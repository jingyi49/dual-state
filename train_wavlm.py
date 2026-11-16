#!/usr/bin/env python3
"""Frozen WavLM-Large features of noisy audio -> TRM -> clean FocalCodec tokens.

Uses the LLaSE-G1 layer-6 feature recipe, keeping TRM and FocalCodec targets.
Reference: https://github.com/Kevin-naticl/LLaSE-G1
"""
import argparse
import json
import logging
import math
import os
from pathlib import Path
import sys
from itertools import islice
from logging.handlers import RotatingFileHandler

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm import tqdm

from dns import DNSDataset
from trm import InputEmbedder, TinyRecursiveModel
from single_state import SingleStateModel

config = {
    "seed": 1234,
    "output_dir": "experiments/trm_wavlm",
    "num_epochs": 50,
    "batch_size": 8,
    "num_workers": 0,
    "lr": 5e-4,
    "weight_decay": 0.01,
    "max_grad_norm": 5.0,
    "use_amp": False,
    "n_steps": 1,
    "T_passes": 1,
    "d_model": 256,
    "vocab_size": 4096,
}
CODEC_CONFIG = "lucadellalib/focalcodec_50hz_4k_causal"


def setup_logger(log_dir, log_filename="train.log", max_bytes=1024 * 1024):
    os.makedirs(log_dir, exist_ok=True)
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    if not logger.handlers:
        formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")
        console = logging.StreamHandler()
        console.setFormatter(formatter)
        logger.addHandler(console)
        log_file = RotatingFileHandler(
            os.path.join(log_dir, log_filename), maxBytes=max_bytes, backupCount=0
        )
        log_file.setFormatter(formatter)
        logger.addHandler(log_file)
    return logger


def seed_everything(seed):
    if seed is not None:
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)


def _run_epoch(model, input_embedder, dataloader, device, *, optimizer=None,
               clip_grad=1.0, use_amp=False, loss_type="ce_bce"):
    if loss_type not in ("ce_bce", "ce_only"):
        raise ValueError(f"Unknown loss type: {loss_type}")
    bce_weight = 1.0 if loss_type == "ce_bce" else 0.0
    training = optimizer is not None
    model.train(training)
    input_embedder.train(training)
    totals = dict(ce=0.0, bce=0.0, final_ce=0.0, token_acc=0.0,
                  seq_acc=0.0, q_mean=0.0, q_mean_pos=0.0, q_mean_neg=0.0)
    samples = positive = negative = 0
    pbar = tqdm(dataloader, desc="train" if training else "valid", leave=False)
    for batch in pbar:
        inputs = batch["x_input"].to(device)
        targets = batch["y_true"].to(device)
        batch_size = len(inputs)
        if training:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(training):
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16,
                                enabled=use_amp, cache_enabled=False):
                x = input_embedder(inputs)
                if isinstance(model, SingleStateModel):
                    logits, q = model(x)
                else:
                    y = torch.zeros(batch_size, inputs.shape[1], model.d, device=device)
                    z = torch.zeros_like(y)
                    _, logits, q = model.deep_recursion(x, y, z)
                with torch.autocast(device_type=device.type, enabled=False):
                    ce = F.cross_entropy(
                        logits.float().transpose(1, 2), targets,
                        reduction="none",
                    ).mean(dim=1)
                    ids = logits.detach().argmax(dim=-1)
                    exact_float = (ids == targets).all(dim=1).float()
                    # Keep BCE as a diagnostic in CE-only runs, but disconnect
                    # the confidence branch completely from backpropagation.
                    q_for_bce = q.float() if bce_weight else q.detach().float()
                    bce = F.binary_cross_entropy(q_for_bce, exact_float, reduction="none")
                    loss = (ce + bce).mean() if bce_weight else ce.mean()
                if training:
                    loss.backward()
                totals["ce"] += ce.detach().sum().item()
                totals["bce"] += bce.detach().sum().item()
        if training:
            if clip_grad is not None:
                nn.utils.clip_grad_norm_(
                    list(model.parameters()) + list(input_embedder.parameters()),
                    clip_grad,
                )
            optimizer.step()
        exact = (ids == targets).all(dim=1)
        final_q = q.detach().float()
        totals["final_ce"] += ce.detach().sum().item()
        totals["token_acc"] += (ids == targets).float().mean(dim=1).sum().item()
        totals["seq_acc"] += exact.sum().item()
        totals["q_mean"] += final_q.sum().item()
        totals["q_mean_pos"] += final_q[exact].sum().item()
        totals["q_mean_neg"] += final_q[~exact].sum().item()
        positive += exact.sum().item()
        negative += (~exact).sum().item()
        samples += batch_size
        pbar.set_postfix(
            loss=f"{(totals['ce'] + bce_weight * totals['bce']) / samples:.4f}",
            tok_acc=f"{totals['token_acc'] / samples:.3f}",
        )
    if not samples:
        raise ValueError("Empty epoch")
    metrics = {key: totals[key] / samples for key in
               ("final_ce", "token_acc", "seq_acc", "q_mean")}
    metrics.update(
        ce=totals["ce"] / samples,
        bce=totals["bce"] / samples,
        loss=(totals["ce"] + bce_weight * totals["bce"]) / samples,
        q_mean_pos=totals["q_mean_pos"] / max(1, positive),
        q_mean_neg=totals["q_mean_neg"] / max(1, negative),
    )
    return metrics


def train_one_epoch(model, input_embedder, dataloader, optimizer, device, **kwargs):
    return _run_epoch(
        model, input_embedder, dataloader, device, optimizer=optimizer, **kwargs
    )


@torch.no_grad()
def validate_one_epoch(model, input_embedder, dataloader, device, **kwargs):
    return _run_epoch(model, input_embedder, dataloader, device, **kwargs)


class FrozenWavLM(nn.Module):
    """Raw 16 kHz audio, no waveform normalization, matching LLaSE-G1 extraction."""
    def __init__(self, checkpoint, repo, layer=6):
        super().__init__()
        repo = Path(repo).resolve()
        if not (repo / 'nnet/WavLM.py').is_file():
            raise FileNotFoundError(f"Missing LLaSE-G1 WavLM implementation in {repo}")
        sys.path.insert(0, str(repo))
        from nnet.WavLM import WavLM, WavLMConfig
        state = torch.load(checkpoint, map_location='cpu', weights_only=False)
        cfg = WavLMConfig(state['cfg'])
        if not 1 <= layer <= cfg.encoder_layers:
            raise ValueError(f'Invalid WavLM layer {layer} for {cfg.encoder_layers} layers')
        self.encoder = WavLM(cfg)
        self.encoder.load_state_dict(state['model'], strict=True)
        self.layer = layer
        self.feature_dim = cfg.encoder_embed_dim
        self.requires_grad_(False)
        self.eval()

    def train(self, mode=True):
        # It stays frozen/eval even when its caller enters training mode.
        return super().train(False)

    @torch.no_grad()
    def forward(self, wav):
        if wav.ndim != 2 or wav.shape[1] < 400:
            raise ValueError('Expected mono 16 kHz audio [B,N], at least 400 samples')
        features = self.encoder.extract_features(wav.float(), output_layer=self.layer,
                                                mask=False, ret_layer_results=False)[0]
        # Native WavLM has 320-sample stride with an unpadded 400-sample receptive
        # field. FocalCodec causal 50 Hz tokens use ceil(N/320) frames. Align from
        # waveform length only; never use clean token CONTENT to construct input.
        frames = math.ceil(wav.shape[1] / 320)
        if features.shape[1] != frames:
            features = F.interpolate(features.transpose(1, 2), size=frames,
                                     mode='linear', align_corners=False).transpose(1, 2)
        return features.contiguous()


class FeatureTokenLoader:
    """Frozen online extraction in the main process; workers only read audio."""
    def __init__(self, loader, wavlm, codec, device, max_batches=None):
        self.loader, self.wavlm, self.codec = loader, wavlm, codec
        self.device, self.max_batches = device, max_batches
        self.reported = False

    def __len__(self):
        return min(len(self.loader), self.max_batches) if self.max_batches else len(self.loader)

    def __iter__(self):
        for batch in islice(self.loader, self.max_batches):
            with torch.no_grad():
                x = self.wavlm(batch['noisy_wav'].to(self.device, non_blocking=True))
                y = self.codec.sig_to_toks(batch['clean_wav'].to(self.device, non_blocking=True)).long()
            if x.ndim != 3 or y.ndim != 2:
                raise ValueError(f'WavLM/codec alignment mismatch: {x.shape}, {y.shape}')
            # WavLM features are at 50 Hz; codec tokens may be lower-rate
            # (50/25/12.5 Hz). Time-align by pooling the feature axis to the
            # exact number of codec token frames.
            if x.shape[1] != y.shape[1]:
                x = F.adaptive_avg_pool1d(x.transpose(1, 2), y.shape[1]).transpose(1, 2)
            if x.shape[:2] != y.shape:
                raise ValueError(f'WavLM/codec alignment mismatch: {x.shape}, {y.shape}')
            if x.shape[1] > 8192 or not torch.isfinite(x).all():
                raise ValueError('Non-finite WavLM features or too many frames (>8192)')
            if not self.reported:
                print(f'WavLM features={tuple(x.shape)}, clean tokens={tuple(y.shape)}, frozen=True', flush=True)
                self.reported = True
            yield {'x_input': x, 'y_true': y}


def main():
    parser = argparse.ArgumentParser(description="Train TRM: frozen noisy WavLM features -> clean FocalCodec tokens")
    parser.add_argument("--data-dir", default="/home/jingyi49/scratch/large_dataset")
    parser.add_argument("--input-dir", help="Input WAV folder (default: data-dir/noisy_train)")
    parser.add_argument("--target-dir", help="Target WAV folder (default: data-dir/clean_train)")
    parser.add_argument("--filename-prefix", default="", help="Only pair WAV filenames starting with this prefix")
    parser.add_argument("--output-dir", default="experiments/trm_wavlm")
    parser.add_argument("--architecture", choices=("dual", "shallow", "deep"), default="dual",
                        help="dual: original 4+4 layers; single-state shallow: 4, deep: 8")
    parser.add_argument("--loss-type", choices=("ce_bce", "ce_only"), default="ce_bce",
                        help="CE-only removes confidence supervision; BCE/q remain diagnostics")
    parser.add_argument("--codec-config", default=CODEC_CONFIG,
                        help="FocalCodec repository or local config JSON")
    parser.add_argument("--codec-revision", help="Pinned Hugging Face codec commit")
    parser.add_argument("--epochs", type=int, default=50)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--crop-sec", type=float, default=10.0)
    parser.add_argument("--val-fraction", type=float, default=0.05)
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--lr", type=float, default=5e-4)
    parser.add_argument("--tf32", action="store_true", help="Allow TF32 for FP32 codec matrix operations")
    parser.add_argument("--amp", action="store_true", help="Use BF16 mixed precision for TRM; codec stays FP32")
    parser.add_argument("--resume", type=Path)
    parser.add_argument("--max-train-batches", type=int)
    parser.add_argument("--max-valid-batches", type=int)
    parser.add_argument("--check-data", action="store_true", help="Check pairing and one batch without downloading the codec")
    parser.add_argument("--wavlm-checkpoint", type=Path, default=Path("/home/jingyi49/scratch/WavLM-Large.pt"))
    parser.add_argument("--wavlm-repo", type=Path, default=Path("/home/jingyi49/scratch/LLaSE-G1"))
    parser.add_argument("--wavlm-layer", type=int, default=6)
    args = parser.parse_args()
    if not 1 <= args.wavlm_layer <= 24:
        parser.error("--wavlm-layer must be between 1 and 24")
    if args.crop_sec < 0.025:
        parser.error("WavLM requires at least 400 samples (0.025 seconds)")
    if args.lr <= 0 or args.num_workers < 0:
        parser.error("Require lr > 0 and num-workers >= 0")
    resume_state = None
    if args.resume:
        resume_state = torch.load(args.resume, map_location="cpu", weights_only=False)
        saved = resume_state.get("config", {})
        if saved.get("codec_config", CODEC_CONFIG) != args.codec_config:
            parser.error("Resume codec mismatch")
        if saved.get("codec_revision") != args.codec_revision:
            parser.error("Resume codec revision mismatch")
        if saved.get("loss_type", "ce_bce") != args.loss_type:
            parser.error("Resume loss mismatch; start a separate experiment for a loss ablation")
        if saved.get("architecture", "dual") != args.architecture:
            parser.error("Resume architecture mismatch")
        if saved.get("input_type") != "wavlm_features":
            parser.error("--resume requires a train_wavlm.py checkpoint, not a token-input checkpoint")
        for key in ("wavlm_layer", "crop_sec"):
            if saved.get(key) != getattr(args, key):
                parser.error(f"Resume configuration mismatch: {key}; use the saved setting")
        if saved.get("wavlm_checkpoint") != str(args.wavlm_checkpoint):
            parser.error("Resume must use the saved --wavlm-checkpoint")
        if resume_state["epoch"] + 1 >= args.epochs:
            parser.error("Checkpoint already reached --epochs; supply a larger TOTAL epoch count")
    if min(args.epochs, args.batch_size) < 1:
        parser.error("Epoch and batch counts must be positive")
    if int(os.environ.get("WORLD_SIZE", "1")) > 1:
        parser.error("This entry point supports one process; run python train_wavlm.py")
    for limit in (args.max_train_batches, args.max_valid_batches):
        if limit is not None and limit < 1:
            parser.error("Batch limits must be positive")
    if args.tf32:
        torch.set_float32_matmul_precision("high")
        torch.backends.cudnn.allow_tf32 = True
    seed_everything(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.amp:
        if device.type != "cuda" or not torch.cuda.is_bf16_supported():
            parser.error("--amp requires a CUDA GPU with BF16 support")
        # Avoid eval Transformer fast paths that bypass autocast.
        torch.backends.mha.set_fastpath_enabled(False)
    dataset_options = dict(crop_sec=args.crop_sec, val_fraction=args.val_fraction,
                           seed=args.seed, input_dir=args.input_dir, target_dir=args.target_dir,
                           filename_prefix=args.filename_prefix)
    train_dataset = DNSDataset(args.data_dir, "train", **dataset_options)
    datasets = {"train": train_dataset,
                "valid": DNSDataset(args.data_dir, "valid", input_names=train_dataset.input_names,
                                    **dataset_options)}
    loaders = {split: DataLoader(ds, batch_size=args.batch_size, shuffle=split == "train",
                                num_workers=args.num_workers, pin_memory=device.type == "cuda",
                                persistent_workers=args.num_workers > 0)
               for split, ds in datasets.items()}
    print(f"DNS train={len(datasets['train'])}, valid={len(datasets['valid'])}, device={device}")
    if args.check_data:
        batch = next(iter(loaders["train"]))
        print({key: tuple(value.shape) if torch.is_tensor(value) else value[:2]
               for key, value in batch.items()})
        return
    logger = setup_logger(args.output_dir)
    output = Path(args.output_dir)
    checkpoint_dir = output / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    config.update(vars(args))
    config.update(input_type="wavlm_features", wavlm_layer=args.wavlm_layer,
                  wavlm_checkpoint=str(args.wavlm_checkpoint), wavlm_repo=str(args.wavlm_repo),
                  wavlm_frozen=True, wavlm_normalize=False,
                  feature_alignment="linear_to_ceil_samples_over_320_align_corners_false",
                  sample_rate=16000, codec_config=args.codec_config)
    config.update(num_epochs=args.epochs, n_steps=1, T_passes=1,
                  use_amp=args.amp and device.type == "cuda")
    config["metrics_version"] = 2
    config["bce_weight"] = 1.0 if args.loss_type == "ce_bce" else 0.0
    config["checkpoint_selection_metric"] = "valid_loss"
    config["transformer_layers"] = 4 if args.architecture == "shallow" else 8
    config["state_count"] = 2 if args.architecture == "dual" else 1
    config["resume"] = str(args.resume) if args.resume else None

    for split, ds in datasets.items():
        (output / f"{split}_files.txt").write_text("\n".join(n.name for n, _ in ds.pairs) + "\n")
    wavlm = FrozenWavLM(args.wavlm_checkpoint, args.wavlm_repo, args.wavlm_layer).to(device)
    config["feature_dim"] = wavlm.feature_dim
    codec_options = {"revision": args.codec_revision} if args.codec_revision else {}
    codec = torch.hub.load("lucadellalib/focalcodec", "focalcodec",
                           config=args.codec_config, **codec_options).to(device)
    codec.eval().requires_grad_(False)
    if codec.sample_rate_input != 16000:
        raise ValueError("Expected 16 kHz FocalCodec encoder input")
    config["vocab_size"] = int(codec.quantizer.codebook_size)
    if resume_state and resume_state["config"]["vocab_size"] != config["vocab_size"]:
        raise ValueError("Resume vocabulary mismatch")
    # Probe length only, without clean audio content; inference uses this same
    # frozen encoder to determine the codec frame grid for a waveform length.
    with torch.no_grad():
        probe = torch.zeros(1, round(args.crop_sec * 16000), device=device)
        config["codec_frames_per_crop"] = int(codec.sig_to_toks(probe).shape[1])
    (output / "config.json").write_text(json.dumps(config, indent=2))
    logger.info("codec=%s vocab_size=%s frames_per_crop=%s", args.codec_config,
                config["vocab_size"], config["codec_frames_per_crop"])
    if args.architecture == "dual":
        model = TinyRecursiveModel(d_model=config["d_model"], vocab_size=config["vocab_size"],
                                   n_layers=4, n_heads=8, dropout=0.1).to(device)
    else:
        model = SingleStateModel(d_model=config["d_model"], vocab_size=config["vocab_size"],
                                 n_layers=config["transformer_layers"], n_heads=8,
                                 dropout=0.1).to(device)
    embedder = InputEmbedder(in_dim=wavlm.feature_dim, d_model=config["d_model"],
                             max_len=8192).to(device)
    logger.info("architecture=%s trainable_parameters=%s", args.architecture,
                sum(p.numel() for p in list(model.parameters()) + list(embedder.parameters())))
    optimizer = torch.optim.AdamW(list(model.parameters()) + list(embedder.parameters()),
                                  lr=args.lr, weight_decay=config["weight_decay"])
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, factor=0.9, patience=1)
    start_epoch, best = 0, float("inf")
    if resume_state is not None:
        model.load_state_dict(resume_state["model"])
        embedder.load_state_dict(resume_state["input_embedder"])
        optimizer.load_state_dict(resume_state["optimizer"])
        scheduler.load_state_dict(resume_state["scheduler"])
        start_epoch, best = resume_state["epoch"] + 1, resume_state["best_loss"]
        del resume_state
    token_loaders = {split: FeatureTokenLoader(loader, wavlm, codec, device,
                    args.max_train_batches if split == "train" else args.max_valid_batches)
                    for split, loader in loaders.items()}
    for epoch in range(start_epoch, args.epochs):
        train_metrics = train_one_epoch(model, embedder, token_loaders["train"], optimizer, device,
            clip_grad=config["max_grad_norm"], use_amp=args.amp and device.type == "cuda",
            loss_type=args.loss_type)
        valid_metrics = validate_one_epoch(model, embedder, token_loaders["valid"], device,
            use_amp=args.amp and device.type == "cuda", loss_type=args.loss_type)
        scheduler.step(valid_metrics["loss"])
        improved = valid_metrics["loss"] < best
        best = min(best, valid_metrics["loss"])
        logger.info("epoch=%s train=%s valid=%s", epoch, train_metrics, valid_metrics)
        state = {"epoch": epoch, "model": model.state_dict(), "input_embedder": embedder.state_dict(),
                 "optimizer": optimizer.state_dict(), "scheduler": scheduler.state_dict(),
                 "config": config, "best_loss": best}
        for name in (["last.pt", "best.pt"] if improved else ["last.pt"]):
            temporary = checkpoint_dir / (name + ".tmp")
            torch.save(state, temporary)
            temporary.replace(checkpoint_dir / name)


if __name__ == "__main__":
    main()
