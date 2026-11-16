#!/usr/bin/env python3
"""Run a train_wavlm.py checkpoint on WAV files listed in a .list file."""
import argparse
import math
from pathlib import Path

import soundfile as sf
import torch
import torch.nn.functional as F
import torchaudio

from dns_refinement import predict_tokens
from train_wavlm import CODEC_CONFIG, FrozenWavLM
from trm import InputEmbedder, TinyRecursiveModel
from single_state import SingleStateModel

OUTPUT_SR = 16000


def collect_inputs(source, output_dir):
    """Resolve a WAV, directory, or UTF-8 path list into output paths."""
    source = source.expanduser().resolve()
    if not source.exists():
        raise ValueError(f"Input does not exist: {source}")
    if source.is_dir():
        files = sorted(
            path for path in source.rglob("*")
            if path.is_file() and path.suffix.lower() == ".wav"
        )
        names = [path.relative_to(source) for path in files]
    elif source.suffix.lower() == ".wav":
        files, names = [source], [Path(source.name)]
    else:
        if source.suffix.lower() not in {".test", ".txt", ".list", ".scp"}:
            raise ValueError(
                "Input must be a WAV, directory, or .test/.txt/.list/.scp path list"
            )
        files = []
        for line_number, line in enumerate(
            source.read_text(encoding="utf-8-sig").splitlines(), 1
        ):
            line = line.strip()
            if not line or line.startswith("#"):
                continue
            path = Path(line).expanduser()
            if not path.is_absolute():
                path = source.parent / path
            path = path.resolve()
            if not path.is_file() or path.suffix.lower() != ".wav":
                raise ValueError(
                    f"{source}:{line_number}: missing WAV or invalid path: {line}"
                )
            files.append(path)
        names = [Path(path.name) for path in files]
    if not files:
        raise ValueError("No WAV files found")
    destinations = [output_dir / name for name in names]
    if len({path.resolve() for path in destinations}) != len(destinations):
        raise ValueError("Duplicate output filenames in input list; use unique WAV basenames")
    sources = {path.resolve() for path in files}
    if any(path.resolve() in sources for path in destinations):
        raise ValueError("Output would replace an input WAV; choose another output directory")
    return files, destinations


@torch.inference_mode()
def enhance(wav, wavlm, codec, model, embedder, *, chunk_samples):
    result = []
    # Determine frame count from length only, never from clean target content.
    frames = codec.sig_to_toks(torch.zeros(1, chunk_samples, device=wav.device)).shape[1]
    for chunk in wav.split(chunk_samples):
        length = chunk.numel()
        chunk = chunk.repeat(math.ceil(chunk_samples / length))[:chunk_samples]
        features = wavlm(chunk.unsqueeze(0))
        if features.shape[1] != frames:
            features = F.adaptive_avg_pool1d(features.transpose(1, 2), frames).transpose(1, 2)
        predicted = predict_tokens(model, embedder, features)
        decoded = codec.toks_to_sig(
            predicted,
            output_length=round(chunk_samples * codec.sample_rate_output /
                                codec.sample_rate_input),
        ).squeeze(0)
        decoded = torchaudio.functional.resample(
            decoded, codec.sample_rate_output, OUTPUT_SR)
        wanted = round(length * OUTPUT_SR / codec.sample_rate_input)
        result.append(F.pad(decoded, (0, max(0, wanted - decoded.numel())))[:wanted].cpu())
    return torch.cat(result)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--input", type=Path, required=True,
                        help="WAV, directory, or .list/.txt/.scp file")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--device", default="auto", help="auto, cpu, cuda, or cuda:N")
    parser.add_argument("--chunk-sec", type=float,
                        help="Default: checkpoint crop_sec")
    parser.add_argument("--wavlm-checkpoint", type=Path,
                        help="Default: checkpoint wavlm_checkpoint")
    parser.add_argument("--wavlm-repo", type=Path,
                        help="Default: checkpoint wavlm_repo")
    parser.add_argument("--wavlm-layer", type=int,
                        help="Default: checkpoint wavlm_layer")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    try:
        files, destinations = collect_inputs(args.input, args.output_dir)
    except (ValueError, OSError, UnicodeError) as exc:
        parser.error(str(exc))
    for src, dst in zip(files, destinations):
        if dst.exists() and not args.overwrite:
            parser.error(f"Output exists: {dst}; use --overwrite to replace")

    device = torch.device(
        ("cuda" if torch.cuda.is_available() else "cpu")
        if args.device == "auto" else args.device
    )
    state = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    cfg = state.get("config", {})
    if cfg.get("input_type") != "wavlm_features":
        parser.error("--checkpoint is not a train_wavlm.py feature-input checkpoint")
    chunk_sec = args.chunk_sec if args.chunk_sec is not None else cfg.get("crop_sec", 10.0)
    wavlm_checkpoint = args.wavlm_checkpoint or Path(cfg["wavlm_checkpoint"])
    wavlm_repo = args.wavlm_repo or Path(cfg["wavlm_repo"])
    wavlm_layer = args.wavlm_layer if args.wavlm_layer is not None else cfg.get("wavlm_layer", 6)
    if not math.isfinite(chunk_sec) or chunk_sec <= 0:
        parser.error("Chunk duration must be positive")

    emb_state = state["input_embedder"]
    feature_dim = emb_state["proj.weight"].shape[1]
    vocab, dim = state["model"]["output_head.weight"].shape
    architecture = cfg.get("architecture", "dual")
    if architecture == "dual":
        model = TinyRecursiveModel(d_model=dim, vocab_size=vocab, n_layers=4,
                                   n_heads=8, dropout=0.1)
    elif architecture in ("shallow", "deep"):
        model = SingleStateModel(d_model=dim, vocab_size=vocab,
                                 n_layers=4 if architecture == "shallow" else 8)
    else:
        parser.error(f"Unknown checkpoint architecture: {architecture}")
    embedder = InputEmbedder(in_dim=feature_dim, d_model=dim,
                             max_len=emb_state["pos.pe.weight"].shape[0])
    model.load_state_dict(state["model"], strict=True)
    embedder.load_state_dict(emb_state, strict=True)
    model.to(device).eval().requires_grad_(False)
    embedder.to(device).eval().requires_grad_(False)
    wavlm = FrozenWavLM(wavlm_checkpoint, wavlm_repo, wavlm_layer).to(device)
    codec = torch.hub.load("lucadellalib/focalcodec", "focalcodec",
                           config=cfg.get("codec_config", CODEC_CONFIG),
                           **({"revision": cfg["codec_revision"]} if cfg.get("codec_revision") else {})).to(device)
    codec.eval().requires_grad_(False)
    if codec.sample_rate_input != 16000:
        raise ValueError(f"Unexpected codec rates: {codec.sample_rate_input} -> {codec.sample_rate_output}")
    chunk_samples = round(chunk_sec * codec.sample_rate_input)
    if chunk_samples < 400:
        parser.error("--chunk-sec must be at least 0.025 seconds for WavLM")
    print(f"Device={device}; WavLM layer={wavlm_layer}; chunk={chunk_sec}s; output={OUTPUT_SR} Hz", flush=True)

    for src, dst in zip(files, destinations):
        audio, sr = sf.read(src, dtype="float32", always_2d=True)
        if len(audio) == 0:
            raise ValueError(f"Empty input: {src}")
        wav = torch.from_numpy(audio.mean(axis=1)).to(device)
        if not torch.isfinite(wav).all():
            raise ValueError(f"Non-finite input: {src}")
        wav = torchaudio.functional.resample(wav, sr, codec.sample_rate_input)
        enhanced = enhance(wav, wavlm, codec, model, embedder,
                           chunk_samples=chunk_samples)
        wanted = max(1, round(len(audio) * OUTPUT_SR / sr))
        enhanced = F.pad(enhanced, (0, max(0, wanted - enhanced.numel())))[:wanted]
        if not torch.isfinite(enhanced).all():
            raise ValueError(f"Non-finite model output: {src}")
        dst.parent.mkdir(parents=True, exist_ok=True)
        sf.write(dst, enhanced.clamp(-1, 1).numpy(), OUTPUT_SR, subtype="PCM_16")
        print(f"{src} -> {dst} ({wanted} samples, {OUTPUT_SR} Hz)", flush=True)


if __name__ == "__main__":
    main()
