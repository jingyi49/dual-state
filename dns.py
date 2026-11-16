"""Filename-paired DNS audio with deterministic train/validation splits."""
from pathlib import Path
import random

import soundfile as sf
import torch
import torchaudio
from torch.utils.data import Dataset


class DNSDataset(Dataset):
    def __init__(self, root, split="train", target_sr=16000, crop_sec=4.0,
                 val_fraction=0.05, seed=1234, input_dir=None, target_dir=None, input_names=None, filename_prefix=""):
        if split not in ("train", "valid"):
            raise ValueError("split must be train or valid")
        if not 0 < val_fraction < 1 or crop_sec <= 0:
            raise ValueError("Require 0 < val_fraction < 1 and crop_sec > 0")
        root = Path(root)
        noisy_dir = Path(input_dir) if input_dir is not None else root / "noisy_train"
        clean_dir = Path(target_dir) if target_dir is not None else root / "clean_train"
        for directory in (noisy_dir, clean_dir):
            if not directory.is_dir():
                raise FileNotFoundError(directory)
        noisy = {p.name: p for p in noisy_dir.iterdir() if p.is_file() and not p.name.startswith(".") and p.suffix.lower() == ".wav"}
        clean = {p.name: p for p in clean_dir.iterdir() if p.is_file() and not p.name.startswith(".") and p.suffix.lower() == ".wav"}
        noisy = {name: path for name, path in noisy.items() if name.startswith(filename_prefix)}
        clean = {name: path for name, path in clean.items() if name.startswith(filename_prefix)}
        if input_names is not None:
            absent = set(input_names) - noisy.keys()
            if absent:
                raise FileNotFoundError(f"Input files disappeared: {sorted(absent)[:5]}")
            noisy = {name: noisy[name] for name in input_names}
        missing = noisy.keys() - clean.keys()
        if missing:
            raise ValueError(f"Missing clean targets for {len(missing)} inputs: {sorted(missing)[:5]}")
        names = sorted(noisy)
        if len(names) < 2:
            raise ValueError("At least two audio pairs are required")
        self.input_names = tuple(names)
        random.Random(seed).shuffle(names)
        count = max(1, min(len(names)-1, round(len(names)*val_fraction)))
        names = names[count:] if split == "train" else names[:count]
        self.pairs = [(noisy[name], clean[name]) for name in names]
        self.training = split == "train"
        self.target_sr = target_sr
        self.crop_len = round(crop_sec * target_sr)
        if self.crop_len < 1:
            raise ValueError("Crop must contain at least one sample")

    def __len__(self):
        return len(self.pairs)

    def __getitem__(self, index):
        noisy_path, clean_path = self.pairs[index]
        # Read only the selected segment, particularly useful for long DNS clips.
        with sf.SoundFile(noisy_path) as nf, sf.SoundFile(clean_path) as cf:
            if nf.samplerate != cf.samplerate:
                raise ValueError(f"Pair sample rate mismatch: {noisy_path.name} "
                                 f"(input={nf.samplerate}, target={cf.samplerate})")
            # Use the common duration and identical crop offsets for both files.
            sr, length = nf.samplerate, min(len(nf), len(cf))
            if length == 0:
                raise ValueError(f"Empty audio: {noisy_path.name}")
            frames = max(1, round(self.crop_len * sr / self.target_sr))
            maximum = max(0, length - frames)
            start = torch.randint(maximum + 1, ()).item() if self.training else maximum // 2
            frames = min(frames, length - start)
            nf.seek(start)
            cf.seek(start)
            noisy = torch.from_numpy(nf.read(frames, dtype="float32", always_2d=True).mean(axis=1))
            clean = torch.from_numpy(cf.read(frames, dtype="float32", always_2d=True).mean(axis=1))
        if sr != self.target_sr:
            noisy = torchaudio.functional.resample(noisy, sr, self.target_sr)
            clean = torchaudio.functional.resample(clean, sr, self.target_sr)
        # Repeat short pairs together to avoid training on batch padding tokens.
        repeats = (self.crop_len + noisy.numel() - 1) // noisy.numel()
        noisy = noisy.repeat(repeats)[:self.crop_len]
        clean = clean.repeat(repeats)[:self.crop_len]
        return {"noisy_wav": noisy, "clean_wav": clean, "utt_id": noisy_path.stem}
