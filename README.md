# Direct Non-Autoregressive Token Prediction for Speech Enhancement

Official implementation and pretrained checkpoints for our study of direct
non-autoregressive (NAR) acoustic-token prediction for speech enhancement.

Rather than generating clean acoustic tokens sequentially, the system predicts
all tokens in parallel from frozen WavLM representations of noisy speech. The
predicted tokens are converted to an enhanced waveform with a pretrained
FocalCodec decoder. We evaluate a dual-state predictor at three codec token
rates: **50 Hz**, **25 Hz**, and **12.5 Hz**.

Audio samples are available on the [demo page](https://jingyi49.github.io/demo/).

## Method

```text
Noisy waveform
      |
      v
Frozen WavLM-Large (layer 6)
      |
      v
Dual-state Transformer predictor
      |
      v
Clean FocalCodec tokens (predicted in parallel)
      |
      v
Frozen FocalCodec decoder
      |
      v
Enhanced waveform
```

FocalCodec is used because it provides high-quality speech reconstruction with
a single codebook, allowing direct parallel token prediction without the
additional inter-codebook dependencies introduced by multi-codebook codecs.

## Repository structure

```text
train_wavlm.py       Training entry point
infer_wavlm.py       Inference entry point
trm.py               Dual-state predictor
single_state.py      Shallow and deep single-state ablations
dns.py               Paired noisy/clean audio dataset
dns_refinement.py    Parallel token-prediction helper
experiments/         Configurations and training logs for each token rate
dnsmos_results/      Per-utterance DNSMOS results
codec_manifest.json  FocalCodec versions used in the experiments
```

## Requirements

- Python 3.10 or later
- PyTorch and TorchAudio
- `soundfile`
- `tqdm`
- [FocalCodec](https://github.com/lucadellalib/focalcodec), loaded through
  `torch.hub`
- [WavLM-Large](https://github.com/microsoft/unilm/tree/master/wavlm) checkpoint
- The WavLM implementation from
  [LLaSE-G1](https://github.com/Kevin-naticl/LLaSE-G1)

A minimal environment can be prepared with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchaudio soundfile tqdm
```

Download `WavLM-Large.pt` and clone LLaSE-G1, then pass their local paths using
`--wavlm-checkpoint` and `--wavlm-repo`.

## Data preparation

Training expects paired noisy and clean WAV files with identical filenames:

```text
data/
├── noisy_train/
│   ├── sample_0001.wav
│   └── sample_0002.wav
└── clean_train/
    ├── sample_0001.wav
    └── sample_0002.wav
```

The loader converts audio to mono 16 kHz and applies identical crops to each
noisy/clean pair. A deterministic 5% validation split is used by default.

You can validate the file pairing before loading the pretrained models:

```bash
python train_wavlm.py \
  --data-dir /path/to/data \
  --check-data
```

## Training

The following commands reproduce the three dual-state codec-rate experiments.
Replace the data and WavLM paths with local paths.

### 50 Hz

```bash
python train_wavlm.py \
  --data-dir /path/to/data \
  --output-dir experiments/50hz \
  --architecture dual \
  --codec-config lucadellalib/focalcodec_50hz \
  --codec-revision d6d6d9524e52155c85193c2c3b8da1cf8842f019 \
  --wavlm-checkpoint /path/to/WavLM-Large.pt \
  --wavlm-repo /path/to/LLaSE-G1 \
  --wavlm-layer 6 \
  --epochs 50 --batch-size 8 --crop-sec 10
```

### 25 Hz

```bash
python train_wavlm.py \
  --data-dir /path/to/data \
  --output-dir experiments/25hz \
  --architecture dual \
  --codec-config lucadellalib/focalcodec_25hz \
  --codec-revision 581aa401901b4985fbbdd3569d80a5a191740c1f \
  --wavlm-checkpoint /path/to/WavLM-Large.pt \
  --wavlm-repo /path/to/LLaSE-G1 \
  --wavlm-layer 6 \
  --epochs 50 --batch-size 8 --crop-sec 10
```

### 12.5 Hz

```bash
python train_wavlm.py \
  --data-dir /path/to/data \
  --output-dir experiments/12_5hz \
  --architecture dual \
  --codec-config lucadellalib/focalcodec_12_5hz \
  --codec-revision 96ddced5e284f109d9022a65d1062fcd92dc33eb \
  --wavlm-checkpoint /path/to/WavLM-Large.pt \
  --wavlm-repo /path/to/LLaSE-G1 \
  --wavlm-layer 6 \
  --epochs 50 --batch-size 8 --crop-sec 10
```

By default, training uses the combined cross-entropy and confidence BCE loss.
Use `--loss-type ce_only` for the CE-only ablation. The architecture ablations
are selected with `--architecture shallow` (one 4-layer state) and
`--architecture deep` (one 8-layer state).

## Pretrained checkpoints

The trained checkpoints are available in the
[checkpoint release](https://github.com/jingyi49/dual-state/releases/tag/checkpoints-v1):

| Token rate | Best checkpoint | Final checkpoint |
|---|---|---|
| 50 Hz | `50hz-best.pt` | `50hz-last.pt` |
| 25 Hz | `25hz-best.pt` | `25hz-last.pt` |
| 12.5 Hz | `12_5hz-best.pt` | `12_5hz-last.pt` |

They can also be downloaded with the GitHub CLI:

```bash
gh release download checkpoints-v1 \
  --repo jingyi49/dual-state \
  --pattern '*.pt' \
  --dir checkpoints
```

## Inference

The input may be a WAV file, a directory of WAV files, or a text file containing
one WAV path per line.

```bash
python infer_wavlm.py \
  --checkpoint checkpoints/12_5hz-best.pt \
  --input /path/to/noisy.wav \
  --output-dir enhanced \
  --wavlm-checkpoint /path/to/WavLM-Large.pt \
  --wavlm-repo /path/to/LLaSE-G1
```

Inference preserves the input duration, converts multi-channel inputs to mono,
and writes 16 kHz PCM WAV files. Use `--device cpu`, `--device cuda`, or
`--device cuda:N` to select the device.

## DNSMOS results

The following values are averages over the 150 utterances in each included
DNSMOS result file. Full per-utterance scores are provided under
`dnsmos_results/`.

| Condition | Token rate | OVRL | SIG | BAK | P.808 |
|---|---:|---:|---:|---:|---:|
| Reverberant | 50 Hz | 3.424 | 3.654 | 4.187 | 4.050 |
| Reverberant | 25 Hz | 3.459 | 3.669 | 4.208 | 4.088 |
| Reverberant | 12.5 Hz | 3.463 | 3.675 | 4.206 | 4.048 |
| Non-reverberant | 50 Hz | 3.429 | 3.651 | 4.190 | 4.013 |
| Non-reverberant | 25 Hz | 3.442 | 3.653 | 4.200 | 4.038 |
| Non-reverberant | 12.5 Hz | 3.445 | 3.657 | 4.201 | 4.018 |

## Citation

If you use this code or the pretrained models, please cite the accompanying
paper. The BibTeX entry will be added after publication.

## Acknowledgements

This repository uses pretrained representations from WavLM and acoustic tokens
from FocalCodec. The WavLM feature-extraction setup follows LLaSE-G1.
