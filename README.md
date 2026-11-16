# Do We Need Autoregression for Token-Based Speech Enhancement?

Official implementation and pretrained checkpoints for:

**Do We Need Autoregression for Token-Based Speech Enhancement?**

Jingyi Li, Luca Della Libera, Cem Subakan, and Mirco Ravanelli

This work investigates whether autoregressive (AR) generation is necessary for
high-quality token-based speech enhancement. We formulate enhancement as direct
non-autoregressive (NAR) prediction of clean acoustic tokens from noisy-speech
representations. All token positions are predicted in parallel and converted to
an enhanced waveform by a frozen neural codec decoder.

- [Audio demo](https://jingyi49.github.io/demo/)
- [Pretrained checkpoints](https://github.com/jingyi49/dual-state/releases/tag/checkpoints-v1)

## Overview

```text
Noisy waveform
      |
      v
Frozen WavLM-Large (layer 6)
      |
      v
Parallel token predictor (shallow, deep, or dual-state)
      |
      v
Clean FocalCodec tokens
      |
      v
Frozen FocalCodec decoder
      |
      v
Enhanced waveform
```

The noisy waveform is encoded into continuous WavLM representations. A
trainable predictor maps these representations directly to a sequence of clean
FocalCodec tokens, with no conditioning on previously generated tokens. During
training, the target tokens are extracted from the corresponding clean waveform
using the frozen FocalCodec tokenizer. Only the predictor is optimized; WavLM
and FocalCodec remain frozen.

We adopt FocalCodec because it provides high-quality speech reconstruction with
a single codebook, enabling direct parallel token prediction without the
additional inter-codebook dependencies introduced by multi-codebook codecs.

## Predictor architectures

We study three predictors under the same WavLM-to-FocalCodec framework:

- **Shallow single-state:** one 4-layer non-causal Transformer.
- **Deep single-state:** one 8-layer non-causal Transformer.
- **Dual-state:** separate 4-layer context and prediction Transformers, each
  applied once. The context state summarizes the noisy representation and the
  prediction state produces the token logits.

The comparison isolates the effect of predictor structure while retaining fully
parallel token generation in all three cases.

## Main results

### DNSMOS

DNSMOS results on the DNS Challenge 2020 test sets and real recordings are
reported as OVRL / SIG / BAK. Higher is better.

| Model | Type | Reverb | No reverb | Real recordings |
|---|---|---:|---:|---:|
| LLaSE-G1 | AR | 3.33 / 3.59 / 4.10 | 3.42 / 3.66 / 4.17 | 3.06 / 3.36 / 3.96 |
| UniSE | AR | 3.41 / 3.68 / 4.12 | 3.42 / 3.67 / 4.13 | 3.23 / 3.56 / 3.96 |
| **Dual-state (12.5 Hz)** | **NAR** | **3.46 / 3.67 / 4.21** | **3.44 / 3.66 / 4.20** | **3.44 / 3.67 / 4.19** |
| **Dual-state (25 Hz)** | **NAR** | **3.46 / 3.67 / 4.21** | **3.44 / 3.65 / 4.20** | **3.42 / 3.65 / 4.18** |
| **Dual-state (50 Hz)** | **NAR** | **3.43 / 3.65 / 4.19** | **3.42 / 3.65 / 4.19** | **3.40 / 3.64 / 4.17** |

The 12.5-Hz and 25-Hz variants obtain the highest reverberant OVRL score, while
the 12.5-Hz model leads all reported DNSMOS components on real recordings.

### Predictor ablation

All predictors achieve similar DNSMOS, but the dual-state model improves
linguistic consistency. Speaker similarity (SS) is higher-is-better and
differential word error rate (dWER) is lower-is-better.

| Predictor | Reverb OVRL / SIG / BAK | No-reverb OVRL / SIG / BAK | dWER | SS |
|---|---:|---:|---:|---:|
| Shallow | 3.43 / 3.65 / 4.19 | 3.43 / 3.64 / 4.20 | 0.44 | 0.93 |
| Deep | 3.43 / 3.65 / 4.19 | 3.42 / 3.64 / 4.19 | 0.16 | 0.97 |
| **Dual-state** | **3.43 / 3.65 / 4.19** | **3.42 / 3.65 / 4.19** | **0.12** | **0.97** |

### Continuous versus discrete prediction

The paper also compares discrete token prediction with continuous latent
prediction using three VAE decoders. The discrete 50-Hz dual-state model
substantially improves DNSMOS across both acoustic conditions.

| Target representation | Reverb OVRL / SIG / BAK | No-reverb OVRL / SIG / BAK |
|---|---:|---:|
| EAR-VAE2 | 2.37 / 2.63 / 3.80 | 2.42 / 2.69 / 3.74 |
| KVAE-Audio | 1.80 / 2.13 / 3.33 | 1.88 / 2.21 / 3.38 |
| SAME-S | 1.66 / 1.98 / 3.19 | 1.51 / 1.74 / 3.23 |
| **FocalCodec tokens (dual-state, 50 Hz)** | **3.43 / 3.65 / 4.19** | **3.42 / 3.65 / 4.19** |

### Computational cost

The real-time factor (RTF) is inference time divided by input duration; lower
values indicate faster processing. Parameters include all inference modules,
including frozen components. MACs and RTFs are measured on 10-second waveforms.

| Model | Parameters (M) | GMACs | CPU RTF | GPU RTF |
|---|---:|---:|---:|---:|
| LLaSE-G1 | 1283.05 | 688.04 | 0.326 | 0.009 |
| QuarkAudio-UniSE | 253.98 | 427.80 | 0.640 | 0.172 |
| Dual-state (50 Hz) | 135.02 | 92.75 | 0.073 | 0.002 |
| Dual-state (25 Hz) | 136.07 | 85.48 | **0.048** | **0.002** |
| Dual-state (12.5 Hz) | 136.59 | **83.66** | 0.068 | **0.002** |

CPU measurements use four threads on an AMD EPYC 9655. GPU measurements use an
NVIDIA H100 MIG 3g.40gb partition. Values are medians of five runs after two
warm-up runs and exclude model loading and file I/O.

## Repository structure

```text
train_wavlm.py       Training entry point
infer_wavlm.py       Inference entry point
trm.py               Dual-state predictor
single_state.py      Shallow and deep single-state predictors
dns.py               Paired noisy/clean audio dataset
dns_refinement.py    Parallel token-prediction helper
experiments/         Configurations and training logs for each token rate
dnsmos_results/      Per-utterance DNSMOS outputs
codec_manifest.json  FocalCodec repositories and pinned revisions
```

## Installation

### Requirements

- Python 3.10 or later
- PyTorch and TorchAudio
- `soundfile`
- `tqdm`
- [FocalCodec](https://github.com/lucadellalib/focalcodec), loaded through
  `torch.hub`
- [WavLM-Large](https://github.com/microsoft/unilm/tree/master/wavlm) checkpoint
- The WavLM implementation from
  [LLaSE-G1](https://github.com/Kevin-naticl/LLaSE-G1)

Create a minimal environment with:

```bash
python -m venv .venv
source .venv/bin/activate
pip install torch torchaudio soundfile tqdm
```

Download `WavLM-Large.pt` and clone LLaSE-G1. Their paths are supplied to the
training and inference scripts with `--wavlm-checkpoint` and `--wavlm-repo`.

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

Audio is converted to mono 16 kHz. Identical 10-second crops are applied to each
noisy/clean pair, and a deterministic 5% validation split is used by default.
Check the file pairing before loading the pretrained models with:

```bash
python train_wavlm.py \
  --data-dir /path/to/data \
  --check-data
```

Custom noisy and clean directories can be provided with `--input-dir` and
`--target-dir`.

## Training

The following command trains the 50-Hz dual-state model:

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
  --epochs 50 \
  --batch-size 8 \
  --crop-sec 10
```

Use the following codec settings for the lower-rate experiments:

| Token rate | `--codec-config` | `--codec-revision` |
|---|---|---|
| 50 Hz | `lucadellalib/focalcodec_50hz` | `d6d6d9524e52155c85193c2c3b8da1cf8842f019` |
| 25 Hz | `lucadellalib/focalcodec_25hz` | `581aa401901b4985fbbdd3569d80a5a191740c1f` |
| 12.5 Hz | `lucadellalib/focalcodec_12_5hz` | `96ddced5e284f109d9022a65d1062fcd92dc33eb` |

Architecture and loss ablations are selected using:

```bash
# Four-layer single-state predictor
python train_wavlm.py ... --architecture shallow

# Eight-layer single-state predictor
python train_wavlm.py ... --architecture deep

# Cross-entropy only
python train_wavlm.py ... --loss-type ce_only
```

The default `ce_bce` setting combines token cross-entropy with the model's
sequence-confidence objective. `best.pt` is selected using validation loss, and
`last.pt` contains the final epoch.

## Pretrained checkpoints

The pretrained dual-state checkpoints are hosted in the
[checkpoint release](https://github.com/jingyi49/dual-state/releases/tag/checkpoints-v1):

| Token rate | Best checkpoint | Final checkpoint |
|---|---|---|
| 50 Hz | `50hz-best.pt` | `50hz-last.pt` |
| 25 Hz | `25hz-best.pt` | `25hz-last.pt` |
| 12.5 Hz | `12_5hz-best.pt` | `12_5hz-last.pt` |

Download all checkpoints with the GitHub CLI:

```bash
gh release download checkpoints-v1 \
  --repo jingyi49/dual-state \
  --pattern '*.pt' \
  --dir checkpoints
```

## Inference

The input may be a WAV file, a directory containing WAV files, or a text file
with one WAV path per line:

```bash
python infer_wavlm.py \
  --checkpoint checkpoints/12_5hz-best.pt \
  --input /path/to/noisy.wav \
  --output-dir enhanced \
  --wavlm-checkpoint /path/to/WavLM-Large.pt \
  --wavlm-repo /path/to/LLaSE-G1
```

Inference preserves the input duration, converts multi-channel inputs to mono,
and writes 16 kHz PCM WAV files. Select the device using `--device cpu`,
`--device cuda`, or `--device cuda:N`.
