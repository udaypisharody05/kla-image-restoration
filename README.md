# KLA / i4C Hackathon — AI-Based Restoration of Degraded Images

Restores noisy, low-resolution grayscale semiconductor-style images to clean,
2x-higher-resolution images: `128x128` noisy LR -> `256x256` restored HR.
This repository is a submission for the **AI-Based Restoration of Degraded
Images** problem statement.

## Overview

- **Problem**: paired `.npy` grayscale images, `NoisyLR` (~128x128,
  signal-dependent noise, no blur, no fixed pattern — see
  [`results/degradation_analysis/degradation_report.md`](results/degradation_analysis/degradation_report.md))
  and `GT` (~256x256, exact 2x). The task is a joint denoise + 2x
  super-resolution problem solved end-to-end by one model trained on paired
  supervision — there is no separate denoising stage.
- **Model**: `ResidualSRNet` (`src/models/residual_sr.py`) — a small residual
  CNN (initial 3x3 conv -> 8 residual blocks -> conv -> `PixelShuffle(2)`),
  64 feature channels, **630,724 trainable parameters**, trained with EMA
  weight averaging (decay 0.999).
- **Final champion**: `checkpoints/exp23_ema_extended90/checkpoint_best.pt`
  (EMA weights), packaged for inference as the tracked public artifact
  [`weights/residualsr_final_ema.pt`](weights/residualsr_final_ema.pt). See
  [EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) for the full ablation history that
  led to this configuration.
- **Alternatives tested, not adopted**: an EDSR-lite (1.37M params), a
  NAFNet-SR-style gated architecture (432K params), and a SwinIR-lite
  windowed-attention transformer (348K params) were all trained and measured
  under matched conditions and **all underperformed** the smaller
  ResidualSR champion (by -0.15 dB, -0.49 dB, and -0.27 dB PSNR respectively)
  — see EXPERIMENT_LOG.md Experiments 9/12/13. Several other components
  (channel attention, RDB blocks, noise conditioning, variance-weighted
  loss, hard-patch sampling) are implemented and unit-tested but were never
  trained to completion, so they are **not** part of the submitted result —
  see [`submission/IDEA_SUBMISSION_CONTENT.md`](submission/IDEA_SUBMISSION_CONTENT.md)
  Slide 5 for the full implemented-vs-benchmarked breakdown.
- **Measured validation results** (640-sample canonical split, seed 42; see
  [Final metrics](#final-metrics)): **27.9893 dB PSNR / 0.756916 SSIM**
  (raw), **28.0355 dB PSNR / 0.758519 SSIM** (+x8 TTA), vs. a **23.1413 dB /
  0.550604** bicubic baseline.

## Quick Start

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
```

Place the hackathon-provided dataset at `data/Data-public/` (expected
layout: `data/Data-public/train/train/{NoisyLR,GT}` for the 3,200 training
pairs, `data/Data-public/Test_NoisyLR/NoisyLR` for the 400 official test
inputs — see [`docs/dataset_notes.md`](docs/dataset_notes.md)). The dataset
itself is not part of this repository.

Then restore the official test set using the tracked final weights
(`weights/residualsr_final_ema.pt`, loaded automatically — no path needs to
be supplied):

```bash
python inference.py --input-dir data/Data-public/Test_NoisyLR/NoisyLR --output-dir restored_test_outputs
```

See [Installation](#installation) below for the full CPU/CUDA install
matrix, [Standalone inference](#standalone-inference) for all inference
flags, and [Validation / evaluation](#validation--evaluation) to reproduce
the PSNR/SSIM/LPIPS numbers in [Final metrics](#final-metrics) against the
same tracked weights.

## Repository structure

```text
inference.py                   Standalone restoration script (the deliverable evaluators run)
train.py                       Training entry point (reproduces the champion)
evaluate_checkpoint.py         PSNR/SSIM/LPIPS evaluation on the validation split
evaluate_baseline.py           Classical bicubic baseline (same metrics)
evaluate_ensemble.py           Multi-checkpoint/alpha-search ensemble evaluation
benchmark_inference.py         Load time / latency / throughput / peak-memory benchmark
export_final_weights.py        Re-exports weights/residualsr_final_ema.pt from the champion checkpoint
validate_restored_outputs.py   Validates restored_test_outputs/ and writes manifest.json
generate_submission_assets.py  Regenerates submission/assets/ figures

src/models/                    Model architectures (ResidualSRNet + experimental variants)
src/                           Library code (models, losses, dataset, transforms, metrics, TTA, ...)
tests/                         pytest suite (unit + integration)

weights/residualsr_final_ema.pt   Final packaged inference weights (~2.4 MiB) -- tracked in Git
checkpoints/                      Full training checkpoints (NOT in Git; see below)
restored_test_outputs/            Final restored outputs for all 400 official test images + manifest.json
results/                          Measured metrics/benchmark JSON (final_metrics.json, final_benchmark.json, ...)
submission/                       Idea-submission slide content + figures (pipeline/metrics/sample panels)
EXPERIMENT_LOG.md                 Full experiment history (30 experiments, what worked and what didn't)
```

## Supported environment

- **Python**: developed and tested with **3.12.10** (Windows). No other
  Python version has been verified.
- **PyTorch**: **2.7.1** (`2.7.1+cu128` in the development GPU environment).
- **CUDA**: used automatically when available
  (`torch.cuda.is_available()` — no hard-coded GPU index, no
  platform-specific logic). Development/benchmarking hardware was an
  **NVIDIA GeForce RTX 4060 Laptop GPU (~8 GB VRAM)** — see
  [Inference benchmark](#inference-benchmark) for numbers measured on that
  hardware specifically. The code makes no GPU-model-specific assumptions;
  it is expected to run on any CUDA device, including a datacenter GPU such
  as an H100.
- **CPU fallback**: fully supported. `inference.py`, `evaluate_checkpoint.py`,
  and `train.py` all fall back to CPU automatically when CUDA is
  unavailable.

## Installation

### Windows PowerShell

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` is a complete pip-freeze snapshot of the training
environment. As installed above, `torch`/`torchvision` resolve to their
**CPU** builds (works everywhere, no GPU required). To use CUDA, install the
matching CUDA build *before* the step above (pip will then see it already
satisfied and leave it alone):

```bash
pip install torch==2.7.1 torchvision==0.22.1 --index-url https://download.pytorch.org/whl/cu128
```

Use the `cuXXX` index matching your CUDA driver (see
[pytorch.org/get-started](https://pytorch.org/get-started/locally/)) — cu128
is what the development machine used, not a strict requirement.

## Model weights

The final submission model is the EMA weights of
`checkpoints/exp23_ema_extended90/checkpoint_best.pt`, repackaged as a small,
self-contained, inference-only file:

```text
weights/residualsr_final_ema.pt   (~2.4 MiB — small enough to track normally in Git)
```

It contains the model's `state_dict` plus every piece of architecture
metadata needed to reconstruct it (`architecture`, `in_channels`,
`out_channels`, `num_features`, `num_blocks`, `scale`, and every optional
ResidualSR variant flag), so **no CLI flags are required** to load it
correctly. `inference.py` loads this file by default — no path needs to be
supplied or edited. To regenerate it from the source checkpoint (which is
never modified):

```bash
python export_final_weights.py
```

The full training checkpoint (`checkpoints/exp23_ema_extended90/checkpoint_best.pt`,
~9.7 MiB, includes optimizer/scheduler/EMA state) is excluded from Git by
`.gitignore` (all of `checkpoints/` is), consistent with every other
experiment checkpoint in this project — only the small packaged inference
file is tracked.

## Input format

- Format: **`.npy`**, one file per image.
- Shape: `[H, W]` — 2D, single channel (no channel axis). NoisyLR is
  ~`128x128`; GT/restored output is ~`256x256` (exact 2x).
- dtype: `float32`.
- Numerical range: **not normalized to `[0,1]`**. Measured NoisyLR values
  span roughly `[-0.28, 2.16]` (~3.4% of pixels fall outside `[0,1]`); GT
  values are within `[0,1]` exactly. See
  [`docs/dataset_notes.md`](docs/dataset_notes.md). The model is fed raw
  values unchanged (no clipping/normalization anywhere in the pipeline), and
  its output is likewise raw/unclipped, matching this convention.

## Standalone inference

**One command, no source edits, no dataset/checkpoint path configuration
required:**

```bash
python inference.py --input-dir data/Data-public/Test_NoisyLR/NoisyLR --output-dir restored_test_outputs
```

- Loads `weights/residualsr_final_ema.pt` automatically.
- Uses CUDA automatically if available, otherwise CPU.
- Processes every `.npy` file under `--input-dir` (deterministic, sorted
  order), writes one `.npy` output per input under `--output-dir` (created
  automatically if missing), preserving filenames.
- Requires no ground truth and no training-dataset path.
- Prints device used, per-file progress, total images processed, total
  inference time, and average time/image.

Optional flags:

```text
--checkpoint <path>     Override the model weights (accepts either a packaged
                        weights/*.pt file or a full train.py checkpoint --
                        both reconstruct the model with zero extra flags).
--tta {none,x8}         See "TTA" below. Default: none.
--device {cuda,cpu}     Override auto-detection.
```

Validate the generated outputs and produce a manifest:

```bash
python validate_restored_outputs.py --input-dir data/Data-public/Test_NoisyLR/NoisyLR --output-dir restored_test_outputs
```

## TTA

`--tta x8` runs an 8-way geometric self-ensemble (`src/tta.py`: all
horizontal/vertical-flip x 90-degree-rotation combinations, averaged) instead
of a single forward pass. Measured on the canonical validation split:

| Mode | PSNR | SSIM | LPIPS | Mean latency (RTX 4060 Laptop, 128x128 input) | Throughput |
| --- | ---: | ---: | ---: | ---: | ---: |
| `none` (default) | 27.9893 dB | 0.756916 | 0.302781 | 15.57 ms | 64.24 img/s |
| `x8` | 28.0355 dB | 0.758519 | 0.309627 | 50.22 ms (**3.23x slower**) | 19.91 img/s |

**`none` is the submission default.** The x8 PSNR/SSIM gain is small
(+0.046 dB / +0.0016) and LPIPS actually gets very slightly *worse* with x8
(0.302781 -> 0.309627 — TTA's averaging trades a small amount of perceptual
sharpness for pixel-fidelity gains), while inference cost more than triples.
`--tta x8` remains available for evaluators who prefer the marginal accuracy
gain over throughput. Full numbers: [`results/final_benchmark.json`](results/final_benchmark.json),
[`results/final_metrics.json`](results/final_metrics.json).

## Training

Reproduce the final champion configuration from scratch (seed 42, ~90
epochs on an RTX 4060 Laptop GPU):

```bash
python train.py --model residual_sr --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --ema --ema-decay 0.999 --epochs 90 --checkpoint-dir checkpoints/exp23_ema_extended90
```

This is the exact recipe recorded for Experiment 23 in
[EXPERIMENT_LOG.md](EXPERIMENT_LOG.md) (itself a continuation of Experiments
6/15/16/19-21 — see that file for the full lineage and every rejected
alternative). `python train.py --help` documents every option, including
several optional/experimental variants (channel attention, multi-scale and
dense-block variants, hard-patch sampling, a denoising stem, alternative
losses) that are **not** part of the submission champion — see
EXPERIMENT_LOG.md's "Next-Experiment Priority Order" for their status.

A tiny smoke run that only verifies the pipeline (not real training):

```bash
python train.py --epochs 1 --max-train-samples 16 --max-val-samples 8 --checkpoint-dir /tmp/smoke
```

## Validation / evaluation

`--checkpoint` accepts either the tracked public artifact
(`weights/residualsr_final_ema.pt` -- what a fresh clone actually has) or a
full `train.py` training checkpoint
(`checkpoints/<exp>/checkpoint_best.pt` -- only present if you trained
locally, since `checkpoints/` is gitignored). Both reconstruct the model
architecture automatically, with zero extra CLI flags:

```bash
python evaluate_checkpoint.py --checkpoint weights/residualsr_final_ema.pt --tta none --lpips
python evaluate_checkpoint.py --checkpoint weights/residualsr_final_ema.pt --tta x8 --lpips
```

Equivalently, against a locally reproduced full training checkpoint:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/exp23_ema_extended90/checkpoint_best.pt --tta none --lpips
```

Reports L1/PSNR/SSIM (`src/metrics.py`, identical formulas/clipping
convention to the bicubic baseline) plus optional LPIPS
(`--lpips`; requires the `lpips` package and its pretrained weights, listed
in `requirements.txt`; grayscale is replicated to 3 channels and mapped from
`[0,1]` to `[-1,1]` for the LPIPS network — see
`src/baseline.py::lpips_input_tensor`). LPIPS runs as a strictly separate,
additive pass and never changes the PSNR/SSIM numbers. Bicubic reference:

```bash
python evaluate_baseline.py --data-dir data/Data-public --val-fraction 0.2 --seed 42 --lpips
```

## Final metrics

Measured on the canonical 640-image validation split (seed 42,
`val_fraction=0.2`), independently reproduced during this submission audit —
see [`results/final_metrics.json`](results/final_metrics.json) and
[`submission/assets/metrics.png`](submission/assets/metrics.png):

| | PSNR (dB, higher better) | SSIM (higher better) | LPIPS (lower better) |
| --- | ---: | ---: | ---: |
| Bicubic | 23.1413 | 0.550604 | 0.4242 |
| ResidualSR (raw) | 27.9893 | 0.756916 | 0.3028 |
| ResidualSR + x8 TTA | **28.0355** | **0.758519** | 0.3096 |

Improvement over bicubic: **+4.8480 dB** PSNR (raw), **+4.8942 dB** (x8).

## Test output generation

```bash
python inference.py --input-dir data/Data-public/Test_NoisyLR/NoisyLR --output-dir restored_test_outputs
python validate_restored_outputs.py --input-dir data/Data-public/Test_NoisyLR/NoisyLR --output-dir restored_test_outputs
```

Produces 400 `.npy` outputs (one per official test input) plus
`restored_test_outputs/manifest.json`. The official test set has no locally
available ground truth, so no PSNR/SSIM/LPIPS is computed for it — see
`manifest.json`'s validation fields (shape/dtype/finiteness/count) instead.

`restored_test_outputs/` contains the final restored outputs for all 400 official
test images plus `manifest.json`, as required for hackathon submission.

These outputs were generated using the tracked final model weights at
`weights/residualsr_final_ema.pt`. The directory is approximately 102 MiB in total.

## Reproducibility

- Split seed: **42** (`src/splits.py::split_pairs`, `val_fraction=0.2` ->
  2,560 train / 640 validation), used identically for training, evaluation,
  and the bicubic baseline.
- `train.py` seeds `random`/`numpy`/`torch` (and `torch.cuda`) from
  `--seed` (default 42).
- Every checkpoint stores its full `model_config`/`training_config`/
  `loss_config`/`scheduler_config`/`ema_config`, so `--resume` reconstructs
  an interrupted run's exact state (see `train.py::load_checkpoint_for_resume`).

## Inference benchmark

```bash
python benchmark_inference.py
```

Measures model load time and, for both `--tta none` and `--tta x8`: mean/
median/min/max latency, throughput, and peak CUDA memory (warm-up iterations
run first; `torch.cuda.synchronize()` brackets every timed region). Writes
[`results/final_benchmark.json`](results/final_benchmark.json) and a Markdown
summary. **Numbers are specific to the machine the script runs on** — see
[TTA](#tta) above for the RTX 4060 Laptop GPU numbers this repository ships
with; re-run on any other machine (including an H100) for numbers specific
to it.

## License / acknowledgment

No license file is currently present in this repository. This project was
built for the KLA / i4C hackathon's **AI-Based Restoration of Degraded
Images** problem statement; add a license before any public release beyond
the hackathon submission if one is required by its rules.
