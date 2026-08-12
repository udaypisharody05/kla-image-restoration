# KLA Image Restoration — Project Handoff

This file did not exist before this milestone; it is created here to serve as the
single onboarding doc for future sessions. Keep it updated as decisions get made,
but do not delete established context below without a strong reason.

## Established project decisions (do not change without a genuine bug)

- Dataset: 3,200 paired grayscale `.npy` training samples (`float32`, `[H,W]`, 1 channel).
  - LR (`NoisyLR`) is `128x128`; GT is `256x256` (exact 2x). Raw values may fall
    outside `[0,1]`; GT is expected within `[0,1]`.
  - Discovery/pairing is generic and directory-name/content driven; see
    `src/dataset_discovery.py` (`discover_layout`, `discover_pairs`). Pairs are
    matched by identical filename stem.
- Canonical split: `src/splits.py::split_pairs(pairs, val_fraction=0.2, seed=42)` →
  2,560 train / 640 validation. Deterministic via `numpy.random.default_rng(42)`.
  Reused everywhere (bicubic baseline, PyTorch datasets, training).
- Scale factor: 2x, always.
- Training crop: LR `64x64` → GT `128x128`, spatially aligned
  (`src/transforms.py::create_training_transform`, default `augment=True` with
  paired flips/90-degree rotations). Validation always uses full images,
  `transform=None`, no augmentation — stays comparable to the bicubic baseline.
- Dataset (`data/`) and checkpoints (`checkpoints/`, `*.pt`, `*.pth`, `*.ckpt`) are
  excluded from Git via `.gitignore`. Do not commit either.
- Bicubic validation baseline (`evaluate_baseline.py`, `results/bicubic_baseline.json`):
  **PSNR 23.1413 dB, SSIM 0.550604** (grayscale SSIM via scikit-image, `data_range=1.0`,
  prediction clipped to `[0,1]` for metrics only). This is the number every learned
  model should be compared against — do not claim improvement without measuring it.
- Test conventions: `pytest -m "not integration"` runs fast, dataset-free,
  GPU-free unit tests (uses `tmp_path` + synthetic arrays). `pytest -m integration`
  exercises the real ~3,200-sample dataset under `data/` (or `SEMICON_DATA_DIR`) and
  is slow (~38 minutes for the full suite); it is not run automatically by agents
  unless a change specifically requires it.
- Docker: `Dockerfile` runs the unit test suite by default; not redesigned here.

## Milestone: first trainable neural baseline (added this session)

Added a small residual CNN and the minimum training/evaluation infrastructure to
verify the learning pipeline end-to-end. Deliberately simple — no GANs, no
perceptual/VGG loss, no transformers, no pretrained weights.

- `src/models/residual_sr.py` — `ResidualSRNet`: conv_in → 4 `ResidualBlock`s (2x
  conv3x3 + identity skip) → conv → global skip around the residual body →
  conv → `nn.PixelShuffle(2)`. Defaults: `in_channels=1, out_channels=1,
  num_features=32, num_blocks=4, scale=2` (matches the grayscale dataset, not the
  "RGB" wording in generic SR task templates — this dataset is 1-channel).
  ~84.7k trainable parameters.
- `src/metrics.py` — `psnr()`/`ssim()` accept batched `[B,C,H,W]` (or unbatched
  `[C,H,W]`) torch tensors and internally call the *existing*
  `src.baseline.peak_signal_noise_ratio`/`structural_similarity_index` per image,
  so neural validation numbers use the exact same formulas as the bicubic
  baseline. No new metric dependency added.
- `train.py` — CLI training entry point. Reuses `discover_layout`/`discover_pairs`,
  `split_pairs`, `PairedRestorationDataset`, `create_dataloader`,
  `create_training_transform`. Defaults: `L1Loss`, Adam, `lr=1e-4`, `seed=42`,
  `val_fraction=0.2`, `crop_size=64`, `scale=2`. Device auto-selects CUDA else CPU
  and prints the choice. Seeds `random`/`numpy`/`torch` for reproducibility.
  Saves `checkpoint_latest.pt` every epoch and `checkpoint_best.pt` whenever
  validation PSNR improves, each containing model/optimizer state, epoch,
  best validation PSNR, and model/training config.
- `evaluate_checkpoint.py` — loads a checkpoint, reconstructs `ResidualSRNet`
  from its stored `model_config`, evaluates the fixed validation split, prints
  L1/PSNR/SSIM plus the delta versus the bicubic baseline. No test-set inference.
- Tests added (all fast, no dataset, no GPU): `tests/test_model_unit.py`,
  `tests/test_metrics_unit.py`, `tests/test_training_unit.py`. 55 unit tests pass
  total (`pytest -m "not integration"`).
- Smoke-verified: `python train.py --epochs 1 --max-train-samples 16
  --max-val-samples 8 --checkpoint-dir <tmp>` against the real dataset — dataset
  loading, forward/backward, optimizer step, validation, and both checkpoints all
  confirmed working. Val PSNR after this trivial smoke run (~6.4 dB) is far below
  bicubic, as expected for 16 samples / 1 epoch with a freshly initialized network
  — not a real training result.

### Next step (not yet run)

A real training experiment, e.g.:

```bash
python train.py --data-dir data/Data-public --epochs 20 --batch-size 16 --lr 1e-4 --checkpoint-dir checkpoints
```

Then compare with:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/checkpoint_best.pt --data-dir data/Data-public
```

Epoch count/architecture width have not been tuned — this milestone only
establishes that the pipeline works, not that the model is competitive.

## Optional GPU thermal guard (infrastructure, not an experiment)

Thermally constrained NVIDIA laptops (this project's dev machine sustains ~87-90°C
under long training runs) can optionally use `src/thermal.py`'s `GpuTemperatureGuard`
via `train.py`:

```bash
python train.py ... --gpu-temp-limit 82 --gpu-temp-resume 78 --gpu-temp-check-interval 5 --gpu-temp-poll-seconds 3
```

Disabled by default (`--gpu-temp-limit 0`, the default) — in that case `train.py` never
calls `nvidia-smi` and behaves exactly as before this feature existed. When enabled, it
pauses only between fully completed training/validation batches (never mid-batch),
reading GPU temperature via `nvidia-smi` every `--gpu-temp-check-interval` batches, and
sleeps in `--gpu-temp-poll-seconds` increments until temperature drops to
`--gpu-temp-resume` (a lower resume threshold than the pause limit avoids rapid
pause/resume toggling). This changes wall-clock time only — model, optimizer, loss,
scheduler, data order, and metrics are completely unaffected, and resuming a checkpoint
with different thermal settings than it was originally trained with is always legal
(thermal settings are recorded in `training_config` for reference only, never checked on
resume).

## Current best model (updated after Experiment 23; see EXPERIMENT_LOG.md for full history)

**Experiment 23 (`checkpoints/exp23_ema_extended90/checkpoint_best.pt`, epoch 90)
is the current champion checkpoint** — ResidualSRNet, 64 features / 8 residual
blocks (630,724 params), same recipe as Experiment 6 but with EMA (decay 0.999)
and training extended to 90 epochs. Independently verified: Val PSNR 27.9893 dB,
Val SSIM 0.756916, Val L1 0.032443 (non-TTA). **Experiment 23 + x8 TTA is the
current best overall pipeline**: Val PSNR **28.0355 dB**, Val SSIM **0.758519**,
Val L1 **0.032264** (+4.8942 dB / +0.207915 over the 23.1413 dB / 0.550604
bicubic baseline). Note `evaluate_checkpoint.py` loads EMA weights by default
when a checkpoint has them, so these numbers come straight from the normal
command.

Experiment 16 (`checkpoints/exp16_extended70/checkpoint_best.pt`, epoch 65) was
the previous champion at 27.7656 dB non-TTA / 27.8154 dB with x8 TTA.

Experiment 6 (`checkpoints/exp6_crop96/checkpoint_best.pt`) remains the original,
from-scratch champion and the recipe every later experiment is still compared
against: ResidualSRNet, 64F/8B, L1, Adam, ReduceLROnPlateau, 96x96 LR crop, 40
epochs, seed 42. Independently verified: Val PSNR 27.7090 dB, Val SSIM 0.745634,
Val L1 0.033420 (+4.5677 dB / +0.195030 over bicubic).

- Experiment 5 (`checkpoints/exp5_l1_ssim/checkpoint_best.pt`) has the highest SSIM on
  record (0.747377) but lower PSNR (27.5282 dB) — kept as the best-SSIM reference only,
  not the overall pick.
- Experiment 7 (`checkpoints/exp7_crop128/checkpoint_best.pt`, full 128x128-image
  training crop) is **completed but neutral**: numerically the highest PSNR
  (27.7101 dB, +0.0011 dB over Experiment 6 — within noise), but slightly worse SSIM
  (0.743748) and substantially slower per epoch (~38-42s vs ~25-27s), for no real gain.
  It does not replace Experiment 6 as the recommended configuration.
- **Recommendation for future experiments: use a 96x96 LR training crop
  (`--crop-size 96`)**, not 64x64 (Experiments 1-5) or the full 128x128 image
  (Experiment 7) — Experiment 6 already captured the benefit of extra crop context;
  going further did not help.

### Experiment 8 (completed — negative result; MSE loss)

`--loss mse` (`checkpoints/exp8_mse/checkpoint_best.pt`, epoch 13) was screened for 15
of the planned 40 epochs and **stopped deliberately** after underperforming Experiment 6
by -0.4931 dB PSNR (27.2159 dB vs 27.7090 dB) and -0.018161 SSIM (0.727473 vs
0.745634), with early signs of plateauing rather than catching up. The checkpoint is
retained (not deleted) for reproducibility, but training was not continued to 40 epochs
since the gap was already decisive. **L1 remains the preferred reconstruction loss** —
Experiments 4 (Charbonnier), 5 (L1+SSIM), and 8 (MSE) have now all been tried against L1
without beating Experiment 6 on PSNR.

### Experiment 9 (completed — negative result; EDSR-lite architecture)

`EDSRLite` (`src/models/edsr_lite.py`, a new architecture -- `ResidualSRNet` is
untouched) was trained with Experiment 6's exact recipe substituted onto a stronger
64-feature/16-block/1,367,553-parameter (2.1682x) EDSR-style network. Independently
verified (`checkpoints/exp9_edsr_lite/checkpoint_best.pt`, epoch 36): Val PSNR
27.5658 dB, Val SSIM 0.742162 -- **0.1432 dB lower and 0.003472 lower** than
Experiment 6, despite 2.17x the parameters and ~2x the epoch time. The model trained
correctly (no implementation failure); the extra depth/capacity simply didn't help
generalization on this dataset size. The checkpoint is retained for reproducibility.
**Experiment 6 remains the practical champion.** A centralized model factory
(`src.models.build_model_config`/`build_model`) now backs `train.py`,
`evaluate_checkpoint.py`, and `infer_test.py` so both architectures share one
reconstruction path -- useful infrastructure kept regardless of this result.

### Experiment 10 (completed — small measured win; x8 geometric TTA)

With both loss substitution (Experiments 4, 5, 8) and architecture scaling
(Experiment 9) failing to beat Experiment 6 on retraining, Experiment 10 tested a
**no-retraining, inference-time** technique instead: x8 geometric self-ensembling
(`src/tta.py::predict_x8`, `--tta x8` on `evaluate_checkpoint.py`/`infer_test.py`).
Measured on the full 640-image validation set: Experiment 6 improved from
27.7090 dB / 0.745634 to **27.7689 dB / 0.747955** (+0.0599 dB, +0.002321 SSIM);
Experiment 9 improved similarly but more modestly (+0.0217 dB, +0.000818 SSIM). Both
gains are real and reproducible, with no SSIM regression, at the cost of ~8x
inference compute (no training cost). **x8 TTA is a worthwhile optional inference
step on top of Experiment 6's checkpoint**, not a replacement for it -- the
underlying champion checkpoint is unchanged. Use `--tta x8` when generating final
predictions if the ~8x inference cost is acceptable; use `--tta none` (default) for
fast iteration/screening.

Experiment 6 (`checkpoints/exp6_crop96/checkpoint_best.pt`, L1, 96x96 crop) remains
the underlying champion checkpoint: Val PSNR 27.7090 dB, Val SSIM 0.745634 (normal),
27.7689 dB / 0.747955 with `--tta x8`.

### Experiment 11 (completed — rejected; model ensembling)

With loss substitution, architecture scaling, and TTA all tried, Experiment 11
tested weighted-averaging Experiment 6's and Experiment 9's **raw predictions**
(`src/ensemble.py::weighted_average_predictions`, `evaluate_ensemble.py`) to see
if their errors were complementary enough to beat Experiment 6 + x8 TTA. They were
not: every weight tested (0.50/0.50, 0.75/0.25, 0.875/0.125 Exp6/Exp9, each with
and without x8 TTA) scored below its single-model reference on PSNR, SSIM, and L1,
converging toward — but never past — Experiment 6 alone as Experiment 9's weight
shrank toward zero. Best configuration (0.875 Exp6 + 0.125 Exp9, x8 TTA): PSNR
27.7561 dB, SSIM 0.747603 — still **-0.0128 dB / -0.000352 SSIM** below Experiment
6 + x8 TTA alone. **Ensembling with Experiment 9 is rejected; Experiment 6 + x8
TTA remains the best inference pipeline.** `src/ensemble.py` and
`evaluate_ensemble.py` are retained as reusable infrastructure for a future
architecturally-different candidate (e.g. NAFNet/SwinIR/Restormer), should one be
trained later.

### Experiment 12 (completed — rejected; NAFNet-SR architecture, stopped @ epoch 32)

A genuinely different feature-processing design: `src/models/nafnet_sr.py`
implements NAFNet-style gated blocks locally (channel-wise `LayerNorm2d`,
`SimpleGate` split-and-multiply as the only in-block nonlinearity — no
ReLU/GELU, simplified channel attention, learnable zero-initialized per-branch
residual scales) wrapped in the same shallow-conv / long-skip /
PixelShuffle-upsample skeleton as `ResidualSRNet`/`EDSRLite`. Wired into the
shared model factory as `architecture="nafnet_sr"`; legacy and `edsr_lite`
checkpoints unaffected, all 5 cross-architecture resume mismatches rejected.

**Sizing note:** an initial 96-feature/12-block candidate (1.23M params) was
rejected by the CUDA sanity check — NAFNet-style blocks carry far more
*activation* memory per parameter than this project's other architectures
(~10 sequential ops per block vs. 2), and needed ~13 GB at batch16/crop96,
exceeding the 8 GB RTX 4060 Laptop GPU. The architecture was resized down
(batch/crop left untouched) to **64 features / 8 NAF blocks, 432,129
parameters**, which fit safely (~6.47 GB peak reserved, ~322 ms/batch).

**Real result:** trained with the full controlled recipe (L1, crop96, batch16,
seed42, Adam, ReduceLROnPlateau) and **stopped deliberately at epoch 32 of 40**
after clear plateauing (+0.0464 dB across epochs 25→32).
`checkpoints/exp12_nafnet_sr/checkpoint_best.pt`, independently re-verified:
**Val PSNR 27.2178 dB, Val SSIM 0.729829, Val L1 0.035279** — **-0.4912 dB PSNR
/ -0.015805 SSIM** below Experiment 6. **NAFNet-SR is rejected.** This is the
third architecture/scaling/ensembling attempt to beat Experiment 6 that has
failed (after Experiment 9 and Experiment 11). **Experiment 6 remains the
champion checkpoint; Experiment 6 + x8 TTA (27.7689 dB / 0.747955) remains the
best inference pipeline.** Both Experiment 12 checkpoints are retained,
unmodified, for reproducibility.

### Experiment 13 (completed — rejected; SwinIR-lite architecture)

Window-based self-attention (`src/models/swinir_lite.py`, embed_dim=60,
depth=6, num_heads=6, window_size=8, mlp_ratio=2.0, 348,421 params), trained
with Experiment 6's exact recipe (L1, crop96, batch16, seed42, Adam,
ReduceLROnPlateau, 40 epochs). Independently verified
(`checkpoints/exp13_swinir_lite/checkpoint_best.pt`, epoch 38): Val PSNR
27.4361 dB, Val SSIM 0.738432 — **0.2729 dB lower and 0.007202 lower** than
Experiment 6. The model trained correctly (no divergence, responded normally
to LR reductions); windowed self-attention simply didn't outperform the
proven residual-CNN design at this scale/dataset size. **Rejected.** This is
the fourth architecture/ensembling attempt (after Experiments 9, 11, 12) that
failed to beat Experiment 6, motivating Experiment 14's pivot away from
architecture search toward the training recipe itself.

### Experiment 14 (completed — rejected; cosine LR schedule)

With four architecture/ensembling attempts exhausted, Experiment 14 held
Experiment 6's architecture fixed and changed exactly one variable:
`ReduceLROnPlateau` → `CosineAnnealingLR` (`T_max=40, eta_min=1e-6`).
Independently verified (`checkpoints/exp14_cosine/checkpoint_best.pt`, epoch
38): Val PSNR 27.6011 dB, Val SSIM 0.742668 — **0.1079 dB lower and 0.002966
lower** than Experiment 6's `ReduceLROnPlateau` result. The smallest gap of
any rejected experiment so far, but still a clear regression, not noise.
**Rejected — `ReduceLROnPlateau` remains this project's scheduler of
choice.** `train.py`'s scheduler infrastructure now cleanly supports both
scheduler types (`build_scheduler_config`/`build_scheduler`/`scheduler_step()`
dispatch, with `T_max` always explicit and never derived from `--epochs`,
and resume compatibility mirroring the existing `loss_config` strict-match
pattern) — useful infrastructure kept regardless of this result.

### Experiments 15 & 16 (completed — champion improved; extended training)

With scheduler substitution (Exp 14) joining architecture/ensembling (Exp 9,
11, 12, 13) as a failed attempt to beat Experiment 6, Experiments 15/16
asked a simpler question instead: does the champion configuration benefit
from continued `ReduceLROnPlateau`-controlled training past its original
40-epoch budget? Both are direct **continuations** of Experiment 6's own
run (`--resume`/`--checkpoint-dir` used as already-independent CLI args, no
code changes), not a new architecture or recipe: Experiment 15 resumed from
`checkpoints/exp6_crop96/checkpoint_latest.pt` (epoch 40, LR 5e-05) to
epoch 60; Experiment 16 resumed from `checkpoints/exp15_extended60/checkpoint_latest.pt`
to epoch 70. Verified results:

- **Exp 15** (epoch 60): Val PSNR 27.7626 dB, Val SSIM 0.748636 — a modest
  +0.0536 dB gain over Experiment 6.
- **Exp 16** (best epoch 65 of 70; scheduler reached `min_lr=1e-6` by epoch
  70): Val PSNR 27.7656 dB, Val SSIM 0.748618 — only +0.0030 dB over Exp 15,
  essentially noise. **Epoch-extension is now saturated** — the one modest
  gain available (Exp 15) has been captured; further blind extension is not
  expected to help.
- **Exp 16 + x8 TTA: PSNR 27.8154 dB, SSIM 0.750571, L1 0.032998** — this is
  now the **current best overall pipeline**, surpassing Experiment 6 + x8
  TTA (27.7689 dB / 0.747955). `checkpoints/exp16_extended70/checkpoint_best.pt`
  (epoch 65) is the new champion checkpoint.

All of Experiments 6, 15, and 16's checkpoints are retained unmodified.

### Experiment 17 (completed — rejected individually, reused in Exp 18; bicubic residual learning)

Tests whether explicitly learning only the restoration residual over a fixed
bicubic 2x upsample (`prediction = fixed_bicubic_upsample(LR) +
learned_residual_branch(LR)`) improves over direct HR prediction.
`src/models/residual_sr_bicubic.py::ResidualSRBicubic` reuses
`ResidualSRNet`'s exact learned-branch topology (same `ResidualBlock`, same
conv/PixelShuffle layout, **630,724 params — identical to Exp 6**, since the
bicubic term has zero trainable parameters) with the sum computed at the
very end and never clipped. The bicubic skip uses `torch.nn.functional.interpolate(
mode="bicubic")`, not this project's PIL-based bicubic baseline
(`src.baseline.bicubic_upscale`, left untouched) — the two are **not**
bit-identical (measured max abs diff 0.0644 on a `[0,1]`-range test image),
a documented, deliberate difference since PIL can't run efficiently inside a
GPU training loop. `residual_sr`↔`residual_sr_bicubic` resume mismatches are
rejected even though the underlying tensor shapes are identical (config
equality, not shape equality, gates resume).

**Real result** (trained from scratch, 60 epochs, `checkpoints/exp17_bicubic_residual/checkpoint_best.pt`,
epoch 60): Val PSNR 27.7460 dB / SSIM 0.748557 (non-TTA); +x8 TTA: PSNR
27.7942 dB / SSIM 0.750434. **Closest any alternative has come to the
champion** (-0.0196 dB non-TTA, -0.0212 dB with x8, vs. Experiment 16) but
still did not win. **Rejected as an individual model**; checkpoint retained
and reused directly as Experiment 18's second ensemble member.

### Experiment 18 (completed — rejected; Exp16 + Exp17 model ensemble)

Inference-only (no training). Reused Experiment 11's ensemble infrastructure
(`src/ensemble.py`, `evaluate_ensemble.py`) unmodified — it already supported
`residual_sr_bicubic` automatically via the shared model factory. Three
pre-declared non-TTA weights tested: 50/50 (27.7845 dB), 75/25 (27.7822 dB),
87.5/12.5 (27.7757 dB). Top two (50/50, 75/25) were within the pre-declared
0.01 dB tie-break threshold, so both were evaluated with x8: 50/50+x8 =
27.8111 dB; **75/25+x8 = 27.8149 dB (best ensemble)**. This is **0.0005 dB
below** Experiment 16 + x8's 27.8154 dB — essentially a tie (SSIM/L1 were
marginally better for the ensemble) but does not clear the pre-declared
primary-metric (PSNR) success bar. **Ensembling rejected.**

**Experiment 16 + x8 TTA remains the overall champion pipeline: PSNR
27.8154 dB, SSIM 0.750571, L1 0.032998.** No further experiments were
started automatically.

### Experiment 19 (prepared, not yet run — EMA weight averaging)

Tests whether an exponential moving average (EMA) of `ResidualSRNet`'s
weights improves validation PSNR, as the sole variable against Experiment
6's proven recipe (from scratch, 60 epochs). `src/ema.py::ExponentialMovingAverage`
is generic training infrastructure (works with any `nn.Module`; `ResidualSRNet`
itself is unmodified) — initializes a shadow copy from the model's current
(never zero) weights, and updates it once per optimizer step:
`ema_param = decay * ema_param + (1 - decay) * live_param` (verified
numerically: 1→3 at decay=0.9 gives exactly 1.2). Training loss always uses
the live weights; **validation (and therefore the scheduler decision and
best-checkpoint selection) uses the EMA shadow instead** — a one-line change
in the training loop (`eval_model = ema.shadow_model if ema else model`),
with `validate()` itself completely unmodified.

Checkpoints gained two new keys: `ema_state_dict` (the EMA shadow's weights,
`None` when disabled) and `ema_config` (`{"enabled": True, "decay": 0.999}`
or `None`) — `model_state_dict` still always means the live/raw weights, in
every checkpoint, so no historical loader changes. `evaluate_checkpoint.py`'s
`load_model()` gained `prefer_ema: bool = True`: when EMA state is present,
those weights load automatically (matching the checkpoint's recorded PSNR)
with zero CLI/code changes needed in `evaluate_checkpoint.py`/`infer_test.py`
themselves. Resume mismatch checks (EMA↔non-EMA, different decay) are
strict and reject cleanly, while historical non-EMA checkpoints resume
exactly as before — `--ema`/`--ema-decay` default off. **Next step: the
real 60-epoch run**, e.g.:

```bash
python train.py --model residual_sr --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --ema --ema-decay 0.999 \
  --epochs 60 --checkpoint-dir checkpoints/exp19_ema
```

then compare against the current champion pipeline, Experiment 16 + x8 TTA
(27.8154 dB / 0.750571 / 0.032998). **This run has not been started.**

### ⚠ Experiments 19-21 (EMA runs) executed but never written up

Observed 2026-08-11 while running the Experiment 21 degradation analysis, from
checkpoint metadata only (nothing re-evaluated, nothing altered):

| checkpoint | epoch | stored `best_val_psnr` |
| --- | ---: | ---: |
| `checkpoints/exp19_ema/` | 60 | 27.828210 |
| `checkpoints/exp20_ema_extended70/` | 70 | 27.884964 |
| `checkpoints/exp21_ema_extended80/` | 80 | 27.938326 |

EMA clearly helped (Exp 16's non-TTA champion was 27.7656 dB) and the trend
continued into Experiment 23. These three runs still need proper write-ups with
an independent `evaluate_checkpoint.py` pass. **Numbering was resolved on
2026-08-11**: the degradation analysis is Experiment 22 (it had briefly shared
"21" with `exp21_ema_extended80`), Experiment 23 is the EMA e90 champion, and
Experiment 24 is synthetic noise augmentation.

### Experiment 22 (completed — analysis only; dataset degradation forensics)

**Not a training experiment.** `src/degradation.py` + `analyze_degradation.py`
characterize the GT 256x256 -> NoisyLR 128x128 process across all 3,200 training
pairs. Full outputs in
[`results/degradation_analysis/`](results/degradation_analysis/degradation_report.md).
Two findings dominate:

1. **Noise is strongly signal dependent** (multiplicative/speckle-like): residual
   std climbs 0.0120 -> 0.1631 across intensity (13.6x), fitting
   `var(I) = -6.19e-05 + 0.00653·I + 0.0201·I²` at **R² = 0.9995**. Plain L1
   assumes homoscedastic noise, so the current recipe mis-weights the image.
   Residual std (0.0897) is ~2.8x the champion's validation L1 — the task is
   **noise-dominated, not resolution-dominated**.
2. **119 repeated-scene groups covering 250 images** (same GT, independent noise
   draws), all recoverable from filenames (ID gap ≤ 2). **36 groups straddle the
   canonical split**, giving 38/640 (5.9%) validation images a train twin — so
   absolute validation numbers are slightly optimistic. Cross-experiment
   comparisons remain valid (identical split throughout) and **the split was not
   changed**.

Ruled out with evidence: fixed-pattern noise (ratio 1.01), pre-downsampling blur
(0.328%), gain/bias calibration (0.14%), spatial correlation (max 0.051),
frequency-domain structure, and discrete degradation regimes.

**Recommendations it produced, ranked:** (1) variance-stabilizing transform or
`1/sqrt(var(I))`-weighted loss — **HIGH**; (2) signal-dependent synthetic-noise
augmentation — **MEDIUM-HIGH**; (3) scene-group-aware training/validation —
**MEDIUM**. Recommendation (2) became Experiment 24; recommendation (3) is now
available as the standalone diagnostic `evaluate_group_aware.py`.

### Experiment 23 (completed — current protected champion; EMA to epoch 90)

`checkpoints/exp23_ema_extended90/checkpoint_best.pt` (epoch 90), ResidualSRNet
64F/8B + EMA decay 0.999. Independently re-verified this session:
**non-TTA 27.9893 dB / 0.756916 / 0.032443**, **+ x8 TTA 28.0355 dB / 0.758519 /
0.032264**. This is the number Experiment 24 must beat.

Secondary group-aware diagnostic (`evaluate_group_aware.py`): 28.2333 dB on a
leakage-free split vs 27.9893 dB canonical. **Do not read that +0.24 dB as
"leakage was costing us accuracy"** — it is a *different subset of images*, so
per-image difficulty dominates the comparison. The group-aware metric is only
meaningful compared against itself across experiments; canonical stays
authoritative.

### Experiment 24 (completed — REJECTED; signal-dependent synthetic noise)

Adds synthetic noisy-LR training inputs drawn from the Experiment 22
degradation model (`src/synthetic_noise.py`), as its MEDIUM-HIGH recommendation.
`--synthetic-noise-prob 0.5` (default `0.0` = historical behavior); validation
stays 100% real by construction — `build_datasets` never hands the augmentation
to the validation dataset.

Key decisions, all evidence-backed by `analyze_synthetic_noise.py`
([report](results/synthetic_noise_analysis/synthetic_noise_report.json)):

- **Gaussian epsilon, not Student-t.** The standardized residual looks
  heavy-tailed (excess kurtosis +3.46 → t with ν≈5.7), but heteroscedastic
  mixing *alone* already produces +2.45 excess kurtosis in the pooled residual.
  Gaussian lands at +2.45 vs the real +3.52; Student-t overshoots to +8.54 and
  is also worse on percentiles (0.0198 vs 0.0144). Much of the apparent tail
  weight is error in `sigma(I)`, not real tail weight in epsilon.
- **Match quality:** synthetic std 0.0850 vs real 0.0848; 37/40 intensity bins
  within 10%; median ratio 1.004.
- **Two documented limitations:** skewness is unmatched (real +0.407 vs ~0), and
  the darkest bin under-noises 2.5x because the fitted quadratic's constant is
  negative and clamps to zero below I=0.0092. `--synthetic-noise-variance-floor
  1.43e-4` fixes the latter (worst bin 0.449 → ~1.21) but the default stays
  `0.0` to reproduce the Experiment 22 model verbatim.
- **Alignment:** substitution happens at full resolution *before* cropping, so
  the existing aligned-crop transform is reused unchanged and GT/LR cannot
  desynchronize.
- **Reproducibility:** `numpy.random.default_rng(SeedSequence([seed, epoch,
  index]))` — no global RNG, worker-count independent, fresh realization each
  epoch via `train_dataset.set_epoch(epoch)`.

**Real run result (`checkpoints/exp24_noise_aug/`, 90 epochs, from scratch):**

| split | PSNR (dB) | SSIM | L1 |
| --- | ---: | ---: | ---: |
| canonical, non-TTA | 27.8721 | 0.747690 | 0.032828 |
| canonical, x8 TTA | 27.9116 | 0.749092 | 0.032667 |
| group-aware (diagnostic) | 28.1264 | 0.748362 | 0.030632 |

Trails the Experiment 23 champion (28.0355 / 0.758519 / 0.032264) on every
metric, canonical and group-aware alike. **Rejected — do not extend this
experiment or tune `--synthetic-noise-prob` further.** Experiment 25 (below)
tests a different hypothesis instead.

### Experiment 25 (prepared, not yet run — noise-conditioned ResidualSR)

Instead of substituting synthetic inputs (Experiment 24's rejected approach),
gives the model an explicit second input channel: a per-pixel
signal-dependent sigma estimate from the same Experiment 22 model, computed
from the real NoisyLR itself. Trains exclusively on real NoisyLR/GT pairs —
mutually exclusive with `--synthetic-noise-prob`.

```
channel 0 = NoisyLR                        (real, never clamped)
channel 1 = sigma(clamp(NoisyLR, 0, 1))     (Experiment 22 model, raw sigma, no normalization)
```

`ResidualSRNet` is unchanged apart from `in_channels=2` (631,300 params vs
630,724). New module `src/noise_conditioning.py` provides
`conditioning_sigma_map`/`prepare_model_input` (reusing
`src/synthetic_noise.py`'s variance/sigma formulas) plus a
`NoiseConditionedModel` wrapper whose standard `nn.Module` composition makes
`--noise-conditioning` work transparently through EMA, x8 TTA,
`evaluate_checkpoint.py`, `infer_test.py`, and `evaluate_group_aware.py` with
**zero changes to any of those files**. x8 TTA spatial consistency (sigma
computed after each D4 transform vs. concatenate-then-transform) verified
numerically identical (`torch.allclose(..., atol=1e-6)`, max diff `0.0`).

35 new tests, full fast suite 560 passed / 8 deselected. CUDA sanity (64F/8B,
`in_channels=2`, batch16, crop96, EMA 0.999): finite forward/backward, EMA
update, peak 211.8 MiB allocated, no OOM. Real-data smoke test (32
train/16 val, 2 epochs, `checkpoints/exp25_noise_conditioned_smoke`, deleted
after): real-only data confirmed, EMA/resume/mismatch-rejection all verified,
`evaluate_checkpoint.py`/`infer_test.py`/`evaluate_group_aware.py` all working
with the conditioned checkpoint. See EXPERIMENT_LOG.md for full detail.

**Next step: the real run** (not started; `checkpoints/exp25_noise_conditioned/`
does not exist):

```bash
python train.py --model residual_sr --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --ema --ema-decay 0.999 --noise-conditioning \
  --epochs 90 --checkpoint-dir checkpoints/exp25_noise_conditioned
```
