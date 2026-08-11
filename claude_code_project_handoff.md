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

## Current best model (updated after Experiment 7; see EXPERIMENT_LOG.md for full history)

**Experiment 6 (`checkpoints/exp6_crop96/checkpoint_best.pt`) is the practical preferred
configuration** — ResidualSRNet, 64 features / 8 residual blocks (630,724 params), L1
loss, Adam, ReduceLROnPlateau, 96x96 LR training crop, 40 epochs, seed 42. Independently
verified: Val PSNR 27.7090 dB, Val SSIM 0.745634, Val L1 0.033420 (+4.5677 dB / +0.195030
over the 23.1413 dB / 0.550604 bicubic baseline).

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

### Experiment 13 (prepared, not yet run — SwinIR-lite architecture)

Next architecture direction: **window-based self-attention** rather than
another convolutional design, on the hypothesis that windowed self-attention
may preserve long-range structural consistency and fine semiconductor edges
better than the CNN-only architectures tried so far (Experiments 1-9, 12).
See `EXPERIMENT_LOG.md`'s Experiment 13 entry for the exact chosen
configuration, CUDA sizing candidates, and the not-yet-run screening command.
**This run has not been started.**
