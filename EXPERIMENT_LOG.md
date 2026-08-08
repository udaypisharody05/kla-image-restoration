# KLA Image Restoration — Experiment Log

This document records validated experiments for the KLA Track 1 image restoration project.

The purpose of this log is to preserve model configurations, training settings, validation results, and conclusions so that future experiments can be compared fairly.

---

# Reference Dataset Configuration

These settings are fixed across experiments unless explicitly documented otherwise.

```text
Training/validation pairs: 3200
Training samples:          2560
Validation samples:         640
Split seed:                  42

Scale factor:                 2×

Training LR crop:          64×64
Training GT crop:        128×128

Validation:
Full images
No random augmentation
LR:  128×128
GT:  256×256
```

All experiments should use the existing canonical split in `src/splits.py`.

The test set must not be used for model selection or hyperparameter tuning.

---

# Classical Reference — Bicubic Baseline

## Method

Standard bicubic 2× upsampling.

## Validation Results

```text
PSNR: 23.1413 dB
SSIM: 0.550604
```

This is the minimum reference that neural restoration models should be compared against.

---

# Experiment 1 — Small Residual CNN Baseline

## Status

**Completed and independently verified**

## Objective

Establish the first trainable neural baseline and verify the complete restoration pipeline:

```text
Dataset
→ aligned preprocessing
→ DataLoader
→ neural network
→ forward pass
→ L1 loss
→ backpropagation
→ optimizer
→ validation
→ PSNR / SSIM
→ checkpointing
→ checkpoint reload
```

The purpose of this experiment was to establish a simple reproducible baseline before attempting stronger architectures or more advanced training strategies.

---

## Architecture

Model:

```text
ResidualSRNet
```

Architecture:

```text
Noisy LR image
      ↓
3×3 Conv
1 → 32 channels
      ↓
4 Residual Blocks
      ↓
3×3 Conv
      ↓
Global residual feature skip
      ↓
Upsampling convolution
32 → 4 channels
      ↓
PixelShuffle ×2
      ↓
Restored grayscale image
```

Each residual block:

```text
Input
  ↓
3×3 Conv
  ↓
ReLU
  ↓
3×3 Conv
  ↓
+ Identity
```

Model configuration:

```text
Input channels:       1
Output channels:      1
Feature channels:    32
Residual blocks:      4
Upsampling:     PixelShuffle ×2

Trainable parameters: 84,708
```

The model uses one input/output channel because the competition dataset contains grayscale `.npy` images.

No pretrained weights or external model downloads are used.

---

# Training Configuration

```text
Loss:              L1Loss
Optimizer:         Adam
Learning rate:     1e-4
Batch size:        16
Epochs:            20

Scale:             2×
LR crop:           64×64
GT crop:           128×128

Split seed:        42

Training samples:  2560
Validation:         640
```

Device used:

```text
NVIDIA GeForce RTX 4060
CUDA enabled
```

Training automatically selects CUDA when available.

---

# Metric Convention

PSNR and SSIM use the same underlying metric implementation and image-range convention as the bicubic baseline.

Image range:

```text
[0, 1]
```

For PSNR and SSIM:

```text
Model prediction → clipped to [0,1] → metric calculation
```

Only the prediction is clipped.

The GT image is not clipped.

L1 loss remains calculated on the raw model output and is therefore not clipped.

This preserves useful training gradients while keeping PSNR/SSIM directly comparable with the bicubic baseline.

---

# Epoch Results

|  Epoch |     Train L1 |       Val L1 | Val PSNR (dB) |     Val SSIM |
| -----: | -----------: | -----------: | ------------: | -----------: |
|      1 |     0.088815 |     0.048694 |       23.6115 |     0.595291 |
|      2 |     0.044181 |     0.042451 |       25.5132 |     0.642734 |
|      3 |     0.039738 |     0.040620 |       25.9597 |     0.659896 |
|      4 |     0.037839 |     0.039191 |       26.2918 |     0.675988 |
|      5 |     0.036669 |     0.038279 |       26.5005 |     0.689808 |
|      6 |     0.035920 |     0.037962 |       26.5759 |     0.698078 |
|      7 |     0.035239 |     0.037680 |       26.6367 |     0.701402 |
|      8 |     0.034826 |     0.037707 |       26.6143 |     0.703875 |
|      9 |     0.034622 |     0.036726 |       26.8533 |     0.710985 |
|     10 |     0.034464 |     0.036565 |       26.8776 |     0.713732 |
|     11 |     0.034193 |     0.036428 |       26.9120 |     0.716011 |
|     12 |     0.034332 |     0.036515 |       26.8788 |     0.715718 |
|     13 |     0.034067 |     0.036597 |       26.8616 |     0.717403 |
|     14 |     0.033837 |     0.036078 |       27.0012 |     0.717686 |
|     15 |     0.033685 |     0.036117 |       26.9800 |     0.721288 |
|     16 |     0.033798 |     0.036091 |       26.9880 |     0.720918 |
|     17 |     0.033595 |     0.036013 |       26.9910 |     0.722599 |
|     18 |     0.033665 |     0.035781 |       27.0595 |     0.722526 |
|     19 |     0.033416 |     0.035878 |       27.0375 |     0.723119 |
| **20** | **0.033444** | **0.035668** |   **27.0870** | **0.725385** |

---

# Best Model

Best validation PSNR occurred at:

```text
Epoch: 20
```

Checkpoint:

```text
checkpoints/checkpoint_best.pt
```

Verified independently using:

```powershell
python evaluate_checkpoint.py --checkpoint checkpoints/checkpoint_best.pt --data-dir data/Data-public
```

Independent evaluation result:

```text
Using device: cuda

Validation samples: 640

Val L1:   0.035668
Val PSNR: 27.0870 dB
Val SSIM: 0.725385
```

The independently loaded checkpoint exactly reproduced the metrics recorded during training.

This confirms that:

```text
Model checkpoint loading       ✅
Model reconstruction           ✅
Validation split               ✅
Validation preprocessing       ✅
PSNR implementation            ✅
SSIM implementation            ✅
Best-checkpoint selection      ✅
```

---

# Comparison With Bicubic

| Metric |    Bicubic |   Residual CNN |    Improvement |
| ------ | ---------: | -------------: | -------------: |
| PSNR   | 23.1413 dB | **27.0870 dB** | **+3.9457 dB** |
| SSIM   |   0.550604 |   **0.725385** |  **+0.174781** |

The first neural baseline therefore clearly outperforms bicubic interpolation on the fixed validation set.

---

# Training Behaviour

The model learned rapidly during the first few epochs.

Representative PSNR progression:

```text
Epoch 1:   23.6115 dB
Epoch 2:   25.5132 dB
Epoch 5:   26.5005 dB
Epoch 10:  26.8776 dB
Epoch 14:  27.0012 dB
Epoch 18:  27.0595 dB
Epoch 20:  27.0870 dB
```

Most of the improvement occurred during the first 10 epochs.

Later epochs continued to produce smaller improvements, indicating that training is approaching a plateau under the current fixed learning rate.

There is currently no strong evidence of severe overfitting.

Training L1:

```text
0.088815 → 0.033444
```

Validation L1:

```text
0.048694 → 0.035668
```

Validation PSNR reached its highest value at the final epoch.

---

# Training Performance

Epoch 1 required approximately:

```text
64.5 seconds
```

Subsequent epochs generally required approximately:

```text
7–8 seconds per epoch
```

The longer first epoch is consistent with CUDA initialization and warm-up overhead.

The RTX 4060 provides sufficient performance for rapid controlled experimentation with the current lightweight architecture.

---

# Checkpointing

Training maintains:

```text
checkpoint_latest.pt
checkpoint_best.pt
```

`checkpoint_latest.pt` is updated after every epoch.

`checkpoint_best.pt` is updated only when validation PSNR improves.

Checkpoints include:

```text
model_state_dict
optimizer_state_dict
epoch
best_val_psnr
model_config
training_config
```

Resume support is implemented through:

```text
--resume
```

Example:

```powershell
python train.py --data-dir data/Data-public --epochs 20 --batch-size 16 --lr 1e-4 --checkpoint-dir checkpoints --resume checkpoints/checkpoint_latest.pt
```

Checkpoint files remain excluded from Git.

---

# Experiment 1 Conclusion

Experiment 1 successfully established the first neural restoration baseline.

The model achieved:

```text
PSNR: 27.0870 dB
SSIM: 0.725385
```

compared with the bicubic baseline:

```text
PSNR: 23.1413 dB
SSIM: 0.550604
```

Therefore:

```text
PSNR improvement: +3.9457 dB
SSIM improvement: +0.174781
```

The complete neural training, validation, metric, checkpoint, and resume pipeline is now considered verified.

Experiment 1 should be treated as the canonical neural baseline for all subsequent experiments.

---

# Next Experiment

## Experiment 2 — Optimization / Learning-Rate Schedule

Objective:

Determine whether the apparent ~27 dB plateau is caused by the current optimization schedule before increasing model capacity.

The architecture should remain unchanged.

Proposed controlled changes:

```text
Model:             unchanged
Residual blocks:   4
Features:          32
Loss:              L1Loss
Optimizer:         Adam
Batch size:        16

Initial LR:        1e-4
Epochs:            40
LR scheduler:      ReduceLROnPlateau
  Mode:            max (monitors validation PSNR)
  Factor:          0.5
  Patience:        3 epochs
  Min LR:          1e-6

Dataset split:     unchanged
Seed:              42
Preprocessing:     unchanged
```

Only optimization behaviour should be changed so that the effect can be measured independently. No architecture, loss, optimizer type, batch size, dataset, split, or preprocessing changes.

Experiment 2 must be compared directly against:

```text
Experiment 1:
PSNR = 27.0870 dB
SSIM = 0.725385
```

Do not overwrite or reinterpret Experiment 1 when recording later experiments.

## Experiment 2 Infrastructure (implemented; training not yet run)

Scheduler support was added to `train.py` behind an explicit CLI flag so Experiment 1
remains reproducible with no code-path change:

```text
--scheduler none      (default; identical fixed-LR behavior to Experiment 1)
--scheduler plateau    (ReduceLROnPlateau, mode="max" on validation PSNR)
--scheduler-factor     (default 0.5)
--scheduler-patience   (default 3)
--min-lr               (default 1e-6)
```

Epoch flow with a scheduler enabled:

```text
train_one_epoch
-> validate (computes Val PSNR)
-> print epoch summary (including the LR used for this epoch)
-> scheduler.step(val_psnr)          [prints "Learning rate reduced: a -> b" if it drops]
-> update best_val_psnr
-> save checkpoint_latest.pt (includes scheduler_state_dict + scheduler_config)
-> save checkpoint_best.pt if improved
```

Checkpoints now carry `scheduler_state_dict` and `scheduler_config` in addition to the
existing `model_state_dict`/`optimizer_state_dict`/`epoch`/`best_val_psnr`/`model_config`/
`training_config`. Both fields are `None` when `--scheduler none` is used. Older
checkpoints saved before this change (including Experiment 1's) simply lack these two
keys; resuming from them is still supported and prints an explicit warning rather than
silently skipping scheduler restoration.

### Checkpoint directory convention

To guarantee Experiment 2 cannot overwrite Experiment 1's reference checkpoint:

```text
checkpoints/checkpoint_best.pt          Experiment 1's original files (untouched)
checkpoints/checkpoint_latest.pt
checkpoints/exp1_baseline/              Byte-identical protective copy of the above,
                                         made before any Experiment 2 code/training
checkpoints/exp2_plateau/               Experiment 2's own checkpoint_latest.pt /
                                         checkpoint_best.pt (to be created by the real run)
```

`--checkpoint-dir` must be passed explicitly as `checkpoints/exp2_plateau` when starting
Experiment 2 -- the CLI default is still bare `checkpoints/` (unchanged, to avoid an
unrelated behavior change), so omitting the flag would write into Experiment 1's files.

### Smoke verification (tiny subset, not a real result)

Verified via `--max-train-samples 24 --max-val-samples 12` against the real dataset with
`--scheduler plateau --scheduler-patience 1`: training/validation/checkpointing all
succeed, a genuine LR reduction was observed and logged, and resuming from a mid-run
checkpoint correctly restored the scheduler's internal history and the reduced learning
rate (continuing at the correct next epoch rather than restarting). See the implementation
report for exact commands/output; smoke checkpoints were deleted afterward and are not
part of any experiment record.

---

# Experiment 2B — Same Model, 40 Epochs, Fixed LR (no scheduler)

## Status

**Completed**

## Objective

Isolate how much of any Experiment-2-vs-Experiment-1 gain comes from simply training
longer (20 to 40 epochs) versus from the LR scheduler itself. Same architecture, same
fixed `lr=1e-4`, no scheduler -- only the epoch count changed relative to Experiment 1.

## Training Configuration

```text
Architecture:      unchanged (32 features, 4 residual blocks, 84,708 parameters)
Loss:              L1Loss
Optimizer:         Adam
Learning rate:     1e-4 (fixed, no scheduler)
Batch size:        16
Epochs:            40
Split seed:        42
```

## Best Result

```text
Best PSNR: 27.2704 dB
SSIM at best-PSNR epoch: 0.731226
```

Checkpoint: `checkpoints/exp2_fixed40/checkpoint_best.pt`

---

# Experiment 2 — Same Model, 40 Epochs, ReduceLROnPlateau

## Status

**Completed**

## Training Configuration

```text
Architecture:      unchanged (32 features, 4 residual blocks, 84,708 parameters)
Loss:              L1Loss
Optimizer:         Adam
Initial LR:        1e-4
Scheduler:         ReduceLROnPlateau
  Mode:            max
  Factor:          0.5
  Patience:        3
  Min LR:          1e-6
Batch size:        16
Epochs:            40
Split seed:        42
```

## Best Result

```text
Best PSNR: 27.2959 dB
SSIM: 0.734007
```

Checkpoint: `checkpoints/exp2_plateau/checkpoint_best.pt`

## Controlled Comparison -- Experiment 1 vs 2B vs 2

| Experiment | Epochs | Scheduler | Best PSNR      | SSIM at best PSNR |
| ---------- | -----: | --------- | --------------: | -----------------: |
| Exp 1      |     20 | none      |      27.0870 dB |            0.725385 |
| Exp 2B     |     40 | none      |      27.2704 dB |            0.731226 |
| Exp 2      |     40 | plateau   |  **27.2959 dB** |        **0.734007** |

```text
20 -> 40 epochs, fixed LR:         +0.1834 dB  (Exp 1 -> Exp 2B)
40 epochs, fixed LR -> scheduled:  +0.0255 dB  (Exp 2B -> Exp 2)
```

## Conclusion

Most of the gain from Experiment 1 to Experiment 2 came from longer training, not from
the learning-rate schedule. The scheduler contributed a small additional improvement on
top of that (+0.0255 dB at the best epoch) and will be retained as the default training
recipe for subsequent experiments, since it is at worst neutral and did provide a
measured (if modest) gain. This result argues against attributing the ~27 dB plateau
primarily to the fixed learning rate -- the next controlled variable to test is model
capacity (Experiment 3).

---

# Experiment 3 — Increased Model Capacity

## Status

**Completed and independently verified.**

## Objective

Test whether additional model capacity improves the current neural baseline, using
Experiment 2 (40 epochs, ReduceLROnPlateau) as the training-recipe reference.

## Changes From Experiment 2

```text
Feature channels:  32 -> 64
Residual blocks:    4 -> 8
```

Everything else (loss, optimizer, initial LR, scheduler config, batch size, epochs,
dataset/split/seed, crop sizes, augmentation, validation preprocessing, metric
implementation and clipping convention, checkpointing/resume behavior) is unchanged from
Experiment 2.

## Architecture

Same `ResidualSRNet` structure as Experiments 1/2 (no new architecture family):

```text
Noisy LR
-> 3x3 conv (1 -> 64 channels)
-> 8 Residual Blocks (conv3x3 -> ReLU -> conv3x3 -> identity skip)
-> 3x3 conv
-> global residual feature skip
-> upsampling 3x3 conv (64 -> 4 channels)
-> PixelShuffle x2
-> restored grayscale output
```

## Parameter Count (verified by direct instantiation, not estimated)

```text
Experiment 2 parameters: 84,708
Experiment 3 parameters: 630,724
Capacity multiplier:     7.446x
```

## CUDA Sanity Check

Synthetic batch `[16, 1, 64, 64]` -> output `[16, 1, 128, 128]` on the RTX 4060 Laptop
GPU: forward, L1 backward, and one Adam optimizer step all succeeded. Peak allocated GPU
memory ~348 MiB / peak reserved ~448 MiB, against ~8188 MiB total (~5.5% utilization at
batch size 16) -- batch size 16 fits comfortably and was left unchanged.

## Checkpoint Directory

```text
checkpoints/exp3_capacity/checkpoint_latest.pt
checkpoints/exp3_capacity/checkpoint_best.pt
```

Separate from `checkpoints/exp1_baseline/`, `checkpoints/exp2_plateau/`, and
`checkpoints/exp2_fixed40/`, all of which remain untouched.

## Result

```text
Best epoch by PSNR: 38

Val L1:   0.033708
Val PSNR: 27.6212 dB
Val SSIM: 0.743619
```

Checkpoint: `checkpoints/exp3_capacity/checkpoint_best.pt`

## Comparison vs Experiment 2

| Metric | Exp 2 (32/4) | Exp 3 (64/8) | Improvement |
| ------ | -----------: | -----------: | ----------: |
| PSNR   |   27.2959 dB |   27.6212 dB | +0.3253 dB  |
| SSIM   |     0.734007 |     0.743619 | +0.009612   |

## Conclusion

Increasing model capacity (7.45x more parameters) produced a measurable improvement over
Experiment 2, larger than the scheduler's own contribution (+0.0255 dB) but smaller than
the gain from longer training alone (+0.1834 dB). Experiment 3 is now the primary
reference configuration for subsequent experiments. Experiment 3 must remain untouched
and reproducible; its checkpoints under `checkpoints/exp3_capacity/` are not modified by
later experiments.

---

# Experiment 4 — Charbonnier Loss

## Status

**Completed and independently verified.**

## Objective

Determine whether Charbonnier reconstruction loss improves validation restoration
quality compared with L1, using Experiment 3 (64 features, 8 residual blocks, 40 epochs,
ReduceLROnPlateau) as both the architecture and the training-recipe reference.

## Change From Experiment 3

```text
Loss: L1Loss -> Charbonnier loss
      mean(sqrt((prediction - target)^2 + epsilon^2))
Charbonnier epsilon: 1e-3
```

Everything else -- architecture (64 features, 8 residual blocks, 630,724 parameters),
optimizer, initial LR, scheduler config, batch size, epochs, seed, dataset/split, crop
sizes, augmentation, validation preprocessing, PSNR/SSIM implementation and clipping
convention, checkpointing/resume semantics -- is unchanged from Experiment 3.

## Implementation

`src/losses.py` adds `CharbonnierLoss` (`nn.Module`) plus `build_loss_config`/
`build_loss`/`loss_label` helpers, mirroring the existing scheduler helper pattern.
`train.py` gains `--loss {l1,charbonnier}` (default `l1`, so Experiments 1-3 remain
exactly reproducible with no flag) and `--charbonnier-eps` (default `1e-3`). Checkpoints
now also store `loss_config`; resuming with a mismatched `--loss`/`--charbonnier-eps`
against a checkpoint's stored config raises a clear error rather than silently switching
objectives, mirroring the existing `model_config` resume-safety check. Checkpoints saved
before this change (Experiments 1-3) have no `loss_config` key and are treated as L1,
consistent with what they actually used. Training/validation logging now prints
`Train L1`/`Val L1` or `Train Charbonnier`/`Val Charbonnier` depending on the selected
loss, instead of always labeling the value "L1". `evaluate_checkpoint.py` always reports
an actual-L1 `Val L1` diagnostic regardless of training loss (so that number stays
comparable across every experiment) and separately prints which loss the checkpoint was
trained with.

## Checkpoint Directory

```text
checkpoints/exp4_charbonnier/checkpoint_latest.pt
checkpoints/exp4_charbonnier/checkpoint_best.pt
```

Separate from `checkpoints/exp1_baseline/`, `checkpoints/exp2_plateau/`,
`checkpoints/exp2_fixed40/`, and `checkpoints/exp3_capacity/`, all of which remain
untouched.

## Success Criterion

Compared directly against Experiment 3 (PSNR = 27.6212 dB, SSIM = 0.743619) using
validation PSNR as the checkpoint-selection metric, exactly as in every prior experiment:

```text
Exp 4 PSNR > 27.6212 dB: possible improvement from Charbonnier
Exp 4 PSNR approximately equal: no meaningful advantage
Exp 4 PSNR lower: L1 remains preferable under this setup
```

Loss magnitude (L1 vs Charbonnier values are not on the same numeric scale near zero
error) is explicitly not the comparison criterion.

## Result

```text
Best epoch by PSNR: 38

Val L1 diagnostic: 0.033817
Val PSNR: 27.5881 dB
Val SSIM: 0.743230
```

Checkpoint: `checkpoints/exp4_charbonnier/checkpoint_best.pt`

## Comparison vs Experiment 3

| Metric | Exp 3 (L1) | Exp 4 (Charbonnier) | Change     |
| ------ | ---------: | -------------------: | ---------: |
| PSNR   | 27.6212 dB |            27.5881 dB| -0.0331 dB |
| SSIM   |   0.743619 |              0.743230 | -0.000389  |

## Conclusion

Charbonnier did not outperform L1 -- both PSNR and SSIM are very slightly lower than
Experiment 3, essentially a wash within run-to-run noise. **Experiment 3 (L1) remains
the canonical best configuration.** This is consistent with Charbonnier's numerical
similarity to L1 away from zero error (see the synthetic sanity check in the Experiment 4
implementation report): for this model/dataset, smoothing the loss near zero error did
not translate into a measurable validation improvement.

---

# Experiment 5 — L1 + SSIM Composite Loss

## Status

**Completed and independently verified.**

## Objective

Test whether explicit structural-similarity optimization improves the current best
64-feature / 8-block model (Experiment 3), by adding a differentiable SSIM term to the
reconstruction loss.

## Change From Experiment 3

```text
Loss: L1Loss -> L1 + lambda * (1 - SSIM)
      lambda (ssim_weight) = 0.1
```

Everything else -- architecture (64 features, 8 residual blocks, 630,724 parameters),
optimizer, initial LR, scheduler config, batch size, epochs, seed, dataset/split, crop
sizes, augmentation, validation preprocessing, PSNR/SSIM implementation and clipping
convention, checkpointing/resume semantics -- is unchanged from Experiment 3.

## Differentiable SSIM Implementation

`src/losses.py` adds a from-scratch, pure-PyTorch local-window (Gaussian) SSIM
(`differentiable_ssim`, Wang et al. 2004 formulation) built entirely from `conv2d` +
elementwise tensor ops -- no NumPy conversion, so gradients flow through it. This is
deliberately separate from the established `src.metrics.ssim` (which converts to NumPy
via scikit-image for evaluation and is not differentiable): training uses the new
differentiable version inside the loss, while validation/evaluation continue to use the
existing scikit-image-based metric unchanged, so Experiment 5's PSNR/SSIM stay directly
comparable to Experiments 1-4. The two implementations can produce slightly different
numeric SSIM values (different window/constant conventions between scikit-image and this
Gaussian-window formulation) -- expected and not a bug, since only the evaluation metric
is used for cross-experiment comparison.

Window size 11, Gaussian sigma 1.5, `C1 = (0.01*L)^2`, `C2 = (0.03*L)^2` with data range
`L = 1.0` -- the standard SSIM constants. `SSIMLoss` wraps it as `1 - differentiable_ssim`;
`L1SSIMLoss` computes `L1 + ssim_weight * SSIMLoss`.

`build_loss_config`/`build_loss`/`loss_label` (already used for `l1`/`charbonnier`) are
extended with `"l1_ssim"` rather than adding parallel logic. `train.py` gains
`--loss l1_ssim` and `--ssim-weight` (default `0.1`); logging prints `Train L1+SSIM`/
`Val L1+SSIM` for this loss. Checkpoints store
`loss_config: {"name": "l1_ssim", "ssim_weight": 0.1}`; resume enforces an exact match
against the checkpoint's stored loss config (mirroring the Experiment 4 safety check),
so an L1, Charbonnier, or L1+SSIM checkpoint can never be silently resumed under a
different objective. `evaluate_checkpoint.py` requires no changes -- it already reports
the checkpoint's training loss generically via `loss_label()` and always scores an
actual-L1 diagnostic regardless of training loss.

## Checkpoint Directory

```text
checkpoints/exp5_l1_ssim/checkpoint_latest.pt
checkpoints/exp5_l1_ssim/checkpoint_best.pt
```

Separate from `checkpoints/exp1_baseline/`, `checkpoints/exp2_plateau/`,
`checkpoints/exp2_fixed40/`, `checkpoints/exp3_capacity/`, and
`checkpoints/exp4_charbonnier/`, all of which remain untouched.

## Success Criterion

Compared directly against Experiment 3 (PSNR = 27.6212 dB, SSIM = 0.743619), interpreted
using both metrics, with validation PSNR remaining the checkpoint-selection metric:

```text
Higher PSNR and higher SSIM: clear improvement
Similar PSNR but noticeably higher SSIM: potential useful tradeoff
Higher SSIM but meaningfully worse PSNR: do not automatically replace Experiment 3
Lower on both: reject composite loss
```

## Result

```text
Best checkpoint epoch: 39

Val L1 diagnostic: 0.034065
Val PSNR: 27.5282 dB
Val SSIM: 0.747377
```

Checkpoint: `checkpoints/exp5_l1_ssim/checkpoint_best.pt`

## Comparison vs Experiment 3

| Metric | Exp 3 (L1) | Exp 5 (L1+SSIM) | Change     |
| ------ | ---------: | ---------------: | ---------: |
| PSNR   | 27.6212 dB |        27.5282 dB| -0.0930 dB |
| SSIM   |   0.743619 |          0.747377 | +0.003758  |

## Conclusion

Experiment 5 improved structural similarity (SSIM +0.003758) but reduced PSNR
(-0.0930 dB) relative to Experiment 3. **Experiment 3 remains the canonical best-PSNR
model; Experiment 5 is the current best-SSIM model.** Since validation PSNR is the
project's checkpoint-selection metric and Experiment 6 changes only crop size (not
loss), Experiment 6 is compared primarily against Experiment 3, with Experiment 5 kept
as the best-SSIM reference.

---

# Experiment 6 — Larger Training Crop

## Status

**Planned / infrastructure verified.** The real 40-epoch training run has not been
executed.

## Objective

Determine whether a larger training crop -- more spatial context per training sample --
improves validation restoration quality for the current best 64-feature / 8-block L1
model (Experiment 3).

## Change From Experiment 3

```text
Training crop: 64x64 LR / 128x128 GT -> 96x96 LR / 192x192 GT
Loss: L1 (unchanged -- Experiment 6 returns to Experiment 3's loss, not Experiment 5's)
```

Everything else -- architecture (64 features, 8 residual blocks, 630,724 parameters),
optimizer, initial LR, scheduler config, batch size, epochs, seed, dataset/split,
augmentation, validation preprocessing, PSNR/SSIM implementation and clipping
convention, checkpointing/resume semantics -- is unchanged from Experiment 3.

## Crop Parameterization

Crop size was **already fully parameterized** end-to-end before this experiment --
no rewrite of the transform implementation was needed:

- `train.py` already exposed `--crop-size` (default `64`, unchanged).
- `src/transforms.py::aligned_paired_crop`/`PairedRandomCrop`/`create_training_transform`
  already accept an arbitrary LR crop size; the GT crop is always derived automatically
  as `crop_size * scale` (never a separate argument), so `--crop-size 96` with the
  existing `scale=2` produces an exact 192x192 GT crop with no new code path.
- `training_config["crop_size"]` was already stored in checkpoints.

Two real gaps were closed for this experiment: (1) the startup log did not print the
configured training crop at all -- `train.py` now prints
`Training crop: LR = 96x96 GT = 192x192 (validation always uses full images)`
(or `64x64`/`128x128` by default) every run; (2) resuming with a different `--crop-size`
than a checkpoint's stored `training_config` was not checked at all -- a new
`warn_on_resume_config_mismatch()` helper (replacing the old inline seed/val-fraction
check, extended rather than duplicated) now also warns clearly, e.g.
`--crop-size (64) differs from the checkpoint's stored training crop_size (96)`,
without blocking the resume, consistent with how seed/val_fraction mismatches are
already handled (a warning, not a hard rejection like model_config/loss_config, since
crop size doesn't break tensor-shape compatibility).

## GPU Memory Sanity Check

Synthetic batch `[16, 1, 96, 96]` -> output `[16, 1, 192, 192]` on the RTX 4060 Laptop
GPU: forward, L1 backward, and one Adam optimizer step all succeeded, all outputs/
gradients finite. Peak allocated ~774 MiB / peak reserved ~876 MiB, against ~8188 MiB
total (~10.7% utilization) -- batch size 16 fits comfortably and was **not** reduced.

## Checkpoint Directory

```text
checkpoints/exp6_crop96/checkpoint_latest.pt
checkpoints/exp6_crop96/checkpoint_best.pt
```

Separate from `checkpoints/exp1_baseline/`, `checkpoints/exp2_plateau/`,
`checkpoints/exp2_fixed40/`, `checkpoints/exp3_capacity/`, `checkpoints/exp4_charbonnier/`,
and `checkpoints/exp5_l1_ssim/`, all of which remain untouched.

## Benchmark

Compared primarily against Experiment 3 (L1, same architecture, same everything except
crop size):

```text
Experiment 3: PSNR = 27.6212 dB, SSIM = 0.743619
```

Experiment 5 remains the best-SSIM reference (PSNR = 27.5282 dB, SSIM = 0.747377).

```text
Higher PSNR and SSIM: clear improvement
Higher PSNR, similar SSIM: useful improvement
Similar PSNR: larger crop likely not worth the additional training cost
Lower PSNR: retain 64x64 crop
```

## Result

TBD -- the real 40-epoch Experiment 6 run has not been started.

---

# Experiment History

| Experiment               | Main Change                           |      Best PSNR |    Best SSIM | Status   |
| ------------------------ | -------------------------------------- | -------------: | -----------: | -------- |
| Bicubic                  | Classical interpolation               |     23.1413 dB |     0.550604 | Complete |
| Exp 1 — Residual CNN     | First neural baseline (20 epochs)     |     27.0870 dB |      0.725385 | Complete |
| Exp 2B — Longer training | 40 epochs, fixed LR                   |     27.2704 dB |      0.731226 | Complete |
| Exp 2 — Optimization     | 40 epochs, ReduceLROnPlateau          |     27.2959 dB |      0.734007 | Complete |
| Exp 3 — Capacity         | 64 features / 8 blocks (7.45x params) | **27.6212 dB** |      0.743619 | Complete |
| Exp 4 — Charbonnier loss | L1 -> Charbonnier (eps=1e-3)          |     27.5881 dB |      0.743230 | Complete |
| Exp 5 — L1+SSIM loss     | L1 -> L1 + 0.1*(1-SSIM)               |     27.5282 dB |  **0.747377** | Complete |
| Exp 6 — Larger crop      | 64x64 -> 96x96 LR crop                |             TBD |           TBD | Planned  |

---

## Experiment Logging Rule

For every future experiment record:

1. Experiment objective
2. Exact change from previous experiment
3. Architecture
4. Parameter count
5. Loss
6. Optimizer
7. Learning-rate configuration
8. Batch size
9. Epoch count
10. Data augmentation
11. Best epoch
12. Validation L1
13. Validation PSNR
14. Validation SSIM
15. Improvement/regression against Experiment 1
16. Checkpoint used
17. Observations
18. Conclusion
19. Next experiment decision

Whenever possible, change only one major experimental variable at a time.
