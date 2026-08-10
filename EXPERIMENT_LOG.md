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

**Completed and independently verified.**

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

```text
Best epoch by PSNR: 38

Val L1: 0.033420
Val PSNR: 27.7090 dB
Val SSIM: 0.745634
```

Checkpoint: `checkpoints/exp6_crop96/checkpoint_best.pt`

Independently verified using `evaluate_checkpoint.py`:

```text
Val L1 = 0.033420
Val PSNR = 27.7090 dB
Val SSIM = 0.745634
Bicubic PSNR = 23.1413 dB
Bicubic SSIM = 0.550604
PSNR improvement over bicubic = +4.5677 dB
SSIM improvement over bicubic = +0.195030
```

## Comparison vs Experiment 3 and Experiment 5

| Metric | Exp 3 (64x64 crop) | Exp 5 (L1+SSIM) | Exp 6 (96x96 crop) | Exp 6 vs Exp 3 |
| ------ | ------------------: | ----------------: | -------------------: | --------------: |
| PSNR   |          27.6212 dB |         27.5282 dB |        **27.7090 dB** |      +0.0878 dB |
| SSIM   |            0.743619 |         **0.747377** |              0.745634 |        +0.002015 |

## Conclusion

Experiment 6 improved **both** PSNR and SSIM relative to Experiment 3 -- a clear
improvement per the benchmark criteria (higher PSNR and SSIM). **Experiment 6 is now
the best-PSNR model overall**, surpassing Experiment 3. Experiment 5 retains a very
slightly higher SSIM (0.747377 vs 0.745634, a difference of 0.001743) but at a lower
PSNR than Experiment 6, so it remains the best-SSIM reference only, not the overall
best model. The larger 96x96 LR training crop (more spatial context per sample) was a
worthwhile change and did not require any batch-size reduction (fit comfortably in
GPU memory; see the GPU Memory Sanity Check above).

---

# Experiment 7 — Full-Image Training Crop

## Status

**Completed and independently verified.**

## Objective

Test whether training on the full 128x128 LR image (256x256 GT) -- the maximum possible
spatial context, no random crop at all -- improves validation restoration quality beyond
Experiment 6's 96x96 crop, using Experiment 6's exact training recipe otherwise.

## Configuration

```text
Architecture:      ResidualSRNet, unchanged (64 features, 8 residual blocks, 630,724 parameters)
PixelShuffle:       x2
Scale:              2

Loss:               L1Loss
Optimizer:          Adam
Initial LR:         1e-4
Scheduler:          ReduceLROnPlateau
  Mode:             max
  Factor:           0.5
  Patience:         3
  Min LR:           1e-6

Batch size:         16
Epochs:             40
Seed:               42

Training crop:      LR 128x128 / GT 256x256 (full image -- no random crop offset possible)
Validation:         full images, unchanged (128x128 LR / 256x256 GT, no augmentation)

Train samples:      2560
Validation samples: 640
```

Change from Experiment 6: training crop only (96x96 -> 128x128, i.e. the full LR image).
Everything else -- architecture, loss, optimizer, scheduler, batch size, epoch count,
seed, dataset/split, validation preprocessing, metric implementation and clipping
convention, checkpointing/resume semantics -- unchanged.

## Result

```text
Best epoch by PSNR: 38

Val L1: 0.033430
Val PSNR: 27.7101 dB
Val SSIM: 0.743748
```

Checkpoint: `checkpoints/exp7_crop128/checkpoint_best.pt`

Independently verified using `evaluate_checkpoint.py`:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/exp7_crop128/checkpoint_best.pt --data-dir data/Data-public
```

```text
Using device: cuda
Loaded checkpoint checkpoints\exp7_crop128\checkpoint_best.pt (epoch=38, best_val_psnr=27.71012558457606)
Training loss: L1 ({'name': 'l1'})
Validation samples: 640
Val L1 (diagnostic, always L1 regardless of training loss): 0.033430
Val PSNR: 27.7101 dB
Val SSIM: 0.743748
Bicubic PSNR: 23.1413 dB
Bicubic SSIM: 0.550604
PSNR vs bicubic: +4.5688 dB
SSIM vs bicubic: +0.193144
```

Independently reproduced exactly (epoch 38, Val L1 0.033430, Val PSNR 27.7101 dB,
Val SSIM 0.743748) -- confirms checkpoint loading, model reconstruction, and the
validation pipeline all remain correct.

## Comparison vs Experiment 6

| Metric      | Exp 6 (96x96 crop) | Exp 7 (128x128 crop) | Exp 7 vs Exp 6 |
| ----------- | -------------------: | ----------------------: | --------------: |
| Val L1      |              0.033420 |                 0.033430 |     +0.000010 (negligible) |
| Val PSNR    |            27.7090 dB |               27.7101 dB |         +0.0011 dB |
| Val SSIM    |              0.745634 |                 0.743748 |          -0.001886 |
| Epoch time  |          ~25-27 s     |             ~38-42 s     |    substantially slower |

## Conclusion

Experiment 7 produced **no meaningful improvement** over Experiment 6. The PSNR
difference (+0.0011 dB) is negligible -- effectively tied -- while SSIM is slightly
*worse* (-0.001886) and Val L1 is essentially unchanged. At the same time, training on
the full 128x128 image was substantially slower per epoch (~38-42s vs ~25-27s) for no
corresponding quality gain, since a full-image batch does strictly more compute per
step than a 96x96 crop at the same batch size. **96x96 remains the preferred training
crop size for future experiments** -- Experiment 6 stays the practical best-PSNR
configuration; Experiment 7 is retained as a completed, neutral (non-improving) result
that closes off "more crop context" as a further avenue worth pursuing at this
architecture/data scale.

## Checkpoint Directory

```text
checkpoints/exp7_crop128/checkpoint_latest.pt
checkpoints/exp7_crop128/checkpoint_best.pt
```

Separate from `checkpoints/exp1_baseline/`, `checkpoints/exp2_plateau/`,
`checkpoints/exp2_fixed40/`, `checkpoints/exp3_capacity/`, `checkpoints/exp4_charbonnier/`,
`checkpoints/exp5_l1_ssim/`, and `checkpoints/exp6_crop96/`, all of which remain
untouched. Verified present and byte-identical (SHA-256) before and after this task's
independent evaluation.

---

# Experiment 8 — MSE Loss

## Status

**COMPLETED — STOPPED AFTER SCREENING.** Run as a 15-epoch screening budget (not the
full 40-epoch schedule) to cheaply test the MSE hypothesis before committing full
training time. Screening results were decisively worse than Experiment 6 at a
comparable or later point in training, so the run was deliberately stopped at 15
epochs rather than continued to 40. `checkpoints/exp8_mse/` is retained for
reproducibility.

## Objective

Test whether directly optimizing mean squared error (rather than L1) improves
validation PSNR. PSNR is a direct (monotonic, log-scaled) function of MSE, so directly
minimizing MSE during training may align the training objective more closely with the
checkpoint-selection metric than L1 does.

## Configuration (identical to Experiment 6 except loss and epoch budget)

```text
Architecture:      ResidualSRNet, unchanged (64 features, 8 residual blocks, 630,724 parameters)
PixelShuffle:       x2
Scale:              2

Loss:               L1Loss -> MSELoss (torch.nn.MSELoss)
Optimizer:          Adam
Initial LR:         1e-4
Scheduler:          ReduceLROnPlateau
  Mode:             max
  Factor:           0.5
  Patience:         3
  Min LR:           1e-6

Batch size:         16
Epochs:             15 (screening budget; full schedule would have been 40)
Seed:               42

Training crop:      LR 96x96 / GT 192x192 (Experiment 6's crop, not Experiment 7's)
Validation:         full images, unchanged (128x128 LR / 256x256 GT, no augmentation)

Train samples:      2560
Validation samples: 640
```

Change from Experiment 6: reconstruction loss only (L1 -> MSE), plus a deliberately
shortened screening budget (15 epochs instead of 40) to test the hypothesis cheaply.
Everything else -- architecture, optimizer, scheduler, batch size, seed, crop size,
dataset/split, validation preprocessing, metric implementation and clipping convention,
checkpointing/resume semantics -- unchanged.

## Screening Results

|  Epoch | Val PSNR (dB) |     Val SSIM |             |
| -----: | ------------: | -----------: | ----------- |
|      1 |       26.0065 |     0.659793 |             |
|      5 |       27.0446 |     0.720486 |             |
|      9 |       27.2118 |     0.723958 |             |
| **13** |   **27.2159** | **0.727473** | best PSNR   |
|     15 |       27.1834 |     0.729404 | run stopped |

Best checkpoint: epoch 13, `checkpoints/exp8_mse/checkpoint_best.pt`.

Independently verified using `evaluate_checkpoint.py`:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/exp8_mse/checkpoint_best.pt --data-dir data/Data-public
```

```text
Using device: cuda
Loaded checkpoint checkpoints\exp8_mse\checkpoint_best.pt (epoch=13, best_val_psnr=27.215915462443217)
Training loss: MSE ({'name': 'mse'})
Validation samples: 640
Val L1 (diagnostic, always L1 regardless of training loss): 0.035239
Val PSNR: 27.2159 dB
Val SSIM: 0.727473
Bicubic PSNR: 23.1413 dB
Bicubic SSIM: 0.550604
PSNR vs bicubic: +4.0746 dB
SSIM vs bicubic: +0.176869
```

Independently reproduced exactly (epoch 13, Val PSNR 27.2159 dB, Val SSIM 0.727473,
Val L1 diagnostic 0.035239, +4.0746 dB / +0.176869 over bicubic) -- confirms checkpoint
loading, model reconstruction, and the validation pipeline all remain correct.

## Comparison vs Experiment 6

| Metric   | Exp 6 (L1, epoch 38) | Exp 8 (MSE, epoch 13) | Exp 8 vs Exp 6 |
| -------- | ---------------------: | -----------------------: | --------------: |
| Val PSNR |             27.7090 dB |               27.2159 dB |     -0.4931 dB |
| Val SSIM |               0.745634 |                 0.727473 |       -0.018161 |

MSE also underperforms L1 at comparable epoch counts, not only at its own best epoch --
Experiment 6 had already reached 27.0012 dB by its own epoch 14 and kept improving
steadily through epoch 38, while Experiment 8's PSNR peaked at epoch 13 (27.2159 dB)
and had already begun to plateau/wobble by epoch 15 (27.1834 dB, a slight regression),
suggesting MSE was not simply "behind schedule" but converging to a worse optimum for
this model/data combination.

## Conclusion

MSE substantially underperformed L1 in this controlled comparison: -0.4931 dB PSNR and
-0.018161 SSIM versus Experiment 6, with early signs of plateauing rather than
continued improvement. Continuing to the full 40-epoch budget was judged very unlikely
to close a gap of this size, so the run was **deliberately stopped at 15 epochs**
(a screening budget, not a resource failure or bug) rather than spending the remaining
~25 epochs of compute chasing an already-decisive negative result. **Experiment 8 is
recorded as a completed negative/neutral result; L1 remains the preferred
reconstruction loss.** `checkpoints/exp8_mse/` is kept (not deleted) for
reproducibility. The next research direction is architecture improvement, not another
simple loss substitution -- Experiments 4 (Charbonnier), 5 (L1+SSIM), and 8 (MSE) have
now all been tried against L1 without producing a PSNR improvement over Experiment 6.

## Implementation

`src/losses.py` extends `build_loss_config`/`build_loss`/`loss_label` with `"mse"` (no
parallel logic; same pattern as `l1`/`charbonnier`/`l1_ssim`). `train.py`'s `--loss`
choices become `{l1, mse, charbonnier, l1_ssim}`; default remains `l1`, so all prior
experiments stay exactly reproducible with no flag. MSE checkpoints store
`loss_config: {"name": "mse"}`. Because `load_checkpoint_for_resume`'s loss-config check
already compares the stored and requested `loss_config` dicts generically, MSE
automatically got the same hard-reject resume protection as every other loss with no
extra code -- verified live in both directions (an MSE checkpoint rejects `--loss l1`,
and the real Experiment 6 L1 checkpoint rejects `--loss mse`). `evaluate_checkpoint.py`
required **no changes** -- it already reports the checkpoint's training loss generically
via `loss_label()` and always scores an actual-L1 diagnostic regardless of training loss
(so `Val L1` stays comparable across every experiment, MSE included).

## Verification Performed (infrastructure only, not training results)

- Unit tests: MSE math correctness, identical-inputs -> 0, differentiability/finite
  gradients, `loss_config` format, resume accept/reject in both directions (including
  against a real Experiment 6-shaped legacy checkpoint), regression coverage for
  L1/Charbonnier/L1+SSIM, `evaluate_checkpoint.load_model` reads an MSE checkpoint.
- CUDA sanity: `[16,1,96,96]` -> `[16,1,192,192]` on the RTX 4060 Laptop GPU, forward/
  MSE/backward/optimizer step all succeeded, finite throughout. Peak allocated ~776 MiB
  / peak reserved ~876 MiB of ~8188 MiB total -- consistent with Experiment 6/7's memory
  footprint (loss choice doesn't materially affect memory).
- Real-data smoke run (`checkpoints/exp8_mse_smoke/`, deleted afterward -- **not**
  `checkpoints/exp8_mse/`): 32 train / 16 val samples, 1-2 epochs, `--loss mse
  --crop-size 96 --seed 42`, same scheduler as Experiment 6. Training/validation/
  checkpointing/resume all worked; checkpoint correctly stored
  `loss_config: {"name": "mse"}`; `evaluate_checkpoint.py` correctly read it back and
  printed `Training loss: MSE ({'name': 'mse'})`.

## Checkpoint Directory

```text
checkpoints/exp8_mse/checkpoint_latest.pt   (epoch 15, run end)
checkpoints/exp8_mse/checkpoint_best.pt     (epoch 13, best Val PSNR)
```

Both retained (not deleted) for reproducibility. Separate from all prior experiment
directories, all of which remain untouched. Checkpoint file hashes (SHA-256) were
verified identical before and after independent evaluation in this task.

---

# Experiment 9 — EDSR-lite Architecture

## Status

**COMPLETED and independently verified.**

## Hypothesis

Experiments 4, 5, and 8 all tried substituting the reconstruction loss (Charbonnier,
L1+SSIM, MSE) against Experiment 6's L1 baseline without beating it on PSNR. This
suggests the ~27.7 dB plateau may be a **model-capacity/architecture** limit rather
than a loss-function one. Experiment 9 tests that by introducing a stronger,
EDSR-style architecture (`EDSRLite`) while keeping Experiment 6's entire training
recipe (loss, crop, batch size, seed, optimizer, scheduler) unchanged, so any PSNR
change can be attributed to architecture alone.

## Chosen Architecture: EDSRLite

Implemented as a **new** model (`src/models/edsr_lite.py`) rather than modifying
`ResidualSRNet` in place -- `ResidualSRNet` is completely untouched, so Experiments
1-8's checkpoints remain fully compatible and reproducible.

```text
Noisy LR
  -> 3x3 conv (in_channels -> num_features)
  -> num_blocks x EDSRResidualBlock:
       Conv3x3 -> ReLU -> Conv3x3, added back as: x + residual_scale * residual
       (no BatchNorm anywhere in the model)
  -> 3x3 conv ("conv_after_body")
  -> + long/global residual connection back to the post-conv_in features
  -> 3x3 conv (num_features -> num_features * scale^2) -> PixelShuffle(scale)
  -> 3x3 reconstruction conv (num_features -> out_channels)
  -> restored grayscale output
```

Unlike `ResidualSRNet` (which fuses upsampling and reconstruction into a single
conv+PixelShuffle step), EDSRLite keeps a separate feature-space upsample and a
dedicated image-space reconstruction convolution, matching classical EDSR (Lim et
al., 2017) structure. No attention, no transformers, no GAN components, no
pretrained weights -- channel attention is explicitly deferred to a later,
separate experiment per the task brief.

## Experiment 9 Configuration

```text
Architecture:       edsr_lite
Feature channels:   64
Residual blocks:    16
Residual scale:     0.1 (fixed constant, not trainable)
Scale:              2
Parameters:         1,367,553 (verified by direct instantiation)
Ratio vs ResidualSRNet champion (630,724): 2.1682x
```

Why this configuration: 64 features keeps the same width as the Experiment 6 champion
(isolating "depth + EDSR design" as the tested change rather than also changing
width), while doubling the residual-block count (8 -> 16) meaningfully increases
capacity and receptive field. This lands at ~1.37M parameters -- solidly inside the
task's suggested 1-3M "reasonable" region, roughly 2.2x the current champion, without
approaching a research-scale EDSR (32 blocks x 256 features, ~43M parameters). A
CUDA sanity check (below) confirmed this fits comfortably in the RTX 4060 Laptop
GPU's ~8GB VRAM at the full batch size of 16, and a short runtime measurement
estimated a full 40-epoch run at roughly 25-35 minutes -- practical for this
dataset/hardware, not requiring an architecture search or a larger model.

Training recipe (unchanged from Experiment 6):

```text
Loss:               L1Loss
Optimizer:          Adam
Initial LR:         1e-4
Scheduler:          ReduceLROnPlateau (mode=max, factor=0.5, patience=3, min_lr=1e-6)
Batch size:         16
Seed:               42
Training crop:      LR 96x96 / GT 192x192
Validation:         full images, unchanged
Train samples:      2560
Validation samples: 640
```

## Implementation

`src/models/__init__.py` adds a centralized model factory,
`build_model_config`/`build_model`, mirroring `src/losses.py`'s existing
`build_loss_config`/`build_loss` pattern -- `train.py`, `evaluate_checkpoint.py`,
and `infer_test.py` (via `evaluate_checkpoint.load_model`) now share one
reconstruction path instead of duplicating `ResidualSRNet(**model_config)`.

**Critical backward-compatibility detail:** `build_model_config("residual_sr", ...)`
deliberately **omits** an `"architecture"` key, producing a dict byte-identical to
every historical checkpoint's `model_config` (Experiments 1-8 never had this key).
Only `"edsr_lite"` configs carry `"architecture": "edsr_lite"`. This means the
*existing* `model_config` dict-equality check in `load_checkpoint_for_resume`
correctly rejects architecture mismatches in both directions with **zero new
comparison logic** -- an EDSRLite config can never equal a ResidualSRNet config
(different key sets), and `build_model()` treats a missing `"architecture"` key as
`ResidualSRNet`, exactly matching what every historical checkpoint actually is.
Verified live in both directions (see Verification Performed below).

`train.py` gains `--model {residual_sr,edsr_lite}` (default `residual_sr`, so every
prior experiment's command stays exactly reproducible with no flag) and
`--residual-scale` (default `0.1`, ignored unless `--model edsr_lite`). Training
startup now prints the selected architecture, its exact trainable parameter count,
and its full `model_config`.

## Verification Performed (infrastructure only, not training results)

- Unit tests: grayscale input, exact 2x output shape across multiple spatial sizes,
  exact parameter count for the Experiment 9 configuration, no BatchNorm anywhere,
  residual-block structure and exact formula match (`x + residual_scale * residual`,
  including a `residual_scale=0` identity edge case), finite forward output, finite
  backward gradients, model-factory reconstruction of both architectures, legacy
  (no-`"architecture"`-key) configs still load as `ResidualSRNet`, matching-config
  EDSRLite resume succeeds, cross-architecture resume rejected in both directions
  (verified against a real Experiment-6-shaped legacy checkpoint), and
  `evaluate_checkpoint.load_model` (shared by `infer_test.py`) reconstructs an
  EDSRLite checkpoint correctly.
- CUDA sanity: `[16,1,96,96]` -> `[16,1,192,192]` on the RTX 4060 Laptop GPU with the
  exact Experiment 9 config, L1 loss, forward/backward/optimizer step all succeeded,
  finite throughout, no OOM. Peak allocated ~1679 MiB / peak reserved ~2190 MiB of
  ~8188 MiB total (~26.75% utilization) -- comfortably fits at the full batch size of
  16 (not reduced).
- Runtime sanity: ~229 ms/batch measured over 20 iterations after warmup ->
  estimated ~37s/epoch (train-only, 160 batches) -> roughly 25-35 minutes for a full
  40-epoch run including validation -- practical for this experiment.
- Real-data smoke run (`checkpoints/exp9_edsr_smoke/`, deleted afterward -- **not**
  `checkpoints/exp9_edsr_lite/`): 32 train / 16 val samples, `--model edsr_lite
  --num-features 64 --num-blocks 16 --residual-scale 0.1 --loss l1 --crop-size 96
  --seed 42`, same scheduler as Experiment 6. Training/validation/checkpointing/
  resume all worked; checkpoint correctly stored
  `model_config["architecture"] == "edsr_lite"`; `evaluate_checkpoint.py` and
  `infer_test.py` both correctly reconstructed and ran the checkpoint.

## Checkpoint Directory

```text
checkpoints/exp9_edsr_lite/checkpoint_latest.pt
checkpoints/exp9_edsr_lite/checkpoint_best.pt   (epoch 36, best Val PSNR)
```

Separate from all prior experiment directories, all of which remain untouched.

## Result

```text
Best epoch by PSNR: 36

Val L1: 0.033854
Val PSNR: 27.5658 dB
Val SSIM: 0.742162
```

Independently verified using `evaluate_checkpoint.py`:

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/exp9_edsr_lite/checkpoint_best.pt --data-dir data/Data-public
```

```text
Using device: cuda
Loaded checkpoint checkpoints\exp9_edsr_lite\checkpoint_best.pt (epoch=36, best_val_psnr=27.565757212200424)
Training loss: L1 ({'name': 'l1'})
Validation samples: 640
Val L1 (diagnostic, always L1 regardless of training loss): 0.033854
Val PSNR: 27.5658 dB
Val SSIM: 0.742162
Bicubic PSNR: 23.1413 dB
Bicubic SSIM: 0.550604
PSNR vs bicubic: +4.4245 dB
SSIM vs bicubic: +0.191558
```

Independently reproduced exactly -- confirms checkpoint loading, model-factory
reconstruction of EDSRLite, and the validation pipeline all remain correct.

## Comparison vs Experiment 6

| Metric        | Exp 6 (ResidualSRNet, 630,724 params) | Exp 9 (EDSRLite, 1,367,553 params) | Exp 9 vs Exp 6 |
| ------------- | --------------------------------------: | -------------------------------------: | ---------------: |
| Val L1        |                                 0.033420 |                                0.033854 |         +0.000434 |
| Val PSNR      |                               27.7090 dB |                              27.5658 dB |         -0.1432 dB |
| Val SSIM      |                                 0.745634 |                                0.742162 |          -0.003472 |
| Best epoch    |                                       38 |                                      36 |                 -- |
| Parameters    |                                  630,724 |                               1,367,553 |             2.1682x |
| Epoch time    |                             ~25-27s     |                             ~49-54s     |     ~2x slower    |

## Conclusion

Experiment 9 did **not** improve over Experiment 6 -- PSNR is 0.1432 dB lower and SSIM
is 0.003472 lower, despite using 2.1682x as many parameters and roughly twice the
per-epoch training time. The model trained correctly (it reached a comparable
bicubic-relative improvement, +4.4245 dB, and benefited from scheduler LR reductions
during training), so this is **not an implementation failure** -- the additional
depth/capacity of EDSRLite simply did not translate into better validation
generalization on this ~2,560-sample dataset. **Experiment 6 remains the practical
champion.** `checkpoints/exp9_edsr_lite/` is retained (not deleted) as a completed,
non-improving architecture experiment for reproducibility and future reference (e.g.
as a component in a possible future ensemble). This result argues against simply
scaling up residual depth/width as the next step; combined with Experiments 4/5/8
(loss substitutions) also failing to beat Experiment 6, small isolated
loss/capacity changes to this architecture family appear to have reached a plateau
around ~27.6-27.7 dB.

---

# Experiment 10 — x8 Geometric Self-Ensemble (Test-Time Augmentation)

## Status

**COMPLETED and measured.** No retraining performed -- this experiment only changes
how existing checkpoints are evaluated at inference time.

## Objective

Test whether averaging predictions from the 8 dihedral (D4) transforms of each LR
validation image -- identity, three rotations, and their four flipped
counterparts -- reduces orientation-specific prediction error and improves
validation PSNR/SSIM, with zero additional training. Tested on the canonical
640-image validation set only; the official (GT-less) test set was not used to make
any decision.

## Implementation

New module `src/tta.py`:

```text
d4_transforms()       -> the 8 (flip, rotation_k) pairs: {no-flip, h-flip} x {0,90,180,270}
forward_transform(x, flip, k)  -> flip (if any), then rotate by 90*k degrees
inverse_transform(y, flip, k)  -> un-rotate by -k, then un-flip (flip is self-inverse)
predict_x8(model, inputs)      -> mean over the 8 (transform -> model -> inverse-transform)
                                   raw predictions; never clips before averaging
```

Each of the 8 transforms was verified to be an exact algebraic inverse of itself
applied in reverse order (`inverse_transform(forward_transform(x)) == x` bit-for-bit,
for odd/even/square/rectangular tensors alike), and all 8 are pairwise distinct --
no accidental duplicates. `predict_x8` works for any `[N,C,H,W]` batch (grayscale,
batch size > 1, not assumed square), always in `torch.inference_mode()`, and restores
the model's original `training`/`eval` mode on exit.

**Raw-averaging order preserved as specified:** individual per-transform predictions
are stacked and averaged *before* any clipping; clipping (when it happens at all)
occurs only inside the existing `src.metrics.psnr`/`ssim` (their default
`clip_prediction=True`, unchanged) when scoring the already-averaged result --
verified directly by a unit test using a constant-valued mock model whose raw output
(2.0, outside `[0,1]`) survives `predict_x8` unclipped.

`evaluate_checkpoint.py` gains `--tta {none,x8}` (default `none`) and a new
`validate_x8()` function mirroring `train.validate()`'s exact aggregation but scoring
`predict_x8`'s output; `--tta none` calls the pre-existing `validate()` unchanged, so
default behavior is byte-for-byte identical to before TTA existed. `infer_test.py`
gains the same `--tta {none,x8}` flag on `run_inference()`, with identical
default-preserving behavior; output filenames, ordering, dtype, and directory
structure are all unchanged. No second PSNR/SSIM implementation was introduced --
both paths call the same `src.metrics.psnr`/`ssim`.

## CUDA Sanity Check

`checkpoints/exp6_crop96/checkpoint_best.pt`, batch of 4 validation-sized
`[4,1,128,128]` inputs, on the RTX 4060 Laptop GPU:

```text
Normal:  output [4,1,256,256], finite, ~10.48 ms/call (warmed-up average of 10)
x8 TTA:  output [4,1,256,256], finite, ~83.81 ms/call (warmed-up average of 10)
Compute-only ratio: ~8.00x (expected -- 8 forward passes vs 1)
Peak allocated: ~100.30 MiB / peak reserved: ~114.00 MiB -- no OOM
```

## Full 640-Image Validation Comparison

### Experiment 6 (ResidualSRNet, 64F/8B, 630,724 params)

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/exp6_crop96/checkpoint_best.pt --data-dir data/Data-public --tta none
python evaluate_checkpoint.py --checkpoint checkpoints/exp6_crop96/checkpoint_best.pt --data-dir data/Data-public --tta x8
```

| Metric        |     Normal |         x8 |     Delta |
| ------------- | ---------: | ---------: | --------: |
| Val L1        |   0.033420 |   0.033176 | -0.000244 |
| Val PSNR      | 27.7090 dB | 27.7689 dB | +0.0599 dB |
| Val SSIM      |   0.745634 |   0.747955 | +0.002321 |
| Wall time     |    16.806s |    33.195s |    ~1.98x |

### Experiment 9 (EDSRLite, 64F/16B, 1,367,553 params)

```bash
python evaluate_checkpoint.py --checkpoint checkpoints/exp9_edsr_lite/checkpoint_best.pt --data-dir data/Data-public --tta none
python evaluate_checkpoint.py --checkpoint checkpoints/exp9_edsr_lite/checkpoint_best.pt --data-dir data/Data-public --tta x8
```

| Metric        |     Normal |         x8 |     Delta |
| ------------- | ---------: | ---------: | --------: |
| Val L1        |   0.033854 |   0.033766 | -0.000088 |
| Val PSNR      | 27.5658 dB | 27.5875 dB | +0.0217 dB |
| Val SSIM      |   0.742162 |   0.742980 | +0.000818 |
| Wall time     |    19.929s |    58.406s |    ~2.93x |

(End-to-end script wall-time ratios are lower than the ~8x compute-only ratio above
because a large, roughly-fixed fraction of each run -- dataset discovery, checkpoint
loading, CUDA initialization -- doesn't scale with the number of forward passes.)

## Conclusion

x8 geometric self-ensembling produced a **reproducible improvement in both PSNR and
SSIM, with no regression, on both checkpoints tested** -- exactly the success
criterion this experiment specified. Experiment 6 gained **+0.0599 dB PSNR** and
**+0.002321 SSIM**, comfortably inside the "even +0.03 to +0.10 dB could be useful"
range called out in advance (and well below the often-cited +0.2 to +0.5 dB, as
expected -- that figure was not assumed and was not observed). Experiment 9 gained a
smaller but still positive **+0.0217 dB PSNR** and **+0.000818 SSIM**. Neither
model's L1 diagnostic regressed. The cost is purely computational (~8x inference
compute, ~2-3x wall time on a small 640-image validation run) and requires no
retraining, no architecture change, and no change to checkpoint-selection criteria.

**Decision: x8 TTA is worth retaining as an optional inference-time technique**,
particularly for Experiment 6 where the gain is clearest. It does not change which
checkpoint is the "champion" (that remains determined by normal validation PSNR, per
the project's established checkpoint-selection convention) -- it is an *optional
inference-time post-processing step* available via `--tta x8` on top of whichever
checkpoint is already selected, not a replacement for model selection or further
training.

---

# Official Test-Set Inference Sanity Check (infrastructure, not a new experiment)

After independently verifying Experiment 6, a small inference sanity check was run
using `infer_test.py` against `checkpoints/exp6_crop96/checkpoint_best.pt` on the first
10 official competition test images (`data/Data-public/Test_NoisyLR/NoisyLR/000000.npy`
through `000009.npy`, sorted filename order -- deterministic, not random).

The official test set has **no locally available ground truth**. Test PSNR and SSIM are
therefore **not available and were not computed** -- do not infer them from this note.
This was a visual/numerical sanity check only (finite outputs, correct 128x128 ->
256x256 shapes, bicubic-vs-neural comparison images), not model evaluation or selection.
Results: `results/test_sanity_exp6/` (raw predictions, bicubic baselines, per-image
comparison PNGs, one contact sheet, and `sanity_stats.json`). All 10 predictions were
finite with no NaN/Inf. This does not change Experiment 6's validation-based ranking
above, which remains the only quantitative comparison in this log.

---

# Experiment History

| Experiment               | Main Change                           |      Best PSNR |    Best SSIM | Status   |
| ------------------------ | -------------------------------------- | -------------: | -----------: | -------- |
| Bicubic                  | Classical interpolation               |     23.1413 dB |     0.550604 | Complete |
| Exp 1 — Residual CNN     | First neural baseline (20 epochs)     |     27.0870 dB |      0.725385 | Complete |
| Exp 2B — Longer training | 40 epochs, fixed LR                   |     27.2704 dB |      0.731226 | Complete |
| Exp 2 — Optimization     | 40 epochs, ReduceLROnPlateau          |     27.2959 dB |      0.734007 | Complete |
| Exp 3 — Capacity         | 64 features / 8 blocks (7.45x params) |     27.6212 dB |      0.743619 | Complete |
| Exp 4 — Charbonnier loss | L1 -> Charbonnier (eps=1e-3)          |     27.5881 dB |      0.743230 | Complete |
| Exp 5 — L1+SSIM loss     | L1 -> L1 + 0.1*(1-SSIM)               |     27.5282 dB |  **0.747377** | Complete |
| Exp 6 — Larger crop      | 64x64 -> 96x96 LR crop                |     27.7090 dB |      0.745634 | Complete |
| Exp 7 — Full-image crop  | 96x96 -> 128x128 (full image) LR crop | **27.7101 dB** |      0.743748 | Complete |
| Exp 8 — MSE loss         | L1 -> MSE, stopped after 15-epoch screen |    27.2159 dB |      0.727473 | Complete |
| Exp 9 — EDSR-lite arch   | ResidualSRNet -> EDSRLite (64F/16B, 1.37M params) | 27.5658 dB | 0.742162 | Complete |
| Exp 10 — x8 geometric TTA | Inference-only self-ensemble on Exp 6 checkpoint | **27.7689 dB** | **0.747955** | Complete |

Note: Exp 10 is not a trained model -- it is Experiment 6's checkpoint evaluated with
x8 test-time augmentation (+0.0599 dB / +0.002321 SSIM over Exp 6 alone). It is an
optional inference-time post-processing step, not a new checkpoint-selection
candidate; Experiment 6's checkpoint remains the underlying "champion" model.

Note: Exp 7's PSNR is numerically the highest on record, but the margin over Exp 6
(+0.0011 dB) is negligible, Exp 7's SSIM/L1 are both slightly worse than Exp 6, and
Exp 7 is substantially slower per epoch. **Experiment 6 (96x96 crop, L1 loss) remains
the practical preferred configuration**; Experiment 7 is a completed, neutral result
that does not change that recommendation. Experiment 8 (MSE loss) is a completed
negative result -- stopped at a 15-epoch screening budget after underperforming
Experiment 6 by -0.4931 dB PSNR / -0.018161 SSIM with early signs of plateauing;
**L1 remains the preferred reconstruction loss**. The next research direction is
architecture improvement, not another loss substitution.

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
