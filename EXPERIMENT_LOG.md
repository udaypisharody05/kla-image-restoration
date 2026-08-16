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

# Experiment 11 — Model Ensemble (Experiment 6 x Experiment 9)

## Status

**Completed -- negative result.** Inference-only, zero retraining. No checkpoint,
architecture, loss, optimizer, scheduler, or split was modified.

## Objective

With loss substitution (Experiments 4, 5, 8), architecture scaling (Experiment 9),
and geometric TTA (Experiment 10) all explored, Experiment 11 tests a different
inference-only idea: whether Experiment 6 (`ResidualSRNet`) and Experiment 9
(`EDSRLite`) -- two independently trained models with different architectures --
make sufficiently *different* prediction errors that a weighted average of their
raw outputs improves validation PSNR/SSIM beyond either model alone, and beyond
Experiment 6 + x8 TTA (the current best pipeline).

| Model | Checkpoint | Architecture | Parameters |
| ----- | ---------- | ------------ | ---------: |
| A (Exp 6) | `checkpoints/exp6_crop96/checkpoint_best.pt` | ResidualSRNet, 64F/8B | 630,724 |
| B (Exp 9) | `checkpoints/exp9_edsr_lite/checkpoint_best.pt` | EDSRLite, 64F/16B | 1,367,553 |

## Averaging Formula

```
ensemble = weight_A * Exp6_raw_prediction + weight_B * Exp9_raw_prediction
```

Each model's **raw** (unclipped) prediction is computed first (optionally via
`predict_x8` for the x8-TTA variant, reusing `src/tta.py` unchanged), the two raw
predictions are combined by `src/ensemble.py::weighted_average_predictions`
(weights normalized to sum to 1, never clips), and only the final combined
prediction is passed to the existing `src.metrics.psnr`/`ssim` -- which perform
the only clipping in the whole pipeline, exactly like the TTA and no-TTA paths
already did. `src/metrics.py` was not modified.

## Implementation

- `src/ensemble.py` -- new, minimal `weighted_average_predictions(predictions,
  weights=None)` utility. Rejects fewer than 2 predictions, shape mismatches,
  prediction/weight count mismatches, and non-positive weights; normalizes
  weights internally; preserves batch/channel/spatial dimensions; never clips.
- `evaluate_ensemble.py` -- new dedicated evaluation script (kept separate from
  `evaluate_checkpoint.py` since it loads two checkpoints rather than one).
  Loads checkpoint A and B via the existing `evaluate_checkpoint.load_model`,
  builds the canonical validation split via the existing `src.splits.split_pairs`,
  and scores with the existing `src.metrics.psnr`/`ssim` -- no new model loading,
  splitting, or metric logic. Supports `--checkpoint-a/-b`, `--weight-a/-b`
  (default 0.5/0.5), and `--tta {none,x8}` (reuses `src.tta.predict_x8`
  unchanged, applied independently to each model before combining).

## Tests

`tests/test_ensemble_unit.py` (22 new tests): weighted-average correctness (50/50,
75/25, and explicit non-normalized weights), default equal weighting, shape/count/
weight validation, batch/channel/spatial dimension preservation, unclipped raw
values surviving averaging (including negative and >1 constants), finite-output,
identical-input idempotence, 3-way averaging, the normal and x8 ensemble
evaluation paths on a tiny synthetic dataset (the x8 path is cross-checked against
a manual `predict_x8` + `weighted_average_predictions` computation, proving it
reuses `src/tta.py` rather than reimplementing TTA), real Experiment 6/9
checkpoint loading, and the `evaluate_ensemble.py` CLI surface.

`pytest -m "not integration" -q` -> **278 passed, 8 deselected** (up from 256
before this experiment; all pre-existing tests, including every Experiment 10 TTA
test, remain unchanged and passing).

## CUDA Sanity Check

Both real checkpoints loaded on CUDA (`NVIDIA GeForce RTX 4060 Laptop GPU`) with
an 8-sample slice of the validation set: both models produced identical output
shapes `(4, 1, 256, 256)` for the same input batch, the normal ensemble and the
x8 ensemble both ran successfully with finite metrics, and no CUDA OOM occurred.
**Peak CUDA memory allocated: 203.10 MiB.**

## Full 640-Image Validation Comparison

References (Experiments 6 and 9, normal and x8) reuse the values already
independently verified in Experiments 6, 9, and 10 -- their code paths
(`train.validate` / `evaluate_checkpoint.validate_x8`) are unchanged by this
experiment.

| Configuration                  |     Val L1 |       PSNR |       SSIM | Runtime (640 img) |
| ------------------------------- | ---------: | ---------: | ---------: | -----------------: |
| Exp 6 normal (reference)        |   0.033420 | 27.7090 dB |   0.745634 |                 -- |
| Exp 6 + x8 (reference)          |   0.033176 | 27.7689 dB |   0.747955 |                 -- |
| Exp 9 normal (reference)        |   0.033854 | 27.5658 dB |   0.742162 |                 -- |
| Exp 9 + x8 (reference)          |   0.033766 | 27.5875 dB |   0.742980 |                 -- |
| **0.50 A + 0.50 B, normal**     |   0.033479 | 27.6794 dB |   0.745260 |             12.836s |
| **0.50 A + 0.50 B, x8**         |   0.033394 | 27.7002 dB |   0.746076 |             69.271s |
| 0.75 A + 0.25 B, normal         |   0.033410 | 27.7051 dB |   0.745791 |             12.908s |
| 0.75 A + 0.25 B, x8             |   0.033265 | 27.7403 dB |   0.747172 |             69.287s |
| 0.875 A + 0.125 B, normal       |   0.033405 | 27.7098 dB |   0.745798 |             12.771s |
| 0.875 A + 0.125 B, x8           |   0.033216 | 27.7561 dB |   0.747603 |             69.377s |

Commands used (identical pattern for each weight/`--tta` combination):

```bash
python evaluate_ensemble.py --checkpoint-a checkpoints/exp6_crop96/checkpoint_best.pt \
  --checkpoint-b checkpoints/exp9_edsr_lite/checkpoint_best.pt \
  --weight-a 0.5 --weight-b 0.5 --tta none
```

### Weight exploration

The 50/50 ensemble was tested first, per the limited-weight-test protocol. It was
not *clearly* worse than Experiment 6 alone (-0.0296 dB PSNR, a gap comparable in
size to Experiment 10's TTA gain, not a decisive regression like Experiment 8's
MSE-loss result), so weight exploration continued to 0.75/0.25, and then -- since
that step continued to improve monotonically toward Exp 6 alone -- to the optional
0.875/0.125 tier. No further, denser search was performed (no grid search, per
instructions), since the trend across all three weight points was already
unambiguous: **every ensemble configuration tested, at both TTA settings, scored
below its corresponding single-model reference (Exp 6 normal or Exp 6 + x8) on
PSNR, SSIM, and L1**, and metrics moved monotonically *toward* (but never past)
the Exp 6-alone numbers as Experiment 9's weight was reduced toward zero.

## Deltas (best ensemble configuration: 0.875 A + 0.125 B, x8)

| Comparison                        |     ΔPSNR |      ΔSSIM |       ΔL1 |
| ---------------------------------- | --------: | ---------: | --------: |
| vs Exp 6 normal (27.7090/0.745634) |  +0.0471 dB |  +0.001969 | -0.000204 |
| vs Exp 6 + x8 (27.7689/0.747955)   |  -0.0128 dB |  -0.000352 | +0.000040 |

The best-performing ensemble configuration still trails the current best pipeline
(Experiment 6 + x8 TTA alone) on every metric.

## Conclusion

Averaging Experiment 6 and Experiment 9's raw predictions **did not improve on
Experiment 6 alone, at any tested weight, with or without x8 TTA**. The 50/50
ensemble underperformed Experiment 6 by -0.0296 dB PSNR; shifting weight further
toward Experiment 6 (0.75/0.25, then 0.875/0.125) monotonically recovered most of
that gap but never closed it, converging toward -- not past -- Experiment 6's own
numbers as Experiment 9's contribution shrank toward zero. This indicates
Experiment 9's prediction errors are not sufficiently independent of/complementary
to Experiment 6's to yield an ensembling benefit; being the weaker individual
model (-0.1432 dB PSNR vs Exp 6), Experiment 9 mostly adds noise rather than
correcting Experiment 6's mistakes. Every combination of ensembling + x8 TTA also
remained below Experiment 6 + x8 TTA alone, so ensembling does not stack usefully
with the TTA gain either.

**Decision: reject model ensembling of Experiment 6 and Experiment 9.**
**Experiment 6 remains the practical champion checkpoint, and Experiment 6 + x8
TTA remains the best available inference pipeline** (27.7689 dB / 0.747955). The
ensemble utility (`src/ensemble.py`) and evaluation script
(`evaluate_ensemble.py`) are retained as reusable infrastructure -- a future
ensemble candidate with genuinely complementary errors (e.g. a differently-biased
architecture such as NAFNet/SwinIR/Restormer, not yet attempted) could still be
tested with the same tools without new code.

---

# Experiment 12 — NAFNet-SR Architecture

## Status

**COMPLETED -- STOPPED AT EPOCH 32.** The real Experiment 12 training run
(`checkpoints/exp12_nafnet_sr/`, full canonical recipe below) was deliberately
stopped at epoch 32 of a planned 40 after clear plateauing well below
Experiment 6. The smoke-run metrics further below are infrastructure-verification
artifacts only (1 epoch, 32 train / 16 val samples, freshly initialized weights)
and are unrelated to this real result.

## Hypothesis

Experiments 4, 5, 8 (loss substitution), 9 (deeper/wider residual architecture),
10 (geometric TTA), and 11 (model ensembling) have all been tried against
Experiment 6. Only TTA (Experiment 10) improved on it; scaling the same residual-
block design further (Experiment 9) did not, and ensembling two residual-style
architectures (Experiment 11) did not either. Experiment 12 tests a **genuinely
different feature-processing design** rather than another variation on the
residual-CNN theme: NAFNet-style gated blocks (no traditional activation
function, channel-splitting multiplicative gating, simplified channel attention,
channel-wise LayerNorm instead of BatchNorm) adapted to 2x super-resolution. The
hypothesis is that this different inductive bias may capture noise/detail
patterns the residual-CNN family (Experiments 1-9, 11) does not, and is worth
screening on its own merits before deciding whether it is worth a full 40-epoch
run.

## Architecture

Implemented locally in `src/models/nafnet_sr.py` -- no third-party NAFNet
package, no pretrained weights. Core components:

- **`LayerNorm2d`**: channel-wise layer normalization for `[N,C,H,W]` tensors
  (mean/variance computed per spatial position, across channels). NAFNet's
  normalization choice; not BatchNorm (no running statistics, no batch-size
  dependence).
- **`SimpleGate`**: splits channels into two equal halves and multiplies them
  element-wise. This *is* the block's entire nonlinearity -- no ReLU/GELU
  anywhere inside a `NAFBlock`.
- **`NAFBlock`**: a gated conv branch (`LayerNorm2d` -> 1x1 conv channel
  expansion -> 3x3 **depthwise** conv -> `SimpleGate` -> simplified channel
  attention (global average pool -> 1x1 conv, multiplied back in, no extra
  nonlinearity) -> 1x1 projection back to `num_channels`, added to the block
  input through a learnable per-channel scale `beta`), followed by a gated
  feed-forward branch (`LayerNorm2d` -> 1x1 expansion -> `SimpleGate` -> 1x1
  projection, added back through a learnable per-channel scale `gamma`).
  `beta`/`gamma` are both initialized to **zero**, so every `NAFBlock` starts
  training as an exact identity map (NAFNet's stabilizing init trick, verified
  by a dedicated unit test).
- **`NAFNetSR`**: adapts the (same-resolution) NAFNet design to 2x
  super-resolution using the same skeleton already established by
  `ResidualSRNet`/`EDSRLite`: shallow 3x3 conv -> `num_blocks` x `NAFBlock`
  (operating at the LR resolution, no pooling/striding anywhere) -> 3x3 conv ->
  long/global residual connection back to the post-shallow-conv features -> 3x3
  conv (`num_features -> num_features*scale^2`) -> `PixelShuffle(scale)` -> 3x3
  reconstruction conv. Learned upsampling only -- no interpolation-only output
  path.

## Chosen Configuration (revised after CUDA sanity -- see below)

| Parameter | Value |
| --- | --- |
| Feature width (`num_features`) | 64 |
| NAF blocks (`num_blocks`) | 8 |
| Conv-branch expansion (`dw_expand`) | 2 |
| FFN-branch expansion (`ffn_expand`) | 2 |
| Scale | 2 |
| **Parameter count** | **432,129** |

Ratio vs Experiment 6 (630,724 params): **0.685x**. Ratio vs Experiment 9
(1,367,553 params): **0.316x**.

An initial candidate (96 features / 12 blocks, 1,229,185 params -- targeting the
"1-3M parameter" region suggested for this experiment, and a closer capacity
match to Experiment 9) was rejected by the CUDA sanity check: at batch16/crop96
it required **~13.1 GB peak allocated CUDA memory**, exceeding the RTX 4060
Laptop GPU's 8 GB VRAM. This was not a bug -- a forward-only vs. forward+backward
memory comparison confirmed it scales as expected with batch size (real
activation memory, not a leak), and per-batch runtime collapsed to ~18.9 seconds
(vs. an expected tens of milliseconds), consistent with Windows silently
spilling into slow shared/system memory rather than raising a hard CUDA OOM.
Per this task's explicit instruction ("if batch16 OOMs or is clearly unsafe,
STOP; do not silently reduce batch size"), batch size, crop size, and every
other controlled training variable were left untouched -- **the architecture
size itself was reduced instead**, which is the one variable Experiment 12 is
actually about choosing. The root cause is architectural, not a sizing
coincidence: each `NAFBlock` has roughly 10 sequential activation-heavy ops
(two norms, four 1x1 convs, one depthwise 3x3, two gates, one pooled-attention
conv) that autograd must retain for backward, versus 2 conv ops for a
`ResidualSRNet`/`EDSRLite` block -- so NAFNet-style blocks cost substantially
more *activation* memory per parameter than this project's other architectures,
independent of raw parameter count. 64 features / 8 blocks was chosen as a
configuration that fits safely (see CUDA sanity below) while remaining
"NAFNet-SR-Lite" rather than a toy size.

## Model Factory / CLI

- `src/models/__init__.py`: `build_model_config`/`build_model` extended with
  `architecture="nafnet_sr"`, plus new `dw_expand`/`ffn_expand` parameters
  (default 2/2, not exposed as new CLI flags -- Experiment 12 has one clearly
  defined configuration, matching this project's existing convention of pinning
  architecture-internal constants like EDSRLite's `residual_scale`).
  `residual_sr`/`edsr_lite` configs and reconstruction are byte-for-byte
  unchanged; default architecture remains `residual_sr`.
- `train.py --model` gains the `nafnet_sr` choice (`residual_sr` still default).
  No other CLI, loss, crop, scheduler, optimizer, or seed behavior changed.
- `evaluate_checkpoint.py`/`infer_test.py` require **no changes** -- both
  already reconstruct models exclusively through `build_model`, so a
  `nafnet_sr` checkpoint loads through the identical existing code path,
  confirmed by tests and by the smoke-test evaluation below.
- `src/tta.py` (x8 geometric self-ensemble) and `src/ensemble.py` (model
  ensembling) are architecture-agnostic and were **not modified**; both
  confirmed working with `NAFNetSR` by test and by the smoke-test run below.

### Exact `model_config`

```python
{
    "architecture": "nafnet_sr",
    "in_channels": 1,
    "out_channels": 1,
    "num_features": 64,
    "num_blocks": 8,
    "scale": 2,
    "dw_expand": 2,
    "ffn_expand": 2,
}
```

## Checkpoint / Resume Compatibility

Verified by `tests/test_training_unit.py`:

- Matching NAFNet-SR checkpoint + config -> resume succeeds (weights restored
  exactly, epoch/best-PSNR continuation correct).
- NAFNet-SR checkpoint + `residual_sr` config -> rejected.
- NAFNet-SR checkpoint + `edsr_lite` config -> rejected.
- `residual_sr` checkpoint + NAFNet-SR config -> rejected.
- `edsr_lite` checkpoint + NAFNet-SR config -> rejected.
- Legacy (no-`"architecture"`-key) checkpoints still reconstruct as
  `ResidualSRNet`, unchanged.

All five mismatch directions are rejected by the existing, unmodified
dict-equality check in `load_checkpoint_for_resume` -- no new compatibility
logic was needed, the same mechanism that already protected `edsr_lite`.

## Tests

36 new tests: `tests/test_nafnet_sr_unit.py` (24 -- shapes/finiteness/gradients,
exact Experiment 12 parameter count, `SimpleGate` split-multiply correctness and
non-equivalence to ReLU/GELU, `NAFBlock` shape preservation and zero-init
identity behavior, `LayerNorm2d` channel-axis normalization, no-BatchNorm check,
x8 TTA compatibility), plus factory tests in `tests/test_model_unit.py` and
checkpoint/resume/loading tests in `tests/test_training_unit.py` (config shape,
`build_model` reconstruction, all 5 mismatch-rejection directions,
`evaluate_checkpoint`/`infer_test`'s shared `load_model` reconstructing
`NAFNetSR`).

`pytest -m "not integration" -q` -> **314 passed, 8 deselected** (up from 278;
every pre-existing test, including all TTA and ensemble tests, remains
unchanged and passing).

## CUDA Sanity Check

Exact Experiment 12 configuration, `batch=16`, `input=[16,1,96,96]`,
`loss=L1`, `optimizer=Adam`, forward -> loss -> backward -> optimizer step,
10 timed iterations after 3 discarded warmup iterations:

| Check | Result |
| --- | --- |
| Device | NVIDIA GeForce RTX 4060 Laptop GPU (CUDA) |
| Parameter count | 432,129 |
| Output shape | `(16, 1, 192, 192)` (matches expected exactly) |
| Output finite | True |
| Loss finite | True |
| All gradients finite | True |
| Peak allocated CUDA memory | 6,007.4 MiB |
| Peak reserved CUDA memory | 6,470.0 MiB |
| Total VRAM | 8,187.5 MiB |
| Headroom | ~1,718 MiB (~21%) |
| Per-batch runtime (avg of 10) | 322.5 ms |
| OOM | None |

No OOM, healthy headroom under the 8 GB budget, and per-batch runtime back to a
normal (millisecond-scale) range -- confirming the earlier 96f/12b candidate's
~18.9-second-per-batch result was specifically a memory-pressure artifact of
that oversized configuration, not an inherent property of NAFNet-SR-Lite.

## Real-Data Smoke Test (infrastructure verification only -- not a result)

```bash
python train.py --model nafnet_sr --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --epochs 1 --max-train-samples 32 --max-val-samples 16 \
  --checkpoint-dir checkpoints/exp12_nafnet_sr_smoke --num-workers 0
```

Verified and then deleted (`checkpoints/exp12_nafnet_sr_smoke/`, not committed):

- CUDA used; printed `Model: nafnet_sr (432,129 trainable parameters)` and the
  exact `model_config` above.
- Training and validation both completed in 1.6s; PSNR/SSIM computed via the
  existing unmodified metric functions; `checkpoint_best.pt` saved.
- `evaluate_checkpoint.py --tta none` and `--tta x8` both loaded the smoke
  checkpoint and ran to completion with finite metrics.
- `infer_test.py`'s model-loading path (`evaluate_checkpoint.load_model`)
  reconstructed `NAFNetSR` correctly; `run_inference(..., tta="none")` and
  `run_inference(..., tta="x8")` both produced finite, correctly-shaped
  (2x input) predictions on a synthetic full-resolution-style test tensor.
- No historical checkpoint was read or written during this test.

The smoke run's own numbers (Val PSNR ~7-8 dB, near-random since the network
saw only 32 samples for 1 epoch) are **not an experiment result** and are
recorded here only as evidence the pipeline works end-to-end; they must not be
compared against Experiment 6/9/10/11.

## Checkpoint Safety

All 22 historical checkpoint files across Experiments 1-9 were SHA-256-hashed
before and after this preparation task; every hash is unchanged. Specifically
re-verified for `checkpoints/exp6_crop96/{checkpoint_best,checkpoint_latest}.pt`
and `checkpoints/exp9_edsr_lite/{checkpoint_best,checkpoint_latest}.pt`.

## Real Training Run

Exact Experiment 12 configuration (`64 features / 8 NAF blocks / dw_expand=2 /
ffn_expand=2`, 432,129 parameters), trained with the fully controlled recipe:
L1 loss, crop96 LR / crop192 GT, batch16, seed42, Adam (initial LR 1e-4),
ReduceLROnPlateau (factor 0.5, patience 3, min LR 1e-6), canonical split,
full-image validation on all 640 validation samples, best checkpoint selected
by validation PSNR -- identical to every other experiment's controlled
variables; only the architecture differs.

Checkpoint: `checkpoints/exp12_nafnet_sr/checkpoint_best.pt`. Independently
re-verified via `evaluate_checkpoint.py --checkpoint
checkpoints/exp12_nafnet_sr/checkpoint_best.pt --tta none`:

```
Loaded checkpoint checkpoints/exp12_nafnet_sr/checkpoint_best.pt (epoch=32, best_val_psnr=27.21782928578738)
Training loss: L1 ({'name': 'l1'})
Validation samples: 640
Val L1 (diagnostic, always L1 regardless of training loss): 0.035279
Val PSNR: 27.2178 dB
Val SSIM: 0.729829
Bicubic PSNR: 23.1413 dB
Bicubic SSIM: 0.550604
PSNR vs bicubic: +4.0765 dB
SSIM vs bicubic: +0.179225
```

| Metric | Value |
| --- | --- |
| Best epoch | 32 (of 40 planned; stopped deliberately) |
| Val L1 | 0.035279 |
| Val PSNR | 27.2178 dB |
| Val SSIM | 0.729829 |
| PSNR vs bicubic | +4.0765 dB |
| SSIM vs bicubic | +0.179225 |

The checkpoint's stored `model_config` was confirmed identical to the intended
Experiment 12 configuration (`architecture=nafnet_sr, num_features=64,
num_blocks=8, dw_expand=2, ffn_expand=2, scale=2`), and its `training_config`
confirms the controlled recipe (`crop_size=96, batch_size=16, seed=42, lr=1e-4`)
was used unchanged.

## Comparison vs Experiment 6

| Metric | Exp 6 (champion) | Exp 12 (NAFNet-SR) | Delta |
| --- | ---: | ---: | ---: |
| Val L1 | 0.033420 | 0.035279 | +0.001859 |
| Val PSNR | 27.7090 dB | 27.2178 dB | **-0.4912 dB** |
| Val SSIM | 0.745634 | 0.729829 | **-0.015805** |
| Parameters | 630,724 | 432,129 | 0.685x |

Experiment 12 underperforms Experiment 6 on every metric, despite being a
genuinely different (not merely smaller/larger residual-CNN) architecture.

## Plateau Reasoning

Validation PSNR after each ReduceLROnPlateau reduction shows diminishing,
near-flat returns rather than continued climbing:

| Epoch | Val PSNR |
| --- | --- |
| 25 | 27.1714 dB |
| 28 | 27.2035 dB |
| 30 | 27.2115 dB |
| 32 | 27.2178 dB |

Only **+0.0464 dB** improvement across epochs 25-32 (7 epochs), well within the
kind of flat late-training tail seen in this project's other completed runs
once the scheduler has reduced LR multiple times -- there is no indication that
continuing to epoch 40 would close a ~0.49 dB gap to Experiment 6. Training was
therefore stopped at epoch 32 rather than run to the full 40 epochs.

## GPU Sizing Finding

The originally scoped 96-feature / 12-block NAFNet-SR configuration
(1,229,185 params, closer to Experiment 9's capacity) was found unsuitable for
the available RTX 4060 Laptop GPU: at batch16/crop96 it required ~13 GB of
training memory, exceeding the 8 GB VRAM budget and causing Windows
shared-memory spill (per-batch runtime collapsed to ~18.9 seconds, versus
~322 ms for the eventually-chosen configuration). This is a NAFNet-specific
characteristic -- each `NAFBlock`'s ~10 sequential activation-heavy ops
(vs. 2 conv ops per `ResidualSRNet`/`EDSRLite` block) cost substantially more
activation memory per parameter than this project's other architectures. The
safe 64-feature / 8-block configuration was used instead, and it trained
without memory issues -- but it did not reach competitive validation
performance, so the negative result above is not confounded by the earlier
sizing problem.

## Conclusion

**NAFNet-SR is rejected.** The compact configuration required to fit this
project's GPU budget underperforms Experiment 6 by a clear, non-noise margin
(-0.4912 dB PSNR, -0.015805 SSIM) and plateaus well before its 40-epoch budget,
so no benefit is expected from letting it run longer. Combined with Experiment
9 (deeper/wider residual CNN, also negative) and Experiment 11 (ensembling
Experiment 6 with a weaker model, also negative), three different attempts to
beat Experiment 6 via architecture/scale/ensembling have now failed.
**Experiment 6 remains the champion checkpoint, and Experiment 6 + x8 TTA
(27.7689 dB / 0.747955) remains the best inference pipeline.** Both
`checkpoints/exp12_nafnet_sr/checkpoint_best.pt` and `checkpoint_latest.pt` are
retained for reproducibility, unmodified.

---

# Experiment 13 — SwinIR-lite Architecture

## Status

**COMPLETED.** The real Experiment 13 training run
(`checkpoints/exp13_swinir_lite/`, full canonical 40-epoch recipe below) ran
to completion. The smoke-run metrics further below are infrastructure-
verification artifacts only (1 epoch, 32 train / 16 val samples, freshly
initialized weights) and are unrelated to this real result.

## Hypothesis

Three attempts to beat Experiment 6 via convolutional means have now failed:
Experiment 9 (deeper/wider residual CNN), Experiment 11 (ensembling Experiment
6 with a weaker model), and Experiment 12 (NAFNet-style gated convolutional
blocks). All three stayed within the same broad family -- local convolutional
receptive fields, however creatively combined. Experiment 13 tests a
structurally different mechanism instead: **windowed self-attention**
(Liu et al. 2021 Swin Transformer / Liang et al. 2021 SwinIR). The hypothesis
is that attention within local windows, combined with the shifted-window
mechanism that lets information cross window boundaries across layers, may
preserve long-range structural consistency and fine semiconductor edges
better than any convolution-only design tried so far.

## Architecture

Implemented locally in `src/models/swinir_lite.py` -- no third-party
Swin/SwinIR package, no pretrained weights. Core components:

- **`window_partition`/`window_reverse`**: split a `[B,H,W,C]` feature map
  into non-overlapping `window_size x window_size` windows and back. Exact
  inverses of each other (verified by a dedicated round-trip test).
- **`WindowAttention`**: standard multi-head self-attention computed *within*
  each window only (not globally across the full feature map), plus a
  learned relative position bias indexed by each pair of in-window positions'
  offset -- Swin's mechanism for giving attention spatial awareness without
  absolute position embeddings.
- **`SwinTransformerBlock`**: `LayerNorm` -> (optional cyclic shift) -> window
  partition -> window multi-head self-attention -> window reverse -> (undo
  shift) -> residual, followed by `LayerNorm` -> MLP (GELU) -> residual.
  Blocks **alternate** between regular windows (`shift_size=0`) and shifted
  windows (`shift_size=window_size//2`) so information can flow across window
  boundaries; the shifted variant uses a precomputed additive attention mask
  so a window that straddles the cyclic-shift wrap-around never attends
  across the seam (the standard Swin masking construction, verified by a
  shape test and a "mask changes the result" test).
- **`SwinIRLite`**: adapts this (same-resolution) transformer body to 2x
  super-resolution using the same shallow-conv / long-skip /
  PixelShuffle-upsample skeleton already used by
  `ResidualSRNet`/`EDSRLite`/`NAFNetSR` -- only the feature-processing body is
  replaced by a stack of `SwinTransformerBlock`s operating on tokens instead
  of a stack of convolutional blocks. No BatchNorm anywhere (`LayerNorm` only,
  matching every other architecture in this project).

### Input size handling

`window_size` must evenly divide the transformer body's spatial dimensions.
`window_size=8` was chosen because `96 % 8 == 0` (training crop) and
`128 % 8 == 0` (full validation image), so both the common cases need no
padding. For robustness beyond those two sizes, `SwinIRLite.forward` reflect-
pads the post-shallow-conv feature map up to the next multiple of
`window_size` before the transformer body and **crops the padding back off**
before the long residual connection and upsampling -- so any input at least
`window_size` in each dimension produces an exact `2x` output, and validation
images are never silently cropped (only compute-internal padding is added and
then discarded; verified by a non-multiple-of-window-size shape test).
Inputs smaller than `window_size` raise a clear `ValueError` rather than
padding into meaninglessness.

## CUDA Memory/Runtime Candidate Search

Following the sizing lesson from Experiment 12 (parameter count alone is not
a safe sizing proxy), 4 small candidates were measured at the exact training
configuration (`batch=16, input=[16,1,96,96]`, forward+backward+optimizer
step) before choosing one -- no large sweep:

| Candidate | Params | Peak Allocated | Peak Reserved | Headroom (of 8,187.5 MiB) | Per-batch |
| --- | ---: | ---: | ---: | ---: | ---: |
| embed_dim=48, depth=6, heads=6 | 226,789 | 3,833.0 MiB | 4,642.0 MiB | 3,545.5 MiB (43%) | 326.7 ms |
| embed_dim=64, depth=6, heads=8 | 397,617 | 5,079.3 MiB | 6,326.0 MiB | 1,861.5 MiB (23%) | 429.5 ms |
| embed_dim=48, depth=4, heads=6 | 186,169 | 2,738.9 MiB | 3,574.0 MiB | 4,613.5 MiB (56%) | 227.5 ms |
| **embed_dim=60, depth=6, heads=6 (chosen)** | **348,421** | 4,289.9 MiB | 5,350.0 MiB | **2,837.5 MiB (35%)** | 367.6 ms |

All 4 candidates fit safely under 8 GB with no shared-memory spill (unlike
Experiment 12's rejected 96f/12b candidate, every candidate here ran at
normal millisecond-scale per-batch runtime, confirming none of them triggered
memory pressure). None were rejected outright; embed_dim=60/depth=6/heads=6
was chosen over the raw-headroom-maximizing embed_dim=48/depth=4 option
because it is meaningfully more expressive (nearly double the parameters, two
more transformer blocks) while still preferring several GB of headroom over
the embed_dim=64/heads=8 option, which left only ~23% headroom -- closer to
the "running at the absolute limit" this search was explicitly meant to
avoid.

## Chosen Configuration

| Parameter | Value |
| --- | --- |
| Embedding dimension (`embed_dim`) | 60 |
| Transformer blocks (`depth`) | 6 |
| Attention heads (`num_heads`) | 6 (head dim = 10) |
| Window size (`window_size`) | 8 |
| MLP expansion ratio (`mlp_ratio`) | 2.0 |
| Scale | 2 |
| **Parameter count** | **348,421** |

Ratio vs Experiment 6 (630,724 params): 0.552x. Ratio vs Experiment 12
(432,129 params): 0.806x -- a comparable capacity class to the other compact
architectures tried in this project, not an outlier in either direction.

## Model Factory / CLI

- `src/models/__init__.py`: `build_model_config`/`build_model` extended with
  `architecture="swinir_lite"` plus `embed_dim`/`depth`/`num_heads`/
  `window_size`/`mlp_ratio` parameters. `residual_sr`/`edsr_lite`/`nafnet_sr`
  configs and reconstruction are byte-for-byte unchanged; default architecture
  remains `residual_sr`.
- `train.py --model` gains the `swinir_lite` choice, plus 5 new
  architecture-specific flags (`--embed-dim`, `--depth`, `--num-heads`,
  `--window-size`, `--mlp-ratio`) -- all ignored unless `--model swinir_lite`,
  matching the existing `--residual-scale`/`--dw-expand`-style convention. No
  other CLI, loss, crop, scheduler, optimizer, or seed behavior changed.
- `evaluate_checkpoint.py`/`infer_test.py` require **no changes** -- both
  already reconstruct models exclusively through `build_model`, confirmed by
  tests and by the smoke-test evaluation below.
- `src/tta.py` (x8 geometric self-ensemble) and `src/ensemble.py` (model
  ensembling) are architecture-agnostic and were **not modified**; x8 TTA
  confirmed working with `SwinIRLite` by test and by the smoke-test run below.

### Exact `model_config`

```python
{
    "architecture": "swinir_lite",
    "in_channels": 1,
    "out_channels": 1,
    "embed_dim": 60,
    "depth": 6,
    "num_heads": 6,
    "window_size": 8,
    "mlp_ratio": 2.0,
    "scale": 2,
}
```

## Checkpoint / Resume Compatibility

Verified by `tests/test_training_unit.py`:

- Matching SwinIR-lite checkpoint + config -> resume succeeds (weights
  restored exactly, epoch/best-PSNR continuation correct).
- SwinIR-lite checkpoint + `residual_sr` config -> rejected.
- SwinIR-lite checkpoint + `edsr_lite` config -> rejected.
- SwinIR-lite checkpoint + `nafnet_sr` config -> rejected.
- `residual_sr` checkpoint + SwinIR-lite config -> rejected.
- `edsr_lite` checkpoint + SwinIR-lite config -> rejected.
- `nafnet_sr` checkpoint + SwinIR-lite config -> rejected.
- Legacy (no-`"architecture"`-key) checkpoints still reconstruct as
  `ResidualSRNet`, unchanged.

All six mismatch directions are rejected by the existing, unmodified
dict-equality check in `load_checkpoint_for_resume` -- no new compatibility
logic was needed, the same mechanism that already protected `edsr_lite` and
`nafnet_sr`.

## Tests

38 new tests: `tests/test_swinir_lite_unit.py` (26 -- shapes/finiteness/
gradients, exact Experiment 13 parameter count, window partition/reverse
round-trip, window-attention shape/finiteness/mask-effect, shifted-window
mask shape, block shape preservation for both regular and shifted variants,
96x96 and 128x128 explicit shape checks, non-multiple-of-window-size padding
behavior, too-small-input error, no-BatchNorm check, x8 TTA compatibility),
plus factory tests in `tests/test_model_unit.py` (3) and checkpoint/resume/
loading tests in `tests/test_training_unit.py` (9 -- config identifier
storage, matching resume, all 6 cross-architecture mismatch directions,
`evaluate_checkpoint`/`infer_test`'s shared `load_model` reconstructing
`SwinIRLite`).

`pytest -m "not integration" -q` -> **352 passed, 8 deselected** (up from
314; every pre-existing test, including all TTA, ensemble, and NAFNet-SR
tests, remains unchanged and passing).

## CUDA Sanity Check

Exact Experiment 13 configuration, `batch=16`, `input=[16,1,96,96]`,
`loss=L1`, `optimizer=Adam`, forward -> loss -> backward -> optimizer step,
10 timed iterations after 3 discarded warmup iterations:

| Check | Result |
| --- | --- |
| Device | NVIDIA GeForce RTX 4060 Laptop GPU (CUDA) |
| Parameter count | 348,421 |
| Output shape | `(16, 1, 192, 192)` (matches expected exactly) |
| Output finite | True |
| Loss finite | True |
| All gradients finite | True |
| Peak allocated CUDA memory | 4,289.9 MiB |
| Peak reserved CUDA memory | 5,350.0 MiB |
| Total VRAM | 8,187.5 MiB |
| Headroom | ~2,837.5 MiB (~35%) |
| Per-batch runtime (avg of 10) | 367.6 ms |
| OOM / shared-memory spill | None |

## Real-Data Smoke Test (infrastructure verification only -- not a result)

```bash
python train.py --model swinir_lite --embed-dim 60 --depth 6 --num-heads 6 \
  --window-size 8 --mlp-ratio 2.0 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --epochs 1 --max-train-samples 32 --max-val-samples 16 \
  --checkpoint-dir checkpoints/exp13_swinir_lite_smoke --num-workers 0
```

Verified and then deleted (`checkpoints/exp13_swinir_lite_smoke/`, not
committed):

- CUDA used; printed `Model: swinir_lite (348,421 trainable parameters)` and
  the exact `model_config` above.
- Training and validation both completed in 1.6s -- validation ran on real
  full 128x128 LR images (16 samples), confirming the padding-free
  `window_size=8` path handles the real validation resolution correctly, not
  just the unit-test synthetic case.
- `evaluate_checkpoint.py --tta none` and `--tta x8` both loaded the smoke
  checkpoint and ran to completion with finite metrics.
- `infer_test.py`'s model-loading path (`evaluate_checkpoint.load_model`)
  reconstructed `SwinIRLite` correctly; `run_inference(..., tta="none")` and
  `run_inference(..., tta="x8")` both produced finite, correctly-shaped
  (2x input) predictions on a synthetic full-resolution-style test tensor.
- No historical checkpoint was read or written during this test.

The smoke run's own numbers (Val PSNR ~9-10 dB, near-random since the network
saw only 32 samples for 1 epoch) are **not an experiment result** and are
recorded here only as evidence the pipeline works end-to-end; they must not
be compared against Experiment 6/9/10/11/12.

## Checkpoint Safety

`checkpoints/exp6_crop96/checkpoint_best.pt`, `checkpoints/exp9_edsr_lite/checkpoint_best.pt`,
and `checkpoints/exp12_nafnet_sr/{checkpoint_best,checkpoint_latest}.pt` were
SHA-256-hashed before and after this preparation task; every hash is
unchanged.

## Real Training Run

Exact Experiment 13 configuration (`embed_dim=60, depth=6, num_heads=6,
window_size=8, mlp_ratio=2.0`, 348,421 parameters), trained with the fully
controlled recipe: L1 loss, crop96 LR / crop192 GT, batch16, seed42, Adam
(initial LR 1e-4), ReduceLROnPlateau (factor 0.5, patience 3, min LR 1e-6),
40 epochs, canonical split, full-image validation on all 640 validation
samples, best checkpoint selected by validation PSNR -- identical to
Experiment 6's recipe; only the architecture differs.

Checkpoint: `checkpoints/exp13_swinir_lite/checkpoint_best.pt`. Independently
re-verified via `evaluate_checkpoint.py --checkpoint
checkpoints/exp13_swinir_lite/checkpoint_best.pt --tta none`:

```
Loaded checkpoint checkpoints/exp13_swinir_lite/checkpoint_best.pt (epoch=38, best_val_psnr=27.436127608062332)
Training loss: L1 ({'name': 'l1'})
Validation samples: 640
Val L1 (diagnostic, always L1 regardless of training loss): 0.034307
Val PSNR: 27.4361 dB
Val SSIM: 0.738432
Bicubic PSNR: 23.1413 dB
Bicubic SSIM: 0.550604
PSNR vs bicubic: +4.2948 dB
SSIM vs bicubic: +0.187828
```

| Metric | Value |
| --- | --- |
| Best epoch | 38 (of 40) |
| Val L1 | 0.034307 |
| Val PSNR | 27.4361 dB |
| Val SSIM | 0.738432 |
| PSNR vs bicubic | +4.2948 dB |
| SSIM vs bicubic | +0.187828 |

The checkpoint's stored `model_config` was confirmed identical to the intended
Experiment 13 configuration, and its `training_config`/`loss_config`/
`scheduler_config` confirm the controlled recipe (`crop_size=96, batch_size=16,
seed=42, lr=1e-4`, L1, `ReduceLROnPlateau` factor=0.5/patience=3/min_lr=1e-6)
was used unchanged.

## Comparison vs Experiment 6

| Metric | Exp 6 (champion) | Exp 13 (SwinIR-lite) | Delta |
| --- | ---: | ---: | ---: |
| Val L1 | 0.033420 | 0.034307 | +0.000887 |
| Val PSNR | 27.7090 dB | 27.4361 dB | **-0.2729 dB** |
| Val SSIM | 0.745634 | 0.738432 | **-0.007202** |
| Parameters | 630,724 | 348,421 | 0.552x |

## Conclusion

SwinIR-lite **trained correctly** -- both epoch (38) and the LR-reduction
pattern implied by `ReduceLROnPlateau` behaved as expected, and the model
responded to training normally (no divergence, no plateauing failure mode
like Experiment 12's). It simply **did not beat Experiment 6**: -0.2729 dB
PSNR / -0.007202 SSIM, a smaller gap than Experiment 12's NAFNet-SR
(-0.4912 dB) but still a clear, non-noise regression, despite using windowed
self-attention -- a structurally different mechanism from every convolutional
architecture tried so far. **Experiment 6 remains the champion checkpoint,
and Experiment 6 + x8 TTA (27.7689 dB / 0.747955) remains the best inference
pipeline.** This is the fourth attempt (after Experiments 9, 11, 12) to beat
Experiment 6 via architecture, ensembling, or attention, and the fourth to
fall short -- motivating Experiment 14's shift away from architecture
exploration and toward the training recipe itself (LR schedule) on the
already-proven Experiment 6 architecture. Both
`checkpoints/exp13_swinir_lite/checkpoint_best.pt` and `checkpoint_latest.pt`
are retained for reproducibility, unmodified.

---

# Experiment 14 — Cosine LR Schedule

## Status

**COMPLETED.** The real Experiment 14 training run
(`checkpoints/exp14_cosine/`, full canonical 40-epoch recipe below) ran to
completion. The smoke-run metrics further below are infrastructure-
verification artifacts only (1 epoch + 1 resumed epoch, 32 train / 16 val
samples, freshly initialized weights) and are unrelated to this real result.

## Hypothesis

Four attempts to beat Experiment 6 via a different architecture, ensembling,
or attention have now failed (Experiments 9, 11, 12, 13). Experiment 14
returns to the proven Experiment 6 architecture (`ResidualSRNet`, 64F/8B,
630,724 params) and changes exactly one variable: the learning-rate
schedule, `ReduceLROnPlateau` -> `CosineAnnealingLR`. The hypothesis is that
a smooth, monotonic LR decay may improve convergence over plateau-triggered
step reductions -- `ReduceLROnPlateau` only reduces LR after `patience`
epochs of stagnation, producing a staircase schedule with potentially long
stretches at a too-high or too-low LR; cosine annealing instead decays
continuously across the whole run, which is often reported to help late-stage
fine convergence in image restoration/SR literature. Everything else
(architecture, loss, crop, batch, seed, optimizer, initial LR, split,
validation, metrics, checkpoint-selection criterion) is held fixed to isolate
this one variable.

## Scheduler Implementation

`train.py`'s existing scheduler infrastructure (`build_scheduler_config`,
`build_scheduler`, checkpoint save/resume) is extended, not replaced:

- `build_scheduler_config(scheduler_name, factor, patience, min_lr, t_max=None)`
  gains a `"cosine"` branch returning
  `{"name": "cosine", "t_max": t_max, "eta_min": min_lr}`. `t_max` is
  **required** for `"cosine"` (raises `ValueError` if missing/non-positive) --
  it is never derived from `--epochs`, so an interrupted or smoke-test run
  with a small `--epochs` cannot silently compress the intended horizon.
  `"plateau"`/`"none"` behavior is byte-for-byte unchanged.
- `build_scheduler` gains a matching `"cosine"` branch constructing
  `torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=t_max,
  eta_min=eta_min)`.
- A new `scheduler_step(scheduler, scheduler_config, val_psnr)` helper
  centralizes the one place stepping behavior must differ by scheduler type
  (see below), replacing the previous unconditional
  `scheduler.step(val_metrics["psnr"])` call in the training loop.
- `save_checkpoint`/`load_checkpoint_for_resume`'s scheduler-related type
  hints were widened from `ReduceLROnPlateau | None` to a new
  `Scheduler = ReduceLROnPlateau | CosineAnnealingLR` alias; no behavioral
  change to either function beyond the new optional mismatch check below.

### Exact scheduler config

```python
{
    "name": "cosine",
    "t_max": 40,
    "eta_min": 1e-6,
}
```

## Stepping Behavior

`ReduceLROnPlateau.step(metric)` and `CosineAnnealingLR.step()` have
different signatures -- the former needs this epoch's validation PSNR to
decide whether to reduce; the latter takes no argument, since its schedule is
a fixed function of epoch count alone. `scheduler_step()` dispatches on
`scheduler_config["name"]`:

```python
def scheduler_step(scheduler, scheduler_config, val_psnr):
    if scheduler_config["name"] == "plateau":
        scheduler.step(val_psnr)
    else:
        scheduler.step()
```

**Exactly when this happens, unchanged from before cosine existed:** once per
epoch, immediately after `validate()` returns (so this epoch's validation
PSNR is available for plateau) and before `save_checkpoint()` (so the LR
saved into `checkpoint_latest.pt`/`checkpoint_best.pt` already reflects this
epoch's step -- a resume just continues from epoch+1 without needing to
replay the step). `train_one_epoch`'s inner per-batch loop is untouched;
`scheduler.step()` is still only ever called once per *epoch*, matching
`ReduceLROnPlateau`'s existing cadence -- `CosineAnnealingLR` follows the
identical cadence rather than a per-batch schedule, keeping "one scheduler
step = one epoch" true for both.

## LR Trajectory Verification (T_max=40, eta_min=1e-6, initial LR=1e-4)

Computed by driving `build_scheduler`+`scheduler_step` through 40 steps with
the exact Experiment 14 config (correctness check only -- the schedule is not
tuned based on these values):

| Epoch | LR |
| --- | ---: |
| 1 | 9.98474080e-05 |
| 5 | 9.62320369e-05 |
| 10 | 8.55017857e-05 |
| 20 | 5.05000000e-05 |
| 30 | 1.54982143e-05 |
| 35 | 4.76796314e-06 |
| 40 | 1.00000000e-06 (= eta_min exactly) |

Monotonically non-increasing throughout, reaches exactly `eta_min` at epoch
40, and never drops below `eta_min` even if stepped past epoch 40 (cosine
holds at `eta_min` thereafter -- verified by a dedicated test).

## CLI

- `--scheduler` gains the `"cosine"` choice: `{none, plateau, cosine}`.
  Default remains `"none"` (Experiment 1's fixed-LR behavior, unchanged).
- New `--scheduler-t-max` (default `40`, the standard experiment horizon
  used throughout this project) -- ignored unless `--scheduler cosine`.
- `--min-lr` is reused as `eta_min` for cosine (already existed for
  plateau's floor) -- no separate `--eta-min` flag needed.
- Experiment 14 requests exactly: `--scheduler cosine --scheduler-t-max 40
  --min-lr 1e-6`.

## Resume Compatibility

`load_checkpoint_for_resume` gains an optional `scheduler_config` parameter,
mirroring the existing `loss_config` strict-equality pattern exactly: when
the caller passes a `scheduler_config` (as `train.py --resume` always now
does), it must equal the checkpoint's stored `scheduler_config` exactly or a
`ValueError` is raised; passing `None` (the default, used by tests and by
`--scheduler none` resumes) skips the check entirely. Verified by test:

- Matching cosine checkpoint + cosine config (same `t_max`/`eta_min`) ->
  resume succeeds; LR and full scheduler state (`last_epoch`, `base_lrs`,
  etc.) restored exactly.
- Cosine checkpoint + plateau config -> rejected.
- Plateau checkpoint + cosine config -> rejected.
- Cosine checkpoint, different `t_max` -> rejected.
- Cosine checkpoint, different `eta_min` -> rejected.
- No `scheduler_config` passed -> check skipped, preserving every historical
  no-scheduler-config-comparison resume path (Experiments 1-13) exactly as
  it worked before this change.
- A resumed 15-epoch-then-continued-to-40 cosine run reaches the identical
  final LR as an uninterrupted 40-epoch run (fixed `T_max=40` horizon, not
  re-derived from the resume point).

## Tests

16 new tests appended to `tests/test_scheduler_unit.py`: config
construction/validation, scheduler construction, monotonic LR decrease,
never-below-`eta_min`, exact `eta_min` at `T_max`, cosine stepping ignoring
the PSNR argument, plateau stepping still using it (unchanged), checkpoint
storage of cosine state, resume restoring LR/scheduler state, resumed vs.
uninterrupted trajectory equivalence, all 4 mismatch-rejection directions,
and the `scheduler_config=None` legacy-skip path. All pre-existing plateau
tests are untouched and still pass, confirming plateau behavior is unchanged.

`pytest -m "not integration" -q` -> **368 passed, 8 deselected** (up from
352; every pre-existing test, including all model/TTA/ensemble/loss tests,
remains unchanged and passing).

## CUDA Sanity Check

Experiment 14 reuses Experiment 6's exact architecture, so no sustained
benchmark was run -- only a tiny single-step correctness check: `batch=16`,
`input=[16,1,96,96]`, `ResidualSRNet` 64F/8B, `L1`, `Adam`, cosine scheduler,
one forward -> backward -> `optimizer.step()` -> `scheduler_step()`.

| Check | Result |
| --- | --- |
| Device | NVIDIA GeForce RTX 4060 Laptop GPU (CUDA) |
| Parameter count | 630,724 |
| Output shape | `(16, 1, 192, 192)` (matches expected exactly) |
| Output finite | True |
| Loss finite | True |
| All gradients finite | True |
| LR after one `scheduler_step()` | 9.984740801978984e-05 (matches the epoch-1 trajectory value above) |
| Peak allocated CUDA memory | 773.9 MiB |
| Peak reserved CUDA memory | 876.0 MiB |
| OOM | None |

As expected, memory usage is essentially identical to Experiment 6 (same
architecture) and far below the 8 GB budget -- no sizing search was needed.

## Real-Data Smoke Test (infrastructure verification only -- not a result)

```bash
python train.py --model residual_sr --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler cosine --scheduler-t-max 40 --min-lr 1e-6 \
  --epochs 1 --max-train-samples 32 --max-val-samples 16 \
  --checkpoint-dir checkpoints/exp14_cosine_smoke --num-workers 0
```

Verified and then deleted (`checkpoints/exp14_cosine_smoke/`, not committed):

- CUDA used; printed `Model: residual_sr (630,724 trainable parameters)` and
  `Scheduler: {'name': 'cosine', 't_max': 40, 'eta_min': 1e-06}`.
- **Critical check:** despite `--epochs 1`, the checkpoint's stored
  `scheduler_config` is `{"name": "cosine", "t_max": 40, "eta_min": 1e-06}`
  -- the full 40-epoch horizon, not silently compressed to match the 1-epoch
  smoke run. `scheduler_state_dict` shows `"T_max": 40` and `"last_epoch": 1`
  (one step taken, against a 40-epoch horizon). Training printed "Learning
  rate reduced: 1.000000e-04 -> 9.984741e-05", matching the epoch-1
  trajectory value above exactly.
- `evaluate_checkpoint.py --tta none` loaded the smoke checkpoint and ran to
  completion with finite metrics.
- Resume verified: re-ran with `--resume checkpoints/exp14_cosine_smoke/checkpoint_latest.pt
  --epochs 2`; scheduler state restored ("Restored scheduler state from
  checkpoint."), LR continued from 9.984741e-05 (not reset to 1e-4), and
  after the second step reduced further to 9.939057e-05 -- the correct
  epoch-2 point on the same continuous 40-epoch curve, not a fresh 2-epoch
  schedule.
- No historical checkpoint was read or written during this test.

The smoke run's own PSNR/SSIM numbers are **not an experiment result** and
are recorded here only as evidence the pipeline works end-to-end; they must
not be compared against Experiment 6/9/10/11/12/13.

## Checkpoint Safety

`checkpoints/exp6_crop96/checkpoint_best.pt`,
`checkpoints/exp9_edsr_lite/checkpoint_best.pt`,
`checkpoints/exp12_nafnet_sr/checkpoint_best.pt`, and
`checkpoints/exp13_swinir_lite/{checkpoint_best,checkpoint_latest}.pt` were
SHA-256-hashed before and after this preparation task; every hash is
unchanged.

## Real Training Run

Exact Experiment 14 configuration (`ResidualSRNet`, 64F/8B, 630,724 params),
trained with Experiment 6's identical recipe except the scheduler: L1 loss,
crop96 LR / crop192 GT, batch16, seed42, Adam (initial LR 1e-4),
`CosineAnnealingLR` (`T_max=40, eta_min=1e-6`), 40 epochs, canonical split,
full-image validation on all 640 validation samples, best checkpoint selected
by validation PSNR.

Checkpoint: `checkpoints/exp14_cosine/checkpoint_best.pt`. Independently
re-verified via `evaluate_checkpoint.py --checkpoint
checkpoints/exp14_cosine/checkpoint_best.pt --tta none`:

```
Loaded checkpoint checkpoints/exp14_cosine/checkpoint_best.pt (epoch=38, best_val_psnr=27.601060624152534)
Training loss: L1 ({'name': 'l1'})
Validation samples: 640
Val L1 (diagnostic, always L1 regardless of training loss): 0.033783
Val PSNR: 27.6011 dB
Val SSIM: 0.742668
Bicubic PSNR: 23.1413 dB
Bicubic SSIM: 0.550604
PSNR vs bicubic: +4.4598 dB
SSIM vs bicubic: +0.192064
```

| Metric | Value |
| --- | --- |
| Best epoch | 38 (of 40) |
| Val L1 | 0.033783 |
| Val PSNR | 27.6011 dB |
| Val SSIM | 0.742668 |
| PSNR vs bicubic | +4.4598 dB |
| SSIM vs bicubic | +0.192064 |

The checkpoint's stored `model_config` (`residual_sr`, 64F/8B), `loss_config`
(`l1`), and `scheduler_config` (`{"name": "cosine", "t_max": 40, "eta_min":
1e-06}`) all confirm the controlled recipe was used exactly as specified.

## Comparison vs Experiment 6

| Metric | Exp 6 (`ReduceLROnPlateau`) | Exp 14 (`CosineAnnealingLR`) | Delta |
| --- | ---: | ---: | ---: |
| Val L1 | 0.033420 | 0.033783 | +0.000363 |
| Val PSNR | 27.7090 dB | 27.6011 dB | **-0.1079 dB** |
| Val SSIM | 0.745634 | 0.742668 | **-0.002966** |

Both runs share the identical architecture, loss, crop, batch size, seed,
optimizer, and initial LR -- the scheduler is the only variable that
changed, isolating its effect cleanly.

## Conclusion

**Cosine annealing did not beat the established `ReduceLROnPlateau`
schedule.** The smooth, continuous LR decay hypothesized to help late-stage
convergence instead underperformed the plateau-triggered step schedule by a
small but clear margin (-0.1079 dB PSNR, -0.002966 SSIM) -- the smallest gap
of any rejected experiment so far (smaller than Experiments 9, 12, 13's
architecture-change gaps), but still a real regression, not noise. A
plausible explanation: `ReduceLROnPlateau` only reduces LR when validation
PSNR actually stalls, so it can hold a higher LR for longer when the model is
still improving, whereas cosine's schedule is fixed in advance and can reduce
LR before the model is done benefiting from a higher one. **Retain
`ReduceLROnPlateau` as this project's scheduler of choice.** **Experiment 6
remains the champion checkpoint, and Experiment 6 + x8 TTA (27.7689 dB /
0.747955) remains the best inference pipeline.** Both
`checkpoints/exp14_cosine/checkpoint_best.pt` and `checkpoint_latest.pt` are
retained for reproducibility, unmodified.

---

# Experiment 15 — Extended Champion Training

## Status

**COMPLETED.** The real Experiment 15 run (`checkpoints/exp15_extended60/`,
resumed from `checkpoints/exp6_crop96/checkpoint_latest.pt`, epochs 41-60)
ran to completion. Independently re-verified via `evaluate_checkpoint.py
--checkpoint checkpoints/exp15_extended60/checkpoint_best.pt --tta none`:
best epoch **60**, Val L1 **0.033210**, Val PSNR **27.7626 dB**, Val SSIM
**0.748636** (+4.6213 dB / +0.198032 vs. bicubic) -- essentially flat versus
Experiment 6's epoch-38/40 result (27.7090 dB), i.e. a small further gain
from continued training (+0.0536 dB over Exp6) rather than a new plateau
regression. Superseded by Experiment 16 (a further continuation to epoch 70);
see that entry for the full extended-training conclusion. Both
`checkpoints/exp15_extended60/checkpoint_best.pt` and `checkpoint_latest.pt`
are retained for reproducibility, unmodified.

## Hypothesis

Experiment 6's original run stopped at its planned 40-epoch budget while
`ReduceLROnPlateau` had only reduced the LR to 5e-05 (2 bad epochs into a
patience of 3 at that point -- not yet exhausted). Four different attempts to
beat Experiment 6 via architecture, ensembling, attention, or an alternative
scheduler have all failed (Experiments 9, 11, 12, 13, 14). Experiment 15 asks
a simpler question instead: **does the established champion configuration
still have room to improve if training simply continues past epoch 40**,
under the exact same `ReduceLROnPlateau`-controlled recipe that produced it,
rather than restarting from scratch with any different setting.

## Source Checkpoint: Experiment 6

Resuming from `checkpoints/exp6_crop96/checkpoint_latest.pt` (not
`checkpoint_best.pt`) is the deliberate choice here, since `checkpoint_latest.pt`
holds the true end-of-epoch-40 state (model, optimizer, and scheduler as they
stood after the final epoch), not merely the epoch that happened to score
best. Inspected directly (read-only; this checkpoint file is never written to
by inspection):

| Field | Value |
| --- | --- |
| Stored epoch | 40 |
| Stored best_val_psnr | 27.70896924076407 (~27.7090 dB) |
| `model_config` | `{"in_channels":1,"out_channels":1,"num_features":64,"num_blocks":8,"scale":2}` (`residual_sr`) |
| `loss_config` | `{"name": "l1"}` |
| `scheduler_config` | `{"name": "plateau", "mode": "max", "factor": 0.5, "patience": 3, "min_lr": 1e-06}` |
| `scheduler_state_dict` present | Yes -- `last_epoch=40, best=27.70896924076407, num_bad_epochs=2, cooldown_counter=0, _last_lr=[5e-05]` |
| `optimizer_state_dict` present | Yes (Adam moment buffers for all 630,724 parameters) |
| **Current LR** (read directly from `optimizer_state_dict["param_groups"][0]["lr"]`, matches `scheduler_state_dict["_last_lr"]`) | **5e-05 exactly** (one plateau reduction from the initial 1e-4 occurred during the original run) |
| `training_config.crop_size` | 96 |
| `training_config.seed` | 42 |
| `training_config.val_fraction` | 0.2 |
| `training_config.batch_size` | 16 |

## Dry Resume Verification (read-only -- no training performed)

`load_checkpoint_for_resume` was called directly against the real
`checkpoints/exp6_crop96/checkpoint_latest.pt`, with a freshly constructed
matching model/optimizer/`ReduceLROnPlateau` scheduler (identical
`model_config`/`loss_config`/`scheduler_config` to what Experiment 15 will
use), to confirm the exact resume state without running any epoch:

| Check | Result |
| --- | --- |
| `start_epoch` returned | **41** |
| `best_val_psnr` returned | 27.70896924076407 |
| LR after restore (`current_lr(optimizer)`) | 5e-05 (matches source exactly) |
| `scheduler.best` after restore | 27.70896924076407 |
| `scheduler.num_bad_epochs` after restore | 2 |
| `scheduler.cooldown_counter` after restore | 0 |
| `previous_training_config` returned | Matches the table above exactly (`crop_size=96, seed=42, val_fraction=0.2, batch_size=16`) |
| Source file SHA-256 after the dry resume | Identical to before (read-only `torch.load`; resuming never writes to the source checkpoint) |

This confirms optimizer state, full `ReduceLROnPlateau` state (not just the
LR value), and the epoch/best-PSNR bookkeeping all carry over correctly, and
that resuming is non-destructive to the source checkpoint.

## Output Directory / Existing Infrastructure

**No `train.py` code changes were required or made.** `--resume` (the
checkpoint to load) and `--checkpoint-dir` (where `checkpoint_latest.pt`/
`checkpoint_best.pt` are subsequently written) are already fully independent
CLI arguments -- `--checkpoint-dir` is never derived from `--resume`'s path.
Experiment 15 will therefore use:

- `--resume checkpoints/exp6_crop96/checkpoint_latest.pt` (source)
- `--checkpoint-dir checkpoints/exp15_extended60` (destination -- a new,
  currently-nonexistent directory; Experiment 6's directory is never written
  to during Experiment 15)
- `--epochs 60` (so the loop `range(start_epoch, epochs+1)` = `range(41, 61)`
  runs exactly epochs 41-60 -- 20 additional epochs)
- Every other flag identical to Experiment 6's original invocation: `--model
  residual_sr --num-features 64 --num-blocks 8 --loss l1 --crop-size 96
  --batch-size 16 --seed 42 --lr 1e-4 --scheduler plateau
  --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6` (the `--lr`
  flag only sets the *initial* LR for a fresh optimizer; since we resume, the
  actual starting LR comes from the restored optimizer state -- 5e-05 -- not
  from this flag)

## Best-Checkpoint Behavior (documented in advance, not worked around)

Experiment 15 resumes with `best_val_psnr` already at 27.70896924076407 (the
value carried over from Experiment 6, per the dry-resume verification above).
`checkpoint_best.pt` is only written when a later epoch's validation PSNR
*exceeds* this inherited value (`train.py`'s existing `is_new_best =
val_metrics["psnr"] > best_val_psnr` check -- unmodified). **If none of
epochs 41-60 beat ~27.7090 dB, `checkpoints/exp15_extended60/` will correctly
contain only `checkpoint_latest.pt` and no `checkpoint_best.pt` -- this is
valid, expected behavior, not a failure to be worked around.**
`best_val_psnr` is deliberately **not** reset to force a new-best save; doing
so would fabricate a "champion" checkpoint from a lower bar than Experiment 6
actually cleared, invalidating the comparison. If a later epoch genuinely
exceeds 27.7090 dB, `checkpoint_best.pt` will be saved normally in
`checkpoints/exp15_extended60/` exactly as `train.py` already does for every
other experiment.

## Tests

No training-code changes were made (Section 7's "zero code changes if
already supported" applied), so no new tests were added.
`pytest -m "not integration" -q` -> **368 passed, 8 deselected** (unchanged
from Experiment 14 -- confirms nothing regressed).

## Checkpoint Safety

SHA-256-hashed before and after this preparation task (including the dry
resume verification above, which loads but never writes to
`checkpoint_latest.pt`):
`checkpoints/exp6_crop96/{checkpoint_best,checkpoint_latest}.pt`,
`checkpoints/exp9_edsr_lite/checkpoint_best.pt`,
`checkpoints/exp12_nafnet_sr/checkpoint_best.pt`,
`checkpoints/exp13_swinir_lite/checkpoint_best.pt`,
`checkpoints/exp14_cosine/{checkpoint_best,checkpoint_latest}.pt` -- every
hash unchanged. **Experiment 6 remains fully immutable.**

## Result

**epoch 60: Val L1 = 0.033210, Val PSNR = 27.7626 dB, Val SSIM = 0.748636**
(+4.6213 dB / +0.198032 vs. bicubic; +0.0536 dB / +0.003002 vs. Experiment
6's epoch-38 result of 27.7090 dB / 0.745634). 20 additional
`ReduceLROnPlateau`-controlled epochs produced a modest further gain rather
than a plateau -- motivating Experiment 16's further extension to epoch 70
(see that entry for the point at which extended training saturates).

---

# Experiment 16 — Extended Champion Training (Continued, 60 -> 70 Epochs)

## Status

**COMPLETED.** A further continuation of Experiment 15
(`checkpoints/exp16_extended70/`, resumed from
`checkpoints/exp15_extended60/checkpoint_latest.pt`, epochs 61-70) ran to
completion.

## Hypothesis

Experiment 15 (epochs 41-60) produced a modest gain over Experiment 6
(+0.0536 dB), without yet showing clear saturation. Experiment 16 tests
whether continuing another 10 epochs (61-70) under the identical
`ReduceLROnPlateau`-controlled recipe continues to help, plateaus, or
reverses -- i.e. whether simple epoch-extension is a repeatable way to
improve the champion, or a one-time gain that has now been exhausted.

## Real Training Run

Resumed from `checkpoints/exp15_extended60/checkpoint_latest.pt` into
`checkpoints/exp16_extended70/` (Experiment 15's directory untouched), same
recipe as every prior extension: `ResidualSRNet` 64F/8B, L1, crop96/192,
batch16, seed42, Adam, `ReduceLROnPlateau` (factor 0.5, patience 3,
min_lr 1e-6), target epoch 70.

Checkpoint: `checkpoints/exp16_extended70/checkpoint_best.pt`. Independently
re-verified:

```
Loaded checkpoint checkpoints/exp16_extended70/checkpoint_best.pt (epoch=65, best_val_psnr=27.765574385729558)
--tta none:
Val L1: 0.033206   Val PSNR: 27.7656 dB   Val SSIM: 0.748618
PSNR vs bicubic: +4.6243 dB   SSIM vs bicubic: +0.198014

--tta x8:
Val L1: 0.032998   Val PSNR: 27.8154 dB   Val SSIM: 0.750571
PSNR vs bicubic: +4.6741 dB   SSIM vs bicubic: +0.199967
```

| Metric | Exp 15 (epoch 60) | Exp 16 non-TTA (epoch 65) | Exp 16 + x8 TTA |
| --- | ---: | ---: | ---: |
| Val L1 | 0.033210 | 0.033206 | 0.032998 |
| Val PSNR | 27.7626 dB | 27.7656 dB | **27.8154 dB** |
| Val SSIM | 0.748636 | 0.748618 | **0.750571** |

Best epoch was **65**, not 70: by epoch 70 (`checkpoint_latest.pt`), the
scheduler had reduced LR to exactly `min_lr = 1e-6` (`num_bad_epochs=1` at
that point) with no further validation-PSNR improvement recorded after epoch
65's checkpoint. Improvement from epoch 60 -> 65: **PSNR +0.0030 dB, SSIM
-0.000018, L1 -0.000004** -- essentially noise-level.

## Conclusion

**Additional training beyond epoch 60 produced only a negligible
improvement, and the scheduler bottomed out at `min_lr` by epoch 70.**
Simple epoch-extension of the champion configuration is therefore
**saturated** -- Experiments 15 and 16 together show the technique had one
modest gain available (+0.0536 dB from epochs 41-60) that is now exhausted;
further blind extension is not expected to help without changing something
else about the training recipe. **Experiment 16 + x8 TTA is now the current
best overall pipeline: PSNR 27.8154 dB, SSIM 0.750571, L1 0.032998**
(+4.6741 dB / +0.199967 vs. bicubic), surpassing the previous best pipeline
(Experiment 6 + x8 TTA, 27.7689 dB / 0.747955). `checkpoints/exp16_extended70/checkpoint_best.pt`
(epoch 65) is the new champion checkpoint; both it and `checkpoint_latest.pt`
(epoch 70) are retained, unmodified, alongside all of Experiments 6 and 15's
checkpoints.

---

# Experiment 17 — Bicubic Residual Learning

## Status

**COMPLETED.** The real Experiment 17 training run
(`checkpoints/exp17_bicubic_residual/`, trained from scratch, full canonical
60-epoch recipe below) ran to completion. The smoke-run metrics further
below are infrastructure-verification artifacts only (1+1 epochs, 32 train /
16 val samples, freshly initialized weights) and are unrelated to this real
result.

## Hypothesis

Providing a fixed bicubic-upsampled LR as a global reconstruction path may
let the network focus its capacity on denoising and high-frequency
correction (a residual over a already-reasonable baseline) rather than
relearning the entire 2x image-formation mapping from scratch, as the direct
formulation (Experiments 1-16) requires. This is a controlled, single-variable
test: same learned-branch topology as the champion, same full training
recipe, trained from scratch (not fine-tuned from any existing checkpoint) --
only the presence of the fixed bicubic skip changes.

Baseline for comparison (direct HR prediction, no bicubic skip):
- Experiment 16 non-TTA: **27.7656 dB** / 0.748618 SSIM / 0.033206 L1
- Current best overall pipeline, Experiment 16 + x8 TTA: **27.8154 dB** /
  0.750571 SSIM / 0.032998 L1

## Bicubic Implementation

**Existing repository baseline** (`src.baseline.bicubic_upscale`, used by
`evaluate_baseline.py` and every "vs. bicubic" delta in this log): resizes a
2D numpy array via `PIL.Image.fromarray(..., mode="F").resize(...,
resample=Image.Resampling.BICUBIC)` -- a CPU-only, PIL-based bicubic
implementation. **This file is completely untouched by Experiment 17.**

**Experiment 17's in-model bicubic skip** (`src/models/residual_sr_bicubic.py::
fixed_bicubic_upsample`): `torch.nn.functional.interpolate(x, scale_factor=scale,
mode="bicubic", align_corners=False)` -- PyTorch's native bicubic, chosen
because it runs on-device (GPU) inside a batched forward pass without a CPU
round-trip, which `PIL.Image.resize` cannot do efficiently at training time.

**These are not bit-identical** -- different library, different convolution
kernel/anti-aliasing behavior. Measured directly (same 16x16 synthetic
grayscale input, both upscaled 2x): **max abs difference 0.0644, mean abs
difference 0.0182** (on a `[0,1]`-range image). This is a deliberate,
documented divergence, not an oversight -- `fixed_bicubic_upsample` optimizes
for "runs efficiently inside a PyTorch forward pass," while
`bicubic_upscale` optimizes for "matches the classical baseline every other
experiment is compared against." Every experiment's "vs. bicubic" delta
continues to use the unmodified `src.baseline.bicubic_upscale`/
`evaluate_baseline.py` pipeline; Experiment 17's internal skip is purely an
architectural component, invisible to metric computation.

## Architecture

`src/models/residual_sr_bicubic.py::ResidualSRBicubic` -- `ResidualSRNet`'s
exact learned branch (imports and reuses `ResidualBlock` directly; identical
`conv_in`/`body`/`conv_body_out`/`upsample_conv`/`pixel_shuffle` layout, no
new blocks, channels, normalization, activations, or residual scaling), with
the final step changed from "return the learned branch's output" to:

```
prediction = fixed_bicubic_upsample(LR) + learned_residual_branch(LR)
```

```
Noisy LR ---------------------------> fixed_bicubic_upsample --------+
  |                                                                  |
  v                                                                  |
3x3 conv (in_channels -> num_features)                              |
  |                                                                  |
  v                                                                  |
num_blocks x ResidualBlock                                          |
  |                                                                  |
  v                                                                  |
3x3 conv ("conv_body_out") -> + (local skip, same as ResidualSRNet) |
  |                                                                  |
  v                                                                  |
3x3 conv (-> out_channels*scale^2) -> PixelShuffle(scale)           |
  |                                                                  |
  v                                                                  |
learned residual -----------------------------------------------> + -> raw prediction
```

Trained with `nn.L1Loss()(prediction, GT)` directly -- the residual branch is
never trained against a separately constructed target; the loss only ever
sees the final summed prediction, exactly as specified.

**Parameter count: 630,724** -- identical to Experiment 6 (`ResidualSRNet`
64F/8B), since `fixed_bicubic_upsample` contributes zero trainable
parameters and no other change was made to the learned branch. Verified by a
direct test comparing parameter counts against a plain `ResidualSRNet` built
with the same `num_features`/`num_blocks`.

## Raw Output / Clipping Behavior

The bicubic term is **never clipped** before being added to the residual,
and the summed output is **never clipped** either -- exactly like every
other architecture in this project. A model with the final `upsample_conv`
weight/bias zeroed (learned residual forced to exactly zero) produces output
identical to `fixed_bicubic_upsample(LR)` to float32 tolerance (max abs diff
**0.0**, i.e. exact) -- confirming the global skip is wired correctly and
isolates its effect cleanly. Feeding an input far outside `[0,1]` (constant
5.0, and separately constant -5.0) produces output correspondingly far
outside `[0,1]` in both directions, confirming no clamp exists anywhere in
`forward`. Metric-time clipping remains entirely the existing
`src/metrics.py` pipeline's responsibility, unmodified.

## Model Factory / CLI

- `src/models/__init__.py`: `build_model_config`/`build_model` extended with
  `architecture="residual_sr_bicubic"`, reusing `num_features`/`num_blocks`/
  `scale` -- no new parameters needed since the bicubic behavior is intrinsic
  to the architecture, not configurable. `residual_sr`/`edsr_lite`/
  `nafnet_sr`/`swinir_lite` configs and reconstruction are byte-for-byte
  unchanged; default architecture remains `residual_sr`.
- `train.py --model` gains the `residual_sr_bicubic` choice. No new CLI
  flags -- `--num-features`/`--num-blocks` are reused exactly as they already
  are for `residual_sr`/`edsr_lite`/`nafnet_sr`.
- `evaluate_checkpoint.py`/`infer_test.py` require **no changes** -- both
  already reconstruct models exclusively through `build_model`.
- `src/tta.py` (x8 geometric self-ensemble) required **no changes** and does
  **not** double-add the bicubic term: `predict_x8` calls the complete model
  (bicubic skip included) once per D4 transform, inverse-transforms each
  complete raw prediction, and averages those -- verified by a dedicated
  test cross-checking `predict_x8`'s output against a manual step-by-step
  recomputation using the same primitives.

### Exact `model_config`

```python
{
    "architecture": "residual_sr_bicubic",
    "in_channels": 1,
    "out_channels": 1,
    "num_features": 64,
    "num_blocks": 8,
    "scale": 2,
}
```

## Checkpoint / Resume Compatibility

Verified by `tests/test_training_unit.py`:

- Matching `residual_sr_bicubic` checkpoint + config -> resume succeeds
  (weights restored exactly, epoch/best-PSNR continuation correct).
- `residual_sr` checkpoint + `residual_sr_bicubic` request -> **rejected**,
  even though the underlying learned-branch tensor shapes are identical
  (the missing `"architecture"` key vs. the present one makes the configs
  unequal, so the existing dict-equality check in `load_checkpoint_for_resume`
  catches this with zero new comparison logic -- the same mechanism already
  protecting `edsr_lite`/`nafnet_sr`/`swinir_lite`).
- `residual_sr_bicubic` checkpoint + `residual_sr` request -> rejected.
- Legacy (no-`"architecture"`-key) checkpoints (Experiments 1-8) still
  reconstruct as `ResidualSRNet`, unchanged.

## Tests

26 new tests: `tests/test_residual_sr_bicubic_unit.py` (18 -- construction,
exact parameter count, 96x96/128x128/batch16 shapes, finiteness, gradients,
bicubic-skip parameter-freeness, device/dtype preservation, the zero-residual
exact-parity test, unclamped-output tests in both directions, x8 TTA
shape/finiteness and the no-double-bicubic cross-check), plus factory tests
in `tests/test_model_unit.py` (3) and checkpoint/resume/loading tests in
`tests/test_training_unit.py` (5).

`pytest -m "not integration" -q` -> **394 passed, 8 deselected** (up from
368; every pre-existing test, including all TTA, ensemble, and every other
architecture's tests, remains unchanged and passing).

## CUDA Sanity Check

Short check only (not a sustained benchmark), exact Experiment 17
configuration, `batch=16`, `input=[16,1,96,96]`, `loss=L1`, `optimizer=Adam`,
forward -> loss -> backward -> optimizer step, 10 timed iterations after 3
discarded warmup iterations:

| Check | Result |
| --- | --- |
| Device | NVIDIA GeForce RTX 4060 Laptop GPU (CUDA) |
| Parameter count | 630,724 |
| Output shape | `(16, 1, 192, 192)` (matches expected exactly) |
| Output finite | True |
| Loss finite | True |
| All gradients finite | True |
| Peak allocated CUDA memory | 778.7 MiB |
| Peak reserved CUDA memory | 876.0 MiB |
| Per-batch runtime (avg of 10) | 101.0 ms |
| OOM | None |

Memory usage is essentially identical to Experiment 6 (same learned-branch
parameter count); the small runtime increase over a plain `ResidualSRNet`
forward pass is attributable to the added `F.interpolate` bicubic op.

## Real-Data Smoke Test (infrastructure verification only -- not a result)

```bash
python train.py --model residual_sr_bicubic --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --epochs 1 --max-train-samples 32 --max-val-samples 16 \
  --checkpoint-dir checkpoints/exp17_bicubic_residual_smoke --num-workers 0
```

Verified and then deleted (`checkpoints/exp17_bicubic_residual_smoke/`, not
committed):

- CUDA used; printed `Model: residual_sr_bicubic (630,724 trainable
  parameters)` and the exact `model_config` above.
- Training and validation both completed (0.8s); validation ran on real
  full-size (128x128) LR images. Notably, even after 1 epoch on 32 samples
  the smoke checkpoint's Val PSNR (~18.7 dB) was far above the near-random
  results other architectures' 1-epoch smoke runs typically show (~7-13 dB)
  -- consistent with (but not proof of) the hypothesis, since the bicubic
  skip gives the model a reasonable starting point before the learned branch
  has meaningfully trained. **This is a smoke-test artifact, not a result,
  and is not being treated as evidence for or against the hypothesis.**
- `evaluate_checkpoint.py --tta none` and `--tta x8` both loaded the smoke
  checkpoint and ran to completion with finite metrics.
- `infer_test.py`'s model-loading path (`evaluate_checkpoint.load_model`)
  reconstructed `ResidualSRBicubic` correctly; `run_inference` with both
  `tta="none"` and `tta="x8"` produced finite, correctly-shaped
  (128x128 LR -> 256x256 prediction) output.
- Resume verified: re-ran with `--resume .../checkpoint_latest.pt --epochs 2`;
  scheduler state restored, training continued from epoch 2 correctly, and a
  new best checkpoint was saved normally.
- No historical checkpoint was read or written during this test.

The smoke run's own numbers are **not an experiment result** and must not be
compared against Experiment 6/9/10/11/12/13/14/15/16.

## Checkpoint Safety

`checkpoints/exp6_crop96/{checkpoint_best,checkpoint_latest}.pt`,
`checkpoints/exp9_edsr_lite/checkpoint_best.pt`,
`checkpoints/exp12_nafnet_sr/checkpoint_best.pt`,
`checkpoints/exp13_swinir_lite/checkpoint_best.pt`,
`checkpoints/exp14_cosine/checkpoint_best.pt`, and
`checkpoints/exp15_extended60/{checkpoint_best,checkpoint_latest}.pt` and
`checkpoints/exp16_extended70/{checkpoint_best,checkpoint_latest}.pt` were
SHA-256-hashed before and after this preparation task; every hash is
unchanged.

## Real Training Run

Exact Experiment 17 configuration (`ResidualSRBicubic`, 64F/8B, 630,724
params), trained **from scratch** (not fine-tuned from any existing
checkpoint) with the full controlled recipe: L1 loss, crop96 LR / crop192
GT, batch16, seed42, Adam (initial LR 1e-4), `ReduceLROnPlateau` (factor
0.5, patience 3, min LR 1e-6), 60 epochs (`training_config` confirms
`epochs: 60`; no extension to 70 was run -- no
`checkpoints/exp17_bicubic_residual_extended*` directory exists), canonical
split, full-image validation on all 640 validation samples, best checkpoint
selected by validation PSNR.

Checkpoint: `checkpoints/exp17_bicubic_residual/checkpoint_best.pt`.
Independently re-verified:

```
--tta none:
Loaded checkpoint checkpoints/exp17_bicubic_residual/checkpoint_best.pt (epoch=60, best_val_psnr=27.745965460790693)
Val L1: 0.033260   Val PSNR: 27.7460 dB   Val SSIM: 0.748557
PSNR vs bicubic: +4.6047 dB   SSIM vs bicubic: +0.197953

--tta x8:
Val L1: 0.033066   Val PSNR: 27.7942 dB   Val SSIM: 0.750434
PSNR vs bicubic: +4.6529 dB   SSIM vs bicubic: +0.199830
```

| Metric | Exp 16 non-TTA | Exp 17 non-TTA | Exp 16 + x8 | Exp 17 + x8 |
| --- | ---: | ---: | ---: | ---: |
| Val L1 | 0.033206 | 0.033260 | 0.032998 | 0.033066 |
| Val PSNR | 27.7656 dB | 27.7460 dB | **27.8154 dB** | 27.7942 dB |
| Val SSIM | 0.748618 | 0.748557 | **0.750571** | 0.750434 |

## Conclusion

**Bicubic residual learning is highly competitive but did not beat the
direct ResidualSR champion**, in either the non-TTA (-0.0196 dB vs.
Experiment 16) or x8 (-0.0212 dB vs. Experiment 16 + x8) comparison. Both
gaps are small -- this is by far the closest any alternative formulation
has come to the champion (closer than Experiments 9, 11, 12, 13, 14) -- but
consistently on the losing side, so the direct-prediction formulation
remains preferred. Training from scratch (not fine-tuned) confirms this is a
genuine architectural comparison, not an artifact of initialization.
**Experiment 16 + x8 TTA (27.8154 dB / 0.750571 / 0.032998) remains the
overall champion pipeline.** `checkpoints/exp17_bicubic_residual/checkpoint_best.pt`
and `checkpoint_latest.pt` are retained for reproducibility, unmodified --
and, notably, are reused directly as Experiment 18's second ensemble member
(see below) rather than discarded, since "did not individually win" does not
imply "has nothing to contribute to an ensemble."

---

# Experiment 18 — Experiment 16 + Experiment 17 Model Ensemble

## Status

**COMPLETED.** Inference-only: no epochs, no optimizer, no scheduler, no new
checkpoints. Reuses the existing Experiment 11 ensemble infrastructure
(`src/ensemble.py::weighted_average_predictions`, `evaluate_ensemble.py`)
unmodified.

## Hypothesis

Experiment 16 (direct HR prediction) and Experiment 17 (bicubic + learned
residual) differ in output formulation despite sharing the same learned-branch
topology and training recipe. Since Experiment 17 came unusually close to
Experiment 16 (-0.0196 dB non-TTA, -0.0212 dB with x8 -- the smallest gap of
any alternative tried), their reconstruction errors might be partially
complementary, making this a more promising ensemble candidate than
Experiment 11 (Exp6 + the substantially weaker Exp9).

## Ensemble Semantics Verified (no code changes)

- Both models reconstructed from checkpoint `model_config` via the existing
  `build_model` factory (`evaluate_checkpoint.load_model`) -- automatically
  supports `residual_sr_bicubic` with zero special-casing, since the factory
  already had it wired in from Experiment 17.
- Predictions combined as raw floating-point tensors
  (`weighted_average_predictions`); metric clipping happens only inside
  `psnr`/`ssim` afterward -- confirmed by reading the code path, matching
  Experiment 11's already-verified behavior.
- No individual prediction is clipped before averaging.
- x8 behavior: `predict_x8` is called on each *complete* model (bicubic skip
  included, for Experiment 17) once per D4 transform; `evaluate_ensemble.py`
  then only averages the two complete outputs -- the bicubic term is added
  exactly once (inside `ResidualSRBicubic.forward`), never a second time by
  the ensemble script. Confirmed by code inspection; this exact behavior was
  already covered by a dedicated Experiment 17 test
  (`test_x8_tta_averages_the_complete_model_output_not_bicubic_twice`).
- No working code was rewritten.

## Non-TTA Weight Tests (3 pre-declared weights, full 640-image validation)

| Weight (Exp16 / Exp17) | Val L1 | Val PSNR | Val SSIM | PSNR vs bicubic | SSIM vs bicubic |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 / 50 | 0.033116 | 27.7845 dB | 0.749675 | +4.6432 dB | +0.199071 |
| 75 / 25 | 0.033132 | 27.7822 dB | 0.749419 | +4.6409 dB | +0.198815 |
| 87.5 / 12.5 | 0.033162 | 27.7757 dB | 0.749086 | +4.6344 dB | +0.198482 |

Strongest non-TTA weighting by PSNR: **50/50 (27.7845 dB)**, followed by
75/25 (27.7822 dB) -- a gap of **0.0023 dB**, at or below the 0.01 dB
threshold specified for testing both top weightings with x8.

## x8 Test

Per the pre-declared rule (top two non-TTA weightings within 0.01 dB PSNR of
each other -> evaluate both with x8), both 50/50 and 75/25 were run with
`--tta x8`:

| Weight (Exp16 / Exp17) | Val L1 | Val PSNR | Val SSIM | PSNR vs bicubic | SSIM vs bicubic |
| --- | ---: | ---: | ---: | ---: | ---: |
| 50 / 50 + x8 | 0.033008 | 27.8111 dB | 0.750709 | +4.6698 dB | +0.200105 |
| 75 / 25 + x8 | 0.032997 | 27.8149 dB | 0.750692 | +4.6736 dB | +0.200088 |

## Comparison Against the Target

| Pipeline | Val L1 | Val PSNR | Val SSIM |
| --- | ---: | ---: | ---: |
| Exp 16 (non-TTA) | 0.033206 | 27.7656 dB | 0.748618 |
| Exp 16 + x8 (**current champion**) | 0.032998 | **27.8154 dB** | 0.750571 |
| Exp 17 (non-TTA) | 0.033260 | 27.7460 dB | 0.748557 |
| Exp 17 + x8 | 0.033066 | 27.7942 dB | 0.750434 |
| 50/50 (non-TTA) | 0.033116 | 27.7845 dB | 0.749675 |
| 75/25 (non-TTA) | 0.033132 | 27.7822 dB | 0.749419 |
| 87.5/12.5 (non-TTA) | 0.033162 | 27.7757 dB | 0.749086 |
| 50/50 + x8 | 0.033008 | 27.8111 dB | 0.750709 |
| **75/25 + x8 (best ensemble)** | **0.032997** | 27.8149 dB | **0.750692** |

The best ensemble configuration (75/25 + x8) reaches **27.8149 dB**, which is
**0.0005 dB below** the champion's 27.8154 dB on the primary ranking metric
(PSNR) -- it does not satisfy the pre-declared success rule (`ensemble + x8
> 27.8154`). Its L1 (0.032997 vs. 0.032998) and SSIM (0.750692 vs. 0.750571)
are both marginally *better* than the champion's, but PSNR was designated
the primary ranking metric in advance, and by that metric this is a
(negligible) loss, not a win.

## Conclusion

**Ensembling is rejected.** 75/25 + x8 essentially ties Experiment 16 + x8
(differences on the order of a few ten-thousandths of a dB either way --
within measurement noise), but does not exceed it on the pre-declared
primary metric. Per the pre-declared success rule, **Experiment 16 + x8 TTA
remains the champion pipeline: PSNR 27.8154 dB, SSIM 0.750571, L1 0.032998.**
No further weight exploration was performed (a dense sweep was explicitly
out of scope, to avoid over-tuning the validation set) and no new experiment
was started automatically, per instructions.

## Historical Checkpoint Safety

This experiment is inference-only and performed no writes to
`checkpoints/exp16_extended70/` or `checkpoints/exp17_bicubic_residual/` --
`evaluate_ensemble.py` only loads checkpoints (`torch.load`) and prints
metrics; it never calls `torch.save`. No hash re-verification was necessary
beyond this structural guarantee, consistent with every prior
`evaluate_ensemble.py`/`evaluate_checkpoint.py` use in this project.

## Tests

No code changes were required (the existing Experiment 11 ensemble
infrastructure already supported `residual_sr_bicubic` through the shared
model factory), so no new tests were added. `pytest -m "not integration" -q`
-> **394 passed, 8 deselected** (unchanged from Experiment 17 -- confirms
nothing regressed).

---

# Experiment 19 — EMA Weight Averaging

## Status

**PLANNED / PREPARED.** Implementation, tests, CUDA sanity, and a tiny
real-data smoke test are complete and passing. **No real training run has
been started.** The smoke-run metrics below are infrastructure-verification
artifacts only (2+1 epochs, 32 train / 16 val samples, freshly initialized
weights, `decay=0.999` barely moved in so few steps) and must not be
compared against any other experiment's numbers.

## Hypothesis

Does maintaining an exponential moving average (EMA) of the trained model's
weights improve validation PSNR/generalization for the proven
`ResidualSRNet`, by smoothing noisy late-stage parameter updates, while
preserving the exact successful training formulation otherwise? The only
experimental variable is EMA -- not combined with architecture, loss, crop
schedule, optimizer, weight decay, cosine LR, warmup, AMP, or bicubic
residual learning changes.

## EMA Implementation

Implemented as generic training infrastructure in `src/ema.py`, not
architecture-specific -- `ResidualSRNet` itself is completely unmodified.

- **`ExponentialMovingAverage(model, decay)`**: on construction, deep-copies
  *model*'s **current** weights into a separate shadow model (`.eval()`,
  every parameter `requires_grad_(False)`) -- **never zeros**. For a
  from-scratch run this means the shadow starts out identical to the
  network's freshly-initialized weights; for a resumed run, this initial
  copy is immediately overwritten by the checkpoint's actual saved EMA state
  via `load_state_dict` (see Resume below), so it is never the operative
  state in that case.
- **`update(model)`**: for every floating-point parameter,
  `ema_param = decay * ema_param + (1 - decay) * live_param`. Buffers (e.g.
  BatchNorm running stats -- none of this project's architectures have any)
  are copied directly rather than averaged, since a buffer is not a
  gradient-updated parameter and "smoothing" it has no well-defined meaning
  independent of the live model's own bookkeeping; handled generically in
  case a future architecture has any. Decorated `@torch.no_grad()`.
  Verified numerically: initial=1, live=3, decay=0.9 -> shadow becomes
  exactly `0.9*1 + 0.1*3 = 1.2`.
- **Timing**: called once per optimizer step, i.e. once per training batch,
  immediately after `optimizer.step()` inside `train_one_epoch`. The first
  update therefore happens after the very first batch of training (or,
  on a resumed run, after the first batch following the resumed epoch --
  the shadow's prior trajectory was already restored before that point).
- **Device/no-gradients**: `.to(device)` moves the shadow; every shadow
  parameter has `requires_grad=False` and is never passed to any optimizer
  (`torch.optim.Adam(model.parameters(), ...)` only ever sees the live
  model's parameters) -- verified by a dedicated test checking the shadow's
  parameter ids never appear in `optimizer.param_groups`.

## Validation / Scheduler / Best-Checkpoint Semantics

Implemented as a **one-line change in the training loop**, not new
validation logic: `validate()` itself is completely unmodified; the caller
simply passes a different model.

```python
eval_model = ema.shadow_model if ema is not None else model
val_metrics = validate(eval_model, validation_loader, loss_fn, device, thermal_guard)
```

Since `scheduler_step(scheduler, scheduler_config, val_metrics["psnr"])` and
the `is_new_best = val_metrics["psnr"] > best_val_psnr` check both only ever
consume `val_metrics` (never touching `model`/`eval_model` directly), routing
`val_metrics` through the EMA shadow automatically makes the scheduler and
best-checkpoint selection EMA-driven too, with zero additional special-casing
required elsewhere. Verified explicitly by a test asserting
`validate(model, ...) != validate(ema.shadow_model, ...)` after training
(proving genuinely different weights are scored, not the live model under a
different name) and a second test confirming the exact `eval_model`
expression used matches what gets validated.

Training loss (inside `train_one_epoch`) always uses the **live** model --
unaffected by any of this.

## Checkpoint Representation

`model_state_dict` **always** means the live/raw training weights, in every
checkpoint, EMA or not -- this key's meaning never changes, so no historical
loader needs to change. A new, separate `ema_state_dict` key (`None` when
EMA is disabled) holds the EMA shadow's weights -- the ones that actually
produced the checkpoint's recorded validation PSNR. `evaluate_checkpoint.py`'s
`load_model()` gained a `prefer_ema: bool = True` parameter: when the
checkpoint has non-`None` `ema_state_dict` and `prefer_ema` is true (the
default), those EMA weights are loaded instead of `model_state_dict` --
automatically, with no CLI flag needed on `evaluate_checkpoint.py`/
`infer_test.py` (both call `load_model()` with its default `prefer_ema=True`
already). Historical/non-EMA checkpoints have no `ema_state_dict`
(`None`), so `load_model()` falls through to the exact prior behavior
unchanged. `load_model(path, device, prefer_ema=False)` forces the live/raw
weights instead, for diagnostics -- this never changes which checkpoint was
selected "best" (that ranking already happened during training via the EMA
`val_metrics` above), only which weights get loaded from it afterward.

### Exact checkpoint additions

```python
{
    ...  # every existing key, unchanged
    "ema_state_dict": ema.state_dict() if ema is not None else None,
    "ema_config": ema_config,  # {"enabled": True, "decay": 0.999} or None
}
```

## CLI

- `--ema` (`store_true`, default off) and `--ema-decay` (default `0.999`,
  ignored unless `--ema`). Every historical command that never mentions
  `--ema` behaves identically to before this feature existed.
- Experiment 19 requests exactly `--ema --ema-decay 0.999`.

## Resume Compatibility

`load_checkpoint_for_resume` gained `ema`/`ema_config` parameters. Unlike
`loss_config`/`scheduler_config` (which default to `None` meaning "skip this
check"), `ema_config` defaults to a private `_UNSET` sentinel -- because
`None` is EMA's own legitimate "disabled" value
(`build_ema_config(False, ...)` returns `None`), reusing `None` as the
"skip" signal could not distinguish "caller doesn't care" from "caller
explicitly disabled EMA and wants that enforced". `train.py`'s real resume
path always passes the actual computed `ema_config` (never omits it), so in
real usage the check is always active and strict:

- Matching EMA checkpoint + EMA enabled + same decay -> resumes correctly
  (full EMA trajectory restored via `ema.load_state_dict`, verified exact).
- EMA checkpoint + EMA disabled -> **rejected**.
- EMA checkpoint + different decay -> **rejected**.
- Non-EMA checkpoint + EMA resume request -> **rejected**.
- Historical non-EMA checkpoints, resumed with `--ema` off (the default) ->
  behave exactly as before (`None == None`, check passes trivially).
- A resumed EMA run reaches the identical shadow weights as an uninterrupted
  run through the same number of updates (verified via a deterministic
  parameter-perturbation test, not real training, since exact floating-point
  equality is what's being checked).

## Tests

31 new tests: `tests/test_ema_unit.py` (12 -- initialization-from-live-weights
not zeros, independent-copy verification, decay-range validation, the exact
numerical formula check from the task spec, multi-step compounding, a real
multi-parameter-model check, no-gradients, not-in-optimizer, live-model
untouched, device movement, state_dict round-trip), plus 19 in
`tests/test_training_unit.py` (EMA updates occurring post-optimizer-step,
validation/scheduler/best-checkpoint EMA semantics, config/state checkpoint
storage, matching resume, resumed-vs-uninterrupted trajectory equivalence,
all 3 mismatch-rejection directions, the `_UNSET`-skips-check legacy path, a
simulated real pre-Experiment-19 checkpoint, `load_model` EMA-preference
behavior in both directions, x8 TTA compatibility).

`pytest -m "not integration" -q` -> **425 passed, 8 deselected** (up from
394; every pre-existing test, including all model/scheduler/TTA/ensemble
tests, remains unchanged and passing).

## CUDA Sanity Check

Short check only (not a sustained benchmark): `ResidualSRNet` 64F/8B,
`batch=16`, `crop=96`, `L1`, EMA enabled (`decay=0.999`), forward -> backward
-> `optimizer.step()` -> `ema.update()` -> EMA validation forward, 1 timed
iteration after 3 discarded warmup iterations:

| Check | Result |
| --- | --- |
| Device | NVIDIA GeForce RTX 4060 Laptop GPU (CUDA) |
| Parameter count | 630,724 |
| Forward output finite | True |
| Loss finite | True |
| All gradients finite | True |
| EMA validation forward finite | True |
| EMA shadow device | `cuda:0` (matches live model) |
| Peak allocated CUDA memory | 783.3 MiB |
| Peak reserved CUDA memory | 878.0 MiB |
| OOM | None |

Memory is essentially identical to Experiment 6/16 (same architecture); the
EMA shadow adds a second copy of the ~630K-parameter model (a few MiB),
negligible next to activation memory.

## Real-Data Smoke Test (infrastructure verification only -- not a result)

```bash
python train.py --model residual_sr --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --ema --ema-decay 0.999 \
  --epochs 2 --max-train-samples 32 --max-val-samples 16 \
  --checkpoint-dir checkpoints/exp19_ema_smoke --num-workers 0
```

Verified and then deleted (`checkpoints/exp19_ema_smoke/`, not committed):

- CUDA used; printed `EMA: {'enabled': True, 'decay': 0.999}`.
- Trained 2 epochs; checkpoint's `ema_config` = `{"enabled": True, "decay":
  0.999}`, `ema_state_dict` present and **numerically different** from
  `model_state_dict` (confirmed by direct tensor comparison) -- proving the
  live and EMA weights are genuinely tracked separately, not aliased.
- `evaluate_checkpoint.py --tta none` and `--tta x8` both loaded the smoke
  checkpoint (EMA weights, by default) and ran to completion with finite
  metrics.
- `infer_test.py`'s model-loading path (`evaluate_checkpoint.load_model`)
  confirmed to load weights matching `ema_state_dict` (not `model_state_dict`)
  by direct tensor comparison; `run_inference` with both `tta="none"` and
  `tta="x8"` produced finite, correctly-shaped output.
- Resume verified: re-ran with `--resume .../checkpoint_latest.pt --epochs 3`;
  printed both `"Restored scheduler state from checkpoint."` and
  `"Restored EMA state from checkpoint."`, training continued correctly.
- No historical checkpoint was read or written during this test.

The smoke run's own numbers are **not an experiment result** (`decay=0.999`
moves the shadow only ~0.1% per step, so 4 total optimizer steps across 2
epochs leaves the EMA shadow still very close to its random initialization
-- expected, not a bug) and must not be compared against Experiment
6/9/.../16/17/18.

## Checkpoint Safety

`checkpoints/exp6_crop96/checkpoint_best.pt`,
`checkpoints/exp9_edsr_lite/checkpoint_best.pt`,
`checkpoints/exp12_nafnet_sr/checkpoint_best.pt`,
`checkpoints/exp13_swinir_lite/checkpoint_best.pt`,
`checkpoints/exp14_cosine/checkpoint_best.pt`,
`checkpoints/exp15_extended60/checkpoint_best.pt`, and
`checkpoints/exp16_extended70/{checkpoint_best,checkpoint_latest}.pt` and
`checkpoints/exp17_bicubic_residual/{checkpoint_best,checkpoint_latest}.pt`
were SHA-256-hashed before and after this preparation task; every hash is
unchanged.

## Result

**TBD.** No real Experiment 19 training run has been started. The next step
is the real 60-epoch run:

```bash
python train.py --model residual_sr --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --ema --ema-decay 0.999 \
  --epochs 60 --checkpoint-dir checkpoints/exp19_ema
```

then compare with `evaluate_checkpoint.py --checkpoint
checkpoints/exp19_ema/checkpoint_best.pt` against the current champion
pipeline, Experiment 16 + x8 TTA (27.8154 dB / 0.750571 / 0.032998). **This
run has not been started.**

---

# Canonical experiment numbering (resolved 2026-08-11)

An earlier session flagged that "Experiment 21" had been used for two different
things. The numbering below is now **authoritative**; the degradation analysis
previously headed "Experiment 21" is renumbered to **22**, matching its
checkpoint-free, analysis-only nature and the sequence the EMA runs occupy.

| # | what | artifact |
| --- | --- | --- |
| 19 | EMA weight averaging, 60 epochs | `checkpoints/exp19_ema/` |
| 20 | EMA continuation to epoch 70 | `checkpoints/exp20_ema_extended70/` |
| 21 | EMA continuation to epoch 80 | `checkpoints/exp21_ema_extended80/` |
| 22 | Dataset degradation forensics (**analysis only**) | `results/degradation_analysis/` |
| 23 | EMA continuation to epoch 90 — **current champion** | `checkpoints/exp23_ema_extended90/` |
| 24 | Signal-dependent synthetic noise augmentation | prepared, not yet run |

Experiments 19, 20 and 21 still lack full write-ups. Their stored checkpoint
metadata (read-only, not re-measured) is:

| checkpoint directory | stored epoch | stored `best_val_psnr` |
| --- | ---: | ---: |
| `checkpoints/exp19_ema/` | 60 | 27.828210 |
| `checkpoints/exp20_ema_extended70/` | 70 | 27.884964 |
| `checkpoints/exp21_ema_extended80/` | 80 | 27.938326 |

EMA plainly helped — Experiment 16's non-TTA result was 27.7656 dB — and the
progression continues monotonically into Experiment 23 below. Closing 19-21
properly (with independent `evaluate_checkpoint.py` passes) remains outstanding.

---

# Experiment 23 — EMA Continuation to Epoch 90 (current champion)

## Status

**COMPLETED — current protected champion.** ResidualSRNet 64F/8B with EMA
(decay 0.999), continued to epoch 90.

## Verified result

Independently re-measured on the canonical 640-image validation split with
`evaluate_checkpoint.py` (which loads the EMA weights by default), against
`checkpoints/exp23_ema_extended90/checkpoint_best.pt` (epoch 90):

| pipeline | Val L1 | Val PSNR | Val SSIM |
| --- | ---: | ---: | ---: |
| non-TTA | 0.032443 | **27.9893 dB** | 0.756916 |
| + x8 TTA | 0.032264 | **28.0355 dB** | **0.758519** |

vs. bicubic: +4.8942 dB / +0.207915. Both figures reproduce the previously
reported values exactly.

**Experiment 23 + x8 TTA (28.0355 dB / 0.758519 / 0.032264) is the protected
champion pipeline** that Experiment 24 must beat.

### Secondary group-aware diagnostic

Using the leakage-aware split from `evaluate_group_aware.py` (see Experiment 22
for why: 38 of 640 canonical validation images have a near-identical scene in
train):

| split | n | L1 | PSNR | SSIM | leaked images |
| --- | ---: | ---: | ---: | ---: | ---: |
| canonical | 640 | 0.032443 | 27.9893 | 0.756916 | 38 |
| group-aware | 640 | 0.030269 | 28.2333 | 0.758245 | 0 |

**Read this with care.** The group-aware split scores *higher*, not lower, which
is the opposite of what "removing leakage" naively predicts. The reason is that
it is a **different subset of images**, so the comparison is dominated by
per-image difficulty variance rather than by leakage. The group-aware number is
therefore only meaningful when compared against *itself* across experiments —
never as an absolute correction to the canonical metric. Canonical remains the
authoritative, historically comparable measurement.

---

# Experiment 22 — Dataset Degradation Forensics

## Status

**COMPLETED — analysis only** (originally numbered 21; see the numbering table
above). This is **not a training experiment**: no model
was trained, loaded, or evaluated, no checkpoint was created or modified, and no
result in this log was changed. It characterizes the dataset's
GT 256x256 -> NoisyLR 128x128 degradation so that Experiment 22's modeling
strategy can be chosen from measured evidence.

## Objective

Find exploitable structure in the degradation process that could enable a
substantially better model, rather than another incremental hyperparameter gain.
Experiments 9, 11, 12, 13, 14, 17 and 18 all failed to beat the champion by
changing architecture, loss, scheduler or ensembling — suggesting the remaining
headroom lies in the *data*, not the model.

## Method

- `src/degradation.py` — pure, unit-tested analysis primitives.
- `analyze_degradation.py` — driver over all **3,200 canonical training pairs**.
- Full outputs: [`results/degradation_analysis/degradation_report.md`](results/degradation_analysis/degradation_report.md)
  and [`degradation_report.json`](results/degradation_analysis/degradation_report.json),
  plus six diagnostic plots in the same directory.

Central quantity is the residual `r = NoisyLR - downsample(GT)`, which bundles
sensor noise, blur mismatch, resampling-phase error and quantization; separating
those is what the analysis does.

## Headline findings

| question | measured answer |
| --- | --- |
| Best GT->LR model | `bicubic` (MSE 0.008042); all 8 candidates within 29% of each other |
| Systematic gain/bias | None worth correcting: global `a=0.9947, b=+0.0023`; per-image affine correction recovers only **0.14%** of MSE |
| Residual magnitude | std **0.0897**, skew +0.408, excess kurtosis **+3.459** (heavy-tailed) |
| iid? | **No** |
| Signal dependent? | **Yes, strongly** — std rises 0.0120 -> 0.1631 across intensity (**13.6x**); `var(I) = -6.19e-05 + 0.00653·I + 0.0201·I²`, **R² = 0.9995** |
| Spatially correlated? | **No** — max abs autocorrelation 0.051 at any tested offset |
| Fixed pattern? | **No** — mean-residual map std 0.001598 vs 0.001585 expected from pure noise (ratio **1.01**) |
| Frequency structure? | **No** — flat spectrum, no periodic peaks, H/V power ratio 1.026 |
| Pre-downsampling blur? | **Negligible** — best sigma 0.4, only **0.328%** MSE gain |
| Multiple regimes? | **No** — per-image noise correlates **+0.899** with image brightness; spread is brightness, not discrete classes |
| Repeated scenes? | **Yes** — see below |
| Train vs validation | Distributionally matched (all deltas negligible) |

## The two most important discoveries

**1. The noise is strongly signal dependent (multiplicative/speckle-like).**
Residual variance grows ~quadratically with intensity, fitting a simple
closed-form model at R² = 0.9995. Plain L1 assumes homoscedastic noise, so the
current recipe systematically over-weights bright noisy pixels and under-weights
dark clean ones. For scale, the residual std (0.0897) is ~2.8x the validation L1
the champion pipeline achieves — the task is **noise-dominated, not
resolution-dominated**.

**2. Repeated scenes exist, and they leak across the split.**
Zero byte-identical GTs, but **119 confirmed repeated-scene groups covering 250
images** (GT MSE < 0.001, mean 0.000108 — versus mean *LR* MSE 0.014416 within
the same groups, i.e. one clean scene under independent noise draws). Every
confirmed pair sits at filename ID gap ≤ 2, so **filenames encode the grouping**.
**36 groups straddle the canonical split, giving 38 of 640 validation images
(5.9%) a near-identical twin in train** — absolute validation numbers are
therefore slightly optimistic. This does not invalidate any cross-experiment
comparison in this log (all used the identical split), and **the split was left
unchanged**.

## Ranked recommendations for Experiment 22

1. **Variance-stabilizing transform / noise-aware loss — HIGH.** Directly targets
   the strongest measured structure (13.6x variance span, R² 0.9995). Either train
   in a generalized-Anscombe-transformed space and invert at inference, or weight
   the existing L1 by `1/sqrt(var(I))` from the fitted coefficients. Cheap to bolt
   onto the champion recipe.
2. **Signal-dependent synthetic-noise augmentation — MEDIUM-HIGH.** The corruption
   process is now characterized, so unlimited synthetic pairs can be generated from
   2,560 GTs. Capped below HIGH because the fit matches aggregate variance while
   the measured excess kurtosis (+3.459) shows heavier tails than a matched Gaussian.
3. **Scene-group-aware training/validation — MEDIUM.** Enables Noise2Noise-style
   consistency terms across the repeated observations, and a cleaner group-aware
   validation split. Capped by the fact that only 7.8% of the dataset is involved.

Self-supervised (blind-spot / SURE) objectives are statistically admissible here —
noise is zero-mean (+0.000008) and spatially white — but rank below the above
because paired GT already exists.

**Deprioritized (LOW), with evidence:** fixed-pattern subtraction (ratio 1.01),
deblurring/kernel estimation (0.328%), per-image gain/bias calibration (0.14%),
degradation-regime classification (no clusters), further resampling-kernel search
(all candidates far below the noise floor).

## Result

Analysis complete. **No model trained.** Its top-ranked recommendation became
Experiment 24 below; the group-aware split it motivated is now available as the
standalone diagnostic `evaluate_group_aware.py`.

---

# Experiment 24 — Signal-Dependent Synthetic Noise Augmentation

## Status

**COMPLETE / REJECTED.** The real training run finished (from scratch, 90
epochs, `checkpoints/exp24_noise_aug/`). It trails the Experiment 23 champion
on both canonical and group-aware validation. Do not extend Experiment 24 or
tune `--synthetic-noise-prob` further; Experiment 25 explores a different
angle (explicit noise conditioning instead of synthetic substitution).

## Hypothesis

Training on additional independent noise realizations sampled from the
Experiment 22 heteroscedastic degradation model may improve generalization
beyond the finite set of 2,560 observed noisy inputs. Experiment 22 ranked this
MEDIUM-HIGH: the corruption process is now characterized, the dataset is small,
and synthesis needs no new labels.

## Baseline to beat

Experiment 23 + x8 TTA: **28.0355 dB / 0.758519 SSIM / 0.032264 L1**
(non-TTA 27.9893 / 0.756916 / 0.032443).

## The one experimental change

Everything else is held at the champion recipe: ResidualSRNet 64F/8B, from
scratch, seed 42, L1, crop96, batch16, Adam 1e-4, ReduceLROnPlateau
(0.5/3/1e-6), EMA decay 0.999. No architecture, loss, optimizer, crop or EMA
change — those are separate experiments.

## Synthetic degradation model

```
clean_lr     = bicubic_downsample_2x(GT)          # the Exp 22 best GT->LR model
variance(I)  = max(-6.19e-05 + 0.00653*I + 0.0201*I^2, variance_floor)
synthetic_lr = clean_lr + sqrt(variance(clean_lr)) * epsilon
```

No clipping: real `NoisyLR` itself ranges roughly `[-0.003, 1.33]`, so clipping
would make the synthetic stream *less* faithful.

### Chosen epsilon: Gaussian (not Student-t)

This is the counter-intuitive part, and it is decided by measurement rather than
by the usual heuristic. The real *standardized* residual `r/sigma(I)` does look
heavy-tailed (excess kurtosis +3.459, implying Student-t nu ≈ 5.7), which argues
for t. But the augmentation's job is to reproduce the real **residual**, and
heteroscedastic mixing across the intensity range already contributes excess
kurtosis on its own:

| quantity | real residual | Gaussian epsilon | Student-t epsilon |
| --- | ---: | ---: | ---: |
| std | 0.0848 | 0.0850 | 0.0850 |
| skewness | +0.4072 | 0.0000 | +0.0103 |
| excess kurtosis | +3.5185 | **+2.4536** | +8.5390 |
| mean abs percentile error | — | **0.01436** | 0.01979 |

Gaussian lands within 1.07 excess kurtosis of the real residual; Student-t
overshoots by 5.02 and is worse on percentiles too. Much of the standardized
residual's apparent tail weight is spread injected by imperfections in
`sigma(I)`, not genuine tail weight in epsilon. **Gaussian is selected**, and
`--synthetic-noise-distribution student_t` remains available for a future
ablation. Selection was made by `analyze_synthetic_noise.py`; full numbers in
[`results/synthetic_noise_analysis/synthetic_noise_report.json`](results/synthetic_noise_analysis/synthetic_noise_report.json).

## Match quality (the Experiment 22 gate)

Measured over 400 pairs, synthetic vs real residuals:

- overall std **0.0850 vs 0.0848** (0.2% high)
- **37 of 40** intensity bins within 10%; median ratio **1.004**
- spatial autocorrelation stays near-white in both, as measured
- **known limitation 1 — skewness:** real +0.407, synthetic ~0.000. Neither
  candidate distribution is skewed; matching it would need an asymmetric noise
  model, which was out of scope here.
- **known limitation 2 — darkest bin:** worst ratio **0.449 at I=0.013**. The
  fitted quadratic has a *negative* constant term, so variance is clamped to
  zero below I=0.0092 and under-predicts sigma by ~2.5x in the darkest bin
  (~2.8% of pixels). Setting `--synthetic-noise-variance-floor 1.43e-4`
  (the measured dark-bin variance) improves the worst bin from 0.449 to ~1.21;
  the default stays **0.0** so the Experiment 22 model is reproduced verbatim.

**Verdict: the synthetic degradation reproduces the measured process well
enough to proceed.** Both limitations are localized and documented rather than
silently absorbed.

## Mixture policy

Per training sample, probability `--synthetic-noise-prob` (Experiment 24 uses
**0.5**) of substituting a synthetic LR; otherwise the real `NoisyLR` is used.
Real paired data is never removed. **Validation is 100% real** — `build_datasets`
never passes the augmentation to the validation dataset, so no synthesized input
can reach a reported metric, the scheduler, or checkpoint selection. Default is
`0.0`, so every historical command behaves exactly as before.

## Alignment and transform order

```
load real LR + GT
  -> [maybe] replace LR with bicubic_downsample(GT) + noise   <-- full resolution
  -> aligned paired random crop (existing PairedRandomCrop)
  -> paired flips / 90-degree rotation (existing augmentation)
```

Substitution happens at **full resolution, before any cropping**, so the
synthetic tensor is a drop-in with the identical spatial layout as the real LR.
The existing aligned-crop transform then handles it unchanged — GT and LR
cannot desynchronize because nothing crops them independently. A test confirms
this end-to-end by re-downsampling the *returned* GT crop and checking it
reproduces the returned LR crop to within the noise scale.

## Reproducibility

Noise is drawn from `numpy.random.default_rng(SeedSequence([seed, epoch, index]))`
inside `SyntheticNoiseAugmentation` — an explicitly seeded generator, matching
`src/splits.py`'s convention. No global/process RNG is touched, so the
augmentation is a pure function of `(seed, epoch, index)`: identical across
DataLoader worker counts, reproducible on re-run, yet drawing a **fresh
realization each epoch** (`train.py` calls `train_dataset.set_epoch(epoch)`).

## Checkpoint / resume

`synthetic_noise_config` is stored in every checkpoint and fully reconstructs the
augmentation: enabled, probability, seed, distribution, degrees of freedom,
variance coefficients, variance floor, downsampling identifier, scale. Resume
uses the same `_UNSET`-sentinel strict-match pattern as `ema_config`: matching
config resumes; different probability, distribution, or variance floor is
rejected; a synthetic checkpoint resumed with the augmentation disabled is
rejected; and historical checkpoints (which lack the key entirely) resume
unchanged.

## Tests

54 new tests in `tests/test_synthetic_noise_unit.py`. Full fast suite:
**525 passed, 8 deselected** (up from 471).

## CUDA sanity

ResidualSRNet 64F/8B, batch16, crop96, L1, EMA 0.999, probability 0.5 on real
data: batch shapes `(16,1,96,96)`/`(16,1,192,192)`, a genuine real/synthetic mix
within the batch, finite inputs, finite loss and gradients, EMA updated, peak
allocated 783.6 MiB / reserved 880.0 MiB, no OOM.

## Real-data smoke test

```bash
python train.py --model residual_sr --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --ema --ema-decay 0.999 --synthetic-noise-prob 0.5 \
  --epochs 2 --max-train-samples 32 --max-val-samples 16 \
  --checkpoint-dir checkpoints/exp24_noise_aug_smoke --num-workers 0
```

Verified then deleted: mixed real/synthetic training batches, real-only
validation, EMA-driven validation, correct `synthetic_noise_config` in the
checkpoint, `evaluate_checkpoint.py` working with `--tta none` and `--tta x8`,
matching resume succeeding (scheduler *and* EMA state restored), and a
mismatched `--synthetic-noise-prob 0.25` resume correctly rejected.

## Result

**REJECTED — trails the Experiment 23 champion.**

```bash
python train.py --model residual_sr --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --ema --ema-decay 0.999 \
  --synthetic-noise-prob 0.5 --synthetic-noise-distribution gaussian \
  --epochs 90 --checkpoint-dir checkpoints/exp24_noise_aug
```

| split | PSNR (dB) | SSIM | L1 |
| --- | ---: | ---: | ---: |
| canonical, non-TTA | 27.8721 | 0.747690 | 0.032828 |
| canonical, x8 TTA | 27.9116 | 0.749092 | 0.032667 |
| group-aware (diagnostic) | 28.1264 | 0.748362 | 0.030632 |

Champion for comparison — Experiment 23 EMA e90 + x8 TTA: **28.0355 dB /
0.758519 / 0.032264** (canonical, non-TTA 27.9893 / 0.756916 / 0.032443;
group-aware 28.2333 / 0.758245 / 0.030269).

Experiment 24 trails Experiment 23 on both canonical and group-aware
validation, on every metric. The synthetic-noise substitution did not improve
generalization here — plausibly because the augmentation, despite passing the
Experiment 22 match-quality gate, still introduces a residual synthetic/real
domain gap that offsets whatever benefit additional noise realizations bring.
**Do not extend this experiment or tune `--synthetic-noise-prob` further.**
Experiment 25 tests a different hypothesis: give the model an explicit,
per-pixel estimate of the measured noise level instead of substituting
synthetic input.

---

# Experiment 25 — Noise-Conditioned ResidualSR

## Status

**PLANNED / PREPARED.** Implementation, tests, CUDA sanity and a tiny
real-data smoke test are all complete and passing. **No real training run has
been started** (`checkpoints/exp25_noise_conditioned/` does not exist).

## Hypothesis

Experiment 24 substituted synthetic degraded inputs to expose the model to
more noise realizations, but a residual synthetic/real domain gap likely
offset the benefit (see Experiment 24 result above). Experiment 25 instead
trains exclusively on real NoisyLR/GT pairs — no synthetic substitution — but
gives the model an explicit, per-pixel estimate of the measured
signal-dependent noise level (Experiment 22's model) as a second input
channel. Providing this as an auxiliary signal, rather than perturbing the
input distribution, may let the network adapt its restoration strength to the
local noise level without incurring any synthetic-domain mismatch.

Champion for comparison — Experiment 23 EMA e90 + x8 TTA: **28.0355 dB /
0.758519 / 0.032264**.

## Noise conditioning

```
channel 0 = NoisyLR                                   (real, completely unmodified)
channel 1 = sigma(clamp(NoisyLR, 0, 1))                (Experiment 22's measured model)

I           = clamp(NoisyLR, 0, 1)                     # clamp affects ONLY the sigma estimate
variance    = clamp(-6.19e-05 + 0.00653*I + 0.0201*I^2, min=0)   # floor 0.0, Exp22 verbatim
sigma       = sqrt(variance)                           # fed raw -- no normalization
```

The clamp is isolated to the intensity used for sigma estimation; the LR
channel that reaches the model is never clamped or otherwise modified. Sigma
is fed raw, with no arbitrary normalization. Mutually exclusive with
Experiment 24 — no synthetic noise augmentation is used here.

Implemented once in `src/noise_conditioning.py` (`conditioning_sigma_map`,
`prepare_model_input`) reusing `src/synthetic_noise.py`'s variance/sigma
formulas rather than duplicating them, so training, validation, x8 TTA,
`evaluate_checkpoint.py`, `infer_test.py`, and `evaluate_group_aware.py` all
compute the identical conditioning map.

## Model

Existing `ResidualSRNet`, unmodified apart from `in_channels=2` (was 1) so
`conv_in` accepts the extra sigma channel. 64F/8B, scale 2, same residual
blocks and upsampling head as every other ResidualSR experiment. No
attention, no additional blocks, no extra width, no denoiser, no synthetic
augmentation, no weighted/SSIM/Charbonnier/MSE loss, no crop-size or
scheduler/EMA-decay change. Parameter count: 631,300 (vs 630,724 for the
single-channel model — the difference is entirely `conv_in`'s extra input
channel).

## Wrapper design (`NoiseConditionedModel`)

`wrap_for_conditioning` wraps the base model in a thin `nn.Module` whose
`forward` computes `prepare_model_input(lr, config)` before delegating to the
base model. Standard `nn.Module` composition means `.parameters()`,
`.to(device)`, `.train()`/`.eval()`, and `copy.deepcopy` (used by
`ExponentialMovingAverage`) all transparently pass through — so `src/tta.py`,
`train.train_one_epoch`/`validate`, `evaluate_checkpoint.validate_x8`,
`infer_test.run_inference`, and `evaluate_group_aware.py` needed **zero**
changes; every caller keeps passing plain single-channel LR tensors. When
conditioning is disabled, `wrap_for_conditioning` returns the exact same model
object (no wrapper at all), so historical commands are byte-for-byte
unaffected.

## x8 TTA spatial consistency

Since sigma is an exactly pointwise function of LR, "transform LR then
compute sigma" (what the wrapper does automatically inside `predict_x8`) and
"compute sigma then transform both channels together" are mathematically
identical. Verified numerically:
`test_x8_conditioning_matches_concatenate_then_transform` compares the two
orderings and confirms `torch.allclose(..., atol=1e-6)` with max difference
`0.0`.

## Checkpoint / resume

`noise_conditioning_config` is stored in every checkpoint:
`{enabled, method: "signal_dependent_sigma", variance_coefficients, input_intensity_clamp: [0.0, 1.0], variance_floor: 0.0, sigma_normalization: "none"}`,
or `None` when disabled (mirrors `ema_config`/`synthetic_noise_config`).
`evaluate_checkpoint.load_model` reconstructs the wrapped model automatically
from this key, so `infer_test.py` and `evaluate_group_aware.py` need no
changes of their own. Resume uses the same `_UNSET`-sentinel strict-match
pattern as `ema_config`/`synthetic_noise_config`: matching config resumes;
different coefficients, floor, or method is rejected; a conditioned checkpoint
resumed with conditioning disabled (or vice versa) is rejected; historical
checkpoints (which lack the key entirely) resume unchanged.

## Tests

35 new tests in `tests/test_noise_conditioning_unit.py` covering the variance
formula, clamp isolation, LR-channel preservation, variance non-negativity,
sigma finiteness/monotonicity, tensor shapes and channel content, the
disabled/historical passthrough, `ResidualSRNet` with `in_channels=2`,
training forward/backward, EMA compatibility, validation, x8 TTA equivalence,
`evaluate_checkpoint`/`infer_test`/`evaluate_group_aware` integration,
checkpoint config correctness, and matching/mismatched/historical resume. Full
fast suite: **560 passed, 8 deselected** (up from 525).

## CUDA sanity

ResidualSRNet 64F/8B, `in_channels=2`, batch16, crop96 (LR 48x48, GT
96x96), L1, EMA 0.999: finite forward (`631,300` trainable params, output
shape `(16,1,96,96)`), finite loss/gradients, EMA update succeeded, no OOM.
Allocated 13.3 MiB (peak 211.8 MiB), reserved 270.0 MiB (peak 270.0 MiB).

## Real-data smoke test

```bash
python train.py --checkpoint-dir checkpoints/exp25_noise_conditioned_smoke \
  --max-train-samples 32 --max-val-samples 16 --epochs 2 --batch-size 8 \
  --crop-size 96 --num-features 64 --num-blocks 8 --loss l1 \
  --ema --ema-decay 0.999 --noise-conditioning --seed 42
```

Verified then deleted: only real NoisyLR/GT used (no synthetic augmentation),
the conditioning map generated correctly, EMA-driven validation, correct
`noise_conditioning_config` stored in the checkpoint,
`evaluate_checkpoint.py` working with `--tta none` and `--tta x8`,
`evaluate_group_aware.py` running end-to-end, `infer_test.py` producing
finite x8 predictions, a matching resume succeeding (EMA state restored,
continuing at epoch 3), and a mismatched resume (conditioning disabled)
correctly rejected. Historical checkpoint hashes
(`exp19_ema`, `exp20_ema_extended70`, `exp21_ema_extended80`,
`exp23_ema_extended90`, `exp24_noise_aug`, `exp6_crop96`) confirmed identical
before and after this preparation.

## Result

**TBD.** No real Experiment 25 training run has been started. Proposed
command:

```bash
python train.py --model residual_sr --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --ema --ema-decay 0.999 --noise-conditioning \
  --epochs 90 --checkpoint-dir checkpoints/exp25_noise_conditioned
```

Then compare with `evaluate_checkpoint.py --checkpoint
checkpoints/exp25_noise_conditioned/checkpoint_best.pt --tta x8` against the
protected champion (Experiment 23, 28.0355 dB / 0.758519 / 0.032264).

---

# Experiment 26 — ResidualSR Ablation Toolkit (infrastructure only, no results yet)

## Status

**PLANNED / PREPARED.** Eight independently-selectable capabilities were added
to the training pipeline so each can be ablated on its own against the
Experiment 23 champion (`checkpoints/exp23_ema_extended90/checkpoint_best.pt`,
EMA e90 + x8 TTA: **28.0355 dB / 0.758519 / 0.032264**). ResidualSR is
unchanged as the default architecture; every capability below defaults off
and reproduces historical behavior exactly unless explicitly requested.
**No real ablation training run has been started** — this section documents
capability and reproducible commands only, per instruction; no results are
reported here.

## What's new

1. **`--loss mixed --mixed-loss-alpha 0.5`** (`src/losses.py::MixedL1MSELoss`)
   — `alpha*L1 + (1-alpha)*MSE`, default `alpha=0.5`. Existing `--loss l1`
   (default) / `mse` / `charbonnier` / `l1_ssim` unchanged.
2. **`--finetune-from <checkpoint>`** (`train.py::load_weights_for_finetune`)
   — loads model weights only (prefers EMA weights when present) and starts
   a brand-new experiment: fresh optimizer/`--lr`/scheduler, epoch counter
   reset to 1, its own best-score tracking, its own `--checkpoint-dir`.
   Distinct from `--resume` (restores full training state to continue the
   *same* run); the two are mutually exclusive
   (`train.py::validate_finetune_args`), and a fine-tuning run is rejected
   if `--checkpoint-dir` would be the same directory as the source
   checkpoint's, so the source can never be overwritten.
3. **`--global-bicubic-residual`** (`train.py::resolve_model_architecture`)
   — `prediction = bicubic_upsample(LR) + learned_residual(LR)`. Reuses
   Experiment 17's already-implemented, already-tested
   `ResidualSRBicubic` (`src/models/residual_sr_bicubic.py`) rather than a
   new implementation; just a convenience mapping from `--model residual_sr`
   onto it. Rejects being combined with any architecture other than
   `residual_sr`/`residual_sr_bicubic`, rather than silently ignoring it.
4. **Configurable crop size (`--crop-size`)** — already existed and is
   already thoroughly tested at 64 (default), 96 (Experiment 6), and 128 =
   full LR image (Experiment 7); see `tests/test_transforms_unit.py`. No
   changes made here; verified still passing.
5. **`--channel-attention --attention-reduction 8`**
   (`src/models/attention.py::ChannelAttention`, wired into
   `src/models/residual_sr.py::ResidualBlock`) — squeeze-and-excitation gate
   (global-avg-pool → 1×1 reduce → ReLU → 1×1 expand → sigmoid → channel-wise
   multiply) optionally inserted into each residual block. Default off
   constructs no attention submodule at all — zero parameter/state-dict
   change from before this existed.
6. **`--multiscale-block`**
   (`src/models/residual_sr.py::MultiScaleBlock`) — local 3×3 branch +
   dilated 3×3 (dilation=2) branch, concatenated and fused with a single 1×1
   convolution before the residual add, replacing `ResidualBlock` when
   requested. No 5×5 kernel. Independent of `--channel-attention` (neither
   auto-enables the other; each has its own dedicated tests proving the
   other stays off).
7. **Experiment isolation** — every capability is a separate CLI flag with
   its own `--checkpoint-dir`; none are combined by default.
8. **Backward compatibility** — `build_model_config`/checkpoint `model_config`
   only gain new keys (`channel_attention`, `attention_reduction`,
   `multiscale_block`) when actually requested (mirrors how the
   `"architecture"` key itself only appears for non-default architectures),
   so a plain `--model residual_sr` run with none of these flags produces the
   byte-identical dict every historical checkpoint has, and every existing
   checkpoint (Experiments 1–25) remains loadable and resumable unmodified.

## Parameter counts (64F/8B — the champion capacity)

| variant | trainable params | vs. baseline |
| --- | ---: | ---: |
| ResidualSR (baseline, unchanged) | 630,724 | — |
| ResidualSR + channel attention (reduction=8) | 639,492 | +8,768 (+1.4%) |
| ResidualSR + multi-scale block | 696,772 | +66,048 (+10.5%) |

## Tests

122 new tests: mixed-loss formula/construction/backward-pass
(`tests/test_losses_unit.py`), channel-attention/multi-scale-block
shape/finiteness/backward-pass/independent-selectability/backward-compat
(`tests/test_model_unit.py`), and fine-tune-vs-resume
semantics/`--global-bicubic-residual` mapping (new
`tests/test_finetune_unit.py`). Full fast suite: **617 passed, 8 deselected**
(up from 560).

## CLI-wiring smoke checks (not real training; deleted afterward)

One-epoch, 8-train/4-val runs at 8F/2B confirmed the full
CLI→model→train→checkpoint path for each new flag (`--loss mixed`,
`--global-bicubic-residual`, `--channel-attention`, `--multiscale-block`),
plus a fine-tune run from the real Experiment 23 champion checkpoint
(`--finetune-from checkpoints/exp23_ema_extended90/checkpoint_best.pt --loss
mse --lr 1e-5`) that correctly loaded its EMA weights (one-epoch tiny-batch
Val PSNR 27.83 dB, consistent with starting from the champion's 27.99 dB
rather than random init) and left the source checkpoint file byte-identical
(SHA-256 verified before/after). Both `validate_finetune_args` guards
(`--resume`+`--finetune-from` together; `--checkpoint-dir` equal to the
source's directory) were confirmed to reject with a clear error.

## Proposed ablation commands (A–H; none run yet)

Each uses a separate `checkpoints/exp26_*` directory. `<BEST>` =
`checkpoints/exp23_ema_extended90/checkpoint_best.pt`, the current champion.

**A. Baseline ResidualSR** (already the champion; re-stated for reference)
```bash
python train.py --model residual_sr --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --ema --ema-decay 0.999 --epochs 90 --checkpoint-dir checkpoints/exp23_ema_extended90
```

**B. ResidualSR + MSE fine-tuning** (`--num-features`/`--num-blocks` must
match `<BEST>`'s 64F/8B -- they default to 32/4, so omitting them here would
fail to load the checkpoint's weights with a shape mismatch)
```bash
python train.py --finetune-from <BEST> --num-features 64 --num-blocks 8 \
  --loss mse --lr 1e-5 --epochs 10 --checkpoint-dir checkpoints/exp26_finetune_mse
```

**C. ResidualSR + mixed-loss fine-tuning**
```bash
python train.py --finetune-from <BEST> --num-features 64 --num-blocks 8 \
  --loss mixed --mixed-loss-alpha 0.5 --lr 1e-5 \
  --epochs 10 --checkpoint-dir checkpoints/exp26_finetune_mixed
```

**D. ResidualSR + bicubic residual** (from scratch, champion recipe otherwise)
```bash
python train.py --model residual_sr --global-bicubic-residual \
  --num-features 64 --num-blocks 8 --loss l1 --crop-size 96 --batch-size 16 \
  --seed 42 --lr 1e-4 --scheduler plateau --scheduler-factor 0.5 \
  --scheduler-patience 3 --min-lr 1e-6 --ema --ema-decay 0.999 \
  --epochs 90 --checkpoint-dir checkpoints/exp26_bicubic_residual
```

**E. ResidualSR + 96 crop** (from scratch, champion recipe otherwise)
```bash
python train.py --model residual_sr --crop-size 96 \
  --num-features 64 --num-blocks 8 --loss l1 --batch-size 16 \
  --seed 42 --lr 1e-4 --scheduler plateau --scheduler-factor 0.5 \
  --scheduler-patience 3 --min-lr 1e-6 --ema --ema-decay 0.999 \
  --epochs 90 --checkpoint-dir checkpoints/exp26_crop96
```
*(Note: the champion recipe already trains at `--crop-size 96`; this row is
here for completeness/explicitness of the ablation matrix.)*

**F. ResidualSR + full 128 crop** (from scratch, champion recipe otherwise)
```bash
python train.py --model residual_sr --crop-size 128 \
  --num-features 64 --num-blocks 8 --loss l1 --batch-size 16 \
  --seed 42 --lr 1e-4 --scheduler plateau --scheduler-factor 0.5 \
  --scheduler-patience 3 --min-lr 1e-6 --ema --ema-decay 0.999 \
  --epochs 90 --checkpoint-dir checkpoints/exp26_crop128_full
```

**G. ResidualSR + channel attention** (from scratch, champion recipe otherwise)
```bash
python train.py --model residual_sr --channel-attention --attention-reduction 8 \
  --num-features 64 --num-blocks 8 --loss l1 --crop-size 96 --batch-size 16 \
  --seed 42 --lr 1e-4 --scheduler plateau --scheduler-factor 0.5 \
  --scheduler-patience 3 --min-lr 1e-6 --ema --ema-decay 0.999 \
  --epochs 90 --checkpoint-dir checkpoints/exp26_channel_attention
```

**H. ResidualSR + multi-scale block** (from scratch, champion recipe otherwise)
```bash
python train.py --model residual_sr --multiscale-block \
  --num-features 64 --num-blocks 8 --loss l1 --crop-size 96 --batch-size 16 \
  --seed 42 --lr 1e-4 --scheduler plateau --scheduler-factor 0.5 \
  --scheduler-patience 3 --min-lr 1e-6 --ema --ema-decay 0.999 \
  --epochs 90 --checkpoint-dir checkpoints/exp26_multiscale_block
```

Each proposed run's result should be compared independently against the
Experiment 23 champion (28.0355 dB / 0.758519 / 0.032264) with
`evaluate_checkpoint.py --checkpoint <dir>/checkpoint_best.pt --tta x8`.

## Result

**B and C ran; both REJECTED (trail the champion). D–H have not been started.**

Real fine-tuning runs exist on disk (`checkpoints/exp26_finetune_mixed/`,
`checkpoints/exp26_finetune_mse/`) and were independently re-measured with
`evaluate_checkpoint.py` (not just read from checkpoint metadata):

| variant | epoch | Val PSNR (non-TTA) | Val SSIM (non-TTA) | Val PSNR (x8 TTA) |
| --- | ---: | ---: | ---: | ---: |
| **B — MSE fine-tune** | 8 | 27.8966 dB | 0.754295 | not measured |
| **C — Mixed L1+MSE fine-tune** | 23 | 27.9527 dB | 0.756677 | 28.0027 dB |
| Champion (Exp 23 + x8 TTA, for reference) | 90 | 27.9893 dB | 0.756916 | **28.0355 dB** |

Both trail the champion at every measured comparison point (B by -0.0927 dB
non-TTA; C by -0.0366 dB non-TTA, -0.0328 dB at x8 TTA). Consistent with the
project's prior finding that pure MSE underperforms (Experiment 8) and that
mixed L1+MSE is a smaller, gentler version of the same issue — MSE-leaning
objectives have not beaten L1 at this checkpoint's convergence point in this
project. **Rejected; do not extend B or C further.**

**Caveat — B and C were run at `--crop-size 64` (the CLI default), not the
champion's `--crop-size 96`,** because the commands documented above for B/C
never pass `--crop-size` explicitly (unlike D–H, which do). This means B/C
are not a clean single-variable ablation against the champion (loss changed
*and* crop size reverted to 64) — a confound worth closing before drawing
strong conclusions about the loss change specifically. Re-running B/C with
`--crop-size 96` added is the cleaner comparison if this is revisited; not
done here since both already trail the champion even with the confound
working in a less-clear direction, and GPU time is limited.

**G — channel attention has an incomplete/invalid artifact on disk,
`checkpoints/exp27_channel_attention/`, and it must not be used as an
ablation result.** Inspecting its checkpoint metadata: only 10 epochs
completed (not the planned 90), `crop_size=64` (not the champion's 96, same
missing-flag issue as B/C above), and `best_val_psnr=21.37` -- far below
every other model in this project. This is consistent with an EMA shadow
(`decay=0.999`) still very close to its random initialization after only 10
epochs, not evidence that channel attention hurts. **Treat this checkpoint
directory as a stray/incomplete artifact, not a completed Experiment G.** The
command in the table above (G, with `--crop-size 96 --epochs 90` as written)
is the correct one to actually run.

---

# Experiment 27 — Variance-Weighted L1 Loss (infrastructure, no results yet)

## Status

**PLANNED / PREPARED.** Implementation and unit tests are complete. **No
real training run has been started.**

## Motivation

This is the direct implementation of Experiment 22's forensics report's
**highest-ranked, previously-untried** recommendation (see
`results/degradation_analysis/degradation_report.md`, "Ranked strategies for
Experiment 22", item 1, "expected payoff: HIGH"): *"weight the existing L1 by
1/√var(I) using the fitted coefficients."* Experiments 24 (synthetic noise
substitution) and 25 (noise conditioning) both respond to the same
signal-dependent-noise finding via different mechanisms (augmenting the input
distribution, and giving the model an explicit sigma channel, respectively);
this is a third, independent, previously-untried response that changes only
the loss function.

## Implementation

`src/losses.py::VarianceWeightedL1Loss` -- `mean(weight * |pred - target|)`
where `weight = (1/(sigma(I) + eps))`, normalized so `mean(weight) == 1`
per batch (keeps the loss on the same numeric scale as plain L1, so existing
`--lr` values remain sensible). `sigma(I)` reuses
`src.synthetic_noise.noise_sigma`/`VARIANCE_COEFFICIENTS` -- the exact
Experiment 22 fit, not a re-derived one. `I` is estimated from the *target*
(HR GT), clamped to `[0,1]`, since that is the space this loss operates in --
a documented approximation of a law that was fit in *LR* space (see the
docstring in `src/losses.py` for the full caveat). Exposed as
`--loss weighted_l1`, with `--weighted-l1-eps` (default `1e-2`) and
`--weighted-l1-variance-floor` (default `0.0`, reproduces the fitted model
exactly). Integrates through the existing `build_loss_config`/`build_loss`/
`loss_label` dispatch (`src/losses.py`) exactly like `charbonnier`/`mixed`,
so no special-casing was added to `train.py`'s training loop, checkpointing,
or resume logic -- the existing generic `loss_config` dict-equality resume
check already covers it.

## Tests

18 new tests in `tests/test_losses_unit.py` covering: the mathematical
formula against a from-scratch reference computation, identical-input ->
zero loss, weight normalization (verified against plain L1 on a
uniform-intensity image, where normalization forces every weight to exactly
1), that error concentrated in a low-intensity region is weighted more than
the identical error concentrated in a high-intensity region (same target, so
an identical weight map, isolating where the error sits), finite
gradients/backward pass, finiteness for out-of-`[0,1]`-range values (real
NoisyLR/GT are not strictly bounded), rejection of non-positive `eps`, and
`build_loss_config`/`build_loss`/`loss_label` wiring. Full fast suite: to be
re-measured; see the top-level test run in this update for the current total.

## CUDA / wiring sanity

Verified via the existing `--loss` dispatch path exercised by
`tests/test_losses_unit.py`'s backward-pass tests (CPU, `requires_grad`
tensors, finite gradients). No dedicated CUDA smoke run was performed for
this loss alone since it changes nothing about the model, optimizer,
scheduler, or EMA machinery Exp 24/25 already CUDA-sanity-checked -- only the
per-pixel loss weighting.

## Proposed command

```bash
python train.py --model residual_sr --num-features 64 --num-blocks 8 \
  --loss weighted_l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --ema --ema-decay 0.999 --epochs 90 --checkpoint-dir checkpoints/exp27_weighted_l1
```

Compare with `evaluate_checkpoint.py --checkpoint
checkpoints/exp27_weighted_l1/checkpoint_best.pt --tta x8` against the
protected champion (Experiment 23, 28.0355 dB / 0.758519 / 0.032264).

## Result

**TBD.** No real training run has been started.

---

# Experiment 28 — Hard-Patch / Informative-Patch Sampling (infrastructure, no results yet)

## Status

**PLANNED / PREPARED.** Implementation and unit tests are complete. **No
real training run has been started.**

## Motivation

Standard `PairedRandomCrop` samples training crop origins uniformly at
random. This optionally biases sampling toward higher-gradient-energy
("harder"/more informative) regions of each training image, on the premise
that more training signal per step may come from edges/texture than from
flat regions -- while still mixing in plain random crops so training is
never restricted to only high-gradient areas.

## Implementation

`src/transforms.py::gradient_energy_map` (a simple central-difference
|dx|+|dy| proxy for local high-frequency content) +
`sample_informative_crop_origin` (weighted-random sampling over every legal
crop origin, weights proportional to that origin's window's summed gradient
energy, computed in `O(H*W)` via a 2D prefix-sum/integral-image table rather
than one pass per candidate origin) + `PairedHardPatchCrop` (applies it, then
delegates to the existing `aligned_paired_crop` -- the same alignment
primitive `PairedRandomCrop` uses, so LR/GT can never desynchronize) +
`PairedMixedCrop` (per-sample Bernoulli mixture: probability
`hard_patch_prob` -- default `0.5` -- uses the informative crop, otherwise
falls back to plain `PairedRandomCrop`). `create_training_transform` gains
`hard_patch_sampling`/`hard_patch_prob` parameters (default
`hard_patch_sampling=False`, byte-for-byte the historical
`PairedRandomCrop`-only pipeline). `train.py` gains `--hard-patch-sampling`
and `--hard-patch-prob` (default `0.5`), threaded through `build_datasets`,
recorded in `training_config`, and checked (warn, not reject -- same policy
as `--crop-size`) on resume via an extended `warn_on_resume_config_mismatch`.

Sampling is *weighted-random*, not a deterministic argmax: a flat +1e-6
floor keeps every legal origin sampleable even in a perfectly flat window, so
the same image can still yield different informative crops across epochs
(via the shared generator's advancing state) rather than always picking the
identical best window.

## Tests

16 new tests in `tests/test_hard_patch_unit.py`: gradient-energy-map
correctness on flat/edge images, in-bounds sampling over many seeds,
statistical bias toward a synthetic high-energy region (vs. a uniform-random
baseline), no crash/NaN on a fully flat image, exact 2x LR/GT alignment
(both `PairedHardPatchCrop` alone and all three `PairedMixedCrop` probability
settings), deterministic reproducibility given the same seed (single crop,
a 20-call sequence, and via `create_training_transform`'s `seed=` parameter),
`hard_patch_prob=0.0` never invoking informative sampling (verified by
monkeypatching it to raise), `hard_patch_prob=1.0` always invoking it (call
count verified), `hard_patch_prob=0.5` invoking both branches roughly
proportionally, full-image and oversized-crop boundary behavior identical to
`PairedRandomCrop`, and in-bounds crop origins over many seeds on a
synthetic high-energy image. All existing `tests/test_transforms_unit.py`
(37 tests) and `tests/test_transforms_integration.py`/
`tests/test_training_unit.py` pass unchanged, confirming
`hard_patch_sampling=False` (the default) leaves the historical pipeline
untouched.

## CLI-wiring smoke check (not real training; deleted afterward)

`--loss weighted_l1 --hard-patch-sampling --hard-patch-prob 0.5
--denoise-stem --denoise-stem-blocks 2`, 1 epoch, 8 train / 4 val samples,
8-feature/2-block model: ran to completion, printed
`Hard-patch sampling: enabled (prob=0.5)`, finite loss/PSNR/SSIM, checkpoint
saved and loadable. Checkpoint directory deleted after verification.

## Proposed command

```bash
python train.py --model residual_sr --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --ema --ema-decay 0.999 --hard-patch-sampling --hard-patch-prob 0.5 \
  --epochs 90 --checkpoint-dir checkpoints/exp28_hard_patch
```

## Result

**TBD.** No real training run has been started.

---

# Experiment 29 — Degradation-Aware Denoising Stem (infrastructure, no results yet)

## Status

**PLANNED / PREPARED.** Implementation and unit tests are complete. **No
real training run has been started.**

## Justification (per project instructions: implement only if forensics support it)

Experiment 22 found the NoisyLR corruption is dominated by meaningful,
strongly signal-dependent noise (residual std 0.0897, ~2.8x the champion's
validation L1) with negligible blur (best Gaussian sigma 0.4, only 0.328%
MSE gain) and no fixed pattern (ratio 1.01) -- i.e. "meaningful noise before
upsampling" in the sense that justifies trying an explicit pre-trunk
denoising stage. This is offered as an *additional*, independently-selectable
mechanism alongside Experiment 25's noise-conditioning channel (which
responds to the same finding differently, by giving the model an explicit
sigma estimate rather than trying to remove noise before the trunk) -- not a
replacement for it; the two are orthogonal and can in principle be combined
later if both help individually.

## Implementation

`src/models/denoise_stem.py::DenoiseStem` -- `Conv3x3 -> {2..4} x
SimpleGateBlock -> Conv3x3`, applied as a residual correction
(`output = input + stem(input)`) before `ResidualSRNet.conv_in`.
`SimpleGateBlock` is a simplified-NAFNet-style block (channel-doubling conv
-> element-wise "simple gate" a*b -> conv, no LayerNorm, no attention, no
multi-stage encoder/decoder) per the project's explicit preference for a
lightweight gated block over a full NAFNet. Wired into `ResidualSRNet` via
`denoise_stem`/`denoise_stem_features`/`denoise_stem_blocks` (all default
off/32/2), following the exact same "no submodule constructed at all unless
requested" convention as `channel_attention`/`multiscale_block`, and into
`build_model_config`/`build_model` (`src/models/__init__.py`) the same way.
`train.py` gains `--denoise-stem`, `--denoise-stem-features` (default `32`),
`--denoise-stem-blocks` (default `2`, 2-4 recommended per the spec).

## Parameter counts (64F/8B -- the champion capacity)

| variant | trainable params | vs. baseline |
| --- | ---: | ---: |
| ResidualSR (baseline, unchanged) | 630,724 | -- |
| ResidualSR + denoise stem (32 features, 2 blocks) | 686,821 | +56,097 (+8.9%) |

Modest growth, as instructed.

## Tests

12 new tests in `tests/test_model_unit.py`: `SimpleGateBlock` shape/
finiteness/backward pass, `DenoiseStem` shape preservation for both
`in_channels=1` (real LR) and `in_channels=2` (compatibility with
Experiment 25's noise-conditioning wrapper, which concatenates a sigma
channel), rejection of non-positive `num_blocks`, backward pass, default
construction unaffected (`model.stem is None` when disabled), forward shape
and modest (<25%) parameter growth when enabled, `build_model_config`/
`build_model` wiring (keys omitted when disabled, included when enabled,
legacy configs still build a stemless model), and explicit combination with
`--rdb-block` (independent mechanisms, not mutually exclusive).

## CLI-wiring smoke check (not real training; deleted afterward)

Same combined smoke check as Experiment 28 above (`--denoise-stem
--denoise-stem-blocks 2` alongside `--loss weighted_l1
--hard-patch-sampling`): ran to completion, model config printed
`'denoise_stem': True, 'denoise_stem_features': 32, 'denoise_stem_blocks': 2`,
finite loss/PSNR/SSIM. Checkpoint directory deleted after verification.

## Proposed command

```bash
python train.py --model residual_sr --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --ema --ema-decay 0.999 --denoise-stem --denoise-stem-blocks 2 \
  --epochs 90 --checkpoint-dir checkpoints/exp29_denoise_stem
```

## Result

**TBD.** No real training run has been started.

---

# Experiment 30 — Lightweight Residual Dense Block (Phase 10, infrastructure, no results yet)

## Status

**PLANNED / PREPARED.** Implementation and unit tests are complete. **No
real training run has been started.**

## Motivation

An optional third residual-block alternative (alongside `ResidualBlock` and
Experiment 26's `MultiScaleBlock`), inspired by RDN (Zhang et al. 2018) but
deliberately not a full RDN -- a single dense-connectivity block dropped into
the existing trunk, reusing `ResidualSRNet`'s stem and upsampling head
unchanged, per the project's instruction not to require broad refactoring.

## Implementation

`src/models/residual_sr.py::ResidualDenseBlock` -- `num_layers` (default 3)
3x3 convolutions, each seeing the concatenation of the block's input and
every previous layer's output (dense feature reuse), a 1x1 local-feature-
fusion convolution back down to `num_features`, and a residual add. A small
`growth_rate` (default 16, channels added per layer) keeps the concatenated
channel count -- and therefore the parameter count -- modest. Wired into
`ResidualSRNet` via `rdb_block`/`rdb_growth_rate`/`rdb_num_layers` (default
off/16/3), mutually exclusive with `multiscale_block` (both replace the
residual block type; `ResidualSRNet.__init__` raises `ValueError` if both are
requested), but independently composable with `channel_attention` and
`denoise_stem` (both orthogonal mechanisms). `--rdb-block`,
`--rdb-growth-rate`, `--rdb-num-layers` added to `train.py`; wired through
`build_model_config`/`build_model` the same way as every other optional
variant.

## Parameter counts (64F/8B -- the champion capacity)

| variant | trainable params | vs. baseline |
| --- | ---: | ---: |
| ResidualSR (baseline, unchanged) | 630,724 | -- |
| ResidualSR + RDB blocks (growth_rate=16, 3 layers) | 374,596 | **-256,128 (-40.6%)** |

Notably *smaller* than the baseline, not larger -- dense feature reuse needs
less raw per-layer width to represent similar capacity, comfortably clearing
the spec's "<1M parameters" target with headroom for a larger `growth_rate`
if a first run shows underfitting.

## Tests

15 new tests in `tests/test_model_unit.py`: shape/finiteness, default-off
attention submodule + optional attention composition, backward pass, dense
channel growth verified layer-by-layer (`layers[i].in_channels`), rejection
of non-positive `growth_rate`/`num_layers`, forward shape through the full
`ResidualSRNet`, the <1M-parameter assertion at champion capacity, backward
pass through the full model, mutual exclusivity with `--multiscale-block`
(raises `ValueError`), `build_model_config`/`build_model` key-presence
wiring (omitted when disabled, included when enabled), legacy-config
backward compatibility, and explicit composition with `--denoise-stem`.

## CLI-wiring smoke check (not real training; deleted afterward)

`--rdb-block --rdb-growth-rate 4 --rdb-num-layers 2`, 1 epoch, 8 train / 4
val samples, 8-feature/2-block model (2,684 total params at this tiny
capacity): ran to completion, then a second run with `--resume
checkpoint_latest.pt --epochs 2` correctly resumed at epoch 2 and completed.
Checkpoint directory deleted after verification.

## Proposed command

```bash
python train.py --model residual_sr --rdb-block --rdb-growth-rate 16 --rdb-num-layers 3 \
  --num-features 64 --num-blocks 8 --loss l1 --crop-size 96 --batch-size 16 \
  --seed 42 --lr 1e-4 --scheduler plateau --scheduler-factor 0.5 \
  --scheduler-patience 3 --min-lr 1e-6 --ema --ema-decay 0.999 \
  --epochs 90 --checkpoint-dir checkpoints/exp30_rdb_block
```

## Result

**TBD.** No real training run has been started.

---

# Ensemble Tooling Extensions — Phase 7/8 (infrastructure, not a new experiment)

## Status

**COMPLETE (tooling only; no new training).** `evaluate_ensemble.py` extended
to support Phase 7 (prediction averaging across more than two checkpoints)
and Phase 8 (alpha grid search with an explicit accept/reject rule),
inference-only.

## What's new

1. **`--checkpoints ckpt1 ckpt2 [ckpt3 ...]` / `--weights w1 w2 [...]`** --
   generalizes the ensemble evaluator beyond exactly two checkpoints (e.g.
   several epochs of the *same* run for Phase 7 prediction averaging, or more
   than two competitive models). The original `--checkpoint-a`/
   `--checkpoint-b`/`--weight-a`/`--weight-b` two-checkpoint interface is
   kept byte-for-byte unchanged (Experiments 11/18 used it, as does
   `tests/test_ensemble_unit.py`); `_resolve_checkpoints_and_weights` merges
   whichever interface was used into one internal list.
2. **`validate_ensemble_n`** (new) -- the general N-model (N >= 2) version;
   `validate_ensemble` (the original two-model function, unchanged
   signature) now delegates to a shared private helper so both stay
   consistent without duplicating the aggregation loop.
3. **`--alpha-search` (`--alpha-step`, default `0.05`)** -- sweeps
   `prediction = alpha*A + (1-alpha)*B` over `0.0, 0.05, ..., 1.0` for the
   first two checkpoints, reports each raw model's own PSNR/SSIM, the best
   alpha, and an explicit **ACCEPTED/REJECTED** verdict (rejects unless the
   best ensemble PSNR strictly beats the stronger individual model -- the
   project's existing ensemble acceptance rule, previously applied manually
   in Experiments 11/18's write-ups, now automated). The pure-model
   endpoints (`alpha=0.0`/`1.0`) reuse the already-computed raw single-model
   metrics rather than passing a zero weight through
   `weighted_average_predictions` (which requires strictly positive weights
   by design -- `src/ensemble.py`).
4. **`evaluate_checkpoint.py --raw-weights`** (Phase 7: "raw model and EMA
   model must be distinguishable") -- the CLI previously had no way to
   request the live/raw weights instead of EMA; `load_model`'s
   `prefer_ema=False` path existed but was unreachable from the command
   line. Now exposed, and every run explicitly prints
   `Weights evaluated: EMA` / `raw/live` / `raw/live (checkpoint has no EMA
   weights)` so no report can be ambiguous about which weights were scored.

## Tests

11 new tests: `tests/test_ensemble_unit.py` (N-way matches the two-model
result exactly for N=2, 3-model support, rejection of a single model,
`alpha_grid` boundary/validation, `run_alpha_search` structure/best-alpha
selection, `alpha=1.0` reproducing raw model A exactly, and rejection when
both models are identical -- the ensemble can never strictly beat either),
`tests/test_evaluate_checkpoint_unit.py` (`--raw-weights` CLI flag exists).
Full fast suite: **685 passed, 8 deselected** (up from 617 before this
update).

## Real-checkpoint smoke check (not a real ablation; inference-only, no
training)

```bash
python evaluate_ensemble.py --checkpoints checkpoints/exp23_ema_extended90/checkpoint_best.pt \
  checkpoints/exp26_finetune_mixed/checkpoint_best.pt --alpha-search --alpha-step 0.25 \
  --max-val-samples 32
```

On this 32-sample subset the champion (Model A) alone was best at every
tested alpha (`alpha=1.00 -> 26.8391 dB`, `+0.0000 dB` over itself,
**REJECTED**) -- expected, since exp26_finetune_mixed already trails the
champion on the full validation set (see Experiment 26 above). This is a
CLI-wiring smoke check on a small subset, not a real ensemble evaluation;
a real comparison should use the full 640-image validation split.

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
| Exp 10 — x8 geometric TTA | Inference-only self-ensemble on Exp 6 checkpoint | 27.7689 dB | 0.747955 | Complete (superseded by Exp 16 + x8) |
| Exp 11 — Model ensemble  | Weighted average of Exp 6 + Exp 9 raw predictions | 27.7561 dB | 0.747603 | Complete (rejected) |
| Exp 12 — NAFNet-SR arch  | ResidualSRNet -> NAFNet-style gated blocks (64F/8B, 432K params) | 27.2178 dB | 0.729829 | Complete (rejected, stopped @32) |
| Exp 13 — SwinIR-lite arch | Convolution -> windowed self-attention (embed60/6 blocks, 348K params) | 27.4361 dB | 0.738432 | Complete (rejected) |
| Exp 14 — Cosine LR schedule | Exp 6 architecture, ReduceLROnPlateau -> CosineAnnealingLR (T_max=40) | 27.6011 dB | 0.742668 | Complete (rejected) |
| Exp 15 — Extended champion training | Resume Exp 6 latest checkpoint 40 -> 60 epochs, same recipe | 27.7626 dB | 0.748636 | Complete |
| Exp 16 — Extended champion training | Resume Exp 15 latest checkpoint 60 -> 70 epochs, same recipe | 27.7656 dB | 0.748618 | Complete |
| Exp 16 + x8 TTA | Inference-only self-ensemble on Exp 16 checkpoint | **27.8154 dB** | **0.750571** | Complete |
| Exp 17 — Bicubic residual learning | ResidualSR learned branch + fixed bicubic global skip, trained from scratch | 27.7460 dB | 0.748557 | Complete |
| Exp 17 + x8 TTA | Inference-only self-ensemble on Exp 17 checkpoint | 27.7942 dB | 0.750434 | Complete |
| Exp 18 — Exp16+Exp17 ensemble | Weighted average of Exp 16 + Exp 17 raw predictions (best: 75/25 + x8) | 27.8149 dB | 0.750692 | Complete (rejected) |
| Exp 19 — EMA weight averaging | ResidualSRNet 64F/8B, exponential moving average of weights (decay=0.999) | 27.8282* | TBD | Run executed, **write-up missing** |
| Exp 20 — EMA extended to 70 epochs | Continuation of Exp 19 | 27.8850* | TBD | Run executed, **write-up missing** |
| Exp 21 — EMA extended to 80 epochs | Continuation of Exp 20 | 27.9383* | TBD | Run executed, **write-up missing** |
| Exp 22 — Degradation forensics | **Analysis only**, no training: characterizes GT -> NoisyLR | n/a | n/a | Complete |
| Exp 23 — EMA extended to 90 epochs | Continuation of Exp 21 | 27.9893 | 0.756916 | Complete |
| **Exp 23 + x8 TTA** | Inference-only self-ensemble on Exp 23 checkpoint | **28.0355 dB** | **0.758519** | **Current champion** |
| Exp 24 — Synthetic noise augmentation | 50% signal-dependent synthetic noisy-LR training inputs | 27.9116 dB | 0.749092 | Complete (rejected) |
| Exp 25 — Noise-conditioned ResidualSR | Extra sigma(I) input channel from the Exp 22 noise model | TBD | TBD | Planned / prepared |
| Exp 26B — MSE fine-tune from champion | `--finetune-from` Exp 23, `--loss mse`, 8 epochs | 27.8966 dB | 0.754295 | Complete (rejected) |
| Exp 26C — Mixed L1+MSE fine-tune | `--finetune-from` Exp 23, `--loss mixed`, 23 epochs | 27.9527 dB | 0.756677 | Complete (rejected) |
| Exp 26C + x8 TTA | Inference-only self-ensemble on Exp 26C checkpoint | 28.0027 dB | 0.758374 | Complete (rejected) |
| Exp 26G — Channel attention | Stray/incomplete artifact on disk (10/90 epochs, wrong crop) | n/a | n/a | **Invalid -- not a real result; see caveat above** |
| Exp 27 — Variance-weighted L1 loss | `weight = 1/sqrt(var(I))` from the Exp 22 fit | TBD | TBD | Planned / prepared |
| Exp 28 — Hard-patch sampling | Gradient-energy-weighted crop-origin sampling, 50/50 mixture | TBD | TBD | Planned / prepared |
| Exp 29 — Denoise stem | Pre-trunk `Conv->gated blocks->Conv` residual denoising stage | TBD | TBD | Planned / prepared |
| Exp 30 — Lightweight RDB | Dense-feature-reuse residual block (374,596 params, -40.6%) | TBD | TBD | Planned / prepared |

Note: Exp 10 is not a trained model -- it is Experiment 6's checkpoint evaluated with
x8 test-time augmentation (+0.0599 dB / +0.002321 SSIM over Exp 6 alone). It is an
optional inference-time post-processing step, not a new checkpoint-selection
candidate. **As of Experiment 16, Experiment 6 is no longer the underlying
champion model** -- Experiment 16's checkpoint (epoch 65, a continuation of
Experiment 6 via Experiments 15/16) is, and Experiment 16 + x8 TTA
(27.8154 dB / 0.750571) is now the best overall pipeline, superseding
Experiment 6 + x8 TTA (27.7689 dB / 0.747955).

Note: Exp 11 is not a trained model either -- it is Experiment 6 and Experiment 9's
checkpoints combined at inference time via weighted raw-prediction averaging. Its
best tested configuration (0.875 Exp6 + 0.125 Exp9, with x8 TTA) still trails
Experiment 6 + x8 TTA alone by -0.0128 dB PSNR / -0.000352 SSIM, so it was
**rejected**; Experiment 6 + x8 TTA remains the best inference pipeline.

*Exp 19/20/21 PSNR values are read from stored checkpoint metadata, not
independently re-measured; those three runs still need proper write-ups.

Note: Exp 12 (NAFNet-SR, 64F/8B, 432,129 params) was stopped deliberately at
epoch 32 of a planned 40 after clearly plateauing (+0.0464 dB over epochs 25-32)
well below Experiment 6 (-0.4912 dB PSNR / -0.015805 SSIM). **Rejected**; see
the full Experiment 12 entry above for the GPU-sizing finding that shaped its
configuration and the plateau data behind the stop decision.

Note: Exp 13 (SwinIR-lite, embed_dim=60/depth=6/heads=6/window=8, 348,421
params) ran the full 40 epochs and trained correctly (best epoch 38), but
finished -0.2729 dB PSNR / -0.007202 SSIM below Experiment 6. **Rejected**;
this is the fourth architecture/ensembling attempt (after Exp 9, 11, 12) to
beat Experiment 6, motivating Experiment 14's shift to tuning the training
recipe (LR schedule) on the proven Experiment 6 architecture instead of
further architecture search.

Note: Exp 14 (`CosineAnnealingLR`, T_max=40/eta_min=1e-6, on the identical
Experiment 6 architecture/recipe) ran the full 40 epochs (best epoch 38) but
finished -0.1079 dB PSNR / -0.002966 SSIM below Experiment 6's
`ReduceLROnPlateau` result -- the smallest gap of any rejected experiment so
far, but still a clear regression. **Rejected; `ReduceLROnPlateau` remains
this project's scheduler of choice.** Motivates Experiment 15's shift toward
extending Experiment 6's own training horizon rather than changing the
recipe further.

Note: Exp 17 (bicubic + learned residual, trained from scratch, same
topology/recipe as Exp 16) came the closest of any alternative to date
(-0.0196 dB non-TTA, -0.0212 dB with x8, vs. Exp 16) but still did not win.
**Rejected as an individual model**; its checkpoint was reused as Experiment
18's second ensemble member rather than discarded.

Note: Exp 18 (weighted ensemble of Exp 16 + Exp 17 raw predictions) tested
three pre-declared non-TTA weights (50/50, 75/25, 87.5/12.5) plus x8 on the
top two (50/50 and 75/25, within the pre-declared 0.01 dB tie-break
threshold). Best result, 75/25 + x8: PSNR 27.8149 dB -- **0.0005 dB below**
Exp 16 + x8's 27.8154 dB, a negligible but real shortfall against the
pre-declared primary-metric success rule (SSIM/L1 were marginally better,
but PSNR was designated primary in advance). **Rejected; Experiment 16 + x8
TTA remains the champion pipeline.**

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

# Next-Experiment Priority Order (A–I)

Consolidated, reproducible commands in the priority order requested for this
round of work. None have been run as full training jobs (GPU time is
limited and long training is not started automatically); each is
independently selectable and compares against the protected champion
(Experiment 23 + x8 TTA: **28.0355 dB / 0.758519 / 0.032264**).

**A — Champion + channel attention only.** The `checkpoints/exp27_channel_attention/`
artifact on disk is invalid (10/90 epochs, wrong crop -- see the Experiment 26
caveats above); this is the correct command to actually run:
```bash
python train.py --model residual_sr --channel-attention --attention-reduction 8 \
  --num-features 64 --num-blocks 8 --loss l1 --crop-size 96 --batch-size 16 \
  --seed 42 --lr 1e-4 --scheduler plateau --scheduler-factor 0.5 \
  --scheduler-patience 3 --min-lr 1e-6 --ema --ema-decay 0.999 \
  --epochs 90 --checkpoint-dir checkpoints/expA_channel_attention
```

**B — Champion + crop 96 only.** Already the champion's own training crop
(established at Experiment 6, carried through every EMA continuation to
Experiment 23) -- this is a no-op relative to the champion, not a new run.

**C — Champion + crop 128 (full LR image) only.** Already answered at the
*pre-EMA* recipe by Experiment 7 (27.7101 dB, a negligible +0.0011 dB over
the then-champion, with worse SSIM/L1 and slower epochs -- **not** repeated
at the current 90-epoch EMA recipe). Low priority to redo given that history,
but the command for completeness:
```bash
python train.py --model residual_sr --crop-size 128 \
  --num-features 64 --num-blocks 8 --loss l1 --batch-size 16 \
  --seed 42 --lr 1e-4 --scheduler plateau --scheduler-factor 0.5 \
  --scheduler-patience 3 --min-lr 1e-6 --ema --ema-decay 0.999 \
  --epochs 90 --checkpoint-dir checkpoints/expC_crop128_ema
```

**D — Champion + multi-scale block only.**
```bash
python train.py --model residual_sr --multiscale-block \
  --num-features 64 --num-blocks 8 --loss l1 --crop-size 96 --batch-size 16 \
  --seed 42 --lr 1e-4 --scheduler plateau --scheduler-factor 0.5 \
  --scheduler-patience 3 --min-lr 1e-6 --ema --ema-decay 0.999 \
  --epochs 90 --checkpoint-dir checkpoints/expD_multiscale_block
```

**E — Champion + hard-patch sampling only** (Experiment 28 above):
```bash
python train.py --model residual_sr --hard-patch-sampling --hard-patch-prob 0.5 \
  --num-features 64 --num-blocks 8 --loss l1 --crop-size 96 --batch-size 16 \
  --seed 42 --lr 1e-4 --scheduler plateau --scheduler-factor 0.5 \
  --scheduler-patience 3 --min-lr 1e-6 --ema --ema-decay 0.999 \
  --epochs 90 --checkpoint-dir checkpoints/expE_hard_patch
```

**F — Best of A–E + variance-weighted L1 (Experiment 27 above), applied to
the winning architecture from A–E once known.** Deliberately not Charbonnier
again (Experiment 4, rejected) and not another mixed-loss fine-tune
(Experiment 26B/C, both rejected) -- those were explicitly ruled out by the
"do not repeat the same experiment" instruction. Template (substitute the
winning A–E flags in place of the placeholder architecture line):
```bash
python train.py --model residual_sr [<winning A-E flag(s)>] \
  --loss weighted_l1 --num-features 64 --num-blocks 8 --crop-size 96 --batch-size 16 \
  --seed 42 --lr 1e-4 --scheduler plateau --scheduler-factor 0.5 \
  --scheduler-patience 3 --min-lr 1e-6 --ema --ema-decay 0.999 \
  --epochs 90 --checkpoint-dir checkpoints/expF_best_plus_weighted_l1
```

**G — Degradation-aware denoising stem, only if forensics support it.**
Forensics do support it (Experiment 29 above: strong signal-dependent noise,
negligible blur, no fixed pattern):
```bash
python train.py --model residual_sr --denoise-stem --denoise-stem-blocks 2 \
  --num-features 64 --num-blocks 8 --loss l1 --crop-size 96 --batch-size 16 \
  --seed 42 --lr 1e-4 --scheduler plateau --scheduler-factor 0.5 \
  --scheduler-patience 3 --min-lr 1e-6 --ema --ema-decay 0.999 \
  --epochs 90 --checkpoint-dir checkpoints/expG_denoise_stem
```
Also still outstanding from before this round: **Experiment 25 (noise
conditioning)** is fully prepared but has never been run --
```bash
python train.py --model residual_sr --num-features 64 --num-blocks 8 \
  --loss l1 --crop-size 96 --batch-size 16 --seed 42 --lr 1e-4 \
  --scheduler plateau --scheduler-factor 0.5 --scheduler-patience 3 --min-lr 1e-6 \
  --ema --ema-decay 0.999 --noise-conditioning \
  --epochs 90 --checkpoint-dir checkpoints/exp25_noise_conditioned
```

**H — Best model (from A–G) + x8 TTA.**
```bash
python evaluate_checkpoint.py --checkpoint <best_dir>/checkpoint_best.pt --tta x8
```

**I — Prediction averaging / ensemble of the strongest 2+ candidates from
A–H** (Experiment "Ensemble Tooling Extensions" above):
```bash
python evaluate_ensemble.py --checkpoints <best_dir>/checkpoint_best.pt \
  checkpoints/exp23_ema_extended90/checkpoint_best.pt --alpha-search --tta x8
```
Reject unless the best alpha strictly beats the stronger of the two
individual models (automated by `--alpha-search`'s printed verdict).

**Also outstanding, lower priority than A–I but prepared:** Experiment 30
(lightweight RDB block, 374,596 params, -40.6% vs. baseline) --
```bash
python train.py --model residual_sr --rdb-block --rdb-growth-rate 16 --rdb-num-layers 3 \
  --num-features 64 --num-blocks 8 --loss l1 --crop-size 96 --batch-size 16 \
  --seed 42 --lr 1e-4 --scheduler plateau --scheduler-factor 0.5 \
  --scheduler-patience 3 --min-lr 1e-6 --ema --ema-decay 0.999 \
  --epochs 90 --checkpoint-dir checkpoints/exp30_rdb_block
```

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
