# Experiment 21 — Dataset Degradation Forensics

Analysis only: **no model was trained, loaded, or evaluated.** All numbers
below describe the dataset's GT -> NoisyLR degradation, measured over the
canonical **3,200 training pairs** (GT 256x256 -> NoisyLR 128x128).

Generated: 2026-08-11T12:02:50+00:00

## 1. Which GT -> LR downsampling model best explains the observed LR?

| model | MAE | MSE | PSNR (dB) | correlation | bias |
| --- | ---: | ---: | ---: | ---: | ---: |
| `bicubic` **(best)** | 0.061243 | 0.008042 | 22.1080 | 0.88549 | +0.000008 |
| `bilinear` | 0.061713 | 0.008125 | 22.0616 | 0.88442 | +0.000008 |
| `area` | 0.061713 | 0.008125 | 22.0616 | 0.88442 | +0.000008 |
| `bicubic_antialias` | 0.063117 | 0.008449 | 21.8845 | 0.87908 | +0.000008 |
| `bilinear_antialias` | 0.064617 | 0.008799 | 21.6811 | 0.87449 | +0.000008 |
| `subsample_odd` | 0.068843 | 0.010336 | 21.0144 | 0.85360 | -0.000014 |
| `nearest` | 0.068843 | 0.010336 | 21.0144 | 0.85360 | -0.000014 |
| `subsample_even` | 0.068898 | 0.010377 | 21.0280 | 0.85382 | +0.000029 |

These are **degradation-model agreement scores, not restoration performance** —
both sides of each comparison are known quantities.

**Best model: `bicubic`.** All residual analysis below uses it.

## 2. Systematic gain / bias

- Global dataset-wide fit `LR ≈ a·downsample(GT) + b`: **a = 0.994675, b = +0.002316**
- Per-image gain: mean 0.988045, std 0.017182 (p5 0.9580, p95 1.0049)
- Per-image bias: mean +0.004849, std 0.008135
- Mean residual MSE before affine correction: 0.008042
- Mean residual MSE after per-image affine correction: 0.008031 (**0.14%** reduction)

## 3. Residual noise magnitude

- mean **+0.000008**, std **0.089679**, variance 0.008042
- range [-0.7646, 1.1805]
- skewness +0.4080, excess kurtosis +3.4585
- percentiles: p0.1 -0.3508, p1 -0.2362, p5 -0.1442, p25 -0.0412, p50 -0.0018, p75 +0.0367, p95 +0.1523, p99 +0.2702, p99.9 +0.4377
- per-image residual std: mean 0.084395, range [0.009296, 0.205340]

## 4-5. Is the noise iid / signal dependent?

Fitted `var(I) = -6.1886e-05 + 0.00653067·I + 0.0201442·I²`  (R² = 0.9995)

| clean-LR intensity | residual mean | residual std | count |
| ---: | ---: | ---: | ---: |
| 0.013 | +0.00167 | 0.01196 | 1,470,941 |
| 0.113 | +0.00193 | 0.02797 | 1,704,374 |
| 0.213 | +0.00157 | 0.04664 | 1,611,265 |
| 0.312 | +0.00087 | 0.06383 | 1,507,235 |
| 0.413 | +0.00009 | 0.07899 | 1,479,782 |
| 0.512 | -0.00057 | 0.09349 | 1,423,231 |
| 0.613 | -0.00142 | 0.10719 | 1,307,019 |
| 0.713 | -0.00209 | 0.12112 | 1,101,604 |
| 0.812 | -0.00204 | 0.13471 | 965,134 |
| 0.913 | -0.00171 | 0.15070 | 809,478 |

Noise std grows from ~0.0120 in the darkest bin to ~0.1631 in the brightest — **strongly signal dependent, not homoscedastic**.

## 6. Spatial correlation

| offset (dy,dx) | normalized autocorrelation |
| --- | ---: |
| (0,1) | -0.04435 |
| (0,2) | +0.00041 |
| (0,3) | +0.00543 |
| (0,4) | +0.00487 |
| (0,8) | +0.00264 |
| (1,0) | -0.05109 |
| (1,1) | +0.00242 |
| (2,0) | -0.00103 |
| (2,2) | +0.00027 |
| (3,0) | +0.00466 |
| (4,0) | +0.00484 |
| (8,0) | +0.00321 |

## 7. Fixed-pattern structure

- residual mean map: min -0.007057, max +0.006703, std across pixels 0.001598
- expected std of a pure-noise mean map (σ/√N): 0.001585
- ratio observed/expected: **1.008**
- row-mean spread 0.000147, column-mean spread 0.000156

## 8. Frequency domain

- radial spectrum low/high ratio: 0.8598
- max/median radial power ratio: 1.1436
- horizontal vs vertical power ratio: 1.0259

## 9. Edge / texture dependence

| gradient magnitude bin | mean abs residual | count |
| --- | ---: | ---: |
| [0, 0.002) | 0.06483 | 2,254,498 |
| [0.002, 0.005) | 0.05887 | 4,266,436 |
| [0.005, 0.01) | 0.05558 | 5,896,609 |
| [0.01, 0.02) | 0.05608 | 8,768,232 |
| [0.02, 0.05) | 0.05857 | 14,140,591 |
| [0.05, 0.1) | 0.06179 | 9,104,903 |
| [0.1, 0.2) | 0.06951 | 5,642,277 |
| [0.2, 0.5) | 0.08895 | 2,317,988 |
| [0.5, 1) | 0.13274 | 37,246 |

## 10. Pre-downsampling blur

| Gaussian sigma | residual MSE |
| ---: | ---: |
| 0.0 | 0.007267 |
| 0.2 | 0.007267 |
| 0.3 | 0.007262 |
| 0.4 **(best)** | 0.007243 |
| 0.5 | 0.007293 |
| 0.7 | 0.007593 |
| 0.9 | 0.007972 |
| 1.2 | 0.008599 |
| 1.6 | 0.009415 |

Best sigma **0.4**; improvement over no blur: 0.328%.

## 11. Repeated scenes

- exact byte-identical GT groups: **0** (3,200 unique hashes / 3,200 pairs)
- average-hash *candidate* groups: 180 (candidates only — an 8x8 hash also collides for unrelated images)
- **confirmed** near-duplicate pairs (GT MSE < 0.001): **143**
- confirmed repeated-scene groups: **119** covering **250 images**, sizes {'2': 107, '3': 12}
- within-group mean GT MSE 0.000108 vs within-group mean **LR** MSE 0.014416 — same clean scene, **independent noise realizations**
- filename structure: all confirmed pairs sit at ID gap ≤ 2 (max 2, median 1, 100% within 2) — **filenames encode the grouping**
- groups spanning train *and* validation: **36**, leaking **38 of 640** validation images (5.9%)

## 12. Degradation regimes

- per-image residual std 2-means: centers [0.06270743731110354, 0.11100153042972323], sizes [1763, 1437], variance explained 0.6277
- correlation(per-image residual std, mean intensity) = **+0.8989**
- correlation(per-image residual std, gradient energy) = +0.2606

## 13. Train vs validation

| statistic | train | validation | Δ |
| --- | ---: | ---: | ---: |
| residual_std | 0.084428 | 0.084260 | -0.000168 |
| residual_mse | 0.008054 | 0.007997 | -0.000057 |
| gain | 0.988267 | 0.987155 | -0.001112 |
| bias | 0.004719 | 0.005370 | +0.000651 |
| estimate_mean | 0.433528 | 0.433532 | +0.000004 |
| gradient_energy | 0.007110 | 0.007497 | +0.000386 |

Train n=2,560, validation n=640 (canonical split, unchanged).

## Plots

- `results/degradation_analysis/residual_mean_map.png` — residual mean map
- `results/degradation_analysis/residual_std_map.png` — residual std map
- `results/degradation_analysis/residual_power_spectrum.png` — residual power spectrum
- `results/degradation_analysis/noise_vs_intensity.png` — noise vs intensity
- `results/degradation_analysis/residual_row_column_profiles.png` — residual row column profiles
- `results/degradation_analysis/per_image_noise_distribution.png` — per image noise distribution

## Conclusions and Experiment 22 candidates

### What the evidence says

1. **Downsampling model** — `bicubic` explains the clean component of
   NoisyLR best, but only marginally better than the neighbouring resampling kernels. The
   spread across all candidates is small relative to the residual itself, which means the
   residual is dominated by **noise, not by resampling-kernel mismatch**.
2. **Gain / bias** — the global affine fit is a = 0.994675, b = +0.002316,
   i.e. essentially identity. Per-image affine correction reduces residual MSE by only
   0.14%. **There is no systematic gain or
   offset worth correcting.**
3. **Noise magnitude** — residual std is 0.0897 over the whole dataset, with excess
   kurtosis +3.459 (heavier-tailed than Gaussian). For scale, that is roughly
   2.8x the validation L1 the current champion pipeline already
   achieves (0.032607) — the corruption to be removed is far larger than the error that
   remains, confirming the task is **noise-dominated rather than resolution-dominated**.
4. **Not iid, strongly signal dependent** — residual std climbs from 0.0120 in the darkest
   intensity bin to 0.1631 in the brightest, a **13.6x** spread. The fitted variance model
   `var(I) = -6.189e-05 + 0.006531·I + 0.02014·I²` achieves
   R² = 0.9995, describing predominantly **multiplicative/speckle-like** (variance grows ~quadratically with intensity).
   **This is the single most exploitable structure found.**
5. **Spatially near-white** — the largest |autocorrelation| at any tested offset is
   0.0511. The small negative lag-1 terms are an expected artifact of estimating the
   clean signal by smoothing, not evidence of correlated noise. Treat the noise as spatially
   independent.
6. **No usable fixed pattern** — the dataset-average residual map has std 0.001598
   versus 0.001585 expected from pure noise averaging
   (ratio 1.01). There is no repeatable per-sensor-position offset to subtract.
7. **No pre-downsampling blur** — the best Gaussian sigma is 0.4, improving MSE by
   0.328%. The degradation is **not** blur-then-downsample.
8. **Repeated scenes DO exist** — there are zero byte-identical GTs, but
   **119 confirmed repeated-scene groups covering
   250 images** (GT MSE <
   0.001; within-group GT MSE
   0.000108 against within-group *LR* MSE
   0.014416, i.e. one clean scene observed under
   independent noise draws). **Every** confirmed pair sits at consecutive-ish filename indices
   (max ID gap 2), so the grouping is recoverable directly from
   filenames. Critically, 36 of these
   groups straddle the canonical split, giving
   38 of
   640 validation images
   (5.9%) a near-identical twin in train.
9. **One continuous degradation regime** — per-image residual std correlates
   +0.899 with mean image intensity. The apparent
   spread in per-image noise level is explained by **image brightness**, not by discrete noise-level
   classes. 2-means on residual std explains only
   0.628 of variance and shows no separation gap.
10. **Train and validation match distributionally** — every compared degradation statistic differs by a
    negligible margin (see the table above), so the two splits are drawn from the same process. The one
    caveat is the 5.9% twin overlap in (8):
    absolute validation numbers are very slightly optimistic. This does **not** invalidate any
    cross-experiment comparison in this log, since every experiment used the identical split, and the
    split is deliberately left unchanged.

### Ranked strategies for Experiment 22

**1. Variance-stabilizing transform / noise-aware loss — expected payoff: HIGH.**
The measured noise variance spans 13.6x across the intensity range with R² = 0.9995
against a simple closed-form model. Plain L1 implicitly assumes homoscedastic noise, so it
currently over-weights bright, noisy pixels and under-weights dark, clean ones — where most of
the recoverable detail actually lives. Two concrete variants: (a) train on a
generalized-Anscombe-transformed signal and invert at inference; (b) keep the pixel space but
weight the loss by 1/√var(I) using the fitted coefficients. This directly targets the strongest
structure in the data and is cheap to implement on top of the existing champion recipe.

**2. Signal-dependent synthetic-noise augmentation — expected payoff: MEDIUM-HIGH.**
With `var(I)` now measured to R² = 0.9995, unlimited extra
(GT, synthetic-NoisyLR) pairs can be synthesized by drawing noise from the fitted
signal-dependent model, multiplying the effective 2,560-sample training set without any new
labels. The dataset is small and the corruption process is now characterized, which is exactly
the regime where this works. Held back from HIGH only because the fit captures the *aggregate*
variance, while the residual's excess kurtosis of +3.459 shows the true
per-pixel distribution is heavier-tailed than a Gaussian of the same variance — so the synthetic
corruption would be slightly wrong in its tails. Calibrating against the measured percentiles in
§3 would mitigate that.

**3. Scene-group-aware training and validation — expected payoff: MEDIUM.**
119 repeated-scene groups covering
250 images are now identified, and they are recoverable from
filenames alone (all confirmed pairs within ID gap 2). Two distinct uses:
(a) *modelling* — the multiple independent noise draws over one clean scene permit
multi-observation averaging or a consistency/Noise2Noise-style term that plain paired L1 cannot
express; (b) *measurement hygiene* — 36
groups currently straddle the split, so
5.9% of validation images have a
near-identical twin in train. A group-aware split would give a slightly harder but cleaner
validation signal. MEDIUM because only
7.8% of the dataset is
involved, capping the achievable gain. **Do not change the canonical split casually** — doing so
makes new numbers incomparable with Experiments 1-20.

**Also worth noting — self-supervised objectives are statistically admissible here.** The noise is
spatially near-white (max |autocorrelation| 0.0511) and zero-mean
(+0.000008), which is precisely the condition blind-spot / Noise2Void and
Stein-unbiased-risk methods require. Ranked below the three above only because paired GT is already
available, so these would add regularization rather than new information.

**Explicitly deprioritized — LOW payoff, with reasons:** fixed-pattern subtraction (no pattern
exists, ratio 1.01); deblurring or kernel estimation (best sigma
0.4, only 0.328% gain); per-image gain/bias calibration
(0.14% MSE reduction); degradation-regime
classification (no discrete clusters — apparent spread is just brightness); and further
resampling-kernel search (all candidates within
29%
of each other, and all far below the noise floor).
