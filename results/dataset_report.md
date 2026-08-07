# Dataset inspection report

Generated: 2026-08-06T17:16:07.867402+00:00

## Layout

- Dataset root: `data/Data-public`
- Training inputs: `train/train/NoisyLR`
- Ground truths: `train/train/GT`
- Test inputs: `Test_NoisyLR/NoisyLR`
- Physical files: 13606
- Extracted size: 1026.90 MiB

## Pairing

- Valid pairs: 3200 (3200 inputs, 3200 targets)
- Test images: 400
- Convention: exact complete filename stem match
- Missing targets: 0
- Missing inputs: 0
- Duplicate input IDs: 0
- Duplicate target IDs: 0

## Array findings

### Training degraded

Inspected 3200 of 3200 files.

- Shapes: `{'[128, 128]': 3200}`; dtypes: `{'float32': 3200}`; channels: `{'1': 3200}`; grayscale: True
- Min/max: -0.278563052 / 2.158005; mean/std: 0.433536288 / 0.284786611
- Below 0: 149367 (0.284895%); above 1: 1629724 (3.108452%)
- Outside [0,1]: 1779091 (3.393347%)
- NaN/Inf: 0 / 0
- Constant/nearly constant: 0; load failures: 0

### Ground truth

Inspected 3200 of 3200 files.

- Shapes: `{'[256, 256]': 3200}`; dtypes: `{'float32': 3200}`; channels: `{'1': 3200}`; grayscale: True
- Min/max: 0 / 1; mean/std: 0.433528441 / 0.27264606
- Below 0: 0 (0.000000%); above 1: 0 (0.000000%)
- Outside [0,1]: 0 (0.000000%)
- NaN/Inf: 0 / 0
- Constant/nearly constant: 0; load failures: 0

### Test degraded

Inspected 400 of 400 files.

- Shapes: `{'[128, 128]': 400}`; dtypes: `{'float32': 400}`; channels: `{'1': 400}`; grayscale: True
- Min/max: -0.224880666 / 2.15801597; mean/std: 0.442742028 / 0.284269049
- Below 0: 43262 (0.660126%); above 1: 201860 (3.080139%)
- Outside [0,1]: 245122 (3.740265%)
- NaN/Inf: 0 / 0
- Constant/nearly constant: 0; load failures: 0

## Geometry

- Scale factors: `{'2x2': 3200}`
- Consistent: True

## Warnings

- None
