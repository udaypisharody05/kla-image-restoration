# Dataset inspection report

Generated: 2026-08-07T16:26:10.640505+00:00

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

Inspected 100 of 3200 files.

- Shapes: `{'[128, 128]': 100}`; dtypes: `{'float32': 100}`; channels: `{'1': 100}`; grayscale: True
- Min/max: -0.131092951 / 1.8896383; mean/std: 0.417404399 / 0.29574788
- Below 0: 9537 (0.582092%); above 1: 55503 (3.387634%)
- Outside [0,1]: 65040 (3.969727%)
- NaN/Inf: 0 / 0
- Constant/nearly constant: 0; load failures: 0

### Ground truth

Inspected 100 of 3200 files.

- Shapes: `{'[256, 256]': 100}`; dtypes: `{'float32': 100}`; channels: `{'1': 100}`; grayscale: True
- Min/max: 0 / 1; mean/std: 0.417487799 / 0.284305691
- Below 0: 0 (0.000000%); above 1: 0 (0.000000%)
- Outside [0,1]: 0 (0.000000%)
- NaN/Inf: 0 / 0
- Constant/nearly constant: 0; load failures: 0

### Test degraded

Inspected 100 of 400 files.

- Shapes: `{'[128, 128]': 100}`; dtypes: `{'float32': 100}`; channels: `{'1': 100}`; grayscale: True
- Min/max: -0.15555042 / 2.15801597; mean/std: 0.447342668 / 0.283826703
- Below 0: 8257 (0.503967%); above 1: 50883 (3.105652%)
- Outside [0,1]: 59140 (3.609619%)
- NaN/Inf: 0 / 0
- Constant/nearly constant: 0; load failures: 0

## Geometry

- Scale factors: `{'2x2': 100}`
- Consistent: True

## Warnings

- None
