# Dataset notes

Verified dataset documentation is recorded in `results/dataset_report.md` and `results/dataset_report.json`.

## Observed facts

- Root: `data/Data-public`.
- Training inputs: `train/train/NoisyLR`; ground truths: `train/train/GT`.
- Test inputs: `Test_NoisyLR/NoisyLR`.
- There are 3,200 complete training pairs and 400 test inputs. No pair is missing, duplicated, unmatched, or corrupt.
- Samples use `.npy`, `float32`, grayscale arrays. Inputs are 128 by 128; targets are 256 by 256; the spatial scale is consistently 2 by 2.
- Training degraded values span -0.278563052 to 2.158005. Of 52,428,800 values, 1,779,091 (3.393347%) lie outside `[0,1]`.
- Test degraded values span -0.224880666 to 2.15801597; 3.740265% lie outside `[0,1]`.
- Ground-truth values span exactly 0 to 1, with no out-of-range values.
- No array contains NaN/Inf or is constant/nearly constant.
- `__MACOSX`, `.DS_Store`, and `._*` files are archive metadata, not samples.

## Assumptions

- Exact complete filename stems are pair identifiers. All 3,200 observed pairs support this.
- Array axes are spatial `(height, width)` because every array is two-dimensional.

## Preprocessing cautions and future decisions

- Preserve raw `float32` values. Never clip, normalize, resize, or overwrite source arrays in the generic loader.
- Keep metadata sidecars out of discovery.
- Before implementing a PyTorch dataset, choose crop sampling and augmentation policies, and explicitly decide whether clipping belongs inside the model pipeline. Preserve the fixed 2x supervision relationship.
