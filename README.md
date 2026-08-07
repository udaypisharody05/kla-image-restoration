# KLA Image Restoration

Utilities for inspecting and validating the paired image-restoration dataset.

## Installation

The project is tested with **Python 3.12.13**. Other Python versions have not been verified.

On Windows PowerShell:

```powershell
python -m venv .venv
.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

On macOS or Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

`requirements.txt` contains the exact dependency versions used for verification, including pytest for this small repository's development and test workflow.

## Quick Start with Docker

Install Docker Desktop (Windows/macOS) or Docker Engine (Linux). Python and the project dependencies do not need to be installed on the host.

Build the image from the repository root:

```bash
docker build -t semicon-restoration .
```

Run the portable unit tests (the image's default command):

```bash
docker run --rm semicon-restoration
```

The dataset is not included in Git or in the Docker image. Download and extract it separately under the repository's `data/` directory using the structure in [Dataset Setup](#dataset-setup), then mount it read-only when running dataset inspection. Mount `results/` as well to persist generated reports on the host.

### Windows PowerShell

```powershell
docker run --rm `
  -v "${PWD}/data:/app/data:ro" `
  -v "${PWD}/results:/app/results" `
  semicon-restoration `
  python inspect_dataset.py --data-dir /app/data --results-dir /app/results --max-samples 100
```

### macOS/Linux

```bash
docker run --rm \
  --user "$(id -u):$(id -g)" \
  -v "$(pwd)/data:/app/data:ro" \
  -v "$(pwd)/results:/app/results" \
  semicon-restoration \
  python inspect_dataset.py --data-dir /app/data --results-dir /app/results --max-samples 100
```

The macOS/Linux command uses your host user ID so reports created in the bind-mounted `results/` directory remain editable by your account. Docker Desktop handles bind-mount permissions for the PowerShell command.

Commands after the image name override the default, so other utilities work the same way. For example, `python visualize_samples.py --data-dir /app/data` runs sample visualization. The existing `SEMICON_DATA_DIR` override is also supported; set it with Docker's `-e` option when mounting the dataset somewhere other than `/app/data`.

## Dataset Setup

The full hackathon dataset is not committed to Git and is not required for development or unit testing. Download it separately and extract it without changing its internal structure so it appears under:

```text
data/Data-public/train/train/NoisyLR/
data/Data-public/train/train/GT/
data/Data-public/Test_NoisyLR/NoisyLR/
```

The scripts discover the dataset recursively, so `SEMICON_DATA_DIR` may instead point to another directory containing the extracted structure.

## Running Tests

Unit tests create tiny NumPy arrays in pytest temporary directories. They require no external dataset or network access:

```bash
pytest -m "not integration"
```

Integration tests exercise the same discovery, loading, geometry, and reporting behavior against the full hackathon dataset. By default they look under `<repository-root>/data`, independently of the current working directory:

```bash
pytest -m integration
```

To keep the dataset elsewhere, set `SEMICON_DATA_DIR` to either an absolute path or a path relative to the repository root. For example, in PowerShell:

```powershell
$env:SEMICON_DATA_DIR = "D:\datasets\semicon"
pytest -m integration
```

On Bash-compatible shells:

```bash
SEMICON_DATA_DIR=/datasets/semicon pytest -m integration
```

Run every available test with:

```bash
pytest
```

If the full dataset is absent or its structure is invalid, integration tests report a clear skip reason while unit tests continue to run and pass.

## Bicubic Validation Baseline

The first reproducible reference is a classical 2x bicubic interpolation baseline. It uses the existing deterministic NoisyLR-to-GT pairing and splits the pairs into 80% training and 20% validation with seed `42` by default. Only the validation subset is evaluated; the competition test set has no GT and is not used.

```bash
python evaluate_baseline.py \
  --data-dir data/Data-public \
  --val-fraction 0.2 \
  --seed 42
```

The command reports PSNR (dB), grayscale SSIM, interpolation-only CPU timing, and throughput, and saves `results/bicubic_baseline.json`. The raw float32 bicubic result is retained unchanged. By default, only the prediction passed to metrics is clipped to `[0,1]`, because GT represents valid intensities in that range; use `--no-clip-prediction` to measure without metric-time clipping.

LPIPS is available through `--lpips` only when a compatible optional PyTorch/LPIPS installation and its pretrained weights are already available. The grayscale metric input is replicated to three channels and mapped from `[0,1]` to `[-1,1]` inside the LPIPS adapter only. The default dependency set intentionally excludes this large, download-dependent stack; if unavailable, the baseline completes and records LPIPS as unavailable rather than failing.

This baseline establishes the performance that future learned restoration models should beat.

## PyTorch Data Pipeline

The lazy PyTorch datasets reuse the repository's canonical discovery and seeded split. They store paths and metadata only; arrays are loaded when a sample is indexed.

```python
from pathlib import Path

from src.dataset import PairedRestorationDataset, create_dataloader
from src.dataset_discovery import discover_layout, discover_pairs
from src.splits import split_pairs

layout = discover_layout(Path("data"))
pairs = discover_pairs(layout).pairs
train_pairs, validation_pairs = split_pairs(
    pairs,
    val_fraction=0.2,
    seed=42,
)

train_dataset = PairedRestorationDataset(train_pairs)
validation_dataset = PairedRestorationDataset(validation_pairs)

train_loader = create_dataloader(
    train_dataset,
    batch_size=8,
    shuffle=True,
    seed=42,
)
validation_loader = create_dataloader(
    validation_dataset,
    batch_size=8,
    shuffle=False,
)
```

Each paired sample contains `input`, `target`, and `filename`. Input tensors are raw NoisyLR `torch.float32` values in `[1,H,W]` format, including legitimate values below 0 or above 1. GT tensors are `torch.float32 [1,2H,2W]`. No clipping, normalization, resizing, cropping, padding, or augmentation occurs.

For competition inputs without GT, construct `RestorationTestDataset(image_files(layout.test_input_dir))`; samples contain only `input` and `filename`. `create_dataloader` supports `batch_size`, `shuffle`, `num_workers`, `pin_memory`, `drop_last`, and an optional shuffle `seed`.

## Training Preprocessing

Training uses a spatially aligned random crop followed by paired geometric augmentation:

```text
128x128 LR  -> random 64x64 LR crop at (y, x)
256x256 GT  -> corresponding 128x128 GT crop at (2y, 2x)
             -> identical paired flips/right-angle rotation
```

```python
from src.transforms import create_training_transform

training_transform = create_training_transform(
    crop_size=64,
    scale=2,
    augment=True,
)
training_dataset = PairedRestorationDataset(
    train_pairs,
    scale=2,
    transform=training_transform,
)

# Validation stays deterministic and directly comparable with the full-image
# bicubic baseline: no crop and no augmentation.
validation_dataset = PairedRestorationDataset(validation_pairs, scale=2)
```

The default augmentation policy independently applies horizontal and vertical flips with probability 0.5 and uniformly selects a rotation from 0, 90, 180, or 270 degrees. No interpolation is used. There are no intensity changes, clipping, normalization, resizing, padding, or quantization; raw float32 NoisyLR values are preserved.

By default transforms use PyTorch's process-local RNG, which DataLoader workers seed in the standard way. Pass `seed=` or a `torch.Generator` to `create_training_transform` for deterministic single-worker tests or runs. Generator state advances on every access, so a sample is not permanently assigned one crop. Validation should continue using `transform=None` and complete `128x128`/`256x256` images so neural metrics remain comparable with the bicubic PSNR and SSIM baseline.
