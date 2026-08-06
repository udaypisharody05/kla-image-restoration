# KLA Image Restoration

Utilities for inspecting and validating the paired image-restoration dataset.

## Dataset Setup

The full hackathon dataset is not required for development or unit testing. To use it for inspection or integration testing, download and extract it without changing its internal structure so it appears under:

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
