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
