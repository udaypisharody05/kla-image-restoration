"""Deterministic discovery and identifier-based pairing for extracted datasets.

Assumption: paired files share their complete filename stem. Extensions may differ.
macOS archive metadata (``__MACOSX``, ``.DS_Store``, and ``._*``) is never data.
"""

from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

from .io_utils import SUPPORTED_EXTENSIONS


@dataclass(frozen=True, order=True)
class ImagePair:
    """A training sample linked by a stable filename identifier."""

    pair_id: str
    input_path: Path
    target_path: Path


@dataclass(frozen=True)
class DatasetLayout:
    root: Path
    train_input_dir: Path
    target_dir: Path
    test_input_dir: Path


@dataclass(frozen=True)
class PairDiscovery:
    pairs: tuple[ImagePair, ...]
    missing_targets: tuple[str, ...]
    missing_inputs: tuple[str, ...]
    duplicate_input_ids: dict[str, tuple[str, ...]]
    duplicate_target_ids: dict[str, tuple[str, ...]]


def is_data_file(path: Path) -> bool:
    return (
        path.is_file()
        and path.suffix.lower() in SUPPORTED_EXTENSIONS
        and not path.name.startswith("._")
        and ".DS_Store" not in path.name
        and "__MACOSX" not in path.parts
    )


def image_files(directory: Path) -> list[Path]:
    return sorted((p for p in Path(directory).rglob("*") if is_data_file(p)), key=lambda p: p.as_posix())


def _score(directory: Path, role: str) -> int:
    name = directory.name.lower().replace("-", "").replace("_", "")
    ancestors = " ".join(p.name.lower() for p in directory.parents)
    score = len(image_files(directory))
    if role == "target" and any(x in name for x in ("gt", "groundtruth", "target", "clean", "hr")):
        score += 1_000_000
    if role == "input" and any(x in name for x in ("noisylr", "degraded", "input", "lr")):
        score += 1_000_000
    if role == "test" and "test" in ancestors and any(x in name for x in ("noisylr", "degraded", "input", "lr")):
        score += 2_000_000
    if role != "test" and "train" in ancestors:
        score += 500_000
    return score


def discover_layout(data_dir: Path) -> DatasetLayout:
    """Find leaf image directories from names plus their actual contents."""
    data_dir = Path(data_dir).resolve()
    if not data_dir.is_dir():
        raise FileNotFoundError(f"Data directory does not exist: {data_dir}")
    candidates = [d for d in data_dir.rglob("*") if d.is_dir() and image_files(d)]
    leaves = [d for d in candidates if any(is_data_file(p) for p in d.iterdir())]
    if len(leaves) < 3:
        raise RuntimeError(f"Could not find training input, target, and test directories under {data_dir}")
    target = max(leaves, key=lambda d: (_score(d, "target"), d.as_posix()))
    test = max((d for d in leaves if d != target), key=lambda d: (_score(d, "test"), d.as_posix()))
    train = max((d for d in leaves if d not in (target, test)), key=lambda d: (_score(d, "input"), d.as_posix()))
    common = Path(Path(*Path(train).parts[: len(Path(train).parts)]))
    common = Path(__import__("os").path.commonpath([train, target, test]))
    return DatasetLayout(common, train, target, test)


def _by_id(directory: Path) -> dict[str, list[Path]]:
    grouped: dict[str, list[Path]] = defaultdict(list)
    for path in image_files(directory):
        grouped[path.stem].append(path)
    return grouped


def discover_pairs(layout: DatasetLayout) -> PairDiscovery:
    inputs, targets = _by_id(layout.train_input_dir), _by_id(layout.target_dir)
    duplicate_inputs = {k: tuple(str(p) for p in v) for k, v in inputs.items() if len(v) > 1}
    duplicate_targets = {k: tuple(str(p) for p in v) for k, v in targets.items() if len(v) > 1}
    valid_ids = sorted(inputs.keys() & targets.keys() - duplicate_inputs.keys() - duplicate_targets.keys())
    pairs = tuple(ImagePair(i, inputs[i][0], targets[i][0]) for i in valid_ids)
    return PairDiscovery(
        pairs,
        tuple(sorted(inputs.keys() - targets.keys())),
        tuple(sorted(targets.keys() - inputs.keys())),
        duplicate_inputs,
        duplicate_targets,
    )
