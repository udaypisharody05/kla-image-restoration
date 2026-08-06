"""Lossless array loading helpers for scientific and conventional image files."""

from pathlib import Path

import imageio.v3 as iio
import numpy as np

SUPPORTED_EXTENSIONS = {".npy", ".npz", ".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}


def load_image_array(path: Path) -> np.ndarray:
    """Load *path* as an array without normalization, clipping, or dtype conversion.

    An NPZ archive must contain exactly one array: silently choosing among multiple
    members is unsafe because their meaning is unknown. Pickled NumPy objects are
    deliberately rejected.
    """
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Image file does not exist: {path}")
    suffix = path.suffix.lower()
    if suffix not in SUPPORTED_EXTENSIONS:
        raise ValueError(f"Unsupported image extension {suffix!r}: {path}")
    try:
        if suffix == ".npy":
            array = np.load(path, allow_pickle=False)
        elif suffix == ".npz":
            with np.load(path, allow_pickle=False) as archive:
                keys = archive.files
                if len(keys) != 1:
                    raise ValueError(
                        f"NPZ file must contain exactly one array; found {keys}: {path}"
                    )
                array = np.array(archive[keys[0]], copy=True)
        else:
            array = iio.imread(path)
    except (OSError, ValueError, TypeError) as exc:
        raise ValueError(f"Failed to load image array {path}: {exc}") from exc
    array = np.asarray(array)
    if array.ndim < 2:
        raise ValueError(f"Expected at least two dimensions, got {array.shape}: {path}")
    if array.dtype == object:
        raise ValueError(f"Object arrays are not supported: {path}")
    return array
