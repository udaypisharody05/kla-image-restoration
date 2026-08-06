"""Save deterministic side-by-side visualizations of valid restoration pairs."""

import argparse
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from src.dataset_discovery import discover_layout, discover_pairs
from src.io_utils import load_image_array


def _display(array: np.ndarray) -> np.ndarray:
    return array if array.ndim == 2 else array[..., 0]


def create_visualization(data_dir: Path, num_pairs: int, output: Path) -> None:
    pairs = discover_pairs(discover_layout(data_dir)).pairs[:num_pairs]
    if not pairs:
        raise RuntimeError("No valid training pairs were discovered")
    fig, axes = plt.subplots(len(pairs), 2, figsize=(10, 4 * len(pairs)), squeeze=False)
    for row, pair in enumerate(pairs):
        for col, (label, path) in enumerate((("Degraded", pair.input_path), ("Ground truth", pair.target_path))):
            array = load_image_array(path)
            finite = array[np.isfinite(array)]
            if not finite.size:
                raise ValueError(f"Cannot visualize array without finite values: {path}")
            if col == 0:
                vmin, vmax = np.percentile(finite, [1, 99])
            else:
                vmin, vmax = float(finite.min()), float(finite.max())
            axes[row, col].imshow(_display(array), cmap="gray", vmin=vmin, vmax=vmax)
            axes[row, col].set_title(f"{label}: {pair.pair_id}\n{array.shape}, min={finite.min():.5g}, max={finite.max():.5g}")
            axes[row, col].axis("off")
    fig.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=150, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("data"))
    parser.add_argument("--num-pairs", type=int, default=5)
    parser.add_argument("--output", type=Path, default=Path("results/sample_pairs.png"))
    args = parser.parse_args()
    if args.num_pairs < 1: parser.error("--num-pairs must be positive")
    create_visualization(args.data_dir, args.num_pairs, args.output)
    print(f"Saved {args.output}")


if __name__ == "__main__":
    main()
