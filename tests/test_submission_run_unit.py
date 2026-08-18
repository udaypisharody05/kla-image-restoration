"""Compliance tests for the final evaluator package,
``submission/SuperconductorSemistars/run.py``.

Fast and dataset-free: builds tiny synthetic ``.npy`` inputs so none of
these tests require the full 400-image official dataset or a GPU. Covers
the organizer's hard requirements: the ``python run.py <in> <out>`` CLI
contract, exact filename mapping, correct 2x output shape derived from each
input (not hard-coded), output value range/dtype/finiteness, automatic
output-directory creation, and that the package is self-contained (no
dependency on the parent repository's ``src/``).
"""

import subprocess
import sys
from pathlib import Path

import numpy as np
import pytest

SUBMISSION_DIR = Path(__file__).resolve().parents[1] / "submission" / "SuperconductorSemistars"

sys.path.insert(0, str(SUBMISSION_DIR))
import run as submission_run  # noqa: E402


@pytest.fixture(autouse=True)
def _isolate_submission_sys_path():
    """Each test module-imports `run`; keep sys.path tidy across the suite."""
    yield


def _write_npy_inputs(input_dir: Path, shapes: dict[str, tuple[int, ...]]) -> None:
    input_dir.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    for name, shape in shapes.items():
        array = rng.random(shape, dtype=np.float32)
        np.save(input_dir / name, array)


# --- CLI contract ---


def test_cli_requires_exactly_two_positional_args(tmp_path: Path) -> None:
    assert submission_run.main([]) == 2
    assert submission_run.main([str(tmp_path)]) == 2
    assert submission_run.main([str(tmp_path), str(tmp_path), "extra"]) == 2


def test_cli_end_to_end_via_subprocess(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_npy_inputs(input_dir, {"a.npy": (12, 16)})

    result = subprocess.run(
        [sys.executable, str(SUBMISSION_DIR / "run.py"), str(input_dir), str(output_dir)],
        capture_output=True,
        text=True,
        timeout=120,
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "a.npy").is_file()


def test_cli_runs_from_a_different_working_directory(tmp_path: Path) -> None:
    """The evaluator command must not depend on CWD == the submission folder."""
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_npy_inputs(input_dir, {"a.npy": (8, 8)})

    result = subprocess.run(
        [sys.executable, str(SUBMISSION_DIR / "run.py"), str(input_dir), str(output_dir)],
        capture_output=True,
        text=True,
        timeout=120,
        cwd=str(tmp_path),  # deliberately NOT the submission directory
    )
    assert result.returncode == 0, result.stderr
    assert (output_dir / "a.npy").is_file()


# --- Filename mapping ---


def test_output_filenames_exactly_match_input_filenames(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_npy_inputs(input_dir, {"a.npy": (8, 8), "b.npy": (8, 8)})

    assert submission_run.main([str(input_dir), str(output_dir)]) == 0

    assert {p.name for p in output_dir.glob("*.npy")} == {"a.npy", "b.npy"}


def test_no_npy_files_produces_clear_error(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    input_dir.mkdir()
    output_dir = tmp_path / "output"
    assert submission_run.main([str(input_dir), str(output_dir)]) == 2


def test_missing_input_dir_produces_clear_error(tmp_path: Path) -> None:
    assert submission_run.main([str(tmp_path / "nope"), str(tmp_path / "out")]) == 2


def test_identical_input_and_output_dir_is_rejected(tmp_path: Path) -> None:
    _write_npy_inputs(tmp_path, {"a.npy": (8, 8)})
    assert submission_run.main([str(tmp_path), str(tmp_path)]) == 2
    # source file must be untouched
    assert np.load(tmp_path / "a.npy").shape == (8, 8)


# --- Output directory creation ---


def test_output_directory_is_created_automatically(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "nested" / "does" / "not" / "exist"
    _write_npy_inputs(input_dir, {"a.npy": (8, 8)})
    assert not output_dir.exists()

    assert submission_run.main([str(input_dir), str(output_dir)]) == 0
    assert output_dir.is_dir()


# --- Output dimensions (derived per-file, not hard-coded) ---


@pytest.mark.parametrize("shape", [(32, 48), (8, 8), (17, 23)])
def test_output_dimensions_are_exactly_2x_input(tmp_path: Path, shape: tuple[int, int]) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    _write_npy_inputs(input_dir, {"a.npy": shape})

    assert submission_run.main([str(input_dir), str(output_dir)]) == 0

    output_array = np.load(output_dir / "a.npy")
    assert output_array.shape == (shape[0] * 2, shape[1] * 2)


def test_supports_hw1_and_1hw_input_shapes(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    rng = np.random.default_rng(1)
    np.save(input_dir / "hw1.npy", rng.random((8, 8, 1), dtype=np.float32))
    np.save(input_dir / "onehw.npy", rng.random((1, 8, 8), dtype=np.float32))

    assert submission_run.main([str(input_dir), str(output_dir)]) == 0

    assert np.load(output_dir / "hw1.npy").shape == (16, 16)
    assert np.load(output_dir / "onehw.npy").shape == (16, 16)


def test_rejects_ambiguous_input_shape(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    np.save(input_dir / "bad.npy", np.zeros((8, 8, 3), dtype=np.float32))

    with pytest.raises(ValueError):
        submission_run.main([str(input_dir), str(output_dir)])


# --- Output range / numerical safety / dtype ---


def test_output_values_are_clipped_to_unit_range_and_finite(tmp_path: Path) -> None:
    input_dir = tmp_path / "input"
    output_dir = tmp_path / "output"
    input_dir.mkdir()
    # Deliberately out-of-[0,1] input, matching the real dataset's raw NoisyLR convention.
    np.save(input_dir / "a.npy", np.array([[-0.4, 1.6], [0.0, 1.0]], dtype=np.float32))

    assert submission_run.main([str(input_dir), str(output_dir)]) == 0

    output_array = np.load(output_dir / "a.npy")
    assert np.isfinite(output_array).all()
    assert output_array.min() >= 0.0
    assert output_array.max() <= 1.0
    assert output_array.dtype == np.float32


def test_restore_helper_clips_out_of_range_model_output(tmp_path: Path) -> None:
    """Unit-level check on restore(): synthetic model that returns out-of-range values."""
    import torch

    class _FixedOutputModel(torch.nn.Module):
        def forward(self, x: torch.Tensor) -> torch.Tensor:
            batch, _, height, width = x.shape
            out = torch.full((batch, 1, height * 2, width * 2), 5.0)
            out[0, 0, 0, 0] = -3.0
            return out

    model = _FixedOutputModel()
    array = np.zeros((4, 4), dtype=np.float32)
    prediction = submission_run.restore(model, array, torch.device("cpu"), scale=2)
    assert prediction.dtype == np.float32
    assert prediction.min() >= 0.0
    assert prediction.max() <= 1.0
    assert np.isfinite(prediction).all()


# --- GPU/CPU independence ---


def test_select_device_does_not_require_cuda() -> None:
    device = submission_run.select_device()
    assert device.type in ("cuda", "cpu")


def test_load_model_works_on_cpu() -> None:
    import torch

    model, scale = submission_run.load_model(torch.device("cpu"))
    assert scale == 2
    assert sum(p.numel() for p in model.parameters()) == 630_724


# --- Self-contained package (no dependency on the parent repo's src/) ---


def test_run_module_does_not_import_parent_src_package() -> None:
    assert not hasattr(submission_run, "src")
    source = (SUBMISSION_DIR / "run.py").read_text(encoding="utf-8")
    assert "from src" not in source
    assert "import src" not in source


def test_model_state_dict_loads_strictly_from_packaged_weights() -> None:
    import torch

    from models.residual_sr import ResidualSRNet

    package = torch.load(SUBMISSION_DIR / "models" / "residualsr_final_ema.pt", map_location="cpu", weights_only=False)
    model = ResidualSRNet(
        in_channels=package["in_channels"],
        out_channels=package["out_channels"],
        num_features=package["num_features"],
        num_blocks=package["num_blocks"],
        scale=package["scale"],
    )
    model.load_state_dict(package["model_state_dict"], strict=True)  # raises on any mismatch
