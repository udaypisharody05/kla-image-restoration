"""Fast, mock-only tests for the optional GPU thermal guard (src/thermal.py).

No real GPU, CUDA, nvidia-smi, or actual waiting is required or used: all
subprocess calls and time.sleep are mocked/injected.
"""

import subprocess
from unittest.mock import patch

import pytest

from src.thermal import (
    DEFAULT_NVIDIA_SMI_COMMAND,
    GpuTemperatureGuard,
    read_gpu_temperature,
    validate_gpu_temperature_settings,
)


def _completed_process(stdout: str, returncode: int = 0, stderr: str = "") -> subprocess.CompletedProcess:
    return subprocess.CompletedProcess(
        args=list(DEFAULT_NVIDIA_SMI_COMMAND), returncode=returncode, stdout=stdout, stderr=stderr
    )


# --- Temperature reading ---


def test_read_gpu_temperature_parses_plain_number() -> None:
    with patch("subprocess.run", return_value=_completed_process("82\n")) as mock_run:
        assert read_gpu_temperature() == 82.0
    mock_run.assert_called_once()
    called_args, called_kwargs = mock_run.call_args
    assert called_args[0] == list(DEFAULT_NVIDIA_SMI_COMMAND)
    assert called_kwargs.get("shell", False) is False
    assert called_kwargs.get("capture_output") is True
    assert called_kwargs.get("text") is True


def test_read_gpu_temperature_handles_surrounding_whitespace() -> None:
    with patch("subprocess.run", return_value=_completed_process("  82  \r\n")):
        assert read_gpu_temperature() == 82.0


def test_read_gpu_temperature_selects_first_gpu_of_multiple_lines() -> None:
    with patch("subprocess.run", return_value=_completed_process("82\n75\n")):
        assert read_gpu_temperature() == 82.0


def test_read_gpu_temperature_fails_clearly_on_non_numeric_output() -> None:
    with patch("subprocess.run", return_value=_completed_process("not-a-number\n")):
        with pytest.raises(RuntimeError, match="parse"):
            read_gpu_temperature()


def test_read_gpu_temperature_fails_clearly_when_nvidia_smi_missing() -> None:
    with patch("subprocess.run", side_effect=FileNotFoundError("no such file")):
        with pytest.raises(RuntimeError, match="Could not run nvidia-smi"):
            read_gpu_temperature()


def test_read_gpu_temperature_fails_clearly_on_nonzero_return_code() -> None:
    with patch(
        "subprocess.run", return_value=_completed_process("", returncode=1, stderr="boom")
    ):
        with pytest.raises(RuntimeError, match="nvidia-smi exited with code 1"):
            read_gpu_temperature()


def test_read_gpu_temperature_fails_clearly_on_empty_output() -> None:
    with patch("subprocess.run", return_value=_completed_process("")):
        with pytest.raises(RuntimeError, match="no GPU temperature output"):
            read_gpu_temperature()


# --- Disabled behavior ---


def test_default_gpu_temp_limit_is_zero() -> None:
    import train

    with patch("sys.argv", ["train.py"]):
        parsed = train.parse_args()
    assert parsed.gpu_temp_limit == 0.0
    assert parsed.gpu_temp_resume == 78.0
    assert parsed.gpu_temp_check_interval == 5
    assert parsed.gpu_temp_poll_seconds == 3.0


def test_disabled_guard_never_queries_temperature() -> None:
    reader_calls = []
    guard = GpuTemperatureGuard(
        limit=0.0,
        resume_threshold=78.0,
        check_interval=1,
        poll_seconds=3.0,
        temperature_reader=lambda: reader_calls.append(1) or 999.0,
    )
    for _ in range(10):
        guard.on_batch_complete()
    assert reader_calls == []


def test_disabled_guard_never_sleeps() -> None:
    sleep_calls = []
    guard = GpuTemperatureGuard(
        limit=0.0,
        resume_threshold=78.0,
        check_interval=1,
        poll_seconds=3.0,
        temperature_reader=lambda: 999.0,
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
    )
    for _ in range(10):
        guard.on_batch_complete()
    assert sleep_calls == []


def test_disabled_guard_verify_monitoring_is_a_no_op() -> None:
    reader_calls = []
    guard = GpuTemperatureGuard(
        limit=0.0,
        resume_threshold=78.0,
        check_interval=1,
        poll_seconds=3.0,
        temperature_reader=lambda: reader_calls.append(1) or 999.0,
    )
    guard.verify_monitoring()
    assert reader_calls == []


# --- Threshold / hysteresis behavior ---


def test_temperature_below_limit_does_nothing() -> None:
    sleep_calls = []
    guard = GpuTemperatureGuard(
        limit=82.0,
        resume_threshold=78.0,
        check_interval=1,
        poll_seconds=3.0,
        temperature_reader=lambda: 80.0,
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
    )
    guard.on_batch_complete()
    assert sleep_calls == []


def test_temperature_exactly_at_limit_enters_pause() -> None:
    sequence = iter([82.0, 78.0])
    sleep_calls = []
    guard = GpuTemperatureGuard(
        limit=82.0,
        resume_threshold=78.0,
        check_interval=1,
        poll_seconds=3.0,
        temperature_reader=lambda: next(sequence),
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
    )
    guard.on_batch_complete()
    assert sleep_calls == [3.0]


def test_temperature_above_limit_enters_pause() -> None:
    sequence = iter([90.0, 78.0])
    sleep_calls = []
    guard = GpuTemperatureGuard(
        limit=82.0,
        resume_threshold=78.0,
        check_interval=1,
        poll_seconds=3.0,
        temperature_reader=lambda: next(sequence),
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
    )
    guard.on_batch_complete()
    assert sleep_calls == [3.0]


def test_pause_continues_until_temperature_reaches_resume_threshold() -> None:
    sequence = iter([82.0, 81.0, 79.0, 78.0])
    reads = []
    sleep_calls = []

    def reader() -> float:
        value = next(sequence)
        reads.append(value)
        return value

    guard = GpuTemperatureGuard(
        limit=82.0,
        resume_threshold=78.0,
        check_interval=1,
        poll_seconds=3.0,
        temperature_reader=reader,
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
    )
    guard.on_batch_complete()
    assert reads == [82.0, 81.0, 79.0, 78.0]
    assert sleep_calls == [3.0, 3.0, 3.0]  # remains paused at 81 and 79, resumes at 78


def test_sleep_receives_configured_polling_interval() -> None:
    sequence = iter([85.0, 85.0, 78.0])
    sleep_calls = []
    guard = GpuTemperatureGuard(
        limit=82.0,
        resume_threshold=78.0,
        check_interval=1,
        poll_seconds=7.5,
        temperature_reader=lambda: next(sequence),
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
    )
    guard.on_batch_complete()
    assert sleep_calls == [7.5, 7.5]


def test_does_not_resume_prematurely_between_limit_and_resume_threshold() -> None:
    """82 -> pause; must remain paused through 81/80/79, only resume at 78."""
    sequence = iter([82.0, 81.0, 80.0, 79.0, 78.0])
    reads = []

    def reader() -> float:
        value = next(sequence)
        reads.append(value)
        return value

    guard = GpuTemperatureGuard(
        limit=82.0,
        resume_threshold=78.0,
        check_interval=1,
        poll_seconds=1.0,
        temperature_reader=reader,
        sleep_fn=lambda seconds: None,
    )
    guard.on_batch_complete()
    assert reads == [82.0, 81.0, 80.0, 79.0, 78.0]


# --- Mocked full-sequence smoke test (no real waiting) ---


def test_mocked_thermal_sequence_normal_then_pause_then_resume() -> None:
    """80 (normal) -> 82 (threshold reached) -> 81 -> 79 -> 78 (resume)."""
    sequence = iter([80.0, 82.0, 81.0, 79.0, 78.0])
    reads = []
    sleep_calls = []

    def reader() -> float:
        value = next(sequence)
        reads.append(value)
        return value

    guard = GpuTemperatureGuard(
        limit=82.0,
        resume_threshold=78.0,
        check_interval=1,
        poll_seconds=3.0,
        temperature_reader=reader,
        sleep_fn=lambda seconds: sleep_calls.append(seconds),
    )
    guard.on_batch_complete()  # reads 80 -> below limit, no pause
    assert reads == [80.0]
    assert sleep_calls == []

    guard.on_batch_complete()  # reads 82, 81, 79, 78 -> pause then resume
    assert reads == [80.0, 82.0, 81.0, 79.0, 78.0]
    assert sleep_calls == [3.0, 3.0, 3.0]


# --- Training-loop integration: no extra/missing optimizer operations ---


def test_thermal_pause_does_not_change_optimizer_step_count() -> None:
    import torch
    from torch import nn

    from src.dataset import PairedRestorationDataset, create_dataloader
    from src.dataset_discovery import ImagePair
    from train import train_one_epoch

    import numpy as np
    import tempfile
    from pathlib import Path

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        input_dir, target_dir = root / "NoisyLR", root / "GT"
        input_dir.mkdir(parents=True)
        target_dir.mkdir(parents=True)
        pairs = []
        for index in range(6):
            degraded = np.random.default_rng(index).uniform(0, 1, size=(8, 8)).astype(np.float32)
            target = np.random.default_rng(index + 100).uniform(0, 1, size=(16, 16)).astype(
                np.float32
            )
            input_path = input_dir / f"{index:06d}.npy"
            target_path = target_dir / f"{index:06d}.npy"
            np.save(input_path, degraded)
            np.save(target_path, target)
            pairs.append(ImagePair(f"{index:06d}", input_path, target_path))

        dataset = PairedRestorationDataset(pairs)
        loader = create_dataloader(dataset, batch_size=2, shuffle=False)  # 3 batches

        model = nn.Conv2d(1, 4, 3, padding=1)  # trivial stand-in model, shape-agnostic loss below
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
        step_calls = []
        real_step = optimizer.step

        def counting_step():
            step_calls.append(1)
            return real_step()

        optimizer.step = counting_step

        def trivial_loss_fn(outputs, targets):
            return outputs.mean() * 0.0 + 1.0  # constant, differentiable-safe stand-in

        guard = GpuTemperatureGuard(
            limit=1.0,  # absurdly low: every check will "pause"
            resume_threshold=0.0,
            check_interval=1,
            poll_seconds=0.001,
            temperature_reader=iter([50.0, -1.0] * 10).__next__,  # first hot, then instantly cool
            sleep_fn=lambda seconds: None,
        )

        train_one_epoch(model, loader, optimizer, trivial_loss_fn, torch.device("cpu"), guard)
        assert len(step_calls) == 3  # exactly one optimizer.step() per batch, no more/less


# --- CLI validation ---


def test_validate_rejects_negative_limit() -> None:
    with pytest.raises(ValueError, match="gpu-temp-limit"):
        validate_gpu_temperature_settings(-1.0, 78.0, 5, 3.0)


def test_validate_rejects_negative_resume() -> None:
    with pytest.raises(ValueError, match="gpu-temp-resume"):
        validate_gpu_temperature_settings(82.0, -1.0, 5, 3.0)


def test_validate_rejects_non_positive_check_interval() -> None:
    with pytest.raises(ValueError, match="gpu-temp-check-interval"):
        validate_gpu_temperature_settings(82.0, 78.0, 0, 3.0)


def test_validate_rejects_non_positive_poll_seconds() -> None:
    with pytest.raises(ValueError, match="gpu-temp-poll-seconds"):
        validate_gpu_temperature_settings(82.0, 78.0, 5, 0.0)


def test_validate_rejects_resume_equal_to_limit_when_enabled() -> None:
    with pytest.raises(ValueError, match="strictly less than"):
        validate_gpu_temperature_settings(82.0, 82.0, 5, 3.0)


def test_validate_rejects_resume_above_limit_when_enabled() -> None:
    with pytest.raises(ValueError, match="strictly less than"):
        validate_gpu_temperature_settings(82.0, 85.0, 5, 3.0)


def test_validate_allows_arbitrary_resume_when_limit_is_disabled() -> None:
    validate_gpu_temperature_settings(0.0, 999.0, 5, 3.0)  # must not raise


def test_validate_accepts_recommended_configuration() -> None:
    validate_gpu_temperature_settings(82.0, 78.0, 5, 3.0)  # must not raise


# --- Check cadence ---


def test_temperature_checked_only_at_configured_batch_interval() -> None:
    reader_calls = []
    guard = GpuTemperatureGuard(
        limit=82.0,
        resume_threshold=78.0,
        check_interval=5,
        poll_seconds=3.0,
        temperature_reader=lambda: reader_calls.append(1) or 20.0,  # always cool
        sleep_fn=lambda seconds: None,
    )
    for _ in range(4):
        guard.on_batch_complete()
    assert reader_calls == []  # not yet at the 5th batch

    guard.on_batch_complete()  # 5th completed batch -> exactly one check
    assert len(reader_calls) == 1

    for _ in range(4):
        guard.on_batch_complete()
    assert len(reader_calls) == 1  # still not due again

    guard.on_batch_complete()  # 10th completed batch -> second check
    assert len(reader_calls) == 2
