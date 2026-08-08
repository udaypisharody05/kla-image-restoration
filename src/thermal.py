"""Optional GPU thermal guard: pauses training between completed batches when
the GPU gets too hot, resumes once it has cooled down.

Disabled by default (``limit <= 0``). When disabled, nothing in this module
touches ``subprocess`` or ``time.sleep`` -- training behaves exactly as if
this feature did not exist. This is a wall-clock scheduling aid only; it must
never change model/optimizer/loss/scheduler/data behavior.
"""

from collections.abc import Callable, Sequence
import subprocess
import time


DEFAULT_NVIDIA_SMI_COMMAND: tuple[str, ...] = (
    "nvidia-smi",
    "--query-gpu=temperature.gpu",
    "--format=csv,noheader,nounits",
)


def read_gpu_temperature(command: Sequence[str] = DEFAULT_NVIDIA_SMI_COMMAND) -> float:
    """Query the current GPU temperature (Celsius) via ``nvidia-smi``.

    On a multi-GPU machine, only the *first* reported temperature is used --
    this project targets a single-GPU laptop, so no per-GPU selection logic
    is implemented. Raises ``RuntimeError`` with a clear message if
    ``nvidia-smi`` is missing, exits non-zero, or its output cannot be parsed;
    never returns a fabricated value.
    """
    try:
        result = subprocess.run(list(command), capture_output=True, text=True, check=False)
    except (OSError, FileNotFoundError) as exc:
        raise RuntimeError(f"Could not run nvidia-smi to read GPU temperature: {exc}") from exc

    if result.returncode != 0:
        raise RuntimeError(
            f"nvidia-smi exited with code {result.returncode}: {result.stderr.strip()}"
        )

    lines = [line.strip() for line in result.stdout.splitlines() if line.strip()]
    if not lines:
        raise RuntimeError("nvidia-smi returned no GPU temperature output")

    first_gpu_line = lines[0]  # multiple GPUs: only the first is used (see docstring)
    try:
        return float(first_gpu_line)
    except ValueError as exc:
        raise RuntimeError(
            f"Could not parse GPU temperature from nvidia-smi output: {first_gpu_line!r}"
        ) from exc


def validate_gpu_temperature_settings(
    limit: float, resume_threshold: float, check_interval: int, poll_seconds: float
) -> None:
    """Validate thermal-guard CLI settings, independent of constructing a guard."""
    if limit < 0:
        raise ValueError(f"--gpu-temp-limit must be >= 0, got {limit}")
    if resume_threshold < 0:
        raise ValueError(f"--gpu-temp-resume must be >= 0, got {resume_threshold}")
    if check_interval <= 0:
        raise ValueError(f"--gpu-temp-check-interval must be > 0, got {check_interval}")
    if poll_seconds <= 0:
        raise ValueError(f"--gpu-temp-poll-seconds must be > 0, got {poll_seconds}")
    # limit == 0 means the guard is disabled; the resume threshold is then
    # irrelevant and intentionally not checked against it (an arbitrary
    # --gpu-temp-resume alongside a disabled guard is allowed).
    if limit > 0 and resume_threshold >= limit:
        raise ValueError(
            f"--gpu-temp-resume ({resume_threshold}) must be strictly less than "
            f"--gpu-temp-limit ({limit}) when the guard is enabled."
        )


class GpuTemperatureGuard:
    """Cooperative thermal pause, checked between completed training/validation batches.

    Call :meth:`on_batch_complete` once after each fully completed batch
    (i.e. after ``optimizer.step()`` for training, or after a completed
    validation batch). Every ``check_interval`` calls, the GPU temperature is
    read once; if it is at or above ``limit``, training sleeps in
    ``poll_seconds`` increments -- rechecking each time -- until the
    temperature drops to or below ``resume_threshold`` (hysteresis: a lower
    resume threshold than the pause limit avoids rapidly toggling pause/resume
    near a single temperature).

    When ``limit <= 0`` the guard is disabled: ``on_batch_complete`` is then a
    true no-op that never calls the temperature reader or sleeps.
    """

    def __init__(
        self,
        limit: float,
        resume_threshold: float,
        check_interval: int,
        poll_seconds: float,
        temperature_reader: Callable[[], float] = read_gpu_temperature,
        sleep_fn: Callable[[float], None] = time.sleep,
    ) -> None:
        validate_gpu_temperature_settings(limit, resume_threshold, check_interval, poll_seconds)
        self.limit = limit
        self.resume_threshold = resume_threshold
        self.check_interval = check_interval
        self.poll_seconds = poll_seconds
        self._read_temperature = temperature_reader
        self._sleep = sleep_fn
        self._batches_since_check = 0

    @property
    def enabled(self) -> bool:
        return self.limit > 0

    def verify_monitoring(self) -> None:
        """Read once so a broken nvidia-smi fails fast at startup, not mid-run.

        No-op when the guard is disabled.
        """
        if self.enabled:
            self._read_temperature()

    def on_batch_complete(self) -> None:
        """Call after a fully completed batch. No-op when the guard is disabled."""
        if not self.enabled:
            return
        self._batches_since_check += 1
        if self._batches_since_check < self.check_interval:
            return
        self._batches_since_check = 0
        self._check_and_pause_if_needed()

    def _check_and_pause_if_needed(self) -> None:
        # No torch.cuda.synchronize() here: train.py already calls loss.item()
        # every batch (to accumulate the running loss), and .item() on a CUDA
        # tensor is itself a blocking/synchronizing call -- by the time this
        # method runs, this batch's GPU work has already completed. Adding an
        # explicit synchronize would be redundant overhead, not a correctness fix.
        temperature = self._read_temperature()
        if temperature < self.limit:
            return
        print(f"GPU temperature {temperature:.0f}°C reached limit {self.limit:.0f}°C.")
        print(f"Cooling until <= {self.resume_threshold:.0f}°C...")
        while temperature > self.resume_threshold:
            self._sleep(self.poll_seconds)
            temperature = self._read_temperature()
        print(f"GPU temperature {temperature:.0f}°C. Resuming training.")


def build_gpu_temperature_guard(
    limit: float, resume_threshold: float, check_interval: int, poll_seconds: float
) -> GpuTemperatureGuard:
    """Construct and validate a :class:`GpuTemperatureGuard` from CLI-style values."""
    return GpuTemperatureGuard(limit, resume_threshold, check_interval, poll_seconds)
