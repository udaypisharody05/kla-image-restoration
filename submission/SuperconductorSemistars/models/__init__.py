"""Self-contained model package for the submission runtime (no dependency on the parent repository's ``src/``)."""

from .residual_sr import ResidualSRNet

__all__ = ["ResidualSRNet"]
