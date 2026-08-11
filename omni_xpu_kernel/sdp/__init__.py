"""Standalone scaled dot-product attention kernel wrapper."""

import os
from pathlib import Path

import torch


def _get_native():
    from .. import _load_extension
    return _load_extension().sdp


def _sidecar_candidates():
    sidecar_dir = Path(__file__).resolve().parent.parent / "lgrf_uni"
    if not sidecar_dir.is_dir():
        return ()
    pattern = "lgrf_sdp*.pyd" if os.name == "nt" else "lgrf_sdp*.so"
    return tuple(sidecar_dir.glob(pattern))


_sidecar_cached = None


def _sidecar_candidates_cached():
    """Return the sidecar list, scanned once (glob is ~0.5 ms per call)."""
    global _sidecar_cached
    if _sidecar_cached is None:
        _sidecar_cached = _sidecar_candidates()
    return _sidecar_cached


def is_available() -> bool:
    """Return whether the target-specific attention sidecar is packaged."""
    return bool(_sidecar_candidates_cached())


def sdp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    if not _sidecar_candidates_cached():
        raise RuntimeError(
            "omni_xpu_kernel SDP sidecar is unavailable for this build; "
            "A770/DG2 builds currently use PyTorch SDPA for attention"
        )
    return _get_native().sdp(q, k, v)


__all__ = ["sdp", "is_available"]
