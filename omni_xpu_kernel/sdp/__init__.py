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


def sdp_bhld(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """BHLD-direct SDP (DG2): q/k/v [B=1, H, L, D], returns [B, L, H, D].

    Reads the raw [B, H, L, D] contiguous buffers directly (heads-first),
    avoiding the three permute+copy layout conversions of :func:`sdp`.
    The output is [B, L, H, D] contiguous, so a reshape to [B, L, H*D]
    is a copy-free view.
    """
    if not _sidecar_candidates_cached():
        raise RuntimeError("omni_xpu_kernel SDP sidecar is unavailable for this build")
    return _get_native().sdp_bhld(q, k, v)


def clear_cache() -> None:
    """Release sidecar-owned packed Q/K/V USM buffers.

    ComfyUI's ``unload_all_models`` / ``torch.xpu.empty_cache`` cannot see
    these allocations. Call this before starting a new workflow run when
    VRAM pressure is observed.
    """
    if not _sidecar_candidates_cached():
        return
    _get_native().clear_cache()


__all__ = ["sdp", "sdp_bhld", "is_available", "clear_cache"]
