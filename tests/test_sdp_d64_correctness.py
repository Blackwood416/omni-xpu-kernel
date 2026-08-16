"""A770 DG2 D64 DPAS attention correctness sweep (v4.1 D64 port).

Run standalone in an isolated process:
  python tests/test_sdp_d64_correctness.py [--seq 1 16 ...]

Compares omni_xpu_kernel.sdp.sdp against torch SDPA on [1, L, H, 64]
fp16/bf16 (MiniMax H3 VideoVAE contract).
"""

from __future__ import annotations

import argparse

import torch

from omni_xpu_kernel import sdp as omni_sdp


def check(seq: int, heads: int, dtype: torch.dtype, seed: int = 20260816):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    q = torch.randn(1, seq, heads, 64, dtype=dtype, generator=gen).to("xpu")
    k = torch.randn(1, seq, heads, 64, dtype=dtype, generator=gen).to("xpu")
    v = torch.randn(1, seq, heads, 64, dtype=dtype, generator=gen).to("xpu")

    # torch SDPA's 4D convention is [B, H, L, D]; the sidecar uses
    # [B, L, H, D] (ComfyUI layout), so transpose for the reference.
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.transpose(1, 2), k.transpose(1, 2), v.transpose(1, 2)
    ).transpose(1, 2)
    out = omni_sdp.sdp(q, k, v)

    d = (out.float() - ref.float()).abs()
    denom = ref.float().abs().clamp_min(1e-6)
    rel = (d / denom).max().item()
    ok = bool(torch.isfinite(out).all())
    print(
        f"seq={seq:>6} heads={heads} {str(dtype):>7}: "
        f"max_abs={d.max().item():.3e} max_rel={rel:.3e} finite={ok}",
        flush=True,
    )
    assert ok, "non-finite output"
    assert d.max().item() <= 0.02, f"abs error too large: {d.max().item()}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--seq",
        nargs="+",
        type=int,
        default=[1, 16, 31, 65, 129, 300, 512, 1024, 1797, 4096, 8192, 20683],
    )
    args = parser.parse_args()

    print("device:", torch.xpu.get_device_name(0), "torch:", torch.__version__)
    for dtype in (torch.float16, torch.bfloat16):
        for seq in args.seq:
            check(seq, 32, dtype)
    print("D64 correctness OK")


if __name__ == "__main__":
    main()
