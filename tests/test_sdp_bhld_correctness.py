"""Correctness coverage for the DG2 SDP BHLD-direct entry (sdp_bhld).

Regression test for the padding/stride bug review:
  - sdp_bhld pads K/V to a multiple of 16 but passed the ORIGINAL kv_len to
    the kernel; in BHLD layout the per-head stride becomes (L+pad)*D while the
    kernel addresses K/V as headIdx * L * D -> wrong data for every head > 0
    whenever kv_len % 16 != 0.
Fix: BHLD path does NOT pad (DG2 v4 kernels mask out-of-range rows).
"""

import pytest
import torch
import torch.nn.functional as F


def has_dg2_sdp_bhld():
    try:
        import omni_xpu_kernel
        from omni_xpu_kernel import sdp

        return (
            torch.xpu.is_available()
            and omni_xpu_kernel.__xpu_target__ == "dg2"
            and sdp is not None
            and hasattr(sdp, "sdp_bhld")
        )
    except Exception:
        return False


@pytest.mark.skipif(
    not has_dg2_sdp_bhld(), reason="DG2 sdp_bhld sidecar unavailable"
)
@pytest.mark.parametrize(
    "seq,heads,dtype",
    [
        (1, 56, torch.bfloat16),
        (17, 56, torch.bfloat16),
        (1459, 56, torch.bfloat16),
        (9814, 56, torch.bfloat16),
        (20683, 56, torch.bfloat16),
        (1459, 48, torch.float16),
        (20683, 48, torch.float16),
        (10811, 48, torch.bfloat16),
    ],
)
def test_sdp_bhld_matches_sdp_and_torch(seq, heads, dtype):
    from omni_xpu_kernel import sdp

    torch.xpu.manual_seed_all(20260818)
    D = 128
    q = torch.randn(1, heads, seq, D, device="xpu", dtype=dtype) * 0.1
    k = torch.randn(1, heads, seq, D, device="xpu", dtype=dtype) * 0.1
    v = torch.randn(1, heads, seq, D, device="xpu", dtype=dtype) * 0.1

    # BHLD-direct (fixed path: no padding)
    out_bhld = sdp.sdp_bhld(q, k, v)  # [B, L, H, D]
    torch.xpu.synchronize()
    assert out_bhld.shape == (1, seq, heads, D)
    assert torch.isfinite(out_bhld).all()

    # Old permute path (BLHD input, pads K/V — head stride unaffected)
    qb = q.permute(0, 2, 1, 3).contiguous()
    kb = k.permute(0, 2, 1, 3).contiguous()
    vb = v.permute(0, 2, 1, 3).contiguous()
    out_old = sdp.sdp(qb, kb, vb)  # [B, L, H, D]
    torch.xpu.synchronize()

    # Torch SDPA reference: BLHD -> [B, H, L, D]
    ref = F.scaled_dot_product_attention(qb, kb, vb)
    torch.xpu.synchronize()

    rel = lambda a, b: (a.float() - b.float()).abs().mean() / (
        b.float().abs().mean() + 1e-6
    )
    # 修复后的 BHLD 路径必须与旧路径一致（fp16 累加噪声级别）
    r_bhld_old = rel(out_bhld, out_old)
    r_bhld_torch = rel(out_bhld.transpose(1, 2), ref)
    r_old_torch = rel(out_old.transpose(1, 2), ref)

    print(
        f"seq={seq} H={heads} {dtype}: bhld-vs-old={r_bhld_old:.5f} "
        f"bhld-vs-torch={r_bhld_torch:.5f} old-vs-torch={r_old_torch:.5f}"
    )
    assert r_bhld_old < 0.05, f"BHLD vs old diverged: {r_bhld_old:.5f}"
    assert r_bhld_torch < 0.10, f"BHLD vs torch diverged: {r_bhld_torch:.5f}"
