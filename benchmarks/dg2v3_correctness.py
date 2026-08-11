"""DG2 v3 SDP full-pipeline correctness matrix.

Compares the sidecar output against torch SDPA (XPU) and an fp32 CPU
reference across multi-tile / multi-q-tile shapes and both fp16/bf16.

Usage:
    python dg2v3_correctness.py
"""

import torch

import omni_xpu_kernel.sdp as sdp


def run_case(b, qlen, kvlen, h, d, dtype, seed, verbose=True):
    torch.manual_seed(seed)
    q = torch.randn(b, qlen, h, d, device="cpu", dtype=dtype) * 0.5
    k = torch.randn(b, kvlen, h, d, device="cpu", dtype=dtype) * 0.5
    v = torch.randn(b, kvlen, h, d, device="cpu", dtype=dtype) * 0.5

    qx, kx, vx = q.to("xpu"), k.to("xpu"), v.to("xpu")
    out = sdp.sdp(qx, kx, vx)
    torch.xpu.synchronize()

    ref = torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3),
        k.permute(0, 2, 1, 3),
        v.permute(0, 2, 1, 3),
    ).permute(0, 2, 1, 3)

    # fp32 CPU ground truth
    scale = 1.0 / (d**0.5)
    qf = q.permute(0, 2, 1, 3).float()   # [B, H, L, D]
    kf = k.permute(0, 2, 1, 3).float()
    vf = v.permute(0, 2, 1, 3).float()
    scores = (qf @ kf.transpose(-2, -1)) * scale   # [B, H, L, kv]
    p = torch.softmax(scores, dim=-1)
    gt = (p @ vf).permute(0, 2, 1, 3)

    err_torch = (out.cpu().float() - ref.float()).abs().max().item()
    err_gt = (out.cpu().float() - gt).abs().max().item()
    finite = bool(torch.isfinite(out).all())
    if verbose:
        print(
            f"L={qlen:4d} kv={kvlen:4d} H={h:2d} "
            f"{str(dtype):6s} seed={seed}: vs_torch={err_torch:.6f} "
            f"vs_gt={err_gt:.6f} finite={finite}"
        )
    return finite, err_torch, err_gt


def main():
    if not sdp.is_available():
        raise SystemExit("sidecar unavailable")
    cases = []
    for qlen in (1, 16, 63, 64, 65, 128, 129, 256, 300):
        for kv in (1, 16, 64, 65, 128, 300):
            cases.append((1, qlen, kv, 1, torch.float16, qlen * 7 + kv))
    for h in (8, 12):
        for qlen in (16, 128, 257):
            for kv in (64, 129, 300):
                cases.append((1, qlen, kv, h, torch.float16, 1000 + h * 31 + qlen + kv))
    for qlen in (16, 128, 256):
        for kv in (64, 128, 300):
            cases.append((1, qlen, kv, 8, torch.bfloat16, 5000 + qlen + kv))

    worst = 0.0
    failures = 0
    for b, qlen, kv, h, dtype, seed in cases:
        finite, err_torch, err_gt = run_case(b, qlen, kv, h, 128, dtype, seed)
        if not finite or err_gt > 0.05:
            failures += 1
            print(f"  FAIL L={qlen} kv={kv} H={h} {dtype} seed={seed}")
        worst = max(worst, err_gt)
    print(f"\n{cases.__len__()} cases; failures={failures}; worst vs GT={worst:.6f}")
    if failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
