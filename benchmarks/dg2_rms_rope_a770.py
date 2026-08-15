"""A770 RMS-RoPE A/B: Kitchen eager in-place vs omni native in-place.

Targets the exact MiniMax H3 contract used by ComfyUI:
  q/k = [1, S, 56, 128] bf16 views into a packed [S, 3*7168] qkv buffer
  freqs_cis = [1, S, 1, 48, 2, 2] bf16
  rot_dim = 96, split-half, in-place, epsilon = 1e-5

Run with the A770 (dg2) wheel:
  python benchmarks/dg2_rms_rope_a770.py [--seq 1024 4096 ...] [--iters 30]
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch


def make_contract(seq: int, seed: int = 1234):
    """Return (qkv, q, k, freqs_cis, q_scale, k_scale) with H3 strides."""
    gen = torch.Generator(device="cpu").manual_seed(seed)
    qkv = torch.randn((seq, 3 * 7168), dtype=torch.bfloat16, generator=gen).to("xpu")
    q = qkv[:, :7168].view(1, seq, 56, 128)
    k = qkv[:, 7168 : 2 * 7168].view(1, seq, 56, 128)

    angles = torch.randn((seq, 96), dtype=torch.float32, generator=gen).to("xpu") * 0.5
    half = angles[:, :48]
    c, s = torch.cos(half), torch.sin(half)
    freqs = (
        torch.stack([c, -s, s, c], dim=-1)
        .reshape(1, seq, 1, 48, 2, 2)
        .to(torch.bfloat16)
    )
    qw = torch.randn(128, dtype=torch.bfloat16, generator=gen).to("xpu")
    kw = torch.randn(128, dtype=torch.bfloat16, generator=gen).to("xpu")
    return qkv, q, k, freqs, qw, kw


def eager_impl(qkv, q, k, freqs, qw, kw, seq):
    from comfy_kitchen.backends.eager.rope import rms_rope_split_half_

    rms_rope_split_half_(q, k, freqs, qw, kw, epsilon=1e-5, rot_dim=96)


def native_impl(qkv, q, k, freqs, qw, kw, seq):
    from omni_xpu_kernel import rotary

    rotary.rms_kitchen_rope_split_half_(q, k, freqs, qw, kw, epsilon=1e-5, rot_dim=96)


def check(qkv_e, qkv_n, seq):
    q_e = qkv_e[:, :7168].view(1, seq, 56, 128)
    k_e = qkv_e[:, 7168 : 2 * 7168].view(1, seq, 56, 128)
    q_n = qkv_n[:, :7168].view(1, seq, 56, 128)
    k_n = qkv_n[:, 7168 : 2 * 7168].view(1, seq, 56, 128)
    dq = (q_e.float() - q_n.float()).abs().max().item()
    dk = (k_e.float() - k_n.float()).abs().max().item()
    return dq, dk


def check_tolerance(seq: int) -> None:
    """Assert the omni path stays inside the repo's 2% bf16 contract."""
    qkv_e, q_e, k_e, freqs, qw, kw = make_contract(seq)
    qkv_n = qkv_e.clone()
    q_n = qkv_n[:, :7168].view(1, seq, 56, 128)
    k_n = qkv_n[:, 7168 : 2 * 7168].view(1, seq, 56, 128)
    eager_impl(qkv_e, q_e, k_e, freqs, qw, kw, seq)
    native_impl(qkv_n, q_n, k_n, freqs, qw, kw, seq)
    qe = qkv_e[:, :7168].view(1, seq, 56, 128)
    ke = qkv_e[:, 7168 : 2 * 7168].view(1, seq, 56, 128)
    qn = qkv_n[:, :7168].view(1, seq, 56, 128)
    kn = qkv_n[:, 7168 : 2 * 7168].view(1, seq, 56, 128)
    dq = (qe.float() - qn.float()).abs().max().item()
    dk = (ke.float() - kn.float()).abs().max().item()
    q_rel = ((qe.float() - qn.float()).abs() / (qe.float().abs() + 1e-6)).max().item()
    k_rel = ((ke.float() - kn.float()).abs() / (ke.float().abs() + 1e-6)).max().item()
    print(f"seq={seq}: max_abs q={dq:.3e} k={dk:.3e}  max_rel q={q_rel:.3e} k={k_rel:.3e}")
    torch.testing.assert_close(qn, qe, rtol=0.02, atol=0.02)
    torch.testing.assert_close(kn, ke, rtol=0.02, atol=0.02)
    print("  tolerance OK (rtol=0.02, atol=0.02)")


def diagnose(seq: int, seed: int) -> None:
    """Compare eager and native against an independent fp32 reference."""
    qkv_e, q_e, k_e, freqs, qw, kw = make_contract(seq, seed=seed)
    qkv_n = qkv_e.clone()
    q_n = qkv_n[:, :7168].view(1, seq, 56, 128)
    k_n = qkv_n[:, 7168 : 2 * 7168].view(1, seq, 56, 128)
    q_before = q_e.clone()
    k_before = k_e.clone()

    eager_impl(qkv_e, q_e, k_e, freqs, qw, kw, seq)
    native_impl(qkv_n, q_n, k_n, freqs, qw, kw, seq)

    def ref(x, scale):
        xf = x.float()
        inv = torch.rsqrt(xf.square().mean(-1, keepdim=True) + 1e-5)
        normalized = (xf * inv * scale.float()).to(x.dtype)
        rotated = (
            normalized[..., :96]
            .reshape(*normalized.shape[:-1], 2, -1)
            .movedim(-2, -1)
            .unsqueeze(-2)
            .to(freqs.dtype)
        )
        rot = freqs[..., 0] * rotated[..., 0] + freqs[..., 1] * rotated[..., 1]
        rot = rot.movedim(-1, -2).reshape_as(normalized[..., :96]).type_as(normalized)
        return torch.cat((rot, normalized[..., 96:]), dim=-1)

    q_ref = ref(q_before, qw)
    k_ref = ref(k_before, kw)

    for name, actual in (("q", q_e), ("k", k_e), ("q", q_n), ("k", k_n)):
        expected = q_ref if name == "q" else k_ref
        diff = (actual.float() - expected.float()).abs()
        rel = diff / (expected.float().abs() + 1e-6)
        idx = torch.argmax(diff)
        unravel = torch.unravel_index(idx, diff.shape)
        print(
            f"{name} max_abs={diff.max().item():.3e} max_rel={rel.max().item():.3e} "
            f"at={tuple(i.item() for i in unravel)}",
            flush=True,
        )

    # direct eager vs native mismatch detail
    diff = (q_e.float() - q_n.float()).abs()
    idx = torch.unravel_index(torch.argmax(diff), diff.shape)
    i = tuple(i.item() for i in idx)
    print("eager-vs-native worst:", i)
    print("  eager:", q_e[i].item(), " native:", q_n[i].item(),
          " ref:", q_ref[i].item())
    print("  row head/token:", i[1], i[2], " col:", i[3])


def timed(impl, qkv, q, k, freqs, qw, kw, seq, warmup, iters):
    for _ in range(warmup):
        impl(qkv, q, k, freqs, qw, kw, seq)
    torch.xpu.synchronize()

    device_times: list[float] = []
    wall_times: list[float] = []
    for _ in range(iters):
        start_ev = torch.xpu.Event(enable_timing=True)
        end_ev = torch.xpu.Event(enable_timing=True)
        torch.xpu.synchronize()
        t0 = time.perf_counter()
        start_ev.record()
        impl(qkv, q, k, freqs, qw, kw, seq)
        end_ev.record()
        end_ev.synchronize()
        wall_times.append((time.perf_counter() - t0) * 1e3)
        device_times.append(start_ev.elapsed_time(end_ev))

    def med(vals):
        return statistics.median(vals)

    return med(device_times), med(wall_times), device_times, wall_times


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seq", nargs="+", type=int, default=[1024, 4096, 8192, 16384, 20685, 32768])
    parser.add_argument("--warmup", type=int, default=5)
    parser.add_argument("--iters", type=int, default=30)
    parser.add_argument("--check", action="store_true", help="run tolerance check only")
    parser.add_argument("--diag", type=int, metavar="SEQ", help="run mismatch diagnosis")
    parser.add_argument("--seed", type=int, default=1234)
    args = parser.parse_args()

    print(f"device: {torch.xpu.get_device_name(0)}  torch: {torch.__version__}")
    if args.diag:
        diagnose(args.diag, args.seed)
        return
    if args.check:
        for seq in args.seq:
            check_tolerance(seq)
        return
    print(f"warmup={args.warmup} iters={args.iters}")
    print(f"{'seq':>7} {'eager_dev':>10} {'eager_wall':>11} {'omni_dev':>10} "
          f"{'omni_wall':>11} {'speedup_dev':>11} {'speedup_wall':>12} {'dq':>10} {'dk':>10}")

    for seq in args.seq:
        qkv_e, q_e, k_e, freqs, qw, kw = make_contract(seq)
        qkv_n = qkv_e.clone()
        q_n = qkv_n[:, :7168].view(1, seq, 56, 128)
        k_n = qkv_n[:, 7168 : 2 * 7168].view(1, seq, 56, 128)

        eager_impl(qkv_e, q_e, k_e, freqs, qw, kw, seq)
        native_impl(qkv_n, q_n, k_n, freqs, qw, kw, seq)
        dq, dk = check(qkv_e, qkv_n, seq)

        # reset buffers for timing (values are irrelevant)
        qkv_e.zero_()
        qkv_n.zero_()
        e_dev, e_wall, _, _ = timed(eager_impl, qkv_e, q_e, k_e, freqs, qw, kw, seq, args.warmup, args.iters)
        n_dev, n_wall, _, _ = timed(native_impl, qkv_n, q_n, k_n, freqs, qw, kw, seq, args.warmup, args.iters)
        print(
            f"{seq:>7} {e_dev:>10.3f} {e_wall:>11.3f} {n_dev:>10.3f} "
            f"{n_wall:>11.3f} {e_dev / n_dev:>11.2f}x {e_wall / n_wall:>12.2f}x "
            f"{dq:>10.2e} {dk:>10.2e}",
            flush=True,
        )


if __name__ == "__main__":
    main()
