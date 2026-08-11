"""DG2 v3/v2/torch SDPA wall-time benchmark (A770, D=128).

Protocol: warmup, >= 20 samples, median + p10/p90, correctness check per
shape. Call with the target sidecar installed (v2 or v3 build) and compare
against torch SDPA in the same process.
"""

import statistics
import sys
import time

import torch

import omni_xpu_kernel.sdp as sdp


SHAPES = [
    # (qlen, kvlen, heads) — MiniMax H3 GGUF prefill-style D=128 shapes
    (512, 512, 32),
    (1024, 1024, 32),
    (2048, 2048, 32),
    (4096, 4096, 32),
    (8192, 8192, 32),
]


def bench(mode, q, k, v, samples=30):
    # correctness first
    ref = torch.nn.functional.scaled_dot_product_attention(
        q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
    ).permute(0, 2, 1, 3)
    torch.xpu.synchronize()
    if mode == "torch":
        fn = lambda: torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3), k.permute(0, 2, 1, 3), v.permute(0, 2, 1, 3)
        )
    else:
        fn = lambda: sdp.sdp(q, k, v)
    out = fn()
    if mode == "torch":
        out = out.permute(0, 2, 1, 3)
    torch.xpu.synchronize()
    err = (out.float() - ref.float()).abs().max().item()

    for _ in range(5):
        fn()
    torch.xpu.synchronize()
    times = []
    for _ in range(samples):
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        times.append((time.perf_counter() - t0) * 1e3)
    times.sort()
    med = statistics.median(times)
    p10 = times[int(len(times) * 0.1)]
    p90 = times[int(len(times) * 0.9)]
    return med, p10, p90, err


def main():
    mode = sys.argv[1] if len(sys.argv) > 1 else "sidecar"
    dtype = torch.float16
    print(f"mode={mode} sidecar_available={sdp.is_available()}")
    for qlen, kvlen, h in SHAPES:
        torch.manual_seed(0)
        q = (torch.randn(1, qlen, h, 128, dtype=dtype) * 0.5).to("xpu")
        k = (torch.randn(1, kvlen, h, 128, dtype=dtype) * 0.5).to("xpu")
        v = (torch.randn(1, kvlen, h, 128, dtype=dtype) * 0.5).to("xpu")
        med, p10, p90, err = bench(mode, q, k, v)
        print(
            f"L={qlen:5d} kv={kvlen:5d} H={h:2d}: "
            f"median={med:.3f} ms p10={p10:.3f} p90={p90:.3f} err={err:.6f}"
        )


if __name__ == "__main__":
    main()
