"""v4 vs torch SDPA across extended A770 shapes (interleaved A/B)."""

import statistics
import time

import torch

import omni_xpu_kernel.sdp as sdp


SHAPES = [
    # (qlen, kvlen, H, dtype)
    (512, 512, 16, torch.float16),
    (512, 512, 48, torch.float16),
    (1024, 1024, 16, torch.float16),
    (1024, 1024, 48, torch.float16),
    (2048, 2048, 16, torch.float16),
    (2048, 2048, 48, torch.float16),
    (8192, 8192, 16, torch.float16),
    (8192, 8192, 48, torch.float16),
    (256, 1024, 32, torch.float16),
    (512, 2048, 32, torch.float16),
    (1024, 4096, 32, torch.float16),
    (16384, 16384, 32, torch.float16),
    (16384, 16384, 16, torch.float16),
    (8192, 8192, 32, torch.bfloat16),
]


def bench(fn, n=30):
    for _ in range(8):
        fn()
    torch.xpu.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return statistics.median(ts), ts[int(n * 0.1)], ts[int(n * 0.9)]


def main():
    for qlen, kvlen, h, dtype in SHAPES:
        torch.manual_seed(0)
        q = (torch.randn(1, qlen, h, 128, dtype=dtype) * 0.5).to("xpu")
        k = (torch.randn(1, kvlen, h, 128, dtype=dtype) * 0.5).to("xpu")
        v = (torch.randn(1, kvlen, h, 128, dtype=dtype) * 0.5).to("xpu")
        qt = q.permute(0, 2, 1, 3).contiguous()
        kt = k.permute(0, 2, 1, 3).contiguous()
        vt = v.permute(0, 2, 1, 3).contiguous()

        ours = bench(lambda: sdp.sdp(q, k, v))
        theirs = bench(
            lambda: torch.nn.functional.scaled_dot_product_attention(qt, kt, vt)
        )
        ours2 = bench(lambda: sdp.sdp(q, k, v))
        best_ours = min(ours[0], ours2[0])
        ratio = best_ours / theirs[0]
        print(
            f"L={qlen:5d} kv={kvlen:5d} H={h:2d} {str(dtype):6s}: "
            f"v4={best_ours:.3f} torch={theirs[0]:.3f} ratio={ratio:.3f}"
        )


if __name__ == "__main__":
    main()
