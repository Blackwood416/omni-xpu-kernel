"""A770 probe for the H3 phase-0 INT8 linear shape seen in comfy-info8.log.

Shape: x=[20683,5376] bf16, weight=[28672,5376] int8, per-channel weight
scales [28672,1], dynamic rowwise activation quantization, bf16 output.
This is the single largest instrumented cost in the first 200 blocks.
"""

import os
import statistics
import time

os.environ.setdefault("ONEDNN_VERBOSE", "profile,dispatch")
os.environ.setdefault("OMNI_DEBUG", "1")

import torch

M, K, N = 20683, 5376, 28672


def median_ms(fn, warmup=2, samples=5):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    vals = []
    for _ in range(samples):
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        vals.append((time.perf_counter() - t0) * 1000.0)
    return statistics.median(vals), min(vals), max(vals)


def main():
    print("torch", torch.__version__, "xpu", torch.xpu.is_available())
    print("M,K,N", M, K, N)
    torch.manual_seed(0)
    x = torch.randn(M, K, device="xpu", dtype=torch.bfloat16)
    weight = torch.randint(-127, 128, (N, K), device="xpu", dtype=torch.int8)
    weight_scale = torch.rand(N, 1, device="xpu", dtype=torch.float32) + 0.5

    from omni_xpu_kernel import int8

    def ours_full():
        return int8.int8_linear(
            x, weight, weight_scale, None, torch.bfloat16, False, 256
        )

    def ours_full_convrot():
        return int8.int8_linear(
            x, weight, weight_scale, None, torch.bfloat16, True, 256
        )

    x_int8, x_scale = int8.quantize_int8_rowwise(x)

    def ours_prequant():
        return int8.int8_linear_prequantized(
            x_int8, x_scale, weight, weight_scale, None, torch.bfloat16
        )

    def raw_mm():
        s32 = int8.mm_int8(x_int8, weight.t().contiguous())
        return s32

    def bf16_linear():
        return torch.nn.functional.linear(
            x, weight.to(torch.bfloat16) / weight_scale.to(torch.bfloat16)
        )

    def rotate():
        return int8.rotate_convrot(x, 256)

    def quant():
        return int8.quantize_int8_rowwise(x)

    print("cache before", int8.int8_cache_stats())
    med, lo, hi = median_ms(ours_full)
    print(f"int8_linear full       median={med:9.2f} ms  min={lo:8.2f} max={hi:8.2f}")
    med, lo, hi = median_ms(ours_full_convrot)
    print(f"int8_linear convrot    median={med:9.2f} ms  min={lo:8.2f} max={hi:8.2f}")
    med, lo, hi = median_ms(ours_prequant)
    print(f"int8_prequantized      median={med:9.2f} ms  min={lo:8.2f} max={hi:8.2f}")
    med, lo, hi = median_ms(raw_mm)
    print(f"mm_int8 raw s32        median={med:9.2f} ms  min={lo:8.2f} max={hi:8.2f}")
    med, lo, hi = median_ms(rotate)
    print(f"rotate_convrot g256    median={med:9.2f} ms  min={lo:8.2f} max={hi:8.2f}")
    med, lo, hi = median_ms(quant)
    print(f"quantize_rowwise       median={med:9.2f} ms  min={lo:8.2f} max={hi:8.2f}")
    med, lo, hi = median_ms(bf16_linear)
    print(f"bf16 linear ref        median={med:9.2f} ms  min={lo:8.2f} max={hi:8.2f}")
    print("cache after", int8.int8_cache_stats())

    # Correctness sanity against bf16 reference on a smaller row subset.
    ref = torch.nn.functional.linear(
        x[:8], weight.to(torch.bfloat16) / weight_scale.to(torch.bfloat16)
    )
    out = ours_full()[:8]
    err = (out.float() - ref.float()).abs()
    print(
        "sample max_abs_err",
        float(err.max()),
        "mean",
        float(err.mean()),
        "ref_absmax",
        float(ref.abs().max()),
    )


if __name__ == "__main__":
    main()
