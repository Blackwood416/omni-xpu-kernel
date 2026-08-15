"""DG2 fused ConvRot + rowwise INT8 quantization correctness/benchmark.

Run standalone (``python tests/test_int8_convrot_fused_dg2.py``) or under
pytest. Requires the A770/DG2 wheel exposing
``omni_xpu_kernel.int8.quantize_int8_convrot_fused``.

The reference composes the exact current path: cached Hadamard matmul (with
bf16 materialization) followed by rowwise INT8 quantization.
"""

from __future__ import annotations

import argparse
import statistics
import time

import torch

from omni_xpu_kernel import int8 as omni_int8


def _hadamard(group_size: int, dtype: torch.dtype, device: torch.device):
    h4 = torch.tensor(
        [1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, -1, 1, 1, 1],
        dtype=dtype,
        device=device,
    ).reshape(4, 4)
    h = h4
    size = 4
    while size < group_size:
        h = (
            h.unsqueeze(1).unsqueeze(3) * h4.unsqueeze(0).unsqueeze(2)
        ).reshape(size * 4, size * 4)
        size *= 4
    return (h / group_size ** 0.5).contiguous()


def _reference(x: torch.Tensor, group_size: int):
    m, k = x.shape
    grouped = x.reshape(m, k // group_size, group_size)
    rotated = torch.matmul(grouped, _hadamard(group_size, x.dtype, x.device))
    rotated = rotated.reshape(m, k)
    row_max = rotated.abs().amax(dim=1, keepdim=True).clamp_min(1e-30)
    scale = row_max / 127.0
    q = torch.clamp(torch.round(rotated / scale), -128, 127).to(torch.int8)
    return q, scale.squeeze(1)


def _check_case(m, k, group_size, dtype, seed):
    gen = torch.Generator(device="cpu").manual_seed(seed)
    x = torch.randn((m, k), dtype=dtype, generator=gen).to("xpu")
    q_ref, s_ref = _reference(x, group_size)
    q_act, s_act = omni_int8.quantize_int8_convrot_fused(x, group_size)
    s_act = s_act.reshape(-1)

    assert q_act.shape == q_ref.shape
    assert s_act.shape == s_ref.shape
    diff = (q_act.to(torch.int16) - q_ref.to(torch.int16)).abs()
    exact = (diff == 0).float().mean().item()
    s_rel = ((s_act - s_ref).abs() / s_ref.clamp_min(1e-30)).max().item()
    print(
        f"m={m:>5} k={k:>6} g={group_size} {str(dtype):>7}: "
        f"exact={exact * 100:.5f}% max_qdiff={diff.max().item()} max_srel={s_rel:.2e}",
        flush=True,
    )
    # Skipping the composed path's bf16 rotation materialization can move a
    # rare element by up to 2 INT8 LSB; the vast majority stay bit-identical.
    assert diff.max().item() <= 2, f"q differs by {diff.max().item()} at {m}x{k}"
    # The fused butterfly and torch's bf16 matmul round to the same bf16
    # rotated values in nearly all positions; a different fp32 reduction
    # order can flip the last bf16 bit on a few percent of elements, which
    # shows up as an INT8 difference of exactly 1.
    assert exact >= 0.85, f"too many mismatches: {exact * 100:.4f}% exact"
    assert s_rel <= 0.02, f"scale mismatch {s_rel}"


def _bench(rows_list, cols_list):
    print(f"{'rows':>7} {'cols':>7} {'composed_ms':>12} {'fused_ms':>10} "
          f"{'speedup':>8} {'GB_s':>8}")
    for rows in rows_list:
        for cols in cols_list:
            x = torch.randn((rows, cols), dtype=torch.bfloat16, device="xpu")
            q_ref, s_ref = _reference(x, 256)

            for _ in range(3):
                omni_int8.quantize_int8_convrot_fused(x, 256)
            torch.xpu.synchronize()
            comp = []
            fused = []
            for _ in range(20):
                torch.xpu.synchronize()
                t0 = time.perf_counter()
                q_ref, s_ref = _reference(x, 256)
                torch.xpu.synchronize()
                comp.append((time.perf_counter() - t0) * 1e3)

                torch.xpu.synchronize()
                t0 = time.perf_counter()
                omni_int8.quantize_int8_convrot_fused(x, 256)
                torch.xpu.synchronize()
                fused.append((time.perf_counter() - t0) * 1e3)

            cm, fm = statistics.median(comp), statistics.median(fused)
            traffic = rows * cols * 2 / 1e9
            print(
                f"{rows:>7} {cols:>7} {cm:>12.2f} {fm:>10.2f} "
                f"{cm / fm:>7.2f}x {traffic / (fm / 1e3):>8.0f}",
                flush=True,
            )


def _debug_small():
    """Print q for a tiny fully-visible case to diagnose the butterfly."""
    m, k, g = 1, 256, 256
    x = torch.arange(k, dtype=torch.bfloat16, device="xpu").reshape(m, k) / 8.0
    q_ref, s_ref = _reference(x, g)
    q_act, s_act = omni_int8.quantize_int8_convrot_fused(x, g)
    print("scale ref/act:", s_ref.item(), s_act.item())
    print("q_ref :", q_ref[0, :32].tolist())
    print("q_act :", q_act[0, :32].tolist())
    print("q_ref[32:64]:", q_ref[0, 32:64].tolist())
    print("q_act[32:64]:", q_act[0, 32:64].tolist())
    print("q_ref[96:128]:", q_ref[0, 96:128].tolist())
    print("q_act[96:128]:", q_act[0, 96:128].tolist())


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--bench", action="store_true")
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--rows", nargs="+", type=int, default=[1024, 4096, 20685])
    parser.add_argument("--cols", nargs="+", type=int, default=[7168, 14336])
    args = parser.parse_args()

    print("device:", torch.xpu.get_device_name(0), "torch:", torch.__version__)
    if args.debug:
        _debug_small()
        return
    cases = [
        (1, 14336, 256, torch.bfloat16, 1),
        (37, 7168, 256, torch.bfloat16, 2),
        (256, 14336, 256, torch.bfloat16, 3),
        (20685, 14336, 256, torch.bfloat16, 4),
        (256, 7168, 256, torch.float16, 5),
        (64, 4096, 64, torch.bfloat16, 6),
    ]
    for case in cases:
        _check_case(*case)
    print("correctness OK")

    if args.bench:
        _bench(args.rows, args.cols)


if __name__ == "__main__":
    main()
