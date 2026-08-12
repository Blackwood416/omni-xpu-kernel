"""A770/H3 main-phase microbenchmark (exact ComfyUI shapes).

Shapes taken from comfy-info14.log for the MiniMax H3 4-step workflow:

    seq     = 20683
    hidden  = 5376
    heads   = 56, dim = 128 (bf16 self attention)
    qkv     = (20683, 5376) x (21504, 5376)
    fc1     = (20683, 5376) x (28672, 5376)
    out     = (20683, 7168) x (5376, 7168)

Measures:
    * omni ESIMD SDP on BLHD vs the adapter's real BHLD->permute->SDP->reshape
      pipeline vs Torch SDPA on BHLD
    * omni int8_linear for qkv/fc1/out (with and without ConvRot) vs a bf16
      Torch linear reference
    * omni ESIMD RMSNorm

Run with the ComfyUI portable interpreter, GPU otherwise idle:

    C:\\Users\\Administrator\\ComfyUI_windows_portable\\python_embeded\\python.exe ^
        omni\\omni_xpu_kernel\\benchmarks\\h3_main_phase_a770.py
"""

from __future__ import annotations

import statistics
import time

import torch

try:
    import omni_xpu_kernel as omni
    from omni_xpu_kernel import int8 as omni_int8
    from omni_xpu_kernel import norm as omni_norm
    from omni_xpu_kernel import sdp as omni_sdp
except ImportError:
    omni = omni_int8 = omni_norm = omni_sdp = None


SEQ = 20683
HIDDEN = 5376
FFN = 28672
QKV = 21504
OUT_IN = 7168
HEADS = 56
DIM = 128


def bench(fn, n=20, warmup=5):
    for _ in range(warmup):
        fn()
    torch.xpu.synchronize()
    ts = []
    for _ in range(n):
        t0 = time.perf_counter()
        fn()
        torch.xpu.synchronize()
        ts.append((time.perf_counter() - t0) * 1e3)
    ts.sort()
    return {
        "median": statistics.median(ts),
        "p10": ts[int(n * 0.1)],
        "p90": ts[int(n * 0.9)],
    }


def fmt(name, r):
    print(f"{name:55s} median={r['median']:9.1f}ms  p10={r['p10']:9.1f}  p90={r['p90']:9.1f}")


def main():
    if omni is None:
        raise SystemExit("omni_xpu_kernel not importable in this interpreter")
    print(f"omni_xpu_kernel {omni.__version__} target={omni.__xpu_target__}")
    device = torch.device("xpu")

    # --- inputs -----------------------------------------------------------------
    x = (torch.randn(SEQ, HIDDEN, dtype=torch.bfloat16) * 0.1).to(device)
    x_out = (torch.randn(SEQ, OUT_IN, dtype=torch.bfloat16) * 0.1).to(device)

    def make_w(n, k):
        wf = (torch.randn(n, k) * 0.02).to(device)
        wq, ws = omni_int8.quantize_int8_tensorwise(wf)
        return wq, ws, wf

    w_qkv, s_qkv, wr_qkv = make_w(QKV, HIDDEN)
    w_fc1, s_fc1, wr_fc1 = make_w(FFN, HIDDEN)
    w_out, s_out, wr_out = make_w(HIDDEN, OUT_IN)
    # fc2: 28672-wide fc1 output is SwiGLU'd to 14336, then projected to 5376.
    w_fc2, s_fc2, wr_fc2 = make_w(HIDDEN, 14336)

    # BLHD contiguous (what the sidecar expects)
    q_blhd = (torch.randn(1, SEQ, HEADS, DIM, dtype=torch.bfloat16) * 0.1).to(device)
    k_blhd = (torch.randn(1, SEQ, HEADS, DIM, dtype=torch.bfloat16) * 0.1).to(device)
    v_blhd = (torch.randn(1, SEQ, HEADS, DIM, dtype=torch.bfloat16) * 0.1).to(device)
    # BHLD contiguous (what optimized_attention receives after reshape)
    q_bhld = q_blhd.permute(0, 2, 1, 3).contiguous()
    k_bhld = k_blhd.permute(0, 2, 1, 3).contiguous()
    v_bhld = v_blhd.permute(0, 2, 1, 3).contiguous()

    norm_w = torch.randn(HIDDEN, dtype=torch.bfloat16, device=device) + 1.0

    # --- attention ---------------------------------------------------------------
    print("\n== attention (1,20683,56,128 bf16) ==")
    fmt("omni sdp (BLHD direct)", bench(lambda: omni_sdp.sdp(q_blhd, k_blhd, v_blhd)))

    def adapter_pipeline():
        q = q_bhld.permute(0, 2, 1, 3).contiguous()
        k = k_bhld.permute(0, 2, 1, 3).contiguous()
        v = v_bhld.permute(0, 2, 1, 3).contiguous()
        out = omni_sdp.sdp(q, k, v)
        return out.reshape(1, SEQ, HEADS * DIM)

    fmt("omni adapter pipeline (permute x3 + sdp + reshape)", bench(adapter_pipeline))
    fmt(
        "torch SDPA (BHLD)",
        bench(
            lambda: torch.nn.functional.scaled_dot_product_attention(
                q_bhld, k_bhld, v_bhld, dropout_p=0.0, is_causal=False
            )
        ),
    )

    # --- int8 linears -------------------------------------------------------------
    print("\n== int8 linears (omni_dg2_compat) ==")
    for label, xx, ww, ss in (
        ("qkv (5376->21504)", x, w_qkv, s_qkv),
        ("fc1 (5376->28672)", x, w_fc1, s_fc1),
        ("out (7168->5376)", x_out, w_out, s_out),
    ):
        fmt(
            f"omni int8_linear {label} (no convrot)",
            bench(
                lambda xx=xx, ww=ww, ss=ss: omni_int8.int8_linear(
                    xx, ww, ss, bias=None, out_dtype=torch.bfloat16,
                    convrot=False, convrot_groupsize=256, input_act=None,
                )
            ),
        )
        fmt(
            f"omni int8_linear {label} (convrot g256)",
            bench(
                lambda xx=xx, ww=ww, ss=ss: omni_int8.int8_linear(
                    xx, ww, ss, bias=None, out_dtype=torch.bfloat16,
                    convrot=True, convrot_groupsize=256, input_act=None,
                )
            ),
        )
        fmt(
            f"torch bf16 linear {label} (dequant ref)",
            bench(lambda xx=xx, wf=wr_qkv if ww is w_qkv else (wr_fc1 if ww is w_fc1 else wr_out): torch.nn.functional.linear(xx, wf.to(torch.bfloat16))),
        )

    # --- rms norm --------------------------------------------------------------
    print("\n== rms_norm (20683,5376) bf16 ==")
    fmt("omni esimd rms_norm", bench(lambda: omni_norm.rms_norm(norm_w, x)))
    x_fc1 = (torch.randn(SEQ, 2 * 14336, dtype=torch.bfloat16) * 0.1).to(device)
    fmt(
        "omni int8_linear fc2 (14336->5376, swiglu+convrot)",
        bench(
            lambda: omni_int8.int8_linear(
                x_fc1,
                w_fc2, s_fc2, bias=None, out_dtype=torch.bfloat16,
                convrot=True, convrot_groupsize=256, input_act="swiglu",
            )
        ),
    )

    # --- one full block pipeline (async, one sync) ------------------------------
    print("\n== full block pipeline x20 (async, one sync per block) ==")

    def block():
        h = omni_norm.rms_norm(norm_w, x)
        h = omni_int8.int8_linear(
            h, w_qkv, s_qkv, bias=None, out_dtype=torch.bfloat16,
            convrot=True, convrot_groupsize=256, input_act=None,
        )
        # The sidecar expects contiguous [B, L, H, D]; h rows are tokens, so
        # the per-head slices are already contiguous BLHD views.
        q = h[:, :HEADS * DIM].reshape(1, SEQ, HEADS, DIM).contiguous()
        k = h[:, HEADS * DIM:2 * HEADS * DIM].reshape(1, SEQ, HEADS, DIM).contiguous()
        v = h[:, 2 * HEADS * DIM:].reshape(1, SEQ, HEADS, DIM).contiguous()
        a = omni_sdp.sdp(q, k, v).reshape(SEQ, HEADS * DIM)
        h = omni_int8.int8_linear(
            a, w_out, s_out, bias=None, out_dtype=torch.bfloat16,
            convrot=True, convrot_groupsize=256, input_act=None,
        )
        h = omni_norm.rms_norm(norm_w, h)
        h = omni_int8.int8_linear(
            h, w_fc1, s_fc1, bias=None, out_dtype=torch.bfloat16,
            convrot=True, convrot_groupsize=256, input_act=None,
        )
        return omni_int8.int8_linear(
            h, w_fc2, s_fc2, bias=None, out_dtype=torch.bfloat16,
            convrot=True, convrot_groupsize=256, input_act="swiglu",
        )

    for _ in range(3):
        block()
    torch.xpu.synchronize()
    t0 = time.perf_counter()
    for _ in range(20):
        block()
    torch.xpu.synchronize()
    dt = (time.perf_counter() - t0) / 20
    print(f"one block (rms+qkv+attn+out+rms+fc1) = {dt * 1e3:.1f} ms")


if __name__ == "__main__":
    main()
