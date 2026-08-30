"""cute / CUTLASS-SYCL fused Flash Attention (torch op).

Drop-in for :func:`omni_xpu_kernel.sdp.sdp` — same signature and layout::

    from omni_xpu_kernel import cute
    out = cute.sdp(q, k, v)   # self-attn [B, L, H, D] (B==1, D==128), fp16/bf16

PTL-H and BMG wheels also expose a workflow-tuned D120 entry point that
consumes dense packed-BHLD or BLHD-backed BHLD layouts without intermediate
copies::

    out = cute.sdp_bhld_d120(q, k, v)  # [B, H, L, 120]

BMG wheels additionally expose the exact Wan 2.2 14B T2V Turbo 720p
cross-attention contract through ``sdp_wan22_cross`` and a batched,
rectangular D128 BHLD entry point through ``sdp_bhld_d128``. The structural
MiniMax H3 VideoVAE D64 tile family is exposed separately through
``sdp_minimax_h3_vae_d64``.

Unlike the ESIMD ``sdp`` kernel (fp16 accumulator + adaptive V-scaling), the cute
FMHA accumulates QK and P*V in fp32, so it does not overflow on large-magnitude
activations (e.g. Qwen-Image). It is AOT-compiled into ``cute_fmha_torch.so`` and
exposes ``torch.ops.cute_fmha.sdp``. The generic entry point accepts
self-attention only; validated rectangular workflow contracts use dedicated
entry points.
"""

import glob
import os

import torch

_loaded = False


def _find_so():
    """Locate the cute FMHA .so.

    setuptools names it with the Python ABI suffix (cute_fmha_torch.cpython-*.so);
    a hand build may drop a plain cute_fmha_torch.so. OMNI_CUTE_FMHA_SO overrides.
    """
    env = os.environ.get("OMNI_CUTE_FMHA_SO", "")
    if env:
        return env
    here = os.path.dirname(os.path.abspath(__file__))
    cands = [os.path.join(here, "cute_fmha_torch.so")]
    cands += sorted(glob.glob(os.path.join(here, "cute_fmha_torch*.so")))
    for c in cands:
        if os.path.exists(c):
            return c
    return ""


def _ensure_loaded():
    global _loaded
    if _loaded:
        return
    so = _find_so()
    if not so or not os.path.exists(so):
        raise ImportError(
            "cute_fmha_torch .so not found next to omni_xpu_kernel.cute "
            "(set OMNI_CUTE_FMHA_SO to override)"
        )
    torch.ops.load_library(so)
    _loaded = True


def is_available():
    try:
        _ensure_loaded()
        return True
    except Exception:
        return False


def sdp(q: torch.Tensor, k: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Fused scaled-dot-product attention. Inputs [B, L, H, D] (B==1, D==128)."""
    _ensure_loaded()
    return torch.ops.cute_fmha.sdp(q, k, v)


# ──────────────────────────────────────────────────────────────────────────────
# Sol-Attn (sparse attention) — A770/DG2 ESIMD backend.
# 接口对齐 xiangyuT/ComfyUI-SolAttn_xpu 的 cute.sol_attn / supports_sol_attn。
# prepare 用 torch 在主机侧算路由（质心/阈值/掩码），forward 调 lgrf_sol_attn
# sidecar（DG2 ESIMD，WG=32、fp32 累加）。
# 契约：B=1、bf16、BTHD、D=128、自注意力、无 mask。
# ──────────────────────────────────────────────────────────────────────────────
import ctypes
from pathlib import Path

_sol_attn_lib = None
_SOL_ATTN_BLOCK = 64


def _sol_attn_library():
    global _sol_attn_lib
    if _sol_attn_lib is not None:
        return _sol_attn_lib
    here = Path(__file__).resolve().parent.parent / "lgrf_uni"
    pattern = "lgrf_sol_attn.pyd" if os.name == "nt" else "lgrf_sol_attn.so"
    cands = sorted(here.glob(pattern))
    if not cands:
        raise RuntimeError("lgrf_sol_attn sidecar not found in %s" % here)
    lib = ctypes.CDLL(str(cands[0]))
    lib.sol_attn_forward.argtypes = [
        ctypes.c_void_p] * 11 + [ctypes.c_int] * 5 + [ctypes.c_float]
    lib.sol_attn_forward.restype = None
    _sol_attn_lib = lib
    return lib


_sol_attn_sum_lib = None
_sol_attn_exact_lib = None


def _sol_attn_sum_library():
    """DPAS 摘要 sidecar（lgrf_sol_attn_sum*.pyd）。"""
    global _sol_attn_sum_lib
    if _sol_attn_sum_lib is not None:
        return _sol_attn_sum_lib
    here = Path(__file__).resolve().parent.parent / "lgrf_uni"
    cands = sorted(here.glob("lgrf_sol_attn_sum*.pyd" if os.name == "nt"
                             else "lgrf_sol_attn_sum*.so"))
    if not cands:
        raise RuntimeError("lgrf_sol_attn_sum sidecar not found in %s" % here)
    _sol_attn_sum_lib = ctypes.CDLL(str(cands[0]))
    return _sol_attn_sum_lib


def _sol_attn_exact_library():
    """FMA 精确 sidecar（lgrf_sol_attn_exact*.pyd）。"""
    global _sol_attn_exact_lib
    if _sol_attn_exact_lib is not None:
        return _sol_attn_exact_lib
    here = Path(__file__).resolve().parent.parent / "lgrf_uni"
    cands = sorted(here.glob("lgrf_sol_attn_exact*.pyd" if os.name == "nt"
                             else "lgrf_sol_attn_exact*.so"))
    if not cands:
        raise RuntimeError("lgrf_sol_attn_exact sidecar not found in %s"
                          % here)
    _sol_attn_exact_lib = ctypes.CDLL(str(cands[0]))
    return _sol_attn_exact_lib


def supports_sol_attn() -> bool:
    """Whether the DG2 Sol-Attn ESIMD sidecar is packaged."""
    try:
        _sol_attn_library()
        return True
    except Exception:
        return False


def _sol_attn_prepare(q, k, v, scale, tau, sink_blocks, sink_q):
    """路由准备：返回 (kc, vc, routes)。
    kc: [B,H,N,D] bf16 块均值；vc: [B,H,N,D] bf16 V 块和；
    routes: [B,H,QN,N] uint8 精确掩码。语义与 CUDA Sol-Attn 一致。
    """
    B, T, H, D = q.shape
    NB = (T + _SOL_ATTN_BLOCK - 1) // _SOL_ATTN_BLOCK
    scale_log2 = float(scale) * 1.4426950408889634
    pad = NB * _SOL_ATTN_BLOCK - T

    def block_reduce(x, mode):
        xf = x.float()
        if pad:
            xf = torch.nn.functional.pad(xf, (0, 0, 0, 0, 0, pad))
        xb = xf.reshape(B, NB, _SOL_ATTN_BLOCK, H, D)
        s = xb.sum(dim=2)  # [B,N,H,D]
        if mode == "mean":
            lens = torch.full((NB,), _SOL_ATTN_BLOCK, device=x.device,
                              dtype=torch.float32)
            lens[-1] = T - (NB - 1) * _SOL_ATTN_BLOCK
            s = s / lens.view(1, NB, 1, 1)
        return s.permute(0, 2, 1, 3)  # [B,H,N,D]

    qc = block_reduce(q, "mean")  # f32
    # block_reduce 返回 permute 视图（[B,N,H,D] 内存布局），kernel 按连续
    # [B,H,N,D] 寻址；H>=2 时非连续会让 kc/vc 读错位，这里强制连续。
    kc = block_reduce(k, "mean").to(torch.bfloat16).contiguous()
    vc = block_reduce(v, "sum").to(torch.bfloat16).contiguous()
    k_mean = kc.float().mean(dim=2, keepdim=True)
    k_var = (kc.float().square().mean(dim=2, keepdim=True) -
             k_mean.square()).clamp_min(0)
    raw_mean = (qc * k_mean).sum(dim=-1)
    raw_var = (qc.square() * k_var).sum(dim=-1)
    thr = raw_mean * scale_log2 + tau * torch.sqrt(
        raw_var * scale_log2 * scale_log2 + 1e-6)
    if sink_q[1] > sink_q[0]:
        thr[:, :, sink_q[0]:sink_q[1]] = float("-inf")
    qb = torch.arange(NB, device=q.device)
    score = torch.einsum("bhqd,bhkd->bhqk", qc, kc.float()) * scale_log2
    routes = (score > thr.unsqueeze(-1)) | (
        (qb[:, None] - qb[None, :]).abs()[None, None] <= 1)
    if sink_blocks[1] > sink_blocks[0]:
        routes[:, :, :, sink_blocks[0]:sink_blocks[1]] = True
    return kc, vc, routes.to(torch.uint8)


def _sol_attn_csr(routes):
    """由路由掩码构建精确/近似块 CSR 索引（kernel 无分支循环用）。"""
    B, H, QN, N = routes.shape
    rb = routes.bool().reshape(B * H * QN, N)
    n_exact = rb.sum(dim=1, dtype=torch.int32)          # [rows]
    n_approx = N - n_exact
    ar = torch.arange(N, device=routes.device).unsqueeze(0)
    order_ex = torch.argsort(~rb, dim=1, stable=True).to(torch.int32)
    order_ap = torch.argsort(rb, dim=1, stable=True).to(torch.int32)
    exact_idx = torch.where(ar < n_exact.unsqueeze(1), order_ex, 0)
    approx_idx = torch.where(ar < n_approx.unsqueeze(1), order_ap, 0)
    shape = (B, H, QN, N)
    return (exact_idx.reshape(shape), approx_idx.reshape(shape),
            n_exact.reshape(B, H, QN), n_approx.reshape(B, H, QN))


def sol_attn(q, k, v, *, scale=None, tau=1.0,
             sink_blocks=(0, 0), sink_q=(0, 0)):
    """Sparse Sol-Attn (BF16 BTHD D128, B==1, self-attention, no mask)."""
    if q.dim() != 4 or q.shape[0] != 1 or q.shape[-1] != 128:
        raise ValueError("sol_attn requires [1, T, H, 128] bf16 inputs")
    if q.dtype != torch.bfloat16:
        raise ValueError("sol_attn is bf16-only")
    if q.shape != k.shape or q.shape != v.shape:
        raise ValueError("q/k/v must share shape")
    if scale is None:
        scale = q.shape[-1] ** -0.5
    # H>=2 原生直读（kc/vc 已在 _sol_attn_prepare 中 contiguous）；
    # 2026-08-30 定位：之前"多 head miscompile"实为 kc/vc permute 非连续
    # 视图导致 kernel 寻址错位，H=1 时两种布局重合所以一直通过。
    q = q.contiguous()
    k = k.contiguous()
    v = v.contiguous()
    B, T, H, D = q.shape
    NB = (T + _SOL_ATTN_BLOCK - 1) // _SOL_ATTN_BLOCK
    kc, vc, routes = _sol_attn_prepare(
        q, k, v, scale, float(tau),
        tuple(int(x) for x in sink_blocks),
        tuple(int(x) for x in sink_q))
    queue = torch.xpu.current_stream().sycl_queue
    try:
        # ── A770 两阶段路径：DPAS 摘要（近似块）+ FMA 精确（稀疏保留块）──
        # 摘要：pack_q/pack_kc/pack_vc + summary，输出 m/l/acc（bf16 在线
        # softmax 初始值）；精确：per-qblock CSR + init 合并，输出最终结果。
        sum_lib = _sol_attn_sum_library()
        exact_lib = _sol_attn_exact_library()
    except Exception:
        sum_lib = exact_lib = None
    if sum_lib is not None and exact_lib is not None:
        n_pad = ((NB + 63) // 64) * 64
        q_tiles = (T + 255) // 256
        qn_pad = q_tiles * 4
        kc_p = torch.nn.functional.pad(
            kc, (0, 0, 0, n_pad - NB)).contiguous()
        vc_p = torch.nn.functional.pad(
            vc, (0, 0, 0, n_pad - NB)).contiguous()
        routes_p = torch.nn.functional.pad(
            routes, (0, n_pad - NB, 0, qn_pad - routes.shape[2])
        ).contiguous()
        packed_q = torch.empty(H * q_tiles * 32 * 8 * 128,
                               device=q.device, dtype=torch.bfloat16)
        n_groups = n_pad // 8
        pg_total = n_pad // 16
        packed_kc = torch.empty(H * n_groups * 8 * 64, device=q.device,
                                dtype=torch.int32)
        packed_vc = torch.empty(H * pg_total * 8 * 2 * 64,
                                device=q.device, dtype=torch.int32)
        m = torch.empty(T, H, device=q.device, dtype=torch.float32)
        l = torch.empty(T, H, device=q.device, dtype=torch.float32)
        acc = torch.empty(T, H, 128, device=q.device, dtype=torch.bfloat16)
        sum_lib.sol_attn_pack_q(
            ctypes.c_void_p(queue), ctypes.c_void_p(q.data_ptr()),
            ctypes.c_void_p(packed_q.data_ptr()),
            ctypes.c_int(T), ctypes.c_int(H))
        sum_lib.sol_attn_pack_kc(
            ctypes.c_void_p(queue), ctypes.c_void_p(kc_p.data_ptr()),
            ctypes.c_void_p(packed_kc.data_ptr()),
            ctypes.c_int(H), ctypes.c_int(n_groups))
        sum_lib.sol_attn_pack_vc(
            ctypes.c_void_p(queue), ctypes.c_void_p(vc_p.data_ptr()),
            ctypes.c_void_p(packed_vc.data_ptr()),
            ctypes.c_int(H), ctypes.c_int(pg_total))
        sum_lib.sol_attn_summary(
            ctypes.c_void_p(queue),
            ctypes.c_void_p(packed_q.data_ptr()),
            ctypes.c_void_p(packed_kc.data_ptr()),
            ctypes.c_void_p(packed_vc.data_ptr()),
            ctypes.c_void_p(routes_p.data_ptr()),
            ctypes.c_void_p(m.data_ptr()), ctypes.c_void_p(l.data_ptr()),
            ctypes.c_void_p(acc.data_ptr()),
            ctypes.c_int(T), ctypes.c_int(T), ctypes.c_int(H),
            ctypes.c_int(NB), ctypes.c_int(qn_pad), ctypes.c_int(n_pad))
        exact_idx, _, n_exact, _ = _sol_attn_csr(routes)
        out = torch.empty_like(q)
        exact_lib.sol_attn_exact(
            ctypes.c_void_p(queue),
            ctypes.c_void_p(q.data_ptr()), ctypes.c_void_p(k.data_ptr()),
            ctypes.c_void_p(v.data_ptr()),
            ctypes.c_void_p(exact_idx.data_ptr()),
            ctypes.c_void_p(n_exact.data_ptr()),
            ctypes.c_void_p(m.data_ptr()), ctypes.c_void_p(l.data_ptr()),
            ctypes.c_void_p(acc.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(T), ctypes.c_int(T), ctypes.c_int(H),
            ctypes.c_int(NB), ctypes.c_int(NB), ctypes.c_float(float(scale)))
        torch.xpu.synchronize()
        return out

    # 回退：单 kernel FMA 全流程
    lib = _sol_attn_library()
    exact_idx, approx_idx, n_exact, n_approx = _sol_attn_csr(routes)
    out = torch.empty_like(q)
    args = [ctypes.c_void_p(queue),
            ctypes.c_void_p(q.data_ptr()),
            ctypes.c_void_p(k.data_ptr()),
            ctypes.c_void_p(v.data_ptr()),
            ctypes.c_void_p(kc.data_ptr()),
            ctypes.c_void_p(vc.data_ptr()),
            ctypes.c_void_p(exact_idx.data_ptr()),
            ctypes.c_void_p(approx_idx.data_ptr()),
            ctypes.c_void_p(n_exact.data_ptr()),
            ctypes.c_void_p(n_approx.data_ptr()),
            ctypes.c_void_p(out.data_ptr()),
            ctypes.c_int(T), ctypes.c_int(T),
            ctypes.c_int(H), ctypes.c_int(NB),
            ctypes.c_int(NB),
            ctypes.c_float(float(scale))]
    lib.sol_attn_forward(*args)
    torch.xpu.synchronize()
    return out


def supports_wan22_cross() -> bool:
    """Whether this BMG sidecar exports the exact Wan 2.2 cross kernel."""
    try:
        _ensure_loaded()
        return hasattr(torch.ops.cute_fmha, "sdp_wan22_cross")
    except Exception:
        return False


def sdp_wan22_cross(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Wan 2.2 14B T2V Turbo 720p FP16 cross-attention."""
    _ensure_loaded()
    if not hasattr(torch.ops.cute_fmha, "sdp_wan22_cross"):
        raise RuntimeError(
            "CUTE Wan 2.2 cross-attention kernel is unavailable "
            "in this sidecar"
        )
    return torch.ops.cute_fmha.sdp_wan22_cross(q, k, v)


def supports_d128_bhld() -> bool:
    """Whether this BMG sidecar exports batched/rectangular D128 BHLD."""
    try:
        _ensure_loaded()
        return hasattr(torch.ops.cute_fmha, "sdp_bhld_d128")
    except Exception:
        return False


def sdp_bhld_d128(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Attention for supported dense or H3 QKV-backed ``[B,H,L,128]`` inputs."""
    _ensure_loaded()
    if not hasattr(torch.ops.cute_fmha, "sdp_bhld_d128"):
        raise RuntimeError(
            "CUTE D128 BHLD attention kernel is unavailable "
            "in this sidecar"
        )
    return torch.ops.cute_fmha.sdp_bhld_d128(q, k, v)


def supports_minimax_h3_vae_d64() -> bool:
    """Whether this BMG sidecar exports MiniMax H3 VideoVAE D64 tiles."""
    try:
        _ensure_loaded()
        return hasattr(torch.ops.cute_fmha, "sdp_minimax_h3_vae_d64")
    except Exception:
        return False


def sdp_minimax_h3_vae_d64(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """MiniMax H3 VideoVAE FP16 ``[1,32,S,64]`` tile attention.

    ``S`` is derived by the decoder from the temporal/spatial tile extent;
    Q/K use the ``H*D`` sequence stride and V remains a view into the
    three-wide QKV projection.
    """
    _ensure_loaded()
    if not hasattr(torch.ops.cute_fmha, "sdp_minimax_h3_vae_d64"):
        raise RuntimeError(
            "CUTE MiniMax H3 VideoVAE D64 kernel is unavailable "
            "in this sidecar"
        )
    return torch.ops.cute_fmha.sdp_minimax_h3_vae_d64(q, k, v)


def supports_d120_bhld() -> bool:
    """Whether this target sidecar exports the workflow-tuned D120 kernel."""
    try:
        _ensure_loaded()
        return hasattr(torch.ops.cute_fmha, "sdp_bhld_d120")
    except Exception:
        return False


def sdp_bhld_d120(
    q: torch.Tensor, k: torch.Tensor, v: torch.Tensor
) -> torch.Tensor:
    """Fused self-attention for validated dense ``[B,H,L,120]`` inputs."""
    _ensure_loaded()
    if not hasattr(torch.ops.cute_fmha, "sdp_bhld_d120"):
        raise RuntimeError("CUTE D120 BHLD kernel is unavailable in this sidecar")
    return torch.ops.cute_fmha.sdp_bhld_d120(q, k, v)


__all__ = [
    "sdp",
    "sdp_wan22_cross",
    "supports_wan22_cross",
    "sdp_bhld_d128",
    "supports_d128_bhld",
    "sdp_minimax_h3_vae_d64",
    "supports_minimax_h3_vae_d64",
    "sdp_bhld_d120",
    "supports_d120_bhld",
    "sol_attn",
    "supports_sol_attn",
    "is_available",
]
