# omni_xpu_kernel (A770/DG2 fork)

High-performance Intel XPU kernels for PyTorch, with A770/DG2 Windows support
added on top of the upstream `omni_xpu_kernel` maintained inside
[intel/llm-scaler](https://github.com/intel/llm-scaler)
(`omni/omni_xpu_kernel`).

This repository is the standalone fork used by
[ComfyUI-OmniXPU](https://github.com/Blackwood416/ComfyUI-OmniXPU). The official
upstream README is preserved in [`UPSTREAM_README.md`](UPSTREAM_README.md);
this file only describes the fork and its differences.

## Documentation layout

| File | Content |
|---|---|
| `README.md` | Fork status, A770/DG2 differences, runtime requirements |
| `UPSTREAM_README.md` | Official upstream README, preserved verbatim |
| `WHL_BUILD_INSTALL.md` | Windows wheel build matrix, installation, acceptance |
| `CHANGELOG.md` | Versioned local changes and measured results |

## Fork changes

### DG2 SDP attention sidecar

Windows A770 wheels package `lgrf_uni/lgrf_sdp*.pyd` with DG2-native ESIMD
kernels:

- D128 v4.1 DPAS (`flash.attn.b.mha.dg2.dpas4.h`) is the default
  large-sequence route. On A770 it reaches or beats PyTorch SDPA for the
  MiniMax H3 workload shapes.
- D64 v4.1 DPAS port (`flash.attn.b.mha.dg2.dpas4.d64.h`) is correct and
  stable, but still 0.57-0.92x PyTorch SDPA on the measured shapes, so the
  ComfyUI adapter keeps FP16/D64 on the torch fallback.

Inputs are contiguous `[B, L, H, D]`, `B=1`, `D` in {64, 128}, FP16 or BF16.
Enable with `OMNI_ATTN_BACKEND=esimd`; the adapter retains PyTorch SDPA for
every contract the kernel does not win.

Measured on A770 (driver 32.0.101.8860, oneAPI 2026.1, D=128, FP16, H=32,
40-sample interleaved wall medians): L512 0.54 vs 0.60 ms, L2048 2.75 vs
3.29 ms, L4096 8.8 vs 12.6 ms, L8192 34.7 vs 41.4 ms, 1024x4096 2.91 vs
3.20 ms. H3 `(1,20683,56,128)` BF16: 405-410 vs 431-451 ms.

### DG2 fused ConvRot quantization

`int8_convrot_quant_dg2.cpp` (SLM) and `int8_convrot_quant_esimd.cpp`
(register butterfly) fuse the cached Hadamard matmul into rowwise INT8
quantization. The register variant is the production path: `20685x14336`
4.3 ms vs 7.3 ms for matmul+quantize (1.7x), with no DEVICE_LOST in the
pressure runs. Routed through `OMNIXPU_DG2_CONVROT_FUSED=1` (default on).

### RMS-RoPE bridge support

`rotary.rms_kitchen_rope_split_half_` (and the functional variant) provide
the fused RMSNorm + partial split-half RoPE used by MiniMax H3 Q/K. On A770
the Kitchen eager path is measured 4-11x slower than this fused kernel.

### INT8 fast paths and packaging

- `mixed_precision.Linear` INT8 fast path (`omni_dg2_compat_fast`) and cached
  qdata/scale copies for offloaded TensorWise INT8 weights.
- Torch 2.13 support (`torch213.dg2` wheels), oneAPI 2026.1, `dg2` target with
  `a770`/`arc-a770` aliases, and a private wheel build counter
  (`OMNI_XPU_BUILD_NUMBER`).

## Windows A770/DG2 runtime requirements

The Windows wheels import `sycl9.dll`, `dnnl.dll`, `torch_xpu.dll`, and
`c10_xpu.dll`. They are built and validated with:

- oneAPI DPC++/C++ Compiler 2026.1.0;
- Intel Arc driver 32.0.101.8860;
- PyTorch 2.13.0+xpu from the official XPU index.

At runtime, install the torch XPU pip packages; they pull the complete SYCL /
Unified Runtime (`intel-sycl-rt`, `intel-cmplr-lib-ur`, `intel-cmplr-lib-rt`,
...) including the Level Zero PI backend (`pi_level_zero.dll`). A partial
oneAPI installation that lacks `pi_level_zero.dll` cannot load `_C`; see
[issue #1](https://github.com/Blackwood416/omni-xpu-kernel/issues/1).

The ComfyUI portable environment already satisfies these requirements through
its torch installation; no separate oneAPI install is needed there.

## Status vs upstream

- Upstream A770 builds are core-only (no SDP sidecar; callers retain PyTorch
  SDPA). This fork adds the DG2 sidecar and the A770-tuned variants above.
- Dispatch boundaries here are A770 measurements only. Other GPUs must be
  re-measured before adopting any fork-specific route.

See `CHANGELOG.md` for the full change history and negative results.
