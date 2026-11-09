# omni_xpu_kernel 0.2.0b3 build 1 (torch 2.15 / DG2 A770 / oneAPI 2026.2)

A770 / DG2 maintenance release for **PyTorch 2.15.0+xpu** and **Intel oneAPI 2026.2 Toolkit** (Python 3.13, Windows 10/11 x64).
This release migrates the compilation toolchain and runtime bindings to Intel oneAPI 2026.2, incorporates upstream kernel optimizations from `intel/llm-scaler` PR #786 (packed W4A8 decode, shared Qwen INT8 preparation, d128 RMS-RoPE fast path), and validates end-to-end W4A8 diffusion model generation while retaining the canonical base version `0.2.0b3`.

## What changed

### 1. Intel oneAPI 2026.2 Toolkit Migration & Runtime Unification
- Fully migrated compiler and linkage toolchain to Intel oneAPI DPC++/C++ 2026.2.0 (Build 20261001).
- Unified Windows SYCL runtime packages (`intel-sycl-rt`, `dpcpp-cpp-rt`, `intel-cmplr-lib-ur` 2026.2.0) to eliminate symbol discrepancies and `0xc0000139` load failures.
- Updated Windows dynamic loader path resolution to automatically support `ur_loader_shared.dll` and oneAPI 2026.2 / TBB runtime search paths.

### 2. Upstream Kernel Optimizations (intel/llm-scaler PR #786)
- **Packed W4A8 Decode**: Added `csrc/kitchen_w4a8.cpp` and `int8.w4a8_linear_prequantized`, enabling direct in-kernel nibble unpacking and scaled INT8 GEMM for decode / low-row inference (1-8 rows) without materializing full INT8 weights.
- **Butterfly Stride & Quantizer Refinement**: Replaced stride division with bit-shift calculations and unified per-work-item reciprocal multiplication in `csrc/kitchen_gated_quant.cpp`.
- **Qwen d128 Fast RMS-RoPE**: Integrated specialized indexed RMS-RoPE fast path for head dimension 128 in `csrc/kitchen_rms_rope_sycl.cpp`.
- **Retained Architecture Safeguards**: Kept DG2-specific INT8 residual acceleration path in `csrc/onednn_int8.cpp`.

### 3. W4A8 Diffusion Model Real-World Verification
- Successfully converted full `qwen_image_2.1_bf16.safetensors` (18.08 GB) into packed `qwen_image_2.1_w4a8.safetensors` (3.76 GB, **4.8x compression ratio** / 79% VRAM footprint reduction).
- Added safe XPU embedding dequantization route (`_dequant_int4_grouped_to_int8_xpu`) to prevent PyTorch 2.15 advanced indexing driver faults on DG2.
- Verified end-to-end 25-step diffusion image generation on Intel Arc A770 at steady **2.98 s/step**.

## Measured on A770 (ComfyUI 0.44.2 + PyTorch 2.15.0+xpu + oneAPI 2026.2)

Intel Arc A770 16 GB, PyTorch 2.15.0+xpu, ComfyUI v0.44.2, ComfyUI-OmniXPU, comfy-aimdo 0.6.0:

| Workflow | Spec | Steady step | Total time | Status |
|---|---|---|---|---|
| Z-Image Turbo | 1024², 8 steps | **0.85 s** | 10.09 s (hot) | **PASSED** |
| Krea2 Turbo | 1024², 8 steps | **1.80 s** | 19.38 s (hot) | **PASSED** |
| Krea2 Turbo + LoRA | 1024², 8 steps, strength 0.8 | **2.33 s** | 25.77 s (hot) | **PASSED** |
| Qwen-2.1 T2I | 1024², 25 steps | **0.828 s** | 24.85 s (hot) | **PASSED** |
| Qwen-2.1 ImageEdit | 1024², 25 steps, Dual-Ref | **1.52 s** | 44.49 s (hot) | **PASSED** |
| MiniMax H3 | 864×480, 56 frames, 20 steps | **9.65 s** | 270.61 s (cold) | **PASSED** |
| MiniMax H3 + LoRA | 864×480, 124 frames, 8 steps | **22.18 s** | 244.66 s (cold) | **PASSED** |
| Qwen-2.1 W4A8 (New) | 1024², 25 steps, 3.76 GB model | **2.98 s** | 101.12 s (cold) | **PASSED** |

## Build identity

- **Target**: DG2 (Intel Arc A770)
- **Version**: `0.2.0b3+torch215.dg2.1`
- **Platform**: Windows 10/11, win_amd64
- **Toolchain**: Intel oneAPI DPC++/C++ Compiler 2026.2.0 + MSVC 19.44 + PyTorch 2.15.0+xpu
- **Wheel**: `omni_xpu_kernel-0.2.0b3+torch215.dg2.1-cp313-cp313-win_amd64.whl`
- **SHA256**: `8665F1DB1E9A5A82C1E2C6FF950FF8CA2A40C960592642F97EEA866E2EEFC2CD`

## Install

```powershell
pip install --force-reinstall --no-deps omni_xpu_kernel-0.2.0b3+torch215.dg2.1-cp313-cp313-win_amd64.whl
```
