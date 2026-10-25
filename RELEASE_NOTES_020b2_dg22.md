# omni_xpu_kernel 0.2.0b2 build 2 (torch 2.14 / DG2 A770)

A770 / DG2 build for **PyTorch 2.14.0+xpu** (Python 3.13, oneAPI 2026.1, Windows 10/11 x64).
This release builds upon `0.2.0b2+torch214.dg2.1` and synchronizes upstream `intel/llm-scaler` enhancements (#746, #747, #749, #767, #768) including native SiTU activation quantization, Kitchen decode operators, persistent Hadamard cache allocation context, and robust tensor shape/device guards.

## What changed

### 1. Native SiTU & Gated INT8 Quantization
- **Fused SiTU / SwiGLU Gated Activation**: Added `csrc/kitchen_gated_quant.cpp` which executes fused gated activation (SiTU or SwiGLU), optional online ConvRot rotation, and rowwise INT8 quantization in a single kernel dispatch.
- Directly accelerates MOSS-Image INT8 inference and ComfyUI DiT pipelines without temporary activation re-allocations.

### 2. Kitchen Decode & Deferred-State Operators
- Added native Kitchen decode implementations: `decode_attention`, `decode_step_merge`, `int8_decode_update`, `int8_decode`, `deltanet_conv_deferred`, and `gated_delta_decode_deferred` (`csrc/kitchen_decode.cpp`, `csrc/kitchen_decode.h`, `csrc/kitchen_deferred.cpp`, `csrc/kitchen_int8_decode.cpp`).
- Fully bound and registered under `omni_xpu_kernel.kitchen` with PyTorch 2.14 compile meta ops.

### 3. Allocation Context & Numerical Robustness Guards
- **ConvRot Cache Management**: Exposed `prepare_convrot_hadamard`, `clear_convrot_hadamard_cache`, and `set_allocation_context_factory` in `omni_xpu_kernel.int8` to prevent persistent matrix allocations from leaking into temporary graph memory contexts.
- **Scaleback Bias Guard**: Added explicit dimension and XPU device equality verification for fused scaleback bias in `csrc/int8_scaleback_esimd.cpp`.
- **oneDNN INT4 Grouping Guard**: Applied safe grouping division and positive/divisible group-size constraints before oneDNN primitive construction in `csrc/onednn_int4_gemm.cpp`.

## Measured on A770 (ComfyUI 0.42.5 + Driver 9150)

Intel Arc A770 16 GB, driver 32.0.101.9150, oneAPI 2026.1, torch 2.14.0+xpu:

| Workflow | Spec | Steady step | Rounds |
|---|---|---|---|
| Z-Image Turbo | 1024², 8 steps | 0.88 s | 2/2 |
| Krea2 Turbo | 1024², 8 steps | 1.91 s | 2/2 |
| Krea2 Turbo + LoRA | 1024², 8 steps, strength 0.8 | 2.07 s | 2/2 |
| Qwen-2.1 T2I | 1024², 25 steps | 1.27 s | 2/2 |
| Qwen-2.1 ImageEdit | 1024², 25 steps, Dual-Ref | 5.00 s | 2/2 |
| MOSS-Image | INT8 ConvRot DiT, 50 steps | 4.12 s | Verified |
| MiniMax H3 + LoRA | 864×480, 124 frames, 8 steps | 26–28 s | 8/8 steps |

## Build identity

- **Target**: DG2 (Intel Arc A770)
- **Version**: `0.2.0b2+torch214.dg2.2`
- **Platform**: Windows 10/11, win_amd64
- **Toolchain**: Intel oneAPI DPC++/C++ Compiler 2026.1.0 + MSVC 19.44
- **Wheel**: `omni_xpu_kernel-0.2.0b2+torch214.dg2.2-cp313-cp313-win_amd64.whl`
- **SHA256**: `50ED97C494850C83C09D860186275C294A98D6F9A8C90E83F6E1ED9344A61646`

## Install

```powershell
pip install --force-reinstall --no-deps omni_xpu_kernel-0.2.0b2+torch214.dg2.2-cp313-cp313-win_amd64.whl
```
