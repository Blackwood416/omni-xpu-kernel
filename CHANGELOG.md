# Changelog

Notable user-facing changes will be recorded here when a public
`omni_xpu_kernel` release is prepared.

## Unreleased

## 0.2.0b1+torch213.dg2.1 - 2026-08-16

- Add the A770/DG2 D64 DPAS attention port (attention v4.1, fp16 and bf16
  D64 kernels) and the DG2 fused ConvRot rotation+quantize path via ESIMD
  register butterfly, including stability and compiler-fold fixes.
- Add the A770 MiniMax H3 fused RMS-RoPE bridge kernels and the matching
  A/B benchmark and VLINT profiles.
- Add an optional private rebuild counter to the wheel version
  (`OMNI_XPU_BUILD_NUMBER=1` builds `0.2.0b1+torch213.dg2.1`); the public
  base version keeps following the upstream image version.

## 0.2.0b1 - 2026-08-10

- Add the focused ComfyUI v0.31 image with pinned XPU Kitchen and AIMDO
  integrations.
- Include the accepted BMG and PTL-H kernel routes accumulated for the
  preview milestone.

## 0.1.0b9.dev1 - 2026-08-05

- Vendor the validated oneDNN 3.9.1 runtime and redistribution notices in
  Windows wheels.
- Prefer the wheel-private `dnnl.dll` at runtime and retain system oneAPI only
  as a source-checkout and legacy-wheel fallback.
- Remove the ineffective Windows `onednn` Python package dependency while
  preserving the Linux dependency and runtime layout.
