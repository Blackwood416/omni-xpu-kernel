"""A770/DG2 SDP regression gate.

The Xe2 lgrf SDP kernel family device-lost on A770 (driver 32.0.101.8860).
Root cause found with the standalone harness plus Windows LiveKernelEvent
logs: ESIMD work-group size 512 triggers a GPU TDR (event 141) even for a
trivial kernel, while WG=32 is stable. The DG2-native kernel in
`lgrf_uni/single_kernels/flash.attn.b.mha.dg2.h` therefore uses WG=32 and
passes this gate.

Keep this file as the regression gate for any future A770 SDP kernel change:
it must print finite output and exit 0. It is also the minimal repro for the
historical device-lost bug.

Usage (after AOT-building the DG2 sidecar next to the package):

    python repro_a770_sdp_device_lost.py

Expected result on the recorded stack: UR_RESULT_ERROR_DEVICE_LOST before any
output is produced. Keep this file as the regression gate for the next
DG2-native SDP kernel; it must pass (finite output) before any A770 dispatch
rule is enabled.
"""

import torch

import omni_xpu_kernel
import omni_xpu_kernel.sdp as sdp


def main() -> None:
    print("target:", getattr(omni_xpu_kernel, "__xpu_target__", None))
    print("sdp available:", sdp.is_available())
    if not sdp.is_available():
        raise SystemExit("DG2 sidecar not present; nothing to reproduce.")

    def check(q, k, v, label):
        out = sdp.sdp(q, k, v)
        torch.xpu.synchronize()
        if not bool(torch.isfinite(out).all()):
            raise SystemExit(f"non-finite output ({label})")
        ref = torch.nn.functional.scaled_dot_product_attention(
            q.permute(0, 2, 1, 3).contiguous(),
            k.permute(0, 2, 1, 3).contiguous(),
            v.permute(0, 2, 1, 3).contiguous(),
        ).permute(0, 2, 1, 3).contiguous()
        torch.xpu.synchronize()
        err = (out.float() - ref.float()).abs().max().item()
        if err > 0.02:
            raise SystemExit(f"correctness regression ({label}): max_abs={err:.5f}")
        print(f"{label}: max_abs={err:.5f}")

    torch.manual_seed(0)
    q = torch.randn(1, 16, 8, 128, device="xpu", dtype=torch.float16) * 0.5
    k = torch.randn_like(q)
    v = torch.randn_like(q)
    check(q, k, v, "basic 16x16")

    # Multi-tile + padding coverage (kvZero global-flag path).
    for qlen, kvlen, h in ((65, 130, 1), (129, 300, 8), (300, 300, 12)):
        torch.manual_seed(7 + qlen)
        q = torch.randn(1, qlen, h, 128, device="xpu", dtype=torch.float16) * 0.5
        k = torch.randn(1, kvlen, h, 128, device="xpu", dtype=torch.float16) * 0.5
        v = torch.randn(1, kvlen, h, 128, device="xpu", dtype=torch.float16) * 0.5
        check(q, k, v, f"L={qlen} kv={kvlen} H={h}")

    torch.manual_seed(99)
    q = torch.randn(1, 256, 8, 128, device="xpu", dtype=torch.bfloat16) * 0.5
    k = torch.randn(1, 300, 8, 128, device="xpu", dtype=torch.bfloat16) * 0.5
    v = torch.randn(1, 300, 8, 128, device="xpu", dtype=torch.bfloat16) * 0.5
    check(q, k, v, "bf16 multi-tile")

    print("finite output; device-lost regression fixed")


if __name__ == "__main__":
    main()
