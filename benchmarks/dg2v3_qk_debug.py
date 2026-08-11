"""DG2 v3 QK layout forensics.

Runs the sidecar with DG2V3_ATTN_QK_ONLY + DG2V3_DUMP_QK and reconstructs the
hardware DPAS operand layout by brute-force comparison against the dumped
q/K/partial-C data.

Usage:
    set DG2V3_ATTN_QK_ONLY=1 DG2V3_DUMP_QK=1   (done by this script)
    python dg2v3_qk_debug.py
"""

import os
import struct

import torch

import omni_xpu_kernel.sdp as sdp


SCALE = 1.0 / (128.0 ** 0.5)


def load_dump(path):
    data = open(path, "rb").read()
    return struct.unpack("<8192f", data)


def main():
    inputs = torch.load(r"C:\Temp\dg2v3_input.pt", map_location="cpu", weights_only=False)
    q = inputs["q"].to("xpu")  # [1, 16, 1, 128] = [B, L, H, D]
    k = inputs["k"].to("xpu")
    v = inputs["v"].to("xpu")

    qcpu = inputs["q"].float()[0, :, 0, :]  # [16, 128]
    kcpu = inputs["k"].float()[0, :, 0, :]  # [16, 128]
    torch_scores = qcpu @ kcpu.t() * SCALE  # [16, 16]

    os.environ["DG2V3_ATTN_QK_ONLY"] = "1"
    os.environ["DG2V3_DUMP_QK"] = "1"
    out = sdp.sdp(q, k, v)
    torch.xpu.synchronize()

    dbg = load_dump(r"C:\Temp\dg2v3_qk.bin")

    # ---- region 1: q rows 0..3 (4 x 128 floats) ----
    qdump = [list(dbg[0 + r * 128 : 0 + (r + 1) * 128]) for r in range(4)]
    for r in range(4):
        ref = qcpu[r].tolist()
        err = max(abs(a - b) for a, b in zip(qdump[r], ref))
        print(f"q dump row{r} vs torch: max_abs={err:.6f}")
    for r in range(4):
        best = min(range(16), key=lambda rr: max(abs(a - b) for a, b in zip(qdump[r], qcpu[rr].tolist())))
        besterr = max(abs(a - b) for a, b in zip(qdump[r], qcpu[best].tolist()))
        print(f"q dump row{r} best matches torch row {best} (max_abs={besterr:.6f})")

    # ---- region 2: K blocks (g=0..1, c=0..7), raw VNNI fp16 order ----
    # block (g,c): kf[2*(dp*8+r)+h] = K[g*8+r][c*16 + 2*dp + h]
    kdump = [[0.0] * 128 for _ in range(16)]
    for g in range(2):
        for c in range(8):
            base = 512 + (g * 8 + c) * 128
            kf = dbg[base : base + 128]
            for dp in range(8):
                for r in range(8):
                    for h in range(2):
                        kdump[g * 8 + r][c * 16 + 2 * dp + h] = kf[2 * (dp * 8 + r) + h]
    errs = []
    for row in range(16):
        ref = kcpu[row].tolist()
        errs.append(max(abs(a - b) for a, b in zip(kdump[row], ref)))
    print("K dump max_abs per row:", [f"{e:.6f}" for e in errs])

    # ---- region 3: SC scores (4 rows x BN kv) ----
    sc = [[0.0] * 128 for _ in range(4)]
    for r in range(4):
        base = 4608 + r * 128
        sc[r] = list(dbg[base : base + 128])

    # Full-score check against SC region for the best hypothesis.
    print("\nSC dump (r0..3, kv0..15):")
    for r in range(4):
        print(f"  qrow{r}: " + " ".join(f"{x:+.4f}" for x in sc[r]))
    print("torch q0 scores:      " + " ".join(f"{x:+.4f}" for x in torch_scores[0]))
    print("torch q1 scores:      " + " ".join(f"{x:+.4f}" for x in torch_scores[1]))
    print("torch q2 scores:      " + " ".join(f"{x:+.4f}" for x in torch_scores[2]))
    print("torch q3 scores:      " + " ".join(f"{x:+.4f}" for x in torch_scores[3]))


if __name__ == "__main__":
    main()
