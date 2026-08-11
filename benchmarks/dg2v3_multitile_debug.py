"""DG2 v3 multi-tile state forensics (L=1, kv=65)."""

import os
import struct

import torch

import omni_xpu_kernel.sdp as sdp


SCALE = 1.0 / (128.0 ** 0.5)


def load_dump(path):
    data = open(path, "rb").read()
    return struct.unpack("<8192f", data)


def main():
    torch.manual_seed(72)
    q = torch.randn(1, 1, 1, 128, dtype=torch.float16) * 0.5
    k = torch.randn(1, 65, 1, 128, dtype=torch.float16) * 0.5
    v = torch.randn(1, 65, 1, 128, dtype=torch.float16) * 0.5

    os.environ["DG2V3_ATTN_QK_ONLY"] = "1"
    os.environ["DG2V3_DUMP_QK"] = "1"
    sdp.sdp(q.to("xpu"), k.to("xpu"), v.to("xpu"))
    torch.xpu.synchronize()
    dbg = load_dump(r"C:\Temp\dg2v3_qk.bin")

    qf = q.float()[0, 0, 0, :]
    kf = k.float()[0, :, 0, :]  # [65, 128]
    vf = v.float()[0, :, 0, :]

    # SC per tile
    for t in range(2):
        base = 4608 + t * 64
        sc = dbg[base : base + 64]
        rows = kf[t * 64 : t * 64 + 64]
        torch_scores = (qf @ rows.t()) * SCALE
        err = max(abs(a - b) for a, b in zip(sc[:16], torch_scores[:16]))
        print(f"tile {t}: SC err (first 16)={err:.6f}")
        print("  sc0  :", " ".join(f"{x:+.4f}" for x in sc[:16]))
        print("  torch:", " ".join(f"{x:+.4f}" for x in torch_scores[:16]))
        print("  sc16 :", " ".join(f"{x:+.4f}" for x in sc[16:32]))
        print("  torch:", " ".join(f"{x:+.4f}" for x in torch_scores[16:32]))

    # expected online state
    all_scores = (qf @ kf.t()) * SCALE  # [65]
    m_ref = -3.402823466e38
    l_ref = 0.0
    acc_ref = torch.zeros(128)
    m_hist = []
    l_hist = []
    for t in range(2):
        seg = all_scores[t * 64 : t * 64 + 64]
        m_tile = seg.max().item()
        p = torch.exp2((seg - m_tile).double() * 1.4426950408889634)
        l_tile = p.sum().item()
        rescale = torch.exp2(torch.tensor((m_ref - m_tile) * 1.4426950408889634).double()).item()
        acc_ref = acc_ref * rescale
        l_ref = l_ref * rescale + l_tile
        m_ref = m_tile
        m_hist.append(m_tile)
        l_hist.append(l_tile)
    gt = torch.softmax(all_scores, dim=-1).float() @ vf

    print("\nkernel final m:", [f"{dbg[5500+r]:.6f}" for r in range(4)])
    print("expected m    :", [f"{m_ref:.6f}"] * 4)
    print("kernel final l:", [f"{dbg[5504+r]:.6f}" for r in range(4)])
    print("expected l    :", [f"{l_ref:.6f}"] * 4)
    print("kernel acc row0[0:8]:", [f"{dbg[5508+i]:.6f}" for i in range(8)])
    print("expected acc row0[0:8]:", [f"{x:.6f}" for x in acc_ref.tolist()[:8]])
    print("gt[0:8]:", [f"{x:.6f}" for x in gt.tolist()[:8]])



if __name__ == "__main__":
    main()
