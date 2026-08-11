"""Minimal torch SDPA loop for VTune GPU comparison (no sidecar)."""

import torch


def main():
    torch.manual_seed(0)
    q = torch.randn(1, 32, 8192, 128, dtype=torch.float16, device="xpu") * 0.5
    k = torch.randn(1, 32, 8192, 128, dtype=torch.float16, device="xpu") * 0.5
    v = torch.randn(1, 32, 8192, 128, dtype=torch.float16, device="xpu") * 0.5
    for _ in range(12):
        torch.nn.functional.scaled_dot_product_attention(q, k, v)
    torch.xpu.synchronize()


if __name__ == "__main__":
    main()
