"""Convert Qwen Image 2.1 BF16 diffusion model to Comfy Kitchen AsymW4A8 format."""
import gc
import json
import os
import sys
import time
from pathlib import Path

import torch
from safetensors import safe_open
from safetensors.torch import save_file
from comfy_kitchen.tensor.w4a8_int8 import quantize_w4a8_int8_weight

SRC_PATH = r"C:\Users\Administrator\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\qwen_image_2.1_bf16.safetensors"
DST_PATH = r"C:\Users\Administrator\ComfyUI_windows_portable\ComfyUI\models\diffusion_models\qwen_image_2.1_w4a8.safetensors"

COMFY_QUANT_BYTES = json.dumps({
    "format": "asym_w4a8_int8",
    "group_size": 16,
    "convrot": True,
    "convrot_groupsize": 256,
}).encode("utf-8")


def is_quantizable(key: str, shape: list[int]) -> bool:
    if len(shape) != 2:
        return False
    if not key.endswith(".weight"):
        return False
    # Preserve embedding and small IO projections in unquantized precision
    if any(ign in key for ign in ["img_in", "proj_out", "time_text_embed", "timestep_embedder"]):
        return False
    n, k = shape
    return k >= 256 and (k % 256 == 0) and (k % 16 == 0)


def main():
    t0 = time.time()
    print(f"[*] Starting W4A8 quantization of {SRC_PATH}")
    print(f"[*] Target output: {DST_PATH}")

    output_tensors = {}
    with safe_open(SRC_PATH, framework="pt", device="cpu") as f:
        keys = list(f.keys())
        total = len(keys)
        quant_count = 0
        pass_count = 0

        for idx, key in enumerate(keys, 1):
            slice_obj = f.get_slice(key)
            shape = slice_obj.get_shape()

            if is_quantizable(key, shape):
                tensor = f.get_tensor(key).float()
                # Run w4a8 grouped quantization with convrot
                q, s_rel, s_chan, cb, corr = quantize_w4a8_int8_weight(
                    tensor, group_size=16, convrot_groupsize=256,
                    symmetric=True, scale_dtype=torch.float8_e4m3fn,
                    codebook=True,
                )
                output_tensors[key] = q.contiguous()
                output_tensors[f"{key}_s_rel"] = s_rel.contiguous()
                output_tensors[f"{key}_s_channel"] = s_chan.contiguous()
                if corr is not None:
                    # In symmetric mode with codebook, the codebook tensor is returned as corr
                    output_tensors[f"{key}_codebook"] = corr.contiguous()
                elif cb is not None:
                    output_tensors[f"{key}_codebook"] = cb.contiguous()

                prefix = key[:-len(".weight")]
                output_tensors[f"{prefix}.comfy_quant"] = torch.tensor(
                    list(COMFY_QUANT_BYTES), dtype=torch.uint8
                )
                quant_count += 1
                if quant_count % 10 == 0 or quant_count == 1:
                    elapsed = time.time() - t0
                    print(f"  [{idx}/{total}] Quantized {key} {shape} -> packed {list(q.shape)} ({elapsed:.1f}s)")
            else:
                tensor = f.get_tensor(key)
                output_tensors[key] = tensor
                pass_count += 1

    print(f"[*] All layers processed: {quant_count} quantized to W4A8, {pass_count} preserved.")
    print(f"[*] Saving {len(output_tensors)} tensors to {DST_PATH}...")
    save_file(output_tensors, DST_PATH)
    file_size_gb = os.path.getsize(DST_PATH) / (1024 ** 3)
    elapsed = time.time() - t0
    print(f"[+] W4A8 quantization complete in {elapsed:.1f}s! Output size: {file_size_gb:.2f} GB")


if __name__ == "__main__":
    main()
