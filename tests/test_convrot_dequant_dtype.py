"""CPU-reference contract for the dtype-aware inverse ConvRot path."""
import pytest
import torch

from omni_xpu_kernel import int8

pytestmark = pytest.mark.skipif(not torch.xpu.is_available(), reason="XPU required")


def reference(q, scale, group_size, dtype):
    # FP32 dequantization precedes the inverse rotation in Kitchen.
    h4 = torch.tensor([[1, 1, 1, -1], [1, 1, -1, 1],
                       [1, -1, 1, 1], [-1, 1, 1, 1]], dtype=torch.float64)
    h = h4
    while h.shape[0] < group_size:
        h = torch.kron(h, h4)
    h = h / group_size**0.5
    values = (q.float() * scale.float()).double()
    return (values.reshape(-1, group_size) @ h).reshape(q.shape).to(dtype)


@pytest.mark.parametrize("rows,cols,group", [
    (1, 64, 64), (3, 256, 256), (31, 512, 256),
    (257, 5376, 256), (33, 14336, 256), (17, 7168, 256),
    (3, 128, 64), (3, 128, 16),
])
@pytest.mark.parametrize("dtype", [torch.float32, torch.float16, torch.bfloat16])
def test_inverse_rotation_matches_cpu(rows, cols, group, dtype):
    generator = torch.Generator().manual_seed(90673001)
    q_cpu = torch.randint(-128, 128, (rows, cols), dtype=torch.int8, generator=generator)
    scale_cpu = torch.rand(rows, 1, generator=generator) * 0.01
    expected = reference(q_cpu, scale_cpu, group, dtype)
    actual = int8.dequantize_int8_convrot_weight_dtype(
        q_cpu.to("xpu"), scale_cpu.to("xpu"), group, dtype)
    torch.xpu.current_stream().synchronize()
    actual_cpu = actual.cpu()
    tolerance = {torch.float32: (1e-5, 2e-6), torch.float16: (1e-3, 2e-5),
                 torch.bfloat16: (0.008, 2e-5)}[dtype]
    torch.testing.assert_close(actual_cpu, expected, rtol=tolerance[0], atol=tolerance[1])


@pytest.mark.parametrize("offset", [0, 1, 17])
def test_storage_offset_and_noncontiguous_views(offset):
    q_storage = torch.randint(-128, 128, (offset + 3*512,), dtype=torch.int8)
    q_cpu = q_storage[offset:].reshape(3, 512)[:, ::2]
    scales = torch.tensor([[0.0], [0.01], [0.025]])
    q = q_storage.to("xpu")[offset:].reshape(3, 512)[:, ::2]
    actual = int8.dequantize_int8_convrot_weight_dtype(q, scales.to("xpu"), 256)
    torch.xpu.current_stream().synchronize()
    torch.testing.assert_close(actual.cpu(), reference(q_cpu, scales, 256, torch.bfloat16),
                               rtol=0.008, atol=2e-5)


@pytest.mark.parametrize("group", [0, 3, 128, 1024])
def test_rejects_invalid_group_geometry(group):
    q = torch.ones(3, 256, dtype=torch.int8, device="xpu")
    scale = torch.ones(3, 1, device="xpu")
    with pytest.raises((RuntimeError, ValueError)):
        int8.dequantize_int8_convrot_weight_dtype(q, scale, group)


def test_invalid_output_dtype():
    q = torch.ones(1, 256, dtype=torch.int8, device="xpu")
    with pytest.raises(ValueError, match="ConvRot output"):
        int8.dequantize_int8_convrot_weight_dtype(q, torch.ones(1, 1, device="xpu"), 256, torch.int8)


def test_empty_rows():
    result = int8.dequantize_int8_convrot_weight_dtype(
        torch.empty(0, 256, dtype=torch.int8, device="xpu"),
        torch.empty(0, 1, device="xpu"), 256)
    assert result.shape == (0, 256) and result.dtype == torch.bfloat16
