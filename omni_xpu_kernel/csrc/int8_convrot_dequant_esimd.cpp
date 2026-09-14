#include <torch/extension.h>
#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "utils.h"

using namespace sycl::ext::intel::esimd;

namespace omni_xpu {
namespace int8_ops {
namespace {

#ifndef OMNI_CONVROT_DEQUANT_WG_SIZE
#if defined(OMNI_XPU_ARCH_PTL_H)
#define OMNI_CONVROT_DEQUANT_WG_SIZE 1
#elif defined(OMNI_XPU_ARCH_BMG)
#define OMNI_CONVROT_DEQUANT_WG_SIZE 1
#elif defined(OMNI_XPU_ARCH_DG2)
#define OMNI_CONVROT_DEQUANT_WG_SIZE 1
#else
#error "Define a supported OMNI_XPU_ARCH target"
#endif
#endif

template<int Stride, int GroupSize>
inline void radix4_hadamard_stage(simd<float, GroupSize>& values) {
#pragma unroll
    for (int base = 0; base < GroupSize; base += 4 * Stride) {
        simd<float, Stride> a =
            values.template select<Stride, 1>(base);
        simd<float, Stride> b =
            values.template select<Stride, 1>(base + Stride);
        simd<float, Stride> c =
            values.template select<Stride, 1>(base + 2 * Stride);
        simd<float, Stride> d =
            values.template select<Stride, 1>(base + 3 * Stride);
        values.template select<Stride, 1>(base) = a + b + c - d;
        values.template select<Stride, 1>(base + Stride) = a + b - c + d;
        values.template select<Stride, 1>(base + 2 * Stride) = a - b + c + d;
        values.template select<Stride, 1>(base + 3 * Stride) = -a + b + c + d;
    }
    if constexpr (Stride * 4 < GroupSize) {
        radix4_hadamard_stage<Stride * 4, GroupSize>(values);
    }
}

template<int GroupSize, typename OutputT>
void dequantize_convrot_kernel(
    const int8_t* __restrict__ input,
    const float* __restrict__ scales,
    OutputT* __restrict__ output,
    int64_t rows,
    int64_t groups,
    const at::Device& device) {
    constexpr int WorkGroupSize = OMNI_CONVROT_DEQUANT_WG_SIZE;
    const int64_t work_items = rows * groups;
    const int64_t padded =
        (work_items + WorkGroupSize - 1) / WorkGroupSize * WorkGroupSize;
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(padded), sycl::range<1>(WorkGroupSize)),
            [=](sycl::nd_item<1> item) SYCL_ESIMD_KERNEL {
                const int64_t work_item = item.get_global_id(0);
                if (work_item >= work_items) return;
                const int64_t row = work_item / groups;
                const int64_t group = work_item % groups;
                const int64_t first =
                    row * groups * GroupSize + group * GroupSize;
                simd<int8_t, GroupSize> quantized =
                    block_load<int8_t, GroupSize>(input + first);
                simd<float, GroupSize> values = quantized;
                values *= scales[row];
                radix4_hadamard_stage<1, GroupSize>(values);
                values *= 1.0f / sycl::sqrt(static_cast<float>(GroupSize));
                simd<OutputT, GroupSize> converted = values;
                block_store<OutputT, GroupSize>(output + first, converted);
            });
    };
    utils::submit_kernel(cgf, device, "int8_convrot_dequant_fused");
}

template<typename OutputT>
void dispatch_dequantize(
    const torch::Tensor& input, const torch::Tensor& scale,
    torch::Tensor& output, int64_t group_size) {
    const int64_t groups = input.size(1) / group_size;
    auto* output_ptr = reinterpret_cast<OutputT*>(output.data_ptr());
    if (group_size == 64) {
        dequantize_convrot_kernel<64, OutputT>(
            input.data_ptr<int8_t>(), scale.data_ptr<float>(), output_ptr,
            input.size(0), groups, input.device());
    } else {
        dequantize_convrot_kernel<256, OutputT>(
            input.data_ptr<int8_t>(), scale.data_ptr<float>(), output_ptr,
            input.size(0), groups, input.device());
    }
}

}  // namespace

torch::Tensor dequantize_int8_convrot_fused_dtype(
    torch::Tensor input,
    torch::Tensor scale,
    int64_t group_size,
    int64_t output_dtype_code) {
    TORCH_CHECK(input.device().is_xpu(), "input must be on XPU");
    TORCH_CHECK(input.scalar_type() == torch::kInt8, "input must be int8");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(group_size == 64 || group_size == 256,
                "fused ConvRot dequantization requires group size 64 or 256");
    TORCH_CHECK(input.size(1) % group_size == 0,
                "input features must be divisible by group size");
    TORCH_CHECK(scale.numel() == input.size(0), "scale must be rowwise");
    TORCH_CHECK(output_dtype_code >= 0 && output_dtype_code <= 2,
                "output dtype code must be 0 (fp32), 1 (fp16), or 2 (bf16)");
    input = input.contiguous();
    auto scale_f = scale.to(input.device(), torch::kFloat).contiguous();
    const auto output_dtype = output_dtype_code == 0 ? torch::kFloat
        : output_dtype_code == 1 ? torch::kHalf : torch::kBFloat16;
    auto output = torch::empty(input.sizes(), input.options().dtype(output_dtype));
    if (input.numel() == 0) return output;
    if (output_dtype_code == 0) {
        dispatch_dequantize<float>(input, scale_f, output, group_size);
    } else if (output_dtype_code == 1) {
        dispatch_dequantize<sycl::half>(input, scale_f, output, group_size);
    } else {
        dispatch_dequantize<sycl::ext::oneapi::bfloat16>(input, scale_f, output, group_size);
    }
    return output;
}

torch::Tensor dequantize_int8_convrot_fused(
    torch::Tensor input, torch::Tensor scale, int64_t group_size) {
    return dequantize_int8_convrot_fused_dtype(input, scale, group_size, 0);
}

}  // namespace int8_ops
}  // namespace omni_xpu
