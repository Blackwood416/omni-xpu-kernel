// DG2/A770 fused ConvRot activation rotation + rowwise INT8 quantization.
//
// The A770 H3 INT8 path currently rotates activations with a cached
// torch.matmul against the dense [G,G] Hadamard matrix. For K=14336 that
// path is launch- and GEMM-efficiency-bound (~165 GB/s effective on the
// 0.7MP workload). This implementation replaces the matmul with a radix-4
// fast-Hadamard butterfly and fuses rowwise quantization:
//
//   kernel A: per (row, group) work-group rotates the group in SLM and
//             writes the unnormalized group absmax to a [rows, groups]
//             scratch buffer;
//   kernel B: one subgroup per row reduces the group maxima into the row
//             scale (row_max / sqrt(G) / 127);
//   kernel C: per (row, group) work-group recomputes the rotation and
//             quantizes with the completed row scale.
//
// The normalization 1/sqrt(G) is folded into scale/quant_inv instead of
// materializing a rounded rotated value: q = round(raw * 127 / row_max_raw).
// This skips the bf16 rounding boundary of the composed matmul path (INT8
// differences of at most 1 are expected and covered by the test tolerance).
//
// Design constraints for A770 (see the A770 kernel skill):
//   - WG=256 (8 subgroups x 32 lanes), one group per subgroup;
//   - SLM = 8 * group_size floats (<= 8KB for G=256), high occupancy;
//   - no 2D LSC, no named barriers, no ESIMD DPAS in this path.
//
// Negative results recorded 2026-08-15 (A770, driver 32.0.101.8860):
//   1. row-SLM single kernel (WG=256, 57KB SLM, one row/WG): numerically
//      correct but only 0.39-0.78x the composed path due to occupancy.
//   2. early-returning invalid subgroups before item.barrier() produced
//      UR_RESULT_ERROR_DEVICE_LOST on K=7168/256. Never return before a
//      barrier; clamp invalid groups and mask stores.
//   3. an in-place bf16 "round" pass after the butterfly (reading and
//      rewriting every SLM element) caused the compiler to fold away both
//      passes: the kernel returned the raw input. The round pass is
//      therefore omitted and normalization is folded into quant_inv.
//   4. the shared helper-function variant showed the same folding; rotate
//      code is inlined into each kernel body.

#include <torch/extension.h>
#include <sycl/sycl.hpp>

#include <cmath>
#include <tuple>

#include "utils.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

namespace omni_xpu {
namespace int8_ops {
namespace {

constexpr int32_t DG2_CONVROT_SG = 32;
constexpr int32_t DG2_CONVROT_WG = 256;
constexpr int32_t DG2_CONVROT_SUBS = DG2_CONVROT_WG / DG2_CONVROT_SG;

template <typename InputT>
void convrot_rotate_max_kernel(
    const InputT* __restrict__ input,
    float* __restrict__ group_maxes,
    int64_t rows,
    int64_t groups,
    int64_t group_size,
    int64_t stages,
    const at::Device& device) {
    const int64_t groups_per_wg = DG2_CONVROT_SUBS;
    const int64_t wgs = (groups + groups_per_wg - 1) / groups_per_wg;
    const size_t global = static_cast<size_t>(rows) * wgs * DG2_CONVROT_WG;
    auto cgf = [&](sycl::handler& handle) {
        sycl::local_accessor<float, 1> slm(
            sycl::range<1>(groups_per_wg * group_size), handle);
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(global), sycl::range<1>(DG2_CONVROT_WG)),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(DG2_CONVROT_SG)]] {
                auto sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int sub = static_cast<int>(sg.get_group_linear_id());
                const int64_t row =
                    static_cast<int64_t>(item.get_group(0) / wgs);
                const int64_t wg_group =
                    static_cast<int64_t>(item.get_group(0) % wgs);
                const int64_t raw_group = wg_group * groups_per_wg + sub;
                // Clamp invalid groups instead of early-returning: every
                // work-item must reach every barrier.
                const int64_t group =
                    raw_group < groups ? raw_group : (groups - 1);
                const bool valid = raw_group < groups;
                const int64_t group_offset = group * group_size;
                const int64_t slm_offset = sub * group_size;
                const InputT* __restrict__ row_ptr =
                    input + row * groups * group_size;

                auto rotate_group = [&]() {
                    constexpr int VEC = 8;
                    const int64_t lane_start =
                        static_cast<int64_t>(lane) * VEC;
                    if (group_size % VEC == 0) {
                        using VecT = sycl::vec<InputT, VEC>;
                        for (int64_t k = lane_start; k < group_size;
                             k += DG2_CONVROT_SG * VEC) {
                            const VecT values =
                                *reinterpret_cast<const VecT*>(
                                    row_ptr + group_offset + k);
                            for (int i = 0; i < VEC; ++i) {
                                slm[slm_offset + k + i] =
                                    static_cast<float>(values[i]);
                            }
                        }
                    } else {
                        for (int64_t k = lane; k < group_size;
                             k += DG2_CONVROT_SG) {
                            slm[slm_offset + k] = static_cast<float>(
                                row_ptr[group_offset + k]);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // Radix-4 Hadamard butterfly in place. Base enumeration
                    // avoids a runtime `continue` (see negative result 3).
                    for (int64_t stage = 0; stage < stages; ++stage) {
                        const int64_t stride =
                            static_cast<int64_t>(1) << (2 * stage);
                        for (int64_t r = lane; r < stride; r += DG2_CONVROT_SG) {
                            for (int64_t b = r; b < group_size;
                                 b += 4 * stride) {
                                const float x0 = slm[slm_offset + b];
                                const float x1 =
                                    slm[slm_offset + b + stride];
                                const float x2 =
                                    slm[slm_offset + b + 2 * stride];
                                const float x3 =
                                    slm[slm_offset + b + 3 * stride];
                                slm[slm_offset + b] = x0 + x1 + x2 - x3;
                                slm[slm_offset + b + stride] =
                                    x0 + x1 - x2 + x3;
                                slm[slm_offset + b + 2 * stride] =
                                    x0 - x1 + x2 + x3;
                                slm[slm_offset + b + 3 * stride] =
                                    -x0 + x1 + x2 + x3;
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                };

                rotate_group();

                float local_max = 0.0f;
                for (int64_t k = lane; k < group_size;
                     k += DG2_CONVROT_SG) {
                    local_max = sycl::fmax(
                        local_max, sycl::fabs(slm[slm_offset + k]));
                }
                const float group_max = sycl::reduce_over_group(
                    sg, local_max, sycl::maximum<float>());
                if (lane == 0 && valid) {
                    group_maxes[row * groups + group] = group_max;
                }
            });
    };
    utils::submit_kernel(cgf, device, "int8_convrot_rotate_max_dg2");
}

template <typename InputT>
void convrot_row_reduce_kernel(
    const float* __restrict__ group_maxes,
    float* __restrict__ row_scales,
    int64_t rows,
    int64_t groups,
    float inv_sqrt_group,
    const at::Device& device) {
    // One subgroup (32 lanes) per row; no SLM, no atomics, no barrier.
    const size_t global = static_cast<size_t>(rows) * DG2_CONVROT_SG;
    auto cgf = [&](sycl::handler& handle) {
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(global), sycl::range<1>(DG2_CONVROT_SG)),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(DG2_CONVROT_SG)]] {
                auto sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int64_t row =
                    static_cast<int64_t>(item.get_group(0));
                if (row >= rows) return;

                float local_max = 0.0f;
                for (int64_t g = lane; g < groups; g += DG2_CONVROT_SG) {
                    local_max = sycl::fmax(
                        local_max, group_maxes[row * groups + g]);
                }
                const float row_max = sycl::reduce_over_group(
                    sg, local_max, sycl::maximum<float>());
                if (lane == 0) {
                    // scale = max(rotated)/127 = row_max_raw/sqrt(G)/127.
                    const float scale =
                        row_max > 0.0f
                            ? row_max * inv_sqrt_group / 127.0f
                            : 0.0f;
                    row_scales[row] = scale;
                }
            });
    };
    utils::submit_kernel(cgf, device, "int8_convrot_row_reduce_dg2");
}

template <typename InputT>
void convrot_quantize_kernel(
    const InputT* __restrict__ input,
    const float* __restrict__ row_scales,
    int8_t* __restrict__ output,
    int64_t rows,
    int64_t groups,
    int64_t group_size,
    int64_t stages,
    float inv_sqrt_group,
    const at::Device& device) {
    const int64_t groups_per_wg = DG2_CONVROT_SUBS;
    const int64_t wgs = (groups + groups_per_wg - 1) / groups_per_wg;
    const size_t global = static_cast<size_t>(rows) * wgs * DG2_CONVROT_WG;
    auto cgf = [&](sycl::handler& handle) {
        sycl::local_accessor<float, 1> slm(
            sycl::range<1>(groups_per_wg * group_size), handle);
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(global), sycl::range<1>(DG2_CONVROT_WG)),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(DG2_CONVROT_SG)]] {
                auto sg = item.get_sub_group();
                const int lane = static_cast<int>(sg.get_local_linear_id());
                const int sub = static_cast<int>(sg.get_group_linear_id());
                const int64_t row =
                    static_cast<int64_t>(item.get_group(0) / wgs);
                const int64_t wg_group =
                    static_cast<int64_t>(item.get_group(0) % wgs);
                const int64_t raw_group = wg_group * groups_per_wg + sub;
                const int64_t group =
                    raw_group < groups ? raw_group : (groups - 1);
                const bool valid = raw_group < groups;
                const int64_t group_offset = group * group_size;
                const int64_t slm_offset = sub * group_size;
                const InputT* __restrict__ row_ptr =
                    input + row * groups * group_size;

                auto rotate_group = [&]() {
                    constexpr int VEC = 8;
                    const int64_t lane_start =
                        static_cast<int64_t>(lane) * VEC;
                    if (group_size % VEC == 0) {
                        using VecT = sycl::vec<InputT, VEC>;
                        for (int64_t k = lane_start; k < group_size;
                             k += DG2_CONVROT_SG * VEC) {
                            const VecT values =
                                *reinterpret_cast<const VecT*>(
                                    row_ptr + group_offset + k);
                            for (int i = 0; i < VEC; ++i) {
                                slm[slm_offset + k + i] =
                                    static_cast<float>(values[i]);
                            }
                        }
                    } else {
                        for (int64_t k = lane; k < group_size;
                             k += DG2_CONVROT_SG) {
                            slm[slm_offset + k] = static_cast<float>(
                                row_ptr[group_offset + k]);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    for (int64_t stage = 0; stage < stages; ++stage) {
                        const int64_t stride =
                            static_cast<int64_t>(1) << (2 * stage);
                        for (int64_t r = lane; r < stride; r += DG2_CONVROT_SG) {
                            for (int64_t b = r; b < group_size;
                                 b += 4 * stride) {
                                const float x0 = slm[slm_offset + b];
                                const float x1 =
                                    slm[slm_offset + b + stride];
                                const float x2 =
                                    slm[slm_offset + b + 2 * stride];
                                const float x3 =
                                    slm[slm_offset + b + 3 * stride];
                                slm[slm_offset + b] = x0 + x1 + x2 - x3;
                                slm[slm_offset + b + stride] =
                                    x0 + x1 - x2 + x3;
                                slm[slm_offset + b + 2 * stride] =
                                    x0 - x1 + x2 + x3;
                                slm[slm_offset + b + 3 * stride] =
                                    -x0 + x1 + x2 + x3;
                            }
                        }
                        item.barrier(sycl::access::fence_space::local_space);
                    }
                };

                rotate_group();

                const float scale = row_scales[row];
                // q = round(raw * 127 / row_max_raw); row_max_raw =
                // scale * sqrt(G) * 127, so quant_inv = inv_sqrt_group/scale.
                const float quant_inv =
                    scale > 0.0f ? inv_sqrt_group / scale : 0.0f;
                int8_t* __restrict__ out_ptr =
                    output + row * groups * group_size;
                if (valid) {
                    for (int64_t k = lane; k < group_size;
                         k += DG2_CONVROT_SG) {
                        float q = sycl::rint(
                            slm[slm_offset + k] * quant_inv);
                        q = sycl::fmax(-128.0f, sycl::fmin(127.0f, q));
                        out_ptr[group_offset + k] = static_cast<int8_t>(
                            static_cast<int32_t>(q));
                    }
                }
            });
    };
    utils::submit_kernel(cgf, device, "int8_convrot_quantize_dg2");
}

template <typename InputT>
void dispatch_dg2(
    const torch::Tensor& input,
    torch::Tensor& output,
    torch::Tensor& scales,
    torch::Tensor& group_maxes,
    int64_t group_size) {
    const int64_t rows = input.size(0);
    const int64_t groups = input.size(1) / group_size;
    const int64_t stages =
        static_cast<int64_t>(std::log(static_cast<double>(group_size)) /
                             std::log(4.0) + 0.5);
    const float inv_sqrt_group =
        1.0f / std::sqrt(static_cast<float>(group_size));
    const auto* input_ptr = reinterpret_cast<const InputT*>(input.data_ptr());
    auto* output_ptr = reinterpret_cast<int8_t*>(output.data_ptr());
    auto* scales_ptr = scales.data_ptr<float>();
    auto* group_maxes_ptr = group_maxes.data_ptr<float>();

    convrot_rotate_max_kernel<InputT>(
        input_ptr, group_maxes_ptr, rows, groups, group_size, stages,
        input.device());
    convrot_row_reduce_kernel<InputT>(
        group_maxes_ptr, scales_ptr, rows, groups, inv_sqrt_group,
        input.device());
    convrot_quantize_kernel<InputT>(
        input_ptr, scales_ptr, output_ptr, rows, groups, group_size, stages,
        inv_sqrt_group, input.device());
}

}  // namespace

std::tuple<torch::Tensor, torch::Tensor> quantize_int8_convrot_fused_dg2(
    torch::Tensor input,
    int64_t group_size) {
    TORCH_CHECK(input.device().is_xpu(), "input must be on XPU");
    TORCH_CHECK(input.dim() == 2, "input must be 2D");
    TORCH_CHECK(
        input.scalar_type() == torch::kBFloat16 ||
            input.scalar_type() == torch::kHalf,
        "input must be bf16 or fp16");
    TORCH_CHECK(
        group_size == 64 || group_size == 256,
        "DG2 fused ConvRot quantization supports group sizes 64 and 256");
    TORCH_CHECK(
        input.size(1) % group_size == 0, "invalid group size");
    TORCH_CHECK(
        input.size(1) <= 16384,
        "DG2 fused ConvRot quantization supports K <= 16384");

    input = input.contiguous();
    auto output = torch::empty_like(
        input, input.options().dtype(torch::kInt8));
    auto scales = torch::empty(
        {input.size(0)}, input.options().dtype(torch::kFloat));
    auto group_maxes = torch::empty(
        {input.size(0), input.size(1) / group_size},
        input.options().dtype(torch::kFloat));
    if (input.numel() == 0) return {output, scales};

    if (input.scalar_type() == torch::kBFloat16) {
        dispatch_dg2<bf16>(input, output, scales, group_maxes, group_size);
    } else {
        dispatch_dg2<fp16>(input, output, scales, group_maxes, group_size);
    }
    return {output, scales};
}

}  // namespace int8_ops
}  // namespace omni_xpu
