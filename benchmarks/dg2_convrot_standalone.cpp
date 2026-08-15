// Standalone A770 ConvRot butterfly harness (no torch/pybind).
//
// Compile:
//   icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device dg2" -O2 \
//        dg2_convrot_standalone.cpp -o dg2_convrot_standalone.exe

#include <sycl/sycl.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using bf16 = sycl::ext::oneapi::bfloat16;

constexpr int32_t SG = 32;
constexpr int32_t WG = 256;
constexpr int32_t SUBS = WG / SG;

void rotate_quantize_kernel(
    const bf16* __restrict__ input,
    int8_t* __restrict__ output,
    int64_t rows,
    int64_t groups,
    int64_t group_size,
    float inv_sqrt_group,
    float scale,
    sycl::queue& queue) {
    const int64_t groups_per_wg = SUBS;
    const int64_t wgs = (groups + groups_per_wg - 1) / groups_per_wg;
    const size_t global = static_cast<size_t>(rows) * wgs * WG;
    queue.submit([&](sycl::handler& handle) {
        sycl::local_accessor<float, 1> slm(
            sycl::range<1>(groups_per_wg * group_size), handle);
        handle.parallel_for(
            sycl::nd_range<1>(
                sycl::range<1>(global), sycl::range<1>(WG)),
            [=](sycl::nd_item<1> item)
                [[sycl::reqd_sub_group_size(SG)]] {
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
                const bf16* __restrict__ row_ptr =
                    input + row * groups * group_size;

                auto rotate_group = [&]() {
                    constexpr int VEC = 8;
                    const int64_t lane_start =
                        static_cast<int64_t>(lane) * VEC;
                    if (group_size % VEC == 0) {
                        using VecT = sycl::vec<bf16, VEC>;
                        for (int64_t k = lane_start; k < group_size;
                             k += SG * VEC) {
                            const VecT values =
                                *reinterpret_cast<const VecT*>(
                                    row_ptr + group_offset + k);
                            for (int i = 0; i < VEC; ++i) {
                                slm[slm_offset + k + i] =
                                    static_cast<float>(values[i]);
                            }
                        }
                    } else {
                        for (int64_t k = lane; k < group_size; k += SG) {
                            slm[slm_offset + k] = static_cast<float>(
                                row_ptr[group_offset + k]);
                        }
                    }
                    item.barrier(sycl::access::fence_space::local_space);

                    // Compile-time unrolled radix-4 butterfly (G=256).
                    const int64_t strides[4] = {1, 4, 16, 64};
                    for (int64_t stage = 0; stage < 4; ++stage) {
                        const int64_t stride = strides[stage];
                        for (int64_t r = lane; r < stride; r += SG) {
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

                    // DEBUG: round without the bf16 cast (pure float *).
                    for (int64_t k = lane; k < group_size; k += SG) {
                        slm[slm_offset + k] =
                            slm[slm_offset + k] * inv_sqrt_group;
                    }
                    item.barrier(sycl::access::fence_space::local_space);
                };

                rotate_group();

                const float quant_inv = 127.0f / scale;
                int8_t* __restrict__ out_ptr =
                    output + row * groups * group_size;
                if (valid) {
                    for (int64_t k = lane; k < group_size; k += SG) {
                        float q = sycl::rint(
                            slm[slm_offset + k] * quant_inv);
                        q = sycl::fmax(-128.0f, sycl::fmin(127.0f, q));
                        out_ptr[group_offset + k] = static_cast<int8_t>(
                            static_cast<int32_t>(q));
                    }
                }
            });
    });
}

int main() {
    try {
        sycl::queue queue(sycl::gpu_selector_v);
        std::printf(
            "device: %s\n",
            queue.get_device().get_info<sycl::info::device::name>().c_str());
        const int64_t G = 256;
        const int64_t K = G;
        const int64_t groups = 1;
        const int64_t rows = 1;
        const float inv_sqrt = 1.0f / 16.0f;

        std::vector<bf16> host_in(K);
        for (int64_t i = 0; i < K; ++i) {
            host_in[i] = static_cast<bf16>(static_cast<float>(i) / 8.0f);
        }
        std::vector<int8_t> host_out(K, -99);

        bf16* in = sycl::malloc_device<bf16>(K, queue);
        int8_t* out = sycl::malloc_device<int8_t>(K, queue);
        queue.memcpy(in, host_in.data(), K * sizeof(bf16)).wait();

        // Correct scale for this ramp: max |rotated| = 32 -> 32/127.
        rotate_quantize_kernel(
            in, out, rows, groups, G, inv_sqrt, 32.0f / 127.0f, queue);
        queue.wait();
        queue.memcpy(host_out.data(), out, K * sizeof(int8_t)).wait();

        std::printf("out[0..15]:");
        for (int i = 0; i < 16; ++i) std::printf(" %d", host_out[i]);
        std::printf("\n");

        // CPU butterfly reference.
        std::vector<float> raw(K);
        for (int64_t i = 0; i < K; ++i) raw[i] = static_cast<float>(host_in[i]);
        const int64_t strides[4] = {1, 4, 16, 64};
        for (int64_t stage = 0; stage < 4; ++stage) {
            const int64_t stride = strides[stage];
            std::vector<float> next = raw;
            for (int64_t r = 0; r < stride; ++r) {
                for (int64_t b = r; b < K; b += 4 * stride) {
                    const float x0 = raw[b], x1 = raw[b + stride];
                    const float x2 = raw[b + 2 * stride],
                                x3 = raw[b + 3 * stride];
                    next[b] = x0 + x1 + x2 - x3;
                    next[b + stride] = x0 + x1 - x2 + x3;
                    next[b + 2 * stride] = x0 - x1 + x2 + x3;
                    next[b + 3 * stride] = -x0 + x1 + x2 + x3;
                }
            }
            raw.swap(next);
        }
        int exact = 0;
        for (int64_t i = 0; i < K; ++i) {
            const float rotated = raw[i] * inv_sqrt;
            const int expected_q =
                static_cast<int>(std::rint(rotated * 127.0f / 32.0f));
            if (host_out[i] == expected_q) ++exact;
        }
        std::printf("exact %d/%lld\n", exact, static_cast<long long>(K));

        sycl::free(in, queue);
        sycl::free(out, queue);
        return 0;
    } catch (const sycl::exception& e) {
        std::printf("SYCL error: %s\n", e.what());
        return 1;
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
}
