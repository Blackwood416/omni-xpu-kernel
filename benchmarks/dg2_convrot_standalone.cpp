// Standalone A770 ConvRot fused-kernel configuration sweep.
//
// Compile:
//   icpx -fsycl -fsycl-targets=spir64_gen -Xs "-device dg2" -O2 \
//        dg2_convrot_standalone.cpp -o dg2_convrot_standalone.exe

#include <sycl/sycl.hpp>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using bf16 = sycl::ext::oneapi::bfloat16;

template <int WG, int SG>
void rotate_max_kernel(
    const bf16* __restrict__ input,
    float* __restrict__ group_maxes,
    int64_t rows,
    int64_t groups,
    int64_t group_size,
    int64_t stages,
    sycl::queue& queue) {
    constexpr int SUBS = WG / SG;
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
                    // group_size is fixed at 256 in this sweep.
                    constexpr int VEC = 256 / SG;
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

                    for (int64_t stage = 0; stage < stages; ++stage) {
                        const int64_t stride =
                            static_cast<int64_t>(1) << (2 * stage);
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
                };

                rotate_group();

                float local_max = 0.0f;
                for (int64_t k = lane; k < group_size; k += SG) {
                    local_max = sycl::fmax(
                        local_max, sycl::fabs(slm[slm_offset + k]));
                }
                const float group_max = sycl::reduce_over_group(
                    sg, local_max, sycl::maximum<float>());
                if (lane == 0 && valid) {
                    group_maxes[row * groups + group] = group_max;
                }
            });
    });
}

template <int WG, int SG>
double run_a(const bf16* in, float* scratch, int64_t rows, int64_t groups,
             int64_t group_size, sycl::queue& queue, int iters) {
    const int64_t stages =
        static_cast<int64_t>(std::log(static_cast<double>(group_size)) /
                             std::log(4.0) + 0.5);
    for (int i = 0; i < 3; ++i)
        rotate_max_kernel<WG, SG>(in, scratch, rows, groups, group_size,
                                  stages, queue);
    queue.wait();
    auto t0 = std::chrono::steady_clock::now();
    for (int i = 0; i < iters; ++i)
        rotate_max_kernel<WG, SG>(in, scratch, rows, groups, group_size,
                                  stages, queue);
    queue.wait();
    auto t1 = std::chrono::steady_clock::now();
    return std::chrono::duration<double, std::milli>(t1 - t0).count() / iters;
}

int main() {
    try {
        sycl::queue queue(sycl::gpu_selector_v);
        std::printf(
            "device: %s\n",
            queue.get_device().get_info<sycl::info::device::name>().c_str());

        const int64_t G = 256;
        const int64_t rows = 4096;
        const int64_t groups = 56;  // K = 14336
        const int64_t K = groups * G;
        std::vector<bf16> host_in(rows * K);
        for (int64_t i = 0; i < rows * K; ++i) {
            host_in[i] = static_cast<bf16>(
                static_cast<float>((i % 256)) / 8.0f);
        }
        bf16* in = sycl::malloc_device<bf16>(rows * K, queue);
        float* scratch = sycl::malloc_device<float>(rows * groups, queue);
        queue.memcpy(in, host_in.data(), rows * K * sizeof(bf16)).wait();

        const double read_gb = static_cast<double>(rows * K * 2) / 1e9;
        std::printf("input %.3f GB read per kernel A call\n", read_gb);
        std::printf("%-14s %10s %10s\n", "config", "ms/call", "GB/s");
        const int iters = 20;
        std::printf("%-14s %10.3f %10.1f\n", "WG256/SG32",
                    run_a<256, 32>(in, scratch, rows, groups, G, queue, iters),
                    read_gb / (run_a<256, 32>(in, scratch, rows, groups, G,
                                              queue, iters) / 1e3));
        std::printf("%-14s %10.3f %10.1f\n", "WG128/SG32",
                    run_a<128, 32>(in, scratch, rows, groups, G, queue, iters),
                    read_gb / (run_a<128, 32>(in, scratch, rows, groups, G,
                                              queue, iters) / 1e3));
        std::printf("%-14s %10.3f %10.1f\n", "WG64/SG32",
                    run_a<64, 32>(in, scratch, rows, groups, G, queue, iters),
                    read_gb / (run_a<64, 32>(in, scratch, rows, groups, G,
                                             queue, iters) / 1e3));
        std::printf("%-14s %10.3f %10.1f\n", "WG32/SG32",
                    run_a<32, 32>(in, scratch, rows, groups, G, queue, iters),
                    read_gb / (run_a<32, 32>(in, scratch, rows, groups, G,
                                             queue, iters) / 1e3));
        std::printf("%-14s %10.3f %10.1f\n", "WG256/SG16",
                    run_a<256, 16>(in, scratch, rows, groups, G, queue, iters),
                    read_gb / (run_a<256, 16>(in, scratch, rows, groups, G,
                                              queue, iters) / 1e3));

        sycl::free(in, queue);
        sycl::free(scratch, queue);
        return 0;
    } catch (const sycl::exception& e) {
        std::printf("SYCL error: %s\n", e.what());
        return 1;
    } catch (const std::exception& e) {
        std::printf("error: %s\n", e.what());
        return 1;
    }
}
