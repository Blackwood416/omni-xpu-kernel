// Measures per-submit wall overhead of an empty ESIMD kernel on A770.
//
// Build (Windows):
//   icx -fsycl -fsycl-targets=spir64_gen -Xs "-device dg2 -options -doubleGRF"
//       /O2 /std:c++17 /EHsc empty_kernel_overhead.cpp /Fe:empty_kernel_overhead.exe

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <chrono>
#include <cstdio>

using namespace sycl::ext::intel::esimd;

int main() {
  constexpr int ITERS = 200;
  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
  // Warm up.
  for (int i = 0; i < 20; i++) {
    q.submit([&](sycl::handler& cgh) {
      cgh.single_task([=]() SYCL_ESIMD_KERNEL {});
    });
  }
  q.wait();

  auto t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < ITERS; i++) {
    q.submit([&](sycl::handler& cgh) {
      cgh.single_task([=]() SYCL_ESIMD_KERNEL {});
    });
  }
  q.wait();
  auto t1 = std::chrono::steady_clock::now();
  std::printf("submit-only loop: %.3f us/call\n",
              std::chrono::duration<double, std::micro>(t1 - t0).count() /
                  ITERS);

  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < ITERS; i++) {
    q.submit([&](sycl::handler& cgh) {
      cgh.single_task([=]() SYCL_ESIMD_KERNEL {});
    }).wait();
  }
  t1 = std::chrono::steady_clock::now();
  std::printf("submit+wait:      %.3f us/call\n",
              std::chrono::duration<double, std::micro>(t1 - t0).count() /
                  ITERS);

  t0 = std::chrono::steady_clock::now();
  for (int i = 0; i < ITERS; i++) {
    q.submit([&](sycl::handler& cgh) {
      cgh.single_task([=]() SYCL_ESIMD_KERNEL {});
    });
    q.submit([&](sycl::handler& cgh) {
      cgh.single_task([=]() SYCL_ESIMD_KERNEL {});
    });
    q.wait();
  }
  t1 = std::chrono::steady_clock::now();
  std::printf("2-submit+wait:    %.3f us/call\n",
              std::chrono::duration<double, std::micro>(t1 - t0).count() /
                  ITERS);
  return 0;
}
