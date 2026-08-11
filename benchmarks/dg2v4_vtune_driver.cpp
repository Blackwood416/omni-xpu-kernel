// Standalone A770 driver for VTune profiling of the DG2 v4 SDP sidecar.
//
// Loads the installed lgrf_sdp sidecar and calls sdp_fp16 repeatedly on a
// fixed L/KV/H=32/D=128 fp16 input, so VTune sees only the pack/attn
// kernels without python/torch host noise.
//
// Build (Windows, oneAPI env + VS env):
//   icx -fsycl /O2 /std:c++17 /EHsc /DNOMINMAX /DWIN32_LEAN_AND_MEAN
//       dg2v3_vtune_driver.cpp /Fe:dg2v3_vtune_driver.exe
//
// Profile:
//   vtune -collect gpu-hotspots -knob characterization-mode=instruction-count
//         -result-dir C:/Temp/vtune_v3 -- dg2v3_vtune_driver.exe

#include <sycl/sycl.hpp>

#include <windows.h>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <chrono>
#include <random>
#include <vector>

using fp16 = sycl::half;

using sdp_kernel_fn = void (*)(void*, void*, void*, void*, void*, int, int,
                               int, int, void*);

int main(int argc, char** argv) {
  const int L = argc > 1 ? std::atoi(argv[1]) : 8192;
  const int KV = argc > 2 ? std::atoi(argv[2]) : L;
  constexpr int H = 32;
  constexpr int D = 128;
  constexpr int ITERS = 12;

  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
  std::printf("device: %s\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());

  auto alloc = [&](size_t bytes) {
    return static_cast<fp16*>(sycl::aligned_alloc_device(64, bytes, q));
  };
  fp16* qs = alloc((size_t)L * H * D * sizeof(fp16));
  fp16* ks = alloc((size_t)KV * H * D * sizeof(fp16));
  fp16* vs = alloc((size_t)KV * H * D * sizeof(fp16));
  fp16* out = alloc((size_t)L * H * D * sizeof(fp16));
  float* alpha = static_cast<float*>(
      sycl::aligned_alloc_device(64, (size_t)H * D * sizeof(float), q));

  std::vector<fp16> hq((size_t)L * H * D), hk((size_t)KV * H * D),
      hv((size_t)KV * H * D);
  std::mt19937 rng(0);
  std::normal_distribution<float> dist(0.0f, 0.5f);
  for (auto& x : hq) x = fp16(dist(rng));
  for (auto& x : hk) x = fp16(dist(rng));
  for (auto& x : hv) x = fp16(dist(rng));
  q.memcpy(qs, hq.data(), hq.size() * sizeof(fp16)).wait();
  q.memcpy(ks, hk.data(), hk.size() * sizeof(fp16)).wait();
  q.memcpy(vs, hv.data(), hv.size() * sizeof(fp16)).wait();
  std::vector<float> ha((size_t)H * D, 1.0f);
  q.memcpy(alpha, ha.data(), ha.size() * sizeof(float)).wait();

  const char* pyd =
      "C:/Users/Administrator/ComfyUI_windows_portable/python_embeded/"
      "Lib/site-packages/omni_xpu_kernel/lgrf_uni/"
      "lgrf_sdp.cp313-win_amd64.pyd";
  HMODULE lib = LoadLibraryA(pyd);
  if (!lib) {
    std::printf("LoadLibrary failed: %lu\n", (unsigned long)GetLastError());
    return 1;
  }
  auto sdp_fp16 = reinterpret_cast<sdp_kernel_fn>(
      GetProcAddress(lib, "sdp_fp16"));
  if (!sdp_fp16) {
    std::printf("sdp_fp16 not found\n");
    return 1;
  }

  void* qp = &q;
  auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < ITERS; it++) {
    sdp_fp16(qs, ks, vs, alpha, out, L, KV, H, H, qp);
  }
  auto t1 = std::chrono::steady_clock::now();
  std::printf("per-call wall: %.3f ms\n",
              std::chrono::duration<double, std::milli>(t1 - t0).count() /
                  ITERS);
  // Steady-state: warm up 4, then time 8.
  for (int it = 0; it < 4; it++) {
    sdp_fp16(qs, ks, vs, alpha, out, L, KV, H, H, qp);
  }
  auto t2 = std::chrono::steady_clock::now();
  for (int it = 0; it < 8; it++) {
    sdp_fp16(qs, ks, vs, alpha, out, L, KV, H, H, qp);
  }
  auto t3 = std::chrono::steady_clock::now();
  std::printf("steady per-call: %.3f ms\n",
              std::chrono::duration<double, std::milli>(t3 - t2).count() / 8);
  std::printf("done %d iterations\n", ITERS);
  q.wait();  // async sidecar: keep USM alive until all kernels finish

  sycl::free(qs, q);
  sycl::free(ks, q);
  sycl::free(vs, q);
  sycl::free(out, q);
  sycl::free(alpha, q);
  return 0;
}
