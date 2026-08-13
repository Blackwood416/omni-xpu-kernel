// Standalone A770 driver for VTune profiling of the DG2 v4 SDP sidecar on
// the exact MiniMax H3 main-model contract: bf16, L=KV=20683, H=56, D=128.
//
// Loads the installed lgrf_sdp sidecar and calls sdp_bf16io repeatedly so
// VTune sees only the pack/attn kernels without python/torch host noise.
//
// Build (Windows, oneAPI env + VS env):
//   icx -fsycl /O2 /std:c++17 /EHsc /DNOMINMAX /DWIN32_LEAN_AND_MEAN
//       dg2v4_h3_vtune_driver.cpp /Fe:dg2v4_h3_vtune_driver.exe
//
// Profile:
//   vtune -collect gpu-hotspots -knob characterization-mode=instruction-count
//         -result-dir C:/Temp/vtune_h3 -- dg2v4_h3_vtune_driver.exe

#include <sycl/sycl.hpp>

#include <windows.h>

#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <random>
#include <vector>

using bf16 = sycl::ext::oneapi::bfloat16;

using sdp_kernel_fn = void (*)(void*, void*, void*, void*, void*, int, int,
                               int, int, void*);

int main(int argc, char** argv) {
  const int L = argc > 1 ? std::atoi(argv[1]) : 20683;
  const int KV = argc > 2 ? std::atoi(argv[2]) : L;
  constexpr int H = 56;
  constexpr int D = 128;
  const int ITERS = argc > 3 ? std::atoi(argv[3]) : 16;

  sycl::queue q(sycl::gpu_selector_v, sycl::property::queue::in_order());
  std::printf("device: %s\n",
              q.get_device().get_info<sycl::info::device::name>().c_str());
  std::printf("shape: L=%d KV=%d H=%d D=%d bf16 iters=%d\n", L, KV, H, D,
              ITERS);

  auto alloc = [&](size_t bytes) {
    return static_cast<bf16*>(sycl::aligned_alloc_device(64, bytes, q));
  };
  bf16* qs = alloc((size_t)L * H * D * sizeof(bf16));
  bf16* ks = alloc((size_t)KV * H * D * sizeof(bf16));
  bf16* vs = alloc((size_t)KV * H * D * sizeof(bf16));
  bf16* out = alloc((size_t)L * H * D * sizeof(bf16));
  float* alpha = static_cast<float*>(
      sycl::aligned_alloc_device(64, (size_t)H * D * sizeof(float), q));

  std::vector<bf16> hq((size_t)L * H * D), hk((size_t)KV * H * D),
      hv((size_t)KV * H * D);
  std::mt19937 rng(0);
  std::normal_distribution<float> dist(0.0f, 0.25f);
  for (auto& x : hq) x = bf16(dist(rng));
  for (auto& x : hk) x = bf16(dist(rng));
  for (auto& x : hv) x = bf16(dist(rng));
  q.memcpy(qs, hq.data(), hq.size() * sizeof(bf16)).wait();
  q.memcpy(ks, hk.data(), hk.size() * sizeof(bf16)).wait();
  q.memcpy(vs, hv.data(), hv.size() * sizeof(bf16)).wait();
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
  auto sdp_bf16io = reinterpret_cast<sdp_kernel_fn>(
      GetProcAddress(lib, "sdp_bf16io"));
  if (!sdp_bf16io) {
    std::printf("sdp_bf16io not found\n");
    return 1;
  }

  void* qp = &q;
  // Warm up 4 calls, then time ITERS.
  for (int it = 0; it < 4; it++) {
    sdp_bf16io(qs, ks, vs, alpha, out, L, KV, H, H, qp);
    q.wait();
  }
  auto t0 = std::chrono::steady_clock::now();
  for (int it = 0; it < ITERS; it++) {
    sdp_bf16io(qs, ks, vs, alpha, out, L, KV, H, H, qp);
    q.wait();
  }
  auto t1 = std::chrono::steady_clock::now();
  std::printf("steady per-call: %.3f ms\n",
              std::chrono::duration<double, std::milli>(t1 - t0).count() /
                  ITERS);
  q.wait();

  sycl::free(qs, q);
  sycl::free(ks, q);
  sycl::free(vs, q);
  sycl::free(out, q);
  sycl::free(alpha, q);
  return 0;
}
