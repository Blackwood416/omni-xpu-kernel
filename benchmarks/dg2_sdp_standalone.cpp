// Standalone DG2 SDP kernel harness (no torch). Fills the output with a
// sentinel before launch so a missing store is immediately visible, runs the
// kernel several times, and compares against a CPU reference.
//
// Build (Windows, oneAPI 2026.1, VS env initialized):
//   icx -fsycl -fsycl-targets=spir64_gen -Xs "-device dg2" /O2 /std:c++17
//       /Fe:dg2_sdp_standalone.exe benchmarks\dg2_sdp_standalone.cpp

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

#define __ESIMD_NS  sycl::ext::intel::esimd
#define __ESIMD_ENS sycl::ext::intel::experimental::esimd
#undef  ESIMD_INLINE
#define ESIMD_INLINE inline __attribute__((always_inline))

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl::ext::intel::experimental::esimd;

namespace dg2 {
#include "../omni_xpu_kernel/lgrf_uni/single_kernels/flash.attn.b.mha.dg2.h"
}

constexpr int QLEN = 32;
constexpr int KV = 32;
constexpr int H = 1;
constexpr int D = 128;
constexpr float SENTINEL = 12345.0f;

static void cpu_ref(const std::vector<fp16>& q, const std::vector<fp16>& k,
                    const std::vector<fp16>& v, std::vector<float>& out) {
  const float scale = 1.0f / std::sqrt((float)D);
  for (int r = 0; r < QLEN; r++) {
    for (int h = 0; h < H; h++) {
      float mx = -1e30f, sum = 0.0f;
      std::vector<float> p(KV);
      for (int j = 0; j < KV; j++) {
        float s = 0.0f;
        for (int d = 0; d < D; d++) {
          s += float(q[(r * H + h) * D + d]) * float(k[(j * H + h) * D + d]);
        }
        s *= scale;
        p[j] = s;
        mx = std::max(mx, s);
      }
      for (int j = 0; j < KV; j++) {
        p[j] = std::exp(p[j] - mx);
        sum += p[j];
      }
      for (int d = 0; d < D; d++) {
        float acc = 0.0f;
        for (int j = 0; j < KV; j++) {
          acc += p[j] * float(v[(j * H + h) * D + d]);
        }
        out[(r * H + h) * D + d] = acc / sum;
      }
    }
  }
}

static ESIMD_INLINE void bisect0(uint8_t* qState, uint8_t* kState,
                                 uint8_t* vState, uint8_t* normAlpha,
                                 uint8_t* out, uint32_t actLen, uint32_t kvLen,
                                 uint32_t headQ, uint32_t headKv,
                                 sycl::nd_item<2>& ndi) {
  constexpr int BM = 32;
  constexpr int BN = 32;
  constexpr int VLEN = 8;
  constexpr int DGROUPS = D / VLEN;
  constexpr int SLM_TOTAL_BYTES = 2 * BN * D * static_cast<int>(sizeof(fp16));
  const int lid = static_cast<int>(ndi.get_local_id(0));
  const int headIdx = static_cast<int>(ndi.get_group(0)) % static_cast<int>(headQ);
  const int qTile = static_cast<int>(ndi.get_group(1));
  const int qrow = qTile * BM + lid / DGROUPS;
  const int dbase = (lid % DGROUPS) * VLEN;
  slm_init(SLM_TOTAL_BYTES);
  if (qrow == 0 && dbase == 0) {
    simd<fp16, 8> m = fp16(1.0f);
    block_store(reinterpret_cast<fp16*>(out), m, overaligned_tag<16>{});
  }
  (void)actLen;
  (void)kvLen;
  (void)headKv;
  (void)normAlpha;
  (void)qState;
  (void)kState;
  (void)vState;
  (void)qrow;
  (void)qTile;
  (void)headIdx;
  (void)lid;
}

int main() {
  std::printf("main start\n");
  std::fflush(stdout);
  sycl::queue q;
  std::printf("device: %s\n", q.get_device().get_info<sycl::info::device::name>().c_str());

  auto* qd = sycl::malloc_device<fp16>(QLEN * H * D, q);
  auto* kd = sycl::malloc_device<fp16>(KV * H * D, q);
  auto* vd = sycl::malloc_device<fp16>(KV * H * D, q);
  auto* od = sycl::malloc_device<fp16>(QLEN * H * D, q);
  auto* ad = sycl::malloc_device<float>(H * D, q);
  std::vector<fp16> qh(QLEN * H * D), kh(KV * H * D), vh(KV * H * D);
  std::vector<float> ref(QLEN * H * D), outh(QLEN * H * D);
  std::vector<float> alpha(H * D, 1.0f);

  for (int i = 0; i < QLEN * H * D; i++) {
    qh[i] = fp16(std::sin(0.01f * i) * 0.5f);
    vh[i] = fp16(std::cos(0.013f * i) * 0.5f);
  }
  for (int i = 0; i < KV * H * D; i++) {
    kh[i] = fp16(std::sin(0.017f * i) * 0.5f);
  }
  cpu_ref(qh, kh, vh, ref);

  q.memcpy(qd, qh.data(), qh.size() * sizeof(fp16)).wait();
  q.memcpy(kd, kh.data(), kh.size() * sizeof(fp16)).wait();
  q.memcpy(vd, vh.data(), vh.size() * sizeof(fp16)).wait();
  q.memcpy(ad, alpha.data(), alpha.size() * sizeof(float)).wait();

  // Control: trivial ESIMD kernel writes a constant through the same store
  // path. If the sentinel survives here, the harness/toolchain is the issue.
  {
    std::vector<fp16> sentinel(QLEN * H * D, fp16(SENTINEL));
    q.memcpy(od, sentinel.data(), sentinel.size() * sizeof(fp16)).wait();
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(sycl::range<1>(1), [=](sycl::id<1>) SYCL_ESIMD_KERNEL {
        simd<fp16, 8> x = fp16(3.25f);
        block_store(reinterpret_cast<fp16*>(od), x, overaligned_tag<16>{});
      });
    }).wait();
    std::vector<fp16> ctl(QLEN * H * D);
    q.memcpy(ctl.data(), od, ctl.size() * sizeof(fp16)).wait();
    std::printf("control: out0=%f (expect 3.25)\n", float(ctl[0]));
  }

  {
    std::vector<fp16> sentinel(QLEN * H * D, fp16(SENTINEL));
    q.memcpy(od, sentinel.data(), sentinel.size() * sizeof(fp16)).wait();
    // WG=32 probe: ESIMD + WG=512 triggered GPU TDR (LiveKernelEvent 141).
    sycl::nd_range<2> ndr({32, 1}, {32, 1});
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(ndr, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
        bisect0(reinterpret_cast<uint8_t*>(qd), reinterpret_cast<uint8_t*>(kd),
                reinterpret_cast<uint8_t*>(vd), reinterpret_cast<uint8_t*>(ad),
                reinterpret_cast<uint8_t*>(od), QLEN, KV, H, H, ndi);
      });
    }).wait();
    std::vector<fp16> raw(QLEN * H * D);
    q.memcpy(raw.data(), od, raw.size() * sizeof(fp16)).wait();
    std::printf("bisect0: out0=%f\n", float(raw[0]));
  }

  for (int run = 0; run < 5; run++) {
    std::vector<fp16> sentinel(QLEN * H * D, fp16(SENTINEL));
    q.memcpy(od, sentinel.data(), sentinel.size() * sizeof(fp16)).wait();

    sycl::nd_range<2> ndr = dg2::flash_ndr(QLEN, H, D);
    q.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(ndr, [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
        dg2::flashAttnDg2Precomputed<fp16, D>(
            reinterpret_cast<uint8_t*>(qd), reinterpret_cast<uint8_t*>(kd),
            reinterpret_cast<uint8_t*>(vd), reinterpret_cast<uint8_t*>(ad),
            reinterpret_cast<uint8_t*>(od), QLEN, KV, H, H, ndi);
      });
    }).wait();

    std::vector<fp16> raw(QLEN * H * D);
    q.memcpy(raw.data(), od, raw.size() * sizeof(fp16)).wait();
    for (int i = 0; i < (int)raw.size(); i++) outh[i] = float(raw[i]);

    int sentinel_hits = 0;
    float max_err = 0.0f;
    for (int i = 0; i < (int)outh.size(); i++) {
      if (std::abs(outh[i] - SENTINEL) < 1.0f) sentinel_hits++;
      max_err = std::max(max_err, std::abs(outh[i] - ref[i]));
    }
    std::printf("run %d: sentinel_hits=%d max_err=%.5f out0=%f,%f,%f,%f\n",
                run, sentinel_hits, max_err, outh[0], outh[1], outh[2], outh[3]);
  }

  sycl::free(qd, q);
  sycl::free(kd, q);
  sycl::free(vd, q);
  sycl::free(od, q);
  sycl::free(ad, q);
  return 0;
}
