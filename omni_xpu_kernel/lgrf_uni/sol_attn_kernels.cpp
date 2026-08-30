// Sol-Attn DG2 ESIMD sparse Flash Attention — fp32 accumulation, WG=32.
//
// 按 A770 kernel skill 的证据纪律实现：
//  * 精确路径逐行照搬 DG2 SDP kernel（flash.attn.b.mha.dg2.h）的成熟结构：
//    SLM 半块协作搬入（16KB，零填充）+ m/l/rescale 在线 softmax + f32 累加。
//  * 稀疏只做最小增量：每 64-token KV 块按路由掩码选择精确（SLM 路径）或
//    近似（K 质心点积 + V 块和，分母按块长加权）。
//  * 原生多 head 网格（headIdx = get_group(0) % headQ），一次启动全算。
//  * [TOOLCHAIN] 多 head 布局直读在此前 self-invent 版本上被 implicit-ESIMD
//    编译器错误处理；本版以 DG2 验证过的显式 simd + SLM 结构规避。
// 契约：B=1、bf16、BTHD、D=128、自注意力、无 mask。

#include <algorithm>
#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "esimd_kernel_api.h"

using fp16 = sycl::half;
using bf16 = sycl::ext::oneapi::bfloat16;

#define __ESIMD_NS  sycl::ext::intel::esimd
#define __ESIMD_ENS sycl::ext::intel::experimental::esimd
#undef  ESIMD_INLINE
#define ESIMD_INLINE inline __attribute__((always_inline))

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl::ext::intel::experimental::esimd;

namespace sol_attn {

constexpr int SA_BM = 32;    // query rows per work-group
constexpr int SA_BN = 64;    // tokens per KV route block
constexpr int SA_WG = 32;    // work-group size (A770 TDR-safe)
constexpr int SA_HD = 128;   // head dim

template <typename T> struct IsFp16 : std::false_type {};
template <> struct IsFp16<sycl::half> : std::true_type {};

template <int N> ESIMD_INLINE float dg2_sum(simd<float, N> v) {
  constexpr int H = N / 2;
  simd<float, H> lo = v.template select<H, 1>(0);
  simd<float, H> hi = v.template select<H, 1>(H);
  lo += hi;
  if constexpr (H == 1) {
    return lo[0];
  } else {
    return dg2_sum<H>(lo);
  }
}

inline sycl::nd_range<2> sol_attn_ndr(int q_len, int headQ) {
  const int qTiles = (q_len + SA_BM - 1) / SA_BM;
  return sycl::nd_range<2>(
      {(size_t)(SA_WG * headQ), (size_t)qTiles}, {(size_t)SA_WG, 1});
}

template <typename ElemT, int HD>
ESIMD_INLINE void solAttnDg2(
    uint8_t* qState, uint8_t* kState, uint8_t* vState,
    uint8_t* kcState, uint8_t* vcState,
    uint8_t* exactIdx, uint8_t* approxIdx, uint8_t* nExact, uint8_t* nApprox,
    uint8_t* outState,
    uint32_t qLen, uint32_t kvLen, uint32_t headQ, uint32_t nBlocks,
    uint32_t qBlocks, uint32_t headIdxParam, float scale,
    sycl::nd_item<2>& ndi) {
  constexpr int BM = SA_BM;
  constexpr int BN = SA_BN;
  constexpr int WG = SA_WG;
  constexpr int SLM_K_BYTES = BN * HD * static_cast<int>(sizeof(ElemT));
  constexpr int SLM_TOTAL_BYTES = 2 * SLM_K_BYTES;
  constexpr float SCALE = 0.08838834764831844f;  // 1/sqrt(128)
  constexpr float LOG2E = 1.4426950408889634f;

  const int lid = static_cast<int>(ndi.get_local_id(0));
  const int headIdx = static_cast<int>(headIdxParam) +
                      static_cast<int>(ndi.get_group(0) % headQ);
  const int qTile = static_cast<int>(ndi.get_group(1));
  const int qrow = qTile * BM + lid;
  const int qblock = qTile / 2;  // 64 query rows per route block
  const size_t rowStride = static_cast<size_t>(headQ) * HD;
  const size_t headOffset = static_cast<size_t>(headIdx) * nBlocks * HD;

  const ElemT* q = reinterpret_cast<const ElemT*>(qState);
  const ElemT* kBase = reinterpret_cast<const ElemT*>(kState);
  const ElemT* vBase = reinterpret_cast<const ElemT*>(vState);
  const bf16* kc = reinterpret_cast<const bf16*>(kcState) + headOffset;
  const bf16* vc = reinterpret_cast<const bf16*>(vcState) + headOffset;
  const int32_t* exactList =
      reinterpret_cast<const int32_t*>(exactIdx) +
      static_cast<size_t>(headIdx) * qBlocks * nBlocks +
      static_cast<size_t>(qblock) * nBlocks;
  const int32_t* approxList =
      reinterpret_cast<const int32_t*>(approxIdx) +
      static_cast<size_t>(headIdx) * qBlocks * nBlocks +
      static_cast<size_t>(qblock) * nBlocks;
  const int32_t* nExactPtr = reinterpret_cast<const int32_t*>(nExact) +
                             static_cast<size_t>(headIdx) * qBlocks +
                             static_cast<size_t>(qblock);
  const int32_t* nApproxPtr = reinterpret_cast<const int32_t*>(nApprox) +
                              static_cast<size_t>(headIdx) * qBlocks +
                              static_cast<size_t>(qblock);
  const int nx = nExactPtr[0];
  const int na = nApproxPtr[0];

  slm_init(SLM_TOTAL_BYTES);

  simd<float, HD> qReg = 0;
  if (qrow < static_cast<int>(qLen)) {
    const ElemT* qp = q + static_cast<size_t>(qrow) * rowStride +
                      static_cast<size_t>(headIdx) * HD;
#pragma unroll
    for (int c = 0; c < HD / 16; ++c) {
      simd<float, 16> chunk = block_load<ElemT, 16>(
          qp + c * 16, overaligned_tag<16>{});
      qReg.template select<16, 1>(c * 16) = chunk;
    }
  }

  simd<float, HD> acc = 0;
  float m = -3.402823466e+38f;
  float l = 0.0f;

  // ── 精确块（无分支循环）──
  for (int ei = 0; ei < nx; ++ei) {
    const int kb = exactList[ei];
    {
      // ── 精确路径：DG2 逐行结构，整块（BN=64）一次搬入 SLM，每块 2 barrier ──
      const int kstart = kb * BN;
      int klen = static_cast<int>(kvLen) - kstart;
      if (klen > BN) klen = BN;
      if (klen < 0) klen = 0;
      const size_t gBase = static_cast<size_t>(kstart) * rowStride +
                           static_cast<size_t>(headIdx) * HD;
      const int validElems = klen * HD;
      for (int i = lid * 16; i < BN * HD; i += WG * 16) {
        if (i < validElems) {
          const int col = i % HD;
          simd<ElemT, 16> kchunk = block_load<ElemT, 16>(
              kBase + gBase + static_cast<size_t>(i / HD) * rowStride + col,
              overaligned_tag<16>{});
          slm_block_store(i * static_cast<int>(sizeof(ElemT)), kchunk,
                          overaligned_tag<16>{});
          simd<ElemT, 16> vchunk = block_load<ElemT, 16>(
              vBase + gBase + static_cast<size_t>(i / HD) * rowStride + col,
              overaligned_tag<16>{});
          slm_block_store(
              SLM_K_BYTES + i * static_cast<int>(sizeof(ElemT)), vchunk,
              overaligned_tag<16>{});
        } else {
          slm_block_store(i * static_cast<int>(sizeof(ElemT)),
                          simd<ElemT, 16>(0), overaligned_tag<16>{});
          slm_block_store(
              SLM_K_BYTES + i * static_cast<int>(sizeof(ElemT)),
              simd<ElemT, 16>(0), overaligned_tag<16>{});
        }
      }
      barrier();
      for (int j = 0; j < klen; ++j) {
        simd<float, HD> dotv = 0;
#pragma unroll
        for (int c = 0; c < HD / 16; ++c) {
          simd<float, 16> kf = slm_block_load<ElemT, 16>(
              (j * HD + c * 16) * static_cast<int>(sizeof(ElemT)),
              overaligned_tag<16>{});
          dotv.template select<16, 1>(c * 16) =
              dotv.template select<16, 1>(c * 16) +
              qReg.template select<16, 1>(c * 16) * kf;
        }
        float s = dg2_sum<HD>(dotv) * SCALE;
        const float mNew = (s > m) ? s : m;
        const float rescale = sycl::exp2((m - mNew) * LOG2E);
        const float p = sycl::exp2((s - mNew) * LOG2E);
        l = l * rescale + p;
#pragma unroll
        for (int c = 0; c < HD / 16; ++c) {
          simd<float, 16> vf = slm_block_load<ElemT, 16>(
              SLM_K_BYTES + (j * HD + c * 16) *
                  static_cast<int>(sizeof(ElemT)),
              overaligned_tag<16>{});
          acc.template select<16, 1>(c * 16) =
              acc.template select<16, 1>(c * 16) * rescale + p * vf;
        }
        m = mNew;
      }
      barrier();
    }
  }
  // ── 近似块（无分支循环）──
  for (int ai = 0; ai < na; ++ai) {
    const int kb = approxList[ai];
    {
      // ── 近似路径：K 质心点积 + V 块和，分母按块长加权 ──
      const int kstart = kb * BN;
      int klen = static_cast<int>(kvLen) - kstart;
      if (klen > BN) klen = BN;
      if (klen < 0) klen = 0;
      const bf16* kcp = kc + static_cast<size_t>(kb) * HD;
      const bf16* vcp = vc + static_cast<size_t>(kb) * HD;
      simd<float, HD> dotv = 0;
#pragma unroll
      for (int c = 0; c < HD / 16; ++c) {
        simd<float, 16> kf = block_load<bf16, 16>(
            kcp + c * 16, overaligned_tag<16>{});
        dotv.template select<16, 1>(c * 16) =
            dotv.template select<16, 1>(c * 16) +
            qReg.template select<16, 1>(c * 16) * kf;
      }
      float s = dg2_sum<HD>(dotv) * SCALE;
      const float mNew = (s > m) ? s : m;
      const float rescale = sycl::exp2((m - mNew) * LOG2E);
      const float p = sycl::exp2((s - mNew) * LOG2E);
      l = l * rescale + p * static_cast<float>(klen);
#pragma unroll
      for (int c = 0; c < HD / 16; ++c) {
        simd<float, 16> vf = block_load<bf16, 16>(
            vcp + c * 16, overaligned_tag<16>{});
      acc.template select<16, 1>(c * 16) =
          acc.template select<16, 1>(c * 16) * rescale + p * vf;
      }
      m = mNew;
    }
  }

  // ── 输出归一化 ──
  if (qrow < static_cast<int>(qLen) && l > 0.0f) {
    simd<float, HD> outv = acc * (1.0f / l);
    if constexpr (IsFp16<ElemT>::value) {
      outv.merge(65504.0f, outv > 65504.0f);
      outv.merge(-65504.0f, outv < -65504.0f);
    }
    ElemT* op = reinterpret_cast<ElemT*>(outState) +
                static_cast<size_t>(qrow) * rowStride +
                static_cast<size_t>(headIdx) * HD;
#pragma unroll
    for (int c = 0; c < HD / 16; ++c) {
      simd<float, 16> f16 = outv.template select<16, 1>(c * 16);
      simd<ElemT, 16> o16;
#pragma unroll
      for (int i = 0; i < 16; ++i) {
        o16[i] = static_cast<ElemT>(f16[i]);
      }
      block_store(op + c * 16, o16);
    }
  }
}

}  // namespace sol_attn

// ──────────────────────────────────────────────────────────────────────────────
// sol_attn_forward: BF16 BTHD D=128 sparse attention。
//   kc/vc: [B,H,N,D] bf16（K 块均值 / V 块和），routes: [B,H,QN,N] u8。
//   原生多 head：网格按 headQ 展开，一次启动。
// ──────────────────────────────────────────────────────────────────────────────
extern "C" ESIMD_KERNEL_API void sol_attn_forward(
    void* sycl_queue_ptr,
    void* q, void* k, void* v,
    void* kc, void* vc, void* exact_idx, void* approx_idx,
    void* n_exact, void* n_approx, void* out,
    int q_len, int kv_len, int heads, int n_blocks, int q_blocks, float scale) {
  sycl::queue& queue = *reinterpret_cast<sycl::queue*>(sycl_queue_ptr);
  for (int h = 0; h < heads; ++h) {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sol_attn::sol_attn_ndr(q_len, 1),
          [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            sol_attn::solAttnDg2<bf16, sol_attn::SA_HD>(
                static_cast<uint8_t*>(q), static_cast<uint8_t*>(k),
                static_cast<uint8_t*>(v), static_cast<uint8_t*>(kc),
                static_cast<uint8_t*>(vc),
                static_cast<uint8_t*>(exact_idx),
                static_cast<uint8_t*>(approx_idx),
                static_cast<uint8_t*>(n_exact),
                static_cast<uint8_t*>(n_approx),
                static_cast<uint8_t*>(out),
                static_cast<uint32_t>(q_len), static_cast<uint32_t>(kv_len),
                static_cast<uint32_t>(heads),
                static_cast<uint32_t>(n_blocks),
                static_cast<uint32_t>(q_blocks),
                static_cast<uint32_t>(h), scale, ndi);
          });
    }).wait();
  }
}


// 诊断探针：两段半块搬入（模拟精确分支），读回 row0/row31/row32/row63
extern "C" ESIMD_KERNEL_API void sol_attn_slm_probe2(
    void* sycl_queue_ptr, void* k, void* out, int heads) {
  sycl::queue& queue = *reinterpret_cast<sycl::queue*>(sycl_queue_ptr);
  constexpr int HD = sol_attn::SA_HD;
  constexpr int SLM_BN = 32;
  constexpr int SLM_K_BYTES = SLM_BN * HD * 2;
  constexpr int SLM_TOTAL = 2 * SLM_K_BYTES;
  constexpr int WG = 32;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<2>({(size_t)WG, 1}, {(size_t)WG, 1}),
        [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
          const int lid = static_cast<int>(ndi.get_local_id(0));
          const size_t rowStride = static_cast<size_t>(heads) * HD;
          const bf16* kBase = reinterpret_cast<const bf16*>(k);
          slm_init(SLM_TOTAL);
          for (int half = 0; half < 2; ++half) {
            const int hstart = half * SLM_BN;
            const int validElems = SLM_BN * HD;
            for (int i = lid * 16; i < SLM_BN * HD; i += WG * 16) {
              const int row = hstart + i / HD;
              const int col = i % HD;
              simd<bf16, 16> kchunk = block_load<bf16, 16>(
                  kBase + static_cast<size_t>(row) * rowStride + col,
                  overaligned_tag<16>{});
              slm_block_store(i * 2, kchunk, overaligned_tag<16>{});
            }
            barrier();
            if (lid == 0) {
              bf16* op = reinterpret_cast<bf16*>(out);
              for (int rr = 0; rr < 32; rr += 31) {
                simd<bf16, 16> r = slm_block_load<bf16, 16>(
                    (rr * HD) * 2, overaligned_tag<16>{});
                const int slot = half * 2 + (rr == 0 ? 0 : 1);
                block_store(op + slot * 32, r);
                simd<bf16, 16> r2 = slm_block_load<bf16, 16>(
                    (rr * HD + 16) * 2, overaligned_tag<16>{});
                block_store(op + slot * 32 + 16, r2);
              }
            }
            barrier();
          }
        });
  }).wait();
}
