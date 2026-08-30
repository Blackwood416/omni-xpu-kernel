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

constexpr int SA_BM = 64;    // query rows per work-group（每线程 2 行，K/V SLM 复用减半）
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
    uint8_t* exactIdx, uint8_t* approxIdx, uint8_t* nExact, uint8_t* nApprox,
    const float* initM, const float* initL, const bf16* initAcc,
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
  const int qrowA = qTile * BM + lid * 2;
  const int qrowB = qrowA + 1;
  const int qblock = qTile;  // 64 query rows = 1 route block
  const size_t rowStride = static_cast<size_t>(headQ) * HD;
  const size_t headOffset = static_cast<size_t>(headIdx) * nBlocks * HD;

  const ElemT* q = reinterpret_cast<const ElemT*>(qState);
  const ElemT* kBase = reinterpret_cast<const ElemT*>(kState);
  const ElemT* vBase = reinterpret_cast<const ElemT*>(vState);
  const int32_t* exactList =
      reinterpret_cast<const int32_t*>(exactIdx) +
      static_cast<size_t>(headIdx) * qBlocks * nBlocks +
      static_cast<size_t>(qblock) * nBlocks;
  const int32_t* nExactPtr = reinterpret_cast<const int32_t*>(nExact) +
                             static_cast<size_t>(headIdx) * qBlocks +
                             static_cast<size_t>(qblock);
  const int nx = nExactPtr[0];

  slm_init(SLM_TOTAL_BYTES);

  simd<float, HD> qRegA = 0;
  simd<float, HD> qRegB = 0;
  if (qrowA < static_cast<int>(qLen)) {
    const ElemT* qp = q + static_cast<size_t>(qrowA) * rowStride +
                      static_cast<size_t>(headIdx) * HD;
#pragma unroll
    for (int c = 0; c < HD / 16; ++c) {
      simd<float, 16> chunk = block_load<ElemT, 16>(
          qp + c * 16, overaligned_tag<16>{});
      qRegA.template select<16, 1>(c * 16) = chunk;
    }
  }
  if (qrowB < static_cast<int>(qLen)) {
    const ElemT* qpb = q + static_cast<size_t>(qrowB) * rowStride +
                       static_cast<size_t>(headIdx) * HD;
#pragma unroll
    for (int c = 0; c < HD / 16; ++c) {
      simd<float, 16> chunk = block_load<ElemT, 16>(
          qpb + c * 16, overaligned_tag<16>{});
      qRegB.template select<16, 1>(c * 16) = chunk;
    }
  }

  simd<float, HD> accA = 0;
  simd<float, HD> accB = 0;
  float mA = -3.402823466e+38f;
  float mB = -3.402823466e+38f;
  float lA = 0.0f;
  float lB = 0.0f;
  if (qrowA < static_cast<int>(qLen) && initM != nullptr) {
    const size_t rowOffA = static_cast<size_t>(qrowA) * headQ + headIdx;
    mA = initM[rowOffA];
    lA = initL[rowOffA];
    const bf16* ia = initAcc + rowOffA * HD;
#pragma unroll
    for (int c = 0; c < HD / 16; ++c) {
      accA.template select<16, 1>(c * 16) =
          block_load<bf16, 16>(ia + c * 16, overaligned_tag<16>{});
    }
  }
  if (qrowB < static_cast<int>(qLen) && initM != nullptr) {
    const size_t rowOffB = static_cast<size_t>(qrowB) * headQ + headIdx;
    mB = initM[rowOffB];
    lB = initL[rowOffB];
    const bf16* ib = initAcc + rowOffB * HD;
#pragma unroll
    for (int c = 0; c < HD / 16; ++c) {
      accB.template select<16, 1>(c * 16) =
          block_load<bf16, 16>(ib + c * 16, overaligned_tag<16>{});
    }
  }

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
      // 固定 64 次循环 + 越界掩码 + 部分展开；每线程 2 行共用 K/V SLM 读取，
      // 两个 softmax 链交错隐藏延迟。SLM 无效行已零填充；j>=klen 掩到 -1e30。
#pragma unroll 2
      for (int j = 0; j < BN; ++j) {
        simd<float, HD> dotvA = 0;
        simd<float, HD> dotvB = 0;
#pragma unroll
        for (int c = 0; c < HD / 64; ++c) {
          simd<float, 64> kf = convert<float>(slm_block_load<ElemT, 64>(
              (j * HD + c * 64) * static_cast<int>(sizeof(ElemT)),
              overaligned_tag<16>{}));
          dotvA.template select<64, 1>(c * 64) =
              dotvA.template select<64, 1>(c * 64) +
              qRegA.template select<64, 1>(c * 64) * kf;
          dotvB.template select<64, 1>(c * 64) =
              dotvB.template select<64, 1>(c * 64) +
              qRegB.template select<64, 1>(c * 64) * kf;
        }
        float sA = dg2_sum<HD>(dotvA) * SCALE;
        float sB = dg2_sum<HD>(dotvB) * SCALE;
        if (j >= klen) {
          sA = -1.0e30f;
          sB = -1.0e30f;
        }
        const float mNewA = (sA > mA) ? sA : mA;
        const float rescaleA = sycl::exp2((mA - mNewA) * LOG2E);
        const float pA = sycl::exp2((sA - mNewA) * LOG2E);
        const float mNewB = (sB > mB) ? sB : mB;
        const float rescaleB = sycl::exp2((mB - mNewB) * LOG2E);
        const float pB = sycl::exp2((sB - mNewB) * LOG2E);
        lA = lA * rescaleA + pA;
        lB = lB * rescaleB + pB;
#pragma unroll
        for (int c = 0; c < HD / 64; ++c) {
          simd<float, 64> vf = convert<float>(slm_block_load<ElemT, 64>(
              SLM_K_BYTES + (j * HD + c * 64) *
                  static_cast<int>(sizeof(ElemT)),
              overaligned_tag<16>{}));
          accA.template select<64, 1>(c * 64) =
              accA.template select<64, 1>(c * 64) * rescaleA + pA * vf;
          accB.template select<64, 1>(c * 64) =
              accB.template select<64, 1>(c * 64) * rescaleB + pB * vf;
        }
        mA = mNewA;
        mB = mNewB;
      }
      barrier();
    }
  }
  // ── 输出归一化 ──
  for (int rr = 0; rr < 2; ++rr) {
    const int qrow = (rr == 0) ? qrowA : qrowB;
    const simd<float, HD>& accR = (rr == 0) ? accA : accB;
    const float lR = (rr == 0) ? lA : lB;
    if (qrow < static_cast<int>(qLen) && lR > 0.0f) {
      simd<float, HD> outv = accR * (1.0f / lR);
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
}

// ── 4 行/线程 + union 精确列表版本（128 行/WG = 2 qblock）──
// 每线程 4 行同属一个 qblock（lid*4..+3 → qblock = qTile*2 + lid/16），
// K/V SLM 读取降为 1/4；WG 遍历宿主预计算的 union 精确块列表，
// 非本 qblock 精确的块按 routes 掩码整块跳过（分数 -1e30）。
template <typename ElemT, int HD>
ESIMD_INLINE void solAttnDg2Union4(
    uint8_t* qState, uint8_t* kState, uint8_t* vState,
    const uint8_t* routes, const int32_t* unionIdx, const int32_t* nUnion,
    const float* initM, const float* initL, const bf16* initAcc,
    uint8_t* outState,
    uint32_t qLen, uint32_t kvLen, uint32_t headQ, uint32_t qBlocks,
    uint32_t nPad, sycl::nd_item<1>& ndi) {
  constexpr int BM4 = 128;  // query rows per work-group
  constexpr int BN = SA_BN;
  constexpr int WG = SA_WG;
  constexpr int SLM_K_BYTES = BN * HD * static_cast<int>(sizeof(ElemT));
  constexpr int SLM_TOTAL_BYTES = 2 * SLM_K_BYTES;
  constexpr float SCALE = 0.08838834764831844f;
  constexpr float LOG2E = 1.4426950408889634f;

  const int gid = static_cast<int>(ndi.get_group_linear_id());
  const int lid = static_cast<int>(ndi.get_local_linear_id());
  const int qTiles4 = (static_cast<int>(qLen) + BM4 - 1) / BM4;
  const int headIdx = gid / qTiles4;
  const int qTile = gid % qTiles4;
  const int qrowA = qTile * BM4 + lid * 4;
  const int qrowB = qrowA + 1;
  const int qrowC = qrowA + 2;
  const int qrowD = qrowA + 3;
  const int qblock = qTile * 2 + lid / 16;
  const size_t rowStride = static_cast<size_t>(headQ) * HD;

  const ElemT* q = reinterpret_cast<const ElemT*>(qState);
  const ElemT* kBase = reinterpret_cast<const ElemT*>(kState);
  const ElemT* vBase = reinterpret_cast<const ElemT*>(vState);
  const int32_t* ulist =
      reinterpret_cast<const int32_t*>(unionIdx) +
      (static_cast<size_t>(headIdx) * qTiles4 + qTile) * 192;
  const int nu = nUnion[static_cast<size_t>(headIdx) * qTiles4 + qTile];

  slm_init(SLM_TOTAL_BYTES);

  simd<float, HD> qRegA = 0;
  simd<float, HD> qRegB = 0;
  simd<float, HD> qRegC = 0;
  simd<float, HD> qRegD = 0;
  if (qrowA < static_cast<int>(qLen)) {
    const ElemT* qp = q + static_cast<size_t>(qrowA) * rowStride +
                      static_cast<size_t>(headIdx) * HD;
#pragma unroll
    for (int c = 0; c < HD / 16; ++c) {
      qRegA.template select<16, 1>(c * 16) =
          block_load<ElemT, 16>(qp + c * 16, overaligned_tag<16>{});
    }
  }
  if (qrowB < static_cast<int>(qLen)) {
    const ElemT* qp = q + static_cast<size_t>(qrowB) * rowStride +
                      static_cast<size_t>(headIdx) * HD;
#pragma unroll
    for (int c = 0; c < HD / 16; ++c) {
      qRegB.template select<16, 1>(c * 16) =
          block_load<ElemT, 16>(qp + c * 16, overaligned_tag<16>{});
    }
  }
  if (qrowC < static_cast<int>(qLen)) {
    const ElemT* qp = q + static_cast<size_t>(qrowC) * rowStride +
                      static_cast<size_t>(headIdx) * HD;
#pragma unroll
    for (int c = 0; c < HD / 16; ++c) {
      qRegC.template select<16, 1>(c * 16) =
          block_load<ElemT, 16>(qp + c * 16, overaligned_tag<16>{});
    }
  }
  if (qrowD < static_cast<int>(qLen)) {
    const ElemT* qp = q + static_cast<size_t>(qrowD) * rowStride +
                      static_cast<size_t>(headIdx) * HD;
#pragma unroll
    for (int c = 0; c < HD / 16; ++c) {
      qRegD.template select<16, 1>(c * 16) =
          block_load<ElemT, 16>(qp + c * 16, overaligned_tag<16>{});
    }
  }

  simd<float, HD> accA = 0;
  simd<float, HD> accB = 0;
  simd<float, HD> accC = 0;
  simd<float, HD> accD = 0;
  float mA = -3.402823466e+38f;
  float mB = -3.402823466e+38f;
  float mC = -3.402823466e+38f;
  float mD = -3.402823466e+38f;
  float lA = 0.0f;
  float lB = 0.0f;
  float lC = 0.0f;
  float lD = 0.0f;
  if (initM != nullptr) {
    if (qrowA < static_cast<int>(qLen)) {
      const size_t ro = static_cast<size_t>(qrowA) * headQ + headIdx;
      mA = initM[ro];
      lA = initL[ro];
      const bf16* ia = initAcc + ro * HD;
#pragma unroll
      for (int c = 0; c < HD / 16; ++c) {
        accA.template select<16, 1>(c * 16) =
            block_load<bf16, 16>(ia + c * 16, overaligned_tag<16>{});
      }
    }
    if (qrowB < static_cast<int>(qLen)) {
      const size_t ro = static_cast<size_t>(qrowB) * headQ + headIdx;
      mB = initM[ro];
      lB = initL[ro];
      const bf16* ia = initAcc + ro * HD;
#pragma unroll
      for (int c = 0; c < HD / 16; ++c) {
        accB.template select<16, 1>(c * 16) =
            block_load<bf16, 16>(ia + c * 16, overaligned_tag<16>{});
      }
    }
    if (qrowC < static_cast<int>(qLen)) {
      const size_t ro = static_cast<size_t>(qrowC) * headQ + headIdx;
      mC = initM[ro];
      lC = initL[ro];
      const bf16* ia = initAcc + ro * HD;
#pragma unroll
      for (int c = 0; c < HD / 16; ++c) {
        accC.template select<16, 1>(c * 16) =
            block_load<bf16, 16>(ia + c * 16, overaligned_tag<16>{});
      }
    }
    if (qrowD < static_cast<int>(qLen)) {
      const size_t ro = static_cast<size_t>(qrowD) * headQ + headIdx;
      mD = initM[ro];
      lD = initL[ro];
      const bf16* ia = initAcc + ro * HD;
#pragma unroll
      for (int c = 0; c < HD / 16; ++c) {
        accD.template select<16, 1>(c * 16) =
            block_load<bf16, 16>(ia + c * 16, overaligned_tag<16>{});
      }
    }
  }

  for (int ei = 0; ei < nu; ++ei) {
    const int kb = ulist[ei];
    {
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

      const float exf = static_cast<float>(routes[
          (static_cast<size_t>(headIdx) * qBlocks + qblock) * nPad + kb]);

#pragma unroll 2
      for (int j = 0; j < BN; ++j) {
        simd<float, HD> dotA = 0;
        simd<float, HD> dotB = 0;
        simd<float, HD> dotC = 0;
        simd<float, HD> dotD = 0;
#pragma unroll
        for (int c = 0; c < HD / 16; ++c) {
          simd<float, 16> kf = slm_block_load<ElemT, 16>(
              (j * HD + c * 16) * static_cast<int>(sizeof(ElemT)),
              overaligned_tag<16>{});
          dotA.template select<16, 1>(c * 16) =
              dotA.template select<16, 1>(c * 16) +
              qRegA.template select<16, 1>(c * 16) * kf;
          dotB.template select<16, 1>(c * 16) =
              dotB.template select<16, 1>(c * 16) +
              qRegB.template select<16, 1>(c * 16) * kf;
          dotC.template select<16, 1>(c * 16) =
              dotC.template select<16, 1>(c * 16) +
              qRegC.template select<16, 1>(c * 16) * kf;
          dotD.template select<16, 1>(c * 16) =
              dotD.template select<16, 1>(c * 16) +
              qRegD.template select<16, 1>(c * 16) * kf;
        }
        float sA = dg2_sum<HD>(dotA) * SCALE;
        float sB = dg2_sum<HD>(dotB) * SCALE;
        float sC = dg2_sum<HD>(dotC) * SCALE;
        float sD = dg2_sum<HD>(dotD) * SCALE;
        if (j >= klen) {
          sA = sB = sC = sD = -1.0e30f;
        }
        // 非本 qblock 精确：整块跳过（m/l/acc 均不参与）
        sA = sA * exf - (1.0f - exf) * 1.0e30f;
        sB = sB * exf - (1.0f - exf) * 1.0e30f;
        sC = sC * exf - (1.0f - exf) * 1.0e30f;
        sD = sD * exf - (1.0f - exf) * 1.0e30f;
        const float mNA = (sA > mA) ? sA : mA;
        const float rA = sycl::exp2((mA - mNA) * LOG2E);
        const float pA = sycl::exp2((sA - mNA) * LOG2E);
        const float mNB = (sB > mB) ? sB : mB;
        const float rB = sycl::exp2((mB - mNB) * LOG2E);
        const float pB = sycl::exp2((sB - mNB) * LOG2E);
        const float mNC = (sC > mC) ? sC : mC;
        const float rC = sycl::exp2((mC - mNC) * LOG2E);
        const float pC = sycl::exp2((sC - mNC) * LOG2E);
        const float mND = (sD > mD) ? sD : mD;
        const float rD = sycl::exp2((mD - mND) * LOG2E);
        const float pD = sycl::exp2((sD - mND) * LOG2E);
        lA = lA * rA + pA;
        lB = lB * rB + pB;
        lC = lC * rC + pC;
        lD = lD * rD + pD;
#pragma unroll
        for (int c = 0; c < HD / 16; ++c) {
          simd<float, 16> vf = slm_block_load<ElemT, 16>(
              SLM_K_BYTES + (j * HD + c * 16) *
                  static_cast<int>(sizeof(ElemT)),
              overaligned_tag<16>{});
          accA.template select<16, 1>(c * 16) =
              accA.template select<16, 1>(c * 16) * rA + pA * vf;
          accB.template select<16, 1>(c * 16) =
              accB.template select<16, 1>(c * 16) * rB + pB * vf;
          accC.template select<16, 1>(c * 16) =
              accC.template select<16, 1>(c * 16) * rC + pC * vf;
          accD.template select<16, 1>(c * 16) =
              accD.template select<16, 1>(c * 16) * rD + pD * vf;
        }
        mA = mNA;
        mB = mNB;
        mC = mNC;
        mD = mND;
      }
      barrier();
    }
  }

  // ── 输出归一化 ──
#pragma unroll
  for (int rr = 0; rr < 4; ++rr) {
    const int qrow = qrowA + rr;
    const float lR = (rr == 0) ? lA : (rr == 1) ? lB : (rr == 2) ? lC : lD;
    if (qrow < static_cast<int>(qLen) && lR > 0.0f) {
      simd<float, HD> outv = 0;
      if (rr == 0) {
        outv = accA * (1.0f / lA);
      } else if (rr == 1) {
        outv = accB * (1.0f / lB);
      } else if (rr == 2) {
        outv = accC * (1.0f / lC);
      } else {
        outv = accD * (1.0f / lD);
      }
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
}

}  // namespace sol_attn

// ──────────────────────────────────────────────────────────────────────────────
// sol_attn_exact: 精确阶段，以摘要阶段输出的 m/l/acc 为在线 softmax 初始值。
//   init_m/init_l: [T,H] f32；init_acc: [T,H,128] bf16（摘要输出）。
//   只处理精确块（CSR），不再内联近似路径。原生多 head：网格按 headQ 展开。
// ──────────────────────────────────────────────────────────────────────────────
extern "C" ESIMD_KERNEL_API void sol_attn_exact(
    void* sycl_queue_ptr,
    void* q, void* k, void* v,
    void* exact_idx, void* n_exact,
    void* init_m, void* init_l, void* init_acc, void* out,
    int q_len, int kv_len, int heads, int n_blocks, int q_blocks, float scale) {
  sycl::queue& queue = *reinterpret_cast<sycl::queue*>(sycl_queue_ptr);
  for (int h = 0; h < heads; ++h) {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sol_attn::sol_attn_ndr(q_len, 1),
          [=](sycl::nd_item<2> ndi) SYCL_ESIMD_KERNEL {
            sol_attn::solAttnDg2<bf16, sol_attn::SA_HD>(
                static_cast<uint8_t*>(q), static_cast<uint8_t*>(k),
                static_cast<uint8_t*>(v),
                static_cast<uint8_t*>(exact_idx),
                nullptr,
                static_cast<uint8_t*>(n_exact),
                nullptr,
                static_cast<const float*>(init_m),
                static_cast<const float*>(init_l),
                static_cast<const bf16*>(init_acc),
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

// ── union 版本：4 行/线程，128 行/WG（2 qblock），routes 掩码 ──
extern "C" ESIMD_KERNEL_API void sol_attn_exact_union(
    void* sycl_queue_ptr,
    void* q, void* k, void* v,
    void* routes, void* union_idx, void* n_union,
    void* init_m, void* init_l, void* init_acc, void* out,
    int q_len, int kv_len, int heads, int q_blocks, int n_pad) {
  sycl::queue& queue = *reinterpret_cast<sycl::queue*>(sycl_queue_ptr);
  const int qTiles4 = (q_len + 127) / 128;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            {(size_t)(sol_attn::SA_WG * heads * qTiles4)},
            {(size_t)sol_attn::SA_WG}),
        [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
          sol_attn::solAttnDg2Union4<bf16, sol_attn::SA_HD>(
              static_cast<uint8_t*>(q), static_cast<uint8_t*>(k),
              static_cast<uint8_t*>(v),
              static_cast<const uint8_t*>(routes),
              static_cast<const int32_t*>(union_idx),
              static_cast<const int32_t*>(n_union),
              static_cast<const float*>(init_m),
              static_cast<const float*>(init_l),
              static_cast<const bf16*>(init_acc),
              static_cast<uint8_t*>(out),
              static_cast<uint32_t>(q_len),
              static_cast<uint32_t>(kv_len),
              static_cast<uint32_t>(heads),
              static_cast<uint32_t>(q_blocks),
              static_cast<uint32_t>(n_pad), ndi);
        });
  }).wait();
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
