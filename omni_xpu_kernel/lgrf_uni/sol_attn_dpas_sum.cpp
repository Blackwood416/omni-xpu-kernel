// Sol-Attn A770 DPAS summary kernel — bf16 BTHD D128 sparse attention。
//
// 结构完全复刻已验证的 DG2 DPAS4 模板（flash.attn.b.mha.dg2.dpas4.h）：
//   * DPAS ExecutionSize=8 / N=8 / fp32 累加（ExecutionSize=16 在 A770 数值错误）
//   * VNNI 打包布局与 packQDg2/packKvDg2 逐字节一致
//   * WG=32、Q_TILE=256（RPT=8）、doubleGRF；任何 spill 都可能 DEVICE_LOST
//   * scores 就地存 bf16 pChunk（与 dpas4 相同），单遍 QK
//
// 本文件只做"近似摘要阶段"：scores = Q@Kc^T（DPAS）→ 行在线 softmax（fp32）→
// acc = P@Vc（DPAS，Vc 为块和）。输出每行 m/l/acc（fp32/bf16），供精确阶段
// 作为在线 softmax 初始值。精确块（routes 掩码=1）在摘要里保留真实分数参与
// m（保证全局 max 一致），但 p 置零不贡献 l/acc——精确阶段再补算一次，
// 避免双重计数。kc/vc 由宿主预先 padding 到 32/64 行整倍数
// （零行），pack kernel 无运行时分支。

#include <cstdint>
#include <type_traits>

#include <sycl/sycl.hpp>
#include <sycl/ext/intel/esimd.hpp>

#include "esimd_kernel_api.h"

using bf16 = sycl::ext::oneapi::bfloat16;

#define __ESIMD_NS  sycl::ext::intel::esimd
#define __ESIMD_ENS sycl::ext::intel::experimental::esimd
#undef ESIMD_INLINE
#define ESIMD_INLINE inline __attribute__((always_inline))

using namespace sycl::ext::intel::esimd;
using namespace sycl::ext::intel::esimd::xmx;
using namespace sycl::ext::intel::experimental::esimd;

namespace sol_sum {

constexpr int WG = 32;
constexpr int RPT = 8;        // query rows per thread (Q_TILE=256)
constexpr int QGRP = WG * RPT;
constexpr int DCHUNKS = 8;    // 128 / 16
constexpr int NCHUNK = 32;    // summary n-columns per online chunk
constexpr int PGCH = NCHUNK / 16;  // P 16-row groups per chunk
constexpr float SCALE = 0.08838834764831844f;   // 1/sqrt(128)
constexpr float LOG2E = 1.4426950408889634f;
constexpr float FP32_MIN = -3.402823466e+38f;

// Full 8x8 DPAS with 8 distinct rows in A (K=16 per instruction).
template <typename ElemT>
ESIMD_INLINE simd<float, 64> dpas8x8(const simd<float, 64>& c,
                                     const simd<uint32_t, 64>& braw,
                                     const simd<ElemT, 128>& a) {
  simd<uint32_t, 64> brawLocal = braw;
  simd<ElemT, 128> b = brawLocal.template bit_cast_view<ElemT>();
  return dpas<8, 8, float>(c, b, a);
}

// ── pack Q：BTHD [1,T,H,128] → packedQ [head][qTile][thread][c][r][16] bf16 ──
// 与 DG2 DPAS4 的 packQDg2（RPT=8, DCHUNKS=8, BTHD）逐字节一致。
ESIMD_INLINE void packQDg2(
    uint8_t* packedQ, const uint8_t* qState,
    uint32_t qLen, uint32_t headQ, sycl::nd_item<1>& ndi) {
  constexpr int R = RPT;
  constexpr int DC = DCHUNKS;
  const int gid = static_cast<int>(ndi.get_group_linear_id());
  const int lid = static_cast<int>(ndi.get_local_linear_id());
  const int qTiles = (static_cast<int>(qLen) + QGRP - 1) / QGRP;
  const int headIdx = gid / qTiles;
  const int qTile = gid % qTiles;
  const int qrowBase = qTile * QGRP + lid * R;
  const size_t threadOff =
      ((static_cast<size_t>(headIdx) * qTiles + qTile) * WG + lid) * R * 128;

  simd<bf16, R * 128> qrows = 0;
#pragma unroll
  for (int r = 0; r < R; r++) {
    const int qrow = qrowBase + r;
    if (qrow < static_cast<int>(qLen)) {
      const bf16* qp = reinterpret_cast<const bf16*>(qState) +
                       (static_cast<size_t>(qrow) * headQ + headIdx) * 128;
#pragma unroll
      for (int c = 0; c < DC; c++) {
        qrows.template select<16, 1>(r * 128 + c * 16) =
            block_load<bf16, 16>(qp + c * 16, overaligned_tag<16>{});
      }
    }
  }
#pragma unroll
  for (int c = 0; c < DC; c++) {
    simd<bf16, 128> block = 0;
#pragma unroll
    for (int r = 0; r < R; r++) {
      block.template select<16, 1>(r * 16) =
          qrows.template select<16, 1>(r * 128 + c * 16);
    }
    block_store<bf16, 128>(reinterpret_cast<bf16*>(packedQ) + threadOff + c * 128,
                           block, overaligned_tag<16>{});
  }
}

// ── pack Kc：kc [1,H,N,128] → packedKc [head][g][c][64] uint32（VNNI）──
// 每 8 个 kc 行一组（g），组内 8 个 d-chunk；word[dp*8+r] = pack(kc[g*8+r][c*16+2dp],
// kc[g*8+r][c*16+2dp+1])。与 DPAS4 packKvDg2 的 K pack 布局一致。
// 宿主把 kc padding 到 32 行整倍数（nGroups = nPad/8 为 4 的倍数）。
ESIMD_INLINE void packKcDg2(
    uint8_t* packedKc, const uint8_t* kcState,
    uint32_t headIdx, uint32_t nGroups, uint32_t g,
    uint32_t c, sycl::nd_item<1>& ndi) {
  const bf16* kcBase = reinterpret_cast<const bf16*>(kcState) +
                       static_cast<size_t>(headIdx) * nGroups * 8 * 128;
  simd<bf16, 128> rows = 0;
#pragma unroll
  for (int r = 0; r < 8; r++) {
    const int n = static_cast<int>(g) * 8 + r;
    rows.template select<16, 1>(r * 16) = block_load<bf16, 16>(
        kcBase + static_cast<size_t>(n) * 128 + c * 16,
        overaligned_tag<16>{});
  }
  simd<uint32_t, 64> words;
#pragma unroll
  for (int dp = 0; dp < 8; dp++) {
#pragma unroll
    for (int r = 0; r < 8; r++) {
      const bf16 loVal = static_cast<bf16>(rows[r * 16 + 2 * dp]);
      const bf16 hiVal = static_cast<bf16>(rows[r * 16 + 2 * dp + 1]);
      const uint32_t lo = static_cast<uint32_t>(sycl::bit_cast<uint16_t>(loVal));
      const uint32_t hi = static_cast<uint32_t>(sycl::bit_cast<uint16_t>(hiVal));
      words[dp * 8 + r] = lo | (hi << 16);
    }
  }
  const size_t off = static_cast<size_t>(headIdx) * nGroups * 8 * 64 +
                     (static_cast<size_t>(g) * DCHUNKS + c) * 64;
  block_store<uint32_t, 64>(reinterpret_cast<uint32_t*>(packedKc) + off,
                            words, overaligned_tag<16>{});
}

// ── pack Vc：vc [1,H,N,128] → packedVc [head][p][c][h][64] uint32（VNNI）──
// 每 16 个 vc 行一组（p），组内 8 对行 × 8 d-chunk × 2 d-half；与 DPAS4
// packKvDg2 的 V pack 布局一致。宿主把 vc padding 到 64 行整倍数
// （pgTotal = nPad/16 为 4 的倍数）。
ESIMD_INLINE void packVcDg2(
    uint8_t* packedVc, const uint8_t* vcState,
    uint32_t headIdx, uint32_t pgTotal, uint32_t p,
    uint32_t pair8, sycl::nd_item<1>& ndi) {
  const bf16* vcBase = reinterpret_cast<const bf16*>(vcState) +
                       static_cast<size_t>(headIdx) * pgTotal * 16 * 128;
  const int row0 = static_cast<int>(p) * 16 + static_cast<int>(pair8) * 2;
  const int row1 = row0 + 1;
#pragma unroll
  for (int c = 0; c < DCHUNKS; c++) {
    simd<bf16, 16> lo = block_load<bf16, 16>(
        vcBase + static_cast<size_t>(row0) * 128 + c * 16,
        overaligned_tag<16>{});
    simd<bf16, 16> hi = block_load<bf16, 16>(
        vcBase + static_cast<size_t>(row1) * 128 + c * 16,
        overaligned_tag<16>{});
#pragma unroll
    for (int h = 0; h < 2; h++) {
      simd<bf16, 16> loHalf = 0;
      simd<bf16, 16> hiHalf = 0;
      for (int i = 0; i < 8; i++) {
        loHalf[i] = lo[8 * h + i];
        hiHalf[i] = hi[8 * h + i];
      }
      simd<uint32_t, 8> w;
#pragma unroll
      for (int i = 0; i < 8; i++) {
        const uint32_t a = static_cast<uint32_t>(
            sycl::bit_cast<uint16_t>(static_cast<bf16>(loHalf[i])));
        const uint32_t b = static_cast<uint32_t>(
            sycl::bit_cast<uint16_t>(static_cast<bf16>(hiHalf[i])));
        w[i] = a | (b << 16);
      }
      const size_t off = static_cast<size_t>(headIdx) * pgTotal * 8 * 2 * 64 +
                         ((static_cast<size_t>(p) * DCHUNKS + c) * 2 + h) * 64 +
                         static_cast<size_t>(pair8) * 8;
      block_store<uint32_t, 8>(reinterpret_cast<uint32_t*>(packedVc) + off,
                               w, overaligned_tag<16>{});
    }
  }
}

// ── summary：per (head, qTile 256 行) 全块在线 softmax 摘要 ──
// 输入 colmean（查询块质心分数 [H,QN,N] f32，log2 域，来自 prepare 的
// score 矩阵；本 kernel 转线性域）、packedVc（已打包、N 已 padding）。
// 每行共享其查询块的质心分数（kitchen centroid-tail：routing 64x 缩小，
// 精度代价 ~5e-4 cosine）。
// 输出：m/l fp32 [T,H]；acc bf16 [T,H,128]（在线 softmax 初始值）。
ESIMD_INLINE void summaryDg2(
    const float* colmean, const uint8_t* packedVc,
    const uint8_t* routes,
    float* outM, float* outL, bf16* outAcc,
    uint32_t qLen, uint32_t kvLen, uint32_t headQ, uint32_t nBlocks,
    uint32_t qBlocks, uint32_t nPad, sycl::nd_item<1>& ndi) {
  const int gid = static_cast<int>(ndi.get_group_linear_id());
  const int lid = static_cast<int>(ndi.get_local_linear_id());
  const int qTiles = (static_cast<int>(qLen) + QGRP - 1) / QGRP;
  const int headIdx = gid / qTiles;
  const int qTile = gid % qTiles;
  const int qrowBase = qTile * QGRP + lid * RPT;
  const int pgTotal = static_cast<int>(nPad) / 16;
  const int nChunks = (static_cast<int>(nBlocks) + NCHUNK - 1) / NCHUNK;
  // 每线程 8 行落在同一个 64 行 qblock 内
  const int qblock = qTile * 4 + lid / RPT;

  float mArr[RPT];
  float lArr[RPT];
  simd<float, RPT * 128> acc = 0;
  simd<bf16, RPT * NCHUNK> pChunkAll = 0;
#pragma unroll
  for (int r = 0; r < RPT; r++) {
    mArr[r] = FP32_MIN;
    lArr[r] = 0.0f;
  }

  for (int nc = 0; nc < nChunks; nc++) {
    const int nBase = nc * NCHUNK;
    // chunk 级块长（len 加权；padding 列 0）
    simd<float, NCHUNK> lens;
    for (int i = 0; i < NCHUNK; i++) {
      const int n = nBase + i;
      const int remain = static_cast<int>(kvLen) - n * 64;
      const float len = remain > 64 ? 64.0f : (remain > 0 ? (float)remain : 0.0f);
      lens[i] = (n >= static_cast<int>(nBlocks)) ? 0.0f : len;
    }

    // ── 质心分数：查询块 colmean（log2 域）转线性域写 pChunk（每行共享）──
    {
      const float* cp = colmean +
          (static_cast<size_t>(headIdx) * qBlocks + qblock) * nPad + nBase;
      simd<float, NCHUNK> svec_c = block_load<float, NCHUNK>(
          cp, overaligned_tag<16>{}) * (1.0f / LOG2E);
      simd_mask<NCHUNK> cmask;
      for (int i = 0; i < NCHUNK; i++) {
        cmask[i] = (nBase + i) >= static_cast<int>(nBlocks);
      }
      svec_c.merge(-1.0e30f, cmask);
      simd<bf16, NCHUNK> cph = convert<bf16>(svec_c);
#pragma unroll
      for (int r = 0; r < RPT; r++) {
        for (int p = 0; p < PGCH; p++) {
          pChunkAll.template select<16, 1>(p * RPT * 16 + r * 16) =
              cph.template select<16, 1>(p * 16);
        }
      }
    }

    // ── 行在线 softmax（就地转 p；len 加权；acc 按 rescalePrev 缩放）──
#pragma unroll
    for (int r = 0; r < RPT; r++) {
      simd<float, NCHUNK> svec = 0;
#pragma unroll
      for (int p = 0; p < PGCH; p++) {
        simd<bf16, 16> sh =
            pChunkAll.template select<16, 1>(p * RPT * 16 + r * 16);
        svec.template select<16, 1>(p * 16) = convert<float>(sh);
      }
      const float mTile = __ESIMD_NS::detail::reduce<
          float, float, NCHUNK, __ESIMD_NS::detail::esimd_apply_reduced_max>(
          svec);
      const float oldM = mArr[r];
      float rescalePrev = 1.0f;
      float rescaleCur = 1.0f;
      if (mTile > oldM) {
        rescalePrev = __ESIMD_NS::exp2<float, 1>(
            simd<float, 1>((oldM - mTile) * LOG2E))[0];
      } else if (mTile < oldM) {
        rescaleCur = __ESIMD_NS::exp2<float, 1>(
            simd<float, 1>((mTile - oldM) * LOG2E))[0];
      }
      mArr[r] = mTile > oldM ? mTile : oldM;
      lArr[r] = lArr[r] * rescalePrev;
      acc.template select<128, 1>(r * 128) =
          acc.template select<128, 1>(r * 128) * rescalePrev;
      svec = (svec - mTile) * LOG2E;
      simd<float, NCHUNK> pvec = __ESIMD_NS::exp2<float, NCHUNK>(svec);
      // 精确块（routes=1）不贡献 l/acc（m 已含其分数）
      {
        const uint8_t* rp = routes +
            (static_cast<size_t>(headIdx) * qBlocks + qblock) * nPad + nBase;
        simd<uint8_t, NCHUNK> emask = block_load<uint8_t, NCHUNK>(
            rp, overaligned_tag<16>{});
        simd_mask<NCHUNK> exact = emask != static_cast<uint8_t>(0);
        pvec.merge(0.0f, exact);
      }
      const float lTile = __ESIMD_NS::detail::reduce<
          float, float, NCHUNK, __ESIMD_NS::detail::esimd_apply_sum>(
          pvec * lens);
      lArr[r] = lArr[r] + lTile * rescaleCur;
#pragma unroll
      for (int p = 0; p < PGCH; p++) {
        simd<float, 16> p16 = pvec.template select<16, 1>(p * 16);
        simd<bf16, 16> ph = convert<bf16>(p16 * rescaleCur);
        pChunkAll.template select<16, 1>(p * RPT * 16 + r * 16) = ph;
      }
    }

    // ── S*V：acc += P @ Vc（32 n × 128 d）──
#pragma unroll
    for (int c = 0; c < DCHUNKS; c++) {
      simd<float, 64> accH0 = 0;
      simd<float, 64> accH1 = 0;
#pragma unroll
      for (int r = 0; r < RPT; r++) {
        accH0.template select<8, 1>(r * 8) =
            acc.template select<8, 1>(r * 128 + c * 16);
        accH1.template select<8, 1>(r * 8) =
            acc.template select<8, 1>(r * 128 + c * 16 + 8);
      }
#pragma unroll
      for (int p = 0; p < PGCH; p++) {
        simd<bf16, 128> a;
        a.template select<RPT * 16, 1>(0) =
            pChunkAll.template select<RPT * 16, 1>(p * RPT * 16);
        const int pGlobal = nc * PGCH + p;
        const size_t bOff =
            (static_cast<size_t>(headIdx) * pgTotal + pGlobal) * 8 * 2 * 64 +
            c * 2 * 64;
        simd<uint32_t, 64> b0 = block_load<uint32_t, 64>(
            reinterpret_cast<const uint32_t*>(packedVc) + bOff,
            overaligned_tag<16>{});
        simd<uint32_t, 64> b1 = block_load<uint32_t, 64>(
            reinterpret_cast<const uint32_t*>(packedVc) + bOff + 64,
            overaligned_tag<16>{});
        accH0 = dpas8x8<bf16>(accH0, b0, a);
        accH1 = dpas8x8<bf16>(accH1, b1, a);
      }
#pragma unroll
      for (int r = 0; r < RPT; r++) {
        acc.template select<8, 1>(r * 128 + c * 16) =
            accH0.template select<8, 1>(r * 8);
        acc.template select<8, 1>(r * 128 + c * 16 + 8) =
            accH1.template select<8, 1>(r * 8);
      }
    }
  }

  // ── 输出：m/l fp32 [T,H]；acc bf16 [T,H,128] ──
#pragma unroll
  for (int r = 0; r < RPT; r++) {
    const int qrow = qrowBase + r;
    if (qrow < static_cast<int>(qLen)) {
      const size_t rowOff = static_cast<size_t>(qrow) * headQ + headIdx;
      outM[rowOff] = mArr[r];
      outL[rowOff] = lArr[r];
      bf16* op = outAcc + rowOff * 128;
#pragma unroll
      for (int c = 0; c < DCHUNKS; c++) {
        simd<float, 16> outv = acc.template select<16, 1>(r * 128 + c * 16);
        simd<bf16, 16> o16;
        bf16 arr[16];
        float farr[16];
        outv.copy_to(farr);
        for (int i = 0; i < 16; i++) {
          arr[i] = static_cast<bf16>(farr[i]);
        }
        o16.copy_from(arr);
        block_store(op + c * 16, o16);
      }
    }
  }
}

}  // namespace sol_sum

// ──────────────────────────────────────────────────────────────────────────────
// 导出入口
// ──────────────────────────────────────────────────────────────────────────────
extern "C" ESIMD_KERNEL_API void sol_attn_pack_q(
    void* sycl_queue_ptr, void* q, void* packed_q, int q_len, int heads) {
  sycl::queue& queue = *reinterpret_cast<sycl::queue*>(sycl_queue_ptr);
  const int qTiles = (q_len + sol_sum::QGRP - 1) / sol_sum::QGRP;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            {(size_t)(sol_sum::WG * heads * qTiles)}, {(size_t)sol_sum::WG}),
        [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
          sol_sum::packQDg2(static_cast<uint8_t*>(packed_q),
                            static_cast<const uint8_t*>(q),
                            static_cast<uint32_t>(q_len),
                            static_cast<uint32_t>(heads), ndi);
        });
  }).wait();
}

extern "C" ESIMD_KERNEL_API void sol_attn_pack_kc(
    void* sycl_queue_ptr, void* kc, void* packed_kc,
    int heads, int n_groups_pad) {
  sycl::queue& queue = *reinterpret_cast<sycl::queue*>(sycl_queue_ptr);
  // n_groups_pad 必须是 4 的倍数（宿主把 kc padding 到 32 行整倍数）。
  const int wgsPerHead = n_groups_pad / 4;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>({(size_t)(sol_sum::WG * heads * wgsPerHead)},
                          {(size_t)sol_sum::WG}),
        [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
          const int gid = static_cast<int>(ndi.get_group_linear_id());
          const int lid = static_cast<int>(ndi.get_local_linear_id());
          const int headIdx = gid / wgsPerHead;
          const int g = (gid % wgsPerHead) * 4 + lid / 8;
          const int c = lid % 8;
          sol_sum::packKcDg2(static_cast<uint8_t*>(packed_kc),
                             static_cast<const uint8_t*>(kc),
                             static_cast<uint32_t>(headIdx),
                             static_cast<uint32_t>(n_groups_pad),
                             static_cast<uint32_t>(g),
                             static_cast<uint32_t>(c), ndi);
        });
  }).wait();
}

extern "C" ESIMD_KERNEL_API void sol_attn_pack_vc(
    void* sycl_queue_ptr, void* vc, void* packed_vc,
    int heads, int pg_pad) {
  sycl::queue& queue = *reinterpret_cast<sycl::queue*>(sycl_queue_ptr);
  // pg_pad 必须是 4 的倍数（宿主把 vc padding 到 64 行整倍数）。
  const int wgsPerHead = pg_pad / 4;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>({(size_t)(sol_sum::WG * heads * wgsPerHead)},
                          {(size_t)sol_sum::WG}),
        [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
          const int gid = static_cast<int>(ndi.get_group_linear_id());
          const int lid = static_cast<int>(ndi.get_local_linear_id());
          const int headIdx = gid / wgsPerHead;
          const int p = (gid % wgsPerHead) * 4 + lid / 8;
          const int pair8 = lid % 8;
          sol_sum::packVcDg2(static_cast<uint8_t*>(packed_vc),
                             static_cast<const uint8_t*>(vc),
                             static_cast<uint32_t>(headIdx),
                             static_cast<uint32_t>(pg_pad),
                             static_cast<uint32_t>(p),
                             static_cast<uint32_t>(pair8), ndi);
        });
  }).wait();
}

extern "C" ESIMD_KERNEL_API void sol_attn_summary(
    void* sycl_queue_ptr,
    void* colmean, void* packed_vc,
    void* routes,
    void* out_m, void* out_l, void* out_acc,
    int q_len, int kv_len, int heads, int n_blocks, int q_blocks, int n_pad) {
  sycl::queue& queue = *reinterpret_cast<sycl::queue*>(sycl_queue_ptr);
  const int qTiles = (q_len + sol_sum::QGRP - 1) / sol_sum::QGRP;
  queue.submit([&](sycl::handler& cgh) {
    cgh.parallel_for(
        sycl::nd_range<1>(
            {(size_t)(sol_sum::WG * heads * qTiles)}, {(size_t)sol_sum::WG}),
        [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
          sol_sum::summaryDg2(
              static_cast<const float*>(colmean),
              static_cast<const uint8_t*>(packed_vc),
              static_cast<const uint8_t*>(routes),
              static_cast<float*>(out_m), static_cast<float*>(out_l),
              static_cast<bf16*>(out_acc),
              static_cast<uint32_t>(q_len), static_cast<uint32_t>(kv_len),
              static_cast<uint32_t>(heads),
              static_cast<uint32_t>(n_blocks),
              static_cast<uint32_t>(q_blocks),
              static_cast<uint32_t>(n_pad), ndi);
        });
  }).wait();
}
