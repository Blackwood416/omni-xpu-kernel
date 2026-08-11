// DG2-native Flash Attention v4 — packed-KV DPAS design.
//
// v2 showed correct DPAS math but no speedup: each thread still serially
// visited every K/V row and replicated one query row across 8 DPAS rows
// (1/8 XMX utilization). v3 fixes both:
//
//   Kernel 1 (pack): one pass over K/V per head writes the VNNI-packed
//   operand layout to global buffers (packedK / packedV / kvZero), so every
//   query work-group reuses the same packed data instead of repacking it.
//   Layouts are identical to v2's SLM blocks.
//
//   Kernel 2 (attn): WG=32, each thread owns 4 query rows. DPAS is M=8 with
//   4 *distinct* rows (50% XMX rows; RPT=8 was attempted and device-lost —
//   see the RPT constant note below). The packed K/V tile is staged
//   cooperatively into SLM once per (query-group, kv-tile); the per-thread
//   fp32 accumulator, query rows, scores, and softmax values live in
//   registers (SLM is shared by all 32 lanes, so per-lane state must not be
//   staged there). q_len is processed 128 rows per work-group.
//
// SLM layout (2*BN*64*4 bytes: 32 KB at BN=64, 64 KB at BN=128; K and V are
// staged into separate regions at the top of each tile, which removes the
// QK -> S*V barrier):
//   KSTAGE  packed K tile
//   VSTAGE  packed V tile
//   kvZero flags are NOT in SLM: they are loaded per tile from the global
//   packed buffer into registers (the wrapper passes the padded kv_len, so
//   zero-row flags are the only reliable padding mask; BN=128 also leaves no
//   SLM room for a flags region).
//
// Only HD=128 is implemented; D64 keeps the v1/v2 kernels.
//
// v4 dispatch (host side):
//   qLen <= 1024 -> fused single kernel (attnDg2<true>): K/V are packed per
//   work-group from the raw row-major buffers into SLM (BN=64 layout with a
//   kvZero flags region), so small shapes pay one submit instead of two.
//   qLen > 1024  -> two kernels (packKvDg2 + attnDg2<false>): K/V packed
//   once into global buffers (BN=128 layout, flags read from global).
//   Both paths are async (no internal queue.wait in release builds); the
//   caller's torch stream synchronizes, matching torch op semantics.
//
// Measured on A770 (driver 32.0.101.8860, oneAPI 2026.1, doubleGRF),
// H=32/D=128 fp16, wall median:
//   L= 512: v4 0.57 ms vs torch 0.66 ms
//   L=1024: v4 1.34 ms vs torch 1.36 ms
//   L=2048: v4 2.9  ms vs torch 3.78 ms
//   L=4096: v4 9.8  ms vs torch 13.5 ms
//   L=8192: v4 40   ms vs torch 42.2 ms
// v4 wins on every benchmarked shape. History: v3 BN=128 79 ms, v3 BN=64
// 105 ms, v2 110 ms.
//
// VTune-driven changes (instruction-count then full-compute):
//   - DPAS A operand chunks stored chunk-major in one simd (qChunkAll /
//     pChunkAll) so the QK/S*V A operand is a contiguous 64-element select.
//   - K and V staged into separate SLM regions at the top of each tile,
//     removing the QK -> S*V barrier (3 barriers/tile -> 2; SLM 64 KB at
//     BN=128 is exactly K 32 KB + V 32 KB).
//   - Softmax exp2 vectorized with the ESIMD hardware native_exp2 (was a
//     scalar per-element software sequence: SP instructions 130G -> 41G).
//   - Softmax max reduced with a vector tree, lTile via vector reduce, and
//     p written chunk-major straight from the exp2 vector.
//   - Scores live directly in per-row simd vectors (svecArr), no scalar
//     staging round trip between QK and softmax.
//   - DPAS A rows 4..7 are not zero-filled (their C rows are never read),
//     removing 4-GRF fills per DPAS.
//   Total attn GPU instructions 404G -> 194G per 12 launches; XVE stall was
//   57% (SLM read 5.56 TB/s near the roof) before these changes.
//   Small-shape overhead was then removed by: fusing pack into attn for
//   qLen<=1024, dropping the per-call queue.wait (async dispatch), and
//   caching the python sidecar glob (was ~0.5 ms per call).
// The remaining gap to the hardware roof is the 50% zero-row waste in DPAS
// (RPT>4 device-losts whenever the compiler spills) plus SLM operand relay.
//
// Negative results recorded for this stack (do not re-run without a
// watchdog):
//   - RPT=6 and RPT=8 both triggered UR_RESULT_ERROR_DEVICE_LOST at
//     L=129/kv=300 whenever the compiler spilled (7.5-16 KB). Any spill is
//     the suspected TDR trigger; RPT=4 with zero spill is stable.
//   - fp16 DPAS accumulator is rejected by dpas.hpp for ExecutionSize=8
//     (fp16 C requires N=16, which is numerically wrong on A770).
//   - Reading packedV directly from global instead of staging via SLM
//     measured 172 ms vs 105 ms at BN=64 (cooperative SLM staging wins).

#include <mutex>
#include <cstdlib>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace dg2v3 {

#ifndef DG2V3_RPT
#define DG2V3_RPT 4
#endif
#ifndef DG2V3_BN
// Measured best on A770 (driver 32.0.101.8860, oneAPI 2026.1, doubleGRF):
// L=8192/H=32/D=128 fp16 79 ms (BN=128) vs 105 ms (BN=64); torch SDPA 43 ms.
// BN=128 uses the full 64 KB per-WG SLM cap (K+V tiles), so kvZero flags are
// loaded per tile from the global packed buffer instead of SLM.
#define DG2V3_BN 128
#endif
#ifndef DG2V3_FUSED_MAX_Q
#define DG2V3_FUSED_MAX_Q 1024
#endif

template <typename T> struct IsFp16V3 : std::false_type {};
template <> struct IsFp16V3<sycl::half> : std::true_type {};

// Full 8x8 DPAS with 8 distinct rows in A.
template <typename ElemT>
ESIMD_INLINE simd<float, 64> dpas8x8(const simd<float, 64>& c,
                                     const simd<uint32_t, 64>& braw,
                                     const simd<ElemT, 128>& a) {
  simd<uint32_t, 64> brawLocal = braw;
  simd<ElemT, 128> b = brawLocal.template bit_cast_view<ElemT>();
  return dpas<8, 8, float>(c, b, a);
}

// ---------------------------------------------------------------------------
// Kernel 1: pack K/V into global VNNI operand layout.
// Grid: (32 * headQ * nKvTiles) threads, WG=32.
//   group -> head = gid / nKvTiles, tile = gid % nKvTiles
// ---------------------------------------------------------------------------
template <typename ElemT>
ESIMD_INLINE void packKvDg2(
    uint8_t* packedK,
    uint8_t* packedV,
    uint8_t* kvZero,
    const uint8_t* kState,
    const uint8_t* vState,
    uint32_t kvLen,
    uint32_t headQ,
    uint32_t headKv,
    sycl::nd_item<1>& ndi) {
  constexpr int BN = DG2V3_BN;
  constexpr int KG = BN / 8;     // K row groups of 8
  constexpr int VP = BN / 2;     // V row pairs
  constexpr int DCHUNKS = 8;
  constexpr int WG = 32;
  const int gid = static_cast<int>(ndi.get_group_linear_id());
  const int lid = static_cast<int>(ndi.get_local_linear_id());
  const int nTiles = (static_cast<int>(kvLen) + BN - 1) / BN;
  const int headIdx = gid / nTiles;
  const int tile = gid % nTiles;
  const int kvBase = tile * BN;
  const size_t tileKOff =
      (static_cast<size_t>(headIdx) * nTiles + tile) * BN * 64;
  const size_t tileVOff =
      (static_cast<size_t>(headIdx) * nTiles + tile) * BN * 64;

  // ---- K pack: thread -> (kvGroup = lid/4, dChunks = (lid%4)*2, +1) ----
  {
#pragma unroll
    for (int g = lid / 4; g < KG; g += WG / 4) {
      const int c0 = (lid % 4) * 2;
      simd<float, 8> rowAbs = 0;
#pragma unroll
      for (int cc = 0; cc < 2; cc++) {
        const int c = c0 + cc;
        simd<ElemT, 128> rows = 0;
#pragma unroll
        for (int r = 0; r < 8; r++) {
          const int row = kvBase + g * 8 + r;
          simd<ElemT, 16> chunk = 0;
          if (row < static_cast<int>(kvLen)) {
            chunk = block_load<ElemT, 16>(
                reinterpret_cast<const ElemT*>(kState) +
                    static_cast<size_t>(row) * headKv * 128 +
                    static_cast<size_t>(headIdx) * 128 + c * 16,
                overaligned_tag<16>{});
          }
          rows.template select<16, 1>(r * 16) = chunk;
          if (cc == 0) {
            simd<float, 16> f = chunk;
            rowAbs[r] = dg2::dg2_sum<16>(__ESIMD_NS::abs(f));
          }
        }
        simd<uint32_t, 64> words;
#pragma unroll
        for (int dp = 0; dp < 8; dp++) {
#pragma unroll
          for (int r = 0; r < 8; r++) {
            const ElemT loVal = static_cast<ElemT>(rows[r * 16 + 2 * dp]);
            const ElemT hiVal = static_cast<ElemT>(rows[r * 16 + 2 * dp + 1]);
            const uint32_t lo =
                static_cast<uint32_t>(sycl::bit_cast<uint16_t>(loVal));
            const uint32_t hi =
                static_cast<uint32_t>(sycl::bit_cast<uint16_t>(hiVal));
            words[dp * 8 + r] = lo | (hi << 16);
          }
        }
        block_store<uint32_t, 64>(
            reinterpret_cast<uint32_t*>(packedK) + tileKOff +
                (g * DCHUNKS + c) * 64,
            words,
            overaligned_tag<16>{});
      }
      simd<uint32_t, 8> flagGroup;
#pragma unroll
      for (int r = 0; r < 8; r++) {
        flagGroup[r] = (rowAbs[r] > 0.0f) ? 1u : 0u;
      }
      block_store<uint32_t, 8>(
          // Layout matches the attn kernel: kvZero[head * nTiles * BN + row].
          reinterpret_cast<uint32_t*>(kvZero) +
              static_cast<size_t>(headIdx) * nTiles * BN + kvBase + g * 8,
          flagGroup,
          overaligned_tag<16>{});
    }
  }

  // ---- V pack: thread = row pair ----
  {
#pragma unroll
    for (int pair = lid; pair < VP; pair += WG) {
      const int row0 = kvBase + pair * 2;
      const int row1 = row0 + 1;
#pragma unroll
      for (int c = 0; c < DCHUNKS; c++) {
        simd<ElemT, 16> lo = 0;
        simd<ElemT, 16> hi = 0;
        if (row0 < static_cast<int>(kvLen)) {
          lo = block_load<ElemT, 16>(
              reinterpret_cast<const ElemT*>(vState) +
                  static_cast<size_t>(row0) * headKv * 128 +
                  static_cast<size_t>(headIdx) * 128 + c * 16,
              overaligned_tag<16>{});
        }
        if (row1 < static_cast<int>(kvLen)) {
          hi = block_load<ElemT, 16>(
              reinterpret_cast<const ElemT*>(vState) +
                  static_cast<size_t>(row1) * headKv * 128 +
                  static_cast<size_t>(headIdx) * 128 + c * 16,
              overaligned_tag<16>{});
        }
#pragma unroll
        for (int h = 0; h < 2; h++) {
          simd<ElemT, 16> loHalf = 0;
          simd<ElemT, 16> hiHalf = 0;
          for (int i = 0; i < 8; i++) {
            loHalf[i] = lo[8 * h + i];
            hiHalf[i] = hi[8 * h + i];
          }
          simd<uint32_t, 8> w;
#pragma unroll
          for (int i = 0; i < 8; i++) {
            const uint32_t a =
                static_cast<uint32_t>(sycl::bit_cast<uint16_t>(
                    static_cast<ElemT>(loHalf[i])));
            const uint32_t b =
                static_cast<uint32_t>(sycl::bit_cast<uint16_t>(
                    static_cast<ElemT>(hiHalf[i])));
            w[i] = a | (b << 16);
          }
          block_store<uint32_t, 8>(
              reinterpret_cast<uint32_t*>(packedV) + tileVOff +
                  ((pair / 8) * DCHUNKS + c) * 2 * 64 + h * 64 +
                  (pair % 8) * 8,
              w,
              overaligned_tag<16>{});
        }
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Kernel 2: flash attention over packed K/V.
// Grid: (32 * headQ * qTiles) threads, WG=32; 256 query rows per group.
// ---------------------------------------------------------------------------
template <typename ElemT, bool FUSED = false>
ESIMD_INLINE void attnDg2(
    const uint8_t* qState,
    const uint8_t* kState,
    const uint8_t* vState,
    const uint8_t* packedK,
    const uint8_t* packedV,
    const uint8_t* kvZero,
    const float* normAlpha,
    uint8_t* out,
    float* dbg,
    uint32_t qLen,
    uint32_t kvLen,
    uint32_t headQ,
    uint32_t headKv,
    bool stageOnly,
    bool qkOnly,
    bool dumpState,
    bool svOnly,
    bool svRegAcc,
    bool svConstA,
    sycl::nd_item<1>& ndi) {
  // RPT>4 attempts (6 and 8 rows) both triggered UR_RESULT_ERROR_DEVICE_LOST
  // on A770 at L=129/kv=300 whenever the compiler spilled (7.5-16 KB),
  // driver 32.0.101.8860 / oneAPI 2026.1 / doubleGRF. RPT=4 (50% DPAS rows)
  // is the stable geometry; any spill is the suspected TDR trigger.
  constexpr int RPT = DG2V3_RPT;    // query rows per thread
  constexpr int QGRP = 32 * RPT;    // query rows per work-group
  // The fused single-kernel path (small qLen) packs K/V per work-group from
  // the raw row-major buffers and therefore uses BN=64 so the kvZero flags
  // fit in SLM; the two-kernel path keeps BN=DG2V3_BN (128).
  constexpr int BN = FUSED ? 64 : DG2V3_BN;
  constexpr int KG = BN / 8;        // K row groups of 8
  constexpr int PG = BN / 16;       // V row groups of 16
  constexpr int DCHUNKS = 8;
  constexpr int WG = 32;
  // K and V are staged into separate regions at the top of the tile, so the
  // QK -> S*V transition needs no barrier (3 barriers/tile -> 2). At BN=128
  // this uses the full 64 KB per-WG SLM cap (2 * 32 KB).
  constexpr int KSTAGE = 0;
  constexpr int VSTAGE = BN * 64 * 4;
  constexpr int FLAGS = VSTAGE + BN * 64 * 4;
  constexpr int SLM_TOTAL = FUSED ? (FLAGS + BN * 4)
                                  : (2 * BN * 64 * 4);
  constexpr float SCALE = 0.08838834764831844f;
  constexpr float LOG2E = 1.4426950408889634f;
  constexpr float MASKED_SCORE = -1.0e30f;

  const int gid = static_cast<int>(ndi.get_group_linear_id());
  const int lid = static_cast<int>(ndi.get_local_linear_id());
  const int nTiles = (static_cast<int>(kvLen) + BN - 1) / BN;
  const int qTiles = (static_cast<int>(qLen) + QGRP - 1) / QGRP;
  const int headIdx = gid / qTiles;
  const int qTile = gid % qTiles;
  const int qrowBase = qTile * QGRP + lid * RPT;

  slm_init(SLM_TOTAL);

  // Per-thread query rows stay in registers (SLM is shared by all 32 lanes;
  // staging per-lane rows there would be a write race). qChunkAll is stored
  // chunk-major: [c][r][k] = q[row r][c*16+k] at c*64 + r*16 + k, so the
  // QK DPAS A operand (4 rows x 16 k) is a contiguous 64-element block per
  // chunk instead of 64 cross-stride selects (VTune: "Other" instructions
  // were 60% of the attn kernel).
  simd<ElemT, RPT * DCHUNKS * 16> qChunkAll = 0;
#pragma unroll
  for (int r = 0; r < RPT; r++) {
    const int qrow = qrowBase + r;
    if (qrow < static_cast<int>(qLen)) {
      const ElemT* qp = reinterpret_cast<const ElemT*>(qState) +
                        (static_cast<size_t>(qrow) * headQ + headIdx) * 128;
#pragma unroll
      for (int c = 0; c < DCHUNKS; c++) {
        qChunkAll.template select<16, 1>(c * 64 + r * 16) =
            block_load<ElemT, 16>(qp + c * 16, overaligned_tag<16>{});
      }
    }
  }

  float mArr[RPT];
  float lArr[RPT];
  simd<float, RPT * 128> acc = 0;
  simd<ElemT, RPT * DCHUNKS * 16> pChunkAll = 0;
  // Scores live directly in per-row simd vectors so QK writes and the
  // softmax reads are vector ops (no scalar staging round trip).
  simd<float, BN> svecArr[RPT];
#pragma unroll
  for (int r = 0; r < RPT; r++) {
    mArr[r] = -3.402823466e+38f;
    lArr[r] = 0.0f;
  }

  const size_t tileKOff = static_cast<size_t>(headIdx) * nTiles * BN * 64;
  const size_t tileVOff = static_cast<size_t>(headIdx) * nTiles * BN * 64;
  const size_t zeroOff = static_cast<size_t>(headIdx) * nTiles * BN;

  for (int t = 0; t < nTiles; t++) {
    // ---- Stage packed K and V into separate SLM regions ----
    {
      if constexpr (FUSED) {
        // Pack this 64-row tile directly from the raw K/V buffers into SLM.
        // K: thread -> (kvGroup = lid/4, dChunks = (lid%4)*2, +1).
        const int g = lid / 4;
        const int c0 = (lid % 4) * 2;
        simd<float, 8> rowAbs = 0;
#pragma unroll
        for (int cc = 0; cc < 2; cc++) {
          const int c = c0 + cc;
          simd<ElemT, 128> rows = 0;
#pragma unroll
          for (int r = 0; r < 8; r++) {
            const int row = t * 64 + g * 8 + r;
            simd<ElemT, 16> chunk = 0;
            if (row < static_cast<int>(kvLen)) {
              chunk = block_load<ElemT, 16>(
                  reinterpret_cast<const ElemT*>(kState) +
                      static_cast<size_t>(row) * headKv * 128 +
                      static_cast<size_t>(headIdx) * 128 + c * 16,
                  overaligned_tag<16>{});
            }
            rows.template select<16, 1>(r * 16) = chunk;
            if (cc == 0) {
              simd<float, 16> f = chunk;
              rowAbs[r] = dg2::dg2_sum<16>(__ESIMD_NS::abs(f));
            }
          }
          simd<uint32_t, 64> words;
#pragma unroll
          for (int dp = 0; dp < 8; dp++) {
#pragma unroll
            for (int r = 0; r < 8; r++) {
              const ElemT loVal =
                  static_cast<ElemT>(rows[r * 16 + 2 * dp]);
              const ElemT hiVal =
                  static_cast<ElemT>(rows[r * 16 + 2 * dp + 1]);
              const uint32_t lo = static_cast<uint32_t>(
                  sycl::bit_cast<uint16_t>(loVal));
              const uint32_t hi = static_cast<uint32_t>(
                  sycl::bit_cast<uint16_t>(hiVal));
              words[dp * 8 + r] = lo | (hi << 16);
            }
          }
          slm_block_store(
              KSTAGE + (g * DCHUNKS + c) * 64 * 4,
              words,
              overaligned_tag<16>{});
        }
        simd<uint32_t, 8> flagGroup;
#pragma unroll
        for (int r = 0; r < 8; r++) {
          flagGroup[r] = (rowAbs[r] > 0.0f) ? 1u : 0u;
        }
        slm_block_store(FLAGS + g * 8 * 4, flagGroup, overaligned_tag<16>{});
        // V: thread = row pair, packs 8 words per (dChunk, dHalf).
        const int pair = lid;
        const int row0 = t * 64 + pair * 2;
        const int row1 = row0 + 1;
#pragma unroll
        for (int c = 0; c < DCHUNKS; c++) {
          simd<ElemT, 16> lo = 0;
          simd<ElemT, 16> hi = 0;
          if (row0 < static_cast<int>(kvLen)) {
            lo = block_load<ElemT, 16>(
                reinterpret_cast<const ElemT*>(vState) +
                    static_cast<size_t>(row0) * headKv * 128 +
                    static_cast<size_t>(headIdx) * 128 + c * 16,
                overaligned_tag<16>{});
          }
          if (row1 < static_cast<int>(kvLen)) {
            hi = block_load<ElemT, 16>(
                reinterpret_cast<const ElemT*>(vState) +
                    static_cast<size_t>(row1) * headKv * 128 +
                    static_cast<size_t>(headIdx) * 128 + c * 16,
                overaligned_tag<16>{});
          }
#pragma unroll
          for (int h = 0; h < 2; h++) {
            simd<ElemT, 16> loHalf = 0;
            simd<ElemT, 16> hiHalf = 0;
            for (int i = 0; i < 8; i++) {
              loHalf[i] = lo[8 * h + i];
              hiHalf[i] = hi[8 * h + i];
            }
            simd<uint32_t, 8> w;
#pragma unroll
            for (int i = 0; i < 8; i++) {
              const uint32_t a = static_cast<uint32_t>(
                  sycl::bit_cast<uint16_t>(static_cast<ElemT>(loHalf[i])));
              const uint32_t b = static_cast<uint32_t>(
                  sycl::bit_cast<uint16_t>(static_cast<ElemT>(hiHalf[i])));
              w[i] = a | (b << 16);
            }
            slm_block_store(
                VSTAGE + (((pair / 8) * DCHUNKS + c) * 2 + h) * 64 * 4 +
                    (pair % 8) * 8 * 4,
                w,
                overaligned_tag<16>{});
          }
        }
      } else {
      const size_t tkOff = tileKOff + static_cast<size_t>(t) * BN * 64;
#pragma unroll
      for (int i = 0; i < BN / 32; i++) {
        simd<uint32_t, 64> words = block_load<uint32_t, 64>(
            reinterpret_cast<const uint32_t*>(packedK) + tkOff +
                (lid * (BN / 32) + i) * 64,
            overaligned_tag<16>{});
        slm_block_store(
            KSTAGE + (lid * (BN / 32) + i) * 64 * 4,
            words,
            overaligned_tag<16>{});
      }
      const size_t tvOff = tileVOff + static_cast<size_t>(t) * BN * 64;
#pragma unroll
      for (int i = 0; i < BN / 32; i++) {
        simd<uint32_t, 64> words = block_load<uint32_t, 64>(
            reinterpret_cast<const uint32_t*>(packedV) + tvOff +
                (lid * (BN / 32) + i) * 64,
            overaligned_tag<16>{});
        slm_block_store(
            VSTAGE + (lid * (BN / 32) + i) * 64 * 4,
            words,
            overaligned_tag<16>{});
      }
      }
    }
    barrier();

    if (stageOnly) {
      continue;
    }

    // Fused: flags are in SLM (packed by this work-group). Two-kernel:
    // kvZero flags come from the global packed buffer (BN=128 leaves no SLM
    // room). The wrapper passes the padded kv_len, so the zero-row flags are
    // the only reliable padding mask in both paths.
    simd<uint32_t, 64> flagsLo;
#if DG2V3_BN == 128
    simd<uint32_t, 64> flagsHi;
#endif
    if constexpr (FUSED) {
      flagsLo = slm_block_load<uint32_t, 64>(FLAGS, overaligned_tag<16>{});
    } else {
      flagsLo = block_load<uint32_t, 64>(
          reinterpret_cast<const uint32_t*>(kvZero) + zeroOff +
              static_cast<size_t>(t) * BN,
          overaligned_tag<16>{});
#if DG2V3_BN == 128
      flagsHi = block_load<uint32_t, 64>(
          reinterpret_cast<const uint32_t*>(kvZero) + zeroOff +
              static_cast<size_t>(t) * BN + 64,
          overaligned_tag<16>{});
#endif
    }

    // ---- QK^T: BN scores per query row (kept in registers) ----
#pragma unroll
    for (int g = 0; g < KG; g++) {
      simd<float, 64> c64 = 0;
#pragma unroll
      for (int c = 0; c < DCHUNKS; c++) {
        // Rows 4..7 of C are never read (RPT=4), so only rows 0..3 of A need
        // to be defined; skipping the 4-GRF zero fill cuts per-dpas moves.
        simd<ElemT, 128> a;
        a.template select<RPT * 16, 1>(0) =
            qChunkAll.template select<RPT * 16, 1>(c * RPT * 16);
        simd<uint32_t, 64> braw = slm_block_load<uint32_t, 64>(
            KSTAGE + (g * DCHUNKS + c) * 64 * 4, overaligned_tag<16>{});
        c64 = dpas8x8<ElemT>(c64, braw, a);
      }
#pragma unroll
      for (int r = 0; r < RPT; r++) {
#pragma unroll
        for (int i = 0; i < 8; i++) {
          const int j = g * 8 + i;
          float s = static_cast<float>(c64[r * 8 + i]) * SCALE;
          const uint32_t flag = [&]() -> uint32_t {
            if constexpr (FUSED) {
              return static_cast<uint32_t>(flagsLo[j]);
            } else {
#if DG2V3_BN == 128
              return (j < 64) ? static_cast<uint32_t>(flagsLo[j])
                              : static_cast<uint32_t>(flagsHi[j - 64]);
#else
              return static_cast<uint32_t>(flagsLo[j]);
#endif
            }
          }();
          if (t * BN + j >= static_cast<int>(kvLen) || flag == 0u) {
            s = MASKED_SCORE;
          }
          svecArr[r][j] = s;
        }
      }
    }

    // ---- S*V (V already staged into its own SLM region) ----
    {
      if (qkOnly) {
#if defined(DG2V3_DEBUG)
        // Debug dump into the fp32 dbg buffer (host copies it to a file).
        if (lid == 0 && qTile == 0 && dbg && t < 2) {
          int w = 0;
          // 1. q rows 0..3, 128 fp16 each (as float), dumped once (t==0)
          if (t == 0) {
            for (int r = 0; r < RPT; r++) {
#pragma unroll
              for (int c = 0; c < DCHUNKS; c++) {
                simd<ElemT, 16> qc =
                    qChunkAll.template select<16, 1>(c * RPT * 16 + r * 16);
                for (int i = 0; i < 16; i++) {
                  dbg[w++] = static_cast<float>(static_cast<ElemT>(qc[i]));
                }
              }
            }
          }
          w = 512;
          // 2. K rows 0..15 for THIS tile (per-tile block at 512 + t*2048)
          w += t * 2048;
          {
            for (int g = 0; g < 2; g++) {
              for (int c = 0; c < DCHUNKS; c++) {
                simd<uint32_t, 64> kb = slm_block_load<uint32_t, 64>(
                    KSTAGE + (g * DCHUNKS + c) * 64 * 4,
                    overaligned_tag<16>{});
                simd<ElemT, 128> kf = kb.template bit_cast_view<ElemT>();
                for (int i = 0; i < 128; i++) {
                  dbg[w++] = static_cast<float>(static_cast<ElemT>(kf[i]));
                }
              }
            }
          }
          // 3. SC scores (scaled, masked), rows 0..3 x 16 kv, per tile
          w = 4608 + t * 64;
          {
            for (int r = 0; r < RPT; r++) {
              for (int j = 0; j < BN; j++) {
                dbg[w++] = svecArr[r][j];
              }
            }
          }
        }
#endif
        continue;
      }
    }

    if (!svOnly) {
      // ---- Two-pass softmax per row (registers) ----
#pragma unroll
      for (int r = 0; r < RPT; r++) {
        simd<float, BN> svec = svecArr[r];
        // Vector max reduction instead of BN scalar compares.
        const float mTile = __ESIMD_NS::detail::reduce<
            float, float, BN,
            __ESIMD_NS::detail::esimd_apply_reduced_max>(svec);
        float lTile = 0.0f;
        svec = (svec - mTile) * LOG2E;
        simd<float, BN> pvec = __ESIMD_NS::exp2<float, BN>(svec);
#pragma unroll
        for (int p = 0; p < PG; p++) {
          simd<float, 16> p16 = pvec.template select<16, 1>(p * 16);
          simd<ElemT, 16> ph = convert<ElemT>(p16);
          lTile += __ESIMD_NS::detail::reduce<
              float, float, 16, __ESIMD_NS::detail::esimd_apply_sum>(p16);
          pChunkAll.template select<16, 1>(p * RPT * 16 + r * 16) = ph;
        }
        const float rescale = __ESIMD_NS::exp2<float, 1>(
            simd<float, 1>((mArr[r] - mTile) * LOG2E))[0];
        lArr[r] = lArr[r] * rescale + lTile;
        mArr[r] = mTile;
        acc.template select<128, 1>(r * 128) =
            acc.template select<128, 1>(r * 128) * rescale;
      }
    } else {
      // S*V isolation test: zero p so the DPAS output is 0.
#pragma unroll
      for (int i = 0; i < RPT * DCHUNKS * 16; i++) {
        pChunkAll[i] = static_cast<ElemT>(0);
      }
    }

    // ---- S*V: accumulator in registers, rescaled before this loop ----
#pragma unroll
    for (int c = 0; c < DCHUNKS; c++) {
      // fp16 accumulator (ExecutionSize=8, fp16 C) is rejected by dpas.hpp:
      // with N=8 only fp32 C is supported; N=16 supports fp16 C but is
      // numerically wrong on A770 (see the skill negative result). fp32 C
      // is therefore the only valid A770 form.
      simd<float, 64> accH0 = 0;
      simd<float, 64> accH1 = 0;
#pragma unroll
      for (int r = 0; r < RPT; r++) {
        if (!svRegAcc) {
          accH0.template select<8, 1>(r * 8) =
              acc.template select<8, 1>(r * 128 + c * 16);
          accH1.template select<8, 1>(r * 8) =
              acc.template select<8, 1>(r * 128 + c * 16 + 8);
        }
      }
#pragma unroll
      for (int p = 0; p < PG; p++) {
        simd<ElemT, 128> a;
        if (svConstA) {
          a = static_cast<ElemT>(1);
        } else {
          a.template select<RPT * 16, 1>(0) =
              pChunkAll.template select<RPT * 16, 1>(p * RPT * 16);
        }
        simd<uint32_t, 64> b0 = slm_block_load<uint32_t, 64>(
            VSTAGE + ((p * DCHUNKS + c) * 2) * 64 * 4,
            overaligned_tag<16>{});
        simd<uint32_t, 64> b1 = slm_block_load<uint32_t, 64>(
            VSTAGE + ((p * DCHUNKS + c) * 2 + 1) * 64 * 4,
            overaligned_tag<16>{});
        accH0 = dpas8x8<ElemT>(accH0, b0, a);
        accH1 = dpas8x8<ElemT>(accH1, b1, a);
      }
#pragma unroll
      for (int r = 0; r < RPT; r++) {
        if (!svRegAcc) {
          acc.template select<8, 1>(r * 128 + c * 16) =
              accH0.template select<8, 1>(r * 8);
          acc.template select<8, 1>(r * 128 + c * 16 + 8) =
              accH1.template select<8, 1>(r * 8);
        }
      }
    }

    barrier();
  }

  // ---- Output normalization + store ----
  if ((qkOnly || dumpState) && lid == 0 && qTile == 0 && dbg) {
    // 4. final m/l and accumulator row 0 after all tiles
    for (int r = 0; r < RPT; r++) {
      dbg[5500 + r] = mArr[r];
      dbg[5504 + r] = lArr[r];
    }
    for (int i = 0; i < 128; i++) {
      dbg[5508 + i] = acc[i];
    }
  }
#pragma unroll
  for (int r = 0; r < RPT; r++) {
    const int qrow = qrowBase + r;
    if (qrow < static_cast<int>(qLen)) {
      const float* alphaBase =
          normAlpha + static_cast<size_t>(headIdx) * 128;
      ElemT* op = reinterpret_cast<ElemT*>(out) +
                  (static_cast<size_t>(qrow) * headQ + headIdx) * 128;
#pragma unroll
      for (int c = 0; c < DCHUNKS; c++) {
        simd<float, 16> outv =
            acc.template select<16, 1>(r * 128 + c * 16);
        simd<float, 16> alpha =
            block_load<float, 16>(alphaBase + c * 16, overaligned_tag<16>{});
        outv = outv * (1.0f / lArr[r]) * alpha;
        if constexpr (IsFp16V3<ElemT>::value) {
          outv.merge(65504.0f, outv > 65504.0f);
          outv.merge(-65504.0f, outv < -65504.0f);
        }
        block_store<ElemT, 16>(
            op + c * 16, simd<ElemT, 16>(outv), overaligned_tag<16>{});
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Host orchestration: pack once, then run attention.
// ---------------------------------------------------------------------------
struct V3Buffers {
  void* packedK = nullptr;
  void* packedV = nullptr;
  void* kvZero = nullptr;
};

inline std::unordered_map<uint64_t, V3Buffers>& v3_cache() {
  static std::unordered_map<uint64_t, V3Buffers> cache;
  return cache;
}

inline std::mutex& v3_mutex() {
  static std::mutex m;
  return m;
}

template <typename ElemT>
inline void runSdpV3(
    const void* q,
    const void* k,
    const void* v,
    const void* alpha,
    void* out,
    int qLen,
    int kvLen,
    int headQ,
    int headKv,
    sycl::queue& queue) {
  constexpr int BN = DG2V3_BN;
  const int nTiles = (kvLen + BN - 1) / BN;
  const size_t tileWords = static_cast<size_t>(BN) * 64;
  const uint64_t key =
      (static_cast<uint64_t>(headQ) << 32) | static_cast<uint32_t>(kvLen);

  const int qTiles = (qLen + 127) / 128;
  const int packGroups = headQ * nTiles;
  const int attnGroups = headQ * qTiles;
  constexpr int WG = 32;

#if defined(DG2V3_DEBUG)
  const bool attnOnly = std::getenv("DG2V3_ATTN_ONLY") != nullptr;
  const bool packOnly = std::getenv("DG2V3_PACK_ONLY") != nullptr;
  const bool stageOnly = std::getenv("DG2V3_ATTN_STAGE_ONLY") != nullptr;
  const bool qkOnly = std::getenv("DG2V3_ATTN_QK_ONLY") != nullptr;
  const bool svOnly = std::getenv("DG2V3_ATTN_SV_ONLY") != nullptr;
  const bool svRegAcc = std::getenv("DG2V3_ATTN_SV_REGACC") != nullptr;
  const bool svConstA = std::getenv("DG2V3_ATTN_SV_CONSTA") != nullptr;
  const bool dumpQk = std::getenv("DG2V3_DUMP_QK") != nullptr;
  const bool dumpState = std::getenv("DG2V3_DUMP_STATE") != nullptr;
#else
  const bool attnOnly = false;
  const bool packOnly = false;
  const bool stageOnly = false;
  const bool qkOnly = false;
  const bool svOnly = false;
  const bool svRegAcc = false;
  const bool svConstA = false;
  const bool dumpQk = false;
  const bool dumpState = false;
#endif
  // Small qLen: fuse K/V packing into the attention kernel (one submit, one
  // wait; the per-WG repack cost is small when qTiles is small). Larger
  // qLen keeps the two-kernel path so K/V are packed only once.
  const bool useFused = qLen <= DG2V3_FUSED_MAX_Q && !attnOnly && !packOnly;
  V3Buffers buf;
  if (!useFused) {
    {
      std::lock_guard<std::mutex> guard(v3_mutex());
      auto it = v3_cache().find(key);
      if (it == v3_cache().end()) {
        buf.packedK = sycl::aligned_alloc_device(
            64,
            static_cast<size_t>(headQ) * nTiles * tileWords * 4,
            queue);
        buf.packedV = sycl::aligned_alloc_device(
            64,
            static_cast<size_t>(headQ) * nTiles * tileWords * 4,
            queue);
        buf.kvZero = sycl::aligned_alloc_device(
            64,
            static_cast<size_t>(headQ) * nTiles * BN * 4,
            queue);
        v3_cache()[key] = buf;
      } else {
        buf = it->second;
      }
    }
  }
  float* dbgBuf = nullptr;
#if defined(DG2V3_DEBUG)
  if (dumpQk) {
    dbgBuf = static_cast<float*>(sycl::aligned_alloc_device(
        64, 8192 * sizeof(float), queue));
  }
#endif
  if (!attnOnly && !useFused) {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(packGroups * WG),
                            sycl::range<1>(WG)),
          [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            packKvDg2<ElemT>(
                static_cast<uint8_t*>(buf.packedK),
                static_cast<uint8_t*>(buf.packedV),
                static_cast<uint8_t*>(buf.kvZero),
                static_cast<const uint8_t*>(k),
                static_cast<const uint8_t*>(v),
                static_cast<uint32_t>(kvLen),
                static_cast<uint32_t>(headQ),
                static_cast<uint32_t>(headKv),
                ndi);
        });
    });
  }
  if (!packOnly) {
    if (useFused) {
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(attnGroups * WG),
                              sycl::range<1>(WG)),
            [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
              attnDg2<ElemT, true>(
                  static_cast<const uint8_t*>(q),
                  static_cast<const uint8_t*>(k),
                  static_cast<const uint8_t*>(v),
                  nullptr,
                  nullptr,
                  nullptr,
                  static_cast<const float*>(alpha),
                  static_cast<uint8_t*>(out),
                  dbgBuf,
                  static_cast<uint32_t>(qLen),
                  static_cast<uint32_t>(kvLen),
                  static_cast<uint32_t>(headQ),
                  static_cast<uint32_t>(headKv),
                  stageOnly,
                  qkOnly,
                  dumpState,
                  svOnly,
                  svRegAcc,
                  svConstA,
                  ndi);
          });
      });
    } else {
      queue.submit([&](sycl::handler& cgh) {
        cgh.parallel_for(
            sycl::nd_range<1>(sycl::range<1>(attnGroups * WG),
                              sycl::range<1>(WG)),
            [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
              attnDg2<ElemT, false>(
                  static_cast<const uint8_t*>(q),
                  nullptr,
                  nullptr,
                  static_cast<const uint8_t*>(buf.packedK),
                  static_cast<const uint8_t*>(buf.packedV),
                  static_cast<const uint8_t*>(buf.kvZero),
                  static_cast<const float*>(alpha),
                  static_cast<uint8_t*>(out),
                  dbgBuf,
                  static_cast<uint32_t>(qLen),
                  static_cast<uint32_t>(kvLen),
                  static_cast<uint32_t>(headQ),
                  static_cast<uint32_t>(headKv),
                  stageOnly,
                  qkOnly,
                  dumpState,
                  svOnly,
                  svRegAcc,
                  svConstA,
                  ndi);
          });
      });
    }
  }
#if defined(DG2V3_DEBUG)
  queue.wait();
#else
  // Async dispatch: the caller (torch stream) synchronizes. Keeping the
  // queue non-blocked lets consecutive SDP calls overlap host submit with
  // device execution, matching torch op semantics. Debug paths below need
  // the synchronous wait so their memcpy dumps observe completed kernels.
#endif
#if defined(DG2V3_DEBUG)
  if (dumpQk && dbgBuf) {
    std::vector<float> dbg(8192);
    queue.memcpy(dbg.data(), dbgBuf, 8192 * sizeof(float)).wait();
    std::ofstream fqk("C:/Temp/dg2v3_qk.bin", std::ios::binary);
    fqk.write(reinterpret_cast<const char*>(dbg.data()), 8192 * sizeof(float));
    fqk.close();
    sycl::free(dbgBuf, queue);
  }
#endif
#if defined(DG2V3_DEBUG)
  if (std::getenv("DG2V3_DUMP_PACKED") != nullptr && !attnOnly) {
    std::vector<uint32_t> dumpK(4096), dumpV(4096);
    queue.memcpy(dumpK.data(), buf.packedK, 4096 * 4).wait();
    queue.memcpy(dumpV.data(), buf.packedV, 4096 * 4).wait();
    std::vector<uint32_t> dumpZ(512);
    queue.memcpy(dumpZ.data(), buf.kvZero, 512 * 4).wait();
    std::ofstream fk("C:/Temp/dg2v3_packedK.bin", std::ios::binary);
    fk.write(reinterpret_cast<const char*>(dumpK.data()), 4096 * 4);
    fk.close();
    std::ofstream fv("C:/Temp/dg2v3_packedV.bin", std::ios::binary);
    fv.write(reinterpret_cast<const char*>(dumpV.data()), 4096 * 4);
    fv.close();
    std::ofstream fz("C:/Temp/dg2v3_kvZero.bin", std::ios::binary);
    fz.write(reinterpret_cast<const char*>(dumpZ.data()), 512 * 4);
    fz.close();
  }
#endif
}

}  // namespace dg2v3
