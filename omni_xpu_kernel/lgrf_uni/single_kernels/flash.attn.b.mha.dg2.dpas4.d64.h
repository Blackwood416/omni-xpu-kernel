// DG2-native Flash Attention v4.1 D64 port (MiniMax H3 VideoVAE).
// Generated from flash.attn.b.mha.dg2.dpas4.h with DCHUNKS=4 and
// row width 64; see scripts/gen_dg2_d64_header.py for the mapping.
// Same A770 rules: WG=32, zero spill, fp32 DPAS accumulation.

// DG2-native Flash Attention v4.1 — packed-KV + packed-Q DPAS design.
// Non-fused path prepacks Q into the DPAS A-operand layout (one 256 B block
// per QK operand), which lets RPT=8 run with 100% XMX rows and zero spill.
// Small shapes keep the fused RPT=4 kernel; the host passes ORIGINAL kv_len
// so padded rows are masked by the row-index check (no kvZero flags needed).
//
// Dispatch:
//   qLen <= 1024 && qTilesFused * kvTilesFused <= 128 -> fused
//   attnDg2<true>: K/V are packed per work-group from raw row-major buffers
//   into SLM (BN=64, RPT=4, WG=32), one submit.
//   Otherwise -> packQDg2 + packKvDg2 + attnDg2<false>: Q/K/V are packed
//   once into global operand layouts (BN=64, RPT=8, WG=32), three submits.
//   Both paths are async (no internal queue.wait in release builds); the
//   caller's torch stream synchronizes, matching torch op semantics.
//
// SLM layout: KSTAGE at 0, VSTAGE at BN*64*4. K and V are staged into
// separate regions at the top of each tile, so QK -> S*V needs no extra
// barrier (2 barriers/tile).
//
// Measured on A770 (driver 32.0.101.8860, oneAPI 2026.1, doubleGRF),
// H=32/D=128 fp16 wall median: L=512 0.54 vs 0.60 ms, L=1024 ~1.2-1.4 ms vs
// torch 1.04-1.36 ms, L=2048 2.75 vs 3.29 ms, L=4096 8.8 vs 12.6 ms,
// L=8192 34.7 vs 41.4 ms. Extended 40-sample runs also win 1024x4096
// (2.91 vs 3.20 ms), 1024x1024 H48 (1.43 vs 1.49), and 512x512 H48
// (0.64 vs 0.65).
//
// VTune xpu-offload (1024x4096): attn ~73% of GPU time, packKv ~23%,
// packQ ~3%. The remaining attn cost is K/V SLM staging and operand relay.
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
//   - RPT=6 with BN=128 compiled without spills but DEVICE_LOST on the
//     first real launch; keep RPT8/BN64 for the two-kernel path.
//   - WG=64 for the non-fused attn kernel produced wrong outputs even on a
//     single KV tile (no TDR), so A770 staging/barrier geometry stays WG=32.
//   - Fused BN=128 is numerically wrong on this stack (single-tile error
//     ~0.12); fused stays BN=64.
//   - Fused RPT=6 spills at both BN=64 (1.7-2.0 KB) and BN=32 (0.9-1.3 KB),
//     so fused stays RPT=4.

#include <mutex>
#include <cstdlib>
#include <fstream>
#include <unordered_map>
#include <vector>

namespace dg2v4d64 {

#ifndef DG2V4D64_RPT
// Non-fused D64 uses RPT=8 (100% XMX rows) like the D128 v4.1 path.
// Measured on A770 (driver 32.0.101.8860): correct and stable (20x stress at
// 1797/20683), but 0.84-0.92x of torch SDPA at L>=4096 and 0.57x at L=1797.
// ComfyUI-OmniXPU therefore keeps fp16/D64 routed to torch SDPA on DG2; this
// kernel is retained for other targets and future driver/toolchain retests.
#define DG2V4D64_RPT 8
#endif
#ifndef DG2V4D64_FUSED_RPT
#define DG2V4D64_FUSED_RPT 4
#endif
#ifndef DG2V4_BN
// Measured best on A770 (driver 32.0.101.8860, oneAPI 2026.1, doubleGRF):
// L=8192/H=32/D=128 fp16 79 ms (BN=128) vs 105 ms (BN=64); torch SDPA 43 ms.
// BN=128 uses the full 64 KB per-WG SLM cap (K+V tiles), so kvZero flags are
// loaded per tile from the global packed buffer instead of SLM.
#define DG2V4_BN 64
#endif
#ifndef DG2V4_FUSED_MAX_Q
#define DG2V4_FUSED_MAX_Q 1024
#endif
#ifndef DG2V4_FUSED_MAX_TILES
#define DG2V4_FUSED_MAX_TILES 128
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
// Kernel 0: pack Q into the DPAS A-operand layout.
// Grid: (headQ * qTiles * WG) threads, WG=32. Each thread owns RPT query
// rows; packedQ is [head][qTile][thread][dChunk][row][16] so every QK DPAS A
// operand is one contiguous 256 B block (pre-packed A, zero selects).
// ---------------------------------------------------------------------------
template <typename ElemT>
ESIMD_INLINE void packQDg2(
    uint8_t* packedQ,
    const uint8_t* qState,
    uint32_t qLen,
    uint32_t headQ,
    sycl::nd_item<1>& ndi) {
  constexpr int RPT = DG2V4D64_RPT;
  constexpr int DCHUNKS = 4;
  constexpr int WG = 32;
  constexpr int QGRP = WG * RPT;
  const int gid = static_cast<int>(ndi.get_group_linear_id());
  const int lid = static_cast<int>(ndi.get_local_linear_id());
  const int qTiles = (static_cast<int>(qLen) + QGRP - 1) / QGRP;
  const int headIdx = gid / qTiles;
  const int qTile = gid % qTiles;
  const int qrowBase = qTile * QGRP + lid * RPT;
  const size_t threadOff =
      ((static_cast<size_t>(headIdx) * qTiles + qTile) * WG + lid) *
      RPT * 64;

  simd<ElemT, RPT * 64> qrows = 0;
#pragma unroll
  for (int r = 0; r < RPT; r++) {
    const int qrow = qrowBase + r;
    if (qrow < static_cast<int>(qLen)) {
      const ElemT* qp = reinterpret_cast<const ElemT*>(qState) +
                        (static_cast<size_t>(qrow) * headQ + headIdx) * 64;
#pragma unroll
      for (int c = 0; c < DCHUNKS; c++) {
        qrows.template select<16, 1>(r * 64 + c * 16) =
            block_load<ElemT, 16>(qp + c * 16, overaligned_tag<16>{});
      }
    }
  }
#pragma unroll
  for (int c = 0; c < DCHUNKS; c++) {
    simd<ElemT, 128> block = 0;
#pragma unroll
    for (int r = 0; r < RPT; r++) {
      block.template select<16, 1>(r * 16) =
          qrows.template select<16, 1>(r * 64 + c * 16);
    }
    block_store<ElemT, RPT * 16>(
        reinterpret_cast<ElemT*>(packedQ) + threadOff + c * (RPT * 16),
        block.template select<RPT * 16, 1>(0),
        overaligned_tag<16>{});
  }
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
    const uint8_t* kState,
    const uint8_t* vState,
    uint32_t kvLen,
    uint32_t headQ,
    uint32_t headKv,
    sycl::nd_item<1>& ndi) {
  constexpr int BN = DG2V4_BN;
  constexpr int KG = BN / 8;     // K row groups of 8
  constexpr int VP = BN / 2;     // V row pairs
  constexpr int DCHUNKS = 4;
  constexpr int WG = 32;
  const int gid = static_cast<int>(ndi.get_group_linear_id());
  const int lid = static_cast<int>(ndi.get_local_linear_id());
  const int nTiles = (static_cast<int>(kvLen) + BN - 1) / BN;
  const int headIdx = gid / nTiles;
  const int tile = gid % nTiles;
  const int kvBase = tile * BN;
  const size_t tileKOff =
      (static_cast<size_t>(headIdx) * nTiles + tile) * BN * 32;
  const size_t tileVOff =
      (static_cast<size_t>(headIdx) * nTiles + tile) * BN * 32;

  // ---- K pack: thread -> (kvGroup = lid/4, dChunks = (lid%4)*2, +1) ----
  {
#pragma unroll
    for (int g = lid / 4; g < KG; g += WG / 4) {
      const int c = lid % 4;
      simd<ElemT, 128> rows = 0;
#pragma unroll
      for (int r = 0; r < 8; r++) {
        const int row = kvBase + g * 8 + r;
        simd<ElemT, 16> chunk = 0;
        if (row < static_cast<int>(kvLen)) {
          chunk = block_load<ElemT, 16>(
              reinterpret_cast<const ElemT*>(kState) +
                  static_cast<size_t>(row) * headKv * 64 +
                  static_cast<size_t>(headIdx) * 64 + c * 16,
              overaligned_tag<16>{});
        }
        rows.template select<16, 1>(r * 16) = chunk;
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
                  static_cast<size_t>(row0) * headKv * 64 +
                  static_cast<size_t>(headIdx) * 64 + c * 16,
              overaligned_tag<16>{});
        }
        if (row1 < static_cast<int>(kvLen)) {
          hi = block_load<ElemT, 16>(
              reinterpret_cast<const ElemT*>(vState) +
                  static_cast<size_t>(row1) * headKv * 64 +
                  static_cast<size_t>(headIdx) * 64 + c * 16,
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
    const uint8_t* packedQ,
    const uint8_t* kState,
    const uint8_t* vState,
    const uint8_t* packedK,
    const uint8_t* packedV,
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
  // Fused small-shape path keeps RPT=4 (register-assembled A); the
  // two-kernel path reads a pre-packed A layout and runs RPT=DG2V4D64_RPT.
  constexpr int RPT = FUSED ? DG2V4D64_FUSED_RPT : DG2V4D64_RPT;
  constexpr int QGRP = 32 * RPT;    // query rows per work-group
  // The fused single-kernel path (small qLen) packs K/V per work-group from
  // the raw row-major buffers and therefore uses BN=64 so the kvZero flags
  // fit in SLM; the two-kernel path keeps BN=DG2V4_BN (64).
  constexpr int BN = FUSED ? 64 : DG2V4_BN;
  constexpr int KG = BN / 8;        // K row groups of 8
  constexpr int PG = BN / 16;       // V row groups of 16
  constexpr int DCHUNKS = 4;
  constexpr int WG = 32;
  // K and V are staged into separate regions at the top of the tile, so the
  // QK -> S*V transition needs no barrier (3 barriers/tile -> 2). At BN=128
  // this uses the full 64 KB per-WG SLM cap (2 * 32 KB).
  constexpr int KSTAGE = 0;
  constexpr int VSTAGE = BN * 32 * 4;
  // No flags region: the host passes the ORIGINAL kv_len, so padded zero
  // rows are masked by the row-index check alone.
  constexpr int SLM_TOTAL = 2 * BN * 32 * 4;
  constexpr float SCALE = 0.125f;
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
                        (static_cast<size_t>(qrow) * headQ + headIdx) * 64;
#pragma unroll
      for (int c = 0; c < DCHUNKS; c++) {
        qChunkAll.template select<16, 1>(c * RPT * 16 + r * 16) =
            block_load<ElemT, 16>(qp + c * 16, overaligned_tag<16>{});
      }
    }
  }

  float mArr[RPT];
  float lArr[RPT];
  simd<float, RPT * 64> acc = 0;
  // pChunk holds only the PG kv-row groups; QK writes scaled scores here,
  // softmax converts/exp2s in place, and S*V reads the same 16-kv-row A
  // operand for every d chunk.
  simd<ElemT, RPT * PG * 16> pChunkAll = 0;
#pragma unroll
  for (int r = 0; r < RPT; r++) {
    mArr[r] = -3.402823466e+38f;
    lArr[r] = 0.0f;
  }

  const size_t tileKOff = static_cast<size_t>(headIdx) * nTiles * BN * 32;
  const size_t tileVOff = static_cast<size_t>(headIdx) * nTiles * BN * 32;

  for (int t = 0; t < nTiles; t++) {
    // ---- Stage packed K and V into separate SLM regions ----
    {
      if constexpr (FUSED) {
        // Pack this 64-row tile directly from the raw K/V buffers into SLM.
        // K: thread -> (kvGroup = lid/4, dChunks = (lid%4)*2, +1).
#pragma unroll
        for (int g = lid / 4; g < KG; g += WG / 4) {
          const int c = lid % 4;
          simd<ElemT, 128> rows = 0;
#pragma unroll
          for (int r = 0; r < 8; r++) {
            const int row = t * BN + g * 8 + r;
            simd<ElemT, 16> chunk = 0;
            if (row < static_cast<int>(kvLen)) {
              chunk = block_load<ElemT, 16>(
                  reinterpret_cast<const ElemT*>(kState) +
                      static_cast<size_t>(row) * headKv * 64 +
                      static_cast<size_t>(headIdx) * 64 + c * 16,
                  overaligned_tag<16>{});
            }
            rows.template select<16, 1>(r * 16) = chunk;
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
        // V: thread = row pair, packs 8 words per (dChunk, dHalf).
        const int pair = lid;
        const int row0 = t * BN + pair * 2;
        const int row1 = row0 + 1;
#pragma unroll
        for (int c = 0; c < DCHUNKS; c++) {
          simd<ElemT, 16> lo = 0;
          simd<ElemT, 16> hi = 0;
          if (row0 < static_cast<int>(kvLen)) {
            lo = block_load<ElemT, 16>(
                reinterpret_cast<const ElemT*>(vState) +
                    static_cast<size_t>(row0) * headKv * 64 +
                    static_cast<size_t>(headIdx) * 64 + c * 16,
                overaligned_tag<16>{});
          }
          if (row1 < static_cast<int>(kvLen)) {
            hi = block_load<ElemT, 16>(
                reinterpret_cast<const ElemT*>(vState) +
                    static_cast<size_t>(row1) * headKv * 64 +
                    static_cast<size_t>(headIdx) * 64 + c * 16,
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
      const size_t tkOff = tileKOff + static_cast<size_t>(t) * BN * 32;
#pragma unroll
      for (int i = 0; i < BN / 64; i++) {
        simd<uint32_t, 64> words = block_load<uint32_t, 64>(
            reinterpret_cast<const uint32_t*>(packedK) + tkOff +
                (lid * (BN / 64) + i) * 64,
            overaligned_tag<16>{});
        slm_block_store(
            KSTAGE + (lid * (BN / 64) + i) * 64 * 4,
            words,
            overaligned_tag<16>{});
      }
      const size_t tvOff = tileVOff + static_cast<size_t>(t) * BN * 32;
#pragma unroll
      for (int i = 0; i < BN / 64; i++) {
        simd<uint32_t, 64> words = block_load<uint32_t, 64>(
            reinterpret_cast<const uint32_t*>(packedV) + tvOff +
                (lid * (BN / 64) + i) * 64,
            overaligned_tag<16>{});
        slm_block_store(
            VSTAGE + (lid * (BN / 64) + i) * 64 * 4,
            words,
            overaligned_tag<16>{});
      }
      }
    }
    barrier();

    if (stageOnly) {
      continue;
    }

    // ---- QK^T: BN scores per query row (kept in registers) ----
#pragma unroll
    for (int g = 0; g < KG; g++) {
      simd<float, 64> c64 = 0;
#pragma unroll
      for (int c = 0; c < DCHUNKS; c++) {
        simd<ElemT, 128> a = 0;
        if constexpr (FUSED) {
          a.template select<RPT * 16, 1>(0) =
              qChunkAll.template select<RPT * 16, 1>(c * RPT * 16);
        } else {
          const size_t qPackBase =
              (static_cast<size_t>(headIdx) * qTiles + qTile) * WG + lid;
          // D64 packs RPT real rows per chunk (RPT*16 elements); zero-extend
          // to the full 8-row dpas A operand like the fused path.
          a.template select<RPT * 16, 1>(0) = block_load<ElemT, RPT * 16>(
              reinterpret_cast<const ElemT*>(packedQ) +
                  qPackBase * (RPT * 64) + c * (RPT * 16),
              overaligned_tag<16>{});
        }
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
          if (t * BN + j >= static_cast<int>(kvLen)) {
            s = MASKED_SCORE;
          }
          // Store scores chunk-major directly in pChunk: [j/16][r][j%16].
          pChunkAll[(j / 16) * RPT * 16 + r * 16 + (j % 16)] =
              static_cast<ElemT>(s);
        }
      }
    }

    // ---- S*V (V already staged into its own SLM region) ----
    {
      if (qkOnly) {
#if defined(DG2V4_DEBUG)
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
          // 3. SC scores (scaled, masked), rows 0..RPT-1 x BN kv, per tile
          w = 4608 + t * 64;
          {
            for (int r = 0; r < RPT; r++) {
              for (int j = 0; j < BN; j++) {
                dbg[w++] = static_cast<float>(
                    static_cast<ElemT>(pChunkAll[(j / 16) * RPT * 16 +
                                                 r * 16 + (j % 16)]));
              }
            }
          }
        }
#endif
        continue;
      }
    }

    if (!svOnly) {
      // ---- Two-pass softmax per row (scores/P share pChunk) ----
#pragma unroll
      for (int r = 0; r < RPT; r++) {
        simd<float, BN> svec = 0;
#pragma unroll
        for (int p = 0; p < PG; p++) {
          simd<ElemT, 16> sh =
              pChunkAll.template select<16, 1>(p * RPT * 16 + r * 16);
          svec.template select<16, 1>(p * 16) = convert<float>(sh);
        }
        // Vector max reduction instead of BN scalar compares.
        const float mTile = __ESIMD_NS::detail::reduce<
            float, float, BN,
            __ESIMD_NS::detail::esimd_apply_reduced_max>(svec);
        // Stable online update: never exponentiate a positive difference.
        // If the new tile lowers the row max, scale this tile's p/l by
        // exp(m_new - m_old) (<=1); if it raises the max, scale the previous
        // accumulator/l by exp(m_old - m_new) (<=1). Every rescale is <=1,
        // so large score spans cannot overflow fp32.
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
        float lTile = 0.0f;
        svec = (svec - mTile) * LOG2E;
        simd<float, BN> pvec = __ESIMD_NS::exp2<float, BN>(svec);
#pragma unroll
        for (int p = 0; p < PG; p++) {
          simd<float, 16> p16 = pvec.template select<16, 1>(p * 16);
          simd<ElemT, 16> ph = convert<ElemT>(p16 * rescaleCur);
          lTile += __ESIMD_NS::detail::reduce<
              float, float, 16, __ESIMD_NS::detail::esimd_apply_sum>(p16);
          pChunkAll.template select<16, 1>(p * RPT * 16 + r * 16) = ph;
        }
        lArr[r] = lArr[r] * rescalePrev + lTile * rescaleCur;
        mArr[r] = mTile > oldM ? mTile : oldM;
        acc.template select<64, 1>(r * 64) =
            acc.template select<64, 1>(r * 64) * rescalePrev;
      }
    } else {
      // S*V isolation test: zero p so the DPAS output is 0.
#pragma unroll
      for (int i = 0; i < RPT * PG * 16; i++) {
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
              acc.template select<8, 1>(r * 64 + c * 16);
          accH1.template select<8, 1>(r * 8) =
              acc.template select<8, 1>(r * 64 + c * 16 + 8);
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
          acc.template select<8, 1>(r * 64 + c * 16) =
              accH0.template select<8, 1>(r * 8);
          acc.template select<8, 1>(r * 64 + c * 16 + 8) =
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
          normAlpha + static_cast<size_t>(headIdx) * 64;
      ElemT* op = reinterpret_cast<ElemT*>(out) +
                  (static_cast<size_t>(qrow) * headQ + headIdx) * 64;
#pragma unroll
      for (int c = 0; c < DCHUNKS; c++) {
        simd<float, 16> outv =
            acc.template select<16, 1>(r * 64 + c * 16);
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
struct V4Buffers {
  void* packedQ = nullptr;
  void* packedK = nullptr;
  void* packedV = nullptr;
};

inline std::unordered_map<uint64_t, V4Buffers>& v4_cache() {
  static std::unordered_map<uint64_t, V4Buffers> cache;
  return cache;
}

// H3-style seq=20683 needs ~0.9 GB of packed buffers per shape. Keep only
// the most recent shape so a second workflow run does not pile up device
// memory on top of an already VRAM-pressured ComfyUI model cache.
constexpr size_t V4_CACHE_MAX = 1;

inline std::vector<uint64_t>& v4_lru_order() {
  static std::vector<uint64_t> order;
  return order;
}

inline void v4_touch(uint64_t key) {
  auto& order = v4_lru_order();
  for (auto it = order.begin(); it != order.end(); ++it) {
    if (*it == key) {
      order.erase(it);
      break;
    }
  }
  order.push_back(key);
}

inline std::mutex& v4_mutex() {
  static std::mutex m;
  return m;
}

inline void clearV4Cache(sycl::queue& queue) {
  std::lock_guard<std::mutex> guard(v4_mutex());
  // Sidecar kernels are async; never free USM an in-flight kernel may read.
  queue.wait();
  for (auto& entry : v4_cache()) {
    sycl::free(entry.second.packedQ, queue);
    sycl::free(entry.second.packedK, queue);
    sycl::free(entry.second.packedV, queue);
  }
  v4_cache().clear();
  v4_lru_order().clear();
}

template <typename ElemT>
inline void runSdpV4(
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
  constexpr int BN = DG2V4_BN;
  const int nTiles = (kvLen + BN - 1) / BN;
  const size_t tileWords = static_cast<size_t>(BN) * 32;
  // Cache identity includes headKv too: packQ depends on headQ/qLen, while
  // packK/V buffer sizes depend on headKv/kvLen. Reusing across a GQA head
  // count change would alias differently-sized buffers.
  const uint64_t key =
      (static_cast<uint64_t>(headQ) << 48) |
      (static_cast<uint64_t>(headKv) << 40) |
      (static_cast<uint64_t>(static_cast<uint32_t>(qLen)) << 16) |
      static_cast<uint32_t>(kvLen);

  // Fused uses RPT=DG2V4D64_FUSED_RPT; non-fused uses RPT=DG2V4D64_RPT.
  const int qTilesFused =
      (qLen + 32 * DG2V4D64_FUSED_RPT - 1) / (32 * DG2V4D64_FUSED_RPT);
  const int qTilesPack =
      (qLen + 32 * DG2V4D64_RPT - 1) / (32 * DG2V4D64_RPT);
  const int packGroups = headQ * nTiles;
  const int attnGroups = headQ * qTilesPack;
  constexpr int WG = 32;

#if defined(DG2V4_DEBUG)
  const bool attnOnly = std::getenv("DG2V4_ATTN_ONLY") != nullptr;
  const bool packOnly = std::getenv("DG2V4_PACK_ONLY") != nullptr;
  const bool stageOnly = std::getenv("DG2V4_ATTN_STAGE_ONLY") != nullptr;
  const bool qkOnly = std::getenv("DG2V4_ATTN_QK_ONLY") != nullptr;
  const bool svOnly = std::getenv("DG2V4_ATTN_SV_ONLY") != nullptr;
  const bool svRegAcc = std::getenv("DG2V4_ATTN_SV_REGACC") != nullptr;
  const bool svConstA = std::getenv("DG2V4_ATTN_SV_CONSTA") != nullptr;
  const bool dumpQk = std::getenv("DG2V4_DUMP_QK") != nullptr;
  const bool dumpState = std::getenv("DG2V4_DUMP_STATE") != nullptr;
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
  // Fused packing repacks K/V once per qTile work-group, so its redundancy
  // is qTiles * kvTiles (BN=64 tile count). Keep fused only when that
  // product is small; otherwise the two-kernel path wins because K/V are
  // packed once globally.
  const int kvTilesFused = (kvLen + 63) / 64;
  const bool useFused = qLen <= DG2V4_FUSED_MAX_Q &&
                        qTilesFused * kvTilesFused <= DG2V4_FUSED_MAX_TILES &&
                        !attnOnly && !packOnly;
  V4Buffers buf;
  if (!useFused) {
    {
      std::lock_guard<std::mutex> guard(v4_mutex());
      auto it = v4_cache().find(key);
      if (it == v4_cache().end()) {
        // Bound the sidecar cache: H3-style runs at seq=20685 already hold
        // ~0.9 GB of packed Q/K/V per shape; letting every new shape pile up
        // pushes VRAM pressure and can make the next workflow run slower.
        while (v4_lru_order().size() >= V4_CACHE_MAX &&
               v4_cache().size() >= V4_CACHE_MAX) {
          const uint64_t victim = v4_lru_order().front();
          if (victim == key) {
            break;
          }
          auto vit = v4_cache().find(victim);
          if (vit != v4_cache().end()) {
            // Do not free USM that an earlier async submission may still be
            // reading; shape-change evictions are rare enough to pay a sync.
            queue.wait();
            sycl::free(vit->second.packedQ, queue);
            sycl::free(vit->second.packedK, queue);
            sycl::free(vit->second.packedV, queue);
            v4_cache().erase(vit);
          }
          auto& order = v4_lru_order();
          if (!order.empty()) {
            order.erase(order.begin());
          }
        }
        buf.packedQ = sycl::aligned_alloc_device(
            64,
            static_cast<size_t>(headQ) * qTilesPack * WG *
                (DG2V4D64_RPT * 64) * sizeof(ElemT),
            queue);
        buf.packedK = sycl::aligned_alloc_device(
            64,
            static_cast<size_t>(headQ) * nTiles * tileWords * 4,
            queue);
        buf.packedV = sycl::aligned_alloc_device(
            64,
            static_cast<size_t>(headQ) * nTiles * tileWords * 4,
            queue);
        v4_cache()[key] = buf;
        v4_touch(key);
      } else {
        buf = it->second;
        v4_touch(key);
      }
    }
  }
  float* dbgBuf = nullptr;
#if defined(DG2V4_DEBUG)
  if (dumpQk) {
    dbgBuf = static_cast<float*>(sycl::aligned_alloc_device(
        64, 8192 * sizeof(float), queue));
  }
#endif
  if (!attnOnly && !useFused) {
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(headQ * qTilesPack * WG),
                            sycl::range<1>(WG)),
          [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            packQDg2<ElemT>(
                static_cast<uint8_t*>(buf.packedQ),
                static_cast<const uint8_t*>(q),
                static_cast<uint32_t>(qLen),
                static_cast<uint32_t>(headQ),
                ndi);
          });
    });
    queue.submit([&](sycl::handler& cgh) {
      cgh.parallel_for(
          sycl::nd_range<1>(sycl::range<1>(packGroups * WG),
                            sycl::range<1>(WG)),
          [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
            packKvDg2<ElemT>(
                static_cast<uint8_t*>(buf.packedK),
                static_cast<uint8_t*>(buf.packedV),
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
            sycl::nd_range<1>(sycl::range<1>(headQ * qTilesFused * WG),
                              sycl::range<1>(WG)),
            [=](sycl::nd_item<1> ndi) SYCL_ESIMD_KERNEL {
              attnDg2<ElemT, true>(
                  static_cast<const uint8_t*>(q),
                  nullptr,
                  static_cast<const uint8_t*>(k),
                  static_cast<const uint8_t*>(v),
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
                  nullptr,
                  static_cast<const uint8_t*>(buf.packedQ),
                  nullptr,
                  nullptr,
                  static_cast<const uint8_t*>(buf.packedK),
                  static_cast<const uint8_t*>(buf.packedV),
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
#if defined(DG2V4_DEBUG)
  queue.wait();
#else
  // Async dispatch: the caller (torch stream) synchronizes. Keeping the
  // queue non-blocked lets consecutive SDP calls overlap host submit with
  // device execution, matching torch op semantics. Debug paths below need
  // the synchronous wait so their memcpy dumps observe completed kernels.
#endif
#if defined(DG2V4_DEBUG)
  if (dumpQk && dbgBuf) {
    std::vector<float> dbg(8192);
    queue.memcpy(dbg.data(), dbgBuf, 8192 * sizeof(float)).wait();
    std::ofstream fqk("C:/Temp/dg2v4_qk.bin", std::ios::binary);
    fqk.write(reinterpret_cast<const char*>(dbg.data()), 8192 * sizeof(float));
    fqk.close();
    sycl::free(dbgBuf, queue);
  }
#endif
#if defined(DG2V4_DEBUG)
  if (std::getenv("DG2V4_DUMP_PACKED") != nullptr && !attnOnly) {
    std::vector<uint32_t> dumpK(4096), dumpV(4096);
    queue.memcpy(dumpK.data(), buf.packedK, 4096 * 4).wait();
    queue.memcpy(dumpV.data(), buf.packedV, 4096 * 4).wait();
    std::ofstream fk("C:/Temp/dg2v4_packedK.bin", std::ios::binary);
    fk.write(reinterpret_cast<const char*>(dumpK.data()), 4096 * 4);
    fk.close();
    std::ofstream fv("C:/Temp/dg2v4_packedV.bin", std::ios::binary);
    fv.write(reinterpret_cast<const char*>(dumpV.data()), 4096 * 4);
    fv.close();
  }
#endif
}

}  // namespace dg2v4d64
