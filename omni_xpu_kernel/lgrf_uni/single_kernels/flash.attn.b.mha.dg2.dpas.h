// DG2-native Flash Attention v2 — ESIMD DPAS (XMX) QK^T and S*V.
//
// v1 (flash.attn.b.mha.dg2.h) is correct and stable but 5-10x slower than
// Torch SDPA because QK^T/S*V are per-element FMAs. v2 keeps the A770-safe
// geometry (WG=32, one query row per thread, no 2D LSC, no named barriers)
// and moves both matmuls to the measured A770 DPAS contract:
//
//   dpas<8, 1, float>(c, b, a)
//     A = 1 x 16 (fp16/bf16, row-major)
//     B = 16 x 8 (fp16/bf16, VNNI-packed, 64 u32 words)
//     C = 1 x 8  (fp32 accumulator)
//
// VNNI word layout (from the measured A770 GEMM contract):
//   word(k/2, n) = low16(B[2*(k/2)][n]) | high16(B[2*(k/2)+1][n])
//
// SLM (single-buffered, 33 KB total):
//   packedK [(kvGroup=8) x (dChunk=8) x 64 words]  16 KB
//     block (g,c): word = (d/2)*8 + kv_local
//   packedV [(p=4) x (dChunk=8) x (dHalf=2) x 64 words]  16 KB
//     block (p,c,h): word = (kvPair)*8 + d, kvPair in 0..7
//   kvZero  [64 x u32]                              256 B
//
// Online softmax is two-pass per tile (max, then exp2) with cross-tile
// rescaling in fp32. Only HD=128 is implemented; D64 keeps the v1 kernel.

#include <type_traits>

namespace dg2dpas {

template <typename T> struct IsFp16Dpas : std::false_type {};
template <> struct IsFp16Dpas<sycl::half> : std::true_type {};

// VNNI-pack 8 output columns from two 16-element rows.
// w[i] = (low 16 bits of lo[i]) | (high 16 bits of hi[i]).
template <typename ElemT>
ESIMD_INLINE simd<uint32_t, 8> pack8(const simd<ElemT, 16>& lo,
                                     const simd<ElemT, 16>& hi) {
  simd<uint32_t, 8> w;
#pragma unroll
  for (int i = 0; i < 8; i++) {
    const uint32_t a = static_cast<uint32_t>(sycl::bit_cast<uint16_t>(lo[i]));
    const uint32_t b = static_cast<uint32_t>(sycl::bit_cast<uint16_t>(hi[i]));
    w[i] = a | (b << 16);
  }
  return w;
}

// dpas<8, 8, float> with the single query row (or 16 p values) replicated
// across the 8 DPAS rows. Only output row 0 is used. M=8 is the measured
// A770 form; M=1 produced score noise in validation.
template <typename ElemT>
ESIMD_INLINE simd<float, 8> dpas8x16x8(const simd<float, 8>& c,
                                       const simd<uint32_t, 64>& braw,
                                       const simd<ElemT, 16>& a16) {
  simd<uint32_t, 64> brawLocal = braw;
  simd<ElemT, 128> b = brawLocal.template bit_cast_view<ElemT>();
  simd<ElemT, 128> a;
#pragma unroll
  for (int r = 0; r < 8; r++) {
    a.template select<16, 1>(r * 16) = a16;
  }
  simd<float, 64> c64 = 0;
  c64.template select<8, 1>(0) = c;
  simd<float, 64> out64 = dpas<8, 8, float>(c64, b, a);
  return out64.template select<8, 1>(0);
}

template <typename ElemT, int HD>
ESIMD_INLINE void flashAttnDg2DpasPrecomputed(
    uint8_t* qState,
    uint8_t* kState,
    uint8_t* vState,
    uint8_t* normAlpha,
    uint8_t* out,
    uint32_t activationLength,
    uint32_t kvSeqLen,
    uint32_t headQ,
    uint32_t headKv,
    sycl::nd_item<2>& ndi) {
  static_assert(HD == 128, "v2 DPAS kernel is implemented for HD=128 only");
  constexpr int BM = 32;
  constexpr int BN = 64;
  constexpr int WG = 32;
  constexpr int DCHUNKS = HD / 16;   // 8
  constexpr int KGROUPS = BN / 8;    // 8 kv groups of 8 rows
  constexpr int VPAIRS = BN / 2;     // 32 row pairs
  constexpr int SLM_K_BYTES = KGROUPS * DCHUNKS * 64 * 4;  // 16 KB
  constexpr int SLM_V_BYTES = 4 * DCHUNKS * 2 * 64 * 4;    // 16 KB
  constexpr int SLM_ZERO_BYTES = BN * 4;                   // 256 B
  constexpr int SLM_TOTAL_BYTES = SLM_K_BYTES + SLM_V_BYTES + SLM_ZERO_BYTES;
  constexpr float SCALE = 0.08838834764831844f;  // 1/sqrt(128)
  constexpr float LOG2E = 1.4426950408889634f;
  constexpr float MASKED_SCORE = -1.0e30f;

  const int lid = static_cast<int>(ndi.get_local_id(0));
  const int headIdx = static_cast<int>(ndi.get_group(0)) % static_cast<int>(headQ);
  const int qTile = static_cast<int>(ndi.get_group(1));
  const int qrow = qTile * BM + lid;

  slm_init(SLM_TOTAL_BYTES);

  // Query row in fp16/bf16 registers (A operand for QK^T).
  simd<ElemT, HD> qReg = 0;
  if (qrow < static_cast<int>(activationLength)) {
    const ElemT* qp = reinterpret_cast<const ElemT*>(qState) +
                      (static_cast<size_t>(qrow) * headQ + headIdx) * HD;
#pragma unroll
    for (int c = 0; c < DCHUNKS; c++) {
      qReg.template select<16, 1>(c * 16) =
          block_load<ElemT, 16>(qp + c * 16, overaligned_tag<16>{});
    }
  }

  const ElemT* kBase = reinterpret_cast<const ElemT*>(kState);
  const ElemT* vBase = reinterpret_cast<const ElemT*>(vState);
  const size_t kvRowStride = static_cast<size_t>(headKv) * HD;
  const int nkvTiles = (static_cast<int>(kvSeqLen) + BN - 1) / BN;

  simd<float, HD> acc = 0;
  float m = -3.402823466e+38f;
  float l = 0.0f;

  for (int t = 0; t < nkvTiles; t++) {
    // ---- Pack K/V tiles into VNNI SLM layout (uniform bounds) ----
    {
      int tileValidRows = static_cast<int>(kvSeqLen) - t * BN;
      if (tileValidRows > BN) {
        tileValidRows = BN;
      }
      if (tileValidRows < 0) {
        continue;
      }
      const int kvBase = t * BN;

      // K: thread -> (kvGroup = lid/4, dChunks = (lid%4)*2, +1). Each thread
      // loads 8 rows x 16 d per chunk, packs 64 VNNI words (word = (d/2)*8 +
      // kv_local), and computes per-row |K| sums for zero-row masking.
      {
        const int g = lid / 4;
        const int c0 = (lid % 4) * 2;
        simd<float, 8> rowAbs = 0;
        bool firstChunk = true;
#pragma unroll
        for (int cc = 0; cc < 2; cc++) {
          const int c = c0 + cc;
          simd<ElemT, 128> rows = 0;
#pragma unroll
          for (int r = 0; r < 8; r++) {
            const int row = kvBase + g * 8 + r;
            simd<ElemT, 16> chunk = 0;
            if (row < static_cast<int>(kvSeqLen)) {
              chunk = block_load<ElemT, 16>(
                  kBase + static_cast<size_t>(row) * kvRowStride +
                      static_cast<size_t>(headIdx) * HD + c * 16,
                  overaligned_tag<16>{});
            }
            rows.template select<16, 1>(r * 16) = chunk;
            if (firstChunk) {
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
              (g * DCHUNKS + c) * 64 * 4, words, overaligned_tag<16>{});
          firstChunk = false;
        }
        simd<uint32_t, 8> flagGroup;
#pragma unroll
        for (int r = 0; r < 8; r++) {
          flagGroup[r] = (rowAbs[r] > 0.0f) ? 1u : 0u;
        }
        slm_block_store(
            SLM_K_BYTES + SLM_V_BYTES + g * 8 * 4,
            flagGroup,
            overaligned_tag<16>{});
      }

      // V: thread = row pair (2 rows), packs 8 words per (dChunk, dHalf).
      // Block (c,h) is 64 words; word = kvPair*8 + d.
      {
        const int pair = lid;
        const int row0 = kvBase + pair * 2;
        const int row1 = row0 + 1;
#pragma unroll
        for (int c = 0; c < DCHUNKS; c++) {
          simd<ElemT, 16> lo = 0;
          simd<ElemT, 16> hi = 0;
          if (row0 < static_cast<int>(kvSeqLen)) {
            lo = block_load<ElemT, 16>(
                vBase + static_cast<size_t>(row0) * kvRowStride +
                    static_cast<size_t>(headIdx) * HD + c * 16,
                overaligned_tag<16>{});
          }
          if (row1 < static_cast<int>(kvSeqLen)) {
            hi = block_load<ElemT, 16>(
                vBase + static_cast<size_t>(row1) * kvRowStride +
                    static_cast<size_t>(headIdx) * HD + c * 16,
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
            simd<uint32_t, 8> w = pack8(loHalf, hiHalf);
            slm_block_store(
                SLM_K_BYTES +
                    (((pair / 8) * DCHUNKS + c) * 2 + h) * 64 * 4 +
                    (pair % 8) * 8 * 4,
                w,
                overaligned_tag<16>{});
          }
        }
      }
    }

    barrier();

    // ---- QK^T via DPAS: 64 scores ----
    float scores[BN];
#pragma unroll
    for (int g = 0; g < KGROUPS; g++) {
      simd<float, 8> scoreG = 0;
#pragma unroll
      for (int c = 0; c < DCHUNKS; c++) {
        simd<uint32_t, 64> braw = slm_block_load<uint32_t, 64>(
            (g * DCHUNKS + c) * 64 * 4, overaligned_tag<16>{});
        simd<ElemT, 16> a = qReg.template select<16, 1>(c * 16);
        scoreG = dpas8x16x8<ElemT>(scoreG, braw, a);
      }
#pragma unroll
      for (int i = 0; i < 8; i++) {
        scores[g * 8 + i] = static_cast<float>(scoreG[i]) * SCALE;
      }
    }

    // Mask zero-K rows (padding) and out-of-range rows.
#pragma unroll
    for (int j = 0; j < BN; j++) {
      simd<uint32_t, 8> flagGroup = slm_block_load<uint32_t, 8>(
          SLM_K_BYTES + SLM_V_BYTES + (j / 8) * 8 * 4,
          overaligned_tag<16>{});
      const int absRow = t * BN + j;
      if (!(static_cast<uint32_t>(flagGroup[j % 8]) != 0u &&
            absRow < static_cast<int>(kvSeqLen))) {
        scores[j] = MASKED_SCORE;
      }
    }

    // ---- Two-pass online softmax for this tile ----
    float mTile = -3.402823466e+38f;
#pragma unroll
    for (int j = 0; j < BN; j++) {
      mTile = (scores[j] > mTile) ? scores[j] : mTile;
    }
    float lTile = 0.0f;
    simd<ElemT, BN> pAll = 0;
#pragma unroll
    for (int j = 0; j < BN; j++) {
      const float p = sycl::exp2((scores[j] - mTile) * LOG2E);
      lTile += p;
      pAll[j] = static_cast<ElemT>(p);
    }
    const float rescale = sycl::exp2((m - mTile) * LOG2E);
    l = l * rescale + lTile;
    acc = acc * rescale;
    m = mTile;

    // ---- S*V via DPAS: 4 row-pairs-of-16 x 8 d-chunks x 2 halves ----
#pragma unroll
    for (int p = 0; p < 4; p++) {
      simd<ElemT, 16> a = pAll.template select<16, 1>(p * 16);
#pragma unroll
      for (int c = 0; c < DCHUNKS; c++) {
#pragma unroll
        for (int h = 0; h < 2; h++) {
          simd<uint32_t, 64> braw = slm_block_load<uint32_t, 64>(
              SLM_K_BYTES + ((p * DCHUNKS + c) * 2 + h) * 64 * 4,
              overaligned_tag<16>{});
          simd<float, 8> acc8 = acc.template select<8, 1>(c * 16 + h * 8);
          acc8 = dpas8x16x8<ElemT>(acc8, braw, a);
          acc.template select<8, 1>(c * 16 + h * 8) = acc8;
        }
      }
    }

    barrier();
  }

  // ---- Output normalization + store ----
  if (qrow < static_cast<int>(activationLength)) {
    simd<float, HD> outv = acc * (1.0f / l);
    if constexpr (IsFp16Dpas<ElemT>::value) {
      outv.merge(65504.0f, outv > 65504.0f);
      outv.merge(-65504.0f, outv < -65504.0f);
    }
    const float* alphaBase = reinterpret_cast<const float*>(normAlpha) +
                             static_cast<size_t>(headIdx) * HD;
    ElemT* op = reinterpret_cast<ElemT*>(out) +
                (static_cast<size_t>(qrow) * headQ + headIdx) * HD;
#pragma unroll
    for (int c = 0; c < DCHUNKS; c++) {
      simd<float, 16> alpha =
          block_load<float, 16>(alphaBase + c * 16, overaligned_tag<16>{});
      simd<float, 16> o16 = outv.template select<16, 1>(c * 16) * alpha;
      if constexpr (IsFp16Dpas<ElemT>::value) {
        o16.merge(65504.0f, o16 > 65504.0f);
        o16.merge(-65504.0f, o16 < -65504.0f);
      }
      block_store<ElemT, 16>(op + c * 16, simd<ElemT, 16>(o16),
                             overaligned_tag<16>{});
    }
  }
}

}  // namespace dg2dpas
