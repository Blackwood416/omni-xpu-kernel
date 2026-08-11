// DG2-native ESIMD Flash Attention — fp16/bf16 I/O, fp32 accumulation.
//
// A770/DG2 constraints learned from the preserved negative results:
//   * Xe2 lgrf SDP family: fp16 S*V DPAS accumulator rejected by DG2 VISA;
//     after switching to fp32 accumulation it device-lost on A770.
//   * ESIMD work-group size 512 triggers GPU TDR (LiveKernelEvent 141) on
//     A770 with driver 32.0.101.8860 — verified with a trivial slm_init
//     kernel. This kernel therefore uses WG=32 (one thread per query row),
//     matching the measured-safe ESIMD work-group sizes on A770.
//   * 2D LSC loads/named barriers are avoided (measured A770 hazards).
//
// Structure: BM=32 query rows per work-group, BN=32 KV rows per SLM tile,
// one thread per query row with the full head dim in registers. K/V tiles
// are staged cooperatively through SLM; online max/sum runs in fp32; the
// output is (acc / l) * normAlpha, clamped for fp16.
//
// Layout [B=1, L, H, D] contiguous for Q/K/V/O; headQ == headKv (enforced by
// the torch wrapper). KV rows beyond kvSeqLen are masked with a large
// negative score (-1e30) instead of -inf so exp2 never sees NaN.

#include <type_traits>

// Work-group geometry shared with the sidecar entry points.
constexpr int DG2_BM = 32;
constexpr int DG2_BN = 32;
constexpr int DG2_WG = 32;

inline sycl::nd_range<2> flash_ndr(int q_len, int headQ, int /*head_dim*/) {
  const int qTiles = (q_len + DG2_BM - 1) / DG2_BM;
  return sycl::nd_range<2>(
      {(size_t)(DG2_WG * headQ), (size_t)qTiles}, {(size_t)DG2_WG, 1});
}

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

template <typename ElemT, int HD>
ESIMD_INLINE void flashAttnDg2Precomputed(
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
  constexpr int BM = DG2_BM;
  constexpr int BN = DG2_BN;
  constexpr int WG = DG2_WG;
  constexpr int SLM_K_BYTES = BN * HD * static_cast<int>(sizeof(ElemT));
  constexpr int SLM_TOTAL_BYTES = 2 * SLM_K_BYTES;
  constexpr float SCALE =
      (HD == 128) ? 0.08838834764831844f : 0.125f;  // 1/sqrt(128), 1/sqrt(64)
  constexpr float LOG2E = 1.4426950408889634f;
  constexpr float MASKED_SCORE = -1.0e30f;

  const int lid = static_cast<int>(ndi.get_local_id(0));
  const int headIdx = static_cast<int>(ndi.get_group(0)) % static_cast<int>(headQ);
  const int qTile = static_cast<int>(ndi.get_group(1));
  const int qrow = qTile * BM + lid;

  slm_init(SLM_TOTAL_BYTES);

  // Query row -> registers (fp32). OOB rows keep zeros and skip the store.
  simd<float, HD> qReg = 0;
  if (qrow < static_cast<int>(activationLength)) {
    const ElemT* qp = reinterpret_cast<const ElemT*>(qState) +
                      (static_cast<size_t>(qrow) * headQ + headIdx) * HD;
#pragma unroll
    for (int c = 0; c < HD / 16; c++) {
      simd<float, 16> chunk = block_load<ElemT, 16>(
          qp + c * 16, overaligned_tag<16>{});
      qReg.template select<16, 1>(c * 16) = chunk;
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
    // ---- Cooperative K/V tile copy into SLM (uniform bounds) ----
    {
      int tileValidRows = static_cast<int>(kvSeqLen) - t * BN;
      if (tileValidRows > BN) {
        tileValidRows = BN;
      }
      if (tileValidRows < 0) {
        continue;
      }
      const int tileValidElems = tileValidRows * HD;
      const size_t globalBase =
          (static_cast<size_t>(t) * BN * headKv + headIdx) * HD;

      for (int i = lid * 16; i < tileValidElems; i += WG * 16) {
        const int row = i / HD;
        const int col = i % HD;
        simd<ElemT, 16> kchunk = block_load<ElemT, 16>(
            kBase + globalBase + static_cast<size_t>(row) * kvRowStride + col,
            overaligned_tag<16>{});
        slm_block_store(
            i * static_cast<int>(sizeof(ElemT)), kchunk, overaligned_tag<16>{});
        simd<ElemT, 16> vchunk = block_load<ElemT, 16>(
            vBase + globalBase + static_cast<size_t>(row) * kvRowStride + col,
            overaligned_tag<16>{});
        slm_block_store(
            SLM_K_BYTES + i * static_cast<int>(sizeof(ElemT)),
            vchunk,
            overaligned_tag<16>{});
      }
      for (int i = tileValidElems + lid * 16; i < BN * HD; i += WG * 16) {
        slm_block_store(
            i * static_cast<int>(sizeof(ElemT)),
            simd<ElemT, 16>(0),
            overaligned_tag<16>{});
        slm_block_store(
            SLM_K_BYTES + i * static_cast<int>(sizeof(ElemT)),
            simd<ElemT, 16>(0),
            overaligned_tag<16>{});
      }
    }

    barrier();

    // ---- One KV tile of online softmax + S*V ----
#pragma unroll
    for (int j = 0; j < BN; j++) {
      const int absRow = t * BN + j;
      // sdp.cpp pads K/V to a multiple of 16 with zero rows and passes the
      // padded length; those rows must not consume softmax mass. A zero K
      // row cannot be distinguished from the padding by length alone, so
      // mask rows whose K row is exactly zero (the padding rows are zero in
      // both K and V). Real all-zero K rows are masked too -- an acceptable
      // v1 tradeoff, documented in the header.

      // q·k over the full head dim (fp32).
      simd<float, HD> dotv = 0;
      simd<float, HD> kAbs = 0;
#pragma unroll
      for (int c = 0; c < HD / 16; c++) {
        simd<float, 16> kf = slm_block_load<ElemT, 16>(
            (j * HD + c * 16) * static_cast<int>(sizeof(ElemT)),
            overaligned_tag<16>{});
        kAbs.template select<16, 1>(c * 16) += __ESIMD_NS::abs(kf);
        dotv.template select<16, 1>(c * 16) =
            dotv.template select<16, 1>(c * 16) +
            qReg.template select<16, 1>(c * 16) * kf;
      }
      float s = dg2_sum<HD>(dotv) * SCALE;
      const bool valid = (dg2_sum<HD>(kAbs) > 0.0f) &&
                         (absRow < static_cast<int>(kvSeqLen));
      if (!valid) {
        s = MASKED_SCORE;
      }

      const float mNew = (s > m) ? s : m;
      const float rescale = sycl::exp2((m - mNew) * LOG2E);
      const float p = sycl::exp2((s - mNew) * LOG2E);
      l = l * rescale + p;

#pragma unroll
      for (int c = 0; c < HD / 16; c++) {
        simd<float, 16> vf = slm_block_load<ElemT, 16>(
            (SLM_K_BYTES + (j * HD + c * 16) * static_cast<int>(sizeof(ElemT))),
            overaligned_tag<16>{});
        acc.template select<16, 1>(c * 16) =
            acc.template select<16, 1>(c * 16) * rescale + p * vf;
      }
      m = mNew;
    }

    barrier();
  }

  // ---- Output normalization + store ----
  if (qrow < static_cast<int>(activationLength)) {
    simd<float, HD> outv = acc * (1.0f / l);
    if constexpr (IsFp16<ElemT>::value) {
      outv.merge(65504.0f, outv > 65504.0f);
      outv.merge(-65504.0f, outv < -65504.0f);
    }
    const float* alphaBase = reinterpret_cast<const float*>(normAlpha) +
                             static_cast<size_t>(headIdx) * HD;
    ElemT* op = reinterpret_cast<ElemT*>(out) +
                (static_cast<size_t>(qrow) * headQ + headIdx) * HD;
#pragma unroll
    for (int c = 0; c < HD / 16; c++) {
      simd<float, 16> alpha =
          block_load<float, 16>(alphaBase + c * 16, overaligned_tag<16>{});
      simd<float, 16> o16 = outv.template select<16, 1>(c * 16) * alpha;
      if constexpr (IsFp16<ElemT>::value) {
        o16.merge(65504.0f, o16 > 65504.0f);
        o16.merge(-65504.0f, o16 < -65504.0f);
      }
      block_store<ElemT, 16>(op + c * 16, simd<ElemT, 16>(o16),
                             overaligned_tag<16>{});
    }
  }
}
