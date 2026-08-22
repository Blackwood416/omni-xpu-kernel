// ============================================================================
// oneDNN S8 x U4 GEMM — W4A8 primitive (also serves W4A4 s8-compute path)
// ============================================================================
//   act:        [M, K]  int8    — symmetric activation (zp=0)
//   xscales:    [M, K/gs] f32   — src scales,  attr mask (1<<0)|(1<<1), group {1,gs}
//                                  (f32 not f16: QW-class bf16 activations can
//                                   overflow an f16 scale to inf -> NaN -> black)
//   packed_u4:  [N, K/2] uint8  — u4 weights, signed->unsigned (^0x88), zp=8
//   scales_f16: [K/gs, N] f16   — wei scales, attr mask (1<<0)|(1<<1), group {gs,1}
//   out:        [M, N]  f32/f16/bf16
// ============================================================================

#include <torch/extension.h>
#include <oneapi/dnnl/dnnl.hpp>
#include <oneapi/dnnl/dnnl_sycl.hpp>

#include <cstdio>
#include <map>
#include <mutex>
#include <optional>
#include <tuple>
#include <unordered_map>

#include "utils.h"

namespace omni_xpu {
namespace svdq {

using DT = dnnl::memory::data_type;
using CacheKey = std::tuple<int64_t, int64_t, int64_t, int64_t, int>;  // M,K,N,gs,dst_dt

struct CachedPrimitive {
    dnnl::engine eng;
    dnnl::stream strm;
    dnnl::matmul prim;
    dnnl::memory::desc src_md;
    dnnl::memory::desc wei_md;
    dnnl::memory::desc dst_md;
    dnnl::memory::desc xscale_md;
    dnnl::memory::desc wscale_md;
    dnnl::memory::desc zp_md;
};

static std::map<CacheKey, CachedPrimitive> g_cache;
static std::mutex g_cache_mutex;

static std::map<int, std::pair<dnnl::engine, dnnl::stream>>& engine_map() {
    static std::map<int, std::pair<dnnl::engine, dnnl::stream>> map;
    return map;
}

static std::pair<dnnl::engine, dnnl::stream>& ensure_engine(const torch::Device& device) {
    int idx = device.index();
    auto& map = engine_map();
    auto it = map.find(idx);
    if (it != map.end()) return it->second;
    sycl::queue& q = omni_xpu::utils::get_queue(device);
    auto eng = dnnl::sycl_interop::make_engine(q.get_device(), q.get_context());
    auto strm = dnnl::sycl_interop::make_stream(eng, q);
    auto [ins, _] = map.emplace(idx, std::make_pair(std::move(eng), std::move(strm)));
    return ins->second;
}

static DT dt_of(torch::ScalarType t) {
    switch (t) {
        case torch::kFloat: return DT::f32;
        case torch::kHalf:  return DT::f16;
        default:            return DT::bf16;
    }
}

static CachedPrimitive& get_or_create(
    const CacheKey& key, int64_t M, int64_t K, int64_t N,
    int64_t act_gs, int64_t wei_gs, bool per_block_zp,
    DT dst_dt, const dnnl::engine& eng, const dnnl::stream& strm) {
    auto it = g_cache.find(key);
    if (it != g_cache.end()) return it->second;

    CachedPrimitive cp;
    cp.eng = eng;
    cp.strm = strm;
    cp.src_md   = dnnl::memory::desc({M, K}, DT::s8, dnnl::memory::format_tag::ab);
    cp.wei_md   = dnnl::memory::desc({K, N}, DT::u4, dnnl::memory::format_tag::ba);
    // dst dtype follows the caller's out_dtype: fp16 for small-scale models,
    // bf16 for wide-exponent (QW-class) models, f32 for the safest fallback.
    // fp16 must only be requested when the model values fit fp16 range.
    cp.dst_md   = dnnl::memory::desc({M, N}, dst_dt, dnnl::memory::format_tag::ab);
    const int64_t G_src = K / act_gs;
    const int64_t G_wei = K / wei_gs;
    cp.xscale_md = dnnl::memory::desc({M, G_src}, DT::f32, dnnl::memory::format_tag::ab);
    // wei scale 用 f16（与 a16 tint4 gs=128 的可用路径一致；
    // f32 wei scale 在 gs=128 下输出全零）。src scale 保持 f32 防激活溢出。
    cp.wscale_md = dnnl::memory::desc({G_wei, N}, DT::f16, dnnl::memory::format_tag::ab);
    cp.zp_md     = per_block_zp
        ? dnnl::memory::desc({G_wei, N}, DT::u8, dnnl::memory::format_tag::ab)
        : dnnl::memory::desc({1}, DT::u8, dnnl::memory::format_tag::a);

    dnnl::primitive_attr attr;
    // src scales: per-row per-K-group — mask3 {1, gs}
    // src/wei 分组解耦：src 侧 oneDNN 稳定支持 32/64；wei 侧可到 128。
    attr.set_scales(DNNL_ARG_SRC, (1 << 0) | (1 << 1), {1, act_gs}, DT::f32);
    // weight scales: per-group per-N — mask3 {gs, 1}
    attr.set_scales(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1), {wei_gs, 1}, DT::f16);
    if (per_block_zp) {
        // TINT4/torchao 非对称：per-block zero points（mask3 {gs, 1}），
        // w = (q - zp) * scale 在 oneDNN 内完成，无 Python 修正项。
        attr.set_zero_points(DNNL_ARG_WEIGHTS, (1 << 0) | (1 << 1),
                             {wei_gs, 1}, DT::u8);
    } else {
        // signed int4 -> u4 需要标量 zp=8
        attr.set_zero_points(DNNL_ARG_WEIGHTS, 0, {}, DT::u8);
    }
    // A770 u4 matmul: narrow (fp16-class) accumulation overflows to NaN when
    // K >= 5376 (Qwen-style K=12288). fpmath_mode::any with the second arg
    // true allows the bf16 accumulation the jit:gemm DPAS path needs — wide
    // exponent, no overflow (recorded A770 fix; see AI-doc int4int8).
    attr.set_fpmath_mode(dnnl::fpmath_mode::any, true);

    dnnl::matmul::primitive_desc pd(cp.eng, cp.src_md, cp.wei_md, cp.dst_md, attr);
    std::string impl_info = pd.impl_info_str();
    fprintf(stderr, "[onednn_s8u4_gemm] CACHE MISS: impl=%s (M=%lld K=%lld N=%lld act_gs=%lld wei_gs=%lld)\n",
            impl_info.c_str(), (long long)M, (long long)K, (long long)N,
            (long long)act_gs, (long long)wei_gs);
    fprintf(stderr, "[onednn_s8u4_gemm] scratchpad=%zu B (%.1f MB)\n",
            pd.scratchpad_desc().get_size(), pd.scratchpad_desc().get_size() / 1048576.0);
    if (impl_info.find("ref") != std::string::npos) {
        fprintf(stderr, "[onednn_s8u4_gemm] WARNING: reference fallback (slow)\n");
    }
    cp.prim = dnnl::matmul(pd);
    auto [ins, _] = g_cache.emplace(key, std::move(cp));
    return ins->second;
}

template <DT DstDT>
static void launch(
    const torch::Tensor& act, const torch::Tensor& xscales,
    const torch::Tensor& packed_u4, const torch::Tensor& wscales,
    void* zp_ptr, bool per_block_zp, torch::Tensor& output,
    int64_t M, int64_t K, int64_t N, int64_t act_gs, int64_t wei_gs,
    const torch::Device& device) {
    CacheKey key(M, K, N,
                 (act_gs * 1000 + wei_gs) + (per_block_zp ? (1 << 21) : 0),
                 (int)DstDT);
    CachedPrimitive* cached = nullptr;
    {
        std::lock_guard<std::mutex> lock(g_cache_mutex);
        auto& [eng, strm] = ensure_engine(device);
        cached = &get_or_create(key, M, K, N, act_gs, wei_gs, per_block_zp,
                                DstDT, eng, strm);
    }
    std::unordered_map<int, dnnl::memory> args = {
        {DNNL_ARG_SRC,                            dnnl::memory(cached->src_md,   cached->eng, act.data_ptr())},
        {DNNL_ARG_WEIGHTS,                        dnnl::memory(cached->wei_md,   cached->eng, packed_u4.data_ptr())},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_SRC,     dnnl::memory(cached->xscale_md, cached->eng, xscales.data_ptr())},
        {DNNL_ARG_ATTR_SCALES | DNNL_ARG_WEIGHTS, dnnl::memory(cached->wscale_md, cached->eng, wscales.data_ptr())},
        {DNNL_ARG_ATTR_ZERO_POINTS | DNNL_ARG_WEIGHTS, dnnl::memory(cached->zp_md, cached->eng, zp_ptr)},
        {DNNL_ARG_DST,                            dnnl::memory(cached->dst_md,   cached->eng, output.data_ptr())},
    };
    cached->prim.execute(cached->strm, args);
}

torch::Tensor onednn_s8u4_gemm(
    const torch::Tensor& act,
    const torch::Tensor& xscales,
    const torch::Tensor& packed_u4,
    const torch::Tensor& scales_f16,
    torch::ScalarType out_dtype,
    std::optional<torch::Tensor> zp_u8) {
    TORCH_CHECK(act.dim() == 2, "act must be [M, K]");
    TORCH_CHECK(act.scalar_type() == torch::kInt8, "act must be int8");
    TORCH_CHECK(act.device().is_xpu(), "act must be on XPU");
    TORCH_CHECK(packed_u4.scalar_type() == torch::kUInt8, "packed_u4 must be uint8");
    TORCH_CHECK(xscales.scalar_type() == torch::kFloat16 ||
                    xscales.scalar_type() == torch::kFloat,
                "xscales must be f16 or f32");
    TORCH_CHECK(scales_f16.scalar_type() == torch::kFloat16 ||
                    scales_f16.scalar_type() == torch::kFloat,
                "scales_f16 must be f16 or f32");

    const int64_t M = act.size(0);
    const int64_t K = act.size(1);
    const int64_t N = packed_u4.size(0);
    TORCH_CHECK(packed_u4.size(1) == K / 2, "packed_u4.size(1) must equal K/2");
    const int64_t G_src = xscales.size(1);
    const int64_t G_wei = scales_f16.size(0);
    TORCH_CHECK(G_src > 0, "xscales must have at least one group column");
    TORCH_CHECK(xscales.size(0) == M, "xscales must be [M, G_src]");
    TORCH_CHECK(scales_f16.size(1) == N, "scales_f16 must be [G_wei, N]");
    const int64_t act_gs = K / G_src;
    const int64_t wei_gs = K / G_wei;
    TORCH_CHECK(act_gs * G_src == K, "K must be divisible by G_src");
    TORCH_CHECK(wei_gs * G_wei == K, "K must be divisible by G_wei");

    const bool per_block_zp = zp_u8.has_value()
        && zp_u8->defined() && zp_u8->numel() > 0;
    torch::Tensor zp_c;   // 持久的连续副本（per-block）
    static torch::Tensor zp_scalar;
    const void* zp_ptr;
    if (per_block_zp) {
        const torch::Tensor& zpt = zp_u8.value();
        TORCH_CHECK(zpt.scalar_type() == torch::kUInt8, "zp_u8 must be uint8");
        TORCH_CHECK(zpt.dim() == 2, "zp_u8 must be [G_wei, N]");
        TORCH_CHECK(zpt.size(0) == G_wei && zpt.size(1) == N,
                    "zp_u8 must be [G_wei, N]");
        zp_c = zpt.contiguous();
        zp_ptr = zp_c.data_ptr();
    } else {
        if (!zp_scalar.defined() || zp_scalar.device() != act.device()) {
            zp_scalar = torch::tensor({8},
                torch::TensorOptions().dtype(torch::kUInt8).device(act.device()));
        }
        zp_ptr = zp_scalar.data_ptr();
    }

    // dst dtype follows out_dtype (fp16/bf16/f32). Accumulation stays wide
    // (fpmath any -> bf16 accumulate, no overflow at Qwen-scale K).
    torch::Tensor output = torch::empty({M, N},
        torch::TensorOptions().dtype(out_dtype).device(act.device()));

    const auto act_c = act.contiguous();
    const auto xs_c = xscales.contiguous();
    const auto wu_c = packed_u4.contiguous();
    const auto ws_c = scales_f16.contiguous();
    const auto xs_f32 = xs_c.scalar_type() == torch::kFloat
                            ? xs_c
                            : xs_c.to(torch::kFloat);

    switch (out_dtype) {
        case torch::kFloat:
            launch<DT::f32>(act_c, xs_f32, wu_c, ws_c,
                const_cast<void*>(zp_ptr), per_block_zp,
                output, M, K, N, act_gs, wei_gs, act_c.device());
            break;
        case torch::kHalf:
            launch<DT::f16>(act_c, xs_f32, wu_c, ws_c,
                const_cast<void*>(zp_ptr), per_block_zp,
                output, M, K, N, act_gs, wei_gs, act_c.device());
            break;
        default:
            launch<DT::bf16>(act_c, xs_f32, wu_c, ws_c,
                const_cast<void*>(zp_ptr), per_block_zp,
                output, M, K, N, act_gs, wei_gs, act_c.device());
            break;
    }
    return output;
}

}  // namespace svdq
}  // namespace omni_xpu
