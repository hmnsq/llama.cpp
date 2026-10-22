#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HTP_MATMUL_OPS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MATMUL_OPS_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-vtcm.h"

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
ਬਾਹਰੀ "C" {
#ਜੇ_ਅੰਤ

// --- HMX Tile Constraints ---
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_HMX_TILE_ਮਾਪ   (32 * 32 * ਆਕਾਰ(__fp16)) // 2048 bytes
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_HMX_TILE_N_ELMS 1024
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_HMX_MIN_NROWS   4

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_ਭਾਰ_TILE_K_Q2_0             64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_ਭਾਰ_TILE_ਮਾਪ_Q2_0          HTP_TENSOR_TILE_ਮਾਪ_Q2_0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_ਭਾਰ_ALIGNED_TILE_ਮਾਪ_Q2_0  640

// --- Activation Tiled Block Sizes (including padding) ---
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_ACT_TILE_ਮਾਪ_Q8_0      1152
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_ACT_TILE_ਮਾਪ_Q8_1      1280
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_ACT_TILE_ਮਾਪ_F16       2048

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_MAX_PREFETCH 16

// --- Solver Cost Model Penalty Weights (HMX-specific) ---
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_HMX_ਲਾਗਤ_W_DEQUANT 3 // cost penalty for quantized weight loading/dequantization
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_HMX_ਲਾਗਤ_A_CONVERT 2 // cost penalty for activation loading/conversion

// --- DMA Activation Transfer Configuration ---
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_DMA_ACT_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਕਦਮ 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_MM_DMA_ACT_MULTIPLIER    (2 * HTP_MM_DMA_ACT_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_ਕਦਮ)

ਗਿਣਤੀ htp_mm_kernel_type {
    HTP_MM_KERNEL_UNSUPPORTED = 0,

    // HMX paths
    HTP_MM_KERNEL_HMX_2D,
    HTP_MM_KERNEL_HMX_BATCHED,

    // HVX floating-point paths
    HTP_MM_KERNEL_HVX_F16_F16_VTCM,
    HTP_MM_KERNEL_HVX_F32_F32_VTCM,
    HTP_MM_KERNEL_HVX_BF16_F32_VTCM,

    // HVX quantized paths
    HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ,      // standard row-wise parallel quantization
    HTP_MM_KERNEL_HVX_QUANT_ਖੰਡ,    // parallel block-wise quantization
};

// Op-specific struct for precomputed matmul params
ਬਣਤਰ htp_mm_kernel_params {
    uint8_t  kernel_type;        // enum htp_mm_kernel_type
    uint8_t  pipeline;           // 1 = pipelined execution, 0 = standard
    uint8_t  collapse;           // 1 = collapse outer dims into 2D, 0 = standard
    uint8_t  n_hmx;              // 1 = use HMX, 0 = use HVX

    uint8_t  n_ਧਾਗੇ;          // Number of threads to spawn
    uint8_t  n_act_ਧਾਗੇ;      // Number of threads for activation preparation
    uint8_t  n_prefetch;         // Prefetch lookahead buffers/rows in VTCM
    uint8_t  n_weights;          // Number of weights for fused NX

    int32_t  m_chunk;            // Row chunk size (M chunk)
    int32_t  n_chunk;            // Col chunk size (N chunk)
    int32_t  tile_ਮਾਪ;          // Weight tile size
    int32_t  aligned_tile_ਮਾਪ;  // Aligned weight tile size (padded to 128)
    int32_t  act_ਪੰਕਤੀ_ਮਾਪ;       // Row size for activation scratchpad
    int32_t  vtcm_ਮਾਪ;          // Total required scratchpad size in VTCM
    int32_t  vtcm_src0_ਮਾਪ;     // src0 scratchpad size in VTCM
    int32_t  vtcm_act_ਮਾਪ;      // activation scratchpad size in VTCM
    int32_t  vtcm_bias_ਮਾਪ;     // bias scratchpad size in VTCM (fused only)
    int32_t  vtcm_dst_ਮਾਪ;      // dst scratchpad size in VTCM

    // Precomputed division values
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne12_ne1;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne1;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_r2;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_r3;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne12;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_n_act_ਧਾਗੇ;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne00_padded;
};

#ਜੇ defined(__cplusplus)
static_assert(ਆਕਾਰ(ਬਣਤਰ htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#ਨਹੀਂਤਾਂ
_Static_assert(ਆਕਾਰ(ਬਣਤਰ htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#ਜੇ_ਅੰਤ

ਬਣਤਰ mmid_ਪੰਕਤੀ_mapping {
    uint32_t i1;
    uint32_t i2;
};

// Search for optimal (mc, nc) chunk sizes within VTCM budget.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ htp_mm_hmx_compute_chunks(size_t   vtcm_ਕੁੱਲ,
                              size_t   overhead,
                              size_t   ਪ੍ਰਤੀ_n_ਲਾਗਤ,
                              size_t   ਪ੍ਰਤੀ_m_ਲਾਗਤ,
                              size_t   ਪ੍ਰਤੀ_mn_ਲਾਗਤ,
                              size_t   m,
                              size_t   n,
                              size_t   m_ਖੰਡ_ਲਾਗਤ,
                              size_t   n_ਖੰਡ_ਲਾਗਤ,
                              size_t * m_chunk_ਬਾਹਰਲਾ,
                              size_t * n_chunk_ਬਾਹਰਲਾ,
                              size_t * ਕੁੱਲ_ਬਾਹਰਲਾ) {
    ਜੇ (m == 0 || n == 0) ਵਾਪਸ -1;
    ਜੇ (vtcm_ਕੁੱਲ <= overhead) ਵਾਪਸ -1;
    ਜੇ (ਪ੍ਰਤੀ_n_ਲਾਗਤ == 0 || ਪ੍ਰਤੀ_m_ਲਾਗਤ == 0 || ਪ੍ਰਤੀ_mn_ਲਾਗਤ == 0) ਵਾਪਸ -1;

    ਅਚਲ size_t usable = vtcm_ਕੁੱਲ - overhead;

    size_t ਉੱਤਮ_ਲਾਗਤ = SIZE_MAX;
    size_t ਉੱਤਮ_ਪੂਛ_waste = SIZE_MAX;
    size_t ਉੱਤਮ_mn   = 0;
    size_t ਉੱਤਮ_m = 0, ਉੱਤਮ_n = 0;

    ਅਚਲ size_t max_nc_budget = (usable / ਪ੍ਰਤੀ_n_ਲਾਗਤ);
    ਅਚਲ size_t n_max = hex_align_ਥੱਲੇ(hex_smin((size_t)n, max_nc_budget), HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ);
    ਲਈ (size_t nc = n_max; nc >= HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ; nc -= HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) {
        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        ਜੇ (hex_mul_ਉਛਾਲ(nc, ਪ੍ਰਤੀ_n_ਲਾਗਤ, &n_fixed)) ਜਾਰੀ;
        ਜੇ (n_fixed >= usable) ਜਾਓ ਅਗਲਾ_nc;

        ਜੇ (hex_mul_ਉਛਾਲ(nc, ਪ੍ਰਤੀ_mn_ਲਾਗਤ, &ncmn)) ਜਾਓ ਅਗਲਾ_nc;
        ਜੇ (hex_ਜੋੜੋ_ਉਛਾਲ(ਪ੍ਰਤੀ_m_ਲਾਗਤ, ncmn, &mc_denom) || mc_denom == 0) ਜਾਓ ਅਗਲਾ_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hex_align_ਥੱਲੇ(mc, HTP_MM_HMX_TILE_N_ਪੰਕਤੀਆਂ);
            mc = hex_smin(mc, m);

            ਜੇ (mc == 0) {
                ਜਾਓ ਅਗਲਾ_nc;
            }

            size_t mblocks = ((size_t) m + mc - 1) / mc;
            size_t nblocks = ((size_t) n + nc - 1) / nc;
            size_t ਲਾਗਤ    = mblocks * m_ਖੰਡ_ਲਾਗਤ + nblocks * n_ਖੰਡ_ਲਾਗਤ;
            size_t rem     = n % nc;
            size_t ਪੂਛ_waste = (rem == 0) ? 0 : (nc - rem);
            size_t mn      = mc * nc;
            ਜੇ (ਲਾਗਤ < ਉੱਤਮ_ਲਾਗਤ ||
                (ਲਾਗਤ == ਉੱਤਮ_ਲਾਗਤ && ਪੂਛ_waste < ਉੱਤਮ_ਪੂਛ_waste) ||
                (ਲਾਗਤ == ਉੱਤਮ_ਲਾਗਤ && ਪੂਛ_waste == ਉੱਤਮ_ਪੂਛ_waste && mn > ਉੱਤਮ_mn)) {
                ਉੱਤਮ_ਲਾਗਤ       = ਲਾਗਤ;
                ਉੱਤਮ_ਪੂਛ_waste = ਪੂਛ_waste;
                ਉੱਤਮ_mn         = mn;
                ਉੱਤਮ_m          = mc;
                ਉੱਤਮ_n          = nc;
            }
        }

ਅਗਲਾ_nc:
        ਜੇ (nc == HTP_MM_HMX_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ) ਰੋਕੋ;  // avoid size_t underflow
    }

    ਜੇ (ਉੱਤਮ_m == 0 || ਉੱਤਮ_n == 0) ਵਾਪਸ -1;

    // Compute exact total (with overflow checks)
    size_t t0 = 0, t1 = 0, t2 = 0, mn = 0, ਕੁੱਲ = 0;
    ਜੇ (hex_mul_ਉਛਾਲ(ਉੱਤਮ_n, ਪ੍ਰਤੀ_n_ਲਾਗਤ, &t0)) ਵਾਪਸ -1;
    ਜੇ (hex_mul_ਉਛਾਲ(ਉੱਤਮ_m, ਪ੍ਰਤੀ_m_ਲਾਗਤ, &t1)) ਵਾਪਸ -1;
    ਜੇ (hex_mul_ਉਛਾਲ(ਉੱਤਮ_m, ਉੱਤਮ_n, &mn))     ਵਾਪਸ -1;
    ਜੇ (hex_mul_ਉਛਾਲ(mn, ਪ੍ਰਤੀ_mn_ਲਾਗਤ, &t2))    ਵਾਪਸ -1;
    ਜੇ (hex_ਜੋੜੋ_ਉਛਾਲ(t0, t1, &ਕੁੱਲ))          ਵਾਪਸ -1;
    ਜੇ (hex_ਜੋੜੋ_ਉਛਾਲ(ਕੁੱਲ, t2, &ਕੁੱਲ))       ਵਾਪਸ -1;
    ਜੇ (hex_ਜੋੜੋ_ਉਛਾਲ(ਕੁੱਲ, overhead, &ਕੁੱਲ)) ਵਾਪਸ -1;

    *m_chunk_ਬਾਹਰਲਾ = ਉੱਤਮ_m;
    *n_chunk_ਬਾਹਰਲਾ = ਉੱਤਮ_n;
    *ਕੁੱਲ_ਬਾਹਰਲਾ   = ਕੁੱਲ;
    ਵਾਪਸ 0;
}

// weight types whose tiles carry a per-block offset (x = d * q + m): the activations need block sums (q8_1)
// (Q2_K: per-16 k sums, q8_1_s16)
ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_mm_ਭਾਰ_has_ਵਿੱਥ(ਪੂਰਨ ਭਾਰ_type) {
    ਵਾਪਸ ਭਾਰ_type == HTP_TYPE_Q4_1 || ਭਾਰ_type == HTP_TYPE_Q4_K || ਭਾਰ_type == HTP_TYPE_Q5_K ||
           ਭਾਰ_type == HTP_TYPE_Q2_K || ਭਾਰ_type == HTP_TYPE_Q2_0;
}

// --- Activation/Row Size Helpers ---
ਸਥਿਰ ਸਤਰਅੰਦਰ size_t htp_mm_q8_0_tiled_ਪੰਕਤੀ_ਮਾਪ(uint32_t ne) {
    ਅਚਲ uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ਅਚਲ uint32_t nb_32 = ne_padded / 32;
    ਵਾਪਸ nb_32 * HTP_MM_ACT_TILE_ਮਾਪ_Q8_0;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t htp_mm_q8_1_tiled_ਪੰਕਤੀ_ਮਾਪ(uint32_t ne) {
    ਅਚਲ uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ਅਚਲ uint32_t nb_32 = ne_padded / 32;
    ਵਾਪਸ nb_32 * HTP_MM_ACT_TILE_ਮਾਪ_Q8_1;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t htp_mm_f16_tiled_ਪੰਕਤੀ_ਮਾਪ(uint32_t ne) {
    ਅਚਲ uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ਅਚਲ uint32_t nb_32 = ne_padded / 32;
    ਵਾਪਸ nb_32 * HTP_MM_ACT_TILE_ਮਾਪ_F16;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t htp_mm_tiled_act_ਪੰਕਤੀ_ਮਾਪ(ਪੂਰਨ ਭਾਰ_type, uint32_t ne) {
    ਜੇ (ਭਾਰ_type == HTP_TYPE_F16) {
        ਵਾਪਸ htp_mm_f16_tiled_ਪੰਕਤੀ_ਮਾਪ(ne);
    }
    ਵਾਪਸ htp_mm_ਭਾਰ_has_ਵਿੱਥ(ਭਾਰ_type) ? htp_mm_q8_1_tiled_ਪੰਕਤੀ_ਮਾਪ(ne) : htp_mm_q8_0_tiled_ਪੰਕਤੀ_ਮਾਪ(ne);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t htp_mm_ਗੇੜਾ_ਉੱਪਰ(size_t n, size_t m) {
    ਵਾਪਸ ((n + m - 1) / m) * m;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_mm_hmx_pipeline(uint32_t m) {
    ਵਾਪਸ m > 32;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_mm_hmx_get_2d_chunk_costs(
    ਪੂਰਨ wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_ਮਾਪ,
    size_t * ਮਾਪ_ਪ੍ਰਤੀ_n_ਬਾਹਰਲਾ, size_t * ਮਾਪ_ਪ੍ਰਤੀ_m_ਬਾਹਰਲਾ, size_t * ਮਾਪ_ਪ੍ਰਤੀ_mn_ਬਾਹਰਲਾ
) {
    ਅਚਲ bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
    ਅਚਲ size_t ਪੰਕਤੀ_stride = htp_tensor_get_ਪੰਕਤੀ_ਮਾਪ(wtype, k);
    ਅਚਲ size_t vec_dot_ਮਾਪ = k * ਆਕਾਰ(uint16_t);
    ਅਚਲ uint32_t n_ਭਾਰ_tiles = k / htp_tensor_get_tile_k(wtype);
    ਅਚਲ size_t qweight_ਪੰਕਤੀ_stride = is_quant ? (size_t)(n_ਭਾਰ_tiles * aligned_tile_ਮਾਪ) / 32 : 0;

    *ਮਾਪ_ਪ੍ਰਤੀ_n_ਬਾਹਰਲਾ = (pipeline ? 2 : 1) * (is_quant ? qweight_ਪੰਕਤੀ_stride : ਪੰਕਤੀ_stride) +
                      (pipeline ? 2 * vec_dot_ਮਾਪ : vec_dot_ਮਾਪ);
    *ਮਾਪ_ਪ੍ਰਤੀ_m_ਬਾਹਰਲਾ = vec_dot_ਮਾਪ;
    *ਮਾਪ_ਪ੍ਰਤੀ_mn_ਬਾਹਰਲਾ = (pipeline ? 2 : 1) * ਆਕਾਰ(uint16_t);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_mm_hmx_get_batched_chunk_costs(
    uint32_t k, uint32_t ਸਮੂਹ_ਮਾਪ,
    size_t * ਮਾਪ_ਪ੍ਰਤੀ_n_ਬਾਹਰਲਾ, size_t * ਮਾਪ_ਪ੍ਰਤੀ_m_ਬਾਹਰਲਾ, size_t * ਮਾਪ_ਪ੍ਰਤੀ_mn_ਬਾਹਰਲਾ
) {
    ਅਚਲ size_t vec_dot_ਮਾਪ = k * ਆਕਾਰ(uint16_t);
    *ਮਾਪ_ਪ੍ਰਤੀ_n_ਬਾਹਰਲਾ = 3 * vec_dot_ਮਾਪ;
    *ਮਾਪ_ਪ੍ਰਤੀ_m_ਬਾਹਰਲਾ = ਸਮੂਹ_ਮਾਪ * vec_dot_ਮਾਪ;
    *ਮਾਪ_ਪ੍ਰਤੀ_mn_ਬਾਹਰਲਾ = ਆਕਾਰ(uint16_t);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t htp_mm_hmx_get_2d_overhead(bool pipeline, bool is_matmul_id) {
    size_t ਅੰਕ_regions = pipeline ? 7 : (is_matmul_id ? 4 : 5);
    ਵਾਪਸ ਅੰਕ_regions * HTP_MM_HMX_TILE_ਮਾਪ + 256;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t htp_mm_hmx_get_batched_overhead(ਖਾਲੀ) {
    ਵਾਪਸ 5 * HTP_MM_HMX_TILE_ਮਾਪ + 256;
}

ਬਣਤਰ htp_mm_hmx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_ਭਾਰ[2];     // [1] is only used when pipelined
    size_t off_act;
    size_t off_act_f32;       // fp32 activation conversion scratch
    size_t off_dst[2];        // [1] is only used when pipelined
    size_t off_scratch[2];    // dequantization scratch pads
    size_t off_scales;        // HMX scales (256 bytes)
    size_t off_bias;          // bias in VTCM

    // Cached sizes of regions for HMX kernel use
    size_t ਭਾਰ_area_ਬਾਈਟਾਂ;
    size_t act_area_ਬਾਈਟਾਂ;
    size_t act_f32_ਬਾਈਟਾਂ;
    size_t output_area_ਬਾਈਟਾਂ;
    size_t scratch_ਬਾਈਟਾਂ[2];
    size_t act_ਸਿਰ_stride;
    size_t bias_ਬਾਈਟਾਂ;

    size_t ਕੁੱਲ_ਬਾਈਟਾਂ;
};

ਬਣਤਰ htp_mm_hvx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_act;           // vtcm_act (activation)
    size_t off_src0;          // vtcm_src0 (weight/Wk)
    size_t off_bias;          // vtcm_bias (bias / fused add only)
    size_t off_dst;           // vtcm_dst (output scratch)
    size_t off_act_raw;       // vtcm_act_raw (raw activation DMA staging)

    // Cached sizes
    size_t src0_ਬਾਈਟਾਂ;
    size_t act_ਬਾਈਟਾਂ;
    size_t bias_ਬਾਈਟਾਂ;
    size_t dst_ਬਾਈਟਾਂ;
    size_t act_raw_ਬਾਈਟਾਂ;

    size_t ਕੁੱਲ_ਬਾਈਟਾਂ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_mm_hmx_vtcm_layout_build(
    ਬਣਤਰ htp_mm_hmx_vtcm_layout * L,
    ਪੂਰਨ kernel_type,
    ਪੂਰਨ wtype,
    uint32_t k,
    size_t mc,
    size_t nc,
    uint32_t ਸਮੂਹ_ਮਾਪ,
    bool pipeline,
    uint32_t act_ਧਾਗੇ,
    uint32_t aligned_tile_ਮਾਪ,
    size_t bias_ਮਾਪ
) {
    size_t off = 0;

    ਜੇ (kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        ਅਚਲ size_t vec_dot_ਮਾਪ     = k * ਆਕਾਰ(uint16_t);
        ਅਚਲ size_t act_ਸਿਰ_stride   = mc * k;
        ਅਚਲ size_t ਭਾਰ_area_ਮਾਪ  = hex_align_ਉੱਪਰ(nc * vec_dot_ਮਾਪ, HTP_MM_HMX_TILE_ਮਾਪ);
        ਅਚਲ size_t activation_area_ਮਾਪ = hex_align_ਉੱਪਰ(ਸਮੂਹ_ਮਾਪ * act_ਸਿਰ_stride * ਆਕਾਰ(uint16_t), HTP_MM_HMX_TILE_ਮਾਪ);
        ਅਚਲ size_t output_area_ਮਾਪ  = hex_align_ਉੱਪਰ(ਸਮੂਹ_ਮਾਪ * mc * nc * ਆਕਾਰ(uint16_t), HTP_MM_HMX_TILE_ਮਾਪ);
        ਅਚਲ size_t scratch_area_ਮਾਪ = hex_align_ਉੱਪਰ(nc * vec_dot_ਮਾਪ, HTP_MM_HMX_TILE_ਮਾਪ);
        ਅਚਲ size_t min_f32_ਮਾਪ = hex_align_ਉੱਪਰ(act_ਧਾਗੇ * HTP_MM_DMA_ACT_MULTIPLIER * k * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);

        // Group A: Permanent activation tiles and scales
        size_t off_ਸਮੂਹ_a = 0;
        VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_a, off_act, activation_area_ਮਾਪ);
        VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_a, off_scales, HTP_MM_HMX_TILE_ਮਾਪ); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ਸਮੂਹ_a, off_bias, hex_align_ਉੱਪਰ(bias_ਮਾਪ, HTP_MM_HMX_TILE_ਮਾਪ), bias_ਮਾਪ > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_ਸਮੂਹ_b = off_ਸਮੂਹ_a;
        VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_ਭਾਰ[0], ਭਾਰ_area_ਮਾਪ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ਸਮੂਹ_b, off_ਭਾਰ[1], ਭਾਰ_area_ਮਾਪ, false);
        VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_dst[0], output_area_ਮਾਪ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ਸਮੂਹ_b, off_dst[1], output_area_ਮਾਪ, false);
        VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_scratch[0], scratch_area_ਮਾਪ);
        VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_scratch[1], scratch_area_ਮਾਪ);

        ਅਚਲ size_t ਸਮੂਹ_b_ਮਾਪ = off_ਸਮੂਹ_b - off_ਸਮੂਹ_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ਅਚਲ size_t max_f32_ਮਾਪ = act_ਧਾਗੇ * 64 * k * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਅਚਲ size_t act_f32_ਮਾਪ = hex_align_ਉੱਪਰ(hex_smin(max_f32_ਮਾਪ, hex_smax(min_f32_ਮਾਪ, ਸਮੂਹ_b_ਮਾਪ)), 128);
        size_t off_ਸਮੂਹ_c = off_ਸਮੂਹ_a;
        VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_c, off_act_f32, act_f32_ਮਾਪ);

        ਅਚਲ size_t ਸਮੂਹ_c_ਮਾਪ = off_ਸਮੂਹ_c - off_ਸਮੂਹ_a;

        L->ਭਾਰ_area_ਬਾਈਟਾਂ = ਭਾਰ_area_ਮਾਪ;
        L->act_area_ਬਾਈਟਾਂ    = activation_area_ਮਾਪ;
        L->act_f32_ਬਾਈਟਾਂ     = act_f32_ਮਾਪ;
        L->output_area_ਬਾਈਟਾਂ = output_area_ਮਾਪ;
        L->scratch_ਬਾਈਟਾਂ[0]  = scratch_area_ਮਾਪ;
        L->scratch_ਬਾਈਟਾਂ[1]  = scratch_area_ਮਾਪ;
        L->act_ਸਿਰ_stride   = act_ਸਿਰ_stride;
        L->bias_ਬਾਈਟਾਂ        = bias_ਮਾਪ;

        off = off_ਸਮੂਹ_a + hex_smax(ਸਮੂਹ_b_ਮਾਪ, ਸਮੂਹ_c_ਮਾਪ);
    } ਨਹੀਂਤਾਂ {
        // HTP_MM_KERNEL_HMX_2D
        ਅਚਲ bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
        ਅਚਲ size_t ਪੰਕਤੀ_stride = htp_tensor_get_ਪੰਕਤੀ_ਮਾਪ(wtype, k);
        ਅਚਲ size_t vec_dot_ਮਾਪ = k * ਆਕਾਰ(uint16_t);
        ਅਚਲ uint32_t n_ਭਾਰ_tiles = k / htp_tensor_get_tile_k(wtype);

        ਅਚਲ size_t min_f32_ਮਾਪ = hex_align_ਉੱਪਰ(act_ਧਾਗੇ * HTP_MM_DMA_ACT_MULTIPLIER * k * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);
        ਅਚਲ size_t ਭਾਰ_area_ਮਾਪ = is_quant
            ? hex_align_ਉੱਪਰ((nc / 32) * n_ਭਾਰ_tiles * aligned_tile_ਮਾਪ, HTP_MM_HMX_TILE_ਮਾਪ)
            : hex_align_ਉੱਪਰ(nc * ਪੰਕਤੀ_stride, HTP_MM_HMX_TILE_ਮਾਪ);
        ਅਚਲ size_t act_area_ਮਾਪ    = hex_align_ਉੱਪਰ(mc * vec_dot_ਮਾਪ, HTP_MM_HMX_TILE_ਮਾਪ);
        ਅਚਲ size_t output_area_ਮਾਪ = hex_align_ਉੱਪਰ(mc * nc * ਆਕਾਰ(__fp16), HTP_MM_HMX_TILE_ਮਾਪ);

        ਅਚਲ size_t scratch0_ਮਾਪ = hex_align_ਉੱਪਰ(nc * vec_dot_ਮਾਪ, HTP_MM_HMX_TILE_ਮਾਪ);
        ਅਚਲ size_t scratch1_ਮਾਪ = pipeline ? scratch0_ਮਾਪ : 0;

        // Group A:  Scales and activation tiles (must not overlap with Group B or C)
        size_t off_ਸਮੂਹ_a = 0;
        VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_a, off_scales, HTP_MM_HMX_TILE_ਮਾਪ); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_a, off_act, act_area_ਮਾਪ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ਸਮੂਹ_a, off_bias, hex_align_ਉੱਪਰ(bias_ਮਾਪ, HTP_MM_HMX_TILE_ਮਾਪ), bias_ਮਾਪ > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_ਸਮੂਹ_b = off_ਸਮੂਹ_a;
        VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_ਭਾਰ[0], ਭਾਰ_area_ਮਾਪ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ਸਮੂਹ_b, off_ਭਾਰ[1], ਭਾਰ_area_ਮਾਪ, pipeline);
        VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_dst[0], output_area_ਮਾਪ);
        VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_scratch[0], scratch0_ਮਾਪ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ਸਮੂਹ_b, off_scratch[1], scratch0_ਮਾਪ, pipeline);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ਸਮੂਹ_b, off_dst[1], output_area_ਮਾਪ, pipeline);

        ਅਚਲ size_t ਸਮੂਹ_b_ਮਾਪ = off_ਸਮੂਹ_b - off_ਸਮੂਹ_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ਅਚਲ size_t max_f32_ਮਾਪ = act_ਧਾਗੇ * 64 * k * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਅਚਲ size_t act_f32_ਮਾਪ = hex_align_ਉੱਪਰ(hex_smin(max_f32_ਮਾਪ, hex_smax(min_f32_ਮਾਪ, ਸਮੂਹ_b_ਮਾਪ)), 128);
        size_t off_ਸਮੂਹ_c = off_ਸਮੂਹ_a;
        VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_c, off_act_f32, act_f32_ਮਾਪ);

        ਅਚਲ size_t ਸਮੂਹ_c_ਮਾਪ = off_ਸਮੂਹ_c - off_ਸਮੂਹ_a;

        L->ਭਾਰ_area_ਬਾਈਟਾਂ = ਭਾਰ_area_ਮਾਪ;
        L->act_area_ਬਾਈਟਾਂ    = act_area_ਮਾਪ;
        L->act_f32_ਬਾਈਟਾਂ     = act_f32_ਮਾਪ;
        L->output_area_ਬਾਈਟਾਂ = output_area_ਮਾਪ;
        L->scratch_ਬਾਈਟਾਂ[0]  = scratch0_ਮਾਪ;
        L->scratch_ਬਾਈਟਾਂ[1]  = scratch1_ਮਾਪ;
        L->act_ਸਿਰ_stride   = 0;
        L->bias_ਬਾਈਟਾਂ        = bias_ਮਾਪ;

        off = off_ਸਮੂਹ_a + hex_smax(ਸਮੂਹ_b_ਮਾਪ, ਸਮੂਹ_c_ਮਾਪ);
    }

    L->ਕੁੱਲ_ਬਾਈਟਾਂ = off;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ htp_mm_hvx_vtcm_layout_build(
    ਬਣਤਰ htp_mm_hvx_vtcm_layout * L,
    ਪੂਰਨ kernel_type,
    ਪੂਰਨ wtype,
    uint32_t ne10,       // k
    uint32_t act_nrows,  // m_total
    uint32_t n_ਧਾਗੇ,
    size_t dst_ਪੰਕਤੀ_ਮਾਪ,
    size_t src0_ਪੰਕਤੀ_ਮਾਪ,
    size_t act_ਪੰਕਤੀ_ਮਾਪ,
    size_t bias_ਪੰਕਤੀ_ਮਾਪ,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_nx
) {
    (ਖਾਲੀ)act_ਪੰਕਤੀ_ਮਾਪ;
    size_t src0_sz    = 0;
    size_t act_sz     = 0;
    size_t bias_sz    = bias_ਪੰਕਤੀ_ਮਾਪ > 0 ? htp_mm_ਗੇੜਾ_ਉੱਪਰ(bias_ਪੰਕਤੀ_ਮਾਪ, 128) : 0;
    size_t dst_sz     = 0;
    size_t act_raw_sz = 0;

    ਜੇ (is_fused_nx) {
        uint32_t aligned_tile_ਮਾਪ = htp_tensor_get_tile_padded_ਮਾਪ(wtype);
        uint32_t n_k_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        uint32_t tile_ਪੰਕਤੀ_ਮਾਪ = n_k_tiles * aligned_tile_ਮਾਪ;
        size_t ਭਾਰ_sz_ਪ੍ਰਤੀ_thread = hex_ਗੇੜਾ_ਉੱਪਰ(n_prefetch * tile_ਪੰਕਤੀ_ਮਾਪ, 128);

        size_t tiled_act_ਪੰਕਤੀ_ਮਾਪ = htp_mm_tiled_act_ਪੰਕਤੀ_ਮਾਪ(wtype, ne10);
        size_t q_act_sz = hex_ਗੇੜਾ_ਉੱਪਰ(tiled_act_ਪੰਕਤੀ_ਮਾਪ * act_nrows, 128);
        size_t raw_ਪੰਕਤੀ_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(ne10 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), QK_Q8_0_TILED * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

        src0_sz    = ਭਾਰ_sz_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ; // shared single-weight prefetch buffer
        act_sz     = q_act_sz;                         // quantized activation buffer
        bias_sz    = 0;
        dst_sz     = 0;
        act_raw_sz = hex_ਗੇੜਾ_ਉੱਪਰ(raw_ਪੰਕਤੀ_ਮਾਪ * act_nrows, 128);
    } ਨਹੀਂਤਾਂ ਜੇ (is_matmul_id) {
        ਅਚਲ size_t act_ਪੰਕਤੀ_ਮਾਪ_tiled = htp_mm_tiled_act_ਪੰਕਤੀ_ਮਾਪ(wtype, ne10);
        ਅਚਲ uint32_t aligned_tile_ਮਾਪ = htp_tensor_get_tile_padded_ਮਾਪ(wtype);
        ਅਚਲ uint32_t n_k_tiles         = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        ਅਚਲ uint32_t tile_ਪੰਕਤੀ_ਮਾਪ     = n_k_tiles * aligned_tile_ਮਾਪ;
        size_t src0_sz_ਪ੍ਰਤੀ_thread        = htp_mm_ਗੇੜਾ_ਉੱਪਰ(n_prefetch * tile_ਪੰਕਤੀ_ਮਾਪ, 256);

        act_sz = htp_mm_ਗੇੜਾ_ਉੱਪਰ(act_ਪੰਕਤੀ_ਮਾਪ_tiled * act_nrows, 256);

        size_t raw_ਪੰਕਤੀ_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(ne10 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), QK_Q8_0_TILED * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

        src0_sz    = src0_sz_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
        dst_sz     = 0;
        bias_sz    = 0;
        act_raw_sz = hex_ਗੇੜਾ_ਉੱਪਰ(raw_ਪੰਕਤੀ_ਮਾਪ * act_nrows, 128);
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ_padded = htp_mm_ਗੇੜਾ_ਉੱਪਰ(src0_ਪੰਕਤੀ_ਮਾਪ, 128);
        ਅਚਲ size_t dst_nrows = (act_nrows > 1) ? 0 : 1;

        ਚੋਣ (kernel_type) {
            ਹਾਲਤ HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                size_t f16_act_ਪੰਕਤੀ_ਮਾਪ = htp_mm_ਗੇੜਾ_ਉੱਪਰ(ne10 * 2, 128);
                act_sz     = htp_mm_ਗੇੜਾ_ਉੱਪਰ(f16_act_ਪੰਕਤੀ_ਮਾਪ * act_nrows, 256);
                src0_sz    = htp_mm_ਗੇੜਾ_ਉੱਪਰ(n_prefetch * src0_ਪੰਕਤੀ_ਮਾਪ_padded, 256) * n_ਧਾਗੇ;
                dst_sz     = dst_nrows > 0 ? htp_mm_ਗੇੜਾ_ਉੱਪਰ(dst_ਪੰਕਤੀ_ਮਾਪ, 128) * n_ਧਾਗੇ : 0;
                act_raw_sz = hex_ਗੇੜਾ_ਉੱਪਰ(hex_ਗੇੜਾ_ਉੱਪਰ(ne10 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128) * act_nrows, 128);
                ਰੋਕੋ;
            }
            ਹਾਲਤ HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            ਹਾਲਤ HTP_MM_KERNEL_HVX_BF16_F32_VTCM: {
                size_t f32_act_ਪੰਕਤੀ_ਮਾਪ = htp_mm_ਗੇੜਾ_ਉੱਪਰ(ne10 * 4, 128);
                act_sz     = htp_mm_ਗੇੜਾ_ਉੱਪਰ(f32_act_ਪੰਕਤੀ_ਮਾਪ * act_nrows, 256);
                src0_sz    = htp_mm_ਗੇੜਾ_ਉੱਪਰ(n_prefetch * src0_ਪੰਕਤੀ_ਮਾਪ_padded, 256) * n_ਧਾਗੇ;
                dst_sz     = dst_nrows > 0 ? htp_mm_ਗੇੜਾ_ਉੱਪਰ(dst_ਪੰਕਤੀ_ਮਾਪ, 128) * n_ਧਾਗੇ : 0;
                act_raw_sz = 0;
                ਰੋਕੋ;
            }
            ਹਾਲਤ HTP_MM_KERNEL_HVX_QUANT_ਖੰਡ:
            ਹਾਲਤ HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ: {
                size_t q_act_ਪੰਕਤੀ_ਮਾਪ = htp_mm_tiled_act_ਪੰਕਤੀ_ਮਾਪ(wtype, ne10);
                uint32_t aligned_tile_ਮਾਪ = htp_tensor_get_tile_padded_ਮਾਪ(wtype);
                uint32_t n_ਭਾਰ_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
                uint32_t tile_ਪੰਕਤੀ_ਮਾਪ = n_ਭਾਰ_tiles * aligned_tile_ਮਾਪ;
                size_t tiled_vtcm_ਮਾਪ = htp_mm_ਗੇੜਾ_ਉੱਪਰ(n_prefetch * tile_ਪੰਕਤੀ_ਮਾਪ, 256);

                src0_sz = tiled_vtcm_ਮਾਪ * n_ਧਾਗੇ;
                act_sz  = htp_mm_ਗੇੜਾ_ਉੱਪਰ(q_act_ਪੰਕਤੀ_ਮਾਪ * act_nrows, 256);

                size_t dst_ਟੁਕੜਾ_ਪ੍ਰਤੀ_thread = (dst_nrows > 0 && act_nrows == 1) ? htp_mm_ਗੇੜਾ_ਉੱਪਰ((dst_ਪੰਕਤੀ_ਮਾਪ + n_ਧਾਗੇ - 1) / n_ਧਾਗੇ, 128) : 0;
                dst_sz = dst_ਟੁਕੜਾ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ;
                size_t raw_ਪੰਕਤੀ_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(ne10 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), QK_Q8_0_TILED * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                act_raw_sz = hex_ਗੇੜਾ_ਉੱਪਰ(raw_ਪੰਕਤੀ_ਮਾਪ * act_nrows, 128);
                ਰੋਕੋ;
            }
            ਮੂਲ:
                ਰੋਕੋ;
        }
    }

    // Group A: Persistent buffers across chunk compute
    size_t off_ਸਮੂਹ_a = 0;
    VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_a, off_act, act_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ਸਮੂਹ_a, off_bias, bias_sz, bias_sz > 0);

    // Group B: Compute-only buffers (starts at off_group_a)
    size_t off_ਸਮੂਹ_b = off_ਸਮੂਹ_a;
    VTCM_LAYOUT_ALLOC(off_ਸਮੂਹ_b, off_src0, src0_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ਸਮੂਹ_b, off_dst, dst_sz, dst_sz > 0);
    ਅਚਲ size_t ਸਮੂਹ_b_ਮਾਪ = off_ਸਮੂਹ_b - off_ਸਮੂਹ_a;

    // Group C: Raw activation staging buffer (overlaps Group B, starts at off_group_a)
    size_t off_ਸਮੂਹ_c = off_ਸਮੂਹ_a;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ਸਮੂਹ_c, off_act_raw, act_raw_sz, act_raw_sz > 0);
    ਅਚਲ size_t ਸਮੂਹ_c_ਮਾਪ = off_ਸਮੂਹ_c - off_ਸਮੂਹ_a;

    L->src0_ਬਾਈਟਾਂ    = src0_sz;
    L->act_ਬਾਈਟਾਂ     = act_sz;
    L->bias_ਬਾਈਟਾਂ    = bias_sz;
    L->dst_ਬਾਈਟਾਂ     = dst_sz;
    L->act_raw_ਬਾਈਟਾਂ = act_raw_sz;
    L->ਕੁੱਲ_ਬਾਈਟਾਂ   = off_ਸਮੂਹ_a + hex_smax(ਸਮੂਹ_b_ਮਾਪ, ਸਮੂਹ_c_ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_mm_hvx_solve_vtcm_params(
    ਪੂਰਨ kernel_type,
    ਪੂਰਨ wtype,
    uint32_t ne10,
    uint32_t act_nrows,
    uint32_t n_ਧਾਗੇ,
    size_t dst_ਪੰਕਤੀ_ਮਾਪ,
    size_t src0_ਪੰਕਤੀ_ਮਾਪ,
    size_t act_ਪੰਕਤੀ_ਮਾਪ,
    size_t bias_ਪੰਕਤੀ_ਮਾਪ,
    uint32_t n_prefetch,
    size_t vtcm_budget,
    ਬਣਤਰ htp_mm_hvx_vtcm_layout * L_ਬਾਹਰਲਾ,
    uint32_t * m_chunk_ਬਾਹਰਲਾ
) {
    ਬਣਤਰ htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, act_nrows, n_ਧਾਗੇ,
        dst_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, act_ਪੰਕਤੀ_ਮਾਪ, bias_ਪੰਕਤੀ_ਮਾਪ, n_prefetch, false, false
    );

    ਜੇ (L.ਕੁੱਲ_ਬਾਈਟਾਂ <= vtcm_budget) {
        *L_ਬਾਹਰਲਾ = L;
        *m_chunk_ਬਾਹਰਲਾ = act_nrows;
        ਵਾਪਸ true;
    }

    ਅਚਲ size_t fixed_ਬਾਈਟਾਂ = L.src0_ਬਾਈਟਾਂ + L.bias_ਬਾਈਟਾਂ + L.dst_ਬਾਈਟਾਂ;
    ਜੇ (vtcm_budget <= fixed_ਬਾਈਟਾਂ) {
        ਵਾਪਸ false;
    }

    ਅਚਲ size_t avail_act = vtcm_budget - fixed_ਬਾਈਟਾਂ;
    size_t ਪੰਕਤੀ_ਮਾਪ = 0;
    ਜੇ (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ || kernel_type == HTP_MM_KERNEL_HVX_QUANT_ਖੰਡ) {
        ਪੰਕਤੀ_ਮਾਪ = htp_mm_tiled_act_ਪੰਕਤੀ_ਮਾਪ(wtype, ne10);
    } ਨਹੀਂਤਾਂ ਜੇ (kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        ਪੰਕਤੀ_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(ne10 * 2, 128);
    } ਨਹੀਂਤਾਂ {
        ਪੰਕਤੀ_ਮਾਪ = hex_ਗੇੜਾ_ਉੱਪਰ(ne10 * 4, 128);
    }
    ਜੇ (ਪੰਕਤੀ_ਮਾਪ == 0) {
        ਵਾਪਸ false;
    }

    size_t eff_ਪੰਕਤੀ_ਮਾਪ = ਪੰਕਤੀ_ਮਾਪ;
    ਜੇ (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ਪੰਕਤੀ || kernel_type == HTP_MM_KERNEL_HVX_QUANT_ਖੰਡ) {
        eff_ਪੰਕਤੀ_ਮਾਪ += hex_ਗੇੜਾ_ਉੱਪਰ(ne10 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), QK_Q8_0_TILED * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    }

    uint32_t m_chunk = (uint32_t) (avail_act / eff_ਪੰਕਤੀ_ਮਾਪ);
    ਜੇ (m_chunk > 1) {
        m_chunk &= ~1U;
    }
    ਜੇ (m_chunk > act_nrows) {
        m_chunk = act_nrows;
    }
    ਜੇ (m_chunk < 1) {
        ਵਾਪਸ false;
    }

    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, m_chunk, n_ਧਾਗੇ,
        dst_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, act_ਪੰਕਤੀ_ਮਾਪ, bias_ਪੰਕਤੀ_ਮਾਪ, n_prefetch, false, false
    );

    ਜਦੋਂ (m_chunk > 2 && L.ਕੁੱਲ_ਬਾਈਟਾਂ > vtcm_budget) {
        m_chunk -= 2;
        htp_mm_hvx_vtcm_layout_build(
            &L, kernel_type, wtype, ne10, m_chunk, n_ਧਾਗੇ,
            dst_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, act_ਪੰਕਤੀ_ਮਾਪ, bias_ਪੰਕਤੀ_ਮਾਪ, n_prefetch, false, false
        );
    }

    ਜੇ (L.ਕੁੱਲ_ਬਾਈਟਾਂ <= vtcm_budget) {
        *L_ਬਾਹਰਲਾ = L;
        *m_chunk_ਬਾਹਰਲਾ = m_chunk;
        ਵਾਪਸ true;
    }

    ਵਾਪਸ false;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t htp_mm_hmx_get_2d_vtcm_ਮਾਪ(
    ਪੂਰਨ wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_ਧਾਗੇ, uint32_t aligned_tile_ਮਾਪ, size_t bias_ਮਾਪ
) {
    ਬਣਤਰ htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_ਧਾਗੇ, aligned_tile_ਮਾਪ, bias_ਮਾਪ);
    ਵਾਪਸ L.ਕੁੱਲ_ਬਾਈਟਾਂ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t htp_mm_hmx_get_batched_vtcm_ਮਾਪ(
    ਪੂਰਨ wtype, uint32_t k, size_t mc, size_t nc, uint32_t ਸਮੂਹ_ਮਾਪ, bool pipeline, uint32_t act_ਧਾਗੇ, size_t bias_ਮਾਪ) {
    (ਖਾਲੀ)pipeline;
    ਬਣਤਰ htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, wtype, k, mc, nc, ਸਮੂਹ_ਮਾਪ, false, act_ਧਾਗੇ, 0, bias_ਮਾਪ);
    ਵਾਪਸ L.ਕੁੱਲ_ਬਾਈਟਾਂ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_mm_hmx_solve_batched_params(
    ਪੂਰਨ wtype,
    uint32_t k,
    uint32_t ne01_padded,
    uint32_t ne11,
    uint32_t ਸਮੂਹ_ਮਾਪ,
    ਪੂਰਨ n_ਧਾਗੇ,
    bool pipeline,
    size_t bias_ਮਾਪ,
    size_t vtcm_budget,
    size_t * m_chunk_ਬਾਹਰਲਾ,
    size_t * n_chunk_ਬਾਹਰਲਾ,
    ਪੂਰਨ * act_ਧਾਗੇ_ਬਾਹਰਲਾ,
    size_t * vtcm_ਮਾਪ_ਬਾਹਰਲਾ
) {
    size_t ਉੱਤਮ_mblocks = SIZE_MAX;
    ਪੂਰਨ ਉੱਤਮ_act_ਧਾਗੇ = 0;
    size_t ਉੱਤਮ_m_chunk = 0;
    size_t ਉੱਤਮ_n_chunk = 0;
    size_t ਉੱਤਮ_vtcm_ਮਾਪ = 0;

    ਪੂਰਨ act_ਧਾਗੇ = n_ਧਾਗੇ;
    ਜਦੋਂ (act_ਧਾਗੇ >= 1) {
        size_t ਸਮੂਹ_overhead = htp_mm_hmx_get_batched_overhead() + (bias_ਮਾਪ > 0 ? hex_align_ਉੱਪਰ(bias_ਮਾਪ, HTP_MM_HMX_TILE_ਮਾਪ) : 0);
        size_t ਸਮੂਹ_ਮਾਪ_ਪ੍ਰਤੀ_n, ਸਮੂਹ_ਮਾਪ_ਪ੍ਰਤੀ_m, ਸਮੂਹ_ਮਾਪ_ਪ੍ਰਤੀ_mn;
        htp_mm_hmx_get_batched_chunk_costs(k, ਸਮੂਹ_ਮਾਪ, &ਸਮੂਹ_ਮਾਪ_ਪ੍ਰਤੀ_n, &ਸਮੂਹ_ਮਾਪ_ਪ੍ਰਤੀ_m, &ਸਮੂਹ_ਮਾਪ_ਪ੍ਰਤੀ_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_ਮਾਪ_candidate = 0;

        ਜੇ (htp_mm_hmx_compute_chunks(vtcm_budget, ਸਮੂਹ_overhead, ਸਮੂਹ_ਮਾਪ_ਪ੍ਰਤੀ_n, ਸਮੂਹ_ਮਾਪ_ਪ੍ਰਤੀ_m, ਸਮੂਹ_ਮਾਪ_ਪ੍ਰਤੀ_mn, hex_align_ਉੱਪਰ(ne11, 32), ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_ਲਾਗਤ_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_ਲਾਗਤ_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_ਮਾਪ_candidate) == 0) {
            size_t exact_ਮਾਪ = htp_mm_hmx_get_batched_vtcm_ਮਾਪ(wtype, k, m_chunk_candidate, n_chunk_candidate, ਸਮੂਹ_ਮਾਪ, pipeline, act_ਧਾਗੇ, bias_ਮਾਪ);
            ਜੇ (exact_ਮਾਪ <= vtcm_budget) {
                size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                ਜੇ (mblocks < ਉੱਤਮ_mblocks || (mblocks == ਉੱਤਮ_mblocks && act_ਧਾਗੇ > ਉੱਤਮ_act_ਧਾਗੇ)) {
                    ਉੱਤਮ_mblocks = mblocks;
                    ਉੱਤਮ_act_ਧਾਗੇ = act_ਧਾਗੇ;
                    ਉੱਤਮ_m_chunk = m_chunk_candidate;
                    ਉੱਤਮ_n_chunk = n_chunk_candidate;
                    ਉੱਤਮ_vtcm_ਮਾਪ = exact_ਮਾਪ;
                }
            }
        }
        ਜੇ (act_ਧਾਗੇ == 1) {
            act_ਧਾਗੇ = 0;
        } ਨਹੀਂਤਾਂ {
            act_ਧਾਗੇ /= 2;
        }
    }

    ਜੇ (ਉੱਤਮ_act_ਧਾਗੇ > 0) {
        *m_chunk_ਬਾਹਰਲਾ = ਉੱਤਮ_m_chunk;
        *n_chunk_ਬਾਹਰਲਾ = ਉੱਤਮ_n_chunk;
        *vtcm_ਮਾਪ_ਬਾਹਰਲਾ = ਉੱਤਮ_vtcm_ਮਾਪ;
        *act_ਧਾਗੇ_ਬਾਹਰਲਾ = ਉੱਤਮ_act_ਧਾਗੇ;
        ਵਾਪਸ true;
    }
    ਵਾਪਸ false;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ bool htp_mm_hmx_solve_2d_params(
    ਪੂਰਨ wtype,
    uint32_t k,
    uint32_t m_id_ਪੰਕਤੀਆਂ,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_ਲਾਗਤ,
    ਪੂਰਨ n_ਧਾਗੇ,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_ਮਾਪ,
    size_t bias_ਮਾਪ,
    size_t vtcm_budget,
    size_t * m_chunk_ਬਾਹਰਲਾ,
    size_t * n_chunk_ਬਾਹਰਲਾ,
    ਪੂਰਨ * act_ਧਾਗੇ_ਬਾਹਰਲਾ,
    size_t * vtcm_ਮਾਪ_ਬਾਹਰਲਾ
) {
    size_t ਉੱਤਮ_mblocks = SIZE_MAX;
    ਪੂਰਨ ਉੱਤਮ_act_ਧਾਗੇ = 0;
    size_t ਉੱਤਮ_m_chunk = 0;
    size_t ਉੱਤਮ_n_chunk = 0;
    size_t ਉੱਤਮ_vtcm_ਮਾਪ = 0;

    ਅਚਲ ਪੂਰਨ m_for_chunks = is_matmul_id ? hex_align_ਉੱਪਰ(m_id_ਪੰਕਤੀਆਂ, 32) : ne11_padded;

    ਪੂਰਨ act_ਧਾਗੇ = n_ਧਾਗੇ;
    ਜਦੋਂ (act_ਧਾਗੇ >= 1) {
        size_t simple_2d_overhead = htp_mm_hmx_get_2d_overhead(pipeline, is_matmul_id) + (bias_ਮਾਪ > 0 ? hex_align_ਉੱਪਰ(bias_ਮਾਪ, HTP_MM_HMX_TILE_ਮਾਪ) : 0);
        size_t simple_2d_ਮਾਪ_ਪ੍ਰਤੀ_n, simple_2d_ਮਾਪ_ਪ੍ਰਤੀ_m, simple_2d_ਮਾਪ_ਪ੍ਰਤੀ_mn;
        htp_mm_hmx_get_2d_chunk_costs(wtype, k, pipeline, aligned_tile_ਮਾਪ, &simple_2d_ਮਾਪ_ਪ੍ਰਤੀ_n, &simple_2d_ਮਾਪ_ਪ੍ਰਤੀ_m, &simple_2d_ਮਾਪ_ਪ੍ਰਤੀ_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_ਮਾਪ_candidate = 0;

        ਜੇ (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_ਮਾਪ_ਪ੍ਰਤੀ_n, simple_2d_ਮਾਪ_ਪ੍ਰਤੀ_m, simple_2d_ਮਾਪ_ਪ੍ਰਤੀ_mn, m_for_chunks, ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_ਲਾਗਤ_W_DEQUANT, (size_t) m_for_ਲਾਗਤ * HTP_MM_HMX_ਲਾਗਤ_A_CONVERT,
                               &m_chunk_candidate, &n_chunk_candidate, &vtcm_ਮਾਪ_candidate) == 0) {
            size_t exact_ਮਾਪ = htp_mm_hmx_get_2d_vtcm_ਮਾਪ(wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_ਧਾਗੇ, aligned_tile_ਮਾਪ, bias_ਮਾਪ);
            ਜੇ (exact_ਮਾਪ <= vtcm_budget) {
                size_t mblocks = ((size_t) m_for_ਲਾਗਤ + m_chunk_candidate - 1) / m_chunk_candidate;
                ਜੇ (mblocks < ਉੱਤਮ_mblocks || (mblocks == ਉੱਤਮ_mblocks && act_ਧਾਗੇ > ਉੱਤਮ_act_ਧਾਗੇ)) {
                    ਉੱਤਮ_mblocks = mblocks;
                    ਉੱਤਮ_act_ਧਾਗੇ = act_ਧਾਗੇ;
                    ਉੱਤਮ_m_chunk = m_chunk_candidate;
                    ਉੱਤਮ_n_chunk = n_chunk_candidate;
                    ਉੱਤਮ_vtcm_ਮਾਪ = exact_ਮਾਪ;
                }
            }
        }
        ਜੇ (act_ਧਾਗੇ == 1) {
            act_ਧਾਗੇ = 0;
        } ਨਹੀਂਤਾਂ {
            act_ਧਾਗੇ /= 2;
        }
    }

    ਜੇ (ਉੱਤਮ_act_ਧਾਗੇ > 0) {
        *m_chunk_ਬਾਹਰਲਾ = ਉੱਤਮ_m_chunk;
        *n_chunk_ਬਾਹਰਲਾ = ਉੱਤਮ_n_chunk;
        *vtcm_ਮਾਪ_ਬਾਹਰਲਾ = ਉੱਤਮ_vtcm_ਮਾਪ;
        *act_ਧਾਗੇ_ਬਾਹਰਲਾ = ਉੱਤਮ_act_ਧਾਗੇ;
        ਵਾਪਸ true;
    }
    ਵਾਪਸ false;
}

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ __cplusplus
}
#ਜੇ_ਅੰਤ

#ਜੇ_ਅੰਤ // HTP_MATMUL_OPS_H
