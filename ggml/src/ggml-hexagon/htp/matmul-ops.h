#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HTP_MATMUL_OPS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MATMUL_OPS_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
ಬಾಹ್ಯ "C" {
#ಆದರೆ_ಅಂತ್ಯ

// --- HMX Tile Constraints ---
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ 32
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು 32
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_HMX_TILE_ಗಾತ್ರ   (32 * 32 * ಗಾತ್ರ(__fp16)) // 2048 bytes
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_HMX_TILE_N_ELMS 1024
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_HMX_MIN_NROWS   4

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_ತೂಕ_TILE_K_Q2_0             64
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_ತೂಕ_TILE_ಗಾತ್ರ_Q2_0          HTP_TENSOR_TILE_ಗಾತ್ರ_Q2_0
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_ತೂಕ_ALIGNED_TILE_ಗಾತ್ರ_Q2_0  640

// --- Activation Tiled Block Sizes (including padding) ---
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_ACT_TILE_ಗಾತ್ರ_Q8_0      1152
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_ACT_TILE_ಗಾತ್ರ_Q8_1      1280
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_ACT_TILE_ಗಾತ್ರ_F16       2048

#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_MAX_PREFETCH 16

// --- Solver Cost Model Penalty Weights (HMX-specific) ---
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_HMX_ವೆಚ್ಚ_W_DEQUANT 3 // cost penalty for quantized weight loading/dequantization
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_HMX_ವೆಚ್ಚ_A_CONVERT 2 // cost penalty for activation loading/conversion

// --- DMA Activation Transfer Configuration ---
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_DMA_ACT_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಹೆಜ್ಜೆ 2
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_MM_DMA_ACT_MULTIPLIER    (2 * HTP_MM_DMA_ACT_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_ಹೆಜ್ಜೆ)

ಎಣಿಕೆ htp_mm_kernel_type {
    HTP_MM_KERNEL_UNSUPPORTED = 0,

    // HMX paths
    HTP_MM_KERNEL_HMX_2D,
    HTP_MM_KERNEL_HMX_BATCHED,

    // HVX floating-point paths
    HTP_MM_KERNEL_HVX_F16_F16_VTCM,
    HTP_MM_KERNEL_HVX_F32_F32_VTCM,
    HTP_MM_KERNEL_HVX_BF16_F32_VTCM,

    // HVX quantized paths
    HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು,      // standard row-wise parallel quantization
    HTP_MM_KERNEL_HVX_QUANT_ಖಂಡ,    // parallel block-wise quantization
};

// Op-specific struct for precomputed matmul params
ರಚನೆ htp_mm_kernel_params {
    uint8_t  kernel_type;        // enum htp_mm_kernel_type
    uint8_t  pipeline;           // 1 = pipelined execution, 0 = standard
    uint8_t  collapse;           // 1 = collapse outer dims into 2D, 0 = standard
    uint8_t  n_hmx;              // 1 = use HMX, 0 = use HVX

    uint8_t  n_ಎಳೆಗಳು;          // Number of threads to spawn
    uint8_t  n_act_ಎಳೆಗಳು;      // Number of threads for activation preparation
    uint8_t  n_prefetch;         // Prefetch lookahead buffers/rows in VTCM
    uint8_t  n_weights;          // Number of weights for fused NX

    int32_t  m_chunk;            // Row chunk size (M chunk)
    int32_t  n_chunk;            // Col chunk size (N chunk)
    int32_t  tile_ಗಾತ್ರ;          // Weight tile size
    int32_t  aligned_tile_ಗಾತ್ರ;  // Aligned weight tile size (padded to 128)
    int32_t  act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;       // Row size for activation scratchpad
    int32_t  vtcm_ಗಾತ್ರ;          // Total required scratchpad size in VTCM
    int32_t  vtcm_src0_ಗಾತ್ರ;     // src0 scratchpad size in VTCM
    int32_t  vtcm_act_ಗಾತ್ರ;      // activation scratchpad size in VTCM
    int32_t  vtcm_bias_ಗಾತ್ರ;     // bias scratchpad size in VTCM (fused only)
    int32_t  vtcm_dst_ಗಾತ್ರ;      // dst scratchpad size in VTCM

    // Precomputed division values
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne12_ne1;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne1;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_r2;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_r3;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne12;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_n_act_ಎಳೆಗಳು;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne00_padded;
};

#ಆದರೆ defined(__cplusplus)
static_assert(ಗಾತ್ರ(ರಚನೆ htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#ಇಲ್ಲದಿದ್ದರೆ
_Static_assert(ಗಾತ್ರ(ರಚನೆ htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#ಆದರೆ_ಅಂತ್ಯ

ರಚನೆ mmid_ಅಡ್ಡಸಾಲು_mapping {
    uint32_t i1;
    uint32_t i2;
};

// Search for optimal (mc, nc) chunk sizes within VTCM budget.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ htp_mm_hmx_compute_chunks(size_t   vtcm_ಒಟ್ಟು,
                              size_t   overhead,
                              size_t   ಪ್ರತಿ_n_ವೆಚ್ಚ,
                              size_t   ಪ್ರತಿ_m_ವೆಚ್ಚ,
                              size_t   ಪ್ರತಿ_mn_ವೆಚ್ಚ,
                              size_t   m,
                              size_t   n,
                              size_t   m_ಖಂಡ_ವೆಚ್ಚ,
                              size_t   n_ಖಂಡ_ವೆಚ್ಚ,
                              size_t * m_chunk_ಹೊರಹೋಗು,
                              size_t * n_chunk_ಹೊರಹೋಗು,
                              size_t * ಒಟ್ಟು_ಹೊರಹೋಗು) {
    ಆದರೆ (m == 0 || n == 0) ಹಿಂತಿರುಗಿ -1;
    ಆದರೆ (vtcm_ಒಟ್ಟು <= overhead) ಹಿಂತಿರುಗಿ -1;
    ಆದರೆ (ಪ್ರತಿ_n_ವೆಚ್ಚ == 0 || ಪ್ರತಿ_m_ವೆಚ್ಚ == 0 || ಪ್ರತಿ_mn_ವೆಚ್ಚ == 0) ಹಿಂತಿರುಗಿ -1;

    ಸ್ಥಿರಾಂಕ size_t usable = vtcm_ಒಟ್ಟು - overhead;

    size_t ಉತ್ತಮ_ವೆಚ್ಚ = SIZE_MAX;
    size_t ಉತ್ತಮ_ಬಾಲ_waste = SIZE_MAX;
    size_t ಉತ್ತಮ_mn   = 0;
    size_t ಉತ್ತಮ_m = 0, ಉತ್ತಮ_n = 0;

    ಸ್ಥಿರಾಂಕ size_t max_nc_budget = (usable / ಪ್ರತಿ_n_ವೆಚ್ಚ);
    ಸ್ಥಿರಾಂಕ size_t n_max = hex_align_ಕೆಳಕ್ಕೆ(hex_smin((size_t)n, max_nc_budget), HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
    ಗಾಗಿ (size_t nc = n_max; nc >= HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ; nc -= HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) {
        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        ಆದರೆ (hex_mul_ಅತಿಪ್ರವಾಹ(nc, ಪ್ರತಿ_n_ವೆಚ್ಚ, &n_fixed)) ಮುಂದುವರಿ;
        ಆದರೆ (n_fixed >= usable) ಹೋಗು ಮುಂದಿನ_nc;

        ಆದರೆ (hex_mul_ಅತಿಪ್ರವಾಹ(nc, ಪ್ರತಿ_mn_ವೆಚ್ಚ, &ncmn)) ಹೋಗು ಮುಂದಿನ_nc;
        ಆದರೆ (hex_ಕೂಡಿಸು_ಅತಿಪ್ರವಾಹ(ಪ್ರತಿ_m_ವೆಚ್ಚ, ncmn, &mc_denom) || mc_denom == 0) ಹೋಗು ಮುಂದಿನ_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hex_align_ಕೆಳಕ್ಕೆ(mc, HTP_MM_HMX_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);
            mc = hex_smin(mc, m);

            ಆದರೆ (mc == 0) {
                ಹೋಗು ಮುಂದಿನ_nc;
            }

            size_t mblocks = ((size_t) m + mc - 1) / mc;
            size_t nblocks = ((size_t) n + nc - 1) / nc;
            size_t ವೆಚ್ಚ    = mblocks * m_ಖಂಡ_ವೆಚ್ಚ + nblocks * n_ಖಂಡ_ವೆಚ್ಚ;
            size_t rem     = n % nc;
            size_t ಬಾಲ_waste = (rem == 0) ? 0 : (nc - rem);
            size_t mn      = mc * nc;
            ಆದರೆ (ವೆಚ್ಚ < ಉತ್ತಮ_ವೆಚ್ಚ ||
                (ವೆಚ್ಚ == ಉತ್ತಮ_ವೆಚ್ಚ && ಬಾಲ_waste < ಉತ್ತಮ_ಬಾಲ_waste) ||
                (ವೆಚ್ಚ == ಉತ್ತಮ_ವೆಚ್ಚ && ಬಾಲ_waste == ಉತ್ತಮ_ಬಾಲ_waste && mn > ಉತ್ತಮ_mn)) {
                ಉತ್ತಮ_ವೆಚ್ಚ       = ವೆಚ್ಚ;
                ಉತ್ತಮ_ಬಾಲ_waste = ಬಾಲ_waste;
                ಉತ್ತಮ_mn         = mn;
                ಉತ್ತಮ_m          = mc;
                ಉತ್ತಮ_n          = nc;
            }
        }

ಮುಂದಿನ_nc:
        ಆದರೆ (nc == HTP_MM_HMX_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ) ನಿಲ್ಲಿಸು;  // avoid size_t underflow
    }

    ಆದರೆ (ಉತ್ತಮ_m == 0 || ಉತ್ತಮ_n == 0) ಹಿಂತಿರುಗಿ -1;

    // Compute exact total (with overflow checks)
    size_t t0 = 0, t1 = 0, t2 = 0, mn = 0, ಒಟ್ಟು = 0;
    ಆದರೆ (hex_mul_ಅತಿಪ್ರವಾಹ(ಉತ್ತಮ_n, ಪ್ರತಿ_n_ವೆಚ್ಚ, &t0)) ಹಿಂತಿರುಗಿ -1;
    ಆದರೆ (hex_mul_ಅತಿಪ್ರವಾಹ(ಉತ್ತಮ_m, ಪ್ರತಿ_m_ವೆಚ್ಚ, &t1)) ಹಿಂತಿರುಗಿ -1;
    ಆದರೆ (hex_mul_ಅತಿಪ್ರವಾಹ(ಉತ್ತಮ_m, ಉತ್ತಮ_n, &mn))     ಹಿಂತಿರುಗಿ -1;
    ಆದರೆ (hex_mul_ಅತಿಪ್ರವಾಹ(mn, ಪ್ರತಿ_mn_ವೆಚ್ಚ, &t2))    ಹಿಂತಿರುಗಿ -1;
    ಆದರೆ (hex_ಕೂಡಿಸು_ಅತಿಪ್ರವಾಹ(t0, t1, &ಒಟ್ಟು))          ಹಿಂತಿರುಗಿ -1;
    ಆದರೆ (hex_ಕೂಡಿಸು_ಅತಿಪ್ರವಾಹ(ಒಟ್ಟು, t2, &ಒಟ್ಟು))       ಹಿಂತಿರುಗಿ -1;
    ಆದರೆ (hex_ಕೂಡಿಸು_ಅತಿಪ್ರವಾಹ(ಒಟ್ಟು, overhead, &ಒಟ್ಟು)) ಹಿಂತಿರುಗಿ -1;

    *m_chunk_ಹೊರಹೋಗು = ಉತ್ತಮ_m;
    *n_chunk_ಹೊರಹೋಗು = ಉತ್ತಮ_n;
    *ಒಟ್ಟು_ಹೊರಹೋಗು   = ಒಟ್ಟು;
    ಹಿಂತಿರುಗಿ 0;
}

// weight types whose tiles carry a per-block offset (x = d * q + m): the activations need block sums (q8_1)
// (Q2_K: per-16 k sums, q8_1_s16)
ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_mm_ತೂಕ_has_ಸ್ಥಾನಾಂತರ(ಪೂರ್ಣಾಂಕ ತೂಕ_type) {
    ಹಿಂತಿರುಗಿ ತೂಕ_type == HTP_TYPE_Q4_1 || ತೂಕ_type == HTP_TYPE_Q4_K || ತೂಕ_type == HTP_TYPE_Q5_K ||
           ತೂಕ_type == HTP_TYPE_Q2_K || ತೂಕ_type == HTP_TYPE_Q2_0;
}

// --- Activation/Row Size Helpers ---
ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t htp_mm_q8_0_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(uint32_t ne) {
    ಸ್ಥಿರಾಂಕ uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ಸ್ಥಿರಾಂಕ uint32_t nb_32 = ne_padded / 32;
    ಹಿಂತಿರುಗಿ nb_32 * HTP_MM_ACT_TILE_ಗಾತ್ರ_Q8_0;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t htp_mm_q8_1_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(uint32_t ne) {
    ಸ್ಥಿರಾಂಕ uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ಸ್ಥಿರಾಂಕ uint32_t nb_32 = ne_padded / 32;
    ಹಿಂತಿರುಗಿ nb_32 * HTP_MM_ACT_TILE_ಗಾತ್ರ_Q8_1;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t htp_mm_f16_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(uint32_t ne) {
    ಸ್ಥಿರಾಂಕ uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ಸ್ಥಿರಾಂಕ uint32_t nb_32 = ne_padded / 32;
    ಹಿಂತಿರುಗಿ nb_32 * HTP_MM_ACT_TILE_ಗಾತ್ರ_F16;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t htp_mm_tiled_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ ತೂಕ_type, uint32_t ne) {
    ಆದರೆ (ತೂಕ_type == HTP_TYPE_F16) {
        ಹಿಂತಿರುಗಿ htp_mm_f16_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ne);
    }
    ಹಿಂತಿರುಗಿ htp_mm_ತೂಕ_has_ಸ್ಥಾನಾಂತರ(ತೂಕ_type) ? htp_mm_q8_1_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ne) : htp_mm_q8_0_tiled_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(ne);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t htp_mm_ಸುತ್ತು_ಮೇಲೆ(size_t n, size_t m) {
    ಹಿಂತಿರುಗಿ ((n + m - 1) / m) * m;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_mm_hmx_pipeline(uint32_t m) {
    ಹಿಂತಿರುಗಿ m > 32;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_mm_hmx_get_2d_chunk_costs(
    ಪೂರ್ಣಾಂಕ wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_ಗಾತ್ರ,
    size_t * ಗಾತ್ರ_ಪ್ರತಿ_n_ಹೊರಹೋಗು, size_t * ಗಾತ್ರ_ಪ್ರತಿ_m_ಹೊರಹೋಗು, size_t * ಗಾತ್ರ_ಪ್ರತಿ_mn_ಹೊರಹೋಗು
) {
    ಸ್ಥಿರಾಂಕ bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride = htp_tensor_get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(wtype, k);
    ಸ್ಥಿರಾಂಕ size_t vec_dot_ಗಾತ್ರ = k * ಗಾತ್ರ(uint16_t);
    ಸ್ಥಿರಾಂಕ uint32_t n_ತೂಕ_tiles = k / htp_tensor_get_tile_k(wtype);
    ಸ್ಥಿರಾಂಕ size_t qweight_ಅಡ್ಡಸಾಲು_stride = is_quant ? (size_t)(n_ತೂಕ_tiles * aligned_tile_ಗಾತ್ರ) / 32 : 0;

    *ಗಾತ್ರ_ಪ್ರತಿ_n_ಹೊರಹೋಗು = (pipeline ? 2 : 1) * (is_quant ? qweight_ಅಡ್ಡಸಾಲು_stride : ಅಡ್ಡಸಾಲು_stride) +
                      (pipeline ? 2 * vec_dot_ಗಾತ್ರ : vec_dot_ಗಾತ್ರ);
    *ಗಾತ್ರ_ಪ್ರತಿ_m_ಹೊರಹೋಗು = vec_dot_ಗಾತ್ರ;
    *ಗಾತ್ರ_ಪ್ರತಿ_mn_ಹೊರಹೋಗು = (pipeline ? 2 : 1) * ಗಾತ್ರ(uint16_t);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_mm_hmx_get_batched_chunk_costs(
    uint32_t k, uint32_t ಗುಂಪು_ಗಾತ್ರ,
    size_t * ಗಾತ್ರ_ಪ್ರತಿ_n_ಹೊರಹೋಗು, size_t * ಗಾತ್ರ_ಪ್ರತಿ_m_ಹೊರಹೋಗು, size_t * ಗಾತ್ರ_ಪ್ರತಿ_mn_ಹೊರಹೋಗು
) {
    ಸ್ಥಿರಾಂಕ size_t vec_dot_ಗಾತ್ರ = k * ಗಾತ್ರ(uint16_t);
    *ಗಾತ್ರ_ಪ್ರತಿ_n_ಹೊರಹೋಗು = 3 * vec_dot_ಗಾತ್ರ;
    *ಗಾತ್ರ_ಪ್ರತಿ_m_ಹೊರಹೋಗು = ಗುಂಪು_ಗಾತ್ರ * vec_dot_ಗಾತ್ರ;
    *ಗಾತ್ರ_ಪ್ರತಿ_mn_ಹೊರಹೋಗು = ಗಾತ್ರ(uint16_t);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t htp_mm_hmx_get_2d_overhead(bool pipeline, bool is_matmul_id) {
    size_t ಅಂಕೆ_regions = pipeline ? 7 : (is_matmul_id ? 4 : 5);
    ಹಿಂತಿರುಗಿ ಅಂಕೆ_regions * HTP_MM_HMX_TILE_ಗಾತ್ರ + 256;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t htp_mm_hmx_get_batched_overhead(ಶೂನ್ಯ) {
    ಹಿಂತಿರುಗಿ 5 * HTP_MM_HMX_TILE_ಗಾತ್ರ + 256;
}

ರಚನೆ htp_mm_hmx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_ತೂಕ[2];     // [1] is only used when pipelined
    size_t off_act;
    size_t off_act_f32;       // fp32 activation conversion scratch
    size_t off_dst[2];        // [1] is only used when pipelined
    size_t off_scratch[2];    // dequantization scratch pads
    size_t off_scales;        // HMX scales (256 bytes)
    size_t off_bias;          // bias in VTCM

    // Cached sizes of regions for HMX kernel use
    size_t ತೂಕ_area_ಬೈಟುಗಳು;
    size_t act_area_ಬೈಟುಗಳು;
    size_t act_f32_ಬೈಟುಗಳು;
    size_t output_area_ಬೈಟುಗಳು;
    size_t scratch_ಬೈಟುಗಳು[2];
    size_t act_ತಲೆ_stride;
    size_t bias_ಬೈಟುಗಳು;

    size_t ಒಟ್ಟು_ಬೈಟುಗಳು;
};

ರಚನೆ htp_mm_hvx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_act;           // vtcm_act (activation)
    size_t off_src0;          // vtcm_src0 (weight/Wk)
    size_t off_bias;          // vtcm_bias (bias / fused add only)
    size_t off_dst;           // vtcm_dst (output scratch)
    size_t off_act_raw;       // vtcm_act_raw (raw activation DMA staging)

    // Cached sizes
    size_t src0_ಬೈಟುಗಳು;
    size_t act_ಬೈಟುಗಳು;
    size_t bias_ಬೈಟುಗಳು;
    size_t dst_ಬೈಟುಗಳು;
    size_t act_raw_ಬೈಟುಗಳು;

    size_t ಒಟ್ಟು_ಬೈಟುಗಳು;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_mm_hmx_vtcm_layout_build(
    ರಚನೆ htp_mm_hmx_vtcm_layout * L,
    ಪೂರ್ಣಾಂಕ kernel_type,
    ಪೂರ್ಣಾಂಕ wtype,
    uint32_t k,
    size_t mc,
    size_t nc,
    uint32_t ಗುಂಪು_ಗಾತ್ರ,
    bool pipeline,
    uint32_t act_ಎಳೆಗಳು,
    uint32_t aligned_tile_ಗಾತ್ರ,
    size_t bias_ಗಾತ್ರ
) {
    size_t off = 0;

    ಆದರೆ (kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        ಸ್ಥಿರಾಂಕ size_t vec_dot_ಗಾತ್ರ     = k * ಗಾತ್ರ(uint16_t);
        ಸ್ಥಿರಾಂಕ size_t act_ತಲೆ_stride   = mc * k;
        ಸ್ಥಿರಾಂಕ size_t ತೂಕ_area_ಗಾತ್ರ  = hex_align_ಮೇಲೆ(nc * vec_dot_ಗಾತ್ರ, HTP_MM_HMX_TILE_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ size_t activation_area_ಗಾತ್ರ = hex_align_ಮೇಲೆ(ಗುಂಪು_ಗಾತ್ರ * act_ತಲೆ_stride * ಗಾತ್ರ(uint16_t), HTP_MM_HMX_TILE_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ size_t output_area_ಗಾತ್ರ  = hex_align_ಮೇಲೆ(ಗುಂಪು_ಗಾತ್ರ * mc * nc * ಗಾತ್ರ(uint16_t), HTP_MM_HMX_TILE_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ size_t scratch_area_ಗಾತ್ರ = hex_align_ಮೇಲೆ(nc * vec_dot_ಗಾತ್ರ, HTP_MM_HMX_TILE_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ size_t min_f32_ಗಾತ್ರ = hex_align_ಮೇಲೆ(act_ಎಳೆಗಳು * HTP_MM_DMA_ACT_MULTIPLIER * k * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);

        // Group A: Permanent activation tiles and scales
        size_t off_ಗುಂಪು_a = 0;
        VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_a, off_act, activation_area_ಗಾತ್ರ);
        VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_a, off_scales, HTP_MM_HMX_TILE_ಗಾತ್ರ); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ಗುಂಪು_a, off_bias, hex_align_ಮೇಲೆ(bias_ಗಾತ್ರ, HTP_MM_HMX_TILE_ಗಾತ್ರ), bias_ಗಾತ್ರ > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_ಗುಂಪು_b = off_ಗುಂಪು_a;
        VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_ತೂಕ[0], ತೂಕ_area_ಗಾತ್ರ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ಗುಂಪು_b, off_ತೂಕ[1], ತೂಕ_area_ಗಾತ್ರ, false);
        VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_dst[0], output_area_ಗಾತ್ರ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ಗುಂಪು_b, off_dst[1], output_area_ಗಾತ್ರ, false);
        VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_scratch[0], scratch_area_ಗಾತ್ರ);
        VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_scratch[1], scratch_area_ಗಾತ್ರ);

        ಸ್ಥಿರಾಂಕ size_t ಗುಂಪು_b_ಗಾತ್ರ = off_ಗುಂಪು_b - off_ಗುಂಪು_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ಸ್ಥಿರಾಂಕ size_t max_f32_ಗಾತ್ರ = act_ಎಳೆಗಳು * 64 * k * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಸ್ಥಿರಾಂಕ size_t act_f32_ಗಾತ್ರ = hex_align_ಮೇಲೆ(hex_smin(max_f32_ಗಾತ್ರ, hex_smax(min_f32_ಗಾತ್ರ, ಗುಂಪು_b_ಗಾತ್ರ)), 128);
        size_t off_ಗುಂಪು_c = off_ಗುಂಪು_a;
        VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_c, off_act_f32, act_f32_ಗಾತ್ರ);

        ಸ್ಥಿರಾಂಕ size_t ಗುಂಪು_c_ಗಾತ್ರ = off_ಗುಂಪು_c - off_ಗುಂಪು_a;

        L->ತೂಕ_area_ಬೈಟುಗಳು = ತೂಕ_area_ಗಾತ್ರ;
        L->act_area_ಬೈಟುಗಳು    = activation_area_ಗಾತ್ರ;
        L->act_f32_ಬೈಟುಗಳು     = act_f32_ಗಾತ್ರ;
        L->output_area_ಬೈಟುಗಳು = output_area_ಗಾತ್ರ;
        L->scratch_ಬೈಟುಗಳು[0]  = scratch_area_ಗಾತ್ರ;
        L->scratch_ಬೈಟುಗಳು[1]  = scratch_area_ಗಾತ್ರ;
        L->act_ತಲೆ_stride   = act_ತಲೆ_stride;
        L->bias_ಬೈಟುಗಳು        = bias_ಗಾತ್ರ;

        off = off_ಗುಂಪು_a + hex_smax(ಗುಂಪು_b_ಗಾತ್ರ, ಗುಂಪು_c_ಗಾತ್ರ);
    } ಇಲ್ಲದಿದ್ದರೆ {
        // HTP_MM_KERNEL_HMX_2D
        ಸ್ಥಿರಾಂಕ bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
        ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride = htp_tensor_get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(wtype, k);
        ಸ್ಥಿರಾಂಕ size_t vec_dot_ಗಾತ್ರ = k * ಗಾತ್ರ(uint16_t);
        ಸ್ಥಿರಾಂಕ uint32_t n_ತೂಕ_tiles = k / htp_tensor_get_tile_k(wtype);

        ಸ್ಥಿರಾಂಕ size_t min_f32_ಗಾತ್ರ = hex_align_ಮೇಲೆ(act_ಎಳೆಗಳು * HTP_MM_DMA_ACT_MULTIPLIER * k * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);
        ಸ್ಥಿರಾಂಕ size_t ತೂಕ_area_ಗಾತ್ರ = is_quant
            ? hex_align_ಮೇಲೆ((nc / 32) * n_ತೂಕ_tiles * aligned_tile_ಗಾತ್ರ, HTP_MM_HMX_TILE_ಗಾತ್ರ)
            : hex_align_ಮೇಲೆ(nc * ಅಡ್ಡಸಾಲು_stride, HTP_MM_HMX_TILE_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ size_t act_area_ಗಾತ್ರ    = hex_align_ಮೇಲೆ(mc * vec_dot_ಗಾತ್ರ, HTP_MM_HMX_TILE_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ size_t output_area_ಗಾತ್ರ = hex_align_ಮೇಲೆ(mc * nc * ಗಾತ್ರ(__fp16), HTP_MM_HMX_TILE_ಗಾತ್ರ);

        ಸ್ಥಿರಾಂಕ size_t scratch0_ಗಾತ್ರ = hex_align_ಮೇಲೆ(nc * vec_dot_ಗಾತ್ರ, HTP_MM_HMX_TILE_ಗಾತ್ರ);
        ಸ್ಥಿರಾಂಕ size_t scratch1_ಗಾತ್ರ = pipeline ? scratch0_ಗಾತ್ರ : 0;

        // Group A:  Scales and activation tiles (must not overlap with Group B or C)
        size_t off_ಗುಂಪು_a = 0;
        VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_a, off_scales, HTP_MM_HMX_TILE_ಗಾತ್ರ); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_a, off_act, act_area_ಗಾತ್ರ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ಗುಂಪು_a, off_bias, hex_align_ಮೇಲೆ(bias_ಗಾತ್ರ, HTP_MM_HMX_TILE_ಗಾತ್ರ), bias_ಗಾತ್ರ > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_ಗುಂಪು_b = off_ಗುಂಪು_a;
        VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_ತೂಕ[0], ತೂಕ_area_ಗಾತ್ರ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ಗುಂಪು_b, off_ತೂಕ[1], ತೂಕ_area_ಗಾತ್ರ, pipeline);
        VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_dst[0], output_area_ಗಾತ್ರ);
        VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_scratch[0], scratch0_ಗಾತ್ರ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ಗುಂಪು_b, off_scratch[1], scratch0_ಗಾತ್ರ, pipeline);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ಗುಂಪು_b, off_dst[1], output_area_ಗಾತ್ರ, pipeline);

        ಸ್ಥಿರಾಂಕ size_t ಗುಂಪು_b_ಗಾತ್ರ = off_ಗುಂಪು_b - off_ಗುಂಪು_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ಸ್ಥಿರಾಂಕ size_t max_f32_ಗಾತ್ರ = act_ಎಳೆಗಳು * 64 * k * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ಸ್ಥಿರಾಂಕ size_t act_f32_ಗಾತ್ರ = hex_align_ಮೇಲೆ(hex_smin(max_f32_ಗಾತ್ರ, hex_smax(min_f32_ಗಾತ್ರ, ಗುಂಪು_b_ಗಾತ್ರ)), 128);
        size_t off_ಗುಂಪು_c = off_ಗುಂಪು_a;
        VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_c, off_act_f32, act_f32_ಗಾತ್ರ);

        ಸ್ಥಿರಾಂಕ size_t ಗುಂಪು_c_ಗಾತ್ರ = off_ಗುಂಪು_c - off_ಗುಂಪು_a;

        L->ತೂಕ_area_ಬೈಟುಗಳು = ತೂಕ_area_ಗಾತ್ರ;
        L->act_area_ಬೈಟುಗಳು    = act_area_ಗಾತ್ರ;
        L->act_f32_ಬೈಟುಗಳು     = act_f32_ಗಾತ್ರ;
        L->output_area_ಬೈಟುಗಳು = output_area_ಗಾತ್ರ;
        L->scratch_ಬೈಟುಗಳು[0]  = scratch0_ಗಾತ್ರ;
        L->scratch_ಬೈಟುಗಳು[1]  = scratch1_ಗಾತ್ರ;
        L->act_ತಲೆ_stride   = 0;
        L->bias_ಬೈಟುಗಳು        = bias_ಗಾತ್ರ;

        off = off_ಗುಂಪು_a + hex_smax(ಗುಂಪು_b_ಗಾತ್ರ, ಗುಂಪು_c_ಗಾತ್ರ);
    }

    L->ಒಟ್ಟು_ಬೈಟುಗಳು = off;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ htp_mm_hvx_vtcm_layout_build(
    ರಚನೆ htp_mm_hvx_vtcm_layout * L,
    ಪೂರ್ಣಾಂಕ kernel_type,
    ಪೂರ್ಣಾಂಕ wtype,
    uint32_t ne10,       // k
    uint32_t act_nrows,  // m_total
    uint32_t n_ಎಳೆಗಳು,
    size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    size_t act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    size_t bias_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_nx
) {
    (ಶೂನ್ಯ)act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    size_t src0_sz    = 0;
    size_t act_sz     = 0;
    size_t bias_sz    = bias_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ > 0 ? htp_mm_ಸುತ್ತು_ಮೇಲೆ(bias_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128) : 0;
    size_t dst_sz     = 0;
    size_t act_raw_sz = 0;

    ಆದರೆ (is_fused_nx) {
        uint32_t aligned_tile_ಗಾತ್ರ = htp_tensor_get_tile_padded_ಗಾತ್ರ(wtype);
        uint32_t n_k_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        uint32_t tile_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = n_k_tiles * aligned_tile_ಗಾತ್ರ;
        size_t ತೂಕ_sz_ಪ್ರತಿ_thread = hex_ಸುತ್ತು_ಮೇಲೆ(n_prefetch * tile_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128);

        size_t tiled_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_mm_tiled_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(wtype, ne10);
        size_t q_act_sz = hex_ಸುತ್ತು_ಮೇಲೆ(tiled_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ * act_nrows, 128);
        size_t raw_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(ne10 * ಗಾತ್ರ(ದಶಮಾಂಶ), QK_Q8_0_TILED * ಗಾತ್ರ(ದಶಮಾಂಶ));

        src0_sz    = ತೂಕ_sz_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು; // shared single-weight prefetch buffer
        act_sz     = q_act_sz;                         // quantized activation buffer
        bias_sz    = 0;
        dst_sz     = 0;
        act_raw_sz = hex_ಸುತ್ತು_ಮೇಲೆ(raw_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ * act_nrows, 128);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (is_matmul_id) {
        ಸ್ಥಿರಾಂಕ size_t act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_tiled = htp_mm_tiled_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(wtype, ne10);
        ಸ್ಥಿರಾಂಕ uint32_t aligned_tile_ಗಾತ್ರ = htp_tensor_get_tile_padded_ಗಾತ್ರ(wtype);
        ಸ್ಥಿರಾಂಕ uint32_t n_k_tiles         = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        ಸ್ಥಿರಾಂಕ uint32_t tile_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ     = n_k_tiles * aligned_tile_ಗಾತ್ರ;
        size_t src0_sz_ಪ್ರತಿ_thread        = htp_mm_ಸುತ್ತು_ಮೇಲೆ(n_prefetch * tile_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 256);

        act_sz = htp_mm_ಸುತ್ತು_ಮೇಲೆ(act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_tiled * act_nrows, 256);

        size_t raw_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(ne10 * ಗಾತ್ರ(ದಶಮಾಂಶ), QK_Q8_0_TILED * ಗಾತ್ರ(ದಶಮಾಂಶ));

        src0_sz    = src0_sz_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
        dst_sz     = 0;
        bias_sz    = 0;
        act_raw_sz = hex_ಸುತ್ತು_ಮೇಲೆ(raw_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ * act_nrows, 128);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded = htp_mm_ಸುತ್ತು_ಮೇಲೆ(src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128);
        ಸ್ಥಿರಾಂಕ size_t dst_nrows = (act_nrows > 1) ? 0 : 1;

        ಆಯ್ಕೆ (kernel_type) {
            ಸ್ಥಿತಿ HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                size_t f16_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_mm_ಸುತ್ತು_ಮೇಲೆ(ne10 * 2, 128);
                act_sz     = htp_mm_ಸುತ್ತು_ಮೇಲೆ(f16_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ * act_nrows, 256);
                src0_sz    = htp_mm_ಸುತ್ತು_ಮೇಲೆ(n_prefetch * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded, 256) * n_ಎಳೆಗಳು;
                dst_sz     = dst_nrows > 0 ? htp_mm_ಸುತ್ತು_ಮೇಲೆ(dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128) * n_ಎಳೆಗಳು : 0;
                act_raw_sz = hex_ಸುತ್ತು_ಮೇಲೆ(hex_ಸುತ್ತು_ಮೇಲೆ(ne10 * ಗಾತ್ರ(ದಶಮಾಂಶ), 128) * act_nrows, 128);
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            ಸ್ಥಿತಿ HTP_MM_KERNEL_HVX_BF16_F32_VTCM: {
                size_t f32_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_mm_ಸುತ್ತು_ಮೇಲೆ(ne10 * 4, 128);
                act_sz     = htp_mm_ಸುತ್ತು_ಮೇಲೆ(f32_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ * act_nrows, 256);
                src0_sz    = htp_mm_ಸುತ್ತು_ಮೇಲೆ(n_prefetch * src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_padded, 256) * n_ಎಳೆಗಳು;
                dst_sz     = dst_nrows > 0 ? htp_mm_ಸುತ್ತು_ಮೇಲೆ(dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 128) * n_ಎಳೆಗಳು : 0;
                act_raw_sz = 0;
                ನಿಲ್ಲಿಸು;
            }
            ಸ್ಥಿತಿ HTP_MM_KERNEL_HVX_QUANT_ಖಂಡ:
            ಸ್ಥಿತಿ HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು: {
                size_t q_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_mm_tiled_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(wtype, ne10);
                uint32_t aligned_tile_ಗಾತ್ರ = htp_tensor_get_tile_padded_ಗಾತ್ರ(wtype);
                uint32_t n_ತೂಕ_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
                uint32_t tile_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = n_ತೂಕ_tiles * aligned_tile_ಗಾತ್ರ;
                size_t tiled_vtcm_ಗಾತ್ರ = htp_mm_ಸುತ್ತು_ಮೇಲೆ(n_prefetch * tile_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 256);

                src0_sz = tiled_vtcm_ಗಾತ್ರ * n_ಎಳೆಗಳು;
                act_sz  = htp_mm_ಸುತ್ತು_ಮೇಲೆ(q_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ * act_nrows, 256);

                size_t dst_ತುಂಡು_ಪ್ರತಿ_thread = (dst_nrows > 0 && act_nrows == 1) ? htp_mm_ಸುತ್ತು_ಮೇಲೆ((dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ + n_ಎಳೆಗಳು - 1) / n_ಎಳೆಗಳು, 128) : 0;
                dst_sz = dst_ತುಂಡು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು;
                size_t raw_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(ne10 * ಗಾತ್ರ(ದಶಮಾಂಶ), QK_Q8_0_TILED * ಗಾತ್ರ(ದಶಮಾಂಶ));
                act_raw_sz = hex_ಸುತ್ತು_ಮೇಲೆ(raw_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ * act_nrows, 128);
                ನಿಲ್ಲಿಸು;
            }
            ಪೂರ್ವನಿಯೋಜಿತ:
                ನಿಲ್ಲಿಸು;
        }
    }

    // Group A: Persistent buffers across chunk compute
    size_t off_ಗುಂಪು_a = 0;
    VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_a, off_act, act_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ಗುಂಪು_a, off_bias, bias_sz, bias_sz > 0);

    // Group B: Compute-only buffers (starts at off_group_a)
    size_t off_ಗುಂಪು_b = off_ಗುಂಪು_a;
    VTCM_LAYOUT_ALLOC(off_ಗುಂಪು_b, off_src0, src0_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ಗುಂಪು_b, off_dst, dst_sz, dst_sz > 0);
    ಸ್ಥಿರಾಂಕ size_t ಗುಂಪು_b_ಗಾತ್ರ = off_ಗುಂಪು_b - off_ಗುಂಪು_a;

    // Group C: Raw activation staging buffer (overlaps Group B, starts at off_group_a)
    size_t off_ಗುಂಪು_c = off_ಗುಂಪು_a;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ಗುಂಪು_c, off_act_raw, act_raw_sz, act_raw_sz > 0);
    ಸ್ಥಿರಾಂಕ size_t ಗುಂಪು_c_ಗಾತ್ರ = off_ಗುಂಪು_c - off_ಗುಂಪು_a;

    L->src0_ಬೈಟುಗಳು    = src0_sz;
    L->act_ಬೈಟುಗಳು     = act_sz;
    L->bias_ಬೈಟುಗಳು    = bias_sz;
    L->dst_ಬೈಟುಗಳು     = dst_sz;
    L->act_raw_ಬೈಟುಗಳು = act_raw_sz;
    L->ಒಟ್ಟು_ಬೈಟುಗಳು   = off_ಗುಂಪು_a + hex_smax(ಗುಂಪು_b_ಗಾತ್ರ, ಗುಂಪು_c_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_mm_hvx_solve_vtcm_params(
    ಪೂರ್ಣಾಂಕ kernel_type,
    ಪೂರ್ಣಾಂಕ wtype,
    uint32_t ne10,
    uint32_t act_nrows,
    uint32_t n_ಎಳೆಗಳು,
    size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    size_t act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    size_t bias_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ,
    uint32_t n_prefetch,
    size_t vtcm_budget,
    ರಚನೆ htp_mm_hvx_vtcm_layout * L_ಹೊರಹೋಗು,
    uint32_t * m_chunk_ಹೊರಹೋಗು
) {
    ರಚನೆ htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, act_nrows, n_ಎಳೆಗಳು,
        dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, bias_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, n_prefetch, false, false
    );

    ಆದರೆ (L.ಒಟ್ಟು_ಬೈಟುಗಳು <= vtcm_budget) {
        *L_ಹೊರಹೋಗು = L;
        *m_chunk_ಹೊರಹೋಗು = act_nrows;
        ಹಿಂತಿರುಗಿ true;
    }

    ಸ್ಥಿರಾಂಕ size_t fixed_ಬೈಟುಗಳು = L.src0_ಬೈಟುಗಳು + L.bias_ಬೈಟುಗಳು + L.dst_ಬೈಟುಗಳು;
    ಆದರೆ (vtcm_budget <= fixed_ಬೈಟುಗಳು) {
        ಹಿಂತಿರುಗಿ false;
    }

    ಸ್ಥಿರಾಂಕ size_t avail_act = vtcm_budget - fixed_ಬೈಟುಗಳು;
    size_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = 0;
    ಆದರೆ (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು || kernel_type == HTP_MM_KERNEL_HVX_QUANT_ಖಂಡ) {
        ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = htp_mm_tiled_act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(wtype, ne10);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(ne10 * 2, 128);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = hex_ಸುತ್ತು_ಮೇಲೆ(ne10 * 4, 128);
    }
    ಆದರೆ (ಅಡ್ಡಸಾಲು_ಗಾತ್ರ == 0) {
        ಹಿಂತಿರುಗಿ false;
    }

    size_t eff_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    ಆದರೆ (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ಅಡ್ಡಸಾಲು || kernel_type == HTP_MM_KERNEL_HVX_QUANT_ಖಂಡ) {
        eff_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ += hex_ಸುತ್ತು_ಮೇಲೆ(ne10 * ಗಾತ್ರ(ದಶಮಾಂಶ), QK_Q8_0_TILED * ಗಾತ್ರ(ದಶಮಾಂಶ));
    }

    uint32_t m_chunk = (uint32_t) (avail_act / eff_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ);
    ಆದರೆ (m_chunk > 1) {
        m_chunk &= ~1U;
    }
    ಆದರೆ (m_chunk > act_nrows) {
        m_chunk = act_nrows;
    }
    ಆದರೆ (m_chunk < 1) {
        ಹಿಂತಿರುಗಿ false;
    }

    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, m_chunk, n_ಎಳೆಗಳು,
        dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, bias_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, n_prefetch, false, false
    );

    ಆಗ (m_chunk > 2 && L.ಒಟ್ಟು_ಬೈಟುಗಳು > vtcm_budget) {
        m_chunk -= 2;
        htp_mm_hvx_vtcm_layout_build(
            &L, kernel_type, wtype, ne10, m_chunk, n_ಎಳೆಗಳು,
            dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, act_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, bias_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, n_prefetch, false, false
        );
    }

    ಆದರೆ (L.ಒಟ್ಟು_ಬೈಟುಗಳು <= vtcm_budget) {
        *L_ಹೊರಹೋಗು = L;
        *m_chunk_ಹೊರಹೋಗು = m_chunk;
        ಹಿಂತಿರುಗಿ true;
    }

    ಹಿಂತಿರುಗಿ false;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t htp_mm_hmx_get_2d_vtcm_ಗಾತ್ರ(
    ಪೂರ್ಣಾಂಕ wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_ಎಳೆಗಳು, uint32_t aligned_tile_ಗಾತ್ರ, size_t bias_ಗಾತ್ರ
) {
    ರಚನೆ htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_ಎಳೆಗಳು, aligned_tile_ಗಾತ್ರ, bias_ಗಾತ್ರ);
    ಹಿಂತಿರುಗಿ L.ಒಟ್ಟು_ಬೈಟುಗಳು;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t htp_mm_hmx_get_batched_vtcm_ಗಾತ್ರ(
    ಪೂರ್ಣಾಂಕ wtype, uint32_t k, size_t mc, size_t nc, uint32_t ಗುಂಪು_ಗಾತ್ರ, bool pipeline, uint32_t act_ಎಳೆಗಳು, size_t bias_ಗಾತ್ರ) {
    (ಶೂನ್ಯ)pipeline;
    ರಚನೆ htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, wtype, k, mc, nc, ಗುಂಪು_ಗಾತ್ರ, false, act_ಎಳೆಗಳು, 0, bias_ಗಾತ್ರ);
    ಹಿಂತಿರುಗಿ L.ಒಟ್ಟು_ಬೈಟುಗಳು;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_mm_hmx_solve_batched_params(
    ಪೂರ್ಣಾಂಕ wtype,
    uint32_t k,
    uint32_t ne01_padded,
    uint32_t ne11,
    uint32_t ಗುಂಪು_ಗಾತ್ರ,
    ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು,
    bool pipeline,
    size_t bias_ಗಾತ್ರ,
    size_t vtcm_budget,
    size_t * m_chunk_ಹೊರಹೋಗು,
    size_t * n_chunk_ಹೊರಹೋಗು,
    ಪೂರ್ಣಾಂಕ * act_ಎಳೆಗಳು_ಹೊರಹೋಗು,
    size_t * vtcm_ಗಾತ್ರ_ಹೊರಹೋಗು
) {
    size_t ಉತ್ತಮ_mblocks = SIZE_MAX;
    ಪೂರ್ಣಾಂಕ ಉತ್ತಮ_act_ಎಳೆಗಳು = 0;
    size_t ಉತ್ತಮ_m_chunk = 0;
    size_t ಉತ್ತಮ_n_chunk = 0;
    size_t ಉತ್ತಮ_vtcm_ಗಾತ್ರ = 0;

    ಪೂರ್ಣಾಂಕ act_ಎಳೆಗಳು = n_ಎಳೆಗಳು;
    ಆಗ (act_ಎಳೆಗಳು >= 1) {
        size_t ಗುಂಪು_overhead = htp_mm_hmx_get_batched_overhead() + (bias_ಗಾತ್ರ > 0 ? hex_align_ಮೇಲೆ(bias_ಗಾತ್ರ, HTP_MM_HMX_TILE_ಗಾತ್ರ) : 0);
        size_t ಗುಂಪು_ಗಾತ್ರ_ಪ್ರತಿ_n, ಗುಂಪು_ಗಾತ್ರ_ಪ್ರತಿ_m, ಗುಂಪು_ಗಾತ್ರ_ಪ್ರತಿ_mn;
        htp_mm_hmx_get_batched_chunk_costs(k, ಗುಂಪು_ಗಾತ್ರ, &ಗುಂಪು_ಗಾತ್ರ_ಪ್ರತಿ_n, &ಗುಂಪು_ಗಾತ್ರ_ಪ್ರತಿ_m, &ಗುಂಪು_ಗಾತ್ರ_ಪ್ರತಿ_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_ಗಾತ್ರ_candidate = 0;

        ಆದರೆ (htp_mm_hmx_compute_chunks(vtcm_budget, ಗುಂಪು_overhead, ಗುಂಪು_ಗಾತ್ರ_ಪ್ರತಿ_n, ಗುಂಪು_ಗಾತ್ರ_ಪ್ರತಿ_m, ಗುಂಪು_ಗಾತ್ರ_ಪ್ರತಿ_mn, hex_align_ಮೇಲೆ(ne11, 32), ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_ವೆಚ್ಚ_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_ವೆಚ್ಚ_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_ಗಾತ್ರ_candidate) == 0) {
            size_t exact_ಗಾತ್ರ = htp_mm_hmx_get_batched_vtcm_ಗಾತ್ರ(wtype, k, m_chunk_candidate, n_chunk_candidate, ಗುಂಪು_ಗಾತ್ರ, pipeline, act_ಎಳೆಗಳು, bias_ಗಾತ್ರ);
            ಆದರೆ (exact_ಗಾತ್ರ <= vtcm_budget) {
                size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                ಆದರೆ (mblocks < ಉತ್ತಮ_mblocks || (mblocks == ಉತ್ತಮ_mblocks && act_ಎಳೆಗಳು > ಉತ್ತಮ_act_ಎಳೆಗಳು)) {
                    ಉತ್ತಮ_mblocks = mblocks;
                    ಉತ್ತಮ_act_ಎಳೆಗಳು = act_ಎಳೆಗಳು;
                    ಉತ್ತಮ_m_chunk = m_chunk_candidate;
                    ಉತ್ತಮ_n_chunk = n_chunk_candidate;
                    ಉತ್ತಮ_vtcm_ಗಾತ್ರ = exact_ಗಾತ್ರ;
                }
            }
        }
        ಆದರೆ (act_ಎಳೆಗಳು == 1) {
            act_ಎಳೆಗಳು = 0;
        } ಇಲ್ಲದಿದ್ದರೆ {
            act_ಎಳೆಗಳು /= 2;
        }
    }

    ಆದರೆ (ಉತ್ತಮ_act_ಎಳೆಗಳು > 0) {
        *m_chunk_ಹೊರಹೋಗು = ಉತ್ತಮ_m_chunk;
        *n_chunk_ಹೊರಹೋಗು = ಉತ್ತಮ_n_chunk;
        *vtcm_ಗಾತ್ರ_ಹೊರಹೋಗು = ಉತ್ತಮ_vtcm_ಗಾತ್ರ;
        *act_ಎಳೆಗಳು_ಹೊರಹೋಗು = ಉತ್ತಮ_act_ಎಳೆಗಳು;
        ಹಿಂತಿರುಗಿ true;
    }
    ಹಿಂತಿರುಗಿ false;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool htp_mm_hmx_solve_2d_params(
    ಪೂರ್ಣಾಂಕ wtype,
    uint32_t k,
    uint32_t m_id_ಅಡ್ಡಸಾಲುಗಳು,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_ವೆಚ್ಚ,
    ಪೂರ್ಣಾಂಕ n_ಎಳೆಗಳು,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_ಗಾತ್ರ,
    size_t bias_ಗಾತ್ರ,
    size_t vtcm_budget,
    size_t * m_chunk_ಹೊರಹೋಗು,
    size_t * n_chunk_ಹೊರಹೋಗು,
    ಪೂರ್ಣಾಂಕ * act_ಎಳೆಗಳು_ಹೊರಹೋಗು,
    size_t * vtcm_ಗಾತ್ರ_ಹೊರಹೋಗು
) {
    size_t ಉತ್ತಮ_mblocks = SIZE_MAX;
    ಪೂರ್ಣಾಂಕ ಉತ್ತಮ_act_ಎಳೆಗಳು = 0;
    size_t ಉತ್ತಮ_m_chunk = 0;
    size_t ಉತ್ತಮ_n_chunk = 0;
    size_t ಉತ್ತಮ_vtcm_ಗಾತ್ರ = 0;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ m_for_chunks = is_matmul_id ? hex_align_ಮೇಲೆ(m_id_ಅಡ್ಡಸಾಲುಗಳು, 32) : ne11_padded;

    ಪೂರ್ಣಾಂಕ act_ಎಳೆಗಳು = n_ಎಳೆಗಳು;
    ಆಗ (act_ಎಳೆಗಳು >= 1) {
        size_t simple_2d_overhead = htp_mm_hmx_get_2d_overhead(pipeline, is_matmul_id) + (bias_ಗಾತ್ರ > 0 ? hex_align_ಮೇಲೆ(bias_ಗಾತ್ರ, HTP_MM_HMX_TILE_ಗಾತ್ರ) : 0);
        size_t simple_2d_ಗಾತ್ರ_ಪ್ರತಿ_n, simple_2d_ಗಾತ್ರ_ಪ್ರತಿ_m, simple_2d_ಗಾತ್ರ_ಪ್ರತಿ_mn;
        htp_mm_hmx_get_2d_chunk_costs(wtype, k, pipeline, aligned_tile_ಗಾತ್ರ, &simple_2d_ಗಾತ್ರ_ಪ್ರತಿ_n, &simple_2d_ಗಾತ್ರ_ಪ್ರತಿ_m, &simple_2d_ಗಾತ್ರ_ಪ್ರತಿ_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_ಗಾತ್ರ_candidate = 0;

        ಆದರೆ (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_ಗಾತ್ರ_ಪ್ರತಿ_n, simple_2d_ಗಾತ್ರ_ಪ್ರತಿ_m, simple_2d_ಗಾತ್ರ_ಪ್ರತಿ_mn, m_for_chunks, ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_ವೆಚ್ಚ_W_DEQUANT, (size_t) m_for_ವೆಚ್ಚ * HTP_MM_HMX_ವೆಚ್ಚ_A_CONVERT,
                               &m_chunk_candidate, &n_chunk_candidate, &vtcm_ಗಾತ್ರ_candidate) == 0) {
            size_t exact_ಗಾತ್ರ = htp_mm_hmx_get_2d_vtcm_ಗಾತ್ರ(wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_ಎಳೆಗಳು, aligned_tile_ಗಾತ್ರ, bias_ಗಾತ್ರ);
            ಆದರೆ (exact_ಗಾತ್ರ <= vtcm_budget) {
                size_t mblocks = ((size_t) m_for_ವೆಚ್ಚ + m_chunk_candidate - 1) / m_chunk_candidate;
                ಆದರೆ (mblocks < ಉತ್ತಮ_mblocks || (mblocks == ಉತ್ತಮ_mblocks && act_ಎಳೆಗಳು > ಉತ್ತಮ_act_ಎಳೆಗಳು)) {
                    ಉತ್ತಮ_mblocks = mblocks;
                    ಉತ್ತಮ_act_ಎಳೆಗಳು = act_ಎಳೆಗಳು;
                    ಉತ್ತಮ_m_chunk = m_chunk_candidate;
                    ಉತ್ತಮ_n_chunk = n_chunk_candidate;
                    ಉತ್ತಮ_vtcm_ಗಾತ್ರ = exact_ಗಾತ್ರ;
                }
            }
        }
        ಆದರೆ (act_ಎಳೆಗಳು == 1) {
            act_ಎಳೆಗಳು = 0;
        } ಇಲ್ಲದಿದ್ದರೆ {
            act_ಎಳೆಗಳು /= 2;
        }
    }

    ಆದರೆ (ಉತ್ತಮ_act_ಎಳೆಗಳು > 0) {
        *m_chunk_ಹೊರಹೋಗು = ಉತ್ತಮ_m_chunk;
        *n_chunk_ಹೊರಹೋಗು = ಉತ್ತಮ_n_chunk;
        *vtcm_ಗಾತ್ರ_ಹೊರಹೋಗು = ಉತ್ತಮ_vtcm_ಗಾತ್ರ;
        *act_ಎಳೆಗಳು_ಹೊರಹೋಗು = ಉತ್ತಮ_act_ಎಳೆಗಳು;
        ಹಿಂತಿರುಗಿ true;
    }
    ಹಿಂತಿರುಗಿ false;
}

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ __cplusplus
}
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ_ಅಂತ್ಯ // HTP_MATMUL_OPS_H
