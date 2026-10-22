#यदि_अपरिभाषितम् HTP_MATMUL_OPS_H
#परिभाषय HTP_MATMUL_OPS_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "htp-vtcm.h"

#यदि_परिभाषितम् __cplusplus
बाह्य "C" {
#यदि_अन्तः

// --- HMX Tile Constraints ---
#परिभाषय HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या 32
#परिभाषय HTP_MM_HMX_TILE_N_पङ्क्तिकाः 32
#परिभाषय HTP_MM_HMX_TILE_आकार   (32 * 32 * परिमाण(__fp16)) // 2048 bytes
#परिभाषय HTP_MM_HMX_TILE_N_ELMS 1024
#परिभाषय HTP_MM_HMX_MIN_NROWS   4

#परिभाषय HTP_MM_WEIGHT_TILE_K_Q2_0             64
#परिभाषय HTP_MM_WEIGHT_TILE_आकार_Q2_0          HTP_TENSOR_TILE_आकार_Q2_0
#परिभाषय HTP_MM_WEIGHT_ALIGNED_TILE_आकार_Q2_0  640

// --- Activation Tiled Block Sizes (including padding) ---
#परिभाषय HTP_MM_ACT_TILE_आकार_Q8_0      1152
#परिभाषय HTP_MM_ACT_TILE_आकार_Q8_1      1280
#परिभाषय HTP_MM_ACT_TILE_आकार_F16       2048

#परिभाषय HTP_MM_MAX_PREFETCH 16

// --- Solver Cost Model Penalty Weights (HMX-specific) ---
#परिभाषय HTP_MM_HMX_COST_W_DEQUANT 3 // cost penalty for quantized weight loading/dequantization
#परिभाषय HTP_MM_HMX_COST_A_CONVERT 2 // cost penalty for activation loading/conversion

// --- DMA Activation Transfer Configuration ---
#परिभाषय HTP_MM_DMA_ACT_पङ्क्तिकाः_प्रत्येकम्_चरणः 2
#परिभाषय HTP_MM_DMA_ACT_MULTIPLIER    (2 * HTP_MM_DMA_ACT_पङ्क्तिकाः_प्रत्येकम्_चरणः)

गणना htp_mm_kernel_type {
    HTP_MM_KERNEL_UNSUPPORTED = 0,

    // HMX paths
    HTP_MM_KERNEL_HMX_2D,
    HTP_MM_KERNEL_HMX_BATCHED,

    // HVX floating-point paths
    HTP_MM_KERNEL_HVX_F16_F16_VTCM,
    HTP_MM_KERNEL_HVX_F32_F32_VTCM,
    HTP_MM_KERNEL_HVX_BF16_F32_VTCM,

    // HVX quantized paths
    HTP_MM_KERNEL_HVX_QUANT_ROW,      // standard row-wise parallel quantization
    HTP_MM_KERNEL_HVX_QUANT_खण्डम्,    // parallel block-wise quantization
};

// Op-specific struct for precomputed matmul params
संरचना htp_mm_kernel_params {
    uint8_t  kernel_type;        // enum htp_mm_kernel_type
    uint8_t  pipeline;           // 1 = pipelined execution, 0 = standard
    uint8_t  collapse;           // 1 = collapse outer dims into 2D, 0 = standard
    uint8_t  n_hmx;              // 1 = use HMX, 0 = use HVX

    uint8_t  n_तन्तवः;          // Number of threads to spawn
    uint8_t  n_act_तन्तवः;      // Number of threads for activation preparation
    uint8_t  n_prefetch;         // Prefetch lookahead buffers/rows in VTCM
    uint8_t  n_weights;          // Number of weights for fused NX

    int32_t  m_chunk;            // Row chunk size (M chunk)
    int32_t  n_chunk;            // Col chunk size (N chunk)
    int32_t  tile_आकार;          // Weight tile size
    int32_t  aligned_tile_आकार;  // Aligned weight tile size (padded to 128)
    int32_t  act_row_आकार;       // Row size for activation scratchpad
    int32_t  vtcm_आकार;          // Total required scratchpad size in VTCM
    int32_t  vtcm_src0_आकार;     // src0 scratchpad size in VTCM
    int32_t  vtcm_act_आकार;      // activation scratchpad size in VTCM
    int32_t  vtcm_bias_आकार;     // bias scratchpad size in VTCM (fused only)
    int32_t  vtcm_dst_आकार;      // dst scratchpad size in VTCM

    // Precomputed division values
    संरचना fastdiv_मानानि div_ne12_ne1;
    संरचना fastdiv_मानानि div_ne1;
    संरचना fastdiv_मानानि div_r2;
    संरचना fastdiv_मानानि div_r3;
    संरचना fastdiv_मानानि div_ne12;
    संरचना fastdiv_मानानि div_n_act_तन्तवः;
    संरचना fastdiv_मानानि div_ne00_padded;
};

#यदि defined(__cplusplus)
static_assert(परिमाण(संरचना htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(परिमाण(संरचना htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#यदि_अन्तः

संरचना mmid_row_mapping {
    uint32_t i1;
    uint32_t i2;
};

// Search for optimal (mc, nc) chunk sizes within VTCM budget.
स्थिर अन्तःस्थ पूर्णाङ्क htp_mm_hmx_compute_chunks(size_t   vtcm_योग,
                              size_t   overhead,
                              size_t   प्रत्येकम्_n_cost,
                              size_t   प्रत्येकम्_m_cost,
                              size_t   प्रत्येकम्_mn_cost,
                              size_t   m,
                              size_t   n,
                              size_t   m_खण्डम्_cost,
                              size_t   n_खण्डम्_cost,
                              size_t * m_chunk_बहिर्गमनम्,
                              size_t * n_chunk_बहिर्गमनम्,
                              size_t * योग_बहिर्गमनम्) {
    यदि (m == 0 || n == 0) प्रत्यागम -1;
    यदि (vtcm_योग <= overhead) प्रत्यागम -1;
    यदि (प्रत्येकम्_n_cost == 0 || प्रत्येकम्_m_cost == 0 || प्रत्येकम्_mn_cost == 0) प्रत्यागम -1;

    ध्रुव size_t usable = vtcm_योग - overhead;

    size_t best_cost = SIZE_MAX;
    size_t best_पुच्छ_waste = SIZE_MAX;
    size_t best_mn   = 0;
    size_t best_m = 0, best_n = 0;

    ध्रुव size_t max_nc_budget = (usable / प्रत्येकम्_n_cost);
    ध्रुव size_t n_max = hex_align_अधस्तात्(hex_smin((size_t)n, max_nc_budget), HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या);
    प्रति (size_t nc = n_max; nc >= HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या; nc -= HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) {
        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        यदि (hex_mul_अतिप्रवाहः(nc, प्रत्येकम्_n_cost, &n_fixed)) अनुवर्त;
        यदि (n_fixed >= usable) गच्छ अग्र_nc;

        यदि (hex_mul_अतिप्रवाहः(nc, प्रत्येकम्_mn_cost, &ncmn)) गच्छ अग्र_nc;
        यदि (hex_सङ्कलय_अतिप्रवाहः(प्रत्येकम्_m_cost, ncmn, &mc_denom) || mc_denom == 0) गच्छ अग्र_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hex_align_अधस्तात्(mc, HTP_MM_HMX_TILE_N_पङ्क्तिकाः);
            mc = hex_smin(mc, m);

            यदि (mc == 0) {
                गच्छ अग्र_nc;
            }

            size_t mblocks = ((size_t) m + mc - 1) / mc;
            size_t nblocks = ((size_t) n + nc - 1) / nc;
            size_t cost    = mblocks * m_खण्डम्_cost + nblocks * n_खण्डम्_cost;
            size_t rem     = n % nc;
            size_t पुच्छ_waste = (rem == 0) ? 0 : (nc - rem);
            size_t mn      = mc * nc;
            यदि (cost < best_cost ||
                (cost == best_cost && पुच्छ_waste < best_पुच्छ_waste) ||
                (cost == best_cost && पुच्छ_waste == best_पुच्छ_waste && mn > best_mn)) {
                best_cost       = cost;
                best_पुच्छ_waste = पुच्छ_waste;
                best_mn         = mn;
                best_m          = mc;
                best_n          = nc;
            }
        }

अग्र_nc:
        यदि (nc == HTP_MM_HMX_TILE_N_स्तम्भसङ्ख्या) भङ्ग;  // avoid size_t underflow
    }

    यदि (best_m == 0 || best_n == 0) प्रत्यागम -1;

    // Compute exact total (with overflow checks)
    size_t t0 = 0, t1 = 0, t2 = 0, mn = 0, योग = 0;
    यदि (hex_mul_अतिप्रवाहः(best_n, प्रत्येकम्_n_cost, &t0)) प्रत्यागम -1;
    यदि (hex_mul_अतिप्रवाहः(best_m, प्रत्येकम्_m_cost, &t1)) प्रत्यागम -1;
    यदि (hex_mul_अतिप्रवाहः(best_m, best_n, &mn))     प्रत्यागम -1;
    यदि (hex_mul_अतिप्रवाहः(mn, प्रत्येकम्_mn_cost, &t2))    प्रत्यागम -1;
    यदि (hex_सङ्कलय_अतिप्रवाहः(t0, t1, &योग))          प्रत्यागम -1;
    यदि (hex_सङ्कलय_अतिप्रवाहः(योग, t2, &योग))       प्रत्यागम -1;
    यदि (hex_सङ्कलय_अतिप्रवाहः(योग, overhead, &योग)) प्रत्यागम -1;

    *m_chunk_बहिर्गमनम् = best_m;
    *n_chunk_बहिर्गमनम् = best_n;
    *योग_बहिर्गमनम्   = योग;
    प्रत्यागम 0;
}

// weight types whose tiles carry a per-block offset (x = d * q + m): the activations need block sums (q8_1)
// (Q2_K: per-16 k sums, q8_1_s16)
स्थिर अन्तःस्थ bool htp_mm_weight_has_विस्थापनम्(पूर्णाङ्क weight_type) {
    प्रत्यागम weight_type == HTP_TYPE_Q4_1 || weight_type == HTP_TYPE_Q4_K || weight_type == HTP_TYPE_Q5_K ||
           weight_type == HTP_TYPE_Q2_K || weight_type == HTP_TYPE_Q2_0;
}

// --- Activation/Row Size Helpers ---
स्थिर अन्तःस्थ size_t htp_mm_q8_0_tiled_row_आकार(uint32_t ne) {
    ध्रुव uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ध्रुव uint32_t nb_32 = ne_padded / 32;
    प्रत्यागम nb_32 * HTP_MM_ACT_TILE_आकार_Q8_0;
}

स्थिर अन्तःस्थ size_t htp_mm_q8_1_tiled_row_आकार(uint32_t ne) {
    ध्रुव uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ध्रुव uint32_t nb_32 = ne_padded / 32;
    प्रत्यागम nb_32 * HTP_MM_ACT_TILE_आकार_Q8_1;
}

स्थिर अन्तःस्थ size_t htp_mm_f16_tiled_row_आकार(uint32_t ne) {
    ध्रुव uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ध्रुव uint32_t nb_32 = ne_padded / 32;
    प्रत्यागम nb_32 * HTP_MM_ACT_TILE_आकार_F16;
}

स्थिर अन्तःस्थ size_t htp_mm_tiled_act_row_आकार(पूर्णाङ्क weight_type, uint32_t ne) {
    यदि (weight_type == HTP_TYPE_F16) {
        प्रत्यागम htp_mm_f16_tiled_row_आकार(ne);
    }
    प्रत्यागम htp_mm_weight_has_विस्थापनम्(weight_type) ? htp_mm_q8_1_tiled_row_आकार(ne) : htp_mm_q8_0_tiled_row_आकार(ne);
}

स्थिर अन्तःस्थ size_t htp_mm_आवर्तनम्_ऊर्ध्वम्(size_t n, size_t m) {
    प्रत्यागम ((n + m - 1) / m) * m;
}

स्थिर अन्तःस्थ bool htp_mm_hmx_pipeline(uint32_t m) {
    प्रत्यागम m > 32;
}

स्थिर अन्तःस्थ शून्य htp_mm_hmx_get_2d_chunk_costs(
    पूर्णाङ्क wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_आकार,
    size_t * आकार_प्रत्येकम्_n_बहिर्गमनम्, size_t * आकार_प्रत्येकम्_m_बहिर्गमनम्, size_t * आकार_प्रत्येकम्_mn_बहिर्गमनम्
) {
    ध्रुव bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
    ध्रुव size_t row_stride = htp_tensor_get_row_आकार(wtype, k);
    ध्रुव size_t vec_dot_आकार = k * परिमाण(uint16_t);
    ध्रुव uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(wtype);
    ध्रुव size_t qweight_row_stride = is_quant ? (size_t)(n_weight_tiles * aligned_tile_आकार) / 32 : 0;

    *आकार_प्रत्येकम्_n_बहिर्गमनम् = (pipeline ? 2 : 1) * (is_quant ? qweight_row_stride : row_stride) +
                      (pipeline ? 2 * vec_dot_आकार : vec_dot_आकार);
    *आकार_प्रत्येकम्_m_बहिर्गमनम् = vec_dot_आकार;
    *आकार_प्रत्येकम्_mn_बहिर्गमनम् = (pipeline ? 2 : 1) * परिमाण(uint16_t);
}

स्थिर अन्तःस्थ शून्य htp_mm_hmx_get_batched_chunk_costs(
    uint32_t k, uint32_t गणः_आकार,
    size_t * आकार_प्रत्येकम्_n_बहिर्गमनम्, size_t * आकार_प्रत्येकम्_m_बहिर्गमनम्, size_t * आकार_प्रत्येकम्_mn_बहिर्गमनम्
) {
    ध्रुव size_t vec_dot_आकार = k * परिमाण(uint16_t);
    *आकार_प्रत्येकम्_n_बहिर्गमनम् = 3 * vec_dot_आकार;
    *आकार_प्रत्येकम्_m_बहिर्गमनम् = गणः_आकार * vec_dot_आकार;
    *आकार_प्रत्येकम्_mn_बहिर्गमनम् = परिमाण(uint16_t);
}

स्थिर अन्तःस्थ size_t htp_mm_hmx_get_2d_overhead(bool pipeline, bool is_matmul_id) {
    size_t अङ्कः_regions = pipeline ? 7 : (is_matmul_id ? 4 : 5);
    प्रत्यागम अङ्कः_regions * HTP_MM_HMX_TILE_आकार + 256;
}

स्थिर अन्तःस्थ size_t htp_mm_hmx_get_batched_overhead(शून्य) {
    प्रत्यागम 5 * HTP_MM_HMX_TILE_आकार + 256;
}

संरचना htp_mm_hmx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_weight[2];     // [1] is only used when pipelined
    size_t off_act;
    size_t off_act_f32;       // fp32 activation conversion scratch
    size_t off_dst[2];        // [1] is only used when pipelined
    size_t off_scratch[2];    // dequantization scratch pads
    size_t off_scales;        // HMX scales (256 bytes)
    size_t off_bias;          // bias in VTCM

    // Cached sizes of regions for HMX kernel use
    size_t weight_area_अष्टकानि;
    size_t act_area_अष्टकानि;
    size_t act_f32_अष्टकानि;
    size_t output_area_अष्टकानि;
    size_t scratch_अष्टकानि[2];
    size_t act_शीर्ष_stride;
    size_t bias_अष्टकानि;

    size_t योग_अष्टकानि;
};

संरचना htp_mm_hvx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_act;           // vtcm_act (activation)
    size_t off_src0;          // vtcm_src0 (weight/Wk)
    size_t off_bias;          // vtcm_bias (bias / fused add only)
    size_t off_dst;           // vtcm_dst (output scratch)
    size_t off_act_raw;       // vtcm_act_raw (raw activation DMA staging)

    // Cached sizes
    size_t src0_अष्टकानि;
    size_t act_अष्टकानि;
    size_t bias_अष्टकानि;
    size_t dst_अष्टकानि;
    size_t act_raw_अष्टकानि;

    size_t योग_अष्टकानि;
};

स्थिर अन्तःस्थ शून्य htp_mm_hmx_vtcm_layout_build(
    संरचना htp_mm_hmx_vtcm_layout * L,
    पूर्णाङ्क kernel_type,
    पूर्णाङ्क wtype,
    uint32_t k,
    size_t mc,
    size_t nc,
    uint32_t गणः_आकार,
    bool pipeline,
    uint32_t act_तन्तवः,
    uint32_t aligned_tile_आकार,
    size_t bias_आकार
) {
    size_t off = 0;

    यदि (kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        ध्रुव size_t vec_dot_आकार     = k * परिमाण(uint16_t);
        ध्रुव size_t act_शीर्ष_stride   = mc * k;
        ध्रुव size_t weight_area_आकार  = hex_align_ऊर्ध्वम्(nc * vec_dot_आकार, HTP_MM_HMX_TILE_आकार);
        ध्रुव size_t activation_area_आकार = hex_align_ऊर्ध्वम्(गणः_आकार * act_शीर्ष_stride * परिमाण(uint16_t), HTP_MM_HMX_TILE_आकार);
        ध्रुव size_t output_area_आकार  = hex_align_ऊर्ध्वम्(गणः_आकार * mc * nc * परिमाण(uint16_t), HTP_MM_HMX_TILE_आकार);
        ध्रुव size_t scratch_area_आकार = hex_align_ऊर्ध्वम्(nc * vec_dot_आकार, HTP_MM_HMX_TILE_आकार);
        ध्रुव size_t min_f32_आकार = hex_align_ऊर्ध्वम्(act_तन्तवः * HTP_MM_DMA_ACT_MULTIPLIER * k * परिमाण(भिन्न), 128);

        // Group A: Permanent activation tiles and scales
        size_t off_गणः_a = 0;
        VTCM_LAYOUT_ALLOC(off_गणः_a, off_act, activation_area_आकार);
        VTCM_LAYOUT_ALLOC(off_गणः_a, off_scales, HTP_MM_HMX_TILE_आकार); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_गणः_a, off_bias, hex_align_ऊर्ध्वम्(bias_आकार, HTP_MM_HMX_TILE_आकार), bias_आकार > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_गणः_b = off_गणः_a;
        VTCM_LAYOUT_ALLOC(off_गणः_b, off_weight[0], weight_area_आकार);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_गणः_b, off_weight[1], weight_area_आकार, false);
        VTCM_LAYOUT_ALLOC(off_गणः_b, off_dst[0], output_area_आकार);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_गणः_b, off_dst[1], output_area_आकार, false);
        VTCM_LAYOUT_ALLOC(off_गणः_b, off_scratch[0], scratch_area_आकार);
        VTCM_LAYOUT_ALLOC(off_गणः_b, off_scratch[1], scratch_area_आकार);

        ध्रुव size_t गणः_b_आकार = off_गणः_b - off_गणः_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ध्रुव size_t max_f32_आकार = act_तन्तवः * 64 * k * परिमाण(भिन्न);
        ध्रुव size_t act_f32_आकार = hex_align_ऊर्ध्वम्(hex_smin(max_f32_आकार, hex_smax(min_f32_आकार, गणः_b_आकार)), 128);
        size_t off_गणः_c = off_गणः_a;
        VTCM_LAYOUT_ALLOC(off_गणः_c, off_act_f32, act_f32_आकार);

        ध्रुव size_t गणः_c_आकार = off_गणः_c - off_गणः_a;

        L->weight_area_अष्टकानि = weight_area_आकार;
        L->act_area_अष्टकानि    = activation_area_आकार;
        L->act_f32_अष्टकानि     = act_f32_आकार;
        L->output_area_अष्टकानि = output_area_आकार;
        L->scratch_अष्टकानि[0]  = scratch_area_आकार;
        L->scratch_अष्टकानि[1]  = scratch_area_आकार;
        L->act_शीर्ष_stride   = act_शीर्ष_stride;
        L->bias_अष्टकानि        = bias_आकार;

        off = off_गणः_a + hex_smax(गणः_b_आकार, गणः_c_आकार);
    } अन्यथा {
        // HTP_MM_KERNEL_HMX_2D
        ध्रुव bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
        ध्रुव size_t row_stride = htp_tensor_get_row_आकार(wtype, k);
        ध्रुव size_t vec_dot_आकार = k * परिमाण(uint16_t);
        ध्रुव uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(wtype);

        ध्रुव size_t min_f32_आकार = hex_align_ऊर्ध्वम्(act_तन्तवः * HTP_MM_DMA_ACT_MULTIPLIER * k * परिमाण(भिन्न), 128);
        ध्रुव size_t weight_area_आकार = is_quant
            ? hex_align_ऊर्ध्वम्((nc / 32) * n_weight_tiles * aligned_tile_आकार, HTP_MM_HMX_TILE_आकार)
            : hex_align_ऊर्ध्वम्(nc * row_stride, HTP_MM_HMX_TILE_आकार);
        ध्रुव size_t act_area_आकार    = hex_align_ऊर्ध्वम्(mc * vec_dot_आकार, HTP_MM_HMX_TILE_आकार);
        ध्रुव size_t output_area_आकार = hex_align_ऊर्ध्वम्(mc * nc * परिमाण(__fp16), HTP_MM_HMX_TILE_आकार);

        ध्रुव size_t scratch0_आकार = hex_align_ऊर्ध्वम्(nc * vec_dot_आकार, HTP_MM_HMX_TILE_आकार);
        ध्रुव size_t scratch1_आकार = pipeline ? scratch0_आकार : 0;

        // Group A:  Scales and activation tiles (must not overlap with Group B or C)
        size_t off_गणः_a = 0;
        VTCM_LAYOUT_ALLOC(off_गणः_a, off_scales, HTP_MM_HMX_TILE_आकार); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC(off_गणः_a, off_act, act_area_आकार);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_गणः_a, off_bias, hex_align_ऊर्ध्वम्(bias_आकार, HTP_MM_HMX_TILE_आकार), bias_आकार > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_गणः_b = off_गणः_a;
        VTCM_LAYOUT_ALLOC(off_गणः_b, off_weight[0], weight_area_आकार);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_गणः_b, off_weight[1], weight_area_आकार, pipeline);
        VTCM_LAYOUT_ALLOC(off_गणः_b, off_dst[0], output_area_आकार);
        VTCM_LAYOUT_ALLOC(off_गणः_b, off_scratch[0], scratch0_आकार);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_गणः_b, off_scratch[1], scratch0_आकार, pipeline);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_गणः_b, off_dst[1], output_area_आकार, pipeline);

        ध्रुव size_t गणः_b_आकार = off_गणः_b - off_गणः_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ध्रुव size_t max_f32_आकार = act_तन्तवः * 64 * k * परिमाण(भिन्न);
        ध्रुव size_t act_f32_आकार = hex_align_ऊर्ध्वम्(hex_smin(max_f32_आकार, hex_smax(min_f32_आकार, गणः_b_आकार)), 128);
        size_t off_गणः_c = off_गणः_a;
        VTCM_LAYOUT_ALLOC(off_गणः_c, off_act_f32, act_f32_आकार);

        ध्रुव size_t गणः_c_आकार = off_गणः_c - off_गणः_a;

        L->weight_area_अष्टकानि = weight_area_आकार;
        L->act_area_अष्टकानि    = act_area_आकार;
        L->act_f32_अष्टकानि     = act_f32_आकार;
        L->output_area_अष्टकानि = output_area_आकार;
        L->scratch_अष्टकानि[0]  = scratch0_आकार;
        L->scratch_अष्टकानि[1]  = scratch1_आकार;
        L->act_शीर्ष_stride   = 0;
        L->bias_अष्टकानि        = bias_आकार;

        off = off_गणः_a + hex_smax(गणः_b_आकार, गणः_c_आकार);
    }

    L->योग_अष्टकानि = off;
}

स्थिर अन्तःस्थ शून्य htp_mm_hvx_vtcm_layout_build(
    संरचना htp_mm_hvx_vtcm_layout * L,
    पूर्णाङ्क kernel_type,
    पूर्णाङ्क wtype,
    uint32_t ne10,       // k
    uint32_t act_nrows,  // m_total
    uint32_t n_तन्तवः,
    size_t dst_row_आकार,
    size_t src0_row_आकार,
    size_t act_row_आकार,
    size_t bias_row_आकार,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_nx
) {
    (शून्य)act_row_आकार;
    size_t src0_sz    = 0;
    size_t act_sz     = 0;
    size_t bias_sz    = bias_row_आकार > 0 ? htp_mm_आवर्तनम्_ऊर्ध्वम्(bias_row_आकार, 128) : 0;
    size_t dst_sz     = 0;
    size_t act_raw_sz = 0;

    यदि (is_fused_nx) {
        uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(wtype);
        uint32_t n_k_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        uint32_t tile_row_आकार = n_k_tiles * aligned_tile_आकार;
        size_t weight_sz_प्रत्येकम्_thread = hex_आवर्तनम्_ऊर्ध्वम्(n_prefetch * tile_row_आकार, 128);

        size_t tiled_act_row_आकार = htp_mm_tiled_act_row_आकार(wtype, ne10);
        size_t q_act_sz = hex_आवर्तनम्_ऊर्ध्वम्(tiled_act_row_आकार * act_nrows, 128);
        size_t raw_row_आकार = hex_आवर्तनम्_ऊर्ध्वम्(ne10 * परिमाण(भिन्न), QK_Q8_0_TILED * परिमाण(भिन्न));

        src0_sz    = weight_sz_प्रत्येकम्_thread * n_तन्तवः; // shared single-weight prefetch buffer
        act_sz     = q_act_sz;                         // quantized activation buffer
        bias_sz    = 0;
        dst_sz     = 0;
        act_raw_sz = hex_आवर्तनम्_ऊर्ध्वम्(raw_row_आकार * act_nrows, 128);
    } अन्यथा यदि (is_matmul_id) {
        ध्रुव size_t act_row_आकार_tiled = htp_mm_tiled_act_row_आकार(wtype, ne10);
        ध्रुव uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(wtype);
        ध्रुव uint32_t n_k_tiles         = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        ध्रुव uint32_t tile_row_आकार     = n_k_tiles * aligned_tile_आकार;
        size_t src0_sz_प्रत्येकम्_thread        = htp_mm_आवर्तनम्_ऊर्ध्वम्(n_prefetch * tile_row_आकार, 256);

        act_sz = htp_mm_आवर्तनम्_ऊर्ध्वम्(act_row_आकार_tiled * act_nrows, 256);

        size_t raw_row_आकार = hex_आवर्तनम्_ऊर्ध्वम्(ne10 * परिमाण(भिन्न), QK_Q8_0_TILED * परिमाण(भिन्न));

        src0_sz    = src0_sz_प्रत्येकम्_thread * n_तन्तवः;
        dst_sz     = 0;
        bias_sz    = 0;
        act_raw_sz = hex_आवर्तनम्_ऊर्ध्वम्(raw_row_आकार * act_nrows, 128);
    } अन्यथा {
        ध्रुव size_t src0_row_आकार_padded = htp_mm_आवर्तनम्_ऊर्ध्वम्(src0_row_आकार, 128);
        ध्रुव size_t dst_nrows = (act_nrows > 1) ? 0 : 1;

        विकल्प (kernel_type) {
            पक्ष HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                size_t f16_act_row_आकार = htp_mm_आवर्तनम्_ऊर्ध्वम्(ne10 * 2, 128);
                act_sz     = htp_mm_आवर्तनम्_ऊर्ध्वम्(f16_act_row_आकार * act_nrows, 256);
                src0_sz    = htp_mm_आवर्तनम्_ऊर्ध्वम्(n_prefetch * src0_row_आकार_padded, 256) * n_तन्तवः;
                dst_sz     = dst_nrows > 0 ? htp_mm_आवर्तनम्_ऊर्ध्वम्(dst_row_आकार, 128) * n_तन्तवः : 0;
                act_raw_sz = hex_आवर्तनम्_ऊर्ध्वम्(hex_आवर्तनम्_ऊर्ध्वम्(ne10 * परिमाण(भिन्न), 128) * act_nrows, 128);
                भङ्ग;
            }
            पक्ष HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            पक्ष HTP_MM_KERNEL_HVX_BF16_F32_VTCM: {
                size_t f32_act_row_आकार = htp_mm_आवर्तनम्_ऊर्ध्वम्(ne10 * 4, 128);
                act_sz     = htp_mm_आवर्तनम्_ऊर्ध्वम्(f32_act_row_आकार * act_nrows, 256);
                src0_sz    = htp_mm_आवर्तनम्_ऊर्ध्वम्(n_prefetch * src0_row_आकार_padded, 256) * n_तन्तवः;
                dst_sz     = dst_nrows > 0 ? htp_mm_आवर्तनम्_ऊर्ध्वम्(dst_row_आकार, 128) * n_तन्तवः : 0;
                act_raw_sz = 0;
                भङ्ग;
            }
            पक्ष HTP_MM_KERNEL_HVX_QUANT_खण्डम्:
            पक्ष HTP_MM_KERNEL_HVX_QUANT_ROW: {
                size_t q_act_row_आकार = htp_mm_tiled_act_row_आकार(wtype, ne10);
                uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(wtype);
                uint32_t n_weight_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
                uint32_t tile_row_आकार = n_weight_tiles * aligned_tile_आकार;
                size_t tiled_vtcm_आकार = htp_mm_आवर्तनम्_ऊर्ध्वम्(n_prefetch * tile_row_आकार, 256);

                src0_sz = tiled_vtcm_आकार * n_तन्तवः;
                act_sz  = htp_mm_आवर्तनम्_ऊर्ध्वम्(q_act_row_आकार * act_nrows, 256);

                size_t dst_खण्डिका_प्रत्येकम्_thread = (dst_nrows > 0 && act_nrows == 1) ? htp_mm_आवर्तनम्_ऊर्ध्वम्((dst_row_आकार + n_तन्तवः - 1) / n_तन्तवः, 128) : 0;
                dst_sz = dst_खण्डिका_प्रत्येकम्_thread * n_तन्तवः;
                size_t raw_row_आकार = hex_आवर्तनम्_ऊर्ध्वम्(ne10 * परिमाण(भिन्न), QK_Q8_0_TILED * परिमाण(भिन्न));
                act_raw_sz = hex_आवर्तनम्_ऊर्ध्वम्(raw_row_आकार * act_nrows, 128);
                भङ्ग;
            }
            स्वभाव:
                भङ्ग;
        }
    }

    // Group A: Persistent buffers across chunk compute
    size_t off_गणः_a = 0;
    VTCM_LAYOUT_ALLOC(off_गणः_a, off_act, act_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_गणः_a, off_bias, bias_sz, bias_sz > 0);

    // Group B: Compute-only buffers (starts at off_group_a)
    size_t off_गणः_b = off_गणः_a;
    VTCM_LAYOUT_ALLOC(off_गणः_b, off_src0, src0_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_गणः_b, off_dst, dst_sz, dst_sz > 0);
    ध्रुव size_t गणः_b_आकार = off_गणः_b - off_गणः_a;

    // Group C: Raw activation staging buffer (overlaps Group B, starts at off_group_a)
    size_t off_गणः_c = off_गणः_a;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_गणः_c, off_act_raw, act_raw_sz, act_raw_sz > 0);
    ध्रुव size_t गणः_c_आकार = off_गणः_c - off_गणः_a;

    L->src0_अष्टकानि    = src0_sz;
    L->act_अष्टकानि     = act_sz;
    L->bias_अष्टकानि    = bias_sz;
    L->dst_अष्टकानि     = dst_sz;
    L->act_raw_अष्टकानि = act_raw_sz;
    L->योग_अष्टकानि   = off_गणः_a + hex_smax(गणः_b_आकार, गणः_c_आकार);
}

स्थिर अन्तःस्थ bool htp_mm_hvx_solve_vtcm_params(
    पूर्णाङ्क kernel_type,
    पूर्णाङ्क wtype,
    uint32_t ne10,
    uint32_t act_nrows,
    uint32_t n_तन्तवः,
    size_t dst_row_आकार,
    size_t src0_row_आकार,
    size_t act_row_आकार,
    size_t bias_row_आकार,
    uint32_t n_prefetch,
    size_t vtcm_budget,
    संरचना htp_mm_hvx_vtcm_layout * L_बहिर्गमनम्,
    uint32_t * m_chunk_बहिर्गमनम्
) {
    संरचना htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, act_nrows, n_तन्तवः,
        dst_row_आकार, src0_row_आकार, act_row_आकार, bias_row_आकार, n_prefetch, false, false
    );

    यदि (L.योग_अष्टकानि <= vtcm_budget) {
        *L_बहिर्गमनम् = L;
        *m_chunk_बहिर्गमनम् = act_nrows;
        प्रत्यागम true;
    }

    ध्रुव size_t fixed_अष्टकानि = L.src0_अष्टकानि + L.bias_अष्टकानि + L.dst_अष्टकानि;
    यदि (vtcm_budget <= fixed_अष्टकानि) {
        प्रत्यागम false;
    }

    ध्रुव size_t avail_act = vtcm_budget - fixed_अष्टकानि;
    size_t row_आकार = 0;
    यदि (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kernel_type == HTP_MM_KERNEL_HVX_QUANT_खण्डम्) {
        row_आकार = htp_mm_tiled_act_row_आकार(wtype, ne10);
    } अन्यथा यदि (kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        row_आकार = hex_आवर्तनम्_ऊर्ध्वम्(ne10 * 2, 128);
    } अन्यथा {
        row_आकार = hex_आवर्तनम्_ऊर्ध्वम्(ne10 * 4, 128);
    }
    यदि (row_आकार == 0) {
        प्रत्यागम false;
    }

    size_t eff_row_आकार = row_आकार;
    यदि (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kernel_type == HTP_MM_KERNEL_HVX_QUANT_खण्डम्) {
        eff_row_आकार += hex_आवर्तनम्_ऊर्ध्वम्(ne10 * परिमाण(भिन्न), QK_Q8_0_TILED * परिमाण(भिन्न));
    }

    uint32_t m_chunk = (uint32_t) (avail_act / eff_row_आकार);
    यदि (m_chunk > 1) {
        m_chunk &= ~1U;
    }
    यदि (m_chunk > act_nrows) {
        m_chunk = act_nrows;
    }
    यदि (m_chunk < 1) {
        प्रत्यागम false;
    }

    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, m_chunk, n_तन्तवः,
        dst_row_आकार, src0_row_आकार, act_row_आकार, bias_row_आकार, n_prefetch, false, false
    );

    यावत् (m_chunk > 2 && L.योग_अष्टकानि > vtcm_budget) {
        m_chunk -= 2;
        htp_mm_hvx_vtcm_layout_build(
            &L, kernel_type, wtype, ne10, m_chunk, n_तन्तवः,
            dst_row_आकार, src0_row_आकार, act_row_आकार, bias_row_आकार, n_prefetch, false, false
        );
    }

    यदि (L.योग_अष्टकानि <= vtcm_budget) {
        *L_बहिर्गमनम् = L;
        *m_chunk_बहिर्गमनम् = m_chunk;
        प्रत्यागम true;
    }

    प्रत्यागम false;
}

स्थिर अन्तःस्थ size_t htp_mm_hmx_get_2d_vtcm_आकार(
    पूर्णाङ्क wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_तन्तवः, uint32_t aligned_tile_आकार, size_t bias_आकार
) {
    संरचना htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_तन्तवः, aligned_tile_आकार, bias_आकार);
    प्रत्यागम L.योग_अष्टकानि;
}

स्थिर अन्तःस्थ size_t htp_mm_hmx_get_batched_vtcm_आकार(
    पूर्णाङ्क wtype, uint32_t k, size_t mc, size_t nc, uint32_t गणः_आकार, bool pipeline, uint32_t act_तन्तवः, size_t bias_आकार) {
    (शून्य)pipeline;
    संरचना htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, wtype, k, mc, nc, गणः_आकार, false, act_तन्तवः, 0, bias_आकार);
    प्रत्यागम L.योग_अष्टकानि;
}

स्थिर अन्तःस्थ bool htp_mm_hmx_solve_batched_params(
    पूर्णाङ्क wtype,
    uint32_t k,
    uint32_t ne01_padded,
    uint32_t ne11,
    uint32_t गणः_आकार,
    पूर्णाङ्क n_तन्तवः,
    bool pipeline,
    size_t bias_आकार,
    size_t vtcm_budget,
    size_t * m_chunk_बहिर्गमनम्,
    size_t * n_chunk_बहिर्गमनम्,
    पूर्णाङ्क * act_तन्तवः_बहिर्गमनम्,
    size_t * vtcm_आकार_बहिर्गमनम्
) {
    size_t best_mblocks = SIZE_MAX;
    पूर्णाङ्क best_act_तन्तवः = 0;
    size_t best_m_chunk = 0;
    size_t best_n_chunk = 0;
    size_t best_vtcm_आकार = 0;

    पूर्णाङ्क act_तन्तवः = n_तन्तवः;
    यावत् (act_तन्तवः >= 1) {
        size_t गणः_overhead = htp_mm_hmx_get_batched_overhead() + (bias_आकार > 0 ? hex_align_ऊर्ध्वम्(bias_आकार, HTP_MM_HMX_TILE_आकार) : 0);
        size_t गणः_आकार_प्रत्येकम्_n, गणः_आकार_प्रत्येकम्_m, गणः_आकार_प्रत्येकम्_mn;
        htp_mm_hmx_get_batched_chunk_costs(k, गणः_आकार, &गणः_आकार_प्रत्येकम्_n, &गणः_आकार_प्रत्येकम्_m, &गणः_आकार_प्रत्येकम्_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_आकार_candidate = 0;

        यदि (htp_mm_hmx_compute_chunks(vtcm_budget, गणः_overhead, गणः_आकार_प्रत्येकम्_n, गणः_आकार_प्रत्येकम्_m, गणः_आकार_प्रत्येकम्_mn, hex_align_ऊर्ध्वम्(ne11, 32), ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_COST_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_आकार_candidate) == 0) {
            size_t exact_आकार = htp_mm_hmx_get_batched_vtcm_आकार(wtype, k, m_chunk_candidate, n_chunk_candidate, गणः_आकार, pipeline, act_तन्तवः, bias_आकार);
            यदि (exact_आकार <= vtcm_budget) {
                size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                यदि (mblocks < best_mblocks || (mblocks == best_mblocks && act_तन्तवः > best_act_तन्तवः)) {
                    best_mblocks = mblocks;
                    best_act_तन्तवः = act_तन्तवः;
                    best_m_chunk = m_chunk_candidate;
                    best_n_chunk = n_chunk_candidate;
                    best_vtcm_आकार = exact_आकार;
                }
            }
        }
        यदि (act_तन्तवः == 1) {
            act_तन्तवः = 0;
        } अन्यथा {
            act_तन्तवः /= 2;
        }
    }

    यदि (best_act_तन्तवः > 0) {
        *m_chunk_बहिर्गमनम् = best_m_chunk;
        *n_chunk_बहिर्गमनम् = best_n_chunk;
        *vtcm_आकार_बहिर्गमनम् = best_vtcm_आकार;
        *act_तन्तवः_बहिर्गमनम् = best_act_तन्तवः;
        प्रत्यागम true;
    }
    प्रत्यागम false;
}

स्थिर अन्तःस्थ bool htp_mm_hmx_solve_2d_params(
    पूर्णाङ्क wtype,
    uint32_t k,
    uint32_t m_id_पङ्क्तिकाः,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_cost,
    पूर्णाङ्क n_तन्तवः,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_आकार,
    size_t bias_आकार,
    size_t vtcm_budget,
    size_t * m_chunk_बहिर्गमनम्,
    size_t * n_chunk_बहिर्गमनम्,
    पूर्णाङ्क * act_तन्तवः_बहिर्गमनम्,
    size_t * vtcm_आकार_बहिर्गमनम्
) {
    size_t best_mblocks = SIZE_MAX;
    पूर्णाङ्क best_act_तन्तवः = 0;
    size_t best_m_chunk = 0;
    size_t best_n_chunk = 0;
    size_t best_vtcm_आकार = 0;

    ध्रुव पूर्णाङ्क m_for_chunks = is_matmul_id ? hex_align_ऊर्ध्वम्(m_id_पङ्क्तिकाः, 32) : ne11_padded;

    पूर्णाङ्क act_तन्तवः = n_तन्तवः;
    यावत् (act_तन्तवः >= 1) {
        size_t simple_2d_overhead = htp_mm_hmx_get_2d_overhead(pipeline, is_matmul_id) + (bias_आकार > 0 ? hex_align_ऊर्ध्वम्(bias_आकार, HTP_MM_HMX_TILE_आकार) : 0);
        size_t simple_2d_आकार_प्रत्येकम्_n, simple_2d_आकार_प्रत्येकम्_m, simple_2d_आकार_प्रत्येकम्_mn;
        htp_mm_hmx_get_2d_chunk_costs(wtype, k, pipeline, aligned_tile_आकार, &simple_2d_आकार_प्रत्येकम्_n, &simple_2d_आकार_प्रत्येकम्_m, &simple_2d_आकार_प्रत्येकम्_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_आकार_candidate = 0;

        यदि (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_आकार_प्रत्येकम्_n, simple_2d_आकार_प्रत्येकम्_m, simple_2d_आकार_प्रत्येकम्_mn, m_for_chunks, ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (size_t) m_for_cost * HTP_MM_HMX_COST_A_CONVERT,
                               &m_chunk_candidate, &n_chunk_candidate, &vtcm_आकार_candidate) == 0) {
            size_t exact_आकार = htp_mm_hmx_get_2d_vtcm_आकार(wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_तन्तवः, aligned_tile_आकार, bias_आकार);
            यदि (exact_आकार <= vtcm_budget) {
                size_t mblocks = ((size_t) m_for_cost + m_chunk_candidate - 1) / m_chunk_candidate;
                यदि (mblocks < best_mblocks || (mblocks == best_mblocks && act_तन्तवः > best_act_तन्तवः)) {
                    best_mblocks = mblocks;
                    best_act_तन्तवः = act_तन्तवः;
                    best_m_chunk = m_chunk_candidate;
                    best_n_chunk = n_chunk_candidate;
                    best_vtcm_आकार = exact_आकार;
                }
            }
        }
        यदि (act_तन्तवः == 1) {
            act_तन्तवः = 0;
        } अन्यथा {
            act_तन्तवः /= 2;
        }
    }

    यदि (best_act_तन्तवः > 0) {
        *m_chunk_बहिर्गमनम् = best_m_chunk;
        *n_chunk_बहिर्गमनम् = best_n_chunk;
        *vtcm_आकार_बहिर्गमनम् = best_vtcm_आकार;
        *act_तन्तवः_बहिर्गमनम् = best_act_तन्तवः;
        प्रत्यागम true;
    }
    प्रत्यागम false;
}

#यदि_परिभाषितम् __cplusplus
}
#यदि_अन्तः

#यदि_अन्तः // HTP_MATMUL_OPS_H
