#నిర్వచితం_కాకపోతే HTP_MATMUL_OPS_H
#నిర్వచించు HTP_MATMUL_OPS_H

#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "hex-fastdiv.h"
#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "htp-vtcm.h"

#నిర్వచితమైతే __cplusplus
బాహ్య "C" {
#అయితే_ముగింపు

// --- HMX Tile Constraints ---
#నిర్వచించు HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య 32
#నిర్వచించు HTP_MM_HMX_TILE_N_అడ్డువరుసలు 32
#నిర్వచించు HTP_MM_HMX_TILE_కొలత   (32 * 32 * పరిమాణం(__fp16)) // 2048 bytes
#నిర్వచించు HTP_MM_HMX_TILE_N_ELMS 1024
#నిర్వచించు HTP_MM_HMX_MIN_NROWS   4

#నిర్వచించు HTP_MM_బరువు_TILE_K_Q2_0             64
#నిర్వచించు HTP_MM_బరువు_TILE_కొలత_Q2_0          HTP_TENSOR_TILE_కొలత_Q2_0
#నిర్వచించు HTP_MM_బరువు_ALIGNED_TILE_కొలత_Q2_0  640

// --- Activation Tiled Block Sizes (including padding) ---
#నిర్వచించు HTP_MM_ACT_TILE_కొలత_Q8_0      1152
#నిర్వచించు HTP_MM_ACT_TILE_కొలత_Q8_1      1280
#నిర్వచించు HTP_MM_ACT_TILE_కొలత_F16       2048

#నిర్వచించు HTP_MM_MAX_PREFETCH 16

// --- Solver Cost Model Penalty Weights (HMX-specific) ---
#నిర్వచించు HTP_MM_HMX_ఖర్చు_W_DEQUANT 3 // cost penalty for quantized weight loading/dequantization
#నిర్వచించు HTP_MM_HMX_ఖర్చు_A_CONVERT 2 // cost penalty for activation loading/conversion

// --- DMA Activation Transfer Configuration ---
#నిర్వచించు HTP_MM_DMA_ACT_అడ్డువరుసలు_చొప్పున_అడుగు 2
#నిర్వచించు HTP_MM_DMA_ACT_MULTIPLIER    (2 * HTP_MM_DMA_ACT_అడ్డువరుసలు_చొప్పున_అడుగు)

గణన htp_mm_kernel_type {
    HTP_MM_KERNEL_UNSUPPORTED = 0,

    // HMX paths
    HTP_MM_KERNEL_HMX_2D,
    HTP_MM_KERNEL_HMX_BATCHED,

    // HVX floating-point paths
    HTP_MM_KERNEL_HVX_F16_F16_VTCM,
    HTP_MM_KERNEL_HVX_F32_F32_VTCM,
    HTP_MM_KERNEL_HVX_BF16_F32_VTCM,

    // HVX quantized paths
    HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస,      // standard row-wise parallel quantization
    HTP_MM_KERNEL_HVX_QUANT_ఖండం,    // parallel block-wise quantization
};

// Op-specific struct for precomputed matmul params
నిర్మాణం htp_mm_kernel_params {
    uint8_t  kernel_type;        // enum htp_mm_kernel_type
    uint8_t  pipeline;           // 1 = pipelined execution, 0 = standard
    uint8_t  collapse;           // 1 = collapse outer dims into 2D, 0 = standard
    uint8_t  n_hmx;              // 1 = use HMX, 0 = use HVX

    uint8_t  n_దారాలు;          // Number of threads to spawn
    uint8_t  n_act_దారాలు;      // Number of threads for activation preparation
    uint8_t  n_prefetch;         // Prefetch lookahead buffers/rows in VTCM
    uint8_t  n_weights;          // Number of weights for fused NX

    int32_t  m_chunk;            // Row chunk size (M chunk)
    int32_t  n_chunk;            // Col chunk size (N chunk)
    int32_t  tile_కొలత;          // Weight tile size
    int32_t  aligned_tile_కొలత;  // Aligned weight tile size (padded to 128)
    int32_t  act_అడ్డువరుస_కొలత;       // Row size for activation scratchpad
    int32_t  vtcm_కొలత;          // Total required scratchpad size in VTCM
    int32_t  vtcm_src0_కొలత;     // src0 scratchpad size in VTCM
    int32_t  vtcm_act_కొలత;      // activation scratchpad size in VTCM
    int32_t  vtcm_bias_కొలత;     // bias scratchpad size in VTCM (fused only)
    int32_t  vtcm_dst_కొలత;      // dst scratchpad size in VTCM

    // Precomputed division values
    నిర్మాణం fastdiv_విలువలు div_ne12_ne1;
    నిర్మాణం fastdiv_విలువలు div_ne1;
    నిర్మాణం fastdiv_విలువలు div_r2;
    నిర్మాణం fastdiv_విలువలు div_r3;
    నిర్మాణం fastdiv_విలువలు div_ne12;
    నిర్మాణం fastdiv_విలువలు div_n_act_దారాలు;
    నిర్మాణం fastdiv_విలువలు div_ne00_padded;
};

#ఒకవేళ defined(__cplusplus)
static_assert(పరిమాణం(నిర్మాణం htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#లేకపోతే
_Static_assert(పరిమాణం(నిర్మాణం htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#అయితే_ముగింపు

నిర్మాణం mmid_అడ్డువరుస_mapping {
    uint32_t i1;
    uint32_t i2;
};

// Search for optimal (mc, nc) chunk sizes within VTCM budget.
స్థిర అంతఃవరుస పూర్ణాంకం htp_mm_hmx_compute_chunks(size_t   vtcm_మొత్తం,
                              size_t   overhead,
                              size_t   చొప్పున_n_ఖర్చు,
                              size_t   చొప్పున_m_ఖర్చు,
                              size_t   చొప్పున_mn_ఖర్చు,
                              size_t   m,
                              size_t   n,
                              size_t   m_ఖండం_ఖర్చు,
                              size_t   n_ఖండం_ఖర్చు,
                              size_t * m_chunk_బహిర్గతం,
                              size_t * n_chunk_బహిర్గతం,
                              size_t * మొత్తం_బహిర్గతం) {
    ఒకవేళ (m == 0 || n == 0) తిరిగి -1;
    ఒకవేళ (vtcm_మొత్తం <= overhead) తిరిగి -1;
    ఒకవేళ (చొప్పున_n_ఖర్చు == 0 || చొప్పున_m_ఖర్చు == 0 || చొప్పున_mn_ఖర్చు == 0) తిరిగి -1;

    స్థిరరాశి size_t usable = vtcm_మొత్తం - overhead;

    size_t ఉత్తమ_ఖర్చు = SIZE_MAX;
    size_t ఉత్తమ_తోక_waste = SIZE_MAX;
    size_t ఉత్తమ_mn   = 0;
    size_t ఉత్తమ_m = 0, ఉత్తమ_n = 0;

    స్థిరరాశి size_t max_nc_budget = (usable / చొప్పున_n_ఖర్చు);
    స్థిరరాశి size_t n_max = hex_align_కిందికి(hex_smin((size_t)n, max_nc_budget), HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య);
    కొరకు (size_t nc = n_max; nc >= HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య; nc -= HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) {
        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        ఒకవేళ (hex_mul_అధికప్రవాహం(nc, చొప్పున_n_ఖర్చు, &n_fixed)) కొనసాగు;
        ఒకవేళ (n_fixed >= usable) వెళ్ళు తదుపరి_nc;

        ఒకవేళ (hex_mul_అధికప్రవాహం(nc, చొప్పున_mn_ఖర్చు, &ncmn)) వెళ్ళు తదుపరి_nc;
        ఒకవేళ (hex_కూడు_అధికప్రవాహం(చొప్పున_m_ఖర్చు, ncmn, &mc_denom) || mc_denom == 0) వెళ్ళు తదుపరి_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hex_align_కిందికి(mc, HTP_MM_HMX_TILE_N_అడ్డువరుసలు);
            mc = hex_smin(mc, m);

            ఒకవేళ (mc == 0) {
                వెళ్ళు తదుపరి_nc;
            }

            size_t mblocks = ((size_t) m + mc - 1) / mc;
            size_t nblocks = ((size_t) n + nc - 1) / nc;
            size_t ఖర్చు    = mblocks * m_ఖండం_ఖర్చు + nblocks * n_ఖండం_ఖర్చు;
            size_t rem     = n % nc;
            size_t తోక_waste = (rem == 0) ? 0 : (nc - rem);
            size_t mn      = mc * nc;
            ఒకవేళ (ఖర్చు < ఉత్తమ_ఖర్చు ||
                (ఖర్చు == ఉత్తమ_ఖర్చు && తోక_waste < ఉత్తమ_తోక_waste) ||
                (ఖర్చు == ఉత్తమ_ఖర్చు && తోక_waste == ఉత్తమ_తోక_waste && mn > ఉత్తమ_mn)) {
                ఉత్తమ_ఖర్చు       = ఖర్చు;
                ఉత్తమ_తోక_waste = తోక_waste;
                ఉత్తమ_mn         = mn;
                ఉత్తమ_m          = mc;
                ఉత్తమ_n          = nc;
            }
        }

తదుపరి_nc:
        ఒకవేళ (nc == HTP_MM_HMX_TILE_N_నిలువుల_సంఖ్య) ఆపు;  // avoid size_t underflow
    }

    ఒకవేళ (ఉత్తమ_m == 0 || ఉత్తమ_n == 0) తిరిగి -1;

    // Compute exact total (with overflow checks)
    size_t t0 = 0, t1 = 0, t2 = 0, mn = 0, మొత్తం = 0;
    ఒకవేళ (hex_mul_అధికప్రవాహం(ఉత్తమ_n, చొప్పున_n_ఖర్చు, &t0)) తిరిగి -1;
    ఒకవేళ (hex_mul_అధికప్రవాహం(ఉత్తమ_m, చొప్పున_m_ఖర్చు, &t1)) తిరిగి -1;
    ఒకవేళ (hex_mul_అధికప్రవాహం(ఉత్తమ_m, ఉత్తమ_n, &mn))     తిరిగి -1;
    ఒకవేళ (hex_mul_అధికప్రవాహం(mn, చొప్పున_mn_ఖర్చు, &t2))    తిరిగి -1;
    ఒకవేళ (hex_కూడు_అధికప్రవాహం(t0, t1, &మొత్తం))          తిరిగి -1;
    ఒకవేళ (hex_కూడు_అధికప్రవాహం(మొత్తం, t2, &మొత్తం))       తిరిగి -1;
    ఒకవేళ (hex_కూడు_అధికప్రవాహం(మొత్తం, overhead, &మొత్తం)) తిరిగి -1;

    *m_chunk_బహిర్గతం = ఉత్తమ_m;
    *n_chunk_బహిర్గతం = ఉత్తమ_n;
    *మొత్తం_బహిర్గతం   = మొత్తం;
    తిరిగి 0;
}

// weight types whose tiles carry a per-block offset (x = d * q + m): the activations need block sums (q8_1)
// (Q2_K: per-16 k sums, q8_1_s16)
స్థిర అంతఃవరుస bool htp_mm_బరువు_has_స్థానభ్రంశం(పూర్ణాంకం బరువు_type) {
    తిరిగి బరువు_type == HTP_TYPE_Q4_1 || బరువు_type == HTP_TYPE_Q4_K || బరువు_type == HTP_TYPE_Q5_K ||
           బరువు_type == HTP_TYPE_Q2_K || బరువు_type == HTP_TYPE_Q2_0;
}

// --- Activation/Row Size Helpers ---
స్థిర అంతఃవరుస size_t htp_mm_q8_0_tiled_అడ్డువరుస_కొలత(uint32_t ne) {
    స్థిరరాశి uint32_t ne_padded = ((ne + 127) / 128) * 128;
    స్థిరరాశి uint32_t nb_32 = ne_padded / 32;
    తిరిగి nb_32 * HTP_MM_ACT_TILE_కొలత_Q8_0;
}

స్థిర అంతఃవరుస size_t htp_mm_q8_1_tiled_అడ్డువరుస_కొలత(uint32_t ne) {
    స్థిరరాశి uint32_t ne_padded = ((ne + 127) / 128) * 128;
    స్థిరరాశి uint32_t nb_32 = ne_padded / 32;
    తిరిగి nb_32 * HTP_MM_ACT_TILE_కొలత_Q8_1;
}

స్థిర అంతఃవరుస size_t htp_mm_f16_tiled_అడ్డువరుస_కొలత(uint32_t ne) {
    స్థిరరాశి uint32_t ne_padded = ((ne + 127) / 128) * 128;
    స్థిరరాశి uint32_t nb_32 = ne_padded / 32;
    తిరిగి nb_32 * HTP_MM_ACT_TILE_కొలత_F16;
}

స్థిర అంతఃవరుస size_t htp_mm_tiled_act_అడ్డువరుస_కొలత(పూర్ణాంకం బరువు_type, uint32_t ne) {
    ఒకవేళ (బరువు_type == HTP_TYPE_F16) {
        తిరిగి htp_mm_f16_tiled_అడ్డువరుస_కొలత(ne);
    }
    తిరిగి htp_mm_బరువు_has_స్థానభ్రంశం(బరువు_type) ? htp_mm_q8_1_tiled_అడ్డువరుస_కొలత(ne) : htp_mm_q8_0_tiled_అడ్డువరుస_కొలత(ne);
}

స్థిర అంతఃవరుస size_t htp_mm_విడత_పైకి(size_t n, size_t m) {
    తిరిగి ((n + m - 1) / m) * m;
}

స్థిర అంతఃవరుస bool htp_mm_hmx_pipeline(uint32_t m) {
    తిరిగి m > 32;
}

స్థిర అంతఃవరుస శూన్య htp_mm_hmx_get_2d_chunk_costs(
    పూర్ణాంకం wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_కొలత,
    size_t * కొలత_చొప్పున_n_బహిర్గతం, size_t * కొలత_చొప్పున_m_బహిర్గతం, size_t * కొలత_చొప్పున_mn_బహిర్గతం
) {
    స్థిరరాశి bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
    స్థిరరాశి size_t అడ్డువరుస_stride = htp_tensor_get_అడ్డువరుస_కొలత(wtype, k);
    స్థిరరాశి size_t vec_dot_కొలత = k * పరిమాణం(uint16_t);
    స్థిరరాశి uint32_t n_బరువు_tiles = k / htp_tensor_get_tile_k(wtype);
    స్థిరరాశి size_t qweight_అడ్డువరుస_stride = is_quant ? (size_t)(n_బరువు_tiles * aligned_tile_కొలత) / 32 : 0;

    *కొలత_చొప్పున_n_బహిర్గతం = (pipeline ? 2 : 1) * (is_quant ? qweight_అడ్డువరుస_stride : అడ్డువరుస_stride) +
                      (pipeline ? 2 * vec_dot_కొలత : vec_dot_కొలత);
    *కొలత_చొప్పున_m_బహిర్గతం = vec_dot_కొలత;
    *కొలత_చొప్పున_mn_బహిర్గతం = (pipeline ? 2 : 1) * పరిమాణం(uint16_t);
}

స్థిర అంతఃవరుస శూన్య htp_mm_hmx_get_batched_chunk_costs(
    uint32_t k, uint32_t సమూహం_కొలత,
    size_t * కొలత_చొప్పున_n_బహిర్గతం, size_t * కొలత_చొప్పున_m_బహిర్గతం, size_t * కొలత_చొప్పున_mn_బహిర్గతం
) {
    స్థిరరాశి size_t vec_dot_కొలత = k * పరిమాణం(uint16_t);
    *కొలత_చొప్పున_n_బహిర్గతం = 3 * vec_dot_కొలత;
    *కొలత_చొప్పున_m_బహిర్గతం = సమూహం_కొలత * vec_dot_కొలత;
    *కొలత_చొప్పున_mn_బహిర్గతం = పరిమాణం(uint16_t);
}

స్థిర అంతఃవరుస size_t htp_mm_hmx_get_2d_overhead(bool pipeline, bool is_matmul_id) {
    size_t అంకె_regions = pipeline ? 7 : (is_matmul_id ? 4 : 5);
    తిరిగి అంకె_regions * HTP_MM_HMX_TILE_కొలత + 256;
}

స్థిర అంతఃవరుస size_t htp_mm_hmx_get_batched_overhead(శూన్య) {
    తిరిగి 5 * HTP_MM_HMX_TILE_కొలత + 256;
}

నిర్మాణం htp_mm_hmx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_బరువు[2];     // [1] is only used when pipelined
    size_t off_act;
    size_t off_act_f32;       // fp32 activation conversion scratch
    size_t off_dst[2];        // [1] is only used when pipelined
    size_t off_scratch[2];    // dequantization scratch pads
    size_t off_scales;        // HMX scales (256 bytes)
    size_t off_bias;          // bias in VTCM

    // Cached sizes of regions for HMX kernel use
    size_t బరువు_area_బైట్లు;
    size_t act_area_బైట్లు;
    size_t act_f32_బైట్లు;
    size_t output_area_బైట్లు;
    size_t scratch_బైట్లు[2];
    size_t act_తల_stride;
    size_t bias_బైట్లు;

    size_t మొత్తం_బైట్లు;
};

నిర్మాణం htp_mm_hvx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_act;           // vtcm_act (activation)
    size_t off_src0;          // vtcm_src0 (weight/Wk)
    size_t off_bias;          // vtcm_bias (bias / fused add only)
    size_t off_dst;           // vtcm_dst (output scratch)
    size_t off_act_raw;       // vtcm_act_raw (raw activation DMA staging)

    // Cached sizes
    size_t src0_బైట్లు;
    size_t act_బైట్లు;
    size_t bias_బైట్లు;
    size_t dst_బైట్లు;
    size_t act_raw_బైట్లు;

    size_t మొత్తం_బైట్లు;
};

స్థిర అంతఃవరుస శూన్య htp_mm_hmx_vtcm_layout_build(
    నిర్మాణం htp_mm_hmx_vtcm_layout * L,
    పూర్ణాంకం kernel_type,
    పూర్ణాంకం wtype,
    uint32_t k,
    size_t mc,
    size_t nc,
    uint32_t సమూహం_కొలత,
    bool pipeline,
    uint32_t act_దారాలు,
    uint32_t aligned_tile_కొలత,
    size_t bias_కొలత
) {
    size_t off = 0;

    ఒకవేళ (kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        స్థిరరాశి size_t vec_dot_కొలత     = k * పరిమాణం(uint16_t);
        స్థిరరాశి size_t act_తల_stride   = mc * k;
        స్థిరరాశి size_t బరువు_area_కొలత  = hex_align_పైకి(nc * vec_dot_కొలత, HTP_MM_HMX_TILE_కొలత);
        స్థిరరాశి size_t activation_area_కొలత = hex_align_పైకి(సమూహం_కొలత * act_తల_stride * పరిమాణం(uint16_t), HTP_MM_HMX_TILE_కొలత);
        స్థిరరాశి size_t output_area_కొలత  = hex_align_పైకి(సమూహం_కొలత * mc * nc * పరిమాణం(uint16_t), HTP_MM_HMX_TILE_కొలత);
        స్థిరరాశి size_t scratch_area_కొలత = hex_align_పైకి(nc * vec_dot_కొలత, HTP_MM_HMX_TILE_కొలత);
        స్థిరరాశి size_t min_f32_కొలత = hex_align_పైకి(act_దారాలు * HTP_MM_DMA_ACT_MULTIPLIER * k * పరిమాణం(దశాంశం), 128);

        // Group A: Permanent activation tiles and scales
        size_t off_సమూహం_a = 0;
        VTCM_LAYOUT_ALLOC(off_సమూహం_a, off_act, activation_area_కొలత);
        VTCM_LAYOUT_ALLOC(off_సమూహం_a, off_scales, HTP_MM_HMX_TILE_కొలత); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_సమూహం_a, off_bias, hex_align_పైకి(bias_కొలత, HTP_MM_HMX_TILE_కొలత), bias_కొలత > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_సమూహం_b = off_సమూహం_a;
        VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_బరువు[0], బరువు_area_కొలత);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_సమూహం_b, off_బరువు[1], బరువు_area_కొలత, false);
        VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_dst[0], output_area_కొలత);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_సమూహం_b, off_dst[1], output_area_కొలత, false);
        VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_scratch[0], scratch_area_కొలత);
        VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_scratch[1], scratch_area_కొలత);

        స్థిరరాశి size_t సమూహం_b_కొలత = off_సమూహం_b - off_సమూహం_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        స్థిరరాశి size_t max_f32_కొలత = act_దారాలు * 64 * k * పరిమాణం(దశాంశం);
        స్థిరరాశి size_t act_f32_కొలత = hex_align_పైకి(hex_smin(max_f32_కొలత, hex_smax(min_f32_కొలత, సమూహం_b_కొలత)), 128);
        size_t off_సమూహం_c = off_సమూహం_a;
        VTCM_LAYOUT_ALLOC(off_సమూహం_c, off_act_f32, act_f32_కొలత);

        స్థిరరాశి size_t సమూహం_c_కొలత = off_సమూహం_c - off_సమూహం_a;

        L->బరువు_area_బైట్లు = బరువు_area_కొలత;
        L->act_area_బైట్లు    = activation_area_కొలత;
        L->act_f32_బైట్లు     = act_f32_కొలత;
        L->output_area_బైట్లు = output_area_కొలత;
        L->scratch_బైట్లు[0]  = scratch_area_కొలత;
        L->scratch_బైట్లు[1]  = scratch_area_కొలత;
        L->act_తల_stride   = act_తల_stride;
        L->bias_బైట్లు        = bias_కొలత;

        off = off_సమూహం_a + hex_smax(సమూహం_b_కొలత, సమూహం_c_కొలత);
    } లేకపోతే {
        // HTP_MM_KERNEL_HMX_2D
        స్థిరరాశి bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
        స్థిరరాశి size_t అడ్డువరుస_stride = htp_tensor_get_అడ్డువరుస_కొలత(wtype, k);
        స్థిరరాశి size_t vec_dot_కొలత = k * పరిమాణం(uint16_t);
        స్థిరరాశి uint32_t n_బరువు_tiles = k / htp_tensor_get_tile_k(wtype);

        స్థిరరాశి size_t min_f32_కొలత = hex_align_పైకి(act_దారాలు * HTP_MM_DMA_ACT_MULTIPLIER * k * పరిమాణం(దశాంశం), 128);
        స్థిరరాశి size_t బరువు_area_కొలత = is_quant
            ? hex_align_పైకి((nc / 32) * n_బరువు_tiles * aligned_tile_కొలత, HTP_MM_HMX_TILE_కొలత)
            : hex_align_పైకి(nc * అడ్డువరుస_stride, HTP_MM_HMX_TILE_కొలత);
        స్థిరరాశి size_t act_area_కొలత    = hex_align_పైకి(mc * vec_dot_కొలత, HTP_MM_HMX_TILE_కొలత);
        స్థిరరాశి size_t output_area_కొలత = hex_align_పైకి(mc * nc * పరిమాణం(__fp16), HTP_MM_HMX_TILE_కొలత);

        స్థిరరాశి size_t scratch0_కొలత = hex_align_పైకి(nc * vec_dot_కొలత, HTP_MM_HMX_TILE_కొలత);
        స్థిరరాశి size_t scratch1_కొలత = pipeline ? scratch0_కొలత : 0;

        // Group A:  Scales and activation tiles (must not overlap with Group B or C)
        size_t off_సమూహం_a = 0;
        VTCM_LAYOUT_ALLOC(off_సమూహం_a, off_scales, HTP_MM_HMX_TILE_కొలత); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC(off_సమూహం_a, off_act, act_area_కొలత);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_సమూహం_a, off_bias, hex_align_పైకి(bias_కొలత, HTP_MM_HMX_TILE_కొలత), bias_కొలత > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_సమూహం_b = off_సమూహం_a;
        VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_బరువు[0], బరువు_area_కొలత);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_సమూహం_b, off_బరువు[1], బరువు_area_కొలత, pipeline);
        VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_dst[0], output_area_కొలత);
        VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_scratch[0], scratch0_కొలత);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_సమూహం_b, off_scratch[1], scratch0_కొలత, pipeline);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_సమూహం_b, off_dst[1], output_area_కొలత, pipeline);

        స్థిరరాశి size_t సమూహం_b_కొలత = off_సమూహం_b - off_సమూహం_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        స్థిరరాశి size_t max_f32_కొలత = act_దారాలు * 64 * k * పరిమాణం(దశాంశం);
        స్థిరరాశి size_t act_f32_కొలత = hex_align_పైకి(hex_smin(max_f32_కొలత, hex_smax(min_f32_కొలత, సమూహం_b_కొలత)), 128);
        size_t off_సమూహం_c = off_సమూహం_a;
        VTCM_LAYOUT_ALLOC(off_సమూహం_c, off_act_f32, act_f32_కొలత);

        స్థిరరాశి size_t సమూహం_c_కొలత = off_సమూహం_c - off_సమూహం_a;

        L->బరువు_area_బైట్లు = బరువు_area_కొలత;
        L->act_area_బైట్లు    = act_area_కొలత;
        L->act_f32_బైట్లు     = act_f32_కొలత;
        L->output_area_బైట్లు = output_area_కొలత;
        L->scratch_బైట్లు[0]  = scratch0_కొలత;
        L->scratch_బైట్లు[1]  = scratch1_కొలత;
        L->act_తల_stride   = 0;
        L->bias_బైట్లు        = bias_కొలత;

        off = off_సమూహం_a + hex_smax(సమూహం_b_కొలత, సమూహం_c_కొలత);
    }

    L->మొత్తం_బైట్లు = off;
}

స్థిర అంతఃవరుస శూన్య htp_mm_hvx_vtcm_layout_build(
    నిర్మాణం htp_mm_hvx_vtcm_layout * L,
    పూర్ణాంకం kernel_type,
    పూర్ణాంకం wtype,
    uint32_t ne10,       // k
    uint32_t act_nrows,  // m_total
    uint32_t n_దారాలు,
    size_t dst_అడ్డువరుస_కొలత,
    size_t src0_అడ్డువరుస_కొలత,
    size_t act_అడ్డువరుస_కొలత,
    size_t bias_అడ్డువరుస_కొలత,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_nx
) {
    (శూన్య)act_అడ్డువరుస_కొలత;
    size_t src0_sz    = 0;
    size_t act_sz     = 0;
    size_t bias_sz    = bias_అడ్డువరుస_కొలత > 0 ? htp_mm_విడత_పైకి(bias_అడ్డువరుస_కొలత, 128) : 0;
    size_t dst_sz     = 0;
    size_t act_raw_sz = 0;

    ఒకవేళ (is_fused_nx) {
        uint32_t aligned_tile_కొలత = htp_tensor_get_tile_padded_కొలత(wtype);
        uint32_t n_k_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        uint32_t tile_అడ్డువరుస_కొలత = n_k_tiles * aligned_tile_కొలత;
        size_t బరువు_sz_చొప్పున_thread = hex_విడత_పైకి(n_prefetch * tile_అడ్డువరుస_కొలత, 128);

        size_t tiled_act_అడ్డువరుస_కొలత = htp_mm_tiled_act_అడ్డువరుస_కొలత(wtype, ne10);
        size_t q_act_sz = hex_విడత_పైకి(tiled_act_అడ్డువరుస_కొలత * act_nrows, 128);
        size_t raw_అడ్డువరుస_కొలత = hex_విడత_పైకి(ne10 * పరిమాణం(దశాంశం), QK_Q8_0_TILED * పరిమాణం(దశాంశం));

        src0_sz    = బరువు_sz_చొప్పున_thread * n_దారాలు; // shared single-weight prefetch buffer
        act_sz     = q_act_sz;                         // quantized activation buffer
        bias_sz    = 0;
        dst_sz     = 0;
        act_raw_sz = hex_విడత_పైకి(raw_అడ్డువరుస_కొలత * act_nrows, 128);
    } లేకపోతే ఒకవేళ (is_matmul_id) {
        స్థిరరాశి size_t act_అడ్డువరుస_కొలత_tiled = htp_mm_tiled_act_అడ్డువరుస_కొలత(wtype, ne10);
        స్థిరరాశి uint32_t aligned_tile_కొలత = htp_tensor_get_tile_padded_కొలత(wtype);
        స్థిరరాశి uint32_t n_k_tiles         = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        స్థిరరాశి uint32_t tile_అడ్డువరుస_కొలత     = n_k_tiles * aligned_tile_కొలత;
        size_t src0_sz_చొప్పున_thread        = htp_mm_విడత_పైకి(n_prefetch * tile_అడ్డువరుస_కొలత, 256);

        act_sz = htp_mm_విడత_పైకి(act_అడ్డువరుస_కొలత_tiled * act_nrows, 256);

        size_t raw_అడ్డువరుస_కొలత = hex_విడత_పైకి(ne10 * పరిమాణం(దశాంశం), QK_Q8_0_TILED * పరిమాణం(దశాంశం));

        src0_sz    = src0_sz_చొప్పున_thread * n_దారాలు;
        dst_sz     = 0;
        bias_sz    = 0;
        act_raw_sz = hex_విడత_పైకి(raw_అడ్డువరుస_కొలత * act_nrows, 128);
    } లేకపోతే {
        స్థిరరాశి size_t src0_అడ్డువరుస_కొలత_padded = htp_mm_విడత_పైకి(src0_అడ్డువరుస_కొలత, 128);
        స్థిరరాశి size_t dst_nrows = (act_nrows > 1) ? 0 : 1;

        ఎంపిక (kernel_type) {
            స్థితి HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                size_t f16_act_అడ్డువరుస_కొలత = htp_mm_విడత_పైకి(ne10 * 2, 128);
                act_sz     = htp_mm_విడత_పైకి(f16_act_అడ్డువరుస_కొలత * act_nrows, 256);
                src0_sz    = htp_mm_విడత_పైకి(n_prefetch * src0_అడ్డువరుస_కొలత_padded, 256) * n_దారాలు;
                dst_sz     = dst_nrows > 0 ? htp_mm_విడత_పైకి(dst_అడ్డువరుస_కొలత, 128) * n_దారాలు : 0;
                act_raw_sz = hex_విడత_పైకి(hex_విడత_పైకి(ne10 * పరిమాణం(దశాంశం), 128) * act_nrows, 128);
                ఆపు;
            }
            స్థితి HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            స్థితి HTP_MM_KERNEL_HVX_BF16_F32_VTCM: {
                size_t f32_act_అడ్డువరుస_కొలత = htp_mm_విడత_పైకి(ne10 * 4, 128);
                act_sz     = htp_mm_విడత_పైకి(f32_act_అడ్డువరుస_కొలత * act_nrows, 256);
                src0_sz    = htp_mm_విడత_పైకి(n_prefetch * src0_అడ్డువరుస_కొలత_padded, 256) * n_దారాలు;
                dst_sz     = dst_nrows > 0 ? htp_mm_విడత_పైకి(dst_అడ్డువరుస_కొలత, 128) * n_దారాలు : 0;
                act_raw_sz = 0;
                ఆపు;
            }
            స్థితి HTP_MM_KERNEL_HVX_QUANT_ఖండం:
            స్థితి HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస: {
                size_t q_act_అడ్డువరుస_కొలత = htp_mm_tiled_act_అడ్డువరుస_కొలత(wtype, ne10);
                uint32_t aligned_tile_కొలత = htp_tensor_get_tile_padded_కొలత(wtype);
                uint32_t n_బరువు_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
                uint32_t tile_అడ్డువరుస_కొలత = n_బరువు_tiles * aligned_tile_కొలత;
                size_t tiled_vtcm_కొలత = htp_mm_విడత_పైకి(n_prefetch * tile_అడ్డువరుస_కొలత, 256);

                src0_sz = tiled_vtcm_కొలత * n_దారాలు;
                act_sz  = htp_mm_విడత_పైకి(q_act_అడ్డువరుస_కొలత * act_nrows, 256);

                size_t dst_ముక్క_చొప్పున_thread = (dst_nrows > 0 && act_nrows == 1) ? htp_mm_విడత_పైకి((dst_అడ్డువరుస_కొలత + n_దారాలు - 1) / n_దారాలు, 128) : 0;
                dst_sz = dst_ముక్క_చొప్పున_thread * n_దారాలు;
                size_t raw_అడ్డువరుస_కొలత = hex_విడత_పైకి(ne10 * పరిమాణం(దశాంశం), QK_Q8_0_TILED * పరిమాణం(దశాంశం));
                act_raw_sz = hex_విడత_పైకి(raw_అడ్డువరుస_కొలత * act_nrows, 128);
                ఆపు;
            }
            అప్రమేయ:
                ఆపు;
        }
    }

    // Group A: Persistent buffers across chunk compute
    size_t off_సమూహం_a = 0;
    VTCM_LAYOUT_ALLOC(off_సమూహం_a, off_act, act_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_సమూహం_a, off_bias, bias_sz, bias_sz > 0);

    // Group B: Compute-only buffers (starts at off_group_a)
    size_t off_సమూహం_b = off_సమూహం_a;
    VTCM_LAYOUT_ALLOC(off_సమూహం_b, off_src0, src0_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_సమూహం_b, off_dst, dst_sz, dst_sz > 0);
    స్థిరరాశి size_t సమూహం_b_కొలత = off_సమూహం_b - off_సమూహం_a;

    // Group C: Raw activation staging buffer (overlaps Group B, starts at off_group_a)
    size_t off_సమూహం_c = off_సమూహం_a;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_సమూహం_c, off_act_raw, act_raw_sz, act_raw_sz > 0);
    స్థిరరాశి size_t సమూహం_c_కొలత = off_సమూహం_c - off_సమూహం_a;

    L->src0_బైట్లు    = src0_sz;
    L->act_బైట్లు     = act_sz;
    L->bias_బైట్లు    = bias_sz;
    L->dst_బైట్లు     = dst_sz;
    L->act_raw_బైట్లు = act_raw_sz;
    L->మొత్తం_బైట్లు   = off_సమూహం_a + hex_smax(సమూహం_b_కొలత, సమూహం_c_కొలత);
}

స్థిర అంతఃవరుస bool htp_mm_hvx_solve_vtcm_params(
    పూర్ణాంకం kernel_type,
    పూర్ణాంకం wtype,
    uint32_t ne10,
    uint32_t act_nrows,
    uint32_t n_దారాలు,
    size_t dst_అడ్డువరుస_కొలత,
    size_t src0_అడ్డువరుస_కొలత,
    size_t act_అడ్డువరుస_కొలత,
    size_t bias_అడ్డువరుస_కొలత,
    uint32_t n_prefetch,
    size_t vtcm_budget,
    నిర్మాణం htp_mm_hvx_vtcm_layout * L_బహిర్గతం,
    uint32_t * m_chunk_బహిర్గతం
) {
    నిర్మాణం htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, act_nrows, n_దారాలు,
        dst_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, act_అడ్డువరుస_కొలత, bias_అడ్డువరుస_కొలత, n_prefetch, false, false
    );

    ఒకవేళ (L.మొత్తం_బైట్లు <= vtcm_budget) {
        *L_బహిర్గతం = L;
        *m_chunk_బహిర్గతం = act_nrows;
        తిరిగి true;
    }

    స్థిరరాశి size_t fixed_బైట్లు = L.src0_బైట్లు + L.bias_బైట్లు + L.dst_బైట్లు;
    ఒకవేళ (vtcm_budget <= fixed_బైట్లు) {
        తిరిగి false;
    }

    స్థిరరాశి size_t avail_act = vtcm_budget - fixed_బైట్లు;
    size_t అడ్డువరుస_కొలత = 0;
    ఒకవేళ (kernel_type == HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస || kernel_type == HTP_MM_KERNEL_HVX_QUANT_ఖండం) {
        అడ్డువరుస_కొలత = htp_mm_tiled_act_అడ్డువరుస_కొలత(wtype, ne10);
    } లేకపోతే ఒకవేళ (kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        అడ్డువరుస_కొలత = hex_విడత_పైకి(ne10 * 2, 128);
    } లేకపోతే {
        అడ్డువరుస_కొలత = hex_విడత_పైకి(ne10 * 4, 128);
    }
    ఒకవేళ (అడ్డువరుస_కొలత == 0) {
        తిరిగి false;
    }

    size_t eff_అడ్డువరుస_కొలత = అడ్డువరుస_కొలత;
    ఒకవేళ (kernel_type == HTP_MM_KERNEL_HVX_QUANT_అడ్డువరుస || kernel_type == HTP_MM_KERNEL_HVX_QUANT_ఖండం) {
        eff_అడ్డువరుస_కొలత += hex_విడత_పైకి(ne10 * పరిమాణం(దశాంశం), QK_Q8_0_TILED * పరిమాణం(దశాంశం));
    }

    uint32_t m_chunk = (uint32_t) (avail_act / eff_అడ్డువరుస_కొలత);
    ఒకవేళ (m_chunk > 1) {
        m_chunk &= ~1U;
    }
    ఒకవేళ (m_chunk > act_nrows) {
        m_chunk = act_nrows;
    }
    ఒకవేళ (m_chunk < 1) {
        తిరిగి false;
    }

    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, m_chunk, n_దారాలు,
        dst_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, act_అడ్డువరుస_కొలత, bias_అడ్డువరుస_కొలత, n_prefetch, false, false
    );

    అయితే (m_chunk > 2 && L.మొత్తం_బైట్లు > vtcm_budget) {
        m_chunk -= 2;
        htp_mm_hvx_vtcm_layout_build(
            &L, kernel_type, wtype, ne10, m_chunk, n_దారాలు,
            dst_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, act_అడ్డువరుస_కొలత, bias_అడ్డువరుస_కొలత, n_prefetch, false, false
        );
    }

    ఒకవేళ (L.మొత్తం_బైట్లు <= vtcm_budget) {
        *L_బహిర్గతం = L;
        *m_chunk_బహిర్గతం = m_chunk;
        తిరిగి true;
    }

    తిరిగి false;
}

స్థిర అంతఃవరుస size_t htp_mm_hmx_get_2d_vtcm_కొలత(
    పూర్ణాంకం wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_దారాలు, uint32_t aligned_tile_కొలత, size_t bias_కొలత
) {
    నిర్మాణం htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_దారాలు, aligned_tile_కొలత, bias_కొలత);
    తిరిగి L.మొత్తం_బైట్లు;
}

స్థిర అంతఃవరుస size_t htp_mm_hmx_get_batched_vtcm_కొలత(
    పూర్ణాంకం wtype, uint32_t k, size_t mc, size_t nc, uint32_t సమూహం_కొలత, bool pipeline, uint32_t act_దారాలు, size_t bias_కొలత) {
    (శూన్య)pipeline;
    నిర్మాణం htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, wtype, k, mc, nc, సమూహం_కొలత, false, act_దారాలు, 0, bias_కొలత);
    తిరిగి L.మొత్తం_బైట్లు;
}

స్థిర అంతఃవరుస bool htp_mm_hmx_solve_batched_params(
    పూర్ణాంకం wtype,
    uint32_t k,
    uint32_t ne01_padded,
    uint32_t ne11,
    uint32_t సమూహం_కొలత,
    పూర్ణాంకం n_దారాలు,
    bool pipeline,
    size_t bias_కొలత,
    size_t vtcm_budget,
    size_t * m_chunk_బహిర్గతం,
    size_t * n_chunk_బహిర్గతం,
    పూర్ణాంకం * act_దారాలు_బహిర్గతం,
    size_t * vtcm_కొలత_బహిర్గతం
) {
    size_t ఉత్తమ_mblocks = SIZE_MAX;
    పూర్ణాంకం ఉత్తమ_act_దారాలు = 0;
    size_t ఉత్తమ_m_chunk = 0;
    size_t ఉత్తమ_n_chunk = 0;
    size_t ఉత్తమ_vtcm_కొలత = 0;

    పూర్ణాంకం act_దారాలు = n_దారాలు;
    అయితే (act_దారాలు >= 1) {
        size_t సమూహం_overhead = htp_mm_hmx_get_batched_overhead() + (bias_కొలత > 0 ? hex_align_పైకి(bias_కొలత, HTP_MM_HMX_TILE_కొలత) : 0);
        size_t సమూహం_కొలత_చొప్పున_n, సమూహం_కొలత_చొప్పున_m, సమూహం_కొలత_చొప్పున_mn;
        htp_mm_hmx_get_batched_chunk_costs(k, సమూహం_కొలత, &సమూహం_కొలత_చొప్పున_n, &సమూహం_కొలత_చొప్పున_m, &సమూహం_కొలత_చొప్పున_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_కొలత_candidate = 0;

        ఒకవేళ (htp_mm_hmx_compute_chunks(vtcm_budget, సమూహం_overhead, సమూహం_కొలత_చొప్పున_n, సమూహం_కొలత_చొప్పున_m, సమూహం_కొలత_చొప్పున_mn, hex_align_పైకి(ne11, 32), ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_ఖర్చు_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_ఖర్చు_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_కొలత_candidate) == 0) {
            size_t exact_కొలత = htp_mm_hmx_get_batched_vtcm_కొలత(wtype, k, m_chunk_candidate, n_chunk_candidate, సమూహం_కొలత, pipeline, act_దారాలు, bias_కొలత);
            ఒకవేళ (exact_కొలత <= vtcm_budget) {
                size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                ఒకవేళ (mblocks < ఉత్తమ_mblocks || (mblocks == ఉత్తమ_mblocks && act_దారాలు > ఉత్తమ_act_దారాలు)) {
                    ఉత్తమ_mblocks = mblocks;
                    ఉత్తమ_act_దారాలు = act_దారాలు;
                    ఉత్తమ_m_chunk = m_chunk_candidate;
                    ఉత్తమ_n_chunk = n_chunk_candidate;
                    ఉత్తమ_vtcm_కొలత = exact_కొలత;
                }
            }
        }
        ఒకవేళ (act_దారాలు == 1) {
            act_దారాలు = 0;
        } లేకపోతే {
            act_దారాలు /= 2;
        }
    }

    ఒకవేళ (ఉత్తమ_act_దారాలు > 0) {
        *m_chunk_బహిర్గతం = ఉత్తమ_m_chunk;
        *n_chunk_బహిర్గతం = ఉత్తమ_n_chunk;
        *vtcm_కొలత_బహిర్గతం = ఉత్తమ_vtcm_కొలత;
        *act_దారాలు_బహిర్గతం = ఉత్తమ_act_దారాలు;
        తిరిగి true;
    }
    తిరిగి false;
}

స్థిర అంతఃవరుస bool htp_mm_hmx_solve_2d_params(
    పూర్ణాంకం wtype,
    uint32_t k,
    uint32_t m_id_అడ్డువరుసలు,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_ఖర్చు,
    పూర్ణాంకం n_దారాలు,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_కొలత,
    size_t bias_కొలత,
    size_t vtcm_budget,
    size_t * m_chunk_బహిర్గతం,
    size_t * n_chunk_బహిర్గతం,
    పూర్ణాంకం * act_దారాలు_బహిర్గతం,
    size_t * vtcm_కొలత_బహిర్గతం
) {
    size_t ఉత్తమ_mblocks = SIZE_MAX;
    పూర్ణాంకం ఉత్తమ_act_దారాలు = 0;
    size_t ఉత్తమ_m_chunk = 0;
    size_t ఉత్తమ_n_chunk = 0;
    size_t ఉత్తమ_vtcm_కొలత = 0;

    స్థిరరాశి పూర్ణాంకం m_for_chunks = is_matmul_id ? hex_align_పైకి(m_id_అడ్డువరుసలు, 32) : ne11_padded;

    పూర్ణాంకం act_దారాలు = n_దారాలు;
    అయితే (act_దారాలు >= 1) {
        size_t simple_2d_overhead = htp_mm_hmx_get_2d_overhead(pipeline, is_matmul_id) + (bias_కొలత > 0 ? hex_align_పైకి(bias_కొలత, HTP_MM_HMX_TILE_కొలత) : 0);
        size_t simple_2d_కొలత_చొప్పున_n, simple_2d_కొలత_చొప్పున_m, simple_2d_కొలత_చొప్పున_mn;
        htp_mm_hmx_get_2d_chunk_costs(wtype, k, pipeline, aligned_tile_కొలత, &simple_2d_కొలత_చొప్పున_n, &simple_2d_కొలత_చొప్పున_m, &simple_2d_కొలత_చొప్పున_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_కొలత_candidate = 0;

        ఒకవేళ (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_కొలత_చొప్పున_n, simple_2d_కొలత_చొప్పున_m, simple_2d_కొలత_చొప్పున_mn, m_for_chunks, ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_ఖర్చు_W_DEQUANT, (size_t) m_for_ఖర్చు * HTP_MM_HMX_ఖర్చు_A_CONVERT,
                               &m_chunk_candidate, &n_chunk_candidate, &vtcm_కొలత_candidate) == 0) {
            size_t exact_కొలత = htp_mm_hmx_get_2d_vtcm_కొలత(wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_దారాలు, aligned_tile_కొలత, bias_కొలత);
            ఒకవేళ (exact_కొలత <= vtcm_budget) {
                size_t mblocks = ((size_t) m_for_ఖర్చు + m_chunk_candidate - 1) / m_chunk_candidate;
                ఒకవేళ (mblocks < ఉత్తమ_mblocks || (mblocks == ఉత్తమ_mblocks && act_దారాలు > ఉత్తమ_act_దారాలు)) {
                    ఉత్తమ_mblocks = mblocks;
                    ఉత్తమ_act_దారాలు = act_దారాలు;
                    ఉత్తమ_m_chunk = m_chunk_candidate;
                    ఉత్తమ_n_chunk = n_chunk_candidate;
                    ఉత్తమ_vtcm_కొలత = exact_కొలత;
                }
            }
        }
        ఒకవేళ (act_దారాలు == 1) {
            act_దారాలు = 0;
        } లేకపోతే {
            act_దారాలు /= 2;
        }
    }

    ఒకవేళ (ఉత్తమ_act_దారాలు > 0) {
        *m_chunk_బహిర్గతం = ఉత్తమ_m_chunk;
        *n_chunk_బహిర్గతం = ఉత్తమ_n_chunk;
        *vtcm_కొలత_బహిర్గతం = ఉత్తమ_vtcm_కొలత;
        *act_దారాలు_బహిర్గతం = ఉత్తమ_act_దారాలు;
        తిరిగి true;
    }
    తిరిగి false;
}

#నిర్వచితమైతే __cplusplus
}
#అయితే_ముగింపు

#అయితే_ముగింపు // HTP_MATMUL_OPS_H
