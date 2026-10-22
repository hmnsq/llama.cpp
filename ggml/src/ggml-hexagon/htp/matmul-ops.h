#اگر_غیر_تعریف_شدہ HTP_MATMUL_OPS_H
#تعریف_کرو HTP_MATMUL_OPS_H

#شامل <stdint.h>
#شامل <stddef.h>
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "hex-fastdiv.h"
#شامل "hex-common.h"
#شامل "htp-vtcm.h"

#اگر_تعریف_شدہ __cplusplus
خارجی "C" {
#اگر_ختم

// --- HMX Tile Constraints ---
#تعریف_کرو HTP_MM_HMX_TILE_N_ستون_تعداد 32
#تعریف_کرو HTP_MM_HMX_TILE_N_صفیں 32
#تعریف_کرو HTP_MM_HMX_TILE_سائز   (32 * 32 * حجم(__fp16)) // 2048 bytes
#تعریف_کرو HTP_MM_HMX_TILE_N_ELMS 1024
#تعریف_کرو HTP_MM_HMX_MIN_NROWS   4

#تعریف_کرو HTP_MM_وزن_TILE_K_Q2_0             64
#تعریف_کرو HTP_MM_وزن_TILE_سائز_Q2_0          HTP_TENSOR_TILE_سائز_Q2_0
#تعریف_کرو HTP_MM_وزن_ALIGNED_TILE_سائز_Q2_0  640

// --- Activation Tiled Block Sizes (including padding) ---
#تعریف_کرو HTP_MM_ACT_TILE_سائز_Q8_0      1152
#تعریف_کرو HTP_MM_ACT_TILE_سائز_Q8_1      1280
#تعریف_کرو HTP_MM_ACT_TILE_سائز_F16       2048

#تعریف_کرو HTP_MM_MAX_PREFETCH 16

// --- Solver Cost Model Penalty Weights (HMX-specific) ---
#تعریف_کرو HTP_MM_HMX_لاگت_W_DEQUANT 3 // cost penalty for quantized weight loading/dequantization
#تعریف_کرو HTP_MM_HMX_لاگت_A_CONVERT 2 // cost penalty for activation loading/conversion

// --- DMA Activation Transfer Configuration ---
#تعریف_کرو HTP_MM_DMA_ACT_صفیں_فی_قدم 2
#تعریف_کرو HTP_MM_DMA_ACT_MULTIPLIER    (2 * HTP_MM_DMA_ACT_صفیں_فی_قدم)

شمار htp_mm_kernel_type {
    HTP_MM_KERNEL_UNSUPPORTED = 0,

    // HMX paths
    HTP_MM_KERNEL_HMX_2D,
    HTP_MM_KERNEL_HMX_BATCHED,

    // HVX floating-point paths
    HTP_MM_KERNEL_HVX_F16_F16_VTCM,
    HTP_MM_KERNEL_HVX_F32_F32_VTCM,
    HTP_MM_KERNEL_HVX_BF16_F32_VTCM,

    // HVX quantized paths
    HTP_MM_KERNEL_HVX_QUANT_صف,      // standard row-wise parallel quantization
    HTP_MM_KERNEL_HVX_QUANT_بلاک,    // parallel block-wise quantization
};

// Op-specific struct for precomputed matmul params
ساخت htp_mm_kernel_params {
    uint8_t  kernel_type;        // enum htp_mm_kernel_type
    uint8_t  pipeline;           // 1 = pipelined execution, 0 = standard
    uint8_t  collapse;           // 1 = collapse outer dims into 2D, 0 = standard
    uint8_t  n_hmx;              // 1 = use HMX, 0 = use HVX

    uint8_t  n_دھاگے;          // Number of threads to spawn
    uint8_t  n_act_دھاگے;      // Number of threads for activation preparation
    uint8_t  n_prefetch;         // Prefetch lookahead buffers/rows in VTCM
    uint8_t  n_weights;          // Number of weights for fused NX

    int32_t  m_chunk;            // Row chunk size (M chunk)
    int32_t  n_chunk;            // Col chunk size (N chunk)
    int32_t  tile_سائز;          // Weight tile size
    int32_t  aligned_tile_سائز;  // Aligned weight tile size (padded to 128)
    int32_t  act_صف_سائز;       // Row size for activation scratchpad
    int32_t  vtcm_سائز;          // Total required scratchpad size in VTCM
    int32_t  vtcm_src0_سائز;     // src0 scratchpad size in VTCM
    int32_t  vtcm_act_سائز;      // activation scratchpad size in VTCM
    int32_t  vtcm_bias_سائز;     // bias scratchpad size in VTCM (fused only)
    int32_t  vtcm_dst_سائز;      // dst scratchpad size in VTCM

    // Precomputed division values
    ساخت fastdiv_اقدار div_ne12_ne1;
    ساخت fastdiv_اقدار div_ne1;
    ساخت fastdiv_اقدار div_r2;
    ساخت fastdiv_اقدار div_r3;
    ساخت fastdiv_اقدار div_ne12;
    ساخت fastdiv_اقدار div_n_act_دھاگے;
    ساخت fastdiv_اقدار div_ne00_padded;
};

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#ورنہ
_Static_assert(حجم(ساخت htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#اگر_ختم

ساخت mmid_صف_mapping {
    uint32_t i1;
    uint32_t i2;
};

// Search for optimal (mc, nc) chunk sizes within VTCM budget.
ساکن اندرونی صحیح htp_mm_hmx_compute_chunks(size_t   vtcm_کل,
                              size_t   overhead,
                              size_t   فی_n_لاگت,
                              size_t   فی_m_لاگت,
                              size_t   فی_mn_لاگت,
                              size_t   m,
                              size_t   n,
                              size_t   m_بلاک_لاگت,
                              size_t   n_بلاک_لاگت,
                              size_t * m_chunk_برآمد,
                              size_t * n_chunk_برآمد,
                              size_t * کل_برآمد) {
    اگر (m == 0 || n == 0) واپس -1;
    اگر (vtcm_کل <= overhead) واپس -1;
    اگر (فی_n_لاگت == 0 || فی_m_لاگت == 0 || فی_mn_لاگت == 0) واپس -1;

    ثابت size_t usable = vtcm_کل - overhead;

    size_t بہترین_لاگت = SIZE_MAX;
    size_t بہترین_دم_waste = SIZE_MAX;
    size_t بہترین_mn   = 0;
    size_t بہترین_m = 0, بہترین_n = 0;

    ثابت size_t max_nc_budget = (usable / فی_n_لاگت);
    ثابت size_t n_max = hex_align_نیچے_کو(hex_smin((size_t)n, max_nc_budget), HTP_MM_HMX_TILE_N_ستون_تعداد);
    برائے (size_t nc = n_max; nc >= HTP_MM_HMX_TILE_N_ستون_تعداد; nc -= HTP_MM_HMX_TILE_N_ستون_تعداد) {
        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        اگر (hex_mul_حد_سے_تجاوز(nc, فی_n_لاگت, &n_fixed)) جاری;
        اگر (n_fixed >= usable) جاؤ اگلا_nc;

        اگر (hex_mul_حد_سے_تجاوز(nc, فی_mn_لاگت, &ncmn)) جاؤ اگلا_nc;
        اگر (hex_جوڑو_حد_سے_تجاوز(فی_m_لاگت, ncmn, &mc_denom) || mc_denom == 0) جاؤ اگلا_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hex_align_نیچے_کو(mc, HTP_MM_HMX_TILE_N_صفیں);
            mc = hex_smin(mc, m);

            اگر (mc == 0) {
                جاؤ اگلا_nc;
            }

            size_t mblocks = ((size_t) m + mc - 1) / mc;
            size_t nblocks = ((size_t) n + nc - 1) / nc;
            size_t لاگت    = mblocks * m_بلاک_لاگت + nblocks * n_بلاک_لاگت;
            size_t rem     = n % nc;
            size_t دم_waste = (rem == 0) ? 0 : (nc - rem);
            size_t mn      = mc * nc;
            اگر (لاگت < بہترین_لاگت ||
                (لاگت == بہترین_لاگت && دم_waste < بہترین_دم_waste) ||
                (لاگت == بہترین_لاگت && دم_waste == بہترین_دم_waste && mn > بہترین_mn)) {
                بہترین_لاگت       = لاگت;
                بہترین_دم_waste = دم_waste;
                بہترین_mn         = mn;
                بہترین_m          = mc;
                بہترین_n          = nc;
            }
        }

اگلا_nc:
        اگر (nc == HTP_MM_HMX_TILE_N_ستون_تعداد) توڑو;  // avoid size_t underflow
    }

    اگر (بہترین_m == 0 || بہترین_n == 0) واپس -1;

    // Compute exact total (with overflow checks)
    size_t t0 = 0, t1 = 0, t2 = 0, mn = 0, کل = 0;
    اگر (hex_mul_حد_سے_تجاوز(بہترین_n, فی_n_لاگت, &t0)) واپس -1;
    اگر (hex_mul_حد_سے_تجاوز(بہترین_m, فی_m_لاگت, &t1)) واپس -1;
    اگر (hex_mul_حد_سے_تجاوز(بہترین_m, بہترین_n, &mn))     واپس -1;
    اگر (hex_mul_حد_سے_تجاوز(mn, فی_mn_لاگت, &t2))    واپس -1;
    اگر (hex_جوڑو_حد_سے_تجاوز(t0, t1, &کل))          واپس -1;
    اگر (hex_جوڑو_حد_سے_تجاوز(کل, t2, &کل))       واپس -1;
    اگر (hex_جوڑو_حد_سے_تجاوز(کل, overhead, &کل)) واپس -1;

    *m_chunk_برآمد = بہترین_m;
    *n_chunk_برآمد = بہترین_n;
    *کل_برآمد   = کل;
    واپس 0;
}

// weight types whose tiles carry a per-block offset (x = d * q + m): the activations need block sums (q8_1)
// (Q2_K: per-16 k sums, q8_1_s16)
ساکن اندرونی bool htp_mm_وزن_has_انحراف(صحیح وزن_type) {
    واپس وزن_type == HTP_TYPE_Q4_1 || وزن_type == HTP_TYPE_Q4_K || وزن_type == HTP_TYPE_Q5_K ||
           وزن_type == HTP_TYPE_Q2_K || وزن_type == HTP_TYPE_Q2_0;
}

// --- Activation/Row Size Helpers ---
ساکن اندرونی size_t htp_mm_q8_0_tiled_صف_سائز(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    واپس nb_32 * HTP_MM_ACT_TILE_سائز_Q8_0;
}

ساکن اندرونی size_t htp_mm_q8_1_tiled_صف_سائز(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    واپس nb_32 * HTP_MM_ACT_TILE_سائز_Q8_1;
}

ساکن اندرونی size_t htp_mm_f16_tiled_صف_سائز(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    واپس nb_32 * HTP_MM_ACT_TILE_سائز_F16;
}

ساکن اندرونی size_t htp_mm_tiled_act_صف_سائز(صحیح وزن_type, uint32_t ne) {
    اگر (وزن_type == HTP_TYPE_F16) {
        واپس htp_mm_f16_tiled_صف_سائز(ne);
    }
    واپس htp_mm_وزن_has_انحراف(وزن_type) ? htp_mm_q8_1_tiled_صف_سائز(ne) : htp_mm_q8_0_tiled_صف_سائز(ne);
}

ساکن اندرونی size_t htp_mm_باری_اوپر(size_t n, size_t m) {
    واپس ((n + m - 1) / m) * m;
}

ساکن اندرونی bool htp_mm_hmx_pipeline(uint32_t m) {
    واپس m > 32;
}

ساکن اندرونی خالی htp_mm_hmx_get_2d_chunk_costs(
    صحیح wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_سائز,
    size_t * سائز_فی_n_برآمد, size_t * سائز_فی_m_برآمد, size_t * سائز_فی_mn_برآمد
) {
    ثابت bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
    ثابت size_t صف_stride = htp_tensor_get_صف_سائز(wtype, k);
    ثابت size_t vec_dot_سائز = k * حجم(uint16_t);
    ثابت uint32_t n_وزن_tiles = k / htp_tensor_get_tile_k(wtype);
    ثابت size_t qweight_صف_stride = is_quant ? (size_t)(n_وزن_tiles * aligned_tile_سائز) / 32 : 0;

    *سائز_فی_n_برآمد = (pipeline ? 2 : 1) * (is_quant ? qweight_صف_stride : صف_stride) +
                      (pipeline ? 2 * vec_dot_سائز : vec_dot_سائز);
    *سائز_فی_m_برآمد = vec_dot_سائز;
    *سائز_فی_mn_برآمد = (pipeline ? 2 : 1) * حجم(uint16_t);
}

ساکن اندرونی خالی htp_mm_hmx_get_batched_chunk_costs(
    uint32_t k, uint32_t گروہ_سائز,
    size_t * سائز_فی_n_برآمد, size_t * سائز_فی_m_برآمد, size_t * سائز_فی_mn_برآمد
) {
    ثابت size_t vec_dot_سائز = k * حجم(uint16_t);
    *سائز_فی_n_برآمد = 3 * vec_dot_سائز;
    *سائز_فی_m_برآمد = گروہ_سائز * vec_dot_سائز;
    *سائز_فی_mn_برآمد = حجم(uint16_t);
}

ساکن اندرونی size_t htp_mm_hmx_get_2d_overhead(bool pipeline, bool is_matmul_id) {
    size_t عدد_regions = pipeline ? 7 : (is_matmul_id ? 4 : 5);
    واپس عدد_regions * HTP_MM_HMX_TILE_سائز + 256;
}

ساکن اندرونی size_t htp_mm_hmx_get_batched_overhead(خالی) {
    واپس 5 * HTP_MM_HMX_TILE_سائز + 256;
}

ساخت htp_mm_hmx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_وزن[2];     // [1] is only used when pipelined
    size_t off_act;
    size_t off_act_f32;       // fp32 activation conversion scratch
    size_t off_dst[2];        // [1] is only used when pipelined
    size_t off_scratch[2];    // dequantization scratch pads
    size_t off_scales;        // HMX scales (256 bytes)
    size_t off_bias;          // bias in VTCM

    // Cached sizes of regions for HMX kernel use
    size_t وزن_area_بائٹیں;
    size_t act_area_بائٹیں;
    size_t act_f32_بائٹیں;
    size_t output_area_بائٹیں;
    size_t scratch_بائٹیں[2];
    size_t act_سر_stride;
    size_t bias_بائٹیں;

    size_t کل_بائٹیں;
};

ساخت htp_mm_hvx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_act;           // vtcm_act (activation)
    size_t off_src0;          // vtcm_src0 (weight/Wk)
    size_t off_bias;          // vtcm_bias (bias / fused add only)
    size_t off_dst;           // vtcm_dst (output scratch)
    size_t off_act_raw;       // vtcm_act_raw (raw activation DMA staging)

    // Cached sizes
    size_t src0_بائٹیں;
    size_t act_بائٹیں;
    size_t bias_بائٹیں;
    size_t dst_بائٹیں;
    size_t act_raw_بائٹیں;

    size_t کل_بائٹیں;
};

ساکن اندرونی خالی htp_mm_hmx_vtcm_layout_build(
    ساخت htp_mm_hmx_vtcm_layout * L,
    صحیح kernel_type,
    صحیح wtype,
    uint32_t k,
    size_t mc,
    size_t nc,
    uint32_t گروہ_سائز,
    bool pipeline,
    uint32_t act_دھاگے,
    uint32_t aligned_tile_سائز,
    size_t bias_سائز
) {
    size_t off = 0;

    اگر (kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        ثابت size_t vec_dot_سائز     = k * حجم(uint16_t);
        ثابت size_t act_سر_stride   = mc * k;
        ثابت size_t وزن_area_سائز  = hex_align_اوپر(nc * vec_dot_سائز, HTP_MM_HMX_TILE_سائز);
        ثابت size_t activation_area_سائز = hex_align_اوپر(گروہ_سائز * act_سر_stride * حجم(uint16_t), HTP_MM_HMX_TILE_سائز);
        ثابت size_t output_area_سائز  = hex_align_اوپر(گروہ_سائز * mc * nc * حجم(uint16_t), HTP_MM_HMX_TILE_سائز);
        ثابت size_t scratch_area_سائز = hex_align_اوپر(nc * vec_dot_سائز, HTP_MM_HMX_TILE_سائز);
        ثابت size_t min_f32_سائز = hex_align_اوپر(act_دھاگے * HTP_MM_DMA_ACT_MULTIPLIER * k * حجم(اعشاری), 128);

        // Group A: Permanent activation tiles and scales
        size_t off_گروہ_a = 0;
        VTCM_LAYOUT_ALLOC(off_گروہ_a, off_act, activation_area_سائز);
        VTCM_LAYOUT_ALLOC(off_گروہ_a, off_scales, HTP_MM_HMX_TILE_سائز); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_a, off_bias, hex_align_اوپر(bias_سائز, HTP_MM_HMX_TILE_سائز), bias_سائز > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_گروہ_b = off_گروہ_a;
        VTCM_LAYOUT_ALLOC(off_گروہ_b, off_وزن[0], وزن_area_سائز);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_وزن[1], وزن_area_سائز, false);
        VTCM_LAYOUT_ALLOC(off_گروہ_b, off_dst[0], output_area_سائز);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_dst[1], output_area_سائز, false);
        VTCM_LAYOUT_ALLOC(off_گروہ_b, off_scratch[0], scratch_area_سائز);
        VTCM_LAYOUT_ALLOC(off_گروہ_b, off_scratch[1], scratch_area_سائز);

        ثابت size_t گروہ_b_سائز = off_گروہ_b - off_گروہ_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ثابت size_t max_f32_سائز = act_دھاگے * 64 * k * حجم(اعشاری);
        ثابت size_t act_f32_سائز = hex_align_اوپر(hex_smin(max_f32_سائز, hex_smax(min_f32_سائز, گروہ_b_سائز)), 128);
        size_t off_گروہ_c = off_گروہ_a;
        VTCM_LAYOUT_ALLOC(off_گروہ_c, off_act_f32, act_f32_سائز);

        ثابت size_t گروہ_c_سائز = off_گروہ_c - off_گروہ_a;

        L->وزن_area_بائٹیں = وزن_area_سائز;
        L->act_area_بائٹیں    = activation_area_سائز;
        L->act_f32_بائٹیں     = act_f32_سائز;
        L->output_area_بائٹیں = output_area_سائز;
        L->scratch_بائٹیں[0]  = scratch_area_سائز;
        L->scratch_بائٹیں[1]  = scratch_area_سائز;
        L->act_سر_stride   = act_سر_stride;
        L->bias_بائٹیں        = bias_سائز;

        off = off_گروہ_a + hex_smax(گروہ_b_سائز, گروہ_c_سائز);
    } ورنہ {
        // HTP_MM_KERNEL_HMX_2D
        ثابت bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
        ثابت size_t صف_stride = htp_tensor_get_صف_سائز(wtype, k);
        ثابت size_t vec_dot_سائز = k * حجم(uint16_t);
        ثابت uint32_t n_وزن_tiles = k / htp_tensor_get_tile_k(wtype);

        ثابت size_t min_f32_سائز = hex_align_اوپر(act_دھاگے * HTP_MM_DMA_ACT_MULTIPLIER * k * حجم(اعشاری), 128);
        ثابت size_t وزن_area_سائز = is_quant
            ? hex_align_اوپر((nc / 32) * n_وزن_tiles * aligned_tile_سائز, HTP_MM_HMX_TILE_سائز)
            : hex_align_اوپر(nc * صف_stride, HTP_MM_HMX_TILE_سائز);
        ثابت size_t act_area_سائز    = hex_align_اوپر(mc * vec_dot_سائز, HTP_MM_HMX_TILE_سائز);
        ثابت size_t output_area_سائز = hex_align_اوپر(mc * nc * حجم(__fp16), HTP_MM_HMX_TILE_سائز);

        ثابت size_t scratch0_سائز = hex_align_اوپر(nc * vec_dot_سائز, HTP_MM_HMX_TILE_سائز);
        ثابت size_t scratch1_سائز = pipeline ? scratch0_سائز : 0;

        // Group A:  Scales and activation tiles (must not overlap with Group B or C)
        size_t off_گروہ_a = 0;
        VTCM_LAYOUT_ALLOC(off_گروہ_a, off_scales, HTP_MM_HMX_TILE_سائز); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC(off_گروہ_a, off_act, act_area_سائز);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_a, off_bias, hex_align_اوپر(bias_سائز, HTP_MM_HMX_TILE_سائز), bias_سائز > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_گروہ_b = off_گروہ_a;
        VTCM_LAYOUT_ALLOC(off_گروہ_b, off_وزن[0], وزن_area_سائز);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_وزن[1], وزن_area_سائز, pipeline);
        VTCM_LAYOUT_ALLOC(off_گروہ_b, off_dst[0], output_area_سائز);
        VTCM_LAYOUT_ALLOC(off_گروہ_b, off_scratch[0], scratch0_سائز);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_scratch[1], scratch0_سائز, pipeline);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_dst[1], output_area_سائز, pipeline);

        ثابت size_t گروہ_b_سائز = off_گروہ_b - off_گروہ_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ثابت size_t max_f32_سائز = act_دھاگے * 64 * k * حجم(اعشاری);
        ثابت size_t act_f32_سائز = hex_align_اوپر(hex_smin(max_f32_سائز, hex_smax(min_f32_سائز, گروہ_b_سائز)), 128);
        size_t off_گروہ_c = off_گروہ_a;
        VTCM_LAYOUT_ALLOC(off_گروہ_c, off_act_f32, act_f32_سائز);

        ثابت size_t گروہ_c_سائز = off_گروہ_c - off_گروہ_a;

        L->وزن_area_بائٹیں = وزن_area_سائز;
        L->act_area_بائٹیں    = act_area_سائز;
        L->act_f32_بائٹیں     = act_f32_سائز;
        L->output_area_بائٹیں = output_area_سائز;
        L->scratch_بائٹیں[0]  = scratch0_سائز;
        L->scratch_بائٹیں[1]  = scratch1_سائز;
        L->act_سر_stride   = 0;
        L->bias_بائٹیں        = bias_سائز;

        off = off_گروہ_a + hex_smax(گروہ_b_سائز, گروہ_c_سائز);
    }

    L->کل_بائٹیں = off;
}

ساکن اندرونی خالی htp_mm_hvx_vtcm_layout_build(
    ساخت htp_mm_hvx_vtcm_layout * L,
    صحیح kernel_type,
    صحیح wtype,
    uint32_t ne10,       // k
    uint32_t act_nrows,  // m_total
    uint32_t n_دھاگے,
    size_t dst_صف_سائز,
    size_t src0_صف_سائز,
    size_t act_صف_سائز,
    size_t bias_صف_سائز,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_nx
) {
    (خالی)act_صف_سائز;
    size_t src0_sz    = 0;
    size_t act_sz     = 0;
    size_t bias_sz    = bias_صف_سائز > 0 ? htp_mm_باری_اوپر(bias_صف_سائز, 128) : 0;
    size_t dst_sz     = 0;
    size_t act_raw_sz = 0;

    اگر (is_fused_nx) {
        uint32_t aligned_tile_سائز = htp_tensor_get_tile_padded_سائز(wtype);
        uint32_t n_k_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        uint32_t tile_صف_سائز = n_k_tiles * aligned_tile_سائز;
        size_t وزن_sz_فی_thread = hex_باری_اوپر(n_prefetch * tile_صف_سائز, 128);

        size_t tiled_act_صف_سائز = htp_mm_tiled_act_صف_سائز(wtype, ne10);
        size_t q_act_sz = hex_باری_اوپر(tiled_act_صف_سائز * act_nrows, 128);
        size_t raw_صف_سائز = hex_باری_اوپر(ne10 * حجم(اعشاری), QK_Q8_0_TILED * حجم(اعشاری));

        src0_sz    = وزن_sz_فی_thread * n_دھاگے; // shared single-weight prefetch buffer
        act_sz     = q_act_sz;                         // quantized activation buffer
        bias_sz    = 0;
        dst_sz     = 0;
        act_raw_sz = hex_باری_اوپر(raw_صف_سائز * act_nrows, 128);
    } ورنہ اگر (is_matmul_id) {
        ثابت size_t act_صف_سائز_tiled = htp_mm_tiled_act_صف_سائز(wtype, ne10);
        ثابت uint32_t aligned_tile_سائز = htp_tensor_get_tile_padded_سائز(wtype);
        ثابت uint32_t n_k_tiles         = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        ثابت uint32_t tile_صف_سائز     = n_k_tiles * aligned_tile_سائز;
        size_t src0_sz_فی_thread        = htp_mm_باری_اوپر(n_prefetch * tile_صف_سائز, 256);

        act_sz = htp_mm_باری_اوپر(act_صف_سائز_tiled * act_nrows, 256);

        size_t raw_صف_سائز = hex_باری_اوپر(ne10 * حجم(اعشاری), QK_Q8_0_TILED * حجم(اعشاری));

        src0_sz    = src0_sz_فی_thread * n_دھاگے;
        dst_sz     = 0;
        bias_sz    = 0;
        act_raw_sz = hex_باری_اوپر(raw_صف_سائز * act_nrows, 128);
    } ورنہ {
        ثابت size_t src0_صف_سائز_padded = htp_mm_باری_اوپر(src0_صف_سائز, 128);
        ثابت size_t dst_nrows = (act_nrows > 1) ? 0 : 1;

        انتخاب (kernel_type) {
            حالت HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                size_t f16_act_صف_سائز = htp_mm_باری_اوپر(ne10 * 2, 128);
                act_sz     = htp_mm_باری_اوپر(f16_act_صف_سائز * act_nrows, 256);
                src0_sz    = htp_mm_باری_اوپر(n_prefetch * src0_صف_سائز_padded, 256) * n_دھاگے;
                dst_sz     = dst_nrows > 0 ? htp_mm_باری_اوپر(dst_صف_سائز, 128) * n_دھاگے : 0;
                act_raw_sz = hex_باری_اوپر(hex_باری_اوپر(ne10 * حجم(اعشاری), 128) * act_nrows, 128);
                توڑو;
            }
            حالت HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            حالت HTP_MM_KERNEL_HVX_BF16_F32_VTCM: {
                size_t f32_act_صف_سائز = htp_mm_باری_اوپر(ne10 * 4, 128);
                act_sz     = htp_mm_باری_اوپر(f32_act_صف_سائز * act_nrows, 256);
                src0_sz    = htp_mm_باری_اوپر(n_prefetch * src0_صف_سائز_padded, 256) * n_دھاگے;
                dst_sz     = dst_nrows > 0 ? htp_mm_باری_اوپر(dst_صف_سائز, 128) * n_دھاگے : 0;
                act_raw_sz = 0;
                توڑو;
            }
            حالت HTP_MM_KERNEL_HVX_QUANT_بلاک:
            حالت HTP_MM_KERNEL_HVX_QUANT_صف: {
                size_t q_act_صف_سائز = htp_mm_tiled_act_صف_سائز(wtype, ne10);
                uint32_t aligned_tile_سائز = htp_tensor_get_tile_padded_سائز(wtype);
                uint32_t n_وزن_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
                uint32_t tile_صف_سائز = n_وزن_tiles * aligned_tile_سائز;
                size_t tiled_vtcm_سائز = htp_mm_باری_اوپر(n_prefetch * tile_صف_سائز, 256);

                src0_sz = tiled_vtcm_سائز * n_دھاگے;
                act_sz  = htp_mm_باری_اوپر(q_act_صف_سائز * act_nrows, 256);

                size_t dst_قاش_فی_thread = (dst_nrows > 0 && act_nrows == 1) ? htp_mm_باری_اوپر((dst_صف_سائز + n_دھاگے - 1) / n_دھاگے, 128) : 0;
                dst_sz = dst_قاش_فی_thread * n_دھاگے;
                size_t raw_صف_سائز = hex_باری_اوپر(ne10 * حجم(اعشاری), QK_Q8_0_TILED * حجم(اعشاری));
                act_raw_sz = hex_باری_اوپر(raw_صف_سائز * act_nrows, 128);
                توڑو;
            }
            طےشدہ:
                توڑو;
        }
    }

    // Group A: Persistent buffers across chunk compute
    size_t off_گروہ_a = 0;
    VTCM_LAYOUT_ALLOC(off_گروہ_a, off_act, act_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_a, off_bias, bias_sz, bias_sz > 0);

    // Group B: Compute-only buffers (starts at off_group_a)
    size_t off_گروہ_b = off_گروہ_a;
    VTCM_LAYOUT_ALLOC(off_گروہ_b, off_src0, src0_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_dst, dst_sz, dst_sz > 0);
    ثابت size_t گروہ_b_سائز = off_گروہ_b - off_گروہ_a;

    // Group C: Raw activation staging buffer (overlaps Group B, starts at off_group_a)
    size_t off_گروہ_c = off_گروہ_a;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_c, off_act_raw, act_raw_sz, act_raw_sz > 0);
    ثابت size_t گروہ_c_سائز = off_گروہ_c - off_گروہ_a;

    L->src0_بائٹیں    = src0_sz;
    L->act_بائٹیں     = act_sz;
    L->bias_بائٹیں    = bias_sz;
    L->dst_بائٹیں     = dst_sz;
    L->act_raw_بائٹیں = act_raw_sz;
    L->کل_بائٹیں   = off_گروہ_a + hex_smax(گروہ_b_سائز, گروہ_c_سائز);
}

ساکن اندرونی bool htp_mm_hvx_solve_vtcm_params(
    صحیح kernel_type,
    صحیح wtype,
    uint32_t ne10,
    uint32_t act_nrows,
    uint32_t n_دھاگے,
    size_t dst_صف_سائز,
    size_t src0_صف_سائز,
    size_t act_صف_سائز,
    size_t bias_صف_سائز,
    uint32_t n_prefetch,
    size_t vtcm_budget,
    ساخت htp_mm_hvx_vtcm_layout * L_برآمد,
    uint32_t * m_chunk_برآمد
) {
    ساخت htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, act_nrows, n_دھاگے,
        dst_صف_سائز, src0_صف_سائز, act_صف_سائز, bias_صف_سائز, n_prefetch, false, false
    );

    اگر (L.کل_بائٹیں <= vtcm_budget) {
        *L_برآمد = L;
        *m_chunk_برآمد = act_nrows;
        واپس true;
    }

    ثابت size_t fixed_بائٹیں = L.src0_بائٹیں + L.bias_بائٹیں + L.dst_بائٹیں;
    اگر (vtcm_budget <= fixed_بائٹیں) {
        واپس false;
    }

    ثابت size_t avail_act = vtcm_budget - fixed_بائٹیں;
    size_t صف_سائز = 0;
    اگر (kernel_type == HTP_MM_KERNEL_HVX_QUANT_صف || kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک) {
        صف_سائز = htp_mm_tiled_act_صف_سائز(wtype, ne10);
    } ورنہ اگر (kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        صف_سائز = hex_باری_اوپر(ne10 * 2, 128);
    } ورنہ {
        صف_سائز = hex_باری_اوپر(ne10 * 4, 128);
    }
    اگر (صف_سائز == 0) {
        واپس false;
    }

    size_t eff_صف_سائز = صف_سائز;
    اگر (kernel_type == HTP_MM_KERNEL_HVX_QUANT_صف || kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک) {
        eff_صف_سائز += hex_باری_اوپر(ne10 * حجم(اعشاری), QK_Q8_0_TILED * حجم(اعشاری));
    }

    uint32_t m_chunk = (uint32_t) (avail_act / eff_صف_سائز);
    اگر (m_chunk > 1) {
        m_chunk &= ~1U;
    }
    اگر (m_chunk > act_nrows) {
        m_chunk = act_nrows;
    }
    اگر (m_chunk < 1) {
        واپس false;
    }

    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, m_chunk, n_دھاگے,
        dst_صف_سائز, src0_صف_سائز, act_صف_سائز, bias_صف_سائز, n_prefetch, false, false
    );

    جب (m_chunk > 2 && L.کل_بائٹیں > vtcm_budget) {
        m_chunk -= 2;
        htp_mm_hvx_vtcm_layout_build(
            &L, kernel_type, wtype, ne10, m_chunk, n_دھاگے,
            dst_صف_سائز, src0_صف_سائز, act_صف_سائز, bias_صف_سائز, n_prefetch, false, false
        );
    }

    اگر (L.کل_بائٹیں <= vtcm_budget) {
        *L_برآمد = L;
        *m_chunk_برآمد = m_chunk;
        واپس true;
    }

    واپس false;
}

ساکن اندرونی size_t htp_mm_hmx_get_2d_vtcm_سائز(
    صحیح wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_دھاگے, uint32_t aligned_tile_سائز, size_t bias_سائز
) {
    ساخت htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_دھاگے, aligned_tile_سائز, bias_سائز);
    واپس L.کل_بائٹیں;
}

ساکن اندرونی size_t htp_mm_hmx_get_batched_vtcm_سائز(
    صحیح wtype, uint32_t k, size_t mc, size_t nc, uint32_t گروہ_سائز, bool pipeline, uint32_t act_دھاگے, size_t bias_سائز) {
    (خالی)pipeline;
    ساخت htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, wtype, k, mc, nc, گروہ_سائز, false, act_دھاگے, 0, bias_سائز);
    واپس L.کل_بائٹیں;
}

ساکن اندرونی bool htp_mm_hmx_solve_batched_params(
    صحیح wtype,
    uint32_t k,
    uint32_t ne01_padded,
    uint32_t ne11,
    uint32_t گروہ_سائز,
    صحیح n_دھاگے,
    bool pipeline,
    size_t bias_سائز,
    size_t vtcm_budget,
    size_t * m_chunk_برآمد,
    size_t * n_chunk_برآمد,
    صحیح * act_دھاگے_برآمد,
    size_t * vtcm_سائز_برآمد
) {
    size_t بہترین_mblocks = SIZE_MAX;
    صحیح بہترین_act_دھاگے = 0;
    size_t بہترین_m_chunk = 0;
    size_t بہترین_n_chunk = 0;
    size_t بہترین_vtcm_سائز = 0;

    صحیح act_دھاگے = n_دھاگے;
    جب (act_دھاگے >= 1) {
        size_t گروہ_overhead = htp_mm_hmx_get_batched_overhead() + (bias_سائز > 0 ? hex_align_اوپر(bias_سائز, HTP_MM_HMX_TILE_سائز) : 0);
        size_t گروہ_سائز_فی_n, گروہ_سائز_فی_m, گروہ_سائز_فی_mn;
        htp_mm_hmx_get_batched_chunk_costs(k, گروہ_سائز, &گروہ_سائز_فی_n, &گروہ_سائز_فی_m, &گروہ_سائز_فی_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_سائز_candidate = 0;

        اگر (htp_mm_hmx_compute_chunks(vtcm_budget, گروہ_overhead, گروہ_سائز_فی_n, گروہ_سائز_فی_m, گروہ_سائز_فی_mn, hex_align_اوپر(ne11, 32), ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_لاگت_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_لاگت_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_سائز_candidate) == 0) {
            size_t exact_سائز = htp_mm_hmx_get_batched_vtcm_سائز(wtype, k, m_chunk_candidate, n_chunk_candidate, گروہ_سائز, pipeline, act_دھاگے, bias_سائز);
            اگر (exact_سائز <= vtcm_budget) {
                size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                اگر (mblocks < بہترین_mblocks || (mblocks == بہترین_mblocks && act_دھاگے > بہترین_act_دھاگے)) {
                    بہترین_mblocks = mblocks;
                    بہترین_act_دھاگے = act_دھاگے;
                    بہترین_m_chunk = m_chunk_candidate;
                    بہترین_n_chunk = n_chunk_candidate;
                    بہترین_vtcm_سائز = exact_سائز;
                }
            }
        }
        اگر (act_دھاگے == 1) {
            act_دھاگے = 0;
        } ورنہ {
            act_دھاگے /= 2;
        }
    }

    اگر (بہترین_act_دھاگے > 0) {
        *m_chunk_برآمد = بہترین_m_chunk;
        *n_chunk_برآمد = بہترین_n_chunk;
        *vtcm_سائز_برآمد = بہترین_vtcm_سائز;
        *act_دھاگے_برآمد = بہترین_act_دھاگے;
        واپس true;
    }
    واپس false;
}

ساکن اندرونی bool htp_mm_hmx_solve_2d_params(
    صحیح wtype,
    uint32_t k,
    uint32_t m_id_صفیں,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_لاگت,
    صحیح n_دھاگے,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_سائز,
    size_t bias_سائز,
    size_t vtcm_budget,
    size_t * m_chunk_برآمد,
    size_t * n_chunk_برآمد,
    صحیح * act_دھاگے_برآمد,
    size_t * vtcm_سائز_برآمد
) {
    size_t بہترین_mblocks = SIZE_MAX;
    صحیح بہترین_act_دھاگے = 0;
    size_t بہترین_m_chunk = 0;
    size_t بہترین_n_chunk = 0;
    size_t بہترین_vtcm_سائز = 0;

    ثابت صحیح m_for_chunks = is_matmul_id ? hex_align_اوپر(m_id_صفیں, 32) : ne11_padded;

    صحیح act_دھاگے = n_دھاگے;
    جب (act_دھاگے >= 1) {
        size_t simple_2d_overhead = htp_mm_hmx_get_2d_overhead(pipeline, is_matmul_id) + (bias_سائز > 0 ? hex_align_اوپر(bias_سائز, HTP_MM_HMX_TILE_سائز) : 0);
        size_t simple_2d_سائز_فی_n, simple_2d_سائز_فی_m, simple_2d_سائز_فی_mn;
        htp_mm_hmx_get_2d_chunk_costs(wtype, k, pipeline, aligned_tile_سائز, &simple_2d_سائز_فی_n, &simple_2d_سائز_فی_m, &simple_2d_سائز_فی_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_سائز_candidate = 0;

        اگر (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_سائز_فی_n, simple_2d_سائز_فی_m, simple_2d_سائز_فی_mn, m_for_chunks, ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_لاگت_W_DEQUANT, (size_t) m_for_لاگت * HTP_MM_HMX_لاگت_A_CONVERT,
                               &m_chunk_candidate, &n_chunk_candidate, &vtcm_سائز_candidate) == 0) {
            size_t exact_سائز = htp_mm_hmx_get_2d_vtcm_سائز(wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_دھاگے, aligned_tile_سائز, bias_سائز);
            اگر (exact_سائز <= vtcm_budget) {
                size_t mblocks = ((size_t) m_for_لاگت + m_chunk_candidate - 1) / m_chunk_candidate;
                اگر (mblocks < بہترین_mblocks || (mblocks == بہترین_mblocks && act_دھاگے > بہترین_act_دھاگے)) {
                    بہترین_mblocks = mblocks;
                    بہترین_act_دھاگے = act_دھاگے;
                    بہترین_m_chunk = m_chunk_candidate;
                    بہترین_n_chunk = n_chunk_candidate;
                    بہترین_vtcm_سائز = exact_سائز;
                }
            }
        }
        اگر (act_دھاگے == 1) {
            act_دھاگے = 0;
        } ورنہ {
            act_دھاگے /= 2;
        }
    }

    اگر (بہترین_act_دھاگے > 0) {
        *m_chunk_برآمد = بہترین_m_chunk;
        *n_chunk_برآمد = بہترین_n_chunk;
        *vtcm_سائز_برآمد = بہترین_vtcm_سائز;
        *act_دھاگے_برآمد = بہترین_act_دھاگے;
        واپس true;
    }
    واپس false;
}

#اگر_تعریف_شدہ __cplusplus
}
#اگر_ختم

#اگر_ختم // HTP_MATMUL_OPS_H
