#جيڪڏهن_وصف_نه_ٿيل HTP_MATMUL_OPS_H
#وصف_ڪريو HTP_MATMUL_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hex-fastdiv.h"
#ضمن "hex-common.h"
#ضمن "htp-vtcm.h"

#جيڪڏهن_وصف_ٿيل __cplusplus
خارجي "C" {
#جيڪڏهن_پورو

// --- HMX Tile Constraints ---
#وصف_ڪريو HTP_MM_HMX_TILE_N_کانن_ڳڻپ 32
#وصف_ڪريو HTP_MM_HMX_TILE_N_صفون 32
#وصف_ڪريو HTP_MM_HMX_TILE_سائيز   (32 * 32 * حجم(__fp16)) // 2048 bytes
#وصف_ڪريو HTP_MM_HMX_TILE_N_ELMS 1024
#وصف_ڪريو HTP_MM_HMX_MIN_NROWS   4

#وصف_ڪريو HTP_MM_وزن_TILE_K_Q2_0             64
#وصف_ڪريو HTP_MM_وزن_TILE_سائيز_Q2_0          HTP_TENSOR_TILE_سائيز_Q2_0
#وصف_ڪريو HTP_MM_وزن_ALIGNED_TILE_سائيز_Q2_0  640

// --- Activation Tiled Block Sizes (including padding) ---
#وصف_ڪريو HTP_MM_ACT_TILE_سائيز_Q8_0      1152
#وصف_ڪريو HTP_MM_ACT_TILE_سائيز_Q8_1      1280
#وصف_ڪريو HTP_MM_ACT_TILE_سائيز_F16       2048

#وصف_ڪريو HTP_MM_MAX_PREFETCH 16

// --- Solver Cost Model Penalty Weights (HMX-specific) ---
#وصف_ڪريو HTP_MM_HMX_لاڳت_W_DEQUANT 3 // cost penalty for quantized weight loading/dequantization
#وصف_ڪريو HTP_MM_HMX_لاڳت_A_CONVERT 2 // cost penalty for activation loading/conversion

// --- DMA Activation Transfer Configuration ---
#وصف_ڪريو HTP_MM_DMA_ACT_صفون_في_قدم 2
#وصف_ڪريو HTP_MM_DMA_ACT_MULTIPLIER    (2 * HTP_MM_DMA_ACT_صفون_في_قدم)

تعداد htp_mm_kernel_type {
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
    HTP_MM_KERNEL_HVX_QUANT_بلاڪ,    // parallel block-wise quantization
};

// Op-specific struct for precomputed matmul params
بنية htp_mm_kernel_params {
    uint8_t  kernel_type;        // enum htp_mm_kernel_type
    uint8_t  pipeline;           // 1 = pipelined execution, 0 = standard
    uint8_t  collapse;           // 1 = collapse outer dims into 2D, 0 = standard
    uint8_t  n_hmx;              // 1 = use HMX, 0 = use HVX

    uint8_t  n_ڌاڳا;          // Number of threads to spawn
    uint8_t  n_act_ڌاڳا;      // Number of threads for activation preparation
    uint8_t  n_prefetch;         // Prefetch lookahead buffers/rows in VTCM
    uint8_t  n_weights;          // Number of weights for fused NX

    int32_t  m_chunk;            // Row chunk size (M chunk)
    int32_t  n_chunk;            // Col chunk size (N chunk)
    int32_t  tile_سائيز;          // Weight tile size
    int32_t  aligned_tile_سائيز;  // Aligned weight tile size (padded to 128)
    int32_t  act_صف_سائيز;       // Row size for activation scratchpad
    int32_t  vtcm_سائيز;          // Total required scratchpad size in VTCM
    int32_t  vtcm_src0_سائيز;     // src0 scratchpad size in VTCM
    int32_t  vtcm_act_سائيز;      // activation scratchpad size in VTCM
    int32_t  vtcm_bias_سائيز;     // bias scratchpad size in VTCM (fused only)
    int32_t  vtcm_dst_سائيز;      // dst scratchpad size in VTCM

    // Precomputed division values
    بنية fastdiv_قدرون div_ne12_ne1;
    بنية fastdiv_قدرون div_ne1;
    بنية fastdiv_قدرون div_r2;
    بنية fastdiv_قدرون div_r3;
    بنية fastdiv_قدرون div_ne12;
    بنية fastdiv_قدرون div_n_act_ڌاڳا;
    بنية fastdiv_قدرون div_ne00_padded;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#جيڪڏهن_پورو

بنية mmid_صف_mapping {
    uint32_t i1;
    uint32_t i2;
};

// Search for optimal (mc, nc) chunk sizes within VTCM budget.
ساكن مضمن صحيح htp_mm_hmx_compute_chunks(size_t   vtcm_ڪل,
                              size_t   overhead,
                              size_t   في_n_لاڳت,
                              size_t   في_m_لاڳت,
                              size_t   في_mn_لاڳت,
                              size_t   m,
                              size_t   n,
                              size_t   m_بلاڪ_لاڳت,
                              size_t   n_بلاڪ_لاڳت,
                              size_t * m_chunk_ٻاهريون,
                              size_t * n_chunk_ٻاهريون,
                              size_t * ڪل_ٻاهريون) {
    إذا (m == 0 || n == 0) أرجع -1;
    إذا (vtcm_ڪل <= overhead) أرجع -1;
    إذا (في_n_لاڳت == 0 || في_m_لاڳت == 0 || في_mn_لاڳت == 0) أرجع -1;

    ثابت size_t usable = vtcm_ڪل - overhead;

    size_t بهترين_لاڳت = SIZE_MAX;
    size_t بهترين_پڇ_waste = SIZE_MAX;
    size_t بهترين_mn   = 0;
    size_t بهترين_m = 0, بهترين_n = 0;

    ثابت size_t max_nc_budget = (usable / في_n_لاڳت);
    ثابت size_t n_max = hex_align_هيٺ_ڏانهن(hex_smin((size_t)n, max_nc_budget), HTP_MM_HMX_TILE_N_کانن_ڳڻپ);
    لكل (size_t nc = n_max; nc >= HTP_MM_HMX_TILE_N_کانن_ڳڻپ; nc -= HTP_MM_HMX_TILE_N_کانن_ڳڻپ) {
        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        إذا (hex_mul_حد_کان_مٿي(nc, في_n_لاڳت, &n_fixed)) استمر;
        إذا (n_fixed >= usable) اذهب ايندڙ_nc;

        إذا (hex_mul_حد_کان_مٿي(nc, في_mn_لاڳت, &ncmn)) اذهب ايندڙ_nc;
        إذا (hex_جوڙيو_حد_کان_مٿي(في_m_لاڳت, ncmn, &mc_denom) || mc_denom == 0) اذهب ايندڙ_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hex_align_هيٺ_ڏانهن(mc, HTP_MM_HMX_TILE_N_صفون);
            mc = hex_smin(mc, m);

            إذا (mc == 0) {
                اذهب ايندڙ_nc;
            }

            size_t mblocks = ((size_t) m + mc - 1) / mc;
            size_t nblocks = ((size_t) n + nc - 1) / nc;
            size_t لاڳت    = mblocks * m_بلاڪ_لاڳت + nblocks * n_بلاڪ_لاڳت;
            size_t rem     = n % nc;
            size_t پڇ_waste = (rem == 0) ? 0 : (nc - rem);
            size_t mn      = mc * nc;
            إذا (لاڳت < بهترين_لاڳت ||
                (لاڳت == بهترين_لاڳت && پڇ_waste < بهترين_پڇ_waste) ||
                (لاڳت == بهترين_لاڳت && پڇ_waste == بهترين_پڇ_waste && mn > بهترين_mn)) {
                بهترين_لاڳت       = لاڳت;
                بهترين_پڇ_waste = پڇ_waste;
                بهترين_mn         = mn;
                بهترين_m          = mc;
                بهترين_n          = nc;
            }
        }

ايندڙ_nc:
        إذا (nc == HTP_MM_HMX_TILE_N_کانن_ڳڻپ) اكسر;  // avoid size_t underflow
    }

    إذا (بهترين_m == 0 || بهترين_n == 0) أرجع -1;

    // Compute exact total (with overflow checks)
    size_t t0 = 0, t1 = 0, t2 = 0, mn = 0, ڪل = 0;
    إذا (hex_mul_حد_کان_مٿي(بهترين_n, في_n_لاڳت, &t0)) أرجع -1;
    إذا (hex_mul_حد_کان_مٿي(بهترين_m, في_m_لاڳت, &t1)) أرجع -1;
    إذا (hex_mul_حد_کان_مٿي(بهترين_m, بهترين_n, &mn))     أرجع -1;
    إذا (hex_mul_حد_کان_مٿي(mn, في_mn_لاڳت, &t2))    أرجع -1;
    إذا (hex_جوڙيو_حد_کان_مٿي(t0, t1, &ڪل))          أرجع -1;
    إذا (hex_جوڙيو_حد_کان_مٿي(ڪل, t2, &ڪل))       أرجع -1;
    إذا (hex_جوڙيو_حد_کان_مٿي(ڪل, overhead, &ڪل)) أرجع -1;

    *m_chunk_ٻاهريون = بهترين_m;
    *n_chunk_ٻاهريون = بهترين_n;
    *ڪل_ٻاهريون   = ڪل;
    أرجع 0;
}

// weight types whose tiles carry a per-block offset (x = d * q + m): the activations need block sums (q8_1)
// (Q2_K: per-16 k sums, q8_1_s16)
ساكن مضمن bool htp_mm_وزن_has_وٿي(صحيح وزن_type) {
    أرجع وزن_type == HTP_TYPE_Q4_1 || وزن_type == HTP_TYPE_Q4_K || وزن_type == HTP_TYPE_Q5_K ||
           وزن_type == HTP_TYPE_Q2_K || وزن_type == HTP_TYPE_Q2_0;
}

// --- Activation/Row Size Helpers ---
ساكن مضمن size_t htp_mm_q8_0_tiled_صف_سائيز(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    أرجع nb_32 * HTP_MM_ACT_TILE_سائيز_Q8_0;
}

ساكن مضمن size_t htp_mm_q8_1_tiled_صف_سائيز(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    أرجع nb_32 * HTP_MM_ACT_TILE_سائيز_Q8_1;
}

ساكن مضمن size_t htp_mm_f16_tiled_صف_سائيز(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    أرجع nb_32 * HTP_MM_ACT_TILE_سائيز_F16;
}

ساكن مضمن size_t htp_mm_tiled_act_صف_سائيز(صحيح وزن_type, uint32_t ne) {
    إذا (وزن_type == HTP_TYPE_F16) {
        أرجع htp_mm_f16_tiled_صف_سائيز(ne);
    }
    أرجع htp_mm_وزن_has_وٿي(وزن_type) ? htp_mm_q8_1_tiled_صف_سائيز(ne) : htp_mm_q8_0_tiled_صف_سائيز(ne);
}

ساكن مضمن size_t htp_mm_ڦيرو_مٿي(size_t n, size_t m) {
    أرجع ((n + m - 1) / m) * m;
}

ساكن مضمن bool htp_mm_hmx_pipeline(uint32_t m) {
    أرجع m > 32;
}

ساكن مضمن فراغ htp_mm_hmx_get_2d_chunk_costs(
    صحيح wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_سائيز,
    size_t * سائيز_في_n_ٻاهريون, size_t * سائيز_في_m_ٻاهريون, size_t * سائيز_في_mn_ٻاهريون
) {
    ثابت bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
    ثابت size_t صف_stride = htp_tensor_get_صف_سائيز(wtype, k);
    ثابت size_t vec_dot_سائيز = k * حجم(uint16_t);
    ثابت uint32_t n_وزن_tiles = k / htp_tensor_get_tile_k(wtype);
    ثابت size_t qweight_صف_stride = is_quant ? (size_t)(n_وزن_tiles * aligned_tile_سائيز) / 32 : 0;

    *سائيز_في_n_ٻاهريون = (pipeline ? 2 : 1) * (is_quant ? qweight_صف_stride : صف_stride) +
                      (pipeline ? 2 * vec_dot_سائيز : vec_dot_سائيز);
    *سائيز_في_m_ٻاهريون = vec_dot_سائيز;
    *سائيز_في_mn_ٻاهريون = (pipeline ? 2 : 1) * حجم(uint16_t);
}

ساكن مضمن فراغ htp_mm_hmx_get_batched_chunk_costs(
    uint32_t k, uint32_t گروهه_سائيز,
    size_t * سائيز_في_n_ٻاهريون, size_t * سائيز_في_m_ٻاهريون, size_t * سائيز_في_mn_ٻاهريون
) {
    ثابت size_t vec_dot_سائيز = k * حجم(uint16_t);
    *سائيز_في_n_ٻاهريون = 3 * vec_dot_سائيز;
    *سائيز_في_m_ٻاهريون = گروهه_سائيز * vec_dot_سائيز;
    *سائيز_في_mn_ٻاهريون = حجم(uint16_t);
}

ساكن مضمن size_t htp_mm_hmx_get_2d_overhead(bool pipeline, bool is_matmul_id) {
    size_t انگ_regions = pipeline ? 7 : (is_matmul_id ? 4 : 5);
    أرجع انگ_regions * HTP_MM_HMX_TILE_سائيز + 256;
}

ساكن مضمن size_t htp_mm_hmx_get_batched_overhead(فراغ) {
    أرجع 5 * HTP_MM_HMX_TILE_سائيز + 256;
}

بنية htp_mm_hmx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_وزن[2];     // [1] is only used when pipelined
    size_t off_act;
    size_t off_act_f32;       // fp32 activation conversion scratch
    size_t off_dst[2];        // [1] is only used when pipelined
    size_t off_scratch[2];    // dequantization scratch pads
    size_t off_scales;        // HMX scales (256 bytes)
    size_t off_bias;          // bias in VTCM

    // Cached sizes of regions for HMX kernel use
    size_t وزن_area_بائيٽون;
    size_t act_area_بائيٽون;
    size_t act_f32_بائيٽون;
    size_t output_area_بائيٽون;
    size_t scratch_بائيٽون[2];
    size_t act_سر_stride;
    size_t bias_بائيٽون;

    size_t ڪل_بائيٽون;
};

بنية htp_mm_hvx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_act;           // vtcm_act (activation)
    size_t off_src0;          // vtcm_src0 (weight/Wk)
    size_t off_bias;          // vtcm_bias (bias / fused add only)
    size_t off_dst;           // vtcm_dst (output scratch)
    size_t off_act_raw;       // vtcm_act_raw (raw activation DMA staging)

    // Cached sizes
    size_t src0_بائيٽون;
    size_t act_بائيٽون;
    size_t bias_بائيٽون;
    size_t dst_بائيٽون;
    size_t act_raw_بائيٽون;

    size_t ڪل_بائيٽون;
};

ساكن مضمن فراغ htp_mm_hmx_vtcm_layout_build(
    بنية htp_mm_hmx_vtcm_layout * L,
    صحيح kernel_type,
    صحيح wtype,
    uint32_t k,
    size_t mc,
    size_t nc,
    uint32_t گروهه_سائيز,
    bool pipeline,
    uint32_t act_ڌاڳا,
    uint32_t aligned_tile_سائيز,
    size_t bias_سائيز
) {
    size_t off = 0;

    إذا (kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        ثابت size_t vec_dot_سائيز     = k * حجم(uint16_t);
        ثابت size_t act_سر_stride   = mc * k;
        ثابت size_t وزن_area_سائيز  = hex_align_مٿي(nc * vec_dot_سائيز, HTP_MM_HMX_TILE_سائيز);
        ثابت size_t activation_area_سائيز = hex_align_مٿي(گروهه_سائيز * act_سر_stride * حجم(uint16_t), HTP_MM_HMX_TILE_سائيز);
        ثابت size_t output_area_سائيز  = hex_align_مٿي(گروهه_سائيز * mc * nc * حجم(uint16_t), HTP_MM_HMX_TILE_سائيز);
        ثابت size_t scratch_area_سائيز = hex_align_مٿي(nc * vec_dot_سائيز, HTP_MM_HMX_TILE_سائيز);
        ثابت size_t min_f32_سائيز = hex_align_مٿي(act_ڌاڳا * HTP_MM_DMA_ACT_MULTIPLIER * k * حجم(عائم), 128);

        // Group A: Permanent activation tiles and scales
        size_t off_گروهه_a = 0;
        VTCM_LAYOUT_ALLOC(off_گروهه_a, off_act, activation_area_سائيز);
        VTCM_LAYOUT_ALLOC(off_گروهه_a, off_scales, HTP_MM_HMX_TILE_سائيز); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروهه_a, off_bias, hex_align_مٿي(bias_سائيز, HTP_MM_HMX_TILE_سائيز), bias_سائيز > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_گروهه_b = off_گروهه_a;
        VTCM_LAYOUT_ALLOC(off_گروهه_b, off_وزن[0], وزن_area_سائيز);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروهه_b, off_وزن[1], وزن_area_سائيز, false);
        VTCM_LAYOUT_ALLOC(off_گروهه_b, off_dst[0], output_area_سائيز);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروهه_b, off_dst[1], output_area_سائيز, false);
        VTCM_LAYOUT_ALLOC(off_گروهه_b, off_scratch[0], scratch_area_سائيز);
        VTCM_LAYOUT_ALLOC(off_گروهه_b, off_scratch[1], scratch_area_سائيز);

        ثابت size_t گروهه_b_سائيز = off_گروهه_b - off_گروهه_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ثابت size_t max_f32_سائيز = act_ڌاڳا * 64 * k * حجم(عائم);
        ثابت size_t act_f32_سائيز = hex_align_مٿي(hex_smin(max_f32_سائيز, hex_smax(min_f32_سائيز, گروهه_b_سائيز)), 128);
        size_t off_گروهه_c = off_گروهه_a;
        VTCM_LAYOUT_ALLOC(off_گروهه_c, off_act_f32, act_f32_سائيز);

        ثابت size_t گروهه_c_سائيز = off_گروهه_c - off_گروهه_a;

        L->وزن_area_بائيٽون = وزن_area_سائيز;
        L->act_area_بائيٽون    = activation_area_سائيز;
        L->act_f32_بائيٽون     = act_f32_سائيز;
        L->output_area_بائيٽون = output_area_سائيز;
        L->scratch_بائيٽون[0]  = scratch_area_سائيز;
        L->scratch_بائيٽون[1]  = scratch_area_سائيز;
        L->act_سر_stride   = act_سر_stride;
        L->bias_بائيٽون        = bias_سائيز;

        off = off_گروهه_a + hex_smax(گروهه_b_سائيز, گروهه_c_سائيز);
    } وإلا {
        // HTP_MM_KERNEL_HMX_2D
        ثابت bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
        ثابت size_t صف_stride = htp_tensor_get_صف_سائيز(wtype, k);
        ثابت size_t vec_dot_سائيز = k * حجم(uint16_t);
        ثابت uint32_t n_وزن_tiles = k / htp_tensor_get_tile_k(wtype);

        ثابت size_t min_f32_سائيز = hex_align_مٿي(act_ڌاڳا * HTP_MM_DMA_ACT_MULTIPLIER * k * حجم(عائم), 128);
        ثابت size_t وزن_area_سائيز = is_quant
            ? hex_align_مٿي((nc / 32) * n_وزن_tiles * aligned_tile_سائيز, HTP_MM_HMX_TILE_سائيز)
            : hex_align_مٿي(nc * صف_stride, HTP_MM_HMX_TILE_سائيز);
        ثابت size_t act_area_سائيز    = hex_align_مٿي(mc * vec_dot_سائيز, HTP_MM_HMX_TILE_سائيز);
        ثابت size_t output_area_سائيز = hex_align_مٿي(mc * nc * حجم(__fp16), HTP_MM_HMX_TILE_سائيز);

        ثابت size_t scratch0_سائيز = hex_align_مٿي(nc * vec_dot_سائيز, HTP_MM_HMX_TILE_سائيز);
        ثابت size_t scratch1_سائيز = pipeline ? scratch0_سائيز : 0;

        // Group A:  Scales and activation tiles (must not overlap with Group B or C)
        size_t off_گروهه_a = 0;
        VTCM_LAYOUT_ALLOC(off_گروهه_a, off_scales, HTP_MM_HMX_TILE_سائيز); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC(off_گروهه_a, off_act, act_area_سائيز);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروهه_a, off_bias, hex_align_مٿي(bias_سائيز, HTP_MM_HMX_TILE_سائيز), bias_سائيز > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_گروهه_b = off_گروهه_a;
        VTCM_LAYOUT_ALLOC(off_گروهه_b, off_وزن[0], وزن_area_سائيز);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروهه_b, off_وزن[1], وزن_area_سائيز, pipeline);
        VTCM_LAYOUT_ALLOC(off_گروهه_b, off_dst[0], output_area_سائيز);
        VTCM_LAYOUT_ALLOC(off_گروهه_b, off_scratch[0], scratch0_سائيز);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروهه_b, off_scratch[1], scratch0_سائيز, pipeline);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروهه_b, off_dst[1], output_area_سائيز, pipeline);

        ثابت size_t گروهه_b_سائيز = off_گروهه_b - off_گروهه_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ثابت size_t max_f32_سائيز = act_ڌاڳا * 64 * k * حجم(عائم);
        ثابت size_t act_f32_سائيز = hex_align_مٿي(hex_smin(max_f32_سائيز, hex_smax(min_f32_سائيز, گروهه_b_سائيز)), 128);
        size_t off_گروهه_c = off_گروهه_a;
        VTCM_LAYOUT_ALLOC(off_گروهه_c, off_act_f32, act_f32_سائيز);

        ثابت size_t گروهه_c_سائيز = off_گروهه_c - off_گروهه_a;

        L->وزن_area_بائيٽون = وزن_area_سائيز;
        L->act_area_بائيٽون    = act_area_سائيز;
        L->act_f32_بائيٽون     = act_f32_سائيز;
        L->output_area_بائيٽون = output_area_سائيز;
        L->scratch_بائيٽون[0]  = scratch0_سائيز;
        L->scratch_بائيٽون[1]  = scratch1_سائيز;
        L->act_سر_stride   = 0;
        L->bias_بائيٽون        = bias_سائيز;

        off = off_گروهه_a + hex_smax(گروهه_b_سائيز, گروهه_c_سائيز);
    }

    L->ڪل_بائيٽون = off;
}

ساكن مضمن فراغ htp_mm_hvx_vtcm_layout_build(
    بنية htp_mm_hvx_vtcm_layout * L,
    صحيح kernel_type,
    صحيح wtype,
    uint32_t ne10,       // k
    uint32_t act_nrows,  // m_total
    uint32_t n_ڌاڳا,
    size_t dst_صف_سائيز,
    size_t src0_صف_سائيز,
    size_t act_صف_سائيز,
    size_t bias_صف_سائيز,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_nx
) {
    (فراغ)act_صف_سائيز;
    size_t src0_sz    = 0;
    size_t act_sz     = 0;
    size_t bias_sz    = bias_صف_سائيز > 0 ? htp_mm_ڦيرو_مٿي(bias_صف_سائيز, 128) : 0;
    size_t dst_sz     = 0;
    size_t act_raw_sz = 0;

    إذا (is_fused_nx) {
        uint32_t aligned_tile_سائيز = htp_tensor_get_tile_padded_سائيز(wtype);
        uint32_t n_k_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        uint32_t tile_صف_سائيز = n_k_tiles * aligned_tile_سائيز;
        size_t وزن_sz_في_thread = hex_ڦيرو_مٿي(n_prefetch * tile_صف_سائيز, 128);

        size_t tiled_act_صف_سائيز = htp_mm_tiled_act_صف_سائيز(wtype, ne10);
        size_t q_act_sz = hex_ڦيرو_مٿي(tiled_act_صف_سائيز * act_nrows, 128);
        size_t raw_صف_سائيز = hex_ڦيرو_مٿي(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

        src0_sz    = وزن_sz_في_thread * n_ڌاڳا; // shared single-weight prefetch buffer
        act_sz     = q_act_sz;                         // quantized activation buffer
        bias_sz    = 0;
        dst_sz     = 0;
        act_raw_sz = hex_ڦيرو_مٿي(raw_صف_سائيز * act_nrows, 128);
    } وإلا إذا (is_matmul_id) {
        ثابت size_t act_صف_سائيز_tiled = htp_mm_tiled_act_صف_سائيز(wtype, ne10);
        ثابت uint32_t aligned_tile_سائيز = htp_tensor_get_tile_padded_سائيز(wtype);
        ثابت uint32_t n_k_tiles         = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        ثابت uint32_t tile_صف_سائيز     = n_k_tiles * aligned_tile_سائيز;
        size_t src0_sz_في_thread        = htp_mm_ڦيرو_مٿي(n_prefetch * tile_صف_سائيز, 256);

        act_sz = htp_mm_ڦيرو_مٿي(act_صف_سائيز_tiled * act_nrows, 256);

        size_t raw_صف_سائيز = hex_ڦيرو_مٿي(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

        src0_sz    = src0_sz_في_thread * n_ڌاڳا;
        dst_sz     = 0;
        bias_sz    = 0;
        act_raw_sz = hex_ڦيرو_مٿي(raw_صف_سائيز * act_nrows, 128);
    } وإلا {
        ثابت size_t src0_صف_سائيز_padded = htp_mm_ڦيرو_مٿي(src0_صف_سائيز, 128);
        ثابت size_t dst_nrows = (act_nrows > 1) ? 0 : 1;

        بدل (kernel_type) {
            حالة HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                size_t f16_act_صف_سائيز = htp_mm_ڦيرو_مٿي(ne10 * 2, 128);
                act_sz     = htp_mm_ڦيرو_مٿي(f16_act_صف_سائيز * act_nrows, 256);
                src0_sz    = htp_mm_ڦيرو_مٿي(n_prefetch * src0_صف_سائيز_padded, 256) * n_ڌاڳا;
                dst_sz     = dst_nrows > 0 ? htp_mm_ڦيرو_مٿي(dst_صف_سائيز, 128) * n_ڌاڳا : 0;
                act_raw_sz = hex_ڦيرو_مٿي(hex_ڦيرو_مٿي(ne10 * حجم(عائم), 128) * act_nrows, 128);
                اكسر;
            }
            حالة HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            حالة HTP_MM_KERNEL_HVX_BF16_F32_VTCM: {
                size_t f32_act_صف_سائيز = htp_mm_ڦيرو_مٿي(ne10 * 4, 128);
                act_sz     = htp_mm_ڦيرو_مٿي(f32_act_صف_سائيز * act_nrows, 256);
                src0_sz    = htp_mm_ڦيرو_مٿي(n_prefetch * src0_صف_سائيز_padded, 256) * n_ڌاڳا;
                dst_sz     = dst_nrows > 0 ? htp_mm_ڦيرو_مٿي(dst_صف_سائيز, 128) * n_ڌاڳا : 0;
                act_raw_sz = 0;
                اكسر;
            }
            حالة HTP_MM_KERNEL_HVX_QUANT_بلاڪ:
            حالة HTP_MM_KERNEL_HVX_QUANT_صف: {
                size_t q_act_صف_سائيز = htp_mm_tiled_act_صف_سائيز(wtype, ne10);
                uint32_t aligned_tile_سائيز = htp_tensor_get_tile_padded_سائيز(wtype);
                uint32_t n_وزن_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
                uint32_t tile_صف_سائيز = n_وزن_tiles * aligned_tile_سائيز;
                size_t tiled_vtcm_سائيز = htp_mm_ڦيرو_مٿي(n_prefetch * tile_صف_سائيز, 256);

                src0_sz = tiled_vtcm_سائيز * n_ڌاڳا;
                act_sz  = htp_mm_ڦيرو_مٿي(q_act_صف_سائيز * act_nrows, 256);

                size_t dst_ٽڪرو_في_thread = (dst_nrows > 0 && act_nrows == 1) ? htp_mm_ڦيرو_مٿي((dst_صف_سائيز + n_ڌاڳا - 1) / n_ڌاڳا, 128) : 0;
                dst_sz = dst_ٽڪرو_في_thread * n_ڌاڳا;
                size_t raw_صف_سائيز = hex_ڦيرو_مٿي(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));
                act_raw_sz = hex_ڦيرو_مٿي(raw_صف_سائيز * act_nrows, 128);
                اكسر;
            }
            افتراضي:
                اكسر;
        }
    }

    // Group A: Persistent buffers across chunk compute
    size_t off_گروهه_a = 0;
    VTCM_LAYOUT_ALLOC(off_گروهه_a, off_act, act_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروهه_a, off_bias, bias_sz, bias_sz > 0);

    // Group B: Compute-only buffers (starts at off_group_a)
    size_t off_گروهه_b = off_گروهه_a;
    VTCM_LAYOUT_ALLOC(off_گروهه_b, off_src0, src0_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروهه_b, off_dst, dst_sz, dst_sz > 0);
    ثابت size_t گروهه_b_سائيز = off_گروهه_b - off_گروهه_a;

    // Group C: Raw activation staging buffer (overlaps Group B, starts at off_group_a)
    size_t off_گروهه_c = off_گروهه_a;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروهه_c, off_act_raw, act_raw_sz, act_raw_sz > 0);
    ثابت size_t گروهه_c_سائيز = off_گروهه_c - off_گروهه_a;

    L->src0_بائيٽون    = src0_sz;
    L->act_بائيٽون     = act_sz;
    L->bias_بائيٽون    = bias_sz;
    L->dst_بائيٽون     = dst_sz;
    L->act_raw_بائيٽون = act_raw_sz;
    L->ڪل_بائيٽون   = off_گروهه_a + hex_smax(گروهه_b_سائيز, گروهه_c_سائيز);
}

ساكن مضمن bool htp_mm_hvx_solve_vtcm_params(
    صحيح kernel_type,
    صحيح wtype,
    uint32_t ne10,
    uint32_t act_nrows,
    uint32_t n_ڌاڳا,
    size_t dst_صف_سائيز,
    size_t src0_صف_سائيز,
    size_t act_صف_سائيز,
    size_t bias_صف_سائيز,
    uint32_t n_prefetch,
    size_t vtcm_budget,
    بنية htp_mm_hvx_vtcm_layout * L_ٻاهريون,
    uint32_t * m_chunk_ٻاهريون
) {
    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, act_nrows, n_ڌاڳا,
        dst_صف_سائيز, src0_صف_سائيز, act_صف_سائيز, bias_صف_سائيز, n_prefetch, false, false
    );

    إذا (L.ڪل_بائيٽون <= vtcm_budget) {
        *L_ٻاهريون = L;
        *m_chunk_ٻاهريون = act_nrows;
        أرجع true;
    }

    ثابت size_t fixed_بائيٽون = L.src0_بائيٽون + L.bias_بائيٽون + L.dst_بائيٽون;
    إذا (vtcm_budget <= fixed_بائيٽون) {
        أرجع false;
    }

    ثابت size_t avail_act = vtcm_budget - fixed_بائيٽون;
    size_t صف_سائيز = 0;
    إذا (kernel_type == HTP_MM_KERNEL_HVX_QUANT_صف || kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاڪ) {
        صف_سائيز = htp_mm_tiled_act_صف_سائيز(wtype, ne10);
    } وإلا إذا (kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        صف_سائيز = hex_ڦيرو_مٿي(ne10 * 2, 128);
    } وإلا {
        صف_سائيز = hex_ڦيرو_مٿي(ne10 * 4, 128);
    }
    إذا (صف_سائيز == 0) {
        أرجع false;
    }

    size_t eff_صف_سائيز = صف_سائيز;
    إذا (kernel_type == HTP_MM_KERNEL_HVX_QUANT_صف || kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاڪ) {
        eff_صف_سائيز += hex_ڦيرو_مٿي(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));
    }

    uint32_t m_chunk = (uint32_t) (avail_act / eff_صف_سائيز);
    إذا (m_chunk > 1) {
        m_chunk &= ~1U;
    }
    إذا (m_chunk > act_nrows) {
        m_chunk = act_nrows;
    }
    إذا (m_chunk < 1) {
        أرجع false;
    }

    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, m_chunk, n_ڌاڳا,
        dst_صف_سائيز, src0_صف_سائيز, act_صف_سائيز, bias_صف_سائيز, n_prefetch, false, false
    );

    طالما (m_chunk > 2 && L.ڪل_بائيٽون > vtcm_budget) {
        m_chunk -= 2;
        htp_mm_hvx_vtcm_layout_build(
            &L, kernel_type, wtype, ne10, m_chunk, n_ڌاڳا,
            dst_صف_سائيز, src0_صف_سائيز, act_صف_سائيز, bias_صف_سائيز, n_prefetch, false, false
        );
    }

    إذا (L.ڪل_بائيٽون <= vtcm_budget) {
        *L_ٻاهريون = L;
        *m_chunk_ٻاهريون = m_chunk;
        أرجع true;
    }

    أرجع false;
}

ساكن مضمن size_t htp_mm_hmx_get_2d_vtcm_سائيز(
    صحيح wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_ڌاڳا, uint32_t aligned_tile_سائيز, size_t bias_سائيز
) {
    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_ڌاڳا, aligned_tile_سائيز, bias_سائيز);
    أرجع L.ڪل_بائيٽون;
}

ساكن مضمن size_t htp_mm_hmx_get_batched_vtcm_سائيز(
    صحيح wtype, uint32_t k, size_t mc, size_t nc, uint32_t گروهه_سائيز, bool pipeline, uint32_t act_ڌاڳا, size_t bias_سائيز) {
    (فراغ)pipeline;
    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, wtype, k, mc, nc, گروهه_سائيز, false, act_ڌاڳا, 0, bias_سائيز);
    أرجع L.ڪل_بائيٽون;
}

ساكن مضمن bool htp_mm_hmx_solve_batched_params(
    صحيح wtype,
    uint32_t k,
    uint32_t ne01_padded,
    uint32_t ne11,
    uint32_t گروهه_سائيز,
    صحيح n_ڌاڳا,
    bool pipeline,
    size_t bias_سائيز,
    size_t vtcm_budget,
    size_t * m_chunk_ٻاهريون,
    size_t * n_chunk_ٻاهريون,
    صحيح * act_ڌاڳا_ٻاهريون,
    size_t * vtcm_سائيز_ٻاهريون
) {
    size_t بهترين_mblocks = SIZE_MAX;
    صحيح بهترين_act_ڌاڳا = 0;
    size_t بهترين_m_chunk = 0;
    size_t بهترين_n_chunk = 0;
    size_t بهترين_vtcm_سائيز = 0;

    صحيح act_ڌاڳا = n_ڌاڳا;
    طالما (act_ڌاڳا >= 1) {
        size_t گروهه_overhead = htp_mm_hmx_get_batched_overhead() + (bias_سائيز > 0 ? hex_align_مٿي(bias_سائيز, HTP_MM_HMX_TILE_سائيز) : 0);
        size_t گروهه_سائيز_في_n, گروهه_سائيز_في_m, گروهه_سائيز_في_mn;
        htp_mm_hmx_get_batched_chunk_costs(k, گروهه_سائيز, &گروهه_سائيز_في_n, &گروهه_سائيز_في_m, &گروهه_سائيز_في_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_سائيز_candidate = 0;

        إذا (htp_mm_hmx_compute_chunks(vtcm_budget, گروهه_overhead, گروهه_سائيز_في_n, گروهه_سائيز_في_m, گروهه_سائيز_في_mn, hex_align_مٿي(ne11, 32), ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_لاڳت_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_لاڳت_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_سائيز_candidate) == 0) {
            size_t exact_سائيز = htp_mm_hmx_get_batched_vtcm_سائيز(wtype, k, m_chunk_candidate, n_chunk_candidate, گروهه_سائيز, pipeline, act_ڌاڳا, bias_سائيز);
            إذا (exact_سائيز <= vtcm_budget) {
                size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                إذا (mblocks < بهترين_mblocks || (mblocks == بهترين_mblocks && act_ڌاڳا > بهترين_act_ڌاڳا)) {
                    بهترين_mblocks = mblocks;
                    بهترين_act_ڌاڳا = act_ڌاڳا;
                    بهترين_m_chunk = m_chunk_candidate;
                    بهترين_n_chunk = n_chunk_candidate;
                    بهترين_vtcm_سائيز = exact_سائيز;
                }
            }
        }
        إذا (act_ڌاڳا == 1) {
            act_ڌاڳا = 0;
        } وإلا {
            act_ڌاڳا /= 2;
        }
    }

    إذا (بهترين_act_ڌاڳا > 0) {
        *m_chunk_ٻاهريون = بهترين_m_chunk;
        *n_chunk_ٻاهريون = بهترين_n_chunk;
        *vtcm_سائيز_ٻاهريون = بهترين_vtcm_سائيز;
        *act_ڌاڳا_ٻاهريون = بهترين_act_ڌاڳا;
        أرجع true;
    }
    أرجع false;
}

ساكن مضمن bool htp_mm_hmx_solve_2d_params(
    صحيح wtype,
    uint32_t k,
    uint32_t m_id_صفون,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_لاڳت,
    صحيح n_ڌاڳا,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_سائيز,
    size_t bias_سائيز,
    size_t vtcm_budget,
    size_t * m_chunk_ٻاهريون,
    size_t * n_chunk_ٻاهريون,
    صحيح * act_ڌاڳا_ٻاهريون,
    size_t * vtcm_سائيز_ٻاهريون
) {
    size_t بهترين_mblocks = SIZE_MAX;
    صحيح بهترين_act_ڌاڳا = 0;
    size_t بهترين_m_chunk = 0;
    size_t بهترين_n_chunk = 0;
    size_t بهترين_vtcm_سائيز = 0;

    ثابت صحيح m_for_chunks = is_matmul_id ? hex_align_مٿي(m_id_صفون, 32) : ne11_padded;

    صحيح act_ڌاڳا = n_ڌاڳا;
    طالما (act_ڌاڳا >= 1) {
        size_t simple_2d_overhead = htp_mm_hmx_get_2d_overhead(pipeline, is_matmul_id) + (bias_سائيز > 0 ? hex_align_مٿي(bias_سائيز, HTP_MM_HMX_TILE_سائيز) : 0);
        size_t simple_2d_سائيز_في_n, simple_2d_سائيز_في_m, simple_2d_سائيز_في_mn;
        htp_mm_hmx_get_2d_chunk_costs(wtype, k, pipeline, aligned_tile_سائيز, &simple_2d_سائيز_في_n, &simple_2d_سائيز_في_m, &simple_2d_سائيز_في_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_سائيز_candidate = 0;

        إذا (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_سائيز_في_n, simple_2d_سائيز_في_m, simple_2d_سائيز_في_mn, m_for_chunks, ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_لاڳت_W_DEQUANT, (size_t) m_for_لاڳت * HTP_MM_HMX_لاڳت_A_CONVERT,
                               &m_chunk_candidate, &n_chunk_candidate, &vtcm_سائيز_candidate) == 0) {
            size_t exact_سائيز = htp_mm_hmx_get_2d_vtcm_سائيز(wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_ڌاڳا, aligned_tile_سائيز, bias_سائيز);
            إذا (exact_سائيز <= vtcm_budget) {
                size_t mblocks = ((size_t) m_for_لاڳت + m_chunk_candidate - 1) / m_chunk_candidate;
                إذا (mblocks < بهترين_mblocks || (mblocks == بهترين_mblocks && act_ڌاڳا > بهترين_act_ڌاڳا)) {
                    بهترين_mblocks = mblocks;
                    بهترين_act_ڌاڳا = act_ڌاڳا;
                    بهترين_m_chunk = m_chunk_candidate;
                    بهترين_n_chunk = n_chunk_candidate;
                    بهترين_vtcm_سائيز = exact_سائيز;
                }
            }
        }
        إذا (act_ڌاڳا == 1) {
            act_ڌاڳا = 0;
        } وإلا {
            act_ڌاڳا /= 2;
        }
    }

    إذا (بهترين_act_ڌاڳا > 0) {
        *m_chunk_ٻاهريون = بهترين_m_chunk;
        *n_chunk_ٻاهريون = بهترين_n_chunk;
        *vtcm_سائيز_ٻاهريون = بهترين_vtcm_سائيز;
        *act_ڌاڳا_ٻاهريون = بهترين_act_ڌاڳا;
        أرجع true;
    }
    أرجع false;
}

#جيڪڏهن_وصف_ٿيل __cplusplus
}
#جيڪڏهن_پورو

#جيڪڏهن_پورو // HTP_MATMUL_OPS_H
