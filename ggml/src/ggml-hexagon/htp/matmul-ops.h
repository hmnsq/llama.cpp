#اگر_تعریف_نشده HTP_MATMUL_OPS_H
#تعریف_کن HTP_MATMUL_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hex-fastdiv.h"
#ضمن "hex-common.h"
#ضمن "htp-vtcm.h"

#اگر_تعریف_شده __cplusplus
خارجي "C" {
#پایان_اگر

// --- HMX Tile Constraints ---
#تعریف_کن HTP_MM_HMX_TILE_N_تعداد_ستون 32
#تعریف_کن HTP_MM_HMX_TILE_N_ردیفها 32
#تعریف_کن HTP_MM_HMX_TILE_اندازه   (32 * 32 * حجم(__fp16)) // 2048 bytes
#تعریف_کن HTP_MM_HMX_TILE_N_ELMS 1024
#تعریف_کن HTP_MM_HMX_MIN_NROWS   4

#تعریف_کن HTP_MM_وزن_TILE_K_Q2_0             64
#تعریف_کن HTP_MM_وزن_TILE_اندازه_Q2_0          HTP_TENSOR_TILE_اندازه_Q2_0
#تعریف_کن HTP_MM_وزن_ALIGNED_TILE_اندازه_Q2_0  640

// --- Activation Tiled Block Sizes (including padding) ---
#تعریف_کن HTP_MM_ACT_TILE_اندازه_Q8_0      1152
#تعریف_کن HTP_MM_ACT_TILE_اندازه_Q8_1      1280
#تعریف_کن HTP_MM_ACT_TILE_اندازه_F16       2048

#تعریف_کن HTP_MM_MAX_PREFETCH 16

// --- Solver Cost Model Penalty Weights (HMX-specific) ---
#تعریف_کن HTP_MM_HMX_هزینه_W_DEQUANT 3 // cost penalty for quantized weight loading/dequantization
#تعریف_کن HTP_MM_HMX_هزینه_A_CONVERT 2 // cost penalty for activation loading/conversion

// --- DMA Activation Transfer Configuration ---
#تعریف_کن HTP_MM_DMA_ACT_ردیفها_در_هر_گام 2
#تعریف_کن HTP_MM_DMA_ACT_MULTIPLIER    (2 * HTP_MM_DMA_ACT_ردیفها_در_هر_گام)

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
    HTP_MM_KERNEL_HVX_QUANT_ردیف,      // standard row-wise parallel quantization
    HTP_MM_KERNEL_HVX_QUANT_بلوک,    // parallel block-wise quantization
};

// Op-specific struct for precomputed matmul params
بنية htp_mm_kernel_params {
    uint8_t  kernel_type;        // enum htp_mm_kernel_type
    uint8_t  pipeline;           // 1 = pipelined execution, 0 = standard
    uint8_t  collapse;           // 1 = collapse outer dims into 2D, 0 = standard
    uint8_t  n_hmx;              // 1 = use HMX, 0 = use HVX

    uint8_t  n_نخها;          // Number of threads to spawn
    uint8_t  n_act_نخها;      // Number of threads for activation preparation
    uint8_t  n_prefetch;         // Prefetch lookahead buffers/rows in VTCM
    uint8_t  n_weights;          // Number of weights for fused NX

    int32_t  m_chunk;            // Row chunk size (M chunk)
    int32_t  n_chunk;            // Col chunk size (N chunk)
    int32_t  tile_اندازه;          // Weight tile size
    int32_t  aligned_tile_اندازه;  // Aligned weight tile size (padded to 128)
    int32_t  act_ردیف_اندازه;       // Row size for activation scratchpad
    int32_t  vtcm_اندازه;          // Total required scratchpad size in VTCM
    int32_t  vtcm_src0_اندازه;     // src0 scratchpad size in VTCM
    int32_t  vtcm_act_اندازه;      // activation scratchpad size in VTCM
    int32_t  vtcm_bias_اندازه;     // bias scratchpad size in VTCM (fused only)
    int32_t  vtcm_dst_اندازه;      // dst scratchpad size in VTCM

    // Precomputed division values
    بنية fastdiv_مقادیر div_ne12_ne1;
    بنية fastdiv_مقادیر div_ne1;
    بنية fastdiv_مقادیر div_r2;
    بنية fastdiv_مقادیر div_r3;
    بنية fastdiv_مقادیر div_ne12;
    بنية fastdiv_مقادیر div_n_act_نخها;
    بنية fastdiv_مقادیر div_ne00_padded;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#پایان_اگر

بنية mmid_ردیف_mapping {
    uint32_t i1;
    uint32_t i2;
};

// Search for optimal (mc, nc) chunk sizes within VTCM budget.
ساكن مضمن صحيح htp_mm_hmx_compute_chunks(size_t   vtcm_مجموع,
                              size_t   overhead,
                              size_t   در_هر_n_هزینه,
                              size_t   در_هر_m_هزینه,
                              size_t   در_هر_mn_هزینه,
                              size_t   m,
                              size_t   n,
                              size_t   m_بلوک_هزینه,
                              size_t   n_بلوک_هزینه,
                              size_t * m_chunk_بیرون,
                              size_t * n_chunk_بیرون,
                              size_t * مجموع_بیرون) {
    إذا (m == 0 || n == 0) أرجع -1;
    إذا (vtcm_مجموع <= overhead) أرجع -1;
    إذا (در_هر_n_هزینه == 0 || در_هر_m_هزینه == 0 || در_هر_mn_هزینه == 0) أرجع -1;

    ثابت size_t usable = vtcm_مجموع - overhead;

    size_t بهترین_هزینه = SIZE_MAX;
    size_t بهترین_دم_waste = SIZE_MAX;
    size_t بهترین_mn   = 0;
    size_t بهترین_m = 0, بهترین_n = 0;

    ثابت size_t max_nc_budget = (usable / در_هر_n_هزینه);
    ثابت size_t n_max = hex_align_پایین(hex_smin((size_t)n, max_nc_budget), HTP_MM_HMX_TILE_N_تعداد_ستون);
    لكل (size_t nc = n_max; nc >= HTP_MM_HMX_TILE_N_تعداد_ستون; nc -= HTP_MM_HMX_TILE_N_تعداد_ستون) {
        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        إذا (hex_mul_سرریز(nc, در_هر_n_هزینه, &n_fixed)) استمر;
        إذا (n_fixed >= usable) اذهب بعدی_nc;

        إذا (hex_mul_سرریز(nc, در_هر_mn_هزینه, &ncmn)) اذهب بعدی_nc;
        إذا (hex_جمع_کن_سرریز(در_هر_m_هزینه, ncmn, &mc_denom) || mc_denom == 0) اذهب بعدی_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hex_align_پایین(mc, HTP_MM_HMX_TILE_N_ردیفها);
            mc = hex_smin(mc, m);

            إذا (mc == 0) {
                اذهب بعدی_nc;
            }

            size_t mblocks = ((size_t) m + mc - 1) / mc;
            size_t nblocks = ((size_t) n + nc - 1) / nc;
            size_t هزینه    = mblocks * m_بلوک_هزینه + nblocks * n_بلوک_هزینه;
            size_t rem     = n % nc;
            size_t دم_waste = (rem == 0) ? 0 : (nc - rem);
            size_t mn      = mc * nc;
            إذا (هزینه < بهترین_هزینه ||
                (هزینه == بهترین_هزینه && دم_waste < بهترین_دم_waste) ||
                (هزینه == بهترین_هزینه && دم_waste == بهترین_دم_waste && mn > بهترین_mn)) {
                بهترین_هزینه       = هزینه;
                بهترین_دم_waste = دم_waste;
                بهترین_mn         = mn;
                بهترین_m          = mc;
                بهترین_n          = nc;
            }
        }

بعدی_nc:
        إذا (nc == HTP_MM_HMX_TILE_N_تعداد_ستون) اكسر;  // avoid size_t underflow
    }

    إذا (بهترین_m == 0 || بهترین_n == 0) أرجع -1;

    // Compute exact total (with overflow checks)
    size_t t0 = 0, t1 = 0, t2 = 0, mn = 0, مجموع = 0;
    إذا (hex_mul_سرریز(بهترین_n, در_هر_n_هزینه, &t0)) أرجع -1;
    إذا (hex_mul_سرریز(بهترین_m, در_هر_m_هزینه, &t1)) أرجع -1;
    إذا (hex_mul_سرریز(بهترین_m, بهترین_n, &mn))     أرجع -1;
    إذا (hex_mul_سرریز(mn, در_هر_mn_هزینه, &t2))    أرجع -1;
    إذا (hex_جمع_کن_سرریز(t0, t1, &مجموع))          أرجع -1;
    إذا (hex_جمع_کن_سرریز(مجموع, t2, &مجموع))       أرجع -1;
    إذا (hex_جمع_کن_سرریز(مجموع, overhead, &مجموع)) أرجع -1;

    *m_chunk_بیرون = بهترین_m;
    *n_chunk_بیرون = بهترین_n;
    *مجموع_بیرون   = مجموع;
    أرجع 0;
}

// weight types whose tiles carry a per-block offset (x = d * q + m): the activations need block sums (q8_1)
// (Q2_K: per-16 k sums, q8_1_s16)
ساكن مضمن bool htp_mm_وزن_has_جابه_جایی(صحيح وزن_type) {
    أرجع وزن_type == HTP_TYPE_Q4_1 || وزن_type == HTP_TYPE_Q4_K || وزن_type == HTP_TYPE_Q5_K ||
           وزن_type == HTP_TYPE_Q2_K || وزن_type == HTP_TYPE_Q2_0;
}

// --- Activation/Row Size Helpers ---
ساكن مضمن size_t htp_mm_q8_0_tiled_ردیف_اندازه(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    أرجع nb_32 * HTP_MM_ACT_TILE_اندازه_Q8_0;
}

ساكن مضمن size_t htp_mm_q8_1_tiled_ردیف_اندازه(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    أرجع nb_32 * HTP_MM_ACT_TILE_اندازه_Q8_1;
}

ساكن مضمن size_t htp_mm_f16_tiled_ردیف_اندازه(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    أرجع nb_32 * HTP_MM_ACT_TILE_اندازه_F16;
}

ساكن مضمن size_t htp_mm_tiled_act_ردیف_اندازه(صحيح وزن_type, uint32_t ne) {
    إذا (وزن_type == HTP_TYPE_F16) {
        أرجع htp_mm_f16_tiled_ردیف_اندازه(ne);
    }
    أرجع htp_mm_وزن_has_جابه_جایی(وزن_type) ? htp_mm_q8_1_tiled_ردیف_اندازه(ne) : htp_mm_q8_0_tiled_ردیف_اندازه(ne);
}

ساكن مضمن size_t htp_mm_دور_به_بالا(size_t n, size_t m) {
    أرجع ((n + m - 1) / m) * m;
}

ساكن مضمن bool htp_mm_hmx_pipeline(uint32_t m) {
    أرجع m > 32;
}

ساكن مضمن فراغ htp_mm_hmx_get_2d_chunk_costs(
    صحيح wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_اندازه,
    size_t * اندازه_در_هر_n_بیرون, size_t * اندازه_در_هر_m_بیرون, size_t * اندازه_در_هر_mn_بیرون
) {
    ثابت bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
    ثابت size_t ردیف_stride = htp_tensor_get_ردیف_اندازه(wtype, k);
    ثابت size_t vec_dot_اندازه = k * حجم(uint16_t);
    ثابت uint32_t n_وزن_tiles = k / htp_tensor_get_tile_k(wtype);
    ثابت size_t qweight_ردیف_stride = is_quant ? (size_t)(n_وزن_tiles * aligned_tile_اندازه) / 32 : 0;

    *اندازه_در_هر_n_بیرون = (pipeline ? 2 : 1) * (is_quant ? qweight_ردیف_stride : ردیف_stride) +
                      (pipeline ? 2 * vec_dot_اندازه : vec_dot_اندازه);
    *اندازه_در_هر_m_بیرون = vec_dot_اندازه;
    *اندازه_در_هر_mn_بیرون = (pipeline ? 2 : 1) * حجم(uint16_t);
}

ساكن مضمن فراغ htp_mm_hmx_get_batched_chunk_costs(
    uint32_t k, uint32_t گروه_اندازه,
    size_t * اندازه_در_هر_n_بیرون, size_t * اندازه_در_هر_m_بیرون, size_t * اندازه_در_هر_mn_بیرون
) {
    ثابت size_t vec_dot_اندازه = k * حجم(uint16_t);
    *اندازه_در_هر_n_بیرون = 3 * vec_dot_اندازه;
    *اندازه_در_هر_m_بیرون = گروه_اندازه * vec_dot_اندازه;
    *اندازه_در_هر_mn_بیرون = حجم(uint16_t);
}

ساكن مضمن size_t htp_mm_hmx_get_2d_overhead(bool pipeline, bool is_matmul_id) {
    size_t رقم_regions = pipeline ? 7 : (is_matmul_id ? 4 : 5);
    أرجع رقم_regions * HTP_MM_HMX_TILE_اندازه + 256;
}

ساكن مضمن size_t htp_mm_hmx_get_batched_overhead(فراغ) {
    أرجع 5 * HTP_MM_HMX_TILE_اندازه + 256;
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
    size_t وزن_area_بایتها;
    size_t act_area_بایتها;
    size_t act_f32_بایتها;
    size_t output_area_بایتها;
    size_t scratch_بایتها[2];
    size_t act_سر_stride;
    size_t bias_بایتها;

    size_t مجموع_بایتها;
};

بنية htp_mm_hvx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_act;           // vtcm_act (activation)
    size_t off_src0;          // vtcm_src0 (weight/Wk)
    size_t off_bias;          // vtcm_bias (bias / fused add only)
    size_t off_dst;           // vtcm_dst (output scratch)
    size_t off_act_raw;       // vtcm_act_raw (raw activation DMA staging)

    // Cached sizes
    size_t src0_بایتها;
    size_t act_بایتها;
    size_t bias_بایتها;
    size_t dst_بایتها;
    size_t act_raw_بایتها;

    size_t مجموع_بایتها;
};

ساكن مضمن فراغ htp_mm_hmx_vtcm_layout_build(
    بنية htp_mm_hmx_vtcm_layout * L,
    صحيح kernel_type,
    صحيح wtype,
    uint32_t k,
    size_t mc,
    size_t nc,
    uint32_t گروه_اندازه,
    bool pipeline,
    uint32_t act_نخها,
    uint32_t aligned_tile_اندازه,
    size_t bias_اندازه
) {
    size_t off = 0;

    إذا (kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        ثابت size_t vec_dot_اندازه     = k * حجم(uint16_t);
        ثابت size_t act_سر_stride   = mc * k;
        ثابت size_t وزن_area_اندازه  = hex_align_به_بالا(nc * vec_dot_اندازه, HTP_MM_HMX_TILE_اندازه);
        ثابت size_t activation_area_اندازه = hex_align_به_بالا(گروه_اندازه * act_سر_stride * حجم(uint16_t), HTP_MM_HMX_TILE_اندازه);
        ثابت size_t output_area_اندازه  = hex_align_به_بالا(گروه_اندازه * mc * nc * حجم(uint16_t), HTP_MM_HMX_TILE_اندازه);
        ثابت size_t scratch_area_اندازه = hex_align_به_بالا(nc * vec_dot_اندازه, HTP_MM_HMX_TILE_اندازه);
        ثابت size_t min_f32_اندازه = hex_align_به_بالا(act_نخها * HTP_MM_DMA_ACT_MULTIPLIER * k * حجم(عائم), 128);

        // Group A: Permanent activation tiles and scales
        size_t off_گروه_a = 0;
        VTCM_LAYOUT_ALLOC(off_گروه_a, off_act, activation_area_اندازه);
        VTCM_LAYOUT_ALLOC(off_گروه_a, off_scales, HTP_MM_HMX_TILE_اندازه); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروه_a, off_bias, hex_align_به_بالا(bias_اندازه, HTP_MM_HMX_TILE_اندازه), bias_اندازه > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_گروه_b = off_گروه_a;
        VTCM_LAYOUT_ALLOC(off_گروه_b, off_وزن[0], وزن_area_اندازه);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروه_b, off_وزن[1], وزن_area_اندازه, false);
        VTCM_LAYOUT_ALLOC(off_گروه_b, off_dst[0], output_area_اندازه);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروه_b, off_dst[1], output_area_اندازه, false);
        VTCM_LAYOUT_ALLOC(off_گروه_b, off_scratch[0], scratch_area_اندازه);
        VTCM_LAYOUT_ALLOC(off_گروه_b, off_scratch[1], scratch_area_اندازه);

        ثابت size_t گروه_b_اندازه = off_گروه_b - off_گروه_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ثابت size_t max_f32_اندازه = act_نخها * 64 * k * حجم(عائم);
        ثابت size_t act_f32_اندازه = hex_align_به_بالا(hex_smin(max_f32_اندازه, hex_smax(min_f32_اندازه, گروه_b_اندازه)), 128);
        size_t off_گروه_c = off_گروه_a;
        VTCM_LAYOUT_ALLOC(off_گروه_c, off_act_f32, act_f32_اندازه);

        ثابت size_t گروه_c_اندازه = off_گروه_c - off_گروه_a;

        L->وزن_area_بایتها = وزن_area_اندازه;
        L->act_area_بایتها    = activation_area_اندازه;
        L->act_f32_بایتها     = act_f32_اندازه;
        L->output_area_بایتها = output_area_اندازه;
        L->scratch_بایتها[0]  = scratch_area_اندازه;
        L->scratch_بایتها[1]  = scratch_area_اندازه;
        L->act_سر_stride   = act_سر_stride;
        L->bias_بایتها        = bias_اندازه;

        off = off_گروه_a + hex_smax(گروه_b_اندازه, گروه_c_اندازه);
    } وإلا {
        // HTP_MM_KERNEL_HMX_2D
        ثابت bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
        ثابت size_t ردیف_stride = htp_tensor_get_ردیف_اندازه(wtype, k);
        ثابت size_t vec_dot_اندازه = k * حجم(uint16_t);
        ثابت uint32_t n_وزن_tiles = k / htp_tensor_get_tile_k(wtype);

        ثابت size_t min_f32_اندازه = hex_align_به_بالا(act_نخها * HTP_MM_DMA_ACT_MULTIPLIER * k * حجم(عائم), 128);
        ثابت size_t وزن_area_اندازه = is_quant
            ? hex_align_به_بالا((nc / 32) * n_وزن_tiles * aligned_tile_اندازه, HTP_MM_HMX_TILE_اندازه)
            : hex_align_به_بالا(nc * ردیف_stride, HTP_MM_HMX_TILE_اندازه);
        ثابت size_t act_area_اندازه    = hex_align_به_بالا(mc * vec_dot_اندازه, HTP_MM_HMX_TILE_اندازه);
        ثابت size_t output_area_اندازه = hex_align_به_بالا(mc * nc * حجم(__fp16), HTP_MM_HMX_TILE_اندازه);

        ثابت size_t scratch0_اندازه = hex_align_به_بالا(nc * vec_dot_اندازه, HTP_MM_HMX_TILE_اندازه);
        ثابت size_t scratch1_اندازه = pipeline ? scratch0_اندازه : 0;

        // Group A:  Scales and activation tiles (must not overlap with Group B or C)
        size_t off_گروه_a = 0;
        VTCM_LAYOUT_ALLOC(off_گروه_a, off_scales, HTP_MM_HMX_TILE_اندازه); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC(off_گروه_a, off_act, act_area_اندازه);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروه_a, off_bias, hex_align_به_بالا(bias_اندازه, HTP_MM_HMX_TILE_اندازه), bias_اندازه > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_گروه_b = off_گروه_a;
        VTCM_LAYOUT_ALLOC(off_گروه_b, off_وزن[0], وزن_area_اندازه);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروه_b, off_وزن[1], وزن_area_اندازه, pipeline);
        VTCM_LAYOUT_ALLOC(off_گروه_b, off_dst[0], output_area_اندازه);
        VTCM_LAYOUT_ALLOC(off_گروه_b, off_scratch[0], scratch0_اندازه);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروه_b, off_scratch[1], scratch0_اندازه, pipeline);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروه_b, off_dst[1], output_area_اندازه, pipeline);

        ثابت size_t گروه_b_اندازه = off_گروه_b - off_گروه_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ثابت size_t max_f32_اندازه = act_نخها * 64 * k * حجم(عائم);
        ثابت size_t act_f32_اندازه = hex_align_به_بالا(hex_smin(max_f32_اندازه, hex_smax(min_f32_اندازه, گروه_b_اندازه)), 128);
        size_t off_گروه_c = off_گروه_a;
        VTCM_LAYOUT_ALLOC(off_گروه_c, off_act_f32, act_f32_اندازه);

        ثابت size_t گروه_c_اندازه = off_گروه_c - off_گروه_a;

        L->وزن_area_بایتها = وزن_area_اندازه;
        L->act_area_بایتها    = act_area_اندازه;
        L->act_f32_بایتها     = act_f32_اندازه;
        L->output_area_بایتها = output_area_اندازه;
        L->scratch_بایتها[0]  = scratch0_اندازه;
        L->scratch_بایتها[1]  = scratch1_اندازه;
        L->act_سر_stride   = 0;
        L->bias_بایتها        = bias_اندازه;

        off = off_گروه_a + hex_smax(گروه_b_اندازه, گروه_c_اندازه);
    }

    L->مجموع_بایتها = off;
}

ساكن مضمن فراغ htp_mm_hvx_vtcm_layout_build(
    بنية htp_mm_hvx_vtcm_layout * L,
    صحيح kernel_type,
    صحيح wtype,
    uint32_t ne10,       // k
    uint32_t act_nrows,  // m_total
    uint32_t n_نخها,
    size_t dst_ردیف_اندازه,
    size_t src0_ردیف_اندازه,
    size_t act_ردیف_اندازه,
    size_t bias_ردیف_اندازه,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_nx
) {
    (فراغ)act_ردیف_اندازه;
    size_t src0_sz    = 0;
    size_t act_sz     = 0;
    size_t bias_sz    = bias_ردیف_اندازه > 0 ? htp_mm_دور_به_بالا(bias_ردیف_اندازه, 128) : 0;
    size_t dst_sz     = 0;
    size_t act_raw_sz = 0;

    إذا (is_fused_nx) {
        uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(wtype);
        uint32_t n_k_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        uint32_t tile_ردیف_اندازه = n_k_tiles * aligned_tile_اندازه;
        size_t وزن_sz_در_هر_thread = hex_دور_به_بالا(n_prefetch * tile_ردیف_اندازه, 128);

        size_t tiled_act_ردیف_اندازه = htp_mm_tiled_act_ردیف_اندازه(wtype, ne10);
        size_t q_act_sz = hex_دور_به_بالا(tiled_act_ردیف_اندازه * act_nrows, 128);
        size_t raw_ردیف_اندازه = hex_دور_به_بالا(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

        src0_sz    = وزن_sz_در_هر_thread * n_نخها; // shared single-weight prefetch buffer
        act_sz     = q_act_sz;                         // quantized activation buffer
        bias_sz    = 0;
        dst_sz     = 0;
        act_raw_sz = hex_دور_به_بالا(raw_ردیف_اندازه * act_nrows, 128);
    } وإلا إذا (is_matmul_id) {
        ثابت size_t act_ردیف_اندازه_tiled = htp_mm_tiled_act_ردیف_اندازه(wtype, ne10);
        ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(wtype);
        ثابت uint32_t n_k_tiles         = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        ثابت uint32_t tile_ردیف_اندازه     = n_k_tiles * aligned_tile_اندازه;
        size_t src0_sz_در_هر_thread        = htp_mm_دور_به_بالا(n_prefetch * tile_ردیف_اندازه, 256);

        act_sz = htp_mm_دور_به_بالا(act_ردیف_اندازه_tiled * act_nrows, 256);

        size_t raw_ردیف_اندازه = hex_دور_به_بالا(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

        src0_sz    = src0_sz_در_هر_thread * n_نخها;
        dst_sz     = 0;
        bias_sz    = 0;
        act_raw_sz = hex_دور_به_بالا(raw_ردیف_اندازه * act_nrows, 128);
    } وإلا {
        ثابت size_t src0_ردیف_اندازه_padded = htp_mm_دور_به_بالا(src0_ردیف_اندازه, 128);
        ثابت size_t dst_nrows = (act_nrows > 1) ? 0 : 1;

        بدل (kernel_type) {
            حالة HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                size_t f16_act_ردیف_اندازه = htp_mm_دور_به_بالا(ne10 * 2, 128);
                act_sz     = htp_mm_دور_به_بالا(f16_act_ردیف_اندازه * act_nrows, 256);
                src0_sz    = htp_mm_دور_به_بالا(n_prefetch * src0_ردیف_اندازه_padded, 256) * n_نخها;
                dst_sz     = dst_nrows > 0 ? htp_mm_دور_به_بالا(dst_ردیف_اندازه, 128) * n_نخها : 0;
                act_raw_sz = hex_دور_به_بالا(hex_دور_به_بالا(ne10 * حجم(عائم), 128) * act_nrows, 128);
                اكسر;
            }
            حالة HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            حالة HTP_MM_KERNEL_HVX_BF16_F32_VTCM: {
                size_t f32_act_ردیف_اندازه = htp_mm_دور_به_بالا(ne10 * 4, 128);
                act_sz     = htp_mm_دور_به_بالا(f32_act_ردیف_اندازه * act_nrows, 256);
                src0_sz    = htp_mm_دور_به_بالا(n_prefetch * src0_ردیف_اندازه_padded, 256) * n_نخها;
                dst_sz     = dst_nrows > 0 ? htp_mm_دور_به_بالا(dst_ردیف_اندازه, 128) * n_نخها : 0;
                act_raw_sz = 0;
                اكسر;
            }
            حالة HTP_MM_KERNEL_HVX_QUANT_بلوک:
            حالة HTP_MM_KERNEL_HVX_QUANT_ردیف: {
                size_t q_act_ردیف_اندازه = htp_mm_tiled_act_ردیف_اندازه(wtype, ne10);
                uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(wtype);
                uint32_t n_وزن_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
                uint32_t tile_ردیف_اندازه = n_وزن_tiles * aligned_tile_اندازه;
                size_t tiled_vtcm_اندازه = htp_mm_دور_به_بالا(n_prefetch * tile_ردیف_اندازه, 256);

                src0_sz = tiled_vtcm_اندازه * n_نخها;
                act_sz  = htp_mm_دور_به_بالا(q_act_ردیف_اندازه * act_nrows, 256);

                size_t dst_برش_در_هر_thread = (dst_nrows > 0 && act_nrows == 1) ? htp_mm_دور_به_بالا((dst_ردیف_اندازه + n_نخها - 1) / n_نخها, 128) : 0;
                dst_sz = dst_برش_در_هر_thread * n_نخها;
                size_t raw_ردیف_اندازه = hex_دور_به_بالا(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));
                act_raw_sz = hex_دور_به_بالا(raw_ردیف_اندازه * act_nrows, 128);
                اكسر;
            }
            افتراضي:
                اكسر;
        }
    }

    // Group A: Persistent buffers across chunk compute
    size_t off_گروه_a = 0;
    VTCM_LAYOUT_ALLOC(off_گروه_a, off_act, act_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروه_a, off_bias, bias_sz, bias_sz > 0);

    // Group B: Compute-only buffers (starts at off_group_a)
    size_t off_گروه_b = off_گروه_a;
    VTCM_LAYOUT_ALLOC(off_گروه_b, off_src0, src0_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروه_b, off_dst, dst_sz, dst_sz > 0);
    ثابت size_t گروه_b_اندازه = off_گروه_b - off_گروه_a;

    // Group C: Raw activation staging buffer (overlaps Group B, starts at off_group_a)
    size_t off_گروه_c = off_گروه_a;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروه_c, off_act_raw, act_raw_sz, act_raw_sz > 0);
    ثابت size_t گروه_c_اندازه = off_گروه_c - off_گروه_a;

    L->src0_بایتها    = src0_sz;
    L->act_بایتها     = act_sz;
    L->bias_بایتها    = bias_sz;
    L->dst_بایتها     = dst_sz;
    L->act_raw_بایتها = act_raw_sz;
    L->مجموع_بایتها   = off_گروه_a + hex_smax(گروه_b_اندازه, گروه_c_اندازه);
}

ساكن مضمن bool htp_mm_hvx_solve_vtcm_params(
    صحيح kernel_type,
    صحيح wtype,
    uint32_t ne10,
    uint32_t act_nrows,
    uint32_t n_نخها,
    size_t dst_ردیف_اندازه,
    size_t src0_ردیف_اندازه,
    size_t act_ردیف_اندازه,
    size_t bias_ردیف_اندازه,
    uint32_t n_prefetch,
    size_t vtcm_budget,
    بنية htp_mm_hvx_vtcm_layout * L_بیرون,
    uint32_t * m_chunk_بیرون
) {
    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, act_nrows, n_نخها,
        dst_ردیف_اندازه, src0_ردیف_اندازه, act_ردیف_اندازه, bias_ردیف_اندازه, n_prefetch, false, false
    );

    إذا (L.مجموع_بایتها <= vtcm_budget) {
        *L_بیرون = L;
        *m_chunk_بیرون = act_nrows;
        أرجع true;
    }

    ثابت size_t fixed_بایتها = L.src0_بایتها + L.bias_بایتها + L.dst_بایتها;
    إذا (vtcm_budget <= fixed_بایتها) {
        أرجع false;
    }

    ثابت size_t avail_act = vtcm_budget - fixed_بایتها;
    size_t ردیف_اندازه = 0;
    إذا (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ردیف || kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلوک) {
        ردیف_اندازه = htp_mm_tiled_act_ردیف_اندازه(wtype, ne10);
    } وإلا إذا (kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        ردیف_اندازه = hex_دور_به_بالا(ne10 * 2, 128);
    } وإلا {
        ردیف_اندازه = hex_دور_به_بالا(ne10 * 4, 128);
    }
    إذا (ردیف_اندازه == 0) {
        أرجع false;
    }

    size_t eff_ردیف_اندازه = ردیف_اندازه;
    إذا (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ردیف || kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلوک) {
        eff_ردیف_اندازه += hex_دور_به_بالا(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));
    }

    uint32_t m_chunk = (uint32_t) (avail_act / eff_ردیف_اندازه);
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
        &L, kernel_type, wtype, ne10, m_chunk, n_نخها,
        dst_ردیف_اندازه, src0_ردیف_اندازه, act_ردیف_اندازه, bias_ردیف_اندازه, n_prefetch, false, false
    );

    طالما (m_chunk > 2 && L.مجموع_بایتها > vtcm_budget) {
        m_chunk -= 2;
        htp_mm_hvx_vtcm_layout_build(
            &L, kernel_type, wtype, ne10, m_chunk, n_نخها,
            dst_ردیف_اندازه, src0_ردیف_اندازه, act_ردیف_اندازه, bias_ردیف_اندازه, n_prefetch, false, false
        );
    }

    إذا (L.مجموع_بایتها <= vtcm_budget) {
        *L_بیرون = L;
        *m_chunk_بیرون = m_chunk;
        أرجع true;
    }

    أرجع false;
}

ساكن مضمن size_t htp_mm_hmx_get_2d_vtcm_اندازه(
    صحيح wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_نخها, uint32_t aligned_tile_اندازه, size_t bias_اندازه
) {
    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_نخها, aligned_tile_اندازه, bias_اندازه);
    أرجع L.مجموع_بایتها;
}

ساكن مضمن size_t htp_mm_hmx_get_batched_vtcm_اندازه(
    صحيح wtype, uint32_t k, size_t mc, size_t nc, uint32_t گروه_اندازه, bool pipeline, uint32_t act_نخها, size_t bias_اندازه) {
    (فراغ)pipeline;
    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, wtype, k, mc, nc, گروه_اندازه, false, act_نخها, 0, bias_اندازه);
    أرجع L.مجموع_بایتها;
}

ساكن مضمن bool htp_mm_hmx_solve_batched_params(
    صحيح wtype,
    uint32_t k,
    uint32_t ne01_padded,
    uint32_t ne11,
    uint32_t گروه_اندازه,
    صحيح n_نخها,
    bool pipeline,
    size_t bias_اندازه,
    size_t vtcm_budget,
    size_t * m_chunk_بیرون,
    size_t * n_chunk_بیرون,
    صحيح * act_نخها_بیرون,
    size_t * vtcm_اندازه_بیرون
) {
    size_t بهترین_mblocks = SIZE_MAX;
    صحيح بهترین_act_نخها = 0;
    size_t بهترین_m_chunk = 0;
    size_t بهترین_n_chunk = 0;
    size_t بهترین_vtcm_اندازه = 0;

    صحيح act_نخها = n_نخها;
    طالما (act_نخها >= 1) {
        size_t گروه_overhead = htp_mm_hmx_get_batched_overhead() + (bias_اندازه > 0 ? hex_align_به_بالا(bias_اندازه, HTP_MM_HMX_TILE_اندازه) : 0);
        size_t گروه_اندازه_در_هر_n, گروه_اندازه_در_هر_m, گروه_اندازه_در_هر_mn;
        htp_mm_hmx_get_batched_chunk_costs(k, گروه_اندازه, &گروه_اندازه_در_هر_n, &گروه_اندازه_در_هر_m, &گروه_اندازه_در_هر_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_اندازه_candidate = 0;

        إذا (htp_mm_hmx_compute_chunks(vtcm_budget, گروه_overhead, گروه_اندازه_در_هر_n, گروه_اندازه_در_هر_m, گروه_اندازه_در_هر_mn, hex_align_به_بالا(ne11, 32), ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_هزینه_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_هزینه_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_اندازه_candidate) == 0) {
            size_t exact_اندازه = htp_mm_hmx_get_batched_vtcm_اندازه(wtype, k, m_chunk_candidate, n_chunk_candidate, گروه_اندازه, pipeline, act_نخها, bias_اندازه);
            إذا (exact_اندازه <= vtcm_budget) {
                size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                إذا (mblocks < بهترین_mblocks || (mblocks == بهترین_mblocks && act_نخها > بهترین_act_نخها)) {
                    بهترین_mblocks = mblocks;
                    بهترین_act_نخها = act_نخها;
                    بهترین_m_chunk = m_chunk_candidate;
                    بهترین_n_chunk = n_chunk_candidate;
                    بهترین_vtcm_اندازه = exact_اندازه;
                }
            }
        }
        إذا (act_نخها == 1) {
            act_نخها = 0;
        } وإلا {
            act_نخها /= 2;
        }
    }

    إذا (بهترین_act_نخها > 0) {
        *m_chunk_بیرون = بهترین_m_chunk;
        *n_chunk_بیرون = بهترین_n_chunk;
        *vtcm_اندازه_بیرون = بهترین_vtcm_اندازه;
        *act_نخها_بیرون = بهترین_act_نخها;
        أرجع true;
    }
    أرجع false;
}

ساكن مضمن bool htp_mm_hmx_solve_2d_params(
    صحيح wtype,
    uint32_t k,
    uint32_t m_id_ردیفها,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_هزینه,
    صحيح n_نخها,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_اندازه,
    size_t bias_اندازه,
    size_t vtcm_budget,
    size_t * m_chunk_بیرون,
    size_t * n_chunk_بیرون,
    صحيح * act_نخها_بیرون,
    size_t * vtcm_اندازه_بیرون
) {
    size_t بهترین_mblocks = SIZE_MAX;
    صحيح بهترین_act_نخها = 0;
    size_t بهترین_m_chunk = 0;
    size_t بهترین_n_chunk = 0;
    size_t بهترین_vtcm_اندازه = 0;

    ثابت صحيح m_for_chunks = is_matmul_id ? hex_align_به_بالا(m_id_ردیفها, 32) : ne11_padded;

    صحيح act_نخها = n_نخها;
    طالما (act_نخها >= 1) {
        size_t simple_2d_overhead = htp_mm_hmx_get_2d_overhead(pipeline, is_matmul_id) + (bias_اندازه > 0 ? hex_align_به_بالا(bias_اندازه, HTP_MM_HMX_TILE_اندازه) : 0);
        size_t simple_2d_اندازه_در_هر_n, simple_2d_اندازه_در_هر_m, simple_2d_اندازه_در_هر_mn;
        htp_mm_hmx_get_2d_chunk_costs(wtype, k, pipeline, aligned_tile_اندازه, &simple_2d_اندازه_در_هر_n, &simple_2d_اندازه_در_هر_m, &simple_2d_اندازه_در_هر_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_اندازه_candidate = 0;

        إذا (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_اندازه_در_هر_n, simple_2d_اندازه_در_هر_m, simple_2d_اندازه_در_هر_mn, m_for_chunks, ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_هزینه_W_DEQUANT, (size_t) m_for_هزینه * HTP_MM_HMX_هزینه_A_CONVERT,
                               &m_chunk_candidate, &n_chunk_candidate, &vtcm_اندازه_candidate) == 0) {
            size_t exact_اندازه = htp_mm_hmx_get_2d_vtcm_اندازه(wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_نخها, aligned_tile_اندازه, bias_اندازه);
            إذا (exact_اندازه <= vtcm_budget) {
                size_t mblocks = ((size_t) m_for_هزینه + m_chunk_candidate - 1) / m_chunk_candidate;
                إذا (mblocks < بهترین_mblocks || (mblocks == بهترین_mblocks && act_نخها > بهترین_act_نخها)) {
                    بهترین_mblocks = mblocks;
                    بهترین_act_نخها = act_نخها;
                    بهترین_m_chunk = m_chunk_candidate;
                    بهترین_n_chunk = n_chunk_candidate;
                    بهترین_vtcm_اندازه = exact_اندازه;
                }
            }
        }
        إذا (act_نخها == 1) {
            act_نخها = 0;
        } وإلا {
            act_نخها /= 2;
        }
    }

    إذا (بهترین_act_نخها > 0) {
        *m_chunk_بیرون = بهترین_m_chunk;
        *n_chunk_بیرون = بهترین_n_chunk;
        *vtcm_اندازه_بیرون = بهترین_vtcm_اندازه;
        *act_نخها_بیرون = بهترین_act_نخها;
        أرجع true;
    }
    أرجع false;
}

#اگر_تعریف_شده __cplusplus
}
#پایان_اگر

#پایان_اگر // HTP_MATMUL_OPS_H
