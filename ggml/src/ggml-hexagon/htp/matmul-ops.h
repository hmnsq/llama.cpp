#که_نه_تعریف_شوی HTP_MATMUL_OPS_H
#تعریف_کړه HTP_MATMUL_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hex-fastdiv.h"
#ضمن "hex-common.h"
#ضمن "htp-vtcm.h"

#که_تعریف_شوی __cplusplus
خارجي "C" {
#د_که_پای

// --- HMX Tile Constraints ---
#تعریف_کړه HTP_MM_HMX_TILE_N_ستون_شمېر 32
#تعریف_کړه HTP_MM_HMX_TILE_N_لیکې 32
#تعریف_کړه HTP_MM_HMX_TILE_اندازه   (32 * 32 * حجم(__fp16)) // 2048 bytes
#تعریف_کړه HTP_MM_HMX_TILE_N_ELMS 1024
#تعریف_کړه HTP_MM_HMX_MIN_NROWS   4

#تعریف_کړه HTP_MM_وزن_TILE_K_Q2_0             64
#تعریف_کړه HTP_MM_وزن_TILE_اندازه_Q2_0          HTP_TENSOR_TILE_اندازه_Q2_0
#تعریف_کړه HTP_MM_وزن_ALIGNED_TILE_اندازه_Q2_0  640

// --- Activation Tiled Block Sizes (including padding) ---
#تعریف_کړه HTP_MM_ACT_TILE_اندازه_Q8_0      1152
#تعریف_کړه HTP_MM_ACT_TILE_اندازه_Q8_1      1280
#تعریف_کړه HTP_MM_ACT_TILE_اندازه_F16       2048

#تعریف_کړه HTP_MM_MAX_PREFETCH 16

// --- Solver Cost Model Penalty Weights (HMX-specific) ---
#تعریف_کړه HTP_MM_HMX_لګښت_W_DEQUANT 3 // cost penalty for quantized weight loading/dequantization
#تعریف_کړه HTP_MM_HMX_لګښت_A_CONVERT 2 // cost penalty for activation loading/conversion

// --- DMA Activation Transfer Configuration ---
#تعریف_کړه HTP_MM_DMA_ACT_لیکې_فی_ګام 2
#تعریف_کړه HTP_MM_DMA_ACT_MULTIPLIER    (2 * HTP_MM_DMA_ACT_لیکې_فی_ګام)

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
    HTP_MM_KERNEL_HVX_QUANT_لیکه,      // standard row-wise parallel quantization
    HTP_MM_KERNEL_HVX_QUANT_بلاک,    // parallel block-wise quantization
};

// Op-specific struct for precomputed matmul params
بنية htp_mm_kernel_params {
    uint8_t  kernel_type;        // enum htp_mm_kernel_type
    uint8_t  pipeline;           // 1 = pipelined execution, 0 = standard
    uint8_t  collapse;           // 1 = collapse outer dims into 2D, 0 = standard
    uint8_t  n_hmx;              // 1 = use HMX, 0 = use HVX

    uint8_t  n_تارونه;          // Number of threads to spawn
    uint8_t  n_act_تارونه;      // Number of threads for activation preparation
    uint8_t  n_prefetch;         // Prefetch lookahead buffers/rows in VTCM
    uint8_t  n_weights;          // Number of weights for fused NX

    int32_t  m_chunk;            // Row chunk size (M chunk)
    int32_t  n_chunk;            // Col chunk size (N chunk)
    int32_t  tile_اندازه;          // Weight tile size
    int32_t  aligned_tile_اندازه;  // Aligned weight tile size (padded to 128)
    int32_t  act_لیکه_اندازه;       // Row size for activation scratchpad
    int32_t  vtcm_اندازه;          // Total required scratchpad size in VTCM
    int32_t  vtcm_src0_اندازه;     // src0 scratchpad size in VTCM
    int32_t  vtcm_act_اندازه;      // activation scratchpad size in VTCM
    int32_t  vtcm_bias_اندازه;     // bias scratchpad size in VTCM (fused only)
    int32_t  vtcm_dst_اندازه;      // dst scratchpad size in VTCM

    // Precomputed division values
    بنية fastdiv_ارزښتونه div_ne12_ne1;
    بنية fastdiv_ارزښتونه div_ne1;
    بنية fastdiv_ارزښتونه div_r2;
    بنية fastdiv_ارزښتونه div_r3;
    بنية fastdiv_ارزښتونه div_ne12;
    بنية fastdiv_ارزښتونه div_n_act_تارونه;
    بنية fastdiv_ارزښتونه div_ne00_padded;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#د_که_پای

بنية mmid_لیکه_mapping {
    uint32_t i1;
    uint32_t i2;
};

// Search for optimal (mc, nc) chunk sizes within VTCM budget.
ساكن مضمن صحيح htp_mm_hmx_compute_chunks(size_t   vtcm_ټول,
                              size_t   overhead,
                              size_t   فی_n_لګښت,
                              size_t   فی_m_لګښت,
                              size_t   فی_mn_لګښت,
                              size_t   m,
                              size_t   n,
                              size_t   m_بلاک_لګښت,
                              size_t   n_بلاک_لګښت,
                              size_t * m_chunk_بهر,
                              size_t * n_chunk_بهر,
                              size_t * ټول_بهر) {
    إذا (m == 0 || n == 0) أرجع -1;
    إذا (vtcm_ټول <= overhead) أرجع -1;
    إذا (فی_n_لګښت == 0 || فی_m_لګښت == 0 || فی_mn_لګښت == 0) أرجع -1;

    ثابت size_t usable = vtcm_ټول - overhead;

    size_t غوره_لګښت = SIZE_MAX;
    size_t غوره_لکۍ_waste = SIZE_MAX;
    size_t غوره_mn   = 0;
    size_t غوره_m = 0, غوره_n = 0;

    ثابت size_t max_nc_budget = (usable / فی_n_لګښت);
    ثابت size_t n_max = hex_align_ښکته(hex_smin((size_t)n, max_nc_budget), HTP_MM_HMX_TILE_N_ستون_شمېر);
    لكل (size_t nc = n_max; nc >= HTP_MM_HMX_TILE_N_ستون_شمېر; nc -= HTP_MM_HMX_TILE_N_ستون_شمېر) {
        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        إذا (hex_mul_ډکېدنه(nc, فی_n_لګښت, &n_fixed)) استمر;
        إذا (n_fixed >= usable) اذهب بل_nc;

        إذا (hex_mul_ډکېدنه(nc, فی_mn_لګښت, &ncmn)) اذهب بل_nc;
        إذا (hex_جمع_کړه_ډکېدنه(فی_m_لګښت, ncmn, &mc_denom) || mc_denom == 0) اذهب بل_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hex_align_ښکته(mc, HTP_MM_HMX_TILE_N_لیکې);
            mc = hex_smin(mc, m);

            إذا (mc == 0) {
                اذهب بل_nc;
            }

            size_t mblocks = ((size_t) m + mc - 1) / mc;
            size_t nblocks = ((size_t) n + nc - 1) / nc;
            size_t لګښت    = mblocks * m_بلاک_لګښت + nblocks * n_بلاک_لګښت;
            size_t rem     = n % nc;
            size_t لکۍ_waste = (rem == 0) ? 0 : (nc - rem);
            size_t mn      = mc * nc;
            إذا (لګښت < غوره_لګښت ||
                (لګښت == غوره_لګښت && لکۍ_waste < غوره_لکۍ_waste) ||
                (لګښت == غوره_لګښت && لکۍ_waste == غوره_لکۍ_waste && mn > غوره_mn)) {
                غوره_لګښت       = لګښت;
                غوره_لکۍ_waste = لکۍ_waste;
                غوره_mn         = mn;
                غوره_m          = mc;
                غوره_n          = nc;
            }
        }

بل_nc:
        إذا (nc == HTP_MM_HMX_TILE_N_ستون_شمېر) اكسر;  // avoid size_t underflow
    }

    إذا (غوره_m == 0 || غوره_n == 0) أرجع -1;

    // Compute exact total (with overflow checks)
    size_t t0 = 0, t1 = 0, t2 = 0, mn = 0, ټول = 0;
    إذا (hex_mul_ډکېدنه(غوره_n, فی_n_لګښت, &t0)) أرجع -1;
    إذا (hex_mul_ډکېدنه(غوره_m, فی_m_لګښت, &t1)) أرجع -1;
    إذا (hex_mul_ډکېدنه(غوره_m, غوره_n, &mn))     أرجع -1;
    إذا (hex_mul_ډکېدنه(mn, فی_mn_لګښت, &t2))    أرجع -1;
    إذا (hex_جمع_کړه_ډکېدنه(t0, t1, &ټول))          أرجع -1;
    إذا (hex_جمع_کړه_ډکېدنه(ټول, t2, &ټول))       أرجع -1;
    إذا (hex_جمع_کړه_ډکېدنه(ټول, overhead, &ټول)) أرجع -1;

    *m_chunk_بهر = غوره_m;
    *n_chunk_بهر = غوره_n;
    *ټول_بهر   = ټول;
    أرجع 0;
}

// weight types whose tiles carry a per-block offset (x = d * q + m): the activations need block sums (q8_1)
// (Q2_K: per-16 k sums, q8_1_s16)
ساكن مضمن bool htp_mm_وزن_has_دننه_واټن(صحيح وزن_type) {
    أرجع وزن_type == HTP_TYPE_Q4_1 || وزن_type == HTP_TYPE_Q4_K || وزن_type == HTP_TYPE_Q5_K ||
           وزن_type == HTP_TYPE_Q2_K || وزن_type == HTP_TYPE_Q2_0;
}

// --- Activation/Row Size Helpers ---
ساكن مضمن size_t htp_mm_q8_0_tiled_لیکه_اندازه(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    أرجع nb_32 * HTP_MM_ACT_TILE_اندازه_Q8_0;
}

ساكن مضمن size_t htp_mm_q8_1_tiled_لیکه_اندازه(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    أرجع nb_32 * HTP_MM_ACT_TILE_اندازه_Q8_1;
}

ساكن مضمن size_t htp_mm_f16_tiled_لیکه_اندازه(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    أرجع nb_32 * HTP_MM_ACT_TILE_اندازه_F16;
}

ساكن مضمن size_t htp_mm_tiled_act_لیکه_اندازه(صحيح وزن_type, uint32_t ne) {
    إذا (وزن_type == HTP_TYPE_F16) {
        أرجع htp_mm_f16_tiled_لیکه_اندازه(ne);
    }
    أرجع htp_mm_وزن_has_دننه_واټن(وزن_type) ? htp_mm_q8_1_tiled_لیکه_اندازه(ne) : htp_mm_q8_0_tiled_لیکه_اندازه(ne);
}

ساكن مضمن size_t htp_mm_پړاو_پورته(size_t n, size_t m) {
    أرجع ((n + m - 1) / m) * m;
}

ساكن مضمن bool htp_mm_hmx_pipeline(uint32_t m) {
    أرجع m > 32;
}

ساكن مضمن فراغ htp_mm_hmx_get_2d_chunk_costs(
    صحيح wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_اندازه,
    size_t * اندازه_فی_n_بهر, size_t * اندازه_فی_m_بهر, size_t * اندازه_فی_mn_بهر
) {
    ثابت bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
    ثابت size_t لیکه_stride = htp_tensor_get_لیکه_اندازه(wtype, k);
    ثابت size_t vec_dot_اندازه = k * حجم(uint16_t);
    ثابت uint32_t n_وزن_tiles = k / htp_tensor_get_tile_k(wtype);
    ثابت size_t qweight_لیکه_stride = is_quant ? (size_t)(n_وزن_tiles * aligned_tile_اندازه) / 32 : 0;

    *اندازه_فی_n_بهر = (pipeline ? 2 : 1) * (is_quant ? qweight_لیکه_stride : لیکه_stride) +
                      (pipeline ? 2 * vec_dot_اندازه : vec_dot_اندازه);
    *اندازه_فی_m_بهر = vec_dot_اندازه;
    *اندازه_فی_mn_بهر = (pipeline ? 2 : 1) * حجم(uint16_t);
}

ساكن مضمن فراغ htp_mm_hmx_get_batched_chunk_costs(
    uint32_t k, uint32_t ډله_اندازه,
    size_t * اندازه_فی_n_بهر, size_t * اندازه_فی_m_بهر, size_t * اندازه_فی_mn_بهر
) {
    ثابت size_t vec_dot_اندازه = k * حجم(uint16_t);
    *اندازه_فی_n_بهر = 3 * vec_dot_اندازه;
    *اندازه_فی_m_بهر = ډله_اندازه * vec_dot_اندازه;
    *اندازه_فی_mn_بهر = حجم(uint16_t);
}

ساكن مضمن size_t htp_mm_hmx_get_2d_overhead(bool pipeline, bool is_matmul_id) {
    size_t عدد_regions = pipeline ? 7 : (is_matmul_id ? 4 : 5);
    أرجع عدد_regions * HTP_MM_HMX_TILE_اندازه + 256;
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
    size_t وزن_area_بایټونه;
    size_t act_area_بایټونه;
    size_t act_f32_بایټونه;
    size_t output_area_بایټونه;
    size_t scratch_بایټونه[2];
    size_t act_سر_stride;
    size_t bias_بایټونه;

    size_t ټول_بایټونه;
};

بنية htp_mm_hvx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_act;           // vtcm_act (activation)
    size_t off_src0;          // vtcm_src0 (weight/Wk)
    size_t off_bias;          // vtcm_bias (bias / fused add only)
    size_t off_dst;           // vtcm_dst (output scratch)
    size_t off_act_raw;       // vtcm_act_raw (raw activation DMA staging)

    // Cached sizes
    size_t src0_بایټونه;
    size_t act_بایټونه;
    size_t bias_بایټونه;
    size_t dst_بایټونه;
    size_t act_raw_بایټونه;

    size_t ټول_بایټونه;
};

ساكن مضمن فراغ htp_mm_hmx_vtcm_layout_build(
    بنية htp_mm_hmx_vtcm_layout * L,
    صحيح kernel_type,
    صحيح wtype,
    uint32_t k,
    size_t mc,
    size_t nc,
    uint32_t ډله_اندازه,
    bool pipeline,
    uint32_t act_تارونه,
    uint32_t aligned_tile_اندازه,
    size_t bias_اندازه
) {
    size_t off = 0;

    إذا (kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        ثابت size_t vec_dot_اندازه     = k * حجم(uint16_t);
        ثابت size_t act_سر_stride   = mc * k;
        ثابت size_t وزن_area_اندازه  = hex_align_پورته(nc * vec_dot_اندازه, HTP_MM_HMX_TILE_اندازه);
        ثابت size_t activation_area_اندازه = hex_align_پورته(ډله_اندازه * act_سر_stride * حجم(uint16_t), HTP_MM_HMX_TILE_اندازه);
        ثابت size_t output_area_اندازه  = hex_align_پورته(ډله_اندازه * mc * nc * حجم(uint16_t), HTP_MM_HMX_TILE_اندازه);
        ثابت size_t scratch_area_اندازه = hex_align_پورته(nc * vec_dot_اندازه, HTP_MM_HMX_TILE_اندازه);
        ثابت size_t min_f32_اندازه = hex_align_پورته(act_تارونه * HTP_MM_DMA_ACT_MULTIPLIER * k * حجم(عائم), 128);

        // Group A: Permanent activation tiles and scales
        size_t off_ډله_a = 0;
        VTCM_LAYOUT_ALLOC(off_ډله_a, off_act, activation_area_اندازه);
        VTCM_LAYOUT_ALLOC(off_ډله_a, off_scales, HTP_MM_HMX_TILE_اندازه); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ډله_a, off_bias, hex_align_پورته(bias_اندازه, HTP_MM_HMX_TILE_اندازه), bias_اندازه > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_ډله_b = off_ډله_a;
        VTCM_LAYOUT_ALLOC(off_ډله_b, off_وزن[0], وزن_area_اندازه);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ډله_b, off_وزن[1], وزن_area_اندازه, false);
        VTCM_LAYOUT_ALLOC(off_ډله_b, off_dst[0], output_area_اندازه);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ډله_b, off_dst[1], output_area_اندازه, false);
        VTCM_LAYOUT_ALLOC(off_ډله_b, off_scratch[0], scratch_area_اندازه);
        VTCM_LAYOUT_ALLOC(off_ډله_b, off_scratch[1], scratch_area_اندازه);

        ثابت size_t ډله_b_اندازه = off_ډله_b - off_ډله_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ثابت size_t max_f32_اندازه = act_تارونه * 64 * k * حجم(عائم);
        ثابت size_t act_f32_اندازه = hex_align_پورته(hex_smin(max_f32_اندازه, hex_smax(min_f32_اندازه, ډله_b_اندازه)), 128);
        size_t off_ډله_c = off_ډله_a;
        VTCM_LAYOUT_ALLOC(off_ډله_c, off_act_f32, act_f32_اندازه);

        ثابت size_t ډله_c_اندازه = off_ډله_c - off_ډله_a;

        L->وزن_area_بایټونه = وزن_area_اندازه;
        L->act_area_بایټونه    = activation_area_اندازه;
        L->act_f32_بایټونه     = act_f32_اندازه;
        L->output_area_بایټونه = output_area_اندازه;
        L->scratch_بایټونه[0]  = scratch_area_اندازه;
        L->scratch_بایټونه[1]  = scratch_area_اندازه;
        L->act_سر_stride   = act_سر_stride;
        L->bias_بایټونه        = bias_اندازه;

        off = off_ډله_a + hex_smax(ډله_b_اندازه, ډله_c_اندازه);
    } وإلا {
        // HTP_MM_KERNEL_HMX_2D
        ثابت bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
        ثابت size_t لیکه_stride = htp_tensor_get_لیکه_اندازه(wtype, k);
        ثابت size_t vec_dot_اندازه = k * حجم(uint16_t);
        ثابت uint32_t n_وزن_tiles = k / htp_tensor_get_tile_k(wtype);

        ثابت size_t min_f32_اندازه = hex_align_پورته(act_تارونه * HTP_MM_DMA_ACT_MULTIPLIER * k * حجم(عائم), 128);
        ثابت size_t وزن_area_اندازه = is_quant
            ? hex_align_پورته((nc / 32) * n_وزن_tiles * aligned_tile_اندازه, HTP_MM_HMX_TILE_اندازه)
            : hex_align_پورته(nc * لیکه_stride, HTP_MM_HMX_TILE_اندازه);
        ثابت size_t act_area_اندازه    = hex_align_پورته(mc * vec_dot_اندازه, HTP_MM_HMX_TILE_اندازه);
        ثابت size_t output_area_اندازه = hex_align_پورته(mc * nc * حجم(__fp16), HTP_MM_HMX_TILE_اندازه);

        ثابت size_t scratch0_اندازه = hex_align_پورته(nc * vec_dot_اندازه, HTP_MM_HMX_TILE_اندازه);
        ثابت size_t scratch1_اندازه = pipeline ? scratch0_اندازه : 0;

        // Group A:  Scales and activation tiles (must not overlap with Group B or C)
        size_t off_ډله_a = 0;
        VTCM_LAYOUT_ALLOC(off_ډله_a, off_scales, HTP_MM_HMX_TILE_اندازه); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC(off_ډله_a, off_act, act_area_اندازه);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ډله_a, off_bias, hex_align_پورته(bias_اندازه, HTP_MM_HMX_TILE_اندازه), bias_اندازه > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_ډله_b = off_ډله_a;
        VTCM_LAYOUT_ALLOC(off_ډله_b, off_وزن[0], وزن_area_اندازه);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ډله_b, off_وزن[1], وزن_area_اندازه, pipeline);
        VTCM_LAYOUT_ALLOC(off_ډله_b, off_dst[0], output_area_اندازه);
        VTCM_LAYOUT_ALLOC(off_ډله_b, off_scratch[0], scratch0_اندازه);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ډله_b, off_scratch[1], scratch0_اندازه, pipeline);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_ډله_b, off_dst[1], output_area_اندازه, pipeline);

        ثابت size_t ډله_b_اندازه = off_ډله_b - off_ډله_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ثابت size_t max_f32_اندازه = act_تارونه * 64 * k * حجم(عائم);
        ثابت size_t act_f32_اندازه = hex_align_پورته(hex_smin(max_f32_اندازه, hex_smax(min_f32_اندازه, ډله_b_اندازه)), 128);
        size_t off_ډله_c = off_ډله_a;
        VTCM_LAYOUT_ALLOC(off_ډله_c, off_act_f32, act_f32_اندازه);

        ثابت size_t ډله_c_اندازه = off_ډله_c - off_ډله_a;

        L->وزن_area_بایټونه = وزن_area_اندازه;
        L->act_area_بایټونه    = act_area_اندازه;
        L->act_f32_بایټونه     = act_f32_اندازه;
        L->output_area_بایټونه = output_area_اندازه;
        L->scratch_بایټونه[0]  = scratch0_اندازه;
        L->scratch_بایټونه[1]  = scratch1_اندازه;
        L->act_سر_stride   = 0;
        L->bias_بایټونه        = bias_اندازه;

        off = off_ډله_a + hex_smax(ډله_b_اندازه, ډله_c_اندازه);
    }

    L->ټول_بایټونه = off;
}

ساكن مضمن فراغ htp_mm_hvx_vtcm_layout_build(
    بنية htp_mm_hvx_vtcm_layout * L,
    صحيح kernel_type,
    صحيح wtype,
    uint32_t ne10,       // k
    uint32_t act_nrows,  // m_total
    uint32_t n_تارونه,
    size_t dst_لیکه_اندازه,
    size_t src0_لیکه_اندازه,
    size_t act_لیکه_اندازه,
    size_t bias_لیکه_اندازه,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_nx
) {
    (فراغ)act_لیکه_اندازه;
    size_t src0_sz    = 0;
    size_t act_sz     = 0;
    size_t bias_sz    = bias_لیکه_اندازه > 0 ? htp_mm_پړاو_پورته(bias_لیکه_اندازه, 128) : 0;
    size_t dst_sz     = 0;
    size_t act_raw_sz = 0;

    إذا (is_fused_nx) {
        uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(wtype);
        uint32_t n_k_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        uint32_t tile_لیکه_اندازه = n_k_tiles * aligned_tile_اندازه;
        size_t وزن_sz_فی_thread = hex_پړاو_پورته(n_prefetch * tile_لیکه_اندازه, 128);

        size_t tiled_act_لیکه_اندازه = htp_mm_tiled_act_لیکه_اندازه(wtype, ne10);
        size_t q_act_sz = hex_پړاو_پورته(tiled_act_لیکه_اندازه * act_nrows, 128);
        size_t raw_لیکه_اندازه = hex_پړاو_پورته(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

        src0_sz    = وزن_sz_فی_thread * n_تارونه; // shared single-weight prefetch buffer
        act_sz     = q_act_sz;                         // quantized activation buffer
        bias_sz    = 0;
        dst_sz     = 0;
        act_raw_sz = hex_پړاو_پورته(raw_لیکه_اندازه * act_nrows, 128);
    } وإلا إذا (is_matmul_id) {
        ثابت size_t act_لیکه_اندازه_tiled = htp_mm_tiled_act_لیکه_اندازه(wtype, ne10);
        ثابت uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(wtype);
        ثابت uint32_t n_k_tiles         = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        ثابت uint32_t tile_لیکه_اندازه     = n_k_tiles * aligned_tile_اندازه;
        size_t src0_sz_فی_thread        = htp_mm_پړاو_پورته(n_prefetch * tile_لیکه_اندازه, 256);

        act_sz = htp_mm_پړاو_پورته(act_لیکه_اندازه_tiled * act_nrows, 256);

        size_t raw_لیکه_اندازه = hex_پړاو_پورته(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

        src0_sz    = src0_sz_فی_thread * n_تارونه;
        dst_sz     = 0;
        bias_sz    = 0;
        act_raw_sz = hex_پړاو_پورته(raw_لیکه_اندازه * act_nrows, 128);
    } وإلا {
        ثابت size_t src0_لیکه_اندازه_padded = htp_mm_پړاو_پورته(src0_لیکه_اندازه, 128);
        ثابت size_t dst_nrows = (act_nrows > 1) ? 0 : 1;

        بدل (kernel_type) {
            حالة HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                size_t f16_act_لیکه_اندازه = htp_mm_پړاو_پورته(ne10 * 2, 128);
                act_sz     = htp_mm_پړاو_پورته(f16_act_لیکه_اندازه * act_nrows, 256);
                src0_sz    = htp_mm_پړاو_پورته(n_prefetch * src0_لیکه_اندازه_padded, 256) * n_تارونه;
                dst_sz     = dst_nrows > 0 ? htp_mm_پړاو_پورته(dst_لیکه_اندازه, 128) * n_تارونه : 0;
                act_raw_sz = hex_پړاو_پورته(hex_پړاو_پورته(ne10 * حجم(عائم), 128) * act_nrows, 128);
                اكسر;
            }
            حالة HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            حالة HTP_MM_KERNEL_HVX_BF16_F32_VTCM: {
                size_t f32_act_لیکه_اندازه = htp_mm_پړاو_پورته(ne10 * 4, 128);
                act_sz     = htp_mm_پړاو_پورته(f32_act_لیکه_اندازه * act_nrows, 256);
                src0_sz    = htp_mm_پړاو_پورته(n_prefetch * src0_لیکه_اندازه_padded, 256) * n_تارونه;
                dst_sz     = dst_nrows > 0 ? htp_mm_پړاو_پورته(dst_لیکه_اندازه, 128) * n_تارونه : 0;
                act_raw_sz = 0;
                اكسر;
            }
            حالة HTP_MM_KERNEL_HVX_QUANT_بلاک:
            حالة HTP_MM_KERNEL_HVX_QUANT_لیکه: {
                size_t q_act_لیکه_اندازه = htp_mm_tiled_act_لیکه_اندازه(wtype, ne10);
                uint32_t aligned_tile_اندازه = htp_tensor_get_tile_padded_اندازه(wtype);
                uint32_t n_وزن_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
                uint32_t tile_لیکه_اندازه = n_وزن_tiles * aligned_tile_اندازه;
                size_t tiled_vtcm_اندازه = htp_mm_پړاو_پورته(n_prefetch * tile_لیکه_اندازه, 256);

                src0_sz = tiled_vtcm_اندازه * n_تارونه;
                act_sz  = htp_mm_پړاو_پورته(q_act_لیکه_اندازه * act_nrows, 256);

                size_t dst_ټوټه_فی_thread = (dst_nrows > 0 && act_nrows == 1) ? htp_mm_پړاو_پورته((dst_لیکه_اندازه + n_تارونه - 1) / n_تارونه, 128) : 0;
                dst_sz = dst_ټوټه_فی_thread * n_تارونه;
                size_t raw_لیکه_اندازه = hex_پړاو_پورته(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));
                act_raw_sz = hex_پړاو_پورته(raw_لیکه_اندازه * act_nrows, 128);
                اكسر;
            }
            افتراضي:
                اكسر;
        }
    }

    // Group A: Persistent buffers across chunk compute
    size_t off_ډله_a = 0;
    VTCM_LAYOUT_ALLOC(off_ډله_a, off_act, act_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ډله_a, off_bias, bias_sz, bias_sz > 0);

    // Group B: Compute-only buffers (starts at off_group_a)
    size_t off_ډله_b = off_ډله_a;
    VTCM_LAYOUT_ALLOC(off_ډله_b, off_src0, src0_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ډله_b, off_dst, dst_sz, dst_sz > 0);
    ثابت size_t ډله_b_اندازه = off_ډله_b - off_ډله_a;

    // Group C: Raw activation staging buffer (overlaps Group B, starts at off_group_a)
    size_t off_ډله_c = off_ډله_a;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_ډله_c, off_act_raw, act_raw_sz, act_raw_sz > 0);
    ثابت size_t ډله_c_اندازه = off_ډله_c - off_ډله_a;

    L->src0_بایټونه    = src0_sz;
    L->act_بایټونه     = act_sz;
    L->bias_بایټونه    = bias_sz;
    L->dst_بایټونه     = dst_sz;
    L->act_raw_بایټونه = act_raw_sz;
    L->ټول_بایټونه   = off_ډله_a + hex_smax(ډله_b_اندازه, ډله_c_اندازه);
}

ساكن مضمن bool htp_mm_hvx_solve_vtcm_params(
    صحيح kernel_type,
    صحيح wtype,
    uint32_t ne10,
    uint32_t act_nrows,
    uint32_t n_تارونه,
    size_t dst_لیکه_اندازه,
    size_t src0_لیکه_اندازه,
    size_t act_لیکه_اندازه,
    size_t bias_لیکه_اندازه,
    uint32_t n_prefetch,
    size_t vtcm_budget,
    بنية htp_mm_hvx_vtcm_layout * L_بهر,
    uint32_t * m_chunk_بهر
) {
    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, act_nrows, n_تارونه,
        dst_لیکه_اندازه, src0_لیکه_اندازه, act_لیکه_اندازه, bias_لیکه_اندازه, n_prefetch, false, false
    );

    إذا (L.ټول_بایټونه <= vtcm_budget) {
        *L_بهر = L;
        *m_chunk_بهر = act_nrows;
        أرجع true;
    }

    ثابت size_t fixed_بایټونه = L.src0_بایټونه + L.bias_بایټونه + L.dst_بایټونه;
    إذا (vtcm_budget <= fixed_بایټونه) {
        أرجع false;
    }

    ثابت size_t avail_act = vtcm_budget - fixed_بایټونه;
    size_t لیکه_اندازه = 0;
    إذا (kernel_type == HTP_MM_KERNEL_HVX_QUANT_لیکه || kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک) {
        لیکه_اندازه = htp_mm_tiled_act_لیکه_اندازه(wtype, ne10);
    } وإلا إذا (kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        لیکه_اندازه = hex_پړاو_پورته(ne10 * 2, 128);
    } وإلا {
        لیکه_اندازه = hex_پړاو_پورته(ne10 * 4, 128);
    }
    إذا (لیکه_اندازه == 0) {
        أرجع false;
    }

    size_t eff_لیکه_اندازه = لیکه_اندازه;
    إذا (kernel_type == HTP_MM_KERNEL_HVX_QUANT_لیکه || kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک) {
        eff_لیکه_اندازه += hex_پړاو_پورته(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));
    }

    uint32_t m_chunk = (uint32_t) (avail_act / eff_لیکه_اندازه);
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
        &L, kernel_type, wtype, ne10, m_chunk, n_تارونه,
        dst_لیکه_اندازه, src0_لیکه_اندازه, act_لیکه_اندازه, bias_لیکه_اندازه, n_prefetch, false, false
    );

    طالما (m_chunk > 2 && L.ټول_بایټونه > vtcm_budget) {
        m_chunk -= 2;
        htp_mm_hvx_vtcm_layout_build(
            &L, kernel_type, wtype, ne10, m_chunk, n_تارونه,
            dst_لیکه_اندازه, src0_لیکه_اندازه, act_لیکه_اندازه, bias_لیکه_اندازه, n_prefetch, false, false
        );
    }

    إذا (L.ټول_بایټونه <= vtcm_budget) {
        *L_بهر = L;
        *m_chunk_بهر = m_chunk;
        أرجع true;
    }

    أرجع false;
}

ساكن مضمن size_t htp_mm_hmx_get_2d_vtcm_اندازه(
    صحيح wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_تارونه, uint32_t aligned_tile_اندازه, size_t bias_اندازه
) {
    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_تارونه, aligned_tile_اندازه, bias_اندازه);
    أرجع L.ټول_بایټونه;
}

ساكن مضمن size_t htp_mm_hmx_get_batched_vtcm_اندازه(
    صحيح wtype, uint32_t k, size_t mc, size_t nc, uint32_t ډله_اندازه, bool pipeline, uint32_t act_تارونه, size_t bias_اندازه) {
    (فراغ)pipeline;
    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, wtype, k, mc, nc, ډله_اندازه, false, act_تارونه, 0, bias_اندازه);
    أرجع L.ټول_بایټونه;
}

ساكن مضمن bool htp_mm_hmx_solve_batched_params(
    صحيح wtype,
    uint32_t k,
    uint32_t ne01_padded,
    uint32_t ne11,
    uint32_t ډله_اندازه,
    صحيح n_تارونه,
    bool pipeline,
    size_t bias_اندازه,
    size_t vtcm_budget,
    size_t * m_chunk_بهر,
    size_t * n_chunk_بهر,
    صحيح * act_تارونه_بهر,
    size_t * vtcm_اندازه_بهر
) {
    size_t غوره_mblocks = SIZE_MAX;
    صحيح غوره_act_تارونه = 0;
    size_t غوره_m_chunk = 0;
    size_t غوره_n_chunk = 0;
    size_t غوره_vtcm_اندازه = 0;

    صحيح act_تارونه = n_تارونه;
    طالما (act_تارونه >= 1) {
        size_t ډله_overhead = htp_mm_hmx_get_batched_overhead() + (bias_اندازه > 0 ? hex_align_پورته(bias_اندازه, HTP_MM_HMX_TILE_اندازه) : 0);
        size_t ډله_اندازه_فی_n, ډله_اندازه_فی_m, ډله_اندازه_فی_mn;
        htp_mm_hmx_get_batched_chunk_costs(k, ډله_اندازه, &ډله_اندازه_فی_n, &ډله_اندازه_فی_m, &ډله_اندازه_فی_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_اندازه_candidate = 0;

        إذا (htp_mm_hmx_compute_chunks(vtcm_budget, ډله_overhead, ډله_اندازه_فی_n, ډله_اندازه_فی_m, ډله_اندازه_فی_mn, hex_align_پورته(ne11, 32), ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_لګښت_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_لګښت_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_اندازه_candidate) == 0) {
            size_t exact_اندازه = htp_mm_hmx_get_batched_vtcm_اندازه(wtype, k, m_chunk_candidate, n_chunk_candidate, ډله_اندازه, pipeline, act_تارونه, bias_اندازه);
            إذا (exact_اندازه <= vtcm_budget) {
                size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                إذا (mblocks < غوره_mblocks || (mblocks == غوره_mblocks && act_تارونه > غوره_act_تارونه)) {
                    غوره_mblocks = mblocks;
                    غوره_act_تارونه = act_تارونه;
                    غوره_m_chunk = m_chunk_candidate;
                    غوره_n_chunk = n_chunk_candidate;
                    غوره_vtcm_اندازه = exact_اندازه;
                }
            }
        }
        إذا (act_تارونه == 1) {
            act_تارونه = 0;
        } وإلا {
            act_تارونه /= 2;
        }
    }

    إذا (غوره_act_تارونه > 0) {
        *m_chunk_بهر = غوره_m_chunk;
        *n_chunk_بهر = غوره_n_chunk;
        *vtcm_اندازه_بهر = غوره_vtcm_اندازه;
        *act_تارونه_بهر = غوره_act_تارونه;
        أرجع true;
    }
    أرجع false;
}

ساكن مضمن bool htp_mm_hmx_solve_2d_params(
    صحيح wtype,
    uint32_t k,
    uint32_t m_id_لیکې,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_لګښت,
    صحيح n_تارونه,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_اندازه,
    size_t bias_اندازه,
    size_t vtcm_budget,
    size_t * m_chunk_بهر,
    size_t * n_chunk_بهر,
    صحيح * act_تارونه_بهر,
    size_t * vtcm_اندازه_بهر
) {
    size_t غوره_mblocks = SIZE_MAX;
    صحيح غوره_act_تارونه = 0;
    size_t غوره_m_chunk = 0;
    size_t غوره_n_chunk = 0;
    size_t غوره_vtcm_اندازه = 0;

    ثابت صحيح m_for_chunks = is_matmul_id ? hex_align_پورته(m_id_لیکې, 32) : ne11_padded;

    صحيح act_تارونه = n_تارونه;
    طالما (act_تارونه >= 1) {
        size_t simple_2d_overhead = htp_mm_hmx_get_2d_overhead(pipeline, is_matmul_id) + (bias_اندازه > 0 ? hex_align_پورته(bias_اندازه, HTP_MM_HMX_TILE_اندازه) : 0);
        size_t simple_2d_اندازه_فی_n, simple_2d_اندازه_فی_m, simple_2d_اندازه_فی_mn;
        htp_mm_hmx_get_2d_chunk_costs(wtype, k, pipeline, aligned_tile_اندازه, &simple_2d_اندازه_فی_n, &simple_2d_اندازه_فی_m, &simple_2d_اندازه_فی_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_اندازه_candidate = 0;

        إذا (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_اندازه_فی_n, simple_2d_اندازه_فی_m, simple_2d_اندازه_فی_mn, m_for_chunks, ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_لګښت_W_DEQUANT, (size_t) m_for_لګښت * HTP_MM_HMX_لګښت_A_CONVERT,
                               &m_chunk_candidate, &n_chunk_candidate, &vtcm_اندازه_candidate) == 0) {
            size_t exact_اندازه = htp_mm_hmx_get_2d_vtcm_اندازه(wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_تارونه, aligned_tile_اندازه, bias_اندازه);
            إذا (exact_اندازه <= vtcm_budget) {
                size_t mblocks = ((size_t) m_for_لګښت + m_chunk_candidate - 1) / m_chunk_candidate;
                إذا (mblocks < غوره_mblocks || (mblocks == غوره_mblocks && act_تارونه > غوره_act_تارونه)) {
                    غوره_mblocks = mblocks;
                    غوره_act_تارونه = act_تارونه;
                    غوره_m_chunk = m_chunk_candidate;
                    غوره_n_chunk = n_chunk_candidate;
                    غوره_vtcm_اندازه = exact_اندازه;
                }
            }
        }
        إذا (act_تارونه == 1) {
            act_تارونه = 0;
        } وإلا {
            act_تارونه /= 2;
        }
    }

    إذا (غوره_act_تارونه > 0) {
        *m_chunk_بهر = غوره_m_chunk;
        *n_chunk_بهر = غوره_n_chunk;
        *vtcm_اندازه_بهر = غوره_vtcm_اندازه;
        *act_تارونه_بهر = غوره_act_تارونه;
        أرجع true;
    }
    أرجع false;
}

#که_تعریف_شوی __cplusplus
}
#د_که_پای

#د_که_پای // HTP_MATMUL_OPS_H
