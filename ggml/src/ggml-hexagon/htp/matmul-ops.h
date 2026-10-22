#جے_بنا_تعریف HTP_MATMUL_OPS_H
#تعریف_کرو HTP_MATMUL_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hex-fastdiv.h"
#ضمن "hex-common.h"
#ضمن "htp-vtcm.h"

#جے_تعریف_شدہ __cplusplus
خارجي "C" {
#جے_انت

// --- HMX Tile Constraints ---
#تعریف_کرو HTP_MM_HMX_TILE_N_خانے_گنتی 32
#تعریف_کرو HTP_MM_HMX_TILE_N_صفاں 32
#تعریف_کرو HTP_MM_HMX_TILE_ناپ   (32 * 32 * حجم(__fp16)) // 2048 bytes
#تعریف_کرو HTP_MM_HMX_TILE_N_ELMS 1024
#تعریف_کرو HTP_MM_HMX_MIN_NROWS   4

#تعریف_کرو HTP_MM_بھار_TILE_K_Q2_0             64
#تعریف_کرو HTP_MM_بھار_TILE_ناپ_Q2_0          HTP_TENSOR_TILE_ناپ_Q2_0
#تعریف_کرو HTP_MM_بھار_ALIGNED_TILE_ناپ_Q2_0  640

// --- Activation Tiled Block Sizes (including padding) ---
#تعریف_کرو HTP_MM_ACT_TILE_ناپ_Q8_0      1152
#تعریف_کرو HTP_MM_ACT_TILE_ناپ_Q8_1      1280
#تعریف_کرو HTP_MM_ACT_TILE_ناپ_F16       2048

#تعریف_کرو HTP_MM_MAX_PREFETCH 16

// --- Solver Cost Model Penalty Weights (HMX-specific) ---
#تعریف_کرو HTP_MM_HMX_لاگت_W_DEQUANT 3 // cost penalty for quantized weight loading/dequantization
#تعریف_کرو HTP_MM_HMX_لاگت_A_CONVERT 2 // cost penalty for activation loading/conversion

// --- DMA Activation Transfer Configuration ---
#تعریف_کرو HTP_MM_DMA_ACT_صفاں_فی_قدم 2
#تعریف_کرو HTP_MM_DMA_ACT_MULTIPLIER    (2 * HTP_MM_DMA_ACT_صفاں_فی_قدم)

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
    HTP_MM_KERNEL_HVX_QUANT_بلاک,    // parallel block-wise quantization
};

// Op-specific struct for precomputed matmul params
بنية htp_mm_kernel_params {
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
    int32_t  tile_ناپ;          // Weight tile size
    int32_t  aligned_tile_ناپ;  // Aligned weight tile size (padded to 128)
    int32_t  act_صف_ناپ;       // Row size for activation scratchpad
    int32_t  vtcm_ناپ;          // Total required scratchpad size in VTCM
    int32_t  vtcm_src0_ناپ;     // src0 scratchpad size in VTCM
    int32_t  vtcm_act_ناپ;      // activation scratchpad size in VTCM
    int32_t  vtcm_bias_ناپ;     // bias scratchpad size in VTCM (fused only)
    int32_t  vtcm_dst_ناپ;      // dst scratchpad size in VTCM

    // Precomputed division values
    بنية fastdiv_اقدار div_ne12_ne1;
    بنية fastdiv_اقدار div_ne1;
    بنية fastdiv_اقدار div_r2;
    بنية fastdiv_اقدار div_r3;
    بنية fastdiv_اقدار div_ne12;
    بنية fastdiv_اقدار div_n_act_دھاگے;
    بنية fastdiv_اقدار div_ne00_padded;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#وإلا
_Static_assert(حجم(بنية htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#جے_انت

بنية mmid_صف_mapping {
    uint32_t i1;
    uint32_t i2;
};

// Search for optimal (mc, nc) chunk sizes within VTCM budget.
ساكن مضمن صحيح htp_mm_hmx_compute_chunks(size_t   vtcm_کل,
                              size_t   overhead,
                              size_t   فی_n_لاگت,
                              size_t   فی_m_لاگت,
                              size_t   فی_mn_لاگت,
                              size_t   m,
                              size_t   n,
                              size_t   m_بلاک_لاگت,
                              size_t   n_بلاک_لاگت,
                              size_t * m_chunk_باہرلا,
                              size_t * n_chunk_باہرلا,
                              size_t * کل_باہرلا) {
    إذا (m == 0 || n == 0) أرجع -1;
    إذا (vtcm_کل <= overhead) أرجع -1;
    إذا (فی_n_لاگت == 0 || فی_m_لاگت == 0 || فی_mn_لاگت == 0) أرجع -1;

    ثابت size_t usable = vtcm_کل - overhead;

    size_t بہترین_لاگت = SIZE_MAX;
    size_t بہترین_پوچھ_waste = SIZE_MAX;
    size_t بہترین_mn   = 0;
    size_t بہترین_m = 0, بہترین_n = 0;

    ثابت size_t max_nc_budget = (usable / فی_n_لاگت);
    ثابت size_t n_max = hex_align_تھلے(hex_smin((size_t)n, max_nc_budget), HTP_MM_HMX_TILE_N_خانے_گنتی);
    لكل (size_t nc = n_max; nc >= HTP_MM_HMX_TILE_N_خانے_گنتی; nc -= HTP_MM_HMX_TILE_N_خانے_گنتی) {
        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        إذا (hex_mul_حد_توں_ودھ(nc, فی_n_لاگت, &n_fixed)) استمر;
        إذا (n_fixed >= usable) اذهب اگلا_nc;

        إذا (hex_mul_حد_توں_ودھ(nc, فی_mn_لاگت, &ncmn)) اذهب اگلا_nc;
        إذا (hex_جوڑو_حد_توں_ودھ(فی_m_لاگت, ncmn, &mc_denom) || mc_denom == 0) اذهب اگلا_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hex_align_تھلے(mc, HTP_MM_HMX_TILE_N_صفاں);
            mc = hex_smin(mc, m);

            إذا (mc == 0) {
                اذهب اگلا_nc;
            }

            size_t mblocks = ((size_t) m + mc - 1) / mc;
            size_t nblocks = ((size_t) n + nc - 1) / nc;
            size_t لاگت    = mblocks * m_بلاک_لاگت + nblocks * n_بلاک_لاگت;
            size_t rem     = n % nc;
            size_t پوچھ_waste = (rem == 0) ? 0 : (nc - rem);
            size_t mn      = mc * nc;
            إذا (لاگت < بہترین_لاگت ||
                (لاگت == بہترین_لاگت && پوچھ_waste < بہترین_پوچھ_waste) ||
                (لاگت == بہترین_لاگت && پوچھ_waste == بہترین_پوچھ_waste && mn > بہترین_mn)) {
                بہترین_لاگت       = لاگت;
                بہترین_پوچھ_waste = پوچھ_waste;
                بہترین_mn         = mn;
                بہترین_m          = mc;
                بہترین_n          = nc;
            }
        }

اگلا_nc:
        إذا (nc == HTP_MM_HMX_TILE_N_خانے_گنتی) اكسر;  // avoid size_t underflow
    }

    إذا (بہترین_m == 0 || بہترین_n == 0) أرجع -1;

    // Compute exact total (with overflow checks)
    size_t t0 = 0, t1 = 0, t2 = 0, mn = 0, کل = 0;
    إذا (hex_mul_حد_توں_ودھ(بہترین_n, فی_n_لاگت, &t0)) أرجع -1;
    إذا (hex_mul_حد_توں_ودھ(بہترین_m, فی_m_لاگت, &t1)) أرجع -1;
    إذا (hex_mul_حد_توں_ودھ(بہترین_m, بہترین_n, &mn))     أرجع -1;
    إذا (hex_mul_حد_توں_ودھ(mn, فی_mn_لاگت, &t2))    أرجع -1;
    إذا (hex_جوڑو_حد_توں_ودھ(t0, t1, &کل))          أرجع -1;
    إذا (hex_جوڑو_حد_توں_ودھ(کل, t2, &کل))       أرجع -1;
    إذا (hex_جوڑو_حد_توں_ودھ(کل, overhead, &کل)) أرجع -1;

    *m_chunk_باہرلا = بہترین_m;
    *n_chunk_باہرلا = بہترین_n;
    *کل_باہرلا   = کل;
    أرجع 0;
}

// weight types whose tiles carry a per-block offset (x = d * q + m): the activations need block sums (q8_1)
// (Q2_K: per-16 k sums, q8_1_s16)
ساكن مضمن bool htp_mm_بھار_has_وتھ(صحيح بھار_type) {
    أرجع بھار_type == HTP_TYPE_Q4_1 || بھار_type == HTP_TYPE_Q4_K || بھار_type == HTP_TYPE_Q5_K ||
           بھار_type == HTP_TYPE_Q2_K || بھار_type == HTP_TYPE_Q2_0;
}

// --- Activation/Row Size Helpers ---
ساكن مضمن size_t htp_mm_q8_0_tiled_صف_ناپ(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    أرجع nb_32 * HTP_MM_ACT_TILE_ناپ_Q8_0;
}

ساكن مضمن size_t htp_mm_q8_1_tiled_صف_ناپ(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    أرجع nb_32 * HTP_MM_ACT_TILE_ناپ_Q8_1;
}

ساكن مضمن size_t htp_mm_f16_tiled_صف_ناپ(uint32_t ne) {
    ثابت uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ثابت uint32_t nb_32 = ne_padded / 32;
    أرجع nb_32 * HTP_MM_ACT_TILE_ناپ_F16;
}

ساكن مضمن size_t htp_mm_tiled_act_صف_ناپ(صحيح بھار_type, uint32_t ne) {
    إذا (بھار_type == HTP_TYPE_F16) {
        أرجع htp_mm_f16_tiled_صف_ناپ(ne);
    }
    أرجع htp_mm_بھار_has_وتھ(بھار_type) ? htp_mm_q8_1_tiled_صف_ناپ(ne) : htp_mm_q8_0_tiled_صف_ناپ(ne);
}

ساكن مضمن size_t htp_mm_گیڑا_اتے(size_t n, size_t m) {
    أرجع ((n + m - 1) / m) * m;
}

ساكن مضمن bool htp_mm_hmx_pipeline(uint32_t m) {
    أرجع m > 32;
}

ساكن مضمن فراغ htp_mm_hmx_get_2d_chunk_costs(
    صحيح wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_ناپ,
    size_t * ناپ_فی_n_باہرلا, size_t * ناپ_فی_m_باہرلا, size_t * ناپ_فی_mn_باہرلا
) {
    ثابت bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
    ثابت size_t صف_stride = htp_tensor_get_صف_ناپ(wtype, k);
    ثابت size_t vec_dot_ناپ = k * حجم(uint16_t);
    ثابت uint32_t n_بھار_tiles = k / htp_tensor_get_tile_k(wtype);
    ثابت size_t qweight_صف_stride = is_quant ? (size_t)(n_بھار_tiles * aligned_tile_ناپ) / 32 : 0;

    *ناپ_فی_n_باہرلا = (pipeline ? 2 : 1) * (is_quant ? qweight_صف_stride : صف_stride) +
                      (pipeline ? 2 * vec_dot_ناپ : vec_dot_ناپ);
    *ناپ_فی_m_باہرلا = vec_dot_ناپ;
    *ناپ_فی_mn_باہرلا = (pipeline ? 2 : 1) * حجم(uint16_t);
}

ساكن مضمن فراغ htp_mm_hmx_get_batched_chunk_costs(
    uint32_t k, uint32_t گروہ_ناپ,
    size_t * ناپ_فی_n_باہرلا, size_t * ناپ_فی_m_باہرلا, size_t * ناپ_فی_mn_باہرلا
) {
    ثابت size_t vec_dot_ناپ = k * حجم(uint16_t);
    *ناپ_فی_n_باہرلا = 3 * vec_dot_ناپ;
    *ناپ_فی_m_باہرلا = گروہ_ناپ * vec_dot_ناپ;
    *ناپ_فی_mn_باہرلا = حجم(uint16_t);
}

ساكن مضمن size_t htp_mm_hmx_get_2d_overhead(bool pipeline, bool is_matmul_id) {
    size_t عدد_regions = pipeline ? 7 : (is_matmul_id ? 4 : 5);
    أرجع عدد_regions * HTP_MM_HMX_TILE_ناپ + 256;
}

ساكن مضمن size_t htp_mm_hmx_get_batched_overhead(فراغ) {
    أرجع 5 * HTP_MM_HMX_TILE_ناپ + 256;
}

بنية htp_mm_hmx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_بھار[2];     // [1] is only used when pipelined
    size_t off_act;
    size_t off_act_f32;       // fp32 activation conversion scratch
    size_t off_dst[2];        // [1] is only used when pipelined
    size_t off_scratch[2];    // dequantization scratch pads
    size_t off_scales;        // HMX scales (256 bytes)
    size_t off_bias;          // bias in VTCM

    // Cached sizes of regions for HMX kernel use
    size_t بھار_area_بائٹاں;
    size_t act_area_بائٹاں;
    size_t act_f32_بائٹاں;
    size_t output_area_بائٹاں;
    size_t scratch_بائٹاں[2];
    size_t act_سر_stride;
    size_t bias_بائٹاں;

    size_t کل_بائٹاں;
};

بنية htp_mm_hvx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_act;           // vtcm_act (activation)
    size_t off_src0;          // vtcm_src0 (weight/Wk)
    size_t off_bias;          // vtcm_bias (bias / fused add only)
    size_t off_dst;           // vtcm_dst (output scratch)
    size_t off_act_raw;       // vtcm_act_raw (raw activation DMA staging)

    // Cached sizes
    size_t src0_بائٹاں;
    size_t act_بائٹاں;
    size_t bias_بائٹاں;
    size_t dst_بائٹاں;
    size_t act_raw_بائٹاں;

    size_t کل_بائٹاں;
};

ساكن مضمن فراغ htp_mm_hmx_vtcm_layout_build(
    بنية htp_mm_hmx_vtcm_layout * L,
    صحيح kernel_type,
    صحيح wtype,
    uint32_t k,
    size_t mc,
    size_t nc,
    uint32_t گروہ_ناپ,
    bool pipeline,
    uint32_t act_دھاگے,
    uint32_t aligned_tile_ناپ,
    size_t bias_ناپ
) {
    size_t off = 0;

    إذا (kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        ثابت size_t vec_dot_ناپ     = k * حجم(uint16_t);
        ثابت size_t act_سر_stride   = mc * k;
        ثابت size_t بھار_area_ناپ  = hex_align_اتے(nc * vec_dot_ناپ, HTP_MM_HMX_TILE_ناپ);
        ثابت size_t activation_area_ناپ = hex_align_اتے(گروہ_ناپ * act_سر_stride * حجم(uint16_t), HTP_MM_HMX_TILE_ناپ);
        ثابت size_t output_area_ناپ  = hex_align_اتے(گروہ_ناپ * mc * nc * حجم(uint16_t), HTP_MM_HMX_TILE_ناپ);
        ثابت size_t scratch_area_ناپ = hex_align_اتے(nc * vec_dot_ناپ, HTP_MM_HMX_TILE_ناپ);
        ثابت size_t min_f32_ناپ = hex_align_اتے(act_دھاگے * HTP_MM_DMA_ACT_MULTIPLIER * k * حجم(عائم), 128);

        // Group A: Permanent activation tiles and scales
        size_t off_گروہ_a = 0;
        VTCM_LAYOUT_ALLOC(off_گروہ_a, off_act, activation_area_ناپ);
        VTCM_LAYOUT_ALLOC(off_گروہ_a, off_scales, HTP_MM_HMX_TILE_ناپ); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_a, off_bias, hex_align_اتے(bias_ناپ, HTP_MM_HMX_TILE_ناپ), bias_ناپ > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_گروہ_b = off_گروہ_a;
        VTCM_LAYOUT_ALLOC(off_گروہ_b, off_بھار[0], بھار_area_ناپ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_بھار[1], بھار_area_ناپ, false);
        VTCM_LAYOUT_ALLOC(off_گروہ_b, off_dst[0], output_area_ناپ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_dst[1], output_area_ناپ, false);
        VTCM_LAYOUT_ALLOC(off_گروہ_b, off_scratch[0], scratch_area_ناپ);
        VTCM_LAYOUT_ALLOC(off_گروہ_b, off_scratch[1], scratch_area_ناپ);

        ثابت size_t گروہ_b_ناپ = off_گروہ_b - off_گروہ_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ثابت size_t max_f32_ناپ = act_دھاگے * 64 * k * حجم(عائم);
        ثابت size_t act_f32_ناپ = hex_align_اتے(hex_smin(max_f32_ناپ, hex_smax(min_f32_ناپ, گروہ_b_ناپ)), 128);
        size_t off_گروہ_c = off_گروہ_a;
        VTCM_LAYOUT_ALLOC(off_گروہ_c, off_act_f32, act_f32_ناپ);

        ثابت size_t گروہ_c_ناپ = off_گروہ_c - off_گروہ_a;

        L->بھار_area_بائٹاں = بھار_area_ناپ;
        L->act_area_بائٹاں    = activation_area_ناپ;
        L->act_f32_بائٹاں     = act_f32_ناپ;
        L->output_area_بائٹاں = output_area_ناپ;
        L->scratch_بائٹاں[0]  = scratch_area_ناپ;
        L->scratch_بائٹاں[1]  = scratch_area_ناپ;
        L->act_سر_stride   = act_سر_stride;
        L->bias_بائٹاں        = bias_ناپ;

        off = off_گروہ_a + hex_smax(گروہ_b_ناپ, گروہ_c_ناپ);
    } وإلا {
        // HTP_MM_KERNEL_HMX_2D
        ثابت bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
        ثابت size_t صف_stride = htp_tensor_get_صف_ناپ(wtype, k);
        ثابت size_t vec_dot_ناپ = k * حجم(uint16_t);
        ثابت uint32_t n_بھار_tiles = k / htp_tensor_get_tile_k(wtype);

        ثابت size_t min_f32_ناپ = hex_align_اتے(act_دھاگے * HTP_MM_DMA_ACT_MULTIPLIER * k * حجم(عائم), 128);
        ثابت size_t بھار_area_ناپ = is_quant
            ? hex_align_اتے((nc / 32) * n_بھار_tiles * aligned_tile_ناپ, HTP_MM_HMX_TILE_ناپ)
            : hex_align_اتے(nc * صف_stride, HTP_MM_HMX_TILE_ناپ);
        ثابت size_t act_area_ناپ    = hex_align_اتے(mc * vec_dot_ناپ, HTP_MM_HMX_TILE_ناپ);
        ثابت size_t output_area_ناپ = hex_align_اتے(mc * nc * حجم(__fp16), HTP_MM_HMX_TILE_ناپ);

        ثابت size_t scratch0_ناپ = hex_align_اتے(nc * vec_dot_ناپ, HTP_MM_HMX_TILE_ناپ);
        ثابت size_t scratch1_ناپ = pipeline ? scratch0_ناپ : 0;

        // Group A:  Scales and activation tiles (must not overlap with Group B or C)
        size_t off_گروہ_a = 0;
        VTCM_LAYOUT_ALLOC(off_گروہ_a, off_scales, HTP_MM_HMX_TILE_ناپ); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC(off_گروہ_a, off_act, act_area_ناپ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_a, off_bias, hex_align_اتے(bias_ناپ, HTP_MM_HMX_TILE_ناپ), bias_ناپ > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_گروہ_b = off_گروہ_a;
        VTCM_LAYOUT_ALLOC(off_گروہ_b, off_بھار[0], بھار_area_ناپ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_بھار[1], بھار_area_ناپ, pipeline);
        VTCM_LAYOUT_ALLOC(off_گروہ_b, off_dst[0], output_area_ناپ);
        VTCM_LAYOUT_ALLOC(off_گروہ_b, off_scratch[0], scratch0_ناپ);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_scratch[1], scratch0_ناپ, pipeline);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_dst[1], output_area_ناپ, pipeline);

        ثابت size_t گروہ_b_ناپ = off_گروہ_b - off_گروہ_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ثابت size_t max_f32_ناپ = act_دھاگے * 64 * k * حجم(عائم);
        ثابت size_t act_f32_ناپ = hex_align_اتے(hex_smin(max_f32_ناپ, hex_smax(min_f32_ناپ, گروہ_b_ناپ)), 128);
        size_t off_گروہ_c = off_گروہ_a;
        VTCM_LAYOUT_ALLOC(off_گروہ_c, off_act_f32, act_f32_ناپ);

        ثابت size_t گروہ_c_ناپ = off_گروہ_c - off_گروہ_a;

        L->بھار_area_بائٹاں = بھار_area_ناپ;
        L->act_area_بائٹاں    = act_area_ناپ;
        L->act_f32_بائٹاں     = act_f32_ناپ;
        L->output_area_بائٹاں = output_area_ناپ;
        L->scratch_بائٹاں[0]  = scratch0_ناپ;
        L->scratch_بائٹاں[1]  = scratch1_ناپ;
        L->act_سر_stride   = 0;
        L->bias_بائٹاں        = bias_ناپ;

        off = off_گروہ_a + hex_smax(گروہ_b_ناپ, گروہ_c_ناپ);
    }

    L->کل_بائٹاں = off;
}

ساكن مضمن فراغ htp_mm_hvx_vtcm_layout_build(
    بنية htp_mm_hvx_vtcm_layout * L,
    صحيح kernel_type,
    صحيح wtype,
    uint32_t ne10,       // k
    uint32_t act_nrows,  // m_total
    uint32_t n_دھاگے,
    size_t dst_صف_ناپ,
    size_t src0_صف_ناپ,
    size_t act_صف_ناپ,
    size_t bias_صف_ناپ,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_nx
) {
    (فراغ)act_صف_ناپ;
    size_t src0_sz    = 0;
    size_t act_sz     = 0;
    size_t bias_sz    = bias_صف_ناپ > 0 ? htp_mm_گیڑا_اتے(bias_صف_ناپ, 128) : 0;
    size_t dst_sz     = 0;
    size_t act_raw_sz = 0;

    إذا (is_fused_nx) {
        uint32_t aligned_tile_ناپ = htp_tensor_get_tile_padded_ناپ(wtype);
        uint32_t n_k_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        uint32_t tile_صف_ناپ = n_k_tiles * aligned_tile_ناپ;
        size_t بھار_sz_فی_thread = hex_گیڑا_اتے(n_prefetch * tile_صف_ناپ, 128);

        size_t tiled_act_صف_ناپ = htp_mm_tiled_act_صف_ناپ(wtype, ne10);
        size_t q_act_sz = hex_گیڑا_اتے(tiled_act_صف_ناپ * act_nrows, 128);
        size_t raw_صف_ناپ = hex_گیڑا_اتے(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

        src0_sz    = بھار_sz_فی_thread * n_دھاگے; // shared single-weight prefetch buffer
        act_sz     = q_act_sz;                         // quantized activation buffer
        bias_sz    = 0;
        dst_sz     = 0;
        act_raw_sz = hex_گیڑا_اتے(raw_صف_ناپ * act_nrows, 128);
    } وإلا إذا (is_matmul_id) {
        ثابت size_t act_صف_ناپ_tiled = htp_mm_tiled_act_صف_ناپ(wtype, ne10);
        ثابت uint32_t aligned_tile_ناپ = htp_tensor_get_tile_padded_ناپ(wtype);
        ثابت uint32_t n_k_tiles         = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        ثابت uint32_t tile_صف_ناپ     = n_k_tiles * aligned_tile_ناپ;
        size_t src0_sz_فی_thread        = htp_mm_گیڑا_اتے(n_prefetch * tile_صف_ناپ, 256);

        act_sz = htp_mm_گیڑا_اتے(act_صف_ناپ_tiled * act_nrows, 256);

        size_t raw_صف_ناپ = hex_گیڑا_اتے(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));

        src0_sz    = src0_sz_فی_thread * n_دھاگے;
        dst_sz     = 0;
        bias_sz    = 0;
        act_raw_sz = hex_گیڑا_اتے(raw_صف_ناپ * act_nrows, 128);
    } وإلا {
        ثابت size_t src0_صف_ناپ_padded = htp_mm_گیڑا_اتے(src0_صف_ناپ, 128);
        ثابت size_t dst_nrows = (act_nrows > 1) ? 0 : 1;

        بدل (kernel_type) {
            حالة HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                size_t f16_act_صف_ناپ = htp_mm_گیڑا_اتے(ne10 * 2, 128);
                act_sz     = htp_mm_گیڑا_اتے(f16_act_صف_ناپ * act_nrows, 256);
                src0_sz    = htp_mm_گیڑا_اتے(n_prefetch * src0_صف_ناپ_padded, 256) * n_دھاگے;
                dst_sz     = dst_nrows > 0 ? htp_mm_گیڑا_اتے(dst_صف_ناپ, 128) * n_دھاگے : 0;
                act_raw_sz = hex_گیڑا_اتے(hex_گیڑا_اتے(ne10 * حجم(عائم), 128) * act_nrows, 128);
                اكسر;
            }
            حالة HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            حالة HTP_MM_KERNEL_HVX_BF16_F32_VTCM: {
                size_t f32_act_صف_ناپ = htp_mm_گیڑا_اتے(ne10 * 4, 128);
                act_sz     = htp_mm_گیڑا_اتے(f32_act_صف_ناپ * act_nrows, 256);
                src0_sz    = htp_mm_گیڑا_اتے(n_prefetch * src0_صف_ناپ_padded, 256) * n_دھاگے;
                dst_sz     = dst_nrows > 0 ? htp_mm_گیڑا_اتے(dst_صف_ناپ, 128) * n_دھاگے : 0;
                act_raw_sz = 0;
                اكسر;
            }
            حالة HTP_MM_KERNEL_HVX_QUANT_بلاک:
            حالة HTP_MM_KERNEL_HVX_QUANT_صف: {
                size_t q_act_صف_ناپ = htp_mm_tiled_act_صف_ناپ(wtype, ne10);
                uint32_t aligned_tile_ناپ = htp_tensor_get_tile_padded_ناپ(wtype);
                uint32_t n_بھار_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
                uint32_t tile_صف_ناپ = n_بھار_tiles * aligned_tile_ناپ;
                size_t tiled_vtcm_ناپ = htp_mm_گیڑا_اتے(n_prefetch * tile_صف_ناپ, 256);

                src0_sz = tiled_vtcm_ناپ * n_دھاگے;
                act_sz  = htp_mm_گیڑا_اتے(q_act_صف_ناپ * act_nrows, 256);

                size_t dst_ٹکڑا_فی_thread = (dst_nrows > 0 && act_nrows == 1) ? htp_mm_گیڑا_اتے((dst_صف_ناپ + n_دھاگے - 1) / n_دھاگے, 128) : 0;
                dst_sz = dst_ٹکڑا_فی_thread * n_دھاگے;
                size_t raw_صف_ناپ = hex_گیڑا_اتے(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));
                act_raw_sz = hex_گیڑا_اتے(raw_صف_ناپ * act_nrows, 128);
                اكسر;
            }
            افتراضي:
                اكسر;
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
    ثابت size_t گروہ_b_ناپ = off_گروہ_b - off_گروہ_a;

    // Group C: Raw activation staging buffer (overlaps Group B, starts at off_group_a)
    size_t off_گروہ_c = off_گروہ_a;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_c, off_act_raw, act_raw_sz, act_raw_sz > 0);
    ثابت size_t گروہ_c_ناپ = off_گروہ_c - off_گروہ_a;

    L->src0_بائٹاں    = src0_sz;
    L->act_بائٹاں     = act_sz;
    L->bias_بائٹاں    = bias_sz;
    L->dst_بائٹاں     = dst_sz;
    L->act_raw_بائٹاں = act_raw_sz;
    L->کل_بائٹاں   = off_گروہ_a + hex_smax(گروہ_b_ناپ, گروہ_c_ناپ);
}

ساكن مضمن bool htp_mm_hvx_solve_vtcm_params(
    صحيح kernel_type,
    صحيح wtype,
    uint32_t ne10,
    uint32_t act_nrows,
    uint32_t n_دھاگے,
    size_t dst_صف_ناپ,
    size_t src0_صف_ناپ,
    size_t act_صف_ناپ,
    size_t bias_صف_ناپ,
    uint32_t n_prefetch,
    size_t vtcm_budget,
    بنية htp_mm_hvx_vtcm_layout * L_باہرلا,
    uint32_t * m_chunk_باہرلا
) {
    بنية htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, act_nrows, n_دھاگے,
        dst_صف_ناپ, src0_صف_ناپ, act_صف_ناپ, bias_صف_ناپ, n_prefetch, false, false
    );

    إذا (L.کل_بائٹاں <= vtcm_budget) {
        *L_باہرلا = L;
        *m_chunk_باہرلا = act_nrows;
        أرجع true;
    }

    ثابت size_t fixed_بائٹاں = L.src0_بائٹاں + L.bias_بائٹاں + L.dst_بائٹاں;
    إذا (vtcm_budget <= fixed_بائٹاں) {
        أرجع false;
    }

    ثابت size_t avail_act = vtcm_budget - fixed_بائٹاں;
    size_t صف_ناپ = 0;
    إذا (kernel_type == HTP_MM_KERNEL_HVX_QUANT_صف || kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک) {
        صف_ناپ = htp_mm_tiled_act_صف_ناپ(wtype, ne10);
    } وإلا إذا (kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        صف_ناپ = hex_گیڑا_اتے(ne10 * 2, 128);
    } وإلا {
        صف_ناپ = hex_گیڑا_اتے(ne10 * 4, 128);
    }
    إذا (صف_ناپ == 0) {
        أرجع false;
    }

    size_t eff_صف_ناپ = صف_ناپ;
    إذا (kernel_type == HTP_MM_KERNEL_HVX_QUANT_صف || kernel_type == HTP_MM_KERNEL_HVX_QUANT_بلاک) {
        eff_صف_ناپ += hex_گیڑا_اتے(ne10 * حجم(عائم), QK_Q8_0_TILED * حجم(عائم));
    }

    uint32_t m_chunk = (uint32_t) (avail_act / eff_صف_ناپ);
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
        &L, kernel_type, wtype, ne10, m_chunk, n_دھاگے,
        dst_صف_ناپ, src0_صف_ناپ, act_صف_ناپ, bias_صف_ناپ, n_prefetch, false, false
    );

    طالما (m_chunk > 2 && L.کل_بائٹاں > vtcm_budget) {
        m_chunk -= 2;
        htp_mm_hvx_vtcm_layout_build(
            &L, kernel_type, wtype, ne10, m_chunk, n_دھاگے,
            dst_صف_ناپ, src0_صف_ناپ, act_صف_ناپ, bias_صف_ناپ, n_prefetch, false, false
        );
    }

    إذا (L.کل_بائٹاں <= vtcm_budget) {
        *L_باہرلا = L;
        *m_chunk_باہرلا = m_chunk;
        أرجع true;
    }

    أرجع false;
}

ساكن مضمن size_t htp_mm_hmx_get_2d_vtcm_ناپ(
    صحيح wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_دھاگے, uint32_t aligned_tile_ناپ, size_t bias_ناپ
) {
    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_دھاگے, aligned_tile_ناپ, bias_ناپ);
    أرجع L.کل_بائٹاں;
}

ساكن مضمن size_t htp_mm_hmx_get_batched_vtcm_ناپ(
    صحيح wtype, uint32_t k, size_t mc, size_t nc, uint32_t گروہ_ناپ, bool pipeline, uint32_t act_دھاگے, size_t bias_ناپ) {
    (فراغ)pipeline;
    بنية htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, wtype, k, mc, nc, گروہ_ناپ, false, act_دھاگے, 0, bias_ناپ);
    أرجع L.کل_بائٹاں;
}

ساكن مضمن bool htp_mm_hmx_solve_batched_params(
    صحيح wtype,
    uint32_t k,
    uint32_t ne01_padded,
    uint32_t ne11,
    uint32_t گروہ_ناپ,
    صحيح n_دھاگے,
    bool pipeline,
    size_t bias_ناپ,
    size_t vtcm_budget,
    size_t * m_chunk_باہرلا,
    size_t * n_chunk_باہرلا,
    صحيح * act_دھاگے_باہرلا,
    size_t * vtcm_ناپ_باہرلا
) {
    size_t بہترین_mblocks = SIZE_MAX;
    صحيح بہترین_act_دھاگے = 0;
    size_t بہترین_m_chunk = 0;
    size_t بہترین_n_chunk = 0;
    size_t بہترین_vtcm_ناپ = 0;

    صحيح act_دھاگے = n_دھاگے;
    طالما (act_دھاگے >= 1) {
        size_t گروہ_overhead = htp_mm_hmx_get_batched_overhead() + (bias_ناپ > 0 ? hex_align_اتے(bias_ناپ, HTP_MM_HMX_TILE_ناپ) : 0);
        size_t گروہ_ناپ_فی_n, گروہ_ناپ_فی_m, گروہ_ناپ_فی_mn;
        htp_mm_hmx_get_batched_chunk_costs(k, گروہ_ناپ, &گروہ_ناپ_فی_n, &گروہ_ناپ_فی_m, &گروہ_ناپ_فی_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_ناپ_candidate = 0;

        إذا (htp_mm_hmx_compute_chunks(vtcm_budget, گروہ_overhead, گروہ_ناپ_فی_n, گروہ_ناپ_فی_m, گروہ_ناپ_فی_mn, hex_align_اتے(ne11, 32), ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_لاگت_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_لاگت_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_ناپ_candidate) == 0) {
            size_t exact_ناپ = htp_mm_hmx_get_batched_vtcm_ناپ(wtype, k, m_chunk_candidate, n_chunk_candidate, گروہ_ناپ, pipeline, act_دھاگے, bias_ناپ);
            إذا (exact_ناپ <= vtcm_budget) {
                size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                إذا (mblocks < بہترین_mblocks || (mblocks == بہترین_mblocks && act_دھاگے > بہترین_act_دھاگے)) {
                    بہترین_mblocks = mblocks;
                    بہترین_act_دھاگے = act_دھاگے;
                    بہترین_m_chunk = m_chunk_candidate;
                    بہترین_n_chunk = n_chunk_candidate;
                    بہترین_vtcm_ناپ = exact_ناپ;
                }
            }
        }
        إذا (act_دھاگے == 1) {
            act_دھاگے = 0;
        } وإلا {
            act_دھاگے /= 2;
        }
    }

    إذا (بہترین_act_دھاگے > 0) {
        *m_chunk_باہرلا = بہترین_m_chunk;
        *n_chunk_باہرلا = بہترین_n_chunk;
        *vtcm_ناپ_باہرلا = بہترین_vtcm_ناپ;
        *act_دھاگے_باہرلا = بہترین_act_دھاگے;
        أرجع true;
    }
    أرجع false;
}

ساكن مضمن bool htp_mm_hmx_solve_2d_params(
    صحيح wtype,
    uint32_t k,
    uint32_t m_id_صفاں,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_لاگت,
    صحيح n_دھاگے,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_ناپ,
    size_t bias_ناپ,
    size_t vtcm_budget,
    size_t * m_chunk_باہرلا,
    size_t * n_chunk_باہرلا,
    صحيح * act_دھاگے_باہرلا,
    size_t * vtcm_ناپ_باہرلا
) {
    size_t بہترین_mblocks = SIZE_MAX;
    صحيح بہترین_act_دھاگے = 0;
    size_t بہترین_m_chunk = 0;
    size_t بہترین_n_chunk = 0;
    size_t بہترین_vtcm_ناپ = 0;

    ثابت صحيح m_for_chunks = is_matmul_id ? hex_align_اتے(m_id_صفاں, 32) : ne11_padded;

    صحيح act_دھاگے = n_دھاگے;
    طالما (act_دھاگے >= 1) {
        size_t simple_2d_overhead = htp_mm_hmx_get_2d_overhead(pipeline, is_matmul_id) + (bias_ناپ > 0 ? hex_align_اتے(bias_ناپ, HTP_MM_HMX_TILE_ناپ) : 0);
        size_t simple_2d_ناپ_فی_n, simple_2d_ناپ_فی_m, simple_2d_ناپ_فی_mn;
        htp_mm_hmx_get_2d_chunk_costs(wtype, k, pipeline, aligned_tile_ناپ, &simple_2d_ناپ_فی_n, &simple_2d_ناپ_فی_m, &simple_2d_ناپ_فی_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_ناپ_candidate = 0;

        إذا (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_ناپ_فی_n, simple_2d_ناپ_فی_m, simple_2d_ناپ_فی_mn, m_for_chunks, ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_لاگت_W_DEQUANT, (size_t) m_for_لاگت * HTP_MM_HMX_لاگت_A_CONVERT,
                               &m_chunk_candidate, &n_chunk_candidate, &vtcm_ناپ_candidate) == 0) {
            size_t exact_ناپ = htp_mm_hmx_get_2d_vtcm_ناپ(wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_دھاگے, aligned_tile_ناپ, bias_ناپ);
            إذا (exact_ناپ <= vtcm_budget) {
                size_t mblocks = ((size_t) m_for_لاگت + m_chunk_candidate - 1) / m_chunk_candidate;
                إذا (mblocks < بہترین_mblocks || (mblocks == بہترین_mblocks && act_دھاگے > بہترین_act_دھاگے)) {
                    بہترین_mblocks = mblocks;
                    بہترین_act_دھاگے = act_دھاگے;
                    بہترین_m_chunk = m_chunk_candidate;
                    بہترین_n_chunk = n_chunk_candidate;
                    بہترین_vtcm_ناپ = exact_ناپ;
                }
            }
        }
        إذا (act_دھاگے == 1) {
            act_دھاگے = 0;
        } وإلا {
            act_دھاگے /= 2;
        }
    }

    إذا (بہترین_act_دھاگے > 0) {
        *m_chunk_باہرلا = بہترین_m_chunk;
        *n_chunk_باہرلا = بہترین_n_chunk;
        *vtcm_ناپ_باہرلا = بہترین_vtcm_ناپ;
        *act_دھاگے_باہرلا = بہترین_act_دھاگے;
        أرجع true;
    }
    أرجع false;
}

#جے_تعریف_شدہ __cplusplus
}
#جے_انت

#جے_انت // HTP_MATMUL_OPS_H
