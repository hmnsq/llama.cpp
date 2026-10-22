#ifndef HTP_MATMUL_OPS_H
#define HTP_MATMUL_OPS_H

#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <stddef.h>
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"
#অন্তর্ভাব "hex-fastdiv.h"
#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "htp-vtcm.h"

#ifdef __cplusplus
বাহ্যিক "C" {
#endif

// --- HMX Tile Constraints ---
#define HTP_MM_HMX_TILE_N_COLS 32
#define HTP_MM_HMX_TILE_N_ROWS 32
#define HTP_MM_HMX_TILE_আকার   (32 * 32 * আকার(__fp16)) // 2048 bytes
#define HTP_MM_HMX_TILE_N_ELMS 1024
#define HTP_MM_HMX_MIN_NROWS   4

#define HTP_MM_WEIGHT_TILE_K_Q2_0             64
#define HTP_MM_WEIGHT_TILE_আকার_Q2_0          HTP_TENSOR_TILE_আকার_Q2_0
#define HTP_MM_WEIGHT_ALIGNED_TILE_আকার_Q2_0  640

// --- Activation Tiled Block Sizes (including padding) ---
#define HTP_MM_ACT_TILE_আকার_Q8_0      1152
#define HTP_MM_ACT_TILE_আকার_Q8_1      1280
#define HTP_MM_ACT_TILE_আকার_F16       2048

#define HTP_MM_MAX_PREFETCH 16

// --- Solver Cost Model Penalty Weights (HMX-specific) ---
#define HTP_MM_HMX_COST_W_DEQUANT 3 // cost penalty for quantized weight loading/dequantization
#define HTP_MM_HMX_COST_A_CONVERT 2 // cost penalty for activation loading/conversion

// --- DMA Activation Transfer Configuration ---
#define HTP_MM_DMA_ACT_ROWS_PER_STEP 2
#define HTP_MM_DMA_ACT_MULTIPLIER    (2 * HTP_MM_DMA_ACT_ROWS_PER_STEP)

গণনা htp_mm_kernel_type {
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
    HTP_MM_KERNEL_HVX_QUANT_BLOCK,    // parallel block-wise quantization
};

// Op-specific struct for precomputed matmul params
কাঠামো htp_mm_kernel_params {
    uint8_t  kernel_type;        // enum htp_mm_kernel_type
    uint8_t  pipeline;           // 1 = pipelined execution, 0 = standard
    uint8_t  collapse;           // 1 = collapse outer dims into 2D, 0 = standard
    uint8_t  n_hmx;              // 1 = use HMX, 0 = use HVX

    uint8_t  n_threads;          // Number of threads to spawn
    uint8_t  n_act_threads;      // Number of threads for activation preparation
    uint8_t  n_prefetch;         // Prefetch lookahead buffers/rows in VTCM
    uint8_t  n_weights;          // Number of weights for fused NX

    int32_t  m_chunk;            // Row chunk size (M chunk)
    int32_t  n_chunk;            // Col chunk size (N chunk)
    int32_t  tile_আকার;          // Weight tile size
    int32_t  aligned_tile_আকার;  // Aligned weight tile size (padded to 128)
    int32_t  act_row_আকার;       // Row size for activation scratchpad
    int32_t  vtcm_আকার;          // Total required scratchpad size in VTCM
    int32_t  vtcm_src0_আকার;     // src0 scratchpad size in VTCM
    int32_t  vtcm_act_আকার;      // activation scratchpad size in VTCM
    int32_t  vtcm_bias_আকার;     // bias scratchpad size in VTCM (fused only)
    int32_t  vtcm_dst_আকার;      // dst scratchpad size in VTCM

    // Precomputed division values
    কাঠামো fastdiv_মানসমূহ div_ne12_ne1;
    কাঠামো fastdiv_মানসমূহ div_ne1;
    কাঠামো fastdiv_মানসমূহ div_r2;
    কাঠামো fastdiv_মানসমূহ div_r3;
    কাঠামো fastdiv_মানসমূহ div_ne12;
    কাঠামো fastdiv_মানসমূহ div_n_act_threads;
    কাঠামো fastdiv_মানসমূহ div_ne00_padded;
};

#যদি defined(__cplusplus)
static_assert(আকার(কাঠামো htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#অন্যথা
_Static_assert(আকার(কাঠামো htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#endif

কাঠামো mmid_row_mapping {
    uint32_t i1;
    uint32_t i2;
};

// Search for optimal (mc, nc) chunk sizes within VTCM budget.
স্থির অন্তঃরেখ পূর্ণসংখ্যা htp_mm_hmx_compute_chunks(size_t   vtcm_যোগ,
                              size_t   overhead,
                              size_t   per_n_cost,
                              size_t   per_m_cost,
                              size_t   per_mn_cost,
                              size_t   m,
                              size_t   n,
                              size_t   m_block_cost,
                              size_t   n_block_cost,
                              size_t * m_chunk_out,
                              size_t * n_chunk_out,
                              size_t * যোগ_out) {
    যদি (m == 0 || n == 0) ফেরত -1;
    যদি (vtcm_যোগ <= overhead) ফেরত -1;
    যদি (per_n_cost == 0 || per_m_cost == 0 || per_mn_cost == 0) ফেরত -1;

    ধ্রুবক size_t usable = vtcm_যোগ - overhead;

    size_t best_cost = SIZE_MAX;
    size_t best_লেজ_waste = SIZE_MAX;
    size_t best_mn   = 0;
    size_t best_m = 0, best_n = 0;

    ধ্রুবক size_t max_nc_budget = (usable / per_n_cost);
    ধ্রুবক size_t n_max = hex_align_down(hex_smin((size_t)n, max_nc_budget), HTP_MM_HMX_TILE_N_COLS);
    জন্য (size_t nc = n_max; nc >= HTP_MM_HMX_TILE_N_COLS; nc -= HTP_MM_HMX_TILE_N_COLS) {
        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        যদি (hex_mul_overflow(nc, per_n_cost, &n_fixed)) চালু;
        যদি (n_fixed >= usable) যাও পরবর্তী_nc;

        যদি (hex_mul_overflow(nc, per_mn_cost, &ncmn)) যাও পরবর্তী_nc;
        যদি (hex_add_overflow(per_m_cost, ncmn, &mc_denom) || mc_denom == 0) যাও পরবর্তী_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hex_align_down(mc, HTP_MM_HMX_TILE_N_ROWS);
            mc = hex_smin(mc, m);

            যদি (mc == 0) {
                যাও পরবর্তী_nc;
            }

            size_t mblocks = ((size_t) m + mc - 1) / mc;
            size_t nblocks = ((size_t) n + nc - 1) / nc;
            size_t cost    = mblocks * m_block_cost + nblocks * n_block_cost;
            size_t rem     = n % nc;
            size_t লেজ_waste = (rem == 0) ? 0 : (nc - rem);
            size_t mn      = mc * nc;
            যদি (cost < best_cost ||
                (cost == best_cost && লেজ_waste < best_লেজ_waste) ||
                (cost == best_cost && লেজ_waste == best_লেজ_waste && mn > best_mn)) {
                best_cost       = cost;
                best_লেজ_waste = লেজ_waste;
                best_mn         = mn;
                best_m          = mc;
                best_n          = nc;
            }
        }

পরবর্তী_nc:
        যদি (nc == HTP_MM_HMX_TILE_N_COLS) বিরতি;  // avoid size_t underflow
    }

    যদি (best_m == 0 || best_n == 0) ফেরত -1;

    // Compute exact total (with overflow checks)
    size_t t0 = 0, t1 = 0, t2 = 0, mn = 0, যোগ = 0;
    যদি (hex_mul_overflow(best_n, per_n_cost, &t0)) ফেরত -1;
    যদি (hex_mul_overflow(best_m, per_m_cost, &t1)) ফেরত -1;
    যদি (hex_mul_overflow(best_m, best_n, &mn))     ফেরত -1;
    যদি (hex_mul_overflow(mn, per_mn_cost, &t2))    ফেরত -1;
    যদি (hex_add_overflow(t0, t1, &যোগ))          ফেরত -1;
    যদি (hex_add_overflow(যোগ, t2, &যোগ))       ফেরত -1;
    যদি (hex_add_overflow(যোগ, overhead, &যোগ)) ফেরত -1;

    *m_chunk_out = best_m;
    *n_chunk_out = best_n;
    *যোগ_out   = যোগ;
    ফেরত 0;
}

// weight types whose tiles carry a per-block offset (x = d * q + m): the activations need block sums (q8_1)
// (Q2_K: per-16 k sums, q8_1_s16)
স্থির অন্তঃরেখ bool htp_mm_weight_has_offset(পূর্ণসংখ্যা weight_type) {
    ফেরত weight_type == HTP_TYPE_Q4_1 || weight_type == HTP_TYPE_Q4_K || weight_type == HTP_TYPE_Q5_K ||
           weight_type == HTP_TYPE_Q2_K || weight_type == HTP_TYPE_Q2_0;
}

// --- Activation/Row Size Helpers ---
স্থির অন্তঃরেখ size_t htp_mm_q8_0_tiled_row_আকার(uint32_t ne) {
    ধ্রুবক uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ধ্রুবক uint32_t nb_32 = ne_padded / 32;
    ফেরত nb_32 * HTP_MM_ACT_TILE_আকার_Q8_0;
}

স্থির অন্তঃরেখ size_t htp_mm_q8_1_tiled_row_আকার(uint32_t ne) {
    ধ্রুবক uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ধ্রুবক uint32_t nb_32 = ne_padded / 32;
    ফেরত nb_32 * HTP_MM_ACT_TILE_আকার_Q8_1;
}

স্থির অন্তঃরেখ size_t htp_mm_f16_tiled_row_আকার(uint32_t ne) {
    ধ্রুবক uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ধ্রুবক uint32_t nb_32 = ne_padded / 32;
    ফেরত nb_32 * HTP_MM_ACT_TILE_আকার_F16;
}

স্থির অন্তঃরেখ size_t htp_mm_tiled_act_row_আকার(পূর্ণসংখ্যা weight_type, uint32_t ne) {
    যদি (weight_type == HTP_TYPE_F16) {
        ফেরত htp_mm_f16_tiled_row_আকার(ne);
    }
    ফেরত htp_mm_weight_has_offset(weight_type) ? htp_mm_q8_1_tiled_row_আকার(ne) : htp_mm_q8_0_tiled_row_আকার(ne);
}

স্থির অন্তঃরেখ size_t htp_mm_round_up(size_t n, size_t m) {
    ফেরত ((n + m - 1) / m) * m;
}

স্থির অন্তঃরেখ bool htp_mm_hmx_pipeline(uint32_t m) {
    ফেরত m > 32;
}

স্থির অন্তঃরেখ শূন্য htp_mm_hmx_get_2d_chunk_costs(
    পূর্ণসংখ্যা wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_আকার,
    size_t * আকার_per_n_out, size_t * আকার_per_m_out, size_t * আকার_per_mn_out
) {
    ধ্রুবক bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
    ধ্রুবক size_t row_stride = htp_tensor_get_row_আকার(wtype, k);
    ধ্রুবক size_t vec_dot_আকার = k * আকার(uint16_t);
    ধ্রুবক uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(wtype);
    ধ্রুবক size_t qweight_row_stride = is_quant ? (size_t)(n_weight_tiles * aligned_tile_আকার) / 32 : 0;

    *আকার_per_n_out = (pipeline ? 2 : 1) * (is_quant ? qweight_row_stride : row_stride) +
                      (pipeline ? 2 * vec_dot_আকার : vec_dot_আকার);
    *আকার_per_m_out = vec_dot_আকার;
    *আকার_per_mn_out = (pipeline ? 2 : 1) * আকার(uint16_t);
}

স্থির অন্তঃরেখ শূন্য htp_mm_hmx_get_batched_chunk_costs(
    uint32_t k, uint32_t group_আকার,
    size_t * আকার_per_n_out, size_t * আকার_per_m_out, size_t * আকার_per_mn_out
) {
    ধ্রুবক size_t vec_dot_আকার = k * আকার(uint16_t);
    *আকার_per_n_out = 3 * vec_dot_আকার;
    *আকার_per_m_out = group_আকার * vec_dot_আকার;
    *আকার_per_mn_out = আকার(uint16_t);
}

স্থির অন্তঃরেখ size_t htp_mm_hmx_get_2d_overhead(bool pipeline, bool is_matmul_id) {
    size_t অঙ্ক_regions = pipeline ? 7 : (is_matmul_id ? 4 : 5);
    ফেরত অঙ্ক_regions * HTP_MM_HMX_TILE_আকার + 256;
}

স্থির অন্তঃরেখ size_t htp_mm_hmx_get_batched_overhead(শূন্য) {
    ফেরত 5 * HTP_MM_HMX_TILE_আকার + 256;
}

কাঠামো htp_mm_hmx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_weight[2];     // [1] is only used when pipelined
    size_t off_act;
    size_t off_act_f32;       // fp32 activation conversion scratch
    size_t off_dst[2];        // [1] is only used when pipelined
    size_t off_scratch[2];    // dequantization scratch pads
    size_t off_scales;        // HMX scales (256 bytes)
    size_t off_bias;          // bias in VTCM

    // Cached sizes of regions for HMX kernel use
    size_t weight_area_bytes;
    size_t act_area_bytes;
    size_t act_f32_bytes;
    size_t output_area_bytes;
    size_t scratch_bytes[2];
    size_t act_শীর্ষ_stride;
    size_t bias_bytes;

    size_t যোগ_bytes;
};

কাঠামো htp_mm_hvx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    size_t off_act;           // vtcm_act (activation)
    size_t off_src0;          // vtcm_src0 (weight/Wk)
    size_t off_bias;          // vtcm_bias (bias / fused add only)
    size_t off_dst;           // vtcm_dst (output scratch)
    size_t off_act_raw;       // vtcm_act_raw (raw activation DMA staging)

    // Cached sizes
    size_t src0_bytes;
    size_t act_bytes;
    size_t bias_bytes;
    size_t dst_bytes;
    size_t act_raw_bytes;

    size_t যোগ_bytes;
};

স্থির অন্তঃরেখ শূন্য htp_mm_hmx_vtcm_layout_build(
    কাঠামো htp_mm_hmx_vtcm_layout * L,
    পূর্ণসংখ্যা kernel_type,
    পূর্ণসংখ্যা wtype,
    uint32_t k,
    size_t mc,
    size_t nc,
    uint32_t group_আকার,
    bool pipeline,
    uint32_t act_threads,
    uint32_t aligned_tile_আকার,
    size_t bias_আকার
) {
    size_t off = 0;

    যদি (kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        ধ্রুবক size_t vec_dot_আকার     = k * আকার(uint16_t);
        ধ্রুবক size_t act_শীর্ষ_stride   = mc * k;
        ধ্রুবক size_t weight_area_আকার  = hex_align_up(nc * vec_dot_আকার, HTP_MM_HMX_TILE_আকার);
        ধ্রুবক size_t activation_area_আকার = hex_align_up(group_আকার * act_শীর্ষ_stride * আকার(uint16_t), HTP_MM_HMX_TILE_আকার);
        ধ্রুবক size_t output_area_আকার  = hex_align_up(group_আকার * mc * nc * আকার(uint16_t), HTP_MM_HMX_TILE_আকার);
        ধ্রুবক size_t scratch_area_আকার = hex_align_up(nc * vec_dot_আকার, HTP_MM_HMX_TILE_আকার);
        ধ্রুবক size_t min_f32_আকার = hex_align_up(act_threads * HTP_MM_DMA_ACT_MULTIPLIER * k * আকার(দশমিক), 128);

        // Group A: Permanent activation tiles and scales
        size_t off_group_a = 0;
        VTCM_LAYOUT_ALLOC(off_group_a, off_act, activation_area_আকার);
        VTCM_LAYOUT_ALLOC(off_group_a, off_scales, HTP_MM_HMX_TILE_আকার); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_bias, hex_align_up(bias_আকার, HTP_MM_HMX_TILE_আকার), bias_আকার > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_group_b = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_b, off_weight[0], weight_area_আকার);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_weight[1], weight_area_আকার, false);
        VTCM_LAYOUT_ALLOC(off_group_b, off_dst[0], output_area_আকার);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_dst[1], output_area_আকার, false);
        VTCM_LAYOUT_ALLOC(off_group_b, off_scratch[0], scratch_area_আকার);
        VTCM_LAYOUT_ALLOC(off_group_b, off_scratch[1], scratch_area_আকার);

        ধ্রুবক size_t group_b_আকার = off_group_b - off_group_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ধ্রুবক size_t max_f32_আকার = act_threads * 64 * k * আকার(দশমিক);
        ধ্রুবক size_t act_f32_আকার = hex_align_up(hex_smin(max_f32_আকার, hex_smax(min_f32_আকার, group_b_আকার)), 128);
        size_t off_group_c = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_c, off_act_f32, act_f32_আকার);

        ধ্রুবক size_t group_c_আকার = off_group_c - off_group_a;

        L->weight_area_bytes = weight_area_আকার;
        L->act_area_bytes    = activation_area_আকার;
        L->act_f32_bytes     = act_f32_আকার;
        L->output_area_bytes = output_area_আকার;
        L->scratch_bytes[0]  = scratch_area_আকার;
        L->scratch_bytes[1]  = scratch_area_আকার;
        L->act_শীর্ষ_stride   = act_শীর্ষ_stride;
        L->bias_bytes        = bias_আকার;

        off = off_group_a + hex_smax(group_b_আকার, group_c_আকার);
    } অন্যথা {
        // HTP_MM_KERNEL_HMX_2D
        ধ্রুবক bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
        ধ্রুবক size_t row_stride = htp_tensor_get_row_আকার(wtype, k);
        ধ্রুবক size_t vec_dot_আকার = k * আকার(uint16_t);
        ধ্রুবক uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(wtype);

        ধ্রুবক size_t min_f32_আকার = hex_align_up(act_threads * HTP_MM_DMA_ACT_MULTIPLIER * k * আকার(দশমিক), 128);
        ধ্রুবক size_t weight_area_আকার = is_quant
            ? hex_align_up((nc / 32) * n_weight_tiles * aligned_tile_আকার, HTP_MM_HMX_TILE_আকার)
            : hex_align_up(nc * row_stride, HTP_MM_HMX_TILE_আকার);
        ধ্রুবক size_t act_area_আকার    = hex_align_up(mc * vec_dot_আকার, HTP_MM_HMX_TILE_আকার);
        ধ্রুবক size_t output_area_আকার = hex_align_up(mc * nc * আকার(__fp16), HTP_MM_HMX_TILE_আকার);

        ধ্রুবক size_t scratch0_আকার = hex_align_up(nc * vec_dot_আকার, HTP_MM_HMX_TILE_আকার);
        ধ্রুবক size_t scratch1_আকার = pipeline ? scratch0_আকার : 0;

        // Group A:  Scales and activation tiles (must not overlap with Group B or C)
        size_t off_group_a = 0;
        VTCM_LAYOUT_ALLOC(off_group_a, off_scales, HTP_MM_HMX_TILE_আকার); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC(off_group_a, off_act, act_area_আকার);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_bias, hex_align_up(bias_আকার, HTP_MM_HMX_TILE_আকার), bias_আকার > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_group_b = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_b, off_weight[0], weight_area_আকার);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_weight[1], weight_area_আকার, pipeline);
        VTCM_LAYOUT_ALLOC(off_group_b, off_dst[0], output_area_আকার);
        VTCM_LAYOUT_ALLOC(off_group_b, off_scratch[0], scratch0_আকার);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_scratch[1], scratch0_আকার, pipeline);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_dst[1], output_area_আকার, pipeline);

        ধ্রুবক size_t group_b_আকার = off_group_b - off_group_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ধ্রুবক size_t max_f32_আকার = act_threads * 64 * k * আকার(দশমিক);
        ধ্রুবক size_t act_f32_আকার = hex_align_up(hex_smin(max_f32_আকার, hex_smax(min_f32_আকার, group_b_আকার)), 128);
        size_t off_group_c = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_c, off_act_f32, act_f32_আকার);

        ধ্রুবক size_t group_c_আকার = off_group_c - off_group_a;

        L->weight_area_bytes = weight_area_আকার;
        L->act_area_bytes    = act_area_আকার;
        L->act_f32_bytes     = act_f32_আকার;
        L->output_area_bytes = output_area_আকার;
        L->scratch_bytes[0]  = scratch0_আকার;
        L->scratch_bytes[1]  = scratch1_আকার;
        L->act_শীর্ষ_stride   = 0;
        L->bias_bytes        = bias_আকার;

        off = off_group_a + hex_smax(group_b_আকার, group_c_আকার);
    }

    L->যোগ_bytes = off;
}

স্থির অন্তঃরেখ শূন্য htp_mm_hvx_vtcm_layout_build(
    কাঠামো htp_mm_hvx_vtcm_layout * L,
    পূর্ণসংখ্যা kernel_type,
    পূর্ণসংখ্যা wtype,
    uint32_t ne10,       // k
    uint32_t act_nrows,  // m_total
    uint32_t n_threads,
    size_t dst_row_আকার,
    size_t src0_row_আকার,
    size_t act_row_আকার,
    size_t bias_row_আকার,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_nx
) {
    (শূন্য)act_row_আকার;
    size_t src0_sz    = 0;
    size_t act_sz     = 0;
    size_t bias_sz    = bias_row_আকার > 0 ? htp_mm_round_up(bias_row_আকার, 128) : 0;
    size_t dst_sz     = 0;
    size_t act_raw_sz = 0;

    যদি (is_fused_nx) {
        uint32_t aligned_tile_আকার = htp_tensor_get_tile_padded_আকার(wtype);
        uint32_t n_k_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        uint32_t tile_row_আকার = n_k_tiles * aligned_tile_আকার;
        size_t weight_sz_per_thread = hex_round_up(n_prefetch * tile_row_আকার, 128);

        size_t tiled_act_row_আকার = htp_mm_tiled_act_row_আকার(wtype, ne10);
        size_t q_act_sz = hex_round_up(tiled_act_row_আকার * act_nrows, 128);
        size_t raw_row_আকার = hex_round_up(ne10 * আকার(দশমিক), QK_Q8_0_TILED * আকার(দশমিক));

        src0_sz    = weight_sz_per_thread * n_threads; // shared single-weight prefetch buffer
        act_sz     = q_act_sz;                         // quantized activation buffer
        bias_sz    = 0;
        dst_sz     = 0;
        act_raw_sz = hex_round_up(raw_row_আকার * act_nrows, 128);
    } অন্যথা যদি (is_matmul_id) {
        ধ্রুবক size_t act_row_আকার_tiled = htp_mm_tiled_act_row_আকার(wtype, ne10);
        ধ্রুবক uint32_t aligned_tile_আকার = htp_tensor_get_tile_padded_আকার(wtype);
        ধ্রুবক uint32_t n_k_tiles         = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        ধ্রুবক uint32_t tile_row_আকার     = n_k_tiles * aligned_tile_আকার;
        size_t src0_sz_per_thread        = htp_mm_round_up(n_prefetch * tile_row_আকার, 256);

        act_sz = htp_mm_round_up(act_row_আকার_tiled * act_nrows, 256);

        size_t raw_row_আকার = hex_round_up(ne10 * আকার(দশমিক), QK_Q8_0_TILED * আকার(দশমিক));

        src0_sz    = src0_sz_per_thread * n_threads;
        dst_sz     = 0;
        bias_sz    = 0;
        act_raw_sz = hex_round_up(raw_row_আকার * act_nrows, 128);
    } অন্যথা {
        ধ্রুবক size_t src0_row_আকার_padded = htp_mm_round_up(src0_row_আকার, 128);
        ধ্রুবক size_t dst_nrows = (act_nrows > 1) ? 0 : 1;

        নির্বাচন (kernel_type) {
            ক্ষেত্র HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                size_t f16_act_row_আকার = htp_mm_round_up(ne10 * 2, 128);
                act_sz     = htp_mm_round_up(f16_act_row_আকার * act_nrows, 256);
                src0_sz    = htp_mm_round_up(n_prefetch * src0_row_আকার_padded, 256) * n_threads;
                dst_sz     = dst_nrows > 0 ? htp_mm_round_up(dst_row_আকার, 128) * n_threads : 0;
                act_raw_sz = hex_round_up(hex_round_up(ne10 * আকার(দশমিক), 128) * act_nrows, 128);
                বিরতি;
            }
            ক্ষেত্র HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            ক্ষেত্র HTP_MM_KERNEL_HVX_BF16_F32_VTCM: {
                size_t f32_act_row_আকার = htp_mm_round_up(ne10 * 4, 128);
                act_sz     = htp_mm_round_up(f32_act_row_আকার * act_nrows, 256);
                src0_sz    = htp_mm_round_up(n_prefetch * src0_row_আকার_padded, 256) * n_threads;
                dst_sz     = dst_nrows > 0 ? htp_mm_round_up(dst_row_আকার, 128) * n_threads : 0;
                act_raw_sz = 0;
                বিরতি;
            }
            ক্ষেত্র HTP_MM_KERNEL_HVX_QUANT_BLOCK:
            ক্ষেত্র HTP_MM_KERNEL_HVX_QUANT_ROW: {
                size_t q_act_row_আকার = htp_mm_tiled_act_row_আকার(wtype, ne10);
                uint32_t aligned_tile_আকার = htp_tensor_get_tile_padded_আকার(wtype);
                uint32_t n_weight_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
                uint32_t tile_row_আকার = n_weight_tiles * aligned_tile_আকার;
                size_t tiled_vtcm_আকার = htp_mm_round_up(n_prefetch * tile_row_আকার, 256);

                src0_sz = tiled_vtcm_আকার * n_threads;
                act_sz  = htp_mm_round_up(q_act_row_আকার * act_nrows, 256);

                size_t dst_slice_per_thread = (dst_nrows > 0 && act_nrows == 1) ? htp_mm_round_up((dst_row_আকার + n_threads - 1) / n_threads, 128) : 0;
                dst_sz = dst_slice_per_thread * n_threads;
                size_t raw_row_আকার = hex_round_up(ne10 * আকার(দশমিক), QK_Q8_0_TILED * আকার(দশমিক));
                act_raw_sz = hex_round_up(raw_row_আকার * act_nrows, 128);
                বিরতি;
            }
            পূর্বনির্ধারিত:
                বিরতি;
        }
    }

    // Group A: Persistent buffers across chunk compute
    size_t off_group_a = 0;
    VTCM_LAYOUT_ALLOC(off_group_a, off_act, act_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_bias, bias_sz, bias_sz > 0);

    // Group B: Compute-only buffers (starts at off_group_a)
    size_t off_group_b = off_group_a;
    VTCM_LAYOUT_ALLOC(off_group_b, off_src0, src0_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_dst, dst_sz, dst_sz > 0);
    ধ্রুবক size_t group_b_আকার = off_group_b - off_group_a;

    // Group C: Raw activation staging buffer (overlaps Group B, starts at off_group_a)
    size_t off_group_c = off_group_a;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_c, off_act_raw, act_raw_sz, act_raw_sz > 0);
    ধ্রুবক size_t group_c_আকার = off_group_c - off_group_a;

    L->src0_bytes    = src0_sz;
    L->act_bytes     = act_sz;
    L->bias_bytes    = bias_sz;
    L->dst_bytes     = dst_sz;
    L->act_raw_bytes = act_raw_sz;
    L->যোগ_bytes   = off_group_a + hex_smax(group_b_আকার, group_c_আকার);
}

স্থির অন্তঃরেখ bool htp_mm_hvx_solve_vtcm_params(
    পূর্ণসংখ্যা kernel_type,
    পূর্ণসংখ্যা wtype,
    uint32_t ne10,
    uint32_t act_nrows,
    uint32_t n_threads,
    size_t dst_row_আকার,
    size_t src0_row_আকার,
    size_t act_row_আকার,
    size_t bias_row_আকার,
    uint32_t n_prefetch,
    size_t vtcm_budget,
    কাঠামো htp_mm_hvx_vtcm_layout * L_out,
    uint32_t * m_chunk_out
) {
    কাঠামো htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, act_nrows, n_threads,
        dst_row_আকার, src0_row_আকার, act_row_আকার, bias_row_আকার, n_prefetch, false, false
    );

    যদি (L.যোগ_bytes <= vtcm_budget) {
        *L_out = L;
        *m_chunk_out = act_nrows;
        ফেরত true;
    }

    ধ্রুবক size_t fixed_bytes = L.src0_bytes + L.bias_bytes + L.dst_bytes;
    যদি (vtcm_budget <= fixed_bytes) {
        ফেরত false;
    }

    ধ্রুবক size_t avail_act = vtcm_budget - fixed_bytes;
    size_t row_আকার = 0;
    যদি (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        row_আকার = htp_mm_tiled_act_row_আকার(wtype, ne10);
    } অন্যথা যদি (kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        row_আকার = hex_round_up(ne10 * 2, 128);
    } অন্যথা {
        row_আকার = hex_round_up(ne10 * 4, 128);
    }
    যদি (row_আকার == 0) {
        ফেরত false;
    }

    size_t eff_row_আকার = row_আকার;
    যদি (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        eff_row_আকার += hex_round_up(ne10 * আকার(দশমিক), QK_Q8_0_TILED * আকার(দশমিক));
    }

    uint32_t m_chunk = (uint32_t) (avail_act / eff_row_আকার);
    যদি (m_chunk > 1) {
        m_chunk &= ~1U;
    }
    যদি (m_chunk > act_nrows) {
        m_chunk = act_nrows;
    }
    যদি (m_chunk < 1) {
        ফেরত false;
    }

    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, m_chunk, n_threads,
        dst_row_আকার, src0_row_আকার, act_row_আকার, bias_row_আকার, n_prefetch, false, false
    );

    যখন (m_chunk > 2 && L.যোগ_bytes > vtcm_budget) {
        m_chunk -= 2;
        htp_mm_hvx_vtcm_layout_build(
            &L, kernel_type, wtype, ne10, m_chunk, n_threads,
            dst_row_আকার, src0_row_আকার, act_row_আকার, bias_row_আকার, n_prefetch, false, false
        );
    }

    যদি (L.যোগ_bytes <= vtcm_budget) {
        *L_out = L;
        *m_chunk_out = m_chunk;
        ফেরত true;
    }

    ফেরত false;
}

স্থির অন্তঃরেখ size_t htp_mm_hmx_get_2d_vtcm_আকার(
    পূর্ণসংখ্যা wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_threads, uint32_t aligned_tile_আকার, size_t bias_আকার
) {
    কাঠামো htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_threads, aligned_tile_আকার, bias_আকার);
    ফেরত L.যোগ_bytes;
}

স্থির অন্তঃরেখ size_t htp_mm_hmx_get_batched_vtcm_আকার(
    পূর্ণসংখ্যা wtype, uint32_t k, size_t mc, size_t nc, uint32_t group_আকার, bool pipeline, uint32_t act_threads, size_t bias_আকার) {
    (শূন্য)pipeline;
    কাঠামো htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, wtype, k, mc, nc, group_আকার, false, act_threads, 0, bias_আকার);
    ফেরত L.যোগ_bytes;
}

স্থির অন্তঃরেখ bool htp_mm_hmx_solve_batched_params(
    পূর্ণসংখ্যা wtype,
    uint32_t k,
    uint32_t ne01_padded,
    uint32_t ne11,
    uint32_t group_আকার,
    পূর্ণসংখ্যা n_threads,
    bool pipeline,
    size_t bias_আকার,
    size_t vtcm_budget,
    size_t * m_chunk_out,
    size_t * n_chunk_out,
    পূর্ণসংখ্যা * act_threads_out,
    size_t * vtcm_আকার_out
) {
    size_t best_mblocks = SIZE_MAX;
    পূর্ণসংখ্যা best_act_threads = 0;
    size_t best_m_chunk = 0;
    size_t best_n_chunk = 0;
    size_t best_vtcm_আকার = 0;

    পূর্ণসংখ্যা act_threads = n_threads;
    যখন (act_threads >= 1) {
        size_t group_overhead = htp_mm_hmx_get_batched_overhead() + (bias_আকার > 0 ? hex_align_up(bias_আকার, HTP_MM_HMX_TILE_আকার) : 0);
        size_t group_আকার_per_n, group_আকার_per_m, group_আকার_per_mn;
        htp_mm_hmx_get_batched_chunk_costs(k, group_আকার, &group_আকার_per_n, &group_আকার_per_m, &group_আকার_per_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_আকার_candidate = 0;

        যদি (htp_mm_hmx_compute_chunks(vtcm_budget, group_overhead, group_আকার_per_n, group_আকার_per_m, group_আকার_per_mn, hex_align_up(ne11, 32), ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_COST_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_আকার_candidate) == 0) {
            size_t exact_আকার = htp_mm_hmx_get_batched_vtcm_আকার(wtype, k, m_chunk_candidate, n_chunk_candidate, group_আকার, pipeline, act_threads, bias_আকার);
            যদি (exact_আকার <= vtcm_budget) {
                size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                যদি (mblocks < best_mblocks || (mblocks == best_mblocks && act_threads > best_act_threads)) {
                    best_mblocks = mblocks;
                    best_act_threads = act_threads;
                    best_m_chunk = m_chunk_candidate;
                    best_n_chunk = n_chunk_candidate;
                    best_vtcm_আকার = exact_আকার;
                }
            }
        }
        যদি (act_threads == 1) {
            act_threads = 0;
        } অন্যথা {
            act_threads /= 2;
        }
    }

    যদি (best_act_threads > 0) {
        *m_chunk_out = best_m_chunk;
        *n_chunk_out = best_n_chunk;
        *vtcm_আকার_out = best_vtcm_আকার;
        *act_threads_out = best_act_threads;
        ফেরত true;
    }
    ফেরত false;
}

স্থির অন্তঃরেখ bool htp_mm_hmx_solve_2d_params(
    পূর্ণসংখ্যা wtype,
    uint32_t k,
    uint32_t m_id_rows,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_cost,
    পূর্ণসংখ্যা n_threads,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_আকার,
    size_t bias_আকার,
    size_t vtcm_budget,
    size_t * m_chunk_out,
    size_t * n_chunk_out,
    পূর্ণসংখ্যা * act_threads_out,
    size_t * vtcm_আকার_out
) {
    size_t best_mblocks = SIZE_MAX;
    পূর্ণসংখ্যা best_act_threads = 0;
    size_t best_m_chunk = 0;
    size_t best_n_chunk = 0;
    size_t best_vtcm_আকার = 0;

    ধ্রুবক পূর্ণসংখ্যা m_for_chunks = is_matmul_id ? hex_align_up(m_id_rows, 32) : ne11_padded;

    পূর্ণসংখ্যা act_threads = n_threads;
    যখন (act_threads >= 1) {
        size_t simple_2d_overhead = htp_mm_hmx_get_2d_overhead(pipeline, is_matmul_id) + (bias_আকার > 0 ? hex_align_up(bias_আকার, HTP_MM_HMX_TILE_আকার) : 0);
        size_t simple_2d_আকার_per_n, simple_2d_আকার_per_m, simple_2d_আকার_per_mn;
        htp_mm_hmx_get_2d_chunk_costs(wtype, k, pipeline, aligned_tile_আকার, &simple_2d_আকার_per_n, &simple_2d_আকার_per_m, &simple_2d_আকার_per_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_আকার_candidate = 0;

        যদি (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_আকার_per_n, simple_2d_আকার_per_m, simple_2d_আকার_per_mn, m_for_chunks, ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (size_t) m_for_cost * HTP_MM_HMX_COST_A_CONVERT,
                               &m_chunk_candidate, &n_chunk_candidate, &vtcm_আকার_candidate) == 0) {
            size_t exact_আকার = htp_mm_hmx_get_2d_vtcm_আকার(wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_threads, aligned_tile_আকার, bias_আকার);
            যদি (exact_আকার <= vtcm_budget) {
                size_t mblocks = ((size_t) m_for_cost + m_chunk_candidate - 1) / m_chunk_candidate;
                যদি (mblocks < best_mblocks || (mblocks == best_mblocks && act_threads > best_act_threads)) {
                    best_mblocks = mblocks;
                    best_act_threads = act_threads;
                    best_m_chunk = m_chunk_candidate;
                    best_n_chunk = n_chunk_candidate;
                    best_vtcm_আকার = exact_আকার;
                }
            }
        }
        যদি (act_threads == 1) {
            act_threads = 0;
        } অন্যথা {
            act_threads /= 2;
        }
    }

    যদি (best_act_threads > 0) {
        *m_chunk_out = best_m_chunk;
        *n_chunk_out = best_n_chunk;
        *vtcm_আকার_out = best_vtcm_আকার;
        *act_threads_out = best_act_threads;
        ফেরত true;
    }
    ফেরত false;
}

#ifdef __cplusplus
}
#endif

#endif // HTP_MATMUL_OPS_H
