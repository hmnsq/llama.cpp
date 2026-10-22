#ifndef HTP_MATMUL_OPS_H
#define HTP_MATMUL_OPS_H

#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <stddef.h>
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"
#ଅନ୍ତର୍ଭାଵ "hex-fastdiv.h"
#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "htp-vtcm.h"

#ifdef __cplusplus
ବାହ୍ୟ "C" {
#endif

// --- HMX Tile Constraints ---
#define HTP_MM_HMX_TILE_N_COLS 32
#define HTP_MM_HMX_TILE_N_ROWS 32
#define HTP_MM_HMX_TILE_SIZE   (32 * 32 * ଆକାର(__fp16)) // 2048 bytes
#define HTP_MM_HMX_TILE_N_ELMS 1024
#define HTP_MM_HMX_MIN_NROWS   4

#define HTP_MM_WEIGHT_TILE_K_Q2_0             64
#define HTP_MM_WEIGHT_TILE_SIZE_Q2_0          HTP_TENSOR_TILE_SIZE_Q2_0
#define HTP_MM_WEIGHT_ALIGNED_TILE_SIZE_Q2_0  640

// --- Activation Tiled Block Sizes (including padding) ---
#define HTP_MM_ACT_TILE_SIZE_Q8_0      1152
#define HTP_MM_ACT_TILE_SIZE_Q8_1      1280
#define HTP_MM_ACT_TILE_SIZE_F16       2048

#define HTP_MM_MAX_PREFETCH 16

// --- Solver Cost Model Penalty Weights (HMX-specific) ---
#define HTP_MM_HMX_COST_W_DEQUANT 3 // cost penalty for quantized weight loading/dequantization
#define HTP_MM_HMX_COST_A_CONVERT 2 // cost penalty for activation loading/conversion

// --- DMA Activation Transfer Configuration ---
#define HTP_MM_DMA_ACT_ROWS_PER_STEP 2
#define HTP_MM_DMA_ACT_MULTIPLIER    (2 * HTP_MM_DMA_ACT_ROWS_PER_STEP)

ଗଣନା htp_mm_kernel_type {
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
ସଂରଚନା htp_mm_kernel_params {
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
    int32_t  tile_size;          // Weight tile size
    int32_t  aligned_tile_size;  // Aligned weight tile size (padded to 128)
    int32_t  act_row_size;       // Row size for activation scratchpad
    int32_t  vtcm_size;          // Total required scratchpad size in VTCM
    int32_t  vtcm_src0_size;     // src0 scratchpad size in VTCM
    int32_t  vtcm_act_size;      // activation scratchpad size in VTCM
    int32_t  vtcm_bias_size;     // bias scratchpad size in VTCM (fused only)
    int32_t  vtcm_dst_size;      // dst scratchpad size in VTCM

    // Precomputed division values
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne12_ne1;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne1;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_r2;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_r3;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne12;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_n_act_threads;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne00_padded;
};

#ଯଦି defined(__cplusplus)
static_assert(ଆକାର(ସଂରଚନା htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#ଅନ୍ୟଥା
_Static_assert(ଆକାର(ସଂରଚନା htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#endif

ସଂରଚନା mmid_row_mapping {
    uint32_t i1;
    uint32_t i2;
};

// Search for optimal (mc, nc) chunk sizes within VTCM budget.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା htp_mm_hmx_compute_chunks(size_t   vtcm_total,
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
                              size_t * total_out) {
    ଯଦି (m == 0 || n == 0) ଫେରନ୍ତୁ -1;
    ଯଦି (vtcm_total <= overhead) ଫେରନ୍ତୁ -1;
    ଯଦି (per_n_cost == 0 || per_m_cost == 0 || per_mn_cost == 0) ଫେରନ୍ତୁ -1;

    ଅଚଳ size_t usable = vtcm_total - overhead;

    size_t best_cost = SIZE_MAX;
    size_t best_tail_waste = SIZE_MAX;
    size_t best_mn   = 0;
    size_t best_m = 0, best_n = 0;

    ଅଚଳ size_t max_nc_budget = (usable / per_n_cost);
    ଅଚଳ size_t n_max = hex_align_down(hex_smin((size_t)n, max_nc_budget), HTP_MM_HMX_TILE_N_COLS);
    ପାଇଁ (size_t nc = n_max; nc >= HTP_MM_HMX_TILE_N_COLS; nc -= HTP_MM_HMX_TILE_N_COLS) {
        size_t n_fixed = 0, ncmn = 0, mc_denom = 0;
        ଯଦି (hex_mul_overflow(nc, per_n_cost, &n_fixed)) ଜାରି;
        ଯଦି (n_fixed >= usable) ଯାଅ next_nc;

        ଯଦି (hex_mul_overflow(nc, per_mn_cost, &ncmn)) ଯାଅ next_nc;
        ଯଦି (hex_add_overflow(per_m_cost, ncmn, &mc_denom) || mc_denom == 0) ଯାଅ next_nc;

        {
            size_t remain = usable - n_fixed;
            size_t mc = remain / mc_denom;
            mc = hex_align_down(mc, HTP_MM_HMX_TILE_N_ROWS);
            mc = hex_smin(mc, m);

            ଯଦି (mc == 0) {
                ଯାଅ next_nc;
            }

            size_t mblocks = ((size_t) m + mc - 1) / mc;
            size_t nblocks = ((size_t) n + nc - 1) / nc;
            size_t cost    = mblocks * m_block_cost + nblocks * n_block_cost;
            size_t rem     = n % nc;
            size_t tail_waste = (rem == 0) ? 0 : (nc - rem);
            size_t mn      = mc * nc;
            ଯଦି (cost < best_cost ||
                (cost == best_cost && tail_waste < best_tail_waste) ||
                (cost == best_cost && tail_waste == best_tail_waste && mn > best_mn)) {
                best_cost       = cost;
                best_tail_waste = tail_waste;
                best_mn         = mn;
                best_m          = mc;
                best_n          = nc;
            }
        }

next_nc:
        ଯଦି (nc == HTP_MM_HMX_TILE_N_COLS) ଭାଙ୍ଗ;  // avoid size_t underflow
    }

    ଯଦି (best_m == 0 || best_n == 0) ଫେରନ୍ତୁ -1;

    // Compute exact total (with overflow checks)
    size_t t0 = 0, t1 = 0, t2 = 0, mn = 0, total = 0;
    ଯଦି (hex_mul_overflow(best_n, per_n_cost, &t0)) ଫେରନ୍ତୁ -1;
    ଯଦି (hex_mul_overflow(best_m, per_m_cost, &t1)) ଫେରନ୍ତୁ -1;
    ଯଦି (hex_mul_overflow(best_m, best_n, &mn))     ଫେରନ୍ତୁ -1;
    ଯଦି (hex_mul_overflow(mn, per_mn_cost, &t2))    ଫେରନ୍ତୁ -1;
    ଯଦି (hex_add_overflow(t0, t1, &total))          ଫେରନ୍ତୁ -1;
    ଯଦି (hex_add_overflow(total, t2, &total))       ଫେରନ୍ତୁ -1;
    ଯଦି (hex_add_overflow(total, overhead, &total)) ଫେରନ୍ତୁ -1;

    *m_chunk_out = best_m;
    *n_chunk_out = best_n;
    *total_out   = total;
    ଫେରନ୍ତୁ 0;
}

// weight types whose tiles carry a per-block offset (x = d * q + m): the activations need block sums (q8_1)
// (Q2_K: per-16 k sums, q8_1_s16)
ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_mm_weight_has_offset(ପୂର୍ଣ୍ଣସଂଖ୍ୟା weight_type) {
    ଫେରନ୍ତୁ weight_type == HTP_TYPE_Q4_1 || weight_type == HTP_TYPE_Q4_K || weight_type == HTP_TYPE_Q5_K ||
           weight_type == HTP_TYPE_Q2_K || weight_type == HTP_TYPE_Q2_0;
}

// --- Activation/Row Size Helpers ---
ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t htp_mm_q8_0_tiled_row_size(uint32_t ne) {
    ଅଚଳ uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ଅଚଳ uint32_t nb_32 = ne_padded / 32;
    ଫେରନ୍ତୁ nb_32 * HTP_MM_ACT_TILE_SIZE_Q8_0;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t htp_mm_q8_1_tiled_row_size(uint32_t ne) {
    ଅଚଳ uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ଅଚଳ uint32_t nb_32 = ne_padded / 32;
    ଫେରନ୍ତୁ nb_32 * HTP_MM_ACT_TILE_SIZE_Q8_1;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t htp_mm_f16_tiled_row_size(uint32_t ne) {
    ଅଚଳ uint32_t ne_padded = ((ne + 127) / 128) * 128;
    ଅଚଳ uint32_t nb_32 = ne_padded / 32;
    ଫେରନ୍ତୁ nb_32 * HTP_MM_ACT_TILE_SIZE_F16;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t htp_mm_tiled_act_row_size(ପୂର୍ଣ୍ଣସଂଖ୍ୟା weight_type, uint32_t ne) {
    ଯଦି (weight_type == HTP_TYPE_F16) {
        ଫେରନ୍ତୁ htp_mm_f16_tiled_row_size(ne);
    }
    ଫେରନ୍ତୁ htp_mm_weight_has_offset(weight_type) ? htp_mm_q8_1_tiled_row_size(ne) : htp_mm_q8_0_tiled_row_size(ne);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t htp_mm_round_up(size_t n, size_t m) {
    ଫେରନ୍ତୁ ((n + m - 1) / m) * m;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_mm_hmx_pipeline(uint32_t m) {
    ଫେରନ୍ତୁ m > 32;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_mm_hmx_get_2d_chunk_costs(
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_size,
    size_t * size_per_n_out, size_t * size_per_m_out, size_t * size_per_mn_out
) {
    ଅଚଳ bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
    ଅଚଳ size_t row_stride = htp_tensor_get_row_size(wtype, k);
    ଅଚଳ size_t vec_dot_size = k * ଆକାର(uint16_t);
    ଅଚଳ uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(wtype);
    ଅଚଳ size_t qweight_row_stride = is_quant ? (size_t)(n_weight_tiles * aligned_tile_size) / 32 : 0;

    *size_per_n_out = (pipeline ? 2 : 1) * (is_quant ? qweight_row_stride : row_stride) +
                      (pipeline ? 2 * vec_dot_size : vec_dot_size);
    *size_per_m_out = vec_dot_size;
    *size_per_mn_out = (pipeline ? 2 : 1) * ଆକାର(uint16_t);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_mm_hmx_get_batched_chunk_costs(
    uint32_t k, uint32_t group_size,
    size_t * size_per_n_out, size_t * size_per_m_out, size_t * size_per_mn_out
) {
    ଅଚଳ size_t vec_dot_size = k * ଆକାର(uint16_t);
    *size_per_n_out = 3 * vec_dot_size;
    *size_per_m_out = group_size * vec_dot_size;
    *size_per_mn_out = ଆକାର(uint16_t);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t htp_mm_hmx_get_2d_overhead(bool pipeline, bool is_matmul_id) {
    size_t ଅଙ୍କ_regions = pipeline ? 7 : (is_matmul_id ? 4 : 5);
    ଫେରନ୍ତୁ ଅଙ୍କ_regions * HTP_MM_HMX_TILE_SIZE + 256;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t htp_mm_hmx_get_batched_overhead(ଶୂନ୍ୟ) {
    ଫେରନ୍ତୁ 5 * HTP_MM_HMX_TILE_SIZE + 256;
}

ସଂରଚନା htp_mm_hmx_vtcm_layout {
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
    size_t act_head_stride;
    size_t bias_bytes;

    size_t total_bytes;
};

ସଂରଚନା htp_mm_hvx_vtcm_layout {
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

    size_t total_bytes;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_mm_hmx_vtcm_layout_build(
    ସଂରଚନା htp_mm_hmx_vtcm_layout * L,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_type,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା wtype,
    uint32_t k,
    size_t mc,
    size_t nc,
    uint32_t group_size,
    bool pipeline,
    uint32_t act_threads,
    uint32_t aligned_tile_size,
    size_t bias_size
) {
    size_t off = 0;

    ଯଦି (kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        ଅଚଳ size_t vec_dot_size     = k * ଆକାର(uint16_t);
        ଅଚଳ size_t act_head_stride   = mc * k;
        ଅଚଳ size_t weight_area_size  = hex_align_up(nc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
        ଅଚଳ size_t activation_area_size = hex_align_up(group_size * act_head_stride * ଆକାର(uint16_t), HTP_MM_HMX_TILE_SIZE);
        ଅଚଳ size_t output_area_size  = hex_align_up(group_size * mc * nc * ଆକାର(uint16_t), HTP_MM_HMX_TILE_SIZE);
        ଅଚଳ size_t scratch_area_size = hex_align_up(nc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
        ଅଚଳ size_t min_f32_size = hex_align_up(act_threads * HTP_MM_DMA_ACT_MULTIPLIER * k * ଆକାର(ଦଶମିକ), 128);

        // Group A: Permanent activation tiles and scales
        size_t off_group_a = 0;
        VTCM_LAYOUT_ALLOC(off_group_a, off_act, activation_area_size);
        VTCM_LAYOUT_ALLOC(off_group_a, off_scales, HTP_MM_HMX_TILE_SIZE); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_bias, hex_align_up(bias_size, HTP_MM_HMX_TILE_SIZE), bias_size > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_group_b = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_b, off_weight[0], weight_area_size);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_weight[1], weight_area_size, false);
        VTCM_LAYOUT_ALLOC(off_group_b, off_dst[0], output_area_size);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_dst[1], output_area_size, false);
        VTCM_LAYOUT_ALLOC(off_group_b, off_scratch[0], scratch_area_size);
        VTCM_LAYOUT_ALLOC(off_group_b, off_scratch[1], scratch_area_size);

        ଅଚଳ size_t group_b_size = off_group_b - off_group_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ଅଚଳ size_t max_f32_size = act_threads * 64 * k * ଆକାର(ଦଶମିକ);
        ଅଚଳ size_t act_f32_size = hex_align_up(hex_smin(max_f32_size, hex_smax(min_f32_size, group_b_size)), 128);
        size_t off_group_c = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_c, off_act_f32, act_f32_size);

        ଅଚଳ size_t group_c_size = off_group_c - off_group_a;

        L->weight_area_bytes = weight_area_size;
        L->act_area_bytes    = activation_area_size;
        L->act_f32_bytes     = act_f32_size;
        L->output_area_bytes = output_area_size;
        L->scratch_bytes[0]  = scratch_area_size;
        L->scratch_bytes[1]  = scratch_area_size;
        L->act_head_stride   = act_head_stride;
        L->bias_bytes        = bias_size;

        off = off_group_a + hex_smax(group_b_size, group_c_size);
    } ଅନ୍ୟଥା {
        // HTP_MM_KERNEL_HMX_2D
        ଅଚଳ bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
        ଅଚଳ size_t row_stride = htp_tensor_get_row_size(wtype, k);
        ଅଚଳ size_t vec_dot_size = k * ଆକାର(uint16_t);
        ଅଚଳ uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(wtype);

        ଅଚଳ size_t min_f32_size = hex_align_up(act_threads * HTP_MM_DMA_ACT_MULTIPLIER * k * ଆକାର(ଦଶମିକ), 128);
        ଅଚଳ size_t weight_area_size = is_quant
            ? hex_align_up((nc / 32) * n_weight_tiles * aligned_tile_size, HTP_MM_HMX_TILE_SIZE)
            : hex_align_up(nc * row_stride, HTP_MM_HMX_TILE_SIZE);
        ଅଚଳ size_t act_area_size    = hex_align_up(mc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
        ଅଚଳ size_t output_area_size = hex_align_up(mc * nc * ଆକାର(__fp16), HTP_MM_HMX_TILE_SIZE);

        ଅଚଳ size_t scratch0_size = hex_align_up(nc * vec_dot_size, HTP_MM_HMX_TILE_SIZE);
        ଅଚଳ size_t scratch1_size = pipeline ? scratch0_size : 0;

        // Group A:  Scales and activation tiles (must not overlap with Group B or C)
        size_t off_group_a = 0;
        VTCM_LAYOUT_ALLOC(off_group_a, off_scales, HTP_MM_HMX_TILE_SIZE); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC(off_group_a, off_act, act_area_size);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_bias, hex_align_up(bias_size, HTP_MM_HMX_TILE_SIZE), bias_size > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        size_t off_group_b = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_b, off_weight[0], weight_area_size);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_weight[1], weight_area_size, pipeline);
        VTCM_LAYOUT_ALLOC(off_group_b, off_dst[0], output_area_size);
        VTCM_LAYOUT_ALLOC(off_group_b, off_scratch[0], scratch0_size);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_scratch[1], scratch0_size, pipeline);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_dst[1], output_area_size, pipeline);

        ଅଚଳ size_t group_b_size = off_group_b - off_group_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        ଅଚଳ size_t max_f32_size = act_threads * 64 * k * ଆକାର(ଦଶମିକ);
        ଅଚଳ size_t act_f32_size = hex_align_up(hex_smin(max_f32_size, hex_smax(min_f32_size, group_b_size)), 128);
        size_t off_group_c = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_c, off_act_f32, act_f32_size);

        ଅଚଳ size_t group_c_size = off_group_c - off_group_a;

        L->weight_area_bytes = weight_area_size;
        L->act_area_bytes    = act_area_size;
        L->act_f32_bytes     = act_f32_size;
        L->output_area_bytes = output_area_size;
        L->scratch_bytes[0]  = scratch0_size;
        L->scratch_bytes[1]  = scratch1_size;
        L->act_head_stride   = 0;
        L->bias_bytes        = bias_size;

        off = off_group_a + hex_smax(group_b_size, group_c_size);
    }

    L->total_bytes = off;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ htp_mm_hvx_vtcm_layout_build(
    ସଂରଚନା htp_mm_hvx_vtcm_layout * L,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_type,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା wtype,
    uint32_t ne10,       // k
    uint32_t act_nrows,  // m_total
    uint32_t n_threads,
    size_t dst_row_size,
    size_t src0_row_size,
    size_t act_row_size,
    size_t bias_row_size,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_nx
) {
    (ଶୂନ୍ୟ)act_row_size;
    size_t src0_sz    = 0;
    size_t act_sz     = 0;
    size_t bias_sz    = bias_row_size > 0 ? htp_mm_round_up(bias_row_size, 128) : 0;
    size_t dst_sz     = 0;
    size_t act_raw_sz = 0;

    ଯଦି (is_fused_nx) {
        uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);
        uint32_t n_k_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        uint32_t tile_row_size = n_k_tiles * aligned_tile_size;
        size_t weight_sz_per_thread = hex_round_up(n_prefetch * tile_row_size, 128);

        size_t tiled_act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
        size_t q_act_sz = hex_round_up(tiled_act_row_size * act_nrows, 128);
        size_t raw_row_size = hex_round_up(ne10 * ଆକାର(ଦଶମିକ), QK_Q8_0_TILED * ଆକାର(ଦଶମିକ));

        src0_sz    = weight_sz_per_thread * n_threads; // shared single-weight prefetch buffer
        act_sz     = q_act_sz;                         // quantized activation buffer
        bias_sz    = 0;
        dst_sz     = 0;
        act_raw_sz = hex_round_up(raw_row_size * act_nrows, 128);
    } ଅନ୍ୟଥା ଯଦି (is_matmul_id) {
        ଅଚଳ size_t act_row_size_tiled = htp_mm_tiled_act_row_size(wtype, ne10);
        ଅଚଳ uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);
        ଅଚଳ uint32_t n_k_tiles         = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        ଅଚଳ uint32_t tile_row_size     = n_k_tiles * aligned_tile_size;
        size_t src0_sz_per_thread        = htp_mm_round_up(n_prefetch * tile_row_size, 256);

        act_sz = htp_mm_round_up(act_row_size_tiled * act_nrows, 256);

        size_t raw_row_size = hex_round_up(ne10 * ଆକାର(ଦଶମିକ), QK_Q8_0_TILED * ଆକାର(ଦଶମିକ));

        src0_sz    = src0_sz_per_thread * n_threads;
        dst_sz     = 0;
        bias_sz    = 0;
        act_raw_sz = hex_round_up(raw_row_size * act_nrows, 128);
    } ଅନ୍ୟଥା {
        ଅଚଳ size_t src0_row_size_padded = htp_mm_round_up(src0_row_size, 128);
        ଅଚଳ size_t dst_nrows = (act_nrows > 1) ? 0 : 1;

        ବାଛ (kernel_type) {
            ସ୍ଥିତି HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                size_t f16_act_row_size = htp_mm_round_up(ne10 * 2, 128);
                act_sz     = htp_mm_round_up(f16_act_row_size * act_nrows, 256);
                src0_sz    = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256) * n_threads;
                dst_sz     = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_threads : 0;
                act_raw_sz = hex_round_up(hex_round_up(ne10 * ଆକାର(ଦଶମିକ), 128) * act_nrows, 128);
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            ସ୍ଥିତି HTP_MM_KERNEL_HVX_BF16_F32_VTCM: {
                size_t f32_act_row_size = htp_mm_round_up(ne10 * 4, 128);
                act_sz     = htp_mm_round_up(f32_act_row_size * act_nrows, 256);
                src0_sz    = htp_mm_round_up(n_prefetch * src0_row_size_padded, 256) * n_threads;
                dst_sz     = dst_nrows > 0 ? htp_mm_round_up(dst_row_size, 128) * n_threads : 0;
                act_raw_sz = 0;
                ଭାଙ୍ଗ;
            }
            ସ୍ଥିତି HTP_MM_KERNEL_HVX_QUANT_BLOCK:
            ସ୍ଥିତି HTP_MM_KERNEL_HVX_QUANT_ROW: {
                size_t q_act_row_size = htp_mm_tiled_act_row_size(wtype, ne10);
                uint32_t aligned_tile_size = htp_tensor_get_tile_padded_size(wtype);
                uint32_t n_weight_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
                uint32_t tile_row_size = n_weight_tiles * aligned_tile_size;
                size_t tiled_vtcm_size = htp_mm_round_up(n_prefetch * tile_row_size, 256);

                src0_sz = tiled_vtcm_size * n_threads;
                act_sz  = htp_mm_round_up(q_act_row_size * act_nrows, 256);

                size_t dst_slice_per_thread = (dst_nrows > 0 && act_nrows == 1) ? htp_mm_round_up((dst_row_size + n_threads - 1) / n_threads, 128) : 0;
                dst_sz = dst_slice_per_thread * n_threads;
                size_t raw_row_size = hex_round_up(ne10 * ଆକାର(ଦଶମିକ), QK_Q8_0_TILED * ଆକାର(ଦଶମିକ));
                act_raw_sz = hex_round_up(raw_row_size * act_nrows, 128);
                ଭାଙ୍ଗ;
            }
            ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
                ଭାଙ୍ଗ;
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
    ଅଚଳ size_t group_b_size = off_group_b - off_group_a;

    // Group C: Raw activation staging buffer (overlaps Group B, starts at off_group_a)
    size_t off_group_c = off_group_a;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_c, off_act_raw, act_raw_sz, act_raw_sz > 0);
    ଅଚଳ size_t group_c_size = off_group_c - off_group_a;

    L->src0_bytes    = src0_sz;
    L->act_bytes     = act_sz;
    L->bias_bytes    = bias_sz;
    L->dst_bytes     = dst_sz;
    L->act_raw_bytes = act_raw_sz;
    L->total_bytes   = off_group_a + hex_smax(group_b_size, group_c_size);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_mm_hvx_solve_vtcm_params(
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_type,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା wtype,
    uint32_t ne10,
    uint32_t act_nrows,
    uint32_t n_threads,
    size_t dst_row_size,
    size_t src0_row_size,
    size_t act_row_size,
    size_t bias_row_size,
    uint32_t n_prefetch,
    size_t vtcm_budget,
    ସଂରଚନା htp_mm_hvx_vtcm_layout * L_out,
    uint32_t * m_chunk_out
) {
    ସଂରଚନା htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, act_nrows, n_threads,
        dst_row_size, src0_row_size, act_row_size, bias_row_size, n_prefetch, false, false
    );

    ଯଦି (L.total_bytes <= vtcm_budget) {
        *L_out = L;
        *m_chunk_out = act_nrows;
        ଫେରନ୍ତୁ true;
    }

    ଅଚଳ size_t fixed_bytes = L.src0_bytes + L.bias_bytes + L.dst_bytes;
    ଯଦି (vtcm_budget <= fixed_bytes) {
        ଫେରନ୍ତୁ false;
    }

    ଅଚଳ size_t avail_act = vtcm_budget - fixed_bytes;
    size_t row_size = 0;
    ଯଦି (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        row_size = htp_mm_tiled_act_row_size(wtype, ne10);
    } ଅନ୍ୟଥା ଯଦି (kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        row_size = hex_round_up(ne10 * 2, 128);
    } ଅନ୍ୟଥା {
        row_size = hex_round_up(ne10 * 4, 128);
    }
    ଯଦି (row_size == 0) {
        ଫେରନ୍ତୁ false;
    }

    size_t eff_row_size = row_size;
    ଯଦି (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        eff_row_size += hex_round_up(ne10 * ଆକାର(ଦଶମିକ), QK_Q8_0_TILED * ଆକାର(ଦଶମିକ));
    }

    uint32_t m_chunk = (uint32_t) (avail_act / eff_row_size);
    ଯଦି (m_chunk > 1) {
        m_chunk &= ~1U;
    }
    ଯଦି (m_chunk > act_nrows) {
        m_chunk = act_nrows;
    }
    ଯଦି (m_chunk < 1) {
        ଫେରନ୍ତୁ false;
    }

    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, m_chunk, n_threads,
        dst_row_size, src0_row_size, act_row_size, bias_row_size, n_prefetch, false, false
    );

    ଯେତେବେଳେ (m_chunk > 2 && L.total_bytes > vtcm_budget) {
        m_chunk -= 2;
        htp_mm_hvx_vtcm_layout_build(
            &L, kernel_type, wtype, ne10, m_chunk, n_threads,
            dst_row_size, src0_row_size, act_row_size, bias_row_size, n_prefetch, false, false
        );
    }

    ଯଦି (L.total_bytes <= vtcm_budget) {
        *L_out = L;
        *m_chunk_out = m_chunk;
        ଫେରନ୍ତୁ true;
    }

    ଫେରନ୍ତୁ false;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t htp_mm_hmx_get_2d_vtcm_size(
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା wtype, uint32_t k, size_t mc, size_t nc, bool pipeline, uint32_t act_threads, uint32_t aligned_tile_size, size_t bias_size
) {
    ସଂରଚନା htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_threads, aligned_tile_size, bias_size);
    ଫେରନ୍ତୁ L.total_bytes;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t htp_mm_hmx_get_batched_vtcm_size(
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା wtype, uint32_t k, size_t mc, size_t nc, uint32_t group_size, bool pipeline, uint32_t act_threads, size_t bias_size) {
    (ଶୂନ୍ୟ)pipeline;
    ସଂରଚନା htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, wtype, k, mc, nc, group_size, false, act_threads, 0, bias_size);
    ଫେରନ୍ତୁ L.total_bytes;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_mm_hmx_solve_batched_params(
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା wtype,
    uint32_t k,
    uint32_t ne01_padded,
    uint32_t ne11,
    uint32_t group_size,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads,
    bool pipeline,
    size_t bias_size,
    size_t vtcm_budget,
    size_t * m_chunk_out,
    size_t * n_chunk_out,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା * act_threads_out,
    size_t * vtcm_size_out
) {
    size_t best_mblocks = SIZE_MAX;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା best_act_threads = 0;
    size_t best_m_chunk = 0;
    size_t best_n_chunk = 0;
    size_t best_vtcm_size = 0;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା act_threads = n_threads;
    ଯେତେବେଳେ (act_threads >= 1) {
        size_t group_overhead = htp_mm_hmx_get_batched_overhead() + (bias_size > 0 ? hex_align_up(bias_size, HTP_MM_HMX_TILE_SIZE) : 0);
        size_t group_size_per_n, group_size_per_m, group_size_per_mn;
        htp_mm_hmx_get_batched_chunk_costs(k, group_size, &group_size_per_n, &group_size_per_m, &group_size_per_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_size_candidate = 0;

        ଯଦି (htp_mm_hmx_compute_chunks(vtcm_budget, group_overhead, group_size_per_n, group_size_per_m, group_size_per_mn, hex_align_up(ne11, 32), ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (size_t) ne11 * HTP_MM_HMX_COST_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_size_candidate) == 0) {
            size_t exact_size = htp_mm_hmx_get_batched_vtcm_size(wtype, k, m_chunk_candidate, n_chunk_candidate, group_size, pipeline, act_threads, bias_size);
            ଯଦି (exact_size <= vtcm_budget) {
                size_t mblocks = ((size_t) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                ଯଦି (mblocks < best_mblocks || (mblocks == best_mblocks && act_threads > best_act_threads)) {
                    best_mblocks = mblocks;
                    best_act_threads = act_threads;
                    best_m_chunk = m_chunk_candidate;
                    best_n_chunk = n_chunk_candidate;
                    best_vtcm_size = exact_size;
                }
            }
        }
        ଯଦି (act_threads == 1) {
            act_threads = 0;
        } ଅନ୍ୟଥା {
            act_threads /= 2;
        }
    }

    ଯଦି (best_act_threads > 0) {
        *m_chunk_out = best_m_chunk;
        *n_chunk_out = best_n_chunk;
        *vtcm_size_out = best_vtcm_size;
        *act_threads_out = best_act_threads;
        ଫେରନ୍ତୁ true;
    }
    ଫେରନ୍ତୁ false;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ bool htp_mm_hmx_solve_2d_params(
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା wtype,
    uint32_t k,
    uint32_t m_id_rows,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_cost,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_threads,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_size,
    size_t bias_size,
    size_t vtcm_budget,
    size_t * m_chunk_out,
    size_t * n_chunk_out,
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା * act_threads_out,
    size_t * vtcm_size_out
) {
    size_t best_mblocks = SIZE_MAX;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା best_act_threads = 0;
    size_t best_m_chunk = 0;
    size_t best_n_chunk = 0;
    size_t best_vtcm_size = 0;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା m_for_chunks = is_matmul_id ? hex_align_up(m_id_rows, 32) : ne11_padded;

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା act_threads = n_threads;
    ଯେତେବେଳେ (act_threads >= 1) {
        size_t simple_2d_overhead = htp_mm_hmx_get_2d_overhead(pipeline, is_matmul_id) + (bias_size > 0 ? hex_align_up(bias_size, HTP_MM_HMX_TILE_SIZE) : 0);
        size_t simple_2d_size_per_n, simple_2d_size_per_m, simple_2d_size_per_mn;
        htp_mm_hmx_get_2d_chunk_costs(wtype, k, pipeline, aligned_tile_size, &simple_2d_size_per_n, &simple_2d_size_per_m, &simple_2d_size_per_mn);

        size_t m_chunk_candidate = 0;
        size_t n_chunk_candidate = 0;
        size_t vtcm_size_candidate = 0;

        ଯଦି (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_size_per_n, simple_2d_size_per_m, simple_2d_size_per_mn, m_for_chunks, ne01_padded,
                               (size_t) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (size_t) m_for_cost * HTP_MM_HMX_COST_A_CONVERT,
                               &m_chunk_candidate, &n_chunk_candidate, &vtcm_size_candidate) == 0) {
            size_t exact_size = htp_mm_hmx_get_2d_vtcm_size(wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_threads, aligned_tile_size, bias_size);
            ଯଦି (exact_size <= vtcm_budget) {
                size_t mblocks = ((size_t) m_for_cost + m_chunk_candidate - 1) / m_chunk_candidate;
                ଯଦି (mblocks < best_mblocks || (mblocks == best_mblocks && act_threads > best_act_threads)) {
                    best_mblocks = mblocks;
                    best_act_threads = act_threads;
                    best_m_chunk = m_chunk_candidate;
                    best_n_chunk = n_chunk_candidate;
                    best_vtcm_size = exact_size;
                }
            }
        }
        ଯଦି (act_threads == 1) {
            act_threads = 0;
        } ଅନ୍ୟଥା {
            act_threads /= 2;
        }
    }

    ଯଦି (best_act_threads > 0) {
        *m_chunk_out = best_m_chunk;
        *n_chunk_out = best_n_chunk;
        *vtcm_size_out = best_vtcm_size;
        *act_threads_out = best_act_threads;
        ଫେରନ୍ତୁ true;
    }
    ଫେରନ୍ତୁ false;
}

#ifdef __cplusplus
}
#endif

#endif // HTP_MATMUL_OPS_H
