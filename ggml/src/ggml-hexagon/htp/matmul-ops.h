#अगर_अघोषित HTP_MATMUL_OPS_H
#घोषणा HTP_MATMUL_OPS_H

#समावेश <stdint.h>
#समावेश <stddef.h>
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"
#समावेश "hex-fastdiv.h"
#समावेश "hex-common.h"
#समावेश "htp-vtcm.h"

#अगर_घोषित __cplusplus
बाह्य "C" {
#पूर्ण_अगर

// --- HMX Tile Constraints ---
#घोषणा HTP_MM_HMX_TILE_N_COLS 32
#घोषणा HTP_MM_HMX_TILE_N_ROWS 32
#घोषणा HTP_MM_HMX_TILE_आकार   (32 * 32 * माप(__fp16)) // 2048 bytes
#घोषणा HTP_MM_HMX_TILE_N_ELMS 1024
#घोषणा HTP_MM_HMX_MIN_NROWS   4

#घोषणा HTP_MM_WEIGHT_TILE_K_Q2_0             64
#घोषणा HTP_MM_WEIGHT_TILE_आकार_Q2_0          HTP_TENSOR_TILE_आकार_Q2_0
#घोषणा HTP_MM_WEIGHT_ALIGNED_TILE_आकार_Q2_0  640

// --- Activation Tiled Block Sizes (including padding) ---
#घोषणा HTP_MM_ACT_TILE_आकार_Q8_0      1152
#घोषणा HTP_MM_ACT_TILE_आकार_Q8_1      1280
#घोषणा HTP_MM_ACT_TILE_आकार_F16       2048

#घोषणा HTP_MM_MAX_PREFETCH 16

// --- Solver Cost Model Penalty Weights (HMX-specific) ---
#घोषणा HTP_MM_HMX_COST_W_DEQUANT 3 // cost penalty for quantized weight loading/dequantization
#घोषणा HTP_MM_HMX_COST_A_CONVERT 2 // cost penalty for activation loading/conversion

// --- DMA Activation Transfer Configuration ---
#घोषणा HTP_MM_DMA_ACT_ROWS_PER_STEP 2
#घोषणा HTP_MM_DMA_ACT_MULTIPLIER    (2 * HTP_MM_DMA_ACT_ROWS_PER_STEP)

क्रमागत htp_mm_kernel_type {
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
काष्ठा htp_mm_kernel_params {
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
    int32_t  tile_आकार;          // Weight tile size
    int32_t  aligned_tile_आकार;  // Aligned weight tile size (padded to 128)
    int32_t  act_row_आकार;       // Row size for activation scratchpad
    int32_t  vtcm_आकार;          // Total required scratchpad size in VTCM
    int32_t  vtcm_src0_आकार;     // src0 scratchpad size in VTCM
    int32_t  vtcm_act_आकार;      // activation scratchpad size in VTCM
    int32_t  vtcm_bias_आकार;     // bias scratchpad size in VTCM (fused only)
    int32_t  vtcm_dst_आकार;      // dst scratchpad size in VTCM

    // Precomputed division values
    काष्ठा fastdiv_मूल्यावली div_ne12_ne1;
    काष्ठा fastdiv_मूल्यावली div_ne1;
    काष्ठा fastdiv_मूल्यावली div_r2;
    काष्ठा fastdiv_मूल्यावली div_r3;
    काष्ठा fastdiv_मूल्यावली div_ne12;
    काष्ठा fastdiv_मूल्यावली div_n_act_threads;
    काष्ठा fastdiv_मूल्यावली div_ne00_padded;
};

#अगर defined(__cplusplus)
static_assert(माप(काष्ठा htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#अन्यथा
_Static_assert(माप(काष्ठा htp_mm_kernel_params) <= 128, "htp_matmul_kernel_params is too large for kernel_params blob");
#पूर्ण_अगर

काष्ठा mmid_row_mapping {
    uint32_t i1;
    uint32_t i2;
};

// Search for optimal (mc, nc) chunk sizes within VTCM budget.
अटल अंतरभूत पूर्णांक htp_mm_hmx_compute_chunks(माप_प्रकार   vtcm_योग,
                              माप_प्रकार   overhead,
                              माप_प्रकार   per_n_cost,
                              माप_प्रकार   per_m_cost,
                              माप_प्रकार   per_mn_cost,
                              माप_प्रकार   m,
                              माप_प्रकार   n,
                              माप_प्रकार   m_block_cost,
                              माप_प्रकार   n_block_cost,
                              माप_प्रकार * m_chunk_out,
                              माप_प्रकार * n_chunk_out,
                              माप_प्रकार * योग_out) {
    अगर (m == 0 || n == 0) वापस -1;
    अगर (vtcm_योग <= overhead) वापस -1;
    अगर (per_n_cost == 0 || per_m_cost == 0 || per_mn_cost == 0) वापस -1;

    स्थिर माप_प्रकार usable = vtcm_योग - overhead;

    माप_प्रकार best_cost = SIZE_MAX;
    माप_प्रकार best_पुच्छ_waste = SIZE_MAX;
    माप_प्रकार best_mn   = 0;
    माप_प्रकार best_m = 0, best_n = 0;

    स्थिर माप_प्रकार max_nc_budget = (usable / per_n_cost);
    स्थिर माप_प्रकार n_max = hex_align_down(hex_smin((माप_प्रकार)n, max_nc_budget), HTP_MM_HMX_TILE_N_COLS);
    क्रम (माप_प्रकार nc = n_max; nc >= HTP_MM_HMX_TILE_N_COLS; nc -= HTP_MM_HMX_TILE_N_COLS) {
        माप_प्रकार n_fixed = 0, ncmn = 0, mc_denom = 0;
        अगर (hex_mul_overflow(nc, per_n_cost, &n_fixed)) जारी;
        अगर (n_fixed >= usable) जाओ अग्र_nc;

        अगर (hex_mul_overflow(nc, per_mn_cost, &ncmn)) जाओ अग्र_nc;
        अगर (hex_add_overflow(per_m_cost, ncmn, &mc_denom) || mc_denom == 0) जाओ अग्र_nc;

        {
            माप_प्रकार remain = usable - n_fixed;
            माप_प्रकार mc = remain / mc_denom;
            mc = hex_align_down(mc, HTP_MM_HMX_TILE_N_ROWS);
            mc = hex_smin(mc, m);

            अगर (mc == 0) {
                जाओ अग्र_nc;
            }

            माप_प्रकार mblocks = ((माप_प्रकार) m + mc - 1) / mc;
            माप_प्रकार nblocks = ((माप_प्रकार) n + nc - 1) / nc;
            माप_प्रकार cost    = mblocks * m_block_cost + nblocks * n_block_cost;
            माप_प्रकार rem     = n % nc;
            माप_प्रकार पुच्छ_waste = (rem == 0) ? 0 : (nc - rem);
            माप_प्रकार mn      = mc * nc;
            अगर (cost < best_cost ||
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
        अगर (nc == HTP_MM_HMX_TILE_N_COLS) अवरोध;  // avoid size_t underflow
    }

    अगर (best_m == 0 || best_n == 0) वापस -1;

    // Compute exact total (with overflow checks)
    माप_प्रकार t0 = 0, t1 = 0, t2 = 0, mn = 0, योग = 0;
    अगर (hex_mul_overflow(best_n, per_n_cost, &t0)) वापस -1;
    अगर (hex_mul_overflow(best_m, per_m_cost, &t1)) वापस -1;
    अगर (hex_mul_overflow(best_m, best_n, &mn))     वापस -1;
    अगर (hex_mul_overflow(mn, per_mn_cost, &t2))    वापस -1;
    अगर (hex_add_overflow(t0, t1, &योग))          वापस -1;
    अगर (hex_add_overflow(योग, t2, &योग))       वापस -1;
    अगर (hex_add_overflow(योग, overhead, &योग)) वापस -1;

    *m_chunk_out = best_m;
    *n_chunk_out = best_n;
    *योग_out   = योग;
    वापस 0;
}

// weight types whose tiles carry a per-block offset (x = d * q + m): the activations need block sums (q8_1)
// (Q2_K: per-16 k sums, q8_1_s16)
अटल अंतरभूत bool htp_mm_weight_has_offset(पूर्णांक weight_type) {
    वापस weight_type == HTP_TYPE_Q4_1 || weight_type == HTP_TYPE_Q4_K || weight_type == HTP_TYPE_Q5_K ||
           weight_type == HTP_TYPE_Q2_K || weight_type == HTP_TYPE_Q2_0;
}

// --- Activation/Row Size Helpers ---
अटल अंतरभूत माप_प्रकार htp_mm_q8_0_tiled_row_आकार(uint32_t ne) {
    स्थिर uint32_t ne_padded = ((ne + 127) / 128) * 128;
    स्थिर uint32_t nb_32 = ne_padded / 32;
    वापस nb_32 * HTP_MM_ACT_TILE_आकार_Q8_0;
}

अटल अंतरभूत माप_प्रकार htp_mm_q8_1_tiled_row_आकार(uint32_t ne) {
    स्थिर uint32_t ne_padded = ((ne + 127) / 128) * 128;
    स्थिर uint32_t nb_32 = ne_padded / 32;
    वापस nb_32 * HTP_MM_ACT_TILE_आकार_Q8_1;
}

अटल अंतरभूत माप_प्रकार htp_mm_f16_tiled_row_आकार(uint32_t ne) {
    स्थिर uint32_t ne_padded = ((ne + 127) / 128) * 128;
    स्थिर uint32_t nb_32 = ne_padded / 32;
    वापस nb_32 * HTP_MM_ACT_TILE_आकार_F16;
}

अटल अंतरभूत माप_प्रकार htp_mm_tiled_act_row_आकार(पूर्णांक weight_type, uint32_t ne) {
    अगर (weight_type == HTP_TYPE_F16) {
        वापस htp_mm_f16_tiled_row_आकार(ne);
    }
    वापस htp_mm_weight_has_offset(weight_type) ? htp_mm_q8_1_tiled_row_आकार(ne) : htp_mm_q8_0_tiled_row_आकार(ne);
}

अटल अंतरभूत माप_प्रकार htp_mm_round_up(माप_प्रकार n, माप_प्रकार m) {
    वापस ((n + m - 1) / m) * m;
}

अटल अंतरभूत bool htp_mm_hmx_pipeline(uint32_t m) {
    वापस m > 32;
}

अटल अंतरभूत व्योम htp_mm_hmx_get_2d_chunk_costs(
    पूर्णांक wtype, uint32_t k, bool pipeline, uint32_t aligned_tile_आकार,
    माप_प्रकार * आकार_per_n_out, माप_प्रकार * आकार_per_m_out, माप_प्रकार * आकार_per_mn_out
) {
    स्थिर bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
    स्थिर माप_प्रकार row_stride = htp_tensor_get_row_आकार(wtype, k);
    स्थिर माप_प्रकार vec_dot_आकार = k * माप(uint16_t);
    स्थिर uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(wtype);
    स्थिर माप_प्रकार qweight_row_stride = is_quant ? (माप_प्रकार)(n_weight_tiles * aligned_tile_आकार) / 32 : 0;

    *आकार_per_n_out = (pipeline ? 2 : 1) * (is_quant ? qweight_row_stride : row_stride) +
                      (pipeline ? 2 * vec_dot_आकार : vec_dot_आकार);
    *आकार_per_m_out = vec_dot_आकार;
    *आकार_per_mn_out = (pipeline ? 2 : 1) * माप(uint16_t);
}

अटल अंतरभूत व्योम htp_mm_hmx_get_batched_chunk_costs(
    uint32_t k, uint32_t group_आकार,
    माप_प्रकार * आकार_per_n_out, माप_प्रकार * आकार_per_m_out, माप_प्रकार * आकार_per_mn_out
) {
    स्थिर माप_प्रकार vec_dot_आकार = k * माप(uint16_t);
    *आकार_per_n_out = 3 * vec_dot_आकार;
    *आकार_per_m_out = group_आकार * vec_dot_आकार;
    *आकार_per_mn_out = माप(uint16_t);
}

अटल अंतरभूत माप_प्रकार htp_mm_hmx_get_2d_overhead(bool pipeline, bool is_matmul_id) {
    माप_प्रकार अंक_regions = pipeline ? 7 : (is_matmul_id ? 4 : 5);
    वापस अंक_regions * HTP_MM_HMX_TILE_आकार + 256;
}

अटल अंतरभूत माप_प्रकार htp_mm_hmx_get_batched_overhead(व्योम) {
    वापस 5 * HTP_MM_HMX_TILE_आकार + 256;
}

काष्ठा htp_mm_hmx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    माप_प्रकार off_weight[2];     // [1] is only used when pipelined
    माप_प्रकार off_act;
    माप_प्रकार off_act_f32;       // fp32 activation conversion scratch
    माप_प्रकार off_dst[2];        // [1] is only used when pipelined
    माप_प्रकार off_scratch[2];    // dequantization scratch pads
    माप_प्रकार off_scales;        // HMX scales (256 bytes)
    माप_प्रकार off_bias;          // bias in VTCM

    // Cached sizes of regions for HMX kernel use
    माप_प्रकार weight_area_bytes;
    माप_प्रकार act_area_bytes;
    माप_प्रकार act_f32_bytes;
    माप_प्रकार output_area_bytes;
    माप_प्रकार scratch_bytes[2];
    माप_प्रकार act_शीर्ष_stride;
    माप_प्रकार bias_bytes;

    माप_प्रकार योग_bytes;
};

काष्ठा htp_mm_hvx_vtcm_layout {
    // Byte offsets from vtcm_base for each region
    माप_प्रकार off_act;           // vtcm_act (activation)
    माप_प्रकार off_src0;          // vtcm_src0 (weight/Wk)
    माप_प्रकार off_bias;          // vtcm_bias (bias / fused add only)
    माप_प्रकार off_dst;           // vtcm_dst (output scratch)
    माप_प्रकार off_act_raw;       // vtcm_act_raw (raw activation DMA staging)

    // Cached sizes
    माप_प्रकार src0_bytes;
    माप_प्रकार act_bytes;
    माप_प्रकार bias_bytes;
    माप_प्रकार dst_bytes;
    माप_प्रकार act_raw_bytes;

    माप_प्रकार योग_bytes;
};

अटल अंतरभूत व्योम htp_mm_hmx_vtcm_layout_build(
    काष्ठा htp_mm_hmx_vtcm_layout * L,
    पूर्णांक kernel_type,
    पूर्णांक wtype,
    uint32_t k,
    माप_प्रकार mc,
    माप_प्रकार nc,
    uint32_t group_आकार,
    bool pipeline,
    uint32_t act_threads,
    uint32_t aligned_tile_आकार,
    माप_प्रकार bias_आकार
) {
    माप_प्रकार off = 0;

    अगर (kernel_type == HTP_MM_KERNEL_HMX_BATCHED) {
        स्थिर माप_प्रकार vec_dot_आकार     = k * माप(uint16_t);
        स्थिर माप_प्रकार act_शीर्ष_stride   = mc * k;
        स्थिर माप_प्रकार weight_area_आकार  = hex_align_up(nc * vec_dot_आकार, HTP_MM_HMX_TILE_आकार);
        स्थिर माप_प्रकार activation_area_आकार = hex_align_up(group_आकार * act_शीर्ष_stride * माप(uint16_t), HTP_MM_HMX_TILE_आकार);
        स्थिर माप_प्रकार output_area_आकार  = hex_align_up(group_आकार * mc * nc * माप(uint16_t), HTP_MM_HMX_TILE_आकार);
        स्थिर माप_प्रकार scratch_area_आकार = hex_align_up(nc * vec_dot_आकार, HTP_MM_HMX_TILE_आकार);
        स्थिर माप_प्रकार min_f32_आकार = hex_align_up(act_threads * HTP_MM_DMA_ACT_MULTIPLIER * k * माप(भग्न), 128);

        // Group A: Permanent activation tiles and scales
        माप_प्रकार off_group_a = 0;
        VTCM_LAYOUT_ALLOC(off_group_a, off_act, activation_area_आकार);
        VTCM_LAYOUT_ALLOC(off_group_a, off_scales, HTP_MM_HMX_TILE_आकार); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_bias, hex_align_up(bias_आकार, HTP_MM_HMX_TILE_आकार), bias_आकार > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        माप_प्रकार off_group_b = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_b, off_weight[0], weight_area_आकार);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_weight[1], weight_area_आकार, false);
        VTCM_LAYOUT_ALLOC(off_group_b, off_dst[0], output_area_आकार);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_dst[1], output_area_आकार, false);
        VTCM_LAYOUT_ALLOC(off_group_b, off_scratch[0], scratch_area_आकार);
        VTCM_LAYOUT_ALLOC(off_group_b, off_scratch[1], scratch_area_आकार);

        स्थिर माप_प्रकार group_b_आकार = off_group_b - off_group_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        स्थिर माप_प्रकार max_f32_आकार = act_threads * 64 * k * माप(भग्न);
        स्थिर माप_प्रकार act_f32_आकार = hex_align_up(hex_smin(max_f32_आकार, hex_smax(min_f32_आकार, group_b_आकार)), 128);
        माप_प्रकार off_group_c = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_c, off_act_f32, act_f32_आकार);

        स्थिर माप_प्रकार group_c_आकार = off_group_c - off_group_a;

        L->weight_area_bytes = weight_area_आकार;
        L->act_area_bytes    = activation_area_आकार;
        L->act_f32_bytes     = act_f32_आकार;
        L->output_area_bytes = output_area_आकार;
        L->scratch_bytes[0]  = scratch_area_आकार;
        L->scratch_bytes[1]  = scratch_area_आकार;
        L->act_शीर्ष_stride   = act_शीर्ष_stride;
        L->bias_bytes        = bias_आकार;

        off = off_group_a + hex_smax(group_b_आकार, group_c_आकार);
    } अन्यथा {
        // HTP_MM_KERNEL_HMX_2D
        स्थिर bool is_quant = (wtype != HTP_TYPE_F16 && wtype != HTP_TYPE_F32 && wtype != HTP_TYPE_BF16);
        स्थिर माप_प्रकार row_stride = htp_tensor_get_row_आकार(wtype, k);
        स्थिर माप_प्रकार vec_dot_आकार = k * माप(uint16_t);
        स्थिर uint32_t n_weight_tiles = k / htp_tensor_get_tile_k(wtype);

        स्थिर माप_प्रकार min_f32_आकार = hex_align_up(act_threads * HTP_MM_DMA_ACT_MULTIPLIER * k * माप(भग्न), 128);
        स्थिर माप_प्रकार weight_area_आकार = is_quant
            ? hex_align_up((nc / 32) * n_weight_tiles * aligned_tile_आकार, HTP_MM_HMX_TILE_आकार)
            : hex_align_up(nc * row_stride, HTP_MM_HMX_TILE_आकार);
        स्थिर माप_प्रकार act_area_आकार    = hex_align_up(mc * vec_dot_आकार, HTP_MM_HMX_TILE_आकार);
        स्थिर माप_प्रकार output_area_आकार = hex_align_up(mc * nc * माप(__fp16), HTP_MM_HMX_TILE_आकार);

        स्थिर माप_प्रकार scratch0_आकार = hex_align_up(nc * vec_dot_आकार, HTP_MM_HMX_TILE_आकार);
        स्थिर माप_प्रकार scratch1_आकार = pipeline ? scratch0_आकार : 0;

        // Group A:  Scales and activation tiles (must not overlap with Group B or C)
        माप_प्रकार off_group_a = 0;
        VTCM_LAYOUT_ALLOC(off_group_a, off_scales, HTP_MM_HMX_TILE_आकार); // Padded to 2K for alignment and future persistent data
        VTCM_LAYOUT_ALLOC(off_group_a, off_act, act_area_आकार);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_bias, hex_align_up(bias_आकार, HTP_MM_HMX_TILE_आकार), bias_आकार > 0);

        // Group B: Compute-only buffers (starts at off_group_a)
        माप_प्रकार off_group_b = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_b, off_weight[0], weight_area_आकार);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_weight[1], weight_area_आकार, pipeline);
        VTCM_LAYOUT_ALLOC(off_group_b, off_dst[0], output_area_आकार);
        VTCM_LAYOUT_ALLOC(off_group_b, off_scratch[0], scratch0_आकार);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_scratch[1], scratch0_आकार, pipeline);
        VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_dst[1], output_area_आकार, pipeline);

        स्थिर माप_प्रकार group_b_आकार = off_group_b - off_group_a;

        // Group C: Activation prep temporary buffer (overlaps Group B, starting at off_group_a)
        स्थिर माप_प्रकार max_f32_आकार = act_threads * 64 * k * माप(भग्न);
        स्थिर माप_प्रकार act_f32_आकार = hex_align_up(hex_smin(max_f32_आकार, hex_smax(min_f32_आकार, group_b_आकार)), 128);
        माप_प्रकार off_group_c = off_group_a;
        VTCM_LAYOUT_ALLOC(off_group_c, off_act_f32, act_f32_आकार);

        स्थिर माप_प्रकार group_c_आकार = off_group_c - off_group_a;

        L->weight_area_bytes = weight_area_आकार;
        L->act_area_bytes    = act_area_आकार;
        L->act_f32_bytes     = act_f32_आकार;
        L->output_area_bytes = output_area_आकार;
        L->scratch_bytes[0]  = scratch0_आकार;
        L->scratch_bytes[1]  = scratch1_आकार;
        L->act_शीर्ष_stride   = 0;
        L->bias_bytes        = bias_आकार;

        off = off_group_a + hex_smax(group_b_आकार, group_c_आकार);
    }

    L->योग_bytes = off;
}

अटल अंतरभूत व्योम htp_mm_hvx_vtcm_layout_build(
    काष्ठा htp_mm_hvx_vtcm_layout * L,
    पूर्णांक kernel_type,
    पूर्णांक wtype,
    uint32_t ne10,       // k
    uint32_t act_nrows,  // m_total
    uint32_t n_threads,
    माप_प्रकार dst_row_आकार,
    माप_प्रकार src0_row_आकार,
    माप_प्रकार act_row_आकार,
    माप_प्रकार bias_row_आकार,
    uint32_t n_prefetch,
    bool is_matmul_id,
    bool is_fused_nx
) {
    (व्योम)act_row_आकार;
    माप_प्रकार src0_sz    = 0;
    माप_प्रकार act_sz     = 0;
    माप_प्रकार bias_sz    = bias_row_आकार > 0 ? htp_mm_round_up(bias_row_आकार, 128) : 0;
    माप_प्रकार dst_sz     = 0;
    माप_प्रकार act_raw_sz = 0;

    अगर (is_fused_nx) {
        uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(wtype);
        uint32_t n_k_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        uint32_t tile_row_आकार = n_k_tiles * aligned_tile_आकार;
        माप_प्रकार weight_sz_per_thread = hex_round_up(n_prefetch * tile_row_आकार, 128);

        माप_प्रकार tiled_act_row_आकार = htp_mm_tiled_act_row_आकार(wtype, ne10);
        माप_प्रकार q_act_sz = hex_round_up(tiled_act_row_आकार * act_nrows, 128);
        माप_प्रकार raw_row_आकार = hex_round_up(ne10 * माप(भग्न), QK_Q8_0_TILED * माप(भग्न));

        src0_sz    = weight_sz_per_thread * n_threads; // shared single-weight prefetch buffer
        act_sz     = q_act_sz;                         // quantized activation buffer
        bias_sz    = 0;
        dst_sz     = 0;
        act_raw_sz = hex_round_up(raw_row_आकार * act_nrows, 128);
    } अन्यथा अगर (is_matmul_id) {
        स्थिर माप_प्रकार act_row_आकार_tiled = htp_mm_tiled_act_row_आकार(wtype, ne10);
        स्थिर uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(wtype);
        स्थिर uint32_t n_k_tiles         = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
        स्थिर uint32_t tile_row_आकार     = n_k_tiles * aligned_tile_आकार;
        माप_प्रकार src0_sz_per_thread        = htp_mm_round_up(n_prefetch * tile_row_आकार, 256);

        act_sz = htp_mm_round_up(act_row_आकार_tiled * act_nrows, 256);

        माप_प्रकार raw_row_आकार = hex_round_up(ne10 * माप(भग्न), QK_Q8_0_TILED * माप(भग्न));

        src0_sz    = src0_sz_per_thread * n_threads;
        dst_sz     = 0;
        bias_sz    = 0;
        act_raw_sz = hex_round_up(raw_row_आकार * act_nrows, 128);
    } अन्यथा {
        स्थिर माप_प्रकार src0_row_आकार_padded = htp_mm_round_up(src0_row_आकार, 128);
        स्थिर माप_प्रकार dst_nrows = (act_nrows > 1) ? 0 : 1;

        चयन (kernel_type) {
            हाल HTP_MM_KERNEL_HVX_F16_F16_VTCM: {
                माप_प्रकार f16_act_row_आकार = htp_mm_round_up(ne10 * 2, 128);
                act_sz     = htp_mm_round_up(f16_act_row_आकार * act_nrows, 256);
                src0_sz    = htp_mm_round_up(n_prefetch * src0_row_आकार_padded, 256) * n_threads;
                dst_sz     = dst_nrows > 0 ? htp_mm_round_up(dst_row_आकार, 128) * n_threads : 0;
                act_raw_sz = hex_round_up(hex_round_up(ne10 * माप(भग्न), 128) * act_nrows, 128);
                अवरोध;
            }
            हाल HTP_MM_KERNEL_HVX_F32_F32_VTCM:
            हाल HTP_MM_KERNEL_HVX_BF16_F32_VTCM: {
                माप_प्रकार f32_act_row_आकार = htp_mm_round_up(ne10 * 4, 128);
                act_sz     = htp_mm_round_up(f32_act_row_आकार * act_nrows, 256);
                src0_sz    = htp_mm_round_up(n_prefetch * src0_row_आकार_padded, 256) * n_threads;
                dst_sz     = dst_nrows > 0 ? htp_mm_round_up(dst_row_आकार, 128) * n_threads : 0;
                act_raw_sz = 0;
                अवरोध;
            }
            हाल HTP_MM_KERNEL_HVX_QUANT_BLOCK:
            हाल HTP_MM_KERNEL_HVX_QUANT_ROW: {
                माप_प्रकार q_act_row_आकार = htp_mm_tiled_act_row_आकार(wtype, ne10);
                uint32_t aligned_tile_आकार = htp_tensor_get_tile_padded_आकार(wtype);
                uint32_t n_weight_tiles = (ne10 + htp_tensor_get_tile_k(wtype) - 1) / htp_tensor_get_tile_k(wtype);
                uint32_t tile_row_आकार = n_weight_tiles * aligned_tile_आकार;
                माप_प्रकार tiled_vtcm_आकार = htp_mm_round_up(n_prefetch * tile_row_आकार, 256);

                src0_sz = tiled_vtcm_आकार * n_threads;
                act_sz  = htp_mm_round_up(q_act_row_आकार * act_nrows, 256);

                माप_प्रकार dst_slice_per_thread = (dst_nrows > 0 && act_nrows == 1) ? htp_mm_round_up((dst_row_आकार + n_threads - 1) / n_threads, 128) : 0;
                dst_sz = dst_slice_per_thread * n_threads;
                माप_प्रकार raw_row_आकार = hex_round_up(ne10 * माप(भग्न), QK_Q8_0_TILED * माप(भग्न));
                act_raw_sz = hex_round_up(raw_row_आकार * act_nrows, 128);
                अवरोध;
            }
            शेष:
                अवरोध;
        }
    }

    // Group A: Persistent buffers across chunk compute
    माप_प्रकार off_group_a = 0;
    VTCM_LAYOUT_ALLOC(off_group_a, off_act, act_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_a, off_bias, bias_sz, bias_sz > 0);

    // Group B: Compute-only buffers (starts at off_group_a)
    माप_प्रकार off_group_b = off_group_a;
    VTCM_LAYOUT_ALLOC(off_group_b, off_src0, src0_sz);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_b, off_dst, dst_sz, dst_sz > 0);
    स्थिर माप_प्रकार group_b_आकार = off_group_b - off_group_a;

    // Group C: Raw activation staging buffer (overlaps Group B, starts at off_group_a)
    माप_प्रकार off_group_c = off_group_a;
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_group_c, off_act_raw, act_raw_sz, act_raw_sz > 0);
    स्थिर माप_प्रकार group_c_आकार = off_group_c - off_group_a;

    L->src0_bytes    = src0_sz;
    L->act_bytes     = act_sz;
    L->bias_bytes    = bias_sz;
    L->dst_bytes     = dst_sz;
    L->act_raw_bytes = act_raw_sz;
    L->योग_bytes   = off_group_a + hex_smax(group_b_आकार, group_c_आकार);
}

अटल अंतरभूत bool htp_mm_hvx_solve_vtcm_params(
    पूर्णांक kernel_type,
    पूर्णांक wtype,
    uint32_t ne10,
    uint32_t act_nrows,
    uint32_t n_threads,
    माप_प्रकार dst_row_आकार,
    माप_प्रकार src0_row_आकार,
    माप_प्रकार act_row_आकार,
    माप_प्रकार bias_row_आकार,
    uint32_t n_prefetch,
    माप_प्रकार vtcm_budget,
    काष्ठा htp_mm_hvx_vtcm_layout * L_out,
    uint32_t * m_chunk_out
) {
    काष्ठा htp_mm_hvx_vtcm_layout L;
    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, act_nrows, n_threads,
        dst_row_आकार, src0_row_आकार, act_row_आकार, bias_row_आकार, n_prefetch, false, false
    );

    अगर (L.योग_bytes <= vtcm_budget) {
        *L_out = L;
        *m_chunk_out = act_nrows;
        वापस true;
    }

    स्थिर माप_प्रकार fixed_bytes = L.src0_bytes + L.bias_bytes + L.dst_bytes;
    अगर (vtcm_budget <= fixed_bytes) {
        वापस false;
    }

    स्थिर माप_प्रकार avail_act = vtcm_budget - fixed_bytes;
    माप_प्रकार row_आकार = 0;
    अगर (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        row_आकार = htp_mm_tiled_act_row_आकार(wtype, ne10);
    } अन्यथा अगर (kernel_type == HTP_MM_KERNEL_HVX_F16_F16_VTCM) {
        row_आकार = hex_round_up(ne10 * 2, 128);
    } अन्यथा {
        row_आकार = hex_round_up(ne10 * 4, 128);
    }
    अगर (row_आकार == 0) {
        वापस false;
    }

    माप_प्रकार eff_row_आकार = row_आकार;
    अगर (kernel_type == HTP_MM_KERNEL_HVX_QUANT_ROW || kernel_type == HTP_MM_KERNEL_HVX_QUANT_BLOCK) {
        eff_row_आकार += hex_round_up(ne10 * माप(भग्न), QK_Q8_0_TILED * माप(भग्न));
    }

    uint32_t m_chunk = (uint32_t) (avail_act / eff_row_आकार);
    अगर (m_chunk > 1) {
        m_chunk &= ~1U;
    }
    अगर (m_chunk > act_nrows) {
        m_chunk = act_nrows;
    }
    अगर (m_chunk < 1) {
        वापस false;
    }

    htp_mm_hvx_vtcm_layout_build(
        &L, kernel_type, wtype, ne10, m_chunk, n_threads,
        dst_row_आकार, src0_row_आकार, act_row_आकार, bias_row_आकार, n_prefetch, false, false
    );

    जबतक (m_chunk > 2 && L.योग_bytes > vtcm_budget) {
        m_chunk -= 2;
        htp_mm_hvx_vtcm_layout_build(
            &L, kernel_type, wtype, ne10, m_chunk, n_threads,
            dst_row_आकार, src0_row_आकार, act_row_आकार, bias_row_आकार, n_prefetch, false, false
        );
    }

    अगर (L.योग_bytes <= vtcm_budget) {
        *L_out = L;
        *m_chunk_out = m_chunk;
        वापस true;
    }

    वापस false;
}

अटल अंतरभूत माप_प्रकार htp_mm_hmx_get_2d_vtcm_आकार(
    पूर्णांक wtype, uint32_t k, माप_प्रकार mc, माप_प्रकार nc, bool pipeline, uint32_t act_threads, uint32_t aligned_tile_आकार, माप_प्रकार bias_आकार
) {
    काष्ठा htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_2D, wtype, k, mc, nc, 1, pipeline, act_threads, aligned_tile_आकार, bias_आकार);
    वापस L.योग_bytes;
}

अटल अंतरभूत माप_प्रकार htp_mm_hmx_get_batched_vtcm_आकार(
    पूर्णांक wtype, uint32_t k, माप_प्रकार mc, माप_प्रकार nc, uint32_t group_आकार, bool pipeline, uint32_t act_threads, माप_प्रकार bias_आकार) {
    (व्योम)pipeline;
    काष्ठा htp_mm_hmx_vtcm_layout L;
    htp_mm_hmx_vtcm_layout_build(&L, HTP_MM_KERNEL_HMX_BATCHED, wtype, k, mc, nc, group_आकार, false, act_threads, 0, bias_आकार);
    वापस L.योग_bytes;
}

अटल अंतरभूत bool htp_mm_hmx_solve_batched_params(
    पूर्णांक wtype,
    uint32_t k,
    uint32_t ne01_padded,
    uint32_t ne11,
    uint32_t group_आकार,
    पूर्णांक n_threads,
    bool pipeline,
    माप_प्रकार bias_आकार,
    माप_प्रकार vtcm_budget,
    माप_प्रकार * m_chunk_out,
    माप_प्रकार * n_chunk_out,
    पूर्णांक * act_threads_out,
    माप_प्रकार * vtcm_आकार_out
) {
    माप_प्रकार best_mblocks = SIZE_MAX;
    पूर्णांक best_act_threads = 0;
    माप_प्रकार best_m_chunk = 0;
    माप_प्रकार best_n_chunk = 0;
    माप_प्रकार best_vtcm_आकार = 0;

    पूर्णांक act_threads = n_threads;
    जबतक (act_threads >= 1) {
        माप_प्रकार group_overhead = htp_mm_hmx_get_batched_overhead() + (bias_आकार > 0 ? hex_align_up(bias_आकार, HTP_MM_HMX_TILE_आकार) : 0);
        माप_प्रकार group_आकार_per_n, group_आकार_per_m, group_आकार_per_mn;
        htp_mm_hmx_get_batched_chunk_costs(k, group_आकार, &group_आकार_per_n, &group_आकार_per_m, &group_आकार_per_mn);

        माप_प्रकार m_chunk_candidate = 0;
        माप_प्रकार n_chunk_candidate = 0;
        माप_प्रकार vtcm_आकार_candidate = 0;

        अगर (htp_mm_hmx_compute_chunks(vtcm_budget, group_overhead, group_आकार_per_n, group_आकार_per_m, group_आकार_per_mn, hex_align_up(ne11, 32), ne01_padded,
                               (माप_प्रकार) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (माप_प्रकार) ne11 * HTP_MM_HMX_COST_A_CONVERT,
                                &m_chunk_candidate, &n_chunk_candidate, &vtcm_आकार_candidate) == 0) {
            माप_प्रकार exact_आकार = htp_mm_hmx_get_batched_vtcm_आकार(wtype, k, m_chunk_candidate, n_chunk_candidate, group_आकार, pipeline, act_threads, bias_आकार);
            अगर (exact_आकार <= vtcm_budget) {
                माप_प्रकार mblocks = ((माप_प्रकार) ne11 + m_chunk_candidate - 1) / m_chunk_candidate;
                अगर (mblocks < best_mblocks || (mblocks == best_mblocks && act_threads > best_act_threads)) {
                    best_mblocks = mblocks;
                    best_act_threads = act_threads;
                    best_m_chunk = m_chunk_candidate;
                    best_n_chunk = n_chunk_candidate;
                    best_vtcm_आकार = exact_आकार;
                }
            }
        }
        अगर (act_threads == 1) {
            act_threads = 0;
        } अन्यथा {
            act_threads /= 2;
        }
    }

    अगर (best_act_threads > 0) {
        *m_chunk_out = best_m_chunk;
        *n_chunk_out = best_n_chunk;
        *vtcm_आकार_out = best_vtcm_आकार;
        *act_threads_out = best_act_threads;
        वापस true;
    }
    वापस false;
}

अटल अंतरभूत bool htp_mm_hmx_solve_2d_params(
    पूर्णांक wtype,
    uint32_t k,
    uint32_t m_id_rows,
    uint32_t ne01_padded,
    uint32_t ne11_padded,
    uint32_t m_for_cost,
    पूर्णांक n_threads,
    bool pipeline,
    bool is_matmul_id,
    uint32_t aligned_tile_आकार,
    माप_प्रकार bias_आकार,
    माप_प्रकार vtcm_budget,
    माप_प्रकार * m_chunk_out,
    माप_प्रकार * n_chunk_out,
    पूर्णांक * act_threads_out,
    माप_प्रकार * vtcm_आकार_out
) {
    माप_प्रकार best_mblocks = SIZE_MAX;
    पूर्णांक best_act_threads = 0;
    माप_प्रकार best_m_chunk = 0;
    माप_प्रकार best_n_chunk = 0;
    माप_प्रकार best_vtcm_आकार = 0;

    स्थिर पूर्णांक m_for_chunks = is_matmul_id ? hex_align_up(m_id_rows, 32) : ne11_padded;

    पूर्णांक act_threads = n_threads;
    जबतक (act_threads >= 1) {
        माप_प्रकार simple_2d_overhead = htp_mm_hmx_get_2d_overhead(pipeline, is_matmul_id) + (bias_आकार > 0 ? hex_align_up(bias_आकार, HTP_MM_HMX_TILE_आकार) : 0);
        माप_प्रकार simple_2d_आकार_per_n, simple_2d_आकार_per_m, simple_2d_आकार_per_mn;
        htp_mm_hmx_get_2d_chunk_costs(wtype, k, pipeline, aligned_tile_आकार, &simple_2d_आकार_per_n, &simple_2d_आकार_per_m, &simple_2d_आकार_per_mn);

        माप_प्रकार m_chunk_candidate = 0;
        माप_प्रकार n_chunk_candidate = 0;
        माप_प्रकार vtcm_आकार_candidate = 0;

        अगर (htp_mm_hmx_compute_chunks(vtcm_budget, simple_2d_overhead, simple_2d_आकार_per_n, simple_2d_आकार_per_m, simple_2d_आकार_per_mn, m_for_chunks, ne01_padded,
                               (माप_प्रकार) ne01_padded * HTP_MM_HMX_COST_W_DEQUANT, (माप_प्रकार) m_for_cost * HTP_MM_HMX_COST_A_CONVERT,
                               &m_chunk_candidate, &n_chunk_candidate, &vtcm_आकार_candidate) == 0) {
            माप_प्रकार exact_आकार = htp_mm_hmx_get_2d_vtcm_आकार(wtype, k, m_chunk_candidate, n_chunk_candidate, pipeline, is_matmul_id ? 0 : act_threads, aligned_tile_आकार, bias_आकार);
            अगर (exact_आकार <= vtcm_budget) {
                माप_प्रकार mblocks = ((माप_प्रकार) m_for_cost + m_chunk_candidate - 1) / m_chunk_candidate;
                अगर (mblocks < best_mblocks || (mblocks == best_mblocks && act_threads > best_act_threads)) {
                    best_mblocks = mblocks;
                    best_act_threads = act_threads;
                    best_m_chunk = m_chunk_candidate;
                    best_n_chunk = n_chunk_candidate;
                    best_vtcm_आकार = exact_आकार;
                }
            }
        }
        अगर (act_threads == 1) {
            act_threads = 0;
        } अन्यथा {
            act_threads /= 2;
        }
    }

    अगर (best_act_threads > 0) {
        *m_chunk_out = best_m_chunk;
        *n_chunk_out = best_n_chunk;
        *vtcm_आकार_out = best_vtcm_आकार;
        *act_threads_out = best_act_threads;
        वापस true;
    }
    वापस false;
}

#अगर_घोषित __cplusplus
}
#पूर्ण_अगर

#पूर्ण_अगर // HTP_MATMUL_OPS_H
