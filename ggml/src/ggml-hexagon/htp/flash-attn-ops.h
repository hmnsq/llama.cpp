#اگر_غیر_تعریف_شدہ HTP_FLASH_ATTN_OPS_H
#تعریف_کرو HTP_FLASH_ATTN_OPS_H

#شامل <stdint.h>
#شامل <stddef.h>
#شامل <stdbool.h>

#شامل "hex-fastdiv.h"
#شامل "hex-common.h"
#شامل "htp-vtcm.h"

#اگر_تعریف_شدہ __cplusplus
خارجی "C" {
#اگر_ختم

// Tile constants (mirrored from hmx-utils.h for use on host side if needed)
#تعریف_کرو HTP_FA_HMX_TILE_سائز   2048
#تعریف_کرو HMX_FP16_TILE_سائز     2048
#تعریف_کرو HMX_FP16_TILE_N_صفیں   32
#تعریف_کرو HMX_FP16_TILE_N_ستون_تعداد   32
#تعریف_کرو HMX_FP16_TILE_N_ELMS   1024

#تعریف_کرو HVX_FA_DMA_CACHE_سائز  128
#تعریف_کرو HMX_FA_DMA_CACHE_سائز  4


#تعریف_کرو HTP_FA_M_INITIAL_VAL  -10000.0f

شمار htp_fa_kernel_type {
    HTP_FA_KERNEL_UNSUPPORTED = 0,
    HTP_FA_KERNEL_HVX,
    HTP_FA_KERNEL_HMX
};

ساخت htp_fa_kernel_params {
    uint8_t  kernel_type;        // enum htp_fa_kernel_type
    uint8_t  سر_بانٹو;         // 1 = partition by KV heads in multicore, 0 = token partition
    uint8_t  flags;              // reserved
    uint8_t  n_دھاگے;          // Number of threads to run

    // Common parameters
    uint16_t Br;
    uint16_t Bc;
    uint16_t n_kv_بلاکیں;        // also HVX's n_blocks
    uint16_t G;                  // GQA factor (n_heads / n_kv_heads)

    اعشاری    scale;
    اعشاری    max_bias;
    اعشاری    logit_softcap;
    uint32_t vtcm_سائز;

    uint32_t qrows;
    uint32_t qrows_فی_thread;
    uint32_t qrow_آغاز;
    اعشاری    m0;
    اعشاری    m1;
    uint32_t n_سر_log2;

    ساخت fastdiv_اقدار src3_div2;
    ساخت fastdiv_اقدار src3_div3;

    ساخت fastdiv_اقدار نشر_rk2;
    ساخت fastdiv_اقدار نشر_rk3;
    ساخت fastdiv_اقدار نشر_rv2;
    ساخت fastdiv_اقدار نشر_rv3;

    اتحاد {
        ساخت {
            uint32_t g_br;
            uint32_t صف_buf_stride;
            uint32_t نقاب_buf_صف_stride;
            int32_t  نقاب_نشر;
            int32_t  pipeline;
            ساخت fastdiv_اقدار div_G;
        } hmx;
        ساخت {
            uint32_t سائز_q_صف_padded;
            uint32_t سائز_k_صف_padded;
            uint32_t سائز_v_صف_padded;
            ساخت fastdiv_اقدار src0_div21;
            ساخت fastdiv_اقدار src0_div1;
        } hvx;
    } u;
};

#اگر defined(__cplusplus)
static_assert(حجم(ساخت htp_fa_kernel_params) <= 128, "htp_fa_kernel_params is too large for kernel_params blob");
#اگر_ختم

// VTCM region layout for the HMX flash-attention kernel.
//
// Single source of truth for both the host (which needs the total size to pick a
// (Br, Bc) tiling that fits the VTCM budget) and the device (which needs the actual
// byte offsets to place each scratch buffer). Building the layout once and reading
// offsets/total from it makes host estimate and device allocation impossible to
// desync -- previously they were duplicated formulas in two files and drifted.
//
// All fields are byte offsets / byte sizes -- no HVX_Vector type is named here so the
// header stays host-includable. The device casts (base + off_*) to the proper type.
// An offset of 0 marks a region that is not allocated for this configuration (only
// off_v_tiles[1], which exists only when pipelining); the device sets such pointers NULL.
ساخت hmx_fa_vtcm_layout {
    // Byte offsets from vtcm_base for each region.
    size_t off_q_tiles;
    size_t off_q_dma;
    size_t off_o_tiles[2];
    size_t off_k_fp16[2];
    size_t off_v_fp16[2];
    size_t off_k_tiles[2];
    size_t off_v_tiles[2];
    size_t off_s_tiles[2];
    size_t off_p_tiles[2];
    size_t off_d_tiles[2];
    size_t off_d_inv_l;
    size_t off_m_vec;
    size_t off_l_vec;
    size_t off_s_rowmax;
    size_t off_p_rowsum;
    size_t off_صف_bufs;
    size_t off_hmx_scales_id;
    size_t off_hmx_scales_qk;
    size_t off_نقاب_buf;
    size_t off_slopes;
    size_t off_sinks;

    // Region byte sizes reused by the device at runtime (not just for allocation).
    size_t q_tile_بائٹیں;
    size_t o_tile_بائٹیں;
    size_t k_tile_بائٹیں;
    size_t s_tile_بائٹیں;       // S and P tiles (same size)
    size_t d_tile_بائٹیں;       // d_tiles[0..1] + d_inv_l, allocated back to back
    size_t m_سطر_بائٹیں;       // one mask row
    size_t m_buf_خانہ_بائٹیں;   // one dma_cache slot = align_up(Br * m_line_bytes, 4096)
    size_t ستون_vec_بائٹیں;
    size_t sinks_بائٹیں;

    // Derived strides.
    size_t صف_buf_stride;       // HVX vectors (128B) per row buffer
    size_t نقاب_buf_صف_stride;  // __fp16 elements per row in the mask buffer

    bool   pipeline;
    size_t کل_بائٹیں;
};

#تعریف_کرو FA_MAX_KV_بلاکیں 1024

// Build the VTCM layout.

ساکن اندرونی خالی hmx_fa_vtcm_layout_build(ساخت hmx_fa_vtcm_layout * L,
                                            size_t gqa_ضربی_عدد, size_t DK, size_t DV,
                                            size_t Br, size_t Bc, size_t n_دھاگے,
                                            bool pipeline, bool is_q_fp32,
                                            bool has_sinks, size_t n_heads) {
    ثابت size_t g_br         = hex_align_اوپر(gqa_ضربی_عدد * Br, HMX_FP16_TILE_N_صفیں);
    ثابت size_t q_tile_سائز  = hex_align_اوپر(g_br * DK   * حجم(__fp16), HTP_FA_HMX_TILE_سائز);
    ثابت size_t o_tile_سائز  = hex_align_اوپر(g_br * DV   * حجم(__fp16), HTP_FA_HMX_TILE_سائز);
    ثابت size_t k_tile_سائز  = hex_align_اوپر(Bc   * DK   * حجم(__fp16), HTP_FA_HMX_TILE_سائز);
    ثابت size_t v_tile_سائز  = hex_align_اوپر(Bc   * DV   * حجم(__fp16), HTP_FA_HMX_TILE_سائز);
    ثابت size_t s_tile_سائز  = hex_align_اوپر(g_br * Bc   * حجم(__fp16), HTP_FA_HMX_TILE_سائز);

    // The rescale matrices are diagonal: the HMX kernels only ever load the g_br/32
    // tiles that sit on the diagonal, so store just those, packed back to back with
    // a stride of one tile.  The old [g_br, g_br] square layout allocated g_br/32
    // times more than it used, which is also why a second D buffer was unaffordable.
    ثابت size_t d_tile_سائز = (g_br / HMX_FP16_TILE_N_صفیں) * HTP_FA_HMX_TILE_سائز;

    ثابت size_t q_dma_سائز   = hex_align_اوپر(g_br * DK * (is_q_fp32 ? حجم(اعشاری) : حجم(__fp16)), 128);
    ثابت size_t k_dma_سائز   = hex_align_اوپر(Bc * hex_باری_اوپر(DK * حجم(__fp16), 128), 128);
    ثابت size_t v_dma_سائز   = hex_align_اوپر(Bc * hex_باری_اوپر(DV * حجم(__fp16), 128), 128);
    ثابت size_t ستون_vec_سائز = hex_align_اوپر(g_br * حجم(اعشاری),  256);
    ثابت size_t صف_vec_سائز = hex_align_اوپر(Bc   * حجم(__fp16), 256);
    ثابت size_t m_سطر_سائز  = hex_align_اوپر(Bc   * حجم(__fp16), 128);
    ثابت size_t m_buf_خانہ   = hex_align_اوپر(Br * m_سطر_سائز, 256);
    ثابت size_t m_buf_سائز   = m_buf_خانہ * HMX_FA_DMA_CACHE_سائز;
    ثابت size_t slopes_سائز  = hex_align_اوپر(g_br * حجم(__fp16), 128);
    ثابت size_t sinks_سائز   = hex_باری_اوپر(n_heads * حجم(اعشاری), 128);

    size_t off = 0;

    // Group A (Part 1 - HMX Tiled buffers)
    VTCM_LAYOUT_ALLOC(off, off_q_tiles,       q_tile_سائز);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[0],    o_tile_سائز);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[1],    o_tile_سائز);
    VTCM_LAYOUT_ALLOC(off, off_d_tiles[0],    d_tile_سائز);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_d_tiles[1], d_tile_سائز, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_d_inv_l,       d_tile_سائز);

    // Group B & C share start offset (Group B tiles must be 2KB aligned)
    size_t off_گروہ_b_c = hex_align_اوپر(off, HTP_FA_HMX_TILE_سائز);

    // Group B: Compute-only buffers
    size_t off_گروہ_b = off_گروہ_b_c;
    VTCM_LAYOUT_ALLOC(off_گروہ_b, off_k_tiles[0],    k_tile_سائز);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_k_tiles[1], k_tile_سائز, pipeline);
    VTCM_LAYOUT_ALLOC(off_گروہ_b, off_v_tiles[0],    v_tile_سائز);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_v_tiles[1], v_tile_سائز, pipeline);
    VTCM_LAYOUT_ALLOC(off_گروہ_b, off_s_tiles[0],    s_tile_سائز);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_s_tiles[1], s_tile_سائز, pipeline);
    VTCM_LAYOUT_ALLOC(off_گروہ_b, off_p_tiles[0],    s_tile_سائز);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروہ_b, off_p_tiles[1], s_tile_سائز, pipeline);
    VTCM_LAYOUT_ALLOC(off_گروہ_b, off_s_rowmax,      ستون_vec_سائز);
    VTCM_LAYOUT_ALLOC(off_گروہ_b, off_p_rowsum,      ستون_vec_سائز);
    VTCM_LAYOUT_ALLOC(off_گروہ_b, off_صف_bufs,      صف_vec_سائز * 2 * n_دھاگے);

    ثابت size_t گروہ_b_سائز = off_گروہ_b - off_گروہ_b_c;

    // Group C: Q fetch DMA buffer
    size_t off_گروہ_c = off_گروہ_b_c;
    VTCM_LAYOUT_ALLOC(off_گروہ_c, off_q_dma,         q_dma_سائز);

    ثابت size_t گروہ_c_سائز = off_گروہ_c - off_گروہ_b_c;

    off = off_گروہ_b_c + hex_smax(گروہ_b_سائز, گروہ_c_سائز);

    // Group A (Part 2 - remaining non-HMX buffers)
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[0],     k_dma_سائز);
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[1],     k_dma_سائز);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[0],     v_dma_سائز);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[1],     v_dma_سائز);
    VTCM_LAYOUT_ALLOC(off, off_m_vec,         ستون_vec_سائز);
    VTCM_LAYOUT_ALLOC(off, off_l_vec,         ستون_vec_سائز);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_id, 256);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_qk, 256);
    VTCM_LAYOUT_ALLOC(off, off_نقاب_buf,      m_buf_سائز);
    VTCM_LAYOUT_ALLOC(off, off_slopes,        slopes_سائز);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, sinks_سائز, has_sinks);

    L->q_tile_بائٹیں        = q_tile_سائز;
    L->o_tile_بائٹیں        = o_tile_سائز;
    L->k_tile_بائٹیں        = k_tile_سائز;
    L->ستون_vec_بائٹیں       = ستون_vec_سائز;
    L->s_tile_بائٹیں        = s_tile_سائز;
    // Measured from the actual offsets rather than assumed to be N * d_tile_size, so
    // that inserting a region between them (or adding padding to VTCM_LAYOUT_ALLOC)
    // cannot silently leave the tail of the run unzeroed.
    L->d_tile_بائٹیں        = (L->off_d_inv_l + d_tile_سائز) - L->off_d_tiles[0];
    L->m_سطر_بائٹیں        = m_سطر_سائز;
    L->m_buf_خانہ_بائٹیں    = m_buf_خانہ;
    L->صف_buf_stride      = صف_vec_سائز / 128;
    L->نقاب_buf_صف_stride = m_سطر_سائز / حجم(__fp16);
    L->sinks_بائٹیں         = has_sinks ? sinks_سائز : 0;
    L->pipeline            = pipeline;
    L->کل_بائٹیں         = off;
}

// Exact VTCM usage for a given (gqa_factor, DK, DV, Br, Bc) configuration.
ساکن اندرونی size_t hmx_fa_compute_vtcm_usage(size_t gqa_ضربی_عدد, size_t DK, size_t DV, size_t Br, size_t Bc, size_t n_دھاگے, bool pipeline, bool is_q_fp32, bool has_sinks, size_t n_heads) {
    ساخت hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, gqa_ضربی_عدد, DK, DV, Br, Bc, n_دھاگے, pipeline, is_q_fp32, has_sinks, n_heads);
    واپس L.کل_بائٹیں;
}

#تعریف_کرو FA_HVX_بلاک_سائز 64
#تعریف_کرو FA_HVX_G_MAX      8

ساخت hvx_fa_vtcm_layout {
    size_t off_q;
    size_t off_k;
    size_t off_v;
    size_t off_m;
    size_t off_a;
    size_t off_sinks;

    size_t سائز_q_بلاک;
    size_t سائز_k_بلاک;
    size_t سائز_v_بلاک;
    size_t سائز_m_بلاک;
    size_t سائز_vkq_acc;
    size_t سائز_sinks;

    size_t کل_بائٹیں;
};

ساکن اندرونی خالی hvx_fa_vtcm_layout_build(ساخت hvx_fa_vtcm_layout * L,
                                            size_t DK, size_t DV,
                                            bool is_q_fp32, bool has_نقاب,
                                            bool has_sinks, size_t n_heads,
                                            size_t n_دھاگے) {
    ثابت size_t سائز_q_صف_padded = hex_باری_اوپر(DK * (is_q_fp32 ? 4 : 2), 128);
    ثابت size_t سائز_k_صف_padded = hex_باری_اوپر(DK * حجم(__fp16), 128);
    ثابت size_t سائز_v_صف_padded = hex_باری_اوپر(DV * حجم(__fp16), 128);

    ثابت size_t سائز_q_بلاک = سائز_q_صف_padded * FA_HVX_G_MAX;
    ثابت size_t سائز_k_بلاک = سائز_k_صف_padded * FA_HVX_بلاک_سائز;
    ثابت size_t سائز_v_بلاک = سائز_v_صف_padded * FA_HVX_بلاک_سائز;
    ثابت size_t سائز_m_بلاک = hex_باری_اوپر(FA_HVX_بلاک_سائز * حجم(__fp16), 128);
    ثابت size_t سائز_vkq_acc = hex_باری_اوپر(DV * حجم(اعشاری), 128) * FA_HVX_G_MAX;
    ثابت size_t سائز_sinks   = hex_باری_اوپر(n_heads * حجم(اعشاری), 128);

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_q, سائز_q_بلاک * n_دھاگے);
    VTCM_LAYOUT_ALLOC(off, off_k, سائز_k_بلاک * 2 * n_دھاگے);
    VTCM_LAYOUT_ALLOC(off, off_v, سائز_v_بلاک * 2 * n_دھاگے);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_m, سائز_m_بلاک * HVX_FA_DMA_CACHE_سائز * n_دھاگے, has_نقاب);
    VTCM_LAYOUT_ALLOC(off, off_a, سائز_vkq_acc * n_دھاگے);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, سائز_sinks, has_sinks);

    L->سائز_q_بلاک = سائز_q_بلاک;
    L->سائز_k_بلاک = سائز_k_بلاک;
    L->سائز_v_بلاک = سائز_v_بلاک;
    L->سائز_m_بلاک = سائز_m_بلاک;
    L->سائز_vkq_acc = سائز_vkq_acc;
    L->سائز_sinks   = has_sinks ? سائز_sinks : 0;
    L->کل_بائٹیں  = off;
}

ساکن اندرونی size_t hvx_fa_compute_vtcm_usage(size_t DK, size_t DV, bool is_q_fp32, bool has_نقاب, bool has_sinks, size_t n_heads, size_t n_دھاگے) {
    ساخت hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, DK, DV, is_q_fp32, has_نقاب, has_sinks, n_heads, n_دھاگے);
    واپس L.کل_بائٹیں;
}

#تعریف_کرو FA_MIN_KV_بلاکیں 3

// Cost-based (Br, Bc) search for flash attention with pipeline constraint.
ساکن اندرونی صحیح hmx_fa_ڈھونڈو_chunk_سائز(size_t * Br_برآمد,
                                         size_t * Bc_برآمد,
                                         size_t   gqa_ضربی_عدد,
                                         size_t   DK,
                                         size_t   DV,
                                         size_t   qo_len,
                                         size_t   kv_len,
                                         size_t   vtcm_budget,
                                         size_t   n_دھاگے,
                                         bool     is_q_fp32,
                                         bool     has_sinks,
                                         size_t   n_heads) {
    ثابت size_t T       = HMX_FP16_TILE_N_صفیں;  // 32
    ثابت size_t br_unit = hmx_ceil_div(T, gqa_ضربی_عدد);
    ثابت size_t bc_unit = HMX_FP16_TILE_N_ستون_تعداد * 2;  // 64
    ثابت bool   can_pipeline = (kv_len >= FA_MIN_KV_بلاکیں * bc_unit && n_دھاگے >= 2);

    // Br_max: largest Br aligned to br_unit that does not exceed qo_len.
    ثابت size_t Br_max = qo_len >= br_unit ? hex_align_نیچے_کو(qo_len, br_unit) : br_unit;

    // Pipeline constraint: cap Bc so n_kv_blocks >= FA_MIN_KV_BLOCKS.
    // Only relax when kv_len is too short to form enough blocks.
    size_t Bc_حد_بندی     = can_pipeline ? hex_align_نیچے_کو(kv_len / FA_MIN_KV_بلاکیں, bc_unit) :
                                         (kv_len >= bc_unit ? hex_align_نیچے_کو(kv_len, bc_unit) : bc_unit);
    // For small-Q SWA-sized KV caches (qo_len <= 16, kv_len <= 2048), cap Bc at 256 for block-skipping granularity
    اگر (qo_len <= 16 && kv_len <= 2048 && Bc_حد_بندی > 256) {
        Bc_حد_بندی = 256;
    }
    // Cost coefficients calibrated from profiling
    ثابت size_t c_q_fixed    = 800;   // per-Q-block: q_load + epilogue o_update + o_norm + o_store
    ثابت size_t c_iter_بنیاد  = 200;   // per-KV-iter base (HMX dot/update + DMA)
    ثابت size_t c_softmax    = 600;   // per 64-row vector chunk on HVX

    size_t بہترین_لاگت = SIZE_MAX, بہترین_mn = 0;
    size_t بہترین_Br = 0, بہترین_Bc = 0;

    برائے (size_t Br = Br_max; Br >= br_unit; Br -= br_unit) {
        // Try all Bc candidates from Bc_limit down to bc_unit
        برائے (size_t Bc = Bc_حد_بندی; Bc >= bc_unit; Bc -= bc_unit) {
            size_t vtcm_needed = hmx_fa_compute_vtcm_usage(gqa_ضربی_عدد, DK, DV, Br, Bc, n_دھاگے, can_pipeline, is_q_fp32, has_sinks, n_heads);
            اگر (vtcm_needed <= vtcm_budget) {
                // This Bc fits for this Br!
                ثابت size_t q_بلاکیں       = (qo_len + Br - 1) / Br;
                ثابت size_t kv_بلاکیں      = (kv_len + Bc - 1) / Bc;
                ثابت size_t actual_دھاگے = (kv_بلاکیں >= 3 && n_دھاگے >= 2) ? n_دھاگے : 1;
                ثابت size_t n_صفیں_g       = Br * gqa_ضربی_عدد;
                ثابت size_t n_صف_vec_cnt  = (n_صفیں_g + 63) / 64;
                ثابت size_t n_use          = n_صف_vec_cnt < actual_دھاگے ? n_صف_vec_cnt : actual_دھاگے;
                ثابت size_t vecs_فی_t     = n_use > 0 ? (n_صف_vec_cnt + n_use - 1) / n_use : 1;

                ثابت size_t c_iter_actual  = c_iter_بنیاد + c_softmax * vecs_فی_t;
                ثابت size_t لاگت           = q_بلاکیں * (c_q_fixed + kv_بلاکیں * c_iter_actual);
                ثابت size_t mn             = Br * Bc;

                اگر (لاگت < بہترین_لاگت || (لاگت == بہترین_لاگت && mn > بہترین_mn)) {
                    بہترین_لاگت = لاگت;
                    بہترین_mn   = mn;
                    بہترین_Br   = Br;
                    بہترین_Bc   = Bc;
                }
                // Since we iterate Bc from largest to smallest, this is the largest Bc that fits
                // for this Br. We can break to the next Br.
                توڑو;
            }
        }

        اگر (Br == br_unit) {
            توڑو;
        }
    }

    اگر (بہترین_Br == 0 || بہترین_Bc == 0) {
        واپس -1;
    }

    *Br_برآمد = بہترین_Br;
    *Bc_برآمد = بہترین_Bc;
    واپس 0;
}

#اگر_تعریف_شدہ __cplusplus
}
#اگر_ختم

#اگر_ختم /* HTP_FLASH_ATTN_OPS_H */
