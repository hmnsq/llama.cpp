#اگر_تعریف_نشده HTP_FLASH_ATTN_OPS_H
#تعریف_کن HTP_FLASH_ATTN_OPS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>

#ضمن "hex-fastdiv.h"
#ضمن "hex-common.h"
#ضمن "htp-vtcm.h"

#اگر_تعریف_شده __cplusplus
خارجي "C" {
#پایان_اگر

// Tile constants (mirrored from hmx-utils.h for use on host side if needed)
#تعریف_کن HTP_FA_HMX_TILE_اندازه   2048
#تعریف_کن HMX_FP16_TILE_اندازه     2048
#تعریف_کن HMX_FP16_TILE_N_ردیفها   32
#تعریف_کن HMX_FP16_TILE_N_تعداد_ستون   32
#تعریف_کن HMX_FP16_TILE_N_ELMS   1024

#تعریف_کن HVX_FA_DMA_CACHE_اندازه  128
#تعریف_کن HMX_FA_DMA_CACHE_اندازه  4


#تعریف_کن HTP_FA_M_INITIAL_VAL  -10000.0f

تعداد htp_fa_kernel_type {
    HTP_FA_KERNEL_UNSUPPORTED = 0,
    HTP_FA_KERNEL_HVX,
    HTP_FA_KERNEL_HMX
};

بنية htp_fa_kernel_params {
    uint8_t  kernel_type;        // enum htp_fa_kernel_type
    uint8_t  سر_تقسیم;         // 1 = partition by KV heads in multicore, 0 = token partition
    uint8_t  flags;              // reserved
    uint8_t  n_نخها;          // Number of threads to run

    // Common parameters
    uint16_t Br;
    uint16_t Bc;
    uint16_t n_kv_بلوکها;        // also HVX's n_blocks
    uint16_t G;                  // GQA factor (n_heads / n_kv_heads)

    عائم    scale;
    عائم    max_bias;
    عائم    logit_softcap;
    uint32_t vtcm_اندازه;

    uint32_t qrows;
    uint32_t qrows_در_هر_thread;
    uint32_t qrow_شروع;
    عائم    m0;
    عائم    m1;
    uint32_t n_سر_log2;

    بنية fastdiv_مقادیر src3_div2;
    بنية fastdiv_مقادیر src3_div3;

    بنية fastdiv_مقادیر پخش_rk2;
    بنية fastdiv_مقادیر پخش_rk3;
    بنية fastdiv_مقادیر پخش_rv2;
    بنية fastdiv_مقادیر پخش_rv3;

    اتحاد {
        بنية {
            uint32_t g_br;
            uint32_t ردیف_buf_stride;
            uint32_t ماسک_buf_ردیف_stride;
            int32_t  ماسک_پخش;
            int32_t  pipeline;
            بنية fastdiv_مقادیر div_G;
        } hmx;
        بنية {
            uint32_t اندازه_q_ردیف_padded;
            uint32_t اندازه_k_ردیف_padded;
            uint32_t اندازه_v_ردیف_padded;
            بنية fastdiv_مقادیر src0_div21;
            بنية fastdiv_مقادیر src0_div1;
        } hvx;
    } u;
};

#إذا defined(__cplusplus)
static_assert(حجم(بنية htp_fa_kernel_params) <= 128, "htp_fa_kernel_params is too large for kernel_params blob");
#پایان_اگر

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
بنية hmx_fa_vtcm_layout {
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
    size_t off_ردیف_bufs;
    size_t off_hmx_scales_id;
    size_t off_hmx_scales_qk;
    size_t off_ماسک_buf;
    size_t off_slopes;
    size_t off_sinks;

    // Region byte sizes reused by the device at runtime (not just for allocation).
    size_t q_tile_بایتها;
    size_t o_tile_بایتها;
    size_t k_tile_بایتها;
    size_t s_tile_بایتها;       // S and P tiles (same size)
    size_t d_tile_بایتها;       // d_tiles[0..1] + d_inv_l, allocated back to back
    size_t m_سطر_بایتها;       // one mask row
    size_t m_buf_جایگاه_بایتها;   // one dma_cache slot = align_up(Br * m_line_bytes, 4096)
    size_t ستون_vec_بایتها;
    size_t sinks_بایتها;

    // Derived strides.
    size_t ردیف_buf_stride;       // HVX vectors (128B) per row buffer
    size_t ماسک_buf_ردیف_stride;  // __fp16 elements per row in the mask buffer

    bool   pipeline;
    size_t مجموع_بایتها;
};

#تعریف_کن FA_MAX_KV_بلوکها 1024

// Build the VTCM layout.

ساكن مضمن فراغ hmx_fa_vtcm_layout_build(بنية hmx_fa_vtcm_layout * L,
                                            size_t gqa_ضریب, size_t DK, size_t DV,
                                            size_t Br, size_t Bc, size_t n_نخها,
                                            bool pipeline, bool is_q_fp32,
                                            bool has_sinks, size_t n_heads) {
    ثابت size_t g_br         = hex_align_به_بالا(gqa_ضریب * Br, HMX_FP16_TILE_N_ردیفها);
    ثابت size_t q_tile_اندازه  = hex_align_به_بالا(g_br * DK   * حجم(__fp16), HTP_FA_HMX_TILE_اندازه);
    ثابت size_t o_tile_اندازه  = hex_align_به_بالا(g_br * DV   * حجم(__fp16), HTP_FA_HMX_TILE_اندازه);
    ثابت size_t k_tile_اندازه  = hex_align_به_بالا(Bc   * DK   * حجم(__fp16), HTP_FA_HMX_TILE_اندازه);
    ثابت size_t v_tile_اندازه  = hex_align_به_بالا(Bc   * DV   * حجم(__fp16), HTP_FA_HMX_TILE_اندازه);
    ثابت size_t s_tile_اندازه  = hex_align_به_بالا(g_br * Bc   * حجم(__fp16), HTP_FA_HMX_TILE_اندازه);

    // The rescale matrices are diagonal: the HMX kernels only ever load the g_br/32
    // tiles that sit on the diagonal, so store just those, packed back to back with
    // a stride of one tile.  The old [g_br, g_br] square layout allocated g_br/32
    // times more than it used, which is also why a second D buffer was unaffordable.
    ثابت size_t d_tile_اندازه = (g_br / HMX_FP16_TILE_N_ردیفها) * HTP_FA_HMX_TILE_اندازه;

    ثابت size_t q_dma_اندازه   = hex_align_به_بالا(g_br * DK * (is_q_fp32 ? حجم(عائم) : حجم(__fp16)), 128);
    ثابت size_t k_dma_اندازه   = hex_align_به_بالا(Bc * hex_دور_به_بالا(DK * حجم(__fp16), 128), 128);
    ثابت size_t v_dma_اندازه   = hex_align_به_بالا(Bc * hex_دور_به_بالا(DV * حجم(__fp16), 128), 128);
    ثابت size_t ستون_vec_اندازه = hex_align_به_بالا(g_br * حجم(عائم),  256);
    ثابت size_t ردیف_vec_اندازه = hex_align_به_بالا(Bc   * حجم(__fp16), 256);
    ثابت size_t m_سطر_اندازه  = hex_align_به_بالا(Bc   * حجم(__fp16), 128);
    ثابت size_t m_buf_جایگاه   = hex_align_به_بالا(Br * m_سطر_اندازه, 256);
    ثابت size_t m_buf_اندازه   = m_buf_جایگاه * HMX_FA_DMA_CACHE_اندازه;
    ثابت size_t slopes_اندازه  = hex_align_به_بالا(g_br * حجم(__fp16), 128);
    ثابت size_t sinks_اندازه   = hex_دور_به_بالا(n_heads * حجم(عائم), 128);

    size_t off = 0;

    // Group A (Part 1 - HMX Tiled buffers)
    VTCM_LAYOUT_ALLOC(off, off_q_tiles,       q_tile_اندازه);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[0],    o_tile_اندازه);
    VTCM_LAYOUT_ALLOC(off, off_o_tiles[1],    o_tile_اندازه);
    VTCM_LAYOUT_ALLOC(off, off_d_tiles[0],    d_tile_اندازه);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_d_tiles[1], d_tile_اندازه, pipeline);
    VTCM_LAYOUT_ALLOC(off, off_d_inv_l,       d_tile_اندازه);

    // Group B & C share start offset (Group B tiles must be 2KB aligned)
    size_t off_گروه_b_c = hex_align_به_بالا(off, HTP_FA_HMX_TILE_اندازه);

    // Group B: Compute-only buffers
    size_t off_گروه_b = off_گروه_b_c;
    VTCM_LAYOUT_ALLOC(off_گروه_b, off_k_tiles[0],    k_tile_اندازه);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروه_b, off_k_tiles[1], k_tile_اندازه, pipeline);
    VTCM_LAYOUT_ALLOC(off_گروه_b, off_v_tiles[0],    v_tile_اندازه);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروه_b, off_v_tiles[1], v_tile_اندازه, pipeline);
    VTCM_LAYOUT_ALLOC(off_گروه_b, off_s_tiles[0],    s_tile_اندازه);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروه_b, off_s_tiles[1], s_tile_اندازه, pipeline);
    VTCM_LAYOUT_ALLOC(off_گروه_b, off_p_tiles[0],    s_tile_اندازه);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off_گروه_b, off_p_tiles[1], s_tile_اندازه, pipeline);
    VTCM_LAYOUT_ALLOC(off_گروه_b, off_s_rowmax,      ستون_vec_اندازه);
    VTCM_LAYOUT_ALLOC(off_گروه_b, off_p_rowsum,      ستون_vec_اندازه);
    VTCM_LAYOUT_ALLOC(off_گروه_b, off_ردیف_bufs,      ردیف_vec_اندازه * 2 * n_نخها);

    ثابت size_t گروه_b_اندازه = off_گروه_b - off_گروه_b_c;

    // Group C: Q fetch DMA buffer
    size_t off_گروه_c = off_گروه_b_c;
    VTCM_LAYOUT_ALLOC(off_گروه_c, off_q_dma,         q_dma_اندازه);

    ثابت size_t گروه_c_اندازه = off_گروه_c - off_گروه_b_c;

    off = off_گروه_b_c + hex_smax(گروه_b_اندازه, گروه_c_اندازه);

    // Group A (Part 2 - remaining non-HMX buffers)
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[0],     k_dma_اندازه);
    VTCM_LAYOUT_ALLOC(off, off_k_fp16[1],     k_dma_اندازه);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[0],     v_dma_اندازه);
    VTCM_LAYOUT_ALLOC(off, off_v_fp16[1],     v_dma_اندازه);
    VTCM_LAYOUT_ALLOC(off, off_m_vec,         ستون_vec_اندازه);
    VTCM_LAYOUT_ALLOC(off, off_l_vec,         ستون_vec_اندازه);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_id, 256);
    VTCM_LAYOUT_ALLOC(off, off_hmx_scales_qk, 256);
    VTCM_LAYOUT_ALLOC(off, off_ماسک_buf,      m_buf_اندازه);
    VTCM_LAYOUT_ALLOC(off, off_slopes,        slopes_اندازه);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, sinks_اندازه, has_sinks);

    L->q_tile_بایتها        = q_tile_اندازه;
    L->o_tile_بایتها        = o_tile_اندازه;
    L->k_tile_بایتها        = k_tile_اندازه;
    L->ستون_vec_بایتها       = ستون_vec_اندازه;
    L->s_tile_بایتها        = s_tile_اندازه;
    // Measured from the actual offsets rather than assumed to be N * d_tile_size, so
    // that inserting a region between them (or adding padding to VTCM_LAYOUT_ALLOC)
    // cannot silently leave the tail of the run unzeroed.
    L->d_tile_بایتها        = (L->off_d_inv_l + d_tile_اندازه) - L->off_d_tiles[0];
    L->m_سطر_بایتها        = m_سطر_اندازه;
    L->m_buf_جایگاه_بایتها    = m_buf_جایگاه;
    L->ردیف_buf_stride      = ردیف_vec_اندازه / 128;
    L->ماسک_buf_ردیف_stride = m_سطر_اندازه / حجم(__fp16);
    L->sinks_بایتها         = has_sinks ? sinks_اندازه : 0;
    L->pipeline            = pipeline;
    L->مجموع_بایتها         = off;
}

// Exact VTCM usage for a given (gqa_factor, DK, DV, Br, Bc) configuration.
ساكن مضمن size_t hmx_fa_compute_vtcm_usage(size_t gqa_ضریب, size_t DK, size_t DV, size_t Br, size_t Bc, size_t n_نخها, bool pipeline, bool is_q_fp32, bool has_sinks, size_t n_heads) {
    بنية hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, gqa_ضریب, DK, DV, Br, Bc, n_نخها, pipeline, is_q_fp32, has_sinks, n_heads);
    أرجع L.مجموع_بایتها;
}

#تعریف_کن FA_HVX_بلوک_اندازه 64
#تعریف_کن FA_HVX_G_MAX      8

بنية hvx_fa_vtcm_layout {
    size_t off_q;
    size_t off_k;
    size_t off_v;
    size_t off_m;
    size_t off_a;
    size_t off_sinks;

    size_t اندازه_q_بلوک;
    size_t اندازه_k_بلوک;
    size_t اندازه_v_بلوک;
    size_t اندازه_m_بلوک;
    size_t اندازه_vkq_acc;
    size_t اندازه_sinks;

    size_t مجموع_بایتها;
};

ساكن مضمن فراغ hvx_fa_vtcm_layout_build(بنية hvx_fa_vtcm_layout * L,
                                            size_t DK, size_t DV,
                                            bool is_q_fp32, bool has_ماسک,
                                            bool has_sinks, size_t n_heads,
                                            size_t n_نخها) {
    ثابت size_t اندازه_q_ردیف_padded = hex_دور_به_بالا(DK * (is_q_fp32 ? 4 : 2), 128);
    ثابت size_t اندازه_k_ردیف_padded = hex_دور_به_بالا(DK * حجم(__fp16), 128);
    ثابت size_t اندازه_v_ردیف_padded = hex_دور_به_بالا(DV * حجم(__fp16), 128);

    ثابت size_t اندازه_q_بلوک = اندازه_q_ردیف_padded * FA_HVX_G_MAX;
    ثابت size_t اندازه_k_بلوک = اندازه_k_ردیف_padded * FA_HVX_بلوک_اندازه;
    ثابت size_t اندازه_v_بلوک = اندازه_v_ردیف_padded * FA_HVX_بلوک_اندازه;
    ثابت size_t اندازه_m_بلوک = hex_دور_به_بالا(FA_HVX_بلوک_اندازه * حجم(__fp16), 128);
    ثابت size_t اندازه_vkq_acc = hex_دور_به_بالا(DV * حجم(عائم), 128) * FA_HVX_G_MAX;
    ثابت size_t اندازه_sinks   = hex_دور_به_بالا(n_heads * حجم(عائم), 128);

    size_t off = 0;

    VTCM_LAYOUT_ALLOC(off, off_q, اندازه_q_بلوک * n_نخها);
    VTCM_LAYOUT_ALLOC(off, off_k, اندازه_k_بلوک * 2 * n_نخها);
    VTCM_LAYOUT_ALLOC(off, off_v, اندازه_v_بلوک * 2 * n_نخها);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_m, اندازه_m_بلوک * HVX_FA_DMA_CACHE_اندازه * n_نخها, has_ماسک);
    VTCM_LAYOUT_ALLOC(off, off_a, اندازه_vkq_acc * n_نخها);
    VTCM_LAYOUT_ALLOC_OPTIONAL(off, off_sinks, اندازه_sinks, has_sinks);

    L->اندازه_q_بلوک = اندازه_q_بلوک;
    L->اندازه_k_بلوک = اندازه_k_بلوک;
    L->اندازه_v_بلوک = اندازه_v_بلوک;
    L->اندازه_m_بلوک = اندازه_m_بلوک;
    L->اندازه_vkq_acc = اندازه_vkq_acc;
    L->اندازه_sinks   = has_sinks ? اندازه_sinks : 0;
    L->مجموع_بایتها  = off;
}

ساكن مضمن size_t hvx_fa_compute_vtcm_usage(size_t DK, size_t DV, bool is_q_fp32, bool has_ماسک, bool has_sinks, size_t n_heads, size_t n_نخها) {
    بنية hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, DK, DV, is_q_fp32, has_ماسک, has_sinks, n_heads, n_نخها);
    أرجع L.مجموع_بایتها;
}

#تعریف_کن FA_MIN_KV_بلوکها 3

// Cost-based (Br, Bc) search for flash attention with pipeline constraint.
ساكن مضمن صحيح hmx_fa_پیدا_کن_chunk_اندازه(size_t * Br_بیرون,
                                         size_t * Bc_بیرون,
                                         size_t   gqa_ضریب,
                                         size_t   DK,
                                         size_t   DV,
                                         size_t   qo_len,
                                         size_t   kv_len,
                                         size_t   vtcm_budget,
                                         size_t   n_نخها,
                                         bool     is_q_fp32,
                                         bool     has_sinks,
                                         size_t   n_heads) {
    ثابت size_t T       = HMX_FP16_TILE_N_ردیفها;  // 32
    ثابت size_t br_unit = hmx_ceil_div(T, gqa_ضریب);
    ثابت size_t bc_unit = HMX_FP16_TILE_N_تعداد_ستون * 2;  // 64
    ثابت bool   can_pipeline = (kv_len >= FA_MIN_KV_بلوکها * bc_unit && n_نخها >= 2);

    // Br_max: largest Br aligned to br_unit that does not exceed qo_len.
    ثابت size_t Br_max = qo_len >= br_unit ? hex_align_پایین(qo_len, br_unit) : br_unit;

    // Pipeline constraint: cap Bc so n_kv_blocks >= FA_MIN_KV_BLOCKS.
    // Only relax when kv_len is too short to form enough blocks.
    size_t Bc_حد     = can_pipeline ? hex_align_پایین(kv_len / FA_MIN_KV_بلوکها, bc_unit) :
                                         (kv_len >= bc_unit ? hex_align_پایین(kv_len, bc_unit) : bc_unit);
    // For small-Q SWA-sized KV caches (qo_len <= 16, kv_len <= 2048), cap Bc at 256 for block-skipping granularity
    إذا (qo_len <= 16 && kv_len <= 2048 && Bc_حد > 256) {
        Bc_حد = 256;
    }
    // Cost coefficients calibrated from profiling
    ثابت size_t c_q_fixed    = 800;   // per-Q-block: q_load + epilogue o_update + o_norm + o_store
    ثابت size_t c_iter_پایه  = 200;   // per-KV-iter base (HMX dot/update + DMA)
    ثابت size_t c_softmax    = 600;   // per 64-row vector chunk on HVX

    size_t بهترین_هزینه = SIZE_MAX, بهترین_mn = 0;
    size_t بهترین_Br = 0, بهترین_Bc = 0;

    لكل (size_t Br = Br_max; Br >= br_unit; Br -= br_unit) {
        // Try all Bc candidates from Bc_limit down to bc_unit
        لكل (size_t Bc = Bc_حد; Bc >= bc_unit; Bc -= bc_unit) {
            size_t vtcm_needed = hmx_fa_compute_vtcm_usage(gqa_ضریب, DK, DV, Br, Bc, n_نخها, can_pipeline, is_q_fp32, has_sinks, n_heads);
            إذا (vtcm_needed <= vtcm_budget) {
                // This Bc fits for this Br!
                ثابت size_t q_بلوکها       = (qo_len + Br - 1) / Br;
                ثابت size_t kv_بلوکها      = (kv_len + Bc - 1) / Bc;
                ثابت size_t actual_نخها = (kv_بلوکها >= 3 && n_نخها >= 2) ? n_نخها : 1;
                ثابت size_t n_ردیفها_g       = Br * gqa_ضریب;
                ثابت size_t n_ردیف_vec_cnt  = (n_ردیفها_g + 63) / 64;
                ثابت size_t n_use          = n_ردیف_vec_cnt < actual_نخها ? n_ردیف_vec_cnt : actual_نخها;
                ثابت size_t vecs_در_هر_t     = n_use > 0 ? (n_ردیف_vec_cnt + n_use - 1) / n_use : 1;

                ثابت size_t c_iter_actual  = c_iter_پایه + c_softmax * vecs_در_هر_t;
                ثابت size_t هزینه           = q_بلوکها * (c_q_fixed + kv_بلوکها * c_iter_actual);
                ثابت size_t mn             = Br * Bc;

                إذا (هزینه < بهترین_هزینه || (هزینه == بهترین_هزینه && mn > بهترین_mn)) {
                    بهترین_هزینه = هزینه;
                    بهترین_mn   = mn;
                    بهترین_Br   = Br;
                    بهترین_Bc   = Bc;
                }
                // Since we iterate Bc from largest to smallest, this is the largest Bc that fits
                // for this Br. We can break to the next Br.
                اكسر;
            }
        }

        إذا (Br == br_unit) {
            اكسر;
        }
    }

    إذا (بهترین_Br == 0 || بهترین_Bc == 0) {
        أرجع -1;
    }

    *Br_بیرون = بهترین_Br;
    *Bc_بیرون = بهترین_Bc;
    أرجع 0;
}

#اگر_تعریف_شده __cplusplus
}
#پایان_اگر

#پایان_اگر /* HTP_FLASH_ATTN_OPS_H */
