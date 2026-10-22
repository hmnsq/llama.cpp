#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <assert.h>
#अन्तर्भाव <HAP_compute_res.h>
#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव <stdatomic.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hex-fastdiv.h"
#अन्तर्भाव "hex-profile.h"
#अन्तर्भाव "hmx-queue.h"
#अन्तर्भाव "hmx-utils.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "hvx-dump.h"
#अन्तर्भाव "hvx-copy.h"
#अन्तर्भाव "hvx-reduce.h"
#अन्तर्भाव "hvx-flash-attn.h"
#अन्तर्भाव "htp-vtcm.h"
#अन्तर्भाव "work-queue.h"

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "hvx-quant.h"

#अन्तर्भाव "flash-attn-ops.h"
#अन्तर्भाव "hvx-fa-kernels.h"
#अन्तर्भाव "hmx-fa-kernels.h"

// Must be multiple of 32
#define FLASH_ATTN_BLOCK_SIZE (32 * 2)

संरचना htp_fa_context {
    स्थिरांक संरचना htp_ops_context * octx;

    संरचना fastdiv_मूल्ये src0_div21;
    संरचना fastdiv_मूल्ये src0_div1;

    संरचना fastdiv_मूल्ये broadcast_rk2;
    संरचना fastdiv_मूल्ये broadcast_rk3;
    संरचना fastdiv_मूल्ये broadcast_rv2;
    संरचना fastdiv_मूल्ये broadcast_rv3;

    संरचना fastdiv_मूल्ये src3_div2;
    संरचना fastdiv_मूल्ये src3_div3;

    दशांश scale;
    दशांश max_bias;
    bool  has_softcap;
    __fp16 logit_softcap;

    uint32_t n_head_log2;
    दशांश m0;
    दशांश m1;
    __fp16 slopes[512];

    uint32_t n_blocks;
    uint32_t n_active_blocks;
    bool     shared_active_blocks;

    माप_प्रकार size_q_row_padded;
    माप_प्रकार size_k_row_padded;
    माप_प्रकार size_v_row_padded;

    माप_प्रकार size_k_block;
    माप_प्रकार size_v_block;
    माप_प्रकार size_m_block;

    uint32_t qrows;
    uint32_t qrows_per_thread;
    uint32_t qrow_start;

    bool is_q_fp32;

    माप_प्रकार size_q_block;
    माप_प्रकार size_vkq_acc;

    uint8_t *  spad_q;
    uint8_t *  spad_k;
    uint8_t *  spad_v;
    uint8_t *  spad_m;
    uint8_t *  spad_a;
    दशांश *    spad_sinks;
    uint16_t   active_blocks[FA_MAX_KV_BLOCKS];

    स्थिरांक संरचना htp_tensor * k;
    स्थिरांक संरचना htp_tensor * v;
};

संरचना hmx_fa_context {
    स्थिरांक संरचना htp_ops_context * octx;
    स्थिरांक संरचना htp_tensor *      sinks;  // attention sinks (src[4]), NULL if absent
    स्थिरांक संरचना htp_tensor *      k;
    स्थिरांक संरचना htp_tensor *      v;
    bool         pipeline;  // true when n_kv_blocks >= FA_MIN_KV_BLOCKS && n_threads >= 2
    uint32_t     n_threads;

    // Op parameters
    __fp16       scale;
    दशांश        max_bias;
    bool         has_softcap;
    __fp16       logit_softcap;
    uint32_t     n_head_log2;
    दशांश        m0, m1;

    // Dimensions
    uint32_t     DK, DV;
    uint32_t     DK_pad, DV_pad;  // head_dim rounded up to 64 for HMX tiling
    uint32_t     n_kv;        // kv_len
    uint32_t     n_kv_heads;  // number of KV heads
    uint32_t     n_heads;     // number of Q heads
    uint32_t     G;           // GQA factor = n_heads / n_kv_heads
    संरचना fastdiv_मूल्ये div_G;
    संरचना fastdiv_मूल्ये src3_div2;
    संरचना fastdiv_मूल्ये src3_div3;
    uint32_t     n_kv_blocks;
    uint32_t     neq1;        // Q token count

    // Types
    bool         is_q_fp32;
    bool         is_dst_fp32;

    // Dynamic block sizes
    uint32_t     Br;    // Q tokens per block (before GQA expansion)
    uint32_t     Bc;
    uint32_t     g_br;  // hex_align_up(G * Br, 32) - actual tile row dim

    // VTCM buffers (allocated by vtcm_seq_alloc)
    __fp16 *     vtcm_q_dma;           // Q DMA fetch buffer
    __fp16 *     vtcm_q_tiles;         // Q tile format [g_br, D]
    __fp16 *     vtcm_o_tiles[2];      // O ping-pong [g_br, D]
    __fp16 *     vtcm_k_fp16[2];       // K DMA double-buffer [Bc, D]
    __fp16 *     vtcm_v_fp16[2];       // V DMA double-buffer [Bc, D]
    __fp16 *     vtcm_k_tiles[2];      // K tiles (transposed, double-buffered)
    __fp16 *     vtcm_v_tiles[2];      // V tiles (column-major, double-buffered)
    __fp16 *     vtcm_s_tiles[2];      // S = QK^T [g_br, Bc] (double-buffered)
    __fp16 *     vtcm_p_tiles[2];      // P = softmax(S) [g_br, Bc]
    __fp16 *     vtcm_d_tiles[2];      // Diagonal rescale, g_br/32 packed diagonal tiles (double-buffered)
    __fp16 *     vtcm_d_inv_l;         // Diagonal rescale (1/l), same packed layout
    HVX_Vector * vtcm_m_vec;           // Row max [g_br]
    HVX_Vector * vtcm_l_vec;           // Row sum [g_br]
    HVX_Vector * vtcm_s_rowmax;        // Softmax intermediate [g_br]
    HVX_Vector * vtcm_p_rowsum;        // Softmax intermediate [g_br]
    HVX_Vector * vtcm_row_bufs;        // Per-thread softmax row scratch [n_threads][2][Bc/64]
    uint8_t *    vtcm_hmx_scales_id;   // HMX output scales (identity)
    uint8_t *    vtcm_hmx_scales_qk;   // HMX output scales (qk_scale)
    __fp16 *     vtcm_mask_buf;        // VTCM mask buffer [Br * m_line], DMA'd per KV block
    __fp16 *     vtcm_slopes;          // ALiBi slopes [g_br]
    दशांश *      vtcm_sinks;           // Attention sinks
    माप_प्रकार       row_buf_stride;       // HVX vectors per row buffer (Bc/64)
    माप_प्रकार       mask_buf_row_stride;  // elements (__fp16) per row in mask buffer
    माप_प्रकार       q_tile_bytes;
    माप_प्रकार       k_tile_bytes;
    माप_प्रकार       o_tile_bytes;
    माप_प्रकार       col_vec_bytes;
    माप_प्रकार       d_tile_bytes;
    bool         mask_broadcast;       // true when mask->ne[2] == 1 (head-independent, single 2D DMA)
    dma_cache_fa m_cache;
};

// Scan mask rows [q_start, q_start + n_rows_q) and build compact list of KV blocks with any entry != -inf.
स्थिर uint32_t fa_scan_active_blocks(
        dma_queue *               dma_q,
        स्थिरांक संरचना htp_tensor * mask,
        uint32_t                  q_start,
        uint32_t                  n_rows_q,
        uint32_t                  im2,
        uint32_t                  im3,
        uint32_t                  nek1,
        uint32_t                  Bc,
        uint32_t                  n_kv_blocks,
        शून्य *                    vtcm_scratch,
        माप_प्रकार                    scratch_bytes,
        uint16_t *                active_blocks) {
    स्थिरांक माप_प्रकार row_bytes  = (माप_प्रकार) nek1 * आकार(__fp16);
    स्थिरांक माप_प्रकार row_stride = hex_round_up(row_bytes, 128);

    जर (!mask || n_kv_blocks <= 1 || n_kv_blocks > FA_MAX_KV_BLOCKS ||
        (माप_प्रकार) n_rows_q * row_stride > scratch_bytes) {
        #pragma clang loop vectorize(disable)
        साठी (uint32_t b = 0; b < n_kv_blocks; ++b) {
            active_blocks[b] = (uint16_t) b;
        }
        परत n_kv_blocks;
    }

    स्थिरांक dma_addr_t m_src = mask->data + q_start * mask->nb[1] + im2 * mask->nb[2] + im3 * mask->nb[3];
    dma_queue_ढकला(dma_q, dma_make_data(vtcm_scratch, m_src), row_stride, mask->nb[1], row_bytes, n_rows_q);
    स्थिरांक uint8_t * m_buf = (स्थिरांक uint8_t *) dma_queue_काढा(dma_q).dst;

    स्थिरांक HVX_Vector v_ninf   = Q6_Vh_vsplat_R(0xFC00);
    uint32_t         n_active = 0;

    साठी (uint32_t b = 0; b < n_kv_blocks; ++b) {
        स्थिरांक uint32_t col0 = b * Bc;
        स्थिरांक uint32_t cols = MIN(Bc, nek1 - col0);

        साठी (uint32_t r = 0; r < n_rows_q; ++r) {
            स्थिरांक __fp16 * row_ptr = (स्थिरांक __fp16 *) (m_buf + r * row_stride) + col0;
            HVX_Vector v_any = Q6_V_vzero();
            uint32_t c = 0;
            साठी (; c + 64 <= cols; c += 64) {
                स्थिरांक HVX_Vector v_m = *(स्थिरांक HVX_Vector *) (row_ptr + c);
                v_any = Q6_V_vor_VV(v_any, Q6_V_vxor_VV(v_m, v_ninf));
            }
            जर (c < cols) {
                स्थिरांक uint32_t rem = cols - c;
                स्थिरांक HVX_VectorPred q_rem = Q6_Q_vsetq2_R(rem * आकार(__fp16));
                स्थिरांक HVX_Vector v_m = *(स्थिरांक HVX_Vector *) (row_ptr + c);
                स्थिरांक HVX_Vector v_diff = Q6_V_vand_QV(q_rem, Q6_V_vxor_VV(v_m, v_ninf));
                v_any = Q6_V_vor_VV(v_any, v_diff);
            }
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 64));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 32));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 16));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 8));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 4));
            जर (Q6_R_vextract_VR(v_any, 0) != 0) {
                active_blocks[n_active++] = (uint16_t) b;
                थांबा;
            }
        }
    }

    जर (n_active == 0 && n_kv_blocks > 0) {
        active_blocks[0] = 0;
        n_active = 1;
    }
    परत n_active;
}

स्थिर शून्य flash_attn_ext_f16_thread(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, शून्य * data) {
    संरचना htp_fa_context * factx = (संरचना htp_fa_context *) data;
    स्थिरांक संरचना htp_ops_context * octx = factx->octx;
    स्थिरांक संरचना htp_tensor * q     = octx->src[0];
    स्थिरांक संरचना htp_tensor * k     = octx->src[1];
    स्थिरांक संरचना htp_tensor * v     = octx->src[2];
    स्थिरांक संरचना htp_tensor * mask  = octx->src[3];
    स्थिरांक संरचना htp_tensor * sinks = octx->src[4];
    स्थिरांक संरचना htp_tensor * dst   = octx->dst;

    स्थिरांक uint32_t neq0 = q->ne[0];
    स्थिरांक uint32_t neq1 = q->ne[1];
    स्थिरांक uint32_t neq2 = q->ne[2];
    स्थिरांक uint32_t neq3 = q->ne[3];

    स्थिरांक uint32_t nek0 = k->ne[0];
    स्थिरांक uint32_t nek1 = k->ne[1];
    स्थिरांक uint32_t nek2 = k->ne[2];
    स्थिरांक uint32_t nek3 = k->ne[3];

    स्थिरांक uint32_t nev0 = v->ne[0];
    स्थिरांक uint32_t nev1 = v->ne[1];
    स्थिरांक uint32_t nev2 = v->ne[2];
    स्थिरांक uint32_t nev3 = v->ne[3];

    स्थिरांक uint32_t nbq1 = q->nb[1];
    स्थिरांक uint32_t nbq2 = q->nb[2];
    स्थिरांक uint32_t nbq3 = q->nb[3];

    स्थिरांक uint32_t nbk1 = k->nb[1];
    स्थिरांक uint32_t nbk2 = k->nb[2];
    स्थिरांक uint32_t nbk3 = k->nb[3];

    स्थिरांक uint32_t nbv1 = v->nb[1];
    स्थिरांक uint32_t nbv2 = v->nb[2];
    स्थिरांक uint32_t nbv3 = v->nb[3];

    स्थिरांक uint32_t ne1 = dst->ne[1];
    स्थिरांक uint32_t ne2 = dst->ne[2];
    स्थिरांक uint32_t ne3 = dst->ne[3];

    स्थिरांक uint32_t nb1 = dst->nb[1];
    स्थिरांक uint32_t nb2 = dst->nb[2];
    स्थिरांक uint32_t nb3 = dst->nb[3];

    // total rows in q
    स्थिरांक uint32_t dr  = factx->qrows_per_thread;
    स्थिरांक uint32_t ir0 = factx->qrow_start + dr * ith;
    स्थिरांक uint32_t ir1 = MIN(ir0 + dr, factx->qrow_start + factx->qrows);

    जर (ir0 >= ir1) परत;

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    dma_queue * dma_q = octx->ctx->dma[ith];

    स्थिरांक uint32_t DK = nek0;
    स्थिरांक uint32_t DV = nev0;

    स्थिरांक माप_प्रकार size_q_row = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    स्थिरांक माप_प्रकार size_k_row = htp_tensor_get_row_size(k->type, DK);
    स्थिरांक माप_प्रकार size_v_row = htp_tensor_get_row_size(v->type, DV);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = factx->spad_q + factx->size_q_block * ith;
    uint8_t * spad_k = factx->spad_k + factx->size_k_block * 2 * ith;
    uint8_t * spad_v = factx->spad_v + factx->size_v_block * 2 * ith;
    uint8_t * spad_m = factx->spad_m + (mask ? factx->size_m_block * HVX_FA_DMA_CACHE_SIZE : 0) * ith;
    uint8_t * spad_a = factx->spad_a + factx->size_vkq_acc * ith;

    dma_cache_dm m_cache;
    dma_cache_dm_init(&m_cache, spad_m, factx->size_m_block, HVX_FA_DMA_CACHE_SIZE);

    uint16_t thread_active_blocks[FA_MAX_KV_BLOCKS];
    स्थिरांक uint16_t * active_blocks = factx->shared_active_blocks ? factx->active_blocks : thread_active_blocks;
    uint32_t n_active_blocks       = factx->n_active_blocks;
    dma_addr_t prev_mp_base        = 0;

    जर (!factx->shared_active_blocks && !mask) {
        #pragma clang loop vectorize(disable)
        साठी (uint32_t b = 0; b < factx->n_blocks; ++b) {
            thread_active_blocks[b] = (uint16_t) b;
        }
    }

    स्थिरांक माप_प्रकार size_vkq_acc_single = hex_round_up(DV * आकार(दशांश), 128);

    uint32_t ir = ir0;
    पर्यंत (ir < ir1) {
        स्थिरांक uint32_t iq3 = fastdiv(ir, &factx->src0_div21);
        स्थिरांक uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &factx->src0_div1);
        स्थिरांक uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        स्थिरांक uint32_t ik3 = fastdiv(iq3, &factx->broadcast_rk3);
        स्थिरांक uint32_t ik2 = fastdiv(iq2, &factx->broadcast_rk2);

        स्थिरांक uint32_t iv3 = fastdiv(iq3, &factx->broadcast_rv3);
        स्थिरांक uint32_t iv2 = fastdiv(iq2, &factx->broadcast_rv2);

        uint32_t G_local = 1;
        जर (neq1 == 1 && (mask == रिक्त || mask->ne[2] == 1)) {
            पर्यंत (ir + G_local < ir1 && G_local < FA_HVX_G_MAX) {
                स्थिरांक uint32_t next_ir = ir + G_local;
                स्थिरांक uint32_t next_iq3 = fastdiv(next_ir, &factx->src0_div21);
                स्थिरांक uint32_t next_iq2 = fastdiv(next_ir - next_iq3*neq2*neq1, &factx->src0_div1);
                स्थिरांक uint32_t next_iq1 = (next_ir - next_iq3*neq2*neq1 - next_iq2 * neq1);

                स्थिरांक uint32_t next_ik3 = fastdiv(next_iq3, &factx->broadcast_rk3);
                स्थिरांक uint32_t next_ik2 = fastdiv(next_iq2, &factx->broadcast_rk2);

                स्थिरांक uint32_t next_iv3 = fastdiv(next_iq3, &factx->broadcast_rv3);
                स्थिरांक uint32_t next_iv2 = fastdiv(next_iq2, &factx->broadcast_rv2);

                जर (next_ik2 != ik2 || next_ik3 != ik3 || next_iv2 != iv2 || next_iv3 != iv3 || next_iq1 != iq1 || next_iq3 != iq3) {
                    थांबा;
                }
                G_local++;
            }
        }

        dma_addr_t mp_base = 0;
        जर (mask) {
            स्थिरांक uint32_t im2 = fastmodulo(iq2, mask->ne[2], &factx->src3_div2);
            स्थिरांक uint32_t im3 = fastmodulo(iq3, mask->ne[3], &factx->src3_div3);
            mp_base = mask->data + iq1*mask->nb[1] + im2*mask->nb[2] + im3*mask->nb[3];
            जर (!factx->shared_active_blocks && mp_base != prev_mp_base) {
                n_active_blocks = fa_scan_active_blocks(
                    dma_q, mask, iq1, 1, im2, im3, nek1, FLASH_ATTN_BLOCK_SIZE,
                    factx->n_blocks, spad_k, factx->size_k_block * 2, thread_active_blocks);
                prev_mp_base = mp_base;
            }
        }

        uint32_t heads[FA_HVX_G_MAX];
        HVX_Vector slope_vecs[FA_HVX_G_MAX] __attribute__((aligned(128)));
        HVX_Vector S_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        HVX_Vector M_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        uint8_t * q_ptrs[FA_HVX_G_MAX];
        दशांश * vkq_ptrs[FA_HVX_G_MAX];

        साठी (uint32_t g = 0; g < G_local; ++g) {
            स्थिरांक uint32_t r = ir + g;
            स्थिरांक uint32_t r_iq3 = fastdiv(r, &factx->src0_div21);
            स्थिरांक uint32_t r_iq2 = fastdiv(r - r_iq3*neq2*neq1, &factx->src0_div1);
            स्थिरांक uint32_t r_iq1 = (r - r_iq3*neq2*neq1 - r_iq2 * neq1);

            heads[g] = r_iq2;
            स्थिरांक __fp16 slope = factx->slopes[r_iq2];
            slope_vecs[g] = hvx_vec_splat_f16(slope);

            S_vec[g] = hvx_vec_splat_f32(0.0f);
            M_vec[g] = hvx_vec_splat_f32(HTP_FA_M_INITIAL_VAL);

            uint8_t * q_dst = spad_q + g * factx->size_q_row_padded;
            q_ptrs[g] = q_dst;

            दशांश * vkq_dst = (दशांश *)(spad_a + g * size_vkq_acc_single);
            vkq_ptrs[g] = vkq_dst;
            hvx_splat_f32_a((uint8_t *) vkq_dst, 0, DV);

            // Fetch Q row g
            स्थिरांक dma_addr_t q_row_ptr = q->data + r_iq1*nbq1 + r_iq2*nbq2 + r_iq3*nbq3;
            dma_queue_ढकला(dma_q, dma_make_data(q_dst, q_row_ptr), factx->size_q_row_padded, nbq1, size_q_row, 1);
        }

        // Prefetch first two active blocks
        साठी (uint32_t step = 0; step < MIN(n_active_blocks, 2); ++step) {
            स्थिरांक uint32_t ib = active_blocks[step];
            स्थिरांक uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
            स्थिरांक uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

            // K
            स्थिरांक dma_addr_t k_src = k->data + ic_start*nbk1 + ik2*nbk2 + ik3*nbk3;
            uint8_t * k_dst = spad_k + (step % 2) * factx->size_k_block;
            dma_queue_ढकला(dma_q, dma_make_data(k_dst, k_src), factx->size_k_row_padded, nbk1, size_k_row, current_block_size);

            // V
            स्थिरांक dma_addr_t v_src = v->data + ic_start*nbv1 + iv2*nbv2 + iv3*nbv3;
            uint8_t * v_dst = spad_v + (step % 2) * factx->size_v_block;
            dma_queue_ढकला(dma_q, dma_make_data(v_dst, v_src), factx->size_v_row_padded, nbv1, size_v_row, current_block_size);

            // Mask
            जर (mask) {
                स्थिरांक dma_addr_t m_src = mp_base + ic_start * आकार(__fp16);
                dma_cache_dm_ढकला(dma_q, &m_cache, ib, m_src, current_block_size * 2, current_block_size * 2, current_block_size * 2, 1);
            }
        }

        // Pop all Q rows
        साठी (uint32_t g = 0; g < G_local; ++g) {
            uint8_t * q_ptr_vtcm = (शून्य *) dma_queue_काढा(dma_q).dst;
            जर (factx->is_q_fp32) {
                hvx_copy_f16_f32_aa(q_ptr_vtcm, q_ptr_vtcm, DK);
            }
        }

        स्थिरांक HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
        स्थिरांक bool has_softcap     = factx->has_softcap;
        स्थिरांक HVX_Vector v_cap     = has_softcap ? hvx_vec_splat_f16(factx->logit_softcap) : Q6_V_vzero();
        स्थिरांक HVX_Vector vinf      = Q6_Vh_vsplat_R(0xFC00);
        स्थिरांक HVX_Vector vmin      = Q6_Vh_vsplat_R(0xFBFF);
        स्थिरांक HVX_Vector v_log2e   = hvx_vec_splat_f16(EXP_LOG2E_F);
        स्थिरांक uint32_t stride_v2   = factx->size_v_row_padded * 2;

        साठी (uint32_t step = 0; step < n_active_blocks; ++step) {
            स्थिरांक uint32_t ib = active_blocks[step];
            स्थिरांक uint32_t ic_start = ib * FLASH_ATTN_BLOCK_SIZE;
            स्थिरांक uint32_t current_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - ic_start);

            // Wait for DMA
            uint8_t * k_base = (शून्य *) dma_queue_काढा(dma_q).dst; // K
            uint8_t * v_base = (शून्य *) dma_queue_काढा(dma_q).dst; // V
            __fp16  * m_base = mask ? (__fp16 *) dma_queue_काढा(dma_q).dst : रिक्त; // M

            जर (factx->k->type == HTP_TYPE_Q8_0) {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
                साठी (uint32_t r = 0; r < current_block_size; ++r) {
                    __fp16 * row_k = (__fp16 *)(k_base + r * factx->size_k_row_padded);
                    hvx_dequantize_row_flat_q8_0_f16(row_k, row_k, DK);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
            }
            जर (factx->v->type == HTP_TYPE_Q8_0) {
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
                साठी (uint32_t r = 0; r < current_block_size; ++r) {
                    __fp16 * row_v = (__fp16 *)(v_base + r * factx->size_v_row_padded);
                    hvx_dequantize_row_flat_q8_0_f16(row_v, row_v, DV);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
            }

            साठी (uint32_t g = 0; g < G_local; ++g) {
                स्थिरांक uint32_t head_ir = ir + g;
                uint8_t * q_ptr_vtcm = q_ptrs[g];
                दशांश * VKQ32 = vkq_ptrs[g];
                स्थिरांक HVX_Vector slope_vec = slope_vecs[g];

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_QK, head_ir);

                HVX_Vector scores_f16 = Q6_V_vzero();
                जर (current_block_size > 0) {
                    HVX_Vector scores0 = hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_base, factx->size_k_row_padded, DK, factx->scale);
                    HVX_Vector scores1 = (current_block_size > 32) ? hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_base + 32 * factx->size_k_row_padded, factx->size_k_row_padded, DK, factx->scale) : Q6_V_vzero();
                    scores_f16 = hvx_vec_f32_to_f16(scores0, scores1);
                }

                जर (has_softcap) {
                    scores_f16 = hvx_vec_tanh_f16(scores_f16);
                    scores_f16 = hvx_vec_mul_f16_f16(scores_f16, v_cap);
                }

                HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(current_block_size * आकार(__fp16));

                जर (mask) {
                    HVX_Vector m_vals_f16 = *(स्थिरांक HVX_UVector *) m_base;
                    HVX_VectorPred is_inf = Q6_Q_vcmp_eq_VhVh(m_vals_f16, vinf);
                    m_vals_f16 = Q6_V_vmux_QVV(is_inf, vmin, m_vals_f16);

                    HVX_Vector m_scaled = hvx_vec_mul_f16_f16(m_vals_f16, slope_vec);
                    scores_f16 = Q6_V_vmux_QVV(q_tail_keep, hvx_vec_add_f16_f16(scores_f16, m_scaled), v_neg_inf);
                } अन्यथा {
                    scores_f16 = Q6_V_vmux_QVV(q_tail_keep, scores_f16, v_neg_inf);
                }

                HVX_Vector v_max_f16 = hvx_vec_reduce_max_f16(scores_f16);
                HVX_Vector v_max     = Q6_V_lo_W(hvx_vec_f16_to_f32(v_max_f16));
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_QK, head_ir);

                // prefetch K for step + 2 after last head finished QK
                जर (g + 1 == G_local && step + 2 < n_active_blocks) {
                    स्थिरांक uint32_t next_ib = active_blocks[step + 2];
                    स्थिरांक uint32_t next_ic_start = next_ib * FLASH_ATTN_BLOCK_SIZE;
                    स्थिरांक uint32_t next_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - next_ic_start);

                    स्थिरांक dma_addr_t k_src = k->data + next_ic_start*nbk1 + ik2*nbk2 + ik3*nbk3;
                    dma_queue_ढकला(dma_q, dma_make_data(k_base, k_src), factx->size_k_row_padded, nbk1, size_k_row, next_block_size);
                }

                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_SFM, head_ir);
                {
                    HVX_Vector M_new_vec = Q6_Vsf_vmax_VsfVsf(v_max, M_vec[g]);
                    HVX_Vector diff_vec  = HVX_OP_SUB_F32(M_vec[g], M_new_vec);

                    HVX_Vector diff_f16   = hvx_vec_f32_to_f16(diff_vec, diff_vec);
                    HVX_Vector diff_base2 = hvx_vec_mul_f16_f16(diff_f16, v_log2e);
                    HVX_Vector ms_f16     = hvx_vec_exp2_f16(diff_base2);
                    HVX_Vector ms_vec     = Q6_V_lo_W(hvx_vec_f16_to_f32(ms_f16));

                    M_vec[g] = M_new_vec;

                    HVX_Vector v_m_vec_f16 = hvx_vec_f32_to_f16(M_vec[g], M_vec[g]);
                    HVX_Vector v_s_minus_m = Q6_Vqf16_vsub_VhfVhf(scores_f16, v_m_vec_f16);
                    HVX_Vector v_s_minus_m_base2 = hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m), v_log2e);

                    HVX_Vector P = hvx_vec_exp2_f16(v_s_minus_m_base2);
                    P = Q6_V_vmux_QVV(q_tail_keep, P, Q6_V_vzero());

                    HVX_VectorPair P_pair = hvx_vec_f16_to_f32(P);
                    HVX_Vector P0 = Q6_V_lo_W(P_pair);
                    HVX_Vector P1 = Q6_V_hi_W(P_pair);
                    HVX_Vector p_sum_vec = hvx_vec_reduce_sum_f32(HVX_OP_ADD_F32(P0, P1));

                    S_vec[g] = HVX_OP_ADD_F32(HVX_OP_MUL_F32(S_vec[g], ms_vec), p_sum_vec);

                    स्थिरांक uint8_t * v_ptr = v_base;

                    जर (DV == 64) {
                        HVX_VectorPair vkq0 = *((स्थिरांक HVX_VectorPair *) VKQ32);
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );

                        साठी (uint32_t j = 0; j < current_block_size; j += 2) {
                            HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            स्थिरांक HVX_Vector * vx0 = (स्थिरांक HVX_Vector *) v_ptr;
                            जर (j + 1 == current_block_size) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                थांबा;
                            }

                            HVX_Vector S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            स्थिरांक HVX_Vector * vx1 = (स्थिरांक HVX_Vector *) (v_ptr + factx->size_v_row_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            v_ptr += stride_v2;
                        }

                        *((HVX_VectorPair *) VKQ32) = vkq0;
                    } अन्यथा जर (DV == 128) {
                        HVX_VectorPair vkq0 = ((स्थिरांक HVX_VectorPair *) VKQ32)[0];
                        HVX_VectorPair vkq1 = ((स्थिरांक HVX_VectorPair *) VKQ32)[1];
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );
                        vkq1 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq1), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq1), ms_vec)
                        );

                        साठी (uint32_t j = 0; j < current_block_size; j += 2) {
                            HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            स्थिरांक HVX_Vector * vx0 = (स्थिरांक HVX_Vector *) v_ptr;
                            जर (j + 1 == current_block_size) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                थांबा;
                            }

                            HVX_Vector S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            स्थिरांक HVX_Vector * vx1 = (स्थिरांक HVX_Vector *) (v_ptr + factx->size_v_row_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx1[1]), S1);
                            v_ptr += stride_v2;
                        }

                        ((HVX_VectorPair *) VKQ32)[0] = vkq0;
                        ((HVX_VectorPair *) VKQ32)[1] = vkq1;
                    } अन्यथा जर (DV == 256) {
                        HVX_VectorPair vkq0 = ((स्थिरांक HVX_VectorPair *) VKQ32)[0];
                        HVX_VectorPair vkq1 = ((स्थिरांक HVX_VectorPair *) VKQ32)[1];
                        HVX_VectorPair vkq2 = ((स्थिरांक HVX_VectorPair *) VKQ32)[2];
                        HVX_VectorPair vkq3 = ((स्थिरांक HVX_VectorPair *) VKQ32)[3];
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );
                        vkq1 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq1), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq1), ms_vec)
                        );
                        vkq2 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq2), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq2), ms_vec)
                        );
                        vkq3 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq3), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq3), ms_vec)
                        );

                        साठी (uint32_t j = 0; j < current_block_size; j += 2) {
                            HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            स्थिरांक HVX_Vector * vx0 = (स्थिरांक HVX_Vector *) v_ptr;
                            जर (j + 1 == current_block_size) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                vkq2 = hvx_vec_mpyacc_f32_f16(vkq2, Q6_Vh_vshuff_Vh(vx0[2]), S0);
                                vkq3 = hvx_vec_mpyacc_f32_f16(vkq3, Q6_Vh_vshuff_Vh(vx0[3]), S0);
                                थांबा;
                            }

                            HVX_Vector S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            स्थिरांक HVX_Vector * vx1 = (स्थिरांक HVX_Vector *) (v_ptr + factx->size_v_row_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx1[1]), S1);
                            vkq2 = hvx_vec_mpyacc_f32_f16(vkq2, Q6_Vh_vshuff_Vh(vx0[2]), S0);
                            vkq2 = hvx_vec_mpyacc_f32_f16(vkq2, Q6_Vh_vshuff_Vh(vx1[2]), S1);
                            vkq3 = hvx_vec_mpyacc_f32_f16(vkq3, Q6_Vh_vshuff_Vh(vx0[3]), S0);
                            vkq3 = hvx_vec_mpyacc_f32_f16(vkq3, Q6_Vh_vshuff_Vh(vx1[3]), S1);
                            v_ptr += stride_v2;
                        }

                        ((HVX_VectorPair *) VKQ32)[0] = vkq0;
                        ((HVX_VectorPair *) VKQ32)[1] = vkq1;
                        ((HVX_VectorPair *) VKQ32)[2] = vkq2;
                        ((HVX_VectorPair *) VKQ32)[3] = vkq3;
                    } अन्यथा {
                        hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (स्थिरांक uint8_t *) VKQ32, DV, ms_vec);

                        साठी (uint32_t j = 0; j < current_block_size; j += 2) {
                            जर (j + 1 == current_block_size) {
                                HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                                hvx_mad_f32_f16_aa_vec(VKQ32, v_ptr, S0, DV);
                                थांबा;
                            }

                            HVX_Vector S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            HVX_Vector S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));

                            hvx_mad_f32_f16_aa_rx2_vec(VKQ32, v_ptr, v_ptr + factx->size_v_row_padded, S0, S1, DV);
                            v_ptr += stride_v2;
                        }
                    }
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, head_ir);

                // prefetch V and mask for step + 2 after last head finished V accumulation
                जर (g + 1 == G_local && step + 2 < n_active_blocks) {
                    स्थिरांक uint32_t next_ib = active_blocks[step + 2];
                    स्थिरांक uint32_t next_ic_start = next_ib * FLASH_ATTN_BLOCK_SIZE;
                    स्थिरांक uint32_t next_block_size = MIN(FLASH_ATTN_BLOCK_SIZE, nek1 - next_ic_start);

                    // V
                    स्थिरांक dma_addr_t v_src = v->data + next_ic_start*nbv1 + iv2*nbv2 + iv3*nbv3;
                    dma_queue_ढकला(dma_q, dma_make_data(v_base, v_src), factx->size_v_row_padded, nbv1, size_v_row, next_block_size);

                    // Mask
                    जर (mask) {
                        स्थिरांक dma_addr_t m_src = mp_base + next_ic_start * आकार(__fp16);
                        dma_cache_dm_ढकला(dma_q, &m_cache, next_ib, m_src, next_block_size * 2, next_block_size * 2, next_block_size * 2, 1);
                    }
                }
            } // end for g
        } // end for step

        साठी (uint32_t g = 0; g < G_local; ++g) {
            स्थिरांक uint32_t head_ir = ir + g;
            स्थिरांक uint32_t h = heads[g];
            दशांश * VKQ32 = vkq_ptrs[g];

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, head_ir);

            दशांश M = hvx_vec_get_f32(M_vec[g]);
            दशांश S = hvx_vec_get_f32(S_vec[g]);

            जर (sinks) {
                स्थिरांक दशांश s = factx->spad_sinks[h];

                दशांश vs = 1.0f;

                जर (s > M) {
                    HVX_Vector diff_vec = hvx_vec_splat_f32(M - s);
                    HVX_Vector ms_vec   = hvx_vec_exp_f32(diff_vec);
                    hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (स्थिरांक uint8_t *) VKQ32, DV, ms_vec);

                    दशांश ms = hvx_vec_get_f32(ms_vec);
                    S = S * ms + vs;
                } अन्यथा {
                    HVX_Vector diff_vec = hvx_vec_splat_f32(s - M);
                    vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                    S += vs;
                }
            }

            स्थिरांक दशांश S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            hvx_scale_f32_aa((uint8_t *) VKQ32, (स्थिरांक uint8_t *) VKQ32, DV, S_inv);

            स्थिरांक uint32_t r_iq3 = fastdiv(head_ir, &factx->src0_div21);
            स्थिरांक uint32_t r_iq2 = fastdiv(head_ir - r_iq3*neq2*neq1, &factx->src0_div1);
            स्थिरांक uint32_t r_iq1 = (head_ir - r_iq3*neq2*neq1 - r_iq2 * neq1);

            uint8_t * dst_ptr = (uint8_t *) dst->data + r_iq2 * dst->nb[1] + r_iq1 * dst->nb[2] + r_iq3 * dst->nb[3];

            जर (dst->type == HTP_TYPE_F32) {
                hvx_copy_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
            } अन्यथा जर (dst->type == HTP_TYPE_F16) {
                hvx_copy_f16_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, head_ir);
        }

        ir += G_local;
    }
}

// ============================================================================
// HMX Phase args and thread logic
// ============================================================================

प्रकारव्याख्या संरचना {
    संरचना hmx_fa_context * factx;
    uint32_t                kv_rows;
    माप_प्रकार                  src_stride;
    शून्य *                  curr_k;
    uint32_t                kv_start;
    uint32_t                rows_per_t;
    माप_प्रकार                  buf_idx;
} fa_k_int_args_t;

स्थिर शून्य fa_k_interleave_thread(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, शून्य * data) {
    fa_k_int_args_t *       args  = (fa_k_int_args_t *) data;
    संरचना hmx_fa_context * factx = args->factx;

    स्थिरांक uint32_t total_rows = args->kv_rows;
    स्थिरांक uint32_t rows_per_t = args->rows_per_t;
    स्थिरांक uint32_t start      = i * rows_per_t;
    स्थिरांक uint32_t end        = (uint32_t) hex_smin(start + rows_per_t, total_rows);

    जर (start >= total_rows) {
        परत;
    }

    संरचना htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (args->kv_start + start));
    जर (factx->k->type == HTP_TYPE_Q8_0) {
        साठी (uint32_t r = start; r < end; ++r) {
            __fp16 * row_k = (__fp16 *)((अक्षर *)args->curr_k + r * args->src_stride * आकार(__fp16));
            hvx_dequantize_row_flat_q8_0_f16(row_k, row_k, factx->DK);
        }
    }
    hmx_interleave_rows_to_tiles(factx->vtcm_k_tiles[args->buf_idx], (स्थिरांक __fp16 *) args->curr_k, total_rows, factx->DK_pad,
                             args->src_stride, start, end);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (args->kv_start + start));
}

स्थिर शून्य fa_phase_k_interleave(संरचना hmx_fa_context * factx, uint32_t kv_rows, माप_प्रकार src_stride, शून्य * curr_k, uint32_t kv_start, माप_प्रकार buf_idx) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    जर (factx->n_threads > 1 && kv_rows >= factx->n_threads * 2) {
        n = factx->n_threads;
    }
    uint32_t rows_per_t = hex_align_up(hmx_ceil_div(kv_rows, n), 2);
    fa_k_int_args_t args = { factx, kv_rows, src_stride, curr_k, kv_start, rows_per_t, buf_idx };
    जर (n > 1) {
        work_queue_run(wp, fa_k_interleave_thread, &args, n);
    } अन्यथा {
        fa_k_interleave_thread(1, 0, &args);
    }
}

प्रकारव्याख्या संरचना {
    संरचना hmx_fa_context * factx;
    uint32_t                kv_rows;
    माप_प्रकार                  src_stride;
    शून्य *                  v_src;
    शून्य *                  v_tiles_dst;
    माप_प्रकार                  n_col_tiles;
    uint32_t                kv_start;
    uint32_t                rows_per_t;
} fa_v_int_args_t;

स्थिर शून्य fa_v_interleave_thread(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, शून्य * data) {
    fa_v_int_args_t *       args  = (fa_v_int_args_t *) data;
    संरचना hmx_fa_context * factx = args->factx;

    स्थिरांक uint32_t total_rows = args->kv_rows;
    स्थिरांक uint32_t rows_per_t = args->rows_per_t;
    स्थिरांक uint32_t start      = i * rows_per_t;
    स्थिरांक uint32_t end        = (uint32_t) hex_smin(start + rows_per_t, total_rows);

    जर (start >= total_rows) {
        परत;
    }

    __fp16 * v_tiles_dst = (__fp16 *) args->v_tiles_dst;

    संरचना htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (args->kv_start + start));
    जर (factx->v->type == HTP_TYPE_Q8_0) {
        साठी (uint32_t r = start; r < end; ++r) {
            __fp16 * row_v = (__fp16 *)((अक्षर *)args->v_src + r * args->src_stride * आकार(__fp16));
            hvx_dequantize_row_flat_q8_0_f16(row_v, row_v, factx->DV);
        }
    }
    hmx_interleave_cols_to_tiles(v_tiles_dst, (स्थिरांक __fp16 *) args->v_src, total_rows, factx->DV_pad,
                             args->src_stride, (uint32_t) args->n_col_tiles, start, end);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (args->kv_start + start));
}

स्थिर शून्य fa_phase_v_interleave(संरचना hmx_fa_context * factx,
                                  uint32_t                kv_rows,
                                  माप_प्रकार                  src_stride,
                                  शून्य *                  v_src,
                                  शून्य *                  v_tiles_dst,
                                  माप_प्रकार                  n_col_tiles,
                                  uint32_t                kv_start) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    जर (factx->n_threads > 1 && kv_rows >= factx->n_threads * 2) {
        n = factx->n_threads;
    }
    uint32_t rows_per_t = hex_align_up(hmx_ceil_div(kv_rows, n), 2);
    fa_v_int_args_t args = { factx, kv_rows, src_stride, v_src, v_tiles_dst, n_col_tiles, kv_start, rows_per_t };
    जर (n > 1) {
        work_queue_run(wp, fa_v_interleave_thread, &args, n);
    } अन्यथा {
        fa_v_interleave_thread(1, 0, &args);
    }
}

प्रकारव्याख्या संरचना {
    संरचना hmx_fa_context *   factx;
    स्थिरांक संरचना htp_tensor * q;
    uint32_t                  q_start;
    uint32_t                  kv_head;
    uint32_t                  ib3;
    माप_प्रकार                    n_rows_g;
    माप_प्रकार                    rows_per_t;
    माप_प्रकार                    n_rows_q;
    bool                      q_transposed;
    atomic_uint               barrier;
} fa_q_load_args_t;

स्थिर शून्य fa_q_load_thread(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, शून्य * data) {
    fa_q_load_args_t *      args  = (fa_q_load_args_t *) data;
    संरचना hmx_fa_context * factx = args->factx;

    स्थिरांक माप_प्रकार n_rows_g = args->n_rows_g;
    स्थिरांक माप_प्रकार G        = factx->G;
    स्थिरांक माप_प्रकार DK       = factx->DK;

    // Partition the padded Q rows (g_br) across threads.
    // Keep start/end even so r and r+1 are always in the same thread's range.
    स्थिरांक माप_प्रकार rows_per_t = args->rows_per_t;
    स्थिरांक माप_प्रकार start      = (माप_प्रकार) i * rows_per_t;
    स्थिरांक माप_प्रकार end        = hex_smin(start + rows_per_t, factx->g_br);

    संरचना htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (args->q_start * G + start));

    // Parallel initialization of per-block state
    {
        स्थिरांक uint32_t g_br = factx->g_br;
        स्थिरांक uint32_t DV   = factx->DV;

        स्थिरांक माप_प्रकार col_vec_bytes = factx->col_vec_bytes;
        स्थिरांक माप_प्रकार d_tile_bytes  = factx->d_tile_bytes;

        // Initialize vtcm_l_vec & vtcm_m_vec
        स्थिरांक माप_प्रकार l_bytes_per_t = hex_align_up(col_vec_bytes / n, 128);
        स्थिरांक माप_प्रकार l_start       = i * l_bytes_per_t;
        स्थिरांक माप_प्रकार l_end         = hex_smin(l_start + l_bytes_per_t, col_vec_bytes);

        स्थिरांक माप_प्रकार m_bytes_per_t = hex_align_up(col_vec_bytes / n, 128);
        स्थिरांक माप_प्रकार m_start       = i * m_bytes_per_t;
        स्थिरांक माप_प्रकार m_end         = hex_smin(m_start + m_bytes_per_t, col_vec_bytes);

        जर (l_start < col_vec_bytes) {
            hvx_splat_u8_a((अक्षर *) factx->vtcm_l_vec + l_start, 0, l_end - l_start);
        }
        जर (m_start < col_vec_bytes) {
            hvx_splat_f32_a((अक्षर *) factx->vtcm_m_vec + m_start, HTP_FA_M_INITIAL_VAL, (m_end - m_start) / आकार(दशांश));
        }

        // Zero the whole rescale region: vtcm_d_tiles[0], the optional vtcm_d_tiles[1]
        // and vtcm_d_inv_l are equal-sized and allocated back to back, so one run covers
        // them all.  The scatter only ever writes the diagonal, ignore the rest.
        स्थिरांक माप_प्रकार d_bytes_per_t = hex_align_up(d_tile_bytes / n, 128);
        स्थिरांक माप_प्रकार d_start       = i * d_bytes_per_t;
        स्थिरांक माप_प्रकार d_end         = hex_smin(d_start + d_bytes_per_t, d_tile_bytes);
        जर (d_start < d_tile_bytes) {
            hvx_splat_u8_a((अक्षर *) factx->vtcm_d_tiles[0] + d_start, 0, d_end - d_start);
        }
    }

    जर (start < factx->g_br) {
        स्थिरांक संरचना htp_tensor * q       = args->q;
        स्थिरांक uint32_t            q_start = args->q_start;
        स्थिरांक uint32_t            kv_head = args->kv_head;
        स्थिरांक uint32_t            ib3     = args->ib3;

        स्थिरांक bool use_q_dma = (factx->vtcm_q_dma != रिक्त);

        __fp16 * q_tiles = factx->vtcm_q_tiles;
        स्थिरांक माप_प्रकार DK_pad = factx->DK_pad;
        जर (use_q_dma) {
            स्थिरांक माप_प्रकार g_rows_end = hex_smin(end, n_rows_g);
            स्थिरांक uint32_t d_limit = factx->is_q_fp32 ? DK / 32 : DK / 64;

            uint8_t * q_flat  = (uint8_t *) factx->vtcm_q_dma;
            जर (DK_pad != DK) {
                जर (factx->is_q_fp32) {
                    hmx_fa_q_prep_fp32_pad(q_tiles, q_flat, start, end, g_rows_end, DK, DK_pad, G, args->n_rows_q, &factx->div_G, args->q_transposed);
                } अन्यथा {
                    hmx_fa_q_prep_fp16_pad(q_tiles, q_flat, start, end, g_rows_end, DK, DK_pad, G, args->n_rows_q, &factx->div_G, args->q_transposed);
                }
            } अन्यथा जर (factx->is_q_fp32) {
                निवड (d_limit) {
                स्थिती 2:  hmx_fa_q_prep_fp32_d2(q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, args->q_transposed); थांबा;
                स्थिती 4:  hmx_fa_q_prep_fp32_d4(q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, args->q_transposed); थांबा;
                पूर्वनिर्धारित: hmx_fa_q_prep_fp32(   q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, d_limit, args->q_transposed); थांबा;
                }
            } अन्यथा {
                निवड (d_limit) {
                स्थिती 1:  hmx_fa_q_prep_fp16_d1(q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, args->q_transposed); थांबा;
                स्थिती 2:  hmx_fa_q_prep_fp16_d2(q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, args->q_transposed); थांबा;
                पूर्वनिर्धारित: hmx_fa_q_prep_fp16(   q_tiles, q_flat, start, end, g_rows_end, DK, G, args->n_rows_q, &factx->div_G, d_limit, args->q_transposed); थांबा;
                }
            }
        } अन्यथा {
            // Fallback: direct-from-DDR/L2 path
            hmx_fa_q_prep_fallback(q_tiles, q->data, q->nb[1], q->nb[2], q->nb[3],
                                   q_start, kv_head, ib3, start, end, n_rows_g, G, DK, DK_pad, factx->is_q_fp32, &factx->div_G);
        }
    }

    // Synchronize threads before zeroing out vtcm_o_tiles[0] to prevent race condition
    जर (n > 1) {
        atomic_fetch_sub(&args->barrier, 1);
        पर्यंत (atomic_load(&args->barrier) > 0) {
            // spin wait
        }
    }

    // Zero out vtcm_o_tiles[0] as it was used as temp_q_vtcm
    {
        स्थिरांक uint32_t g_br = factx->g_br;
        स्थिरांक uint32_t DV   = factx->DV;
        स्थिरांक माप_प्रकार o_tile_bytes  = factx->o_tile_bytes;
        स्थिरांक माप_प्रकार o_bytes_per_t = hex_align_up(o_tile_bytes / n, 128);
        स्थिरांक माप_प्रकार o_start       = i * o_bytes_per_t;
        स्थिरांक माप_प्रकार o_end         = hex_smin(o_start + o_bytes_per_t, o_tile_bytes);
        जर (o_start < o_tile_bytes) {
            hvx_splat_u8_a((अक्षर *) factx->vtcm_o_tiles[0] + o_start, 0, o_end - o_start);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (args->q_start * G + start));
}

स्थिर शून्य fa_phase_q_load(संरचना hmx_fa_context *   factx,
                            स्थिरांक संरचना htp_tensor * q,
                            uint32_t                  q_start,
                            uint32_t                  kv_head,
                            uint32_t                  ib3,
                            माप_प्रकार                    n_rows_g) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    जर (factx->n_threads > 1 && n_rows_g >= (माप_प्रकार) (factx->n_threads * 2)) {
        n = factx->n_threads;
    }
    माप_प्रकार rows_per_t = hex_align_up(hmx_ceil_div(factx->g_br, n), 2);
    स्थिरांक uint32_t n_rows_q = hex_smin(factx->Br, factx->neq1 - q_start);
    fa_q_load_args_t args;
    args.factx = factx;
    args.q = q;
    args.q_start = q_start;
    args.kv_head = kv_head;
    args.ib3 = ib3;
    args.n_rows_g = n_rows_g;
    args.rows_per_t = rows_per_t;
    args.n_rows_q = n_rows_q;
    args.q_transposed = q->nb[1] < q->nb[2];
    atomic_init(&args.barrier, n);
    जर (n > 1) {
        work_queue_run(wp, fa_q_load_thread, &args, n);
    } अन्यथा {
        fa_q_load_thread(1, 0, &args);
    }
}

प्रकारव्याख्या संरचना {
    संरचना hmx_fa_context *   factx;
    स्थिरांक संरचना htp_tensor * dst;
    स्थिरांक __fp16 *            o_tile_src;
    uint32_t                  q_start;
    uint32_t                  kv_head;
    uint32_t                  ib3;
    माप_प्रकार                    n_rows_g;
    माप_प्रकार                    rows_per_t;
} fa_o_store_args_t;

स्थिर शून्य fa_o_store_thread_f32(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, शून्य * data) {
    fa_o_store_args_t *     args  = (fa_o_store_args_t *) data;
    संरचना hmx_fa_context * factx = args->factx;

    स्थिरांक माप_प्रकार n_rows_g = args->n_rows_g;
    स्थिरांक माप_प्रकार G        = factx->G;
    स्थिरांक माप_प्रकार DV       = factx->DV;

    स्थिरांक माप_प्रकार rows_per_t = args->rows_per_t;
    स्थिरांक माप_प्रकार start      = (माप_प्रकार) i * rows_per_t;
    स्थिरांक माप_प्रकार end        = hex_smin(start + rows_per_t, n_rows_g);

    जर (start >= n_rows_g) {
        परत;
    }

    संरचना htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));

    स्थिरांक संरचना htp_tensor * dst        = args->dst;
    स्थिरांक __fp16 *            o_tile_src = args->o_tile_src;
    स्थिरांक uint32_t            q_start    = args->q_start;
    स्थिरांक uint32_t            kv_head    = args->kv_head;
    स्थिरांक uint32_t            ib3        = args->ib3;

    स्थिरांक माप_प्रकार DV_pad = factx->DV_pad;

    माप_प्रकार q_idx = fastdiv(start, &factx->div_G);
    माप_प्रकार h_idx = fastmodulo(start, G, &factx->div_G);

    साठी (माप_प्रकार r = start; r < end; ++r) {
        दशांश * out = (दशांश *) ((uint8_t *) dst->data + (kv_head * G + h_idx) * dst->nb[1] +
                                 (q_start + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        माप_प्रकार         r0            = r / HMX_FP16_TILE_N_ROWS;
        माप_प्रकार         r1            = r % HMX_FP16_TILE_N_ROWS;
        स्थिरांक __fp16 * tile_row_base = o_tile_src + r0 * HMX_FP16_TILE_N_ROWS * DV_pad;

        साठी (uint32_t d = 0; d < DV / 32; ++d) {
            स्थिरांक HVX_Vector * in_tile = (स्थिरांक HVX_Vector *) (tile_row_base + d * HMX_FP16_TILE_N_ELMS);
            HVX_VectorPair     vp      = hvx_vec_f16_to_f32_shuff(in_tile[r1 / 2]);
            जर (r1 % 2 == 0) {
                *(HVX_UVector *) (out + d * 32) = Q6_V_lo_W(vp);
            } अन्यथा {
                *(HVX_UVector *) (out + d * 32) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail: DV not a multiple of 32 (e.g. 72 -> last 8 lanes). Partial vector-write
        // for the remaining (DV % 32) floats.
        स्थिरांक uint32_t d_tail = DV / 32;
        स्थिरांक uint32_t rem    = DV - d_tail * 32;
        जर (rem) {
            स्थिरांक HVX_Vector * in_tile = (स्थिरांक HVX_Vector *) (tile_row_base + d_tail * HMX_FP16_TILE_N_ELMS);
            HVX_VectorPair     vp      = hvx_vec_f16_to_f32_shuff(in_tile[r1 / 2]);
            HVX_Vector         vd      = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((शून्य *) (out + d_tail * 32), rem * आकार(दशांश), vd);
        }

        h_idx++;
        जर (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));
}

स्थिर शून्य fa_o_store_thread_f16(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, शून्य * data) {
    fa_o_store_args_t *     args  = (fa_o_store_args_t *) data;
    संरचना hmx_fa_context * factx = args->factx;

    स्थिरांक माप_प्रकार n_rows_g   = args->n_rows_g;
    स्थिरांक माप_प्रकार rows_per_t = args->rows_per_t;
    स्थिरांक माप_प्रकार G          = factx->G;
    स्थिरांक माप_प्रकार DV         = factx->DV;
    स्थिरांक माप_प्रकार start      = (माप_प्रकार) i * rows_per_t;
    स्थिरांक माप_प्रकार end        = hex_smin(start + rows_per_t, n_rows_g);

    जर (start >= n_rows_g) {
        परत;
    }

    संरचना htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));

    स्थिरांक संरचना htp_tensor * dst        = args->dst;
    स्थिरांक __fp16 *            o_tile_src = args->o_tile_src;
    स्थिरांक uint32_t            q_start    = args->q_start;
    स्थिरांक uint32_t            kv_head    = args->kv_head;
    स्थिरांक uint32_t            ib3        = args->ib3;

    // O-tiles use the padded head dim (DV_pad); dst holds the real DV lanes.
    स्थिरांक माप_प्रकार DV_pad = factx->DV_pad;

    माप_प्रकार q_idx = fastdiv(start, &factx->div_G);
    माप_प्रकार h_idx = fastmodulo(start, G, &factx->div_G);

    साठी (माप_प्रकार r = start; r < end; ++r) {
        __fp16 * out = (__fp16 *) ((uint8_t *) dst->data + (kv_head * G + h_idx) * dst->nb[1] +
                                   (q_start + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        माप_प्रकार         r0            = r / HMX_FP16_TILE_N_ROWS;
        माप_प्रकार         r1            = r % HMX_FP16_TILE_N_ROWS;
        स्थिरांक __fp16 * tile_row_base = o_tile_src + r0 * HMX_FP16_TILE_N_ROWS * DV_pad;

        साठी (uint32_t d = 0; d < DV / 64; ++d) {
            स्थिरांक __fp16 *     in_dtile = tile_row_base + d * HMX_FP16_TILE_N_ELMS * 2;
            स्थिरांक HVX_Vector * pv_in0   = ((स्थिरांक HVX_Vector *) in_dtile) + r1 / 2;
            स्थिरांक HVX_Vector * pv_in1   = pv_in0 + 16;
            HVX_VectorPair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            जर (r1 % 2 == 0) {
                *(HVX_UVector *) (out + d * 64) = Q6_V_lo_W(vp);
            } अन्यथा {
                *(HVX_UVector *) (out + d * 64) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail when DV is not a multiple of 64.
        स्थिरांक uint32_t d_tail = DV / 64;
        स्थिरांक uint32_t rem    = DV - d_tail * 64;
        जर (rem) {
            स्थिरांक __fp16 *     in_dtile = tile_row_base + d_tail * HMX_FP16_TILE_N_ELMS * 2;
            स्थिरांक HVX_Vector * pv_in0   = ((स्थिरांक HVX_Vector *) in_dtile) + r1 / 2;
            स्थिरांक HVX_Vector * pv_in1   = pv_in0 + 16;
            HVX_VectorPair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            HVX_Vector         vd       = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((शून्य *) (out + d_tail * 64), rem * आकार(__fp16), vd);
        }

        h_idx++;
        जर (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (args->q_start * G + start));
}

स्थिर शून्य fa_phase_o_store(संरचना hmx_fa_context *   factx,
                             स्थिरांक संरचना htp_tensor * dst,
                             स्थिरांक __fp16 *            o_tile_src,
                             uint32_t                  q_start,
                             uint32_t                  kv_head,
                             uint32_t                  ib3,
                             माप_प्रकार                    n_rows_g) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    uint32_t n = 1;
    जर (factx->n_threads > 1 && n_rows_g >= (माप_प्रकार) (factx->n_threads * 2)) {
        n = factx->n_threads;
    }
    माप_प्रकार rows_per_t = hmx_ceil_div(n_rows_g, n);
    fa_o_store_args_t args = { factx, dst, o_tile_src, q_start, kv_head, ib3, n_rows_g, rows_per_t };
    worker_callback_t store_fn = factx->is_dst_fp32 ? fa_o_store_thread_f32 : fa_o_store_thread_f16;
    जर (n > 1) {
        work_queue_run(wp, store_fn, &args, n);
    } अन्यथा {
        store_fn(1, 0, &args);
    }
}

प्रकारव्याख्या संरचना {
    संरचना hmx_fa_context *   factx;
    माप_प्रकार                    buf_idx;
    माप_प्रकार                    kv_rows;
    माप_प्रकार                    n_rows_g;
    माप_प्रकार                    n_col_tiles;
    माप_प्रकार                    n_tiles_per_bc;
    माप_प्रकार                    n_row_tiles;
    माप_प्रकार                    n_row_tiles_g_br;
    uint32_t                  Bc;
    uint32_t                  G;
    uint32_t                  kv_head;
    uint32_t                  kv_start;
    uint32_t                  q_start;
    uint32_t                  ib3;
    bool                      has_alibi;  // true when max_bias != 0 (need slope * mask + add)
    __fp16 *                  slopes;
    स्थिरांक संरचना htp_tensor * mask;
    स्थिरांक __fp16 *            mask_vtcm;             // VTCM mask buffer base (NULL = DDR fallback)
    माप_प्रकार                    mask_vtcm_row_stride;  // elements (__fp16) per row in VTCM mask buffer
    संरचना fastdiv_मूल्ये     thread_div;
} fa_softmax_args_t;

स्थिर ओळीत शून्य fa_softmax_impl(
    अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, शून्य * data,
    स्थिरांक bool has_mask,
    स्थिरांक bool mask_broadcast,
    स्थिरांक bool is_g1,
    स्थिरांक bool has_alibi,
    स्थिरांक bool has_softcap
) {
    fa_softmax_args_t *     args  = (fa_softmax_args_t *) data;
    संरचना hmx_fa_context * factx = args->factx;

    स्थिरांक माप_प्रकार n_rows_g       = args->n_rows_g;
    स्थिरांक माप_प्रकार kv_rows        = args->kv_rows;
    स्थिरांक माप_प्रकार Bc             = args->Bc;
    स्थिरांक माप_प्रकार G              = args->G;
    स्थिरांक माप_प्रकार n_tiles_per_bc = args->n_tiles_per_bc;
    स्थिरांक माप_प्रकार n_row_vec_cnt  = hmx_ceil_div(n_rows_g, 64);
    स्थिरांक uint32_t im3          = has_mask ? fastmodulo(args->ib3, args->mask->ne[3], &factx->src3_div3) : 0;

    माप_प्रकार vec_start = 0;
    माप_प्रकार vec_end   = n_row_vec_cnt;
    जर (n > 1) {
        स्थिरांक माप_प्रकार vecs_per_t = fastdiv(n_row_vec_cnt + n - 1, &args->thread_div);
        vec_start = i * vecs_per_t;
        vec_end   = hex_smin(vec_start + vecs_per_t, n_row_vec_cnt);
    }

    जर (vec_start >= n_row_vec_cnt) {
        परत;
    }

    संरचना htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (args->q_start * G + vec_start * 64));

    // Per-thread row scratch: thread i uses bufs at offset i * 2 * stride
    स्थिरांक माप_प्रकार row_buf_stride = factx->row_buf_stride;
    HVX_Vector * my_row_buf0    = factx->vtcm_row_bufs + i * 2 * row_buf_stride;
    HVX_Vector * my_row_buf1    = my_row_buf0 + row_buf_stride;

    स्थिरांक HVX_Vector v_neg_inf = Q6_Vh_vsplat_R(0xfbff);

    साठी (माप_प्रकार r_vec_idx = vec_start; r_vec_idx < vec_end; ++r_vec_idx) {
        HVX_Vector rowmax_acc_v = v_neg_inf;
        HVX_Vector rowsum_acc_v = Q6_V_vzero();
        HVX_Vector m_prev_v0    = factx->vtcm_m_vec[r_vec_idx * 2 + 0];
        HVX_Vector m_prev_v1    = factx->vtcm_m_vec[r_vec_idx * 2 + 1];

        HVX_Vector v_slopes = Q6_V_vzero();
        जर (has_alibi) {
            v_slopes = hvx_vmem(args->slopes + r_vec_idx * 64);
        }

        साठी (uint32_t r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            uint32_t r = r_vec_idx * 64 + r_vec_off;
            जर (r >= hex_align_up(n_rows_g, 2)) {
                थांबा;
            }

            uint32_t r0 = r / HMX_FP16_TILE_N_ROWS;
            uint32_t r1 = r % HMX_FP16_TILE_N_ROWS;

            स्थिरांक __fp16 * s_ld_base = factx->vtcm_s_tiles[args->buf_idx] + r0 * HMX_FP16_TILE_N_ROWS * Bc;
            __fp16 *       p_st_base = factx->vtcm_p_tiles[args->buf_idx] + r0 * HMX_FP16_TILE_N_ROWS * Bc;

            // Decode 2 rows from S tiles into per-thread row buffers
            जर (has_softcap) {
                स्थिरांक HVX_Vector v_cap = hvx_vec_splat_f16(factx->logit_softcap);
                साठी (माप_प्रकार c = 0; c < kv_rows; c += 64) {
                    माप_प्रकार             ci       = c / 64;
                    स्थिरांक __fp16 *     in_dtile = s_ld_base + ci * HMX_FP16_TILE_N_ELMS * 2;
                    स्थिरांक HVX_Vector * pv_s_in0 = ((स्थिरांक HVX_Vector *) in_dtile) + r1 / 2;
                    स्थिरांक HVX_Vector * pv_s_in1 = pv_s_in0 + 16;

                    HVX_VectorPair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    HVX_Vector     v_s_row0  = Q6_V_lo_W(vp_s_drow);
                    HVX_Vector     v_s_row1  = Q6_V_hi_W(vp_s_drow);

                    HVX_Vector t0   = hvx_vec_tanh_f16(v_s_row0);
                    my_row_buf0[ci] = hvx_vec_mul_f16_f16(t0, v_cap);

                    HVX_Vector t1   = hvx_vec_tanh_f16(v_s_row1);
                    my_row_buf1[ci] = hvx_vec_mul_f16_f16(t1, v_cap);
                }
            } अन्यथा {
                माप_प्रकार c = 0;
                साठी (; c + 64 < kv_rows; c += 128) {
                    माप_प्रकार             ci0       = c / 64;
                    माप_प्रकार             ci1       = ci0 + 1;
                    स्थिरांक __fp16 *     in_dtile0 = s_ld_base + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                    स्थिरांक __fp16 *     in_dtile1 = s_ld_base + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                    स्थिरांक HVX_Vector * pv_s_in0_0 = ((स्थिरांक HVX_Vector *) in_dtile0) + r1 / 2;
                    स्थिरांक HVX_Vector * pv_s_in1_0 = pv_s_in0_0 + 16;
                    स्थिरांक HVX_Vector * pv_s_in0_1 = ((स्थिरांक HVX_Vector *) in_dtile1) + r1 / 2;
                    स्थिरांक HVX_Vector * pv_s_in1_1 = pv_s_in0_1 + 16;

                    HVX_VectorPair vp_s_drow0 = Q6_W_vdeal_VVR(*pv_s_in1_0, *pv_s_in0_0, -2);
                    my_row_buf0[ci0]          = Q6_V_lo_W(vp_s_drow0);
                    my_row_buf1[ci0]          = Q6_V_hi_W(vp_s_drow0);

                    HVX_VectorPair vp_s_drow1 = Q6_W_vdeal_VVR(*pv_s_in1_1, *pv_s_in0_1, -2);
                    my_row_buf0[ci1]          = Q6_V_lo_W(vp_s_drow1);
                    my_row_buf1[ci1]          = Q6_V_hi_W(vp_s_drow1);
                }
                साठी (; c < kv_rows; c += 64) {
                    माप_प्रकार             ci       = c / 64;
                    स्थिरांक __fp16 *     in_dtile = s_ld_base + ci * HMX_FP16_TILE_N_ELMS * 2;
                    स्थिरांक HVX_Vector * pv_s_in0 = ((स्थिरांक HVX_Vector *) in_dtile) + r1 / 2;
                    स्थिरांक HVX_Vector * pv_s_in1 = pv_s_in0 + 16;

                    HVX_VectorPair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    my_row_buf0[ci]          = Q6_V_lo_W(vp_s_drow);
                    my_row_buf1[ci]          = Q6_V_hi_W(vp_s_drow);
                }
            }

            // Apply mask & compute rowmax(S)
            HVX_Vector v_slope0 = Q6_V_vzero();
            HVX_Vector v_slope1 = Q6_V_vzero();
            जर (has_alibi) {
                v_slope0 = hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, r_vec_off * 2));
                v_slope1 = (r + 1 < n_rows_g) ? hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, (r_vec_off + 1) * 2)) : Q6_V_vzero();
            }

            स्थिरांक HVX_Vector v_threshold = Q6_Vh_vsplat_R(0xcc00);  // fp16 -16.0

            HVX_Vector v_s_rowmax0 = v_neg_inf;
            HVX_Vector v_s_rowmax1 = v_neg_inf;
            जर (has_mask) {
                साठी (माप_प्रकार c = 0; c < kv_rows; c += 64) {
                    माप_प्रकार         ci          = c / 64;
                    स्थिरांक माप_प्रकार   ne          = hex_smin(kv_rows - c, 64);
                    HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(ne * आकार(__fp16));

                    HVX_Vector v_mask0, v_mask1;

                    जर (mask_broadcast) {
                        जर (is_g1) {
                            स्थिरांक माप_प्रकार qi0 = r + 0;
                            v_mask0 = *(स्थिरांक HVX_Vector *) (args->mask_vtcm + qi0 * args->mask_vtcm_row_stride + c);
                            v_mask1 = v_neg_inf;
                            जर (r + 1 < n_rows_g) {
                                स्थिरांक माप_प्रकार qi1 = r + 1;
                                v_mask1 = *(स्थिरांक HVX_Vector *) (args->mask_vtcm + qi1 * args->mask_vtcm_row_stride + c);
                            }
                        } अन्यथा {
                            स्थिरांक माप_प्रकार qi0 = fastdiv(r + 0, &factx->div_G);
                            v_mask0 = *(स्थिरांक HVX_Vector *) (args->mask_vtcm + qi0 * args->mask_vtcm_row_stride + c);
                            v_mask1 = v_neg_inf;
                            जर (r + 1 < n_rows_g) {
                                स्थिरांक माप_प्रकार qi1 = fastdiv(r + 1, &factx->div_G);
                                जर (qi1 == qi0) {
                                    v_mask1 = v_mask0;
                                } अन्यथा {
                                    v_mask1 = *(स्थिरांक HVX_Vector *) (args->mask_vtcm + qi1 * args->mask_vtcm_row_stride + c);
                                }
                            }
                        }
                    } अन्यथा {
                        // Head-dependent mask: pre-interleaved per row r.
                        स्थिरांक माप_प्रकार r0 = r + 0;
                        v_mask0 = *(स्थिरांक HVX_Vector *) (args->mask_vtcm + r0 * args->mask_vtcm_row_stride + c);
                        v_mask1 = v_neg_inf;
                        जर (r + 1 < n_rows_g) {
                            स्थिरांक माप_प्रकार r1 = r + 1;
                            v_mask1 = *(स्थिरांक HVX_Vector *) (args->mask_vtcm + r1 * args->mask_vtcm_row_stride + c);
                        }
                    }

                    // Threshold: mask values below -16.0 are treated as -inf (causal mask).
                    HVX_VectorPred q_keep0 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_threshold), q_tail_keep);
                    HVX_VectorPred q_keep1 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_threshold), q_tail_keep);

                    // Scale mask values by log2(e) for base-2 calculations
                    स्थिरांक HVX_Vector v_log2e = hvx_vec_splat_f16(EXP_LOG2E_F);
                    HVX_Vector v_mask0_scaled = hvx_vec_mul_f16_f16(v_mask0, v_log2e);
                    HVX_Vector v_mask1_scaled = hvx_vec_mul_f16_f16(v_mask1, v_log2e);

                    जर (has_alibi) {
                        HVX_Vector v_sm0 = hvx_vec_mul_f16_f16(v_mask0_scaled, v_slope0);
                        HVX_Vector v_sm1 = hvx_vec_mul_f16_f16(v_mask1_scaled, v_slope1);
                        my_row_buf0[ci]  = Q6_V_vmux_QVV(q_keep0, hvx_vec_add_f16_f16(my_row_buf0[ci], v_sm0), v_neg_inf);
                        my_row_buf1[ci]  = Q6_V_vmux_QVV(q_keep1, hvx_vec_add_f16_f16(my_row_buf1[ci], v_sm1), v_neg_inf);
                    } अन्यथा {
                        my_row_buf0[ci] = Q6_V_vmux_QVV(q_keep0, hvx_vec_add_f16_f16(my_row_buf0[ci], v_mask0_scaled), v_neg_inf);
                        my_row_buf1[ci] = Q6_V_vmux_QVV(q_keep1, hvx_vec_add_f16_f16(my_row_buf1[ci], v_mask1_scaled), v_neg_inf);
                    }

                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci]);
                }
            } अन्यथा {
                माप_प्रकार c = 0;
                साठी (; c + 64 < kv_rows; c += 128) {
                    माप_प्रकार ci0 = c / 64;
                    माप_प्रकार ci1 = ci0 + 1;
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci0]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci0]);
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci1]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci1]);
                }
                साठी (; c < kv_rows; c += 64) {
                    माप_प्रकार         ci          = c / 64;
                    स्थिरांक माप_प्रकार   ne          = hex_smin(kv_rows - c, 64);
                    HVX_VectorPred q_tail_keep = Q6_Q_vsetq2_R(ne * आकार(__fp16));
                    जर (ne < 64) {
                        my_row_buf0[ci] = Q6_V_vmux_QVV(q_tail_keep, my_row_buf0[ci], v_neg_inf);
                        my_row_buf1[ci] = Q6_V_vmux_QVV(q_tail_keep, my_row_buf1[ci], v_neg_inf);
                    }
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_row_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_row_buf1[ci]);
                }
            }

            v_s_rowmax0 = hvx_vec_reduce_max_f16(v_s_rowmax0);
            v_s_rowmax1 = hvx_vec_reduce_max_f16(v_s_rowmax1);

            // Splat m_prev[r], m_prev[r+1] from the float per-row accumulators and convert to fp16 vectors
            HVX_Vector v_m_prev0, v_m_prev1;
            जर (r_vec_off < 32) {
                HVX_Vector v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, r_vec_off * 4));
                v_m_prev0 = hvx_vec_f32_to_f16(v0, v0);
                जर (r + 1 < n_rows_g) {
                    HVX_Vector v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, (r_vec_off + 1) * 4));
                    v_m_prev1 = hvx_vec_f32_to_f16(v1, v1);
                } अन्यथा {
                    v_m_prev1 = Q6_V_vzero();
                }
            } अन्यथा {
                HVX_Vector v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v1, (r_vec_off - 32) * 4));
                v_m_prev0 = hvx_vec_f32_to_f16(v0, v0);
                जर (r + 1 < n_rows_g) {
                    HVX_Vector v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v1, (r_vec_off + 1 - 32) * 4));
                    v_m_prev1 = hvx_vec_f32_to_f16(v1, v1);
                } अन्यथा {
                    v_m_prev1 = Q6_V_vzero();
                }
            }

            HVX_Vector v_dup_m0 = Q6_Vhf_vmax_VhfVhf(v_m_prev0, v_s_rowmax0);
            HVX_Vector v_dup_m1 = Q6_Vhf_vmax_VhfVhf(v_m_prev1, v_s_rowmax1);

            // Insert row r, r+1 rowmax into rowmax_acc_v
            {
                HVX_VectorPred p_start = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_VectorPred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_VectorPred p_end   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_VectorPred p_lane0 = Q6_Q_and_QQn(p_mid, p_start);
                HVX_VectorPred p_lane1 = Q6_Q_and_QQn(p_end, p_mid);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane0, v_dup_m0, rowmax_acc_v);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane1, v_dup_m1, rowmax_acc_v);
            }

            // Compute P = exp(S - m_new)
            स्थिरांक HVX_Vector v_zero      = Q6_V_vzero();
            HVX_Vector       v_p_rowsum0 = v_zero;
            HVX_Vector       v_p_rowsum1 = v_zero;

            माप_प्रकार c = 0;
            साठी (; c + 64 < kv_rows; c += 128) {
                माप_प्रकार     ci0          = c / 64;
                माप_प्रकार     ci1          = ci0 + 1;

                HVX_Vector v_s_minus_m0_0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci0], v_dup_m0);
                HVX_Vector v_s_minus_m1_0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci0], v_dup_m1);
                HVX_Vector v_s_minus_m0_1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci1], v_dup_m0);
                HVX_Vector v_s_minus_m1_1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci1], v_dup_m1);

                HVX_Vector v_p_row0_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_0));
                HVX_Vector v_p_row1_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_0));
                HVX_Vector v_p_row0_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_1));
                HVX_Vector v_p_row1_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_1));

                __fp16 *     out_dtile0  = p_st_base + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                __fp16 *     out_dtile1  = p_st_base + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_p_out0_0  = ((HVX_Vector *) out_dtile0) + r1 / 2;
                HVX_Vector * pv_p_out1_0  = pv_p_out0_0 + 16;
                HVX_Vector * pv_p_out0_1  = ((HVX_Vector *) out_dtile1) + r1 / 2;
                HVX_Vector * pv_p_out1_1  = pv_p_out0_1 + 16;

                HVX_VectorPair vp_p_dual0 = Q6_W_vshuff_VVR(v_p_row1_hf_0, v_p_row0_hf_0, -2);
                *pv_p_out0_0               = Q6_V_lo_W(vp_p_dual0);
                *pv_p_out1_0               = Q6_V_hi_W(vp_p_dual0);

                HVX_VectorPair vp_p_dual1 = Q6_W_vshuff_VVR(v_p_row1_hf_1, v_p_row0_hf_1, -2);
                *pv_p_out0_1               = Q6_V_lo_W(vp_p_dual1);
                *pv_p_out1_1               = Q6_V_hi_W(vp_p_dual1);

                HVX_VectorPair vp_p0_0 = hvx_vec_f16_to_f32_shuff(v_p_row0_hf_0);
                HVX_VectorPair vp_p1_0 = hvx_vec_f16_to_f32_shuff(v_p_row1_hf_0);
                HVX_VectorPair vp_p0_1 = hvx_vec_f16_to_f32_shuff(v_p_row0_hf_1);
                HVX_VectorPair vp_p1_1 = hvx_vec_f16_to_f32_shuff(v_p_row1_hf_1);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_0), Q6_V_hi_W(vp_p0_0)));
                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_1), Q6_V_hi_W(vp_p0_1)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_0), Q6_V_hi_W(vp_p1_0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_1), Q6_V_hi_W(vp_p1_1)));
            }
            साठी (माप_प्रकार c_rem = c; c_rem < kv_rows; c_rem += 64) {
                माप_प्रकार     ci           = c_rem / 64;
                HVX_Vector v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(my_row_buf0[ci], v_dup_m0);
                HVX_Vector v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(my_row_buf1[ci], v_dup_m1);

                HVX_Vector v_p_row0_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0));
                HVX_Vector v_p_row1_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1));
                __fp16 *     out_dtile  = p_st_base + ci * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_p_out0  = ((HVX_Vector *) out_dtile) + r1 / 2;
                HVX_Vector * pv_p_out1  = pv_p_out0 + 16;

                HVX_VectorPair vp_p_dual = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
                *pv_p_out0               = Q6_V_lo_W(vp_p_dual);
                *pv_p_out1               = Q6_V_hi_W(vp_p_dual);

                HVX_VectorPair vp_p0 = hvx_vec_f16_to_f32_shuff(v_p_row0_hf);
                HVX_VectorPair vp_p1 = hvx_vec_f16_to_f32_shuff(v_p_row1_hf);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0), Q6_V_hi_W(vp_p0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1), Q6_V_hi_W(vp_p1)));
            }

            HVX_Vector rowsum0_sf = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum0));
            HVX_Vector rowsum1_sf = hvx_vec_reduce_sum_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum1));
            {
                HVX_Vector rv0_v = hvx_vec_f32_to_f16(rowsum0_sf, rowsum0_sf);
                HVX_Vector rv1_v = hvx_vec_f32_to_f16(rowsum1_sf, rowsum1_sf);

                HVX_VectorPred p_start = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_VectorPred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_VectorPred p_end   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_VectorPred p_lane0 = Q6_Q_and_QQn(p_mid, p_start);
                HVX_VectorPred p_lane1 = Q6_Q_and_QQn(p_end, p_mid);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane0, rv0_v, rowsum_acc_v);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane1, rv1_v, rowsum_acc_v);
            }
        }

        // Inline fa_ml_update_and_build_d for this vector (lock-free and in parallel)
        HVX_VectorPair rowmax_acc_pair    = hvx_vec_f16_to_f32(rowmax_acc_v);
        HVX_Vector     v_rowmax_acc_f32_0 = Q6_V_lo_W(rowmax_acc_pair);
        HVX_Vector     v_rowmax_acc_f32_1 = Q6_V_hi_W(rowmax_acc_pair);

        HVX_Vector v_m_curr0 = Q6_Vsf_vmax_VsfVsf(m_prev_v0, v_rowmax_acc_f32_0);
        HVX_Vector v_m_curr1 = Q6_Vsf_vmax_VsfVsf(m_prev_v1, v_rowmax_acc_f32_1);

        HVX_Vector v_m_diff0 = HVX_OP_SUB_F32(m_prev_v0, v_m_curr0);
        HVX_Vector v_m_diff1 = HVX_OP_SUB_F32(m_prev_v1, v_m_curr1);

        HVX_Vector v_m_diff_f16   = hvx_vec_f32_to_f16(v_m_diff0, v_m_diff1);
        HVX_Vector exp_m_diff_f16 = hvx_vec_exp2_f16(v_m_diff_f16);

        HVX_VectorPair exp_m_diff_pair = hvx_vec_f16_to_f32(exp_m_diff_f16);
        HVX_Vector exp_m_diff0 = Q6_V_lo_W(exp_m_diff_pair);
        HVX_Vector exp_m_diff1 = Q6_V_hi_W(exp_m_diff_pair);

        HVX_VectorPair rowsum_acc_pair = hvx_vec_f16_to_f32(rowsum_acc_v);
        HVX_Vector     v_rowsum_acc_f32_0 = Q6_V_lo_W(rowsum_acc_pair);
        HVX_Vector     v_rowsum_acc_f32_1 = Q6_V_hi_W(rowsum_acc_pair);

        HVX_Vector l_prev_v0 = factx->vtcm_l_vec[r_vec_idx * 2 + 0];
        HVX_Vector l_prev_v1 = factx->vtcm_l_vec[r_vec_idx * 2 + 1];
        HVX_Vector v_l_curr0 = HVX_OP_ADD_F32(HVX_OP_MUL_F32(l_prev_v0, exp_m_diff0), v_rowsum_acc_f32_0);
        HVX_Vector v_l_curr1 = HVX_OP_ADD_F32(HVX_OP_MUL_F32(l_prev_v1, exp_m_diff1), v_rowsum_acc_f32_1);

        factx->vtcm_m_vec[r_vec_idx * 2 + 0] = v_m_curr0;
        factx->vtcm_m_vec[r_vec_idx * 2 + 1] = v_m_curr1;
        factx->vtcm_l_vec[r_vec_idx * 2 + 0] = v_l_curr0;
        factx->vtcm_l_vec[r_vec_idx * 2 + 1] = v_l_curr1;

        // Build diagonal tile D = diag(exp(m_diff))
        स्थिरांक HVX_Vector     v_offsets = *(स्थिरांक HVX_Vector *) d_tile_scatter_offsets;
        स्थिरांक HVX_VectorPred q_32_mask = Q6_Q_vsetq_R(32 * आकार(__fp16));
        HVX_Vector           v_exp_m_diff = exp_m_diff_f16;

        __fp16 * स्थिरांक d_tiles_out = factx->vtcm_d_tiles[args->buf_idx];

        माप_प्रकार t0 = r_vec_idx * 2;
        जर (t0 < args->n_row_tiles) {
            स्थिरांक HVX_Vector v_content = v_exp_m_diff;
            __fp16 *         out_base  = d_tiles_out + t0 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_mask, (माप_प्रकार) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
        }

        माप_प्रकार t1 = r_vec_idx * 2 + 1;
        जर (t1 < args->n_row_tiles) {
            स्थिरांक HVX_Vector v_content = Q6_V_vror_VR(v_exp_m_diff, 64);
            __fp16 *         out_base  = d_tiles_out + t1 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_mask, (माप_प्रकार) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (args->q_start * G + vec_start * 64));
}

स्थिर शून्य fa_softmax_thread_nomask(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, शून्य * data) {
    fa_softmax_impl(n, i, data,
                    /*has_mask=*/false,
                    /*mask_broadcast=*/false,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

स्थिर शून्य fa_softmax_thread_mask_broadcast_g1(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, शून्य * data) {
    fa_softmax_impl(n, i, data,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/true,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

स्थिर शून्य fa_softmax_thread_mask_broadcast_gn(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, शून्य * data) {
    fa_softmax_impl(n, i, data,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

स्थिर शून्य fa_softmax_thread(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, शून्य * data) {
    fa_softmax_args_t *     args  = (fa_softmax_args_t *) data;
    संरचना hmx_fa_context * factx = args->factx;

    स्थिरांक bool has_mask       = (args->mask != रिक्त);
    स्थिरांक bool mask_broadcast = factx->mask_broadcast;
    स्थिरांक bool is_g1          = (args->G == 1);
    स्थिरांक bool has_alibi      = args->has_alibi;
    स्थिरांक bool has_softcap    = factx->has_softcap;

    fa_softmax_impl(n, i, data, has_mask, mask_broadcast, is_g1, has_alibi, has_softcap);
}

स्थिर ओळीत HVX_Vector hvx_fa_load_sinks(स्थिरांक दशांश * sinks_data, माप_प्रकार idx, माप_प्रकार मोजणी) {
    स्थिरांक HVX_Vector * v_src    = (स्थिरांक HVX_Vector *) sinks_data + (idx >> 5);
    स्थिरांक माप_प्रकार       lane_off = idx & 31;
    स्थिरांक माप_प्रकार       byte_off = lane_off * आकार(दशांश);
    परत (lane_off + मोजणी > 32)
        ? Q6_V_valign_VVR(v_src[1], v_src[0], byte_off)
        : Q6_V_vror_VR(v_src[0], byte_off);
}

स्थिर __attribute__((noinline)) शून्य fa_build_d_diag_inv_l(संरचना hmx_fa_context * factx,
                                                            माप_प्रकार                  n_row_tiles,
                                                            माप_प्रकार                  n_row_tiles_g_br,
                                                            uint32_t                kv_head,
                                                            माप_प्रकार                  n_rows_g) {
    (शून्य) n_row_tiles_g_br;

    // sinks: l += exp(sink - m * ln2) in f32, m and l are in the log2 domain
    जर (factx->sinks) {
        स्थिरांक दशांश *    sinks_data   = factx->vtcm_sinks;
        स्थिरांक माप_प्रकार     G            = factx->G;
        स्थिरांक माप_प्रकार     head_base    = (माप_प्रकार) kv_head * G;
        स्थिरांक HVX_Vector v_ln2        = Q6_V_vsplat_R(EXP_LOGN2);
        स्थिरांक HVX_Vector v_head_sinks = (G <= 32) ? hvx_fa_load_sinks(sinks_data, head_base, G) : Q6_V_vzero();
        HVX_Vector       v_sink       = Q6_V_vzero();
        माप_प्रकार           g_prev       = SIZE_MAX;

        साठी (माप_प्रकार i = 0; i < n_row_tiles; ++i) {
            // the sink pattern of a tile only depends on (i * 32) % G, rebuild it when that changes
            स्थिरांक माप_प्रकार r0 = i * 32;
            स्थिरांक माप_प्रकार g0 = fastmodulo(r0, G, &factx->div_G);
            जर (g0 != g_prev) {
                स्थिरांक माप_प्रकार rem_g = G - g0;
                जर (G <= 32) {
                    स्थिरांक HVX_Vector v_lo = Q6_V_vror_VR(v_head_sinks, g0 * आकार(दशांश));
                    स्थिरांक HVX_Vector v_hi = Q6_V_vror_VR(v_head_sinks, 128 - rem_g * आकार(दशांश));
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(rem_g * आकार(दशांश)), v_lo, v_hi);
                    साठी (माप_प्रकार len = G; len < 32; len <<= 1) {
                        स्थिरांक HVX_VectorPred q_keep    = Q6_Q_vsetq_R(len * आकार(दशांश));
                        स्थिरांक HVX_Vector     v_shifted = Q6_V_vror_VR(v_sink, 128 - len * आकार(दशांश));
                        v_sink = Q6_V_vmux_QVV(q_keep, v_sink, v_shifted);
                    }
                } अन्यथा जर (rem_g >= 32) {
                    v_sink = hvx_fa_load_sinks(sinks_data, head_base + g0, 32);
                } अन्यथा {
                    स्थिरांक HVX_Vector v_lo = hvx_fa_load_sinks(sinks_data, head_base + g0, rem_g);
                    स्थिरांक HVX_Vector v_hi = hvx_fa_load_sinks(sinks_data, head_base, 32 - rem_g);
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq_R(rem_g * आकार(दशांश)), v_lo, Q6_V_vror_VR(v_hi, 128 - rem_g * आकार(दशांश)));
                }
                g_prev = g0;
            }

            HVX_Vector v_d = HVX_OP_SUB_F32(v_sink, HVX_OP_MUL_F32(factx->vtcm_m_vec[i], v_ln2));
            जर (r0 + 32 > n_rows_g) {
                स्थिरांक HVX_VectorPred q_valid = Q6_Q_vsetq_R((n_rows_g - r0) * आकार(दशांश));
                v_d = Q6_V_vmux_QVV(q_valid, v_d, Q6_V_vzero());
            }
            factx->vtcm_l_vec[i] = HVX_OP_ADD_F32(factx->vtcm_l_vec[i], hvx_vec_exp_f32(v_d));
        }
    }

    स्थिरांक HVX_Vector     v_offsets = *(स्थिरांक HVX_Vector *) d_tile_scatter_offsets;
    स्थिरांक HVX_VectorPred q_32_mask = Q6_Q_vsetq_R(32 * आकार(__fp16));
    स्थिरांक HVX_Vector     one       = hvx_vec_splat_f32(1.0f);

    HVX_Vector v_content = Q6_V_vzero();
    साठी (माप_प्रकार i = 0; i < n_row_tiles; ++i) {
        जर ((i % 2) == 0) {
            HVX_Vector inv_lo = HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i]));
            HVX_Vector inv_hi = (i + 1 < n_row_tiles) ? HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i + 1])) : Q6_V_vzero();
            v_content = hvx_vec_f32_to_f16(inv_lo, inv_hi);
        } अन्यथा {
            v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 * out_base = factx->vtcm_d_inv_l + i * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_mask, (माप_प्रकार) out_base, HMX_FP16_TILE_SIZE - 1, v_offsets, v_content);
    }
}

स्थिर शून्य fa_phase_softmax_and_build_d(संरचना hmx_fa_context * factx,
                                         fa_softmax_args_t *     sargs,
                                         माप_प्रकार                  n_row_tiles,
                                         माप_प्रकार                  n_row_tiles_g_br) {
    work_queue_t wp = factx->octx->ctx->work_queue;
    स्थिरांक माप_प्रकार n_row_vec_cnt = hmx_ceil_div(sargs->n_rows_g, 64);

    worker_callback_t softmax_fn = fa_softmax_thread;
    जर (sargs->mask == रिक्त && !factx->has_softcap && !sargs->has_alibi) {
        softmax_fn = fa_softmax_thread_nomask;
    } अन्यथा जर (sargs->mask != रिक्त && factx->mask_broadcast && !factx->has_softcap && !sargs->has_alibi) {
        जर (sargs->G == 1) {
            softmax_fn = fa_softmax_thread_mask_broadcast_g1;
        } अन्यथा {
            softmax_fn = fa_softmax_thread_mask_broadcast_gn;
        }
    }

    जर (factx->n_threads > 1 && n_row_vec_cnt >= 2) {
        uint32_t n_use = (uint32_t) hex_smin((माप_प्रकार) factx->n_threads, n_row_vec_cnt);
        sargs->thread_div = init_fastdiv_मूल्ये(n_use);
        work_queue_run(wp, softmax_fn, sargs, n_use);
    } अन्यथा {
        softmax_fn(1, 0, sargs);
    }
}

// ============================================================================
// HMX job structs and worker functions
// ============================================================================

प्रकारव्याख्या संरचना {
    स्थिरांक __fp16 * q_tiles;
    स्थिरांक __fp16 * k_tiles;
    __fp16 *       s_tiles;
    माप_प्रकार         n_row_tiles;
    माप_प्रकार         n_col_tiles;
    माप_प्रकार         n_dot_tiles;  // DK / 32
    माप_प्रकार         n_tiles_per_bc;
    uint8_t *      hmx_scales;
} hmx_fa_qk_job_t;

स्थिर शून्य hmx_fa_qk_dot_worker(शून्य * data) {
    hmx_fa_qk_job_t * job            = (hmx_fa_qk_job_t *) data;
    स्थिरांक माप_प्रकार      n_row_tiles    = job->n_row_tiles;
    स्थिरांक माप_प्रकार      n_col_tiles    = job->n_col_tiles;
    स्थिरांक माप_प्रकार      n_dot_tiles    = job->n_dot_tiles;
    स्थिरांक माप_प्रकार      n_tiles_per_bc = job->n_tiles_per_bc;
    स्थिरांक __fp16 * restrict q_tiles  = job->q_tiles;
    स्थिरांक __fp16 * restrict k_tiles  = job->k_tiles;
    __fp16 * restrict s_tiles        = job->s_tiles;
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    यंत्र अस्थिर(HMX_SET_BIAS("%0") :: "r"((अचिन्हित पूर्णांक)job->hmx_scales));
    स्थिरांक माप_प्रकार dot_stride = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    साठी (माप_प्रकार r = 0; r < n_row_tiles; ++r) {
        स्थिरांक __fp16 * row_tiles = q_tiles + r * dot_stride;
        स्थिरांक __fp16 * col_tiles = k_tiles;
        __fp16 *       out_tile  = s_tiles + r * n_tiles_per_bc * HMX_FP16_TILE_N_ELMS;

        साठी (माप_प्रकार c = 0; c < n_col_tiles; ++c) {
            hmx_fa_qk_dot_tile(row_tiles, col_tiles, out_tile, n_dot_tiles);
            col_tiles += dot_stride;
            out_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

प्रकारव्याख्या संरचना {
    __fp16 *       o_curr;
    स्थिरांक __fp16 * o_prev;
    स्थिरांक __fp16 * p_tiles;
    स्थिरांक __fp16 * v_tiles;
    स्थिरांक __fp16 * d_tiles;
    uint8_t *      hmx_scales;
    माप_प्रकार         n_row_tiles;
    माप_प्रकार         n_col_tiles;
    माप_प्रकार         n_row_tiles_g_br;
    माप_प्रकार         n_tiles_per_bc;
    माप_प्रकार         DV;
} hmx_fa_o_update_job_t;

स्थिर शून्य hmx_fa_o_update_worker(शून्य * data) {
    hmx_fa_o_update_job_t * job              = (hmx_fa_o_update_job_t *) data;
    स्थिरांक माप_प्रकार            n_row_tiles      = job->n_row_tiles;
    स्थिरांक माप_प्रकार            n_col_tiles      = job->n_col_tiles;
    स्थिरांक माप_प्रकार            n_row_tiles_g_br = job->n_row_tiles_g_br;
    स्थिरांक माप_प्रकार            n_tiles_per_bc   = job->n_tiles_per_bc;
    स्थिरांक माप_प्रकार            DV_tiles         = job->DV / 32;
    स्थिरांक __fp16 * restrict d_tiles          = job->d_tiles;
    स्थिरांक __fp16 * restrict p_tiles          = job->p_tiles;
    स्थिरांक __fp16 * restrict v_tiles          = job->v_tiles;
    स्थिरांक __fp16 * restrict o_prev           = job->o_prev;
    __fp16 * restrict o_curr                 = job->o_curr;
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(n_col_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    यंत्र अस्थिर(HMX_SET_BIAS("%0") :: "r"((अचिन्हित पूर्णांक)job->hmx_scales));
    स्थिरांक माप_प्रकार o_stride = n_row_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    स्थिरांक माप_प्रकार v_stride = n_tiles_per_bc * HMX_FP16_TILE_N_ELMS;
    साठी (माप_प्रकार r = 0; r < n_row_tiles; ++r) {
        स्थिरांक __fp16 * d_diag     = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        स्थिरांक __fp16 * p_tile_in  = p_tiles + (r * n_tiles_per_bc) * HMX_FP16_TILE_N_ELMS;
        स्थिरांक __fp16 * o_rc       = o_prev + r * HMX_FP16_TILE_N_ELMS;
        स्थिरांक __fp16 * v_tile_in  = v_tiles;
        __fp16       * o_tile_out = o_curr + r * HMX_FP16_TILE_N_ELMS;

        साठी (माप_प्रकार c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_update_tile(d_diag, o_rc, p_tile_in, v_tile_in, o_tile_out, n_col_tiles);
            o_rc       += o_stride;
            v_tile_in  += v_stride;
            o_tile_out += o_stride;
        }
    }
}

प्रकारव्याख्या संरचना {
    __fp16 *       o_curr;   // output (row-major tile layout)
    स्थिरांक __fp16 * o_prev;   // input (column-major tile layout)
    स्थिरांक __fp16 * d_tiles;  // diag(1/l) tiles
    uint8_t *      hmx_scales;
    माप_प्रकार         n_row_tiles;
    माप_प्रकार         n_row_tiles_g_br;
    माप_प्रकार         DV;
} hmx_fa_o_norm_job_t;

स्थिर शून्य hmx_fa_o_norm_worker(शून्य * data) {
    hmx_fa_o_norm_job_t * job              = (hmx_fa_o_norm_job_t *) data;
    स्थिरांक माप_प्रकार          n_row_tiles      = job->n_row_tiles;
    स्थिरांक माप_प्रकार          n_row_tiles_g_br = job->n_row_tiles_g_br;
    स्थिरांक माप_प्रकार          DV_tiles         = job->DV / 32;
    स्थिरांक __fp16 * restrict d_tiles        = job->d_tiles;
    स्थिरांक __fp16 * restrict o_prev         = job->o_prev;
    __fp16 * restrict o_curr               = job->o_curr;
    __builtin_assume(n_row_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    यंत्र अस्थिर(HMX_SET_BIAS("%0") :: "r"((अचिन्हित पूर्णांक)job->hmx_scales));
    स्थिरांक माप_प्रकार o_stride = n_row_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    साठी (माप_प्रकार r = 0; r < n_row_tiles; ++r) {
        स्थिरांक __fp16 * d_diag = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        स्थिरांक __fp16 * o_rc = o_prev + r * HMX_FP16_TILE_N_ELMS;
        __fp16 *       o_out = o_curr + r * DV_tiles * HMX_FP16_TILE_N_ELMS;

        साठी (माप_प्रकार c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_norm_tile(d_diag, o_rc, o_out);
            o_rc  += o_stride;
            o_out += HMX_FP16_TILE_N_ELMS;
        }
    }
}

// Populate per-GQA-row ALiBi slopes for a given KV head.
स्थिर __attribute__((noinline)) शून्य fa_compute_slopes(
                              स्थिरांक संरचना hmx_fa_context * factx,
                              uint32_t                      kv_head,
                              माप_प्रकार                        n_rows_g) {
    __fp16 * slopes = factx->vtcm_slopes;
    जर (factx->max_bias == 0.0f) {
        hvx_splat_f16_a(slopes, 1.0f, n_rows_g);
        परत;
    }

    स्थिरांक uint32_t G           = factx->G;
    स्थिरांक uint32_t n_head_log2 = factx->n_head_log2;
    स्थिरांक दशांश    m0          = factx->m0;
    स्थिरांक दशांश    m1          = factx->m1;

    __fp16 temp_slopes[512] __attribute__((aligned(128)));
    जर (G <= 32) {
        // Fast path: Compute G unique slope values in vector registers
        HVX_Vector v_val = hvx_alibi_slopes(kv_head, G, n_head_log2, m0, m1);

        __fp16 temp_slopes_aligned[64] __attribute__((aligned(128)));
        hvx_vmem(temp_slopes_aligned) = hvx_vec_f32_to_f16(v_val, Q6_V_vzero());

        साठी (uint32_t i = 0; i < G; ++i) {
            temp_slopes[i] = temp_slopes_aligned[i];
        }
    } अन्यथा {
        // Fallback path: G > 32 (rare configurations)
        साठी (uint32_t i = 0; i < G; ++i) {
            temp_slopes[i] = (__fp16)alibi_slope(kv_head * G + i, n_head_log2, m0, m1);
        }
    }

    // Allocate stack buffer to avoid scalar writes to VTCM (which generates L2 misses)
    __fp16 local_slopes[n_rows_g] __attribute__((aligned(128)));
    साठी (माप_प्रकार r = 0; r < n_rows_g; ++r) {
        local_slopes[r] = temp_slopes[fastmodulo(r, G, &factx->div_G)];
    }

    // Copy to VTCM slopes using HVX block copy (both are aligned to 128 bytes)
    hvx_copy_f16_aa((uint8_t *)slopes, (स्थिरांक uint8_t *)local_slopes, n_rows_g);
}

स्थिर शून्य fa_ढकला_mask_dma_gqa(
    dma_queue *               dma_q,
    स्थिरांक संरचना htp_tensor * mask,
    uint32_t                  q_start,
    uint32_t                  im3,
    uint32_t                  kv_start,
    uint32_t                  kv_head,
    uint32_t                  G,
    uint32_t                  m_ओळ_bytes,
    uint32_t                  kv_rows,
    uint32_t                  n_rows_q,
    संरचना hmx_fa_context *   factx
) {
    साठी (uint32_t g = 0; g < G; ++g) {
        स्थिरांक uint32_t h_idx = kv_head * G + g;
        स्थिरांक uint32_t im2 = fastmodulo(h_idx, mask->ne[2], &factx->src3_div2);
        स्थिरांक dma_addr_t ms_src = mask->data + q_start * mask->nb[1] +
                                  im2 * mask->nb[2] + im3 * mask->nb[3] + kv_start * आकार(__fp16);
        uint8_t * ms_dst = (uint8_t *) factx->vtcm_mask_buf + g * m_ओळ_bytes;
        dma_queue_ढकला(dma_q, dma_make_data(ms_dst, ms_src), G * m_ओळ_bytes, mask->nb[1], kv_rows * आकार(__fp16), n_rows_q);
    }
}

स्थिर शून्य fa_काढा_mask_dma_gqa(dma_queue * dma_q, uint32_t G) {
    साठी (uint32_t g = 0; g < G; ++g) {
        dma_queue_काढा(dma_q);
    }
}

स्थिर ओळीत शून्य fa_prefetch_block(dma_queue * dma_q, स्थिरांक संरचना htp_tensor * k, स्थिरांक संरचना htp_tensor * v, स्थिरांक संरचना htp_tensor * mask,
                                     uint32_t b, माप_प्रकार Bc, माप_प्रकार size_k_row_padded, माप_प्रकार size_k_row, माप_प्रकार size_v_row_padded, माप_प्रकार size_v_row,
                                     uint32_t ik2, uint32_t ik3, uint32_t iv2, uint32_t iv3, uint32_t q_start, uint32_t im3, uint32_t kv_head, uint32_t G,
                                     माप_प्रकार m_ओळ_bytes, माप_प्रकार n_rows_q, माप_प्रकार nek1, माप_प्रकार prefetch_buf, संरचना hmx_fa_context * factx) {
    स्थिरांक uint32_t  prefetch_start = b * Bc;
    स्थिरांक uint32_t  prefetch_rows  = hex_smin(Bc, nek1 - prefetch_start);
    स्थिरांक dma_addr_t k_prefetch_src = k->data + prefetch_start * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
    dma_queue_ढकला(dma_q, dma_make_data(factx->vtcm_k_fp16[prefetch_buf], k_prefetch_src), size_k_row_padded, k->nb[1], size_k_row, prefetch_rows);
    स्थिरांक dma_addr_t v_prefetch_src = v->data + prefetch_start * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
    dma_queue_ढकला(dma_q, dma_make_data(factx->vtcm_v_fp16[prefetch_buf], v_prefetch_src), size_v_row_padded, v->nb[1], size_v_row, prefetch_rows);

    जर (mask) {
        जर (__builtin_expect(factx->mask_broadcast, true)) {
            स्थिरांक dma_addr_t ms_src = mask->data + q_start * mask->nb[1] + im3 * mask->nb[3] + prefetch_start * आकार(__fp16);
            dma_cache_fa_ढकला(dma_q, &factx->m_cache, ms_src, m_ओळ_bytes, mask->nb[1], prefetch_rows * आकार(__fp16), n_rows_q);
        } अन्यथा {
            fa_ढकला_mask_dma_gqa(dma_q, mask, q_start, im3, prefetch_start, kv_head, G, m_ओळ_bytes, prefetch_rows, n_rows_q, factx);
        }
    }
}

// ============================================================================
// Core HMX flash attention algorithm (GQA-merged)
// ============================================================================

पूर्णांक hmx_flash_attn_ext(संरचना htp_ops_context * octx) {
    संरचना htp_thread_trace * tr_hvx = &octx->ctx->trace[0];
    संरचना htp_thread_trace * tr_hmx = &octx->ctx->trace[HTP_MAX_NTHREADS];
    स्थिरांक संरचना htp_tensor * q    = octx->src[0];
    स्थिरांक संरचना htp_tensor * k    = octx->src[1];
    स्थिरांक संरचना htp_tensor * v    = octx->src[2];
    स्थिरांक संरचना htp_tensor * mask = (octx->src[3] && octx->src[3]->data) ? octx->src[3] : रिक्त;
    स्थिरांक संरचना htp_tensor * dst  = octx->dst;

    संरचना htp_context * स्थिरांक ctx = octx->ctx;

    जर (!ctx->hmx_enabled) {
        परत HTP_STATUS_NO_SUPPORT;
    }

    // Dimensions
    स्थिरांक uint32_t neq0 = q->ne[0];  // head_dim (DK)
    स्थिरांक uint32_t neq1 = q->ne[1];  // n_tokens
    स्थिरांक uint32_t neq2 = q->ne[2];  // n_heads
    स्थिरांक uint32_t neq3 = q->ne[3];  // n_seqs

    स्थिरांक uint32_t nek0 = k->ne[0];  // head_dim
    स्थिरांक uint32_t nek1 = k->ne[1];  // kv_len

    स्थिरांक uint32_t nev0 = v->ne[0];  // head_dim (DV)

    स्थिरांक uint32_t DK = neq0;
    स्थिरांक uint32_t DV = nev0;

    // HMX tiles head_dim in units of 64. head_dim need not be 64- (or 32-) aligned:
    // we can operate on DK/DV rounded up to 64 with tail lanes [D, D_pad) zero-filled.
    स्थिरांक uint32_t DK_pad = hex_round_up(DK, 64);
    स्थिरांक uint32_t DV_pad = hex_round_up(DV, 64);
    जर (DK == 0 || DV == 0) {
        परत HTP_STATUS_NO_SUPPORT;
    }

    स्थिरांक संरचना htp_fa_kernel_params * kparams = (स्थिरांक संरचना htp_fa_kernel_params *) octx->kernel_params;
    स्थिरांक uint32_t n_kv_heads = k->ne[2];

    // ======== Build context ========
    संरचना hmx_fa_context factx;
    स_रखो(&factx, 0, आकार(factx));
    factx.octx           = octx;
    factx.sinks          = octx->src[4];  // NULL if this op has no attention sinks
    factx.k              = k;
    factx.v              = v;
    factx.n_threads      = kparams->n_threads;
    factx.DK             = DK;
    factx.DV             = DV;
    factx.DK_pad         = DK_pad;
    factx.DV_pad         = DV_pad;
    factx.n_kv           = nek1;
    factx.n_kv_heads     = n_kv_heads;
    factx.n_heads        = neq2;
    factx.G              = kparams->G;
    factx.div_G          = kparams->u.hmx.div_G;
    factx.neq1           = neq1;
    factx.Br             = kparams->Br;
    factx.Bc             = kparams->Bc;
    factx.g_br           = kparams->u.hmx.g_br;
    factx.n_kv_blocks    = kparams->n_kv_blocks;
    factx.is_q_fp32      = (q->type == HTP_TYPE_F32);
    factx.is_dst_fp32    = (dst->type == HTP_TYPE_F32);
    factx.pipeline       = (kparams->u.hmx.pipeline != 0);
    factx.mask_broadcast = (kparams->u.hmx.mask_broadcast != 0);
    जर (mask) {
        factx.src3_div2  = kparams->src3_div2;
        factx.src3_div3  = kparams->src3_div3;
    }

    factx.has_softcap   = (kparams->logit_softcap != 0.0f);
    factx.scale         = (__fp16) kparams->scale;
    factx.max_bias      = kparams->max_bias;
    factx.logit_softcap = 0;
    जर (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_head_log2 = kparams->n_head_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    स्थिरांक uint32_t Br = factx.Br;
    स्थिरांक uint32_t Bc = factx.Bc;
    स्थिरांक uint32_t g_br = factx.g_br;
    स्थिरांक bool pipeline = factx.pipeline;
    स्थिरांक uint32_t n_threads = factx.n_threads;
    स्थिरांक uint32_t G = factx.G;

    // Multi-device: prefer head-parallel partitioning (each core owns a disjoint head
    // shard), falling back to Q-block (token) split when heads don't divide evenly.
    स्थिरांक uint32_t n_q_blocks = (neq1 + Br - 1) / Br;
    uint32_t q_start_min  = 0;
    uint32_t q_start_max  = neq1;
    uint32_t kv_head_min  = 0;
    uint32_t kv_head_max  = n_kv_heads;

    जर (octx->ctx->mdev.मोजणी > 1) {
        स्थिरांक uint32_t mdev_मोजणी = octx->ctx->mdev.मोजणी;
        स्थिरांक uint32_t mdev_idx   = octx->ctx->mdev.idx;
        स्थिरांक uint32_t dst_e_size = (dst->type == HTP_TYPE_F32) ? आकार(दशांश) : आकार(__fp16);
        स्थिरांक bool can_split      = htp_tensor_can_row_partition(dst, dst_e_size);

        जर (kparams->head_split && can_split && n_kv_heads >= mdev_मोजणी && n_kv_heads % mdev_मोजणी == 0) {
            स्थिरांक uint32_t kv_per_core = n_kv_heads / mdev_मोजणी;
            kv_head_min = mdev_idx * kv_per_core;
            kv_head_max = kv_head_min + kv_per_core;
        } अन्यथा {
            स्थिरांक संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(n_q_blocks, can_split ? 1 : 0, mdev_idx, mdev_मोजणी, &octx->ctx->mdev.मोजणी_div);
            स्थिरांक uint32_t block_start = range.start;
            स्थिरांक uint32_t block_end   = range.start + range.मोजणी;

            जर (block_start >= block_end) {
                परत HTP_STATUS_OK;
            }

            q_start_min = block_start * Br;
            q_start_max = MIN(block_end * Br, neq1);
        }
    }

    // ======== VTCM allocation (GQA-aware) ========
    // K/V row sizes drive the DMA descriptors (not the VTCM layout) and are used
    // throughout the KV loop below. The DMA copies only the real DK/DV columns; the
    // staging rows are padded to hold DK_pad/DV_pad columns (tail zero-filled below)
    // so the HMX interleave/tile logic can operate on 64-aligned head dims.
    स्थिरांक माप_प्रकार size_k_row        = htp_tensor_get_row_size(k->type, DK);
    स्थिरांक माप_प्रकार size_v_row        = htp_tensor_get_row_size(v->type, DV);
    स्थिरांक माप_प्रकार size_k_row_padded = hex_round_up(DK_pad * आकार(__fp16), 128);
    स्थिरांक माप_प्रकार size_v_row_padded = hex_round_up(DV_pad * आकार(__fp16), 128);

    // Build the VTCM layout once (shared with the host estimator) and place every
    // scratch buffer at its computed offset. Padded head dims size the HMX tiles.
    संरचना hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, G, DK_pad, DV_pad, Br, Bc, n_threads, pipeline, factx.is_q_fp32, factx.sinks != रिक्त, factx.n_heads);

    जर (L.total_bytes > ctx->vtcm_size) {
        परत HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * स्थिरांक base = ctx->vtcm_base;

    factx.vtcm_q_dma          = VTCM_LAYOUT_PTR(__fp16, base, L.off_q_dma);
    factx.vtcm_q_tiles        = VTCM_LAYOUT_PTR(__fp16, base, L.off_q_tiles);
    factx.vtcm_o_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_o_tiles[0]);
    factx.vtcm_o_tiles[1]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_o_tiles[1]);
    factx.vtcm_k_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_k_fp16[0]);
    factx.vtcm_k_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_k_fp16[1]);
    factx.vtcm_v_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_v_fp16[0]);
    factx.vtcm_v_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, base, L.off_v_fp16[1]);
    factx.vtcm_k_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_k_tiles[0]);
    factx.vtcm_k_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_k_tiles[1], pipeline);
    factx.vtcm_v_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_v_tiles[0]);
    factx.vtcm_v_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_v_tiles[1], pipeline);
    factx.vtcm_s_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_s_tiles[0]);
    factx.vtcm_s_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_s_tiles[1], pipeline);
    factx.vtcm_p_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_p_tiles[0]);
    factx.vtcm_p_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_p_tiles[1], pipeline);
    factx.vtcm_d_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, base, L.off_d_tiles[0]);
    factx.vtcm_d_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, base, L.off_d_tiles[1], pipeline);
    factx.vtcm_d_inv_l        = VTCM_LAYOUT_PTR(__fp16, base, L.off_d_inv_l);
    factx.vtcm_m_vec          = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_m_vec);
    factx.vtcm_l_vec          = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_l_vec);
    factx.vtcm_s_rowmax       = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_s_rowmax);
    factx.vtcm_p_rowsum       = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_p_rowsum);
    factx.vtcm_row_bufs       = VTCM_LAYOUT_PTR(HVX_Vector, base, L.off_row_bufs);
    factx.row_buf_stride      = L.row_buf_stride;
    factx.vtcm_hmx_scales_id  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_hmx_scales_id);
    factx.vtcm_hmx_scales_qk  = VTCM_LAYOUT_PTR(uint8_t, base, L.off_hmx_scales_qk);
    factx.vtcm_mask_buf       = VTCM_LAYOUT_PTR(__fp16, base, L.off_mask_buf);
    factx.mask_buf_row_stride = L.mask_buf_row_stride;
    factx.q_tile_bytes        = L.q_tile_bytes;
    factx.k_tile_bytes        = L.k_tile_bytes;
    factx.o_tile_bytes        = L.o_tile_bytes;
    factx.col_vec_bytes       = L.col_vec_bytes;
    factx.d_tile_bytes        = L.d_tile_bytes;
    factx.vtcm_slopes         = VTCM_LAYOUT_PTR(__fp16, base, L.off_slopes);
    factx.vtcm_sinks          = VTCM_LAYOUT_PTR_OPTIONAL(दशांश, base, L.off_sinks, factx.sinks != रिक्त);

    स्थिरांक माप_प्रकार m_ओळ_bytes = L.m_ओळ_bytes;  // used by the mask DMAs in the KV loop

    dma_cache_fa_init(&factx.m_cache, (uint8_t *) factx.vtcm_mask_buf, L.m_buf_slot_bytes, HMX_FA_DMA_CACHE_SIZE);

    // Head-dim padding: the K/V DMA staging buffers and the flat-Q buffer are laid out
    // with padded row strides (size_{k,v,q}_row_padded, covering D_pad columns) but the
    // DMA only writes the real D columns per row. Zero the whole staging buffers once up
    // front so tail lanes [D, D_pad) stay zero for all KV blocks. No-op when already aligned.
    जर (DK_pad != DK || DV_pad != DV) {
        स्थिरांक माप_प्रकार k_buf_bytes = (माप_प्रकार) factx.Bc * size_k_row_padded;
        स्थिरांक माप_प्रकार v_buf_bytes = (माप_प्रकार) factx.Bc * size_v_row_padded;
        hvx_splat_u8_a((अक्षर *) factx.vtcm_k_fp16[0], 0, k_buf_bytes);
        hvx_splat_u8_a((अक्षर *) factx.vtcm_k_fp16[1], 0, k_buf_bytes);
        hvx_splat_u8_a((अक्षर *) factx.vtcm_v_fp16[0], 0, v_buf_bytes);
        hvx_splat_u8_a((अक्षर *) factx.vtcm_v_fp16[1], 0, v_buf_bytes);
        // Flat-Q DMA scratch
        जर (factx.vtcm_q_dma) {
            स्थिरांक माप_प्रकार q_dma_bytes = hex_align_up(factx.g_br * DK * (factx.is_q_fp32 ? आकार(दशांश) : आकार(__fp16)), 128);
            hvx_splat_u8_a((अक्षर *) factx.vtcm_q_dma, 0, q_dma_bytes);
        }
    }

    // ======== Initialize HMX output scales ========
    hmx_init_column_scales(factx.vtcm_hmx_scales_id, Q6_V_vsplat_R(0x3c00)); // 1.0
    {
        संघ { __fp16 f; uint16_t u; } sc = { .f = factx.scale };
        hmx_init_column_scales(factx.vtcm_hmx_scales_qk, Q6_V_vsplat_R(sc.u));
    }

    // ======== DMA setup ========
    dma_queue * स्थिरांक dma_q = ctx->dma[0];

    जर (factx.sinks) {
        dma_queue_ढकला(dma_q, dma_make_data(factx.vtcm_sinks, factx.sinks->data), L.sinks_bytes, 0, factx.sinks->size, 1);
        dma_queue_काढा(dma_q);
    }

    स्थिरांक माप_प्रकार n_row_tiles_g_br = g_br / HMX_FP16_TILE_N_ROWS;
    स्थिरांक माप_प्रकार n_tiles_per_bc   = Bc / HMX_FP16_TILE_N_COLS;

    स्थिरांक माप_प्रकार qo_element_size = factx.is_q_fp32 ? आकार(दशांश) : आकार(__fp16);

    स्थिरांक bool q_transposed                 = q->nb[1] < q->nb[2];
    स्थिरांक माप_प्रकार q_src_stride               = q_transposed ? q->nb[2] : q->nb[1];
    स्थिरांक माप_प्रकार q_row_bytes_untransposed   = factx.G * factx.DK * qo_element_size;
    स्थिरांक माप_प्रकार q_row_bytes_trans_factor   = factx.DK * qo_element_size;

    // ======== Reusable job descriptors for pipeline ========
    hmx_fa_qk_job_t       qk_job;
    hmx_fa_o_update_job_t ou_job;
    hmx_fa_o_norm_job_t   on_job;

    // ======== Main loop ========
    साठी (uint32_t ib3 = 0; ib3 < neq3; ++ib3) {
        स्थिरांक uint32_t im3 = mask ? fastmodulo(ib3, mask->ne[3], &factx.src3_div3) : 0;
        साठी (uint32_t q_start = q_start_min; q_start < q_start_max; q_start += Br) {
            स्थिरांक uint32_t n_rows_q    = hex_smin(Br, neq1 - q_start);
            स्थिरांक माप_प्रकार   n_rows_g    = n_rows_q * G;
            स्थिरांक माप_प्रकार   g_br_actual = hex_align_up(n_rows_g, HMX_FP16_TILE_N_ROWS);
            स्थिरांक माप_प्रकार   n_row_tiles = g_br_actual / HMX_FP16_TILE_N_ROWS;

            uint16_t active_blocks[FA_MAX_KV_BLOCKS];
            uint32_t n_active_blocks = factx.n_kv_blocks;
            जर (mask && factx.mask_broadcast && factx.n_kv_blocks > 1) {
                स्थिरांक माप_प्रकार scratch_bytes = factx.k_tile_bytes * (factx.pipeline ? 2 : 1);
                n_active_blocks = fa_scan_active_blocks(
                    dma_q, mask, q_start, n_rows_q, 0, im3, nek1, Bc,
                    factx.n_kv_blocks, factx.vtcm_k_tiles[0], scratch_bytes, active_blocks);
            } अन्यथा {
                #pragma clang loop vectorize(disable)
                साठी (uint32_t b = 0; b < factx.n_kv_blocks; ++b) {
                    active_blocks[b] = (uint16_t) b;
                }
            }
            स्थिरांक uint32_t b0       = (n_active_blocks > 0) ? active_blocks[0] : 0;
            स्थिरांक uint32_t b0_start = b0 * Bc;
            स्थिरांक uint32_t b0_rows  = hex_smin(Bc, nek1 - b0_start);

            साठी (uint32_t kv_head = kv_head_min; kv_head < kv_head_max; ++kv_head) {
                स्थिरांक uint32_t ik2 = kv_head;
                स्थिरांक uint32_t ik3 = fastdiv(ib3, &kparams->broadcast_rk3);
                स्थिरांक uint32_t iv2 = kv_head;
                स्थिरांक uint32_t iv3 = fastdiv(ib3, &kparams->broadcast_rv3);

                // 1. Push Q and KV DMAs for the first kv_head of this q_start block.
                // Subsequent kv_head iterations are enqueued early at the end of the previous iteration.
                जर (kv_head == kv_head_min) {
                    स्थिरांक dma_addr_t q_ptr = q->data + q_start * q->nb[1] +
                                            (kv_head * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    स्थिरांक माप_प्रकार q_row_bytes = q_transposed ? n_rows_q * q_row_bytes_trans_factor : q_row_bytes_untransposed;
                    स्थिरांक माप_प्रकार n_rows      = q_transposed ? factx.G : n_rows_q;
                    dma_queue_ढकला(dma_q, dma_make_data(factx.vtcm_q_dma, q_ptr), q_row_bytes, hex_smax(q_src_stride, q_row_bytes), q_row_bytes, n_rows);

                    जर (n_active_blocks > 0) {
                        स्थिरांक dma_addr_t k_src = k->data + b0_start * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_queue_ढकला(dma_q, dma_make_data(factx.vtcm_k_fp16[0], k_src), size_k_row_padded, k->nb[1], size_k_row, b0_rows);

                        स्थिरांक dma_addr_t v_src = v->data + b0_start * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_queue_ढकला(dma_q, dma_make_data(factx.vtcm_v_fp16[0], v_src), size_v_row_padded, v->nb[1], size_v_row, b0_rows);

                        जर (factx.pipeline && mask) {
                            जर (__builtin_expect(factx.mask_broadcast, true)) {
                                स्थिरांक dma_addr_t ms_src = mask->data + q_start * mask->nb[1] + im3 * mask->nb[3] + b0_start * आकार(__fp16);
                                dma_cache_fa_ढकला(dma_q, &factx.m_cache, ms_src, m_ओळ_bytes, mask->nb[1], b0_rows * आकार(__fp16), n_rows_q);
                            } अन्यथा {
                                fa_ढकला_mask_dma_gqa(dma_q, mask, q_start, im3, b0_start, kv_head, G, m_ओळ_bytes, b0_rows, n_rows_q, &factx);
                            }
                        }
                    }
                }

                // 2. Pop Q DMA (blocks until Q is loaded)
                dma_queue_काढा(dma_q);

                // ---- Load Q block & Initialize per-block state ----
                fa_phase_q_load(&factx, q, q_start, kv_head, ib3, n_rows_g);

                __fp16 * o_tile_prev = factx.vtcm_o_tiles[0];
                __fp16 * o_tile_curr = factx.vtcm_o_tiles[1];

                // ---- KV block loop with DMA double-buffering ----
                माप_प्रकार buf_idx = 0;

                htp_trace_event_start(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_start);
                fa_compute_slopes(&factx, kv_head, n_rows_g);
                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_start);

                स्थिरांक माप_प्रकार k_src_stride = size_k_row_padded / आकार(__fp16);
                स्थिरांक माप_प्रकार v_src_stride = size_v_row_padded / आकार(__fp16);

                hmx_queue_t hmx_q = ctx->hmx_queue;

                जर (factx.pipeline) {
                    // Double-buffered job structs because HMX queue runs asynchronously
                    hmx_fa_qk_job_t qk_job[2];
                    hmx_fa_o_update_job_t ou_job[2];

                    // Prefetch second active block early if there are multiple active blocks
                    जर (n_active_blocks > 1) {
                        fa_prefetch_block(dma_q, k, v, mask, active_blocks[1], Bc, size_k_row_padded, size_k_row, size_v_row_padded, size_v_row,
                                          ik2, ik3, iv2, iv3, q_start, im3, kv_head, G, m_ओळ_bytes, n_rows_q, nek1, 1, &factx);
                    }

                    // Prep and start QK-dot(0)
                    शून्य * curr_k0 = (शून्य *) dma_queue_काढा(dma_q).dst;
                    fa_phase_k_interleave(&factx, b0_rows, k_src_stride, curr_k0, b0_start, 0);

                    qk_job[0].q_tiles        = factx.vtcm_q_tiles;
                    qk_job[0].k_tiles        = factx.vtcm_k_tiles[0];
                    qk_job[0].s_tiles        = factx.vtcm_s_tiles[0];
                    qk_job[0].n_row_tiles    = n_row_tiles;
                    qk_job[0].n_col_tiles    = hmx_ceil_div(b0_rows, HMX_FP16_TILE_N_COLS);
                    qk_job[0].n_dot_tiles    = DK_pad / 32;
                    qk_job[0].n_tiles_per_bc = n_tiles_per_bc;
                    qk_job[0].hmx_scales     = factx.vtcm_hmx_scales_qk;
                    hmx_queue_ढकला(hmx_q, hmx_queue_make_desc(hmx_fa_qk_dot_worker, &qk_job[0]));

                    साठी (uint32_t step = 0; step < n_active_blocks; ++step) {
                        स्थिरांक uint32_t kv_blk      = active_blocks[step];
                        स्थिरांक uint32_t kv_start    = kv_blk * Bc;
                        स्थिरांक uint32_t kv_rows     = hex_smin(Bc, nek1 - kv_start);
                        स्थिरांक माप_प्रकार   n_col_tiles = hmx_ceil_div(kv_rows, HMX_FP16_TILE_N_COLS);

                        // ---- 1. Pop and run V-prep for current block ----
                        शून्य * curr_v = (शून्य *) dma_queue_काढा(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_rows, v_src_stride, curr_v, factx.vtcm_v_tiles[buf_idx], n_tiles_per_bc, kv_start);

                        // ---- 2. Pop and run mask-prep for current block ----
                        __fp16 * current_mask_vtcm = रिक्त;
                        जर (mask) {
                            जर (__builtin_expect(factx.mask_broadcast, true)) {
                                current_mask_vtcm = (__fp16 *) dma_queue_काढा(dma_q).dst;
                            } अन्यथा {
                                fa_काढा_mask_dma_gqa(dma_q, G);
                                current_mask_vtcm = factx.vtcm_mask_buf;
                            }
                        }

                        // ---- 3. Start HMX O update for step - 1 (reads P[1 - buf_idx], V[1 - buf_idx], D) ----
                        // O update relies on the previous block's P and V tiles.
                        // O update MUST be pushed before the next block's QK-dot: hmx_queue_pop() retires the
                        // oldest descriptor, so push order alone decides which pop waits for which job.
                        // If OU went in after QK(i+1), the pop below would retire QK(i+1) and leave
                        // OU(i-1) in flight into the next iteration, where V-prep overwrites V[prev_buf].
                        जर (step > 0) {
                            स्थिरांक uint32_t prev_blk      = active_blocks[step - 1];
                            स्थिरांक माप_प्रकार prev_buf        = 1 - buf_idx;
                            ou_job[prev_buf].o_curr      = o_tile_curr;
                            ou_job[prev_buf].o_prev      = o_tile_prev;
                            ou_job[prev_buf].p_tiles     = factx.vtcm_p_tiles[prev_buf];
                            ou_job[prev_buf].v_tiles     = factx.vtcm_v_tiles[prev_buf];
                            ou_job[prev_buf].d_tiles     = factx.vtcm_d_tiles[prev_buf];
                            ou_job[prev_buf].hmx_scales  = factx.vtcm_hmx_scales_id;
                            ou_job[prev_buf].n_row_tiles = n_row_tiles;
                            ou_job[prev_buf].n_col_tiles =
                                hmx_ceil_div(hex_smin(Bc, nek1 - prev_blk * Bc), HMX_FP16_TILE_N_COLS);
                            ou_job[prev_buf].n_row_tiles_g_br = n_row_tiles_g_br;
                            ou_job[prev_buf].n_tiles_per_bc   = n_tiles_per_bc;
                            ou_job[prev_buf].DV               = DV_pad;
                            hmx_queue_ढकला(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job[prev_buf]));
                        }

                        // ---- 4. Pop and run K-prep for next active block & push next QK-dot ----
                        जर (step + 1 < n_active_blocks) {
                            स्थिरांक uint32_t next_blk   = active_blocks[step + 1];
                            स्थिरांक uint32_t next_start = next_blk * Bc;
                            स्थिरांक uint32_t next_rows  = hex_smin(Bc, nek1 - next_start);
                            स्थिरांक माप_प्रकार   next_buf   = 1 - buf_idx;

                            शून्य * next_k = (शून्य *) dma_queue_काढा(dma_q).dst;
                            fa_phase_k_interleave(&factx, next_rows, k_src_stride, next_k, next_start, next_buf);

                            qk_job[next_buf].q_tiles        = factx.vtcm_q_tiles;
                            qk_job[next_buf].k_tiles        = factx.vtcm_k_tiles[next_buf];
                            qk_job[next_buf].s_tiles        = factx.vtcm_s_tiles[next_buf];
                            qk_job[next_buf].n_row_tiles    = n_row_tiles;
                            qk_job[next_buf].n_col_tiles    = hmx_ceil_div(next_rows, HMX_FP16_TILE_N_COLS);
                            qk_job[next_buf].n_dot_tiles    = DK_pad / 32;
                            qk_job[next_buf].n_tiles_per_bc = n_tiles_per_bc;
                            qk_job[next_buf].hmx_scales     = factx.vtcm_hmx_scales_qk;
                            hmx_queue_ढकला(hmx_q, hmx_queue_make_desc(hmx_fa_qk_dot_worker, &qk_job[next_buf]));
                        }

                        // ---- 5. Wait for current block's QK-dot to finish ----
                        hmx_queue_काढा(hmx_q);

                        // ---- 6. Phase 2: softmax + build_D ----
                        fa_softmax_args_t sargs;
                        स_रखो(&sargs, 0, आकार(sargs));
                        sargs.factx                = &factx;
                        sargs.buf_idx              = buf_idx;
                        sargs.kv_rows              = kv_rows;
                        sargs.n_rows_g             = n_rows_g;
                        sargs.n_col_tiles          = n_col_tiles;
                        sargs.n_tiles_per_bc       = n_tiles_per_bc;
                        sargs.n_row_tiles          = n_row_tiles;
                        sargs.n_row_tiles_g_br     = n_row_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_head              = kv_head;
                        sargs.kv_start             = kv_start;
                        sargs.q_start              = q_start;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.mask                 = mask;
                        sargs.mask_vtcm            = current_mask_vtcm;
                        sargs.mask_vtcm_row_stride = factx.mask_buf_row_stride;
                        sargs.slopes               = factx.vtcm_slopes;

                        // Run Softmax on HVX (blocking call)
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_row_tiles, n_row_tiles_g_br);

                        // Wait for HMX O update for step - 1 to finish
                        जर (step > 0) {
                            hmx_queue_काढा(hmx_q);
                            hex_swap_ptr((शून्य **) &o_tile_curr, (शून्य **) &o_tile_prev);
                        }

                        // Prefetch block step + 2
                        जर (step + 2 < n_active_blocks) {
                            fa_prefetch_block(dma_q, k, v, mask, active_blocks[step + 2], Bc, size_k_row_padded, size_k_row, size_v_row_padded, size_v_row,
                                              ik2, ik3, iv2, iv3, q_start, im3, kv_head, G, m_ओळ_bytes, n_rows_q, nek1, buf_idx, &factx);
                        }

                        buf_idx = 1 - buf_idx;
                    }

                    // Epilogue
                    जर (n_active_blocks > 0) {
                        स्थिरांक uint32_t last_blk = active_blocks[n_active_blocks - 1];
                        स्थिरांक माप_प्रकार last_cols  = hmx_ceil_div(hex_smin(Bc, nek1 - last_blk * Bc), HMX_FP16_TILE_N_COLS);
                        ou_job[0].o_curr           = o_tile_curr;
                        ou_job[0].o_prev           = o_tile_prev;
                        ou_job[0].p_tiles          = factx.vtcm_p_tiles[1 - buf_idx];
                        ou_job[0].v_tiles          = factx.vtcm_v_tiles[1 - buf_idx];
                        ou_job[0].d_tiles          = factx.vtcm_d_tiles[1 - buf_idx];
                        ou_job[0].hmx_scales       = factx.vtcm_hmx_scales_id;
                        ou_job[0].n_row_tiles      = n_row_tiles;
                        ou_job[0].n_col_tiles      = last_cols;
                        ou_job[0].n_row_tiles_g_br = n_row_tiles_g_br;
                        ou_job[0].n_tiles_per_bc   = n_tiles_per_bc;
                        ou_job[0].DV               = DV_pad;
                        hmx_queue_ढकला(hmx_q, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job[0]));

                        // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                        htp_trace_event_start(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_start);
                        fa_build_d_diag_inv_l(&factx, n_row_tiles, n_row_tiles_g_br, kv_head, n_rows_g);
                        htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_start);
                        hmx_queue_काढा(hmx_q);

                        hex_swap_ptr((शून्य **) &o_tile_curr, (शून्य **) &o_tile_prev);
                    }

                } अन्यथा {
                    // Fallback path
                    साठी (uint32_t step = 0; step < n_active_blocks; ++step) {
                        स्थिरांक uint32_t kv_blk      = active_blocks[step];
                        स्थिरांक uint32_t kv_start    = kv_blk * Bc;
                        स्थिरांक uint32_t kv_rows     = hex_smin(Bc, nek1 - kv_start);
                        स्थिरांक माप_प्रकार   n_col_tiles = hmx_ceil_div(kv_rows, HMX_FP16_TILE_N_COLS);

                        जर (mask) {
                            जर (__builtin_expect(factx.mask_broadcast, true)) {
                                स्थिरांक dma_addr_t ms_src = mask->data + q_start * mask->nb[1] + im3 * mask->nb[3] + kv_start * आकार(__fp16);
                                dma_cache_fa_ढकला(dma_q, &factx.m_cache, ms_src, m_ओळ_bytes, mask->nb[1], kv_rows * आकार(__fp16), n_rows_q);
                            } अन्यथा {
                                fa_ढकला_mask_dma_gqa(dma_q, mask, q_start, im3, kv_start, kv_head, G, m_ओळ_bytes, kv_rows, n_rows_q, &factx);
                            }
                        }

                        जर (step + 1 < n_active_blocks) {
                            स्थिरांक uint32_t  prefetch_blk   = active_blocks[step + 1];
                            स्थिरांक uint32_t  prefetch_start = prefetch_blk * Bc;
                            स्थिरांक uint32_t  prefetch_rows  = hex_smin(Bc, nek1 - prefetch_start);
                            स्थिरांक माप_प्रकार    prefetch_buf   = 1 - buf_idx;
                            स्थिरांक dma_addr_t k_prefetch_src = k->data + prefetch_start * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                            dma_queue_ढकला(dma_q, dma_make_data(factx.vtcm_k_fp16[prefetch_buf], k_prefetch_src), size_k_row_padded, k->nb[1], size_k_row, prefetch_rows);
                            स्थिरांक dma_addr_t v_prefetch_src = v->data + prefetch_start * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                            dma_queue_ढकला(dma_q, dma_make_data(factx.vtcm_v_fp16[prefetch_buf], v_prefetch_src), size_v_row_padded, v->nb[1], size_v_row, prefetch_rows);
                        }

                        // Wait for current K DMA and interleave
                        शून्य * curr_k = (शून्य *) dma_queue_काढा(dma_q).dst;
                        fa_phase_k_interleave(&factx, kv_rows, k_src_stride, curr_k, kv_start, 0);

                        {
                            qk_job.q_tiles        = factx.vtcm_q_tiles;
                            qk_job.k_tiles        = factx.vtcm_k_tiles[0];
                            qk_job.s_tiles        = factx.vtcm_s_tiles[0];
                            qk_job.n_row_tiles    = n_row_tiles;
                            qk_job.n_col_tiles    = n_col_tiles;
                            qk_job.n_dot_tiles    = (माप_प्रकार) (DK_pad / 32);
                            qk_job.n_tiles_per_bc = n_tiles_per_bc;
                            qk_job.hmx_scales     = factx.vtcm_hmx_scales_qk;

                            hmx_queue_ढकला(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_qk_dot_worker, &qk_job));
                            hmx_queue_काढा(ctx->hmx_queue);
                        }

                        // Wait for current V DMA and interleave
                        शून्य * curr_v = (शून्य *) dma_queue_काढा(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_rows, v_src_stride, curr_v, factx.vtcm_v_tiles[0], n_tiles_per_bc, kv_start);

                        // ---- Phase 3: softmax + build_D ----
                        __fp16 * current_mask_vtcm = रिक्त;
                        जर (mask) {
                            जर (__builtin_expect(factx.mask_broadcast, true)) {
                                current_mask_vtcm = (__fp16 *) dma_queue_काढा(dma_q).dst;
                            } अन्यथा {
                                fa_काढा_mask_dma_gqa(dma_q, G);
                                current_mask_vtcm = factx.vtcm_mask_buf;
                            }
                        }

                        fa_softmax_args_t sargs;
                        स_रखो(&sargs, 0, आकार(sargs));
                        sargs.factx                = &factx;
                        sargs.kv_rows              = kv_rows;
                        sargs.n_rows_g             = n_rows_g;
                        sargs.n_col_tiles          = n_col_tiles;
                        sargs.n_tiles_per_bc       = n_tiles_per_bc;
                        sargs.n_row_tiles          = n_row_tiles;
                        sargs.n_row_tiles_g_br     = n_row_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_head              = kv_head;
                        sargs.kv_start             = kv_start;
                        sargs.q_start              = q_start;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.mask                 = mask;
                        sargs.mask_vtcm            = current_mask_vtcm;
                        sargs.mask_vtcm_row_stride = factx.mask_buf_row_stride;
                        sargs.slopes               = factx.vtcm_slopes;
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_row_tiles, n_row_tiles_g_br);

                        {
                            ou_job.o_curr           = o_tile_curr;
                            ou_job.o_prev           = o_tile_prev;
                            ou_job.p_tiles          = factx.vtcm_p_tiles[0];
                            ou_job.v_tiles          = factx.vtcm_v_tiles[0];
                            ou_job.d_tiles          = factx.vtcm_d_tiles[0];
                            ou_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                            ou_job.n_row_tiles      = n_row_tiles;
                            ou_job.n_col_tiles      = n_col_tiles;
                            ou_job.n_row_tiles_g_br = n_row_tiles_g_br;
                            ou_job.n_tiles_per_bc   = n_tiles_per_bc;
                            ou_job.DV               = DV_pad;

                            hmx_queue_ढकला(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_o_update_worker, &ou_job));
                            जर (step + 1 == n_active_blocks) {
                                // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                                htp_trace_event_start(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_start);
                                fa_build_d_diag_inv_l(&factx, n_row_tiles, n_row_tiles_g_br, kv_head, n_rows_g);
                                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_start);
                            }
                            hmx_queue_काढा(ctx->hmx_queue);

                            hex_swap_ptr((शून्य **) &o_tile_curr, (शून्य **) &o_tile_prev);
                        }

                        buf_idx = 1 - buf_idx;
                    }
                }

                // Enqueue DMAs for the next kv_head early so they overlap with O-PROC
                स्थिरांक uint32_t next_kv_head = kv_head + 1;
                स्थिरांक bool has_next = (next_kv_head < kv_head_max);

                जर (has_next) {
                    स्थिरांक dma_addr_t next_q_ptr = q->data + q_start * q->nb[1] + (next_kv_head * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    स्थिरांक माप_प्रकार next_q_row_bytes = q_transposed ? n_rows_q * q_row_bytes_trans_factor : q_row_bytes_untransposed;
                    स्थिरांक माप_प्रकार next_n_rows      = q_transposed ? factx.G : n_rows_q;
                    dma_queue_ढकला(dma_q, dma_make_data(factx.vtcm_q_dma, next_q_ptr), next_q_row_bytes, hex_smax(q_src_stride, next_q_row_bytes), next_q_row_bytes, next_n_rows);

                    जर (n_active_blocks > 0) {
                        स्थिरांक uint32_t next_ik2 = next_kv_head;
                        स्थिरांक uint32_t next_iv2 = next_kv_head;

                        स्थिरांक dma_addr_t next_k_src = k->data + b0_start * k->nb[1] + next_ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_queue_ढकला(dma_q, dma_make_data(factx.vtcm_k_fp16[0], next_k_src), size_k_row_padded, k->nb[1], size_k_row, b0_rows);

                        स्थिरांक dma_addr_t next_v_src = v->data + b0_start * v->nb[1] + next_iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_queue_ढकला(dma_q, dma_make_data(factx.vtcm_v_fp16[0], next_v_src), size_v_row_padded, v->nb[1], size_v_row, b0_rows);

                        जर (factx.pipeline && mask) {
                            जर (__builtin_expect(factx.mask_broadcast, true)) {
                                स्थिरांक dma_addr_t ms_src = mask->data + q_start * mask->nb[1] + im3 * mask->nb[3] + b0_start * आकार(__fp16);
                                dma_cache_fa_ढकला(dma_q, &factx.m_cache, ms_src, m_ओळ_bytes, mask->nb[1], b0_rows * आकार(__fp16), n_rows_q);
                            } अन्यथा {
                                fa_ढकला_mask_dma_gqa(dma_q, mask, q_start, im3, b0_start, next_kv_head, G, m_ओळ_bytes, b0_rows, n_rows_q, &factx);
                            }
                        }
                    }
                }

                // ---- Final normalization ----
                {
                    on_job.o_curr           = o_tile_curr;
                    on_job.o_prev           = o_tile_prev;
                    on_job.d_tiles          = factx.vtcm_d_inv_l;
                    on_job.hmx_scales       = factx.vtcm_hmx_scales_id;
                    on_job.n_row_tiles      = n_row_tiles;
                    on_job.n_row_tiles_g_br = n_row_tiles_g_br;
                    on_job.DV               = DV_pad;
                    hmx_queue_ढकला(ctx->hmx_queue, hmx_queue_make_desc(hmx_fa_o_norm_worker, &on_job));
                    hmx_queue_काढा(ctx->hmx_queue);
                }

                // ---- Store O block ----
                fa_phase_o_store(&factx, dst, o_tile_curr, q_start, kv_head, ib3, n_rows_g);
            }
        }
    }

    परत HTP_STATUS_OK;
}

पूर्णांक op_flash_attn_ext(संरचना htp_ops_context * octx) {
    स्थिरांक संरचना htp_tensor * q    = octx->src[0];
    स्थिरांक संरचना htp_tensor * k    = octx->src[1];
    स्थिरांक संरचना htp_tensor * v    = octx->src[2];
    स्थिरांक संरचना htp_tensor * mask = octx->src[3];
    स्थिरांक संरचना htp_tensor * dst  = octx->dst;

    // Check support
    जर ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) ||
        (k->type != HTP_TYPE_F16 && k->type != HTP_TYPE_Q8_0) ||
        (v->type != HTP_TYPE_F16 && v->type != HTP_TYPE_Q8_0)) {
        परत HTP_STATUS_NO_SUPPORT;
    }

    जर (htp_tensor_is_extended(dst)) {
        परत HTP_STATUS_NO_SUPPORT;
    }

    स्थिरांक संरचना htp_fa_kernel_params * kparams = (स्थिरांक संरचना htp_fa_kernel_params *) octx->kernel_params;

    जर (kparams->kernel_type == HTP_FA_KERNEL_UNSUPPORTED || kparams->n_kv_blocks > FA_MAX_KV_BLOCKS) {
        परत HTP_STATUS_NO_SUPPORT;
    }

    जर (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        परत HTP_STATUS_INVAL_PARAMS;
    }

    जर (kparams->kernel_type == HTP_FA_KERNEL_HMX) {
        परत hmx_flash_attn_ext(octx);
    }

    संरचना htp_fa_context factx;
    factx.octx = octx;
    factx.k = k;
    factx.v = v;

    factx.src0_div21 = kparams->u.hvx.src0_div21;
    factx.src0_div1  = kparams->u.hvx.src0_div1;

    factx.broadcast_rk2 = kparams->broadcast_rk2;
    factx.broadcast_rk3 = kparams->broadcast_rk3;
    factx.broadcast_rv2 = kparams->broadcast_rv2;
    factx.broadcast_rv3 = kparams->broadcast_rv3;

    जर (mask) {
        factx.src3_div2 = kparams->src3_div2;
        factx.src3_div3 = kparams->src3_div3;
    }

    factx.is_q_fp32 = (q->type == HTP_TYPE_F32);
    factx.size_q_row_padded = kparams->u.hvx.size_q_row_padded;
    factx.size_k_row_padded = kparams->u.hvx.size_k_row_padded;
    factx.size_v_row_padded = kparams->u.hvx.size_v_row_padded;

    factx.n_blocks = kparams->n_kv_blocks;

    factx.scale = kparams->scale;
    factx.max_bias = kparams->max_bias;
    factx.has_softcap = (kparams->logit_softcap != 0.0f);
    factx.logit_softcap = 0;
    जर (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_head_log2 = kparams->n_head_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    स्थिरांक uint32_t n_head = q->ne[2];
    जर (n_head > 512) {
        परत HTP_STATUS_NO_SUPPORT;
    }
    साठी (uint32_t h = 0; h < n_head; ++h) {
        factx.slopes[h] = (__fp16) ((kparams->max_bias > 0.0f) ? alibi_slope(h, factx.n_head_log2, factx.m0, factx.m1) : 1.0f);
    }

    // total rows in q
    स्थिरांक uint32_t neq1 = q->ne[1];
    स्थिरांक uint32_t neq2 = q->ne[2];
    स्थिरांक uint32_t neq3 = q->ne[3];
    स्थिरांक uint32_t total_qrows = neq1 * neq2 * neq3;

    uint32_t qrow_start = 0;
    uint32_t qrows      = total_qrows;

    जर (octx->ctx->mdev.मोजणी > 1) {
        स्थिरांक uint32_t mdev_मोजणी = octx->ctx->mdev.मोजणी;
        स्थिरांक uint32_t mdev_idx   = octx->ctx->mdev.idx;
        स्थिरांक uint32_t n_kv_heads = k->ne[2];
        स्थिरांक uint32_t dst_e_size = (dst->type == HTP_TYPE_F32) ? आकार(दशांश) : आकार(__fp16);
        स्थिरांक bool can_split      = htp_tensor_can_row_partition(dst, dst_e_size);

        // head range is contiguous in flat row space only when neq3 == 1
        जर (kparams->head_split && can_split && neq3 == 1 && n_kv_heads >= mdev_मोजणी && n_kv_heads % mdev_मोजणी == 0) {
            स्थिरांक uint32_t G              = kparams->G;
            स्थिरांक uint32_t kv_per_core    = n_kv_heads / mdev_मोजणी;
            स्थिरांक uint32_t heads_per_core = kv_per_core * G;
            स्थिरांक uint32_t head_start     = mdev_idx * heads_per_core;
            qrow_start = head_start * neq1;
            qrows      = heads_per_core * neq1;
        } अन्यथा {
            स्थिरांक संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_qrows, can_split ? 1 : 0, mdev_idx, mdev_मोजणी, &octx->ctx->mdev.मोजणी_div);
            qrow_start = range.start;
            qrows      = range.मोजणी;
        }
    }

    जर (qrows == 0) {
        परत HTP_STATUS_OK;
    }

    स्थिरांक uint32_t n_threads = octx->n_threads;

    factx.qrows            = qrows;
    factx.qrow_start       = qrow_start;
    factx.qrows_per_thread = fastdiv(qrows + n_threads - 1, &octx->n_threads_div);

    स्थिरांक bool has_mask  = (mask != रिक्त);
    स्थिरांक bool has_sinks = (octx->src[4] != रिक्त);
    संरचना hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, k->ne[0], v->ne[0], factx.is_q_fp32, has_mask, has_sinks, n_head, n_threads);

    जर (L.total_bytes > octx->ctx->vtcm_size) {
        परत HTP_STATUS_VTCM_TOO_SMALL;
    }

    factx.size_q_block = L.size_q_block;
    factx.size_k_block = L.size_k_block;
    factx.size_v_block = L.size_v_block;
    factx.size_m_block = L.size_m_block;
    factx.size_vkq_acc = L.size_vkq_acc;

    uint8_t * स्थिरांक base = octx->ctx->vtcm_base;

    factx.spad_q               = VTCM_LAYOUT_PTR(uint8_t, base, L.off_q);
    factx.spad_k               = VTCM_LAYOUT_PTR(uint8_t, base, L.off_k);
    factx.spad_v               = VTCM_LAYOUT_PTR(uint8_t, base, L.off_v);
    factx.spad_m               = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, base, L.off_m, has_mask);
    factx.spad_a               = VTCM_LAYOUT_PTR(uint8_t, base, L.off_a);
    factx.spad_sinks           = VTCM_LAYOUT_PTR_OPTIONAL(दशांश, base, L.off_sinks, has_sinks);
    factx.n_active_blocks      = factx.n_blocks;
    factx.shared_active_blocks = false;

    जर (has_mask && neq1 == 1 && neq3 == 1 && mask->ne[2] == 1 && factx.n_blocks > 1 && factx.n_blocks <= FA_MAX_KV_BLOCKS) {
        factx.n_active_blocks = fa_scan_active_blocks(
            octx->ctx->dma[0], mask, 0, 1, 0, 0, k->ne[1], FLASH_ATTN_BLOCK_SIZE,
            factx.n_blocks, factx.spad_k, factx.size_k_block * 2, factx.active_blocks);
        factx.shared_active_blocks = true;
    }

    जर (has_sinks) {
        स्थिरांक संरचना htp_tensor * sinks = octx->src[4];
        dma_queue * dma_q = octx->ctx->dma[0];
        dma_queue_ढकला(dma_q, dma_make_data(factx.spad_sinks, sinks->data), L.size_sinks, 0, sinks->size, 1);
        dma_queue_काढा(dma_q);
    }

    work_queue_run(octx->ctx->work_queue, flash_attn_ext_f16_thread, &factx, n_threads);

    परत HTP_STATUS_OK;
}
