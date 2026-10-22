#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <HAP_compute_res.h>
#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdatomic.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "hmx-queue.h"
#ಅನ್ತರ್ಭಾವ "hmx-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx-dump.h"
#ಅನ್ತರ್ಭಾವ "hvx-copy.h"
#ಅನ್ತರ್ಭಾವ "hvx-reduce.h"
#ಅನ್ತರ್ಭಾವ "hvx-flash-attn.h"
#ಅನ್ತರ್ಭಾವ "htp-vtcm.h"
#ಅನ್ತರ್ಭಾವ "work-queue.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "hvx-quant.h"

#ಅನ್ತರ್ಭಾವ "flash-attn-ops.h"
#ಅನ್ತರ್ಭಾವ "hvx-fa-kernels.h"
#ಅನ್ತರ್ಭಾವ "hmx-fa-kernels.h"

// Must be multiple of 32
#ವ್ಯಾಖ್ಯಾನಿಸು FLASH_ATTN_ಖಂಡ_ಗಾತ್ರ (32 * 2)

ರಚನೆ htp_fa_context {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ops_context * octx;

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src0_div21;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src0_div1;

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು ಪ್ರಸಾರ_rk2;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು ಪ್ರಸಾರ_rk3;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು ಪ್ರಸಾರ_rv2;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು ಪ್ರಸಾರ_rv3;

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src3_div2;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src3_div3;

    ದಶಮಾಂಶ scale;
    ದಶಮಾಂಶ max_bias;
    bool  has_softcap;
    __fp16 logit_softcap;

    uint32_t n_ತಲೆ_log2;
    ದಶಮಾಂಶ m0;
    ದಶಮಾಂಶ m1;
    __fp16 slopes[512];

    uint32_t n_ಖಂಡಗಳು;
    uint32_t n_ಸಕ್ರಿಯ_ಖಂಡಗಳು;
    bool     shared_ಸಕ್ರಿಯ_ಖಂಡಗಳು;

    size_t ಗಾತ್ರ_q_ಅಡ್ಡಸಾಲು_padded;
    size_t ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded;
    size_t ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded;

    size_t ಗಾತ್ರ_k_ಖಂಡ;
    size_t ಗಾತ್ರ_v_ಖಂಡ;
    size_t ಗಾತ್ರ_m_ಖಂಡ;

    uint32_t qrows;
    uint32_t qrows_ಪ್ರತಿ_thread;
    uint32_t qrow_ಆರಂಭ;

    bool is_q_fp32;

    size_t ಗಾತ್ರ_q_ಖಂಡ;
    size_t ಗಾತ್ರ_vkq_acc;

    uint8_t *  spad_q;
    uint8_t *  spad_k;
    uint8_t *  spad_v;
    uint8_t *  spad_m;
    uint8_t *  spad_a;
    ದಶಮಾಂಶ *    spad_sinks;
    uint16_t   ಸಕ್ರಿಯ_ಖಂಡಗಳು[FA_MAX_KV_ಖಂಡಗಳು];

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * k;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * v;
};

ರಚನೆ hmx_fa_context {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ops_context * octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor *      sinks;  // attention sinks (src[4]), NULL if absent
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor *      k;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor *      v;
    bool         pipeline;  // true when n_kv_blocks >= FA_MIN_KV_BLOCKS && n_threads >= 2
    uint32_t     n_ಎಳೆಗಳು;

    // Op parameters
    __fp16       scale;
    ದಶಮಾಂಶ        max_bias;
    bool         has_softcap;
    __fp16       logit_softcap;
    uint32_t     n_ತಲೆ_log2;
    ದಶಮಾಂಶ        m0, m1;

    // Dimensions
    uint32_t     DK, DV;
    uint32_t     DK_pad, DV_pad;  // head_dim rounded up to 64 for HMX tiling
    uint32_t     n_kv;        // kv_len
    uint32_t     n_kv_heads;  // number of KV heads
    uint32_t     n_heads;     // number of Q heads
    uint32_t     G;           // GQA factor = n_heads / n_kv_heads
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_G;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src3_div2;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು src3_div3;
    uint32_t     n_kv_ಖಂಡಗಳು;
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
    HVX_ಸದಿಶ * vtcm_m_vec;           // Row max [g_br]
    HVX_ಸದಿಶ * vtcm_l_vec;           // Row sum [g_br]
    HVX_ಸದಿಶ * vtcm_s_rowmax;        // Softmax intermediate [g_br]
    HVX_ಸದಿಶ * vtcm_p_rowsum;        // Softmax intermediate [g_br]
    HVX_ಸದಿಶ * vtcm_ಅಡ್ಡಸಾಲು_bufs;        // Per-thread softmax row scratch [n_threads][2][Bc/64]
    uint8_t *    vtcm_hmx_scales_id;   // HMX output scales (identity)
    uint8_t *    vtcm_hmx_scales_qk;   // HMX output scales (qk_scale)
    __fp16 *     vtcm_ಮುಖವಾಡ_buf;        // VTCM mask buffer [Br * m_line], DMA'd per KV block
    __fp16 *     vtcm_slopes;          // ALiBi slopes [g_br]
    ದಶಮಾಂಶ *      vtcm_sinks;           // Attention sinks
    size_t       ಅಡ್ಡಸಾಲು_buf_stride;       // HVX vectors per row buffer (Bc/64)
    size_t       ಮುಖವಾಡ_buf_ಅಡ್ಡಸಾಲು_stride;  // elements (__fp16) per row in mask buffer
    size_t       q_tile_ಬೈಟುಗಳು;
    size_t       k_tile_ಬೈಟುಗಳು;
    size_t       o_tile_ಬೈಟುಗಳು;
    size_t       ಲಂಬಸಾಲು_vec_ಬೈಟುಗಳು;
    size_t       d_tile_ಬೈಟುಗಳು;
    bool         ಮುಖವಾಡ_ಪ್ರಸಾರ;       // true when mask->ne[2] == 1 (head-independent, single 2D DMA)
    dma_cache_fa m_cache;
};

// Scan mask rows [q_start, q_start + n_rows_q) and build compact list of KV blocks with any entry != -inf.
ಸ್ಥಿರ uint32_t fa_ಪರಿಶೀಲನೆ_ಸಕ್ರಿಯ_ಖಂಡಗಳು(
        dma_ಸರತಿ *               dma_q,
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಮುಖವಾಡ,
        uint32_t                  q_ಆರಂಭ,
        uint32_t                  n_ಅಡ್ಡಸಾಲುಗಳು_q,
        uint32_t                  im2,
        uint32_t                  im3,
        uint32_t                  nek1,
        uint32_t                  Bc,
        uint32_t                  n_kv_ಖಂಡಗಳು,
        ಶೂನ್ಯ *                    vtcm_scratch,
        size_t                    scratch_ಬೈಟುಗಳು,
        uint16_t *                ಸಕ್ರಿಯ_ಖಂಡಗಳು) {
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು  = (size_t) nek1 * ಗಾತ್ರ(__fp16);
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_stride = hex_ಸುತ್ತು_ಮೇಲೆ(ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, 128);

    ಆದರೆ (!ಮುಖವಾಡ || n_kv_ಖಂಡಗಳು <= 1 || n_kv_ಖಂಡಗಳು > FA_MAX_KV_ಖಂಡಗಳು ||
        (size_t) n_ಅಡ್ಡಸಾಲುಗಳು_q * ಅಡ್ಡಸಾಲು_stride > scratch_ಬೈಟುಗಳು) {
        #ಸೂಚನೆ clang loop vectorize(disable)
        ಗಾಗಿ (uint32_t b = 0; b < n_kv_ಖಂಡಗಳು; ++b) {
            ಸಕ್ರಿಯ_ಖಂಡಗಳು[b] = (uint16_t) b;
        }
        ಹಿಂತಿರುಗಿ n_kv_ಖಂಡಗಳು;
    }

    ಸ್ಥಿರಾಂಕ dma_addr_t m_src = ಮುಖವಾಡ->ದತ್ತಾಂಶ + q_ಆರಂಭ * ಮುಖವಾಡ->nb[1] + im2 * ಮುಖವಾಡ->nb[2] + im3 * ಮುಖವಾಡ->nb[3];
    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_scratch, m_src), ಅಡ್ಡಸಾಲು_stride, ಮುಖವಾಡ->nb[1], ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, n_ಅಡ್ಡಸಾಲುಗಳು_q);
    ಸ್ಥಿರಾಂಕ uint8_t * m_buf = (ಸ್ಥಿರಾಂಕ uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_ninf   = Q6_Vh_vsplat_R(0xFC00);
    uint32_t         n_ಸಕ್ರಿಯ = 0;

    ಗಾಗಿ (uint32_t b = 0; b < n_kv_ಖಂಡಗಳು; ++b) {
        ಸ್ಥಿರಾಂಕ uint32_t col0 = b * Bc;
        ಸ್ಥಿರಾಂಕ uint32_t ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 = MIN(Bc, nek1 - col0);

        ಗಾಗಿ (uint32_t r = 0; r < n_ಅಡ್ಡಸಾಲುಗಳು_q; ++r) {
            ಸ್ಥಿರಾಂಕ __fp16 * ಅಡ್ಡಸಾಲು_ptr = (ಸ್ಥಿರಾಂಕ __fp16 *) (m_buf + r * ಅಡ್ಡಸಾಲು_stride) + col0;
            HVX_ಸದಿಶ v_any = Q6_V_vzero();
            uint32_t c = 0;
            ಗಾಗಿ (; c + 64 <= ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2; c += 64) {
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_m = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ಅಡ್ಡಸಾಲು_ptr + c);
                v_any = Q6_V_vor_VV(v_any, Q6_V_vxor_VV(v_m, v_ninf));
            }
            ಆದರೆ (c < ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2) {
                ಸ್ಥಿರಾಂಕ uint32_t rem = ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2 - c;
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred q_rem = Q6_Q_vsetq2_R(rem * ಗಾತ್ರ(__fp16));
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_m = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ಅಡ್ಡಸಾಲು_ptr + c);
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_diff = Q6_V_vand_QV(q_rem, Q6_V_vxor_VV(v_m, v_ninf));
                v_any = Q6_V_vor_VV(v_any, v_diff);
            }
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 64));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 32));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 16));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 8));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 4));
            ಆದರೆ (Q6_R_vextract_VR(v_any, 0) != 0) {
                ಸಕ್ರಿಯ_ಖಂಡಗಳು[n_ಸಕ್ರಿಯ++] = (uint16_t) b;
                ನಿಲ್ಲಿಸು;
            }
        }
    }

    ಆದರೆ (n_ಸಕ್ರಿಯ == 0 && n_kv_ಖಂಡಗಳು > 0) {
        ಸಕ್ರಿಯ_ಖಂಡಗಳು[0] = 0;
        n_ಸಕ್ರಿಯ = 1;
    }
    ಹಿಂತಿರುಗಿ n_ಸಕ್ರಿಯ;
}

ಸ್ಥಿರ ಶೂನ್ಯ flash_attn_ext_f16_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_fa_context * factx = (ರಚನೆ htp_fa_context *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ops_context * octx = factx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * q     = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * k     = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * v     = octx->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಮುಖವಾಡ  = octx->src[3];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * sinks = octx->src[4];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst   = octx->dst;

    ಸ್ಥಿರಾಂಕ uint32_t neq0 = q->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t neq1 = q->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t neq2 = q->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t neq3 = q->ne[3];

    ಸ್ಥಿರಾಂಕ uint32_t nek0 = k->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t nek1 = k->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t nek2 = k->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t nek3 = k->ne[3];

    ಸ್ಥಿರಾಂಕ uint32_t nev0 = v->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t nev1 = v->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t nev2 = v->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t nev3 = v->ne[3];

    ಸ್ಥಿರಾಂಕ uint32_t nbq1 = q->nb[1];
    ಸ್ಥಿರಾಂಕ uint32_t nbq2 = q->nb[2];
    ಸ್ಥಿರಾಂಕ uint32_t nbq3 = q->nb[3];

    ಸ್ಥಿರಾಂಕ uint32_t nbk1 = k->nb[1];
    ಸ್ಥಿರಾಂಕ uint32_t nbk2 = k->nb[2];
    ಸ್ಥಿರಾಂಕ uint32_t nbk3 = k->nb[3];

    ಸ್ಥಿರಾಂಕ uint32_t nbv1 = v->nb[1];
    ಸ್ಥಿರಾಂಕ uint32_t nbv2 = v->nb[2];
    ಸ್ಥಿರಾಂಕ uint32_t nbv3 = v->nb[3];

    ಸ್ಥಿರಾಂಕ uint32_t ne1 = dst->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t ne2 = dst->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t ne3 = dst->ne[3];

    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1];
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = dst->nb[2];
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = dst->nb[3];

    // total rows in q
    ಸ್ಥಿರಾಂಕ uint32_t dr  = factx->qrows_ಪ್ರತಿ_thread;
    ಸ್ಥಿರಾಂಕ uint32_t ir0 = factx->qrow_ಆರಂಭ + dr * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ir1 = MIN(ir0 + dr, factx->qrow_ಆರಂಭ + factx->qrows);

    ಆದರೆ (ir0 >= ir1) ಹಿಂತಿರುಗಿ;

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];

    ಸ್ಥಿರಾಂಕ uint32_t DK = nek0;
    ಸ್ಥಿರಾಂಕ uint32_t DV = nev0;

    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_q_ಅಡ್ಡಸಾಲು = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು = htp_tensor_get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(k->type, DK);
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು = htp_tensor_get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(v->type, DV);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = factx->spad_q + factx->ಗಾತ್ರ_q_ಖಂಡ * ith;
    uint8_t * spad_k = factx->spad_k + factx->ಗಾತ್ರ_k_ಖಂಡ * 2 * ith;
    uint8_t * spad_v = factx->spad_v + factx->ಗಾತ್ರ_v_ಖಂಡ * 2 * ith;
    uint8_t * spad_m = factx->spad_m + (ಮುಖವಾಡ ? factx->ಗಾತ್ರ_m_ಖಂಡ * HVX_FA_DMA_CACHE_ಗಾತ್ರ : 0) * ith;
    uint8_t * spad_a = factx->spad_a + factx->ಗಾತ್ರ_vkq_acc * ith;

    dma_cache_dm m_cache;
    dma_cache_dm_init(&m_cache, spad_m, factx->ಗಾತ್ರ_m_ಖಂಡ, HVX_FA_DMA_CACHE_ಗಾತ್ರ);

    uint16_t thread_ಸಕ್ರಿಯ_ಖಂಡಗಳು[FA_MAX_KV_ಖಂಡಗಳು];
    ಸ್ಥಿರಾಂಕ uint16_t * ಸಕ್ರಿಯ_ಖಂಡಗಳು = factx->shared_ಸಕ್ರಿಯ_ಖಂಡಗಳು ? factx->ಸಕ್ರಿಯ_ಖಂಡಗಳು : thread_ಸಕ್ರಿಯ_ಖಂಡಗಳು;
    uint32_t n_ಸಕ್ರಿಯ_ಖಂಡಗಳು       = factx->n_ಸಕ್ರಿಯ_ಖಂಡಗಳು;
    dma_addr_t prev_mp_ತಳ        = 0;

    ಆದರೆ (!factx->shared_ಸಕ್ರಿಯ_ಖಂಡಗಳು && !ಮುಖವಾಡ) {
        #ಸೂಚನೆ clang loop vectorize(disable)
        ಗಾಗಿ (uint32_t b = 0; b < factx->n_ಖಂಡಗಳು; ++b) {
            thread_ಸಕ್ರಿಯ_ಖಂಡಗಳು[b] = (uint16_t) b;
        }
    }

    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_vkq_acc_single = hex_ಸುತ್ತು_ಮೇಲೆ(DV * ಗಾತ್ರ(ದಶಮಾಂಶ), 128);

    uint32_t ir = ir0;
    ಆಗ (ir < ir1) {
        ಸ್ಥಿರಾಂಕ uint32_t iq3 = fastdiv(ir, &factx->src0_div21);
        ಸ್ಥಿರಾಂಕ uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &factx->src0_div1);
        ಸ್ಥಿರಾಂಕ uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        ಸ್ಥಿರಾಂಕ uint32_t ik3 = fastdiv(iq3, &factx->ಪ್ರಸಾರ_rk3);
        ಸ್ಥಿರಾಂಕ uint32_t ik2 = fastdiv(iq2, &factx->ಪ್ರಸಾರ_rk2);

        ಸ್ಥಿರಾಂಕ uint32_t iv3 = fastdiv(iq3, &factx->ಪ್ರಸಾರ_rv3);
        ಸ್ಥಿರಾಂಕ uint32_t iv2 = fastdiv(iq2, &factx->ಪ್ರಸಾರ_rv2);

        uint32_t G_local = 1;
        ಆದರೆ (neq1 == 1 && (ಮುಖವಾಡ == ಖಾಲಿ || ಮುಖವಾಡ->ne[2] == 1)) {
            ಆಗ (ir + G_local < ir1 && G_local < FA_HVX_G_MAX) {
                ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ir = ir + G_local;
                ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_iq3 = fastdiv(ಮುಂದಿನ_ir, &factx->src0_div21);
                ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_iq2 = fastdiv(ಮುಂದಿನ_ir - ಮುಂದಿನ_iq3*neq2*neq1, &factx->src0_div1);
                ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_iq1 = (ಮುಂದಿನ_ir - ಮುಂದಿನ_iq3*neq2*neq1 - ಮುಂದಿನ_iq2 * neq1);

                ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ik3 = fastdiv(ಮುಂದಿನ_iq3, &factx->ಪ್ರಸಾರ_rk3);
                ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ik2 = fastdiv(ಮುಂದಿನ_iq2, &factx->ಪ್ರಸಾರ_rk2);

                ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_iv3 = fastdiv(ಮುಂದಿನ_iq3, &factx->ಪ್ರಸಾರ_rv3);
                ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_iv2 = fastdiv(ಮುಂದಿನ_iq2, &factx->ಪ್ರಸಾರ_rv2);

                ಆದರೆ (ಮುಂದಿನ_ik2 != ik2 || ಮುಂದಿನ_ik3 != ik3 || ಮುಂದಿನ_iv2 != iv2 || ಮುಂದಿನ_iv3 != iv3 || ಮುಂದಿನ_iq1 != iq1 || ಮುಂದಿನ_iq3 != iq3) {
                    ನಿಲ್ಲಿಸು;
                }
                G_local++;
            }
        }

        dma_addr_t mp_ತಳ = 0;
        ಆದರೆ (ಮುಖವಾಡ) {
            ಸ್ಥಿರಾಂಕ uint32_t im2 = fastmodulo(iq2, ಮುಖವಾಡ->ne[2], &factx->src3_div2);
            ಸ್ಥಿರಾಂಕ uint32_t im3 = fastmodulo(iq3, ಮುಖವಾಡ->ne[3], &factx->src3_div3);
            mp_ತಳ = ಮುಖವಾಡ->ದತ್ತಾಂಶ + iq1*ಮುಖವಾಡ->nb[1] + im2*ಮುಖವಾಡ->nb[2] + im3*ಮುಖವಾಡ->nb[3];
            ಆದರೆ (!factx->shared_ಸಕ್ರಿಯ_ಖಂಡಗಳು && mp_ತಳ != prev_mp_ತಳ) {
                n_ಸಕ್ರಿಯ_ಖಂಡಗಳು = fa_ಪರಿಶೀಲನೆ_ಸಕ್ರಿಯ_ಖಂಡಗಳು(
                    dma_q, ಮುಖವಾಡ, iq1, 1, im2, im3, nek1, FLASH_ATTN_ಖಂಡ_ಗಾತ್ರ,
                    factx->n_ಖಂಡಗಳು, spad_k, factx->ಗಾತ್ರ_k_ಖಂಡ * 2, thread_ಸಕ್ರಿಯ_ಖಂಡಗಳು);
                prev_mp_ತಳ = mp_ತಳ;
            }
        }

        uint32_t heads[FA_HVX_G_MAX];
        HVX_ಸದಿಶ ಇಳಿಜಾರು_vecs[FA_HVX_G_MAX] __attribute__((aligned(128)));
        HVX_ಸದಿಶ S_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        HVX_ಸದಿಶ M_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        uint8_t * q_ptrs[FA_HVX_G_MAX];
        ದಶಮಾಂಶ * vkq_ptrs[FA_HVX_G_MAX];

        ಗಾಗಿ (uint32_t g = 0; g < G_local; ++g) {
            ಸ್ಥಿರಾಂಕ uint32_t r = ir + g;
            ಸ್ಥಿರಾಂಕ uint32_t r_iq3 = fastdiv(r, &factx->src0_div21);
            ಸ್ಥಿರಾಂಕ uint32_t r_iq2 = fastdiv(r - r_iq3*neq2*neq1, &factx->src0_div1);
            ಸ್ಥಿರಾಂಕ uint32_t r_iq1 = (r - r_iq3*neq2*neq1 - r_iq2 * neq1);

            heads[g] = r_iq2;
            ಸ್ಥಿರಾಂಕ __fp16 ಇಳಿಜಾರು = factx->slopes[r_iq2];
            ಇಳಿಜಾರು_vecs[g] = hvx_vec_splat_f16(ಇಳಿಜಾರು);

            S_vec[g] = hvx_vec_splat_f32(0.0f);
            M_vec[g] = hvx_vec_splat_f32(HTP_FA_M_INITIAL_VAL);

            uint8_t * q_dst = spad_q + g * factx->ಗಾತ್ರ_q_ಅಡ್ಡಸಾಲು_padded;
            q_ptrs[g] = q_dst;

            ದಶಮಾಂಶ * vkq_dst = (ದಶಮಾಂಶ *)(spad_a + g * ಗಾತ್ರ_vkq_acc_single);
            vkq_ptrs[g] = vkq_dst;
            hvx_splat_f32_a((uint8_t *) vkq_dst, 0, DV);

            // Fetch Q row g
            ಸ್ಥಿರಾಂಕ dma_addr_t q_ಅಡ್ಡಸಾಲು_ptr = q->ದತ್ತಾಂಶ + r_iq1*nbq1 + r_iq2*nbq2 + r_iq3*nbq3;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(q_dst, q_ಅಡ್ಡಸಾಲು_ptr), factx->ಗಾತ್ರ_q_ಅಡ್ಡಸಾಲು_padded, nbq1, ಗಾತ್ರ_q_ಅಡ್ಡಸಾಲು, 1);
        }

        // Prefetch first two active blocks
        ಗಾಗಿ (uint32_t ಹೆಜ್ಜೆ = 0; ಹೆಜ್ಜೆ < MIN(n_ಸಕ್ರಿಯ_ಖಂಡಗಳು, 2); ++ಹೆಜ್ಜೆ) {
            ಸ್ಥಿರಾಂಕ uint32_t ib = ಸಕ್ರಿಯ_ಖಂಡಗಳು[ಹೆಜ್ಜೆ];
            ಸ್ಥಿರಾಂಕ uint32_t ic_ಆರಂಭ = ib * FLASH_ATTN_ಖಂಡ_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ uint32_t current_ಖಂಡ_ಗಾತ್ರ = MIN(FLASH_ATTN_ಖಂಡ_ಗಾತ್ರ, nek1 - ic_ಆರಂಭ);

            // K
            ಸ್ಥಿರಾಂಕ dma_addr_t k_src = k->ದತ್ತಾಂಶ + ic_ಆರಂಭ*nbk1 + ik2*nbk2 + ik3*nbk3;
            uint8_t * k_dst = spad_k + (ಹೆಜ್ಜೆ % 2) * factx->ಗಾತ್ರ_k_ಖಂಡ;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(k_dst, k_src), factx->ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded, nbk1, ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು, current_ಖಂಡ_ಗಾತ್ರ);

            // V
            ಸ್ಥಿರಾಂಕ dma_addr_t v_src = v->ದತ್ತಾಂಶ + ic_ಆರಂಭ*nbv1 + iv2*nbv2 + iv3*nbv3;
            uint8_t * v_dst = spad_v + (ಹೆಜ್ಜೆ % 2) * factx->ಗಾತ್ರ_v_ಖಂಡ;
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(v_dst, v_src), factx->ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded, nbv1, ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು, current_ಖಂಡ_ಗಾತ್ರ);

            // Mask
            ಆದರೆ (ಮುಖವಾಡ) {
                ಸ್ಥಿರಾಂಕ dma_addr_t m_src = mp_ತಳ + ic_ಆರಂಭ * ಗಾತ್ರ(__fp16);
                dma_cache_dm_ತಳ್ಳು(dma_q, &m_cache, ib, m_src, current_ಖಂಡ_ಗಾತ್ರ * 2, current_ಖಂಡ_ಗಾತ್ರ * 2, current_ಖಂಡ_ಗಾತ್ರ * 2, 1);
            }
        }

        // Pop all Q rows
        ಗಾಗಿ (uint32_t g = 0; g < G_local; ++g) {
            uint8_t * q_ptr_vtcm = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
            ಆದರೆ (factx->is_q_fp32) {
                hvx_copy_f16_f32_aa(q_ptr_vtcm, q_ptr_vtcm, DK);
            }
        }

        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
        ಸ್ಥಿರಾಂಕ bool has_softcap     = factx->has_softcap;
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_cap     = has_softcap ? hvx_vec_splat_f16(factx->logit_softcap) : Q6_V_vzero();
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vinf      = Q6_Vh_vsplat_R(0xFC00);
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vmin      = Q6_Vh_vsplat_R(0xFBFF);
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_log2e   = hvx_vec_splat_f16(EXP_LOG2E_F);
        ಸ್ಥಿರಾಂಕ uint32_t stride_v2   = factx->ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded * 2;

        ಗಾಗಿ (uint32_t ಹೆಜ್ಜೆ = 0; ಹೆಜ್ಜೆ < n_ಸಕ್ರಿಯ_ಖಂಡಗಳು; ++ಹೆಜ್ಜೆ) {
            ಸ್ಥಿರಾಂಕ uint32_t ib = ಸಕ್ರಿಯ_ಖಂಡಗಳು[ಹೆಜ್ಜೆ];
            ಸ್ಥಿರಾಂಕ uint32_t ic_ಆರಂಭ = ib * FLASH_ATTN_ಖಂಡ_ಗಾತ್ರ;
            ಸ್ಥಿರಾಂಕ uint32_t current_ಖಂಡ_ಗಾತ್ರ = MIN(FLASH_ATTN_ಖಂಡ_ಗಾತ್ರ, nek1 - ic_ಆರಂಭ);

            // Wait for DMA
            uint8_t * k_ತಳ = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst; // K
            uint8_t * v_ತಳ = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst; // V
            __fp16  * m_ತಳ = ಮುಖವಾಡ ? (__fp16 *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst : ಖಾಲಿ; // M

            ಆದರೆ (factx->k->type == HTP_TYPE_Q8_0) {
                htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
                ಗಾಗಿ (uint32_t r = 0; r < current_ಖಂಡ_ಗಾತ್ರ; ++r) {
                    __fp16 * ಅಡ್ಡಸಾಲು_k = (__fp16 *)(k_ತಳ + r * factx->ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded);
                    hvx_dequantize_ಅಡ್ಡಸಾಲು_flat_q8_0_f16(ಅಡ್ಡಸಾಲು_k, ಅಡ್ಡಸಾಲು_k, DK);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
            }
            ಆದರೆ (factx->v->type == HTP_TYPE_Q8_0) {
                htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
                ಗಾಗಿ (uint32_t r = 0; r < current_ಖಂಡ_ಗಾತ್ರ; ++r) {
                    __fp16 * ಅಡ್ಡಸಾಲು_v = (__fp16 *)(v_ತಳ + r * factx->ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded);
                    hvx_dequantize_ಅಡ್ಡಸಾಲು_flat_q8_0_f16(ಅಡ್ಡಸಾಲು_v, ಅಡ್ಡಸಾಲು_v, DV);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
            }

            ಗಾಗಿ (uint32_t g = 0; g < G_local; ++g) {
                ಸ್ಥಿರಾಂಕ uint32_t ತಲೆ_ir = ir + g;
                uint8_t * q_ptr_vtcm = q_ptrs[g];
                ದಶಮಾಂಶ * VKQ32 = vkq_ptrs[g];
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ಇಳಿಜಾರು_vec = ಇಳಿಜಾರು_vecs[g];

                htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_FA_QK, ತಲೆ_ir);

                HVX_ಸದಿಶ scores_f16 = Q6_V_vzero();
                ಆದರೆ (current_ಖಂಡ_ಗಾತ್ರ > 0) {
                    HVX_ಸದಿಶ scores0 = hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_ತಳ, factx->ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded, DK, factx->scale);
                    HVX_ಸದಿಶ scores1 = (current_ಖಂಡ_ಗಾತ್ರ > 32) ? hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_ತಳ + 32 * factx->ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded, factx->ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded, DK, factx->scale) : Q6_V_vzero();
                    scores_f16 = hvx_vec_f32_ಗಮ್ಯ_f16(scores0, scores1);
                }

                ಆದರೆ (has_softcap) {
                    scores_f16 = hvx_vec_tanh_f16(scores_f16);
                    scores_f16 = hvx_vec_mul_f16_f16(scores_f16, v_cap);
                }

                HVX_ಸದಿಶ_Pred q_ಬಾಲ_keep = Q6_Q_vsetq2_R(current_ಖಂಡ_ಗಾತ್ರ * ಗಾತ್ರ(__fp16));

                ಆದರೆ (ಮುಖವಾಡ) {
                    HVX_ಸದಿಶ m_vals_f16 = *(ಸ್ಥಿರಾಂಕ HVX_UVector *) m_ತಳ;
                    HVX_ಸದಿಶ_Pred is_inf = Q6_Q_vcmp_eq_VhVh(m_vals_f16, vinf);
                    m_vals_f16 = Q6_V_vmux_QVV(is_inf, vmin, m_vals_f16);

                    HVX_ಸದಿಶ m_scaled = hvx_vec_mul_f16_f16(m_vals_f16, ಇಳಿಜಾರು_vec);
                    scores_f16 = Q6_V_vmux_QVV(q_ಬಾಲ_keep, hvx_vec_ಕೂಡಿಸು_f16_f16(scores_f16, m_scaled), v_neg_inf);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    scores_f16 = Q6_V_vmux_QVV(q_ಬಾಲ_keep, scores_f16, v_neg_inf);
                }

                HVX_ಸದಿಶ v_max_f16 = hvx_vec_reduce_max_f16(scores_f16);
                HVX_ಸದಿಶ v_max     = Q6_V_lo_W(hvx_vec_f16_ಗಮ್ಯ_f32(v_max_f16));
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_QK, ತಲೆ_ir);

                // prefetch K for step + 2 after last head finished QK
                ಆದರೆ (g + 1 == G_local && ಹೆಜ್ಜೆ + 2 < n_ಸಕ್ರಿಯ_ಖಂಡಗಳು) {
                    ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ib = ಸಕ್ರಿಯ_ಖಂಡಗಳು[ಹೆಜ್ಜೆ + 2];
                    ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ic_ಆರಂಭ = ಮುಂದಿನ_ib * FLASH_ATTN_ಖಂಡ_ಗಾತ್ರ;
                    ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ = MIN(FLASH_ATTN_ಖಂಡ_ಗಾತ್ರ, nek1 - ಮುಂದಿನ_ic_ಆರಂಭ);

                    ಸ್ಥಿರಾಂಕ dma_addr_t k_src = k->ದತ್ತಾಂಶ + ಮುಂದಿನ_ic_ಆರಂಭ*nbk1 + ik2*nbk2 + ik3*nbk3;
                    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(k_ತಳ, k_src), factx->ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded, nbk1, ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು, ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ);
                }

                htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_FA_SFM, ತಲೆ_ir);
                {
                    HVX_ಸದಿಶ M_new_vec = Q6_Vsf_vmax_VsfVsf(v_max, M_vec[g]);
                    HVX_ಸದಿಶ diff_vec  = HVX_OP_SUB_F32(M_vec[g], M_new_vec);

                    HVX_ಸದಿಶ diff_f16   = hvx_vec_f32_ಗಮ್ಯ_f16(diff_vec, diff_vec);
                    HVX_ಸದಿಶ diff_base2 = hvx_vec_mul_f16_f16(diff_f16, v_log2e);
                    HVX_ಸದಿಶ ms_f16     = hvx_vec_exp2_f16(diff_base2);
                    HVX_ಸದಿಶ ms_vec     = Q6_V_lo_W(hvx_vec_f16_ಗಮ್ಯ_f32(ms_f16));

                    M_vec[g] = M_new_vec;

                    HVX_ಸದಿಶ v_m_vec_f16 = hvx_vec_f32_ಗಮ್ಯ_f16(M_vec[g], M_vec[g]);
                    HVX_ಸದಿಶ v_s_minus_m = Q6_Vqf16_vsub_VhfVhf(scores_f16, v_m_vec_f16);
                    HVX_ಸದಿಶ v_s_minus_m_base2 = hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m), v_log2e);

                    HVX_ಸದಿಶ P = hvx_vec_exp2_f16(v_s_minus_m_base2);
                    P = Q6_V_vmux_QVV(q_ಬಾಲ_keep, P, Q6_V_vzero());

                    HVX_ಸದಿಶ_Pair P_pair = hvx_vec_f16_ಗಮ್ಯ_f32(P);
                    HVX_ಸದಿಶ P0 = Q6_V_lo_W(P_pair);
                    HVX_ಸದಿಶ P1 = Q6_V_hi_W(P_pair);
                    HVX_ಸದಿಶ p_ಮೊತ್ತ_vec = hvx_vec_reduce_ಮೊತ್ತ_f32(HVX_OP_ಕೂಡಿಸು_F32(P0, P1));

                    S_vec[g] = HVX_OP_ಕೂಡಿಸು_F32(HVX_OP_MUL_F32(S_vec[g], ms_vec), p_ಮೊತ್ತ_vec);

                    ಸ್ಥಿರಾಂಕ uint8_t * v_ptr = v_ತಳ;

                    ಆದರೆ (DV == 64) {
                        HVX_ಸದಿಶ_Pair vkq0 = *((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pair *) VKQ32);
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );

                        ಗಾಗಿ (uint32_t j = 0; j < current_ಖಂಡ_ಗಾತ್ರ; j += 2) {
                            HVX_ಸದಿಶ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * vx0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) v_ptr;
                            ಆದರೆ (j + 1 == current_ಖಂಡ_ಗಾತ್ರ) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                ನಿಲ್ಲಿಸು;
                            }

                            HVX_ಸದಿಶ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * vx1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (v_ptr + factx->ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            v_ptr += stride_v2;
                        }

                        *((HVX_ಸದಿಶ_Pair *) VKQ32) = vkq0;
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (DV == 128) {
                        HVX_ಸದಿಶ_Pair vkq0 = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pair *) VKQ32)[0];
                        HVX_ಸದಿಶ_Pair vkq1 = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pair *) VKQ32)[1];
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );
                        vkq1 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq1), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq1), ms_vec)
                        );

                        ಗಾಗಿ (uint32_t j = 0; j < current_ಖಂಡ_ಗಾತ್ರ; j += 2) {
                            HVX_ಸದಿಶ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * vx0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) v_ptr;
                            ಆದರೆ (j + 1 == current_ಖಂಡ_ಗಾತ್ರ) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                ನಿಲ್ಲಿಸು;
                            }

                            HVX_ಸದಿಶ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * vx1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (v_ptr + factx->ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx1[1]), S1);
                            v_ptr += stride_v2;
                        }

                        ((HVX_ಸದಿಶ_Pair *) VKQ32)[0] = vkq0;
                        ((HVX_ಸದಿಶ_Pair *) VKQ32)[1] = vkq1;
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (DV == 256) {
                        HVX_ಸದಿಶ_Pair vkq0 = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pair *) VKQ32)[0];
                        HVX_ಸದಿಶ_Pair vkq1 = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pair *) VKQ32)[1];
                        HVX_ಸದಿಶ_Pair vkq2 = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pair *) VKQ32)[2];
                        HVX_ಸದಿಶ_Pair vkq3 = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pair *) VKQ32)[3];
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

                        ಗಾಗಿ (uint32_t j = 0; j < current_ಖಂಡ_ಗಾತ್ರ; j += 2) {
                            HVX_ಸದಿಶ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * vx0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) v_ptr;
                            ಆದರೆ (j + 1 == current_ಖಂಡ_ಗಾತ್ರ) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                vkq2 = hvx_vec_mpyacc_f32_f16(vkq2, Q6_Vh_vshuff_Vh(vx0[2]), S0);
                                vkq3 = hvx_vec_mpyacc_f32_f16(vkq3, Q6_Vh_vshuff_Vh(vx0[3]), S0);
                                ನಿಲ್ಲಿಸು;
                            }

                            HVX_ಸದಿಶ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * vx1 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (v_ptr + factx->ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded);
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

                        ((HVX_ಸದಿಶ_Pair *) VKQ32)[0] = vkq0;
                        ((HVX_ಸದಿಶ_Pair *) VKQ32)[1] = vkq1;
                        ((HVX_ಸದಿಶ_Pair *) VKQ32)[2] = vkq2;
                        ((HVX_ಸದಿಶ_Pair *) VKQ32)[3] = vkq3;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (ಸ್ಥಿರಾಂಕ uint8_t *) VKQ32, DV, ms_vec);

                        ಗಾಗಿ (uint32_t j = 0; j < current_ಖಂಡ_ಗಾತ್ರ; j += 2) {
                            ಆದರೆ (j + 1 == current_ಖಂಡ_ಗಾತ್ರ) {
                                HVX_ಸದಿಶ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                                hvx_mad_f32_f16_aa_vec(VKQ32, v_ptr, S0, DV);
                                ನಿಲ್ಲಿಸು;
                            }

                            HVX_ಸದಿಶ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            HVX_ಸದಿಶ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));

                            hvx_mad_f32_f16_aa_rx2_vec(VKQ32, v_ptr, v_ptr + factx->ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded, S0, S1, DV);
                            v_ptr += stride_v2;
                        }
                    }
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, ತಲೆ_ir);

                // prefetch V and mask for step + 2 after last head finished V accumulation
                ಆದರೆ (g + 1 == G_local && ಹೆಜ್ಜೆ + 2 < n_ಸಕ್ರಿಯ_ಖಂಡಗಳು) {
                    ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ib = ಸಕ್ರಿಯ_ಖಂಡಗಳು[ಹೆಜ್ಜೆ + 2];
                    ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ic_ಆರಂಭ = ಮುಂದಿನ_ib * FLASH_ATTN_ಖಂಡ_ಗಾತ್ರ;
                    ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ = MIN(FLASH_ATTN_ಖಂಡ_ಗಾತ್ರ, nek1 - ಮುಂದಿನ_ic_ಆರಂಭ);

                    // V
                    ಸ್ಥಿರಾಂಕ dma_addr_t v_src = v->ದತ್ತಾಂಶ + ಮುಂದಿನ_ic_ಆರಂಭ*nbv1 + iv2*nbv2 + iv3*nbv3;
                    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(v_ತಳ, v_src), factx->ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded, nbv1, ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು, ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ);

                    // Mask
                    ಆದರೆ (ಮುಖವಾಡ) {
                        ಸ್ಥಿರಾಂಕ dma_addr_t m_src = mp_ತಳ + ಮುಂದಿನ_ic_ಆರಂಭ * ಗಾತ್ರ(__fp16);
                        dma_cache_dm_ತಳ್ಳು(dma_q, &m_cache, ಮುಂದಿನ_ib, m_src, ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ * 2, ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ * 2, ಮುಂದಿನ_ಖಂಡ_ಗಾತ್ರ * 2, 1);
                    }
                }
            } // end for g
        } // end for step

        ಗಾಗಿ (uint32_t g = 0; g < G_local; ++g) {
            ಸ್ಥಿರಾಂಕ uint32_t ತಲೆ_ir = ir + g;
            ಸ್ಥಿರಾಂಕ uint32_t h = heads[g];
            ದಶಮಾಂಶ * VKQ32 = vkq_ptrs[g];

            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_O_PROC, ತಲೆ_ir);

            ದಶಮಾಂಶ M = hvx_vec_get_f32(M_vec[g]);
            ದಶಮಾಂಶ S = hvx_vec_get_f32(S_vec[g]);

            ಆದರೆ (sinks) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ s = factx->spad_sinks[h];

                ದಶಮಾಂಶ vs = 1.0f;

                ಆದರೆ (s > M) {
                    HVX_ಸದಿಶ diff_vec = hvx_vec_splat_f32(M - s);
                    HVX_ಸದಿಶ ms_vec   = hvx_vec_exp_f32(diff_vec);
                    hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (ಸ್ಥಿರಾಂಕ uint8_t *) VKQ32, DV, ms_vec);

                    ದಶಮಾಂಶ ms = hvx_vec_get_f32(ms_vec);
                    S = S * ms + vs;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    HVX_ಸದಿಶ diff_vec = hvx_vec_splat_f32(s - M);
                    vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                    S += vs;
                }
            }

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            hvx_scale_f32_aa((uint8_t *) VKQ32, (ಸ್ಥಿರಾಂಕ uint8_t *) VKQ32, DV, S_inv);

            ಸ್ಥಿರಾಂಕ uint32_t r_iq3 = fastdiv(ತಲೆ_ir, &factx->src0_div21);
            ಸ್ಥಿರಾಂಕ uint32_t r_iq2 = fastdiv(ತಲೆ_ir - r_iq3*neq2*neq1, &factx->src0_div1);
            ಸ್ಥಿರಾಂಕ uint32_t r_iq1 = (ತಲೆ_ir - r_iq3*neq2*neq1 - r_iq2 * neq1);

            uint8_t * dst_ptr = (uint8_t *) dst->ದತ್ತಾಂಶ + r_iq2 * dst->nb[1] + r_iq1 * dst->nb[2] + r_iq3 * dst->nb[3];

            ಆದರೆ (dst->type == HTP_TYPE_F32) {
                hvx_copy_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (dst->type == HTP_TYPE_F16) {
                hvx_copy_f16_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, ತಲೆ_ir);
        }

        ir += G_local;
    }
}

// ============================================================================
// HMX Phase args and thread logic
// ============================================================================

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ರಚನೆ hmx_fa_context * factx;
    uint32_t                kv_ಅಡ್ಡಸಾಲುಗಳು;
    size_t                  src_stride;
    ಶೂನ್ಯ *                  curr_k;
    uint32_t                kv_ಆರಂಭ;
    uint32_t                ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
    size_t                  buf_idx;
} fa_k_int_ವಾದಗಳು_t;

ಸ್ಥಿರ ಶೂನ್ಯ fa_k_interleave_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    fa_k_int_ವಾದಗಳು_t *       ವಾದಗಳು_2  = (fa_k_int_ವಾದಗಳು_t *) ದತ್ತಾಂಶ;
    ರಚನೆ hmx_fa_context * factx = ವಾದಗಳು_2->factx;

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = ವಾದಗಳು_2->kv_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t = ವಾದಗಳು_2->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ      = i * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ        = (uint32_t) hex_smin(ಆರಂಭ + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);

    ಆದರೆ (ಆರಂಭ >= ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (ವಾದಗಳು_2->kv_ಆರಂಭ + ಆರಂಭ));
    ಆದರೆ (factx->k->type == HTP_TYPE_Q8_0) {
        ಗಾಗಿ (uint32_t r = ಆರಂಭ; r < ಕೊನೆ; ++r) {
            __fp16 * ಅಡ್ಡಸಾಲು_k = (__fp16 *)((ಅಕ್ಷರ *)ವಾದಗಳು_2->curr_k + r * ವಾದಗಳು_2->src_stride * ಗಾತ್ರ(__fp16));
            hvx_dequantize_ಅಡ್ಡಸಾಲು_flat_q8_0_f16(ಅಡ್ಡಸಾಲು_k, ಅಡ್ಡಸಾಲು_k, factx->DK);
        }
    }
    hmx_interleave_ಅಡ್ಡಸಾಲುಗಳು_ಗಮ್ಯ_tiles(factx->vtcm_k_tiles[ವಾದಗಳು_2->buf_idx], (ಸ್ಥಿರಾಂಕ __fp16 *) ವಾದಗಳು_2->curr_k, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, factx->DK_pad,
                             ವಾದಗಳು_2->src_stride, ಆರಂಭ, ಕೊನೆ);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (ವಾದಗಳು_2->kv_ಆರಂಭ + ಆರಂಭ));
}

ಸ್ಥಿರ ಶೂನ್ಯ fa_phase_k_interleave(ರಚನೆ hmx_fa_context * factx, uint32_t kv_ಅಡ್ಡಸಾಲುಗಳು, size_t src_stride, ಶೂನ್ಯ * curr_k, uint32_t kv_ಆರಂಭ, size_t buf_idx) {
    work_ಸರತಿ_t wp = factx->octx->ctx->work_ಸರತಿ;
    uint32_t n = 1;
    ಆದರೆ (factx->n_ಎಳೆಗಳು > 1 && kv_ಅಡ್ಡಸಾಲುಗಳು >= factx->n_ಎಳೆಗಳು * 2) {
        n = factx->n_ಎಳೆಗಳು;
    }
    uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t = hex_align_ಮೇಲೆ(hmx_ceil_div(kv_ಅಡ್ಡಸಾಲುಗಳು, n), 2);
    fa_k_int_ವಾದಗಳು_t ವಾದಗಳು_2 = { factx, kv_ಅಡ್ಡಸಾಲುಗಳು, src_stride, curr_k, kv_ಆರಂಭ, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t, buf_idx };
    ಆದರೆ (n > 1) {
        work_ಸರತಿ_ಸತತ(wp, fa_k_interleave_thread, &ವಾದಗಳು_2, n);
    } ಇಲ್ಲದಿದ್ದರೆ {
        fa_k_interleave_thread(1, 0, &ವಾದಗಳು_2);
    }
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ರಚನೆ hmx_fa_context * factx;
    uint32_t                kv_ಅಡ್ಡಸಾಲುಗಳು;
    size_t                  src_stride;
    ಶೂನ್ಯ *                  v_src;
    ಶೂನ್ಯ *                  v_tiles_dst;
    size_t                  n_ಲಂಬಸಾಲು_tiles;
    uint32_t                kv_ಆರಂಭ;
    uint32_t                ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
} fa_v_int_ವಾದಗಳು_t;

ಸ್ಥಿರ ಶೂನ್ಯ fa_v_interleave_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    fa_v_int_ವಾದಗಳು_t *       ವಾದಗಳು_2  = (fa_v_int_ವಾದಗಳು_t *) ದತ್ತಾಂಶ;
    ರಚನೆ hmx_fa_context * factx = ವಾದಗಳು_2->factx;

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = ವಾದಗಳು_2->kv_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t = ವಾದಗಳು_2->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ      = i * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ        = (uint32_t) hex_smin(ಆರಂಭ + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);

    ಆದರೆ (ಆರಂಭ >= ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
        ಹಿಂತಿರುಗಿ;
    }

    __fp16 * v_tiles_dst = (__fp16 *) ವಾದಗಳು_2->v_tiles_dst;

    ರಚನೆ htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (ವಾದಗಳು_2->kv_ಆರಂಭ + ಆರಂಭ));
    ಆದರೆ (factx->v->type == HTP_TYPE_Q8_0) {
        ಗಾಗಿ (uint32_t r = ಆರಂಭ; r < ಕೊನೆ; ++r) {
            __fp16 * ಅಡ್ಡಸಾಲು_v = (__fp16 *)((ಅಕ್ಷರ *)ವಾದಗಳು_2->v_src + r * ವಾದಗಳು_2->src_stride * ಗಾತ್ರ(__fp16));
            hvx_dequantize_ಅಡ್ಡಸಾಲು_flat_q8_0_f16(ಅಡ್ಡಸಾಲು_v, ಅಡ್ಡಸಾಲು_v, factx->DV);
        }
    }
    hmx_interleave_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಗಮ್ಯ_tiles(v_tiles_dst, (ಸ್ಥಿರಾಂಕ __fp16 *) ವಾದಗಳು_2->v_src, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, factx->DV_pad,
                             ವಾದಗಳು_2->src_stride, (uint32_t) ವಾದಗಳು_2->n_ಲಂಬಸಾಲು_tiles, ಆರಂಭ, ಕೊನೆ);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (ವಾದಗಳು_2->kv_ಆರಂಭ + ಆರಂಭ));
}

ಸ್ಥಿರ ಶೂನ್ಯ fa_phase_v_interleave(ರಚನೆ hmx_fa_context * factx,
                                  uint32_t                kv_ಅಡ್ಡಸಾಲುಗಳು,
                                  size_t                  src_stride,
                                  ಶೂನ್ಯ *                  v_src,
                                  ಶೂನ್ಯ *                  v_tiles_dst,
                                  size_t                  n_ಲಂಬಸಾಲು_tiles,
                                  uint32_t                kv_ಆರಂಭ) {
    work_ಸರತಿ_t wp = factx->octx->ctx->work_ಸರತಿ;
    uint32_t n = 1;
    ಆದರೆ (factx->n_ಎಳೆಗಳು > 1 && kv_ಅಡ್ಡಸಾಲುಗಳು >= factx->n_ಎಳೆಗಳು * 2) {
        n = factx->n_ಎಳೆಗಳು;
    }
    uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t = hex_align_ಮೇಲೆ(hmx_ceil_div(kv_ಅಡ್ಡಸಾಲುಗಳು, n), 2);
    fa_v_int_ವಾದಗಳು_t ವಾದಗಳು_2 = { factx, kv_ಅಡ್ಡಸಾಲುಗಳು, src_stride, v_src, v_tiles_dst, n_ಲಂಬಸಾಲು_tiles, kv_ಆರಂಭ, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t };
    ಆದರೆ (n > 1) {
        work_ಸರತಿ_ಸತತ(wp, fa_v_interleave_thread, &ವಾದಗಳು_2, n);
    } ಇಲ್ಲದಿದ್ದರೆ {
        fa_v_interleave_thread(1, 0, &ವಾದಗಳು_2);
    }
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ರಚನೆ hmx_fa_context *   factx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * q;
    uint32_t                  q_ಆರಂಭ;
    uint32_t                  kv_ತಲೆ;
    uint32_t                  ib3;
    size_t                    n_ಅಡ್ಡಸಾಲುಗಳು_g;
    size_t                    ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
    size_t                    n_ಅಡ್ಡಸಾಲುಗಳು_q;
    bool                      q_transposed;
    atomic_uint               barrier;
} fa_q_ತುಂಬು_ವಾದಗಳು_t;

ಸ್ಥಿರ ಶೂನ್ಯ fa_q_ತುಂಬು_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    fa_q_ತುಂಬು_ವಾದಗಳು_t *      ವಾದಗಳು_2  = (fa_q_ತುಂಬು_ವಾದಗಳು_t *) ದತ್ತಾಂಶ;
    ರಚನೆ hmx_fa_context * factx = ವಾದಗಳು_2->factx;

    ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲುಗಳು_g = ವಾದಗಳು_2->n_ಅಡ್ಡಸಾಲುಗಳು_g;
    ಸ್ಥಿರಾಂಕ size_t G        = factx->G;
    ಸ್ಥಿರಾಂಕ size_t DK       = factx->DK;

    // Partition the padded Q rows (g_br) across threads.
    // Keep start/end even so r and r+1 are always in the same thread's range.
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t = ವಾದಗಳು_2->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
    ಸ್ಥಿರಾಂಕ size_t ಆರಂಭ      = (size_t) i * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
    ಸ್ಥಿರಾಂಕ size_t ಕೊನೆ        = hex_smin(ಆರಂಭ + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t, factx->g_br);

    ರಚನೆ htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (ವಾದಗಳು_2->q_ಆರಂಭ * G + ಆರಂಭ));

    // Parallel initialization of per-block state
    {
        ಸ್ಥಿರಾಂಕ uint32_t g_br = factx->g_br;
        ಸ್ಥಿರಾಂಕ uint32_t DV   = factx->DV;

        ಸ್ಥಿರಾಂಕ size_t ಲಂಬಸಾಲು_vec_ಬೈಟುಗಳು = factx->ಲಂಬಸಾಲು_vec_ಬೈಟುಗಳು;
        ಸ್ಥಿರಾಂಕ size_t d_tile_ಬೈಟುಗಳು  = factx->d_tile_ಬೈಟುಗಳು;

        // Initialize vtcm_l_vec & vtcm_m_vec
        ಸ್ಥಿರಾಂಕ size_t l_ಬೈಟುಗಳು_ಪ್ರತಿ_t = hex_align_ಮೇಲೆ(ಲಂಬಸಾಲು_vec_ಬೈಟುಗಳು / n, 128);
        ಸ್ಥಿರಾಂಕ size_t l_ಆರಂಭ       = i * l_ಬೈಟುಗಳು_ಪ್ರತಿ_t;
        ಸ್ಥಿರಾಂಕ size_t l_ಕೊನೆ         = hex_smin(l_ಆರಂಭ + l_ಬೈಟುಗಳು_ಪ್ರತಿ_t, ಲಂಬಸಾಲು_vec_ಬೈಟುಗಳು);

        ಸ್ಥಿರಾಂಕ size_t m_ಬೈಟುಗಳು_ಪ್ರತಿ_t = hex_align_ಮೇಲೆ(ಲಂಬಸಾಲು_vec_ಬೈಟುಗಳು / n, 128);
        ಸ್ಥಿರಾಂಕ size_t m_ಆರಂಭ       = i * m_ಬೈಟುಗಳು_ಪ್ರತಿ_t;
        ಸ್ಥಿರಾಂಕ size_t m_ಕೊನೆ         = hex_smin(m_ಆರಂಭ + m_ಬೈಟುಗಳು_ಪ್ರತಿ_t, ಲಂಬಸಾಲು_vec_ಬೈಟುಗಳು);

        ಆದರೆ (l_ಆರಂಭ < ಲಂಬಸಾಲು_vec_ಬೈಟುಗಳು) {
            hvx_splat_u8_a((ಅಕ್ಷರ *) factx->vtcm_l_vec + l_ಆರಂಭ, 0, l_ಕೊನೆ - l_ಆರಂಭ);
        }
        ಆದರೆ (m_ಆರಂಭ < ಲಂಬಸಾಲು_vec_ಬೈಟುಗಳು) {
            hvx_splat_f32_a((ಅಕ್ಷರ *) factx->vtcm_m_vec + m_ಆರಂಭ, HTP_FA_M_INITIAL_VAL, (m_ಕೊನೆ - m_ಆರಂಭ) / ಗಾತ್ರ(ದಶಮಾಂಶ));
        }

        // Zero the whole rescale region: vtcm_d_tiles[0], the optional vtcm_d_tiles[1]
        // and vtcm_d_inv_l are equal-sized and allocated back to back, so one run covers
        // them all.  The scatter only ever writes the diagonal, ignore the rest.
        ಸ್ಥಿರಾಂಕ size_t d_ಬೈಟುಗಳು_ಪ್ರತಿ_t = hex_align_ಮೇಲೆ(d_tile_ಬೈಟುಗಳು / n, 128);
        ಸ್ಥಿರಾಂಕ size_t d_ಆರಂಭ       = i * d_ಬೈಟುಗಳು_ಪ್ರತಿ_t;
        ಸ್ಥಿರಾಂಕ size_t d_ಕೊನೆ         = hex_smin(d_ಆರಂಭ + d_ಬೈಟುಗಳು_ಪ್ರತಿ_t, d_tile_ಬೈಟುಗಳು);
        ಆದರೆ (d_ಆರಂಭ < d_tile_ಬೈಟುಗಳು) {
            hvx_splat_u8_a((ಅಕ್ಷರ *) factx->vtcm_d_tiles[0] + d_ಆರಂಭ, 0, d_ಕೊನೆ - d_ಆರಂಭ);
        }
    }

    ಆದರೆ (ಆರಂಭ < factx->g_br) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * q       = ವಾದಗಳು_2->q;
        ಸ್ಥಿರಾಂಕ uint32_t            q_ಆರಂಭ = ವಾದಗಳು_2->q_ಆರಂಭ;
        ಸ್ಥಿರಾಂಕ uint32_t            kv_ತಲೆ = ವಾದಗಳು_2->kv_ತಲೆ;
        ಸ್ಥಿರಾಂಕ uint32_t            ib3     = ವಾದಗಳು_2->ib3;

        ಸ್ಥಿರಾಂಕ bool use_q_dma = (factx->vtcm_q_dma != ಖಾಲಿ);

        __fp16 * q_tiles = factx->vtcm_q_tiles;
        ಸ್ಥಿರಾಂಕ size_t DK_pad = factx->DK_pad;
        ಆದರೆ (use_q_dma) {
            ಸ್ಥಿರಾಂಕ size_t g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ = hex_smin(ಕೊನೆ, n_ಅಡ್ಡಸಾಲುಗಳು_g);
            ಸ್ಥಿರಾಂಕ uint32_t d_ಎಲ್ಲೆ = factx->is_q_fp32 ? DK / 32 : DK / 64;

            uint8_t * q_flat  = (uint8_t *) factx->vtcm_q_dma;
            ಆದರೆ (DK_pad != DK) {
                ಆದರೆ (factx->is_q_fp32) {
                    hmx_fa_q_prep_fp32_pad(q_tiles, q_flat, ಆರಂಭ, ಕೊನೆ, g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ, DK, DK_pad, G, ವಾದಗಳು_2->n_ಅಡ್ಡಸಾಲುಗಳು_q, &factx->div_G, ವಾದಗಳು_2->q_transposed);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    hmx_fa_q_prep_fp16_pad(q_tiles, q_flat, ಆರಂಭ, ಕೊನೆ, g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ, DK, DK_pad, G, ವಾದಗಳು_2->n_ಅಡ್ಡಸಾಲುಗಳು_q, &factx->div_G, ವಾದಗಳು_2->q_transposed);
                }
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (factx->is_q_fp32) {
                ಆಯ್ಕೆ (d_ಎಲ್ಲೆ) {
                ಸ್ಥಿತಿ 2:  hmx_fa_q_prep_fp32_d2(q_tiles, q_flat, ಆರಂಭ, ಕೊನೆ, g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ, DK, G, ವಾದಗಳು_2->n_ಅಡ್ಡಸಾಲುಗಳು_q, &factx->div_G, ವಾದಗಳು_2->q_transposed); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 4:  hmx_fa_q_prep_fp32_d4(q_tiles, q_flat, ಆರಂಭ, ಕೊನೆ, g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ, DK, G, ವಾದಗಳು_2->n_ಅಡ್ಡಸಾಲುಗಳು_q, &factx->div_G, ವಾದಗಳು_2->q_transposed); ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ: hmx_fa_q_prep_fp32(   q_tiles, q_flat, ಆರಂಭ, ಕೊನೆ, g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ, DK, G, ವಾದಗಳು_2->n_ಅಡ್ಡಸಾಲುಗಳು_q, &factx->div_G, d_ಎಲ್ಲೆ, ವಾದಗಳು_2->q_transposed); ನಿಲ್ಲಿಸು;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆಯ್ಕೆ (d_ಎಲ್ಲೆ) {
                ಸ್ಥಿತಿ 1:  hmx_fa_q_prep_fp16_d1(q_tiles, q_flat, ಆರಂಭ, ಕೊನೆ, g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ, DK, G, ವಾದಗಳು_2->n_ಅಡ್ಡಸಾಲುಗಳು_q, &factx->div_G, ವಾದಗಳು_2->q_transposed); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 2:  hmx_fa_q_prep_fp16_d2(q_tiles, q_flat, ಆರಂಭ, ಕೊನೆ, g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ, DK, G, ವಾದಗಳು_2->n_ಅಡ್ಡಸಾಲುಗಳು_q, &factx->div_G, ವಾದಗಳು_2->q_transposed); ನಿಲ್ಲಿಸು;
                ಪೂರ್ವನಿಯೋಜಿತ: hmx_fa_q_prep_fp16(   q_tiles, q_flat, ಆರಂಭ, ಕೊನೆ, g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ, DK, G, ವಾದಗಳು_2->n_ಅಡ್ಡಸಾಲುಗಳು_q, &factx->div_G, d_ಎಲ್ಲೆ, ವಾದಗಳು_2->q_transposed); ನಿಲ್ಲಿಸು;
                }
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            // Fallback: direct-from-DDR/L2 path
            hmx_fa_q_prep_fallback(q_tiles, q->ದತ್ತಾಂಶ, q->nb[1], q->nb[2], q->nb[3],
                                   q_ಆರಂಭ, kv_ತಲೆ, ib3, ಆರಂಭ, ಕೊನೆ, n_ಅಡ್ಡಸಾಲುಗಳು_g, G, DK, DK_pad, factx->is_q_fp32, &factx->div_G);
        }
    }

    // Synchronize threads before zeroing out vtcm_o_tiles[0] to prevent race condition
    ಆದರೆ (n > 1) {
        atomic_fetch_sub(&ವಾದಗಳು_2->barrier, 1);
        ಆಗ (atomic_ತುಂಬು(&ವಾದಗಳು_2->barrier) > 0) {
            // spin wait
        }
    }

    // Zero out vtcm_o_tiles[0] as it was used as temp_q_vtcm
    {
        ಸ್ಥಿರಾಂಕ uint32_t g_br = factx->g_br;
        ಸ್ಥಿರಾಂಕ uint32_t DV   = factx->DV;
        ಸ್ಥಿರಾಂಕ size_t o_tile_ಬೈಟುಗಳು  = factx->o_tile_ಬೈಟುಗಳು;
        ಸ್ಥಿರಾಂಕ size_t o_ಬೈಟುಗಳು_ಪ್ರತಿ_t = hex_align_ಮೇಲೆ(o_tile_ಬೈಟುಗಳು / n, 128);
        ಸ್ಥಿರಾಂಕ size_t o_ಆರಂಭ       = i * o_ಬೈಟುಗಳು_ಪ್ರತಿ_t;
        ಸ್ಥಿರಾಂಕ size_t o_ಕೊನೆ         = hex_smin(o_ಆರಂಭ + o_ಬೈಟುಗಳು_ಪ್ರತಿ_t, o_tile_ಬೈಟುಗಳು);
        ಆದರೆ (o_ಆರಂಭ < o_tile_ಬೈಟುಗಳು) {
            hvx_splat_u8_a((ಅಕ್ಷರ *) factx->vtcm_o_tiles[0] + o_ಆರಂಭ, 0, o_ಕೊನೆ - o_ಆರಂಭ);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (ವಾದಗಳು_2->q_ಆರಂಭ * G + ಆರಂಭ));
}

ಸ್ಥಿರ ಶೂನ್ಯ fa_phase_q_ತುಂಬು(ರಚನೆ hmx_fa_context *   factx,
                            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * q,
                            uint32_t                  q_ಆರಂಭ,
                            uint32_t                  kv_ತಲೆ,
                            uint32_t                  ib3,
                            size_t                    n_ಅಡ್ಡಸಾಲುಗಳು_g) {
    work_ಸರತಿ_t wp = factx->octx->ctx->work_ಸರತಿ;
    uint32_t n = 1;
    ಆದರೆ (factx->n_ಎಳೆಗಳು > 1 && n_ಅಡ್ಡಸಾಲುಗಳು_g >= (size_t) (factx->n_ಎಳೆಗಳು * 2)) {
        n = factx->n_ಎಳೆಗಳು;
    }
    size_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t = hex_align_ಮೇಲೆ(hmx_ceil_div(factx->g_br, n), 2);
    ಸ್ಥಿರಾಂಕ uint32_t n_ಅಡ್ಡಸಾಲುಗಳು_q = hex_smin(factx->Br, factx->neq1 - q_ಆರಂಭ);
    fa_q_ತುಂಬು_ವಾದಗಳು_t ವಾದಗಳು_2;
    ವಾದಗಳು_2.factx = factx;
    ವಾದಗಳು_2.q = q;
    ವಾದಗಳು_2.q_ಆರಂಭ = q_ಆರಂಭ;
    ವಾದಗಳು_2.kv_ತಲೆ = kv_ತಲೆ;
    ವಾದಗಳು_2.ib3 = ib3;
    ವಾದಗಳು_2.n_ಅಡ್ಡಸಾಲುಗಳು_g = n_ಅಡ್ಡಸಾಲುಗಳು_g;
    ವಾದಗಳು_2.ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t = ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
    ವಾದಗಳು_2.n_ಅಡ್ಡಸಾಲುಗಳು_q = n_ಅಡ್ಡಸಾಲುಗಳು_q;
    ವಾದಗಳು_2.q_transposed = q->nb[1] < q->nb[2];
    atomic_init(&ವಾದಗಳು_2.barrier, n);
    ಆದರೆ (n > 1) {
        work_ಸರತಿ_ಸತತ(wp, fa_q_ತುಂಬು_thread, &ವಾದಗಳು_2, n);
    } ಇಲ್ಲದಿದ್ದರೆ {
        fa_q_ತುಂಬು_thread(1, 0, &ವಾದಗಳು_2);
    }
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ರಚನೆ hmx_fa_context *   factx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst;
    ಸ್ಥಿರಾಂಕ __fp16 *            o_tile_src;
    uint32_t                  q_ಆರಂಭ;
    uint32_t                  kv_ತಲೆ;
    uint32_t                  ib3;
    size_t                    n_ಅಡ್ಡಸಾಲುಗಳು_g;
    size_t                    ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
} fa_o_store_ವಾದಗಳು_t;

ಸ್ಥಿರ ಶೂನ್ಯ fa_o_store_thread_f32(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    fa_o_store_ವಾದಗಳು_t *     ವಾದಗಳು_2  = (fa_o_store_ವಾದಗಳು_t *) ದತ್ತಾಂಶ;
    ರಚನೆ hmx_fa_context * factx = ವಾದಗಳು_2->factx;

    ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲುಗಳು_g = ವಾದಗಳು_2->n_ಅಡ್ಡಸಾಲುಗಳು_g;
    ಸ್ಥಿರಾಂಕ size_t G        = factx->G;
    ಸ್ಥಿರಾಂಕ size_t DV       = factx->DV;

    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t = ವಾದಗಳು_2->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
    ಸ್ಥಿರಾಂಕ size_t ಆರಂಭ      = (size_t) i * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
    ಸ್ಥಿರಾಂಕ size_t ಕೊನೆ        = hex_smin(ಆರಂಭ + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t, n_ಅಡ್ಡಸಾಲುಗಳು_g);

    ಆದರೆ (ಆರಂಭ >= n_ಅಡ್ಡಸಾಲುಗಳು_g) {
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (ವಾದಗಳು_2->q_ಆರಂಭ * G + ಆರಂಭ));

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst        = ವಾದಗಳು_2->dst;
    ಸ್ಥಿರಾಂಕ __fp16 *            o_tile_src = ವಾದಗಳು_2->o_tile_src;
    ಸ್ಥಿರಾಂಕ uint32_t            q_ಆರಂಭ    = ವಾದಗಳು_2->q_ಆರಂಭ;
    ಸ್ಥಿರಾಂಕ uint32_t            kv_ತಲೆ    = ವಾದಗಳು_2->kv_ತಲೆ;
    ಸ್ಥಿರಾಂಕ uint32_t            ib3        = ವಾದಗಳು_2->ib3;

    ಸ್ಥಿರಾಂಕ size_t DV_pad = factx->DV_pad;

    size_t q_idx = fastdiv(ಆರಂಭ, &factx->div_G);
    size_t h_idx = fastmodulo(ಆರಂಭ, G, &factx->div_G);

    ಗಾಗಿ (size_t r = ಆರಂಭ; r < ಕೊನೆ; ++r) {
        ದಶಮಾಂಶ * ಹೊರಹೋಗು = (ದಶಮಾಂಶ *) ((uint8_t *) dst->ದತ್ತಾಂಶ + (kv_ತಲೆ * G + h_idx) * dst->nb[1] +
                                 (q_ಆರಂಭ + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        size_t         r1            = r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        ಸ್ಥಿರಾಂಕ __fp16 * tile_ಅಡ್ಡಸಾಲು_ತಳ = o_tile_src + r0 * HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು * DV_pad;

        ಗಾಗಿ (uint32_t d = 0; d < DV / 32; ++d) {
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಒಳ_tile = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (tile_ಅಡ್ಡಸಾಲು_ತಳ + d * HMX_FP16_TILE_N_ELMS);
            HVX_ಸದಿಶ_Pair     vp      = hvx_vec_f16_ಗಮ್ಯ_f32_shuff(ಒಳ_tile[r1 / 2]);
            ಆದರೆ (r1 % 2 == 0) {
                *(HVX_UVector *) (ಹೊರಹೋಗು + d * 32) = Q6_V_lo_W(vp);
            } ಇಲ್ಲದಿದ್ದರೆ {
                *(HVX_UVector *) (ಹೊರಹೋಗು + d * 32) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail: DV not a multiple of 32 (e.g. 72 -> last 8 lanes). Partial vector-write
        // for the remaining (DV % 32) floats.
        ಸ್ಥಿರಾಂಕ uint32_t d_ಬಾಲ = DV / 32;
        ಸ್ಥಿರಾಂಕ uint32_t rem    = DV - d_ಬಾಲ * 32;
        ಆದರೆ (rem) {
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಒಳ_tile = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (tile_ಅಡ್ಡಸಾಲು_ತಳ + d_ಬಾಲ * HMX_FP16_TILE_N_ELMS);
            HVX_ಸದಿಶ_Pair     vp      = hvx_vec_f16_ಗಮ್ಯ_f32_shuff(ಒಳ_tile[r1 / 2]);
            HVX_ಸದಿಶ         vd      = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((ಶೂನ್ಯ *) (ಹೊರಹೋಗು + d_ಬಾಲ * 32), rem * ಗಾತ್ರ(ದಶಮಾಂಶ), vd);
        }

        h_idx++;
        ಆದರೆ (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (ವಾದಗಳು_2->q_ಆರಂಭ * G + ಆರಂಭ));
}

ಸ್ಥಿರ ಶೂನ್ಯ fa_o_store_thread_f16(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    fa_o_store_ವಾದಗಳು_t *     ವಾದಗಳು_2  = (fa_o_store_ವಾದಗಳು_t *) ದತ್ತಾಂಶ;
    ರಚನೆ hmx_fa_context * factx = ವಾದಗಳು_2->factx;

    ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲುಗಳು_g   = ವಾದಗಳು_2->n_ಅಡ್ಡಸಾಲುಗಳು_g;
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t = ವಾದಗಳು_2->ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
    ಸ್ಥಿರಾಂಕ size_t G          = factx->G;
    ಸ್ಥಿರಾಂಕ size_t DV         = factx->DV;
    ಸ್ಥಿರಾಂಕ size_t ಆರಂಭ      = (size_t) i * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t;
    ಸ್ಥಿರಾಂಕ size_t ಕೊನೆ        = hex_smin(ಆರಂಭ + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t, n_ಅಡ್ಡಸಾಲುಗಳು_g);

    ಆದರೆ (ಆರಂಭ >= n_ಅಡ್ಡಸಾಲುಗಳು_g) {
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (ವಾದಗಳು_2->q_ಆರಂಭ * G + ಆರಂಭ));

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst        = ವಾದಗಳು_2->dst;
    ಸ್ಥಿರಾಂಕ __fp16 *            o_tile_src = ವಾದಗಳು_2->o_tile_src;
    ಸ್ಥಿರಾಂಕ uint32_t            q_ಆರಂಭ    = ವಾದಗಳು_2->q_ಆರಂಭ;
    ಸ್ಥಿರಾಂಕ uint32_t            kv_ತಲೆ    = ವಾದಗಳು_2->kv_ತಲೆ;
    ಸ್ಥಿರಾಂಕ uint32_t            ib3        = ವಾದಗಳು_2->ib3;

    // O-tiles use the padded head dim (DV_pad); dst holds the real DV lanes.
    ಸ್ಥಿರಾಂಕ size_t DV_pad = factx->DV_pad;

    size_t q_idx = fastdiv(ಆರಂಭ, &factx->div_G);
    size_t h_idx = fastmodulo(ಆರಂಭ, G, &factx->div_G);

    ಗಾಗಿ (size_t r = ಆರಂಭ; r < ಕೊನೆ; ++r) {
        __fp16 * ಹೊರಹೋಗು = (__fp16 *) ((uint8_t *) dst->ದತ್ತಾಂಶ + (kv_ತಲೆ * G + h_idx) * dst->nb[1] +
                                   (q_ಆರಂಭ + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        size_t         r1            = r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        ಸ್ಥಿರಾಂಕ __fp16 * tile_ಅಡ್ಡಸಾಲು_ತಳ = o_tile_src + r0 * HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು * DV_pad;

        ಗಾಗಿ (uint32_t d = 0; d < DV / 64; ++d) {
            ಸ್ಥಿರಾಂಕ __fp16 *     ಒಳ_dtile = tile_ಅಡ್ಡಸಾಲು_ತಳ + d * HMX_FP16_TILE_N_ELMS * 2;
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in0   = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) ಒಳ_dtile) + r1 / 2;
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in1   = pv_in0 + 16;
            HVX_ಸದಿಶ_Pair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            ಆದರೆ (r1 % 2 == 0) {
                *(HVX_UVector *) (ಹೊರಹೋಗು + d * 64) = Q6_V_lo_W(vp);
            } ಇಲ್ಲದಿದ್ದರೆ {
                *(HVX_UVector *) (ಹೊರಹೋಗು + d * 64) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail when DV is not a multiple of 64.
        ಸ್ಥಿರಾಂಕ uint32_t d_ಬಾಲ = DV / 64;
        ಸ್ಥಿರಾಂಕ uint32_t rem    = DV - d_ಬಾಲ * 64;
        ಆದರೆ (rem) {
            ಸ್ಥಿರಾಂಕ __fp16 *     ಒಳ_dtile = tile_ಅಡ್ಡಸಾಲು_ತಳ + d_ಬಾಲ * HMX_FP16_TILE_N_ELMS * 2;
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in0   = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) ಒಳ_dtile) + r1 / 2;
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in1   = pv_in0 + 16;
            HVX_ಸದಿಶ_Pair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            HVX_ಸದಿಶ         vd       = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((ಶೂನ್ಯ *) (ಹೊರಹೋಗು + d_ಬಾಲ * 64), rem * ಗಾತ್ರ(__fp16), vd);
        }

        h_idx++;
        ಆದರೆ (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (ವಾದಗಳು_2->q_ಆರಂಭ * G + ಆರಂಭ));
}

ಸ್ಥಿರ ಶೂನ್ಯ fa_phase_o_store(ರಚನೆ hmx_fa_context *   factx,
                             ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst,
                             ಸ್ಥಿರಾಂಕ __fp16 *            o_tile_src,
                             uint32_t                  q_ಆರಂಭ,
                             uint32_t                  kv_ತಲೆ,
                             uint32_t                  ib3,
                             size_t                    n_ಅಡ್ಡಸಾಲುಗಳು_g) {
    work_ಸರತಿ_t wp = factx->octx->ctx->work_ಸರತಿ;
    uint32_t n = 1;
    ಆದರೆ (factx->n_ಎಳೆಗಳು > 1 && n_ಅಡ್ಡಸಾಲುಗಳು_g >= (size_t) (factx->n_ಎಳೆಗಳು * 2)) {
        n = factx->n_ಎಳೆಗಳು;
    }
    size_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t = hmx_ceil_div(n_ಅಡ್ಡಸಾಲುಗಳು_g, n);
    fa_o_store_ವಾದಗಳು_t ವಾದಗಳು_2 = { factx, dst, o_tile_src, q_ಆರಂಭ, kv_ತಲೆ, ib3, n_ಅಡ್ಡಸಾಲುಗಳು_g, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_t };
    ಕೆಲಸಗಾರ_callback_t store_fn = factx->is_dst_fp32 ? fa_o_store_thread_f32 : fa_o_store_thread_f16;
    ಆದರೆ (n > 1) {
        work_ಸರತಿ_ಸತತ(wp, store_fn, &ವಾದಗಳು_2, n);
    } ಇಲ್ಲದಿದ್ದರೆ {
        store_fn(1, 0, &ವಾದಗಳು_2);
    }
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ರಚನೆ hmx_fa_context *   factx;
    size_t                    buf_idx;
    size_t                    kv_ಅಡ್ಡಸಾಲುಗಳು;
    size_t                    n_ಅಡ್ಡಸಾಲುಗಳು_g;
    size_t                    n_ಲಂಬಸಾಲು_tiles;
    size_t                    n_tiles_ಪ್ರತಿ_bc;
    size_t                    n_ಅಡ್ಡಸಾಲು_tiles;
    size_t                    n_ಅಡ್ಡಸಾಲು_tiles_g_br;
    uint32_t                  Bc;
    uint32_t                  G;
    uint32_t                  kv_ತಲೆ;
    uint32_t                  kv_ಆರಂಭ;
    uint32_t                  q_ಆರಂಭ;
    uint32_t                  ib3;
    bool                      has_alibi;  // true when max_bias != 0 (need slope * mask + add)
    __fp16 *                  slopes;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಮುಖವಾಡ;
    ಸ್ಥಿರಾಂಕ __fp16 *            ಮುಖವಾಡ_vtcm;             // VTCM mask buffer base (NULL = DDR fallback)
    size_t                    ಮುಖವಾಡ_vtcm_ಅಡ್ಡಸಾಲು_stride;  // elements (__fp16) per row in VTCM mask buffer
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು     thread_div;
} fa_softmax_ವಾದಗಳು_t;

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ fa_softmax_impl(
    ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ,
    ಸ್ಥಿರಾಂಕ bool has_ಮುಖವಾಡ,
    ಸ್ಥಿರಾಂಕ bool ಮುಖವಾಡ_ಪ್ರಸಾರ,
    ಸ್ಥಿರಾಂಕ bool is_g1,
    ಸ್ಥಿರಾಂಕ bool has_alibi,
    ಸ್ಥಿರಾಂಕ bool has_softcap
) {
    fa_softmax_ವಾದಗಳು_t *     ವಾದಗಳು_2  = (fa_softmax_ವಾದಗಳು_t *) ದತ್ತಾಂಶ;
    ರಚನೆ hmx_fa_context * factx = ವಾದಗಳು_2->factx;

    ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲುಗಳು_g       = ವಾದಗಳು_2->n_ಅಡ್ಡಸಾಲುಗಳು_g;
    ಸ್ಥಿರಾಂಕ size_t kv_ಅಡ್ಡಸಾಲುಗಳು        = ವಾದಗಳು_2->kv_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ size_t Bc             = ವಾದಗಳು_2->Bc;
    ಸ್ಥಿರಾಂಕ size_t G              = ವಾದಗಳು_2->G;
    ಸ್ಥಿರಾಂಕ size_t n_tiles_ಪ್ರತಿ_bc = ವಾದಗಳು_2->n_tiles_ಪ್ರತಿ_bc;
    ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲು_vec_cnt  = hmx_ceil_div(n_ಅಡ್ಡಸಾಲುಗಳು_g, 64);
    ಸ್ಥಿರಾಂಕ uint32_t im3          = has_ಮುಖವಾಡ ? fastmodulo(ವಾದಗಳು_2->ib3, ವಾದಗಳು_2->ಮುಖವಾಡ->ne[3], &factx->src3_div3) : 0;

    size_t vec_ಆರಂಭ = 0;
    size_t vec_ಕೊನೆ   = n_ಅಡ್ಡಸಾಲು_vec_cnt;
    ಆದರೆ (n > 1) {
        ಸ್ಥಿರಾಂಕ size_t vecs_ಪ್ರತಿ_t = fastdiv(n_ಅಡ್ಡಸಾಲು_vec_cnt + n - 1, &ವಾದಗಳು_2->thread_div);
        vec_ಆರಂಭ = i * vecs_ಪ್ರತಿ_t;
        vec_ಕೊನೆ   = hex_smin(vec_ಆರಂಭ + vecs_ಪ್ರತಿ_t, n_ಅಡ್ಡಸಾಲು_vec_cnt);
    }

    ಆದರೆ (vec_ಆರಂಭ >= n_ಅಡ್ಡಸಾಲು_vec_cnt) {
        ಹಿಂತಿರುಗಿ;
    }

    ರಚನೆ htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (ವಾದಗಳು_2->q_ಆರಂಭ * G + vec_ಆರಂಭ * 64));

    // Per-thread row scratch: thread i uses bufs at offset i * 2 * stride
    ಸ್ಥಿರಾಂಕ size_t ಅಡ್ಡಸಾಲು_buf_stride = factx->ಅಡ್ಡಸಾಲು_buf_stride;
    HVX_ಸದಿಶ * my_ಅಡ್ಡಸಾಲು_buf0    = factx->vtcm_ಅಡ್ಡಸಾಲು_bufs + i * 2 * ಅಡ್ಡಸಾಲು_buf_stride;
    HVX_ಸದಿಶ * my_ಅಡ್ಡಸಾಲು_buf1    = my_ಅಡ್ಡಸಾಲು_buf0 + ಅಡ್ಡಸಾಲು_buf_stride;

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_neg_inf = Q6_Vh_vsplat_R(0xfbff);

    ಗಾಗಿ (size_t r_vec_idx = vec_ಆರಂಭ; r_vec_idx < vec_ಕೊನೆ; ++r_vec_idx) {
        HVX_ಸದಿಶ rowmax_acc_v = v_neg_inf;
        HVX_ಸದಿಶ rowsum_acc_v = Q6_V_vzero();
        HVX_ಸದಿಶ m_prev_v0    = factx->vtcm_m_vec[r_vec_idx * 2 + 0];
        HVX_ಸದಿಶ m_prev_v1    = factx->vtcm_m_vec[r_vec_idx * 2 + 1];

        HVX_ಸದಿಶ v_slopes = Q6_V_vzero();
        ಆದರೆ (has_alibi) {
            v_slopes = hvx_vmem(ವಾದಗಳು_2->slopes + r_vec_idx * 64);
        }

        ಗಾಗಿ (uint32_t r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            uint32_t r = r_vec_idx * 64 + r_vec_off;
            ಆದರೆ (r >= hex_align_ಮೇಲೆ(n_ಅಡ್ಡಸಾಲುಗಳು_g, 2)) {
                ನಿಲ್ಲಿಸು;
            }

            uint32_t r0 = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
            uint32_t r1 = r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;

            ಸ್ಥಿರಾಂಕ __fp16 * s_ld_ತಳ = factx->vtcm_s_tiles[ವಾದಗಳು_2->buf_idx] + r0 * HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು * Bc;
            __fp16 *       p_st_ತಳ = factx->vtcm_p_tiles[ವಾದಗಳು_2->buf_idx] + r0 * HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು * Bc;

            // Decode 2 rows from S tiles into per-thread row buffers
            ಆದರೆ (has_softcap) {
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_cap = hvx_vec_splat_f16(factx->logit_softcap);
                ಗಾಗಿ (size_t c = 0; c < kv_ಅಡ್ಡಸಾಲುಗಳು; c += 64) {
                    size_t             ci       = c / 64;
                    ಸ್ಥಿರಾಂಕ __fp16 *     ಒಳ_dtile = s_ld_ತಳ + ci * HMX_FP16_TILE_N_ELMS * 2;
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_s_in0 = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) ಒಳ_dtile) + r1 / 2;
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_s_in1 = pv_s_in0 + 16;

                    HVX_ಸದಿಶ_Pair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    HVX_ಸದಿಶ     v_s_row0  = Q6_V_lo_W(vp_s_drow);
                    HVX_ಸದಿಶ     v_s_row1  = Q6_V_hi_W(vp_s_drow);

                    HVX_ಸದಿಶ t0   = hvx_vec_tanh_f16(v_s_row0);
                    my_ಅಡ್ಡಸಾಲು_buf0[ci] = hvx_vec_mul_f16_f16(t0, v_cap);

                    HVX_ಸದಿಶ t1   = hvx_vec_tanh_f16(v_s_row1);
                    my_ಅಡ್ಡಸಾಲು_buf1[ci] = hvx_vec_mul_f16_f16(t1, v_cap);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                size_t c = 0;
                ಗಾಗಿ (; c + 64 < kv_ಅಡ್ಡಸಾಲುಗಳು; c += 128) {
                    size_t             ci0       = c / 64;
                    size_t             ci1       = ci0 + 1;
                    ಸ್ಥಿರಾಂಕ __fp16 *     ಒಳ_dtile0 = s_ld_ತಳ + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                    ಸ್ಥಿರಾಂಕ __fp16 *     ಒಳ_dtile1 = s_ld_ತಳ + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_s_in0_0 = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) ಒಳ_dtile0) + r1 / 2;
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_s_in1_0 = pv_s_in0_0 + 16;
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_s_in0_1 = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) ಒಳ_dtile1) + r1 / 2;
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_s_in1_1 = pv_s_in0_1 + 16;

                    HVX_ಸದಿಶ_Pair vp_s_drow0 = Q6_W_vdeal_VVR(*pv_s_in1_0, *pv_s_in0_0, -2);
                    my_ಅಡ್ಡಸಾಲು_buf0[ci0]          = Q6_V_lo_W(vp_s_drow0);
                    my_ಅಡ್ಡಸಾಲು_buf1[ci0]          = Q6_V_hi_W(vp_s_drow0);

                    HVX_ಸದಿಶ_Pair vp_s_drow1 = Q6_W_vdeal_VVR(*pv_s_in1_1, *pv_s_in0_1, -2);
                    my_ಅಡ್ಡಸಾಲು_buf0[ci1]          = Q6_V_lo_W(vp_s_drow1);
                    my_ಅಡ್ಡಸಾಲು_buf1[ci1]          = Q6_V_hi_W(vp_s_drow1);
                }
                ಗಾಗಿ (; c < kv_ಅಡ್ಡಸಾಲುಗಳು; c += 64) {
                    size_t             ci       = c / 64;
                    ಸ್ಥಿರಾಂಕ __fp16 *     ಒಳ_dtile = s_ld_ತಳ + ci * HMX_FP16_TILE_N_ELMS * 2;
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_s_in0 = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) ಒಳ_dtile) + r1 / 2;
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_s_in1 = pv_s_in0 + 16;

                    HVX_ಸದಿಶ_Pair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    my_ಅಡ್ಡಸಾಲು_buf0[ci]          = Q6_V_lo_W(vp_s_drow);
                    my_ಅಡ್ಡಸಾಲು_buf1[ci]          = Q6_V_hi_W(vp_s_drow);
                }
            }

            // Apply mask & compute rowmax(S)
            HVX_ಸದಿಶ v_slope0 = Q6_V_vzero();
            HVX_ಸದಿಶ v_slope1 = Q6_V_vzero();
            ಆದರೆ (has_alibi) {
                v_slope0 = hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, r_vec_off * 2));
                v_slope1 = (r + 1 < n_ಅಡ್ಡಸಾಲುಗಳು_g) ? hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, (r_vec_off + 1) * 2)) : Q6_V_vzero();
            }

            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_ಮಿತಿ = Q6_Vh_vsplat_R(0xcc00);  // fp16 -16.0

            HVX_ಸದಿಶ v_s_rowmax0 = v_neg_inf;
            HVX_ಸದಿಶ v_s_rowmax1 = v_neg_inf;
            ಆದರೆ (has_ಮುಖವಾಡ) {
                ಗಾಗಿ (size_t c = 0; c < kv_ಅಡ್ಡಸಾಲುಗಳು; c += 64) {
                    size_t         ci          = c / 64;
                    ಸ್ಥಿರಾಂಕ size_t   ne          = hex_smin(kv_ಅಡ್ಡಸಾಲುಗಳು - c, 64);
                    HVX_ಸದಿಶ_Pred q_ಬಾಲ_keep = Q6_Q_vsetq2_R(ne * ಗಾತ್ರ(__fp16));

                    HVX_ಸದಿಶ v_mask0, v_mask1;

                    ಆದರೆ (ಮುಖವಾಡ_ಪ್ರಸಾರ) {
                        ಆದರೆ (is_g1) {
                            ಸ್ಥಿರಾಂಕ size_t qi0 = r + 0;
                            v_mask0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ವಾದಗಳು_2->ಮುಖವಾಡ_vtcm + qi0 * ವಾದಗಳು_2->ಮುಖವಾಡ_vtcm_ಅಡ್ಡಸಾಲು_stride + c);
                            v_mask1 = v_neg_inf;
                            ಆದರೆ (r + 1 < n_ಅಡ್ಡಸಾಲುಗಳು_g) {
                                ಸ್ಥಿರಾಂಕ size_t qi1 = r + 1;
                                v_mask1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ವಾದಗಳು_2->ಮುಖವಾಡ_vtcm + qi1 * ವಾದಗಳು_2->ಮುಖವಾಡ_vtcm_ಅಡ್ಡಸಾಲು_stride + c);
                            }
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            ಸ್ಥಿರಾಂಕ size_t qi0 = fastdiv(r + 0, &factx->div_G);
                            v_mask0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ವಾದಗಳು_2->ಮುಖವಾಡ_vtcm + qi0 * ವಾದಗಳು_2->ಮುಖವಾಡ_vtcm_ಅಡ್ಡಸಾಲು_stride + c);
                            v_mask1 = v_neg_inf;
                            ಆದರೆ (r + 1 < n_ಅಡ್ಡಸಾಲುಗಳು_g) {
                                ಸ್ಥಿರಾಂಕ size_t qi1 = fastdiv(r + 1, &factx->div_G);
                                ಆದರೆ (qi1 == qi0) {
                                    v_mask1 = v_mask0;
                                } ಇಲ್ಲದಿದ್ದರೆ {
                                    v_mask1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ವಾದಗಳು_2->ಮುಖವಾಡ_vtcm + qi1 * ವಾದಗಳು_2->ಮುಖವಾಡ_vtcm_ಅಡ್ಡಸಾಲು_stride + c);
                                }
                            }
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        // Head-dependent mask: pre-interleaved per row r.
                        ಸ್ಥಿರಾಂಕ size_t r0 = r + 0;
                        v_mask0 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ವಾದಗಳು_2->ಮುಖವಾಡ_vtcm + r0 * ವಾದಗಳು_2->ಮುಖವಾಡ_vtcm_ಅಡ್ಡಸಾಲು_stride + c);
                        v_mask1 = v_neg_inf;
                        ಆದರೆ (r + 1 < n_ಅಡ್ಡಸಾಲುಗಳು_g) {
                            ಸ್ಥಿರಾಂಕ size_t r1 = r + 1;
                            v_mask1 = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ವಾದಗಳು_2->ಮುಖವಾಡ_vtcm + r1 * ವಾದಗಳು_2->ಮುಖವಾಡ_vtcm_ಅಡ್ಡಸಾಲು_stride + c);
                        }
                    }

                    // Threshold: mask values below -16.0 are treated as -inf (causal mask).
                    HVX_ಸದಿಶ_Pred q_keep0 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_ಮಿತಿ), q_ಬಾಲ_keep);
                    HVX_ಸದಿಶ_Pred q_keep1 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_ಮಿತಿ), q_ಬಾಲ_keep);

                    // Scale mask values by log2(e) for base-2 calculations
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_log2e = hvx_vec_splat_f16(EXP_LOG2E_F);
                    HVX_ಸದಿಶ v_mask0_scaled = hvx_vec_mul_f16_f16(v_mask0, v_log2e);
                    HVX_ಸದಿಶ v_mask1_scaled = hvx_vec_mul_f16_f16(v_mask1, v_log2e);

                    ಆದರೆ (has_alibi) {
                        HVX_ಸದಿಶ v_sm0 = hvx_vec_mul_f16_f16(v_mask0_scaled, v_slope0);
                        HVX_ಸದಿಶ v_sm1 = hvx_vec_mul_f16_f16(v_mask1_scaled, v_slope1);
                        my_ಅಡ್ಡಸಾಲು_buf0[ci]  = Q6_V_vmux_QVV(q_keep0, hvx_vec_ಕೂಡಿಸು_f16_f16(my_ಅಡ್ಡಸಾಲು_buf0[ci], v_sm0), v_neg_inf);
                        my_ಅಡ್ಡಸಾಲು_buf1[ci]  = Q6_V_vmux_QVV(q_keep1, hvx_vec_ಕೂಡಿಸು_f16_f16(my_ಅಡ್ಡಸಾಲು_buf1[ci], v_sm1), v_neg_inf);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        my_ಅಡ್ಡಸಾಲು_buf0[ci] = Q6_V_vmux_QVV(q_keep0, hvx_vec_ಕೂಡಿಸು_f16_f16(my_ಅಡ್ಡಸಾಲು_buf0[ci], v_mask0_scaled), v_neg_inf);
                        my_ಅಡ್ಡಸಾಲು_buf1[ci] = Q6_V_vmux_QVV(q_keep1, hvx_vec_ಕೂಡಿಸು_f16_f16(my_ಅಡ್ಡಸಾಲು_buf1[ci], v_mask1_scaled), v_neg_inf);
                    }

                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_ಅಡ್ಡಸಾಲು_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_ಅಡ್ಡಸಾಲು_buf1[ci]);
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                size_t c = 0;
                ಗಾಗಿ (; c + 64 < kv_ಅಡ್ಡಸಾಲುಗಳು; c += 128) {
                    size_t ci0 = c / 64;
                    size_t ci1 = ci0 + 1;
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_ಅಡ್ಡಸಾಲು_buf0[ci0]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_ಅಡ್ಡಸಾಲು_buf1[ci0]);
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_ಅಡ್ಡಸಾಲು_buf0[ci1]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_ಅಡ್ಡಸಾಲು_buf1[ci1]);
                }
                ಗಾಗಿ (; c < kv_ಅಡ್ಡಸಾಲುಗಳು; c += 64) {
                    size_t         ci          = c / 64;
                    ಸ್ಥಿರಾಂಕ size_t   ne          = hex_smin(kv_ಅಡ್ಡಸಾಲುಗಳು - c, 64);
                    HVX_ಸದಿಶ_Pred q_ಬಾಲ_keep = Q6_Q_vsetq2_R(ne * ಗಾತ್ರ(__fp16));
                    ಆದರೆ (ne < 64) {
                        my_ಅಡ್ಡಸಾಲು_buf0[ci] = Q6_V_vmux_QVV(q_ಬಾಲ_keep, my_ಅಡ್ಡಸಾಲು_buf0[ci], v_neg_inf);
                        my_ಅಡ್ಡಸಾಲು_buf1[ci] = Q6_V_vmux_QVV(q_ಬಾಲ_keep, my_ಅಡ್ಡಸಾಲು_buf1[ci], v_neg_inf);
                    }
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_ಅಡ್ಡಸಾಲು_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_ಅಡ್ಡಸಾಲು_buf1[ci]);
                }
            }

            v_s_rowmax0 = hvx_vec_reduce_max_f16(v_s_rowmax0);
            v_s_rowmax1 = hvx_vec_reduce_max_f16(v_s_rowmax1);

            // Splat m_prev[r], m_prev[r+1] from the float per-row accumulators and convert to fp16 vectors
            HVX_ಸದಿಶ v_m_prev0, v_m_prev1;
            ಆದರೆ (r_vec_off < 32) {
                HVX_ಸದಿಶ v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, r_vec_off * 4));
                v_m_prev0 = hvx_vec_f32_ಗಮ್ಯ_f16(v0, v0);
                ಆದರೆ (r + 1 < n_ಅಡ್ಡಸಾಲುಗಳು_g) {
                    HVX_ಸದಿಶ v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, (r_vec_off + 1) * 4));
                    v_m_prev1 = hvx_vec_f32_ಗಮ್ಯ_f16(v1, v1);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    v_m_prev1 = Q6_V_vzero();
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                HVX_ಸದಿಶ v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v1, (r_vec_off - 32) * 4));
                v_m_prev0 = hvx_vec_f32_ಗಮ್ಯ_f16(v0, v0);
                ಆದರೆ (r + 1 < n_ಅಡ್ಡಸಾಲುಗಳು_g) {
                    HVX_ಸದಿಶ v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v1, (r_vec_off + 1 - 32) * 4));
                    v_m_prev1 = hvx_vec_f32_ಗಮ್ಯ_f16(v1, v1);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    v_m_prev1 = Q6_V_vzero();
                }
            }

            HVX_ಸದಿಶ v_dup_m0 = Q6_Vhf_vmax_VhfVhf(v_m_prev0, v_s_rowmax0);
            HVX_ಸದಿಶ v_dup_m1 = Q6_Vhf_vmax_VhfVhf(v_m_prev1, v_s_rowmax1);

            // Insert row r, r+1 rowmax into rowmax_acc_v
            {
                HVX_ಸದಿಶ_Pred p_ಆರಂಭ = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_ಸದಿಶ_Pred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_ಸದಿಶ_Pred p_ಕೊನೆ   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_ಸದಿಶ_Pred p_lane0 = Q6_Q_and_QQn(p_mid, p_ಆರಂಭ);
                HVX_ಸದಿಶ_Pred p_lane1 = Q6_Q_and_QQn(p_ಕೊನೆ, p_mid);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane0, v_dup_m0, rowmax_acc_v);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane1, v_dup_m1, rowmax_acc_v);
            }

            // Compute P = exp(S - m_new)
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_zero      = Q6_V_vzero();
            HVX_ಸದಿಶ       v_p_rowsum0 = v_zero;
            HVX_ಸದಿಶ       v_p_rowsum1 = v_zero;

            size_t c = 0;
            ಗಾಗಿ (; c + 64 < kv_ಅಡ್ಡಸಾಲುಗಳು; c += 128) {
                size_t     ci0          = c / 64;
                size_t     ci1          = ci0 + 1;

                HVX_ಸದಿಶ v_s_minus_m0_0 = Q6_Vqf16_vsub_VhfVhf(my_ಅಡ್ಡಸಾಲು_buf0[ci0], v_dup_m0);
                HVX_ಸದಿಶ v_s_minus_m1_0 = Q6_Vqf16_vsub_VhfVhf(my_ಅಡ್ಡಸಾಲು_buf1[ci0], v_dup_m1);
                HVX_ಸದಿಶ v_s_minus_m0_1 = Q6_Vqf16_vsub_VhfVhf(my_ಅಡ್ಡಸಾಲು_buf0[ci1], v_dup_m0);
                HVX_ಸದಿಶ v_s_minus_m1_1 = Q6_Vqf16_vsub_VhfVhf(my_ಅಡ್ಡಸಾಲು_buf1[ci1], v_dup_m1);

                HVX_ಸದಿಶ v_p_row0_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_0));
                HVX_ಸದಿಶ v_p_row1_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_0));
                HVX_ಸದಿಶ v_p_row0_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_1));
                HVX_ಸದಿಶ v_p_row1_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_1));

                __fp16 *     ಹೊರಹೋಗು_dtile0  = p_st_ತಳ + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                __fp16 *     ಹೊರಹೋಗು_dtile1  = p_st_ತಳ + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ಸದಿಶ * pv_p_out0_0  = ((HVX_ಸದಿಶ *) ಹೊರಹೋಗು_dtile0) + r1 / 2;
                HVX_ಸದಿಶ * pv_p_out1_0  = pv_p_out0_0 + 16;
                HVX_ಸದಿಶ * pv_p_out0_1  = ((HVX_ಸದಿಶ *) ಹೊರಹೋಗು_dtile1) + r1 / 2;
                HVX_ಸದಿಶ * pv_p_out1_1  = pv_p_out0_1 + 16;

                HVX_ಸದಿಶ_Pair vp_p_dual0 = Q6_W_vshuff_VVR(v_p_row1_hf_0, v_p_row0_hf_0, -2);
                *pv_p_out0_0               = Q6_V_lo_W(vp_p_dual0);
                *pv_p_out1_0               = Q6_V_hi_W(vp_p_dual0);

                HVX_ಸದಿಶ_Pair vp_p_dual1 = Q6_W_vshuff_VVR(v_p_row1_hf_1, v_p_row0_hf_1, -2);
                *pv_p_out0_1               = Q6_V_lo_W(vp_p_dual1);
                *pv_p_out1_1               = Q6_V_hi_W(vp_p_dual1);

                HVX_ಸದಿಶ_Pair vp_p0_0 = hvx_vec_f16_ಗಮ್ಯ_f32_shuff(v_p_row0_hf_0);
                HVX_ಸದಿಶ_Pair vp_p1_0 = hvx_vec_f16_ಗಮ್ಯ_f32_shuff(v_p_row1_hf_0);
                HVX_ಸದಿಶ_Pair vp_p0_1 = hvx_vec_f16_ಗಮ್ಯ_f32_shuff(v_p_row0_hf_1);
                HVX_ಸದಿಶ_Pair vp_p1_1 = hvx_vec_f16_ಗಮ್ಯ_f32_shuff(v_p_row1_hf_1);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_0), Q6_V_hi_W(vp_p0_0)));
                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_1), Q6_V_hi_W(vp_p0_1)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_0), Q6_V_hi_W(vp_p1_0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_1), Q6_V_hi_W(vp_p1_1)));
            }
            ಗಾಗಿ (size_t c_rem = c; c_rem < kv_ಅಡ್ಡಸಾಲುಗಳು; c_rem += 64) {
                size_t     ci           = c_rem / 64;
                HVX_ಸದಿಶ v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(my_ಅಡ್ಡಸಾಲು_buf0[ci], v_dup_m0);
                HVX_ಸದಿಶ v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(my_ಅಡ್ಡಸಾಲು_buf1[ci], v_dup_m1);

                HVX_ಸದಿಶ v_p_row0_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0));
                HVX_ಸದಿಶ v_p_row1_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1));
                __fp16 *     ಹೊರಹೋಗು_dtile  = p_st_ತಳ + ci * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ಸದಿಶ * pv_p_out0  = ((HVX_ಸದಿಶ *) ಹೊರಹೋಗು_dtile) + r1 / 2;
                HVX_ಸದಿಶ * pv_p_out1  = pv_p_out0 + 16;

                HVX_ಸದಿಶ_Pair vp_p_dual = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
                *pv_p_out0               = Q6_V_lo_W(vp_p_dual);
                *pv_p_out1               = Q6_V_hi_W(vp_p_dual);

                HVX_ಸದಿಶ_Pair vp_p0 = hvx_vec_f16_ಗಮ್ಯ_f32_shuff(v_p_row0_hf);
                HVX_ಸದಿಶ_Pair vp_p1 = hvx_vec_f16_ಗಮ್ಯ_f32_shuff(v_p_row1_hf);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0), Q6_V_hi_W(vp_p0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1), Q6_V_hi_W(vp_p1)));
            }

            HVX_ಸದಿಶ rowsum0_sf = hvx_vec_reduce_ಮೊತ್ತ_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum0));
            HVX_ಸದಿಶ rowsum1_sf = hvx_vec_reduce_ಮೊತ್ತ_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum1));
            {
                HVX_ಸದಿಶ rv0_v = hvx_vec_f32_ಗಮ್ಯ_f16(rowsum0_sf, rowsum0_sf);
                HVX_ಸದಿಶ rv1_v = hvx_vec_f32_ಗಮ್ಯ_f16(rowsum1_sf, rowsum1_sf);

                HVX_ಸದಿಶ_Pred p_ಆರಂಭ = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_ಸದಿಶ_Pred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_ಸದಿಶ_Pred p_ಕೊನೆ   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_ಸದಿಶ_Pred p_lane0 = Q6_Q_and_QQn(p_mid, p_ಆರಂಭ);
                HVX_ಸದಿಶ_Pred p_lane1 = Q6_Q_and_QQn(p_ಕೊನೆ, p_mid);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane0, rv0_v, rowsum_acc_v);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane1, rv1_v, rowsum_acc_v);
            }
        }

        // Inline fa_ml_update_and_build_d for this vector (lock-free and in parallel)
        HVX_ಸದಿಶ_Pair rowmax_acc_pair    = hvx_vec_f16_ಗಮ್ಯ_f32(rowmax_acc_v);
        HVX_ಸದಿಶ     v_rowmax_acc_f32_0 = Q6_V_lo_W(rowmax_acc_pair);
        HVX_ಸದಿಶ     v_rowmax_acc_f32_1 = Q6_V_hi_W(rowmax_acc_pair);

        HVX_ಸದಿಶ v_m_curr0 = Q6_Vsf_vmax_VsfVsf(m_prev_v0, v_rowmax_acc_f32_0);
        HVX_ಸದಿಶ v_m_curr1 = Q6_Vsf_vmax_VsfVsf(m_prev_v1, v_rowmax_acc_f32_1);

        HVX_ಸದಿಶ v_m_diff0 = HVX_OP_SUB_F32(m_prev_v0, v_m_curr0);
        HVX_ಸದಿಶ v_m_diff1 = HVX_OP_SUB_F32(m_prev_v1, v_m_curr1);

        HVX_ಸದಿಶ v_m_diff_f16   = hvx_vec_f32_ಗಮ್ಯ_f16(v_m_diff0, v_m_diff1);
        HVX_ಸದಿಶ exp_m_diff_f16 = hvx_vec_exp2_f16(v_m_diff_f16);

        HVX_ಸದಿಶ_Pair exp_m_diff_pair = hvx_vec_f16_ಗಮ್ಯ_f32(exp_m_diff_f16);
        HVX_ಸದಿಶ exp_m_diff0 = Q6_V_lo_W(exp_m_diff_pair);
        HVX_ಸದಿಶ exp_m_diff1 = Q6_V_hi_W(exp_m_diff_pair);

        HVX_ಸದಿಶ_Pair rowsum_acc_pair = hvx_vec_f16_ಗಮ್ಯ_f32(rowsum_acc_v);
        HVX_ಸದಿಶ     v_rowsum_acc_f32_0 = Q6_V_lo_W(rowsum_acc_pair);
        HVX_ಸದಿಶ     v_rowsum_acc_f32_1 = Q6_V_hi_W(rowsum_acc_pair);

        HVX_ಸದಿಶ l_prev_v0 = factx->vtcm_l_vec[r_vec_idx * 2 + 0];
        HVX_ಸದಿಶ l_prev_v1 = factx->vtcm_l_vec[r_vec_idx * 2 + 1];
        HVX_ಸದಿಶ v_l_curr0 = HVX_OP_ಕೂಡಿಸು_F32(HVX_OP_MUL_F32(l_prev_v0, exp_m_diff0), v_rowsum_acc_f32_0);
        HVX_ಸದಿಶ v_l_curr1 = HVX_OP_ಕೂಡಿಸು_F32(HVX_OP_MUL_F32(l_prev_v1, exp_m_diff1), v_rowsum_acc_f32_1);

        factx->vtcm_m_vec[r_vec_idx * 2 + 0] = v_m_curr0;
        factx->vtcm_m_vec[r_vec_idx * 2 + 1] = v_m_curr1;
        factx->vtcm_l_vec[r_vec_idx * 2 + 0] = v_l_curr0;
        factx->vtcm_l_vec[r_vec_idx * 2 + 1] = v_l_curr1;

        // Build diagonal tile D = diag(exp(m_diff))
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ     v_offsets = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) d_tile_scatter_offsets;
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred q_32_ಮುಖವಾಡ = Q6_Q_vsetq_R(32 * ಗಾತ್ರ(__fp16));
        HVX_ಸದಿಶ           v_exp_m_diff = exp_m_diff_f16;

        __fp16 * ಸ್ಥಿರಾಂಕ d_tiles_ಹೊರಹೋಗು = factx->vtcm_d_tiles[ವಾದಗಳು_2->buf_idx];

        size_t t0 = r_vec_idx * 2;
        ಆದರೆ (t0 < ವಾದಗಳು_2->n_ಅಡ್ಡಸಾಲು_tiles) {
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_content = v_exp_m_diff;
            __fp16 *         ಹೊರಹೋಗು_ತಳ  = d_tiles_ಹೊರಹೋಗು + t0 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_ಮುಖವಾಡ, (size_t) ಹೊರಹೋಗು_ತಳ, HMX_FP16_TILE_ಗಾತ್ರ - 1, v_offsets, v_content);
        }

        size_t t1 = r_vec_idx * 2 + 1;
        ಆದರೆ (t1 < ವಾದಗಳು_2->n_ಅಡ್ಡಸಾಲು_tiles) {
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_content = Q6_V_vror_VR(v_exp_m_diff, 64);
            __fp16 *         ಹೊರಹೋಗು_ತಳ  = d_tiles_ಹೊರಹೋಗು + t1 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_ಮುಖವಾಡ, (size_t) ಹೊರಹೋಗು_ತಳ, HMX_FP16_TILE_ಗಾತ್ರ - 1, v_offsets, v_content);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (ವಾದಗಳು_2->q_ಆರಂಭ * G + vec_ಆರಂಭ * 64));
}

ಸ್ಥಿರ ಶೂನ್ಯ fa_softmax_thread_nomask(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    fa_softmax_impl(n, i, ದತ್ತಾಂಶ,
                    /*has_mask=*/false,
                    /*mask_broadcast=*/false,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ಸ್ಥಿರ ಶೂನ್ಯ fa_softmax_thread_ಮುಖವಾಡ_ಪ್ರಸಾರ_g1(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    fa_softmax_impl(n, i, ದತ್ತಾಂಶ,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/true,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ಸ್ಥಿರ ಶೂನ್ಯ fa_softmax_thread_ಮುಖವಾಡ_ಪ್ರಸಾರ_gn(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    fa_softmax_impl(n, i, ದತ್ತಾಂಶ,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ಸ್ಥಿರ ಶೂನ್ಯ fa_softmax_thread(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    fa_softmax_ವಾದಗಳು_t *     ವಾದಗಳು_2  = (fa_softmax_ವಾದಗಳು_t *) ದತ್ತಾಂಶ;
    ರಚನೆ hmx_fa_context * factx = ವಾದಗಳು_2->factx;

    ಸ್ಥಿರಾಂಕ bool has_ಮುಖವಾಡ       = (ವಾದಗಳು_2->ಮುಖವಾಡ != ಖಾಲಿ);
    ಸ್ಥಿರಾಂಕ bool ಮುಖವಾಡ_ಪ್ರಸಾರ = factx->ಮುಖವಾಡ_ಪ್ರಸಾರ;
    ಸ್ಥಿರಾಂಕ bool is_g1          = (ವಾದಗಳು_2->G == 1);
    ಸ್ಥಿರಾಂಕ bool has_alibi      = ವಾದಗಳು_2->has_alibi;
    ಸ್ಥಿರಾಂಕ bool has_softcap    = factx->has_softcap;

    fa_softmax_impl(n, i, ದತ್ತಾಂಶ, has_ಮುಖವಾಡ, ಮುಖವಾಡ_ಪ್ರಸಾರ, is_g1, has_alibi, has_softcap);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_fa_ತುಂಬು_sinks(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * sinks_ದತ್ತಾಂಶ, size_t idx, size_t ಎಣಿಕೆ_2) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * v_src    = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) sinks_ದತ್ತಾಂಶ + (idx >> 5);
    ಸ್ಥಿರಾಂಕ size_t       lane_off = idx & 31;
    ಸ್ಥಿರಾಂಕ size_t       byte_off = lane_off * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಹಿಂತಿರುಗಿ (lane_off + ಎಣಿಕೆ_2 > 32)
        ? Q6_V_valign_VVR(v_src[1], v_src[0], byte_off)
        : Q6_V_vror_VR(v_src[0], byte_off);
}

ಸ್ಥಿರ __attribute__((noinline)) ಶೂನ್ಯ fa_build_d_diag_inv_l(ರಚನೆ hmx_fa_context * factx,
                                                            size_t                  n_ಅಡ್ಡಸಾಲು_tiles,
                                                            size_t                  n_ಅಡ್ಡಸಾಲು_tiles_g_br,
                                                            uint32_t                kv_ತಲೆ,
                                                            size_t                  n_ಅಡ್ಡಸಾಲುಗಳು_g) {
    (ಶೂನ್ಯ) n_ಅಡ್ಡಸಾಲು_tiles_g_br;

    // sinks: l += exp(sink - m * ln2) in f32, m and l are in the log2 domain
    ಆದರೆ (factx->sinks) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *    sinks_ದತ್ತಾಂಶ   = factx->vtcm_sinks;
        ಸ್ಥಿರಾಂಕ size_t     G            = factx->G;
        ಸ್ಥಿರಾಂಕ size_t     ತಲೆ_ತಳ    = (size_t) kv_ತಲೆ * G;
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_ln2        = Q6_V_vsplat_R(EXP_LOGN2);
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_ತಲೆ_sinks = (G <= 32) ? hvx_fa_ತುಂಬು_sinks(sinks_ದತ್ತಾಂಶ, ತಲೆ_ತಳ, G) : Q6_V_vzero();
        HVX_ಸದಿಶ       v_sink       = Q6_V_vzero();
        size_t           g_prev       = SIZE_MAX;

        ಗಾಗಿ (size_t i = 0; i < n_ಅಡ್ಡಸಾಲು_tiles; ++i) {
            // the sink pattern of a tile only depends on (i * 32) % G, rebuild it when that changes
            ಸ್ಥಿರಾಂಕ size_t r0 = i * 32;
            ಸ್ಥಿರಾಂಕ size_t g0 = fastmodulo(r0, G, &factx->div_G);
            ಆದರೆ (g0 != g_prev) {
                ಸ್ಥಿರಾಂಕ size_t rem_g = G - g0;
                ಆದರೆ (G <= 32) {
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_lo = Q6_V_vror_VR(v_ತಲೆ_sinks, g0 * ಗಾತ್ರ(ದಶಮಾಂಶ));
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_hi = Q6_V_vror_VR(v_ತಲೆ_sinks, 128 - rem_g * ಗಾತ್ರ(ದಶಮಾಂಶ));
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(rem_g * ಗಾತ್ರ(ದಶಮಾಂಶ)), v_lo, v_hi);
                    ಗಾಗಿ (size_t len = G; len < 32; len <<= 1) {
                        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred q_keep    = Q6_Q_vsetq_R(len * ಗಾತ್ರ(ದಶಮಾಂಶ));
                        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ     v_shifted = Q6_V_vror_VR(v_sink, 128 - len * ಗಾತ್ರ(ದಶಮಾಂಶ));
                        v_sink = Q6_V_vmux_QVV(q_keep, v_sink, v_shifted);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (rem_g >= 32) {
                    v_sink = hvx_fa_ತುಂಬು_sinks(sinks_ದತ್ತಾಂಶ, ತಲೆ_ತಳ + g0, 32);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_lo = hvx_fa_ತುಂಬು_sinks(sinks_ದತ್ತಾಂಶ, ತಲೆ_ತಳ + g0, rem_g);
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_hi = hvx_fa_ತುಂಬು_sinks(sinks_ದತ್ತಾಂಶ, ತಲೆ_ತಳ, 32 - rem_g);
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq_R(rem_g * ಗಾತ್ರ(ದಶಮಾಂಶ)), v_lo, Q6_V_vror_VR(v_hi, 128 - rem_g * ಗಾತ್ರ(ದಶಮಾಂಶ)));
                }
                g_prev = g0;
            }

            HVX_ಸದಿಶ v_d = HVX_OP_SUB_F32(v_sink, HVX_OP_MUL_F32(factx->vtcm_m_vec[i], v_ln2));
            ಆದರೆ (r0 + 32 > n_ಅಡ್ಡಸಾಲುಗಳು_g) {
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred q_valid = Q6_Q_vsetq_R((n_ಅಡ್ಡಸಾಲುಗಳು_g - r0) * ಗಾತ್ರ(ದಶಮಾಂಶ));
                v_d = Q6_V_vmux_QVV(q_valid, v_d, Q6_V_vzero());
            }
            factx->vtcm_l_vec[i] = HVX_OP_ಕೂಡಿಸು_F32(factx->vtcm_l_vec[i], hvx_vec_exp_f32(v_d));
        }
    }

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ     v_offsets = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) d_tile_scatter_offsets;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred q_32_ಮುಖವಾಡ = Q6_Q_vsetq_R(32 * ಗಾತ್ರ(__fp16));
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ     one       = hvx_vec_splat_f32(1.0f);

    HVX_ಸದಿಶ v_content = Q6_V_vzero();
    ಗಾಗಿ (size_t i = 0; i < n_ಅಡ್ಡಸಾಲು_tiles; ++i) {
        ಆದರೆ ((i % 2) == 0) {
            HVX_ಸದಿಶ inv_lo = HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i]));
            HVX_ಸದಿಶ inv_hi = (i + 1 < n_ಅಡ್ಡಸಾಲು_tiles) ? HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i + 1])) : Q6_V_vzero();
            v_content = hvx_vec_f32_ಗಮ್ಯ_f16(inv_lo, inv_hi);
        } ಇಲ್ಲದಿದ್ದರೆ {
            v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 * ಹೊರಹೋಗು_ತಳ = factx->vtcm_d_inv_l + i * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_ಮುಖವಾಡ, (size_t) ಹೊರಹೋಗು_ತಳ, HMX_FP16_TILE_ಗಾತ್ರ - 1, v_offsets, v_content);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ fa_phase_softmax_and_build_d(ರಚನೆ hmx_fa_context * factx,
                                         fa_softmax_ವಾದಗಳು_t *     sargs,
                                         size_t                  n_ಅಡ್ಡಸಾಲು_tiles,
                                         size_t                  n_ಅಡ್ಡಸಾಲು_tiles_g_br) {
    work_ಸರತಿ_t wp = factx->octx->ctx->work_ಸರತಿ;
    ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲು_vec_cnt = hmx_ceil_div(sargs->n_ಅಡ್ಡಸಾಲುಗಳು_g, 64);

    ಕೆಲಸಗಾರ_callback_t softmax_fn = fa_softmax_thread;
    ಆದರೆ (sargs->ಮುಖವಾಡ == ಖಾಲಿ && !factx->has_softcap && !sargs->has_alibi) {
        softmax_fn = fa_softmax_thread_nomask;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sargs->ಮುಖವಾಡ != ಖಾಲಿ && factx->ಮುಖವಾಡ_ಪ್ರಸಾರ && !factx->has_softcap && !sargs->has_alibi) {
        ಆದರೆ (sargs->G == 1) {
            softmax_fn = fa_softmax_thread_ಮುಖವಾಡ_ಪ್ರಸಾರ_g1;
        } ಇಲ್ಲದಿದ್ದರೆ {
            softmax_fn = fa_softmax_thread_ಮುಖವಾಡ_ಪ್ರಸಾರ_gn;
        }
    }

    ಆದರೆ (factx->n_ಎಳೆಗಳು > 1 && n_ಅಡ್ಡಸಾಲು_vec_cnt >= 2) {
        uint32_t n_use = (uint32_t) hex_smin((size_t) factx->n_ಎಳೆಗಳು, n_ಅಡ್ಡಸಾಲು_vec_cnt);
        sargs->thread_div = init_fastdiv_ಮೌಲ್ಯಗಳು(n_use);
        work_ಸರತಿ_ಸತತ(wp, softmax_fn, sargs, n_use);
    } ಇಲ್ಲದಿದ್ದರೆ {
        softmax_fn(1, 0, sargs);
    }
}

// ============================================================================
// HMX job structs and worker functions
// ============================================================================

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ಸ್ಥಿರಾಂಕ __fp16 * q_tiles;
    ಸ್ಥಿರಾಂಕ __fp16 * k_tiles;
    __fp16 *       s_tiles;
    size_t         n_ಅಡ್ಡಸಾಲು_tiles;
    size_t         n_ಲಂಬಸಾಲು_tiles;
    size_t         n_dot_tiles;  // DK / 32
    size_t         n_tiles_ಪ್ರತಿ_bc;
    uint8_t *      hmx_scales;
} hmx_fa_qk_ಕೆಲಸ_t;

ಸ್ಥಿರ ಶೂನ್ಯ hmx_fa_qk_dot_ಕೆಲಸಗಾರ(ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    hmx_fa_qk_ಕೆಲಸ_t * ಕೆಲಸ            = (hmx_fa_qk_ಕೆಲಸ_t *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ size_t      n_ಅಡ್ಡಸಾಲು_tiles    = ಕೆಲಸ->n_ಅಡ್ಡಸಾಲು_tiles;
    ಸ್ಥಿರಾಂಕ size_t      n_ಲಂಬಸಾಲು_tiles    = ಕೆಲಸ->n_ಲಂಬಸಾಲು_tiles;
    ಸ್ಥಿರಾಂಕ size_t      n_dot_tiles    = ಕೆಲಸ->n_dot_tiles;
    ಸ್ಥಿರಾಂಕ size_t      n_tiles_ಪ್ರತಿ_bc = ಕೆಲಸ->n_tiles_ಪ್ರತಿ_bc;
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ q_tiles  = ಕೆಲಸ->q_tiles;
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ k_tiles  = ಕೆಲಸ->k_tiles;
    __fp16 * ಸೀಮಿತ s_tiles        = ಕೆಲಸ->s_tiles;
    __builtin_assume(n_ಅಡ್ಡಸಾಲು_tiles > 0);
    __builtin_assume(n_ಲಂಬಸಾಲು_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    asm ಅಸ್ಥಿರ(HMX_ಗಣ_BIAS("%0") :: "r"((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)ಕೆಲಸ->hmx_scales));
    ಸ್ಥಿರಾಂಕ size_t dot_stride = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    ಗಾಗಿ (size_t r = 0; r < n_ಅಡ್ಡಸಾಲು_tiles; ++r) {
        ಸ್ಥಿರಾಂಕ __fp16 * ಅಡ್ಡಸಾಲು_tiles = q_tiles + r * dot_stride;
        ಸ್ಥಿರಾಂಕ __fp16 * ಲಂಬಸಾಲು_tiles = k_tiles;
        __fp16 *       ಹೊರಹೋಗು_tile  = s_tiles + r * n_tiles_ಪ್ರತಿ_bc * HMX_FP16_TILE_N_ELMS;

        ಗಾಗಿ (size_t c = 0; c < n_ಲಂಬಸಾಲು_tiles; ++c) {
            hmx_fa_qk_dot_tile(ಅಡ್ಡಸಾಲು_tiles, ಲಂಬಸಾಲು_tiles, ಹೊರಹೋಗು_tile, n_dot_tiles);
            ಲಂಬಸಾಲು_tiles += dot_stride;
            ಹೊರಹೋಗು_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    __fp16 *       o_curr;
    ಸ್ಥಿರಾಂಕ __fp16 * o_prev;
    ಸ್ಥಿರಾಂಕ __fp16 * p_tiles;
    ಸ್ಥಿರಾಂಕ __fp16 * v_tiles;
    ಸ್ಥಿರಾಂಕ __fp16 * d_tiles;
    uint8_t *      hmx_scales;
    size_t         n_ಅಡ್ಡಸಾಲು_tiles;
    size_t         n_ಲಂಬಸಾಲು_tiles;
    size_t         n_ಅಡ್ಡಸಾಲು_tiles_g_br;
    size_t         n_tiles_ಪ್ರತಿ_bc;
    size_t         DV;
} hmx_fa_o_update_ಕೆಲಸ_t;

ಸ್ಥಿರ ಶೂನ್ಯ hmx_fa_o_update_ಕೆಲಸಗಾರ(ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    hmx_fa_o_update_ಕೆಲಸ_t * ಕೆಲಸ              = (hmx_fa_o_update_ಕೆಲಸ_t *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ size_t            n_ಅಡ್ಡಸಾಲು_tiles      = ಕೆಲಸ->n_ಅಡ್ಡಸಾಲು_tiles;
    ಸ್ಥಿರಾಂಕ size_t            n_ಲಂಬಸಾಲು_tiles      = ಕೆಲಸ->n_ಲಂಬಸಾಲು_tiles;
    ಸ್ಥಿರಾಂಕ size_t            n_ಅಡ್ಡಸಾಲು_tiles_g_br = ಕೆಲಸ->n_ಅಡ್ಡಸಾಲು_tiles_g_br;
    ಸ್ಥಿರಾಂಕ size_t            n_tiles_ಪ್ರತಿ_bc   = ಕೆಲಸ->n_tiles_ಪ್ರತಿ_bc;
    ಸ್ಥಿರಾಂಕ size_t            DV_tiles         = ಕೆಲಸ->DV / 32;
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ d_tiles          = ಕೆಲಸ->d_tiles;
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ p_tiles          = ಕೆಲಸ->p_tiles;
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ v_tiles          = ಕೆಲಸ->v_tiles;
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ o_prev           = ಕೆಲಸ->o_prev;
    __fp16 * ಸೀಮಿತ o_curr                 = ಕೆಲಸ->o_curr;
    __builtin_assume(n_ಅಡ್ಡಸಾಲು_tiles > 0);
    __builtin_assume(n_ಲಂಬಸಾಲು_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm ಅಸ್ಥಿರ(HMX_ಗಣ_BIAS("%0") :: "r"((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)ಕೆಲಸ->hmx_scales));
    ಸ್ಥಿರಾಂಕ size_t o_stride = n_ಅಡ್ಡಸಾಲು_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    ಸ್ಥಿರಾಂಕ size_t v_stride = n_tiles_ಪ್ರತಿ_bc * HMX_FP16_TILE_N_ELMS;
    ಗಾಗಿ (size_t r = 0; r < n_ಅಡ್ಡಸಾಲು_tiles; ++r) {
        ಸ್ಥಿರಾಂಕ __fp16 * d_diag     = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        ಸ್ಥಿರಾಂಕ __fp16 * p_tile_ಒಳ  = p_tiles + (r * n_tiles_ಪ್ರತಿ_bc) * HMX_FP16_TILE_N_ELMS;
        ಸ್ಥಿರಾಂಕ __fp16 * o_rc       = o_prev + r * HMX_FP16_TILE_N_ELMS;
        ಸ್ಥಿರಾಂಕ __fp16 * v_tile_ಒಳ  = v_tiles;
        __fp16       * o_tile_ಹೊರಹೋಗು = o_curr + r * HMX_FP16_TILE_N_ELMS;

        ಗಾಗಿ (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_update_tile(d_diag, o_rc, p_tile_ಒಳ, v_tile_ಒಳ, o_tile_ಹೊರಹೋಗು, n_ಲಂಬಸಾಲು_tiles);
            o_rc       += o_stride;
            v_tile_ಒಳ  += v_stride;
            o_tile_ಹೊರಹೋಗು += o_stride;
        }
    }
}

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    __fp16 *       o_curr;   // output (row-major tile layout)
    ಸ್ಥಿರಾಂಕ __fp16 * o_prev;   // input (column-major tile layout)
    ಸ್ಥಿರಾಂಕ __fp16 * d_tiles;  // diag(1/l) tiles
    uint8_t *      hmx_scales;
    size_t         n_ಅಡ್ಡಸಾಲು_tiles;
    size_t         n_ಅಡ್ಡಸಾಲು_tiles_g_br;
    size_t         DV;
} hmx_fa_o_norm_ಕೆಲಸ_t;

ಸ್ಥಿರ ಶೂನ್ಯ hmx_fa_o_norm_ಕೆಲಸಗಾರ(ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    hmx_fa_o_norm_ಕೆಲಸ_t * ಕೆಲಸ              = (hmx_fa_o_norm_ಕೆಲಸ_t *) ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ size_t          n_ಅಡ್ಡಸಾಲು_tiles      = ಕೆಲಸ->n_ಅಡ್ಡಸಾಲು_tiles;
    ಸ್ಥಿರಾಂಕ size_t          n_ಅಡ್ಡಸಾಲು_tiles_g_br = ಕೆಲಸ->n_ಅಡ್ಡಸಾಲು_tiles_g_br;
    ಸ್ಥಿರಾಂಕ size_t          DV_tiles         = ಕೆಲಸ->DV / 32;
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ d_tiles        = ಕೆಲಸ->d_tiles;
    ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ o_prev         = ಕೆಲಸ->o_prev;
    __fp16 * ಸೀಮಿತ o_curr               = ಕೆಲಸ->o_curr;
    __builtin_assume(n_ಅಡ್ಡಸಾಲು_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm ಅಸ್ಥಿರ(HMX_ಗಣ_BIAS("%0") :: "r"((ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ)ಕೆಲಸ->hmx_scales));
    ಸ್ಥಿರಾಂಕ size_t o_stride = n_ಅಡ್ಡಸಾಲು_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    ಗಾಗಿ (size_t r = 0; r < n_ಅಡ್ಡಸಾಲು_tiles; ++r) {
        ಸ್ಥಿರಾಂಕ __fp16 * d_diag = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        ಸ್ಥಿರಾಂಕ __fp16 * o_rc = o_prev + r * HMX_FP16_TILE_N_ELMS;
        __fp16 *       o_ಹೊರಹೋಗು = o_curr + r * DV_tiles * HMX_FP16_TILE_N_ELMS;

        ಗಾಗಿ (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_norm_tile(d_diag, o_rc, o_ಹೊರಹೋಗು);
            o_rc  += o_stride;
            o_ಹೊರಹೋಗು += HMX_FP16_TILE_N_ELMS;
        }
    }
}

// Populate per-GQA-row ALiBi slopes for a given KV head.
ಸ್ಥಿರ __attribute__((noinline)) ಶೂನ್ಯ fa_compute_slopes(
                              ಸ್ಥಿರಾಂಕ ರಚನೆ hmx_fa_context * factx,
                              uint32_t                      kv_ತಲೆ,
                              size_t                        n_ಅಡ್ಡಸಾಲುಗಳು_g) {
    __fp16 * slopes = factx->vtcm_slopes;
    ಆದರೆ (factx->max_bias == 0.0f) {
        hvx_splat_f16_a(slopes, 1.0f, n_ಅಡ್ಡಸಾಲುಗಳು_g);
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint32_t G           = factx->G;
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ_log2 = factx->n_ತಲೆ_log2;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    m0          = factx->m0;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    m1          = factx->m1;

    __fp16 ತಾತ್ಕಾಲಿಕ_slopes[512] __attribute__((aligned(128)));
    ಆದರೆ (G <= 32) {
        // Fast path: Compute G unique slope values in vector registers
        HVX_ಸದಿಶ v_val = hvx_alibi_slopes(kv_ತಲೆ, G, n_ತಲೆ_log2, m0, m1);

        __fp16 ತಾತ್ಕಾಲಿಕ_slopes_aligned[64] __attribute__((aligned(128)));
        hvx_vmem(ತಾತ್ಕಾಲಿಕ_slopes_aligned) = hvx_vec_f32_ಗಮ್ಯ_f16(v_val, Q6_V_vzero());

        ಗಾಗಿ (uint32_t i = 0; i < G; ++i) {
            ತಾತ್ಕಾಲಿಕ_slopes[i] = ತಾತ್ಕಾಲಿಕ_slopes_aligned[i];
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Fallback path: G > 32 (rare configurations)
        ಗಾಗಿ (uint32_t i = 0; i < G; ++i) {
            ತಾತ್ಕಾಲಿಕ_slopes[i] = (__fp16)alibi_ಇಳಿಜಾರು(kv_ತಲೆ * G + i, n_ತಲೆ_log2, m0, m1);
        }
    }

    // Allocate stack buffer to avoid scalar writes to VTCM (which generates L2 misses)
    __fp16 local_slopes[n_ಅಡ್ಡಸಾಲುಗಳು_g] __attribute__((aligned(128)));
    ಗಾಗಿ (size_t r = 0; r < n_ಅಡ್ಡಸಾಲುಗಳು_g; ++r) {
        local_slopes[r] = ತಾತ್ಕಾಲಿಕ_slopes[fastmodulo(r, G, &factx->div_G)];
    }

    // Copy to VTCM slopes using HVX block copy (both are aligned to 128 bytes)
    hvx_copy_f16_aa((uint8_t *)slopes, (ಸ್ಥಿರಾಂಕ uint8_t *)local_slopes, n_ಅಡ್ಡಸಾಲುಗಳು_g);
}

ಸ್ಥಿರ ಶೂನ್ಯ fa_ತಳ್ಳು_ಮುಖವಾಡ_dma_gqa(
    dma_ಸರತಿ *               dma_q,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಮುಖವಾಡ,
    uint32_t                  q_ಆರಂಭ,
    uint32_t                  im3,
    uint32_t                  kv_ಆರಂಭ,
    uint32_t                  kv_ತಲೆ,
    uint32_t                  G,
    uint32_t                  m_ಸಾಲು_ಬೈಟುಗಳು,
    uint32_t                  kv_ಅಡ್ಡಸಾಲುಗಳು,
    uint32_t                  n_ಅಡ್ಡಸಾಲುಗಳು_q,
    ರಚನೆ hmx_fa_context *   factx
) {
    ಗಾಗಿ (uint32_t g = 0; g < G; ++g) {
        ಸ್ಥಿರಾಂಕ uint32_t h_idx = kv_ತಲೆ * G + g;
        ಸ್ಥಿರಾಂಕ uint32_t im2 = fastmodulo(h_idx, ಮುಖವಾಡ->ne[2], &factx->src3_div2);
        ಸ್ಥಿರಾಂಕ dma_addr_t ms_src = ಮುಖವಾಡ->ದತ್ತಾಂಶ + q_ಆರಂಭ * ಮುಖವಾಡ->nb[1] +
                                  im2 * ಮುಖವಾಡ->nb[2] + im3 * ಮುಖವಾಡ->nb[3] + kv_ಆರಂಭ * ಗಾತ್ರ(__fp16);
        uint8_t * ms_dst = (uint8_t *) factx->vtcm_ಮುಖವಾಡ_buf + g * m_ಸಾಲು_ಬೈಟುಗಳು;
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(ms_dst, ms_src), G * m_ಸಾಲು_ಬೈಟುಗಳು, ಮುಖವಾಡ->nb[1], kv_ಅಡ್ಡಸಾಲುಗಳು * ಗಾತ್ರ(__fp16), n_ಅಡ್ಡಸಾಲುಗಳು_q);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ fa_ತೆಗೆ_ಮುಖವಾಡ_dma_gqa(dma_ಸರತಿ * dma_q, uint32_t G) {
    ಗಾಗಿ (uint32_t g = 0; g < G; ++g) {
        dma_ಸರತಿ_ತೆಗೆ(dma_q);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ fa_prefetch_ಖಂಡ(dma_ಸರತಿ * dma_q, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * k, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * v, ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಮುಖವಾಡ,
                                     uint32_t b, size_t Bc, size_t ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded, size_t ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು, size_t ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded, size_t ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು,
                                     uint32_t ik2, uint32_t ik3, uint32_t iv2, uint32_t iv3, uint32_t q_ಆರಂಭ, uint32_t im3, uint32_t kv_ತಲೆ, uint32_t G,
                                     size_t m_ಸಾಲು_ಬೈಟುಗಳು, size_t n_ಅಡ್ಡಸಾಲುಗಳು_q, size_t nek1, size_t prefetch_buf, ರಚನೆ hmx_fa_context * factx) {
    ಸ್ಥಿರಾಂಕ uint32_t  prefetch_ಆರಂಭ = b * Bc;
    ಸ್ಥಿರಾಂಕ uint32_t  prefetch_ಅಡ್ಡಸಾಲುಗಳು  = hex_smin(Bc, nek1 - prefetch_ಆರಂಭ);
    ಸ್ಥಿರಾಂಕ dma_addr_t k_prefetch_src = k->ದತ್ತಾಂಶ + prefetch_ಆರಂಭ * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(factx->vtcm_k_fp16[prefetch_buf], k_prefetch_src), ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded, k->nb[1], ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು, prefetch_ಅಡ್ಡಸಾಲುಗಳು);
    ಸ್ಥಿರಾಂಕ dma_addr_t v_prefetch_src = v->ದತ್ತಾಂಶ + prefetch_ಆರಂಭ * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(factx->vtcm_v_fp16[prefetch_buf], v_prefetch_src), ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded, v->nb[1], ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು, prefetch_ಅಡ್ಡಸಾಲುಗಳು);

    ಆದರೆ (ಮುಖವಾಡ) {
        ಆದರೆ (__builtin_expect(factx->ಮುಖವಾಡ_ಪ್ರಸಾರ, true)) {
            ಸ್ಥಿರಾಂಕ dma_addr_t ms_src = ಮುಖವಾಡ->ದತ್ತಾಂಶ + q_ಆರಂಭ * ಮುಖವಾಡ->nb[1] + im3 * ಮುಖವಾಡ->nb[3] + prefetch_ಆರಂಭ * ಗಾತ್ರ(__fp16);
            dma_cache_fa_ತಳ್ಳು(dma_q, &factx->m_cache, ms_src, m_ಸಾಲು_ಬೈಟುಗಳು, ಮುಖವಾಡ->nb[1], prefetch_ಅಡ್ಡಸಾಲುಗಳು * ಗಾತ್ರ(__fp16), n_ಅಡ್ಡಸಾಲುಗಳು_q);
        } ಇಲ್ಲದಿದ್ದರೆ {
            fa_ತಳ್ಳು_ಮುಖವಾಡ_dma_gqa(dma_q, ಮುಖವಾಡ, q_ಆರಂಭ, im3, prefetch_ಆರಂಭ, kv_ತಲೆ, G, m_ಸಾಲು_ಬೈಟುಗಳು, prefetch_ಅಡ್ಡಸಾಲುಗಳು, n_ಅಡ್ಡಸಾಲುಗಳು_q, factx);
        }
    }
}

// ============================================================================
// Core HMX flash attention algorithm (GQA-merged)
// ============================================================================

ಪೂರ್ಣಾಂಕ hmx_flash_attn_ext(ರಚನೆ htp_ops_context * octx) {
    ರಚನೆ htp_thread_trace * tr_hvx = &octx->ctx->trace[0];
    ರಚನೆ htp_thread_trace * tr_hmx = &octx->ctx->trace[HTP_MAX_NTHREADS];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * q    = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * k    = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * v    = octx->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಮುಖವಾಡ = (octx->src[3] && octx->src[3]->ದತ್ತಾಂಶ) ? octx->src[3] : ಖಾಲಿ;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ರಚನೆ htp_context * ಸ್ಥಿರಾಂಕ ctx = octx->ctx;

    ಆದರೆ (!ctx->hmx_enabled) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    // Dimensions
    ಸ್ಥಿರಾಂಕ uint32_t neq0 = q->ne[0];  // head_dim (DK)
    ಸ್ಥಿರಾಂಕ uint32_t neq1 = q->ne[1];  // n_tokens
    ಸ್ಥಿರಾಂಕ uint32_t neq2 = q->ne[2];  // n_heads
    ಸ್ಥಿರಾಂಕ uint32_t neq3 = q->ne[3];  // n_seqs

    ಸ್ಥಿರಾಂಕ uint32_t nek0 = k->ne[0];  // head_dim
    ಸ್ಥಿರಾಂಕ uint32_t nek1 = k->ne[1];  // kv_len

    ಸ್ಥಿರಾಂಕ uint32_t nev0 = v->ne[0];  // head_dim (DV)

    ಸ್ಥಿರಾಂಕ uint32_t DK = neq0;
    ಸ್ಥಿರಾಂಕ uint32_t DV = nev0;

    // HMX tiles head_dim in units of 64. head_dim need not be 64- (or 32-) aligned:
    // we can operate on DK/DV rounded up to 64 with tail lanes [D, D_pad) zero-filled.
    ಸ್ಥಿರಾಂಕ uint32_t DK_pad = hex_ಸುತ್ತು_ಮೇಲೆ(DK, 64);
    ಸ್ಥಿರಾಂಕ uint32_t DV_pad = hex_ಸುತ್ತು_ಮೇಲೆ(DV, 64);
    ಆದರೆ (DK == 0 || DV == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_fa_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_fa_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ uint32_t n_kv_heads = k->ne[2];

    // ======== Build context ========
    ರಚನೆ hmx_fa_context factx;
    memset(&factx, 0, ಗಾತ್ರ(factx));
    factx.octx           = octx;
    factx.sinks          = octx->src[4];  // NULL if this op has no attention sinks
    factx.k              = k;
    factx.v              = v;
    factx.n_ಎಳೆಗಳು      = kparams->n_ಎಳೆಗಳು;
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
    factx.n_kv_ಖಂಡಗಳು    = kparams->n_kv_ಖಂಡಗಳು;
    factx.is_q_fp32      = (q->type == HTP_TYPE_F32);
    factx.is_dst_fp32    = (dst->type == HTP_TYPE_F32);
    factx.pipeline       = (kparams->u.hmx.pipeline != 0);
    factx.ಮುಖವಾಡ_ಪ್ರಸಾರ = (kparams->u.hmx.ಮುಖವಾಡ_ಪ್ರಸಾರ != 0);
    ಆದರೆ (ಮುಖವಾಡ) {
        factx.src3_div2  = kparams->src3_div2;
        factx.src3_div3  = kparams->src3_div3;
    }

    factx.has_softcap   = (kparams->logit_softcap != 0.0f);
    factx.scale         = (__fp16) kparams->scale;
    factx.max_bias      = kparams->max_bias;
    factx.logit_softcap = 0;
    ಆದರೆ (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_ತಲೆ_log2 = kparams->n_ತಲೆ_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    ಸ್ಥಿರಾಂಕ uint32_t Br = factx.Br;
    ಸ್ಥಿರಾಂಕ uint32_t Bc = factx.Bc;
    ಸ್ಥಿರಾಂಕ uint32_t g_br = factx.g_br;
    ಸ್ಥಿರಾಂಕ bool pipeline = factx.pipeline;
    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = factx.n_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ uint32_t G = factx.G;

    // Multi-device: prefer head-parallel partitioning (each core owns a disjoint head
    // shard), falling back to Q-block (token) split when heads don't divide evenly.
    ಸ್ಥಿರಾಂಕ uint32_t n_q_ಖಂಡಗಳು = (neq1 + Br - 1) / Br;
    uint32_t q_ಆರಂಭ_min  = 0;
    uint32_t q_ಆರಂಭ_max  = neq1;
    uint32_t kv_ತಲೆ_min  = 0;
    uint32_t kv_ತಲೆ_max  = n_kv_heads;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ uint32_t mdev_ಎಣಿಕೆ = octx->ctx->mdev.ಎಣಿಕೆ_2;
        ಸ್ಥಿರಾಂಕ uint32_t mdev_idx   = octx->ctx->mdev.idx;
        ಸ್ಥಿರಾಂಕ uint32_t dst_e_ಗಾತ್ರ = (dst->type == HTP_TYPE_F32) ? ಗಾತ್ರ(ದಶಮಾಂಶ) : ಗಾತ್ರ(__fp16);
        ಸ್ಥಿರಾಂಕ bool can_ವಿಭಜಿಸು      = htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(dst, dst_e_ಗಾತ್ರ);

        ಆದರೆ (kparams->ತಲೆ_ವಿಭಜಿಸು && can_ವಿಭಜಿಸು && n_kv_heads >= mdev_ಎಣಿಕೆ && n_kv_heads % mdev_ಎಣಿಕೆ == 0) {
            ಸ್ಥಿರಾಂಕ uint32_t kv_ಪ್ರತಿ_core = n_kv_heads / mdev_ಎಣಿಕೆ;
            kv_ತಲೆ_min = mdev_idx * kv_ಪ್ರತಿ_core;
            kv_ತಲೆ_max = kv_ತಲೆ_min + kv_ಪ್ರತಿ_core;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(n_q_ಖಂಡಗಳು, can_ವಿಭಜಿಸು ? 1 : 0, mdev_idx, mdev_ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
            ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಆರಂಭ = range.ಆರಂಭ;
            ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಕೊನೆ   = range.ಆರಂಭ + range.ಎಣಿಕೆ_2;

            ಆದರೆ (ಖಂಡ_ಆರಂಭ >= ಖಂಡ_ಕೊನೆ) {
                ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
            }

            q_ಆರಂಭ_min = ಖಂಡ_ಆರಂಭ * Br;
            q_ಆರಂಭ_max = MIN(ಖಂಡ_ಕೊನೆ * Br, neq1);
        }
    }

    // ======== VTCM allocation (GQA-aware) ========
    // K/V row sizes drive the DMA descriptors (not the VTCM layout) and are used
    // throughout the KV loop below. The DMA copies only the real DK/DV columns; the
    // staging rows are padded to hold DK_pad/DV_pad columns (tail zero-filled below)
    // so the HMX interleave/tile logic can operate on 64-aligned head dims.
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು        = htp_tensor_get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(k->type, DK);
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು        = htp_tensor_get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(v->type, DV);
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded = hex_ಸುತ್ತು_ಮೇಲೆ(DK_pad * ಗಾತ್ರ(__fp16), 128);
    ಸ್ಥಿರಾಂಕ size_t ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded = hex_ಸುತ್ತು_ಮೇಲೆ(DV_pad * ಗಾತ್ರ(__fp16), 128);

    // Build the VTCM layout once (shared with the host estimator) and place every
    // scratch buffer at its computed offset. Padded head dims size the HMX tiles.
    ರಚನೆ hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, G, DK_pad, DV_pad, Br, Bc, n_ಎಳೆಗಳು, pipeline, factx.is_q_fp32, factx.sinks != ಖಾಲಿ, factx.n_heads);

    ಆದರೆ (L.ಒಟ್ಟು_ಬೈಟುಗಳು > ctx->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    uint8_t * ಸ್ಥಿರಾಂಕ ತಳ = ctx->vtcm_ತಳ;

    factx.vtcm_q_dma          = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_q_dma);
    factx.vtcm_q_tiles        = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_q_tiles);
    factx.vtcm_o_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_o_tiles[0]);
    factx.vtcm_o_tiles[1]     = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_o_tiles[1]);
    factx.vtcm_k_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_k_fp16[0]);
    factx.vtcm_k_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_k_fp16[1]);
    factx.vtcm_v_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_v_fp16[0]);
    factx.vtcm_v_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_v_fp16[1]);
    factx.vtcm_k_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_k_tiles[0]);
    factx.vtcm_k_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ತಳ, L.off_k_tiles[1], pipeline);
    factx.vtcm_v_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_v_tiles[0]);
    factx.vtcm_v_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ತಳ, L.off_v_tiles[1], pipeline);
    factx.vtcm_s_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_s_tiles[0]);
    factx.vtcm_s_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ತಳ, L.off_s_tiles[1], pipeline);
    factx.vtcm_p_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_p_tiles[0]);
    factx.vtcm_p_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ತಳ, L.off_p_tiles[1], pipeline);
    factx.vtcm_d_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_d_tiles[0]);
    factx.vtcm_d_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ತಳ, L.off_d_tiles[1], pipeline);
    factx.vtcm_d_inv_l        = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_d_inv_l);
    factx.vtcm_m_vec          = VTCM_LAYOUT_PTR(HVX_ಸದಿಶ, ತಳ, L.off_m_vec);
    factx.vtcm_l_vec          = VTCM_LAYOUT_PTR(HVX_ಸದಿಶ, ತಳ, L.off_l_vec);
    factx.vtcm_s_rowmax       = VTCM_LAYOUT_PTR(HVX_ಸದಿಶ, ತಳ, L.off_s_rowmax);
    factx.vtcm_p_rowsum       = VTCM_LAYOUT_PTR(HVX_ಸದಿಶ, ತಳ, L.off_p_rowsum);
    factx.vtcm_ಅಡ್ಡಸಾಲು_bufs       = VTCM_LAYOUT_PTR(HVX_ಸದಿಶ, ತಳ, L.off_ಅಡ್ಡಸಾಲು_bufs);
    factx.ಅಡ್ಡಸಾಲು_buf_stride      = L.ಅಡ್ಡಸಾಲು_buf_stride;
    factx.vtcm_hmx_scales_id  = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_hmx_scales_id);
    factx.vtcm_hmx_scales_qk  = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_hmx_scales_qk);
    factx.vtcm_ಮುಖವಾಡ_buf       = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_ಮುಖವಾಡ_buf);
    factx.ಮುಖವಾಡ_buf_ಅಡ್ಡಸಾಲು_stride = L.ಮುಖವಾಡ_buf_ಅಡ್ಡಸಾಲು_stride;
    factx.q_tile_ಬೈಟುಗಳು        = L.q_tile_ಬೈಟುಗಳು;
    factx.k_tile_ಬೈಟುಗಳು        = L.k_tile_ಬೈಟುಗಳು;
    factx.o_tile_ಬೈಟುಗಳು        = L.o_tile_ಬೈಟುಗಳು;
    factx.ಲಂಬಸಾಲು_vec_ಬೈಟುಗಳು       = L.ಲಂಬಸಾಲು_vec_ಬೈಟುಗಳು;
    factx.d_tile_ಬೈಟುಗಳು        = L.d_tile_ಬೈಟುಗಳು;
    factx.vtcm_slopes         = VTCM_LAYOUT_PTR(__fp16, ತಳ, L.off_slopes);
    factx.vtcm_sinks          = VTCM_LAYOUT_PTR_OPTIONAL(ದಶಮಾಂಶ, ತಳ, L.off_sinks, factx.sinks != ಖಾಲಿ);

    ಸ್ಥಿರಾಂಕ size_t m_ಸಾಲು_ಬೈಟುಗಳು = L.m_ಸಾಲು_ಬೈಟುಗಳು;  // used by the mask DMAs in the KV loop

    dma_cache_fa_init(&factx.m_cache, (uint8_t *) factx.vtcm_ಮುಖವಾಡ_buf, L.m_buf_ಸ್ಥಾನ_ಬೈಟುಗಳು, HMX_FA_DMA_CACHE_ಗಾತ್ರ);

    // Head-dim padding: the K/V DMA staging buffers and the flat-Q buffer are laid out
    // with padded row strides (size_{k,v,q}_row_padded, covering D_pad columns) but the
    // DMA only writes the real D columns per row. Zero the whole staging buffers once up
    // front so tail lanes [D, D_pad) stay zero for all KV blocks. No-op when already aligned.
    ಆದರೆ (DK_pad != DK || DV_pad != DV) {
        ಸ್ಥಿರಾಂಕ size_t k_buf_ಬೈಟುಗಳು = (size_t) factx.Bc * ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded;
        ಸ್ಥಿರಾಂಕ size_t v_buf_ಬೈಟುಗಳು = (size_t) factx.Bc * ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded;
        hvx_splat_u8_a((ಅಕ್ಷರ *) factx.vtcm_k_fp16[0], 0, k_buf_ಬೈಟುಗಳು);
        hvx_splat_u8_a((ಅಕ್ಷರ *) factx.vtcm_k_fp16[1], 0, k_buf_ಬೈಟುಗಳು);
        hvx_splat_u8_a((ಅಕ್ಷರ *) factx.vtcm_v_fp16[0], 0, v_buf_ಬೈಟುಗಳು);
        hvx_splat_u8_a((ಅಕ್ಷರ *) factx.vtcm_v_fp16[1], 0, v_buf_ಬೈಟುಗಳು);
        // Flat-Q DMA scratch
        ಆದರೆ (factx.vtcm_q_dma) {
            ಸ್ಥಿರಾಂಕ size_t q_dma_ಬೈಟುಗಳು = hex_align_ಮೇಲೆ(factx.g_br * DK * (factx.is_q_fp32 ? ಗಾತ್ರ(ದಶಮಾಂಶ) : ಗಾತ್ರ(__fp16)), 128);
            hvx_splat_u8_a((ಅಕ್ಷರ *) factx.vtcm_q_dma, 0, q_dma_ಬೈಟುಗಳು);
        }
    }

    // ======== Initialize HMX output scales ========
    hmx_init_ಸ್ತಂಭ_scales(factx.vtcm_hmx_scales_id, Q6_V_vsplat_R(0x3c00)); // 1.0
    {
        ಒಕ್ಕೂಟ { __fp16 f; uint16_t u; } sc = { .f = factx.scale };
        hmx_init_ಸ್ತಂಭ_scales(factx.vtcm_hmx_scales_qk, Q6_V_vsplat_R(sc.u));
    }

    // ======== DMA setup ========
    dma_ಸರತಿ * ಸ್ಥಿರಾಂಕ dma_q = ctx->dma[0];

    ಆದರೆ (factx.sinks) {
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(factx.vtcm_sinks, factx.sinks->ದತ್ತಾಂಶ), L.sinks_ಬೈಟುಗಳು, 0, factx.sinks->ಗಾತ್ರ_2, 1);
        dma_ಸರತಿ_ತೆಗೆ(dma_q);
    }

    ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲು_tiles_g_br = g_br / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ size_t n_tiles_ಪ್ರತಿ_bc   = Bc / HMX_FP16_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;

    ಸ್ಥಿರಾಂಕ size_t qo_element_ಗಾತ್ರ = factx.is_q_fp32 ? ಗಾತ್ರ(ದಶಮಾಂಶ) : ಗಾತ್ರ(__fp16);

    ಸ್ಥಿರಾಂಕ bool q_transposed                 = q->nb[1] < q->nb[2];
    ಸ್ಥಿರಾಂಕ size_t q_src_stride               = q_transposed ? q->nb[2] : q->nb[1];
    ಸ್ಥಿರಾಂಕ size_t q_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_untransposed   = factx.G * factx.DK * qo_element_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t q_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_trans_ಗುಣಕ   = factx.DK * qo_element_ಗಾತ್ರ;

    // ======== Reusable job descriptors for pipeline ========
    hmx_fa_qk_ಕೆಲಸ_t       qk_ಕೆಲಸ;
    hmx_fa_o_update_ಕೆಲಸ_t ou_ಕೆಲಸ;
    hmx_fa_o_norm_ಕೆಲಸ_t   on_ಕೆಲಸ;

    // ======== Main loop ========
    ಗಾಗಿ (uint32_t ib3 = 0; ib3 < neq3; ++ib3) {
        ಸ್ಥಿರಾಂಕ uint32_t im3 = ಮುಖವಾಡ ? fastmodulo(ib3, ಮುಖವಾಡ->ne[3], &factx.src3_div3) : 0;
        ಗಾಗಿ (uint32_t q_ಆರಂಭ = q_ಆರಂಭ_min; q_ಆರಂಭ < q_ಆರಂಭ_max; q_ಆರಂಭ += Br) {
            ಸ್ಥಿರಾಂಕ uint32_t n_ಅಡ್ಡಸಾಲುಗಳು_q    = hex_smin(Br, neq1 - q_ಆರಂಭ);
            ಸ್ಥಿರಾಂಕ size_t   n_ಅಡ್ಡಸಾಲುಗಳು_g    = n_ಅಡ್ಡಸಾಲುಗಳು_q * G;
            ಸ್ಥಿರಾಂಕ size_t   g_br_actual = hex_align_ಮೇಲೆ(n_ಅಡ್ಡಸಾಲುಗಳು_g, HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು);
            ಸ್ಥಿರಾಂಕ size_t   n_ಅಡ್ಡಸಾಲು_tiles = g_br_actual / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;

            uint16_t ಸಕ್ರಿಯ_ಖಂಡಗಳು[FA_MAX_KV_ಖಂಡಗಳು];
            uint32_t n_ಸಕ್ರಿಯ_ಖಂಡಗಳು = factx.n_kv_ಖಂಡಗಳು;
            ಆದರೆ (ಮುಖವಾಡ && factx.ಮುಖವಾಡ_ಪ್ರಸಾರ && factx.n_kv_ಖಂಡಗಳು > 1) {
                ಸ್ಥಿರಾಂಕ size_t scratch_ಬೈಟುಗಳು = factx.k_tile_ಬೈಟುಗಳು * (factx.pipeline ? 2 : 1);
                n_ಸಕ್ರಿಯ_ಖಂಡಗಳು = fa_ಪರಿಶೀಲನೆ_ಸಕ್ರಿಯ_ಖಂಡಗಳು(
                    dma_q, ಮುಖವಾಡ, q_ಆರಂಭ, n_ಅಡ್ಡಸಾಲುಗಳು_q, 0, im3, nek1, Bc,
                    factx.n_kv_ಖಂಡಗಳು, factx.vtcm_k_tiles[0], scratch_ಬೈಟುಗಳು, ಸಕ್ರಿಯ_ಖಂಡಗಳು);
            } ಇಲ್ಲದಿದ್ದರೆ {
                #ಸೂಚನೆ clang loop vectorize(disable)
                ಗಾಗಿ (uint32_t b = 0; b < factx.n_kv_ಖಂಡಗಳು; ++b) {
                    ಸಕ್ರಿಯ_ಖಂಡಗಳು[b] = (uint16_t) b;
                }
            }
            ಸ್ಥಿರಾಂಕ uint32_t b0       = (n_ಸಕ್ರಿಯ_ಖಂಡಗಳು > 0) ? ಸಕ್ರಿಯ_ಖಂಡಗಳು[0] : 0;
            ಸ್ಥಿರಾಂಕ uint32_t b0_ಆರಂಭ = b0 * Bc;
            ಸ್ಥಿರಾಂಕ uint32_t b0_ಅಡ್ಡಸಾಲುಗಳು  = hex_smin(Bc, nek1 - b0_ಆರಂಭ);

            ಗಾಗಿ (uint32_t kv_ತಲೆ = kv_ತಲೆ_min; kv_ತಲೆ < kv_ತಲೆ_max; ++kv_ತಲೆ) {
                ಸ್ಥಿರಾಂಕ uint32_t ik2 = kv_ತಲೆ;
                ಸ್ಥಿರಾಂಕ uint32_t ik3 = fastdiv(ib3, &kparams->ಪ್ರಸಾರ_rk3);
                ಸ್ಥಿರಾಂಕ uint32_t iv2 = kv_ತಲೆ;
                ಸ್ಥಿರಾಂಕ uint32_t iv3 = fastdiv(ib3, &kparams->ಪ್ರಸಾರ_rv3);

                // 1. Push Q and KV DMAs for the first kv_head of this q_start block.
                // Subsequent kv_head iterations are enqueued early at the end of the previous iteration.
                ಆದರೆ (kv_ತಲೆ == kv_ತಲೆ_min) {
                    ಸ್ಥಿರಾಂಕ dma_addr_t q_ptr = q->ದತ್ತಾಂಶ + q_ಆರಂಭ * q->nb[1] +
                                            (kv_ತಲೆ * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    ಸ್ಥಿರಾಂಕ size_t q_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = q_transposed ? n_ಅಡ್ಡಸಾಲುಗಳು_q * q_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_trans_ಗುಣಕ : q_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_untransposed;
                    ಸ್ಥಿರಾಂಕ size_t n_ಅಡ್ಡಸಾಲುಗಳು      = q_transposed ? factx.G : n_ಅಡ್ಡಸಾಲುಗಳು_q;
                    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(factx.vtcm_q_dma, q_ptr), q_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, hex_smax(q_src_stride, q_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು), q_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, n_ಅಡ್ಡಸಾಲುಗಳು);

                    ಆದರೆ (n_ಸಕ್ರಿಯ_ಖಂಡಗಳು > 0) {
                        ಸ್ಥಿರಾಂಕ dma_addr_t k_src = k->ದತ್ತಾಂಶ + b0_ಆರಂಭ * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(factx.vtcm_k_fp16[0], k_src), ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded, k->nb[1], ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು, b0_ಅಡ್ಡಸಾಲುಗಳು);

                        ಸ್ಥಿರಾಂಕ dma_addr_t v_src = v->ದತ್ತಾಂಶ + b0_ಆರಂಭ * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(factx.vtcm_v_fp16[0], v_src), ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded, v->nb[1], ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು, b0_ಅಡ್ಡಸಾಲುಗಳು);

                        ಆದರೆ (factx.pipeline && ಮುಖವಾಡ) {
                            ಆದರೆ (__builtin_expect(factx.ಮುಖವಾಡ_ಪ್ರಸಾರ, true)) {
                                ಸ್ಥಿರಾಂಕ dma_addr_t ms_src = ಮುಖವಾಡ->ದತ್ತಾಂಶ + q_ಆರಂಭ * ಮುಖವಾಡ->nb[1] + im3 * ಮುಖವಾಡ->nb[3] + b0_ಆರಂಭ * ಗಾತ್ರ(__fp16);
                                dma_cache_fa_ತಳ್ಳು(dma_q, &factx.m_cache, ms_src, m_ಸಾಲು_ಬೈಟುಗಳು, ಮುಖವಾಡ->nb[1], b0_ಅಡ್ಡಸಾಲುಗಳು * ಗಾತ್ರ(__fp16), n_ಅಡ್ಡಸಾಲುಗಳು_q);
                            } ಇಲ್ಲದಿದ್ದರೆ {
                                fa_ತಳ್ಳು_ಮುಖವಾಡ_dma_gqa(dma_q, ಮುಖವಾಡ, q_ಆರಂಭ, im3, b0_ಆರಂಭ, kv_ತಲೆ, G, m_ಸಾಲು_ಬೈಟುಗಳು, b0_ಅಡ್ಡಸಾಲುಗಳು, n_ಅಡ್ಡಸಾಲುಗಳು_q, &factx);
                            }
                        }
                    }
                }

                // 2. Pop Q DMA (blocks until Q is loaded)
                dma_ಸರತಿ_ತೆಗೆ(dma_q);

                // ---- Load Q block & Initialize per-block state ----
                fa_phase_q_ತುಂಬು(&factx, q, q_ಆರಂಭ, kv_ತಲೆ, ib3, n_ಅಡ್ಡಸಾಲುಗಳು_g);

                __fp16 * o_tile_prev = factx.vtcm_o_tiles[0];
                __fp16 * o_tile_curr = factx.vtcm_o_tiles[1];

                // ---- KV block loop with DMA double-buffering ----
                size_t buf_idx = 0;

                htp_trace_event_ಆರಂಭ(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_ಆರಂಭ);
                fa_compute_slopes(&factx, kv_ತಲೆ, n_ಅಡ್ಡಸಾಲುಗಳು_g);
                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_ಆರಂಭ);

                ಸ್ಥಿರಾಂಕ size_t k_src_stride = ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded / ಗಾತ್ರ(__fp16);
                ಸ್ಥಿರಾಂಕ size_t v_src_stride = ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded / ಗಾತ್ರ(__fp16);

                hmx_ಸರತಿ_t hmx_q = ctx->hmx_ಸರತಿ;

                ಆದರೆ (factx.pipeline) {
                    // Double-buffered job structs because HMX queue runs asynchronously
                    hmx_fa_qk_ಕೆಲಸ_t qk_ಕೆಲಸ[2];
                    hmx_fa_o_update_ಕೆಲಸ_t ou_ಕೆಲಸ[2];

                    // Prefetch second active block early if there are multiple active blocks
                    ಆದರೆ (n_ಸಕ್ರಿಯ_ಖಂಡಗಳು > 1) {
                        fa_prefetch_ಖಂಡ(dma_q, k, v, ಮುಖವಾಡ, ಸಕ್ರಿಯ_ಖಂಡಗಳು[1], Bc, ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded, ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು, ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded, ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು,
                                          ik2, ik3, iv2, iv3, q_ಆರಂಭ, im3, kv_ತಲೆ, G, m_ಸಾಲು_ಬೈಟುಗಳು, n_ಅಡ್ಡಸಾಲುಗಳು_q, nek1, 1, &factx);
                    }

                    // Prep and start QK-dot(0)
                    ಶೂನ್ಯ * curr_k0 = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
                    fa_phase_k_interleave(&factx, b0_ಅಡ್ಡಸಾಲುಗಳು, k_src_stride, curr_k0, b0_ಆರಂಭ, 0);

                    qk_ಕೆಲಸ[0].q_tiles        = factx.vtcm_q_tiles;
                    qk_ಕೆಲಸ[0].k_tiles        = factx.vtcm_k_tiles[0];
                    qk_ಕೆಲಸ[0].s_tiles        = factx.vtcm_s_tiles[0];
                    qk_ಕೆಲಸ[0].n_ಅಡ್ಡಸಾಲು_tiles    = n_ಅಡ್ಡಸಾಲು_tiles;
                    qk_ಕೆಲಸ[0].n_ಲಂಬಸಾಲು_tiles    = hmx_ceil_div(b0_ಅಡ್ಡಸಾಲುಗಳು, HMX_FP16_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                    qk_ಕೆಲಸ[0].n_dot_tiles    = DK_pad / 32;
                    qk_ಕೆಲಸ[0].n_tiles_ಪ್ರತಿ_bc = n_tiles_ಪ್ರತಿ_bc;
                    qk_ಕೆಲಸ[0].hmx_scales     = factx.vtcm_hmx_scales_qk;
                    hmx_ಸರತಿ_ತಳ್ಳು(hmx_q, hmx_ಸರತಿ_make_desc(hmx_fa_qk_dot_ಕೆಲಸಗಾರ, &qk_ಕೆಲಸ[0]));

                    ಗಾಗಿ (uint32_t ಹೆಜ್ಜೆ = 0; ಹೆಜ್ಜೆ < n_ಸಕ್ರಿಯ_ಖಂಡಗಳು; ++ಹೆಜ್ಜೆ) {
                        ಸ್ಥಿರಾಂಕ uint32_t kv_blk      = ಸಕ್ರಿಯ_ಖಂಡಗಳು[ಹೆಜ್ಜೆ];
                        ಸ್ಥಿರಾಂಕ uint32_t kv_ಆರಂಭ    = kv_blk * Bc;
                        ಸ್ಥಿರಾಂಕ uint32_t kv_ಅಡ್ಡಸಾಲುಗಳು     = hex_smin(Bc, nek1 - kv_ಆರಂಭ);
                        ಸ್ಥಿರಾಂಕ size_t   n_ಲಂಬಸಾಲು_tiles = hmx_ceil_div(kv_ಅಡ್ಡಸಾಲುಗಳು, HMX_FP16_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

                        // ---- 1. Pop and run V-prep for current block ----
                        ಶೂನ್ಯ * curr_v = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_ಅಡ್ಡಸಾಲುಗಳು, v_src_stride, curr_v, factx.vtcm_v_tiles[buf_idx], n_tiles_ಪ್ರತಿ_bc, kv_ಆರಂಭ);

                        // ---- 2. Pop and run mask-prep for current block ----
                        __fp16 * current_ಮುಖವಾಡ_vtcm = ಖಾಲಿ;
                        ಆದರೆ (ಮುಖವಾಡ) {
                            ಆದರೆ (__builtin_expect(factx.ಮುಖವಾಡ_ಪ್ರಸಾರ, true)) {
                                current_ಮುಖವಾಡ_vtcm = (__fp16 *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
                            } ಇಲ್ಲದಿದ್ದರೆ {
                                fa_ತೆಗೆ_ಮುಖವಾಡ_dma_gqa(dma_q, G);
                                current_ಮುಖವಾಡ_vtcm = factx.vtcm_ಮುಖವಾಡ_buf;
                            }
                        }

                        // ---- 3. Start HMX O update for step - 1 (reads P[1 - buf_idx], V[1 - buf_idx], D) ----
                        // O update relies on the previous block's P and V tiles.
                        // O update MUST be pushed before the next block's QK-dot: hmx_queue_pop() retires the
                        // oldest descriptor, so push order alone decides which pop waits for which job.
                        // If OU went in after QK(i+1), the pop below would retire QK(i+1) and leave
                        // OU(i-1) in flight into the next iteration, where V-prep overwrites V[prev_buf].
                        ಆದರೆ (ಹೆಜ್ಜೆ > 0) {
                            ಸ್ಥಿರಾಂಕ uint32_t prev_blk      = ಸಕ್ರಿಯ_ಖಂಡಗಳು[ಹೆಜ್ಜೆ - 1];
                            ಸ್ಥಿರಾಂಕ size_t prev_buf        = 1 - buf_idx;
                            ou_ಕೆಲಸ[prev_buf].o_curr      = o_tile_curr;
                            ou_ಕೆಲಸ[prev_buf].o_prev      = o_tile_prev;
                            ou_ಕೆಲಸ[prev_buf].p_tiles     = factx.vtcm_p_tiles[prev_buf];
                            ou_ಕೆಲಸ[prev_buf].v_tiles     = factx.vtcm_v_tiles[prev_buf];
                            ou_ಕೆಲಸ[prev_buf].d_tiles     = factx.vtcm_d_tiles[prev_buf];
                            ou_ಕೆಲಸ[prev_buf].hmx_scales  = factx.vtcm_hmx_scales_id;
                            ou_ಕೆಲಸ[prev_buf].n_ಅಡ್ಡಸಾಲು_tiles = n_ಅಡ್ಡಸಾಲು_tiles;
                            ou_ಕೆಲಸ[prev_buf].n_ಲಂಬಸಾಲು_tiles =
                                hmx_ceil_div(hex_smin(Bc, nek1 - prev_blk * Bc), HMX_FP16_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                            ou_ಕೆಲಸ[prev_buf].n_ಅಡ್ಡಸಾಲು_tiles_g_br = n_ಅಡ್ಡಸಾಲು_tiles_g_br;
                            ou_ಕೆಲಸ[prev_buf].n_tiles_ಪ್ರತಿ_bc   = n_tiles_ಪ್ರತಿ_bc;
                            ou_ಕೆಲಸ[prev_buf].DV               = DV_pad;
                            hmx_ಸರತಿ_ತಳ್ಳು(hmx_q, hmx_ಸರತಿ_make_desc(hmx_fa_o_update_ಕೆಲಸಗಾರ, &ou_ಕೆಲಸ[prev_buf]));
                        }

                        // ---- 4. Pop and run K-prep for next active block & push next QK-dot ----
                        ಆದರೆ (ಹೆಜ್ಜೆ + 1 < n_ಸಕ್ರಿಯ_ಖಂಡಗಳು) {
                            ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_blk   = ಸಕ್ರಿಯ_ಖಂಡಗಳು[ಹೆಜ್ಜೆ + 1];
                            ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಆರಂಭ = ಮುಂದಿನ_blk * Bc;
                            ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಅಡ್ಡಸಾಲುಗಳು  = hex_smin(Bc, nek1 - ಮುಂದಿನ_ಆರಂಭ);
                            ಸ್ಥಿರಾಂಕ size_t   ಮುಂದಿನ_buf   = 1 - buf_idx;

                            ಶೂನ್ಯ * ಮುಂದಿನ_k = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
                            fa_phase_k_interleave(&factx, ಮುಂದಿನ_ಅಡ್ಡಸಾಲುಗಳು, k_src_stride, ಮುಂದಿನ_k, ಮುಂದಿನ_ಆರಂಭ, ಮುಂದಿನ_buf);

                            qk_ಕೆಲಸ[ಮುಂದಿನ_buf].q_tiles        = factx.vtcm_q_tiles;
                            qk_ಕೆಲಸ[ಮುಂದಿನ_buf].k_tiles        = factx.vtcm_k_tiles[ಮುಂದಿನ_buf];
                            qk_ಕೆಲಸ[ಮುಂದಿನ_buf].s_tiles        = factx.vtcm_s_tiles[ಮುಂದಿನ_buf];
                            qk_ಕೆಲಸ[ಮುಂದಿನ_buf].n_ಅಡ್ಡಸಾಲು_tiles    = n_ಅಡ್ಡಸಾಲು_tiles;
                            qk_ಕೆಲಸ[ಮುಂದಿನ_buf].n_ಲಂಬಸಾಲು_tiles    = hmx_ceil_div(ಮುಂದಿನ_ಅಡ್ಡಸಾಲುಗಳು, HMX_FP16_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                            qk_ಕೆಲಸ[ಮುಂದಿನ_buf].n_dot_tiles    = DK_pad / 32;
                            qk_ಕೆಲಸ[ಮುಂದಿನ_buf].n_tiles_ಪ್ರತಿ_bc = n_tiles_ಪ್ರತಿ_bc;
                            qk_ಕೆಲಸ[ಮುಂದಿನ_buf].hmx_scales     = factx.vtcm_hmx_scales_qk;
                            hmx_ಸರತಿ_ತಳ್ಳು(hmx_q, hmx_ಸರತಿ_make_desc(hmx_fa_qk_dot_ಕೆಲಸಗಾರ, &qk_ಕೆಲಸ[ಮುಂದಿನ_buf]));
                        }

                        // ---- 5. Wait for current block's QK-dot to finish ----
                        hmx_ಸರತಿ_ತೆಗೆ(hmx_q);

                        // ---- 6. Phase 2: softmax + build_D ----
                        fa_softmax_ವಾದಗಳು_t sargs;
                        memset(&sargs, 0, ಗಾತ್ರ(sargs));
                        sargs.factx                = &factx;
                        sargs.buf_idx              = buf_idx;
                        sargs.kv_ಅಡ್ಡಸಾಲುಗಳು              = kv_ಅಡ್ಡಸಾಲುಗಳು;
                        sargs.n_ಅಡ್ಡಸಾಲುಗಳು_g             = n_ಅಡ್ಡಸಾಲುಗಳು_g;
                        sargs.n_ಲಂಬಸಾಲು_tiles          = n_ಲಂಬಸಾಲು_tiles;
                        sargs.n_tiles_ಪ್ರತಿ_bc       = n_tiles_ಪ್ರತಿ_bc;
                        sargs.n_ಅಡ್ಡಸಾಲು_tiles          = n_ಅಡ್ಡಸಾಲು_tiles;
                        sargs.n_ಅಡ್ಡಸಾಲು_tiles_g_br     = n_ಅಡ್ಡಸಾಲು_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_ತಲೆ              = kv_ತಲೆ;
                        sargs.kv_ಆರಂಭ             = kv_ಆರಂಭ;
                        sargs.q_ಆರಂಭ              = q_ಆರಂಭ;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.ಮುಖವಾಡ                 = ಮುಖವಾಡ;
                        sargs.ಮುಖವಾಡ_vtcm            = current_ಮುಖವಾಡ_vtcm;
                        sargs.ಮುಖವಾಡ_vtcm_ಅಡ್ಡಸಾಲು_stride = factx.ಮುಖವಾಡ_buf_ಅಡ್ಡಸಾಲು_stride;
                        sargs.slopes               = factx.vtcm_slopes;

                        // Run Softmax on HVX (blocking call)
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_ಅಡ್ಡಸಾಲು_tiles, n_ಅಡ್ಡಸಾಲು_tiles_g_br);

                        // Wait for HMX O update for step - 1 to finish
                        ಆದರೆ (ಹೆಜ್ಜೆ > 0) {
                            hmx_ಸರತಿ_ತೆಗೆ(hmx_q);
                            hex_ಅದಲುಬದಲು_ptr((ಶೂನ್ಯ **) &o_tile_curr, (ಶೂನ್ಯ **) &o_tile_prev);
                        }

                        // Prefetch block step + 2
                        ಆದರೆ (ಹೆಜ್ಜೆ + 2 < n_ಸಕ್ರಿಯ_ಖಂಡಗಳು) {
                            fa_prefetch_ಖಂಡ(dma_q, k, v, ಮುಖವಾಡ, ಸಕ್ರಿಯ_ಖಂಡಗಳು[ಹೆಜ್ಜೆ + 2], Bc, ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded, ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು, ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded, ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು,
                                              ik2, ik3, iv2, iv3, q_ಆರಂಭ, im3, kv_ತಲೆ, G, m_ಸಾಲು_ಬೈಟುಗಳು, n_ಅಡ್ಡಸಾಲುಗಳು_q, nek1, buf_idx, &factx);
                        }

                        buf_idx = 1 - buf_idx;
                    }

                    // Epilogue
                    ಆದರೆ (n_ಸಕ್ರಿಯ_ಖಂಡಗಳು > 0) {
                        ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆಯ_blk = ಸಕ್ರಿಯ_ಖಂಡಗಳು[n_ಸಕ್ರಿಯ_ಖಂಡಗಳು - 1];
                        ಸ್ಥಿರಾಂಕ size_t ಕೊನೆಯ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ  = hmx_ceil_div(hex_smin(Bc, nek1 - ಕೊನೆಯ_blk * Bc), HMX_FP16_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
                        ou_ಕೆಲಸ[0].o_curr           = o_tile_curr;
                        ou_ಕೆಲಸ[0].o_prev           = o_tile_prev;
                        ou_ಕೆಲಸ[0].p_tiles          = factx.vtcm_p_tiles[1 - buf_idx];
                        ou_ಕೆಲಸ[0].v_tiles          = factx.vtcm_v_tiles[1 - buf_idx];
                        ou_ಕೆಲಸ[0].d_tiles          = factx.vtcm_d_tiles[1 - buf_idx];
                        ou_ಕೆಲಸ[0].hmx_scales       = factx.vtcm_hmx_scales_id;
                        ou_ಕೆಲಸ[0].n_ಅಡ್ಡಸಾಲು_tiles      = n_ಅಡ್ಡಸಾಲು_tiles;
                        ou_ಕೆಲಸ[0].n_ಲಂಬಸಾಲು_tiles      = ಕೊನೆಯ_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
                        ou_ಕೆಲಸ[0].n_ಅಡ್ಡಸಾಲು_tiles_g_br = n_ಅಡ್ಡಸಾಲು_tiles_g_br;
                        ou_ಕೆಲಸ[0].n_tiles_ಪ್ರತಿ_bc   = n_tiles_ಪ್ರತಿ_bc;
                        ou_ಕೆಲಸ[0].DV               = DV_pad;
                        hmx_ಸರತಿ_ತಳ್ಳು(hmx_q, hmx_ಸರತಿ_make_desc(hmx_fa_o_update_ಕೆಲಸಗಾರ, &ou_ಕೆಲಸ[0]));

                        // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                        htp_trace_event_ಆರಂಭ(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_ಆರಂಭ);
                        fa_build_d_diag_inv_l(&factx, n_ಅಡ್ಡಸಾಲು_tiles, n_ಅಡ್ಡಸಾಲು_tiles_g_br, kv_ತಲೆ, n_ಅಡ್ಡಸಾಲುಗಳು_g);
                        htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_ಆರಂಭ);
                        hmx_ಸರತಿ_ತೆಗೆ(hmx_q);

                        hex_ಅದಲುಬದಲು_ptr((ಶೂನ್ಯ **) &o_tile_curr, (ಶೂನ್ಯ **) &o_tile_prev);
                    }

                } ಇಲ್ಲದಿದ್ದರೆ {
                    // Fallback path
                    ಗಾಗಿ (uint32_t ಹೆಜ್ಜೆ = 0; ಹೆಜ್ಜೆ < n_ಸಕ್ರಿಯ_ಖಂಡಗಳು; ++ಹೆಜ್ಜೆ) {
                        ಸ್ಥಿರಾಂಕ uint32_t kv_blk      = ಸಕ್ರಿಯ_ಖಂಡಗಳು[ಹೆಜ್ಜೆ];
                        ಸ್ಥಿರಾಂಕ uint32_t kv_ಆರಂಭ    = kv_blk * Bc;
                        ಸ್ಥಿರಾಂಕ uint32_t kv_ಅಡ್ಡಸಾಲುಗಳು     = hex_smin(Bc, nek1 - kv_ಆರಂಭ);
                        ಸ್ಥಿರಾಂಕ size_t   n_ಲಂಬಸಾಲು_tiles = hmx_ceil_div(kv_ಅಡ್ಡಸಾಲುಗಳು, HMX_FP16_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

                        ಆದರೆ (ಮುಖವಾಡ) {
                            ಆದರೆ (__builtin_expect(factx.ಮುಖವಾಡ_ಪ್ರಸಾರ, true)) {
                                ಸ್ಥಿರಾಂಕ dma_addr_t ms_src = ಮುಖವಾಡ->ದತ್ತಾಂಶ + q_ಆರಂಭ * ಮುಖವಾಡ->nb[1] + im3 * ಮುಖವಾಡ->nb[3] + kv_ಆರಂಭ * ಗಾತ್ರ(__fp16);
                                dma_cache_fa_ತಳ್ಳು(dma_q, &factx.m_cache, ms_src, m_ಸಾಲು_ಬೈಟುಗಳು, ಮುಖವಾಡ->nb[1], kv_ಅಡ್ಡಸಾಲುಗಳು * ಗಾತ್ರ(__fp16), n_ಅಡ್ಡಸಾಲುಗಳು_q);
                            } ಇಲ್ಲದಿದ್ದರೆ {
                                fa_ತಳ್ಳು_ಮುಖವಾಡ_dma_gqa(dma_q, ಮುಖವಾಡ, q_ಆರಂಭ, im3, kv_ಆರಂಭ, kv_ತಲೆ, G, m_ಸಾಲು_ಬೈಟುಗಳು, kv_ಅಡ್ಡಸಾಲುಗಳು, n_ಅಡ್ಡಸಾಲುಗಳು_q, &factx);
                            }
                        }

                        ಆದರೆ (ಹೆಜ್ಜೆ + 1 < n_ಸಕ್ರಿಯ_ಖಂಡಗಳು) {
                            ಸ್ಥಿರಾಂಕ uint32_t  prefetch_blk   = ಸಕ್ರಿಯ_ಖಂಡಗಳು[ಹೆಜ್ಜೆ + 1];
                            ಸ್ಥಿರಾಂಕ uint32_t  prefetch_ಆರಂಭ = prefetch_blk * Bc;
                            ಸ್ಥಿರಾಂಕ uint32_t  prefetch_ಅಡ್ಡಸಾಲುಗಳು  = hex_smin(Bc, nek1 - prefetch_ಆರಂಭ);
                            ಸ್ಥಿರಾಂಕ size_t    prefetch_buf   = 1 - buf_idx;
                            ಸ್ಥಿರಾಂಕ dma_addr_t k_prefetch_src = k->ದತ್ತಾಂಶ + prefetch_ಆರಂಭ * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(factx.vtcm_k_fp16[prefetch_buf], k_prefetch_src), ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded, k->nb[1], ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು, prefetch_ಅಡ್ಡಸಾಲುಗಳು);
                            ಸ್ಥಿರಾಂಕ dma_addr_t v_prefetch_src = v->ದತ್ತಾಂಶ + prefetch_ಆರಂಭ * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(factx.vtcm_v_fp16[prefetch_buf], v_prefetch_src), ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded, v->nb[1], ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು, prefetch_ಅಡ್ಡಸಾಲುಗಳು);
                        }

                        // Wait for current K DMA and interleave
                        ಶೂನ್ಯ * curr_k = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
                        fa_phase_k_interleave(&factx, kv_ಅಡ್ಡಸಾಲುಗಳು, k_src_stride, curr_k, kv_ಆರಂಭ, 0);

                        {
                            qk_ಕೆಲಸ.q_tiles        = factx.vtcm_q_tiles;
                            qk_ಕೆಲಸ.k_tiles        = factx.vtcm_k_tiles[0];
                            qk_ಕೆಲಸ.s_tiles        = factx.vtcm_s_tiles[0];
                            qk_ಕೆಲಸ.n_ಅಡ್ಡಸಾಲು_tiles    = n_ಅಡ್ಡಸಾಲು_tiles;
                            qk_ಕೆಲಸ.n_ಲಂಬಸಾಲು_tiles    = n_ಲಂಬಸಾಲು_tiles;
                            qk_ಕೆಲಸ.n_dot_tiles    = (size_t) (DK_pad / 32);
                            qk_ಕೆಲಸ.n_tiles_ಪ್ರತಿ_bc = n_tiles_ಪ್ರತಿ_bc;
                            qk_ಕೆಲಸ.hmx_scales     = factx.vtcm_hmx_scales_qk;

                            hmx_ಸರತಿ_ತಳ್ಳು(ctx->hmx_ಸರತಿ, hmx_ಸರತಿ_make_desc(hmx_fa_qk_dot_ಕೆಲಸಗಾರ, &qk_ಕೆಲಸ));
                            hmx_ಸರತಿ_ತೆಗೆ(ctx->hmx_ಸರತಿ);
                        }

                        // Wait for current V DMA and interleave
                        ಶೂನ್ಯ * curr_v = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_ಅಡ್ಡಸಾಲುಗಳು, v_src_stride, curr_v, factx.vtcm_v_tiles[0], n_tiles_ಪ್ರತಿ_bc, kv_ಆರಂಭ);

                        // ---- Phase 3: softmax + build_D ----
                        __fp16 * current_ಮುಖವಾಡ_vtcm = ಖಾಲಿ;
                        ಆದರೆ (ಮುಖವಾಡ) {
                            ಆದರೆ (__builtin_expect(factx.ಮುಖವಾಡ_ಪ್ರಸಾರ, true)) {
                                current_ಮುಖವಾಡ_vtcm = (__fp16 *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;
                            } ಇಲ್ಲದಿದ್ದರೆ {
                                fa_ತೆಗೆ_ಮುಖವಾಡ_dma_gqa(dma_q, G);
                                current_ಮುಖವಾಡ_vtcm = factx.vtcm_ಮುಖವಾಡ_buf;
                            }
                        }

                        fa_softmax_ವಾದಗಳು_t sargs;
                        memset(&sargs, 0, ಗಾತ್ರ(sargs));
                        sargs.factx                = &factx;
                        sargs.kv_ಅಡ್ಡಸಾಲುಗಳು              = kv_ಅಡ್ಡಸಾಲುಗಳು;
                        sargs.n_ಅಡ್ಡಸಾಲುಗಳು_g             = n_ಅಡ್ಡಸಾಲುಗಳು_g;
                        sargs.n_ಲಂಬಸಾಲು_tiles          = n_ಲಂಬಸಾಲು_tiles;
                        sargs.n_tiles_ಪ್ರತಿ_bc       = n_tiles_ಪ್ರತಿ_bc;
                        sargs.n_ಅಡ್ಡಸಾಲು_tiles          = n_ಅಡ್ಡಸಾಲು_tiles;
                        sargs.n_ಅಡ್ಡಸಾಲು_tiles_g_br     = n_ಅಡ್ಡಸಾಲು_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_ತಲೆ              = kv_ತಲೆ;
                        sargs.kv_ಆರಂಭ             = kv_ಆರಂಭ;
                        sargs.q_ಆರಂಭ              = q_ಆರಂಭ;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.ಮುಖವಾಡ                 = ಮುಖವಾಡ;
                        sargs.ಮುಖವಾಡ_vtcm            = current_ಮುಖವಾಡ_vtcm;
                        sargs.ಮುಖವಾಡ_vtcm_ಅಡ್ಡಸಾಲು_stride = factx.ಮುಖವಾಡ_buf_ಅಡ್ಡಸಾಲು_stride;
                        sargs.slopes               = factx.vtcm_slopes;
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_ಅಡ್ಡಸಾಲು_tiles, n_ಅಡ್ಡಸಾಲು_tiles_g_br);

                        {
                            ou_ಕೆಲಸ.o_curr           = o_tile_curr;
                            ou_ಕೆಲಸ.o_prev           = o_tile_prev;
                            ou_ಕೆಲಸ.p_tiles          = factx.vtcm_p_tiles[0];
                            ou_ಕೆಲಸ.v_tiles          = factx.vtcm_v_tiles[0];
                            ou_ಕೆಲಸ.d_tiles          = factx.vtcm_d_tiles[0];
                            ou_ಕೆಲಸ.hmx_scales       = factx.vtcm_hmx_scales_id;
                            ou_ಕೆಲಸ.n_ಅಡ್ಡಸಾಲು_tiles      = n_ಅಡ್ಡಸಾಲು_tiles;
                            ou_ಕೆಲಸ.n_ಲಂಬಸಾಲು_tiles      = n_ಲಂಬಸಾಲು_tiles;
                            ou_ಕೆಲಸ.n_ಅಡ್ಡಸಾಲು_tiles_g_br = n_ಅಡ್ಡಸಾಲು_tiles_g_br;
                            ou_ಕೆಲಸ.n_tiles_ಪ್ರತಿ_bc   = n_tiles_ಪ್ರತಿ_bc;
                            ou_ಕೆಲಸ.DV               = DV_pad;

                            hmx_ಸರತಿ_ತಳ್ಳು(ctx->hmx_ಸರತಿ, hmx_ಸರತಿ_make_desc(hmx_fa_o_update_ಕೆಲಸಗಾರ, &ou_ಕೆಲಸ));
                            ಆದರೆ (ಹೆಜ್ಜೆ + 1 == n_ಸಕ್ರಿಯ_ಖಂಡಗಳು) {
                                // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                                htp_trace_event_ಆರಂಭ(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_ಆರಂಭ);
                                fa_build_d_diag_inv_l(&factx, n_ಅಡ್ಡಸಾಲು_tiles, n_ಅಡ್ಡಸಾಲು_tiles_g_br, kv_ತಲೆ, n_ಅಡ್ಡಸಾಲುಗಳು_g);
                                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_ಆರಂಭ);
                            }
                            hmx_ಸರತಿ_ತೆಗೆ(ctx->hmx_ಸರತಿ);

                            hex_ಅದಲುಬದಲು_ptr((ಶೂನ್ಯ **) &o_tile_curr, (ಶೂನ್ಯ **) &o_tile_prev);
                        }

                        buf_idx = 1 - buf_idx;
                    }
                }

                // Enqueue DMAs for the next kv_head early so they overlap with O-PROC
                ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_kv_ತಲೆ = kv_ತಲೆ + 1;
                ಸ್ಥಿರಾಂಕ bool has_ಮುಂದಿನ = (ಮುಂದಿನ_kv_ತಲೆ < kv_ತಲೆ_max);

                ಆದರೆ (has_ಮುಂದಿನ) {
                    ಸ್ಥಿರಾಂಕ dma_addr_t ಮುಂದಿನ_q_ptr = q->ದತ್ತಾಂಶ + q_ಆರಂಭ * q->nb[1] + (ಮುಂದಿನ_kv_ತಲೆ * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    ಸ್ಥಿರಾಂಕ size_t ಮುಂದಿನ_q_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = q_transposed ? n_ಅಡ್ಡಸಾಲುಗಳು_q * q_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_trans_ಗುಣಕ : q_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು_untransposed;
                    ಸ್ಥಿರಾಂಕ size_t ಮುಂದಿನ_n_ಅಡ್ಡಸಾಲುಗಳು      = q_transposed ? factx.G : n_ಅಡ್ಡಸಾಲುಗಳು_q;
                    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(factx.vtcm_q_dma, ಮುಂದಿನ_q_ptr), ಮುಂದಿನ_q_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, hex_smax(q_src_stride, ಮುಂದಿನ_q_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು), ಮುಂದಿನ_q_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, ಮುಂದಿನ_n_ಅಡ್ಡಸಾಲುಗಳು);

                    ಆದರೆ (n_ಸಕ್ರಿಯ_ಖಂಡಗಳು > 0) {
                        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ik2 = ಮುಂದಿನ_kv_ತಲೆ;
                        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_iv2 = ಮುಂದಿನ_kv_ತಲೆ;

                        ಸ್ಥಿರಾಂಕ dma_addr_t ಮುಂದಿನ_k_src = k->ದತ್ತಾಂಶ + b0_ಆರಂಭ * k->nb[1] + ಮುಂದಿನ_ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(factx.vtcm_k_fp16[0], ಮುಂದಿನ_k_src), ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded, k->nb[1], ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು, b0_ಅಡ್ಡಸಾಲುಗಳು);

                        ಸ್ಥಿರಾಂಕ dma_addr_t ಮುಂದಿನ_v_src = v->ದತ್ತಾಂಶ + b0_ಆರಂಭ * v->nb[1] + ಮುಂದಿನ_iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(factx.vtcm_v_fp16[0], ಮುಂದಿನ_v_src), ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded, v->nb[1], ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು, b0_ಅಡ್ಡಸಾಲುಗಳು);

                        ಆದರೆ (factx.pipeline && ಮುಖವಾಡ) {
                            ಆದರೆ (__builtin_expect(factx.ಮುಖವಾಡ_ಪ್ರಸಾರ, true)) {
                                ಸ್ಥಿರಾಂಕ dma_addr_t ms_src = ಮುಖವಾಡ->ದತ್ತಾಂಶ + q_ಆರಂಭ * ಮುಖವಾಡ->nb[1] + im3 * ಮುಖವಾಡ->nb[3] + b0_ಆರಂಭ * ಗಾತ್ರ(__fp16);
                                dma_cache_fa_ತಳ್ಳು(dma_q, &factx.m_cache, ms_src, m_ಸಾಲು_ಬೈಟುಗಳು, ಮುಖವಾಡ->nb[1], b0_ಅಡ್ಡಸಾಲುಗಳು * ಗಾತ್ರ(__fp16), n_ಅಡ್ಡಸಾಲುಗಳು_q);
                            } ಇಲ್ಲದಿದ್ದರೆ {
                                fa_ತಳ್ಳು_ಮುಖವಾಡ_dma_gqa(dma_q, ಮುಖವಾಡ, q_ಆರಂಭ, im3, b0_ಆರಂಭ, ಮುಂದಿನ_kv_ತಲೆ, G, m_ಸಾಲು_ಬೈಟುಗಳು, b0_ಅಡ್ಡಸಾಲುಗಳು, n_ಅಡ್ಡಸಾಲುಗಳು_q, &factx);
                            }
                        }
                    }
                }

                // ---- Final normalization ----
                {
                    on_ಕೆಲಸ.o_curr           = o_tile_curr;
                    on_ಕೆಲಸ.o_prev           = o_tile_prev;
                    on_ಕೆಲಸ.d_tiles          = factx.vtcm_d_inv_l;
                    on_ಕೆಲಸ.hmx_scales       = factx.vtcm_hmx_scales_id;
                    on_ಕೆಲಸ.n_ಅಡ್ಡಸಾಲು_tiles      = n_ಅಡ್ಡಸಾಲು_tiles;
                    on_ಕೆಲಸ.n_ಅಡ್ಡಸಾಲು_tiles_g_br = n_ಅಡ್ಡಸಾಲು_tiles_g_br;
                    on_ಕೆಲಸ.DV               = DV_pad;
                    hmx_ಸರತಿ_ತಳ್ಳು(ctx->hmx_ಸರತಿ, hmx_ಸರತಿ_make_desc(hmx_fa_o_norm_ಕೆಲಸಗಾರ, &on_ಕೆಲಸ));
                    hmx_ಸರತಿ_ತೆಗೆ(ctx->hmx_ಸರತಿ);
                }

                // ---- Store O block ----
                fa_phase_o_store(&factx, dst, o_tile_curr, q_ಆರಂಭ, kv_ತಲೆ, ib3, n_ಅಡ್ಡಸಾಲುಗಳು_g);
            }
        }
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಪೂರ್ಣಾಂಕ op_flash_attn_ext(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * q    = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * k    = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * v    = octx->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * ಮುಖವಾಡ = octx->src[3];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    // Check support
    ಆದರೆ ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) ||
        (k->type != HTP_TYPE_F16 && k->type != HTP_TYPE_Q8_0) ||
        (v->type != HTP_TYPE_F16 && v->type != HTP_TYPE_Q8_0)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (htp_tensor_is_extended(dst)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_fa_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_fa_kernel_params *) octx->kernel_params;

    ಆದರೆ (kparams->kernel_type == HTP_FA_KERNEL_UNSUPPORTED || kparams->n_kv_ಖಂಡಗಳು > FA_MAX_KV_ಖಂಡಗಳು) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, kparams->n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಆದರೆ (kparams->kernel_type == HTP_FA_KERNEL_HMX) {
        ಹಿಂತಿರುಗಿ hmx_flash_attn_ext(octx);
    }

    ರಚನೆ htp_fa_context factx;
    factx.octx = octx;
    factx.k = k;
    factx.v = v;

    factx.src0_div21 = kparams->u.hvx.src0_div21;
    factx.src0_div1  = kparams->u.hvx.src0_div1;

    factx.ಪ್ರಸಾರ_rk2 = kparams->ಪ್ರಸಾರ_rk2;
    factx.ಪ್ರಸಾರ_rk3 = kparams->ಪ್ರಸಾರ_rk3;
    factx.ಪ್ರಸಾರ_rv2 = kparams->ಪ್ರಸಾರ_rv2;
    factx.ಪ್ರಸಾರ_rv3 = kparams->ಪ್ರಸಾರ_rv3;

    ಆದರೆ (ಮುಖವಾಡ) {
        factx.src3_div2 = kparams->src3_div2;
        factx.src3_div3 = kparams->src3_div3;
    }

    factx.is_q_fp32 = (q->type == HTP_TYPE_F32);
    factx.ಗಾತ್ರ_q_ಅಡ್ಡಸಾಲು_padded = kparams->u.hvx.ಗಾತ್ರ_q_ಅಡ್ಡಸಾಲು_padded;
    factx.ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded = kparams->u.hvx.ಗಾತ್ರ_k_ಅಡ್ಡಸಾಲು_padded;
    factx.ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded = kparams->u.hvx.ಗಾತ್ರ_v_ಅಡ್ಡಸಾಲು_padded;

    factx.n_ಖಂಡಗಳು = kparams->n_kv_ಖಂಡಗಳು;

    factx.scale = kparams->scale;
    factx.max_bias = kparams->max_bias;
    factx.has_softcap = (kparams->logit_softcap != 0.0f);
    factx.logit_softcap = 0;
    ಆದರೆ (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_ತಲೆ_log2 = kparams->n_ತಲೆ_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ = q->ne[2];
    ಆದರೆ (n_ತಲೆ > 512) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }
    ಗಾಗಿ (uint32_t h = 0; h < n_ತಲೆ; ++h) {
        factx.slopes[h] = (__fp16) ((kparams->max_bias > 0.0f) ? alibi_ಇಳಿಜಾರು(h, factx.n_ತಲೆ_log2, factx.m0, factx.m1) : 1.0f);
    }

    // total rows in q
    ಸ್ಥಿರಾಂಕ uint32_t neq1 = q->ne[1];
    ಸ್ಥಿರಾಂಕ uint32_t neq2 = q->ne[2];
    ಸ್ಥಿರಾಂಕ uint32_t neq3 = q->ne[3];
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_qrows = neq1 * neq2 * neq3;

    uint32_t qrow_ಆರಂಭ = 0;
    uint32_t qrows      = ಒಟ್ಟು_qrows;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ uint32_t mdev_ಎಣಿಕೆ = octx->ctx->mdev.ಎಣಿಕೆ_2;
        ಸ್ಥಿರಾಂಕ uint32_t mdev_idx   = octx->ctx->mdev.idx;
        ಸ್ಥಿರಾಂಕ uint32_t n_kv_heads = k->ne[2];
        ಸ್ಥಿರಾಂಕ uint32_t dst_e_ಗಾತ್ರ = (dst->type == HTP_TYPE_F32) ? ಗಾತ್ರ(ದಶಮಾಂಶ) : ಗಾತ್ರ(__fp16);
        ಸ್ಥಿರಾಂಕ bool can_ವಿಭಜಿಸು      = htp_tensor_can_ಅಡ್ಡಸಾಲು_ವಿಭಜನೆ(dst, dst_e_ಗಾತ್ರ);

        // head range is contiguous in flat row space only when neq3 == 1
        ಆದರೆ (kparams->ತಲೆ_ವಿಭಜಿಸು && can_ವಿಭಜಿಸು && neq3 == 1 && n_kv_heads >= mdev_ಎಣಿಕೆ && n_kv_heads % mdev_ಎಣಿಕೆ == 0) {
            ಸ್ಥಿರಾಂಕ uint32_t G              = kparams->G;
            ಸ್ಥಿರಾಂಕ uint32_t kv_ಪ್ರತಿ_core    = n_kv_heads / mdev_ಎಣಿಕೆ;
            ಸ್ಥಿರಾಂಕ uint32_t heads_ಪ್ರತಿ_core = kv_ಪ್ರತಿ_core * G;
            ಸ್ಥಿರಾಂಕ uint32_t ತಲೆ_ಆರಂಭ     = mdev_idx * heads_ಪ್ರತಿ_core;
            qrow_ಆರಂಭ = ತಲೆ_ಆರಂಭ * neq1;
            qrows      = heads_ಪ್ರತಿ_core * neq1;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(ಒಟ್ಟು_qrows, can_ವಿಭಜಿಸು ? 1 : 0, mdev_idx, mdev_ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
            qrow_ಆರಂಭ = range.ಆರಂಭ;
            qrows      = range.ಎಣಿಕೆ_2;
        }
    }

    ಆದರೆ (qrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

    factx.qrows            = qrows;
    factx.qrow_ಆರಂಭ       = qrow_ಆರಂಭ;
    factx.qrows_ಪ್ರತಿ_thread = fastdiv(qrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);

    ಸ್ಥಿರಾಂಕ bool has_ಮುಖವಾಡ  = (ಮುಖವಾಡ != ಖಾಲಿ);
    ಸ್ಥಿರಾಂಕ bool has_sinks = (octx->src[4] != ಖಾಲಿ);
    ರಚನೆ hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, k->ne[0], v->ne[0], factx.is_q_fp32, has_ಮುಖವಾಡ, has_sinks, n_ತಲೆ, n_ಎಳೆಗಳು);

    ಆದರೆ (L.ಒಟ್ಟು_ಬೈಟುಗಳು > octx->ctx->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    factx.ಗಾತ್ರ_q_ಖಂಡ = L.ಗಾತ್ರ_q_ಖಂಡ;
    factx.ಗಾತ್ರ_k_ಖಂಡ = L.ಗಾತ್ರ_k_ಖಂಡ;
    factx.ಗಾತ್ರ_v_ಖಂಡ = L.ಗಾತ್ರ_v_ಖಂಡ;
    factx.ಗಾತ್ರ_m_ಖಂಡ = L.ಗಾತ್ರ_m_ಖಂಡ;
    factx.ಗಾತ್ರ_vkq_acc = L.ಗಾತ್ರ_vkq_acc;

    uint8_t * ಸ್ಥಿರಾಂಕ ತಳ = octx->ctx->vtcm_ತಳ;

    factx.spad_q               = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_q);
    factx.spad_k               = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_k);
    factx.spad_v               = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_v);
    factx.spad_m               = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, ತಳ, L.off_m, has_ಮುಖವಾಡ);
    factx.spad_a               = VTCM_LAYOUT_PTR(uint8_t, ತಳ, L.off_a);
    factx.spad_sinks           = VTCM_LAYOUT_PTR_OPTIONAL(ದಶಮಾಂಶ, ತಳ, L.off_sinks, has_sinks);
    factx.n_ಸಕ್ರಿಯ_ಖಂಡಗಳು      = factx.n_ಖಂಡಗಳು;
    factx.shared_ಸಕ್ರಿಯ_ಖಂಡಗಳು = false;

    ಆದರೆ (has_ಮುಖವಾಡ && neq1 == 1 && neq3 == 1 && ಮುಖವಾಡ->ne[2] == 1 && factx.n_ಖಂಡಗಳು > 1 && factx.n_ಖಂಡಗಳು <= FA_MAX_KV_ಖಂಡಗಳು) {
        factx.n_ಸಕ್ರಿಯ_ಖಂಡಗಳು = fa_ಪರಿಶೀಲನೆ_ಸಕ್ರಿಯ_ಖಂಡಗಳು(
            octx->ctx->dma[0], ಮುಖವಾಡ, 0, 1, 0, 0, k->ne[1], FLASH_ATTN_ಖಂಡ_ಗಾತ್ರ,
            factx.n_ಖಂಡಗಳು, factx.spad_k, factx.ಗಾತ್ರ_k_ಖಂಡ * 2, factx.ಸಕ್ರಿಯ_ಖಂಡಗಳು);
        factx.shared_ಸಕ್ರಿಯ_ಖಂಡಗಳು = true;
    }

    ಆದರೆ (has_sinks) {
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * sinks = octx->src[4];
        dma_ಸರತಿ * dma_q = octx->ctx->dma[0];
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(factx.spad_sinks, sinks->ದತ್ತಾಂಶ), L.ಗಾತ್ರ_sinks, 0, sinks->ಗಾತ್ರ_2, 1);
        dma_ಸರತಿ_ತೆಗೆ(dma_q);
    }

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, flash_attn_ext_f16_thread, &factx, n_ಎಳೆಗಳು);

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}
