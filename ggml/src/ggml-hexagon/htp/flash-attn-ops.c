#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_compute_res.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdatomic.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hmx-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hmx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-dump.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-copy.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-reduce.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-flash-attn.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-vtcm.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "work-queue.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-quant.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "flash-attn-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-fa-kernels.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hmx-fa-kernels.h"

// Must be multiple of 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FLASH_ATTN_ਖੰਡ_ਮਾਪ (32 * 2)

ਬਣਤਰ htp_fa_context {
    ਅਚਲ ਬਣਤਰ htp_ops_context * octx;

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src0_div21;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src0_div1;

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ ਪ੍ਰਸਾਰਣ_rk2;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ ਪ੍ਰਸਾਰਣ_rk3;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ ਪ੍ਰਸਾਰਣ_rv2;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ ਪ੍ਰਸਾਰਣ_rv3;

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src3_div2;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src3_div3;

    ਦਸ਼ਮਲਵ scale;
    ਦਸ਼ਮਲਵ max_bias;
    bool  has_softcap;
    __fp16 logit_softcap;

    uint32_t n_ਸਿਰ_log2;
    ਦਸ਼ਮਲਵ m0;
    ਦਸ਼ਮਲਵ m1;
    __fp16 slopes[512];

    uint32_t n_ਖੰਡਾਂ;
    uint32_t n_ਸਰਗਰਮ_ਖੰਡਾਂ;
    bool     shared_ਸਰਗਰਮ_ਖੰਡਾਂ;

    size_t ਮਾਪ_q_ਪੰਕਤੀ_padded;
    size_t ਮਾਪ_k_ਪੰਕਤੀ_padded;
    size_t ਮਾਪ_v_ਪੰਕਤੀ_padded;

    size_t ਮਾਪ_k_ਖੰਡ;
    size_t ਮਾਪ_v_ਖੰਡ;
    size_t ਮਾਪ_m_ਖੰਡ;

    uint32_t qrows;
    uint32_t qrows_ਪ੍ਰਤੀ_thread;
    uint32_t qrow_ਸ਼ੁਰੂ;

    bool is_q_fp32;

    size_t ਮਾਪ_q_ਖੰਡ;
    size_t ਮਾਪ_vkq_acc;

    uint8_t *  spad_q;
    uint8_t *  spad_k;
    uint8_t *  spad_v;
    uint8_t *  spad_m;
    uint8_t *  spad_a;
    ਦਸ਼ਮਲਵ *    spad_sinks;
    uint16_t   ਸਰਗਰਮ_ਖੰਡਾਂ[FA_MAX_KV_ਖੰਡਾਂ];

    ਅਚਲ ਬਣਤਰ htp_tensor * k;
    ਅਚਲ ਬਣਤਰ htp_tensor * v;
};

ਬਣਤਰ hmx_fa_context {
    ਅਚਲ ਬਣਤਰ htp_ops_context * octx;
    ਅਚਲ ਬਣਤਰ htp_tensor *      sinks;  // attention sinks (src[4]), NULL if absent
    ਅਚਲ ਬਣਤਰ htp_tensor *      k;
    ਅਚਲ ਬਣਤਰ htp_tensor *      v;
    bool         pipeline;  // true when n_kv_blocks >= FA_MIN_KV_BLOCKS && n_threads >= 2
    uint32_t     n_ਧਾਗੇ;

    // Op parameters
    __fp16       scale;
    ਦਸ਼ਮਲਵ        max_bias;
    bool         has_softcap;
    __fp16       logit_softcap;
    uint32_t     n_ਸਿਰ_log2;
    ਦਸ਼ਮਲਵ        m0, m1;

    // Dimensions
    uint32_t     DK, DV;
    uint32_t     DK_pad, DV_pad;  // head_dim rounded up to 64 for HMX tiling
    uint32_t     n_kv;        // kv_len
    uint32_t     n_kv_heads;  // number of KV heads
    uint32_t     n_heads;     // number of Q heads
    uint32_t     G;           // GQA factor = n_heads / n_kv_heads
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_G;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src3_div2;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ src3_div3;
    uint32_t     n_kv_ਖੰਡਾਂ;
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
    HVX_ਸਦਿਸ਼ * vtcm_m_vec;           // Row max [g_br]
    HVX_ਸਦਿਸ਼ * vtcm_l_vec;           // Row sum [g_br]
    HVX_ਸਦਿਸ਼ * vtcm_s_rowmax;        // Softmax intermediate [g_br]
    HVX_ਸਦਿਸ਼ * vtcm_p_rowsum;        // Softmax intermediate [g_br]
    HVX_ਸਦਿਸ਼ * vtcm_ਪੰਕਤੀ_bufs;        // Per-thread softmax row scratch [n_threads][2][Bc/64]
    uint8_t *    vtcm_hmx_scales_id;   // HMX output scales (identity)
    uint8_t *    vtcm_hmx_scales_qk;   // HMX output scales (qk_scale)
    __fp16 *     vtcm_ਨਕਾਬ_buf;        // VTCM mask buffer [Br * m_line], DMA'd per KV block
    __fp16 *     vtcm_slopes;          // ALiBi slopes [g_br]
    ਦਸ਼ਮਲਵ *      vtcm_sinks;           // Attention sinks
    size_t       ਪੰਕਤੀ_buf_stride;       // HVX vectors per row buffer (Bc/64)
    size_t       ਨਕਾਬ_buf_ਪੰਕਤੀ_stride;  // elements (__fp16) per row in mask buffer
    size_t       q_tile_ਬਾਈਟਾਂ;
    size_t       k_tile_ਬਾਈਟਾਂ;
    size_t       o_tile_ਬਾਈਟਾਂ;
    size_t       ਖਾਨਾ_vec_ਬਾਈਟਾਂ;
    size_t       d_tile_ਬਾਈਟਾਂ;
    bool         ਨਕਾਬ_ਪ੍ਰਸਾਰਣ;       // true when mask->ne[2] == 1 (head-independent, single 2D DMA)
    dma_cache_fa m_cache;
};

// Scan mask rows [q_start, q_start + n_rows_q) and build compact list of KV blocks with any entry != -inf.
ਸਥਿਰ uint32_t fa_ਜਾਂਚ_ਸਰਗਰਮ_ਖੰਡਾਂ(
        dma_ਕਤਾਰ *               dma_q,
        ਅਚਲ ਬਣਤਰ htp_tensor * ਨਕਾਬ,
        uint32_t                  q_ਸ਼ੁਰੂ,
        uint32_t                  n_ਪੰਕਤੀਆਂ_q,
        uint32_t                  im2,
        uint32_t                  im3,
        uint32_t                  nek1,
        uint32_t                  Bc,
        uint32_t                  n_kv_ਖੰਡਾਂ,
        ਖਾਲੀ *                    vtcm_scratch,
        size_t                    scratch_ਬਾਈਟਾਂ,
        uint16_t *                ਸਰਗਰਮ_ਖੰਡਾਂ) {
    ਅਚਲ size_t ਪੰਕਤੀ_ਬਾਈਟਾਂ  = (size_t) nek1 * ਆਕਾਰ(__fp16);
    ਅਚਲ size_t ਪੰਕਤੀ_stride = hex_ਗੇੜਾ_ਉੱਪਰ(ਪੰਕਤੀ_ਬਾਈਟਾਂ, 128);

    ਜੇ (!ਨਕਾਬ || n_kv_ਖੰਡਾਂ <= 1 || n_kv_ਖੰਡਾਂ > FA_MAX_KV_ਖੰਡਾਂ ||
        (size_t) n_ਪੰਕਤੀਆਂ_q * ਪੰਕਤੀ_stride > scratch_ਬਾਈਟਾਂ) {
        #ਹਦਾਇਤ clang loop vectorize(disable)
        ਲਈ (uint32_t b = 0; b < n_kv_ਖੰਡਾਂ; ++b) {
            ਸਰਗਰਮ_ਖੰਡਾਂ[b] = (uint16_t) b;
        }
        ਵਾਪਸ n_kv_ਖੰਡਾਂ;
    }

    ਅਚਲ dma_addr_t m_src = ਨਕਾਬ->ਅੰਕੜੇ + q_ਸ਼ੁਰੂ * ਨਕਾਬ->nb[1] + im2 * ਨਕਾਬ->nb[2] + im3 * ਨਕਾਬ->nb[3];
    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_scratch, m_src), ਪੰਕਤੀ_stride, ਨਕਾਬ->nb[1], ਪੰਕਤੀ_ਬਾਈਟਾਂ, n_ਪੰਕਤੀਆਂ_q);
    ਅਚਲ uint8_t * m_buf = (ਅਚਲ uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

    ਅਚਲ HVX_ਸਦਿਸ਼ v_ninf   = Q6_Vh_vsplat_R(0xFC00);
    uint32_t         n_ਸਰਗਰਮ = 0;

    ਲਈ (uint32_t b = 0; b < n_kv_ਖੰਡਾਂ; ++b) {
        ਅਚਲ uint32_t col0 = b * Bc;
        ਅਚਲ uint32_t ਖਾਨੇ_ਗਿਣਤੀ_2 = MIN(Bc, nek1 - col0);

        ਲਈ (uint32_t r = 0; r < n_ਪੰਕਤੀਆਂ_q; ++r) {
            ਅਚਲ __fp16 * ਪੰਕਤੀ_ptr = (ਅਚਲ __fp16 *) (m_buf + r * ਪੰਕਤੀ_stride) + col0;
            HVX_ਸਦਿਸ਼ v_any = Q6_V_vzero();
            uint32_t c = 0;
            ਲਈ (; c + 64 <= ਖਾਨੇ_ਗਿਣਤੀ_2; c += 64) {
                ਅਚਲ HVX_ਸਦਿਸ਼ v_m = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਪੰਕਤੀ_ptr + c);
                v_any = Q6_V_vor_VV(v_any, Q6_V_vxor_VV(v_m, v_ninf));
            }
            ਜੇ (c < ਖਾਨੇ_ਗਿਣਤੀ_2) {
                ਅਚਲ uint32_t rem = ਖਾਨੇ_ਗਿਣਤੀ_2 - c;
                ਅਚਲ HVX_ਸਦਿਸ਼_Pred q_rem = Q6_Q_vsetq2_R(rem * ਆਕਾਰ(__fp16));
                ਅਚਲ HVX_ਸਦਿਸ਼ v_m = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਪੰਕਤੀ_ptr + c);
                ਅਚਲ HVX_ਸਦਿਸ਼ v_diff = Q6_V_vand_QV(q_rem, Q6_V_vxor_VV(v_m, v_ninf));
                v_any = Q6_V_vor_VV(v_any, v_diff);
            }
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 64));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 32));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 16));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 8));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 4));
            ਜੇ (Q6_R_vextract_VR(v_any, 0) != 0) {
                ਸਰਗਰਮ_ਖੰਡਾਂ[n_ਸਰਗਰਮ++] = (uint16_t) b;
                ਰੋਕੋ;
            }
        }
    }

    ਜੇ (n_ਸਰਗਰਮ == 0 && n_kv_ਖੰਡਾਂ > 0) {
        ਸਰਗਰਮ_ਖੰਡਾਂ[0] = 0;
        n_ਸਰਗਰਮ = 1;
    }
    ਵਾਪਸ n_ਸਰਗਰਮ;
}

ਸਥਿਰ ਖਾਲੀ flash_attn_ext_f16_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_fa_context * factx = (ਬਣਤਰ htp_fa_context *) ਅੰਕੜੇ;
    ਅਚਲ ਬਣਤਰ htp_ops_context * octx = factx->octx;
    ਅਚਲ ਬਣਤਰ htp_tensor * q     = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * k     = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * v     = octx->src[2];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਨਕਾਬ  = octx->src[3];
    ਅਚਲ ਬਣਤਰ htp_tensor * sinks = octx->src[4];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst   = octx->dst;

    ਅਚਲ uint32_t neq0 = q->ne[0];
    ਅਚਲ uint32_t neq1 = q->ne[1];
    ਅਚਲ uint32_t neq2 = q->ne[2];
    ਅਚਲ uint32_t neq3 = q->ne[3];

    ਅਚਲ uint32_t nek0 = k->ne[0];
    ਅਚਲ uint32_t nek1 = k->ne[1];
    ਅਚਲ uint32_t nek2 = k->ne[2];
    ਅਚਲ uint32_t nek3 = k->ne[3];

    ਅਚਲ uint32_t nev0 = v->ne[0];
    ਅਚਲ uint32_t nev1 = v->ne[1];
    ਅਚਲ uint32_t nev2 = v->ne[2];
    ਅਚਲ uint32_t nev3 = v->ne[3];

    ਅਚਲ uint32_t nbq1 = q->nb[1];
    ਅਚਲ uint32_t nbq2 = q->nb[2];
    ਅਚਲ uint32_t nbq3 = q->nb[3];

    ਅਚਲ uint32_t nbk1 = k->nb[1];
    ਅਚਲ uint32_t nbk2 = k->nb[2];
    ਅਚਲ uint32_t nbk3 = k->nb[3];

    ਅਚਲ uint32_t nbv1 = v->nb[1];
    ਅਚਲ uint32_t nbv2 = v->nb[2];
    ਅਚਲ uint32_t nbv3 = v->nb[3];

    ਅਚਲ uint32_t ne1 = dst->ne[1];
    ਅਚਲ uint32_t ne2 = dst->ne[2];
    ਅਚਲ uint32_t ne3 = dst->ne[3];

    ਅਚਲ uint32_t nb1 = dst->nb[1];
    ਅਚਲ uint32_t nb2 = dst->nb[2];
    ਅਚਲ uint32_t nb3 = dst->nb[3];

    // total rows in q
    ਅਚਲ uint32_t dr  = factx->qrows_ਪ੍ਰਤੀ_thread;
    ਅਚਲ uint32_t ir0 = factx->qrow_ਸ਼ੁਰੂ + dr * ith;
    ਅਚਲ uint32_t ir1 = MIN(ir0 + dr, factx->qrow_ਸ਼ੁਰੂ + factx->qrows);

    ਜੇ (ir0 >= ir1) ਵਾਪਸ;

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];

    ਅਚਲ uint32_t DK = nek0;
    ਅਚਲ uint32_t DV = nev0;

    ਅਚਲ size_t ਮਾਪ_q_ਪੰਕਤੀ = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    ਅਚਲ size_t ਮਾਪ_k_ਪੰਕਤੀ = htp_tensor_get_ਪੰਕਤੀ_ਮਾਪ(k->type, DK);
    ਅਚਲ size_t ਮਾਪ_v_ਪੰਕਤੀ = htp_tensor_get_ਪੰਕਤੀ_ਮਾਪ(v->type, DV);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = factx->spad_q + factx->ਮਾਪ_q_ਖੰਡ * ith;
    uint8_t * spad_k = factx->spad_k + factx->ਮਾਪ_k_ਖੰਡ * 2 * ith;
    uint8_t * spad_v = factx->spad_v + factx->ਮਾਪ_v_ਖੰਡ * 2 * ith;
    uint8_t * spad_m = factx->spad_m + (ਨਕਾਬ ? factx->ਮਾਪ_m_ਖੰਡ * HVX_FA_DMA_CACHE_ਮਾਪ : 0) * ith;
    uint8_t * spad_a = factx->spad_a + factx->ਮਾਪ_vkq_acc * ith;

    dma_cache_dm m_cache;
    dma_cache_dm_init(&m_cache, spad_m, factx->ਮਾਪ_m_ਖੰਡ, HVX_FA_DMA_CACHE_ਮਾਪ);

    uint16_t thread_ਸਰਗਰਮ_ਖੰਡਾਂ[FA_MAX_KV_ਖੰਡਾਂ];
    ਅਚਲ uint16_t * ਸਰਗਰਮ_ਖੰਡਾਂ = factx->shared_ਸਰਗਰਮ_ਖੰਡਾਂ ? factx->ਸਰਗਰਮ_ਖੰਡਾਂ : thread_ਸਰਗਰਮ_ਖੰਡਾਂ;
    uint32_t n_ਸਰਗਰਮ_ਖੰਡਾਂ       = factx->n_ਸਰਗਰਮ_ਖੰਡਾਂ;
    dma_addr_t prev_mp_ਅਧਾਰ        = 0;

    ਜੇ (!factx->shared_ਸਰਗਰਮ_ਖੰਡਾਂ && !ਨਕਾਬ) {
        #ਹਦਾਇਤ clang loop vectorize(disable)
        ਲਈ (uint32_t b = 0; b < factx->n_ਖੰਡਾਂ; ++b) {
            thread_ਸਰਗਰਮ_ਖੰਡਾਂ[b] = (uint16_t) b;
        }
    }

    ਅਚਲ size_t ਮਾਪ_vkq_acc_single = hex_ਗੇੜਾ_ਉੱਪਰ(DV * ਆਕਾਰ(ਦਸ਼ਮਲਵ), 128);

    uint32_t ir = ir0;
    ਜਦੋਂ (ir < ir1) {
        ਅਚਲ uint32_t iq3 = fastdiv(ir, &factx->src0_div21);
        ਅਚਲ uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &factx->src0_div1);
        ਅਚਲ uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        ਅਚਲ uint32_t ik3 = fastdiv(iq3, &factx->ਪ੍ਰਸਾਰਣ_rk3);
        ਅਚਲ uint32_t ik2 = fastdiv(iq2, &factx->ਪ੍ਰਸਾਰਣ_rk2);

        ਅਚਲ uint32_t iv3 = fastdiv(iq3, &factx->ਪ੍ਰਸਾਰਣ_rv3);
        ਅਚਲ uint32_t iv2 = fastdiv(iq2, &factx->ਪ੍ਰਸਾਰਣ_rv2);

        uint32_t G_local = 1;
        ਜੇ (neq1 == 1 && (ਨਕਾਬ == ਨਿਰਾ || ਨਕਾਬ->ne[2] == 1)) {
            ਜਦੋਂ (ir + G_local < ir1 && G_local < FA_HVX_G_MAX) {
                ਅਚਲ uint32_t ਅਗਲਾ_ir = ir + G_local;
                ਅਚਲ uint32_t ਅਗਲਾ_iq3 = fastdiv(ਅਗਲਾ_ir, &factx->src0_div21);
                ਅਚਲ uint32_t ਅਗਲਾ_iq2 = fastdiv(ਅਗਲਾ_ir - ਅਗਲਾ_iq3*neq2*neq1, &factx->src0_div1);
                ਅਚਲ uint32_t ਅਗਲਾ_iq1 = (ਅਗਲਾ_ir - ਅਗਲਾ_iq3*neq2*neq1 - ਅਗਲਾ_iq2 * neq1);

                ਅਚਲ uint32_t ਅਗਲਾ_ik3 = fastdiv(ਅਗਲਾ_iq3, &factx->ਪ੍ਰਸਾਰਣ_rk3);
                ਅਚਲ uint32_t ਅਗਲਾ_ik2 = fastdiv(ਅਗਲਾ_iq2, &factx->ਪ੍ਰਸਾਰਣ_rk2);

                ਅਚਲ uint32_t ਅਗਲਾ_iv3 = fastdiv(ਅਗਲਾ_iq3, &factx->ਪ੍ਰਸਾਰਣ_rv3);
                ਅਚਲ uint32_t ਅਗਲਾ_iv2 = fastdiv(ਅਗਲਾ_iq2, &factx->ਪ੍ਰਸਾਰਣ_rv2);

                ਜੇ (ਅਗਲਾ_ik2 != ik2 || ਅਗਲਾ_ik3 != ik3 || ਅਗਲਾ_iv2 != iv2 || ਅਗਲਾ_iv3 != iv3 || ਅਗਲਾ_iq1 != iq1 || ਅਗਲਾ_iq3 != iq3) {
                    ਰੋਕੋ;
                }
                G_local++;
            }
        }

        dma_addr_t mp_ਅਧਾਰ = 0;
        ਜੇ (ਨਕਾਬ) {
            ਅਚਲ uint32_t im2 = fastmodulo(iq2, ਨਕਾਬ->ne[2], &factx->src3_div2);
            ਅਚਲ uint32_t im3 = fastmodulo(iq3, ਨਕਾਬ->ne[3], &factx->src3_div3);
            mp_ਅਧਾਰ = ਨਕਾਬ->ਅੰਕੜੇ + iq1*ਨਕਾਬ->nb[1] + im2*ਨਕਾਬ->nb[2] + im3*ਨਕਾਬ->nb[3];
            ਜੇ (!factx->shared_ਸਰਗਰਮ_ਖੰਡਾਂ && mp_ਅਧਾਰ != prev_mp_ਅਧਾਰ) {
                n_ਸਰਗਰਮ_ਖੰਡਾਂ = fa_ਜਾਂਚ_ਸਰਗਰਮ_ਖੰਡਾਂ(
                    dma_q, ਨਕਾਬ, iq1, 1, im2, im3, nek1, FLASH_ATTN_ਖੰਡ_ਮਾਪ,
                    factx->n_ਖੰਡਾਂ, spad_k, factx->ਮਾਪ_k_ਖੰਡ * 2, thread_ਸਰਗਰਮ_ਖੰਡਾਂ);
                prev_mp_ਅਧਾਰ = mp_ਅਧਾਰ;
            }
        }

        uint32_t heads[FA_HVX_G_MAX];
        HVX_ਸਦਿਸ਼ ਢਲਾਨ_vecs[FA_HVX_G_MAX] __attribute__((aligned(128)));
        HVX_ਸਦਿਸ਼ S_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        HVX_ਸਦਿਸ਼ M_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        uint8_t * q_ptrs[FA_HVX_G_MAX];
        ਦਸ਼ਮਲਵ * vkq_ptrs[FA_HVX_G_MAX];

        ਲਈ (uint32_t g = 0; g < G_local; ++g) {
            ਅਚਲ uint32_t r = ir + g;
            ਅਚਲ uint32_t r_iq3 = fastdiv(r, &factx->src0_div21);
            ਅਚਲ uint32_t r_iq2 = fastdiv(r - r_iq3*neq2*neq1, &factx->src0_div1);
            ਅਚਲ uint32_t r_iq1 = (r - r_iq3*neq2*neq1 - r_iq2 * neq1);

            heads[g] = r_iq2;
            ਅਚਲ __fp16 ਢਲਾਨ = factx->slopes[r_iq2];
            ਢਲਾਨ_vecs[g] = hvx_vec_splat_f16(ਢਲਾਨ);

            S_vec[g] = hvx_vec_splat_f32(0.0f);
            M_vec[g] = hvx_vec_splat_f32(HTP_FA_M_INITIAL_VAL);

            uint8_t * q_dst = spad_q + g * factx->ਮਾਪ_q_ਪੰਕਤੀ_padded;
            q_ptrs[g] = q_dst;

            ਦਸ਼ਮਲਵ * vkq_dst = (ਦਸ਼ਮਲਵ *)(spad_a + g * ਮਾਪ_vkq_acc_single);
            vkq_ptrs[g] = vkq_dst;
            hvx_splat_f32_a((uint8_t *) vkq_dst, 0, DV);

            // Fetch Q row g
            ਅਚਲ dma_addr_t q_ਪੰਕਤੀ_ptr = q->ਅੰਕੜੇ + r_iq1*nbq1 + r_iq2*nbq2 + r_iq3*nbq3;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(q_dst, q_ਪੰਕਤੀ_ptr), factx->ਮਾਪ_q_ਪੰਕਤੀ_padded, nbq1, ਮਾਪ_q_ਪੰਕਤੀ, 1);
        }

        // Prefetch first two active blocks
        ਲਈ (uint32_t ਕਦਮ = 0; ਕਦਮ < MIN(n_ਸਰਗਰਮ_ਖੰਡਾਂ, 2); ++ਕਦਮ) {
            ਅਚਲ uint32_t ib = ਸਰਗਰਮ_ਖੰਡਾਂ[ਕਦਮ];
            ਅਚਲ uint32_t ic_ਸ਼ੁਰੂ = ib * FLASH_ATTN_ਖੰਡ_ਮਾਪ;
            ਅਚਲ uint32_t current_ਖੰਡ_ਮਾਪ = MIN(FLASH_ATTN_ਖੰਡ_ਮਾਪ, nek1 - ic_ਸ਼ੁਰੂ);

            // K
            ਅਚਲ dma_addr_t k_src = k->ਅੰਕੜੇ + ic_ਸ਼ੁਰੂ*nbk1 + ik2*nbk2 + ik3*nbk3;
            uint8_t * k_dst = spad_k + (ਕਦਮ % 2) * factx->ਮਾਪ_k_ਖੰਡ;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(k_dst, k_src), factx->ਮਾਪ_k_ਪੰਕਤੀ_padded, nbk1, ਮਾਪ_k_ਪੰਕਤੀ, current_ਖੰਡ_ਮਾਪ);

            // V
            ਅਚਲ dma_addr_t v_src = v->ਅੰਕੜੇ + ic_ਸ਼ੁਰੂ*nbv1 + iv2*nbv2 + iv3*nbv3;
            uint8_t * v_dst = spad_v + (ਕਦਮ % 2) * factx->ਮਾਪ_v_ਖੰਡ;
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(v_dst, v_src), factx->ਮਾਪ_v_ਪੰਕਤੀ_padded, nbv1, ਮਾਪ_v_ਪੰਕਤੀ, current_ਖੰਡ_ਮਾਪ);

            // Mask
            ਜੇ (ਨਕਾਬ) {
                ਅਚਲ dma_addr_t m_src = mp_ਅਧਾਰ + ic_ਸ਼ੁਰੂ * ਆਕਾਰ(__fp16);
                dma_cache_dm_ਧੱਕੋ(dma_q, &m_cache, ib, m_src, current_ਖੰਡ_ਮਾਪ * 2, current_ਖੰਡ_ਮਾਪ * 2, current_ਖੰਡ_ਮਾਪ * 2, 1);
            }
        }

        // Pop all Q rows
        ਲਈ (uint32_t g = 0; g < G_local; ++g) {
            uint8_t * q_ptr_vtcm = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
            ਜੇ (factx->is_q_fp32) {
                hvx_copy_f16_f32_aa(q_ptr_vtcm, q_ptr_vtcm, DK);
            }
        }

        ਅਚਲ HVX_ਸਦਿਸ਼ v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
        ਅਚਲ bool has_softcap     = factx->has_softcap;
        ਅਚਲ HVX_ਸਦਿਸ਼ v_cap     = has_softcap ? hvx_vec_splat_f16(factx->logit_softcap) : Q6_V_vzero();
        ਅਚਲ HVX_ਸਦਿਸ਼ vinf      = Q6_Vh_vsplat_R(0xFC00);
        ਅਚਲ HVX_ਸਦਿਸ਼ vmin      = Q6_Vh_vsplat_R(0xFBFF);
        ਅਚਲ HVX_ਸਦਿਸ਼ v_log2e   = hvx_vec_splat_f16(EXP_LOG2E_F);
        ਅਚਲ uint32_t stride_v2   = factx->ਮਾਪ_v_ਪੰਕਤੀ_padded * 2;

        ਲਈ (uint32_t ਕਦਮ = 0; ਕਦਮ < n_ਸਰਗਰਮ_ਖੰਡਾਂ; ++ਕਦਮ) {
            ਅਚਲ uint32_t ib = ਸਰਗਰਮ_ਖੰਡਾਂ[ਕਦਮ];
            ਅਚਲ uint32_t ic_ਸ਼ੁਰੂ = ib * FLASH_ATTN_ਖੰਡ_ਮਾਪ;
            ਅਚਲ uint32_t current_ਖੰਡ_ਮਾਪ = MIN(FLASH_ATTN_ਖੰਡ_ਮਾਪ, nek1 - ic_ਸ਼ੁਰੂ);

            // Wait for DMA
            uint8_t * k_ਅਧਾਰ = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst; // K
            uint8_t * v_ਅਧਾਰ = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst; // V
            __fp16  * m_ਅਧਾਰ = ਨਕਾਬ ? (__fp16 *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst : ਨਿਰਾ; // M

            ਜੇ (factx->k->type == HTP_TYPE_Q8_0) {
                htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
                ਲਈ (uint32_t r = 0; r < current_ਖੰਡ_ਮਾਪ; ++r) {
                    __fp16 * ਪੰਕਤੀ_k = (__fp16 *)(k_ਅਧਾਰ + r * factx->ਮਾਪ_k_ਪੰਕਤੀ_padded);
                    hvx_dequantize_ਪੰਕਤੀ_flat_q8_0_f16(ਪੰਕਤੀ_k, ਪੰਕਤੀ_k, DK);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
            }
            ਜੇ (factx->v->type == HTP_TYPE_Q8_0) {
                htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
                ਲਈ (uint32_t r = 0; r < current_ਖੰਡ_ਮਾਪ; ++r) {
                    __fp16 * ਪੰਕਤੀ_v = (__fp16 *)(v_ਅਧਾਰ + r * factx->ਮਾਪ_v_ਪੰਕਤੀ_padded);
                    hvx_dequantize_ਪੰਕਤੀ_flat_q8_0_f16(ਪੰਕਤੀ_v, ਪੰਕਤੀ_v, DV);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
            }

            ਲਈ (uint32_t g = 0; g < G_local; ++g) {
                ਅਚਲ uint32_t ਸਿਰ_ir = ir + g;
                uint8_t * q_ptr_vtcm = q_ptrs[g];
                ਦਸ਼ਮਲਵ * VKQ32 = vkq_ptrs[g];
                ਅਚਲ HVX_ਸਦਿਸ਼ ਢਲਾਨ_vec = ਢਲਾਨ_vecs[g];

                htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_FA_QK, ਸਿਰ_ir);

                HVX_ਸਦਿਸ਼ scores_f16 = Q6_V_vzero();
                ਜੇ (current_ਖੰਡ_ਮਾਪ > 0) {
                    HVX_ਸਦਿਸ਼ scores0 = hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_ਅਧਾਰ, factx->ਮਾਪ_k_ਪੰਕਤੀ_padded, DK, factx->scale);
                    HVX_ਸਦਿਸ਼ scores1 = (current_ਖੰਡ_ਮਾਪ > 32) ? hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_ਅਧਾਰ + 32 * factx->ਮਾਪ_k_ਪੰਕਤੀ_padded, factx->ਮਾਪ_k_ਪੰਕਤੀ_padded, DK, factx->scale) : Q6_V_vzero();
                    scores_f16 = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(scores0, scores1);
                }

                ਜੇ (has_softcap) {
                    scores_f16 = hvx_vec_tanh_f16(scores_f16);
                    scores_f16 = hvx_vec_mul_f16_f16(scores_f16, v_cap);
                }

                HVX_ਸਦਿਸ਼_Pred q_ਪੂਛ_keep = Q6_Q_vsetq2_R(current_ਖੰਡ_ਮਾਪ * ਆਕਾਰ(__fp16));

                ਜੇ (ਨਕਾਬ) {
                    HVX_ਸਦਿਸ਼ m_vals_f16 = *(ਅਚਲ HVX_UVector *) m_ਅਧਾਰ;
                    HVX_ਸਦਿਸ਼_Pred is_inf = Q6_Q_vcmp_eq_VhVh(m_vals_f16, vinf);
                    m_vals_f16 = Q6_V_vmux_QVV(is_inf, vmin, m_vals_f16);

                    HVX_ਸਦਿਸ਼ m_scaled = hvx_vec_mul_f16_f16(m_vals_f16, ਢਲਾਨ_vec);
                    scores_f16 = Q6_V_vmux_QVV(q_ਪੂਛ_keep, hvx_vec_ਜੋੜੋ_f16_f16(scores_f16, m_scaled), v_neg_inf);
                } ਨਹੀਂਤਾਂ {
                    scores_f16 = Q6_V_vmux_QVV(q_ਪੂਛ_keep, scores_f16, v_neg_inf);
                }

                HVX_ਸਦਿਸ਼ v_max_f16 = hvx_vec_reduce_max_f16(scores_f16);
                HVX_ਸਦਿਸ਼ v_max     = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(v_max_f16));
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_QK, ਸਿਰ_ir);

                // prefetch K for step + 2 after last head finished QK
                ਜੇ (g + 1 == G_local && ਕਦਮ + 2 < n_ਸਰਗਰਮ_ਖੰਡਾਂ) {
                    ਅਚਲ uint32_t ਅਗਲਾ_ib = ਸਰਗਰਮ_ਖੰਡਾਂ[ਕਦਮ + 2];
                    ਅਚਲ uint32_t ਅਗਲਾ_ic_ਸ਼ੁਰੂ = ਅਗਲਾ_ib * FLASH_ATTN_ਖੰਡ_ਮਾਪ;
                    ਅਚਲ uint32_t ਅਗਲਾ_ਖੰਡ_ਮਾਪ = MIN(FLASH_ATTN_ਖੰਡ_ਮਾਪ, nek1 - ਅਗਲਾ_ic_ਸ਼ੁਰੂ);

                    ਅਚਲ dma_addr_t k_src = k->ਅੰਕੜੇ + ਅਗਲਾ_ic_ਸ਼ੁਰੂ*nbk1 + ik2*nbk2 + ik3*nbk3;
                    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(k_ਅਧਾਰ, k_src), factx->ਮਾਪ_k_ਪੰਕਤੀ_padded, nbk1, ਮਾਪ_k_ਪੰਕਤੀ, ਅਗਲਾ_ਖੰਡ_ਮਾਪ);
                }

                htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_FA_SFM, ਸਿਰ_ir);
                {
                    HVX_ਸਦਿਸ਼ M_new_vec = Q6_Vsf_vmax_VsfVsf(v_max, M_vec[g]);
                    HVX_ਸਦਿਸ਼ diff_vec  = HVX_OP_SUB_F32(M_vec[g], M_new_vec);

                    HVX_ਸਦਿਸ਼ diff_f16   = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(diff_vec, diff_vec);
                    HVX_ਸਦਿਸ਼ diff_base2 = hvx_vec_mul_f16_f16(diff_f16, v_log2e);
                    HVX_ਸਦਿਸ਼ ms_f16     = hvx_vec_exp2_f16(diff_base2);
                    HVX_ਸਦਿਸ਼ ms_vec     = Q6_V_lo_W(hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(ms_f16));

                    M_vec[g] = M_new_vec;

                    HVX_ਸਦਿਸ਼ v_m_vec_f16 = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(M_vec[g], M_vec[g]);
                    HVX_ਸਦਿਸ਼ v_s_minus_m = Q6_Vqf16_vsub_VhfVhf(scores_f16, v_m_vec_f16);
                    HVX_ਸਦਿਸ਼ v_s_minus_m_base2 = hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m), v_log2e);

                    HVX_ਸਦਿਸ਼ P = hvx_vec_exp2_f16(v_s_minus_m_base2);
                    P = Q6_V_vmux_QVV(q_ਪੂਛ_keep, P, Q6_V_vzero());

                    HVX_ਸਦਿਸ਼_Pair P_pair = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(P);
                    HVX_ਸਦਿਸ਼ P0 = Q6_V_lo_W(P_pair);
                    HVX_ਸਦਿਸ਼ P1 = Q6_V_hi_W(P_pair);
                    HVX_ਸਦਿਸ਼ p_ਜੋੜ_vec = hvx_vec_reduce_ਜੋੜ_f32(HVX_OP_ਜੋੜੋ_F32(P0, P1));

                    S_vec[g] = HVX_OP_ਜੋੜੋ_F32(HVX_OP_MUL_F32(S_vec[g], ms_vec), p_ਜੋੜ_vec);

                    ਅਚਲ uint8_t * v_ptr = v_ਅਧਾਰ;

                    ਜੇ (DV == 64) {
                        HVX_ਸਦਿਸ਼_Pair vkq0 = *((ਅਚਲ HVX_ਸਦਿਸ਼_Pair *) VKQ32);
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );

                        ਲਈ (uint32_t j = 0; j < current_ਖੰਡ_ਮਾਪ; j += 2) {
                            HVX_ਸਦਿਸ਼ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ਅਚਲ HVX_ਸਦਿਸ਼ * vx0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) v_ptr;
                            ਜੇ (j + 1 == current_ਖੰਡ_ਮਾਪ) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                ਰੋਕੋ;
                            }

                            HVX_ਸਦਿਸ਼ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ਅਚਲ HVX_ਸਦਿਸ਼ * vx1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (v_ptr + factx->ਮਾਪ_v_ਪੰਕਤੀ_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            v_ptr += stride_v2;
                        }

                        *((HVX_ਸਦਿਸ਼_Pair *) VKQ32) = vkq0;
                    } ਨਹੀਂਤਾਂ ਜੇ (DV == 128) {
                        HVX_ਸਦਿਸ਼_Pair vkq0 = ((ਅਚਲ HVX_ਸਦਿਸ਼_Pair *) VKQ32)[0];
                        HVX_ਸਦਿਸ਼_Pair vkq1 = ((ਅਚਲ HVX_ਸਦਿਸ਼_Pair *) VKQ32)[1];
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );
                        vkq1 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq1), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq1), ms_vec)
                        );

                        ਲਈ (uint32_t j = 0; j < current_ਖੰਡ_ਮਾਪ; j += 2) {
                            HVX_ਸਦਿਸ਼ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ਅਚਲ HVX_ਸਦਿਸ਼ * vx0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) v_ptr;
                            ਜੇ (j + 1 == current_ਖੰਡ_ਮਾਪ) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                ਰੋਕੋ;
                            }

                            HVX_ਸਦਿਸ਼ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ਅਚਲ HVX_ਸਦਿਸ਼ * vx1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (v_ptr + factx->ਮਾਪ_v_ਪੰਕਤੀ_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx1[1]), S1);
                            v_ptr += stride_v2;
                        }

                        ((HVX_ਸਦਿਸ਼_Pair *) VKQ32)[0] = vkq0;
                        ((HVX_ਸਦਿਸ਼_Pair *) VKQ32)[1] = vkq1;
                    } ਨਹੀਂਤਾਂ ਜੇ (DV == 256) {
                        HVX_ਸਦਿਸ਼_Pair vkq0 = ((ਅਚਲ HVX_ਸਦਿਸ਼_Pair *) VKQ32)[0];
                        HVX_ਸਦਿਸ਼_Pair vkq1 = ((ਅਚਲ HVX_ਸਦਿਸ਼_Pair *) VKQ32)[1];
                        HVX_ਸਦਿਸ਼_Pair vkq2 = ((ਅਚਲ HVX_ਸਦਿਸ਼_Pair *) VKQ32)[2];
                        HVX_ਸਦਿਸ਼_Pair vkq3 = ((ਅਚਲ HVX_ਸਦਿਸ਼_Pair *) VKQ32)[3];
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

                        ਲਈ (uint32_t j = 0; j < current_ਖੰਡ_ਮਾਪ; j += 2) {
                            HVX_ਸਦਿਸ਼ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ਅਚਲ HVX_ਸਦਿਸ਼ * vx0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) v_ptr;
                            ਜੇ (j + 1 == current_ਖੰਡ_ਮਾਪ) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                vkq2 = hvx_vec_mpyacc_f32_f16(vkq2, Q6_Vh_vshuff_Vh(vx0[2]), S0);
                                vkq3 = hvx_vec_mpyacc_f32_f16(vkq3, Q6_Vh_vshuff_Vh(vx0[3]), S0);
                                ਰੋਕੋ;
                            }

                            HVX_ਸਦਿਸ਼ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ਅਚਲ HVX_ਸਦਿਸ਼ * vx1 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (v_ptr + factx->ਮਾਪ_v_ਪੰਕਤੀ_padded);
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

                        ((HVX_ਸਦਿਸ਼_Pair *) VKQ32)[0] = vkq0;
                        ((HVX_ਸਦਿਸ਼_Pair *) VKQ32)[1] = vkq1;
                        ((HVX_ਸਦਿਸ਼_Pair *) VKQ32)[2] = vkq2;
                        ((HVX_ਸਦਿਸ਼_Pair *) VKQ32)[3] = vkq3;
                    } ਨਹੀਂਤਾਂ {
                        hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (ਅਚਲ uint8_t *) VKQ32, DV, ms_vec);

                        ਲਈ (uint32_t j = 0; j < current_ਖੰਡ_ਮਾਪ; j += 2) {
                            ਜੇ (j + 1 == current_ਖੰਡ_ਮਾਪ) {
                                HVX_ਸਦਿਸ਼ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                                hvx_mad_f32_f16_aa_vec(VKQ32, v_ptr, S0, DV);
                                ਰੋਕੋ;
                            }

                            HVX_ਸਦਿਸ਼ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            HVX_ਸਦਿਸ਼ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));

                            hvx_mad_f32_f16_aa_rx2_vec(VKQ32, v_ptr, v_ptr + factx->ਮਾਪ_v_ਪੰਕਤੀ_padded, S0, S1, DV);
                            v_ptr += stride_v2;
                        }
                    }
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, ਸਿਰ_ir);

                // prefetch V and mask for step + 2 after last head finished V accumulation
                ਜੇ (g + 1 == G_local && ਕਦਮ + 2 < n_ਸਰਗਰਮ_ਖੰਡਾਂ) {
                    ਅਚਲ uint32_t ਅਗਲਾ_ib = ਸਰਗਰਮ_ਖੰਡਾਂ[ਕਦਮ + 2];
                    ਅਚਲ uint32_t ਅਗਲਾ_ic_ਸ਼ੁਰੂ = ਅਗਲਾ_ib * FLASH_ATTN_ਖੰਡ_ਮਾਪ;
                    ਅਚਲ uint32_t ਅਗਲਾ_ਖੰਡ_ਮਾਪ = MIN(FLASH_ATTN_ਖੰਡ_ਮਾਪ, nek1 - ਅਗਲਾ_ic_ਸ਼ੁਰੂ);

                    // V
                    ਅਚਲ dma_addr_t v_src = v->ਅੰਕੜੇ + ਅਗਲਾ_ic_ਸ਼ੁਰੂ*nbv1 + iv2*nbv2 + iv3*nbv3;
                    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(v_ਅਧਾਰ, v_src), factx->ਮਾਪ_v_ਪੰਕਤੀ_padded, nbv1, ਮਾਪ_v_ਪੰਕਤੀ, ਅਗਲਾ_ਖੰਡ_ਮਾਪ);

                    // Mask
                    ਜੇ (ਨਕਾਬ) {
                        ਅਚਲ dma_addr_t m_src = mp_ਅਧਾਰ + ਅਗਲਾ_ic_ਸ਼ੁਰੂ * ਆਕਾਰ(__fp16);
                        dma_cache_dm_ਧੱਕੋ(dma_q, &m_cache, ਅਗਲਾ_ib, m_src, ਅਗਲਾ_ਖੰਡ_ਮਾਪ * 2, ਅਗਲਾ_ਖੰਡ_ਮਾਪ * 2, ਅਗਲਾ_ਖੰਡ_ਮਾਪ * 2, 1);
                    }
                }
            } // end for g
        } // end for step

        ਲਈ (uint32_t g = 0; g < G_local; ++g) {
            ਅਚਲ uint32_t ਸਿਰ_ir = ir + g;
            ਅਚਲ uint32_t h = heads[g];
            ਦਸ਼ਮਲਵ * VKQ32 = vkq_ptrs[g];

            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_O_PROC, ਸਿਰ_ir);

            ਦਸ਼ਮਲਵ M = hvx_vec_get_f32(M_vec[g]);
            ਦਸ਼ਮਲਵ S = hvx_vec_get_f32(S_vec[g]);

            ਜੇ (sinks) {
                ਅਚਲ ਦਸ਼ਮਲਵ s = factx->spad_sinks[h];

                ਦਸ਼ਮਲਵ vs = 1.0f;

                ਜੇ (s > M) {
                    HVX_ਸਦਿਸ਼ diff_vec = hvx_vec_splat_f32(M - s);
                    HVX_ਸਦਿਸ਼ ms_vec   = hvx_vec_exp_f32(diff_vec);
                    hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (ਅਚਲ uint8_t *) VKQ32, DV, ms_vec);

                    ਦਸ਼ਮਲਵ ms = hvx_vec_get_f32(ms_vec);
                    S = S * ms + vs;
                } ਨਹੀਂਤਾਂ {
                    HVX_ਸਦਿਸ਼ diff_vec = hvx_vec_splat_f32(s - M);
                    vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                    S += vs;
                }
            }

            ਅਚਲ ਦਸ਼ਮਲਵ S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            hvx_scale_f32_aa((uint8_t *) VKQ32, (ਅਚਲ uint8_t *) VKQ32, DV, S_inv);

            ਅਚਲ uint32_t r_iq3 = fastdiv(ਸਿਰ_ir, &factx->src0_div21);
            ਅਚਲ uint32_t r_iq2 = fastdiv(ਸਿਰ_ir - r_iq3*neq2*neq1, &factx->src0_div1);
            ਅਚਲ uint32_t r_iq1 = (ਸਿਰ_ir - r_iq3*neq2*neq1 - r_iq2 * neq1);

            uint8_t * dst_ptr = (uint8_t *) dst->ਅੰਕੜੇ + r_iq2 * dst->nb[1] + r_iq1 * dst->nb[2] + r_iq3 * dst->nb[3];

            ਜੇ (dst->type == HTP_TYPE_F32) {
                hvx_copy_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
            } ਨਹੀਂਤਾਂ ਜੇ (dst->type == HTP_TYPE_F16) {
                hvx_copy_f16_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, ਸਿਰ_ir);
        }

        ir += G_local;
    }
}

// ============================================================================
// HMX Phase args and thread logic
// ============================================================================

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਬਣਤਰ hmx_fa_context * factx;
    uint32_t                kv_ਪੰਕਤੀਆਂ;
    size_t                  src_stride;
    ਖਾਲੀ *                  curr_k;
    uint32_t                kv_ਸ਼ੁਰੂ;
    uint32_t                ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
    size_t                  buf_idx;
} fa_k_int_ਦਲੀਲਾਂ_t;

ਸਥਿਰ ਖਾਲੀ fa_k_interleave_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    fa_k_int_ਦਲੀਲਾਂ_t *       ਦਲੀਲਾਂ_2  = (fa_k_int_ਦਲੀਲਾਂ_t *) ਅੰਕੜੇ;
    ਬਣਤਰ hmx_fa_context * factx = ਦਲੀਲਾਂ_2->factx;

    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = ਦਲੀਲਾਂ_2->kv_ਪੰਕਤੀਆਂ;
    ਅਚਲ uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t = ਦਲੀਲਾਂ_2->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
    ਅਚਲ uint32_t ਸ਼ੁਰੂ      = i * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
    ਅਚਲ uint32_t ਸਿਰਾ        = (uint32_t) hex_smin(ਸ਼ੁਰੂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t, ਕੁੱਲ_ਪੰਕਤੀਆਂ);

    ਜੇ (ਸ਼ੁਰੂ >= ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
        ਵਾਪਸ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (ਦਲੀਲਾਂ_2->kv_ਸ਼ੁਰੂ + ਸ਼ੁਰੂ));
    ਜੇ (factx->k->type == HTP_TYPE_Q8_0) {
        ਲਈ (uint32_t r = ਸ਼ੁਰੂ; r < ਸਿਰਾ; ++r) {
            __fp16 * ਪੰਕਤੀ_k = (__fp16 *)((ਅੱਖਰ *)ਦਲੀਲਾਂ_2->curr_k + r * ਦਲੀਲਾਂ_2->src_stride * ਆਕਾਰ(__fp16));
            hvx_dequantize_ਪੰਕਤੀ_flat_q8_0_f16(ਪੰਕਤੀ_k, ਪੰਕਤੀ_k, factx->DK);
        }
    }
    hmx_interleave_ਪੰਕਤੀਆਂ_ਮੰਜ਼ਿਲ_tiles(factx->vtcm_k_tiles[ਦਲੀਲਾਂ_2->buf_idx], (ਅਚਲ __fp16 *) ਦਲੀਲਾਂ_2->curr_k, ਕੁੱਲ_ਪੰਕਤੀਆਂ, factx->DK_pad,
                             ਦਲੀਲਾਂ_2->src_stride, ਸ਼ੁਰੂ, ਸਿਰਾ);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (ਦਲੀਲਾਂ_2->kv_ਸ਼ੁਰੂ + ਸ਼ੁਰੂ));
}

ਸਥਿਰ ਖਾਲੀ fa_phase_k_interleave(ਬਣਤਰ hmx_fa_context * factx, uint32_t kv_ਪੰਕਤੀਆਂ, size_t src_stride, ਖਾਲੀ * curr_k, uint32_t kv_ਸ਼ੁਰੂ, size_t buf_idx) {
    work_ਕਤਾਰ_t wp = factx->octx->ctx->work_ਕਤਾਰ;
    uint32_t n = 1;
    ਜੇ (factx->n_ਧਾਗੇ > 1 && kv_ਪੰਕਤੀਆਂ >= factx->n_ਧਾਗੇ * 2) {
        n = factx->n_ਧਾਗੇ;
    }
    uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t = hex_align_ਉੱਪਰ(hmx_ceil_div(kv_ਪੰਕਤੀਆਂ, n), 2);
    fa_k_int_ਦਲੀਲਾਂ_t ਦਲੀਲਾਂ_2 = { factx, kv_ਪੰਕਤੀਆਂ, src_stride, curr_k, kv_ਸ਼ੁਰੂ, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t, buf_idx };
    ਜੇ (n > 1) {
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(wp, fa_k_interleave_thread, &ਦਲੀਲਾਂ_2, n);
    } ਨਹੀਂਤਾਂ {
        fa_k_interleave_thread(1, 0, &ਦਲੀਲਾਂ_2);
    }
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਬਣਤਰ hmx_fa_context * factx;
    uint32_t                kv_ਪੰਕਤੀਆਂ;
    size_t                  src_stride;
    ਖਾਲੀ *                  v_src;
    ਖਾਲੀ *                  v_tiles_dst;
    size_t                  n_ਖਾਨਾ_tiles;
    uint32_t                kv_ਸ਼ੁਰੂ;
    uint32_t                ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
} fa_v_int_ਦਲੀਲਾਂ_t;

ਸਥਿਰ ਖਾਲੀ fa_v_interleave_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    fa_v_int_ਦਲੀਲਾਂ_t *       ਦਲੀਲਾਂ_2  = (fa_v_int_ਦਲੀਲਾਂ_t *) ਅੰਕੜੇ;
    ਬਣਤਰ hmx_fa_context * factx = ਦਲੀਲਾਂ_2->factx;

    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = ਦਲੀਲਾਂ_2->kv_ਪੰਕਤੀਆਂ;
    ਅਚਲ uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t = ਦਲੀਲਾਂ_2->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
    ਅਚਲ uint32_t ਸ਼ੁਰੂ      = i * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
    ਅਚਲ uint32_t ਸਿਰਾ        = (uint32_t) hex_smin(ਸ਼ੁਰੂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t, ਕੁੱਲ_ਪੰਕਤੀਆਂ);

    ਜੇ (ਸ਼ੁਰੂ >= ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
        ਵਾਪਸ;
    }

    __fp16 * v_tiles_dst = (__fp16 *) ਦਲੀਲਾਂ_2->v_tiles_dst;

    ਬਣਤਰ htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (ਦਲੀਲਾਂ_2->kv_ਸ਼ੁਰੂ + ਸ਼ੁਰੂ));
    ਜੇ (factx->v->type == HTP_TYPE_Q8_0) {
        ਲਈ (uint32_t r = ਸ਼ੁਰੂ; r < ਸਿਰਾ; ++r) {
            __fp16 * ਪੰਕਤੀ_v = (__fp16 *)((ਅੱਖਰ *)ਦਲੀਲਾਂ_2->v_src + r * ਦਲੀਲਾਂ_2->src_stride * ਆਕਾਰ(__fp16));
            hvx_dequantize_ਪੰਕਤੀ_flat_q8_0_f16(ਪੰਕਤੀ_v, ਪੰਕਤੀ_v, factx->DV);
        }
    }
    hmx_interleave_ਖਾਨੇ_ਗਿਣਤੀ_ਮੰਜ਼ਿਲ_tiles(v_tiles_dst, (ਅਚਲ __fp16 *) ਦਲੀਲਾਂ_2->v_src, ਕੁੱਲ_ਪੰਕਤੀਆਂ, factx->DV_pad,
                             ਦਲੀਲਾਂ_2->src_stride, (uint32_t) ਦਲੀਲਾਂ_2->n_ਖਾਨਾ_tiles, ਸ਼ੁਰੂ, ਸਿਰਾ);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (ਦਲੀਲਾਂ_2->kv_ਸ਼ੁਰੂ + ਸ਼ੁਰੂ));
}

ਸਥਿਰ ਖਾਲੀ fa_phase_v_interleave(ਬਣਤਰ hmx_fa_context * factx,
                                  uint32_t                kv_ਪੰਕਤੀਆਂ,
                                  size_t                  src_stride,
                                  ਖਾਲੀ *                  v_src,
                                  ਖਾਲੀ *                  v_tiles_dst,
                                  size_t                  n_ਖਾਨਾ_tiles,
                                  uint32_t                kv_ਸ਼ੁਰੂ) {
    work_ਕਤਾਰ_t wp = factx->octx->ctx->work_ਕਤਾਰ;
    uint32_t n = 1;
    ਜੇ (factx->n_ਧਾਗੇ > 1 && kv_ਪੰਕਤੀਆਂ >= factx->n_ਧਾਗੇ * 2) {
        n = factx->n_ਧਾਗੇ;
    }
    uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t = hex_align_ਉੱਪਰ(hmx_ceil_div(kv_ਪੰਕਤੀਆਂ, n), 2);
    fa_v_int_ਦਲੀਲਾਂ_t ਦਲੀਲਾਂ_2 = { factx, kv_ਪੰਕਤੀਆਂ, src_stride, v_src, v_tiles_dst, n_ਖਾਨਾ_tiles, kv_ਸ਼ੁਰੂ, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t };
    ਜੇ (n > 1) {
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(wp, fa_v_interleave_thread, &ਦਲੀਲਾਂ_2, n);
    } ਨਹੀਂਤਾਂ {
        fa_v_interleave_thread(1, 0, &ਦਲੀਲਾਂ_2);
    }
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਬਣਤਰ hmx_fa_context *   factx;
    ਅਚਲ ਬਣਤਰ htp_tensor * q;
    uint32_t                  q_ਸ਼ੁਰੂ;
    uint32_t                  kv_ਸਿਰ;
    uint32_t                  ib3;
    size_t                    n_ਪੰਕਤੀਆਂ_g;
    size_t                    ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
    size_t                    n_ਪੰਕਤੀਆਂ_q;
    bool                      q_transposed;
    atomic_uint               barrier;
} fa_q_ਲੱਦੋ_ਦਲੀਲਾਂ_t;

ਸਥਿਰ ਖਾਲੀ fa_q_ਲੱਦੋ_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    fa_q_ਲੱਦੋ_ਦਲੀਲਾਂ_t *      ਦਲੀਲਾਂ_2  = (fa_q_ਲੱਦੋ_ਦਲੀਲਾਂ_t *) ਅੰਕੜੇ;
    ਬਣਤਰ hmx_fa_context * factx = ਦਲੀਲਾਂ_2->factx;

    ਅਚਲ size_t n_ਪੰਕਤੀਆਂ_g = ਦਲੀਲਾਂ_2->n_ਪੰਕਤੀਆਂ_g;
    ਅਚਲ size_t G        = factx->G;
    ਅਚਲ size_t DK       = factx->DK;

    // Partition the padded Q rows (g_br) across threads.
    // Keep start/end even so r and r+1 are always in the same thread's range.
    ਅਚਲ size_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t = ਦਲੀਲਾਂ_2->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
    ਅਚਲ size_t ਸ਼ੁਰੂ      = (size_t) i * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
    ਅਚਲ size_t ਸਿਰਾ        = hex_smin(ਸ਼ੁਰੂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t, factx->g_br);

    ਬਣਤਰ htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (ਦਲੀਲਾਂ_2->q_ਸ਼ੁਰੂ * G + ਸ਼ੁਰੂ));

    // Parallel initialization of per-block state
    {
        ਅਚਲ uint32_t g_br = factx->g_br;
        ਅਚਲ uint32_t DV   = factx->DV;

        ਅਚਲ size_t ਖਾਨਾ_vec_ਬਾਈਟਾਂ = factx->ਖਾਨਾ_vec_ਬਾਈਟਾਂ;
        ਅਚਲ size_t d_tile_ਬਾਈਟਾਂ  = factx->d_tile_ਬਾਈਟਾਂ;

        // Initialize vtcm_l_vec & vtcm_m_vec
        ਅਚਲ size_t l_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_t = hex_align_ਉੱਪਰ(ਖਾਨਾ_vec_ਬਾਈਟਾਂ / n, 128);
        ਅਚਲ size_t l_ਸ਼ੁਰੂ       = i * l_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_t;
        ਅਚਲ size_t l_ਸਿਰਾ         = hex_smin(l_ਸ਼ੁਰੂ + l_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_t, ਖਾਨਾ_vec_ਬਾਈਟਾਂ);

        ਅਚਲ size_t m_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_t = hex_align_ਉੱਪਰ(ਖਾਨਾ_vec_ਬਾਈਟਾਂ / n, 128);
        ਅਚਲ size_t m_ਸ਼ੁਰੂ       = i * m_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_t;
        ਅਚਲ size_t m_ਸਿਰਾ         = hex_smin(m_ਸ਼ੁਰੂ + m_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_t, ਖਾਨਾ_vec_ਬਾਈਟਾਂ);

        ਜੇ (l_ਸ਼ੁਰੂ < ਖਾਨਾ_vec_ਬਾਈਟਾਂ) {
            hvx_splat_u8_a((ਅੱਖਰ *) factx->vtcm_l_vec + l_ਸ਼ੁਰੂ, 0, l_ਸਿਰਾ - l_ਸ਼ੁਰੂ);
        }
        ਜੇ (m_ਸ਼ੁਰੂ < ਖਾਨਾ_vec_ਬਾਈਟਾਂ) {
            hvx_splat_f32_a((ਅੱਖਰ *) factx->vtcm_m_vec + m_ਸ਼ੁਰੂ, HTP_FA_M_INITIAL_VAL, (m_ਸਿਰਾ - m_ਸ਼ੁਰੂ) / ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        }

        // Zero the whole rescale region: vtcm_d_tiles[0], the optional vtcm_d_tiles[1]
        // and vtcm_d_inv_l are equal-sized and allocated back to back, so one run covers
        // them all.  The scatter only ever writes the diagonal, ignore the rest.
        ਅਚਲ size_t d_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_t = hex_align_ਉੱਪਰ(d_tile_ਬਾਈਟਾਂ / n, 128);
        ਅਚਲ size_t d_ਸ਼ੁਰੂ       = i * d_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_t;
        ਅਚਲ size_t d_ਸਿਰਾ         = hex_smin(d_ਸ਼ੁਰੂ + d_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_t, d_tile_ਬਾਈਟਾਂ);
        ਜੇ (d_ਸ਼ੁਰੂ < d_tile_ਬਾਈਟਾਂ) {
            hvx_splat_u8_a((ਅੱਖਰ *) factx->vtcm_d_tiles[0] + d_ਸ਼ੁਰੂ, 0, d_ਸਿਰਾ - d_ਸ਼ੁਰੂ);
        }
    }

    ਜੇ (ਸ਼ੁਰੂ < factx->g_br) {
        ਅਚਲ ਬਣਤਰ htp_tensor * q       = ਦਲੀਲਾਂ_2->q;
        ਅਚਲ uint32_t            q_ਸ਼ੁਰੂ = ਦਲੀਲਾਂ_2->q_ਸ਼ੁਰੂ;
        ਅਚਲ uint32_t            kv_ਸਿਰ = ਦਲੀਲਾਂ_2->kv_ਸਿਰ;
        ਅਚਲ uint32_t            ib3     = ਦਲੀਲਾਂ_2->ib3;

        ਅਚਲ bool use_q_dma = (factx->vtcm_q_dma != ਨਿਰਾ);

        __fp16 * q_tiles = factx->vtcm_q_tiles;
        ਅਚਲ size_t DK_pad = factx->DK_pad;
        ਜੇ (use_q_dma) {
            ਅਚਲ size_t g_ਪੰਕਤੀਆਂ_ਸਿਰਾ = hex_smin(ਸਿਰਾ, n_ਪੰਕਤੀਆਂ_g);
            ਅਚਲ uint32_t d_ਹੱਦਬੰਦੀ = factx->is_q_fp32 ? DK / 32 : DK / 64;

            uint8_t * q_flat  = (uint8_t *) factx->vtcm_q_dma;
            ਜੇ (DK_pad != DK) {
                ਜੇ (factx->is_q_fp32) {
                    hmx_fa_q_prep_fp32_pad(q_tiles, q_flat, ਸ਼ੁਰੂ, ਸਿਰਾ, g_ਪੰਕਤੀਆਂ_ਸਿਰਾ, DK, DK_pad, G, ਦਲੀਲਾਂ_2->n_ਪੰਕਤੀਆਂ_q, &factx->div_G, ਦਲੀਲਾਂ_2->q_transposed);
                } ਨਹੀਂਤਾਂ {
                    hmx_fa_q_prep_fp16_pad(q_tiles, q_flat, ਸ਼ੁਰੂ, ਸਿਰਾ, g_ਪੰਕਤੀਆਂ_ਸਿਰਾ, DK, DK_pad, G, ਦਲੀਲਾਂ_2->n_ਪੰਕਤੀਆਂ_q, &factx->div_G, ਦਲੀਲਾਂ_2->q_transposed);
                }
            } ਨਹੀਂਤਾਂ ਜੇ (factx->is_q_fp32) {
                ਚੋਣ (d_ਹੱਦਬੰਦੀ) {
                ਹਾਲਤ 2:  hmx_fa_q_prep_fp32_d2(q_tiles, q_flat, ਸ਼ੁਰੂ, ਸਿਰਾ, g_ਪੰਕਤੀਆਂ_ਸਿਰਾ, DK, G, ਦਲੀਲਾਂ_2->n_ਪੰਕਤੀਆਂ_q, &factx->div_G, ਦਲੀਲਾਂ_2->q_transposed); ਰੋਕੋ;
                ਹਾਲਤ 4:  hmx_fa_q_prep_fp32_d4(q_tiles, q_flat, ਸ਼ੁਰੂ, ਸਿਰਾ, g_ਪੰਕਤੀਆਂ_ਸਿਰਾ, DK, G, ਦਲੀਲਾਂ_2->n_ਪੰਕਤੀਆਂ_q, &factx->div_G, ਦਲੀਲਾਂ_2->q_transposed); ਰੋਕੋ;
                ਮੂਲ: hmx_fa_q_prep_fp32(   q_tiles, q_flat, ਸ਼ੁਰੂ, ਸਿਰਾ, g_ਪੰਕਤੀਆਂ_ਸਿਰਾ, DK, G, ਦਲੀਲਾਂ_2->n_ਪੰਕਤੀਆਂ_q, &factx->div_G, d_ਹੱਦਬੰਦੀ, ਦਲੀਲਾਂ_2->q_transposed); ਰੋਕੋ;
                }
            } ਨਹੀਂਤਾਂ {
                ਚੋਣ (d_ਹੱਦਬੰਦੀ) {
                ਹਾਲਤ 1:  hmx_fa_q_prep_fp16_d1(q_tiles, q_flat, ਸ਼ੁਰੂ, ਸਿਰਾ, g_ਪੰਕਤੀਆਂ_ਸਿਰਾ, DK, G, ਦਲੀਲਾਂ_2->n_ਪੰਕਤੀਆਂ_q, &factx->div_G, ਦਲੀਲਾਂ_2->q_transposed); ਰੋਕੋ;
                ਹਾਲਤ 2:  hmx_fa_q_prep_fp16_d2(q_tiles, q_flat, ਸ਼ੁਰੂ, ਸਿਰਾ, g_ਪੰਕਤੀਆਂ_ਸਿਰਾ, DK, G, ਦਲੀਲਾਂ_2->n_ਪੰਕਤੀਆਂ_q, &factx->div_G, ਦਲੀਲਾਂ_2->q_transposed); ਰੋਕੋ;
                ਮੂਲ: hmx_fa_q_prep_fp16(   q_tiles, q_flat, ਸ਼ੁਰੂ, ਸਿਰਾ, g_ਪੰਕਤੀਆਂ_ਸਿਰਾ, DK, G, ਦਲੀਲਾਂ_2->n_ਪੰਕਤੀਆਂ_q, &factx->div_G, d_ਹੱਦਬੰਦੀ, ਦਲੀਲਾਂ_2->q_transposed); ਰੋਕੋ;
                }
            }
        } ਨਹੀਂਤਾਂ {
            // Fallback: direct-from-DDR/L2 path
            hmx_fa_q_prep_fallback(q_tiles, q->ਅੰਕੜੇ, q->nb[1], q->nb[2], q->nb[3],
                                   q_ਸ਼ੁਰੂ, kv_ਸਿਰ, ib3, ਸ਼ੁਰੂ, ਸਿਰਾ, n_ਪੰਕਤੀਆਂ_g, G, DK, DK_pad, factx->is_q_fp32, &factx->div_G);
        }
    }

    // Synchronize threads before zeroing out vtcm_o_tiles[0] to prevent race condition
    ਜੇ (n > 1) {
        atomic_fetch_sub(&ਦਲੀਲਾਂ_2->barrier, 1);
        ਜਦੋਂ (atomic_ਲੱਦੋ(&ਦਲੀਲਾਂ_2->barrier) > 0) {
            // spin wait
        }
    }

    // Zero out vtcm_o_tiles[0] as it was used as temp_q_vtcm
    {
        ਅਚਲ uint32_t g_br = factx->g_br;
        ਅਚਲ uint32_t DV   = factx->DV;
        ਅਚਲ size_t o_tile_ਬਾਈਟਾਂ  = factx->o_tile_ਬਾਈਟਾਂ;
        ਅਚਲ size_t o_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_t = hex_align_ਉੱਪਰ(o_tile_ਬਾਈਟਾਂ / n, 128);
        ਅਚਲ size_t o_ਸ਼ੁਰੂ       = i * o_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_t;
        ਅਚਲ size_t o_ਸਿਰਾ         = hex_smin(o_ਸ਼ੁਰੂ + o_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_t, o_tile_ਬਾਈਟਾਂ);
        ਜੇ (o_ਸ਼ੁਰੂ < o_tile_ਬਾਈਟਾਂ) {
            hvx_splat_u8_a((ਅੱਖਰ *) factx->vtcm_o_tiles[0] + o_ਸ਼ੁਰੂ, 0, o_ਸਿਰਾ - o_ਸ਼ੁਰੂ);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (ਦਲੀਲਾਂ_2->q_ਸ਼ੁਰੂ * G + ਸ਼ੁਰੂ));
}

ਸਥਿਰ ਖਾਲੀ fa_phase_q_ਲੱਦੋ(ਬਣਤਰ hmx_fa_context *   factx,
                            ਅਚਲ ਬਣਤਰ htp_tensor * q,
                            uint32_t                  q_ਸ਼ੁਰੂ,
                            uint32_t                  kv_ਸਿਰ,
                            uint32_t                  ib3,
                            size_t                    n_ਪੰਕਤੀਆਂ_g) {
    work_ਕਤਾਰ_t wp = factx->octx->ctx->work_ਕਤਾਰ;
    uint32_t n = 1;
    ਜੇ (factx->n_ਧਾਗੇ > 1 && n_ਪੰਕਤੀਆਂ_g >= (size_t) (factx->n_ਧਾਗੇ * 2)) {
        n = factx->n_ਧਾਗੇ;
    }
    size_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t = hex_align_ਉੱਪਰ(hmx_ceil_div(factx->g_br, n), 2);
    ਅਚਲ uint32_t n_ਪੰਕਤੀਆਂ_q = hex_smin(factx->Br, factx->neq1 - q_ਸ਼ੁਰੂ);
    fa_q_ਲੱਦੋ_ਦਲੀਲਾਂ_t ਦਲੀਲਾਂ_2;
    ਦਲੀਲਾਂ_2.factx = factx;
    ਦਲੀਲਾਂ_2.q = q;
    ਦਲੀਲਾਂ_2.q_ਸ਼ੁਰੂ = q_ਸ਼ੁਰੂ;
    ਦਲੀਲਾਂ_2.kv_ਸਿਰ = kv_ਸਿਰ;
    ਦਲੀਲਾਂ_2.ib3 = ib3;
    ਦਲੀਲਾਂ_2.n_ਪੰਕਤੀਆਂ_g = n_ਪੰਕਤੀਆਂ_g;
    ਦਲੀਲਾਂ_2.ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t = ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
    ਦਲੀਲਾਂ_2.n_ਪੰਕਤੀਆਂ_q = n_ਪੰਕਤੀਆਂ_q;
    ਦਲੀਲਾਂ_2.q_transposed = q->nb[1] < q->nb[2];
    atomic_init(&ਦਲੀਲਾਂ_2.barrier, n);
    ਜੇ (n > 1) {
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(wp, fa_q_ਲੱਦੋ_thread, &ਦਲੀਲਾਂ_2, n);
    } ਨਹੀਂਤਾਂ {
        fa_q_ਲੱਦੋ_thread(1, 0, &ਦਲੀਲਾਂ_2);
    }
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਬਣਤਰ hmx_fa_context *   factx;
    ਅਚਲ ਬਣਤਰ htp_tensor * dst;
    ਅਚਲ __fp16 *            o_tile_src;
    uint32_t                  q_ਸ਼ੁਰੂ;
    uint32_t                  kv_ਸਿਰ;
    uint32_t                  ib3;
    size_t                    n_ਪੰਕਤੀਆਂ_g;
    size_t                    ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
} fa_o_store_ਦਲੀਲਾਂ_t;

ਸਥਿਰ ਖਾਲੀ fa_o_store_thread_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    fa_o_store_ਦਲੀਲਾਂ_t *     ਦਲੀਲਾਂ_2  = (fa_o_store_ਦਲੀਲਾਂ_t *) ਅੰਕੜੇ;
    ਬਣਤਰ hmx_fa_context * factx = ਦਲੀਲਾਂ_2->factx;

    ਅਚਲ size_t n_ਪੰਕਤੀਆਂ_g = ਦਲੀਲਾਂ_2->n_ਪੰਕਤੀਆਂ_g;
    ਅਚਲ size_t G        = factx->G;
    ਅਚਲ size_t DV       = factx->DV;

    ਅਚਲ size_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t = ਦਲੀਲਾਂ_2->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
    ਅਚਲ size_t ਸ਼ੁਰੂ      = (size_t) i * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
    ਅਚਲ size_t ਸਿਰਾ        = hex_smin(ਸ਼ੁਰੂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t, n_ਪੰਕਤੀਆਂ_g);

    ਜੇ (ਸ਼ੁਰੂ >= n_ਪੰਕਤੀਆਂ_g) {
        ਵਾਪਸ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (ਦਲੀਲਾਂ_2->q_ਸ਼ੁਰੂ * G + ਸ਼ੁਰੂ));

    ਅਚਲ ਬਣਤਰ htp_tensor * dst        = ਦਲੀਲਾਂ_2->dst;
    ਅਚਲ __fp16 *            o_tile_src = ਦਲੀਲਾਂ_2->o_tile_src;
    ਅਚਲ uint32_t            q_ਸ਼ੁਰੂ    = ਦਲੀਲਾਂ_2->q_ਸ਼ੁਰੂ;
    ਅਚਲ uint32_t            kv_ਸਿਰ    = ਦਲੀਲਾਂ_2->kv_ਸਿਰ;
    ਅਚਲ uint32_t            ib3        = ਦਲੀਲਾਂ_2->ib3;

    ਅਚਲ size_t DV_pad = factx->DV_pad;

    size_t q_idx = fastdiv(ਸ਼ੁਰੂ, &factx->div_G);
    size_t h_idx = fastmodulo(ਸ਼ੁਰੂ, G, &factx->div_G);

    ਲਈ (size_t r = ਸ਼ੁਰੂ; r < ਸਿਰਾ; ++r) {
        ਦਸ਼ਮਲਵ * ਬਾਹਰਲਾ = (ਦਸ਼ਮਲਵ *) ((uint8_t *) dst->ਅੰਕੜੇ + (kv_ਸਿਰ * G + h_idx) * dst->nb[1] +
                                 (q_ਸ਼ੁਰੂ + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        size_t         r1            = r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        ਅਚਲ __fp16 * tile_ਪੰਕਤੀ_ਅਧਾਰ = o_tile_src + r0 * HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ * DV_pad;

        ਲਈ (uint32_t d = 0; d < DV / 32; ++d) {
            ਅਚਲ HVX_ਸਦਿਸ਼ * ਵਿੱਚ_tile = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (tile_ਪੰਕਤੀ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS);
            HVX_ਸਦਿਸ਼_Pair     vp      = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32_shuff(ਵਿੱਚ_tile[r1 / 2]);
            ਜੇ (r1 % 2 == 0) {
                *(HVX_UVector *) (ਬਾਹਰਲਾ + d * 32) = Q6_V_lo_W(vp);
            } ਨਹੀਂਤਾਂ {
                *(HVX_UVector *) (ਬਾਹਰਲਾ + d * 32) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail: DV not a multiple of 32 (e.g. 72 -> last 8 lanes). Partial vector-write
        // for the remaining (DV % 32) floats.
        ਅਚਲ uint32_t d_ਪੂਛ = DV / 32;
        ਅਚਲ uint32_t rem    = DV - d_ਪੂਛ * 32;
        ਜੇ (rem) {
            ਅਚਲ HVX_ਸਦਿਸ਼ * ਵਿੱਚ_tile = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (tile_ਪੰਕਤੀ_ਅਧਾਰ + d_ਪੂਛ * HMX_FP16_TILE_N_ELMS);
            HVX_ਸਦਿਸ਼_Pair     vp      = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32_shuff(ਵਿੱਚ_tile[r1 / 2]);
            HVX_ਸਦਿਸ਼         vd      = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((ਖਾਲੀ *) (ਬਾਹਰਲਾ + d_ਪੂਛ * 32), rem * ਆਕਾਰ(ਦਸ਼ਮਲਵ), vd);
        }

        h_idx++;
        ਜੇ (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (ਦਲੀਲਾਂ_2->q_ਸ਼ੁਰੂ * G + ਸ਼ੁਰੂ));
}

ਸਥਿਰ ਖਾਲੀ fa_o_store_thread_f16(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    fa_o_store_ਦਲੀਲਾਂ_t *     ਦਲੀਲਾਂ_2  = (fa_o_store_ਦਲੀਲਾਂ_t *) ਅੰਕੜੇ;
    ਬਣਤਰ hmx_fa_context * factx = ਦਲੀਲਾਂ_2->factx;

    ਅਚਲ size_t n_ਪੰਕਤੀਆਂ_g   = ਦਲੀਲਾਂ_2->n_ਪੰਕਤੀਆਂ_g;
    ਅਚਲ size_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t = ਦਲੀਲਾਂ_2->ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
    ਅਚਲ size_t G          = factx->G;
    ਅਚਲ size_t DV         = factx->DV;
    ਅਚਲ size_t ਸ਼ੁਰੂ      = (size_t) i * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t;
    ਅਚਲ size_t ਸਿਰਾ        = hex_smin(ਸ਼ੁਰੂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t, n_ਪੰਕਤੀਆਂ_g);

    ਜੇ (ਸ਼ੁਰੂ >= n_ਪੰਕਤੀਆਂ_g) {
        ਵਾਪਸ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (ਦਲੀਲਾਂ_2->q_ਸ਼ੁਰੂ * G + ਸ਼ੁਰੂ));

    ਅਚਲ ਬਣਤਰ htp_tensor * dst        = ਦਲੀਲਾਂ_2->dst;
    ਅਚਲ __fp16 *            o_tile_src = ਦਲੀਲਾਂ_2->o_tile_src;
    ਅਚਲ uint32_t            q_ਸ਼ੁਰੂ    = ਦਲੀਲਾਂ_2->q_ਸ਼ੁਰੂ;
    ਅਚਲ uint32_t            kv_ਸਿਰ    = ਦਲੀਲਾਂ_2->kv_ਸਿਰ;
    ਅਚਲ uint32_t            ib3        = ਦਲੀਲਾਂ_2->ib3;

    // O-tiles use the padded head dim (DV_pad); dst holds the real DV lanes.
    ਅਚਲ size_t DV_pad = factx->DV_pad;

    size_t q_idx = fastdiv(ਸ਼ੁਰੂ, &factx->div_G);
    size_t h_idx = fastmodulo(ਸ਼ੁਰੂ, G, &factx->div_G);

    ਲਈ (size_t r = ਸ਼ੁਰੂ; r < ਸਿਰਾ; ++r) {
        __fp16 * ਬਾਹਰਲਾ = (__fp16 *) ((uint8_t *) dst->ਅੰਕੜੇ + (kv_ਸਿਰ * G + h_idx) * dst->nb[1] +
                                   (q_ਸ਼ੁਰੂ + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        size_t         r1            = r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        ਅਚਲ __fp16 * tile_ਪੰਕਤੀ_ਅਧਾਰ = o_tile_src + r0 * HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ * DV_pad;

        ਲਈ (uint32_t d = 0; d < DV / 64; ++d) {
            ਅਚਲ __fp16 *     ਵਿੱਚ_dtile = tile_ਪੰਕਤੀ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS * 2;
            ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in0   = ((ਅਚਲ HVX_ਸਦਿਸ਼ *) ਵਿੱਚ_dtile) + r1 / 2;
            ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in1   = pv_in0 + 16;
            HVX_ਸਦਿਸ਼_Pair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            ਜੇ (r1 % 2 == 0) {
                *(HVX_UVector *) (ਬਾਹਰਲਾ + d * 64) = Q6_V_lo_W(vp);
            } ਨਹੀਂਤਾਂ {
                *(HVX_UVector *) (ਬਾਹਰਲਾ + d * 64) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail when DV is not a multiple of 64.
        ਅਚਲ uint32_t d_ਪੂਛ = DV / 64;
        ਅਚਲ uint32_t rem    = DV - d_ਪੂਛ * 64;
        ਜੇ (rem) {
            ਅਚਲ __fp16 *     ਵਿੱਚ_dtile = tile_ਪੰਕਤੀ_ਅਧਾਰ + d_ਪੂਛ * HMX_FP16_TILE_N_ELMS * 2;
            ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in0   = ((ਅਚਲ HVX_ਸਦਿਸ਼ *) ਵਿੱਚ_dtile) + r1 / 2;
            ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in1   = pv_in0 + 16;
            HVX_ਸਦਿਸ਼_Pair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            HVX_ਸਦਿਸ਼         vd       = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((ਖਾਲੀ *) (ਬਾਹਰਲਾ + d_ਪੂਛ * 64), rem * ਆਕਾਰ(__fp16), vd);
        }

        h_idx++;
        ਜੇ (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (ਦਲੀਲਾਂ_2->q_ਸ਼ੁਰੂ * G + ਸ਼ੁਰੂ));
}

ਸਥਿਰ ਖਾਲੀ fa_phase_o_store(ਬਣਤਰ hmx_fa_context *   factx,
                             ਅਚਲ ਬਣਤਰ htp_tensor * dst,
                             ਅਚਲ __fp16 *            o_tile_src,
                             uint32_t                  q_ਸ਼ੁਰੂ,
                             uint32_t                  kv_ਸਿਰ,
                             uint32_t                  ib3,
                             size_t                    n_ਪੰਕਤੀਆਂ_g) {
    work_ਕਤਾਰ_t wp = factx->octx->ctx->work_ਕਤਾਰ;
    uint32_t n = 1;
    ਜੇ (factx->n_ਧਾਗੇ > 1 && n_ਪੰਕਤੀਆਂ_g >= (size_t) (factx->n_ਧਾਗੇ * 2)) {
        n = factx->n_ਧਾਗੇ;
    }
    size_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t = hmx_ceil_div(n_ਪੰਕਤੀਆਂ_g, n);
    fa_o_store_ਦਲੀਲਾਂ_t ਦਲੀਲਾਂ_2 = { factx, dst, o_tile_src, q_ਸ਼ੁਰੂ, kv_ਸਿਰ, ib3, n_ਪੰਕਤੀਆਂ_g, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_t };
    ਕਾਮਾ_callback_t store_fn = factx->is_dst_fp32 ? fa_o_store_thread_f32 : fa_o_store_thread_f16;
    ਜੇ (n > 1) {
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(wp, store_fn, &ਦਲੀਲਾਂ_2, n);
    } ਨਹੀਂਤਾਂ {
        store_fn(1, 0, &ਦਲੀਲਾਂ_2);
    }
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਬਣਤਰ hmx_fa_context *   factx;
    size_t                    buf_idx;
    size_t                    kv_ਪੰਕਤੀਆਂ;
    size_t                    n_ਪੰਕਤੀਆਂ_g;
    size_t                    n_ਖਾਨਾ_tiles;
    size_t                    n_tiles_ਪ੍ਰਤੀ_bc;
    size_t                    n_ਪੰਕਤੀ_tiles;
    size_t                    n_ਪੰਕਤੀ_tiles_g_br;
    uint32_t                  Bc;
    uint32_t                  G;
    uint32_t                  kv_ਸਿਰ;
    uint32_t                  kv_ਸ਼ੁਰੂ;
    uint32_t                  q_ਸ਼ੁਰੂ;
    uint32_t                  ib3;
    bool                      has_alibi;  // true when max_bias != 0 (need slope * mask + add)
    __fp16 *                  slopes;
    ਅਚਲ ਬਣਤਰ htp_tensor * ਨਕਾਬ;
    ਅਚਲ __fp16 *            ਨਕਾਬ_vtcm;             // VTCM mask buffer base (NULL = DDR fallback)
    size_t                    ਨਕਾਬ_vtcm_ਪੰਕਤੀ_stride;  // elements (__fp16) per row in VTCM mask buffer
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ     thread_div;
} fa_softmax_ਦਲੀਲਾਂ_t;

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ fa_softmax_impl(
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ,
    ਅਚਲ bool has_ਨਕਾਬ,
    ਅਚਲ bool ਨਕਾਬ_ਪ੍ਰਸਾਰਣ,
    ਅਚਲ bool is_g1,
    ਅਚਲ bool has_alibi,
    ਅਚਲ bool has_softcap
) {
    fa_softmax_ਦਲੀਲਾਂ_t *     ਦਲੀਲਾਂ_2  = (fa_softmax_ਦਲੀਲਾਂ_t *) ਅੰਕੜੇ;
    ਬਣਤਰ hmx_fa_context * factx = ਦਲੀਲਾਂ_2->factx;

    ਅਚਲ size_t n_ਪੰਕਤੀਆਂ_g       = ਦਲੀਲਾਂ_2->n_ਪੰਕਤੀਆਂ_g;
    ਅਚਲ size_t kv_ਪੰਕਤੀਆਂ        = ਦਲੀਲਾਂ_2->kv_ਪੰਕਤੀਆਂ;
    ਅਚਲ size_t Bc             = ਦਲੀਲਾਂ_2->Bc;
    ਅਚਲ size_t G              = ਦਲੀਲਾਂ_2->G;
    ਅਚਲ size_t n_tiles_ਪ੍ਰਤੀ_bc = ਦਲੀਲਾਂ_2->n_tiles_ਪ੍ਰਤੀ_bc;
    ਅਚਲ size_t n_ਪੰਕਤੀ_vec_cnt  = hmx_ceil_div(n_ਪੰਕਤੀਆਂ_g, 64);
    ਅਚਲ uint32_t im3          = has_ਨਕਾਬ ? fastmodulo(ਦਲੀਲਾਂ_2->ib3, ਦਲੀਲਾਂ_2->ਨਕਾਬ->ne[3], &factx->src3_div3) : 0;

    size_t vec_ਸ਼ੁਰੂ = 0;
    size_t vec_ਸਿਰਾ   = n_ਪੰਕਤੀ_vec_cnt;
    ਜੇ (n > 1) {
        ਅਚਲ size_t vecs_ਪ੍ਰਤੀ_t = fastdiv(n_ਪੰਕਤੀ_vec_cnt + n - 1, &ਦਲੀਲਾਂ_2->thread_div);
        vec_ਸ਼ੁਰੂ = i * vecs_ਪ੍ਰਤੀ_t;
        vec_ਸਿਰਾ   = hex_smin(vec_ਸ਼ੁਰੂ + vecs_ਪ੍ਰਤੀ_t, n_ਪੰਕਤੀ_vec_cnt);
    }

    ਜੇ (vec_ਸ਼ੁਰੂ >= n_ਪੰਕਤੀ_vec_cnt) {
        ਵਾਪਸ;
    }

    ਬਣਤਰ htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (ਦਲੀਲਾਂ_2->q_ਸ਼ੁਰੂ * G + vec_ਸ਼ੁਰੂ * 64));

    // Per-thread row scratch: thread i uses bufs at offset i * 2 * stride
    ਅਚਲ size_t ਪੰਕਤੀ_buf_stride = factx->ਪੰਕਤੀ_buf_stride;
    HVX_ਸਦਿਸ਼ * my_ਪੰਕਤੀ_buf0    = factx->vtcm_ਪੰਕਤੀ_bufs + i * 2 * ਪੰਕਤੀ_buf_stride;
    HVX_ਸਦਿਸ਼ * my_ਪੰਕਤੀ_buf1    = my_ਪੰਕਤੀ_buf0 + ਪੰਕਤੀ_buf_stride;

    ਅਚਲ HVX_ਸਦਿਸ਼ v_neg_inf = Q6_Vh_vsplat_R(0xfbff);

    ਲਈ (size_t r_vec_idx = vec_ਸ਼ੁਰੂ; r_vec_idx < vec_ਸਿਰਾ; ++r_vec_idx) {
        HVX_ਸਦਿਸ਼ rowmax_acc_v = v_neg_inf;
        HVX_ਸਦਿਸ਼ rowsum_acc_v = Q6_V_vzero();
        HVX_ਸਦਿਸ਼ m_prev_v0    = factx->vtcm_m_vec[r_vec_idx * 2 + 0];
        HVX_ਸਦਿਸ਼ m_prev_v1    = factx->vtcm_m_vec[r_vec_idx * 2 + 1];

        HVX_ਸਦਿਸ਼ v_slopes = Q6_V_vzero();
        ਜੇ (has_alibi) {
            v_slopes = hvx_vmem(ਦਲੀਲਾਂ_2->slopes + r_vec_idx * 64);
        }

        ਲਈ (uint32_t r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            uint32_t r = r_vec_idx * 64 + r_vec_off;
            ਜੇ (r >= hex_align_ਉੱਪਰ(n_ਪੰਕਤੀਆਂ_g, 2)) {
                ਰੋਕੋ;
            }

            uint32_t r0 = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
            uint32_t r1 = r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;

            ਅਚਲ __fp16 * s_ld_ਅਧਾਰ = factx->vtcm_s_tiles[ਦਲੀਲਾਂ_2->buf_idx] + r0 * HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ * Bc;
            __fp16 *       p_st_ਅਧਾਰ = factx->vtcm_p_tiles[ਦਲੀਲਾਂ_2->buf_idx] + r0 * HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ * Bc;

            // Decode 2 rows from S tiles into per-thread row buffers
            ਜੇ (has_softcap) {
                ਅਚਲ HVX_ਸਦਿਸ਼ v_cap = hvx_vec_splat_f16(factx->logit_softcap);
                ਲਈ (size_t c = 0; c < kv_ਪੰਕਤੀਆਂ; c += 64) {
                    size_t             ci       = c / 64;
                    ਅਚਲ __fp16 *     ਵਿੱਚ_dtile = s_ld_ਅਧਾਰ + ci * HMX_FP16_TILE_N_ELMS * 2;
                    ਅਚਲ HVX_ਸਦਿਸ਼ * pv_s_in0 = ((ਅਚਲ HVX_ਸਦਿਸ਼ *) ਵਿੱਚ_dtile) + r1 / 2;
                    ਅਚਲ HVX_ਸਦਿਸ਼ * pv_s_in1 = pv_s_in0 + 16;

                    HVX_ਸਦਿਸ਼_Pair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    HVX_ਸਦਿਸ਼     v_s_row0  = Q6_V_lo_W(vp_s_drow);
                    HVX_ਸਦਿਸ਼     v_s_row1  = Q6_V_hi_W(vp_s_drow);

                    HVX_ਸਦਿਸ਼ t0   = hvx_vec_tanh_f16(v_s_row0);
                    my_ਪੰਕਤੀ_buf0[ci] = hvx_vec_mul_f16_f16(t0, v_cap);

                    HVX_ਸਦਿਸ਼ t1   = hvx_vec_tanh_f16(v_s_row1);
                    my_ਪੰਕਤੀ_buf1[ci] = hvx_vec_mul_f16_f16(t1, v_cap);
                }
            } ਨਹੀਂਤਾਂ {
                size_t c = 0;
                ਲਈ (; c + 64 < kv_ਪੰਕਤੀਆਂ; c += 128) {
                    size_t             ci0       = c / 64;
                    size_t             ci1       = ci0 + 1;
                    ਅਚਲ __fp16 *     ਵਿੱਚ_dtile0 = s_ld_ਅਧਾਰ + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                    ਅਚਲ __fp16 *     ਵਿੱਚ_dtile1 = s_ld_ਅਧਾਰ + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                    ਅਚਲ HVX_ਸਦਿਸ਼ * pv_s_in0_0 = ((ਅਚਲ HVX_ਸਦਿਸ਼ *) ਵਿੱਚ_dtile0) + r1 / 2;
                    ਅਚਲ HVX_ਸਦਿਸ਼ * pv_s_in1_0 = pv_s_in0_0 + 16;
                    ਅਚਲ HVX_ਸਦਿਸ਼ * pv_s_in0_1 = ((ਅਚਲ HVX_ਸਦਿਸ਼ *) ਵਿੱਚ_dtile1) + r1 / 2;
                    ਅਚਲ HVX_ਸਦਿਸ਼ * pv_s_in1_1 = pv_s_in0_1 + 16;

                    HVX_ਸਦਿਸ਼_Pair vp_s_drow0 = Q6_W_vdeal_VVR(*pv_s_in1_0, *pv_s_in0_0, -2);
                    my_ਪੰਕਤੀ_buf0[ci0]          = Q6_V_lo_W(vp_s_drow0);
                    my_ਪੰਕਤੀ_buf1[ci0]          = Q6_V_hi_W(vp_s_drow0);

                    HVX_ਸਦਿਸ਼_Pair vp_s_drow1 = Q6_W_vdeal_VVR(*pv_s_in1_1, *pv_s_in0_1, -2);
                    my_ਪੰਕਤੀ_buf0[ci1]          = Q6_V_lo_W(vp_s_drow1);
                    my_ਪੰਕਤੀ_buf1[ci1]          = Q6_V_hi_W(vp_s_drow1);
                }
                ਲਈ (; c < kv_ਪੰਕਤੀਆਂ; c += 64) {
                    size_t             ci       = c / 64;
                    ਅਚਲ __fp16 *     ਵਿੱਚ_dtile = s_ld_ਅਧਾਰ + ci * HMX_FP16_TILE_N_ELMS * 2;
                    ਅਚਲ HVX_ਸਦਿਸ਼ * pv_s_in0 = ((ਅਚਲ HVX_ਸਦਿਸ਼ *) ਵਿੱਚ_dtile) + r1 / 2;
                    ਅਚਲ HVX_ਸਦਿਸ਼ * pv_s_in1 = pv_s_in0 + 16;

                    HVX_ਸਦਿਸ਼_Pair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    my_ਪੰਕਤੀ_buf0[ci]          = Q6_V_lo_W(vp_s_drow);
                    my_ਪੰਕਤੀ_buf1[ci]          = Q6_V_hi_W(vp_s_drow);
                }
            }

            // Apply mask & compute rowmax(S)
            HVX_ਸਦਿਸ਼ v_slope0 = Q6_V_vzero();
            HVX_ਸਦਿਸ਼ v_slope1 = Q6_V_vzero();
            ਜੇ (has_alibi) {
                v_slope0 = hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, r_vec_off * 2));
                v_slope1 = (r + 1 < n_ਪੰਕਤੀਆਂ_g) ? hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, (r_vec_off + 1) * 2)) : Q6_V_vzero();
            }

            ਅਚਲ HVX_ਸਦਿਸ਼ v_ਹੱਦ = Q6_Vh_vsplat_R(0xcc00);  // fp16 -16.0

            HVX_ਸਦਿਸ਼ v_s_rowmax0 = v_neg_inf;
            HVX_ਸਦਿਸ਼ v_s_rowmax1 = v_neg_inf;
            ਜੇ (has_ਨਕਾਬ) {
                ਲਈ (size_t c = 0; c < kv_ਪੰਕਤੀਆਂ; c += 64) {
                    size_t         ci          = c / 64;
                    ਅਚਲ size_t   ne          = hex_smin(kv_ਪੰਕਤੀਆਂ - c, 64);
                    HVX_ਸਦਿਸ਼_Pred q_ਪੂਛ_keep = Q6_Q_vsetq2_R(ne * ਆਕਾਰ(__fp16));

                    HVX_ਸਦਿਸ਼ v_mask0, v_mask1;

                    ਜੇ (ਨਕਾਬ_ਪ੍ਰਸਾਰਣ) {
                        ਜੇ (is_g1) {
                            ਅਚਲ size_t qi0 = r + 0;
                            v_mask0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਦਲੀਲਾਂ_2->ਨਕਾਬ_vtcm + qi0 * ਦਲੀਲਾਂ_2->ਨਕਾਬ_vtcm_ਪੰਕਤੀ_stride + c);
                            v_mask1 = v_neg_inf;
                            ਜੇ (r + 1 < n_ਪੰਕਤੀਆਂ_g) {
                                ਅਚਲ size_t qi1 = r + 1;
                                v_mask1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਦਲੀਲਾਂ_2->ਨਕਾਬ_vtcm + qi1 * ਦਲੀਲਾਂ_2->ਨਕਾਬ_vtcm_ਪੰਕਤੀ_stride + c);
                            }
                        } ਨਹੀਂਤਾਂ {
                            ਅਚਲ size_t qi0 = fastdiv(r + 0, &factx->div_G);
                            v_mask0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਦਲੀਲਾਂ_2->ਨਕਾਬ_vtcm + qi0 * ਦਲੀਲਾਂ_2->ਨਕਾਬ_vtcm_ਪੰਕਤੀ_stride + c);
                            v_mask1 = v_neg_inf;
                            ਜੇ (r + 1 < n_ਪੰਕਤੀਆਂ_g) {
                                ਅਚਲ size_t qi1 = fastdiv(r + 1, &factx->div_G);
                                ਜੇ (qi1 == qi0) {
                                    v_mask1 = v_mask0;
                                } ਨਹੀਂਤਾਂ {
                                    v_mask1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਦਲੀਲਾਂ_2->ਨਕਾਬ_vtcm + qi1 * ਦਲੀਲਾਂ_2->ਨਕਾਬ_vtcm_ਪੰਕਤੀ_stride + c);
                                }
                            }
                        }
                    } ਨਹੀਂਤਾਂ {
                        // Head-dependent mask: pre-interleaved per row r.
                        ਅਚਲ size_t r0 = r + 0;
                        v_mask0 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਦਲੀਲਾਂ_2->ਨਕਾਬ_vtcm + r0 * ਦਲੀਲਾਂ_2->ਨਕਾਬ_vtcm_ਪੰਕਤੀ_stride + c);
                        v_mask1 = v_neg_inf;
                        ਜੇ (r + 1 < n_ਪੰਕਤੀਆਂ_g) {
                            ਅਚਲ size_t r1 = r + 1;
                            v_mask1 = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਦਲੀਲਾਂ_2->ਨਕਾਬ_vtcm + r1 * ਦਲੀਲਾਂ_2->ਨਕਾਬ_vtcm_ਪੰਕਤੀ_stride + c);
                        }
                    }

                    // Threshold: mask values below -16.0 are treated as -inf (causal mask).
                    HVX_ਸਦਿਸ਼_Pred q_keep0 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_ਹੱਦ), q_ਪੂਛ_keep);
                    HVX_ਸਦਿਸ਼_Pred q_keep1 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_ਹੱਦ), q_ਪੂਛ_keep);

                    // Scale mask values by log2(e) for base-2 calculations
                    ਅਚਲ HVX_ਸਦਿਸ਼ v_log2e = hvx_vec_splat_f16(EXP_LOG2E_F);
                    HVX_ਸਦਿਸ਼ v_mask0_scaled = hvx_vec_mul_f16_f16(v_mask0, v_log2e);
                    HVX_ਸਦਿਸ਼ v_mask1_scaled = hvx_vec_mul_f16_f16(v_mask1, v_log2e);

                    ਜੇ (has_alibi) {
                        HVX_ਸਦਿਸ਼ v_sm0 = hvx_vec_mul_f16_f16(v_mask0_scaled, v_slope0);
                        HVX_ਸਦਿਸ਼ v_sm1 = hvx_vec_mul_f16_f16(v_mask1_scaled, v_slope1);
                        my_ਪੰਕਤੀ_buf0[ci]  = Q6_V_vmux_QVV(q_keep0, hvx_vec_ਜੋੜੋ_f16_f16(my_ਪੰਕਤੀ_buf0[ci], v_sm0), v_neg_inf);
                        my_ਪੰਕਤੀ_buf1[ci]  = Q6_V_vmux_QVV(q_keep1, hvx_vec_ਜੋੜੋ_f16_f16(my_ਪੰਕਤੀ_buf1[ci], v_sm1), v_neg_inf);
                    } ਨਹੀਂਤਾਂ {
                        my_ਪੰਕਤੀ_buf0[ci] = Q6_V_vmux_QVV(q_keep0, hvx_vec_ਜੋੜੋ_f16_f16(my_ਪੰਕਤੀ_buf0[ci], v_mask0_scaled), v_neg_inf);
                        my_ਪੰਕਤੀ_buf1[ci] = Q6_V_vmux_QVV(q_keep1, hvx_vec_ਜੋੜੋ_f16_f16(my_ਪੰਕਤੀ_buf1[ci], v_mask1_scaled), v_neg_inf);
                    }

                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_ਪੰਕਤੀ_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_ਪੰਕਤੀ_buf1[ci]);
                }
            } ਨਹੀਂਤਾਂ {
                size_t c = 0;
                ਲਈ (; c + 64 < kv_ਪੰਕਤੀਆਂ; c += 128) {
                    size_t ci0 = c / 64;
                    size_t ci1 = ci0 + 1;
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_ਪੰਕਤੀ_buf0[ci0]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_ਪੰਕਤੀ_buf1[ci0]);
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_ਪੰਕਤੀ_buf0[ci1]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_ਪੰਕਤੀ_buf1[ci1]);
                }
                ਲਈ (; c < kv_ਪੰਕਤੀਆਂ; c += 64) {
                    size_t         ci          = c / 64;
                    ਅਚਲ size_t   ne          = hex_smin(kv_ਪੰਕਤੀਆਂ - c, 64);
                    HVX_ਸਦਿਸ਼_Pred q_ਪੂਛ_keep = Q6_Q_vsetq2_R(ne * ਆਕਾਰ(__fp16));
                    ਜੇ (ne < 64) {
                        my_ਪੰਕਤੀ_buf0[ci] = Q6_V_vmux_QVV(q_ਪੂਛ_keep, my_ਪੰਕਤੀ_buf0[ci], v_neg_inf);
                        my_ਪੰਕਤੀ_buf1[ci] = Q6_V_vmux_QVV(q_ਪੂਛ_keep, my_ਪੰਕਤੀ_buf1[ci], v_neg_inf);
                    }
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_ਪੰਕਤੀ_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_ਪੰਕਤੀ_buf1[ci]);
                }
            }

            v_s_rowmax0 = hvx_vec_reduce_max_f16(v_s_rowmax0);
            v_s_rowmax1 = hvx_vec_reduce_max_f16(v_s_rowmax1);

            // Splat m_prev[r], m_prev[r+1] from the float per-row accumulators and convert to fp16 vectors
            HVX_ਸਦਿਸ਼ v_m_prev0, v_m_prev1;
            ਜੇ (r_vec_off < 32) {
                HVX_ਸਦਿਸ਼ v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, r_vec_off * 4));
                v_m_prev0 = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(v0, v0);
                ਜੇ (r + 1 < n_ਪੰਕਤੀਆਂ_g) {
                    HVX_ਸਦਿਸ਼ v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v0, (r_vec_off + 1) * 4));
                    v_m_prev1 = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(v1, v1);
                } ਨਹੀਂਤਾਂ {
                    v_m_prev1 = Q6_V_vzero();
                }
            } ਨਹੀਂਤਾਂ {
                HVX_ਸਦਿਸ਼ v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v1, (r_vec_off - 32) * 4));
                v_m_prev0 = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(v0, v0);
                ਜੇ (r + 1 < n_ਪੰਕਤੀਆਂ_g) {
                    HVX_ਸਦਿਸ਼ v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_prev_v1, (r_vec_off + 1 - 32) * 4));
                    v_m_prev1 = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(v1, v1);
                } ਨਹੀਂਤਾਂ {
                    v_m_prev1 = Q6_V_vzero();
                }
            }

            HVX_ਸਦਿਸ਼ v_dup_m0 = Q6_Vhf_vmax_VhfVhf(v_m_prev0, v_s_rowmax0);
            HVX_ਸਦਿਸ਼ v_dup_m1 = Q6_Vhf_vmax_VhfVhf(v_m_prev1, v_s_rowmax1);

            // Insert row r, r+1 rowmax into rowmax_acc_v
            {
                HVX_ਸਦਿਸ਼_Pred p_ਸ਼ੁਰੂ = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_ਸਦਿਸ਼_Pred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_ਸਦਿਸ਼_Pred p_ਸਿਰਾ   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_ਸਦਿਸ਼_Pred p_lane0 = Q6_Q_and_QQn(p_mid, p_ਸ਼ੁਰੂ);
                HVX_ਸਦਿਸ਼_Pred p_lane1 = Q6_Q_and_QQn(p_ਸਿਰਾ, p_mid);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane0, v_dup_m0, rowmax_acc_v);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane1, v_dup_m1, rowmax_acc_v);
            }

            // Compute P = exp(S - m_new)
            ਅਚਲ HVX_ਸਦਿਸ਼ v_zero      = Q6_V_vzero();
            HVX_ਸਦਿਸ਼       v_p_rowsum0 = v_zero;
            HVX_ਸਦਿਸ਼       v_p_rowsum1 = v_zero;

            size_t c = 0;
            ਲਈ (; c + 64 < kv_ਪੰਕਤੀਆਂ; c += 128) {
                size_t     ci0          = c / 64;
                size_t     ci1          = ci0 + 1;

                HVX_ਸਦਿਸ਼ v_s_minus_m0_0 = Q6_Vqf16_vsub_VhfVhf(my_ਪੰਕਤੀ_buf0[ci0], v_dup_m0);
                HVX_ਸਦਿਸ਼ v_s_minus_m1_0 = Q6_Vqf16_vsub_VhfVhf(my_ਪੰਕਤੀ_buf1[ci0], v_dup_m1);
                HVX_ਸਦਿਸ਼ v_s_minus_m0_1 = Q6_Vqf16_vsub_VhfVhf(my_ਪੰਕਤੀ_buf0[ci1], v_dup_m0);
                HVX_ਸਦਿਸ਼ v_s_minus_m1_1 = Q6_Vqf16_vsub_VhfVhf(my_ਪੰਕਤੀ_buf1[ci1], v_dup_m1);

                HVX_ਸਦਿਸ਼ v_p_row0_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_0));
                HVX_ਸਦਿਸ਼ v_p_row1_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_0));
                HVX_ਸਦਿਸ਼ v_p_row0_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_1));
                HVX_ਸਦਿਸ਼ v_p_row1_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_1));

                __fp16 *     ਬਾਹਰਲਾ_dtile0  = p_st_ਅਧਾਰ + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                __fp16 *     ਬਾਹਰਲਾ_dtile1  = p_st_ਅਧਾਰ + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ਸਦਿਸ਼ * pv_p_out0_0  = ((HVX_ਸਦਿਸ਼ *) ਬਾਹਰਲਾ_dtile0) + r1 / 2;
                HVX_ਸਦਿਸ਼ * pv_p_out1_0  = pv_p_out0_0 + 16;
                HVX_ਸਦਿਸ਼ * pv_p_out0_1  = ((HVX_ਸਦਿਸ਼ *) ਬਾਹਰਲਾ_dtile1) + r1 / 2;
                HVX_ਸਦਿਸ਼ * pv_p_out1_1  = pv_p_out0_1 + 16;

                HVX_ਸਦਿਸ਼_Pair vp_p_dual0 = Q6_W_vshuff_VVR(v_p_row1_hf_0, v_p_row0_hf_0, -2);
                *pv_p_out0_0               = Q6_V_lo_W(vp_p_dual0);
                *pv_p_out1_0               = Q6_V_hi_W(vp_p_dual0);

                HVX_ਸਦਿਸ਼_Pair vp_p_dual1 = Q6_W_vshuff_VVR(v_p_row1_hf_1, v_p_row0_hf_1, -2);
                *pv_p_out0_1               = Q6_V_lo_W(vp_p_dual1);
                *pv_p_out1_1               = Q6_V_hi_W(vp_p_dual1);

                HVX_ਸਦਿਸ਼_Pair vp_p0_0 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32_shuff(v_p_row0_hf_0);
                HVX_ਸਦਿਸ਼_Pair vp_p1_0 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32_shuff(v_p_row1_hf_0);
                HVX_ਸਦਿਸ਼_Pair vp_p0_1 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32_shuff(v_p_row0_hf_1);
                HVX_ਸਦਿਸ਼_Pair vp_p1_1 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32_shuff(v_p_row1_hf_1);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_0), Q6_V_hi_W(vp_p0_0)));
                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_1), Q6_V_hi_W(vp_p0_1)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_0), Q6_V_hi_W(vp_p1_0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_1), Q6_V_hi_W(vp_p1_1)));
            }
            ਲਈ (size_t c_rem = c; c_rem < kv_ਪੰਕਤੀਆਂ; c_rem += 64) {
                size_t     ci           = c_rem / 64;
                HVX_ਸਦਿਸ਼ v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(my_ਪੰਕਤੀ_buf0[ci], v_dup_m0);
                HVX_ਸਦਿਸ਼ v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(my_ਪੰਕਤੀ_buf1[ci], v_dup_m1);

                HVX_ਸਦਿਸ਼ v_p_row0_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0));
                HVX_ਸਦਿਸ਼ v_p_row1_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1));
                __fp16 *     ਬਾਹਰਲਾ_dtile  = p_st_ਅਧਾਰ + ci * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ਸਦਿਸ਼ * pv_p_out0  = ((HVX_ਸਦਿਸ਼ *) ਬਾਹਰਲਾ_dtile) + r1 / 2;
                HVX_ਸਦਿਸ਼ * pv_p_out1  = pv_p_out0 + 16;

                HVX_ਸਦਿਸ਼_Pair vp_p_dual = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
                *pv_p_out0               = Q6_V_lo_W(vp_p_dual);
                *pv_p_out1               = Q6_V_hi_W(vp_p_dual);

                HVX_ਸਦਿਸ਼_Pair vp_p0 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32_shuff(v_p_row0_hf);
                HVX_ਸਦਿਸ਼_Pair vp_p1 = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32_shuff(v_p_row1_hf);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0), Q6_V_hi_W(vp_p0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1), Q6_V_hi_W(vp_p1)));
            }

            HVX_ਸਦਿਸ਼ rowsum0_sf = hvx_vec_reduce_ਜੋੜ_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum0));
            HVX_ਸਦਿਸ਼ rowsum1_sf = hvx_vec_reduce_ਜੋੜ_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum1));
            {
                HVX_ਸਦਿਸ਼ rv0_v = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(rowsum0_sf, rowsum0_sf);
                HVX_ਸਦਿਸ਼ rv1_v = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(rowsum1_sf, rowsum1_sf);

                HVX_ਸਦਿਸ਼_Pred p_ਸ਼ੁਰੂ = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_ਸਦਿਸ਼_Pred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_ਸਦਿਸ਼_Pred p_ਸਿਰਾ   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_ਸਦਿਸ਼_Pred p_lane0 = Q6_Q_and_QQn(p_mid, p_ਸ਼ੁਰੂ);
                HVX_ਸਦਿਸ਼_Pred p_lane1 = Q6_Q_and_QQn(p_ਸਿਰਾ, p_mid);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane0, rv0_v, rowsum_acc_v);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane1, rv1_v, rowsum_acc_v);
            }
        }

        // Inline fa_ml_update_and_build_d for this vector (lock-free and in parallel)
        HVX_ਸਦਿਸ਼_Pair rowmax_acc_pair    = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(rowmax_acc_v);
        HVX_ਸਦਿਸ਼     v_rowmax_acc_f32_0 = Q6_V_lo_W(rowmax_acc_pair);
        HVX_ਸਦਿਸ਼     v_rowmax_acc_f32_1 = Q6_V_hi_W(rowmax_acc_pair);

        HVX_ਸਦਿਸ਼ v_m_curr0 = Q6_Vsf_vmax_VsfVsf(m_prev_v0, v_rowmax_acc_f32_0);
        HVX_ਸਦਿਸ਼ v_m_curr1 = Q6_Vsf_vmax_VsfVsf(m_prev_v1, v_rowmax_acc_f32_1);

        HVX_ਸਦਿਸ਼ v_m_diff0 = HVX_OP_SUB_F32(m_prev_v0, v_m_curr0);
        HVX_ਸਦਿਸ਼ v_m_diff1 = HVX_OP_SUB_F32(m_prev_v1, v_m_curr1);

        HVX_ਸਦਿਸ਼ v_m_diff_f16   = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(v_m_diff0, v_m_diff1);
        HVX_ਸਦਿਸ਼ exp_m_diff_f16 = hvx_vec_exp2_f16(v_m_diff_f16);

        HVX_ਸਦਿਸ਼_Pair exp_m_diff_pair = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(exp_m_diff_f16);
        HVX_ਸਦਿਸ਼ exp_m_diff0 = Q6_V_lo_W(exp_m_diff_pair);
        HVX_ਸਦਿਸ਼ exp_m_diff1 = Q6_V_hi_W(exp_m_diff_pair);

        HVX_ਸਦਿਸ਼_Pair rowsum_acc_pair = hvx_vec_f16_ਮੰਜ਼ਿਲ_f32(rowsum_acc_v);
        HVX_ਸਦਿਸ਼     v_rowsum_acc_f32_0 = Q6_V_lo_W(rowsum_acc_pair);
        HVX_ਸਦਿਸ਼     v_rowsum_acc_f32_1 = Q6_V_hi_W(rowsum_acc_pair);

        HVX_ਸਦਿਸ਼ l_prev_v0 = factx->vtcm_l_vec[r_vec_idx * 2 + 0];
        HVX_ਸਦਿਸ਼ l_prev_v1 = factx->vtcm_l_vec[r_vec_idx * 2 + 1];
        HVX_ਸਦਿਸ਼ v_l_curr0 = HVX_OP_ਜੋੜੋ_F32(HVX_OP_MUL_F32(l_prev_v0, exp_m_diff0), v_rowsum_acc_f32_0);
        HVX_ਸਦਿਸ਼ v_l_curr1 = HVX_OP_ਜੋੜੋ_F32(HVX_OP_MUL_F32(l_prev_v1, exp_m_diff1), v_rowsum_acc_f32_1);

        factx->vtcm_m_vec[r_vec_idx * 2 + 0] = v_m_curr0;
        factx->vtcm_m_vec[r_vec_idx * 2 + 1] = v_m_curr1;
        factx->vtcm_l_vec[r_vec_idx * 2 + 0] = v_l_curr0;
        factx->vtcm_l_vec[r_vec_idx * 2 + 1] = v_l_curr1;

        // Build diagonal tile D = diag(exp(m_diff))
        ਅਚਲ HVX_ਸਦਿਸ਼     v_offsets = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) d_tile_scatter_offsets;
        ਅਚਲ HVX_ਸਦਿਸ਼_Pred q_32_ਨਕਾਬ = Q6_Q_vsetq_R(32 * ਆਕਾਰ(__fp16));
        HVX_ਸਦਿਸ਼           v_exp_m_diff = exp_m_diff_f16;

        __fp16 * ਅਚਲ d_tiles_ਬਾਹਰਲਾ = factx->vtcm_d_tiles[ਦਲੀਲਾਂ_2->buf_idx];

        size_t t0 = r_vec_idx * 2;
        ਜੇ (t0 < ਦਲੀਲਾਂ_2->n_ਪੰਕਤੀ_tiles) {
            ਅਚਲ HVX_ਸਦਿਸ਼ v_content = v_exp_m_diff;
            __fp16 *         ਬਾਹਰਲਾ_ਅਧਾਰ  = d_tiles_ਬਾਹਰਲਾ + t0 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_ਨਕਾਬ, (size_t) ਬਾਹਰਲਾ_ਅਧਾਰ, HMX_FP16_TILE_ਮਾਪ - 1, v_offsets, v_content);
        }

        size_t t1 = r_vec_idx * 2 + 1;
        ਜੇ (t1 < ਦਲੀਲਾਂ_2->n_ਪੰਕਤੀ_tiles) {
            ਅਚਲ HVX_ਸਦਿਸ਼ v_content = Q6_V_vror_VR(v_exp_m_diff, 64);
            __fp16 *         ਬਾਹਰਲਾ_ਅਧਾਰ  = d_tiles_ਬਾਹਰਲਾ + t1 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_ਨਕਾਬ, (size_t) ਬਾਹਰਲਾ_ਅਧਾਰ, HMX_FP16_TILE_ਮਾਪ - 1, v_offsets, v_content);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (ਦਲੀਲਾਂ_2->q_ਸ਼ੁਰੂ * G + vec_ਸ਼ੁਰੂ * 64));
}

ਸਥਿਰ ਖਾਲੀ fa_softmax_thread_nomask(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    fa_softmax_impl(n, i, ਅੰਕੜੇ,
                    /*has_mask=*/false,
                    /*mask_broadcast=*/false,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ਸਥਿਰ ਖਾਲੀ fa_softmax_thread_ਨਕਾਬ_ਪ੍ਰਸਾਰਣ_g1(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    fa_softmax_impl(n, i, ਅੰਕੜੇ,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/true,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ਸਥਿਰ ਖਾਲੀ fa_softmax_thread_ਨਕਾਬ_ਪ੍ਰਸਾਰਣ_gn(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    fa_softmax_impl(n, i, ਅੰਕੜੇ,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ਸਥਿਰ ਖਾਲੀ fa_softmax_thread(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    fa_softmax_ਦਲੀਲਾਂ_t *     ਦਲੀਲਾਂ_2  = (fa_softmax_ਦਲੀਲਾਂ_t *) ਅੰਕੜੇ;
    ਬਣਤਰ hmx_fa_context * factx = ਦਲੀਲਾਂ_2->factx;

    ਅਚਲ bool has_ਨਕਾਬ       = (ਦਲੀਲਾਂ_2->ਨਕਾਬ != ਨਿਰਾ);
    ਅਚਲ bool ਨਕਾਬ_ਪ੍ਰਸਾਰਣ = factx->ਨਕਾਬ_ਪ੍ਰਸਾਰਣ;
    ਅਚਲ bool is_g1          = (ਦਲੀਲਾਂ_2->G == 1);
    ਅਚਲ bool has_alibi      = ਦਲੀਲਾਂ_2->has_alibi;
    ਅਚਲ bool has_softcap    = factx->has_softcap;

    fa_softmax_impl(n, i, ਅੰਕੜੇ, has_ਨਕਾਬ, ਨਕਾਬ_ਪ੍ਰਸਾਰਣ, is_g1, has_alibi, has_softcap);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_fa_ਲੱਦੋ_sinks(ਅਚਲ ਦਸ਼ਮਲਵ * sinks_ਅੰਕੜੇ, size_t idx, size_t ਗਿਣਤੀ_2) {
    ਅਚਲ HVX_ਸਦਿਸ਼ * v_src    = (ਅਚਲ HVX_ਸਦਿਸ਼ *) sinks_ਅੰਕੜੇ + (idx >> 5);
    ਅਚਲ size_t       lane_off = idx & 31;
    ਅਚਲ size_t       byte_off = lane_off * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਵਾਪਸ (lane_off + ਗਿਣਤੀ_2 > 32)
        ? Q6_V_valign_VVR(v_src[1], v_src[0], byte_off)
        : Q6_V_vror_VR(v_src[0], byte_off);
}

ਸਥਿਰ __attribute__((noinline)) ਖਾਲੀ fa_build_d_diag_inv_l(ਬਣਤਰ hmx_fa_context * factx,
                                                            size_t                  n_ਪੰਕਤੀ_tiles,
                                                            size_t                  n_ਪੰਕਤੀ_tiles_g_br,
                                                            uint32_t                kv_ਸਿਰ,
                                                            size_t                  n_ਪੰਕਤੀਆਂ_g) {
    (ਖਾਲੀ) n_ਪੰਕਤੀ_tiles_g_br;

    // sinks: l += exp(sink - m * ln2) in f32, m and l are in the log2 domain
    ਜੇ (factx->sinks) {
        ਅਚਲ ਦਸ਼ਮਲਵ *    sinks_ਅੰਕੜੇ   = factx->vtcm_sinks;
        ਅਚਲ size_t     G            = factx->G;
        ਅਚਲ size_t     ਸਿਰ_ਅਧਾਰ    = (size_t) kv_ਸਿਰ * G;
        ਅਚਲ HVX_ਸਦਿਸ਼ v_ln2        = Q6_V_vsplat_R(EXP_LOGN2);
        ਅਚਲ HVX_ਸਦਿਸ਼ v_ਸਿਰ_sinks = (G <= 32) ? hvx_fa_ਲੱਦੋ_sinks(sinks_ਅੰਕੜੇ, ਸਿਰ_ਅਧਾਰ, G) : Q6_V_vzero();
        HVX_ਸਦਿਸ਼       v_sink       = Q6_V_vzero();
        size_t           g_prev       = SIZE_MAX;

        ਲਈ (size_t i = 0; i < n_ਪੰਕਤੀ_tiles; ++i) {
            // the sink pattern of a tile only depends on (i * 32) % G, rebuild it when that changes
            ਅਚਲ size_t r0 = i * 32;
            ਅਚਲ size_t g0 = fastmodulo(r0, G, &factx->div_G);
            ਜੇ (g0 != g_prev) {
                ਅਚਲ size_t rem_g = G - g0;
                ਜੇ (G <= 32) {
                    ਅਚਲ HVX_ਸਦਿਸ਼ v_lo = Q6_V_vror_VR(v_ਸਿਰ_sinks, g0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                    ਅਚਲ HVX_ਸਦਿਸ਼ v_hi = Q6_V_vror_VR(v_ਸਿਰ_sinks, 128 - rem_g * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(rem_g * ਆਕਾਰ(ਦਸ਼ਮਲਵ)), v_lo, v_hi);
                    ਲਈ (size_t len = G; len < 32; len <<= 1) {
                        ਅਚਲ HVX_ਸਦਿਸ਼_Pred q_keep    = Q6_Q_vsetq_R(len * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                        ਅਚਲ HVX_ਸਦਿਸ਼     v_shifted = Q6_V_vror_VR(v_sink, 128 - len * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                        v_sink = Q6_V_vmux_QVV(q_keep, v_sink, v_shifted);
                    }
                } ਨਹੀਂਤਾਂ ਜੇ (rem_g >= 32) {
                    v_sink = hvx_fa_ਲੱਦੋ_sinks(sinks_ਅੰਕੜੇ, ਸਿਰ_ਅਧਾਰ + g0, 32);
                } ਨਹੀਂਤਾਂ {
                    ਅਚਲ HVX_ਸਦਿਸ਼ v_lo = hvx_fa_ਲੱਦੋ_sinks(sinks_ਅੰਕੜੇ, ਸਿਰ_ਅਧਾਰ + g0, rem_g);
                    ਅਚਲ HVX_ਸਦਿਸ਼ v_hi = hvx_fa_ਲੱਦੋ_sinks(sinks_ਅੰਕੜੇ, ਸਿਰ_ਅਧਾਰ, 32 - rem_g);
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq_R(rem_g * ਆਕਾਰ(ਦਸ਼ਮਲਵ)), v_lo, Q6_V_vror_VR(v_hi, 128 - rem_g * ਆਕਾਰ(ਦਸ਼ਮਲਵ)));
                }
                g_prev = g0;
            }

            HVX_ਸਦਿਸ਼ v_d = HVX_OP_SUB_F32(v_sink, HVX_OP_MUL_F32(factx->vtcm_m_vec[i], v_ln2));
            ਜੇ (r0 + 32 > n_ਪੰਕਤੀਆਂ_g) {
                ਅਚਲ HVX_ਸਦਿਸ਼_Pred q_valid = Q6_Q_vsetq_R((n_ਪੰਕਤੀਆਂ_g - r0) * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                v_d = Q6_V_vmux_QVV(q_valid, v_d, Q6_V_vzero());
            }
            factx->vtcm_l_vec[i] = HVX_OP_ਜੋੜੋ_F32(factx->vtcm_l_vec[i], hvx_vec_exp_f32(v_d));
        }
    }

    ਅਚਲ HVX_ਸਦਿਸ਼     v_offsets = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) d_tile_scatter_offsets;
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred q_32_ਨਕਾਬ = Q6_Q_vsetq_R(32 * ਆਕਾਰ(__fp16));
    ਅਚਲ HVX_ਸਦਿਸ਼     one       = hvx_vec_splat_f32(1.0f);

    HVX_ਸਦਿਸ਼ v_content = Q6_V_vzero();
    ਲਈ (size_t i = 0; i < n_ਪੰਕਤੀ_tiles; ++i) {
        ਜੇ ((i % 2) == 0) {
            HVX_ਸਦਿਸ਼ inv_lo = HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i]));
            HVX_ਸਦਿਸ਼ inv_hi = (i + 1 < n_ਪੰਕਤੀ_tiles) ? HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i + 1])) : Q6_V_vzero();
            v_content = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(inv_lo, inv_hi);
        } ਨਹੀਂਤਾਂ {
            v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 * ਬਾਹਰਲਾ_ਅਧਾਰ = factx->vtcm_d_inv_l + i * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_ਨਕਾਬ, (size_t) ਬਾਹਰਲਾ_ਅਧਾਰ, HMX_FP16_TILE_ਮਾਪ - 1, v_offsets, v_content);
    }
}

ਸਥਿਰ ਖਾਲੀ fa_phase_softmax_and_build_d(ਬਣਤਰ hmx_fa_context * factx,
                                         fa_softmax_ਦਲੀਲਾਂ_t *     sargs,
                                         size_t                  n_ਪੰਕਤੀ_tiles,
                                         size_t                  n_ਪੰਕਤੀ_tiles_g_br) {
    work_ਕਤਾਰ_t wp = factx->octx->ctx->work_ਕਤਾਰ;
    ਅਚਲ size_t n_ਪੰਕਤੀ_vec_cnt = hmx_ceil_div(sargs->n_ਪੰਕਤੀਆਂ_g, 64);

    ਕਾਮਾ_callback_t softmax_fn = fa_softmax_thread;
    ਜੇ (sargs->ਨਕਾਬ == ਨਿਰਾ && !factx->has_softcap && !sargs->has_alibi) {
        softmax_fn = fa_softmax_thread_nomask;
    } ਨਹੀਂਤਾਂ ਜੇ (sargs->ਨਕਾਬ != ਨਿਰਾ && factx->ਨਕਾਬ_ਪ੍ਰਸਾਰਣ && !factx->has_softcap && !sargs->has_alibi) {
        ਜੇ (sargs->G == 1) {
            softmax_fn = fa_softmax_thread_ਨਕਾਬ_ਪ੍ਰਸਾਰਣ_g1;
        } ਨਹੀਂਤਾਂ {
            softmax_fn = fa_softmax_thread_ਨਕਾਬ_ਪ੍ਰਸਾਰਣ_gn;
        }
    }

    ਜੇ (factx->n_ਧਾਗੇ > 1 && n_ਪੰਕਤੀ_vec_cnt >= 2) {
        uint32_t n_use = (uint32_t) hex_smin((size_t) factx->n_ਧਾਗੇ, n_ਪੰਕਤੀ_vec_cnt);
        sargs->thread_div = init_fastdiv_ਮੁੱਲਾਂ(n_use);
        work_ਕਤਾਰ_ਲੜੀਵਾਰ(wp, softmax_fn, sargs, n_use);
    } ਨਹੀਂਤਾਂ {
        softmax_fn(1, 0, sargs);
    }
}

// ============================================================================
// HMX job structs and worker functions
// ============================================================================

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਅਚਲ __fp16 * q_tiles;
    ਅਚਲ __fp16 * k_tiles;
    __fp16 *       s_tiles;
    size_t         n_ਪੰਕਤੀ_tiles;
    size_t         n_ਖਾਨਾ_tiles;
    size_t         n_dot_tiles;  // DK / 32
    size_t         n_tiles_ਪ੍ਰਤੀ_bc;
    uint8_t *      hmx_scales;
} hmx_fa_qk_ਕੰਮ_t;

ਸਥਿਰ ਖਾਲੀ hmx_fa_qk_dot_ਕਾਮਾ(ਖਾਲੀ * ਅੰਕੜੇ) {
    hmx_fa_qk_ਕੰਮ_t * ਕੰਮ            = (hmx_fa_qk_ਕੰਮ_t *) ਅੰਕੜੇ;
    ਅਚਲ size_t      n_ਪੰਕਤੀ_tiles    = ਕੰਮ->n_ਪੰਕਤੀ_tiles;
    ਅਚਲ size_t      n_ਖਾਨਾ_tiles    = ਕੰਮ->n_ਖਾਨਾ_tiles;
    ਅਚਲ size_t      n_dot_tiles    = ਕੰਮ->n_dot_tiles;
    ਅਚਲ size_t      n_tiles_ਪ੍ਰਤੀ_bc = ਕੰਮ->n_tiles_ਪ੍ਰਤੀ_bc;
    ਅਚਲ __fp16 * ਸੀਮਿਤ q_tiles  = ਕੰਮ->q_tiles;
    ਅਚਲ __fp16 * ਸੀਮਿਤ k_tiles  = ਕੰਮ->k_tiles;
    __fp16 * ਸੀਮਿਤ s_tiles        = ਕੰਮ->s_tiles;
    __builtin_assume(n_ਪੰਕਤੀ_tiles > 0);
    __builtin_assume(n_ਖਾਨਾ_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    asm ਅਸਥਿਰ(HMX_ਸੈੱਟ_BIAS("%0") :: "r"((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)ਕੰਮ->hmx_scales));
    ਅਚਲ size_t dot_stride = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    ਲਈ (size_t r = 0; r < n_ਪੰਕਤੀ_tiles; ++r) {
        ਅਚਲ __fp16 * ਪੰਕਤੀ_tiles = q_tiles + r * dot_stride;
        ਅਚਲ __fp16 * ਖਾਨਾ_tiles = k_tiles;
        __fp16 *       ਬਾਹਰਲਾ_tile  = s_tiles + r * n_tiles_ਪ੍ਰਤੀ_bc * HMX_FP16_TILE_N_ELMS;

        ਲਈ (size_t c = 0; c < n_ਖਾਨਾ_tiles; ++c) {
            hmx_fa_qk_dot_tile(ਪੰਕਤੀ_tiles, ਖਾਨਾ_tiles, ਬਾਹਰਲਾ_tile, n_dot_tiles);
            ਖਾਨਾ_tiles += dot_stride;
            ਬਾਹਰਲਾ_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    __fp16 *       o_curr;
    ਅਚਲ __fp16 * o_prev;
    ਅਚਲ __fp16 * p_tiles;
    ਅਚਲ __fp16 * v_tiles;
    ਅਚਲ __fp16 * d_tiles;
    uint8_t *      hmx_scales;
    size_t         n_ਪੰਕਤੀ_tiles;
    size_t         n_ਖਾਨਾ_tiles;
    size_t         n_ਪੰਕਤੀ_tiles_g_br;
    size_t         n_tiles_ਪ੍ਰਤੀ_bc;
    size_t         DV;
} hmx_fa_o_update_ਕੰਮ_t;

ਸਥਿਰ ਖਾਲੀ hmx_fa_o_update_ਕਾਮਾ(ਖਾਲੀ * ਅੰਕੜੇ) {
    hmx_fa_o_update_ਕੰਮ_t * ਕੰਮ              = (hmx_fa_o_update_ਕੰਮ_t *) ਅੰਕੜੇ;
    ਅਚਲ size_t            n_ਪੰਕਤੀ_tiles      = ਕੰਮ->n_ਪੰਕਤੀ_tiles;
    ਅਚਲ size_t            n_ਖਾਨਾ_tiles      = ਕੰਮ->n_ਖਾਨਾ_tiles;
    ਅਚਲ size_t            n_ਪੰਕਤੀ_tiles_g_br = ਕੰਮ->n_ਪੰਕਤੀ_tiles_g_br;
    ਅਚਲ size_t            n_tiles_ਪ੍ਰਤੀ_bc   = ਕੰਮ->n_tiles_ਪ੍ਰਤੀ_bc;
    ਅਚਲ size_t            DV_tiles         = ਕੰਮ->DV / 32;
    ਅਚਲ __fp16 * ਸੀਮਿਤ d_tiles          = ਕੰਮ->d_tiles;
    ਅਚਲ __fp16 * ਸੀਮਿਤ p_tiles          = ਕੰਮ->p_tiles;
    ਅਚਲ __fp16 * ਸੀਮਿਤ v_tiles          = ਕੰਮ->v_tiles;
    ਅਚਲ __fp16 * ਸੀਮਿਤ o_prev           = ਕੰਮ->o_prev;
    __fp16 * ਸੀਮਿਤ o_curr                 = ਕੰਮ->o_curr;
    __builtin_assume(n_ਪੰਕਤੀ_tiles > 0);
    __builtin_assume(n_ਖਾਨਾ_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm ਅਸਥਿਰ(HMX_ਸੈੱਟ_BIAS("%0") :: "r"((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)ਕੰਮ->hmx_scales));
    ਅਚਲ size_t o_stride = n_ਪੰਕਤੀ_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    ਅਚਲ size_t v_stride = n_tiles_ਪ੍ਰਤੀ_bc * HMX_FP16_TILE_N_ELMS;
    ਲਈ (size_t r = 0; r < n_ਪੰਕਤੀ_tiles; ++r) {
        ਅਚਲ __fp16 * d_diag     = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        ਅਚਲ __fp16 * p_tile_ਵਿੱਚ  = p_tiles + (r * n_tiles_ਪ੍ਰਤੀ_bc) * HMX_FP16_TILE_N_ELMS;
        ਅਚਲ __fp16 * o_rc       = o_prev + r * HMX_FP16_TILE_N_ELMS;
        ਅਚਲ __fp16 * v_tile_ਵਿੱਚ  = v_tiles;
        __fp16       * o_tile_ਬਾਹਰਲਾ = o_curr + r * HMX_FP16_TILE_N_ELMS;

        ਲਈ (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_update_tile(d_diag, o_rc, p_tile_ਵਿੱਚ, v_tile_ਵਿੱਚ, o_tile_ਬਾਹਰਲਾ, n_ਖਾਨਾ_tiles);
            o_rc       += o_stride;
            v_tile_ਵਿੱਚ  += v_stride;
            o_tile_ਬਾਹਰਲਾ += o_stride;
        }
    }
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    __fp16 *       o_curr;   // output (row-major tile layout)
    ਅਚਲ __fp16 * o_prev;   // input (column-major tile layout)
    ਅਚਲ __fp16 * d_tiles;  // diag(1/l) tiles
    uint8_t *      hmx_scales;
    size_t         n_ਪੰਕਤੀ_tiles;
    size_t         n_ਪੰਕਤੀ_tiles_g_br;
    size_t         DV;
} hmx_fa_o_norm_ਕੰਮ_t;

ਸਥਿਰ ਖਾਲੀ hmx_fa_o_norm_ਕਾਮਾ(ਖਾਲੀ * ਅੰਕੜੇ) {
    hmx_fa_o_norm_ਕੰਮ_t * ਕੰਮ              = (hmx_fa_o_norm_ਕੰਮ_t *) ਅੰਕੜੇ;
    ਅਚਲ size_t          n_ਪੰਕਤੀ_tiles      = ਕੰਮ->n_ਪੰਕਤੀ_tiles;
    ਅਚਲ size_t          n_ਪੰਕਤੀ_tiles_g_br = ਕੰਮ->n_ਪੰਕਤੀ_tiles_g_br;
    ਅਚਲ size_t          DV_tiles         = ਕੰਮ->DV / 32;
    ਅਚਲ __fp16 * ਸੀਮਿਤ d_tiles        = ਕੰਮ->d_tiles;
    ਅਚਲ __fp16 * ਸੀਮਿਤ o_prev         = ਕੰਮ->o_prev;
    __fp16 * ਸੀਮਿਤ o_curr               = ਕੰਮ->o_curr;
    __builtin_assume(n_ਪੰਕਤੀ_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm ਅਸਥਿਰ(HMX_ਸੈੱਟ_BIAS("%0") :: "r"((ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ)ਕੰਮ->hmx_scales));
    ਅਚਲ size_t o_stride = n_ਪੰਕਤੀ_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    ਲਈ (size_t r = 0; r < n_ਪੰਕਤੀ_tiles; ++r) {
        ਅਚਲ __fp16 * d_diag = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        ਅਚਲ __fp16 * o_rc = o_prev + r * HMX_FP16_TILE_N_ELMS;
        __fp16 *       o_ਬਾਹਰਲਾ = o_curr + r * DV_tiles * HMX_FP16_TILE_N_ELMS;

        ਲਈ (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_norm_tile(d_diag, o_rc, o_ਬਾਹਰਲਾ);
            o_rc  += o_stride;
            o_ਬਾਹਰਲਾ += HMX_FP16_TILE_N_ELMS;
        }
    }
}

// Populate per-GQA-row ALiBi slopes for a given KV head.
ਸਥਿਰ __attribute__((noinline)) ਖਾਲੀ fa_compute_slopes(
                              ਅਚਲ ਬਣਤਰ hmx_fa_context * factx,
                              uint32_t                      kv_ਸਿਰ,
                              size_t                        n_ਪੰਕਤੀਆਂ_g) {
    __fp16 * slopes = factx->vtcm_slopes;
    ਜੇ (factx->max_bias == 0.0f) {
        hvx_splat_f16_a(slopes, 1.0f, n_ਪੰਕਤੀਆਂ_g);
        ਵਾਪਸ;
    }

    ਅਚਲ uint32_t G           = factx->G;
    ਅਚਲ uint32_t n_ਸਿਰ_log2 = factx->n_ਸਿਰ_log2;
    ਅਚਲ ਦਸ਼ਮਲਵ    m0          = factx->m0;
    ਅਚਲ ਦਸ਼ਮਲਵ    m1          = factx->m1;

    __fp16 ਆਰਜ਼ੀ_slopes[512] __attribute__((aligned(128)));
    ਜੇ (G <= 32) {
        // Fast path: Compute G unique slope values in vector registers
        HVX_ਸਦਿਸ਼ v_val = hvx_alibi_slopes(kv_ਸਿਰ, G, n_ਸਿਰ_log2, m0, m1);

        __fp16 ਆਰਜ਼ੀ_slopes_aligned[64] __attribute__((aligned(128)));
        hvx_vmem(ਆਰਜ਼ੀ_slopes_aligned) = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16(v_val, Q6_V_vzero());

        ਲਈ (uint32_t i = 0; i < G; ++i) {
            ਆਰਜ਼ੀ_slopes[i] = ਆਰਜ਼ੀ_slopes_aligned[i];
        }
    } ਨਹੀਂਤਾਂ {
        // Fallback path: G > 32 (rare configurations)
        ਲਈ (uint32_t i = 0; i < G; ++i) {
            ਆਰਜ਼ੀ_slopes[i] = (__fp16)alibi_ਢਲਾਨ(kv_ਸਿਰ * G + i, n_ਸਿਰ_log2, m0, m1);
        }
    }

    // Allocate stack buffer to avoid scalar writes to VTCM (which generates L2 misses)
    __fp16 local_slopes[n_ਪੰਕਤੀਆਂ_g] __attribute__((aligned(128)));
    ਲਈ (size_t r = 0; r < n_ਪੰਕਤੀਆਂ_g; ++r) {
        local_slopes[r] = ਆਰਜ਼ੀ_slopes[fastmodulo(r, G, &factx->div_G)];
    }

    // Copy to VTCM slopes using HVX block copy (both are aligned to 128 bytes)
    hvx_copy_f16_aa((uint8_t *)slopes, (ਅਚਲ uint8_t *)local_slopes, n_ਪੰਕਤੀਆਂ_g);
}

ਸਥਿਰ ਖਾਲੀ fa_ਧੱਕੋ_ਨਕਾਬ_dma_gqa(
    dma_ਕਤਾਰ *               dma_q,
    ਅਚਲ ਬਣਤਰ htp_tensor * ਨਕਾਬ,
    uint32_t                  q_ਸ਼ੁਰੂ,
    uint32_t                  im3,
    uint32_t                  kv_ਸ਼ੁਰੂ,
    uint32_t                  kv_ਸਿਰ,
    uint32_t                  G,
    uint32_t                  m_ਸਤਰ_ਬਾਈਟਾਂ,
    uint32_t                  kv_ਪੰਕਤੀਆਂ,
    uint32_t                  n_ਪੰਕਤੀਆਂ_q,
    ਬਣਤਰ hmx_fa_context *   factx
) {
    ਲਈ (uint32_t g = 0; g < G; ++g) {
        ਅਚਲ uint32_t h_idx = kv_ਸਿਰ * G + g;
        ਅਚਲ uint32_t im2 = fastmodulo(h_idx, ਨਕਾਬ->ne[2], &factx->src3_div2);
        ਅਚਲ dma_addr_t ms_src = ਨਕਾਬ->ਅੰਕੜੇ + q_ਸ਼ੁਰੂ * ਨਕਾਬ->nb[1] +
                                  im2 * ਨਕਾਬ->nb[2] + im3 * ਨਕਾਬ->nb[3] + kv_ਸ਼ੁਰੂ * ਆਕਾਰ(__fp16);
        uint8_t * ms_dst = (uint8_t *) factx->vtcm_ਨਕਾਬ_buf + g * m_ਸਤਰ_ਬਾਈਟਾਂ;
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(ms_dst, ms_src), G * m_ਸਤਰ_ਬਾਈਟਾਂ, ਨਕਾਬ->nb[1], kv_ਪੰਕਤੀਆਂ * ਆਕਾਰ(__fp16), n_ਪੰਕਤੀਆਂ_q);
    }
}

ਸਥਿਰ ਖਾਲੀ fa_ਕੱਢੋ_ਨਕਾਬ_dma_gqa(dma_ਕਤਾਰ * dma_q, uint32_t G) {
    ਲਈ (uint32_t g = 0; g < G; ++g) {
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ fa_prefetch_ਖੰਡ(dma_ਕਤਾਰ * dma_q, ਅਚਲ ਬਣਤਰ htp_tensor * k, ਅਚਲ ਬਣਤਰ htp_tensor * v, ਅਚਲ ਬਣਤਰ htp_tensor * ਨਕਾਬ,
                                     uint32_t b, size_t Bc, size_t ਮਾਪ_k_ਪੰਕਤੀ_padded, size_t ਮਾਪ_k_ਪੰਕਤੀ, size_t ਮਾਪ_v_ਪੰਕਤੀ_padded, size_t ਮਾਪ_v_ਪੰਕਤੀ,
                                     uint32_t ik2, uint32_t ik3, uint32_t iv2, uint32_t iv3, uint32_t q_ਸ਼ੁਰੂ, uint32_t im3, uint32_t kv_ਸਿਰ, uint32_t G,
                                     size_t m_ਸਤਰ_ਬਾਈਟਾਂ, size_t n_ਪੰਕਤੀਆਂ_q, size_t nek1, size_t prefetch_buf, ਬਣਤਰ hmx_fa_context * factx) {
    ਅਚਲ uint32_t  prefetch_ਸ਼ੁਰੂ = b * Bc;
    ਅਚਲ uint32_t  prefetch_ਪੰਕਤੀਆਂ  = hex_smin(Bc, nek1 - prefetch_ਸ਼ੁਰੂ);
    ਅਚਲ dma_addr_t k_prefetch_src = k->ਅੰਕੜੇ + prefetch_ਸ਼ੁਰੂ * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(factx->vtcm_k_fp16[prefetch_buf], k_prefetch_src), ਮਾਪ_k_ਪੰਕਤੀ_padded, k->nb[1], ਮਾਪ_k_ਪੰਕਤੀ, prefetch_ਪੰਕਤੀਆਂ);
    ਅਚਲ dma_addr_t v_prefetch_src = v->ਅੰਕੜੇ + prefetch_ਸ਼ੁਰੂ * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(factx->vtcm_v_fp16[prefetch_buf], v_prefetch_src), ਮਾਪ_v_ਪੰਕਤੀ_padded, v->nb[1], ਮਾਪ_v_ਪੰਕਤੀ, prefetch_ਪੰਕਤੀਆਂ);

    ਜੇ (ਨਕਾਬ) {
        ਜੇ (__builtin_expect(factx->ਨਕਾਬ_ਪ੍ਰਸਾਰਣ, true)) {
            ਅਚਲ dma_addr_t ms_src = ਨਕਾਬ->ਅੰਕੜੇ + q_ਸ਼ੁਰੂ * ਨਕਾਬ->nb[1] + im3 * ਨਕਾਬ->nb[3] + prefetch_ਸ਼ੁਰੂ * ਆਕਾਰ(__fp16);
            dma_cache_fa_ਧੱਕੋ(dma_q, &factx->m_cache, ms_src, m_ਸਤਰ_ਬਾਈਟਾਂ, ਨਕਾਬ->nb[1], prefetch_ਪੰਕਤੀਆਂ * ਆਕਾਰ(__fp16), n_ਪੰਕਤੀਆਂ_q);
        } ਨਹੀਂਤਾਂ {
            fa_ਧੱਕੋ_ਨਕਾਬ_dma_gqa(dma_q, ਨਕਾਬ, q_ਸ਼ੁਰੂ, im3, prefetch_ਸ਼ੁਰੂ, kv_ਸਿਰ, G, m_ਸਤਰ_ਬਾਈਟਾਂ, prefetch_ਪੰਕਤੀਆਂ, n_ਪੰਕਤੀਆਂ_q, factx);
        }
    }
}

// ============================================================================
// Core HMX flash attention algorithm (GQA-merged)
// ============================================================================

ਪੂਰਨ hmx_flash_attn_ext(ਬਣਤਰ htp_ops_context * octx) {
    ਬਣਤਰ htp_thread_trace * tr_hvx = &octx->ctx->trace[0];
    ਬਣਤਰ htp_thread_trace * tr_hmx = &octx->ctx->trace[HTP_MAX_NTHREADS];
    ਅਚਲ ਬਣਤਰ htp_tensor * q    = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * k    = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * v    = octx->src[2];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਨਕਾਬ = (octx->src[3] && octx->src[3]->ਅੰਕੜੇ) ? octx->src[3] : ਨਿਰਾ;
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਬਣਤਰ htp_context * ਅਚਲ ctx = octx->ctx;

    ਜੇ (!ctx->hmx_enabled) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    // Dimensions
    ਅਚਲ uint32_t neq0 = q->ne[0];  // head_dim (DK)
    ਅਚਲ uint32_t neq1 = q->ne[1];  // n_tokens
    ਅਚਲ uint32_t neq2 = q->ne[2];  // n_heads
    ਅਚਲ uint32_t neq3 = q->ne[3];  // n_seqs

    ਅਚਲ uint32_t nek0 = k->ne[0];  // head_dim
    ਅਚਲ uint32_t nek1 = k->ne[1];  // kv_len

    ਅਚਲ uint32_t nev0 = v->ne[0];  // head_dim (DV)

    ਅਚਲ uint32_t DK = neq0;
    ਅਚਲ uint32_t DV = nev0;

    // HMX tiles head_dim in units of 64. head_dim need not be 64- (or 32-) aligned:
    // we can operate on DK/DV rounded up to 64 with tail lanes [D, D_pad) zero-filled.
    ਅਚਲ uint32_t DK_pad = hex_ਗੇੜਾ_ਉੱਪਰ(DK, 64);
    ਅਚਲ uint32_t DV_pad = hex_ਗੇੜਾ_ਉੱਪਰ(DV, 64);
    ਜੇ (DK == 0 || DV == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ ਬਣਤਰ htp_fa_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_fa_kernel_params *) octx->kernel_params;
    ਅਚਲ uint32_t n_kv_heads = k->ne[2];

    // ======== Build context ========
    ਬਣਤਰ hmx_fa_context factx;
    memset(&factx, 0, ਆਕਾਰ(factx));
    factx.octx           = octx;
    factx.sinks          = octx->src[4];  // NULL if this op has no attention sinks
    factx.k              = k;
    factx.v              = v;
    factx.n_ਧਾਗੇ      = kparams->n_ਧਾਗੇ;
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
    factx.n_kv_ਖੰਡਾਂ    = kparams->n_kv_ਖੰਡਾਂ;
    factx.is_q_fp32      = (q->type == HTP_TYPE_F32);
    factx.is_dst_fp32    = (dst->type == HTP_TYPE_F32);
    factx.pipeline       = (kparams->u.hmx.pipeline != 0);
    factx.ਨਕਾਬ_ਪ੍ਰਸਾਰਣ = (kparams->u.hmx.ਨਕਾਬ_ਪ੍ਰਸਾਰਣ != 0);
    ਜੇ (ਨਕਾਬ) {
        factx.src3_div2  = kparams->src3_div2;
        factx.src3_div3  = kparams->src3_div3;
    }

    factx.has_softcap   = (kparams->logit_softcap != 0.0f);
    factx.scale         = (__fp16) kparams->scale;
    factx.max_bias      = kparams->max_bias;
    factx.logit_softcap = 0;
    ਜੇ (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_ਸਿਰ_log2 = kparams->n_ਸਿਰ_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    ਅਚਲ uint32_t Br = factx.Br;
    ਅਚਲ uint32_t Bc = factx.Bc;
    ਅਚਲ uint32_t g_br = factx.g_br;
    ਅਚਲ bool pipeline = factx.pipeline;
    ਅਚਲ uint32_t n_ਧਾਗੇ = factx.n_ਧਾਗੇ;
    ਅਚਲ uint32_t G = factx.G;

    // Multi-device: prefer head-parallel partitioning (each core owns a disjoint head
    // shard), falling back to Q-block (token) split when heads don't divide evenly.
    ਅਚਲ uint32_t n_q_ਖੰਡਾਂ = (neq1 + Br - 1) / Br;
    uint32_t q_ਸ਼ੁਰੂ_min  = 0;
    uint32_t q_ਸ਼ੁਰੂ_max  = neq1;
    uint32_t kv_ਸਿਰ_min  = 0;
    uint32_t kv_ਸਿਰ_max  = n_kv_heads;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ uint32_t mdev_ਗਿਣਤੀ = octx->ctx->mdev.ਗਿਣਤੀ_2;
        ਅਚਲ uint32_t mdev_idx   = octx->ctx->mdev.idx;
        ਅਚਲ uint32_t dst_e_ਮਾਪ = (dst->type == HTP_TYPE_F32) ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) : ਆਕਾਰ(__fp16);
        ਅਚਲ bool can_ਵੰਡੋ      = htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(dst, dst_e_ਮਾਪ);

        ਜੇ (kparams->ਸਿਰ_ਵੰਡੋ && can_ਵੰਡੋ && n_kv_heads >= mdev_ਗਿਣਤੀ && n_kv_heads % mdev_ਗਿਣਤੀ == 0) {
            ਅਚਲ uint32_t kv_ਪ੍ਰਤੀ_core = n_kv_heads / mdev_ਗਿਣਤੀ;
            kv_ਸਿਰ_min = mdev_idx * kv_ਪ੍ਰਤੀ_core;
            kv_ਸਿਰ_max = kv_ਸਿਰ_min + kv_ਪ੍ਰਤੀ_core;
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(n_q_ਖੰਡਾਂ, can_ਵੰਡੋ ? 1 : 0, mdev_idx, mdev_ਗਿਣਤੀ, &octx->ctx->mdev.ਗਿਣਤੀ_div);
            ਅਚਲ uint32_t ਖੰਡ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
            ਅਚਲ uint32_t ਖੰਡ_ਸਿਰਾ   = range.ਸ਼ੁਰੂ + range.ਗਿਣਤੀ_2;

            ਜੇ (ਖੰਡ_ਸ਼ੁਰੂ >= ਖੰਡ_ਸਿਰਾ) {
                ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
            }

            q_ਸ਼ੁਰੂ_min = ਖੰਡ_ਸ਼ੁਰੂ * Br;
            q_ਸ਼ੁਰੂ_max = MIN(ਖੰਡ_ਸਿਰਾ * Br, neq1);
        }
    }

    // ======== VTCM allocation (GQA-aware) ========
    // K/V row sizes drive the DMA descriptors (not the VTCM layout) and are used
    // throughout the KV loop below. The DMA copies only the real DK/DV columns; the
    // staging rows are padded to hold DK_pad/DV_pad columns (tail zero-filled below)
    // so the HMX interleave/tile logic can operate on 64-aligned head dims.
    ਅਚਲ size_t ਮਾਪ_k_ਪੰਕਤੀ        = htp_tensor_get_ਪੰਕਤੀ_ਮਾਪ(k->type, DK);
    ਅਚਲ size_t ਮਾਪ_v_ਪੰਕਤੀ        = htp_tensor_get_ਪੰਕਤੀ_ਮਾਪ(v->type, DV);
    ਅਚਲ size_t ਮਾਪ_k_ਪੰਕਤੀ_padded = hex_ਗੇੜਾ_ਉੱਪਰ(DK_pad * ਆਕਾਰ(__fp16), 128);
    ਅਚਲ size_t ਮਾਪ_v_ਪੰਕਤੀ_padded = hex_ਗੇੜਾ_ਉੱਪਰ(DV_pad * ਆਕਾਰ(__fp16), 128);

    // Build the VTCM layout once (shared with the host estimator) and place every
    // scratch buffer at its computed offset. Padded head dims size the HMX tiles.
    ਬਣਤਰ hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, G, DK_pad, DV_pad, Br, Bc, n_ਧਾਗੇ, pipeline, factx.is_q_fp32, factx.sinks != ਨਿਰਾ, factx.n_heads);

    ਜੇ (L.ਕੁੱਲ_ਬਾਈਟਾਂ > ctx->vtcm_ਮਾਪ) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    uint8_t * ਅਚਲ ਅਧਾਰ = ctx->vtcm_ਅਧਾਰ;

    factx.vtcm_q_dma          = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_q_dma);
    factx.vtcm_q_tiles        = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_q_tiles);
    factx.vtcm_o_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_o_tiles[0]);
    factx.vtcm_o_tiles[1]     = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_o_tiles[1]);
    factx.vtcm_k_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_k_fp16[0]);
    factx.vtcm_k_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_k_fp16[1]);
    factx.vtcm_v_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_v_fp16[0]);
    factx.vtcm_v_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_v_fp16[1]);
    factx.vtcm_k_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_k_tiles[0]);
    factx.vtcm_k_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ਅਧਾਰ, L.off_k_tiles[1], pipeline);
    factx.vtcm_v_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_v_tiles[0]);
    factx.vtcm_v_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ਅਧਾਰ, L.off_v_tiles[1], pipeline);
    factx.vtcm_s_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_s_tiles[0]);
    factx.vtcm_s_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ਅਧਾਰ, L.off_s_tiles[1], pipeline);
    factx.vtcm_p_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_p_tiles[0]);
    factx.vtcm_p_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ਅਧਾਰ, L.off_p_tiles[1], pipeline);
    factx.vtcm_d_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_d_tiles[0]);
    factx.vtcm_d_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, ਅਧਾਰ, L.off_d_tiles[1], pipeline);
    factx.vtcm_d_inv_l        = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_d_inv_l);
    factx.vtcm_m_vec          = VTCM_LAYOUT_PTR(HVX_ਸਦਿਸ਼, ਅਧਾਰ, L.off_m_vec);
    factx.vtcm_l_vec          = VTCM_LAYOUT_PTR(HVX_ਸਦਿਸ਼, ਅਧਾਰ, L.off_l_vec);
    factx.vtcm_s_rowmax       = VTCM_LAYOUT_PTR(HVX_ਸਦਿਸ਼, ਅਧਾਰ, L.off_s_rowmax);
    factx.vtcm_p_rowsum       = VTCM_LAYOUT_PTR(HVX_ਸਦਿਸ਼, ਅਧਾਰ, L.off_p_rowsum);
    factx.vtcm_ਪੰਕਤੀ_bufs       = VTCM_LAYOUT_PTR(HVX_ਸਦਿਸ਼, ਅਧਾਰ, L.off_ਪੰਕਤੀ_bufs);
    factx.ਪੰਕਤੀ_buf_stride      = L.ਪੰਕਤੀ_buf_stride;
    factx.vtcm_hmx_scales_id  = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_hmx_scales_id);
    factx.vtcm_hmx_scales_qk  = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_hmx_scales_qk);
    factx.vtcm_ਨਕਾਬ_buf       = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_ਨਕਾਬ_buf);
    factx.ਨਕਾਬ_buf_ਪੰਕਤੀ_stride = L.ਨਕਾਬ_buf_ਪੰਕਤੀ_stride;
    factx.q_tile_ਬਾਈਟਾਂ        = L.q_tile_ਬਾਈਟਾਂ;
    factx.k_tile_ਬਾਈਟਾਂ        = L.k_tile_ਬਾਈਟਾਂ;
    factx.o_tile_ਬਾਈਟਾਂ        = L.o_tile_ਬਾਈਟਾਂ;
    factx.ਖਾਨਾ_vec_ਬਾਈਟਾਂ       = L.ਖਾਨਾ_vec_ਬਾਈਟਾਂ;
    factx.d_tile_ਬਾਈਟਾਂ        = L.d_tile_ਬਾਈਟਾਂ;
    factx.vtcm_slopes         = VTCM_LAYOUT_PTR(__fp16, ਅਧਾਰ, L.off_slopes);
    factx.vtcm_sinks          = VTCM_LAYOUT_PTR_OPTIONAL(ਦਸ਼ਮਲਵ, ਅਧਾਰ, L.off_sinks, factx.sinks != ਨਿਰਾ);

    ਅਚਲ size_t m_ਸਤਰ_ਬਾਈਟਾਂ = L.m_ਸਤਰ_ਬਾਈਟਾਂ;  // used by the mask DMAs in the KV loop

    dma_cache_fa_init(&factx.m_cache, (uint8_t *) factx.vtcm_ਨਕਾਬ_buf, L.m_buf_ਥਾਂ_ਬਾਈਟਾਂ, HMX_FA_DMA_CACHE_ਮਾਪ);

    // Head-dim padding: the K/V DMA staging buffers and the flat-Q buffer are laid out
    // with padded row strides (size_{k,v,q}_row_padded, covering D_pad columns) but the
    // DMA only writes the real D columns per row. Zero the whole staging buffers once up
    // front so tail lanes [D, D_pad) stay zero for all KV blocks. No-op when already aligned.
    ਜੇ (DK_pad != DK || DV_pad != DV) {
        ਅਚਲ size_t k_buf_ਬਾਈਟਾਂ = (size_t) factx.Bc * ਮਾਪ_k_ਪੰਕਤੀ_padded;
        ਅਚਲ size_t v_buf_ਬਾਈਟਾਂ = (size_t) factx.Bc * ਮਾਪ_v_ਪੰਕਤੀ_padded;
        hvx_splat_u8_a((ਅੱਖਰ *) factx.vtcm_k_fp16[0], 0, k_buf_ਬਾਈਟਾਂ);
        hvx_splat_u8_a((ਅੱਖਰ *) factx.vtcm_k_fp16[1], 0, k_buf_ਬਾਈਟਾਂ);
        hvx_splat_u8_a((ਅੱਖਰ *) factx.vtcm_v_fp16[0], 0, v_buf_ਬਾਈਟਾਂ);
        hvx_splat_u8_a((ਅੱਖਰ *) factx.vtcm_v_fp16[1], 0, v_buf_ਬਾਈਟਾਂ);
        // Flat-Q DMA scratch
        ਜੇ (factx.vtcm_q_dma) {
            ਅਚਲ size_t q_dma_ਬਾਈਟਾਂ = hex_align_ਉੱਪਰ(factx.g_br * DK * (factx.is_q_fp32 ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) : ਆਕਾਰ(__fp16)), 128);
            hvx_splat_u8_a((ਅੱਖਰ *) factx.vtcm_q_dma, 0, q_dma_ਬਾਈਟਾਂ);
        }
    }

    // ======== Initialize HMX output scales ========
    hmx_init_ਕਾਲਮ_scales(factx.vtcm_hmx_scales_id, Q6_V_vsplat_R(0x3c00)); // 1.0
    {
        ਸੰਘ { __fp16 f; uint16_t u; } sc = { .f = factx.scale };
        hmx_init_ਕਾਲਮ_scales(factx.vtcm_hmx_scales_qk, Q6_V_vsplat_R(sc.u));
    }

    // ======== DMA setup ========
    dma_ਕਤਾਰ * ਅਚਲ dma_q = ctx->dma[0];

    ਜੇ (factx.sinks) {
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(factx.vtcm_sinks, factx.sinks->ਅੰਕੜੇ), L.sinks_ਬਾਈਟਾਂ, 0, factx.sinks->ਮਾਪ, 1);
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
    }

    ਅਚਲ size_t n_ਪੰਕਤੀ_tiles_g_br = g_br / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
    ਅਚਲ size_t n_tiles_ਪ੍ਰਤੀ_bc   = Bc / HMX_FP16_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;

    ਅਚਲ size_t qo_element_ਮਾਪ = factx.is_q_fp32 ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) : ਆਕਾਰ(__fp16);

    ਅਚਲ bool q_transposed                 = q->nb[1] < q->nb[2];
    ਅਚਲ size_t q_src_stride               = q_transposed ? q->nb[2] : q->nb[1];
    ਅਚਲ size_t q_ਪੰਕਤੀ_ਬਾਈਟਾਂ_untransposed   = factx.G * factx.DK * qo_element_ਮਾਪ;
    ਅਚਲ size_t q_ਪੰਕਤੀ_ਬਾਈਟਾਂ_trans_ਗੁਣਕ   = factx.DK * qo_element_ਮਾਪ;

    // ======== Reusable job descriptors for pipeline ========
    hmx_fa_qk_ਕੰਮ_t       qk_ਕੰਮ;
    hmx_fa_o_update_ਕੰਮ_t ou_ਕੰਮ;
    hmx_fa_o_norm_ਕੰਮ_t   on_ਕੰਮ;

    // ======== Main loop ========
    ਲਈ (uint32_t ib3 = 0; ib3 < neq3; ++ib3) {
        ਅਚਲ uint32_t im3 = ਨਕਾਬ ? fastmodulo(ib3, ਨਕਾਬ->ne[3], &factx.src3_div3) : 0;
        ਲਈ (uint32_t q_ਸ਼ੁਰੂ = q_ਸ਼ੁਰੂ_min; q_ਸ਼ੁਰੂ < q_ਸ਼ੁਰੂ_max; q_ਸ਼ੁਰੂ += Br) {
            ਅਚਲ uint32_t n_ਪੰਕਤੀਆਂ_q    = hex_smin(Br, neq1 - q_ਸ਼ੁਰੂ);
            ਅਚਲ size_t   n_ਪੰਕਤੀਆਂ_g    = n_ਪੰਕਤੀਆਂ_q * G;
            ਅਚਲ size_t   g_br_actual = hex_align_ਉੱਪਰ(n_ਪੰਕਤੀਆਂ_g, HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ);
            ਅਚਲ size_t   n_ਪੰਕਤੀ_tiles = g_br_actual / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;

            uint16_t ਸਰਗਰਮ_ਖੰਡਾਂ[FA_MAX_KV_ਖੰਡਾਂ];
            uint32_t n_ਸਰਗਰਮ_ਖੰਡਾਂ = factx.n_kv_ਖੰਡਾਂ;
            ਜੇ (ਨਕਾਬ && factx.ਨਕਾਬ_ਪ੍ਰਸਾਰਣ && factx.n_kv_ਖੰਡਾਂ > 1) {
                ਅਚਲ size_t scratch_ਬਾਈਟਾਂ = factx.k_tile_ਬਾਈਟਾਂ * (factx.pipeline ? 2 : 1);
                n_ਸਰਗਰਮ_ਖੰਡਾਂ = fa_ਜਾਂਚ_ਸਰਗਰਮ_ਖੰਡਾਂ(
                    dma_q, ਨਕਾਬ, q_ਸ਼ੁਰੂ, n_ਪੰਕਤੀਆਂ_q, 0, im3, nek1, Bc,
                    factx.n_kv_ਖੰਡਾਂ, factx.vtcm_k_tiles[0], scratch_ਬਾਈਟਾਂ, ਸਰਗਰਮ_ਖੰਡਾਂ);
            } ਨਹੀਂਤਾਂ {
                #ਹਦਾਇਤ clang loop vectorize(disable)
                ਲਈ (uint32_t b = 0; b < factx.n_kv_ਖੰਡਾਂ; ++b) {
                    ਸਰਗਰਮ_ਖੰਡਾਂ[b] = (uint16_t) b;
                }
            }
            ਅਚਲ uint32_t b0       = (n_ਸਰਗਰਮ_ਖੰਡਾਂ > 0) ? ਸਰਗਰਮ_ਖੰਡਾਂ[0] : 0;
            ਅਚਲ uint32_t b0_ਸ਼ੁਰੂ = b0 * Bc;
            ਅਚਲ uint32_t b0_ਪੰਕਤੀਆਂ  = hex_smin(Bc, nek1 - b0_ਸ਼ੁਰੂ);

            ਲਈ (uint32_t kv_ਸਿਰ = kv_ਸਿਰ_min; kv_ਸਿਰ < kv_ਸਿਰ_max; ++kv_ਸਿਰ) {
                ਅਚਲ uint32_t ik2 = kv_ਸਿਰ;
                ਅਚਲ uint32_t ik3 = fastdiv(ib3, &kparams->ਪ੍ਰਸਾਰਣ_rk3);
                ਅਚਲ uint32_t iv2 = kv_ਸਿਰ;
                ਅਚਲ uint32_t iv3 = fastdiv(ib3, &kparams->ਪ੍ਰਸਾਰਣ_rv3);

                // 1. Push Q and KV DMAs for the first kv_head of this q_start block.
                // Subsequent kv_head iterations are enqueued early at the end of the previous iteration.
                ਜੇ (kv_ਸਿਰ == kv_ਸਿਰ_min) {
                    ਅਚਲ dma_addr_t q_ptr = q->ਅੰਕੜੇ + q_ਸ਼ੁਰੂ * q->nb[1] +
                                            (kv_ਸਿਰ * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    ਅਚਲ size_t q_ਪੰਕਤੀ_ਬਾਈਟਾਂ = q_transposed ? n_ਪੰਕਤੀਆਂ_q * q_ਪੰਕਤੀ_ਬਾਈਟਾਂ_trans_ਗੁਣਕ : q_ਪੰਕਤੀ_ਬਾਈਟਾਂ_untransposed;
                    ਅਚਲ size_t n_ਪੰਕਤੀਆਂ      = q_transposed ? factx.G : n_ਪੰਕਤੀਆਂ_q;
                    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(factx.vtcm_q_dma, q_ptr), q_ਪੰਕਤੀ_ਬਾਈਟਾਂ, hex_smax(q_src_stride, q_ਪੰਕਤੀ_ਬਾਈਟਾਂ), q_ਪੰਕਤੀ_ਬਾਈਟਾਂ, n_ਪੰਕਤੀਆਂ);

                    ਜੇ (n_ਸਰਗਰਮ_ਖੰਡਾਂ > 0) {
                        ਅਚਲ dma_addr_t k_src = k->ਅੰਕੜੇ + b0_ਸ਼ੁਰੂ * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(factx.vtcm_k_fp16[0], k_src), ਮਾਪ_k_ਪੰਕਤੀ_padded, k->nb[1], ਮਾਪ_k_ਪੰਕਤੀ, b0_ਪੰਕਤੀਆਂ);

                        ਅਚਲ dma_addr_t v_src = v->ਅੰਕੜੇ + b0_ਸ਼ੁਰੂ * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(factx.vtcm_v_fp16[0], v_src), ਮਾਪ_v_ਪੰਕਤੀ_padded, v->nb[1], ਮਾਪ_v_ਪੰਕਤੀ, b0_ਪੰਕਤੀਆਂ);

                        ਜੇ (factx.pipeline && ਨਕਾਬ) {
                            ਜੇ (__builtin_expect(factx.ਨਕਾਬ_ਪ੍ਰਸਾਰਣ, true)) {
                                ਅਚਲ dma_addr_t ms_src = ਨਕਾਬ->ਅੰਕੜੇ + q_ਸ਼ੁਰੂ * ਨਕਾਬ->nb[1] + im3 * ਨਕਾਬ->nb[3] + b0_ਸ਼ੁਰੂ * ਆਕਾਰ(__fp16);
                                dma_cache_fa_ਧੱਕੋ(dma_q, &factx.m_cache, ms_src, m_ਸਤਰ_ਬਾਈਟਾਂ, ਨਕਾਬ->nb[1], b0_ਪੰਕਤੀਆਂ * ਆਕਾਰ(__fp16), n_ਪੰਕਤੀਆਂ_q);
                            } ਨਹੀਂਤਾਂ {
                                fa_ਧੱਕੋ_ਨਕਾਬ_dma_gqa(dma_q, ਨਕਾਬ, q_ਸ਼ੁਰੂ, im3, b0_ਸ਼ੁਰੂ, kv_ਸਿਰ, G, m_ਸਤਰ_ਬਾਈਟਾਂ, b0_ਪੰਕਤੀਆਂ, n_ਪੰਕਤੀਆਂ_q, &factx);
                            }
                        }
                    }
                }

                // 2. Pop Q DMA (blocks until Q is loaded)
                dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);

                // ---- Load Q block & Initialize per-block state ----
                fa_phase_q_ਲੱਦੋ(&factx, q, q_ਸ਼ੁਰੂ, kv_ਸਿਰ, ib3, n_ਪੰਕਤੀਆਂ_g);

                __fp16 * o_tile_prev = factx.vtcm_o_tiles[0];
                __fp16 * o_tile_curr = factx.vtcm_o_tiles[1];

                // ---- KV block loop with DMA double-buffering ----
                size_t buf_idx = 0;

                htp_trace_event_ਸ਼ੁਰੂ(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_ਸ਼ੁਰੂ);
                fa_compute_slopes(&factx, kv_ਸਿਰ, n_ਪੰਕਤੀਆਂ_g);
                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_ਸ਼ੁਰੂ);

                ਅਚਲ size_t k_src_stride = ਮਾਪ_k_ਪੰਕਤੀ_padded / ਆਕਾਰ(__fp16);
                ਅਚਲ size_t v_src_stride = ਮਾਪ_v_ਪੰਕਤੀ_padded / ਆਕਾਰ(__fp16);

                hmx_ਕਤਾਰ_t hmx_q = ctx->hmx_ਕਤਾਰ;

                ਜੇ (factx.pipeline) {
                    // Double-buffered job structs because HMX queue runs asynchronously
                    hmx_fa_qk_ਕੰਮ_t qk_ਕੰਮ[2];
                    hmx_fa_o_update_ਕੰਮ_t ou_ਕੰਮ[2];

                    // Prefetch second active block early if there are multiple active blocks
                    ਜੇ (n_ਸਰਗਰਮ_ਖੰਡਾਂ > 1) {
                        fa_prefetch_ਖੰਡ(dma_q, k, v, ਨਕਾਬ, ਸਰਗਰਮ_ਖੰਡਾਂ[1], Bc, ਮਾਪ_k_ਪੰਕਤੀ_padded, ਮਾਪ_k_ਪੰਕਤੀ, ਮਾਪ_v_ਪੰਕਤੀ_padded, ਮਾਪ_v_ਪੰਕਤੀ,
                                          ik2, ik3, iv2, iv3, q_ਸ਼ੁਰੂ, im3, kv_ਸਿਰ, G, m_ਸਤਰ_ਬਾਈਟਾਂ, n_ਪੰਕਤੀਆਂ_q, nek1, 1, &factx);
                    }

                    // Prep and start QK-dot(0)
                    ਖਾਲੀ * curr_k0 = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
                    fa_phase_k_interleave(&factx, b0_ਪੰਕਤੀਆਂ, k_src_stride, curr_k0, b0_ਸ਼ੁਰੂ, 0);

                    qk_ਕੰਮ[0].q_tiles        = factx.vtcm_q_tiles;
                    qk_ਕੰਮ[0].k_tiles        = factx.vtcm_k_tiles[0];
                    qk_ਕੰਮ[0].s_tiles        = factx.vtcm_s_tiles[0];
                    qk_ਕੰਮ[0].n_ਪੰਕਤੀ_tiles    = n_ਪੰਕਤੀ_tiles;
                    qk_ਕੰਮ[0].n_ਖਾਨਾ_tiles    = hmx_ceil_div(b0_ਪੰਕਤੀਆਂ, HMX_FP16_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ);
                    qk_ਕੰਮ[0].n_dot_tiles    = DK_pad / 32;
                    qk_ਕੰਮ[0].n_tiles_ਪ੍ਰਤੀ_bc = n_tiles_ਪ੍ਰਤੀ_bc;
                    qk_ਕੰਮ[0].hmx_scales     = factx.vtcm_hmx_scales_qk;
                    hmx_ਕਤਾਰ_ਧੱਕੋ(hmx_q, hmx_ਕਤਾਰ_make_desc(hmx_fa_qk_dot_ਕਾਮਾ, &qk_ਕੰਮ[0]));

                    ਲਈ (uint32_t ਕਦਮ = 0; ਕਦਮ < n_ਸਰਗਰਮ_ਖੰਡਾਂ; ++ਕਦਮ) {
                        ਅਚਲ uint32_t kv_blk      = ਸਰਗਰਮ_ਖੰਡਾਂ[ਕਦਮ];
                        ਅਚਲ uint32_t kv_ਸ਼ੁਰੂ    = kv_blk * Bc;
                        ਅਚਲ uint32_t kv_ਪੰਕਤੀਆਂ     = hex_smin(Bc, nek1 - kv_ਸ਼ੁਰੂ);
                        ਅਚਲ size_t   n_ਖਾਨਾ_tiles = hmx_ceil_div(kv_ਪੰਕਤੀਆਂ, HMX_FP16_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ);

                        // ---- 1. Pop and run V-prep for current block ----
                        ਖਾਲੀ * curr_v = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_ਪੰਕਤੀਆਂ, v_src_stride, curr_v, factx.vtcm_v_tiles[buf_idx], n_tiles_ਪ੍ਰਤੀ_bc, kv_ਸ਼ੁਰੂ);

                        // ---- 2. Pop and run mask-prep for current block ----
                        __fp16 * current_ਨਕਾਬ_vtcm = ਨਿਰਾ;
                        ਜੇ (ਨਕਾਬ) {
                            ਜੇ (__builtin_expect(factx.ਨਕਾਬ_ਪ੍ਰਸਾਰਣ, true)) {
                                current_ਨਕਾਬ_vtcm = (__fp16 *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
                            } ਨਹੀਂਤਾਂ {
                                fa_ਕੱਢੋ_ਨਕਾਬ_dma_gqa(dma_q, G);
                                current_ਨਕਾਬ_vtcm = factx.vtcm_ਨਕਾਬ_buf;
                            }
                        }

                        // ---- 3. Start HMX O update for step - 1 (reads P[1 - buf_idx], V[1 - buf_idx], D) ----
                        // O update relies on the previous block's P and V tiles.
                        // O update MUST be pushed before the next block's QK-dot: hmx_queue_pop() retires the
                        // oldest descriptor, so push order alone decides which pop waits for which job.
                        // If OU went in after QK(i+1), the pop below would retire QK(i+1) and leave
                        // OU(i-1) in flight into the next iteration, where V-prep overwrites V[prev_buf].
                        ਜੇ (ਕਦਮ > 0) {
                            ਅਚਲ uint32_t prev_blk      = ਸਰਗਰਮ_ਖੰਡਾਂ[ਕਦਮ - 1];
                            ਅਚਲ size_t prev_buf        = 1 - buf_idx;
                            ou_ਕੰਮ[prev_buf].o_curr      = o_tile_curr;
                            ou_ਕੰਮ[prev_buf].o_prev      = o_tile_prev;
                            ou_ਕੰਮ[prev_buf].p_tiles     = factx.vtcm_p_tiles[prev_buf];
                            ou_ਕੰਮ[prev_buf].v_tiles     = factx.vtcm_v_tiles[prev_buf];
                            ou_ਕੰਮ[prev_buf].d_tiles     = factx.vtcm_d_tiles[prev_buf];
                            ou_ਕੰਮ[prev_buf].hmx_scales  = factx.vtcm_hmx_scales_id;
                            ou_ਕੰਮ[prev_buf].n_ਪੰਕਤੀ_tiles = n_ਪੰਕਤੀ_tiles;
                            ou_ਕੰਮ[prev_buf].n_ਖਾਨਾ_tiles =
                                hmx_ceil_div(hex_smin(Bc, nek1 - prev_blk * Bc), HMX_FP16_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ);
                            ou_ਕੰਮ[prev_buf].n_ਪੰਕਤੀ_tiles_g_br = n_ਪੰਕਤੀ_tiles_g_br;
                            ou_ਕੰਮ[prev_buf].n_tiles_ਪ੍ਰਤੀ_bc   = n_tiles_ਪ੍ਰਤੀ_bc;
                            ou_ਕੰਮ[prev_buf].DV               = DV_pad;
                            hmx_ਕਤਾਰ_ਧੱਕੋ(hmx_q, hmx_ਕਤਾਰ_make_desc(hmx_fa_o_update_ਕਾਮਾ, &ou_ਕੰਮ[prev_buf]));
                        }

                        // ---- 4. Pop and run K-prep for next active block & push next QK-dot ----
                        ਜੇ (ਕਦਮ + 1 < n_ਸਰਗਰਮ_ਖੰਡਾਂ) {
                            ਅਚਲ uint32_t ਅਗਲਾ_blk   = ਸਰਗਰਮ_ਖੰਡਾਂ[ਕਦਮ + 1];
                            ਅਚਲ uint32_t ਅਗਲਾ_ਸ਼ੁਰੂ = ਅਗਲਾ_blk * Bc;
                            ਅਚਲ uint32_t ਅਗਲਾ_ਪੰਕਤੀਆਂ  = hex_smin(Bc, nek1 - ਅਗਲਾ_ਸ਼ੁਰੂ);
                            ਅਚਲ size_t   ਅਗਲਾ_buf   = 1 - buf_idx;

                            ਖਾਲੀ * ਅਗਲਾ_k = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
                            fa_phase_k_interleave(&factx, ਅਗਲਾ_ਪੰਕਤੀਆਂ, k_src_stride, ਅਗਲਾ_k, ਅਗਲਾ_ਸ਼ੁਰੂ, ਅਗਲਾ_buf);

                            qk_ਕੰਮ[ਅਗਲਾ_buf].q_tiles        = factx.vtcm_q_tiles;
                            qk_ਕੰਮ[ਅਗਲਾ_buf].k_tiles        = factx.vtcm_k_tiles[ਅਗਲਾ_buf];
                            qk_ਕੰਮ[ਅਗਲਾ_buf].s_tiles        = factx.vtcm_s_tiles[ਅਗਲਾ_buf];
                            qk_ਕੰਮ[ਅਗਲਾ_buf].n_ਪੰਕਤੀ_tiles    = n_ਪੰਕਤੀ_tiles;
                            qk_ਕੰਮ[ਅਗਲਾ_buf].n_ਖਾਨਾ_tiles    = hmx_ceil_div(ਅਗਲਾ_ਪੰਕਤੀਆਂ, HMX_FP16_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ);
                            qk_ਕੰਮ[ਅਗਲਾ_buf].n_dot_tiles    = DK_pad / 32;
                            qk_ਕੰਮ[ਅਗਲਾ_buf].n_tiles_ਪ੍ਰਤੀ_bc = n_tiles_ਪ੍ਰਤੀ_bc;
                            qk_ਕੰਮ[ਅਗਲਾ_buf].hmx_scales     = factx.vtcm_hmx_scales_qk;
                            hmx_ਕਤਾਰ_ਧੱਕੋ(hmx_q, hmx_ਕਤਾਰ_make_desc(hmx_fa_qk_dot_ਕਾਮਾ, &qk_ਕੰਮ[ਅਗਲਾ_buf]));
                        }

                        // ---- 5. Wait for current block's QK-dot to finish ----
                        hmx_ਕਤਾਰ_ਕੱਢੋ(hmx_q);

                        // ---- 6. Phase 2: softmax + build_D ----
                        fa_softmax_ਦਲੀਲਾਂ_t sargs;
                        memset(&sargs, 0, ਆਕਾਰ(sargs));
                        sargs.factx                = &factx;
                        sargs.buf_idx              = buf_idx;
                        sargs.kv_ਪੰਕਤੀਆਂ              = kv_ਪੰਕਤੀਆਂ;
                        sargs.n_ਪੰਕਤੀਆਂ_g             = n_ਪੰਕਤੀਆਂ_g;
                        sargs.n_ਖਾਨਾ_tiles          = n_ਖਾਨਾ_tiles;
                        sargs.n_tiles_ਪ੍ਰਤੀ_bc       = n_tiles_ਪ੍ਰਤੀ_bc;
                        sargs.n_ਪੰਕਤੀ_tiles          = n_ਪੰਕਤੀ_tiles;
                        sargs.n_ਪੰਕਤੀ_tiles_g_br     = n_ਪੰਕਤੀ_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_ਸਿਰ              = kv_ਸਿਰ;
                        sargs.kv_ਸ਼ੁਰੂ             = kv_ਸ਼ੁਰੂ;
                        sargs.q_ਸ਼ੁਰੂ              = q_ਸ਼ੁਰੂ;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.ਨਕਾਬ                 = ਨਕਾਬ;
                        sargs.ਨਕਾਬ_vtcm            = current_ਨਕਾਬ_vtcm;
                        sargs.ਨਕਾਬ_vtcm_ਪੰਕਤੀ_stride = factx.ਨਕਾਬ_buf_ਪੰਕਤੀ_stride;
                        sargs.slopes               = factx.vtcm_slopes;

                        // Run Softmax on HVX (blocking call)
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_ਪੰਕਤੀ_tiles, n_ਪੰਕਤੀ_tiles_g_br);

                        // Wait for HMX O update for step - 1 to finish
                        ਜੇ (ਕਦਮ > 0) {
                            hmx_ਕਤਾਰ_ਕੱਢੋ(hmx_q);
                            hex_ਅਦਲਾ_ਬਦਲੀ_ptr((ਖਾਲੀ **) &o_tile_curr, (ਖਾਲੀ **) &o_tile_prev);
                        }

                        // Prefetch block step + 2
                        ਜੇ (ਕਦਮ + 2 < n_ਸਰਗਰਮ_ਖੰਡਾਂ) {
                            fa_prefetch_ਖੰਡ(dma_q, k, v, ਨਕਾਬ, ਸਰਗਰਮ_ਖੰਡਾਂ[ਕਦਮ + 2], Bc, ਮਾਪ_k_ਪੰਕਤੀ_padded, ਮਾਪ_k_ਪੰਕਤੀ, ਮਾਪ_v_ਪੰਕਤੀ_padded, ਮਾਪ_v_ਪੰਕਤੀ,
                                              ik2, ik3, iv2, iv3, q_ਸ਼ੁਰੂ, im3, kv_ਸਿਰ, G, m_ਸਤਰ_ਬਾਈਟਾਂ, n_ਪੰਕਤੀਆਂ_q, nek1, buf_idx, &factx);
                        }

                        buf_idx = 1 - buf_idx;
                    }

                    // Epilogue
                    ਜੇ (n_ਸਰਗਰਮ_ਖੰਡਾਂ > 0) {
                        ਅਚਲ uint32_t ਆਖਰੀ_blk = ਸਰਗਰਮ_ਖੰਡਾਂ[n_ਸਰਗਰਮ_ਖੰਡਾਂ - 1];
                        ਅਚਲ size_t ਆਖਰੀ_ਖਾਨੇ_ਗਿਣਤੀ  = hmx_ceil_div(hex_smin(Bc, nek1 - ਆਖਰੀ_blk * Bc), HMX_FP16_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ);
                        ou_ਕੰਮ[0].o_curr           = o_tile_curr;
                        ou_ਕੰਮ[0].o_prev           = o_tile_prev;
                        ou_ਕੰਮ[0].p_tiles          = factx.vtcm_p_tiles[1 - buf_idx];
                        ou_ਕੰਮ[0].v_tiles          = factx.vtcm_v_tiles[1 - buf_idx];
                        ou_ਕੰਮ[0].d_tiles          = factx.vtcm_d_tiles[1 - buf_idx];
                        ou_ਕੰਮ[0].hmx_scales       = factx.vtcm_hmx_scales_id;
                        ou_ਕੰਮ[0].n_ਪੰਕਤੀ_tiles      = n_ਪੰਕਤੀ_tiles;
                        ou_ਕੰਮ[0].n_ਖਾਨਾ_tiles      = ਆਖਰੀ_ਖਾਨੇ_ਗਿਣਤੀ;
                        ou_ਕੰਮ[0].n_ਪੰਕਤੀ_tiles_g_br = n_ਪੰਕਤੀ_tiles_g_br;
                        ou_ਕੰਮ[0].n_tiles_ਪ੍ਰਤੀ_bc   = n_tiles_ਪ੍ਰਤੀ_bc;
                        ou_ਕੰਮ[0].DV               = DV_pad;
                        hmx_ਕਤਾਰ_ਧੱਕੋ(hmx_q, hmx_ਕਤਾਰ_make_desc(hmx_fa_o_update_ਕਾਮਾ, &ou_ਕੰਮ[0]));

                        // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                        htp_trace_event_ਸ਼ੁਰੂ(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_ਸ਼ੁਰੂ);
                        fa_build_d_diag_inv_l(&factx, n_ਪੰਕਤੀ_tiles, n_ਪੰਕਤੀ_tiles_g_br, kv_ਸਿਰ, n_ਪੰਕਤੀਆਂ_g);
                        htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_ਸ਼ੁਰੂ);
                        hmx_ਕਤਾਰ_ਕੱਢੋ(hmx_q);

                        hex_ਅਦਲਾ_ਬਦਲੀ_ptr((ਖਾਲੀ **) &o_tile_curr, (ਖਾਲੀ **) &o_tile_prev);
                    }

                } ਨਹੀਂਤਾਂ {
                    // Fallback path
                    ਲਈ (uint32_t ਕਦਮ = 0; ਕਦਮ < n_ਸਰਗਰਮ_ਖੰਡਾਂ; ++ਕਦਮ) {
                        ਅਚਲ uint32_t kv_blk      = ਸਰਗਰਮ_ਖੰਡਾਂ[ਕਦਮ];
                        ਅਚਲ uint32_t kv_ਸ਼ੁਰੂ    = kv_blk * Bc;
                        ਅਚਲ uint32_t kv_ਪੰਕਤੀਆਂ     = hex_smin(Bc, nek1 - kv_ਸ਼ੁਰੂ);
                        ਅਚਲ size_t   n_ਖਾਨਾ_tiles = hmx_ceil_div(kv_ਪੰਕਤੀਆਂ, HMX_FP16_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ);

                        ਜੇ (ਨਕਾਬ) {
                            ਜੇ (__builtin_expect(factx.ਨਕਾਬ_ਪ੍ਰਸਾਰਣ, true)) {
                                ਅਚਲ dma_addr_t ms_src = ਨਕਾਬ->ਅੰਕੜੇ + q_ਸ਼ੁਰੂ * ਨਕਾਬ->nb[1] + im3 * ਨਕਾਬ->nb[3] + kv_ਸ਼ੁਰੂ * ਆਕਾਰ(__fp16);
                                dma_cache_fa_ਧੱਕੋ(dma_q, &factx.m_cache, ms_src, m_ਸਤਰ_ਬਾਈਟਾਂ, ਨਕਾਬ->nb[1], kv_ਪੰਕਤੀਆਂ * ਆਕਾਰ(__fp16), n_ਪੰਕਤੀਆਂ_q);
                            } ਨਹੀਂਤਾਂ {
                                fa_ਧੱਕੋ_ਨਕਾਬ_dma_gqa(dma_q, ਨਕਾਬ, q_ਸ਼ੁਰੂ, im3, kv_ਸ਼ੁਰੂ, kv_ਸਿਰ, G, m_ਸਤਰ_ਬਾਈਟਾਂ, kv_ਪੰਕਤੀਆਂ, n_ਪੰਕਤੀਆਂ_q, &factx);
                            }
                        }

                        ਜੇ (ਕਦਮ + 1 < n_ਸਰਗਰਮ_ਖੰਡਾਂ) {
                            ਅਚਲ uint32_t  prefetch_blk   = ਸਰਗਰਮ_ਖੰਡਾਂ[ਕਦਮ + 1];
                            ਅਚਲ uint32_t  prefetch_ਸ਼ੁਰੂ = prefetch_blk * Bc;
                            ਅਚਲ uint32_t  prefetch_ਪੰਕਤੀਆਂ  = hex_smin(Bc, nek1 - prefetch_ਸ਼ੁਰੂ);
                            ਅਚਲ size_t    prefetch_buf   = 1 - buf_idx;
                            ਅਚਲ dma_addr_t k_prefetch_src = k->ਅੰਕੜੇ + prefetch_ਸ਼ੁਰੂ * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(factx.vtcm_k_fp16[prefetch_buf], k_prefetch_src), ਮਾਪ_k_ਪੰਕਤੀ_padded, k->nb[1], ਮਾਪ_k_ਪੰਕਤੀ, prefetch_ਪੰਕਤੀਆਂ);
                            ਅਚਲ dma_addr_t v_prefetch_src = v->ਅੰਕੜੇ + prefetch_ਸ਼ੁਰੂ * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(factx.vtcm_v_fp16[prefetch_buf], v_prefetch_src), ਮਾਪ_v_ਪੰਕਤੀ_padded, v->nb[1], ਮਾਪ_v_ਪੰਕਤੀ, prefetch_ਪੰਕਤੀਆਂ);
                        }

                        // Wait for current K DMA and interleave
                        ਖਾਲੀ * curr_k = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
                        fa_phase_k_interleave(&factx, kv_ਪੰਕਤੀਆਂ, k_src_stride, curr_k, kv_ਸ਼ੁਰੂ, 0);

                        {
                            qk_ਕੰਮ.q_tiles        = factx.vtcm_q_tiles;
                            qk_ਕੰਮ.k_tiles        = factx.vtcm_k_tiles[0];
                            qk_ਕੰਮ.s_tiles        = factx.vtcm_s_tiles[0];
                            qk_ਕੰਮ.n_ਪੰਕਤੀ_tiles    = n_ਪੰਕਤੀ_tiles;
                            qk_ਕੰਮ.n_ਖਾਨਾ_tiles    = n_ਖਾਨਾ_tiles;
                            qk_ਕੰਮ.n_dot_tiles    = (size_t) (DK_pad / 32);
                            qk_ਕੰਮ.n_tiles_ਪ੍ਰਤੀ_bc = n_tiles_ਪ੍ਰਤੀ_bc;
                            qk_ਕੰਮ.hmx_scales     = factx.vtcm_hmx_scales_qk;

                            hmx_ਕਤਾਰ_ਧੱਕੋ(ctx->hmx_ਕਤਾਰ, hmx_ਕਤਾਰ_make_desc(hmx_fa_qk_dot_ਕਾਮਾ, &qk_ਕੰਮ));
                            hmx_ਕਤਾਰ_ਕੱਢੋ(ctx->hmx_ਕਤਾਰ);
                        }

                        // Wait for current V DMA and interleave
                        ਖਾਲੀ * curr_v = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_ਪੰਕਤੀਆਂ, v_src_stride, curr_v, factx.vtcm_v_tiles[0], n_tiles_ਪ੍ਰਤੀ_bc, kv_ਸ਼ੁਰੂ);

                        // ---- Phase 3: softmax + build_D ----
                        __fp16 * current_ਨਕਾਬ_vtcm = ਨਿਰਾ;
                        ਜੇ (ਨਕਾਬ) {
                            ਜੇ (__builtin_expect(factx.ਨਕਾਬ_ਪ੍ਰਸਾਰਣ, true)) {
                                current_ਨਕਾਬ_vtcm = (__fp16 *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;
                            } ਨਹੀਂਤਾਂ {
                                fa_ਕੱਢੋ_ਨਕਾਬ_dma_gqa(dma_q, G);
                                current_ਨਕਾਬ_vtcm = factx.vtcm_ਨਕਾਬ_buf;
                            }
                        }

                        fa_softmax_ਦਲੀਲਾਂ_t sargs;
                        memset(&sargs, 0, ਆਕਾਰ(sargs));
                        sargs.factx                = &factx;
                        sargs.kv_ਪੰਕਤੀਆਂ              = kv_ਪੰਕਤੀਆਂ;
                        sargs.n_ਪੰਕਤੀਆਂ_g             = n_ਪੰਕਤੀਆਂ_g;
                        sargs.n_ਖਾਨਾ_tiles          = n_ਖਾਨਾ_tiles;
                        sargs.n_tiles_ਪ੍ਰਤੀ_bc       = n_tiles_ਪ੍ਰਤੀ_bc;
                        sargs.n_ਪੰਕਤੀ_tiles          = n_ਪੰਕਤੀ_tiles;
                        sargs.n_ਪੰਕਤੀ_tiles_g_br     = n_ਪੰਕਤੀ_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_ਸਿਰ              = kv_ਸਿਰ;
                        sargs.kv_ਸ਼ੁਰੂ             = kv_ਸ਼ੁਰੂ;
                        sargs.q_ਸ਼ੁਰੂ              = q_ਸ਼ੁਰੂ;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.ਨਕਾਬ                 = ਨਕਾਬ;
                        sargs.ਨਕਾਬ_vtcm            = current_ਨਕਾਬ_vtcm;
                        sargs.ਨਕਾਬ_vtcm_ਪੰਕਤੀ_stride = factx.ਨਕਾਬ_buf_ਪੰਕਤੀ_stride;
                        sargs.slopes               = factx.vtcm_slopes;
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_ਪੰਕਤੀ_tiles, n_ਪੰਕਤੀ_tiles_g_br);

                        {
                            ou_ਕੰਮ.o_curr           = o_tile_curr;
                            ou_ਕੰਮ.o_prev           = o_tile_prev;
                            ou_ਕੰਮ.p_tiles          = factx.vtcm_p_tiles[0];
                            ou_ਕੰਮ.v_tiles          = factx.vtcm_v_tiles[0];
                            ou_ਕੰਮ.d_tiles          = factx.vtcm_d_tiles[0];
                            ou_ਕੰਮ.hmx_scales       = factx.vtcm_hmx_scales_id;
                            ou_ਕੰਮ.n_ਪੰਕਤੀ_tiles      = n_ਪੰਕਤੀ_tiles;
                            ou_ਕੰਮ.n_ਖਾਨਾ_tiles      = n_ਖਾਨਾ_tiles;
                            ou_ਕੰਮ.n_ਪੰਕਤੀ_tiles_g_br = n_ਪੰਕਤੀ_tiles_g_br;
                            ou_ਕੰਮ.n_tiles_ਪ੍ਰਤੀ_bc   = n_tiles_ਪ੍ਰਤੀ_bc;
                            ou_ਕੰਮ.DV               = DV_pad;

                            hmx_ਕਤਾਰ_ਧੱਕੋ(ctx->hmx_ਕਤਾਰ, hmx_ਕਤਾਰ_make_desc(hmx_fa_o_update_ਕਾਮਾ, &ou_ਕੰਮ));
                            ਜੇ (ਕਦਮ + 1 == n_ਸਰਗਰਮ_ਖੰਡਾਂ) {
                                // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                                htp_trace_event_ਸ਼ੁਰੂ(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_ਸ਼ੁਰੂ);
                                fa_build_d_diag_inv_l(&factx, n_ਪੰਕਤੀ_tiles, n_ਪੰਕਤੀ_tiles_g_br, kv_ਸਿਰ, n_ਪੰਕਤੀਆਂ_g);
                                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_ਸ਼ੁਰੂ);
                            }
                            hmx_ਕਤਾਰ_ਕੱਢੋ(ctx->hmx_ਕਤਾਰ);

                            hex_ਅਦਲਾ_ਬਦਲੀ_ptr((ਖਾਲੀ **) &o_tile_curr, (ਖਾਲੀ **) &o_tile_prev);
                        }

                        buf_idx = 1 - buf_idx;
                    }
                }

                // Enqueue DMAs for the next kv_head early so they overlap with O-PROC
                ਅਚਲ uint32_t ਅਗਲਾ_kv_ਸਿਰ = kv_ਸਿਰ + 1;
                ਅਚਲ bool has_ਅਗਲਾ = (ਅਗਲਾ_kv_ਸਿਰ < kv_ਸਿਰ_max);

                ਜੇ (has_ਅਗਲਾ) {
                    ਅਚਲ dma_addr_t ਅਗਲਾ_q_ptr = q->ਅੰਕੜੇ + q_ਸ਼ੁਰੂ * q->nb[1] + (ਅਗਲਾ_kv_ਸਿਰ * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    ਅਚਲ size_t ਅਗਲਾ_q_ਪੰਕਤੀ_ਬਾਈਟਾਂ = q_transposed ? n_ਪੰਕਤੀਆਂ_q * q_ਪੰਕਤੀ_ਬਾਈਟਾਂ_trans_ਗੁਣਕ : q_ਪੰਕਤੀ_ਬਾਈਟਾਂ_untransposed;
                    ਅਚਲ size_t ਅਗਲਾ_n_ਪੰਕਤੀਆਂ      = q_transposed ? factx.G : n_ਪੰਕਤੀਆਂ_q;
                    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(factx.vtcm_q_dma, ਅਗਲਾ_q_ptr), ਅਗਲਾ_q_ਪੰਕਤੀ_ਬਾਈਟਾਂ, hex_smax(q_src_stride, ਅਗਲਾ_q_ਪੰਕਤੀ_ਬਾਈਟਾਂ), ਅਗਲਾ_q_ਪੰਕਤੀ_ਬਾਈਟਾਂ, ਅਗਲਾ_n_ਪੰਕਤੀਆਂ);

                    ਜੇ (n_ਸਰਗਰਮ_ਖੰਡਾਂ > 0) {
                        ਅਚਲ uint32_t ਅਗਲਾ_ik2 = ਅਗਲਾ_kv_ਸਿਰ;
                        ਅਚਲ uint32_t ਅਗਲਾ_iv2 = ਅਗਲਾ_kv_ਸਿਰ;

                        ਅਚਲ dma_addr_t ਅਗਲਾ_k_src = k->ਅੰਕੜੇ + b0_ਸ਼ੁਰੂ * k->nb[1] + ਅਗਲਾ_ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(factx.vtcm_k_fp16[0], ਅਗਲਾ_k_src), ਮਾਪ_k_ਪੰਕਤੀ_padded, k->nb[1], ਮਾਪ_k_ਪੰਕਤੀ, b0_ਪੰਕਤੀਆਂ);

                        ਅਚਲ dma_addr_t ਅਗਲਾ_v_src = v->ਅੰਕੜੇ + b0_ਸ਼ੁਰੂ * v->nb[1] + ਅਗਲਾ_iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(factx.vtcm_v_fp16[0], ਅਗਲਾ_v_src), ਮਾਪ_v_ਪੰਕਤੀ_padded, v->nb[1], ਮਾਪ_v_ਪੰਕਤੀ, b0_ਪੰਕਤੀਆਂ);

                        ਜੇ (factx.pipeline && ਨਕਾਬ) {
                            ਜੇ (__builtin_expect(factx.ਨਕਾਬ_ਪ੍ਰਸਾਰਣ, true)) {
                                ਅਚਲ dma_addr_t ms_src = ਨਕਾਬ->ਅੰਕੜੇ + q_ਸ਼ੁਰੂ * ਨਕਾਬ->nb[1] + im3 * ਨਕਾਬ->nb[3] + b0_ਸ਼ੁਰੂ * ਆਕਾਰ(__fp16);
                                dma_cache_fa_ਧੱਕੋ(dma_q, &factx.m_cache, ms_src, m_ਸਤਰ_ਬਾਈਟਾਂ, ਨਕਾਬ->nb[1], b0_ਪੰਕਤੀਆਂ * ਆਕਾਰ(__fp16), n_ਪੰਕਤੀਆਂ_q);
                            } ਨਹੀਂਤਾਂ {
                                fa_ਧੱਕੋ_ਨਕਾਬ_dma_gqa(dma_q, ਨਕਾਬ, q_ਸ਼ੁਰੂ, im3, b0_ਸ਼ੁਰੂ, ਅਗਲਾ_kv_ਸਿਰ, G, m_ਸਤਰ_ਬਾਈਟਾਂ, b0_ਪੰਕਤੀਆਂ, n_ਪੰਕਤੀਆਂ_q, &factx);
                            }
                        }
                    }
                }

                // ---- Final normalization ----
                {
                    on_ਕੰਮ.o_curr           = o_tile_curr;
                    on_ਕੰਮ.o_prev           = o_tile_prev;
                    on_ਕੰਮ.d_tiles          = factx.vtcm_d_inv_l;
                    on_ਕੰਮ.hmx_scales       = factx.vtcm_hmx_scales_id;
                    on_ਕੰਮ.n_ਪੰਕਤੀ_tiles      = n_ਪੰਕਤੀ_tiles;
                    on_ਕੰਮ.n_ਪੰਕਤੀ_tiles_g_br = n_ਪੰਕਤੀ_tiles_g_br;
                    on_ਕੰਮ.DV               = DV_pad;
                    hmx_ਕਤਾਰ_ਧੱਕੋ(ctx->hmx_ਕਤਾਰ, hmx_ਕਤਾਰ_make_desc(hmx_fa_o_norm_ਕਾਮਾ, &on_ਕੰਮ));
                    hmx_ਕਤਾਰ_ਕੱਢੋ(ctx->hmx_ਕਤਾਰ);
                }

                // ---- Store O block ----
                fa_phase_o_store(&factx, dst, o_tile_curr, q_ਸ਼ੁਰੂ, kv_ਸਿਰ, ib3, n_ਪੰਕਤੀਆਂ_g);
            }
        }
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਪੂਰਨ op_flash_attn_ext(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * q    = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * k    = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * v    = octx->src[2];
    ਅਚਲ ਬਣਤਰ htp_tensor * ਨਕਾਬ = octx->src[3];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    // Check support
    ਜੇ ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) ||
        (k->type != HTP_TYPE_F16 && k->type != HTP_TYPE_Q8_0) ||
        (v->type != HTP_TYPE_F16 && v->type != HTP_TYPE_Q8_0)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (htp_tensor_is_extended(dst)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ ਬਣਤਰ htp_fa_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_fa_kernel_params *) octx->kernel_params;

    ਜੇ (kparams->kernel_type == HTP_FA_KERNEL_UNSUPPORTED || kparams->n_kv_ਖੰਡਾਂ > FA_MAX_KV_ਖੰਡਾਂ) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, kparams->n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਜੇ (kparams->kernel_type == HTP_FA_KERNEL_HMX) {
        ਵਾਪਸ hmx_flash_attn_ext(octx);
    }

    ਬਣਤਰ htp_fa_context factx;
    factx.octx = octx;
    factx.k = k;
    factx.v = v;

    factx.src0_div21 = kparams->u.hvx.src0_div21;
    factx.src0_div1  = kparams->u.hvx.src0_div1;

    factx.ਪ੍ਰਸਾਰਣ_rk2 = kparams->ਪ੍ਰਸਾਰਣ_rk2;
    factx.ਪ੍ਰਸਾਰਣ_rk3 = kparams->ਪ੍ਰਸਾਰਣ_rk3;
    factx.ਪ੍ਰਸਾਰਣ_rv2 = kparams->ਪ੍ਰਸਾਰਣ_rv2;
    factx.ਪ੍ਰਸਾਰਣ_rv3 = kparams->ਪ੍ਰਸਾਰਣ_rv3;

    ਜੇ (ਨਕਾਬ) {
        factx.src3_div2 = kparams->src3_div2;
        factx.src3_div3 = kparams->src3_div3;
    }

    factx.is_q_fp32 = (q->type == HTP_TYPE_F32);
    factx.ਮਾਪ_q_ਪੰਕਤੀ_padded = kparams->u.hvx.ਮਾਪ_q_ਪੰਕਤੀ_padded;
    factx.ਮਾਪ_k_ਪੰਕਤੀ_padded = kparams->u.hvx.ਮਾਪ_k_ਪੰਕਤੀ_padded;
    factx.ਮਾਪ_v_ਪੰਕਤੀ_padded = kparams->u.hvx.ਮਾਪ_v_ਪੰਕਤੀ_padded;

    factx.n_ਖੰਡਾਂ = kparams->n_kv_ਖੰਡਾਂ;

    factx.scale = kparams->scale;
    factx.max_bias = kparams->max_bias;
    factx.has_softcap = (kparams->logit_softcap != 0.0f);
    factx.logit_softcap = 0;
    ਜੇ (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_ਸਿਰ_log2 = kparams->n_ਸਿਰ_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    ਅਚਲ uint32_t n_ਸਿਰ = q->ne[2];
    ਜੇ (n_ਸਿਰ > 512) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }
    ਲਈ (uint32_t h = 0; h < n_ਸਿਰ; ++h) {
        factx.slopes[h] = (__fp16) ((kparams->max_bias > 0.0f) ? alibi_ਢਲਾਨ(h, factx.n_ਸਿਰ_log2, factx.m0, factx.m1) : 1.0f);
    }

    // total rows in q
    ਅਚਲ uint32_t neq1 = q->ne[1];
    ਅਚਲ uint32_t neq2 = q->ne[2];
    ਅਚਲ uint32_t neq3 = q->ne[3];
    ਅਚਲ uint32_t ਕੁੱਲ_qrows = neq1 * neq2 * neq3;

    uint32_t qrow_ਸ਼ੁਰੂ = 0;
    uint32_t qrows      = ਕੁੱਲ_qrows;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ uint32_t mdev_ਗਿਣਤੀ = octx->ctx->mdev.ਗਿਣਤੀ_2;
        ਅਚਲ uint32_t mdev_idx   = octx->ctx->mdev.idx;
        ਅਚਲ uint32_t n_kv_heads = k->ne[2];
        ਅਚਲ uint32_t dst_e_ਮਾਪ = (dst->type == HTP_TYPE_F32) ? ਆਕਾਰ(ਦਸ਼ਮਲਵ) : ਆਕਾਰ(__fp16);
        ਅਚਲ bool can_ਵੰਡੋ      = htp_tensor_can_ਪੰਕਤੀ_ਹਿੱਸੇ_ਕਰੋ(dst, dst_e_ਮਾਪ);

        // head range is contiguous in flat row space only when neq3 == 1
        ਜੇ (kparams->ਸਿਰ_ਵੰਡੋ && can_ਵੰਡੋ && neq3 == 1 && n_kv_heads >= mdev_ਗਿਣਤੀ && n_kv_heads % mdev_ਗਿਣਤੀ == 0) {
            ਅਚਲ uint32_t G              = kparams->G;
            ਅਚਲ uint32_t kv_ਪ੍ਰਤੀ_core    = n_kv_heads / mdev_ਗਿਣਤੀ;
            ਅਚਲ uint32_t heads_ਪ੍ਰਤੀ_core = kv_ਪ੍ਰਤੀ_core * G;
            ਅਚਲ uint32_t ਸਿਰ_ਸ਼ੁਰੂ     = mdev_idx * heads_ਪ੍ਰਤੀ_core;
            qrow_ਸ਼ੁਰੂ = ਸਿਰ_ਸ਼ੁਰੂ * neq1;
            qrows      = heads_ਪ੍ਰਤੀ_core * neq1;
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(ਕੁੱਲ_qrows, can_ਵੰਡੋ ? 1 : 0, mdev_idx, mdev_ਗਿਣਤੀ, &octx->ctx->mdev.ਗਿਣਤੀ_div);
            qrow_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
            qrows      = range.ਗਿਣਤੀ_2;
        }
    }

    ਜੇ (qrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

    factx.qrows            = qrows;
    factx.qrow_ਸ਼ੁਰੂ       = qrow_ਸ਼ੁਰੂ;
    factx.qrows_ਪ੍ਰਤੀ_thread = fastdiv(qrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);

    ਅਚਲ bool has_ਨਕਾਬ  = (ਨਕਾਬ != ਨਿਰਾ);
    ਅਚਲ bool has_sinks = (octx->src[4] != ਨਿਰਾ);
    ਬਣਤਰ hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, k->ne[0], v->ne[0], factx.is_q_fp32, has_ਨਕਾਬ, has_sinks, n_ਸਿਰ, n_ਧਾਗੇ);

    ਜੇ (L.ਕੁੱਲ_ਬਾਈਟਾਂ > octx->ctx->vtcm_ਮਾਪ) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    factx.ਮਾਪ_q_ਖੰਡ = L.ਮਾਪ_q_ਖੰਡ;
    factx.ਮਾਪ_k_ਖੰਡ = L.ਮਾਪ_k_ਖੰਡ;
    factx.ਮਾਪ_v_ਖੰਡ = L.ਮਾਪ_v_ਖੰਡ;
    factx.ਮਾਪ_m_ਖੰਡ = L.ਮਾਪ_m_ਖੰਡ;
    factx.ਮਾਪ_vkq_acc = L.ਮਾਪ_vkq_acc;

    uint8_t * ਅਚਲ ਅਧਾਰ = octx->ctx->vtcm_ਅਧਾਰ;

    factx.spad_q               = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_q);
    factx.spad_k               = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_k);
    factx.spad_v               = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_v);
    factx.spad_m               = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, ਅਧਾਰ, L.off_m, has_ਨਕਾਬ);
    factx.spad_a               = VTCM_LAYOUT_PTR(uint8_t, ਅਧਾਰ, L.off_a);
    factx.spad_sinks           = VTCM_LAYOUT_PTR_OPTIONAL(ਦਸ਼ਮਲਵ, ਅਧਾਰ, L.off_sinks, has_sinks);
    factx.n_ਸਰਗਰਮ_ਖੰਡਾਂ      = factx.n_ਖੰਡਾਂ;
    factx.shared_ਸਰਗਰਮ_ਖੰਡਾਂ = false;

    ਜੇ (has_ਨਕਾਬ && neq1 == 1 && neq3 == 1 && ਨਕਾਬ->ne[2] == 1 && factx.n_ਖੰਡਾਂ > 1 && factx.n_ਖੰਡਾਂ <= FA_MAX_KV_ਖੰਡਾਂ) {
        factx.n_ਸਰਗਰਮ_ਖੰਡਾਂ = fa_ਜਾਂਚ_ਸਰਗਰਮ_ਖੰਡਾਂ(
            octx->ctx->dma[0], ਨਕਾਬ, 0, 1, 0, 0, k->ne[1], FLASH_ATTN_ਖੰਡ_ਮਾਪ,
            factx.n_ਖੰਡਾਂ, factx.spad_k, factx.ਮਾਪ_k_ਖੰਡ * 2, factx.ਸਰਗਰਮ_ਖੰਡਾਂ);
        factx.shared_ਸਰਗਰਮ_ਖੰਡਾਂ = true;
    }

    ਜੇ (has_sinks) {
        ਅਚਲ ਬਣਤਰ htp_tensor * sinks = octx->src[4];
        dma_ਕਤਾਰ * dma_q = octx->ctx->dma[0];
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(factx.spad_sinks, sinks->ਅੰਕੜੇ), L.ਮਾਪ_sinks, 0, sinks->ਮਾਪ, 1);
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
    }

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, flash_attn_ext_f16_thread, &factx, n_ਧਾਗੇ);

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}
