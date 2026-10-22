#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <assert.h>
#شامل <HAP_compute_res.h>
#شامل <HAP_farf.h>
#شامل <math.h>
#شامل <stdbool.h>
#شامل <stdatomic.h>
#شامل <stddef.h>
#شامل <stdint.h>
#شامل <string.h>

#شامل "dma-queue.h"
#شامل "hex-fastdiv.h"
#شامل "hex-profile.h"
#شامل "hmx-queue.h"
#شامل "hmx-utils.h"
#شامل "hvx-utils.h"
#شامل "hvx-dump.h"
#شامل "hvx-copy.h"
#شامل "hvx-reduce.h"
#شامل "hvx-flash-attn.h"
#شامل "htp-vtcm.h"
#شامل "work-queue.h"

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "hvx-quant.h"

#شامل "flash-attn-ops.h"
#شامل "hvx-fa-kernels.h"
#شامل "hmx-fa-kernels.h"

// Must be multiple of 32
#تعریف_کرو FLASH_ATTN_بلاک_سائز (32 * 2)

ساخت htp_fa_context {
    ثابت ساخت htp_ops_context * octx;

    ساخت fastdiv_اقدار src0_div21;
    ساخت fastdiv_اقدار src0_div1;

    ساخت fastdiv_اقدار نشر_rk2;
    ساخت fastdiv_اقدار نشر_rk3;
    ساخت fastdiv_اقدار نشر_rv2;
    ساخت fastdiv_اقدار نشر_rv3;

    ساخت fastdiv_اقدار src3_div2;
    ساخت fastdiv_اقدار src3_div3;

    اعشاری scale;
    اعشاری max_bias;
    bool  has_softcap;
    __fp16 logit_softcap;

    uint32_t n_سر_log2;
    اعشاری m0;
    اعشاری m1;
    __fp16 slopes[512];

    uint32_t n_بلاکیں;
    uint32_t n_فعال_بلاکیں;
    bool     shared_فعال_بلاکیں;

    size_t سائز_q_صف_padded;
    size_t سائز_k_صف_padded;
    size_t سائز_v_صف_padded;

    size_t سائز_k_بلاک;
    size_t سائز_v_بلاک;
    size_t سائز_m_بلاک;

    uint32_t qrows;
    uint32_t qrows_فی_thread;
    uint32_t qrow_آغاز;

    bool is_q_fp32;

    size_t سائز_q_بلاک;
    size_t سائز_vkq_acc;

    uint8_t *  spad_q;
    uint8_t *  spad_k;
    uint8_t *  spad_v;
    uint8_t *  spad_m;
    uint8_t *  spad_a;
    اعشاری *    spad_sinks;
    uint16_t   فعال_بلاکیں[FA_MAX_KV_بلاکیں];

    ثابت ساخت htp_tensor * k;
    ثابت ساخت htp_tensor * v;
};

ساخت hmx_fa_context {
    ثابت ساخت htp_ops_context * octx;
    ثابت ساخت htp_tensor *      sinks;  // attention sinks (src[4]), NULL if absent
    ثابت ساخت htp_tensor *      k;
    ثابت ساخت htp_tensor *      v;
    bool         pipeline;  // true when n_kv_blocks >= FA_MIN_KV_BLOCKS && n_threads >= 2
    uint32_t     n_دھاگے;

    // Op parameters
    __fp16       scale;
    اعشاری        max_bias;
    bool         has_softcap;
    __fp16       logit_softcap;
    uint32_t     n_سر_log2;
    اعشاری        m0, m1;

    // Dimensions
    uint32_t     DK, DV;
    uint32_t     DK_pad, DV_pad;  // head_dim rounded up to 64 for HMX tiling
    uint32_t     n_kv;        // kv_len
    uint32_t     n_kv_heads;  // number of KV heads
    uint32_t     n_heads;     // number of Q heads
    uint32_t     G;           // GQA factor = n_heads / n_kv_heads
    ساخت fastdiv_اقدار div_G;
    ساخت fastdiv_اقدار src3_div2;
    ساخت fastdiv_اقدار src3_div3;
    uint32_t     n_kv_بلاکیں;
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
    HVX_سمتیہ * vtcm_m_vec;           // Row max [g_br]
    HVX_سمتیہ * vtcm_l_vec;           // Row sum [g_br]
    HVX_سمتیہ * vtcm_s_rowmax;        // Softmax intermediate [g_br]
    HVX_سمتیہ * vtcm_p_rowsum;        // Softmax intermediate [g_br]
    HVX_سمتیہ * vtcm_صف_bufs;        // Per-thread softmax row scratch [n_threads][2][Bc/64]
    uint8_t *    vtcm_hmx_scales_id;   // HMX output scales (identity)
    uint8_t *    vtcm_hmx_scales_qk;   // HMX output scales (qk_scale)
    __fp16 *     vtcm_نقاب_buf;        // VTCM mask buffer [Br * m_line], DMA'd per KV block
    __fp16 *     vtcm_slopes;          // ALiBi slopes [g_br]
    اعشاری *      vtcm_sinks;           // Attention sinks
    size_t       صف_buf_stride;       // HVX vectors per row buffer (Bc/64)
    size_t       نقاب_buf_صف_stride;  // elements (__fp16) per row in mask buffer
    size_t       q_tile_بائٹیں;
    size_t       k_tile_بائٹیں;
    size_t       o_tile_بائٹیں;
    size_t       ستون_vec_بائٹیں;
    size_t       d_tile_بائٹیں;
    bool         نقاب_نشر;       // true when mask->ne[2] == 1 (head-independent, single 2D DMA)
    dma_cache_fa m_cache;
};

// Scan mask rows [q_start, q_start + n_rows_q) and build compact list of KV blocks with any entry != -inf.
ساکن uint32_t fa_جائزہ_فعال_بلاکیں(
        dma_قطار *               dma_q,
        ثابت ساخت htp_tensor * نقاب,
        uint32_t                  q_آغاز,
        uint32_t                  n_صفیں_q,
        uint32_t                  im2,
        uint32_t                  im3,
        uint32_t                  nek1,
        uint32_t                  Bc,
        uint32_t                  n_kv_بلاکیں,
        خالی *                    vtcm_scratch,
        size_t                    scratch_بائٹیں,
        uint16_t *                فعال_بلاکیں) {
    ثابت size_t صف_بائٹیں  = (size_t) nek1 * حجم(__fp16);
    ثابت size_t صف_stride = hex_باری_اوپر(صف_بائٹیں, 128);

    اگر (!نقاب || n_kv_بلاکیں <= 1 || n_kv_بلاکیں > FA_MAX_KV_بلاکیں ||
        (size_t) n_صفیں_q * صف_stride > scratch_بائٹیں) {
        #ہدایت clang loop vectorize(disable)
        برائے (uint32_t b = 0; b < n_kv_بلاکیں; ++b) {
            فعال_بلاکیں[b] = (uint16_t) b;
        }
        واپس n_kv_بلاکیں;
    }

    ثابت dma_addr_t m_src = نقاب->ڈیٹا + q_آغاز * نقاب->nb[1] + im2 * نقاب->nb[2] + im3 * نقاب->nb[3];
    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_scratch, m_src), صف_stride, نقاب->nb[1], صف_بائٹیں, n_صفیں_q);
    ثابت uint8_t * m_buf = (ثابت uint8_t *) dma_قطار_نکالو(dma_q).dst;

    ثابت HVX_سمتیہ v_ninf   = Q6_Vh_vsplat_R(0xFC00);
    uint32_t         n_فعال = 0;

    برائے (uint32_t b = 0; b < n_kv_بلاکیں; ++b) {
        ثابت uint32_t col0 = b * Bc;
        ثابت uint32_t ستون_تعداد_2 = MIN(Bc, nek1 - col0);

        برائے (uint32_t r = 0; r < n_صفیں_q; ++r) {
            ثابت __fp16 * صف_ptr = (ثابت __fp16 *) (m_buf + r * صف_stride) + col0;
            HVX_سمتیہ v_any = Q6_V_vzero();
            uint32_t c = 0;
            برائے (; c + 64 <= ستون_تعداد_2; c += 64) {
                ثابت HVX_سمتیہ v_m = *(ثابت HVX_سمتیہ *) (صف_ptr + c);
                v_any = Q6_V_vor_VV(v_any, Q6_V_vxor_VV(v_m, v_ninf));
            }
            اگر (c < ستون_تعداد_2) {
                ثابت uint32_t rem = ستون_تعداد_2 - c;
                ثابت HVX_سمتیہ_Pred q_rem = Q6_Q_vsetq2_R(rem * حجم(__fp16));
                ثابت HVX_سمتیہ v_m = *(ثابت HVX_سمتیہ *) (صف_ptr + c);
                ثابت HVX_سمتیہ v_diff = Q6_V_vand_QV(q_rem, Q6_V_vxor_VV(v_m, v_ninf));
                v_any = Q6_V_vor_VV(v_any, v_diff);
            }
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 64));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 32));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 16));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 8));
            v_any = Q6_V_vor_VV(v_any, Q6_V_vror_VR(v_any, 4));
            اگر (Q6_R_vextract_VR(v_any, 0) != 0) {
                فعال_بلاکیں[n_فعال++] = (uint16_t) b;
                توڑو;
            }
        }
    }

    اگر (n_فعال == 0 && n_kv_بلاکیں > 0) {
        فعال_بلاکیں[0] = 0;
        n_فعال = 1;
    }
    واپس n_فعال;
}

ساکن خالی flash_attn_ext_f16_thread(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_fa_context * factx = (ساخت htp_fa_context *) ڈیٹا;
    ثابت ساخت htp_ops_context * octx = factx->octx;
    ثابت ساخت htp_tensor * q     = octx->src[0];
    ثابت ساخت htp_tensor * k     = octx->src[1];
    ثابت ساخت htp_tensor * v     = octx->src[2];
    ثابت ساخت htp_tensor * نقاب  = octx->src[3];
    ثابت ساخت htp_tensor * sinks = octx->src[4];
    ثابت ساخت htp_tensor * dst   = octx->dst;

    ثابت uint32_t neq0 = q->ne[0];
    ثابت uint32_t neq1 = q->ne[1];
    ثابت uint32_t neq2 = q->ne[2];
    ثابت uint32_t neq3 = q->ne[3];

    ثابت uint32_t nek0 = k->ne[0];
    ثابت uint32_t nek1 = k->ne[1];
    ثابت uint32_t nek2 = k->ne[2];
    ثابت uint32_t nek3 = k->ne[3];

    ثابت uint32_t nev0 = v->ne[0];
    ثابت uint32_t nev1 = v->ne[1];
    ثابت uint32_t nev2 = v->ne[2];
    ثابت uint32_t nev3 = v->ne[3];

    ثابت uint32_t nbq1 = q->nb[1];
    ثابت uint32_t nbq2 = q->nb[2];
    ثابت uint32_t nbq3 = q->nb[3];

    ثابت uint32_t nbk1 = k->nb[1];
    ثابت uint32_t nbk2 = k->nb[2];
    ثابت uint32_t nbk3 = k->nb[3];

    ثابت uint32_t nbv1 = v->nb[1];
    ثابت uint32_t nbv2 = v->nb[2];
    ثابت uint32_t nbv3 = v->nb[3];

    ثابت uint32_t ne1 = dst->ne[1];
    ثابت uint32_t ne2 = dst->ne[2];
    ثابت uint32_t ne3 = dst->ne[3];

    ثابت uint32_t nb1 = dst->nb[1];
    ثابت uint32_t nb2 = dst->nb[2];
    ثابت uint32_t nb3 = dst->nb[3];

    // total rows in q
    ثابت uint32_t dr  = factx->qrows_فی_thread;
    ثابت uint32_t ir0 = factx->qrow_آغاز + dr * ith;
    ثابت uint32_t ir1 = MIN(ir0 + dr, factx->qrow_آغاز + factx->qrows);

    اگر (ir0 >= ir1) واپس;

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    dma_قطار * dma_q = octx->ctx->dma[ith];

    ثابت uint32_t DK = nek0;
    ثابت uint32_t DV = nev0;

    ثابت size_t سائز_q_صف = DK * ((q->type == HTP_TYPE_F32) ? 4 : 2);
    ثابت size_t سائز_k_صف = htp_tensor_get_صف_سائز(k->type, DK);
    ثابت size_t سائز_v_صف = htp_tensor_get_صف_سائز(v->type, DV);

    // Scratchpad buffers for Q, K, V, Mask, and VKQ32 accumulator
    uint8_t * spad_q = factx->spad_q + factx->سائز_q_بلاک * ith;
    uint8_t * spad_k = factx->spad_k + factx->سائز_k_بلاک * 2 * ith;
    uint8_t * spad_v = factx->spad_v + factx->سائز_v_بلاک * 2 * ith;
    uint8_t * spad_m = factx->spad_m + (نقاب ? factx->سائز_m_بلاک * HVX_FA_DMA_CACHE_سائز : 0) * ith;
    uint8_t * spad_a = factx->spad_a + factx->سائز_vkq_acc * ith;

    dma_cache_dm m_cache;
    dma_cache_dm_init(&m_cache, spad_m, factx->سائز_m_بلاک, HVX_FA_DMA_CACHE_سائز);

    uint16_t thread_فعال_بلاکیں[FA_MAX_KV_بلاکیں];
    ثابت uint16_t * فعال_بلاکیں = factx->shared_فعال_بلاکیں ? factx->فعال_بلاکیں : thread_فعال_بلاکیں;
    uint32_t n_فعال_بلاکیں       = factx->n_فعال_بلاکیں;
    dma_addr_t پچھلا_mp_بنیاد        = 0;

    اگر (!factx->shared_فعال_بلاکیں && !نقاب) {
        #ہدایت clang loop vectorize(disable)
        برائے (uint32_t b = 0; b < factx->n_بلاکیں; ++b) {
            thread_فعال_بلاکیں[b] = (uint16_t) b;
        }
    }

    ثابت size_t سائز_vkq_acc_single = hex_باری_اوپر(DV * حجم(اعشاری), 128);

    uint32_t ir = ir0;
    جب (ir < ir1) {
        ثابت uint32_t iq3 = fastdiv(ir, &factx->src0_div21);
        ثابت uint32_t iq2 = fastdiv(ir - iq3*neq2*neq1, &factx->src0_div1);
        ثابت uint32_t iq1 = (ir - iq3*neq2*neq1 - iq2 * neq1);

        ثابت uint32_t ik3 = fastdiv(iq3, &factx->نشر_rk3);
        ثابت uint32_t ik2 = fastdiv(iq2, &factx->نشر_rk2);

        ثابت uint32_t iv3 = fastdiv(iq3, &factx->نشر_rv3);
        ثابت uint32_t iv2 = fastdiv(iq2, &factx->نشر_rv2);

        uint32_t G_local = 1;
        اگر (neq1 == 1 && (نقاب == تہی || نقاب->ne[2] == 1)) {
            جب (ir + G_local < ir1 && G_local < FA_HVX_G_MAX) {
                ثابت uint32_t اگلا_ir = ir + G_local;
                ثابت uint32_t اگلا_iq3 = fastdiv(اگلا_ir, &factx->src0_div21);
                ثابت uint32_t اگلا_iq2 = fastdiv(اگلا_ir - اگلا_iq3*neq2*neq1, &factx->src0_div1);
                ثابت uint32_t اگلا_iq1 = (اگلا_ir - اگلا_iq3*neq2*neq1 - اگلا_iq2 * neq1);

                ثابت uint32_t اگلا_ik3 = fastdiv(اگلا_iq3, &factx->نشر_rk3);
                ثابت uint32_t اگلا_ik2 = fastdiv(اگلا_iq2, &factx->نشر_rk2);

                ثابت uint32_t اگلا_iv3 = fastdiv(اگلا_iq3, &factx->نشر_rv3);
                ثابت uint32_t اگلا_iv2 = fastdiv(اگلا_iq2, &factx->نشر_rv2);

                اگر (اگلا_ik2 != ik2 || اگلا_ik3 != ik3 || اگلا_iv2 != iv2 || اگلا_iv3 != iv3 || اگلا_iq1 != iq1 || اگلا_iq3 != iq3) {
                    توڑو;
                }
                G_local++;
            }
        }

        dma_addr_t mp_بنیاد = 0;
        اگر (نقاب) {
            ثابت uint32_t im2 = fastmodulo(iq2, نقاب->ne[2], &factx->src3_div2);
            ثابت uint32_t im3 = fastmodulo(iq3, نقاب->ne[3], &factx->src3_div3);
            mp_بنیاد = نقاب->ڈیٹا + iq1*نقاب->nb[1] + im2*نقاب->nb[2] + im3*نقاب->nb[3];
            اگر (!factx->shared_فعال_بلاکیں && mp_بنیاد != پچھلا_mp_بنیاد) {
                n_فعال_بلاکیں = fa_جائزہ_فعال_بلاکیں(
                    dma_q, نقاب, iq1, 1, im2, im3, nek1, FLASH_ATTN_بلاک_سائز,
                    factx->n_بلاکیں, spad_k, factx->سائز_k_بلاک * 2, thread_فعال_بلاکیں);
                پچھلا_mp_بنیاد = mp_بنیاد;
            }
        }

        uint32_t heads[FA_HVX_G_MAX];
        HVX_سمتیہ ڈھلان_vecs[FA_HVX_G_MAX] __attribute__((aligned(128)));
        HVX_سمتیہ S_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        HVX_سمتیہ M_vec[FA_HVX_G_MAX]      __attribute__((aligned(128)));
        uint8_t * q_ptrs[FA_HVX_G_MAX];
        اعشاری * vkq_ptrs[FA_HVX_G_MAX];

        برائے (uint32_t g = 0; g < G_local; ++g) {
            ثابت uint32_t r = ir + g;
            ثابت uint32_t r_iq3 = fastdiv(r, &factx->src0_div21);
            ثابت uint32_t r_iq2 = fastdiv(r - r_iq3*neq2*neq1, &factx->src0_div1);
            ثابت uint32_t r_iq1 = (r - r_iq3*neq2*neq1 - r_iq2 * neq1);

            heads[g] = r_iq2;
            ثابت __fp16 ڈھلان = factx->slopes[r_iq2];
            ڈھلان_vecs[g] = hvx_vec_splat_f16(ڈھلان);

            S_vec[g] = hvx_vec_splat_f32(0.0f);
            M_vec[g] = hvx_vec_splat_f32(HTP_FA_M_INITIAL_VAL);

            uint8_t * q_dst = spad_q + g * factx->سائز_q_صف_padded;
            q_ptrs[g] = q_dst;

            اعشاری * vkq_dst = (اعشاری *)(spad_a + g * سائز_vkq_acc_single);
            vkq_ptrs[g] = vkq_dst;
            hvx_splat_f32_a((uint8_t *) vkq_dst, 0, DV);

            // Fetch Q row g
            ثابت dma_addr_t q_صف_ptr = q->ڈیٹا + r_iq1*nbq1 + r_iq2*nbq2 + r_iq3*nbq3;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(q_dst, q_صف_ptr), factx->سائز_q_صف_padded, nbq1, سائز_q_صف, 1);
        }

        // Prefetch first two active blocks
        برائے (uint32_t قدم = 0; قدم < MIN(n_فعال_بلاکیں, 2); ++قدم) {
            ثابت uint32_t ib = فعال_بلاکیں[قدم];
            ثابت uint32_t ic_آغاز = ib * FLASH_ATTN_بلاک_سائز;
            ثابت uint32_t current_بلاک_سائز = MIN(FLASH_ATTN_بلاک_سائز, nek1 - ic_آغاز);

            // K
            ثابت dma_addr_t k_src = k->ڈیٹا + ic_آغاز*nbk1 + ik2*nbk2 + ik3*nbk3;
            uint8_t * k_dst = spad_k + (قدم % 2) * factx->سائز_k_بلاک;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(k_dst, k_src), factx->سائز_k_صف_padded, nbk1, سائز_k_صف, current_بلاک_سائز);

            // V
            ثابت dma_addr_t v_src = v->ڈیٹا + ic_آغاز*nbv1 + iv2*nbv2 + iv3*nbv3;
            uint8_t * v_dst = spad_v + (قدم % 2) * factx->سائز_v_بلاک;
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(v_dst, v_src), factx->سائز_v_صف_padded, nbv1, سائز_v_صف, current_بلاک_سائز);

            // Mask
            اگر (نقاب) {
                ثابت dma_addr_t m_src = mp_بنیاد + ic_آغاز * حجم(__fp16);
                dma_cache_dm_ڈالو(dma_q, &m_cache, ib, m_src, current_بلاک_سائز * 2, current_بلاک_سائز * 2, current_بلاک_سائز * 2, 1);
            }
        }

        // Pop all Q rows
        برائے (uint32_t g = 0; g < G_local; ++g) {
            uint8_t * q_ptr_vtcm = (خالی *) dma_قطار_نکالو(dma_q).dst;
            اگر (factx->is_q_fp32) {
                hvx_copy_f16_f32_aa(q_ptr_vtcm, q_ptr_vtcm, DK);
            }
        }

        ثابت HVX_سمتیہ v_neg_inf = Q6_Vh_vsplat_R(0xfbff);
        ثابت bool has_softcap     = factx->has_softcap;
        ثابت HVX_سمتیہ v_cap     = has_softcap ? hvx_vec_splat_f16(factx->logit_softcap) : Q6_V_vzero();
        ثابت HVX_سمتیہ vinf      = Q6_Vh_vsplat_R(0xFC00);
        ثابت HVX_سمتیہ vmin      = Q6_Vh_vsplat_R(0xFBFF);
        ثابت HVX_سمتیہ v_log2e   = hvx_vec_splat_f16(EXP_LOG2E_F);
        ثابت uint32_t stride_v2   = factx->سائز_v_صف_padded * 2;

        برائے (uint32_t قدم = 0; قدم < n_فعال_بلاکیں; ++قدم) {
            ثابت uint32_t ib = فعال_بلاکیں[قدم];
            ثابت uint32_t ic_آغاز = ib * FLASH_ATTN_بلاک_سائز;
            ثابت uint32_t current_بلاک_سائز = MIN(FLASH_ATTN_بلاک_سائز, nek1 - ic_آغاز);

            // Wait for DMA
            uint8_t * k_بنیاد = (خالی *) dma_قطار_نکالو(dma_q).dst; // K
            uint8_t * v_بنیاد = (خالی *) dma_قطار_نکالو(dma_q).dst; // V
            __fp16  * m_بنیاد = نقاب ? (__fp16 *) dma_قطار_نکالو(dma_q).dst : تہی; // M

            اگر (factx->k->type == HTP_TYPE_Q8_0) {
                htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
                برائے (uint32_t r = 0; r < current_بلاک_سائز; ++r) {
                    __fp16 * صف_k = (__fp16 *)(k_بنیاد + r * factx->سائز_k_صف_padded);
                    hvx_dequantize_صف_flat_q8_0_f16(صف_k, صف_k, DK);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, ir);
            }
            اگر (factx->v->type == HTP_TYPE_Q8_0) {
                htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
                برائے (uint32_t r = 0; r < current_بلاک_سائز; ++r) {
                    __fp16 * صف_v = (__fp16 *)(v_بنیاد + r * factx->سائز_v_صف_padded);
                    hvx_dequantize_صف_flat_q8_0_f16(صف_v, صف_v, DV);
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, ir);
            }

            برائے (uint32_t g = 0; g < G_local; ++g) {
                ثابت uint32_t سر_ir = ir + g;
                uint8_t * q_ptr_vtcm = q_ptrs[g];
                اعشاری * VKQ32 = vkq_ptrs[g];
                ثابت HVX_سمتیہ ڈھلان_vec = ڈھلان_vecs[g];

                htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_FA_QK, سر_ir);

                HVX_سمتیہ scores_f16 = Q6_V_vzero();
                اگر (current_بلاک_سائز > 0) {
                    HVX_سمتیہ scores0 = hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_بنیاد, factx->سائز_k_صف_padded, DK, factx->scale);
                    HVX_سمتیہ scores1 = (current_بلاک_سائز > 32) ? hvx_dot_f16_f16_aa_rx32(q_ptr_vtcm, k_بنیاد + 32 * factx->سائز_k_صف_padded, factx->سائز_k_صف_padded, DK, factx->scale) : Q6_V_vzero();
                    scores_f16 = hvx_vec_f32_مقصد_f16(scores0, scores1);
                }

                اگر (has_softcap) {
                    scores_f16 = hvx_vec_tanh_f16(scores_f16);
                    scores_f16 = hvx_vec_mul_f16_f16(scores_f16, v_cap);
                }

                HVX_سمتیہ_Pred q_دم_keep = Q6_Q_vsetq2_R(current_بلاک_سائز * حجم(__fp16));

                اگر (نقاب) {
                    HVX_سمتیہ m_vals_f16 = *(ثابت HVX_UVector *) m_بنیاد;
                    HVX_سمتیہ_Pred is_inf = Q6_Q_vcmp_eq_VhVh(m_vals_f16, vinf);
                    m_vals_f16 = Q6_V_vmux_QVV(is_inf, vmin, m_vals_f16);

                    HVX_سمتیہ m_scaled = hvx_vec_mul_f16_f16(m_vals_f16, ڈھلان_vec);
                    scores_f16 = Q6_V_vmux_QVV(q_دم_keep, hvx_vec_جوڑو_f16_f16(scores_f16, m_scaled), v_neg_inf);
                } ورنہ {
                    scores_f16 = Q6_V_vmux_QVV(q_دم_keep, scores_f16, v_neg_inf);
                }

                HVX_سمتیہ v_max_f16 = hvx_vec_reduce_max_f16(scores_f16);
                HVX_سمتیہ v_max     = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(v_max_f16));
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_QK, سر_ir);

                // prefetch K for step + 2 after last head finished QK
                اگر (g + 1 == G_local && قدم + 2 < n_فعال_بلاکیں) {
                    ثابت uint32_t اگلا_ib = فعال_بلاکیں[قدم + 2];
                    ثابت uint32_t اگلا_ic_آغاز = اگلا_ib * FLASH_ATTN_بلاک_سائز;
                    ثابت uint32_t اگلا_بلاک_سائز = MIN(FLASH_ATTN_بلاک_سائز, nek1 - اگلا_ic_آغاز);

                    ثابت dma_addr_t k_src = k->ڈیٹا + اگلا_ic_آغاز*nbk1 + ik2*nbk2 + ik3*nbk3;
                    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(k_بنیاد, k_src), factx->سائز_k_صف_padded, nbk1, سائز_k_صف, اگلا_بلاک_سائز);
                }

                htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_FA_SFM, سر_ir);
                {
                    HVX_سمتیہ M_new_vec = Q6_Vsf_vmax_VsfVsf(v_max, M_vec[g]);
                    HVX_سمتیہ diff_vec  = HVX_OP_SUB_F32(M_vec[g], M_new_vec);

                    HVX_سمتیہ diff_f16   = hvx_vec_f32_مقصد_f16(diff_vec, diff_vec);
                    HVX_سمتیہ diff_base2 = hvx_vec_mul_f16_f16(diff_f16, v_log2e);
                    HVX_سمتیہ ms_f16     = hvx_vec_exp2_f16(diff_base2);
                    HVX_سمتیہ ms_vec     = Q6_V_lo_W(hvx_vec_f16_مقصد_f32(ms_f16));

                    M_vec[g] = M_new_vec;

                    HVX_سمتیہ v_m_vec_f16 = hvx_vec_f32_مقصد_f16(M_vec[g], M_vec[g]);
                    HVX_سمتیہ v_s_minus_m = Q6_Vqf16_vsub_VhfVhf(scores_f16, v_m_vec_f16);
                    HVX_سمتیہ v_s_minus_m_base2 = hvx_vec_mul_f16_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m), v_log2e);

                    HVX_سمتیہ P = hvx_vec_exp2_f16(v_s_minus_m_base2);
                    P = Q6_V_vmux_QVV(q_دم_keep, P, Q6_V_vzero());

                    HVX_سمتیہ_Pair P_pair = hvx_vec_f16_مقصد_f32(P);
                    HVX_سمتیہ P0 = Q6_V_lo_W(P_pair);
                    HVX_سمتیہ P1 = Q6_V_hi_W(P_pair);
                    HVX_سمتیہ p_مجموع_vec = hvx_vec_reduce_مجموع_f32(HVX_OP_جوڑو_F32(P0, P1));

                    S_vec[g] = HVX_OP_جوڑو_F32(HVX_OP_MUL_F32(S_vec[g], ms_vec), p_مجموع_vec);

                    ثابت uint8_t * v_ptr = v_بنیاد;

                    اگر (DV == 64) {
                        HVX_سمتیہ_Pair vkq0 = *((ثابت HVX_سمتیہ_Pair *) VKQ32);
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );

                        برائے (uint32_t j = 0; j < current_بلاک_سائز; j += 2) {
                            HVX_سمتیہ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ثابت HVX_سمتیہ * vx0 = (ثابت HVX_سمتیہ *) v_ptr;
                            اگر (j + 1 == current_بلاک_سائز) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                توڑو;
                            }

                            HVX_سمتیہ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ثابت HVX_سمتیہ * vx1 = (ثابت HVX_سمتیہ *) (v_ptr + factx->سائز_v_صف_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            v_ptr += stride_v2;
                        }

                        *((HVX_سمتیہ_Pair *) VKQ32) = vkq0;
                    } ورنہ اگر (DV == 128) {
                        HVX_سمتیہ_Pair vkq0 = ((ثابت HVX_سمتیہ_Pair *) VKQ32)[0];
                        HVX_سمتیہ_Pair vkq1 = ((ثابت HVX_سمتیہ_Pair *) VKQ32)[1];
                        vkq0 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq0), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq0), ms_vec)
                        );
                        vkq1 = Q6_W_vcombine_VV(
                            HVX_OP_MUL_F32(Q6_V_hi_W(vkq1), ms_vec),
                            HVX_OP_MUL_F32(Q6_V_lo_W(vkq1), ms_vec)
                        );

                        برائے (uint32_t j = 0; j < current_بلاک_سائز; j += 2) {
                            HVX_سمتیہ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ثابت HVX_سمتیہ * vx0 = (ثابت HVX_سمتیہ *) v_ptr;
                            اگر (j + 1 == current_بلاک_سائز) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                توڑو;
                            }

                            HVX_سمتیہ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ثابت HVX_سمتیہ * vx1 = (ثابت HVX_سمتیہ *) (v_ptr + factx->سائز_v_صف_padded);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                            vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx1[0]), S1);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                            vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx1[1]), S1);
                            v_ptr += stride_v2;
                        }

                        ((HVX_سمتیہ_Pair *) VKQ32)[0] = vkq0;
                        ((HVX_سمتیہ_Pair *) VKQ32)[1] = vkq1;
                    } ورنہ اگر (DV == 256) {
                        HVX_سمتیہ_Pair vkq0 = ((ثابت HVX_سمتیہ_Pair *) VKQ32)[0];
                        HVX_سمتیہ_Pair vkq1 = ((ثابت HVX_سمتیہ_Pair *) VKQ32)[1];
                        HVX_سمتیہ_Pair vkq2 = ((ثابت HVX_سمتیہ_Pair *) VKQ32)[2];
                        HVX_سمتیہ_Pair vkq3 = ((ثابت HVX_سمتیہ_Pair *) VKQ32)[3];
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

                        برائے (uint32_t j = 0; j < current_بلاک_سائز; j += 2) {
                            HVX_سمتیہ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            ثابت HVX_سمتیہ * vx0 = (ثابت HVX_سمتیہ *) v_ptr;
                            اگر (j + 1 == current_بلاک_سائز) {
                                vkq0 = hvx_vec_mpyacc_f32_f16(vkq0, Q6_Vh_vshuff_Vh(vx0[0]), S0);
                                vkq1 = hvx_vec_mpyacc_f32_f16(vkq1, Q6_Vh_vshuff_Vh(vx0[1]), S0);
                                vkq2 = hvx_vec_mpyacc_f32_f16(vkq2, Q6_Vh_vshuff_Vh(vx0[2]), S0);
                                vkq3 = hvx_vec_mpyacc_f32_f16(vkq3, Q6_Vh_vshuff_Vh(vx0[3]), S0);
                                توڑو;
                            }

                            HVX_سمتیہ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));
                            ثابت HVX_سمتیہ * vx1 = (ثابت HVX_سمتیہ *) (v_ptr + factx->سائز_v_صف_padded);
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

                        ((HVX_سمتیہ_Pair *) VKQ32)[0] = vkq0;
                        ((HVX_سمتیہ_Pair *) VKQ32)[1] = vkq1;
                        ((HVX_سمتیہ_Pair *) VKQ32)[2] = vkq2;
                        ((HVX_سمتیہ_Pair *) VKQ32)[3] = vkq3;
                    } ورنہ {
                        hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (ثابت uint8_t *) VKQ32, DV, ms_vec);

                        برائے (uint32_t j = 0; j < current_بلاک_سائز; j += 2) {
                            اگر (j + 1 == current_بلاک_سائز) {
                                HVX_سمتیہ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                                hvx_mad_f32_f16_aa_vec(VKQ32, v_ptr, S0, DV);
                                توڑو;
                            }

                            HVX_سمتیہ S0 = hvx_vec_repl_f16(Q6_V_vror_VR(P, j * 2));
                            HVX_سمتیہ S1 = hvx_vec_repl_f16(Q6_V_vror_VR(P, (j + 1) * 2));

                            hvx_mad_f32_f16_aa_rx2_vec(VKQ32, v_ptr, v_ptr + factx->سائز_v_صف_padded, S0, S1, DV);
                            v_ptr += stride_v2;
                        }
                    }
                }
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, سر_ir);

                // prefetch V and mask for step + 2 after last head finished V accumulation
                اگر (g + 1 == G_local && قدم + 2 < n_فعال_بلاکیں) {
                    ثابت uint32_t اگلا_ib = فعال_بلاکیں[قدم + 2];
                    ثابت uint32_t اگلا_ic_آغاز = اگلا_ib * FLASH_ATTN_بلاک_سائز;
                    ثابت uint32_t اگلا_بلاک_سائز = MIN(FLASH_ATTN_بلاک_سائز, nek1 - اگلا_ic_آغاز);

                    // V
                    ثابت dma_addr_t v_src = v->ڈیٹا + اگلا_ic_آغاز*nbv1 + iv2*nbv2 + iv3*nbv3;
                    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(v_بنیاد, v_src), factx->سائز_v_صف_padded, nbv1, سائز_v_صف, اگلا_بلاک_سائز);

                    // Mask
                    اگر (نقاب) {
                        ثابت dma_addr_t m_src = mp_بنیاد + اگلا_ic_آغاز * حجم(__fp16);
                        dma_cache_dm_ڈالو(dma_q, &m_cache, اگلا_ib, m_src, اگلا_بلاک_سائز * 2, اگلا_بلاک_سائز * 2, اگلا_بلاک_سائز * 2, 1);
                    }
                }
            } // end for g
        } // end for step

        برائے (uint32_t g = 0; g < G_local; ++g) {
            ثابت uint32_t سر_ir = ir + g;
            ثابت uint32_t h = heads[g];
            اعشاری * VKQ32 = vkq_ptrs[g];

            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_O_PROC, سر_ir);

            اعشاری M = hvx_vec_get_f32(M_vec[g]);
            اعشاری S = hvx_vec_get_f32(S_vec[g]);

            اگر (sinks) {
                ثابت اعشاری s = factx->spad_sinks[h];

                اعشاری vs = 1.0f;

                اگر (s > M) {
                    HVX_سمتیہ diff_vec = hvx_vec_splat_f32(M - s);
                    HVX_سمتیہ ms_vec   = hvx_vec_exp_f32(diff_vec);
                    hvx_scale_vec_f32_aa((uint8_t *) VKQ32, (ثابت uint8_t *) VKQ32, DV, ms_vec);

                    اعشاری ms = hvx_vec_get_f32(ms_vec);
                    S = S * ms + vs;
                } ورنہ {
                    HVX_سمتیہ diff_vec = hvx_vec_splat_f32(s - M);
                    vs = hvx_vec_get_f32(hvx_vec_exp_f32(diff_vec));
                    S += vs;
                }
            }

            ثابت اعشاری S_inv = S == 0.0f ? 0.0f : 1.0f/S;
            hvx_scale_f32_aa((uint8_t *) VKQ32, (ثابت uint8_t *) VKQ32, DV, S_inv);

            ثابت uint32_t r_iq3 = fastdiv(سر_ir, &factx->src0_div21);
            ثابت uint32_t r_iq2 = fastdiv(سر_ir - r_iq3*neq2*neq1, &factx->src0_div1);
            ثابت uint32_t r_iq1 = (سر_ir - r_iq3*neq2*neq1 - r_iq2 * neq1);

            uint8_t * dst_ptr = (uint8_t *) dst->ڈیٹا + r_iq2 * dst->nb[1] + r_iq1 * dst->nb[2] + r_iq3 * dst->nb[3];

            اگر (dst->type == HTP_TYPE_F32) {
                hvx_copy_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
            } ورنہ اگر (dst->type == HTP_TYPE_F16) {
                hvx_copy_f16_f32_ua(dst_ptr, (uint8_t *) VKQ32, DV);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, سر_ir);
        }

        ir += G_local;
    }
}

// ============================================================================
// HMX Phase args and thread logic
// ============================================================================

قسمتعریف ساخت {
    ساخت hmx_fa_context * factx;
    uint32_t                kv_صفیں;
    size_t                  src_stride;
    خالی *                  curr_k;
    uint32_t                kv_آغاز;
    uint32_t                صفیں_فی_t;
    size_t                  buf_idx;
} fa_k_int_دلائل_t;

ساکن خالی fa_k_interleave_thread(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    fa_k_int_دلائل_t *       دلائل_2  = (fa_k_int_دلائل_t *) ڈیٹا;
    ساخت hmx_fa_context * factx = دلائل_2->factx;

    ثابت uint32_t کل_صفیں = دلائل_2->kv_صفیں;
    ثابت uint32_t صفیں_فی_t = دلائل_2->صفیں_فی_t;
    ثابت uint32_t آغاز      = i * صفیں_فی_t;
    ثابت uint32_t انتہا        = (uint32_t) hex_smin(آغاز + صفیں_فی_t, کل_صفیں);

    اگر (آغاز >= کل_صفیں) {
        واپس;
    }

    ساخت htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (دلائل_2->kv_آغاز + آغاز));
    اگر (factx->k->type == HTP_TYPE_Q8_0) {
        برائے (uint32_t r = آغاز; r < انتہا; ++r) {
            __fp16 * صف_k = (__fp16 *)((حرف *)دلائل_2->curr_k + r * دلائل_2->src_stride * حجم(__fp16));
            hvx_dequantize_صف_flat_q8_0_f16(صف_k, صف_k, factx->DK);
        }
    }
    hmx_interleave_صفیں_مقصد_tiles(factx->vtcm_k_tiles[دلائل_2->buf_idx], (ثابت __fp16 *) دلائل_2->curr_k, کل_صفیں, factx->DK_pad,
                             دلائل_2->src_stride, آغاز, انتہا);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_K_PREP, (uint16_t) (دلائل_2->kv_آغاز + آغاز));
}

ساکن خالی fa_phase_k_interleave(ساخت hmx_fa_context * factx, uint32_t kv_صفیں, size_t src_stride, خالی * curr_k, uint32_t kv_آغاز, size_t buf_idx) {
    work_قطار_t wp = factx->octx->ctx->work_قطار;
    uint32_t n = 1;
    اگر (factx->n_دھاگے > 1 && kv_صفیں >= factx->n_دھاگے * 2) {
        n = factx->n_دھاگے;
    }
    uint32_t صفیں_فی_t = hex_align_اوپر(hmx_ceil_div(kv_صفیں, n), 2);
    fa_k_int_دلائل_t دلائل_2 = { factx, kv_صفیں, src_stride, curr_k, kv_آغاز, صفیں_فی_t, buf_idx };
    اگر (n > 1) {
        work_قطار_تسلسل(wp, fa_k_interleave_thread, &دلائل_2, n);
    } ورنہ {
        fa_k_interleave_thread(1, 0, &دلائل_2);
    }
}

قسمتعریف ساخت {
    ساخت hmx_fa_context * factx;
    uint32_t                kv_صفیں;
    size_t                  src_stride;
    خالی *                  v_src;
    خالی *                  v_tiles_dst;
    size_t                  n_ستون_tiles;
    uint32_t                kv_آغاز;
    uint32_t                صفیں_فی_t;
} fa_v_int_دلائل_t;

ساکن خالی fa_v_interleave_thread(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    fa_v_int_دلائل_t *       دلائل_2  = (fa_v_int_دلائل_t *) ڈیٹا;
    ساخت hmx_fa_context * factx = دلائل_2->factx;

    ثابت uint32_t کل_صفیں = دلائل_2->kv_صفیں;
    ثابت uint32_t صفیں_فی_t = دلائل_2->صفیں_فی_t;
    ثابت uint32_t آغاز      = i * صفیں_فی_t;
    ثابت uint32_t انتہا        = (uint32_t) hex_smin(آغاز + صفیں_فی_t, کل_صفیں);

    اگر (آغاز >= کل_صفیں) {
        واپس;
    }

    __fp16 * v_tiles_dst = (__fp16 *) دلائل_2->v_tiles_dst;

    ساخت htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (دلائل_2->kv_آغاز + آغاز));
    اگر (factx->v->type == HTP_TYPE_Q8_0) {
        برائے (uint32_t r = آغاز; r < انتہا; ++r) {
            __fp16 * صف_v = (__fp16 *)((حرف *)دلائل_2->v_src + r * دلائل_2->src_stride * حجم(__fp16));
            hvx_dequantize_صف_flat_q8_0_f16(صف_v, صف_v, factx->DV);
        }
    }
    hmx_interleave_ستون_تعداد_مقصد_tiles(v_tiles_dst, (ثابت __fp16 *) دلائل_2->v_src, کل_صفیں, factx->DV_pad,
                             دلائل_2->src_stride, (uint32_t) دلائل_2->n_ستون_tiles, آغاز, انتہا);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_V_PREP, (uint16_t) (دلائل_2->kv_آغاز + آغاز));
}

ساکن خالی fa_phase_v_interleave(ساخت hmx_fa_context * factx,
                                  uint32_t                kv_صفیں,
                                  size_t                  src_stride,
                                  خالی *                  v_src,
                                  خالی *                  v_tiles_dst,
                                  size_t                  n_ستون_tiles,
                                  uint32_t                kv_آغاز) {
    work_قطار_t wp = factx->octx->ctx->work_قطار;
    uint32_t n = 1;
    اگر (factx->n_دھاگے > 1 && kv_صفیں >= factx->n_دھاگے * 2) {
        n = factx->n_دھاگے;
    }
    uint32_t صفیں_فی_t = hex_align_اوپر(hmx_ceil_div(kv_صفیں, n), 2);
    fa_v_int_دلائل_t دلائل_2 = { factx, kv_صفیں, src_stride, v_src, v_tiles_dst, n_ستون_tiles, kv_آغاز, صفیں_فی_t };
    اگر (n > 1) {
        work_قطار_تسلسل(wp, fa_v_interleave_thread, &دلائل_2, n);
    } ورنہ {
        fa_v_interleave_thread(1, 0, &دلائل_2);
    }
}

قسمتعریف ساخت {
    ساخت hmx_fa_context *   factx;
    ثابت ساخت htp_tensor * q;
    uint32_t                  q_آغاز;
    uint32_t                  kv_سر;
    uint32_t                  ib3;
    size_t                    n_صفیں_g;
    size_t                    صفیں_فی_t;
    size_t                    n_صفیں_q;
    bool                      q_transposed;
    atomic_uint               barrier;
} fa_q_بھرو_دلائل_t;

ساکن خالی fa_q_بھرو_thread(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    fa_q_بھرو_دلائل_t *      دلائل_2  = (fa_q_بھرو_دلائل_t *) ڈیٹا;
    ساخت hmx_fa_context * factx = دلائل_2->factx;

    ثابت size_t n_صفیں_g = دلائل_2->n_صفیں_g;
    ثابت size_t G        = factx->G;
    ثابت size_t DK       = factx->DK;

    // Partition the padded Q rows (g_br) across threads.
    // Keep start/end even so r and r+1 are always in the same thread's range.
    ثابت size_t صفیں_فی_t = دلائل_2->صفیں_فی_t;
    ثابت size_t آغاز      = (size_t) i * صفیں_فی_t;
    ثابت size_t انتہا        = hex_smin(آغاز + صفیں_فی_t, factx->g_br);

    ساخت htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (دلائل_2->q_آغاز * G + آغاز));

    // Parallel initialization of per-block state
    {
        ثابت uint32_t g_br = factx->g_br;
        ثابت uint32_t DV   = factx->DV;

        ثابت size_t ستون_vec_بائٹیں = factx->ستون_vec_بائٹیں;
        ثابت size_t d_tile_بائٹیں  = factx->d_tile_بائٹیں;

        // Initialize vtcm_l_vec & vtcm_m_vec
        ثابت size_t l_بائٹیں_فی_t = hex_align_اوپر(ستون_vec_بائٹیں / n, 128);
        ثابت size_t l_آغاز       = i * l_بائٹیں_فی_t;
        ثابت size_t l_انتہا         = hex_smin(l_آغاز + l_بائٹیں_فی_t, ستون_vec_بائٹیں);

        ثابت size_t m_بائٹیں_فی_t = hex_align_اوپر(ستون_vec_بائٹیں / n, 128);
        ثابت size_t m_آغاز       = i * m_بائٹیں_فی_t;
        ثابت size_t m_انتہا         = hex_smin(m_آغاز + m_بائٹیں_فی_t, ستون_vec_بائٹیں);

        اگر (l_آغاز < ستون_vec_بائٹیں) {
            hvx_splat_u8_a((حرف *) factx->vtcm_l_vec + l_آغاز, 0, l_انتہا - l_آغاز);
        }
        اگر (m_آغاز < ستون_vec_بائٹیں) {
            hvx_splat_f32_a((حرف *) factx->vtcm_m_vec + m_آغاز, HTP_FA_M_INITIAL_VAL, (m_انتہا - m_آغاز) / حجم(اعشاری));
        }

        // Zero the whole rescale region: vtcm_d_tiles[0], the optional vtcm_d_tiles[1]
        // and vtcm_d_inv_l are equal-sized and allocated back to back, so one run covers
        // them all.  The scatter only ever writes the diagonal, ignore the rest.
        ثابت size_t d_بائٹیں_فی_t = hex_align_اوپر(d_tile_بائٹیں / n, 128);
        ثابت size_t d_آغاز       = i * d_بائٹیں_فی_t;
        ثابت size_t d_انتہا         = hex_smin(d_آغاز + d_بائٹیں_فی_t, d_tile_بائٹیں);
        اگر (d_آغاز < d_tile_بائٹیں) {
            hvx_splat_u8_a((حرف *) factx->vtcm_d_tiles[0] + d_آغاز, 0, d_انتہا - d_آغاز);
        }
    }

    اگر (آغاز < factx->g_br) {
        ثابت ساخت htp_tensor * q       = دلائل_2->q;
        ثابت uint32_t            q_آغاز = دلائل_2->q_آغاز;
        ثابت uint32_t            kv_سر = دلائل_2->kv_سر;
        ثابت uint32_t            ib3     = دلائل_2->ib3;

        ثابت bool use_q_dma = (factx->vtcm_q_dma != تہی);

        __fp16 * q_tiles = factx->vtcm_q_tiles;
        ثابت size_t DK_pad = factx->DK_pad;
        اگر (use_q_dma) {
            ثابت size_t g_صفیں_انتہا = hex_smin(انتہا, n_صفیں_g);
            ثابت uint32_t d_حد_بندی = factx->is_q_fp32 ? DK / 32 : DK / 64;

            uint8_t * q_flat  = (uint8_t *) factx->vtcm_q_dma;
            اگر (DK_pad != DK) {
                اگر (factx->is_q_fp32) {
                    hmx_fa_q_prep_fp32_pad(q_tiles, q_flat, آغاز, انتہا, g_صفیں_انتہا, DK, DK_pad, G, دلائل_2->n_صفیں_q, &factx->div_G, دلائل_2->q_transposed);
                } ورنہ {
                    hmx_fa_q_prep_fp16_pad(q_tiles, q_flat, آغاز, انتہا, g_صفیں_انتہا, DK, DK_pad, G, دلائل_2->n_صفیں_q, &factx->div_G, دلائل_2->q_transposed);
                }
            } ورنہ اگر (factx->is_q_fp32) {
                انتخاب (d_حد_بندی) {
                حالت 2:  hmx_fa_q_prep_fp32_d2(q_tiles, q_flat, آغاز, انتہا, g_صفیں_انتہا, DK, G, دلائل_2->n_صفیں_q, &factx->div_G, دلائل_2->q_transposed); توڑو;
                حالت 4:  hmx_fa_q_prep_fp32_d4(q_tiles, q_flat, آغاز, انتہا, g_صفیں_انتہا, DK, G, دلائل_2->n_صفیں_q, &factx->div_G, دلائل_2->q_transposed); توڑو;
                طےشدہ: hmx_fa_q_prep_fp32(   q_tiles, q_flat, آغاز, انتہا, g_صفیں_انتہا, DK, G, دلائل_2->n_صفیں_q, &factx->div_G, d_حد_بندی, دلائل_2->q_transposed); توڑو;
                }
            } ورنہ {
                انتخاب (d_حد_بندی) {
                حالت 1:  hmx_fa_q_prep_fp16_d1(q_tiles, q_flat, آغاز, انتہا, g_صفیں_انتہا, DK, G, دلائل_2->n_صفیں_q, &factx->div_G, دلائل_2->q_transposed); توڑو;
                حالت 2:  hmx_fa_q_prep_fp16_d2(q_tiles, q_flat, آغاز, انتہا, g_صفیں_انتہا, DK, G, دلائل_2->n_صفیں_q, &factx->div_G, دلائل_2->q_transposed); توڑو;
                طےشدہ: hmx_fa_q_prep_fp16(   q_tiles, q_flat, آغاز, انتہا, g_صفیں_انتہا, DK, G, دلائل_2->n_صفیں_q, &factx->div_G, d_حد_بندی, دلائل_2->q_transposed); توڑو;
                }
            }
        } ورنہ {
            // Fallback: direct-from-DDR/L2 path
            hmx_fa_q_prep_fallback(q_tiles, q->ڈیٹا, q->nb[1], q->nb[2], q->nb[3],
                                   q_آغاز, kv_سر, ib3, آغاز, انتہا, n_صفیں_g, G, DK, DK_pad, factx->is_q_fp32, &factx->div_G);
        }
    }

    // Synchronize threads before zeroing out vtcm_o_tiles[0] to prevent race condition
    اگر (n > 1) {
        atomic_fetch_sub(&دلائل_2->barrier, 1);
        جب (atomic_بھرو(&دلائل_2->barrier) > 0) {
            // spin wait
        }
    }

    // Zero out vtcm_o_tiles[0] as it was used as temp_q_vtcm
    {
        ثابت uint32_t g_br = factx->g_br;
        ثابت uint32_t DV   = factx->DV;
        ثابت size_t o_tile_بائٹیں  = factx->o_tile_بائٹیں;
        ثابت size_t o_بائٹیں_فی_t = hex_align_اوپر(o_tile_بائٹیں / n, 128);
        ثابت size_t o_آغاز       = i * o_بائٹیں_فی_t;
        ثابت size_t o_انتہا         = hex_smin(o_آغاز + o_بائٹیں_فی_t, o_tile_بائٹیں);
        اگر (o_آغاز < o_tile_بائٹیں) {
            hvx_splat_u8_a((حرف *) factx->vtcm_o_tiles[0] + o_آغاز, 0, o_انتہا - o_آغاز);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_Q_PREP, (uint16_t) (دلائل_2->q_آغاز * G + آغاز));
}

ساکن خالی fa_phase_q_بھرو(ساخت hmx_fa_context *   factx,
                            ثابت ساخت htp_tensor * q,
                            uint32_t                  q_آغاز,
                            uint32_t                  kv_سر,
                            uint32_t                  ib3,
                            size_t                    n_صفیں_g) {
    work_قطار_t wp = factx->octx->ctx->work_قطار;
    uint32_t n = 1;
    اگر (factx->n_دھاگے > 1 && n_صفیں_g >= (size_t) (factx->n_دھاگے * 2)) {
        n = factx->n_دھاگے;
    }
    size_t صفیں_فی_t = hex_align_اوپر(hmx_ceil_div(factx->g_br, n), 2);
    ثابت uint32_t n_صفیں_q = hex_smin(factx->Br, factx->neq1 - q_آغاز);
    fa_q_بھرو_دلائل_t دلائل_2;
    دلائل_2.factx = factx;
    دلائل_2.q = q;
    دلائل_2.q_آغاز = q_آغاز;
    دلائل_2.kv_سر = kv_سر;
    دلائل_2.ib3 = ib3;
    دلائل_2.n_صفیں_g = n_صفیں_g;
    دلائل_2.صفیں_فی_t = صفیں_فی_t;
    دلائل_2.n_صفیں_q = n_صفیں_q;
    دلائل_2.q_transposed = q->nb[1] < q->nb[2];
    atomic_init(&دلائل_2.barrier, n);
    اگر (n > 1) {
        work_قطار_تسلسل(wp, fa_q_بھرو_thread, &دلائل_2, n);
    } ورنہ {
        fa_q_بھرو_thread(1, 0, &دلائل_2);
    }
}

قسمتعریف ساخت {
    ساخت hmx_fa_context *   factx;
    ثابت ساخت htp_tensor * dst;
    ثابت __fp16 *            o_tile_src;
    uint32_t                  q_آغاز;
    uint32_t                  kv_سر;
    uint32_t                  ib3;
    size_t                    n_صفیں_g;
    size_t                    صفیں_فی_t;
} fa_o_store_دلائل_t;

ساکن خالی fa_o_store_thread_f32(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    fa_o_store_دلائل_t *     دلائل_2  = (fa_o_store_دلائل_t *) ڈیٹا;
    ساخت hmx_fa_context * factx = دلائل_2->factx;

    ثابت size_t n_صفیں_g = دلائل_2->n_صفیں_g;
    ثابت size_t G        = factx->G;
    ثابت size_t DV       = factx->DV;

    ثابت size_t صفیں_فی_t = دلائل_2->صفیں_فی_t;
    ثابت size_t آغاز      = (size_t) i * صفیں_فی_t;
    ثابت size_t انتہا        = hex_smin(آغاز + صفیں_فی_t, n_صفیں_g);

    اگر (آغاز >= n_صفیں_g) {
        واپس;
    }

    ساخت htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (دلائل_2->q_آغاز * G + آغاز));

    ثابت ساخت htp_tensor * dst        = دلائل_2->dst;
    ثابت __fp16 *            o_tile_src = دلائل_2->o_tile_src;
    ثابت uint32_t            q_آغاز    = دلائل_2->q_آغاز;
    ثابت uint32_t            kv_سر    = دلائل_2->kv_سر;
    ثابت uint32_t            ib3        = دلائل_2->ib3;

    ثابت size_t DV_pad = factx->DV_pad;

    size_t q_idx = fastdiv(آغاز, &factx->div_G);
    size_t h_idx = fastmodulo(آغاز, G, &factx->div_G);

    برائے (size_t r = آغاز; r < انتہا; ++r) {
        اعشاری * برآمد = (اعشاری *) ((uint8_t *) dst->ڈیٹا + (kv_سر * G + h_idx) * dst->nb[1] +
                                 (q_آغاز + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_صفیں;
        size_t         r1            = r % HMX_FP16_TILE_N_صفیں;
        ثابت __fp16 * tile_صف_بنیاد = o_tile_src + r0 * HMX_FP16_TILE_N_صفیں * DV_pad;

        برائے (uint32_t d = 0; d < DV / 32; ++d) {
            ثابت HVX_سمتیہ * اندر_tile = (ثابت HVX_سمتیہ *) (tile_صف_بنیاد + d * HMX_FP16_TILE_N_ELMS);
            HVX_سمتیہ_Pair     vp      = hvx_vec_f16_مقصد_f32_shuff(اندر_tile[r1 / 2]);
            اگر (r1 % 2 == 0) {
                *(HVX_UVector *) (برآمد + d * 32) = Q6_V_lo_W(vp);
            } ورنہ {
                *(HVX_UVector *) (برآمد + d * 32) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail: DV not a multiple of 32 (e.g. 72 -> last 8 lanes). Partial vector-write
        // for the remaining (DV % 32) floats.
        ثابت uint32_t d_دم = DV / 32;
        ثابت uint32_t rem    = DV - d_دم * 32;
        اگر (rem) {
            ثابت HVX_سمتیہ * اندر_tile = (ثابت HVX_سمتیہ *) (tile_صف_بنیاد + d_دم * HMX_FP16_TILE_N_ELMS);
            HVX_سمتیہ_Pair     vp      = hvx_vec_f16_مقصد_f32_shuff(اندر_tile[r1 / 2]);
            HVX_سمتیہ         vd      = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((خالی *) (برآمد + d_دم * 32), rem * حجم(اعشاری), vd);
        }

        h_idx++;
        اگر (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (دلائل_2->q_آغاز * G + آغاز));
}

ساکن خالی fa_o_store_thread_f16(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    fa_o_store_دلائل_t *     دلائل_2  = (fa_o_store_دلائل_t *) ڈیٹا;
    ساخت hmx_fa_context * factx = دلائل_2->factx;

    ثابت size_t n_صفیں_g   = دلائل_2->n_صفیں_g;
    ثابت size_t صفیں_فی_t = دلائل_2->صفیں_فی_t;
    ثابت size_t G          = factx->G;
    ثابت size_t DV         = factx->DV;
    ثابت size_t آغاز      = (size_t) i * صفیں_فی_t;
    ثابت size_t انتہا        = hex_smin(آغاز + صفیں_فی_t, n_صفیں_g);

    اگر (آغاز >= n_صفیں_g) {
        واپس;
    }

    ساخت htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (دلائل_2->q_آغاز * G + آغاز));

    ثابت ساخت htp_tensor * dst        = دلائل_2->dst;
    ثابت __fp16 *            o_tile_src = دلائل_2->o_tile_src;
    ثابت uint32_t            q_آغاز    = دلائل_2->q_آغاز;
    ثابت uint32_t            kv_سر    = دلائل_2->kv_سر;
    ثابت uint32_t            ib3        = دلائل_2->ib3;

    // O-tiles use the padded head dim (DV_pad); dst holds the real DV lanes.
    ثابت size_t DV_pad = factx->DV_pad;

    size_t q_idx = fastdiv(آغاز, &factx->div_G);
    size_t h_idx = fastmodulo(آغاز, G, &factx->div_G);

    برائے (size_t r = آغاز; r < انتہا; ++r) {
        __fp16 * برآمد = (__fp16 *) ((uint8_t *) dst->ڈیٹا + (kv_سر * G + h_idx) * dst->nb[1] +
                                   (q_آغاز + q_idx) * dst->nb[2] + ib3 * dst->nb[3]);

        size_t         r0            = r / HMX_FP16_TILE_N_صفیں;
        size_t         r1            = r % HMX_FP16_TILE_N_صفیں;
        ثابت __fp16 * tile_صف_بنیاد = o_tile_src + r0 * HMX_FP16_TILE_N_صفیں * DV_pad;

        برائے (uint32_t d = 0; d < DV / 64; ++d) {
            ثابت __fp16 *     اندر_dtile = tile_صف_بنیاد + d * HMX_FP16_TILE_N_ELMS * 2;
            ثابت HVX_سمتیہ * pv_in0   = ((ثابت HVX_سمتیہ *) اندر_dtile) + r1 / 2;
            ثابت HVX_سمتیہ * pv_in1   = pv_in0 + 16;
            HVX_سمتیہ_Pair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            اگر (r1 % 2 == 0) {
                *(HVX_UVector *) (برآمد + d * 64) = Q6_V_lo_W(vp);
            } ورنہ {
                *(HVX_UVector *) (برآمد + d * 64) = Q6_V_hi_W(vp);
            }
        }
        // Ragged tail when DV is not a multiple of 64.
        ثابت uint32_t d_دم = DV / 64;
        ثابت uint32_t rem    = DV - d_دم * 64;
        اگر (rem) {
            ثابت __fp16 *     اندر_dtile = tile_صف_بنیاد + d_دم * HMX_FP16_TILE_N_ELMS * 2;
            ثابت HVX_سمتیہ * pv_in0   = ((ثابت HVX_سمتیہ *) اندر_dtile) + r1 / 2;
            ثابت HVX_سمتیہ * pv_in1   = pv_in0 + 16;
            HVX_سمتیہ_Pair     vp       = Q6_W_vdeal_VVR(*pv_in1, *pv_in0, -2);
            HVX_سمتیہ         vd       = (r1 % 2 == 0) ? Q6_V_lo_W(vp) : Q6_V_hi_W(vp);
            hvx_vec_store_u((خالی *) (برآمد + d_دم * 64), rem * حجم(__fp16), vd);
        }

        h_idx++;
        اگر (h_idx == G) {
            h_idx = 0;
            q_idx++;
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) (دلائل_2->q_آغاز * G + آغاز));
}

ساکن خالی fa_phase_o_store(ساخت hmx_fa_context *   factx,
                             ثابت ساخت htp_tensor * dst,
                             ثابت __fp16 *            o_tile_src,
                             uint32_t                  q_آغاز,
                             uint32_t                  kv_سر,
                             uint32_t                  ib3,
                             size_t                    n_صفیں_g) {
    work_قطار_t wp = factx->octx->ctx->work_قطار;
    uint32_t n = 1;
    اگر (factx->n_دھاگے > 1 && n_صفیں_g >= (size_t) (factx->n_دھاگے * 2)) {
        n = factx->n_دھاگے;
    }
    size_t صفیں_فی_t = hmx_ceil_div(n_صفیں_g, n);
    fa_o_store_دلائل_t دلائل_2 = { factx, dst, o_tile_src, q_آغاز, kv_سر, ib3, n_صفیں_g, صفیں_فی_t };
    کارکن_callback_t store_fn = factx->is_dst_fp32 ? fa_o_store_thread_f32 : fa_o_store_thread_f16;
    اگر (n > 1) {
        work_قطار_تسلسل(wp, store_fn, &دلائل_2, n);
    } ورنہ {
        store_fn(1, 0, &دلائل_2);
    }
}

قسمتعریف ساخت {
    ساخت hmx_fa_context *   factx;
    size_t                    buf_idx;
    size_t                    kv_صفیں;
    size_t                    n_صفیں_g;
    size_t                    n_ستون_tiles;
    size_t                    n_tiles_فی_bc;
    size_t                    n_صف_tiles;
    size_t                    n_صف_tiles_g_br;
    uint32_t                  Bc;
    uint32_t                  G;
    uint32_t                  kv_سر;
    uint32_t                  kv_آغاز;
    uint32_t                  q_آغاز;
    uint32_t                  ib3;
    bool                      has_alibi;  // true when max_bias != 0 (need slope * mask + add)
    __fp16 *                  slopes;
    ثابت ساخت htp_tensor * نقاب;
    ثابت __fp16 *            نقاب_vtcm;             // VTCM mask buffer base (NULL = DDR fallback)
    size_t                    نقاب_vtcm_صف_stride;  // elements (__fp16) per row in VTCM mask buffer
    ساخت fastdiv_اقدار     thread_div;
} fa_softmax_دلائل_t;

ساکن اندرونی خالی fa_softmax_impl(
    غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا,
    ثابت bool has_نقاب,
    ثابت bool نقاب_نشر,
    ثابت bool is_g1,
    ثابت bool has_alibi,
    ثابت bool has_softcap
) {
    fa_softmax_دلائل_t *     دلائل_2  = (fa_softmax_دلائل_t *) ڈیٹا;
    ساخت hmx_fa_context * factx = دلائل_2->factx;

    ثابت size_t n_صفیں_g       = دلائل_2->n_صفیں_g;
    ثابت size_t kv_صفیں        = دلائل_2->kv_صفیں;
    ثابت size_t Bc             = دلائل_2->Bc;
    ثابت size_t G              = دلائل_2->G;
    ثابت size_t n_tiles_فی_bc = دلائل_2->n_tiles_فی_bc;
    ثابت size_t n_صف_vec_cnt  = hmx_ceil_div(n_صفیں_g, 64);
    ثابت uint32_t im3          = has_نقاب ? fastmodulo(دلائل_2->ib3, دلائل_2->نقاب->ne[3], &factx->src3_div3) : 0;

    size_t vec_آغاز = 0;
    size_t vec_انتہا   = n_صف_vec_cnt;
    اگر (n > 1) {
        ثابت size_t vecs_فی_t = fastdiv(n_صف_vec_cnt + n - 1, &دلائل_2->thread_div);
        vec_آغاز = i * vecs_فی_t;
        vec_انتہا   = hex_smin(vec_آغاز + vecs_فی_t, n_صف_vec_cnt);
    }

    اگر (vec_آغاز >= n_صف_vec_cnt) {
        واپس;
    }

    ساخت htp_thread_trace * tr = &factx->octx->ctx->trace[i];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (دلائل_2->q_آغاز * G + vec_آغاز * 64));

    // Per-thread row scratch: thread i uses bufs at offset i * 2 * stride
    ثابت size_t صف_buf_stride = factx->صف_buf_stride;
    HVX_سمتیہ * my_صف_buf0    = factx->vtcm_صف_bufs + i * 2 * صف_buf_stride;
    HVX_سمتیہ * my_صف_buf1    = my_صف_buf0 + صف_buf_stride;

    ثابت HVX_سمتیہ v_neg_inf = Q6_Vh_vsplat_R(0xfbff);

    برائے (size_t r_vec_idx = vec_آغاز; r_vec_idx < vec_انتہا; ++r_vec_idx) {
        HVX_سمتیہ rowmax_acc_v = v_neg_inf;
        HVX_سمتیہ rowsum_acc_v = Q6_V_vzero();
        HVX_سمتیہ m_پچھلا_v0    = factx->vtcm_m_vec[r_vec_idx * 2 + 0];
        HVX_سمتیہ m_پچھلا_v1    = factx->vtcm_m_vec[r_vec_idx * 2 + 1];

        HVX_سمتیہ v_slopes = Q6_V_vzero();
        اگر (has_alibi) {
            v_slopes = hvx_vmem(دلائل_2->slopes + r_vec_idx * 64);
        }

        برائے (uint32_t r_vec_off = 0; r_vec_off < 64; r_vec_off += 2) {
            uint32_t r = r_vec_idx * 64 + r_vec_off;
            اگر (r >= hex_align_اوپر(n_صفیں_g, 2)) {
                توڑو;
            }

            uint32_t r0 = r / HMX_FP16_TILE_N_صفیں;
            uint32_t r1 = r % HMX_FP16_TILE_N_صفیں;

            ثابت __fp16 * s_ld_بنیاد = factx->vtcm_s_tiles[دلائل_2->buf_idx] + r0 * HMX_FP16_TILE_N_صفیں * Bc;
            __fp16 *       p_st_بنیاد = factx->vtcm_p_tiles[دلائل_2->buf_idx] + r0 * HMX_FP16_TILE_N_صفیں * Bc;

            // Decode 2 rows from S tiles into per-thread row buffers
            اگر (has_softcap) {
                ثابت HVX_سمتیہ v_cap = hvx_vec_splat_f16(factx->logit_softcap);
                برائے (size_t c = 0; c < kv_صفیں; c += 64) {
                    size_t             ci       = c / 64;
                    ثابت __fp16 *     اندر_dtile = s_ld_بنیاد + ci * HMX_FP16_TILE_N_ELMS * 2;
                    ثابت HVX_سمتیہ * pv_s_in0 = ((ثابت HVX_سمتیہ *) اندر_dtile) + r1 / 2;
                    ثابت HVX_سمتیہ * pv_s_in1 = pv_s_in0 + 16;

                    HVX_سمتیہ_Pair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    HVX_سمتیہ     v_s_row0  = Q6_V_lo_W(vp_s_drow);
                    HVX_سمتیہ     v_s_row1  = Q6_V_hi_W(vp_s_drow);

                    HVX_سمتیہ t0   = hvx_vec_tanh_f16(v_s_row0);
                    my_صف_buf0[ci] = hvx_vec_mul_f16_f16(t0, v_cap);

                    HVX_سمتیہ t1   = hvx_vec_tanh_f16(v_s_row1);
                    my_صف_buf1[ci] = hvx_vec_mul_f16_f16(t1, v_cap);
                }
            } ورنہ {
                size_t c = 0;
                برائے (; c + 64 < kv_صفیں; c += 128) {
                    size_t             ci0       = c / 64;
                    size_t             ci1       = ci0 + 1;
                    ثابت __fp16 *     اندر_dtile0 = s_ld_بنیاد + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                    ثابت __fp16 *     اندر_dtile1 = s_ld_بنیاد + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                    ثابت HVX_سمتیہ * pv_s_in0_0 = ((ثابت HVX_سمتیہ *) اندر_dtile0) + r1 / 2;
                    ثابت HVX_سمتیہ * pv_s_in1_0 = pv_s_in0_0 + 16;
                    ثابت HVX_سمتیہ * pv_s_in0_1 = ((ثابت HVX_سمتیہ *) اندر_dtile1) + r1 / 2;
                    ثابت HVX_سمتیہ * pv_s_in1_1 = pv_s_in0_1 + 16;

                    HVX_سمتیہ_Pair vp_s_drow0 = Q6_W_vdeal_VVR(*pv_s_in1_0, *pv_s_in0_0, -2);
                    my_صف_buf0[ci0]          = Q6_V_lo_W(vp_s_drow0);
                    my_صف_buf1[ci0]          = Q6_V_hi_W(vp_s_drow0);

                    HVX_سمتیہ_Pair vp_s_drow1 = Q6_W_vdeal_VVR(*pv_s_in1_1, *pv_s_in0_1, -2);
                    my_صف_buf0[ci1]          = Q6_V_lo_W(vp_s_drow1);
                    my_صف_buf1[ci1]          = Q6_V_hi_W(vp_s_drow1);
                }
                برائے (; c < kv_صفیں; c += 64) {
                    size_t             ci       = c / 64;
                    ثابت __fp16 *     اندر_dtile = s_ld_بنیاد + ci * HMX_FP16_TILE_N_ELMS * 2;
                    ثابت HVX_سمتیہ * pv_s_in0 = ((ثابت HVX_سمتیہ *) اندر_dtile) + r1 / 2;
                    ثابت HVX_سمتیہ * pv_s_in1 = pv_s_in0 + 16;

                    HVX_سمتیہ_Pair vp_s_drow = Q6_W_vdeal_VVR(*pv_s_in1, *pv_s_in0, -2);
                    my_صف_buf0[ci]          = Q6_V_lo_W(vp_s_drow);
                    my_صف_buf1[ci]          = Q6_V_hi_W(vp_s_drow);
                }
            }

            // Apply mask & compute rowmax(S)
            HVX_سمتیہ v_slope0 = Q6_V_vzero();
            HVX_سمتیہ v_slope1 = Q6_V_vzero();
            اگر (has_alibi) {
                v_slope0 = hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, r_vec_off * 2));
                v_slope1 = (r + 1 < n_صفیں_g) ? hvx_vec_repl_f16(Q6_V_vror_VR(v_slopes, (r_vec_off + 1) * 2)) : Q6_V_vzero();
            }

            ثابت HVX_سمتیہ v_حد = Q6_Vh_vsplat_R(0xcc00);  // fp16 -16.0

            HVX_سمتیہ v_s_rowmax0 = v_neg_inf;
            HVX_سمتیہ v_s_rowmax1 = v_neg_inf;
            اگر (has_نقاب) {
                برائے (size_t c = 0; c < kv_صفیں; c += 64) {
                    size_t         ci          = c / 64;
                    ثابت size_t   ne          = hex_smin(kv_صفیں - c, 64);
                    HVX_سمتیہ_Pred q_دم_keep = Q6_Q_vsetq2_R(ne * حجم(__fp16));

                    HVX_سمتیہ v_mask0, v_mask1;

                    اگر (نقاب_نشر) {
                        اگر (is_g1) {
                            ثابت size_t qi0 = r + 0;
                            v_mask0 = *(ثابت HVX_سمتیہ *) (دلائل_2->نقاب_vtcm + qi0 * دلائل_2->نقاب_vtcm_صف_stride + c);
                            v_mask1 = v_neg_inf;
                            اگر (r + 1 < n_صفیں_g) {
                                ثابت size_t qi1 = r + 1;
                                v_mask1 = *(ثابت HVX_سمتیہ *) (دلائل_2->نقاب_vtcm + qi1 * دلائل_2->نقاب_vtcm_صف_stride + c);
                            }
                        } ورنہ {
                            ثابت size_t qi0 = fastdiv(r + 0, &factx->div_G);
                            v_mask0 = *(ثابت HVX_سمتیہ *) (دلائل_2->نقاب_vtcm + qi0 * دلائل_2->نقاب_vtcm_صف_stride + c);
                            v_mask1 = v_neg_inf;
                            اگر (r + 1 < n_صفیں_g) {
                                ثابت size_t qi1 = fastdiv(r + 1, &factx->div_G);
                                اگر (qi1 == qi0) {
                                    v_mask1 = v_mask0;
                                } ورنہ {
                                    v_mask1 = *(ثابت HVX_سمتیہ *) (دلائل_2->نقاب_vtcm + qi1 * دلائل_2->نقاب_vtcm_صف_stride + c);
                                }
                            }
                        }
                    } ورنہ {
                        // Head-dependent mask: pre-interleaved per row r.
                        ثابت size_t r0 = r + 0;
                        v_mask0 = *(ثابت HVX_سمتیہ *) (دلائل_2->نقاب_vtcm + r0 * دلائل_2->نقاب_vtcm_صف_stride + c);
                        v_mask1 = v_neg_inf;
                        اگر (r + 1 < n_صفیں_g) {
                            ثابت size_t r1 = r + 1;
                            v_mask1 = *(ثابت HVX_سمتیہ *) (دلائل_2->نقاب_vtcm + r1 * دلائل_2->نقاب_vtcm_صف_stride + c);
                        }
                    }

                    // Threshold: mask values below -16.0 are treated as -inf (causal mask).
                    HVX_سمتیہ_Pred q_keep0 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask0, v_حد), q_دم_keep);
                    HVX_سمتیہ_Pred q_keep1 = Q6_Q_and_QQ(Q6_Q_vcmp_gt_VhfVhf(v_mask1, v_حد), q_دم_keep);

                    // Scale mask values by log2(e) for base-2 calculations
                    ثابت HVX_سمتیہ v_log2e = hvx_vec_splat_f16(EXP_LOG2E_F);
                    HVX_سمتیہ v_mask0_scaled = hvx_vec_mul_f16_f16(v_mask0, v_log2e);
                    HVX_سمتیہ v_mask1_scaled = hvx_vec_mul_f16_f16(v_mask1, v_log2e);

                    اگر (has_alibi) {
                        HVX_سمتیہ v_sm0 = hvx_vec_mul_f16_f16(v_mask0_scaled, v_slope0);
                        HVX_سمتیہ v_sm1 = hvx_vec_mul_f16_f16(v_mask1_scaled, v_slope1);
                        my_صف_buf0[ci]  = Q6_V_vmux_QVV(q_keep0, hvx_vec_جوڑو_f16_f16(my_صف_buf0[ci], v_sm0), v_neg_inf);
                        my_صف_buf1[ci]  = Q6_V_vmux_QVV(q_keep1, hvx_vec_جوڑو_f16_f16(my_صف_buf1[ci], v_sm1), v_neg_inf);
                    } ورنہ {
                        my_صف_buf0[ci] = Q6_V_vmux_QVV(q_keep0, hvx_vec_جوڑو_f16_f16(my_صف_buf0[ci], v_mask0_scaled), v_neg_inf);
                        my_صف_buf1[ci] = Q6_V_vmux_QVV(q_keep1, hvx_vec_جوڑو_f16_f16(my_صف_buf1[ci], v_mask1_scaled), v_neg_inf);
                    }

                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_صف_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_صف_buf1[ci]);
                }
            } ورنہ {
                size_t c = 0;
                برائے (; c + 64 < kv_صفیں; c += 128) {
                    size_t ci0 = c / 64;
                    size_t ci1 = ci0 + 1;
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_صف_buf0[ci0]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_صف_buf1[ci0]);
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_صف_buf0[ci1]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_صف_buf1[ci1]);
                }
                برائے (; c < kv_صفیں; c += 64) {
                    size_t         ci          = c / 64;
                    ثابت size_t   ne          = hex_smin(kv_صفیں - c, 64);
                    HVX_سمتیہ_Pred q_دم_keep = Q6_Q_vsetq2_R(ne * حجم(__fp16));
                    اگر (ne < 64) {
                        my_صف_buf0[ci] = Q6_V_vmux_QVV(q_دم_keep, my_صف_buf0[ci], v_neg_inf);
                        my_صف_buf1[ci] = Q6_V_vmux_QVV(q_دم_keep, my_صف_buf1[ci], v_neg_inf);
                    }
                    v_s_rowmax0 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax0, my_صف_buf0[ci]);
                    v_s_rowmax1 = Q6_Vhf_vmax_VhfVhf(v_s_rowmax1, my_صف_buf1[ci]);
                }
            }

            v_s_rowmax0 = hvx_vec_reduce_max_f16(v_s_rowmax0);
            v_s_rowmax1 = hvx_vec_reduce_max_f16(v_s_rowmax1);

            // Splat m_prev[r], m_prev[r+1] from the float per-row accumulators and convert to fp16 vectors
            HVX_سمتیہ v_m_prev0, v_m_prev1;
            اگر (r_vec_off < 32) {
                HVX_سمتیہ v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_پچھلا_v0, r_vec_off * 4));
                v_m_prev0 = hvx_vec_f32_مقصد_f16(v0, v0);
                اگر (r + 1 < n_صفیں_g) {
                    HVX_سمتیہ v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_پچھلا_v0, (r_vec_off + 1) * 4));
                    v_m_prev1 = hvx_vec_f32_مقصد_f16(v1, v1);
                } ورنہ {
                    v_m_prev1 = Q6_V_vzero();
                }
            } ورنہ {
                HVX_سمتیہ v0 = hvx_vec_repl_f32(Q6_V_vror_VR(m_پچھلا_v1, (r_vec_off - 32) * 4));
                v_m_prev0 = hvx_vec_f32_مقصد_f16(v0, v0);
                اگر (r + 1 < n_صفیں_g) {
                    HVX_سمتیہ v1 = hvx_vec_repl_f32(Q6_V_vror_VR(m_پچھلا_v1, (r_vec_off + 1 - 32) * 4));
                    v_m_prev1 = hvx_vec_f32_مقصد_f16(v1, v1);
                } ورنہ {
                    v_m_prev1 = Q6_V_vzero();
                }
            }

            HVX_سمتیہ v_dup_m0 = Q6_Vhf_vmax_VhfVhf(v_m_prev0, v_s_rowmax0);
            HVX_سمتیہ v_dup_m1 = Q6_Vhf_vmax_VhfVhf(v_m_prev1, v_s_rowmax1);

            // Insert row r, r+1 rowmax into rowmax_acc_v
            {
                HVX_سمتیہ_Pred p_آغاز = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_سمتیہ_Pred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_سمتیہ_Pred p_انتہا   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_سمتیہ_Pred p_lane0 = Q6_Q_and_QQn(p_mid, p_آغاز);
                HVX_سمتیہ_Pred p_lane1 = Q6_Q_and_QQn(p_انتہا, p_mid);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane0, v_dup_m0, rowmax_acc_v);
                rowmax_acc_v           = Q6_V_vmux_QVV(p_lane1, v_dup_m1, rowmax_acc_v);
            }

            // Compute P = exp(S - m_new)
            ثابت HVX_سمتیہ v_zero      = Q6_V_vzero();
            HVX_سمتیہ       v_p_rowsum0 = v_zero;
            HVX_سمتیہ       v_p_rowsum1 = v_zero;

            size_t c = 0;
            برائے (; c + 64 < kv_صفیں; c += 128) {
                size_t     ci0          = c / 64;
                size_t     ci1          = ci0 + 1;

                HVX_سمتیہ v_s_minus_m0_0 = Q6_Vqf16_vsub_VhfVhf(my_صف_buf0[ci0], v_dup_m0);
                HVX_سمتیہ v_s_minus_m1_0 = Q6_Vqf16_vsub_VhfVhf(my_صف_buf1[ci0], v_dup_m1);
                HVX_سمتیہ v_s_minus_m0_1 = Q6_Vqf16_vsub_VhfVhf(my_صف_buf0[ci1], v_dup_m0);
                HVX_سمتیہ v_s_minus_m1_1 = Q6_Vqf16_vsub_VhfVhf(my_صف_buf1[ci1], v_dup_m1);

                HVX_سمتیہ v_p_row0_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_0));
                HVX_سمتیہ v_p_row1_hf_0  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_0));
                HVX_سمتیہ v_p_row0_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0_1));
                HVX_سمتیہ v_p_row1_hf_1  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1_1));

                __fp16 *     برآمد_dtile0  = p_st_بنیاد + ci0 * HMX_FP16_TILE_N_ELMS * 2;
                __fp16 *     برآمد_dtile1  = p_st_بنیاد + ci1 * HMX_FP16_TILE_N_ELMS * 2;
                HVX_سمتیہ * pv_p_out0_0  = ((HVX_سمتیہ *) برآمد_dtile0) + r1 / 2;
                HVX_سمتیہ * pv_p_out1_0  = pv_p_out0_0 + 16;
                HVX_سمتیہ * pv_p_out0_1  = ((HVX_سمتیہ *) برآمد_dtile1) + r1 / 2;
                HVX_سمتیہ * pv_p_out1_1  = pv_p_out0_1 + 16;

                HVX_سمتیہ_Pair vp_p_dual0 = Q6_W_vshuff_VVR(v_p_row1_hf_0, v_p_row0_hf_0, -2);
                *pv_p_out0_0               = Q6_V_lo_W(vp_p_dual0);
                *pv_p_out1_0               = Q6_V_hi_W(vp_p_dual0);

                HVX_سمتیہ_Pair vp_p_dual1 = Q6_W_vshuff_VVR(v_p_row1_hf_1, v_p_row0_hf_1, -2);
                *pv_p_out0_1               = Q6_V_lo_W(vp_p_dual1);
                *pv_p_out1_1               = Q6_V_hi_W(vp_p_dual1);

                HVX_سمتیہ_Pair vp_p0_0 = hvx_vec_f16_مقصد_f32_shuff(v_p_row0_hf_0);
                HVX_سمتیہ_Pair vp_p1_0 = hvx_vec_f16_مقصد_f32_shuff(v_p_row1_hf_0);
                HVX_سمتیہ_Pair vp_p0_1 = hvx_vec_f16_مقصد_f32_shuff(v_p_row0_hf_1);
                HVX_سمتیہ_Pair vp_p1_1 = hvx_vec_f16_مقصد_f32_shuff(v_p_row1_hf_1);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_0), Q6_V_hi_W(vp_p0_0)));
                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0_1), Q6_V_hi_W(vp_p0_1)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_0), Q6_V_hi_W(vp_p1_0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1_1), Q6_V_hi_W(vp_p1_1)));
            }
            برائے (size_t c_rem = c; c_rem < kv_صفیں; c_rem += 64) {
                size_t     ci           = c_rem / 64;
                HVX_سمتیہ v_s_minus_m0 = Q6_Vqf16_vsub_VhfVhf(my_صف_buf0[ci], v_dup_m0);
                HVX_سمتیہ v_s_minus_m1 = Q6_Vqf16_vsub_VhfVhf(my_صف_buf1[ci], v_dup_m1);

                HVX_سمتیہ v_p_row0_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m0));
                HVX_سمتیہ v_p_row1_hf  = hvx_vec_exp2_f16(Q6_Vhf_equals_Vqf16(v_s_minus_m1));
                __fp16 *     برآمد_dtile  = p_st_بنیاد + ci * HMX_FP16_TILE_N_ELMS * 2;
                HVX_سمتیہ * pv_p_out0  = ((HVX_سمتیہ *) برآمد_dtile) + r1 / 2;
                HVX_سمتیہ * pv_p_out1  = pv_p_out0 + 16;

                HVX_سمتیہ_Pair vp_p_dual = Q6_W_vshuff_VVR(v_p_row1_hf, v_p_row0_hf, -2);
                *pv_p_out0               = Q6_V_lo_W(vp_p_dual);
                *pv_p_out1               = Q6_V_hi_W(vp_p_dual);

                HVX_سمتیہ_Pair vp_p0 = hvx_vec_f16_مقصد_f32_shuff(v_p_row0_hf);
                HVX_سمتیہ_Pair vp_p1 = hvx_vec_f16_مقصد_f32_shuff(v_p_row1_hf);

                v_p_rowsum0 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum0, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p0), Q6_V_hi_W(vp_p0)));
                v_p_rowsum1 = Q6_Vqf32_vadd_Vqf32Vqf32(v_p_rowsum1, Q6_Vqf32_vadd_VsfVsf(Q6_V_lo_W(vp_p1), Q6_V_hi_W(vp_p1)));
            }

            HVX_سمتیہ rowsum0_sf = hvx_vec_reduce_مجموع_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum0));
            HVX_سمتیہ rowsum1_sf = hvx_vec_reduce_مجموع_f32(Q6_Vsf_equals_Vqf32(v_p_rowsum1));
            {
                HVX_سمتیہ rv0_v = hvx_vec_f32_مقصد_f16(rowsum0_sf, rowsum0_sf);
                HVX_سمتیہ rv1_v = hvx_vec_f32_مقصد_f16(rowsum1_sf, rowsum1_sf);

                HVX_سمتیہ_Pred p_آغاز = Q6_Q_vsetq_R(r_vec_off * 2);
                HVX_سمتیہ_Pred p_mid   = Q6_Q_vsetq_R((r_vec_off + 1) * 2);
                HVX_سمتیہ_Pred p_انتہا   = Q6_Q_vsetq2_R((r_vec_off + 2) * 2);
                HVX_سمتیہ_Pred p_lane0 = Q6_Q_and_QQn(p_mid, p_آغاز);
                HVX_سمتیہ_Pred p_lane1 = Q6_Q_and_QQn(p_انتہا, p_mid);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane0, rv0_v, rowsum_acc_v);
                rowsum_acc_v           = Q6_V_vmux_QVV(p_lane1, rv1_v, rowsum_acc_v);
            }
        }

        // Inline fa_ml_update_and_build_d for this vector (lock-free and in parallel)
        HVX_سمتیہ_Pair rowmax_acc_pair    = hvx_vec_f16_مقصد_f32(rowmax_acc_v);
        HVX_سمتیہ     v_rowmax_acc_f32_0 = Q6_V_lo_W(rowmax_acc_pair);
        HVX_سمتیہ     v_rowmax_acc_f32_1 = Q6_V_hi_W(rowmax_acc_pair);

        HVX_سمتیہ v_m_curr0 = Q6_Vsf_vmax_VsfVsf(m_پچھلا_v0, v_rowmax_acc_f32_0);
        HVX_سمتیہ v_m_curr1 = Q6_Vsf_vmax_VsfVsf(m_پچھلا_v1, v_rowmax_acc_f32_1);

        HVX_سمتیہ v_m_diff0 = HVX_OP_SUB_F32(m_پچھلا_v0, v_m_curr0);
        HVX_سمتیہ v_m_diff1 = HVX_OP_SUB_F32(m_پچھلا_v1, v_m_curr1);

        HVX_سمتیہ v_m_diff_f16   = hvx_vec_f32_مقصد_f16(v_m_diff0, v_m_diff1);
        HVX_سمتیہ exp_m_diff_f16 = hvx_vec_exp2_f16(v_m_diff_f16);

        HVX_سمتیہ_Pair exp_m_diff_pair = hvx_vec_f16_مقصد_f32(exp_m_diff_f16);
        HVX_سمتیہ exp_m_diff0 = Q6_V_lo_W(exp_m_diff_pair);
        HVX_سمتیہ exp_m_diff1 = Q6_V_hi_W(exp_m_diff_pair);

        HVX_سمتیہ_Pair rowsum_acc_pair = hvx_vec_f16_مقصد_f32(rowsum_acc_v);
        HVX_سمتیہ     v_rowsum_acc_f32_0 = Q6_V_lo_W(rowsum_acc_pair);
        HVX_سمتیہ     v_rowsum_acc_f32_1 = Q6_V_hi_W(rowsum_acc_pair);

        HVX_سمتیہ l_پچھلا_v0 = factx->vtcm_l_vec[r_vec_idx * 2 + 0];
        HVX_سمتیہ l_پچھلا_v1 = factx->vtcm_l_vec[r_vec_idx * 2 + 1];
        HVX_سمتیہ v_l_curr0 = HVX_OP_جوڑو_F32(HVX_OP_MUL_F32(l_پچھلا_v0, exp_m_diff0), v_rowsum_acc_f32_0);
        HVX_سمتیہ v_l_curr1 = HVX_OP_جوڑو_F32(HVX_OP_MUL_F32(l_پچھلا_v1, exp_m_diff1), v_rowsum_acc_f32_1);

        factx->vtcm_m_vec[r_vec_idx * 2 + 0] = v_m_curr0;
        factx->vtcm_m_vec[r_vec_idx * 2 + 1] = v_m_curr1;
        factx->vtcm_l_vec[r_vec_idx * 2 + 0] = v_l_curr0;
        factx->vtcm_l_vec[r_vec_idx * 2 + 1] = v_l_curr1;

        // Build diagonal tile D = diag(exp(m_diff))
        ثابت HVX_سمتیہ     v_offsets = *(ثابت HVX_سمتیہ *) d_tile_scatter_offsets;
        ثابت HVX_سمتیہ_Pred q_32_نقاب = Q6_Q_vsetq_R(32 * حجم(__fp16));
        HVX_سمتیہ           v_exp_m_diff = exp_m_diff_f16;

        __fp16 * ثابت d_tiles_برآمد = factx->vtcm_d_tiles[دلائل_2->buf_idx];

        size_t t0 = r_vec_idx * 2;
        اگر (t0 < دلائل_2->n_صف_tiles) {
            ثابت HVX_سمتیہ v_content = v_exp_m_diff;
            __fp16 *         برآمد_بنیاد  = d_tiles_برآمد + t0 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_نقاب, (size_t) برآمد_بنیاد, HMX_FP16_TILE_سائز - 1, v_offsets, v_content);
        }

        size_t t1 = r_vec_idx * 2 + 1;
        اگر (t1 < دلائل_2->n_صف_tiles) {
            ثابت HVX_سمتیہ v_content = Q6_V_vror_VR(v_exp_m_diff, 64);
            __fp16 *         برآمد_بنیاد  = d_tiles_برآمد + t1 * HMX_FP16_TILE_N_ELMS;
            Q6_vscatter_QRMVhV(q_32_نقاب, (size_t) برآمد_بنیاد, HMX_FP16_TILE_سائز - 1, v_offsets, v_content);
        }
    }
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_FA_SFM, (uint16_t) (دلائل_2->q_آغاز * G + vec_آغاز * 64));
}

ساکن خالی fa_softmax_thread_nomask(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    fa_softmax_impl(n, i, ڈیٹا,
                    /*has_mask=*/false,
                    /*mask_broadcast=*/false,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ساکن خالی fa_softmax_thread_نقاب_نشر_g1(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    fa_softmax_impl(n, i, ڈیٹا,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/true,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ساکن خالی fa_softmax_thread_نقاب_نشر_gn(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    fa_softmax_impl(n, i, ڈیٹا,
                    /*has_mask=*/true,
                    /*mask_broadcast=*/true,
                    /*is_g1=*/false,
                    /*has_alibi=*/false,
                    /*has_softcap=*/false);
}

ساکن خالی fa_softmax_thread(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    fa_softmax_دلائل_t *     دلائل_2  = (fa_softmax_دلائل_t *) ڈیٹا;
    ساخت hmx_fa_context * factx = دلائل_2->factx;

    ثابت bool has_نقاب       = (دلائل_2->نقاب != تہی);
    ثابت bool نقاب_نشر = factx->نقاب_نشر;
    ثابت bool is_g1          = (دلائل_2->G == 1);
    ثابت bool has_alibi      = دلائل_2->has_alibi;
    ثابت bool has_softcap    = factx->has_softcap;

    fa_softmax_impl(n, i, ڈیٹا, has_نقاب, نقاب_نشر, is_g1, has_alibi, has_softcap);
}

ساکن اندرونی HVX_سمتیہ hvx_fa_بھرو_sinks(ثابت اعشاری * sinks_ڈیٹا, size_t idx, size_t گنتی) {
    ثابت HVX_سمتیہ * v_src    = (ثابت HVX_سمتیہ *) sinks_ڈیٹا + (idx >> 5);
    ثابت size_t       lane_off = idx & 31;
    ثابت size_t       byte_off = lane_off * حجم(اعشاری);
    واپس (lane_off + گنتی > 32)
        ? Q6_V_valign_VVR(v_src[1], v_src[0], byte_off)
        : Q6_V_vror_VR(v_src[0], byte_off);
}

ساکن __attribute__((noinline)) خالی fa_build_d_diag_inv_l(ساخت hmx_fa_context * factx,
                                                            size_t                  n_صف_tiles,
                                                            size_t                  n_صف_tiles_g_br,
                                                            uint32_t                kv_سر,
                                                            size_t                  n_صفیں_g) {
    (خالی) n_صف_tiles_g_br;

    // sinks: l += exp(sink - m * ln2) in f32, m and l are in the log2 domain
    اگر (factx->sinks) {
        ثابت اعشاری *    sinks_ڈیٹا   = factx->vtcm_sinks;
        ثابت size_t     G            = factx->G;
        ثابت size_t     سر_بنیاد    = (size_t) kv_سر * G;
        ثابت HVX_سمتیہ v_ln2        = Q6_V_vsplat_R(EXP_LOGN2);
        ثابت HVX_سمتیہ v_سر_sinks = (G <= 32) ? hvx_fa_بھرو_sinks(sinks_ڈیٹا, سر_بنیاد, G) : Q6_V_vzero();
        HVX_سمتیہ       v_sink       = Q6_V_vzero();
        size_t           g_پچھلا       = SIZE_MAX;

        برائے (size_t i = 0; i < n_صف_tiles; ++i) {
            // the sink pattern of a tile only depends on (i * 32) % G, rebuild it when that changes
            ثابت size_t r0 = i * 32;
            ثابت size_t g0 = fastmodulo(r0, G, &factx->div_G);
            اگر (g0 != g_پچھلا) {
                ثابت size_t rem_g = G - g0;
                اگر (G <= 32) {
                    ثابت HVX_سمتیہ v_lo = Q6_V_vror_VR(v_سر_sinks, g0 * حجم(اعشاری));
                    ثابت HVX_سمتیہ v_hi = Q6_V_vror_VR(v_سر_sinks, 128 - rem_g * حجم(اعشاری));
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq2_R(rem_g * حجم(اعشاری)), v_lo, v_hi);
                    برائے (size_t len = G; len < 32; len <<= 1) {
                        ثابت HVX_سمتیہ_Pred q_keep    = Q6_Q_vsetq_R(len * حجم(اعشاری));
                        ثابت HVX_سمتیہ     v_shifted = Q6_V_vror_VR(v_sink, 128 - len * حجم(اعشاری));
                        v_sink = Q6_V_vmux_QVV(q_keep, v_sink, v_shifted);
                    }
                } ورنہ اگر (rem_g >= 32) {
                    v_sink = hvx_fa_بھرو_sinks(sinks_ڈیٹا, سر_بنیاد + g0, 32);
                } ورنہ {
                    ثابت HVX_سمتیہ v_lo = hvx_fa_بھرو_sinks(sinks_ڈیٹا, سر_بنیاد + g0, rem_g);
                    ثابت HVX_سمتیہ v_hi = hvx_fa_بھرو_sinks(sinks_ڈیٹا, سر_بنیاد, 32 - rem_g);
                    v_sink = Q6_V_vmux_QVV(Q6_Q_vsetq_R(rem_g * حجم(اعشاری)), v_lo, Q6_V_vror_VR(v_hi, 128 - rem_g * حجم(اعشاری)));
                }
                g_پچھلا = g0;
            }

            HVX_سمتیہ v_d = HVX_OP_SUB_F32(v_sink, HVX_OP_MUL_F32(factx->vtcm_m_vec[i], v_ln2));
            اگر (r0 + 32 > n_صفیں_g) {
                ثابت HVX_سمتیہ_Pred q_valid = Q6_Q_vsetq_R((n_صفیں_g - r0) * حجم(اعشاری));
                v_d = Q6_V_vmux_QVV(q_valid, v_d, Q6_V_vzero());
            }
            factx->vtcm_l_vec[i] = HVX_OP_جوڑو_F32(factx->vtcm_l_vec[i], hvx_vec_exp_f32(v_d));
        }
    }

    ثابت HVX_سمتیہ     v_offsets = *(ثابت HVX_سمتیہ *) d_tile_scatter_offsets;
    ثابت HVX_سمتیہ_Pred q_32_نقاب = Q6_Q_vsetq_R(32 * حجم(__fp16));
    ثابت HVX_سمتیہ     one       = hvx_vec_splat_f32(1.0f);

    HVX_سمتیہ v_content = Q6_V_vzero();
    برائے (size_t i = 0; i < n_صف_tiles; ++i) {
        اگر ((i % 2) == 0) {
            HVX_سمتیہ inv_lo = HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i]));
            HVX_سمتیہ inv_hi = (i + 1 < n_صف_tiles) ? HVX_OP_MUL_F32(one, hvx_vec_inverse_f32(factx->vtcm_l_vec[i + 1])) : Q6_V_vzero();
            v_content = hvx_vec_f32_مقصد_f16(inv_lo, inv_hi);
        } ورنہ {
            v_content = Q6_V_vror_VR(v_content, 64);
        }

        __fp16 * برآمد_بنیاد = factx->vtcm_d_inv_l + i * HMX_FP16_TILE_N_ELMS;
        Q6_vscatter_QRMVhV(q_32_نقاب, (size_t) برآمد_بنیاد, HMX_FP16_TILE_سائز - 1, v_offsets, v_content);
    }
}

ساکن خالی fa_phase_softmax_and_build_d(ساخت hmx_fa_context * factx,
                                         fa_softmax_دلائل_t *     sargs,
                                         size_t                  n_صف_tiles,
                                         size_t                  n_صف_tiles_g_br) {
    work_قطار_t wp = factx->octx->ctx->work_قطار;
    ثابت size_t n_صف_vec_cnt = hmx_ceil_div(sargs->n_صفیں_g, 64);

    کارکن_callback_t softmax_fn = fa_softmax_thread;
    اگر (sargs->نقاب == تہی && !factx->has_softcap && !sargs->has_alibi) {
        softmax_fn = fa_softmax_thread_nomask;
    } ورنہ اگر (sargs->نقاب != تہی && factx->نقاب_نشر && !factx->has_softcap && !sargs->has_alibi) {
        اگر (sargs->G == 1) {
            softmax_fn = fa_softmax_thread_نقاب_نشر_g1;
        } ورنہ {
            softmax_fn = fa_softmax_thread_نقاب_نشر_gn;
        }
    }

    اگر (factx->n_دھاگے > 1 && n_صف_vec_cnt >= 2) {
        uint32_t n_use = (uint32_t) hex_smin((size_t) factx->n_دھاگے, n_صف_vec_cnt);
        sargs->thread_div = init_fastdiv_اقدار(n_use);
        work_قطار_تسلسل(wp, softmax_fn, sargs, n_use);
    } ورنہ {
        softmax_fn(1, 0, sargs);
    }
}

// ============================================================================
// HMX job structs and worker functions
// ============================================================================

قسمتعریف ساخت {
    ثابت __fp16 * q_tiles;
    ثابت __fp16 * k_tiles;
    __fp16 *       s_tiles;
    size_t         n_صف_tiles;
    size_t         n_ستون_tiles;
    size_t         n_dot_tiles;  // DK / 32
    size_t         n_tiles_فی_bc;
    uint8_t *      hmx_scales;
} hmx_fa_qk_کام_t;

ساکن خالی hmx_fa_qk_dot_کارکن(خالی * ڈیٹا) {
    hmx_fa_qk_کام_t * کام            = (hmx_fa_qk_کام_t *) ڈیٹا;
    ثابت size_t      n_صف_tiles    = کام->n_صف_tiles;
    ثابت size_t      n_ستون_tiles    = کام->n_ستون_tiles;
    ثابت size_t      n_dot_tiles    = کام->n_dot_tiles;
    ثابت size_t      n_tiles_فی_bc = کام->n_tiles_فی_bc;
    ثابت __fp16 * محدود q_tiles  = کام->q_tiles;
    ثابت __fp16 * محدود k_tiles  = کام->k_tiles;
    __fp16 * محدود s_tiles        = کام->s_tiles;
    __builtin_assume(n_صف_tiles > 0);
    __builtin_assume(n_ستون_tiles > 0);
    __builtin_assume(n_dot_tiles > 0);

    asm متلون(HMX_سیٹ_BIAS("%0") :: "r"((غیرعلامتی صحیح)کام->hmx_scales));
    ثابت size_t dot_stride = n_dot_tiles * HMX_FP16_TILE_N_ELMS;
    برائے (size_t r = 0; r < n_صف_tiles; ++r) {
        ثابت __fp16 * صف_tiles = q_tiles + r * dot_stride;
        ثابت __fp16 * ستون_tiles = k_tiles;
        __fp16 *       برآمد_tile  = s_tiles + r * n_tiles_فی_bc * HMX_FP16_TILE_N_ELMS;

        برائے (size_t c = 0; c < n_ستون_tiles; ++c) {
            hmx_fa_qk_dot_tile(صف_tiles, ستون_tiles, برآمد_tile, n_dot_tiles);
            ستون_tiles += dot_stride;
            برآمد_tile  += HMX_FP16_TILE_N_ELMS;
        }
    }
}

قسمتعریف ساخت {
    __fp16 *       o_curr;
    ثابت __fp16 * o_پچھلا;
    ثابت __fp16 * p_tiles;
    ثابت __fp16 * v_tiles;
    ثابت __fp16 * d_tiles;
    uint8_t *      hmx_scales;
    size_t         n_صف_tiles;
    size_t         n_ستون_tiles;
    size_t         n_صف_tiles_g_br;
    size_t         n_tiles_فی_bc;
    size_t         DV;
} hmx_fa_o_update_کام_t;

ساکن خالی hmx_fa_o_update_کارکن(خالی * ڈیٹا) {
    hmx_fa_o_update_کام_t * کام              = (hmx_fa_o_update_کام_t *) ڈیٹا;
    ثابت size_t            n_صف_tiles      = کام->n_صف_tiles;
    ثابت size_t            n_ستون_tiles      = کام->n_ستون_tiles;
    ثابت size_t            n_صف_tiles_g_br = کام->n_صف_tiles_g_br;
    ثابت size_t            n_tiles_فی_bc   = کام->n_tiles_فی_bc;
    ثابت size_t            DV_tiles         = کام->DV / 32;
    ثابت __fp16 * محدود d_tiles          = کام->d_tiles;
    ثابت __fp16 * محدود p_tiles          = کام->p_tiles;
    ثابت __fp16 * محدود v_tiles          = کام->v_tiles;
    ثابت __fp16 * محدود o_پچھلا           = کام->o_پچھلا;
    __fp16 * محدود o_curr                 = کام->o_curr;
    __builtin_assume(n_صف_tiles > 0);
    __builtin_assume(n_ستون_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm متلون(HMX_سیٹ_BIAS("%0") :: "r"((غیرعلامتی صحیح)کام->hmx_scales));
    ثابت size_t o_stride = n_صف_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    ثابت size_t v_stride = n_tiles_فی_bc * HMX_FP16_TILE_N_ELMS;
    برائے (size_t r = 0; r < n_صف_tiles; ++r) {
        ثابت __fp16 * d_diag     = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        ثابت __fp16 * p_tile_اندر  = p_tiles + (r * n_tiles_فی_bc) * HMX_FP16_TILE_N_ELMS;
        ثابت __fp16 * o_rc       = o_پچھلا + r * HMX_FP16_TILE_N_ELMS;
        ثابت __fp16 * v_tile_اندر  = v_tiles;
        __fp16       * o_tile_برآمد = o_curr + r * HMX_FP16_TILE_N_ELMS;

        برائے (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_update_tile(d_diag, o_rc, p_tile_اندر, v_tile_اندر, o_tile_برآمد, n_ستون_tiles);
            o_rc       += o_stride;
            v_tile_اندر  += v_stride;
            o_tile_برآمد += o_stride;
        }
    }
}

قسمتعریف ساخت {
    __fp16 *       o_curr;   // output (row-major tile layout)
    ثابت __fp16 * o_پچھلا;   // input (column-major tile layout)
    ثابت __fp16 * d_tiles;  // diag(1/l) tiles
    uint8_t *      hmx_scales;
    size_t         n_صف_tiles;
    size_t         n_صف_tiles_g_br;
    size_t         DV;
} hmx_fa_o_norm_کام_t;

ساکن خالی hmx_fa_o_norm_کارکن(خالی * ڈیٹا) {
    hmx_fa_o_norm_کام_t * کام              = (hmx_fa_o_norm_کام_t *) ڈیٹا;
    ثابت size_t          n_صف_tiles      = کام->n_صف_tiles;
    ثابت size_t          n_صف_tiles_g_br = کام->n_صف_tiles_g_br;
    ثابت size_t          DV_tiles         = کام->DV / 32;
    ثابت __fp16 * محدود d_tiles        = کام->d_tiles;
    ثابت __fp16 * محدود o_پچھلا         = کام->o_پچھلا;
    __fp16 * محدود o_curr               = کام->o_curr;
    __builtin_assume(n_صف_tiles > 0);
    __builtin_assume(DV_tiles > 0);

    asm متلون(HMX_سیٹ_BIAS("%0") :: "r"((غیرعلامتی صحیح)کام->hmx_scales));
    ثابت size_t o_stride = n_صف_tiles_g_br * HMX_FP16_TILE_N_ELMS;
    برائے (size_t r = 0; r < n_صف_tiles; ++r) {
        ثابت __fp16 * d_diag = d_tiles + r * HMX_FP16_TILE_N_ELMS;
        ثابت __fp16 * o_rc = o_پچھلا + r * HMX_FP16_TILE_N_ELMS;
        __fp16 *       o_برآمد = o_curr + r * DV_tiles * HMX_FP16_TILE_N_ELMS;

        برائے (size_t c = 0; c < DV_tiles; ++c) {
            hmx_fa_o_norm_tile(d_diag, o_rc, o_برآمد);
            o_rc  += o_stride;
            o_برآمد += HMX_FP16_TILE_N_ELMS;
        }
    }
}

// Populate per-GQA-row ALiBi slopes for a given KV head.
ساکن __attribute__((noinline)) خالی fa_compute_slopes(
                              ثابت ساخت hmx_fa_context * factx,
                              uint32_t                      kv_سر,
                              size_t                        n_صفیں_g) {
    __fp16 * slopes = factx->vtcm_slopes;
    اگر (factx->max_bias == 0.0f) {
        hvx_splat_f16_a(slopes, 1.0f, n_صفیں_g);
        واپس;
    }

    ثابت uint32_t G           = factx->G;
    ثابت uint32_t n_سر_log2 = factx->n_سر_log2;
    ثابت اعشاری    m0          = factx->m0;
    ثابت اعشاری    m1          = factx->m1;

    __fp16 عارضی_slopes[512] __attribute__((aligned(128)));
    اگر (G <= 32) {
        // Fast path: Compute G unique slope values in vector registers
        HVX_سمتیہ v_val = hvx_alibi_slopes(kv_سر, G, n_سر_log2, m0, m1);

        __fp16 عارضی_slopes_aligned[64] __attribute__((aligned(128)));
        hvx_vmem(عارضی_slopes_aligned) = hvx_vec_f32_مقصد_f16(v_val, Q6_V_vzero());

        برائے (uint32_t i = 0; i < G; ++i) {
            عارضی_slopes[i] = عارضی_slopes_aligned[i];
        }
    } ورنہ {
        // Fallback path: G > 32 (rare configurations)
        برائے (uint32_t i = 0; i < G; ++i) {
            عارضی_slopes[i] = (__fp16)alibi_ڈھلان(kv_سر * G + i, n_سر_log2, m0, m1);
        }
    }

    // Allocate stack buffer to avoid scalar writes to VTCM (which generates L2 misses)
    __fp16 local_slopes[n_صفیں_g] __attribute__((aligned(128)));
    برائے (size_t r = 0; r < n_صفیں_g; ++r) {
        local_slopes[r] = عارضی_slopes[fastmodulo(r, G, &factx->div_G)];
    }

    // Copy to VTCM slopes using HVX block copy (both are aligned to 128 bytes)
    hvx_copy_f16_aa((uint8_t *)slopes, (ثابت uint8_t *)local_slopes, n_صفیں_g);
}

ساکن خالی fa_ڈالو_نقاب_dma_gqa(
    dma_قطار *               dma_q,
    ثابت ساخت htp_tensor * نقاب,
    uint32_t                  q_آغاز,
    uint32_t                  im3,
    uint32_t                  kv_آغاز,
    uint32_t                  kv_سر,
    uint32_t                  G,
    uint32_t                  m_سطر_بائٹیں,
    uint32_t                  kv_صفیں,
    uint32_t                  n_صفیں_q,
    ساخت hmx_fa_context *   factx
) {
    برائے (uint32_t g = 0; g < G; ++g) {
        ثابت uint32_t h_idx = kv_سر * G + g;
        ثابت uint32_t im2 = fastmodulo(h_idx, نقاب->ne[2], &factx->src3_div2);
        ثابت dma_addr_t ms_src = نقاب->ڈیٹا + q_آغاز * نقاب->nb[1] +
                                  im2 * نقاب->nb[2] + im3 * نقاب->nb[3] + kv_آغاز * حجم(__fp16);
        uint8_t * ms_dst = (uint8_t *) factx->vtcm_نقاب_buf + g * m_سطر_بائٹیں;
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(ms_dst, ms_src), G * m_سطر_بائٹیں, نقاب->nb[1], kv_صفیں * حجم(__fp16), n_صفیں_q);
    }
}

ساکن خالی fa_نکالو_نقاب_dma_gqa(dma_قطار * dma_q, uint32_t G) {
    برائے (uint32_t g = 0; g < G; ++g) {
        dma_قطار_نکالو(dma_q);
    }
}

ساکن اندرونی خالی fa_prefetch_بلاک(dma_قطار * dma_q, ثابت ساخت htp_tensor * k, ثابت ساخت htp_tensor * v, ثابت ساخت htp_tensor * نقاب,
                                     uint32_t b, size_t Bc, size_t سائز_k_صف_padded, size_t سائز_k_صف, size_t سائز_v_صف_padded, size_t سائز_v_صف,
                                     uint32_t ik2, uint32_t ik3, uint32_t iv2, uint32_t iv3, uint32_t q_آغاز, uint32_t im3, uint32_t kv_سر, uint32_t G,
                                     size_t m_سطر_بائٹیں, size_t n_صفیں_q, size_t nek1, size_t prefetch_buf, ساخت hmx_fa_context * factx) {
    ثابت uint32_t  prefetch_آغاز = b * Bc;
    ثابت uint32_t  prefetch_صفیں  = hex_smin(Bc, nek1 - prefetch_آغاز);
    ثابت dma_addr_t k_prefetch_src = k->ڈیٹا + prefetch_آغاز * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(factx->vtcm_k_fp16[prefetch_buf], k_prefetch_src), سائز_k_صف_padded, k->nb[1], سائز_k_صف, prefetch_صفیں);
    ثابت dma_addr_t v_prefetch_src = v->ڈیٹا + prefetch_آغاز * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(factx->vtcm_v_fp16[prefetch_buf], v_prefetch_src), سائز_v_صف_padded, v->nb[1], سائز_v_صف, prefetch_صفیں);

    اگر (نقاب) {
        اگر (__builtin_expect(factx->نقاب_نشر, true)) {
            ثابت dma_addr_t ms_src = نقاب->ڈیٹا + q_آغاز * نقاب->nb[1] + im3 * نقاب->nb[3] + prefetch_آغاز * حجم(__fp16);
            dma_cache_fa_ڈالو(dma_q, &factx->m_cache, ms_src, m_سطر_بائٹیں, نقاب->nb[1], prefetch_صفیں * حجم(__fp16), n_صفیں_q);
        } ورنہ {
            fa_ڈالو_نقاب_dma_gqa(dma_q, نقاب, q_آغاز, im3, prefetch_آغاز, kv_سر, G, m_سطر_بائٹیں, prefetch_صفیں, n_صفیں_q, factx);
        }
    }
}

// ============================================================================
// Core HMX flash attention algorithm (GQA-merged)
// ============================================================================

صحیح hmx_flash_attn_ext(ساخت htp_ops_context * octx) {
    ساخت htp_thread_trace * tr_hvx = &octx->ctx->trace[0];
    ساخت htp_thread_trace * tr_hmx = &octx->ctx->trace[HTP_MAX_NTHREADS];
    ثابت ساخت htp_tensor * q    = octx->src[0];
    ثابت ساخت htp_tensor * k    = octx->src[1];
    ثابت ساخت htp_tensor * v    = octx->src[2];
    ثابت ساخت htp_tensor * نقاب = (octx->src[3] && octx->src[3]->ڈیٹا) ? octx->src[3] : تہی;
    ثابت ساخت htp_tensor * dst  = octx->dst;

    ساخت htp_context * ثابت ctx = octx->ctx;

    اگر (!ctx->hmx_enabled) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    // Dimensions
    ثابت uint32_t neq0 = q->ne[0];  // head_dim (DK)
    ثابت uint32_t neq1 = q->ne[1];  // n_tokens
    ثابت uint32_t neq2 = q->ne[2];  // n_heads
    ثابت uint32_t neq3 = q->ne[3];  // n_seqs

    ثابت uint32_t nek0 = k->ne[0];  // head_dim
    ثابت uint32_t nek1 = k->ne[1];  // kv_len

    ثابت uint32_t nev0 = v->ne[0];  // head_dim (DV)

    ثابت uint32_t DK = neq0;
    ثابت uint32_t DV = nev0;

    // HMX tiles head_dim in units of 64. head_dim need not be 64- (or 32-) aligned:
    // we can operate on DK/DV rounded up to 64 with tail lanes [D, D_pad) zero-filled.
    ثابت uint32_t DK_pad = hex_باری_اوپر(DK, 64);
    ثابت uint32_t DV_pad = hex_باری_اوپر(DV, 64);
    اگر (DK == 0 || DV == 0) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت ساخت htp_fa_kernel_params * kparams = (ثابت ساخت htp_fa_kernel_params *) octx->kernel_params;
    ثابت uint32_t n_kv_heads = k->ne[2];

    // ======== Build context ========
    ساخت hmx_fa_context factx;
    memset(&factx, 0, حجم(factx));
    factx.octx           = octx;
    factx.sinks          = octx->src[4];  // NULL if this op has no attention sinks
    factx.k              = k;
    factx.v              = v;
    factx.n_دھاگے      = kparams->n_دھاگے;
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
    factx.n_kv_بلاکیں    = kparams->n_kv_بلاکیں;
    factx.is_q_fp32      = (q->type == HTP_TYPE_F32);
    factx.is_dst_fp32    = (dst->type == HTP_TYPE_F32);
    factx.pipeline       = (kparams->u.hmx.pipeline != 0);
    factx.نقاب_نشر = (kparams->u.hmx.نقاب_نشر != 0);
    اگر (نقاب) {
        factx.src3_div2  = kparams->src3_div2;
        factx.src3_div3  = kparams->src3_div3;
    }

    factx.has_softcap   = (kparams->logit_softcap != 0.0f);
    factx.scale         = (__fp16) kparams->scale;
    factx.max_bias      = kparams->max_bias;
    factx.logit_softcap = 0;
    اگر (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_سر_log2 = kparams->n_سر_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    ثابت uint32_t Br = factx.Br;
    ثابت uint32_t Bc = factx.Bc;
    ثابت uint32_t g_br = factx.g_br;
    ثابت bool pipeline = factx.pipeline;
    ثابت uint32_t n_دھاگے = factx.n_دھاگے;
    ثابت uint32_t G = factx.G;

    // Multi-device: prefer head-parallel partitioning (each core owns a disjoint head
    // shard), falling back to Q-block (token) split when heads don't divide evenly.
    ثابت uint32_t n_q_بلاکیں = (neq1 + Br - 1) / Br;
    uint32_t q_آغاز_min  = 0;
    uint32_t q_آغاز_max  = neq1;
    uint32_t kv_سر_min  = 0;
    uint32_t kv_سر_max  = n_kv_heads;

    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت uint32_t mdev_گنتی = octx->ctx->mdev.گنتی;
        ثابت uint32_t mdev_idx   = octx->ctx->mdev.idx;
        ثابت uint32_t dst_e_سائز = (dst->type == HTP_TYPE_F32) ? حجم(اعشاری) : حجم(__fp16);
        ثابت bool can_بانٹو      = htp_tensor_can_صف_تقسیم(dst, dst_e_سائز);

        اگر (kparams->سر_بانٹو && can_بانٹو && n_kv_heads >= mdev_گنتی && n_kv_heads % mdev_گنتی == 0) {
            ثابت uint32_t kv_فی_core = n_kv_heads / mdev_گنتی;
            kv_سر_min = mdev_idx * kv_فی_core;
            kv_سر_max = kv_سر_min + kv_فی_core;
        } ورنہ {
            ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(n_q_بلاکیں, can_بانٹو ? 1 : 0, mdev_idx, mdev_گنتی, &octx->ctx->mdev.گنتی_div);
            ثابت uint32_t بلاک_آغاز = range.آغاز;
            ثابت uint32_t بلاک_انتہا   = range.آغاز + range.گنتی;

            اگر (بلاک_آغاز >= بلاک_انتہا) {
                واپس HTP_حالت_OK;
            }

            q_آغاز_min = بلاک_آغاز * Br;
            q_آغاز_max = MIN(بلاک_انتہا * Br, neq1);
        }
    }

    // ======== VTCM allocation (GQA-aware) ========
    // K/V row sizes drive the DMA descriptors (not the VTCM layout) and are used
    // throughout the KV loop below. The DMA copies only the real DK/DV columns; the
    // staging rows are padded to hold DK_pad/DV_pad columns (tail zero-filled below)
    // so the HMX interleave/tile logic can operate on 64-aligned head dims.
    ثابت size_t سائز_k_صف        = htp_tensor_get_صف_سائز(k->type, DK);
    ثابت size_t سائز_v_صف        = htp_tensor_get_صف_سائز(v->type, DV);
    ثابت size_t سائز_k_صف_padded = hex_باری_اوپر(DK_pad * حجم(__fp16), 128);
    ثابت size_t سائز_v_صف_padded = hex_باری_اوپر(DV_pad * حجم(__fp16), 128);

    // Build the VTCM layout once (shared with the host estimator) and place every
    // scratch buffer at its computed offset. Padded head dims size the HMX tiles.
    ساخت hmx_fa_vtcm_layout L;
    hmx_fa_vtcm_layout_build(&L, G, DK_pad, DV_pad, Br, Bc, n_دھاگے, pipeline, factx.is_q_fp32, factx.sinks != تہی, factx.n_heads);

    اگر (L.کل_بائٹیں > ctx->vtcm_سائز) {
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * ثابت بنیاد = ctx->vtcm_بنیاد;

    factx.vtcm_q_dma          = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_q_dma);
    factx.vtcm_q_tiles        = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_q_tiles);
    factx.vtcm_o_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_o_tiles[0]);
    factx.vtcm_o_tiles[1]     = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_o_tiles[1]);
    factx.vtcm_k_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_k_fp16[0]);
    factx.vtcm_k_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_k_fp16[1]);
    factx.vtcm_v_fp16[0]      = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_v_fp16[0]);
    factx.vtcm_v_fp16[1]      = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_v_fp16[1]);
    factx.vtcm_k_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_k_tiles[0]);
    factx.vtcm_k_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, بنیاد, L.off_k_tiles[1], pipeline);
    factx.vtcm_v_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_v_tiles[0]);
    factx.vtcm_v_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, بنیاد, L.off_v_tiles[1], pipeline);
    factx.vtcm_s_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_s_tiles[0]);
    factx.vtcm_s_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, بنیاد, L.off_s_tiles[1], pipeline);
    factx.vtcm_p_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_p_tiles[0]);
    factx.vtcm_p_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, بنیاد, L.off_p_tiles[1], pipeline);
    factx.vtcm_d_tiles[0]     = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_d_tiles[0]);
    factx.vtcm_d_tiles[1]     = VTCM_LAYOUT_PTR_OPTIONAL(__fp16, بنیاد, L.off_d_tiles[1], pipeline);
    factx.vtcm_d_inv_l        = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_d_inv_l);
    factx.vtcm_m_vec          = VTCM_LAYOUT_PTR(HVX_سمتیہ, بنیاد, L.off_m_vec);
    factx.vtcm_l_vec          = VTCM_LAYOUT_PTR(HVX_سمتیہ, بنیاد, L.off_l_vec);
    factx.vtcm_s_rowmax       = VTCM_LAYOUT_PTR(HVX_سمتیہ, بنیاد, L.off_s_rowmax);
    factx.vtcm_p_rowsum       = VTCM_LAYOUT_PTR(HVX_سمتیہ, بنیاد, L.off_p_rowsum);
    factx.vtcm_صف_bufs       = VTCM_LAYOUT_PTR(HVX_سمتیہ, بنیاد, L.off_صف_bufs);
    factx.صف_buf_stride      = L.صف_buf_stride;
    factx.vtcm_hmx_scales_id  = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_hmx_scales_id);
    factx.vtcm_hmx_scales_qk  = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_hmx_scales_qk);
    factx.vtcm_نقاب_buf       = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_نقاب_buf);
    factx.نقاب_buf_صف_stride = L.نقاب_buf_صف_stride;
    factx.q_tile_بائٹیں        = L.q_tile_بائٹیں;
    factx.k_tile_بائٹیں        = L.k_tile_بائٹیں;
    factx.o_tile_بائٹیں        = L.o_tile_بائٹیں;
    factx.ستون_vec_بائٹیں       = L.ستون_vec_بائٹیں;
    factx.d_tile_بائٹیں        = L.d_tile_بائٹیں;
    factx.vtcm_slopes         = VTCM_LAYOUT_PTR(__fp16, بنیاد, L.off_slopes);
    factx.vtcm_sinks          = VTCM_LAYOUT_PTR_OPTIONAL(اعشاری, بنیاد, L.off_sinks, factx.sinks != تہی);

    ثابت size_t m_سطر_بائٹیں = L.m_سطر_بائٹیں;  // used by the mask DMAs in the KV loop

    dma_cache_fa_init(&factx.m_cache, (uint8_t *) factx.vtcm_نقاب_buf, L.m_buf_خانہ_بائٹیں, HMX_FA_DMA_CACHE_سائز);

    // Head-dim padding: the K/V DMA staging buffers and the flat-Q buffer are laid out
    // with padded row strides (size_{k,v,q}_row_padded, covering D_pad columns) but the
    // DMA only writes the real D columns per row. Zero the whole staging buffers once up
    // front so tail lanes [D, D_pad) stay zero for all KV blocks. No-op when already aligned.
    اگر (DK_pad != DK || DV_pad != DV) {
        ثابت size_t k_buf_بائٹیں = (size_t) factx.Bc * سائز_k_صف_padded;
        ثابت size_t v_buf_بائٹیں = (size_t) factx.Bc * سائز_v_صف_padded;
        hvx_splat_u8_a((حرف *) factx.vtcm_k_fp16[0], 0, k_buf_بائٹیں);
        hvx_splat_u8_a((حرف *) factx.vtcm_k_fp16[1], 0, k_buf_بائٹیں);
        hvx_splat_u8_a((حرف *) factx.vtcm_v_fp16[0], 0, v_buf_بائٹیں);
        hvx_splat_u8_a((حرف *) factx.vtcm_v_fp16[1], 0, v_buf_بائٹیں);
        // Flat-Q DMA scratch
        اگر (factx.vtcm_q_dma) {
            ثابت size_t q_dma_بائٹیں = hex_align_اوپر(factx.g_br * DK * (factx.is_q_fp32 ? حجم(اعشاری) : حجم(__fp16)), 128);
            hvx_splat_u8_a((حرف *) factx.vtcm_q_dma, 0, q_dma_بائٹیں);
        }
    }

    // ======== Initialize HMX output scales ========
    hmx_init_عمود_scales(factx.vtcm_hmx_scales_id, Q6_V_vsplat_R(0x3c00)); // 1.0
    {
        اتحاد { __fp16 f; uint16_t u; } sc = { .f = factx.scale };
        hmx_init_عمود_scales(factx.vtcm_hmx_scales_qk, Q6_V_vsplat_R(sc.u));
    }

    // ======== DMA setup ========
    dma_قطار * ثابت dma_q = ctx->dma[0];

    اگر (factx.sinks) {
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(factx.vtcm_sinks, factx.sinks->ڈیٹا), L.sinks_بائٹیں, 0, factx.sinks->سائز, 1);
        dma_قطار_نکالو(dma_q);
    }

    ثابت size_t n_صف_tiles_g_br = g_br / HMX_FP16_TILE_N_صفیں;
    ثابت size_t n_tiles_فی_bc   = Bc / HMX_FP16_TILE_N_ستون_تعداد;

    ثابت size_t qo_element_سائز = factx.is_q_fp32 ? حجم(اعشاری) : حجم(__fp16);

    ثابت bool q_transposed                 = q->nb[1] < q->nb[2];
    ثابت size_t q_src_stride               = q_transposed ? q->nb[2] : q->nb[1];
    ثابت size_t q_صف_بائٹیں_untransposed   = factx.G * factx.DK * qo_element_سائز;
    ثابت size_t q_صف_بائٹیں_trans_ضربی_عدد   = factx.DK * qo_element_سائز;

    // ======== Reusable job descriptors for pipeline ========
    hmx_fa_qk_کام_t       qk_کام;
    hmx_fa_o_update_کام_t ou_کام;
    hmx_fa_o_norm_کام_t   on_کام;

    // ======== Main loop ========
    برائے (uint32_t ib3 = 0; ib3 < neq3; ++ib3) {
        ثابت uint32_t im3 = نقاب ? fastmodulo(ib3, نقاب->ne[3], &factx.src3_div3) : 0;
        برائے (uint32_t q_آغاز = q_آغاز_min; q_آغاز < q_آغاز_max; q_آغاز += Br) {
            ثابت uint32_t n_صفیں_q    = hex_smin(Br, neq1 - q_آغاز);
            ثابت size_t   n_صفیں_g    = n_صفیں_q * G;
            ثابت size_t   g_br_actual = hex_align_اوپر(n_صفیں_g, HMX_FP16_TILE_N_صفیں);
            ثابت size_t   n_صف_tiles = g_br_actual / HMX_FP16_TILE_N_صفیں;

            uint16_t فعال_بلاکیں[FA_MAX_KV_بلاکیں];
            uint32_t n_فعال_بلاکیں = factx.n_kv_بلاکیں;
            اگر (نقاب && factx.نقاب_نشر && factx.n_kv_بلاکیں > 1) {
                ثابت size_t scratch_بائٹیں = factx.k_tile_بائٹیں * (factx.pipeline ? 2 : 1);
                n_فعال_بلاکیں = fa_جائزہ_فعال_بلاکیں(
                    dma_q, نقاب, q_آغاز, n_صفیں_q, 0, im3, nek1, Bc,
                    factx.n_kv_بلاکیں, factx.vtcm_k_tiles[0], scratch_بائٹیں, فعال_بلاکیں);
            } ورنہ {
                #ہدایت clang loop vectorize(disable)
                برائے (uint32_t b = 0; b < factx.n_kv_بلاکیں; ++b) {
                    فعال_بلاکیں[b] = (uint16_t) b;
                }
            }
            ثابت uint32_t b0       = (n_فعال_بلاکیں > 0) ? فعال_بلاکیں[0] : 0;
            ثابت uint32_t b0_آغاز = b0 * Bc;
            ثابت uint32_t b0_صفیں  = hex_smin(Bc, nek1 - b0_آغاز);

            برائے (uint32_t kv_سر = kv_سر_min; kv_سر < kv_سر_max; ++kv_سر) {
                ثابت uint32_t ik2 = kv_سر;
                ثابت uint32_t ik3 = fastdiv(ib3, &kparams->نشر_rk3);
                ثابت uint32_t iv2 = kv_سر;
                ثابت uint32_t iv3 = fastdiv(ib3, &kparams->نشر_rv3);

                // 1. Push Q and KV DMAs for the first kv_head of this q_start block.
                // Subsequent kv_head iterations are enqueued early at the end of the previous iteration.
                اگر (kv_سر == kv_سر_min) {
                    ثابت dma_addr_t q_ptr = q->ڈیٹا + q_آغاز * q->nb[1] +
                                            (kv_سر * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    ثابت size_t q_صف_بائٹیں = q_transposed ? n_صفیں_q * q_صف_بائٹیں_trans_ضربی_عدد : q_صف_بائٹیں_untransposed;
                    ثابت size_t n_صفیں      = q_transposed ? factx.G : n_صفیں_q;
                    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(factx.vtcm_q_dma, q_ptr), q_صف_بائٹیں, hex_smax(q_src_stride, q_صف_بائٹیں), q_صف_بائٹیں, n_صفیں);

                    اگر (n_فعال_بلاکیں > 0) {
                        ثابت dma_addr_t k_src = k->ڈیٹا + b0_آغاز * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(factx.vtcm_k_fp16[0], k_src), سائز_k_صف_padded, k->nb[1], سائز_k_صف, b0_صفیں);

                        ثابت dma_addr_t v_src = v->ڈیٹا + b0_آغاز * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(factx.vtcm_v_fp16[0], v_src), سائز_v_صف_padded, v->nb[1], سائز_v_صف, b0_صفیں);

                        اگر (factx.pipeline && نقاب) {
                            اگر (__builtin_expect(factx.نقاب_نشر, true)) {
                                ثابت dma_addr_t ms_src = نقاب->ڈیٹا + q_آغاز * نقاب->nb[1] + im3 * نقاب->nb[3] + b0_آغاز * حجم(__fp16);
                                dma_cache_fa_ڈالو(dma_q, &factx.m_cache, ms_src, m_سطر_بائٹیں, نقاب->nb[1], b0_صفیں * حجم(__fp16), n_صفیں_q);
                            } ورنہ {
                                fa_ڈالو_نقاب_dma_gqa(dma_q, نقاب, q_آغاز, im3, b0_آغاز, kv_سر, G, m_سطر_بائٹیں, b0_صفیں, n_صفیں_q, &factx);
                            }
                        }
                    }
                }

                // 2. Pop Q DMA (blocks until Q is loaded)
                dma_قطار_نکالو(dma_q);

                // ---- Load Q block & Initialize per-block state ----
                fa_phase_q_بھرو(&factx, q, q_آغاز, kv_سر, ib3, n_صفیں_g);

                __fp16 * o_tile_پچھلا = factx.vtcm_o_tiles[0];
                __fp16 * o_tile_curr = factx.vtcm_o_tiles[1];

                // ---- KV block loop with DMA double-buffering ----
                size_t buf_idx = 0;

                htp_trace_event_آغاز(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_آغاز);
                fa_compute_slopes(&factx, kv_سر, n_صفیں_g);
                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_A_PREP, (uint16_t) q_آغاز);

                ثابت size_t k_src_stride = سائز_k_صف_padded / حجم(__fp16);
                ثابت size_t v_src_stride = سائز_v_صف_padded / حجم(__fp16);

                hmx_قطار_t hmx_q = ctx->hmx_قطار;

                اگر (factx.pipeline) {
                    // Double-buffered job structs because HMX queue runs asynchronously
                    hmx_fa_qk_کام_t qk_کام[2];
                    hmx_fa_o_update_کام_t ou_کام[2];

                    // Prefetch second active block early if there are multiple active blocks
                    اگر (n_فعال_بلاکیں > 1) {
                        fa_prefetch_بلاک(dma_q, k, v, نقاب, فعال_بلاکیں[1], Bc, سائز_k_صف_padded, سائز_k_صف, سائز_v_صف_padded, سائز_v_صف,
                                          ik2, ik3, iv2, iv3, q_آغاز, im3, kv_سر, G, m_سطر_بائٹیں, n_صفیں_q, nek1, 1, &factx);
                    }

                    // Prep and start QK-dot(0)
                    خالی * curr_k0 = (خالی *) dma_قطار_نکالو(dma_q).dst;
                    fa_phase_k_interleave(&factx, b0_صفیں, k_src_stride, curr_k0, b0_آغاز, 0);

                    qk_کام[0].q_tiles        = factx.vtcm_q_tiles;
                    qk_کام[0].k_tiles        = factx.vtcm_k_tiles[0];
                    qk_کام[0].s_tiles        = factx.vtcm_s_tiles[0];
                    qk_کام[0].n_صف_tiles    = n_صف_tiles;
                    qk_کام[0].n_ستون_tiles    = hmx_ceil_div(b0_صفیں, HMX_FP16_TILE_N_ستون_تعداد);
                    qk_کام[0].n_dot_tiles    = DK_pad / 32;
                    qk_کام[0].n_tiles_فی_bc = n_tiles_فی_bc;
                    qk_کام[0].hmx_scales     = factx.vtcm_hmx_scales_qk;
                    hmx_قطار_ڈالو(hmx_q, hmx_قطار_make_desc(hmx_fa_qk_dot_کارکن, &qk_کام[0]));

                    برائے (uint32_t قدم = 0; قدم < n_فعال_بلاکیں; ++قدم) {
                        ثابت uint32_t kv_blk      = فعال_بلاکیں[قدم];
                        ثابت uint32_t kv_آغاز    = kv_blk * Bc;
                        ثابت uint32_t kv_صفیں     = hex_smin(Bc, nek1 - kv_آغاز);
                        ثابت size_t   n_ستون_tiles = hmx_ceil_div(kv_صفیں, HMX_FP16_TILE_N_ستون_تعداد);

                        // ---- 1. Pop and run V-prep for current block ----
                        خالی * curr_v = (خالی *) dma_قطار_نکالو(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_صفیں, v_src_stride, curr_v, factx.vtcm_v_tiles[buf_idx], n_tiles_فی_bc, kv_آغاز);

                        // ---- 2. Pop and run mask-prep for current block ----
                        __fp16 * current_نقاب_vtcm = تہی;
                        اگر (نقاب) {
                            اگر (__builtin_expect(factx.نقاب_نشر, true)) {
                                current_نقاب_vtcm = (__fp16 *) dma_قطار_نکالو(dma_q).dst;
                            } ورنہ {
                                fa_نکالو_نقاب_dma_gqa(dma_q, G);
                                current_نقاب_vtcm = factx.vtcm_نقاب_buf;
                            }
                        }

                        // ---- 3. Start HMX O update for step - 1 (reads P[1 - buf_idx], V[1 - buf_idx], D) ----
                        // O update relies on the previous block's P and V tiles.
                        // O update MUST be pushed before the next block's QK-dot: hmx_queue_pop() retires the
                        // oldest descriptor, so push order alone decides which pop waits for which job.
                        // If OU went in after QK(i+1), the pop below would retire QK(i+1) and leave
                        // OU(i-1) in flight into the next iteration, where V-prep overwrites V[prev_buf].
                        اگر (قدم > 0) {
                            ثابت uint32_t پچھلا_blk      = فعال_بلاکیں[قدم - 1];
                            ثابت size_t پچھلا_buf        = 1 - buf_idx;
                            ou_کام[پچھلا_buf].o_curr      = o_tile_curr;
                            ou_کام[پچھلا_buf].o_پچھلا      = o_tile_پچھلا;
                            ou_کام[پچھلا_buf].p_tiles     = factx.vtcm_p_tiles[پچھلا_buf];
                            ou_کام[پچھلا_buf].v_tiles     = factx.vtcm_v_tiles[پچھلا_buf];
                            ou_کام[پچھلا_buf].d_tiles     = factx.vtcm_d_tiles[پچھلا_buf];
                            ou_کام[پچھلا_buf].hmx_scales  = factx.vtcm_hmx_scales_id;
                            ou_کام[پچھلا_buf].n_صف_tiles = n_صف_tiles;
                            ou_کام[پچھلا_buf].n_ستون_tiles =
                                hmx_ceil_div(hex_smin(Bc, nek1 - پچھلا_blk * Bc), HMX_FP16_TILE_N_ستون_تعداد);
                            ou_کام[پچھلا_buf].n_صف_tiles_g_br = n_صف_tiles_g_br;
                            ou_کام[پچھلا_buf].n_tiles_فی_bc   = n_tiles_فی_bc;
                            ou_کام[پچھلا_buf].DV               = DV_pad;
                            hmx_قطار_ڈالو(hmx_q, hmx_قطار_make_desc(hmx_fa_o_update_کارکن, &ou_کام[پچھلا_buf]));
                        }

                        // ---- 4. Pop and run K-prep for next active block & push next QK-dot ----
                        اگر (قدم + 1 < n_فعال_بلاکیں) {
                            ثابت uint32_t اگلا_blk   = فعال_بلاکیں[قدم + 1];
                            ثابت uint32_t اگلا_آغاز = اگلا_blk * Bc;
                            ثابت uint32_t اگلا_صفیں  = hex_smin(Bc, nek1 - اگلا_آغاز);
                            ثابت size_t   اگلا_buf   = 1 - buf_idx;

                            خالی * اگلا_k = (خالی *) dma_قطار_نکالو(dma_q).dst;
                            fa_phase_k_interleave(&factx, اگلا_صفیں, k_src_stride, اگلا_k, اگلا_آغاز, اگلا_buf);

                            qk_کام[اگلا_buf].q_tiles        = factx.vtcm_q_tiles;
                            qk_کام[اگلا_buf].k_tiles        = factx.vtcm_k_tiles[اگلا_buf];
                            qk_کام[اگلا_buf].s_tiles        = factx.vtcm_s_tiles[اگلا_buf];
                            qk_کام[اگلا_buf].n_صف_tiles    = n_صف_tiles;
                            qk_کام[اگلا_buf].n_ستون_tiles    = hmx_ceil_div(اگلا_صفیں, HMX_FP16_TILE_N_ستون_تعداد);
                            qk_کام[اگلا_buf].n_dot_tiles    = DK_pad / 32;
                            qk_کام[اگلا_buf].n_tiles_فی_bc = n_tiles_فی_bc;
                            qk_کام[اگلا_buf].hmx_scales     = factx.vtcm_hmx_scales_qk;
                            hmx_قطار_ڈالو(hmx_q, hmx_قطار_make_desc(hmx_fa_qk_dot_کارکن, &qk_کام[اگلا_buf]));
                        }

                        // ---- 5. Wait for current block's QK-dot to finish ----
                        hmx_قطار_نکالو(hmx_q);

                        // ---- 6. Phase 2: softmax + build_D ----
                        fa_softmax_دلائل_t sargs;
                        memset(&sargs, 0, حجم(sargs));
                        sargs.factx                = &factx;
                        sargs.buf_idx              = buf_idx;
                        sargs.kv_صفیں              = kv_صفیں;
                        sargs.n_صفیں_g             = n_صفیں_g;
                        sargs.n_ستون_tiles          = n_ستون_tiles;
                        sargs.n_tiles_فی_bc       = n_tiles_فی_bc;
                        sargs.n_صف_tiles          = n_صف_tiles;
                        sargs.n_صف_tiles_g_br     = n_صف_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_سر              = kv_سر;
                        sargs.kv_آغاز             = kv_آغاز;
                        sargs.q_آغاز              = q_آغاز;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.نقاب                 = نقاب;
                        sargs.نقاب_vtcm            = current_نقاب_vtcm;
                        sargs.نقاب_vtcm_صف_stride = factx.نقاب_buf_صف_stride;
                        sargs.slopes               = factx.vtcm_slopes;

                        // Run Softmax on HVX (blocking call)
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_صف_tiles, n_صف_tiles_g_br);

                        // Wait for HMX O update for step - 1 to finish
                        اگر (قدم > 0) {
                            hmx_قطار_نکالو(hmx_q);
                            hex_تبادلہ_ptr((خالی **) &o_tile_curr, (خالی **) &o_tile_پچھلا);
                        }

                        // Prefetch block step + 2
                        اگر (قدم + 2 < n_فعال_بلاکیں) {
                            fa_prefetch_بلاک(dma_q, k, v, نقاب, فعال_بلاکیں[قدم + 2], Bc, سائز_k_صف_padded, سائز_k_صف, سائز_v_صف_padded, سائز_v_صف,
                                              ik2, ik3, iv2, iv3, q_آغاز, im3, kv_سر, G, m_سطر_بائٹیں, n_صفیں_q, nek1, buf_idx, &factx);
                        }

                        buf_idx = 1 - buf_idx;
                    }

                    // Epilogue
                    اگر (n_فعال_بلاکیں > 0) {
                        ثابت uint32_t آخری_blk = فعال_بلاکیں[n_فعال_بلاکیں - 1];
                        ثابت size_t آخری_ستون_تعداد  = hmx_ceil_div(hex_smin(Bc, nek1 - آخری_blk * Bc), HMX_FP16_TILE_N_ستون_تعداد);
                        ou_کام[0].o_curr           = o_tile_curr;
                        ou_کام[0].o_پچھلا           = o_tile_پچھلا;
                        ou_کام[0].p_tiles          = factx.vtcm_p_tiles[1 - buf_idx];
                        ou_کام[0].v_tiles          = factx.vtcm_v_tiles[1 - buf_idx];
                        ou_کام[0].d_tiles          = factx.vtcm_d_tiles[1 - buf_idx];
                        ou_کام[0].hmx_scales       = factx.vtcm_hmx_scales_id;
                        ou_کام[0].n_صف_tiles      = n_صف_tiles;
                        ou_کام[0].n_ستون_tiles      = آخری_ستون_تعداد;
                        ou_کام[0].n_صف_tiles_g_br = n_صف_tiles_g_br;
                        ou_کام[0].n_tiles_فی_bc   = n_tiles_فی_bc;
                        ou_کام[0].DV               = DV_pad;
                        hmx_قطار_ڈالو(hmx_q, hmx_قطار_make_desc(hmx_fa_o_update_کارکن, &ou_کام[0]));

                        // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                        htp_trace_event_آغاز(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_آغاز);
                        fa_build_d_diag_inv_l(&factx, n_صف_tiles, n_صف_tiles_g_br, kv_سر, n_صفیں_g);
                        htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_آغاز);
                        hmx_قطار_نکالو(hmx_q);

                        hex_تبادلہ_ptr((خالی **) &o_tile_curr, (خالی **) &o_tile_پچھلا);
                    }

                } ورنہ {
                    // Fallback path
                    برائے (uint32_t قدم = 0; قدم < n_فعال_بلاکیں; ++قدم) {
                        ثابت uint32_t kv_blk      = فعال_بلاکیں[قدم];
                        ثابت uint32_t kv_آغاز    = kv_blk * Bc;
                        ثابت uint32_t kv_صفیں     = hex_smin(Bc, nek1 - kv_آغاز);
                        ثابت size_t   n_ستون_tiles = hmx_ceil_div(kv_صفیں, HMX_FP16_TILE_N_ستون_تعداد);

                        اگر (نقاب) {
                            اگر (__builtin_expect(factx.نقاب_نشر, true)) {
                                ثابت dma_addr_t ms_src = نقاب->ڈیٹا + q_آغاز * نقاب->nb[1] + im3 * نقاب->nb[3] + kv_آغاز * حجم(__fp16);
                                dma_cache_fa_ڈالو(dma_q, &factx.m_cache, ms_src, m_سطر_بائٹیں, نقاب->nb[1], kv_صفیں * حجم(__fp16), n_صفیں_q);
                            } ورنہ {
                                fa_ڈالو_نقاب_dma_gqa(dma_q, نقاب, q_آغاز, im3, kv_آغاز, kv_سر, G, m_سطر_بائٹیں, kv_صفیں, n_صفیں_q, &factx);
                            }
                        }

                        اگر (قدم + 1 < n_فعال_بلاکیں) {
                            ثابت uint32_t  prefetch_blk   = فعال_بلاکیں[قدم + 1];
                            ثابت uint32_t  prefetch_آغاز = prefetch_blk * Bc;
                            ثابت uint32_t  prefetch_صفیں  = hex_smin(Bc, nek1 - prefetch_آغاز);
                            ثابت size_t    prefetch_buf   = 1 - buf_idx;
                            ثابت dma_addr_t k_prefetch_src = k->ڈیٹا + prefetch_آغاز * k->nb[1] + ik2 * k->nb[2] + ik3 * k->nb[3];
                            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(factx.vtcm_k_fp16[prefetch_buf], k_prefetch_src), سائز_k_صف_padded, k->nb[1], سائز_k_صف, prefetch_صفیں);
                            ثابت dma_addr_t v_prefetch_src = v->ڈیٹا + prefetch_آغاز * v->nb[1] + iv2 * v->nb[2] + iv3 * v->nb[3];
                            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(factx.vtcm_v_fp16[prefetch_buf], v_prefetch_src), سائز_v_صف_padded, v->nb[1], سائز_v_صف, prefetch_صفیں);
                        }

                        // Wait for current K DMA and interleave
                        خالی * curr_k = (خالی *) dma_قطار_نکالو(dma_q).dst;
                        fa_phase_k_interleave(&factx, kv_صفیں, k_src_stride, curr_k, kv_آغاز, 0);

                        {
                            qk_کام.q_tiles        = factx.vtcm_q_tiles;
                            qk_کام.k_tiles        = factx.vtcm_k_tiles[0];
                            qk_کام.s_tiles        = factx.vtcm_s_tiles[0];
                            qk_کام.n_صف_tiles    = n_صف_tiles;
                            qk_کام.n_ستون_tiles    = n_ستون_tiles;
                            qk_کام.n_dot_tiles    = (size_t) (DK_pad / 32);
                            qk_کام.n_tiles_فی_bc = n_tiles_فی_bc;
                            qk_کام.hmx_scales     = factx.vtcm_hmx_scales_qk;

                            hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_fa_qk_dot_کارکن, &qk_کام));
                            hmx_قطار_نکالو(ctx->hmx_قطار);
                        }

                        // Wait for current V DMA and interleave
                        خالی * curr_v = (خالی *) dma_قطار_نکالو(dma_q).dst;
                        fa_phase_v_interleave(&factx, kv_صفیں, v_src_stride, curr_v, factx.vtcm_v_tiles[0], n_tiles_فی_bc, kv_آغاز);

                        // ---- Phase 3: softmax + build_D ----
                        __fp16 * current_نقاب_vtcm = تہی;
                        اگر (نقاب) {
                            اگر (__builtin_expect(factx.نقاب_نشر, true)) {
                                current_نقاب_vtcm = (__fp16 *) dma_قطار_نکالو(dma_q).dst;
                            } ورنہ {
                                fa_نکالو_نقاب_dma_gqa(dma_q, G);
                                current_نقاب_vtcm = factx.vtcm_نقاب_buf;
                            }
                        }

                        fa_softmax_دلائل_t sargs;
                        memset(&sargs, 0, حجم(sargs));
                        sargs.factx                = &factx;
                        sargs.kv_صفیں              = kv_صفیں;
                        sargs.n_صفیں_g             = n_صفیں_g;
                        sargs.n_ستون_tiles          = n_ستون_tiles;
                        sargs.n_tiles_فی_bc       = n_tiles_فی_bc;
                        sargs.n_صف_tiles          = n_صف_tiles;
                        sargs.n_صف_tiles_g_br     = n_صف_tiles_g_br;
                        sargs.Bc                   = Bc;
                        sargs.G                    = G;
                        sargs.kv_سر              = kv_سر;
                        sargs.kv_آغاز             = kv_آغاز;
                        sargs.q_آغاز              = q_آغاز;
                        sargs.ib3                  = ib3;
                        sargs.has_alibi            = (factx.max_bias != 0.0f);
                        sargs.نقاب                 = نقاب;
                        sargs.نقاب_vtcm            = current_نقاب_vtcm;
                        sargs.نقاب_vtcm_صف_stride = factx.نقاب_buf_صف_stride;
                        sargs.slopes               = factx.vtcm_slopes;
                        fa_phase_softmax_and_build_d(&factx, &sargs, n_صف_tiles, n_صف_tiles_g_br);

                        {
                            ou_کام.o_curr           = o_tile_curr;
                            ou_کام.o_پچھلا           = o_tile_پچھلا;
                            ou_کام.p_tiles          = factx.vtcm_p_tiles[0];
                            ou_کام.v_tiles          = factx.vtcm_v_tiles[0];
                            ou_کام.d_tiles          = factx.vtcm_d_tiles[0];
                            ou_کام.hmx_scales       = factx.vtcm_hmx_scales_id;
                            ou_کام.n_صف_tiles      = n_صف_tiles;
                            ou_کام.n_ستون_tiles      = n_ستون_tiles;
                            ou_کام.n_صف_tiles_g_br = n_صف_tiles_g_br;
                            ou_کام.n_tiles_فی_bc   = n_tiles_فی_bc;
                            ou_کام.DV               = DV_pad;

                            hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_fa_o_update_کارکن, &ou_کام));
                            اگر (قدم + 1 == n_فعال_بلاکیں) {
                                // Overlapped: run HVX build diag inv L while HMX is busy executing the update
                                htp_trace_event_آغاز(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_آغاز);
                                fa_build_d_diag_inv_l(&factx, n_صف_tiles, n_صف_tiles_g_br, kv_سر, n_صفیں_g);
                                htp_trace_event_stop(tr_hvx, HTP_TRACE_EVT_HVX_O_PROC, (uint16_t) q_آغاز);
                            }
                            hmx_قطار_نکالو(ctx->hmx_قطار);

                            hex_تبادلہ_ptr((خالی **) &o_tile_curr, (خالی **) &o_tile_پچھلا);
                        }

                        buf_idx = 1 - buf_idx;
                    }
                }

                // Enqueue DMAs for the next kv_head early so they overlap with O-PROC
                ثابت uint32_t اگلا_kv_سر = kv_سر + 1;
                ثابت bool has_اگلا = (اگلا_kv_سر < kv_سر_max);

                اگر (has_اگلا) {
                    ثابت dma_addr_t اگلا_q_ptr = q->ڈیٹا + q_آغاز * q->nb[1] + (اگلا_kv_سر * factx.G) * q->nb[2] + ib3 * q->nb[3];
                    ثابت size_t اگلا_q_صف_بائٹیں = q_transposed ? n_صفیں_q * q_صف_بائٹیں_trans_ضربی_عدد : q_صف_بائٹیں_untransposed;
                    ثابت size_t اگلا_n_صفیں      = q_transposed ? factx.G : n_صفیں_q;
                    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(factx.vtcm_q_dma, اگلا_q_ptr), اگلا_q_صف_بائٹیں, hex_smax(q_src_stride, اگلا_q_صف_بائٹیں), اگلا_q_صف_بائٹیں, اگلا_n_صفیں);

                    اگر (n_فعال_بلاکیں > 0) {
                        ثابت uint32_t اگلا_ik2 = اگلا_kv_سر;
                        ثابت uint32_t اگلا_iv2 = اگلا_kv_سر;

                        ثابت dma_addr_t اگلا_k_src = k->ڈیٹا + b0_آغاز * k->nb[1] + اگلا_ik2 * k->nb[2] + ik3 * k->nb[3];
                        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(factx.vtcm_k_fp16[0], اگلا_k_src), سائز_k_صف_padded, k->nb[1], سائز_k_صف, b0_صفیں);

                        ثابت dma_addr_t اگلا_v_src = v->ڈیٹا + b0_آغاز * v->nb[1] + اگلا_iv2 * v->nb[2] + iv3 * v->nb[3];
                        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(factx.vtcm_v_fp16[0], اگلا_v_src), سائز_v_صف_padded, v->nb[1], سائز_v_صف, b0_صفیں);

                        اگر (factx.pipeline && نقاب) {
                            اگر (__builtin_expect(factx.نقاب_نشر, true)) {
                                ثابت dma_addr_t ms_src = نقاب->ڈیٹا + q_آغاز * نقاب->nb[1] + im3 * نقاب->nb[3] + b0_آغاز * حجم(__fp16);
                                dma_cache_fa_ڈالو(dma_q, &factx.m_cache, ms_src, m_سطر_بائٹیں, نقاب->nb[1], b0_صفیں * حجم(__fp16), n_صفیں_q);
                            } ورنہ {
                                fa_ڈالو_نقاب_dma_gqa(dma_q, نقاب, q_آغاز, im3, b0_آغاز, اگلا_kv_سر, G, m_سطر_بائٹیں, b0_صفیں, n_صفیں_q, &factx);
                            }
                        }
                    }
                }

                // ---- Final normalization ----
                {
                    on_کام.o_curr           = o_tile_curr;
                    on_کام.o_پچھلا           = o_tile_پچھلا;
                    on_کام.d_tiles          = factx.vtcm_d_inv_l;
                    on_کام.hmx_scales       = factx.vtcm_hmx_scales_id;
                    on_کام.n_صف_tiles      = n_صف_tiles;
                    on_کام.n_صف_tiles_g_br = n_صف_tiles_g_br;
                    on_کام.DV               = DV_pad;
                    hmx_قطار_ڈالو(ctx->hmx_قطار, hmx_قطار_make_desc(hmx_fa_o_norm_کارکن, &on_کام));
                    hmx_قطار_نکالو(ctx->hmx_قطار);
                }

                // ---- Store O block ----
                fa_phase_o_store(&factx, dst, o_tile_curr, q_آغاز, kv_سر, ib3, n_صفیں_g);
            }
        }
    }

    واپس HTP_حالت_OK;
}

صحیح op_flash_attn_ext(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * q    = octx->src[0];
    ثابت ساخت htp_tensor * k    = octx->src[1];
    ثابت ساخت htp_tensor * v    = octx->src[2];
    ثابت ساخت htp_tensor * نقاب = octx->src[3];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    // Check support
    اگر ((q->type != HTP_TYPE_F16 && q->type != HTP_TYPE_F32) ||
        (k->type != HTP_TYPE_F16 && k->type != HTP_TYPE_Q8_0) ||
        (v->type != HTP_TYPE_F16 && v->type != HTP_TYPE_Q8_0)) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    اگر (htp_tensor_is_extended(dst)) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت ساخت htp_fa_kernel_params * kparams = (ثابت ساخت htp_fa_kernel_params *) octx->kernel_params;

    اگر (kparams->kernel_type == HTP_FA_KERNEL_UNSUPPORTED || kparams->n_kv_بلاکیں > FA_MAX_KV_بلاکیں) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    اگر (!htp_ops_context_سیٹ_n_دھاگے(octx, kparams->n_دھاگے)) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    اگر (kparams->kernel_type == HTP_FA_KERNEL_HMX) {
        واپس hmx_flash_attn_ext(octx);
    }

    ساخت htp_fa_context factx;
    factx.octx = octx;
    factx.k = k;
    factx.v = v;

    factx.src0_div21 = kparams->u.hvx.src0_div21;
    factx.src0_div1  = kparams->u.hvx.src0_div1;

    factx.نشر_rk2 = kparams->نشر_rk2;
    factx.نشر_rk3 = kparams->نشر_rk3;
    factx.نشر_rv2 = kparams->نشر_rv2;
    factx.نشر_rv3 = kparams->نشر_rv3;

    اگر (نقاب) {
        factx.src3_div2 = kparams->src3_div2;
        factx.src3_div3 = kparams->src3_div3;
    }

    factx.is_q_fp32 = (q->type == HTP_TYPE_F32);
    factx.سائز_q_صف_padded = kparams->u.hvx.سائز_q_صف_padded;
    factx.سائز_k_صف_padded = kparams->u.hvx.سائز_k_صف_padded;
    factx.سائز_v_صف_padded = kparams->u.hvx.سائز_v_صف_padded;

    factx.n_بلاکیں = kparams->n_kv_بلاکیں;

    factx.scale = kparams->scale;
    factx.max_bias = kparams->max_bias;
    factx.has_softcap = (kparams->logit_softcap != 0.0f);
    factx.logit_softcap = 0;
    اگر (factx.has_softcap) {
        factx.logit_softcap = (__fp16) kparams->logit_softcap;
    }

    factx.n_سر_log2 = kparams->n_سر_log2;
    factx.m0          = kparams->m0;
    factx.m1          = kparams->m1;

    ثابت uint32_t n_سر = q->ne[2];
    اگر (n_سر > 512) {
        واپس HTP_حالت_NO_SUPPORT;
    }
    برائے (uint32_t h = 0; h < n_سر; ++h) {
        factx.slopes[h] = (__fp16) ((kparams->max_bias > 0.0f) ? alibi_ڈھلان(h, factx.n_سر_log2, factx.m0, factx.m1) : 1.0f);
    }

    // total rows in q
    ثابت uint32_t neq1 = q->ne[1];
    ثابت uint32_t neq2 = q->ne[2];
    ثابت uint32_t neq3 = q->ne[3];
    ثابت uint32_t کل_qrows = neq1 * neq2 * neq3;

    uint32_t qrow_آغاز = 0;
    uint32_t qrows      = کل_qrows;

    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت uint32_t mdev_گنتی = octx->ctx->mdev.گنتی;
        ثابت uint32_t mdev_idx   = octx->ctx->mdev.idx;
        ثابت uint32_t n_kv_heads = k->ne[2];
        ثابت uint32_t dst_e_سائز = (dst->type == HTP_TYPE_F32) ? حجم(اعشاری) : حجم(__fp16);
        ثابت bool can_بانٹو      = htp_tensor_can_صف_تقسیم(dst, dst_e_سائز);

        // head range is contiguous in flat row space only when neq3 == 1
        اگر (kparams->سر_بانٹو && can_بانٹو && neq3 == 1 && n_kv_heads >= mdev_گنتی && n_kv_heads % mdev_گنتی == 0) {
            ثابت uint32_t G              = kparams->G;
            ثابت uint32_t kv_فی_core    = n_kv_heads / mdev_گنتی;
            ثابت uint32_t heads_فی_core = kv_فی_core * G;
            ثابت uint32_t سر_آغاز     = mdev_idx * heads_فی_core;
            qrow_آغاز = سر_آغاز * neq1;
            qrows      = heads_فی_core * neq1;
        } ورنہ {
            ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(کل_qrows, can_بانٹو ? 1 : 0, mdev_idx, mdev_گنتی, &octx->ctx->mdev.گنتی_div);
            qrow_آغاز = range.آغاز;
            qrows      = range.گنتی;
        }
    }

    اگر (qrows == 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    factx.qrows            = qrows;
    factx.qrow_آغاز       = qrow_آغاز;
    factx.qrows_فی_thread = fastdiv(qrows + n_دھاگے - 1, &octx->n_دھاگے_div);

    ثابت bool has_نقاب  = (نقاب != تہی);
    ثابت bool has_sinks = (octx->src[4] != تہی);
    ساخت hvx_fa_vtcm_layout L;
    hvx_fa_vtcm_layout_build(&L, k->ne[0], v->ne[0], factx.is_q_fp32, has_نقاب, has_sinks, n_سر, n_دھاگے);

    اگر (L.کل_بائٹیں > octx->ctx->vtcm_سائز) {
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    factx.سائز_q_بلاک = L.سائز_q_بلاک;
    factx.سائز_k_بلاک = L.سائز_k_بلاک;
    factx.سائز_v_بلاک = L.سائز_v_بلاک;
    factx.سائز_m_بلاک = L.سائز_m_بلاک;
    factx.سائز_vkq_acc = L.سائز_vkq_acc;

    uint8_t * ثابت بنیاد = octx->ctx->vtcm_بنیاد;

    factx.spad_q               = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_q);
    factx.spad_k               = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_k);
    factx.spad_v               = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_v);
    factx.spad_m               = VTCM_LAYOUT_PTR_OPTIONAL(uint8_t, بنیاد, L.off_m, has_نقاب);
    factx.spad_a               = VTCM_LAYOUT_PTR(uint8_t, بنیاد, L.off_a);
    factx.spad_sinks           = VTCM_LAYOUT_PTR_OPTIONAL(اعشاری, بنیاد, L.off_sinks, has_sinks);
    factx.n_فعال_بلاکیں      = factx.n_بلاکیں;
    factx.shared_فعال_بلاکیں = false;

    اگر (has_نقاب && neq1 == 1 && neq3 == 1 && نقاب->ne[2] == 1 && factx.n_بلاکیں > 1 && factx.n_بلاکیں <= FA_MAX_KV_بلاکیں) {
        factx.n_فعال_بلاکیں = fa_جائزہ_فعال_بلاکیں(
            octx->ctx->dma[0], نقاب, 0, 1, 0, 0, k->ne[1], FLASH_ATTN_بلاک_سائز,
            factx.n_بلاکیں, factx.spad_k, factx.سائز_k_بلاک * 2, factx.فعال_بلاکیں);
        factx.shared_فعال_بلاکیں = true;
    }

    اگر (has_sinks) {
        ثابت ساخت htp_tensor * sinks = octx->src[4];
        dma_قطار * dma_q = octx->ctx->dma[0];
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(factx.spad_sinks, sinks->ڈیٹا), L.سائز_sinks, 0, sinks->سائز, 1);
        dma_قطار_نکالو(dma_q);
    }

    work_قطار_تسلسل(octx->ctx->work_قطار, flash_attn_ext_f16_thread, &factx, n_دھاگے);

    واپس HTP_حالت_OK;
}
