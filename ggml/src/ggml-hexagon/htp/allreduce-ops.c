#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>
#شامل <stdatomic.h>
#شامل <math.h>
#شامل <string.h>

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "hvx-utils.h"
#شامل "htp-tensor.h"
#شامل "dma-queue.h"
#شامل "hex-profile.h"
#شامل "allreduce-ops.h"
#شامل "htp-fence.h"

ساخت htp_allreduce_context {
    ساخت htp_ops_context * octx;
    uint32_t n_ranks;
    uint32_t n_dsts;
    uint32_t nelem;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t صف_سائز_aligned;
    uint32_t rank_elem_آغاز;
    uint32_t rank_nelem;
    uint32_t elems_فی_thread;
    uint32_t بلاک_elems;
    uint32_t vtcm_سائز_فی_thread;
    bool     is_صف_bcast;
    uint8_t * src_spad_بنیاد[HTP_ALLREDUCE_MAX_RANKS];
    uint8_t * dst_spad_بنیاد;
    uint8_t * res_spad_بنیاد;
};

#تعریف_کرو DEFINE_ALLREDUCE_THREAD_DMA_1D(SUFFIX, TYPE, HVX_جوڑو_FN, HAS_جوڑو)                                         \
ساکن خالی allreduce_thread_dma_1d_##SUFFIX(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {                   \
    ساخت htp_allreduce_context * actx = (ساخت htp_allreduce_context *) ڈیٹا;                                  \
    ساخت htp_ops_context * octx = actx->octx;                                                                   \
                                                                                                                  \
    ثابت uint32_t n_ranks     = actx->n_ranks;                                                                   \
    ثابت uint32_t n_dsts      = actx->n_dsts;                                                                    \
    ثابت uint32_t بلاک_elems = actx->بلاک_elems;                                                               \
                                                                                                                  \
    ثابت uint32_t dr  = actx->elems_فی_thread;                                                                  \
    ثابت uint32_t ir0 = actx->rank_elem_آغاز + dr * ith;                                                        \
    ثابت uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_آغاز + actx->rank_nelem);                                 \
    اگر (ir0 >= ir1) واپس;                                                                                       \
                                                                                                                  \
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];                                                        \
    dma_قطار * dma_q = octx->ctx->dma[ith];                                                                      \
                                                                                                                  \
    ثابت size_t vtcm_thread_انحراف = ith * actx->vtcm_سائز_فی_thread;                                           \
    uint8_t * dst_spad_بنیاد = actx->dst_spad_بنیاد + vtcm_thread_انحراف;                                           \
    uint8_t * res_spad_بنیاد = HAS_جوڑو ? (actx->res_spad_بنیاد + vtcm_thread_انحراف) : تہی;                        \
                                                                                                                  \
    ثابت size_t spad_half = actx->vtcm_سائز_فی_thread / 2;                                                      \
    uint32_t ir_prefetch = ir0;                                                                                   \
    صحیح spad_idx = 0;                                                                                             \
                                                                                                                  \
    برائے (صحیح k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                            \
        uint32_t cur_elems = MIN(بلاک_elems, ir1 - ir_prefetch);                                                 \
        size_t   cur_بائٹیں = cur_elems * حجم(TYPE);                                                            \
        uint8_t * d_spad = dst_spad_بنیاد + spad_idx * spad_half;                                                  \
        برائے (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->ڈیٹا + ir_prefetch * حجم(TYPE);                                  \
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(d_ddr, d_spad), cur_بائٹیں, cur_بائٹیں, cur_بائٹیں, 0);              \
        }                                                                                                         \
        برائے (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            uint8_t * s_spad = actx->src_spad_بنیاد[s] + vtcm_thread_انحراف + spad_idx * spad_half;                \
            ثابت dma_addr_t s_ddr = octx->src[s]->ڈیٹا + ir_prefetch * حجم(TYPE);                             \
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_spad, s_ddr), cur_بائٹیں, cur_بائٹیں, cur_بائٹیں, 1);              \
        }                                                                                                         \
        اگر (HAS_جوڑو) {                                                                                            \
            uint8_t * r_spad = res_spad_بنیاد + spad_idx * spad_half;                                              \
            ثابت dma_addr_t r_ddr = octx->src[2 * n_ranks]->ڈیٹا + ir_prefetch * حجم(TYPE);                   \
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(r_spad, r_ddr), cur_بائٹیں, cur_بائٹیں, cur_بائٹیں, 1);              \
        }                                                                                                         \
        ir_prefetch += cur_elems;                                                                                 \
        spad_idx ^= 1;                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    صحیح comp_spad_idx = 0;                                                                                        \
    برائے (uint32_t ir = ir0; ir < ir1; ) {                                                                         \
        uint32_t cur_elems = MIN(بلاک_elems, ir1 - ir);                                                          \
        size_t   cur_بائٹیں = cur_elems * حجم(TYPE);                                                            \
        برائے (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_قطار_نکالو(dma_q);                                                                                 \
        }                                                                                                         \
        برائے (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            dma_قطار_نکالو(dma_q);                                                                                 \
        }                                                                                                         \
        اگر (HAS_جوڑو) {                                                                                            \
            dma_قطار_نکالو(dma_q);                                                                                 \
        }                                                                                                         \
        uint8_t * d_spad  = dst_spad_بنیاد + comp_spad_idx * spad_half;                                            \
        uint8_t * s0_spad = actx->src_spad_بنیاد[0] + vtcm_thread_انحراف + comp_spad_idx * spad_half;              \
        uint8_t * s1_spad = actx->src_spad_بنیاد[1] + vtcm_thread_انحراف + comp_spad_idx * spad_half;              \
        uint8_t * r_spad  = HAS_جوڑو ? (res_spad_بنیاد + comp_spad_idx * spad_half) : تہی;                         \
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                         \
        HVX_جوڑو_FN(d_spad, s0_spad, s1_spad, cur_elems);                                                          \
        برائے (uint32_t s = 2; s < n_ranks; s++) {                                                                  \
            uint8_t * ss_spad = actx->src_spad_بنیاد[s] + vtcm_thread_انحراف + comp_spad_idx * spad_half;          \
            HVX_جوڑو_FN(d_spad, d_spad, ss_spad, cur_elems);                                                       \
        }                                                                                                         \
        اگر (HAS_جوڑو) {                                                                                            \
            HVX_جوڑو_FN(d_spad, d_spad, r_spad, cur_elems);                                                        \
        }                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                          \
        برائے (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->ڈیٹا + ir * حجم(TYPE);                                           \
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(d_ddr, d_spad), cur_بائٹیں, cur_بائٹیں, cur_بائٹیں, 1);              \
        }                                                                                                         \
        اگر (ir_prefetch < ir1) {                                                                                  \
            uint32_t اگلا_elems = MIN(بلاک_elems, ir1 - ir_prefetch);                                            \
            size_t   اگلا_بائٹیں = اگلا_elems * حجم(TYPE);                                                      \
            برائے (uint32_t s = 0; s < n_ranks; s++) {                                                              \
                uint8_t * s_spad = actx->src_spad_بنیاد[s] + vtcm_thread_انحراف + comp_spad_idx * spad_half;       \
                ثابت dma_addr_t s_اگلا = octx->src[s]->ڈیٹا + ir_prefetch * حجم(TYPE);                        \
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_spad, s_اگلا), اگلا_بائٹیں, اگلا_بائٹیں, اگلا_بائٹیں, 1);      \
            }                                                                                                     \
            اگر (HAS_جوڑو) {                                                                                        \
                uint8_t * r_spad_اگلا = res_spad_بنیاد + comp_spad_idx * spad_half;                                \
                ثابت dma_addr_t r_اگلا = octx->src[2 * n_ranks]->ڈیٹا + ir_prefetch * حجم(TYPE);              \
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(r_spad_اگلا, r_اگلا), اگلا_بائٹیں, اگلا_بائٹیں, اگلا_بائٹیں, 1); \
            }                                                                                                     \
            ir_prefetch += اگلا_elems;                                                                            \
        }                                                                                                         \
        comp_spad_idx ^= 1;                                                                                       \
        ir += cur_elems;                                                                                          \
    }                                                                                                             \
    dma_قطار_flush(dma_q);                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_جوڑو_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     اعشاری,  hvx_جوڑو_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(جوڑو_f16, __fp16, hvx_جوڑو_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(جوڑو_f32, اعشاری,  hvx_جوڑو_f32_aaa, 1)

#تعریف_کرو DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_جوڑو_FN, HAS_جوڑو, IS_صف_BCAST)                                                           \
ساکن خالی allreduce_thread_dma_2d_##SUFFIX(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {                                                   \
    ساخت htp_allreduce_context * actx = (ساخت htp_allreduce_context *) ڈیٹا;                                                                  \
    ساخت htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    ثابت uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    ثابت uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    ثابت uint32_t ne0              = actx->ne0;                                                                                                  \
    ثابت uint32_t بلاک_صفیں       = actx->بلاک_elems;                                                                                          \
    ثابت uint32_t صف_سائز_aligned = actx->صف_سائز_aligned;                                                                                     \
    ثابت uint32_t صف_بائٹیں        = ne0 * حجم(TYPE);                                                                                         \
                                                                                                                                                  \
    ثابت uint32_t dr  = actx->elems_فی_thread;                                                                                                  \
    ثابت uint32_t r0  = actx->rank_elem_آغاز + dr * ith;                                                                                        \
    ثابت uint32_t r1  = MIN(r0 + dr, actx->rank_elem_آغاز + actx->rank_nelem);                                                                  \
    اگر (r0 >= r1) واپس;                                                                                                                         \
                                                                                                                                                  \
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_قطار * dma_q = octx->ctx->dma[ith];                                                                                                      \
                                                                                                                                                  \
    ثابت size_t vtcm_thread_انحراف = ith * actx->vtcm_سائز_فی_thread;                                                                           \
    uint8_t * dst_spad_بنیاد = actx->dst_spad_بنیاد + vtcm_thread_انحراف;                                                                           \
    uint8_t * res_spad_بنیاد = HAS_جوڑو ? (IS_صف_BCAST ? actx->res_spad_بنیاد : (actx->res_spad_بنیاد + vtcm_thread_انحراف)) : تہی;                 \
                                                                                                                                                  \
    ثابت size_t spad_half = actx->vtcm_سائز_فی_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    صحیح spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    برائے (صحیح k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_صفیں = MIN(بلاک_صفیں, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_بنیاد + spad_idx * spad_half;                                                                                  \
        برائے (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->ڈیٹا + r_prefetch * octx->dsts[d]->nb[1];                                                           \
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(d_ddr, d_spad), octx->dsts[d]->nb[1], صف_سائز_aligned, صف_بائٹیں, 0);                            \
        }                                                                                                                                         \
        برائے (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = actx->src_spad_بنیاد[s] + vtcm_thread_انحراف + spad_idx * spad_half;                                                \
            ثابت dma_addr_t s_ddr = octx->src[s]->ڈیٹا + r_prefetch * octx->src[s]->nb[1];                                                       \
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_spad, s_ddr), صف_سائز_aligned, octx->src[s]->nb[1], صف_بائٹیں, cur_صفیں);                      \
        }                                                                                                                                         \
        اگر (HAS_جوڑو && !IS_صف_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_بنیاد + spad_idx * spad_half;                                                                              \
            ثابت dma_addr_t r_ddr = octx->src[2 * n_ranks]->ڈیٹا + r_prefetch * octx->src[2 * n_ranks]->nb[1];                                   \
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(r_spad, r_ddr), صف_سائز_aligned, octx->src[2 * n_ranks]->nb[1], صف_بائٹیں, cur_صفیں);            \
        }                                                                                                                                         \
        r_prefetch += cur_صفیں;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    صحیح comp_spad_idx = 0;                                                                                                                        \
    برائے (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_صفیں = MIN(بلاک_صفیں, r1 - r);                                                                                              \
        برائے (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_قطار_نکالو(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        برائے (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            dma_قطار_نکالو(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        اگر (HAS_جوڑو && !IS_صف_BCAST) {                                                                                                           \
            dma_قطار_نکالو(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        uint8_t * d_spad  = dst_spad_بنیاد + comp_spad_idx * spad_half;                                                                            \
        uint8_t * s0_spad = actx->src_spad_بنیاد[0] + vtcm_thread_انحراف + comp_spad_idx * spad_half;                                              \
        uint8_t * s1_spad = actx->src_spad_بنیاد[1] + vtcm_thread_انحراف + comp_spad_idx * spad_half;                                              \
        uint8_t * r_spad  = (HAS_جوڑو && !IS_صف_BCAST) ? (res_spad_بنیاد + comp_spad_idx * spad_half) : تہی;                                      \
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        برائے (uint32_t صف = 0; صف < cur_صفیں; صف++) {                                                                                           \
            uint8_t * d_صف = d_spad + صف * صف_سائز_aligned;                                                                                    \
            ثابت uint8_t * s0_صف = s0_spad + صف * صف_سائز_aligned;                                                                            \
            ثابت uint8_t * s1_صف = s1_spad + صف * صف_سائز_aligned;                                                                            \
            HVX_جوڑو_FN(d_صف, s0_صف, s1_صف, ne0);                                                                                               \
            برائے (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                ثابت uint8_t * ss_صف = actx->src_spad_بنیاد[s] + vtcm_thread_انحراف + comp_spad_idx * spad_half + صف * صف_سائز_aligned;        \
                HVX_جوڑو_FN(d_صف, d_صف, ss_صف, ne0);                                                                                            \
            }                                                                                                                                     \
            اگر (HAS_جوڑو) {                                                                                                                        \
                ثابت uint8_t * res_صف = IS_صف_BCAST ? res_spad_بنیاد : (r_spad + صف * صف_سائز_aligned);                                       \
                HVX_جوڑو_FN(d_صف, d_صف, res_صف, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        برائے (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->ڈیٹا + r * octx->dsts[d]->nb[1];                                                                    \
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(d_ddr, d_spad), octx->dsts[d]->nb[1], صف_سائز_aligned, صف_بائٹیں, cur_صفیں);                     \
        }                                                                                                                                         \
        اگر (r_prefetch < r1) {                                                                                                                    \
            uint32_t اگلا_صفیں = MIN(بلاک_صفیں, r1 - r_prefetch);                                                                                \
            برائے (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                uint8_t * s_spad = actx->src_spad_بنیاد[s] + vtcm_thread_انحراف + comp_spad_idx * spad_half;                                       \
                ثابت dma_addr_t s_اگلا = octx->src[s]->ڈیٹا + r_prefetch * octx->src[s]->nb[1];                                                  \
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(s_spad, s_اگلا), صف_سائز_aligned, octx->src[s]->nb[1], صف_بائٹیں, اگلا_صفیں);                \
            }                                                                                                                                     \
            اگر (HAS_جوڑو && !IS_صف_BCAST) {                                                                                                       \
                uint8_t * r_spad_اگلا = res_spad_بنیاد + comp_spad_idx * spad_half;                                                                \
                ثابت dma_addr_t r_اگلا = octx->src[2 * n_ranks]->ڈیٹا + r_prefetch * octx->src[2 * n_ranks]->nb[1];                              \
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(r_spad_اگلا, r_اگلا), صف_سائز_aligned, octx->src[2 * n_ranks]->nb[1], صف_بائٹیں, اگلا_صفیں); \
            }                                                                                                                                     \
            r_prefetch += اگلا_صفیں;                                                                                                              \
        }                                                                                                                                         \
        comp_spad_idx ^= 1;                                                                                                                       \
        r += cur_صفیں;                                                                                                                            \
    }                                                                                                                                             \
    dma_قطار_flush(dma_q);                                                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_جوڑو_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           اعشاری,  hvx_جوڑو_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جوڑو_f16,       __fp16, hvx_جوڑو_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جوڑو_f32,       اعشاری,  hvx_جوڑو_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جوڑو_bcast_f16, __fp16, hvx_جوڑو_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جوڑو_bcast_f32, اعشاری,  hvx_جوڑو_f32_aaa, 1, 1)

ساکن صحیح validate_allreduce(
    ساخت htp_ops_context * octx,
    ثابت ساخت htp_allreduce_kernel_params * kparams,
    uint32_t n_ranks
) {
    اگر (kparams->n_دھاگے == 0 || (uint32_t) kparams->n_دھاگے > octx->ctx->n_دھاگے) {
        واپس HTP_حالت_INVAL_PARAMS;
    }
    octx->n_دھاگے = (uint32_t) kparams->n_دھاگے;

    اگر (kparams->vtcm_سائز_فی_thread <= 0 || kparams->vtcm_سائز <= 0) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    ثابت bool has_جوڑو = (octx->op == HTP_OP_ALLREDUCE_جوڑو);
    ثابت size_t n_vtcm_buffers = htp_allreduce_vtcm_بفر_گنتی(
        n_ranks, octx->n_دھاگے, has_جوڑو, kparams->is_صف_bcast != 0);
    ثابت size_t vtcm_سائز = n_vtcm_buffers * (size_t) kparams->vtcm_سائز_فی_thread;
    اگر (vtcm_سائز != (size_t) kparams->vtcm_سائز) {
        واپس HTP_حالت_INVAL_PARAMS;
    }
    اگر (vtcm_سائز > octx->ctx->vtcm_سائز) {
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    اگر (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    واپس HTP_حالت_OK;
}

صحیح op_allreduce(ساخت htp_ops_context * octx) {
    اگر (octx->ctx->mdev.گنتی > 1 && octx->ctx->mdev.idx > 0) {
        واپس HTP_حالت_OK;
    }

    ثابت ساخت htp_allreduce_kernel_params * kparams = (ثابت ساخت htp_allreduce_kernel_params *) octx->kernel_params;
    ثابت ساخت htp_tensor * dst = octx->dst;

    ثابت uint32_t rank    = (uint32_t) kparams->rank;
    ثابت uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    اگر (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t fence_seq_اندراج = (uint32_t) octx->op_params[0];
    ثابت uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    ثابت ساخت htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->ڈیٹا;

    ثابت صحیح حالت_2 = validate_allreduce(octx, kparams, n_ranks);
    اگر (حالت_2 != HTP_حالت_OK) {
        اگر (حالت_2 == HTP_حالت_NO_SUPPORT) {
            FARF(ERROR, "ggml-hex: allreduce unsupported type %d : rank %u\n", dst->type, rank);
        }
        htp_fence_write(my_fence, fence_seq_exit, حالت_2);
        واپس حالت_2;
    }

    ثابت bool has_جوڑو = (octx->op == HTP_OP_ALLREDUCE_جوڑو);
    ثابت uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت int32_t  منوال  = kparams->منوال;

    // 1. Entry Barrier: Synchronize all ranks before reading
    ساخت htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_آغاز(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_اندراج);

    htp_fence_write(my_fence, fence_seq_اندراج, octx->حالت_2);

    برائے (uint32_t j = 0; j < n_ranks; j++) {
        اگر (j == rank) جاری;
        ثابت ساخت htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->ڈیٹا;
        uint64_t spins = 0;
        جب (1) {
            uint32_t peer_seq;
            uint32_t peer_حالت;
            htp_fence_read(peer_fence, &peer_seq, &peer_حالت);
            اگر ((int32_t)(peer_seq - fence_seq_اندراج) >= 0) {
                اگر (peer_حالت > HTP_حالت_OK) {
                    FARF(ERROR, "ggml-hex: allreduce entry peer %u failed with status %u\n", j, peer_حالت);
                    htp_fence_write(my_fence, fence_seq_exit, peer_حالت);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_اندراج);
                    واپس peer_حالت;
                }
                توڑو;
            }
            اگر (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_اندراج, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_حالت_اندرونی_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_اندراج);
                واپس HTP_حالت_اندرونی_ERR;
            }
            hex_pause();
        }
    }
    asm متلون ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_اندراج);

    // 2. Multi-threaded Reduction across assigned rank chunk
    اگر (nelem > 0) {
        ثابت uint32_t n_دھاگے            = (uint32_t) kparams->n_دھاگے;
        ثابت uint32_t بلاک_elems          = (uint32_t) kparams->بلاک_elems;
        ثابت uint32_t elems_فی_thread     = (uint32_t) kparams->elems_فی_thread;
        ثابت uint32_t vtcm_سائز_فی_thread = (uint32_t) kparams->vtcm_سائز_فی_thread;

        ساخت htp_allreduce_context actx;
        actx.octx                 = octx;
        actx.n_ranks              = n_ranks;
        actx.n_dsts               = (uint32_t) kparams->n_dsts ? (uint32_t) kparams->n_dsts : n_ranks;
        actx.nelem                = nelem;
        actx.ne0                  = (uint32_t) kparams->ne0;
        actx.ne1                  = (uint32_t) kparams->ne1;
        actx.صف_سائز_aligned     = (uint32_t) kparams->صف_سائز_aligned;
        actx.rank_elem_آغاز      = (uint32_t) kparams->rank_elem_آغاز;
        actx.rank_nelem           = (uint32_t) kparams->rank_nelem;
        actx.elems_فی_thread     = elems_فی_thread;
        actx.بلاک_elems          = بلاک_elems;
        actx.vtcm_سائز_فی_thread = vtcm_سائز_فی_thread;
        actx.is_صف_bcast         = (kparams->is_صف_bcast != 0);

        work_قطار_func_t reduce_fun = تہی;
        انتخاب (kparams->kernel_type) {
            حالت HTP_ALLREDUCE_KERNEL_DMA_1D:
                اگر (has_جوڑو) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_جوڑو_f16 : allreduce_thread_dma_1d_جوڑو_f32;
                } ورنہ {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                توڑو;
            حالت HTP_ALLREDUCE_KERNEL_DMA_2D:
                اگر (has_جوڑو) {
                    اگر (kparams->is_صف_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_جوڑو_bcast_f16 : allreduce_thread_dma_2d_جوڑو_bcast_f32;
                    } ورنہ {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_جوڑو_f16 : allreduce_thread_dma_2d_جوڑو_f32;
                    }
                } ورنہ {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_f16 : allreduce_thread_dma_2d_f32;
                }
                توڑو;
            طےشدہ:
                FARF(ERROR, "ggml-hex: allreduce unsupported kernel %d : rank %u\n", kparams->kernel_type, rank);
                htp_fence_write(my_fence, fence_seq_exit, HTP_حالت_NO_SUPPORT);
                واپس HTP_حالت_NO_SUPPORT;
        }

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_بنیاد;
        برائے (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_بنیاد[s] = vtcm_ptr;
            vtcm_ptr += n_دھاگے * vtcm_سائز_فی_thread;
        }
        actx.dst_spad_بنیاد = vtcm_ptr;
        vtcm_ptr += n_دھاگے * vtcm_سائز_فی_thread;
        اگر (has_جوڑو) {
            actx.res_spad_بنیاد = vtcm_ptr;
            vtcm_ptr += (actx.is_صف_bcast ? 1 : n_دھاگے) * vtcm_سائز_فی_thread;
        }

        اگر (has_جوڑو && actx.is_صف_bcast) {
            ثابت dma_addr_t r_ddr = octx->src[2 * n_ranks]->ڈیٹا;
            ثابت uint32_t صف_بائٹیں = actx.ne0 * (dst->type == HTP_TYPE_F16 ? حجم(__fp16) : حجم(اعشاری));
            dma_قطار * dma_q = octx->ctx->dma[0];
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(actx.res_spad_بنیاد, r_ddr), actx.صف_سائز_aligned, 0, صف_بائٹیں, 1);
            dma_قطار_نکالو(dma_q);
        }

        work_قطار_تسلسل(octx->ctx->work_قطار, reduce_fun, &actx, n_دھاگے);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_آغاز(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    // drain fan-out DMA writes before exit barrier
    اگر (منوال == HTP_ALLREDUCE_SHARDED_FANOUT) {
        asm متلون ("syncht" : : : "memory");
    }

    htp_fence_write(my_fence, fence_seq_exit, octx->حالت_2);

    برائے (uint32_t j = 0; j < n_ranks; j++) {
        اگر (j == rank) جاری;
        ثابت ساخت htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->ڈیٹا;
        uint64_t spins = 0;
        جب (1) {
            uint32_t peer_seq;
            uint32_t peer_حالت;
            htp_fence_read(peer_fence, &peer_seq, &peer_حالت);
            اگر ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                اگر (peer_حالت > HTP_حالت_OK) {
                    FARF(ERROR, "ggml-hex: allreduce exit peer %u failed with status %u\n", j, peer_حالت);
                    htp_fence_write(my_fence, fence_seq_exit, peer_حالت);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                    واپس peer_حالت;
                }
                توڑو;
            }
            اگر (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce exit fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_exit, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_حالت_اندرونی_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                واپس HTP_حالت_اندرونی_ERR;
            }
            hex_pause();
        }
    }
    asm متلون ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    واپس octx->حالت_2;
}
