#هدايت clang diagnostic ignored "-Wunused-variable"
#هدايت clang diagnostic ignored "-Wunused-function"
#هدايت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>
#ضمن <stdatomic.h>
#ضمن <math.h>
#ضمن <string.h>

#وصف_ڪريو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "hvx-utils.h"
#ضمن "htp-tensor.h"
#ضمن "dma-queue.h"
#ضمن "hex-profile.h"
#ضمن "allreduce-ops.h"
#ضمن "htp-fence.h"

بنية htp_allreduce_context {
    بنية htp_ops_context * octx;
    uint32_t n_ranks;
    uint32_t n_dsts;
    uint32_t nelem;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t صف_سائيز_aligned;
    uint32_t rank_elem_شروعات;
    uint32_t rank_nelem;
    uint32_t elems_في_thread;
    uint32_t بلاڪ_elems;
    uint32_t vtcm_سائيز_في_thread;
    bool     is_صف_bcast;
    uint8_t * src_spad_بنياد[HTP_ALLREDUCE_MAX_RANKS];
    uint8_t * dst_spad_بنياد;
    uint8_t * res_spad_بنياد;
};

#وصف_ڪريو DEFINE_ALLREDUCE_THREAD_DMA_1D(SUFFIX, TYPE, HVX_جوڙيو_FN, HAS_جوڙيو)                                         \
ساكن فراغ allreduce_thread_dma_1d_##SUFFIX(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {                   \
    بنية htp_allreduce_context * actx = (بنية htp_allreduce_context *) معلومات;                                  \
    بنية htp_ops_context * octx = actx->octx;                                                                   \
                                                                                                                  \
    ثابت uint32_t n_ranks     = actx->n_ranks;                                                                   \
    ثابت uint32_t n_dsts      = actx->n_dsts;                                                                    \
    ثابت uint32_t بلاڪ_elems = actx->بلاڪ_elems;                                                               \
                                                                                                                  \
    ثابت uint32_t dr  = actx->elems_في_thread;                                                                  \
    ثابت uint32_t ir0 = actx->rank_elem_شروعات + dr * ith;                                                        \
    ثابت uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_شروعات + actx->rank_nelem);                                 \
    إذا (ir0 >= ir1) أرجع;                                                                                       \
                                                                                                                  \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                                        \
    dma_قطار * dma_q = octx->ctx->dma[ith];                                                                      \
                                                                                                                  \
    ثابت size_t vtcm_thread_وٿي = ith * actx->vtcm_سائيز_في_thread;                                           \
    uint8_t * dst_spad_بنياد = actx->dst_spad_بنياد + vtcm_thread_وٿي;                                           \
    uint8_t * res_spad_بنياد = HAS_جوڙيو ? (actx->res_spad_بنياد + vtcm_thread_وٿي) : عدم;                        \
                                                                                                                  \
    ثابت size_t spad_half = actx->vtcm_سائيز_في_thread / 2;                                                      \
    uint32_t ir_prefetch = ir0;                                                                                   \
    صحيح spad_idx = 0;                                                                                             \
                                                                                                                  \
    لكل (صحيح k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                            \
        uint32_t cur_elems = MIN(بلاڪ_elems, ir1 - ir_prefetch);                                                 \
        size_t   cur_بائيٽون = cur_elems * حجم(TYPE);                                                            \
        uint8_t * d_spad = dst_spad_بنياد + spad_idx * spad_half;                                                  \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->معلومات + ir_prefetch * حجم(TYPE);                                  \
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(d_ddr, d_spad), cur_بائيٽون, cur_بائيٽون, cur_بائيٽون, 0);              \
        }                                                                                                         \
        لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            uint8_t * s_spad = actx->src_spad_بنياد[s] + vtcm_thread_وٿي + spad_idx * spad_half;                \
            ثابت dma_addr_t s_ddr = octx->src[s]->معلومات + ir_prefetch * حجم(TYPE);                             \
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s_spad, s_ddr), cur_بائيٽون, cur_بائيٽون, cur_بائيٽون, 1);              \
        }                                                                                                         \
        إذا (HAS_جوڙيو) {                                                                                            \
            uint8_t * r_spad = res_spad_بنياد + spad_idx * spad_half;                                              \
            ثابت dma_addr_t r_ddr = octx->src[2 * n_ranks]->معلومات + ir_prefetch * حجم(TYPE);                   \
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(r_spad, r_ddr), cur_بائيٽون, cur_بائيٽون, cur_بائيٽون, 1);              \
        }                                                                                                         \
        ir_prefetch += cur_elems;                                                                                 \
        spad_idx ^= 1;                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    صحيح comp_spad_idx = 0;                                                                                        \
    لكل (uint32_t ir = ir0; ir < ir1; ) {                                                                         \
        uint32_t cur_elems = MIN(بلاڪ_elems, ir1 - ir);                                                          \
        size_t   cur_بائيٽون = cur_elems * حجم(TYPE);                                                            \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_قطار_ڪڍو(dma_q);                                                                                 \
        }                                                                                                         \
        لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            dma_قطار_ڪڍو(dma_q);                                                                                 \
        }                                                                                                         \
        إذا (HAS_جوڙيو) {                                                                                            \
            dma_قطار_ڪڍو(dma_q);                                                                                 \
        }                                                                                                         \
        uint8_t * d_spad  = dst_spad_بنياد + comp_spad_idx * spad_half;                                            \
        uint8_t * s0_spad = actx->src_spad_بنياد[0] + vtcm_thread_وٿي + comp_spad_idx * spad_half;              \
        uint8_t * s1_spad = actx->src_spad_بنياد[1] + vtcm_thread_وٿي + comp_spad_idx * spad_half;              \
        uint8_t * r_spad  = HAS_جوڙيو ? (res_spad_بنياد + comp_spad_idx * spad_half) : عدم;                         \
        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                         \
        HVX_جوڙيو_FN(d_spad, s0_spad, s1_spad, cur_elems);                                                          \
        لكل (uint32_t s = 2; s < n_ranks; s++) {                                                                  \
            uint8_t * ss_spad = actx->src_spad_بنياد[s] + vtcm_thread_وٿي + comp_spad_idx * spad_half;          \
            HVX_جوڙيو_FN(d_spad, d_spad, ss_spad, cur_elems);                                                       \
        }                                                                                                         \
        إذا (HAS_جوڙيو) {                                                                                            \
            HVX_جوڙيو_FN(d_spad, d_spad, r_spad, cur_elems);                                                        \
        }                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                          \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->معلومات + ir * حجم(TYPE);                                           \
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(d_ddr, d_spad), cur_بائيٽون, cur_بائيٽون, cur_بائيٽون, 1);              \
        }                                                                                                         \
        إذا (ir_prefetch < ir1) {                                                                                  \
            uint32_t ايندڙ_elems = MIN(بلاڪ_elems, ir1 - ir_prefetch);                                            \
            size_t   ايندڙ_بائيٽون = ايندڙ_elems * حجم(TYPE);                                                      \
            لكل (uint32_t s = 0; s < n_ranks; s++) {                                                              \
                uint8_t * s_spad = actx->src_spad_بنياد[s] + vtcm_thread_وٿي + comp_spad_idx * spad_half;       \
                ثابت dma_addr_t s_ايندڙ = octx->src[s]->معلومات + ir_prefetch * حجم(TYPE);                        \
                dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s_spad, s_ايندڙ), ايندڙ_بائيٽون, ايندڙ_بائيٽون, ايندڙ_بائيٽون, 1);      \
            }                                                                                                     \
            إذا (HAS_جوڙيو) {                                                                                        \
                uint8_t * r_spad_ايندڙ = res_spad_بنياد + comp_spad_idx * spad_half;                                \
                ثابت dma_addr_t r_ايندڙ = octx->src[2 * n_ranks]->معلومات + ir_prefetch * حجم(TYPE);              \
                dma_قطار_ڌڪو(dma_q, dma_make_معلومات(r_spad_ايندڙ, r_ايندڙ), ايندڙ_بائيٽون, ايندڙ_بائيٽون, ايندڙ_بائيٽون, 1); \
            }                                                                                                     \
            ir_prefetch += ايندڙ_elems;                                                                            \
        }                                                                                                         \
        comp_spad_idx ^= 1;                                                                                       \
        ir += cur_elems;                                                                                          \
    }                                                                                                             \
    dma_قطار_flush(dma_q);                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_جوڙيو_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     عائم,  hvx_جوڙيو_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(جوڙيو_f16, __fp16, hvx_جوڙيو_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(جوڙيو_f32, عائم,  hvx_جوڙيو_f32_aaa, 1)

#وصف_ڪريو DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_جوڙيو_FN, HAS_جوڙيو, IS_صف_BCAST)                                                           \
ساكن فراغ allreduce_thread_dma_2d_##SUFFIX(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {                                                   \
    بنية htp_allreduce_context * actx = (بنية htp_allreduce_context *) معلومات;                                                                  \
    بنية htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    ثابت uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    ثابت uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    ثابت uint32_t ne0              = actx->ne0;                                                                                                  \
    ثابت uint32_t بلاڪ_صفون       = actx->بلاڪ_elems;                                                                                          \
    ثابت uint32_t صف_سائيز_aligned = actx->صف_سائيز_aligned;                                                                                     \
    ثابت uint32_t صف_بائيٽون        = ne0 * حجم(TYPE);                                                                                         \
                                                                                                                                                  \
    ثابت uint32_t dr  = actx->elems_في_thread;                                                                                                  \
    ثابت uint32_t r0  = actx->rank_elem_شروعات + dr * ith;                                                                                        \
    ثابت uint32_t r1  = MIN(r0 + dr, actx->rank_elem_شروعات + actx->rank_nelem);                                                                  \
    إذا (r0 >= r1) أرجع;                                                                                                                         \
                                                                                                                                                  \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_قطار * dma_q = octx->ctx->dma[ith];                                                                                                      \
                                                                                                                                                  \
    ثابت size_t vtcm_thread_وٿي = ith * actx->vtcm_سائيز_في_thread;                                                                           \
    uint8_t * dst_spad_بنياد = actx->dst_spad_بنياد + vtcm_thread_وٿي;                                                                           \
    uint8_t * res_spad_بنياد = HAS_جوڙيو ? (IS_صف_BCAST ? actx->res_spad_بنياد : (actx->res_spad_بنياد + vtcm_thread_وٿي)) : عدم;                 \
                                                                                                                                                  \
    ثابت size_t spad_half = actx->vtcm_سائيز_في_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    صحيح spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    لكل (صحيح k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_صفون = MIN(بلاڪ_صفون, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_بنياد + spad_idx * spad_half;                                                                                  \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->معلومات + r_prefetch * octx->dsts[d]->nb[1];                                                           \
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(d_ddr, d_spad), octx->dsts[d]->nb[1], صف_سائيز_aligned, صف_بائيٽون, 0);                            \
        }                                                                                                                                         \
        لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = actx->src_spad_بنياد[s] + vtcm_thread_وٿي + spad_idx * spad_half;                                                \
            ثابت dma_addr_t s_ddr = octx->src[s]->معلومات + r_prefetch * octx->src[s]->nb[1];                                                       \
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s_spad, s_ddr), صف_سائيز_aligned, octx->src[s]->nb[1], صف_بائيٽون, cur_صفون);                      \
        }                                                                                                                                         \
        إذا (HAS_جوڙيو && !IS_صف_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_بنياد + spad_idx * spad_half;                                                                              \
            ثابت dma_addr_t r_ddr = octx->src[2 * n_ranks]->معلومات + r_prefetch * octx->src[2 * n_ranks]->nb[1];                                   \
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(r_spad, r_ddr), صف_سائيز_aligned, octx->src[2 * n_ranks]->nb[1], صف_بائيٽون, cur_صفون);            \
        }                                                                                                                                         \
        r_prefetch += cur_صفون;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    صحيح comp_spad_idx = 0;                                                                                                                        \
    لكل (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_صفون = MIN(بلاڪ_صفون, r1 - r);                                                                                              \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_قطار_ڪڍو(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            dma_قطار_ڪڍو(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        إذا (HAS_جوڙيو && !IS_صف_BCAST) {                                                                                                           \
            dma_قطار_ڪڍو(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        uint8_t * d_spad  = dst_spad_بنياد + comp_spad_idx * spad_half;                                                                            \
        uint8_t * s0_spad = actx->src_spad_بنياد[0] + vtcm_thread_وٿي + comp_spad_idx * spad_half;                                              \
        uint8_t * s1_spad = actx->src_spad_بنياد[1] + vtcm_thread_وٿي + comp_spad_idx * spad_half;                                              \
        uint8_t * r_spad  = (HAS_جوڙيو && !IS_صف_BCAST) ? (res_spad_بنياد + comp_spad_idx * spad_half) : عدم;                                      \
        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        لكل (uint32_t صف = 0; صف < cur_صفون; صف++) {                                                                                           \
            uint8_t * d_صف = d_spad + صف * صف_سائيز_aligned;                                                                                    \
            ثابت uint8_t * s0_صف = s0_spad + صف * صف_سائيز_aligned;                                                                            \
            ثابت uint8_t * s1_صف = s1_spad + صف * صف_سائيز_aligned;                                                                            \
            HVX_جوڙيو_FN(d_صف, s0_صف, s1_صف, ne0);                                                                                               \
            لكل (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                ثابت uint8_t * ss_صف = actx->src_spad_بنياد[s] + vtcm_thread_وٿي + comp_spad_idx * spad_half + صف * صف_سائيز_aligned;        \
                HVX_جوڙيو_FN(d_صف, d_صف, ss_صف, ne0);                                                                                            \
            }                                                                                                                                     \
            إذا (HAS_جوڙيو) {                                                                                                                        \
                ثابت uint8_t * res_صف = IS_صف_BCAST ? res_spad_بنياد : (r_spad + صف * صف_سائيز_aligned);                                       \
                HVX_جوڙيو_FN(d_صف, d_صف, res_صف, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->معلومات + r * octx->dsts[d]->nb[1];                                                                    \
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(d_ddr, d_spad), octx->dsts[d]->nb[1], صف_سائيز_aligned, صف_بائيٽون, cur_صفون);                     \
        }                                                                                                                                         \
        إذا (r_prefetch < r1) {                                                                                                                    \
            uint32_t ايندڙ_صفون = MIN(بلاڪ_صفون, r1 - r_prefetch);                                                                                \
            لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                uint8_t * s_spad = actx->src_spad_بنياد[s] + vtcm_thread_وٿي + comp_spad_idx * spad_half;                                       \
                ثابت dma_addr_t s_ايندڙ = octx->src[s]->معلومات + r_prefetch * octx->src[s]->nb[1];                                                  \
                dma_قطار_ڌڪو(dma_q, dma_make_معلومات(s_spad, s_ايندڙ), صف_سائيز_aligned, octx->src[s]->nb[1], صف_بائيٽون, ايندڙ_صفون);                \
            }                                                                                                                                     \
            إذا (HAS_جوڙيو && !IS_صف_BCAST) {                                                                                                       \
                uint8_t * r_spad_ايندڙ = res_spad_بنياد + comp_spad_idx * spad_half;                                                                \
                ثابت dma_addr_t r_ايندڙ = octx->src[2 * n_ranks]->معلومات + r_prefetch * octx->src[2 * n_ranks]->nb[1];                              \
                dma_قطار_ڌڪو(dma_q, dma_make_معلومات(r_spad_ايندڙ, r_ايندڙ), صف_سائيز_aligned, octx->src[2 * n_ranks]->nb[1], صف_بائيٽون, ايندڙ_صفون); \
            }                                                                                                                                     \
            r_prefetch += ايندڙ_صفون;                                                                                                              \
        }                                                                                                                                         \
        comp_spad_idx ^= 1;                                                                                                                       \
        r += cur_صفون;                                                                                                                            \
    }                                                                                                                                             \
    dma_قطار_flush(dma_q);                                                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_جوڙيو_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           عائم,  hvx_جوڙيو_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جوڙيو_f16,       __fp16, hvx_جوڙيو_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جوڙيو_f32,       عائم,  hvx_جوڙيو_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جوڙيو_bcast_f16, __fp16, hvx_جوڙيو_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جوڙيو_bcast_f32, عائم,  hvx_جوڙيو_f32_aaa, 1, 1)

ساكن صحيح validate_allreduce(
    بنية htp_ops_context * octx,
    ثابت بنية htp_allreduce_kernel_params * kparams,
    uint32_t n_ranks
) {
    إذا (kparams->n_ڌاڳا == 0 || (uint32_t) kparams->n_ڌاڳا > octx->ctx->n_ڌاڳا) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }
    octx->n_ڌاڳا = (uint32_t) kparams->n_ڌاڳا;

    إذا (kparams->vtcm_سائيز_في_thread <= 0 || kparams->vtcm_سائيز <= 0) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت bool has_جوڙيو = (octx->op == HTP_OP_ALLREDUCE_جوڙيو);
    ثابت size_t n_vtcm_buffers = htp_allreduce_vtcm_بفر_ڳڻپ(
        n_ranks, octx->n_ڌاڳا, has_جوڙيو, kparams->is_صف_bcast != 0);
    ثابت size_t vtcm_سائيز = n_vtcm_buffers * (size_t) kparams->vtcm_سائيز_في_thread;
    إذا (vtcm_سائيز != (size_t) kparams->vtcm_سائيز) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }
    إذا (vtcm_سائيز > octx->ctx->vtcm_سائيز) {
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    إذا (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    أرجع HTP_حالت_OK;
}

صحيح op_allreduce(بنية htp_ops_context * octx) {
    إذا (octx->ctx->mdev.ڳڻپ > 1 && octx->ctx->mdev.idx > 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت بنية htp_allreduce_kernel_params * kparams = (ثابت بنية htp_allreduce_kernel_params *) octx->kernel_params;
    ثابت بنية htp_tensor * dst = octx->dst;

    ثابت uint32_t rank    = (uint32_t) kparams->rank;
    ثابت uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    إذا (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t fence_seq_داخلا = (uint32_t) octx->op_params[0];
    ثابت uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    ثابت بنية htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->معلومات;

    ثابت صحيح حالت = validate_allreduce(octx, kparams, n_ranks);
    إذا (حالت != HTP_حالت_OK) {
        إذا (حالت == HTP_حالت_NO_SUPPORT) {
            FARF(ERROR, "ggml-hex: allreduce unsupported type %d : rank %u\n", dst->type, rank);
        }
        htp_fence_write(my_fence, fence_seq_exit, حالت);
        أرجع حالت;
    }

    ثابت bool has_جوڙيو = (octx->op == HTP_OP_ALLREDUCE_جوڙيو);
    ثابت uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت int32_t  گهڻائي  = kparams->گهڻائي;

    // 1. Entry Barrier: Synchronize all ranks before reading
    بنية htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_شروعات(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_داخلا);

    htp_fence_write(my_fence, fence_seq_داخلا, octx->حالت);

    لكل (uint32_t j = 0; j < n_ranks; j++) {
        إذا (j == rank) استمر;
        ثابت بنية htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->معلومات;
        uint64_t spins = 0;
        طالما (1) {
            uint32_t peer_seq;
            uint32_t peer_حالت;
            htp_fence_read(peer_fence, &peer_seq, &peer_حالت);
            إذا ((int32_t)(peer_seq - fence_seq_داخلا) >= 0) {
                إذا (peer_حالت > HTP_حالت_OK) {
                    FARF(ERROR, "ggml-hex: allreduce entry peer %u failed with status %u\n", j, peer_حالت);
                    htp_fence_write(my_fence, fence_seq_exit, peer_حالت);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_داخلا);
                    أرجع peer_حالت;
                }
                اكسر;
            }
            إذا (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_داخلا, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_حالت_اندروني_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_داخلا);
                أرجع HTP_حالت_اندروني_ERR;
            }
            hex_pause();
        }
    }
    asm متطاير ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_داخلا);

    // 2. Multi-threaded Reduction across assigned rank chunk
    إذا (nelem > 0) {
        ثابت uint32_t n_ڌاڳا            = (uint32_t) kparams->n_ڌاڳا;
        ثابت uint32_t بلاڪ_elems          = (uint32_t) kparams->بلاڪ_elems;
        ثابت uint32_t elems_في_thread     = (uint32_t) kparams->elems_في_thread;
        ثابت uint32_t vtcm_سائيز_في_thread = (uint32_t) kparams->vtcm_سائيز_في_thread;

        بنية htp_allreduce_context actx;
        actx.octx                 = octx;
        actx.n_ranks              = n_ranks;
        actx.n_dsts               = (uint32_t) kparams->n_dsts ? (uint32_t) kparams->n_dsts : n_ranks;
        actx.nelem                = nelem;
        actx.ne0                  = (uint32_t) kparams->ne0;
        actx.ne1                  = (uint32_t) kparams->ne1;
        actx.صف_سائيز_aligned     = (uint32_t) kparams->صف_سائيز_aligned;
        actx.rank_elem_شروعات      = (uint32_t) kparams->rank_elem_شروعات;
        actx.rank_nelem           = (uint32_t) kparams->rank_nelem;
        actx.elems_في_thread     = elems_في_thread;
        actx.بلاڪ_elems          = بلاڪ_elems;
        actx.vtcm_سائيز_في_thread = vtcm_سائيز_في_thread;
        actx.is_صف_bcast         = (kparams->is_صف_bcast != 0);

        work_قطار_func_t reduce_fun = عدم;
        بدل (kparams->kernel_type) {
            حالة HTP_ALLREDUCE_KERNEL_DMA_1D:
                إذا (has_جوڙيو) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_جوڙيو_f16 : allreduce_thread_dma_1d_جوڙيو_f32;
                } وإلا {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                اكسر;
            حالة HTP_ALLREDUCE_KERNEL_DMA_2D:
                إذا (has_جوڙيو) {
                    إذا (kparams->is_صف_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_جوڙيو_bcast_f16 : allreduce_thread_dma_2d_جوڙيو_bcast_f32;
                    } وإلا {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_جوڙيو_f16 : allreduce_thread_dma_2d_جوڙيو_f32;
                    }
                } وإلا {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_f16 : allreduce_thread_dma_2d_f32;
                }
                اكسر;
            افتراضي:
                FARF(ERROR, "ggml-hex: allreduce unsupported kernel %d : rank %u\n", kparams->kernel_type, rank);
                htp_fence_write(my_fence, fence_seq_exit, HTP_حالت_NO_SUPPORT);
                أرجع HTP_حالت_NO_SUPPORT;
        }

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_بنياد;
        لكل (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_بنياد[s] = vtcm_ptr;
            vtcm_ptr += n_ڌاڳا * vtcm_سائيز_في_thread;
        }
        actx.dst_spad_بنياد = vtcm_ptr;
        vtcm_ptr += n_ڌاڳا * vtcm_سائيز_في_thread;
        إذا (has_جوڙيو) {
            actx.res_spad_بنياد = vtcm_ptr;
            vtcm_ptr += (actx.is_صف_bcast ? 1 : n_ڌاڳا) * vtcm_سائيز_في_thread;
        }

        إذا (has_جوڙيو && actx.is_صف_bcast) {
            ثابت dma_addr_t r_ddr = octx->src[2 * n_ranks]->معلومات;
            ثابت uint32_t صف_بائيٽون = actx.ne0 * (dst->type == HTP_TYPE_F16 ? حجم(__fp16) : حجم(عائم));
            dma_قطار * dma_q = octx->ctx->dma[0];
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(actx.res_spad_بنياد, r_ddr), actx.صف_سائيز_aligned, 0, صف_بائيٽون, 1);
            dma_قطار_ڪڍو(dma_q);
        }

        work_قطار_لاڳيتو(octx->ctx->work_قطار, reduce_fun, &actx, n_ڌاڳا);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_شروعات(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    // drain fan-out DMA writes before exit barrier
    إذا (گهڻائي == HTP_ALLREDUCE_SHARDED_FANOUT) {
        asm متطاير ("syncht" : : : "memory");
    }

    htp_fence_write(my_fence, fence_seq_exit, octx->حالت);

    لكل (uint32_t j = 0; j < n_ranks; j++) {
        إذا (j == rank) استمر;
        ثابت بنية htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->معلومات;
        uint64_t spins = 0;
        طالما (1) {
            uint32_t peer_seq;
            uint32_t peer_حالت;
            htp_fence_read(peer_fence, &peer_seq, &peer_حالت);
            إذا ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                إذا (peer_حالت > HTP_حالت_OK) {
                    FARF(ERROR, "ggml-hex: allreduce exit peer %u failed with status %u\n", j, peer_حالت);
                    htp_fence_write(my_fence, fence_seq_exit, peer_حالت);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                    أرجع peer_حالت;
                }
                اكسر;
            }
            إذا (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce exit fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_exit, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_حالت_اندروني_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                أرجع HTP_حالت_اندروني_ERR;
            }
            hex_pause();
        }
    }
    asm متطاير ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    أرجع octx->حالت;
}
