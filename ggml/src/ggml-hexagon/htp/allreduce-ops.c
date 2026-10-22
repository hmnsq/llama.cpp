#رهنمود clang diagnostic ignored "-Wunused-variable"
#رهنمود clang diagnostic ignored "-Wunused-function"
#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>
#ضمن <stdatomic.h>
#ضمن <math.h>
#ضمن <string.h>

#تعریف_کن GGML_COMMON_DECL_C
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
    uint32_t ردیف_اندازه_aligned;
    uint32_t rank_elem_شروع;
    uint32_t rank_nelem;
    uint32_t elems_در_هر_thread;
    uint32_t بلوک_elems;
    uint32_t vtcm_اندازه_در_هر_thread;
    bool     is_ردیف_bcast;
    uint8_t * src_spad_پایه[HTP_ALLREDUCE_MAX_RANKS];
    uint8_t * dst_spad_پایه;
    uint8_t * res_spad_پایه;
};

#تعریف_کن DEFINE_ALLREDUCE_THREAD_DMA_1D(SUFFIX, TYPE, HVX_جمع_کن_FN, HAS_جمع_کن)                                         \
ساكن فراغ allreduce_thread_dma_1d_##SUFFIX(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {                   \
    بنية htp_allreduce_context * actx = (بنية htp_allreduce_context *) داده;                                  \
    بنية htp_ops_context * octx = actx->octx;                                                                   \
                                                                                                                  \
    ثابت uint32_t n_ranks     = actx->n_ranks;                                                                   \
    ثابت uint32_t n_dsts      = actx->n_dsts;                                                                    \
    ثابت uint32_t بلوک_elems = actx->بلوک_elems;                                                               \
                                                                                                                  \
    ثابت uint32_t dr  = actx->elems_در_هر_thread;                                                                  \
    ثابت uint32_t ir0 = actx->rank_elem_شروع + dr * ith;                                                        \
    ثابت uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_شروع + actx->rank_nelem);                                 \
    إذا (ir0 >= ir1) أرجع;                                                                                       \
                                                                                                                  \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                                        \
    dma_صف * dma_q = octx->ctx->dma[ith];                                                                      \
                                                                                                                  \
    ثابت size_t vtcm_thread_جابه_جایی = ith * actx->vtcm_اندازه_در_هر_thread;                                           \
    uint8_t * dst_spad_پایه = actx->dst_spad_پایه + vtcm_thread_جابه_جایی;                                           \
    uint8_t * res_spad_پایه = HAS_جمع_کن ? (actx->res_spad_پایه + vtcm_thread_جابه_جایی) : عدم;                        \
                                                                                                                  \
    ثابت size_t spad_half = actx->vtcm_اندازه_در_هر_thread / 2;                                                      \
    uint32_t ir_prefetch = ir0;                                                                                   \
    صحيح spad_idx = 0;                                                                                             \
                                                                                                                  \
    لكل (صحيح k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                            \
        uint32_t cur_elems = MIN(بلوک_elems, ir1 - ir_prefetch);                                                 \
        size_t   cur_بایتها = cur_elems * حجم(TYPE);                                                            \
        uint8_t * d_spad = dst_spad_پایه + spad_idx * spad_half;                                                  \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->داده + ir_prefetch * حجم(TYPE);                                  \
            dma_صف_افزودن(dma_q, dma_make_داده(d_ddr, d_spad), cur_بایتها, cur_بایتها, cur_بایتها, 0);              \
        }                                                                                                         \
        لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            uint8_t * s_spad = actx->src_spad_پایه[s] + vtcm_thread_جابه_جایی + spad_idx * spad_half;                \
            ثابت dma_addr_t s_ddr = octx->src[s]->داده + ir_prefetch * حجم(TYPE);                             \
            dma_صف_افزودن(dma_q, dma_make_داده(s_spad, s_ddr), cur_بایتها, cur_بایتها, cur_بایتها, 1);              \
        }                                                                                                         \
        إذا (HAS_جمع_کن) {                                                                                            \
            uint8_t * r_spad = res_spad_پایه + spad_idx * spad_half;                                              \
            ثابت dma_addr_t r_ddr = octx->src[2 * n_ranks]->داده + ir_prefetch * حجم(TYPE);                   \
            dma_صف_افزودن(dma_q, dma_make_داده(r_spad, r_ddr), cur_بایتها, cur_بایتها, cur_بایتها, 1);              \
        }                                                                                                         \
        ir_prefetch += cur_elems;                                                                                 \
        spad_idx ^= 1;                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    صحيح comp_spad_idx = 0;                                                                                        \
    لكل (uint32_t ir = ir0; ir < ir1; ) {                                                                         \
        uint32_t cur_elems = MIN(بلوک_elems, ir1 - ir);                                                          \
        size_t   cur_بایتها = cur_elems * حجم(TYPE);                                                            \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_صف_برداشت(dma_q);                                                                                 \
        }                                                                                                         \
        لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            dma_صف_برداشت(dma_q);                                                                                 \
        }                                                                                                         \
        إذا (HAS_جمع_کن) {                                                                                            \
            dma_صف_برداشت(dma_q);                                                                                 \
        }                                                                                                         \
        uint8_t * d_spad  = dst_spad_پایه + comp_spad_idx * spad_half;                                            \
        uint8_t * s0_spad = actx->src_spad_پایه[0] + vtcm_thread_جابه_جایی + comp_spad_idx * spad_half;              \
        uint8_t * s1_spad = actx->src_spad_پایه[1] + vtcm_thread_جابه_جایی + comp_spad_idx * spad_half;              \
        uint8_t * r_spad  = HAS_جمع_کن ? (res_spad_پایه + comp_spad_idx * spad_half) : عدم;                         \
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                         \
        HVX_جمع_کن_FN(d_spad, s0_spad, s1_spad, cur_elems);                                                          \
        لكل (uint32_t s = 2; s < n_ranks; s++) {                                                                  \
            uint8_t * ss_spad = actx->src_spad_پایه[s] + vtcm_thread_جابه_جایی + comp_spad_idx * spad_half;          \
            HVX_جمع_کن_FN(d_spad, d_spad, ss_spad, cur_elems);                                                       \
        }                                                                                                         \
        إذا (HAS_جمع_کن) {                                                                                            \
            HVX_جمع_کن_FN(d_spad, d_spad, r_spad, cur_elems);                                                        \
        }                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                          \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->داده + ir * حجم(TYPE);                                           \
            dma_صف_افزودن(dma_q, dma_make_داده(d_ddr, d_spad), cur_بایتها, cur_بایتها, cur_بایتها, 1);              \
        }                                                                                                         \
        إذا (ir_prefetch < ir1) {                                                                                  \
            uint32_t بعدی_elems = MIN(بلوک_elems, ir1 - ir_prefetch);                                            \
            size_t   بعدی_بایتها = بعدی_elems * حجم(TYPE);                                                      \
            لكل (uint32_t s = 0; s < n_ranks; s++) {                                                              \
                uint8_t * s_spad = actx->src_spad_پایه[s] + vtcm_thread_جابه_جایی + comp_spad_idx * spad_half;       \
                ثابت dma_addr_t s_بعدی = octx->src[s]->داده + ir_prefetch * حجم(TYPE);                        \
                dma_صف_افزودن(dma_q, dma_make_داده(s_spad, s_بعدی), بعدی_بایتها, بعدی_بایتها, بعدی_بایتها, 1);      \
            }                                                                                                     \
            إذا (HAS_جمع_کن) {                                                                                        \
                uint8_t * r_spad_بعدی = res_spad_پایه + comp_spad_idx * spad_half;                                \
                ثابت dma_addr_t r_بعدی = octx->src[2 * n_ranks]->داده + ir_prefetch * حجم(TYPE);              \
                dma_صف_افزودن(dma_q, dma_make_داده(r_spad_بعدی, r_بعدی), بعدی_بایتها, بعدی_بایتها, بعدی_بایتها, 1); \
            }                                                                                                     \
            ir_prefetch += بعدی_elems;                                                                            \
        }                                                                                                         \
        comp_spad_idx ^= 1;                                                                                       \
        ir += cur_elems;                                                                                          \
    }                                                                                                             \
    dma_صف_flush(dma_q);                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_جمع_کن_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     عائم,  hvx_جمع_کن_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(جمع_کن_f16, __fp16, hvx_جمع_کن_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(جمع_کن_f32, عائم,  hvx_جمع_کن_f32_aaa, 1)

#تعریف_کن DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_جمع_کن_FN, HAS_جمع_کن, IS_ردیف_BCAST)                                                           \
ساكن فراغ allreduce_thread_dma_2d_##SUFFIX(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {                                                   \
    بنية htp_allreduce_context * actx = (بنية htp_allreduce_context *) داده;                                                                  \
    بنية htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    ثابت uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    ثابت uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    ثابت uint32_t ne0              = actx->ne0;                                                                                                  \
    ثابت uint32_t بلوک_ردیفها       = actx->بلوک_elems;                                                                                          \
    ثابت uint32_t ردیف_اندازه_aligned = actx->ردیف_اندازه_aligned;                                                                                     \
    ثابت uint32_t ردیف_بایتها        = ne0 * حجم(TYPE);                                                                                         \
                                                                                                                                                  \
    ثابت uint32_t dr  = actx->elems_در_هر_thread;                                                                                                  \
    ثابت uint32_t r0  = actx->rank_elem_شروع + dr * ith;                                                                                        \
    ثابت uint32_t r1  = MIN(r0 + dr, actx->rank_elem_شروع + actx->rank_nelem);                                                                  \
    إذا (r0 >= r1) أرجع;                                                                                                                         \
                                                                                                                                                  \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_صف * dma_q = octx->ctx->dma[ith];                                                                                                      \
                                                                                                                                                  \
    ثابت size_t vtcm_thread_جابه_جایی = ith * actx->vtcm_اندازه_در_هر_thread;                                                                           \
    uint8_t * dst_spad_پایه = actx->dst_spad_پایه + vtcm_thread_جابه_جایی;                                                                           \
    uint8_t * res_spad_پایه = HAS_جمع_کن ? (IS_ردیف_BCAST ? actx->res_spad_پایه : (actx->res_spad_پایه + vtcm_thread_جابه_جایی)) : عدم;                 \
                                                                                                                                                  \
    ثابت size_t spad_half = actx->vtcm_اندازه_در_هر_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    صحيح spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    لكل (صحيح k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_ردیفها = MIN(بلوک_ردیفها, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_پایه + spad_idx * spad_half;                                                                                  \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->داده + r_prefetch * octx->dsts[d]->nb[1];                                                           \
            dma_صف_افزودن(dma_q, dma_make_داده(d_ddr, d_spad), octx->dsts[d]->nb[1], ردیف_اندازه_aligned, ردیف_بایتها, 0);                            \
        }                                                                                                                                         \
        لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = actx->src_spad_پایه[s] + vtcm_thread_جابه_جایی + spad_idx * spad_half;                                                \
            ثابت dma_addr_t s_ddr = octx->src[s]->داده + r_prefetch * octx->src[s]->nb[1];                                                       \
            dma_صف_افزودن(dma_q, dma_make_داده(s_spad, s_ddr), ردیف_اندازه_aligned, octx->src[s]->nb[1], ردیف_بایتها, cur_ردیفها);                      \
        }                                                                                                                                         \
        إذا (HAS_جمع_کن && !IS_ردیف_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_پایه + spad_idx * spad_half;                                                                              \
            ثابت dma_addr_t r_ddr = octx->src[2 * n_ranks]->داده + r_prefetch * octx->src[2 * n_ranks]->nb[1];                                   \
            dma_صف_افزودن(dma_q, dma_make_داده(r_spad, r_ddr), ردیف_اندازه_aligned, octx->src[2 * n_ranks]->nb[1], ردیف_بایتها, cur_ردیفها);            \
        }                                                                                                                                         \
        r_prefetch += cur_ردیفها;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    صحيح comp_spad_idx = 0;                                                                                                                        \
    لكل (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_ردیفها = MIN(بلوک_ردیفها, r1 - r);                                                                                              \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_صف_برداشت(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            dma_صف_برداشت(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        إذا (HAS_جمع_کن && !IS_ردیف_BCAST) {                                                                                                           \
            dma_صف_برداشت(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        uint8_t * d_spad  = dst_spad_پایه + comp_spad_idx * spad_half;                                                                            \
        uint8_t * s0_spad = actx->src_spad_پایه[0] + vtcm_thread_جابه_جایی + comp_spad_idx * spad_half;                                              \
        uint8_t * s1_spad = actx->src_spad_پایه[1] + vtcm_thread_جابه_جایی + comp_spad_idx * spad_half;                                              \
        uint8_t * r_spad  = (HAS_جمع_کن && !IS_ردیف_BCAST) ? (res_spad_پایه + comp_spad_idx * spad_half) : عدم;                                      \
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        لكل (uint32_t ردیف = 0; ردیف < cur_ردیفها; ردیف++) {                                                                                           \
            uint8_t * d_ردیف = d_spad + ردیف * ردیف_اندازه_aligned;                                                                                    \
            ثابت uint8_t * s0_ردیف = s0_spad + ردیف * ردیف_اندازه_aligned;                                                                            \
            ثابت uint8_t * s1_ردیف = s1_spad + ردیف * ردیف_اندازه_aligned;                                                                            \
            HVX_جمع_کن_FN(d_ردیف, s0_ردیف, s1_ردیف, ne0);                                                                                               \
            لكل (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                ثابت uint8_t * ss_ردیف = actx->src_spad_پایه[s] + vtcm_thread_جابه_جایی + comp_spad_idx * spad_half + ردیف * ردیف_اندازه_aligned;        \
                HVX_جمع_کن_FN(d_ردیف, d_ردیف, ss_ردیف, ne0);                                                                                            \
            }                                                                                                                                     \
            إذا (HAS_جمع_کن) {                                                                                                                        \
                ثابت uint8_t * res_ردیف = IS_ردیف_BCAST ? res_spad_پایه : (r_spad + ردیف * ردیف_اندازه_aligned);                                       \
                HVX_جمع_کن_FN(d_ردیف, d_ردیف, res_ردیف, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->داده + r * octx->dsts[d]->nb[1];                                                                    \
            dma_صف_افزودن(dma_q, dma_make_داده(d_ddr, d_spad), octx->dsts[d]->nb[1], ردیف_اندازه_aligned, ردیف_بایتها, cur_ردیفها);                     \
        }                                                                                                                                         \
        إذا (r_prefetch < r1) {                                                                                                                    \
            uint32_t بعدی_ردیفها = MIN(بلوک_ردیفها, r1 - r_prefetch);                                                                                \
            لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                uint8_t * s_spad = actx->src_spad_پایه[s] + vtcm_thread_جابه_جایی + comp_spad_idx * spad_half;                                       \
                ثابت dma_addr_t s_بعدی = octx->src[s]->داده + r_prefetch * octx->src[s]->nb[1];                                                  \
                dma_صف_افزودن(dma_q, dma_make_داده(s_spad, s_بعدی), ردیف_اندازه_aligned, octx->src[s]->nb[1], ردیف_بایتها, بعدی_ردیفها);                \
            }                                                                                                                                     \
            إذا (HAS_جمع_کن && !IS_ردیف_BCAST) {                                                                                                       \
                uint8_t * r_spad_بعدی = res_spad_پایه + comp_spad_idx * spad_half;                                                                \
                ثابت dma_addr_t r_بعدی = octx->src[2 * n_ranks]->داده + r_prefetch * octx->src[2 * n_ranks]->nb[1];                              \
                dma_صف_افزودن(dma_q, dma_make_داده(r_spad_بعدی, r_بعدی), ردیف_اندازه_aligned, octx->src[2 * n_ranks]->nb[1], ردیف_بایتها, بعدی_ردیفها); \
            }                                                                                                                                     \
            r_prefetch += بعدی_ردیفها;                                                                                                              \
        }                                                                                                                                         \
        comp_spad_idx ^= 1;                                                                                                                       \
        r += cur_ردیفها;                                                                                                                            \
    }                                                                                                                                             \
    dma_صف_flush(dma_q);                                                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_جمع_کن_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           عائم,  hvx_جمع_کن_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جمع_کن_f16,       __fp16, hvx_جمع_کن_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جمع_کن_f32,       عائم,  hvx_جمع_کن_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جمع_کن_bcast_f16, __fp16, hvx_جمع_کن_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جمع_کن_bcast_f32, عائم,  hvx_جمع_کن_f32_aaa, 1, 1)

ساكن صحيح validate_allreduce(
    بنية htp_ops_context * octx,
    ثابت بنية htp_allreduce_kernel_params * kparams,
    uint32_t n_ranks
) {
    إذا (kparams->n_نخها == 0 || (uint32_t) kparams->n_نخها > octx->ctx->n_نخها) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }
    octx->n_نخها = (uint32_t) kparams->n_نخها;

    إذا (kparams->vtcm_اندازه_در_هر_thread <= 0 || kparams->vtcm_اندازه <= 0) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    ثابت bool has_جمع_کن = (octx->op == HTP_OP_ALLREDUCE_جمع_کن);
    ثابت size_t n_vtcm_buffers = htp_allreduce_vtcm_میانگیر_شمار(
        n_ranks, octx->n_نخها, has_جمع_کن, kparams->is_ردیف_bcast != 0);
    ثابت size_t vtcm_اندازه = n_vtcm_buffers * (size_t) kparams->vtcm_اندازه_در_هر_thread;
    إذا (vtcm_اندازه != (size_t) kparams->vtcm_اندازه) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }
    إذا (vtcm_اندازه > octx->ctx->vtcm_اندازه) {
        أرجع HTP_وضعیت_VTCM_TOO_SMALL;
    }

    إذا (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    أرجع HTP_وضعیت_OK;
}

صحيح op_allreduce(بنية htp_ops_context * octx) {
    إذا (octx->ctx->mdev.شمار > 1 && octx->ctx->mdev.idx > 0) {
        أرجع HTP_وضعیت_OK;
    }

    ثابت بنية htp_allreduce_kernel_params * kparams = (ثابت بنية htp_allreduce_kernel_params *) octx->kernel_params;
    ثابت بنية htp_tensor * dst = octx->dst;

    ثابت uint32_t rank    = (uint32_t) kparams->rank;
    ثابت uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    إذا (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    ثابت uint32_t fence_seq_مدخل = (uint32_t) octx->op_params[0];
    ثابت uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    ثابت بنية htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->داده;

    ثابت صحيح وضعیت = validate_allreduce(octx, kparams, n_ranks);
    إذا (وضعیت != HTP_وضعیت_OK) {
        إذا (وضعیت == HTP_وضعیت_NO_SUPPORT) {
            FARF(ERROR, "ggml-hex: allreduce unsupported type %d : rank %u\n", dst->type, rank);
        }
        htp_fence_write(my_fence, fence_seq_exit, وضعیت);
        أرجع وضعیت;
    }

    ثابت bool has_جمع_کن = (octx->op == HTP_OP_ALLREDUCE_جمع_کن);
    ثابت uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت int32_t  نما  = kparams->نما;

    // 1. Entry Barrier: Synchronize all ranks before reading
    بنية htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_شروع(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_مدخل);

    htp_fence_write(my_fence, fence_seq_مدخل, octx->وضعیت);

    لكل (uint32_t j = 0; j < n_ranks; j++) {
        إذا (j == rank) استمر;
        ثابت بنية htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->داده;
        uint64_t spins = 0;
        طالما (1) {
            uint32_t peer_seq;
            uint32_t peer_وضعیت;
            htp_fence_read(peer_fence, &peer_seq, &peer_وضعیت);
            إذا ((int32_t)(peer_seq - fence_seq_مدخل) >= 0) {
                إذا (peer_وضعیت > HTP_وضعیت_OK) {
                    FARF(ERROR, "ggml-hex: allreduce entry peer %u failed with status %u\n", j, peer_وضعیت);
                    htp_fence_write(my_fence, fence_seq_exit, peer_وضعیت);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_مدخل);
                    أرجع peer_وضعیت;
                }
                اكسر;
            }
            إذا (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_مدخل, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_وضعیت_داخلی_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_مدخل);
                أرجع HTP_وضعیت_داخلی_ERR;
            }
            hex_pause();
        }
    }
    asm متطاير ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_مدخل);

    // 2. Multi-threaded Reduction across assigned rank chunk
    إذا (nelem > 0) {
        ثابت uint32_t n_نخها            = (uint32_t) kparams->n_نخها;
        ثابت uint32_t بلوک_elems          = (uint32_t) kparams->بلوک_elems;
        ثابت uint32_t elems_در_هر_thread     = (uint32_t) kparams->elems_در_هر_thread;
        ثابت uint32_t vtcm_اندازه_در_هر_thread = (uint32_t) kparams->vtcm_اندازه_در_هر_thread;

        بنية htp_allreduce_context actx;
        actx.octx                 = octx;
        actx.n_ranks              = n_ranks;
        actx.n_dsts               = (uint32_t) kparams->n_dsts ? (uint32_t) kparams->n_dsts : n_ranks;
        actx.nelem                = nelem;
        actx.ne0                  = (uint32_t) kparams->ne0;
        actx.ne1                  = (uint32_t) kparams->ne1;
        actx.ردیف_اندازه_aligned     = (uint32_t) kparams->ردیف_اندازه_aligned;
        actx.rank_elem_شروع      = (uint32_t) kparams->rank_elem_شروع;
        actx.rank_nelem           = (uint32_t) kparams->rank_nelem;
        actx.elems_در_هر_thread     = elems_در_هر_thread;
        actx.بلوک_elems          = بلوک_elems;
        actx.vtcm_اندازه_در_هر_thread = vtcm_اندازه_در_هر_thread;
        actx.is_ردیف_bcast         = (kparams->is_ردیف_bcast != 0);

        work_صف_func_t reduce_fun = عدم;
        بدل (kparams->kernel_type) {
            حالة HTP_ALLREDUCE_KERNEL_DMA_1D:
                إذا (has_جمع_کن) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_جمع_کن_f16 : allreduce_thread_dma_1d_جمع_کن_f32;
                } وإلا {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                اكسر;
            حالة HTP_ALLREDUCE_KERNEL_DMA_2D:
                إذا (has_جمع_کن) {
                    إذا (kparams->is_ردیف_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_جمع_کن_bcast_f16 : allreduce_thread_dma_2d_جمع_کن_bcast_f32;
                    } وإلا {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_جمع_کن_f16 : allreduce_thread_dma_2d_جمع_کن_f32;
                    }
                } وإلا {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_f16 : allreduce_thread_dma_2d_f32;
                }
                اكسر;
            افتراضي:
                FARF(ERROR, "ggml-hex: allreduce unsupported kernel %d : rank %u\n", kparams->kernel_type, rank);
                htp_fence_write(my_fence, fence_seq_exit, HTP_وضعیت_NO_SUPPORT);
                أرجع HTP_وضعیت_NO_SUPPORT;
        }

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_پایه;
        لكل (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_پایه[s] = vtcm_ptr;
            vtcm_ptr += n_نخها * vtcm_اندازه_در_هر_thread;
        }
        actx.dst_spad_پایه = vtcm_ptr;
        vtcm_ptr += n_نخها * vtcm_اندازه_در_هر_thread;
        إذا (has_جمع_کن) {
            actx.res_spad_پایه = vtcm_ptr;
            vtcm_ptr += (actx.is_ردیف_bcast ? 1 : n_نخها) * vtcm_اندازه_در_هر_thread;
        }

        إذا (has_جمع_کن && actx.is_ردیف_bcast) {
            ثابت dma_addr_t r_ddr = octx->src[2 * n_ranks]->داده;
            ثابت uint32_t ردیف_بایتها = actx.ne0 * (dst->type == HTP_TYPE_F16 ? حجم(__fp16) : حجم(عائم));
            dma_صف * dma_q = octx->ctx->dma[0];
            dma_صف_افزودن(dma_q, dma_make_داده(actx.res_spad_پایه, r_ddr), actx.ردیف_اندازه_aligned, 0, ردیف_بایتها, 1);
            dma_صف_برداشت(dma_q);
        }

        work_صف_پیاپی(octx->ctx->work_صف, reduce_fun, &actx, n_نخها);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_شروع(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    // drain fan-out DMA writes before exit barrier
    إذا (نما == HTP_ALLREDUCE_SHARDED_FANOUT) {
        asm متطاير ("syncht" : : : "memory");
    }

    htp_fence_write(my_fence, fence_seq_exit, octx->وضعیت);

    لكل (uint32_t j = 0; j < n_ranks; j++) {
        إذا (j == rank) استمر;
        ثابت بنية htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->داده;
        uint64_t spins = 0;
        طالما (1) {
            uint32_t peer_seq;
            uint32_t peer_وضعیت;
            htp_fence_read(peer_fence, &peer_seq, &peer_وضعیت);
            إذا ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                إذا (peer_وضعیت > HTP_وضعیت_OK) {
                    FARF(ERROR, "ggml-hex: allreduce exit peer %u failed with status %u\n", j, peer_وضعیت);
                    htp_fence_write(my_fence, fence_seq_exit, peer_وضعیت);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                    أرجع peer_وضعیت;
                }
                اكسر;
            }
            إذا (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce exit fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_exit, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_وضعیت_داخلی_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                أرجع HTP_وضعیت_داخلی_ERR;
            }
            hex_pause();
        }
    }
    asm متطاير ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    أرجع octx->وضعیت;
}
