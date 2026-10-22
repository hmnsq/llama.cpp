#لارښوونه clang diagnostic ignored "-Wunused-variable"
#لارښوونه clang diagnostic ignored "-Wunused-function"
#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>
#ضمن <stdatomic.h>
#ضمن <math.h>
#ضمن <string.h>

#تعریف_کړه GGML_COMMON_DECL_C
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
    uint32_t لیکه_اندازه_aligned;
    uint32_t rank_elem_پیل;
    uint32_t rank_nelem;
    uint32_t elems_فی_thread;
    uint32_t بلاک_elems;
    uint32_t vtcm_اندازه_فی_thread;
    bool     is_لیکه_bcast;
    uint8_t * src_spad_بنسټ[HTP_ALLREDUCE_MAX_RANKS];
    uint8_t * dst_spad_بنسټ;
    uint8_t * res_spad_بنسټ;
};

#تعریف_کړه DEFINE_ALLREDUCE_THREAD_DMA_1D(SUFFIX, TYPE, HVX_جمع_کړه_FN, HAS_جمع_کړه)                                         \
ساكن فراغ allreduce_thread_dma_1d_##SUFFIX(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {                   \
    بنية htp_allreduce_context * actx = (بنية htp_allreduce_context *) معلومات;                                  \
    بنية htp_ops_context * octx = actx->octx;                                                                   \
                                                                                                                  \
    ثابت uint32_t n_ranks     = actx->n_ranks;                                                                   \
    ثابت uint32_t n_dsts      = actx->n_dsts;                                                                    \
    ثابت uint32_t بلاک_elems = actx->بلاک_elems;                                                               \
                                                                                                                  \
    ثابت uint32_t dr  = actx->elems_فی_thread;                                                                  \
    ثابت uint32_t ir0 = actx->rank_elem_پیل + dr * ith;                                                        \
    ثابت uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_پیل + actx->rank_nelem);                                 \
    إذا (ir0 >= ir1) أرجع;                                                                                       \
                                                                                                                  \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                                        \
    dma_کتار * dma_q = octx->ctx->dma[ith];                                                                      \
                                                                                                                  \
    ثابت size_t vtcm_thread_دننه_واټن = ith * actx->vtcm_اندازه_فی_thread;                                           \
    uint8_t * dst_spad_بنسټ = actx->dst_spad_بنسټ + vtcm_thread_دننه_واټن;                                           \
    uint8_t * res_spad_بنسټ = HAS_جمع_کړه ? (actx->res_spad_بنسټ + vtcm_thread_دننه_واټن) : عدم;                        \
                                                                                                                  \
    ثابت size_t spad_half = actx->vtcm_اندازه_فی_thread / 2;                                                      \
    uint32_t ir_prefetch = ir0;                                                                                   \
    صحيح spad_idx = 0;                                                                                             \
                                                                                                                  \
    لكل (صحيح k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                            \
        uint32_t cur_elems = MIN(بلاک_elems, ir1 - ir_prefetch);                                                 \
        size_t   cur_بایټونه = cur_elems * حجم(TYPE);                                                            \
        uint8_t * d_spad = dst_spad_بنسټ + spad_idx * spad_half;                                                  \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->معلومات + ir_prefetch * حجم(TYPE);                                  \
            dma_کتار_ټېل(dma_q, dma_make_معلومات(d_ddr, d_spad), cur_بایټونه, cur_بایټونه, cur_بایټونه, 0);              \
        }                                                                                                         \
        لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            uint8_t * s_spad = actx->src_spad_بنسټ[s] + vtcm_thread_دننه_واټن + spad_idx * spad_half;                \
            ثابت dma_addr_t s_ddr = octx->src[s]->معلومات + ir_prefetch * حجم(TYPE);                             \
            dma_کتار_ټېل(dma_q, dma_make_معلومات(s_spad, s_ddr), cur_بایټونه, cur_بایټونه, cur_بایټونه, 1);              \
        }                                                                                                         \
        إذا (HAS_جمع_کړه) {                                                                                            \
            uint8_t * r_spad = res_spad_بنسټ + spad_idx * spad_half;                                              \
            ثابت dma_addr_t r_ddr = octx->src[2 * n_ranks]->معلومات + ir_prefetch * حجم(TYPE);                   \
            dma_کتار_ټېل(dma_q, dma_make_معلومات(r_spad, r_ddr), cur_بایټونه, cur_بایټونه, cur_بایټونه, 1);              \
        }                                                                                                         \
        ir_prefetch += cur_elems;                                                                                 \
        spad_idx ^= 1;                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    صحيح comp_spad_idx = 0;                                                                                        \
    لكل (uint32_t ir = ir0; ir < ir1; ) {                                                                         \
        uint32_t cur_elems = MIN(بلاک_elems, ir1 - ir);                                                          \
        size_t   cur_بایټونه = cur_elems * حجم(TYPE);                                                            \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_کتار_راوباسه(dma_q);                                                                                 \
        }                                                                                                         \
        لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            dma_کتار_راوباسه(dma_q);                                                                                 \
        }                                                                                                         \
        إذا (HAS_جمع_کړه) {                                                                                            \
            dma_کتار_راوباسه(dma_q);                                                                                 \
        }                                                                                                         \
        uint8_t * d_spad  = dst_spad_بنسټ + comp_spad_idx * spad_half;                                            \
        uint8_t * s0_spad = actx->src_spad_بنسټ[0] + vtcm_thread_دننه_واټن + comp_spad_idx * spad_half;              \
        uint8_t * s1_spad = actx->src_spad_بنسټ[1] + vtcm_thread_دننه_واټن + comp_spad_idx * spad_half;              \
        uint8_t * r_spad  = HAS_جمع_کړه ? (res_spad_بنسټ + comp_spad_idx * spad_half) : عدم;                         \
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                         \
        HVX_جمع_کړه_FN(d_spad, s0_spad, s1_spad, cur_elems);                                                          \
        لكل (uint32_t s = 2; s < n_ranks; s++) {                                                                  \
            uint8_t * ss_spad = actx->src_spad_بنسټ[s] + vtcm_thread_دننه_واټن + comp_spad_idx * spad_half;          \
            HVX_جمع_کړه_FN(d_spad, d_spad, ss_spad, cur_elems);                                                       \
        }                                                                                                         \
        إذا (HAS_جمع_کړه) {                                                                                            \
            HVX_جمع_کړه_FN(d_spad, d_spad, r_spad, cur_elems);                                                        \
        }                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                          \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->معلومات + ir * حجم(TYPE);                                           \
            dma_کتار_ټېل(dma_q, dma_make_معلومات(d_ddr, d_spad), cur_بایټونه, cur_بایټونه, cur_بایټونه, 1);              \
        }                                                                                                         \
        إذا (ir_prefetch < ir1) {                                                                                  \
            uint32_t بل_elems = MIN(بلاک_elems, ir1 - ir_prefetch);                                            \
            size_t   بل_بایټونه = بل_elems * حجم(TYPE);                                                      \
            لكل (uint32_t s = 0; s < n_ranks; s++) {                                                              \
                uint8_t * s_spad = actx->src_spad_بنسټ[s] + vtcm_thread_دننه_واټن + comp_spad_idx * spad_half;       \
                ثابت dma_addr_t s_بل = octx->src[s]->معلومات + ir_prefetch * حجم(TYPE);                        \
                dma_کتار_ټېل(dma_q, dma_make_معلومات(s_spad, s_بل), بل_بایټونه, بل_بایټونه, بل_بایټونه, 1);      \
            }                                                                                                     \
            إذا (HAS_جمع_کړه) {                                                                                        \
                uint8_t * r_spad_بل = res_spad_بنسټ + comp_spad_idx * spad_half;                                \
                ثابت dma_addr_t r_بل = octx->src[2 * n_ranks]->معلومات + ir_prefetch * حجم(TYPE);              \
                dma_کتار_ټېل(dma_q, dma_make_معلومات(r_spad_بل, r_بل), بل_بایټونه, بل_بایټونه, بل_بایټونه, 1); \
            }                                                                                                     \
            ir_prefetch += بل_elems;                                                                            \
        }                                                                                                         \
        comp_spad_idx ^= 1;                                                                                       \
        ir += cur_elems;                                                                                          \
    }                                                                                                             \
    dma_کتار_flush(dma_q);                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_جمع_کړه_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     عائم,  hvx_جمع_کړه_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(جمع_کړه_f16, __fp16, hvx_جمع_کړه_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(جمع_کړه_f32, عائم,  hvx_جمع_کړه_f32_aaa, 1)

#تعریف_کړه DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_جمع_کړه_FN, HAS_جمع_کړه, IS_لیکه_BCAST)                                                           \
ساكن فراغ allreduce_thread_dma_2d_##SUFFIX(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {                                                   \
    بنية htp_allreduce_context * actx = (بنية htp_allreduce_context *) معلومات;                                                                  \
    بنية htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    ثابت uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    ثابت uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    ثابت uint32_t ne0              = actx->ne0;                                                                                                  \
    ثابت uint32_t بلاک_لیکې       = actx->بلاک_elems;                                                                                          \
    ثابت uint32_t لیکه_اندازه_aligned = actx->لیکه_اندازه_aligned;                                                                                     \
    ثابت uint32_t لیکه_بایټونه        = ne0 * حجم(TYPE);                                                                                         \
                                                                                                                                                  \
    ثابت uint32_t dr  = actx->elems_فی_thread;                                                                                                  \
    ثابت uint32_t r0  = actx->rank_elem_پیل + dr * ith;                                                                                        \
    ثابت uint32_t r1  = MIN(r0 + dr, actx->rank_elem_پیل + actx->rank_nelem);                                                                  \
    إذا (r0 >= r1) أرجع;                                                                                                                         \
                                                                                                                                                  \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_کتار * dma_q = octx->ctx->dma[ith];                                                                                                      \
                                                                                                                                                  \
    ثابت size_t vtcm_thread_دننه_واټن = ith * actx->vtcm_اندازه_فی_thread;                                                                           \
    uint8_t * dst_spad_بنسټ = actx->dst_spad_بنسټ + vtcm_thread_دننه_واټن;                                                                           \
    uint8_t * res_spad_بنسټ = HAS_جمع_کړه ? (IS_لیکه_BCAST ? actx->res_spad_بنسټ : (actx->res_spad_بنسټ + vtcm_thread_دننه_واټن)) : عدم;                 \
                                                                                                                                                  \
    ثابت size_t spad_half = actx->vtcm_اندازه_فی_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    صحيح spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    لكل (صحيح k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_لیکې = MIN(بلاک_لیکې, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_بنسټ + spad_idx * spad_half;                                                                                  \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->معلومات + r_prefetch * octx->dsts[d]->nb[1];                                                           \
            dma_کتار_ټېل(dma_q, dma_make_معلومات(d_ddr, d_spad), octx->dsts[d]->nb[1], لیکه_اندازه_aligned, لیکه_بایټونه, 0);                            \
        }                                                                                                                                         \
        لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = actx->src_spad_بنسټ[s] + vtcm_thread_دننه_واټن + spad_idx * spad_half;                                                \
            ثابت dma_addr_t s_ddr = octx->src[s]->معلومات + r_prefetch * octx->src[s]->nb[1];                                                       \
            dma_کتار_ټېل(dma_q, dma_make_معلومات(s_spad, s_ddr), لیکه_اندازه_aligned, octx->src[s]->nb[1], لیکه_بایټونه, cur_لیکې);                      \
        }                                                                                                                                         \
        إذا (HAS_جمع_کړه && !IS_لیکه_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_بنسټ + spad_idx * spad_half;                                                                              \
            ثابت dma_addr_t r_ddr = octx->src[2 * n_ranks]->معلومات + r_prefetch * octx->src[2 * n_ranks]->nb[1];                                   \
            dma_کتار_ټېل(dma_q, dma_make_معلومات(r_spad, r_ddr), لیکه_اندازه_aligned, octx->src[2 * n_ranks]->nb[1], لیکه_بایټونه, cur_لیکې);            \
        }                                                                                                                                         \
        r_prefetch += cur_لیکې;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    صحيح comp_spad_idx = 0;                                                                                                                        \
    لكل (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_لیکې = MIN(بلاک_لیکې, r1 - r);                                                                                              \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_کتار_راوباسه(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            dma_کتار_راوباسه(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        إذا (HAS_جمع_کړه && !IS_لیکه_BCAST) {                                                                                                           \
            dma_کتار_راوباسه(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        uint8_t * d_spad  = dst_spad_بنسټ + comp_spad_idx * spad_half;                                                                            \
        uint8_t * s0_spad = actx->src_spad_بنسټ[0] + vtcm_thread_دننه_واټن + comp_spad_idx * spad_half;                                              \
        uint8_t * s1_spad = actx->src_spad_بنسټ[1] + vtcm_thread_دننه_واټن + comp_spad_idx * spad_half;                                              \
        uint8_t * r_spad  = (HAS_جمع_کړه && !IS_لیکه_BCAST) ? (res_spad_بنسټ + comp_spad_idx * spad_half) : عدم;                                      \
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        لكل (uint32_t لیکه = 0; لیکه < cur_لیکې; لیکه++) {                                                                                           \
            uint8_t * d_لیکه = d_spad + لیکه * لیکه_اندازه_aligned;                                                                                    \
            ثابت uint8_t * s0_لیکه = s0_spad + لیکه * لیکه_اندازه_aligned;                                                                            \
            ثابت uint8_t * s1_لیکه = s1_spad + لیکه * لیکه_اندازه_aligned;                                                                            \
            HVX_جمع_کړه_FN(d_لیکه, s0_لیکه, s1_لیکه, ne0);                                                                                               \
            لكل (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                ثابت uint8_t * ss_لیکه = actx->src_spad_بنسټ[s] + vtcm_thread_دننه_واټن + comp_spad_idx * spad_half + لیکه * لیکه_اندازه_aligned;        \
                HVX_جمع_کړه_FN(d_لیکه, d_لیکه, ss_لیکه, ne0);                                                                                            \
            }                                                                                                                                     \
            إذا (HAS_جمع_کړه) {                                                                                                                        \
                ثابت uint8_t * res_لیکه = IS_لیکه_BCAST ? res_spad_بنسټ : (r_spad + لیکه * لیکه_اندازه_aligned);                                       \
                HVX_جمع_کړه_FN(d_لیکه, d_لیکه, res_لیکه, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        لكل (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->معلومات + r * octx->dsts[d]->nb[1];                                                                    \
            dma_کتار_ټېل(dma_q, dma_make_معلومات(d_ddr, d_spad), octx->dsts[d]->nb[1], لیکه_اندازه_aligned, لیکه_بایټونه, cur_لیکې);                     \
        }                                                                                                                                         \
        إذا (r_prefetch < r1) {                                                                                                                    \
            uint32_t بل_لیکې = MIN(بلاک_لیکې, r1 - r_prefetch);                                                                                \
            لكل (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                uint8_t * s_spad = actx->src_spad_بنسټ[s] + vtcm_thread_دننه_واټن + comp_spad_idx * spad_half;                                       \
                ثابت dma_addr_t s_بل = octx->src[s]->معلومات + r_prefetch * octx->src[s]->nb[1];                                                  \
                dma_کتار_ټېل(dma_q, dma_make_معلومات(s_spad, s_بل), لیکه_اندازه_aligned, octx->src[s]->nb[1], لیکه_بایټونه, بل_لیکې);                \
            }                                                                                                                                     \
            إذا (HAS_جمع_کړه && !IS_لیکه_BCAST) {                                                                                                       \
                uint8_t * r_spad_بل = res_spad_بنسټ + comp_spad_idx * spad_half;                                                                \
                ثابت dma_addr_t r_بل = octx->src[2 * n_ranks]->معلومات + r_prefetch * octx->src[2 * n_ranks]->nb[1];                              \
                dma_کتار_ټېل(dma_q, dma_make_معلومات(r_spad_بل, r_بل), لیکه_اندازه_aligned, octx->src[2 * n_ranks]->nb[1], لیکه_بایټونه, بل_لیکې); \
            }                                                                                                                                     \
            r_prefetch += بل_لیکې;                                                                                                              \
        }                                                                                                                                         \
        comp_spad_idx ^= 1;                                                                                                                       \
        r += cur_لیکې;                                                                                                                            \
    }                                                                                                                                             \
    dma_کتار_flush(dma_q);                                                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_جمع_کړه_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           عائم,  hvx_جمع_کړه_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جمع_کړه_f16,       __fp16, hvx_جمع_کړه_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جمع_کړه_f32,       عائم,  hvx_جمع_کړه_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جمع_کړه_bcast_f16, __fp16, hvx_جمع_کړه_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(جمع_کړه_bcast_f32, عائم,  hvx_جمع_کړه_f32_aaa, 1, 1)

ساكن صحيح validate_allreduce(
    بنية htp_ops_context * octx,
    ثابت بنية htp_allreduce_kernel_params * kparams,
    uint32_t n_ranks
) {
    إذا (kparams->n_تارونه == 0 || (uint32_t) kparams->n_تارونه > octx->ctx->n_تارونه) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }
    octx->n_تارونه = (uint32_t) kparams->n_تارونه;

    إذا (kparams->vtcm_اندازه_فی_thread <= 0 || kparams->vtcm_اندازه <= 0) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت bool has_جمع_کړه = (octx->op == HTP_OP_ALLREDUCE_جمع_کړه);
    ثابت size_t n_vtcm_buffers = htp_allreduce_vtcm_بفر_شمېر(
        n_ranks, octx->n_تارونه, has_جمع_کړه, kparams->is_لیکه_bcast != 0);
    ثابت size_t vtcm_اندازه = n_vtcm_buffers * (size_t) kparams->vtcm_اندازه_فی_thread;
    إذا (vtcm_اندازه != (size_t) kparams->vtcm_اندازه) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }
    إذا (vtcm_اندازه > octx->ctx->vtcm_اندازه) {
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    إذا (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    أرجع HTP_حالت_OK;
}

صحيح op_allreduce(بنية htp_ops_context * octx) {
    إذا (octx->ctx->mdev.شمېر > 1 && octx->ctx->mdev.idx > 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت بنية htp_allreduce_kernel_params * kparams = (ثابت بنية htp_allreduce_kernel_params *) octx->kernel_params;
    ثابت بنية htp_tensor * dst = octx->dst;

    ثابت uint32_t rank    = (uint32_t) kparams->rank;
    ثابت uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    إذا (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t fence_seq_ثبت = (uint32_t) octx->op_params[0];
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

    ثابت bool has_جمع_کړه = (octx->op == HTP_OP_ALLREDUCE_جمع_کړه);
    ثابت uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ثابت int32_t  ډېرپېښ  = kparams->ډېرپېښ;

    // 1. Entry Barrier: Synchronize all ranks before reading
    بنية htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_پیل(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_ثبت);

    htp_fence_write(my_fence, fence_seq_ثبت, octx->حالت);

    لكل (uint32_t j = 0; j < n_ranks; j++) {
        إذا (j == rank) استمر;
        ثابت بنية htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->معلومات;
        uint64_t spins = 0;
        طالما (1) {
            uint32_t peer_seq;
            uint32_t peer_حالت;
            htp_fence_read(peer_fence, &peer_seq, &peer_حالت);
            إذا ((int32_t)(peer_seq - fence_seq_ثبت) >= 0) {
                إذا (peer_حالت > HTP_حالت_OK) {
                    FARF(ERROR, "ggml-hex: allreduce entry peer %u failed with status %u\n", j, peer_حالت);
                    htp_fence_write(my_fence, fence_seq_exit, peer_حالت);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_ثبت);
                    أرجع peer_حالت;
                }
                اكسر;
            }
            إذا (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_ثبت, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_حالت_دننی_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_ثبت);
                أرجع HTP_حالت_دننی_ERR;
            }
            hex_pause();
        }
    }
    asm متطاير ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_ثبت);

    // 2. Multi-threaded Reduction across assigned rank chunk
    إذا (nelem > 0) {
        ثابت uint32_t n_تارونه            = (uint32_t) kparams->n_تارونه;
        ثابت uint32_t بلاک_elems          = (uint32_t) kparams->بلاک_elems;
        ثابت uint32_t elems_فی_thread     = (uint32_t) kparams->elems_فی_thread;
        ثابت uint32_t vtcm_اندازه_فی_thread = (uint32_t) kparams->vtcm_اندازه_فی_thread;

        بنية htp_allreduce_context actx;
        actx.octx                 = octx;
        actx.n_ranks              = n_ranks;
        actx.n_dsts               = (uint32_t) kparams->n_dsts ? (uint32_t) kparams->n_dsts : n_ranks;
        actx.nelem                = nelem;
        actx.ne0                  = (uint32_t) kparams->ne0;
        actx.ne1                  = (uint32_t) kparams->ne1;
        actx.لیکه_اندازه_aligned     = (uint32_t) kparams->لیکه_اندازه_aligned;
        actx.rank_elem_پیل      = (uint32_t) kparams->rank_elem_پیل;
        actx.rank_nelem           = (uint32_t) kparams->rank_nelem;
        actx.elems_فی_thread     = elems_فی_thread;
        actx.بلاک_elems          = بلاک_elems;
        actx.vtcm_اندازه_فی_thread = vtcm_اندازه_فی_thread;
        actx.is_لیکه_bcast         = (kparams->is_لیکه_bcast != 0);

        work_کتار_func_t reduce_fun = عدم;
        بدل (kparams->kernel_type) {
            حالة HTP_ALLREDUCE_KERNEL_DMA_1D:
                إذا (has_جمع_کړه) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_جمع_کړه_f16 : allreduce_thread_dma_1d_جمع_کړه_f32;
                } وإلا {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                اكسر;
            حالة HTP_ALLREDUCE_KERNEL_DMA_2D:
                إذا (has_جمع_کړه) {
                    إذا (kparams->is_لیکه_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_جمع_کړه_bcast_f16 : allreduce_thread_dma_2d_جمع_کړه_bcast_f32;
                    } وإلا {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_جمع_کړه_f16 : allreduce_thread_dma_2d_جمع_کړه_f32;
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

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_بنسټ;
        لكل (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_بنسټ[s] = vtcm_ptr;
            vtcm_ptr += n_تارونه * vtcm_اندازه_فی_thread;
        }
        actx.dst_spad_بنسټ = vtcm_ptr;
        vtcm_ptr += n_تارونه * vtcm_اندازه_فی_thread;
        إذا (has_جمع_کړه) {
            actx.res_spad_بنسټ = vtcm_ptr;
            vtcm_ptr += (actx.is_لیکه_bcast ? 1 : n_تارونه) * vtcm_اندازه_فی_thread;
        }

        إذا (has_جمع_کړه && actx.is_لیکه_bcast) {
            ثابت dma_addr_t r_ddr = octx->src[2 * n_ranks]->معلومات;
            ثابت uint32_t لیکه_بایټونه = actx.ne0 * (dst->type == HTP_TYPE_F16 ? حجم(__fp16) : حجم(عائم));
            dma_کتار * dma_q = octx->ctx->dma[0];
            dma_کتار_ټېل(dma_q, dma_make_معلومات(actx.res_spad_بنسټ, r_ddr), actx.لیکه_اندازه_aligned, 0, لیکه_بایټونه, 1);
            dma_کتار_راوباسه(dma_q);
        }

        work_کتار_پرلپسې(octx->ctx->work_کتار, reduce_fun, &actx, n_تارونه);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_پیل(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    // drain fan-out DMA writes before exit barrier
    إذا (ډېرپېښ == HTP_ALLREDUCE_SHARDED_FANOUT) {
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
                htp_fence_write(my_fence, fence_seq_exit, HTP_حالت_دننی_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                أرجع HTP_حالت_دننی_ERR;
            }
            hex_pause();
        }
    }
    asm متطاير ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    أرجع octx->حالت;
}
