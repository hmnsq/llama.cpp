#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdatomic.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "allreduce-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-fence.h"

ਬਣਤਰ htp_allreduce_context {
    ਬਣਤਰ htp_ops_context * octx;
    uint32_t n_ranks;
    uint32_t n_dsts;
    uint32_t nelem;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t ਪੰਕਤੀ_ਮਾਪ_aligned;
    uint32_t rank_elem_ਸ਼ੁਰੂ;
    uint32_t rank_nelem;
    uint32_t elems_ਪ੍ਰਤੀ_thread;
    uint32_t ਖੰਡ_elems;
    uint32_t vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    bool     is_ਪੰਕਤੀ_bcast;
    uint8_t * src_spad_ਅਧਾਰ[HTP_ALLREDUCE_MAX_RANKS];
    uint8_t * dst_spad_ਅਧਾਰ;
    uint8_t * res_spad_ਅਧਾਰ;
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_ALLREDUCE_THREAD_DMA_1D(SUFFIX, TYPE, HVX_ਜੋੜੋ_FN, HAS_ਜੋੜੋ)                                         \
ਸਥਿਰ ਖਾਲੀ allreduce_thread_dma_1d_##SUFFIX(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {                   \
    ਬਣਤਰ htp_allreduce_context * actx = (ਬਣਤਰ htp_allreduce_context *) ਅੰਕੜੇ;                                  \
    ਬਣਤਰ htp_ops_context * octx = actx->octx;                                                                   \
                                                                                                                  \
    ਅਚਲ uint32_t n_ranks     = actx->n_ranks;                                                                   \
    ਅਚਲ uint32_t n_dsts      = actx->n_dsts;                                                                    \
    ਅਚਲ uint32_t ਖੰਡ_elems = actx->ਖੰਡ_elems;                                                               \
                                                                                                                  \
    ਅਚਲ uint32_t dr  = actx->elems_ਪ੍ਰਤੀ_thread;                                                                  \
    ਅਚਲ uint32_t ir0 = actx->rank_elem_ਸ਼ੁਰੂ + dr * ith;                                                        \
    ਅਚਲ uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_ਸ਼ੁਰੂ + actx->rank_nelem);                                 \
    ਜੇ (ir0 >= ir1) ਵਾਪਸ;                                                                                       \
                                                                                                                  \
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];                                                        \
    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];                                                                      \
                                                                                                                  \
    ਅਚਲ size_t vtcm_thread_ਵਿੱਥ = ith * actx->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread;                                           \
    uint8_t * dst_spad_ਅਧਾਰ = actx->dst_spad_ਅਧਾਰ + vtcm_thread_ਵਿੱਥ;                                           \
    uint8_t * res_spad_ਅਧਾਰ = HAS_ਜੋੜੋ ? (actx->res_spad_ਅਧਾਰ + vtcm_thread_ਵਿੱਥ) : ਨਿਰਾ;                        \
                                                                                                                  \
    ਅਚਲ size_t spad_half = actx->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread / 2;                                                      \
    uint32_t ir_prefetch = ir0;                                                                                   \
    ਪੂਰਨ spad_idx = 0;                                                                                             \
                                                                                                                  \
    ਲਈ (ਪੂਰਨ k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                            \
        uint32_t cur_elems = MIN(ਖੰਡ_elems, ir1 - ir_prefetch);                                                 \
        size_t   cur_ਬਾਈਟਾਂ = cur_elems * ਆਕਾਰ(TYPE);                                                            \
        uint8_t * d_spad = dst_spad_ਅਧਾਰ + spad_idx * spad_half;                                                  \
        ਲਈ (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->ਅੰਕੜੇ + ir_prefetch * ਆਕਾਰ(TYPE);                                  \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(d_ddr, d_spad), cur_ਬਾਈਟਾਂ, cur_ਬਾਈਟਾਂ, cur_ਬਾਈਟਾਂ, 0);              \
        }                                                                                                         \
        ਲਈ (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            uint8_t * s_spad = actx->src_spad_ਅਧਾਰ[s] + vtcm_thread_ਵਿੱਥ + spad_idx * spad_half;                \
            ਅਚਲ dma_addr_t s_ddr = octx->src[s]->ਅੰਕੜੇ + ir_prefetch * ਆਕਾਰ(TYPE);                             \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_spad, s_ddr), cur_ਬਾਈਟਾਂ, cur_ਬਾਈਟਾਂ, cur_ਬਾਈਟਾਂ, 1);              \
        }                                                                                                         \
        ਜੇ (HAS_ਜੋੜੋ) {                                                                                            \
            uint8_t * r_spad = res_spad_ਅਧਾਰ + spad_idx * spad_half;                                              \
            ਅਚਲ dma_addr_t r_ddr = octx->src[2 * n_ranks]->ਅੰਕੜੇ + ir_prefetch * ਆਕਾਰ(TYPE);                   \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(r_spad, r_ddr), cur_ਬਾਈਟਾਂ, cur_ਬਾਈਟਾਂ, cur_ਬਾਈਟਾਂ, 1);              \
        }                                                                                                         \
        ir_prefetch += cur_elems;                                                                                 \
        spad_idx ^= 1;                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    ਪੂਰਨ comp_spad_idx = 0;                                                                                        \
    ਲਈ (uint32_t ir = ir0; ir < ir1; ) {                                                                         \
        uint32_t cur_elems = MIN(ਖੰਡ_elems, ir1 - ir);                                                          \
        size_t   cur_ਬਾਈਟਾਂ = cur_elems * ਆਕਾਰ(TYPE);                                                            \
        ਲਈ (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);                                                                                 \
        }                                                                                                         \
        ਲਈ (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);                                                                                 \
        }                                                                                                         \
        ਜੇ (HAS_ਜੋੜੋ) {                                                                                            \
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);                                                                                 \
        }                                                                                                         \
        uint8_t * d_spad  = dst_spad_ਅਧਾਰ + comp_spad_idx * spad_half;                                            \
        uint8_t * s0_spad = actx->src_spad_ਅਧਾਰ[0] + vtcm_thread_ਵਿੱਥ + comp_spad_idx * spad_half;              \
        uint8_t * s1_spad = actx->src_spad_ਅਧਾਰ[1] + vtcm_thread_ਵਿੱਥ + comp_spad_idx * spad_half;              \
        uint8_t * r_spad  = HAS_ਜੋੜੋ ? (res_spad_ਅਧਾਰ + comp_spad_idx * spad_half) : ਨਿਰਾ;                         \
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                         \
        HVX_ਜੋੜੋ_FN(d_spad, s0_spad, s1_spad, cur_elems);                                                          \
        ਲਈ (uint32_t s = 2; s < n_ranks; s++) {                                                                  \
            uint8_t * ss_spad = actx->src_spad_ਅਧਾਰ[s] + vtcm_thread_ਵਿੱਥ + comp_spad_idx * spad_half;          \
            HVX_ਜੋੜੋ_FN(d_spad, d_spad, ss_spad, cur_elems);                                                       \
        }                                                                                                         \
        ਜੇ (HAS_ਜੋੜੋ) {                                                                                            \
            HVX_ਜੋੜੋ_FN(d_spad, d_spad, r_spad, cur_elems);                                                        \
        }                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                          \
        ਲਈ (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->ਅੰਕੜੇ + ir * ਆਕਾਰ(TYPE);                                           \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(d_ddr, d_spad), cur_ਬਾਈਟਾਂ, cur_ਬਾਈਟਾਂ, cur_ਬਾਈਟਾਂ, 1);              \
        }                                                                                                         \
        ਜੇ (ir_prefetch < ir1) {                                                                                  \
            uint32_t ਅਗਲਾ_elems = MIN(ਖੰਡ_elems, ir1 - ir_prefetch);                                            \
            size_t   ਅਗਲਾ_ਬਾਈਟਾਂ = ਅਗਲਾ_elems * ਆਕਾਰ(TYPE);                                                      \
            ਲਈ (uint32_t s = 0; s < n_ranks; s++) {                                                              \
                uint8_t * s_spad = actx->src_spad_ਅਧਾਰ[s] + vtcm_thread_ਵਿੱਥ + comp_spad_idx * spad_half;       \
                ਅਚਲ dma_addr_t s_ਅਗਲਾ = octx->src[s]->ਅੰਕੜੇ + ir_prefetch * ਆਕਾਰ(TYPE);                        \
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_spad, s_ਅਗਲਾ), ਅਗਲਾ_ਬਾਈਟਾਂ, ਅਗਲਾ_ਬਾਈਟਾਂ, ਅਗਲਾ_ਬਾਈਟਾਂ, 1);      \
            }                                                                                                     \
            ਜੇ (HAS_ਜੋੜੋ) {                                                                                        \
                uint8_t * r_spad_ਅਗਲਾ = res_spad_ਅਧਾਰ + comp_spad_idx * spad_half;                                \
                ਅਚਲ dma_addr_t r_ਅਗਲਾ = octx->src[2 * n_ranks]->ਅੰਕੜੇ + ir_prefetch * ਆਕਾਰ(TYPE);              \
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(r_spad_ਅਗਲਾ, r_ਅਗਲਾ), ਅਗਲਾ_ਬਾਈਟਾਂ, ਅਗਲਾ_ਬਾਈਟਾਂ, ਅਗਲਾ_ਬਾਈਟਾਂ, 1); \
            }                                                                                                     \
            ir_prefetch += ਅਗਲਾ_elems;                                                                            \
        }                                                                                                         \
        comp_spad_idx ^= 1;                                                                                       \
        ir += cur_elems;                                                                                          \
    }                                                                                                             \
    dma_ਕਤਾਰ_flush(dma_q);                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_ਜੋੜੋ_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     ਦਸ਼ਮਲਵ,  hvx_ਜੋੜੋ_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(ਜੋੜੋ_f16, __fp16, hvx_ਜੋੜੋ_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(ਜੋੜੋ_f32, ਦਸ਼ਮਲਵ,  hvx_ਜੋੜੋ_f32_aaa, 1)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_ਜੋੜੋ_FN, HAS_ਜੋੜੋ, IS_ਪੰਕਤੀ_BCAST)                                                           \
ਸਥਿਰ ਖਾਲੀ allreduce_thread_dma_2d_##SUFFIX(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {                                                   \
    ਬਣਤਰ htp_allreduce_context * actx = (ਬਣਤਰ htp_allreduce_context *) ਅੰਕੜੇ;                                                                  \
    ਬਣਤਰ htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    ਅਚਲ uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    ਅਚਲ uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    ਅਚਲ uint32_t ne0              = actx->ne0;                                                                                                  \
    ਅਚਲ uint32_t ਖੰਡ_ਪੰਕਤੀਆਂ       = actx->ਖੰਡ_elems;                                                                                          \
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਮਾਪ_aligned = actx->ਪੰਕਤੀ_ਮਾਪ_aligned;                                                                                     \
    ਅਚਲ uint32_t ਪੰਕਤੀ_ਬਾਈਟਾਂ        = ne0 * ਆਕਾਰ(TYPE);                                                                                         \
                                                                                                                                                  \
    ਅਚਲ uint32_t dr  = actx->elems_ਪ੍ਰਤੀ_thread;                                                                                                  \
    ਅਚਲ uint32_t r0  = actx->rank_elem_ਸ਼ੁਰੂ + dr * ith;                                                                                        \
    ਅਚਲ uint32_t r1  = MIN(r0 + dr, actx->rank_elem_ਸ਼ੁਰੂ + actx->rank_nelem);                                                                  \
    ਜੇ (r0 >= r1) ਵਾਪਸ;                                                                                                                         \
                                                                                                                                                  \
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];                                                                                                      \
                                                                                                                                                  \
    ਅਚਲ size_t vtcm_thread_ਵਿੱਥ = ith * actx->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread;                                                                           \
    uint8_t * dst_spad_ਅਧਾਰ = actx->dst_spad_ਅਧਾਰ + vtcm_thread_ਵਿੱਥ;                                                                           \
    uint8_t * res_spad_ਅਧਾਰ = HAS_ਜੋੜੋ ? (IS_ਪੰਕਤੀ_BCAST ? actx->res_spad_ਅਧਾਰ : (actx->res_spad_ਅਧਾਰ + vtcm_thread_ਵਿੱਥ)) : ਨਿਰਾ;                 \
                                                                                                                                                  \
    ਅਚਲ size_t spad_half = actx->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    ਪੂਰਨ spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    ਲਈ (ਪੂਰਨ k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_ਪੰਕਤੀਆਂ = MIN(ਖੰਡ_ਪੰਕਤੀਆਂ, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_ਅਧਾਰ + spad_idx * spad_half;                                                                                  \
        ਲਈ (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->ਅੰਕੜੇ + r_prefetch * octx->dsts[d]->nb[1];                                                           \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(d_ddr, d_spad), octx->dsts[d]->nb[1], ਪੰਕਤੀ_ਮਾਪ_aligned, ਪੰਕਤੀ_ਬਾਈਟਾਂ, 0);                            \
        }                                                                                                                                         \
        ਲਈ (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = actx->src_spad_ਅਧਾਰ[s] + vtcm_thread_ਵਿੱਥ + spad_idx * spad_half;                                                \
            ਅਚਲ dma_addr_t s_ddr = octx->src[s]->ਅੰਕੜੇ + r_prefetch * octx->src[s]->nb[1];                                                       \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_spad, s_ddr), ਪੰਕਤੀ_ਮਾਪ_aligned, octx->src[s]->nb[1], ਪੰਕਤੀ_ਬਾਈਟਾਂ, cur_ਪੰਕਤੀਆਂ);                      \
        }                                                                                                                                         \
        ਜੇ (HAS_ਜੋੜੋ && !IS_ਪੰਕਤੀ_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_ਅਧਾਰ + spad_idx * spad_half;                                                                              \
            ਅਚਲ dma_addr_t r_ddr = octx->src[2 * n_ranks]->ਅੰਕੜੇ + r_prefetch * octx->src[2 * n_ranks]->nb[1];                                   \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(r_spad, r_ddr), ਪੰਕਤੀ_ਮਾਪ_aligned, octx->src[2 * n_ranks]->nb[1], ਪੰਕਤੀ_ਬਾਈਟਾਂ, cur_ਪੰਕਤੀਆਂ);            \
        }                                                                                                                                         \
        r_prefetch += cur_ਪੰਕਤੀਆਂ;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    ਪੂਰਨ comp_spad_idx = 0;                                                                                                                        \
    ਲਈ (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_ਪੰਕਤੀਆਂ = MIN(ਖੰਡ_ਪੰਕਤੀਆਂ, r1 - r);                                                                                              \
        ਲਈ (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        ਲਈ (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        ਜੇ (HAS_ਜੋੜੋ && !IS_ਪੰਕਤੀ_BCAST) {                                                                                                           \
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        uint8_t * d_spad  = dst_spad_ਅਧਾਰ + comp_spad_idx * spad_half;                                                                            \
        uint8_t * s0_spad = actx->src_spad_ਅਧਾਰ[0] + vtcm_thread_ਵਿੱਥ + comp_spad_idx * spad_half;                                              \
        uint8_t * s1_spad = actx->src_spad_ਅਧਾਰ[1] + vtcm_thread_ਵਿੱਥ + comp_spad_idx * spad_half;                                              \
        uint8_t * r_spad  = (HAS_ਜੋੜੋ && !IS_ਪੰਕਤੀ_BCAST) ? (res_spad_ਅਧਾਰ + comp_spad_idx * spad_half) : ਨਿਰਾ;                                      \
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        ਲਈ (uint32_t ਪੰਕਤੀ = 0; ਪੰਕਤੀ < cur_ਪੰਕਤੀਆਂ; ਪੰਕਤੀ++) {                                                                                           \
            uint8_t * d_ਪੰਕਤੀ = d_spad + ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ_aligned;                                                                                    \
            ਅਚਲ uint8_t * s0_ਪੰਕਤੀ = s0_spad + ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ_aligned;                                                                            \
            ਅਚਲ uint8_t * s1_ਪੰਕਤੀ = s1_spad + ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ_aligned;                                                                            \
            HVX_ਜੋੜੋ_FN(d_ਪੰਕਤੀ, s0_ਪੰਕਤੀ, s1_ਪੰਕਤੀ, ne0);                                                                                               \
            ਲਈ (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                ਅਚਲ uint8_t * ss_ਪੰਕਤੀ = actx->src_spad_ਅਧਾਰ[s] + vtcm_thread_ਵਿੱਥ + comp_spad_idx * spad_half + ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ_aligned;        \
                HVX_ਜੋੜੋ_FN(d_ਪੰਕਤੀ, d_ਪੰਕਤੀ, ss_ਪੰਕਤੀ, ne0);                                                                                            \
            }                                                                                                                                     \
            ਜੇ (HAS_ਜੋੜੋ) {                                                                                                                        \
                ਅਚਲ uint8_t * res_ਪੰਕਤੀ = IS_ਪੰਕਤੀ_BCAST ? res_spad_ਅਧਾਰ : (r_spad + ਪੰਕਤੀ * ਪੰਕਤੀ_ਮਾਪ_aligned);                                       \
                HVX_ਜੋੜੋ_FN(d_ਪੰਕਤੀ, d_ਪੰਕਤੀ, res_ਪੰਕਤੀ, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        ਲਈ (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->ਅੰਕੜੇ + r * octx->dsts[d]->nb[1];                                                                    \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(d_ddr, d_spad), octx->dsts[d]->nb[1], ਪੰਕਤੀ_ਮਾਪ_aligned, ਪੰਕਤੀ_ਬਾਈਟਾਂ, cur_ਪੰਕਤੀਆਂ);                     \
        }                                                                                                                                         \
        ਜੇ (r_prefetch < r1) {                                                                                                                    \
            uint32_t ਅਗਲਾ_ਪੰਕਤੀਆਂ = MIN(ਖੰਡ_ਪੰਕਤੀਆਂ, r1 - r_prefetch);                                                                                \
            ਲਈ (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                uint8_t * s_spad = actx->src_spad_ਅਧਾਰ[s] + vtcm_thread_ਵਿੱਥ + comp_spad_idx * spad_half;                                       \
                ਅਚਲ dma_addr_t s_ਅਗਲਾ = octx->src[s]->ਅੰਕੜੇ + r_prefetch * octx->src[s]->nb[1];                                                  \
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(s_spad, s_ਅਗਲਾ), ਪੰਕਤੀ_ਮਾਪ_aligned, octx->src[s]->nb[1], ਪੰਕਤੀ_ਬਾਈਟਾਂ, ਅਗਲਾ_ਪੰਕਤੀਆਂ);                \
            }                                                                                                                                     \
            ਜੇ (HAS_ਜੋੜੋ && !IS_ਪੰਕਤੀ_BCAST) {                                                                                                       \
                uint8_t * r_spad_ਅਗਲਾ = res_spad_ਅਧਾਰ + comp_spad_idx * spad_half;                                                                \
                ਅਚਲ dma_addr_t r_ਅਗਲਾ = octx->src[2 * n_ranks]->ਅੰਕੜੇ + r_prefetch * octx->src[2 * n_ranks]->nb[1];                              \
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(r_spad_ਅਗਲਾ, r_ਅਗਲਾ), ਪੰਕਤੀ_ਮਾਪ_aligned, octx->src[2 * n_ranks]->nb[1], ਪੰਕਤੀ_ਬਾਈਟਾਂ, ਅਗਲਾ_ਪੰਕਤੀਆਂ); \
            }                                                                                                                                     \
            r_prefetch += ਅਗਲਾ_ਪੰਕਤੀਆਂ;                                                                                                              \
        }                                                                                                                                         \
        comp_spad_idx ^= 1;                                                                                                                       \
        r += cur_ਪੰਕਤੀਆਂ;                                                                                                                            \
    }                                                                                                                                             \
    dma_ਕਤਾਰ_flush(dma_q);                                                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_ਜੋੜੋ_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           ਦਸ਼ਮਲਵ,  hvx_ਜੋੜੋ_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(ਜੋੜੋ_f16,       __fp16, hvx_ਜੋੜੋ_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(ਜੋੜੋ_f32,       ਦਸ਼ਮਲਵ,  hvx_ਜੋੜੋ_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(ਜੋੜੋ_bcast_f16, __fp16, hvx_ਜੋੜੋ_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(ਜੋੜੋ_bcast_f32, ਦਸ਼ਮਲਵ,  hvx_ਜੋੜੋ_f32_aaa, 1, 1)

ਸਥਿਰ ਪੂਰਨ validate_allreduce(
    ਬਣਤਰ htp_ops_context * octx,
    ਅਚਲ ਬਣਤਰ htp_allreduce_kernel_params * kparams,
    uint32_t n_ranks
) {
    ਜੇ (kparams->n_ਧਾਗੇ == 0 || (uint32_t) kparams->n_ਧਾਗੇ > octx->ctx->n_ਧਾਗੇ) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }
    octx->n_ਧਾਗੇ = (uint32_t) kparams->n_ਧਾਗੇ;

    ਜੇ (kparams->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread <= 0 || kparams->vtcm_ਮਾਪ <= 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਅਚਲ bool has_ਜੋੜੋ = (octx->op == HTP_OP_ALLREDUCE_ਜੋੜੋ);
    ਅਚਲ size_t n_vtcm_buffers = htp_allreduce_vtcm_ਬਫਰ_ਗਿਣਤੀ(
        n_ranks, octx->n_ਧਾਗੇ, has_ਜੋੜੋ, kparams->is_ਪੰਕਤੀ_bcast != 0);
    ਅਚਲ size_t vtcm_ਮਾਪ = n_vtcm_buffers * (size_t) kparams->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread;
    ਜੇ (vtcm_ਮਾਪ != (size_t) kparams->vtcm_ਮਾਪ) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }
    ਜੇ (vtcm_ਮਾਪ > octx->ctx->vtcm_ਮਾਪ) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    ਜੇ (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਪੂਰਨ op_allreduce(ਬਣਤਰ htp_ops_context * octx) {
    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1 && octx->ctx->mdev.idx > 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ ਬਣਤਰ htp_allreduce_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_allreduce_kernel_params *) octx->kernel_params;
    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;

    ਅਚਲ uint32_t rank    = (uint32_t) kparams->rank;
    ਅਚਲ uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    ਜੇ (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਅਚਲ uint32_t fence_seq_ਇੰਦਰਾਜ = (uint32_t) octx->op_params[0];
    ਅਚਲ uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    ਅਚਲ ਬਣਤਰ htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->ਅੰਕੜੇ;

    ਅਚਲ ਪੂਰਨ ਸਥਿਤੀ = validate_allreduce(octx, kparams, n_ranks);
    ਜੇ (ਸਥਿਤੀ != HTP_ਸਥਿਤੀ_OK) {
        ਜੇ (ਸਥਿਤੀ == HTP_ਸਥਿਤੀ_NO_SUPPORT) {
            FARF(ERROR, "ggml-hex: allreduce unsupported type %d : rank %u\n", dst->type, rank);
        }
        htp_fence_write(my_fence, fence_seq_exit, ਸਥਿਤੀ);
        ਵਾਪਸ ਸਥਿਤੀ;
    }

    ਅਚਲ bool has_ਜੋੜੋ = (octx->op == HTP_OP_ALLREDUCE_ਜੋੜੋ);
    ਅਚਲ uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ਅਚਲ int32_t  ਬਹੁਲਕ  = kparams->ਬਹੁਲਕ;

    // 1. Entry Barrier: Synchronize all ranks before reading
    ਬਣਤਰ htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_ਸ਼ੁਰੂ(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_ਇੰਦਰਾਜ);

    htp_fence_write(my_fence, fence_seq_ਇੰਦਰਾਜ, octx->ਸਥਿਤੀ);

    ਲਈ (uint32_t j = 0; j < n_ranks; j++) {
        ਜੇ (j == rank) ਜਾਰੀ;
        ਅਚਲ ਬਣਤਰ htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->ਅੰਕੜੇ;
        uint64_t spins = 0;
        ਜਦੋਂ (1) {
            uint32_t peer_seq;
            uint32_t peer_ਸਥਿਤੀ;
            htp_fence_read(peer_fence, &peer_seq, &peer_ਸਥਿਤੀ);
            ਜੇ ((int32_t)(peer_seq - fence_seq_ਇੰਦਰਾਜ) >= 0) {
                ਜੇ (peer_ਸਥਿਤੀ > HTP_ਸਥਿਤੀ_OK) {
                    FARF(ERROR, "ggml-hex: allreduce entry peer %u failed with status %u\n", j, peer_ਸਥਿਤੀ);
                    htp_fence_write(my_fence, fence_seq_exit, peer_ਸਥਿਤੀ);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_ਇੰਦਰਾਜ);
                    ਵਾਪਸ peer_ਸਥਿਤੀ;
                }
                ਰੋਕੋ;
            }
            ਜੇ (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_ਇੰਦਰਾਜ, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_ਸਥਿਤੀ_ਅੰਦਰੂਨੀ_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_ਇੰਦਰਾਜ);
                ਵਾਪਸ HTP_ਸਥਿਤੀ_ਅੰਦਰੂਨੀ_ERR;
            }
            hex_pause();
        }
    }
    asm ਅਸਥਿਰ ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_ਇੰਦਰਾਜ);

    // 2. Multi-threaded Reduction across assigned rank chunk
    ਜੇ (nelem > 0) {
        ਅਚਲ uint32_t n_ਧਾਗੇ            = (uint32_t) kparams->n_ਧਾਗੇ;
        ਅਚਲ uint32_t ਖੰਡ_elems          = (uint32_t) kparams->ਖੰਡ_elems;
        ਅਚਲ uint32_t elems_ਪ੍ਰਤੀ_thread     = (uint32_t) kparams->elems_ਪ੍ਰਤੀ_thread;
        ਅਚਲ uint32_t vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread = (uint32_t) kparams->vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread;

        ਬਣਤਰ htp_allreduce_context actx;
        actx.octx                 = octx;
        actx.n_ranks              = n_ranks;
        actx.n_dsts               = (uint32_t) kparams->n_dsts ? (uint32_t) kparams->n_dsts : n_ranks;
        actx.nelem                = nelem;
        actx.ne0                  = (uint32_t) kparams->ne0;
        actx.ne1                  = (uint32_t) kparams->ne1;
        actx.ਪੰਕਤੀ_ਮਾਪ_aligned     = (uint32_t) kparams->ਪੰਕਤੀ_ਮਾਪ_aligned;
        actx.rank_elem_ਸ਼ੁਰੂ      = (uint32_t) kparams->rank_elem_ਸ਼ੁਰੂ;
        actx.rank_nelem           = (uint32_t) kparams->rank_nelem;
        actx.elems_ਪ੍ਰਤੀ_thread     = elems_ਪ੍ਰਤੀ_thread;
        actx.ਖੰਡ_elems          = ਖੰਡ_elems;
        actx.vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread = vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread;
        actx.is_ਪੰਕਤੀ_bcast         = (kparams->is_ਪੰਕਤੀ_bcast != 0);

        work_ਕਤਾਰ_func_t reduce_fun = ਨਿਰਾ;
        ਚੋਣ (kparams->kernel_type) {
            ਹਾਲਤ HTP_ALLREDUCE_KERNEL_DMA_1D:
                ਜੇ (has_ਜੋੜੋ) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_ਜੋੜੋ_f16 : allreduce_thread_dma_1d_ਜੋੜੋ_f32;
                } ਨਹੀਂਤਾਂ {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                ਰੋਕੋ;
            ਹਾਲਤ HTP_ALLREDUCE_KERNEL_DMA_2D:
                ਜੇ (has_ਜੋੜੋ) {
                    ਜੇ (kparams->is_ਪੰਕਤੀ_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_ਜੋੜੋ_bcast_f16 : allreduce_thread_dma_2d_ਜੋੜੋ_bcast_f32;
                    } ਨਹੀਂਤਾਂ {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_ਜੋੜੋ_f16 : allreduce_thread_dma_2d_ਜੋੜੋ_f32;
                    }
                } ਨਹੀਂਤਾਂ {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_f16 : allreduce_thread_dma_2d_f32;
                }
                ਰੋਕੋ;
            ਮੂਲ:
                FARF(ERROR, "ggml-hex: allreduce unsupported kernel %d : rank %u\n", kparams->kernel_type, rank);
                htp_fence_write(my_fence, fence_seq_exit, HTP_ਸਥਿਤੀ_NO_SUPPORT);
                ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
        }

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;
        ਲਈ (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_ਅਧਾਰ[s] = vtcm_ptr;
            vtcm_ptr += n_ਧਾਗੇ * vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread;
        }
        actx.dst_spad_ਅਧਾਰ = vtcm_ptr;
        vtcm_ptr += n_ਧਾਗੇ * vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread;
        ਜੇ (has_ਜੋੜੋ) {
            actx.res_spad_ਅਧਾਰ = vtcm_ptr;
            vtcm_ptr += (actx.is_ਪੰਕਤੀ_bcast ? 1 : n_ਧਾਗੇ) * vtcm_ਮਾਪ_ਪ੍ਰਤੀ_thread;
        }

        ਜੇ (has_ਜੋੜੋ && actx.is_ਪੰਕਤੀ_bcast) {
            ਅਚਲ dma_addr_t r_ddr = octx->src[2 * n_ranks]->ਅੰਕੜੇ;
            ਅਚਲ uint32_t ਪੰਕਤੀ_ਬਾਈਟਾਂ = actx.ne0 * (dst->type == HTP_TYPE_F16 ? ਆਕਾਰ(__fp16) : ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            dma_ਕਤਾਰ * dma_q = octx->ctx->dma[0];
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(actx.res_spad_ਅਧਾਰ, r_ddr), actx.ਪੰਕਤੀ_ਮਾਪ_aligned, 0, ਪੰਕਤੀ_ਬਾਈਟਾਂ, 1);
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        }

        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, reduce_fun, &actx, n_ਧਾਗੇ);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_ਸ਼ੁਰੂ(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    // drain fan-out DMA writes before exit barrier
    ਜੇ (ਬਹੁਲਕ == HTP_ALLREDUCE_SHARDED_FANOUT) {
        asm ਅਸਥਿਰ ("syncht" : : : "memory");
    }

    htp_fence_write(my_fence, fence_seq_exit, octx->ਸਥਿਤੀ);

    ਲਈ (uint32_t j = 0; j < n_ranks; j++) {
        ਜੇ (j == rank) ਜਾਰੀ;
        ਅਚਲ ਬਣਤਰ htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->ਅੰਕੜੇ;
        uint64_t spins = 0;
        ਜਦੋਂ (1) {
            uint32_t peer_seq;
            uint32_t peer_ਸਥਿਤੀ;
            htp_fence_read(peer_fence, &peer_seq, &peer_ਸਥਿਤੀ);
            ਜੇ ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                ਜੇ (peer_ਸਥਿਤੀ > HTP_ਸਥਿਤੀ_OK) {
                    FARF(ERROR, "ggml-hex: allreduce exit peer %u failed with status %u\n", j, peer_ਸਥਿਤੀ);
                    htp_fence_write(my_fence, fence_seq_exit, peer_ਸਥਿਤੀ);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                    ਵਾਪਸ peer_ਸਥਿਤੀ;
                }
                ਰੋਕੋ;
            }
            ਜੇ (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce exit fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_exit, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_ਸਥਿਤੀ_ਅੰਦਰੂਨੀ_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                ਵਾਪਸ HTP_ਸਥਿਤੀ_ਅੰਦਰੂਨੀ_ERR;
            }
            hex_pause();
        }
    }
    asm ਅਸਥਿਰ ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    ਵਾਪਸ octx->ਸਥਿਤੀ;
}
