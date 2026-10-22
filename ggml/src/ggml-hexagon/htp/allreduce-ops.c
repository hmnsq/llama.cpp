#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>
#ಅನ್ತರ್ಭಾವ <stdatomic.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"
#ಅನ್ತರ್ಭಾವ "allreduce-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-fence.h"

ರಚನೆ htp_allreduce_context {
    ರಚನೆ htp_ops_context * octx;
    uint32_t n_ranks;
    uint32_t n_dsts;
    uint32_t nelem;
    uint32_t ne0;
    uint32_t ne1;
    uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    uint32_t rank_elem_ಆರಂಭ;
    uint32_t rank_nelem;
    uint32_t elems_ಪ್ರತಿ_thread;
    uint32_t ಖಂಡ_elems;
    uint32_t vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    bool     is_ಅಡ್ಡಸಾಲು_bcast;
    uint8_t * src_spad_ತಳ[HTP_ALLREDUCE_MAX_RANKS];
    uint8_t * dst_spad_ತಳ;
    uint8_t * res_spad_ತಳ;
};

#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_ALLREDUCE_THREAD_DMA_1D(SUFFIX, TYPE, HVX_ಕೂಡಿಸು_FN, HAS_ಕೂಡಿಸು)                                         \
ಸ್ಥಿರ ಶೂನ್ಯ allreduce_thread_dma_1d_##SUFFIX(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {                   \
    ರಚನೆ htp_allreduce_context * actx = (ರಚನೆ htp_allreduce_context *) ದತ್ತಾಂಶ;                                  \
    ರಚನೆ htp_ops_context * octx = actx->octx;                                                                   \
                                                                                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t n_ranks     = actx->n_ranks;                                                                   \
    ಸ್ಥಿರಾಂಕ uint32_t n_dsts      = actx->n_dsts;                                                                    \
    ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_elems = actx->ಖಂಡ_elems;                                                               \
                                                                                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t dr  = actx->elems_ಪ್ರತಿ_thread;                                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t ir0 = actx->rank_elem_ಆರಂಭ + dr * ith;                                                        \
    ಸ್ಥಿರಾಂಕ uint32_t ir1 = MIN(ir0 + dr, actx->rank_elem_ಆರಂಭ + actx->rank_nelem);                                 \
    ಆದರೆ (ir0 >= ir1) ಹಿಂತಿರುಗಿ;                                                                                       \
                                                                                                                  \
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];                                                        \
    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];                                                                      \
                                                                                                                  \
    ಸ್ಥಿರಾಂಕ size_t vtcm_thread_ಸ್ಥಾನಾಂತರ = ith * actx->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread;                                           \
    uint8_t * dst_spad_ತಳ = actx->dst_spad_ತಳ + vtcm_thread_ಸ್ಥಾನಾಂತರ;                                           \
    uint8_t * res_spad_ತಳ = HAS_ಕೂಡಿಸು ? (actx->res_spad_ತಳ + vtcm_thread_ಸ್ಥಾನಾಂತರ) : ಖಾಲಿ;                        \
                                                                                                                  \
    ಸ್ಥಿರಾಂಕ size_t spad_half = actx->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread / 2;                                                      \
    uint32_t ir_prefetch = ir0;                                                                                   \
    ಪೂರ್ಣಾಂಕ spad_idx = 0;                                                                                             \
                                                                                                                  \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 2 && ir_prefetch < ir1; k++) {                                                            \
        uint32_t cur_elems = MIN(ಖಂಡ_elems, ir1 - ir_prefetch);                                                 \
        size_t   cur_ಬೈಟುಗಳು = cur_elems * ಗಾತ್ರ(TYPE);                                                            \
        uint8_t * d_spad = dst_spad_ತಳ + spad_idx * spad_half;                                                  \
        ಗಾಗಿ (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->ದತ್ತಾಂಶ + ir_prefetch * ಗಾತ್ರ(TYPE);                                  \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(d_ddr, d_spad), cur_ಬೈಟುಗಳು, cur_ಬೈಟುಗಳು, cur_ಬೈಟುಗಳು, 0);              \
        }                                                                                                         \
        ಗಾಗಿ (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            uint8_t * s_spad = actx->src_spad_ತಳ[s] + vtcm_thread_ಸ್ಥಾನಾಂತರ + spad_idx * spad_half;                \
            ಸ್ಥಿರಾಂಕ dma_addr_t s_ddr = octx->src[s]->ದತ್ತಾಂಶ + ir_prefetch * ಗಾತ್ರ(TYPE);                             \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_spad, s_ddr), cur_ಬೈಟುಗಳು, cur_ಬೈಟುಗಳು, cur_ಬೈಟುಗಳು, 1);              \
        }                                                                                                         \
        ಆದರೆ (HAS_ಕೂಡಿಸು) {                                                                                            \
            uint8_t * r_spad = res_spad_ತಳ + spad_idx * spad_half;                                              \
            ಸ್ಥಿರಾಂಕ dma_addr_t r_ddr = octx->src[2 * n_ranks]->ದತ್ತಾಂಶ + ir_prefetch * ಗಾತ್ರ(TYPE);                   \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(r_spad, r_ddr), cur_ಬೈಟುಗಳು, cur_ಬೈಟುಗಳು, cur_ಬೈಟುಗಳು, 1);              \
        }                                                                                                         \
        ir_prefetch += cur_elems;                                                                                 \
        spad_idx ^= 1;                                                                                            \
    }                                                                                                             \
                                                                                                                  \
    ಪೂರ್ಣಾಂಕ comp_spad_idx = 0;                                                                                        \
    ಗಾಗಿ (uint32_t ir = ir0; ir < ir1; ) {                                                                         \
        uint32_t cur_elems = MIN(ಖಂಡ_elems, ir1 - ir);                                                          \
        size_t   cur_ಬೈಟುಗಳು = cur_elems * ಗಾತ್ರ(TYPE);                                                            \
        ಗಾಗಿ (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_ಸರತಿ_ತೆಗೆ(dma_q);                                                                                 \
        }                                                                                                         \
        ಗಾಗಿ (uint32_t s = 0; s < n_ranks; s++) {                                                                  \
            dma_ಸರತಿ_ತೆಗೆ(dma_q);                                                                                 \
        }                                                                                                         \
        ಆದರೆ (HAS_ಕೂಡಿಸು) {                                                                                            \
            dma_ಸರತಿ_ತೆಗೆ(dma_q);                                                                                 \
        }                                                                                                         \
        uint8_t * d_spad  = dst_spad_ತಳ + comp_spad_idx * spad_half;                                            \
        uint8_t * s0_spad = actx->src_spad_ತಳ[0] + vtcm_thread_ಸ್ಥಾನಾಂತರ + comp_spad_idx * spad_half;              \
        uint8_t * s1_spad = actx->src_spad_ತಳ[1] + vtcm_thread_ಸ್ಥಾನಾಂತರ + comp_spad_idx * spad_half;              \
        uint8_t * r_spad  = HAS_ಕೂಡಿಸು ? (res_spad_ತಳ + comp_spad_idx * spad_half) : ಖಾಲಿ;                         \
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                         \
        HVX_ಕೂಡಿಸು_FN(d_spad, s0_spad, s1_spad, cur_elems);                                                          \
        ಗಾಗಿ (uint32_t s = 2; s < n_ranks; s++) {                                                                  \
            uint8_t * ss_spad = actx->src_spad_ತಳ[s] + vtcm_thread_ಸ್ಥಾನಾಂತರ + comp_spad_idx * spad_half;          \
            HVX_ಕೂಡಿಸು_FN(d_spad, d_spad, ss_spad, cur_elems);                                                       \
        }                                                                                                         \
        ಆದರೆ (HAS_ಕೂಡಿಸು) {                                                                                            \
            HVX_ಕೂಡಿಸು_FN(d_spad, d_spad, r_spad, cur_elems);                                                        \
        }                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ir);                                          \
        ಗಾಗಿ (uint32_t d = 0; d < n_dsts; d++) {                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->ದತ್ತಾಂಶ + ir * ಗಾತ್ರ(TYPE);                                           \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(d_ddr, d_spad), cur_ಬೈಟುಗಳು, cur_ಬೈಟುಗಳು, cur_ಬೈಟುಗಳು, 1);              \
        }                                                                                                         \
        ಆದರೆ (ir_prefetch < ir1) {                                                                                  \
            uint32_t ಮುಂದಿನ_elems = MIN(ಖಂಡ_elems, ir1 - ir_prefetch);                                            \
            size_t   ಮುಂದಿನ_ಬೈಟುಗಳು = ಮುಂದಿನ_elems * ಗಾತ್ರ(TYPE);                                                      \
            ಗಾಗಿ (uint32_t s = 0; s < n_ranks; s++) {                                                              \
                uint8_t * s_spad = actx->src_spad_ತಳ[s] + vtcm_thread_ಸ್ಥಾನಾಂತರ + comp_spad_idx * spad_half;       \
                ಸ್ಥಿರಾಂಕ dma_addr_t s_ಮುಂದಿನ = octx->src[s]->ದತ್ತಾಂಶ + ir_prefetch * ಗಾತ್ರ(TYPE);                        \
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_spad, s_ಮುಂದಿನ), ಮುಂದಿನ_ಬೈಟುಗಳು, ಮುಂದಿನ_ಬೈಟುಗಳು, ಮುಂದಿನ_ಬೈಟುಗಳು, 1);      \
            }                                                                                                     \
            ಆದರೆ (HAS_ಕೂಡಿಸು) {                                                                                        \
                uint8_t * r_spad_ಮುಂದಿನ = res_spad_ತಳ + comp_spad_idx * spad_half;                                \
                ಸ್ಥಿರಾಂಕ dma_addr_t r_ಮುಂದಿನ = octx->src[2 * n_ranks]->ದತ್ತಾಂಶ + ir_prefetch * ಗಾತ್ರ(TYPE);              \
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(r_spad_ಮುಂದಿನ, r_ಮುಂದಿನ), ಮುಂದಿನ_ಬೈಟುಗಳು, ಮುಂದಿನ_ಬೈಟುಗಳು, ಮುಂದಿನ_ಬೈಟುಗಳು, 1); \
            }                                                                                                     \
            ir_prefetch += ಮುಂದಿನ_elems;                                                                            \
        }                                                                                                         \
        comp_spad_idx ^= 1;                                                                                       \
        ir += cur_elems;                                                                                          \
    }                                                                                                             \
    dma_ಸರತಿ_flush(dma_q);                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_1D(f16,     __fp16, hvx_ಕೂಡಿಸು_f16_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(f32,     ದಶಮಾಂಶ,  hvx_ಕೂಡಿಸು_f32_aaa, 0)
DEFINE_ALLREDUCE_THREAD_DMA_1D(ಕೂಡಿಸು_f16, __fp16, hvx_ಕೂಡಿಸು_f16_aaa, 1)
DEFINE_ALLREDUCE_THREAD_DMA_1D(ಕೂಡಿಸು_f32, ದಶಮಾಂಶ,  hvx_ಕೂಡಿಸು_f32_aaa, 1)

#ವ್ಯಾಖ್ಯಾನಿಸು DEFINE_ALLREDUCE_THREAD_DMA_2D(SUFFIX, TYPE, HVX_ಕೂಡಿಸು_FN, HAS_ಕೂಡಿಸು, IS_ಅಡ್ಡಸಾಲು_BCAST)                                                           \
ಸ್ಥಿರ ಶೂನ್ಯ allreduce_thread_dma_2d_##SUFFIX(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {                                                   \
    ರಚನೆ htp_allreduce_context * actx = (ರಚನೆ htp_allreduce_context *) ದತ್ತಾಂಶ;                                                                  \
    ರಚನೆ htp_ops_context * octx = actx->octx;                                                                                                   \
                                                                                                                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t n_ranks          = actx->n_ranks;                                                                                              \
    ಸ್ಥಿರಾಂಕ uint32_t n_dsts           = actx->n_dsts;                                                                                               \
    ಸ್ಥಿರಾಂಕ uint32_t ne0              = actx->ne0;                                                                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು       = actx->ಖಂಡ_elems;                                                                                          \
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = actx->ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;                                                                                     \
    ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು        = ne0 * ಗಾತ್ರ(TYPE);                                                                                         \
                                                                                                                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t dr  = actx->elems_ಪ್ರತಿ_thread;                                                                                                  \
    ಸ್ಥಿರಾಂಕ uint32_t r0  = actx->rank_elem_ಆರಂಭ + dr * ith;                                                                                        \
    ಸ್ಥಿರಾಂಕ uint32_t r1  = MIN(r0 + dr, actx->rank_elem_ಆರಂಭ + actx->rank_nelem);                                                                  \
    ಆದರೆ (r0 >= r1) ಹಿಂತಿರುಗಿ;                                                                                                                         \
                                                                                                                                                  \
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];                                                                                        \
    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];                                                                                                      \
                                                                                                                                                  \
    ಸ್ಥಿರಾಂಕ size_t vtcm_thread_ಸ್ಥಾನಾಂತರ = ith * actx->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread;                                                                           \
    uint8_t * dst_spad_ತಳ = actx->dst_spad_ತಳ + vtcm_thread_ಸ್ಥಾನಾಂತರ;                                                                           \
    uint8_t * res_spad_ತಳ = HAS_ಕೂಡಿಸು ? (IS_ಅಡ್ಡಸಾಲು_BCAST ? actx->res_spad_ತಳ : (actx->res_spad_ತಳ + vtcm_thread_ಸ್ಥಾನಾಂತರ)) : ಖಾಲಿ;                 \
                                                                                                                                                  \
    ಸ್ಥಿರಾಂಕ size_t spad_half = actx->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread / 2;                                                                                      \
    uint32_t r_prefetch = r0;                                                                                                                     \
    ಪೂರ್ಣಾಂಕ spad_idx = 0;                                                                                                                             \
                                                                                                                                                  \
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 2 && r_prefetch < r1; k++) {                                                                                              \
        uint32_t cur_ಅಡ್ಡಸಾಲುಗಳು = MIN(ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು, r1 - r_prefetch);                                                                                     \
        uint8_t * d_spad = dst_spad_ತಳ + spad_idx * spad_half;                                                                                  \
        ಗಾಗಿ (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->ದತ್ತಾಂಶ + r_prefetch * octx->dsts[d]->nb[1];                                                           \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(d_ddr, d_spad), octx->dsts[d]->nb[1], ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, 0);                            \
        }                                                                                                                                         \
        ಗಾಗಿ (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            uint8_t * s_spad = actx->src_spad_ತಳ[s] + vtcm_thread_ಸ್ಥಾನಾಂತರ + spad_idx * spad_half;                                                \
            ಸ್ಥಿರಾಂಕ dma_addr_t s_ddr = octx->src[s]->ದತ್ತಾಂಶ + r_prefetch * octx->src[s]->nb[1];                                                       \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_spad, s_ddr), ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, octx->src[s]->nb[1], ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, cur_ಅಡ್ಡಸಾಲುಗಳು);                      \
        }                                                                                                                                         \
        ಆದರೆ (HAS_ಕೂಡಿಸು && !IS_ಅಡ್ಡಸಾಲು_BCAST) {                                                                                                           \
            uint8_t * r_spad = res_spad_ತಳ + spad_idx * spad_half;                                                                              \
            ಸ್ಥಿರಾಂಕ dma_addr_t r_ddr = octx->src[2 * n_ranks]->ದತ್ತಾಂಶ + r_prefetch * octx->src[2 * n_ranks]->nb[1];                                   \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(r_spad, r_ddr), ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, octx->src[2 * n_ranks]->nb[1], ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, cur_ಅಡ್ಡಸಾಲುಗಳು);            \
        }                                                                                                                                         \
        r_prefetch += cur_ಅಡ್ಡಸಾಲುಗಳು;                                                                                                                   \
        spad_idx ^= 1;                                                                                                                            \
    }                                                                                                                                             \
                                                                                                                                                  \
    ಪೂರ್ಣಾಂಕ comp_spad_idx = 0;                                                                                                                        \
    ಗಾಗಿ (uint32_t r = r0; r < r1; ) {                                                                                                             \
        uint32_t cur_ಅಡ್ಡಸಾಲುಗಳು = MIN(ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು, r1 - r);                                                                                              \
        ಗಾಗಿ (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_ಸರತಿ_ತೆಗೆ(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        ಗಾಗಿ (uint32_t s = 0; s < n_ranks; s++) {                                                                                                  \
            dma_ಸರತಿ_ತೆಗೆ(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        ಆದರೆ (HAS_ಕೂಡಿಸು && !IS_ಅಡ್ಡಸಾಲು_BCAST) {                                                                                                           \
            dma_ಸರತಿ_ತೆಗೆ(dma_q);                                                                                                                 \
        }                                                                                                                                         \
        uint8_t * d_spad  = dst_spad_ತಳ + comp_spad_idx * spad_half;                                                                            \
        uint8_t * s0_spad = actx->src_spad_ತಳ[0] + vtcm_thread_ಸ್ಥಾನಾಂತರ + comp_spad_idx * spad_half;                                              \
        uint8_t * s1_spad = actx->src_spad_ತಳ[1] + vtcm_thread_ಸ್ಥಾನಾಂತರ + comp_spad_idx * spad_half;                                              \
        uint8_t * r_spad  = (HAS_ಕೂಡಿಸು && !IS_ಅಡ್ಡಸಾಲು_BCAST) ? (res_spad_ತಳ + comp_spad_idx * spad_half) : ಖಾಲಿ;                                      \
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                          \
        ಗಾಗಿ (uint32_t ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < cur_ಅಡ್ಡಸಾಲುಗಳು; ಅಡ್ಡಸಾಲು++) {                                                                                           \
            uint8_t * d_ಅಡ್ಡಸಾಲು = d_spad + ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;                                                                                    \
            ಸ್ಥಿರಾಂಕ uint8_t * s0_ಅಡ್ಡಸಾಲು = s0_spad + ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;                                                                            \
            ಸ್ಥಿರಾಂಕ uint8_t * s1_ಅಡ್ಡಸಾಲು = s1_spad + ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;                                                                            \
            HVX_ಕೂಡಿಸು_FN(d_ಅಡ್ಡಸಾಲು, s0_ಅಡ್ಡಸಾಲು, s1_ಅಡ್ಡಸಾಲು, ne0);                                                                                               \
            ಗಾಗಿ (uint32_t s = 2; s < n_ranks; s++) {                                                                                              \
                ಸ್ಥಿರಾಂಕ uint8_t * ss_ಅಡ್ಡಸಾಲು = actx->src_spad_ತಳ[s] + vtcm_thread_ಸ್ಥಾನಾಂತರ + comp_spad_idx * spad_half + ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;        \
                HVX_ಕೂಡಿಸು_FN(d_ಅಡ್ಡಸಾಲು, d_ಅಡ್ಡಸಾಲು, ss_ಅಡ್ಡಸಾಲು, ne0);                                                                                            \
            }                                                                                                                                     \
            ಆದರೆ (HAS_ಕೂಡಿಸು) {                                                                                                                        \
                ಸ್ಥಿರಾಂಕ uint8_t * res_ಅಡ್ಡಸಾಲು = IS_ಅಡ್ಡಸಾಲು_BCAST ? res_spad_ತಳ : (r_spad + ಅಡ್ಡಸಾಲು * ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);                                       \
                HVX_ಕೂಡಿಸು_FN(d_ಅಡ್ಡಸಾಲು, d_ಅಡ್ಡಸಾಲು, res_ಅಡ್ಡಸಾಲು, ne0);                                                                                           \
            }                                                                                                                                     \
        }                                                                                                                                         \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);                                                                           \
        ಗಾಗಿ (uint32_t d = 0; d < n_dsts; d++) {                                                                                                   \
            dma_addr_t d_ddr = octx->dsts[d]->ದತ್ತಾಂಶ + r * octx->dsts[d]->nb[1];                                                                    \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(d_ddr, d_spad), octx->dsts[d]->nb[1], ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, cur_ಅಡ್ಡಸಾಲುಗಳು);                     \
        }                                                                                                                                         \
        ಆದರೆ (r_prefetch < r1) {                                                                                                                    \
            uint32_t ಮುಂದಿನ_ಅಡ್ಡಸಾಲುಗಳು = MIN(ಖಂಡ_ಅಡ್ಡಸಾಲುಗಳು, r1 - r_prefetch);                                                                                \
            ಗಾಗಿ (uint32_t s = 0; s < n_ranks; s++) {                                                                                              \
                uint8_t * s_spad = actx->src_spad_ತಳ[s] + vtcm_thread_ಸ್ಥಾನಾಂತರ + comp_spad_idx * spad_half;                                       \
                ಸ್ಥಿರಾಂಕ dma_addr_t s_ಮುಂದಿನ = octx->src[s]->ದತ್ತಾಂಶ + r_prefetch * octx->src[s]->nb[1];                                                  \
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(s_spad, s_ಮುಂದಿನ), ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, octx->src[s]->nb[1], ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, ಮುಂದಿನ_ಅಡ್ಡಸಾಲುಗಳು);                \
            }                                                                                                                                     \
            ಆದರೆ (HAS_ಕೂಡಿಸು && !IS_ಅಡ್ಡಸಾಲು_BCAST) {                                                                                                       \
                uint8_t * r_spad_ಮುಂದಿನ = res_spad_ತಳ + comp_spad_idx * spad_half;                                                                \
                ಸ್ಥಿರಾಂಕ dma_addr_t r_ಮುಂದಿನ = octx->src[2 * n_ranks]->ದತ್ತಾಂಶ + r_prefetch * octx->src[2 * n_ranks]->nb[1];                              \
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(r_spad_ಮುಂದಿನ, r_ಮುಂದಿನ), ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, octx->src[2 * n_ranks]->nb[1], ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, ಮುಂದಿನ_ಅಡ್ಡಸಾಲುಗಳು); \
            }                                                                                                                                     \
            r_prefetch += ಮುಂದಿನ_ಅಡ್ಡಸಾಲುಗಳು;                                                                                                              \
        }                                                                                                                                         \
        comp_spad_idx ^= 1;                                                                                                                       \
        r += cur_ಅಡ್ಡಸಾಲುಗಳು;                                                                                                                            \
    }                                                                                                                                             \
    dma_ಸರತಿ_flush(dma_q);                                                                                                                       \
}

DEFINE_ALLREDUCE_THREAD_DMA_2D(f16,           __fp16, hvx_ಕೂಡಿಸು_f16_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(f32,           ದಶಮಾಂಶ,  hvx_ಕೂಡಿಸು_f32_aaa, 0, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(ಕೂಡಿಸು_f16,       __fp16, hvx_ಕೂಡಿಸು_f16_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(ಕೂಡಿಸು_f32,       ದಶಮಾಂಶ,  hvx_ಕೂಡಿಸು_f32_aaa, 1, 0)
DEFINE_ALLREDUCE_THREAD_DMA_2D(ಕೂಡಿಸು_bcast_f16, __fp16, hvx_ಕೂಡಿಸು_f16_aaa, 1, 1)
DEFINE_ALLREDUCE_THREAD_DMA_2D(ಕೂಡಿಸು_bcast_f32, ದಶಮಾಂಶ,  hvx_ಕೂಡಿಸು_f32_aaa, 1, 1)

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ validate_allreduce(
    ರಚನೆ htp_ops_context * octx,
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_allreduce_kernel_params * kparams,
    uint32_t n_ranks
) {
    ಆದರೆ (kparams->n_ಎಳೆಗಳು == 0 || (uint32_t) kparams->n_ಎಳೆಗಳು > octx->ctx->n_ಎಳೆಗಳು) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }
    octx->n_ಎಳೆಗಳು = (uint32_t) kparams->n_ಎಳೆಗಳು;

    ಆದರೆ (kparams->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread <= 0 || kparams->vtcm_ಗಾತ್ರ <= 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಸ್ಥಿರಾಂಕ bool has_ಕೂಡಿಸು = (octx->op == HTP_OP_ALLREDUCE_ಕೂಡಿಸು);
    ಸ್ಥಿರಾಂಕ size_t n_vtcm_buffers = htp_allreduce_vtcm_ಬಫರ್_ಎಣಿಕೆ(
        n_ranks, octx->n_ಎಳೆಗಳು, has_ಕೂಡಿಸು, kparams->is_ಅಡ್ಡಸಾಲು_bcast != 0);
    ಸ್ಥಿರಾಂಕ size_t vtcm_ಗಾತ್ರ = n_vtcm_buffers * (size_t) kparams->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread;
    ಆದರೆ (vtcm_ಗಾತ್ರ != (size_t) kparams->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }
    ಆದರೆ (vtcm_ಗಾತ್ರ > octx->ctx->vtcm_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    ಆದರೆ (octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_F32) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಪೂರ್ಣಾಂಕ op_allreduce(ರಚನೆ htp_ops_context * octx) {
    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1 && octx->ctx->mdev.idx > 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_allreduce_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_allreduce_kernel_params *) octx->kernel_params;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;

    ಸ್ಥಿರಾಂಕ uint32_t rank    = (uint32_t) kparams->rank;
    ಸ್ಥಿರಾಂಕ uint32_t n_ranks = (uint32_t) kparams->n_ranks;

    ಆದರೆ (n_ranks < 2 || n_ranks > HTP_ALLREDUCE_MAX_RANKS || rank >= n_ranks) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಸ್ಥಿರಾಂಕ uint32_t fence_seq_ನಮೂದು = (uint32_t) octx->op_params[0];
    ಸ್ಥಿರಾಂಕ uint32_t fence_seq_exit  = (uint32_t) octx->op_params[1];

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * my_sync = octx->src[n_ranks + rank];
    atomic_uint * my_fence = (atomic_uint *) (uintptr_t) my_sync->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸ್ಥಿತಿ_2 = validate_allreduce(octx, kparams, n_ranks);
    ಆದರೆ (ಸ್ಥಿತಿ_2 != HTP_ಸ್ಥಿತಿ_OK) {
        ಆದರೆ (ಸ್ಥಿತಿ_2 == HTP_ಸ್ಥಿತಿ_NO_SUPPORT) {
            FARF(ERROR, "ggml-hex: allreduce unsupported type %d : rank %u\n", dst->type, rank);
        }
        htp_fence_write(my_fence, fence_seq_exit, ಸ್ಥಿತಿ_2);
        ಹಿಂತಿರುಗಿ ಸ್ಥಿತಿ_2;
    }

    ಸ್ಥಿರಾಂಕ bool has_ಕೂಡಿಸು = (octx->op == HTP_OP_ALLREDUCE_ಕೂಡಿಸು);
    ಸ್ಥಿರಾಂಕ uint32_t nelem = dst->ne[0] * dst->ne[1] * dst->ne[2] * dst->ne[3];
    ಸ್ಥಿರಾಂಕ int32_t  ಬಹುಲಕ  = kparams->ಬಹುಲಕ;

    // 1. Entry Barrier: Synchronize all ranks before reading
    ರಚನೆ htp_thread_trace * tr0 = &octx->ctx->trace[0];
    htp_trace_event_ಆರಂಭ(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_ನಮೂದು);

    htp_fence_write(my_fence, fence_seq_ನಮೂದು, octx->ಸ್ಥಿತಿ_2);

    ಗಾಗಿ (uint32_t j = 0; j < n_ranks; j++) {
        ಆದರೆ (j == rank) ಮುಂದುವರಿ;
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->ದತ್ತಾಂಶ;
        uint64_t spins = 0;
        ಆಗ (1) {
            uint32_t peer_seq;
            uint32_t peer_ಸ್ಥಿತಿ;
            htp_fence_read(peer_fence, &peer_seq, &peer_ಸ್ಥಿತಿ);
            ಆದರೆ ((int32_t)(peer_seq - fence_seq_ನಮೂದು) >= 0) {
                ಆದರೆ (peer_ಸ್ಥಿತಿ > HTP_ಸ್ಥಿತಿ_OK) {
                    FARF(ERROR, "ggml-hex: allreduce entry peer %u failed with status %u\n", j, peer_ಸ್ಥಿತಿ);
                    htp_fence_write(my_fence, fence_seq_exit, peer_ಸ್ಥಿತಿ);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_ನಮೂದು);
                    ಹಿಂತಿರುಗಿ peer_ಸ್ಥಿತಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಆದರೆ (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce entry fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_ನಮೂದು, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_ಸ್ಥಿತಿ_ಆಂತರಿಕ_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_ನಮೂದು);
                ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_ಆಂತರಿಕ_ERR;
            }
            hex_pause();
        }
    }
    asm ಅಸ್ಥಿರ ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_ನಮೂದು);

    // 2. Multi-threaded Reduction across assigned rank chunk
    ಆದರೆ (nelem > 0) {
        ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು            = (uint32_t) kparams->n_ಎಳೆಗಳು;
        ಸ್ಥಿರಾಂಕ uint32_t ಖಂಡ_elems          = (uint32_t) kparams->ಖಂಡ_elems;
        ಸ್ಥಿರಾಂಕ uint32_t elems_ಪ್ರತಿ_thread     = (uint32_t) kparams->elems_ಪ್ರತಿ_thread;
        ಸ್ಥಿರಾಂಕ uint32_t vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread = (uint32_t) kparams->vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread;

        ರಚನೆ htp_allreduce_context actx;
        actx.octx                 = octx;
        actx.n_ranks              = n_ranks;
        actx.n_dsts               = (uint32_t) kparams->n_dsts ? (uint32_t) kparams->n_dsts : n_ranks;
        actx.nelem                = nelem;
        actx.ne0                  = (uint32_t) kparams->ne0;
        actx.ne1                  = (uint32_t) kparams->ne1;
        actx.ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned     = (uint32_t) kparams->ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
        actx.rank_elem_ಆರಂಭ      = (uint32_t) kparams->rank_elem_ಆರಂಭ;
        actx.rank_nelem           = (uint32_t) kparams->rank_nelem;
        actx.elems_ಪ್ರತಿ_thread     = elems_ಪ್ರತಿ_thread;
        actx.ಖಂಡ_elems          = ಖಂಡ_elems;
        actx.vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread = vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread;
        actx.is_ಅಡ್ಡಸಾಲು_bcast         = (kparams->is_ಅಡ್ಡಸಾಲು_bcast != 0);

        work_ಸರತಿ_func_t reduce_fun = ಖಾಲಿ;
        ಆಯ್ಕೆ (kparams->kernel_type) {
            ಸ್ಥಿತಿ HTP_ALLREDUCE_KERNEL_DMA_1D:
                ಆದರೆ (has_ಕೂಡಿಸು) {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_ಕೂಡಿಸು_f16 : allreduce_thread_dma_1d_ಕೂಡಿಸು_f32;
                } ಇಲ್ಲದಿದ್ದರೆ {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_1d_f16 : allreduce_thread_dma_1d_f32;
                }
                ನಿಲ್ಲಿಸು;
            ಸ್ಥಿತಿ HTP_ALLREDUCE_KERNEL_DMA_2D:
                ಆದರೆ (has_ಕೂಡಿಸು) {
                    ಆದರೆ (kparams->is_ಅಡ್ಡಸಾಲು_bcast) {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_ಕೂಡಿಸು_bcast_f16 : allreduce_thread_dma_2d_ಕೂಡಿಸು_bcast_f32;
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_ಕೂಡಿಸು_f16 : allreduce_thread_dma_2d_ಕೂಡಿಸು_f32;
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    reduce_fun = (dst->type == HTP_TYPE_F16) ? allreduce_thread_dma_2d_f16 : allreduce_thread_dma_2d_f32;
                }
                ನಿಲ್ಲಿಸು;
            ಪೂರ್ವನಿಯೋಜಿತ:
                FARF(ERROR, "ggml-hex: allreduce unsupported kernel %d : rank %u\n", kparams->kernel_type, rank);
                htp_fence_write(my_fence, fence_seq_exit, HTP_ಸ್ಥಿತಿ_NO_SUPPORT);
                ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
        }

        uint8_t * vtcm_ptr = (uint8_t *) octx->ctx->vtcm_ತಳ;
        ಗಾಗಿ (uint32_t s = 0; s < n_ranks; s++) {
            actx.src_spad_ತಳ[s] = vtcm_ptr;
            vtcm_ptr += n_ಎಳೆಗಳು * vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread;
        }
        actx.dst_spad_ತಳ = vtcm_ptr;
        vtcm_ptr += n_ಎಳೆಗಳು * vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread;
        ಆದರೆ (has_ಕೂಡಿಸು) {
            actx.res_spad_ತಳ = vtcm_ptr;
            vtcm_ptr += (actx.is_ಅಡ್ಡಸಾಲು_bcast ? 1 : n_ಎಳೆಗಳು) * vtcm_ಗಾತ್ರ_ಪ್ರತಿ_thread;
        }

        ಆದರೆ (has_ಕೂಡಿಸು && actx.is_ಅಡ್ಡಸಾಲು_bcast) {
            ಸ್ಥಿರಾಂಕ dma_addr_t r_ddr = octx->src[2 * n_ranks]->ದತ್ತಾಂಶ;
            ಸ್ಥಿರಾಂಕ uint32_t ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = actx.ne0 * (dst->type == HTP_TYPE_F16 ? ಗಾತ್ರ(__fp16) : ಗಾತ್ರ(ದಶಮಾಂಶ));
            dma_ಸರತಿ * dma_q = octx->ctx->dma[0];
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(actx.res_spad_ತಳ, r_ddr), actx.ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, 0, ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, 1);
            dma_ಸರತಿ_ತೆಗೆ(dma_q);
        }

        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, reduce_fun, &actx, n_ಎಳೆಗಳು);
    }

    // 4. Exit Barrier: Synchronize all ranks after writing
    htp_trace_event_ಆರಂಭ(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    // drain fan-out DMA writes before exit barrier
    ಆದರೆ (ಬಹುಲಕ == HTP_ALLREDUCE_SHARDED_FANOUT) {
        asm ಅಸ್ಥಿರ ("syncht" : : : "memory");
    }

    htp_fence_write(my_fence, fence_seq_exit, octx->ಸ್ಥಿತಿ_2);

    ಗಾಗಿ (uint32_t j = 0; j < n_ranks; j++) {
        ಆದರೆ (j == rank) ಮುಂದುವರಿ;
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * peer_sync = octx->src[n_ranks + j];
        atomic_uint * peer_fence = (atomic_uint *) (uintptr_t) peer_sync->ದತ್ತಾಂಶ;
        uint64_t spins = 0;
        ಆಗ (1) {
            uint32_t peer_seq;
            uint32_t peer_ಸ್ಥಿತಿ;
            htp_fence_read(peer_fence, &peer_seq, &peer_ಸ್ಥಿತಿ);
            ಆದರೆ ((int32_t)(peer_seq - fence_seq_exit) >= 0) {
                ಆದರೆ (peer_ಸ್ಥಿತಿ > HTP_ಸ್ಥಿತಿ_OK) {
                    FARF(ERROR, "ggml-hex: allreduce exit peer %u failed with status %u\n", j, peer_ಸ್ಥಿತಿ);
                    htp_fence_write(my_fence, fence_seq_exit, peer_ಸ್ಥಿತಿ);
                    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                    ಹಿಂತಿರುಗಿ peer_ಸ್ಥಿತಿ;
                }
                ನಿಲ್ಲಿಸು;
            }
            ಆದರೆ (++spins > HTP_FENCE_TIMEOUT) {
                FARF(ERROR, "ggml-hex: allreduce exit fence-wait TIMEOUT : rank %u waiting on %u fence %p seq 0x%x peer-seq 0x%x\n",
                     rank, j, peer_fence, fence_seq_exit, peer_seq);
                htp_fence_write(my_fence, fence_seq_exit, HTP_ಸ್ಥಿತಿ_ಆಂತರಿಕ_ERR);
                htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);
                ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_ಆಂತರಿಕ_ERR;
            }
            hex_pause();
        }
    }
    asm ಅಸ್ಥಿರ ("syncht" : : : "memory");

    htp_trace_event_stop(tr0, HTP_TRACE_EVT_FENCE, (uint16_t) fence_seq_exit);

    ಹಿಂತಿರುಗಿ octx->ಸ್ಥಿತಿ_2;
}
