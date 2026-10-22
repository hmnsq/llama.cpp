#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "work-queue.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "hex-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx-copy.h"
#ಅನ್ತರ್ಭಾವ "hvx-quant.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "htp/set-rows-ops.h"

#ವ್ಯಾಖ್ಯಾನಿಸು ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_preamble                      \
    ಸ್ಥಿರಾಂಕ uint32_t ne00 = octx->src[0]->ne[0]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne01 = octx->src[0]->ne[1]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = octx->src[0]->ne[2]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t ne10 = octx->src[1]->ne[0]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne11 = octx->src[1]->ne[1]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne12 = octx->src[1]->ne[2]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t nb01 = octx->src[0]->nb[1]; \
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = octx->src[0]->nb[2]; \
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t nb10 = octx->src[1]->nb[0]; \
    ಸ್ಥಿರಾಂಕ uint32_t nb11 = octx->src[1]->nb[1]; \
    ಸ್ಥಿರಾಂಕ uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = octx->dst->nb[1];     \
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = octx->dst->nb[2];     \
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t ne0 = octx->dst->ne[0];     \
    ಸ್ಥಿರಾಂಕ uint32_t ne1 = octx->dst->ne[1];     \
    ಸ್ಥಿರಾಂಕ uint32_t ne2 = octx->dst->ne[2];     \
    ಸ್ಥಿರಾಂಕ uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    ಸ್ಥಿರಾಂಕ uint32_t nr  = ne01;

ರಚನೆ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_context {
    ರಚನೆ htp_ops_context * octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_kernel_params * kparams;
    ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout vtcm_layout;
    uint8_t * vtcm_ತಳ;
    uint32_t task_ಆರಂಭ;
    uint32_t tasks;
    uint32_t tasks_ಪ್ರತಿ_thread;
};

#ವ್ಯಾಖ್ಯಾನಿಸು ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_ಅಭಿವ್ಯಕ್ತಿ)                                                \
ಸ್ಥಿರ ಶೂನ್ಯ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_thread_dma##TYPE_NAME##_##IDX_TYPE(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ *ದತ್ತಾಂಶ) {       \
    ರಚನೆ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_context * srctx = (ರಚನೆ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_context *)ದತ್ತಾಂಶ;                                           \
    ರಚನೆ htp_ops_context * octx = srctx->octx;                                                                 \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_kernel_params * kparams = srctx->kparams;                                          \
    ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_preamble;                                                                                           \
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    ಸ್ಥಿರಾಂಕ uint32_t dr  = srctx->tasks_ಪ್ರತಿ_thread;                                                                \
    ಸ್ಥಿರಾಂಕ uint32_t ir0 = srctx->task_ಆರಂಭ + dr * ith;                                                           \
    ಆದರೆ (ir0 >= srctx->task_ಆರಂಭ + srctx->tasks) {                                                               \
        ಹಿಂತಿರುಗಿ;                                                                                                  \
    }                                                                                                            \
    ಸ್ಥಿರಾಂಕ uint32_t ir1 = MIN(ir0 + dr, srctx->task_ಆರಂಭ + srctx->tasks);                                        \
    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];                                                                     \
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = srctx->vtcm_ತಳ + vtcm_layout->off_src0 + ith * vtcm_layout->src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;   \
    uint8_t * vtcm_dst  = srctx->vtcm_ತಳ + vtcm_layout->off_dst  + ith * vtcm_layout->dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread;    \
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = ne00 * ಗಾತ್ರ(ದಶಮಾಂಶ);                                                         \
    ಸ್ಥಿರಾಂಕ uint32_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ  = htp_tensor_get_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ(octx->dst->type, ne00);                               \
    ಸ್ಥಿರಾಂಕ uint32_t nrows_ಪ್ರತಿ_thread = ir1 - ir0;                                                                 \
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಹೆಜ್ಜೆಗಳು = ne03 * ne02 * nrows_ಪ್ರತಿ_thread;                                                 \
    uint32_t pi_ಹೆಜ್ಜೆ = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    ಗಾಗಿ (uint32_t ಹೆಜ್ಜೆ = 0, spad_idx = 0; ಹೆಜ್ಜೆ < ಒಟ್ಟು_ಹೆಜ್ಜೆಗಳು && spad_idx < 2; ++ಹೆಜ್ಜೆ, spad_idx++) {              \
        uint32_t i = ir0 + pi_ಹೆಜ್ಜೆ;                                                                              \
        ಸ್ಥಿರಾಂಕ dma_addr_t src0_ದತ್ತಾಂಶ = octx->src[0]->ದತ್ತಾಂಶ + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,                                                                                    \
                       dma_make_ದತ್ತಾಂಶ(octx->dst->ದತ್ತಾಂಶ,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_ಗಾತ್ರ),                     \
                       dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, vtcm_layout->dst_spad_half_ಗಾತ್ರ, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0);                          \
        dma_ಸರತಿ_ತಳ್ಳು(dma_q,                                                                                    \
                       dma_make_ದತ್ತಾಂಶ(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_ಗಾತ್ರ,                    \
                                     src0_ದತ್ತಾಂಶ),                                                                 \
                       vtcm_layout->src0_spad_half_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);                       \
        pi_ಹೆಜ್ಜೆ++;                                                                                               \
        ಆದರೆ (pi_ಹೆಜ್ಜೆ == nrows_ಪ್ರತಿ_thread) {                                                                       \
            pi_ಹೆಜ್ಜೆ = 0;                                                                                         \
            pi02++;                                                                                              \
            ಆದರೆ (pi02 == ne02) {                                                                                  \
                pi02 = 0;                                                                                        \
                pi03++;                                                                                          \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    uint32_t ci_ಹೆಜ್ಜೆ = 0;                                                                                        \
    uint32_t ci02 = 0;                                                                                           \
    uint32_t ci03 = 0;                                                                                           \
    uint32_t ci11_ತಳ = 0;                                                                                      \
    uint32_t ci12_ತಳ = 0;                                                                                      \
    ಗಾಗಿ (uint32_t ಹೆಜ್ಜೆ = 0; ಹೆಜ್ಜೆ < ಒಟ್ಟು_ಹೆಜ್ಜೆಗಳು; ++ಹೆಜ್ಜೆ) {                                                        \
        ಶೂನ್ಯ * dst_spad = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).src;                                                     \
        ಶೂನ್ಯ * src_spad = (ಶೂನ್ಯ *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_ಹೆಜ್ಜೆ;                                                                              \
        ಸ್ಥಿರಾಂಕ uintptr_t src1_addr = octx->src[1]->ದತ್ತಾಂಶ + i*nb10 + ci11_ತಳ*nb11 + ci12_ತಳ*nb12;               \
        ಸ್ಥಿರಾಂಕ IDX_TYPE i1 = *(ಸ್ಥಿರಾಂಕ IDX_TYPE *)src1_addr;                                                        \
        ಸ್ಥಿರಾಂಕ bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        ಸ್ಥಿರಾಂಕ uint32_t ಗುರಿ_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, ಹೆಜ್ಜೆ);                                                 \
        ಆದರೆ (valid_i1) {                                                                                          \
            COMPUTE_ಅಭಿವ್ಯಕ್ತಿ;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ಹೆಜ್ಜೆ);                                                  \
        ಆದರೆ (valid_i1) {                                                                                          \
            ಸ್ಥಿರಾಂಕ dma_addr_t dst_ದತ್ತಾಂಶ = octx->dst->ದತ್ತಾಂಶ + ಗುರಿ_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q,                                                                                \
                           dma_make_ದತ್ತಾಂಶ(dst_ದತ್ತಾಂಶ, dst_spad),                                                    \
                           dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, vtcm_layout->dst_spad_half_ಗಾತ್ರ, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);                      \
        } ಇಲ್ಲದಿದ್ದರೆ {                                                                                                 \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q,                                                                                \
                           dma_make_ದತ್ತಾಂಶ(octx->dst->ದತ್ತಾಂಶ, dst_spad),                                             \
                           dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, vtcm_layout->dst_spad_half_ಗಾತ್ರ, dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 0);                      \
        }                                                                                                        \
        ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಹೆಜ್ಜೆ = ಹೆಜ್ಜೆ + 2;                                                                     \
        ಆದರೆ (ಮುಂದಿನ_ಹೆಜ್ಜೆ < ಒಟ್ಟು_ಹೆಜ್ಜೆಗಳು) {                                                                           \
            uint32_t ni = ir0 + pi_ಹೆಜ್ಜೆ;                                                                         \
            ಸ್ಥಿರಾಂಕ dma_addr_t psrc0_ದತ್ತಾಂಶ = octx->src[0]->ದತ್ತಾಂಶ + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_ಸರತಿ_ತಳ್ಳು(dma_q,                                                                                \
                           dma_make_ದತ್ತಾಂಶ(src_spad, psrc0_ದತ್ತಾಂಶ),                                                  \
                           vtcm_layout->src0_spad_half_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, 1);                   \
            pi_ಹೆಜ್ಜೆ++;                                                                                           \
            ಆದರೆ (pi_ಹೆಜ್ಜೆ == nrows_ಪ್ರತಿ_thread) {                                                                   \
                pi_ಹೆಜ್ಜೆ = 0;                                                                                     \
                pi02++;                                                                                          \
                ಆದರೆ (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_ಹೆಜ್ಜೆ++;                                                                                               \
        ಆದರೆ (ci_ಹೆಜ್ಜೆ == nrows_ಪ್ರತಿ_thread) {                                                                       \
            ci_ಹೆಜ್ಜೆ = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_ತಳ++;                                                                                         \
            ಆದರೆ (ci11_ತಳ == ne11) {                                                                             \
                ci11_ತಳ = 0;                                                                                   \
            }                                                                                                    \
            ಆದರೆ (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_ತಳ++;                                                                                     \
                ಆದರೆ (ci12_ತಳ == ne12) {                                                                         \
                    ci12_ತಳ = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_ಸರತಿ_flush(dma_q);                                                                                      \
}

ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ಸ್ಥಿರಾಂಕ uint8_t *)src_spad, ne00); })
ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ಸ್ಥಿರಾಂಕ uint8_t *)src_spad, ne00); })

ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ಸ್ಥಿರಾಂಕ uint8_t *)src_spad, ne00); })
ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ಸ್ಥಿರಾಂಕ uint8_t *)src_spad, ne00); })

ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_ಅಡ್ಡಸಾಲು_q8_0_f32(dst_spad, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src_spad, ne00); })
ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_ಅಡ್ಡಸಾಲು_q8_0_f32(dst_spad, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src_spad, ne00); })

ಪೂರ್ಣಾಂಕ op_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_kernel_params *)octx->kernel_params;
    ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_preamble;

    ಆದರೆ (octx->src[0]->type != HTP_TYPE_F32) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_Q8_0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಆದರೆ (htp_tensor_is_extended(octx->src[1])) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst = octx->dst;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_tasks    = kparams->ಒಟ್ಟು_tasks;

    uint32_t task_ಆರಂಭ = 0;
    uint32_t tasks      = ಒಟ್ಟು_tasks;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಥಿರಾಂಕ bool can_ವಿಭಜಿಸು = htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ - 1)) == 0 && !htp_tensor_is_permuted(dst);
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(ಒಟ್ಟು_tasks, can_ವಿಭಜಿಸು ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        task_ಆರಂಭ = range.ಆರಂಭ;
        tasks      = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (tasks == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, (uint32_t) kparams->n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_ಖಂಡ((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *)octx->src[1]->ದತ್ತಾಂಶ, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    ರಚನೆ ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_ಆರಂಭ = task_ಆರಂಭ;
    srctx.tasks = tasks;
    srctx.tasks_ಪ್ರತಿ_thread = fastdiv(tasks + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);

    htp_ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, n_ಎಳೆಗಳು);
    srctx.vtcm_ತಳ = (uint8_t *)octx->ctx->vtcm_ತಳ;

    work_ಸರತಿ_func_t q_func = ಖಾಲಿ;
    ಸ್ಥಿರಾಂಕ bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    ಆಯ್ಕೆ (octx->dst->type) {
        ಸ್ಥಿತಿ HTP_TYPE_F32:  q_func = is_i32 ? ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_thread_dma_f32_int32_t  : ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_thread_dma_f32_int64_t;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_TYPE_F16:  q_func = is_i32 ? ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_thread_dma_f16_int32_t  : ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_thread_dma_f16_int64_t;  ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ HTP_TYPE_Q8_0: q_func = is_i32 ? ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_thread_dma_q8_0_int32_t : ಗಣ_ಅಡ್ಡಸಾಲುಗಳು_thread_dma_q8_0_int64_t; ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    FARF(ಮೇಲಿನ_2, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_ಬೈಟುಗಳು_ಪ್ರತಿ_thread * n_ಎಳೆಗಳು,
         srctx.vtcm_layout.dst_ಬೈಟುಗಳು_ಪ್ರತಿ_thread  * n_ಎಳೆಗಳು,
         n_ಎಳೆಗಳು);

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, q_func, &srctx, n_ಎಳೆಗಳು);

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}
