#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "work-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-copy.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-quant.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp/set-rows-ops.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸੈੱਟ_ਪੰਕਤੀਆਂ_preamble                      \
    ਅਚਲ uint32_t ne00 = octx->src[0]->ne[0]; \
    ਅਚਲ uint32_t ne01 = octx->src[0]->ne[1]; \
    ਅਚਲ uint32_t ne02 = octx->src[0]->ne[2]; \
    ਅਚਲ uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    ਅਚਲ uint32_t ne10 = octx->src[1]->ne[0]; \
    ਅਚਲ uint32_t ne11 = octx->src[1]->ne[1]; \
    ਅਚਲ uint32_t ne12 = octx->src[1]->ne[2]; \
    ਅਚਲ uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    ਅਚਲ uint32_t nb01 = octx->src[0]->nb[1]; \
    ਅਚਲ uint32_t nb02 = octx->src[0]->nb[2]; \
    ਅਚਲ uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    ਅਚਲ uint32_t nb10 = octx->src[1]->nb[0]; \
    ਅਚਲ uint32_t nb11 = octx->src[1]->nb[1]; \
    ਅਚਲ uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    ਅਚਲ uint32_t nb1 = octx->dst->nb[1];     \
    ਅਚਲ uint32_t nb2 = octx->dst->nb[2];     \
    ਅਚਲ uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    ਅਚਲ uint32_t ne0 = octx->dst->ne[0];     \
    ਅਚਲ uint32_t ne1 = octx->dst->ne[1];     \
    ਅਚਲ uint32_t ne2 = octx->dst->ne[2];     \
    ਅਚਲ uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    ਅਚਲ uint32_t nr  = ne01;

ਬਣਤਰ ਸੈੱਟ_ਪੰਕਤੀਆਂ_context {
    ਬਣਤਰ htp_ops_context * octx;
    ਅਚਲ ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_kernel_params * kparams;
    ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_vtcm_layout vtcm_layout;
    uint8_t * vtcm_ਅਧਾਰ;
    uint32_t task_ਸ਼ੁਰੂ;
    uint32_t tasks;
    uint32_t tasks_ਪ੍ਰਤੀ_thread;
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਸੈੱਟ_ਪੰਕਤੀਆਂ_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_ਵਿਅੰਜਕ)                                                \
ਸਥਿਰ ਖਾਲੀ ਸੈੱਟ_ਪੰਕਤੀਆਂ_thread_dma##TYPE_NAME##_##IDX_TYPE(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ *ਅੰਕੜੇ) {       \
    ਬਣਤਰ ਸੈੱਟ_ਪੰਕਤੀਆਂ_context * srctx = (ਬਣਤਰ ਸੈੱਟ_ਪੰਕਤੀਆਂ_context *)ਅੰਕੜੇ;                                           \
    ਬਣਤਰ htp_ops_context * octx = srctx->octx;                                                                 \
    ਅਚਲ ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_kernel_params * kparams = srctx->kparams;                                          \
    ਸੈੱਟ_ਪੰਕਤੀਆਂ_preamble;                                                                                           \
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    ਅਚਲ uint32_t dr  = srctx->tasks_ਪ੍ਰਤੀ_thread;                                                                \
    ਅਚਲ uint32_t ir0 = srctx->task_ਸ਼ੁਰੂ + dr * ith;                                                           \
    ਜੇ (ir0 >= srctx->task_ਸ਼ੁਰੂ + srctx->tasks) {                                                               \
        ਵਾਪਸ;                                                                                                  \
    }                                                                                                            \
    ਅਚਲ uint32_t ir1 = MIN(ir0 + dr, srctx->task_ਸ਼ੁਰੂ + srctx->tasks);                                        \
    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];                                                                     \
    ਅਚਲ ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = srctx->vtcm_ਅਧਾਰ + vtcm_layout->off_src0 + ith * vtcm_layout->src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;   \
    uint8_t * vtcm_dst  = srctx->vtcm_ਅਧਾਰ + vtcm_layout->off_dst  + ith * vtcm_layout->dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread;    \
    ਅਚਲ uint32_t src0_ਪੰਕਤੀ_ਮਾਪ = ne00 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);                                                         \
    ਅਚਲ uint32_t dst_ਪੰਕਤੀ_ਮਾਪ  = htp_tensor_get_ਪੰਕਤੀ_ਮਾਪ(octx->dst->type, ne00);                               \
    ਅਚਲ uint32_t nrows_ਪ੍ਰਤੀ_thread = ir1 - ir0;                                                                 \
    ਅਚਲ uint32_t ਕੁੱਲ_ਕਦਮਾਂ = ne03 * ne02 * nrows_ਪ੍ਰਤੀ_thread;                                                 \
    uint32_t pi_ਕਦਮ = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    ਲਈ (uint32_t ਕਦਮ = 0, spad_idx = 0; ਕਦਮ < ਕੁੱਲ_ਕਦਮਾਂ && spad_idx < 2; ++ਕਦਮ, spad_idx++) {              \
        uint32_t i = ir0 + pi_ਕਦਮ;                                                                              \
        ਅਚਲ dma_addr_t src0_ਅੰਕੜੇ = octx->src[0]->ਅੰਕੜੇ + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,                                                                                    \
                       dma_make_ਅੰਕੜੇ(octx->dst->ਅੰਕੜੇ,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_ਮਾਪ),                     \
                       dst_ਪੰਕਤੀ_ਮਾਪ, vtcm_layout->dst_spad_half_ਮਾਪ, dst_ਪੰਕਤੀ_ਮਾਪ, 0);                          \
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,                                                                                    \
                       dma_make_ਅੰਕੜੇ(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_ਮਾਪ,                    \
                                     src0_ਅੰਕੜੇ),                                                                 \
                       vtcm_layout->src0_spad_half_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 1);                       \
        pi_ਕਦਮ++;                                                                                               \
        ਜੇ (pi_ਕਦਮ == nrows_ਪ੍ਰਤੀ_thread) {                                                                       \
            pi_ਕਦਮ = 0;                                                                                         \
            pi02++;                                                                                              \
            ਜੇ (pi02 == ne02) {                                                                                  \
                pi02 = 0;                                                                                        \
                pi03++;                                                                                          \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    uint32_t ci_ਕਦਮ = 0;                                                                                        \
    uint32_t ci02 = 0;                                                                                           \
    uint32_t ci03 = 0;                                                                                           \
    uint32_t ci11_ਅਧਾਰ = 0;                                                                                      \
    uint32_t ci12_ਅਧਾਰ = 0;                                                                                      \
    ਲਈ (uint32_t ਕਦਮ = 0; ਕਦਮ < ਕੁੱਲ_ਕਦਮਾਂ; ++ਕਦਮ) {                                                        \
        ਖਾਲੀ * dst_spad = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).src;                                                     \
        ਖਾਲੀ * src_spad = (ਖਾਲੀ *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_ਕਦਮ;                                                                              \
        ਅਚਲ uintptr_t src1_addr = octx->src[1]->ਅੰਕੜੇ + i*nb10 + ci11_ਅਧਾਰ*nb11 + ci12_ਅਧਾਰ*nb12;               \
        ਅਚਲ IDX_TYPE i1 = *(ਅਚਲ IDX_TYPE *)src1_addr;                                                        \
        ਅਚਲ bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        ਅਚਲ uint32_t ਨਿਸ਼ਾਨਾ_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, ਕਦਮ);                                                 \
        ਜੇ (valid_i1) {                                                                                          \
            COMPUTE_ਵਿਅੰਜਕ;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ਕਦਮ);                                                  \
        ਜੇ (valid_i1) {                                                                                          \
            ਅਚਲ dma_addr_t dst_ਅੰਕੜੇ = octx->dst->ਅੰਕੜੇ + ਨਿਸ਼ਾਨਾ_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,                                                                                \
                           dma_make_ਅੰਕੜੇ(dst_ਅੰਕੜੇ, dst_spad),                                                    \
                           dst_ਪੰਕਤੀ_ਮਾਪ, vtcm_layout->dst_spad_half_ਮਾਪ, dst_ਪੰਕਤੀ_ਮਾਪ, 1);                      \
        } ਨਹੀਂਤਾਂ {                                                                                                 \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,                                                                                \
                           dma_make_ਅੰਕੜੇ(octx->dst->ਅੰਕੜੇ, dst_spad),                                             \
                           dst_ਪੰਕਤੀ_ਮਾਪ, vtcm_layout->dst_spad_half_ਮਾਪ, dst_ਪੰਕਤੀ_ਮਾਪ, 0);                      \
        }                                                                                                        \
        ਅਚਲ uint32_t ਅਗਲਾ_ਕਦਮ = ਕਦਮ + 2;                                                                     \
        ਜੇ (ਅਗਲਾ_ਕਦਮ < ਕੁੱਲ_ਕਦਮਾਂ) {                                                                           \
            uint32_t ni = ir0 + pi_ਕਦਮ;                                                                         \
            ਅਚਲ dma_addr_t psrc0_ਅੰਕੜੇ = octx->src[0]->ਅੰਕੜੇ + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q,                                                                                \
                           dma_make_ਅੰਕੜੇ(src_spad, psrc0_ਅੰਕੜੇ),                                                  \
                           vtcm_layout->src0_spad_half_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, src0_ਪੰਕਤੀ_ਮਾਪ, 1);                   \
            pi_ਕਦਮ++;                                                                                           \
            ਜੇ (pi_ਕਦਮ == nrows_ਪ੍ਰਤੀ_thread) {                                                                   \
                pi_ਕਦਮ = 0;                                                                                     \
                pi02++;                                                                                          \
                ਜੇ (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_ਕਦਮ++;                                                                                               \
        ਜੇ (ci_ਕਦਮ == nrows_ਪ੍ਰਤੀ_thread) {                                                                       \
            ci_ਕਦਮ = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_ਅਧਾਰ++;                                                                                         \
            ਜੇ (ci11_ਅਧਾਰ == ne11) {                                                                             \
                ci11_ਅਧਾਰ = 0;                                                                                   \
            }                                                                                                    \
            ਜੇ (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_ਅਧਾਰ++;                                                                                     \
                ਜੇ (ci12_ਅਧਾਰ == ne12) {                                                                         \
                    ci12_ਅਧਾਰ = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_ਕਤਾਰ_flush(dma_q);                                                                                      \
}

ਸੈੱਟ_ਪੰਕਤੀਆਂ_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ਅਚਲ uint8_t *)src_spad, ne00); })
ਸੈੱਟ_ਪੰਕਤੀਆਂ_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ਅਚਲ uint8_t *)src_spad, ne00); })

ਸੈੱਟ_ਪੰਕਤੀਆਂ_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ਅਚਲ uint8_t *)src_spad, ne00); })
ਸੈੱਟ_ਪੰਕਤੀਆਂ_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ਅਚਲ uint8_t *)src_spad, ne00); })

ਸੈੱਟ_ਪੰਕਤੀਆਂ_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_ਪੰਕਤੀ_q8_0_f32(dst_spad, (ਅਚਲ ਦਸ਼ਮਲਵ *)src_spad, ne00); })
ਸੈੱਟ_ਪੰਕਤੀਆਂ_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_ਪੰਕਤੀ_q8_0_f32(dst_spad, (ਅਚਲ ਦਸ਼ਮਲਵ *)src_spad, ne00); })

ਪੂਰਨ op_ਸੈੱਟ_ਪੰਕਤੀਆਂ(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_kernel_params *)octx->kernel_params;
    ਸੈੱਟ_ਪੰਕਤੀਆਂ_preamble;

    ਜੇ (octx->src[0]->type != HTP_TYPE_F32) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_Q8_0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਜੇ (htp_tensor_is_extended(octx->src[1])) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ ਬਣਤਰ htp_tensor * dst = octx->dst;
    ਅਚਲ uint32_t ਕੁੱਲ_tasks    = kparams->ਕੁੱਲ_tasks;

    uint32_t task_ਸ਼ੁਰੂ = 0;
    uint32_t tasks      = ਕੁੱਲ_tasks;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਅਚਲ bool can_ਵੰਡੋ = htp_tensor_mdev_ਅੰਕੜੇ_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ - 1)) == 0 && !htp_tensor_is_permuted(dst);
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(ਕੁੱਲ_tasks, can_ਵੰਡੋ ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        task_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        tasks      = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (tasks == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, (uint32_t) kparams->n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_ਖੰਡ((ਅਚਲ ਖਾਲੀ *)octx->src[1]->ਅੰਕੜੇ, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    ਬਣਤਰ ਸੈੱਟ_ਪੰਕਤੀਆਂ_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_ਸ਼ੁਰੂ = task_ਸ਼ੁਰੂ;
    srctx.tasks = tasks;
    srctx.tasks_ਪ੍ਰਤੀ_thread = fastdiv(tasks + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);

    htp_ਸੈੱਟ_ਪੰਕਤੀਆਂ_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, n_ਧਾਗੇ);
    srctx.vtcm_ਅਧਾਰ = (uint8_t *)octx->ctx->vtcm_ਅਧਾਰ;

    work_ਕਤਾਰ_func_t q_func = ਨਿਰਾ;
    ਅਚਲ bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    ਚੋਣ (octx->dst->type) {
        ਹਾਲਤ HTP_TYPE_F32:  q_func = is_i32 ? ਸੈੱਟ_ਪੰਕਤੀਆਂ_thread_dma_f32_int32_t  : ਸੈੱਟ_ਪੰਕਤੀਆਂ_thread_dma_f32_int64_t;  ਰੋਕੋ;
        ਹਾਲਤ HTP_TYPE_F16:  q_func = is_i32 ? ਸੈੱਟ_ਪੰਕਤੀਆਂ_thread_dma_f16_int32_t  : ਸੈੱਟ_ਪੰਕਤੀਆਂ_thread_dma_f16_int64_t;  ਰੋਕੋ;
        ਹਾਲਤ HTP_TYPE_Q8_0: q_func = is_i32 ? ਸੈੱਟ_ਪੰਕਤੀਆਂ_thread_dma_q8_0_int32_t : ਸੈੱਟ_ਪੰਕਤੀਆਂ_thread_dma_q8_0_int64_t; ਰੋਕੋ;
        ਮੂਲ:            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    FARF(ਉੱਚਾ_2, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread * n_ਧਾਗੇ,
         srctx.vtcm_layout.dst_ਬਾਈਟਾਂ_ਪ੍ਰਤੀ_thread  * n_ਧਾਗੇ,
         n_ਧਾਗੇ);

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, q_func, &srctx, n_ਧਾਗੇ);

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}
