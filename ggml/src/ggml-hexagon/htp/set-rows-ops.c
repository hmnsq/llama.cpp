#निर्देशः clang diagnostic ignored "-Wunused-variable"
#निर्देशः clang diagnostic ignored "-Wunused-function"
#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#अन्तर्भाव <math.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "work-queue.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "hex-utils.h"
#अन्तर्भाव "hvx-copy.h"
#अन्तर्भाव "hvx-quant.h"

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "htp/set-rows-ops.h"

#परिभाषय समुच्चयः_पङ्क्तिकाः_preamble                      \
    ध्रुव uint32_t ne00 = octx->src[0]->ne[0]; \
    ध्रुव uint32_t ne01 = octx->src[0]->ne[1]; \
    ध्रुव uint32_t ne02 = octx->src[0]->ne[2]; \
    ध्रुव uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    ध्रुव uint32_t ne10 = octx->src[1]->ne[0]; \
    ध्रुव uint32_t ne11 = octx->src[1]->ne[1]; \
    ध्रुव uint32_t ne12 = octx->src[1]->ne[2]; \
    ध्रुव uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    ध्रुव uint32_t nb01 = octx->src[0]->nb[1]; \
    ध्रुव uint32_t nb02 = octx->src[0]->nb[2]; \
    ध्रुव uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    ध्रुव uint32_t nb10 = octx->src[1]->nb[0]; \
    ध्रुव uint32_t nb11 = octx->src[1]->nb[1]; \
    ध्रुव uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    ध्रुव uint32_t nb1 = octx->dst->nb[1];     \
    ध्रुव uint32_t nb2 = octx->dst->nb[2];     \
    ध्रुव uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    ध्रुव uint32_t ne0 = octx->dst->ne[0];     \
    ध्रुव uint32_t ne1 = octx->dst->ne[1];     \
    ध्रुव uint32_t ne2 = octx->dst->ne[2];     \
    ध्रुव uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    ध्रुव uint32_t nr  = ne01;

संरचना समुच्चयः_पङ्क्तिकाः_context {
    संरचना htp_ops_context * octx;
    ध्रुव संरचना htp_समुच्चयः_पङ्क्तिकाः_kernel_params * kparams;
    संरचना htp_समुच्चयः_पङ्क्तिकाः_vtcm_layout vtcm_layout;
    uint8_t * vtcm_आधारः;
    uint32_t task_start;
    uint32_t tasks;
    uint32_t tasks_प्रत्येकम्_thread;
};

#परिभाषय समुच्चयः_पङ्क्तिकाः_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_व्यञ्जकम्)                                                \
स्थिर शून्य समुच्चयः_पङ्क्तिकाः_thread_dma##TYPE_NAME##_##IDX_TYPE(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य *दत्त) {       \
    संरचना समुच्चयः_पङ्क्तिकाः_context * srctx = (संरचना समुच्चयः_पङ्क्तिकाः_context *)दत्त;                                           \
    संरचना htp_ops_context * octx = srctx->octx;                                                                 \
    ध्रुव संरचना htp_समुच्चयः_पङ्क्तिकाः_kernel_params * kparams = srctx->kparams;                                          \
    समुच्चयः_पङ्क्तिकाः_preamble;                                                                                           \
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    ध्रुव uint32_t dr  = srctx->tasks_प्रत्येकम्_thread;                                                                \
    ध्रुव uint32_t ir0 = srctx->task_start + dr * ith;                                                           \
    यदि (ir0 >= srctx->task_start + srctx->tasks) {                                                               \
        प्रत्यागम;                                                                                                  \
    }                                                                                                            \
    ध्रुव uint32_t ir1 = MIN(ir0 + dr, srctx->task_start + srctx->tasks);                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                     \
    ध्रुव संरचना htp_समुच्चयः_पङ्क्तिकाः_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = srctx->vtcm_आधारः + vtcm_layout->off_src0 + ith * vtcm_layout->src0_अष्टकानि_प्रत्येकम्_thread;   \
    uint8_t * vtcm_dst  = srctx->vtcm_आधारः + vtcm_layout->off_dst  + ith * vtcm_layout->dst_अष्टकानि_प्रत्येकम्_thread;    \
    ध्रुव uint32_t src0_row_आकार = ne00 * परिमाण(भिन्न);                                                         \
    ध्रुव uint32_t dst_row_आकार  = htp_tensor_get_row_आकार(octx->dst->type, ne00);                               \
    ध्रुव uint32_t nrows_प्रत्येकम्_thread = ir1 - ir0;                                                                 \
    ध्रुव uint32_t योग_चरणाः = ne03 * ne02 * nrows_प्रत्येकम्_thread;                                                 \
    uint32_t pi_चरणः = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    प्रति (uint32_t चरणः = 0, spad_idx = 0; चरणः < योग_चरणाः && spad_idx < 2; ++चरणः, spad_idx++) {              \
        uint32_t i = ir0 + pi_चरणः;                                                                              \
        ध्रुव dma_addr_t src0_दत्त = octx->src[0]->दत्त + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_queue_प्रक्षिप(dma_q,                                                                                    \
                       dma_make_दत्त(octx->dst->दत्त,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_आकार),                     \
                       dst_row_आकार, vtcm_layout->dst_spad_half_आकार, dst_row_आकार, 0);                          \
        dma_queue_प्रक्षिप(dma_q,                                                                                    \
                       dma_make_दत्त(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_आकार,                    \
                                     src0_दत्त),                                                                 \
                       vtcm_layout->src0_spad_half_आकार, src0_row_आकार, src0_row_आकार, 1);                       \
        pi_चरणः++;                                                                                               \
        यदि (pi_चरणः == nrows_प्रत्येकम्_thread) {                                                                       \
            pi_चरणः = 0;                                                                                         \
            pi02++;                                                                                              \
            यदि (pi02 == ne02) {                                                                                  \
                pi02 = 0;                                                                                        \
                pi03++;                                                                                          \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    uint32_t ci_चरणः = 0;                                                                                        \
    uint32_t ci02 = 0;                                                                                           \
    uint32_t ci03 = 0;                                                                                           \
    uint32_t ci11_आधारः = 0;                                                                                      \
    uint32_t ci12_आधारः = 0;                                                                                      \
    प्रति (uint32_t चरणः = 0; चरणः < योग_चरणाः; ++चरणः) {                                                        \
        शून्य * dst_spad = (शून्य *) dma_queue_निष्कासय(dma_q).src;                                                     \
        शून्य * src_spad = (शून्य *) dma_queue_निष्कासय(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_चरणः;                                                                              \
        ध्रुव uintptr_t src1_addr = octx->src[1]->दत्त + i*nb10 + ci11_आधारः*nb11 + ci12_आधारः*nb12;               \
        ध्रुव IDX_TYPE i1 = *(ध्रुव IDX_TYPE *)src1_addr;                                                        \
        ध्रुव bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        ध्रुव uint32_t target_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, चरणः);                                                 \
        यदि (valid_i1) {                                                                                          \
            COMPUTE_व्यञ्जकम्;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, चरणः);                                                  \
        यदि (valid_i1) {                                                                                          \
            ध्रुव dma_addr_t dst_दत्त = octx->dst->दत्त + target_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_queue_प्रक्षिप(dma_q,                                                                                \
                           dma_make_दत्त(dst_दत्त, dst_spad),                                                    \
                           dst_row_आकार, vtcm_layout->dst_spad_half_आकार, dst_row_आकार, 1);                      \
        } अन्यथा {                                                                                                 \
            dma_queue_प्रक्षिप(dma_q,                                                                                \
                           dma_make_दत्त(octx->dst->दत्त, dst_spad),                                             \
                           dst_row_आकार, vtcm_layout->dst_spad_half_आकार, dst_row_आकार, 0);                      \
        }                                                                                                        \
        ध्रुव uint32_t अग्र_चरणः = चरणः + 2;                                                                     \
        यदि (अग्र_चरणः < योग_चरणाः) {                                                                           \
            uint32_t ni = ir0 + pi_चरणः;                                                                         \
            ध्रुव dma_addr_t psrc0_दत्त = octx->src[0]->दत्त + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_queue_प्रक्षिप(dma_q,                                                                                \
                           dma_make_दत्त(src_spad, psrc0_दत्त),                                                  \
                           vtcm_layout->src0_spad_half_आकार, src0_row_आकार, src0_row_आकार, 1);                   \
            pi_चरणः++;                                                                                           \
            यदि (pi_चरणः == nrows_प्रत्येकम्_thread) {                                                                   \
                pi_चरणः = 0;                                                                                     \
                pi02++;                                                                                          \
                यदि (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_चरणः++;                                                                                               \
        यदि (ci_चरणः == nrows_प्रत्येकम्_thread) {                                                                       \
            ci_चरणः = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_आधारः++;                                                                                         \
            यदि (ci11_आधारः == ne11) {                                                                             \
                ci11_आधारः = 0;                                                                                   \
            }                                                                                                    \
            यदि (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_आधारः++;                                                                                     \
                यदि (ci12_आधारः == ne12) {                                                                         \
                    ci12_आधारः = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_queue_flush(dma_q);                                                                                      \
}

समुच्चयः_पङ्क्तिकाः_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ध्रुव uint8_t *)src_spad, ne00); })
समुच्चयः_पङ्क्तिकाः_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ध्रुव uint8_t *)src_spad, ne00); })

समुच्चयः_पङ्क्तिकाः_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ध्रुव uint8_t *)src_spad, ne00); })
समुच्चयः_पङ्क्तिकाः_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ध्रुव uint8_t *)src_spad, ne00); })

समुच्चयः_पङ्क्तिकाः_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_row_q8_0_f32(dst_spad, (ध्रुव भिन्न *)src_spad, ne00); })
समुच्चयः_पङ्क्तिकाः_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_row_q8_0_f32(dst_spad, (ध्रुव भिन्न *)src_spad, ne00); })

पूर्णाङ्क op_समुच्चयः_पङ्क्तिकाः(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_समुच्चयः_पङ्क्तिकाः_kernel_params * kparams = (ध्रुव संरचना htp_समुच्चयः_पङ्क्तिकाः_kernel_params *)octx->kernel_params;
    समुच्चयः_पङ्क्तिकाः_preamble;

    यदि (octx->src[0]->type != HTP_TYPE_F32) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    यदि (octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_Q8_0) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    यदि (htp_tensor_is_extended(octx->src[1])) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव संरचना htp_tensor * dst = octx->dst;
    ध्रुव uint32_t योग_tasks    = kparams->योग_tasks;

    uint32_t task_start = 0;
    uint32_t tasks      = योग_tasks;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        ध्रुव bool can_विभज = htp_tensor_mdev_दत्त_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_पङ्क्तिः_आकार - 1)) == 0 && !htp_tensor_is_permuted(dst);
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_tasks, can_विभज ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        task_start = range.start;
        tasks      = range.गणनम्;
    }

    यदि (tasks == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    यदि (!htp_ops_context_समुच्चयः_n_तन्तवः(octx, (uint32_t) kparams->n_तन्तवः)) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_खण्डम्((ध्रुव शून्य *)octx->src[1]->दत्त, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    संरचना समुच्चयः_पङ्क्तिकाः_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_start = task_start;
    srctx.tasks = tasks;
    srctx.tasks_प्रत्येकम्_thread = fastdiv(tasks + n_तन्तवः - 1, &octx->n_तन्तवः_div);

    htp_समुच्चयः_पङ्क्तिकाः_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, n_तन्तवः);
    srctx.vtcm_आधारः = (uint8_t *)octx->ctx->vtcm_आधारः;

    work_queue_func_t q_func = रिक्त;
    ध्रुव bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    विकल्प (octx->dst->type) {
        पक्ष HTP_TYPE_F32:  q_func = is_i32 ? समुच्चयः_पङ्क्तिकाः_thread_dma_f32_int32_t  : समुच्चयः_पङ्क्तिकाः_thread_dma_f32_int64_t;  भङ्ग;
        पक्ष HTP_TYPE_F16:  q_func = is_i32 ? समुच्चयः_पङ्क्तिकाः_thread_dma_f16_int32_t  : समुच्चयः_पङ्क्तिकाः_thread_dma_f16_int64_t;  भङ्ग;
        पक्ष HTP_TYPE_Q8_0: q_func = is_i32 ? समुच्चयः_पङ्क्तिकाः_thread_dma_q8_0_int32_t : समुच्चयः_पङ्क्तिकाः_thread_dma_q8_0_int64_t; भङ्ग;
        स्वभाव:            प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    FARF(उच्चः_2, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_अष्टकानि_प्रत्येकम्_thread * n_तन्तवः,
         srctx.vtcm_layout.dst_अष्टकानि_प्रत्येकम्_thread  * n_तन्तवः,
         n_तन्तवः);

    work_queue_सततम्(octx->ctx->work_queue, q_func, &srctx, n_तन्तवः);

    प्रत्यागम HTP_स्थितिः_OK;
}
