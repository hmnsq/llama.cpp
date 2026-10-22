#आशय clang diagnostic ignored "-Wunused-variable"
#आशय clang diagnostic ignored "-Wunused-function"
#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <HAP_perf.h>

#समावेश <math.h>
#समावेश <string.h>

#समावेश "dma-queue.h"
#समावेश "work-queue.h"
#समावेश "hvx-utils.h"
#समावेश "hex-utils.h"
#समावेश "hvx-copy.h"
#समावेश "hvx-quant.h"

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"

#समावेश "hex-common.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"
#समावेश "htp/set-rows-ops.h"

#घोषणा set_rows_preamble                      \
    स्थिर uint32_t ne00 = octx->src[0]->ne[0]; \
    स्थिर uint32_t ne01 = octx->src[0]->ne[1]; \
    स्थिर uint32_t ne02 = octx->src[0]->ne[2]; \
    स्थिर uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    स्थिर uint32_t ne10 = octx->src[1]->ne[0]; \
    स्थिर uint32_t ne11 = octx->src[1]->ne[1]; \
    स्थिर uint32_t ne12 = octx->src[1]->ne[2]; \
    स्थिर uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    स्थिर uint32_t nb01 = octx->src[0]->nb[1]; \
    स्थिर uint32_t nb02 = octx->src[0]->nb[2]; \
    स्थिर uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    स्थिर uint32_t nb10 = octx->src[1]->nb[0]; \
    स्थिर uint32_t nb11 = octx->src[1]->nb[1]; \
    स्थिर uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    स्थिर uint32_t nb1 = octx->dst->nb[1];     \
    स्थिर uint32_t nb2 = octx->dst->nb[2];     \
    स्थिर uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    स्थिर uint32_t ne0 = octx->dst->ne[0];     \
    स्थिर uint32_t ne1 = octx->dst->ne[1];     \
    स्थिर uint32_t ne2 = octx->dst->ne[2];     \
    स्थिर uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    स्थिर uint32_t nr  = ne01;

काष्ठा set_rows_context {
    काष्ठा htp_ops_context * octx;
    स्थिर काष्ठा htp_set_rows_kernel_params * kparams;
    काष्ठा htp_set_rows_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;
    uint32_t task_start;
    uint32_t tasks;
    uint32_t tasks_per_thread;
};

#घोषणा SET_ROWS_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_व्यंजक)                                                \
अटल व्योम set_rows_thread_dma_##TYPE_NAME##_##IDX_TYPE(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम *तथ्य) {       \
    काष्ठा set_rows_context * srctx = (काष्ठा set_rows_context *)तथ्य;                                           \
    काष्ठा htp_ops_context * octx = srctx->octx;                                                                 \
    स्थिर काष्ठा htp_set_rows_kernel_params * kparams = srctx->kparams;                                          \
    set_rows_preamble;                                                                                           \
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    स्थिर uint32_t dr  = srctx->tasks_per_thread;                                                                \
    स्थिर uint32_t ir0 = srctx->task_start + dr * ith;                                                           \
    अगर (ir0 >= srctx->task_start + srctx->tasks) {                                                               \
        वापस;                                                                                                  \
    }                                                                                                            \
    स्थिर uint32_t ir1 = MIN(ir0 + dr, srctx->task_start + srctx->tasks);                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                     \
    स्थिर काष्ठा htp_set_rows_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = srctx->vtcm_base + vtcm_layout->off_src0 + ith * vtcm_layout->src0_bytes_per_thread;   \
    uint8_t * vtcm_dst  = srctx->vtcm_base + vtcm_layout->off_dst  + ith * vtcm_layout->dst_bytes_per_thread;    \
    स्थिर uint32_t src0_row_आकार = ne00 * माप(भग्न);                                                         \
    स्थिर uint32_t dst_row_आकार  = htp_tensor_get_row_आकार(octx->dst->type, ne00);                               \
    स्थिर uint32_t nrows_per_thread = ir1 - ir0;                                                                 \
    स्थिर uint32_t योग_steps = ne03 * ne02 * nrows_per_thread;                                                 \
    uint32_t pi_step = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    क्रम (uint32_t step = 0, spad_idx = 0; step < योग_steps && spad_idx < 2; ++step, spad_idx++) {              \
        uint32_t i = ir0 + pi_step;                                                                              \
        स्थिर dma_addr_t src0_तथ्य = octx->src[0]->तथ्य + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_queue_धकेलो(dma_q,                                                                                    \
                       dma_make_तथ्य(octx->dst->तथ्य,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_आकार),                     \
                       dst_row_आकार, vtcm_layout->dst_spad_half_आकार, dst_row_आकार, 0);                          \
        dma_queue_धकेलो(dma_q,                                                                                    \
                       dma_make_तथ्य(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_आकार,                    \
                                     src0_तथ्य),                                                                 \
                       vtcm_layout->src0_spad_half_आकार, src0_row_आकार, src0_row_आकार, 1);                       \
        pi_step++;                                                                                               \
        अगर (pi_step == nrows_per_thread) {                                                                       \
            pi_step = 0;                                                                                         \
            pi02++;                                                                                              \
            अगर (pi02 == ne02) {                                                                                  \
                pi02 = 0;                                                                                        \
                pi03++;                                                                                          \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    uint32_t ci_step = 0;                                                                                        \
    uint32_t ci02 = 0;                                                                                           \
    uint32_t ci03 = 0;                                                                                           \
    uint32_t ci11_base = 0;                                                                                      \
    uint32_t ci12_base = 0;                                                                                      \
    क्रम (uint32_t step = 0; step < योग_steps; ++step) {                                                        \
        व्योम * dst_spad = (व्योम *) dma_queue_निकालो(dma_q).src;                                                     \
        व्योम * src_spad = (व्योम *) dma_queue_निकालो(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_step;                                                                              \
        स्थिर uintptr_t src1_addr = octx->src[1]->तथ्य + i*nb10 + ci11_base*nb11 + ci12_base*nb12;               \
        स्थिर IDX_TYPE i1 = *(स्थिर IDX_TYPE *)src1_addr;                                                        \
        स्थिर bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        स्थिर uint32_t target_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                 \
        अगर (valid_i1) {                                                                                          \
            COMPUTE_व्यंजक;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                  \
        अगर (valid_i1) {                                                                                          \
            स्थिर dma_addr_t dst_तथ्य = octx->dst->तथ्य + target_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_queue_धकेलो(dma_q,                                                                                \
                           dma_make_तथ्य(dst_तथ्य, dst_spad),                                                    \
                           dst_row_आकार, vtcm_layout->dst_spad_half_आकार, dst_row_आकार, 1);                      \
        } अन्यथा {                                                                                                 \
            dma_queue_धकेलो(dma_q,                                                                                \
                           dma_make_तथ्य(octx->dst->तथ्य, dst_spad),                                             \
                           dst_row_आकार, vtcm_layout->dst_spad_half_आकार, dst_row_आकार, 0);                      \
        }                                                                                                        \
        स्थिर uint32_t अग्र_step = step + 2;                                                                     \
        अगर (अग्र_step < योग_steps) {                                                                           \
            uint32_t ni = ir0 + pi_step;                                                                         \
            स्थिर dma_addr_t psrc0_तथ्य = octx->src[0]->तथ्य + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_queue_धकेलो(dma_q,                                                                                \
                           dma_make_तथ्य(src_spad, psrc0_तथ्य),                                                  \
                           vtcm_layout->src0_spad_half_आकार, src0_row_आकार, src0_row_आकार, 1);                   \
            pi_step++;                                                                                           \
            अगर (pi_step == nrows_per_thread) {                                                                   \
                pi_step = 0;                                                                                     \
                pi02++;                                                                                          \
                अगर (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_step++;                                                                                               \
        अगर (ci_step == nrows_per_thread) {                                                                       \
            ci_step = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_base++;                                                                                         \
            अगर (ci11_base == ne11) {                                                                             \
                ci11_base = 0;                                                                                   \
            }                                                                                                    \
            अगर (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_base++;                                                                                     \
                अगर (ci12_base == ne12) {                                                                         \
                    ci12_base = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_queue_flush(dma_q);                                                                                      \
}

SET_ROWS_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (स्थिर uint8_t *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (स्थिर uint8_t *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (स्थिर uint8_t *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (स्थिर uint8_t *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_row_q8_0_f32(dst_spad, (स्थिर भग्न *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_row_q8_0_f32(dst_spad, (स्थिर भग्न *)src_spad, ne00); })

पूर्णांक op_set_rows(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_set_rows_kernel_params * kparams = (स्थिर काष्ठा htp_set_rows_kernel_params *)octx->kernel_params;
    set_rows_preamble;

    अगर (octx->src[0]->type != HTP_TYPE_F32) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    अगर (octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_Q8_0) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    अगर (htp_tensor_is_extended(octx->src[1])) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    स्थिर काष्ठा htp_tensor * dst = octx->dst;
    स्थिर uint32_t योग_tasks    = kparams->योग_tasks;

    uint32_t task_start = 0;
    uint32_t tasks      = योग_tasks;

    अगर (octx->ctx->mdev.गिनती > 1) {
        स्थिर bool can_split = htp_tensor_mdev_तथ्य_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_पंक्ति_आकार - 1)) == 0 && !htp_tensor_is_permuted(dst);
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_tasks, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        task_start = range.start;
        tasks      = range.गिनती;
    }

    अगर (tasks == 0) {
        वापस HTP_STATUS_OK;
    }

    अगर (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    स्थिर uint32_t n_threads = octx->n_threads;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_block((स्थिर व्योम *)octx->src[1]->तथ्य, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    काष्ठा set_rows_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_start = task_start;
    srctx.tasks = tasks;
    srctx.tasks_per_thread = fastdiv(tasks + n_threads - 1, &octx->n_threads_div);

    htp_set_rows_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, n_threads);
    srctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;

    work_queue_func_t q_func = शून्य;
    स्थिर bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    चयन (octx->dst->type) {
        हाल HTP_TYPE_F32:  q_func = is_i32 ? set_rows_thread_dma_f32_int32_t  : set_rows_thread_dma_f32_int64_t;  अवरोध;
        हाल HTP_TYPE_F16:  q_func = is_i32 ? set_rows_thread_dma_f16_int32_t  : set_rows_thread_dma_f16_int64_t;  अवरोध;
        हाल HTP_TYPE_Q8_0: q_func = is_i32 ? set_rows_thread_dma_q8_0_int32_t : set_rows_thread_dma_q8_0_int64_t; अवरोध;
        शेष:            वापस HTP_STATUS_NO_SUPPORT;
    }

    FARF(ऊपरी_2, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_bytes_per_thread * n_threads,
         srctx.vtcm_layout.dst_bytes_per_thread  * n_threads,
         n_threads);

    work_queue_run(octx->ctx->work_queue, q_func, &srctx, n_threads);

    वापस HTP_STATUS_OK;
}
