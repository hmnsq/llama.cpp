#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

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

#define GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "htp/set-rows-ops.h"

#define set_rows_preamble                      \
    अचर uint32_t ne00 = octx->src[0]->ne[0]; \
    अचर uint32_t ne01 = octx->src[0]->ne[1]; \
    अचर uint32_t ne02 = octx->src[0]->ne[2]; \
    अचर uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    अचर uint32_t ne10 = octx->src[1]->ne[0]; \
    अचर uint32_t ne11 = octx->src[1]->ne[1]; \
    अचर uint32_t ne12 = octx->src[1]->ne[2]; \
    अचर uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    अचर uint32_t nb01 = octx->src[0]->nb[1]; \
    अचर uint32_t nb02 = octx->src[0]->nb[2]; \
    अचर uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    अचर uint32_t nb10 = octx->src[1]->nb[0]; \
    अचर uint32_t nb11 = octx->src[1]->nb[1]; \
    अचर uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    अचर uint32_t nb1 = octx->dst->nb[1];     \
    अचर uint32_t nb2 = octx->dst->nb[2];     \
    अचर uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    अचर uint32_t ne0 = octx->dst->ne[0];     \
    अचर uint32_t ne1 = octx->dst->ne[1];     \
    अचर uint32_t ne2 = octx->dst->ne[2];     \
    अचर uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    अचर uint32_t nr  = ne01;

संरचना set_rows_context {
    संरचना htp_ops_context * octx;
    अचर संरचना htp_set_rows_kernel_params * kparams;
    संरचना htp_set_rows_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;
    uint32_t task_start;
    uint32_t tasks;
    uint32_t tasks_per_thread;
};

#define SET_ROWS_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_अभिव्यक्ति)                                                \
स्थिर शून्य set_rows_thread_dma_##TYPE_NAME##_##IDX_TYPE(अचिन्हित पूर्णाङ्क nth, अचिन्हित पूर्णाङ्क ith, शून्य *data) {       \
    संरचना set_rows_context * srctx = (संरचना set_rows_context *)data;                                           \
    संरचना htp_ops_context * octx = srctx->octx;                                                                 \
    अचर संरचना htp_set_rows_kernel_params * kparams = srctx->kparams;                                          \
    set_rows_preamble;                                                                                           \
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    अचर uint32_t dr  = srctx->tasks_per_thread;                                                                \
    अचर uint32_t ir0 = srctx->task_start + dr * ith;                                                           \
    यदि (ir0 >= srctx->task_start + srctx->tasks) {                                                               \
        फर्काऊ;                                                                                                  \
    }                                                                                                            \
    अचर uint32_t ir1 = MIN(ir0 + dr, srctx->task_start + srctx->tasks);                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                     \
    अचर संरचना htp_set_rows_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = srctx->vtcm_base + vtcm_layout->off_src0 + ith * vtcm_layout->src0_bytes_per_thread;   \
    uint8_t * vtcm_dst  = srctx->vtcm_base + vtcm_layout->off_dst  + ith * vtcm_layout->dst_bytes_per_thread;    \
    अचर uint32_t src0_row_size = ne00 * परिमाण(भिन्न);                                                         \
    अचर uint32_t dst_row_size  = htp_tensor_get_row_size(octx->dst->type, ne00);                               \
    अचर uint32_t nrows_per_thread = ir1 - ir0;                                                                 \
    अचर uint32_t total_steps = ne03 * ne02 * nrows_per_thread;                                                 \
    uint32_t pi_step = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    लागि (uint32_t step = 0, spad_idx = 0; step < total_steps && spad_idx < 2; ++step, spad_idx++) {              \
        uint32_t i = ir0 + pi_step;                                                                              \
        अचर dma_addr_t src0_data = octx->src[0]->data + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_queue_थप(dma_q,                                                                                    \
                       dma_make_data(octx->dst->data,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_size),                     \
                       dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 0);                          \
        dma_queue_थप(dma_q,                                                                                    \
                       dma_make_data(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_size,                    \
                                     src0_data),                                                                 \
                       vtcm_layout->src0_spad_half_size, src0_row_size, src0_row_size, 1);                       \
        pi_step++;                                                                                               \
        यदि (pi_step == nrows_per_thread) {                                                                       \
            pi_step = 0;                                                                                         \
            pi02++;                                                                                              \
            यदि (pi02 == ne02) {                                                                                  \
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
    लागि (uint32_t step = 0; step < total_steps; ++step) {                                                        \
        शून्य * dst_spad = (शून्य *) dma_queue_झिक(dma_q).src;                                                     \
        शून्य * src_spad = (शून्य *) dma_queue_झिक(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_step;                                                                              \
        अचर uintptr_t src1_addr = octx->src[1]->data + i*nb10 + ci11_base*nb11 + ci12_base*nb12;               \
        अचर IDX_TYPE i1 = *(अचर IDX_TYPE *)src1_addr;                                                        \
        अचर bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        अचर uint32_t target_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                 \
        यदि (valid_i1) {                                                                                          \
            COMPUTE_अभिव्यक्ति;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                  \
        यदि (valid_i1) {                                                                                          \
            अचर dma_addr_t dst_data = octx->dst->data + target_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_queue_थप(dma_q,                                                                                \
                           dma_make_data(dst_data, dst_spad),                                                    \
                           dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 1);                      \
        } अन्यथा {                                                                                                 \
            dma_queue_थप(dma_q,                                                                                \
                           dma_make_data(octx->dst->data, dst_spad),                                             \
                           dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 0);                      \
        }                                                                                                        \
        अचर uint32_t next_step = step + 2;                                                                     \
        यदि (next_step < total_steps) {                                                                           \
            uint32_t ni = ir0 + pi_step;                                                                         \
            अचर dma_addr_t psrc0_data = octx->src[0]->data + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_queue_थप(dma_q,                                                                                \
                           dma_make_data(src_spad, psrc0_data),                                                  \
                           vtcm_layout->src0_spad_half_size, src0_row_size, src0_row_size, 1);                   \
            pi_step++;                                                                                           \
            यदि (pi_step == nrows_per_thread) {                                                                   \
                pi_step = 0;                                                                                     \
                pi02++;                                                                                          \
                यदि (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_step++;                                                                                               \
        यदि (ci_step == nrows_per_thread) {                                                                       \
            ci_step = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_base++;                                                                                         \
            यदि (ci11_base == ne11) {                                                                             \
                ci11_base = 0;                                                                                   \
            }                                                                                                    \
            यदि (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_base++;                                                                                     \
                यदि (ci12_base == ne12) {                                                                         \
                    ci12_base = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_queue_flush(dma_q);                                                                                      \
}

SET_ROWS_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (अचर uint8_t *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (अचर uint8_t *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (अचर uint8_t *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (अचर uint8_t *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_row_q8_0_f32(dst_spad, (अचर भिन्न *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_row_q8_0_f32(dst_spad, (अचर भिन्न *)src_spad, ne00); })

पूर्णाङ्क op_set_rows(संरचना htp_ops_context * octx) {
    अचर संरचना htp_set_rows_kernel_params * kparams = (अचर संरचना htp_set_rows_kernel_params *)octx->kernel_params;
    set_rows_preamble;

    यदि (octx->src[0]->type != HTP_TYPE_F32) {
        फर्काऊ HTP_STATUS_NO_SUPPORT;
    }

    यदि (octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_Q8_0) {
        फर्काऊ HTP_STATUS_NO_SUPPORT;
    }

    यदि (htp_tensor_is_extended(octx->src[1])) {
        फर्काऊ HTP_STATUS_NO_SUPPORT;
    }

    अचर संरचना htp_tensor * dst = octx->dst;
    अचर uint32_t total_tasks    = kparams->total_tasks;

    uint32_t task_start = 0;
    uint32_t tasks      = total_tasks;

    यदि (octx->ctx->mdev.गन्ती > 1) {
        अचर bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_पङ्क्ति_SIZE - 1)) == 0 && !htp_tensor_is_permuted(dst);
        अचर संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_tasks, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गन्ती, &octx->ctx->mdev.गन्ती_div);
        task_start = range.start;
        tasks      = range.गन्ती;
    }

    यदि (tasks == 0) {
        फर्काऊ HTP_STATUS_OK;
    }

    यदि (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        फर्काऊ HTP_STATUS_INVAL_PARAMS;
    }

    अचर uint32_t n_threads = octx->n_threads;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_block((अचर शून्य *)octx->src[1]->data, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    संरचना set_rows_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_start = task_start;
    srctx.tasks = tasks;
    srctx.tasks_per_thread = fastdiv(tasks + n_threads - 1, &octx->n_threads_div);

    htp_set_rows_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, n_threads);
    srctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;

    work_queue_func_t q_func = रिक्त;
    अचर bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    विकल्प (octx->dst->type) {
        पक्ष HTP_TYPE_F32:  q_func = is_i32 ? set_rows_thread_dma_f32_int32_t  : set_rows_thread_dma_f32_int64_t;  रोक;
        पक्ष HTP_TYPE_F16:  q_func = is_i32 ? set_rows_thread_dma_f16_int32_t  : set_rows_thread_dma_f16_int64_t;  रोक;
        पक्ष HTP_TYPE_Q8_0: q_func = is_i32 ? set_rows_thread_dma_q8_0_int32_t : set_rows_thread_dma_q8_0_int64_t; रोक;
        स्वभाव:            फर्काऊ HTP_STATUS_NO_SUPPORT;
    }

    FARF(माथिल्लो_2, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_bytes_per_thread * n_threads,
         srctx.vtcm_layout.dst_bytes_per_thread  * n_threads,
         n_threads);

    work_queue_run(octx->ctx->work_queue, q_func, &srctx, n_threads);

    फर्काऊ HTP_STATUS_OK;
}
