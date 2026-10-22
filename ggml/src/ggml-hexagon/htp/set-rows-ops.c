#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#அந்தர்பாவ <HAP_farf.h>
#அந்தர்பாவ <HAP_perf.h>

#அந்தர்பாவ <math.h>
#அந்தர்பாவ <string.h>

#அந்தர்பாவ "dma-queue.h"
#அந்தர்பாவ "work-queue.h"
#அந்தர்பாவ "hvx-utils.h"
#அந்தர்பாவ "hex-utils.h"
#அந்தர்பாவ "hvx-copy.h"
#அந்தர்பாவ "hvx-quant.h"

#define GGML_COMMON_DECL_C
#அந்தர்பாவ "ggml-common.h"

#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "htp-ctx.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp-tensor.h"
#அந்தர்பாவ "htp/set-rows-ops.h"

#define set_rows_preamble                      \
    மாறிலி uint32_t ne00 = octx->src[0]->ne[0]; \
    மாறிலி uint32_t ne01 = octx->src[0]->ne[1]; \
    மாறிலி uint32_t ne02 = octx->src[0]->ne[2]; \
    மாறிலி uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    மாறிலி uint32_t ne10 = octx->src[1]->ne[0]; \
    மாறிலி uint32_t ne11 = octx->src[1]->ne[1]; \
    மாறிலி uint32_t ne12 = octx->src[1]->ne[2]; \
    மாறிலி uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    மாறிலி uint32_t nb01 = octx->src[0]->nb[1]; \
    மாறிலி uint32_t nb02 = octx->src[0]->nb[2]; \
    மாறிலி uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    மாறிலி uint32_t nb10 = octx->src[1]->nb[0]; \
    மாறிலி uint32_t nb11 = octx->src[1]->nb[1]; \
    மாறிலி uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    மாறிலி uint32_t nb1 = octx->dst->nb[1];     \
    மாறிலி uint32_t nb2 = octx->dst->nb[2];     \
    மாறிலி uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    மாறிலி uint32_t ne0 = octx->dst->ne[0];     \
    மாறிலி uint32_t ne1 = octx->dst->ne[1];     \
    மாறிலி uint32_t ne2 = octx->dst->ne[2];     \
    மாறிலி uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    மாறிலி uint32_t nr  = ne01;

கட்டமைப்பு set_rows_context {
    கட்டமைப்பு htp_ops_context * octx;
    மாறிலி கட்டமைப்பு htp_set_rows_kernel_params * kparams;
    கட்டமைப்பு htp_set_rows_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;
    uint32_t task_start;
    uint32_t tasks;
    uint32_t tasks_per_thread;
};

#define SET_ROWS_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_கோவை)                                                \
நிலையான வெறுமை set_rows_thread_dma_##TYPE_NAME##_##IDX_TYPE(unsigned முழு nth, unsigned முழு ith, வெறுமை *data) {       \
    கட்டமைப்பு set_rows_context * srctx = (கட்டமைப்பு set_rows_context *)data;                                           \
    கட்டமைப்பு htp_ops_context * octx = srctx->octx;                                                                 \
    மாறிலி கட்டமைப்பு htp_set_rows_kernel_params * kparams = srctx->kparams;                                          \
    set_rows_preamble;                                                                                           \
    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    மாறிலி uint32_t dr  = srctx->tasks_per_thread;                                                                \
    மாறிலி uint32_t ir0 = srctx->task_start + dr * ith;                                                           \
    எனில் (ir0 >= srctx->task_start + srctx->tasks) {                                                               \
        திரும்பு;                                                                                                  \
    }                                                                                                            \
    மாறிலி uint32_t ir1 = MIN(ir0 + dr, srctx->task_start + srctx->tasks);                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                     \
    மாறிலி கட்டமைப்பு htp_set_rows_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = srctx->vtcm_base + vtcm_layout->off_src0 + ith * vtcm_layout->src0_bytes_per_thread;   \
    uint8_t * vtcm_dst  = srctx->vtcm_base + vtcm_layout->off_dst  + ith * vtcm_layout->dst_bytes_per_thread;    \
    மாறிலி uint32_t src0_row_size = ne00 * அளவு(தசம);                                                         \
    மாறிலி uint32_t dst_row_size  = htp_tensor_get_row_size(octx->dst->type, ne00);                               \
    மாறிலி uint32_t nrows_per_thread = ir1 - ir0;                                                                 \
    மாறிலி uint32_t total_steps = ne03 * ne02 * nrows_per_thread;                                                 \
    uint32_t pi_step = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    க்கு (uint32_t step = 0, spad_idx = 0; step < total_steps && spad_idx < 2; ++step, spad_idx++) {              \
        uint32_t i = ir0 + pi_step;                                                                              \
        மாறிலி dma_addr_t src0_data = octx->src[0]->data + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_queue_தள்ளு(dma_q,                                                                                    \
                       dma_make_data(octx->dst->data,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_size),                     \
                       dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 0);                          \
        dma_queue_தள்ளு(dma_q,                                                                                    \
                       dma_make_data(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_size,                    \
                                     src0_data),                                                                 \
                       vtcm_layout->src0_spad_half_size, src0_row_size, src0_row_size, 1);                       \
        pi_step++;                                                                                               \
        எனில் (pi_step == nrows_per_thread) {                                                                       \
            pi_step = 0;                                                                                         \
            pi02++;                                                                                              \
            எனில் (pi02 == ne02) {                                                                                  \
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
    க்கு (uint32_t step = 0; step < total_steps; ++step) {                                                        \
        வெறுமை * dst_spad = (வெறுமை *) dma_queue_எடு(dma_q).src;                                                     \
        வெறுமை * src_spad = (வெறுமை *) dma_queue_எடு(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_step;                                                                              \
        மாறிலி uintptr_t src1_addr = octx->src[1]->data + i*nb10 + ci11_base*nb11 + ci12_base*nb12;               \
        மாறிலி IDX_TYPE i1 = *(மாறிலி IDX_TYPE *)src1_addr;                                                        \
        மாறிலி bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        மாறிலி uint32_t target_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                 \
        எனில் (valid_i1) {                                                                                          \
            COMPUTE_கோவை;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                  \
        எனில் (valid_i1) {                                                                                          \
            மாறிலி dma_addr_t dst_data = octx->dst->data + target_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_queue_தள்ளு(dma_q,                                                                                \
                           dma_make_data(dst_data, dst_spad),                                                    \
                           dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 1);                      \
        } இல்லையெனில் {                                                                                                 \
            dma_queue_தள்ளு(dma_q,                                                                                \
                           dma_make_data(octx->dst->data, dst_spad),                                             \
                           dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 0);                      \
        }                                                                                                        \
        மாறிலி uint32_t next_step = step + 2;                                                                     \
        எனில் (next_step < total_steps) {                                                                           \
            uint32_t ni = ir0 + pi_step;                                                                         \
            மாறிலி dma_addr_t psrc0_data = octx->src[0]->data + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_queue_தள்ளு(dma_q,                                                                                \
                           dma_make_data(src_spad, psrc0_data),                                                  \
                           vtcm_layout->src0_spad_half_size, src0_row_size, src0_row_size, 1);                   \
            pi_step++;                                                                                           \
            எனில் (pi_step == nrows_per_thread) {                                                                   \
                pi_step = 0;                                                                                     \
                pi02++;                                                                                          \
                எனில் (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_step++;                                                                                               \
        எனில் (ci_step == nrows_per_thread) {                                                                       \
            ci_step = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_base++;                                                                                         \
            எனில் (ci11_base == ne11) {                                                                             \
                ci11_base = 0;                                                                                   \
            }                                                                                                    \
            எனில் (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_base++;                                                                                     \
                எனில் (ci12_base == ne12) {                                                                         \
                    ci12_base = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_queue_flush(dma_q);                                                                                      \
}

SET_ROWS_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (மாறிலி uint8_t *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (மாறிலி uint8_t *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (மாறிலி uint8_t *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (மாறிலி uint8_t *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_row_q8_0_f32(dst_spad, (மாறிலி தசம *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_row_q8_0_f32(dst_spad, (மாறிலி தசம *)src_spad, ne00); })

முழு op_set_rows(கட்டமைப்பு htp_ops_context * octx) {
    மாறிலி கட்டமைப்பு htp_set_rows_kernel_params * kparams = (மாறிலி கட்டமைப்பு htp_set_rows_kernel_params *)octx->kernel_params;
    set_rows_preamble;

    எனில் (octx->src[0]->type != HTP_TYPE_F32) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    எனில் (octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_Q8_0) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    எனில் (htp_tensor_is_extended(octx->src[1])) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    மாறிலி கட்டமைப்பு htp_tensor * dst = octx->dst;
    மாறிலி uint32_t total_tasks    = kparams->total_tasks;

    uint32_t task_start = 0;
    uint32_t tasks      = total_tasks;

    எனில் (octx->ctx->mdev.மொத்தம் > 1) {
        மாறிலி bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_வரி_SIZE - 1)) == 0 && !htp_tensor_is_permuted(dst);
        மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_tasks, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
        task_start = range.start;
        tasks      = range.மொத்தம்;
    }

    எனில் (tasks == 0) {
        திரும்பு HTP_STATUS_OK;
    }

    எனில் (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        திரும்பு HTP_STATUS_INVAL_PARAMS;
    }

    மாறிலி uint32_t n_threads = octx->n_threads;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_block((மாறிலி வெறுமை *)octx->src[1]->data, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    கட்டமைப்பு set_rows_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_start = task_start;
    srctx.tasks = tasks;
    srctx.tasks_per_thread = fastdiv(tasks + n_threads - 1, &octx->n_threads_div);

    htp_set_rows_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, n_threads);
    srctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;

    work_queue_func_t q_func = வெற்று;
    மாறிலி bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    தேர்வு (octx->dst->type) {
        நிலை HTP_TYPE_F32:  q_func = is_i32 ? set_rows_thread_dma_f32_int32_t  : set_rows_thread_dma_f32_int64_t;  நிறுத்து;
        நிலை HTP_TYPE_F16:  q_func = is_i32 ? set_rows_thread_dma_f16_int32_t  : set_rows_thread_dma_f16_int64_t;  நிறுத்து;
        நிலை HTP_TYPE_Q8_0: q_func = is_i32 ? set_rows_thread_dma_q8_0_int32_t : set_rows_thread_dma_q8_0_int64_t; நிறுத்து;
        இயல்பு:            திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    FARF(உயர்வு_2, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_bytes_per_thread * n_threads,
         srctx.vtcm_layout.dst_bytes_per_thread  * n_threads,
         n_threads);

    work_queue_run(octx->ctx->work_queue, q_func, &srctx, n_threads);

    திரும்பு HTP_STATUS_OK;
}
