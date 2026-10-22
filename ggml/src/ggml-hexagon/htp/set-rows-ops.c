#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <HAP_perf.h>

#ଅନ୍ତର୍ଭାଵ <math.h>
#ଅନ୍ତର୍ଭାଵ <string.h>

#ଅନ୍ତର୍ଭାଵ "dma-queue.h"
#ଅନ୍ତର୍ଭାଵ "work-queue.h"
#ଅନ୍ତର୍ଭାଵ "hvx-utils.h"
#ଅନ୍ତର୍ଭାଵ "hex-utils.h"
#ଅନ୍ତର୍ଭାଵ "hvx-copy.h"
#ଅନ୍ତର୍ଭାଵ "hvx-quant.h"

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"

#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"
#ଅନ୍ତର୍ଭାଵ "htp/set-rows-ops.h"

#define set_rows_preamble                      \
    ଅଚଳ uint32_t ne00 = octx->src[0]->ne[0]; \
    ଅଚଳ uint32_t ne01 = octx->src[0]->ne[1]; \
    ଅଚଳ uint32_t ne02 = octx->src[0]->ne[2]; \
    ଅଚଳ uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    ଅଚଳ uint32_t ne10 = octx->src[1]->ne[0]; \
    ଅଚଳ uint32_t ne11 = octx->src[1]->ne[1]; \
    ଅଚଳ uint32_t ne12 = octx->src[1]->ne[2]; \
    ଅଚଳ uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    ଅଚଳ uint32_t nb01 = octx->src[0]->nb[1]; \
    ଅଚଳ uint32_t nb02 = octx->src[0]->nb[2]; \
    ଅଚଳ uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    ଅଚଳ uint32_t nb10 = octx->src[1]->nb[0]; \
    ଅଚଳ uint32_t nb11 = octx->src[1]->nb[1]; \
    ଅଚଳ uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    ଅଚଳ uint32_t nb1 = octx->dst->nb[1];     \
    ଅଚଳ uint32_t nb2 = octx->dst->nb[2];     \
    ଅଚଳ uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    ଅଚଳ uint32_t ne0 = octx->dst->ne[0];     \
    ଅଚଳ uint32_t ne1 = octx->dst->ne[1];     \
    ଅଚଳ uint32_t ne2 = octx->dst->ne[2];     \
    ଅଚଳ uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    ଅଚଳ uint32_t nr  = ne01;

ସଂରଚନା set_rows_context {
    ସଂରଚନା htp_ops_context * octx;
    ଅଚଳ ସଂରଚନା htp_set_rows_kernel_params * kparams;
    ସଂରଚନା htp_set_rows_vtcm_layout vtcm_layout;
    uint8_t * vtcm_base;
    uint32_t task_start;
    uint32_t tasks;
    uint32_t tasks_per_thread;
};

#define SET_ROWS_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_ପଦସମଷ୍ଟି)                                                \
ସ୍ଥିର ଶୂନ୍ୟ set_rows_thread_dma_##TYPE_NAME##_##IDX_TYPE(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ *data) {       \
    ସଂରଚନା set_rows_context * srctx = (ସଂରଚନା set_rows_context *)data;                                           \
    ସଂରଚନା htp_ops_context * octx = srctx->octx;                                                                 \
    ଅଚଳ ସଂରଚନା htp_set_rows_kernel_params * kparams = srctx->kparams;                                          \
    set_rows_preamble;                                                                                           \
    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    ଅଚଳ uint32_t dr  = srctx->tasks_per_thread;                                                                \
    ଅଚଳ uint32_t ir0 = srctx->task_start + dr * ith;                                                           \
    ଯଦି (ir0 >= srctx->task_start + srctx->tasks) {                                                               \
        ଫେରନ୍ତୁ;                                                                                                  \
    }                                                                                                            \
    ଅଚଳ uint32_t ir1 = MIN(ir0 + dr, srctx->task_start + srctx->tasks);                                        \
    dma_queue * dma_q = octx->ctx->dma[ith];                                                                     \
    ଅଚଳ ସଂରଚନା htp_set_rows_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = srctx->vtcm_base + vtcm_layout->off_src0 + ith * vtcm_layout->src0_bytes_per_thread;   \
    uint8_t * vtcm_dst  = srctx->vtcm_base + vtcm_layout->off_dst  + ith * vtcm_layout->dst_bytes_per_thread;    \
    ଅଚଳ uint32_t src0_row_size = ne00 * ଆକାର(ଦଶମିକ);                                                         \
    ଅଚଳ uint32_t dst_row_size  = htp_tensor_get_row_size(octx->dst->type, ne00);                               \
    ଅଚଳ uint32_t nrows_per_thread = ir1 - ir0;                                                                 \
    ଅଚଳ uint32_t total_steps = ne03 * ne02 * nrows_per_thread;                                                 \
    uint32_t pi_step = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    ପାଇଁ (uint32_t step = 0, spad_idx = 0; step < total_steps && spad_idx < 2; ++step, spad_idx++) {              \
        uint32_t i = ir0 + pi_step;                                                                              \
        ଅଚଳ dma_addr_t src0_data = octx->src[0]->data + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_queue_ଠେଲ(dma_q,                                                                                    \
                       dma_make_data(octx->dst->data,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_size),                     \
                       dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 0);                          \
        dma_queue_ଠେଲ(dma_q,                                                                                    \
                       dma_make_data(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_size,                    \
                                     src0_data),                                                                 \
                       vtcm_layout->src0_spad_half_size, src0_row_size, src0_row_size, 1);                       \
        pi_step++;                                                                                               \
        ଯଦି (pi_step == nrows_per_thread) {                                                                       \
            pi_step = 0;                                                                                         \
            pi02++;                                                                                              \
            ଯଦି (pi02 == ne02) {                                                                                  \
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
    ପାଇଁ (uint32_t step = 0; step < total_steps; ++step) {                                                        \
        ଶୂନ୍ୟ * dst_spad = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).src;                                                     \
        ଶୂନ୍ୟ * src_spad = (ଶୂନ୍ୟ *) dma_queue_କାଢ଼(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_step;                                                                              \
        ଅଚଳ uintptr_t src1_addr = octx->src[1]->data + i*nb10 + ci11_base*nb11 + ci12_base*nb12;               \
        ଅଚଳ IDX_TYPE i1 = *(ଅଚଳ IDX_TYPE *)src1_addr;                                                        \
        ଅଚଳ bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        ଅଚଳ uint32_t target_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                 \
        ଯଦି (valid_i1) {                                                                                          \
            COMPUTE_ପଦସମଷ୍ଟି;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, step);                                                  \
        ଯଦି (valid_i1) {                                                                                          \
            ଅଚଳ dma_addr_t dst_data = octx->dst->data + target_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_queue_ଠେଲ(dma_q,                                                                                \
                           dma_make_data(dst_data, dst_spad),                                                    \
                           dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 1);                      \
        } ଅନ୍ୟଥା {                                                                                                 \
            dma_queue_ଠେଲ(dma_q,                                                                                \
                           dma_make_data(octx->dst->data, dst_spad),                                             \
                           dst_row_size, vtcm_layout->dst_spad_half_size, dst_row_size, 0);                      \
        }                                                                                                        \
        ଅଚଳ uint32_t next_step = step + 2;                                                                     \
        ଯଦି (next_step < total_steps) {                                                                           \
            uint32_t ni = ir0 + pi_step;                                                                         \
            ଅଚଳ dma_addr_t psrc0_data = octx->src[0]->data + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_queue_ଠେଲ(dma_q,                                                                                \
                           dma_make_data(src_spad, psrc0_data),                                                  \
                           vtcm_layout->src0_spad_half_size, src0_row_size, src0_row_size, 1);                   \
            pi_step++;                                                                                           \
            ଯଦି (pi_step == nrows_per_thread) {                                                                   \
                pi_step = 0;                                                                                     \
                pi02++;                                                                                          \
                ଯଦି (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_step++;                                                                                               \
        ଯଦି (ci_step == nrows_per_thread) {                                                                       \
            ci_step = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_base++;                                                                                         \
            ଯଦି (ci11_base == ne11) {                                                                             \
                ci11_base = 0;                                                                                   \
            }                                                                                                    \
            ଯଦି (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_base++;                                                                                     \
                ଯଦି (ci12_base == ne12) {                                                                         \
                    ci12_base = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_queue_flush(dma_q);                                                                                      \
}

SET_ROWS_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ଅଚଳ uint8_t *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ଅଚଳ uint8_t *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ଅଚଳ uint8_t *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ଅଚଳ uint8_t *)src_spad, ne00); })

SET_ROWS_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_row_q8_0_f32(dst_spad, (ଅଚଳ ଦଶମିକ *)src_spad, ne00); })
SET_ROWS_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_row_q8_0_f32(dst_spad, (ଅଚଳ ଦଶମିକ *)src_spad, ne00); })

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_set_rows(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_set_rows_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_set_rows_kernel_params *)octx->kernel_params;
    set_rows_preamble;

    ଯଦି (octx->src[0]->type != HTP_TYPE_F32) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଯଦି (octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_Q8_0) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଯଦି (htp_tensor_is_extended(octx->src[1])) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଅଚଳ ସଂରଚନା htp_tensor * dst = octx->dst;
    ଅଚଳ uint32_t total_tasks    = kparams->total_tasks;

    uint32_t task_start = 0;
    uint32_t tasks      = total_tasks;

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        ଅଚଳ bool can_split = htp_tensor_mdev_data_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_ଧାଡ଼ି_SIZE - 1)) == 0 && !htp_tensor_is_permuted(dst);
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_tasks, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        task_start = range.start;
        tasks      = range.ଗଣନା_2;
    }

    ଯଦି (tasks == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଯଦି (!htp_ops_context_set_n_threads(octx, (uint32_t) kparams->n_threads)) {
        ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
    }

    ଅଚଳ uint32_t n_threads = octx->n_threads;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_block((ଅଚଳ ଶୂନ୍ୟ *)octx->src[1]->data, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    ସଂରଚନା set_rows_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_start = task_start;
    srctx.tasks = tasks;
    srctx.tasks_per_thread = fastdiv(tasks + n_threads - 1, &octx->n_threads_div);

    htp_set_rows_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, n_threads);
    srctx.vtcm_base = (uint8_t *)octx->ctx->vtcm_base;

    work_queue_func_t q_func = ଖାଲି;
    ଅଚଳ bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    ବାଛ (octx->dst->type) {
        ସ୍ଥିତି HTP_TYPE_F32:  q_func = is_i32 ? set_rows_thread_dma_f32_int32_t  : set_rows_thread_dma_f32_int64_t;  ଭାଙ୍ଗ;
        ସ୍ଥିତି HTP_TYPE_F16:  q_func = is_i32 ? set_rows_thread_dma_f16_int32_t  : set_rows_thread_dma_f16_int64_t;  ଭାଙ୍ଗ;
        ସ୍ଥିତି HTP_TYPE_Q8_0: q_func = is_i32 ? set_rows_thread_dma_q8_0_int32_t : set_rows_thread_dma_q8_0_int64_t; ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:            ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    FARF(ଉପର_2, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_bytes_per_thread * n_threads,
         srctx.vtcm_layout.dst_bytes_per_thread  * n_threads,
         n_threads);

    work_queue_run(octx->ctx->work_queue, q_func, &srctx, n_threads);

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}
