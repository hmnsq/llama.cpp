#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#شامل <math.h>
#شامل <string.h>

#شامل "dma-queue.h"
#شامل "work-queue.h"
#شامل "hvx-utils.h"
#شامل "hex-utils.h"
#شامل "hvx-copy.h"
#شامل "hvx-quant.h"

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"

#شامل "hex-common.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "htp/set-rows-ops.h"

#تعریف_کرو سیٹ_صفیں_preamble                      \
    ثابت uint32_t ne00 = octx->src[0]->ne[0]; \
    ثابت uint32_t ne01 = octx->src[0]->ne[1]; \
    ثابت uint32_t ne02 = octx->src[0]->ne[2]; \
    ثابت uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    ثابت uint32_t ne10 = octx->src[1]->ne[0]; \
    ثابت uint32_t ne11 = octx->src[1]->ne[1]; \
    ثابت uint32_t ne12 = octx->src[1]->ne[2]; \
    ثابت uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    ثابت uint32_t nb01 = octx->src[0]->nb[1]; \
    ثابت uint32_t nb02 = octx->src[0]->nb[2]; \
    ثابت uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    ثابت uint32_t nb10 = octx->src[1]->nb[0]; \
    ثابت uint32_t nb11 = octx->src[1]->nb[1]; \
    ثابت uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    ثابت uint32_t nb1 = octx->dst->nb[1];     \
    ثابت uint32_t nb2 = octx->dst->nb[2];     \
    ثابت uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    ثابت uint32_t ne0 = octx->dst->ne[0];     \
    ثابت uint32_t ne1 = octx->dst->ne[1];     \
    ثابت uint32_t ne2 = octx->dst->ne[2];     \
    ثابت uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    ثابت uint32_t nr  = ne01;

ساخت سیٹ_صفیں_context {
    ساخت htp_ops_context * octx;
    ثابت ساخت htp_سیٹ_صفیں_kernel_params * kparams;
    ساخت htp_سیٹ_صفیں_vtcm_layout vtcm_layout;
    uint8_t * vtcm_بنیاد;
    uint32_t task_آغاز;
    uint32_t tasks;
    uint32_t tasks_فی_thread;
};

#تعریف_کرو سیٹ_صفیں_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_اظہاریہ)                                                \
ساکن خالی سیٹ_صفیں_thread_dma##TYPE_NAME##_##IDX_TYPE(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی *ڈیٹا) {       \
    ساخت سیٹ_صفیں_context * srctx = (ساخت سیٹ_صفیں_context *)ڈیٹا;                                           \
    ساخت htp_ops_context * octx = srctx->octx;                                                                 \
    ثابت ساخت htp_سیٹ_صفیں_kernel_params * kparams = srctx->kparams;                                          \
    سیٹ_صفیں_preamble;                                                                                           \
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    ثابت uint32_t dr  = srctx->tasks_فی_thread;                                                                \
    ثابت uint32_t ir0 = srctx->task_آغاز + dr * ith;                                                           \
    اگر (ir0 >= srctx->task_آغاز + srctx->tasks) {                                                               \
        واپس;                                                                                                  \
    }                                                                                                            \
    ثابت uint32_t ir1 = MIN(ir0 + dr, srctx->task_آغاز + srctx->tasks);                                        \
    dma_قطار * dma_q = octx->ctx->dma[ith];                                                                     \
    ثابت ساخت htp_سیٹ_صفیں_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = srctx->vtcm_بنیاد + vtcm_layout->off_src0 + ith * vtcm_layout->src0_بائٹیں_فی_thread;   \
    uint8_t * vtcm_dst  = srctx->vtcm_بنیاد + vtcm_layout->off_dst  + ith * vtcm_layout->dst_بائٹیں_فی_thread;    \
    ثابت uint32_t src0_صف_سائز = ne00 * حجم(اعشاری);                                                         \
    ثابت uint32_t dst_صف_سائز  = htp_tensor_get_صف_سائز(octx->dst->type, ne00);                               \
    ثابت uint32_t nrows_فی_thread = ir1 - ir0;                                                                 \
    ثابت uint32_t کل_اقدام = ne03 * ne02 * nrows_فی_thread;                                                 \
    uint32_t pi_قدم = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    برائے (uint32_t قدم = 0, spad_idx = 0; قدم < کل_اقدام && spad_idx < 2; ++قدم, spad_idx++) {              \
        uint32_t i = ir0 + pi_قدم;                                                                              \
        ثابت dma_addr_t src0_ڈیٹا = octx->src[0]->ڈیٹا + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_قطار_ڈالو(dma_q,                                                                                    \
                       dma_make_ڈیٹا(octx->dst->ڈیٹا,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_سائز),                     \
                       dst_صف_سائز, vtcm_layout->dst_spad_half_سائز, dst_صف_سائز, 0);                          \
        dma_قطار_ڈالو(dma_q,                                                                                    \
                       dma_make_ڈیٹا(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_سائز,                    \
                                     src0_ڈیٹا),                                                                 \
                       vtcm_layout->src0_spad_half_سائز, src0_صف_سائز, src0_صف_سائز, 1);                       \
        pi_قدم++;                                                                                               \
        اگر (pi_قدم == nrows_فی_thread) {                                                                       \
            pi_قدم = 0;                                                                                         \
            pi02++;                                                                                              \
            اگر (pi02 == ne02) {                                                                                  \
                pi02 = 0;                                                                                        \
                pi03++;                                                                                          \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    uint32_t ci_قدم = 0;                                                                                        \
    uint32_t ci02 = 0;                                                                                           \
    uint32_t ci03 = 0;                                                                                           \
    uint32_t ci11_بنیاد = 0;                                                                                      \
    uint32_t ci12_بنیاد = 0;                                                                                      \
    برائے (uint32_t قدم = 0; قدم < کل_اقدام; ++قدم) {                                                        \
        خالی * dst_spad = (خالی *) dma_قطار_نکالو(dma_q).src;                                                     \
        خالی * src_spad = (خالی *) dma_قطار_نکالو(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_قدم;                                                                              \
        ثابت uintptr_t src1_addr = octx->src[1]->ڈیٹا + i*nb10 + ci11_بنیاد*nb11 + ci12_بنیاد*nb12;               \
        ثابت IDX_TYPE i1 = *(ثابت IDX_TYPE *)src1_addr;                                                        \
        ثابت bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        ثابت uint32_t ہدف_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, قدم);                                                 \
        اگر (valid_i1) {                                                                                          \
            COMPUTE_اظہاریہ;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, قدم);                                                  \
        اگر (valid_i1) {                                                                                          \
            ثابت dma_addr_t dst_ڈیٹا = octx->dst->ڈیٹا + ہدف_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_قطار_ڈالو(dma_q,                                                                                \
                           dma_make_ڈیٹا(dst_ڈیٹا, dst_spad),                                                    \
                           dst_صف_سائز, vtcm_layout->dst_spad_half_سائز, dst_صف_سائز, 1);                      \
        } ورنہ {                                                                                                 \
            dma_قطار_ڈالو(dma_q,                                                                                \
                           dma_make_ڈیٹا(octx->dst->ڈیٹا, dst_spad),                                             \
                           dst_صف_سائز, vtcm_layout->dst_spad_half_سائز, dst_صف_سائز, 0);                      \
        }                                                                                                        \
        ثابت uint32_t اگلا_قدم = قدم + 2;                                                                     \
        اگر (اگلا_قدم < کل_اقدام) {                                                                           \
            uint32_t ni = ir0 + pi_قدم;                                                                         \
            ثابت dma_addr_t psrc0_ڈیٹا = octx->src[0]->ڈیٹا + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_قطار_ڈالو(dma_q,                                                                                \
                           dma_make_ڈیٹا(src_spad, psrc0_ڈیٹا),                                                  \
                           vtcm_layout->src0_spad_half_سائز, src0_صف_سائز, src0_صف_سائز, 1);                   \
            pi_قدم++;                                                                                           \
            اگر (pi_قدم == nrows_فی_thread) {                                                                   \
                pi_قدم = 0;                                                                                     \
                pi02++;                                                                                          \
                اگر (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_قدم++;                                                                                               \
        اگر (ci_قدم == nrows_فی_thread) {                                                                       \
            ci_قدم = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_بنیاد++;                                                                                         \
            اگر (ci11_بنیاد == ne11) {                                                                             \
                ci11_بنیاد = 0;                                                                                   \
            }                                                                                                    \
            اگر (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_بنیاد++;                                                                                     \
                اگر (ci12_بنیاد == ne12) {                                                                         \
                    ci12_بنیاد = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_قطار_flush(dma_q);                                                                                      \
}

سیٹ_صفیں_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })
سیٹ_صفیں_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })

سیٹ_صفیں_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })
سیٹ_صفیں_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })

سیٹ_صفیں_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_صف_q8_0_f32(dst_spad, (ثابت اعشاری *)src_spad, ne00); })
سیٹ_صفیں_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_صف_q8_0_f32(dst_spad, (ثابت اعشاری *)src_spad, ne00); })

صحیح op_سیٹ_صفیں(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_سیٹ_صفیں_kernel_params * kparams = (ثابت ساخت htp_سیٹ_صفیں_kernel_params *)octx->kernel_params;
    سیٹ_صفیں_preamble;

    اگر (octx->src[0]->type != HTP_TYPE_F32) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    اگر (octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_Q8_0) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    اگر (htp_tensor_is_extended(octx->src[1])) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت ساخت htp_tensor * dst = octx->dst;
    ثابت uint32_t کل_tasks    = kparams->کل_tasks;

    uint32_t task_آغاز = 0;
    uint32_t tasks      = کل_tasks;

    اگر (octx->ctx->mdev.گنتی > 1) {
        ثابت bool can_بانٹو = htp_tensor_mdev_ڈیٹا_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_سطر_سائز - 1)) == 0 && !htp_tensor_is_permuted(dst);
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(کل_tasks, can_بانٹو ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        task_آغاز = range.آغاز;
        tasks      = range.گنتی;
    }

    اگر (tasks == 0) {
        واپس HTP_حالت_OK;
    }

    اگر (!htp_ops_context_سیٹ_n_دھاگے(octx, (uint32_t) kparams->n_دھاگے)) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_بلاک((ثابت خالی *)octx->src[1]->ڈیٹا, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    ساخت سیٹ_صفیں_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_آغاز = task_آغاز;
    srctx.tasks = tasks;
    srctx.tasks_فی_thread = fastdiv(tasks + n_دھاگے - 1, &octx->n_دھاگے_div);

    htp_سیٹ_صفیں_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, n_دھاگے);
    srctx.vtcm_بنیاد = (uint8_t *)octx->ctx->vtcm_بنیاد;

    work_قطار_func_t q_func = تہی;
    ثابت bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    انتخاب (octx->dst->type) {
        حالت HTP_TYPE_F32:  q_func = is_i32 ? سیٹ_صفیں_thread_dma_f32_int32_t  : سیٹ_صفیں_thread_dma_f32_int64_t;  توڑو;
        حالت HTP_TYPE_F16:  q_func = is_i32 ? سیٹ_صفیں_thread_dma_f16_int32_t  : سیٹ_صفیں_thread_dma_f16_int64_t;  توڑو;
        حالت HTP_TYPE_Q8_0: q_func = is_i32 ? سیٹ_صفیں_thread_dma_q8_0_int32_t : سیٹ_صفیں_thread_dma_q8_0_int64_t; توڑو;
        طےشدہ:            واپس HTP_حالت_NO_SUPPORT;
    }

    FARF(بلند_2, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_بائٹیں_فی_thread * n_دھاگے,
         srctx.vtcm_layout.dst_بائٹیں_فی_thread  * n_دھاگے,
         n_دھاگے);

    work_قطار_تسلسل(octx->ctx->work_قطار, q_func, &srctx, n_دھاگے);

    واپس HTP_حالت_OK;
}
