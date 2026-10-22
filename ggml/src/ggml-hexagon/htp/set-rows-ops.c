#هدايت clang diagnostic ignored "-Wunused-variable"
#هدايت clang diagnostic ignored "-Wunused-function"
#هدايت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <math.h>
#ضمن <string.h>

#ضمن "dma-queue.h"
#ضمن "work-queue.h"
#ضمن "hvx-utils.h"
#ضمن "hex-utils.h"
#ضمن "hvx-copy.h"
#ضمن "hvx-quant.h"

#وصف_ڪريو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"

#ضمن "hex-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "htp/set-rows-ops.h"

#وصف_ڪريو سيٽ_صفون_preamble                      \
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

بنية سيٽ_صفون_context {
    بنية htp_ops_context * octx;
    ثابت بنية htp_سيٽ_صفون_kernel_params * kparams;
    بنية htp_سيٽ_صفون_vtcm_layout vtcm_layout;
    uint8_t * vtcm_بنياد;
    uint32_t task_شروعات;
    uint32_t tasks;
    uint32_t tasks_في_thread;
};

#وصف_ڪريو سيٽ_صفون_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_اظهار)                                                \
ساكن فراغ سيٽ_صفون_thread_dma##TYPE_NAME##_##IDX_TYPE(بنانشان صحيح nth, بنانشان صحيح ith, فراغ *معلومات) {       \
    بنية سيٽ_صفون_context * srctx = (بنية سيٽ_صفون_context *)معلومات;                                           \
    بنية htp_ops_context * octx = srctx->octx;                                                                 \
    ثابت بنية htp_سيٽ_صفون_kernel_params * kparams = srctx->kparams;                                          \
    سيٽ_صفون_preamble;                                                                                           \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    ثابت uint32_t dr  = srctx->tasks_في_thread;                                                                \
    ثابت uint32_t ir0 = srctx->task_شروعات + dr * ith;                                                           \
    إذا (ir0 >= srctx->task_شروعات + srctx->tasks) {                                                               \
        أرجع;                                                                                                  \
    }                                                                                                            \
    ثابت uint32_t ir1 = MIN(ir0 + dr, srctx->task_شروعات + srctx->tasks);                                        \
    dma_قطار * dma_q = octx->ctx->dma[ith];                                                                     \
    ثابت بنية htp_سيٽ_صفون_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = srctx->vtcm_بنياد + vtcm_layout->off_src0 + ith * vtcm_layout->src0_بائيٽون_في_thread;   \
    uint8_t * vtcm_dst  = srctx->vtcm_بنياد + vtcm_layout->off_dst  + ith * vtcm_layout->dst_بائيٽون_في_thread;    \
    ثابت uint32_t src0_صف_سائيز = ne00 * حجم(عائم);                                                         \
    ثابت uint32_t dst_صف_سائيز  = htp_tensor_get_صف_سائيز(octx->dst->type, ne00);                               \
    ثابت uint32_t nrows_في_thread = ir1 - ir0;                                                                 \
    ثابت uint32_t ڪل_قدمن = ne03 * ne02 * nrows_في_thread;                                                 \
    uint32_t pi_قدم = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    لكل (uint32_t قدم = 0, spad_idx = 0; قدم < ڪل_قدمن && spad_idx < 2; ++قدم, spad_idx++) {              \
        uint32_t i = ir0 + pi_قدم;                                                                              \
        ثابت dma_addr_t src0_معلومات = octx->src[0]->معلومات + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_قطار_ڌڪو(dma_q,                                                                                    \
                       dma_make_معلومات(octx->dst->معلومات,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_سائيز),                     \
                       dst_صف_سائيز, vtcm_layout->dst_spad_half_سائيز, dst_صف_سائيز, 0);                          \
        dma_قطار_ڌڪو(dma_q,                                                                                    \
                       dma_make_معلومات(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_سائيز,                    \
                                     src0_معلومات),                                                                 \
                       vtcm_layout->src0_spad_half_سائيز, src0_صف_سائيز, src0_صف_سائيز, 1);                       \
        pi_قدم++;                                                                                               \
        إذا (pi_قدم == nrows_في_thread) {                                                                       \
            pi_قدم = 0;                                                                                         \
            pi02++;                                                                                              \
            إذا (pi02 == ne02) {                                                                                  \
                pi02 = 0;                                                                                        \
                pi03++;                                                                                          \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    uint32_t ci_قدم = 0;                                                                                        \
    uint32_t ci02 = 0;                                                                                           \
    uint32_t ci03 = 0;                                                                                           \
    uint32_t ci11_بنياد = 0;                                                                                      \
    uint32_t ci12_بنياد = 0;                                                                                      \
    لكل (uint32_t قدم = 0; قدم < ڪل_قدمن; ++قدم) {                                                        \
        فراغ * dst_spad = (فراغ *) dma_قطار_ڪڍو(dma_q).src;                                                     \
        فراغ * src_spad = (فراغ *) dma_قطار_ڪڍو(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_قدم;                                                                              \
        ثابت uintptr_t src1_addr = octx->src[1]->معلومات + i*nb10 + ci11_بنياد*nb11 + ci12_بنياد*nb12;               \
        ثابت IDX_TYPE i1 = *(ثابت IDX_TYPE *)src1_addr;                                                        \
        ثابت bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        ثابت uint32_t هدف_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, قدم);                                                 \
        إذا (valid_i1) {                                                                                          \
            COMPUTE_اظهار;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, قدم);                                                  \
        إذا (valid_i1) {                                                                                          \
            ثابت dma_addr_t dst_معلومات = octx->dst->معلومات + هدف_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_قطار_ڌڪو(dma_q,                                                                                \
                           dma_make_معلومات(dst_معلومات, dst_spad),                                                    \
                           dst_صف_سائيز, vtcm_layout->dst_spad_half_سائيز, dst_صف_سائيز, 1);                      \
        } وإلا {                                                                                                 \
            dma_قطار_ڌڪو(dma_q,                                                                                \
                           dma_make_معلومات(octx->dst->معلومات, dst_spad),                                             \
                           dst_صف_سائيز, vtcm_layout->dst_spad_half_سائيز, dst_صف_سائيز, 0);                      \
        }                                                                                                        \
        ثابت uint32_t ايندڙ_قدم = قدم + 2;                                                                     \
        إذا (ايندڙ_قدم < ڪل_قدمن) {                                                                           \
            uint32_t ni = ir0 + pi_قدم;                                                                         \
            ثابت dma_addr_t psrc0_معلومات = octx->src[0]->معلومات + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_قطار_ڌڪو(dma_q,                                                                                \
                           dma_make_معلومات(src_spad, psrc0_معلومات),                                                  \
                           vtcm_layout->src0_spad_half_سائيز, src0_صف_سائيز, src0_صف_سائيز, 1);                   \
            pi_قدم++;                                                                                           \
            إذا (pi_قدم == nrows_في_thread) {                                                                   \
                pi_قدم = 0;                                                                                     \
                pi02++;                                                                                          \
                إذا (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_قدم++;                                                                                               \
        إذا (ci_قدم == nrows_في_thread) {                                                                       \
            ci_قدم = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_بنياد++;                                                                                         \
            إذا (ci11_بنياد == ne11) {                                                                             \
                ci11_بنياد = 0;                                                                                   \
            }                                                                                                    \
            إذا (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_بنياد++;                                                                                     \
                إذا (ci12_بنياد == ne12) {                                                                         \
                    ci12_بنياد = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_قطار_flush(dma_q);                                                                                      \
}

سيٽ_صفون_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })
سيٽ_صفون_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })

سيٽ_صفون_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })
سيٽ_صفون_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })

سيٽ_صفون_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_صف_q8_0_f32(dst_spad, (ثابت عائم *)src_spad, ne00); })
سيٽ_صفون_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_صف_q8_0_f32(dst_spad, (ثابت عائم *)src_spad, ne00); })

صحيح op_سيٽ_صفون(بنية htp_ops_context * octx) {
    ثابت بنية htp_سيٽ_صفون_kernel_params * kparams = (ثابت بنية htp_سيٽ_صفون_kernel_params *)octx->kernel_params;
    سيٽ_صفون_preamble;

    إذا (octx->src[0]->type != HTP_TYPE_F32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_Q8_0) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(octx->src[1])) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت بنية htp_tensor * dst = octx->dst;
    ثابت uint32_t ڪل_tasks    = kparams->ڪل_tasks;

    uint32_t task_شروعات = 0;
    uint32_t tasks      = ڪل_tasks;

    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        ثابت bool can_ورهايو = htp_tensor_mdev_معلومات_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_سٽ_سائيز - 1)) == 0 && !htp_tensor_is_permuted(dst);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(ڪل_tasks, can_ورهايو ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        task_شروعات = range.شروعات;
        tasks      = range.ڳڻپ;
    }

    إذا (tasks == 0) {
        أرجع HTP_حالت_OK;
    }

    إذا (!htp_ops_context_سيٽ_n_ڌاڳا(octx, (uint32_t) kparams->n_ڌاڳا)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t n_ڌاڳا = octx->n_ڌاڳا;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_بلاڪ((ثابت فراغ *)octx->src[1]->معلومات, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    بنية سيٽ_صفون_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_شروعات = task_شروعات;
    srctx.tasks = tasks;
    srctx.tasks_في_thread = fastdiv(tasks + n_ڌاڳا - 1, &octx->n_ڌاڳا_div);

    htp_سيٽ_صفون_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, n_ڌاڳا);
    srctx.vtcm_بنياد = (uint8_t *)octx->ctx->vtcm_بنياد;

    work_قطار_func_t q_func = عدم;
    ثابت bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    بدل (octx->dst->type) {
        حالة HTP_TYPE_F32:  q_func = is_i32 ? سيٽ_صفون_thread_dma_f32_int32_t  : سيٽ_صفون_thread_dma_f32_int64_t;  اكسر;
        حالة HTP_TYPE_F16:  q_func = is_i32 ? سيٽ_صفون_thread_dma_f16_int32_t  : سيٽ_صفون_thread_dma_f16_int64_t;  اكسر;
        حالة HTP_TYPE_Q8_0: q_func = is_i32 ? سيٽ_صفون_thread_dma_q8_0_int32_t : سيٽ_صفون_thread_dma_q8_0_int64_t; اكسر;
        افتراضي:            أرجع HTP_حالت_NO_SUPPORT;
    }

    FARF(مٿاهون_2, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_بائيٽون_في_thread * n_ڌاڳا,
         srctx.vtcm_layout.dst_بائيٽون_في_thread  * n_ڌاڳا,
         n_ڌاڳا);

    work_قطار_لاڳيتو(octx->ctx->work_قطار, q_func, &srctx, n_ڌاڳا);

    أرجع HTP_حالت_OK;
}
