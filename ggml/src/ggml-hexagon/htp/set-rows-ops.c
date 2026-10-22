#رهنمود clang diagnostic ignored "-Wunused-variable"
#رهنمود clang diagnostic ignored "-Wunused-function"
#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

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

#تعریف_کن GGML_COMMON_DECL_C
#ضمن "ggml-common.h"

#ضمن "hex-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "htp/set-rows-ops.h"

#تعریف_کن مجموعه_ردیفها_preamble                      \
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

بنية مجموعه_ردیفها_context {
    بنية htp_ops_context * octx;
    ثابت بنية htp_مجموعه_ردیفها_kernel_params * kparams;
    بنية htp_مجموعه_ردیفها_vtcm_layout vtcm_layout;
    uint8_t * vtcm_پایه;
    uint32_t task_شروع;
    uint32_t tasks;
    uint32_t tasks_در_هر_thread;
};

#تعریف_کن مجموعه_ردیفها_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_عبارت)                                                \
ساكن فراغ مجموعه_ردیفها_thread_dma##TYPE_NAME##_##IDX_TYPE(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ *داده) {       \
    بنية مجموعه_ردیفها_context * srctx = (بنية مجموعه_ردیفها_context *)داده;                                           \
    بنية htp_ops_context * octx = srctx->octx;                                                                 \
    ثابت بنية htp_مجموعه_ردیفها_kernel_params * kparams = srctx->kparams;                                          \
    مجموعه_ردیفها_preamble;                                                                                           \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    ثابت uint32_t dr  = srctx->tasks_در_هر_thread;                                                                \
    ثابت uint32_t ir0 = srctx->task_شروع + dr * ith;                                                           \
    إذا (ir0 >= srctx->task_شروع + srctx->tasks) {                                                               \
        أرجع;                                                                                                  \
    }                                                                                                            \
    ثابت uint32_t ir1 = MIN(ir0 + dr, srctx->task_شروع + srctx->tasks);                                        \
    dma_صف * dma_q = octx->ctx->dma[ith];                                                                     \
    ثابت بنية htp_مجموعه_ردیفها_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = srctx->vtcm_پایه + vtcm_layout->off_src0 + ith * vtcm_layout->src0_بایتها_در_هر_thread;   \
    uint8_t * vtcm_dst  = srctx->vtcm_پایه + vtcm_layout->off_dst  + ith * vtcm_layout->dst_بایتها_در_هر_thread;    \
    ثابت uint32_t src0_ردیف_اندازه = ne00 * حجم(عائم);                                                         \
    ثابت uint32_t dst_ردیف_اندازه  = htp_tensor_get_ردیف_اندازه(octx->dst->type, ne00);                               \
    ثابت uint32_t nrows_در_هر_thread = ir1 - ir0;                                                                 \
    ثابت uint32_t مجموع_گامها = ne03 * ne02 * nrows_در_هر_thread;                                                 \
    uint32_t pi_گام = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    لكل (uint32_t گام = 0, spad_idx = 0; گام < مجموع_گامها && spad_idx < 2; ++گام, spad_idx++) {              \
        uint32_t i = ir0 + pi_گام;                                                                              \
        ثابت dma_addr_t src0_داده = octx->src[0]->داده + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_صف_افزودن(dma_q,                                                                                    \
                       dma_make_داده(octx->dst->داده,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_اندازه),                     \
                       dst_ردیف_اندازه, vtcm_layout->dst_spad_half_اندازه, dst_ردیف_اندازه, 0);                          \
        dma_صف_افزودن(dma_q,                                                                                    \
                       dma_make_داده(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_اندازه,                    \
                                     src0_داده),                                                                 \
                       vtcm_layout->src0_spad_half_اندازه, src0_ردیف_اندازه, src0_ردیف_اندازه, 1);                       \
        pi_گام++;                                                                                               \
        إذا (pi_گام == nrows_در_هر_thread) {                                                                       \
            pi_گام = 0;                                                                                         \
            pi02++;                                                                                              \
            إذا (pi02 == ne02) {                                                                                  \
                pi02 = 0;                                                                                        \
                pi03++;                                                                                          \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    uint32_t ci_گام = 0;                                                                                        \
    uint32_t ci02 = 0;                                                                                           \
    uint32_t ci03 = 0;                                                                                           \
    uint32_t ci11_پایه = 0;                                                                                      \
    uint32_t ci12_پایه = 0;                                                                                      \
    لكل (uint32_t گام = 0; گام < مجموع_گامها; ++گام) {                                                        \
        فراغ * dst_spad = (فراغ *) dma_صف_برداشت(dma_q).src;                                                     \
        فراغ * src_spad = (فراغ *) dma_صف_برداشت(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_گام;                                                                              \
        ثابت uintptr_t src1_addr = octx->src[1]->داده + i*nb10 + ci11_پایه*nb11 + ci12_پایه*nb12;               \
        ثابت IDX_TYPE i1 = *(ثابت IDX_TYPE *)src1_addr;                                                        \
        ثابت bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        ثابت uint32_t هدف_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, گام);                                                 \
        إذا (valid_i1) {                                                                                          \
            COMPUTE_عبارت;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, گام);                                                  \
        إذا (valid_i1) {                                                                                          \
            ثابت dma_addr_t dst_داده = octx->dst->داده + هدف_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_صف_افزودن(dma_q,                                                                                \
                           dma_make_داده(dst_داده, dst_spad),                                                    \
                           dst_ردیف_اندازه, vtcm_layout->dst_spad_half_اندازه, dst_ردیف_اندازه, 1);                      \
        } وإلا {                                                                                                 \
            dma_صف_افزودن(dma_q,                                                                                \
                           dma_make_داده(octx->dst->داده, dst_spad),                                             \
                           dst_ردیف_اندازه, vtcm_layout->dst_spad_half_اندازه, dst_ردیف_اندازه, 0);                      \
        }                                                                                                        \
        ثابت uint32_t بعدی_گام = گام + 2;                                                                     \
        إذا (بعدی_گام < مجموع_گامها) {                                                                           \
            uint32_t ni = ir0 + pi_گام;                                                                         \
            ثابت dma_addr_t psrc0_داده = octx->src[0]->داده + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_صف_افزودن(dma_q,                                                                                \
                           dma_make_داده(src_spad, psrc0_داده),                                                  \
                           vtcm_layout->src0_spad_half_اندازه, src0_ردیف_اندازه, src0_ردیف_اندازه, 1);                   \
            pi_گام++;                                                                                           \
            إذا (pi_گام == nrows_در_هر_thread) {                                                                   \
                pi_گام = 0;                                                                                     \
                pi02++;                                                                                          \
                إذا (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_گام++;                                                                                               \
        إذا (ci_گام == nrows_در_هر_thread) {                                                                       \
            ci_گام = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_پایه++;                                                                                         \
            إذا (ci11_پایه == ne11) {                                                                             \
                ci11_پایه = 0;                                                                                   \
            }                                                                                                    \
            إذا (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_پایه++;                                                                                     \
                إذا (ci12_پایه == ne12) {                                                                         \
                    ci12_پایه = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_صف_flush(dma_q);                                                                                      \
}

مجموعه_ردیفها_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })
مجموعه_ردیفها_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })

مجموعه_ردیفها_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })
مجموعه_ردیفها_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })

مجموعه_ردیفها_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_ردیف_q8_0_f32(dst_spad, (ثابت عائم *)src_spad, ne00); })
مجموعه_ردیفها_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_ردیف_q8_0_f32(dst_spad, (ثابت عائم *)src_spad, ne00); })

صحيح op_مجموعه_ردیفها(بنية htp_ops_context * octx) {
    ثابت بنية htp_مجموعه_ردیفها_kernel_params * kparams = (ثابت بنية htp_مجموعه_ردیفها_kernel_params *)octx->kernel_params;
    مجموعه_ردیفها_preamble;

    إذا (octx->src[0]->type != HTP_TYPE_F32) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_Q8_0) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    إذا (htp_tensor_is_extended(octx->src[1])) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت بنية htp_tensor * dst = octx->dst;
    ثابت uint32_t مجموع_tasks    = kparams->مجموع_tasks;

    uint32_t task_شروع = 0;
    uint32_t tasks      = مجموع_tasks;

    إذا (octx->ctx->mdev.شمار > 1) {
        ثابت bool can_تقسیم = htp_tensor_mdev_داده_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_سطر_اندازه - 1)) == 0 && !htp_tensor_is_permuted(dst);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(مجموع_tasks, can_تقسیم ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        task_شروع = range.شروع;
        tasks      = range.شمار;
    }

    إذا (tasks == 0) {
        أرجع HTP_وضعیت_OK;
    }

    إذا (!htp_ops_context_مجموعه_n_نخها(octx, (uint32_t) kparams->n_نخها)) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    ثابت uint32_t n_نخها = octx->n_نخها;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_بلوک((ثابت فراغ *)octx->src[1]->داده, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    بنية مجموعه_ردیفها_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_شروع = task_شروع;
    srctx.tasks = tasks;
    srctx.tasks_در_هر_thread = fastdiv(tasks + n_نخها - 1, &octx->n_نخها_div);

    htp_مجموعه_ردیفها_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, n_نخها);
    srctx.vtcm_پایه = (uint8_t *)octx->ctx->vtcm_پایه;

    work_صف_func_t q_func = عدم;
    ثابت bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    بدل (octx->dst->type) {
        حالة HTP_TYPE_F32:  q_func = is_i32 ? مجموعه_ردیفها_thread_dma_f32_int32_t  : مجموعه_ردیفها_thread_dma_f32_int64_t;  اكسر;
        حالة HTP_TYPE_F16:  q_func = is_i32 ? مجموعه_ردیفها_thread_dma_f16_int32_t  : مجموعه_ردیفها_thread_dma_f16_int64_t;  اكسر;
        حالة HTP_TYPE_Q8_0: q_func = is_i32 ? مجموعه_ردیفها_thread_dma_q8_0_int32_t : مجموعه_ردیفها_thread_dma_q8_0_int64_t; اكسر;
        افتراضي:            أرجع HTP_وضعیت_NO_SUPPORT;
    }

    FARF(بالا_2, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_بایتها_در_هر_thread * n_نخها,
         srctx.vtcm_layout.dst_بایتها_در_هر_thread  * n_نخها,
         n_نخها);

    work_صف_پیاپی(octx->ctx->work_صف, q_func, &srctx, n_نخها);

    أرجع HTP_وضعیت_OK;
}
