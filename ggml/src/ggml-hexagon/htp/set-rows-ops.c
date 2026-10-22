#لارښوونه clang diagnostic ignored "-Wunused-variable"
#لارښوونه clang diagnostic ignored "-Wunused-function"
#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

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

#تعریف_کړه GGML_COMMON_DECL_C
#ضمن "ggml-common.h"

#ضمن "hex-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "htp/set-rows-ops.h"

#تعریف_کړه سېټ_لیکې_preamble                      \
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

بنية سېټ_لیکې_context {
    بنية htp_ops_context * octx;
    ثابت بنية htp_سېټ_لیکې_kernel_params * kparams;
    بنية htp_سېټ_لیکې_vtcm_layout vtcm_layout;
    uint8_t * vtcm_بنسټ;
    uint32_t task_پیل;
    uint32_t tasks;
    uint32_t tasks_فی_thread;
};

#تعریف_کړه سېټ_لیکې_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_بیان)                                                \
ساكن فراغ سېټ_لیکې_thread_dma##TYPE_NAME##_##IDX_TYPE(بینښه صحيح nth, بینښه صحيح ith, فراغ *معلومات) {       \
    بنية سېټ_لیکې_context * srctx = (بنية سېټ_لیکې_context *)معلومات;                                           \
    بنية htp_ops_context * octx = srctx->octx;                                                                 \
    ثابت بنية htp_سېټ_لیکې_kernel_params * kparams = srctx->kparams;                                          \
    سېټ_لیکې_preamble;                                                                                           \
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    ثابت uint32_t dr  = srctx->tasks_فی_thread;                                                                \
    ثابت uint32_t ir0 = srctx->task_پیل + dr * ith;                                                           \
    إذا (ir0 >= srctx->task_پیل + srctx->tasks) {                                                               \
        أرجع;                                                                                                  \
    }                                                                                                            \
    ثابت uint32_t ir1 = MIN(ir0 + dr, srctx->task_پیل + srctx->tasks);                                        \
    dma_کتار * dma_q = octx->ctx->dma[ith];                                                                     \
    ثابت بنية htp_سېټ_لیکې_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = srctx->vtcm_بنسټ + vtcm_layout->off_src0 + ith * vtcm_layout->src0_بایټونه_فی_thread;   \
    uint8_t * vtcm_dst  = srctx->vtcm_بنسټ + vtcm_layout->off_dst  + ith * vtcm_layout->dst_بایټونه_فی_thread;    \
    ثابت uint32_t src0_لیکه_اندازه = ne00 * حجم(عائم);                                                         \
    ثابت uint32_t dst_لیکه_اندازه  = htp_tensor_get_لیکه_اندازه(octx->dst->type, ne00);                               \
    ثابت uint32_t nrows_فی_thread = ir1 - ir0;                                                                 \
    ثابت uint32_t ټول_ګامونه = ne03 * ne02 * nrows_فی_thread;                                                 \
    uint32_t pi_ګام = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    لكل (uint32_t ګام = 0, spad_idx = 0; ګام < ټول_ګامونه && spad_idx < 2; ++ګام, spad_idx++) {              \
        uint32_t i = ir0 + pi_ګام;                                                                              \
        ثابت dma_addr_t src0_معلومات = octx->src[0]->معلومات + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_کتار_ټېل(dma_q,                                                                                    \
                       dma_make_معلومات(octx->dst->معلومات,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_اندازه),                     \
                       dst_لیکه_اندازه, vtcm_layout->dst_spad_half_اندازه, dst_لیکه_اندازه, 0);                          \
        dma_کتار_ټېل(dma_q,                                                                                    \
                       dma_make_معلومات(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_اندازه,                    \
                                     src0_معلومات),                                                                 \
                       vtcm_layout->src0_spad_half_اندازه, src0_لیکه_اندازه, src0_لیکه_اندازه, 1);                       \
        pi_ګام++;                                                                                               \
        إذا (pi_ګام == nrows_فی_thread) {                                                                       \
            pi_ګام = 0;                                                                                         \
            pi02++;                                                                                              \
            إذا (pi02 == ne02) {                                                                                  \
                pi02 = 0;                                                                                        \
                pi03++;                                                                                          \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    uint32_t ci_ګام = 0;                                                                                        \
    uint32_t ci02 = 0;                                                                                           \
    uint32_t ci03 = 0;                                                                                           \
    uint32_t ci11_بنسټ = 0;                                                                                      \
    uint32_t ci12_بنسټ = 0;                                                                                      \
    لكل (uint32_t ګام = 0; ګام < ټول_ګامونه; ++ګام) {                                                        \
        فراغ * dst_spad = (فراغ *) dma_کتار_راوباسه(dma_q).src;                                                     \
        فراغ * src_spad = (فراغ *) dma_کتار_راوباسه(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_ګام;                                                                              \
        ثابت uintptr_t src1_addr = octx->src[1]->معلومات + i*nb10 + ci11_بنسټ*nb11 + ci12_بنسټ*nb12;               \
        ثابت IDX_TYPE i1 = *(ثابت IDX_TYPE *)src1_addr;                                                        \
        ثابت bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        ثابت uint32_t موخه_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, ګام);                                                 \
        إذا (valid_i1) {                                                                                          \
            COMPUTE_بیان;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, ګام);                                                  \
        إذا (valid_i1) {                                                                                          \
            ثابت dma_addr_t dst_معلومات = octx->dst->معلومات + موخه_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_کتار_ټېل(dma_q,                                                                                \
                           dma_make_معلومات(dst_معلومات, dst_spad),                                                    \
                           dst_لیکه_اندازه, vtcm_layout->dst_spad_half_اندازه, dst_لیکه_اندازه, 1);                      \
        } وإلا {                                                                                                 \
            dma_کتار_ټېل(dma_q,                                                                                \
                           dma_make_معلومات(octx->dst->معلومات, dst_spad),                                             \
                           dst_لیکه_اندازه, vtcm_layout->dst_spad_half_اندازه, dst_لیکه_اندازه, 0);                      \
        }                                                                                                        \
        ثابت uint32_t بل_ګام = ګام + 2;                                                                     \
        إذا (بل_ګام < ټول_ګامونه) {                                                                           \
            uint32_t ni = ir0 + pi_ګام;                                                                         \
            ثابت dma_addr_t psrc0_معلومات = octx->src[0]->معلومات + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_کتار_ټېل(dma_q,                                                                                \
                           dma_make_معلومات(src_spad, psrc0_معلومات),                                                  \
                           vtcm_layout->src0_spad_half_اندازه, src0_لیکه_اندازه, src0_لیکه_اندازه, 1);                   \
            pi_ګام++;                                                                                           \
            إذا (pi_ګام == nrows_فی_thread) {                                                                   \
                pi_ګام = 0;                                                                                     \
                pi02++;                                                                                          \
                إذا (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_ګام++;                                                                                               \
        إذا (ci_ګام == nrows_فی_thread) {                                                                       \
            ci_ګام = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_بنسټ++;                                                                                         \
            إذا (ci11_بنسټ == ne11) {                                                                             \
                ci11_بنسټ = 0;                                                                                   \
            }                                                                                                    \
            إذا (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_بنسټ++;                                                                                     \
                إذا (ci12_بنسټ == ne12) {                                                                         \
                    ci12_بنسټ = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_کتار_flush(dma_q);                                                                                      \
}

سېټ_لیکې_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })
سېټ_لیکې_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })

سېټ_لیکې_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })
سېټ_لیکې_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (ثابت uint8_t *)src_spad, ne00); })

سېټ_لیکې_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_لیکه_q8_0_f32(dst_spad, (ثابت عائم *)src_spad, ne00); })
سېټ_لیکې_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_لیکه_q8_0_f32(dst_spad, (ثابت عائم *)src_spad, ne00); })

صحيح op_سېټ_لیکې(بنية htp_ops_context * octx) {
    ثابت بنية htp_سېټ_لیکې_kernel_params * kparams = (ثابت بنية htp_سېټ_لیکې_kernel_params *)octx->kernel_params;
    سېټ_لیکې_preamble;

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
    ثابت uint32_t ټول_tasks    = kparams->ټول_tasks;

    uint32_t task_پیل = 0;
    uint32_t tasks      = ټول_tasks;

    إذا (octx->ctx->mdev.شمېر > 1) {
        ثابت bool can_وېشل = htp_tensor_mdev_معلومات_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_کرښه_اندازه - 1)) == 0 && !htp_tensor_is_permuted(dst);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(ټول_tasks, can_وېشل ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        task_پیل = range.پیل;
        tasks      = range.شمېر;
    }

    إذا (tasks == 0) {
        أرجع HTP_حالت_OK;
    }

    إذا (!htp_ops_context_سېټ_n_تارونه(octx, (uint32_t) kparams->n_تارونه)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    ثابت uint32_t n_تارونه = octx->n_تارونه;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_بلاک((ثابت فراغ *)octx->src[1]->معلومات, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    بنية سېټ_لیکې_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_پیل = task_پیل;
    srctx.tasks = tasks;
    srctx.tasks_فی_thread = fastdiv(tasks + n_تارونه - 1, &octx->n_تارونه_div);

    htp_سېټ_لیکې_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, n_تارونه);
    srctx.vtcm_بنسټ = (uint8_t *)octx->ctx->vtcm_بنسټ;

    work_کتار_func_t q_func = عدم;
    ثابت bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    بدل (octx->dst->type) {
        حالة HTP_TYPE_F32:  q_func = is_i32 ? سېټ_لیکې_thread_dma_f32_int32_t  : سېټ_لیکې_thread_dma_f32_int64_t;  اكسر;
        حالة HTP_TYPE_F16:  q_func = is_i32 ? سېټ_لیکې_thread_dma_f16_int32_t  : سېټ_لیکې_thread_dma_f16_int64_t;  اكسر;
        حالة HTP_TYPE_Q8_0: q_func = is_i32 ? سېټ_لیکې_thread_dma_q8_0_int32_t : سېټ_لیکې_thread_dma_q8_0_int64_t; اكسر;
        افتراضي:            أرجع HTP_حالت_NO_SUPPORT;
    }

    FARF(لوړ_2, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_بایټونه_فی_thread * n_تارونه,
         srctx.vtcm_layout.dst_بایټونه_فی_thread  * n_تارونه,
         n_تارونه);

    work_کتار_پرلپسې(octx->ctx->work_کتار, q_func, &srctx, n_تارونه);

    أرجع HTP_حالت_OK;
}
