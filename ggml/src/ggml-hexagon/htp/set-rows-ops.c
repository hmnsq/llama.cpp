#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>

#అన్తర్భావ <math.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "work-queue.h"
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "hex-utils.h"
#అన్తర్భావ "hvx-copy.h"
#అన్తర్భావ "hvx-quant.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"

#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "htp/set-rows-ops.h"

#నిర్వచించు సమితి_అడ్డువరుసలు_preamble                      \
    స్థిరరాశి uint32_t ne00 = octx->src[0]->ne[0]; \
    స్థిరరాశి uint32_t ne01 = octx->src[0]->ne[1]; \
    స్థిరరాశి uint32_t ne02 = octx->src[0]->ne[2]; \
    స్థిరరాశి uint32_t ne03 = octx->src[0]->ne[3]; \
                                               \
    స్థిరరాశి uint32_t ne10 = octx->src[1]->ne[0]; \
    స్థిరరాశి uint32_t ne11 = octx->src[1]->ne[1]; \
    స్థిరరాశి uint32_t ne12 = octx->src[1]->ne[2]; \
    స్థిరరాశి uint32_t ne13 = octx->src[1]->ne[3]; \
                                               \
    స్థిరరాశి uint32_t nb01 = octx->src[0]->nb[1]; \
    స్థిరరాశి uint32_t nb02 = octx->src[0]->nb[2]; \
    స్థిరరాశి uint32_t nb03 = octx->src[0]->nb[3]; \
                                               \
    స్థిరరాశి uint32_t nb10 = octx->src[1]->nb[0]; \
    స్థిరరాశి uint32_t nb11 = octx->src[1]->nb[1]; \
    స్థిరరాశి uint32_t nb12 = octx->src[1]->nb[2]; \
                                               \
    స్థిరరాశి uint32_t nb1 = octx->dst->nb[1];     \
    స్థిరరాశి uint32_t nb2 = octx->dst->nb[2];     \
    స్థిరరాశి uint32_t nb3 = octx->dst->nb[3];     \
                                               \
    స్థిరరాశి uint32_t ne0 = octx->dst->ne[0];     \
    స్థిరరాశి uint32_t ne1 = octx->dst->ne[1];     \
    స్థిరరాశి uint32_t ne2 = octx->dst->ne[2];     \
    స్థిరరాశి uint32_t ne3 = octx->dst->ne[3];     \
                                               \
    స్థిరరాశి uint32_t nr  = ne01;

నిర్మాణం సమితి_అడ్డువరుసలు_context {
    నిర్మాణం htp_ops_context * octx;
    స్థిరరాశి నిర్మాణం htp_సమితి_అడ్డువరుసలు_kernel_params * kparams;
    నిర్మాణం htp_సమితి_అడ్డువరుసలు_vtcm_layout vtcm_layout;
    uint8_t * vtcm_పునాది;
    uint32_t task_ప్రారంభం;
    uint32_t tasks;
    uint32_t tasks_చొప్పున_thread;
};

#నిర్వచించు సమితి_అడ్డువరుసలు_THREAD_DMA_FN(TYPE_NAME, IDX_TYPE, COMPUTE_సమాసం)                                                \
స్థిర శూన్య సమితి_అడ్డువరుసలు_thread_dma##TYPE_NAME##_##IDX_TYPE(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య *దత్తాంశం) {       \
    నిర్మాణం సమితి_అడ్డువరుసలు_context * srctx = (నిర్మాణం సమితి_అడ్డువరుసలు_context *)దత్తాంశం;                                           \
    నిర్మాణం htp_ops_context * octx = srctx->octx;                                                                 \
    స్థిరరాశి నిర్మాణం htp_సమితి_అడ్డువరుసలు_kernel_params * kparams = srctx->kparams;                                          \
    సమితి_అడ్డువరుసలు_preamble;                                                                                           \
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];                                                       \
    స్థిరరాశి uint32_t dr  = srctx->tasks_చొప్పున_thread;                                                                \
    స్థిరరాశి uint32_t ir0 = srctx->task_ప్రారంభం + dr * ith;                                                           \
    ఒకవేళ (ir0 >= srctx->task_ప్రారంభం + srctx->tasks) {                                                               \
        తిరిగి;                                                                                                  \
    }                                                                                                            \
    స్థిరరాశి uint32_t ir1 = MIN(ir0 + dr, srctx->task_ప్రారంభం + srctx->tasks);                                        \
    dma_వరుస * dma_q = octx->ctx->dma[ith];                                                                     \
    స్థిరరాశి నిర్మాణం htp_సమితి_అడ్డువరుసలు_vtcm_layout * vtcm_layout = &srctx->vtcm_layout;                                   \
    uint8_t * vtcm_src0 = srctx->vtcm_పునాది + vtcm_layout->off_src0 + ith * vtcm_layout->src0_బైట్లు_చొప్పున_thread;   \
    uint8_t * vtcm_dst  = srctx->vtcm_పునాది + vtcm_layout->off_dst  + ith * vtcm_layout->dst_బైట్లు_చొప్పున_thread;    \
    స్థిరరాశి uint32_t src0_అడ్డువరుస_కొలత = ne00 * పరిమాణం(దశాంశం);                                                         \
    స్థిరరాశి uint32_t dst_అడ్డువరుస_కొలత  = htp_tensor_get_అడ్డువరుస_కొలత(octx->dst->type, ne00);                               \
    స్థిరరాశి uint32_t nrows_చొప్పున_thread = ir1 - ir0;                                                                 \
    స్థిరరాశి uint32_t మొత్తం_అడుగులు = ne03 * ne02 * nrows_చొప్పున_thread;                                                 \
    uint32_t pi_అడుగు = 0;                                                                                        \
    uint32_t pi02 = 0;                                                                                           \
    uint32_t pi03 = 0;                                                                                           \
    కొరకు (uint32_t అడుగు = 0, spad_idx = 0; అడుగు < మొత్తం_అడుగులు && spad_idx < 2; ++అడుగు, spad_idx++) {              \
        uint32_t i = ir0 + pi_అడుగు;                                                                              \
        స్థిరరాశి dma_addr_t src0_దత్తాంశం = octx->src[0]->దత్తాంశం + i*nb01 + pi02*nb02 + pi03*nb03;                        \
        dma_వరుస_నెట్టు(dma_q,                                                                                    \
                       dma_make_దత్తాంశం(octx->dst->దత్తాంశం,                                                            \
                                     vtcm_dst + spad_idx * vtcm_layout->dst_spad_half_కొలత),                     \
                       dst_అడ్డువరుస_కొలత, vtcm_layout->dst_spad_half_కొలత, dst_అడ్డువరుస_కొలత, 0);                          \
        dma_వరుస_నెట్టు(dma_q,                                                                                    \
                       dma_make_దత్తాంశం(vtcm_src0 + spad_idx * vtcm_layout->src0_spad_half_కొలత,                    \
                                     src0_దత్తాంశం),                                                                 \
                       vtcm_layout->src0_spad_half_కొలత, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 1);                       \
        pi_అడుగు++;                                                                                               \
        ఒకవేళ (pi_అడుగు == nrows_చొప్పున_thread) {                                                                       \
            pi_అడుగు = 0;                                                                                         \
            pi02++;                                                                                              \
            ఒకవేళ (pi02 == ne02) {                                                                                  \
                pi02 = 0;                                                                                        \
                pi03++;                                                                                          \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    uint32_t ci_అడుగు = 0;                                                                                        \
    uint32_t ci02 = 0;                                                                                           \
    uint32_t ci03 = 0;                                                                                           \
    uint32_t ci11_పునాది = 0;                                                                                      \
    uint32_t ci12_పునాది = 0;                                                                                      \
    కొరకు (uint32_t అడుగు = 0; అడుగు < మొత్తం_అడుగులు; ++అడుగు) {                                                        \
        శూన్య * dst_spad = (శూన్య *) dma_వరుస_తీయి(dma_q).src;                                                     \
        శూన్య * src_spad = (శూన్య *) dma_వరుస_తీయి(dma_q).dst;                                                     \
        uint32_t i = ir0 + ci_అడుగు;                                                                              \
        స్థిరరాశి uintptr_t src1_addr = octx->src[1]->దత్తాంశం + i*nb10 + ci11_పునాది*nb11 + ci12_పునాది*nb12;               \
        స్థిరరాశి IDX_TYPE i1 = *(స్థిరరాశి IDX_TYPE *)src1_addr;                                                        \
        స్థిరరాశి bool valid_i1 = ((uint64_t)i1 < (uint64_t)ne1);                                                    \
        స్థిరరాశి uint32_t లక్ష్యం_i1 = (uint32_t)i1;                                                                 \
        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, అడుగు);                                                 \
        ఒకవేళ (valid_i1) {                                                                                          \
            COMPUTE_సమాసం;                                                                                        \
        }                                                                                                        \
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, అడుగు);                                                  \
        ఒకవేళ (valid_i1) {                                                                                          \
            స్థిరరాశి dma_addr_t dst_దత్తాంశం = octx->dst->దత్తాంశం + లక్ష్యం_i1*nb1 + ci02*nb2 + ci03*nb3;                   \
            dma_వరుస_నెట్టు(dma_q,                                                                                \
                           dma_make_దత్తాంశం(dst_దత్తాంశం, dst_spad),                                                    \
                           dst_అడ్డువరుస_కొలత, vtcm_layout->dst_spad_half_కొలత, dst_అడ్డువరుస_కొలత, 1);                      \
        } లేకపోతే {                                                                                                 \
            dma_వరుస_నెట్టు(dma_q,                                                                                \
                           dma_make_దత్తాంశం(octx->dst->దత్తాంశం, dst_spad),                                             \
                           dst_అడ్డువరుస_కొలత, vtcm_layout->dst_spad_half_కొలత, dst_అడ్డువరుస_కొలత, 0);                      \
        }                                                                                                        \
        స్థిరరాశి uint32_t తదుపరి_అడుగు = అడుగు + 2;                                                                     \
        ఒకవేళ (తదుపరి_అడుగు < మొత్తం_అడుగులు) {                                                                           \
            uint32_t ni = ir0 + pi_అడుగు;                                                                         \
            స్థిరరాశి dma_addr_t psrc0_దత్తాంశం = octx->src[0]->దత్తాంశం + ni*nb01 + pi02*nb02 + pi03*nb03;                  \
            dma_వరుస_నెట్టు(dma_q,                                                                                \
                           dma_make_దత్తాంశం(src_spad, psrc0_దత్తాంశం),                                                  \
                           vtcm_layout->src0_spad_half_కొలత, src0_అడ్డువరుస_కొలత, src0_అడ్డువరుస_కొలత, 1);                   \
            pi_అడుగు++;                                                                                           \
            ఒకవేళ (pi_అడుగు == nrows_చొప్పున_thread) {                                                                   \
                pi_అడుగు = 0;                                                                                     \
                pi02++;                                                                                          \
                ఒకవేళ (pi02 == ne02) {                                                                              \
                    pi02 = 0;                                                                                    \
                    pi03++;                                                                                      \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
        ci_అడుగు++;                                                                                               \
        ఒకవేళ (ci_అడుగు == nrows_చొప్పున_thread) {                                                                       \
            ci_అడుగు = 0;                                                                                         \
            ci02++;                                                                                              \
            ci11_పునాది++;                                                                                         \
            ఒకవేళ (ci11_పునాది == ne11) {                                                                             \
                ci11_పునాది = 0;                                                                                   \
            }                                                                                                    \
            ఒకవేళ (ci02 == ne02) {                                                                                  \
                ci02 = 0;                                                                                        \
                ci03++;                                                                                          \
                ci12_పునాది++;                                                                                     \
                ఒకవేళ (ci12_పునాది == ne12) {                                                                         \
                    ci12_పునాది = 0;                                                                               \
                }                                                                                                \
            }                                                                                                    \
        }                                                                                                        \
    }                                                                                                            \
    dma_వరుస_flush(dma_q);                                                                                      \
}

సమితి_అడ్డువరుసలు_THREAD_DMA_FN(f32,  int32_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (స్థిరరాశి uint8_t *)src_spad, ne00); })
సమితి_అడ్డువరుసలు_THREAD_DMA_FN(f32,  int64_t, { hvx_copy_f32_uu((uint8_t *)dst_spad, (స్థిరరాశి uint8_t *)src_spad, ne00); })

సమితి_అడ్డువరుసలు_THREAD_DMA_FN(f16,  int32_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (స్థిరరాశి uint8_t *)src_spad, ne00); })
సమితి_అడ్డువరుసలు_THREAD_DMA_FN(f16,  int64_t, { hvx_copy_f16_f32_uu((uint8_t *)dst_spad, (స్థిరరాశి uint8_t *)src_spad, ne00); })

సమితి_అడ్డువరుసలు_THREAD_DMA_FN(q8_0, int32_t, { hvx_quantize_అడ్డువరుస_q8_0_f32(dst_spad, (స్థిరరాశి దశాంశం *)src_spad, ne00); })
సమితి_అడ్డువరుసలు_THREAD_DMA_FN(q8_0, int64_t, { hvx_quantize_అడ్డువరుస_q8_0_f32(dst_spad, (స్థిరరాశి దశాంశం *)src_spad, ne00); })

పూర్ణాంకం op_సమితి_అడ్డువరుసలు(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_సమితి_అడ్డువరుసలు_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_సమితి_అడ్డువరుసలు_kernel_params *)octx->kernel_params;
    సమితి_అడ్డువరుసలు_preamble;

    ఒకవేళ (octx->src[0]->type != HTP_TYPE_F32) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (octx->dst->type != HTP_TYPE_F32 && octx->dst->type != HTP_TYPE_F16 && octx->dst->type != HTP_TYPE_Q8_0) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    ఒకవేళ (htp_tensor_is_extended(octx->src[1])) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి నిర్మాణం htp_tensor * dst = octx->dst;
    స్థిరరాశి uint32_t మొత్తం_tasks    = kparams->మొత్తం_tasks;

    uint32_t task_ప్రారంభం = 0;
    uint32_t tasks      = మొత్తం_tasks;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        స్థిరరాశి bool can_విభజించు = htp_tensor_mdev_దత్తాంశం_aligned(dst) && (dst->nb[1] & (HTP_TENSOR_MDEV_పంక్తి_కొలత - 1)) == 0 && !htp_tensor_is_permuted(dst);
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(మొత్తం_tasks, can_విభజించు ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        task_ప్రారంభం = range.ప్రారంభం;
        tasks      = range.లెక్క;
    }

    ఒకవేళ (tasks == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, (uint32_t) kparams->n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

    // l2fetch the src1 (indices) tensor in the main thread
    hex_l2fetch_ఖండం((స్థిరరాశి శూన్య *)octx->src[1]->దత్తాంశం, octx->src[1]->ne[3] * octx->src[1]->nb[3]);

    నిర్మాణం సమితి_అడ్డువరుసలు_context srctx;
    srctx.octx = octx;
    srctx.kparams = kparams;
    srctx.task_ప్రారంభం = task_ప్రారంభం;
    srctx.tasks = tasks;
    srctx.tasks_చొప్పున_thread = fastdiv(tasks + n_దారాలు - 1, &octx->n_దారాలు_div);

    htp_సమితి_అడ్డువరుసలు_vtcm_layout_build(&srctx.vtcm_layout, octx->dst->type, ne00, n_దారాలు);
    srctx.vtcm_పునాది = (uint8_t *)octx->ctx->vtcm_పునాది;

    work_వరుస_func_t q_func = ఖాళీ;
    స్థిరరాశి bool is_i32 = (octx->src[1]->type == HTP_TYPE_I32);

    ఎంపిక (octx->dst->type) {
        స్థితి HTP_TYPE_F32:  q_func = is_i32 ? సమితి_అడ్డువరుసలు_thread_dma_f32_int32_t  : సమితి_అడ్డువరుసలు_thread_dma_f32_int64_t;  ఆపు;
        స్థితి HTP_TYPE_F16:  q_func = is_i32 ? సమితి_అడ్డువరుసలు_thread_dma_f16_int32_t  : సమితి_అడ్డువరుసలు_thread_dma_f16_int64_t;  ఆపు;
        స్థితి HTP_TYPE_Q8_0: q_func = is_i32 ? సమితి_అడ్డువరుసలు_thread_dma_q8_0_int32_t : సమితి_అడ్డువరుసలు_thread_dma_q8_0_int64_t; ఆపు;
        అప్రమేయ:            తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    FARF(ఎగువ_2, "set-rows: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : src0-vtcm-size %zu dst-vtcm-size %zu n-threads %d\n",
         octx->src[0]->ne[0], octx->src[0]->ne[1], octx->src[0]->ne[2], octx->src[0]->ne[3],
         octx->src[1]->ne[0], octx->src[1]->ne[1], octx->src[1]->ne[2], octx->src[1]->ne[3],
         octx->dst->ne[0], octx->dst->ne[1], octx->dst->ne[2], octx->dst->ne[3],
         srctx.vtcm_layout.src0_బైట్లు_చొప్పున_thread * n_దారాలు,
         srctx.vtcm_layout.dst_బైట్లు_చొప్పున_thread  * n_దారాలు,
         n_దారాలు);

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, q_func, &srctx, n_దారాలు);

    తిరిగి HTP_స్థితి_OK;
}
