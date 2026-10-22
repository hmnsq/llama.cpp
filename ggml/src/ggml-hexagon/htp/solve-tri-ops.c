#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <string.h>

#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "hex-profile.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "hvx-types.h"
#అన్తర్భావ "hvx-utils.h"

నిర్మాణం htp_solve_tri_context {
    నిర్మాణం htp_ops_context * octx;
    uint32_t                 jobs_చొప్పున_thread;
    uint32_t                 మొత్తం_jobs;
    uint32_t                 పని_ప్రారంభం;
    uint32_t                 k_chunks;
    uint32_t                 నిలువు_ఖండం;
};

స్థిర అంతఃవరుస శూన్య solve_tri_అడ్డువరుస_scalar(స్థిరరాశి దశాంశం * A_అడ్డువరుస,
                                        స్థిరరాశి దశాంశం * B_అడ్డువరుస,
                                        దశాంశం *       X,
                                        uint32_t      అడ్డువరుస,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        దశాంశం         inv_diag) {
    కొరకు (uint32_t నిలువు = col0; నిలువు < col0 + coln; ++నిలువు) {
        దశాంశం కూడిక = 0.0f;
        కొరకు (uint32_t t = 0; t < అడ్డువరుస; ++t) {
            కూడిక += A_అడ్డువరుస[t] * X[t * k + నిలువు];
        }
        X[అడ్డువరుస * k + నిలువు] = (B_అడ్డువరుస[నిలువు] - కూడిక) * inv_diag;
    }
}

స్థిర అంతఃవరుస HVX_సదిశ hvx_ఎక్కించు_partial_f32(స్థిరరాశి దశాంశం * src, uint32_t n) {
    HVX_సదిశ v = *((స్థిరరాశి HVX_UVector *) src);
    HVX_సదిశ_Pred ముసుగు = Q6_Q_vsetq2_R(n * పరిమాణం(దశాంశం));
    తిరిగి Q6_V_vmux_QVV(ముసుగు, v, Q6_V_vzero());
}

స్థిర అంతఃవరుస శూన్య solve_tri_అడ్డువరుస_hvx(స్థిరరాశి దశాంశం * A_అడ్డువరుస,
                                     స్థిరరాశి దశాంశం * B_అడ్డువరుస,
                                     దశాంశం *       X,
                                     uint32_t      అడ్డువరుస,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     దశాంశం         inv_diag) {
    స్థిరరాశి bool నిండిన = (coln == VLEN_FP32);

    HVX_సదిశ కూడిక_v = Q6_V_vzero();
    కొరకు (uint32_t t = 0; t < అడ్డువరుస; ++t) {
        స్థిరరాశి దశాంశం   a         = A_అడ్డువరుస[t];
        స్థిరరాశి దశాంశం * x_అడ్డువరుస_నిలువు = X + t * k + col0;

        HVX_సదిశ x_v = నిండిన ? *((స్థిరరాశి HVX_UVector *) x_అడ్డువరుస_నిలువు) : hvx_ఎక్కించు_partial_f32(x_అడ్డువరుస_నిలువు, coln);
        HVX_సదిశ a_v = hvx_vec_splat_f32(a);
        కూడిక_v          = hvx_vec_కూడు_f32_f32(కూడిక_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    స్థిరరాశి దశాంశం * b_అడ్డువరుస_నిలువు = B_అడ్డువరుస + col0;
    దశాంశం *       x_బహిర్గతం_నిలువు = X + అడ్డువరుస * k + col0;

    HVX_సదిశ b_v        = నిండిన ? *((స్థిరరాశి HVX_UVector *) b_అడ్డువరుస_నిలువు) : hvx_ఎక్కించు_partial_f32(b_అడ్డువరుస_నిలువు, coln);
    HVX_సదిశ inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_సదిశ బహిర్గతం_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, కూడిక_v), inv_diag_v);
    hvx_vec_store_u((శూన్య *) x_బహిర్గతం_నిలువు, coln * పరిమాణం(దశాంశం), బహిర్గతం_v);
}

// Batch-level thread: each job is one full batch.
స్థిర శూన్య solve_tri_batch_thread_f32(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_solve_tri_context * sctx = (నిర్మాణం htp_solve_tri_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context *       octx = sctx->octx;

    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];  // A
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];  // B
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;     // X

    స్థిరరాశి uint32_t n = src0->ne[0];
    స్థిరరాశి uint32_t k = src1->ne[0];

    స్థిరరాశి uint32_t ne02 = src0->ne[2];

    స్థిరరాశి uint32_t నిలువు_ఖండం = VLEN_FP32;
    స్థిరరాశి uint32_t k_నిండిన    = (k / నిలువు_ఖండం) * నిలువు_ఖండం;

    స్థిరరాశి uint32_t ప్రారంభం_batch = sctx->పని_ప్రారంభం + sctx->jobs_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t చివర_batch   = MIN(ప్రారంభం_batch + sctx->jobs_చొప్పున_thread, sctx->పని_ప్రారంభం + sctx->మొత్తం_jobs);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ప్రారంభం_batch);

    కొరకు (uint32_t batch = ప్రారంభం_batch; batch < చివర_batch; ++batch) {
        స్థిరరాశి uint32_t i03 = batch / ne02;
        స్థిరరాశి uint32_t i02 = batch - i03 * ne02;

        స్థిరరాశి దశాంశం * A_batch =
            (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) (uintptr_t) src0->దత్తాంశం + i02 * src0->nb[2] + i03 * src0->nb[3]);
        స్థిరరాశి దశాంశం * B_batch =
            (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) (uintptr_t) src1->దత్తాంశం + i02 * src1->nb[2] + i03 * src1->nb[3]);
        దశాంశం * X_batch = (దశాంశం *) ((uint8_t *) (uintptr_t) dst->దత్తాంశం + i02 * dst->nb[2] + i03 * dst->nb[3]);

        కొరకు (uint32_t అడ్డువరుస = 0; అడ్డువరుస < n; ++అడ్డువరుస) {
            స్థిరరాశి దశాంశం   diag     = A_batch[అడ్డువరుస * n + అడ్డువరుస];
            స్థిరరాశి దశాంశం   inv_diag = 1.0f / diag;
            స్థిరరాశి దశాంశం * A_అడ్డువరుస    = A_batch + అడ్డువరుస * n;
            స్థిరరాశి దశాంశం * B_అడ్డువరుస    = B_batch + అడ్డువరుస * k;

            uint32_t col0 = 0;
            కొరకు (; col0 < k_నిండిన; col0 += నిలువు_ఖండం) {
                solve_tri_అడ్డువరుస_hvx(A_అడ్డువరుస, B_అడ్డువరుస, X_batch, అడ్డువరుస, k, col0, నిలువు_ఖండం, inv_diag);
            }

            ఒకవేళ (col0 < k) {
                స్థిరరాశి uint32_t coln = k - col0;
                ఒకవేళ (coln >= 8) {
                    solve_tri_అడ్డువరుస_hvx(A_అడ్డువరుస, B_అడ్డువరుస, X_batch, అడ్డువరుస, k, col0, coln, inv_diag);
                } లేకపోతే {
                    solve_tri_అడ్డువరుస_scalar(A_అడ్డువరుస, B_అడ్డువరుస, X_batch, అడ్డువరుస, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) చివర_batch);

    FARF(ఎగువ_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, ప్రారంభం_batch, చివర_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
స్థిర శూన్య solve_tri_chunk_thread_f32(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_solve_tri_context * sctx = (నిర్మాణం htp_solve_tri_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context *       octx = sctx->octx;

    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];  // A
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];  // B
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;     // X

    స్థిరరాశి uint32_t n = src0->ne[0];
    స్థిరరాశి uint32_t k = src1->ne[0];

    స్థిరరాశి uint32_t ne02 = src0->ne[2];

    స్థిరరాశి uint32_t ప్రారంభం_పని = sctx->పని_ప్రారంభం + sctx->jobs_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t చివర_పని   = MIN(ప్రారంభం_పని + sctx->jobs_చొప్పున_thread, sctx->పని_ప్రారంభం + sctx->మొత్తం_jobs);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ప్రారంభం_పని);

    కొరకు (uint32_t పని = ప్రారంభం_పని; పని < చివర_పని; ++పని) {
        స్థిరరాశి uint32_t batch = పని / sctx->k_chunks;
        స్థిరరాశి uint32_t chunk = పని - batch * sctx->k_chunks;

        స్థిరరాశి uint32_t i03 = batch / ne02;
        స్థిరరాశి uint32_t i02 = batch - i03 * ne02;

        స్థిరరాశి దశాంశం * A_batch =
            (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) (uintptr_t) src0->దత్తాంశం + i02 * src0->nb[2] + i03 * src0->nb[3]);
        స్థిరరాశి దశాంశం * B_batch =
            (స్థిరరాశి దశాంశం *) ((స్థిరరాశి uint8_t *) (uintptr_t) src1->దత్తాంశం + i02 * src1->nb[2] + i03 * src1->nb[3]);
        దశాంశం * X_batch = (దశాంశం *) ((uint8_t *) (uintptr_t) dst->దత్తాంశం + i02 * dst->nb[2] + i03 * dst->nb[3]);

        స్థిరరాశి uint32_t col0 = chunk * sctx->నిలువు_ఖండం;
        స్థిరరాశి uint32_t coln = MIN(sctx->నిలువు_ఖండం, k - col0);

        కొరకు (uint32_t అడ్డువరుస = 0; అడ్డువరుస < n; ++అడ్డువరుస) {
            స్థిరరాశి దశాంశం diag     = A_batch[అడ్డువరుస * n + అడ్డువరుస];
            స్థిరరాశి దశాంశం inv_diag = 1.0f / diag;

            స్థిరరాశి దశాంశం * A_అడ్డువరుస = A_batch + అడ్డువరుస * n;
            స్థిరరాశి దశాంశం * B_అడ్డువరుస = B_batch + అడ్డువరుస * k;

            ఒకవేళ (coln >= 8) {
                solve_tri_అడ్డువరుస_hvx(A_అడ్డువరుస, B_అడ్డువరుస, X_batch, అడ్డువరుస, k, col0, coln, inv_diag);
            } లేకపోతే {
                solve_tri_అడ్డువరుస_scalar(A_అడ్డువరుస, B_అడ్డువరుస, X_batch, అడ్డువరుస, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) చివర_పని);

    FARF(ఎగువ_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, ప్రారంభం_పని, చివర_పని);
}

పూర్ణాంకం op_solve_tri(నిర్మాణం htp_ops_context * octx) {
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];  // A
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];  // B
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;     // X

    ఒకవేళ (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    ఒకవేళ (src0->ne[0] != src0->ne[1]) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }
    ఒకవేళ (src0->ne[1] != src1->ne[1]) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }
    ఒకవేళ (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }
    ఒకవేళ (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }

    ఒకవేళ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి uint32_t k = src1->ne[0];

    స్థిరరాశి uint32_t నిలువు_ఖండం     = VLEN_FP32;
    స్థిరరాశి uint32_t k_chunks      = (k + నిలువు_ఖండం - 1) / నిలువు_ఖండం;
    స్థిరరాశి uint32_t మొత్తం_batches = src0->ne[2] * src0->ne[3];
    స్థిరరాశి bool     batched       = మొత్తం_batches >= (uint32_t) octx->n_దారాలు;

    FARF(ఎగువ_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    ఒకవేళ (batched) {
        uint32_t పని_ప్రారంభం = 0;
        uint32_t njobs     = మొత్తం_batches;

        ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
            స్థిరరాశి uint32_t batch_కొలత = dst->nb[2];
            స్థిరరాశి uint32_t batches_చొప్పున_chunk = (batch_కొలత > 0) ? (HEX_L2_పంక్తి_కొలత / hex_గరిష్ఠ_సామాన్య_భాజకం_u32(batch_కొలత, HEX_L2_పంక్తి_కొలత)) : 1;
            స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(మొత్తం_batches, htp_tensor_mdev_దత్తాంశం_aligned(dst) ? batches_చొప్పున_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
            పని_ప్రారంభం = range.ప్రారంభం;
            njobs     = range.లెక్క;
        }

        ఒకవేళ (njobs == 0) {
            తిరిగి HTP_స్థితి_OK;
        }

        // Batch-level parallelism
        స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

        నిర్మాణం htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_చొప్పున_thread = fastdiv(njobs + n_దారాలు - 1, &octx->n_దారాలు_div),
            .మొత్తం_jobs      = njobs,
            .పని_ప్రారంభం       = పని_ప్రారంభం,
            .k_chunks        = k_chunks,
            .నిలువు_ఖండం       = నిలువు_ఖండం,
        };

        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, solve_tri_batch_thread_f32, &sctx, n_దారాలు);
    } లేకపోతే {
        // Chunk-level parallelism
        స్థిరరాశి uint32_t మొత్తం_jobs = మొత్తం_batches * k_chunks;

        uint32_t పని_ప్రారంభం = 0;
        uint32_t njobs     = మొత్తం_jobs;

        ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
            స్థిరరాశి bool can_విభజించు = htp_tensor_mdev_దత్తాంశం_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_పంక్తి_కొలత - 1)) == 0);
            స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(మొత్తం_jobs, can_విభజించు ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
            పని_ప్రారంభం = range.ప్రారంభం;
            njobs     = range.లెక్క;
        }

        ఒకవేళ (njobs == 0) {
            తిరిగి HTP_స్థితి_OK;
        }

        స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

        నిర్మాణం htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_చొప్పున_thread = fastdiv(njobs + n_దారాలు - 1, &octx->n_దారాలు_div),
            .మొత్తం_jobs      = njobs,
            .పని_ప్రారంభం       = పని_ప్రారంభం,
            .k_chunks        = k_chunks,
            .నిలువు_ఖండం       = నిలువు_ఖండం,
        };

        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, solve_tri_chunk_thread_f32, &sctx, n_దారాలు);
    }

    తిరిగి HTP_స్థితి_OK;
}
