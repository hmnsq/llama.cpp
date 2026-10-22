#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <string.h>

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "hex-profile.h"

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "hvx-types.h"
#अन्तर्भाव "hvx-utils.h"

संरचना htp_solve_tri_context {
    संरचना htp_ops_context * octx;
    uint32_t                 jobs_प्रत्येकम्_thread;
    uint32_t                 योग_jobs;
    uint32_t                 job_start;
    uint32_t                 k_chunks;
    uint32_t                 col_खण्डम्;
};

स्थिर अन्तःस्थ शून्य solve_tri_row_scalar(ध्रुव भिन्न * A_row,
                                        ध्रुव भिन्न * B_row,
                                        भिन्न *       X,
                                        uint32_t      row,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        भिन्न         inv_diag) {
    प्रति (uint32_t col = col0; col < col0 + coln; ++col) {
        भिन्न योग_2 = 0.0f;
        प्रति (uint32_t t = 0; t < row; ++t) {
            योग_2 += A_row[t] * X[t * k + col];
        }
        X[row * k + col] = (B_row[col] - योग_2) * inv_diag;
    }
}

स्थिर अन्तःस्थ HVX_सदिशम् hvx_आरोपय_partial_f32(ध्रुव भिन्न * src, uint32_t n) {
    HVX_सदिशम् v = *((ध्रुव HVX_UVector *) src);
    HVX_सदिशम्_Pred आवरणम् = Q6_Q_vsetq2_R(n * परिमाण(भिन्न));
    प्रत्यागम Q6_V_vmux_QVV(आवरणम्, v, Q6_V_vzero());
}

स्थिर अन्तःस्थ शून्य solve_tri_row_hvx(ध्रुव भिन्न * A_row,
                                     ध्रुव भिन्न * B_row,
                                     भिन्न *       X,
                                     uint32_t      row,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     भिन्न         inv_diag) {
    ध्रुव bool पूर्णम् = (coln == VLEN_FP32);

    HVX_सदिशम् योग_v = Q6_V_vzero();
    प्रति (uint32_t t = 0; t < row; ++t) {
        ध्रुव भिन्न   a         = A_row[t];
        ध्रुव भिन्न * x_row_col = X + t * k + col0;

        HVX_सदिशम् x_v = पूर्णम् ? *((ध्रुव HVX_UVector *) x_row_col) : hvx_आरोपय_partial_f32(x_row_col, coln);
        HVX_सदिशम् a_v = hvx_vec_splat_f32(a);
        योग_v          = hvx_vec_सङ्कलय_f32_f32(योग_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    ध्रुव भिन्न * b_row_col = B_row + col0;
    भिन्न *       x_बहिर्गमनम्_col = X + row * k + col0;

    HVX_सदिशम् b_v        = पूर्णम् ? *((ध्रुव HVX_UVector *) b_row_col) : hvx_आरोपय_partial_f32(b_row_col, coln);
    HVX_सदिशम् inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_सदिशम् बहिर्गमनम्_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, योग_v), inv_diag_v);
    hvx_vec_store_u((शून्य *) x_बहिर्गमनम्_col, coln * परिमाण(भिन्न), बहिर्गमनम्_v);
}

// Batch-level thread: each job is one full batch.
स्थिर शून्य solve_tri_batch_thread_f32(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_solve_tri_context * sctx = (संरचना htp_solve_tri_context *) दत्त;
    संरचना htp_ops_context *       octx = sctx->octx;

    ध्रुव संरचना htp_tensor * src0 = octx->src[0];  // A
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];  // B
    ध्रुव संरचना htp_tensor * dst  = octx->dst;     // X

    ध्रुव uint32_t n = src0->ne[0];
    ध्रुव uint32_t k = src1->ne[0];

    ध्रुव uint32_t ne02 = src0->ne[2];

    ध्रुव uint32_t col_खण्डम् = VLEN_FP32;
    ध्रुव uint32_t k_पूर्णम्    = (k / col_खण्डम्) * col_खण्डम्;

    ध्रुव uint32_t start_batch = sctx->job_start + sctx->jobs_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t अन्तः_batch   = MIN(start_batch + sctx->jobs_प्रत्येकम्_thread, sctx->job_start + sctx->योग_jobs);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_batch);

    प्रति (uint32_t batch = start_batch; batch < अन्तः_batch; ++batch) {
        ध्रुव uint32_t i03 = batch / ne02;
        ध्रुव uint32_t i02 = batch - i03 * ne02;

        ध्रुव भिन्न * A_batch =
            (ध्रुव भिन्न *) ((ध्रुव uint8_t *) (uintptr_t) src0->दत्त + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ध्रुव भिन्न * B_batch =
            (ध्रुव भिन्न *) ((ध्रुव uint8_t *) (uintptr_t) src1->दत्त + i02 * src1->nb[2] + i03 * src1->nb[3]);
        भिन्न * X_batch = (भिन्न *) ((uint8_t *) (uintptr_t) dst->दत्त + i02 * dst->nb[2] + i03 * dst->nb[3]);

        प्रति (uint32_t row = 0; row < n; ++row) {
            ध्रुव भिन्न   diag     = A_batch[row * n + row];
            ध्रुव भिन्न   inv_diag = 1.0f / diag;
            ध्रुव भिन्न * A_row    = A_batch + row * n;
            ध्रुव भिन्न * B_row    = B_batch + row * k;

            uint32_t col0 = 0;
            प्रति (; col0 < k_पूर्णम्; col0 += col_खण्डम्) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, col_खण्डम्, inv_diag);
            }

            यदि (col0 < k) {
                ध्रुव uint32_t coln = k - col0;
                यदि (coln >= 8) {
                    solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                } अन्यथा {
                    solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) अन्तः_batch);

    FARF(उच्चः_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, start_batch, अन्तः_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
स्थिर शून्य solve_tri_chunk_thread_f32(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_solve_tri_context * sctx = (संरचना htp_solve_tri_context *) दत्त;
    संरचना htp_ops_context *       octx = sctx->octx;

    ध्रुव संरचना htp_tensor * src0 = octx->src[0];  // A
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];  // B
    ध्रुव संरचना htp_tensor * dst  = octx->dst;     // X

    ध्रुव uint32_t n = src0->ne[0];
    ध्रुव uint32_t k = src1->ne[0];

    ध्रुव uint32_t ne02 = src0->ne[2];

    ध्रुव uint32_t start_job = sctx->job_start + sctx->jobs_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t अन्तः_job   = MIN(start_job + sctx->jobs_प्रत्येकम्_thread, sctx->job_start + sctx->योग_jobs);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_job);

    प्रति (uint32_t job = start_job; job < अन्तः_job; ++job) {
        ध्रुव uint32_t batch = job / sctx->k_chunks;
        ध्रुव uint32_t chunk = job - batch * sctx->k_chunks;

        ध्रुव uint32_t i03 = batch / ne02;
        ध्रुव uint32_t i02 = batch - i03 * ne02;

        ध्रुव भिन्न * A_batch =
            (ध्रुव भिन्न *) ((ध्रुव uint8_t *) (uintptr_t) src0->दत्त + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ध्रुव भिन्न * B_batch =
            (ध्रुव भिन्न *) ((ध्रुव uint8_t *) (uintptr_t) src1->दत्त + i02 * src1->nb[2] + i03 * src1->nb[3]);
        भिन्न * X_batch = (भिन्न *) ((uint8_t *) (uintptr_t) dst->दत्त + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ध्रुव uint32_t col0 = chunk * sctx->col_खण्डम्;
        ध्रुव uint32_t coln = MIN(sctx->col_खण्डम्, k - col0);

        प्रति (uint32_t row = 0; row < n; ++row) {
            ध्रुव भिन्न diag     = A_batch[row * n + row];
            ध्रुव भिन्न inv_diag = 1.0f / diag;

            ध्रुव भिन्न * A_row = A_batch + row * n;
            ध्रुव भिन्न * B_row = B_batch + row * k;

            यदि (coln >= 8) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            } अन्यथा {
                solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) अन्तः_job);

    FARF(उच्चः_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, start_job, अन्तः_job);
}

पूर्णाङ्क op_solve_tri(संरचना htp_ops_context * octx) {
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];  // A
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];  // B
    ध्रुव संरचना htp_tensor * dst  = octx->dst;     // X

    यदि (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    यदि (src0->ne[0] != src0->ne[1]) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }
    यदि (src0->ne[1] != src1->ne[1]) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }
    यदि (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }
    यदि (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }

    यदि (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव uint32_t k = src1->ne[0];

    ध्रुव uint32_t col_खण्डम्     = VLEN_FP32;
    ध्रुव uint32_t k_chunks      = (k + col_खण्डम् - 1) / col_खण्डम्;
    ध्रुव uint32_t योग_batches = src0->ne[2] * src0->ne[3];
    ध्रुव bool     batched       = योग_batches >= (uint32_t) octx->n_तन्तवः;

    FARF(उच्चः_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    यदि (batched) {
        uint32_t job_start = 0;
        uint32_t njobs     = योग_batches;

        यदि (octx->ctx->mdev.गणनम् > 1) {
            ध्रुव uint32_t batch_आकार = dst->nb[2];
            ध्रुव uint32_t batches_प्रत्येकम्_chunk = (batch_आकार > 0) ? (HEX_L2_पङ्क्तिः_आकार / hex_gcd_u32(batch_आकार, HEX_L2_पङ्क्तिः_आकार)) : 1;
            ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_batches, htp_tensor_mdev_दत्त_aligned(dst) ? batches_प्रत्येकम्_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
            job_start = range.start;
            njobs     = range.गणनम्;
        }

        यदि (njobs == 0) {
            प्रत्यागम HTP_स्थितिः_OK;
        }

        // Batch-level parallelism
        ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;

        संरचना htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_प्रत्येकम्_thread = fastdiv(njobs + n_तन्तवः - 1, &octx->n_तन्तवः_div),
            .योग_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_खण्डम्       = col_खण्डम्,
        };

        work_queue_सततम्(octx->ctx->work_queue, solve_tri_batch_thread_f32, &sctx, n_तन्तवः);
    } अन्यथा {
        // Chunk-level parallelism
        ध्रुव uint32_t योग_jobs = योग_batches * k_chunks;

        uint32_t job_start = 0;
        uint32_t njobs     = योग_jobs;

        यदि (octx->ctx->mdev.गणनम् > 1) {
            ध्रुव bool can_विभज = htp_tensor_mdev_दत्त_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_पङ्क्तिः_आकार - 1)) == 0);
            ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_jobs, can_विभज ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
            job_start = range.start;
            njobs     = range.गणनम्;
        }

        यदि (njobs == 0) {
            प्रत्यागम HTP_स्थितिः_OK;
        }

        ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;

        संरचना htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_प्रत्येकम्_thread = fastdiv(njobs + n_तन्तवः - 1, &octx->n_तन्तवः_div),
            .योग_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_खण्डम्       = col_खण्डम्,
        };

        work_queue_सततम्(octx->ctx->work_queue, solve_tri_chunk_thread_f32, &sctx, n_तन्तवः);
    }

    प्रत्यागम HTP_स्थितिः_OK;
}
