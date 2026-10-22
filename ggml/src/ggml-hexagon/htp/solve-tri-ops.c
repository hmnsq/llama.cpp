#आशय clang diagnostic ignored "-Wunused-but-set-variable"

#समावेश <HAP_farf.h>
#समावेश <string.h>

#समावेश "hex-common.h"
#समावेश "hex-profile.h"

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"
#समावेश "hvx-types.h"
#समावेश "hvx-utils.h"

काष्ठा htp_solve_tri_context {
    काष्ठा htp_ops_context * octx;
    uint32_t                 jobs_per_thread;
    uint32_t                 योग_jobs;
    uint32_t                 job_start;
    uint32_t                 k_chunks;
    uint32_t                 col_block;
};

अटल अंतरभूत व्योम solve_tri_row_scalar(स्थिर भग्न * A_row,
                                        स्थिर भग्न * B_row,
                                        भग्न *       X,
                                        uint32_t      row,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        भग्न         inv_diag) {
    क्रम (uint32_t col = col0; col < col0 + coln; ++col) {
        भग्न योग_2 = 0.0f;
        क्रम (uint32_t t = 0; t < row; ++t) {
            योग_2 += A_row[t] * X[t * k + col];
        }
        X[row * k + col] = (B_row[col] - योग_2) * inv_diag;
    }
}

अटल अंतरभूत HVX_Vector hvx_load_partial_f32(स्थिर भग्न * src, uint32_t n) {
    HVX_Vector v = *((स्थिर HVX_UVector *) src);
    HVX_VectorPred mask = Q6_Q_vsetq2_R(n * माप(भग्न));
    वापस Q6_V_vmux_QVV(mask, v, Q6_V_vzero());
}

अटल अंतरभूत व्योम solve_tri_row_hvx(स्थिर भग्न * A_row,
                                     स्थिर भग्न * B_row,
                                     भग्न *       X,
                                     uint32_t      row,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     भग्न         inv_diag) {
    स्थिर bool full = (coln == VLEN_FP32);

    HVX_Vector योग_v = Q6_V_vzero();
    क्रम (uint32_t t = 0; t < row; ++t) {
        स्थिर भग्न   a         = A_row[t];
        स्थिर भग्न * x_row_col = X + t * k + col0;

        HVX_Vector x_v = full ? *((स्थिर HVX_UVector *) x_row_col) : hvx_load_partial_f32(x_row_col, coln);
        HVX_Vector a_v = hvx_vec_splat_f32(a);
        योग_v          = hvx_vec_add_f32_f32(योग_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    स्थिर भग्न * b_row_col = B_row + col0;
    भग्न *       x_out_col = X + row * k + col0;

    HVX_Vector b_v        = full ? *((स्थिर HVX_UVector *) b_row_col) : hvx_load_partial_f32(b_row_col, coln);
    HVX_Vector inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_Vector out_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, योग_v), inv_diag_v);
    hvx_vec_store_u((व्योम *) x_out_col, coln * माप(भग्न), out_v);
}

// Batch-level thread: each job is one full batch.
अटल व्योम solve_tri_batch_thread_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_solve_tri_context * sctx = (काष्ठा htp_solve_tri_context *) तथ्य;
    काष्ठा htp_ops_context *       octx = sctx->octx;

    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];  // A
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];  // B
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;     // X

    स्थिर uint32_t n = src0->ne[0];
    स्थिर uint32_t k = src1->ne[0];

    स्थिर uint32_t ne02 = src0->ne[2];

    स्थिर uint32_t col_block = VLEN_FP32;
    स्थिर uint32_t k_full    = (k / col_block) * col_block;

    स्थिर uint32_t start_batch = sctx->job_start + sctx->jobs_per_thread * ith;
    स्थिर uint32_t end_batch   = MIN(start_batch + sctx->jobs_per_thread, sctx->job_start + sctx->योग_jobs);

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_batch);

    क्रम (uint32_t batch = start_batch; batch < end_batch; ++batch) {
        स्थिर uint32_t i03 = batch / ne02;
        स्थिर uint32_t i02 = batch - i03 * ne02;

        स्थिर भग्न * A_batch =
            (स्थिर भग्न *) ((स्थिर uint8_t *) (uintptr_t) src0->तथ्य + i02 * src0->nb[2] + i03 * src0->nb[3]);
        स्थिर भग्न * B_batch =
            (स्थिर भग्न *) ((स्थिर uint8_t *) (uintptr_t) src1->तथ्य + i02 * src1->nb[2] + i03 * src1->nb[3]);
        भग्न * X_batch = (भग्न *) ((uint8_t *) (uintptr_t) dst->तथ्य + i02 * dst->nb[2] + i03 * dst->nb[3]);

        क्रम (uint32_t row = 0; row < n; ++row) {
            स्थिर भग्न   diag     = A_batch[row * n + row];
            स्थिर भग्न   inv_diag = 1.0f / diag;
            स्थिर भग्न * A_row    = A_batch + row * n;
            स्थिर भग्न * B_row    = B_batch + row * k;

            uint32_t col0 = 0;
            क्रम (; col0 < k_full; col0 += col_block) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, col_block, inv_diag);
            }

            अगर (col0 < k) {
                स्थिर uint32_t coln = k - col0;
                अगर (coln >= 8) {
                    solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                } अन्यथा {
                    solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_batch);

    FARF(ऊपरी_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, start_batch, end_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
अटल व्योम solve_tri_chunk_thread_f32(अचिन्हित पूर्णांक nth, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_solve_tri_context * sctx = (काष्ठा htp_solve_tri_context *) तथ्य;
    काष्ठा htp_ops_context *       octx = sctx->octx;

    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];  // A
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];  // B
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;     // X

    स्थिर uint32_t n = src0->ne[0];
    स्थिर uint32_t k = src1->ne[0];

    स्थिर uint32_t ne02 = src0->ne[2];

    स्थिर uint32_t start_job = sctx->job_start + sctx->jobs_per_thread * ith;
    स्थिर uint32_t end_job   = MIN(start_job + sctx->jobs_per_thread, sctx->job_start + sctx->योग_jobs);

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_job);

    क्रम (uint32_t job = start_job; job < end_job; ++job) {
        स्थिर uint32_t batch = job / sctx->k_chunks;
        स्थिर uint32_t chunk = job - batch * sctx->k_chunks;

        स्थिर uint32_t i03 = batch / ne02;
        स्थिर uint32_t i02 = batch - i03 * ne02;

        स्थिर भग्न * A_batch =
            (स्थिर भग्न *) ((स्थिर uint8_t *) (uintptr_t) src0->तथ्य + i02 * src0->nb[2] + i03 * src0->nb[3]);
        स्थिर भग्न * B_batch =
            (स्थिर भग्न *) ((स्थिर uint8_t *) (uintptr_t) src1->तथ्य + i02 * src1->nb[2] + i03 * src1->nb[3]);
        भग्न * X_batch = (भग्न *) ((uint8_t *) (uintptr_t) dst->तथ्य + i02 * dst->nb[2] + i03 * dst->nb[3]);

        स्थिर uint32_t col0 = chunk * sctx->col_block;
        स्थिर uint32_t coln = MIN(sctx->col_block, k - col0);

        क्रम (uint32_t row = 0; row < n; ++row) {
            स्थिर भग्न diag     = A_batch[row * n + row];
            स्थिर भग्न inv_diag = 1.0f / diag;

            स्थिर भग्न * A_row = A_batch + row * n;
            स्थिर भग्न * B_row = B_batch + row * k;

            अगर (coln >= 8) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            } अन्यथा {
                solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_job);

    FARF(ऊपरी_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, start_job, end_job);
}

पूर्णांक op_solve_tri(काष्ठा htp_ops_context * octx) {
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];  // A
    स्थिर काष्ठा htp_tensor * src1 = octx->src[1];  // B
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;     // X

    अगर (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    अगर (src0->ne[0] != src0->ne[1]) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }
    अगर (src0->ne[1] != src1->ne[1]) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }
    अगर (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }
    अगर (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        वापस HTP_STATUS_INVAL_PARAMS;
    }

    अगर (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    स्थिर uint32_t k = src1->ne[0];

    स्थिर uint32_t col_block     = VLEN_FP32;
    स्थिर uint32_t k_chunks      = (k + col_block - 1) / col_block;
    स्थिर uint32_t योग_batches = src0->ne[2] * src0->ne[3];
    स्थिर bool     batched       = योग_batches >= (uint32_t) octx->n_threads;

    FARF(ऊपरी_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    अगर (batched) {
        uint32_t job_start = 0;
        uint32_t njobs     = योग_batches;

        अगर (octx->ctx->mdev.गिनती > 1) {
            स्थिर uint32_t batch_आकार = dst->nb[2];
            स्थिर uint32_t batches_per_chunk = (batch_आकार > 0) ? (HEX_L2_पंक्ति_आकार / hex_gcd_u32(batch_आकार, HEX_L2_पंक्ति_आकार)) : 1;
            स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_batches, htp_tensor_mdev_तथ्य_aligned(dst) ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
            job_start = range.start;
            njobs     = range.गिनती;
        }

        अगर (njobs == 0) {
            वापस HTP_STATUS_OK;
        }

        // Batch-level parallelism
        स्थिर uint32_t n_threads = octx->n_threads;

        काष्ठा htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .योग_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_batch_thread_f32, &sctx, n_threads);
    } अन्यथा {
        // Chunk-level parallelism
        स्थिर uint32_t योग_jobs = योग_batches * k_chunks;

        uint32_t job_start = 0;
        uint32_t njobs     = योग_jobs;

        अगर (octx->ctx->mdev.गिनती > 1) {
            स्थिर bool can_split = htp_tensor_mdev_तथ्य_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_पंक्ति_आकार - 1)) == 0);
            स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(योग_jobs, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
            job_start = range.start;
            njobs     = range.गिनती;
        }

        अगर (njobs == 0) {
            वापस HTP_STATUS_OK;
        }

        स्थिर uint32_t n_threads = octx->n_threads;

        काष्ठा htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .योग_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_chunk_thread_f32, &sctx, n_threads);
    }

    वापस HTP_STATUS_OK;
}
