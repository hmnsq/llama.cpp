#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <string.h>

#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "hex-profile.h"

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"
#ଅନ୍ତର୍ଭାଵ "hvx-types.h"
#ଅନ୍ତର୍ଭାଵ "hvx-utils.h"

ସଂରଚନା htp_solve_tri_context {
    ସଂରଚନା htp_ops_context * octx;
    uint32_t                 jobs_per_thread;
    uint32_t                 total_jobs;
    uint32_t                 job_start;
    uint32_t                 k_chunks;
    uint32_t                 col_block;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ solve_tri_row_scalar(ଅଚଳ ଦଶମିକ * A_row,
                                        ଅଚଳ ଦଶମିକ * B_row,
                                        ଦଶମିକ *       X,
                                        uint32_t      row,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        ଦଶମିକ         inv_diag) {
    ପାଇଁ (uint32_t col = col0; col < col0 + coln; ++col) {
        ଦଶମିକ sum = 0.0f;
        ପାଇଁ (uint32_t t = 0; t < row; ++t) {
            sum += A_row[t] * X[t * k + col];
        }
        X[row * k + col] = (B_row[col] - sum) * inv_diag;
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ HVX_Vector hvx_load_partial_f32(ଅଚଳ ଦଶମିକ * src, uint32_t n) {
    HVX_Vector v = *((ଅଚଳ HVX_UVector *) src);
    HVX_VectorPred mask = Q6_Q_vsetq2_R(n * ଆକାର(ଦଶମିକ));
    ଫେରନ୍ତୁ Q6_V_vmux_QVV(mask, v, Q6_V_vzero());
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ solve_tri_row_hvx(ଅଚଳ ଦଶମିକ * A_row,
                                     ଅଚଳ ଦଶମିକ * B_row,
                                     ଦଶମିକ *       X,
                                     uint32_t      row,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     ଦଶମିକ         inv_diag) {
    ଅଚଳ bool full = (coln == VLEN_FP32);

    HVX_Vector sum_v = Q6_V_vzero();
    ପାଇଁ (uint32_t t = 0; t < row; ++t) {
        ଅଚଳ ଦଶମିକ   a         = A_row[t];
        ଅଚଳ ଦଶମିକ * x_row_col = X + t * k + col0;

        HVX_Vector x_v = full ? *((ଅଚଳ HVX_UVector *) x_row_col) : hvx_load_partial_f32(x_row_col, coln);
        HVX_Vector a_v = hvx_vec_splat_f32(a);
        sum_v          = hvx_vec_add_f32_f32(sum_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    ଅଚଳ ଦଶମିକ * b_row_col = B_row + col0;
    ଦଶମିକ *       x_out_col = X + row * k + col0;

    HVX_Vector b_v        = full ? *((ଅଚଳ HVX_UVector *) b_row_col) : hvx_load_partial_f32(b_row_col, coln);
    HVX_Vector inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_Vector out_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, sum_v), inv_diag_v);
    hvx_vec_store_u((ଶୂନ୍ୟ *) x_out_col, coln * ଆକାର(ଦଶମିକ), out_v);
}

// Batch-level thread: each job is one full batch.
ସ୍ଥିର ଶୂନ୍ୟ solve_tri_batch_thread_f32(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ସଂରଚନା htp_solve_tri_context * sctx = (ସଂରଚନା htp_solve_tri_context *) data;
    ସଂରଚନା htp_ops_context *       octx = sctx->octx;

    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];  // A
    ଅଚଳ ସଂରଚନା htp_tensor * src1 = octx->src[1];  // B
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;     // X

    ଅଚଳ uint32_t n = src0->ne[0];
    ଅଚଳ uint32_t k = src1->ne[0];

    ଅଚଳ uint32_t ne02 = src0->ne[2];

    ଅଚଳ uint32_t col_block = VLEN_FP32;
    ଅଚଳ uint32_t k_full    = (k / col_block) * col_block;

    ଅଚଳ uint32_t start_batch = sctx->job_start + sctx->jobs_per_thread * ith;
    ଅଚଳ uint32_t end_batch   = MIN(start_batch + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_batch);

    ପାଇଁ (uint32_t batch = start_batch; batch < end_batch; ++batch) {
        ଅଚଳ uint32_t i03 = batch / ne02;
        ଅଚଳ uint32_t i02 = batch - i03 * ne02;

        ଅଚଳ ଦଶମିକ * A_batch =
            (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ଅଚଳ ଦଶମିକ * B_batch =
            (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        ଦଶମିକ * X_batch = (ଦଶମିକ *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ପାଇଁ (uint32_t row = 0; row < n; ++row) {
            ଅଚଳ ଦଶମିକ   diag     = A_batch[row * n + row];
            ଅଚଳ ଦଶମିକ   inv_diag = 1.0f / diag;
            ଅଚଳ ଦଶମିକ * A_row    = A_batch + row * n;
            ଅଚଳ ଦଶମିକ * B_row    = B_batch + row * k;

            uint32_t col0 = 0;
            ପାଇଁ (; col0 < k_full; col0 += col_block) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, col_block, inv_diag);
            }

            ଯଦି (col0 < k) {
                ଅଚଳ uint32_t coln = k - col0;
                ଯଦି (coln >= 8) {
                    solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                } ଅନ୍ୟଥା {
                    solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_batch);

    FARF(ଉପର_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, start_batch, end_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
ସ୍ଥିର ଶୂନ୍ୟ solve_tri_chunk_thread_f32(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ସଂରଚନା htp_solve_tri_context * sctx = (ସଂରଚନା htp_solve_tri_context *) data;
    ସଂରଚନା htp_ops_context *       octx = sctx->octx;

    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];  // A
    ଅଚଳ ସଂରଚନା htp_tensor * src1 = octx->src[1];  // B
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;     // X

    ଅଚଳ uint32_t n = src0->ne[0];
    ଅଚଳ uint32_t k = src1->ne[0];

    ଅଚଳ uint32_t ne02 = src0->ne[2];

    ଅଚଳ uint32_t start_job = sctx->job_start + sctx->jobs_per_thread * ith;
    ଅଚଳ uint32_t end_job   = MIN(start_job + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_job);

    ପାଇଁ (uint32_t job = start_job; job < end_job; ++job) {
        ଅଚଳ uint32_t batch = job / sctx->k_chunks;
        ଅଚଳ uint32_t chunk = job - batch * sctx->k_chunks;

        ଅଚଳ uint32_t i03 = batch / ne02;
        ଅଚଳ uint32_t i02 = batch - i03 * ne02;

        ଅଚଳ ଦଶମିକ * A_batch =
            (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ଅଚଳ ଦଶମିକ * B_batch =
            (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        ଦଶମିକ * X_batch = (ଦଶମିକ *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ଅଚଳ uint32_t col0 = chunk * sctx->col_block;
        ଅଚଳ uint32_t coln = MIN(sctx->col_block, k - col0);

        ପାଇଁ (uint32_t row = 0; row < n; ++row) {
            ଅଚଳ ଦଶମିକ diag     = A_batch[row * n + row];
            ଅଚଳ ଦଶମିକ inv_diag = 1.0f / diag;

            ଅଚଳ ଦଶମିକ * A_row = A_batch + row * n;
            ଅଚଳ ଦଶମିକ * B_row = B_batch + row * k;

            ଯଦି (coln >= 8) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            } ଅନ୍ୟଥା {
                solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_job);

    FARF(ଉପର_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, start_job, end_job);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_solve_tri(ସଂରଚନା htp_ops_context * octx) {
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];  // A
    ଅଚଳ ସଂରଚନା htp_tensor * src1 = octx->src[1];  // B
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;     // X

    ଯଦି (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    ଯଦି (src0->ne[0] != src0->ne[1]) {
        ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
    }
    ଯଦି (src0->ne[1] != src1->ne[1]) {
        ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
    }
    ଯଦି (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
    }
    ଯଦି (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
    }

    ଯଦି (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଅଚଳ uint32_t k = src1->ne[0];

    ଅଚଳ uint32_t col_block     = VLEN_FP32;
    ଅଚଳ uint32_t k_chunks      = (k + col_block - 1) / col_block;
    ଅଚଳ uint32_t total_batches = src0->ne[2] * src0->ne[3];
    ଅଚଳ bool     batched       = total_batches >= (uint32_t) octx->n_threads;

    FARF(ଉପର_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    ଯଦି (batched) {
        uint32_t job_start = 0;
        uint32_t njobs     = total_batches;

        ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
            ଅଚଳ uint32_t batch_size = dst->nb[2];
            ଅଚଳ uint32_t batches_per_chunk = (batch_size > 0) ? (HEX_L2_ଧାଡ଼ି_SIZE / hex_gcd_u32(batch_size, HEX_L2_ଧାଡ଼ି_SIZE)) : 1;
            ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, htp_tensor_mdev_data_aligned(dst) ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
            job_start = range.start;
            njobs     = range.ଗଣନା_2;
        }

        ଯଦି (njobs == 0) {
            ଫେରନ୍ତୁ HTP_STATUS_OK;
        }

        // Batch-level parallelism
        ଅଚଳ uint32_t n_threads = octx->n_threads;

        ସଂରଚନା htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_batch_thread_f32, &sctx, n_threads);
    } ଅନ୍ୟଥା {
        // Chunk-level parallelism
        ଅଚଳ uint32_t total_jobs = total_batches * k_chunks;

        uint32_t job_start = 0;
        uint32_t njobs     = total_jobs;

        ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
            ଅଚଳ bool can_split = htp_tensor_mdev_data_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_ଧାଡ଼ି_SIZE - 1)) == 0);
            ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_jobs, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
            job_start = range.start;
            njobs     = range.ଗଣନା_2;
        }

        ଯଦି (njobs == 0) {
            ଫେରନ୍ତୁ HTP_STATUS_OK;
        }

        ଅଚଳ uint32_t n_threads = octx->n_threads;

        ସଂରଚନା htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_chunk_thread_f32, &sctx, n_threads);
    }

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}
