#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাব <HAP_farf.h>
#অন্তর্ভাব <string.h>

#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "hex-profile.h"

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"
#অন্তর্ভাব "hvx-types.h"
#অন্তর্ভাব "hvx-utils.h"

কাঠামো htp_solve_tri_context {
    কাঠামো htp_ops_context * octx;
    uint32_t                 jobs_per_thread;
    uint32_t                 যোগ_jobs;
    uint32_t                 job_start;
    uint32_t                 k_chunks;
    uint32_t                 col_block;
};

স্থির অন্তঃরেখ শূন্য solve_tri_row_scalar(ধ্রুবক দশমিক * A_row,
                                        ধ্রুবক দশমিক * B_row,
                                        দশমিক *       X,
                                        uint32_t      row,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        দশমিক         inv_diag) {
    জন্য (uint32_t col = col0; col < col0 + coln; ++col) {
        দশমিক যোগ_2 = 0.0f;
        জন্য (uint32_t t = 0; t < row; ++t) {
            যোগ_2 += A_row[t] * X[t * k + col];
        }
        X[row * k + col] = (B_row[col] - যোগ_2) * inv_diag;
    }
}

স্থির অন্তঃরেখ HVX_Vector hvx_load_partial_f32(ধ্রুবক দশমিক * src, uint32_t n) {
    HVX_Vector v = *((ধ্রুবক HVX_UVector *) src);
    HVX_VectorPred mask = Q6_Q_vsetq2_R(n * আকার(দশমিক));
    ফেরত Q6_V_vmux_QVV(mask, v, Q6_V_vzero());
}

স্থির অন্তঃরেখ শূন্য solve_tri_row_hvx(ধ্রুবক দশমিক * A_row,
                                     ধ্রুবক দশমিক * B_row,
                                     দশমিক *       X,
                                     uint32_t      row,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     দশমিক         inv_diag) {
    ধ্রুবক bool full = (coln == VLEN_FP32);

    HVX_Vector যোগ_v = Q6_V_vzero();
    জন্য (uint32_t t = 0; t < row; ++t) {
        ধ্রুবক দশমিক   a         = A_row[t];
        ধ্রুবক দশমিক * x_row_col = X + t * k + col0;

        HVX_Vector x_v = full ? *((ধ্রুবক HVX_UVector *) x_row_col) : hvx_load_partial_f32(x_row_col, coln);
        HVX_Vector a_v = hvx_vec_splat_f32(a);
        যোগ_v          = hvx_vec_add_f32_f32(যোগ_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    ধ্রুবক দশমিক * b_row_col = B_row + col0;
    দশমিক *       x_out_col = X + row * k + col0;

    HVX_Vector b_v        = full ? *((ধ্রুবক HVX_UVector *) b_row_col) : hvx_load_partial_f32(b_row_col, coln);
    HVX_Vector inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_Vector out_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, যোগ_v), inv_diag_v);
    hvx_vec_store_u((শূন্য *) x_out_col, coln * আকার(দশমিক), out_v);
}

// Batch-level thread: each job is one full batch.
স্থির শূন্য solve_tri_batch_thread_f32(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো htp_solve_tri_context * sctx = (কাঠামো htp_solve_tri_context *) তথ্য;
    কাঠামো htp_ops_context *       octx = sctx->octx;

    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];  // A
    ধ্রুবক কাঠামো htp_tensor * src1 = octx->src[1];  // B
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;     // X

    ধ্রুবক uint32_t n = src0->ne[0];
    ধ্রুবক uint32_t k = src1->ne[0];

    ধ্রুবক uint32_t ne02 = src0->ne[2];

    ধ্রুবক uint32_t col_block = VLEN_FP32;
    ধ্রুবক uint32_t k_full    = (k / col_block) * col_block;

    ধ্রুবক uint32_t start_batch = sctx->job_start + sctx->jobs_per_thread * ith;
    ধ্রুবক uint32_t end_batch   = MIN(start_batch + sctx->jobs_per_thread, sctx->job_start + sctx->যোগ_jobs);

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_batch);

    জন্য (uint32_t batch = start_batch; batch < end_batch; ++batch) {
        ধ্রুবক uint32_t i03 = batch / ne02;
        ধ্রুবক uint32_t i02 = batch - i03 * ne02;

        ধ্রুবক দশমিক * A_batch =
            (ধ্রুবক দশমিক *) ((ধ্রুবক uint8_t *) (uintptr_t) src0->তথ্য + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ধ্রুবক দশমিক * B_batch =
            (ধ্রুবক দশমিক *) ((ধ্রুবক uint8_t *) (uintptr_t) src1->তথ্য + i02 * src1->nb[2] + i03 * src1->nb[3]);
        দশমিক * X_batch = (দশমিক *) ((uint8_t *) (uintptr_t) dst->তথ্য + i02 * dst->nb[2] + i03 * dst->nb[3]);

        জন্য (uint32_t row = 0; row < n; ++row) {
            ধ্রুবক দশমিক   diag     = A_batch[row * n + row];
            ধ্রুবক দশমিক   inv_diag = 1.0f / diag;
            ধ্রুবক দশমিক * A_row    = A_batch + row * n;
            ধ্রুবক দশমিক * B_row    = B_batch + row * k;

            uint32_t col0 = 0;
            জন্য (; col0 < k_full; col0 += col_block) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, col_block, inv_diag);
            }

            যদি (col0 < k) {
                ধ্রুবক uint32_t coln = k - col0;
                যদি (coln >= 8) {
                    solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                } অন্যথা {
                    solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_batch);

    FARF(উপরের_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, start_batch, end_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
স্থির শূন্য solve_tri_chunk_thread_f32(unsigned পূর্ণসংখ্যা nth, unsigned পূর্ণসংখ্যা ith, শূন্য * তথ্য) {
    কাঠামো htp_solve_tri_context * sctx = (কাঠামো htp_solve_tri_context *) তথ্য;
    কাঠামো htp_ops_context *       octx = sctx->octx;

    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];  // A
    ধ্রুবক কাঠামো htp_tensor * src1 = octx->src[1];  // B
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;     // X

    ধ্রুবক uint32_t n = src0->ne[0];
    ধ্রুবক uint32_t k = src1->ne[0];

    ধ্রুবক uint32_t ne02 = src0->ne[2];

    ধ্রুবক uint32_t start_job = sctx->job_start + sctx->jobs_per_thread * ith;
    ধ্রুবক uint32_t end_job   = MIN(start_job + sctx->jobs_per_thread, sctx->job_start + sctx->যোগ_jobs);

    কাঠামো htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_job);

    জন্য (uint32_t job = start_job; job < end_job; ++job) {
        ধ্রুবক uint32_t batch = job / sctx->k_chunks;
        ধ্রুবক uint32_t chunk = job - batch * sctx->k_chunks;

        ধ্রুবক uint32_t i03 = batch / ne02;
        ধ্রুবক uint32_t i02 = batch - i03 * ne02;

        ধ্রুবক দশমিক * A_batch =
            (ধ্রুবক দশমিক *) ((ধ্রুবক uint8_t *) (uintptr_t) src0->তথ্য + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ধ্রুবক দশমিক * B_batch =
            (ধ্রুবক দশমিক *) ((ধ্রুবক uint8_t *) (uintptr_t) src1->তথ্য + i02 * src1->nb[2] + i03 * src1->nb[3]);
        দশমিক * X_batch = (দশমিক *) ((uint8_t *) (uintptr_t) dst->তথ্য + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ধ্রুবক uint32_t col0 = chunk * sctx->col_block;
        ধ্রুবক uint32_t coln = MIN(sctx->col_block, k - col0);

        জন্য (uint32_t row = 0; row < n; ++row) {
            ধ্রুবক দশমিক diag     = A_batch[row * n + row];
            ধ্রুবক দশমিক inv_diag = 1.0f / diag;

            ধ্রুবক দশমিক * A_row = A_batch + row * n;
            ধ্রুবক দশমিক * B_row = B_batch + row * k;

            যদি (coln >= 8) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            } অন্যথা {
                solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_job);

    FARF(উপরের_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, start_job, end_job);
}

পূর্ণসংখ্যা op_solve_tri(কাঠামো htp_ops_context * octx) {
    ধ্রুবক কাঠামো htp_tensor * src0 = octx->src[0];  // A
    ধ্রুবক কাঠামো htp_tensor * src1 = octx->src[1];  // B
    ধ্রুবক কাঠামো htp_tensor * dst  = octx->dst;     // X

    যদি (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    যদি (src0->ne[0] != src0->ne[1]) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }
    যদি (src0->ne[1] != src1->ne[1]) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }
    যদি (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }
    যদি (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        ফেরত HTP_STATUS_INVAL_PARAMS;
    }

    যদি (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        ফেরত HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুবক uint32_t k = src1->ne[0];

    ধ্রুবক uint32_t col_block     = VLEN_FP32;
    ধ্রুবক uint32_t k_chunks      = (k + col_block - 1) / col_block;
    ধ্রুবক uint32_t যোগ_batches = src0->ne[2] * src0->ne[3];
    ধ্রুবক bool     batched       = যোগ_batches >= (uint32_t) octx->n_threads;

    FARF(উপরের_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    যদি (batched) {
        uint32_t job_start = 0;
        uint32_t njobs     = যোগ_batches;

        যদি (octx->ctx->mdev.গুনতি > 1) {
            ধ্রুবক uint32_t batch_আকার = dst->nb[2];
            ধ্রুবক uint32_t batches_per_chunk = (batch_আকার > 0) ? (HEX_L2_সারি_আকার / hex_gcd_u32(batch_আকার, HEX_L2_সারি_আকার)) : 1;
            ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(যোগ_batches, htp_tensor_mdev_তথ্য_aligned(dst) ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
            job_start = range.start;
            njobs     = range.গুনতি;
        }

        যদি (njobs == 0) {
            ফেরত HTP_STATUS_OK;
        }

        // Batch-level parallelism
        ধ্রুবক uint32_t n_threads = octx->n_threads;

        কাঠামো htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .যোগ_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_batch_thread_f32, &sctx, n_threads);
    } অন্যথা {
        // Chunk-level parallelism
        ধ্রুবক uint32_t যোগ_jobs = যোগ_batches * k_chunks;

        uint32_t job_start = 0;
        uint32_t njobs     = যোগ_jobs;

        যদি (octx->ctx->mdev.গুনতি > 1) {
            ধ্রুবক bool can_split = htp_tensor_mdev_তথ্য_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_সারি_আকার - 1)) == 0);
            ধ্রুবক কাঠামো htp_tensor_mdev_range range = htp_tensor_mdev_partition(যোগ_jobs, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
            job_start = range.start;
            njobs     = range.গুনতি;
        }

        যদি (njobs == 0) {
            ফেরত HTP_STATUS_OK;
        }

        ধ্রুবক uint32_t n_threads = octx->n_threads;

        কাঠামো htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .যোগ_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_chunk_thread_f32, &sctx, n_threads);
    }

    ফেরত HTP_STATUS_OK;
}
