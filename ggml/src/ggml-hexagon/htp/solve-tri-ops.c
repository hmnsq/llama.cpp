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

সংরচনা htp_solve_tri_context {
    সংরচনা htp_ops_context * octx;
    uint32_t                 jobs_per_thread;
    uint32_t                 total_jobs;
    uint32_t                 job_start;
    uint32_t                 k_chunks;
    uint32_t                 col_block;
};

স্থির অন্তঃস্থ শূন্য solve_tri_row_scalar(ধ্রুব ভিন্ন * A_row,
                                        ধ্রুব ভিন্ন * B_row,
                                        ভিন্ন *       X,
                                        uint32_t      row,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        ভিন্ন         inv_diag) {
    প্রতি (uint32_t col = col0; col < col0 + coln; ++col) {
        ভিন্ন sum = 0.0f;
        প্রতি (uint32_t t = 0; t < row; ++t) {
            sum += A_row[t] * X[t * k + col];
        }
        X[row * k + col] = (B_row[col] - sum) * inv_diag;
    }
}

স্থির অন্তঃস্থ HVX_Vector hvx_load_partial_f32(ধ্রুব ভিন্ন * src, uint32_t n) {
    HVX_Vector v = *((ধ্রুব HVX_UVector *) src);
    HVX_VectorPred mask = Q6_Q_vsetq2_R(n * পরিমাণ(ভিন্ন));
    প্রত্যাগম Q6_V_vmux_QVV(mask, v, Q6_V_vzero());
}

স্থির অন্তঃস্থ শূন্য solve_tri_row_hvx(ধ্রুব ভিন্ন * A_row,
                                     ধ্রুব ভিন্ন * B_row,
                                     ভিন্ন *       X,
                                     uint32_t      row,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     ভিন্ন         inv_diag) {
    ধ্রুব bool full = (coln == VLEN_FP32);

    HVX_Vector sum_v = Q6_V_vzero();
    প্রতি (uint32_t t = 0; t < row; ++t) {
        ধ্রুব ভিন্ন   a         = A_row[t];
        ধ্রুব ভিন্ন * x_row_col = X + t * k + col0;

        HVX_Vector x_v = full ? *((ধ্রুব HVX_UVector *) x_row_col) : hvx_load_partial_f32(x_row_col, coln);
        HVX_Vector a_v = hvx_vec_splat_f32(a);
        sum_v          = hvx_vec_add_f32_f32(sum_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    ধ্রুব ভিন্ন * b_row_col = B_row + col0;
    ভিন্ন *       x_out_col = X + row * k + col0;

    HVX_Vector b_v        = full ? *((ধ্রুব HVX_UVector *) b_row_col) : hvx_load_partial_f32(b_row_col, coln);
    HVX_Vector inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_Vector out_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, sum_v), inv_diag_v);
    hvx_vec_store_u((শূন্য *) x_out_col, coln * পরিমাণ(ভিন্ন), out_v);
}

// Batch-level thread: each job is one full batch.
স্থির শূন্য solve_tri_batch_thread_f32(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা htp_solve_tri_context * sctx = (সংরচনা htp_solve_tri_context *) data;
    সংরচনা htp_ops_context *       octx = sctx->octx;

    ধ্রুব সংরচনা htp_tensor * src0 = octx->src[0];  // A
    ধ্রুব সংরচনা htp_tensor * src1 = octx->src[1];  // B
    ধ্রুব সংরচনা htp_tensor * dst  = octx->dst;     // X

    ধ্রুব uint32_t n = src0->ne[0];
    ধ্রুব uint32_t k = src1->ne[0];

    ধ্রুব uint32_t ne02 = src0->ne[2];

    ধ্রুব uint32_t col_block = VLEN_FP32;
    ধ্রুব uint32_t k_full    = (k / col_block) * col_block;

    ধ্রুব uint32_t start_batch = sctx->job_start + sctx->jobs_per_thread * ith;
    ধ্রুব uint32_t end_batch   = MIN(start_batch + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_batch);

    প্রতি (uint32_t batch = start_batch; batch < end_batch; ++batch) {
        ধ্রুব uint32_t i03 = batch / ne02;
        ধ্রুব uint32_t i02 = batch - i03 * ne02;

        ধ্রুব ভিন্ন * A_batch =
            (ধ্রুব ভিন্ন *) ((ধ্রুব uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ধ্রুব ভিন্ন * B_batch =
            (ধ্রুব ভিন্ন *) ((ধ্রুব uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        ভিন্ন * X_batch = (ভিন্ন *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        প্রতি (uint32_t row = 0; row < n; ++row) {
            ধ্রুব ভিন্ন   diag     = A_batch[row * n + row];
            ধ্রুব ভিন্ন   inv_diag = 1.0f / diag;
            ধ্রুব ভিন্ন * A_row    = A_batch + row * n;
            ধ্রুব ভিন্ন * B_row    = B_batch + row * k;

            uint32_t col0 = 0;
            প্রতি (; col0 < k_full; col0 += col_block) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, col_block, inv_diag);
            }

            যদি (col0 < k) {
                ধ্রুব uint32_t coln = k - col0;
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
স্থির শূন্য solve_tri_chunk_thread_f32(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা htp_solve_tri_context * sctx = (সংরচনা htp_solve_tri_context *) data;
    সংরচনা htp_ops_context *       octx = sctx->octx;

    ধ্রুব সংরচনা htp_tensor * src0 = octx->src[0];  // A
    ধ্রুব সংরচনা htp_tensor * src1 = octx->src[1];  // B
    ধ্রুব সংরচনা htp_tensor * dst  = octx->dst;     // X

    ধ্রুব uint32_t n = src0->ne[0];
    ধ্রুব uint32_t k = src1->ne[0];

    ধ্রুব uint32_t ne02 = src0->ne[2];

    ধ্রুব uint32_t start_job = sctx->job_start + sctx->jobs_per_thread * ith;
    ধ্রুব uint32_t end_job   = MIN(start_job + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_job);

    প্রতি (uint32_t job = start_job; job < end_job; ++job) {
        ধ্রুব uint32_t batch = job / sctx->k_chunks;
        ধ্রুব uint32_t chunk = job - batch * sctx->k_chunks;

        ধ্রুব uint32_t i03 = batch / ne02;
        ধ্রুব uint32_t i02 = batch - i03 * ne02;

        ধ্রুব ভিন্ন * A_batch =
            (ধ্রুব ভিন্ন *) ((ধ্রুব uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ধ্রুব ভিন্ন * B_batch =
            (ধ্রুব ভিন্ন *) ((ধ্রুব uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        ভিন্ন * X_batch = (ভিন্ন *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ধ্রুব uint32_t col0 = chunk * sctx->col_block;
        ধ্রুব uint32_t coln = MIN(sctx->col_block, k - col0);

        প্রতি (uint32_t row = 0; row < n; ++row) {
            ধ্রুব ভিন্ন diag     = A_batch[row * n + row];
            ধ্রুব ভিন্ন inv_diag = 1.0f / diag;

            ধ্রুব ভিন্ন * A_row = A_batch + row * n;
            ধ্রুব ভিন্ন * B_row = B_batch + row * k;

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

পূর্ণাঙ্ক op_solve_tri(সংরচনা htp_ops_context * octx) {
    ধ্রুব সংরচনা htp_tensor * src0 = octx->src[0];  // A
    ধ্রুব সংরচনা htp_tensor * src1 = octx->src[1];  // B
    ধ্রুব সংরচনা htp_tensor * dst  = octx->dst;     // X

    যদি (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    যদি (src0->ne[0] != src0->ne[1]) {
        প্রত্যাগম HTP_STATUS_INVAL_PARAMS;
    }
    যদি (src0->ne[1] != src1->ne[1]) {
        প্রত্যাগম HTP_STATUS_INVAL_PARAMS;
    }
    যদি (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        প্রত্যাগম HTP_STATUS_INVAL_PARAMS;
    }
    যদি (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        প্রত্যাগম HTP_STATUS_INVAL_PARAMS;
    }

    যদি (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুব uint32_t k = src1->ne[0];

    ধ্রুব uint32_t col_block     = VLEN_FP32;
    ধ্রুব uint32_t k_chunks      = (k + col_block - 1) / col_block;
    ধ্রুব uint32_t total_batches = src0->ne[2] * src0->ne[3];
    ধ্রুব bool     batched       = total_batches >= (uint32_t) octx->n_threads;

    FARF(উপরের_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    যদি (batched) {
        uint32_t job_start = 0;
        uint32_t njobs     = total_batches;

        যদি (octx->ctx->mdev.গুনতি > 1) {
            ধ্রুব uint32_t batch_size = dst->nb[2];
            ধ্রুব uint32_t batches_per_chunk = (batch_size > 0) ? (HEX_L2_সারি_SIZE / hex_gcd_u32(batch_size, HEX_L2_সারি_SIZE)) : 1;
            ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, htp_tensor_mdev_data_aligned(dst) ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
            job_start = range.start;
            njobs     = range.গুনতি;
        }

        যদি (njobs == 0) {
            প্রত্যাগম HTP_STATUS_OK;
        }

        // Batch-level parallelism
        ধ্রুব uint32_t n_threads = octx->n_threads;

        সংরচনা htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_batch_thread_f32, &sctx, n_threads);
    } অন্যথা {
        // Chunk-level parallelism
        ধ্রুব uint32_t total_jobs = total_batches * k_chunks;

        uint32_t job_start = 0;
        uint32_t njobs     = total_jobs;

        যদি (octx->ctx->mdev.গুনতি > 1) {
            ধ্রুব bool can_split = htp_tensor_mdev_data_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_সারি_SIZE - 1)) == 0);
            ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_jobs, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
            job_start = range.start;
            njobs     = range.গুনতি;
        }

        যদি (njobs == 0) {
            প্রত্যাগম HTP_STATUS_OK;
        }

        ধ্রুব uint32_t n_threads = octx->n_threads;

        সংরচনা htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_chunk_thread_f32, &sctx, n_threads);
    }

    প্রত্যাগম HTP_STATUS_OK;
}
