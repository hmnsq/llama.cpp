#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"

#define GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "hvx-types.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"

ಸಂರಚನಾ htp_solve_tri_context {
    ಸಂರಚನಾ htp_ops_context * octx;
    uint32_t                 jobs_per_thread;
    uint32_t                 total_jobs;
    uint32_t                 job_start;
    uint32_t                 k_chunks;
    uint32_t                 col_block;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ solve_tri_row_scalar(ಧ್ರುವ ಭಿನ್ನ * A_row,
                                        ಧ್ರುವ ಭಿನ್ನ * B_row,
                                        ಭಿನ್ನ *       X,
                                        uint32_t      row,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        ಭಿನ್ನ         inv_diag) {
    ಪ್ರತಿ (uint32_t col = col0; col < col0 + coln; ++col) {
        ಭಿನ್ನ sum = 0.0f;
        ಪ್ರತಿ (uint32_t t = 0; t < row; ++t) {
            sum += A_row[t] * X[t * k + col];
        }
        X[row * k + col] = (B_row[col] - sum) * inv_diag;
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ HVX_Vector hvx_load_partial_f32(ಧ್ರುವ ಭಿನ್ನ * src, uint32_t n) {
    HVX_Vector v = *((ಧ್ರುವ HVX_UVector *) src);
    HVX_VectorPred mask = Q6_Q_vsetq2_R(n * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಪ್ರತ್ಯಾಗಮ Q6_V_vmux_QVV(mask, v, Q6_V_vzero());
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ solve_tri_row_hvx(ಧ್ರುವ ಭಿನ್ನ * A_row,
                                     ಧ್ರುವ ಭಿನ್ನ * B_row,
                                     ಭಿನ್ನ *       X,
                                     uint32_t      row,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     ಭಿನ್ನ         inv_diag) {
    ಧ್ರುವ bool full = (coln == VLEN_FP32);

    HVX_Vector sum_v = Q6_V_vzero();
    ಪ್ರತಿ (uint32_t t = 0; t < row; ++t) {
        ಧ್ರುವ ಭಿನ್ನ   a         = A_row[t];
        ಧ್ರುವ ಭಿನ್ನ * x_row_col = X + t * k + col0;

        HVX_Vector x_v = full ? *((ಧ್ರುವ HVX_UVector *) x_row_col) : hvx_load_partial_f32(x_row_col, coln);
        HVX_Vector a_v = hvx_vec_splat_f32(a);
        sum_v          = hvx_vec_add_f32_f32(sum_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    ಧ್ರುವ ಭಿನ್ನ * b_row_col = B_row + col0;
    ಭಿನ್ನ *       x_out_col = X + row * k + col0;

    HVX_Vector b_v        = full ? *((ಧ್ರುವ HVX_UVector *) b_row_col) : hvx_load_partial_f32(b_row_col, coln);
    HVX_Vector inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_Vector out_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, sum_v), inv_diag_v);
    hvx_vec_store_u((ಶೂನ್ಯ *) x_out_col, coln * ಪರಿಮಾಣ(ಭಿನ್ನ), out_v);
}

// Batch-level thread: each job is one full batch.
ಸ್ಥಿರ ಶೂನ್ಯ solve_tri_batch_thread_f32(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ htp_solve_tri_context * sctx = (ಸಂರಚನಾ htp_solve_tri_context *) data;
    ಸಂರಚನಾ htp_ops_context *       octx = sctx->octx;

    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];  // A
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src1 = octx->src[1];  // B
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst  = octx->dst;     // X

    ಧ್ರುವ uint32_t n = src0->ne[0];
    ಧ್ರುವ uint32_t k = src1->ne[0];

    ಧ್ರುವ uint32_t ne02 = src0->ne[2];

    ಧ್ರುವ uint32_t col_block = VLEN_FP32;
    ಧ್ರುವ uint32_t k_full    = (k / col_block) * col_block;

    ಧ್ರುವ uint32_t start_batch = sctx->job_start + sctx->jobs_per_thread * ith;
    ಧ್ರುವ uint32_t end_batch   = MIN(start_batch + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    ಸಂರಚನಾ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_batch);

    ಪ್ರತಿ (uint32_t batch = start_batch; batch < end_batch; ++batch) {
        ಧ್ರುವ uint32_t i03 = batch / ne02;
        ಧ್ರುವ uint32_t i02 = batch - i03 * ne02;

        ಧ್ರುವ ಭಿನ್ನ * A_batch =
            (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ಧ್ರುವ ಭಿನ್ನ * B_batch =
            (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        ಭಿನ್ನ * X_batch = (ಭಿನ್ನ *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ಪ್ರತಿ (uint32_t row = 0; row < n; ++row) {
            ಧ್ರುವ ಭಿನ್ನ   diag     = A_batch[row * n + row];
            ಧ್ರುವ ಭಿನ್ನ   inv_diag = 1.0f / diag;
            ಧ್ರುವ ಭಿನ್ನ * A_row    = A_batch + row * n;
            ಧ್ರುವ ಭಿನ್ನ * B_row    = B_batch + row * k;

            uint32_t col0 = 0;
            ಪ್ರತಿ (; col0 < k_full; col0 += col_block) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, col_block, inv_diag);
            }

            ಯದಿ (col0 < k) {
                ಧ್ರುವ uint32_t coln = k - col0;
                ಯದಿ (coln >= 8) {
                    solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                } ಅನ್ಯಥಾ {
                    solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_batch);

    FARF(ಮೇಲಿನ_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, start_batch, end_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
ಸ್ಥಿರ ಶೂನ್ಯ solve_tri_chunk_thread_f32(unsigned ಪೂರ್ಣಾಙ್ಕ nth, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ htp_solve_tri_context * sctx = (ಸಂರಚನಾ htp_solve_tri_context *) data;
    ಸಂರಚನಾ htp_ops_context *       octx = sctx->octx;

    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];  // A
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src1 = octx->src[1];  // B
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst  = octx->dst;     // X

    ಧ್ರುವ uint32_t n = src0->ne[0];
    ಧ್ರುವ uint32_t k = src1->ne[0];

    ಧ್ರುವ uint32_t ne02 = src0->ne[2];

    ಧ್ರುವ uint32_t start_job = sctx->job_start + sctx->jobs_per_thread * ith;
    ಧ್ರುವ uint32_t end_job   = MIN(start_job + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    ಸಂರಚನಾ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_job);

    ಪ್ರತಿ (uint32_t job = start_job; job < end_job; ++job) {
        ಧ್ರುವ uint32_t batch = job / sctx->k_chunks;
        ಧ್ರುವ uint32_t chunk = job - batch * sctx->k_chunks;

        ಧ್ರುವ uint32_t i03 = batch / ne02;
        ಧ್ರುವ uint32_t i02 = batch - i03 * ne02;

        ಧ್ರುವ ಭಿನ್ನ * A_batch =
            (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ಧ್ರುವ ಭಿನ್ನ * B_batch =
            (ಧ್ರುವ ಭಿನ್ನ *) ((ಧ್ರುವ uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        ಭಿನ್ನ * X_batch = (ಭಿನ್ನ *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ಧ್ರುವ uint32_t col0 = chunk * sctx->col_block;
        ಧ್ರುವ uint32_t coln = MIN(sctx->col_block, k - col0);

        ಪ್ರತಿ (uint32_t row = 0; row < n; ++row) {
            ಧ್ರುವ ಭಿನ್ನ diag     = A_batch[row * n + row];
            ಧ್ರುವ ಭಿನ್ನ inv_diag = 1.0f / diag;

            ಧ್ರುವ ಭಿನ್ನ * A_row = A_batch + row * n;
            ಧ್ರುವ ಭಿನ್ನ * B_row = B_batch + row * k;

            ಯದಿ (coln >= 8) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            } ಅನ್ಯಥಾ {
                solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_job);

    FARF(ಮೇಲಿನ_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, start_job, end_job);
}

ಪೂರ್ಣಾಙ್ಕ op_solve_tri(ಸಂರಚನಾ htp_ops_context * octx) {
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];  // A
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src1 = octx->src[1];  // B
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst  = octx->dst;     // X

    ಯದಿ (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    ಯದಿ (src0->ne[0] != src0->ne[1]) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_INVAL_PARAMS;
    }
    ಯದಿ (src0->ne[1] != src1->ne[1]) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_INVAL_PARAMS;
    }
    ಯದಿ (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_INVAL_PARAMS;
    }
    ಯದಿ (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_INVAL_PARAMS;
    }

    ಯದಿ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಧ್ರುವ uint32_t k = src1->ne[0];

    ಧ್ರುವ uint32_t col_block     = VLEN_FP32;
    ಧ್ರುವ uint32_t k_chunks      = (k + col_block - 1) / col_block;
    ಧ್ರುವ uint32_t total_batches = src0->ne[2] * src0->ne[3];
    ಧ್ರುವ bool     batched       = total_batches >= (uint32_t) octx->n_threads;

    FARF(ಮೇಲಿನ_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    ಯದಿ (batched) {
        uint32_t job_start = 0;
        uint32_t njobs     = total_batches;

        ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1) {
            ಧ್ರುವ uint32_t batch_size = dst->nb[2];
            ಧ್ರುವ uint32_t batches_per_chunk = (batch_size > 0) ? (HEX_L2_ಸಾಲು_SIZE / hex_gcd_u32(batch_size, HEX_L2_ಸಾಲು_SIZE)) : 1;
            ಧ್ರುವ ಸಂರಚನಾ htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, htp_tensor_mdev_data_aligned(dst) ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
            job_start = range.start;
            njobs     = range.ಎಣಿಕೆ;
        }

        ಯದಿ (njobs == 0) {
            ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
        }

        // Batch-level parallelism
        ಧ್ರುವ uint32_t n_threads = octx->n_threads;

        ಸಂರಚನಾ htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_batch_thread_f32, &sctx, n_threads);
    } ಅನ್ಯಥಾ {
        // Chunk-level parallelism
        ಧ್ರುವ uint32_t total_jobs = total_batches * k_chunks;

        uint32_t job_start = 0;
        uint32_t njobs     = total_jobs;

        ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1) {
            ಧ್ರುವ bool can_split = htp_tensor_mdev_data_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_ಸಾಲು_SIZE - 1)) == 0);
            ಧ್ರುವ ಸಂರಚನಾ htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_jobs, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
            job_start = range.start;
            njobs     = range.ಎಣಿಕೆ;
        }

        ಯದಿ (njobs == 0) {
            ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
        }

        ಧ್ರುವ uint32_t n_threads = octx->n_threads;

        ಸಂರಚನಾ htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_chunk_thread_f32, &sctx, n_threads);
    }

    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
}
