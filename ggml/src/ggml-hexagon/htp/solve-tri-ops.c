#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#அந்தர்பாவ <HAP_farf.h>
#அந்தர்பாவ <string.h>

#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "hex-profile.h"

#define GGML_COMMON_DECL_C
#அந்தர்பாவ "ggml-common.h"
#அந்தர்பாவ "htp-ctx.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp-tensor.h"
#அந்தர்பாவ "hvx-types.h"
#அந்தர்பாவ "hvx-utils.h"

கட்டமைப்பு htp_solve_tri_context {
    கட்டமைப்பு htp_ops_context * octx;
    uint32_t                 jobs_per_thread;
    uint32_t                 total_jobs;
    uint32_t                 job_start;
    uint32_t                 k_chunks;
    uint32_t                 col_block;
};

நிலையான உள்வரி வெறுமை solve_tri_row_scalar(மாறிலி தசம * A_row,
                                        மாறிலி தசம * B_row,
                                        தசம *       X,
                                        uint32_t      row,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        தசம         inv_diag) {
    க்கு (uint32_t col = col0; col < col0 + coln; ++col) {
        தசம sum = 0.0f;
        க்கு (uint32_t t = 0; t < row; ++t) {
            sum += A_row[t] * X[t * k + col];
        }
        X[row * k + col] = (B_row[col] - sum) * inv_diag;
    }
}

நிலையான உள்வரி HVX_Vector hvx_load_partial_f32(மாறிலி தசம * src, uint32_t n) {
    HVX_Vector v = *((மாறிலி HVX_UVector *) src);
    HVX_VectorPred mask = Q6_Q_vsetq2_R(n * அளவு(தசம));
    திரும்பு Q6_V_vmux_QVV(mask, v, Q6_V_vzero());
}

நிலையான உள்வரி வெறுமை solve_tri_row_hvx(மாறிலி தசம * A_row,
                                     மாறிலி தசம * B_row,
                                     தசம *       X,
                                     uint32_t      row,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     தசம         inv_diag) {
    மாறிலி bool full = (coln == VLEN_FP32);

    HVX_Vector sum_v = Q6_V_vzero();
    க்கு (uint32_t t = 0; t < row; ++t) {
        மாறிலி தசம   a         = A_row[t];
        மாறிலி தசம * x_row_col = X + t * k + col0;

        HVX_Vector x_v = full ? *((மாறிலி HVX_UVector *) x_row_col) : hvx_load_partial_f32(x_row_col, coln);
        HVX_Vector a_v = hvx_vec_splat_f32(a);
        sum_v          = hvx_vec_add_f32_f32(sum_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    மாறிலி தசம * b_row_col = B_row + col0;
    தசம *       x_out_col = X + row * k + col0;

    HVX_Vector b_v        = full ? *((மாறிலி HVX_UVector *) b_row_col) : hvx_load_partial_f32(b_row_col, coln);
    HVX_Vector inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_Vector out_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, sum_v), inv_diag_v);
    hvx_vec_store_u((வெறுமை *) x_out_col, coln * அளவு(தசம), out_v);
}

// Batch-level thread: each job is one full batch.
நிலையான வெறுமை solve_tri_batch_thread_f32(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு htp_solve_tri_context * sctx = (கட்டமைப்பு htp_solve_tri_context *) data;
    கட்டமைப்பு htp_ops_context *       octx = sctx->octx;

    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];  // A
    மாறிலி கட்டமைப்பு htp_tensor * src1 = octx->src[1];  // B
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;     // X

    மாறிலி uint32_t n = src0->ne[0];
    மாறிலி uint32_t k = src1->ne[0];

    மாறிலி uint32_t ne02 = src0->ne[2];

    மாறிலி uint32_t col_block = VLEN_FP32;
    மாறிலி uint32_t k_full    = (k / col_block) * col_block;

    மாறிலி uint32_t start_batch = sctx->job_start + sctx->jobs_per_thread * ith;
    மாறிலி uint32_t end_batch   = MIN(start_batch + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_batch);

    க்கு (uint32_t batch = start_batch; batch < end_batch; ++batch) {
        மாறிலி uint32_t i03 = batch / ne02;
        மாறிலி uint32_t i02 = batch - i03 * ne02;

        மாறிலி தசம * A_batch =
            (மாறிலி தசம *) ((மாறிலி uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        மாறிலி தசம * B_batch =
            (மாறிலி தசம *) ((மாறிலி uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        தசம * X_batch = (தசம *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        க்கு (uint32_t row = 0; row < n; ++row) {
            மாறிலி தசம   diag     = A_batch[row * n + row];
            மாறிலி தசம   inv_diag = 1.0f / diag;
            மாறிலி தசம * A_row    = A_batch + row * n;
            மாறிலி தசம * B_row    = B_batch + row * k;

            uint32_t col0 = 0;
            க்கு (; col0 < k_full; col0 += col_block) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, col_block, inv_diag);
            }

            எனில் (col0 < k) {
                மாறிலி uint32_t coln = k - col0;
                எனில் (coln >= 8) {
                    solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                } இல்லையெனில் {
                    solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_batch);

    FARF(உயர்வு_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, start_batch, end_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
நிலையான வெறுமை solve_tri_chunk_thread_f32(unsigned முழு nth, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு htp_solve_tri_context * sctx = (கட்டமைப்பு htp_solve_tri_context *) data;
    கட்டமைப்பு htp_ops_context *       octx = sctx->octx;

    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];  // A
    மாறிலி கட்டமைப்பு htp_tensor * src1 = octx->src[1];  // B
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;     // X

    மாறிலி uint32_t n = src0->ne[0];
    மாறிலி uint32_t k = src1->ne[0];

    மாறிலி uint32_t ne02 = src0->ne[2];

    மாறிலி uint32_t start_job = sctx->job_start + sctx->jobs_per_thread * ith;
    மாறிலி uint32_t end_job   = MIN(start_job + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_job);

    க்கு (uint32_t job = start_job; job < end_job; ++job) {
        மாறிலி uint32_t batch = job / sctx->k_chunks;
        மாறிலி uint32_t chunk = job - batch * sctx->k_chunks;

        மாறிலி uint32_t i03 = batch / ne02;
        மாறிலி uint32_t i02 = batch - i03 * ne02;

        மாறிலி தசம * A_batch =
            (மாறிலி தசம *) ((மாறிலி uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        மாறிலி தசம * B_batch =
            (மாறிலி தசம *) ((மாறிலி uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        தசம * X_batch = (தசம *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        மாறிலி uint32_t col0 = chunk * sctx->col_block;
        மாறிலி uint32_t coln = MIN(sctx->col_block, k - col0);

        க்கு (uint32_t row = 0; row < n; ++row) {
            மாறிலி தசம diag     = A_batch[row * n + row];
            மாறிலி தசம inv_diag = 1.0f / diag;

            மாறிலி தசம * A_row = A_batch + row * n;
            மாறிலி தசம * B_row = B_batch + row * k;

            எனில் (coln >= 8) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            } இல்லையெனில் {
                solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_job);

    FARF(உயர்வு_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, start_job, end_job);
}

முழு op_solve_tri(கட்டமைப்பு htp_ops_context * octx) {
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];  // A
    மாறிலி கட்டமைப்பு htp_tensor * src1 = octx->src[1];  // B
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;     // X

    எனில் (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    எனில் (src0->ne[0] != src0->ne[1]) {
        திரும்பு HTP_STATUS_INVAL_PARAMS;
    }
    எனில் (src0->ne[1] != src1->ne[1]) {
        திரும்பு HTP_STATUS_INVAL_PARAMS;
    }
    எனில் (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        திரும்பு HTP_STATUS_INVAL_PARAMS;
    }
    எனில் (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        திரும்பு HTP_STATUS_INVAL_PARAMS;
    }

    எனில் (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    மாறிலி uint32_t k = src1->ne[0];

    மாறிலி uint32_t col_block     = VLEN_FP32;
    மாறிலி uint32_t k_chunks      = (k + col_block - 1) / col_block;
    மாறிலி uint32_t total_batches = src0->ne[2] * src0->ne[3];
    மாறிலி bool     batched       = total_batches >= (uint32_t) octx->n_threads;

    FARF(உயர்வு_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    எனில் (batched) {
        uint32_t job_start = 0;
        uint32_t njobs     = total_batches;

        எனில் (octx->ctx->mdev.மொத்தம் > 1) {
            மாறிலி uint32_t batch_size = dst->nb[2];
            மாறிலி uint32_t batches_per_chunk = (batch_size > 0) ? (HEX_L2_வரி_SIZE / hex_gcd_u32(batch_size, HEX_L2_வரி_SIZE)) : 1;
            மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, htp_tensor_mdev_data_aligned(dst) ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
            job_start = range.start;
            njobs     = range.மொத்தம்;
        }

        எனில் (njobs == 0) {
            திரும்பு HTP_STATUS_OK;
        }

        // Batch-level parallelism
        மாறிலி uint32_t n_threads = octx->n_threads;

        கட்டமைப்பு htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_batch_thread_f32, &sctx, n_threads);
    } இல்லையெனில் {
        // Chunk-level parallelism
        மாறிலி uint32_t total_jobs = total_batches * k_chunks;

        uint32_t job_start = 0;
        uint32_t njobs     = total_jobs;

        எனில் (octx->ctx->mdev.மொத்தம் > 1) {
            மாறிலி bool can_split = htp_tensor_mdev_data_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_வரி_SIZE - 1)) == 0);
            மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_jobs, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
            job_start = range.start;
            njobs     = range.மொத்தம்;
        }

        எனில் (njobs == 0) {
            திரும்பு HTP_STATUS_OK;
        }

        மாறிலி uint32_t n_threads = octx->n_threads;

        கட்டமைப்பு htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_chunk_thread_f32, &sctx, n_threads);
    }

    திரும்பு HTP_STATUS_OK;
}
