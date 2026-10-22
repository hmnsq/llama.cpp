#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <string.h>

#ضمن "hex-common.h"
#ضمن "hex-profile.h"

#define GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-types.h"
#ضمن "hvx-utils.h"

بنية htp_solve_tri_context {
    بنية htp_ops_context * octx;
    uint32_t                 jobs_per_thread;
    uint32_t                 total_jobs;
    uint32_t                 job_start;
    uint32_t                 k_chunks;
    uint32_t                 col_block;
};

ساكن مضمن فراغ solve_tri_row_scalar(ثابت عائم * A_row,
                                        ثابت عائم * B_row,
                                        عائم *       X,
                                        uint32_t      row,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        عائم         inv_diag) {
    لكل (uint32_t col = col0; col < col0 + coln; ++col) {
        عائم sum = 0.0f;
        لكل (uint32_t t = 0; t < row; ++t) {
            sum += A_row[t] * X[t * k + col];
        }
        X[row * k + col] = (B_row[col] - sum) * inv_diag;
    }
}

ساكن مضمن HVX_Vector hvx_load_partial_f32(ثابت عائم * src, uint32_t n) {
    HVX_Vector v = *((ثابت HVX_UVector *) src);
    HVX_VectorPred mask = Q6_Q_vsetq2_R(n * حجم(عائم));
    أرجع Q6_V_vmux_QVV(mask, v, Q6_V_vzero());
}

ساكن مضمن فراغ solve_tri_row_hvx(ثابت عائم * A_row,
                                     ثابت عائم * B_row,
                                     عائم *       X,
                                     uint32_t      row,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     عائم         inv_diag) {
    ثابت bool full = (coln == VLEN_FP32);

    HVX_Vector sum_v = Q6_V_vzero();
    لكل (uint32_t t = 0; t < row; ++t) {
        ثابت عائم   a         = A_row[t];
        ثابت عائم * x_row_col = X + t * k + col0;

        HVX_Vector x_v = full ? *((ثابت HVX_UVector *) x_row_col) : hvx_load_partial_f32(x_row_col, coln);
        HVX_Vector a_v = hvx_vec_splat_f32(a);
        sum_v          = hvx_vec_add_f32_f32(sum_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    ثابت عائم * b_row_col = B_row + col0;
    عائم *       x_out_col = X + row * k + col0;

    HVX_Vector b_v        = full ? *((ثابت HVX_UVector *) b_row_col) : hvx_load_partial_f32(b_row_col, coln);
    HVX_Vector inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_Vector out_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, sum_v), inv_diag_v);
    hvx_vec_store_u((فراغ *) x_out_col, coln * حجم(عائم), out_v);
}

// Batch-level thread: each job is one full batch.
ساكن فراغ solve_tri_batch_thread_f32(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    بنية htp_solve_tri_context * sctx = (بنية htp_solve_tri_context *) data;
    بنية htp_ops_context *       octx = sctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];  // A
    ثابت بنية htp_tensor * src1 = octx->src[1];  // B
    ثابت بنية htp_tensor * dst  = octx->dst;     // X

    ثابت uint32_t n = src0->ne[0];
    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ne02 = src0->ne[2];

    ثابت uint32_t col_block = VLEN_FP32;
    ثابت uint32_t k_full    = (k / col_block) * col_block;

    ثابت uint32_t start_batch = sctx->job_start + sctx->jobs_per_thread * ith;
    ثابت uint32_t end_batch   = MIN(start_batch + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_batch);

    لكل (uint32_t batch = start_batch; batch < end_batch; ++batch) {
        ثابت uint32_t i03 = batch / ne02;
        ثابت uint32_t i02 = batch - i03 * ne02;

        ثابت عائم * A_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ثابت عائم * B_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        عائم * X_batch = (عائم *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        لكل (uint32_t row = 0; row < n; ++row) {
            ثابت عائم   diag     = A_batch[row * n + row];
            ثابت عائم   inv_diag = 1.0f / diag;
            ثابت عائم * A_row    = A_batch + row * n;
            ثابت عائم * B_row    = B_batch + row * k;

            uint32_t col0 = 0;
            لكل (; col0 < k_full; col0 += col_block) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, col_block, inv_diag);
            }

            إذا (col0 < k) {
                ثابت uint32_t coln = k - col0;
                إذا (coln >= 8) {
                    solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                } وإلا {
                    solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_batch);

    FARF(تيڠݢي_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, start_batch, end_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
ساكن فراغ solve_tri_chunk_thread_f32(unsigned صحيح nth, unsigned صحيح ith, فراغ * data) {
    بنية htp_solve_tri_context * sctx = (بنية htp_solve_tri_context *) data;
    بنية htp_ops_context *       octx = sctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];  // A
    ثابت بنية htp_tensor * src1 = octx->src[1];  // B
    ثابت بنية htp_tensor * dst  = octx->dst;     // X

    ثابت uint32_t n = src0->ne[0];
    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ne02 = src0->ne[2];

    ثابت uint32_t start_job = sctx->job_start + sctx->jobs_per_thread * ith;
    ثابت uint32_t end_job   = MIN(start_job + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_job);

    لكل (uint32_t job = start_job; job < end_job; ++job) {
        ثابت uint32_t batch = job / sctx->k_chunks;
        ثابت uint32_t chunk = job - batch * sctx->k_chunks;

        ثابت uint32_t i03 = batch / ne02;
        ثابت uint32_t i02 = batch - i03 * ne02;

        ثابت عائم * A_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ثابت عائم * B_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        عائم * X_batch = (عائم *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ثابت uint32_t col0 = chunk * sctx->col_block;
        ثابت uint32_t coln = MIN(sctx->col_block, k - col0);

        لكل (uint32_t row = 0; row < n; ++row) {
            ثابت عائم diag     = A_batch[row * n + row];
            ثابت عائم inv_diag = 1.0f / diag;

            ثابت عائم * A_row = A_batch + row * n;
            ثابت عائم * B_row = B_batch + row * k;

            إذا (coln >= 8) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            } وإلا {
                solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_job);

    FARF(تيڠݢي_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, start_job, end_job);
}

صحيح op_solve_tri(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];  // A
    ثابت بنية htp_tensor * src1 = octx->src[1];  // B
    ثابت بنية htp_tensor * dst  = octx->dst;     // X

    إذا (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        أرجع HTP_STATUS_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    إذا (src0->ne[0] != src0->ne[1]) {
        أرجع HTP_STATUS_INVAL_PARAMS;
    }
    إذا (src0->ne[1] != src1->ne[1]) {
        أرجع HTP_STATUS_INVAL_PARAMS;
    }
    إذا (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        أرجع HTP_STATUS_INVAL_PARAMS;
    }
    إذا (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        أرجع HTP_STATUS_INVAL_PARAMS;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        أرجع HTP_STATUS_NO_SUPPORT;
    }

    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t col_block     = VLEN_FP32;
    ثابت uint32_t k_chunks      = (k + col_block - 1) / col_block;
    ثابت uint32_t total_batches = src0->ne[2] * src0->ne[3];
    ثابت bool     batched       = total_batches >= (uint32_t) octx->n_threads;

    FARF(تيڠݢي_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    إذا (batched) {
        uint32_t job_start = 0;
        uint32_t njobs     = total_batches;

        إذا (octx->ctx->mdev.کيراءن > 1) {
            ثابت uint32_t batch_size = dst->nb[2];
            ثابت uint32_t batches_per_chunk = (batch_size > 0) ? (HEX_L2_باريس_SIZE / hex_gcd_u32(batch_size, HEX_L2_باريس_SIZE)) : 1;
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, htp_tensor_mdev_data_aligned(dst) ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.کيراءن, &octx->ctx->mdev.کيراءن_div);
            job_start = range.start;
            njobs     = range.کيراءن;
        }

        إذا (njobs == 0) {
            أرجع HTP_STATUS_OK;
        }

        // Batch-level parallelism
        ثابت uint32_t n_threads = octx->n_threads;

        بنية htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_batch_thread_f32, &sctx, n_threads);
    } وإلا {
        // Chunk-level parallelism
        ثابت uint32_t total_jobs = total_batches * k_chunks;

        uint32_t job_start = 0;
        uint32_t njobs     = total_jobs;

        إذا (octx->ctx->mdev.کيراءن > 1) {
            ثابت bool can_split = htp_tensor_mdev_data_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_باريس_SIZE - 1)) == 0);
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_jobs, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.کيراءن, &octx->ctx->mdev.کيراءن_div);
            job_start = range.start;
            njobs     = range.کيراءن;
        }

        إذا (njobs == 0) {
            أرجع HTP_STATUS_OK;
        }

        ثابت uint32_t n_threads = octx->n_threads;

        بنية htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_chunk_thread_f32, &sctx, n_threads);
    }

    أرجع HTP_STATUS_OK;
}
