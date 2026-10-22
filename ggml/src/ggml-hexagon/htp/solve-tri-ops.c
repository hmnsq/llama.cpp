#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#અન્તર્ભાવ <HAP_farf.h>
#અન્તર્ભાવ <string.h>

#અન્તર્ભાવ "hex-common.h"
#અન્તર્ભાવ "hex-profile.h"

#define GGML_COMMON_DECL_C
#અન્તર્ભાવ "ggml-common.h"
#અન્તર્ભાવ "htp-ctx.h"
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "htp-tensor.h"
#અન્તર્ભાવ "hvx-types.h"
#અન્તર્ભાવ "hvx-utils.h"

સંરચના htp_solve_tri_context {
    સંરચના htp_ops_context * octx;
    uint32_t                 jobs_per_thread;
    uint32_t                 total_jobs;
    uint32_t                 job_start;
    uint32_t                 k_chunks;
    uint32_t                 col_block;
};

સ્થિર અંતઃરેખ શૂન્ય solve_tri_row_scalar(અચલ દશાંશ * A_row,
                                        અચલ દશાંશ * B_row,
                                        દશાંશ *       X,
                                        uint32_t      row,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        દશાંશ         inv_diag) {
    માટે (uint32_t col = col0; col < col0 + coln; ++col) {
        દશાંશ sum = 0.0f;
        માટે (uint32_t t = 0; t < row; ++t) {
            sum += A_row[t] * X[t * k + col];
        }
        X[row * k + col] = (B_row[col] - sum) * inv_diag;
    }
}

સ્થિર અંતઃરેખ HVX_Vector hvx_load_partial_f32(અચલ દશાંશ * src, uint32_t n) {
    HVX_Vector v = *((અચલ HVX_UVector *) src);
    HVX_VectorPred mask = Q6_Q_vsetq2_R(n * કદ(દશાંશ));
    પાછા Q6_V_vmux_QVV(mask, v, Q6_V_vzero());
}

સ્થિર અંતઃરેખ શૂન્ય solve_tri_row_hvx(અચલ દશાંશ * A_row,
                                     અચલ દશાંશ * B_row,
                                     દશાંશ *       X,
                                     uint32_t      row,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     દશાંશ         inv_diag) {
    અચલ bool full = (coln == VLEN_FP32);

    HVX_Vector sum_v = Q6_V_vzero();
    માટે (uint32_t t = 0; t < row; ++t) {
        અચલ દશાંશ   a         = A_row[t];
        અચલ દશાંશ * x_row_col = X + t * k + col0;

        HVX_Vector x_v = full ? *((અચલ HVX_UVector *) x_row_col) : hvx_load_partial_f32(x_row_col, coln);
        HVX_Vector a_v = hvx_vec_splat_f32(a);
        sum_v          = hvx_vec_add_f32_f32(sum_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    અચલ દશાંશ * b_row_col = B_row + col0;
    દશાંશ *       x_out_col = X + row * k + col0;

    HVX_Vector b_v        = full ? *((અચલ HVX_UVector *) b_row_col) : hvx_load_partial_f32(b_row_col, coln);
    HVX_Vector inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_Vector out_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, sum_v), inv_diag_v);
    hvx_vec_store_u((શૂન્ય *) x_out_col, coln * કદ(દશાંશ), out_v);
}

// Batch-level thread: each job is one full batch.
સ્થિર શૂન્ય solve_tri_batch_thread_f32(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    સંરચના htp_solve_tri_context * sctx = (સંરચના htp_solve_tri_context *) data;
    સંરચના htp_ops_context *       octx = sctx->octx;

    અચલ સંરચના htp_tensor * src0 = octx->src[0];  // A
    અચલ સંરચના htp_tensor * src1 = octx->src[1];  // B
    અચલ સંરચના htp_tensor * dst  = octx->dst;     // X

    અચલ uint32_t n = src0->ne[0];
    અચલ uint32_t k = src1->ne[0];

    અચલ uint32_t ne02 = src0->ne[2];

    અચલ uint32_t col_block = VLEN_FP32;
    અચલ uint32_t k_full    = (k / col_block) * col_block;

    અચલ uint32_t start_batch = sctx->job_start + sctx->jobs_per_thread * ith;
    અચલ uint32_t end_batch   = MIN(start_batch + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_batch);

    માટે (uint32_t batch = start_batch; batch < end_batch; ++batch) {
        અચલ uint32_t i03 = batch / ne02;
        અચલ uint32_t i02 = batch - i03 * ne02;

        અચલ દશાંશ * A_batch =
            (અચલ દશાંશ *) ((અચલ uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        અચલ દશાંશ * B_batch =
            (અચલ દશાંશ *) ((અચલ uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        દશાંશ * X_batch = (દશાંશ *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        માટે (uint32_t row = 0; row < n; ++row) {
            અચલ દશાંશ   diag     = A_batch[row * n + row];
            અચલ દશાંશ   inv_diag = 1.0f / diag;
            અચલ દશાંશ * A_row    = A_batch + row * n;
            અચલ દશાંશ * B_row    = B_batch + row * k;

            uint32_t col0 = 0;
            માટે (; col0 < k_full; col0 += col_block) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, col_block, inv_diag);
            }

            જો (col0 < k) {
                અચલ uint32_t coln = k - col0;
                જો (coln >= 8) {
                    solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                } નહીંતર {
                    solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_batch);

    FARF(ઉપલું_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, start_batch, end_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
સ્થિર શૂન્ય solve_tri_chunk_thread_f32(unsigned પૂર્ણાંક nth, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    સંરચના htp_solve_tri_context * sctx = (સંરચના htp_solve_tri_context *) data;
    સંરચના htp_ops_context *       octx = sctx->octx;

    અચલ સંરચના htp_tensor * src0 = octx->src[0];  // A
    અચલ સંરચના htp_tensor * src1 = octx->src[1];  // B
    અચલ સંરચના htp_tensor * dst  = octx->dst;     // X

    અચલ uint32_t n = src0->ne[0];
    અચલ uint32_t k = src1->ne[0];

    અચલ uint32_t ne02 = src0->ne[2];

    અચલ uint32_t start_job = sctx->job_start + sctx->jobs_per_thread * ith;
    અચલ uint32_t end_job   = MIN(start_job + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_job);

    માટે (uint32_t job = start_job; job < end_job; ++job) {
        અચલ uint32_t batch = job / sctx->k_chunks;
        અચલ uint32_t chunk = job - batch * sctx->k_chunks;

        અચલ uint32_t i03 = batch / ne02;
        અચલ uint32_t i02 = batch - i03 * ne02;

        અચલ દશાંશ * A_batch =
            (અચલ દશાંશ *) ((અચલ uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        અચલ દશાંશ * B_batch =
            (અચલ દશાંશ *) ((અચલ uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        દશાંશ * X_batch = (દશાંશ *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        અચલ uint32_t col0 = chunk * sctx->col_block;
        અચલ uint32_t coln = MIN(sctx->col_block, k - col0);

        માટે (uint32_t row = 0; row < n; ++row) {
            અચલ દશાંશ diag     = A_batch[row * n + row];
            અચલ દશાંશ inv_diag = 1.0f / diag;

            અચલ દશાંશ * A_row = A_batch + row * n;
            અચલ દશાંશ * B_row = B_batch + row * k;

            જો (coln >= 8) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            } નહીંતર {
                solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_job);

    FARF(ઉપલું_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, start_job, end_job);
}

પૂર્ણાંક op_solve_tri(સંરચના htp_ops_context * octx) {
    અચલ સંરચના htp_tensor * src0 = octx->src[0];  // A
    અચલ સંરચના htp_tensor * src1 = octx->src[1];  // B
    અચલ સંરચના htp_tensor * dst  = octx->dst;     // X

    જો (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    જો (src0->ne[0] != src0->ne[1]) {
        પાછા HTP_STATUS_INVAL_PARAMS;
    }
    જો (src0->ne[1] != src1->ne[1]) {
        પાછા HTP_STATUS_INVAL_PARAMS;
    }
    જો (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        પાછા HTP_STATUS_INVAL_PARAMS;
    }
    જો (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        પાછા HTP_STATUS_INVAL_PARAMS;
    }

    જો (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    અચલ uint32_t k = src1->ne[0];

    અચલ uint32_t col_block     = VLEN_FP32;
    અચલ uint32_t k_chunks      = (k + col_block - 1) / col_block;
    અચલ uint32_t total_batches = src0->ne[2] * src0->ne[3];
    અચલ bool     batched       = total_batches >= (uint32_t) octx->n_threads;

    FARF(ઉપલું_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    જો (batched) {
        uint32_t job_start = 0;
        uint32_t njobs     = total_batches;

        જો (octx->ctx->mdev.ગણતરી > 1) {
            અચલ uint32_t batch_size = dst->nb[2];
            અચલ uint32_t batches_per_chunk = (batch_size > 0) ? (HEX_L2_લીટી_SIZE / hex_gcd_u32(batch_size, HEX_L2_લીટી_SIZE)) : 1;
            અચલ સંરચના htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, htp_tensor_mdev_data_aligned(dst) ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ગણતરી, &octx->ctx->mdev.ગણતરી_div);
            job_start = range.start;
            njobs     = range.ગણતરી;
        }

        જો (njobs == 0) {
            પાછા HTP_STATUS_OK;
        }

        // Batch-level parallelism
        અચલ uint32_t n_threads = octx->n_threads;

        સંરચના htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_batch_thread_f32, &sctx, n_threads);
    } નહીંતર {
        // Chunk-level parallelism
        અચલ uint32_t total_jobs = total_batches * k_chunks;

        uint32_t job_start = 0;
        uint32_t njobs     = total_jobs;

        જો (octx->ctx->mdev.ગણતરી > 1) {
            અચલ bool can_split = htp_tensor_mdev_data_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_લીટી_SIZE - 1)) == 0);
            અચલ સંરચના htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_jobs, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ગણતરી, &octx->ctx->mdev.ગણતરી_div);
            job_start = range.start;
            njobs     = range.ગણતરી;
        }

        જો (njobs == 0) {
            પાછા HTP_STATUS_OK;
        }

        અચલ uint32_t n_threads = octx->n_threads;

        સંરચના htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_chunk_thread_f32, &sctx, n_threads);
    }

    પાછા HTP_STATUS_OK;
}
