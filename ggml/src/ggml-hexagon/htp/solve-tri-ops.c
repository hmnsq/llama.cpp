#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#അന്തര്ഭാവ <HAP_farf.h>
#അന്തര്ഭാവ <string.h>

#അന്തര്ഭാവ "hex-common.h"
#അന്തര്ഭാവ "hex-profile.h"

#define GGML_COMMON_DECL_C
#അന്തര്ഭാവ "ggml-common.h"
#അന്തര്ഭാവ "htp-ctx.h"
#അന്തര്ഭാവ "htp-ops.h"
#അന്തര്ഭാവ "htp-tensor.h"
#അന്തര്ഭാവ "hvx-types.h"
#അന്തര്ഭാവ "hvx-utils.h"

ഘടന htp_solve_tri_context {
    ഘടന htp_ops_context * octx;
    uint32_t                 jobs_per_thread;
    uint32_t                 total_jobs;
    uint32_t                 job_start;
    uint32_t                 k_chunks;
    uint32_t                 col_block;
};

സ്ഥിതം വരിയിൽ ശൂന്യം solve_tri_row_scalar(സ്ഥിരം ദശാംശം * A_row,
                                        സ്ഥിരം ദശാംശം * B_row,
                                        ദശാംശം *       X,
                                        uint32_t      row,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        ദശാംശം         inv_diag) {
    വേണ്ടി (uint32_t col = col0; col < col0 + coln; ++col) {
        ദശാംശം sum = 0.0f;
        വേണ്ടി (uint32_t t = 0; t < row; ++t) {
            sum += A_row[t] * X[t * k + col];
        }
        X[row * k + col] = (B_row[col] - sum) * inv_diag;
    }
}

സ്ഥിതം വരിയിൽ HVX_Vector hvx_load_partial_f32(സ്ഥിരം ദശാംശം * src, uint32_t n) {
    HVX_Vector v = *((സ്ഥിരം HVX_UVector *) src);
    HVX_VectorPred mask = Q6_Q_vsetq2_R(n * വലുപ്പം(ദശാംശം));
    തിരികെ Q6_V_vmux_QVV(mask, v, Q6_V_vzero());
}

സ്ഥിതം വരിയിൽ ശൂന്യം solve_tri_row_hvx(സ്ഥിരം ദശാംശം * A_row,
                                     സ്ഥിരം ദശാംശം * B_row,
                                     ദശാംശം *       X,
                                     uint32_t      row,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     ദശാംശം         inv_diag) {
    സ്ഥിരം bool full = (coln == VLEN_FP32);

    HVX_Vector sum_v = Q6_V_vzero();
    വേണ്ടി (uint32_t t = 0; t < row; ++t) {
        സ്ഥിരം ദശാംശം   a         = A_row[t];
        സ്ഥിരം ദശാംശം * x_row_col = X + t * k + col0;

        HVX_Vector x_v = full ? *((സ്ഥിരം HVX_UVector *) x_row_col) : hvx_load_partial_f32(x_row_col, coln);
        HVX_Vector a_v = hvx_vec_splat_f32(a);
        sum_v          = hvx_vec_add_f32_f32(sum_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    സ്ഥിരം ദശാംശം * b_row_col = B_row + col0;
    ദശാംശം *       x_out_col = X + row * k + col0;

    HVX_Vector b_v        = full ? *((സ്ഥിരം HVX_UVector *) b_row_col) : hvx_load_partial_f32(b_row_col, coln);
    HVX_Vector inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_Vector out_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, sum_v), inv_diag_v);
    hvx_vec_store_u((ശൂന്യം *) x_out_col, coln * വലുപ്പം(ദശാംശം), out_v);
}

// Batch-level thread: each job is one full batch.
സ്ഥിതം ശൂന്യം solve_tri_batch_thread_f32(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    ഘടന htp_solve_tri_context * sctx = (ഘടന htp_solve_tri_context *) data;
    ഘടന htp_ops_context *       octx = sctx->octx;

    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];  // A
    സ്ഥിരം ഘടന htp_tensor * src1 = octx->src[1];  // B
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;     // X

    സ്ഥിരം uint32_t n = src0->ne[0];
    സ്ഥിരം uint32_t k = src1->ne[0];

    സ്ഥിരം uint32_t ne02 = src0->ne[2];

    സ്ഥിരം uint32_t col_block = VLEN_FP32;
    സ്ഥിരം uint32_t k_full    = (k / col_block) * col_block;

    സ്ഥിരം uint32_t start_batch = sctx->job_start + sctx->jobs_per_thread * ith;
    സ്ഥിരം uint32_t end_batch   = MIN(start_batch + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_batch);

    വേണ്ടി (uint32_t batch = start_batch; batch < end_batch; ++batch) {
        സ്ഥിരം uint32_t i03 = batch / ne02;
        സ്ഥിരം uint32_t i02 = batch - i03 * ne02;

        സ്ഥിരം ദശാംശം * A_batch =
            (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        സ്ഥിരം ദശാംശം * B_batch =
            (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        ദശാംശം * X_batch = (ദശാംശം *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        വേണ്ടി (uint32_t row = 0; row < n; ++row) {
            സ്ഥിരം ദശാംശം   diag     = A_batch[row * n + row];
            സ്ഥിരം ദശാംശം   inv_diag = 1.0f / diag;
            സ്ഥിരം ദശാംശം * A_row    = A_batch + row * n;
            സ്ഥിരം ദശാംശം * B_row    = B_batch + row * k;

            uint32_t col0 = 0;
            വേണ്ടി (; col0 < k_full; col0 += col_block) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, col_block, inv_diag);
            }

            എങ്കിൽ (col0 < k) {
                സ്ഥിരം uint32_t coln = k - col0;
                എങ്കിൽ (coln >= 8) {
                    solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                } അല്ലെങ്കിൽ {
                    solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_batch);

    FARF(മുകളിലെ_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, start_batch, end_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
സ്ഥിതം ശൂന്യം solve_tri_chunk_thread_f32(unsigned പൂർണ്ണസംഖ്യ nth, unsigned പൂർണ്ണസംഖ്യ ith, ശൂന്യം * data) {
    ഘടന htp_solve_tri_context * sctx = (ഘടന htp_solve_tri_context *) data;
    ഘടന htp_ops_context *       octx = sctx->octx;

    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];  // A
    സ്ഥിരം ഘടന htp_tensor * src1 = octx->src[1];  // B
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;     // X

    സ്ഥിരം uint32_t n = src0->ne[0];
    സ്ഥിരം uint32_t k = src1->ne[0];

    സ്ഥിരം uint32_t ne02 = src0->ne[2];

    സ്ഥിരം uint32_t start_job = sctx->job_start + sctx->jobs_per_thread * ith;
    സ്ഥിരം uint32_t end_job   = MIN(start_job + sctx->jobs_per_thread, sctx->job_start + sctx->total_jobs);

    ഘടന htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) start_job);

    വേണ്ടി (uint32_t job = start_job; job < end_job; ++job) {
        സ്ഥിരം uint32_t batch = job / sctx->k_chunks;
        സ്ഥിരം uint32_t chunk = job - batch * sctx->k_chunks;

        സ്ഥിരം uint32_t i03 = batch / ne02;
        സ്ഥിരം uint32_t i02 = batch - i03 * ne02;

        സ്ഥിരം ദശാംശം * A_batch =
            (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം uint8_t *) (uintptr_t) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3]);
        സ്ഥിരം ദശാംശം * B_batch =
            (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം uint8_t *) (uintptr_t) src1->data + i02 * src1->nb[2] + i03 * src1->nb[3]);
        ദശാംശം * X_batch = (ദശാംശം *) ((uint8_t *) (uintptr_t) dst->data + i02 * dst->nb[2] + i03 * dst->nb[3]);

        സ്ഥിരം uint32_t col0 = chunk * sctx->col_block;
        സ്ഥിരം uint32_t coln = MIN(sctx->col_block, k - col0);

        വേണ്ടി (uint32_t row = 0; row < n; ++row) {
            സ്ഥിരം ദശാംശം diag     = A_batch[row * n + row];
            സ്ഥിരം ദശാംശം inv_diag = 1.0f / diag;

            സ്ഥിരം ദശാംശം * A_row = A_batch + row * n;
            സ്ഥിരം ദശാംശം * B_row = B_batch + row * k;

            എങ്കിൽ (coln >= 8) {
                solve_tri_row_hvx(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            } അല്ലെങ്കിൽ {
                solve_tri_row_scalar(A_row, B_row, X_batch, row, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) end_job);

    FARF(മുകളിലെ_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, start_job, end_job);
}

പൂർണ്ണസംഖ്യ op_solve_tri(ഘടന htp_ops_context * octx) {
    സ്ഥിരം ഘടന htp_tensor * src0 = octx->src[0];  // A
    സ്ഥിരം ഘടന htp_tensor * src1 = octx->src[1];  // B
    സ്ഥിരം ഘടന htp_tensor * dst  = octx->dst;     // X

    എങ്കിൽ (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    എങ്കിൽ (src0->ne[0] != src0->ne[1]) {
        തിരികെ HTP_STATUS_INVAL_PARAMS;
    }
    എങ്കിൽ (src0->ne[1] != src1->ne[1]) {
        തിരികെ HTP_STATUS_INVAL_PARAMS;
    }
    എങ്കിൽ (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        തിരികെ HTP_STATUS_INVAL_PARAMS;
    }
    എങ്കിൽ (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        തിരികെ HTP_STATUS_INVAL_PARAMS;
    }

    എങ്കിൽ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        തിരികെ HTP_STATUS_NO_SUPPORT;
    }

    സ്ഥിരം uint32_t k = src1->ne[0];

    സ്ഥിരം uint32_t col_block     = VLEN_FP32;
    സ്ഥിരം uint32_t k_chunks      = (k + col_block - 1) / col_block;
    സ്ഥിരം uint32_t total_batches = src0->ne[2] * src0->ne[3];
    സ്ഥിരം bool     batched       = total_batches >= (uint32_t) octx->n_threads;

    FARF(മുകളിലെ_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    എങ്കിൽ (batched) {
        uint32_t job_start = 0;
        uint32_t njobs     = total_batches;

        എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1) {
            സ്ഥിരം uint32_t batch_size = dst->nb[2];
            സ്ഥിരം uint32_t batches_per_chunk = (batch_size > 0) ? (HEX_L2_വരി_SIZE / hex_gcd_u32(batch_size, HEX_L2_വരി_SIZE)) : 1;
            സ്ഥിരം ഘടന htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_batches, htp_tensor_mdev_data_aligned(dst) ? batches_per_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.എണ്ണം, &octx->ctx->mdev.എണ്ണം_div);
            job_start = range.start;
            njobs     = range.എണ്ണം;
        }

        എങ്കിൽ (njobs == 0) {
            തിരികെ HTP_STATUS_OK;
        }

        // Batch-level parallelism
        സ്ഥിരം uint32_t n_threads = octx->n_threads;

        ഘടന htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_batch_thread_f32, &sctx, n_threads);
    } അല്ലെങ്കിൽ {
        // Chunk-level parallelism
        സ്ഥിരം uint32_t total_jobs = total_batches * k_chunks;

        uint32_t job_start = 0;
        uint32_t njobs     = total_jobs;

        എങ്കിൽ (octx->ctx->mdev.എണ്ണം > 1) {
            സ്ഥിരം bool can_split = htp_tensor_mdev_data_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_വരി_SIZE - 1)) == 0);
            സ്ഥിരം ഘടന htp_tensor_mdev_range range = htp_tensor_mdev_partition(total_jobs, can_split ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.എണ്ണം, &octx->ctx->mdev.എണ്ണം_div);
            job_start = range.start;
            njobs     = range.എണ്ണം;
        }

        എങ്കിൽ (njobs == 0) {
            തിരികെ HTP_STATUS_OK;
        }

        സ്ഥിരം uint32_t n_threads = octx->n_threads;

        ഘടന htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_per_thread = fastdiv(njobs + n_threads - 1, &octx->n_threads_div),
            .total_jobs      = njobs,
            .job_start       = job_start,
            .k_chunks        = k_chunks,
            .col_block       = col_block,
        };

        work_queue_run(octx->ctx->work_queue, solve_tri_chunk_thread_f32, &sctx, n_threads);
    }

    തിരികെ HTP_STATUS_OK;
}
