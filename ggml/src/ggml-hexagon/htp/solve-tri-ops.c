#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <string.h>

#شامل "hex-common.h"
#شامل "hex-profile.h"

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "hvx-types.h"
#شامل "hvx-utils.h"

ساخت htp_solve_tri_context {
    ساخت htp_ops_context * octx;
    uint32_t                 jobs_فی_thread;
    uint32_t                 کل_jobs;
    uint32_t                 کام_آغاز;
    uint32_t                 k_chunks;
    uint32_t                 ستون_بلاک;
};

ساکن اندرونی خالی solve_tri_صف_scalar(ثابت اعشاری * A_صف,
                                        ثابت اعشاری * B_صف,
                                        اعشاری *       X,
                                        uint32_t      صف,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        اعشاری         inv_diag) {
    برائے (uint32_t ستون = col0; ستون < col0 + coln; ++ستون) {
        اعشاری مجموع = 0.0f;
        برائے (uint32_t t = 0; t < صف; ++t) {
            مجموع += A_صف[t] * X[t * k + ستون];
        }
        X[صف * k + ستون] = (B_صف[ستون] - مجموع) * inv_diag;
    }
}

ساکن اندرونی HVX_سمتیہ hvx_بھرو_partial_f32(ثابت اعشاری * src, uint32_t n) {
    HVX_سمتیہ v = *((ثابت HVX_UVector *) src);
    HVX_سمتیہ_Pred نقاب = Q6_Q_vsetq2_R(n * حجم(اعشاری));
    واپس Q6_V_vmux_QVV(نقاب, v, Q6_V_vzero());
}

ساکن اندرونی خالی solve_tri_صف_hvx(ثابت اعشاری * A_صف,
                                     ثابت اعشاری * B_صف,
                                     اعشاری *       X,
                                     uint32_t      صف,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     اعشاری         inv_diag) {
    ثابت bool بھرا = (coln == VLEN_FP32);

    HVX_سمتیہ مجموع_v = Q6_V_vzero();
    برائے (uint32_t t = 0; t < صف; ++t) {
        ثابت اعشاری   a         = A_صف[t];
        ثابت اعشاری * x_صف_ستون = X + t * k + col0;

        HVX_سمتیہ x_v = بھرا ? *((ثابت HVX_UVector *) x_صف_ستون) : hvx_بھرو_partial_f32(x_صف_ستون, coln);
        HVX_سمتیہ a_v = hvx_vec_splat_f32(a);
        مجموع_v          = hvx_vec_جوڑو_f32_f32(مجموع_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    ثابت اعشاری * b_صف_ستون = B_صف + col0;
    اعشاری *       x_برآمد_ستون = X + صف * k + col0;

    HVX_سمتیہ b_v        = بھرا ? *((ثابت HVX_UVector *) b_صف_ستون) : hvx_بھرو_partial_f32(b_صف_ستون, coln);
    HVX_سمتیہ inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_سمتیہ برآمد_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, مجموع_v), inv_diag_v);
    hvx_vec_store_u((خالی *) x_برآمد_ستون, coln * حجم(اعشاری), برآمد_v);
}

// Batch-level thread: each job is one full batch.
ساکن خالی solve_tri_batch_thread_f32(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_solve_tri_context * sctx = (ساخت htp_solve_tri_context *) ڈیٹا;
    ساخت htp_ops_context *       octx = sctx->octx;

    ثابت ساخت htp_tensor * src0 = octx->src[0];  // A
    ثابت ساخت htp_tensor * src1 = octx->src[1];  // B
    ثابت ساخت htp_tensor * dst  = octx->dst;     // X

    ثابت uint32_t n = src0->ne[0];
    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ne02 = src0->ne[2];

    ثابت uint32_t ستون_بلاک = VLEN_FP32;
    ثابت uint32_t k_بھرا    = (k / ستون_بلاک) * ستون_بلاک;

    ثابت uint32_t آغاز_batch = sctx->کام_آغاز + sctx->jobs_فی_thread * ith;
    ثابت uint32_t انتہا_batch   = MIN(آغاز_batch + sctx->jobs_فی_thread, sctx->کام_آغاز + sctx->کل_jobs);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) آغاز_batch);

    برائے (uint32_t batch = آغاز_batch; batch < انتہا_batch; ++batch) {
        ثابت uint32_t i03 = batch / ne02;
        ثابت uint32_t i02 = batch - i03 * ne02;

        ثابت اعشاری * A_batch =
            (ثابت اعشاری *) ((ثابت uint8_t *) (uintptr_t) src0->ڈیٹا + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ثابت اعشاری * B_batch =
            (ثابت اعشاری *) ((ثابت uint8_t *) (uintptr_t) src1->ڈیٹا + i02 * src1->nb[2] + i03 * src1->nb[3]);
        اعشاری * X_batch = (اعشاری *) ((uint8_t *) (uintptr_t) dst->ڈیٹا + i02 * dst->nb[2] + i03 * dst->nb[3]);

        برائے (uint32_t صف = 0; صف < n; ++صف) {
            ثابت اعشاری   diag     = A_batch[صف * n + صف];
            ثابت اعشاری   inv_diag = 1.0f / diag;
            ثابت اعشاری * A_صف    = A_batch + صف * n;
            ثابت اعشاری * B_صف    = B_batch + صف * k;

            uint32_t col0 = 0;
            برائے (; col0 < k_بھرا; col0 += ستون_بلاک) {
                solve_tri_صف_hvx(A_صف, B_صف, X_batch, صف, k, col0, ستون_بلاک, inv_diag);
            }

            اگر (col0 < k) {
                ثابت uint32_t coln = k - col0;
                اگر (coln >= 8) {
                    solve_tri_صف_hvx(A_صف, B_صف, X_batch, صف, k, col0, coln, inv_diag);
                } ورنہ {
                    solve_tri_صف_scalar(A_صف, B_صف, X_batch, صف, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) انتہا_batch);

    FARF(بلند_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, آغاز_batch, انتہا_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
ساکن خالی solve_tri_chunk_thread_f32(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_solve_tri_context * sctx = (ساخت htp_solve_tri_context *) ڈیٹا;
    ساخت htp_ops_context *       octx = sctx->octx;

    ثابت ساخت htp_tensor * src0 = octx->src[0];  // A
    ثابت ساخت htp_tensor * src1 = octx->src[1];  // B
    ثابت ساخت htp_tensor * dst  = octx->dst;     // X

    ثابت uint32_t n = src0->ne[0];
    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ne02 = src0->ne[2];

    ثابت uint32_t آغاز_کام = sctx->کام_آغاز + sctx->jobs_فی_thread * ith;
    ثابت uint32_t انتہا_کام   = MIN(آغاز_کام + sctx->jobs_فی_thread, sctx->کام_آغاز + sctx->کل_jobs);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) آغاز_کام);

    برائے (uint32_t کام = آغاز_کام; کام < انتہا_کام; ++کام) {
        ثابت uint32_t batch = کام / sctx->k_chunks;
        ثابت uint32_t chunk = کام - batch * sctx->k_chunks;

        ثابت uint32_t i03 = batch / ne02;
        ثابت uint32_t i02 = batch - i03 * ne02;

        ثابت اعشاری * A_batch =
            (ثابت اعشاری *) ((ثابت uint8_t *) (uintptr_t) src0->ڈیٹا + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ثابت اعشاری * B_batch =
            (ثابت اعشاری *) ((ثابت uint8_t *) (uintptr_t) src1->ڈیٹا + i02 * src1->nb[2] + i03 * src1->nb[3]);
        اعشاری * X_batch = (اعشاری *) ((uint8_t *) (uintptr_t) dst->ڈیٹا + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ثابت uint32_t col0 = chunk * sctx->ستون_بلاک;
        ثابت uint32_t coln = MIN(sctx->ستون_بلاک, k - col0);

        برائے (uint32_t صف = 0; صف < n; ++صف) {
            ثابت اعشاری diag     = A_batch[صف * n + صف];
            ثابت اعشاری inv_diag = 1.0f / diag;

            ثابت اعشاری * A_صف = A_batch + صف * n;
            ثابت اعشاری * B_صف = B_batch + صف * k;

            اگر (coln >= 8) {
                solve_tri_صف_hvx(A_صف, B_صف, X_batch, صف, k, col0, coln, inv_diag);
            } ورنہ {
                solve_tri_صف_scalar(A_صف, B_صف, X_batch, صف, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) انتہا_کام);

    FARF(بلند_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, آغاز_کام, انتہا_کام);
}

صحیح op_solve_tri(ساخت htp_ops_context * octx) {
    ثابت ساخت htp_tensor * src0 = octx->src[0];  // A
    ثابت ساخت htp_tensor * src1 = octx->src[1];  // B
    ثابت ساخت htp_tensor * dst  = octx->dst;     // X

    اگر (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    اگر (src0->ne[0] != src0->ne[1]) {
        واپس HTP_حالت_INVAL_PARAMS;
    }
    اگر (src0->ne[1] != src1->ne[1]) {
        واپس HTP_حالت_INVAL_PARAMS;
    }
    اگر (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        واپس HTP_حالت_INVAL_PARAMS;
    }
    اگر (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        واپس HTP_حالت_INVAL_PARAMS;
    }

    اگر (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ستون_بلاک     = VLEN_FP32;
    ثابت uint32_t k_chunks      = (k + ستون_بلاک - 1) / ستون_بلاک;
    ثابت uint32_t کل_batches = src0->ne[2] * src0->ne[3];
    ثابت bool     batched       = کل_batches >= (uint32_t) octx->n_دھاگے;

    FARF(بلند_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    اگر (batched) {
        uint32_t کام_آغاز = 0;
        uint32_t njobs     = کل_batches;

        اگر (octx->ctx->mdev.گنتی > 1) {
            ثابت uint32_t batch_سائز = dst->nb[2];
            ثابت uint32_t batches_فی_chunk = (batch_سائز > 0) ? (HEX_L2_سطر_سائز / hex_عاد_اعظم_u32(batch_سائز, HEX_L2_سطر_سائز)) : 1;
            ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(کل_batches, htp_tensor_mdev_ڈیٹا_aligned(dst) ? batches_فی_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
            کام_آغاز = range.آغاز;
            njobs     = range.گنتی;
        }

        اگر (njobs == 0) {
            واپس HTP_حالت_OK;
        }

        // Batch-level parallelism
        ثابت uint32_t n_دھاگے = octx->n_دھاگے;

        ساخت htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_فی_thread = fastdiv(njobs + n_دھاگے - 1, &octx->n_دھاگے_div),
            .کل_jobs      = njobs,
            .کام_آغاز       = کام_آغاز,
            .k_chunks        = k_chunks,
            .ستون_بلاک       = ستون_بلاک,
        };

        work_قطار_تسلسل(octx->ctx->work_قطار, solve_tri_batch_thread_f32, &sctx, n_دھاگے);
    } ورنہ {
        // Chunk-level parallelism
        ثابت uint32_t کل_jobs = کل_batches * k_chunks;

        uint32_t کام_آغاز = 0;
        uint32_t njobs     = کل_jobs;

        اگر (octx->ctx->mdev.گنتی > 1) {
            ثابت bool can_بانٹو = htp_tensor_mdev_ڈیٹا_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_سطر_سائز - 1)) == 0);
            ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(کل_jobs, can_بانٹو ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
            کام_آغاز = range.آغاز;
            njobs     = range.گنتی;
        }

        اگر (njobs == 0) {
            واپس HTP_حالت_OK;
        }

        ثابت uint32_t n_دھاگے = octx->n_دھاگے;

        ساخت htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_فی_thread = fastdiv(njobs + n_دھاگے - 1, &octx->n_دھاگے_div),
            .کل_jobs      = njobs,
            .کام_آغاز       = کام_آغاز,
            .k_chunks        = k_chunks,
            .ستون_بلاک       = ستون_بلاک,
        };

        work_قطار_تسلسل(octx->ctx->work_قطار, solve_tri_chunk_thread_f32, &sctx, n_دھاگے);
    }

    واپس HTP_حالت_OK;
}
