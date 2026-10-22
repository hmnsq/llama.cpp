#رهنمود clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <string.h>

#ضمن "hex-common.h"
#ضمن "hex-profile.h"

#تعریف_کن GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-types.h"
#ضمن "hvx-utils.h"

بنية htp_solve_tri_context {
    بنية htp_ops_context * octx;
    uint32_t                 jobs_در_هر_thread;
    uint32_t                 مجموع_jobs;
    uint32_t                 کار_شروع;
    uint32_t                 k_chunks;
    uint32_t                 ستون_بلوک;
};

ساكن مضمن فراغ solve_tri_ردیف_scalar(ثابت عائم * A_ردیف,
                                        ثابت عائم * B_ردیف,
                                        عائم *       X,
                                        uint32_t      ردیف,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        عائم         inv_diag) {
    لكل (uint32_t ستون = col0; ستون < col0 + coln; ++ستون) {
        عائم جمع = 0.0f;
        لكل (uint32_t t = 0; t < ردیف; ++t) {
            جمع += A_ردیف[t] * X[t * k + ستون];
        }
        X[ردیف * k + ستون] = (B_ردیف[ستون] - جمع) * inv_diag;
    }
}

ساكن مضمن HVX_وکتور hvx_بارگیری_partial_f32(ثابت عائم * src, uint32_t n) {
    HVX_وکتور v = *((ثابت HVX_UVector *) src);
    HVX_وکتور_Pred ماسک = Q6_Q_vsetq2_R(n * حجم(عائم));
    أرجع Q6_V_vmux_QVV(ماسک, v, Q6_V_vzero());
}

ساكن مضمن فراغ solve_tri_ردیف_hvx(ثابت عائم * A_ردیف,
                                     ثابت عائم * B_ردیف,
                                     عائم *       X,
                                     uint32_t      ردیف,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     عائم         inv_diag) {
    ثابت bool پر = (coln == VLEN_FP32);

    HVX_وکتور جمع_v = Q6_V_vzero();
    لكل (uint32_t t = 0; t < ردیف; ++t) {
        ثابت عائم   a         = A_ردیف[t];
        ثابت عائم * x_ردیف_ستون = X + t * k + col0;

        HVX_وکتور x_v = پر ? *((ثابت HVX_UVector *) x_ردیف_ستون) : hvx_بارگیری_partial_f32(x_ردیف_ستون, coln);
        HVX_وکتور a_v = hvx_vec_splat_f32(a);
        جمع_v          = hvx_vec_جمع_کن_f32_f32(جمع_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    ثابت عائم * b_ردیف_ستون = B_ردیف + col0;
    عائم *       x_بیرون_ستون = X + ردیف * k + col0;

    HVX_وکتور b_v        = پر ? *((ثابت HVX_UVector *) b_ردیف_ستون) : hvx_بارگیری_partial_f32(b_ردیف_ستون, coln);
    HVX_وکتور inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_وکتور بیرون_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, جمع_v), inv_diag_v);
    hvx_vec_store_u((فراغ *) x_بیرون_ستون, coln * حجم(عائم), بیرون_v);
}

// Batch-level thread: each job is one full batch.
ساكن فراغ solve_tri_batch_thread_f32(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_solve_tri_context * sctx = (بنية htp_solve_tri_context *) داده;
    بنية htp_ops_context *       octx = sctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];  // A
    ثابت بنية htp_tensor * src1 = octx->src[1];  // B
    ثابت بنية htp_tensor * dst  = octx->dst;     // X

    ثابت uint32_t n = src0->ne[0];
    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ne02 = src0->ne[2];

    ثابت uint32_t ستون_بلوک = VLEN_FP32;
    ثابت uint32_t k_پر    = (k / ستون_بلوک) * ستون_بلوک;

    ثابت uint32_t شروع_batch = sctx->کار_شروع + sctx->jobs_در_هر_thread * ith;
    ثابت uint32_t انتها_batch   = MIN(شروع_batch + sctx->jobs_در_هر_thread, sctx->کار_شروع + sctx->مجموع_jobs);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع_batch);

    لكل (uint32_t batch = شروع_batch; batch < انتها_batch; ++batch) {
        ثابت uint32_t i03 = batch / ne02;
        ثابت uint32_t i02 = batch - i03 * ne02;

        ثابت عائم * A_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src0->داده + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ثابت عائم * B_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src1->داده + i02 * src1->nb[2] + i03 * src1->nb[3]);
        عائم * X_batch = (عائم *) ((uint8_t *) (uintptr_t) dst->داده + i02 * dst->nb[2] + i03 * dst->nb[3]);

        لكل (uint32_t ردیف = 0; ردیف < n; ++ردیف) {
            ثابت عائم   diag     = A_batch[ردیف * n + ردیف];
            ثابت عائم   inv_diag = 1.0f / diag;
            ثابت عائم * A_ردیف    = A_batch + ردیف * n;
            ثابت عائم * B_ردیف    = B_batch + ردیف * k;

            uint32_t col0 = 0;
            لكل (; col0 < k_پر; col0 += ستون_بلوک) {
                solve_tri_ردیف_hvx(A_ردیف, B_ردیف, X_batch, ردیف, k, col0, ستون_بلوک, inv_diag);
            }

            إذا (col0 < k) {
                ثابت uint32_t coln = k - col0;
                إذا (coln >= 8) {
                    solve_tri_ردیف_hvx(A_ردیف, B_ردیف, X_batch, ردیف, k, col0, coln, inv_diag);
                } وإلا {
                    solve_tri_ردیف_scalar(A_ردیف, B_ردیف, X_batch, ردیف, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) انتها_batch);

    FARF(بالا_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, شروع_batch, انتها_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
ساكن فراغ solve_tri_chunk_thread_f32(بیعلامت صحيح nth, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_solve_tri_context * sctx = (بنية htp_solve_tri_context *) داده;
    بنية htp_ops_context *       octx = sctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];  // A
    ثابت بنية htp_tensor * src1 = octx->src[1];  // B
    ثابت بنية htp_tensor * dst  = octx->dst;     // X

    ثابت uint32_t n = src0->ne[0];
    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ne02 = src0->ne[2];

    ثابت uint32_t شروع_کار = sctx->کار_شروع + sctx->jobs_در_هر_thread * ith;
    ثابت uint32_t انتها_کار   = MIN(شروع_کار + sctx->jobs_در_هر_thread, sctx->کار_شروع + sctx->مجموع_jobs);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع_کار);

    لكل (uint32_t کار = شروع_کار; کار < انتها_کار; ++کار) {
        ثابت uint32_t batch = کار / sctx->k_chunks;
        ثابت uint32_t chunk = کار - batch * sctx->k_chunks;

        ثابت uint32_t i03 = batch / ne02;
        ثابت uint32_t i02 = batch - i03 * ne02;

        ثابت عائم * A_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src0->داده + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ثابت عائم * B_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src1->داده + i02 * src1->nb[2] + i03 * src1->nb[3]);
        عائم * X_batch = (عائم *) ((uint8_t *) (uintptr_t) dst->داده + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ثابت uint32_t col0 = chunk * sctx->ستون_بلوک;
        ثابت uint32_t coln = MIN(sctx->ستون_بلوک, k - col0);

        لكل (uint32_t ردیف = 0; ردیف < n; ++ردیف) {
            ثابت عائم diag     = A_batch[ردیف * n + ردیف];
            ثابت عائم inv_diag = 1.0f / diag;

            ثابت عائم * A_ردیف = A_batch + ردیف * n;
            ثابت عائم * B_ردیف = B_batch + ردیف * k;

            إذا (coln >= 8) {
                solve_tri_ردیف_hvx(A_ردیف, B_ردیف, X_batch, ردیف, k, col0, coln, inv_diag);
            } وإلا {
                solve_tri_ردیف_scalar(A_ردیف, B_ردیف, X_batch, ردیف, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) انتها_کار);

    FARF(بالا_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, شروع_کار, انتها_کار);
}

صحيح op_solve_tri(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];  // A
    ثابت بنية htp_tensor * src1 = octx->src[1];  // B
    ثابت بنية htp_tensor * dst  = octx->dst;     // X

    إذا (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    إذا (src0->ne[0] != src0->ne[1]) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }
    إذا (src0->ne[1] != src1->ne[1]) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }
    إذا (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }
    إذا (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        أرجع HTP_وضعیت_INVAL_PARAMS;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ستون_بلوک     = VLEN_FP32;
    ثابت uint32_t k_chunks      = (k + ستون_بلوک - 1) / ستون_بلوک;
    ثابت uint32_t مجموع_batches = src0->ne[2] * src0->ne[3];
    ثابت bool     batched       = مجموع_batches >= (uint32_t) octx->n_نخها;

    FARF(بالا_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    إذا (batched) {
        uint32_t کار_شروع = 0;
        uint32_t njobs     = مجموع_batches;

        إذا (octx->ctx->mdev.شمار > 1) {
            ثابت uint32_t batch_اندازه = dst->nb[2];
            ثابت uint32_t batches_در_هر_chunk = (batch_اندازه > 0) ? (HEX_L2_سطر_اندازه / hex_بزرگترین_مقسوم_علیه_u32(batch_اندازه, HEX_L2_سطر_اندازه)) : 1;
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(مجموع_batches, htp_tensor_mdev_داده_aligned(dst) ? batches_در_هر_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
            کار_شروع = range.شروع;
            njobs     = range.شمار;
        }

        إذا (njobs == 0) {
            أرجع HTP_وضعیت_OK;
        }

        // Batch-level parallelism
        ثابت uint32_t n_نخها = octx->n_نخها;

        بنية htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_در_هر_thread = fastdiv(njobs + n_نخها - 1, &octx->n_نخها_div),
            .مجموع_jobs      = njobs,
            .کار_شروع       = کار_شروع,
            .k_chunks        = k_chunks,
            .ستون_بلوک       = ستون_بلوک,
        };

        work_صف_پیاپی(octx->ctx->work_صف, solve_tri_batch_thread_f32, &sctx, n_نخها);
    } وإلا {
        // Chunk-level parallelism
        ثابت uint32_t مجموع_jobs = مجموع_batches * k_chunks;

        uint32_t کار_شروع = 0;
        uint32_t njobs     = مجموع_jobs;

        إذا (octx->ctx->mdev.شمار > 1) {
            ثابت bool can_تقسیم = htp_tensor_mdev_داده_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_سطر_اندازه - 1)) == 0);
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(مجموع_jobs, can_تقسیم ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
            کار_شروع = range.شروع;
            njobs     = range.شمار;
        }

        إذا (njobs == 0) {
            أرجع HTP_وضعیت_OK;
        }

        ثابت uint32_t n_نخها = octx->n_نخها;

        بنية htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_در_هر_thread = fastdiv(njobs + n_نخها - 1, &octx->n_نخها_div),
            .مجموع_jobs      = njobs,
            .کار_شروع       = کار_شروع,
            .k_chunks        = k_chunks,
            .ستون_بلوک       = ستون_بلوک,
        };

        work_صف_پیاپی(octx->ctx->work_صف, solve_tri_chunk_thread_f32, &sctx, n_نخها);
    }

    أرجع HTP_وضعیت_OK;
}
