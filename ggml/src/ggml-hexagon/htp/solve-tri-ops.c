#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <string.h>

#ضمن "hex-common.h"
#ضمن "hex-profile.h"

#تعریف_کړه GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-types.h"
#ضمن "hvx-utils.h"

بنية htp_solve_tri_context {
    بنية htp_ops_context * octx;
    uint32_t                 jobs_فی_thread;
    uint32_t                 ټول_jobs;
    uint32_t                 دنده_پیل;
    uint32_t                 k_chunks;
    uint32_t                 ستون_بلاک;
};

ساكن مضمن فراغ solve_tri_لیکه_scalar(ثابت عائم * A_لیکه,
                                        ثابت عائم * B_لیکه,
                                        عائم *       X,
                                        uint32_t      لیکه,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        عائم         inv_diag) {
    لكل (uint32_t ستون = col0; ستون < col0 + coln; ++ستون) {
        عائم مجموعه = 0.0f;
        لكل (uint32_t t = 0; t < لیکه; ++t) {
            مجموعه += A_لیکه[t] * X[t * k + ستون];
        }
        X[لیکه * k + ستون] = (B_لیکه[ستون] - مجموعه) * inv_diag;
    }
}

ساكن مضمن HVX_ویکټور hvx_بارول_partial_f32(ثابت عائم * src, uint32_t n) {
    HVX_ویکټور v = *((ثابت HVX_UVector *) src);
    HVX_ویکټور_Pred ماسک = Q6_Q_vsetq2_R(n * حجم(عائم));
    أرجع Q6_V_vmux_QVV(ماسک, v, Q6_V_vzero());
}

ساكن مضمن فراغ solve_tri_لیکه_hvx(ثابت عائم * A_لیکه,
                                     ثابت عائم * B_لیکه,
                                     عائم *       X,
                                     uint32_t      لیکه,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     عائم         inv_diag) {
    ثابت bool ډک = (coln == VLEN_FP32);

    HVX_ویکټور مجموعه_v = Q6_V_vzero();
    لكل (uint32_t t = 0; t < لیکه; ++t) {
        ثابت عائم   a         = A_لیکه[t];
        ثابت عائم * x_لیکه_ستون = X + t * k + col0;

        HVX_ویکټور x_v = ډک ? *((ثابت HVX_UVector *) x_لیکه_ستون) : hvx_بارول_partial_f32(x_لیکه_ستون, coln);
        HVX_ویکټور a_v = hvx_vec_splat_f32(a);
        مجموعه_v          = hvx_vec_جمع_کړه_f32_f32(مجموعه_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    ثابت عائم * b_لیکه_ستون = B_لیکه + col0;
    عائم *       x_بهر_ستون = X + لیکه * k + col0;

    HVX_ویکټور b_v        = ډک ? *((ثابت HVX_UVector *) b_لیکه_ستون) : hvx_بارول_partial_f32(b_لیکه_ستون, coln);
    HVX_ویکټور inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_ویکټور بهر_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, مجموعه_v), inv_diag_v);
    hvx_vec_store_u((فراغ *) x_بهر_ستون, coln * حجم(عائم), بهر_v);
}

// Batch-level thread: each job is one full batch.
ساكن فراغ solve_tri_batch_thread_f32(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_solve_tri_context * sctx = (بنية htp_solve_tri_context *) معلومات;
    بنية htp_ops_context *       octx = sctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];  // A
    ثابت بنية htp_tensor * src1 = octx->src[1];  // B
    ثابت بنية htp_tensor * dst  = octx->dst;     // X

    ثابت uint32_t n = src0->ne[0];
    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ne02 = src0->ne[2];

    ثابت uint32_t ستون_بلاک = VLEN_FP32;
    ثابت uint32_t k_ډک    = (k / ستون_بلاک) * ستون_بلاک;

    ثابت uint32_t پیل_batch = sctx->دنده_پیل + sctx->jobs_فی_thread * ith;
    ثابت uint32_t پای_ټکی_batch   = MIN(پیل_batch + sctx->jobs_فی_thread, sctx->دنده_پیل + sctx->ټول_jobs);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) پیل_batch);

    لكل (uint32_t batch = پیل_batch; batch < پای_ټکی_batch; ++batch) {
        ثابت uint32_t i03 = batch / ne02;
        ثابت uint32_t i02 = batch - i03 * ne02;

        ثابت عائم * A_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src0->معلومات + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ثابت عائم * B_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src1->معلومات + i02 * src1->nb[2] + i03 * src1->nb[3]);
        عائم * X_batch = (عائم *) ((uint8_t *) (uintptr_t) dst->معلومات + i02 * dst->nb[2] + i03 * dst->nb[3]);

        لكل (uint32_t لیکه = 0; لیکه < n; ++لیکه) {
            ثابت عائم   diag     = A_batch[لیکه * n + لیکه];
            ثابت عائم   inv_diag = 1.0f / diag;
            ثابت عائم * A_لیکه    = A_batch + لیکه * n;
            ثابت عائم * B_لیکه    = B_batch + لیکه * k;

            uint32_t col0 = 0;
            لكل (; col0 < k_ډک; col0 += ستون_بلاک) {
                solve_tri_لیکه_hvx(A_لیکه, B_لیکه, X_batch, لیکه, k, col0, ستون_بلاک, inv_diag);
            }

            إذا (col0 < k) {
                ثابت uint32_t coln = k - col0;
                إذا (coln >= 8) {
                    solve_tri_لیکه_hvx(A_لیکه, B_لیکه, X_batch, لیکه, k, col0, coln, inv_diag);
                } وإلا {
                    solve_tri_لیکه_scalar(A_لیکه, B_لیکه, X_batch, لیکه, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) پای_ټکی_batch);

    FARF(لوړ_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, پیل_batch, پای_ټکی_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
ساكن فراغ solve_tri_chunk_thread_f32(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_solve_tri_context * sctx = (بنية htp_solve_tri_context *) معلومات;
    بنية htp_ops_context *       octx = sctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];  // A
    ثابت بنية htp_tensor * src1 = octx->src[1];  // B
    ثابت بنية htp_tensor * dst  = octx->dst;     // X

    ثابت uint32_t n = src0->ne[0];
    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ne02 = src0->ne[2];

    ثابت uint32_t پیل_دنده = sctx->دنده_پیل + sctx->jobs_فی_thread * ith;
    ثابت uint32_t پای_ټکی_دنده   = MIN(پیل_دنده + sctx->jobs_فی_thread, sctx->دنده_پیل + sctx->ټول_jobs);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) پیل_دنده);

    لكل (uint32_t دنده = پیل_دنده; دنده < پای_ټکی_دنده; ++دنده) {
        ثابت uint32_t batch = دنده / sctx->k_chunks;
        ثابت uint32_t chunk = دنده - batch * sctx->k_chunks;

        ثابت uint32_t i03 = batch / ne02;
        ثابت uint32_t i02 = batch - i03 * ne02;

        ثابت عائم * A_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src0->معلومات + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ثابت عائم * B_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src1->معلومات + i02 * src1->nb[2] + i03 * src1->nb[3]);
        عائم * X_batch = (عائم *) ((uint8_t *) (uintptr_t) dst->معلومات + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ثابت uint32_t col0 = chunk * sctx->ستون_بلاک;
        ثابت uint32_t coln = MIN(sctx->ستون_بلاک, k - col0);

        لكل (uint32_t لیکه = 0; لیکه < n; ++لیکه) {
            ثابت عائم diag     = A_batch[لیکه * n + لیکه];
            ثابت عائم inv_diag = 1.0f / diag;

            ثابت عائم * A_لیکه = A_batch + لیکه * n;
            ثابت عائم * B_لیکه = B_batch + لیکه * k;

            إذا (coln >= 8) {
                solve_tri_لیکه_hvx(A_لیکه, B_لیکه, X_batch, لیکه, k, col0, coln, inv_diag);
            } وإلا {
                solve_tri_لیکه_scalar(A_لیکه, B_لیکه, X_batch, لیکه, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) پای_ټکی_دنده);

    FARF(لوړ_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, پیل_دنده, پای_ټکی_دنده);
}

صحيح op_solve_tri(بنية htp_ops_context * octx) {
    ثابت بنية htp_tensor * src0 = octx->src[0];  // A
    ثابت بنية htp_tensor * src1 = octx->src[1];  // B
    ثابت بنية htp_tensor * dst  = octx->dst;     // X

    إذا (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    إذا (src0->ne[0] != src0->ne[1]) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }
    إذا (src0->ne[1] != src1->ne[1]) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }
    إذا (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }
    إذا (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ستون_بلاک     = VLEN_FP32;
    ثابت uint32_t k_chunks      = (k + ستون_بلاک - 1) / ستون_بلاک;
    ثابت uint32_t ټول_batches = src0->ne[2] * src0->ne[3];
    ثابت bool     batched       = ټول_batches >= (uint32_t) octx->n_تارونه;

    FARF(لوړ_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    إذا (batched) {
        uint32_t دنده_پیل = 0;
        uint32_t njobs     = ټول_batches;

        إذا (octx->ctx->mdev.شمېر > 1) {
            ثابت uint32_t batch_اندازه = dst->nb[2];
            ثابت uint32_t batches_فی_chunk = (batch_اندازه > 0) ? (HEX_L2_کرښه_اندازه / hex_لوی_ګډ_وېشونکی_u32(batch_اندازه, HEX_L2_کرښه_اندازه)) : 1;
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(ټول_batches, htp_tensor_mdev_معلومات_aligned(dst) ? batches_فی_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
            دنده_پیل = range.پیل;
            njobs     = range.شمېر;
        }

        إذا (njobs == 0) {
            أرجع HTP_حالت_OK;
        }

        // Batch-level parallelism
        ثابت uint32_t n_تارونه = octx->n_تارونه;

        بنية htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_فی_thread = fastdiv(njobs + n_تارونه - 1, &octx->n_تارونه_div),
            .ټول_jobs      = njobs,
            .دنده_پیل       = دنده_پیل,
            .k_chunks        = k_chunks,
            .ستون_بلاک       = ستون_بلاک,
        };

        work_کتار_پرلپسې(octx->ctx->work_کتار, solve_tri_batch_thread_f32, &sctx, n_تارونه);
    } وإلا {
        // Chunk-level parallelism
        ثابت uint32_t ټول_jobs = ټول_batches * k_chunks;

        uint32_t دنده_پیل = 0;
        uint32_t njobs     = ټول_jobs;

        إذا (octx->ctx->mdev.شمېر > 1) {
            ثابت bool can_وېشل = htp_tensor_mdev_معلومات_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_کرښه_اندازه - 1)) == 0);
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(ټول_jobs, can_وېشل ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
            دنده_پیل = range.پیل;
            njobs     = range.شمېر;
        }

        إذا (njobs == 0) {
            أرجع HTP_حالت_OK;
        }

        ثابت uint32_t n_تارونه = octx->n_تارونه;

        بنية htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_فی_thread = fastdiv(njobs + n_تارونه - 1, &octx->n_تارونه_div),
            .ټول_jobs      = njobs,
            .دنده_پیل       = دنده_پیل,
            .k_chunks        = k_chunks,
            .ستون_بلاک       = ستون_بلاک,
        };

        work_کتار_پرلپسې(octx->ctx->work_کتار, solve_tri_chunk_thread_f32, &sctx, n_تارونه);
    }

    أرجع HTP_حالت_OK;
}
