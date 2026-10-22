#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <string.h>

#ضمن "hex-common.h"
#ضمن "hex-profile.h"

#تعریف_کرو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-types.h"
#ضمن "hvx-utils.h"

بنية htp_solve_tri_context {
    بنية htp_ops_context * octx;
    uint32_t                 jobs_فی_thread;
    uint32_t                 کل_jobs;
    uint32_t                 کم_شروع;
    uint32_t                 k_chunks;
    uint32_t                 خانہ_بلاک;
};

ساكن مضمن فراغ solve_tri_صف_scalar(ثابت عائم * A_صف,
                                        ثابت عائم * B_صف,
                                        عائم *       X,
                                        uint32_t      صف,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        عائم         inv_diag) {
    لكل (uint32_t خانہ = col0; خانہ < col0 + coln; ++خانہ) {
        عائم جوڑ = 0.0f;
        لكل (uint32_t t = 0; t < صف; ++t) {
            جوڑ += A_صف[t] * X[t * k + خانہ];
        }
        X[صف * k + خانہ] = (B_صف[خانہ] - جوڑ) * inv_diag;
    }
}

ساكن مضمن HVX_سمتیہ hvx_لدو_partial_f32(ثابت عائم * src, uint32_t n) {
    HVX_سمتیہ v = *((ثابت HVX_UVector *) src);
    HVX_سمتیہ_Pred نقاب = Q6_Q_vsetq2_R(n * حجم(عائم));
    أرجع Q6_V_vmux_QVV(نقاب, v, Q6_V_vzero());
}

ساكن مضمن فراغ solve_tri_صف_hvx(ثابت عائم * A_صف,
                                     ثابت عائم * B_صف,
                                     عائم *       X,
                                     uint32_t      صف,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     عائم         inv_diag) {
    ثابت bool بھریا = (coln == VLEN_FP32);

    HVX_سمتیہ جوڑ_v = Q6_V_vzero();
    لكل (uint32_t t = 0; t < صف; ++t) {
        ثابت عائم   a         = A_صف[t];
        ثابت عائم * x_صف_خانہ = X + t * k + col0;

        HVX_سمتیہ x_v = بھریا ? *((ثابت HVX_UVector *) x_صف_خانہ) : hvx_لدو_partial_f32(x_صف_خانہ, coln);
        HVX_سمتیہ a_v = hvx_vec_splat_f32(a);
        جوڑ_v          = hvx_vec_جوڑو_f32_f32(جوڑ_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    ثابت عائم * b_صف_خانہ = B_صف + col0;
    عائم *       x_باہرلا_خانہ = X + صف * k + col0;

    HVX_سمتیہ b_v        = بھریا ? *((ثابت HVX_UVector *) b_صف_خانہ) : hvx_لدو_partial_f32(b_صف_خانہ, coln);
    HVX_سمتیہ inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_سمتیہ باہرلا_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, جوڑ_v), inv_diag_v);
    hvx_vec_store_u((فراغ *) x_باہرلا_خانہ, coln * حجم(عائم), باہرلا_v);
}

// Batch-level thread: each job is one full batch.
ساكن فراغ solve_tri_batch_thread_f32(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    بنية htp_solve_tri_context * sctx = (بنية htp_solve_tri_context *) انکڑے;
    بنية htp_ops_context *       octx = sctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];  // A
    ثابت بنية htp_tensor * src1 = octx->src[1];  // B
    ثابت بنية htp_tensor * dst  = octx->dst;     // X

    ثابت uint32_t n = src0->ne[0];
    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ne02 = src0->ne[2];

    ثابت uint32_t خانہ_بلاک = VLEN_FP32;
    ثابت uint32_t k_بھریا    = (k / خانہ_بلاک) * خانہ_بلاک;

    ثابت uint32_t شروع_batch = sctx->کم_شروع + sctx->jobs_فی_thread * ith;
    ثابت uint32_t سرا_batch   = MIN(شروع_batch + sctx->jobs_فی_thread, sctx->کم_شروع + sctx->کل_jobs);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع_batch);

    لكل (uint32_t batch = شروع_batch; batch < سرا_batch; ++batch) {
        ثابت uint32_t i03 = batch / ne02;
        ثابت uint32_t i02 = batch - i03 * ne02;

        ثابت عائم * A_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src0->انکڑے + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ثابت عائم * B_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src1->انکڑے + i02 * src1->nb[2] + i03 * src1->nb[3]);
        عائم * X_batch = (عائم *) ((uint8_t *) (uintptr_t) dst->انکڑے + i02 * dst->nb[2] + i03 * dst->nb[3]);

        لكل (uint32_t صف = 0; صف < n; ++صف) {
            ثابت عائم   diag     = A_batch[صف * n + صف];
            ثابت عائم   inv_diag = 1.0f / diag;
            ثابت عائم * A_صف    = A_batch + صف * n;
            ثابت عائم * B_صف    = B_batch + صف * k;

            uint32_t col0 = 0;
            لكل (; col0 < k_بھریا; col0 += خانہ_بلاک) {
                solve_tri_صف_hvx(A_صف, B_صف, X_batch, صف, k, col0, خانہ_بلاک, inv_diag);
            }

            إذا (col0 < k) {
                ثابت uint32_t coln = k - col0;
                إذا (coln >= 8) {
                    solve_tri_صف_hvx(A_صف, B_صف, X_batch, صف, k, col0, coln, inv_diag);
                } وإلا {
                    solve_tri_صف_scalar(A_صف, B_صف, X_batch, صف, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) سرا_batch);

    FARF(بلند_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, شروع_batch, سرا_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
ساكن فراغ solve_tri_chunk_thread_f32(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    بنية htp_solve_tri_context * sctx = (بنية htp_solve_tri_context *) انکڑے;
    بنية htp_ops_context *       octx = sctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];  // A
    ثابت بنية htp_tensor * src1 = octx->src[1];  // B
    ثابت بنية htp_tensor * dst  = octx->dst;     // X

    ثابت uint32_t n = src0->ne[0];
    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ne02 = src0->ne[2];

    ثابت uint32_t شروع_کم = sctx->کم_شروع + sctx->jobs_فی_thread * ith;
    ثابت uint32_t سرا_کم   = MIN(شروع_کم + sctx->jobs_فی_thread, sctx->کم_شروع + sctx->کل_jobs);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروع_کم);

    لكل (uint32_t کم_2 = شروع_کم; کم_2 < سرا_کم; ++کم_2) {
        ثابت uint32_t batch = کم_2 / sctx->k_chunks;
        ثابت uint32_t chunk = کم_2 - batch * sctx->k_chunks;

        ثابت uint32_t i03 = batch / ne02;
        ثابت uint32_t i02 = batch - i03 * ne02;

        ثابت عائم * A_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src0->انکڑے + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ثابت عائم * B_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src1->انکڑے + i02 * src1->nb[2] + i03 * src1->nb[3]);
        عائم * X_batch = (عائم *) ((uint8_t *) (uintptr_t) dst->انکڑے + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ثابت uint32_t col0 = chunk * sctx->خانہ_بلاک;
        ثابت uint32_t coln = MIN(sctx->خانہ_بلاک, k - col0);

        لكل (uint32_t صف = 0; صف < n; ++صف) {
            ثابت عائم diag     = A_batch[صف * n + صف];
            ثابت عائم inv_diag = 1.0f / diag;

            ثابت عائم * A_صف = A_batch + صف * n;
            ثابت عائم * B_صف = B_batch + صف * k;

            إذا (coln >= 8) {
                solve_tri_صف_hvx(A_صف, B_صف, X_batch, صف, k, col0, coln, inv_diag);
            } وإلا {
                solve_tri_صف_scalar(A_صف, B_صف, X_batch, صف, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) سرا_کم);

    FARF(بلند_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, شروع_کم, سرا_کم);
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

    ثابت uint32_t خانہ_بلاک     = VLEN_FP32;
    ثابت uint32_t k_chunks      = (k + خانہ_بلاک - 1) / خانہ_بلاک;
    ثابت uint32_t کل_batches = src0->ne[2] * src0->ne[3];
    ثابت bool     batched       = کل_batches >= (uint32_t) octx->n_دھاگے;

    FARF(بلند_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    إذا (batched) {
        uint32_t کم_شروع = 0;
        uint32_t njobs     = کل_batches;

        إذا (octx->ctx->mdev.گنتی > 1) {
            ثابت uint32_t batch_ناپ = dst->nb[2];
            ثابت uint32_t batches_فی_chunk = (batch_ناپ > 0) ? (HEX_L2_سطر_ناپ / hex_عاد_اعظم_u32(batch_ناپ, HEX_L2_سطر_ناپ)) : 1;
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(کل_batches, htp_tensor_mdev_انکڑے_aligned(dst) ? batches_فی_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
            کم_شروع = range.شروع;
            njobs     = range.گنتی;
        }

        إذا (njobs == 0) {
            أرجع HTP_حالت_OK;
        }

        // Batch-level parallelism
        ثابت uint32_t n_دھاگے = octx->n_دھاگے;

        بنية htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_فی_thread = fastdiv(njobs + n_دھاگے - 1, &octx->n_دھاگے_div),
            .کل_jobs      = njobs,
            .کم_شروع       = کم_شروع,
            .k_chunks        = k_chunks,
            .خانہ_بلاک       = خانہ_بلاک,
        };

        work_قطار_لگاتار(octx->ctx->work_قطار, solve_tri_batch_thread_f32, &sctx, n_دھاگے);
    } وإلا {
        // Chunk-level parallelism
        ثابت uint32_t کل_jobs = کل_batches * k_chunks;

        uint32_t کم_شروع = 0;
        uint32_t njobs     = کل_jobs;

        إذا (octx->ctx->mdev.گنتی > 1) {
            ثابت bool can_ونڈو = htp_tensor_mdev_انکڑے_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_سطر_ناپ - 1)) == 0);
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(کل_jobs, can_ونڈو ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
            کم_شروع = range.شروع;
            njobs     = range.گنتی;
        }

        إذا (njobs == 0) {
            أرجع HTP_حالت_OK;
        }

        ثابت uint32_t n_دھاگے = octx->n_دھاگے;

        بنية htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_فی_thread = fastdiv(njobs + n_دھاگے - 1, &octx->n_دھاگے_div),
            .کل_jobs      = njobs,
            .کم_شروع       = کم_شروع,
            .k_chunks        = k_chunks,
            .خانہ_بلاک       = خانہ_بلاک,
        };

        work_قطار_لگاتار(octx->ctx->work_قطار, solve_tri_chunk_thread_f32, &sctx, n_دھاگے);
    }

    أرجع HTP_حالت_OK;
}
