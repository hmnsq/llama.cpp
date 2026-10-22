#هدايت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <string.h>

#ضمن "hex-common.h"
#ضمن "hex-profile.h"

#وصف_ڪريو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "hvx-types.h"
#ضمن "hvx-utils.h"

بنية htp_solve_tri_context {
    بنية htp_ops_context * octx;
    uint32_t                 jobs_في_thread;
    uint32_t                 ڪل_jobs;
    uint32_t                 ڪم_شروعات;
    uint32_t                 k_chunks;
    uint32_t                 کانو_بلاڪ;
};

ساكن مضمن فراغ solve_tri_صف_scalar(ثابت عائم * A_صف,
                                        ثابت عائم * B_صف,
                                        عائم *       X,
                                        uint32_t      صف,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        عائم         inv_diag) {
    لكل (uint32_t کانو = col0; کانو < col0 + coln; ++کانو) {
        عائم جوڙ = 0.0f;
        لكل (uint32_t t = 0; t < صف; ++t) {
            جوڙ += A_صف[t] * X[t * k + کانو];
        }
        X[صف * k + کانو] = (B_صف[کانو] - جوڙ) * inv_diag;
    }
}

ساكن مضمن HVX_ويڪٽر hvx_ڀريو_partial_f32(ثابت عائم * src, uint32_t n) {
    HVX_ويڪٽر v = *((ثابت HVX_UVector *) src);
    HVX_ويڪٽر_Pred نقاب = Q6_Q_vsetq2_R(n * حجم(عائم));
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
    ثابت bool ڀريل = (coln == VLEN_FP32);

    HVX_ويڪٽر جوڙ_v = Q6_V_vzero();
    لكل (uint32_t t = 0; t < صف; ++t) {
        ثابت عائم   a         = A_صف[t];
        ثابت عائم * x_صف_کانو = X + t * k + col0;

        HVX_ويڪٽر x_v = ڀريل ? *((ثابت HVX_UVector *) x_صف_کانو) : hvx_ڀريو_partial_f32(x_صف_کانو, coln);
        HVX_ويڪٽر a_v = hvx_vec_splat_f32(a);
        جوڙ_v          = hvx_vec_جوڙيو_f32_f32(جوڙ_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    ثابت عائم * b_صف_کانو = B_صف + col0;
    عائم *       x_ٻاهريون_کانو = X + صف * k + col0;

    HVX_ويڪٽر b_v        = ڀريل ? *((ثابت HVX_UVector *) b_صف_کانو) : hvx_ڀريو_partial_f32(b_صف_کانو, coln);
    HVX_ويڪٽر inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_ويڪٽر ٻاهريون_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, جوڙ_v), inv_diag_v);
    hvx_vec_store_u((فراغ *) x_ٻاهريون_کانو, coln * حجم(عائم), ٻاهريون_v);
}

// Batch-level thread: each job is one full batch.
ساكن فراغ solve_tri_batch_thread_f32(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية htp_solve_tri_context * sctx = (بنية htp_solve_tri_context *) معلومات;
    بنية htp_ops_context *       octx = sctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];  // A
    ثابت بنية htp_tensor * src1 = octx->src[1];  // B
    ثابت بنية htp_tensor * dst  = octx->dst;     // X

    ثابت uint32_t n = src0->ne[0];
    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ne02 = src0->ne[2];

    ثابت uint32_t کانو_بلاڪ = VLEN_FP32;
    ثابت uint32_t k_ڀريل    = (k / کانو_بلاڪ) * کانو_بلاڪ;

    ثابت uint32_t شروعات_batch = sctx->ڪم_شروعات + sctx->jobs_في_thread * ith;
    ثابت uint32_t پڇاڙي_batch   = MIN(شروعات_batch + sctx->jobs_في_thread, sctx->ڪم_شروعات + sctx->ڪل_jobs);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروعات_batch);

    لكل (uint32_t batch = شروعات_batch; batch < پڇاڙي_batch; ++batch) {
        ثابت uint32_t i03 = batch / ne02;
        ثابت uint32_t i02 = batch - i03 * ne02;

        ثابت عائم * A_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src0->معلومات + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ثابت عائم * B_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src1->معلومات + i02 * src1->nb[2] + i03 * src1->nb[3]);
        عائم * X_batch = (عائم *) ((uint8_t *) (uintptr_t) dst->معلومات + i02 * dst->nb[2] + i03 * dst->nb[3]);

        لكل (uint32_t صف = 0; صف < n; ++صف) {
            ثابت عائم   diag     = A_batch[صف * n + صف];
            ثابت عائم   inv_diag = 1.0f / diag;
            ثابت عائم * A_صف    = A_batch + صف * n;
            ثابت عائم * B_صف    = B_batch + صف * k;

            uint32_t col0 = 0;
            لكل (; col0 < k_ڀريل; col0 += کانو_بلاڪ) {
                solve_tri_صف_hvx(A_صف, B_صف, X_batch, صف, k, col0, کانو_بلاڪ, inv_diag);
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

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) پڇاڙي_batch);

    FARF(مٿاهون_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, شروعات_batch, پڇاڙي_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
ساكن فراغ solve_tri_chunk_thread_f32(بنانشان صحيح nth, بنانشان صحيح ith, فراغ * معلومات) {
    بنية htp_solve_tri_context * sctx = (بنية htp_solve_tri_context *) معلومات;
    بنية htp_ops_context *       octx = sctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];  // A
    ثابت بنية htp_tensor * src1 = octx->src[1];  // B
    ثابت بنية htp_tensor * dst  = octx->dst;     // X

    ثابت uint32_t n = src0->ne[0];
    ثابت uint32_t k = src1->ne[0];

    ثابت uint32_t ne02 = src0->ne[2];

    ثابت uint32_t شروعات_ڪم = sctx->ڪم_شروعات + sctx->jobs_في_thread * ith;
    ثابت uint32_t پڇاڙي_ڪم   = MIN(شروعات_ڪم + sctx->jobs_في_thread, sctx->ڪم_شروعات + sctx->ڪل_jobs);

    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) شروعات_ڪم);

    لكل (uint32_t ڪم = شروعات_ڪم; ڪم < پڇاڙي_ڪم; ++ڪم) {
        ثابت uint32_t batch = ڪم / sctx->k_chunks;
        ثابت uint32_t chunk = ڪم - batch * sctx->k_chunks;

        ثابت uint32_t i03 = batch / ne02;
        ثابت uint32_t i02 = batch - i03 * ne02;

        ثابت عائم * A_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src0->معلومات + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ثابت عائم * B_batch =
            (ثابت عائم *) ((ثابت uint8_t *) (uintptr_t) src1->معلومات + i02 * src1->nb[2] + i03 * src1->nb[3]);
        عائم * X_batch = (عائم *) ((uint8_t *) (uintptr_t) dst->معلومات + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ثابت uint32_t col0 = chunk * sctx->کانو_بلاڪ;
        ثابت uint32_t coln = MIN(sctx->کانو_بلاڪ, k - col0);

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

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) پڇاڙي_ڪم);

    FARF(مٿاهون_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, شروعات_ڪم, پڇاڙي_ڪم);
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

    ثابت uint32_t کانو_بلاڪ     = VLEN_FP32;
    ثابت uint32_t k_chunks      = (k + کانو_بلاڪ - 1) / کانو_بلاڪ;
    ثابت uint32_t ڪل_batches = src0->ne[2] * src0->ne[3];
    ثابت bool     batched       = ڪل_batches >= (uint32_t) octx->n_ڌاڳا;

    FARF(مٿاهون_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    إذا (batched) {
        uint32_t ڪم_شروعات = 0;
        uint32_t njobs     = ڪل_batches;

        إذا (octx->ctx->mdev.ڳڻپ > 1) {
            ثابت uint32_t batch_سائيز = dst->nb[2];
            ثابت uint32_t batches_في_chunk = (batch_سائيز > 0) ? (HEX_L2_سٽ_سائيز / hex_وڏو_گڏيل_ونڊيندڙ_u32(batch_سائيز, HEX_L2_سٽ_سائيز)) : 1;
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(ڪل_batches, htp_tensor_mdev_معلومات_aligned(dst) ? batches_في_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
            ڪم_شروعات = range.شروعات;
            njobs     = range.ڳڻپ;
        }

        إذا (njobs == 0) {
            أرجع HTP_حالت_OK;
        }

        // Batch-level parallelism
        ثابت uint32_t n_ڌاڳا = octx->n_ڌاڳا;

        بنية htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_في_thread = fastdiv(njobs + n_ڌاڳا - 1, &octx->n_ڌاڳا_div),
            .ڪل_jobs      = njobs,
            .ڪم_شروعات       = ڪم_شروعات,
            .k_chunks        = k_chunks,
            .کانو_بلاڪ       = کانو_بلاڪ,
        };

        work_قطار_لاڳيتو(octx->ctx->work_قطار, solve_tri_batch_thread_f32, &sctx, n_ڌاڳا);
    } وإلا {
        // Chunk-level parallelism
        ثابت uint32_t ڪل_jobs = ڪل_batches * k_chunks;

        uint32_t ڪم_شروعات = 0;
        uint32_t njobs     = ڪل_jobs;

        إذا (octx->ctx->mdev.ڳڻپ > 1) {
            ثابت bool can_ورهايو = htp_tensor_mdev_معلومات_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_سٽ_سائيز - 1)) == 0);
            ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(ڪل_jobs, can_ورهايو ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
            ڪم_شروعات = range.شروعات;
            njobs     = range.ڳڻپ;
        }

        إذا (njobs == 0) {
            أرجع HTP_حالت_OK;
        }

        ثابت uint32_t n_ڌاڳا = octx->n_ڌاڳا;

        بنية htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_في_thread = fastdiv(njobs + n_ڌاڳا - 1, &octx->n_ڌاڳا_div),
            .ڪل_jobs      = njobs,
            .ڪم_شروعات       = ڪم_شروعات,
            .k_chunks        = k_chunks,
            .کانو_بلاڪ       = کانو_بلاڪ,
        };

        work_قطار_لاڳيتو(octx->ctx->work_قطار, solve_tri_chunk_thread_f32, &sctx, n_ڌاڳا);
    }

    أرجع HTP_حالت_OK;
}
