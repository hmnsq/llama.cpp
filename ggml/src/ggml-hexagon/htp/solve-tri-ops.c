#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-profile.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-types.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"

ਬਣਤਰ htp_solve_tri_context {
    ਬਣਤਰ htp_ops_context * octx;
    uint32_t                 jobs_ਪ੍ਰਤੀ_thread;
    uint32_t                 ਕੁੱਲ_jobs;
    uint32_t                 ਕੰਮ_ਸ਼ੁਰੂ;
    uint32_t                 k_chunks;
    uint32_t                 ਖਾਨਾ_ਖੰਡ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ solve_tri_ਪੰਕਤੀ_scalar(ਅਚਲ ਦਸ਼ਮਲਵ * A_ਪੰਕਤੀ,
                                        ਅਚਲ ਦਸ਼ਮਲਵ * B_ਪੰਕਤੀ,
                                        ਦਸ਼ਮਲਵ *       X,
                                        uint32_t      ਪੰਕਤੀ,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        ਦਸ਼ਮਲਵ         inv_diag) {
    ਲਈ (uint32_t ਖਾਨਾ = col0; ਖਾਨਾ < col0 + coln; ++ਖਾਨਾ) {
        ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
        ਲਈ (uint32_t t = 0; t < ਪੰਕਤੀ; ++t) {
            ਜੋੜ += A_ਪੰਕਤੀ[t] * X[t * k + ਖਾਨਾ];
        }
        X[ਪੰਕਤੀ * k + ਖਾਨਾ] = (B_ਪੰਕਤੀ[ਖਾਨਾ] - ਜੋੜ) * inv_diag;
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ hvx_ਲੱਦੋ_partial_f32(ਅਚਲ ਦਸ਼ਮਲਵ * src, uint32_t n) {
    HVX_ਸਦਿਸ਼ v = *((ਅਚਲ HVX_UVector *) src);
    HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ = Q6_Q_vsetq2_R(n * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਵਾਪਸ Q6_V_vmux_QVV(ਨਕਾਬ, v, Q6_V_vzero());
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ solve_tri_ਪੰਕਤੀ_hvx(ਅਚਲ ਦਸ਼ਮਲਵ * A_ਪੰਕਤੀ,
                                     ਅਚਲ ਦਸ਼ਮਲਵ * B_ਪੰਕਤੀ,
                                     ਦਸ਼ਮਲਵ *       X,
                                     uint32_t      ਪੰਕਤੀ,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     ਦਸ਼ਮਲਵ         inv_diag) {
    ਅਚਲ bool ਭਰਿਆ = (coln == VLEN_FP32);

    HVX_ਸਦਿਸ਼ ਜੋੜ_v = Q6_V_vzero();
    ਲਈ (uint32_t t = 0; t < ਪੰਕਤੀ; ++t) {
        ਅਚਲ ਦਸ਼ਮਲਵ   a         = A_ਪੰਕਤੀ[t];
        ਅਚਲ ਦਸ਼ਮਲਵ * x_ਪੰਕਤੀ_ਖਾਨਾ = X + t * k + col0;

        HVX_ਸਦਿਸ਼ x_v = ਭਰਿਆ ? *((ਅਚਲ HVX_UVector *) x_ਪੰਕਤੀ_ਖਾਨਾ) : hvx_ਲੱਦੋ_partial_f32(x_ਪੰਕਤੀ_ਖਾਨਾ, coln);
        HVX_ਸਦਿਸ਼ a_v = hvx_vec_splat_f32(a);
        ਜੋੜ_v          = hvx_vec_ਜੋੜੋ_f32_f32(ਜੋੜ_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    ਅਚਲ ਦਸ਼ਮਲਵ * b_ਪੰਕਤੀ_ਖਾਨਾ = B_ਪੰਕਤੀ + col0;
    ਦਸ਼ਮਲਵ *       x_ਬਾਹਰਲਾ_ਖਾਨਾ = X + ਪੰਕਤੀ * k + col0;

    HVX_ਸਦਿਸ਼ b_v        = ਭਰਿਆ ? *((ਅਚਲ HVX_UVector *) b_ਪੰਕਤੀ_ਖਾਨਾ) : hvx_ਲੱਦੋ_partial_f32(b_ਪੰਕਤੀ_ਖਾਨਾ, coln);
    HVX_ਸਦਿਸ਼ inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_ਸਦਿਸ਼ ਬਾਹਰਲਾ_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, ਜੋੜ_v), inv_diag_v);
    hvx_vec_store_u((ਖਾਲੀ *) x_ਬਾਹਰਲਾ_ਖਾਨਾ, coln * ਆਕਾਰ(ਦਸ਼ਮਲਵ), ਬਾਹਰਲਾ_v);
}

// Batch-level thread: each job is one full batch.
ਸਥਿਰ ਖਾਲੀ solve_tri_batch_thread_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_solve_tri_context * sctx = (ਬਣਤਰ htp_solve_tri_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context *       octx = sctx->octx;

    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];  // A
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];  // B
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;     // X

    ਅਚਲ uint32_t n = src0->ne[0];
    ਅਚਲ uint32_t k = src1->ne[0];

    ਅਚਲ uint32_t ne02 = src0->ne[2];

    ਅਚਲ uint32_t ਖਾਨਾ_ਖੰਡ = VLEN_FP32;
    ਅਚਲ uint32_t k_ਭਰਿਆ    = (k / ਖਾਨਾ_ਖੰਡ) * ਖਾਨਾ_ਖੰਡ;

    ਅਚਲ uint32_t ਸ਼ੁਰੂ_batch = sctx->ਕੰਮ_ਸ਼ੁਰੂ + sctx->jobs_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਸਿਰਾ_batch   = MIN(ਸ਼ੁਰੂ_batch + sctx->jobs_ਪ੍ਰਤੀ_thread, sctx->ਕੰਮ_ਸ਼ੁਰੂ + sctx->ਕੁੱਲ_jobs);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ਸ਼ੁਰੂ_batch);

    ਲਈ (uint32_t batch = ਸ਼ੁਰੂ_batch; batch < ਸਿਰਾ_batch; ++batch) {
        ਅਚਲ uint32_t i03 = batch / ne02;
        ਅਚਲ uint32_t i02 = batch - i03 * ne02;

        ਅਚਲ ਦਸ਼ਮਲਵ * A_batch =
            (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) (uintptr_t) src0->ਅੰਕੜੇ + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ਅਚਲ ਦਸ਼ਮਲਵ * B_batch =
            (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) (uintptr_t) src1->ਅੰਕੜੇ + i02 * src1->nb[2] + i03 * src1->nb[3]);
        ਦਸ਼ਮਲਵ * X_batch = (ਦਸ਼ਮਲਵ *) ((uint8_t *) (uintptr_t) dst->ਅੰਕੜੇ + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ਲਈ (uint32_t ਪੰਕਤੀ = 0; ਪੰਕਤੀ < n; ++ਪੰਕਤੀ) {
            ਅਚਲ ਦਸ਼ਮਲਵ   diag     = A_batch[ਪੰਕਤੀ * n + ਪੰਕਤੀ];
            ਅਚਲ ਦਸ਼ਮਲਵ   inv_diag = 1.0f / diag;
            ਅਚਲ ਦਸ਼ਮਲਵ * A_ਪੰਕਤੀ    = A_batch + ਪੰਕਤੀ * n;
            ਅਚਲ ਦਸ਼ਮਲਵ * B_ਪੰਕਤੀ    = B_batch + ਪੰਕਤੀ * k;

            uint32_t col0 = 0;
            ਲਈ (; col0 < k_ਭਰਿਆ; col0 += ਖਾਨਾ_ਖੰਡ) {
                solve_tri_ਪੰਕਤੀ_hvx(A_ਪੰਕਤੀ, B_ਪੰਕਤੀ, X_batch, ਪੰਕਤੀ, k, col0, ਖਾਨਾ_ਖੰਡ, inv_diag);
            }

            ਜੇ (col0 < k) {
                ਅਚਲ uint32_t coln = k - col0;
                ਜੇ (coln >= 8) {
                    solve_tri_ਪੰਕਤੀ_hvx(A_ਪੰਕਤੀ, B_ਪੰਕਤੀ, X_batch, ਪੰਕਤੀ, k, col0, coln, inv_diag);
                } ਨਹੀਂਤਾਂ {
                    solve_tri_ਪੰਕਤੀ_scalar(A_ਪੰਕਤੀ, B_ਪੰਕਤੀ, X_batch, ਪੰਕਤੀ, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ਸਿਰਾ_batch);

    FARF(ਉੱਚਾ_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, ਸ਼ੁਰੂ_batch, ਸਿਰਾ_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
ਸਥਿਰ ਖਾਲੀ solve_tri_chunk_thread_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_solve_tri_context * sctx = (ਬਣਤਰ htp_solve_tri_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context *       octx = sctx->octx;

    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];  // A
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];  // B
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;     // X

    ਅਚਲ uint32_t n = src0->ne[0];
    ਅਚਲ uint32_t k = src1->ne[0];

    ਅਚਲ uint32_t ne02 = src0->ne[2];

    ਅਚਲ uint32_t ਸ਼ੁਰੂ_ਕੰਮ = sctx->ਕੰਮ_ਸ਼ੁਰੂ + sctx->jobs_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t ਸਿਰਾ_ਕੰਮ   = MIN(ਸ਼ੁਰੂ_ਕੰਮ + sctx->jobs_ਪ੍ਰਤੀ_thread, sctx->ਕੰਮ_ਸ਼ੁਰੂ + sctx->ਕੁੱਲ_jobs);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ਸ਼ੁਰੂ_ਕੰਮ);

    ਲਈ (uint32_t ਕੰਮ = ਸ਼ੁਰੂ_ਕੰਮ; ਕੰਮ < ਸਿਰਾ_ਕੰਮ; ++ਕੰਮ) {
        ਅਚਲ uint32_t batch = ਕੰਮ / sctx->k_chunks;
        ਅਚਲ uint32_t chunk = ਕੰਮ - batch * sctx->k_chunks;

        ਅਚਲ uint32_t i03 = batch / ne02;
        ਅਚਲ uint32_t i02 = batch - i03 * ne02;

        ਅਚਲ ਦਸ਼ਮਲਵ * A_batch =
            (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) (uintptr_t) src0->ਅੰਕੜੇ + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ਅਚਲ ਦਸ਼ਮਲਵ * B_batch =
            (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ uint8_t *) (uintptr_t) src1->ਅੰਕੜੇ + i02 * src1->nb[2] + i03 * src1->nb[3]);
        ਦਸ਼ਮਲਵ * X_batch = (ਦਸ਼ਮਲਵ *) ((uint8_t *) (uintptr_t) dst->ਅੰਕੜੇ + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ਅਚਲ uint32_t col0 = chunk * sctx->ਖਾਨਾ_ਖੰਡ;
        ਅਚਲ uint32_t coln = MIN(sctx->ਖਾਨਾ_ਖੰਡ, k - col0);

        ਲਈ (uint32_t ਪੰਕਤੀ = 0; ਪੰਕਤੀ < n; ++ਪੰਕਤੀ) {
            ਅਚਲ ਦਸ਼ਮਲਵ diag     = A_batch[ਪੰਕਤੀ * n + ਪੰਕਤੀ];
            ਅਚਲ ਦਸ਼ਮਲਵ inv_diag = 1.0f / diag;

            ਅਚਲ ਦਸ਼ਮਲਵ * A_ਪੰਕਤੀ = A_batch + ਪੰਕਤੀ * n;
            ਅਚਲ ਦਸ਼ਮਲਵ * B_ਪੰਕਤੀ = B_batch + ਪੰਕਤੀ * k;

            ਜੇ (coln >= 8) {
                solve_tri_ਪੰਕਤੀ_hvx(A_ਪੰਕਤੀ, B_ਪੰਕਤੀ, X_batch, ਪੰਕਤੀ, k, col0, coln, inv_diag);
            } ਨਹੀਂਤਾਂ {
                solve_tri_ਪੰਕਤੀ_scalar(A_ਪੰਕਤੀ, B_ਪੰਕਤੀ, X_batch, ਪੰਕਤੀ, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ਸਿਰਾ_ਕੰਮ);

    FARF(ਉੱਚਾ_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, ਸ਼ੁਰੂ_ਕੰਮ, ਸਿਰਾ_ਕੰਮ);
}

ਪੂਰਨ op_solve_tri(ਬਣਤਰ htp_ops_context * octx) {
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];  // A
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];  // B
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;     // X

    ਜੇ (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    ਜੇ (src0->ne[0] != src0->ne[1]) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }
    ਜੇ (src0->ne[1] != src1->ne[1]) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }
    ਜੇ (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }
    ਜੇ (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }

    ਜੇ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ uint32_t k = src1->ne[0];

    ਅਚਲ uint32_t ਖਾਨਾ_ਖੰਡ     = VLEN_FP32;
    ਅਚਲ uint32_t k_chunks      = (k + ਖਾਨਾ_ਖੰਡ - 1) / ਖਾਨਾ_ਖੰਡ;
    ਅਚਲ uint32_t ਕੁੱਲ_batches = src0->ne[2] * src0->ne[3];
    ਅਚਲ bool     batched       = ਕੁੱਲ_batches >= (uint32_t) octx->n_ਧਾਗੇ;

    FARF(ਉੱਚਾ_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    ਜੇ (batched) {
        uint32_t ਕੰਮ_ਸ਼ੁਰੂ = 0;
        uint32_t njobs     = ਕੁੱਲ_batches;

        ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
            ਅਚਲ uint32_t batch_ਮਾਪ = dst->nb[2];
            ਅਚਲ uint32_t batches_ਪ੍ਰਤੀ_chunk = (batch_ਮਾਪ > 0) ? (HEX_L2_ਸਤਰ_ਮਾਪ / hex_ਮਹੱਤਮ_ਸਾਂਝਾ_ਭਾਜਕ_u32(batch_ਮਾਪ, HEX_L2_ਸਤਰ_ਮਾਪ)) : 1;
            ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(ਕੁੱਲ_batches, htp_tensor_mdev_ਅੰਕੜੇ_aligned(dst) ? batches_ਪ੍ਰਤੀ_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
            ਕੰਮ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
            njobs     = range.ਗਿਣਤੀ_2;
        }

        ਜੇ (njobs == 0) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
        }

        // Batch-level parallelism
        ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

        ਬਣਤਰ htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_ਪ੍ਰਤੀ_thread = fastdiv(njobs + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div),
            .ਕੁੱਲ_jobs      = njobs,
            .ਕੰਮ_ਸ਼ੁਰੂ       = ਕੰਮ_ਸ਼ੁਰੂ,
            .k_chunks        = k_chunks,
            .ਖਾਨਾ_ਖੰਡ       = ਖਾਨਾ_ਖੰਡ,
        };

        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, solve_tri_batch_thread_f32, &sctx, n_ਧਾਗੇ);
    } ਨਹੀਂਤਾਂ {
        // Chunk-level parallelism
        ਅਚਲ uint32_t ਕੁੱਲ_jobs = ਕੁੱਲ_batches * k_chunks;

        uint32_t ਕੰਮ_ਸ਼ੁਰੂ = 0;
        uint32_t njobs     = ਕੁੱਲ_jobs;

        ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
            ਅਚਲ bool can_ਵੰਡੋ = htp_tensor_mdev_ਅੰਕੜੇ_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_ਸਤਰ_ਮਾਪ - 1)) == 0);
            ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(ਕੁੱਲ_jobs, can_ਵੰਡੋ ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
            ਕੰਮ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
            njobs     = range.ਗਿਣਤੀ_2;
        }

        ਜੇ (njobs == 0) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
        }

        ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

        ਬਣਤਰ htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_ਪ੍ਰਤੀ_thread = fastdiv(njobs + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div),
            .ਕੁੱਲ_jobs      = njobs,
            .ਕੰਮ_ਸ਼ੁਰੂ       = ਕੰਮ_ਸ਼ੁਰੂ,
            .k_chunks        = k_chunks,
            .ਖਾਨਾ_ਖੰਡ       = ਖਾਨਾ_ਖੰਡ,
        };

        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, solve_tri_chunk_thread_f32, &sctx, n_ਧਾਗੇ);
    }

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}
