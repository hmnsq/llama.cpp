#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "hex-profile.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "hvx-types.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"

ರಚನೆ htp_solve_tri_context {
    ರಚನೆ htp_ops_context * octx;
    uint32_t                 jobs_ಪ್ರತಿ_thread;
    uint32_t                 ಒಟ್ಟು_jobs;
    uint32_t                 ಕೆಲಸ_ಆರಂಭ;
    uint32_t                 k_chunks;
    uint32_t                 ಲಂಬಸಾಲು_ಖಂಡ;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ solve_tri_ಅಡ್ಡಸಾಲು_scalar(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_ಅಡ್ಡಸಾಲು,
                                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B_ಅಡ್ಡಸಾಲು,
                                        ದಶಮಾಂಶ *       X,
                                        uint32_t      ಅಡ್ಡಸಾಲು,
                                        uint32_t      k,
                                        uint32_t      col0,
                                        uint32_t      coln,
                                        ದಶಮಾಂಶ         inv_diag) {
    ಗಾಗಿ (uint32_t ಲಂಬಸಾಲು = col0; ಲಂಬಸಾಲು < col0 + coln; ++ಲಂಬಸಾಲು) {
        ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
        ಗಾಗಿ (uint32_t t = 0; t < ಅಡ್ಡಸಾಲು; ++t) {
            ಮೊತ್ತ += A_ಅಡ್ಡಸಾಲು[t] * X[t * k + ಲಂಬಸಾಲು];
        }
        X[ಅಡ್ಡಸಾಲು * k + ಲಂಬಸಾಲು] = (B_ಅಡ್ಡಸಾಲು[ಲಂಬಸಾಲು] - ಮೊತ್ತ) * inv_diag;
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_ತುಂಬು_partial_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, uint32_t n) {
    HVX_ಸದಿಶ v = *((ಸ್ಥಿರಾಂಕ HVX_UVector *) src);
    HVX_ಸದಿಶ_Pred ಮುಖವಾಡ = Q6_Q_vsetq2_R(n * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಹಿಂತಿರುಗಿ Q6_V_vmux_QVV(ಮುಖವಾಡ, v, Q6_V_vzero());
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ solve_tri_ಅಡ್ಡಸಾಲು_hvx(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_ಅಡ್ಡಸಾಲು,
                                     ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B_ಅಡ್ಡಸಾಲು,
                                     ದಶಮಾಂಶ *       X,
                                     uint32_t      ಅಡ್ಡಸಾಲು,
                                     uint32_t      k,
                                     uint32_t      col0,
                                     uint32_t      coln,
                                     ದಶಮಾಂಶ         inv_diag) {
    ಸ್ಥಿರಾಂಕ bool ತುಂಬಿದ = (coln == VLEN_FP32);

    HVX_ಸದಿಶ ಮೊತ್ತ_v = Q6_V_vzero();
    ಗಾಗಿ (uint32_t t = 0; t < ಅಡ್ಡಸಾಲು; ++t) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   a         = A_ಅಡ್ಡಸಾಲು[t];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_ಅಡ್ಡಸಾಲು_ಲಂಬಸಾಲು = X + t * k + col0;

        HVX_ಸದಿಶ x_v = ತುಂಬಿದ ? *((ಸ್ಥಿರಾಂಕ HVX_UVector *) x_ಅಡ್ಡಸಾಲು_ಲಂಬಸಾಲು) : hvx_ತುಂಬು_partial_f32(x_ಅಡ್ಡಸಾಲು_ಲಂಬಸಾಲು, coln);
        HVX_ಸದಿಶ a_v = hvx_vec_splat_f32(a);
        ಮೊತ್ತ_v          = hvx_vec_ಕೂಡಿಸು_f32_f32(ಮೊತ್ತ_v, hvx_vec_mul_f32_f32(x_v, a_v));
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_ಅಡ್ಡಸಾಲು_ಲಂಬಸಾಲು = B_ಅಡ್ಡಸಾಲು + col0;
    ದಶಮಾಂಶ *       x_ಹೊರಹೋಗು_ಲಂಬಸಾಲು = X + ಅಡ್ಡಸಾಲು * k + col0;

    HVX_ಸದಿಶ b_v        = ತುಂಬಿದ ? *((ಸ್ಥಿರಾಂಕ HVX_UVector *) b_ಅಡ್ಡಸಾಲು_ಲಂಬಸಾಲು) : hvx_ತುಂಬು_partial_f32(b_ಅಡ್ಡಸಾಲು_ಲಂಬಸಾಲು, coln);
    HVX_ಸದಿಶ inv_diag_v = hvx_vec_splat_f32(inv_diag);

    HVX_ಸದಿಶ ಹೊರಹೋಗು_v = hvx_vec_mul_f32_f32(hvx_vec_sub_f32_f32(b_v, ಮೊತ್ತ_v), inv_diag_v);
    hvx_vec_store_u((ಶೂನ್ಯ *) x_ಹೊರಹೋಗು_ಲಂಬಸಾಲು, coln * ಗಾತ್ರ(ದಶಮಾಂಶ), ಹೊರಹೋಗು_v);
}

// Batch-level thread: each job is one full batch.
ಸ್ಥಿರ ಶೂನ್ಯ solve_tri_batch_thread_f32(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_solve_tri_context * sctx = (ರಚನೆ htp_solve_tri_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context *       octx = sctx->octx;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];  // A
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];  // B
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;     // X

    ಸ್ಥಿರಾಂಕ uint32_t n = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t k = src1->ne[0];

    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src0->ne[2];

    ಸ್ಥಿರಾಂಕ uint32_t ಲಂಬಸಾಲು_ಖಂಡ = VLEN_FP32;
    ಸ್ಥಿರಾಂಕ uint32_t k_ತುಂಬಿದ    = (k / ಲಂಬಸಾಲು_ಖಂಡ) * ಲಂಬಸಾಲು_ಖಂಡ;

    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ_batch = sctx->ಕೆಲಸ_ಆರಂಭ + sctx->jobs_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ_batch   = MIN(ಆರಂಭ_batch + sctx->jobs_ಪ್ರತಿ_thread, sctx->ಕೆಲಸ_ಆರಂಭ + sctx->ಒಟ್ಟು_jobs);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ಆರಂಭ_batch);

    ಗಾಗಿ (uint32_t batch = ಆರಂಭ_batch; batch < ಕೊನೆ_batch; ++batch) {
        ಸ್ಥಿರಾಂಕ uint32_t i03 = batch / ne02;
        ಸ್ಥಿರಾಂಕ uint32_t i02 = batch - i03 * ne02;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_batch =
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) src0->ದತ್ತಾಂಶ + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B_batch =
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) src1->ದತ್ತಾಂಶ + i02 * src1->nb[2] + i03 * src1->nb[3]);
        ದಶಮಾಂಶ * X_batch = (ದಶಮಾಂಶ *) ((uint8_t *) (uintptr_t) dst->ದತ್ತಾಂಶ + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ಗಾಗಿ (uint32_t ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < n; ++ಅಡ್ಡಸಾಲು) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   diag     = A_batch[ಅಡ್ಡಸಾಲು * n + ಅಡ್ಡಸಾಲು];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   inv_diag = 1.0f / diag;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_ಅಡ್ಡಸಾಲು    = A_batch + ಅಡ್ಡಸಾಲು * n;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B_ಅಡ್ಡಸಾಲು    = B_batch + ಅಡ್ಡಸಾಲು * k;

            uint32_t col0 = 0;
            ಗಾಗಿ (; col0 < k_ತುಂಬಿದ; col0 += ಲಂಬಸಾಲು_ಖಂಡ) {
                solve_tri_ಅಡ್ಡಸಾಲು_hvx(A_ಅಡ್ಡಸಾಲು, B_ಅಡ್ಡಸಾಲು, X_batch, ಅಡ್ಡಸಾಲು, k, col0, ಲಂಬಸಾಲು_ಖಂಡ, inv_diag);
            }

            ಆದರೆ (col0 < k) {
                ಸ್ಥಿರಾಂಕ uint32_t coln = k - col0;
                ಆದರೆ (coln >= 8) {
                    solve_tri_ಅಡ್ಡಸಾಲು_hvx(A_ಅಡ್ಡಸಾಲು, B_ಅಡ್ಡಸಾಲು, X_batch, ಅಡ್ಡಸಾಲು, k, col0, coln, inv_diag);
                } ಇಲ್ಲದಿದ್ದರೆ {
                    solve_tri_ಅಡ್ಡಸಾಲು_scalar(A_ಅಡ್ಡಸಾಲು, B_ಅಡ್ಡಸಾಲು, X_batch, ಅಡ್ಡಸಾಲು, k, col0, coln, inv_diag);
                }
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ಕೊನೆ_batch);

    FARF(ಮೇಲಿನ_2, "solve-tri-batch %d/%d: A=(%ux%u) B=(%ux%u) batch %u:%u\n",
         ith, nth, n, n, k, n, ಆರಂಭ_batch, ಕೊನೆ_batch);
}

// Chunk-level thread: each job is one (batch, col_chunk) pair.
ಸ್ಥಿರ ಶೂನ್ಯ solve_tri_chunk_thread_f32(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_solve_tri_context * sctx = (ರಚನೆ htp_solve_tri_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context *       octx = sctx->octx;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];  // A
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];  // B
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;     // X

    ಸ್ಥಿರಾಂಕ uint32_t n = src0->ne[0];
    ಸ್ಥಿರಾಂಕ uint32_t k = src1->ne[0];

    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src0->ne[2];

    ಸ್ಥಿರಾಂಕ uint32_t ಆರಂಭ_ಕೆಲಸ = sctx->ಕೆಲಸ_ಆರಂಭ + sctx->jobs_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t ಕೊನೆ_ಕೆಲಸ   = MIN(ಆರಂಭ_ಕೆಲಸ + sctx->jobs_ಪ್ರತಿ_thread, sctx->ಕೆಲಸ_ಆರಂಭ + sctx->ಒಟ್ಟು_jobs);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ಆರಂಭ_ಕೆಲಸ);

    ಗಾಗಿ (uint32_t ಕೆಲಸ = ಆರಂಭ_ಕೆಲಸ; ಕೆಲಸ < ಕೊನೆ_ಕೆಲಸ; ++ಕೆಲಸ) {
        ಸ್ಥಿರಾಂಕ uint32_t batch = ಕೆಲಸ / sctx->k_chunks;
        ಸ್ಥಿರಾಂಕ uint32_t chunk = ಕೆಲಸ - batch * sctx->k_chunks;

        ಸ್ಥಿರಾಂಕ uint32_t i03 = batch / ne02;
        ಸ್ಥಿರಾಂಕ uint32_t i02 = batch - i03 * ne02;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_batch =
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) src0->ದತ್ತಾಂಶ + i02 * src0->nb[2] + i03 * src0->nb[3]);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B_batch =
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ uint8_t *) (uintptr_t) src1->ದತ್ತಾಂಶ + i02 * src1->nb[2] + i03 * src1->nb[3]);
        ದಶಮಾಂಶ * X_batch = (ದಶಮಾಂಶ *) ((uint8_t *) (uintptr_t) dst->ದತ್ತಾಂಶ + i02 * dst->nb[2] + i03 * dst->nb[3]);

        ಸ್ಥಿರಾಂಕ uint32_t col0 = chunk * sctx->ಲಂಬಸಾಲು_ಖಂಡ;
        ಸ್ಥಿರಾಂಕ uint32_t coln = MIN(sctx->ಲಂಬಸಾಲು_ಖಂಡ, k - col0);

        ಗಾಗಿ (uint32_t ಅಡ್ಡಸಾಲು = 0; ಅಡ್ಡಸಾಲು < n; ++ಅಡ್ಡಸಾಲು) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ diag     = A_batch[ಅಡ್ಡಸಾಲು * n + ಅಡ್ಡಸಾಲು];
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_diag = 1.0f / diag;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * A_ಅಡ್ಡಸಾಲು = A_batch + ಅಡ್ಡಸಾಲು * n;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * B_ಅಡ್ಡಸಾಲು = B_batch + ಅಡ್ಡಸಾಲು * k;

            ಆದರೆ (coln >= 8) {
                solve_tri_ಅಡ್ಡಸಾಲು_hvx(A_ಅಡ್ಡಸಾಲು, B_ಅಡ್ಡಸಾಲು, X_batch, ಅಡ್ಡಸಾಲು, k, col0, coln, inv_diag);
            } ಇಲ್ಲದಿದ್ದರೆ {
                solve_tri_ಅಡ್ಡಸಾಲು_scalar(A_ಅಡ್ಡಸಾಲು, B_ಅಡ್ಡಸಾಲು, X_batch, ಅಡ್ಡಸಾಲು, k, col0, coln, inv_diag);
            }
        }
    }

    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ಕೊನೆ_ಕೆಲಸ);

    FARF(ಮೇಲಿನ_2, "solve-tri-chunk %d/%d: A=(%ux%u) B=(%ux%u) jobs %u:%u\n",
         ith, nth, n, n, k, n, ಆರಂಭ_ಕೆಲಸ, ಕೊನೆ_ಕೆಲಸ);
}

ಪೂರ್ಣಾಂಕ op_solve_tri(ರಚನೆ htp_ops_context * octx) {
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];  // A
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];  // B
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;     // X

    ಆದರೆ (src0->type != HTP_TYPE_F32 || src1->type != HTP_TYPE_F32 || dst->type != HTP_TYPE_F32) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    // left=true, lower=true, uni=false only
    ಆದರೆ (src0->ne[0] != src0->ne[1]) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }
    ಆದರೆ (src0->ne[1] != src1->ne[1]) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }
    ಆದರೆ (src0->ne[2] != src1->ne[2] || src0->ne[3] != src1->ne[3]) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }
    ಆದರೆ (dst->ne[0] != src1->ne[0] || dst->ne[1] != src1->ne[1] || dst->ne[2] != src1->ne[2] ||
        dst->ne[3] != src1->ne[3]) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }

    ಆದರೆ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(src1) || htp_tensor_is_extended(dst)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ uint32_t k = src1->ne[0];

    ಸ್ಥಿರಾಂಕ uint32_t ಲಂಬಸಾಲು_ಖಂಡ     = VLEN_FP32;
    ಸ್ಥಿರಾಂಕ uint32_t k_chunks      = (k + ಲಂಬಸಾಲು_ಖಂಡ - 1) / ಲಂಬಸಾಲು_ಖಂಡ;
    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_batches = src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ bool     batched       = ಒಟ್ಟು_batches >= (uint32_t) octx->n_ಎಳೆಗಳು;

    FARF(ಮೇಲಿನ_2, "solve-tri: (%ux%ux%ux%u) x (%ux%ux%ux%u) -> (%ux%ux%ux%u) : batched %d\n",
         src0->ne[0], src0->ne[1], src0->ne[2], src0->ne[3],
         src1->ne[0], src1->ne[1], src1->ne[2], src1->ne[3],
         dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], batched);

    ಆದರೆ (batched) {
        uint32_t ಕೆಲಸ_ಆರಂಭ = 0;
        uint32_t njobs     = ಒಟ್ಟು_batches;

        ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
            ಸ್ಥಿರಾಂಕ uint32_t batch_ಗಾತ್ರ = dst->nb[2];
            ಸ್ಥಿರಾಂಕ uint32_t batches_ಪ್ರತಿ_chunk = (batch_ಗಾತ್ರ > 0) ? (HEX_L2_ಸಾಲು_ಗಾತ್ರ / hex_ಮಹತ್ತಮ_ಸಾಮಾನ್ಯ_ಅಪವರ್ತನ_u32(batch_ಗಾತ್ರ, HEX_L2_ಸಾಲು_ಗಾತ್ರ)) : 1;
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(ಒಟ್ಟು_batches, htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(dst) ? batches_ಪ್ರತಿ_chunk : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
            ಕೆಲಸ_ಆರಂಭ = range.ಆರಂಭ;
            njobs     = range.ಎಣಿಕೆ_2;
        }

        ಆದರೆ (njobs == 0) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
        }

        // Batch-level parallelism
        ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

        ರಚನೆ htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_ಪ್ರತಿ_thread = fastdiv(njobs + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div),
            .ಒಟ್ಟು_jobs      = njobs,
            .ಕೆಲಸ_ಆರಂಭ       = ಕೆಲಸ_ಆರಂಭ,
            .k_chunks        = k_chunks,
            .ಲಂಬಸಾಲು_ಖಂಡ       = ಲಂಬಸಾಲು_ಖಂಡ,
        };

        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, solve_tri_batch_thread_f32, &sctx, n_ಎಳೆಗಳು);
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Chunk-level parallelism
        ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_jobs = ಒಟ್ಟು_batches * k_chunks;

        uint32_t ಕೆಲಸ_ಆರಂಭ = 0;
        uint32_t njobs     = ಒಟ್ಟು_jobs;

        ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
            ಸ್ಥಿರಾಂಕ bool can_ವಿಭಜಿಸು = htp_tensor_mdev_ದತ್ತಾಂಶ_aligned(dst) && ((dst->nb[1] & (HTP_TENSOR_MDEV_ಸಾಲು_ಗಾತ್ರ - 1)) == 0);
            ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(ಒಟ್ಟು_jobs, can_ವಿಭಜಿಸು ? 1 : 0, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
            ಕೆಲಸ_ಆರಂಭ = range.ಆರಂಭ;
            njobs     = range.ಎಣಿಕೆ_2;
        }

        ಆದರೆ (njobs == 0) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
        }

        ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

        ರಚನೆ htp_solve_tri_context sctx = {
            .octx            = octx,
            .jobs_ಪ್ರತಿ_thread = fastdiv(njobs + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div),
            .ಒಟ್ಟು_jobs      = njobs,
            .ಕೆಲಸ_ಆರಂಭ       = ಕೆಲಸ_ಆರಂಭ,
            .k_chunks        = k_chunks,
            .ಲಂಬಸಾಲು_ಖಂಡ       = ಲಂಬಸಾಲು_ಖಂಡ,
        };

        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, solve_tri_chunk_thread_f32, &sctx, n_ಎಳೆಗಳು);
    }

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}
