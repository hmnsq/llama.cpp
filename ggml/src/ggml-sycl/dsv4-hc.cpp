#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "dsv4-hc.hpp"

#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <type_traits>

ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ dsv4_hc_pre_ಖಂಡ_ಗಾತ್ರ = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
ಮಾದರಿ <ತಾರ್ಕಿಕ gated>
ಸ್ಥಿರ ಶೂನ್ಯ dsv4_hc_pre_f32_sycl(
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * weights, ದಶಮಾಂಶ * dst,
        int64_t n_embd, int64_t hc, int64_t n_ಟೋಕನುಗಳು,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        ದಶಮಾಂಶ scale,
        ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nr = n_embd * n_ಟೋಕನುಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (nr + dsv4_hc_pre_ಖಂಡ_ಗಾತ್ರ - 1) / dsv4_hc_pre_ಖಂಡ_ಗಾತ್ರ;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು * dsv4_hc_pre_ಖಂಡ_ಗಾತ್ರ),
                          sycl::range<1>(dsv4_hc_pre_ಖಂಡ_ಗಾತ್ರ)),
        [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) {
            ಸ್ಥಿರಾಂಕ int64_t ir = ಅಂಶ.get_global_id(0);
            ಆದರೆ (ir >= nr) {
                ಹಿಂತಿರುಗಿ;
            }

            ಸ್ಥಿರಾಂಕ int64_t i0 = ir % n_embd;
            ಸ್ಥಿರಾಂಕ int64_t it = ir / n_embd;

            ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
            ಗಾಗಿ (int64_t ih = 0; ih < hc; ++ih) {
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ xv = x[i0*sx0 + ih*sx1 + it*sx2];
                ದಶಮಾಂಶ wv;
                ಆದರೆ constexpr (gated) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::ಘಾತಾಂಕ(-gv));
                } ಇಲ್ಲದಿದ್ದರೆ {
                    wv = weights[ih*sw0 + it*sw1];
                }
                ಮೊತ್ತ += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * ಮೊತ್ತ;
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ dsv4_hc_comb_norm_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ(ದಶಮಾಂಶ * comb, ದಶಮಾಂಶ eps) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idst = 0; idst < DSV4_HC; ++idst) {
        ದಶಮಾಂಶ ಮೊತ್ತ = eps;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ isrc = 0; isrc < DSV4_HC; ++isrc) {
            ಮೊತ್ತ += comb[idst + DSV4_HC*isrc];
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_ಮೊತ್ತ = 1.0f / ಮೊತ್ತ;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_ಮೊತ್ತ;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dsv4_hc_comb_norm_ಅಡ್ಡಸಾಲುಗಳು(ದಶಮಾಂಶ * comb, ದಶಮಾಂಶ eps) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ isrc = 0; isrc < DSV4_HC; ++isrc) {
        ದಶಮಾಂಶ ಮೊತ್ತ = eps;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idst = 0; idst < DSV4_HC; ++idst) {
            ಮೊತ್ತ += comb[idst + DSV4_HC*isrc];
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_ಮೊತ್ತ = 1.0f / ಮೊತ್ತ;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_ಮೊತ್ತ;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dsv4_hc_comb_f32_sycl(
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * mixes,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * scale,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ತಳ,
        ದಶಮಾಂಶ * dst,
        int64_t n_ಟೋಕನುಗಳು,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        ದಶಮಾಂಶ eps,
        int32_t n_iter,
        ಸರತಿ_ptr stream) {
    constexpr ಪೂರ್ಣಾಂಕ comb_ಸ್ಥಾನಾಂತರ = 2*DSV4_HC;

    ಸ್ಥಿರಾಂಕ int64_t ಖಂಡ_ಗಾತ್ರ_3 = 256;
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (n_ಟೋಕನುಗಳು + ಖಂಡ_ಗಾತ್ರ_3 - 1) / ಖಂಡ_ಗಾತ್ರ_3;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು * ಖಂಡ_ಗಾತ್ರ_3), sycl::range<1>(ಖಂಡ_ಗಾತ್ರ_3)),
        [=](sycl::nd_ಅಂಶ<1> ಅಂಶ_ct1) {
            ಸ್ಥಿರಾಂಕ int64_t it = ಅಂಶ_ct1.get_global_id(0);

            ಆದರೆ (it >= n_ಟೋಕನುಗಳು) {
                ಹಿಂತಿರುಗಿ;
            }

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale_comb = scale[2*ss0];
            ದಶಮಾಂಶ comb[DSV4_HC*DSV4_HC];

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ isrc = 0; isrc < DSV4_HC; ++isrc) {
                ದಶಮಾಂಶ max = -INFINITY;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idst = 0; idst < DSV4_HC; ++idst) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx = idst + DSV4_HC*isrc;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = mixes[(comb_ಸ್ಥಾನಾಂತರ + idx)*sm0 + it*sm1] * scale_comb + ತಳ[(comb_ಸ್ಥಾನಾಂತರ + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idst = 0; idst < DSV4_HC; ++idst) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx = idst + DSV4_HC*isrc;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ v = expf(comb[idx] - max);
                    comb[idx] = v;
                    ಮೊತ್ತ += v;
                }

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_ಮೊತ್ತ = 1.0f / ಮೊತ್ತ;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idst = 0; idst < DSV4_HC; ++idst) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_ಮೊತ್ತ + eps;
                }
            }

            dsv4_hc_comb_norm_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ(comb, eps);
            ಗಾಗಿ (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_ಅಡ್ಡಸಾಲುಗಳು(comb, eps);
                dsv4_hc_comb_norm_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ(comb, eps);
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ isrc = 0; isrc < DSV4_HC; ++isrc) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ idst = 0; idst < DSV4_HC; ++idst) {
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ dsv4_hc_post_ಖಂಡ_ಗಾತ್ರ = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
ಮಾದರಿ <ತಾರ್ಕಿಕ has_comb>
ಸ್ಥಿರ ಶೂನ್ಯ dsv4_hc_post_f32_sycl(
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * residual, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * post, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * comb, ದಶಮಾಂಶ * dst,
        int64_t n_embd, int64_t hc, int64_t n_ಟೋಕನುಗಳು,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t nr = n_embd * hc * n_ಟೋಕನುಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t ಖಂಡ_ಗಾತ್ರ_3 = dsv4_hc_post_ಖಂಡ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (nr + ಖಂಡ_ಗಾತ್ರ_3 - 1) / ಖಂಡ_ಗಾತ್ರ_3;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_ಖಂಡಗಳು * ಖಂಡ_ಗಾತ್ರ_3), sycl::range<1>(ಖಂಡ_ಗಾತ್ರ_3)),
        [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) {
            ಸ್ಥಿರಾಂಕ int64_t ir = ಅಂಶ.get_global_id(0);
            ಆದರೆ (ir >= nr) {
                ಹಿಂತಿರುಗಿ;
            }

            ಸ್ಥಿರಾಂಕ int64_t i0   = ir % n_embd;
            ಸ್ಥಿರಾಂಕ int64_t idst = (ir / n_embd) % hc;
            ಸ್ಥಿರಾಂಕ int64_t it   = ir / (n_embd * hc);

            ದಶಮಾಂಶ ಮೊತ್ತ = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
            ಆದರೆ constexpr (has_comb) {
                ಗಾಗಿ (int64_t isrc = 0; isrc < hc; ++isrc) {
                    ಮೊತ್ತ += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಮೊತ್ತ += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = ಮೊತ್ತ;
        });
}

ಶೂನ್ಯ ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಸ್ಥಿರಾಂಕ ggml_tensor * x       = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    ಸ್ಥಿರಾಂಕ int64_t n_embd   = x->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t hc       = x->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = x->ne[2];

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = ggml_get_op_params_f32(dst, 0);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ  gated = ggml_get_op_params_i32(dst, 1) != 0;

    ಸರತಿ_ptr stream = ctx.stream();

    ಆದರೆ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_ಟೋಕನುಗಳು);
        dsv4_hc_pre_f32_sycl<ನಿಜ>(
                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) x->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) weights->ದತ್ತಾಂಶ, (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ,
                n_embd, hc, n_ಟೋಕನುಗಳು,
                nbx0 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbx1 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbx2 / ಗಾತ್ರ(ದಶಮಾಂಶ),
                nbw0 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbw1 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbw2 / ಗಾತ್ರ(ದಶಮಾಂಶ),
                nbd0 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbd1 / ಗಾತ್ರ(ದಶಮಾಂಶ),
                scale, stream);
    } ಇಲ್ಲದಿದ್ದರೆ {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_ಟೋಕನುಗಳು);
        dsv4_hc_pre_f32_sycl<ಸುಳ್ಳು>(
                (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) x->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) weights->ದತ್ತಾಂಶ, (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ,
                n_embd, hc, n_ಟೋಕನುಗಳು,
                nbx0 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbx1 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbx2 / ಗಾತ್ರ(ದಶಮಾಂಶ),
                nbw0 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbw1 / ಗಾತ್ರ(ದಶಮಾಂಶ), /*sw2=*/ 0,
                nbd0 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbd1 / ಗಾತ್ರ(ದಶಮಾಂಶ),
                scale, stream);
    }
}

ಶೂನ್ಯ ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ಸ್ಥಿರಾಂಕ ggml_tensor * mixes = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * scale = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ತಳ  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(ತಳ->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(ತಳ->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, ತಳ,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = mixes->ne[1];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps = ggml_get_op_params_f32(dst, 0);
    ಸ್ಥಿರಾಂಕ int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    ಸರತಿ_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) mixes->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) scale->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ತಳ->ದತ್ತಾಂಶ, (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ,
            n_ಟೋಕನುಗಳು,
            nbm0 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbm1 / ಗಾತ್ರ(ದಶಮಾಂಶ),
            nbs0 / ಗಾತ್ರ(ದಶಮಾಂಶ),
            nbb0 / ಗಾತ್ರ(ದಶಮಾಂಶ),
            nbd0 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbd1 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbd2 / ಗಾತ್ರ(ದಶಮಾಂಶ),
            eps, n_iter, stream);
}

ಶೂನ್ಯ ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    ಸ್ಥಿರಾಂಕ ggml_tensor * x        = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * residual = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * post     = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,        nb);
    GGML_TENSOR_LOCALS(size_t, nbr, residual, nb);
    GGML_TENSOR_LOCALS(size_t, nbp, post,     nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,      nb);

    size_t nbc0 = 0;
    size_t nbc1 = 0;
    size_t nbc2 = 0;
    ಆದರೆ (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    ಸ್ಥಿರಾಂಕ int64_t n_embd   = x->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t n_ಟೋಕನುಗಳು = x->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t hc       = residual->ne[1];

    ಸರತಿ_ptr stream = ctx.stream();

    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ launch = [&](ಸ್ವಯಂಚಾಲಿತ has_comb) {
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value>(
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) x->ದತ್ತಾಂಶ, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) residual->ದತ್ತಾಂಶ,
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) post->ದತ್ತಾಂಶ, comb ? (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) comb->ದತ್ತಾಂಶ : ಶೂನ್ಯಸೂಚಕ, (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ,
            n_embd, hc, n_ಟೋಕನುಗಳು,
            nbx0 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbx1 / ಗಾತ್ರ(ದಶಮಾಂಶ),
            nbr0 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbr1 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbr2 / ಗಾತ್ರ(ದಶಮಾಂಶ),
            nbp0 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbp1 / ಗಾತ್ರ(ದಶಮಾಂಶ),
            nbc0 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbc1 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbc2 / ಗಾತ್ರ(ದಶಮಾಂಶ),
            nbd0 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbd1 / ಗಾತ್ರ(ದಶಮಾಂಶ), nbd2 / ಗಾತ್ರ(ದಶಮಾಂಶ),
            stream);
    };

    ಆದರೆ (comb) {
        launch(std::true_type{});
    } ಇಲ್ಲದಿದ್ದರೆ {
        launch(std::false_type{});
    }
}
