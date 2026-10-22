#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dsv4-hc.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <type_traits>

ਸਥਿਰ constexpr ਪੂਰਨ DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
ਸਥਿਰ constexpr ਪੂਰਨ dsv4_hc_pre_ਖੰਡ_ਮਾਪ = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
ਸਾਂਚਾ <ਤਾਰਕਿਕ gated>
ਸਥਿਰ ਖਾਲੀ dsv4_hc_pre_f32_sycl(
        ਅਚਲ ਦਸ਼ਮਲਵ * x, ਅਚਲ ਦਸ਼ਮਲਵ * weights, ਦਸ਼ਮਲਵ * dst,
        int64_t n_embd, int64_t hc, int64_t n_ਟੋਕਨ,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        ਦਸ਼ਮਲਵ scale,
        ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nr = n_embd * n_ਟੋਕਨ;
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (nr + dsv4_hc_pre_ਖੰਡ_ਮਾਪ - 1) / dsv4_hc_pre_ਖੰਡ_ਮਾਪ;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ * dsv4_hc_pre_ਖੰਡ_ਮਾਪ),
                          sycl::range<1>(dsv4_hc_pre_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) {
            ਅਚਲ int64_t ir = ਵਸਤੂ.get_global_id(0);
            ਜੇ (ir >= nr) {
                ਵਾਪਸ;
            }

            ਅਚਲ int64_t i0 = ir % n_embd;
            ਅਚਲ int64_t it = ir / n_embd;

            ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
            ਲਈ (int64_t ih = 0; ih < hc; ++ih) {
                ਅਚਲ ਦਸ਼ਮਲਵ xv = x[i0*sx0 + ih*sx1 + it*sx2];
                ਦਸ਼ਮਲਵ wv;
                ਜੇ constexpr (gated) {
                    ਅਚਲ ਦਸ਼ਮਲਵ gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::ਘਾਤਾਂਕ(-gv));
                } ਨਹੀਂਤਾਂ {
                    wv = weights[ih*sw0 + it*sw1];
                }
                ਜੋੜ += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * ਜੋੜ;
        });
}

ਸਥਿਰ ਖਾਲੀ dsv4_hc_comb_norm_ਖਾਨੇ_ਗਿਣਤੀ(ਦਸ਼ਮਲਵ * comb, ਦਸ਼ਮਲਵ eps) {
    ਲਈ (ਪੂਰਨ idst = 0; idst < DSV4_HC; ++idst) {
        ਦਸ਼ਮਲਵ ਜੋੜ = eps;
        ਲਈ (ਪੂਰਨ isrc = 0; isrc < DSV4_HC; ++isrc) {
            ਜੋੜ += comb[idst + DSV4_HC*isrc];
        }

        ਅਚਲ ਦਸ਼ਮਲਵ inv_ਜੋੜ = 1.0f / ਜੋੜ;
        ਲਈ (ਪੂਰਨ isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_ਜੋੜ;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ dsv4_hc_comb_norm_ਪੰਕਤੀਆਂ(ਦਸ਼ਮਲਵ * comb, ਦਸ਼ਮਲਵ eps) {
    ਲਈ (ਪੂਰਨ isrc = 0; isrc < DSV4_HC; ++isrc) {
        ਦਸ਼ਮਲਵ ਜੋੜ = eps;
        ਲਈ (ਪੂਰਨ idst = 0; idst < DSV4_HC; ++idst) {
            ਜੋੜ += comb[idst + DSV4_HC*isrc];
        }

        ਅਚਲ ਦਸ਼ਮਲਵ inv_ਜੋੜ = 1.0f / ਜੋੜ;
        ਲਈ (ਪੂਰਨ idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_ਜੋੜ;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ dsv4_hc_comb_f32_sycl(
        ਅਚਲ ਦਸ਼ਮਲਵ * mixes,
        ਅਚਲ ਦਸ਼ਮਲਵ * scale,
        ਅਚਲ ਦਸ਼ਮਲਵ * ਅਧਾਰ,
        ਦਸ਼ਮਲਵ * dst,
        int64_t n_ਟੋਕਨ,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        ਦਸ਼ਮਲਵ eps,
        int32_t n_iter,
        ਕਤਾਰ_ptr stream) {
    constexpr ਪੂਰਨ comb_ਵਿੱਥ = 2*DSV4_HC;

    ਅਚਲ int64_t ਖੰਡ_ਮਾਪ_3 = 256;
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (n_ਟੋਕਨ + ਖੰਡ_ਮਾਪ_3 - 1) / ਖੰਡ_ਮਾਪ_3;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ * ਖੰਡ_ਮਾਪ_3), sycl::range<1>(ਖੰਡ_ਮਾਪ_3)),
        [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ_ct1) {
            ਅਚਲ int64_t it = ਵਸਤੂ_ct1.get_global_id(0);

            ਜੇ (it >= n_ਟੋਕਨ) {
                ਵਾਪਸ;
            }

            ਅਚਲ ਦਸ਼ਮਲਵ scale_comb = scale[2*ss0];
            ਦਸ਼ਮਲਵ comb[DSV4_HC*DSV4_HC];

            ਲਈ (ਪੂਰਨ isrc = 0; isrc < DSV4_HC; ++isrc) {
                ਦਸ਼ਮਲਵ max = -INFINITY;
                ਲਈ (ਪੂਰਨ idst = 0; idst < DSV4_HC; ++idst) {
                    ਅਚਲ ਪੂਰਨ idx = idst + DSV4_HC*isrc;
                    ਅਚਲ ਦਸ਼ਮਲਵ v = mixes[(comb_ਵਿੱਥ + idx)*sm0 + it*sm1] * scale_comb + ਅਧਾਰ[(comb_ਵਿੱਥ + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
                ਲਈ (ਪੂਰਨ idst = 0; idst < DSV4_HC; ++idst) {
                    ਅਚਲ ਪੂਰਨ idx = idst + DSV4_HC*isrc;
                    ਅਚਲ ਦਸ਼ਮਲਵ v = expf(comb[idx] - max);
                    comb[idx] = v;
                    ਜੋੜ += v;
                }

                ਅਚਲ ਦਸ਼ਮਲਵ inv_ਜੋੜ = 1.0f / ਜੋੜ;
                ਲਈ (ਪੂਰਨ idst = 0; idst < DSV4_HC; ++idst) {
                    ਅਚਲ ਪੂਰਨ idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_ਜੋੜ + eps;
                }
            }

            dsv4_hc_comb_norm_ਖਾਨੇ_ਗਿਣਤੀ(comb, eps);
            ਲਈ (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_ਪੰਕਤੀਆਂ(comb, eps);
                dsv4_hc_comb_norm_ਖਾਨੇ_ਗਿਣਤੀ(comb, eps);
            }

            ਲਈ (ਪੂਰਨ isrc = 0; isrc < DSV4_HC; ++isrc) {
                ਲਈ (ਪੂਰਨ idst = 0; idst < DSV4_HC; ++idst) {
                    ਅਚਲ ਪੂਰਨ idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
ਸਥਿਰ constexpr ਪੂਰਨ dsv4_hc_post_ਖੰਡ_ਮਾਪ = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
ਸਾਂਚਾ <ਤਾਰਕਿਕ has_comb>
ਸਥਿਰ ਖਾਲੀ dsv4_hc_post_f32_sycl(
        ਅਚਲ ਦਸ਼ਮਲਵ * x, ਅਚਲ ਦਸ਼ਮਲਵ * residual, ਅਚਲ ਦਸ਼ਮਲਵ * post, ਅਚਲ ਦਸ਼ਮਲਵ * comb, ਦਸ਼ਮਲਵ * dst,
        int64_t n_embd, int64_t hc, int64_t n_ਟੋਕਨ,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t nr = n_embd * hc * n_ਟੋਕਨ;
    ਅਚਲ int64_t ਖੰਡ_ਮਾਪ_3 = dsv4_hc_post_ਖੰਡ_ਮਾਪ;
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (nr + ਖੰਡ_ਮਾਪ_3 - 1) / ਖੰਡ_ਮਾਪ_3;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ਅੰਕ_ਖੰਡਾਂ * ਖੰਡ_ਮਾਪ_3), sycl::range<1>(ਖੰਡ_ਮਾਪ_3)),
        [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) {
            ਅਚਲ int64_t ir = ਵਸਤੂ.get_global_id(0);
            ਜੇ (ir >= nr) {
                ਵਾਪਸ;
            }

            ਅਚਲ int64_t i0   = ir % n_embd;
            ਅਚਲ int64_t idst = (ir / n_embd) % hc;
            ਅਚਲ int64_t it   = ir / (n_embd * hc);

            ਦਸ਼ਮਲਵ ਜੋੜ = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
            ਜੇ constexpr (has_comb) {
                ਲਈ (int64_t isrc = 0; isrc < hc; ++isrc) {
                    ਜੋੜ += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } ਨਹੀਂਤਾਂ {
                ਜੋੜ += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = ਜੋੜ;
        });
}

ਖਾਲੀ ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ਅਚਲ ggml_tensor * x       = dst->src[0];
    ਅਚਲ ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    ਅਚਲ int64_t n_embd   = x->ne[0];
    ਅਚਲ int64_t hc       = x->ne[1];
    ਅਚਲ int64_t n_ਟੋਕਨ = x->ne[2];

    ਅਚਲ ਦਸ਼ਮਲਵ scale = ggml_get_op_params_f32(dst, 0);
    ਅਚਲ ਤਾਰਕਿਕ  gated = ggml_get_op_params_i32(dst, 1) != 0;

    ਕਤਾਰ_ptr stream = ctx.stream();

    ਜੇ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_ਟੋਕਨ);
        dsv4_hc_pre_f32_sycl<ਸੱਚ>(
                (ਅਚਲ ਦਸ਼ਮਲਵ *) x->ਅੰਕੜੇ, (ਅਚਲ ਦਸ਼ਮਲਵ *) weights->ਅੰਕੜੇ, (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ,
                n_embd, hc, n_ਟੋਕਨ,
                nbx0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbx1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbx2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                nbw0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbw1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbw2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                nbd0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbd1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                scale, stream);
    } ਨਹੀਂਤਾਂ {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_ਟੋਕਨ);
        dsv4_hc_pre_f32_sycl<ਝੂਠ>(
                (ਅਚਲ ਦਸ਼ਮਲਵ *) x->ਅੰਕੜੇ, (ਅਚਲ ਦਸ਼ਮਲਵ *) weights->ਅੰਕੜੇ, (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ,
                n_embd, hc, n_ਟੋਕਨ,
                nbx0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbx1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbx2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                nbw0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbw1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), /*sw2=*/ 0,
                nbd0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbd1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
                scale, stream);
    }
}

ਖਾਲੀ ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ਅਚਲ ggml_tensor * mixes = dst->src[0];
    ਅਚਲ ggml_tensor * scale = dst->src[1];
    ਅਚਲ ggml_tensor * ਅਧਾਰ  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(ਅਧਾਰ->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(ਅਧਾਰ->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, ਅਧਾਰ,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    ਅਚਲ int64_t n_ਟੋਕਨ = mixes->ne[1];
    ਅਚਲ ਦਸ਼ਮਲਵ eps = ggml_get_op_params_f32(dst, 0);
    ਅਚਲ int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    ਕਤਾਰ_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (ਅਚਲ ਦਸ਼ਮਲਵ *) mixes->ਅੰਕੜੇ, (ਅਚਲ ਦਸ਼ਮਲਵ *) scale->ਅੰਕੜੇ, (ਅਚਲ ਦਸ਼ਮਲਵ *) ਅਧਾਰ->ਅੰਕੜੇ, (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ,
            n_ਟੋਕਨ,
            nbm0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbm1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
            nbs0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
            nbb0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
            nbd0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbd1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbd2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
            eps, n_iter, stream);
}

ਖਾਲੀ ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    ਅਚਲ ggml_tensor * x        = dst->src[0];
    ਅਚਲ ggml_tensor * residual = dst->src[1];
    ਅਚਲ ggml_tensor * post     = dst->src[2];
    ਅਚਲ ggml_tensor * comb     = dst->src[3];

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
    ਜੇ (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    ਅਚਲ int64_t n_embd   = x->ne[0];
    ਅਚਲ int64_t n_ਟੋਕਨ = x->ne[1];
    ਅਚਲ int64_t hc       = residual->ne[1];

    ਕਤਾਰ_ptr stream = ctx.stream();

    ਅਚਲ ਸਵੈਚਾਲਿਤ launch = [&](ਸਵੈਚਾਲਿਤ has_comb) {
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value>(
            (ਅਚਲ ਦਸ਼ਮਲਵ *) x->ਅੰਕੜੇ, (ਅਚਲ ਦਸ਼ਮਲਵ *) residual->ਅੰਕੜੇ,
            (ਅਚਲ ਦਸ਼ਮਲਵ *) post->ਅੰਕੜੇ, comb ? (ਅਚਲ ਦਸ਼ਮਲਵ *) comb->ਅੰਕੜੇ : ਖਾਲੀ_ਸੂਚਕ, (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ,
            n_embd, hc, n_ਟੋਕਨ,
            nbx0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbx1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
            nbr0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbr1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbr2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
            nbp0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbp1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
            nbc0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbc1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbc2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
            nbd0 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbd1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ), nbd2 / ਆਕਾਰ(ਦਸ਼ਮਲਵ),
            stream);
    };

    ਜੇ (comb) {
        launch(std::true_type{});
    } ਨਹੀਂਤਾਂ {
        launch(std::false_type{});
    }
}
