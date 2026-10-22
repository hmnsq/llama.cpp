#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "dsv4-hc.hpp"

#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <type_traits>

ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ dsv4_hc_pre_block_size = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
template <ತಾರ್ಕಿಕ gated>
ಸ್ಥಿರ ಶೂನ್ಯ dsv4_hc_pre_f32_sycl(
        ಧ್ರುವ ಭಿನ್ನ * x, ಧ್ರುವ ಭಿನ್ನ * weights, ಭಿನ್ನ * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        ಭಿನ್ನ scale,
        queue_ptr stream) {
    ಧ್ರುವ int64_t nr = n_embd * n_tokens;
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (nr + dsv4_hc_pre_block_size - 1) / dsv4_hc_pre_block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks * dsv4_hc_pre_block_size),
                          sycl::range<1>(dsv4_hc_pre_block_size)),
        [=](sycl::nd_item<1> item) {
            ಧ್ರುವ int64_t ir = item.get_global_id(0);
            ಯದಿ (ir >= nr) {
                ಪ್ರತ್ಯಾಗಮ;
            }

            ಧ್ರುವ int64_t i0 = ir % n_embd;
            ಧ್ರುವ int64_t it = ir / n_embd;

            ಭಿನ್ನ sum = 0.0f;
            ಪ್ರತಿ (int64_t ih = 0; ih < hc; ++ih) {
                ಧ್ರುವ ಭಿನ್ನ xv = x[i0*sx0 + ih*sx1 + it*sx2];
                ಭಿನ್ನ wv;
                ಯದಿ constexpr (gated) {
                    ಧ್ರುವ ಭಿನ್ನ gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::exp(-gv));
                } ಅನ್ಯಥಾ {
                    wv = weights[ih*sw0 + it*sw1];
                }
                sum += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * sum;
        });
}

ಸ್ಥಿರ ಶೂನ್ಯ dsv4_hc_comb_norm_cols(ಭಿನ್ನ * comb, ಭಿನ್ನ eps) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idst = 0; idst < DSV4_HC; ++idst) {
        ಭಿನ್ನ sum = eps;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        ಧ್ರುವ ಭಿನ್ನ inv_sum = 1.0f / sum;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dsv4_hc_comb_norm_rows(ಭಿನ್ನ * comb, ಭಿನ್ನ eps) {
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ isrc = 0; isrc < DSV4_HC; ++isrc) {
        ಭಿನ್ನ sum = eps;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        ಧ್ರುವ ಭಿನ್ನ inv_sum = 1.0f / sum;
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dsv4_hc_comb_f32_sycl(
        ಧ್ರುವ ಭಿನ್ನ * mixes,
        ಧ್ರುವ ಭಿನ್ನ * scale,
        ಧ್ರುವ ಭಿನ್ನ * base,
        ಭಿನ್ನ * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        ಭಿನ್ನ eps,
        int32_t n_iter,
        queue_ptr stream) {
    constexpr ಪೂರ್ಣಾಙ್ಕ comb_offset = 2*DSV4_HC;

    ಧ್ರುವ int64_t block_size = 256;
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (n_tokens + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item_ct1) {
            ಧ್ರುವ int64_t it = item_ct1.get_global_id(0);

            ಯದಿ (it >= n_tokens) {
                ಪ್ರತ್ಯಾಗಮ;
            }

            ಧ್ರುವ ಭಿನ್ನ scale_comb = scale[2*ss0];
            ಭಿನ್ನ comb[DSV4_HC*DSV4_HC];

            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ isrc = 0; isrc < DSV4_HC; ++isrc) {
                ಭಿನ್ನ max = -INFINITY;
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idst = 0; idst < DSV4_HC; ++idst) {
                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx = idst + DSV4_HC*isrc;
                    ಧ್ರುವ ಭಿನ್ನ v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                ಭಿನ್ನ sum = 0.0f;
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idst = 0; idst < DSV4_HC; ++idst) {
                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx = idst + DSV4_HC*isrc;
                    ಧ್ರುವ ಭಿನ್ನ v = expf(comb[idx] - max);
                    comb[idx] = v;
                    sum += v;
                }

                ಧ್ರುವ ಭಿನ್ನ inv_sum = 1.0f / sum;
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idst = 0; idst < DSV4_HC; ++idst) {
                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_sum + eps;
                }
            }

            dsv4_hc_comb_norm_cols(comb, eps);
            ಪ್ರತಿ (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_rows(comb, eps);
                dsv4_hc_comb_norm_cols(comb, eps);
            }

            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ isrc = 0; isrc < DSV4_HC; ++isrc) {
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ idst = 0; idst < DSV4_HC; ++idst) {
                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ dsv4_hc_post_block_size = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
template <ತಾರ್ಕಿಕ has_comb>
ಸ್ಥಿರ ಶೂನ್ಯ dsv4_hc_post_f32_sycl(
        ಧ್ರುವ ಭಿನ್ನ * x, ಧ್ರುವ ಭಿನ್ನ * residual, ಧ್ರುವ ಭಿನ್ನ * post, ಧ್ರುವ ಭಿನ್ನ * comb, ಭಿನ್ನ * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        queue_ptr stream) {
    ಧ್ರುವ int64_t nr = n_embd * hc * n_tokens;
    ಧ್ರುವ int64_t block_size = dsv4_hc_post_block_size;
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (nr + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ಅಂಕೆ_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item) {
            ಧ್ರುವ int64_t ir = item.get_global_id(0);
            ಯದಿ (ir >= nr) {
                ಪ್ರತ್ಯಾಗಮ;
            }

            ಧ್ರುವ int64_t i0   = ir % n_embd;
            ಧ್ರುವ int64_t idst = (ir / n_embd) % hc;
            ಧ್ರುವ int64_t it   = ir / (n_embd * hc);

            ಭಿನ್ನ sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
            ಯದಿ constexpr (has_comb) {
                ಪ್ರತಿ (int64_t isrc = 0; isrc < hc; ++isrc) {
                    sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } ಅನ್ಯಥಾ {
                sum += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
        });
}

ಶೂನ್ಯ ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಧ್ರುವ ggml_tensor * x       = dst->src[0];
    ಧ್ರುವ ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    ಧ್ರುವ int64_t n_embd   = x->ne[0];
    ಧ್ರುವ int64_t hc       = x->ne[1];
    ಧ್ರುವ int64_t n_tokens = x->ne[2];

    ಧ್ರುವ ಭಿನ್ನ scale = ggml_get_op_params_f32(dst, 0);
    ಧ್ರುವ ತಾರ್ಕಿಕ  gated = ggml_get_op_params_i32(dst, 1) != 0;

    queue_ptr stream = ctx.stream();

    ಯದಿ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
        dsv4_hc_pre_f32_sycl<ಸತ್ಯ>(
                (ಧ್ರುವ ಭಿನ್ನ *) x->data, (ಧ್ರುವ ಭಿನ್ನ *) weights->data, (ಭಿನ್ನ *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbx1 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbx2 / ಪರಿಮಾಣ(ಭಿನ್ನ),
                nbw0 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbw1 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbw2 / ಪರಿಮಾಣ(ಭಿನ್ನ),
                nbd0 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbd1 / ಪರಿಮಾಣ(ಭಿನ್ನ),
                scale, stream);
    } ಅನ್ಯಥಾ {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        dsv4_hc_pre_f32_sycl<ಅಸತ್ಯ>(
                (ಧ್ರುವ ಭಿನ್ನ *) x->data, (ಧ್ರುವ ಭಿನ್ನ *) weights->data, (ಭಿನ್ನ *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbx1 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbx2 / ಪರಿಮಾಣ(ಭಿನ್ನ),
                nbw0 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbw1 / ಪರಿಮಾಣ(ಭಿನ್ನ), /*sw2=*/ 0,
                nbd0 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbd1 / ಪರಿಮಾಣ(ಭಿನ್ನ),
                scale, stream);
    }
}

ಶೂನ್ಯ ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ಧ್ರುವ ggml_tensor * mixes = dst->src[0];
    ಧ್ರುವ ggml_tensor * scale = dst->src[1];
    ಧ್ರುವ ggml_tensor * base  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(base->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(base->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, base,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    ಧ್ರುವ int64_t n_tokens = mixes->ne[1];
    ಧ್ರುವ ಭಿನ್ನ eps = ggml_get_op_params_f32(dst, 0);
    ಧ್ರುವ int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    queue_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (ಧ್ರುವ ಭಿನ್ನ *) mixes->data, (ಧ್ರುವ ಭಿನ್ನ *) scale->data, (ಧ್ರುವ ಭಿನ್ನ *) base->data, (ಭಿನ್ನ *) dst->data,
            n_tokens,
            nbm0 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbm1 / ಪರಿಮಾಣ(ಭಿನ್ನ),
            nbs0 / ಪರಿಮಾಣ(ಭಿನ್ನ),
            nbb0 / ಪರಿಮಾಣ(ಭಿನ್ನ),
            nbd0 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbd1 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbd2 / ಪರಿಮಾಣ(ಭಿನ್ನ),
            eps, n_iter, stream);
}

ಶೂನ್ಯ ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    ಧ್ರುವ ggml_tensor * x        = dst->src[0];
    ಧ್ರುವ ggml_tensor * residual = dst->src[1];
    ಧ್ರುವ ggml_tensor * post     = dst->src[2];
    ಧ್ರುವ ggml_tensor * comb     = dst->src[3];

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
    ಯದಿ (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    ಧ್ರುವ int64_t n_embd   = x->ne[0];
    ಧ್ರುವ int64_t n_tokens = x->ne[1];
    ಧ್ರುವ int64_t hc       = residual->ne[1];

    queue_ptr stream = ctx.stream();

    ಧ್ರುವ auto launch = [&](auto has_comb) {
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value>(
            (ಧ್ರುವ ಭಿನ್ನ *) x->data, (ಧ್ರುವ ಭಿನ್ನ *) residual->data,
            (ಧ್ರುವ ಭಿನ್ನ *) post->data, comb ? (ಧ್ರುವ ಭಿನ್ನ *) comb->data : nullptr, (ಭಿನ್ನ *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbx1 / ಪರಿಮಾಣ(ಭಿನ್ನ),
            nbr0 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbr1 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbr2 / ಪರಿಮಾಣ(ಭಿನ್ನ),
            nbp0 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbp1 / ಪರಿಮಾಣ(ಭಿನ್ನ),
            nbc0 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbc1 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbc2 / ಪರಿಮಾಣ(ಭಿನ್ನ),
            nbd0 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbd1 / ಪರಿಮಾಣ(ಭಿನ್ನ), nbd2 / ಪರಿಮಾಣ(ಭಿನ್ನ),
            stream);
    };

    ಯದಿ (comb) {
        launch(std::true_type{});
    } ಅನ್ಯಥಾ {
        launch(std::false_type{});
    }
}
