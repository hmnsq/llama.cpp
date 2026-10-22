#അന്തര്ഭാവ "ggml-impl.h"
#അന്തര്ഭാവ "dsv4-hc.hpp"

#അന്തര്ഭാവ <cmath>
#അന്തര്ഭാവ <type_traits>

സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ dsv4_hc_pre_block_size = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
template <തർക്കം gated>
സ്ഥിതം ശൂന്യം dsv4_hc_pre_f32_sycl(
        സ്ഥിരം ദശാംശം * x, സ്ഥിരം ദശാംശം * weights, ദശാംശം * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        ദശാംശം scale,
        queue_ptr stream) {
    സ്ഥിരം int64_t nr = n_embd * n_tokens;
    സ്ഥിരം int64_t അക്കം_blocks = (nr + dsv4_hc_pre_block_size - 1) / dsv4_hc_pre_block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks * dsv4_hc_pre_block_size),
                          sycl::range<1>(dsv4_hc_pre_block_size)),
        [=](sycl::nd_item<1> item) {
            സ്ഥിരം int64_t ir = item.get_global_id(0);
            എങ്കിൽ (ir >= nr) {
                തിരികെ;
            }

            സ്ഥിരം int64_t i0 = ir % n_embd;
            സ്ഥിരം int64_t it = ir / n_embd;

            ദശാംശം sum = 0.0f;
            വേണ്ടി (int64_t ih = 0; ih < hc; ++ih) {
                സ്ഥിരം ദശാംശം xv = x[i0*sx0 + ih*sx1 + it*sx2];
                ദശാംശം wv;
                എങ്കിൽ constexpr (gated) {
                    സ്ഥിരം ദശാംശം gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::exp(-gv));
                } അല്ലെങ്കിൽ {
                    wv = weights[ih*sw0 + it*sw1];
                }
                sum += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * sum;
        });
}

സ്ഥിതം ശൂന്യം dsv4_hc_comb_norm_cols(ദശാംശം * comb, ദശാംശം eps) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ idst = 0; idst < DSV4_HC; ++idst) {
        ദശാംശം sum = eps;
        വേണ്ടി (പൂർണ്ണസംഖ്യ isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        സ്ഥിരം ദശാംശം inv_sum = 1.0f / sum;
        വേണ്ടി (പൂർണ്ണസംഖ്യ isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

സ്ഥിതം ശൂന്യം dsv4_hc_comb_norm_rows(ദശാംശം * comb, ദശാംശം eps) {
    വേണ്ടി (പൂർണ്ണസംഖ്യ isrc = 0; isrc < DSV4_HC; ++isrc) {
        ദശാംശം sum = eps;
        വേണ്ടി (പൂർണ്ണസംഖ്യ idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        സ്ഥിരം ദശാംശം inv_sum = 1.0f / sum;
        വേണ്ടി (പൂർണ്ണസംഖ്യ idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

സ്ഥിതം ശൂന്യം dsv4_hc_comb_f32_sycl(
        സ്ഥിരം ദശാംശം * mixes,
        സ്ഥിരം ദശാംശം * scale,
        സ്ഥിരം ദശാംശം * base,
        ദശാംശം * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        ദശാംശം eps,
        int32_t n_iter,
        queue_ptr stream) {
    constexpr പൂർണ്ണസംഖ്യ comb_offset = 2*DSV4_HC;

    സ്ഥിരം int64_t block_size = 256;
    സ്ഥിരം int64_t അക്കം_blocks = (n_tokens + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item_ct1) {
            സ്ഥിരം int64_t it = item_ct1.get_global_id(0);

            എങ്കിൽ (it >= n_tokens) {
                തിരികെ;
            }

            സ്ഥിരം ദശാംശം scale_comb = scale[2*ss0];
            ദശാംശം comb[DSV4_HC*DSV4_HC];

            വേണ്ടി (പൂർണ്ണസംഖ്യ isrc = 0; isrc < DSV4_HC; ++isrc) {
                ദശാംശം max = -INFINITY;
                വേണ്ടി (പൂർണ്ണസംഖ്യ idst = 0; idst < DSV4_HC; ++idst) {
                    സ്ഥിരം പൂർണ്ണസംഖ്യ idx = idst + DSV4_HC*isrc;
                    സ്ഥിരം ദശാംശം v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                ദശാംശം sum = 0.0f;
                വേണ്ടി (പൂർണ്ണസംഖ്യ idst = 0; idst < DSV4_HC; ++idst) {
                    സ്ഥിരം പൂർണ്ണസംഖ്യ idx = idst + DSV4_HC*isrc;
                    സ്ഥിരം ദശാംശം v = expf(comb[idx] - max);
                    comb[idx] = v;
                    sum += v;
                }

                സ്ഥിരം ദശാംശം inv_sum = 1.0f / sum;
                വേണ്ടി (പൂർണ്ണസംഖ്യ idst = 0; idst < DSV4_HC; ++idst) {
                    സ്ഥിരം പൂർണ്ണസംഖ്യ idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_sum + eps;
                }
            }

            dsv4_hc_comb_norm_cols(comb, eps);
            വേണ്ടി (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_rows(comb, eps);
                dsv4_hc_comb_norm_cols(comb, eps);
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ isrc = 0; isrc < DSV4_HC; ++isrc) {
                വേണ്ടി (പൂർണ്ണസംഖ്യ idst = 0; idst < DSV4_HC; ++idst) {
                    സ്ഥിരം പൂർണ്ണസംഖ്യ idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
സ്ഥിതം constexpr പൂർണ്ണസംഖ്യ dsv4_hc_post_block_size = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
template <തർക്കം has_comb>
സ്ഥിതം ശൂന്യം dsv4_hc_post_f32_sycl(
        സ്ഥിരം ദശാംശം * x, സ്ഥിരം ദശാംശം * residual, സ്ഥിരം ദശാംശം * post, സ്ഥിരം ദശാംശം * comb, ദശാംശം * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        queue_ptr stream) {
    സ്ഥിരം int64_t nr = n_embd * hc * n_tokens;
    സ്ഥിരം int64_t block_size = dsv4_hc_post_block_size;
    സ്ഥിരം int64_t അക്കം_blocks = (nr + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(അക്കം_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item) {
            സ്ഥിരം int64_t ir = item.get_global_id(0);
            എങ്കിൽ (ir >= nr) {
                തിരികെ;
            }

            സ്ഥിരം int64_t i0   = ir % n_embd;
            സ്ഥിരം int64_t idst = (ir / n_embd) % hc;
            സ്ഥിരം int64_t it   = ir / (n_embd * hc);

            ദശാംശം sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
            എങ്കിൽ constexpr (has_comb) {
                വേണ്ടി (int64_t isrc = 0; isrc < hc; ++isrc) {
                    sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } അല്ലെങ്കിൽ {
                sum += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
        });
}

ശൂന്യം ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    സ്ഥിരം ggml_tensor * x       = dst->src[0];
    സ്ഥിരം ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    സ്ഥിരം int64_t n_embd   = x->ne[0];
    സ്ഥിരം int64_t hc       = x->ne[1];
    സ്ഥിരം int64_t n_tokens = x->ne[2];

    സ്ഥിരം ദശാംശം scale = ggml_get_op_params_f32(dst, 0);
    സ്ഥിരം തർക്കം  gated = ggml_get_op_params_i32(dst, 1) != 0;

    queue_ptr stream = ctx.stream();

    എങ്കിൽ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
        dsv4_hc_pre_f32_sycl<സത്യം>(
                (സ്ഥിരം ദശാംശം *) x->data, (സ്ഥിരം ദശാംശം *) weights->data, (ദശാംശം *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / വലുപ്പം(ദശാംശം), nbx1 / വലുപ്പം(ദശാംശം), nbx2 / വലുപ്പം(ദശാംശം),
                nbw0 / വലുപ്പം(ദശാംശം), nbw1 / വലുപ്പം(ദശാംശം), nbw2 / വലുപ്പം(ദശാംശം),
                nbd0 / വലുപ്പം(ദശാംശം), nbd1 / വലുപ്പം(ദശാംശം),
                scale, stream);
    } അല്ലെങ്കിൽ {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        dsv4_hc_pre_f32_sycl<അസത്യം>(
                (സ്ഥിരം ദശാംശം *) x->data, (സ്ഥിരം ദശാംശം *) weights->data, (ദശാംശം *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / വലുപ്പം(ദശാംശം), nbx1 / വലുപ്പം(ദശാംശം), nbx2 / വലുപ്പം(ദശാംശം),
                nbw0 / വലുപ്പം(ദശാംശം), nbw1 / വലുപ്പം(ദശാംശം), /*sw2=*/ 0,
                nbd0 / വലുപ്പം(ദശാംശം), nbd1 / വലുപ്പം(ദശാംശം),
                scale, stream);
    }
}

ശൂന്യം ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    സ്ഥിരം ggml_tensor * mixes = dst->src[0];
    സ്ഥിരം ggml_tensor * scale = dst->src[1];
    സ്ഥിരം ggml_tensor * base  = dst->src[2];

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

    സ്ഥിരം int64_t n_tokens = mixes->ne[1];
    സ്ഥിരം ദശാംശം eps = ggml_get_op_params_f32(dst, 0);
    സ്ഥിരം int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    queue_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (സ്ഥിരം ദശാംശം *) mixes->data, (സ്ഥിരം ദശാംശം *) scale->data, (സ്ഥിരം ദശാംശം *) base->data, (ദശാംശം *) dst->data,
            n_tokens,
            nbm0 / വലുപ്പം(ദശാംശം), nbm1 / വലുപ്പം(ദശാംശം),
            nbs0 / വലുപ്പം(ദശാംശം),
            nbb0 / വലുപ്പം(ദശാംശം),
            nbd0 / വലുപ്പം(ദശാംശം), nbd1 / വലുപ്പം(ദശാംശം), nbd2 / വലുപ്പം(ദശാംശം),
            eps, n_iter, stream);
}

ശൂന്യം ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    സ്ഥിരം ggml_tensor * x        = dst->src[0];
    സ്ഥിരം ggml_tensor * residual = dst->src[1];
    സ്ഥിരം ggml_tensor * post     = dst->src[2];
    സ്ഥിരം ggml_tensor * comb     = dst->src[3];

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
    എങ്കിൽ (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    സ്ഥിരം int64_t n_embd   = x->ne[0];
    സ്ഥിരം int64_t n_tokens = x->ne[1];
    സ്ഥിരം int64_t hc       = residual->ne[1];

    queue_ptr stream = ctx.stream();

    സ്ഥിരം auto launch = [&](auto has_comb) {
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value>(
            (സ്ഥിരം ദശാംശം *) x->data, (സ്ഥിരം ദശാംശം *) residual->data,
            (സ്ഥിരം ദശാംശം *) post->data, comb ? (സ്ഥിരം ദശാംശം *) comb->data : nullptr, (ദശാംശം *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / വലുപ്പം(ദശാംശം), nbx1 / വലുപ്പം(ദശാംശം),
            nbr0 / വലുപ്പം(ദശാംശം), nbr1 / വലുപ്പം(ദശാംശം), nbr2 / വലുപ്പം(ദശാംശം),
            nbp0 / വലുപ്പം(ദശാംശം), nbp1 / വലുപ്പം(ദശാംശം),
            nbc0 / വലുപ്പം(ദശാംശം), nbc1 / വലുപ്പം(ദശാംശം), nbc2 / വലുപ്പം(ദശാംശം),
            nbd0 / വലുപ്പം(ദശാംശം), nbd1 / വലുപ്പം(ദശാംശം), nbd2 / വലുപ്പം(ദശാംശം),
            stream);
    };

    എങ്കിൽ (comb) {
        launch(std::true_type{});
    } അല്ലെങ്കിൽ {
        launch(std::false_type{});
    }
}
