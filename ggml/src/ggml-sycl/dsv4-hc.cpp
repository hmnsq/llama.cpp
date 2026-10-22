#समावेश "ggml-impl.h"
#समावेश "dsv4-hc.hpp"

#समावेश <cmath>
#समावेश <type_traits>

अटल constexpr पूर्णांक DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
अटल constexpr पूर्णांक dsv4_hc_pre_block_आकार = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
ढाँचा <bool gated>
अटल व्योम dsv4_hc_pre_f32_sycl(
        स्थिर भग्न * x, स्थिर भग्न * weights, भग्न * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        भग्न scale,
        queue_ptr stream) {
    स्थिर int64_t nr = n_embd * n_tokens;
    स्थिर int64_t अंक_blocks = (nr + dsv4_hc_pre_block_आकार - 1) / dsv4_hc_pre_block_आकार;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(अंक_blocks * dsv4_hc_pre_block_आकार),
                          sycl::range<1>(dsv4_hc_pre_block_आकार)),
        [=](sycl::nd_item<1> item) {
            स्थिर int64_t ir = item.get_global_id(0);
            अगर (ir >= nr) {
                वापस;
            }

            स्थिर int64_t i0 = ir % n_embd;
            स्थिर int64_t it = ir / n_embd;

            भग्न योग_2 = 0.0f;
            क्रम (int64_t ih = 0; ih < hc; ++ih) {
                स्थिर भग्न xv = x[i0*sx0 + ih*sx1 + it*sx2];
                भग्न wv;
                अगर constexpr (gated) {
                    स्थिर भग्न gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::exp(-gv));
                } अन्यथा {
                    wv = weights[ih*sw0 + it*sw1];
                }
                योग_2 += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * योग_2;
        });
}

अटल व्योम dsv4_hc_comb_norm_cols(भग्न * comb, भग्न eps) {
    क्रम (पूर्णांक idst = 0; idst < DSV4_HC; ++idst) {
        भग्न योग_2 = eps;
        क्रम (पूर्णांक isrc = 0; isrc < DSV4_HC; ++isrc) {
            योग_2 += comb[idst + DSV4_HC*isrc];
        }

        स्थिर भग्न inv_योग = 1.0f / योग_2;
        क्रम (पूर्णांक isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_योग;
        }
    }
}

अटल व्योम dsv4_hc_comb_norm_rows(भग्न * comb, भग्न eps) {
    क्रम (पूर्णांक isrc = 0; isrc < DSV4_HC; ++isrc) {
        भग्न योग_2 = eps;
        क्रम (पूर्णांक idst = 0; idst < DSV4_HC; ++idst) {
            योग_2 += comb[idst + DSV4_HC*isrc];
        }

        स्थिर भग्न inv_योग = 1.0f / योग_2;
        क्रम (पूर्णांक idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_योग;
        }
    }
}

अटल व्योम dsv4_hc_comb_f32_sycl(
        स्थिर भग्न * mixes,
        स्थिर भग्न * scale,
        स्थिर भग्न * base,
        भग्न * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        भग्न eps,
        int32_t n_iter,
        queue_ptr stream) {
    constexpr पूर्णांक comb_offset = 2*DSV4_HC;

    स्थिर int64_t block_आकार_2 = 256;
    स्थिर int64_t अंक_blocks = (n_tokens + block_आकार_2 - 1) / block_आकार_2;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(अंक_blocks * block_आकार_2), sycl::range<1>(block_आकार_2)),
        [=](sycl::nd_item<1> item_ct1) {
            स्थिर int64_t it = item_ct1.get_global_id(0);

            अगर (it >= n_tokens) {
                वापस;
            }

            स्थिर भग्न scale_comb = scale[2*ss0];
            भग्न comb[DSV4_HC*DSV4_HC];

            क्रम (पूर्णांक isrc = 0; isrc < DSV4_HC; ++isrc) {
                भग्न max = -अनन्त;
                क्रम (पूर्णांक idst = 0; idst < DSV4_HC; ++idst) {
                    स्थिर पूर्णांक idx = idst + DSV4_HC*isrc;
                    स्थिर भग्न v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                भग्न योग_2 = 0.0f;
                क्रम (पूर्णांक idst = 0; idst < DSV4_HC; ++idst) {
                    स्थिर पूर्णांक idx = idst + DSV4_HC*isrc;
                    स्थिर भग्न v = expf(comb[idx] - max);
                    comb[idx] = v;
                    योग_2 += v;
                }

                स्थिर भग्न inv_योग = 1.0f / योग_2;
                क्रम (पूर्णांक idst = 0; idst < DSV4_HC; ++idst) {
                    स्थिर पूर्णांक idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_योग + eps;
                }
            }

            dsv4_hc_comb_norm_cols(comb, eps);
            क्रम (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_rows(comb, eps);
                dsv4_hc_comb_norm_cols(comb, eps);
            }

            क्रम (पूर्णांक isrc = 0; isrc < DSV4_HC; ++isrc) {
                क्रम (पूर्णांक idst = 0; idst < DSV4_HC; ++idst) {
                    स्थिर पूर्णांक idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
अटल constexpr पूर्णांक dsv4_hc_post_block_आकार = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
ढाँचा <bool has_comb>
अटल व्योम dsv4_hc_post_f32_sycl(
        स्थिर भग्न * x, स्थिर भग्न * residual, स्थिर भग्न * post, स्थिर भग्न * comb, भग्न * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        queue_ptr stream) {
    स्थिर int64_t nr = n_embd * hc * n_tokens;
    स्थिर int64_t block_आकार_2 = dsv4_hc_post_block_आकार;
    स्थिर int64_t अंक_blocks = (nr + block_आकार_2 - 1) / block_आकार_2;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(अंक_blocks * block_आकार_2), sycl::range<1>(block_आकार_2)),
        [=](sycl::nd_item<1> item) {
            स्थिर int64_t ir = item.get_global_id(0);
            अगर (ir >= nr) {
                वापस;
            }

            स्थिर int64_t i0   = ir % n_embd;
            स्थिर int64_t idst = (ir / n_embd) % hc;
            स्थिर int64_t it   = ir / (n_embd * hc);

            भग्न योग_2 = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
            अगर constexpr (has_comb) {
                क्रम (int64_t isrc = 0; isrc < hc; ++isrc) {
                    योग_2 += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } अन्यथा {
                योग_2 += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = योग_2;
        });
}

व्योम ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    स्थिर ggml_tensor * x       = dst->src[0];
    स्थिर ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(माप_प्रकार, nbx, x,       nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbw, weights, nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbd, dst,     nb);

    स्थिर int64_t n_embd   = x->ne[0];
    स्थिर int64_t hc       = x->ne[1];
    स्थिर int64_t n_tokens = x->ne[2];

    स्थिर भग्न scale = ggml_get_op_params_f32(dst, 0);
    स्थिर bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    queue_ptr stream = ctx.stream();

    अगर (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
        dsv4_hc_pre_f32_sycl<true>(
                (स्थिर भग्न *) x->तथ्य, (स्थिर भग्न *) weights->तथ्य, (भग्न *) dst->तथ्य,
                n_embd, hc, n_tokens,
                nbx0 / माप(भग्न), nbx1 / माप(भग्न), nbx2 / माप(भग्न),
                nbw0 / माप(भग्न), nbw1 / माप(भग्न), nbw2 / माप(भग्न),
                nbd0 / माप(भग्न), nbd1 / माप(भग्न),
                scale, stream);
    } अन्यथा {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        dsv4_hc_pre_f32_sycl<false>(
                (स्थिर भग्न *) x->तथ्य, (स्थिर भग्न *) weights->तथ्य, (भग्न *) dst->तथ्य,
                n_embd, hc, n_tokens,
                nbx0 / माप(भग्न), nbx1 / माप(भग्न), nbx2 / माप(भग्न),
                nbw0 / माप(भग्न), nbw1 / माप(भग्न), /*sw2=*/ 0,
                nbd0 / माप(भग्न), nbd1 / माप(भग्न),
                scale, stream);
    }
}

व्योम ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    स्थिर ggml_tensor * mixes = dst->src[0];
    स्थिर ggml_tensor * scale = dst->src[1];
    स्थिर ggml_tensor * base  = dst->src[2];

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

    GGML_TENSOR_LOCALS(माप_प्रकार, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbs, scale, nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbb, base,  nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbd, dst,   nb);

    स्थिर int64_t n_tokens = mixes->ne[1];
    स्थिर भग्न eps = ggml_get_op_params_f32(dst, 0);
    स्थिर int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    queue_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (स्थिर भग्न *) mixes->तथ्य, (स्थिर भग्न *) scale->तथ्य, (स्थिर भग्न *) base->तथ्य, (भग्न *) dst->तथ्य,
            n_tokens,
            nbm0 / माप(भग्न), nbm1 / माप(भग्न),
            nbs0 / माप(भग्न),
            nbb0 / माप(भग्न),
            nbd0 / माप(भग्न), nbd1 / माप(भग्न), nbd2 / माप(भग्न),
            eps, n_iter, stream);
}

व्योम ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    स्थिर ggml_tensor * x        = dst->src[0];
    स्थिर ggml_tensor * residual = dst->src[1];
    स्थिर ggml_tensor * post     = dst->src[2];
    स्थिर ggml_tensor * comb     = dst->src[3];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(residual->type == GGML_TYPE_F32);
    GGML_ASSERT(post->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(माप_प्रकार, nbx, x,        nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbr, residual, nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbp, post,     nb);
    GGML_TENSOR_LOCALS(माप_प्रकार, nbd, dst,      nb);

    माप_प्रकार nbc0 = 0;
    माप_प्रकार nbc1 = 0;
    माप_प्रकार nbc2 = 0;
    अगर (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    स्थिर int64_t n_embd   = x->ne[0];
    स्थिर int64_t n_tokens = x->ne[1];
    स्थिर int64_t hc       = residual->ne[1];

    queue_ptr stream = ctx.stream();

    स्थिर स्वतः launch = [&](स्वतः has_comb) {
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value>(
            (स्थिर भग्न *) x->तथ्य, (स्थिर भग्न *) residual->तथ्य,
            (स्थिर भग्न *) post->तथ्य, comb ? (स्थिर भग्न *) comb->तथ्य : nullptr, (भग्न *) dst->तथ्य,
            n_embd, hc, n_tokens,
            nbx0 / माप(भग्न), nbx1 / माप(भग्न),
            nbr0 / माप(भग्न), nbr1 / माप(भग्न), nbr2 / माप(भग्न),
            nbp0 / माप(भग्न), nbp1 / माप(भग्न),
            nbc0 / माप(भग्न), nbc1 / माप(भग्न), nbc2 / माप(भग्न),
            nbd0 / माप(भग्न), nbd1 / माप(भग्न), nbd2 / माप(भग्न),
            stream);
    };

    अगर (comb) {
        launch(std::true_type{});
    } अन्यथा {
        launch(std::false_type{});
    }
}
