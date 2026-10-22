#अन्तर्भाव "ggml-impl.h"
#अन्तर्भाव "dsv4-hc.hpp"

#अन्तर्भाव <cmath>
#अन्तर्भाव <type_traits>

स्थिर constexpr पूर्णाङ्क DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
स्थिर constexpr पूर्णाङ्क dsv4_hc_pre_block_size = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
template <तार्किक gated>
स्थिर शून्य dsv4_hc_pre_f32_sycl(
        ध्रुव भिन्न * x, ध्रुव भिन्न * weights, भिन्न * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        भिन्न scale,
        queue_ptr stream) {
    ध्रुव int64_t nr = n_embd * n_tokens;
    ध्रुव int64_t अंक_blocks = (nr + dsv4_hc_pre_block_size - 1) / dsv4_hc_pre_block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(अंक_blocks * dsv4_hc_pre_block_size),
                          sycl::range<1>(dsv4_hc_pre_block_size)),
        [=](sycl::nd_item<1> item) {
            ध्रुव int64_t ir = item.get_global_id(0);
            यदि (ir >= nr) {
                प्रत्यागम;
            }

            ध्रुव int64_t i0 = ir % n_embd;
            ध्रुव int64_t it = ir / n_embd;

            भिन्न sum = 0.0f;
            प्रति (int64_t ih = 0; ih < hc; ++ih) {
                ध्रुव भिन्न xv = x[i0*sx0 + ih*sx1 + it*sx2];
                भिन्न wv;
                यदि constexpr (gated) {
                    ध्रुव भिन्न gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::exp(-gv));
                } अन्यथा {
                    wv = weights[ih*sw0 + it*sw1];
                }
                sum += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * sum;
        });
}

स्थिर शून्य dsv4_hc_comb_norm_cols(भिन्न * comb, भिन्न eps) {
    प्रति (पूर्णाङ्क idst = 0; idst < DSV4_HC; ++idst) {
        भिन्न sum = eps;
        प्रति (पूर्णाङ्क isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        ध्रुव भिन्न inv_sum = 1.0f / sum;
        प्रति (पूर्णाङ्क isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

स्थिर शून्य dsv4_hc_comb_norm_rows(भिन्न * comb, भिन्न eps) {
    प्रति (पूर्णाङ्क isrc = 0; isrc < DSV4_HC; ++isrc) {
        भिन्न sum = eps;
        प्रति (पूर्णाङ्क idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        ध्रुव भिन्न inv_sum = 1.0f / sum;
        प्रति (पूर्णाङ्क idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

स्थिर शून्य dsv4_hc_comb_f32_sycl(
        ध्रुव भिन्न * mixes,
        ध्रुव भिन्न * scale,
        ध्रुव भिन्न * base,
        भिन्न * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        भिन्न eps,
        int32_t n_iter,
        queue_ptr stream) {
    constexpr पूर्णाङ्क comb_offset = 2*DSV4_HC;

    ध्रुव int64_t block_size = 256;
    ध्रुव int64_t अंक_blocks = (n_tokens + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(अंक_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item_ct1) {
            ध्रुव int64_t it = item_ct1.get_global_id(0);

            यदि (it >= n_tokens) {
                प्रत्यागम;
            }

            ध्रुव भिन्न scale_comb = scale[2*ss0];
            भिन्न comb[DSV4_HC*DSV4_HC];

            प्रति (पूर्णाङ्क isrc = 0; isrc < DSV4_HC; ++isrc) {
                भिन्न max = -INFINITY;
                प्रति (पूर्णाङ्क idst = 0; idst < DSV4_HC; ++idst) {
                    ध्रुव पूर्णाङ्क idx = idst + DSV4_HC*isrc;
                    ध्रुव भिन्न v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                भिन्न sum = 0.0f;
                प्रति (पूर्णाङ्क idst = 0; idst < DSV4_HC; ++idst) {
                    ध्रुव पूर्णाङ्क idx = idst + DSV4_HC*isrc;
                    ध्रुव भिन्न v = expf(comb[idx] - max);
                    comb[idx] = v;
                    sum += v;
                }

                ध्रुव भिन्न inv_sum = 1.0f / sum;
                प्रति (पूर्णाङ्क idst = 0; idst < DSV4_HC; ++idst) {
                    ध्रुव पूर्णाङ्क idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_sum + eps;
                }
            }

            dsv4_hc_comb_norm_cols(comb, eps);
            प्रति (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_rows(comb, eps);
                dsv4_hc_comb_norm_cols(comb, eps);
            }

            प्रति (पूर्णाङ्क isrc = 0; isrc < DSV4_HC; ++isrc) {
                प्रति (पूर्णाङ्क idst = 0; idst < DSV4_HC; ++idst) {
                    ध्रुव पूर्णाङ्क idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
स्थिर constexpr पूर्णाङ्क dsv4_hc_post_block_size = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
template <तार्किक has_comb>
स्थिर शून्य dsv4_hc_post_f32_sycl(
        ध्रुव भिन्न * x, ध्रुव भिन्न * residual, ध्रुव भिन्न * post, ध्रुव भिन्न * comb, भिन्न * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        queue_ptr stream) {
    ध्रुव int64_t nr = n_embd * hc * n_tokens;
    ध्रुव int64_t block_size = dsv4_hc_post_block_size;
    ध्रुव int64_t अंक_blocks = (nr + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(अंक_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item) {
            ध्रुव int64_t ir = item.get_global_id(0);
            यदि (ir >= nr) {
                प्रत्यागम;
            }

            ध्रुव int64_t i0   = ir % n_embd;
            ध्रुव int64_t idst = (ir / n_embd) % hc;
            ध्रुव int64_t it   = ir / (n_embd * hc);

            भिन्न sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
            यदि constexpr (has_comb) {
                प्रति (int64_t isrc = 0; isrc < hc; ++isrc) {
                    sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } अन्यथा {
                sum += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
        });
}

शून्य ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ध्रुव ggml_tensor * x       = dst->src[0];
    ध्रुव ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    ध्रुव int64_t n_embd   = x->ne[0];
    ध्रुव int64_t hc       = x->ne[1];
    ध्रुव int64_t n_tokens = x->ne[2];

    ध्रुव भिन्न scale = ggml_get_op_params_f32(dst, 0);
    ध्रुव तार्किक  gated = ggml_get_op_params_i32(dst, 1) != 0;

    queue_ptr stream = ctx.stream();

    यदि (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
        dsv4_hc_pre_f32_sycl<सत्य>(
                (ध्रुव भिन्न *) x->data, (ध्रुव भिन्न *) weights->data, (भिन्न *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / परिमाण(भिन्न), nbx1 / परिमाण(भिन्न), nbx2 / परिमाण(भिन्न),
                nbw0 / परिमाण(भिन्न), nbw1 / परिमाण(भिन्न), nbw2 / परिमाण(भिन्न),
                nbd0 / परिमाण(भिन्न), nbd1 / परिमाण(भिन्न),
                scale, stream);
    } अन्यथा {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        dsv4_hc_pre_f32_sycl<असत्य>(
                (ध्रुव भिन्न *) x->data, (ध्रुव भिन्न *) weights->data, (भिन्न *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / परिमाण(भिन्न), nbx1 / परिमाण(भिन्न), nbx2 / परिमाण(भिन्न),
                nbw0 / परिमाण(भिन्न), nbw1 / परिमाण(भिन्न), /*sw2=*/ 0,
                nbd0 / परिमाण(भिन्न), nbd1 / परिमाण(भिन्न),
                scale, stream);
    }
}

शून्य ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ध्रुव ggml_tensor * mixes = dst->src[0];
    ध्रुव ggml_tensor * scale = dst->src[1];
    ध्रुव ggml_tensor * base  = dst->src[2];

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

    ध्रुव int64_t n_tokens = mixes->ne[1];
    ध्रुव भिन्न eps = ggml_get_op_params_f32(dst, 0);
    ध्रुव int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    queue_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (ध्रुव भिन्न *) mixes->data, (ध्रुव भिन्न *) scale->data, (ध्रुव भिन्न *) base->data, (भिन्न *) dst->data,
            n_tokens,
            nbm0 / परिमाण(भिन्न), nbm1 / परिमाण(भिन्न),
            nbs0 / परिमाण(भिन्न),
            nbb0 / परिमाण(भिन्न),
            nbd0 / परिमाण(भिन्न), nbd1 / परिमाण(भिन्न), nbd2 / परिमाण(भिन्न),
            eps, n_iter, stream);
}

शून्य ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    ध्रुव ggml_tensor * x        = dst->src[0];
    ध्रुव ggml_tensor * residual = dst->src[1];
    ध्रुव ggml_tensor * post     = dst->src[2];
    ध्रुव ggml_tensor * comb     = dst->src[3];

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
    यदि (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    ध्रुव int64_t n_embd   = x->ne[0];
    ध्रुव int64_t n_tokens = x->ne[1];
    ध्रुव int64_t hc       = residual->ne[1];

    queue_ptr stream = ctx.stream();

    ध्रुव auto launch = [&](auto has_comb) {
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value>(
            (ध्रुव भिन्न *) x->data, (ध्रुव भिन्न *) residual->data,
            (ध्रुव भिन्न *) post->data, comb ? (ध्रुव भिन्न *) comb->data : nullptr, (भिन्न *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / परिमाण(भिन्न), nbx1 / परिमाण(भिन्न),
            nbr0 / परिमाण(भिन्न), nbr1 / परिमाण(भिन्न), nbr2 / परिमाण(भिन्न),
            nbp0 / परिमाण(भिन्न), nbp1 / परिमाण(भिन्न),
            nbc0 / परिमाण(भिन्न), nbc1 / परिमाण(भिन्न), nbc2 / परिमाण(भिन्न),
            nbd0 / परिमाण(भिन्न), nbd1 / परिमाण(भिन्न), nbd2 / परिमाण(भिन्न),
            stream);
    };

    यदि (comb) {
        launch(std::true_type{});
    } अन्यथा {
        launch(std::false_type{});
    }
}
