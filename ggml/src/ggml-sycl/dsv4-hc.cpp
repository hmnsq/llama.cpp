#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "dsv4-hc.hpp"

#அந்தர்பாவ <cmath>
#அந்தர்பாவ <type_traits>

நிலையான constexpr முழு DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
நிலையான constexpr முழு dsv4_hc_pre_block_size = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
template <தருக்க gated>
நிலையான வெறுமை dsv4_hc_pre_f32_sycl(
        மாறிலி தசம * x, மாறிலி தசம * weights, தசம * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        தசம scale,
        queue_ptr stream) {
    மாறிலி int64_t nr = n_embd * n_tokens;
    மாறிலி int64_t இலக்கம்_blocks = (nr + dsv4_hc_pre_block_size - 1) / dsv4_hc_pre_block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks * dsv4_hc_pre_block_size),
                          sycl::range<1>(dsv4_hc_pre_block_size)),
        [=](sycl::nd_item<1> item) {
            மாறிலி int64_t ir = item.get_global_id(0);
            எனில் (ir >= nr) {
                திரும்பு;
            }

            மாறிலி int64_t i0 = ir % n_embd;
            மாறிலி int64_t it = ir / n_embd;

            தசம sum = 0.0f;
            க்கு (int64_t ih = 0; ih < hc; ++ih) {
                மாறிலி தசம xv = x[i0*sx0 + ih*sx1 + it*sx2];
                தசம wv;
                எனில் constexpr (gated) {
                    மாறிலி தசம gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::exp(-gv));
                } இல்லையெனில் {
                    wv = weights[ih*sw0 + it*sw1];
                }
                sum += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * sum;
        });
}

நிலையான வெறுமை dsv4_hc_comb_norm_cols(தசம * comb, தசம eps) {
    க்கு (முழு idst = 0; idst < DSV4_HC; ++idst) {
        தசம sum = eps;
        க்கு (முழு isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        மாறிலி தசம inv_sum = 1.0f / sum;
        க்கு (முழு isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

நிலையான வெறுமை dsv4_hc_comb_norm_rows(தசம * comb, தசம eps) {
    க்கு (முழு isrc = 0; isrc < DSV4_HC; ++isrc) {
        தசம sum = eps;
        க்கு (முழு idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        மாறிலி தசம inv_sum = 1.0f / sum;
        க்கு (முழு idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

நிலையான வெறுமை dsv4_hc_comb_f32_sycl(
        மாறிலி தசம * mixes,
        மாறிலி தசம * scale,
        மாறிலி தசம * base,
        தசம * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        தசம eps,
        int32_t n_iter,
        queue_ptr stream) {
    constexpr முழு comb_offset = 2*DSV4_HC;

    மாறிலி int64_t block_size = 256;
    மாறிலி int64_t இலக்கம்_blocks = (n_tokens + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item_ct1) {
            மாறிலி int64_t it = item_ct1.get_global_id(0);

            எனில் (it >= n_tokens) {
                திரும்பு;
            }

            மாறிலி தசம scale_comb = scale[2*ss0];
            தசம comb[DSV4_HC*DSV4_HC];

            க்கு (முழு isrc = 0; isrc < DSV4_HC; ++isrc) {
                தசம max = -INFINITY;
                க்கு (முழு idst = 0; idst < DSV4_HC; ++idst) {
                    மாறிலி முழு idx = idst + DSV4_HC*isrc;
                    மாறிலி தசம v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                தசம sum = 0.0f;
                க்கு (முழு idst = 0; idst < DSV4_HC; ++idst) {
                    மாறிலி முழு idx = idst + DSV4_HC*isrc;
                    மாறிலி தசம v = expf(comb[idx] - max);
                    comb[idx] = v;
                    sum += v;
                }

                மாறிலி தசம inv_sum = 1.0f / sum;
                க்கு (முழு idst = 0; idst < DSV4_HC; ++idst) {
                    மாறிலி முழு idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_sum + eps;
                }
            }

            dsv4_hc_comb_norm_cols(comb, eps);
            க்கு (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_rows(comb, eps);
                dsv4_hc_comb_norm_cols(comb, eps);
            }

            க்கு (முழு isrc = 0; isrc < DSV4_HC; ++isrc) {
                க்கு (முழு idst = 0; idst < DSV4_HC; ++idst) {
                    மாறிலி முழு idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
நிலையான constexpr முழு dsv4_hc_post_block_size = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
template <தருக்க has_comb>
நிலையான வெறுமை dsv4_hc_post_f32_sycl(
        மாறிலி தசம * x, மாறிலி தசம * residual, மாறிலி தசம * post, மாறிலி தசம * comb, தசம * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        queue_ptr stream) {
    மாறிலி int64_t nr = n_embd * hc * n_tokens;
    மாறிலி int64_t block_size = dsv4_hc_post_block_size;
    மாறிலி int64_t இலக்கம்_blocks = (nr + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(இலக்கம்_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item) {
            மாறிலி int64_t ir = item.get_global_id(0);
            எனில் (ir >= nr) {
                திரும்பு;
            }

            மாறிலி int64_t i0   = ir % n_embd;
            மாறிலி int64_t idst = (ir / n_embd) % hc;
            மாறிலி int64_t it   = ir / (n_embd * hc);

            தசம sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
            எனில் constexpr (has_comb) {
                க்கு (int64_t isrc = 0; isrc < hc; ++isrc) {
                    sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } இல்லையெனில் {
                sum += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
        });
}

வெறுமை ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    மாறிலி ggml_tensor * x       = dst->src[0];
    மாறிலி ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    மாறிலி int64_t n_embd   = x->ne[0];
    மாறிலி int64_t hc       = x->ne[1];
    மாறிலி int64_t n_tokens = x->ne[2];

    மாறிலி தசம scale = ggml_get_op_params_f32(dst, 0);
    மாறிலி தருக்க  gated = ggml_get_op_params_i32(dst, 1) != 0;

    queue_ptr stream = ctx.stream();

    எனில் (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
        dsv4_hc_pre_f32_sycl<உண்மை>(
                (மாறிலி தசம *) x->data, (மாறிலி தசம *) weights->data, (தசம *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / அளவு(தசம), nbx1 / அளவு(தசம), nbx2 / அளவு(தசம),
                nbw0 / அளவு(தசம), nbw1 / அளவு(தசம), nbw2 / அளவு(தசம),
                nbd0 / அளவு(தசம), nbd1 / அளவு(தசம),
                scale, stream);
    } இல்லையெனில் {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        dsv4_hc_pre_f32_sycl<பொய்>(
                (மாறிலி தசம *) x->data, (மாறிலி தசம *) weights->data, (தசம *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / அளவு(தசம), nbx1 / அளவு(தசம), nbx2 / அளவு(தசம),
                nbw0 / அளவு(தசம), nbw1 / அளவு(தசம), /*sw2=*/ 0,
                nbd0 / அளவு(தசம), nbd1 / அளவு(தசம),
                scale, stream);
    }
}

வெறுமை ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    மாறிலி ggml_tensor * mixes = dst->src[0];
    மாறிலி ggml_tensor * scale = dst->src[1];
    மாறிலி ggml_tensor * base  = dst->src[2];

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

    மாறிலி int64_t n_tokens = mixes->ne[1];
    மாறிலி தசம eps = ggml_get_op_params_f32(dst, 0);
    மாறிலி int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    queue_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (மாறிலி தசம *) mixes->data, (மாறிலி தசம *) scale->data, (மாறிலி தசம *) base->data, (தசம *) dst->data,
            n_tokens,
            nbm0 / அளவு(தசம), nbm1 / அளவு(தசம),
            nbs0 / அளவு(தசம),
            nbb0 / அளவு(தசம),
            nbd0 / அளவு(தசம), nbd1 / அளவு(தசம), nbd2 / அளவு(தசம),
            eps, n_iter, stream);
}

வெறுமை ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    மாறிலி ggml_tensor * x        = dst->src[0];
    மாறிலி ggml_tensor * residual = dst->src[1];
    மாறிலி ggml_tensor * post     = dst->src[2];
    மாறிலி ggml_tensor * comb     = dst->src[3];

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
    எனில் (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    மாறிலி int64_t n_embd   = x->ne[0];
    மாறிலி int64_t n_tokens = x->ne[1];
    மாறிலி int64_t hc       = residual->ne[1];

    queue_ptr stream = ctx.stream();

    மாறிலி auto launch = [&](auto has_comb) {
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value>(
            (மாறிலி தசம *) x->data, (மாறிலி தசம *) residual->data,
            (மாறிலி தசம *) post->data, comb ? (மாறிலி தசம *) comb->data : nullptr, (தசம *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / அளவு(தசம), nbx1 / அளவு(தசம),
            nbr0 / அளவு(தசம), nbr1 / அளவு(தசம), nbr2 / அளவு(தசம),
            nbp0 / அளவு(தசம), nbp1 / அளவு(தசம),
            nbc0 / அளவு(தசம), nbc1 / அளவு(தசம), nbc2 / அளவு(தசம),
            nbd0 / அளவு(தசம), nbd1 / அளவு(தசம), nbd2 / அளவு(தசம),
            stream);
    };

    எனில் (comb) {
        launch(std::true_type{});
    } இல்லையெனில் {
        launch(std::false_type{});
    }
}
