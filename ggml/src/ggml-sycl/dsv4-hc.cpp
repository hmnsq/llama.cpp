#ضمن "ggml-impl.h"
#ضمن "dsv4-hc.hpp"

#ضمن <cmath>
#ضمن <type_traits>

ساكن constexpr صحيح DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
ساكن constexpr صحيح dsv4_hc_pre_block_size = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
template <منطقي gated>
ساكن فراغ dsv4_hc_pre_f32_sycl(
        ثابت عائم * x, ثابت عائم * weights, عائم * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        عائم scale,
        queue_ptr stream) {
    ثابت int64_t nr = n_embd * n_tokens;
    ثابت int64_t ژمارە_blocks = (nr + dsv4_hc_pre_block_size - 1) / dsv4_hc_pre_block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ژمارە_blocks * dsv4_hc_pre_block_size),
                          sycl::range<1>(dsv4_hc_pre_block_size)),
        [=](sycl::nd_item<1> item) {
            ثابت int64_t ir = item.get_global_id(0);
            إذا (ir >= nr) {
                أرجع;
            }

            ثابت int64_t i0 = ir % n_embd;
            ثابت int64_t it = ir / n_embd;

            عائم sum = 0.0f;
            لكل (int64_t ih = 0; ih < hc; ++ih) {
                ثابت عائم xv = x[i0*sx0 + ih*sx1 + it*sx2];
                عائم wv;
                إذا constexpr (gated) {
                    ثابت عائم gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::exp(-gv));
                } وإلا {
                    wv = weights[ih*sw0 + it*sw1];
                }
                sum += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * sum;
        });
}

ساكن فراغ dsv4_hc_comb_norm_cols(عائم * comb, عائم eps) {
    لكل (صحيح idst = 0; idst < DSV4_HC; ++idst) {
        عائم sum = eps;
        لكل (صحيح isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        ثابت عائم inv_sum = 1.0f / sum;
        لكل (صحيح isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

ساكن فراغ dsv4_hc_comb_norm_rows(عائم * comb, عائم eps) {
    لكل (صحيح isrc = 0; isrc < DSV4_HC; ++isrc) {
        عائم sum = eps;
        لكل (صحيح idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        ثابت عائم inv_sum = 1.0f / sum;
        لكل (صحيح idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

ساكن فراغ dsv4_hc_comb_f32_sycl(
        ثابت عائم * mixes,
        ثابت عائم * scale,
        ثابت عائم * base,
        عائم * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        عائم eps,
        int32_t n_iter,
        queue_ptr stream) {
    constexpr صحيح comb_offset = 2*DSV4_HC;

    ثابت int64_t block_size = 256;
    ثابت int64_t ژمارە_blocks = (n_tokens + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ژمارە_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item_ct1) {
            ثابت int64_t it = item_ct1.get_global_id(0);

            إذا (it >= n_tokens) {
                أرجع;
            }

            ثابت عائم scale_comb = scale[2*ss0];
            عائم comb[DSV4_HC*DSV4_HC];

            لكل (صحيح isrc = 0; isrc < DSV4_HC; ++isrc) {
                عائم max = -INFINITY;
                لكل (صحيح idst = 0; idst < DSV4_HC; ++idst) {
                    ثابت صحيح idx = idst + DSV4_HC*isrc;
                    ثابت عائم v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                عائم sum = 0.0f;
                لكل (صحيح idst = 0; idst < DSV4_HC; ++idst) {
                    ثابت صحيح idx = idst + DSV4_HC*isrc;
                    ثابت عائم v = expf(comb[idx] - max);
                    comb[idx] = v;
                    sum += v;
                }

                ثابت عائم inv_sum = 1.0f / sum;
                لكل (صحيح idst = 0; idst < DSV4_HC; ++idst) {
                    ثابت صحيح idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_sum + eps;
                }
            }

            dsv4_hc_comb_norm_cols(comb, eps);
            لكل (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_rows(comb, eps);
                dsv4_hc_comb_norm_cols(comb, eps);
            }

            لكل (صحيح isrc = 0; isrc < DSV4_HC; ++isrc) {
                لكل (صحيح idst = 0; idst < DSV4_HC; ++idst) {
                    ثابت صحيح idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
ساكن constexpr صحيح dsv4_hc_post_block_size = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
template <منطقي has_comb>
ساكن فراغ dsv4_hc_post_f32_sycl(
        ثابت عائم * x, ثابت عائم * residual, ثابت عائم * post, ثابت عائم * comb, عائم * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        queue_ptr stream) {
    ثابت int64_t nr = n_embd * hc * n_tokens;
    ثابت int64_t block_size = dsv4_hc_post_block_size;
    ثابت int64_t ژمارە_blocks = (nr + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(ژمارە_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item) {
            ثابت int64_t ir = item.get_global_id(0);
            إذا (ir >= nr) {
                أرجع;
            }

            ثابت int64_t i0   = ir % n_embd;
            ثابت int64_t idst = (ir / n_embd) % hc;
            ثابت int64_t it   = ir / (n_embd * hc);

            عائم sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
            إذا constexpr (has_comb) {
                لكل (int64_t isrc = 0; isrc < hc; ++isrc) {
                    sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } وإلا {
                sum += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
        });
}

فراغ ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * x       = dst->src[0];
    ثابت ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    ثابت int64_t n_embd   = x->ne[0];
    ثابت int64_t hc       = x->ne[1];
    ثابت int64_t n_tokens = x->ne[2];

    ثابت عائم scale = ggml_get_op_params_f32(dst, 0);
    ثابت منطقي  gated = ggml_get_op_params_i32(dst, 1) != 0;

    queue_ptr stream = ctx.stream();

    إذا (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
        dsv4_hc_pre_f32_sycl<true>(
                (ثابت عائم *) x->data, (ثابت عائم *) weights->data, (عائم *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / حجم(عائم), nbx1 / حجم(عائم), nbx2 / حجم(عائم),
                nbw0 / حجم(عائم), nbw1 / حجم(عائم), nbw2 / حجم(عائم),
                nbd0 / حجم(عائم), nbd1 / حجم(عائم),
                scale, stream);
    } وإلا {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        dsv4_hc_pre_f32_sycl<خطأ>(
                (ثابت عائم *) x->data, (ثابت عائم *) weights->data, (عائم *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / حجم(عائم), nbx1 / حجم(عائم), nbx2 / حجم(عائم),
                nbw0 / حجم(عائم), nbw1 / حجم(عائم), /*sw2=*/ 0,
                nbd0 / حجم(عائم), nbd1 / حجم(عائم),
                scale, stream);
    }
}

فراغ ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ثابت ggml_tensor * mixes = dst->src[0];
    ثابت ggml_tensor * scale = dst->src[1];
    ثابت ggml_tensor * base  = dst->src[2];

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

    ثابت int64_t n_tokens = mixes->ne[1];
    ثابت عائم eps = ggml_get_op_params_f32(dst, 0);
    ثابت int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    queue_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (ثابت عائم *) mixes->data, (ثابت عائم *) scale->data, (ثابت عائم *) base->data, (عائم *) dst->data,
            n_tokens,
            nbm0 / حجم(عائم), nbm1 / حجم(عائم),
            nbs0 / حجم(عائم),
            nbb0 / حجم(عائم),
            nbd0 / حجم(عائم), nbd1 / حجم(عائم), nbd2 / حجم(عائم),
            eps, n_iter, stream);
}

فراغ ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    ثابت ggml_tensor * x        = dst->src[0];
    ثابت ggml_tensor * residual = dst->src[1];
    ثابت ggml_tensor * post     = dst->src[2];
    ثابت ggml_tensor * comb     = dst->src[3];

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
    إذا (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    ثابت int64_t n_embd   = x->ne[0];
    ثابت int64_t n_tokens = x->ne[1];
    ثابت int64_t hc       = residual->ne[1];

    queue_ptr stream = ctx.stream();

    ثابت auto launch = [&](auto has_comb) {
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value>(
            (ثابت عائم *) x->data, (ثابت عائم *) residual->data,
            (ثابت عائم *) post->data, comb ? (ثابت عائم *) comb->data : nullptr, (عائم *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / حجم(عائم), nbx1 / حجم(عائم),
            nbr0 / حجم(عائم), nbr1 / حجم(عائم), nbr2 / حجم(عائم),
            nbp0 / حجم(عائم), nbp1 / حجم(عائم),
            nbc0 / حجم(عائم), nbc1 / حجم(عائم), nbc2 / حجم(عائم),
            nbd0 / حجم(عائم), nbd1 / حجم(عائم), nbd2 / حجم(عائم),
            stream);
    };

    إذا (comb) {
        launch(std::true_type{});
    } وإلا {
        launch(std::false_type{});
    }
}
