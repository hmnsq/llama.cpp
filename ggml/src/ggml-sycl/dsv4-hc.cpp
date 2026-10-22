#شامل "ggml-impl.h"
#شامل "dsv4-hc.hpp"

#شامل <cmath>
#شامل <type_traits>

ساکن constexpr صحیح DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
ساکن constexpr صحیح dsv4_hc_pre_بلاک_سائز = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
سانچہ <منطقی gated>
ساکن خالی dsv4_hc_pre_f32_sycl(
        ثابت اعشاری * x, ثابت اعشاری * weights, اعشاری * dst,
        int64_t n_embd, int64_t hc, int64_t n_علامات,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        اعشاری scale,
        قطار_ptr stream) {
    ثابت int64_t nr = n_embd * n_علامات;
    ثابت int64_t عدد_بلاکیں = (nr + dsv4_hc_pre_بلاک_سائز - 1) / dsv4_hc_pre_بلاک_سائز;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(عدد_بلاکیں * dsv4_hc_pre_بلاک_سائز),
                          sycl::range<1>(dsv4_hc_pre_بلاک_سائز)),
        [=](sycl::nd_شے<1> شے) {
            ثابت int64_t ir = شے.get_global_id(0);
            اگر (ir >= nr) {
                واپس;
            }

            ثابت int64_t i0 = ir % n_embd;
            ثابت int64_t it = ir / n_embd;

            اعشاری مجموع = 0.0f;
            برائے (int64_t ih = 0; ih < hc; ++ih) {
                ثابت اعشاری xv = x[i0*sx0 + ih*sx1 + it*sx2];
                اعشاری wv;
                اگر constexpr (gated) {
                    ثابت اعشاری gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::قوت_نما(-gv));
                } ورنہ {
                    wv = weights[ih*sw0 + it*sw1];
                }
                مجموع += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * مجموع;
        });
}

ساکن خالی dsv4_hc_comb_norm_ستون_تعداد(اعشاری * comb, اعشاری eps) {
    برائے (صحیح idst = 0; idst < DSV4_HC; ++idst) {
        اعشاری مجموع = eps;
        برائے (صحیح isrc = 0; isrc < DSV4_HC; ++isrc) {
            مجموع += comb[idst + DSV4_HC*isrc];
        }

        ثابت اعشاری inv_مجموع = 1.0f / مجموع;
        برائے (صحیح isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_مجموع;
        }
    }
}

ساکن خالی dsv4_hc_comb_norm_صفیں(اعشاری * comb, اعشاری eps) {
    برائے (صحیح isrc = 0; isrc < DSV4_HC; ++isrc) {
        اعشاری مجموع = eps;
        برائے (صحیح idst = 0; idst < DSV4_HC; ++idst) {
            مجموع += comb[idst + DSV4_HC*isrc];
        }

        ثابت اعشاری inv_مجموع = 1.0f / مجموع;
        برائے (صحیح idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_مجموع;
        }
    }
}

ساکن خالی dsv4_hc_comb_f32_sycl(
        ثابت اعشاری * mixes,
        ثابت اعشاری * scale,
        ثابت اعشاری * بنیاد,
        اعشاری * dst,
        int64_t n_علامات,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        اعشاری eps,
        int32_t n_iter,
        قطار_ptr stream) {
    constexpr صحیح comb_انحراف = 2*DSV4_HC;

    ثابت int64_t بلاک_سائز_3 = 256;
    ثابت int64_t عدد_بلاکیں = (n_علامات + بلاک_سائز_3 - 1) / بلاک_سائز_3;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(عدد_بلاکیں * بلاک_سائز_3), sycl::range<1>(بلاک_سائز_3)),
        [=](sycl::nd_شے<1> شے_ct1) {
            ثابت int64_t it = شے_ct1.get_global_id(0);

            اگر (it >= n_علامات) {
                واپس;
            }

            ثابت اعشاری scale_comb = scale[2*ss0];
            اعشاری comb[DSV4_HC*DSV4_HC];

            برائے (صحیح isrc = 0; isrc < DSV4_HC; ++isrc) {
                اعشاری max = -INFINITY;
                برائے (صحیح idst = 0; idst < DSV4_HC; ++idst) {
                    ثابت صحیح idx = idst + DSV4_HC*isrc;
                    ثابت اعشاری v = mixes[(comb_انحراف + idx)*sm0 + it*sm1] * scale_comb + بنیاد[(comb_انحراف + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                اعشاری مجموع = 0.0f;
                برائے (صحیح idst = 0; idst < DSV4_HC; ++idst) {
                    ثابت صحیح idx = idst + DSV4_HC*isrc;
                    ثابت اعشاری v = expf(comb[idx] - max);
                    comb[idx] = v;
                    مجموع += v;
                }

                ثابت اعشاری inv_مجموع = 1.0f / مجموع;
                برائے (صحیح idst = 0; idst < DSV4_HC; ++idst) {
                    ثابت صحیح idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_مجموع + eps;
                }
            }

            dsv4_hc_comb_norm_ستون_تعداد(comb, eps);
            برائے (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_صفیں(comb, eps);
                dsv4_hc_comb_norm_ستون_تعداد(comb, eps);
            }

            برائے (صحیح isrc = 0; isrc < DSV4_HC; ++isrc) {
                برائے (صحیح idst = 0; idst < DSV4_HC; ++idst) {
                    ثابت صحیح idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
ساکن constexpr صحیح dsv4_hc_post_بلاک_سائز = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
سانچہ <منطقی has_comb>
ساکن خالی dsv4_hc_post_f32_sycl(
        ثابت اعشاری * x, ثابت اعشاری * residual, ثابت اعشاری * post, ثابت اعشاری * comb, اعشاری * dst,
        int64_t n_embd, int64_t hc, int64_t n_علامات,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        قطار_ptr stream) {
    ثابت int64_t nr = n_embd * hc * n_علامات;
    ثابت int64_t بلاک_سائز_3 = dsv4_hc_post_بلاک_سائز;
    ثابت int64_t عدد_بلاکیں = (nr + بلاک_سائز_3 - 1) / بلاک_سائز_3;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(عدد_بلاکیں * بلاک_سائز_3), sycl::range<1>(بلاک_سائز_3)),
        [=](sycl::nd_شے<1> شے) {
            ثابت int64_t ir = شے.get_global_id(0);
            اگر (ir >= nr) {
                واپس;
            }

            ثابت int64_t i0   = ir % n_embd;
            ثابت int64_t idst = (ir / n_embd) % hc;
            ثابت int64_t it   = ir / (n_embd * hc);

            اعشاری مجموع = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
            اگر constexpr (has_comb) {
                برائے (int64_t isrc = 0; isrc < hc; ++isrc) {
                    مجموع += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } ورنہ {
                مجموع += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = مجموع;
        });
}

خالی ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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
    ثابت int64_t n_علامات = x->ne[2];

    ثابت اعشاری scale = ggml_get_op_params_f32(dst, 0);
    ثابت منطقی  gated = ggml_get_op_params_i32(dst, 1) != 0;

    قطار_ptr stream = ctx.stream();

    اگر (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_علامات);
        dsv4_hc_pre_f32_sycl<سچ>(
                (ثابت اعشاری *) x->ڈیٹا, (ثابت اعشاری *) weights->ڈیٹا, (اعشاری *) dst->ڈیٹا,
                n_embd, hc, n_علامات,
                nbx0 / حجم(اعشاری), nbx1 / حجم(اعشاری), nbx2 / حجم(اعشاری),
                nbw0 / حجم(اعشاری), nbw1 / حجم(اعشاری), nbw2 / حجم(اعشاری),
                nbd0 / حجم(اعشاری), nbd1 / حجم(اعشاری),
                scale, stream);
    } ورنہ {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_علامات);
        dsv4_hc_pre_f32_sycl<جھوٹ>(
                (ثابت اعشاری *) x->ڈیٹا, (ثابت اعشاری *) weights->ڈیٹا, (اعشاری *) dst->ڈیٹا,
                n_embd, hc, n_علامات,
                nbx0 / حجم(اعشاری), nbx1 / حجم(اعشاری), nbx2 / حجم(اعشاری),
                nbw0 / حجم(اعشاری), nbw1 / حجم(اعشاری), /*sw2=*/ 0,
                nbd0 / حجم(اعشاری), nbd1 / حجم(اعشاری),
                scale, stream);
    }
}

خالی ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ثابت ggml_tensor * mixes = dst->src[0];
    ثابت ggml_tensor * scale = dst->src[1];
    ثابت ggml_tensor * بنیاد  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(بنیاد->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(بنیاد->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, بنیاد,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    ثابت int64_t n_علامات = mixes->ne[1];
    ثابت اعشاری eps = ggml_get_op_params_f32(dst, 0);
    ثابت int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    قطار_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (ثابت اعشاری *) mixes->ڈیٹا, (ثابت اعشاری *) scale->ڈیٹا, (ثابت اعشاری *) بنیاد->ڈیٹا, (اعشاری *) dst->ڈیٹا,
            n_علامات,
            nbm0 / حجم(اعشاری), nbm1 / حجم(اعشاری),
            nbs0 / حجم(اعشاری),
            nbb0 / حجم(اعشاری),
            nbd0 / حجم(اعشاری), nbd1 / حجم(اعشاری), nbd2 / حجم(اعشاری),
            eps, n_iter, stream);
}

خالی ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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
    اگر (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    ثابت int64_t n_embd   = x->ne[0];
    ثابت int64_t n_علامات = x->ne[1];
    ثابت int64_t hc       = residual->ne[1];

    قطار_ptr stream = ctx.stream();

    ثابت خودکار launch = [&](خودکار has_comb) {
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value>(
            (ثابت اعشاری *) x->ڈیٹا, (ثابت اعشاری *) residual->ڈیٹا,
            (ثابت اعشاری *) post->ڈیٹا, comb ? (ثابت اعشاری *) comb->ڈیٹا : خالی_اشاریہ, (اعشاری *) dst->ڈیٹا,
            n_embd, hc, n_علامات,
            nbx0 / حجم(اعشاری), nbx1 / حجم(اعشاری),
            nbr0 / حجم(اعشاری), nbr1 / حجم(اعشاری), nbr2 / حجم(اعشاری),
            nbp0 / حجم(اعشاری), nbp1 / حجم(اعشاری),
            nbc0 / حجم(اعشاری), nbc1 / حجم(اعشاری), nbc2 / حجم(اعشاری),
            nbd0 / حجم(اعشاری), nbd1 / حجم(اعشاری), nbd2 / حجم(اعشاری),
            stream);
    };

    اگر (comb) {
        launch(std::true_type{});
    } ورنہ {
        launch(std::false_type{});
    }
}
