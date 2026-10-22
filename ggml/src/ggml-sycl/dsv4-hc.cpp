#অন্তর্ভাব "ggml-impl.h"
#অন্তর্ভাব "dsv4-hc.hpp"

#অন্তর্ভাব <cmath>
#অন্তর্ভাব <type_traits>

স্থির constexpr পূর্ণসংখ্যা DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
স্থির constexpr পূর্ণসংখ্যা dsv4_hc_pre_block_আকার = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
template <যৌক্তিক gated>
স্থির শূন্য dsv4_hc_pre_f32_sycl(
        ধ্রুবক দশমিক * x, ধ্রুবক দশমিক * weights, দশমিক * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        দশমিক scale,
        queue_ptr stream) {
    ধ্রুবক int64_t nr = n_embd * n_tokens;
    ধ্রুবক int64_t অঙ্ক_blocks = (nr + dsv4_hc_pre_block_আকার - 1) / dsv4_hc_pre_block_আকার;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks * dsv4_hc_pre_block_আকার),
                          sycl::range<1>(dsv4_hc_pre_block_আকার)),
        [=](sycl::nd_item<1> item) {
            ধ্রুবক int64_t ir = item.get_global_id(0);
            যদি (ir >= nr) {
                ফেরত;
            }

            ধ্রুবক int64_t i0 = ir % n_embd;
            ধ্রুবক int64_t it = ir / n_embd;

            দশমিক যোগ_2 = 0.0f;
            জন্য (int64_t ih = 0; ih < hc; ++ih) {
                ধ্রুবক দশমিক xv = x[i0*sx0 + ih*sx1 + it*sx2];
                দশমিক wv;
                যদি constexpr (gated) {
                    ধ্রুবক দশমিক gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::exp(-gv));
                } অন্যথা {
                    wv = weights[ih*sw0 + it*sw1];
                }
                যোগ_2 += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * যোগ_2;
        });
}

স্থির শূন্য dsv4_hc_comb_norm_cols(দশমিক * comb, দশমিক eps) {
    জন্য (পূর্ণসংখ্যা idst = 0; idst < DSV4_HC; ++idst) {
        দশমিক যোগ_2 = eps;
        জন্য (পূর্ণসংখ্যা isrc = 0; isrc < DSV4_HC; ++isrc) {
            যোগ_2 += comb[idst + DSV4_HC*isrc];
        }

        ধ্রুবক দশমিক inv_যোগ = 1.0f / যোগ_2;
        জন্য (পূর্ণসংখ্যা isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_যোগ;
        }
    }
}

স্থির শূন্য dsv4_hc_comb_norm_rows(দশমিক * comb, দশমিক eps) {
    জন্য (পূর্ণসংখ্যা isrc = 0; isrc < DSV4_HC; ++isrc) {
        দশমিক যোগ_2 = eps;
        জন্য (পূর্ণসংখ্যা idst = 0; idst < DSV4_HC; ++idst) {
            যোগ_2 += comb[idst + DSV4_HC*isrc];
        }

        ধ্রুবক দশমিক inv_যোগ = 1.0f / যোগ_2;
        জন্য (পূর্ণসংখ্যা idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_যোগ;
        }
    }
}

স্থির শূন্য dsv4_hc_comb_f32_sycl(
        ধ্রুবক দশমিক * mixes,
        ধ্রুবক দশমিক * scale,
        ধ্রুবক দশমিক * base,
        দশমিক * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        দশমিক eps,
        int32_t n_iter,
        queue_ptr stream) {
    constexpr পূর্ণসংখ্যা comb_offset = 2*DSV4_HC;

    ধ্রুবক int64_t block_আকার_2 = 256;
    ধ্রুবক int64_t অঙ্ক_blocks = (n_tokens + block_আকার_2 - 1) / block_আকার_2;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks * block_আকার_2), sycl::range<1>(block_আকার_2)),
        [=](sycl::nd_item<1> item_ct1) {
            ধ্রুবক int64_t it = item_ct1.get_global_id(0);

            যদি (it >= n_tokens) {
                ফেরত;
            }

            ধ্রুবক দশমিক scale_comb = scale[2*ss0];
            দশমিক comb[DSV4_HC*DSV4_HC];

            জন্য (পূর্ণসংখ্যা isrc = 0; isrc < DSV4_HC; ++isrc) {
                দশমিক max = -INFINITY;
                জন্য (পূর্ণসংখ্যা idst = 0; idst < DSV4_HC; ++idst) {
                    ধ্রুবক পূর্ণসংখ্যা idx = idst + DSV4_HC*isrc;
                    ধ্রুবক দশমিক v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                দশমিক যোগ_2 = 0.0f;
                জন্য (পূর্ণসংখ্যা idst = 0; idst < DSV4_HC; ++idst) {
                    ধ্রুবক পূর্ণসংখ্যা idx = idst + DSV4_HC*isrc;
                    ধ্রুবক দশমিক v = expf(comb[idx] - max);
                    comb[idx] = v;
                    যোগ_2 += v;
                }

                ধ্রুবক দশমিক inv_যোগ = 1.0f / যোগ_2;
                জন্য (পূর্ণসংখ্যা idst = 0; idst < DSV4_HC; ++idst) {
                    ধ্রুবক পূর্ণসংখ্যা idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_যোগ + eps;
                }
            }

            dsv4_hc_comb_norm_cols(comb, eps);
            জন্য (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_rows(comb, eps);
                dsv4_hc_comb_norm_cols(comb, eps);
            }

            জন্য (পূর্ণসংখ্যা isrc = 0; isrc < DSV4_HC; ++isrc) {
                জন্য (পূর্ণসংখ্যা idst = 0; idst < DSV4_HC; ++idst) {
                    ধ্রুবক পূর্ণসংখ্যা idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
স্থির constexpr পূর্ণসংখ্যা dsv4_hc_post_block_আকার = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
template <যৌক্তিক has_comb>
স্থির শূন্য dsv4_hc_post_f32_sycl(
        ধ্রুবক দশমিক * x, ধ্রুবক দশমিক * residual, ধ্রুবক দশমিক * post, ধ্রুবক দশমিক * comb, দশমিক * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        queue_ptr stream) {
    ধ্রুবক int64_t nr = n_embd * hc * n_tokens;
    ধ্রুবক int64_t block_আকার_2 = dsv4_hc_post_block_আকার;
    ধ্রুবক int64_t অঙ্ক_blocks = (nr + block_আকার_2 - 1) / block_আকার_2;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks * block_আকার_2), sycl::range<1>(block_আকার_2)),
        [=](sycl::nd_item<1> item) {
            ধ্রুবক int64_t ir = item.get_global_id(0);
            যদি (ir >= nr) {
                ফেরত;
            }

            ধ্রুবক int64_t i0   = ir % n_embd;
            ধ্রুবক int64_t idst = (ir / n_embd) % hc;
            ধ্রুবক int64_t it   = ir / (n_embd * hc);

            দশমিক যোগ_2 = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
            যদি constexpr (has_comb) {
                জন্য (int64_t isrc = 0; isrc < hc; ++isrc) {
                    যোগ_2 += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } অন্যথা {
                যোগ_2 += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = যোগ_2;
        });
}

শূন্য ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ধ্রুবক ggml_tensor * x       = dst->src[0];
    ধ্রুবক ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    ধ্রুবক int64_t n_embd   = x->ne[0];
    ধ্রুবক int64_t hc       = x->ne[1];
    ধ্রুবক int64_t n_tokens = x->ne[2];

    ধ্রুবক দশমিক scale = ggml_get_op_params_f32(dst, 0);
    ধ্রুবক যৌক্তিক  gated = ggml_get_op_params_i32(dst, 1) != 0;

    queue_ptr stream = ctx.stream();

    যদি (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
        dsv4_hc_pre_f32_sycl<সত্য>(
                (ধ্রুবক দশমিক *) x->তথ্য, (ধ্রুবক দশমিক *) weights->তথ্য, (দশমিক *) dst->তথ্য,
                n_embd, hc, n_tokens,
                nbx0 / আকার(দশমিক), nbx1 / আকার(দশমিক), nbx2 / আকার(দশমিক),
                nbw0 / আকার(দশমিক), nbw1 / আকার(দশমিক), nbw2 / আকার(দশমিক),
                nbd0 / আকার(দশমিক), nbd1 / আকার(দশমিক),
                scale, stream);
    } অন্যথা {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        dsv4_hc_pre_f32_sycl<মিথ্যা>(
                (ধ্রুবক দশমিক *) x->তথ্য, (ধ্রুবক দশমিক *) weights->তথ্য, (দশমিক *) dst->তথ্য,
                n_embd, hc, n_tokens,
                nbx0 / আকার(দশমিক), nbx1 / আকার(দশমিক), nbx2 / আকার(দশমিক),
                nbw0 / আকার(দশমিক), nbw1 / আকার(দশমিক), /*sw2=*/ 0,
                nbd0 / আকার(দশমিক), nbd1 / আকার(দশমিক),
                scale, stream);
    }
}

শূন্য ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ধ্রুবক ggml_tensor * mixes = dst->src[0];
    ধ্রুবক ggml_tensor * scale = dst->src[1];
    ধ্রুবক ggml_tensor * base  = dst->src[2];

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

    ধ্রুবক int64_t n_tokens = mixes->ne[1];
    ধ্রুবক দশমিক eps = ggml_get_op_params_f32(dst, 0);
    ধ্রুবক int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    queue_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (ধ্রুবক দশমিক *) mixes->তথ্য, (ধ্রুবক দশমিক *) scale->তথ্য, (ধ্রুবক দশমিক *) base->তথ্য, (দশমিক *) dst->তথ্য,
            n_tokens,
            nbm0 / আকার(দশমিক), nbm1 / আকার(দশমিক),
            nbs0 / আকার(দশমিক),
            nbb0 / আকার(দশমিক),
            nbd0 / আকার(দশমিক), nbd1 / আকার(দশমিক), nbd2 / আকার(দশমিক),
            eps, n_iter, stream);
}

শূন্য ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    ধ্রুবক ggml_tensor * x        = dst->src[0];
    ধ্রুবক ggml_tensor * residual = dst->src[1];
    ধ্রুবক ggml_tensor * post     = dst->src[2];
    ধ্রুবক ggml_tensor * comb     = dst->src[3];

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
    যদি (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    ধ্রুবক int64_t n_embd   = x->ne[0];
    ধ্রুবক int64_t n_tokens = x->ne[1];
    ধ্রুবক int64_t hc       = residual->ne[1];

    queue_ptr stream = ctx.stream();

    ধ্রুবক auto launch = [&](auto has_comb) {
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value>(
            (ধ্রুবক দশমিক *) x->তথ্য, (ধ্রুবক দশমিক *) residual->তথ্য,
            (ধ্রুবক দশমিক *) post->তথ্য, comb ? (ধ্রুবক দশমিক *) comb->তথ্য : nullptr, (দশমিক *) dst->তথ্য,
            n_embd, hc, n_tokens,
            nbx0 / আকার(দশমিক), nbx1 / আকার(দশমিক),
            nbr0 / আকার(দশমিক), nbr1 / আকার(দশমিক), nbr2 / আকার(দশমিক),
            nbp0 / আকার(দশমিক), nbp1 / আকার(দশমিক),
            nbc0 / আকার(দশমিক), nbc1 / আকার(দশমিক), nbc2 / আকার(দশমিক),
            nbd0 / আকার(দশমিক), nbd1 / আকার(দশমিক), nbd2 / আকার(দশমিক),
            stream);
    };

    যদি (comb) {
        launch(std::true_type{});
    } অন্যথা {
        launch(std::false_type{});
    }
}
