#অন্তর্ভাৱ "ggml-impl.h"
#অন্তর্ভাৱ "dsv4-hc.hpp"

#অন্তর্ভাৱ <cmath>
#অন্তর্ভাৱ <type_traits>

স্থির constexpr পূর্ণাঙ্ক DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
স্থির constexpr পূর্ণাঙ্ক dsv4_hc_pre_block_size = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
template <তার্কিক gated>
স্থির শূন্য dsv4_hc_pre_f32_sycl(
        ধ্রুৱ ভিন্ন * x, ধ্রুৱ ভিন্ন * weights, ভিন্ন * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        ভিন্ন scale,
        queue_ptr stream) {
    ধ্রুৱ int64_t nr = n_embd * n_tokens;
    ধ্রুৱ int64_t অঙ্ক_blocks = (nr + dsv4_hc_pre_block_size - 1) / dsv4_hc_pre_block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks * dsv4_hc_pre_block_size),
                          sycl::range<1>(dsv4_hc_pre_block_size)),
        [=](sycl::nd_item<1> item) {
            ধ্রুৱ int64_t ir = item.get_global_id(0);
            যদি (ir >= nr) {
                ঘূৰাই;
            }

            ধ্রুৱ int64_t i0 = ir % n_embd;
            ধ্রুৱ int64_t it = ir / n_embd;

            ভিন্ন sum = 0.0f;
            বাবে (int64_t ih = 0; ih < hc; ++ih) {
                ধ্রুৱ ভিন্ন xv = x[i0*sx0 + ih*sx1 + it*sx2];
                ভিন্ন wv;
                যদি constexpr (gated) {
                    ধ্রুৱ ভিন্ন gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::exp(-gv));
                } অন্যথা {
                    wv = weights[ih*sw0 + it*sw1];
                }
                sum += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * sum;
        });
}

স্থির শূন্য dsv4_hc_comb_norm_cols(ভিন্ন * comb, ভিন্ন eps) {
    বাবে (পূর্ণাঙ্ক idst = 0; idst < DSV4_HC; ++idst) {
        ভিন্ন sum = eps;
        বাবে (পূর্ণাঙ্ক isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        ধ্রুৱ ভিন্ন inv_sum = 1.0f / sum;
        বাবে (পূর্ণাঙ্ক isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

স্থির শূন্য dsv4_hc_comb_norm_rows(ভিন্ন * comb, ভিন্ন eps) {
    বাবে (পূর্ণাঙ্ক isrc = 0; isrc < DSV4_HC; ++isrc) {
        ভিন্ন sum = eps;
        বাবে (পূর্ণাঙ্ক idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        ধ্রুৱ ভিন্ন inv_sum = 1.0f / sum;
        বাবে (পূর্ণাঙ্ক idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

স্থির শূন্য dsv4_hc_comb_f32_sycl(
        ধ্রুৱ ভিন্ন * mixes,
        ধ্রুৱ ভিন্ন * scale,
        ধ্রুৱ ভিন্ন * base,
        ভিন্ন * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        ভিন্ন eps,
        int32_t n_iter,
        queue_ptr stream) {
    constexpr পূর্ণাঙ্ক comb_offset = 2*DSV4_HC;

    ধ্রুৱ int64_t block_size = 256;
    ধ্রুৱ int64_t অঙ্ক_blocks = (n_tokens + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item_ct1) {
            ধ্রুৱ int64_t it = item_ct1.get_global_id(0);

            যদি (it >= n_tokens) {
                ঘূৰাই;
            }

            ধ্রুৱ ভিন্ন scale_comb = scale[2*ss0];
            ভিন্ন comb[DSV4_HC*DSV4_HC];

            বাবে (পূর্ণাঙ্ক isrc = 0; isrc < DSV4_HC; ++isrc) {
                ভিন্ন max = -INFINITY;
                বাবে (পূর্ণাঙ্ক idst = 0; idst < DSV4_HC; ++idst) {
                    ধ্রুৱ পূর্ণাঙ্ক idx = idst + DSV4_HC*isrc;
                    ধ্রুৱ ভিন্ন v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                ভিন্ন sum = 0.0f;
                বাবে (পূর্ণাঙ্ক idst = 0; idst < DSV4_HC; ++idst) {
                    ধ্রুৱ পূর্ণাঙ্ক idx = idst + DSV4_HC*isrc;
                    ধ্রুৱ ভিন্ন v = expf(comb[idx] - max);
                    comb[idx] = v;
                    sum += v;
                }

                ধ্রুৱ ভিন্ন inv_sum = 1.0f / sum;
                বাবে (পূর্ণাঙ্ক idst = 0; idst < DSV4_HC; ++idst) {
                    ধ্রুৱ পূর্ণাঙ্ক idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_sum + eps;
                }
            }

            dsv4_hc_comb_norm_cols(comb, eps);
            বাবে (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_rows(comb, eps);
                dsv4_hc_comb_norm_cols(comb, eps);
            }

            বাবে (পূর্ণাঙ্ক isrc = 0; isrc < DSV4_HC; ++isrc) {
                বাবে (পূর্ণাঙ্ক idst = 0; idst < DSV4_HC; ++idst) {
                    ধ্রুৱ পূর্ণাঙ্ক idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
স্থির constexpr পূর্ণাঙ্ক dsv4_hc_post_block_size = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
template <তার্কিক has_comb>
স্থির শূন্য dsv4_hc_post_f32_sycl(
        ধ্রুৱ ভিন্ন * x, ধ্রুৱ ভিন্ন * residual, ধ্রুৱ ভিন্ন * post, ধ্রুৱ ভিন্ন * comb, ভিন্ন * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        queue_ptr stream) {
    ধ্রুৱ int64_t nr = n_embd * hc * n_tokens;
    ধ্রুৱ int64_t block_size = dsv4_hc_post_block_size;
    ধ্রুৱ int64_t অঙ্ক_blocks = (nr + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(অঙ্ক_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item) {
            ধ্রুৱ int64_t ir = item.get_global_id(0);
            যদি (ir >= nr) {
                ঘূৰাই;
            }

            ধ্রুৱ int64_t i0   = ir % n_embd;
            ধ্রুৱ int64_t idst = (ir / n_embd) % hc;
            ধ্রুৱ int64_t it   = ir / (n_embd * hc);

            ভিন্ন sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
            যদি constexpr (has_comb) {
                বাবে (int64_t isrc = 0; isrc < hc; ++isrc) {
                    sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } অন্যথা {
                sum += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
        });
}

শূন্য ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ধ্রুৱ ggml_tensor * x       = dst->src[0];
    ধ্রুৱ ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    ধ্রুৱ int64_t n_embd   = x->ne[0];
    ধ্রুৱ int64_t hc       = x->ne[1];
    ধ্রুৱ int64_t n_tokens = x->ne[2];

    ধ্রুৱ ভিন্ন scale = ggml_get_op_params_f32(dst, 0);
    ধ্রুৱ তার্কিক  gated = ggml_get_op_params_i32(dst, 1) != 0;

    queue_ptr stream = ctx.stream();

    যদি (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
        dsv4_hc_pre_f32_sycl<সঁচা>(
                (ধ্রুৱ ভিন্ন *) x->data, (ধ্রুৱ ভিন্ন *) weights->data, (ভিন্ন *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / পরিমাণ(ভিন্ন), nbx1 / পরিমাণ(ভিন্ন), nbx2 / পরিমাণ(ভিন্ন),
                nbw0 / পরিমাণ(ভিন্ন), nbw1 / পরিমাণ(ভিন্ন), nbw2 / পরিমাণ(ভিন্ন),
                nbd0 / পরিমাণ(ভিন্ন), nbd1 / পরিমাণ(ভিন্ন),
                scale, stream);
    } অন্যথা {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        dsv4_hc_pre_f32_sycl<মিছা>(
                (ধ্রুৱ ভিন্ন *) x->data, (ধ্রুৱ ভিন্ন *) weights->data, (ভিন্ন *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / পরিমাণ(ভিন্ন), nbx1 / পরিমাণ(ভিন্ন), nbx2 / পরিমাণ(ভিন্ন),
                nbw0 / পরিমাণ(ভিন্ন), nbw1 / পরিমাণ(ভিন্ন), /*sw2=*/ 0,
                nbd0 / পরিমাণ(ভিন্ন), nbd1 / পরিমাণ(ভিন্ন),
                scale, stream);
    }
}

শূন্য ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ধ্রুৱ ggml_tensor * mixes = dst->src[0];
    ধ্রুৱ ggml_tensor * scale = dst->src[1];
    ধ্রুৱ ggml_tensor * base  = dst->src[2];

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

    ধ্রুৱ int64_t n_tokens = mixes->ne[1];
    ধ্রুৱ ভিন্ন eps = ggml_get_op_params_f32(dst, 0);
    ধ্রুৱ int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    queue_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (ধ্রুৱ ভিন্ন *) mixes->data, (ধ্রুৱ ভিন্ন *) scale->data, (ধ্রুৱ ভিন্ন *) base->data, (ভিন্ন *) dst->data,
            n_tokens,
            nbm0 / পরিমাণ(ভিন্ন), nbm1 / পরিমাণ(ভিন্ন),
            nbs0 / পরিমাণ(ভিন্ন),
            nbb0 / পরিমাণ(ভিন্ন),
            nbd0 / পরিমাণ(ভিন্ন), nbd1 / পরিমাণ(ভিন্ন), nbd2 / পরিমাণ(ভিন্ন),
            eps, n_iter, stream);
}

শূন্য ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    ধ্রুৱ ggml_tensor * x        = dst->src[0];
    ধ্রুৱ ggml_tensor * residual = dst->src[1];
    ধ্রুৱ ggml_tensor * post     = dst->src[2];
    ধ্রুৱ ggml_tensor * comb     = dst->src[3];

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

    ধ্রুৱ int64_t n_embd   = x->ne[0];
    ধ্রুৱ int64_t n_tokens = x->ne[1];
    ধ্রুৱ int64_t hc       = residual->ne[1];

    queue_ptr stream = ctx.stream();

    ধ্রুৱ auto launch = [&](auto has_comb) {
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value>(
            (ধ্রুৱ ভিন্ন *) x->data, (ধ্রুৱ ভিন্ন *) residual->data,
            (ধ্রুৱ ভিন্ন *) post->data, comb ? (ধ্রুৱ ভিন্ন *) comb->data : nullptr, (ভিন্ন *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / পরিমাণ(ভিন্ন), nbx1 / পরিমাণ(ভিন্ন),
            nbr0 / পরিমাণ(ভিন্ন), nbr1 / পরিমাণ(ভিন্ন), nbr2 / পরিমাণ(ভিন্ন),
            nbp0 / পরিমাণ(ভিন্ন), nbp1 / পরিমাণ(ভিন্ন),
            nbc0 / পরিমাণ(ভিন্ন), nbc1 / পরিমাণ(ভিন্ন), nbc2 / পরিমাণ(ভিন্ন),
            nbd0 / পরিমাণ(ভিন্ন), nbd1 / পরিমাণ(ভিন্ন), nbd2 / পরিমাণ(ভিন্ন),
            stream);
    };

    যদি (comb) {
        launch(std::true_type{});
    } অন্যথা {
        launch(std::false_type{});
    }
}
