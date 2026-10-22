#અન્તર્ભાવ "ggml-impl.h"
#અન્તર્ભાવ "dsv4-hc.hpp"

#અન્તર્ભાવ <cmath>
#અન્તર્ભાવ <type_traits>

સ્થિર constexpr પૂર્ણાંક DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
સ્થિર constexpr પૂર્ણાંક dsv4_hc_pre_block_size = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
template <તાર્કિક gated>
સ્થિર શૂન્ય dsv4_hc_pre_f32_sycl(
        અચલ દશાંશ * x, અચલ દશાંશ * weights, દશાંશ * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        દશાંશ scale,
        queue_ptr stream) {
    અચલ int64_t nr = n_embd * n_tokens;
    અચલ int64_t અંક_blocks = (nr + dsv4_hc_pre_block_size - 1) / dsv4_hc_pre_block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(અંક_blocks * dsv4_hc_pre_block_size),
                          sycl::range<1>(dsv4_hc_pre_block_size)),
        [=](sycl::nd_item<1> item) {
            અચલ int64_t ir = item.get_global_id(0);
            જો (ir >= nr) {
                પાછા;
            }

            અચલ int64_t i0 = ir % n_embd;
            અચલ int64_t it = ir / n_embd;

            દશાંશ sum = 0.0f;
            માટે (int64_t ih = 0; ih < hc; ++ih) {
                અચલ દશાંશ xv = x[i0*sx0 + ih*sx1 + it*sx2];
                દશાંશ wv;
                જો constexpr (gated) {
                    અચલ દશાંશ gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::exp(-gv));
                } નહીંતર {
                    wv = weights[ih*sw0 + it*sw1];
                }
                sum += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * sum;
        });
}

સ્થિર શૂન્ય dsv4_hc_comb_norm_cols(દશાંશ * comb, દશાંશ eps) {
    માટે (પૂર્ણાંક idst = 0; idst < DSV4_HC; ++idst) {
        દશાંશ sum = eps;
        માટે (પૂર્ણાંક isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        અચલ દશાંશ inv_sum = 1.0f / sum;
        માટે (પૂર્ણાંક isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

સ્થિર શૂન્ય dsv4_hc_comb_norm_rows(દશાંશ * comb, દશાંશ eps) {
    માટે (પૂર્ણાંક isrc = 0; isrc < DSV4_HC; ++isrc) {
        દશાંશ sum = eps;
        માટે (પૂર્ણાંક idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        અચલ દશાંશ inv_sum = 1.0f / sum;
        માટે (પૂર્ણાંક idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

સ્થિર શૂન્ય dsv4_hc_comb_f32_sycl(
        અચલ દશાંશ * mixes,
        અચલ દશાંશ * scale,
        અચલ દશાંશ * base,
        દશાંશ * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        દશાંશ eps,
        int32_t n_iter,
        queue_ptr stream) {
    constexpr પૂર્ણાંક comb_offset = 2*DSV4_HC;

    અચલ int64_t block_size = 256;
    અચલ int64_t અંક_blocks = (n_tokens + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(અંક_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item_ct1) {
            અચલ int64_t it = item_ct1.get_global_id(0);

            જો (it >= n_tokens) {
                પાછા;
            }

            અચલ દશાંશ scale_comb = scale[2*ss0];
            દશાંશ comb[DSV4_HC*DSV4_HC];

            માટે (પૂર્ણાંક isrc = 0; isrc < DSV4_HC; ++isrc) {
                દશાંશ max = -INFINITY;
                માટે (પૂર્ણાંક idst = 0; idst < DSV4_HC; ++idst) {
                    અચલ પૂર્ણાંક idx = idst + DSV4_HC*isrc;
                    અચલ દશાંશ v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                દશાંશ sum = 0.0f;
                માટે (પૂર્ણાંક idst = 0; idst < DSV4_HC; ++idst) {
                    અચલ પૂર્ણાંક idx = idst + DSV4_HC*isrc;
                    અચલ દશાંશ v = expf(comb[idx] - max);
                    comb[idx] = v;
                    sum += v;
                }

                અચલ દશાંશ inv_sum = 1.0f / sum;
                માટે (પૂર્ણાંક idst = 0; idst < DSV4_HC; ++idst) {
                    અચલ પૂર્ણાંક idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_sum + eps;
                }
            }

            dsv4_hc_comb_norm_cols(comb, eps);
            માટે (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_rows(comb, eps);
                dsv4_hc_comb_norm_cols(comb, eps);
            }

            માટે (પૂર્ણાંક isrc = 0; isrc < DSV4_HC; ++isrc) {
                માટે (પૂર્ણાંક idst = 0; idst < DSV4_HC; ++idst) {
                    અચલ પૂર્ણાંક idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
સ્થિર constexpr પૂર્ણાંક dsv4_hc_post_block_size = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
template <તાર્કિક has_comb>
સ્થિર શૂન્ય dsv4_hc_post_f32_sycl(
        અચલ દશાંશ * x, અચલ દશાંશ * residual, અચલ દશાંશ * post, અચલ દશાંશ * comb, દશાંશ * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        queue_ptr stream) {
    અચલ int64_t nr = n_embd * hc * n_tokens;
    અચલ int64_t block_size = dsv4_hc_post_block_size;
    અચલ int64_t અંક_blocks = (nr + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(અંક_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item) {
            અચલ int64_t ir = item.get_global_id(0);
            જો (ir >= nr) {
                પાછા;
            }

            અચલ int64_t i0   = ir % n_embd;
            અચલ int64_t idst = (ir / n_embd) % hc;
            અચલ int64_t it   = ir / (n_embd * hc);

            દશાંશ sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
            જો constexpr (has_comb) {
                માટે (int64_t isrc = 0; isrc < hc; ++isrc) {
                    sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } નહીંતર {
                sum += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
        });
}

શૂન્ય ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    અચલ ggml_tensor * x       = dst->src[0];
    અચલ ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    અચલ int64_t n_embd   = x->ne[0];
    અચલ int64_t hc       = x->ne[1];
    અચલ int64_t n_tokens = x->ne[2];

    અચલ દશાંશ scale = ggml_get_op_params_f32(dst, 0);
    અચલ તાર્કિક  gated = ggml_get_op_params_i32(dst, 1) != 0;

    queue_ptr stream = ctx.stream();

    જો (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
        dsv4_hc_pre_f32_sycl<સાચું>(
                (અચલ દશાંશ *) x->data, (અચલ દશાંશ *) weights->data, (દશાંશ *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / કદ(દશાંશ), nbx1 / કદ(દશાંશ), nbx2 / કદ(દશાંશ),
                nbw0 / કદ(દશાંશ), nbw1 / કદ(દશાંશ), nbw2 / કદ(દશાંશ),
                nbd0 / કદ(દશાંશ), nbd1 / કદ(દશાંશ),
                scale, stream);
    } નહીંતર {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        dsv4_hc_pre_f32_sycl<ખોટું>(
                (અચલ દશાંશ *) x->data, (અચલ દશાંશ *) weights->data, (દશાંશ *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / કદ(દશાંશ), nbx1 / કદ(દશાંશ), nbx2 / કદ(દશાંશ),
                nbw0 / કદ(દશાંશ), nbw1 / કદ(દશાંશ), /*sw2=*/ 0,
                nbd0 / કદ(દશાંશ), nbd1 / કદ(દશાંશ),
                scale, stream);
    }
}

શૂન્ય ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    અચલ ggml_tensor * mixes = dst->src[0];
    અચલ ggml_tensor * scale = dst->src[1];
    અચલ ggml_tensor * base  = dst->src[2];

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

    અચલ int64_t n_tokens = mixes->ne[1];
    અચલ દશાંશ eps = ggml_get_op_params_f32(dst, 0);
    અચલ int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    queue_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (અચલ દશાંશ *) mixes->data, (અચલ દશાંશ *) scale->data, (અચલ દશાંશ *) base->data, (દશાંશ *) dst->data,
            n_tokens,
            nbm0 / કદ(દશાંશ), nbm1 / કદ(દશાંશ),
            nbs0 / કદ(દશાંશ),
            nbb0 / કદ(દશાંશ),
            nbd0 / કદ(દશાંશ), nbd1 / કદ(દશાંશ), nbd2 / કદ(દશાંશ),
            eps, n_iter, stream);
}

શૂન્ય ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    અચલ ggml_tensor * x        = dst->src[0];
    અચલ ggml_tensor * residual = dst->src[1];
    અચલ ggml_tensor * post     = dst->src[2];
    અચલ ggml_tensor * comb     = dst->src[3];

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
    જો (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    અચલ int64_t n_embd   = x->ne[0];
    અચલ int64_t n_tokens = x->ne[1];
    અચલ int64_t hc       = residual->ne[1];

    queue_ptr stream = ctx.stream();

    અચલ auto launch = [&](auto has_comb) {
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value>(
            (અચલ દશાંશ *) x->data, (અચલ દશાંશ *) residual->data,
            (અચલ દશાંશ *) post->data, comb ? (અચલ દશાંશ *) comb->data : nullptr, (દશાંશ *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / કદ(દશાંશ), nbx1 / કદ(દશાંશ),
            nbr0 / કદ(દશાંશ), nbr1 / કદ(દશાંશ), nbr2 / કદ(દશાંશ),
            nbp0 / કદ(દશાંશ), nbp1 / કદ(દશાંશ),
            nbc0 / કદ(દશાંશ), nbc1 / કદ(દશાંશ), nbc2 / કદ(દશાંશ),
            nbd0 / કદ(દશાંશ), nbd1 / કદ(દશાંશ), nbd2 / કદ(દશાંશ),
            stream);
    };

    જો (comb) {
        launch(std::true_type{});
    } નહીંતર {
        launch(std::false_type{});
    }
}
