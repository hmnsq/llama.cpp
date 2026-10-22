#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "dsv4-hc.hpp"

#అన్తర్భావ <cmath>
#అన్తర్భావ <type_traits>

స్థిర constexpr పూర్ణాంకం DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
స్థిర constexpr పూర్ణాంకం dsv4_hc_pre_ఖండం_కొలత = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
మూస <తార్కిక gated>
స్థిర శూన్య dsv4_hc_pre_f32_sycl(
        స్థిరరాశి దశాంశం * x, స్థిరరాశి దశాంశం * weights, దశాంశం * dst,
        int64_t n_embd, int64_t hc, int64_t n_టోకెన్లు,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        దశాంశం scale,
        వరుస_ptr stream) {
    స్థిరరాశి int64_t nr = n_embd * n_టోకెన్లు;
    స్థిరరాశి int64_t అంకె_ఖండాలు = (nr + dsv4_hc_pre_ఖండం_కొలత - 1) / dsv4_hc_pre_ఖండం_కొలత;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(అంకె_ఖండాలు * dsv4_hc_pre_ఖండం_కొలత),
                          sycl::range<1>(dsv4_hc_pre_ఖండం_కొలత)),
        [=](sycl::nd_అంశం<1> అంశం) {
            స్థిరరాశి int64_t ir = అంశం.get_global_id(0);
            ఒకవేళ (ir >= nr) {
                తిరిగి;
            }

            స్థిరరాశి int64_t i0 = ir % n_embd;
            స్థిరరాశి int64_t it = ir / n_embd;

            దశాంశం కూడిక = 0.0f;
            కొరకు (int64_t ih = 0; ih < hc; ++ih) {
                స్థిరరాశి దశాంశం xv = x[i0*sx0 + ih*sx1 + it*sx2];
                దశాంశం wv;
                ఒకవేళ constexpr (gated) {
                    స్థిరరాశి దశాంశం gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::ఘాతాంకం(-gv));
                } లేకపోతే {
                    wv = weights[ih*sw0 + it*sw1];
                }
                కూడిక += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * కూడిక;
        });
}

స్థిర శూన్య dsv4_hc_comb_norm_నిలువుల_సంఖ్య(దశాంశం * comb, దశాంశం eps) {
    కొరకు (పూర్ణాంకం idst = 0; idst < DSV4_HC; ++idst) {
        దశాంశం కూడిక = eps;
        కొరకు (పూర్ణాంకం isrc = 0; isrc < DSV4_HC; ++isrc) {
            కూడిక += comb[idst + DSV4_HC*isrc];
        }

        స్థిరరాశి దశాంశం inv_కూడిక = 1.0f / కూడిక;
        కొరకు (పూర్ణాంకం isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_కూడిక;
        }
    }
}

స్థిర శూన్య dsv4_hc_comb_norm_అడ్డువరుసలు(దశాంశం * comb, దశాంశం eps) {
    కొరకు (పూర్ణాంకం isrc = 0; isrc < DSV4_HC; ++isrc) {
        దశాంశం కూడిక = eps;
        కొరకు (పూర్ణాంకం idst = 0; idst < DSV4_HC; ++idst) {
            కూడిక += comb[idst + DSV4_HC*isrc];
        }

        స్థిరరాశి దశాంశం inv_కూడిక = 1.0f / కూడిక;
        కొరకు (పూర్ణాంకం idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_కూడిక;
        }
    }
}

స్థిర శూన్య dsv4_hc_comb_f32_sycl(
        స్థిరరాశి దశాంశం * mixes,
        స్థిరరాశి దశాంశం * scale,
        స్థిరరాశి దశాంశం * పునాది,
        దశాంశం * dst,
        int64_t n_టోకెన్లు,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        దశాంశం eps,
        int32_t n_iter,
        వరుస_ptr stream) {
    constexpr పూర్ణాంకం comb_స్థానభ్రంశం = 2*DSV4_HC;

    స్థిరరాశి int64_t ఖండం_కొలత_3 = 256;
    స్థిరరాశి int64_t అంకె_ఖండాలు = (n_టోకెన్లు + ఖండం_కొలత_3 - 1) / ఖండం_కొలత_3;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(అంకె_ఖండాలు * ఖండం_కొలత_3), sycl::range<1>(ఖండం_కొలత_3)),
        [=](sycl::nd_అంశం<1> అంశం_ct1) {
            స్థిరరాశి int64_t it = అంశం_ct1.get_global_id(0);

            ఒకవేళ (it >= n_టోకెన్లు) {
                తిరిగి;
            }

            స్థిరరాశి దశాంశం scale_comb = scale[2*ss0];
            దశాంశం comb[DSV4_HC*DSV4_HC];

            కొరకు (పూర్ణాంకం isrc = 0; isrc < DSV4_HC; ++isrc) {
                దశాంశం max = -INFINITY;
                కొరకు (పూర్ణాంకం idst = 0; idst < DSV4_HC; ++idst) {
                    స్థిరరాశి పూర్ణాంకం idx = idst + DSV4_HC*isrc;
                    స్థిరరాశి దశాంశం v = mixes[(comb_స్థానభ్రంశం + idx)*sm0 + it*sm1] * scale_comb + పునాది[(comb_స్థానభ్రంశం + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                దశాంశం కూడిక = 0.0f;
                కొరకు (పూర్ణాంకం idst = 0; idst < DSV4_HC; ++idst) {
                    స్థిరరాశి పూర్ణాంకం idx = idst + DSV4_HC*isrc;
                    స్థిరరాశి దశాంశం v = expf(comb[idx] - max);
                    comb[idx] = v;
                    కూడిక += v;
                }

                స్థిరరాశి దశాంశం inv_కూడిక = 1.0f / కూడిక;
                కొరకు (పూర్ణాంకం idst = 0; idst < DSV4_HC; ++idst) {
                    స్థిరరాశి పూర్ణాంకం idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_కూడిక + eps;
                }
            }

            dsv4_hc_comb_norm_నిలువుల_సంఖ్య(comb, eps);
            కొరకు (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_అడ్డువరుసలు(comb, eps);
                dsv4_hc_comb_norm_నిలువుల_సంఖ్య(comb, eps);
            }

            కొరకు (పూర్ణాంకం isrc = 0; isrc < DSV4_HC; ++isrc) {
                కొరకు (పూర్ణాంకం idst = 0; idst < DSV4_HC; ++idst) {
                    స్థిరరాశి పూర్ణాంకం idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
స్థిర constexpr పూర్ణాంకం dsv4_hc_post_ఖండం_కొలత = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
మూస <తార్కిక has_comb>
స్థిర శూన్య dsv4_hc_post_f32_sycl(
        స్థిరరాశి దశాంశం * x, స్థిరరాశి దశాంశం * residual, స్థిరరాశి దశాంశం * post, స్థిరరాశి దశాంశం * comb, దశాంశం * dst,
        int64_t n_embd, int64_t hc, int64_t n_టోకెన్లు,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        వరుస_ptr stream) {
    స్థిరరాశి int64_t nr = n_embd * hc * n_టోకెన్లు;
    స్థిరరాశి int64_t ఖండం_కొలత_3 = dsv4_hc_post_ఖండం_కొలత;
    స్థిరరాశి int64_t అంకె_ఖండాలు = (nr + ఖండం_కొలత_3 - 1) / ఖండం_కొలత_3;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(అంకె_ఖండాలు * ఖండం_కొలత_3), sycl::range<1>(ఖండం_కొలత_3)),
        [=](sycl::nd_అంశం<1> అంశం) {
            స్థిరరాశి int64_t ir = అంశం.get_global_id(0);
            ఒకవేళ (ir >= nr) {
                తిరిగి;
            }

            స్థిరరాశి int64_t i0   = ir % n_embd;
            స్థిరరాశి int64_t idst = (ir / n_embd) % hc;
            స్థిరరాశి int64_t it   = ir / (n_embd * hc);

            దశాంశం కూడిక = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
            ఒకవేళ constexpr (has_comb) {
                కొరకు (int64_t isrc = 0; isrc < hc; ++isrc) {
                    కూడిక += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } లేకపోతే {
                కూడిక += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = కూడిక;
        });
}

శూన్య ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    స్థిరరాశి ggml_tensor * x       = dst->src[0];
    స్థిరరాశి ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    స్థిరరాశి int64_t n_embd   = x->ne[0];
    స్థిరరాశి int64_t hc       = x->ne[1];
    స్థిరరాశి int64_t n_టోకెన్లు = x->ne[2];

    స్థిరరాశి దశాంశం scale = ggml_get_op_params_f32(dst, 0);
    స్థిరరాశి తార్కిక  gated = ggml_get_op_params_i32(dst, 1) != 0;

    వరుస_ptr stream = ctx.stream();

    ఒకవేళ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_టోకెన్లు);
        dsv4_hc_pre_f32_sycl<నిజం>(
                (స్థిరరాశి దశాంశం *) x->దత్తాంశం, (స్థిరరాశి దశాంశం *) weights->దత్తాంశం, (దశాంశం *) dst->దత్తాంశం,
                n_embd, hc, n_టోకెన్లు,
                nbx0 / పరిమాణం(దశాంశం), nbx1 / పరిమాణం(దశాంశం), nbx2 / పరిమాణం(దశాంశం),
                nbw0 / పరిమాణం(దశాంశం), nbw1 / పరిమాణం(దశాంశం), nbw2 / పరిమాణం(దశాంశం),
                nbd0 / పరిమాణం(దశాంశం), nbd1 / పరిమాణం(దశాంశం),
                scale, stream);
    } లేకపోతే {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_టోకెన్లు);
        dsv4_hc_pre_f32_sycl<అబద్ధం>(
                (స్థిరరాశి దశాంశం *) x->దత్తాంశం, (స్థిరరాశి దశాంశం *) weights->దత్తాంశం, (దశాంశం *) dst->దత్తాంశం,
                n_embd, hc, n_టోకెన్లు,
                nbx0 / పరిమాణం(దశాంశం), nbx1 / పరిమాణం(దశాంశం), nbx2 / పరిమాణం(దశాంశం),
                nbw0 / పరిమాణం(దశాంశం), nbw1 / పరిమాణం(దశాంశం), /*sw2=*/ 0,
                nbd0 / పరిమాణం(దశాంశం), nbd1 / పరిమాణం(దశాంశం),
                scale, stream);
    }
}

శూన్య ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    స్థిరరాశి ggml_tensor * mixes = dst->src[0];
    స్థిరరాశి ggml_tensor * scale = dst->src[1];
    స్థిరరాశి ggml_tensor * పునాది  = dst->src[2];

    GGML_ASSERT(mixes->type == GGML_TYPE_F32);
    GGML_ASSERT(scale->type == GGML_TYPE_F32);
    GGML_ASSERT(పునాది->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    constexpr int64_t hc_mix_dim = (2 + DSV4_HC)*DSV4_HC;

    GGML_ASSERT(mixes->ne[0] == hc_mix_dim);
    GGML_ASSERT(dst->ne[0] == DSV4_HC);
    GGML_ASSERT(dst->ne[1] == DSV4_HC);
    GGML_ASSERT(dst->ne[2] == mixes->ne[1]);
    GGML_ASSERT(scale->ne[0] >= 3);
    GGML_ASSERT(పునాది->ne[0] == hc_mix_dim);

    GGML_TENSOR_LOCALS(size_t, nbm, mixes, nb);
    GGML_TENSOR_LOCALS(size_t, nbs, scale, nb);
    GGML_TENSOR_LOCALS(size_t, nbb, పునాది,  nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,   nb);

    స్థిరరాశి int64_t n_టోకెన్లు = mixes->ne[1];
    స్థిరరాశి దశాంశం eps = ggml_get_op_params_f32(dst, 0);
    స్థిరరాశి int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    వరుస_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (స్థిరరాశి దశాంశం *) mixes->దత్తాంశం, (స్థిరరాశి దశాంశం *) scale->దత్తాంశం, (స్థిరరాశి దశాంశం *) పునాది->దత్తాంశం, (దశాంశం *) dst->దత్తాంశం,
            n_టోకెన్లు,
            nbm0 / పరిమాణం(దశాంశం), nbm1 / పరిమాణం(దశాంశం),
            nbs0 / పరిమాణం(దశాంశం),
            nbb0 / పరిమాణం(దశాంశం),
            nbd0 / పరిమాణం(దశాంశం), nbd1 / పరిమాణం(దశాంశం), nbd2 / పరిమాణం(దశాంశం),
            eps, n_iter, stream);
}

శూన్య ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    స్థిరరాశి ggml_tensor * x        = dst->src[0];
    స్థిరరాశి ggml_tensor * residual = dst->src[1];
    స్థిరరాశి ggml_tensor * post     = dst->src[2];
    స్థిరరాశి ggml_tensor * comb     = dst->src[3];

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
    ఒకవేళ (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    స్థిరరాశి int64_t n_embd   = x->ne[0];
    స్థిరరాశి int64_t n_టోకెన్లు = x->ne[1];
    స్థిరరాశి int64_t hc       = residual->ne[1];

    వరుస_ptr stream = ctx.stream();

    స్థిరరాశి స్వయంచాలక launch = [&](స్వయంచాలక has_comb) {
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value>(
            (స్థిరరాశి దశాంశం *) x->దత్తాంశం, (స్థిరరాశి దశాంశం *) residual->దత్తాంశం,
            (స్థిరరాశి దశాంశం *) post->దత్తాంశం, comb ? (స్థిరరాశి దశాంశం *) comb->దత్తాంశం : శూన్యసూచిక, (దశాంశం *) dst->దత్తాంశం,
            n_embd, hc, n_టోకెన్లు,
            nbx0 / పరిమాణం(దశాంశం), nbx1 / పరిమాణం(దశాంశం),
            nbr0 / పరిమాణం(దశాంశం), nbr1 / పరిమాణం(దశాంశం), nbr2 / పరిమాణం(దశాంశం),
            nbp0 / పరిమాణం(దశాంశం), nbp1 / పరిమాణం(దశాంశం),
            nbc0 / పరిమాణం(దశాంశం), nbc1 / పరిమాణం(దశాంశం), nbc2 / పరిమాణం(దశాంశం),
            nbd0 / పరిమాణం(దశాంశం), nbd1 / పరిమాణం(దశాంశం), nbd2 / పరిమాణం(దశాంశం),
            stream);
    };

    ఒకవేళ (comb) {
        launch(std::true_type{});
    } లేకపోతే {
        launch(std::false_type{});
    }
}
