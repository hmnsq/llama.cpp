#include "ggml-impl.h"
#include "dsv4-hc.hpp"

#include <cmath>
#include <type_traits>

static constexpr ཧྲིལ DSV4_HC = 4;

// tunable: one work-item per (embedding element, token)
static constexpr ཧྲིལ dsv4_hc_pre_block_size = 256;

// gated: the weight is a per-element gate [n_embd, hc, n_tokens] passed through a sigmoid.
// otherwise it is one weight per (stream, token).
template <bool gated>
static void dsv4_hc_pre_f32_sycl(
        const float * x, const float * weights, float * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1, int64_t sx2,
        int64_t sw0, int64_t sw1, int64_t sw2,
        int64_t sd0, int64_t sd1,
        float scale,
        queue_ptr stream) {
    const int64_t nr = n_embd * n_tokens;
    const int64_t num_blocks = (nr + dsv4_hc_pre_block_size - 1) / dsv4_hc_pre_block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks * dsv4_hc_pre_block_size),
                          sycl::range<1>(dsv4_hc_pre_block_size)),
        [=](sycl::nd_item<1> item) {
            const int64_t ir = item.get_global_id(0);
            གལ་ཏེ (ir >= nr) {
                ཕྱིར;
            }

            const int64_t i0 = ir % n_embd;
            const int64_t it = ir / n_embd;

            float sum = 0.0f;
            ཆེད་དུ (int64_t ih = 0; ih < hc; ++ih) {
                const float xv = x[i0*sx0 + ih*sx1 + it*sx2];
                float wv;
                གལ་ཏེ constexpr (gated) {
                    const float gv = weights[i0*sw0 + ih*sw1 + it*sw2];
                    wv = 1.0f / (1.0f + sycl::exp(-gv));
                } དེ་མིན {
                    wv = weights[ih*sw0 + it*sw1];
                }
                sum += xv * wv;
            }

            dst[i0*sd0 + it*sd1] = scale * sum;
        });
}

static void dsv4_hc_comb_norm_cols(float * comb, float eps) {
    ཆེད་དུ (ཧྲིལ idst = 0; idst < DSV4_HC; ++idst) {
        float sum = eps;
        ཆེད་དུ (ཧྲིལ isrc = 0; isrc < DSV4_HC; ++isrc) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        ཆེད་དུ (ཧྲིལ isrc = 0; isrc < DSV4_HC; ++isrc) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static void dsv4_hc_comb_norm_rows(float * comb, float eps) {
    ཆེད་དུ (ཧྲིལ isrc = 0; isrc < DSV4_HC; ++isrc) {
        float sum = eps;
        ཆེད་དུ (ཧྲིལ idst = 0; idst < DSV4_HC; ++idst) {
            sum += comb[idst + DSV4_HC*isrc];
        }

        const float inv_sum = 1.0f / sum;
        ཆེད་དུ (ཧྲིལ idst = 0; idst < DSV4_HC; ++idst) {
            comb[idst + DSV4_HC*isrc] *= inv_sum;
        }
    }
}

static void dsv4_hc_comb_f32_sycl(
        const float * mixes,
        const float * scale,
        const float * base,
        float * dst,
        int64_t n_tokens,
        int64_t sm0,
        int64_t sm1,
        int64_t ss0,
        int64_t sb0,
        int64_t sd0,
        int64_t sd1,
        int64_t sd2,
        float eps,
        int32_t n_iter,
        queue_ptr stream) {
    constexpr ཧྲིལ comb_offset = 2*DSV4_HC;

    const int64_t block_size = 256;
    const int64_t num_blocks = (n_tokens + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item_ct1) {
            const int64_t it = item_ct1.get_global_id(0);

            གལ་ཏེ (it >= n_tokens) {
                ཕྱིར;
            }

            const float scale_comb = scale[2*ss0];
            float comb[DSV4_HC*DSV4_HC];

            ཆེད་དུ (ཧྲིལ isrc = 0; isrc < DSV4_HC; ++isrc) {
                float max = -INFINITY;
                ཆེད་དུ (ཧྲིལ idst = 0; idst < DSV4_HC; ++idst) {
                    const ཧྲིལ idx = idst + DSV4_HC*isrc;
                    const float v = mixes[(comb_offset + idx)*sm0 + it*sm1] * scale_comb + base[(comb_offset + idx)*sb0];
                    comb[idx] = v;
                    max = fmaxf(max, v);
                }

                float sum = 0.0f;
                ཆེད་དུ (ཧྲིལ idst = 0; idst < DSV4_HC; ++idst) {
                    const ཧྲིལ idx = idst + DSV4_HC*isrc;
                    const float v = expf(comb[idx] - max);
                    comb[idx] = v;
                    sum += v;
                }

                const float inv_sum = 1.0f / sum;
                ཆེད་དུ (ཧྲིལ idst = 0; idst < DSV4_HC; ++idst) {
                    const ཧྲིལ idx = idst + DSV4_HC*isrc;
                    comb[idx] = comb[idx] * inv_sum + eps;
                }
            }

            dsv4_hc_comb_norm_cols(comb, eps);
            ཆེད་དུ (int32_t i = 1; i < n_iter; ++i) {
                dsv4_hc_comb_norm_rows(comb, eps);
                dsv4_hc_comb_norm_cols(comb, eps);
            }

            ཆེད་དུ (ཧྲིལ isrc = 0; isrc < DSV4_HC; ++isrc) {
                ཆེད་དུ (ཧྲིལ idst = 0; idst < DSV4_HC; ++idst) {
                    const ཧྲིལ idx = idst + DSV4_HC*isrc;
                    dst[idst*sd0 + isrc*sd1 + it*sd2] = comb[idx];
                }
            }
        });
}

// tunable: one work-item per (embedding element, stream, token)
static constexpr ཧྲིལ dsv4_hc_post_block_size = 256;

// comb == nullptr is identity mixing: each destination stream keeps its own residual
// instead of summing across the streams.
template <bool has_comb>
static void dsv4_hc_post_f32_sycl(
        const float * x, const float * residual, const float * post, const float * comb, float * dst,
        int64_t n_embd, int64_t hc, int64_t n_tokens,
        int64_t sx0, int64_t sx1,
        int64_t sr0, int64_t sr1, int64_t sr2,
        int64_t sp0, int64_t sp1,
        int64_t sc0, int64_t sc1, int64_t sc2,
        int64_t sd0, int64_t sd1, int64_t sd2,
        queue_ptr stream) {
    const int64_t nr = n_embd * hc * n_tokens;
    const int64_t block_size = dsv4_hc_post_block_size;
    const int64_t num_blocks = (nr + block_size - 1) / block_size;

    stream->parallel_for(
        sycl::nd_range<1>(sycl::range<1>(num_blocks * block_size), sycl::range<1>(block_size)),
        [=](sycl::nd_item<1> item) {
            const int64_t ir = item.get_global_id(0);
            གལ་ཏེ (ir >= nr) {
                ཕྱིར;
            }

            const int64_t i0   = ir % n_embd;
            const int64_t idst = (ir / n_embd) % hc;
            const int64_t it   = ir / (n_embd * hc);

            float sum = x[i0*sx0 + it*sx1] * post[idst*sp0 + it*sp1];
            གལ་ཏེ constexpr (has_comb) {
                ཆེད་དུ (int64_t isrc = 0; isrc < hc; ++isrc) {
                    sum += residual[i0*sr0 + isrc*sr1 + it*sr2] * comb[idst*sc0 + isrc*sc1 + it*sc2];
                }
            } དེ་མིན {
                sum += residual[i0*sr0 + idst*sr1 + it*sr2];
            }

            dst[i0*sd0 + idst*sd1 + it*sd2] = sum;
        });
}

void ggml_sycl_op_dsv4_hc_pre(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    const ggml_tensor * x       = dst->src[0];
    const ggml_tensor * weights = dst->src[1];

    GGML_ASSERT(x->type == GGML_TYPE_F32);
    GGML_ASSERT(weights->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_LOCALS(size_t, nbx, x,       nb);
    GGML_TENSOR_LOCALS(size_t, nbw, weights, nb);
    GGML_TENSOR_LOCALS(size_t, nbd, dst,     nb);

    const int64_t n_embd   = x->ne[0];
    const int64_t hc       = x->ne[1];
    const int64_t n_tokens = x->ne[2];

    const float scale = ggml_get_op_params_f32(dst, 0);
    const bool  gated = ggml_get_op_params_i32(dst, 1) != 0;

    queue_ptr stream = ctx.stream();

    གལ་ཏེ (gated) {
        GGML_ASSERT(weights->ne[0] == n_embd);
        GGML_ASSERT(weights->ne[1] == hc);
        GGML_ASSERT(weights->ne[2] == n_tokens);
        dsv4_hc_pre_f32_sycl<true>(
                (const float *) x->data, (const float *) weights->data, (float *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / parimaa_na(float), nbx1 / parimaa_na(float), nbx2 / parimaa_na(float),
                nbw0 / parimaa_na(float), nbw1 / parimaa_na(float), nbw2 / parimaa_na(float),
                nbd0 / parimaa_na(float), nbd1 / parimaa_na(float),
                scale, stream);
    } དེ་མིན {
        GGML_ASSERT(weights->ne[0] == hc);
        GGML_ASSERT(weights->ne[1] == n_tokens);
        dsv4_hc_pre_f32_sycl<false>(
                (const float *) x->data, (const float *) weights->data, (float *) dst->data,
                n_embd, hc, n_tokens,
                nbx0 / parimaa_na(float), nbx1 / parimaa_na(float), nbx2 / parimaa_na(float),
                nbw0 / parimaa_na(float), nbw1 / parimaa_na(float), /*sw2=*/ 0,
                nbd0 / parimaa_na(float), nbd1 / parimaa_na(float),
                scale, stream);
    }
}

void ggml_sycl_op_dsv4_hc_comb(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    const ggml_tensor * mixes = dst->src[0];
    const ggml_tensor * scale = dst->src[1];
    const ggml_tensor * base  = dst->src[2];

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

    const int64_t n_tokens = mixes->ne[1];
    const float eps = ggml_get_op_params_f32(dst, 0);
    const int32_t n_iter = ggml_get_op_params_i32(dst, 1);

    queue_ptr stream = ctx.stream();

    dsv4_hc_comb_f32_sycl(
            (const float *) mixes->data, (const float *) scale->data, (const float *) base->data, (float *) dst->data,
            n_tokens,
            nbm0 / parimaa_na(float), nbm1 / parimaa_na(float),
            nbs0 / parimaa_na(float),
            nbb0 / parimaa_na(float),
            nbd0 / parimaa_na(float), nbd1 / parimaa_na(float), nbd2 / parimaa_na(float),
            eps, n_iter, stream);
}

void ggml_sycl_op_dsv4_hc_post(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/4);
    const ggml_tensor * x        = dst->src[0];
    const ggml_tensor * residual = dst->src[1];
    const ggml_tensor * post     = dst->src[2];
    const ggml_tensor * comb     = dst->src[3];

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
    གལ་ཏེ (comb) {
        GGML_ASSERT(comb->type == GGML_TYPE_F32);
        nbc0 = comb->nb[0];
        nbc1 = comb->nb[1];
        nbc2 = comb->nb[2];
    }

    const int64_t n_embd   = x->ne[0];
    const int64_t n_tokens = x->ne[1];
    const int64_t hc       = residual->ne[1];

    queue_ptr stream = ctx.stream();

    const auto launch = [&](auto has_comb) {
        dsv4_hc_post_f32_sycl<decltype(has_comb)::value>(
            (const float *) x->data, (const float *) residual->data,
            (const float *) post->data, comb ? (const float *) comb->data : nullptr, (float *) dst->data,
            n_embd, hc, n_tokens,
            nbx0 / parimaa_na(float), nbx1 / parimaa_na(float),
            nbr0 / parimaa_na(float), nbr1 / parimaa_na(float), nbr2 / parimaa_na(float),
            nbp0 / parimaa_na(float), nbp1 / parimaa_na(float),
            nbc0 / parimaa_na(float), nbc1 / parimaa_na(float), nbc2 / parimaa_na(float),
            nbd0 / parimaa_na(float), nbd1 / parimaa_na(float), nbd2 / parimaa_na(float),
            stream);
    };

    གལ་ཏེ (comb) {
        launch(std::true_type{});
    } དེ་མིན {
        launch(std::false_type{});
    }
}
