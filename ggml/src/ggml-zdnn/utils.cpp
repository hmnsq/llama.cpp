#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "utils.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdlib>

zdnn_ਅੰਕੜੇ_types ggml_zdnn_type_mapping(ggml_type type) {
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_F32:
            ਵਾਪਸ FP32;
        ਹਾਲਤ GGML_TYPE_F16:
            ਵਾਪਸ FP16;
        ਹਾਲਤ GGML_TYPE_BF16:
            ਵਾਪਸ BFLOAT;
        ਹਾਲਤ GGML_TYPE_I8:
            ਵਾਪਸ INT8;
        ਹਾਲਤ GGML_TYPE_I32:
            ਵਾਪਸ INT32;
        ਮੂਲ:
            GGML_ABORT("%s: fatal: unable to determine zTensor data type",
                       __func__);
            ਰੋਕੋ;
    }
}

ਖਾਲੀ ggml_zdnn_create_tensor(zdnn_tensor_desc  & pre_tfm_desc,
                             zdnn_tensor_desc  & tfm_desc,
                             zdnn_ztensor      & ztensor,
                       ਅਚਲ ggml_tensor       * src,
                       ਅਚਲ int64_t           * ne,
                       ਅਚਲ zdnn_ਅੰਕੜੇ_layouts   layout) {
    zdnn_init_pre_transformed_desc(
        layout,
        ggml_zdnn_type_mapping(src->type),
        &pre_tfm_desc,
        ne[3], ne[2], ne[1], ne[0]
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_tfm_desc, &tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_tfm_desc, &tfm_desc, &ztensor));
}

ਖਾਲੀ ggml_zdnn_ਲੱਦੋ_tensor(zdnn_ztensor & ztensor, ਖਾਲੀ * ਬਫਰ) {
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor, ਬਫਰ));
}

// size of the groups along ne0 that get their own scale, or 0 for one scale per row
// the groups are spread evenly over ne0 and rounded up to 128 values, the last group is zero padded
int64_t ggml_zdnn_q_ਸਮੂਹ_ਮਾਪ(ਅਚਲ ggml_tensor * tensor) {
    // benchmarked on Llama-3.2-1B and Qwen2.5-1.5B Q8_0 with the channel scales:
    // 2048 is 15-18% faster than 1024, and 1024 lowers the KL divergence to the CPU by 13-19%
    // GGML_ZDNN_HIGH_PREC=1 selects 1024 for users that want the lower loss
    // always re-measure with different models to see if different sizes need a dispatching logic
    ਸਥਿਰ ਅਚਲ ਤਾਰਕਿਕ ਉੱਚਾ_prec = std::getenv("GGML_ZDNN_HIGH_PREC") != ਖਾਲੀ_ਸੂਚਕ &&
                                  std::atoi(std::getenv("GGML_ZDNN_HIGH_PREC")) != 0;
    ਅਚਲ int64_t ਸਮੂਹ_ਮਾਪ_max = ਉੱਚਾ_prec ? 1024 : 2048;

    ਅਚਲ int64_t ne0 = tensor->ne[0];
    ਜੇ (ne0 <= ਸਮੂਹ_ਮਾਪ_max) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ int64_t n_groups = (ne0 + ਸਮੂਹ_ਮਾਪ_max - 1) / ਸਮੂਹ_ਮਾਪ_max;
    ਵਾਪਸ GGML_PAD((ne0 + n_groups - 1) / n_groups, 128);
}

ਖਾਲੀ ggml_zdnn_ਲੱਦੋ_tensor_q(ggml_backend_zdnn_ਬਫਰ * ਬਫਰ, ਅਚਲ ggml_tensor * tensor) {
    ਅਚਲ int64_t ne0 = tensor->ne[0];
    ਅਚਲ int64_t ne1 = tensor->ne[1];

    ਅਚਲ ggml_ਮੰਜ਼ਿਲ_float_t ਮੰਜ਼ਿਲ_float = ggml_get_type_traits(tensor->type)->ਮੰਜ਼ਿਲ_float;
    GGML_ASSERT(ਮੰਜ਼ਿਲ_float != ਖਾਲੀ_ਸੂਚਕ);

    // zDNN takes one scale per zTensor, so requantize each group of each row to int8 with its own scale
    // a group is the whole row, unless the weights are stacked into groups along ne0
    // rows are written transposed as zDNN computes inputs (ne11, ne0) x weights (ne0, ne1)
    ਅਚਲ ਤਾਰਕਿਕ    stacked  = ਬਫਰ->pre_tfm_desc.layout == ZDNN_3DS;
    ਅਚਲ int64_t n_groups = stacked ? ਬਫਰ->pre_tfm_desc.dim3 : 1;
    ਅਚਲ int64_t gs       = stacked ? ਬਫਰ->pre_tfm_desc.dim2 : ne0;

    // benchmarked: 256 rows fill a 256-byte cache line per write, padding each row by a line avoids cache set conflicts
    constexpr int64_t n_tile   = 256;
    ਅਚਲ     int64_t x_stride = ne0 + 64;

    // the rows past ne0 of the last group stay zero as padding
    std::vector<int8_t> q(n_groups * gs * ne1);
    std::vector<ਦਸ਼ਮਲਵ>  x(n_tile * x_stride);
    std::vector<ਦਸ਼ਮਲਵ>  id(n_tile * n_groups);

    // a few input channels with larger weights than the rest would set the scale of every group,
    // so divide each input channel by its RMS over the rows and multiply the inputs by it instead
    // with groups of 1024, this lowered the KL divergence to the CPU by 10-18% on Llama-3.2-1B and Qwen2.5-1.5B Q8_0 at the same speed
    std::vector<ਦੁੱਗਣਾ> ਜੋੜ_sq(ne0, 0.0);
    ਲਈ (int64_t i1 = 0; i1 < ne1; i1++) {
        ਮੰਜ਼ਿਲ_float((ਅਚਲ ਅੱਖਰ *)tensor->ਅੰਕੜੇ + i1*tensor->nb[1], x.ਅੰਕੜੇ(), ne0);
        ਲਈ (int64_t i0 = 0; i0 < ne0; i0++) {
            ਜੋੜ_sq[i0] += (ਦੁੱਗਣਾ)x[i0] * x[i0];
        }
    }

    std::vector<ਦਸ਼ਮਲਵ> channel_id(ne0);
    ਲਈ (int64_t i0 = 0; i0 < ne0; i0++) {
        ਅਚਲ ਦਸ਼ਮਲਵ c = (ਦਸ਼ਮਲਵ)std::ਵਰਗਮੂਲ(ਜੋੜ_sq[i0] / ne1);
        ਬਫਰ->channel_scales[i0] = c ? c : 1.0f;
        channel_id[i0] = 1.0f / ਬਫਰ->channel_scales[i0];
    }

    ਲਈ (int64_t i1 = 0; i1 < ne1; i1 += n_tile) {
        ਅਚਲ int64_t nt = std::min(n_tile, ne1 - i1);

        ਲਈ (int64_t t = 0; t < nt; t++) {
            ਦਸ਼ਮਲਵ * xt = x.ਅੰਕੜੇ() + t*x_stride;
            ਮੰਜ਼ਿਲ_float((ਅਚਲ ਅੱਖਰ *)tensor->ਅੰਕੜੇ + (i1 + t)*tensor->nb[1], xt, ne0);

            ਲਈ (int64_t i0 = 0; i0 < ne0; i0++) {
                xt[i0] *= channel_id[i0];
            }

            ਲਈ (int64_t g = 0; g < n_groups; g++) {
                ਦਸ਼ਮਲਵ amax = 0.0f;
                ਲਈ (int64_t i0 = g*gs; i0 < std::min((g + 1)*gs, ne0); i0++) {
                    amax = std::max(amax, fabsf(xt[i0]));
                }

                ਅਚਲ ਦਸ਼ਮਲਵ d = amax / 127.0f;
                ਬਫਰ->scales[g*ne1 + i1 + t] = d;
                id[t*n_groups + g] = d ? 1.0f/d : 0.0f;
            }
        }

        ਲਈ (int64_t i0 = 0; i0 < ne0; i0++) {
            ਅਚਲ int64_t g = i0 / gs;
            int8_t * qi = q.ਅੰਕੜੇ() + i0*ne1 + i1;
            ਲਈ (int64_t t = 0; t < nt; t++) {
                qi[t] = (int8_t)roundf(x[t*x_stride + i0] * id[t*n_groups + g]);
            }
        }
    }

    ਜੇ (ਬਫਰ->ztensor.is_transformed) {
        zdnn_reset_ztensor(&ਬਫਰ->ztensor);
    }
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&ਬਫਰ->ztensor, ਝੂਠ, -127, 127, q.ਅੰਕੜੇ()));
}

ਖਾਲੀ ggml_zdnn_init_tensor(ggml_backend_zdnn_ਬਫਰ * ਬਫਰ, ਅਚਲ ggml_tensor * tensor) {
    ਜੇ (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(ggml_is_ਮੈਟ੍ਰਿਕਸ(tensor));

        ਅਚਲ int64_t gs = ggml_zdnn_q_ਸਮੂਹ_ਮਾਪ(tensor);

        ਜੇ (gs > 0) {
            // stack the weights into groups along ne0, each group of each row gets its own scale
            ਅਚਲ int64_t n_groups = (tensor->ne[0] + gs - 1) / gs;

            zdnn_init_pre_transformed_desc(
                ZDNN_3DS,
                INT8,
                &ਬਫਰ->pre_tfm_desc,
                n_groups, gs, tensor->ne[1]
            );

            ਬਫਰ->scales.resize(n_groups * tensor->ne[1]);
        } ਨਹੀਂਤਾਂ {
            zdnn_init_pre_transformed_desc(
                ZDNN_2D,
                INT8,
                &ਬਫਰ->pre_tfm_desc,
                tensor->ne[0], tensor->ne[1]
            );

            ਬਫਰ->scales.resize(tensor->ne[1]);
        }

        ਬਫਰ->channel_scales.resize(tensor->ne[0]);

        ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&ਬਫਰ->pre_tfm_desc, QUANTIZED_WEIGHTS_INT8, &ਬਫਰ->tfm_desc));
        ZDNN_CHECK(zdnn_init_quantized_ztensor_with_malloc(&ਬਫਰ->pre_tfm_desc, &ਬਫਰ->tfm_desc, 1.0f, 0.0f, &ਬਫਰ->ztensor));
        ਵਾਪਸ;
    }

    ਚੋਣ (tensor->op) {
        ਹਾਲਤ GGML_OP_MUL_MAT:
            {
                zdnn_init_pre_transformed_desc(
                    ZDNN_2D,
                    ggml_zdnn_type_mapping(tensor->type),
                    &ਬਫਰ->pre_tfm_desc,
                    tensor->ne[1], tensor->ne[0]
                );
            } ਰੋਕੋ;

        ਮੂਲ:
            {
                // For 4D tensors, GGML uses NCHW layout. However, because zDNN
                // automatically transforms everything to NHWC, we will use it
                // directly to avoid the performance penalty changing the
                // layout and reshaping the tensor.
                zdnn_init_pre_transformed_desc(
                    ZDNN_NHWC,
                    ggml_zdnn_type_mapping(tensor->type),
                    &ਬਫਰ->pre_tfm_desc,
                    tensor->ne[3], tensor->ne[2], tensor->ne[1], tensor->ne[0]
                );

                // TODO: Consider adding a ggml check.
                // TODO: If tensor = 4D, use ZDNN_NCHW by default.
                // TODO: If tensor = 2D, use ZDNN_NHWC by default.
            } ਰੋਕੋ;
    }

    ZDNN_CHECK(zdnn_generate_transformed_desc(&ਬਫਰ->pre_tfm_desc, &ਬਫਰ->tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&ਬਫਰ->pre_tfm_desc, &ਬਫਰ->tfm_desc, &ਬਫਰ->ztensor));
}
