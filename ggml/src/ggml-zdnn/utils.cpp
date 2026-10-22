#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "utils.hpp"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstdlib>

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type) {
    ವಿಕಲ್ಪ (type) {
        ಪಕ್ಷ GGML_TYPE_F32:
            ಪ್ರತ್ಯಾಗಮ FP32;
        ಪಕ್ಷ GGML_TYPE_F16:
            ಪ್ರತ್ಯಾಗಮ FP16;
        ಪಕ್ಷ GGML_TYPE_BF16:
            ಪ್ರತ್ಯಾಗಮ BFLOAT;
        ಪಕ್ಷ GGML_TYPE_I8:
            ಪ್ರತ್ಯಾಗಮ INT8;
        ಪಕ್ಷ GGML_TYPE_I32:
            ಪ್ರತ್ಯಾಗಮ INT32;
        ಸ್ವಭಾವ:
            GGML_ABORT("%s: fatal: unable to determine zTensor data type",
                       __func__);
            ಭಙ್ಗ;
    }
}

ಶೂನ್ಯ ggml_zdnn_create_tensor(zdnn_tensor_desc  & pre_tfm_desc,
                             zdnn_tensor_desc  & tfm_desc,
                             zdnn_ztensor      & ztensor,
                       ಧ್ರುವ ggml_tensor       * src,
                       ಧ್ರುವ int64_t           * ne,
                       ಧ್ರುವ zdnn_data_layouts   layout) {
    zdnn_init_pre_transformed_desc(
        layout,
        ggml_zdnn_type_mapping(src->type),
        &pre_tfm_desc,
        ne[3], ne[2], ne[1], ne[0]
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_tfm_desc, &tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_tfm_desc, &tfm_desc, &ztensor));
}

ಶೂನ್ಯ ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, ಶೂನ್ಯ * buffer) {
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor, buffer));
}

// size of the groups along ne0 that get their own scale, or 0 for one scale per row
// the groups are spread evenly over ne0 and rounded up to 128 values, the last group is zero padded
int64_t ggml_zdnn_q_group_size(ಧ್ರುವ ggml_tensor * tensor) {
    // benchmarked on Llama-3.2-1B and Qwen2.5-1.5B Q8_0 with the channel scales:
    // 2048 is 15-18% faster than 1024, and 1024 lowers the KL divergence to the CPU by 13-19%
    // GGML_ZDNN_HIGH_PREC=1 selects 1024 for users that want the lower loss
    // always re-measure with different models to see if different sizes need a dispatching logic
    ಸ್ಥಿರ ಧ್ರುವ ತಾರ್ಕಿಕ ಮೇಲಿನ_prec = std::getenv("GGML_ZDNN_HIGH_PREC") != nullptr &&
                                  std::atoi(std::getenv("GGML_ZDNN_HIGH_PREC")) != 0;
    ಧ್ರುವ int64_t group_size_max = ಮೇಲಿನ_prec ? 1024 : 2048;

    ಧ್ರುವ int64_t ne0 = tensor->ne[0];
    ಯದಿ (ne0 <= group_size_max) {
        ಪ್ರತ್ಯಾಗಮ 0;
    }

    ಧ್ರುವ int64_t n_groups = (ne0 + group_size_max - 1) / group_size_max;
    ಪ್ರತ್ಯಾಗಮ GGML_PAD((ne0 + n_groups - 1) / n_groups, 128);
}

ಶೂನ್ಯ ggml_zdnn_load_tensor_q(ggml_backend_zdnn_buffer * buffer, ಧ್ರುವ ggml_tensor * tensor) {
    ಧ್ರುವ int64_t ne0 = tensor->ne[0];
    ಧ್ರುವ int64_t ne1 = tensor->ne[1];

    ಧ್ರುವ ggml_to_float_t to_float = ggml_get_type_traits(tensor->type)->to_float;
    GGML_ASSERT(to_float != nullptr);

    // zDNN takes one scale per zTensor, so requantize each group of each row to int8 with its own scale
    // a group is the whole row, unless the weights are stacked into groups along ne0
    // rows are written transposed as zDNN computes inputs (ne11, ne0) x weights (ne0, ne1)
    ಧ್ರುವ ತಾರ್ಕಿಕ    stacked  = buffer->pre_tfm_desc.layout == ZDNN_3DS;
    ಧ್ರುವ int64_t n_groups = stacked ? buffer->pre_tfm_desc.dim3 : 1;
    ಧ್ರುವ int64_t gs       = stacked ? buffer->pre_tfm_desc.dim2 : ne0;

    // benchmarked: 256 rows fill a 256-byte cache line per write, padding each row by a line avoids cache set conflicts
    constexpr int64_t n_tile   = 256;
    ಧ್ರುವ     int64_t x_stride = ne0 + 64;

    // the rows past ne0 of the last group stay zero as padding
    std::vector<int8_t> q(n_groups * gs * ne1);
    std::vector<ಭಿನ್ನ>  x(n_tile * x_stride);
    std::vector<ಭಿನ್ನ>  id(n_tile * n_groups);

    // a few input channels with larger weights than the rest would set the scale of every group,
    // so divide each input channel by its RMS over the rows and multiply the inputs by it instead
    // with groups of 1024, this lowered the KL divergence to the CPU by 10-18% on Llama-3.2-1B and Qwen2.5-1.5B Q8_0 at the same speed
    std::vector<ದ್ವಿಗುಣ> sum_sq(ne0, 0.0);
    ಪ್ರತಿ (int64_t i1 = 0; i1 < ne1; i1++) {
        to_float((ಧ್ರುವ ವರ್ಣ *)tensor->data + i1*tensor->nb[1], x.data(), ne0);
        ಪ್ರತಿ (int64_t i0 = 0; i0 < ne0; i0++) {
            sum_sq[i0] += (ದ್ವಿಗುಣ)x[i0] * x[i0];
        }
    }

    std::vector<ಭಿನ್ನ> channel_id(ne0);
    ಪ್ರತಿ (int64_t i0 = 0; i0 < ne0; i0++) {
        ಧ್ರುವ ಭಿನ್ನ c = (ಭಿನ್ನ)std::sqrt(sum_sq[i0] / ne1);
        buffer->channel_scales[i0] = c ? c : 1.0f;
        channel_id[i0] = 1.0f / buffer->channel_scales[i0];
    }

    ಪ್ರತಿ (int64_t i1 = 0; i1 < ne1; i1 += n_tile) {
        ಧ್ರುವ int64_t nt = std::min(n_tile, ne1 - i1);

        ಪ್ರತಿ (int64_t t = 0; t < nt; t++) {
            ಭಿನ್ನ * xt = x.data() + t*x_stride;
            to_float((ಧ್ರುವ ವರ್ಣ *)tensor->data + (i1 + t)*tensor->nb[1], xt, ne0);

            ಪ್ರತಿ (int64_t i0 = 0; i0 < ne0; i0++) {
                xt[i0] *= channel_id[i0];
            }

            ಪ್ರತಿ (int64_t g = 0; g < n_groups; g++) {
                ಭಿನ್ನ amax = 0.0f;
                ಪ್ರತಿ (int64_t i0 = g*gs; i0 < std::min((g + 1)*gs, ne0); i0++) {
                    amax = std::max(amax, fabsf(xt[i0]));
                }

                ಧ್ರುವ ಭಿನ್ನ d = amax / 127.0f;
                buffer->scales[g*ne1 + i1 + t] = d;
                id[t*n_groups + g] = d ? 1.0f/d : 0.0f;
            }
        }

        ಪ್ರತಿ (int64_t i0 = 0; i0 < ne0; i0++) {
            ಧ್ರುವ int64_t g = i0 / gs;
            int8_t * qi = q.data() + i0*ne1 + i1;
            ಪ್ರತಿ (int64_t t = 0; t < nt; t++) {
                qi[t] = (int8_t)roundf(x[t*x_stride + i0] * id[t*n_groups + g]);
            }
        }
    }

    ಯದಿ (buffer->ztensor.is_transformed) {
        zdnn_reset_ztensor(&buffer->ztensor);
    }
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&buffer->ztensor, ಅಸತ್ಯ, -127, 127, q.data()));
}

ಶೂನ್ಯ ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, ಧ್ರುವ ggml_tensor * tensor) {
    ಯದಿ (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(ggml_is_matrix(tensor));

        ಧ್ರುವ int64_t gs = ggml_zdnn_q_group_size(tensor);

        ಯದಿ (gs > 0) {
            // stack the weights into groups along ne0, each group of each row gets its own scale
            ಧ್ರುವ int64_t n_groups = (tensor->ne[0] + gs - 1) / gs;

            zdnn_init_pre_transformed_desc(
                ZDNN_3DS,
                INT8,
                &buffer->pre_tfm_desc,
                n_groups, gs, tensor->ne[1]
            );

            buffer->scales.resize(n_groups * tensor->ne[1]);
        } ಅನ್ಯಥಾ {
            zdnn_init_pre_transformed_desc(
                ZDNN_2D,
                INT8,
                &buffer->pre_tfm_desc,
                tensor->ne[0], tensor->ne[1]
            );

            buffer->scales.resize(tensor->ne[1]);
        }

        buffer->channel_scales.resize(tensor->ne[0]);

        ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&buffer->pre_tfm_desc, QUANTIZED_WEIGHTS_INT8, &buffer->tfm_desc));
        ZDNN_CHECK(zdnn_init_quantized_ztensor_with_malloc(&buffer->pre_tfm_desc, &buffer->tfm_desc, 1.0f, 0.0f, &buffer->ztensor));
        ಪ್ರತ್ಯಾಗಮ;
    }

    ವಿಕಲ್ಪ (tensor->op) {
        ಪಕ್ಷ GGML_OP_MUL_MAT:
            {
                zdnn_init_pre_transformed_desc(
                    ZDNN_2D,
                    ggml_zdnn_type_mapping(tensor->type),
                    &buffer->pre_tfm_desc,
                    tensor->ne[1], tensor->ne[0]
                );
            } ಭಙ್ಗ;

        ಸ್ವಭಾವ:
            {
                // For 4D tensors, GGML uses NCHW layout. However, because zDNN
                // automatically transforms everything to NHWC, we will use it
                // directly to avoid the performance penalty changing the
                // layout and reshaping the tensor.
                zdnn_init_pre_transformed_desc(
                    ZDNN_NHWC,
                    ggml_zdnn_type_mapping(tensor->type),
                    &buffer->pre_tfm_desc,
                    tensor->ne[3], tensor->ne[2], tensor->ne[1], tensor->ne[0]
                );

                // TODO: Consider adding a ggml check.
                // TODO: If tensor = 4D, use ZDNN_NCHW by default.
                // TODO: If tensor = 2D, use ZDNN_NHWC by default.
            } ಭಙ್ಗ;
    }

    ZDNN_CHECK(zdnn_generate_transformed_desc(&buffer->pre_tfm_desc, &buffer->tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&buffer->pre_tfm_desc, &buffer->tfm_desc, &buffer->ztensor));
}
