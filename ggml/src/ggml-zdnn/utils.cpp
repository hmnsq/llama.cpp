#अन्तर्भाव "ggml.h"
#अन्तर्भाव "utils.hpp"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cmath>
#अन्तर्भाव <cstdlib>

zdnn_data_types ggml_zdnn_type_mapping(ggml_type type) {
    विकल्प (type) {
        पक्ष GGML_TYPE_F32:
            प्रत्यागम FP32;
        पक्ष GGML_TYPE_F16:
            प्रत्यागम FP16;
        पक्ष GGML_TYPE_BF16:
            प्रत्यागम BFLOAT;
        पक्ष GGML_TYPE_I8:
            प्रत्यागम INT8;
        पक्ष GGML_TYPE_I32:
            प्रत्यागम INT32;
        स्वभाव:
            GGML_ABORT("%s: fatal: unable to determine zTensor data type",
                       __func__);
            भङ्ग;
    }
}

शून्य ggml_zdnn_create_tensor(zdnn_tensor_desc  & pre_tfm_desc,
                             zdnn_tensor_desc  & tfm_desc,
                             zdnn_ztensor      & ztensor,
                       ध्रुव ggml_tensor       * src,
                       ध्रुव int64_t           * ne,
                       ध्रुव zdnn_data_layouts   layout) {
    zdnn_init_pre_transformed_desc(
        layout,
        ggml_zdnn_type_mapping(src->type),
        &pre_tfm_desc,
        ne[3], ne[2], ne[1], ne[0]
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_tfm_desc, &tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_tfm_desc, &tfm_desc, &ztensor));
}

शून्य ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, शून्य * buffer) {
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor, buffer));
}

// size of the groups along ne0 that get their own scale, or 0 for one scale per row
// the groups are spread evenly over ne0 and rounded up to 128 values, the last group is zero padded
int64_t ggml_zdnn_q_group_size(ध्रुव ggml_tensor * tensor) {
    // benchmarked on Llama-3.2-1B and Qwen2.5-1.5B Q8_0 with the channel scales:
    // 2048 is 15-18% faster than 1024, and 1024 lowers the KL divergence to the CPU by 13-19%
    // GGML_ZDNN_HIGH_PREC=1 selects 1024 for users that want the lower loss
    // always re-measure with different models to see if different sizes need a dispatching logic
    स्थिर ध्रुव तार्किक ऊपरी_prec = std::getenv("GGML_ZDNN_HIGH_PREC") != nullptr &&
                                  std::atoi(std::getenv("GGML_ZDNN_HIGH_PREC")) != 0;
    ध्रुव int64_t group_size_max = ऊपरी_prec ? 1024 : 2048;

    ध्रुव int64_t ne0 = tensor->ne[0];
    यदि (ne0 <= group_size_max) {
        प्रत्यागम 0;
    }

    ध्रुव int64_t n_groups = (ne0 + group_size_max - 1) / group_size_max;
    प्रत्यागम GGML_PAD((ne0 + n_groups - 1) / n_groups, 128);
}

शून्य ggml_zdnn_load_tensor_q(ggml_backend_zdnn_buffer * buffer, ध्रुव ggml_tensor * tensor) {
    ध्रुव int64_t ne0 = tensor->ne[0];
    ध्रुव int64_t ne1 = tensor->ne[1];

    ध्रुव ggml_to_float_t to_float = ggml_get_type_traits(tensor->type)->to_float;
    GGML_ASSERT(to_float != nullptr);

    // zDNN takes one scale per zTensor, so requantize each group of each row to int8 with its own scale
    // a group is the whole row, unless the weights are stacked into groups along ne0
    // rows are written transposed as zDNN computes inputs (ne11, ne0) x weights (ne0, ne1)
    ध्रुव तार्किक    stacked  = buffer->pre_tfm_desc.layout == ZDNN_3DS;
    ध्रुव int64_t n_groups = stacked ? buffer->pre_tfm_desc.dim3 : 1;
    ध्रुव int64_t gs       = stacked ? buffer->pre_tfm_desc.dim2 : ne0;

    // benchmarked: 256 rows fill a 256-byte cache line per write, padding each row by a line avoids cache set conflicts
    constexpr int64_t n_tile   = 256;
    ध्रुव     int64_t x_stride = ne0 + 64;

    // the rows past ne0 of the last group stay zero as padding
    std::vector<int8_t> q(n_groups * gs * ne1);
    std::vector<भिन्न>  x(n_tile * x_stride);
    std::vector<भिन्न>  id(n_tile * n_groups);

    // a few input channels with larger weights than the rest would set the scale of every group,
    // so divide each input channel by its RMS over the rows and multiply the inputs by it instead
    // with groups of 1024, this lowered the KL divergence to the CPU by 10-18% on Llama-3.2-1B and Qwen2.5-1.5B Q8_0 at the same speed
    std::vector<द्विगुण> sum_sq(ne0, 0.0);
    प्रति (int64_t i1 = 0; i1 < ne1; i1++) {
        to_float((ध्रुव वर्ण *)tensor->data + i1*tensor->nb[1], x.data(), ne0);
        प्रति (int64_t i0 = 0; i0 < ne0; i0++) {
            sum_sq[i0] += (द्विगुण)x[i0] * x[i0];
        }
    }

    std::vector<भिन्न> channel_id(ne0);
    प्रति (int64_t i0 = 0; i0 < ne0; i0++) {
        ध्रुव भिन्न c = (भिन्न)std::sqrt(sum_sq[i0] / ne1);
        buffer->channel_scales[i0] = c ? c : 1.0f;
        channel_id[i0] = 1.0f / buffer->channel_scales[i0];
    }

    प्रति (int64_t i1 = 0; i1 < ne1; i1 += n_tile) {
        ध्रुव int64_t nt = std::min(n_tile, ne1 - i1);

        प्रति (int64_t t = 0; t < nt; t++) {
            भिन्न * xt = x.data() + t*x_stride;
            to_float((ध्रुव वर्ण *)tensor->data + (i1 + t)*tensor->nb[1], xt, ne0);

            प्रति (int64_t i0 = 0; i0 < ne0; i0++) {
                xt[i0] *= channel_id[i0];
            }

            प्रति (int64_t g = 0; g < n_groups; g++) {
                भिन्न amax = 0.0f;
                प्रति (int64_t i0 = g*gs; i0 < std::min((g + 1)*gs, ne0); i0++) {
                    amax = std::max(amax, fabsf(xt[i0]));
                }

                ध्रुव भिन्न d = amax / 127.0f;
                buffer->scales[g*ne1 + i1 + t] = d;
                id[t*n_groups + g] = d ? 1.0f/d : 0.0f;
            }
        }

        प्रति (int64_t i0 = 0; i0 < ne0; i0++) {
            ध्रुव int64_t g = i0 / gs;
            int8_t * qi = q.data() + i0*ne1 + i1;
            प्रति (int64_t t = 0; t < nt; t++) {
                qi[t] = (int8_t)roundf(x[t*x_stride + i0] * id[t*n_groups + g]);
            }
        }
    }

    यदि (buffer->ztensor.is_transformed) {
        zdnn_reset_ztensor(&buffer->ztensor);
    }
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&buffer->ztensor, असत्य, -127, 127, q.data()));
}

शून्य ggml_zdnn_init_tensor(ggml_backend_zdnn_buffer * buffer, ध्रुव ggml_tensor * tensor) {
    यदि (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(ggml_is_matrix(tensor));

        ध्रुव int64_t gs = ggml_zdnn_q_group_size(tensor);

        यदि (gs > 0) {
            // stack the weights into groups along ne0, each group of each row gets its own scale
            ध्रुव int64_t n_groups = (tensor->ne[0] + gs - 1) / gs;

            zdnn_init_pre_transformed_desc(
                ZDNN_3DS,
                INT8,
                &buffer->pre_tfm_desc,
                n_groups, gs, tensor->ne[1]
            );

            buffer->scales.resize(n_groups * tensor->ne[1]);
        } अन्यथा {
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
        प्रत्यागम;
    }

    विकल्प (tensor->op) {
        पक्ष GGML_OP_MUL_MAT:
            {
                zdnn_init_pre_transformed_desc(
                    ZDNN_2D,
                    ggml_zdnn_type_mapping(tensor->type),
                    &buffer->pre_tfm_desc,
                    tensor->ne[1], tensor->ne[0]
                );
            } भङ्ग;

        स्वभाव:
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
            } भङ्ग;
    }

    ZDNN_CHECK(zdnn_generate_transformed_desc(&buffer->pre_tfm_desc, &buffer->tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&buffer->pre_tfm_desc, &buffer->tfm_desc, &buffer->ztensor));
}
