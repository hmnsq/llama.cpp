#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "utils.hpp"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cmath>
#অন্তর্ভাব <cstdlib>

zdnn_তথ্য_types ggml_zdnn_type_mapping(ggml_type type) {
    নির্বাচন (type) {
        ক্ষেত্র GGML_TYPE_F32:
            ফেরত FP32;
        ক্ষেত্র GGML_TYPE_F16:
            ফেরত FP16;
        ক্ষেত্র GGML_TYPE_BF16:
            ফেরত BFLOAT;
        ক্ষেত্র GGML_TYPE_I8:
            ফেরত INT8;
        ক্ষেত্র GGML_TYPE_I32:
            ফেরত INT32;
        পূর্বনির্ধারিত:
            GGML_ABORT("%s: fatal: unable to determine zTensor data type",
                       __func__);
            বিরতি;
    }
}

শূন্য ggml_zdnn_create_tensor(zdnn_tensor_desc  & pre_tfm_desc,
                             zdnn_tensor_desc  & tfm_desc,
                             zdnn_ztensor      & ztensor,
                       ধ্রুবক ggml_tensor       * src,
                       ধ্রুবক int64_t           * ne,
                       ধ্রুবক zdnn_তথ্য_layouts   layout) {
    zdnn_init_pre_transformed_desc(
        layout,
        ggml_zdnn_type_mapping(src->type),
        &pre_tfm_desc,
        ne[3], ne[2], ne[1], ne[0]
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_tfm_desc, &tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_tfm_desc, &tfm_desc, &ztensor));
}

শূন্য ggml_zdnn_load_tensor(zdnn_ztensor & ztensor, শূন্য * বাফার) {
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor, বাফার));
}

// size of the groups along ne0 that get their own scale, or 0 for one scale per row
// the groups are spread evenly over ne0 and rounded up to 128 values, the last group is zero padded
int64_t ggml_zdnn_q_group_আকার(ধ্রুবক ggml_tensor * tensor) {
    // benchmarked on Llama-3.2-1B and Qwen2.5-1.5B Q8_0 with the channel scales:
    // 2048 is 15-18% faster than 1024, and 1024 lowers the KL divergence to the CPU by 13-19%
    // GGML_ZDNN_HIGH_PREC=1 selects 1024 for users that want the lower loss
    // always re-measure with different models to see if different sizes need a dispatching logic
    স্থির ধ্রুবক যৌক্তিক উপরের_prec = std::getenv("GGML_ZDNN_HIGH_PREC") != nullptr &&
                                  std::atoi(std::getenv("GGML_ZDNN_HIGH_PREC")) != 0;
    ধ্রুবক int64_t group_আকার_max = উপরের_prec ? 1024 : 2048;

    ধ্রুবক int64_t ne0 = tensor->ne[0];
    যদি (ne0 <= group_আকার_max) {
        ফেরত 0;
    }

    ধ্রুবক int64_t n_groups = (ne0 + group_আকার_max - 1) / group_আকার_max;
    ফেরত GGML_PAD((ne0 + n_groups - 1) / n_groups, 128);
}

শূন্য ggml_zdnn_load_tensor_q(ggml_backend_zdnn_বাফার * বাফার, ধ্রুবক ggml_tensor * tensor) {
    ধ্রুবক int64_t ne0 = tensor->ne[0];
    ধ্রুবক int64_t ne1 = tensor->ne[1];

    ধ্রুবক ggml_to_float_t to_float = ggml_get_type_traits(tensor->type)->to_float;
    GGML_ASSERT(to_float != nullptr);

    // zDNN takes one scale per zTensor, so requantize each group of each row to int8 with its own scale
    // a group is the whole row, unless the weights are stacked into groups along ne0
    // rows are written transposed as zDNN computes inputs (ne11, ne0) x weights (ne0, ne1)
    ধ্রুবক যৌক্তিক    stacked  = বাফার->pre_tfm_desc.layout == ZDNN_3DS;
    ধ্রুবক int64_t n_groups = stacked ? বাফার->pre_tfm_desc.dim3 : 1;
    ধ্রুবক int64_t gs       = stacked ? বাফার->pre_tfm_desc.dim2 : ne0;

    // benchmarked: 256 rows fill a 256-byte cache line per write, padding each row by a line avoids cache set conflicts
    constexpr int64_t n_tile   = 256;
    ধ্রুবক     int64_t x_stride = ne0 + 64;

    // the rows past ne0 of the last group stay zero as padding
    std::vector<int8_t> q(n_groups * gs * ne1);
    std::vector<দশমিক>  x(n_tile * x_stride);
    std::vector<দশমিক>  id(n_tile * n_groups);

    // a few input channels with larger weights than the rest would set the scale of every group,
    // so divide each input channel by its RMS over the rows and multiply the inputs by it instead
    // with groups of 1024, this lowered the KL divergence to the CPU by 10-18% on Llama-3.2-1B and Qwen2.5-1.5B Q8_0 at the same speed
    std::vector<দ্বিগুণ> যোগ_sq(ne0, 0.0);
    জন্য (int64_t i1 = 0; i1 < ne1; i1++) {
        to_float((ধ্রুবক অক্ষর *)tensor->তথ্য + i1*tensor->nb[1], x.তথ্য(), ne0);
        জন্য (int64_t i0 = 0; i0 < ne0; i0++) {
            যোগ_sq[i0] += (দ্বিগুণ)x[i0] * x[i0];
        }
    }

    std::vector<দশমিক> channel_id(ne0);
    জন্য (int64_t i0 = 0; i0 < ne0; i0++) {
        ধ্রুবক দশমিক c = (দশমিক)std::sqrt(যোগ_sq[i0] / ne1);
        বাফার->channel_scales[i0] = c ? c : 1.0f;
        channel_id[i0] = 1.0f / বাফার->channel_scales[i0];
    }

    জন্য (int64_t i1 = 0; i1 < ne1; i1 += n_tile) {
        ধ্রুবক int64_t nt = std::min(n_tile, ne1 - i1);

        জন্য (int64_t t = 0; t < nt; t++) {
            দশমিক * xt = x.তথ্য() + t*x_stride;
            to_float((ধ্রুবক অক্ষর *)tensor->তথ্য + (i1 + t)*tensor->nb[1], xt, ne0);

            জন্য (int64_t i0 = 0; i0 < ne0; i0++) {
                xt[i0] *= channel_id[i0];
            }

            জন্য (int64_t g = 0; g < n_groups; g++) {
                দশমিক amax = 0.0f;
                জন্য (int64_t i0 = g*gs; i0 < std::min((g + 1)*gs, ne0); i0++) {
                    amax = std::max(amax, fabsf(xt[i0]));
                }

                ধ্রুবক দশমিক d = amax / 127.0f;
                বাফার->scales[g*ne1 + i1 + t] = d;
                id[t*n_groups + g] = d ? 1.0f/d : 0.0f;
            }
        }

        জন্য (int64_t i0 = 0; i0 < ne0; i0++) {
            ধ্রুবক int64_t g = i0 / gs;
            int8_t * qi = q.তথ্য() + i0*ne1 + i1;
            জন্য (int64_t t = 0; t < nt; t++) {
                qi[t] = (int8_t)roundf(x[t*x_stride + i0] * id[t*n_groups + g]);
            }
        }
    }

    যদি (বাফার->ztensor.is_transformed) {
        zdnn_reset_ztensor(&বাফার->ztensor);
    }
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&বাফার->ztensor, মিথ্যা, -127, 127, q.তথ্য()));
}

শূন্য ggml_zdnn_init_tensor(ggml_backend_zdnn_বাফার * বাফার, ধ্রুবক ggml_tensor * tensor) {
    যদি (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(ggml_is_matrix(tensor));

        ধ্রুবক int64_t gs = ggml_zdnn_q_group_আকার(tensor);

        যদি (gs > 0) {
            // stack the weights into groups along ne0, each group of each row gets its own scale
            ধ্রুবক int64_t n_groups = (tensor->ne[0] + gs - 1) / gs;

            zdnn_init_pre_transformed_desc(
                ZDNN_3DS,
                INT8,
                &বাফার->pre_tfm_desc,
                n_groups, gs, tensor->ne[1]
            );

            বাফার->scales.resize(n_groups * tensor->ne[1]);
        } অন্যথা {
            zdnn_init_pre_transformed_desc(
                ZDNN_2D,
                INT8,
                &বাফার->pre_tfm_desc,
                tensor->ne[0], tensor->ne[1]
            );

            বাফার->scales.resize(tensor->ne[1]);
        }

        বাফার->channel_scales.resize(tensor->ne[0]);

        ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&বাফার->pre_tfm_desc, QUANTIZED_WEIGHTS_INT8, &বাফার->tfm_desc));
        ZDNN_CHECK(zdnn_init_quantized_ztensor_with_malloc(&বাফার->pre_tfm_desc, &বাফার->tfm_desc, 1.0f, 0.0f, &বাফার->ztensor));
        ফেরত;
    }

    নির্বাচন (tensor->op) {
        ক্ষেত্র GGML_OP_MUL_MAT:
            {
                zdnn_init_pre_transformed_desc(
                    ZDNN_2D,
                    ggml_zdnn_type_mapping(tensor->type),
                    &বাফার->pre_tfm_desc,
                    tensor->ne[1], tensor->ne[0]
                );
            } বিরতি;

        পূর্বনির্ধারিত:
            {
                // For 4D tensors, GGML uses NCHW layout. However, because zDNN
                // automatically transforms everything to NHWC, we will use it
                // directly to avoid the performance penalty changing the
                // layout and reshaping the tensor.
                zdnn_init_pre_transformed_desc(
                    ZDNN_NHWC,
                    ggml_zdnn_type_mapping(tensor->type),
                    &বাফার->pre_tfm_desc,
                    tensor->ne[3], tensor->ne[2], tensor->ne[1], tensor->ne[0]
                );

                // TODO: Consider adding a ggml check.
                // TODO: If tensor = 4D, use ZDNN_NCHW by default.
                // TODO: If tensor = 2D, use ZDNN_NHWC by default.
            } বিরতি;
    }

    ZDNN_CHECK(zdnn_generate_transformed_desc(&বাফার->pre_tfm_desc, &বাফার->tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&বাফার->pre_tfm_desc, &বাফার->tfm_desc, &বাফার->ztensor));
}
