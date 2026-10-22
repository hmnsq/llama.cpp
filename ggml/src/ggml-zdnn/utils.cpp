#شامل "ggml.h"
#شامل "utils.hpp"

#شامل <algorithm>
#شامل <cmath>
#شامل <cstdlib>

zdnn_ڈیٹا_types ggml_zdnn_type_mapping(ggml_type type) {
    انتخاب (type) {
        حالت GGML_TYPE_F32:
            واپس FP32;
        حالت GGML_TYPE_F16:
            واپس FP16;
        حالت GGML_TYPE_BF16:
            واپس BFLOAT;
        حالت GGML_TYPE_I8:
            واپس INT8;
        حالت GGML_TYPE_I32:
            واپس INT32;
        طےشدہ:
            GGML_ABORT("%s: fatal: unable to determine zTensor data type",
                       __func__);
            توڑو;
    }
}

خالی ggml_zdnn_create_tensor(zdnn_tensor_desc  & pre_tfm_desc,
                             zdnn_tensor_desc  & tfm_desc,
                             zdnn_ztensor      & ztensor,
                       ثابت ggml_tensor       * src,
                       ثابت int64_t           * ne,
                       ثابت zdnn_ڈیٹا_layouts   layout) {
    zdnn_init_pre_transformed_desc(
        layout,
        ggml_zdnn_type_mapping(src->type),
        &pre_tfm_desc,
        ne[3], ne[2], ne[1], ne[0]
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_tfm_desc, &tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_tfm_desc, &tfm_desc, &ztensor));
}

خالی ggml_zdnn_بھرو_tensor(zdnn_ztensor & ztensor, خالی * بفر) {
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor, بفر));
}

// size of the groups along ne0 that get their own scale, or 0 for one scale per row
// the groups are spread evenly over ne0 and rounded up to 128 values, the last group is zero padded
int64_t ggml_zdnn_q_گروہ_سائز(ثابت ggml_tensor * tensor) {
    // benchmarked on Llama-3.2-1B and Qwen2.5-1.5B Q8_0 with the channel scales:
    // 2048 is 15-18% faster than 1024, and 1024 lowers the KL divergence to the CPU by 13-19%
    // GGML_ZDNN_HIGH_PREC=1 selects 1024 for users that want the lower loss
    // always re-measure with different models to see if different sizes need a dispatching logic
    ساکن ثابت منطقی بلند_prec = std::getenv("GGML_ZDNN_HIGH_PREC") != خالی_اشاریہ &&
                                  std::atoi(std::getenv("GGML_ZDNN_HIGH_PREC")) != 0;
    ثابت int64_t گروہ_سائز_max = بلند_prec ? 1024 : 2048;

    ثابت int64_t ne0 = tensor->ne[0];
    اگر (ne0 <= گروہ_سائز_max) {
        واپس 0;
    }

    ثابت int64_t n_groups = (ne0 + گروہ_سائز_max - 1) / گروہ_سائز_max;
    واپس GGML_PAD((ne0 + n_groups - 1) / n_groups, 128);
}

خالی ggml_zdnn_بھرو_tensor_q(ggml_backend_zdnn_بفر * بفر, ثابت ggml_tensor * tensor) {
    ثابت int64_t ne0 = tensor->ne[0];
    ثابت int64_t ne1 = tensor->ne[1];

    ثابت ggml_مقصد_float_t مقصد_float = ggml_get_type_traits(tensor->type)->مقصد_float;
    GGML_ASSERT(مقصد_float != خالی_اشاریہ);

    // zDNN takes one scale per zTensor, so requantize each group of each row to int8 with its own scale
    // a group is the whole row, unless the weights are stacked into groups along ne0
    // rows are written transposed as zDNN computes inputs (ne11, ne0) x weights (ne0, ne1)
    ثابت منطقی    stacked  = بفر->pre_tfm_desc.layout == ZDNN_3DS;
    ثابت int64_t n_groups = stacked ? بفر->pre_tfm_desc.dim3 : 1;
    ثابت int64_t gs       = stacked ? بفر->pre_tfm_desc.dim2 : ne0;

    // benchmarked: 256 rows fill a 256-byte cache line per write, padding each row by a line avoids cache set conflicts
    constexpr int64_t n_tile   = 256;
    ثابت     int64_t x_stride = ne0 + 64;

    // the rows past ne0 of the last group stay zero as padding
    std::vector<int8_t> q(n_groups * gs * ne1);
    std::vector<اعشاری>  x(n_tile * x_stride);
    std::vector<اعشاری>  id(n_tile * n_groups);

    // a few input channels with larger weights than the rest would set the scale of every group,
    // so divide each input channel by its RMS over the rows and multiply the inputs by it instead
    // with groups of 1024, this lowered the KL divergence to the CPU by 10-18% on Llama-3.2-1B and Qwen2.5-1.5B Q8_0 at the same speed
    std::vector<دوہرا> مجموع_sq(ne0, 0.0);
    برائے (int64_t i1 = 0; i1 < ne1; i1++) {
        مقصد_float((ثابت حرف *)tensor->ڈیٹا + i1*tensor->nb[1], x.ڈیٹا(), ne0);
        برائے (int64_t i0 = 0; i0 < ne0; i0++) {
            مجموع_sq[i0] += (دوہرا)x[i0] * x[i0];
        }
    }

    std::vector<اعشاری> channel_id(ne0);
    برائے (int64_t i0 = 0; i0 < ne0; i0++) {
        ثابت اعشاری c = (اعشاری)std::جذر(مجموع_sq[i0] / ne1);
        بفر->channel_scales[i0] = c ? c : 1.0f;
        channel_id[i0] = 1.0f / بفر->channel_scales[i0];
    }

    برائے (int64_t i1 = 0; i1 < ne1; i1 += n_tile) {
        ثابت int64_t nt = std::min(n_tile, ne1 - i1);

        برائے (int64_t t = 0; t < nt; t++) {
            اعشاری * xt = x.ڈیٹا() + t*x_stride;
            مقصد_float((ثابت حرف *)tensor->ڈیٹا + (i1 + t)*tensor->nb[1], xt, ne0);

            برائے (int64_t i0 = 0; i0 < ne0; i0++) {
                xt[i0] *= channel_id[i0];
            }

            برائے (int64_t g = 0; g < n_groups; g++) {
                اعشاری amax = 0.0f;
                برائے (int64_t i0 = g*gs; i0 < std::min((g + 1)*gs, ne0); i0++) {
                    amax = std::max(amax, fabsf(xt[i0]));
                }

                ثابت اعشاری d = amax / 127.0f;
                بفر->scales[g*ne1 + i1 + t] = d;
                id[t*n_groups + g] = d ? 1.0f/d : 0.0f;
            }
        }

        برائے (int64_t i0 = 0; i0 < ne0; i0++) {
            ثابت int64_t g = i0 / gs;
            int8_t * qi = q.ڈیٹا() + i0*ne1 + i1;
            برائے (int64_t t = 0; t < nt; t++) {
                qi[t] = (int8_t)roundf(x[t*x_stride + i0] * id[t*n_groups + g]);
            }
        }
    }

    اگر (بفر->ztensor.is_transformed) {
        zdnn_reset_ztensor(&بفر->ztensor);
    }
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&بفر->ztensor, جھوٹ, -127, 127, q.ڈیٹا()));
}

خالی ggml_zdnn_init_tensor(ggml_backend_zdnn_بفر * بفر, ثابت ggml_tensor * tensor) {
    اگر (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(ggml_is_میٹرکس(tensor));

        ثابت int64_t gs = ggml_zdnn_q_گروہ_سائز(tensor);

        اگر (gs > 0) {
            // stack the weights into groups along ne0, each group of each row gets its own scale
            ثابت int64_t n_groups = (tensor->ne[0] + gs - 1) / gs;

            zdnn_init_pre_transformed_desc(
                ZDNN_3DS,
                INT8,
                &بفر->pre_tfm_desc,
                n_groups, gs, tensor->ne[1]
            );

            بفر->scales.resize(n_groups * tensor->ne[1]);
        } ورنہ {
            zdnn_init_pre_transformed_desc(
                ZDNN_2D,
                INT8,
                &بفر->pre_tfm_desc,
                tensor->ne[0], tensor->ne[1]
            );

            بفر->scales.resize(tensor->ne[1]);
        }

        بفر->channel_scales.resize(tensor->ne[0]);

        ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&بفر->pre_tfm_desc, QUANTIZED_WEIGHTS_INT8, &بفر->tfm_desc));
        ZDNN_CHECK(zdnn_init_quantized_ztensor_with_malloc(&بفر->pre_tfm_desc, &بفر->tfm_desc, 1.0f, 0.0f, &بفر->ztensor));
        واپس;
    }

    انتخاب (tensor->op) {
        حالت GGML_OP_MUL_MAT:
            {
                zdnn_init_pre_transformed_desc(
                    ZDNN_2D,
                    ggml_zdnn_type_mapping(tensor->type),
                    &بفر->pre_tfm_desc,
                    tensor->ne[1], tensor->ne[0]
                );
            } توڑو;

        طےشدہ:
            {
                // For 4D tensors, GGML uses NCHW layout. However, because zDNN
                // automatically transforms everything to NHWC, we will use it
                // directly to avoid the performance penalty changing the
                // layout and reshaping the tensor.
                zdnn_init_pre_transformed_desc(
                    ZDNN_NHWC,
                    ggml_zdnn_type_mapping(tensor->type),
                    &بفر->pre_tfm_desc,
                    tensor->ne[3], tensor->ne[2], tensor->ne[1], tensor->ne[0]
                );

                // TODO: Consider adding a ggml check.
                // TODO: If tensor = 4D, use ZDNN_NCHW by default.
                // TODO: If tensor = 2D, use ZDNN_NHWC by default.
            } توڑو;
    }

    ZDNN_CHECK(zdnn_generate_transformed_desc(&بفر->pre_tfm_desc, &بفر->tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&بفر->pre_tfm_desc, &بفر->tfm_desc, &بفر->ztensor));
}
