#ضمن "ggml.h"
#ضمن "utils.hpp"

#ضمن <algorithm>
#ضمن <cmath>
#ضمن <cstdlib>

zdnn_داده_types ggml_zdnn_type_mapping(ggml_type type) {
    بدل (type) {
        حالة GGML_TYPE_F32:
            أرجع FP32;
        حالة GGML_TYPE_F16:
            أرجع FP16;
        حالة GGML_TYPE_BF16:
            أرجع BFLOAT;
        حالة GGML_TYPE_I8:
            أرجع INT8;
        حالة GGML_TYPE_I32:
            أرجع INT32;
        افتراضي:
            GGML_ABORT("%s: fatal: unable to determine zTensor data type",
                       __func__);
            اكسر;
    }
}

فراغ ggml_zdnn_create_tensor(zdnn_tensor_desc  & pre_tfm_desc,
                             zdnn_tensor_desc  & tfm_desc,
                             zdnn_ztensor      & ztensor,
                       ثابت ggml_tensor       * src,
                       ثابت int64_t           * ne,
                       ثابت zdnn_داده_layouts   layout) {
    zdnn_init_pre_transformed_desc(
        layout,
        ggml_zdnn_type_mapping(src->type),
        &pre_tfm_desc,
        ne[3], ne[2], ne[1], ne[0]
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_tfm_desc, &tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_tfm_desc, &tfm_desc, &ztensor));
}

فراغ ggml_zdnn_بارگیری_tensor(zdnn_ztensor & ztensor, فراغ * میانگیر) {
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor, میانگیر));
}

// size of the groups along ne0 that get their own scale, or 0 for one scale per row
// the groups are spread evenly over ne0 and rounded up to 128 values, the last group is zero padded
int64_t ggml_zdnn_q_گروه_اندازه(ثابت ggml_tensor * tensor) {
    // benchmarked on Llama-3.2-1B and Qwen2.5-1.5B Q8_0 with the channel scales:
    // 2048 is 15-18% faster than 1024, and 1024 lowers the KL divergence to the CPU by 13-19%
    // GGML_ZDNN_HIGH_PREC=1 selects 1024 for users that want the lower loss
    // always re-measure with different models to see if different sizes need a dispatching logic
    ساكن ثابت منطقي بالا_prec = std::getenv("GGML_ZDNN_HIGH_PREC") != اشاره_تهی &&
                                  std::atoi(std::getenv("GGML_ZDNN_HIGH_PREC")) != 0;
    ثابت int64_t گروه_اندازه_max = بالا_prec ? 1024 : 2048;

    ثابت int64_t ne0 = tensor->ne[0];
    إذا (ne0 <= گروه_اندازه_max) {
        أرجع 0;
    }

    ثابت int64_t n_groups = (ne0 + گروه_اندازه_max - 1) / گروه_اندازه_max;
    أرجع GGML_PAD((ne0 + n_groups - 1) / n_groups, 128);
}

فراغ ggml_zdnn_بارگیری_tensor_q(ggml_backend_zdnn_میانگیر * میانگیر, ثابت ggml_tensor * tensor) {
    ثابت int64_t ne0 = tensor->ne[0];
    ثابت int64_t ne1 = tensor->ne[1];

    ثابت ggml_مقصد_float_t مقصد_float = ggml_get_type_traits(tensor->type)->مقصد_float;
    GGML_ASSERT(مقصد_float != اشاره_تهی);

    // zDNN takes one scale per zTensor, so requantize each group of each row to int8 with its own scale
    // a group is the whole row, unless the weights are stacked into groups along ne0
    // rows are written transposed as zDNN computes inputs (ne11, ne0) x weights (ne0, ne1)
    ثابت منطقي    stacked  = میانگیر->pre_tfm_desc.layout == ZDNN_3DS;
    ثابت int64_t n_groups = stacked ? میانگیر->pre_tfm_desc.dim3 : 1;
    ثابت int64_t gs       = stacked ? میانگیر->pre_tfm_desc.dim2 : ne0;

    // benchmarked: 256 rows fill a 256-byte cache line per write, padding each row by a line avoids cache set conflicts
    constexpr int64_t n_tile   = 256;
    ثابت     int64_t x_stride = ne0 + 64;

    // the rows past ne0 of the last group stay zero as padding
    std::vector<int8_t> q(n_groups * gs * ne1);
    std::vector<عائم>  x(n_tile * x_stride);
    std::vector<عائم>  id(n_tile * n_groups);

    // a few input channels with larger weights than the rest would set the scale of every group,
    // so divide each input channel by its RMS over the rows and multiply the inputs by it instead
    // with groups of 1024, this lowered the KL divergence to the CPU by 10-18% on Llama-3.2-1B and Qwen2.5-1.5B Q8_0 at the same speed
    std::vector<مزدوج> جمع_sq(ne0, 0.0);
    لكل (int64_t i1 = 0; i1 < ne1; i1++) {
        مقصد_float((ثابت حرف *)tensor->داده + i1*tensor->nb[1], x.داده(), ne0);
        لكل (int64_t i0 = 0; i0 < ne0; i0++) {
            جمع_sq[i0] += (مزدوج)x[i0] * x[i0];
        }
    }

    std::vector<عائم> channel_id(ne0);
    لكل (int64_t i0 = 0; i0 < ne0; i0++) {
        ثابت عائم c = (عائم)std::جذر(جمع_sq[i0] / ne1);
        میانگیر->channel_scales[i0] = c ? c : 1.0f;
        channel_id[i0] = 1.0f / میانگیر->channel_scales[i0];
    }

    لكل (int64_t i1 = 0; i1 < ne1; i1 += n_tile) {
        ثابت int64_t nt = std::min(n_tile, ne1 - i1);

        لكل (int64_t t = 0; t < nt; t++) {
            عائم * xt = x.داده() + t*x_stride;
            مقصد_float((ثابت حرف *)tensor->داده + (i1 + t)*tensor->nb[1], xt, ne0);

            لكل (int64_t i0 = 0; i0 < ne0; i0++) {
                xt[i0] *= channel_id[i0];
            }

            لكل (int64_t g = 0; g < n_groups; g++) {
                عائم amax = 0.0f;
                لكل (int64_t i0 = g*gs; i0 < std::min((g + 1)*gs, ne0); i0++) {
                    amax = std::max(amax, fabsf(xt[i0]));
                }

                ثابت عائم d = amax / 127.0f;
                میانگیر->scales[g*ne1 + i1 + t] = d;
                id[t*n_groups + g] = d ? 1.0f/d : 0.0f;
            }
        }

        لكل (int64_t i0 = 0; i0 < ne0; i0++) {
            ثابت int64_t g = i0 / gs;
            int8_t * qi = q.داده() + i0*ne1 + i1;
            لكل (int64_t t = 0; t < nt; t++) {
                qi[t] = (int8_t)roundf(x[t*x_stride + i0] * id[t*n_groups + g]);
            }
        }
    }

    إذا (میانگیر->ztensor.is_transformed) {
        zdnn_reset_ztensor(&میانگیر->ztensor);
    }
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&میانگیر->ztensor, خطأ, -127, 127, q.داده()));
}

فراغ ggml_zdnn_init_tensor(ggml_backend_zdnn_میانگیر * میانگیر, ثابت ggml_tensor * tensor) {
    إذا (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(ggml_is_ماتریس(tensor));

        ثابت int64_t gs = ggml_zdnn_q_گروه_اندازه(tensor);

        إذا (gs > 0) {
            // stack the weights into groups along ne0, each group of each row gets its own scale
            ثابت int64_t n_groups = (tensor->ne[0] + gs - 1) / gs;

            zdnn_init_pre_transformed_desc(
                ZDNN_3DS,
                INT8,
                &میانگیر->pre_tfm_desc,
                n_groups, gs, tensor->ne[1]
            );

            میانگیر->scales.resize(n_groups * tensor->ne[1]);
        } وإلا {
            zdnn_init_pre_transformed_desc(
                ZDNN_2D,
                INT8,
                &میانگیر->pre_tfm_desc,
                tensor->ne[0], tensor->ne[1]
            );

            میانگیر->scales.resize(tensor->ne[1]);
        }

        میانگیر->channel_scales.resize(tensor->ne[0]);

        ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&میانگیر->pre_tfm_desc, QUANTIZED_WEIGHTS_INT8, &میانگیر->tfm_desc));
        ZDNN_CHECK(zdnn_init_quantized_ztensor_with_malloc(&میانگیر->pre_tfm_desc, &میانگیر->tfm_desc, 1.0f, 0.0f, &میانگیر->ztensor));
        أرجع;
    }

    بدل (tensor->op) {
        حالة GGML_OP_MUL_MAT:
            {
                zdnn_init_pre_transformed_desc(
                    ZDNN_2D,
                    ggml_zdnn_type_mapping(tensor->type),
                    &میانگیر->pre_tfm_desc,
                    tensor->ne[1], tensor->ne[0]
                );
            } اكسر;

        افتراضي:
            {
                // For 4D tensors, GGML uses NCHW layout. However, because zDNN
                // automatically transforms everything to NHWC, we will use it
                // directly to avoid the performance penalty changing the
                // layout and reshaping the tensor.
                zdnn_init_pre_transformed_desc(
                    ZDNN_NHWC,
                    ggml_zdnn_type_mapping(tensor->type),
                    &میانگیر->pre_tfm_desc,
                    tensor->ne[3], tensor->ne[2], tensor->ne[1], tensor->ne[0]
                );

                // TODO: Consider adding a ggml check.
                // TODO: If tensor = 4D, use ZDNN_NCHW by default.
                // TODO: If tensor = 2D, use ZDNN_NHWC by default.
            } اكسر;
    }

    ZDNN_CHECK(zdnn_generate_transformed_desc(&میانگیر->pre_tfm_desc, &میانگیر->tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&میانگیر->pre_tfm_desc, &میانگیر->tfm_desc, &میانگیر->ztensor));
}
