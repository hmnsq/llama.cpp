#అన్తర్భావ "ggml.h"
#అన్తర్భావ "utils.hpp"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cmath>
#అన్తర్భావ <cstdlib>

zdnn_దత్తాంశం_types ggml_zdnn_type_mapping(ggml_type type) {
    ఎంపిక (type) {
        స్థితి GGML_TYPE_F32:
            తిరిగి FP32;
        స్థితి GGML_TYPE_F16:
            తిరిగి FP16;
        స్థితి GGML_TYPE_BF16:
            తిరిగి BFLOAT;
        స్థితి GGML_TYPE_I8:
            తిరిగి INT8;
        స్థితి GGML_TYPE_I32:
            తిరిగి INT32;
        అప్రమేయ:
            GGML_ABORT("%s: fatal: unable to determine zTensor data type",
                       __func__);
            ఆపు;
    }
}

శూన్య ggml_zdnn_create_tensor(zdnn_tensor_desc  & pre_tfm_desc,
                             zdnn_tensor_desc  & tfm_desc,
                             zdnn_ztensor      & ztensor,
                       స్థిరరాశి ggml_tensor       * src,
                       స్థిరరాశి int64_t           * ne,
                       స్థిరరాశి zdnn_దత్తాంశం_layouts   layout) {
    zdnn_init_pre_transformed_desc(
        layout,
        ggml_zdnn_type_mapping(src->type),
        &pre_tfm_desc,
        ne[3], ne[2], ne[1], ne[0]
    );

    ZDNN_CHECK(zdnn_generate_transformed_desc(&pre_tfm_desc, &tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&pre_tfm_desc, &tfm_desc, &ztensor));
}

శూన్య ggml_zdnn_ఎక్కించు_tensor(zdnn_ztensor & ztensor, శూన్య * బఫరు) {
    ZDNN_CHECK(zdnn_transform_ztensor(&ztensor, బఫరు));
}

// size of the groups along ne0 that get their own scale, or 0 for one scale per row
// the groups are spread evenly over ne0 and rounded up to 128 values, the last group is zero padded
int64_t ggml_zdnn_q_సమూహం_కొలత(స్థిరరాశి ggml_tensor * tensor) {
    // benchmarked on Llama-3.2-1B and Qwen2.5-1.5B Q8_0 with the channel scales:
    // 2048 is 15-18% faster than 1024, and 1024 lowers the KL divergence to the CPU by 13-19%
    // GGML_ZDNN_HIGH_PREC=1 selects 1024 for users that want the lower loss
    // always re-measure with different models to see if different sizes need a dispatching logic
    స్థిర స్థిరరాశి తార్కిక ఎగువ_prec = std::getenv("GGML_ZDNN_HIGH_PREC") != శూన్యసూచిక &&
                                  std::atoi(std::getenv("GGML_ZDNN_HIGH_PREC")) != 0;
    స్థిరరాశి int64_t సమూహం_కొలత_max = ఎగువ_prec ? 1024 : 2048;

    స్థిరరాశి int64_t ne0 = tensor->ne[0];
    ఒకవేళ (ne0 <= సమూహం_కొలత_max) {
        తిరిగి 0;
    }

    స్థిరరాశి int64_t n_groups = (ne0 + సమూహం_కొలత_max - 1) / సమూహం_కొలత_max;
    తిరిగి GGML_PAD((ne0 + n_groups - 1) / n_groups, 128);
}

శూన్య ggml_zdnn_ఎక్కించు_tensor_q(ggml_backend_zdnn_బఫరు * బఫరు, స్థిరరాశి ggml_tensor * tensor) {
    స్థిరరాశి int64_t ne0 = tensor->ne[0];
    స్థిరరాశి int64_t ne1 = tensor->ne[1];

    స్థిరరాశి ggml_గమ్యం_float_t గమ్యం_float = ggml_get_type_traits(tensor->type)->గమ్యం_float;
    GGML_ASSERT(గమ్యం_float != శూన్యసూచిక);

    // zDNN takes one scale per zTensor, so requantize each group of each row to int8 with its own scale
    // a group is the whole row, unless the weights are stacked into groups along ne0
    // rows are written transposed as zDNN computes inputs (ne11, ne0) x weights (ne0, ne1)
    స్థిరరాశి తార్కిక    stacked  = బఫరు->pre_tfm_desc.layout == ZDNN_3DS;
    స్థిరరాశి int64_t n_groups = stacked ? బఫరు->pre_tfm_desc.dim3 : 1;
    స్థిరరాశి int64_t gs       = stacked ? బఫరు->pre_tfm_desc.dim2 : ne0;

    // benchmarked: 256 rows fill a 256-byte cache line per write, padding each row by a line avoids cache set conflicts
    constexpr int64_t n_tile   = 256;
    స్థిరరాశి     int64_t x_stride = ne0 + 64;

    // the rows past ne0 of the last group stay zero as padding
    std::vector<int8_t> q(n_groups * gs * ne1);
    std::vector<దశాంశం>  x(n_tile * x_stride);
    std::vector<దశాంశం>  id(n_tile * n_groups);

    // a few input channels with larger weights than the rest would set the scale of every group,
    // so divide each input channel by its RMS over the rows and multiply the inputs by it instead
    // with groups of 1024, this lowered the KL divergence to the CPU by 10-18% on Llama-3.2-1B and Qwen2.5-1.5B Q8_0 at the same speed
    std::vector<ద్విగుణ> కూడిక_sq(ne0, 0.0);
    కొరకు (int64_t i1 = 0; i1 < ne1; i1++) {
        గమ్యం_float((స్థిరరాశి అక్షరం *)tensor->దత్తాంశం + i1*tensor->nb[1], x.దత్తాంశం(), ne0);
        కొరకు (int64_t i0 = 0; i0 < ne0; i0++) {
            కూడిక_sq[i0] += (ద్విగుణ)x[i0] * x[i0];
        }
    }

    std::vector<దశాంశం> channel_id(ne0);
    కొరకు (int64_t i0 = 0; i0 < ne0; i0++) {
        స్థిరరాశి దశాంశం c = (దశాంశం)std::వర్గమూలం(కూడిక_sq[i0] / ne1);
        బఫరు->channel_scales[i0] = c ? c : 1.0f;
        channel_id[i0] = 1.0f / బఫరు->channel_scales[i0];
    }

    కొరకు (int64_t i1 = 0; i1 < ne1; i1 += n_tile) {
        స్థిరరాశి int64_t nt = std::min(n_tile, ne1 - i1);

        కొరకు (int64_t t = 0; t < nt; t++) {
            దశాంశం * xt = x.దత్తాంశం() + t*x_stride;
            గమ్యం_float((స్థిరరాశి అక్షరం *)tensor->దత్తాంశం + (i1 + t)*tensor->nb[1], xt, ne0);

            కొరకు (int64_t i0 = 0; i0 < ne0; i0++) {
                xt[i0] *= channel_id[i0];
            }

            కొరకు (int64_t g = 0; g < n_groups; g++) {
                దశాంశం amax = 0.0f;
                కొరకు (int64_t i0 = g*gs; i0 < std::min((g + 1)*gs, ne0); i0++) {
                    amax = std::max(amax, fabsf(xt[i0]));
                }

                స్థిరరాశి దశాంశం d = amax / 127.0f;
                బఫరు->scales[g*ne1 + i1 + t] = d;
                id[t*n_groups + g] = d ? 1.0f/d : 0.0f;
            }
        }

        కొరకు (int64_t i0 = 0; i0 < ne0; i0++) {
            స్థిరరాశి int64_t g = i0 / gs;
            int8_t * qi = q.దత్తాంశం() + i0*ne1 + i1;
            కొరకు (int64_t t = 0; t < nt; t++) {
                qi[t] = (int8_t)roundf(x[t*x_stride + i0] * id[t*n_groups + g]);
            }
        }
    }

    ఒకవేళ (బఫరు->ztensor.is_transformed) {
        zdnn_reset_ztensor(&బఫరు->ztensor);
    }
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&బఫరు->ztensor, అబద్ధం, -127, 127, q.దత్తాంశం()));
}

శూన్య ggml_zdnn_init_tensor(ggml_backend_zdnn_బఫరు * బఫరు, స్థిరరాశి ggml_tensor * tensor) {
    ఒకవేళ (ggml_is_quantized(tensor->type)) {
        GGML_ASSERT(ggml_is_మాతృక(tensor));

        స్థిరరాశి int64_t gs = ggml_zdnn_q_సమూహం_కొలత(tensor);

        ఒకవేళ (gs > 0) {
            // stack the weights into groups along ne0, each group of each row gets its own scale
            స్థిరరాశి int64_t n_groups = (tensor->ne[0] + gs - 1) / gs;

            zdnn_init_pre_transformed_desc(
                ZDNN_3DS,
                INT8,
                &బఫరు->pre_tfm_desc,
                n_groups, gs, tensor->ne[1]
            );

            బఫరు->scales.resize(n_groups * tensor->ne[1]);
        } లేకపోతే {
            zdnn_init_pre_transformed_desc(
                ZDNN_2D,
                INT8,
                &బఫరు->pre_tfm_desc,
                tensor->ne[0], tensor->ne[1]
            );

            బఫరు->scales.resize(tensor->ne[1]);
        }

        బఫరు->channel_scales.resize(tensor->ne[0]);

        ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&బఫరు->pre_tfm_desc, QUANTIZED_WEIGHTS_INT8, &బఫరు->tfm_desc));
        ZDNN_CHECK(zdnn_init_quantized_ztensor_with_malloc(&బఫరు->pre_tfm_desc, &బఫరు->tfm_desc, 1.0f, 0.0f, &బఫరు->ztensor));
        తిరిగి;
    }

    ఎంపిక (tensor->op) {
        స్థితి GGML_OP_MUL_MAT:
            {
                zdnn_init_pre_transformed_desc(
                    ZDNN_2D,
                    ggml_zdnn_type_mapping(tensor->type),
                    &బఫరు->pre_tfm_desc,
                    tensor->ne[1], tensor->ne[0]
                );
            } ఆపు;

        అప్రమేయ:
            {
                // For 4D tensors, GGML uses NCHW layout. However, because zDNN
                // automatically transforms everything to NHWC, we will use it
                // directly to avoid the performance penalty changing the
                // layout and reshaping the tensor.
                zdnn_init_pre_transformed_desc(
                    ZDNN_NHWC,
                    ggml_zdnn_type_mapping(tensor->type),
                    &బఫరు->pre_tfm_desc,
                    tensor->ne[3], tensor->ne[2], tensor->ne[1], tensor->ne[0]
                );

                // TODO: Consider adding a ggml check.
                // TODO: If tensor = 4D, use ZDNN_NCHW by default.
                // TODO: If tensor = 2D, use ZDNN_NHWC by default.
            } ఆపు;
    }

    ZDNN_CHECK(zdnn_generate_transformed_desc(&బఫరు->pre_tfm_desc, &బఫరు->tfm_desc));
    ZDNN_CHECK(zdnn_init_ztensor_with_malloc(&బఫరు->pre_tfm_desc, &బఫరు->tfm_desc, &బఫరు->ztensor));
}
