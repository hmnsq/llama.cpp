#అన్తర్భావ "ggml.h"
#అన్తర్భావ "mmf.hpp"
#అన్తర్భావ "utils.hpp"

శూన్య ggml_zdnn_mul_mat_f(
    స్థిరరాశి ggml_backend_zdnn_context * ctx,
    స్థిరరాశి               ggml_tensor * src0,
    స్థిరరాశి               ggml_tensor * src1,
                        ggml_tensor * dst) {
    GGML_TENSOR_ద్విమాన_OP_LOCALS;

    స్థిరరాశి గణన ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_కొలత(type));
    GGML_ASSERT(nb10 == ggml_type_కొలత(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    స్థిరరాశి ggml_tensor * weights = src0;
    స్థిరరాశి ggml_tensor * inputs  = src1;
          ggml_tensor * output  = dst;

    ggml_backend_zdnn_బఫరు * weights_extra = (ggml_backend_zdnn_బఫరు *)weights->extra;
    ggml_backend_zdnn_బఫరు * inputs_extra  = (ggml_backend_zdnn_బఫరు *)inputs->extra;
    ggml_backend_zdnn_బఫరు * output_extra  = (ggml_backend_zdnn_బఫరు *)output->extra;
    ggml_backend_zdnn_బఫరు * bias_extra    = (ggml_backend_zdnn_బఫరు *)output_extra->extra;

    స్థిరరాశి int64_t weights_అడ్డువరుసలు = ne01;
    స్థిరరాశి int64_t weights_నిలువుల_సంఖ్య = ne00;
    స్థిరరాశి int64_t inputs_అడ్డువరుసలు  = ne11;
    స్థిరరాశి int64_t inputs_నిలువుల_సంఖ్య  = ne10;

    ప్రతిజ్ఞా(inputs_నిలువుల_సంఖ్య == weights_నిలువుల_సంఖ్య);

    స్థిరరాశి int64_t output_అడ్డువరుసలు = ne1;
    స్థిరరాశి int64_t output_నిలువుల_సంఖ్య = ne0;

    // GGML_LOG_INFO("%s: tensor '%s' tensor dimensions: [%ld, %ld, %ld, %ld] pre_tfm_desc dimensions: [%ld, %ld, %ld, %ld]\n",
    //               __func__, weights_extra->name,
    //               weights->ne[3], weights->ne[2], weights->ne[1], weights->ne[0],
    //               weights_extra->pre_tfm_desc.dim1,
    //               weights_extra->pre_tfm_desc.dim2,
    //               weights_extra->pre_tfm_desc.dim3,
    //               weights_extra->pre_tfm_desc.dim4);

    // GGML_LOG_INFO("%s: tensor '%s' tensor dimensions: [%ld, %ld, %ld, %ld] pre_tfm_desc dimensions: [%ld, %ld, %ld, %ld]\n",
    //               __func__, inputs_extra->name,
    //               inputs->ne[3], inputs->ne[2], inputs->ne[1], inputs->ne[0],
    //               inputs_extra->pre_tfm_desc.dim1,
    //               inputs_extra->pre_tfm_desc.dim2,
    //               inputs_extra->pre_tfm_desc.dim3,
    //               inputs_extra->pre_tfm_desc.dim4);

    GGML_ASSERT(weights_extra->pre_tfm_desc.dim1 == weights->ne[0] && "weights_extra->pre_tfm_desc.dim1 must match weights->ne[0]");
    GGML_ASSERT(weights_extra->pre_tfm_desc.dim2 == weights->ne[1] && "weights_extra->pre_tfm_desc.dim2 must match weights->ne[1]");
    GGML_ASSERT(inputs_extra->pre_tfm_desc.dim1  == inputs->ne[0]  && "inputs_extra->pre_tfm_desc.dim1 must match inputs->ne[0]");
    GGML_ASSERT(inputs_extra->pre_tfm_desc.dim2  == inputs->ne[1]  && "inputs_extra->pre_tfm_desc.dim2 must match inputs->ne[1]");

    // compute tensors are transformed on first use
    ఒకవేళ (!weights_extra->ztensor.is_transformed) {
        ggml_zdnn_ఎక్కించు_tensor(weights_extra->ztensor, weights->దత్తాంశం);
    }
    ఒకవేళ (!inputs_extra->ztensor.is_transformed) {
        ggml_zdnn_ఎక్కించు_tensor(inputs_extra->ztensor, inputs->దత్తాంశం);
    }

    ZDNN_CHECK(zdnn_matmul_transpose_op(&inputs_extra->ztensor, &weights_extra->ztensor, &bias_extra->ztensor,
                                        అబద్ధం, నిజం, MATMUL_OP_ADDITION, &output_extra->ztensor));
    // TODO: Remove in the future as we are currently DLF16 -> FP32 then in the next op, FP32 -> DLF16 again. Inefficient.
    ZDNN_CHECK(zdnn_transform_origtensor(&output_extra->ztensor, output->దత్తాంశం));

    GGML_వాడని(ctx);
    GGML_వాడని(weights_అడ్డువరుసలు);
    GGML_వాడని(weights_నిలువుల_సంఖ్య);
    GGML_వాడని(inputs_అడ్డువరుసలు);
    GGML_వాడని(inputs_నిలువుల_సంఖ్య);
    GGML_వాడని(output_అడ్డువరుసలు);
    GGML_వాడని(output_నిలువుల_సంఖ్య);
}
