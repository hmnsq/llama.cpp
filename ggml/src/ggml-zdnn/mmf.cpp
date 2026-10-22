#अन्तर्भाव "ggml.h"
#अन्तर्भाव "mmf.hpp"
#अन्तर्भाव "utils.hpp"

शून्य ggml_zdnn_mul_mat_f(
    ध्रुव ggml_backend_zdnn_context * ctx,
    ध्रुव               ggml_tensor * src0,
    ध्रुव               ggml_tensor * src1,
                        ggml_tensor * dst) {
    GGML_TENSOR_द्विमान_OP_LOCALS;

    ध्रुव गणना ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_आकार(type));
    GGML_ASSERT(nb10 == ggml_type_आकार(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == परिमाण(भिन्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ध्रुव ggml_tensor * weights = src0;
    ध्रुव ggml_tensor * inputs  = src1;
          ggml_tensor * output  = dst;

    ggml_backend_zdnn_बफर * weights_extra = (ggml_backend_zdnn_बफर *)weights->extra;
    ggml_backend_zdnn_बफर * inputs_extra  = (ggml_backend_zdnn_बफर *)inputs->extra;
    ggml_backend_zdnn_बफर * output_extra  = (ggml_backend_zdnn_बफर *)output->extra;
    ggml_backend_zdnn_बफर * bias_extra    = (ggml_backend_zdnn_बफर *)output_extra->extra;

    ध्रुव int64_t weights_पङ्क्तिकाः = ne01;
    ध्रुव int64_t weights_स्तम्भसङ्ख्या = ne00;
    ध्रुव int64_t inputs_पङ्क्तिकाः  = ne11;
    ध्रुव int64_t inputs_स्तम्भसङ्ख्या  = ne10;

    प्रतिज्ञा(inputs_स्तम्भसङ्ख्या == weights_स्तम्भसङ्ख्या);

    ध्रुव int64_t output_पङ्क्तिकाः = ne1;
    ध्रुव int64_t output_स्तम्भसङ्ख्या = ne0;

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
    यदि (!weights_extra->ztensor.is_transformed) {
        ggml_zdnn_आरोपय_tensor(weights_extra->ztensor, weights->दत्त);
    }
    यदि (!inputs_extra->ztensor.is_transformed) {
        ggml_zdnn_आरोपय_tensor(inputs_extra->ztensor, inputs->दत्त);
    }

    ZDNN_CHECK(zdnn_matmul_transpose_op(&inputs_extra->ztensor, &weights_extra->ztensor, &bias_extra->ztensor,
                                        असत्य, सत्य, MATMUL_OP_ADDITION, &output_extra->ztensor));
    // TODO: Remove in the future as we are currently DLF16 -> FP32 then in the next op, FP32 -> DLF16 again. Inefficient.
    ZDNN_CHECK(zdnn_transform_origtensor(&output_extra->ztensor, output->दत्त));

    GGML_अप्रयुक्तम्(ctx);
    GGML_अप्रयुक्तम्(weights_पङ्क्तिकाः);
    GGML_अप्रयुक्तम्(weights_स्तम्भसङ्ख्या);
    GGML_अप्रयुक्तम्(inputs_पङ्क्तिकाः);
    GGML_अप्रयुक्तम्(inputs_स्तम्भसङ्ख्या);
    GGML_अप्रयुक्तम्(output_पङ्क्तिकाः);
    GGML_अप्रयुक्तम्(output_स्तम्भसङ्ख्या);
}
