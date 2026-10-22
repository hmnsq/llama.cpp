#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "mmf.hpp"
#ಅನ್ತರ್ಭಾವ "utils.hpp"

ಶೂನ್ಯ ggml_zdnn_mul_mat_f(
    ಸ್ಥಿರಾಂಕ ggml_backend_zdnn_context * ctx,
    ಸ್ಥಿರಾಂಕ               ggml_tensor * src0,
    ಸ್ಥಿರಾಂಕ               ggml_tensor * src1,
                        ggml_tensor * dst) {
    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS;

    ಸ್ಥಿರಾಂಕ ಎಣಿಕೆ ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ಗಾತ್ರ(type));
    GGML_ASSERT(nb10 == ggml_type_ಗಾತ್ರ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ಗಾತ್ರ(ದಶಮಾಂಶ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ಸ್ಥಿರಾಂಕ ggml_tensor * weights = src0;
    ಸ್ಥಿರಾಂಕ ggml_tensor * inputs  = src1;
          ggml_tensor * output  = dst;

    ggml_backend_zdnn_ಬಫರ್ * weights_extra = (ggml_backend_zdnn_ಬಫರ್ *)weights->extra;
    ggml_backend_zdnn_ಬಫರ್ * inputs_extra  = (ggml_backend_zdnn_ಬಫರ್ *)inputs->extra;
    ggml_backend_zdnn_ಬಫರ್ * output_extra  = (ggml_backend_zdnn_ಬಫರ್ *)output->extra;
    ggml_backend_zdnn_ಬಫರ್ * bias_extra    = (ggml_backend_zdnn_ಬಫರ್ *)output_extra->extra;

    ಸ್ಥಿರಾಂಕ int64_t weights_ಅಡ್ಡಸಾಲುಗಳು = ne01;
    ಸ್ಥಿರಾಂಕ int64_t weights_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = ne00;
    ಸ್ಥಿರಾಂಕ int64_t inputs_ಅಡ್ಡಸಾಲುಗಳು  = ne11;
    ಸ್ಥಿರಾಂಕ int64_t inputs_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ  = ne10;

    ಪ್ರತಿಜ್ಞಾ(inputs_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ == weights_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);

    ಸ್ಥಿರಾಂಕ int64_t output_ಅಡ್ಡಸಾಲುಗಳು = ne1;
    ಸ್ಥಿರಾಂಕ int64_t output_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ = ne0;

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
    ಆದರೆ (!weights_extra->ztensor.is_transformed) {
        ggml_zdnn_ತುಂಬು_tensor(weights_extra->ztensor, weights->ದತ್ತಾಂಶ);
    }
    ಆದರೆ (!inputs_extra->ztensor.is_transformed) {
        ggml_zdnn_ತುಂಬು_tensor(inputs_extra->ztensor, inputs->ದತ್ತಾಂಶ);
    }

    ZDNN_CHECK(zdnn_matmul_transpose_op(&inputs_extra->ztensor, &weights_extra->ztensor, &bias_extra->ztensor,
                                        ಸುಳ್ಳು, ನಿಜ, MATMUL_OP_ADDITION, &output_extra->ztensor));
    // TODO: Remove in the future as we are currently DLF16 -> FP32 then in the next op, FP32 -> DLF16 again. Inefficient.
    ZDNN_CHECK(zdnn_transform_origtensor(&output_extra->ztensor, output->ದತ್ತಾಂಶ));

    GGML_ಬಳಸದ(ctx);
    GGML_ಬಳಸದ(weights_ಅಡ್ಡಸಾಲುಗಳು);
    GGML_ಬಳಸದ(weights_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
    GGML_ಬಳಸದ(inputs_ಅಡ್ಡಸಾಲುಗಳು);
    GGML_ಬಳಸದ(inputs_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
    GGML_ಬಳಸದ(output_ಅಡ್ಡಸಾಲುಗಳು);
    GGML_ಬಳಸದ(output_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ);
}
