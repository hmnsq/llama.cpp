#অন্তর্ভাব "ggml.h"
#অন্তর্ভাব "mmf.hpp"
#অন্তর্ভাব "utils.hpp"

শূন্য ggml_zdnn_mul_mat_f(
    ধ্রুবক ggml_backend_zdnn_context * ctx,
    ধ্রুবক               ggml_tensor * src0,
    ধ্রুবক               ggml_tensor * src1,
                        ggml_tensor * dst) {
    GGML_TENSOR_দ্বিমিক_OP_LOCALS;

    ধ্রুবক গণনা ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_আকার(type));
    GGML_ASSERT(nb10 == ggml_type_আকার(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == আকার(দশমিক));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ধ্রুবক ggml_tensor * weights = src0;
    ধ্রুবক ggml_tensor * inputs  = src1;
          ggml_tensor * output  = dst;

    ggml_backend_zdnn_বাফার * weights_extra = (ggml_backend_zdnn_বাফার *)weights->extra;
    ggml_backend_zdnn_বাফার * inputs_extra  = (ggml_backend_zdnn_বাফার *)inputs->extra;
    ggml_backend_zdnn_বাফার * output_extra  = (ggml_backend_zdnn_বাফার *)output->extra;
    ggml_backend_zdnn_বাফার * bias_extra    = (ggml_backend_zdnn_বাফার *)output_extra->extra;

    ধ্রুবক int64_t weights_rows = ne01;
    ধ্রুবক int64_t weights_cols = ne00;
    ধ্রুবক int64_t inputs_rows  = ne11;
    ধ্রুবক int64_t inputs_cols  = ne10;

    প্রতিজ্ঞা(inputs_cols == weights_cols);

    ধ্রুবক int64_t output_rows = ne1;
    ধ্রুবক int64_t output_cols = ne0;

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
    যদি (!weights_extra->ztensor.is_transformed) {
        ggml_zdnn_load_tensor(weights_extra->ztensor, weights->তথ্য);
    }
    যদি (!inputs_extra->ztensor.is_transformed) {
        ggml_zdnn_load_tensor(inputs_extra->ztensor, inputs->তথ্য);
    }

    ZDNN_CHECK(zdnn_matmul_transpose_op(&inputs_extra->ztensor, &weights_extra->ztensor, &bias_extra->ztensor,
                                        মিথ্যা, সত্য, MATMUL_OP_ADDITION, &output_extra->ztensor));
    // TODO: Remove in the future as we are currently DLF16 -> FP32 then in the next op, FP32 -> DLF16 again. Inefficient.
    ZDNN_CHECK(zdnn_transform_origtensor(&output_extra->ztensor, output->তথ্য));

    GGML_UNUSED(ctx);
    GGML_UNUSED(weights_rows);
    GGML_UNUSED(weights_cols);
    GGML_UNUSED(inputs_rows);
    GGML_UNUSED(inputs_cols);
    GGML_UNUSED(output_rows);
    GGML_UNUSED(output_cols);
}
