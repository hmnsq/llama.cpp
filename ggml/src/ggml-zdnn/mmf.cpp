#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "mmf.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "utils.hpp"

ਖਾਲੀ ggml_zdnn_mul_mat_f(
    ਅਚਲ ggml_backend_zdnn_context * ctx,
    ਅਚਲ               ggml_tensor * src0,
    ਅਚਲ               ggml_tensor * src1,
                        ggml_tensor * dst) {
    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS;

    ਅਚਲ ਗਿਣਤੀ ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ਮਾਪ(type));
    GGML_ASSERT(nb10 == ggml_type_ਮਾਪ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ਅਚਲ ggml_tensor * weights = src0;
    ਅਚਲ ggml_tensor * inputs  = src1;
          ggml_tensor * output  = dst;

    ggml_backend_zdnn_ਬਫਰ * weights_extra = (ggml_backend_zdnn_ਬਫਰ *)weights->extra;
    ggml_backend_zdnn_ਬਫਰ * inputs_extra  = (ggml_backend_zdnn_ਬਫਰ *)inputs->extra;
    ggml_backend_zdnn_ਬਫਰ * output_extra  = (ggml_backend_zdnn_ਬਫਰ *)output->extra;
    ggml_backend_zdnn_ਬਫਰ * bias_extra    = (ggml_backend_zdnn_ਬਫਰ *)output_extra->extra;

    ਅਚਲ int64_t weights_ਪੰਕਤੀਆਂ = ne01;
    ਅਚਲ int64_t weights_ਖਾਨੇ_ਗਿਣਤੀ = ne00;
    ਅਚਲ int64_t inputs_ਪੰਕਤੀਆਂ  = ne11;
    ਅਚਲ int64_t inputs_ਖਾਨੇ_ਗਿਣਤੀ  = ne10;

    ਪ੍ਰਤਿਜ੍ਞਾ(inputs_ਖਾਨੇ_ਗਿਣਤੀ == weights_ਖਾਨੇ_ਗਿਣਤੀ);

    ਅਚਲ int64_t output_ਪੰਕਤੀਆਂ = ne1;
    ਅਚਲ int64_t output_ਖਾਨੇ_ਗਿਣਤੀ = ne0;

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
    ਜੇ (!weights_extra->ztensor.is_transformed) {
        ggml_zdnn_ਲੱਦੋ_tensor(weights_extra->ztensor, weights->ਅੰਕੜੇ);
    }
    ਜੇ (!inputs_extra->ztensor.is_transformed) {
        ggml_zdnn_ਲੱਦੋ_tensor(inputs_extra->ztensor, inputs->ਅੰਕੜੇ);
    }

    ZDNN_CHECK(zdnn_matmul_transpose_op(&inputs_extra->ztensor, &weights_extra->ztensor, &bias_extra->ztensor,
                                        ਝੂਠ, ਸੱਚ, MATMUL_OP_ADDITION, &output_extra->ztensor));
    // TODO: Remove in the future as we are currently DLF16 -> FP32 then in the next op, FP32 -> DLF16 again. Inefficient.
    ZDNN_CHECK(zdnn_transform_origtensor(&output_extra->ztensor, output->ਅੰਕੜੇ));

    GGML_ਅਣਵਰਤਿਆ(ctx);
    GGML_ਅਣਵਰਤਿਆ(weights_ਪੰਕਤੀਆਂ);
    GGML_ਅਣਵਰਤਿਆ(weights_ਖਾਨੇ_ਗਿਣਤੀ);
    GGML_ਅਣਵਰਤਿਆ(inputs_ਪੰਕਤੀਆਂ);
    GGML_ਅਣਵਰਤਿਆ(inputs_ਖਾਨੇ_ਗਿਣਤੀ);
    GGML_ਅਣਵਰਤਿਆ(output_ਪੰਕਤੀਆਂ);
    GGML_ਅਣਵਰਤਿਆ(output_ਖਾਨੇ_ਗਿਣਤੀ);
}
