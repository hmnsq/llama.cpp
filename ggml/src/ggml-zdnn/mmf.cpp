#ضمن "ggml.h"
#ضمن "mmf.hpp"
#ضمن "utils.hpp"

فراغ ggml_zdnn_mul_mat_f(
    ثابت ggml_backend_zdnn_context * ctx,
    ثابت               ggml_tensor * src0,
    ثابت               ggml_tensor * src1,
                        ggml_tensor * dst) {
    GGML_TENSOR_ثنائی_OP_LOCALS;

    ثابت تعداد ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_ناپ(type));
    GGML_ASSERT(nb10 == ggml_type_ناپ(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ثابت ggml_tensor * weights = src0;
    ثابت ggml_tensor * inputs  = src1;
          ggml_tensor * output  = dst;

    ggml_backend_zdnn_بفر * weights_extra = (ggml_backend_zdnn_بفر *)weights->extra;
    ggml_backend_zdnn_بفر * inputs_extra  = (ggml_backend_zdnn_بفر *)inputs->extra;
    ggml_backend_zdnn_بفر * output_extra  = (ggml_backend_zdnn_بفر *)output->extra;
    ggml_backend_zdnn_بفر * bias_extra    = (ggml_backend_zdnn_بفر *)output_extra->extra;

    ثابت int64_t weights_صفاں = ne01;
    ثابت int64_t weights_خانے_گنتی = ne00;
    ثابت int64_t inputs_صفاں  = ne11;
    ثابت int64_t inputs_خانے_گنتی  = ne10;

    أكد(inputs_خانے_گنتی == weights_خانے_گنتی);

    ثابت int64_t output_صفاں = ne1;
    ثابت int64_t output_خانے_گنتی = ne0;

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
    إذا (!weights_extra->ztensor.is_transformed) {
        ggml_zdnn_لدو_tensor(weights_extra->ztensor, weights->انکڑے);
    }
    إذا (!inputs_extra->ztensor.is_transformed) {
        ggml_zdnn_لدو_tensor(inputs_extra->ztensor, inputs->انکڑے);
    }

    ZDNN_CHECK(zdnn_matmul_transpose_op(&inputs_extra->ztensor, &weights_extra->ztensor, &bias_extra->ztensor,
                                        خطأ, سچ, MATMUL_OP_ADDITION, &output_extra->ztensor));
    // TODO: Remove in the future as we are currently DLF16 -> FP32 then in the next op, FP32 -> DLF16 again. Inefficient.
    ZDNN_CHECK(zdnn_transform_origtensor(&output_extra->ztensor, output->انکڑے));

    GGML_بن_ورتیا(ctx);
    GGML_بن_ورتیا(weights_صفاں);
    GGML_بن_ورتیا(weights_خانے_گنتی);
    GGML_بن_ورتیا(inputs_صفاں);
    GGML_بن_ورتیا(inputs_خانے_گنتی);
    GGML_بن_ورتیا(output_صفاں);
    GGML_بن_ورتیا(output_خانے_گنتی);
}
