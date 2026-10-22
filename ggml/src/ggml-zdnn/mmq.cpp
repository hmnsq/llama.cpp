#समावेश "ggml.h"
#समावेश "mmq.hpp"
#समावेश "quantize.hpp"

#समावेश <algorithm>
#समावेश <future>
#समावेश <vector>

// the weights are stacked into groups along ne00, and each group of each row of the inputs and weights has its own scale
// zDNN multiplies the groups in stacked calls, then the partial results of each group are scaled and summed here
अटल व्योम ggml_zdnn_mul_mat_q_grouped(ggml_backend_zdnn_context * ctx,
                                        int32_t                     n_threads,
                                  स्थिर ggml_tensor               * inputs,
                                  स्थिर ggml_backend_zdnn_बफर  * weights_extra,
                                        ggml_tensor               * output) {
    स्थिर int64_t ne10 = inputs->ne[0];
    स्थिर int64_t ne11 = inputs->ne[1];
    स्थिर int64_t ne0  = output->ne[0];

    स्थिर int64_t n_groups = weights_extra->pre_tfm_desc.dim3;
    स्थिर int64_t gs       = weights_extra->pre_tfm_desc.dim2;

    // the last group is zero padded when gs does not divide ne10
    GGML_ASSERT(n_groups * gs >= ne10 && (n_groups - 1) * gs < ne10);

    // the rounding errors of the inputs are stacked after the inputs, see ggml_zdnn_quantize_inputs
    ggml_zdnn_pool_alloc<int8_t> inputs_q_alloc(ctx->pool, 2 * n_groups * gs * ne11);
    int8_t * inputs_q = inputs_q_alloc.लो();
    std::vector<भग्न> inputs_scales(n_groups * ne11);
    ggml_zdnn_quantize_inputs(n_threads, inputs, gs, weights_extra->channel_scales.तथ्य(),
                            inputs_q, inputs_q + n_groups * gs * ne11, inputs_scales.तथ्य());

    zdnn_tensor_desc              inputs_pre_tfm_desc;
    zdnn_tensor_desc              inputs_tfm_desc;
    zdnn_ztensor                  inputs_ztensor;
    ggml_zdnn_pool_alloc<uint8_t> inputs_ztensor_alloc;

    zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &inputs_pre_tfm_desc, 2 * n_groups, ne11, gs);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&inputs_pre_tfm_desc, QUANTIZED_INT8, &inputs_tfm_desc));
    ggml_zdnn_init_scratch_ztensor(ctx, &inputs_pre_tfm_desc, &inputs_tfm_desc, &inputs_ztensor, inputs_ztensor_alloc);
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&inputs_ztensor, false, -127, 127, inputs_q));

    // each group of the partial results is read back as its own 2D ztensor
    zdnn_tensor_desc group_pre_tfm_desc;
    zdnn_tensor_desc group_tfm_desc;

    zdnn_init_pre_transformed_desc(ZDNN_2D, FP32, &group_pre_tfm_desc, ne11, ne0);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&group_pre_tfm_desc, &group_tfm_desc));

    स्थिर uint64_t group_bytes = zdnn_getsize_ztensor(&group_tfm_desc);

    // stacked calls with a large output are much slower per group, so run the groups in chunks
    // 8 MiB per chunk was benchmarked to be the best on Llama-3.2-1B Q8_0
    // always re-measure with different models to see if different sizes need a dispatching logic
    constexpr int64_t chunk_bytes = 8*1024*1024;

    स्थिर int64_t n_chunk = std::clamp<int64_t>(chunk_bytes / (int64_t)group_bytes, 1, n_groups);

    // all zeros, so it is already the pre-computed bias for every chunk
    स्थिर zdnn_ztensor & bias_ztensor = ggml_zdnn_zero_bias(ctx, n_chunk, ne0);

    // two partial buffers, so the next chunk runs on the NNPA while this one is summed
    // each holds the results of a chunk of inputs, followed by the results of their rounding errors
    zdnn_tensor_desc              partial_pre_tfm_desc;
    zdnn_tensor_desc              partial_tfm_desc;
    zdnn_ztensor                  partial_ztensor[2];
    ggml_zdnn_pool_alloc<uint8_t> partial_ztensor_alloc[2];

    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &partial_pre_tfm_desc, 2 * n_chunk, ne11, ne0);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&partial_pre_tfm_desc, QUANTIZED_DLFLOAT16, &partial_tfm_desc));
    GGML_ASSERT(zdnn_getsize_ztensor(&partial_tfm_desc) == (uint64_t)(2 * n_chunk) * group_bytes);
    क्रम (int32_t i = 0; i < 2; i++) {
        ggml_zdnn_init_scratch_ztensor(ctx, &partial_pre_tfm_desc, &partial_tfm_desc, &partial_ztensor[i], partial_ztensor_alloc[i]);
    }

    स्थिर भग्न * weights_scales = weights_extra->scales.तथ्य();
    ggml_zdnn_pool_alloc<भग्न> partial_alloc(ctx->pool, 2 * ne11 * ne0);
    भग्न * partial     = partial_alloc.लो();
    भग्न * partial_res = partial + ne11 * ne0;

    स्थिर स्वतः योग_chunk = [&](स्थिर zdnn_ztensor & chunk, int64_t g0, int64_t n) {
        zdnn_ztensor group_ztensor;
        zdnn_init_ztensor(&group_pre_tfm_desc, &group_tfm_desc, &group_ztensor);

        क्रम (int64_t g = g0; g < g0 + n; g++) {
            group_ztensor.बफर_आकार    = group_bytes;
            group_ztensor.is_transformed = true;

            group_ztensor.बफर = (अक्षर *)chunk.बफर + (g - g0)*group_bytes;
            ZDNN_CHECK(zdnn_transform_origtensor(&group_ztensor, partial));

            group_ztensor.बफर = (अक्षर *)chunk.बफर + (n_chunk + g - g0)*group_bytes;
            ZDNN_CHECK(zdnn_transform_origtensor(&group_ztensor, partial_res));

            स्थिर भग्न * sw = weights_scales + g*ne0;
            क्रम (int64_t i1 = 0; i1 < ne11; i1++) {
                स्थिर भग्न   sa = inputs_scales[g*ne11 + i1];
                स्थिर भग्न * p  = partial     + i1*ne0;
                स्थिर भग्न * pr = partial_res + i1*ne0;
                भग्न       * y  = (भग्न *)output->तथ्य + i1*ne0;

                अगर (g == 0) {
                    क्रम (int64_t i0 = 0; i0 < ne0; i0++) {
                        y[i0] = sa * sw[i0] * (p[i0] + pr[i0] * (1.0f/254.0f));
                    }
                } अन्यथा {
                    क्रम (int64_t i0 = 0; i0 < ne0; i0++) {
                        y[i0] += sa * sw[i0] * (p[i0] + pr[i0] * (1.0f/254.0f));
                    }
                }
            }
        }
    };

    zdnn_tensor_desc view_pre_tfm_desc[6];
    zdnn_tensor_desc view_tfm_desc[6];
    std::future<व्योम> योग_task;

    क्रम (int64_t c = 0, g0 = 0; g0 < n_groups; c++, g0 += n_chunk) {
        स्थिर int64_t n = std::min(n_chunk, n_groups - g0);

        स्थिर zdnn_ztensor inputs_view     = ggml_zdnn_ढेर_view(&view_pre_tfm_desc[0], &view_tfm_desc[0], inputs_ztensor,         2 * n_groups, g0,            n);
        स्थिर zdnn_ztensor inputs_res_view = ggml_zdnn_ढेर_view(&view_pre_tfm_desc[1], &view_tfm_desc[1], inputs_ztensor,         2 * n_groups, n_groups + g0, n);
        स्थिर zdnn_ztensor weights_view    = ggml_zdnn_ढेर_view(&view_pre_tfm_desc[2], &view_tfm_desc[2], weights_extra->ztensor, n_groups,     g0,            n);
        स्थिर zdnn_ztensor bias_view       = ggml_zdnn_ढेर_view(&view_pre_tfm_desc[3], &view_tfm_desc[3], bias_ztensor,           n_chunk,      0,             n);

        // the partial buffer of chunk c - 2 is summed by now, as sum_task is waited on before starting the next sum
        zdnn_ztensor & chunk = partial_ztensor[c % 2];
        chunk.pre_transformed_desc = &partial_pre_tfm_desc;
        chunk.transformed_desc     = &partial_tfm_desc;
        chunk.is_transformed       = false;

        zdnn_ztensor chunk_view     = ggml_zdnn_ढेर_view(&view_pre_tfm_desc[4], &view_tfm_desc[4], chunk, 2 * n_chunk, 0,       n);
        zdnn_ztensor chunk_res_view = ggml_zdnn_ढेर_view(&view_pre_tfm_desc[5], &view_tfm_desc[5], chunk, 2 * n_chunk, n_chunk, n);
        ZDNN_CHECK(zdnn_quantized_matmul_op(&inputs_view, &weights_view, &bias_view,
                                            MATMUL_OP_ADDITION, -127, 127, true, false, true,
                                            nullptr, &chunk_view));
        ZDNN_CHECK(zdnn_quantized_matmul_op(&inputs_res_view, &weights_view, &bias_view,
                                            MATMUL_OP_ADDITION, -127, 127, true, false, true,
                                            nullptr, &chunk_res_view));

        अगर (योग_task.valid()) {
            योग_task.लो();
        }

        अगर (n_threads > 1) {
            योग_task = std::async(std::launch::async, योग_chunk, chunk, g0, n);
        } अन्यथा {
            योग_chunk(chunk, g0, n);
        }
    }

    अगर (योग_task.valid()) {
        योग_task.लो();
    }

}

व्योम ggml_zdnn_mul_mat_q(
          ggml_backend_zdnn_context * ctx,
    स्थिर               ggml_tensor * src0,
    स्थिर               ggml_tensor * src1,
                        ggml_tensor * dst) {
    GGML_TENSOR_द्विआधारी_OP_LOCALS;

    स्थिर क्रमागत ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_आकार(type));
    GGML_ASSERT(nb10 == ggml_type_आकार(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == माप(भग्न));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    स्थिर ggml_tensor * weights = src0;
    स्थिर ggml_tensor * inputs  = src1;
          ggml_tensor * output  = dst;

    GGML_ASSERT(inputs->type == GGML_TYPE_F32);

    ggml_backend_zdnn_बफर * weights_extra = (ggml_backend_zdnn_बफर *)weights->extra;
    ggml_backend_zdnn_बफर * output_extra  = (ggml_backend_zdnn_बफर *)output->extra;

    // starting threads costs more than it saves for small inputs, e.g. during token generation
    स्थिर int32_t n_threads = ne10*ne11 >= 65536 ? ctx->n_threads : 1;

    अगर (weights_extra->pre_tfm_desc.layout == ZDNN_3DS) {
        GGML_ASSERT(weights_extra->pre_tfm_desc.dim1 == weights->ne[1] && "weights_extra->pre_tfm_desc.dim1 must match weights->ne[1]");

        ggml_zdnn_mul_mat_q_grouped(ctx, n_threads, inputs, weights_extra, output);

        // the result is written to output->data directly, so the output ztensor is stale
        output_extra->ztensor.is_transformed = false;
        वापस;
    }

    ggml_backend_zdnn_बफर * bias_extra = (ggml_backend_zdnn_बफर *)output_extra->extra;

    GGML_ASSERT(weights_extra->pre_tfm_desc.dim2 == weights->ne[0] && "weights_extra->pre_tfm_desc.dim2 must match weights->ne[0]");
    GGML_ASSERT(weights_extra->pre_tfm_desc.dim1 == weights->ne[1] && "weights_extra->pre_tfm_desc.dim1 must match weights->ne[1]");

    // zDNN takes one scale per zTensor, so quantize each row of the inputs with its own scale
    // the rounding errors of the inputs are stacked after the inputs, see ggml_zdnn_quantize_inputs
    ggml_zdnn_pool_alloc<int8_t> inputs_q_alloc(ctx->pool, 2 * ne10 * ne11);
    int8_t * inputs_q = inputs_q_alloc.लो();
    std::vector<भग्न> inputs_scales(ne11);
    ggml_zdnn_quantize_inputs(n_threads, inputs, ne10, weights_extra->channel_scales.तथ्य(),
                            inputs_q, inputs_q + ne10 * ne11, inputs_scales.तथ्य());

    zdnn_tensor_desc              inputs_pre_tfm_desc;
    zdnn_tensor_desc              inputs_tfm_desc;
    zdnn_ztensor                  inputs_ztensor;
    ggml_zdnn_pool_alloc<uint8_t> inputs_ztensor_alloc;

    zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &inputs_pre_tfm_desc, 2, ne11, ne10);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&inputs_pre_tfm_desc, QUANTIZED_INT8, &inputs_tfm_desc));
    ggml_zdnn_init_scratch_ztensor(ctx, &inputs_pre_tfm_desc, &inputs_tfm_desc, &inputs_ztensor, inputs_ztensor_alloc);
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&inputs_ztensor, false, -127, 127, inputs_q));

    zdnn_tensor_desc              partial_pre_tfm_desc;
    zdnn_tensor_desc              partial_tfm_desc;
    zdnn_ztensor                  partial_ztensor;
    ggml_zdnn_pool_alloc<uint8_t> partial_ztensor_alloc;

    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &partial_pre_tfm_desc, 2, ne11, ne0);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&partial_pre_tfm_desc, QUANTIZED_DLFLOAT16, &partial_tfm_desc));
    ggml_zdnn_init_scratch_ztensor(ctx, &partial_pre_tfm_desc, &partial_tfm_desc, &partial_ztensor, partial_ztensor_alloc);

    // both stacks are multiplied by the same unstacked weights in one broadcast call
    // the bias is all zeros, so it is already the pre-computed bias
    ZDNN_CHECK(zdnn_quantized_matmul_op(&inputs_ztensor, &weights_extra->ztensor, &bias_extra->ztensor,
                                        MATMUL_OP_ADDITION, -127, 127, true, false, true,
                                        nullptr, &partial_ztensor));

    // each stack of the partial results is read back as its own 2D ztensor
    zdnn_tensor_desc ढेर_pre_tfm_desc;
    zdnn_tensor_desc ढेर_tfm_desc;

    zdnn_init_pre_transformed_desc(ZDNN_2D, FP32, &ढेर_pre_tfm_desc, ne11, ne0);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&ढेर_pre_tfm_desc, &ढेर_tfm_desc));

    स्थिर uint64_t ढेर_bytes = zdnn_getsize_ztensor(&ढेर_tfm_desc);
    GGML_ASSERT(partial_ztensor.बफर_आकार == 2*ढेर_bytes);

    ggml_zdnn_pool_alloc<भग्न> partial_res_alloc(ctx->pool, ne11 * ne0);
    भग्न * partial[2] = { (भग्न *)output->तथ्य, partial_res_alloc.लो() };

    ggml_zdnn_parallel_for(n_threads, 2, [&](int64_t s_start, int64_t s_end) {
        zdnn_ztensor ढेर_ztensor;
        zdnn_init_ztensor(&ढेर_pre_tfm_desc, &ढेर_tfm_desc, &ढेर_ztensor);

        क्रम (int64_t s = s_start; s < s_end; s++) {
            ढेर_ztensor.बफर         = (अक्षर *)partial_ztensor.बफर + s*ढेर_bytes;
            ढेर_ztensor.बफर_आकार    = ढेर_bytes;
            ढेर_ztensor.is_transformed = true;
            ZDNN_CHECK(zdnn_transform_origtensor(&ढेर_ztensor, partial[s]));
        }
    });

    // the result is written to output->data directly, so the output ztensor is stale
    output_extra->ztensor.is_transformed = false;

    स्थिर भग्न * weights_scales = weights_extra->scales.तथ्य();
    ggml_zdnn_parallel_for(n_threads, ne1, [&](int64_t i1_start, int64_t i1_end) {
        क्रम (int64_t i1 = i1_start; i1 < i1_end; i1++) {
            स्थिर भग्न   sa = inputs_scales[i1];
            स्थिर भग्न * pr = partial[1] + i1*ne0;
            भग्न       * y  = partial[0] + i1*ne0;
            क्रम (int64_t i0 = 0; i0 < ne0; i0++) {
                y[i0] = sa * weights_scales[i0] * (y[i0] + pr[i0] * (1.0f/254.0f));
            }
        }
    });
}
