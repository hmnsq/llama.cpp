#ضمن "ggml.h"
#ضمن "mmq.hpp"
#ضمن "quantize.hpp"

#ضمن <algorithm>
#ضمن <future>
#ضمن <vector>

// the weights are stacked into groups along ne00, and each group of each row of the inputs and weights has its own scale
// zDNN multiplies the groups in stacked calls, then the partial results of each group are scaled and summed here
ساكن فراغ ggml_zdnn_mul_mat_q_grouped(ggml_backend_zdnn_context * ctx,
                                        int32_t                     n_threads,
                                  ثابت ggml_tensor               * inputs,
                                  ثابت ggml_backend_zdnn_buffer  * weights_extra,
                                        ggml_tensor               * output) {
    ثابت int64_t ne10 = inputs->ne[0];
    ثابت int64_t ne11 = inputs->ne[1];
    ثابت int64_t ne0  = output->ne[0];

    ثابت int64_t n_groups = weights_extra->pre_tfm_desc.dim3;
    ثابت int64_t gs       = weights_extra->pre_tfm_desc.dim2;

    // the last group is zero padded when gs does not divide ne10
    GGML_ASSERT(n_groups * gs >= ne10 && (n_groups - 1) * gs < ne10);

    // the rounding errors of the inputs are stacked after the inputs, see ggml_zdnn_quantize_inputs
    ggml_zdnn_pool_alloc<int8_t> inputs_q_alloc(ctx->pool, 2 * n_groups * gs * ne11);
    int8_t * inputs_q = inputs_q_alloc.get();
    std::vector<عائم> inputs_scales(n_groups * ne11);
    ggml_zdnn_quantize_inputs(n_threads, inputs, gs, weights_extra->channel_scales.data(),
                            inputs_q, inputs_q + n_groups * gs * ne11, inputs_scales.data());

    zdnn_tensor_desc              inputs_pre_tfm_desc;
    zdnn_tensor_desc              inputs_tfm_desc;
    zdnn_ztensor                  inputs_ztensor;
    ggml_zdnn_pool_alloc<uint8_t> inputs_ztensor_alloc;

    zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &inputs_pre_tfm_desc, 2 * n_groups, ne11, gs);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&inputs_pre_tfm_desc, QUANTIZED_INT8, &inputs_tfm_desc));
    ggml_zdnn_init_scratch_ztensor(ctx, &inputs_pre_tfm_desc, &inputs_tfm_desc, &inputs_ztensor, inputs_ztensor_alloc);
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&inputs_ztensor, خطأ, -127, 127, inputs_q));

    // each group of the partial results is read back as its own 2D ztensor
    zdnn_tensor_desc group_pre_tfm_desc;
    zdnn_tensor_desc group_tfm_desc;

    zdnn_init_pre_transformed_desc(ZDNN_2D, FP32, &group_pre_tfm_desc, ne11, ne0);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&group_pre_tfm_desc, &group_tfm_desc));

    ثابت uint64_t group_bytes = zdnn_getsize_ztensor(&group_tfm_desc);

    // stacked calls with a large output are much slower per group, so run the groups in chunks
    // 8 MiB per chunk was benchmarked to be the best on Llama-3.2-1B Q8_0
    // always re-measure with different models to see if different sizes need a dispatching logic
    constexpr int64_t chunk_bytes = 8*1024*1024;

    ثابت int64_t n_chunk = std::clamp<int64_t>(chunk_bytes / (int64_t)group_bytes, 1, n_groups);

    // all zeros, so it is already the pre-computed bias for every chunk
    ثابت zdnn_ztensor & bias_ztensor = ggml_zdnn_zero_bias(ctx, n_chunk, ne0);

    // two partial buffers, so the next chunk runs on the NNPA while this one is summed
    // each holds the results of a chunk of inputs, followed by the results of their rounding errors
    zdnn_tensor_desc              partial_pre_tfm_desc;
    zdnn_tensor_desc              partial_tfm_desc;
    zdnn_ztensor                  partial_ztensor[2];
    ggml_zdnn_pool_alloc<uint8_t> partial_ztensor_alloc[2];

    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &partial_pre_tfm_desc, 2 * n_chunk, ne11, ne0);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&partial_pre_tfm_desc, QUANTIZED_DLFLOAT16, &partial_tfm_desc));
    GGML_ASSERT(zdnn_getsize_ztensor(&partial_tfm_desc) == (uint64_t)(2 * n_chunk) * group_bytes);
    لكل (int32_t i = 0; i < 2; i++) {
        ggml_zdnn_init_scratch_ztensor(ctx, &partial_pre_tfm_desc, &partial_tfm_desc, &partial_ztensor[i], partial_ztensor_alloc[i]);
    }

    ثابت عائم * weights_scales = weights_extra->scales.data();
    ggml_zdnn_pool_alloc<عائم> partial_alloc(ctx->pool, 2 * ne11 * ne0);
    عائم * partial     = partial_alloc.get();
    عائم * partial_res = partial + ne11 * ne0;

    ثابت auto sum_chunk = [&](ثابت zdnn_ztensor & chunk, int64_t g0, int64_t n) {
        zdnn_ztensor group_ztensor;
        zdnn_init_ztensor(&group_pre_tfm_desc, &group_tfm_desc, &group_ztensor);

        لكل (int64_t g = g0; g < g0 + n; g++) {
            group_ztensor.buffer_size    = group_bytes;
            group_ztensor.is_transformed = true;

            group_ztensor.buffer = (حرف *)chunk.buffer + (g - g0)*group_bytes;
            ZDNN_CHECK(zdnn_transform_origtensor(&group_ztensor, partial));

            group_ztensor.buffer = (حرف *)chunk.buffer + (n_chunk + g - g0)*group_bytes;
            ZDNN_CHECK(zdnn_transform_origtensor(&group_ztensor, partial_res));

            ثابت عائم * sw = weights_scales + g*ne0;
            لكل (int64_t i1 = 0; i1 < ne11; i1++) {
                ثابت عائم   sa = inputs_scales[g*ne11 + i1];
                ثابت عائم * p  = partial     + i1*ne0;
                ثابت عائم * pr = partial_res + i1*ne0;
                عائم       * y  = (عائم *)output->data + i1*ne0;

                إذا (g == 0) {
                    لكل (int64_t i0 = 0; i0 < ne0; i0++) {
                        y[i0] = sa * sw[i0] * (p[i0] + pr[i0] * (1.0f/254.0f));
                    }
                } وإلا {
                    لكل (int64_t i0 = 0; i0 < ne0; i0++) {
                        y[i0] += sa * sw[i0] * (p[i0] + pr[i0] * (1.0f/254.0f));
                    }
                }
            }
        }
    };

    zdnn_tensor_desc view_pre_tfm_desc[6];
    zdnn_tensor_desc view_tfm_desc[6];
    std::future<فراغ> sum_task;

    لكل (int64_t c = 0, g0 = 0; g0 < n_groups; c++, g0 += n_chunk) {
        ثابت int64_t n = std::min(n_chunk, n_groups - g0);

        ثابت zdnn_ztensor inputs_view     = ggml_zdnn_دۆۋە_view(&view_pre_tfm_desc[0], &view_tfm_desc[0], inputs_ztensor,         2 * n_groups, g0,            n);
        ثابت zdnn_ztensor inputs_res_view = ggml_zdnn_دۆۋە_view(&view_pre_tfm_desc[1], &view_tfm_desc[1], inputs_ztensor,         2 * n_groups, n_groups + g0, n);
        ثابت zdnn_ztensor weights_view    = ggml_zdnn_دۆۋە_view(&view_pre_tfm_desc[2], &view_tfm_desc[2], weights_extra->ztensor, n_groups,     g0,            n);
        ثابت zdnn_ztensor bias_view       = ggml_zdnn_دۆۋە_view(&view_pre_tfm_desc[3], &view_tfm_desc[3], bias_ztensor,           n_chunk,      0,             n);

        // the partial buffer of chunk c - 2 is summed by now, as sum_task is waited on before starting the next sum
        zdnn_ztensor & chunk = partial_ztensor[c % 2];
        chunk.pre_transformed_desc = &partial_pre_tfm_desc;
        chunk.transformed_desc     = &partial_tfm_desc;
        chunk.is_transformed       = خطأ;

        zdnn_ztensor chunk_view     = ggml_zdnn_دۆۋە_view(&view_pre_tfm_desc[4], &view_tfm_desc[4], chunk, 2 * n_chunk, 0,       n);
        zdnn_ztensor chunk_res_view = ggml_zdnn_دۆۋە_view(&view_pre_tfm_desc[5], &view_tfm_desc[5], chunk, 2 * n_chunk, n_chunk, n);
        ZDNN_CHECK(zdnn_quantized_matmul_op(&inputs_view, &weights_view, &bias_view,
                                            MATMUL_OP_ADDITION, -127, 127, true, خطأ, true,
                                            nullptr, &chunk_view));
        ZDNN_CHECK(zdnn_quantized_matmul_op(&inputs_res_view, &weights_view, &bias_view,
                                            MATMUL_OP_ADDITION, -127, 127, true, خطأ, true,
                                            nullptr, &chunk_res_view));

        إذا (sum_task.valid()) {
            sum_task.get();
        }

        إذا (n_threads > 1) {
            sum_task = std::async(std::launch::async, sum_chunk, chunk, g0, n);
        } وإلا {
            sum_chunk(chunk, g0, n);
        }
    }

    إذا (sum_task.valid()) {
        sum_task.get();
    }

}

فراغ ggml_zdnn_mul_mat_q(
          ggml_backend_zdnn_context * ctx,
    ثابت               ggml_tensor * src0,
    ثابت               ggml_tensor * src1,
                        ggml_tensor * dst) {
    GGML_TENSOR_ئىككىلىك_OP_LOCALS;

    ثابت تعداد ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == حجم(عائم));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    ثابت ggml_tensor * weights = src0;
    ثابت ggml_tensor * inputs  = src1;
          ggml_tensor * output  = dst;

    GGML_ASSERT(inputs->type == GGML_TYPE_F32);

    ggml_backend_zdnn_buffer * weights_extra = (ggml_backend_zdnn_buffer *)weights->extra;
    ggml_backend_zdnn_buffer * output_extra  = (ggml_backend_zdnn_buffer *)output->extra;

    // starting threads costs more than it saves for small inputs, e.g. during token generation
    ثابت int32_t n_threads = ne10*ne11 >= 65536 ? ctx->n_threads : 1;

    إذا (weights_extra->pre_tfm_desc.layout == ZDNN_3DS) {
        GGML_ASSERT(weights_extra->pre_tfm_desc.dim1 == weights->ne[1] && "weights_extra->pre_tfm_desc.dim1 must match weights->ne[1]");

        ggml_zdnn_mul_mat_q_grouped(ctx, n_threads, inputs, weights_extra, output);

        // the result is written to output->data directly, so the output ztensor is stale
        output_extra->ztensor.is_transformed = خطأ;
        أرجع;
    }

    ggml_backend_zdnn_buffer * bias_extra = (ggml_backend_zdnn_buffer *)output_extra->extra;

    GGML_ASSERT(weights_extra->pre_tfm_desc.dim2 == weights->ne[0] && "weights_extra->pre_tfm_desc.dim2 must match weights->ne[0]");
    GGML_ASSERT(weights_extra->pre_tfm_desc.dim1 == weights->ne[1] && "weights_extra->pre_tfm_desc.dim1 must match weights->ne[1]");

    // zDNN takes one scale per zTensor, so quantize each row of the inputs with its own scale
    // the rounding errors of the inputs are stacked after the inputs, see ggml_zdnn_quantize_inputs
    ggml_zdnn_pool_alloc<int8_t> inputs_q_alloc(ctx->pool, 2 * ne10 * ne11);
    int8_t * inputs_q = inputs_q_alloc.get();
    std::vector<عائم> inputs_scales(ne11);
    ggml_zdnn_quantize_inputs(n_threads, inputs, ne10, weights_extra->channel_scales.data(),
                            inputs_q, inputs_q + ne10 * ne11, inputs_scales.data());

    zdnn_tensor_desc              inputs_pre_tfm_desc;
    zdnn_tensor_desc              inputs_tfm_desc;
    zdnn_ztensor                  inputs_ztensor;
    ggml_zdnn_pool_alloc<uint8_t> inputs_ztensor_alloc;

    zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &inputs_pre_tfm_desc, 2, ne11, ne10);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&inputs_pre_tfm_desc, QUANTIZED_INT8, &inputs_tfm_desc));
    ggml_zdnn_init_scratch_ztensor(ctx, &inputs_pre_tfm_desc, &inputs_tfm_desc, &inputs_ztensor, inputs_ztensor_alloc);
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&inputs_ztensor, خطأ, -127, 127, inputs_q));

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
                                        MATMUL_OP_ADDITION, -127, 127, true, خطأ, true,
                                        nullptr, &partial_ztensor));

    // each stack of the partial results is read back as its own 2D ztensor
    zdnn_tensor_desc دۆۋە_pre_tfm_desc;
    zdnn_tensor_desc دۆۋە_tfm_desc;

    zdnn_init_pre_transformed_desc(ZDNN_2D, FP32, &دۆۋە_pre_tfm_desc, ne11, ne0);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&دۆۋە_pre_tfm_desc, &دۆۋە_tfm_desc));

    ثابت uint64_t دۆۋە_bytes = zdnn_getsize_ztensor(&دۆۋە_tfm_desc);
    GGML_ASSERT(partial_ztensor.buffer_size == 2*دۆۋە_bytes);

    ggml_zdnn_pool_alloc<عائم> partial_res_alloc(ctx->pool, ne11 * ne0);
    عائم * partial[2] = { (عائم *)output->data, partial_res_alloc.get() };

    ggml_zdnn_parallel_for(n_threads, 2, [&](int64_t s_start, int64_t s_end) {
        zdnn_ztensor دۆۋە_ztensor;
        zdnn_init_ztensor(&دۆۋە_pre_tfm_desc, &دۆۋە_tfm_desc, &دۆۋە_ztensor);

        لكل (int64_t s = s_start; s < s_end; s++) {
            دۆۋە_ztensor.buffer         = (حرف *)partial_ztensor.buffer + s*دۆۋە_bytes;
            دۆۋە_ztensor.buffer_size    = دۆۋە_bytes;
            دۆۋە_ztensor.is_transformed = true;
            ZDNN_CHECK(zdnn_transform_origtensor(&دۆۋە_ztensor, partial[s]));
        }
    });

    // the result is written to output->data directly, so the output ztensor is stale
    output_extra->ztensor.is_transformed = خطأ;

    ثابت عائم * weights_scales = weights_extra->scales.data();
    ggml_zdnn_parallel_for(n_threads, ne1, [&](int64_t i1_start, int64_t i1_end) {
        لكل (int64_t i1 = i1_start; i1 < i1_end; i1++) {
            ثابت عائم   sa = inputs_scales[i1];
            ثابت عائم * pr = partial[1] + i1*ne0;
            عائم       * y  = partial[0] + i1*ne0;
            لكل (int64_t i0 = 0; i0 < ne0; i0++) {
                y[i0] = sa * weights_scales[i0] * (y[i0] + pr[i0] * (1.0f/254.0f));
            }
        }
    });
}
