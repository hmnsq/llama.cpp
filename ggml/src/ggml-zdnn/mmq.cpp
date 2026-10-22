#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "mmq.hpp"
#અન્તર્ભાવ "quantize.hpp"

#અન્તર્ભાવ <algorithm>
#અન્તર્ભાવ <future>
#અન્તર્ભાવ <vector>

// the weights are stacked into groups along ne00, and each group of each row of the inputs and weights has its own scale
// zDNN multiplies the groups in stacked calls, then the partial results of each group are scaled and summed here
સ્થિર શૂન્ય ggml_zdnn_mul_mat_q_grouped(ggml_backend_zdnn_context * ctx,
                                        int32_t                     n_threads,
                                  અચલ ggml_tensor               * inputs,
                                  અચલ ggml_backend_zdnn_buffer  * weights_extra,
                                        ggml_tensor               * output) {
    અચલ int64_t ne10 = inputs->ne[0];
    અચલ int64_t ne11 = inputs->ne[1];
    અચલ int64_t ne0  = output->ne[0];

    અચલ int64_t n_groups = weights_extra->pre_tfm_desc.dim3;
    અચલ int64_t gs       = weights_extra->pre_tfm_desc.dim2;

    // the last group is zero padded when gs does not divide ne10
    GGML_ASSERT(n_groups * gs >= ne10 && (n_groups - 1) * gs < ne10);

    // the rounding errors of the inputs are stacked after the inputs, see ggml_zdnn_quantize_inputs
    ggml_zdnn_pool_alloc<int8_t> inputs_q_alloc(ctx->pool, 2 * n_groups * gs * ne11);
    int8_t * inputs_q = inputs_q_alloc.get();
    std::vector<દશાંશ> inputs_scales(n_groups * ne11);
    ggml_zdnn_quantize_inputs(n_threads, inputs, gs, weights_extra->channel_scales.data(),
                            inputs_q, inputs_q + n_groups * gs * ne11, inputs_scales.data());

    zdnn_tensor_desc              inputs_pre_tfm_desc;
    zdnn_tensor_desc              inputs_tfm_desc;
    zdnn_ztensor                  inputs_ztensor;
    ggml_zdnn_pool_alloc<uint8_t> inputs_ztensor_alloc;

    zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &inputs_pre_tfm_desc, 2 * n_groups, ne11, gs);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&inputs_pre_tfm_desc, QUANTIZED_INT8, &inputs_tfm_desc));
    ggml_zdnn_init_scratch_ztensor(ctx, &inputs_pre_tfm_desc, &inputs_tfm_desc, &inputs_ztensor, inputs_ztensor_alloc);
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&inputs_ztensor, ખોટું, -127, 127, inputs_q));

    // each group of the partial results is read back as its own 2D ztensor
    zdnn_tensor_desc group_pre_tfm_desc;
    zdnn_tensor_desc group_tfm_desc;

    zdnn_init_pre_transformed_desc(ZDNN_2D, FP32, &group_pre_tfm_desc, ne11, ne0);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&group_pre_tfm_desc, &group_tfm_desc));

    અચલ uint64_t group_bytes = zdnn_getsize_ztensor(&group_tfm_desc);

    // stacked calls with a large output are much slower per group, so run the groups in chunks
    // 8 MiB per chunk was benchmarked to be the best on Llama-3.2-1B Q8_0
    // always re-measure with different models to see if different sizes need a dispatching logic
    constexpr int64_t chunk_bytes = 8*1024*1024;

    અચલ int64_t n_chunk = std::clamp<int64_t>(chunk_bytes / (int64_t)group_bytes, 1, n_groups);

    // all zeros, so it is already the pre-computed bias for every chunk
    અચલ zdnn_ztensor & bias_ztensor = ggml_zdnn_zero_bias(ctx, n_chunk, ne0);

    // two partial buffers, so the next chunk runs on the NNPA while this one is summed
    // each holds the results of a chunk of inputs, followed by the results of their rounding errors
    zdnn_tensor_desc              partial_pre_tfm_desc;
    zdnn_tensor_desc              partial_tfm_desc;
    zdnn_ztensor                  partial_ztensor[2];
    ggml_zdnn_pool_alloc<uint8_t> partial_ztensor_alloc[2];

    zdnn_init_pre_transformed_desc(ZDNN_3DS, FP32, &partial_pre_tfm_desc, 2 * n_chunk, ne11, ne0);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&partial_pre_tfm_desc, QUANTIZED_DLFLOAT16, &partial_tfm_desc));
    GGML_ASSERT(zdnn_getsize_ztensor(&partial_tfm_desc) == (uint64_t)(2 * n_chunk) * group_bytes);
    માટે (int32_t i = 0; i < 2; i++) {
        ggml_zdnn_init_scratch_ztensor(ctx, &partial_pre_tfm_desc, &partial_tfm_desc, &partial_ztensor[i], partial_ztensor_alloc[i]);
    }

    અચલ દશાંશ * weights_scales = weights_extra->scales.data();
    ggml_zdnn_pool_alloc<દશાંશ> partial_alloc(ctx->pool, 2 * ne11 * ne0);
    દશાંશ * partial     = partial_alloc.get();
    દશાંશ * partial_res = partial + ne11 * ne0;

    અચલ auto sum_chunk = [&](અચલ zdnn_ztensor & chunk, int64_t g0, int64_t n) {
        zdnn_ztensor group_ztensor;
        zdnn_init_ztensor(&group_pre_tfm_desc, &group_tfm_desc, &group_ztensor);

        માટે (int64_t g = g0; g < g0 + n; g++) {
            group_ztensor.buffer_size    = group_bytes;
            group_ztensor.is_transformed = સાચું;

            group_ztensor.buffer = (અક્ષર *)chunk.buffer + (g - g0)*group_bytes;
            ZDNN_CHECK(zdnn_transform_origtensor(&group_ztensor, partial));

            group_ztensor.buffer = (અક્ષર *)chunk.buffer + (n_chunk + g - g0)*group_bytes;
            ZDNN_CHECK(zdnn_transform_origtensor(&group_ztensor, partial_res));

            અચલ દશાંશ * sw = weights_scales + g*ne0;
            માટે (int64_t i1 = 0; i1 < ne11; i1++) {
                અચલ દશાંશ   sa = inputs_scales[g*ne11 + i1];
                અચલ દશાંશ * p  = partial     + i1*ne0;
                અચલ દશાંશ * pr = partial_res + i1*ne0;
                દશાંશ       * y  = (દશાંશ *)output->data + i1*ne0;

                જો (g == 0) {
                    માટે (int64_t i0 = 0; i0 < ne0; i0++) {
                        y[i0] = sa * sw[i0] * (p[i0] + pr[i0] * (1.0f/254.0f));
                    }
                } નહીંતર {
                    માટે (int64_t i0 = 0; i0 < ne0; i0++) {
                        y[i0] += sa * sw[i0] * (p[i0] + pr[i0] * (1.0f/254.0f));
                    }
                }
            }
        }
    };

    zdnn_tensor_desc view_pre_tfm_desc[6];
    zdnn_tensor_desc view_tfm_desc[6];
    std::future<શૂન્ય> sum_task;

    માટે (int64_t c = 0, g0 = 0; g0 < n_groups; c++, g0 += n_chunk) {
        અચલ int64_t n = std::min(n_chunk, n_groups - g0);

        અચલ zdnn_ztensor inputs_view     = ggml_zdnn_ઢગલો_view(&view_pre_tfm_desc[0], &view_tfm_desc[0], inputs_ztensor,         2 * n_groups, g0,            n);
        અચલ zdnn_ztensor inputs_res_view = ggml_zdnn_ઢગલો_view(&view_pre_tfm_desc[1], &view_tfm_desc[1], inputs_ztensor,         2 * n_groups, n_groups + g0, n);
        અચલ zdnn_ztensor weights_view    = ggml_zdnn_ઢગલો_view(&view_pre_tfm_desc[2], &view_tfm_desc[2], weights_extra->ztensor, n_groups,     g0,            n);
        અચલ zdnn_ztensor bias_view       = ggml_zdnn_ઢગલો_view(&view_pre_tfm_desc[3], &view_tfm_desc[3], bias_ztensor,           n_chunk,      0,             n);

        // the partial buffer of chunk c - 2 is summed by now, as sum_task is waited on before starting the next sum
        zdnn_ztensor & chunk = partial_ztensor[c % 2];
        chunk.pre_transformed_desc = &partial_pre_tfm_desc;
        chunk.transformed_desc     = &partial_tfm_desc;
        chunk.is_transformed       = ખોટું;

        zdnn_ztensor chunk_view     = ggml_zdnn_ઢગલો_view(&view_pre_tfm_desc[4], &view_tfm_desc[4], chunk, 2 * n_chunk, 0,       n);
        zdnn_ztensor chunk_res_view = ggml_zdnn_ઢગલો_view(&view_pre_tfm_desc[5], &view_tfm_desc[5], chunk, 2 * n_chunk, n_chunk, n);
        ZDNN_CHECK(zdnn_quantized_matmul_op(&inputs_view, &weights_view, &bias_view,
                                            MATMUL_OP_ADDITION, -127, 127, સાચું, ખોટું, સાચું,
                                            nullptr, &chunk_view));
        ZDNN_CHECK(zdnn_quantized_matmul_op(&inputs_res_view, &weights_view, &bias_view,
                                            MATMUL_OP_ADDITION, -127, 127, સાચું, ખોટું, સાચું,
                                            nullptr, &chunk_res_view));

        જો (sum_task.valid()) {
            sum_task.get();
        }

        જો (n_threads > 1) {
            sum_task = std::async(std::launch::async, sum_chunk, chunk, g0, n);
        } નહીંતર {
            sum_chunk(chunk, g0, n);
        }
    }

    જો (sum_task.valid()) {
        sum_task.get();
    }

}

શૂન્ય ggml_zdnn_mul_mat_q(
          ggml_backend_zdnn_context * ctx,
    અચલ               ggml_tensor * src0,
    અચલ               ggml_tensor * src1,
                        ggml_tensor * dst) {
    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS;

    અચલ ગણના ggml_type type = src0->type;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb00 == ggml_type_size(type));
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == કદ(દશાંશ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    અચલ ggml_tensor * weights = src0;
    અચલ ggml_tensor * inputs  = src1;
          ggml_tensor * output  = dst;

    GGML_ASSERT(inputs->type == GGML_TYPE_F32);

    ggml_backend_zdnn_buffer * weights_extra = (ggml_backend_zdnn_buffer *)weights->extra;
    ggml_backend_zdnn_buffer * output_extra  = (ggml_backend_zdnn_buffer *)output->extra;

    // starting threads costs more than it saves for small inputs, e.g. during token generation
    અચલ int32_t n_threads = ne10*ne11 >= 65536 ? ctx->n_threads : 1;

    જો (weights_extra->pre_tfm_desc.layout == ZDNN_3DS) {
        GGML_ASSERT(weights_extra->pre_tfm_desc.dim1 == weights->ne[1] && "weights_extra->pre_tfm_desc.dim1 must match weights->ne[1]");

        ggml_zdnn_mul_mat_q_grouped(ctx, n_threads, inputs, weights_extra, output);

        // the result is written to output->data directly, so the output ztensor is stale
        output_extra->ztensor.is_transformed = ખોટું;
        પાછા;
    }

    ggml_backend_zdnn_buffer * bias_extra = (ggml_backend_zdnn_buffer *)output_extra->extra;

    GGML_ASSERT(weights_extra->pre_tfm_desc.dim2 == weights->ne[0] && "weights_extra->pre_tfm_desc.dim2 must match weights->ne[0]");
    GGML_ASSERT(weights_extra->pre_tfm_desc.dim1 == weights->ne[1] && "weights_extra->pre_tfm_desc.dim1 must match weights->ne[1]");

    // zDNN takes one scale per zTensor, so quantize each row of the inputs with its own scale
    // the rounding errors of the inputs are stacked after the inputs, see ggml_zdnn_quantize_inputs
    ggml_zdnn_pool_alloc<int8_t> inputs_q_alloc(ctx->pool, 2 * ne10 * ne11);
    int8_t * inputs_q = inputs_q_alloc.get();
    std::vector<દશાંશ> inputs_scales(ne11);
    ggml_zdnn_quantize_inputs(n_threads, inputs, ne10, weights_extra->channel_scales.data(),
                            inputs_q, inputs_q + ne10 * ne11, inputs_scales.data());

    zdnn_tensor_desc              inputs_pre_tfm_desc;
    zdnn_tensor_desc              inputs_tfm_desc;
    zdnn_ztensor                  inputs_ztensor;
    ggml_zdnn_pool_alloc<uint8_t> inputs_ztensor_alloc;

    zdnn_init_pre_transformed_desc(ZDNN_3DS, INT8, &inputs_pre_tfm_desc, 2, ne11, ne10);
    ZDNN_CHECK(zdnn_generate_quantized_transformed_desc(&inputs_pre_tfm_desc, QUANTIZED_INT8, &inputs_tfm_desc));
    ggml_zdnn_init_scratch_ztensor(ctx, &inputs_pre_tfm_desc, &inputs_tfm_desc, &inputs_ztensor, inputs_ztensor_alloc);
    ZDNN_CHECK(zdnn_transform_quantized_ztensor(&inputs_ztensor, ખોટું, -127, 127, inputs_q));

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
                                        MATMUL_OP_ADDITION, -127, 127, સાચું, ખોટું, સાચું,
                                        nullptr, &partial_ztensor));

    // each stack of the partial results is read back as its own 2D ztensor
    zdnn_tensor_desc ઢગલો_pre_tfm_desc;
    zdnn_tensor_desc ઢગલો_tfm_desc;

    zdnn_init_pre_transformed_desc(ZDNN_2D, FP32, &ઢગલો_pre_tfm_desc, ne11, ne0);
    ZDNN_CHECK(zdnn_generate_transformed_desc(&ઢગલો_pre_tfm_desc, &ઢગલો_tfm_desc));

    અચલ uint64_t ઢગલો_bytes = zdnn_getsize_ztensor(&ઢગલો_tfm_desc);
    GGML_ASSERT(partial_ztensor.buffer_size == 2*ઢગલો_bytes);

    ggml_zdnn_pool_alloc<દશાંશ> partial_res_alloc(ctx->pool, ne11 * ne0);
    દશાંશ * partial[2] = { (દશાંશ *)output->data, partial_res_alloc.get() };

    ggml_zdnn_parallel_for(n_threads, 2, [&](int64_t s_start, int64_t s_end) {
        zdnn_ztensor ઢગલો_ztensor;
        zdnn_init_ztensor(&ઢગલો_pre_tfm_desc, &ઢગલો_tfm_desc, &ઢગલો_ztensor);

        માટે (int64_t s = s_start; s < s_end; s++) {
            ઢગલો_ztensor.buffer         = (અક્ષર *)partial_ztensor.buffer + s*ઢગલો_bytes;
            ઢગલો_ztensor.buffer_size    = ઢગલો_bytes;
            ઢગલો_ztensor.is_transformed = સાચું;
            ZDNN_CHECK(zdnn_transform_origtensor(&ઢગલો_ztensor, partial[s]));
        }
    });

    // the result is written to output->data directly, so the output ztensor is stale
    output_extra->ztensor.is_transformed = ખોટું;

    અચલ દશાંશ * weights_scales = weights_extra->scales.data();
    ggml_zdnn_parallel_for(n_threads, ne1, [&](int64_t i1_start, int64_t i1_end) {
        માટે (int64_t i1 = i1_start; i1 < i1_end; i1++) {
            અચલ દશાંશ   sa = inputs_scales[i1];
            અચલ દશાંશ * pr = partial[1] + i1*ne0;
            દશાંશ       * y  = partial[0] + i1*ne0;
            માટે (int64_t i0 = 0; i0 < ne0; i0++) {
                y[i0] = sa * weights_scales[i0] * (y[i0] + pr[i0] * (1.0f/254.0f));
            }
        }
    });
}
