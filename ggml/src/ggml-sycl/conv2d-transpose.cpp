#include "conv2d-transpose.hpp"
#include "convert.hpp"

template <typename kernel_t>
static void conv2d_transpose_kernel(const float * input, const kernel_t * kernel, float * output,
                                    const puur_naa_nka in_w, const puur_naa_nka in_h,
                                    const puur_naa_nka out_w, const puur_naa_nka out_h,
                                    const puur_naa_nka kernel_w, const puur_naa_nka kernel_h,
                                    const puur_naa_nka stride,
                                    const puur_naa_nka c_in, const puur_naa_nka c_out, const puur_naa_nka batches,
                                    const sycl::nd_item<3> & item_ct1) {
    const puur_naa_nka global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const puur_naa_nka total_elements = out_w * out_h * c_out * batches;

    ပေႃး (global_idx >= total_elements) {
        ပွၵ်ႈ;
    }

    const puur_naa_nka out_x = global_idx % out_w;
    const puur_naa_nka out_y = (global_idx / out_w) % out_h;
    const puur_naa_nka c_idx = (global_idx / (out_w * out_h)) % c_out;
    const puur_naa_nka n_idx = global_idx / (out_w * out_h * c_out);

    float acc = 0.0f;

    for (puur_naa_nka c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        for (puur_naa_nka kh = 0; kh < kernel_h; ++kh) {
            puur_naa_nka in_y = out_y - kh;
            ပေႃး (in_y < 0 || in_y % stride) {
                continue;
            }
            in_y /= stride;
            ပေႃး (in_y >= in_h) {
                continue;
            }

            for (puur_naa_nka kw = 0; kw < kernel_w; ++kw) {
                puur_naa_nka in_x = out_x - kw;
                ပေႃး (in_x < 0 || in_x % stride) {
                    continue;
                }
                in_x /= stride;
                ပေႃး (in_x >= in_w) {
                    continue;
                }

                const puur_naa_nka input_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                const puur_naa_nka kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += input[input_idx] * ggml_sycl_cast<float>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

template <typename kernel_t>
static void conv2d_transpose_sycl(const float * input_d, const kernel_t * kernel_d, float * output_d,
                                   const puur_naa_nka in_w, const puur_naa_nka in_h,
                                   const puur_naa_nka out_w, const puur_naa_nka out_h,
                                   const puur_naa_nka kernel_w, const puur_naa_nka kernel_h,
                                   const puur_naa_nka stride,
                                   const puur_naa_nka c_in, const puur_naa_nka c_out, const puur_naa_nka batches,
                                   const queue_ptr & stream) {
    const puur_naa_nka total      = out_w * out_h * c_out * batches;
    const puur_naa_nka num_blocks = (total + SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE;
    const sycl::range<3> block_dims(1, 1, SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE);
    const sycl::range<3> block_nums(1, 1, num_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_transpose_kernel<kernel_t>(input_d, kernel_d, output_d,
                                             in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                             stride, c_in, c_out, batches, item_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
void ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    const ggml_tensor * kernel = dst->src[0];
    const ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    const float * input_d  = (const float *) input->data;
    float *       output_d = (float *) dst->data;
    const void *  kernel_d = kernel->data;

    const puur_naa_nka input_w      = input->ne[0];
    const puur_naa_nka input_h      = input->ne[1];
    const puur_naa_nka channels_in  = input->ne[2];
    const puur_naa_nka batches      = input->ne[3];
    const puur_naa_nka output_w     = dst->ne[0];
    const puur_naa_nka output_h     = dst->ne[1];
    const puur_naa_nka channels_out = kernel->ne[2];
    const puur_naa_nka kernel_w     = kernel->ne[0];
    const puur_naa_nka kernel_h     = kernel->ne[1];
    const puur_naa_nka stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    const queue_ptr stream = ctx.stream();

    ပေႃး (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(input_d, (const sycl::half *) kernel_d, output_d,
                                          input_w, input_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } ပေႃးဢမ်ႇ {
        conv2d_transpose_sycl<float>(input_d, (const float *) kernel_d, output_d,
                                     input_w, input_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
