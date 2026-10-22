#include "conv2d-transpose.hpp"
#include "convert.hpp"

template <typename kernel_t>
static void conv2d_transpose_kernel(const float * input, const kernel_t * kernel, float * output,
                                    const ကိန်းပြည့် in_w, const ကိန်းပြည့် in_h,
                                    const ကိန်းပြည့် out_w, const ကိန်းပြည့် out_h,
                                    const ကိန်းပြည့် kernel_w, const ကိန်းပြည့် kernel_h,
                                    const ကိန်းပြည့် stride,
                                    const ကိန်းပြည့် c_in, const ကိန်းပြည့် c_out, const ကိန်းပြည့် batches,
                                    const sycl::nd_item<3> & item_ct1) {
    const ကိန်းပြည့် global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    const ကိန်းပြည့် total_elements = out_w * out_h * c_out * batches;

    အကယ်၍ (global_idx >= total_elements) {
        ပြန်ပေး;
    }

    const ကိန်းပြည့် out_x = global_idx % out_w;
    const ကိန်းပြည့် out_y = (global_idx / out_w) % out_h;
    const ကိန်းပြည့် c_idx = (global_idx / (out_w * out_h)) % c_out;
    const ကိန်းပြည့် n_idx = global_idx / (out_w * out_h * c_out);

    float acc = 0.0f;

    အတွက် (ကိန်းပြည့် c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        အတွက် (ကိန်းပြည့် kh = 0; kh < kernel_h; ++kh) {
            ကိန်းပြည့် in_y = out_y - kh;
            အကယ်၍ (in_y < 0 || in_y % stride) {
                continue;
            }
            in_y /= stride;
            အကယ်၍ (in_y >= in_h) {
                continue;
            }

            အတွက် (ကိန်းပြည့် kw = 0; kw < kernel_w; ++kw) {
                ကိန်းပြည့် in_x = out_x - kw;
                အကယ်၍ (in_x < 0 || in_x % stride) {
                    continue;
                }
                in_x /= stride;
                အကယ်၍ (in_x >= in_w) {
                    continue;
                }

                const ကိန်းပြည့် input_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                const ကိန်းပြည့် kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += input[input_idx] * ggml_sycl_cast<float>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

template <typename kernel_t>
static void conv2d_transpose_sycl(const float * input_d, const kernel_t * kernel_d, float * output_d,
                                   const ကိန်းပြည့် in_w, const ကိန်းပြည့် in_h,
                                   const ကိန်းပြည့် out_w, const ကိန်းပြည့် out_h,
                                   const ကိန်းပြည့် kernel_w, const ကိန်းပြည့် kernel_h,
                                   const ကိန်းပြည့် stride,
                                   const ကိန်းပြည့် c_in, const ကိန်းပြည့် c_out, const ကိန်းပြည့် batches,
                                   const queue_ptr & stream) {
    const ကိန်းပြည့် total      = out_w * out_h * c_out * batches;
    const ကိန်းပြည့် num_blocks = (total + SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE;
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

    const ကိန်းပြည့် input_w      = input->ne[0];
    const ကိန်းပြည့် input_h      = input->ne[1];
    const ကိန်းပြည့် channels_in  = input->ne[2];
    const ကိန်းပြည့် batches      = input->ne[3];
    const ကိန်းပြည့် output_w     = dst->ne[0];
    const ကိန်းပြည့် output_h     = dst->ne[1];
    const ကိန်းပြည့် channels_out = kernel->ne[2];
    const ကိန်းပြည့် kernel_w     = kernel->ne[0];
    const ကိန်းပြည့် kernel_h     = kernel->ne[1];
    const ကိန်းပြည့် stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    const queue_ptr stream = ctx.stream();

    အကယ်၍ (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(input_d, (const sycl::half *) kernel_d, output_d,
                                          input_w, input_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } မဟုတ်ပါက {
        conv2d_transpose_sycl<float>(input_d, (const float *) kernel_d, output_d,
                                     input_w, input_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
