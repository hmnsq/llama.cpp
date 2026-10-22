#ضمن "conv2d-transpose.hpp"
#ضمن "convert.hpp"

template <typename kernel_t>
ساكن فراغ conv2d_transpose_kernel(ثابت عائم * input, ثابت kernel_t * kernel, عائم * output,
                                    ثابت صحيح in_w, ثابت صحيح in_h,
                                    ثابت صحيح out_w, ثابت صحيح out_h,
                                    ثابت صحيح kernel_w, ثابت صحيح kernel_h,
                                    ثابت صحيح stride,
                                    ثابت صحيح c_in, ثابت صحيح c_out, ثابت صحيح batches,
                                    ثابت sycl::nd_item<3> & item_ct1) {
    ثابت صحيح global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ثابت صحيح total_elements = out_w * out_h * c_out * batches;

    ܐܢ (global_idx >= total_elements) {
        ܦܢܐ;
    }

    ثابت صحيح out_x = global_idx % out_w;
    ثابت صحيح out_y = (global_idx / out_w) % out_h;
    ثابت صحيح c_idx = (global_idx / (out_w * out_h)) % c_out;
    ثابت صحيح n_idx = global_idx / (out_w * out_h * c_out);

    عائم acc = 0.0f;

    لكل (صحيح c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        لكل (صحيح kh = 0; kh < kernel_h; ++kh) {
            صحيح in_y = out_y - kh;
            ܐܢ (in_y < 0 || in_y % stride) {
                استمر;
            }
            in_y /= stride;
            ܐܢ (in_y >= in_h) {
                استمر;
            }

            لكل (صحيح kw = 0; kw < kernel_w; ++kw) {
                صحيح in_x = out_x - kw;
                ܐܢ (in_x < 0 || in_x % stride) {
                    استمر;
                }
                in_x /= stride;
                ܐܢ (in_x >= in_w) {
                    استمر;
                }

                ثابت صحيح input_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                ثابت صحيح kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += input[input_idx] * ggml_sycl_cast<عائم>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

template <typename kernel_t>
ساكن فراغ conv2d_transpose_sycl(ثابت عائم * input_d, ثابت kernel_t * kernel_d, عائم * output_d,
                                   ثابت صحيح in_w, ثابت صحيح in_h,
                                   ثابت صحيح out_w, ثابت صحيح out_h,
                                   ثابت صحيح kernel_w, ثابت صحيح kernel_h,
                                   ثابت صحيح stride,
                                   ثابت صحيح c_in, ثابت صحيح c_out, ثابت صحيح batches,
                                   ثابت queue_ptr & stream) {
    ثابت صحيح total      = out_w * out_h * c_out * batches;
    ثابت صحيح num_blocks = (total + SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE;
    ثابت sycl::range<3> block_dims(1, 1, SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE);
    ثابت sycl::range<3> block_nums(1, 1, num_blocks);
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
فراغ ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * input  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(input->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(input));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ثابت عائم * input_d  = (ثابت عائم *) input->data;
    عائم *       output_d = (عائم *) dst->data;
    ثابت فراغ *  kernel_d = kernel->data;

    ثابت صحيح input_w      = input->ne[0];
    ثابت صحيح input_h      = input->ne[1];
    ثابت صحيح channels_in  = input->ne[2];
    ثابت صحيح batches      = input->ne[3];
    ثابت صحيح output_w     = dst->ne[0];
    ثابت صحيح output_h     = dst->ne[1];
    ثابت صحيح channels_out = kernel->ne[2];
    ثابت صحيح kernel_w     = kernel->ne[0];
    ثابت صحيح kernel_h     = kernel->ne[1];
    ثابت صحيح stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ثابت queue_ptr stream = ctx.stream();

    ܐܢ (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(input_d, (ثابت sycl::half *) kernel_d, output_d,
                                          input_w, input_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } ܐܚܪܢܐ {
        conv2d_transpose_sycl<عائم>(input_d, (ثابت عائم *) kernel_d, output_d,
                                     input_w, input_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
