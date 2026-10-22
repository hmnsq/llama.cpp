#অন্তর্ভাব "conv2d-transpose.hpp"
#অন্তর্ভাব "convert.hpp"

template <typename kernel_t>
স্থির শূন্য conv2d_transpose_kernel(ধ্রুবক দশমিক * প্রবেশ, ধ্রুবক kernel_t * kernel, দশমিক * output,
                                    ধ্রুবক পূর্ণসংখ্যা in_w, ধ্রুবক পূর্ণসংখ্যা in_h,
                                    ধ্রুবক পূর্ণসংখ্যা out_w, ধ্রুবক পূর্ণসংখ্যা out_h,
                                    ধ্রুবক পূর্ণসংখ্যা kernel_w, ধ্রুবক পূর্ণসংখ্যা kernel_h,
                                    ধ্রুবক পূর্ণসংখ্যা stride,
                                    ধ্রুবক পূর্ণসংখ্যা c_in, ধ্রুবক পূর্ণসংখ্যা c_out, ধ্রুবক পূর্ণসংখ্যা batches,
                                    ধ্রুবক sycl::nd_item<3> & item_ct1) {
    ধ্রুবক পূর্ণসংখ্যা global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ধ্রুবক পূর্ণসংখ্যা যোগ_elements = out_w * out_h * c_out * batches;

    যদি (global_idx >= যোগ_elements) {
        ফেরত;
    }

    ধ্রুবক পূর্ণসংখ্যা out_x = global_idx % out_w;
    ধ্রুবক পূর্ণসংখ্যা out_y = (global_idx / out_w) % out_h;
    ধ্রুবক পূর্ণসংখ্যা c_idx = (global_idx / (out_w * out_h)) % c_out;
    ধ্রুবক পূর্ণসংখ্যা n_idx = global_idx / (out_w * out_h * c_out);

    দশমিক acc = 0.0f;

    জন্য (পূর্ণসংখ্যা c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        জন্য (পূর্ণসংখ্যা kh = 0; kh < kernel_h; ++kh) {
            পূর্ণসংখ্যা in_y = out_y - kh;
            যদি (in_y < 0 || in_y % stride) {
                চালু;
            }
            in_y /= stride;
            যদি (in_y >= in_h) {
                চালু;
            }

            জন্য (পূর্ণসংখ্যা kw = 0; kw < kernel_w; ++kw) {
                পূর্ণসংখ্যা in_x = out_x - kw;
                যদি (in_x < 0 || in_x % stride) {
                    চালু;
                }
                in_x /= stride;
                যদি (in_x >= in_w) {
                    চালু;
                }

                ধ্রুবক পূর্ণসংখ্যা প্রবেশ_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                ধ্রুবক পূর্ণসংখ্যা kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += প্রবেশ[প্রবেশ_idx] * ggml_sycl_cast<দশমিক>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

template <typename kernel_t>
স্থির শূন্য conv2d_transpose_sycl(ধ্রুবক দশমিক * প্রবেশ_d, ধ্রুবক kernel_t * kernel_d, দশমিক * output_d,
                                   ধ্রুবক পূর্ণসংখ্যা in_w, ধ্রুবক পূর্ণসংখ্যা in_h,
                                   ধ্রুবক পূর্ণসংখ্যা out_w, ধ্রুবক পূর্ণসংখ্যা out_h,
                                   ধ্রুবক পূর্ণসংখ্যা kernel_w, ধ্রুবক পূর্ণসংখ্যা kernel_h,
                                   ধ্রুবক পূর্ণসংখ্যা stride,
                                   ধ্রুবক পূর্ণসংখ্যা c_in, ধ্রুবক পূর্ণসংখ্যা c_out, ধ্রুবক পূর্ণসংখ্যা batches,
                                   ধ্রুবক queue_ptr & stream) {
    ধ্রুবক পূর্ণসংখ্যা যোগ      = out_w * out_h * c_out * batches;
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (যোগ + SYCL_CONV2_D_TRANSPOSE_BLOCK_আকার - 1) / SYCL_CONV2_D_TRANSPOSE_BLOCK_আকার;
    ধ্রুবক sycl::range<3> block_dims(1, 1, SYCL_CONV2_D_TRANSPOSE_BLOCK_আকার);
    ধ্রুবক sycl::range<3> block_nums(1, 1, অঙ্ক_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_transpose_kernel<kernel_t>(প্রবেশ_d, kernel_d, output_d,
                                             in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                             stride, c_in, c_out, batches, item_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
শূন্য ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ধ্রুবক ggml_tensor * kernel = dst->src[0];
    ধ্রুবক ggml_tensor * প্রবেশ  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(প্রবেশ->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(প্রবেশ));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ধ্রুবক দশমিক * প্রবেশ_d  = (ধ্রুবক দশমিক *) প্রবেশ->তথ্য;
    দশমিক *       output_d = (দশমিক *) dst->তথ্য;
    ধ্রুবক শূন্য *  kernel_d = kernel->তথ্য;

    ধ্রুবক পূর্ণসংখ্যা প্রবেশ_w      = প্রবেশ->ne[0];
    ধ্রুবক পূর্ণসংখ্যা প্রবেশ_h      = প্রবেশ->ne[1];
    ধ্রুবক পূর্ণসংখ্যা channels_in  = প্রবেশ->ne[2];
    ধ্রুবক পূর্ণসংখ্যা batches      = প্রবেশ->ne[3];
    ধ্রুবক পূর্ণসংখ্যা output_w     = dst->ne[0];
    ধ্রুবক পূর্ণসংখ্যা output_h     = dst->ne[1];
    ধ্রুবক পূর্ণসংখ্যা channels_out = kernel->ne[2];
    ধ্রুবক পূর্ণসংখ্যা kernel_w     = kernel->ne[0];
    ধ্রুবক পূর্ণসংখ্যা kernel_h     = kernel->ne[1];
    ধ্রুবক পূর্ণসংখ্যা stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ধ্রুবক queue_ptr stream = ctx.stream();

    যদি (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(প্রবেশ_d, (ধ্রুবক sycl::half *) kernel_d, output_d,
                                          প্রবেশ_w, প্রবেশ_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } অন্যথা {
        conv2d_transpose_sycl<দশমিক>(প্রবেশ_d, (ধ্রুবক দশমিক *) kernel_d, output_d,
                                     প্রবেশ_w, প্রবেশ_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
