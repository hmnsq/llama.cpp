#অন্তর্ভাব "conv2d-transpose.hpp"
#অন্তর্ভাব "convert.hpp"

template <typename kernel_t>
স্থির শূন্য conv2d_transpose_kernel(ধ্রুব ভিন্ন * প্রবেশ, ধ্রুব kernel_t * kernel, ভিন্ন * output,
                                    ধ্রুব পূর্ণাঙ্ক in_w, ধ্রুব পূর্ণাঙ্ক in_h,
                                    ধ্রুব পূর্ণাঙ্ক out_w, ধ্রুব পূর্ণাঙ্ক out_h,
                                    ধ্রুব পূর্ণাঙ্ক kernel_w, ধ্রুব পূর্ণাঙ্ক kernel_h,
                                    ধ্রুব পূর্ণাঙ্ক stride,
                                    ধ্রুব পূর্ণাঙ্ক c_in, ধ্রুব পূর্ণাঙ্ক c_out, ধ্রুব পূর্ণাঙ্ক batches,
                                    ধ্রুব sycl::nd_item<3> & item_ct1) {
    ধ্রুব পূর্ণাঙ্ক global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ধ্রুব পূর্ণাঙ্ক total_elements = out_w * out_h * c_out * batches;

    যদি (global_idx >= total_elements) {
        প্রত্যাগম;
    }

    ধ্রুব পূর্ণাঙ্ক out_x = global_idx % out_w;
    ধ্রুব পূর্ণাঙ্ক out_y = (global_idx / out_w) % out_h;
    ধ্রুব পূর্ণাঙ্ক c_idx = (global_idx / (out_w * out_h)) % c_out;
    ধ্রুব পূর্ণাঙ্ক n_idx = global_idx / (out_w * out_h * c_out);

    ভিন্ন acc = 0.0f;

    প্রতি (পূর্ণাঙ্ক c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        প্রতি (পূর্ণাঙ্ক kh = 0; kh < kernel_h; ++kh) {
            পূর্ণাঙ্ক in_y = out_y - kh;
            যদি (in_y < 0 || in_y % stride) {
                অনুবর্ত;
            }
            in_y /= stride;
            যদি (in_y >= in_h) {
                অনুবর্ত;
            }

            প্রতি (পূর্ণাঙ্ক kw = 0; kw < kernel_w; ++kw) {
                পূর্ণাঙ্ক in_x = out_x - kw;
                যদি (in_x < 0 || in_x % stride) {
                    অনুবর্ত;
                }
                in_x /= stride;
                যদি (in_x >= in_w) {
                    অনুবর্ত;
                }

                ধ্রুব পূর্ণাঙ্ক প্রবেশ_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                ধ্রুব পূর্ণাঙ্ক kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += প্রবেশ[প্রবেশ_idx] * ggml_sycl_cast<ভিন্ন>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

template <typename kernel_t>
স্থির শূন্য conv2d_transpose_sycl(ধ্রুব ভিন্ন * প্রবেশ_d, ধ্রুব kernel_t * kernel_d, ভিন্ন * output_d,
                                   ধ্রুব পূর্ণাঙ্ক in_w, ধ্রুব পূর্ণাঙ্ক in_h,
                                   ধ্রুব পূর্ণাঙ্ক out_w, ধ্রুব পূর্ণাঙ্ক out_h,
                                   ধ্রুব পূর্ণাঙ্ক kernel_w, ধ্রুব পূর্ণাঙ্ক kernel_h,
                                   ধ্রুব পূর্ণাঙ্ক stride,
                                   ধ্রুব পূর্ণাঙ্ক c_in, ধ্রুব পূর্ণাঙ্ক c_out, ধ্রুব পূর্ণাঙ্ক batches,
                                   ধ্রুব queue_ptr & stream) {
    ধ্রুব পূর্ণাঙ্ক total      = out_w * out_h * c_out * batches;
    ধ্রুব পূর্ণাঙ্ক অঙ্ক_blocks = (total + SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE;
    ধ্রুব sycl::range<3> block_dims(1, 1, SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE);
    ধ্রুব sycl::range<3> block_nums(1, 1, অঙ্ক_blocks);
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

    ধ্রুব ggml_tensor * kernel = dst->src[0];
    ধ্রুব ggml_tensor * প্রবেশ  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(প্রবেশ->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(প্রবেশ));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ধ্রুব ভিন্ন * প্রবেশ_d  = (ধ্রুব ভিন্ন *) প্রবেশ->data;
    ভিন্ন *       output_d = (ভিন্ন *) dst->data;
    ধ্রুব শূন্য *  kernel_d = kernel->data;

    ধ্রুব পূর্ণাঙ্ক প্রবেশ_w      = প্রবেশ->ne[0];
    ধ্রুব পূর্ণাঙ্ক প্রবেশ_h      = প্রবেশ->ne[1];
    ধ্রুব পূর্ণাঙ্ক channels_in  = প্রবেশ->ne[2];
    ধ্রুব পূর্ণাঙ্ক batches      = প্রবেশ->ne[3];
    ধ্রুব পূর্ণাঙ্ক output_w     = dst->ne[0];
    ধ্রুব পূর্ণাঙ্ক output_h     = dst->ne[1];
    ধ্রুব পূর্ণাঙ্ক channels_out = kernel->ne[2];
    ধ্রুব পূর্ণাঙ্ক kernel_w     = kernel->ne[0];
    ধ্রুব পূর্ণাঙ্ক kernel_h     = kernel->ne[1];
    ধ্রুব পূর্ণাঙ্ক stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ধ্রুব queue_ptr stream = ctx.stream();

    যদি (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(প্রবেশ_d, (ধ্রুব sycl::half *) kernel_d, output_d,
                                          প্রবেশ_w, প্রবেশ_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } অন্যথা {
        conv2d_transpose_sycl<ভিন্ন>(প্রবেশ_d, (ধ্রুব ভিন্ন *) kernel_d, output_d,
                                     প্রবেশ_w, প্রবেশ_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
