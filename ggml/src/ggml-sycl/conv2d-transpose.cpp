#ଅନ୍ତର୍ଭାଵ "conv2d-transpose.hpp"
#ଅନ୍ତର୍ଭାଵ "convert.hpp"

template <typename kernel_t>
ସ୍ଥିର ଶୂନ୍ୟ conv2d_transpose_kernel(ଅଚଳ ଦଶମିକ * ନିବେଶ, ଅଚଳ kernel_t * kernel, ଦଶମିକ * output,
                                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା in_w, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା in_h,
                                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_w, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_h,
                                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_w, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_h,
                                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride,
                                    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା c_in, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା c_out, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା batches,
                                    ଅଚଳ sycl::nd_item<3> & item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା total_elements = out_w * out_h * c_out * batches;

    ଯଦି (global_idx >= total_elements) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_x = global_idx % out_w;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_y = (global_idx / out_w) % out_h;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା c_idx = (global_idx / (out_w * out_h)) % c_out;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_idx = global_idx / (out_w * out_h * c_out);

    ଦଶମିକ acc = 0.0f;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kh = 0; kh < kernel_h; ++kh) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା in_y = out_y - kh;
            ଯଦି (in_y < 0 || in_y % stride) {
                ଜାରି;
            }
            in_y /= stride;
            ଯଦି (in_y >= in_h) {
                ଜାରି;
            }

            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kw = 0; kw < kernel_w; ++kw) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା in_x = out_x - kw;
                ଯଦି (in_x < 0 || in_x % stride) {
                    ଜାରି;
                }
                in_x /= stride;
                ଯଦି (in_x >= in_w) {
                    ଜାରି;
                }

                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += ନିବେଶ[ନିବେଶ_idx] * ggml_sycl_cast<ଦଶମିକ>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

template <typename kernel_t>
ସ୍ଥିର ଶୂନ୍ୟ conv2d_transpose_sycl(ଅଚଳ ଦଶମିକ * ନିବେଶ_d, ଅଚଳ kernel_t * kernel_d, ଦଶମିକ * output_d,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା in_w, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା in_h,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_w, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_h,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_w, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_h,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride,
                                   ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା c_in, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା c_out, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା batches,
                                   ଅଚଳ queue_ptr & stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା total      = out_w * out_h * c_out * batches;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (total + SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE;
    ଅଚଳ sycl::range<3> block_dims(1, 1, SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE);
    ଅଚଳ sycl::range<3> block_nums(1, 1, ଅଙ୍କ_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_transpose_kernel<kernel_t>(ନିବେଶ_d, kernel_d, output_d,
                                             in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                             stride, c_in, c_out, batches, item_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
ଶୂନ୍ୟ ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ଅଚଳ ggml_tensor * kernel = dst->src[0];
    ଅଚଳ ggml_tensor * ନିବେଶ  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(ନିବେଶ->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(ନିବେଶ));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ଅଚଳ ଦଶମିକ * ନିବେଶ_d  = (ଅଚଳ ଦଶମିକ *) ନିବେଶ->data;
    ଦଶମିକ *       output_d = (ଦଶମିକ *) dst->data;
    ଅଚଳ ଶୂନ୍ୟ *  kernel_d = kernel->data;

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_w      = ନିବେଶ->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_h      = ନିବେଶ->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା channels_in  = ନିବେଶ->ne[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା batches      = ନିବେଶ->ne[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା output_w     = dst->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା output_h     = dst->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା channels_out = kernel->ne[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_w     = kernel->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_h     = kernel->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ଅଚଳ queue_ptr stream = ctx.stream();

    ଯଦି (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(ନିବେଶ_d, (ଅଚଳ sycl::half *) kernel_d, output_d,
                                          ନିବେଶ_w, ନିବେଶ_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } ଅନ୍ୟଥା {
        conv2d_transpose_sycl<ଦଶମିକ>(ନିବେଶ_d, (ଅଚଳ ଦଶମିକ *) kernel_d, output_d,
                                     ନିବେଶ_w, ନିବେଶ_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
