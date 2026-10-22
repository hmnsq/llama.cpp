#अन्तर्भाव "conv2d-transpose.hpp"
#अन्तर्भाव "convert.hpp"

template <typename kernel_t>
स्थिर शून्य conv2d_transpose_kernel(ध्रुव भिन्न * निवेशः, ध्रुव kernel_t * kernel, भिन्न * output,
                                    ध्रुव पूर्णाङ्क in_w, ध्रुव पूर्णाङ्क in_h,
                                    ध्रुव पूर्णाङ्क out_w, ध्रुव पूर्णाङ्क out_h,
                                    ध्रुव पूर्णाङ्क kernel_w, ध्रुव पूर्णाङ्क kernel_h,
                                    ध्रुव पूर्णाङ्क stride,
                                    ध्रुव पूर्णाङ्क c_in, ध्रुव पूर्णाङ्क c_out, ध्रुव पूर्णाङ्क batches,
                                    ध्रुव sycl::nd_item<3> & item_ct1) {
    ध्रुव पूर्णाङ्क global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ध्रुव पूर्णाङ्क total_elements = out_w * out_h * c_out * batches;

    सचे (global_idx >= total_elements) {
        पटिनिवत्त;
    }

    ध्रुव पूर्णाङ्क out_x = global_idx % out_w;
    ध्रुव पूर्णाङ्क out_y = (global_idx / out_w) % out_h;
    ध्रुव पूर्णाङ्क c_idx = (global_idx / (out_w * out_h)) % c_out;
    ध्रुव पूर्णाङ्क n_idx = global_idx / (out_w * out_h * c_out);

    भिन्न acc = 0.0f;

    अत्थाय (पूर्णाङ्क c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        अत्थाय (पूर्णाङ्क kh = 0; kh < kernel_h; ++kh) {
            पूर्णाङ्क in_y = out_y - kh;
            सचे (in_y < 0 || in_y % stride) {
                अनुवर्त;
            }
            in_y /= stride;
            सचे (in_y >= in_h) {
                अनुवर्त;
            }

            अत्थाय (पूर्णाङ्क kw = 0; kw < kernel_w; ++kw) {
                पूर्णाङ्क in_x = out_x - kw;
                सचे (in_x < 0 || in_x % stride) {
                    अनुवर्त;
                }
                in_x /= stride;
                सचे (in_x >= in_w) {
                    अनुवर्त;
                }

                ध्रुव पूर्णाङ्क निवेशः_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                ध्रुव पूर्णाङ्क kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += निवेशः[निवेशः_idx] * ggml_sycl_cast<भिन्न>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

template <typename kernel_t>
स्थिर शून्य conv2d_transpose_sycl(ध्रुव भिन्न * निवेशः_d, ध्रुव kernel_t * kernel_d, भिन्न * output_d,
                                   ध्रुव पूर्णाङ्क in_w, ध्रुव पूर्णाङ्क in_h,
                                   ध्रुव पूर्णाङ्क out_w, ध्रुव पूर्णाङ्क out_h,
                                   ध्रुव पूर्णाङ्क kernel_w, ध्रुव पूर्णाङ्क kernel_h,
                                   ध्रुव पूर्णाङ्क stride,
                                   ध्रुव पूर्णाङ्क c_in, ध्रुव पूर्णाङ्क c_out, ध्रुव पूर्णाङ्क batches,
                                   ध्रुव queue_ptr & stream) {
    ध्रुव पूर्णाङ्क total      = out_w * out_h * c_out * batches;
    ध्रुव पूर्णाङ्क अङ्कः_blocks = (total + SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE;
    ध्रुव sycl::range<3> block_dims(1, 1, SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE);
    ध्रुव sycl::range<3> block_nums(1, 1, अङ्कः_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_transpose_kernel<kernel_t>(निवेशः_d, kernel_d, output_d,
                                             in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                             stride, c_in, c_out, batches, item_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
शून्य ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ध्रुव ggml_tensor * kernel = dst->src[0];
    ध्रुव ggml_tensor * निवेशः  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(निवेशः->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(निवेशः));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ध्रुव भिन्न * निवेशः_d  = (ध्रुव भिन्न *) निवेशः->data;
    भिन्न *       output_d = (भिन्न *) dst->data;
    ध्रुव शून्य *  kernel_d = kernel->data;

    ध्रुव पूर्णाङ्क निवेशः_w      = निवेशः->ne[0];
    ध्रुव पूर्णाङ्क निवेशः_h      = निवेशः->ne[1];
    ध्रुव पूर्णाङ्क channels_in  = निवेशः->ne[2];
    ध्रुव पूर्णाङ्क batches      = निवेशः->ne[3];
    ध्रुव पूर्णाङ्क output_w     = dst->ne[0];
    ध्रुव पूर्णाङ्क output_h     = dst->ne[1];
    ध्रुव पूर्णाङ्क channels_out = kernel->ne[2];
    ध्रुव पूर्णाङ्क kernel_w     = kernel->ne[0];
    ध्रुव पूर्णाङ्क kernel_h     = kernel->ne[1];
    ध्रुव पूर्णाङ्क stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ध्रुव queue_ptr stream = ctx.stream();

    सचे (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(निवेशः_d, (ध्रुव sycl::half *) kernel_d, output_d,
                                          निवेशः_w, निवेशः_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } अञ्ञथा {
        conv2d_transpose_sycl<भिन्न>(निवेशः_d, (ध्रुव भिन्न *) kernel_d, output_d,
                                     निवेशः_w, निवेशः_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
