#अन्तर्भाव "conv2d-transpose.hpp"
#अन्तर्भाव "convert.hpp"

ढाँचा <typename kernel_t>
स्थिर शून्य conv2d_transpose_kernel(अचर भिन्न * आगत, अचर kernel_t * kernel, भिन्न * output,
                                    अचर पूर्णाङ्क in_w, अचर पूर्णाङ्क in_h,
                                    अचर पूर्णाङ्क out_w, अचर पूर्णाङ्क out_h,
                                    अचर पूर्णाङ्क kernel_w, अचर पूर्णाङ्क kernel_h,
                                    अचर पूर्णाङ्क stride,
                                    अचर पूर्णाङ्क c_in, अचर पूर्णाङ्क c_out, अचर पूर्णाङ्क batches,
                                    अचर sycl::nd_item<3> & item_ct1) {
    अचर पूर्णाङ्क global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    अचर पूर्णाङ्क total_elements = out_w * out_h * c_out * batches;

    यदि (global_idx >= total_elements) {
        फर्काऊ;
    }

    अचर पूर्णाङ्क out_x = global_idx % out_w;
    अचर पूर्णाङ्क out_y = (global_idx / out_w) % out_h;
    अचर पूर्णाङ्क c_idx = (global_idx / (out_w * out_h)) % c_out;
    अचर पूर्णाङ्क n_idx = global_idx / (out_w * out_h * c_out);

    भिन्न acc = 0.0f;

    लागि (पूर्णाङ्क c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        लागि (पूर्णाङ्क kh = 0; kh < kernel_h; ++kh) {
            पूर्णाङ्क in_y = out_y - kh;
            यदि (in_y < 0 || in_y % stride) {
                जारी;
            }
            in_y /= stride;
            यदि (in_y >= in_h) {
                जारी;
            }

            लागि (पूर्णाङ्क kw = 0; kw < kernel_w; ++kw) {
                पूर्णाङ्क in_x = out_x - kw;
                यदि (in_x < 0 || in_x % stride) {
                    जारी;
                }
                in_x /= stride;
                यदि (in_x >= in_w) {
                    जारी;
                }

                अचर पूर्णाङ्क आगत_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                अचर पूर्णाङ्क kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += आगत[आगत_idx] * ggml_sycl_cast<भिन्न>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

ढाँचा <typename kernel_t>
स्थिर शून्य conv2d_transpose_sycl(अचर भिन्न * आगत_d, अचर kernel_t * kernel_d, भिन्न * output_d,
                                   अचर पूर्णाङ्क in_w, अचर पूर्णाङ्क in_h,
                                   अचर पूर्णाङ्क out_w, अचर पूर्णाङ्क out_h,
                                   अचर पूर्णाङ्क kernel_w, अचर पूर्णाङ्क kernel_h,
                                   अचर पूर्णाङ्क stride,
                                   अचर पूर्णाङ्क c_in, अचर पूर्णाङ्क c_out, अचर पूर्णाङ्क batches,
                                   अचर queue_ptr & stream) {
    अचर पूर्णाङ्क total      = out_w * out_h * c_out * batches;
    अचर पूर्णाङ्क अङ्क_blocks = (total + SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE;
    अचर sycl::range<3> block_dims(1, 1, SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE);
    अचर sycl::range<3> block_nums(1, 1, अङ्क_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_transpose_kernel<kernel_t>(आगत_d, kernel_d, output_d,
                                             in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                             stride, c_in, c_out, batches, item_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
शून्य ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    अचर ggml_tensor * kernel = dst->src[0];
    अचर ggml_tensor * आगत  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(आगत->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(आगत));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    अचर भिन्न * आगत_d  = (अचर भिन्न *) आगत->data;
    भिन्न *       output_d = (भिन्न *) dst->data;
    अचर शून्य *  kernel_d = kernel->data;

    अचर पूर्णाङ्क आगत_w      = आगत->ne[0];
    अचर पूर्णाङ्क आगत_h      = आगत->ne[1];
    अचर पूर्णाङ्क channels_in  = आगत->ne[2];
    अचर पूर्णाङ्क batches      = आगत->ne[3];
    अचर पूर्णाङ्क output_w     = dst->ne[0];
    अचर पूर्णाङ्क output_h     = dst->ne[1];
    अचर पूर्णाङ्क channels_out = kernel->ne[2];
    अचर पूर्णाङ्क kernel_w     = kernel->ne[0];
    अचर पूर्णाङ्क kernel_h     = kernel->ne[1];
    अचर पूर्णाङ्क stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    अचर queue_ptr stream = ctx.stream();

    यदि (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(आगत_d, (अचर sycl::half *) kernel_d, output_d,
                                          आगत_w, आगत_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } अन्यथा {
        conv2d_transpose_sycl<भिन्न>(आगत_d, (अचर भिन्न *) kernel_d, output_d,
                                     आगत_w, आगत_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
