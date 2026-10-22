#समावेश "conv2d-transpose.hpp"
#समावेश "convert.hpp"

ढाँचा <typename kernel_t>
अटल व्योम conv2d_transpose_kernel(स्थिर भग्न * निवेश, स्थिर kernel_t * kernel, भग्न * output,
                                    स्थिर पूर्णांक in_w, स्थिर पूर्णांक in_h,
                                    स्थिर पूर्णांक out_w, स्थिर पूर्णांक out_h,
                                    स्थिर पूर्णांक kernel_w, स्थिर पूर्णांक kernel_h,
                                    स्थिर पूर्णांक stride,
                                    स्थिर पूर्णांक c_in, स्थिर पूर्णांक c_out, स्थिर पूर्णांक batches,
                                    स्थिर sycl::nd_item<3> & item_ct1) {
    स्थिर पूर्णांक global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    स्थिर पूर्णांक योग_elements = out_w * out_h * c_out * batches;

    अगर (global_idx >= योग_elements) {
        वापस;
    }

    स्थिर पूर्णांक out_x = global_idx % out_w;
    स्थिर पूर्णांक out_y = (global_idx / out_w) % out_h;
    स्थिर पूर्णांक c_idx = (global_idx / (out_w * out_h)) % c_out;
    स्थिर पूर्णांक n_idx = global_idx / (out_w * out_h * c_out);

    भग्न acc = 0.0f;

    क्रम (पूर्णांक c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        क्रम (पूर्णांक kh = 0; kh < kernel_h; ++kh) {
            पूर्णांक in_y = out_y - kh;
            अगर (in_y < 0 || in_y % stride) {
                जारी;
            }
            in_y /= stride;
            अगर (in_y >= in_h) {
                जारी;
            }

            क्रम (पूर्णांक kw = 0; kw < kernel_w; ++kw) {
                पूर्णांक in_x = out_x - kw;
                अगर (in_x < 0 || in_x % stride) {
                    जारी;
                }
                in_x /= stride;
                अगर (in_x >= in_w) {
                    जारी;
                }

                स्थिर पूर्णांक निवेश_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                स्थिर पूर्णांक kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += निवेश[निवेश_idx] * ggml_sycl_cast<भग्न>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

ढाँचा <typename kernel_t>
अटल व्योम conv2d_transpose_sycl(स्थिर भग्न * निवेश_d, स्थिर kernel_t * kernel_d, भग्न * output_d,
                                   स्थिर पूर्णांक in_w, स्थिर पूर्णांक in_h,
                                   स्थिर पूर्णांक out_w, स्थिर पूर्णांक out_h,
                                   स्थिर पूर्णांक kernel_w, स्थिर पूर्णांक kernel_h,
                                   स्थिर पूर्णांक stride,
                                   स्थिर पूर्णांक c_in, स्थिर पूर्णांक c_out, स्थिर पूर्णांक batches,
                                   स्थिर queue_ptr & stream) {
    स्थिर पूर्णांक योग      = out_w * out_h * c_out * batches;
    स्थिर पूर्णांक अंक_blocks = (योग + SYCL_CONV2_D_TRANSPOSE_BLOCK_आकार - 1) / SYCL_CONV2_D_TRANSPOSE_BLOCK_आकार;
    स्थिर sycl::range<3> block_dims(1, 1, SYCL_CONV2_D_TRANSPOSE_BLOCK_आकार);
    स्थिर sycl::range<3> block_nums(1, 1, अंक_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_transpose_kernel<kernel_t>(निवेश_d, kernel_d, output_d,
                                             in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                             stride, c_in, c_out, batches, item_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
व्योम ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    स्थिर ggml_tensor * kernel = dst->src[0];
    स्थिर ggml_tensor * निवेश  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(निवेश->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(निवेश));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    स्थिर भग्न * निवेश_d  = (स्थिर भग्न *) निवेश->तथ्य;
    भग्न *       output_d = (भग्न *) dst->तथ्य;
    स्थिर व्योम *  kernel_d = kernel->तथ्य;

    स्थिर पूर्णांक निवेश_w      = निवेश->ne[0];
    स्थिर पूर्णांक निवेश_h      = निवेश->ne[1];
    स्थिर पूर्णांक channels_in  = निवेश->ne[2];
    स्थिर पूर्णांक batches      = निवेश->ne[3];
    स्थिर पूर्णांक output_w     = dst->ne[0];
    स्थिर पूर्णांक output_h     = dst->ne[1];
    स्थिर पूर्णांक channels_out = kernel->ne[2];
    स्थिर पूर्णांक kernel_w     = kernel->ne[0];
    स्थिर पूर्णांक kernel_h     = kernel->ne[1];
    स्थिर पूर्णांक stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    स्थिर queue_ptr stream = ctx.stream();

    अगर (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(निवेश_d, (स्थिर sycl::half *) kernel_d, output_d,
                                          निवेश_w, निवेश_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } अन्यथा {
        conv2d_transpose_sycl<भग्न>(निवेश_d, (स्थिर भग्न *) kernel_d, output_d,
                                     निवेश_w, निवेश_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
