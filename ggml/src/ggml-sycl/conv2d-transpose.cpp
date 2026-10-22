#அந்தர்பாவ "conv2d-transpose.hpp"
#அந்தர்பாவ "convert.hpp"

template <typename kernel_t>
நிலையான வெறுமை conv2d_transpose_kernel(மாறிலி தசம * உள்ளீடு, மாறிலி kernel_t * kernel, தசம * output,
                                    மாறிலி முழு in_w, மாறிலி முழு in_h,
                                    மாறிலி முழு out_w, மாறிலி முழு out_h,
                                    மாறிலி முழு kernel_w, மாறிலி முழு kernel_h,
                                    மாறிலி முழு stride,
                                    மாறிலி முழு c_in, மாறிலி முழு c_out, மாறிலி முழு batches,
                                    மாறிலி sycl::nd_item<3> & item_ct1) {
    மாறிலி முழு global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    மாறிலி முழு total_elements = out_w * out_h * c_out * batches;

    எனில் (global_idx >= total_elements) {
        திரும்பு;
    }

    மாறிலி முழு out_x = global_idx % out_w;
    மாறிலி முழு out_y = (global_idx / out_w) % out_h;
    மாறிலி முழு c_idx = (global_idx / (out_w * out_h)) % c_out;
    மாறிலி முழு n_idx = global_idx / (out_w * out_h * c_out);

    தசம acc = 0.0f;

    க்கு (முழு c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        க்கு (முழு kh = 0; kh < kernel_h; ++kh) {
            முழு in_y = out_y - kh;
            எனில் (in_y < 0 || in_y % stride) {
                தொடர்;
            }
            in_y /= stride;
            எனில் (in_y >= in_h) {
                தொடர்;
            }

            க்கு (முழு kw = 0; kw < kernel_w; ++kw) {
                முழு in_x = out_x - kw;
                எனில் (in_x < 0 || in_x % stride) {
                    தொடர்;
                }
                in_x /= stride;
                எனில் (in_x >= in_w) {
                    தொடர்;
                }

                மாறிலி முழு உள்ளீடு_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                மாறிலி முழு kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += உள்ளீடு[உள்ளீடு_idx] * ggml_sycl_cast<தசம>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

template <typename kernel_t>
நிலையான வெறுமை conv2d_transpose_sycl(மாறிலி தசம * உள்ளீடு_d, மாறிலி kernel_t * kernel_d, தசம * output_d,
                                   மாறிலி முழு in_w, மாறிலி முழு in_h,
                                   மாறிலி முழு out_w, மாறிலி முழு out_h,
                                   மாறிலி முழு kernel_w, மாறிலி முழு kernel_h,
                                   மாறிலி முழு stride,
                                   மாறிலி முழு c_in, மாறிலி முழு c_out, மாறிலி முழு batches,
                                   மாறிலி queue_ptr & stream) {
    மாறிலி முழு total      = out_w * out_h * c_out * batches;
    மாறிலி முழு இலக்கம்_blocks = (total + SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE;
    மாறிலி sycl::range<3> block_dims(1, 1, SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE);
    மாறிலி sycl::range<3> block_nums(1, 1, இலக்கம்_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_transpose_kernel<kernel_t>(உள்ளீடு_d, kernel_d, output_d,
                                             in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                             stride, c_in, c_out, batches, item_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
வெறுமை ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    மாறிலி ggml_tensor * kernel = dst->src[0];
    மாறிலி ggml_tensor * உள்ளீடு  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(உள்ளீடு->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(உள்ளீடு));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    மாறிலி தசம * உள்ளீடு_d  = (மாறிலி தசம *) உள்ளீடு->data;
    தசம *       output_d = (தசம *) dst->data;
    மாறிலி வெறுமை *  kernel_d = kernel->data;

    மாறிலி முழு உள்ளீடு_w      = உள்ளீடு->ne[0];
    மாறிலி முழு உள்ளீடு_h      = உள்ளீடு->ne[1];
    மாறிலி முழு channels_in  = உள்ளீடு->ne[2];
    மாறிலி முழு batches      = உள்ளீடு->ne[3];
    மாறிலி முழு output_w     = dst->ne[0];
    மாறிலி முழு output_h     = dst->ne[1];
    மாறிலி முழு channels_out = kernel->ne[2];
    மாறிலி முழு kernel_w     = kernel->ne[0];
    மாறிலி முழு kernel_h     = kernel->ne[1];
    மாறிலி முழு stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    மாறிலி queue_ptr stream = ctx.stream();

    எனில் (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(உள்ளீடு_d, (மாறிலி sycl::half *) kernel_d, output_d,
                                          உள்ளீடு_w, உள்ளீடு_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } இல்லையெனில் {
        conv2d_transpose_sycl<தசம>(உள்ளீடு_d, (மாறிலி தசம *) kernel_d, output_d,
                                     உள்ளீடு_w, உள்ளீடு_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
