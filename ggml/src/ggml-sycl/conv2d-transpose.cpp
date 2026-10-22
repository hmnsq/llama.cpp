#ಅನ್ತರ್ಭಾವ "conv2d-transpose.hpp"
#ಅನ್ತರ್ಭಾವ "convert.hpp"

template <typename kernel_t>
ಸ್ಥಿರ ಶೂನ್ಯ conv2d_transpose_kernel(ಧ್ರುವ ಭಿನ್ನ * ಒಳಹರಿವು, ಧ್ರುವ kernel_t * kernel, ಭಿನ್ನ * output,
                                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ in_w, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ in_h,
                                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ out_w, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ out_h,
                                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kernel_w, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kernel_h,
                                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ stride,
                                    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ c_in, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ c_out, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ batches,
                                    ಧ್ರುವ sycl::nd_item<3> & item_ct1) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ total_elements = out_w * out_h * c_out * batches;

    ಯದಿ (global_idx >= total_elements) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ out_x = global_idx % out_w;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ out_y = (global_idx / out_w) % out_h;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ c_idx = (global_idx / (out_w * out_h)) % c_out;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n_idx = global_idx / (out_w * out_h * c_out);

    ಭಿನ್ನ acc = 0.0f;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ kh = 0; kh < kernel_h; ++kh) {
            ಪೂರ್ಣಾಙ್ಕ in_y = out_y - kh;
            ಯದಿ (in_y < 0 || in_y % stride) {
                ಅನುವರ್ತ;
            }
            in_y /= stride;
            ಯದಿ (in_y >= in_h) {
                ಅನುವರ್ತ;
            }

            ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ kw = 0; kw < kernel_w; ++kw) {
                ಪೂರ್ಣಾಙ್ಕ in_x = out_x - kw;
                ಯದಿ (in_x < 0 || in_x % stride) {
                    ಅನುವರ್ತ;
                }
                in_x /= stride;
                ಯದಿ (in_x >= in_w) {
                    ಅನುವರ್ತ;
                }

                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += ಒಳಹರಿವು[ಒಳಹರಿವು_idx] * ggml_sycl_cast<ಭಿನ್ನ>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

template <typename kernel_t>
ಸ್ಥಿರ ಶೂನ್ಯ conv2d_transpose_sycl(ಧ್ರುವ ಭಿನ್ನ * ಒಳಹರಿವು_d, ಧ್ರುವ kernel_t * kernel_d, ಭಿನ್ನ * output_d,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ in_w, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ in_h,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ out_w, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ out_h,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kernel_w, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kernel_h,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ stride,
                                   ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ c_in, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ c_out, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ batches,
                                   ಧ್ರುವ queue_ptr & stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ total      = out_w * out_h * c_out * batches;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (total + SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE;
    ಧ್ರುವ sycl::range<3> block_dims(1, 1, SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE);
    ಧ್ರುವ sycl::range<3> block_nums(1, 1, ಅಂಕೆ_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_transpose_kernel<kernel_t>(ಒಳಹರಿವು_d, kernel_d, output_d,
                                             in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                             stride, c_in, c_out, batches, item_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
ಶೂನ್ಯ ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ಧ್ರುವ ggml_tensor * kernel = dst->src[0];
    ಧ್ರುವ ggml_tensor * ಒಳಹರಿವು  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(ಒಳಹರಿವು->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(ಒಳಹರಿವು));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ಧ್ರುವ ಭಿನ್ನ * ಒಳಹರಿವು_d  = (ಧ್ರುವ ಭಿನ್ನ *) ಒಳಹರಿವು->data;
    ಭಿನ್ನ *       output_d = (ಭಿನ್ನ *) dst->data;
    ಧ್ರುವ ಶೂನ್ಯ *  kernel_d = kernel->data;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_w      = ಒಳಹರಿವು->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_h      = ಒಳಹರಿವು->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ channels_in  = ಒಳಹರಿವು->ne[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ batches      = ಒಳಹರಿವು->ne[3];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ output_w     = dst->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ output_h     = dst->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ channels_out = kernel->ne[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kernel_w     = kernel->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kernel_h     = kernel->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ಧ್ರುವ queue_ptr stream = ctx.stream();

    ಯದಿ (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(ಒಳಹರಿವು_d, (ಧ್ರುವ sycl::half *) kernel_d, output_d,
                                          ಒಳಹರಿವು_w, ಒಳಹರಿವು_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } ಅನ್ಯಥಾ {
        conv2d_transpose_sycl<ಭಿನ್ನ>(ಒಳಹರಿವು_d, (ಧ್ರುವ ಭಿನ್ನ *) kernel_d, output_d,
                                     ಒಳಹರಿವು_w, ಒಳಹರಿವು_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
