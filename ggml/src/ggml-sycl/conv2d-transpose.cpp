#अन्तर्भाव "conv2d-transpose.hpp"
#अन्तर्भाव "convert.hpp"

ढाँचा <typename kernel_t>
स्थिर शून्य conv2d_transpose_kernel(स्थिरांक दशांश * आदान, स्थिरांक kernel_t * kernel, दशांश * output,
                                    स्थिरांक पूर्णांक in_w, स्थिरांक पूर्णांक in_h,
                                    स्थिरांक पूर्णांक out_w, स्थिरांक पूर्णांक out_h,
                                    स्थिरांक पूर्णांक kernel_w, स्थिरांक पूर्णांक kernel_h,
                                    स्थिरांक पूर्णांक stride,
                                    स्थिरांक पूर्णांक c_in, स्थिरांक पूर्णांक c_out, स्थिरांक पूर्णांक batches,
                                    स्थिरांक sycl::nd_item<3> & item_ct1) {
    स्थिरांक पूर्णांक global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    स्थिरांक पूर्णांक total_elements = out_w * out_h * c_out * batches;

    जर (global_idx >= total_elements) {
        परत;
    }

    स्थिरांक पूर्णांक out_x = global_idx % out_w;
    स्थिरांक पूर्णांक out_y = (global_idx / out_w) % out_h;
    स्थिरांक पूर्णांक c_idx = (global_idx / (out_w * out_h)) % c_out;
    स्थिरांक पूर्णांक n_idx = global_idx / (out_w * out_h * c_out);

    दशांश acc = 0.0f;

    साठी (पूर्णांक c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        साठी (पूर्णांक kh = 0; kh < kernel_h; ++kh) {
            पूर्णांक in_y = out_y - kh;
            जर (in_y < 0 || in_y % stride) {
                सुरूठेवा;
            }
            in_y /= stride;
            जर (in_y >= in_h) {
                सुरूठेवा;
            }

            साठी (पूर्णांक kw = 0; kw < kernel_w; ++kw) {
                पूर्णांक in_x = out_x - kw;
                जर (in_x < 0 || in_x % stride) {
                    सुरूठेवा;
                }
                in_x /= stride;
                जर (in_x >= in_w) {
                    सुरूठेवा;
                }

                स्थिरांक पूर्णांक आदान_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                स्थिरांक पूर्णांक kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += आदान[आदान_idx] * ggml_sycl_cast<दशांश>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

ढाँचा <typename kernel_t>
स्थिर शून्य conv2d_transpose_sycl(स्थिरांक दशांश * आदान_d, स्थिरांक kernel_t * kernel_d, दशांश * output_d,
                                   स्थिरांक पूर्णांक in_w, स्थिरांक पूर्णांक in_h,
                                   स्थिरांक पूर्णांक out_w, स्थिरांक पूर्णांक out_h,
                                   स्थिरांक पूर्णांक kernel_w, स्थिरांक पूर्णांक kernel_h,
                                   स्थिरांक पूर्णांक stride,
                                   स्थिरांक पूर्णांक c_in, स्थिरांक पूर्णांक c_out, स्थिरांक पूर्णांक batches,
                                   स्थिरांक queue_ptr & stream) {
    स्थिरांक पूर्णांक total      = out_w * out_h * c_out * batches;
    स्थिरांक पूर्णांक अंक_blocks = (total + SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE;
    स्थिरांक sycl::range<3> block_dims(1, 1, SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE);
    स्थिरांक sycl::range<3> block_nums(1, 1, अंक_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_transpose_kernel<kernel_t>(आदान_d, kernel_d, output_d,
                                             in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                             stride, c_in, c_out, batches, item_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
शून्य ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    स्थिरांक ggml_tensor * kernel = dst->src[0];
    स्थिरांक ggml_tensor * आदान  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(आदान->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(आदान));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    स्थिरांक दशांश * आदान_d  = (स्थिरांक दशांश *) आदान->data;
    दशांश *       output_d = (दशांश *) dst->data;
    स्थिरांक शून्य *  kernel_d = kernel->data;

    स्थिरांक पूर्णांक आदान_w      = आदान->ne[0];
    स्थिरांक पूर्णांक आदान_h      = आदान->ne[1];
    स्थिरांक पूर्णांक channels_in  = आदान->ne[2];
    स्थिरांक पूर्णांक batches      = आदान->ne[3];
    स्थिरांक पूर्णांक output_w     = dst->ne[0];
    स्थिरांक पूर्णांक output_h     = dst->ne[1];
    स्थिरांक पूर्णांक channels_out = kernel->ne[2];
    स्थिरांक पूर्णांक kernel_w     = kernel->ne[0];
    स्थिरांक पूर्णांक kernel_h     = kernel->ne[1];
    स्थिरांक पूर्णांक stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    स्थिरांक queue_ptr stream = ctx.stream();

    जर (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(आदान_d, (स्थिरांक sycl::half *) kernel_d, output_d,
                                          आदान_w, आदान_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } अन्यथा {
        conv2d_transpose_sycl<दशांश>(आदान_d, (स्थिरांक दशांश *) kernel_d, output_d,
                                     आदान_w, आदान_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
