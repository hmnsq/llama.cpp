#अन्तर्भाव "conv2d-transpose.hpp"
#अन्तर्भाव "convert.hpp"

प्रतिरूपम् <प्रकारनाम kernel_t>
स्थिर शून्य conv2d_transpose_kernel(ध्रुव भिन्न * निवेशः, ध्रुव kernel_t * kernel, भिन्न * output,
                                    ध्रुव पूर्णाङ्क अन्तर्_w, ध्रुव पूर्णाङ्क अन्तर्_h,
                                    ध्रुव पूर्णाङ्क बहिर्गमनम्_w, ध्रुव पूर्णाङ्क बहिर्गमनम्_h,
                                    ध्रुव पूर्णाङ्क kernel_w, ध्रुव पूर्णाङ्क kernel_h,
                                    ध्रुव पूर्णाङ्क stride,
                                    ध्रुव पूर्णाङ्क c_अन्तर्, ध्रुव पूर्णाङ्क c_बहिर्गमनम्, ध्रुव पूर्णाङ्क batches,
                                    ध्रुव sycl::nd_item<3> & item_ct1) {
    ध्रुव पूर्णाङ्क global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_गणः(2) * item_ct1.get_local_range(2);
    ध्रुव पूर्णाङ्क योग_elements = बहिर्गमनम्_w * बहिर्गमनम्_h * c_बहिर्गमनम् * batches;

    यदि (global_idx >= योग_elements) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क बहिर्गमनम्_x = global_idx % बहिर्गमनम्_w;
    ध्रुव पूर्णाङ्क बहिर्गमनम्_y = (global_idx / बहिर्गमनम्_w) % बहिर्गमनम्_h;
    ध्रुव पूर्णाङ्क c_idx = (global_idx / (बहिर्गमनम्_w * बहिर्गमनम्_h)) % c_बहिर्गमनम्;
    ध्रुव पूर्णाङ्क n_idx = global_idx / (बहिर्गमनम्_w * बहिर्गमनम्_h * c_बहिर्गमनम्);

    भिन्न acc = 0.0f;

    प्रति (पूर्णाङ्क c_अन्तर्_idx = 0; c_अन्तर्_idx < c_अन्तर्; ++c_अन्तर्_idx) {
        प्रति (पूर्णाङ्क kh = 0; kh < kernel_h; ++kh) {
            पूर्णाङ्क अन्तर्_y = बहिर्गमनम्_y - kh;
            यदि (अन्तर्_y < 0 || अन्तर्_y % stride) {
                अनुवर्त;
            }
            अन्तर्_y /= stride;
            यदि (अन्तर्_y >= अन्तर्_h) {
                अनुवर्त;
            }

            प्रति (पूर्णाङ्क kw = 0; kw < kernel_w; ++kw) {
                पूर्णाङ्क अन्तर्_x = बहिर्गमनम्_x - kw;
                यदि (अन्तर्_x < 0 || अन्तर्_x % stride) {
                    अनुवर्त;
                }
                अन्तर्_x /= stride;
                यदि (अन्तर्_x >= अन्तर्_w) {
                    अनुवर्त;
                }

                ध्रुव पूर्णाङ्क निवेशः_idx  = (अन्तर्_w * अन्तर्_h * c_अन्तर्) * n_idx + (अन्तर्_w * अन्तर्_h) * c_अन्तर्_idx + अन्तर्_w * अन्तर्_y + अन्तर्_x;
                ध्रुव पूर्णाङ्क kernel_idx = (kernel_h * kernel_w * c_बहिर्गमनम्) * c_अन्तर्_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += निवेशः[निवेशः_idx] * ggml_sycl_cast<भिन्न>(kernel[kernel_idx]);
            }
        }
    }

    output[(बहिर्गमनम्_w * बहिर्गमनम्_h * c_बहिर्गमनम्) * n_idx + (बहिर्गमनम्_w * बहिर्गमनम्_h) * c_idx + बहिर्गमनम्_w * बहिर्गमनम्_y + बहिर्गमनम्_x] = acc;
}

प्रतिरूपम् <प्रकारनाम kernel_t>
स्थिर शून्य conv2d_transpose_sycl(ध्रुव भिन्न * निवेशः_d, ध्रुव kernel_t * kernel_d, भिन्न * output_d,
                                   ध्रुव पूर्णाङ्क अन्तर्_w, ध्रुव पूर्णाङ्क अन्तर्_h,
                                   ध्रुव पूर्णाङ्क बहिर्गमनम्_w, ध्रुव पूर्णाङ्क बहिर्गमनम्_h,
                                   ध्रुव पूर्णाङ्क kernel_w, ध्रुव पूर्णाङ्क kernel_h,
                                   ध्रुव पूर्णाङ्क stride,
                                   ध्रुव पूर्णाङ्क c_अन्तर्, ध्रुव पूर्णाङ्क c_बहिर्गमनम्, ध्रुव पूर्णाङ्क batches,
                                   ध्रुव queue_ptr & stream) {
    ध्रुव पूर्णाङ्क योग      = बहिर्गमनम्_w * बहिर्गमनम्_h * c_बहिर्गमनम् * batches;
    ध्रुव पूर्णाङ्क अङ्कः_खण्डानि = (योग + SYCL_CONV2_D_TRANSPOSE_खण्डम्_आकार - 1) / SYCL_CONV2_D_TRANSPOSE_खण्डम्_आकार;
    ध्रुव sycl::range<3> खण्डम्_dims(1, 1, SYCL_CONV2_D_TRANSPOSE_खण्डम्_आकार);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, अङ्कः_खण्डानि);
    stream->parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_transpose_kernel<kernel_t>(निवेशः_d, kernel_d, output_d,
                                             अन्तर्_w, अन्तर्_h, बहिर्गमनम्_w, बहिर्गमनम्_h, kernel_w, kernel_h,
                                             stride, c_अन्तर्, c_बहिर्गमनम्, batches, item_ct1);
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

    ध्रुव भिन्न * निवेशः_d  = (ध्रुव भिन्न *) निवेशः->दत्त;
    भिन्न *       output_d = (भिन्न *) dst->दत्त;
    ध्रुव शून्य *  kernel_d = kernel->दत्त;

    ध्रुव पूर्णाङ्क निवेशः_w      = निवेशः->ne[0];
    ध्रुव पूर्णाङ्क निवेशः_h      = निवेशः->ne[1];
    ध्रुव पूर्णाङ्क channels_अन्तर्  = निवेशः->ne[2];
    ध्रुव पूर्णाङ्क batches      = निवेशः->ne[3];
    ध्रुव पूर्णाङ्क output_w     = dst->ne[0];
    ध्रुव पूर्णाङ्क output_h     = dst->ne[1];
    ध्रुव पूर्णाङ्क channels_बहिर्गमनम् = kernel->ne[2];
    ध्रुव पूर्णाङ्क kernel_w     = kernel->ne[0];
    ध्रुव पूर्णाङ्क kernel_h     = kernel->ne[1];
    ध्रुव पूर्णाङ्क stride       = dst->op_params[0];

    GGML_ASSERT(channels_अन्तर् == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ध्रुव queue_ptr stream = ctx.stream();

    यदि (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(निवेशः_d, (ध्रुव sycl::half *) kernel_d, output_d,
                                          निवेशः_w, निवेशः_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_अन्तर्, channels_बहिर्गमनम्, batches, stream);
    } अन्यथा {
        conv2d_transpose_sycl<भिन्न>(निवेशः_d, (ध्रुव भिन्न *) kernel_d, output_d,
                                     निवेशः_w, निवेशः_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_अन्तर्, channels_बहिर्गमनम्, batches, stream);
    }
}
