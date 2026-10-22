#ضمن "conv2d-transpose.hpp"
#ضمن "convert.hpp"

سانچو <قسم_نالو kernel_t>
ساكن فراغ conv2d_transpose_kernel(ثابت عائم * داخلا_2, ثابت kernel_t * kernel, عائم * output,
                                    ثابت صحيح اندريون_w, ثابت صحيح اندريون_h,
                                    ثابت صحيح ٻاهريون_w, ثابت صحيح ٻاهريون_h,
                                    ثابت صحيح kernel_w, ثابت صحيح kernel_h,
                                    ثابت صحيح stride,
                                    ثابت صحيح c_اندريون, ثابت صحيح c_ٻاهريون, ثابت صحيح batches,
                                    ثابت sycl::nd_شي<3> & شي_ct1) {
    ثابت صحيح global_idx     = شي_ct1.get_local_id(2) +
                               شي_ct1.get_گروهه(2) * شي_ct1.get_local_range(2);
    ثابت صحيح ڪل_elements = ٻاهريون_w * ٻاهريون_h * c_ٻاهريون * batches;

    إذا (global_idx >= ڪل_elements) {
        أرجع;
    }

    ثابت صحيح ٻاهريون_x = global_idx % ٻاهريون_w;
    ثابت صحيح ٻاهريون_y = (global_idx / ٻاهريون_w) % ٻاهريون_h;
    ثابت صحيح c_idx = (global_idx / (ٻاهريون_w * ٻاهريون_h)) % c_ٻاهريون;
    ثابت صحيح n_idx = global_idx / (ٻاهريون_w * ٻاهريون_h * c_ٻاهريون);

    عائم acc = 0.0f;

    لكل (صحيح c_اندريون_idx = 0; c_اندريون_idx < c_اندريون; ++c_اندريون_idx) {
        لكل (صحيح kh = 0; kh < kernel_h; ++kh) {
            صحيح اندريون_y = ٻاهريون_y - kh;
            إذا (اندريون_y < 0 || اندريون_y % stride) {
                استمر;
            }
            اندريون_y /= stride;
            إذا (اندريون_y >= اندريون_h) {
                استمر;
            }

            لكل (صحيح kw = 0; kw < kernel_w; ++kw) {
                صحيح اندريون_x = ٻاهريون_x - kw;
                إذا (اندريون_x < 0 || اندريون_x % stride) {
                    استمر;
                }
                اندريون_x /= stride;
                إذا (اندريون_x >= اندريون_w) {
                    استمر;
                }

                ثابت صحيح داخلا_idx  = (اندريون_w * اندريون_h * c_اندريون) * n_idx + (اندريون_w * اندريون_h) * c_اندريون_idx + اندريون_w * اندريون_y + اندريون_x;
                ثابت صحيح kernel_idx = (kernel_h * kernel_w * c_ٻاهريون) * c_اندريون_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += داخلا_2[داخلا_idx] * ggml_sycl_cast<عائم>(kernel[kernel_idx]);
            }
        }
    }

    output[(ٻاهريون_w * ٻاهريون_h * c_ٻاهريون) * n_idx + (ٻاهريون_w * ٻاهريون_h) * c_idx + ٻاهريون_w * ٻاهريون_y + ٻاهريون_x] = acc;
}

سانچو <قسم_نالو kernel_t>
ساكن فراغ conv2d_transpose_sycl(ثابت عائم * داخلا_d, ثابت kernel_t * kernel_d, عائم * output_d,
                                   ثابت صحيح اندريون_w, ثابت صحيح اندريون_h,
                                   ثابت صحيح ٻاهريون_w, ثابت صحيح ٻاهريون_h,
                                   ثابت صحيح kernel_w, ثابت صحيح kernel_h,
                                   ثابت صحيح stride,
                                   ثابت صحيح c_اندريون, ثابت صحيح c_ٻاهريون, ثابت صحيح batches,
                                   ثابت قطار_ptr & stream) {
    ثابت صحيح ڪل      = ٻاهريون_w * ٻاهريون_h * c_ٻاهريون * batches;
    ثابت صحيح انگ_بلاڪون = (ڪل + SYCL_CONV2_D_TRANSPOSE_بلاڪ_سائيز - 1) / SYCL_CONV2_D_TRANSPOSE_بلاڪ_سائيز;
    ثابت sycl::range<3> بلاڪ_dims(1, 1, SYCL_CONV2_D_TRANSPOSE_بلاڪ_سائيز);
    ثابت sycl::range<3> بلاڪ_nums(1, 1, انگ_بلاڪون);
    stream->parallel_for(sycl::nd_range<3>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
        [=](sycl::nd_شي<3> شي_ct1) {
            conv2d_transpose_kernel<kernel_t>(داخلا_d, kernel_d, output_d,
                                             اندريون_w, اندريون_h, ٻاهريون_w, ٻاهريون_h, kernel_w, kernel_h,
                                             stride, c_اندريون, c_ٻاهريون, batches, شي_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
فراغ ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * داخلا_2  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(داخلا_2->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(داخلا_2));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ثابت عائم * داخلا_d  = (ثابت عائم *) داخلا_2->معلومات;
    عائم *       output_d = (عائم *) dst->معلومات;
    ثابت فراغ *  kernel_d = kernel->معلومات;

    ثابت صحيح داخلا_w      = داخلا_2->ne[0];
    ثابت صحيح داخلا_h      = داخلا_2->ne[1];
    ثابت صحيح channels_اندريون  = داخلا_2->ne[2];
    ثابت صحيح batches      = داخلا_2->ne[3];
    ثابت صحيح output_w     = dst->ne[0];
    ثابت صحيح output_h     = dst->ne[1];
    ثابت صحيح channels_ٻاهريون = kernel->ne[2];
    ثابت صحيح kernel_w     = kernel->ne[0];
    ثابت صحيح kernel_h     = kernel->ne[1];
    ثابت صحيح stride       = dst->op_params[0];

    GGML_ASSERT(channels_اندريون == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ثابت قطار_ptr stream = ctx.stream();

    إذا (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(داخلا_d, (ثابت sycl::half *) kernel_d, output_d,
                                          داخلا_w, داخلا_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_اندريون, channels_ٻاهريون, batches, stream);
    } وإلا {
        conv2d_transpose_sycl<عائم>(داخلا_d, (ثابت عائم *) kernel_d, output_d,
                                     داخلا_w, داخلا_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_اندريون, channels_ٻاهريون, batches, stream);
    }
}
