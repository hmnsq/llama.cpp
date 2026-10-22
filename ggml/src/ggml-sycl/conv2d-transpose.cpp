#ضمن "conv2d-transpose.hpp"
#ضمن "convert.hpp"

کينډۍ <ډول_نوم kernel_t>
ساكن فراغ conv2d_transpose_kernel(ثابت عائم * ننوت, ثابت kernel_t * kernel, عائم * output,
                                    ثابت صحيح دننه_w, ثابت صحيح دننه_h,
                                    ثابت صحيح بهر_w, ثابت صحيح بهر_h,
                                    ثابت صحيح kernel_w, ثابت صحيح kernel_h,
                                    ثابت صحيح stride,
                                    ثابت صحيح c_دننه, ثابت صحيح c_بهر, ثابت صحيح batches,
                                    ثابت sycl::nd_توکی<3> & توکی_ct1) {
    ثابت صحيح global_idx     = توکی_ct1.get_local_id(2) +
                               توکی_ct1.get_ډله(2) * توکی_ct1.get_local_range(2);
    ثابت صحيح ټول_elements = بهر_w * بهر_h * c_بهر * batches;

    إذا (global_idx >= ټول_elements) {
        أرجع;
    }

    ثابت صحيح بهر_x = global_idx % بهر_w;
    ثابت صحيح بهر_y = (global_idx / بهر_w) % بهر_h;
    ثابت صحيح c_idx = (global_idx / (بهر_w * بهر_h)) % c_بهر;
    ثابت صحيح n_idx = global_idx / (بهر_w * بهر_h * c_بهر);

    عائم acc = 0.0f;

    لكل (صحيح c_دننه_idx = 0; c_دننه_idx < c_دننه; ++c_دننه_idx) {
        لكل (صحيح kh = 0; kh < kernel_h; ++kh) {
            صحيح دننه_y = بهر_y - kh;
            إذا (دننه_y < 0 || دننه_y % stride) {
                استمر;
            }
            دننه_y /= stride;
            إذا (دننه_y >= دننه_h) {
                استمر;
            }

            لكل (صحيح kw = 0; kw < kernel_w; ++kw) {
                صحيح دننه_x = بهر_x - kw;
                إذا (دننه_x < 0 || دننه_x % stride) {
                    استمر;
                }
                دننه_x /= stride;
                إذا (دننه_x >= دننه_w) {
                    استمر;
                }

                ثابت صحيح ننوت_idx  = (دننه_w * دننه_h * c_دننه) * n_idx + (دننه_w * دننه_h) * c_دننه_idx + دننه_w * دننه_y + دننه_x;
                ثابت صحيح kernel_idx = (kernel_h * kernel_w * c_بهر) * c_دننه_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += ننوت[ننوت_idx] * ggml_sycl_cast<عائم>(kernel[kernel_idx]);
            }
        }
    }

    output[(بهر_w * بهر_h * c_بهر) * n_idx + (بهر_w * بهر_h) * c_idx + بهر_w * بهر_y + بهر_x] = acc;
}

کينډۍ <ډول_نوم kernel_t>
ساكن فراغ conv2d_transpose_sycl(ثابت عائم * ننوت_d, ثابت kernel_t * kernel_d, عائم * output_d,
                                   ثابت صحيح دننه_w, ثابت صحيح دننه_h,
                                   ثابت صحيح بهر_w, ثابت صحيح بهر_h,
                                   ثابت صحيح kernel_w, ثابت صحيح kernel_h,
                                   ثابت صحيح stride,
                                   ثابت صحيح c_دننه, ثابت صحيح c_بهر, ثابت صحيح batches,
                                   ثابت کتار_ptr & stream) {
    ثابت صحيح ټول      = بهر_w * بهر_h * c_بهر * batches;
    ثابت صحيح عدد_بلاکونه = (ټول + SYCL_CONV2_D_TRANSPOSE_بلاک_اندازه - 1) / SYCL_CONV2_D_TRANSPOSE_بلاک_اندازه;
    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_CONV2_D_TRANSPOSE_بلاک_اندازه);
    ثابت sycl::range<3> بلاک_nums(1, 1, عدد_بلاکونه);
    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_توکی<3> توکی_ct1) {
            conv2d_transpose_kernel<kernel_t>(ننوت_d, kernel_d, output_d,
                                             دننه_w, دننه_h, بهر_w, بهر_h, kernel_w, kernel_h,
                                             stride, c_دننه, c_بهر, batches, توکی_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
فراغ ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * ننوت  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(ننوت->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(ننوت));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ثابت عائم * ننوت_d  = (ثابت عائم *) ننوت->معلومات;
    عائم *       output_d = (عائم *) dst->معلومات;
    ثابت فراغ *  kernel_d = kernel->معلومات;

    ثابت صحيح ننوت_w      = ننوت->ne[0];
    ثابت صحيح ننوت_h      = ننوت->ne[1];
    ثابت صحيح channels_دننه  = ننوت->ne[2];
    ثابت صحيح batches      = ننوت->ne[3];
    ثابت صحيح output_w     = dst->ne[0];
    ثابت صحيح output_h     = dst->ne[1];
    ثابت صحيح channels_بهر = kernel->ne[2];
    ثابت صحيح kernel_w     = kernel->ne[0];
    ثابت صحيح kernel_h     = kernel->ne[1];
    ثابت صحيح stride       = dst->op_params[0];

    GGML_ASSERT(channels_دننه == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ثابت کتار_ptr stream = ctx.stream();

    إذا (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(ننوت_d, (ثابت sycl::half *) kernel_d, output_d,
                                          ننوت_w, ننوت_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_دننه, channels_بهر, batches, stream);
    } وإلا {
        conv2d_transpose_sycl<عائم>(ننوت_d, (ثابت عائم *) kernel_d, output_d,
                                     ننوت_w, ننوت_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_دننه, channels_بهر, batches, stream);
    }
}
