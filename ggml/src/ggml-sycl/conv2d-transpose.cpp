#ضمن "conv2d-transpose.hpp"
#ضمن "convert.hpp"

قالب <نام_نوع kernel_t>
ساكن فراغ conv2d_transpose_kernel(ثابت عائم * درون_داد, ثابت kernel_t * kernel, عائم * output,
                                    ثابت صحيح درون_w, ثابت صحيح درون_h,
                                    ثابت صحيح بیرون_w, ثابت صحيح بیرون_h,
                                    ثابت صحيح kernel_w, ثابت صحيح kernel_h,
                                    ثابت صحيح stride,
                                    ثابت صحيح c_درون, ثابت صحيح c_بیرون, ثابت صحيح batches,
                                    ثابت sycl::nd_قلم<3> & قلم_ct1) {
    ثابت صحيح global_idx     = قلم_ct1.get_local_id(2) +
                               قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(2);
    ثابت صحيح مجموع_elements = بیرون_w * بیرون_h * c_بیرون * batches;

    إذا (global_idx >= مجموع_elements) {
        أرجع;
    }

    ثابت صحيح بیرون_x = global_idx % بیرون_w;
    ثابت صحيح بیرون_y = (global_idx / بیرون_w) % بیرون_h;
    ثابت صحيح c_idx = (global_idx / (بیرون_w * بیرون_h)) % c_بیرون;
    ثابت صحيح n_idx = global_idx / (بیرون_w * بیرون_h * c_بیرون);

    عائم acc = 0.0f;

    لكل (صحيح c_درون_idx = 0; c_درون_idx < c_درون; ++c_درون_idx) {
        لكل (صحيح kh = 0; kh < kernel_h; ++kh) {
            صحيح درون_y = بیرون_y - kh;
            إذا (درون_y < 0 || درون_y % stride) {
                استمر;
            }
            درون_y /= stride;
            إذا (درون_y >= درون_h) {
                استمر;
            }

            لكل (صحيح kw = 0; kw < kernel_w; ++kw) {
                صحيح درون_x = بیرون_x - kw;
                إذا (درون_x < 0 || درون_x % stride) {
                    استمر;
                }
                درون_x /= stride;
                إذا (درون_x >= درون_w) {
                    استمر;
                }

                ثابت صحيح درون_داد_idx  = (درون_w * درون_h * c_درون) * n_idx + (درون_w * درون_h) * c_درون_idx + درون_w * درون_y + درون_x;
                ثابت صحيح kernel_idx = (kernel_h * kernel_w * c_بیرون) * c_درون_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += درون_داد[درون_داد_idx] * ggml_sycl_cast<عائم>(kernel[kernel_idx]);
            }
        }
    }

    output[(بیرون_w * بیرون_h * c_بیرون) * n_idx + (بیرون_w * بیرون_h) * c_idx + بیرون_w * بیرون_y + بیرون_x] = acc;
}

قالب <نام_نوع kernel_t>
ساكن فراغ conv2d_transpose_sycl(ثابت عائم * درون_داد_d, ثابت kernel_t * kernel_d, عائم * output_d,
                                   ثابت صحيح درون_w, ثابت صحيح درون_h,
                                   ثابت صحيح بیرون_w, ثابت صحيح بیرون_h,
                                   ثابت صحيح kernel_w, ثابت صحيح kernel_h,
                                   ثابت صحيح stride,
                                   ثابت صحيح c_درون, ثابت صحيح c_بیرون, ثابت صحيح batches,
                                   ثابت صف_ptr & stream) {
    ثابت صحيح مجموع      = بیرون_w * بیرون_h * c_بیرون * batches;
    ثابت صحيح رقم_بلوکها = (مجموع + SYCL_CONV2_D_TRANSPOSE_بلوک_اندازه - 1) / SYCL_CONV2_D_TRANSPOSE_بلوک_اندازه;
    ثابت sycl::range<3> بلوک_dims(1, 1, SYCL_CONV2_D_TRANSPOSE_بلوک_اندازه);
    ثابت sycl::range<3> بلوک_nums(1, 1, رقم_بلوکها);
    stream->parallel_for(sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
        [=](sycl::nd_قلم<3> قلم_ct1) {
            conv2d_transpose_kernel<kernel_t>(درون_داد_d, kernel_d, output_d,
                                             درون_w, درون_h, بیرون_w, بیرون_h, kernel_w, kernel_h,
                                             stride, c_درون, c_بیرون, batches, قلم_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
فراغ ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * درون_داد  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(درون_داد->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(درون_داد));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ثابت عائم * درون_داد_d  = (ثابت عائم *) درون_داد->داده;
    عائم *       output_d = (عائم *) dst->داده;
    ثابت فراغ *  kernel_d = kernel->داده;

    ثابت صحيح درون_داد_w      = درون_داد->ne[0];
    ثابت صحيح درون_داد_h      = درون_داد->ne[1];
    ثابت صحيح channels_درون  = درون_داد->ne[2];
    ثابت صحيح batches      = درون_داد->ne[3];
    ثابت صحيح output_w     = dst->ne[0];
    ثابت صحيح output_h     = dst->ne[1];
    ثابت صحيح channels_بیرون = kernel->ne[2];
    ثابت صحيح kernel_w     = kernel->ne[0];
    ثابت صحيح kernel_h     = kernel->ne[1];
    ثابت صحيح stride       = dst->op_params[0];

    GGML_ASSERT(channels_درون == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ثابت صف_ptr stream = ctx.stream();

    إذا (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(درون_داد_d, (ثابت sycl::half *) kernel_d, output_d,
                                          درون_داد_w, درون_داد_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_درون, channels_بیرون, batches, stream);
    } وإلا {
        conv2d_transpose_sycl<عائم>(درون_داد_d, (ثابت عائم *) kernel_d, output_d,
                                     درون_داد_w, درون_داد_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_درون, channels_بیرون, batches, stream);
    }
}
