#ضمن "conv2d-transpose.hpp"
#ضمن "convert.hpp"

سانچا <قسم_ناں kernel_t>
ساكن فراغ conv2d_transpose_kernel(ثابت عائم * داخلہ, ثابت kernel_t * kernel, عائم * output,
                                    ثابت صحيح وچ_w, ثابت صحيح وچ_h,
                                    ثابت صحيح باہرلا_w, ثابت صحيح باہرلا_h,
                                    ثابت صحيح kernel_w, ثابت صحيح kernel_h,
                                    ثابت صحيح stride,
                                    ثابت صحيح c_وچ, ثابت صحيح c_باہرلا, ثابت صحيح batches,
                                    ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت صحيح global_idx     = شے_ct1.get_local_id(2) +
                               شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2);
    ثابت صحيح کل_elements = باہرلا_w * باہرلا_h * c_باہرلا * batches;

    إذا (global_idx >= کل_elements) {
        أرجع;
    }

    ثابت صحيح باہرلا_x = global_idx % باہرلا_w;
    ثابت صحيح باہرلا_y = (global_idx / باہرلا_w) % باہرلا_h;
    ثابت صحيح c_idx = (global_idx / (باہرلا_w * باہرلا_h)) % c_باہرلا;
    ثابت صحيح n_idx = global_idx / (باہرلا_w * باہرلا_h * c_باہرلا);

    عائم acc = 0.0f;

    لكل (صحيح c_وچ_idx = 0; c_وچ_idx < c_وچ; ++c_وچ_idx) {
        لكل (صحيح kh = 0; kh < kernel_h; ++kh) {
            صحيح وچ_y = باہرلا_y - kh;
            إذا (وچ_y < 0 || وچ_y % stride) {
                استمر;
            }
            وچ_y /= stride;
            إذا (وچ_y >= وچ_h) {
                استمر;
            }

            لكل (صحيح kw = 0; kw < kernel_w; ++kw) {
                صحيح وچ_x = باہرلا_x - kw;
                إذا (وچ_x < 0 || وچ_x % stride) {
                    استمر;
                }
                وچ_x /= stride;
                إذا (وچ_x >= وچ_w) {
                    استمر;
                }

                ثابت صحيح داخلہ_idx  = (وچ_w * وچ_h * c_وچ) * n_idx + (وچ_w * وچ_h) * c_وچ_idx + وچ_w * وچ_y + وچ_x;
                ثابت صحيح kernel_idx = (kernel_h * kernel_w * c_باہرلا) * c_وچ_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += داخلہ[داخلہ_idx] * ggml_sycl_cast<عائم>(kernel[kernel_idx]);
            }
        }
    }

    output[(باہرلا_w * باہرلا_h * c_باہرلا) * n_idx + (باہرلا_w * باہرلا_h) * c_idx + باہرلا_w * باہرلا_y + باہرلا_x] = acc;
}

سانچا <قسم_ناں kernel_t>
ساكن فراغ conv2d_transpose_sycl(ثابت عائم * داخلہ_d, ثابت kernel_t * kernel_d, عائم * output_d,
                                   ثابت صحيح وچ_w, ثابت صحيح وچ_h,
                                   ثابت صحيح باہرلا_w, ثابت صحيح باہرلا_h,
                                   ثابت صحيح kernel_w, ثابت صحيح kernel_h,
                                   ثابت صحيح stride,
                                   ثابت صحيح c_وچ, ثابت صحيح c_باہرلا, ثابت صحيح batches,
                                   ثابت قطار_ptr & stream) {
    ثابت صحيح کل      = باہرلا_w * باہرلا_h * c_باہرلا * batches;
    ثابت صحيح عدد_بلاکاں = (کل + SYCL_CONV2_D_TRANSPOSE_بلاک_ناپ - 1) / SYCL_CONV2_D_TRANSPOSE_بلاک_ناپ;
    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_CONV2_D_TRANSPOSE_بلاک_ناپ);
    ثابت sycl::range<3> بلاک_nums(1, 1, عدد_بلاکاں);
    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) {
            conv2d_transpose_kernel<kernel_t>(داخلہ_d, kernel_d, output_d,
                                             وچ_w, وچ_h, باہرلا_w, باہرلا_h, kernel_w, kernel_h,
                                             stride, c_وچ, c_باہرلا, batches, شے_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
فراغ ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * داخلہ  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(داخلہ->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(داخلہ));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ثابت عائم * داخلہ_d  = (ثابت عائم *) داخلہ->انکڑے;
    عائم *       output_d = (عائم *) dst->انکڑے;
    ثابت فراغ *  kernel_d = kernel->انکڑے;

    ثابت صحيح داخلہ_w      = داخلہ->ne[0];
    ثابت صحيح داخلہ_h      = داخلہ->ne[1];
    ثابت صحيح channels_وچ  = داخلہ->ne[2];
    ثابت صحيح batches      = داخلہ->ne[3];
    ثابت صحيح output_w     = dst->ne[0];
    ثابت صحيح output_h     = dst->ne[1];
    ثابت صحيح channels_باہرلا = kernel->ne[2];
    ثابت صحيح kernel_w     = kernel->ne[0];
    ثابت صحيح kernel_h     = kernel->ne[1];
    ثابت صحيح stride       = dst->op_params[0];

    GGML_ASSERT(channels_وچ == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ثابت قطار_ptr stream = ctx.stream();

    إذا (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(داخلہ_d, (ثابت sycl::half *) kernel_d, output_d,
                                          داخلہ_w, داخلہ_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_وچ, channels_باہرلا, batches, stream);
    } وإلا {
        conv2d_transpose_sycl<عائم>(داخلہ_d, (ثابت عائم *) kernel_d, output_d,
                                     داخلہ_w, داخلہ_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_وچ, channels_باہرلا, batches, stream);
    }
}
