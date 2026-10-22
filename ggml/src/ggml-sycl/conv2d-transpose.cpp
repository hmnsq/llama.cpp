#شامل "conv2d-transpose.hpp"
#شامل "convert.hpp"

سانچہ <قسم_نام kernel_t>
ساکن خالی conv2d_transpose_kernel(ثابت اعشاری * داخلہ, ثابت kernel_t * kernel, اعشاری * output,
                                    ثابت صحیح اندر_w, ثابت صحیح اندر_h,
                                    ثابت صحیح برآمد_w, ثابت صحیح برآمد_h,
                                    ثابت صحیح kernel_w, ثابت صحیح kernel_h,
                                    ثابت صحیح stride,
                                    ثابت صحیح c_اندر, ثابت صحیح c_برآمد, ثابت صحیح batches,
                                    ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت صحیح global_idx     = شے_ct1.get_local_id(2) +
                               شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2);
    ثابت صحیح کل_elements = برآمد_w * برآمد_h * c_برآمد * batches;

    اگر (global_idx >= کل_elements) {
        واپس;
    }

    ثابت صحیح برآمد_x = global_idx % برآمد_w;
    ثابت صحیح برآمد_y = (global_idx / برآمد_w) % برآمد_h;
    ثابت صحیح c_idx = (global_idx / (برآمد_w * برآمد_h)) % c_برآمد;
    ثابت صحیح n_idx = global_idx / (برآمد_w * برآمد_h * c_برآمد);

    اعشاری acc = 0.0f;

    برائے (صحیح c_اندر_idx = 0; c_اندر_idx < c_اندر; ++c_اندر_idx) {
        برائے (صحیح kh = 0; kh < kernel_h; ++kh) {
            صحیح اندر_y = برآمد_y - kh;
            اگر (اندر_y < 0 || اندر_y % stride) {
                جاری;
            }
            اندر_y /= stride;
            اگر (اندر_y >= اندر_h) {
                جاری;
            }

            برائے (صحیح kw = 0; kw < kernel_w; ++kw) {
                صحیح اندر_x = برآمد_x - kw;
                اگر (اندر_x < 0 || اندر_x % stride) {
                    جاری;
                }
                اندر_x /= stride;
                اگر (اندر_x >= اندر_w) {
                    جاری;
                }

                ثابت صحیح داخلہ_idx  = (اندر_w * اندر_h * c_اندر) * n_idx + (اندر_w * اندر_h) * c_اندر_idx + اندر_w * اندر_y + اندر_x;
                ثابت صحیح kernel_idx = (kernel_h * kernel_w * c_برآمد) * c_اندر_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += داخلہ[داخلہ_idx] * ggml_sycl_cast<اعشاری>(kernel[kernel_idx]);
            }
        }
    }

    output[(برآمد_w * برآمد_h * c_برآمد) * n_idx + (برآمد_w * برآمد_h) * c_idx + برآمد_w * برآمد_y + برآمد_x] = acc;
}

سانچہ <قسم_نام kernel_t>
ساکن خالی conv2d_transpose_sycl(ثابت اعشاری * داخلہ_d, ثابت kernel_t * kernel_d, اعشاری * output_d,
                                   ثابت صحیح اندر_w, ثابت صحیح اندر_h,
                                   ثابت صحیح برآمد_w, ثابت صحیح برآمد_h,
                                   ثابت صحیح kernel_w, ثابت صحیح kernel_h,
                                   ثابت صحیح stride,
                                   ثابت صحیح c_اندر, ثابت صحیح c_برآمد, ثابت صحیح batches,
                                   ثابت قطار_ptr & stream) {
    ثابت صحیح کل      = برآمد_w * برآمد_h * c_برآمد * batches;
    ثابت صحیح عدد_بلاکیں = (کل + SYCL_CONV2_D_TRANSPOSE_بلاک_سائز - 1) / SYCL_CONV2_D_TRANSPOSE_بلاک_سائز;
    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_CONV2_D_TRANSPOSE_بلاک_سائز);
    ثابت sycl::range<3> بلاک_nums(1, 1, عدد_بلاکیں);
    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) {
            conv2d_transpose_kernel<kernel_t>(داخلہ_d, kernel_d, output_d,
                                             اندر_w, اندر_h, برآمد_w, برآمد_h, kernel_w, kernel_h,
                                             stride, c_اندر, c_برآمد, batches, شے_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
خالی ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * داخلہ  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(داخلہ->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(داخلہ));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ثابت اعشاری * داخلہ_d  = (ثابت اعشاری *) داخلہ->ڈیٹا;
    اعشاری *       output_d = (اعشاری *) dst->ڈیٹا;
    ثابت خالی *  kernel_d = kernel->ڈیٹا;

    ثابت صحیح داخلہ_w      = داخلہ->ne[0];
    ثابت صحیح داخلہ_h      = داخلہ->ne[1];
    ثابت صحیح channels_اندر  = داخلہ->ne[2];
    ثابت صحیح batches      = داخلہ->ne[3];
    ثابت صحیح output_w     = dst->ne[0];
    ثابت صحیح output_h     = dst->ne[1];
    ثابت صحیح channels_برآمد = kernel->ne[2];
    ثابت صحیح kernel_w     = kernel->ne[0];
    ثابت صحیح kernel_h     = kernel->ne[1];
    ثابت صحیح stride       = dst->op_params[0];

    GGML_ASSERT(channels_اندر == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ثابت قطار_ptr stream = ctx.stream();

    اگر (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(داخلہ_d, (ثابت sycl::half *) kernel_d, output_d,
                                          داخلہ_w, داخلہ_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_اندر, channels_برآمد, batches, stream);
    } ورنہ {
        conv2d_transpose_sycl<اعشاری>(داخلہ_d, (ثابت اعشاری *) kernel_d, output_d,
                                     داخلہ_w, داخلہ_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_اندر, channels_برآمد, batches, stream);
    }
}
