#ਅਨ੍ਤਰ੍ਭਾਵ "conv2d-transpose.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "convert.hpp"

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ kernel_t>
ਸਥਿਰ ਖਾਲੀ conv2d_transpose_kernel(ਅਚਲ ਦਸ਼ਮਲਵ * ਨਿਵੇਸ਼, ਅਚਲ kernel_t * kernel, ਦਸ਼ਮਲਵ * output,
                                    ਅਚਲ ਪੂਰਨ ਵਿੱਚ_w, ਅਚਲ ਪੂਰਨ ਵਿੱਚ_h,
                                    ਅਚਲ ਪੂਰਨ ਬਾਹਰਲਾ_w, ਅਚਲ ਪੂਰਨ ਬਾਹਰਲਾ_h,
                                    ਅਚਲ ਪੂਰਨ kernel_w, ਅਚਲ ਪੂਰਨ kernel_h,
                                    ਅਚਲ ਪੂਰਨ stride,
                                    ਅਚਲ ਪੂਰਨ c_ਵਿੱਚ, ਅਚਲ ਪੂਰਨ c_ਬਾਹਰਲਾ, ਅਚਲ ਪੂਰਨ batches,
                                    ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ global_idx     = ਵਸਤੂ_ct1.get_local_id(2) +
                               ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
    ਅਚਲ ਪੂਰਨ ਕੁੱਲ_elements = ਬਾਹਰਲਾ_w * ਬਾਹਰਲਾ_h * c_ਬਾਹਰਲਾ * batches;

    ਜੇ (global_idx >= ਕੁੱਲ_elements) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ ਬਾਹਰਲਾ_x = global_idx % ਬਾਹਰਲਾ_w;
    ਅਚਲ ਪੂਰਨ ਬਾਹਰਲਾ_y = (global_idx / ਬਾਹਰਲਾ_w) % ਬਾਹਰਲਾ_h;
    ਅਚਲ ਪੂਰਨ c_idx = (global_idx / (ਬਾਹਰਲਾ_w * ਬਾਹਰਲਾ_h)) % c_ਬਾਹਰਲਾ;
    ਅਚਲ ਪੂਰਨ n_idx = global_idx / (ਬਾਹਰਲਾ_w * ਬਾਹਰਲਾ_h * c_ਬਾਹਰਲਾ);

    ਦਸ਼ਮਲਵ acc = 0.0f;

    ਲਈ (ਪੂਰਨ c_ਵਿੱਚ_idx = 0; c_ਵਿੱਚ_idx < c_ਵਿੱਚ; ++c_ਵਿੱਚ_idx) {
        ਲਈ (ਪੂਰਨ kh = 0; kh < kernel_h; ++kh) {
            ਪੂਰਨ ਵਿੱਚ_y = ਬਾਹਰਲਾ_y - kh;
            ਜੇ (ਵਿੱਚ_y < 0 || ਵਿੱਚ_y % stride) {
                ਜਾਰੀ;
            }
            ਵਿੱਚ_y /= stride;
            ਜੇ (ਵਿੱਚ_y >= ਵਿੱਚ_h) {
                ਜਾਰੀ;
            }

            ਲਈ (ਪੂਰਨ kw = 0; kw < kernel_w; ++kw) {
                ਪੂਰਨ ਵਿੱਚ_x = ਬਾਹਰਲਾ_x - kw;
                ਜੇ (ਵਿੱਚ_x < 0 || ਵਿੱਚ_x % stride) {
                    ਜਾਰੀ;
                }
                ਵਿੱਚ_x /= stride;
                ਜੇ (ਵਿੱਚ_x >= ਵਿੱਚ_w) {
                    ਜਾਰੀ;
                }

                ਅਚਲ ਪੂਰਨ ਨਿਵੇਸ਼_idx  = (ਵਿੱਚ_w * ਵਿੱਚ_h * c_ਵਿੱਚ) * n_idx + (ਵਿੱਚ_w * ਵਿੱਚ_h) * c_ਵਿੱਚ_idx + ਵਿੱਚ_w * ਵਿੱਚ_y + ਵਿੱਚ_x;
                ਅਚਲ ਪੂਰਨ kernel_idx = (kernel_h * kernel_w * c_ਬਾਹਰਲਾ) * c_ਵਿੱਚ_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += ਨਿਵੇਸ਼[ਨਿਵੇਸ਼_idx] * ggml_sycl_cast<ਦਸ਼ਮਲਵ>(kernel[kernel_idx]);
            }
        }
    }

    output[(ਬਾਹਰਲਾ_w * ਬਾਹਰਲਾ_h * c_ਬਾਹਰਲਾ) * n_idx + (ਬਾਹਰਲਾ_w * ਬਾਹਰਲਾ_h) * c_idx + ਬਾਹਰਲਾ_w * ਬਾਹਰਲਾ_y + ਬਾਹਰਲਾ_x] = acc;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ kernel_t>
ਸਥਿਰ ਖਾਲੀ conv2d_transpose_sycl(ਅਚਲ ਦਸ਼ਮਲਵ * ਨਿਵੇਸ਼_d, ਅਚਲ kernel_t * kernel_d, ਦਸ਼ਮਲਵ * output_d,
                                   ਅਚਲ ਪੂਰਨ ਵਿੱਚ_w, ਅਚਲ ਪੂਰਨ ਵਿੱਚ_h,
                                   ਅਚਲ ਪੂਰਨ ਬਾਹਰਲਾ_w, ਅਚਲ ਪੂਰਨ ਬਾਹਰਲਾ_h,
                                   ਅਚਲ ਪੂਰਨ kernel_w, ਅਚਲ ਪੂਰਨ kernel_h,
                                   ਅਚਲ ਪੂਰਨ stride,
                                   ਅਚਲ ਪੂਰਨ c_ਵਿੱਚ, ਅਚਲ ਪੂਰਨ c_ਬਾਹਰਲਾ, ਅਚਲ ਪੂਰਨ batches,
                                   ਅਚਲ ਕਤਾਰ_ptr & stream) {
    ਅਚਲ ਪੂਰਨ ਕੁੱਲ      = ਬਾਹਰਲਾ_w * ਬਾਹਰਲਾ_h * c_ਬਾਹਰਲਾ * batches;
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ਕੁੱਲ + SYCL_CONV2_D_TRANSPOSE_ਖੰਡ_ਮਾਪ - 1) / SYCL_CONV2_D_TRANSPOSE_ਖੰਡ_ਮਾਪ;
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, SYCL_CONV2_D_TRANSPOSE_ਖੰਡ_ਮਾਪ);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਅੰਕ_ਖੰਡਾਂ);
    stream->parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            conv2d_transpose_kernel<kernel_t>(ਨਿਵੇਸ਼_d, kernel_d, output_d,
                                             ਵਿੱਚ_w, ਵਿੱਚ_h, ਬਾਹਰਲਾ_w, ਬਾਹਰਲਾ_h, kernel_w, kernel_h,
                                             stride, c_ਵਿੱਚ, c_ਬਾਹਰਲਾ, batches, ਵਸਤੂ_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
ਖਾਲੀ ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ਅਚਲ ggml_tensor * kernel = dst->src[0];
    ਅਚਲ ggml_tensor * ਨਿਵੇਸ਼  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(ਨਿਵੇਸ਼->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(ਨਿਵੇਸ਼));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ਅਚਲ ਦਸ਼ਮਲਵ * ਨਿਵੇਸ਼_d  = (ਅਚਲ ਦਸ਼ਮਲਵ *) ਨਿਵੇਸ਼->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *       output_d = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
    ਅਚਲ ਖਾਲੀ *  kernel_d = kernel->ਅੰਕੜੇ;

    ਅਚਲ ਪੂਰਨ ਨਿਵੇਸ਼_w      = ਨਿਵੇਸ਼->ne[0];
    ਅਚਲ ਪੂਰਨ ਨਿਵੇਸ਼_h      = ਨਿਵੇਸ਼->ne[1];
    ਅਚਲ ਪੂਰਨ channels_ਵਿੱਚ  = ਨਿਵੇਸ਼->ne[2];
    ਅਚਲ ਪੂਰਨ batches      = ਨਿਵੇਸ਼->ne[3];
    ਅਚਲ ਪੂਰਨ output_w     = dst->ne[0];
    ਅਚਲ ਪੂਰਨ output_h     = dst->ne[1];
    ਅਚਲ ਪੂਰਨ channels_ਬਾਹਰਲਾ = kernel->ne[2];
    ਅਚਲ ਪੂਰਨ kernel_w     = kernel->ne[0];
    ਅਚਲ ਪੂਰਨ kernel_h     = kernel->ne[1];
    ਅਚਲ ਪੂਰਨ stride       = dst->op_params[0];

    GGML_ASSERT(channels_ਵਿੱਚ == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ਅਚਲ ਕਤਾਰ_ptr stream = ctx.stream();

    ਜੇ (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(ਨਿਵੇਸ਼_d, (ਅਚਲ sycl::half *) kernel_d, output_d,
                                          ਨਿਵੇਸ਼_w, ਨਿਵੇਸ਼_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_ਵਿੱਚ, channels_ਬਾਹਰਲਾ, batches, stream);
    } ਨਹੀਂਤਾਂ {
        conv2d_transpose_sycl<ਦਸ਼ਮਲਵ>(ਨਿਵੇਸ਼_d, (ਅਚਲ ਦਸ਼ਮਲਵ *) kernel_d, output_d,
                                     ਨਿਵੇਸ਼_w, ਨਿਵੇਸ਼_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_ਵਿੱਚ, channels_ਬਾਹਰਲਾ, batches, stream);
    }
}
