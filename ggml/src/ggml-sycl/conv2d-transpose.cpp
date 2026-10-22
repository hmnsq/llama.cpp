#شامل "conv2d-transpose.hpp"
#شامل "convert.hpp"

template <typename kernel_t>
ایستا تهی conv2d_transpose_kernel(ثابت اعشاری * درون_داد, ثابت kernel_t * kernel, اعشاری * output,
                                    ثابت صحیح in_w, ثابت صحیح in_h,
                                    ثابت صحیح out_w, ثابت صحیح out_h,
                                    ثابت صحیح kernel_w, ثابت صحیح kernel_h,
                                    ثابت صحیح stride,
                                    ثابت صحیح c_in, ثابت صحیح c_out, ثابت صحیح batches,
                                    ثابت sycl::nd_item<3> & item_ct1) {
    ثابت صحیح global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ثابت صحیح total_elements = out_w * out_h * c_out * batches;

    اگر (global_idx >= total_elements) {
        بازگشت;
    }

    ثابت صحیح out_x = global_idx % out_w;
    ثابت صحیح out_y = (global_idx / out_w) % out_h;
    ثابت صحیح c_idx = (global_idx / (out_w * out_h)) % c_out;
    ثابت صحیح n_idx = global_idx / (out_w * out_h * c_out);

    اعشاری acc = 0.0f;

    برای (صحیح c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        برای (صحیح kh = 0; kh < kernel_h; ++kh) {
            صحیح in_y = out_y - kh;
            اگر (in_y < 0 || in_y % stride) {
                ادامه;
            }
            in_y /= stride;
            اگر (in_y >= in_h) {
                ادامه;
            }

            برای (صحیح kw = 0; kw < kernel_w; ++kw) {
                صحیح in_x = out_x - kw;
                اگر (in_x < 0 || in_x % stride) {
                    ادامه;
                }
                in_x /= stride;
                اگر (in_x >= in_w) {
                    ادامه;
                }

                ثابت صحیح درون_داد_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                ثابت صحیح kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += درون_داد[درون_داد_idx] * ggml_sycl_cast<اعشاری>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

template <typename kernel_t>
ایستا تهی conv2d_transpose_sycl(ثابت اعشاری * درون_داد_d, ثابت kernel_t * kernel_d, اعشاری * output_d,
                                   ثابت صحیح in_w, ثابت صحیح in_h,
                                   ثابت صحیح out_w, ثابت صحیح out_h,
                                   ثابت صحیح kernel_w, ثابت صحیح kernel_h,
                                   ثابت صحیح stride,
                                   ثابت صحیح c_in, ثابت صحیح c_out, ثابت صحیح batches,
                                   ثابت queue_ptr & stream) {
    ثابت صحیح total      = out_w * out_h * c_out * batches;
    ثابت صحیح رقم_blocks = (total + SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE;
    ثابت sycl::range<3> block_dims(1, 1, SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE);
    ثابت sycl::range<3> block_nums(1, 1, رقم_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_transpose_kernel<kernel_t>(درون_داد_d, kernel_d, output_d,
                                             in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                             stride, c_in, c_out, batches, item_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
تهی ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * درون_داد  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(درون_داد->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(درون_داد));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ثابت اعشاری * درون_داد_d  = (ثابت اعشاری *) درون_داد->data;
    اعشاری *       output_d = (اعشاری *) dst->data;
    ثابت تهی *  kernel_d = kernel->data;

    ثابت صحیح درون_داد_w      = درون_داد->ne[0];
    ثابت صحیح درون_داد_h      = درون_داد->ne[1];
    ثابت صحیح channels_in  = درون_داد->ne[2];
    ثابت صحیح batches      = درون_داد->ne[3];
    ثابت صحیح output_w     = dst->ne[0];
    ثابت صحیح output_h     = dst->ne[1];
    ثابت صحیح channels_out = kernel->ne[2];
    ثابت صحیح kernel_w     = kernel->ne[0];
    ثابت صحیح kernel_h     = kernel->ne[1];
    ثابت صحیح stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ثابت queue_ptr stream = ctx.stream();

    اگر (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(درون_داد_d, (ثابت sycl::half *) kernel_d, output_d,
                                          درون_داد_w, درون_داد_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } وگرنه {
        conv2d_transpose_sycl<اعشاری>(درون_داد_d, (ثابت اعشاری *) kernel_d, output_d,
                                     درون_داد_w, درون_داد_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
