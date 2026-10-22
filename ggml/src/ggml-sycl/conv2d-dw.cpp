#অন্তর্ভাব "conv2d-dw.hpp"

কাঠামো conv2d_dw_params {
    পূর্ণসংখ্যা in_w, in_h;
    পূর্ণসংখ্যা out_w, out_h;
    পূর্ণসংখ্যা kernel_w, kernel_h;
    পূর্ণসংখ্যা stride_x, stride_y;
    পূর্ণসংখ্যা padding_x, padding_y;
    পূর্ণসংখ্যা dilation_x, dilation_y;
    পূর্ণসংখ্যা channels, batches;
};

কাঠামো conv2d_dw_kernel_bounds {
    পূর্ণসংখ্যা y_min, y_max;
    পূর্ণসংখ্যা x_min, x_max;
};

স্থির অন্তঃরেখ conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(পূর্ণসংখ্যা out_x, পূর্ণসংখ্যা out_y,
                                                                  ধ্রুবক conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    ফেরত bounds;
}

স্থির অন্তঃরেখ পূর্ণসংখ্যা dw_calculate_প্রবেশ_coord(পূর্ণসংখ্যা out_coord, পূর্ণসংখ্যা kern_coord, পূর্ণসংখ্যা stride, পূর্ণসংখ্যা dilation, পূর্ণসংখ্যা padding) {
    ফেরত out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
কাঠামো dw_whcn_layout {
    স্থির পূর্ণসংখ্যা প্রবেশ_সূচক(পূর্ণসংখ্যা n, পূর্ণসংখ্যা c, পূর্ণসংখ্যা y, পূর্ণসংখ্যা x, ধ্রুবক conv2d_dw_params & p) {
        ফেরত n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    স্থির পূর্ণসংখ্যা kernel_সূচক(পূর্ণসংখ্যা c, পূর্ণসংখ্যা ky, পূর্ণসংখ্যা kx, ধ্রুবক conv2d_dw_params & p) {
        ফেরত c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    স্থির পূর্ণসংখ্যা output_সূচক(পূর্ণসংখ্যা n, পূর্ণসংখ্যা c, পূর্ণসংখ্যা y, পূর্ণসংখ্যা x, ধ্রুবক conv2d_dw_params & p) {
        ফেরত n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    স্থির শূন্য unpack_indices(পূর্ণসংখ্যা global_idx, ধ্রুবক conv2d_dw_params & p,
                               পূর্ণসংখ্যা & n, পূর্ণসংখ্যা & c, পূর্ণসংখ্যা & out_y, পূর্ণসংখ্যা & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
কাঠামো dw_cwhn_layout {
    স্থির পূর্ণসংখ্যা প্রবেশ_সূচক(পূর্ণসংখ্যা n, পূর্ণসংখ্যা c, পূর্ণসংখ্যা y, পূর্ণসংখ্যা x, ধ্রুবক conv2d_dw_params & p) {
        ফেরত n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    স্থির পূর্ণসংখ্যা kernel_সূচক(পূর্ণসংখ্যা c, পূর্ণসংখ্যা ky, পূর্ণসংখ্যা kx, ধ্রুবক conv2d_dw_params & p) {
        ফেরত (ky * p.kernel_w + kx) * p.channels + c;
    }
    স্থির পূর্ণসংখ্যা output_সূচক(পূর্ণসংখ্যা n, পূর্ণসংখ্যা c, পূর্ণসংখ্যা y, পূর্ণসংখ্যা x, ধ্রুবক conv2d_dw_params & p) {
        ফেরত n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    স্থির শূন্য unpack_indices(পূর্ণসংখ্যা global_idx, ধ্রুবক conv2d_dw_params & p,
                               পূর্ণসংখ্যা & n, পূর্ণসংখ্যা & c, পূর্ণসংখ্যা & out_y, পূর্ণসংখ্যা & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

template <typename KernelT, typename Layout>
স্থির শূন্য conv2d_dw_kernel(ধ্রুবক দশমিক * প্রবেশ, ধ্রুবক KernelT * kernel, দশমিক * output,
                             ধ্রুবক conv2d_dw_params p, ধ্রুবক sycl::nd_item<3> & item_ct1) {
    ধ্রুবক পূর্ণসংখ্যা global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ধ্রুবক পূর্ণসংখ্যা যোগ_elements = p.batches * p.channels * p.out_h * p.out_w;

    যদি (global_idx >= যোগ_elements) {
        ফেরত;
    }

    পূর্ণসংখ্যা n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    দশমিক acc = 0.0f;
    ধ্রুবক conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    জন্য (পূর্ণসংখ্যা ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ধ্রুবক পূর্ণসংখ্যা in_y = dw_calculate_প্রবেশ_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        জন্য (পূর্ণসংখ্যা kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ধ্রুবক পূর্ণসংখ্যা in_x = dw_calculate_প্রবেশ_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += প্রবেশ[Layout::প্রবেশ_সূচক(n, c, in_y, in_x, p)] *
                   static_cast<দশমিক>(kernel[Layout::kernel_সূচক(c, ky, kx, p)]);
        }
    }

    output[Layout::output_সূচক(n, c, out_y, out_x, p)] = acc;
}

template <typename KernelT, typename Layout>
স্থির শূন্য conv2d_dw_sycl(ধ্রুবক দশমিক * x_d, ধ্রুবক KernelT * w_d, দশমিক * y_d,
                            ধ্রুবক conv2d_dw_params p, ধ্রুবক queue_ptr & stream) {
    ধ্রুবক পূর্ণসংখ্যা যোগ      = p.batches * p.channels * p.out_h * p.out_w;
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (যোগ + SYCL_CONV2_D_DW_BLOCK_আকার - 1) / SYCL_CONV2_D_DW_BLOCK_আকার;
    ধ্রুবক sycl::range<3> block_dims(1, 1, SYCL_CONV2_D_DW_BLOCK_আকার);
    ধ্রুবক sycl::range<3> block_nums(1, 1, অঙ্ক_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

শূন্য ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ধ্রুবক ggml_tensor * kernel = dst->src[0];
    ধ্রুবক ggml_tensor * প্রবেশ  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                প্রবেশ->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ধ্রুবক দশমিক * x_d = (ধ্রুবক দশমিক *) প্রবেশ->তথ্য;
    দশমিক *       y_d = (দশমিক *) dst->তথ্য;

    ধ্রুবক int32_t * p          = (ধ্রুবক int32_t *) dst->op_params;
    ধ্রুবক পূর্ণসংখ্যা       stride_x   = p[0];
    ধ্রুবক পূর্ণসংখ্যা       stride_y   = p[1];
    ধ্রুবক পূর্ণসংখ্যা       padding_x  = p[2];
    ধ্রুবক পূর্ণসংখ্যা       padding_y  = p[3];
    ধ্রুবক পূর্ণসংখ্যা       dilation_x = p[4];
    ধ্রুবক পূর্ণসংখ্যা       dilation_y = p[5];

    ধ্রুবক পূর্ণসংখ্যা in_w     = প্রবেশ->ne[0];
    ধ্রুবক পূর্ণসংখ্যা in_h     = প্রবেশ->ne[1];
    ধ্রুবক পূর্ণসংখ্যা kernel_w = kernel->ne[0];
    ধ্রুবক পূর্ণসংখ্যা kernel_h = kernel->ne[1];
    ধ্রুবক পূর্ণসংখ্যা out_w    = dst->ne[0];
    ধ্রুবক পূর্ণসংখ্যা out_h    = dst->ne[1];
    ধ্রুবক পূর্ণসংখ্যা channels = dst->ne[2];
    ধ্রুবক পূর্ণসংখ্যা batches  = dst->ne[3];

    ধ্রুবক conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ধ্রুবক queue_ptr stream = ctx.stream();

    যদি (kernel->type == GGML_TYPE_F16) {
        ধ্রুবক sycl::half * w_d = (ধ্রুবক sycl::half *) kernel->তথ্য;
        যদি (ggml_is_contiguous(প্রবেশ)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } অন্যথা যদি (ggml_is_contiguous_channels(প্রবেশ)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } অন্যথা {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } অন্যথা {
        ধ্রুবক দশমিক * w_d = (ধ্রুবক দশমিক *) kernel->তথ্য;
        যদি (ggml_is_contiguous(প্রবেশ)) {
            conv2d_dw_sycl<দশমিক, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } অন্যথা যদি (ggml_is_contiguous_channels(প্রবেশ)) {
            conv2d_dw_sycl<দশমিক, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } অন্যথা {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
