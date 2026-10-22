#অন্তর্ভাৱ "conv2d-dw.hpp"

সংরচনা conv2d_dw_params {
    পূর্ণাঙ্ক in_w, in_h;
    পূর্ণাঙ্ক out_w, out_h;
    পূর্ণাঙ্ক kernel_w, kernel_h;
    পূর্ণাঙ্ক stride_x, stride_y;
    পূর্ণাঙ্ক padding_x, padding_y;
    পূর্ণাঙ্ক dilation_x, dilation_y;
    পূর্ণাঙ্ক channels, batches;
};

সংরচনা conv2d_dw_kernel_bounds {
    পূর্ণাঙ্ক y_min, y_max;
    পূর্ণাঙ্ক x_min, x_max;
};

স্থির অন্তঃস্থ conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(পূর্ণাঙ্ক out_x, পূর্ণাঙ্ক out_y,
                                                                  ধ্রুৱ conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    ঘূৰাই bounds;
}

স্থির অন্তঃস্থ পূর্ণাঙ্ক dw_calculate_প্ৰবেশ_coord(পূর্ণাঙ্ক out_coord, পূর্ণাঙ্ক kern_coord, পূর্ণাঙ্ক stride, পূর্ণাঙ্ক dilation, পূর্ণাঙ্ক padding) {
    ঘূৰাই out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
সংরচনা dw_whcn_layout {
    স্থির পূর্ণাঙ্ক প্ৰবেশ_index(পূর্ণাঙ্ক n, পূর্ণাঙ্ক c, পূর্ণাঙ্ক y, পূর্ণাঙ্ক x, ধ্রুৱ conv2d_dw_params & p) {
        ঘূৰাই n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    স্থির পূর্ণাঙ্ক kernel_index(পূর্ণাঙ্ক c, পূর্ণাঙ্ক ky, পূর্ণাঙ্ক kx, ধ্রুৱ conv2d_dw_params & p) {
        ঘূৰাই c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    স্থির পূর্ণাঙ্ক output_index(পূর্ণাঙ্ক n, পূর্ণাঙ্ক c, পূর্ণাঙ্ক y, পূর্ণাঙ্ক x, ধ্রুৱ conv2d_dw_params & p) {
        ঘূৰাই n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    স্থির শূন্য unpack_indices(পূর্ণাঙ্ক global_idx, ধ্রুৱ conv2d_dw_params & p,
                               পূর্ণাঙ্ক & n, পূর্ণাঙ্ক & c, পূর্ণাঙ্ক & out_y, পূর্ণাঙ্ক & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
সংরচনা dw_cwhn_layout {
    স্থির পূর্ণাঙ্ক প্ৰবেশ_index(পূর্ণাঙ্ক n, পূর্ণাঙ্ক c, পূর্ণাঙ্ক y, পূর্ণাঙ্ক x, ধ্রুৱ conv2d_dw_params & p) {
        ঘূৰাই n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    স্থির পূর্ণাঙ্ক kernel_index(পূর্ণাঙ্ক c, পূর্ণাঙ্ক ky, পূর্ণাঙ্ক kx, ধ্রুৱ conv2d_dw_params & p) {
        ঘূৰাই (ky * p.kernel_w + kx) * p.channels + c;
    }
    স্থির পূর্ণাঙ্ক output_index(পূর্ণাঙ্ক n, পূর্ণাঙ্ক c, পূর্ণাঙ্ক y, পূর্ণাঙ্ক x, ধ্রুৱ conv2d_dw_params & p) {
        ঘূৰাই n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    স্থির শূন্য unpack_indices(পূর্ণাঙ্ক global_idx, ধ্রুৱ conv2d_dw_params & p,
                               পূর্ণাঙ্ক & n, পূর্ণাঙ্ক & c, পূর্ণাঙ্ক & out_y, পূর্ণাঙ্ক & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

template <typename KernelT, typename Layout>
স্থির শূন্য conv2d_dw_kernel(ধ্রুৱ ভিন্ন * প্ৰবেশ, ধ্রুৱ KernelT * kernel, ভিন্ন * output,
                             ধ্রুৱ conv2d_dw_params p, ধ্রুৱ sycl::nd_item<3> & item_ct1) {
    ধ্রুৱ পূর্ণাঙ্ক global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ধ্রুৱ পূর্ণাঙ্ক total_elements = p.batches * p.channels * p.out_h * p.out_w;

    যদি (global_idx >= total_elements) {
        ঘূৰাই;
    }

    পূর্ণাঙ্ক n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    ভিন্ন acc = 0.0f;
    ধ্রুৱ conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    বাবে (পূর্ণাঙ্ক ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ধ্রুৱ পূর্ণাঙ্ক in_y = dw_calculate_প্ৰবেশ_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        বাবে (পূর্ণাঙ্ক kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ধ্রুৱ পূর্ণাঙ্ক in_x = dw_calculate_প্ৰবেশ_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += প্ৰবেশ[Layout::প্ৰবেশ_index(n, c, in_y, in_x, p)] *
                   static_cast<ভিন্ন>(kernel[Layout::kernel_index(c, ky, kx, p)]);
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

template <typename KernelT, typename Layout>
স্থির শূন্য conv2d_dw_sycl(ধ্রুৱ ভিন্ন * x_d, ধ্রুৱ KernelT * w_d, ভিন্ন * y_d,
                            ধ্রুৱ conv2d_dw_params p, ধ্রুৱ queue_ptr & stream) {
    ধ্রুৱ পূর্ণাঙ্ক total      = p.batches * p.channels * p.out_h * p.out_w;
    ধ্রুৱ পূর্ণাঙ্ক অঙ্ক_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
    ধ্রুৱ sycl::range<3> block_dims(1, 1, SYCL_CONV2D_DW_BLOCK_SIZE);
    ধ্রুৱ sycl::range<3> block_nums(1, 1, অঙ্ক_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

শূন্য ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ধ্রুৱ ggml_tensor * kernel = dst->src[0];
    ধ্রুৱ ggml_tensor * প্ৰবেশ  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                প্ৰবেশ->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ধ্রুৱ ভিন্ন * x_d = (ধ্রুৱ ভিন্ন *) প্ৰবেশ->data;
    ভিন্ন *       y_d = (ভিন্ন *) dst->data;

    ধ্রুৱ int32_t * p          = (ধ্রুৱ int32_t *) dst->op_params;
    ধ্রুৱ পূর্ণাঙ্ক       stride_x   = p[0];
    ধ্রুৱ পূর্ণাঙ্ক       stride_y   = p[1];
    ধ্রুৱ পূর্ণাঙ্ক       padding_x  = p[2];
    ধ্রুৱ পূর্ণাঙ্ক       padding_y  = p[3];
    ধ্রুৱ পূর্ণাঙ্ক       dilation_x = p[4];
    ধ্রুৱ পূর্ণাঙ্ক       dilation_y = p[5];

    ধ্রুৱ পূর্ণাঙ্ক in_w     = প্ৰবেশ->ne[0];
    ধ্রুৱ পূর্ণাঙ্ক in_h     = প্ৰবেশ->ne[1];
    ধ্রুৱ পূর্ণাঙ্ক kernel_w = kernel->ne[0];
    ধ্রুৱ পূর্ণাঙ্ক kernel_h = kernel->ne[1];
    ধ্রুৱ পূর্ণাঙ্ক out_w    = dst->ne[0];
    ধ্রুৱ পূর্ণাঙ্ক out_h    = dst->ne[1];
    ধ্রুৱ পূর্ণাঙ্ক channels = dst->ne[2];
    ধ্রুৱ পূর্ণাঙ্ক batches  = dst->ne[3];

    ধ্রুৱ conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ধ্রুৱ queue_ptr stream = ctx.stream();

    যদি (kernel->type == GGML_TYPE_F16) {
        ধ্রুৱ sycl::half * w_d = (ধ্রুৱ sycl::half *) kernel->data;
        যদি (ggml_is_contiguous(প্ৰবেশ)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } অন্যথা যদি (ggml_is_contiguous_channels(প্ৰবেশ)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } অন্যথা {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } অন্যথা {
        ধ্রুৱ ভিন্ন * w_d = (ধ্রুৱ ভিন্ন *) kernel->data;
        যদি (ggml_is_contiguous(প্ৰবেশ)) {
            conv2d_dw_sycl<ভিন্ন, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } অন্যথা যদি (ggml_is_contiguous_channels(প্ৰবেশ)) {
            conv2d_dw_sycl<ভিন্ন, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } অন্যথা {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
