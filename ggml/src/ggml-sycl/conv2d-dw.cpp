#अन्तर्भाव "conv2d-dw.hpp"

संरचना conv2d_dw_params {
    पूर्णांक in_w, in_h;
    पूर्णांक out_w, out_h;
    पूर्णांक kernel_w, kernel_h;
    पूर्णांक stride_x, stride_y;
    पूर्णांक padding_x, padding_y;
    पूर्णांक dilation_x, dilation_y;
    पूर्णांक channels, batches;
};

संरचना conv2d_dw_kernel_bounds {
    पूर्णांक y_min, y_max;
    पूर्णांक x_min, x_max;
};

स्थिर ओळीत conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(पूर्णांक out_x, पूर्णांक out_y,
                                                                  स्थिरांक conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    परत bounds;
}

स्थिर ओळीत पूर्णांक dw_calculate_आदान_coord(पूर्णांक out_coord, पूर्णांक kern_coord, पूर्णांक stride, पूर्णांक dilation, पूर्णांक padding) {
    परत out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
संरचना dw_whcn_layout {
    स्थिर पूर्णांक आदान_index(पूर्णांक n, पूर्णांक c, पूर्णांक y, पूर्णांक x, स्थिरांक conv2d_dw_params & p) {
        परत n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    स्थिर पूर्णांक kernel_index(पूर्णांक c, पूर्णांक ky, पूर्णांक kx, स्थिरांक conv2d_dw_params & p) {
        परत c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    स्थिर पूर्णांक output_index(पूर्णांक n, पूर्णांक c, पूर्णांक y, पूर्णांक x, स्थिरांक conv2d_dw_params & p) {
        परत n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    स्थिर शून्य unpack_indices(पूर्णांक global_idx, स्थिरांक conv2d_dw_params & p,
                               पूर्णांक & n, पूर्णांक & c, पूर्णांक & out_y, पूर्णांक & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
संरचना dw_cwhn_layout {
    स्थिर पूर्णांक आदान_index(पूर्णांक n, पूर्णांक c, पूर्णांक y, पूर्णांक x, स्थिरांक conv2d_dw_params & p) {
        परत n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    स्थिर पूर्णांक kernel_index(पूर्णांक c, पूर्णांक ky, पूर्णांक kx, स्थिरांक conv2d_dw_params & p) {
        परत (ky * p.kernel_w + kx) * p.channels + c;
    }
    स्थिर पूर्णांक output_index(पूर्णांक n, पूर्णांक c, पूर्णांक y, पूर्णांक x, स्थिरांक conv2d_dw_params & p) {
        परत n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    स्थिर शून्य unpack_indices(पूर्णांक global_idx, स्थिरांक conv2d_dw_params & p,
                               पूर्णांक & n, पूर्णांक & c, पूर्णांक & out_y, पूर्णांक & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

ढाँचा <typename KernelT, typename Layout>
स्थिर शून्य conv2d_dw_kernel(स्थिरांक दशांश * आदान, स्थिरांक KernelT * kernel, दशांश * output,
                             स्थिरांक conv2d_dw_params p, स्थिरांक sycl::nd_item<3> & item_ct1) {
    स्थिरांक पूर्णांक global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    स्थिरांक पूर्णांक total_elements = p.batches * p.channels * p.out_h * p.out_w;

    जर (global_idx >= total_elements) {
        परत;
    }

    पूर्णांक n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    दशांश acc = 0.0f;
    स्थिरांक conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    साठी (पूर्णांक ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        स्थिरांक पूर्णांक in_y = dw_calculate_आदान_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        साठी (पूर्णांक kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            स्थिरांक पूर्णांक in_x = dw_calculate_आदान_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += आदान[Layout::आदान_index(n, c, in_y, in_x, p)] *
                   static_cast<दशांश>(kernel[Layout::kernel_index(c, ky, kx, p)]);
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

ढाँचा <typename KernelT, typename Layout>
स्थिर शून्य conv2d_dw_sycl(स्थिरांक दशांश * x_d, स्थिरांक KernelT * w_d, दशांश * y_d,
                            स्थिरांक conv2d_dw_params p, स्थिरांक queue_ptr & stream) {
    स्थिरांक पूर्णांक total      = p.batches * p.channels * p.out_h * p.out_w;
    स्थिरांक पूर्णांक अंक_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
    स्थिरांक sycl::range<3> block_dims(1, 1, SYCL_CONV2D_DW_BLOCK_SIZE);
    स्थिरांक sycl::range<3> block_nums(1, 1, अंक_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

शून्य ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    स्थिरांक ggml_tensor * kernel = dst->src[0];
    स्थिरांक ggml_tensor * आदान  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                आदान->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    स्थिरांक दशांश * x_d = (स्थिरांक दशांश *) आदान->data;
    दशांश *       y_d = (दशांश *) dst->data;

    स्थिरांक int32_t * p          = (स्थिरांक int32_t *) dst->op_params;
    स्थिरांक पूर्णांक       stride_x   = p[0];
    स्थिरांक पूर्णांक       stride_y   = p[1];
    स्थिरांक पूर्णांक       padding_x  = p[2];
    स्थिरांक पूर्णांक       padding_y  = p[3];
    स्थिरांक पूर्णांक       dilation_x = p[4];
    स्थिरांक पूर्णांक       dilation_y = p[5];

    स्थिरांक पूर्णांक in_w     = आदान->ne[0];
    स्थिरांक पूर्णांक in_h     = आदान->ne[1];
    स्थिरांक पूर्णांक kernel_w = kernel->ne[0];
    स्थिरांक पूर्णांक kernel_h = kernel->ne[1];
    स्थिरांक पूर्णांक out_w    = dst->ne[0];
    स्थिरांक पूर्णांक out_h    = dst->ne[1];
    स्थिरांक पूर्णांक channels = dst->ne[2];
    स्थिरांक पूर्णांक batches  = dst->ne[3];

    स्थिरांक conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    स्थिरांक queue_ptr stream = ctx.stream();

    जर (kernel->type == GGML_TYPE_F16) {
        स्थिरांक sycl::half * w_d = (स्थिरांक sycl::half *) kernel->data;
        जर (ggml_is_contiguous(आदान)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा जर (ggml_is_contiguous_channels(आदान)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } अन्यथा {
        स्थिरांक दशांश * w_d = (स्थिरांक दशांश *) kernel->data;
        जर (ggml_is_contiguous(आदान)) {
            conv2d_dw_sycl<दशांश, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा जर (ggml_is_contiguous_channels(आदान)) {
            conv2d_dw_sycl<दशांश, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
