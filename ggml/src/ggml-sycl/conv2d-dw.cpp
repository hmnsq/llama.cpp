#ಅನ್ತರ್ಭಾವ "conv2d-dw.hpp"

ಸಂರಚನಾ conv2d_dw_params {
    ಪೂರ್ಣಾಙ್ಕ in_w, in_h;
    ಪೂರ್ಣಾಙ್ಕ out_w, out_h;
    ಪೂರ್ಣಾಙ್ಕ kernel_w, kernel_h;
    ಪೂರ್ಣಾಙ್ಕ stride_x, stride_y;
    ಪೂರ್ಣಾಙ್ಕ padding_x, padding_y;
    ಪೂರ್ಣಾಙ್ಕ dilation_x, dilation_y;
    ಪೂರ್ಣಾಙ್ಕ channels, batches;
};

ಸಂರಚನಾ conv2d_dw_kernel_bounds {
    ಪೂರ್ಣಾಙ್ಕ y_min, y_max;
    ಪೂರ್ಣಾಙ್ಕ x_min, x_max;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(ಪೂರ್ಣಾಙ್ಕ out_x, ಪೂರ್ಣಾಙ್ಕ out_y,
                                                                  ಧ್ರುವ conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    ಪ್ರತ್ಯಾಗಮ bounds;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ dw_calculate_ಒಳಹರಿವು_coord(ಪೂರ್ಣಾಙ್ಕ out_coord, ಪೂರ್ಣಾಙ್ಕ kern_coord, ಪೂರ್ಣಾಙ್ಕ stride, ಪೂರ್ಣಾಙ್ಕ dilation, ಪೂರ್ಣಾಙ್ಕ padding) {
    ಪ್ರತ್ಯಾಗಮ out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
ಸಂರಚನಾ dw_whcn_layout {
    ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_index(ಪೂರ್ಣಾಙ್ಕ n, ಪೂರ್ಣಾಙ್ಕ c, ಪೂರ್ಣಾಙ್ಕ y, ಪೂರ್ಣಾಙ್ಕ x, ಧ್ರುವ conv2d_dw_params & p) {
        ಪ್ರತ್ಯಾಗಮ n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ kernel_index(ಪೂರ್ಣಾಙ್ಕ c, ಪೂರ್ಣಾಙ್ಕ ky, ಪೂರ್ಣಾಙ್ಕ kx, ಧ್ರುವ conv2d_dw_params & p) {
        ಪ್ರತ್ಯಾಗಮ c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ output_index(ಪೂರ್ಣಾಙ್ಕ n, ಪೂರ್ಣಾಙ್ಕ c, ಪೂರ್ಣಾಙ್ಕ y, ಪೂರ್ಣಾಙ್ಕ x, ಧ್ರುವ conv2d_dw_params & p) {
        ಪ್ರತ್ಯಾಗಮ n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    ಸ್ಥಿರ ಶೂನ್ಯ unpack_indices(ಪೂರ್ಣಾಙ್ಕ global_idx, ಧ್ರುವ conv2d_dw_params & p,
                               ಪೂರ್ಣಾಙ್ಕ & n, ಪೂರ್ಣಾಙ್ಕ & c, ಪೂರ್ಣಾಙ್ಕ & out_y, ಪೂರ್ಣಾಙ್ಕ & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
ಸಂರಚನಾ dw_cwhn_layout {
    ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ ಒಳಹರಿವು_index(ಪೂರ್ಣಾಙ್ಕ n, ಪೂರ್ಣಾಙ್ಕ c, ಪೂರ್ಣಾಙ್ಕ y, ಪೂರ್ಣಾಙ್ಕ x, ಧ್ರುವ conv2d_dw_params & p) {
        ಪ್ರತ್ಯಾಗಮ n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ kernel_index(ಪೂರ್ಣಾಙ್ಕ c, ಪೂರ್ಣಾಙ್ಕ ky, ಪೂರ್ಣಾಙ್ಕ kx, ಧ್ರುವ conv2d_dw_params & p) {
        ಪ್ರತ್ಯಾಗಮ (ky * p.kernel_w + kx) * p.channels + c;
    }
    ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ output_index(ಪೂರ್ಣಾಙ್ಕ n, ಪೂರ್ಣಾಙ್ಕ c, ಪೂರ್ಣಾಙ್ಕ y, ಪೂರ್ಣಾಙ್ಕ x, ಧ್ರುವ conv2d_dw_params & p) {
        ಪ್ರತ್ಯಾಗಮ n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    ಸ್ಥಿರ ಶೂನ್ಯ unpack_indices(ಪೂರ್ಣಾಙ್ಕ global_idx, ಧ್ರುವ conv2d_dw_params & p,
                               ಪೂರ್ಣಾಙ್ಕ & n, ಪೂರ್ಣಾಙ್ಕ & c, ಪೂರ್ಣಾಙ್ಕ & out_y, ಪೂರ್ಣಾಙ್ಕ & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

template <typename KernelT, typename Layout>
ಸ್ಥಿರ ಶೂನ್ಯ conv2d_dw_kernel(ಧ್ರುವ ಭಿನ್ನ * ಒಳಹರಿವು, ಧ್ರುವ KernelT * kernel, ಭಿನ್ನ * output,
                             ಧ್ರುವ conv2d_dw_params p, ಧ್ರುವ sycl::nd_item<3> & item_ct1) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ total_elements = p.batches * p.channels * p.out_h * p.out_w;

    ಯದಿ (global_idx >= total_elements) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಪೂರ್ಣಾಙ್ಕ n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    ಭಿನ್ನ acc = 0.0f;
    ಧ್ರುವ conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ in_y = dw_calculate_ಒಳಹರಿವು_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ in_x = dw_calculate_ಒಳಹರಿವು_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += ಒಳಹರಿವು[Layout::ಒಳಹರಿವು_index(n, c, in_y, in_x, p)] *
                   static_cast<ಭಿನ್ನ>(kernel[Layout::kernel_index(c, ky, kx, p)]);
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

template <typename KernelT, typename Layout>
ಸ್ಥಿರ ಶೂನ್ಯ conv2d_dw_sycl(ಧ್ರುವ ಭಿನ್ನ * x_d, ಧ್ರುವ KernelT * w_d, ಭಿನ್ನ * y_d,
                            ಧ್ರುವ conv2d_dw_params p, ಧ್ರುವ queue_ptr & stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ total      = p.batches * p.channels * p.out_h * p.out_w;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
    ಧ್ರುವ sycl::range<3> block_dims(1, 1, SYCL_CONV2D_DW_BLOCK_SIZE);
    ಧ್ರುವ sycl::range<3> block_nums(1, 1, ಅಂಕೆ_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

ಶೂನ್ಯ ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ಧ್ರುವ ggml_tensor * kernel = dst->src[0];
    ಧ್ರುವ ggml_tensor * ಒಳಹರಿವು  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                ಒಳಹರಿವು->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ಧ್ರುವ ಭಿನ್ನ * x_d = (ಧ್ರುವ ಭಿನ್ನ *) ಒಳಹರಿವು->data;
    ಭಿನ್ನ *       y_d = (ಭಿನ್ನ *) dst->data;

    ಧ್ರುವ int32_t * p          = (ಧ್ರುವ int32_t *) dst->op_params;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       stride_x   = p[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       stride_y   = p[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       padding_x  = p[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       padding_y  = p[3];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       dilation_x = p[4];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       dilation_y = p[5];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ in_w     = ಒಳಹರಿವು->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ in_h     = ಒಳಹರಿವು->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kernel_w = kernel->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ kernel_h = kernel->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ out_w    = dst->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ out_h    = dst->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ channels = dst->ne[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ batches  = dst->ne[3];

    ಧ್ರುವ conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ಧ್ರುವ queue_ptr stream = ctx.stream();

    ಯದಿ (kernel->type == GGML_TYPE_F16) {
        ಧ್ರುವ sycl::half * w_d = (ಧ್ರುವ sycl::half *) kernel->data;
        ಯದಿ (ggml_is_contiguous(ಒಳಹರಿವು)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } ಅನ್ಯಥಾ ಯದಿ (ggml_is_contiguous_channels(ಒಳಹರಿವು)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } ಅನ್ಯಥಾ {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } ಅನ್ಯಥಾ {
        ಧ್ರುವ ಭಿನ್ನ * w_d = (ಧ್ರುವ ಭಿನ್ನ *) kernel->data;
        ಯದಿ (ggml_is_contiguous(ಒಳಹರಿವು)) {
            conv2d_dw_sycl<ಭಿನ್ನ, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } ಅನ್ಯಥಾ ಯದಿ (ggml_is_contiguous_channels(ಒಳಹರಿವು)) {
            conv2d_dw_sycl<ಭಿನ್ನ, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } ಅನ್ಯಥಾ {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
