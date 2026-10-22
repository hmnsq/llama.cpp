#ಅನ್ತರ್ಭಾವ "conv2d-dw.hpp"

ರಚನೆ conv2d_dw_params {
    ಪೂರ್ಣಾಂಕ ಒಳ_w, ಒಳ_h;
    ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_w, ಹೊರಹೋಗು_h;
    ಪೂರ್ಣಾಂಕ kernel_w, kernel_h;
    ಪೂರ್ಣಾಂಕ stride_x, stride_y;
    ಪೂರ್ಣಾಂಕ padding_x, padding_y;
    ಪೂರ್ಣಾಂಕ dilation_x, dilation_y;
    ಪೂರ್ಣಾಂಕ channels, batches;
};

ರಚನೆ conv2d_dw_kernel_bounds {
    ಪೂರ್ಣಾಂಕ y_min, y_max;
    ಪೂರ್ಣಾಂಕ x_min, x_max;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_x, ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_y,
                                                                  ಸ್ಥಿರಾಂಕ conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - ಹೊರಹೋಗು_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.ಒಳ_h + p.padding_y - ಹೊರಹೋಗು_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - ಹೊರಹೋಗು_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.ಒಳ_w + p.padding_x - ಹೊರಹೋಗು_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    ಹಿಂತಿರುಗಿ bounds;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ dw_calculate_ಒಳಹರಿವು_coord(ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_coord, ಪೂರ್ಣಾಂಕ kern_coord, ಪೂರ್ಣಾಂಕ stride, ಪೂರ್ಣಾಂಕ dilation, ಪೂರ್ಣಾಂಕ padding) {
    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
ರಚನೆ dw_whcn_layout {
    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ(ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ c, ಪೂರ್ಣಾಂಕ y, ಪೂರ್ಣಾಂಕ x, ಸ್ಥಿರಾಂಕ conv2d_dw_params & p) {
        ಹಿಂತಿರುಗಿ n * (p.channels * p.ಒಳ_w * p.ಒಳ_h) + c * p.ಒಳ_w * p.ಒಳ_h + y * p.ಒಳ_w + x;
    }
    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ kernel_ಸೂಚ್ಯಂಕ(ಪೂರ್ಣಾಂಕ c, ಪೂರ್ಣಾಂಕ ky, ಪೂರ್ಣಾಂಕ kx, ಸ್ಥಿರಾಂಕ conv2d_dw_params & p) {
        ಹಿಂತಿರುಗಿ c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ output_ಸೂಚ್ಯಂಕ(ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ c, ಪೂರ್ಣಾಂಕ y, ಪೂರ್ಣಾಂಕ x, ಸ್ಥಿರಾಂಕ conv2d_dw_params & p) {
        ಹಿಂತಿರುಗಿ n * (p.channels * p.ಹೊರಹೋಗು_w * p.ಹೊರಹೋಗು_h) + c * p.ಹೊರಹೋಗು_w * p.ಹೊರಹೋಗು_h + y * p.ಹೊರಹೋಗು_w + x;
    }
    ಸ್ಥಿರ ಶೂನ್ಯ unpack_indices(ಪೂರ್ಣಾಂಕ global_idx, ಸ್ಥಿರಾಂಕ conv2d_dw_params & p,
                               ಪೂರ್ಣಾಂಕ & n, ಪೂರ್ಣಾಂಕ & c, ಪೂರ್ಣಾಂಕ & ಹೊರಹೋಗು_y, ಪೂರ್ಣಾಂಕ & ಹೊರಹೋಗು_x) {
        ಹೊರಹೋಗು_x  = global_idx % p.ಹೊರಹೋಗು_w;
        ಹೊರಹೋಗು_y  = (global_idx / p.ಹೊರಹೋಗು_w) % p.ಹೊರಹೋಗು_h;
        c      = (global_idx / (p.ಹೊರಹೋಗು_w * p.ಹೊರಹೋಗು_h)) % p.channels;
        n      = global_idx / (p.ಹೊರಹೋಗು_w * p.ಹೊರಹೋಗು_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
ರಚನೆ dw_cwhn_layout {
    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ(ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ c, ಪೂರ್ಣಾಂಕ y, ಪೂರ್ಣಾಂಕ x, ಸ್ಥಿರಾಂಕ conv2d_dw_params & p) {
        ಹಿಂತಿರುಗಿ n * (p.channels * p.ಒಳ_w * p.ಒಳ_h) + (y * p.ಒಳ_w + x) * p.channels + c;
    }
    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ kernel_ಸೂಚ್ಯಂಕ(ಪೂರ್ಣಾಂಕ c, ಪೂರ್ಣಾಂಕ ky, ಪೂರ್ಣಾಂಕ kx, ಸ್ಥಿರಾಂಕ conv2d_dw_params & p) {
        ಹಿಂತಿರುಗಿ (ky * p.kernel_w + kx) * p.channels + c;
    }
    ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ output_ಸೂಚ್ಯಂಕ(ಪೂರ್ಣಾಂಕ n, ಪೂರ್ಣಾಂಕ c, ಪೂರ್ಣಾಂಕ y, ಪೂರ್ಣಾಂಕ x, ಸ್ಥಿರಾಂಕ conv2d_dw_params & p) {
        ಹಿಂತಿರುಗಿ n * (p.channels * p.ಹೊರಹೋಗು_w * p.ಹೊರಹೋಗು_h) + y * (p.ಹೊರಹೋಗು_w * p.channels) + x * p.channels + c;
    }
    ಸ್ಥಿರ ಶೂನ್ಯ unpack_indices(ಪೂರ್ಣಾಂಕ global_idx, ಸ್ಥಿರಾಂಕ conv2d_dw_params & p,
                               ಪೂರ್ಣಾಂಕ & n, ಪೂರ್ಣಾಂಕ & c, ಪೂರ್ಣಾಂಕ & ಹೊರಹೋಗು_y, ಪೂರ್ಣಾಂಕ & ಹೊರಹೋಗು_x) {
        c      = global_idx % p.channels;
        ಹೊರಹೋಗು_x  = (global_idx / p.channels) % p.ಹೊರಹೋಗು_w;
        ಹೊರಹೋಗು_y  = (global_idx / (p.channels * p.ಹೊರಹೋಗು_w)) % p.ಹೊರಹೋಗು_h;
        n      = global_idx / (p.channels * p.ಹೊರಹೋಗು_w * p.ಹೊರಹೋಗು_h);
    }
};

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು KernelT, ಬಗೆ_ಹೆಸರು Layout>
ಸ್ಥಿರ ಶೂನ್ಯ conv2d_dw_kernel(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಒಳಹರಿವು, ಸ್ಥಿರಾಂಕ KernelT * kernel, ದಶಮಾಂಶ * output,
                             ಸ್ಥಿರಾಂಕ conv2d_dw_params p, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ global_idx     = ಅಂಶ_ct1.get_local_id(2) +
                               ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು_elements = p.batches * p.channels * p.ಹೊರಹೋಗು_h * p.ಹೊರಹೋಗು_w;

    ಆದರೆ (global_idx >= ಒಟ್ಟು_elements) {
        ಹಿಂತಿರುಗಿ;
    }

    ಪೂರ್ಣಾಂಕ n, c, ಹೊರಹೋಗು_y, ಹೊರಹೋಗು_x;
    Layout::unpack_indices(global_idx, p, n, c, ಹೊರಹೋಗು_y, ಹೊರಹೋಗು_x);

    ದಶಮಾಂಶ acc = 0.0f;
    ಸ್ಥಿರಾಂಕ conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(ಹೊರಹೋಗು_x, ಹೊರಹೋಗು_y, p);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ_y = dw_calculate_ಒಳಹರಿವು_coord(ಹೊರಹೋಗು_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ_x = dw_calculate_ಒಳಹರಿವು_coord(ಹೊರಹೋಗು_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += ಒಳಹರಿವು[Layout::ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ(n, c, ಒಳ_y, ಒಳ_x, p)] *
                   static_cast<ದಶಮಾಂಶ>(kernel[Layout::kernel_ಸೂಚ್ಯಂಕ(c, ky, kx, p)]);
        }
    }

    output[Layout::output_ಸೂಚ್ಯಂಕ(n, c, ಹೊರಹೋಗು_y, ಹೊರಹೋಗು_x, p)] = acc;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು KernelT, ಬಗೆ_ಹೆಸರು Layout>
ಸ್ಥಿರ ಶೂನ್ಯ conv2d_dw_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_d, ಸ್ಥಿರಾಂಕ KernelT * w_d, ದಶಮಾಂಶ * y_d,
                            ಸ್ಥಿರಾಂಕ conv2d_dw_params p, ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr & stream) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು      = p.batches * p.channels * p.ಹೊರಹೋಗು_h * p.ಹೊರಹೋಗು_w;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = (ಒಟ್ಟು + SYCL_CONV2_D_DW_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_CONV2_D_DW_ಖಂಡ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, SYCL_CONV2_D_DW_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಅಂಕೆ_ಖಂಡಗಳು);
    stream->parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, ಅಂಶ_ct1);
        });
}

ಶೂನ್ಯ ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ಸ್ಥಿರಾಂಕ ggml_tensor * kernel = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಒಳಹರಿವು  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                ಒಳಹರಿವು->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಒಳಹರಿವು->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *       y_d = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ int32_t * p          = (ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       stride_x   = p[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       stride_y   = p[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       padding_x  = p[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       padding_y  = p[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       dilation_x = p[4];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       dilation_y = p[5];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ_w     = ಒಳಹರಿವು->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ_h     = ಒಳಹರಿವು->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kernel_w = kernel->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kernel_h = kernel->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_w    = dst->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_h    = dst->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channels = dst->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ batches  = dst->ne[3];

    ಸ್ಥಿರಾಂಕ conv2d_dw_params params = { ಒಳ_w, ಒಳ_h, ಹೊರಹೋಗು_w, ಹೊರಹೋಗು_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream = ctx.stream();

    ಆದರೆ (kernel->type == GGML_TYPE_F16) {
        ಸ್ಥಿರಾಂಕ sycl::half * w_d = (ಸ್ಥಿರಾಂಕ sycl::half *) kernel->ದತ್ತಾಂಶ;
        ಆದರೆ (ggml_is_contiguous(ಒಳಹರಿವು)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_is_contiguous_channels(ಒಳಹರಿವು)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * w_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) kernel->ದತ್ತಾಂಶ;
        ಆದರೆ (ggml_is_contiguous(ಒಳಹರಿವು)) {
            conv2d_dw_sycl<ದಶಮಾಂಶ, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ggml_is_contiguous_channels(ಒಳಹರಿವು)) {
            conv2d_dw_sycl<ದಶಮಾಂಶ, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } ಇಲ್ಲದಿದ್ದರೆ {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
