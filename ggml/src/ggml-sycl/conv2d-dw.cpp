#अन्तर्भाव "conv2d-dw.hpp"

संरचना conv2d_dw_params {
    पूर्णाङ्क in_w, in_h;
    पूर्णाङ्क out_w, out_h;
    पूर्णाङ्क kernel_w, kernel_h;
    पूर्णाङ्क stride_x, stride_y;
    पूर्णाङ्क padding_x, padding_y;
    पूर्णाङ्क dilation_x, dilation_y;
    पूर्णाङ्क channels, batches;
};

संरचना conv2d_dw_kernel_bounds {
    पूर्णाङ्क y_min, y_max;
    पूर्णाङ्क x_min, x_max;
};

स्थिर अन्तःस्थ conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(पूर्णाङ्क out_x, पूर्णाङ्क out_y,
                                                                  अचर conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    फर्काऊ bounds;
}

स्थिर अन्तःस्थ पूर्णाङ्क dw_calculate_आगत_coord(पूर्णाङ्क out_coord, पूर्णाङ्क kern_coord, पूर्णाङ्क stride, पूर्णाङ्क dilation, पूर्णाङ्क padding) {
    फर्काऊ out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
संरचना dw_whcn_layout {
    स्थिर पूर्णाङ्क आगत_index(पूर्णाङ्क n, पूर्णाङ्क c, पूर्णाङ्क y, पूर्णाङ्क x, अचर conv2d_dw_params & p) {
        फर्काऊ n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    स्थिर पूर्णाङ्क kernel_index(पूर्णाङ्क c, पूर्णाङ्क ky, पूर्णाङ्क kx, अचर conv2d_dw_params & p) {
        फर्काऊ c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    स्थिर पूर्णाङ्क output_index(पूर्णाङ्क n, पूर्णाङ्क c, पूर्णाङ्क y, पूर्णाङ्क x, अचर conv2d_dw_params & p) {
        फर्काऊ n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    स्थिर शून्य unpack_indices(पूर्णाङ्क global_idx, अचर conv2d_dw_params & p,
                               पूर्णाङ्क & n, पूर्णाङ्क & c, पूर्णाङ्क & out_y, पूर्णाङ्क & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
संरचना dw_cwhn_layout {
    स्थिर पूर्णाङ्क आगत_index(पूर्णाङ्क n, पूर्णाङ्क c, पूर्णाङ्क y, पूर्णाङ्क x, अचर conv2d_dw_params & p) {
        फर्काऊ n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    स्थिर पूर्णाङ्क kernel_index(पूर्णाङ्क c, पूर्णाङ्क ky, पूर्णाङ्क kx, अचर conv2d_dw_params & p) {
        फर्काऊ (ky * p.kernel_w + kx) * p.channels + c;
    }
    स्थिर पूर्णाङ्क output_index(पूर्णाङ्क n, पूर्णाङ्क c, पूर्णाङ्क y, पूर्णाङ्क x, अचर conv2d_dw_params & p) {
        फर्काऊ n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    स्थिर शून्य unpack_indices(पूर्णाङ्क global_idx, अचर conv2d_dw_params & p,
                               पूर्णाङ्क & n, पूर्णाङ्क & c, पूर्णाङ्क & out_y, पूर्णाङ्क & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

ढाँचा <typename KernelT, typename Layout>
स्थिर शून्य conv2d_dw_kernel(अचर भिन्न * आगत, अचर KernelT * kernel, भिन्न * output,
                             अचर conv2d_dw_params p, अचर sycl::nd_item<3> & item_ct1) {
    अचर पूर्णाङ्क global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    अचर पूर्णाङ्क total_elements = p.batches * p.channels * p.out_h * p.out_w;

    यदि (global_idx >= total_elements) {
        फर्काऊ;
    }

    पूर्णाङ्क n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    भिन्न acc = 0.0f;
    अचर conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    लागि (पूर्णाङ्क ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        अचर पूर्णाङ्क in_y = dw_calculate_आगत_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        लागि (पूर्णाङ्क kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            अचर पूर्णाङ्क in_x = dw_calculate_आगत_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += आगत[Layout::आगत_index(n, c, in_y, in_x, p)] *
                   static_cast<भिन्न>(kernel[Layout::kernel_index(c, ky, kx, p)]);
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

ढाँचा <typename KernelT, typename Layout>
स्थिर शून्य conv2d_dw_sycl(अचर भिन्न * x_d, अचर KernelT * w_d, भिन्न * y_d,
                            अचर conv2d_dw_params p, अचर queue_ptr & stream) {
    अचर पूर्णाङ्क total      = p.batches * p.channels * p.out_h * p.out_w;
    अचर पूर्णाङ्क अङ्क_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
    अचर sycl::range<3> block_dims(1, 1, SYCL_CONV2D_DW_BLOCK_SIZE);
    अचर sycl::range<3> block_nums(1, 1, अङ्क_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

शून्य ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    अचर ggml_tensor * kernel = dst->src[0];
    अचर ggml_tensor * आगत  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                आगत->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    अचर भिन्न * x_d = (अचर भिन्न *) आगत->data;
    भिन्न *       y_d = (भिन्न *) dst->data;

    अचर int32_t * p          = (अचर int32_t *) dst->op_params;
    अचर पूर्णाङ्क       stride_x   = p[0];
    अचर पूर्णाङ्क       stride_y   = p[1];
    अचर पूर्णाङ्क       padding_x  = p[2];
    अचर पूर्णाङ्क       padding_y  = p[3];
    अचर पूर्णाङ्क       dilation_x = p[4];
    अचर पूर्णाङ्क       dilation_y = p[5];

    अचर पूर्णाङ्क in_w     = आगत->ne[0];
    अचर पूर्णाङ्क in_h     = आगत->ne[1];
    अचर पूर्णाङ्क kernel_w = kernel->ne[0];
    अचर पूर्णाङ्क kernel_h = kernel->ne[1];
    अचर पूर्णाङ्क out_w    = dst->ne[0];
    अचर पूर्णाङ्क out_h    = dst->ne[1];
    अचर पूर्णाङ्क channels = dst->ne[2];
    अचर पूर्णाङ्क batches  = dst->ne[3];

    अचर conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    अचर queue_ptr stream = ctx.stream();

    यदि (kernel->type == GGML_TYPE_F16) {
        अचर sycl::half * w_d = (अचर sycl::half *) kernel->data;
        यदि (ggml_is_contiguous(आगत)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा यदि (ggml_is_contiguous_channels(आगत)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } अन्यथा {
        अचर भिन्न * w_d = (अचर भिन्न *) kernel->data;
        यदि (ggml_is_contiguous(आगत)) {
            conv2d_dw_sycl<भिन्न, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा यदि (ggml_is_contiguous_channels(आगत)) {
            conv2d_dw_sycl<भिन्न, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
