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
                                                                  ध्रुव conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    प्रत्यागम bounds;
}

स्थिर अन्तःस्थ पूर्णाङ्क dw_calculate_निवेश_coord(पूर्णाङ्क out_coord, पूर्णाङ्क kern_coord, पूर्णाङ्क stride, पूर्णाङ्क dilation, पूर्णाङ्क padding) {
    प्रत्यागम out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
संरचना dw_whcn_layout {
    स्थिर पूर्णाङ्क निवेश_index(पूर्णाङ्क n, पूर्णाङ्क c, पूर्णाङ्क y, पूर्णाङ्क x, ध्रुव conv2d_dw_params & p) {
        प्रत्यागम n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    स्थिर पूर्णाङ्क kernel_index(पूर्णाङ्क c, पूर्णाङ्क ky, पूर्णाङ्क kx, ध्रुव conv2d_dw_params & p) {
        प्रत्यागम c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    स्थिर पूर्णाङ्क output_index(पूर्णाङ्क n, पूर्णाङ्क c, पूर्णाङ्क y, पूर्णाङ्क x, ध्रुव conv2d_dw_params & p) {
        प्रत्यागम n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    स्थिर शून्य unpack_indices(पूर्णाङ्क global_idx, ध्रुव conv2d_dw_params & p,
                               पूर्णाङ्क & n, पूर्णाङ्क & c, पूर्णाङ्क & out_y, पूर्णाङ्क & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
संरचना dw_cwhn_layout {
    स्थिर पूर्णाङ्क निवेश_index(पूर्णाङ्क n, पूर्णाङ्क c, पूर्णाङ्क y, पूर्णाङ्क x, ध्रुव conv2d_dw_params & p) {
        प्रत्यागम n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    स्थिर पूर्णाङ्क kernel_index(पूर्णाङ्क c, पूर्णाङ्क ky, पूर्णाङ्क kx, ध्रुव conv2d_dw_params & p) {
        प्रत्यागम (ky * p.kernel_w + kx) * p.channels + c;
    }
    स्थिर पूर्णाङ्क output_index(पूर्णाङ्क n, पूर्णाङ्क c, पूर्णाङ्क y, पूर्णाङ्क x, ध्रुव conv2d_dw_params & p) {
        प्रत्यागम n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    स्थिर शून्य unpack_indices(पूर्णाङ्क global_idx, ध्रुव conv2d_dw_params & p,
                               पूर्णाङ्क & n, पूर्णाङ्क & c, पूर्णाङ्क & out_y, पूर्णाङ्क & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

template <typename KernelT, typename Layout>
स्थिर शून्य conv2d_dw_kernel(ध्रुव भिन्न * निवेश, ध्रुव KernelT * kernel, भिन्न * output,
                             ध्रुव conv2d_dw_params p, ध्रुव sycl::nd_item<3> & item_ct1) {
    ध्रुव पूर्णाङ्क global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ध्रुव पूर्णाङ्क total_elements = p.batches * p.channels * p.out_h * p.out_w;

    यदि (global_idx >= total_elements) {
        प्रत्यागम;
    }

    पूर्णाङ्क n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    भिन्न acc = 0.0f;
    ध्रुव conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    प्रति (पूर्णाङ्क ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ध्रुव पूर्णाङ्क in_y = dw_calculate_निवेश_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        प्रति (पूर्णाङ्क kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ध्रुव पूर्णाङ्क in_x = dw_calculate_निवेश_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += निवेश[Layout::निवेश_index(n, c, in_y, in_x, p)] *
                   static_cast<भिन्न>(kernel[Layout::kernel_index(c, ky, kx, p)]);
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

template <typename KernelT, typename Layout>
स्थिर शून्य conv2d_dw_sycl(ध्रुव भिन्न * x_d, ध्रुव KernelT * w_d, भिन्न * y_d,
                            ध्रुव conv2d_dw_params p, ध्रुव queue_ptr & stream) {
    ध्रुव पूर्णाङ्क total      = p.batches * p.channels * p.out_h * p.out_w;
    ध्रुव पूर्णाङ्क अंक_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
    ध्रुव sycl::range<3> block_dims(1, 1, SYCL_CONV2D_DW_BLOCK_SIZE);
    ध्रुव sycl::range<3> block_nums(1, 1, अंक_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

शून्य ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ध्रुव ggml_tensor * kernel = dst->src[0];
    ध्रुव ggml_tensor * निवेश  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                निवेश->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ध्रुव भिन्न * x_d = (ध्रुव भिन्न *) निवेश->data;
    भिन्न *       y_d = (भिन्न *) dst->data;

    ध्रुव int32_t * p          = (ध्रुव int32_t *) dst->op_params;
    ध्रुव पूर्णाङ्क       stride_x   = p[0];
    ध्रुव पूर्णाङ्क       stride_y   = p[1];
    ध्रुव पूर्णाङ्क       padding_x  = p[2];
    ध्रुव पूर्णाङ्क       padding_y  = p[3];
    ध्रुव पूर्णाङ्क       dilation_x = p[4];
    ध्रुव पूर्णाङ्क       dilation_y = p[5];

    ध्रुव पूर्णाङ्क in_w     = निवेश->ne[0];
    ध्रुव पूर्णाङ्क in_h     = निवेश->ne[1];
    ध्रुव पूर्णाङ्क kernel_w = kernel->ne[0];
    ध्रुव पूर्णाङ्क kernel_h = kernel->ne[1];
    ध्रुव पूर्णाङ्क out_w    = dst->ne[0];
    ध्रुव पूर्णाङ्क out_h    = dst->ne[1];
    ध्रुव पूर्णाङ्क channels = dst->ne[2];
    ध्रुव पूर्णाङ्क batches  = dst->ne[3];

    ध्रुव conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ध्रुव queue_ptr stream = ctx.stream();

    यदि (kernel->type == GGML_TYPE_F16) {
        ध्रुव sycl::half * w_d = (ध्रुव sycl::half *) kernel->data;
        यदि (ggml_is_contiguous(निवेश)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा यदि (ggml_is_contiguous_channels(निवेश)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } अन्यथा {
        ध्रुव भिन्न * w_d = (ध्रुव भिन्न *) kernel->data;
        यदि (ggml_is_contiguous(निवेश)) {
            conv2d_dw_sycl<भिन्न, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा यदि (ggml_is_contiguous_channels(निवेश)) {
            conv2d_dw_sycl<भिन्न, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
