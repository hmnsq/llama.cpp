#समावेश "conv2d-dw.hpp"

काष्ठा conv2d_dw_params {
    पूर्णांक in_w, in_h;
    पूर्णांक out_w, out_h;
    पूर्णांक kernel_w, kernel_h;
    पूर्णांक stride_x, stride_y;
    पूर्णांक padding_x, padding_y;
    पूर्णांक dilation_x, dilation_y;
    पूर्णांक channels, batches;
};

काष्ठा conv2d_dw_kernel_bounds {
    पूर्णांक y_min, y_max;
    पूर्णांक x_min, x_max;
};

अटल अंतरभूत conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(पूर्णांक out_x, पूर्णांक out_y,
                                                                  स्थिर conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    वापस bounds;
}

अटल अंतरभूत पूर्णांक dw_calculate_निवेश_coord(पूर्णांक out_coord, पूर्णांक kern_coord, पूर्णांक stride, पूर्णांक dilation, पूर्णांक padding) {
    वापस out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
काष्ठा dw_whcn_layout {
    अटल पूर्णांक निवेश_सूचकांक(पूर्णांक n, पूर्णांक c, पूर्णांक y, पूर्णांक x, स्थिर conv2d_dw_params & p) {
        वापस n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    अटल पूर्णांक kernel_सूचकांक(पूर्णांक c, पूर्णांक ky, पूर्णांक kx, स्थिर conv2d_dw_params & p) {
        वापस c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    अटल पूर्णांक output_सूचकांक(पूर्णांक n, पूर्णांक c, पूर्णांक y, पूर्णांक x, स्थिर conv2d_dw_params & p) {
        वापस n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    अटल व्योम unpack_indices(पूर्णांक global_idx, स्थिर conv2d_dw_params & p,
                               पूर्णांक & n, पूर्णांक & c, पूर्णांक & out_y, पूर्णांक & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
काष्ठा dw_cwhn_layout {
    अटल पूर्णांक निवेश_सूचकांक(पूर्णांक n, पूर्णांक c, पूर्णांक y, पूर्णांक x, स्थिर conv2d_dw_params & p) {
        वापस n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    अटल पूर्णांक kernel_सूचकांक(पूर्णांक c, पूर्णांक ky, पूर्णांक kx, स्थिर conv2d_dw_params & p) {
        वापस (ky * p.kernel_w + kx) * p.channels + c;
    }
    अटल पूर्णांक output_सूचकांक(पूर्णांक n, पूर्णांक c, पूर्णांक y, पूर्णांक x, स्थिर conv2d_dw_params & p) {
        वापस n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    अटल व्योम unpack_indices(पूर्णांक global_idx, स्थिर conv2d_dw_params & p,
                               पूर्णांक & n, पूर्णांक & c, पूर्णांक & out_y, पूर्णांक & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

ढाँचा <typename KernelT, typename Layout>
अटल व्योम conv2d_dw_kernel(स्थिर भग्न * निवेश, स्थिर KernelT * kernel, भग्न * output,
                             स्थिर conv2d_dw_params p, स्थिर sycl::nd_item<3> & item_ct1) {
    स्थिर पूर्णांक global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    स्थिर पूर्णांक योग_elements = p.batches * p.channels * p.out_h * p.out_w;

    अगर (global_idx >= योग_elements) {
        वापस;
    }

    पूर्णांक n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    भग्न acc = 0.0f;
    स्थिर conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    क्रम (पूर्णांक ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        स्थिर पूर्णांक in_y = dw_calculate_निवेश_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        क्रम (पूर्णांक kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            स्थिर पूर्णांक in_x = dw_calculate_निवेश_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += निवेश[Layout::निवेश_सूचकांक(n, c, in_y, in_x, p)] *
                   static_cast<भग्न>(kernel[Layout::kernel_सूचकांक(c, ky, kx, p)]);
        }
    }

    output[Layout::output_सूचकांक(n, c, out_y, out_x, p)] = acc;
}

ढाँचा <typename KernelT, typename Layout>
अटल व्योम conv2d_dw_sycl(स्थिर भग्न * x_d, स्थिर KernelT * w_d, भग्न * y_d,
                            स्थिर conv2d_dw_params p, स्थिर queue_ptr & stream) {
    स्थिर पूर्णांक योग      = p.batches * p.channels * p.out_h * p.out_w;
    स्थिर पूर्णांक अंक_blocks = (योग + SYCL_CONV2_D_DW_BLOCK_आकार - 1) / SYCL_CONV2_D_DW_BLOCK_आकार;
    स्थिर sycl::range<3> block_dims(1, 1, SYCL_CONV2_D_DW_BLOCK_आकार);
    स्थिर sycl::range<3> block_nums(1, 1, अंक_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

व्योम ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    स्थिर ggml_tensor * kernel = dst->src[0];
    स्थिर ggml_tensor * निवेश  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                निवेश->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    स्थिर भग्न * x_d = (स्थिर भग्न *) निवेश->तथ्य;
    भग्न *       y_d = (भग्न *) dst->तथ्य;

    स्थिर int32_t * p          = (स्थिर int32_t *) dst->op_params;
    स्थिर पूर्णांक       stride_x   = p[0];
    स्थिर पूर्णांक       stride_y   = p[1];
    स्थिर पूर्णांक       padding_x  = p[2];
    स्थिर पूर्णांक       padding_y  = p[3];
    स्थिर पूर्णांक       dilation_x = p[4];
    स्थिर पूर्णांक       dilation_y = p[5];

    स्थिर पूर्णांक in_w     = निवेश->ne[0];
    स्थिर पूर्णांक in_h     = निवेश->ne[1];
    स्थिर पूर्णांक kernel_w = kernel->ne[0];
    स्थिर पूर्णांक kernel_h = kernel->ne[1];
    स्थिर पूर्णांक out_w    = dst->ne[0];
    स्थिर पूर्णांक out_h    = dst->ne[1];
    स्थिर पूर्णांक channels = dst->ne[2];
    स्थिर पूर्णांक batches  = dst->ne[3];

    स्थिर conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    स्थिर queue_ptr stream = ctx.stream();

    अगर (kernel->type == GGML_TYPE_F16) {
        स्थिर sycl::half * w_d = (स्थिर sycl::half *) kernel->तथ्य;
        अगर (ggml_is_contiguous(निवेश)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा अगर (ggml_is_contiguous_channels(निवेश)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } अन्यथा {
        स्थिर भग्न * w_d = (स्थिर भग्न *) kernel->तथ्य;
        अगर (ggml_is_contiguous(निवेश)) {
            conv2d_dw_sycl<भग्न, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा अगर (ggml_is_contiguous_channels(निवेश)) {
            conv2d_dw_sycl<भग्न, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
