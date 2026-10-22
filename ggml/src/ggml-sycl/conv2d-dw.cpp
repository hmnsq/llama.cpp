#अन्तर्भाव "conv2d-dw.hpp"

संरचना conv2d_dw_params {
    पूर्णाङ्क अन्तर्_w, अन्तर्_h;
    पूर्णाङ्क बहिर्गमनम्_w, बहिर्गमनम्_h;
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

स्थिर अन्तःस्थ conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(पूर्णाङ्क बहिर्गमनम्_x, पूर्णाङ्क बहिर्गमनम्_y,
                                                                  ध्रुव conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - बहिर्गमनम्_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.अन्तर्_h + p.padding_y - बहिर्गमनम्_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - बहिर्गमनम्_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.अन्तर्_w + p.padding_x - बहिर्गमनम्_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    प्रत्यागम bounds;
}

स्थिर अन्तःस्थ पूर्णाङ्क dw_calculate_निवेशः_coord(पूर्णाङ्क बहिर्गमनम्_coord, पूर्णाङ्क kern_coord, पूर्णाङ्क stride, पूर्णाङ्क dilation, पूर्णाङ्क padding) {
    प्रत्यागम बहिर्गमनम्_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
संरचना dw_whcn_layout {
    स्थिर पूर्णाङ्क निवेशः_सूचकांक(पूर्णाङ्क n, पूर्णाङ्क c, पूर्णाङ्क y, पूर्णाङ्क x, ध्रुव conv2d_dw_params & p) {
        प्रत्यागम n * (p.channels * p.अन्तर्_w * p.अन्तर्_h) + c * p.अन्तर्_w * p.अन्तर्_h + y * p.अन्तर्_w + x;
    }
    स्थिर पूर्णाङ्क kernel_सूचकांक(पूर्णाङ्क c, पूर्णाङ्क ky, पूर्णाङ्क kx, ध्रुव conv2d_dw_params & p) {
        प्रत्यागम c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    स्थिर पूर्णाङ्क output_सूचकांक(पूर्णाङ्क n, पूर्णाङ्क c, पूर्णाङ्क y, पूर्णाङ्क x, ध्रुव conv2d_dw_params & p) {
        प्रत्यागम n * (p.channels * p.बहिर्गमनम्_w * p.बहिर्गमनम्_h) + c * p.बहिर्गमनम्_w * p.बहिर्गमनम्_h + y * p.बहिर्गमनम्_w + x;
    }
    स्थिर शून्य unpack_indices(पूर्णाङ्क global_idx, ध्रुव conv2d_dw_params & p,
                               पूर्णाङ्क & n, पूर्णाङ्क & c, पूर्णाङ्क & बहिर्गमनम्_y, पूर्णाङ्क & बहिर्गमनम्_x) {
        बहिर्गमनम्_x  = global_idx % p.बहिर्गमनम्_w;
        बहिर्गमनम्_y  = (global_idx / p.बहिर्गमनम्_w) % p.बहिर्गमनम्_h;
        c      = (global_idx / (p.बहिर्गमनम्_w * p.बहिर्गमनम्_h)) % p.channels;
        n      = global_idx / (p.बहिर्गमनम्_w * p.बहिर्गमनम्_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
संरचना dw_cwhn_layout {
    स्थिर पूर्णाङ्क निवेशः_सूचकांक(पूर्णाङ्क n, पूर्णाङ्क c, पूर्णाङ्क y, पूर्णाङ्क x, ध्रुव conv2d_dw_params & p) {
        प्रत्यागम n * (p.channels * p.अन्तर्_w * p.अन्तर्_h) + (y * p.अन्तर्_w + x) * p.channels + c;
    }
    स्थिर पूर्णाङ्क kernel_सूचकांक(पूर्णाङ्क c, पूर्णाङ्क ky, पूर्णाङ्क kx, ध्रुव conv2d_dw_params & p) {
        प्रत्यागम (ky * p.kernel_w + kx) * p.channels + c;
    }
    स्थिर पूर्णाङ्क output_सूचकांक(पूर्णाङ्क n, पूर्णाङ्क c, पूर्णाङ्क y, पूर्णाङ्क x, ध्रुव conv2d_dw_params & p) {
        प्रत्यागम n * (p.channels * p.बहिर्गमनम्_w * p.बहिर्गमनम्_h) + y * (p.बहिर्गमनम्_w * p.channels) + x * p.channels + c;
    }
    स्थिर शून्य unpack_indices(पूर्णाङ्क global_idx, ध्रुव conv2d_dw_params & p,
                               पूर्णाङ्क & n, पूर्णाङ्क & c, पूर्णाङ्क & बहिर्गमनम्_y, पूर्णाङ्क & बहिर्गमनम्_x) {
        c      = global_idx % p.channels;
        बहिर्गमनम्_x  = (global_idx / p.channels) % p.बहिर्गमनम्_w;
        बहिर्गमनम्_y  = (global_idx / (p.channels * p.बहिर्गमनम्_w)) % p.बहिर्गमनम्_h;
        n      = global_idx / (p.channels * p.बहिर्गमनम्_w * p.बहिर्गमनम्_h);
    }
};

प्रतिरूपम् <प्रकारनाम KernelT, प्रकारनाम Layout>
स्थिर शून्य conv2d_dw_kernel(ध्रुव भिन्न * निवेशः, ध्रुव KernelT * kernel, भिन्न * output,
                             ध्रुव conv2d_dw_params p, ध्रुव sycl::nd_item<3> & item_ct1) {
    ध्रुव पूर्णाङ्क global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_गणः(2) * item_ct1.get_local_range(2);
    ध्रुव पूर्णाङ्क योग_elements = p.batches * p.channels * p.बहिर्गमनम्_h * p.बहिर्गमनम्_w;

    यदि (global_idx >= योग_elements) {
        प्रत्यागम;
    }

    पूर्णाङ्क n, c, बहिर्गमनम्_y, बहिर्गमनम्_x;
    Layout::unpack_indices(global_idx, p, n, c, बहिर्गमनम्_y, बहिर्गमनम्_x);

    भिन्न acc = 0.0f;
    ध्रुव conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(बहिर्गमनम्_x, बहिर्गमनम्_y, p);

    प्रति (पूर्णाङ्क ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ध्रुव पूर्णाङ्क अन्तर्_y = dw_calculate_निवेशः_coord(बहिर्गमनम्_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        प्रति (पूर्णाङ्क kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ध्रुव पूर्णाङ्क अन्तर्_x = dw_calculate_निवेशः_coord(बहिर्गमनम्_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += निवेशः[Layout::निवेशः_सूचकांक(n, c, अन्तर्_y, अन्तर्_x, p)] *
                   static_cast<भिन्न>(kernel[Layout::kernel_सूचकांक(c, ky, kx, p)]);
        }
    }

    output[Layout::output_सूचकांक(n, c, बहिर्गमनम्_y, बहिर्गमनम्_x, p)] = acc;
}

प्रतिरूपम् <प्रकारनाम KernelT, प्रकारनाम Layout>
स्थिर शून्य conv2d_dw_sycl(ध्रुव भिन्न * x_d, ध्रुव KernelT * w_d, भिन्न * y_d,
                            ध्रुव conv2d_dw_params p, ध्रुव queue_ptr & stream) {
    ध्रुव पूर्णाङ्क योग      = p.batches * p.channels * p.बहिर्गमनम्_h * p.बहिर्गमनम्_w;
    ध्रुव पूर्णाङ्क अङ्कः_खण्डानि = (योग + SYCL_CONV2_D_DW_खण्डम्_आकार - 1) / SYCL_CONV2_D_DW_खण्डम्_आकार;
    ध्रुव sycl::range<3> खण्डम्_dims(1, 1, SYCL_CONV2_D_DW_खण्डम्_आकार);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, अङ्कः_खण्डानि);
    stream->parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

शून्य ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ध्रुव ggml_tensor * kernel = dst->src[0];
    ध्रुव ggml_tensor * निवेशः  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                निवेशः->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ध्रुव भिन्न * x_d = (ध्रुव भिन्न *) निवेशः->दत्त;
    भिन्न *       y_d = (भिन्न *) dst->दत्त;

    ध्रुव int32_t * p          = (ध्रुव int32_t *) dst->op_params;
    ध्रुव पूर्णाङ्क       stride_x   = p[0];
    ध्रुव पूर्णाङ्क       stride_y   = p[1];
    ध्रुव पूर्णाङ्क       padding_x  = p[2];
    ध्रुव पूर्णाङ्क       padding_y  = p[3];
    ध्रुव पूर्णाङ्क       dilation_x = p[4];
    ध्रुव पूर्णाङ्क       dilation_y = p[5];

    ध्रुव पूर्णाङ्क अन्तर्_w     = निवेशः->ne[0];
    ध्रुव पूर्णाङ्क अन्तर्_h     = निवेशः->ne[1];
    ध्रुव पूर्णाङ्क kernel_w = kernel->ne[0];
    ध्रुव पूर्णाङ्क kernel_h = kernel->ne[1];
    ध्रुव पूर्णाङ्क बहिर्गमनम्_w    = dst->ne[0];
    ध्रुव पूर्णाङ्क बहिर्गमनम्_h    = dst->ne[1];
    ध्रुव पूर्णाङ्क channels = dst->ne[2];
    ध्रुव पूर्णाङ्क batches  = dst->ne[3];

    ध्रुव conv2d_dw_params params = { अन्तर्_w, अन्तर्_h, बहिर्गमनम्_w, बहिर्गमनम्_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ध्रुव queue_ptr stream = ctx.stream();

    यदि (kernel->type == GGML_TYPE_F16) {
        ध्रुव sycl::half * w_d = (ध्रुव sycl::half *) kernel->दत्त;
        यदि (ggml_is_contiguous(निवेशः)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा यदि (ggml_is_contiguous_channels(निवेशः)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } अन्यथा {
        ध्रुव भिन्न * w_d = (ध्रुव भिन्न *) kernel->दत्त;
        यदि (ggml_is_contiguous(निवेशः)) {
            conv2d_dw_sycl<भिन्न, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा यदि (ggml_is_contiguous_channels(निवेशः)) {
            conv2d_dw_sycl<भिन्न, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } अन्यथा {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
