#ଅନ୍ତର୍ଭାଵ "conv2d-dw.hpp"

ସଂରଚନା conv2d_dw_params {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା in_w, in_h;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_w, out_h;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_w, kernel_h;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride_x, stride_y;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା padding_x, padding_y;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା dilation_x, dilation_y;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା channels, batches;
};

ସଂରଚନା conv2d_dw_kernel_bounds {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା y_min, y_max;
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା x_min, x_max;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_x, ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_y,
                                                                  ଅଚଳ conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    ଫେରନ୍ତୁ bounds;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା dw_calculate_ନିବେଶ_coord(ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_coord, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kern_coord, ପୂର୍ଣ୍ଣସଂଖ୍ୟା stride, ପୂର୍ଣ୍ଣସଂଖ୍ୟା dilation, ପୂର୍ଣ୍ଣସଂଖ୍ୟା padding) {
    ଫେରନ୍ତୁ out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
ସଂରଚନା dw_whcn_layout {
    ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_index(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା c, ପୂର୍ଣ୍ଣସଂଖ୍ୟା y, ପୂର୍ଣ୍ଣସଂଖ୍ୟା x, ଅଚଳ conv2d_dw_params & p) {
        ଫେରନ୍ତୁ n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_index(ପୂର୍ଣ୍ଣସଂଖ୍ୟା c, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ky, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kx, ଅଚଳ conv2d_dw_params & p) {
        ଫେରନ୍ତୁ c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା output_index(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା c, ପୂର୍ଣ୍ଣସଂଖ୍ୟା y, ପୂର୍ଣ୍ଣସଂଖ୍ୟା x, ଅଚଳ conv2d_dw_params & p) {
        ଫେରନ୍ତୁ n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    ସ୍ଥିର ଶୂନ୍ୟ unpack_indices(ପୂର୍ଣ୍ଣସଂଖ୍ୟା global_idx, ଅଚଳ conv2d_dw_params & p,
                               ପୂର୍ଣ୍ଣସଂଖ୍ୟା & n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା & c, ପୂର୍ଣ୍ଣସଂଖ୍ୟା & out_y, ପୂର୍ଣ୍ଣସଂଖ୍ୟା & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
ସଂରଚନା dw_cwhn_layout {
    ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା ନିବେଶ_index(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା c, ପୂର୍ଣ୍ଣସଂଖ୍ୟା y, ପୂର୍ଣ୍ଣସଂଖ୍ୟା x, ଅଚଳ conv2d_dw_params & p) {
        ଫେରନ୍ତୁ n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_index(ପୂର୍ଣ୍ଣସଂଖ୍ୟା c, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ky, ପୂର୍ଣ୍ଣସଂଖ୍ୟା kx, ଅଚଳ conv2d_dw_params & p) {
        ଫେରନ୍ତୁ (ky * p.kernel_w + kx) * p.channels + c;
    }
    ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା output_index(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା c, ପୂର୍ଣ୍ଣସଂଖ୍ୟା y, ପୂର୍ଣ୍ଣସଂଖ୍ୟା x, ଅଚଳ conv2d_dw_params & p) {
        ଫେରନ୍ତୁ n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    ସ୍ଥିର ଶୂନ୍ୟ unpack_indices(ପୂର୍ଣ୍ଣସଂଖ୍ୟା global_idx, ଅଚଳ conv2d_dw_params & p,
                               ପୂର୍ଣ୍ଣସଂଖ୍ୟା & n, ପୂର୍ଣ୍ଣସଂଖ୍ୟା & c, ପୂର୍ଣ୍ଣସଂଖ୍ୟା & out_y, ପୂର୍ଣ୍ଣସଂଖ୍ୟା & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

template <typename KernelT, typename Layout>
ସ୍ଥିର ଶୂନ୍ୟ conv2d_dw_kernel(ଅଚଳ ଦଶମିକ * ନିବେଶ, ଅଚଳ KernelT * kernel, ଦଶମିକ * output,
                             ଅଚଳ conv2d_dw_params p, ଅଚଳ sycl::nd_item<3> & item_ct1) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା total_elements = p.batches * p.channels * p.out_h * p.out_w;

    ଯଦି (global_idx >= total_elements) {
        ଫେରନ୍ତୁ;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    ଦଶମିକ acc = 0.0f;
    ଅଚଳ conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା in_y = dw_calculate_ନିବେଶ_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା in_x = dw_calculate_ନିବେଶ_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += ନିବେଶ[Layout::ନିବେଶ_index(n, c, in_y, in_x, p)] *
                   static_cast<ଦଶମିକ>(kernel[Layout::kernel_index(c, ky, kx, p)]);
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

template <typename KernelT, typename Layout>
ସ୍ଥିର ଶୂନ୍ୟ conv2d_dw_sycl(ଅଚଳ ଦଶମିକ * x_d, ଅଚଳ KernelT * w_d, ଦଶମିକ * y_d,
                            ଅଚଳ conv2d_dw_params p, ଅଚଳ queue_ptr & stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା total      = p.batches * p.channels * p.out_h * p.out_w;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
    ଅଚଳ sycl::range<3> block_dims(1, 1, SYCL_CONV2D_DW_BLOCK_SIZE);
    ଅଚଳ sycl::range<3> block_nums(1, 1, ଅଙ୍କ_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

ଶୂନ୍ୟ ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ଅଚଳ ggml_tensor * kernel = dst->src[0];
    ଅଚଳ ggml_tensor * ନିବେଶ  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                ନିବେଶ->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ଅଚଳ ଦଶମିକ * x_d = (ଅଚଳ ଦଶମିକ *) ନିବେଶ->data;
    ଦଶମିକ *       y_d = (ଦଶମିକ *) dst->data;

    ଅଚଳ int32_t * p          = (ଅଚଳ int32_t *) dst->op_params;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       stride_x   = p[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       stride_y   = p[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       padding_x  = p[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       padding_y  = p[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       dilation_x = p[4];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       dilation_y = p[5];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା in_w     = ନିବେଶ->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା in_h     = ନିବେଶ->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_w = kernel->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା kernel_h = kernel->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_w    = dst->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା out_h    = dst->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା channels = dst->ne[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା batches  = dst->ne[3];

    ଅଚଳ conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ଅଚଳ queue_ptr stream = ctx.stream();

    ଯଦି (kernel->type == GGML_TYPE_F16) {
        ଅଚଳ sycl::half * w_d = (ଅଚଳ sycl::half *) kernel->data;
        ଯଦି (ggml_is_contiguous(ନିବେଶ)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } ଅନ୍ୟଥା ଯଦି (ggml_is_contiguous_channels(ନିବେଶ)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } ଅନ୍ୟଥା {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } ଅନ୍ୟଥା {
        ଅଚଳ ଦଶମିକ * w_d = (ଅଚଳ ଦଶମିକ *) kernel->data;
        ଯଦି (ggml_is_contiguous(ନିବେଶ)) {
            conv2d_dw_sycl<ଦଶମିକ, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } ଅନ୍ୟଥା ଯଦି (ggml_is_contiguous_channels(ନିବେଶ)) {
            conv2d_dw_sycl<ଦଶମିକ, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } ଅନ୍ୟଥା {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
