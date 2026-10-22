#அந்தர்பாவ "conv2d-dw.hpp"

கட்டமைப்பு conv2d_dw_params {
    முழு in_w, in_h;
    முழு out_w, out_h;
    முழு kernel_w, kernel_h;
    முழு stride_x, stride_y;
    முழு padding_x, padding_y;
    முழு dilation_x, dilation_y;
    முழு channels, batches;
};

கட்டமைப்பு conv2d_dw_kernel_bounds {
    முழு y_min, y_max;
    முழு x_min, x_max;
};

நிலையான உள்வரி conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(முழு out_x, முழு out_y,
                                                                  மாறிலி conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    திரும்பு bounds;
}

நிலையான உள்வரி முழு dw_calculate_உள்ளீடு_coord(முழு out_coord, முழு kern_coord, முழு stride, முழு dilation, முழு padding) {
    திரும்பு out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
கட்டமைப்பு dw_whcn_layout {
    நிலையான முழு உள்ளீடு_index(முழு n, முழு c, முழு y, முழு x, மாறிலி conv2d_dw_params & p) {
        திரும்பு n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    நிலையான முழு kernel_index(முழு c, முழு ky, முழு kx, மாறிலி conv2d_dw_params & p) {
        திரும்பு c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    நிலையான முழு output_index(முழு n, முழு c, முழு y, முழு x, மாறிலி conv2d_dw_params & p) {
        திரும்பு n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    நிலையான வெறுமை unpack_indices(முழு global_idx, மாறிலி conv2d_dw_params & p,
                               முழு & n, முழு & c, முழு & out_y, முழு & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
கட்டமைப்பு dw_cwhn_layout {
    நிலையான முழு உள்ளீடு_index(முழு n, முழு c, முழு y, முழு x, மாறிலி conv2d_dw_params & p) {
        திரும்பு n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    நிலையான முழு kernel_index(முழு c, முழு ky, முழு kx, மாறிலி conv2d_dw_params & p) {
        திரும்பு (ky * p.kernel_w + kx) * p.channels + c;
    }
    நிலையான முழு output_index(முழு n, முழு c, முழு y, முழு x, மாறிலி conv2d_dw_params & p) {
        திரும்பு n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    நிலையான வெறுமை unpack_indices(முழு global_idx, மாறிலி conv2d_dw_params & p,
                               முழு & n, முழு & c, முழு & out_y, முழு & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

template <typename KernelT, typename Layout>
நிலையான வெறுமை conv2d_dw_kernel(மாறிலி தசம * உள்ளீடு, மாறிலி KernelT * kernel, தசம * output,
                             மாறிலி conv2d_dw_params p, மாறிலி sycl::nd_item<3> & item_ct1) {
    மாறிலி முழு global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    மாறிலி முழு total_elements = p.batches * p.channels * p.out_h * p.out_w;

    எனில் (global_idx >= total_elements) {
        திரும்பு;
    }

    முழு n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    தசம acc = 0.0f;
    மாறிலி conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    க்கு (முழு ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        மாறிலி முழு in_y = dw_calculate_உள்ளீடு_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        க்கு (முழு kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            மாறிலி முழு in_x = dw_calculate_உள்ளீடு_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += உள்ளீடு[Layout::உள்ளீடு_index(n, c, in_y, in_x, p)] *
                   static_cast<தசம>(kernel[Layout::kernel_index(c, ky, kx, p)]);
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

template <typename KernelT, typename Layout>
நிலையான வெறுமை conv2d_dw_sycl(மாறிலி தசம * x_d, மாறிலி KernelT * w_d, தசம * y_d,
                            மாறிலி conv2d_dw_params p, மாறிலி queue_ptr & stream) {
    மாறிலி முழு total      = p.batches * p.channels * p.out_h * p.out_w;
    மாறிலி முழு இலக்கம்_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
    மாறிலி sycl::range<3> block_dims(1, 1, SYCL_CONV2D_DW_BLOCK_SIZE);
    மாறிலி sycl::range<3> block_nums(1, 1, இலக்கம்_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

வெறுமை ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    மாறிலி ggml_tensor * kernel = dst->src[0];
    மாறிலி ggml_tensor * உள்ளீடு  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                உள்ளீடு->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    மாறிலி தசம * x_d = (மாறிலி தசம *) உள்ளீடு->data;
    தசம *       y_d = (தசம *) dst->data;

    மாறிலி int32_t * p          = (மாறிலி int32_t *) dst->op_params;
    மாறிலி முழு       stride_x   = p[0];
    மாறிலி முழு       stride_y   = p[1];
    மாறிலி முழு       padding_x  = p[2];
    மாறிலி முழு       padding_y  = p[3];
    மாறிலி முழு       dilation_x = p[4];
    மாறிலி முழு       dilation_y = p[5];

    மாறிலி முழு in_w     = உள்ளீடு->ne[0];
    மாறிலி முழு in_h     = உள்ளீடு->ne[1];
    மாறிலி முழு kernel_w = kernel->ne[0];
    மாறிலி முழு kernel_h = kernel->ne[1];
    மாறிலி முழு out_w    = dst->ne[0];
    மாறிலி முழு out_h    = dst->ne[1];
    மாறிலி முழு channels = dst->ne[2];
    மாறிலி முழு batches  = dst->ne[3];

    மாறிலி conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    மாறிலி queue_ptr stream = ctx.stream();

    எனில் (kernel->type == GGML_TYPE_F16) {
        மாறிலி sycl::half * w_d = (மாறிலி sycl::half *) kernel->data;
        எனில் (ggml_is_contiguous(உள்ளீடு)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } இல்லையெனில் எனில் (ggml_is_contiguous_channels(உள்ளீடு)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } இல்லையெனில் {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } இல்லையெனில் {
        மாறிலி தசம * w_d = (மாறிலி தசம *) kernel->data;
        எனில் (ggml_is_contiguous(உள்ளீடு)) {
            conv2d_dw_sycl<தசம, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } இல்லையெனில் எனில் (ggml_is_contiguous_channels(உள்ளீடு)) {
            conv2d_dw_sycl<தசம, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } இல்லையெனில் {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
