#شامل "conv2d-dw.hpp"

ساختار conv2d_dw_params {
    صحیح in_w, in_h;
    صحیح out_w, out_h;
    صحیح kernel_w, kernel_h;
    صحیح stride_x, stride_y;
    صحیح padding_x, padding_y;
    صحیح dilation_x, dilation_y;
    صحیح channels, batches;
};

ساختار conv2d_dw_kernel_bounds {
    صحیح y_min, y_max;
    صحیح x_min, x_max;
};

ایستا درونخطی conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(صحیح out_x, صحیح out_y,
                                                                  ثابت conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    بازگشت bounds;
}

ایستا درونخطی صحیح dw_calculate_درون_داد_coord(صحیح out_coord, صحیح kern_coord, صحیح stride, صحیح dilation, صحیح padding) {
    بازگشت out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
ساختار dw_whcn_layout {
    ایستا صحیح درون_داد_index(صحیح n, صحیح c, صحیح y, صحیح x, ثابت conv2d_dw_params & p) {
        بازگشت n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    ایستا صحیح kernel_index(صحیح c, صحیح ky, صحیح kx, ثابت conv2d_dw_params & p) {
        بازگشت c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    ایستا صحیح output_index(صحیح n, صحیح c, صحیح y, صحیح x, ثابت conv2d_dw_params & p) {
        بازگشت n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    ایستا تهی unpack_indices(صحیح global_idx, ثابت conv2d_dw_params & p,
                               صحیح & n, صحیح & c, صحیح & out_y, صحیح & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
ساختار dw_cwhn_layout {
    ایستا صحیح درون_داد_index(صحیح n, صحیح c, صحیح y, صحیح x, ثابت conv2d_dw_params & p) {
        بازگشت n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    ایستا صحیح kernel_index(صحیح c, صحیح ky, صحیح kx, ثابت conv2d_dw_params & p) {
        بازگشت (ky * p.kernel_w + kx) * p.channels + c;
    }
    ایستا صحیح output_index(صحیح n, صحیح c, صحیح y, صحیح x, ثابت conv2d_dw_params & p) {
        بازگشت n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    ایستا تهی unpack_indices(صحیح global_idx, ثابت conv2d_dw_params & p,
                               صحیح & n, صحیح & c, صحیح & out_y, صحیح & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

template <typename KernelT, typename Layout>
ایستا تهی conv2d_dw_kernel(ثابت اعشاری * درون_داد, ثابت KernelT * kernel, اعشاری * output,
                             ثابت conv2d_dw_params p, ثابت sycl::nd_item<3> & item_ct1) {
    ثابت صحیح global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ثابت صحیح total_elements = p.batches * p.channels * p.out_h * p.out_w;

    اگر (global_idx >= total_elements) {
        بازگشت;
    }

    صحیح n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    اعشاری acc = 0.0f;
    ثابت conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    برای (صحیح ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ثابت صحیح in_y = dw_calculate_درون_داد_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        برای (صحیح kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ثابت صحیح in_x = dw_calculate_درون_داد_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += درون_داد[Layout::درون_داد_index(n, c, in_y, in_x, p)] *
                   static_cast<اعشاری>(kernel[Layout::kernel_index(c, ky, kx, p)]);
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

template <typename KernelT, typename Layout>
ایستا تهی conv2d_dw_sycl(ثابت اعشاری * x_d, ثابت KernelT * w_d, اعشاری * y_d,
                            ثابت conv2d_dw_params p, ثابت queue_ptr & stream) {
    ثابت صحیح total      = p.batches * p.channels * p.out_h * p.out_w;
    ثابت صحیح رقم_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
    ثابت sycl::range<3> block_dims(1, 1, SYCL_CONV2D_DW_BLOCK_SIZE);
    ثابت sycl::range<3> block_nums(1, 1, رقم_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

تهی ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * درون_داد  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                درون_داد->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ثابت اعشاری * x_d = (ثابت اعشاری *) درون_داد->data;
    اعشاری *       y_d = (اعشاری *) dst->data;

    ثابت int32_t * p          = (ثابت int32_t *) dst->op_params;
    ثابت صحیح       stride_x   = p[0];
    ثابت صحیح       stride_y   = p[1];
    ثابت صحیح       padding_x  = p[2];
    ثابت صحیح       padding_y  = p[3];
    ثابت صحیح       dilation_x = p[4];
    ثابت صحیح       dilation_y = p[5];

    ثابت صحیح in_w     = درون_داد->ne[0];
    ثابت صحیح in_h     = درون_داد->ne[1];
    ثابت صحیح kernel_w = kernel->ne[0];
    ثابت صحیح kernel_h = kernel->ne[1];
    ثابت صحیح out_w    = dst->ne[0];
    ثابت صحیح out_h    = dst->ne[1];
    ثابت صحیح channels = dst->ne[2];
    ثابت صحیح batches  = dst->ne[3];

    ثابت conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ثابت queue_ptr stream = ctx.stream();

    اگر (kernel->type == GGML_TYPE_F16) {
        ثابت sycl::half * w_d = (ثابت sycl::half *) kernel->data;
        اگر (ggml_is_contiguous(درون_داد)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } وگرنه اگر (ggml_is_contiguous_channels(درون_داد)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } وگرنه {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } وگرنه {
        ثابت اعشاری * w_d = (ثابت اعشاری *) kernel->data;
        اگر (ggml_is_contiguous(درون_داد)) {
            conv2d_dw_sycl<اعشاری, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } وگرنه اگر (ggml_is_contiguous_channels(درون_داد)) {
            conv2d_dw_sycl<اعشاری, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } وگرنه {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
