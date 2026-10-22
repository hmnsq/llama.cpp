#شامل "conv2d-dw.hpp"

ساخت conv2d_dw_params {
    صحیح اندر_w, اندر_h;
    صحیح برآمد_w, برآمد_h;
    صحیح kernel_w, kernel_h;
    صحیح stride_x, stride_y;
    صحیح padding_x, padding_y;
    صحیح dilation_x, dilation_y;
    صحیح channels, batches;
};

ساخت conv2d_dw_kernel_bounds {
    صحیح y_min, y_max;
    صحیح x_min, x_max;
};

ساکن اندرونی conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(صحیح برآمد_x, صحیح برآمد_y,
                                                                  ثابت conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - برآمد_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.اندر_h + p.padding_y - برآمد_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - برآمد_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.اندر_w + p.padding_x - برآمد_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    واپس bounds;
}

ساکن اندرونی صحیح dw_calculate_داخلہ_coord(صحیح برآمد_coord, صحیح kern_coord, صحیح stride, صحیح dilation, صحیح padding) {
    واپس برآمد_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
ساخت dw_whcn_layout {
    ساکن صحیح داخلہ_اشاریہ(صحیح n, صحیح c, صحیح y, صحیح x, ثابت conv2d_dw_params & p) {
        واپس n * (p.channels * p.اندر_w * p.اندر_h) + c * p.اندر_w * p.اندر_h + y * p.اندر_w + x;
    }
    ساکن صحیح kernel_اشاریہ(صحیح c, صحیح ky, صحیح kx, ثابت conv2d_dw_params & p) {
        واپس c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    ساکن صحیح output_اشاریہ(صحیح n, صحیح c, صحیح y, صحیح x, ثابت conv2d_dw_params & p) {
        واپس n * (p.channels * p.برآمد_w * p.برآمد_h) + c * p.برآمد_w * p.برآمد_h + y * p.برآمد_w + x;
    }
    ساکن خالی unpack_indices(صحیح global_idx, ثابت conv2d_dw_params & p,
                               صحیح & n, صحیح & c, صحیح & برآمد_y, صحیح & برآمد_x) {
        برآمد_x  = global_idx % p.برآمد_w;
        برآمد_y  = (global_idx / p.برآمد_w) % p.برآمد_h;
        c      = (global_idx / (p.برآمد_w * p.برآمد_h)) % p.channels;
        n      = global_idx / (p.برآمد_w * p.برآمد_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
ساخت dw_cwhn_layout {
    ساکن صحیح داخلہ_اشاریہ(صحیح n, صحیح c, صحیح y, صحیح x, ثابت conv2d_dw_params & p) {
        واپس n * (p.channels * p.اندر_w * p.اندر_h) + (y * p.اندر_w + x) * p.channels + c;
    }
    ساکن صحیح kernel_اشاریہ(صحیح c, صحیح ky, صحیح kx, ثابت conv2d_dw_params & p) {
        واپس (ky * p.kernel_w + kx) * p.channels + c;
    }
    ساکن صحیح output_اشاریہ(صحیح n, صحیح c, صحیح y, صحیح x, ثابت conv2d_dw_params & p) {
        واپس n * (p.channels * p.برآمد_w * p.برآمد_h) + y * (p.برآمد_w * p.channels) + x * p.channels + c;
    }
    ساکن خالی unpack_indices(صحیح global_idx, ثابت conv2d_dw_params & p,
                               صحیح & n, صحیح & c, صحیح & برآمد_y, صحیح & برآمد_x) {
        c      = global_idx % p.channels;
        برآمد_x  = (global_idx / p.channels) % p.برآمد_w;
        برآمد_y  = (global_idx / (p.channels * p.برآمد_w)) % p.برآمد_h;
        n      = global_idx / (p.channels * p.برآمد_w * p.برآمد_h);
    }
};

سانچہ <قسم_نام KernelT, قسم_نام Layout>
ساکن خالی conv2d_dw_kernel(ثابت اعشاری * داخلہ, ثابت KernelT * kernel, اعشاری * output,
                             ثابت conv2d_dw_params p, ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت صحیح global_idx     = شے_ct1.get_local_id(2) +
                               شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2);
    ثابت صحیح کل_elements = p.batches * p.channels * p.برآمد_h * p.برآمد_w;

    اگر (global_idx >= کل_elements) {
        واپس;
    }

    صحیح n, c, برآمد_y, برآمد_x;
    Layout::unpack_indices(global_idx, p, n, c, برآمد_y, برآمد_x);

    اعشاری acc = 0.0f;
    ثابت conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(برآمد_x, برآمد_y, p);

    برائے (صحیح ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ثابت صحیح اندر_y = dw_calculate_داخلہ_coord(برآمد_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        برائے (صحیح kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ثابت صحیح اندر_x = dw_calculate_داخلہ_coord(برآمد_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += داخلہ[Layout::داخلہ_اشاریہ(n, c, اندر_y, اندر_x, p)] *
                   static_cast<اعشاری>(kernel[Layout::kernel_اشاریہ(c, ky, kx, p)]);
        }
    }

    output[Layout::output_اشاریہ(n, c, برآمد_y, برآمد_x, p)] = acc;
}

سانچہ <قسم_نام KernelT, قسم_نام Layout>
ساکن خالی conv2d_dw_sycl(ثابت اعشاری * x_d, ثابت KernelT * w_d, اعشاری * y_d,
                            ثابت conv2d_dw_params p, ثابت قطار_ptr & stream) {
    ثابت صحیح کل      = p.batches * p.channels * p.برآمد_h * p.برآمد_w;
    ثابت صحیح عدد_بلاکیں = (کل + SYCL_CONV2_D_DW_بلاک_سائز - 1) / SYCL_CONV2_D_DW_بلاک_سائز;
    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_CONV2_D_DW_بلاک_سائز);
    ثابت sycl::range<3> بلاک_nums(1, 1, عدد_بلاکیں);
    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, شے_ct1);
        });
}

خالی ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * داخلہ  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                داخلہ->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ثابت اعشاری * x_d = (ثابت اعشاری *) داخلہ->ڈیٹا;
    اعشاری *       y_d = (اعشاری *) dst->ڈیٹا;

    ثابت int32_t * p          = (ثابت int32_t *) dst->op_params;
    ثابت صحیح       stride_x   = p[0];
    ثابت صحیح       stride_y   = p[1];
    ثابت صحیح       padding_x  = p[2];
    ثابت صحیح       padding_y  = p[3];
    ثابت صحیح       dilation_x = p[4];
    ثابت صحیح       dilation_y = p[5];

    ثابت صحیح اندر_w     = داخلہ->ne[0];
    ثابت صحیح اندر_h     = داخلہ->ne[1];
    ثابت صحیح kernel_w = kernel->ne[0];
    ثابت صحیح kernel_h = kernel->ne[1];
    ثابت صحیح برآمد_w    = dst->ne[0];
    ثابت صحیح برآمد_h    = dst->ne[1];
    ثابت صحیح channels = dst->ne[2];
    ثابت صحیح batches  = dst->ne[3];

    ثابت conv2d_dw_params params = { اندر_w, اندر_h, برآمد_w, برآمد_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ثابت قطار_ptr stream = ctx.stream();

    اگر (kernel->type == GGML_TYPE_F16) {
        ثابت sycl::half * w_d = (ثابت sycl::half *) kernel->ڈیٹا;
        اگر (ggml_is_contiguous(داخلہ)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } ورنہ اگر (ggml_is_contiguous_channels(داخلہ)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } ورنہ {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } ورنہ {
        ثابت اعشاری * w_d = (ثابت اعشاری *) kernel->ڈیٹا;
        اگر (ggml_is_contiguous(داخلہ)) {
            conv2d_dw_sycl<اعشاری, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } ورنہ اگر (ggml_is_contiguous_channels(داخلہ)) {
            conv2d_dw_sycl<اعشاری, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } ورنہ {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
