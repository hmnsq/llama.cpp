#ضمن "conv2d-dw.hpp"

بنية conv2d_dw_params {
    صحيح وچ_w, وچ_h;
    صحيح باہرلا_w, باہرلا_h;
    صحيح kernel_w, kernel_h;
    صحيح stride_x, stride_y;
    صحيح padding_x, padding_y;
    صحيح dilation_x, dilation_y;
    صحيح channels, batches;
};

بنية conv2d_dw_kernel_bounds {
    صحيح y_min, y_max;
    صحيح x_min, x_max;
};

ساكن مضمن conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(صحيح باہرلا_x, صحيح باہرلا_y,
                                                                  ثابت conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - باہرلا_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.وچ_h + p.padding_y - باہرلا_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - باہرلا_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.وچ_w + p.padding_x - باہرلا_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    أرجع bounds;
}

ساكن مضمن صحيح dw_calculate_داخلہ_coord(صحيح باہرلا_coord, صحيح kern_coord, صحيح stride, صحيح dilation, صحيح padding) {
    أرجع باہرلا_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
بنية dw_whcn_layout {
    ساكن صحيح داخلہ_اشاریہ(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.وچ_w * p.وچ_h) + c * p.وچ_w * p.وچ_h + y * p.وچ_w + x;
    }
    ساكن صحيح kernel_اشاریہ(صحيح c, صحيح ky, صحيح kx, ثابت conv2d_dw_params & p) {
        أرجع c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    ساكن صحيح output_اشاریہ(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.باہرلا_w * p.باہرلا_h) + c * p.باہرلا_w * p.باہرلا_h + y * p.باہرلا_w + x;
    }
    ساكن فراغ unpack_indices(صحيح global_idx, ثابت conv2d_dw_params & p,
                               صحيح & n, صحيح & c, صحيح & باہرلا_y, صحيح & باہرلا_x) {
        باہرلا_x  = global_idx % p.باہرلا_w;
        باہرلا_y  = (global_idx / p.باہرلا_w) % p.باہرلا_h;
        c      = (global_idx / (p.باہرلا_w * p.باہرلا_h)) % p.channels;
        n      = global_idx / (p.باہرلا_w * p.باہرلا_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
بنية dw_cwhn_layout {
    ساكن صحيح داخلہ_اشاریہ(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.وچ_w * p.وچ_h) + (y * p.وچ_w + x) * p.channels + c;
    }
    ساكن صحيح kernel_اشاریہ(صحيح c, صحيح ky, صحيح kx, ثابت conv2d_dw_params & p) {
        أرجع (ky * p.kernel_w + kx) * p.channels + c;
    }
    ساكن صحيح output_اشاریہ(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.باہرلا_w * p.باہرلا_h) + y * (p.باہرلا_w * p.channels) + x * p.channels + c;
    }
    ساكن فراغ unpack_indices(صحيح global_idx, ثابت conv2d_dw_params & p,
                               صحيح & n, صحيح & c, صحيح & باہرلا_y, صحيح & باہرلا_x) {
        c      = global_idx % p.channels;
        باہرلا_x  = (global_idx / p.channels) % p.باہرلا_w;
        باہرلا_y  = (global_idx / (p.channels * p.باہرلا_w)) % p.باہرلا_h;
        n      = global_idx / (p.channels * p.باہرلا_w * p.باہرلا_h);
    }
};

سانچا <قسم_ناں KernelT, قسم_ناں Layout>
ساكن فراغ conv2d_dw_kernel(ثابت عائم * داخلہ, ثابت KernelT * kernel, عائم * output,
                             ثابت conv2d_dw_params p, ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت صحيح global_idx     = شے_ct1.get_local_id(2) +
                               شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2);
    ثابت صحيح کل_elements = p.batches * p.channels * p.باہرلا_h * p.باہرلا_w;

    إذا (global_idx >= کل_elements) {
        أرجع;
    }

    صحيح n, c, باہرلا_y, باہرلا_x;
    Layout::unpack_indices(global_idx, p, n, c, باہرلا_y, باہرلا_x);

    عائم acc = 0.0f;
    ثابت conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(باہرلا_x, باہرلا_y, p);

    لكل (صحيح ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ثابت صحيح وچ_y = dw_calculate_داخلہ_coord(باہرلا_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        لكل (صحيح kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ثابت صحيح وچ_x = dw_calculate_داخلہ_coord(باہرلا_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += داخلہ[Layout::داخلہ_اشاریہ(n, c, وچ_y, وچ_x, p)] *
                   static_cast<عائم>(kernel[Layout::kernel_اشاریہ(c, ky, kx, p)]);
        }
    }

    output[Layout::output_اشاریہ(n, c, باہرلا_y, باہرلا_x, p)] = acc;
}

سانچا <قسم_ناں KernelT, قسم_ناں Layout>
ساكن فراغ conv2d_dw_sycl(ثابت عائم * x_d, ثابت KernelT * w_d, عائم * y_d,
                            ثابت conv2d_dw_params p, ثابت قطار_ptr & stream) {
    ثابت صحيح کل      = p.batches * p.channels * p.باہرلا_h * p.باہرلا_w;
    ثابت صحيح عدد_بلاکاں = (کل + SYCL_CONV2_D_DW_بلاک_ناپ - 1) / SYCL_CONV2_D_DW_بلاک_ناپ;
    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_CONV2_D_DW_بلاک_ناپ);
    ثابت sycl::range<3> بلاک_nums(1, 1, عدد_بلاکاں);
    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, شے_ct1);
        });
}

فراغ ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * داخلہ  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                داخلہ->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ثابت عائم * x_d = (ثابت عائم *) داخلہ->انکڑے;
    عائم *       y_d = (عائم *) dst->انکڑے;

    ثابت int32_t * p          = (ثابت int32_t *) dst->op_params;
    ثابت صحيح       stride_x   = p[0];
    ثابت صحيح       stride_y   = p[1];
    ثابت صحيح       padding_x  = p[2];
    ثابت صحيح       padding_y  = p[3];
    ثابت صحيح       dilation_x = p[4];
    ثابت صحيح       dilation_y = p[5];

    ثابت صحيح وچ_w     = داخلہ->ne[0];
    ثابت صحيح وچ_h     = داخلہ->ne[1];
    ثابت صحيح kernel_w = kernel->ne[0];
    ثابت صحيح kernel_h = kernel->ne[1];
    ثابت صحيح باہرلا_w    = dst->ne[0];
    ثابت صحيح باہرلا_h    = dst->ne[1];
    ثابت صحيح channels = dst->ne[2];
    ثابت صحيح batches  = dst->ne[3];

    ثابت conv2d_dw_params params = { وچ_w, وچ_h, باہرلا_w, باہرلا_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ثابت قطار_ptr stream = ctx.stream();

    إذا (kernel->type == GGML_TYPE_F16) {
        ثابت sycl::half * w_d = (ثابت sycl::half *) kernel->انکڑے;
        إذا (ggml_is_contiguous(داخلہ)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا إذا (ggml_is_contiguous_channels(داخلہ)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } وإلا {
        ثابت عائم * w_d = (ثابت عائم *) kernel->انکڑے;
        إذا (ggml_is_contiguous(داخلہ)) {
            conv2d_dw_sycl<عائم, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا إذا (ggml_is_contiguous_channels(داخلہ)) {
            conv2d_dw_sycl<عائم, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
