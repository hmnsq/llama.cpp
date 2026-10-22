#ضمن "conv2d-dw.hpp"

بنية conv2d_dw_params {
    صحيح دننه_w, دننه_h;
    صحيح بهر_w, بهر_h;
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

ساكن مضمن conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(صحيح بهر_x, صحيح بهر_y,
                                                                  ثابت conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - بهر_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.دننه_h + p.padding_y - بهر_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - بهر_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.دننه_w + p.padding_x - بهر_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    أرجع bounds;
}

ساكن مضمن صحيح dw_calculate_ننوت_coord(صحيح بهر_coord, صحيح kern_coord, صحيح stride, صحيح dilation, صحيح padding) {
    أرجع بهر_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
بنية dw_whcn_layout {
    ساكن صحيح ننوت_شاخص(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.دننه_w * p.دننه_h) + c * p.دننه_w * p.دننه_h + y * p.دننه_w + x;
    }
    ساكن صحيح kernel_شاخص(صحيح c, صحيح ky, صحيح kx, ثابت conv2d_dw_params & p) {
        أرجع c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    ساكن صحيح output_شاخص(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.بهر_w * p.بهر_h) + c * p.بهر_w * p.بهر_h + y * p.بهر_w + x;
    }
    ساكن فراغ unpack_indices(صحيح global_idx, ثابت conv2d_dw_params & p,
                               صحيح & n, صحيح & c, صحيح & بهر_y, صحيح & بهر_x) {
        بهر_x  = global_idx % p.بهر_w;
        بهر_y  = (global_idx / p.بهر_w) % p.بهر_h;
        c      = (global_idx / (p.بهر_w * p.بهر_h)) % p.channels;
        n      = global_idx / (p.بهر_w * p.بهر_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
بنية dw_cwhn_layout {
    ساكن صحيح ننوت_شاخص(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.دننه_w * p.دننه_h) + (y * p.دننه_w + x) * p.channels + c;
    }
    ساكن صحيح kernel_شاخص(صحيح c, صحيح ky, صحيح kx, ثابت conv2d_dw_params & p) {
        أرجع (ky * p.kernel_w + kx) * p.channels + c;
    }
    ساكن صحيح output_شاخص(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.بهر_w * p.بهر_h) + y * (p.بهر_w * p.channels) + x * p.channels + c;
    }
    ساكن فراغ unpack_indices(صحيح global_idx, ثابت conv2d_dw_params & p,
                               صحيح & n, صحيح & c, صحيح & بهر_y, صحيح & بهر_x) {
        c      = global_idx % p.channels;
        بهر_x  = (global_idx / p.channels) % p.بهر_w;
        بهر_y  = (global_idx / (p.channels * p.بهر_w)) % p.بهر_h;
        n      = global_idx / (p.channels * p.بهر_w * p.بهر_h);
    }
};

کينډۍ <ډول_نوم KernelT, ډول_نوم Layout>
ساكن فراغ conv2d_dw_kernel(ثابت عائم * ننوت, ثابت KernelT * kernel, عائم * output,
                             ثابت conv2d_dw_params p, ثابت sycl::nd_توکی<3> & توکی_ct1) {
    ثابت صحيح global_idx     = توکی_ct1.get_local_id(2) +
                               توکی_ct1.get_ډله(2) * توکی_ct1.get_local_range(2);
    ثابت صحيح ټول_elements = p.batches * p.channels * p.بهر_h * p.بهر_w;

    إذا (global_idx >= ټول_elements) {
        أرجع;
    }

    صحيح n, c, بهر_y, بهر_x;
    Layout::unpack_indices(global_idx, p, n, c, بهر_y, بهر_x);

    عائم acc = 0.0f;
    ثابت conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(بهر_x, بهر_y, p);

    لكل (صحيح ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ثابت صحيح دننه_y = dw_calculate_ننوت_coord(بهر_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        لكل (صحيح kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ثابت صحيح دننه_x = dw_calculate_ننوت_coord(بهر_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += ننوت[Layout::ننوت_شاخص(n, c, دننه_y, دننه_x, p)] *
                   static_cast<عائم>(kernel[Layout::kernel_شاخص(c, ky, kx, p)]);
        }
    }

    output[Layout::output_شاخص(n, c, بهر_y, بهر_x, p)] = acc;
}

کينډۍ <ډول_نوم KernelT, ډول_نوم Layout>
ساكن فراغ conv2d_dw_sycl(ثابت عائم * x_d, ثابت KernelT * w_d, عائم * y_d,
                            ثابت conv2d_dw_params p, ثابت کتار_ptr & stream) {
    ثابت صحيح ټول      = p.batches * p.channels * p.بهر_h * p.بهر_w;
    ثابت صحيح عدد_بلاکونه = (ټول + SYCL_CONV2_D_DW_بلاک_اندازه - 1) / SYCL_CONV2_D_DW_بلاک_اندازه;
    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_CONV2_D_DW_بلاک_اندازه);
    ثابت sycl::range<3> بلاک_nums(1, 1, عدد_بلاکونه);
    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_توکی<3> توکی_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, توکی_ct1);
        });
}

فراغ ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * ننوت  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                ننوت->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ثابت عائم * x_d = (ثابت عائم *) ننوت->معلومات;
    عائم *       y_d = (عائم *) dst->معلومات;

    ثابت int32_t * p          = (ثابت int32_t *) dst->op_params;
    ثابت صحيح       stride_x   = p[0];
    ثابت صحيح       stride_y   = p[1];
    ثابت صحيح       padding_x  = p[2];
    ثابت صحيح       padding_y  = p[3];
    ثابت صحيح       dilation_x = p[4];
    ثابت صحيح       dilation_y = p[5];

    ثابت صحيح دننه_w     = ننوت->ne[0];
    ثابت صحيح دننه_h     = ننوت->ne[1];
    ثابت صحيح kernel_w = kernel->ne[0];
    ثابت صحيح kernel_h = kernel->ne[1];
    ثابت صحيح بهر_w    = dst->ne[0];
    ثابت صحيح بهر_h    = dst->ne[1];
    ثابت صحيح channels = dst->ne[2];
    ثابت صحيح batches  = dst->ne[3];

    ثابت conv2d_dw_params params = { دننه_w, دننه_h, بهر_w, بهر_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ثابت کتار_ptr stream = ctx.stream();

    إذا (kernel->type == GGML_TYPE_F16) {
        ثابت sycl::half * w_d = (ثابت sycl::half *) kernel->معلومات;
        إذا (ggml_is_contiguous(ننوت)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا إذا (ggml_is_contiguous_channels(ننوت)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } وإلا {
        ثابت عائم * w_d = (ثابت عائم *) kernel->معلومات;
        إذا (ggml_is_contiguous(ننوت)) {
            conv2d_dw_sycl<عائم, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا إذا (ggml_is_contiguous_channels(ننوت)) {
            conv2d_dw_sycl<عائم, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
