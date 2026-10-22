#ضمن "conv2d-dw.hpp"

بنية conv2d_dw_params {
    صحيح اندريون_w, اندريون_h;
    صحيح ٻاهريون_w, ٻاهريون_h;
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

ساكن مضمن conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(صحيح ٻاهريون_x, صحيح ٻاهريون_y,
                                                                  ثابت conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - ٻاهريون_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.اندريون_h + p.padding_y - ٻاهريون_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - ٻاهريون_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.اندريون_w + p.padding_x - ٻاهريون_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    أرجع bounds;
}

ساكن مضمن صحيح dw_calculate_داخلا_coord(صحيح ٻاهريون_coord, صحيح kern_coord, صحيح stride, صحيح dilation, صحيح padding) {
    أرجع ٻاهريون_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
بنية dw_whcn_layout {
    ساكن صحيح داخلا_اشاريو(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.اندريون_w * p.اندريون_h) + c * p.اندريون_w * p.اندريون_h + y * p.اندريون_w + x;
    }
    ساكن صحيح kernel_اشاريو(صحيح c, صحيح ky, صحيح kx, ثابت conv2d_dw_params & p) {
        أرجع c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    ساكن صحيح output_اشاريو(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.ٻاهريون_w * p.ٻاهريون_h) + c * p.ٻاهريون_w * p.ٻاهريون_h + y * p.ٻاهريون_w + x;
    }
    ساكن فراغ unpack_indices(صحيح global_idx, ثابت conv2d_dw_params & p,
                               صحيح & n, صحيح & c, صحيح & ٻاهريون_y, صحيح & ٻاهريون_x) {
        ٻاهريون_x  = global_idx % p.ٻاهريون_w;
        ٻاهريون_y  = (global_idx / p.ٻاهريون_w) % p.ٻاهريون_h;
        c      = (global_idx / (p.ٻاهريون_w * p.ٻاهريون_h)) % p.channels;
        n      = global_idx / (p.ٻاهريون_w * p.ٻاهريون_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
بنية dw_cwhn_layout {
    ساكن صحيح داخلا_اشاريو(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.اندريون_w * p.اندريون_h) + (y * p.اندريون_w + x) * p.channels + c;
    }
    ساكن صحيح kernel_اشاريو(صحيح c, صحيح ky, صحيح kx, ثابت conv2d_dw_params & p) {
        أرجع (ky * p.kernel_w + kx) * p.channels + c;
    }
    ساكن صحيح output_اشاريو(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.ٻاهريون_w * p.ٻاهريون_h) + y * (p.ٻاهريون_w * p.channels) + x * p.channels + c;
    }
    ساكن فراغ unpack_indices(صحيح global_idx, ثابت conv2d_dw_params & p,
                               صحيح & n, صحيح & c, صحيح & ٻاهريون_y, صحيح & ٻاهريون_x) {
        c      = global_idx % p.channels;
        ٻاهريون_x  = (global_idx / p.channels) % p.ٻاهريون_w;
        ٻاهريون_y  = (global_idx / (p.channels * p.ٻاهريون_w)) % p.ٻاهريون_h;
        n      = global_idx / (p.channels * p.ٻاهريون_w * p.ٻاهريون_h);
    }
};

سانچو <قسم_نالو KernelT, قسم_نالو Layout>
ساكن فراغ conv2d_dw_kernel(ثابت عائم * داخلا_2, ثابت KernelT * kernel, عائم * output,
                             ثابت conv2d_dw_params p, ثابت sycl::nd_شي<3> & شي_ct1) {
    ثابت صحيح global_idx     = شي_ct1.get_local_id(2) +
                               شي_ct1.get_گروهه(2) * شي_ct1.get_local_range(2);
    ثابت صحيح ڪل_elements = p.batches * p.channels * p.ٻاهريون_h * p.ٻاهريون_w;

    إذا (global_idx >= ڪل_elements) {
        أرجع;
    }

    صحيح n, c, ٻاهريون_y, ٻاهريون_x;
    Layout::unpack_indices(global_idx, p, n, c, ٻاهريون_y, ٻاهريون_x);

    عائم acc = 0.0f;
    ثابت conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(ٻاهريون_x, ٻاهريون_y, p);

    لكل (صحيح ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ثابت صحيح اندريون_y = dw_calculate_داخلا_coord(ٻاهريون_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        لكل (صحيح kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ثابت صحيح اندريون_x = dw_calculate_داخلا_coord(ٻاهريون_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += داخلا_2[Layout::داخلا_اشاريو(n, c, اندريون_y, اندريون_x, p)] *
                   static_cast<عائم>(kernel[Layout::kernel_اشاريو(c, ky, kx, p)]);
        }
    }

    output[Layout::output_اشاريو(n, c, ٻاهريون_y, ٻاهريون_x, p)] = acc;
}

سانچو <قسم_نالو KernelT, قسم_نالو Layout>
ساكن فراغ conv2d_dw_sycl(ثابت عائم * x_d, ثابت KernelT * w_d, عائم * y_d,
                            ثابت conv2d_dw_params p, ثابت قطار_ptr & stream) {
    ثابت صحيح ڪل      = p.batches * p.channels * p.ٻاهريون_h * p.ٻاهريون_w;
    ثابت صحيح انگ_بلاڪون = (ڪل + SYCL_CONV2_D_DW_بلاڪ_سائيز - 1) / SYCL_CONV2_D_DW_بلاڪ_سائيز;
    ثابت sycl::range<3> بلاڪ_dims(1, 1, SYCL_CONV2_D_DW_بلاڪ_سائيز);
    ثابت sycl::range<3> بلاڪ_nums(1, 1, انگ_بلاڪون);
    stream->parallel_for(sycl::nd_range<3>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
        [=](sycl::nd_شي<3> شي_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, شي_ct1);
        });
}

فراغ ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * داخلا_2  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                داخلا_2->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ثابت عائم * x_d = (ثابت عائم *) داخلا_2->معلومات;
    عائم *       y_d = (عائم *) dst->معلومات;

    ثابت int32_t * p          = (ثابت int32_t *) dst->op_params;
    ثابت صحيح       stride_x   = p[0];
    ثابت صحيح       stride_y   = p[1];
    ثابت صحيح       padding_x  = p[2];
    ثابت صحيح       padding_y  = p[3];
    ثابت صحيح       dilation_x = p[4];
    ثابت صحيح       dilation_y = p[5];

    ثابت صحيح اندريون_w     = داخلا_2->ne[0];
    ثابت صحيح اندريون_h     = داخلا_2->ne[1];
    ثابت صحيح kernel_w = kernel->ne[0];
    ثابت صحيح kernel_h = kernel->ne[1];
    ثابت صحيح ٻاهريون_w    = dst->ne[0];
    ثابت صحيح ٻاهريون_h    = dst->ne[1];
    ثابت صحيح channels = dst->ne[2];
    ثابت صحيح batches  = dst->ne[3];

    ثابت conv2d_dw_params params = { اندريون_w, اندريون_h, ٻاهريون_w, ٻاهريون_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ثابت قطار_ptr stream = ctx.stream();

    إذا (kernel->type == GGML_TYPE_F16) {
        ثابت sycl::half * w_d = (ثابت sycl::half *) kernel->معلومات;
        إذا (ggml_is_contiguous(داخلا_2)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا إذا (ggml_is_contiguous_channels(داخلا_2)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } وإلا {
        ثابت عائم * w_d = (ثابت عائم *) kernel->معلومات;
        إذا (ggml_is_contiguous(داخلا_2)) {
            conv2d_dw_sycl<عائم, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا إذا (ggml_is_contiguous_channels(داخلا_2)) {
            conv2d_dw_sycl<عائم, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
