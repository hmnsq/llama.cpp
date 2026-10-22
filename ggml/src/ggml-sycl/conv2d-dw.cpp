#ضمن "conv2d-dw.hpp"

بنية conv2d_dw_params {
    صحيح درون_w, درون_h;
    صحيح بیرون_w, بیرون_h;
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

ساكن مضمن conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(صحيح بیرون_x, صحيح بیرون_y,
                                                                  ثابت conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - بیرون_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.درون_h + p.padding_y - بیرون_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - بیرون_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.درون_w + p.padding_x - بیرون_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    أرجع bounds;
}

ساكن مضمن صحيح dw_calculate_درون_داد_coord(صحيح بیرون_coord, صحيح kern_coord, صحيح stride, صحيح dilation, صحيح padding) {
    أرجع بیرون_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
بنية dw_whcn_layout {
    ساكن صحيح درون_داد_شاخص(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.درون_w * p.درون_h) + c * p.درون_w * p.درون_h + y * p.درون_w + x;
    }
    ساكن صحيح kernel_شاخص(صحيح c, صحيح ky, صحيح kx, ثابت conv2d_dw_params & p) {
        أرجع c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    ساكن صحيح output_شاخص(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.بیرون_w * p.بیرون_h) + c * p.بیرون_w * p.بیرون_h + y * p.بیرون_w + x;
    }
    ساكن فراغ unpack_indices(صحيح global_idx, ثابت conv2d_dw_params & p,
                               صحيح & n, صحيح & c, صحيح & بیرون_y, صحيح & بیرون_x) {
        بیرون_x  = global_idx % p.بیرون_w;
        بیرون_y  = (global_idx / p.بیرون_w) % p.بیرون_h;
        c      = (global_idx / (p.بیرون_w * p.بیرون_h)) % p.channels;
        n      = global_idx / (p.بیرون_w * p.بیرون_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
بنية dw_cwhn_layout {
    ساكن صحيح درون_داد_شاخص(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.درون_w * p.درون_h) + (y * p.درون_w + x) * p.channels + c;
    }
    ساكن صحيح kernel_شاخص(صحيح c, صحيح ky, صحيح kx, ثابت conv2d_dw_params & p) {
        أرجع (ky * p.kernel_w + kx) * p.channels + c;
    }
    ساكن صحيح output_شاخص(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.بیرون_w * p.بیرون_h) + y * (p.بیرون_w * p.channels) + x * p.channels + c;
    }
    ساكن فراغ unpack_indices(صحيح global_idx, ثابت conv2d_dw_params & p,
                               صحيح & n, صحيح & c, صحيح & بیرون_y, صحيح & بیرون_x) {
        c      = global_idx % p.channels;
        بیرون_x  = (global_idx / p.channels) % p.بیرون_w;
        بیرون_y  = (global_idx / (p.channels * p.بیرون_w)) % p.بیرون_h;
        n      = global_idx / (p.channels * p.بیرون_w * p.بیرون_h);
    }
};

قالب <نام_نوع KernelT, نام_نوع Layout>
ساكن فراغ conv2d_dw_kernel(ثابت عائم * درون_داد, ثابت KernelT * kernel, عائم * output,
                             ثابت conv2d_dw_params p, ثابت sycl::nd_قلم<3> & قلم_ct1) {
    ثابت صحيح global_idx     = قلم_ct1.get_local_id(2) +
                               قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(2);
    ثابت صحيح مجموع_elements = p.batches * p.channels * p.بیرون_h * p.بیرون_w;

    إذا (global_idx >= مجموع_elements) {
        أرجع;
    }

    صحيح n, c, بیرون_y, بیرون_x;
    Layout::unpack_indices(global_idx, p, n, c, بیرون_y, بیرون_x);

    عائم acc = 0.0f;
    ثابت conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(بیرون_x, بیرون_y, p);

    لكل (صحيح ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ثابت صحيح درون_y = dw_calculate_درون_داد_coord(بیرون_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        لكل (صحيح kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ثابت صحيح درون_x = dw_calculate_درون_داد_coord(بیرون_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += درون_داد[Layout::درون_داد_شاخص(n, c, درون_y, درون_x, p)] *
                   static_cast<عائم>(kernel[Layout::kernel_شاخص(c, ky, kx, p)]);
        }
    }

    output[Layout::output_شاخص(n, c, بیرون_y, بیرون_x, p)] = acc;
}

قالب <نام_نوع KernelT, نام_نوع Layout>
ساكن فراغ conv2d_dw_sycl(ثابت عائم * x_d, ثابت KernelT * w_d, عائم * y_d,
                            ثابت conv2d_dw_params p, ثابت صف_ptr & stream) {
    ثابت صحيح مجموع      = p.batches * p.channels * p.بیرون_h * p.بیرون_w;
    ثابت صحيح رقم_بلوکها = (مجموع + SYCL_CONV2_D_DW_بلوک_اندازه - 1) / SYCL_CONV2_D_DW_بلوک_اندازه;
    ثابت sycl::range<3> بلوک_dims(1, 1, SYCL_CONV2_D_DW_بلوک_اندازه);
    ثابت sycl::range<3> بلوک_nums(1, 1, رقم_بلوکها);
    stream->parallel_for(sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
        [=](sycl::nd_قلم<3> قلم_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, قلم_ct1);
        });
}

فراغ ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * درون_داد  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                درون_داد->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ثابت عائم * x_d = (ثابت عائم *) درون_داد->داده;
    عائم *       y_d = (عائم *) dst->داده;

    ثابت int32_t * p          = (ثابت int32_t *) dst->op_params;
    ثابت صحيح       stride_x   = p[0];
    ثابت صحيح       stride_y   = p[1];
    ثابت صحيح       padding_x  = p[2];
    ثابت صحيح       padding_y  = p[3];
    ثابت صحيح       dilation_x = p[4];
    ثابت صحيح       dilation_y = p[5];

    ثابت صحيح درون_w     = درون_داد->ne[0];
    ثابت صحيح درون_h     = درون_داد->ne[1];
    ثابت صحيح kernel_w = kernel->ne[0];
    ثابت صحيح kernel_h = kernel->ne[1];
    ثابت صحيح بیرون_w    = dst->ne[0];
    ثابت صحيح بیرون_h    = dst->ne[1];
    ثابت صحيح channels = dst->ne[2];
    ثابت صحيح batches  = dst->ne[3];

    ثابت conv2d_dw_params params = { درون_w, درون_h, بیرون_w, بیرون_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ثابت صف_ptr stream = ctx.stream();

    إذا (kernel->type == GGML_TYPE_F16) {
        ثابت sycl::half * w_d = (ثابت sycl::half *) kernel->داده;
        إذا (ggml_is_contiguous(درون_داد)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا إذا (ggml_is_contiguous_channels(درون_داد)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } وإلا {
        ثابت عائم * w_d = (ثابت عائم *) kernel->داده;
        إذا (ggml_is_contiguous(درون_داد)) {
            conv2d_dw_sycl<عائم, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا إذا (ggml_is_contiguous_channels(درون_داد)) {
            conv2d_dw_sycl<عائم, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
