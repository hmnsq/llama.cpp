#ਅਨ੍ਤਰ੍ਭਾਵ "conv2d-dw.hpp"

ਬਣਤਰ conv2d_dw_params {
    ਪੂਰਨ ਵਿੱਚ_w, ਵਿੱਚ_h;
    ਪੂਰਨ ਬਾਹਰਲਾ_w, ਬਾਹਰਲਾ_h;
    ਪੂਰਨ kernel_w, kernel_h;
    ਪੂਰਨ stride_x, stride_y;
    ਪੂਰਨ padding_x, padding_y;
    ਪੂਰਨ dilation_x, dilation_y;
    ਪੂਰਨ channels, batches;
};

ਬਣਤਰ conv2d_dw_kernel_bounds {
    ਪੂਰਨ y_min, y_max;
    ਪੂਰਨ x_min, x_max;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(ਪੂਰਨ ਬਾਹਰਲਾ_x, ਪੂਰਨ ਬਾਹਰਲਾ_y,
                                                                  ਅਚਲ conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - ਬਾਹਰਲਾ_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.ਵਿੱਚ_h + p.padding_y - ਬਾਹਰਲਾ_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - ਬਾਹਰਲਾ_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.ਵਿੱਚ_w + p.padding_x - ਬਾਹਰਲਾ_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    ਵਾਪਸ bounds;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ dw_calculate_ਨਿਵੇਸ਼_coord(ਪੂਰਨ ਬਾਹਰਲਾ_coord, ਪੂਰਨ kern_coord, ਪੂਰਨ stride, ਪੂਰਨ dilation, ਪੂਰਨ padding) {
    ਵਾਪਸ ਬਾਹਰਲਾ_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
ਬਣਤਰ dw_whcn_layout {
    ਸਥਿਰ ਪੂਰਨ ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ(ਪੂਰਨ n, ਪੂਰਨ c, ਪੂਰਨ y, ਪੂਰਨ x, ਅਚਲ conv2d_dw_params & p) {
        ਵਾਪਸ n * (p.channels * p.ਵਿੱਚ_w * p.ਵਿੱਚ_h) + c * p.ਵਿੱਚ_w * p.ਵਿੱਚ_h + y * p.ਵਿੱਚ_w + x;
    }
    ਸਥਿਰ ਪੂਰਨ kernel_ਸੂਚਕਾਂਕ(ਪੂਰਨ c, ਪੂਰਨ ky, ਪੂਰਨ kx, ਅਚਲ conv2d_dw_params & p) {
        ਵਾਪਸ c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    ਸਥਿਰ ਪੂਰਨ output_ਸੂਚਕਾਂਕ(ਪੂਰਨ n, ਪੂਰਨ c, ਪੂਰਨ y, ਪੂਰਨ x, ਅਚਲ conv2d_dw_params & p) {
        ਵਾਪਸ n * (p.channels * p.ਬਾਹਰਲਾ_w * p.ਬਾਹਰਲਾ_h) + c * p.ਬਾਹਰਲਾ_w * p.ਬਾਹਰਲਾ_h + y * p.ਬਾਹਰਲਾ_w + x;
    }
    ਸਥਿਰ ਖਾਲੀ unpack_indices(ਪੂਰਨ global_idx, ਅਚਲ conv2d_dw_params & p,
                               ਪੂਰਨ & n, ਪੂਰਨ & c, ਪੂਰਨ & ਬਾਹਰਲਾ_y, ਪੂਰਨ & ਬਾਹਰਲਾ_x) {
        ਬਾਹਰਲਾ_x  = global_idx % p.ਬਾਹਰਲਾ_w;
        ਬਾਹਰਲਾ_y  = (global_idx / p.ਬਾਹਰਲਾ_w) % p.ਬਾਹਰਲਾ_h;
        c      = (global_idx / (p.ਬਾਹਰਲਾ_w * p.ਬਾਹਰਲਾ_h)) % p.channels;
        n      = global_idx / (p.ਬਾਹਰਲਾ_w * p.ਬਾਹਰਲਾ_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
ਬਣਤਰ dw_cwhn_layout {
    ਸਥਿਰ ਪੂਰਨ ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ(ਪੂਰਨ n, ਪੂਰਨ c, ਪੂਰਨ y, ਪੂਰਨ x, ਅਚਲ conv2d_dw_params & p) {
        ਵਾਪਸ n * (p.channels * p.ਵਿੱਚ_w * p.ਵਿੱਚ_h) + (y * p.ਵਿੱਚ_w + x) * p.channels + c;
    }
    ਸਥਿਰ ਪੂਰਨ kernel_ਸੂਚਕਾਂਕ(ਪੂਰਨ c, ਪੂਰਨ ky, ਪੂਰਨ kx, ਅਚਲ conv2d_dw_params & p) {
        ਵਾਪਸ (ky * p.kernel_w + kx) * p.channels + c;
    }
    ਸਥਿਰ ਪੂਰਨ output_ਸੂਚਕਾਂਕ(ਪੂਰਨ n, ਪੂਰਨ c, ਪੂਰਨ y, ਪੂਰਨ x, ਅਚਲ conv2d_dw_params & p) {
        ਵਾਪਸ n * (p.channels * p.ਬਾਹਰਲਾ_w * p.ਬਾਹਰਲਾ_h) + y * (p.ਬਾਹਰਲਾ_w * p.channels) + x * p.channels + c;
    }
    ਸਥਿਰ ਖਾਲੀ unpack_indices(ਪੂਰਨ global_idx, ਅਚਲ conv2d_dw_params & p,
                               ਪੂਰਨ & n, ਪੂਰਨ & c, ਪੂਰਨ & ਬਾਹਰਲਾ_y, ਪੂਰਨ & ਬਾਹਰਲਾ_x) {
        c      = global_idx % p.channels;
        ਬਾਹਰਲਾ_x  = (global_idx / p.channels) % p.ਬਾਹਰਲਾ_w;
        ਬਾਹਰਲਾ_y  = (global_idx / (p.channels * p.ਬਾਹਰਲਾ_w)) % p.ਬਾਹਰਲਾ_h;
        n      = global_idx / (p.channels * p.ਬਾਹਰਲਾ_w * p.ਬਾਹਰਲਾ_h);
    }
};

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ KernelT, ਕਿਸਮ_ਨਾਂ Layout>
ਸਥਿਰ ਖਾਲੀ conv2d_dw_kernel(ਅਚਲ ਦਸ਼ਮਲਵ * ਨਿਵੇਸ਼, ਅਚਲ KernelT * kernel, ਦਸ਼ਮਲਵ * output,
                             ਅਚਲ conv2d_dw_params p, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਅਚਲ ਪੂਰਨ global_idx     = ਵਸਤੂ_ct1.get_local_id(2) +
                               ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
    ਅਚਲ ਪੂਰਨ ਕੁੱਲ_elements = p.batches * p.channels * p.ਬਾਹਰਲਾ_h * p.ਬਾਹਰਲਾ_w;

    ਜੇ (global_idx >= ਕੁੱਲ_elements) {
        ਵਾਪਸ;
    }

    ਪੂਰਨ n, c, ਬਾਹਰਲਾ_y, ਬਾਹਰਲਾ_x;
    Layout::unpack_indices(global_idx, p, n, c, ਬਾਹਰਲਾ_y, ਬਾਹਰਲਾ_x);

    ਦਸ਼ਮਲਵ acc = 0.0f;
    ਅਚਲ conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(ਬਾਹਰਲਾ_x, ਬਾਹਰਲਾ_y, p);

    ਲਈ (ਪੂਰਨ ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ਅਚਲ ਪੂਰਨ ਵਿੱਚ_y = dw_calculate_ਨਿਵੇਸ਼_coord(ਬਾਹਰਲਾ_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        ਲਈ (ਪੂਰਨ kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ਅਚਲ ਪੂਰਨ ਵਿੱਚ_x = dw_calculate_ਨਿਵੇਸ਼_coord(ਬਾਹਰਲਾ_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += ਨਿਵੇਸ਼[Layout::ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ(n, c, ਵਿੱਚ_y, ਵਿੱਚ_x, p)] *
                   static_cast<ਦਸ਼ਮਲਵ>(kernel[Layout::kernel_ਸੂਚਕਾਂਕ(c, ky, kx, p)]);
        }
    }

    output[Layout::output_ਸੂਚਕਾਂਕ(n, c, ਬਾਹਰਲਾ_y, ਬਾਹਰਲਾ_x, p)] = acc;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ KernelT, ਕਿਸਮ_ਨਾਂ Layout>
ਸਥਿਰ ਖਾਲੀ conv2d_dw_sycl(ਅਚਲ ਦਸ਼ਮਲਵ * x_d, ਅਚਲ KernelT * w_d, ਦਸ਼ਮਲਵ * y_d,
                            ਅਚਲ conv2d_dw_params p, ਅਚਲ ਕਤਾਰ_ptr & stream) {
    ਅਚਲ ਪੂਰਨ ਕੁੱਲ      = p.batches * p.channels * p.ਬਾਹਰਲਾ_h * p.ਬਾਹਰਲਾ_w;
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ਕੁੱਲ + SYCL_CONV2_D_DW_ਖੰਡ_ਮਾਪ - 1) / SYCL_CONV2_D_DW_ਖੰਡ_ਮਾਪ;
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, SYCL_CONV2_D_DW_ਖੰਡ_ਮਾਪ);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਅੰਕ_ਖੰਡਾਂ);
    stream->parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, ਵਸਤੂ_ct1);
        });
}

ਖਾਲੀ ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ਅਚਲ ggml_tensor * kernel = dst->src[0];
    ਅਚਲ ggml_tensor * ਨਿਵੇਸ਼  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                ਨਿਵੇਸ਼->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ਅਚਲ ਦਸ਼ਮਲਵ * x_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) ਨਿਵੇਸ਼->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *       y_d = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਅਚਲ int32_t * p          = (ਅਚਲ int32_t *) dst->op_params;
    ਅਚਲ ਪੂਰਨ       stride_x   = p[0];
    ਅਚਲ ਪੂਰਨ       stride_y   = p[1];
    ਅਚਲ ਪੂਰਨ       padding_x  = p[2];
    ਅਚਲ ਪੂਰਨ       padding_y  = p[3];
    ਅਚਲ ਪੂਰਨ       dilation_x = p[4];
    ਅਚਲ ਪੂਰਨ       dilation_y = p[5];

    ਅਚਲ ਪੂਰਨ ਵਿੱਚ_w     = ਨਿਵੇਸ਼->ne[0];
    ਅਚਲ ਪੂਰਨ ਵਿੱਚ_h     = ਨਿਵੇਸ਼->ne[1];
    ਅਚਲ ਪੂਰਨ kernel_w = kernel->ne[0];
    ਅਚਲ ਪੂਰਨ kernel_h = kernel->ne[1];
    ਅਚਲ ਪੂਰਨ ਬਾਹਰਲਾ_w    = dst->ne[0];
    ਅਚਲ ਪੂਰਨ ਬਾਹਰਲਾ_h    = dst->ne[1];
    ਅਚਲ ਪੂਰਨ channels = dst->ne[2];
    ਅਚਲ ਪੂਰਨ batches  = dst->ne[3];

    ਅਚਲ conv2d_dw_params params = { ਵਿੱਚ_w, ਵਿੱਚ_h, ਬਾਹਰਲਾ_w, ਬਾਹਰਲਾ_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ਅਚਲ ਕਤਾਰ_ptr stream = ctx.stream();

    ਜੇ (kernel->type == GGML_TYPE_F16) {
        ਅਚਲ sycl::half * w_d = (ਅਚਲ sycl::half *) kernel->ਅੰਕੜੇ;
        ਜੇ (ggml_is_contiguous(ਨਿਵੇਸ਼)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } ਨਹੀਂਤਾਂ ਜੇ (ggml_is_contiguous_channels(ਨਿਵੇਸ਼)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } ਨਹੀਂਤਾਂ {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } ਨਹੀਂਤਾਂ {
        ਅਚਲ ਦਸ਼ਮਲਵ * w_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) kernel->ਅੰਕੜੇ;
        ਜੇ (ggml_is_contiguous(ਨਿਵੇਸ਼)) {
            conv2d_dw_sycl<ਦਸ਼ਮਲਵ, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } ਨਹੀਂਤਾਂ ਜੇ (ggml_is_contiguous_channels(ਨਿਵੇਸ਼)) {
            conv2d_dw_sycl<ਦਸ਼ਮਲਵ, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } ਨਹੀਂਤਾਂ {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
