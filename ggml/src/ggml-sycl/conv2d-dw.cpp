#ضمن "conv2d-dw.hpp"

بنية conv2d_dw_params {
    صحيح in_w, in_h;
    صحيح out_w, out_h;
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

ساكن مضمن conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(صحيح out_x, صحيح out_y,
                                                                  ثابت conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    أرجع bounds;
}

ساكن مضمن صحيح dw_calculate_كىرگۈزۈش_coord(صحيح out_coord, صحيح kern_coord, صحيح stride, صحيح dilation, صحيح padding) {
    أرجع out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
بنية dw_whcn_layout {
    ساكن صحيح كىرگۈزۈش_index(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    ساكن صحيح kernel_index(صحيح c, صحيح ky, صحيح kx, ثابت conv2d_dw_params & p) {
        أرجع c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    ساكن صحيح output_index(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    ساكن فراغ unpack_indices(صحيح global_idx, ثابت conv2d_dw_params & p,
                               صحيح & n, صحيح & c, صحيح & out_y, صحيح & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
بنية dw_cwhn_layout {
    ساكن صحيح كىرگۈزۈش_index(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    ساكن صحيح kernel_index(صحيح c, صحيح ky, صحيح kx, ثابت conv2d_dw_params & p) {
        أرجع (ky * p.kernel_w + kx) * p.channels + c;
    }
    ساكن صحيح output_index(صحيح n, صحيح c, صحيح y, صحيح x, ثابت conv2d_dw_params & p) {
        أرجع n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    ساكن فراغ unpack_indices(صحيح global_idx, ثابت conv2d_dw_params & p,
                               صحيح & n, صحيح & c, صحيح & out_y, صحيح & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

template <typename KernelT, typename Layout>
ساكن فراغ conv2d_dw_kernel(ثابت عائم * كىرگۈزۈش, ثابت KernelT * kernel, عائم * output,
                             ثابت conv2d_dw_params p, ثابت sycl::nd_item<3> & item_ct1) {
    ثابت صحيح global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ثابت صحيح total_elements = p.batches * p.channels * p.out_h * p.out_w;

    إذا (global_idx >= total_elements) {
        أرجع;
    }

    صحيح n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    عائم acc = 0.0f;
    ثابت conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    لكل (صحيح ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        ثابت صحيح in_y = dw_calculate_كىرگۈزۈش_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        لكل (صحيح kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            ثابت صحيح in_x = dw_calculate_كىرگۈزۈش_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += كىرگۈزۈش[Layout::كىرگۈزۈش_index(n, c, in_y, in_x, p)] *
                   static_cast<عائم>(kernel[Layout::kernel_index(c, ky, kx, p)]);
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

template <typename KernelT, typename Layout>
ساكن فراغ conv2d_dw_sycl(ثابت عائم * x_d, ثابت KernelT * w_d, عائم * y_d,
                            ثابت conv2d_dw_params p, ثابت queue_ptr & stream) {
    ثابت صحيح total      = p.batches * p.channels * p.out_h * p.out_w;
    ثابت صحيح رەقەم_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
    ثابت sycl::range<3> block_dims(1, 1, SYCL_CONV2D_DW_BLOCK_SIZE);
    ثابت sycl::range<3> block_nums(1, 1, رەقەم_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

فراغ ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * كىرگۈزۈش  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                كىرگۈزۈش->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    ثابت عائم * x_d = (ثابت عائم *) كىرگۈزۈش->data;
    عائم *       y_d = (عائم *) dst->data;

    ثابت int32_t * p          = (ثابت int32_t *) dst->op_params;
    ثابت صحيح       stride_x   = p[0];
    ثابت صحيح       stride_y   = p[1];
    ثابت صحيح       padding_x  = p[2];
    ثابت صحيح       padding_y  = p[3];
    ثابت صحيح       dilation_x = p[4];
    ثابت صحيح       dilation_y = p[5];

    ثابت صحيح in_w     = كىرگۈزۈش->ne[0];
    ثابت صحيح in_h     = كىرگۈزۈش->ne[1];
    ثابت صحيح kernel_w = kernel->ne[0];
    ثابت صحيح kernel_h = kernel->ne[1];
    ثابت صحيح out_w    = dst->ne[0];
    ثابت صحيح out_h    = dst->ne[1];
    ثابت صحيح channels = dst->ne[2];
    ثابت صحيح batches  = dst->ne[3];

    ثابت conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    ثابت queue_ptr stream = ctx.stream();

    إذا (kernel->type == GGML_TYPE_F16) {
        ثابت sycl::half * w_d = (ثابت sycl::half *) kernel->data;
        إذا (ggml_is_contiguous(كىرگۈزۈش)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا إذا (ggml_is_contiguous_channels(كىرگۈزۈش)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } وإلا {
        ثابت عائم * w_d = (ثابت عائم *) kernel->data;
        إذا (ggml_is_contiguous(كىرگۈزۈش)) {
            conv2d_dw_sycl<عائم, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا إذا (ggml_is_contiguous_channels(كىرگۈزۈش)) {
            conv2d_dw_sycl<عائم, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } وإلا {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
