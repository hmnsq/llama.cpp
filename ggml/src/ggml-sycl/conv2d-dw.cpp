#അന്തര്ഭാവ "conv2d-dw.hpp"

ഘടന conv2d_dw_params {
    പൂർണ്ണസംഖ്യ in_w, in_h;
    പൂർണ്ണസംഖ്യ out_w, out_h;
    പൂർണ്ണസംഖ്യ kernel_w, kernel_h;
    പൂർണ്ണസംഖ്യ stride_x, stride_y;
    പൂർണ്ണസംഖ്യ padding_x, padding_y;
    പൂർണ്ണസംഖ്യ dilation_x, dilation_y;
    പൂർണ്ണസംഖ്യ channels, batches;
};

ഘടന conv2d_dw_kernel_bounds {
    പൂർണ്ണസംഖ്യ y_min, y_max;
    പൂർണ്ണസംഖ്യ x_min, x_max;
};

സ്ഥിതം വരിയിൽ conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(പൂർണ്ണസംഖ്യ out_x, പൂർണ്ണസംഖ്യ out_y,
                                                                  സ്ഥിരം conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    തിരികെ bounds;
}

സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ dw_calculate_നിവേശം_coord(പൂർണ്ണസംഖ്യ out_coord, പൂർണ്ണസംഖ്യ kern_coord, പൂർണ്ണസംഖ്യ stride, പൂർണ്ണസംഖ്യ dilation, പൂർണ്ണസംഖ്യ padding) {
    തിരികെ out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
ഘടന dw_whcn_layout {
    സ്ഥിതം പൂർണ്ണസംഖ്യ നിവേശം_index(പൂർണ്ണസംഖ്യ n, പൂർണ്ണസംഖ്യ c, പൂർണ്ണസംഖ്യ y, പൂർണ്ണസംഖ്യ x, സ്ഥിരം conv2d_dw_params & p) {
        തിരികെ n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    സ്ഥിതം പൂർണ്ണസംഖ്യ kernel_index(പൂർണ്ണസംഖ്യ c, പൂർണ്ണസംഖ്യ ky, പൂർണ്ണസംഖ്യ kx, സ്ഥിരം conv2d_dw_params & p) {
        തിരികെ c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    സ്ഥിതം പൂർണ്ണസംഖ്യ output_index(പൂർണ്ണസംഖ്യ n, പൂർണ്ണസംഖ്യ c, പൂർണ്ണസംഖ്യ y, പൂർണ്ണസംഖ്യ x, സ്ഥിരം conv2d_dw_params & p) {
        തിരികെ n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    സ്ഥിതം ശൂന്യം unpack_indices(പൂർണ്ണസംഖ്യ global_idx, സ്ഥിരം conv2d_dw_params & p,
                               പൂർണ്ണസംഖ്യ & n, പൂർണ്ണസംഖ്യ & c, പൂർണ്ണസംഖ്യ & out_y, പൂർണ്ണസംഖ്യ & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
ഘടന dw_cwhn_layout {
    സ്ഥിതം പൂർണ്ണസംഖ്യ നിവേശം_index(പൂർണ്ണസംഖ്യ n, പൂർണ്ണസംഖ്യ c, പൂർണ്ണസംഖ്യ y, പൂർണ്ണസംഖ്യ x, സ്ഥിരം conv2d_dw_params & p) {
        തിരികെ n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    സ്ഥിതം പൂർണ്ണസംഖ്യ kernel_index(പൂർണ്ണസംഖ്യ c, പൂർണ്ണസംഖ്യ ky, പൂർണ്ണസംഖ്യ kx, സ്ഥിരം conv2d_dw_params & p) {
        തിരികെ (ky * p.kernel_w + kx) * p.channels + c;
    }
    സ്ഥിതം പൂർണ്ണസംഖ്യ output_index(പൂർണ്ണസംഖ്യ n, പൂർണ്ണസംഖ്യ c, പൂർണ്ണസംഖ്യ y, പൂർണ്ണസംഖ്യ x, സ്ഥിരം conv2d_dw_params & p) {
        തിരികെ n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    സ്ഥിതം ശൂന്യം unpack_indices(പൂർണ്ണസംഖ്യ global_idx, സ്ഥിരം conv2d_dw_params & p,
                               പൂർണ്ണസംഖ്യ & n, പൂർണ്ണസംഖ്യ & c, പൂർണ്ണസംഖ്യ & out_y, പൂർണ്ണസംഖ്യ & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

template <typename KernelT, typename Layout>
സ്ഥിതം ശൂന്യം conv2d_dw_kernel(സ്ഥിരം ദശാംശം * നിവേശം, സ്ഥിരം KernelT * kernel, ദശാംശം * output,
                             സ്ഥിരം conv2d_dw_params p, സ്ഥിരം sycl::nd_item<3> & item_ct1) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ total_elements = p.batches * p.channels * p.out_h * p.out_w;

    എങ്കിൽ (global_idx >= total_elements) {
        തിരികെ;
    }

    പൂർണ്ണസംഖ്യ n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    ദശാംശം acc = 0.0f;
    സ്ഥിരം conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    വേണ്ടി (പൂർണ്ണസംഖ്യ ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        സ്ഥിരം പൂർണ്ണസംഖ്യ in_y = dw_calculate_നിവേശം_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        വേണ്ടി (പൂർണ്ണസംഖ്യ kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            സ്ഥിരം പൂർണ്ണസംഖ്യ in_x = dw_calculate_നിവേശം_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += നിവേശം[Layout::നിവേശം_index(n, c, in_y, in_x, p)] *
                   static_cast<ദശാംശം>(kernel[Layout::kernel_index(c, ky, kx, p)]);
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

template <typename KernelT, typename Layout>
സ്ഥിതം ശൂന്യം conv2d_dw_sycl(സ്ഥിരം ദശാംശം * x_d, സ്ഥിരം KernelT * w_d, ദശാംശം * y_d,
                            സ്ഥിരം conv2d_dw_params p, സ്ഥിരം queue_ptr & stream) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ total      = p.batches * p.channels * p.out_h * p.out_w;
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
    സ്ഥിരം sycl::range<3> block_dims(1, 1, SYCL_CONV2D_DW_BLOCK_SIZE);
    സ്ഥിരം sycl::range<3> block_nums(1, 1, അക്കം_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

ശൂന്യം ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    സ്ഥിരം ggml_tensor * kernel = dst->src[0];
    സ്ഥിരം ggml_tensor * നിവേശം  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                നിവേശം->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    സ്ഥിരം ദശാംശം * x_d = (സ്ഥിരം ദശാംശം *) നിവേശം->data;
    ദശാംശം *       y_d = (ദശാംശം *) dst->data;

    സ്ഥിരം int32_t * p          = (സ്ഥിരം int32_t *) dst->op_params;
    സ്ഥിരം പൂർണ്ണസംഖ്യ       stride_x   = p[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ       stride_y   = p[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ       padding_x  = p[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ       padding_y  = p[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ       dilation_x = p[4];
    സ്ഥിരം പൂർണ്ണസംഖ്യ       dilation_y = p[5];

    സ്ഥിരം പൂർണ്ണസംഖ്യ in_w     = നിവേശം->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ in_h     = നിവേശം->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ kernel_w = kernel->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ kernel_h = kernel->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ out_w    = dst->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ out_h    = dst->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ channels = dst->ne[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ batches  = dst->ne[3];

    സ്ഥിരം conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    സ്ഥിരം queue_ptr stream = ctx.stream();

    എങ്കിൽ (kernel->type == GGML_TYPE_F16) {
        സ്ഥിരം sycl::half * w_d = (സ്ഥിരം sycl::half *) kernel->data;
        എങ്കിൽ (ggml_is_contiguous(നിവേശം)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } അല്ലെങ്കിൽ എങ്കിൽ (ggml_is_contiguous_channels(നിവേശം)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } അല്ലെങ്കിൽ {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } അല്ലെങ്കിൽ {
        സ്ഥിരം ദശാംശം * w_d = (സ്ഥിരം ദശാംശം *) kernel->data;
        എങ്കിൽ (ggml_is_contiguous(നിവേശം)) {
            conv2d_dw_sycl<ദശാംശം, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } അല്ലെങ്കിൽ എങ്കിൽ (ggml_is_contiguous_channels(നിവേശം)) {
            conv2d_dw_sycl<ദശാംശം, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } അല്ലെങ്കിൽ {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
