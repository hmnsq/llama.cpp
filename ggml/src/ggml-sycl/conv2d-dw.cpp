#અન્તર્ભાવ "conv2d-dw.hpp"

સંરચના conv2d_dw_params {
    પૂર્ણાંક in_w, in_h;
    પૂર્ણાંક out_w, out_h;
    પૂર્ણાંક kernel_w, kernel_h;
    પૂર્ણાંક stride_x, stride_y;
    પૂર્ણાંક padding_x, padding_y;
    પૂર્ણાંક dilation_x, dilation_y;
    પૂર્ણાંક channels, batches;
};

સંરચના conv2d_dw_kernel_bounds {
    પૂર્ણાંક y_min, y_max;
    પૂર્ણાંક x_min, x_max;
};

સ્થિર અંતઃરેખ conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(પૂર્ણાંક out_x, પૂર્ણાંક out_y,
                                                                  અચલ conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.in_h + p.padding_y - out_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.in_w + p.padding_x - out_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    પાછા bounds;
}

સ્થિર અંતઃરેખ પૂર્ણાંક dw_calculate_આગત_coord(પૂર્ણાંક out_coord, પૂર્ણાંક kern_coord, પૂર્ણાંક stride, પૂર્ણાંક dilation, પૂર્ણાંક padding) {
    પાછા out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
સંરચના dw_whcn_layout {
    સ્થિર પૂર્ણાંક આગત_index(પૂર્ણાંક n, પૂર્ણાંક c, પૂર્ણાંક y, પૂર્ણાંક x, અચલ conv2d_dw_params & p) {
        પાછા n * (p.channels * p.in_w * p.in_h) + c * p.in_w * p.in_h + y * p.in_w + x;
    }
    સ્થિર પૂર્ણાંક kernel_index(પૂર્ણાંક c, પૂર્ણાંક ky, પૂર્ણાંક kx, અચલ conv2d_dw_params & p) {
        પાછા c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    સ્થિર પૂર્ણાંક output_index(પૂર્ણાંક n, પૂર્ણાંક c, પૂર્ણાંક y, પૂર્ણાંક x, અચલ conv2d_dw_params & p) {
        પાછા n * (p.channels * p.out_w * p.out_h) + c * p.out_w * p.out_h + y * p.out_w + x;
    }
    સ્થિર શૂન્ય unpack_indices(પૂર્ણાંક global_idx, અચલ conv2d_dw_params & p,
                               પૂર્ણાંક & n, પૂર્ણાંક & c, પૂર્ણાંક & out_y, પૂર્ણાંક & out_x) {
        out_x  = global_idx % p.out_w;
        out_y  = (global_idx / p.out_w) % p.out_h;
        c      = (global_idx / (p.out_w * p.out_h)) % p.channels;
        n      = global_idx / (p.out_w * p.out_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
સંરચના dw_cwhn_layout {
    સ્થિર પૂર્ણાંક આગત_index(પૂર્ણાંક n, પૂર્ણાંક c, પૂર્ણાંક y, પૂર્ણાંક x, અચલ conv2d_dw_params & p) {
        પાછા n * (p.channels * p.in_w * p.in_h) + (y * p.in_w + x) * p.channels + c;
    }
    સ્થિર પૂર્ણાંક kernel_index(પૂર્ણાંક c, પૂર્ણાંક ky, પૂર્ણાંક kx, અચલ conv2d_dw_params & p) {
        પાછા (ky * p.kernel_w + kx) * p.channels + c;
    }
    સ્થિર પૂર્ણાંક output_index(પૂર્ણાંક n, પૂર્ણાંક c, પૂર્ણાંક y, પૂર્ણાંક x, અચલ conv2d_dw_params & p) {
        પાછા n * (p.channels * p.out_w * p.out_h) + y * (p.out_w * p.channels) + x * p.channels + c;
    }
    સ્થિર શૂન્ય unpack_indices(પૂર્ણાંક global_idx, અચલ conv2d_dw_params & p,
                               પૂર્ણાંક & n, પૂર્ણાંક & c, પૂર્ણાંક & out_y, પૂર્ણાંક & out_x) {
        c      = global_idx % p.channels;
        out_x  = (global_idx / p.channels) % p.out_w;
        out_y  = (global_idx / (p.channels * p.out_w)) % p.out_h;
        n      = global_idx / (p.channels * p.out_w * p.out_h);
    }
};

template <typename KernelT, typename Layout>
સ્થિર શૂન્ય conv2d_dw_kernel(અચલ દશાંશ * આગત, અચલ KernelT * kernel, દશાંશ * output,
                             અચલ conv2d_dw_params p, અચલ sycl::nd_item<3> & item_ct1) {
    અચલ પૂર્ણાંક global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    અચલ પૂર્ણાંક total_elements = p.batches * p.channels * p.out_h * p.out_w;

    જો (global_idx >= total_elements) {
        પાછા;
    }

    પૂર્ણાંક n, c, out_y, out_x;
    Layout::unpack_indices(global_idx, p, n, c, out_y, out_x);

    દશાંશ acc = 0.0f;
    અચલ conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(out_x, out_y, p);

    માટે (પૂર્ણાંક ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        અચલ પૂર્ણાંક in_y = dw_calculate_આગત_coord(out_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        માટે (પૂર્ણાંક kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            અચલ પૂર્ણાંક in_x = dw_calculate_આગત_coord(out_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += આગત[Layout::આગત_index(n, c, in_y, in_x, p)] *
                   static_cast<દશાંશ>(kernel[Layout::kernel_index(c, ky, kx, p)]);
        }
    }

    output[Layout::output_index(n, c, out_y, out_x, p)] = acc;
}

template <typename KernelT, typename Layout>
સ્થિર શૂન્ય conv2d_dw_sycl(અચલ દશાંશ * x_d, અચલ KernelT * w_d, દશાંશ * y_d,
                            અચલ conv2d_dw_params p, અચલ queue_ptr & stream) {
    અચલ પૂર્ણાંક total      = p.batches * p.channels * p.out_h * p.out_w;
    અચલ પૂર્ણાંક અંક_blocks = (total + SYCL_CONV2D_DW_BLOCK_SIZE - 1) / SYCL_CONV2D_DW_BLOCK_SIZE;
    અચલ sycl::range<3> block_dims(1, 1, SYCL_CONV2D_DW_BLOCK_SIZE);
    અચલ sycl::range<3> block_nums(1, 1, અંક_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, item_ct1);
        });
}

શૂન્ય ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    અચલ ggml_tensor * kernel = dst->src[0];
    અચલ ggml_tensor * આગત  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                આગત->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    અચલ દશાંશ * x_d = (અચલ દશાંશ *) આગત->data;
    દશાંશ *       y_d = (દશાંશ *) dst->data;

    અચલ int32_t * p          = (અચલ int32_t *) dst->op_params;
    અચલ પૂર્ણાંક       stride_x   = p[0];
    અચલ પૂર્ણાંક       stride_y   = p[1];
    અચલ પૂર્ણાંક       padding_x  = p[2];
    અચલ પૂર્ણાંક       padding_y  = p[3];
    અચલ પૂર્ણાંક       dilation_x = p[4];
    અચલ પૂર્ણાંક       dilation_y = p[5];

    અચલ પૂર્ણાંક in_w     = આગત->ne[0];
    અચલ પૂર્ણાંક in_h     = આગત->ne[1];
    અચલ પૂર્ણાંક kernel_w = kernel->ne[0];
    અચલ પૂર્ણાંક kernel_h = kernel->ne[1];
    અચલ પૂર્ણાંક out_w    = dst->ne[0];
    અચલ પૂર્ણાંક out_h    = dst->ne[1];
    અચલ પૂર્ણાંક channels = dst->ne[2];
    અચલ પૂર્ણાંક batches  = dst->ne[3];

    અચલ conv2d_dw_params params = { in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    અચલ queue_ptr stream = ctx.stream();

    જો (kernel->type == GGML_TYPE_F16) {
        અચલ sycl::half * w_d = (અચલ sycl::half *) kernel->data;
        જો (ggml_is_contiguous(આગત)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } નહીંતર જો (ggml_is_contiguous_channels(આગત)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } નહીંતર {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } નહીંતર {
        અચલ દશાંશ * w_d = (અચલ દશાંશ *) kernel->data;
        જો (ggml_is_contiguous(આગત)) {
            conv2d_dw_sycl<દશાંશ, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } નહીંતર જો (ggml_is_contiguous_channels(આગત)) {
            conv2d_dw_sycl<દશાંશ, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } નહીંતર {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
