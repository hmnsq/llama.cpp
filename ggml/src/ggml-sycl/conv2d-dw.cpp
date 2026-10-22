#అన్తర్భావ "conv2d-dw.hpp"

నిర్మాణం conv2d_dw_params {
    పూర్ణాంకం లోపల_w, లోపల_h;
    పూర్ణాంకం బహిర్గతం_w, బహిర్గతం_h;
    పూర్ణాంకం kernel_w, kernel_h;
    పూర్ణాంకం stride_x, stride_y;
    పూర్ణాంకం padding_x, padding_y;
    పూర్ణాంకం dilation_x, dilation_y;
    పూర్ణాంకం channels, batches;
};

నిర్మాణం conv2d_dw_kernel_bounds {
    పూర్ణాంకం y_min, y_max;
    పూర్ణాంకం x_min, x_max;
};

స్థిర అంతఃవరుస conv2d_dw_kernel_bounds dw_calculate_kernel_bounds(పూర్ణాంకం బహిర్గతం_x, పూర్ణాంకం బహిర్గతం_y,
                                                                  స్థిరరాశి conv2d_dw_params & p) {
    conv2d_dw_kernel_bounds bounds;
    bounds.y_min = sycl::max(0, (p.padding_y - బహిర్గతం_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.y_max = sycl::min(p.kernel_h,
                             (p.లోపల_h + p.padding_y - బహిర్గతం_y * p.stride_y + p.dilation_y - 1) / p.dilation_y);
    bounds.x_min = sycl::max(0, (p.padding_x - బహిర్గతం_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    bounds.x_max = sycl::min(p.kernel_w,
                             (p.లోపల_w + p.padding_x - బహిర్గతం_x * p.stride_x + p.dilation_x - 1) / p.dilation_x);
    తిరిగి bounds;
}

స్థిర అంతఃవరుస పూర్ణాంకం dw_calculate_ఆదానం_coord(పూర్ణాంకం బహిర్గతం_coord, పూర్ణాంకం kern_coord, పూర్ణాంకం stride, పూర్ణాంకం dilation, పూర్ణాంకం padding) {
    తిరిగి బహిర్గతం_coord * stride + kern_coord * dilation - padding;
}

// whcn layout: input/output stored as [N, C, H, W]
నిర్మాణం dw_whcn_layout {
    స్థిర పూర్ణాంకం ఆదానం_సూచీ(పూర్ణాంకం n, పూర్ణాంకం c, పూర్ణాంకం y, పూర్ణాంకం x, స్థిరరాశి conv2d_dw_params & p) {
        తిరిగి n * (p.channels * p.లోపల_w * p.లోపల_h) + c * p.లోపల_w * p.లోపల_h + y * p.లోపల_w + x;
    }
    స్థిర పూర్ణాంకం kernel_సూచీ(పూర్ణాంకం c, పూర్ణాంకం ky, పూర్ణాంకం kx, స్థిరరాశి conv2d_dw_params & p) {
        తిరిగి c * p.kernel_h * p.kernel_w + ky * p.kernel_w + kx;
    }
    స్థిర పూర్ణాంకం output_సూచీ(పూర్ణాంకం n, పూర్ణాంకం c, పూర్ణాంకం y, పూర్ణాంకం x, స్థిరరాశి conv2d_dw_params & p) {
        తిరిగి n * (p.channels * p.బహిర్గతం_w * p.బహిర్గతం_h) + c * p.బహిర్గతం_w * p.బహిర్గతం_h + y * p.బహిర్గతం_w + x;
    }
    స్థిర శూన్య unpack_indices(పూర్ణాంకం global_idx, స్థిరరాశి conv2d_dw_params & p,
                               పూర్ణాంకం & n, పూర్ణాంకం & c, పూర్ణాంకం & బహిర్గతం_y, పూర్ణాంకం & బహిర్గతం_x) {
        బహిర్గతం_x  = global_idx % p.బహిర్గతం_w;
        బహిర్గతం_y  = (global_idx / p.బహిర్గతం_w) % p.బహిర్గతం_h;
        c      = (global_idx / (p.బహిర్గతం_w * p.బహిర్గతం_h)) % p.channels;
        n      = global_idx / (p.బహిర్గతం_w * p.బహిర్గతం_h * p.channels);
    }
};

// cwhn layout: input/output stored as [N, H, W, C]
నిర్మాణం dw_cwhn_layout {
    స్థిర పూర్ణాంకం ఆదానం_సూచీ(పూర్ణాంకం n, పూర్ణాంకం c, పూర్ణాంకం y, పూర్ణాంకం x, స్థిరరాశి conv2d_dw_params & p) {
        తిరిగి n * (p.channels * p.లోపల_w * p.లోపల_h) + (y * p.లోపల_w + x) * p.channels + c;
    }
    స్థిర పూర్ణాంకం kernel_సూచీ(పూర్ణాంకం c, పూర్ణాంకం ky, పూర్ణాంకం kx, స్థిరరాశి conv2d_dw_params & p) {
        తిరిగి (ky * p.kernel_w + kx) * p.channels + c;
    }
    స్థిర పూర్ణాంకం output_సూచీ(పూర్ణాంకం n, పూర్ణాంకం c, పూర్ణాంకం y, పూర్ణాంకం x, స్థిరరాశి conv2d_dw_params & p) {
        తిరిగి n * (p.channels * p.బహిర్గతం_w * p.బహిర్గతం_h) + y * (p.బహిర్గతం_w * p.channels) + x * p.channels + c;
    }
    స్థిర శూన్య unpack_indices(పూర్ణాంకం global_idx, స్థిరరాశి conv2d_dw_params & p,
                               పూర్ణాంకం & n, పూర్ణాంకం & c, పూర్ణాంకం & బహిర్గతం_y, పూర్ణాంకం & బహిర్గతం_x) {
        c      = global_idx % p.channels;
        బహిర్గతం_x  = (global_idx / p.channels) % p.బహిర్గతం_w;
        బహిర్గతం_y  = (global_idx / (p.channels * p.బహిర్గతం_w)) % p.బహిర్గతం_h;
        n      = global_idx / (p.channels * p.బహిర్గతం_w * p.బహిర్గతం_h);
    }
};

మూస <రకం_పేరు KernelT, రకం_పేరు Layout>
స్థిర శూన్య conv2d_dw_kernel(స్థిరరాశి దశాంశం * ఆదానం, స్థిరరాశి KernelT * kernel, దశాంశం * output,
                             స్థిరరాశి conv2d_dw_params p, స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం global_idx     = అంశం_ct1.get_local_id(2) +
                               అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
    స్థిరరాశి పూర్ణాంకం మొత్తం_elements = p.batches * p.channels * p.బహిర్గతం_h * p.బహిర్గతం_w;

    ఒకవేళ (global_idx >= మొత్తం_elements) {
        తిరిగి;
    }

    పూర్ణాంకం n, c, బహిర్గతం_y, బహిర్గతం_x;
    Layout::unpack_indices(global_idx, p, n, c, బహిర్గతం_y, బహిర్గతం_x);

    దశాంశం acc = 0.0f;
    స్థిరరాశి conv2d_dw_kernel_bounds bounds = dw_calculate_kernel_bounds(బహిర్గతం_x, బహిర్గతం_y, p);

    కొరకు (పూర్ణాంకం ky = bounds.y_min; ky < bounds.y_max; ++ky) {
        స్థిరరాశి పూర్ణాంకం లోపల_y = dw_calculate_ఆదానం_coord(బహిర్గతం_y, ky, p.stride_y, p.dilation_y, p.padding_y);
        కొరకు (పూర్ణాంకం kx = bounds.x_min; kx < bounds.x_max; ++kx) {
            స్థిరరాశి పూర్ణాంకం లోపల_x = dw_calculate_ఆదానం_coord(బహిర్గతం_x, kx, p.stride_x, p.dilation_x, p.padding_x);
            acc += ఆదానం[Layout::ఆదానం_సూచీ(n, c, లోపల_y, లోపల_x, p)] *
                   static_cast<దశాంశం>(kernel[Layout::kernel_సూచీ(c, ky, kx, p)]);
        }
    }

    output[Layout::output_సూచీ(n, c, బహిర్గతం_y, బహిర్గతం_x, p)] = acc;
}

మూస <రకం_పేరు KernelT, రకం_పేరు Layout>
స్థిర శూన్య conv2d_dw_sycl(స్థిరరాశి దశాంశం * x_d, స్థిరరాశి KernelT * w_d, దశాంశం * y_d,
                            స్థిరరాశి conv2d_dw_params p, స్థిరరాశి వరుస_ptr & stream) {
    స్థిరరాశి పూర్ణాంకం మొత్తం      = p.batches * p.channels * p.బహిర్గతం_h * p.బహిర్గతం_w;
    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = (మొత్తం + SYCL_CONV2_D_DW_ఖండం_కొలత - 1) / SYCL_CONV2_D_DW_ఖండం_కొలత;
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, SYCL_CONV2_D_DW_ఖండం_కొలత);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, అంకె_ఖండాలు);
    stream->parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            conv2d_dw_kernel<KernelT, Layout>(x_d, w_d, y_d, p, అంశం_ct1);
        });
}

శూన్య ggml_sycl_op_conv2d_dw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    స్థిరరాశి ggml_tensor * kernel = dst->src[0];
    స్థిరరాశి ggml_tensor * ఆదానం  = dst->src[1];

    GGML_ASSERT((kernel->type == GGML_TYPE_F32 || kernel->type == GGML_TYPE_F16) &&
                ఆదానం->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    స్థిరరాశి దశాంశం * x_d = (స్థిరరాశి దశాంశం *) ఆదానం->దత్తాంశం;
    దశాంశం *       y_d = (దశాంశం *) dst->దత్తాంశం;

    స్థిరరాశి int32_t * p          = (స్థిరరాశి int32_t *) dst->op_params;
    స్థిరరాశి పూర్ణాంకం       stride_x   = p[0];
    స్థిరరాశి పూర్ణాంకం       stride_y   = p[1];
    స్థిరరాశి పూర్ణాంకం       padding_x  = p[2];
    స్థిరరాశి పూర్ణాంకం       padding_y  = p[3];
    స్థిరరాశి పూర్ణాంకం       dilation_x = p[4];
    స్థిరరాశి పూర్ణాంకం       dilation_y = p[5];

    స్థిరరాశి పూర్ణాంకం లోపల_w     = ఆదానం->ne[0];
    స్థిరరాశి పూర్ణాంకం లోపల_h     = ఆదానం->ne[1];
    స్థిరరాశి పూర్ణాంకం kernel_w = kernel->ne[0];
    స్థిరరాశి పూర్ణాంకం kernel_h = kernel->ne[1];
    స్థిరరాశి పూర్ణాంకం బహిర్గతం_w    = dst->ne[0];
    స్థిరరాశి పూర్ణాంకం బహిర్గతం_h    = dst->ne[1];
    స్థిరరాశి పూర్ణాంకం channels = dst->ne[2];
    స్థిరరాశి పూర్ణాంకం batches  = dst->ne[3];

    స్థిరరాశి conv2d_dw_params params = { లోపల_w, లోపల_h, బహిర్గతం_w, బహిర్గతం_h, kernel_w, kernel_h,
                                      stride_x, stride_y, padding_x, padding_y,
                                      dilation_x, dilation_y, channels, batches };

    స్థిరరాశి వరుస_ptr stream = ctx.stream();

    ఒకవేళ (kernel->type == GGML_TYPE_F16) {
        స్థిరరాశి sycl::half * w_d = (స్థిరరాశి sycl::half *) kernel->దత్తాంశం;
        ఒకవేళ (ggml_is_contiguous(ఆదానం)) {
            conv2d_dw_sycl<sycl::half, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } లేకపోతే ఒకవేళ (ggml_is_contiguous_channels(ఆదానం)) {
            conv2d_dw_sycl<sycl::half, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } లేకపోతే {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    } లేకపోతే {
        స్థిరరాశి దశాంశం * w_d = (స్థిరరాశి దశాంశం *) kernel->దత్తాంశం;
        ఒకవేళ (ggml_is_contiguous(ఆదానం)) {
            conv2d_dw_sycl<దశాంశం, dw_whcn_layout>(x_d, w_d, y_d, params, stream);
        } లేకపోతే ఒకవేళ (ggml_is_contiguous_channels(ఆదానం)) {
            conv2d_dw_sycl<దశాంశం, dw_cwhn_layout>(x_d, w_d, y_d, params, stream);
        } లేకపోతే {
            GGML_ABORT("Unsupported memory layout for conv2d_dw");
        }
    }
}
