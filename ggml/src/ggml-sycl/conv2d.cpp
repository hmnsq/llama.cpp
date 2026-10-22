#అన్తర్భావ "conv2d.hpp"
#అన్తర్భావ "convert.hpp"

నిర్మాణం conv2d_params {
    స్థిరరాశి int64_t IW, IH;
    స్థిరరాశి int64_t OW, OH;
    స్థిరరాశి int64_t KW, KH;
    స్థిరరాశి int64_t ST_X, ST_Y;
    స్థిరరాశి int64_t PD_X, PD_Y;
    స్థిరరాశి int64_t DL_X, DL_Y;
    స్థిరరాశి int64_t IC, OC;
    స్థిరరాశి int64_t B;
    స్థిరరాశి int64_t మొత్తం_3;
};

నిర్మాణం conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

స్థిర అంతఃవరుస int64_t conv2d_max64(int64_t a, int64_t b) {
    తిరిగి (a > b) ? a : b;
}

స్థిర అంతఃవరుస int64_t conv2d_min64(int64_t a, int64_t b) {
    తిరిగి (a < b) ? a : b;
}

స్థిర అంతఃవరుస conv2d_kernel_bounds calculate_kernel_bounds(int64_t బహిర్గతం_x, int64_t బహిర్గతం_y, స్థిరరాశి conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - బహిర్గతం_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - బహిర్గతం_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - బహిర్గతం_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - బహిర్గతం_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    తిరిగి bounds;
}

స్థిర అంతఃవరుస పూర్ణాంకం calculate_ఆదానం_coord(int64_t బహిర్గతం_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    తిరిగి బహిర్గతం_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
స్థిర అంతఃవరుస int64_t whcn_ఆదానం_సూచీ(int64_t n, int64_t c, int64_t y, int64_t x, స్థిరరాశి conv2d_params & P) {
    తిరిగి n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

స్థిర అంతఃవరుస int64_t whcn_kernel_సూచీ(int64_t c_బహిర్గతం, int64_t c_లోపల, int64_t ky, int64_t kx, స్థిరరాశి conv2d_params & P) {
    తిరిగి c_బహిర్గతం * (P.IC * P.KH * P.KW) + c_లోపల * (P.KH * P.KW) + ky * P.KW + kx;
}

స్థిర అంతఃవరుస int64_t whcn_output_సూచీ(int64_t n, int64_t c, int64_t y, int64_t x, స్థిరరాశి conv2d_params & P) {
    తిరిగి n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
స్థిర అంతఃవరుస int64_t cwhn_ఆదానం_సూచీ(int64_t n, int64_t c, int64_t y, int64_t x, స్థిరరాశి conv2d_params & P) {
    తిరిగి ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

స్థిర అంతఃవరుస int64_t cwhn_kernel_సూచీ(int64_t c_బహిర్గతం, int64_t c_లోపల, int64_t ky, int64_t kx, స్థిరరాశి conv2d_params & P) {
    తిరిగి ((ky * P.KW + kx) * P.IC + c_లోపల) * P.OC + c_బహిర్గతం;
}

మూస <రకం_పేరు T, తార్కిక CWHN>
స్థిర శూన్య conv2d_kernel(స్థిరరాశి దశాంశం * ఆదానం, స్థిరరాశి T * kernel, దశాంశం * output,
                          స్థిరరాశి conv2d_params P, స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {
    స్థిరరాశి int64_t global_idx = అంశం_ct1.get_local_id(2) +
                               అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);

    ఒకవేళ (global_idx >= P.మొత్తం_3) {
        తిరిగి;
    }

    స్థిరరాశి int64_t బహిర్గతం_x  = global_idx % P.OW;
    స్థిరరాశి int64_t బహిర్గతం_y  = (global_idx / P.OW) % P.OH;
    స్థిరరాశి int64_t c_బహిర్గతం  = (global_idx / (P.OW * P.OH)) % P.OC;
    స్థిరరాశి int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    దశాంశం acc = 0.0f;

    స్థిరరాశి conv2d_kernel_bounds bounds = calculate_kernel_bounds(బహిర్గతం_x, బహిర్గతం_y, P);

    కొరకు (int64_t c_లోపల = 0; c_లోపల < P.IC; ++c_లోపల) {
        కొరకు (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            స్థిరరాశి int64_t లోపల_y = calculate_ఆదానం_coord(బహిర్గతం_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            కొరకు (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                స్థిరరాశి int64_t లోపల_x = calculate_ఆదానం_coord(బహిర్గతం_x, kx, P.ST_X, P.DL_X, P.PD_X);
                స్థిరరాశి దశాంశం ఆదానం_val  = ఆదానం[CWHN ? cwhn_ఆదానం_సూచీ(n, c_లోపల, లోపల_y, లోపల_x, P) : whcn_ఆదానం_సూచీ(n, c_లోపల, లోపల_y, లోపల_x, P)];
                స్థిరరాశి T     kernel_val = kernel[CWHN ? cwhn_kernel_సూచీ(c_బహిర్గతం, c_లోపల, ky, kx, P) : whcn_kernel_సూచీ(c_బహిర్గతం, c_లోపల, ky, kx, P)];
                acc += ఆదానం_val * ggml_sycl_cast<దశాంశం>(kernel_val);
            }
        }
    }

    output[whcn_output_సూచీ(n, c_బహిర్గతం, బహిర్గతం_y, బహిర్గతం_x, P)] = acc;
}

మూస <రకం_పేరు T, తార్కిక CWHN>
స్థిర శూన్య conv2d_sycl(స్థిరరాశి దశాంశం * X_D, స్థిరరాశి T * K_D, దశాంశం * Y_D,
                        స్థిరరాశి conv2d_params P, స్థిరరాశి వరుస_ptr & stream) {
    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = (P.మొత్తం_3 + SYCL_CONV2_D_ఖండం_కొలత - 1) / SYCL_CONV2_D_ఖండం_కొలత;
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, SYCL_CONV2_D_ఖండం_కొలత);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, అంకె_ఖండాలు);
    stream->parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, అంశం_ct1);
        });
}

శూన్య ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    స్థిరరాశి ggml_tensor * kernel = dst->src[0];
    స్థిరరాశి ggml_tensor * ఆదానం  = dst->src[1];
    స్థిరరాశి దశాంశం *       K_D    = (స్థిరరాశి దశాంశం *) kernel->దత్తాంశం;
    స్థిరరాశి దశాంశం *       X_D    = (స్థిరరాశి దశాంశం *) ఆదానం->దత్తాంశం;
    దశాంశం *             Y_D    = (దశాంశం *) dst->దత్తాంశం;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    స్థిరరాశి తార్కిక whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(ఆదానం);
    స్థిరరాశి తార్కిక cwhn = ఆదానం->nb[2]  == ggml_type_కొలత(ఆదానం->type) &&
                      kernel->nb[3] == ggml_type_కొలత(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    స్థిరరాశి తార్కిక cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(ఆదానం->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(ఆదానం->ne[2] == kernel->ne[2]);

    స్థిరరాశి వరుస_ptr stream = ctx.stream();

    స్థిరరాశి int32_t * p    = (స్థిరరాశి int32_t *) dst->op_params;
    స్థిరరాశి పూర్ణాంకం       ST_X = p[0];
    స్థిరరాశి పూర్ణాంకం       ST_Y = p[1];
    స్థిరరాశి పూర్ణాంకం       PD_X = p[2];
    స్థిరరాశి పూర్ణాంకం       PD_Y = p[3];
    స్థిరరాశి పూర్ణాంకం       DL_X = p[4];
    స్థిరరాశి పూర్ణాంకం       DL_Y = p[5];

    స్థిరరాశి పూర్ణాంకం IW = ఆదానం->ne[0];
    స్థిరరాశి పూర్ణాంకం IH = ఆదానం->ne[1];
    స్థిరరాశి పూర్ణాంకం OW = dst->ne[0];
    స్థిరరాశి పూర్ణాంకం OH = dst->ne[1];
    స్థిరరాశి పూర్ణాంకం KW = kernel->ne[0];
    స్థిరరాశి పూర్ణాంకం KH = kernel->ne[1];
    స్థిరరాశి పూర్ణాంకం IC = ఆదానం->ne[2];
    స్థిరరాశి పూర్ణాంకం OC = kernel->ne[3];
    స్థిరరాశి పూర్ణాంకం B  = ఆదానం->ne[3];

    స్థిరరాశి int64_t     మొత్తం  = (int64_t) B * OC * OH * OW;
    స్థిరరాశి conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, మొత్తం };

    ఒకవేళ (cwhn_layout) {
        ఒకవేళ (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, నిజం>(X_D, (స్థిరరాశి sycl::half *) K_D, Y_D, params, stream);
        } లేకపోతే {
            conv2d_sycl<దశాంశం, నిజం>(X_D, K_D, Y_D, params, stream);
        }
    } లేకపోతే {
        ఒకవేళ (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, అబద్ధం>(X_D, (స్థిరరాశి sycl::half *) K_D, Y_D, params, stream);
        } లేకపోతే {
            conv2d_sycl<దశాంశం, అబద్ధం>(X_D, K_D, Y_D, params, stream);
        }
    }
}
