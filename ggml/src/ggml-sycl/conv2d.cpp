#ಅನ್ತರ್ಭಾವ "conv2d.hpp"
#ಅನ್ತರ್ಭಾವ "convert.hpp"

ರಚನೆ conv2d_params {
    ಸ್ಥಿರಾಂಕ int64_t IW, IH;
    ಸ್ಥಿರಾಂಕ int64_t OW, OH;
    ಸ್ಥಿರಾಂಕ int64_t KW, KH;
    ಸ್ಥಿರಾಂಕ int64_t ST_X, ST_Y;
    ಸ್ಥಿರಾಂಕ int64_t PD_X, PD_Y;
    ಸ್ಥಿರಾಂಕ int64_t DL_X, DL_Y;
    ಸ್ಥಿರಾಂಕ int64_t IC, OC;
    ಸ್ಥಿರಾಂಕ int64_t B;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_3;
};

ರಚನೆ conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int64_t conv2d_max64(int64_t a, int64_t b) {
    ಹಿಂತಿರುಗಿ (a > b) ? a : b;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int64_t conv2d_min64(int64_t a, int64_t b) {
    ಹಿಂತಿರುಗಿ (a < b) ? a : b;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ conv2d_kernel_bounds calculate_kernel_bounds(int64_t ಹೊರಹೋಗು_x, int64_t ಹೊರಹೋಗು_y, ಸ್ಥಿರಾಂಕ conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - ಹೊರಹೋಗು_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - ಹೊರಹೋಗು_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - ಹೊರಹೋಗು_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - ಹೊರಹೋಗು_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    ಹಿಂತಿರುಗಿ bounds;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ calculate_ಒಳಹರಿವು_coord(int64_t ಹೊರಹೋಗು_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
ಸ್ಥಿರ ಸಾಲೊಳಗೆ int64_t whcn_ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ(int64_t n, int64_t c, int64_t y, int64_t x, ಸ್ಥಿರಾಂಕ conv2d_params & P) {
    ಹಿಂತಿರುಗಿ n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int64_t whcn_kernel_ಸೂಚ್ಯಂಕ(int64_t c_ಹೊರಹೋಗು, int64_t c_ಒಳ, int64_t ky, int64_t kx, ಸ್ಥಿರಾಂಕ conv2d_params & P) {
    ಹಿಂತಿರುಗಿ c_ಹೊರಹೋಗು * (P.IC * P.KH * P.KW) + c_ಒಳ * (P.KH * P.KW) + ky * P.KW + kx;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int64_t whcn_output_ಸೂಚ್ಯಂಕ(int64_t n, int64_t c, int64_t y, int64_t x, ಸ್ಥಿರಾಂಕ conv2d_params & P) {
    ಹಿಂತಿರುಗಿ n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
ಸ್ಥಿರ ಸಾಲೊಳಗೆ int64_t cwhn_ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ(int64_t n, int64_t c, int64_t y, int64_t x, ಸ್ಥಿರಾಂಕ conv2d_params & P) {
    ಹಿಂತಿರುಗಿ ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ int64_t cwhn_kernel_ಸೂಚ್ಯಂಕ(int64_t c_ಹೊರಹೋಗು, int64_t c_ಒಳ, int64_t ky, int64_t kx, ಸ್ಥಿರಾಂಕ conv2d_params & P) {
    ಹಿಂತಿರುಗಿ ((ky * P.KW + kx) * P.IC + c_ಒಳ) * P.OC + c_ಹೊರಹೋಗು;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ತಾರ್ಕಿಕ CWHN>
ಸ್ಥಿರ ಶೂನ್ಯ conv2d_kernel(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಒಳಹರಿವು, ಸ್ಥಿರಾಂಕ T * kernel, ದಶಮಾಂಶ * output,
                          ಸ್ಥಿರಾಂಕ conv2d_params P, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ int64_t global_idx = ಅಂಶ_ct1.get_local_id(2) +
                               ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);

    ಆದರೆ (global_idx >= P.ಒಟ್ಟು_3) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int64_t ಹೊರಹೋಗು_x  = global_idx % P.OW;
    ಸ್ಥಿರಾಂಕ int64_t ಹೊರಹೋಗು_y  = (global_idx / P.OW) % P.OH;
    ಸ್ಥಿರಾಂಕ int64_t c_ಹೊರಹೋಗು  = (global_idx / (P.OW * P.OH)) % P.OC;
    ಸ್ಥಿರಾಂಕ int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    ದಶಮಾಂಶ acc = 0.0f;

    ಸ್ಥಿರಾಂಕ conv2d_kernel_bounds bounds = calculate_kernel_bounds(ಹೊರಹೋಗು_x, ಹೊರಹೋಗು_y, P);

    ಗಾಗಿ (int64_t c_ಒಳ = 0; c_ಒಳ < P.IC; ++c_ಒಳ) {
        ಗಾಗಿ (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            ಸ್ಥಿರಾಂಕ int64_t ಒಳ_y = calculate_ಒಳಹರಿವು_coord(ಹೊರಹೋಗು_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            ಗಾಗಿ (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                ಸ್ಥಿರಾಂಕ int64_t ಒಳ_x = calculate_ಒಳಹರಿವು_coord(ಹೊರಹೋಗು_x, kx, P.ST_X, P.DL_X, P.PD_X);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಒಳಹರಿವು_val  = ಒಳಹರಿವು[CWHN ? cwhn_ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ(n, c_ಒಳ, ಒಳ_y, ಒಳ_x, P) : whcn_ಒಳಹರಿವು_ಸೂಚ್ಯಂಕ(n, c_ಒಳ, ಒಳ_y, ಒಳ_x, P)];
                ಸ್ಥಿರಾಂಕ T     kernel_val = kernel[CWHN ? cwhn_kernel_ಸೂಚ್ಯಂಕ(c_ಹೊರಹೋಗು, c_ಒಳ, ky, kx, P) : whcn_kernel_ಸೂಚ್ಯಂಕ(c_ಹೊರಹೋಗು, c_ಒಳ, ky, kx, P)];
                acc += ಒಳಹರಿವು_val * ggml_sycl_cast<ದಶಮಾಂಶ>(kernel_val);
            }
        }
    }

    output[whcn_output_ಸೂಚ್ಯಂಕ(n, c_ಹೊರಹೋಗು, ಹೊರಹೋಗು_y, ಹೊರಹೋಗು_x, P)] = acc;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ತಾರ್ಕಿಕ CWHN>
ಸ್ಥಿರ ಶೂನ್ಯ conv2d_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * X_D, ಸ್ಥಿರಾಂಕ T * K_D, ದಶಮಾಂಶ * Y_D,
                        ಸ್ಥಿರಾಂಕ conv2d_params P, ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr & stream) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = (P.ಒಟ್ಟು_3 + SYCL_CONV2_D_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_CONV2_D_ಖಂಡ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, SYCL_CONV2_D_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಅಂಕೆ_ಖಂಡಗಳು);
    stream->parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, ಅಂಶ_ct1);
        });
}

ಶೂನ್ಯ ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ಸ್ಥಿರಾಂಕ ggml_tensor * kernel = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಒಳಹರಿವು  = dst->src[1];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *       K_D    = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) kernel->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *       X_D    = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಒಳಹರಿವು->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *             Y_D    = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(ಒಳಹರಿವು);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cwhn = ಒಳಹರಿವು->nb[2]  == ggml_type_ಗಾತ್ರ(ಒಳಹರಿವು->type) &&
                      kernel->nb[3] == ggml_type_ಗಾತ್ರ(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(ಒಳಹರಿವು->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(ಒಳಹರಿವು->ne[2] == kernel->ne[2]);

    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream = ctx.stream();

    ಸ್ಥಿರಾಂಕ int32_t * p    = (ಸ್ಥಿರಾಂಕ int32_t *) dst->op_params;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ST_X = p[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       ST_Y = p[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       PD_X = p[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       PD_Y = p[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       DL_X = p[4];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       DL_Y = p[5];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ IW = ಒಳಹರಿವು->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ IH = ಒಳಹರಿವು->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ OW = dst->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ OH = dst->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ KW = kernel->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ KH = kernel->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ IC = ಒಳಹರಿವು->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ OC = kernel->ne[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ B  = ಒಳಹರಿವು->ne[3];

    ಸ್ಥಿರಾಂಕ int64_t     ಒಟ್ಟು  = (int64_t) B * OC * OH * OW;
    ಸ್ಥಿರಾಂಕ conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, ಒಟ್ಟು };

    ಆದರೆ (cwhn_layout) {
        ಆದರೆ (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, ನಿಜ>(X_D, (ಸ್ಥಿರಾಂಕ sycl::half *) K_D, Y_D, params, stream);
        } ಇಲ್ಲದಿದ್ದರೆ {
            conv2d_sycl<ದಶಮಾಂಶ, ನಿಜ>(X_D, K_D, Y_D, params, stream);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, ಸುಳ್ಳು>(X_D, (ಸ್ಥಿರಾಂಕ sycl::half *) K_D, Y_D, params, stream);
        } ಇಲ್ಲದಿದ್ದರೆ {
            conv2d_sycl<ದಶಮಾಂಶ, ಸುಳ್ಳು>(X_D, K_D, Y_D, params, stream);
        }
    }
}
