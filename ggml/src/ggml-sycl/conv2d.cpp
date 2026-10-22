#ಅನ್ತರ್ಭಾವ "conv2d.hpp"
#ಅನ್ತರ್ಭಾವ "convert.hpp"

ಸಂರಚನಾ conv2d_params {
    ಧ್ರುವ int64_t IW, IH;
    ಧ್ರುವ int64_t OW, OH;
    ಧ್ರುವ int64_t KW, KH;
    ಧ್ರುವ int64_t ST_X, ST_Y;
    ಧ್ರುವ int64_t PD_X, PD_Y;
    ಧ್ರುವ int64_t DL_X, DL_Y;
    ಧ್ರುವ int64_t IC, OC;
    ಧ್ರುವ int64_t B;
    ಧ್ರುವ int64_t TOTAL;
};

ಸಂರಚನಾ conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int64_t conv2d_max64(int64_t a, int64_t b) {
    ಪ್ರತ್ಯಾಗಮ (a > b) ? a : b;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int64_t conv2d_min64(int64_t a, int64_t b) {
    ಪ್ರತ್ಯಾಗಮ (a < b) ? a : b;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ conv2d_kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, ಧ್ರುವ conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    ಪ್ರತ್ಯಾಗಮ bounds;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ calculate_ಒಳಹರಿವು_coord(int64_t out_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    ಪ್ರತ್ಯಾಗಮ out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int64_t whcn_ಒಳಹರಿವು_index(int64_t n, int64_t c, int64_t y, int64_t x, ಧ್ರುವ conv2d_params & P) {
    ಪ್ರತ್ಯಾಗಮ n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int64_t whcn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, ಧ್ರುವ conv2d_params & P) {
    ಪ್ರತ್ಯಾಗಮ c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int64_t whcn_output_index(int64_t n, int64_t c, int64_t y, int64_t x, ಧ್ರುವ conv2d_params & P) {
    ಪ್ರತ್ಯಾಗಮ n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int64_t cwhn_ಒಳಹರಿವು_index(int64_t n, int64_t c, int64_t y, int64_t x, ಧ್ರುವ conv2d_params & P) {
    ಪ್ರತ್ಯಾಗಮ ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ int64_t cwhn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, ಧ್ರುವ conv2d_params & P) {
    ಪ್ರತ್ಯಾಗಮ ((ky * P.KW + kx) * P.IC + c_in) * P.OC + c_out;
}

template <typename T, ತಾರ್ಕಿಕ CWHN>
ಸ್ಥಿರ ಶೂನ್ಯ conv2d_kernel(ಧ್ರುವ ಭಿನ್ನ * ಒಳಹರಿವು, ಧ್ರುವ T * kernel, ಭಿನ್ನ * output,
                          ಧ್ರುವ conv2d_params P, ಧ್ರುವ sycl::nd_item<3> & item_ct1) {
    ಧ್ರುವ int64_t global_idx = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);

    ಯದಿ (global_idx >= P.TOTAL) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ int64_t out_x  = global_idx % P.OW;
    ಧ್ರುವ int64_t out_y  = (global_idx / P.OW) % P.OH;
    ಧ್ರುವ int64_t c_out  = (global_idx / (P.OW * P.OH)) % P.OC;
    ಧ್ರುವ int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    ಭಿನ್ನ acc = 0.0f;

    ಧ್ರುವ conv2d_kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

    ಪ್ರತಿ (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        ಪ್ರತಿ (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            ಧ್ರುವ int64_t in_y = calculate_ಒಳಹರಿವು_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            ಪ್ರತಿ (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                ಧ್ರುವ int64_t in_x = calculate_ಒಳಹರಿವು_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);
                ಧ್ರುವ ಭಿನ್ನ ಒಳಹರಿವು_val  = ಒಳಹರಿವು[CWHN ? cwhn_ಒಳಹರಿವು_index(n, c_in, in_y, in_x, P) : whcn_ಒಳಹರಿವು_index(n, c_in, in_y, in_x, P)];
                ಧ್ರುವ T     kernel_val = kernel[CWHN ? cwhn_kernel_index(c_out, c_in, ky, kx, P) : whcn_kernel_index(c_out, c_in, ky, kx, P)];
                acc += ಒಳಹರಿವು_val * ggml_sycl_cast<ಭಿನ್ನ>(kernel_val);
            }
        }
    }

    output[whcn_output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T, ತಾರ್ಕಿಕ CWHN>
ಸ್ಥಿರ ಶೂನ್ಯ conv2d_sycl(ಧ್ರುವ ಭಿನ್ನ * X_D, ಧ್ರುವ T * K_D, ಭಿನ್ನ * Y_D,
                        ಧ್ರುವ conv2d_params P, ಧ್ರುವ queue_ptr & stream) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ಅಂಕೆ_blocks = (P.TOTAL + SYCL_CONV2D_BLOCK_SIZE - 1) / SYCL_CONV2D_BLOCK_SIZE;
    ಧ್ರುವ sycl::range<3> block_dims(1, 1, SYCL_CONV2D_BLOCK_SIZE);
    ಧ್ರುವ sycl::range<3> block_nums(1, 1, ಅಂಕೆ_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, item_ct1);
        });
}

ಶೂನ್ಯ ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ಧ್ರುವ ggml_tensor * kernel = dst->src[0];
    ಧ್ರುವ ggml_tensor * ಒಳಹರಿವು  = dst->src[1];
    ಧ್ರುವ ಭಿನ್ನ *       K_D    = (ಧ್ರುವ ಭಿನ್ನ *) kernel->data;
    ಧ್ರುವ ಭಿನ್ನ *       X_D    = (ಧ್ರುವ ಭಿನ್ನ *) ಒಳಹರಿವು->data;
    ಭಿನ್ನ *             Y_D    = (ಭಿನ್ನ *) dst->data;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    ಧ್ರುವ ತಾರ್ಕಿಕ whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(ಒಳಹರಿವು);
    ಧ್ರುವ ತಾರ್ಕಿಕ cwhn = ಒಳಹರಿವು->nb[2]  == ggml_type_size(ಒಳಹರಿವು->type) &&
                      kernel->nb[3] == ggml_type_size(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    ಧ್ರುವ ತಾರ್ಕಿಕ cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(ಒಳಹರಿವು->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(ಒಳಹರಿವು->ne[2] == kernel->ne[2]);

    ಧ್ರುವ queue_ptr stream = ctx.stream();

    ಧ್ರುವ int32_t * p    = (ಧ್ರುವ int32_t *) dst->op_params;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ST_X = p[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       ST_Y = p[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       PD_X = p[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       PD_Y = p[3];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       DL_X = p[4];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ       DL_Y = p[5];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ IW = ಒಳಹರಿವು->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ IH = ಒಳಹರಿವು->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ OW = dst->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ OH = dst->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ KW = kernel->ne[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ KH = kernel->ne[1];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ IC = ಒಳಹರಿವು->ne[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ OC = kernel->ne[3];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ B  = ಒಳಹರಿವು->ne[3];

    ಧ್ರುವ int64_t     total  = (int64_t) B * OC * OH * OW;
    ಧ್ರುವ conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    ಯದಿ (cwhn_layout) {
        ಯದಿ (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, ಸತ್ಯ>(X_D, (ಧ್ರುವ sycl::half *) K_D, Y_D, params, stream);
        } ಅನ್ಯಥಾ {
            conv2d_sycl<ಭಿನ್ನ, ಸತ್ಯ>(X_D, K_D, Y_D, params, stream);
        }
    } ಅನ್ಯಥಾ {
        ಯದಿ (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, ಅಸತ್ಯ>(X_D, (ಧ್ರುವ sycl::half *) K_D, Y_D, params, stream);
        } ಅನ್ಯಥಾ {
            conv2d_sycl<ಭಿನ್ನ, ಅಸತ್ಯ>(X_D, K_D, Y_D, params, stream);
        }
    }
}
