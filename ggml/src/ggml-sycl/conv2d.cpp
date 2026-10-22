#ਅਨ੍ਤਰ੍ਭਾਵ "conv2d.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "convert.hpp"

ਬਣਤਰ conv2d_params {
    ਅਚਲ int64_t IW, IH;
    ਅਚਲ int64_t OW, OH;
    ਅਚਲ int64_t KW, KH;
    ਅਚਲ int64_t ST_X, ST_Y;
    ਅਚਲ int64_t PD_X, PD_Y;
    ਅਚਲ int64_t DL_X, DL_Y;
    ਅਚਲ int64_t IC, OC;
    ਅਚਲ int64_t B;
    ਅਚਲ int64_t ਕੁੱਲ_3;
};

ਬਣਤਰ conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ int64_t conv2d_max64(int64_t a, int64_t b) {
    ਵਾਪਸ (a > b) ? a : b;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ int64_t conv2d_min64(int64_t a, int64_t b) {
    ਵਾਪਸ (a < b) ? a : b;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ conv2d_kernel_bounds calculate_kernel_bounds(int64_t ਬਾਹਰਲਾ_x, int64_t ਬਾਹਰਲਾ_y, ਅਚਲ conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - ਬਾਹਰਲਾ_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - ਬਾਹਰਲਾ_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - ਬਾਹਰਲਾ_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - ਬਾਹਰਲਾ_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    ਵਾਪਸ bounds;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ calculate_ਨਿਵੇਸ਼_coord(int64_t ਬਾਹਰਲਾ_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    ਵਾਪਸ ਬਾਹਰਲਾ_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
ਸਥਿਰ ਸਤਰਅੰਦਰ int64_t whcn_ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ(int64_t n, int64_t c, int64_t y, int64_t x, ਅਚਲ conv2d_params & P) {
    ਵਾਪਸ n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ int64_t whcn_kernel_ਸੂਚਕਾਂਕ(int64_t c_ਬਾਹਰਲਾ, int64_t c_ਵਿੱਚ, int64_t ky, int64_t kx, ਅਚਲ conv2d_params & P) {
    ਵਾਪਸ c_ਬਾਹਰਲਾ * (P.IC * P.KH * P.KW) + c_ਵਿੱਚ * (P.KH * P.KW) + ky * P.KW + kx;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ int64_t whcn_output_ਸੂਚਕਾਂਕ(int64_t n, int64_t c, int64_t y, int64_t x, ਅਚਲ conv2d_params & P) {
    ਵਾਪਸ n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
ਸਥਿਰ ਸਤਰਅੰਦਰ int64_t cwhn_ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ(int64_t n, int64_t c, int64_t y, int64_t x, ਅਚਲ conv2d_params & P) {
    ਵਾਪਸ ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ int64_t cwhn_kernel_ਸੂਚਕਾਂਕ(int64_t c_ਬਾਹਰਲਾ, int64_t c_ਵਿੱਚ, int64_t ky, int64_t kx, ਅਚਲ conv2d_params & P) {
    ਵਾਪਸ ((ky * P.KW + kx) * P.IC + c_ਵਿੱਚ) * P.OC + c_ਬਾਹਰਲਾ;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਤਾਰਕਿਕ CWHN>
ਸਥਿਰ ਖਾਲੀ conv2d_kernel(ਅਚਲ ਦਸ਼ਮਲਵ * ਨਿਵੇਸ਼, ਅਚਲ T * kernel, ਦਸ਼ਮਲਵ * output,
                          ਅਚਲ conv2d_params P, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਅਚਲ int64_t global_idx = ਵਸਤੂ_ct1.get_local_id(2) +
                               ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);

    ਜੇ (global_idx >= P.ਕੁੱਲ_3) {
        ਵਾਪਸ;
    }

    ਅਚਲ int64_t ਬਾਹਰਲਾ_x  = global_idx % P.OW;
    ਅਚਲ int64_t ਬਾਹਰਲਾ_y  = (global_idx / P.OW) % P.OH;
    ਅਚਲ int64_t c_ਬਾਹਰਲਾ  = (global_idx / (P.OW * P.OH)) % P.OC;
    ਅਚਲ int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    ਦਸ਼ਮਲਵ acc = 0.0f;

    ਅਚਲ conv2d_kernel_bounds bounds = calculate_kernel_bounds(ਬਾਹਰਲਾ_x, ਬਾਹਰਲਾ_y, P);

    ਲਈ (int64_t c_ਵਿੱਚ = 0; c_ਵਿੱਚ < P.IC; ++c_ਵਿੱਚ) {
        ਲਈ (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            ਅਚਲ int64_t ਵਿੱਚ_y = calculate_ਨਿਵੇਸ਼_coord(ਬਾਹਰਲਾ_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            ਲਈ (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                ਅਚਲ int64_t ਵਿੱਚ_x = calculate_ਨਿਵੇਸ਼_coord(ਬਾਹਰਲਾ_x, kx, P.ST_X, P.DL_X, P.PD_X);
                ਅਚਲ ਦਸ਼ਮਲਵ ਨਿਵੇਸ਼_val  = ਨਿਵੇਸ਼[CWHN ? cwhn_ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ(n, c_ਵਿੱਚ, ਵਿੱਚ_y, ਵਿੱਚ_x, P) : whcn_ਨਿਵੇਸ਼_ਸੂਚਕਾਂਕ(n, c_ਵਿੱਚ, ਵਿੱਚ_y, ਵਿੱਚ_x, P)];
                ਅਚਲ T     kernel_val = kernel[CWHN ? cwhn_kernel_ਸੂਚਕਾਂਕ(c_ਬਾਹਰਲਾ, c_ਵਿੱਚ, ky, kx, P) : whcn_kernel_ਸੂਚਕਾਂਕ(c_ਬਾਹਰਲਾ, c_ਵਿੱਚ, ky, kx, P)];
                acc += ਨਿਵੇਸ਼_val * ggml_sycl_cast<ਦਸ਼ਮਲਵ>(kernel_val);
            }
        }
    }

    output[whcn_output_ਸੂਚਕਾਂਕ(n, c_ਬਾਹਰਲਾ, ਬਾਹਰਲਾ_y, ਬਾਹਰਲਾ_x, P)] = acc;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਤਾਰਕਿਕ CWHN>
ਸਥਿਰ ਖਾਲੀ conv2d_sycl(ਅਚਲ ਦਸ਼ਮਲਵ * X_D, ਅਚਲ T * K_D, ਦਸ਼ਮਲਵ * Y_D,
                        ਅਚਲ conv2d_params P, ਅਚਲ ਕਤਾਰ_ptr & stream) {
    ਅਚਲ ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (P.ਕੁੱਲ_3 + SYCL_CONV2_D_ਖੰਡ_ਮਾਪ - 1) / SYCL_CONV2_D_ਖੰਡ_ਮਾਪ;
    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, SYCL_CONV2_D_ਖੰਡ_ਮਾਪ);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, 1, ਅੰਕ_ਖੰਡਾਂ);
    stream->parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, ਵਸਤੂ_ct1);
        });
}

ਖਾਲੀ ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ਅਚਲ ggml_tensor * kernel = dst->src[0];
    ਅਚਲ ggml_tensor * ਨਿਵੇਸ਼  = dst->src[1];
    ਅਚਲ ਦਸ਼ਮਲਵ *       K_D    = (ਅਚਲ ਦਸ਼ਮਲਵ *) kernel->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ *       X_D    = (ਅਚਲ ਦਸ਼ਮਲਵ *) ਨਿਵੇਸ਼->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *             Y_D    = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    ਅਚਲ ਤਾਰਕਿਕ whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(ਨਿਵੇਸ਼);
    ਅਚਲ ਤਾਰਕਿਕ cwhn = ਨਿਵੇਸ਼->nb[2]  == ggml_type_ਮਾਪ(ਨਿਵੇਸ਼->type) &&
                      kernel->nb[3] == ggml_type_ਮਾਪ(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    ਅਚਲ ਤਾਰਕਿਕ cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(ਨਿਵੇਸ਼->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(ਨਿਵੇਸ਼->ne[2] == kernel->ne[2]);

    ਅਚਲ ਕਤਾਰ_ptr stream = ctx.stream();

    ਅਚਲ int32_t * p    = (ਅਚਲ int32_t *) dst->op_params;
    ਅਚਲ ਪੂਰਨ       ST_X = p[0];
    ਅਚਲ ਪੂਰਨ       ST_Y = p[1];
    ਅਚਲ ਪੂਰਨ       PD_X = p[2];
    ਅਚਲ ਪੂਰਨ       PD_Y = p[3];
    ਅਚਲ ਪੂਰਨ       DL_X = p[4];
    ਅਚਲ ਪੂਰਨ       DL_Y = p[5];

    ਅਚਲ ਪੂਰਨ IW = ਨਿਵੇਸ਼->ne[0];
    ਅਚਲ ਪੂਰਨ IH = ਨਿਵੇਸ਼->ne[1];
    ਅਚਲ ਪੂਰਨ OW = dst->ne[0];
    ਅਚਲ ਪੂਰਨ OH = dst->ne[1];
    ਅਚਲ ਪੂਰਨ KW = kernel->ne[0];
    ਅਚਲ ਪੂਰਨ KH = kernel->ne[1];
    ਅਚਲ ਪੂਰਨ IC = ਨਿਵੇਸ਼->ne[2];
    ਅਚਲ ਪੂਰਨ OC = kernel->ne[3];
    ਅਚਲ ਪੂਰਨ B  = ਨਿਵੇਸ਼->ne[3];

    ਅਚਲ int64_t     ਕੁੱਲ  = (int64_t) B * OC * OH * OW;
    ਅਚਲ conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, ਕੁੱਲ };

    ਜੇ (cwhn_layout) {
        ਜੇ (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, ਸੱਚ>(X_D, (ਅਚਲ sycl::half *) K_D, Y_D, params, stream);
        } ਨਹੀਂਤਾਂ {
            conv2d_sycl<ਦਸ਼ਮਲਵ, ਸੱਚ>(X_D, K_D, Y_D, params, stream);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, ਝੂਠ>(X_D, (ਅਚਲ sycl::half *) K_D, Y_D, params, stream);
        } ਨਹੀਂਤਾਂ {
            conv2d_sycl<ਦਸ਼ਮਲਵ, ਝੂਠ>(X_D, K_D, Y_D, params, stream);
        }
    }
}
