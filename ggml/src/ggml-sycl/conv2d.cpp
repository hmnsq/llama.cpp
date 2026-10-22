#ضمن "conv2d.hpp"
#ضمن "convert.hpp"

بنية conv2d_params {
    ثابت int64_t IW, IH;
    ثابت int64_t OW, OH;
    ثابت int64_t KW, KH;
    ثابت int64_t ST_X, ST_Y;
    ثابت int64_t PD_X, PD_Y;
    ثابت int64_t DL_X, DL_Y;
    ثابت int64_t IC, OC;
    ثابت int64_t B;
    ثابت int64_t مجموع_3;
};

بنية conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

ساكن مضمن int64_t conv2d_max64(int64_t a, int64_t b) {
    أرجع (a > b) ? a : b;
}

ساكن مضمن int64_t conv2d_min64(int64_t a, int64_t b) {
    أرجع (a < b) ? a : b;
}

ساكن مضمن conv2d_kernel_bounds calculate_kernel_bounds(int64_t بیرون_x, int64_t بیرون_y, ثابت conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - بیرون_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - بیرون_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - بیرون_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - بیرون_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    أرجع bounds;
}

ساكن مضمن صحيح calculate_درون_داد_coord(int64_t بیرون_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    أرجع بیرون_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
ساكن مضمن int64_t whcn_درون_داد_شاخص(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    أرجع n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

ساكن مضمن int64_t whcn_kernel_شاخص(int64_t c_بیرون, int64_t c_درون, int64_t ky, int64_t kx, ثابت conv2d_params & P) {
    أرجع c_بیرون * (P.IC * P.KH * P.KW) + c_درون * (P.KH * P.KW) + ky * P.KW + kx;
}

ساكن مضمن int64_t whcn_output_شاخص(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    أرجع n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
ساكن مضمن int64_t cwhn_درون_داد_شاخص(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    أرجع ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

ساكن مضمن int64_t cwhn_kernel_شاخص(int64_t c_بیرون, int64_t c_درون, int64_t ky, int64_t kx, ثابت conv2d_params & P) {
    أرجع ((ky * P.KW + kx) * P.IC + c_درون) * P.OC + c_بیرون;
}

قالب <نام_نوع T, منطقي CWHN>
ساكن فراغ conv2d_kernel(ثابت عائم * درون_داد, ثابت T * kernel, عائم * output,
                          ثابت conv2d_params P, ثابت sycl::nd_قلم<3> & قلم_ct1) {
    ثابت int64_t global_idx = قلم_ct1.get_local_id(2) +
                               قلم_ct1.get_گروه(2) * قلم_ct1.get_local_range(2);

    إذا (global_idx >= P.مجموع_3) {
        أرجع;
    }

    ثابت int64_t بیرون_x  = global_idx % P.OW;
    ثابت int64_t بیرون_y  = (global_idx / P.OW) % P.OH;
    ثابت int64_t c_بیرون  = (global_idx / (P.OW * P.OH)) % P.OC;
    ثابت int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    عائم acc = 0.0f;

    ثابت conv2d_kernel_bounds bounds = calculate_kernel_bounds(بیرون_x, بیرون_y, P);

    لكل (int64_t c_درون = 0; c_درون < P.IC; ++c_درون) {
        لكل (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            ثابت int64_t درون_y = calculate_درون_داد_coord(بیرون_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            لكل (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                ثابت int64_t درون_x = calculate_درون_داد_coord(بیرون_x, kx, P.ST_X, P.DL_X, P.PD_X);
                ثابت عائم درون_داد_val  = درون_داد[CWHN ? cwhn_درون_داد_شاخص(n, c_درون, درون_y, درون_x, P) : whcn_درون_داد_شاخص(n, c_درون, درون_y, درون_x, P)];
                ثابت T     kernel_val = kernel[CWHN ? cwhn_kernel_شاخص(c_بیرون, c_درون, ky, kx, P) : whcn_kernel_شاخص(c_بیرون, c_درون, ky, kx, P)];
                acc += درون_داد_val * ggml_sycl_cast<عائم>(kernel_val);
            }
        }
    }

    output[whcn_output_شاخص(n, c_بیرون, بیرون_y, بیرون_x, P)] = acc;
}

قالب <نام_نوع T, منطقي CWHN>
ساكن فراغ conv2d_sycl(ثابت عائم * X_D, ثابت T * K_D, عائم * Y_D,
                        ثابت conv2d_params P, ثابت صف_ptr & stream) {
    ثابت صحيح رقم_بلوکها = (P.مجموع_3 + SYCL_CONV2_D_بلوک_اندازه - 1) / SYCL_CONV2_D_بلوک_اندازه;
    ثابت sycl::range<3> بلوک_dims(1, 1, SYCL_CONV2_D_بلوک_اندازه);
    ثابت sycl::range<3> بلوک_nums(1, 1, رقم_بلوکها);
    stream->parallel_for(sycl::nd_range<3>(بلوک_nums * بلوک_dims, بلوک_dims),
        [=](sycl::nd_قلم<3> قلم_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, قلم_ct1);
        });
}

فراغ ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * درون_داد  = dst->src[1];
    ثابت عائم *       K_D    = (ثابت عائم *) kernel->داده;
    ثابت عائم *       X_D    = (ثابت عائم *) درون_داد->داده;
    عائم *             Y_D    = (عائم *) dst->داده;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    ثابت منطقي whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(درون_داد);
    ثابت منطقي cwhn = درون_داد->nb[2]  == ggml_type_اندازه(درون_داد->type) &&
                      kernel->nb[3] == ggml_type_اندازه(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    ثابت منطقي cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(درون_داد->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(درون_داد->ne[2] == kernel->ne[2]);

    ثابت صف_ptr stream = ctx.stream();

    ثابت int32_t * p    = (ثابت int32_t *) dst->op_params;
    ثابت صحيح       ST_X = p[0];
    ثابت صحيح       ST_Y = p[1];
    ثابت صحيح       PD_X = p[2];
    ثابت صحيح       PD_Y = p[3];
    ثابت صحيح       DL_X = p[4];
    ثابت صحيح       DL_Y = p[5];

    ثابت صحيح IW = درون_داد->ne[0];
    ثابت صحيح IH = درون_داد->ne[1];
    ثابت صحيح OW = dst->ne[0];
    ثابت صحيح OH = dst->ne[1];
    ثابت صحيح KW = kernel->ne[0];
    ثابت صحيح KH = kernel->ne[1];
    ثابت صحيح IC = درون_داد->ne[2];
    ثابت صحيح OC = kernel->ne[3];
    ثابت صحيح B  = درون_داد->ne[3];

    ثابت int64_t     مجموع  = (int64_t) B * OC * OH * OW;
    ثابت conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, مجموع };

    إذا (cwhn_layout) {
        إذا (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, درست>(X_D, (ثابت sycl::half *) K_D, Y_D, params, stream);
        } وإلا {
            conv2d_sycl<عائم, درست>(X_D, K_D, Y_D, params, stream);
        }
    } وإلا {
        إذا (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, خطأ>(X_D, (ثابت sycl::half *) K_D, Y_D, params, stream);
        } وإلا {
            conv2d_sycl<عائم, خطأ>(X_D, K_D, Y_D, params, stream);
        }
    }
}
