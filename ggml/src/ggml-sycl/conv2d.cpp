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
    ثابت int64_t ټول_3;
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

ساكن مضمن conv2d_kernel_bounds calculate_kernel_bounds(int64_t بهر_x, int64_t بهر_y, ثابت conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - بهر_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - بهر_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - بهر_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - بهر_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    أرجع bounds;
}

ساكن مضمن صحيح calculate_ننوت_coord(int64_t بهر_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    أرجع بهر_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
ساكن مضمن int64_t whcn_ننوت_شاخص(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    أرجع n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

ساكن مضمن int64_t whcn_kernel_شاخص(int64_t c_بهر, int64_t c_دننه, int64_t ky, int64_t kx, ثابت conv2d_params & P) {
    أرجع c_بهر * (P.IC * P.KH * P.KW) + c_دننه * (P.KH * P.KW) + ky * P.KW + kx;
}

ساكن مضمن int64_t whcn_output_شاخص(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    أرجع n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
ساكن مضمن int64_t cwhn_ننوت_شاخص(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    أرجع ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

ساكن مضمن int64_t cwhn_kernel_شاخص(int64_t c_بهر, int64_t c_دننه, int64_t ky, int64_t kx, ثابت conv2d_params & P) {
    أرجع ((ky * P.KW + kx) * P.IC + c_دننه) * P.OC + c_بهر;
}

کينډۍ <ډول_نوم T, منطقي CWHN>
ساكن فراغ conv2d_kernel(ثابت عائم * ننوت, ثابت T * kernel, عائم * output,
                          ثابت conv2d_params P, ثابت sycl::nd_توکی<3> & توکی_ct1) {
    ثابت int64_t global_idx = توکی_ct1.get_local_id(2) +
                               توکی_ct1.get_ډله(2) * توکی_ct1.get_local_range(2);

    إذا (global_idx >= P.ټول_3) {
        أرجع;
    }

    ثابت int64_t بهر_x  = global_idx % P.OW;
    ثابت int64_t بهر_y  = (global_idx / P.OW) % P.OH;
    ثابت int64_t c_بهر  = (global_idx / (P.OW * P.OH)) % P.OC;
    ثابت int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    عائم acc = 0.0f;

    ثابت conv2d_kernel_bounds bounds = calculate_kernel_bounds(بهر_x, بهر_y, P);

    لكل (int64_t c_دننه = 0; c_دننه < P.IC; ++c_دننه) {
        لكل (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            ثابت int64_t دننه_y = calculate_ننوت_coord(بهر_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            لكل (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                ثابت int64_t دننه_x = calculate_ننوت_coord(بهر_x, kx, P.ST_X, P.DL_X, P.PD_X);
                ثابت عائم ننوت_val  = ننوت[CWHN ? cwhn_ننوت_شاخص(n, c_دننه, دننه_y, دننه_x, P) : whcn_ننوت_شاخص(n, c_دننه, دننه_y, دننه_x, P)];
                ثابت T     kernel_val = kernel[CWHN ? cwhn_kernel_شاخص(c_بهر, c_دننه, ky, kx, P) : whcn_kernel_شاخص(c_بهر, c_دننه, ky, kx, P)];
                acc += ننوت_val * ggml_sycl_cast<عائم>(kernel_val);
            }
        }
    }

    output[whcn_output_شاخص(n, c_بهر, بهر_y, بهر_x, P)] = acc;
}

کينډۍ <ډول_نوم T, منطقي CWHN>
ساكن فراغ conv2d_sycl(ثابت عائم * X_D, ثابت T * K_D, عائم * Y_D,
                        ثابت conv2d_params P, ثابت کتار_ptr & stream) {
    ثابت صحيح عدد_بلاکونه = (P.ټول_3 + SYCL_CONV2_D_بلاک_اندازه - 1) / SYCL_CONV2_D_بلاک_اندازه;
    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_CONV2_D_بلاک_اندازه);
    ثابت sycl::range<3> بلاک_nums(1, 1, عدد_بلاکونه);
    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_توکی<3> توکی_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, توکی_ct1);
        });
}

فراغ ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * ننوت  = dst->src[1];
    ثابت عائم *       K_D    = (ثابت عائم *) kernel->معلومات;
    ثابت عائم *       X_D    = (ثابت عائم *) ننوت->معلومات;
    عائم *             Y_D    = (عائم *) dst->معلومات;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    ثابت منطقي whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(ننوت);
    ثابت منطقي cwhn = ننوت->nb[2]  == ggml_type_اندازه(ننوت->type) &&
                      kernel->nb[3] == ggml_type_اندازه(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    ثابت منطقي cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(ننوت->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(ننوت->ne[2] == kernel->ne[2]);

    ثابت کتار_ptr stream = ctx.stream();

    ثابت int32_t * p    = (ثابت int32_t *) dst->op_params;
    ثابت صحيح       ST_X = p[0];
    ثابت صحيح       ST_Y = p[1];
    ثابت صحيح       PD_X = p[2];
    ثابت صحيح       PD_Y = p[3];
    ثابت صحيح       DL_X = p[4];
    ثابت صحيح       DL_Y = p[5];

    ثابت صحيح IW = ننوت->ne[0];
    ثابت صحيح IH = ننوت->ne[1];
    ثابت صحيح OW = dst->ne[0];
    ثابت صحيح OH = dst->ne[1];
    ثابت صحيح KW = kernel->ne[0];
    ثابت صحيح KH = kernel->ne[1];
    ثابت صحيح IC = ننوت->ne[2];
    ثابت صحيح OC = kernel->ne[3];
    ثابت صحيح B  = ننوت->ne[3];

    ثابت int64_t     ټول  = (int64_t) B * OC * OH * OW;
    ثابت conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, ټول };

    إذا (cwhn_layout) {
        إذا (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, ریښتیا>(X_D, (ثابت sycl::half *) K_D, Y_D, params, stream);
        } وإلا {
            conv2d_sycl<عائم, ریښتیا>(X_D, K_D, Y_D, params, stream);
        }
    } وإلا {
        إذا (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, خطأ>(X_D, (ثابت sycl::half *) K_D, Y_D, params, stream);
        } وإلا {
            conv2d_sycl<عائم, خطأ>(X_D, K_D, Y_D, params, stream);
        }
    }
}
