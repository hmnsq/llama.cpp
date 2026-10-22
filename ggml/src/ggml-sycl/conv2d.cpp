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
    ثابت int64_t ڪل_3;
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

ساكن مضمن conv2d_kernel_bounds calculate_kernel_bounds(int64_t ٻاهريون_x, int64_t ٻاهريون_y, ثابت conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - ٻاهريون_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - ٻاهريون_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - ٻاهريون_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - ٻاهريون_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    أرجع bounds;
}

ساكن مضمن صحيح calculate_داخلا_coord(int64_t ٻاهريون_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    أرجع ٻاهريون_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
ساكن مضمن int64_t whcn_داخلا_اشاريو(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    أرجع n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

ساكن مضمن int64_t whcn_kernel_اشاريو(int64_t c_ٻاهريون, int64_t c_اندريون, int64_t ky, int64_t kx, ثابت conv2d_params & P) {
    أرجع c_ٻاهريون * (P.IC * P.KH * P.KW) + c_اندريون * (P.KH * P.KW) + ky * P.KW + kx;
}

ساكن مضمن int64_t whcn_output_اشاريو(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    أرجع n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
ساكن مضمن int64_t cwhn_داخلا_اشاريو(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    أرجع ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

ساكن مضمن int64_t cwhn_kernel_اشاريو(int64_t c_ٻاهريون, int64_t c_اندريون, int64_t ky, int64_t kx, ثابت conv2d_params & P) {
    أرجع ((ky * P.KW + kx) * P.IC + c_اندريون) * P.OC + c_ٻاهريون;
}

سانچو <قسم_نالو T, منطقي CWHN>
ساكن فراغ conv2d_kernel(ثابت عائم * داخلا_2, ثابت T * kernel, عائم * output,
                          ثابت conv2d_params P, ثابت sycl::nd_شي<3> & شي_ct1) {
    ثابت int64_t global_idx = شي_ct1.get_local_id(2) +
                               شي_ct1.get_گروهه(2) * شي_ct1.get_local_range(2);

    إذا (global_idx >= P.ڪل_3) {
        أرجع;
    }

    ثابت int64_t ٻاهريون_x  = global_idx % P.OW;
    ثابت int64_t ٻاهريون_y  = (global_idx / P.OW) % P.OH;
    ثابت int64_t c_ٻاهريون  = (global_idx / (P.OW * P.OH)) % P.OC;
    ثابت int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    عائم acc = 0.0f;

    ثابت conv2d_kernel_bounds bounds = calculate_kernel_bounds(ٻاهريون_x, ٻاهريون_y, P);

    لكل (int64_t c_اندريون = 0; c_اندريون < P.IC; ++c_اندريون) {
        لكل (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            ثابت int64_t اندريون_y = calculate_داخلا_coord(ٻاهريون_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            لكل (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                ثابت int64_t اندريون_x = calculate_داخلا_coord(ٻاهريون_x, kx, P.ST_X, P.DL_X, P.PD_X);
                ثابت عائم داخلا_val  = داخلا_2[CWHN ? cwhn_داخلا_اشاريو(n, c_اندريون, اندريون_y, اندريون_x, P) : whcn_داخلا_اشاريو(n, c_اندريون, اندريون_y, اندريون_x, P)];
                ثابت T     kernel_val = kernel[CWHN ? cwhn_kernel_اشاريو(c_ٻاهريون, c_اندريون, ky, kx, P) : whcn_kernel_اشاريو(c_ٻاهريون, c_اندريون, ky, kx, P)];
                acc += داخلا_val * ggml_sycl_cast<عائم>(kernel_val);
            }
        }
    }

    output[whcn_output_اشاريو(n, c_ٻاهريون, ٻاهريون_y, ٻاهريون_x, P)] = acc;
}

سانچو <قسم_نالو T, منطقي CWHN>
ساكن فراغ conv2d_sycl(ثابت عائم * X_D, ثابت T * K_D, عائم * Y_D,
                        ثابت conv2d_params P, ثابت قطار_ptr & stream) {
    ثابت صحيح انگ_بلاڪون = (P.ڪل_3 + SYCL_CONV2_D_بلاڪ_سائيز - 1) / SYCL_CONV2_D_بلاڪ_سائيز;
    ثابت sycl::range<3> بلاڪ_dims(1, 1, SYCL_CONV2_D_بلاڪ_سائيز);
    ثابت sycl::range<3> بلاڪ_nums(1, 1, انگ_بلاڪون);
    stream->parallel_for(sycl::nd_range<3>(بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
        [=](sycl::nd_شي<3> شي_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, شي_ct1);
        });
}

فراغ ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * داخلا_2  = dst->src[1];
    ثابت عائم *       K_D    = (ثابت عائم *) kernel->معلومات;
    ثابت عائم *       X_D    = (ثابت عائم *) داخلا_2->معلومات;
    عائم *             Y_D    = (عائم *) dst->معلومات;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    ثابت منطقي whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(داخلا_2);
    ثابت منطقي cwhn = داخلا_2->nb[2]  == ggml_type_سائيز(داخلا_2->type) &&
                      kernel->nb[3] == ggml_type_سائيز(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    ثابت منطقي cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(داخلا_2->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(داخلا_2->ne[2] == kernel->ne[2]);

    ثابت قطار_ptr stream = ctx.stream();

    ثابت int32_t * p    = (ثابت int32_t *) dst->op_params;
    ثابت صحيح       ST_X = p[0];
    ثابت صحيح       ST_Y = p[1];
    ثابت صحيح       PD_X = p[2];
    ثابت صحيح       PD_Y = p[3];
    ثابت صحيح       DL_X = p[4];
    ثابت صحيح       DL_Y = p[5];

    ثابت صحيح IW = داخلا_2->ne[0];
    ثابت صحيح IH = داخلا_2->ne[1];
    ثابت صحيح OW = dst->ne[0];
    ثابت صحيح OH = dst->ne[1];
    ثابت صحيح KW = kernel->ne[0];
    ثابت صحيح KH = kernel->ne[1];
    ثابت صحيح IC = داخلا_2->ne[2];
    ثابت صحيح OC = kernel->ne[3];
    ثابت صحيح B  = داخلا_2->ne[3];

    ثابت int64_t     ڪل  = (int64_t) B * OC * OH * OW;
    ثابت conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, ڪل };

    إذا (cwhn_layout) {
        إذا (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, سچ>(X_D, (ثابت sycl::half *) K_D, Y_D, params, stream);
        } وإلا {
            conv2d_sycl<عائم, سچ>(X_D, K_D, Y_D, params, stream);
        }
    } وإلا {
        إذا (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, خطأ>(X_D, (ثابت sycl::half *) K_D, Y_D, params, stream);
        } وإلا {
            conv2d_sycl<عائم, خطأ>(X_D, K_D, Y_D, params, stream);
        }
    }
}
