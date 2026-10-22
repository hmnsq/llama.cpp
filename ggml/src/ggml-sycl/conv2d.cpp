#شامل "conv2d.hpp"
#شامل "convert.hpp"

ساخت conv2d_params {
    ثابت int64_t IW, IH;
    ثابت int64_t OW, OH;
    ثابت int64_t KW, KH;
    ثابت int64_t ST_X, ST_Y;
    ثابت int64_t PD_X, PD_Y;
    ثابت int64_t DL_X, DL_Y;
    ثابت int64_t IC, OC;
    ثابت int64_t B;
    ثابت int64_t کل_3;
};

ساخت conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

ساکن اندرونی int64_t conv2d_max64(int64_t a, int64_t b) {
    واپس (a > b) ? a : b;
}

ساکن اندرونی int64_t conv2d_min64(int64_t a, int64_t b) {
    واپس (a < b) ? a : b;
}

ساکن اندرونی conv2d_kernel_bounds calculate_kernel_bounds(int64_t برآمد_x, int64_t برآمد_y, ثابت conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - برآمد_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - برآمد_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - برآمد_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - برآمد_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    واپس bounds;
}

ساکن اندرونی صحیح calculate_داخلہ_coord(int64_t برآمد_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    واپس برآمد_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
ساکن اندرونی int64_t whcn_داخلہ_اشاریہ(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    واپس n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

ساکن اندرونی int64_t whcn_kernel_اشاریہ(int64_t c_برآمد, int64_t c_اندر, int64_t ky, int64_t kx, ثابت conv2d_params & P) {
    واپس c_برآمد * (P.IC * P.KH * P.KW) + c_اندر * (P.KH * P.KW) + ky * P.KW + kx;
}

ساکن اندرونی int64_t whcn_output_اشاریہ(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    واپس n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
ساکن اندرونی int64_t cwhn_داخلہ_اشاریہ(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    واپس ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

ساکن اندرونی int64_t cwhn_kernel_اشاریہ(int64_t c_برآمد, int64_t c_اندر, int64_t ky, int64_t kx, ثابت conv2d_params & P) {
    واپس ((ky * P.KW + kx) * P.IC + c_اندر) * P.OC + c_برآمد;
}

سانچہ <قسم_نام T, منطقی CWHN>
ساکن خالی conv2d_kernel(ثابت اعشاری * داخلہ, ثابت T * kernel, اعشاری * output,
                          ثابت conv2d_params P, ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت int64_t global_idx = شے_ct1.get_local_id(2) +
                               شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2);

    اگر (global_idx >= P.کل_3) {
        واپس;
    }

    ثابت int64_t برآمد_x  = global_idx % P.OW;
    ثابت int64_t برآمد_y  = (global_idx / P.OW) % P.OH;
    ثابت int64_t c_برآمد  = (global_idx / (P.OW * P.OH)) % P.OC;
    ثابت int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    اعشاری acc = 0.0f;

    ثابت conv2d_kernel_bounds bounds = calculate_kernel_bounds(برآمد_x, برآمد_y, P);

    برائے (int64_t c_اندر = 0; c_اندر < P.IC; ++c_اندر) {
        برائے (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            ثابت int64_t اندر_y = calculate_داخلہ_coord(برآمد_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            برائے (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                ثابت int64_t اندر_x = calculate_داخلہ_coord(برآمد_x, kx, P.ST_X, P.DL_X, P.PD_X);
                ثابت اعشاری داخلہ_val  = داخلہ[CWHN ? cwhn_داخلہ_اشاریہ(n, c_اندر, اندر_y, اندر_x, P) : whcn_داخلہ_اشاریہ(n, c_اندر, اندر_y, اندر_x, P)];
                ثابت T     kernel_val = kernel[CWHN ? cwhn_kernel_اشاریہ(c_برآمد, c_اندر, ky, kx, P) : whcn_kernel_اشاریہ(c_برآمد, c_اندر, ky, kx, P)];
                acc += داخلہ_val * ggml_sycl_cast<اعشاری>(kernel_val);
            }
        }
    }

    output[whcn_output_اشاریہ(n, c_برآمد, برآمد_y, برآمد_x, P)] = acc;
}

سانچہ <قسم_نام T, منطقی CWHN>
ساکن خالی conv2d_sycl(ثابت اعشاری * X_D, ثابت T * K_D, اعشاری * Y_D,
                        ثابت conv2d_params P, ثابت قطار_ptr & stream) {
    ثابت صحیح عدد_بلاکیں = (P.کل_3 + SYCL_CONV2_D_بلاک_سائز - 1) / SYCL_CONV2_D_بلاک_سائز;
    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_CONV2_D_بلاک_سائز);
    ثابت sycl::range<3> بلاک_nums(1, 1, عدد_بلاکیں);
    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, شے_ct1);
        });
}

خالی ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * داخلہ  = dst->src[1];
    ثابت اعشاری *       K_D    = (ثابت اعشاری *) kernel->ڈیٹا;
    ثابت اعشاری *       X_D    = (ثابت اعشاری *) داخلہ->ڈیٹا;
    اعشاری *             Y_D    = (اعشاری *) dst->ڈیٹا;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    ثابت منطقی whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(داخلہ);
    ثابت منطقی cwhn = داخلہ->nb[2]  == ggml_type_سائز(داخلہ->type) &&
                      kernel->nb[3] == ggml_type_سائز(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    ثابت منطقی cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(داخلہ->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(داخلہ->ne[2] == kernel->ne[2]);

    ثابت قطار_ptr stream = ctx.stream();

    ثابت int32_t * p    = (ثابت int32_t *) dst->op_params;
    ثابت صحیح       ST_X = p[0];
    ثابت صحیح       ST_Y = p[1];
    ثابت صحیح       PD_X = p[2];
    ثابت صحیح       PD_Y = p[3];
    ثابت صحیح       DL_X = p[4];
    ثابت صحیح       DL_Y = p[5];

    ثابت صحیح IW = داخلہ->ne[0];
    ثابت صحیح IH = داخلہ->ne[1];
    ثابت صحیح OW = dst->ne[0];
    ثابت صحیح OH = dst->ne[1];
    ثابت صحیح KW = kernel->ne[0];
    ثابت صحیح KH = kernel->ne[1];
    ثابت صحیح IC = داخلہ->ne[2];
    ثابت صحیح OC = kernel->ne[3];
    ثابت صحیح B  = داخلہ->ne[3];

    ثابت int64_t     کل  = (int64_t) B * OC * OH * OW;
    ثابت conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, کل };

    اگر (cwhn_layout) {
        اگر (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, سچ>(X_D, (ثابت sycl::half *) K_D, Y_D, params, stream);
        } ورنہ {
            conv2d_sycl<اعشاری, سچ>(X_D, K_D, Y_D, params, stream);
        }
    } ورنہ {
        اگر (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, جھوٹ>(X_D, (ثابت sycl::half *) K_D, Y_D, params, stream);
        } ورنہ {
            conv2d_sycl<اعشاری, جھوٹ>(X_D, K_D, Y_D, params, stream);
        }
    }
}
