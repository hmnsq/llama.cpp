#شامل "conv2d.hpp"
#شامل "convert.hpp"

ساختار conv2d_params {
    ثابت int64_t IW, IH;
    ثابت int64_t OW, OH;
    ثابت int64_t KW, KH;
    ثابت int64_t ST_X, ST_Y;
    ثابت int64_t PD_X, PD_Y;
    ثابت int64_t DL_X, DL_Y;
    ثابت int64_t IC, OC;
    ثابت int64_t B;
    ثابت int64_t TOTAL;
};

ساختار conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

ایستا درونخطی int64_t conv2d_max64(int64_t a, int64_t b) {
    بازگشت (a > b) ? a : b;
}

ایستا درونخطی int64_t conv2d_min64(int64_t a, int64_t b) {
    بازگشت (a < b) ? a : b;
}

ایستا درونخطی conv2d_kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, ثابت conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    بازگشت bounds;
}

ایستا درونخطی صحیح calculate_درون_داد_coord(int64_t out_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    بازگشت out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
ایستا درونخطی int64_t whcn_درون_داد_index(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    بازگشت n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

ایستا درونخطی int64_t whcn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, ثابت conv2d_params & P) {
    بازگشت c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
}

ایستا درونخطی int64_t whcn_output_index(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    بازگشت n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
ایستا درونخطی int64_t cwhn_درون_داد_index(int64_t n, int64_t c, int64_t y, int64_t x, ثابت conv2d_params & P) {
    بازگشت ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

ایستا درونخطی int64_t cwhn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, ثابت conv2d_params & P) {
    بازگشت ((ky * P.KW + kx) * P.IC + c_in) * P.OC + c_out;
}

template <typename T, منطقی CWHN>
ایستا تهی conv2d_kernel(ثابت اعشاری * درون_داد, ثابت T * kernel, اعشاری * output,
                          ثابت conv2d_params P, ثابت sycl::nd_item<3> & item_ct1) {
    ثابت int64_t global_idx = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);

    اگر (global_idx >= P.TOTAL) {
        بازگشت;
    }

    ثابت int64_t out_x  = global_idx % P.OW;
    ثابت int64_t out_y  = (global_idx / P.OW) % P.OH;
    ثابت int64_t c_out  = (global_idx / (P.OW * P.OH)) % P.OC;
    ثابت int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    اعشاری acc = 0.0f;

    ثابت conv2d_kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

    برای (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        برای (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            ثابت int64_t in_y = calculate_درون_داد_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            برای (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                ثابت int64_t in_x = calculate_درون_داد_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);
                ثابت اعشاری درون_داد_val  = درون_داد[CWHN ? cwhn_درون_داد_index(n, c_in, in_y, in_x, P) : whcn_درون_داد_index(n, c_in, in_y, in_x, P)];
                ثابت T     kernel_val = kernel[CWHN ? cwhn_kernel_index(c_out, c_in, ky, kx, P) : whcn_kernel_index(c_out, c_in, ky, kx, P)];
                acc += درون_داد_val * ggml_sycl_cast<اعشاری>(kernel_val);
            }
        }
    }

    output[whcn_output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T, منطقی CWHN>
ایستا تهی conv2d_sycl(ثابت اعشاری * X_D, ثابت T * K_D, اعشاری * Y_D,
                        ثابت conv2d_params P, ثابت queue_ptr & stream) {
    ثابت صحیح رقم_blocks = (P.TOTAL + SYCL_CONV2D_BLOCK_SIZE - 1) / SYCL_CONV2D_BLOCK_SIZE;
    ثابت sycl::range<3> block_dims(1, 1, SYCL_CONV2D_BLOCK_SIZE);
    ثابت sycl::range<3> block_nums(1, 1, رقم_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, item_ct1);
        });
}

تهی ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ثابت ggml_tensor * kernel = dst->src[0];
    ثابت ggml_tensor * درون_داد  = dst->src[1];
    ثابت اعشاری *       K_D    = (ثابت اعشاری *) kernel->data;
    ثابت اعشاری *       X_D    = (ثابت اعشاری *) درون_داد->data;
    اعشاری *             Y_D    = (اعشاری *) dst->data;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    ثابت منطقی whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(درون_داد);
    ثابت منطقی cwhn = درون_داد->nb[2]  == ggml_type_size(درون_داد->type) &&
                      kernel->nb[3] == ggml_type_size(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    ثابت منطقی cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(درون_داد->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(درون_داد->ne[2] == kernel->ne[2]);

    ثابت queue_ptr stream = ctx.stream();

    ثابت int32_t * p    = (ثابت int32_t *) dst->op_params;
    ثابت صحیح       ST_X = p[0];
    ثابت صحیح       ST_Y = p[1];
    ثابت صحیح       PD_X = p[2];
    ثابت صحیح       PD_Y = p[3];
    ثابت صحیح       DL_X = p[4];
    ثابت صحیح       DL_Y = p[5];

    ثابت صحیح IW = درون_داد->ne[0];
    ثابت صحیح IH = درون_داد->ne[1];
    ثابت صحیح OW = dst->ne[0];
    ثابت صحیح OH = dst->ne[1];
    ثابت صحیح KW = kernel->ne[0];
    ثابت صحیح KH = kernel->ne[1];
    ثابت صحیح IC = درون_داد->ne[2];
    ثابت صحیح OC = kernel->ne[3];
    ثابت صحیح B  = درون_داد->ne[3];

    ثابت int64_t     total  = (int64_t) B * OC * OH * OW;
    ثابت conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    اگر (cwhn_layout) {
        اگر (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, درست>(X_D, (ثابت sycl::half *) K_D, Y_D, params, stream);
        } وگرنه {
            conv2d_sycl<اعشاری, درست>(X_D, K_D, Y_D, params, stream);
        }
    } وگرنه {
        اگر (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, نادرست>(X_D, (ثابت sycl::half *) K_D, Y_D, params, stream);
        } وگرنه {
            conv2d_sycl<اعشاری, نادرست>(X_D, K_D, Y_D, params, stream);
        }
    }
}
