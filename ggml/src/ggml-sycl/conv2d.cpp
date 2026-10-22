#অন্তর্ভাব "conv2d.hpp"
#অন্তর্ভাব "convert.hpp"

কাঠামো conv2d_params {
    ধ্রুবক int64_t IW, IH;
    ধ্রুবক int64_t OW, OH;
    ধ্রুবক int64_t KW, KH;
    ধ্রুবক int64_t ST_X, ST_Y;
    ধ্রুবক int64_t PD_X, PD_Y;
    ধ্রুবক int64_t DL_X, DL_Y;
    ধ্রুবক int64_t IC, OC;
    ধ্রুবক int64_t B;
    ধ্রুবক int64_t যোগ_5;
};

কাঠামো conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

স্থির অন্তঃরেখ int64_t conv2d_max64(int64_t a, int64_t b) {
    ফেরত (a > b) ? a : b;
}

স্থির অন্তঃরেখ int64_t conv2d_min64(int64_t a, int64_t b) {
    ফেরত (a < b) ? a : b;
}

স্থির অন্তঃরেখ conv2d_kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, ধ্রুবক conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    ফেরত bounds;
}

স্থির অন্তঃরেখ পূর্ণসংখ্যা calculate_প্রবেশ_coord(int64_t out_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    ফেরত out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
স্থির অন্তঃরেখ int64_t whcn_প্রবেশ_সূচক(int64_t n, int64_t c, int64_t y, int64_t x, ধ্রুবক conv2d_params & P) {
    ফেরত n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

স্থির অন্তঃরেখ int64_t whcn_kernel_সূচক(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, ধ্রুবক conv2d_params & P) {
    ফেরত c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
}

স্থির অন্তঃরেখ int64_t whcn_output_সূচক(int64_t n, int64_t c, int64_t y, int64_t x, ধ্রুবক conv2d_params & P) {
    ফেরত n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
স্থির অন্তঃরেখ int64_t cwhn_প্রবেশ_সূচক(int64_t n, int64_t c, int64_t y, int64_t x, ধ্রুবক conv2d_params & P) {
    ফেরত ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

স্থির অন্তঃরেখ int64_t cwhn_kernel_সূচক(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, ধ্রুবক conv2d_params & P) {
    ফেরত ((ky * P.KW + kx) * P.IC + c_in) * P.OC + c_out;
}

template <typename T, যৌক্তিক CWHN>
স্থির শূন্য conv2d_kernel(ধ্রুবক দশমিক * প্রবেশ, ধ্রুবক T * kernel, দশমিক * output,
                          ধ্রুবক conv2d_params P, ধ্রুবক sycl::nd_item<3> & item_ct1) {
    ধ্রুবক int64_t global_idx = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);

    যদি (global_idx >= P.যোগ_5) {
        ফেরত;
    }

    ধ্রুবক int64_t out_x  = global_idx % P.OW;
    ধ্রুবক int64_t out_y  = (global_idx / P.OW) % P.OH;
    ধ্রুবক int64_t c_out  = (global_idx / (P.OW * P.OH)) % P.OC;
    ধ্রুবক int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    দশমিক acc = 0.0f;

    ধ্রুবক conv2d_kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

    জন্য (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        জন্য (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            ধ্রুবক int64_t in_y = calculate_প্রবেশ_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            জন্য (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                ধ্রুবক int64_t in_x = calculate_প্রবেশ_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);
                ধ্রুবক দশমিক প্রবেশ_val  = প্রবেশ[CWHN ? cwhn_প্রবেশ_সূচক(n, c_in, in_y, in_x, P) : whcn_প্রবেশ_সূচক(n, c_in, in_y, in_x, P)];
                ধ্রুবক T     kernel_val = kernel[CWHN ? cwhn_kernel_সূচক(c_out, c_in, ky, kx, P) : whcn_kernel_সূচক(c_out, c_in, ky, kx, P)];
                acc += প্রবেশ_val * ggml_sycl_cast<দশমিক>(kernel_val);
            }
        }
    }

    output[whcn_output_সূচক(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T, যৌক্তিক CWHN>
স্থির শূন্য conv2d_sycl(ধ্রুবক দশমিক * X_D, ধ্রুবক T * K_D, দশমিক * Y_D,
                        ধ্রুবক conv2d_params P, ধ্রুবক queue_ptr & stream) {
    ধ্রুবক পূর্ণসংখ্যা অঙ্ক_blocks = (P.যোগ_5 + SYCL_CONV2_D_BLOCK_আকার - 1) / SYCL_CONV2_D_BLOCK_আকার;
    ধ্রুবক sycl::range<3> block_dims(1, 1, SYCL_CONV2_D_BLOCK_আকার);
    ধ্রুবক sycl::range<3> block_nums(1, 1, অঙ্ক_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, item_ct1);
        });
}

শূন্য ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ধ্রুবক ggml_tensor * kernel = dst->src[0];
    ধ্রুবক ggml_tensor * প্রবেশ  = dst->src[1];
    ধ্রুবক দশমিক *       K_D    = (ধ্রুবক দশমিক *) kernel->তথ্য;
    ধ্রুবক দশমিক *       X_D    = (ধ্রুবক দশমিক *) প্রবেশ->তথ্য;
    দশমিক *             Y_D    = (দশমিক *) dst->তথ্য;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    ধ্রুবক যৌক্তিক whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(প্রবেশ);
    ধ্রুবক যৌক্তিক cwhn = প্রবেশ->nb[2]  == ggml_type_আকার(প্রবেশ->type) &&
                      kernel->nb[3] == ggml_type_আকার(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    ধ্রুবক যৌক্তিক cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(প্রবেশ->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(প্রবেশ->ne[2] == kernel->ne[2]);

    ধ্রুবক queue_ptr stream = ctx.stream();

    ধ্রুবক int32_t * p    = (ধ্রুবক int32_t *) dst->op_params;
    ধ্রুবক পূর্ণসংখ্যা       ST_X = p[0];
    ধ্রুবক পূর্ণসংখ্যা       ST_Y = p[1];
    ধ্রুবক পূর্ণসংখ্যা       PD_X = p[2];
    ধ্রুবক পূর্ণসংখ্যা       PD_Y = p[3];
    ধ্রুবক পূর্ণসংখ্যা       DL_X = p[4];
    ধ্রুবক পূর্ণসংখ্যা       DL_Y = p[5];

    ধ্রুবক পূর্ণসংখ্যা IW = প্রবেশ->ne[0];
    ধ্রুবক পূর্ণসংখ্যা IH = প্রবেশ->ne[1];
    ধ্রুবক পূর্ণসংখ্যা OW = dst->ne[0];
    ধ্রুবক পূর্ণসংখ্যা OH = dst->ne[1];
    ধ্রুবক পূর্ণসংখ্যা KW = kernel->ne[0];
    ধ্রুবক পূর্ণসংখ্যা KH = kernel->ne[1];
    ধ্রুবক পূর্ণসংখ্যা IC = প্রবেশ->ne[2];
    ধ্রুবক পূর্ণসংখ্যা OC = kernel->ne[3];
    ধ্রুবক পূর্ণসংখ্যা B  = প্রবেশ->ne[3];

    ধ্রুবক int64_t     যোগ  = (int64_t) B * OC * OH * OW;
    ধ্রুবক conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, যোগ };

    যদি (cwhn_layout) {
        যদি (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, সত্য>(X_D, (ধ্রুবক sycl::half *) K_D, Y_D, params, stream);
        } অন্যথা {
            conv2d_sycl<দশমিক, সত্য>(X_D, K_D, Y_D, params, stream);
        }
    } অন্যথা {
        যদি (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, মিথ্যা>(X_D, (ধ্রুবক sycl::half *) K_D, Y_D, params, stream);
        } অন্যথা {
            conv2d_sycl<দশমিক, মিথ্যা>(X_D, K_D, Y_D, params, stream);
        }
    }
}
