#অন্তর্ভাব "conv2d.hpp"
#অন্তর্ভাব "convert.hpp"

সংরচনা conv2d_params {
    ধ্রুব int64_t IW, IH;
    ধ্রুব int64_t OW, OH;
    ধ্রুব int64_t KW, KH;
    ধ্রুব int64_t ST_X, ST_Y;
    ধ্রুব int64_t PD_X, PD_Y;
    ধ্রুব int64_t DL_X, DL_Y;
    ধ্রুব int64_t IC, OC;
    ধ্রুব int64_t B;
    ধ্রুব int64_t TOTAL;
};

সংরচনা conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

স্থির অন্তঃস্থ int64_t conv2d_max64(int64_t a, int64_t b) {
    প্রত্যাগম (a > b) ? a : b;
}

স্থির অন্তঃস্থ int64_t conv2d_min64(int64_t a, int64_t b) {
    প্রত্যাগম (a < b) ? a : b;
}

স্থির অন্তঃস্থ conv2d_kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, ধ্রুব conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    প্রত্যাগম bounds;
}

স্থির অন্তঃস্থ পূর্ণাঙ্ক calculate_প্রবেশ_coord(int64_t out_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    প্রত্যাগম out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
স্থির অন্তঃস্থ int64_t whcn_প্রবেশ_index(int64_t n, int64_t c, int64_t y, int64_t x, ধ্রুব conv2d_params & P) {
    প্রত্যাগম n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

স্থির অন্তঃস্থ int64_t whcn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, ধ্রুব conv2d_params & P) {
    প্রত্যাগম c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
}

স্থির অন্তঃস্থ int64_t whcn_output_index(int64_t n, int64_t c, int64_t y, int64_t x, ধ্রুব conv2d_params & P) {
    প্রত্যাগম n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
স্থির অন্তঃস্থ int64_t cwhn_প্রবেশ_index(int64_t n, int64_t c, int64_t y, int64_t x, ধ্রুব conv2d_params & P) {
    প্রত্যাগম ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

স্থির অন্তঃস্থ int64_t cwhn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, ধ্রুব conv2d_params & P) {
    প্রত্যাগম ((ky * P.KW + kx) * P.IC + c_in) * P.OC + c_out;
}

template <typename T, তার্কিক CWHN>
স্থির শূন্য conv2d_kernel(ধ্রুব ভিন্ন * প্রবেশ, ধ্রুব T * kernel, ভিন্ন * output,
                          ধ্রুব conv2d_params P, ধ্রুব sycl::nd_item<3> & item_ct1) {
    ধ্রুব int64_t global_idx = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);

    যদি (global_idx >= P.TOTAL) {
        প্রত্যাগম;
    }

    ধ্রুব int64_t out_x  = global_idx % P.OW;
    ধ্রুব int64_t out_y  = (global_idx / P.OW) % P.OH;
    ধ্রুব int64_t c_out  = (global_idx / (P.OW * P.OH)) % P.OC;
    ধ্রুব int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    ভিন্ন acc = 0.0f;

    ধ্রুব conv2d_kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

    প্রতি (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        প্রতি (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            ধ্রুব int64_t in_y = calculate_প্রবেশ_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            প্রতি (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                ধ্রুব int64_t in_x = calculate_প্রবেশ_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);
                ধ্রুব ভিন্ন প্রবেশ_val  = প্রবেশ[CWHN ? cwhn_প্রবেশ_index(n, c_in, in_y, in_x, P) : whcn_প্রবেশ_index(n, c_in, in_y, in_x, P)];
                ধ্রুব T     kernel_val = kernel[CWHN ? cwhn_kernel_index(c_out, c_in, ky, kx, P) : whcn_kernel_index(c_out, c_in, ky, kx, P)];
                acc += প্রবেশ_val * ggml_sycl_cast<ভিন্ন>(kernel_val);
            }
        }
    }

    output[whcn_output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T, তার্কিক CWHN>
স্থির শূন্য conv2d_sycl(ধ্রুব ভিন্ন * X_D, ধ্রুব T * K_D, ভিন্ন * Y_D,
                        ধ্রুব conv2d_params P, ধ্রুব queue_ptr & stream) {
    ধ্রুব পূর্ণাঙ্ক অঙ্ক_blocks = (P.TOTAL + SYCL_CONV2D_BLOCK_SIZE - 1) / SYCL_CONV2D_BLOCK_SIZE;
    ধ্রুব sycl::range<3> block_dims(1, 1, SYCL_CONV2D_BLOCK_SIZE);
    ধ্রুব sycl::range<3> block_nums(1, 1, অঙ্ক_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, item_ct1);
        });
}

শূন্য ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ধ্রুব ggml_tensor * kernel = dst->src[0];
    ধ্রুব ggml_tensor * প্রবেশ  = dst->src[1];
    ধ্রুব ভিন্ন *       K_D    = (ধ্রুব ভিন্ন *) kernel->data;
    ধ্রুব ভিন্ন *       X_D    = (ধ্রুব ভিন্ন *) প্রবেশ->data;
    ভিন্ন *             Y_D    = (ভিন্ন *) dst->data;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    ধ্রুব তার্কিক whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(প্রবেশ);
    ধ্রুব তার্কিক cwhn = প্রবেশ->nb[2]  == ggml_type_size(প্রবেশ->type) &&
                      kernel->nb[3] == ggml_type_size(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    ধ্রুব তার্কিক cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(প্রবেশ->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(প্রবেশ->ne[2] == kernel->ne[2]);

    ধ্রুব queue_ptr stream = ctx.stream();

    ধ্রুব int32_t * p    = (ধ্রুব int32_t *) dst->op_params;
    ধ্রুব পূর্ণাঙ্ক       ST_X = p[0];
    ধ্রুব পূর্ণাঙ্ক       ST_Y = p[1];
    ধ্রুব পূর্ণাঙ্ক       PD_X = p[2];
    ধ্রুব পূর্ণাঙ্ক       PD_Y = p[3];
    ধ্রুব পূর্ণাঙ্ক       DL_X = p[4];
    ধ্রুব পূর্ণাঙ্ক       DL_Y = p[5];

    ধ্রুব পূর্ণাঙ্ক IW = প্রবেশ->ne[0];
    ধ্রুব পূর্ণাঙ্ক IH = প্রবেশ->ne[1];
    ধ্রুব পূর্ণাঙ্ক OW = dst->ne[0];
    ধ্রুব পূর্ণাঙ্ক OH = dst->ne[1];
    ধ্রুব পূর্ণাঙ্ক KW = kernel->ne[0];
    ধ্রুব পূর্ণাঙ্ক KH = kernel->ne[1];
    ধ্রুব পূর্ণাঙ্ক IC = প্রবেশ->ne[2];
    ধ্রুব পূর্ণাঙ্ক OC = kernel->ne[3];
    ধ্রুব পূর্ণাঙ্ক B  = প্রবেশ->ne[3];

    ধ্রুব int64_t     total  = (int64_t) B * OC * OH * OW;
    ধ্রুব conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    যদি (cwhn_layout) {
        যদি (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, সত্য>(X_D, (ধ্রুব sycl::half *) K_D, Y_D, params, stream);
        } অন্যথা {
            conv2d_sycl<ভিন্ন, সত্য>(X_D, K_D, Y_D, params, stream);
        }
    } অন্যথা {
        যদি (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, অসত্য>(X_D, (ধ্রুব sycl::half *) K_D, Y_D, params, stream);
        } অন্যথা {
            conv2d_sycl<ভিন্ন, অসত্য>(X_D, K_D, Y_D, params, stream);
        }
    }
}
