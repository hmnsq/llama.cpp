#अन्तर्भाव "conv2d.hpp"
#अन्तर्भाव "convert.hpp"

संरचना conv2d_params {
    स्थिरांक int64_t IW, IH;
    स्थिरांक int64_t OW, OH;
    स्थिरांक int64_t KW, KH;
    स्थिरांक int64_t ST_X, ST_Y;
    स्थिरांक int64_t PD_X, PD_Y;
    स्थिरांक int64_t DL_X, DL_Y;
    स्थिरांक int64_t IC, OC;
    स्थिरांक int64_t B;
    स्थिरांक int64_t TOTAL;
};

संरचना conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

स्थिर ओळीत int64_t conv2d_max64(int64_t a, int64_t b) {
    परत (a > b) ? a : b;
}

स्थिर ओळीत int64_t conv2d_min64(int64_t a, int64_t b) {
    परत (a < b) ? a : b;
}

स्थिर ओळीत conv2d_kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, स्थिरांक conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    परत bounds;
}

स्थिर ओळीत पूर्णांक calculate_आदान_coord(int64_t out_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    परत out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
स्थिर ओळीत int64_t whcn_आदान_index(int64_t n, int64_t c, int64_t y, int64_t x, स्थिरांक conv2d_params & P) {
    परत n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

स्थिर ओळीत int64_t whcn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, स्थिरांक conv2d_params & P) {
    परत c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
}

स्थिर ओळीत int64_t whcn_output_index(int64_t n, int64_t c, int64_t y, int64_t x, स्थिरांक conv2d_params & P) {
    परत n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
स्थिर ओळीत int64_t cwhn_आदान_index(int64_t n, int64_t c, int64_t y, int64_t x, स्थिरांक conv2d_params & P) {
    परत ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

स्थिर ओळीत int64_t cwhn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, स्थिरांक conv2d_params & P) {
    परत ((ky * P.KW + kx) * P.IC + c_in) * P.OC + c_out;
}

ढाँचा <typename T, तार्किक CWHN>
स्थिर शून्य conv2d_kernel(स्थिरांक दशांश * आदान, स्थिरांक T * kernel, दशांश * output,
                          स्थिरांक conv2d_params P, स्थिरांक sycl::nd_item<3> & item_ct1) {
    स्थिरांक int64_t global_idx = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);

    जर (global_idx >= P.TOTAL) {
        परत;
    }

    स्थिरांक int64_t out_x  = global_idx % P.OW;
    स्थिरांक int64_t out_y  = (global_idx / P.OW) % P.OH;
    स्थिरांक int64_t c_out  = (global_idx / (P.OW * P.OH)) % P.OC;
    स्थिरांक int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    दशांश acc = 0.0f;

    स्थिरांक conv2d_kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

    साठी (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        साठी (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            स्थिरांक int64_t in_y = calculate_आदान_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            साठी (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                स्थिरांक int64_t in_x = calculate_आदान_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);
                स्थिरांक दशांश आदान_val  = आदान[CWHN ? cwhn_आदान_index(n, c_in, in_y, in_x, P) : whcn_आदान_index(n, c_in, in_y, in_x, P)];
                स्थिरांक T     kernel_val = kernel[CWHN ? cwhn_kernel_index(c_out, c_in, ky, kx, P) : whcn_kernel_index(c_out, c_in, ky, kx, P)];
                acc += आदान_val * ggml_sycl_cast<दशांश>(kernel_val);
            }
        }
    }

    output[whcn_output_index(n, c_out, out_y, out_x, P)] = acc;
}

ढाँचा <typename T, तार्किक CWHN>
स्थिर शून्य conv2d_sycl(स्थिरांक दशांश * X_D, स्थिरांक T * K_D, दशांश * Y_D,
                        स्थिरांक conv2d_params P, स्थिरांक queue_ptr & stream) {
    स्थिरांक पूर्णांक अंक_blocks = (P.TOTAL + SYCL_CONV2D_BLOCK_SIZE - 1) / SYCL_CONV2D_BLOCK_SIZE;
    स्थिरांक sycl::range<3> block_dims(1, 1, SYCL_CONV2D_BLOCK_SIZE);
    स्थिरांक sycl::range<3> block_nums(1, 1, अंक_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, item_ct1);
        });
}

शून्य ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    स्थिरांक ggml_tensor * kernel = dst->src[0];
    स्थिरांक ggml_tensor * आदान  = dst->src[1];
    स्थिरांक दशांश *       K_D    = (स्थिरांक दशांश *) kernel->data;
    स्थिरांक दशांश *       X_D    = (स्थिरांक दशांश *) आदान->data;
    दशांश *             Y_D    = (दशांश *) dst->data;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    स्थिरांक तार्किक whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(आदान);
    स्थिरांक तार्किक cwhn = आदान->nb[2]  == ggml_type_size(आदान->type) &&
                      kernel->nb[3] == ggml_type_size(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    स्थिरांक तार्किक cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(आदान->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(आदान->ne[2] == kernel->ne[2]);

    स्थिरांक queue_ptr stream = ctx.stream();

    स्थिरांक int32_t * p    = (स्थिरांक int32_t *) dst->op_params;
    स्थिरांक पूर्णांक       ST_X = p[0];
    स्थिरांक पूर्णांक       ST_Y = p[1];
    स्थिरांक पूर्णांक       PD_X = p[2];
    स्थिरांक पूर्णांक       PD_Y = p[3];
    स्थिरांक पूर्णांक       DL_X = p[4];
    स्थिरांक पूर्णांक       DL_Y = p[5];

    स्थिरांक पूर्णांक IW = आदान->ne[0];
    स्थिरांक पूर्णांक IH = आदान->ne[1];
    स्थिरांक पूर्णांक OW = dst->ne[0];
    स्थिरांक पूर्णांक OH = dst->ne[1];
    स्थिरांक पूर्णांक KW = kernel->ne[0];
    स्थिरांक पूर्णांक KH = kernel->ne[1];
    स्थिरांक पूर्णांक IC = आदान->ne[2];
    स्थिरांक पूर्णांक OC = kernel->ne[3];
    स्थिरांक पूर्णांक B  = आदान->ne[3];

    स्थिरांक int64_t     total  = (int64_t) B * OC * OH * OW;
    स्थिरांक conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    जर (cwhn_layout) {
        जर (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, सत्य>(X_D, (स्थिरांक sycl::half *) K_D, Y_D, params, stream);
        } अन्यथा {
            conv2d_sycl<दशांश, सत्य>(X_D, K_D, Y_D, params, stream);
        }
    } अन्यथा {
        जर (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, असत्य>(X_D, (स्थिरांक sycl::half *) K_D, Y_D, params, stream);
        } अन्यथा {
            conv2d_sycl<दशांश, असत्य>(X_D, K_D, Y_D, params, stream);
        }
    }
}
