#अन्तर्भाव "conv2d.hpp"
#अन्तर्भाव "convert.hpp"

संरचना conv2d_params {
    ध्रुव int64_t IW, IH;
    ध्रुव int64_t OW, OH;
    ध्रुव int64_t KW, KH;
    ध्रुव int64_t ST_X, ST_Y;
    ध्रुव int64_t PD_X, PD_Y;
    ध्रुव int64_t DL_X, DL_Y;
    ध्रुव int64_t IC, OC;
    ध्रुव int64_t B;
    ध्रुव int64_t TOTAL;
};

संरचना conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

स्थिर अन्तःस्थ int64_t conv2d_max64(int64_t a, int64_t b) {
    प्रत्यागम (a > b) ? a : b;
}

स्थिर अन्तःस्थ int64_t conv2d_min64(int64_t a, int64_t b) {
    प्रत्यागम (a < b) ? a : b;
}

स्थिर अन्तःस्थ conv2d_kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, ध्रुव conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    प्रत्यागम bounds;
}

स्थिर अन्तःस्थ पूर्णाङ्क calculate_निवेशः_coord(int64_t out_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    प्रत्यागम out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
स्थिर अन्तःस्थ int64_t whcn_निवेशः_index(int64_t n, int64_t c, int64_t y, int64_t x, ध्रुव conv2d_params & P) {
    प्रत्यागम n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

स्थिर अन्तःस्थ int64_t whcn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, ध्रुव conv2d_params & P) {
    प्रत्यागम c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
}

स्थिर अन्तःस्थ int64_t whcn_output_index(int64_t n, int64_t c, int64_t y, int64_t x, ध्रुव conv2d_params & P) {
    प्रत्यागम n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
स्थिर अन्तःस्थ int64_t cwhn_निवेशः_index(int64_t n, int64_t c, int64_t y, int64_t x, ध्रुव conv2d_params & P) {
    प्रत्यागम ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

स्थिर अन्तःस्थ int64_t cwhn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, ध्रुव conv2d_params & P) {
    प्रत्यागम ((ky * P.KW + kx) * P.IC + c_in) * P.OC + c_out;
}

template <typename T, तार्किक CWHN>
स्थिर शून्य conv2d_kernel(ध्रुव भिन्न * निवेशः, ध्रुव T * kernel, भिन्न * output,
                          ध्रुव conv2d_params P, ध्रुव sycl::nd_item<3> & item_ct1) {
    ध्रुव int64_t global_idx = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);

    जइ (global_idx >= P.TOTAL) {
        प्रत्यागम;
    }

    ध्रुव int64_t out_x  = global_idx % P.OW;
    ध्रुव int64_t out_y  = (global_idx / P.OW) % P.OH;
    ध्रुव int64_t c_out  = (global_idx / (P.OW * P.OH)) % P.OC;
    ध्रुव int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    भिन्न acc = 0.0f;

    ध्रुव conv2d_kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

    प्रति (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        प्रति (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            ध्रुव int64_t in_y = calculate_निवेशः_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            प्रति (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                ध्रुव int64_t in_x = calculate_निवेशः_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);
                ध्रुव भिन्न निवेशः_val  = निवेशः[CWHN ? cwhn_निवेशः_index(n, c_in, in_y, in_x, P) : whcn_निवेशः_index(n, c_in, in_y, in_x, P)];
                ध्रुव T     kernel_val = kernel[CWHN ? cwhn_kernel_index(c_out, c_in, ky, kx, P) : whcn_kernel_index(c_out, c_in, ky, kx, P)];
                acc += निवेशः_val * ggml_sycl_cast<भिन्न>(kernel_val);
            }
        }
    }

    output[whcn_output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T, तार्किक CWHN>
स्थिर शून्य conv2d_sycl(ध्रुव भिन्न * X_D, ध्रुव T * K_D, भिन्न * Y_D,
                        ध्रुव conv2d_params P, ध्रुव queue_ptr & stream) {
    ध्रुव पूर्णाङ्क अङ्कः_blocks = (P.TOTAL + SYCL_CONV2D_BLOCK_SIZE - 1) / SYCL_CONV2D_BLOCK_SIZE;
    ध्रुव sycl::range<3> block_dims(1, 1, SYCL_CONV2D_BLOCK_SIZE);
    ध्रुव sycl::range<3> block_nums(1, 1, अङ्कः_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, item_ct1);
        });
}

शून्य ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ध्रुव ggml_tensor * kernel = dst->src[0];
    ध्रुव ggml_tensor * निवेशः  = dst->src[1];
    ध्रुव भिन्न *       K_D    = (ध्रुव भिन्न *) kernel->data;
    ध्रुव भिन्न *       X_D    = (ध्रुव भिन्न *) निवेशः->data;
    भिन्न *             Y_D    = (भिन्न *) dst->data;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    ध्रुव तार्किक whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(निवेशः);
    ध्रुव तार्किक cwhn = निवेशः->nb[2]  == ggml_type_size(निवेशः->type) &&
                      kernel->nb[3] == ggml_type_size(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    ध्रुव तार्किक cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(निवेशः->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(निवेशः->ne[2] == kernel->ne[2]);

    ध्रुव queue_ptr stream = ctx.stream();

    ध्रुव int32_t * p    = (ध्रुव int32_t *) dst->op_params;
    ध्रुव पूर्णाङ्क       ST_X = p[0];
    ध्रुव पूर्णाङ्क       ST_Y = p[1];
    ध्रुव पूर्णाङ्क       PD_X = p[2];
    ध्रुव पूर्णाङ्क       PD_Y = p[3];
    ध्रुव पूर्णाङ्क       DL_X = p[4];
    ध्रुव पूर्णाङ्क       DL_Y = p[5];

    ध्रुव पूर्णाङ्क IW = निवेशः->ne[0];
    ध्रुव पूर्णाङ्क IH = निवेशः->ne[1];
    ध्रुव पूर्णाङ्क OW = dst->ne[0];
    ध्रुव पूर्णाङ्क OH = dst->ne[1];
    ध्रुव पूर्णाङ्क KW = kernel->ne[0];
    ध्रुव पूर्णाङ्क KH = kernel->ne[1];
    ध्रुव पूर्णाङ्क IC = निवेशः->ne[2];
    ध्रुव पूर्णाङ्क OC = kernel->ne[3];
    ध्रुव पूर्णाङ्क B  = निवेशः->ne[3];

    ध्रुव int64_t     total  = (int64_t) B * OC * OH * OW;
    ध्रुव conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    जइ (cwhn_layout) {
        जइ (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, सच्च>(X_D, (ध्रुव sycl::half *) K_D, Y_D, params, stream);
        } अण्णहा {
            conv2d_sycl<भिन्न, सच्च>(X_D, K_D, Y_D, params, stream);
        }
    } अण्णहा {
        जइ (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, मिच्छ>(X_D, (ध्रुव sycl::half *) K_D, Y_D, params, stream);
        } अण्णहा {
            conv2d_sycl<भिन्न, मिच्छ>(X_D, K_D, Y_D, params, stream);
        }
    }
}
