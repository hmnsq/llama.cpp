#अन्तर्भाव "conv2d.hpp"
#अन्तर्भाव "convert.hpp"

संरचना conv2d_params {
    अचर int64_t IW, IH;
    अचर int64_t OW, OH;
    अचर int64_t KW, KH;
    अचर int64_t ST_X, ST_Y;
    अचर int64_t PD_X, PD_Y;
    अचर int64_t DL_X, DL_Y;
    अचर int64_t IC, OC;
    अचर int64_t B;
    अचर int64_t TOTAL;
};

संरचना conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

स्थिर अन्तःस्थ int64_t conv2d_max64(int64_t a, int64_t b) {
    फर्काऊ (a > b) ? a : b;
}

स्थिर अन्तःस्थ int64_t conv2d_min64(int64_t a, int64_t b) {
    फर्काऊ (a < b) ? a : b;
}

स्थिर अन्तःस्थ conv2d_kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, अचर conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    फर्काऊ bounds;
}

स्थिर अन्तःस्थ पूर्णाङ्क calculate_आगत_coord(int64_t out_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    फर्काऊ out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
स्थिर अन्तःस्थ int64_t whcn_आगत_index(int64_t n, int64_t c, int64_t y, int64_t x, अचर conv2d_params & P) {
    फर्काऊ n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

स्थिर अन्तःस्थ int64_t whcn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, अचर conv2d_params & P) {
    फर्काऊ c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
}

स्थिर अन्तःस्थ int64_t whcn_output_index(int64_t n, int64_t c, int64_t y, int64_t x, अचर conv2d_params & P) {
    फर्काऊ n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
स्थिर अन्तःस्थ int64_t cwhn_आगत_index(int64_t n, int64_t c, int64_t y, int64_t x, अचर conv2d_params & P) {
    फर्काऊ ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

स्थिर अन्तःस्थ int64_t cwhn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, अचर conv2d_params & P) {
    फर्काऊ ((ky * P.KW + kx) * P.IC + c_in) * P.OC + c_out;
}

ढाँचा <typename T, तार्किक CWHN>
स्थिर शून्य conv2d_kernel(अचर भिन्न * आगत, अचर T * kernel, भिन्न * output,
                          अचर conv2d_params P, अचर sycl::nd_item<3> & item_ct1) {
    अचर int64_t global_idx = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);

    यदि (global_idx >= P.TOTAL) {
        फर्काऊ;
    }

    अचर int64_t out_x  = global_idx % P.OW;
    अचर int64_t out_y  = (global_idx / P.OW) % P.OH;
    अचर int64_t c_out  = (global_idx / (P.OW * P.OH)) % P.OC;
    अचर int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    भिन्न acc = 0.0f;

    अचर conv2d_kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

    लागि (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        लागि (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            अचर int64_t in_y = calculate_आगत_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            लागि (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                अचर int64_t in_x = calculate_आगत_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);
                अचर भिन्न आगत_val  = आगत[CWHN ? cwhn_आगत_index(n, c_in, in_y, in_x, P) : whcn_आगत_index(n, c_in, in_y, in_x, P)];
                अचर T     kernel_val = kernel[CWHN ? cwhn_kernel_index(c_out, c_in, ky, kx, P) : whcn_kernel_index(c_out, c_in, ky, kx, P)];
                acc += आगत_val * ggml_sycl_cast<भिन्न>(kernel_val);
            }
        }
    }

    output[whcn_output_index(n, c_out, out_y, out_x, P)] = acc;
}

ढाँचा <typename T, तार्किक CWHN>
स्थिर शून्य conv2d_sycl(अचर भिन्न * X_D, अचर T * K_D, भिन्न * Y_D,
                        अचर conv2d_params P, अचर queue_ptr & stream) {
    अचर पूर्णाङ्क अङ्क_blocks = (P.TOTAL + SYCL_CONV2D_BLOCK_SIZE - 1) / SYCL_CONV2D_BLOCK_SIZE;
    अचर sycl::range<3> block_dims(1, 1, SYCL_CONV2D_BLOCK_SIZE);
    अचर sycl::range<3> block_nums(1, 1, अङ्क_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, item_ct1);
        });
}

शून्य ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    अचर ggml_tensor * kernel = dst->src[0];
    अचर ggml_tensor * आगत  = dst->src[1];
    अचर भिन्न *       K_D    = (अचर भिन्न *) kernel->data;
    अचर भिन्न *       X_D    = (अचर भिन्न *) आगत->data;
    भिन्न *             Y_D    = (भिन्न *) dst->data;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    अचर तार्किक whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(आगत);
    अचर तार्किक cwhn = आगत->nb[2]  == ggml_type_size(आगत->type) &&
                      kernel->nb[3] == ggml_type_size(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    अचर तार्किक cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(आगत->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(आगत->ne[2] == kernel->ne[2]);

    अचर queue_ptr stream = ctx.stream();

    अचर int32_t * p    = (अचर int32_t *) dst->op_params;
    अचर पूर्णाङ्क       ST_X = p[0];
    अचर पूर्णाङ्क       ST_Y = p[1];
    अचर पूर्णाङ्क       PD_X = p[2];
    अचर पूर्णाङ्क       PD_Y = p[3];
    अचर पूर्णाङ्क       DL_X = p[4];
    अचर पूर्णाङ्क       DL_Y = p[5];

    अचर पूर्णाङ्क IW = आगत->ne[0];
    अचर पूर्णाङ्क IH = आगत->ne[1];
    अचर पूर्णाङ्क OW = dst->ne[0];
    अचर पूर्णाङ्क OH = dst->ne[1];
    अचर पूर्णाङ्क KW = kernel->ne[0];
    अचर पूर्णाङ्क KH = kernel->ne[1];
    अचर पूर्णाङ्क IC = आगत->ne[2];
    अचर पूर्णाङ्क OC = kernel->ne[3];
    अचर पूर्णाङ्क B  = आगत->ne[3];

    अचर int64_t     total  = (int64_t) B * OC * OH * OW;
    अचर conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    यदि (cwhn_layout) {
        यदि (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, सत्य>(X_D, (अचर sycl::half *) K_D, Y_D, params, stream);
        } अन्यथा {
            conv2d_sycl<भिन्न, सत्य>(X_D, K_D, Y_D, params, stream);
        }
    } अन्यथा {
        यदि (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, असत्य>(X_D, (अचर sycl::half *) K_D, Y_D, params, stream);
        } अन्यथा {
            conv2d_sycl<भिन्न, असत्य>(X_D, K_D, Y_D, params, stream);
        }
    }
}
