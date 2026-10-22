#समावेश "conv2d.hpp"
#समावेश "convert.hpp"

काष्ठा conv2d_params {
    स्थिर int64_t IW, IH;
    स्थिर int64_t OW, OH;
    स्थिर int64_t KW, KH;
    स्थिर int64_t ST_X, ST_Y;
    स्थिर int64_t PD_X, PD_Y;
    स्थिर int64_t DL_X, DL_Y;
    स्थिर int64_t IC, OC;
    स्थिर int64_t B;
    स्थिर int64_t योग_5;
};

काष्ठा conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

अटल अंतरभूत int64_t conv2d_max64(int64_t a, int64_t b) {
    वापस (a > b) ? a : b;
}

अटल अंतरभूत int64_t conv2d_min64(int64_t a, int64_t b) {
    वापस (a < b) ? a : b;
}

अटल अंतरभूत conv2d_kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, स्थिर conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    वापस bounds;
}

अटल अंतरभूत पूर्णांक calculate_निवेश_coord(int64_t out_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    वापस out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
अटल अंतरभूत int64_t whcn_निवेश_सूचकांक(int64_t n, int64_t c, int64_t y, int64_t x, स्थिर conv2d_params & P) {
    वापस n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

अटल अंतरभूत int64_t whcn_kernel_सूचकांक(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, स्थिर conv2d_params & P) {
    वापस c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
}

अटल अंतरभूत int64_t whcn_output_सूचकांक(int64_t n, int64_t c, int64_t y, int64_t x, स्थिर conv2d_params & P) {
    वापस n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
अटल अंतरभूत int64_t cwhn_निवेश_सूचकांक(int64_t n, int64_t c, int64_t y, int64_t x, स्थिर conv2d_params & P) {
    वापस ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

अटल अंतरभूत int64_t cwhn_kernel_सूचकांक(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, स्थिर conv2d_params & P) {
    वापस ((ky * P.KW + kx) * P.IC + c_in) * P.OC + c_out;
}

ढाँचा <typename T, bool CWHN>
अटल व्योम conv2d_kernel(स्थिर भग्न * निवेश, स्थिर T * kernel, भग्न * output,
                          स्थिर conv2d_params P, स्थिर sycl::nd_item<3> & item_ct1) {
    स्थिर int64_t global_idx = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);

    अगर (global_idx >= P.योग_5) {
        वापस;
    }

    स्थिर int64_t out_x  = global_idx % P.OW;
    स्थिर int64_t out_y  = (global_idx / P.OW) % P.OH;
    स्थिर int64_t c_out  = (global_idx / (P.OW * P.OH)) % P.OC;
    स्थिर int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    भग्न acc = 0.0f;

    स्थिर conv2d_kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

    क्रम (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        क्रम (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            स्थिर int64_t in_y = calculate_निवेश_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            क्रम (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                स्थिर int64_t in_x = calculate_निवेश_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);
                स्थिर भग्न निवेश_val  = निवेश[CWHN ? cwhn_निवेश_सूचकांक(n, c_in, in_y, in_x, P) : whcn_निवेश_सूचकांक(n, c_in, in_y, in_x, P)];
                स्थिर T     kernel_val = kernel[CWHN ? cwhn_kernel_सूचकांक(c_out, c_in, ky, kx, P) : whcn_kernel_सूचकांक(c_out, c_in, ky, kx, P)];
                acc += निवेश_val * ggml_sycl_cast<भग्न>(kernel_val);
            }
        }
    }

    output[whcn_output_सूचकांक(n, c_out, out_y, out_x, P)] = acc;
}

ढाँचा <typename T, bool CWHN>
अटल व्योम conv2d_sycl(स्थिर भग्न * X_D, स्थिर T * K_D, भग्न * Y_D,
                        स्थिर conv2d_params P, स्थिर queue_ptr & stream) {
    स्थिर पूर्णांक अंक_blocks = (P.योग_5 + SYCL_CONV2_D_BLOCK_आकार - 1) / SYCL_CONV2_D_BLOCK_आकार;
    स्थिर sycl::range<3> block_dims(1, 1, SYCL_CONV2_D_BLOCK_आकार);
    स्थिर sycl::range<3> block_nums(1, 1, अंक_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, item_ct1);
        });
}

व्योम ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    स्थिर ggml_tensor * kernel = dst->src[0];
    स्थिर ggml_tensor * निवेश  = dst->src[1];
    स्थिर भग्न *       K_D    = (स्थिर भग्न *) kernel->तथ्य;
    स्थिर भग्न *       X_D    = (स्थिर भग्न *) निवेश->तथ्य;
    भग्न *             Y_D    = (भग्न *) dst->तथ्य;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    स्थिर bool whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(निवेश);
    स्थिर bool cwhn = निवेश->nb[2]  == ggml_type_आकार(निवेश->type) &&
                      kernel->nb[3] == ggml_type_आकार(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    स्थिर bool cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(निवेश->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(निवेश->ne[2] == kernel->ne[2]);

    स्थिर queue_ptr stream = ctx.stream();

    स्थिर int32_t * p    = (स्थिर int32_t *) dst->op_params;
    स्थिर पूर्णांक       ST_X = p[0];
    स्थिर पूर्णांक       ST_Y = p[1];
    स्थिर पूर्णांक       PD_X = p[2];
    स्थिर पूर्णांक       PD_Y = p[3];
    स्थिर पूर्णांक       DL_X = p[4];
    स्थिर पूर्णांक       DL_Y = p[5];

    स्थिर पूर्णांक IW = निवेश->ne[0];
    स्थिर पूर्णांक IH = निवेश->ne[1];
    स्थिर पूर्णांक OW = dst->ne[0];
    स्थिर पूर्णांक OH = dst->ne[1];
    स्थिर पूर्णांक KW = kernel->ne[0];
    स्थिर पूर्णांक KH = kernel->ne[1];
    स्थिर पूर्णांक IC = निवेश->ne[2];
    स्थिर पूर्णांक OC = kernel->ne[3];
    स्थिर पूर्णांक B  = निवेश->ne[3];

    स्थिर int64_t     योग  = (int64_t) B * OC * OH * OW;
    स्थिर conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, योग };

    अगर (cwhn_layout) {
        अगर (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, true>(X_D, (स्थिर sycl::half *) K_D, Y_D, params, stream);
        } अन्यथा {
            conv2d_sycl<भग्न, true>(X_D, K_D, Y_D, params, stream);
        }
    } अन्यथा {
        अगर (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, false>(X_D, (स्थिर sycl::half *) K_D, Y_D, params, stream);
        } अन्यथा {
            conv2d_sycl<भग्न, false>(X_D, K_D, Y_D, params, stream);
        }
    }
}
