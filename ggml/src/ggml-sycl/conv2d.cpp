#ଅନ୍ତର୍ଭାଵ "conv2d.hpp"
#ଅନ୍ତର୍ଭାଵ "convert.hpp"

ସଂରଚନା conv2d_params {
    ଅଚଳ int64_t IW, IH;
    ଅଚଳ int64_t OW, OH;
    ଅଚଳ int64_t KW, KH;
    ଅଚଳ int64_t ST_X, ST_Y;
    ଅଚଳ int64_t PD_X, PD_Y;
    ଅଚଳ int64_t DL_X, DL_Y;
    ଅଚଳ int64_t IC, OC;
    ଅଚଳ int64_t B;
    ଅଚଳ int64_t TOTAL;
};

ସଂରଚନା conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ int64_t conv2d_max64(int64_t a, int64_t b) {
    ଫେରନ୍ତୁ (a > b) ? a : b;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ int64_t conv2d_min64(int64_t a, int64_t b) {
    ଫେରନ୍ତୁ (a < b) ? a : b;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ conv2d_kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, ଅଚଳ conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    ଫେରନ୍ତୁ bounds;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ପୂର୍ଣ୍ଣସଂଖ୍ୟା calculate_ନିବେଶ_coord(int64_t out_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    ଫେରନ୍ତୁ out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
ସ୍ଥିର ଧାଡ଼ିଭିତରେ int64_t whcn_ନିବେଶ_index(int64_t n, int64_t c, int64_t y, int64_t x, ଅଚଳ conv2d_params & P) {
    ଫେରନ୍ତୁ n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ int64_t whcn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, ଅଚଳ conv2d_params & P) {
    ଫେରନ୍ତୁ c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ int64_t whcn_output_index(int64_t n, int64_t c, int64_t y, int64_t x, ଅଚଳ conv2d_params & P) {
    ଫେରନ୍ତୁ n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
ସ୍ଥିର ଧାଡ଼ିଭିତରେ int64_t cwhn_ନିବେଶ_index(int64_t n, int64_t c, int64_t y, int64_t x, ଅଚଳ conv2d_params & P) {
    ଫେରନ୍ତୁ ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ int64_t cwhn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, ଅଚଳ conv2d_params & P) {
    ଫେରନ୍ତୁ ((ky * P.KW + kx) * P.IC + c_in) * P.OC + c_out;
}

template <typename T, ତାର୍କିକ CWHN>
ସ୍ଥିର ଶୂନ୍ୟ conv2d_kernel(ଅଚଳ ଦଶମିକ * ନିବେଶ, ଅଚଳ T * kernel, ଦଶମିକ * output,
                          ଅଚଳ conv2d_params P, ଅଚଳ sycl::nd_item<3> & item_ct1) {
    ଅଚଳ int64_t global_idx = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);

    ଯଦି (global_idx >= P.TOTAL) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ int64_t out_x  = global_idx % P.OW;
    ଅଚଳ int64_t out_y  = (global_idx / P.OW) % P.OH;
    ଅଚଳ int64_t c_out  = (global_idx / (P.OW * P.OH)) % P.OC;
    ଅଚଳ int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    ଦଶମିକ acc = 0.0f;

    ଅଚଳ conv2d_kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

    ପାଇଁ (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        ପାଇଁ (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            ଅଚଳ int64_t in_y = calculate_ନିବେଶ_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            ପାଇଁ (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                ଅଚଳ int64_t in_x = calculate_ନିବେଶ_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);
                ଅଚଳ ଦଶମିକ ନିବେଶ_val  = ନିବେଶ[CWHN ? cwhn_ନିବେଶ_index(n, c_in, in_y, in_x, P) : whcn_ନିବେଶ_index(n, c_in, in_y, in_x, P)];
                ଅଚଳ T     kernel_val = kernel[CWHN ? cwhn_kernel_index(c_out, c_in, ky, kx, P) : whcn_kernel_index(c_out, c_in, ky, kx, P)];
                acc += ନିବେଶ_val * ggml_sycl_cast<ଦଶମିକ>(kernel_val);
            }
        }
    }

    output[whcn_output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T, ତାର୍କିକ CWHN>
ସ୍ଥିର ଶୂନ୍ୟ conv2d_sycl(ଅଚଳ ଦଶମିକ * X_D, ଅଚଳ T * K_D, ଦଶମିକ * Y_D,
                        ଅଚଳ conv2d_params P, ଅଚଳ queue_ptr & stream) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_blocks = (P.TOTAL + SYCL_CONV2D_BLOCK_SIZE - 1) / SYCL_CONV2D_BLOCK_SIZE;
    ଅଚଳ sycl::range<3> block_dims(1, 1, SYCL_CONV2D_BLOCK_SIZE);
    ଅଚଳ sycl::range<3> block_nums(1, 1, ଅଙ୍କ_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, item_ct1);
        });
}

ଶୂନ୍ୟ ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ଅଚଳ ggml_tensor * kernel = dst->src[0];
    ଅଚଳ ggml_tensor * ନିବେଶ  = dst->src[1];
    ଅଚଳ ଦଶମିକ *       K_D    = (ଅଚଳ ଦଶମିକ *) kernel->data;
    ଅଚଳ ଦଶମିକ *       X_D    = (ଅଚଳ ଦଶମିକ *) ନିବେଶ->data;
    ଦଶମିକ *             Y_D    = (ଦଶମିକ *) dst->data;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    ଅଚଳ ତାର୍କିକ whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(ନିବେଶ);
    ଅଚଳ ତାର୍କିକ cwhn = ନିବେଶ->nb[2]  == ggml_type_size(ନିବେଶ->type) &&
                      kernel->nb[3] == ggml_type_size(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    ଅଚଳ ତାର୍କିକ cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(ନିବେଶ->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(ନିବେଶ->ne[2] == kernel->ne[2]);

    ଅଚଳ queue_ptr stream = ctx.stream();

    ଅଚଳ int32_t * p    = (ଅଚଳ int32_t *) dst->op_params;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ST_X = p[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       ST_Y = p[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       PD_X = p[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       PD_Y = p[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       DL_X = p[4];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       DL_Y = p[5];

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା IW = ନିବେଶ->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା IH = ନିବେଶ->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା OW = dst->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା OH = dst->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା KW = kernel->ne[0];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା KH = kernel->ne[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା IC = ନିବେଶ->ne[2];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା OC = kernel->ne[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା B  = ନିବେଶ->ne[3];

    ଅଚଳ int64_t     total  = (int64_t) B * OC * OH * OW;
    ଅଚଳ conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    ଯଦି (cwhn_layout) {
        ଯଦି (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, ସତ୍ୟ>(X_D, (ଅଚଳ sycl::half *) K_D, Y_D, params, stream);
        } ଅନ୍ୟଥା {
            conv2d_sycl<ଦଶମିକ, ସତ୍ୟ>(X_D, K_D, Y_D, params, stream);
        }
    } ଅନ୍ୟଥା {
        ଯଦି (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, ମିଥ୍ୟା>(X_D, (ଅଚଳ sycl::half *) K_D, Y_D, params, stream);
        } ଅନ୍ୟଥା {
            conv2d_sycl<ଦଶମିକ, ମିଥ୍ୟା>(X_D, K_D, Y_D, params, stream);
        }
    }
}
