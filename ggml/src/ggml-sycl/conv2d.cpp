#અન્તર્ભાવ "conv2d.hpp"
#અન્તર્ભાવ "convert.hpp"

સંરચના conv2d_params {
    અચલ int64_t IW, IH;
    અચલ int64_t OW, OH;
    અચલ int64_t KW, KH;
    અચલ int64_t ST_X, ST_Y;
    અચલ int64_t PD_X, PD_Y;
    અચલ int64_t DL_X, DL_Y;
    અચલ int64_t IC, OC;
    અચલ int64_t B;
    અચલ int64_t TOTAL;
};

સંરચના conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

સ્થિર અંતઃરેખ int64_t conv2d_max64(int64_t a, int64_t b) {
    પાછા (a > b) ? a : b;
}

સ્થિર અંતઃરેખ int64_t conv2d_min64(int64_t a, int64_t b) {
    પાછા (a < b) ? a : b;
}

સ્થિર અંતઃરેખ conv2d_kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, અચલ conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    પાછા bounds;
}

સ્થિર અંતઃરેખ પૂર્ણાંક calculate_આગત_coord(int64_t out_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    પાછા out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
સ્થિર અંતઃરેખ int64_t whcn_આગત_index(int64_t n, int64_t c, int64_t y, int64_t x, અચલ conv2d_params & P) {
    પાછા n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

સ્થિર અંતઃરેખ int64_t whcn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, અચલ conv2d_params & P) {
    પાછા c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
}

સ્થિર અંતઃરેખ int64_t whcn_output_index(int64_t n, int64_t c, int64_t y, int64_t x, અચલ conv2d_params & P) {
    પાછા n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
સ્થિર અંતઃરેખ int64_t cwhn_આગત_index(int64_t n, int64_t c, int64_t y, int64_t x, અચલ conv2d_params & P) {
    પાછા ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

સ્થિર અંતઃરેખ int64_t cwhn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, અચલ conv2d_params & P) {
    પાછા ((ky * P.KW + kx) * P.IC + c_in) * P.OC + c_out;
}

template <typename T, તાર્કિક CWHN>
સ્થિર શૂન્ય conv2d_kernel(અચલ દશાંશ * આગત, અચલ T * kernel, દશાંશ * output,
                          અચલ conv2d_params P, અચલ sycl::nd_item<3> & item_ct1) {
    અચલ int64_t global_idx = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);

    જો (global_idx >= P.TOTAL) {
        પાછા;
    }

    અચલ int64_t out_x  = global_idx % P.OW;
    અચલ int64_t out_y  = (global_idx / P.OW) % P.OH;
    અચલ int64_t c_out  = (global_idx / (P.OW * P.OH)) % P.OC;
    અચલ int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    દશાંશ acc = 0.0f;

    અચલ conv2d_kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

    માટે (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        માટે (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            અચલ int64_t in_y = calculate_આગત_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            માટે (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                અચલ int64_t in_x = calculate_આગત_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);
                અચલ દશાંશ આગત_val  = આગત[CWHN ? cwhn_આગત_index(n, c_in, in_y, in_x, P) : whcn_આગત_index(n, c_in, in_y, in_x, P)];
                અચલ T     kernel_val = kernel[CWHN ? cwhn_kernel_index(c_out, c_in, ky, kx, P) : whcn_kernel_index(c_out, c_in, ky, kx, P)];
                acc += આગત_val * ggml_sycl_cast<દશાંશ>(kernel_val);
            }
        }
    }

    output[whcn_output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T, તાર્કિક CWHN>
સ્થિર શૂન્ય conv2d_sycl(અચલ દશાંશ * X_D, અચલ T * K_D, દશાંશ * Y_D,
                        અચલ conv2d_params P, અચલ queue_ptr & stream) {
    અચલ પૂર્ણાંક અંક_blocks = (P.TOTAL + SYCL_CONV2D_BLOCK_SIZE - 1) / SYCL_CONV2D_BLOCK_SIZE;
    અચલ sycl::range<3> block_dims(1, 1, SYCL_CONV2D_BLOCK_SIZE);
    અચલ sycl::range<3> block_nums(1, 1, અંક_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, item_ct1);
        });
}

શૂન્ય ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    અચલ ggml_tensor * kernel = dst->src[0];
    અચલ ggml_tensor * આગત  = dst->src[1];
    અચલ દશાંશ *       K_D    = (અચલ દશાંશ *) kernel->data;
    અચલ દશાંશ *       X_D    = (અચલ દશાંશ *) આગત->data;
    દશાંશ *             Y_D    = (દશાંશ *) dst->data;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    અચલ તાર્કિક whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(આગત);
    અચલ તાર્કિક cwhn = આગત->nb[2]  == ggml_type_size(આગત->type) &&
                      kernel->nb[3] == ggml_type_size(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    અચલ તાર્કિક cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(આગત->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(આગત->ne[2] == kernel->ne[2]);

    અચલ queue_ptr stream = ctx.stream();

    અચલ int32_t * p    = (અચલ int32_t *) dst->op_params;
    અચલ પૂર્ણાંક       ST_X = p[0];
    અચલ પૂર્ણાંક       ST_Y = p[1];
    અચલ પૂર્ણાંક       PD_X = p[2];
    અચલ પૂર્ણાંક       PD_Y = p[3];
    અચલ પૂર્ણાંક       DL_X = p[4];
    અચલ પૂર્ણાંક       DL_Y = p[5];

    અચલ પૂર્ણાંક IW = આગત->ne[0];
    અચલ પૂર્ણાંક IH = આગત->ne[1];
    અચલ પૂર્ણાંક OW = dst->ne[0];
    અચલ પૂર્ણાંક OH = dst->ne[1];
    અચલ પૂર્ણાંક KW = kernel->ne[0];
    અચલ પૂર્ણાંક KH = kernel->ne[1];
    અચલ પૂર્ણાંક IC = આગત->ne[2];
    અચલ પૂર્ણાંક OC = kernel->ne[3];
    અચલ પૂર્ણાંક B  = આગત->ne[3];

    અચલ int64_t     total  = (int64_t) B * OC * OH * OW;
    અચલ conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    જો (cwhn_layout) {
        જો (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, સાચું>(X_D, (અચલ sycl::half *) K_D, Y_D, params, stream);
        } નહીંતર {
            conv2d_sycl<દશાંશ, સાચું>(X_D, K_D, Y_D, params, stream);
        }
    } નહીંતર {
        જો (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, ખોટું>(X_D, (અચલ sycl::half *) K_D, Y_D, params, stream);
        } નહીંતર {
            conv2d_sycl<દશાંશ, ખોટું>(X_D, K_D, Y_D, params, stream);
        }
    }
}
