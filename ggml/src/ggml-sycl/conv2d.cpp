#அந்தர்பாவ "conv2d.hpp"
#அந்தர்பாவ "convert.hpp"

கட்டமைப்பு conv2d_params {
    மாறிலி int64_t IW, IH;
    மாறிலி int64_t OW, OH;
    மாறிலி int64_t KW, KH;
    மாறிலி int64_t ST_X, ST_Y;
    மாறிலி int64_t PD_X, PD_Y;
    மாறிலி int64_t DL_X, DL_Y;
    மாறிலி int64_t IC, OC;
    மாறிலி int64_t B;
    மாறிலி int64_t TOTAL;
};

கட்டமைப்பு conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

நிலையான உள்வரி int64_t conv2d_max64(int64_t a, int64_t b) {
    திரும்பு (a > b) ? a : b;
}

நிலையான உள்வரி int64_t conv2d_min64(int64_t a, int64_t b) {
    திரும்பு (a < b) ? a : b;
}

நிலையான உள்வரி conv2d_kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, மாறிலி conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    திரும்பு bounds;
}

நிலையான உள்வரி முழு calculate_உள்ளீடு_coord(int64_t out_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    திரும்பு out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
நிலையான உள்வரி int64_t whcn_உள்ளீடு_index(int64_t n, int64_t c, int64_t y, int64_t x, மாறிலி conv2d_params & P) {
    திரும்பு n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

நிலையான உள்வரி int64_t whcn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, மாறிலி conv2d_params & P) {
    திரும்பு c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
}

நிலையான உள்வரி int64_t whcn_output_index(int64_t n, int64_t c, int64_t y, int64_t x, மாறிலி conv2d_params & P) {
    திரும்பு n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
நிலையான உள்வரி int64_t cwhn_உள்ளீடு_index(int64_t n, int64_t c, int64_t y, int64_t x, மாறிலி conv2d_params & P) {
    திரும்பு ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

நிலையான உள்வரி int64_t cwhn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, மாறிலி conv2d_params & P) {
    திரும்பு ((ky * P.KW + kx) * P.IC + c_in) * P.OC + c_out;
}

template <typename T, தருக்க CWHN>
நிலையான வெறுமை conv2d_kernel(மாறிலி தசம * உள்ளீடு, மாறிலி T * kernel, தசம * output,
                          மாறிலி conv2d_params P, மாறிலி sycl::nd_item<3> & item_ct1) {
    மாறிலி int64_t global_idx = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);

    எனில் (global_idx >= P.TOTAL) {
        திரும்பு;
    }

    மாறிலி int64_t out_x  = global_idx % P.OW;
    மாறிலி int64_t out_y  = (global_idx / P.OW) % P.OH;
    மாறிலி int64_t c_out  = (global_idx / (P.OW * P.OH)) % P.OC;
    மாறிலி int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    தசம acc = 0.0f;

    மாறிலி conv2d_kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

    க்கு (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        க்கு (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            மாறிலி int64_t in_y = calculate_உள்ளீடு_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            க்கு (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                மாறிலி int64_t in_x = calculate_உள்ளீடு_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);
                மாறிலி தசம உள்ளீடு_val  = உள்ளீடு[CWHN ? cwhn_உள்ளீடு_index(n, c_in, in_y, in_x, P) : whcn_உள்ளீடு_index(n, c_in, in_y, in_x, P)];
                மாறிலி T     kernel_val = kernel[CWHN ? cwhn_kernel_index(c_out, c_in, ky, kx, P) : whcn_kernel_index(c_out, c_in, ky, kx, P)];
                acc += உள்ளீடு_val * ggml_sycl_cast<தசம>(kernel_val);
            }
        }
    }

    output[whcn_output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T, தருக்க CWHN>
நிலையான வெறுமை conv2d_sycl(மாறிலி தசம * X_D, மாறிலி T * K_D, தசம * Y_D,
                        மாறிலி conv2d_params P, மாறிலி queue_ptr & stream) {
    மாறிலி முழு இலக்கம்_blocks = (P.TOTAL + SYCL_CONV2D_BLOCK_SIZE - 1) / SYCL_CONV2D_BLOCK_SIZE;
    மாறிலி sycl::range<3> block_dims(1, 1, SYCL_CONV2D_BLOCK_SIZE);
    மாறிலி sycl::range<3> block_nums(1, 1, இலக்கம்_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, item_ct1);
        });
}

வெறுமை ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    மாறிலி ggml_tensor * kernel = dst->src[0];
    மாறிலி ggml_tensor * உள்ளீடு  = dst->src[1];
    மாறிலி தசம *       K_D    = (மாறிலி தசம *) kernel->data;
    மாறிலி தசம *       X_D    = (மாறிலி தசம *) உள்ளீடு->data;
    தசம *             Y_D    = (தசம *) dst->data;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    மாறிலி தருக்க whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(உள்ளீடு);
    மாறிலி தருக்க cwhn = உள்ளீடு->nb[2]  == ggml_type_size(உள்ளீடு->type) &&
                      kernel->nb[3] == ggml_type_size(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    மாறிலி தருக்க cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(உள்ளீடு->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(உள்ளீடு->ne[2] == kernel->ne[2]);

    மாறிலி queue_ptr stream = ctx.stream();

    மாறிலி int32_t * p    = (மாறிலி int32_t *) dst->op_params;
    மாறிலி முழு       ST_X = p[0];
    மாறிலி முழு       ST_Y = p[1];
    மாறிலி முழு       PD_X = p[2];
    மாறிலி முழு       PD_Y = p[3];
    மாறிலி முழு       DL_X = p[4];
    மாறிலி முழு       DL_Y = p[5];

    மாறிலி முழு IW = உள்ளீடு->ne[0];
    மாறிலி முழு IH = உள்ளீடு->ne[1];
    மாறிலி முழு OW = dst->ne[0];
    மாறிலி முழு OH = dst->ne[1];
    மாறிலி முழு KW = kernel->ne[0];
    மாறிலி முழு KH = kernel->ne[1];
    மாறிலி முழு IC = உள்ளீடு->ne[2];
    மாறிலி முழு OC = kernel->ne[3];
    மாறிலி முழு B  = உள்ளீடு->ne[3];

    மாறிலி int64_t     total  = (int64_t) B * OC * OH * OW;
    மாறிலி conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    எனில் (cwhn_layout) {
        எனில் (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, உண்மை>(X_D, (மாறிலி sycl::half *) K_D, Y_D, params, stream);
        } இல்லையெனில் {
            conv2d_sycl<தசம, உண்மை>(X_D, K_D, Y_D, params, stream);
        }
    } இல்லையெனில் {
        எனில் (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, பொய்>(X_D, (மாறிலி sycl::half *) K_D, Y_D, params, stream);
        } இல்லையெனில் {
            conv2d_sycl<தசம, பொய்>(X_D, K_D, Y_D, params, stream);
        }
    }
}
