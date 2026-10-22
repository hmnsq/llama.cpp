#അന്തര്ഭാവ "conv2d.hpp"
#അന്തര്ഭാവ "convert.hpp"

ഘടന conv2d_params {
    സ്ഥിരം int64_t IW, IH;
    സ്ഥിരം int64_t OW, OH;
    സ്ഥിരം int64_t KW, KH;
    സ്ഥിരം int64_t ST_X, ST_Y;
    സ്ഥിരം int64_t PD_X, PD_Y;
    സ്ഥിരം int64_t DL_X, DL_Y;
    സ്ഥിരം int64_t IC, OC;
    സ്ഥിരം int64_t B;
    സ്ഥിരം int64_t TOTAL;
};

ഘടന conv2d_kernel_bounds {
    int64_t y_min, y_max;
    int64_t x_min, x_max;
};

സ്ഥിതം വരിയിൽ int64_t conv2d_max64(int64_t a, int64_t b) {
    തിരികെ (a > b) ? a : b;
}

സ്ഥിതം വരിയിൽ int64_t conv2d_min64(int64_t a, int64_t b) {
    തിരികെ (a < b) ? a : b;
}

സ്ഥിതം വരിയിൽ conv2d_kernel_bounds calculate_kernel_bounds(int64_t out_x, int64_t out_y, സ്ഥിരം conv2d_params & P) {
    conv2d_kernel_bounds bounds;
    bounds.y_min = conv2d_max64(0, (P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.y_max = conv2d_min64(P.KH, (P.IH + P.PD_Y - out_y * P.ST_Y + P.DL_Y - 1) / P.DL_Y);
    bounds.x_min = conv2d_max64(0, (P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    bounds.x_max = conv2d_min64(P.KW, (P.IW + P.PD_X - out_x * P.ST_X + P.DL_X - 1) / P.DL_X);
    തിരികെ bounds;
}

സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ calculate_നിവേശം_coord(int64_t out_coord, int64_t kern_coord, int64_t stride,
                                        int64_t dilation, int64_t padding) {
    തിരികെ out_coord * stride + kern_coord * dilation - padding;
}

// whcn layout helpers (matching ggml tensor memory order)
സ്ഥിതം വരിയിൽ int64_t whcn_നിവേശം_index(int64_t n, int64_t c, int64_t y, int64_t x, സ്ഥിരം conv2d_params & P) {
    തിരികെ n * (P.IC * P.IW * P.IH) + c * P.IW * P.IH + y * P.IW + x;
}

സ്ഥിതം വരിയിൽ int64_t whcn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, സ്ഥിരം conv2d_params & P) {
    തിരികെ c_out * (P.IC * P.KH * P.KW) + c_in * (P.KH * P.KW) + ky * P.KW + kx;
}

സ്ഥിതം വരിയിൽ int64_t whcn_output_index(int64_t n, int64_t c, int64_t y, int64_t x, സ്ഥിരം conv2d_params & P) {
    തിരികെ n * (P.OC * P.OW * P.OH) + c * P.OW * P.OH + y * P.OW + x;
}

// cwhn layout helpers: buffers are [N][H][W][C] input and [KH][KW][IC][OC] kernel
സ്ഥിതം വരിയിൽ int64_t cwhn_നിവേശം_index(int64_t n, int64_t c, int64_t y, int64_t x, സ്ഥിരം conv2d_params & P) {
    തിരികെ ((n * P.IH + y) * P.IW + x) * P.IC + c;
}

സ്ഥിതം വരിയിൽ int64_t cwhn_kernel_index(int64_t c_out, int64_t c_in, int64_t ky, int64_t kx, സ്ഥിരം conv2d_params & P) {
    തിരികെ ((ky * P.KW + kx) * P.IC + c_in) * P.OC + c_out;
}

template <typename T, തർക്കം CWHN>
സ്ഥിതം ശൂന്യം conv2d_kernel(സ്ഥിരം ദശാംശം * നിവേശം, സ്ഥിരം T * kernel, ദശാംശം * output,
                          സ്ഥിരം conv2d_params P, സ്ഥിരം sycl::nd_item<3> & item_ct1) {
    സ്ഥിരം int64_t global_idx = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);

    എങ്കിൽ (global_idx >= P.TOTAL) {
        തിരികെ;
    }

    സ്ഥിരം int64_t out_x  = global_idx % P.OW;
    സ്ഥിരം int64_t out_y  = (global_idx / P.OW) % P.OH;
    സ്ഥിരം int64_t c_out  = (global_idx / (P.OW * P.OH)) % P.OC;
    സ്ഥിരം int64_t n      = global_idx / (P.OW * P.OH * P.OC);

    ദശാംശം acc = 0.0f;

    സ്ഥിരം conv2d_kernel_bounds bounds = calculate_kernel_bounds(out_x, out_y, P);

    വേണ്ടി (int64_t c_in = 0; c_in < P.IC; ++c_in) {
        വേണ്ടി (int64_t ky = bounds.y_min; ky < bounds.y_max; ++ky) {
            സ്ഥിരം int64_t in_y = calculate_നിവേശം_coord(out_y, ky, P.ST_Y, P.DL_Y, P.PD_Y);
            വേണ്ടി (int64_t kx = bounds.x_min; kx < bounds.x_max; ++kx) {
                സ്ഥിരം int64_t in_x = calculate_നിവേശം_coord(out_x, kx, P.ST_X, P.DL_X, P.PD_X);
                സ്ഥിരം ദശാംശം നിവേശം_val  = നിവേശം[CWHN ? cwhn_നിവേശം_index(n, c_in, in_y, in_x, P) : whcn_നിവേശം_index(n, c_in, in_y, in_x, P)];
                സ്ഥിരം T     kernel_val = kernel[CWHN ? cwhn_kernel_index(c_out, c_in, ky, kx, P) : whcn_kernel_index(c_out, c_in, ky, kx, P)];
                acc += നിവേശം_val * ggml_sycl_cast<ദശാംശം>(kernel_val);
            }
        }
    }

    output[whcn_output_index(n, c_out, out_y, out_x, P)] = acc;
}

template <typename T, തർക്കം CWHN>
സ്ഥിതം ശൂന്യം conv2d_sycl(സ്ഥിരം ദശാംശം * X_D, സ്ഥിരം T * K_D, ദശാംശം * Y_D,
                        സ്ഥിരം conv2d_params P, സ്ഥിരം queue_ptr & stream) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = (P.TOTAL + SYCL_CONV2D_BLOCK_SIZE - 1) / SYCL_CONV2D_BLOCK_SIZE;
    സ്ഥിരം sycl::range<3> block_dims(1, 1, SYCL_CONV2D_BLOCK_SIZE);
    സ്ഥിരം sycl::range<3> block_nums(1, 1, അക്കം_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_kernel<T, CWHN>(X_D, K_D, Y_D, P, item_ct1);
        });
}

ശൂന്യം ggml_sycl_op_conv2d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    സ്ഥിരം ggml_tensor * kernel = dst->src[0];
    സ്ഥിരം ggml_tensor * നിവേശം  = dst->src[1];
    സ്ഥിരം ദശാംശം *       K_D    = (സ്ഥിരം ദശാംശം *) kernel->data;
    സ്ഥിരം ദശാംശം *       X_D    = (സ്ഥിരം ദശാംശം *) നിവേശം->data;
    ദശാംശം *             Y_D    = (ദശാംശം *) dst->data;

    // WHCN: contiguous [KW,KH,IC,OC] kernel and [W,H,C,N] input
    // CWHN: channel-fastest buffer [N][H][W][C] input and [KH][KW][IC][OC] kernel (OC fastest)
    സ്ഥിരം തർക്കം whcn = ggml_is_contiguous(kernel) && ggml_is_contiguous(നിവേശം);
    സ്ഥിരം തർക്കം cwhn = നിവേശം->nb[2]  == ggml_type_size(നിവേശം->type) &&
                      kernel->nb[3] == ggml_type_size(kernel->type);
    GGML_ASSERT(whcn || cwhn);
    സ്ഥിരം തർക്കം cwhn_layout = !whcn && cwhn; // prefer whcn when both match (degenerate size-1 dims)

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(നിവേശം->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    // same number of input channels
    GGML_ASSERT(നിവേശം->ne[2] == kernel->ne[2]);

    സ്ഥിരം queue_ptr stream = ctx.stream();

    സ്ഥിരം int32_t * p    = (സ്ഥിരം int32_t *) dst->op_params;
    സ്ഥിരം പൂർണ്ണസംഖ്യ       ST_X = p[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ       ST_Y = p[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ       PD_X = p[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ       PD_Y = p[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ       DL_X = p[4];
    സ്ഥിരം പൂർണ്ണസംഖ്യ       DL_Y = p[5];

    സ്ഥിരം പൂർണ്ണസംഖ്യ IW = നിവേശം->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ IH = നിവേശം->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ OW = dst->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ OH = dst->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ KW = kernel->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ KH = kernel->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ IC = നിവേശം->ne[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ OC = kernel->ne[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ B  = നിവേശം->ne[3];

    സ്ഥിരം int64_t     total  = (int64_t) B * OC * OH * OW;
    സ്ഥിരം conv2d_params params = { IW, IH, OW, OH, KW, KH, ST_X, ST_Y, PD_X, PD_Y, DL_X, DL_Y, IC, OC, B, total };

    എങ്കിൽ (cwhn_layout) {
        എങ്കിൽ (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, സത്യം>(X_D, (സ്ഥിരം sycl::half *) K_D, Y_D, params, stream);
        } അല്ലെങ്കിൽ {
            conv2d_sycl<ദശാംശം, സത്യം>(X_D, K_D, Y_D, params, stream);
        }
    } അല്ലെങ്കിൽ {
        എങ്കിൽ (kernel->type == GGML_TYPE_F16) {
            conv2d_sycl<sycl::half, അസത്യം>(X_D, (സ്ഥിരം sycl::half *) K_D, Y_D, params, stream);
        } അല്ലെങ്കിൽ {
            conv2d_sycl<ദശാംശം, അസത്യം>(X_D, K_D, Y_D, params, stream);
        }
    }
}
