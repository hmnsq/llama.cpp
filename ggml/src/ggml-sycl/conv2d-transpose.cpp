#അന്തര്ഭാവ "conv2d-transpose.hpp"
#അന്തര്ഭാവ "convert.hpp"

template <typename kernel_t>
സ്ഥിതം ശൂന്യം conv2d_transpose_kernel(സ്ഥിരം ദശാംശം * നിവേശം, സ്ഥിരം kernel_t * kernel, ദശാംശം * output,
                                    സ്ഥിരം പൂർണ്ണസംഖ്യ in_w, സ്ഥിരം പൂർണ്ണസംഖ്യ in_h,
                                    സ്ഥിരം പൂർണ്ണസംഖ്യ out_w, സ്ഥിരം പൂർണ്ണസംഖ്യ out_h,
                                    സ്ഥിരം പൂർണ്ണസംഖ്യ kernel_w, സ്ഥിരം പൂർണ്ണസംഖ്യ kernel_h,
                                    സ്ഥിരം പൂർണ്ണസംഖ്യ stride,
                                    സ്ഥിരം പൂർണ്ണസംഖ്യ c_in, സ്ഥിരം പൂർണ്ണസംഖ്യ c_out, സ്ഥിരം പൂർണ്ണസംഖ്യ batches,
                                    സ്ഥിരം sycl::nd_item<3> & item_ct1) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    സ്ഥിരം പൂർണ്ണസംഖ്യ total_elements = out_w * out_h * c_out * batches;

    എങ്കിൽ (global_idx >= total_elements) {
        തിരികെ;
    }

    സ്ഥിരം പൂർണ്ണസംഖ്യ out_x = global_idx % out_w;
    സ്ഥിരം പൂർണ്ണസംഖ്യ out_y = (global_idx / out_w) % out_h;
    സ്ഥിരം പൂർണ്ണസംഖ്യ c_idx = (global_idx / (out_w * out_h)) % c_out;
    സ്ഥിരം പൂർണ്ണസംഖ്യ n_idx = global_idx / (out_w * out_h * c_out);

    ദശാംശം acc = 0.0f;

    വേണ്ടി (പൂർണ്ണസംഖ്യ c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        വേണ്ടി (പൂർണ്ണസംഖ്യ kh = 0; kh < kernel_h; ++kh) {
            പൂർണ്ണസംഖ്യ in_y = out_y - kh;
            എങ്കിൽ (in_y < 0 || in_y % stride) {
                തുടരുക;
            }
            in_y /= stride;
            എങ്കിൽ (in_y >= in_h) {
                തുടരുക;
            }

            വേണ്ടി (പൂർണ്ണസംഖ്യ kw = 0; kw < kernel_w; ++kw) {
                പൂർണ്ണസംഖ്യ in_x = out_x - kw;
                എങ്കിൽ (in_x < 0 || in_x % stride) {
                    തുടരുക;
                }
                in_x /= stride;
                എങ്കിൽ (in_x >= in_w) {
                    തുടരുക;
                }

                സ്ഥിരം പൂർണ്ണസംഖ്യ നിവേശം_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                സ്ഥിരം പൂർണ്ണസംഖ്യ kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += നിവേശം[നിവേശം_idx] * ggml_sycl_cast<ദശാംശം>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

template <typename kernel_t>
സ്ഥിതം ശൂന്യം conv2d_transpose_sycl(സ്ഥിരം ദശാംശം * നിവേശം_d, സ്ഥിരം kernel_t * kernel_d, ദശാംശം * output_d,
                                   സ്ഥിരം പൂർണ്ണസംഖ്യ in_w, സ്ഥിരം പൂർണ്ണസംഖ്യ in_h,
                                   സ്ഥിരം പൂർണ്ണസംഖ്യ out_w, സ്ഥിരം പൂർണ്ണസംഖ്യ out_h,
                                   സ്ഥിരം പൂർണ്ണസംഖ്യ kernel_w, സ്ഥിരം പൂർണ്ണസംഖ്യ kernel_h,
                                   സ്ഥിരം പൂർണ്ണസംഖ്യ stride,
                                   സ്ഥിരം പൂർണ്ണസംഖ്യ c_in, സ്ഥിരം പൂർണ്ണസംഖ്യ c_out, സ്ഥിരം പൂർണ്ണസംഖ്യ batches,
                                   സ്ഥിരം queue_ptr & stream) {
    സ്ഥിരം പൂർണ്ണസംഖ്യ total      = out_w * out_h * c_out * batches;
    സ്ഥിരം പൂർണ്ണസംഖ്യ അക്കം_blocks = (total + SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE;
    സ്ഥിരം sycl::range<3> block_dims(1, 1, SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE);
    സ്ഥിരം sycl::range<3> block_nums(1, 1, അക്കം_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_transpose_kernel<kernel_t>(നിവേശം_d, kernel_d, output_d,
                                             in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                             stride, c_in, c_out, batches, item_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
ശൂന്യം ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    സ്ഥിരം ggml_tensor * kernel = dst->src[0];
    സ്ഥിരം ggml_tensor * നിവേശം  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(നിവേശം->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(നിവേശം));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    സ്ഥിരം ദശാംശം * നിവേശം_d  = (സ്ഥിരം ദശാംശം *) നിവേശം->data;
    ദശാംശം *       output_d = (ദശാംശം *) dst->data;
    സ്ഥിരം ശൂന്യം *  kernel_d = kernel->data;

    സ്ഥിരം പൂർണ്ണസംഖ്യ നിവേശം_w      = നിവേശം->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ നിവേശം_h      = നിവേശം->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ channels_in  = നിവേശം->ne[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ batches      = നിവേശം->ne[3];
    സ്ഥിരം പൂർണ്ണസംഖ്യ output_w     = dst->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ output_h     = dst->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ channels_out = kernel->ne[2];
    സ്ഥിരം പൂർണ്ണസംഖ്യ kernel_w     = kernel->ne[0];
    സ്ഥിരം പൂർണ്ണസംഖ്യ kernel_h     = kernel->ne[1];
    സ്ഥിരം പൂർണ്ണസംഖ്യ stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    സ്ഥിരം queue_ptr stream = ctx.stream();

    എങ്കിൽ (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(നിവേശം_d, (സ്ഥിരം sycl::half *) kernel_d, output_d,
                                          നിവേശം_w, നിവേശം_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } അല്ലെങ്കിൽ {
        conv2d_transpose_sycl<ദശാംശം>(നിവേശം_d, (സ്ഥിരം ദശാംശം *) kernel_d, output_d,
                                     നിവേശം_w, നിവേശം_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
