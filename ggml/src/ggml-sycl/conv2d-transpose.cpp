#અન્તર્ભાવ "conv2d-transpose.hpp"
#અન્તર્ભાવ "convert.hpp"

template <typename kernel_t>
સ્થિર શૂન્ય conv2d_transpose_kernel(અચલ દશાંશ * આગત, અચલ kernel_t * kernel, દશાંશ * output,
                                    અચલ પૂર્ણાંક in_w, અચલ પૂર્ણાંક in_h,
                                    અચલ પૂર્ણાંક out_w, અચલ પૂર્ણાંક out_h,
                                    અચલ પૂર્ણાંક kernel_w, અચલ પૂર્ણાંક kernel_h,
                                    અચલ પૂર્ણાંક stride,
                                    અચલ પૂર્ણાંક c_in, અચલ પૂર્ણાંક c_out, અચલ પૂર્ણાંક batches,
                                    અચલ sycl::nd_item<3> & item_ct1) {
    અચલ પૂર્ણાંક global_idx     = item_ct1.get_local_id(2) +
                               item_ct1.get_group(2) * item_ct1.get_local_range(2);
    અચલ પૂર્ણાંક total_elements = out_w * out_h * c_out * batches;

    જો (global_idx >= total_elements) {
        પાછા;
    }

    અચલ પૂર્ણાંક out_x = global_idx % out_w;
    અચલ પૂર્ણાંક out_y = (global_idx / out_w) % out_h;
    અચલ પૂર્ણાંક c_idx = (global_idx / (out_w * out_h)) % c_out;
    અચલ પૂર્ણાંક n_idx = global_idx / (out_w * out_h * c_out);

    દશાંશ acc = 0.0f;

    માટે (પૂર્ણાંક c_in_idx = 0; c_in_idx < c_in; ++c_in_idx) {
        માટે (પૂર્ણાંક kh = 0; kh < kernel_h; ++kh) {
            પૂર્ણાંક in_y = out_y - kh;
            જો (in_y < 0 || in_y % stride) {
                ચાલુ;
            }
            in_y /= stride;
            જો (in_y >= in_h) {
                ચાલુ;
            }

            માટે (પૂર્ણાંક kw = 0; kw < kernel_w; ++kw) {
                પૂર્ણાંક in_x = out_x - kw;
                જો (in_x < 0 || in_x % stride) {
                    ચાલુ;
                }
                in_x /= stride;
                જો (in_x >= in_w) {
                    ચાલુ;
                }

                અચલ પૂર્ણાંક આગત_idx  = (in_w * in_h * c_in) * n_idx + (in_w * in_h) * c_in_idx + in_w * in_y + in_x;
                અચલ પૂર્ણાંક kernel_idx = (kernel_h * kernel_w * c_out) * c_in_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += આગત[આગત_idx] * ggml_sycl_cast<દશાંશ>(kernel[kernel_idx]);
            }
        }
    }

    output[(out_w * out_h * c_out) * n_idx + (out_w * out_h) * c_idx + out_w * out_y + out_x] = acc;
}

template <typename kernel_t>
સ્થિર શૂન્ય conv2d_transpose_sycl(અચલ દશાંશ * આગત_d, અચલ kernel_t * kernel_d, દશાંશ * output_d,
                                   અચલ પૂર્ણાંક in_w, અચલ પૂર્ણાંક in_h,
                                   અચલ પૂર્ણાંક out_w, અચલ પૂર્ણાંક out_h,
                                   અચલ પૂર્ણાંક kernel_w, અચલ પૂર્ણાંક kernel_h,
                                   અચલ પૂર્ણાંક stride,
                                   અચલ પૂર્ણાંક c_in, અચલ પૂર્ણાંક c_out, અચલ પૂર્ણાંક batches,
                                   અચલ queue_ptr & stream) {
    અચલ પૂર્ણાંક total      = out_w * out_h * c_out * batches;
    અચલ પૂર્ણાંક અંક_blocks = (total + SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE - 1) / SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE;
    અચલ sycl::range<3> block_dims(1, 1, SYCL_CONV2D_TRANSPOSE_BLOCK_SIZE);
    અચલ sycl::range<3> block_nums(1, 1, અંક_blocks);
    stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv2d_transpose_kernel<kernel_t>(આગત_d, kernel_d, output_d,
                                             in_w, in_h, out_w, out_h, kernel_w, kernel_h,
                                             stride, c_in, c_out, batches, item_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
શૂન્ય ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    અચલ ggml_tensor * kernel = dst->src[0];
    અચલ ggml_tensor * આગત  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(આગત->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(આગત));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    અચલ દશાંશ * આગત_d  = (અચલ દશાંશ *) આગત->data;
    દશાંશ *       output_d = (દશાંશ *) dst->data;
    અચલ શૂન્ય *  kernel_d = kernel->data;

    અચલ પૂર્ણાંક આગત_w      = આગત->ne[0];
    અચલ પૂર્ણાંક આગત_h      = આગત->ne[1];
    અચલ પૂર્ણાંક channels_in  = આગત->ne[2];
    અચલ પૂર્ણાંક batches      = આગત->ne[3];
    અચલ પૂર્ણાંક output_w     = dst->ne[0];
    અચલ પૂર્ણાંક output_h     = dst->ne[1];
    અચલ પૂર્ણાંક channels_out = kernel->ne[2];
    અચલ પૂર્ણાંક kernel_w     = kernel->ne[0];
    અચલ પૂર્ણાંક kernel_h     = kernel->ne[1];
    અચલ પૂર્ણાંક stride       = dst->op_params[0];

    GGML_ASSERT(channels_in == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    અચલ queue_ptr stream = ctx.stream();

    જો (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(આગત_d, (અચલ sycl::half *) kernel_d, output_d,
                                          આગત_w, આગત_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_in, channels_out, batches, stream);
    } નહીંતર {
        conv2d_transpose_sycl<દશાંશ>(આગત_d, (અચલ દશાંશ *) kernel_d, output_d,
                                     આગત_w, આગત_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_in, channels_out, batches, stream);
    }
}
