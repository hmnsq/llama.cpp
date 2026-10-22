#అన్తర్భావ "conv2d-transpose.hpp"
#అన్తర్భావ "convert.hpp"

మూస <రకం_పేరు kernel_t>
స్థిర శూన్య conv2d_transpose_kernel(స్థిరరాశి దశాంశం * ఆదానం, స్థిరరాశి kernel_t * kernel, దశాంశం * output,
                                    స్థిరరాశి పూర్ణాంకం లోపల_w, స్థిరరాశి పూర్ణాంకం లోపల_h,
                                    స్థిరరాశి పూర్ణాంకం బహిర్గతం_w, స్థిరరాశి పూర్ణాంకం బహిర్గతం_h,
                                    స్థిరరాశి పూర్ణాంకం kernel_w, స్థిరరాశి పూర్ణాంకం kernel_h,
                                    స్థిరరాశి పూర్ణాంకం stride,
                                    స్థిరరాశి పూర్ణాంకం c_లోపల, స్థిరరాశి పూర్ణాంకం c_బహిర్గతం, స్థిరరాశి పూర్ణాంకం batches,
                                    స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం global_idx     = అంశం_ct1.get_local_id(2) +
                               అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
    స్థిరరాశి పూర్ణాంకం మొత్తం_elements = బహిర్గతం_w * బహిర్గతం_h * c_బహిర్గతం * batches;

    ఒకవేళ (global_idx >= మొత్తం_elements) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం బహిర్గతం_x = global_idx % బహిర్గతం_w;
    స్థిరరాశి పూర్ణాంకం బహిర్గతం_y = (global_idx / బహిర్గతం_w) % బహిర్గతం_h;
    స్థిరరాశి పూర్ణాంకం c_idx = (global_idx / (బహిర్గతం_w * బహిర్గతం_h)) % c_బహిర్గతం;
    స్థిరరాశి పూర్ణాంకం n_idx = global_idx / (బహిర్గతం_w * బహిర్గతం_h * c_బహిర్గతం);

    దశాంశం acc = 0.0f;

    కొరకు (పూర్ణాంకం c_లోపల_idx = 0; c_లోపల_idx < c_లోపల; ++c_లోపల_idx) {
        కొరకు (పూర్ణాంకం kh = 0; kh < kernel_h; ++kh) {
            పూర్ణాంకం లోపల_y = బహిర్గతం_y - kh;
            ఒకవేళ (లోపల_y < 0 || లోపల_y % stride) {
                కొనసాగు;
            }
            లోపల_y /= stride;
            ఒకవేళ (లోపల_y >= లోపల_h) {
                కొనసాగు;
            }

            కొరకు (పూర్ణాంకం kw = 0; kw < kernel_w; ++kw) {
                పూర్ణాంకం లోపల_x = బహిర్గతం_x - kw;
                ఒకవేళ (లోపల_x < 0 || లోపల_x % stride) {
                    కొనసాగు;
                }
                లోపల_x /= stride;
                ఒకవేళ (లోపల_x >= లోపల_w) {
                    కొనసాగు;
                }

                స్థిరరాశి పూర్ణాంకం ఆదానం_idx  = (లోపల_w * లోపల_h * c_లోపల) * n_idx + (లోపల_w * లోపల_h) * c_లోపల_idx + లోపల_w * లోపల_y + లోపల_x;
                స్థిరరాశి పూర్ణాంకం kernel_idx = (kernel_h * kernel_w * c_బహిర్గతం) * c_లోపల_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += ఆదానం[ఆదానం_idx] * ggml_sycl_cast<దశాంశం>(kernel[kernel_idx]);
            }
        }
    }

    output[(బహిర్గతం_w * బహిర్గతం_h * c_బహిర్గతం) * n_idx + (బహిర్గతం_w * బహిర్గతం_h) * c_idx + బహిర్గతం_w * బహిర్గతం_y + బహిర్గతం_x] = acc;
}

మూస <రకం_పేరు kernel_t>
స్థిర శూన్య conv2d_transpose_sycl(స్థిరరాశి దశాంశం * ఆదానం_d, స్థిరరాశి kernel_t * kernel_d, దశాంశం * output_d,
                                   స్థిరరాశి పూర్ణాంకం లోపల_w, స్థిరరాశి పూర్ణాంకం లోపల_h,
                                   స్థిరరాశి పూర్ణాంకం బహిర్గతం_w, స్థిరరాశి పూర్ణాంకం బహిర్గతం_h,
                                   స్థిరరాశి పూర్ణాంకం kernel_w, స్థిరరాశి పూర్ణాంకం kernel_h,
                                   స్థిరరాశి పూర్ణాంకం stride,
                                   స్థిరరాశి పూర్ణాంకం c_లోపల, స్థిరరాశి పూర్ణాంకం c_బహిర్గతం, స్థిరరాశి పూర్ణాంకం batches,
                                   స్థిరరాశి వరుస_ptr & stream) {
    స్థిరరాశి పూర్ణాంకం మొత్తం      = బహిర్గతం_w * బహిర్గతం_h * c_బహిర్గతం * batches;
    స్థిరరాశి పూర్ణాంకం అంకె_ఖండాలు = (మొత్తం + SYCL_CONV2_D_TRANSPOSE_ఖండం_కొలత - 1) / SYCL_CONV2_D_TRANSPOSE_ఖండం_కొలత;
    స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, SYCL_CONV2_D_TRANSPOSE_ఖండం_కొలత);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, 1, అంకె_ఖండాలు);
    stream->parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            conv2d_transpose_kernel<kernel_t>(ఆదానం_d, kernel_d, output_d,
                                             లోపల_w, లోపల_h, బహిర్గతం_w, బహిర్గతం_h, kernel_w, kernel_h,
                                             stride, c_లోపల, c_బహిర్గతం, batches, అంశం_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
శూన్య ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    స్థిరరాశి ggml_tensor * kernel = dst->src[0];
    స్థిరరాశి ggml_tensor * ఆదానం  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(ఆదానం->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(ఆదానం));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    స్థిరరాశి దశాంశం * ఆదానం_d  = (స్థిరరాశి దశాంశం *) ఆదానం->దత్తాంశం;
    దశాంశం *       output_d = (దశాంశం *) dst->దత్తాంశం;
    స్థిరరాశి శూన్య *  kernel_d = kernel->దత్తాంశం;

    స్థిరరాశి పూర్ణాంకం ఆదానం_w      = ఆదానం->ne[0];
    స్థిరరాశి పూర్ణాంకం ఆదానం_h      = ఆదానం->ne[1];
    స్థిరరాశి పూర్ణాంకం channels_లోపల  = ఆదానం->ne[2];
    స్థిరరాశి పూర్ణాంకం batches      = ఆదానం->ne[3];
    స్థిరరాశి పూర్ణాంకం output_w     = dst->ne[0];
    స్థిరరాశి పూర్ణాంకం output_h     = dst->ne[1];
    స్థిరరాశి పూర్ణాంకం channels_బహిర్గతం = kernel->ne[2];
    స్థిరరాశి పూర్ణాంకం kernel_w     = kernel->ne[0];
    స్థిరరాశి పూర్ణాంకం kernel_h     = kernel->ne[1];
    స్థిరరాశి పూర్ణాంకం stride       = dst->op_params[0];

    GGML_ASSERT(channels_లోపల == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    స్థిరరాశి వరుస_ptr stream = ctx.stream();

    ఒకవేళ (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(ఆదానం_d, (స్థిరరాశి sycl::half *) kernel_d, output_d,
                                          ఆదానం_w, ఆదానం_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_లోపల, channels_బహిర్గతం, batches, stream);
    } లేకపోతే {
        conv2d_transpose_sycl<దశాంశం>(ఆదానం_d, (స్థిరరాశి దశాంశం *) kernel_d, output_d,
                                     ఆదానం_w, ఆదానం_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_లోపల, channels_బహిర్గతం, batches, stream);
    }
}
