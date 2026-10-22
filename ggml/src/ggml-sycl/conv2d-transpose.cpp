#ಅನ್ತರ್ಭಾವ "conv2d-transpose.hpp"
#ಅನ್ತರ್ಭಾವ "convert.hpp"

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು kernel_t>
ಸ್ಥಿರ ಶೂನ್ಯ conv2d_transpose_kernel(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಒಳಹರಿವು, ಸ್ಥಿರಾಂಕ kernel_t * kernel, ದಶಮಾಂಶ * output,
                                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ_w, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ_h,
                                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_w, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_h,
                                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kernel_w, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kernel_h,
                                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride,
                                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c_ಒಳ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c_ಹೊರಹೋಗು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ batches,
                                    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ global_idx     = ಅಂಶ_ct1.get_local_id(2) +
                               ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು_elements = ಹೊರಹೋಗು_w * ಹೊರಹೋಗು_h * c_ಹೊರಹೋಗು * batches;

    ಆದರೆ (global_idx >= ಒಟ್ಟು_elements) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_x = global_idx % ಹೊರಹೋಗು_w;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_y = (global_idx / ಹೊರಹೋಗು_w) % ಹೊರಹೋಗು_h;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c_idx = (global_idx / (ಹೊರಹೋಗು_w * ಹೊರಹೋಗು_h)) % c_ಹೊರಹೋಗು;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n_idx = global_idx / (ಹೊರಹೋಗು_w * ಹೊರಹೋಗು_h * c_ಹೊರಹೋಗು);

    ದಶಮಾಂಶ acc = 0.0f;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c_ಒಳ_idx = 0; c_ಒಳ_idx < c_ಒಳ; ++c_ಒಳ_idx) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kh = 0; kh < kernel_h; ++kh) {
            ಪೂರ್ಣಾಂಕ ಒಳ_y = ಹೊರಹೋಗು_y - kh;
            ಆದರೆ (ಒಳ_y < 0 || ಒಳ_y % stride) {
                ಮುಂದುವರಿ;
            }
            ಒಳ_y /= stride;
            ಆದರೆ (ಒಳ_y >= ಒಳ_h) {
                ಮುಂದುವರಿ;
            }

            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ kw = 0; kw < kernel_w; ++kw) {
                ಪೂರ್ಣಾಂಕ ಒಳ_x = ಹೊರಹೋಗು_x - kw;
                ಆದರೆ (ಒಳ_x < 0 || ಒಳ_x % stride) {
                    ಮುಂದುವರಿ;
                }
                ಒಳ_x /= stride;
                ಆದರೆ (ಒಳ_x >= ಒಳ_w) {
                    ಮುಂದುವರಿ;
                }

                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_idx  = (ಒಳ_w * ಒಳ_h * c_ಒಳ) * n_idx + (ಒಳ_w * ಒಳ_h) * c_ಒಳ_idx + ಒಳ_w * ಒಳ_y + ಒಳ_x;
                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kernel_idx = (kernel_h * kernel_w * c_ಹೊರಹೋಗು) * c_ಒಳ_idx + (kernel_h * kernel_w) * c_idx +
                                       kernel_w * kh + kw;

                acc += ಒಳಹರಿವು[ಒಳಹರಿವು_idx] * ggml_sycl_cast<ದಶಮಾಂಶ>(kernel[kernel_idx]);
            }
        }
    }

    output[(ಹೊರಹೋಗು_w * ಹೊರಹೋಗು_h * c_ಹೊರಹೋಗು) * n_idx + (ಹೊರಹೋಗು_w * ಹೊರಹೋಗು_h) * c_idx + ಹೊರಹೋಗು_w * ಹೊರಹೋಗು_y + ಹೊರಹೋಗು_x] = acc;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು kernel_t>
ಸ್ಥಿರ ಶೂನ್ಯ conv2d_transpose_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಒಳಹರಿವು_d, ಸ್ಥಿರಾಂಕ kernel_t * kernel_d, ದಶಮಾಂಶ * output_d,
                                   ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ_w, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ_h,
                                   ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_w, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_h,
                                   ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kernel_w, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kernel_h,
                                   ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride,
                                   ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c_ಒಳ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ c_ಹೊರಹೋಗು, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ batches,
                                   ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr & stream) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಟ್ಟು      = ಹೊರಹೋಗು_w * ಹೊರಹೋಗು_h * c_ಹೊರಹೋಗು * batches;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = (ಒಟ್ಟು + SYCL_CONV2_D_TRANSPOSE_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_CONV2_D_TRANSPOSE_ಖಂಡ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, SYCL_CONV2_D_TRANSPOSE_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಅಂಕೆ_ಖಂಡಗಳು);
    stream->parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            conv2d_transpose_kernel<kernel_t>(ಒಳಹರಿವು_d, kernel_d, output_d,
                                             ಒಳ_w, ಒಳ_h, ಹೊರಹೋಗು_w, ಹೊರಹೋಗು_h, kernel_w, kernel_h,
                                             stride, c_ಒಳ, c_ಹೊರಹೋಗು, batches, ಅಂಶ_ct1);
        });
}

// input:  (W, H, C_in, N)
// kernel: (W, H, C_out, C_in)
// output: (W, H, C_out, N)
ಶೂನ್ಯ ggml_sycl_op_conv2d_transpose(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ಸ್ಥಿರಾಂಕ ggml_tensor * kernel = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * ಒಳಹರಿವು  = dst->src[1];

    GGML_ASSERT(kernel->type == GGML_TYPE_F16 || kernel->type == GGML_TYPE_F32);
    GGML_ASSERT(ಒಳಹರಿವು->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(ಒಳಹರಿವು));
    GGML_ASSERT(ggml_is_contiguous(kernel));
    GGML_ASSERT(ggml_is_contiguous(dst));

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಒಳಹರಿವು_d  = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ಒಳಹರಿವು->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *       output_d = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *  kernel_d = kernel->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_w      = ಒಳಹರಿವು->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_h      = ಒಳಹರಿವು->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channels_ಒಳ  = ಒಳಹರಿವು->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ batches      = ಒಳಹರಿವು->ne[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ output_w     = dst->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ output_h     = dst->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ channels_ಹೊರಹೋಗು = kernel->ne[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kernel_w     = kernel->ne[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ kernel_h     = kernel->ne[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ stride       = dst->op_params[0];

    GGML_ASSERT(channels_ಒಳ == kernel->ne[3]);
    GGML_ASSERT(stride > 0);

    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr stream = ctx.stream();

    ಆದರೆ (kernel->type == GGML_TYPE_F16) {
        conv2d_transpose_sycl<sycl::half>(ಒಳಹರಿವು_d, (ಸ್ಥಿರಾಂಕ sycl::half *) kernel_d, output_d,
                                          ಒಳಹರಿವು_w, ಒಳಹರಿವು_h, output_w, output_h, kernel_w, kernel_h,
                                          stride, channels_ಒಳ, channels_ಹೊರಹೋಗು, batches, stream);
    } ಇಲ್ಲದಿದ್ದರೆ {
        conv2d_transpose_sycl<ದಶಮಾಂಶ>(ಒಳಹರಿವು_d, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) kernel_d, output_d,
                                     ಒಳಹರಿವು_w, ಒಳಹರಿವು_h, output_w, output_h, kernel_w, kernel_h,
                                     stride, channels_ಒಳ, channels_ಹೊರಹೋಗು, batches, stream);
    }
}
