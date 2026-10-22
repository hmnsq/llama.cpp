#अन्तर्भाव "softmax.hpp"
#अन्तर्भाव <cstdint>
#अन्तर्भाव <utility>
#अन्तर्भाव <cmath>


प्रतिरूपम् <प्रकारनाम T> स्थिर __dpct_inline__ भिन्न t2f32(T val) {
    प्रत्यागम (भिन्न) val;
}

प्रतिरूपम् <> भिन्न __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  प्रत्यागम sycl::vec<sycl::half, 1>(val)
      .convert<भिन्न, sycl::rounding_बहुलकम्::automatic>()[0];
}

संरचना soft_max_params {

    int64_t nheads;
    uint32_t n_शीर्ष_log2;
    int64_t ncols;
    int64_t nrows_x;
    int64_t nrows_y;
    int64_t ne00;
    int64_t ne01;
    int64_t ne02;
    int64_t ne03;
    int64_t nb11;
    int64_t nb12;
    int64_t nb13;

    int64_t ne12;
    int64_t ne13;
    भिन्न scale;
    भिन्न max_bias;
    भिन्न m0;
    भिन्न m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#यदि_परिभाषितम् __clang__
#निर्देशः clang diagnostic प्रक्षिप
#निर्देशः clang diagnostic ignored "-Wpass-failed"
#यदि_अन्तः // __clang__
प्रतिरूपम् <तार्किक use_shared, पूर्णाङ्क ncols_template, पूर्णाङ्क खण्डम्_आकार_template, प्रकारनाम T>
स्थिर शून्य soft_max_f32(ध्रुव भिन्न *         x,
                         ध्रुव T *             आवरणम्,
                         ध्रुव भिन्न *         sinks,
                         भिन्न *               dst,
                         ध्रुव soft_max_params p,
                         uint8_t *             dpct_local) {
    स्वचालित      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव पूर्णाङ्क ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    ध्रुव पूर्णाङ्क खण्डम्_आकार_3 = खण्डम्_आकार_template == 0
                               ? item_ct1.get_local_range(2)
                               : खण्डम्_आकार_template;
    ध्रुव पूर्णाङ्क nthreads = खण्डम्_आकार_3;
    ध्रुव पूर्णाङ्क nwarps = nthreads / WARP_आकार;
    ध्रुव size_t nreduce = nwarps / WARP_आकार;

    ध्रुव पूर्णाङ्क tid = item_ct1.get_local_id(2);

    ध्रुव int64_t i03 = item_ct1.get_गणः(0);
    ध्रुव int64_t i02 = item_ct1.get_गणः(1);
    ध्रुव int64_t i01 = item_ct1.get_गणः(2);

    //TODO: noncontigous inputs/outputs
    ध्रुव पूर्णाङ्क rowx = item_ct1.get_गणः(2) +
                     item_ct1.get_गणः(1) * item_ct1.get_गणः_range(2) +
                     item_ct1.get_गणः(0) * item_ct1.get_गणः_range(2) *
                         item_ct1.get_गणः_range(1);

    ध्रुव int64_t i11 = i01;
    ध्रुव int64_t i12 = i02 % p.ne12;
    ध्रुव int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    आवरणम् += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / परिमाण(T) * (आवरणम् != शून्यसूचकः);
    dst  += int64_t(rowx)*ncols;

    ध्रुव पूर्णाङ्क warp_id = item_ct1.get_local_id(2) / WARP_आकार;
    ध्रुव पूर्णाङ्क lane_id = item_ct1.get_local_id(2) % WARP_आकार;

    ध्रुव भिन्न नतिः = get_alibi_नतिः(p.max_bias, i02, p.n_शीर्ष_log2, p.m0, p.m1);

    भिन्न * buf_iw = (भिन्न *) dpct_local;

    // shared memory buffer to cache values between iterations:
    भिन्न *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_आकार) : dst;
    भिन्न max_val = sinks ? sinks[i02] : -INFINITY;
#निर्देशः unroll
    प्रति (पूर्णाङ्क col0 = 0; col0 < ncols; col0 += खण्डम्_आकार_3) {
        ध्रुव पूर्णाङ्क col = col0 + tid;

        यदि (ncols_template == 0 && col >= ncols) {
            भङ्ग;
        }

        ध्रुव भिन्न val = x[col]*p.scale + (आवरणम् ? नतिः*t2f32(आवरणम्[col]) : 0.0f);

        vals[col] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_आकार>(max_val);

    यदि (खण्डम्_आकार_3 > WARP_आकार) {
        यदि (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        item_ct1.barrier();

        max_val = -INFINITY;
        प्रति (पूर्णाङ्क i = lane_id; i < nwarps; i += WARP_आकार) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_आकार>(max_val);
    }
    भिन्न tmp = 0.0f; // partial sum

#निर्देशः unroll
    प्रति (पूर्णाङ्क col0 = 0; col0 < ncols; col0 += खण्डम्_आकार_3) {
        ध्रुव पूर्णाङ्क col = col0 + tid;

        यदि (ncols_template == 0 && col >= ncols) {
            भङ्ग;
        }

        ध्रुव भिन्न val = sycl::native::घातांकः(sycl::max(vals[col] - max_val, -80.0f));
        tmp += val;
        vals[col] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_योग<WARP_आकार>(tmp);
    यदि (खण्डम्_आकार_3 > WARP_आकार) {
        item_ct1.barrier();
        यदि (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            प्रति (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_आकार] = 0.f;
            }
        }
        item_ct1.barrier();

        यदि (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        item_ct1.barrier();

        tmp = buf_iw[lane_id];
        प्रति (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_आकार];
        }
        tmp = warp_reduce_योग<WARP_आकार>(tmp);
    }
    यदि (sinks) {
        tmp += sycl::native::घातांकः(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    ध्रुव भिन्न inv_योग = 1.0f / tmp;

#निर्देशः unroll
    प्रति (पूर्णाङ्क col0 = 0; col0 < ncols; col0 += खण्डम्_आकार_3) {
        ध्रुव पूर्णाङ्क col = col0 + tid;

        यदि (ncols_template == 0 && col >= ncols) {
            प्रत्यागम;
        }

        dst[col] = vals[col] * inv_योग;
    }
}
#यदि_परिभाषितम् __clang__
#निर्देशः clang diagnostic निष्कासय
#यदि_अन्तः // __clang__

स्थिर शून्य soft_max_पृष्ठभागः_f32(ध्रुव भिन्न *grad, ध्रुव भिन्न *dstf, भिन्न *dst,
                              ध्रुव पूर्णाङ्क ncols, ध्रुव भिन्न scale) {
    स्वचालित      item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव पूर्णाङ्क tid      = item_ct1.get_local_id(2);
    ध्रुव पूर्णाङ्क rowx     = item_ct1.get_गणः(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    भिन्न dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    प्रति (पूर्णाङ्क col = tid; col < ncols; col += WARP_आकार) {
        dgf_dot += dstf[col]*grad[col];
    }

    dgf_dot = warp_reduce_योग<WARP_आकार>(dgf_dot);

    प्रति (पूर्णाङ्क col = tid; col < ncols; col += WARP_आकार) {
        dst[col] = scale * (grad[col] - dgf_dot) * dstf[col];
    }
}

प्रतिरूपम् <पूर्णाङ्क... Ns, प्रकारनाम T>
स्थिर शून्य launch_soft_max_kernels(ध्रुव भिन्न *           x,
                                    ध्रुव T *               आवरणम्,
                                    ध्रुव भिन्न *           sinks,
                                    भिन्न *                 dst,
                                    ध्रुव soft_max_params & p,
                                    dpct::queue_ptr         stream,
                                    dpct::dim3              खण्डम्_dims,
                                    dpct::dim3              खण्डम्_nums,
                                    size_t                  nbytes_shared)
{
    स्वचालित launch_kernel = [=](स्वचालित I) -> तार्किक {
        constexpr पूर्णाङ्क ncols = decltype(I)::value;
        constexpr पूर्णाङ्क खण्डम् = (ncols > 1024 ? 1024 : ncols);
        यदि (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                    [=](sycl::nd_item<3> item_ct1) [[sycl::reqd_sub_गणः_आकार(
                        WARP_आकार)]] {
                        soft_max_f32<सत्य, ncols, खण्डम्>(
                            x, आवरणम्, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_अप्रयुक्तम्(item_ct1);
                    });
            });
            प्रत्यागम सत्य;
        }
        प्रत्यागम असत्य;
    };

    // unary fold over launch_kernel
    यदि ((launch_kernel(std::integral_constant<पूर्णाङ्क, Ns>{}) || ...)) {
        प्रत्यागम;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1)
                [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                    soft_max_f32<सत्य, 0, 0>(
                        x, आवरणम्, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_अप्रयुक्तम्(item_ct1);
                });
    });
}

प्रतिरूपम् <प्रकारनाम T>
स्थिर शून्य soft_max_f32_sycl(ध्रुव भिन्न *x, ध्रुव T *आवरणम्,
                              ध्रुव भिन्न *sinks, भिन्न *dst,
                              ध्रुव soft_max_params &params,
                              dpct::queue_ptr stream, पूर्णाङ्क device) {
    पूर्णाङ्क nth = WARP_आकार;
    पूर्णाङ्क max_खण्डम्_आकार = ggml_sycl_info().max_work_गणः_sizes[device];
    ध्रुव int64_t ncols_x = params.ncols;

    यावत् (nth < ncols_x && nth < max_खण्डम्_आकार) nth *= 2;
    यदि (nth>max_खण्डम्_आकार) nth = max_खण्डम्_आकार;

    ध्रुव dpct::dim3 खण्डम्_dims(nth, 1, 1);
    ध्रुव dpct::dim3 खण्डम्_nums(params.ne01, params.ne02, params.ne03);
    ध्रुव size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_आकार) + WARP_आकार) * परिमाण(भिन्न);

    ध्रुव पूर्णाङ्क id       = get_current_device_id();
    ध्रुव size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    यदि (nbytes_shared <= smpbo && ncols_x <= max_खण्डम्_आकार) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, आवरणम्, sinks, dst, params, stream, खण्डम्_dims, खण्डम्_nums,
            nbytes_shared);
    } अन्यथा {
        ध्रुव size_t nbytes_shared_नीचः = WARP_आकार * परिमाण(भिन्न);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_नीचः), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                [=](sycl::nd_item<3> item_ct1)
                    [[sycl::reqd_sub_गणः_आकार(WARP_आकार)]] {
                    soft_max_f32<असत्य, 0, 0>(
                        x, आवरणम्, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_अप्रयुक्तम्(item_ct1);
                });
        });
    }
}

स्थिर शून्य soft_max_पृष्ठभागः_f32_sycl(ध्रुव भिन्न *   grad,
                                   ध्रुव भिन्न *   dstf,
                                   भिन्न *         dst,
                                   ध्रुव पूर्णाङ्क       ncols,
                                   ध्रुव पूर्णाङ्क       nrows,
                                   ध्रुव भिन्न     scale,
                                   dpct::queue_ptr stream) {
    ध्रुव dpct::dim3 खण्डम्_dims(WARP_आकार, 1, 1);
    ध्रुव dpct::dim3 खण्डम्_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                         [=](sycl::nd_item<3> item_ct1) {
                             soft_max_पृष्ठभागः_f32(grad, dstf, dst, ncols, scale);
                             GGML_अप्रयुक्तम्(item_ct1);
                         });
}

शून्य ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];
    ध्रुव ggml_tensor * src2 = dst->src[2];

    ध्रुव भिन्न * src0_d = (ध्रुव भिन्न *) src0->दत्त;
    ध्रुव शून्य  * src1_d = src1 ? (ध्रुव शून्य *) src1->दत्त : शून्यसूचकः;
    ध्रुव शून्य  * src2_d = src2 ? (ध्रुव शून्य *) src2->दत्त : शून्यसूचकः;
    भिन्न       *  dst_d = (भिन्न *) dst->दत्त;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    ध्रुव int64_t nrows_x = ggml_nrows(src0);
    ध्रुव int64_t nrows_y = src0->ne[1];

    ध्रुव int64_t ne00 = src0->ne[0];

    भिन्न scale    = 1.0f;
    भिन्न max_bias = 0.0f;

    स्मृतिं_प्रतिलिख(&scale,    (ध्रुव भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&max_bias, (ध्रुव भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

    ध्रुव तार्किक use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    ध्रुव int64_t nb11 = src1 ? src1->nb[1] : 1;
    ध्रुव int64_t nb12 = src1 ? src1->nb[2] : 1;
    ध्रुव int64_t nb13 = src1 ? src1->nb[3] : 1;

    ध्रुव int64_t ne12 = src1 ? src1->ne[2] : 1;
    ध्रुव int64_t ne13 = src1 ? src1->ne[3] : 1;

    ध्रुव uint32_t n_शीर्ष      = src0->ne[2];
    ध्रुव uint32_t n_शीर्ष_log2 = 1u << (uint32_t) floorf(log2f((भिन्न) n_शीर्ष));

    ध्रुव भिन्न m0 = powf(2.0f, -(max_bias       ) / n_शीर्ष_log2);
    ध्रुव भिन्न m1 = powf(2.0f, -(max_bias / 2.0f) / n_शीर्ष_log2);


    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_शीर्ष_log2 = n_शीर्ष_log2;
    params.ncols = ne00;
    params.nrows_x = nrows_x;
    params.nrows_y = nrows_y;
    params.ne00 = src0->ne[0];
    params.ne01 = src0->ne[1];
    params.ne02 = src0->ne[2];
    params.ne03 = src0->ne[3];
    params.nb11 = nb11;
    params.nb12 = nb12;
    params.nb13 = nb13;
    params.ne12 = ne12;
    params.ne13 = ne13;
    params.scale = scale;
    params.max_bias = max_bias;
    params.m0 = m0;
    params.m1 = m1;

    यदि (use_f16) {
        soft_max_f32_sycl(src0_d, (ध्रुव sycl::half *)src1_d,
                          (ध्रुव भिन्न *)src2_d, dst_d, params, stream,
                          ctx.device);
    } अन्यथा {
        soft_max_f32_sycl(src0_d, (ध्रुव भिन्न *)src1_d, (ध्रुव भिन्न *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

शून्य ggml_sycl_op_soft_max_पृष्ठभागः(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ध्रुव ggml_tensor * src0 = dst->src[0]; // grad
    ध्रुव ggml_tensor * src1 = dst->src[1]; // forward pass output

    ध्रुव भिन्न * src0_d = (ध्रुव भिन्न *) src0->दत्त;
    ध्रुव भिन्न * src1_d = (ध्रुव भिन्न *) src1->दत्त;
    भिन्न       * dst_d  = (भिन्न       *) dst->दत्त;

    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    ध्रुव int64_t ncols = src0->ne[0];
    ध्रुव int64_t nrows = ggml_nrows(src0);

    भिन्न scale    = 1.0f;
    भिन्न max_bias = 0.0f;

    स्मृतिं_प्रतिलिख(&scale,    (ध्रुव भिन्न *) dst->op_params + 0, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&max_bias, (ध्रुव भिन्न *) dst->op_params + 1, परिमाण(भिन्न));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_पृष्ठभागः_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
