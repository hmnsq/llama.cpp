#అన్తర్భావ "softmax.hpp"
#అన్తర్భావ <cstdint>
#అన్తర్భావ <utility>
#అన్తర్భావ <cmath>


మూస <రకం_పేరు T> స్థిర __dpct_inline__ దశాంశం t2f32(T val) {
    తిరిగి (దశాంశం) val;
}

మూస <> దశాంశం __dpct_inline__ t2f32<sycl::half>(sycl::half val) {
  తిరిగి sycl::vec<sycl::half, 1>(val)
      .convert<దశాంశం, sycl::rounding_బహుళకం::automatic>()[0];
}

నిర్మాణం soft_max_params {

    int64_t nheads;
    uint32_t n_తల_log2;
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
    దశాంశం scale;
    దశాంశం max_bias;
    దశాంశం m0;
    దశాంశం m1;
};

// When ncols_template == 0 the bounds for the loops in this function are not known and can't be unrolled.
// As we want to keep pragma unroll for all other cases we suppress the clang transformation warning here.
#నిర్వచితమైతే __clang__
#సూచన clang diagnostic నెట్టు
#సూచన clang diagnostic ignored "-Wpass-failed"
#అయితే_ముగింపు // __clang__
మూస <తార్కిక use_shared, పూర్ణాంకం ncols_template, పూర్ణాంకం ఖండం_కొలత_template, రకం_పేరు T>
స్థిర శూన్య soft_max_f32(స్థిరరాశి దశాంశం *         x,
                         స్థిరరాశి T *             ముసుగు,
                         స్థిరరాశి దశాంశం *         sinks,
                         దశాంశం *               dst,
                         స్థిరరాశి soft_max_params p,
                         uint8_t *             dpct_local) {
    స్వయంచాలక      అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి పూర్ణాంకం ncols    = ncols_template == 0 ? p.ncols : ncols_template;
    స్థిరరాశి పూర్ణాంకం ఖండం_కొలత_3 = ఖండం_కొలత_template == 0
                               ? అంశం_ct1.get_local_range(2)
                               : ఖండం_కొలత_template;
    స్థిరరాశి పూర్ణాంకం nthreads = ఖండం_కొలత_3;
    స్థిరరాశి పూర్ణాంకం nwarps = nthreads / WARP_కొలత;
    స్థిరరాశి size_t nreduce = nwarps / WARP_కొలత;

    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2);

    స్థిరరాశి int64_t i03 = అంశం_ct1.get_సమూహం(0);
    స్థిరరాశి int64_t i02 = అంశం_ct1.get_సమూహం(1);
    స్థిరరాశి int64_t i01 = అంశం_ct1.get_సమూహం(2);

    //TODO: noncontigous inputs/outputs
    స్థిరరాశి పూర్ణాంకం rowx = అంశం_ct1.get_సమూహం(2) +
                     అంశం_ct1.get_సమూహం(1) * అంశం_ct1.get_సమూహం_range(2) +
                     అంశం_ct1.get_సమూహం(0) * అంశం_ct1.get_సమూహం_range(2) *
                         అంశం_ct1.get_సమూహం_range(1);

    స్థిరరాశి int64_t i11 = i01;
    స్థిరరాశి int64_t i12 = i02 % p.ne12;
    స్థిరరాశి int64_t i13 = i03 % p.ne13;

    x    += int64_t(rowx)*ncols;
    ముసుగు += (i11*p.nb11 + i12*p.nb12 + i13*p.nb13) / పరిమాణం(T) * (ముసుగు != శూన్యసూచిక);
    dst  += int64_t(rowx)*ncols;

    స్థిరరాశి పూర్ణాంకం warp_id = అంశం_ct1.get_local_id(2) / WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం lane_id = అంశం_ct1.get_local_id(2) % WARP_కొలత;

    స్థిరరాశి దశాంశం వాలు = get_alibi_వాలు(p.max_bias, i02, p.n_తల_log2, p.m0, p.m1);

    దశాంశం * buf_iw = (దశాంశం *) dpct_local;

    // shared memory buffer to cache values between iterations:
    దశాంశం *vals = use_shared ? buf_iw + sycl::max(nwarps, WARP_కొలత) : dst;
    దశాంశం max_val = sinks ? sinks[i02] : -INFINITY;
#సూచన unroll
    కొరకు (పూర్ణాంకం col0 = 0; col0 < ncols; col0 += ఖండం_కొలత_3) {
        స్థిరరాశి పూర్ణాంకం నిలువు = col0 + tid;

        ఒకవేళ (ncols_template == 0 && నిలువు >= ncols) {
            ఆపు;
        }

        స్థిరరాశి దశాంశం val = x[నిలువు]*p.scale + (ముసుగు ? వాలు*t2f32(ముసుగు[నిలువు]) : 0.0f);

        vals[నిలువు] = val;
        max_val   = sycl::max(max_val, val);
    }
    // find the max value in the block
    max_val = warp_reduce_max<WARP_కొలత>(max_val);

    ఒకవేళ (ఖండం_కొలత_3 > WARP_కొలత) {
        ఒకవేళ (lane_id == 0) {
            buf_iw[warp_id] = max_val;
        }
        అంశం_ct1.barrier();

        max_val = -INFINITY;
        కొరకు (పూర్ణాంకం i = lane_id; i < nwarps; i += WARP_కొలత) {
            max_val = sycl::max(max_val, buf_iw[i]);
        }
        max_val = warp_reduce_max<WARP_కొలత>(max_val);
    }
    దశాంశం tmp = 0.0f; // partial sum

#సూచన unroll
    కొరకు (పూర్ణాంకం col0 = 0; col0 < ncols; col0 += ఖండం_కొలత_3) {
        స్థిరరాశి పూర్ణాంకం నిలువు = col0 + tid;

        ఒకవేళ (ncols_template == 0 && నిలువు >= ncols) {
            ఆపు;
        }

        స్థిరరాశి దశాంశం val = sycl::native::ఘాతాంకం(sycl::max(vals[నిలువు] - max_val, -80.0f));
        tmp += val;
        vals[నిలువు] = val;
    }
    // find the sum of exps in the block
    tmp = warp_reduce_కూడిక<WARP_కొలత>(tmp);
    ఒకవేళ (ఖండం_కొలత_3 > WARP_కొలత) {
        అంశం_ct1.barrier();
        ఒకవేళ (warp_id == 0) {
            buf_iw[lane_id] = 0.0f;
            కొరకు (size_t i = 1; i < nreduce; i += 1) {
                buf_iw[lane_id + i * WARP_కొలత] = 0.f;
            }
        }
        అంశం_ct1.barrier();

        ఒకవేళ (lane_id == 0) {
            buf_iw[warp_id] = tmp;
        }
        అంశం_ct1.barrier();

        tmp = buf_iw[lane_id];
        కొరకు (size_t i = 1; i < nreduce; i += 1) {
            tmp += buf_iw[lane_id + i * WARP_కొలత];
        }
        tmp = warp_reduce_కూడిక<WARP_కొలత>(tmp);
    }
    ఒకవేళ (sinks) {
        tmp += sycl::native::ఘాతాంకం(sycl::max(sinks[i02] - max_val, -80.0f));
    }
    స్థిరరాశి దశాంశం inv_కూడిక = 1.0f / tmp;

#సూచన unroll
    కొరకు (పూర్ణాంకం col0 = 0; col0 < ncols; col0 += ఖండం_కొలత_3) {
        స్థిరరాశి పూర్ణాంకం నిలువు = col0 + tid;

        ఒకవేళ (ncols_template == 0 && నిలువు >= ncols) {
            తిరిగి;
        }

        dst[నిలువు] = vals[నిలువు] * inv_కూడిక;
    }
}
#నిర్వచితమైతే __clang__
#సూచన clang diagnostic తీయి
#అయితే_ముగింపు // __clang__

స్థిర శూన్య soft_max_వెనుక_f32(స్థిరరాశి దశాంశం *grad, స్థిరరాశి దశాంశం *dstf, దశాంశం *dst,
                              స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి దశాంశం scale) {
    స్వయంచాలక      అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి పూర్ణాంకం tid      = అంశం_ct1.get_local_id(2);
    స్థిరరాశి పూర్ణాంకం rowx     = అంశం_ct1.get_సమూహం(2);

    grad += int64_t(rowx)*ncols;
    dstf += int64_t(rowx)*ncols;
    dst  += int64_t(rowx)*ncols;

    దశాంశం dgf_dot = 0.0f; // dot product of dst from forward pass and gradients

    కొరకు (పూర్ణాంకం నిలువు = tid; నిలువు < ncols; నిలువు += WARP_కొలత) {
        dgf_dot += dstf[నిలువు]*grad[నిలువు];
    }

    dgf_dot = warp_reduce_కూడిక<WARP_కొలత>(dgf_dot);

    కొరకు (పూర్ణాంకం నిలువు = tid; నిలువు < ncols; నిలువు += WARP_కొలత) {
        dst[నిలువు] = scale * (grad[నిలువు] - dgf_dot) * dstf[నిలువు];
    }
}

మూస <పూర్ణాంకం... Ns, రకం_పేరు T>
స్థిర శూన్య launch_soft_max_kernels(స్థిరరాశి దశాంశం *           x,
                                    స్థిరరాశి T *               ముసుగు,
                                    స్థిరరాశి దశాంశం *           sinks,
                                    దశాంశం *                 dst,
                                    స్థిరరాశి soft_max_params & p,
                                    dpct::వరుస_ptr         stream,
                                    dpct::dim3              ఖండం_dims,
                                    dpct::dim3              ఖండం_nums,
                                    size_t                  nbytes_shared)
{
    స్వయంచాలక launch_kernel = [=](స్వయంచాలక I) -> తార్కిక {
        constexpr పూర్ణాంకం ncols = decltype(I)::value;
        constexpr పూర్ణాంకం ఖండం = (ncols > 1024 ? 1024 : ncols);
        ఒకవేళ (p.ncols == ncols) {
            stream->submit([&](sycl::handler &cgh) {
                sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                    sycl::range<1>(nbytes_shared), cgh);

                cgh.parallel_for(
                    sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                    [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(
                        WARP_కొలత)]] {
                        soft_max_f32<నిజం, ncols, ఖండం>(
                            x, ముసుగు, sinks, dst, p,
                            dpct_local_acc_ct1
                                .get_multi_ptr<sycl::access::decorated::no>()
                                .get());
                        GGML_వాడని(అంశం_ct1);
                    });
            });
            తిరిగి నిజం;
        }
        తిరిగి అబద్ధం;
    };

    // unary fold over launch_kernel
    ఒకవేళ ((launch_kernel(std::integral_constant<పూర్ణాంకం, Ns>{}) || ...)) {
        తిరిగి;
    }

    stream->submit([&](sycl::handler &cgh) {
        sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
            sycl::range<1>(nbytes_shared), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    soft_max_f32<నిజం, 0, 0>(
                        x, ముసుగు, sinks, dst, p,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_వాడని(అంశం_ct1);
                });
    });
}

మూస <రకం_పేరు T>
స్థిర శూన్య soft_max_f32_sycl(స్థిరరాశి దశాంశం *x, స్థిరరాశి T *ముసుగు,
                              స్థిరరాశి దశాంశం *sinks, దశాంశం *dst,
                              స్థిరరాశి soft_max_params &params,
                              dpct::వరుస_ptr stream, పూర్ణాంకం device) {
    పూర్ణాంకం nth = WARP_కొలత;
    పూర్ణాంకం max_ఖండం_కొలత = ggml_sycl_info().max_work_సమూహం_sizes[device];
    స్థిరరాశి int64_t ncols_x = params.ncols;

    అయితే (nth < ncols_x && nth < max_ఖండం_కొలత) nth *= 2;
    ఒకవేళ (nth>max_ఖండం_కొలత) nth = max_ఖండం_కొలత;

    స్థిరరాశి dpct::dim3 ఖండం_dims(nth, 1, 1);
    స్థిరరాశి dpct::dim3 ఖండం_nums(params.ne01, params.ne02, params.ne03);
    స్థిరరాశి size_t nbytes_shared =
        (GGML_PAD(ncols_x, WARP_కొలత) + WARP_కొలత) * పరిమాణం(దశాంశం);

    స్థిరరాశి పూర్ణాంకం id       = get_current_device_id();
    స్థిరరాశి size_t smpbo = ggml_sycl_info().devices[id].smpbo;

    ఒకవేళ (nbytes_shared <= smpbo && ncols_x <= max_ఖండం_కొలత) {
        launch_soft_max_kernels<32, 64, 128, 256, 512, 1024, 2048, 4096>(
            x, ముసుగు, sinks, dst, params, stream, ఖండం_dims, ఖండం_nums,
            nbytes_shared);
    } లేకపోతే {
        స్థిరరాశి size_t nbytes_shared_దిగువ = WARP_కొలత * పరిమాణం(దశాంశం);

        stream->submit([&](sycl::handler &cgh) {
            sycl::local_accessor<uint8_t, 1> dpct_local_acc_ct1(
                sycl::range<1>(nbytes_shared_దిగువ), cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                    [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    soft_max_f32<అబద్ధం, 0, 0>(
                        x, ముసుగు, sinks, dst, params,
                        dpct_local_acc_ct1
                            .get_multi_ptr<sycl::access::decorated::no>()
                            .get());
                    GGML_వాడని(అంశం_ct1);
                });
        });
    }
}

స్థిర శూన్య soft_max_వెనుక_f32_sycl(స్థిరరాశి దశాంశం *   grad,
                                   స్థిరరాశి దశాంశం *   dstf,
                                   దశాంశం *         dst,
                                   స్థిరరాశి పూర్ణాంకం       ncols,
                                   స్థిరరాశి పూర్ణాంకం       nrows,
                                   స్థిరరాశి దశాంశం     scale,
                                   dpct::వరుస_ptr stream) {
    స్థిరరాశి dpct::dim3 ఖండం_dims(WARP_కొలత, 1, 1);
    స్థిరరాశి dpct::dim3 ఖండం_nums(nrows, 1, 1);

    stream->parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> అంశం_ct1) {
                             soft_max_వెనుక_f32(grad, dstf, dst, ncols, scale);
                             GGML_వాడని(అంశం_ct1);
                         });
}

శూన్య ggml_sycl_op_soft_max(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    స్థిరరాశి ggml_tensor * src2 = dst->src[2];

    స్థిరరాశి దశాంశం * src0_d = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం;
    స్థిరరాశి శూన్య  * src1_d = src1 ? (స్థిరరాశి శూన్య *) src1->దత్తాంశం : శూన్యసూచిక;
    స్థిరరాశి శూన్య  * src2_d = src2 ? (స్థిరరాశి శూన్య *) src2->దత్తాంశం : శూన్యసూచిక;
    దశాంశం       *  dst_d = (దశాంశం *) dst->దత్తాంశం;

    dpct::వరుస_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    // src1 contains mask and it is optional
    GGML_ASSERT(!src1 || src1->type == GGML_TYPE_F16 || src1->type == GGML_TYPE_F32);

    స్థిరరాశి int64_t nrows_x = ggml_nrows(src0);
    స్థిరరాశి int64_t nrows_y = src0->ne[1];

    స్థిరరాశి int64_t ne00 = src0->ne[0];

    దశాంశం scale    = 1.0f;
    దశాంశం max_bias = 0.0f;

    జ్ఞాపకం_నకలు(&scale,    (స్థిరరాశి దశాంశం *) dst->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max_bias, (స్థిరరాశి దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));

    స్థిరరాశి తార్కిక use_f16 = (src1 && src1->type == GGML_TYPE_F16);

    స్థిరరాశి int64_t nb11 = src1 ? src1->nb[1] : 1;
    స్థిరరాశి int64_t nb12 = src1 ? src1->nb[2] : 1;
    స్థిరరాశి int64_t nb13 = src1 ? src1->nb[3] : 1;

    స్థిరరాశి int64_t ne12 = src1 ? src1->ne[2] : 1;
    స్థిరరాశి int64_t ne13 = src1 ? src1->ne[3] : 1;

    స్థిరరాశి uint32_t n_తల      = src0->ne[2];
    స్థిరరాశి uint32_t n_తల_log2 = 1u << (uint32_t) floorf(log2f((దశాంశం) n_తల));

    స్థిరరాశి దశాంశం m0 = powf(2.0f, -(max_bias       ) / n_తల_log2);
    స్థిరరాశి దశాంశం m1 = powf(2.0f, -(max_bias / 2.0f) / n_తల_log2);


    soft_max_params params = {};
    params.nheads = src0->ne[2];
    params.n_తల_log2 = n_తల_log2;
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

    ఒకవేళ (use_f16) {
        soft_max_f32_sycl(src0_d, (స్థిరరాశి sycl::half *)src1_d,
                          (స్థిరరాశి దశాంశం *)src2_d, dst_d, params, stream,
                          ctx.device);
    } లేకపోతే {
        soft_max_f32_sycl(src0_d, (స్థిరరాశి దశాంశం *)src1_d, (స్థిరరాశి దశాంశం *)src2_d,
                          dst_d, params, stream, ctx.device);
    }
}

శూన్య ggml_sycl_op_soft_max_వెనుక(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    స్థిరరాశి ggml_tensor * src0 = dst->src[0]; // grad
    స్థిరరాశి ggml_tensor * src1 = dst->src[1]; // forward pass output

    స్థిరరాశి దశాంశం * src0_d = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం;
    స్థిరరాశి దశాంశం * src1_d = (స్థిరరాశి దశాంశం *) src1->దత్తాంశం;
    దశాంశం       * dst_d  = (దశాంశం       *) dst->దత్తాంశం;

    dpct::వరుస_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    స్థిరరాశి int64_t ncols = src0->ne[0];
    స్థిరరాశి int64_t nrows = ggml_nrows(src0);

    దశాంశం scale    = 1.0f;
    దశాంశం max_bias = 0.0f;

    జ్ఞాపకం_నకలు(&scale,    (స్థిరరాశి దశాంశం *) dst->op_params + 0, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&max_bias, (స్థిరరాశి దశాంశం *) dst->op_params + 1, పరిమాణం(దశాంశం));

    GGML_ASSERT(max_bias == 0.0f);

    soft_max_వెనుక_f32_sycl(src0_d, src1_d, dst_d, ncols, nrows, scale, stream);
}
