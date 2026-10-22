#అన్తర్భావ "norm.hpp"
#అన్తర్భావ "ggml-sycl/common.hpp"
#అన్తర్భావ "ggml-sycl/presets.hpp"

స్థిర శూన్య norm_f32(స్థిరరాశి దశాంశం* x, దశాంశం* dst, స్థిరరాశి పూర్ణాంకం ncols,
    స్థిరరాశి int64_t src_stride_నిలువు, స్థిరరాశి int64_t src_stride_అడ్డువరుస, స్థిరరాశి int64_t src_stride_channel, స్థిరరాశి int64_t src_stride_sample,
    స్థిరరాశి int64_t dst_stride_నిలువు, స్థిరరాశి int64_t dst_stride_అడ్డువరుస, స్థిరరాశి int64_t dst_stride_channel, స్థిరరాశి int64_t dst_stride_sample,
    స్థిరరాశి దశాంశం eps, స్థిరరాశి sycl::nd_అంశం<3>& అంశం_ct1, sycl::float2* s_కూడిక, పూర్ణాంకం ఖండం_కొలత_3) {

    స్థిరరాశి పూర్ణాంకం nthreads = అంశం_ct1.get_local_range(2);
    స్థిరరాశి పూర్ణాంకం sample  = అంశం_ct1.get_సమూహం(0);
    స్థిరరాశి పూర్ణాంకం channel = అంశం_ct1.get_సమూహం(1);
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస     = అంశం_ct1.get_సమూహం(2);

    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి పూర్ణాంకం nwarps = nthreads / WARP_కొలత;

    స్థిరరాశి స్వయంచాలక src_స్థానభ్రంశం = calculate_స్థానభ్రంశం<3>({src_stride_sample, src_stride_channel, src_stride_అడ్డువరుస}, {sample, channel, అడ్డువరుస});
    స్థిరరాశి స్వయంచాలక dst_స్థానభ్రంశం = calculate_స్థానభ్రంశం<3>({dst_stride_sample, dst_stride_channel, dst_stride_అడ్డువరుస}, {sample, channel, అడ్డువరుస});

    x += src_స్థానభ్రంశం;
    dst += dst_స్థానభ్రంశం;

    sycl::float2 సగటు_var = sycl::float2(0.f, 0.f);

    కొరకు (పూర్ణాంకం నిలువు = tid; నిలువు < ncols; నిలువు += ఖండం_కొలత_3) {
        స్థిరరాశి దశాంశం xi = x[నిలువు * src_stride_నిలువు];
        సగటు_var.x() += xi;
        సగటు_var.y() += xi * xi;
    }

    // sum up partial sums
    సగటు_var = warp_reduce_కూడిక(సగటు_var, అంశం_ct1);
    ఒకవేళ  (ఖండం_కొలత_3 > WARP_కొలత) {
        స్థిరరాశి స్వయంచాలక sub_సమూహం = అంశం_ct1.get_sub_సమూహం();
        స్థిరరాశి స్వయంచాలక sg_id = sub_సమూహం.get_సమూహం_రేఖీయ_id();
        స్థిరరాశి స్వయంచాలక wi_లోపల_sg = sub_సమూహం.get_local_రేఖీయ_id();
        ఒకవేళ (wi_లోపల_sg == 0) {
            s_కూడిక[sg_id] = సగటు_var;
        }
        అంశం_ct1.barrier(sycl::access::fence_space::local_space);
        సగటు_var = 0.f;
        స్థిరరాశి size_t nreduce = ceil_div(nwarps, WARP_కొలత);
        కొరకు (size_t i = 0; i < nreduce; i += 1)
        {
            సగటు_var += s_కూడిక[wi_లోపల_sg + i * WARP_కొలత];
        }
        సగటు_var = warp_reduce_కూడిక(సగటు_var, అంశం_ct1);
    }

    స్థిరరాశి దశాంశం సగటు_2 = సగటు_var.x() / ncols;
    స్థిరరాశి దశాంశం var = సగటు_var.y() / ncols - సగటు_2 * సగటు_2;
    స్థిరరాశి దశాంశం inv_std = sycl::rsqrt(var + eps);

    కొరకు (పూర్ణాంకం నిలువు = tid; నిలువు < ncols; నిలువు += ఖండం_కొలత_3) {
        dst[నిలువు * dst_stride_నిలువు] = (x[నిలువు * src_stride_నిలువు] - సగటు_2) * inv_std;
    }
}

స్థిర శూన్య సమూహం_norm_f32(స్థిరరాశి దశాంశం* x, దశాంశం* dst, స్థిరరాశి పూర్ణాంకం సమూహం_కొలత, స్థిరరాశి పూర్ణాంకం ne_elements, స్థిరరాశి దశాంశం eps,
    స్థిరరాశి sycl::nd_అంశం<3>& అంశం_ct1, దశాంశం* s_కూడిక, పూర్ణాంకం ఖండం_కొలత_3) {
    పూర్ణాంకం start = అంశం_ct1.get_సమూహం(2) * సమూహం_కొలత;
    పూర్ణాంకం end = start + సమూహం_కొలత;
    స్థిరరాశి పూర్ణాంకం nthreads = అంశం_ct1.get_local_range(2);
    స్థిరరాశి పూర్ణాంకం nwarps = nthreads / WARP_కొలత;
    start += అంశం_ct1.get_local_id(2);
    size_t nreduce = nwarps / WARP_కొలత;

    ఒకవేళ (end >= ne_elements) {
        end = ne_elements;
    }

    దశాంశం tmp = 0.0f; // partial sum for thread in warp

    కొరకు (పూర్ణాంకం j = start; j < end; j += ఖండం_కొలత_3) {
        tmp += x[j];
    }

    tmp = warp_reduce_కూడిక(tmp, అంశం_ct1);
    ఒకవేళ (ఖండం_కొలత_3 > WARP_కొలత) {

        పూర్ణాంకం warp_id = అంశం_ct1.get_local_id(2) / WARP_కొలత;
        పూర్ణాంకం lane_id = అంశం_ct1.get_local_id(2) % WARP_కొలత;
        ఒకవేళ (lane_id == 0) {
            s_కూడిక[warp_id] = tmp;
        }
        /*
        DPCT1118:1: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:54: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        అంశం_ct1.barrier();
        tmp = 0.f;
        కొరకు (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_కూడిక[lane_id + i * WARP_కొలత];
        }
        tmp = warp_reduce_కూడిక(tmp, అంశం_ct1);
    }

    దశాంశం సగటు_2 = tmp / సమూహం_కొలత;
    tmp = 0.0f;

    కొరకు (పూర్ణాంకం j = start; j < end; j += ఖండం_కొలత_3) {
        దశాంశం xi = x[j] - సగటు_2;
        dst[j] = xi;
        tmp += xi * xi;
    }

    tmp = warp_reduce_కూడిక(tmp, అంశం_ct1);
    ఒకవేళ (ఖండం_కొలత_3 > WARP_కొలత) {

        పూర్ణాంకం warp_id = అంశం_ct1.get_local_id(2) / WARP_కొలత;
        పూర్ణాంకం lane_id = అంశం_ct1.get_local_id(2) % WARP_కొలత;
        ఒకవేళ (lane_id == 0) {
            s_కూడిక[warp_id] = tmp;
        }
        /*
        DPCT1118:2: SYCL group functions and algorithms must be encountered in
        converged control flow. You may need to adjust the code.
        */
        /*
        DPCT1065:55: Consider replacing sycl::nd_item::barrier() with
        sycl::nd_item::barrier(sycl::access::fence_space::local_space) for
        better performance if there is no access to global memory.
        */
        అంశం_ct1.barrier();
        tmp = 0.f;
        కొరకు (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_కూడిక[lane_id + i * WARP_కొలత];
        }
        tmp = warp_reduce_కూడిక(tmp, అంశం_ct1);
    }

    దశాంశం variance = tmp / సమూహం_కొలత;
    దశాంశం scale = sycl::rsqrt(variance + eps);
    కొరకు (పూర్ణాంకం j = start; j < end; j += ఖండం_కొలత_3) {
        dst[j] *= scale;
    }
}

మూస <తార్కిక do_multiply = అబద్ధం, తార్కిక do_కూడు = అబద్ధం>
స్థిర శూన్య rms_norm_f32(స్థిరరాశి దశాంశం* x, దశాంశం* dst, స్థిరరాశి పూర్ణాంకం ncols,
    స్థిరరాశి int64_t src_stride_నిలువు, స్థిరరాశి int64_t src_stride_అడ్డువరుస, స్థిరరాశి int64_t src_stride_channel, స్థిరరాశి int64_t src_stride_sample,
    స్థిరరాశి int64_t dst_stride_నిలువు, స్థిరరాశి int64_t dst_stride_అడ్డువరుస, స్థిరరాశి int64_t dst_stride_channel, స్థిరరాశి int64_t dst_stride_sample,
    స్థిరరాశి దశాంశం eps, స్థిరరాశి sycl::nd_అంశం<3>& అంశం_ct1, దశాంశం* s_కూడిక, పూర్ణాంకం ఖండం_కొలత_3,
    స్థిరరాశి దశాంశం* mul = శూన్యసూచిక, స్థిరరాశి int64_t mul_stride_అడ్డువరుస = 0, స్థిరరాశి int64_t mul_stride_channel = 0,
    స్థిరరాశి int64_t mul_stride_sample = 0, స్థిరరాశి పూర్ణాంకం mul_nrows = 0, స్థిరరాశి పూర్ణాంకం mul_nchannels = 0, స్థిరరాశి పూర్ణాంకం mul_nsamples = 0,
    స్థిరరాశి దశాంశం* కూడు_2 = శూన్యసూచిక, స్థిరరాశి int64_t కూడు_stride_అడ్డువరుస = 0, స్థిరరాశి int64_t కూడు_stride_channel = 0,
    స్థిరరాశి int64_t కూడు_stride_sample = 0, స్థిరరాశి పూర్ణాంకం కూడు_nrows = 0, స్థిరరాశి పూర్ణాంకం కూడు_nchannels = 0, స్థిరరాశి పూర్ణాంకం కూడు_nsamples = 0,
    స్థిరరాశి దశాంశం scale_mul = 1.0f) {

    static_assert(!do_కూడు || do_multiply, "fusing add is not supported without multiplying");

    స్థిరరాశి పూర్ణాంకం sample  = అంశం_ct1.get_సమూహం(0);
    స్థిరరాశి పూర్ణాంకం channel = అంశం_ct1.get_సమూహం(1);
    స్థిరరాశి పూర్ణాంకం అడ్డువరుస     = అంశం_ct1.get_సమూహం(2);

    స్థిరరాశి పూర్ణాంకం nthreads = అంశం_ct1.get_local_range(2);

    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి పూర్ణాంకం nwarps = nthreads / WARP_కొలత;

    స్థిరరాశి స్వయంచాలక src_స్థానభ్రంశం = calculate_స్థానభ్రంశం<3>({src_stride_sample, src_stride_channel, src_stride_అడ్డువరుస}, {sample, channel, అడ్డువరుస});
    స్థిరరాశి స్వయంచాలక dst_స్థానభ్రంశం = calculate_స్థానభ్రంశం<3>({dst_stride_sample, dst_stride_channel, dst_stride_అడ్డువరుస}, {sample, channel, అడ్డువరుస});

    x   += src_స్థానభ్రంశం;
    dst += dst_స్థానభ్రంశం;

    ఒకవేళ constexpr (do_multiply) {
        స్థిరరాశి పూర్ణాంకం mul_అడ్డువరుస     = అడ్డువరుస     % mul_nrows;
        స్థిరరాశి పూర్ణాంకం mul_channel = channel % mul_nchannels;
        స్థిరరాశి పూర్ణాంకం mul_sample  = sample  % mul_nsamples;
        mul += mul_sample * mul_stride_sample + mul_channel * mul_stride_channel + mul_అడ్డువరుస * mul_stride_అడ్డువరుస;
    }

    ఒకవేళ constexpr (do_కూడు) {
        స్థిరరాశి పూర్ణాంకం కూడు_అడ్డువరుస     = అడ్డువరుస     % కూడు_nrows;
        స్థిరరాశి పూర్ణాంకం కూడు_channel = channel % కూడు_nchannels;
        స్థిరరాశి పూర్ణాంకం కూడు_sample  = sample  % కూడు_nsamples;
        కూడు_2 += కూడు_sample * కూడు_stride_sample + కూడు_channel * కూడు_stride_channel + కూడు_అడ్డువరుస * కూడు_stride_అడ్డువరుస;
    }

    దశాంశం tmp = 0.0f; // partial sum for thread in warp

    కొరకు (పూర్ణాంకం నిలువు = tid; నిలువు < ncols; నిలువు += ఖండం_కొలత_3) {
        స్థిరరాశి దశాంశం xi = x[నిలువు * src_stride_నిలువు];
        tmp += xi * xi;
    }

    // sum up partial sums
    tmp = warp_reduce_కూడిక(tmp, అంశం_ct1);
    ఒకవేళ (ఖండం_కొలత_3 > WARP_కొలత) {
        స్థిరరాశి స్వయంచాలక sub_సమూహం = అంశం_ct1.get_sub_సమూహం();
        స్థిరరాశి స్వయంచాలక sg_id = sub_సమూహం.get_సమూహం_రేఖీయ_id();
        స్థిరరాశి స్వయంచాలక wi_లోపల_sg = sub_సమూహం.get_local_రేఖీయ_id();
        ఒకవేళ (wi_లోపల_sg == 0) {
            s_కూడిక[sg_id] = tmp;
        }

        అంశం_ct1.barrier(sycl::access::fence_space::local_space);
        స్థిరరాశి size_t nreduce = ceil_div(nwarps, WARP_కొలత);
        tmp = 0.f;
        కొరకు (size_t i = 0; i < nreduce; i += 1)
        {
            tmp += s_కూడిక[wi_లోపల_sg + i * WARP_కొలత];
        }
        tmp = warp_reduce_కూడిక(tmp, అంశం_ct1);
    }

    స్థిరరాశి దశాంశం సగటు_2 = tmp / ncols;
    స్థిరరాశి దశాంశం scale = sycl::rsqrt(సగటు_2 + eps);

    కొరకు (పూర్ణాంకం నిలువు = tid; నిలువు < ncols; నిలువు += ఖండం_కొలత_3) {
        ఒకవేళ constexpr (do_multiply && do_కూడు) {
            dst[నిలువు * dst_stride_నిలువు] = scale * x[నిలువు * src_stride_నిలువు] * mul[నిలువు] + కూడు_2[నిలువు];
        } లేకపోతే ఒకవేళ constexpr (do_multiply) {
            dst[నిలువు * dst_stride_నిలువు] = scale * x[నిలువు * src_stride_నిలువు] * mul[నిలువు];
        } లేకపోతే {
            // folded epilogue of a fused GGML_OP_SCALE consumer (qwen35 GDN l2 norms);
            // the explicit temporary keeps the float evaluation order identical to
            // running rms_norm and scale as two separate kernels
            స్థిరరాశి దశాంశం v = scale * x[నిలువు * src_stride_నిలువు];
            dst[నిలువు * dst_stride_నిలువు] = v * scale_mul;
        }
    }
}

మూస<పూర్ణాంకం warp_కొలత>
స్థిర శూన్య l2_norm_f32(స్థిరరాశి దశాంశం * x, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols,
    స్థిరరాశి int64_t src_stride_నిలువు, స్థిరరాశి int64_t src_stride_అడ్డువరుస, స్థిరరాశి int64_t src_stride_channel,
    స్థిరరాశి int64_t src_stride_sample, స్థిరరాశి int64_t dst_stride_నిలువు, స్థిరరాశి int64_t dst_stride_అడ్డువరుస,
    స్థిరరాశి int64_t dst_stride_channel, స్థిరరాశి int64_t dst_stride_sample, స్థిరరాశి దశాంశం eps,
    స్థిరరాశి sycl::nd_అంశం<3>& అంశం_ct1, దశాంశం* s_కూడిక, స్థిరరాశి పూర్ణాంకం ఖండం_కొలత_3) {

    స్థిరరాశి పూర్ణాంకం అడ్డువరుస     = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి పూర్ణాంకం channel = అంశం_ct1.get_సమూహం(1);
    స్థిరరాశి పూర్ణాంకం sample  = అంశం_ct1.get_సమూహం(0);
    స్థిరరాశి పూర్ణాంకం tid     = అంశం_ct1.get_local_id(2);

    x   += sample*src_stride_sample + channel*src_stride_channel + అడ్డువరుస*src_stride_అడ్డువరుస;
    dst += sample*dst_stride_sample + channel*dst_stride_channel + అడ్డువరుస*dst_stride_అడ్డువరుస;

    దశాంశం tmp = 0.0f; // partial sum for thread in warp

    కొరకు (పూర్ణాంకం నిలువు = tid; నిలువు < ncols; నిలువు += ఖండం_కొలత_3) {
        స్థిరరాశి దశాంశం xi = x[నిలువు * src_stride_నిలువు];
        tmp += xi * xi;
    }

    tmp = ఖండం_reduce<ఖండం_reduce_పద్ధతి::కూడిక_3, warp_కొలత>(tmp, s_కూడిక, ఖండం_కొలత_3);
    స్థిరరాశి దశాంశం scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));

    కొరకు (పూర్ణాంకం నిలువు = tid; నిలువు < ncols; నిలువు += ఖండం_కొలత_3) {
        dst[నిలువు * dst_stride_నిలువు] = scale * x[నిలువు * src_stride_నిలువు];
    }
}

స్థిర శూన్య norm_f32_sycl(స్థిరరాశి దశాంశం * x, దశాంశం * dst, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం nchannels, స్థిరరాశి పూర్ణాంకం nsamples,
    స్థిరరాశి int64_t src_stride_నిలువు, స్థిరరాశి int64_t src_stride_అడ్డువరుస, స్థిరరాశి int64_t src_stride_channel, స్థిరరాశి int64_t src_stride_sample,
    స్థిరరాశి int64_t dst_stride_నిలువు, స్థిరరాశి int64_t dst_stride_అడ్డువరుస, స్థిరరాశి int64_t dst_stride_channel, స్థిరరాశి int64_t dst_stride_sample,
        స్థిరరాశి దశాంశం eps, వరుస_ptr stream, పూర్ణాంకం device) {

    స్థిరరాశి sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ఒకవేళ (ncols < 1024) {
        స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, WARP_కొలత);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_నిలువు, src_stride_అడ్డువరుస, src_stride_channel, src_stride_sample,
                        dst_stride_నిలువు, dst_stride_అడ్డువరుస, dst_stride_channel, dst_stride_sample,
                        eps, అంశం_ct1, శూన్యసూచిక, WARP_కొలత);
                });
            });
    }
    లేకపోతే {
        స్థిరరాశి పూర్ణాంకం work_సమూహం_కొలత = ggml_sycl_info().max_work_సమూహం_sizes[device];
        ప్రతిజ్ఞా(work_సమూహం_కొలత % (WARP_కొలత * WARP_కొలత) == 0);
        స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, work_సమూహం_కొలత);
        /*
        DPCT1049:17: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<sycl::float2, 1> s_కూడిక_acc_ct1(
                            sycl::range<1>(work_సమూహం_కొలత / WARP_కొలత), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    norm_f32(x, dst, ncols,
                        src_stride_నిలువు, src_stride_అడ్డువరుస, src_stride_channel, src_stride_sample,
                        dst_stride_నిలువు, dst_stride_అడ్డువరుస, dst_stride_channel, dst_stride_sample,
                        eps, అంశం_ct1, get_సూచిక(s_కూడిక_acc_ct1), work_సమూహం_కొలత);
                });
            });
    }
}

స్థిర శూన్య సమూహం_norm_f32_sycl(స్థిరరాశి దశాంశం* x, దశాంశం* dst,
    స్థిరరాశి పూర్ణాంకం అంకె_groups, స్థిరరాశి దశాంశం eps, స్థిరరాశి పూర్ణాంకం సమూహం_కొలత,
    స్థిరరాశి పూర్ణాంకం ne_elements, వరుస_ptr stream, పూర్ణాంకం device) {
    ఒకవేళ (సమూహం_కొలత < 1024) {
        స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, WARP_కొలత);
        stream->submit([&](sycl::handler& cgh) {
            స్థిరరాశి దశాంశం eps_ct4 = eps;
            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, అంకె_groups) * ఖండం_dims,
                    ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    సమూహం_norm_f32(
                        x, dst, సమూహం_కొలత, ne_elements, eps_ct4, అంశం_ct1,
                        శూన్యసూచిక, WARP_కొలత);
                });
            });
    }
    లేకపోతే {
        స్థిరరాశి పూర్ణాంకం work_సమూహం_కొలత = ggml_sycl_info().max_work_సమూహం_sizes[device];
        ప్రతిజ్ఞా(work_సమూహం_కొలత % (WARP_కొలత * WARP_కొలత) == 0);
        స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, work_సమూహం_కొలత);
        /*
        DPCT1049:18: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */

        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<దశాంశం, 1> s_కూడిక_acc_ct1(sycl::range<1>(work_సమూహం_కొలత / WARP_కొలత),
                cgh);

            స్థిరరాశి దశాంశం eps_ct4 = eps;

            cgh.parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, అంకె_groups) * ఖండం_dims,
                    ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    సమూహం_norm_f32(x, dst, సమూహం_కొలత, ne_elements,
                        eps_ct4, అంశం_ct1,
                        get_సూచిక(s_కూడిక_acc_ct1), work_సమూహం_కొలత);
                });
            });
    }
}

స్థిర శూన్య rms_norm_f32_sycl(స్థిరరాశి దశాంశం* x, దశాంశం* dst, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం nchannels, స్థిరరాశి పూర్ణాంకం nsamples,
    స్థిరరాశి int64_t src_stride_నిలువు, స్థిరరాశి int64_t src_stride_అడ్డువరుస, స్థిరరాశి int64_t src_stride_channel, స్థిరరాశి int64_t src_stride_sample,
    స్థిరరాశి int64_t dst_stride_నిలువు, స్థిరరాశి int64_t dst_stride_అడ్డువరుస, స్థిరరాశి int64_t dst_stride_channel, స్థిరరాశి int64_t dst_stride_sample,
    స్థిరరాశి దశాంశం eps, వరుస_ptr stream, పూర్ణాంకం device) {
    // printf("%s ncols=%d, nrows=%d, WARP_SIZE=%d\n", __func__, ncols, nrows, WARP_SIZE);

    స్థిరరాశి sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ఒకవేళ (ncols < 1024) {
        స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, WARP_కొలత);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_నిలువు, src_stride_అడ్డువరుస, src_stride_channel, src_stride_sample,
                        dst_stride_నిలువు, dst_stride_అడ్డువరుస, dst_stride_channel, dst_stride_sample,
                        eps, అంశం_ct1, శూన్యసూచిక, WARP_కొలత);
                });
            });
    }
    లేకపోతే {
        స్థిరరాశి పూర్ణాంకం work_సమూహం_కొలత = ggml_sycl_info().max_work_సమూహం_sizes[device];
        ప్రతిజ్ఞా(work_సమూహం_కొలత % (WARP_కొలత * WARP_కొలత) == 0);
        స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, work_సమూహం_కొలత);
        /*
        DPCT1049:19: The work-group size passed to the SYCL kernel may exceed
        the limit. To get the device limit, query
        info::device::max_work_group_size. Adjust the work-group size if needed.
        */
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<దశాంశం, 1> s_కూడిక_acc_ct1(sycl::range<1>(work_సమూహం_కొలత / WARP_కొలత),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_నిలువు, src_stride_అడ్డువరుస, src_stride_channel, src_stride_sample,
                        dst_stride_నిలువు, dst_stride_అడ్డువరుస, dst_stride_channel, dst_stride_sample,
                        eps, అంశం_ct1, get_సూచిక(s_కూడిక_acc_ct1), work_సమూహం_కొలత);
                });
            });
    }
}

స్థిర శూన్య rms_norm_scale_f32_sycl(స్థిరరాశి దశాంశం* x, దశాంశం* dst, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
    స్థిరరాశి పూర్ణాంకం nchannels, స్థిరరాశి పూర్ణాంకం nsamples,
    స్థిరరాశి int64_t src_stride_నిలువు, స్థిరరాశి int64_t src_stride_అడ్డువరుస, స్థిరరాశి int64_t src_stride_channel, స్థిరరాశి int64_t src_stride_sample,
    స్థిరరాశి int64_t dst_stride_నిలువు, స్థిరరాశి int64_t dst_stride_అడ్డువరుస, స్థిరరాశి int64_t dst_stride_channel, స్థిరరాశి int64_t dst_stride_sample,
    స్థిరరాశి దశాంశం eps, స్థిరరాశి దశాంశం scale_mul, వరుస_ptr stream, పూర్ణాంకం device) {
    స్థిరరాశి sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ఒకవేళ (ncols < 1024) {
        స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, WARP_కొలత);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_నిలువు, src_stride_అడ్డువరుస, src_stride_channel, src_stride_sample,
                        dst_stride_నిలువు, dst_stride_అడ్డువరుస, dst_stride_channel, dst_stride_sample,
                        eps, అంశం_ct1, శూన్యసూచిక, WARP_కొలత,
                        శూన్యసూచిక, 0, 0, 0, 0, 0, 0,
                        శూన్యసూచిక, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
    లేకపోతే {
        స్థిరరాశి పూర్ణాంకం work_సమూహం_కొలత = ggml_sycl_info().max_work_సమూహం_sizes[device];
        ప్రతిజ్ఞా(work_సమూహం_కొలత % (WARP_కొలత * WARP_కొలత) == 0);
        స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, work_సమూహం_కొలత);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<దశాంశం, 1> s_కూడిక_acc_ct1(sycl::range<1>(work_సమూహం_కొలత / WARP_కొలత),
                cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    rms_norm_f32(x, dst, ncols,
                        src_stride_నిలువు, src_stride_అడ్డువరుస, src_stride_channel, src_stride_sample,
                        dst_stride_నిలువు, dst_stride_అడ్డువరుస, dst_stride_channel, dst_stride_sample,
                        eps, అంశం_ct1, get_సూచిక(s_కూడిక_acc_ct1), work_సమూహం_కొలత,
                        శూన్యసూచిక, 0, 0, 0, 0, 0, 0,
                        శూన్యసూచిక, 0, 0, 0, 0, 0, 0,
                        scale_mul);
                });
            });
    }
}

స్థిర శూన్య rms_norm_mul_f32_sycl(స్థిరరాశి దశాంశం* x, స్థిరరాశి దశాంశం* mul, దశాంశం* dst, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows,
        స్థిరరాశి పూర్ణాంకం nchannels, స్థిరరాశి పూర్ణాంకం nsamples,
        స్థిరరాశి int64_t src_stride_నిలువు, స్థిరరాశి int64_t src_stride_అడ్డువరుస, స్థిరరాశి int64_t src_stride_channel, స్థిరరాశి int64_t src_stride_sample,
        స్థిరరాశి int64_t dst_stride_నిలువు, స్థిరరాశి int64_t dst_stride_అడ్డువరుస, స్థిరరాశి int64_t dst_stride_channel, స్థిరరాశి int64_t dst_stride_sample,
        స్థిరరాశి int64_t mul_stride_అడ్డువరుస, స్థిరరాశి int64_t mul_stride_channel, స్థిరరాశి int64_t mul_stride_sample,
        స్థిరరాశి పూర్ణాంకం mul_nrows, స్థిరరాశి పూర్ణాంకం mul_nchannels, స్థిరరాశి పూర్ణాంకం mul_nsamples,
        స్థిరరాశి దశాంశం eps, వరుస_ptr stream, పూర్ణాంకం device) {
    స్థిరరాశి sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ఒకవేళ (ncols < 1024) {
        స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, WARP_కొలత);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    rms_norm_f32<నిజం>(x, dst, ncols,
                        src_stride_నిలువు, src_stride_అడ్డువరుస, src_stride_channel, src_stride_sample,
                        dst_stride_నిలువు, dst_stride_అడ్డువరుస, dst_stride_channel, dst_stride_sample,
                        eps, అంశం_ct1, శూన్యసూచిక, WARP_కొలత,
                        mul, mul_stride_అడ్డువరుస, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
    లేకపోతే {
        స్థిరరాశి పూర్ణాంకం work_సమూహం_కొలత = ggml_sycl_info().max_work_సమూహం_sizes[device];
        ప్రతిజ్ఞా(work_సమూహం_కొలత % (WARP_కొలత * WARP_కొలత) == 0);
        స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, work_సమూహం_కొలత);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<దశాంశం, 1> s_కూడిక_acc_ct1(sycl::range<1>(work_సమూహం_కొలత / WARP_కొలత), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    rms_norm_f32<నిజం>(x, dst, ncols,
                        src_stride_నిలువు, src_stride_అడ్డువరుస, src_stride_channel, src_stride_sample,
                        dst_stride_నిలువు, dst_stride_అడ్డువరుస, dst_stride_channel, dst_stride_sample,
                        eps, అంశం_ct1, get_సూచిక(s_కూడిక_acc_ct1), work_సమూహం_కొలత,
                        mul, mul_stride_అడ్డువరుస, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples);
                });
            });
    }
}

స్థిర శూన్య rms_norm_mul_కూడు_f32_sycl(స్థిరరాశి దశాంశం* x, స్థిరరాశి దశాంశం* mul, స్థిరరాశి దశాంశం* కూడు_2, దశాంశం* dst,
        స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows, స్థిరరాశి పూర్ణాంకం nchannels, స్థిరరాశి పూర్ణాంకం nsamples,
        స్థిరరాశి int64_t src_stride_నిలువు, స్థిరరాశి int64_t src_stride_అడ్డువరుస, స్థిరరాశి int64_t src_stride_channel, స్థిరరాశి int64_t src_stride_sample,
        స్థిరరాశి int64_t dst_stride_నిలువు, స్థిరరాశి int64_t dst_stride_అడ్డువరుస, స్థిరరాశి int64_t dst_stride_channel, స్థిరరాశి int64_t dst_stride_sample,
        స్థిరరాశి int64_t mul_stride_అడ్డువరుస, స్థిరరాశి int64_t mul_stride_channel, స్థిరరాశి int64_t mul_stride_sample,
        స్థిరరాశి పూర్ణాంకం mul_nrows, స్థిరరాశి పూర్ణాంకం mul_nchannels, స్థిరరాశి పూర్ణాంకం mul_nsamples,
        స్థిరరాశి int64_t కూడు_stride_అడ్డువరుస, స్థిరరాశి int64_t కూడు_stride_channel, స్థిరరాశి int64_t కూడు_stride_sample,
        స్థిరరాశి పూర్ణాంకం కూడు_nrows, స్థిరరాశి పూర్ణాంకం కూడు_nchannels, స్థిరరాశి పూర్ణాంకం కూడు_nsamples,
        స్థిరరాశి దశాంశం eps, వరుస_ptr stream, పూర్ణాంకం device) {
    స్థిరరాశి sycl::range<3> global_dims(nsamples, nchannels, nrows);
    ఒకవేళ (ncols < 1024) {
        స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, WARP_కొలత);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    rms_norm_f32<నిజం, నిజం>(x, dst, ncols,
                        src_stride_నిలువు, src_stride_అడ్డువరుస, src_stride_channel, src_stride_sample,
                        dst_stride_నిలువు, dst_stride_అడ్డువరుస, dst_stride_channel, dst_stride_sample,
                        eps, అంశం_ct1, శూన్యసూచిక, WARP_కొలత,
                        mul, mul_stride_అడ్డువరుస, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        కూడు_2, కూడు_stride_అడ్డువరుస, కూడు_stride_channel, కూడు_stride_sample, కూడు_nrows, కూడు_nchannels, కూడు_nsamples);
                });
            });
    }
    లేకపోతే {
        స్థిరరాశి పూర్ణాంకం work_సమూహం_కొలత = ggml_sycl_info().max_work_సమూహం_sizes[device];
        ప్రతిజ్ఞా(work_సమూహం_కొలత % (WARP_కొలత * WARP_కొలత) == 0);
        స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, work_సమూహం_కొలత);
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<దశాంశం, 1> s_కూడిక_acc_ct1(sycl::range<1>(work_సమూహం_కొలత / WARP_కొలత), cgh);
            cgh.parallel_for(
                sycl::nd_range<3>(global_dims * ఖండం_dims, ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                    rms_norm_f32<నిజం, నిజం>(x, dst, ncols,
                        src_stride_నిలువు, src_stride_అడ్డువరుస, src_stride_channel, src_stride_sample,
                        dst_stride_నిలువు, dst_stride_అడ్డువరుస, dst_stride_channel, dst_stride_sample,
                        eps, అంశం_ct1, get_సూచిక(s_కూడిక_acc_ct1), work_సమూహం_కొలత,
                        mul, mul_stride_అడ్డువరుస, mul_stride_channel, mul_stride_sample, mul_nrows, mul_nchannels, mul_nsamples,
                        కూడు_2, కూడు_stride_అడ్డువరుస, కూడు_stride_channel, కూడు_stride_sample, కూడు_nrows, కూడు_nchannels, కూడు_nsamples);
                });
            });
    }
}

మూస<పూర్ణాంకం warp_కొలత>
స్థిర శూన్య l2_norm_f32_sycl(స్థిరరాశి దశాంశం *   x,
                             దశాంశం *         dst,
                             స్థిరరాశి పూర్ణాంకం       ncols,
                             స్థిరరాశి పూర్ణాంకం       nrows,
                             స్థిరరాశి పూర్ణాంకం       nchannels,
                             స్థిరరాశి పూర్ణాంకం       nsamples,
                             స్థిరరాశి int64_t   src_stride_నిలువు,
                             స్థిరరాశి int64_t   src_stride_అడ్డువరుస,
                             స్థిరరాశి int64_t   src_stride_channel,
                             స్థిరరాశి int64_t   src_stride_sample,
                             స్థిరరాశి int64_t   dst_stride_నిలువు,
                             స్థిరరాశి int64_t   dst_stride_అడ్డువరుస,
                             స్థిరరాశి int64_t   dst_stride_channel,
                             స్థిరరాశి int64_t   dst_stride_sample,
                             స్థిరరాశి దశాంశం     eps,
                             వరుస_ptr       stream,
                             పూర్ణాంకం             device) {
    స్థిరరాశి dpct::dim3 ఖండాలు_అంకె(nrows, nchannels, nsamples);

    ఒకవేళ (ncols < 1024) {
        స్థిరరాశి dpct::dim3 ఖండం_dims(warp_కొలత, 1, 1);
        stream->submit([&](sycl::handler& cgh) {
            cgh.parallel_for(
                sycl::nd_range<3>(ఖండాలు_అంకె * ఖండం_dims,
                    ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(warp_కొలత)]] {
                    l2_norm_f32<warp_కొలత>(x, dst, ncols,
                        src_stride_నిలువు, src_stride_అడ్డువరుస, src_stride_channel, src_stride_sample,
                        dst_stride_నిలువు, dst_stride_అడ్డువరుస, dst_stride_channel, dst_stride_sample,
                        eps, అంశం_ct1,
                        శూన్యసూచిక, warp_కొలత);
                });
            });
    }
    లేకపోతే {
        స్థిరరాశి పూర్ణాంకం work_సమూహం_కొలత = ggml_sycl_info().max_work_సమూహం_sizes[device];
        ప్రతిజ్ఞా(work_సమూహం_కొలత % (warp_కొలత * warp_కొలత) == 0);
        స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, work_సమూహం_కొలత);
        పూర్ణాంకం lsm_కొలత =  ఖండం_dims[2] > warp_కొలత ? work_సమూహం_కొలత / warp_కొలత * పరిమాణం(దశాంశం): 0;
        stream->submit([&](sycl::handler& cgh) {
            sycl::local_accessor<దశాంశం, 1> s_కూడిక_acc_ct1(sycl::range<1>(lsm_కొలత),
                cgh);

            cgh.parallel_for(
                sycl::nd_range<3>(ఖండాలు_అంకె * ఖండం_dims,
                    ఖండం_dims),
                [=](sycl::nd_అంశం<3> అంశం_ct1)
                [[sycl::reqd_sub_సమూహం_కొలత(warp_కొలత)]] {
                    l2_norm_f32<warp_కొలత>(x, dst, ncols,
                        src_stride_నిలువు, src_stride_అడ్డువరుస, src_stride_channel, src_stride_sample,
                        dst_stride_నిలువు, dst_stride_అడ్డువరుస, dst_stride_channel, dst_stride_sample,
                        eps, అంశం_ct1, get_సూచిక(s_కూడిక_acc_ct1), work_సమూహం_కొలత);
                });
            });
    }
}

// Batched L2 norm: N independent same-shape F32 tensors in one launch; the tensor
// index is folded into grid dim0 and each row's reduction is identical to the
// single-tensor kernel, so the result is bit-exact.
నిర్మాణం l2_batch_ptrs {
    స్థిరరాశి దశాంశం * src[GGML_SYCL_L2_BATCH_MAX];
    దశాంశం *       dst[GGML_SYCL_L2_BATCH_MAX];
};

// One stride set shared by the whole batch: the caller only groups tensors whose nb[]
// all match, so per-tensor state stays two pointers.
నిర్మాణం l2_batch_strides {
    పూర్ణాంకం     ne1, ne2;
    int64_t ss0, ss1, ss2, ss3;
    int64_t ds0, ds1, ds2, ds3;
};

మూస <పూర్ణాంకం warp_కొలత>
స్థిర శూన్య l2_norm_f32_batch(l2_batch_ptrs p, l2_batch_strides st, స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి దశాంశం eps,
                              స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం t   = అంశం_ct1.get_సమూహం(0);  // tensor index
    స్థిరరాశి పూర్ణాంకం r   = అంశం_ct1.get_సమూహం(2);  // flattened row over ne1*ne2*ne3
    స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2);

    స్థిరరాశి పూర్ణాంకం i1 = r % st.ne1;
    స్థిరరాశి పూర్ణాంకం i2 = (r / st.ne1) % st.ne2;
    స్థిరరాశి పూర్ణాంకం i3 = r / (st.ne1 * st.ne2);

    స్థిరరాశి దశాంశం * x   = p.src[t] + i3 * st.ss3 + i2 * st.ss2 + i1 * st.ss1;
    దశాంశం *       dst = p.dst[t] + i3 * st.ds3 + i2 * st.ds2 + i1 * st.ds1;

    దశాంశం tmp = 0.0f;
    కొరకు (పూర్ణాంకం నిలువు = tid; నిలువు < ncols; నిలువు += warp_కొలత) {
        స్థిరరాశి దశాంశం xi = x[నిలువు * st.ss0];
        tmp += xi * xi;
    }
    tmp = ఖండం_reduce<ఖండం_reduce_పద్ధతి::కూడిక_3, warp_కొలత>(tmp, (దశాంశం *) శూన్యసూచిక, warp_కొలత);
    స్థిరరాశి దశాంశం scale = sycl::rsqrt(sycl::fmax(tmp, eps * eps));
    కొరకు (పూర్ణాంకం నిలువు = tid; నిలువు < ncols; నిలువు += warp_కొలత) {
        dst[నిలువు * st.ds0] = scale * x[నిలువు * st.ss0];
    }
}

మూస <పూర్ణాంకం warp_కొలత>
స్థిర శూన్య l2_norm_f32_batch_sycl(l2_batch_ptrs p, l2_batch_strides st, స్థిరరాశి పూర్ణాంకం n_tensors,
                                   స్థిరరాశి పూర్ణాంకం ncols, స్థిరరాశి పూర్ణాంకం nrows_మొత్తం, స్థిరరాశి దశాంశం eps,
                                   వరుస_ptr stream) {
    స్థిరరాశి dpct::dim3 ఖండాలు_అంకె(nrows_మొత్తం, 1, n_tensors);
    స్థిరరాశి dpct::dim3 ఖండం_dims(warp_కొలత, 1, 1);
    stream->submit([&](sycl::handler & cgh) {
        cgh.parallel_for(sycl::nd_range<3>(ఖండాలు_అంకె * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(warp_కొలత)]] {
                l2_norm_f32_batch<warp_కొలత>(p, st, ncols, eps, అంశం_ct1);
            });
    });
}

శూన్య ggml_sycl_op_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS
    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));
    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(dst->src[0]->దత్తాంశం);
    దశాంశం *       dst_dd  = static_cast<దశాంశం *>(dst->దత్తాంశం);

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, dst->op_params, పరిమాణం(దశాంశం));
    GGML_ASSERT(eps >= 0.0f);
    స్థిరరాశి size_t ts0 = ggml_type_కొలత(src0->type);
    స్థిరరాశి size_t tdst = ggml_type_కొలత(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    స్థిరరాశి int64_t ss0 = nb00 / ts0;
    స్థిరరాశి int64_t ss1 = nb01 / ts0;
    స్థిరరాశి int64_t ss2 = nb02 / ts0;
    స్థిరరాశి int64_t ss3 = nb03 / ts0;
    స్థిరరాశి int64_t ds0 = nb0 / tdst;
    స్థిరరాశి int64_t ds1 = nb1 / tdst;
    స్థిరరాశి int64_t ds2 = nb2 / tdst;
    స్థిరరాశి int64_t ds3 = nb3 / tdst;

    norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_ప్రవాహం, ctx.device);
}

శూన్య ggml_sycl_op_సమూహం_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    పూర్ణాంకం అంకె_groups = dst->op_params[0];
    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(dst->src[0]->దత్తాంశం);
    దశాంశం *       dst_dd  = static_cast<దశాంశం *>(dst->దత్తాంశం);

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, dst->op_params + 1, పరిమాణం(దశాంశం));

    పూర్ణాంకం సమూహం_కొలత = dst->src[0]->ne[0] * dst->src[0]->ne[1] * ((dst->src[0]->ne[2] + అంకె_groups - 1) / అంకె_groups);
    సమూహం_norm_f32_sycl(src0_dd, dst_dd, అంకె_groups, eps, సమూహం_కొలత, dst->src[0]->ne[0] * dst->src[0]->ne[1] * dst->src[0]->ne[2], main_ప్రవాహం, ctx.device);
}

శూన్య ggml_sycl_op_rms_norm(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {

    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(dst->src[0]->దత్తాంశం);
    దశాంశం *       dst_dd  = static_cast<దశాంశం *>(dst->దత్తాంశం);

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, dst->op_params, పరిమాణం(దశాంశం));

    GGML_TENSOR_UNARY_OP_LOCALS
    స్థిరరాశి size_t ts0 = ggml_type_కొలత(src0->type);
    స్థిరరాశి size_t tdst = ggml_type_కొలత(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    స్థిరరాశి int64_t ss0 = nb00 / ts0;
    స్థిరరాశి int64_t ss1 = nb01 / ts0;
    స్థిరరాశి int64_t ss2 = nb02 / ts0;
    స్థిరరాశి int64_t ss3 = nb03 / ts0;
    స్థిరరాశి int64_t ds0 = nb0 / tdst;
    స్థిరరాశి int64_t ds1 = nb1 / tdst;
    స్థిరరాశి int64_t ds2 = nb2 / tdst;
    స్థిరరాశి int64_t ds3 = nb3 / tdst;
    rms_norm_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, main_ప్రవాహం, ctx.device);
}

// Fused rms_norm + scale (the qwen35 GDN l2-norm pair build_gdn_l2_norm emits):
// the scale factor is a host scalar in the GGML_OP_SCALE node's op_params, so
// unlike the mul variants there is no second device tensor to wire up.
శూన్య ggml_sycl_op_rms_norm_scale_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                       ggml_tensor * scale_tensor) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(scale_tensor->type == GGML_TYPE_F32);

    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    స్థిరరాశి దశాంశం * src0_dd = static_cast<స్థిరరాశి దశాంశం *>(src0->దత్తాంశం);
    దశాంశం *       dst_dd  = static_cast<దశాంశం *>(scale_tensor->దత్తాంశం);

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, dst->op_params, పరిమాణం(దశాంశం));
    దశాంశం scale_mul;
    జ్ఞాపకం_నకలు(&scale_mul, scale_tensor->op_params, పరిమాణం(దశాంశం));
    GGML_ASSERT(scale_mul >= 0.0f);

    GGML_TENSOR_UNARY_OP_LOCALS
    స్థిరరాశి size_t ts0 = ggml_type_కొలత(src0->type);
    స్థిరరాశి size_t tdst = ggml_type_కొలత(scale_tensor->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(scale_tensor->nb[0] % tdst == 0 && scale_tensor->nb[1] % tdst == 0 &&
                scale_tensor->nb[2] % tdst == 0 && scale_tensor->nb[3] % tdst == 0);
    స్థిరరాశి int64_t ss0 = nb00 / ts0;
    స్థిరరాశి int64_t ss1 = nb01 / ts0;
    స్థిరరాశి int64_t ss2 = nb02 / ts0;
    స్థిరరాశి int64_t ss3 = nb03 / ts0;
    స్థిరరాశి int64_t ds0 = scale_tensor->nb[0] / tdst;
    స్థిరరాశి int64_t ds1 = scale_tensor->nb[1] / tdst;
    స్థిరరాశి int64_t ds2 = scale_tensor->nb[2] / tdst;
    స్థిరరాశి int64_t ds3 = scale_tensor->nb[3] / tdst;
    rms_norm_scale_f32_sycl(src0_dd, dst_dd, ne00, ne01, ne02, ne03,
        ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, scale_mul, main_ప్రవాహం, ctx.device);
}

శూన్య ggml_sycl_op_rms_norm_fused(ggml_backend_sycl_context & ctx, ggml_tensor * dst, ggml_tensor * mul_tensor) {
    స్థిరరాశి ggml_tensor * rms_norm_src = dst->src[0];
    దశాంశం eps = 0.0f;
    జ్ఞాపకం_నకలు(&eps, dst->op_params, పరిమాణం(దశాంశం));

    స్థిరరాశి దశాంశం *       src0_dd = static_cast<స్థిరరాశి దశాంశం *>(rms_norm_src->దత్తాంశం);
    స్థిరరాశి దశాంశం *       mul_dd  = శూన్యసూచిక;
    స్థిరరాశి ggml_tensor * mul_src = శూన్యసూచిక;
    ఒకవేళ (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<స్థిరరాశి దశాంశం *>(mul_tensor->src[1]->దత్తాంశం);
        mul_src = mul_tensor->src[1];
    } లేకపోతే ఒకవేళ (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<స్థిరరాశి దశాంశం *>(mul_tensor->src[0]->దత్తాంశం);
        mul_src = mul_tensor->src[0];
    } లేకపోతే {
        GGML_ASSERT(అబద్ధం);
    }
    దశాంశం * dst_dd = static_cast<దశాంశం *>(mul_tensor->దత్తాంశం);

    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    స్థిరరాశి int64_t ne00 = rms_norm_src->ne[0];
    స్థిరరాశి int64_t ne01 = rms_norm_src->ne[1];
    స్థిరరాశి int64_t ne02 = rms_norm_src->ne[2];
    స్థిరరాశి int64_t ne03 = rms_norm_src->ne[3];

    స్థిరరాశి size_t ts0 = ggml_type_కొలత(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    స్థిరరాశి int64_t s00 = rms_norm_src->nb[0] / ts0;
    స్థిరరాశి int64_t s01 = rms_norm_src->nb[1] / ts0;
    స్థిరరాశి int64_t s02 = rms_norm_src->nb[2] / ts0;
    స్థిరరాశి int64_t s03 = rms_norm_src->nb[3] / ts0;

    స్థిరరాశి size_t tdst = ggml_type_కొలత(mul_tensor->type);
    GGML_ASSERT(mul_tensor->nb[0] == tdst);
    స్థిరరాశి int64_t d00 = mul_tensor->nb[0] / tdst;
    స్థిరరాశి int64_t d01 = mul_tensor->nb[1] / tdst;
    స్థిరరాశి int64_t d02 = mul_tensor->nb[2] / tdst;
    స్థిరరాశి int64_t d03 = mul_tensor->nb[3] / tdst;

    స్థిరరాశి size_t ts_mul = ggml_type_కొలత(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    స్థిరరాశి int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    స్థిరరాశి int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    స్థిరరాశి int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    స్థిరరాశి పూర్ణాంకం mul_nrows     = mul_src->ne[1];
    స్థిరరాశి పూర్ణాంకం mul_nchannels = mul_src->ne[2];
    స్థిరరాశి పూర్ణాంకం mul_nsamples  = mul_src->ne[3];

    rms_norm_mul_f32_sycl(src0_dd, mul_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples, eps, main_ప్రవాహం, ctx.device);
}

శూన్య ggml_sycl_op_rms_norm_fused_కూడు(ggml_backend_sycl_context & ctx, ggml_tensor * dst,
                                     ggml_tensor * mul_tensor, ggml_tensor * కూడు_tensor) {
    స్థిరరాశి ggml_tensor * rms_norm_src = dst->src[0];
    దశాంశం eps = 0.0f;
    జ్ఞాపకం_నకలు(&eps, dst->op_params, పరిమాణం(దశాంశం));

    స్థిరరాశి దశాంశం *       src0_dd = static_cast<స్థిరరాశి దశాంశం *>(rms_norm_src->దత్తాంశం);
    స్థిరరాశి దశాంశం *       mul_dd  = శూన్యసూచిక;
    స్థిరరాశి ggml_tensor * mul_src = శూన్యసూచిక;
    ఒకవేళ (mul_tensor->src[0] == dst) {
        mul_dd  = static_cast<స్థిరరాశి దశాంశం *>(mul_tensor->src[1]->దత్తాంశం);
        mul_src = mul_tensor->src[1];
    } లేకపోతే ఒకవేళ (mul_tensor->src[1] == dst) {
        mul_dd  = static_cast<స్థిరరాశి దశాంశం *>(mul_tensor->src[0]->దత్తాంశం);
        mul_src = mul_tensor->src[0];
    } లేకపోతే {
        GGML_ASSERT(అబద్ధం);
    }

    స్థిరరాశి దశాంశం *       కూడు_dd  = శూన్యసూచిక;
    స్థిరరాశి ggml_tensor * కూడు_src = శూన్యసూచిక;
    ఒకవేళ (కూడు_tensor->src[0] == mul_tensor) {
        కూడు_dd  = static_cast<స్థిరరాశి దశాంశం *>(కూడు_tensor->src[1]->దత్తాంశం);
        కూడు_src = కూడు_tensor->src[1];
    } లేకపోతే ఒకవేళ (కూడు_tensor->src[1] == mul_tensor) {
        కూడు_dd  = static_cast<స్థిరరాశి దశాంశం *>(కూడు_tensor->src[0]->దత్తాంశం);
        కూడు_src = కూడు_tensor->src[0];
    } లేకపోతే {
        GGML_ASSERT(అబద్ధం);
    }

    దశాంశం * dst_dd = static_cast<దశాంశం *>(కూడు_tensor->దత్తాంశం);

    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    GGML_ASSERT(rms_norm_src->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    GGML_ASSERT(mul_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(కూడు_tensor->type == GGML_TYPE_F32);
    GGML_ASSERT(eps >= 0.0f);

    స్థిరరాశి int64_t ne00 = rms_norm_src->ne[0];
    స్థిరరాశి int64_t ne01 = rms_norm_src->ne[1];
    స్థిరరాశి int64_t ne02 = rms_norm_src->ne[2];
    స్థిరరాశి int64_t ne03 = rms_norm_src->ne[3];

    స్థిరరాశి size_t ts0 = ggml_type_కొలత(rms_norm_src->type);
    GGML_ASSERT(rms_norm_src->nb[0] == ts0);
    స్థిరరాశి int64_t s00 = rms_norm_src->nb[0] / ts0;
    స్థిరరాశి int64_t s01 = rms_norm_src->nb[1] / ts0;
    స్థిరరాశి int64_t s02 = rms_norm_src->nb[2] / ts0;
    స్థిరరాశి int64_t s03 = rms_norm_src->nb[3] / ts0;

    స్థిరరాశి size_t tdst = ggml_type_కొలత(కూడు_tensor->type);
    GGML_ASSERT(కూడు_tensor->nb[0] == tdst);
    స్థిరరాశి int64_t d00 = కూడు_tensor->nb[0] / tdst;
    స్థిరరాశి int64_t d01 = కూడు_tensor->nb[1] / tdst;
    స్థిరరాశి int64_t d02 = కూడు_tensor->nb[2] / tdst;
    స్థిరరాశి int64_t d03 = కూడు_tensor->nb[3] / tdst;

    స్థిరరాశి size_t ts_mul = ggml_type_కొలత(mul_src->type);
    GGML_ASSERT(mul_src->nb[0] == ts_mul);
    స్థిరరాశి int64_t mul_s01 = mul_src->nb[1] / ts_mul;
    స్థిరరాశి int64_t mul_s02 = mul_src->nb[2] / ts_mul;
    స్థిరరాశి int64_t mul_s03 = mul_src->nb[3] / ts_mul;
    స్థిరరాశి పూర్ణాంకం mul_nrows     = mul_src->ne[1];
    స్థిరరాశి పూర్ణాంకం mul_nchannels = mul_src->ne[2];
    స్థిరరాశి పూర్ణాంకం mul_nsamples  = mul_src->ne[3];

    స్థిరరాశి size_t ts_కూడు = ggml_type_కొలత(కూడు_src->type);
    GGML_ASSERT(కూడు_src->nb[0] == ts_కూడు);
    స్థిరరాశి int64_t కూడు_s01 = కూడు_src->nb[1] / ts_కూడు;
    స్థిరరాశి int64_t కూడు_s02 = కూడు_src->nb[2] / ts_కూడు;
    స్థిరరాశి int64_t కూడు_s03 = కూడు_src->nb[3] / ts_కూడు;
    స్థిరరాశి పూర్ణాంకం కూడు_nrows     = కూడు_src->ne[1];
    స్థిరరాశి పూర్ణాంకం కూడు_nchannels = కూడు_src->ne[2];
    స్థిరరాశి పూర్ణాంకం కూడు_nsamples  = కూడు_src->ne[3];

    rms_norm_mul_కూడు_f32_sycl(src0_dd, mul_dd, కూడు_dd, dst_dd, ne00, ne01, ne02, ne03,
        s00, s01, s02, s03, d00, d01, d02, d03,
        mul_s01, mul_s02, mul_s03, mul_nrows, mul_nchannels, mul_nsamples,
        కూడు_s01, కూడు_s02, కూడు_s03, కూడు_nrows, కూడు_nchannels, కూడు_nsamples, eps, main_ప్రవాహం, ctx.device);
}

శూన్య ggml_sycl_op_rms_norm_వెనుక(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);

    GGML_ASSERT(dst->src[0]->type == GGML_TYPE_F32); // dz
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_F32); // x
    GGML_ASSERT(dst->type         == GGML_TYPE_F32);

    దశాంశం eps = 1e-5f;
    std::జ్ఞాపకం_నకలు(&eps, dst->op_params, పరిమాణం(దశాంశం));
    ఒకవేళ (!(eps > 0.0f) || !std::isfinite(eps)) eps = 1e-5f;

    స్థిరరాశి దశాంశం * g_పునాది  = static_cast<స్థిరరాశి దశాంశం *>(dst->src[0]->దత్తాంశం); // dz
    స్థిరరాశి దశాంశం * x_పునాది  = static_cast<స్థిరరాశి దశాంశం *>(dst->src[1]->దత్తాంశం); // x
          దశాంశం * dx_పునాది = static_cast<      దశాంశం *>(dst->దత్తాంశం);

    స్థిరరాశి int64_t D  = dst->ne[0];
    స్థిరరాశి int64_t n1 = dst->ne[1], n2 = dst->ne[2], n3 = dst->ne[3]; (శూన్య) n3;
    స్థిరరాశి int64_t N  = ggml_nrows(dst);
    ఒకవేళ (D == 0 || N == 0) తిరిగి;

    స్థిరరాశి ggml_tensor *G = dst->src[0];
    స్థిరరాశి ggml_tensor *X = dst->src[1];
    స్థిరరాశి పూర్ణాంకం ts = (పూర్ణాంకం) ggml_type_కొలత(X->type);
    GGML_ASSERT((size_t) X->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) G->nb[0]   == (size_t) ts);
    GGML_ASSERT((size_t) dst->nb[0] == (size_t) ts);

    స్థిరరాశి int64_t xs1 = X->nb[1] / ts, xs2 = X->nb[2] / ts, xs3 = X->nb[3] / ts;
    స్థిరరాశి int64_t gs1 = G->nb[1] / ts, gs2 = G->nb[2] / ts, gs3 = G->nb[3] / ts;
    స్థిరరాశి int64_t ds1 = dst->nb[1] / ts, ds2 = dst->nb[2] / ts, ds3 = dst->nb[3] / ts;

    dpct::వరుస_ptr q = ctx.stream();

    // work-group size: multiple of WARP_SIZE, capped by device and 256, and not larger than D
    స్థిరరాశి పూర్ణాంకం device_max_wg = ggml_sycl_info().max_work_సమూహం_sizes[ctx.device];
    స్వయంచాలక roundup = [](పూర్ణాంకం v, పూర్ణాంకం m) { తిరిగి ((v + m - 1) / m) * m; };
    పూర్ణాంకం wg_cap = 256;
    ఒకవేళ (device_max_wg > 0) wg_cap = std::min(wg_cap, device_max_wg);
    పూర్ణాంకం WG = std::max(WARP_కొలత, std::min(roundup((పూర్ణాంకం)std::min<int64_t>(D, wg_cap), WARP_కొలత), wg_cap));

    // FP32 path: per-thread compensated accumulation + hierarchical reduction
    q->submit([&](sycl::handler &cgh) {
        స్థిరరాశి పూర్ణాంకం nwarps_loc = std::max(1, WG / WARP_కొలత);
        // store one partial value per warp (xx and xg) for cross-warp reduction
        స్వయంచాలక l_xx   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);
        స్వయంచాలక l_xg   = sycl::local_accessor<sycl::float2, 1>(sycl::range<1>(nwarps_loc), cgh);

        cgh.parallel_for(
            sycl::nd_range<3>(sycl::range<3>(1, 1, N) * sycl::range<3>(1, 1, WG),
                              sycl::range<3>(1, 1, WG)),
            [=](sycl::nd_అంశం<3> అంశం_ct1) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                స్థిరరాశి పూర్ణాంకం అడ్డువరుస = అంశం_ct1.get_సమూహం(2);
                స్థిరరాశి పూర్ణాంకం tid = అంశం_ct1.get_local_id(2);

                స్థిరరాశి int64_t i1 = అడ్డువరుస % n1;
                స్థిరరాశి int64_t i2 = (అడ్డువరుస / n1) % n2;
                స్థిరరాశి int64_t i3 = అడ్డువరుస / (n1 * n2);

                స్థిరరాశి దశాంశం *__restrict x_అడ్డువరుస = x_పునాది + i3 * xs3 + i2 * xs2 + i1 * xs1;
                స్థిరరాశి దశాంశం *__restrict g_అడ్డువరుస = g_పునాది + i3 * gs3 + i2 * gs2 + i1 * gs1;
                దశాంశం *__restrict d_అడ్డువరుస       = dx_పునాది + i3 * ds3 + i2 * ds2 + i1 * ds1;

                // per-thread accumulation (compensated by default)
                దశాంశం కూడిక_xx = 0.f, కూడిక_xg = 0.f;
#నిర్వచితం_కాకపోతే GGML_SYCL_RMS_వెనుక_FAST
                దశాంశం c_xx = 0.f, c_xg = 0.f;
#అయితే_ముగింపు
                కొరకు (int64_t నిలువు = tid; నిలువు < D; నిలువు += WG) {
                    స్థిరరాశి దశాంశం xv = x_అడ్డువరుస[నిలువు];
                    స్థిరరాశి దశాంశం gv = g_అడ్డువరుస[నిలువు];
#నిర్వచితమైతే GGML_SYCL_RMS_వెనుక_FAST
                    కూడిక_xx += xv * xv;
                    కూడిక_xg += xv * gv;
#లేకపోతే
                    దశాంశం y1 = xv * xv - c_xx;
                    దశాంశం t1 = కూడిక_xx + y1;
                    c_xx = (t1 - కూడిక_xx) - y1;
                    కూడిక_xx = t1;

                    దశాంశం y2 = xv * gv - c_xg;
                    దశాంశం t2 = కూడిక_xg + y2;
                    c_xg = (t2 - కూడిక_xg) - y2;
                    కూడిక_xg = t2;
#అయితే_ముగింపు
                }

                // warp-level reduction
                sycl::float2 xx = sycl::float2(కూడిక_xx,
#నిర్వచితం_కాకపోతే GGML_SYCL_RMS_వెనుక_FAST
                    c_xx
#లేకపోతే
                    0.f
#అయితే_ముగింపు
                );
                sycl::float2 xg = sycl::float2(కూడిక_xg,
#నిర్వచితం_కాకపోతే GGML_SYCL_RMS_వెనుక_FAST
                    c_xg
#లేకపోతే
                    0.f
#అయితే_ముగింపు
                );
                xx = warp_reduce_కూడిక(xx, అంశం_ct1);
                xg = warp_reduce_కూడిక(xg, అంశం_ct1);

                // cross-warp reduction using local memory (single barrier)
                స్థిరరాశి స్వయంచాలక sub_సమూహం = అంశం_ct1.get_sub_సమూహం();
                స్థిరరాశి స్వయంచాలక sg_id     = sub_సమూహం.get_సమూహం_రేఖీయ_id();
                స్థిరరాశి స్వయంచాలక wi_లోపల_sg  = sub_సమూహం.get_local_రేఖీయ_id();
                స్థిరరాశి పూర్ణాంకం nthreads   = అంశం_ct1.get_local_range(2);
                స్థిరరాశి పూర్ణాంకం nwarps     = nthreads / WARP_కొలత;

                sycl::float2 xx_మొత్తం = xx;
                sycl::float2 xg_మొత్తం = xg;
                ఒకవేళ (nwarps > 1) {
                    ఒకవేళ (wi_లోపల_sg == 0) {
                        l_xx[sg_id] = xx;
                        l_xg[sg_id] = xg;
                    }
                    అంశం_ct1.barrier(sycl::access::fence_space::local_space);

                    ఒకవేళ (sg_id == 0) {
                        స్థిరరాశి గుర్తులేని wi_u = wi_లోపల_sg;
                        sycl::float2 xx_మొదటి = (wi_u < static_cast<గుర్తులేని>(nwarps)) ? l_xx[wi_u] : sycl::float2(0.f, 0.f);
                        sycl::float2 xg_మొదటి = (wi_u < static_cast<గుర్తులేని>(nwarps)) ? l_xg[wi_u] : sycl::float2(0.f, 0.f);
                        xx_మొత్తం = warp_reduce_కూడిక(xx_మొదటి, అంశం_ct1);
                        xg_మొత్తం = warp_reduce_కూడిక(xg_మొదటి, అంశం_ct1);
                    } లేకపోతే {
                        // other subgroups keep their local totals; they'll be ignored
                        xx_మొత్తం = xx;
                        xg_మొత్తం = xg;
                    }
                    // ensure all threads see the first-subgroup result via broadcast below
                }

                // compute inv_r and coeff once per row and broadcast to the whole work-group
                దశాంశం inv_r = 0.f;
                దశాంశం coeff = 0.f;
                ఒకవేళ (tid == 0) {
                    స్థిరరాశి దశాంశం కూడిక_xx_f  = xx_మొత్తం.x() + xx_మొత్తం.y();
                    స్థిరరాశి దశాంశం కూడిక_xdz_f = xg_మొత్తం.x() + xg_మొత్తం.y();
                    స్థిరరాశి దశాంశం సగటు_eps  = కూడిక_xx_f / (దశాంశం) D + eps;
                    స్థిరరాశి దశాంశం కూడిక_eps   = కూడిక_xx_f + eps * (దశాంశం) D;
                    inv_r = sycl::rsqrt(సగటు_eps);
                    coeff = -కూడిక_xdz_f / కూడిక_eps;
                }
                inv_r = sycl::సమూహం_ప్రసారం(అంశం_ct1.get_సమూహం(), inv_r);
                coeff = sycl::సమూహం_ప్రసారం(అంశం_ct1.get_సమూహం(), coeff);

                కొరకు (int64_t నిలువు = tid; నిలువు < D; నిలువు += WG) {
                    d_అడ్డువరుస[నిలువు] = (g_అడ్డువరుస[నిలువు] + coeff * x_అడ్డువరుస[నిలువు]) * inv_r;
                }
            });
    });

}

శూన్య ggml_sycl_op_l2_norm(ggml_backend_sycl_context& ctx, ggml_tensor* dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి దశాంశం * src0_d = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం;
    దశాంశం * dst_d = (దశాంశం *) dst->దత్తాంశం;
    dpct::వరుస_ptr     stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_TENSOR_UNARY_OP_LOCALS;

    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, dst->op_params, పరిమాణం(దశాంశం));
    GGML_ASSERT(eps >= 0.0f);

    స్థిరరాశి size_t ts0 = ggml_type_కొలత(src0->type);
    స్థిరరాశి size_t tdst = ggml_type_కొలత(dst->type);
    GGML_ASSERT(nb00 % ts0 == 0 && nb01 % ts0 == 0 && nb02 % ts0 == 0 && nb03 % ts0 == 0);
    GGML_ASSERT(nb0 % tdst == 0 && nb1 % tdst == 0 && nb2 % tdst == 0 && nb3 % tdst == 0);
    స్థిరరాశి int64_t ss0 = nb00 / ts0;
    స్థిరరాశి int64_t ss1 = nb01 / ts0;
    స్థిరరాశి int64_t ss2 = nb02 / ts0;
    స్థిరరాశి int64_t ss3 = nb03 / ts0;
    స్థిరరాశి int64_t ds0 = nb0 / tdst;
    స్థిరరాశి int64_t ds1 = nb1 / tdst;
    స్థిరరాశి int64_t ds2 = nb2 / tdst;
    స్థిరరాశి int64_t ds3 = nb3 / tdst;

    /*support both WARP_SIZE or WARP_32_SIZE in code
      choose by hardware for better performance
    */
    l2_norm_f32_sycl<WARP_కొలత>(src0_d, dst_d, ne00, ne01, ne02, ne03,
            ss0, ss1, ss2, ss3, ds0, ds1, ds2, ds3, eps, stream, ctx.device);
}

// nodes[0..count) are independent, same-shape, same-eps, same-nb L2_NORM ops validated
// by the caller; requires ncols < 1024 (the warp reduction path).
శూన్య ggml_sycl_l2_norm_batch(ggml_backend_sycl_context & ctx, ggml_tensor ** nodes, పూర్ణాంకం count) {
    స్థిరరాశి ggml_tensor * s0 = nodes[0]->src[0];
    స్థిరరాశి పూర్ణాంకం ncols       = (పూర్ణాంకం) s0->ne[0];
    స్థిరరాశి పూర్ణాంకం nrows_మొత్తం = (పూర్ణాంకం) ggml_nrows(s0);
    దశాంశం eps;
    జ్ఞాపకం_నకలు(&eps, nodes[0]->op_params, పరిమాణం(దశాంశం));
    GGML_ASSERT(eps >= 0.0f);

    l2_batch_ptrs p{};
    కొరకు (పూర్ణాంకం t = 0; t < count; ++t) {
        p.src[t] = (స్థిరరాశి దశాంశం *) nodes[t]->src[0]->దత్తాంశం;
        p.dst[t] = (దశాంశం *) nodes[t]->దత్తాంశం;
    }

    స్థిరరాశి ggml_tensor * d0 = nodes[0];
    స్థిరరాశి size_t        ts = ggml_type_కొలత(GGML_TYPE_F32);
    l2_batch_strides    st{};
    st.ne1 = (పూర్ణాంకం) s0->ne[1];
    st.ne2 = (పూర్ణాంకం) s0->ne[2];
    st.ss0 = s0->nb[0] / ts; st.ss1 = s0->nb[1] / ts; st.ss2 = s0->nb[2] / ts; st.ss3 = s0->nb[3] / ts;
    st.ds0 = d0->nb[0] / ts; st.ds1 = d0->nb[1] / ts; st.ds2 = d0->nb[2] / ts; st.ds3 = d0->nb[3] / ts;

    l2_norm_f32_batch_sycl<WARP_కొలత>(p, st, count, ncols, nrows_మొత్తం, eps, ctx.stream());
}
