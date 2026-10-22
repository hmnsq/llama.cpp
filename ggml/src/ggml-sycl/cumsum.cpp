#అన్తర్భావ "cumsum.hpp"
#అన్తర్భావ "common.hpp"

#అన్తర్భావ <algorithm>

#నిర్వచించు SYCL_CUMSUM_ఖండం_కొలత 256

స్థిర __dpct_inline__ దశాంశం warp_ఉపసర్గ_inclusive_కూడిక_f32(దశాంశం x, స్థిరరాశి sycl::nd_అంశం<3> & అంశం) {
    తిరిగి sycl::inclusive_పరిశీలన_over_సమూహం(అంశం.get_sub_సమూహం(), x, sycl::plus<దశాంశం>());
}

స్థిర శూన్య cumsum_f32_kernel(
        స్థిరరాశి దశాంశం * __restrict__ src, దశాంశం * __restrict__ dst,
        స్థిరరాశి int64_t ne00, స్థిరరాశి int64_t ne01, స్థిరరాశి int64_t ne02, స్థిరరాశి int64_t ne03,
        స్థిరరాశి int64_t s01, స్థిరరాశి int64_t s02, స్థిరరాశి int64_t s03,
        స్థిరరాశి int64_t  d1, స్థిరరాశి int64_t  d2, స్థిరరాశి int64_t  d3,
        స్థిరరాశి sycl::nd_అంశం<3> & అంశం, దశాంశం * smem) {

    స్థిరరాశి పూర్ణాంకం tid = అంశం.get_local_id(2);
    స్థిరరాశి పూర్ణాంకం ఖండం_కొలత_3 = అంశం.get_local_range(2);
    స్థిరరాశి పూర్ణాంకం lane = tid % WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం warp = tid / WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం warps_చొప్పున_ఖండం = ఖండం_కొలత_3 / WARP_కొలత;

    దశాంశం * s_vals      = smem;
    దశాంశం * s_warp_sums = smem + ఖండం_కొలత_3;
    దశాంశం * s_వాహకం     = smem + ఖండం_కొలత_3 + warps_చొప్పున_ఖండం;

    ఒకవేళ (tid == 0) {
        s_వాహకం[0] = 0.0f;
    }
    అంశం.barrier(sycl::access::fence_space::local_space);

    స్థిరరాశి int64_t i3 = అంశం.get_సమూహం(0);
    స్థిరరాశి int64_t i2 = అంశం.get_సమూహం(1);
    స్థిరరాశి int64_t i1 = అంశం.get_సమూహం(2);
    ఒకవేళ (i3 >= ne03 || i2 >= ne02 || i1 >= ne01) {
        తిరిగి;
    }

    స్థిరరాశి దశాంశం * src_అడ్డువరుస = src + i1 * s01 + i2 * s02 + i3 * s03;
    దశాంశం       * dst_అడ్డువరుస = dst + i1 * d1  + i2 * d2  + i3 * d3;

    constexpr పూర్ణాంకం అంకె_unroll = 4;
    దశాంశం తాత్కాలిక[అంకె_unroll];

    కొరకు (int64_t i = 0; i < ne00; i += అంకె_unroll * ఖండం_కొలత_3) {
        int64_t idx = i + tid * అంకె_unroll;

        తాత్కాలిక[0] = (idx < ne00 ? src_అడ్డువరుస[idx] : 0.0f);
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 1; j < అంకె_unroll; j++) {
            తాత్కాలిక[j] = తాత్కాలిక[j - 1];
            ఒకవేళ (idx + j < ne00) {
                తాత్కాలిక[j] += src_అడ్డువరుస[idx + j];
            }
        }

        దశాంశం val = (idx < ne00) ? తాత్కాలిక[అంకె_unroll - 1] : 0.0f;

        val = warp_ఉపసర్గ_inclusive_కూడిక_f32(val, అంశం);
        s_vals[tid] = val;

        ఒకవేళ (lane == WARP_కొలత - 1) {
            s_warp_sums[warp] = val;
        }
        అంశం.barrier(sycl::access::fence_space::local_space);

        ఒకవేళ (warp == 0) {
            దశాంశం w = (tid < warps_చొప్పున_ఖండం) ? s_warp_sums[tid] : 0.0f;
            దశాంశం inc = warp_ఉపసర్గ_inclusive_కూడిక_f32(w, అంశం);
            ఒకవేళ (tid < warps_చొప్పున_ఖండం) {
                s_warp_sums[tid] = inc - w;
            }
            ఒకవేళ (tid == warps_చొప్పున_ఖండం - 1) {
                s_వాహకం[1] = inc;
            }
        }
        అంశం.barrier(sycl::access::fence_space::local_space);

        దశాంశం వాహకం = s_వాహకం[0];
        దశాంశం final_స్థానభ్రంశం = s_vals[tid] + s_warp_sums[warp] + వాహకం - తాత్కాలిక[అంకె_unroll - 1];

#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < అంకె_unroll; j++) {
            ఒకవేళ (idx + j < ne00) {
                dst_అడ్డువరుస[idx + j] = తాత్కాలిక[j] + final_స్థానభ్రంశం;
            }
        }

        అంశం.barrier(sycl::access::fence_space::local_space);

        ఒకవేళ (tid == 0) {
            s_వాహకం[0] += s_వాహకం[1];
        }
    }
}

అంతఃవరుస శూన్య ggml_sycl_op_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    dpct::వరుస_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    స్థిరరాశి దశాంశం * src_d = static_cast<స్థిరరాశి దశాంశం *>(src0->దత్తాంశం);
    దశాంశం       * dst_d = static_cast<దశాంశం *>(dst->దత్తాంశం);

    స్థిరరాశి int64_t ne00 = src0->ne[0];
    స్థిరరాశి int64_t ne01 = src0->ne[1];
    స్థిరరాశి int64_t ne02 = src0->ne[2];
    స్థిరరాశి int64_t ne03 = src0->ne[3];

    స్థిరరాశి size_t ts = పరిమాణం(దశాంశం);
    స్థిరరాశి int64_t s01 = src0->nb[1] / ts;
    స్థిరరాశి int64_t s02 = src0->nb[2] / ts;
    స్థిరరాశి int64_t s03 = src0->nb[3] / ts;
    స్థిరరాశి int64_t d1  = dst->nb[1] / ts;
    స్థిరరాశి int64_t d2  = dst->nb[2] / ts;
    స్థిరరాశి int64_t d3  = dst->nb[3] / ts;

    స్థిరరాశి పూర్ణాంకం అంకె_warps = (ne00 + WARP_కొలత - 1) / WARP_కొలత;
    పూర్ణాంకం ఖండం_కొలత_3 = అంకె_warps * WARP_కొలత;
    ఖండం_కొలత_3 = std::min(ఖండం_కొలత_3, SYCL_CUMSUM_ఖండం_కొలత);
    స్థిరరాశి పూర్ణాంకం warps_చొప్పున_ఖండం = ఖండం_కొలత_3 / WARP_కొలత;
    స్థిరరాశి పూర్ణాంకం smem_కొలత = ఖండం_కొలత_3 + warps_చొప్పున_ఖండం + 2;

    స్థిరరాశి sycl::range<3> grid(ne03, ne02, ne01);
    స్థిరరాశి sycl::range<3> ఖండం(1, 1, ఖండం_కొలత_3);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<దశాంశం, 1> smem_acc(sycl::range<1>(smem_కొలత), cgh);
        cgh.parallel_for(
            sycl::nd_range<3>(grid * ఖండం, ఖండం),
            [=](sycl::nd_అంశం<3> అంశం) [[sycl::reqd_sub_సమూహం_కొలత(WARP_కొలత)]] {
                cumsum_f32_kernel(src_d, dst_d, ne00, ne01, ne02, ne03,
                                  s01, s02, s03, d1, d2, d3,
                                  అంశం, get_సూచిక(smem_acc));
            });
    });
}

శూన్య ggml_sycl_cumsum(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_cumsum(ctx, dst);
}
