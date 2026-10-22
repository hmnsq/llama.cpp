#అన్తర్భావ "col2im-1d.hpp"

మూస <రకం_పేరు T>
స్థిర శూన్య col2im_1d_sycl(
        స్థిరరాశి T * నిలువు,
        T * dst,
        స్థిరరాశి పూర్ణాంకం T_లోపల,
        స్థిరరాశి sycl::uint3 T_బహిర్గతం_fd,
        స్థిరరాశి పూర్ణాంకం K,
        స్థిరరాశి పూర్ణాంకం K_OC,
        స్థిరరాశి int32_t s0,
        స్థిరరాశి int32_t p0,
        స్థిరరాశి పూర్ణాంకం మొత్తం,
        dpct::వరుస_ptr stream) {

    స్థిరరాశి uint32_t ఖండం_కొలత_3 = SYCL_COL2_IM_1_D_ఖండం_కొలత;
    స్థిరరాశి uint32_t అంకె_ఖండాలు = (uint32_t) ((మొత్తం + ఖండం_కొలత_3 - 1) / ఖండం_కొలత_3);

    stream->parallel_for(
        sycl::nd_range<3>(
            sycl::range<3>(1, 1, అంకె_ఖండాలు * ఖండం_కొలత_3),
            sycl::range<3>(1, 1, ఖండం_కొలత_3)),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            స్థిరరాశి పూర్ణాంకం idx = (పూర్ణాంకం) అంశం_ct1.get_global_id(2);
            ఒకవేళ (idx >= మొత్తం) {
                తిరిగి;
            }

            స్థిరరాశి sycl::uint2 qr = fast_div_modulo((uint32_t) idx, T_బహిర్గతం_fd);
            స్థిరరాశి పూర్ణాంకం oc    = (పూర్ణాంకం) qr.x();
            స్థిరరాశి పూర్ణాంకం t_బహిర్గతం = (పూర్ణాంకం) qr.y();
            స్థిరరాశి పూర్ణాంకం t_abs = t_బహిర్గతం + p0;

            పూర్ణాంకం t_లోపల_min = (t_abs - K + s0) / s0;
            ఒకవేళ (t_లోపల_min < 0) {
                t_లోపల_min = 0;
            }
            పూర్ణాంకం t_లోపల_max = t_abs / s0;
            ఒకవేళ (t_లోపల_max >= T_లోపల) {
                t_లోపల_max = T_లోపల - 1;
            }

            దశాంశం కూడిక = 0.0f;
            కొరకు (పూర్ణాంకం t_లోపల = t_లోపల_min; t_లోపల <= t_లోపల_max; ++t_లోపల) {
                స్థిరరాశి పూర్ణాంకం k = t_abs - t_లోపల * s0;
                కూడిక += static_cast<దశాంశం>(నిలువు[(oc * K + k) + t_లోపల * K_OC]);
            }

            dst[idx] = static_cast<T>(కూడిక);
        });
}

శూన్య ggml_sycl_op_col2im_1d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(src0 != శూన్యసూచిక);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->type == dst->type);

    స్థిరరాశి int32_t s0 = ((స్థిరరాశి int32_t *) dst->op_params)[0];
    స్థిరరాశి int32_t OC = ((స్థిరరాశి int32_t *) dst->op_params)[1];
    స్థిరరాశి int32_t p0 = ((స్థిరరాశి int32_t *) dst->op_params)[2];

    స్థిరరాశి పూర్ణాంకం K_OC  = (పూర్ణాంకం) src0->ne[0];
    స్థిరరాశి పూర్ణాంకం T_లోపల  = (పూర్ణాంకం) src0->ne[1];
    స్థిరరాశి పూర్ణాంకం K     = K_OC / OC;
    స్థిరరాశి పూర్ణాంకం T_బహిర్గతం = (పూర్ణాంకం) dst->ne[0];

    GGML_ASSERT(OC > 0);
    GGML_ASSERT(K_OC % OC == 0);

    స్థిరరాశి sycl::uint3 T_బహిర్గతం_fd = init_fastdiv_విలువలు((uint32_t) T_బహిర్గతం);

    స్థిరరాశి పూర్ణాంకం మొత్తం = T_బహిర్గతం * OC;

    dpct::వరుస_ptr stream = ctx.stream();

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F32:
            col2im_1d_sycl<దశాంశం>(
                (స్థిరరాశి దశాంశం *) src0->దత్తాంశం,
                (దశాంశం *) dst->దత్తాంశం,
                T_లోపల, T_బహిర్గతం_fd, K, K_OC, s0, p0, మొత్తం, stream);
            ఆపు;
        స్థితి GGML_TYPE_F16:
            col2im_1d_sycl<sycl::half>(
                (స్థిరరాశి sycl::half *) src0->దత్తాంశం,
                (sycl::half *) dst->దత్తాంశం,
                T_లోపల, T_బహిర్గతం_fd, K, K_OC, s0, p0, మొత్తం, stream);
            ఆపు;
#నిర్వచితమైతే GGML_SYCL_HAS_BF16
        స్థితి GGML_TYPE_BF16:
            col2im_1d_sycl<sycl::ext::oneapi::bfloat16>(
                (స్థిరరాశి sycl::ext::oneapi::bfloat16 *) src0->దత్తాంశం,
                (sycl::ext::oneapi::bfloat16 *) dst->దత్తాంశం,
                T_లోపల, T_బహిర్గతం_fd, K, K_OC, s0, p0, మొత్తం, stream);
            ఆపు;
#అయితే_ముగింపు
        అప్రమేయ:
            GGML_ABORT("col2im_1d: unsupported type %d", src0->type);
    }
}
