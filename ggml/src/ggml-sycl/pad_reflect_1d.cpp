#అన్తర్భావ "pad_reflect_1d.hpp"

స్థిర శూన్య pad_reflect_1d_kernel_f32(
    స్థిరరాశి శూన్య *__restrict__ src0, శూన్య *__restrict__ dst, స్థిరరాశి int64_t ne0,
    స్థిరరాశి int64_t ne00, స్థిరరాశి sycl::uint3 ne01, స్థిరరాశి int64_t ne02,
    స్థిరరాశి int64_t ne03, స్థిరరాశి int64_t nb00, స్థిరరాశి int64_t nb01,
    స్థిరరాశి int64_t nb02, స్థిరరాశి int64_t nb03, స్థిరరాశి int64_t nb0,
    స్థిరరాశి int64_t nb1, స్థిరరాశి int64_t nb2, స్థిరరాశి int64_t nb3, స్థిరరాశి పూర్ణాంకం p0,
    స్థిరరాశి పూర్ణాంకం p1, sycl::nd_అంశం<3> అంశం_ct1) {

    స్థిరరాశి int64_t i3 = అంశం_ct1.get_సమూహం(0);
    స్థిరరాశి int64_t i2 = అంశం_ct1.get_సమూహం(1);

    స్థిరరాశి sycl::uint2 div_mod_packed =
        fast_div_modulo(అంశం_ct1.get_సమూహం(2), ne01);
    స్థిరరాశి int64_t tile1 = div_mod_packed.y();
    స్థిరరాశి int64_t tile0 = div_mod_packed.x();
    స్థిరరాశి int64_t i1 = tile1;
    స్థిరరాశి int64_t i0 =
        అంశం_ct1.get_local_id(2) + tile0 * అంశం_ct1.get_local_range(2);

    ఒకవేళ (i0 >= ne0 || i1 >= ne01.z() || i2 >= ne02 || i3 >= ne03) {
        తిరిగి;
    }

    స్థిరరాశి అక్షరం *src0_ptr =
        (స్థిరరాశి అక్షరం *)src0 + i3 * nb03 + i2 * nb02 + i1 * nb01;
    అక్షరం *dst_ptr = (అక్షరం *)dst + i3 * nb3 + i2 * nb2 + i1 * nb1;

    స్థిరరాశి int64_t rel_i0 = i0 - p0; // relative i0 in src0
    int64_t src_idx;

    ఒకవేళ (rel_i0 < 0) {
        // Left padding - reflect
        src_idx = -rel_i0;
    } లేకపోతే ఒకవేళ (rel_i0 < ne00) {
        // Middle - copy
        src_idx = rel_i0;
    } లేకపోతే {
        // Right padding - reflect
        src_idx = 2 * ne00 - 2 - rel_i0;
    }
    స్థిరరాశి దశాంశం value = *(స్థిరరాశి దశాంశం *)(src0_ptr + src_idx * nb00);
    *(దశాంశం *)(dst_ptr + i0 * nb0) = value;

    GGML_వాడని(p1);
}

శూన్య ggml_sycl_op_pad_reflect_1d(ggml_backend_sycl_context &ctx,
                                 ggml_tensor *dst) {

    స్థిరరాశి ggml_tensor *src0 = dst->src[0];
    dpct::వరుస_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    స్థిరరాశి int32_t *opts = (స్థిరరాశి int32_t *)dst->op_params;
    స్థిరరాశి పూర్ణాంకం p0 = opts[0];
    స్థిరరాశి పూర్ణాంకం p1 = opts[1];

    స్థిరరాశి int64_t ne00 = src0->ne[0];
    స్థిరరాశి int64_t ne01 = src0->ne[1];
    స్థిరరాశి sycl::uint3 ne01_packed = init_fastdiv_విలువలు(ne01);
    స్థిరరాశి int64_t ne02 = src0->ne[2];
    స్థిరరాశి int64_t ne03 = src0->ne[3];

    స్థిరరాశి int64_t ne0 = dst->ne[0];

    GGML_ASSERT(ne0 == ne00 + p0 + p1);

    constexpr int64_t bx = SYCL_PAD_REFLECT_1_D_ఖండం_కొలత;
    స్థిరరాశి int64_t tiles0 = (ne0 + bx - 1) / bx;
    స్థిరరాశి dpct::dim3 grid_dims((గుర్తులేని)(ne01 * tiles0), (గుర్తులేని)ne02,
                               (గుర్తులేని)ne03);
    స్థిరరాశి dpct::dim3 ఖండం_dims((గుర్తులేని)bx, 1, 1);

    stream->submit([&](sycl::handler &cgh) {
        స్వయంచాలక src0_దత్తాంశం_ct0 = src0->దత్తాంశం;
        స్వయంచాలక dst_దత్తాంశం_ct1 = dst->దత్తాంశం;
        స్వయంచాలక src0_nb_ct7 = src0->nb[0];
        స్వయంచాలక src0_nb_ct8 = src0->nb[1];
        స్వయంచాలక src0_nb_ct9 = src0->nb[2];
        స్వయంచాలక src0_nb_ct10 = src0->nb[3];
        స్వయంచాలక dst_nb_ct11 = dst->nb[0];
        స్వయంచాలక dst_nb_ct12 = dst->nb[1];
        స్వయంచాలక dst_nb_ct13 = dst->nb[2];
        స్వయంచాలక dst_nb_ct14 = dst->nb[3];

        cgh.parallel_for(sycl::nd_range<3>(grid_dims * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> అంశం_ct1) {
                             pad_reflect_1d_kernel_f32(
                                 src0_దత్తాంశం_ct0, dst_దత్తాంశం_ct1, ne0, ne00,
                                 ne01_packed, ne02, ne03, src0_nb_ct7,
                                 src0_nb_ct8, src0_nb_ct9, src0_nb_ct10,
                                 dst_nb_ct11, dst_nb_ct12, dst_nb_ct13,
                                 dst_nb_ct14, p0, p1, అంశం_ct1);
                         });
    });
}
