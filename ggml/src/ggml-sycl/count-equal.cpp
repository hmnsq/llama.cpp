#అన్తర్భావ "count-equal.hpp"

#అన్తర్భావ <cstdint>

మూస <రకం_పేరు T>
స్థిర శూన్య లెక్క_equal(స్థిరరాశి T *__restrict__ x, స్థిరరాశి T *__restrict__ y,
                        int64_t *__restrict__ dst, స్థిరరాశి int64_t dk,
                        స్థిరరాశి int64_t k) {
    స్వయంచాలక అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి int64_t i0 = (int64_t)అంశం_ct1.get_సమూహం(2) * dk;
    స్థిరరాశి int64_t i1 = sycl::min(i0 + dk, k);

    పూర్ణాంకం nequal = 0;

    కొరకు (int64_t i = i0 + అంశం_ct1.get_local_id(2); i < i1; i += WARP_కొలత) {
        స్థిరరాశి T xi = x[i];
        స్థిరరాశి T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_కూడిక<WARP_కొలత>(nequal);

    ఒకవేళ (అంశం_ct1.get_local_id(2) != 0) {
        తిరిగి;
    }

    dpct::atomic_fetch_add<sycl::access::చిరునామా_space::generic_space>(
        (పూర్ణాంకం *)dst, nequal);
}

శూన్య ggml_sycl_లెక్క_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->దత్తాంశం;

    dpct::వరుస_ptr stream = ctx.stream();
    స్థిరరాశి పూర్ణాంకం id       = get_current_device_id();
    స్థిరరాశి పూర్ణాంకం nsm = ggml_sycl_info().devices[id].nsm;

    స్థిరరాశి int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    స్థిరరాశి int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_లెక్క_EQUAL_CHUNK_కొలత);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dst_d, 0, ggml_nbytes(dst))));

    స్థిరరాశి dpct::dim3 ఖండం_dims(WARP_కొలత, 1, 1);
    స్థిరరాశి dpct::dim3 ఖండం_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_లెక్క_EQUAL_CHUNK_కొలత - 1) /
                                       SYCL_లెక్క_EQUAL_CHUNK_కొలత),
        1, 1);

    ఎంపిక (src0->type) {
    స్థితి GGML_TYPE_I32: {
        స్థిరరాశి పూర్ణాంకం *src0_d = (స్థిరరాశి పూర్ణాంకం *)src0->దత్తాంశం;
        స్థిరరాశి పూర్ణాంకం *src1_d = (స్థిరరాశి పూర్ణాంకం *)src1->దత్తాంశం;
        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) {
                లెక్క_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_వాడని(అంశం_ct1);
            });

    } ఆపు;
    అప్రమేయ:
        GGML_ASSERT(అబద్ధం);
    }
}
