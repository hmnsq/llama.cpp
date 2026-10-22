#అన్తర్భావ "diag.hpp"
#అన్తర్భావ "common.hpp"

#నిర్వచించు SYCL_DIAG_ఖండం_కొలత 256

మూస <రకం_పేరు T>
స్థిర శూన్య diag_kernel(T * __restrict__ dst, స్థిరరాశి T * __restrict__ src,
                        స్థిరరాశి int64_t ne0, స్థిరరాశి int64_t ne1,
                        స్థిరరాశి int64_t ne2, స్థిరరాశి int64_t ne3,
                        స్థిరరాశి int64_t మొత్తం_elements,
                        స్థిరరాశి sycl::nd_అంశం<1> & అంశం) {
    స్థిరరాశి int64_t i = అంశం.get_global_id(0);
    ఒకవేళ (i >= మొత్తం_elements) {
        తిరిగి;
    }

    స్థిరరాశి int64_t i0 = i % ne0;
    స్థిరరాశి int64_t i1 = (i / ne0) % ne1;
    స్థిరరాశి int64_t i2 = (i / (ne0 * ne1)) % ne2;
    స్థిరరాశి int64_t i3 = i / (ne0 * ne1 * ne2);

    స్థిరరాశి int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    ఒకవేళ (i0 == i1) {
        స్థిరరాశి int64_t batch_idx = i3 * ne2 + i2;
        dst[dst_idx] = src[batch_idx * ne0 + i0];
    } లేకపోతే {
        dst[dst_idx] = T(0);
    }

    (శూన్య)ne3;
}

అంతఃవరుస శూన్య ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[1] == 1);

    dpct::వరుస_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    స్థిరరాశి శూన్య * src0_d = src0->దత్తాంశం;
    శూన్య * dst_d = dst->దత్తాంశం;

    స్థిరరాశి int64_t ne0 = dst->ne[0];
    స్థిరరాశి int64_t ne1 = dst->ne[1];
    స్థిరరాశి int64_t ne2 = dst->ne[2];
    స్థిరరాశి int64_t ne3 = dst->ne[3];
    స్థిరరాశి int64_t n_elems = ggml_nelements(dst);
    స్థిరరాశి int64_t అంకె_ఖండాలు = (n_elems + SYCL_DIAG_ఖండం_కొలత - 1) / SYCL_DIAG_ఖండం_కొలత;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    stream->parallel_for(
        sycl::nd_range<1>(అంకె_ఖండాలు * SYCL_DIAG_ఖండం_కొలత, SYCL_DIAG_ఖండం_కొలత),
        [=](sycl::nd_అంశం<1> అంశం) {
            diag_kernel(static_cast<దశాంశం *>(dst_d),
                        static_cast<స్థిరరాశి దశాంశం *>(src0_d),
                        ne0, ne1, ne2, ne3, n_elems, అంశం);
        });
}

శూన్య ggml_sycl_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag(ctx, dst);
}
