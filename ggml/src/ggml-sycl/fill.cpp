#అన్తర్భావ "fill.hpp"
#అన్తర్భావ "common.hpp"

#నిర్వచించు SYCL_FILL_ఖండం_కొలత 256

మూస <రకం_పేరు T>
స్థిర శూన్య fill_kernel(T * dst, స్థిరరాశి int64_t k, స్థిరరాశి T value,
                        స్థిరరాశి sycl::nd_అంశం<1> & అంశం) {
    స్థిరరాశి int64_t i = (int64_t)అంశం.get_global_id(0);
    ఒకవేళ (i >= k) {
        తిరిగి;
    }
    dst[i] = value;
}

అంతఃవరుస శూన్య ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::వరుస_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    దశాంశం value;
    జ్ఞాపకం_నకలు(&value, dst->op_params, పరిమాణం(దశాంశం));

    స్థిరరాశి int64_t k = ggml_nelements(dst);
    స్థిరరాశి int64_t అంకె_ఖండాలు = (k + SYCL_FILL_ఖండం_కొలత - 1) / SYCL_FILL_ఖండం_కొలత;
    శూన్య * dst_d = dst->దత్తాంశం;

    ఎంపిక (dst->type) {
        స్థితి GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(అంకె_ఖండాలు * SYCL_FILL_ఖండం_కొలత, SYCL_FILL_ఖండం_కొలత),
                [=](sycl::nd_అంశం<1> అంశం) {
                    fill_kernel(static_cast<దశాంశం *>(dst_d), k, value, అంశం);
                });
            ఆపు;
        స్థితి GGML_TYPE_F16:
            {
                sycl::half h_విలువ = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(అంకె_ఖండాలు * SYCL_FILL_ఖండం_కొలత, SYCL_FILL_ఖండం_కొలత),
                    [=](sycl::nd_అంశం<1> అంశం) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_విలువ, అంశం);
                    });
            }
            ఆపు;
        అప్రమేయ:
            GGML_ABORT("unsupported type");
    }
}

శూన్య ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
