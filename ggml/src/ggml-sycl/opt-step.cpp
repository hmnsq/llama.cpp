#అన్తర్భావ "opt-step.hpp"

#నిర్వచించు SYCL_OPT_అడుగు_ఖండం_కొలత 256

మూస <రకం_పేరు T>
స్థిర శూన్య opt_అడుగు_adamw_f32_kernel(
    T * __restrict__ x,
    స్థిరరాశి T * __restrict__ g,
    T * __restrict__ g_m,
    T * __restrict__ g_v,
    స్థిరరాశి T * __restrict__ pars,
    స్థిరరాశి int64_t k,
    స్థిరరాశి sycl::nd_అంశం<1> & అంశం) {

    స్థిరరాశి int64_t i = (int64_t) అంశం.get_global_id(0);
    ఒకవేళ (i >= k) {
        తిరిగి;
    }

    స్థిరరాశి దశాంశం alpha  = pars[0];
    స్థిరరాశి దశాంశం beta1  = pars[1];
    స్థిరరాశి దశాంశం beta2  = pars[2];
    స్థిరరాశి దశాంశం eps    = pars[3];
    స్థిరరాశి దశాంశం wd     = pars[4];
    స్థిరరాశి దశాంశం beta1h = pars[5];
    స్థిరరాశి దశాంశం beta2h = pars[6];

    స్థిరరాశి దశాంశం gi  = g[i];
    స్థిరరాశి దశాంశం gmi = g_m[i] * beta1 +      gi * (1.0f - beta1);
    స్థిరరాశి దశాంశం gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    స్థిరరాశి దశాంశం mh = gmi * beta1h;
    స్థిరరాశి దశాంశం vh = sycl::వర్గమూలం(gvi * beta2h) + eps;

    x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
}

మూస <రకం_పేరు T>
స్థిర శూన్య opt_అడుగు_sgd_f32_kernel(
    T * __restrict__ x,
    స్థిరరాశి T * __restrict__ g,
    స్థిరరాశి T * __restrict__ pars,
    స్థిరరాశి int64_t k,
    స్థిరరాశి sycl::nd_అంశం<1> & అంశం) {

    స్థిరరాశి int64_t i = (int64_t) అంశం.get_global_id(0);
    ఒకవేళ (i >= k) {
        తిరిగి;
    }

    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

శూన్య ggml_sycl_opt_అడుగు_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    స్థిరరాశి ggml_tensor * src0         = dst->src[0];
    స్థిరరాశి ggml_tensor * src0_grad    = dst->src[1];
    స్థిరరాశి ggml_tensor * src0_grad_m  = dst->src[2];
    స్థిరరాశి ggml_tensor * src0_grad_v  = dst->src[3];
    స్థిరరాశి ggml_tensor * adamw_params = dst->src[4];

    GGML_ASSERT(src0->type         == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad->type    == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad_m->type  == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad_v->type  == GGML_TYPE_F32);
    GGML_ASSERT(adamw_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src0_grad));
    GGML_ASSERT(ggml_is_contiguous(src0_grad_m));
    GGML_ASSERT(ggml_is_contiguous(src0_grad_v));
    GGML_ASSERT(ggml_is_contiguous(adamw_params));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_m));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad_v));
    GGML_ASSERT(ggml_nelements(adamw_params) == 7);

    dpct::వరుస_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    దశాంశం       * src0_d         = (దశాంశం       *) src0->దత్తాంశం;
    స్థిరరాశి దశాంశం * src0_grad_d    = (స్థిరరాశి దశాంశం *) src0_grad->దత్తాంశం;
    దశాంశం       * src0_grad_m_d  = (దశాంశం       *) src0_grad_m->దత్తాంశం;
    దశాంశం       * src0_grad_v_d  = (దశాంశం       *) src0_grad_v->దత్తాంశం;
    స్థిరరాశి దశాంశం * adamw_params_d = (స్థిరరాశి దశాంశం *) adamw_params->దత్తాంశం;

    స్థిరరాశి int64_t ne = ggml_nelements(src0);
    స్థిరరాశి int64_t అంకె_ఖండాలు = (ne + SYCL_OPT_అడుగు_ఖండం_కొలత - 1) / SYCL_OPT_అడుగు_ఖండం_కొలత;

    stream->parallel_for(
        sycl::nd_range<1>(అంకె_ఖండాలు * SYCL_OPT_అడుగు_ఖండం_కొలత, SYCL_OPT_అడుగు_ఖండం_కొలత),
        [=](sycl::nd_అంశం<1> అంశం) {
            opt_అడుగు_adamw_f32_kernel(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, అంశం);
        });
}

శూన్య ggml_sycl_opt_అడుగు_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    స్థిరరాశి ggml_tensor * src0       = dst->src[0];
    స్థిరరాశి ggml_tensor * src0_grad  = dst->src[1];
    స్థిరరాశి ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(src0->type      == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad->type == GGML_TYPE_F32);
    GGML_ASSERT(sgd_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src0_grad));
    GGML_ASSERT(ggml_is_contiguous(sgd_params));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    dpct::వరుస_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    దశాంశం       * src0_d      = (దశాంశం       *) src0->దత్తాంశం;
    స్థిరరాశి దశాంశం * src0_grad_d = (స్థిరరాశి దశాంశం *) src0_grad->దత్తాంశం;
    స్థిరరాశి దశాంశం * sgd_params_d = (స్థిరరాశి దశాంశం *) sgd_params->దత్తాంశం;

    స్థిరరాశి int64_t ne = ggml_nelements(src0);
    స్థిరరాశి int64_t అంకె_ఖండాలు = (ne + SYCL_OPT_అడుగు_ఖండం_కొలత - 1) / SYCL_OPT_అడుగు_ఖండం_కొలత;

    stream->parallel_for(
        sycl::nd_range<1>(అంకె_ఖండాలు * SYCL_OPT_అడుగు_ఖండం_కొలత, SYCL_OPT_అడుగు_ఖండం_కొలత),
        [=](sycl::nd_అంశం<1> అంశం) {
            opt_అడుగు_sgd_f32_kernel(src0_d, src0_grad_d, sgd_params_d, ne, అంశం);
        });
}
