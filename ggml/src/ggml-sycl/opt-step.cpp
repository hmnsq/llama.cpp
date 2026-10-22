#ਅਨ੍ਤਰ੍ਭਾਵ "opt-step.hpp"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_OPT_ਕਦਮ_ਖੰਡ_ਮਾਪ 256

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ opt_ਕਦਮ_adamw_f32_kernel(
    T * __restrict__ x,
    ਅਚਲ T * __restrict__ g,
    T * __restrict__ g_m,
    T * __restrict__ g_v,
    ਅਚਲ T * __restrict__ pars,
    ਅਚਲ int64_t k,
    ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ) {

    ਅਚਲ int64_t i = (int64_t) ਵਸਤੂ.get_global_id(0);
    ਜੇ (i >= k) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ alpha  = pars[0];
    ਅਚਲ ਦਸ਼ਮਲਵ beta1  = pars[1];
    ਅਚਲ ਦਸ਼ਮਲਵ beta2  = pars[2];
    ਅਚਲ ਦਸ਼ਮਲਵ eps    = pars[3];
    ਅਚਲ ਦਸ਼ਮਲਵ wd     = pars[4];
    ਅਚਲ ਦਸ਼ਮਲਵ beta1h = pars[5];
    ਅਚਲ ਦਸ਼ਮਲਵ beta2h = pars[6];

    ਅਚਲ ਦਸ਼ਮਲਵ gi  = g[i];
    ਅਚਲ ਦਸ਼ਮਲਵ gmi = g_m[i] * beta1 +      gi * (1.0f - beta1);
    ਅਚਲ ਦਸ਼ਮਲਵ gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    ਅਚਲ ਦਸ਼ਮਲਵ mh = gmi * beta1h;
    ਅਚਲ ਦਸ਼ਮਲਵ vh = sycl::ਵਰਗਮੂਲ(gvi * beta2h) + eps;

    x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ opt_ਕਦਮ_sgd_f32_kernel(
    T * __restrict__ x,
    ਅਚਲ T * __restrict__ g,
    ਅਚਲ T * __restrict__ pars,
    ਅਚਲ int64_t k,
    ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ) {

    ਅਚਲ int64_t i = (int64_t) ਵਸਤੂ.get_global_id(0);
    ਜੇ (i >= k) {
        ਵਾਪਸ;
    }

    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

ਖਾਲੀ ggml_sycl_opt_ਕਦਮ_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    ਅਚਲ ggml_tensor * src0         = dst->src[0];
    ਅਚਲ ggml_tensor * src0_grad    = dst->src[1];
    ਅਚਲ ggml_tensor * src0_grad_m  = dst->src[2];
    ਅਚਲ ggml_tensor * src0_grad_v  = dst->src[3];
    ਅਚਲ ggml_tensor * adamw_params = dst->src[4];

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

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਦਸ਼ਮਲਵ       * src0_d         = (ਦਸ਼ਮਲਵ       *) src0->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_grad_d    = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0_grad->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ       * src0_grad_m_d  = (ਦਸ਼ਮਲਵ       *) src0_grad_m->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ       * src0_grad_v_d  = (ਦਸ਼ਮਲਵ       *) src0_grad_v->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * adamw_params_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) adamw_params->ਅੰਕੜੇ;

    ਅਚਲ int64_t ne = ggml_nelements(src0);
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_OPT_ਕਦਮ_ਖੰਡ_ਮਾਪ - 1) / SYCL_OPT_ਕਦਮ_ਖੰਡ_ਮਾਪ;

    stream->parallel_for(
        sycl::nd_range<1>(ਅੰਕ_ਖੰਡਾਂ * SYCL_OPT_ਕਦਮ_ਖੰਡ_ਮਾਪ, SYCL_OPT_ਕਦਮ_ਖੰਡ_ਮਾਪ),
        [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) {
            opt_ਕਦਮ_adamw_f32_kernel(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, ਵਸਤੂ);
        });
}

ਖਾਲੀ ggml_sycl_opt_ਕਦਮ_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ਅਚਲ ggml_tensor * src0       = dst->src[0];
    ਅਚਲ ggml_tensor * src0_grad  = dst->src[1];
    ਅਚਲ ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(src0->type      == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad->type == GGML_TYPE_F32);
    GGML_ASSERT(sgd_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src0_grad));
    GGML_ASSERT(ggml_is_contiguous(sgd_params));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਦਸ਼ਮਲਵ       * src0_d      = (ਦਸ਼ਮਲਵ       *) src0->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_grad_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0_grad->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * sgd_params_d = (ਅਚਲ ਦਸ਼ਮਲਵ *) sgd_params->ਅੰਕੜੇ;

    ਅਚਲ int64_t ne = ggml_nelements(src0);
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (ne + SYCL_OPT_ਕਦਮ_ਖੰਡ_ਮਾਪ - 1) / SYCL_OPT_ਕਦਮ_ਖੰਡ_ਮਾਪ;

    stream->parallel_for(
        sycl::nd_range<1>(ਅੰਕ_ਖੰਡਾਂ * SYCL_OPT_ਕਦਮ_ਖੰਡ_ਮਾਪ, SYCL_OPT_ਕਦਮ_ਖੰਡ_ਮਾਪ),
        [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) {
            opt_ਕਦਮ_sgd_f32_kernel(src0_d, src0_grad_d, sgd_params_d, ne, ਵਸਤੂ);
        });
}
