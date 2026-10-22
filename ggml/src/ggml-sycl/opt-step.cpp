#ಅನ್ತರ್ಭಾವ "opt-step.hpp"

#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_OPT_ಹೆಜ್ಜೆ_ಖಂಡ_ಗಾತ್ರ 256

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ opt_ಹೆಜ್ಜೆ_adamw_f32_kernel(
    T * __restrict__ x,
    ಸ್ಥಿರಾಂಕ T * __restrict__ g,
    T * __restrict__ g_m,
    T * __restrict__ g_v,
    ಸ್ಥಿರಾಂಕ T * __restrict__ pars,
    ಸ್ಥಿರಾಂಕ int64_t k,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ) {

    ಸ್ಥಿರಾಂಕ int64_t i = (int64_t) ಅಂಶ.get_global_id(0);
    ಆದರೆ (i >= k) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha  = pars[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta1  = pars[1];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta2  = pars[2];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ eps    = pars[3];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ wd     = pars[4];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta1h = pars[5];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ beta2h = pars[6];

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ gi  = g[i];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ gmi = g_m[i] * beta1 +      gi * (1.0f - beta1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ mh = gmi * beta1h;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ vh = sycl::ವರ್ಗಮೂಲ(gvi * beta2h) + eps;

    x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ opt_ಹೆಜ್ಜೆ_sgd_f32_kernel(
    T * __restrict__ x,
    ಸ್ಥಿರಾಂಕ T * __restrict__ g,
    ಸ್ಥಿರಾಂಕ T * __restrict__ pars,
    ಸ್ಥಿರಾಂಕ int64_t k,
    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ) {

    ಸ್ಥಿರಾಂಕ int64_t i = (int64_t) ಅಂಶ.get_global_id(0);
    ಆದರೆ (i >= k) {
        ಹಿಂತಿರುಗಿ;
    }

    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

ಶೂನ್ಯ ggml_sycl_opt_ಹೆಜ್ಜೆ_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0         = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0_grad    = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0_grad_m  = dst->src[2];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0_grad_v  = dst->src[3];
    ಸ್ಥಿರಾಂಕ ggml_tensor * adamw_params = dst->src[4];

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

    dpct::ಸರತಿ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ದಶಮಾಂಶ       * src0_d         = (ದಶಮಾಂಶ       *) src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_grad_d    = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0_grad->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ       * src0_grad_m_d  = (ದಶಮಾಂಶ       *) src0_grad_m->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ       * src0_grad_v_d  = (ದಶಮಾಂಶ       *) src0_grad_v->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * adamw_params_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) adamw_params->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ int64_t ne = ggml_nelements(src0);
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (ne + SYCL_OPT_ಹೆಜ್ಜೆ_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_OPT_ಹೆಜ್ಜೆ_ಖಂಡ_ಗಾತ್ರ;

    stream->parallel_for(
        sycl::nd_range<1>(ಅಂಕೆ_ಖಂಡಗಳು * SYCL_OPT_ಹೆಜ್ಜೆ_ಖಂಡ_ಗಾತ್ರ, SYCL_OPT_ಹೆಜ್ಜೆ_ಖಂಡ_ಗಾತ್ರ),
        [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) {
            opt_ಹೆಜ್ಜೆ_adamw_f32_kernel(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, ಅಂಶ);
        });
}

ಶೂನ್ಯ ggml_sycl_opt_ಹೆಜ್ಜೆ_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ಸ್ಥಿರಾಂಕ ggml_tensor * src0       = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0_grad  = dst->src[1];
    ಸ್ಥಿರಾಂಕ ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(src0->type      == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad->type == GGML_TYPE_F32);
    GGML_ASSERT(sgd_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src0_grad));
    GGML_ASSERT(ggml_is_contiguous(sgd_params));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    dpct::ಸರತಿ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ದಶಮಾಂಶ       * src0_d      = (ದಶಮಾಂಶ       *) src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_grad_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0_grad->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * sgd_params_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) sgd_params->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ int64_t ne = ggml_nelements(src0);
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (ne + SYCL_OPT_ಹೆಜ್ಜೆ_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_OPT_ಹೆಜ್ಜೆ_ಖಂಡ_ಗಾತ್ರ;

    stream->parallel_for(
        sycl::nd_range<1>(ಅಂಕೆ_ಖಂಡಗಳು * SYCL_OPT_ಹೆಜ್ಜೆ_ಖಂಡ_ಗಾತ್ರ, SYCL_OPT_ಹೆಜ್ಜೆ_ಖಂಡ_ಗಾತ್ರ),
        [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) {
            opt_ಹೆಜ್ಜೆ_sgd_f32_kernel(src0_d, src0_grad_d, sgd_params_d, ne, ಅಂಶ);
        });
}
