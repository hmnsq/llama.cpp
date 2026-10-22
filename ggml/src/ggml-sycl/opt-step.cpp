#ಅನ್ತರ್ಭಾವ "opt-step.hpp"

#define SYCL_OPT_STEP_BLOCK_SIZE 256

template <typename T>
ಸ್ಥಿರ ಶೂನ್ಯ opt_step_adamw_f32_kernel(
    T * __restrict__ x,
    ಧ್ರುವ T * __restrict__ g,
    T * __restrict__ g_m,
    T * __restrict__ g_v,
    ಧ್ರುವ T * __restrict__ pars,
    ಧ್ರುವ int64_t k,
    ಧ್ರುವ sycl::nd_item<1> & item) {

    ಧ್ರುವ int64_t i = (int64_t) item.get_global_id(0);
    ಯದಿ (i >= k) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ ಭಿನ್ನ alpha  = pars[0];
    ಧ್ರುವ ಭಿನ್ನ beta1  = pars[1];
    ಧ್ರುವ ಭಿನ್ನ beta2  = pars[2];
    ಧ್ರುವ ಭಿನ್ನ eps    = pars[3];
    ಧ್ರುವ ಭಿನ್ನ wd     = pars[4];
    ಧ್ರುವ ಭಿನ್ನ beta1h = pars[5];
    ಧ್ರುವ ಭಿನ್ನ beta2h = pars[6];

    ಧ್ರುವ ಭಿನ್ನ gi  = g[i];
    ಧ್ರುವ ಭಿನ್ನ gmi = g_m[i] * beta1 +      gi * (1.0f - beta1);
    ಧ್ರುವ ಭಿನ್ನ gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    ಧ್ರುವ ಭಿನ್ನ mh = gmi * beta1h;
    ಧ್ರುವ ಭಿನ್ನ vh = sycl::sqrt(gvi * beta2h) + eps;

    x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
}

template <typename T>
ಸ್ಥಿರ ಶೂನ್ಯ opt_step_sgd_f32_kernel(
    T * __restrict__ x,
    ಧ್ರುವ T * __restrict__ g,
    ಧ್ರುವ T * __restrict__ pars,
    ಧ್ರುವ int64_t k,
    ಧ್ರುವ sycl::nd_item<1> & item) {

    ಧ್ರುವ int64_t i = (int64_t) item.get_global_id(0);
    ಯದಿ (i >= k) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

ಶೂನ್ಯ ggml_sycl_opt_step_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    ಧ್ರುವ ggml_tensor * src0         = dst->src[0];
    ಧ್ರುವ ggml_tensor * src0_grad    = dst->src[1];
    ಧ್ರುವ ggml_tensor * src0_grad_m  = dst->src[2];
    ಧ್ರುವ ggml_tensor * src0_grad_v  = dst->src[3];
    ಧ್ರುವ ggml_tensor * adamw_params = dst->src[4];

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

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ಭಿನ್ನ       * src0_d         = (ಭಿನ್ನ       *) src0->data;
    ಧ್ರುವ ಭಿನ್ನ * src0_grad_d    = (ಧ್ರುವ ಭಿನ್ನ *) src0_grad->data;
    ಭಿನ್ನ       * src0_grad_m_d  = (ಭಿನ್ನ       *) src0_grad_m->data;
    ಭಿನ್ನ       * src0_grad_v_d  = (ಭಿನ್ನ       *) src0_grad_v->data;
    ಧ್ರುವ ಭಿನ್ನ * adamw_params_d = (ಧ್ರುವ ಭಿನ್ನ *) adamw_params->data;

    ಧ್ರುವ int64_t ne = ggml_nelements(src0);
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (ne + SYCL_OPT_STEP_BLOCK_SIZE - 1) / SYCL_OPT_STEP_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(ಅಂಕೆ_blocks * SYCL_OPT_STEP_BLOCK_SIZE, SYCL_OPT_STEP_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            opt_step_adamw_f32_kernel(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, item);
        });
}

ಶೂನ್ಯ ggml_sycl_opt_step_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ಧ್ರುವ ggml_tensor * src0       = dst->src[0];
    ಧ್ರುವ ggml_tensor * src0_grad  = dst->src[1];
    ಧ್ರುವ ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(src0->type      == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad->type == GGML_TYPE_F32);
    GGML_ASSERT(sgd_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src0_grad));
    GGML_ASSERT(ggml_is_contiguous(sgd_params));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ಭಿನ್ನ       * src0_d      = (ಭಿನ್ನ       *) src0->data;
    ಧ್ರುವ ಭಿನ್ನ * src0_grad_d = (ಧ್ರುವ ಭಿನ್ನ *) src0_grad->data;
    ಧ್ರುವ ಭಿನ್ನ * sgd_params_d = (ಧ್ರುವ ಭಿನ್ನ *) sgd_params->data;

    ಧ್ರುವ int64_t ne = ggml_nelements(src0);
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (ne + SYCL_OPT_STEP_BLOCK_SIZE - 1) / SYCL_OPT_STEP_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(ಅಂಕೆ_blocks * SYCL_OPT_STEP_BLOCK_SIZE, SYCL_OPT_STEP_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            opt_step_sgd_f32_kernel(src0_d, src0_grad_d, sgd_params_d, ne, item);
        });
}
