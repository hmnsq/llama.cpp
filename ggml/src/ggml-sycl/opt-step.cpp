#અન્તર્ભાવ "opt-step.hpp"

#define SYCL_OPT_STEP_BLOCK_SIZE 256

template <typename T>
સ્થિર શૂન્ય opt_step_adamw_f32_kernel(
    T * __restrict__ x,
    અચલ T * __restrict__ g,
    T * __restrict__ g_m,
    T * __restrict__ g_v,
    અચલ T * __restrict__ pars,
    અચલ int64_t k,
    અચલ sycl::nd_item<1> & item) {

    અચલ int64_t i = (int64_t) item.get_global_id(0);
    જો (i >= k) {
        પાછા;
    }

    અચલ દશાંશ alpha  = pars[0];
    અચલ દશાંશ beta1  = pars[1];
    અચલ દશાંશ beta2  = pars[2];
    અચલ દશાંશ eps    = pars[3];
    અચલ દશાંશ wd     = pars[4];
    અચલ દશાંશ beta1h = pars[5];
    અચલ દશાંશ beta2h = pars[6];

    અચલ દશાંશ gi  = g[i];
    અચલ દશાંશ gmi = g_m[i] * beta1 +      gi * (1.0f - beta1);
    અચલ દશાંશ gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    અચલ દશાંશ mh = gmi * beta1h;
    અચલ દશાંશ vh = sycl::sqrt(gvi * beta2h) + eps;

    x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
}

template <typename T>
સ્થિર શૂન્ય opt_step_sgd_f32_kernel(
    T * __restrict__ x,
    અચલ T * __restrict__ g,
    અચલ T * __restrict__ pars,
    અચલ int64_t k,
    અચલ sycl::nd_item<1> & item) {

    અચલ int64_t i = (int64_t) item.get_global_id(0);
    જો (i >= k) {
        પાછા;
    }

    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

શૂન્ય ggml_sycl_opt_step_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    અચલ ggml_tensor * src0         = dst->src[0];
    અચલ ggml_tensor * src0_grad    = dst->src[1];
    અચલ ggml_tensor * src0_grad_m  = dst->src[2];
    અચલ ggml_tensor * src0_grad_v  = dst->src[3];
    અચલ ggml_tensor * adamw_params = dst->src[4];

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

    દશાંશ       * src0_d         = (દશાંશ       *) src0->data;
    અચલ દશાંશ * src0_grad_d    = (અચલ દશાંશ *) src0_grad->data;
    દશાંશ       * src0_grad_m_d  = (દશાંશ       *) src0_grad_m->data;
    દશાંશ       * src0_grad_v_d  = (દશાંશ       *) src0_grad_v->data;
    અચલ દશાંશ * adamw_params_d = (અચલ દશાંશ *) adamw_params->data;

    અચલ int64_t ne = ggml_nelements(src0);
    અચલ int64_t અંક_blocks = (ne + SYCL_OPT_STEP_BLOCK_SIZE - 1) / SYCL_OPT_STEP_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(અંક_blocks * SYCL_OPT_STEP_BLOCK_SIZE, SYCL_OPT_STEP_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            opt_step_adamw_f32_kernel(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, item);
        });
}

શૂન્ય ggml_sycl_opt_step_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    અચલ ggml_tensor * src0       = dst->src[0];
    અચલ ggml_tensor * src0_grad  = dst->src[1];
    અચલ ggml_tensor * sgd_params = dst->src[2];

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

    દશાંશ       * src0_d      = (દશાંશ       *) src0->data;
    અચલ દશાંશ * src0_grad_d = (અચલ દશાંશ *) src0_grad->data;
    અચલ દશાંશ * sgd_params_d = (અચલ દશાંશ *) sgd_params->data;

    અચલ int64_t ne = ggml_nelements(src0);
    અચલ int64_t અંક_blocks = (ne + SYCL_OPT_STEP_BLOCK_SIZE - 1) / SYCL_OPT_STEP_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(અંક_blocks * SYCL_OPT_STEP_BLOCK_SIZE, SYCL_OPT_STEP_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            opt_step_sgd_f32_kernel(src0_d, src0_grad_d, sgd_params_d, ne, item);
        });
}
