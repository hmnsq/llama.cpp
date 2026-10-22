#ଅନ୍ତର୍ଭାଵ "opt-step.hpp"

#define SYCL_OPT_STEP_BLOCK_SIZE 256

template <typename T>
ସ୍ଥିର ଶୂନ୍ୟ opt_step_adamw_f32_kernel(
    T * __restrict__ x,
    ଅଚଳ T * __restrict__ g,
    T * __restrict__ g_m,
    T * __restrict__ g_v,
    ଅଚଳ T * __restrict__ pars,
    ଅଚଳ int64_t k,
    ଅଚଳ sycl::nd_item<1> & item) {

    ଅଚଳ int64_t i = (int64_t) item.get_global_id(0);
    ଯଦି (i >= k) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ ଦଶମିକ alpha  = pars[0];
    ଅଚଳ ଦଶମିକ beta1  = pars[1];
    ଅଚଳ ଦଶମିକ beta2  = pars[2];
    ଅଚଳ ଦଶମିକ eps    = pars[3];
    ଅଚଳ ଦଶମିକ wd     = pars[4];
    ଅଚଳ ଦଶମିକ beta1h = pars[5];
    ଅଚଳ ଦଶମିକ beta2h = pars[6];

    ଅଚଳ ଦଶମିକ gi  = g[i];
    ଅଚଳ ଦଶମିକ gmi = g_m[i] * beta1 +      gi * (1.0f - beta1);
    ଅଚଳ ଦଶମିକ gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    ଅଚଳ ଦଶମିକ mh = gmi * beta1h;
    ଅଚଳ ଦଶମିକ vh = sycl::sqrt(gvi * beta2h) + eps;

    x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
}

template <typename T>
ସ୍ଥିର ଶୂନ୍ୟ opt_step_sgd_f32_kernel(
    T * __restrict__ x,
    ଅଚଳ T * __restrict__ g,
    ଅଚଳ T * __restrict__ pars,
    ଅଚଳ int64_t k,
    ଅଚଳ sycl::nd_item<1> & item) {

    ଅଚଳ int64_t i = (int64_t) item.get_global_id(0);
    ଯଦି (i >= k) {
        ଫେରନ୍ତୁ;
    }

    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

ଶୂନ୍ୟ ggml_sycl_opt_step_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    ଅଚଳ ggml_tensor * src0         = dst->src[0];
    ଅଚଳ ggml_tensor * src0_grad    = dst->src[1];
    ଅଚଳ ggml_tensor * src0_grad_m  = dst->src[2];
    ଅଚଳ ggml_tensor * src0_grad_v  = dst->src[3];
    ଅଚଳ ggml_tensor * adamw_params = dst->src[4];

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

    ଦଶମିକ       * src0_d         = (ଦଶମିକ       *) src0->data;
    ଅଚଳ ଦଶମିକ * src0_grad_d    = (ଅଚଳ ଦଶମିକ *) src0_grad->data;
    ଦଶମିକ       * src0_grad_m_d  = (ଦଶମିକ       *) src0_grad_m->data;
    ଦଶମିକ       * src0_grad_v_d  = (ଦଶମିକ       *) src0_grad_v->data;
    ଅଚଳ ଦଶମିକ * adamw_params_d = (ଅଚଳ ଦଶମିକ *) adamw_params->data;

    ଅଚଳ int64_t ne = ggml_nelements(src0);
    ଅଚଳ int64_t ଅଙ୍କ_blocks = (ne + SYCL_OPT_STEP_BLOCK_SIZE - 1) / SYCL_OPT_STEP_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(ଅଙ୍କ_blocks * SYCL_OPT_STEP_BLOCK_SIZE, SYCL_OPT_STEP_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            opt_step_adamw_f32_kernel(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, item);
        });
}

ଶୂନ୍ୟ ggml_sycl_opt_step_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ଅଚଳ ggml_tensor * src0       = dst->src[0];
    ଅଚଳ ggml_tensor * src0_grad  = dst->src[1];
    ଅଚଳ ggml_tensor * sgd_params = dst->src[2];

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

    ଦଶମିକ       * src0_d      = (ଦଶମିକ       *) src0->data;
    ଅଚଳ ଦଶମିକ * src0_grad_d = (ଅଚଳ ଦଶମିକ *) src0_grad->data;
    ଅଚଳ ଦଶମିକ * sgd_params_d = (ଅଚଳ ଦଶମିକ *) sgd_params->data;

    ଅଚଳ int64_t ne = ggml_nelements(src0);
    ଅଚଳ int64_t ଅଙ୍କ_blocks = (ne + SYCL_OPT_STEP_BLOCK_SIZE - 1) / SYCL_OPT_STEP_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(ଅଙ୍କ_blocks * SYCL_OPT_STEP_BLOCK_SIZE, SYCL_OPT_STEP_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            opt_step_sgd_f32_kernel(src0_d, src0_grad_d, sgd_params_d, ne, item);
        });
}
