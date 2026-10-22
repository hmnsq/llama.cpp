#अन्तर्भाव "opt-step.hpp"

#define SYCL_OPT_STEP_BLOCK_SIZE 256

template <typename T>
स्थिर शून्य opt_step_adamw_f32_kernel(
    T * __restrict__ x,
    ध्रुव T * __restrict__ g,
    T * __restrict__ g_m,
    T * __restrict__ g_v,
    ध्रुव T * __restrict__ pars,
    ध्रुव int64_t k,
    ध्रुव sycl::nd_item<1> & item) {

    ध्रुव int64_t i = (int64_t) item.get_global_id(0);
    यदि (i >= k) {
        प्रत्यागम;
    }

    ध्रुव भिन्न alpha  = pars[0];
    ध्रुव भिन्न beta1  = pars[1];
    ध्रुव भिन्न beta2  = pars[2];
    ध्रुव भिन्न eps    = pars[3];
    ध्रुव भिन्न wd     = pars[4];
    ध्रुव भिन्न beta1h = pars[5];
    ध्रुव भिन्न beta2h = pars[6];

    ध्रुव भिन्न gi  = g[i];
    ध्रुव भिन्न gmi = g_m[i] * beta1 +      gi * (1.0f - beta1);
    ध्रुव भिन्न gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    ध्रुव भिन्न mh = gmi * beta1h;
    ध्रुव भिन्न vh = sycl::sqrt(gvi * beta2h) + eps;

    x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
}

template <typename T>
स्थिर शून्य opt_step_sgd_f32_kernel(
    T * __restrict__ x,
    ध्रुव T * __restrict__ g,
    ध्रुव T * __restrict__ pars,
    ध्रुव int64_t k,
    ध्रुव sycl::nd_item<1> & item) {

    ध्रुव int64_t i = (int64_t) item.get_global_id(0);
    यदि (i >= k) {
        प्रत्यागम;
    }

    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

शून्य ggml_sycl_opt_step_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    ध्रुव ggml_tensor * src0         = dst->src[0];
    ध्रुव ggml_tensor * src0_grad    = dst->src[1];
    ध्रुव ggml_tensor * src0_grad_m  = dst->src[2];
    ध्रुव ggml_tensor * src0_grad_v  = dst->src[3];
    ध्रुव ggml_tensor * adamw_params = dst->src[4];

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

    भिन्न       * src0_d         = (भिन्न       *) src0->data;
    ध्रुव भिन्न * src0_grad_d    = (ध्रुव भिन्न *) src0_grad->data;
    भिन्न       * src0_grad_m_d  = (भिन्न       *) src0_grad_m->data;
    भिन्न       * src0_grad_v_d  = (भिन्न       *) src0_grad_v->data;
    ध्रुव भिन्न * adamw_params_d = (ध्रुव भिन्न *) adamw_params->data;

    ध्रुव int64_t ne = ggml_nelements(src0);
    ध्रुव int64_t अंक_blocks = (ne + SYCL_OPT_STEP_BLOCK_SIZE - 1) / SYCL_OPT_STEP_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(अंक_blocks * SYCL_OPT_STEP_BLOCK_SIZE, SYCL_OPT_STEP_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            opt_step_adamw_f32_kernel(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, item);
        });
}

शून्य ggml_sycl_opt_step_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ध्रुव ggml_tensor * src0       = dst->src[0];
    ध्रुव ggml_tensor * src0_grad  = dst->src[1];
    ध्रुव ggml_tensor * sgd_params = dst->src[2];

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

    भिन्न       * src0_d      = (भिन्न       *) src0->data;
    ध्रुव भिन्न * src0_grad_d = (ध्रुव भिन्न *) src0_grad->data;
    ध्रुव भिन्न * sgd_params_d = (ध्रुव भिन्न *) sgd_params->data;

    ध्रुव int64_t ne = ggml_nelements(src0);
    ध्रुव int64_t अंक_blocks = (ne + SYCL_OPT_STEP_BLOCK_SIZE - 1) / SYCL_OPT_STEP_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(अंक_blocks * SYCL_OPT_STEP_BLOCK_SIZE, SYCL_OPT_STEP_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            opt_step_sgd_f32_kernel(src0_d, src0_grad_d, sgd_params_d, ne, item);
        });
}
