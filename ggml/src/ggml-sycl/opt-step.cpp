#अन्तर्भाव "opt-step.hpp"

#define SYCL_OPT_STEP_BLOCK_SIZE 256

ढाँचा <typename T>
स्थिर शून्य opt_step_adamw_f32_kernel(
    T * __restrict__ x,
    अचर T * __restrict__ g,
    T * __restrict__ g_m,
    T * __restrict__ g_v,
    अचर T * __restrict__ pars,
    अचर int64_t k,
    अचर sycl::nd_item<1> & item) {

    अचर int64_t i = (int64_t) item.get_global_id(0);
    यदि (i >= k) {
        फर्काऊ;
    }

    अचर भिन्न alpha  = pars[0];
    अचर भिन्न beta1  = pars[1];
    अचर भिन्न beta2  = pars[2];
    अचर भिन्न eps    = pars[3];
    अचर भिन्न wd     = pars[4];
    अचर भिन्न beta1h = pars[5];
    अचर भिन्न beta2h = pars[6];

    अचर भिन्न gi  = g[i];
    अचर भिन्न gmi = g_m[i] * beta1 +      gi * (1.0f - beta1);
    अचर भिन्न gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    अचर भिन्न mh = gmi * beta1h;
    अचर भिन्न vh = sycl::वर्ग_मूल(gvi * beta2h) + eps;

    x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
}

ढाँचा <typename T>
स्थिर शून्य opt_step_sgd_f32_kernel(
    T * __restrict__ x,
    अचर T * __restrict__ g,
    अचर T * __restrict__ pars,
    अचर int64_t k,
    अचर sycl::nd_item<1> & item) {

    अचर int64_t i = (int64_t) item.get_global_id(0);
    यदि (i >= k) {
        फर्काऊ;
    }

    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

शून्य ggml_sycl_opt_step_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    अचर ggml_tensor * src0         = dst->src[0];
    अचर ggml_tensor * src0_grad    = dst->src[1];
    अचर ggml_tensor * src0_grad_m  = dst->src[2];
    अचर ggml_tensor * src0_grad_v  = dst->src[3];
    अचर ggml_tensor * adamw_params = dst->src[4];

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
    अचर भिन्न * src0_grad_d    = (अचर भिन्न *) src0_grad->data;
    भिन्न       * src0_grad_m_d  = (भिन्न       *) src0_grad_m->data;
    भिन्न       * src0_grad_v_d  = (भिन्न       *) src0_grad_v->data;
    अचर भिन्न * adamw_params_d = (अचर भिन्न *) adamw_params->data;

    अचर int64_t ne = ggml_nelements(src0);
    अचर int64_t अङ्क_blocks = (ne + SYCL_OPT_STEP_BLOCK_SIZE - 1) / SYCL_OPT_STEP_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(अङ्क_blocks * SYCL_OPT_STEP_BLOCK_SIZE, SYCL_OPT_STEP_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            opt_step_adamw_f32_kernel(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, item);
        });
}

शून्य ggml_sycl_opt_step_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    अचर ggml_tensor * src0       = dst->src[0];
    अचर ggml_tensor * src0_grad  = dst->src[1];
    अचर ggml_tensor * sgd_params = dst->src[2];

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
    अचर भिन्न * src0_grad_d = (अचर भिन्न *) src0_grad->data;
    अचर भिन्न * sgd_params_d = (अचर भिन्न *) sgd_params->data;

    अचर int64_t ne = ggml_nelements(src0);
    अचर int64_t अङ्क_blocks = (ne + SYCL_OPT_STEP_BLOCK_SIZE - 1) / SYCL_OPT_STEP_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(अङ्क_blocks * SYCL_OPT_STEP_BLOCK_SIZE, SYCL_OPT_STEP_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            opt_step_sgd_f32_kernel(src0_d, src0_grad_d, sgd_params_d, ne, item);
        });
}
