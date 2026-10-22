#समावेश "opt-step.hpp"

#घोषणा SYCL_OPT_STEP_BLOCK_आकार 256

ढाँचा <typename T>
अटल व्योम opt_step_adamw_f32_kernel(
    T * __restrict__ x,
    स्थिर T * __restrict__ g,
    T * __restrict__ g_m,
    T * __restrict__ g_v,
    स्थिर T * __restrict__ pars,
    स्थिर int64_t k,
    स्थिर sycl::nd_item<1> & item) {

    स्थिर int64_t i = (int64_t) item.get_global_id(0);
    अगर (i >= k) {
        वापस;
    }

    स्थिर भग्न alpha  = pars[0];
    स्थिर भग्न beta1  = pars[1];
    स्थिर भग्न beta2  = pars[2];
    स्थिर भग्न eps    = pars[3];
    स्थिर भग्न wd     = pars[4];
    स्थिर भग्न beta1h = pars[5];
    स्थिर भग्न beta2h = pars[6];

    स्थिर भग्न gi  = g[i];
    स्थिर भग्न gmi = g_m[i] * beta1 +      gi * (1.0f - beta1);
    स्थिर भग्न gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    स्थिर भग्न mh = gmi * beta1h;
    स्थिर भग्न vh = sycl::वर्ग_मूल(gvi * beta2h) + eps;

    x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
}

ढाँचा <typename T>
अटल व्योम opt_step_sgd_f32_kernel(
    T * __restrict__ x,
    स्थिर T * __restrict__ g,
    स्थिर T * __restrict__ pars,
    स्थिर int64_t k,
    स्थिर sycl::nd_item<1> & item) {

    स्थिर int64_t i = (int64_t) item.get_global_id(0);
    अगर (i >= k) {
        वापस;
    }

    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

व्योम ggml_sycl_opt_step_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    स्थिर ggml_tensor * src0         = dst->src[0];
    स्थिर ggml_tensor * src0_grad    = dst->src[1];
    स्थिर ggml_tensor * src0_grad_m  = dst->src[2];
    स्थिर ggml_tensor * src0_grad_v  = dst->src[3];
    स्थिर ggml_tensor * adamw_params = dst->src[4];

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

    भग्न       * src0_d         = (भग्न       *) src0->तथ्य;
    स्थिर भग्न * src0_grad_d    = (स्थिर भग्न *) src0_grad->तथ्य;
    भग्न       * src0_grad_m_d  = (भग्न       *) src0_grad_m->तथ्य;
    भग्न       * src0_grad_v_d  = (भग्न       *) src0_grad_v->तथ्य;
    स्थिर भग्न * adamw_params_d = (स्थिर भग्न *) adamw_params->तथ्य;

    स्थिर int64_t ne = ggml_nelements(src0);
    स्थिर int64_t अंक_blocks = (ne + SYCL_OPT_STEP_BLOCK_आकार - 1) / SYCL_OPT_STEP_BLOCK_आकार;

    stream->parallel_for(
        sycl::nd_range<1>(अंक_blocks * SYCL_OPT_STEP_BLOCK_आकार, SYCL_OPT_STEP_BLOCK_आकार),
        [=](sycl::nd_item<1> item) {
            opt_step_adamw_f32_kernel(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, item);
        });
}

व्योम ggml_sycl_opt_step_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    स्थिर ggml_tensor * src0       = dst->src[0];
    स्थिर ggml_tensor * src0_grad  = dst->src[1];
    स्थिर ggml_tensor * sgd_params = dst->src[2];

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

    भग्न       * src0_d      = (भग्न       *) src0->तथ्य;
    स्थिर भग्न * src0_grad_d = (स्थिर भग्न *) src0_grad->तथ्य;
    स्थिर भग्न * sgd_params_d = (स्थिर भग्न *) sgd_params->तथ्य;

    स्थिर int64_t ne = ggml_nelements(src0);
    स्थिर int64_t अंक_blocks = (ne + SYCL_OPT_STEP_BLOCK_आकार - 1) / SYCL_OPT_STEP_BLOCK_आकार;

    stream->parallel_for(
        sycl::nd_range<1>(अंक_blocks * SYCL_OPT_STEP_BLOCK_आकार, SYCL_OPT_STEP_BLOCK_आकार),
        [=](sycl::nd_item<1> item) {
            opt_step_sgd_f32_kernel(src0_d, src0_grad_d, sgd_params_d, ne, item);
        });
}
