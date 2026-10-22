#ضمن "opt-step.hpp"

#تعریف_کن SYCL_OPT_گام_بلوک_اندازه 256

قالب <نام_نوع T>
ساكن فراغ opt_گام_adamw_f32_kernel(
    T * __restrict__ x,
    ثابت T * __restrict__ g,
    T * __restrict__ g_m,
    T * __restrict__ g_v,
    ثابت T * __restrict__ pars,
    ثابت int64_t k,
    ثابت sycl::nd_قلم<1> & قلم) {

    ثابت int64_t i = (int64_t) قلم.get_global_id(0);
    إذا (i >= k) {
        أرجع;
    }

    ثابت عائم alpha  = pars[0];
    ثابت عائم beta1  = pars[1];
    ثابت عائم beta2  = pars[2];
    ثابت عائم eps    = pars[3];
    ثابت عائم wd     = pars[4];
    ثابت عائم beta1h = pars[5];
    ثابت عائم beta2h = pars[6];

    ثابت عائم gi  = g[i];
    ثابت عائم gmi = g_m[i] * beta1 +      gi * (1.0f - beta1);
    ثابت عائم gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    ثابت عائم mh = gmi * beta1h;
    ثابت عائم vh = sycl::جذر(gvi * beta2h) + eps;

    x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
}

قالب <نام_نوع T>
ساكن فراغ opt_گام_sgd_f32_kernel(
    T * __restrict__ x,
    ثابت T * __restrict__ g,
    ثابت T * __restrict__ pars,
    ثابت int64_t k,
    ثابت sycl::nd_قلم<1> & قلم) {

    ثابت int64_t i = (int64_t) قلم.get_global_id(0);
    إذا (i >= k) {
        أرجع;
    }

    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

فراغ ggml_sycl_opt_گام_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    ثابت ggml_tensor * src0         = dst->src[0];
    ثابت ggml_tensor * src0_grad    = dst->src[1];
    ثابت ggml_tensor * src0_grad_m  = dst->src[2];
    ثابت ggml_tensor * src0_grad_v  = dst->src[3];
    ثابت ggml_tensor * adamw_params = dst->src[4];

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

    dpct::صف_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));

    عائم       * src0_d         = (عائم       *) src0->داده;
    ثابت عائم * src0_grad_d    = (ثابت عائم *) src0_grad->داده;
    عائم       * src0_grad_m_d  = (عائم       *) src0_grad_m->داده;
    عائم       * src0_grad_v_d  = (عائم       *) src0_grad_v->داده;
    ثابت عائم * adamw_params_d = (ثابت عائم *) adamw_params->داده;

    ثابت int64_t ne = ggml_nelements(src0);
    ثابت int64_t رقم_بلوکها = (ne + SYCL_OPT_گام_بلوک_اندازه - 1) / SYCL_OPT_گام_بلوک_اندازه;

    stream->parallel_for(
        sycl::nd_range<1>(رقم_بلوکها * SYCL_OPT_گام_بلوک_اندازه, SYCL_OPT_گام_بلوک_اندازه),
        [=](sycl::nd_قلم<1> قلم) {
            opt_گام_adamw_f32_kernel(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, قلم);
        });
}

فراغ ggml_sycl_opt_گام_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ثابت ggml_tensor * src0       = dst->src[0];
    ثابت ggml_tensor * src0_grad  = dst->src[1];
    ثابت ggml_tensor * sgd_params = dst->src[2];

    GGML_ASSERT(src0->type      == GGML_TYPE_F32);
    GGML_ASSERT(src0_grad->type == GGML_TYPE_F32);
    GGML_ASSERT(sgd_params->type == GGML_TYPE_F32);
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src0_grad));
    GGML_ASSERT(ggml_is_contiguous(sgd_params));
    GGML_ASSERT(ggml_are_same_shape(src0, src0_grad));
    GGML_ASSERT(ggml_nelements(sgd_params) == 2);

    dpct::صف_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));

    عائم       * src0_d      = (عائم       *) src0->داده;
    ثابت عائم * src0_grad_d = (ثابت عائم *) src0_grad->داده;
    ثابت عائم * sgd_params_d = (ثابت عائم *) sgd_params->داده;

    ثابت int64_t ne = ggml_nelements(src0);
    ثابت int64_t رقم_بلوکها = (ne + SYCL_OPT_گام_بلوک_اندازه - 1) / SYCL_OPT_گام_بلوک_اندازه;

    stream->parallel_for(
        sycl::nd_range<1>(رقم_بلوکها * SYCL_OPT_گام_بلوک_اندازه, SYCL_OPT_گام_بلوک_اندازه),
        [=](sycl::nd_قلم<1> قلم) {
            opt_گام_sgd_f32_kernel(src0_d, src0_grad_d, sgd_params_d, ne, قلم);
        });
}
