#شامل "opt-step.hpp"

#تعریف_کرو SYCL_OPT_قدم_بلاک_سائز 256

سانچہ <قسم_نام T>
ساکن خالی opt_قدم_adamw_f32_kernel(
    T * __restrict__ x,
    ثابت T * __restrict__ g,
    T * __restrict__ g_m,
    T * __restrict__ g_v,
    ثابت T * __restrict__ pars,
    ثابت int64_t k,
    ثابت sycl::nd_شے<1> & شے) {

    ثابت int64_t i = (int64_t) شے.get_global_id(0);
    اگر (i >= k) {
        واپس;
    }

    ثابت اعشاری alpha  = pars[0];
    ثابت اعشاری beta1  = pars[1];
    ثابت اعشاری beta2  = pars[2];
    ثابت اعشاری eps    = pars[3];
    ثابت اعشاری wd     = pars[4];
    ثابت اعشاری beta1h = pars[5];
    ثابت اعشاری beta2h = pars[6];

    ثابت اعشاری gi  = g[i];
    ثابت اعشاری gmi = g_m[i] * beta1 +      gi * (1.0f - beta1);
    ثابت اعشاری gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    ثابت اعشاری mh = gmi * beta1h;
    ثابت اعشاری vh = sycl::جذر(gvi * beta2h) + eps;

    x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
}

سانچہ <قسم_نام T>
ساکن خالی opt_قدم_sgd_f32_kernel(
    T * __restrict__ x,
    ثابت T * __restrict__ g,
    ثابت T * __restrict__ pars,
    ثابت int64_t k,
    ثابت sycl::nd_شے<1> & شے) {

    ثابت int64_t i = (int64_t) شے.get_global_id(0);
    اگر (i >= k) {
        واپس;
    }

    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

خالی ggml_sycl_opt_قدم_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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

    dpct::قطار_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    اعشاری       * src0_d         = (اعشاری       *) src0->ڈیٹا;
    ثابت اعشاری * src0_grad_d    = (ثابت اعشاری *) src0_grad->ڈیٹا;
    اعشاری       * src0_grad_m_d  = (اعشاری       *) src0_grad_m->ڈیٹا;
    اعشاری       * src0_grad_v_d  = (اعشاری       *) src0_grad_v->ڈیٹا;
    ثابت اعشاری * adamw_params_d = (ثابت اعشاری *) adamw_params->ڈیٹا;

    ثابت int64_t ne = ggml_nelements(src0);
    ثابت int64_t عدد_بلاکیں = (ne + SYCL_OPT_قدم_بلاک_سائز - 1) / SYCL_OPT_قدم_بلاک_سائز;

    stream->parallel_for(
        sycl::nd_range<1>(عدد_بلاکیں * SYCL_OPT_قدم_بلاک_سائز, SYCL_OPT_قدم_بلاک_سائز),
        [=](sycl::nd_شے<1> شے) {
            opt_قدم_adamw_f32_kernel(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, شے);
        });
}

خالی ggml_sycl_opt_قدم_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
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

    dpct::قطار_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    اعشاری       * src0_d      = (اعشاری       *) src0->ڈیٹا;
    ثابت اعشاری * src0_grad_d = (ثابت اعشاری *) src0_grad->ڈیٹا;
    ثابت اعشاری * sgd_params_d = (ثابت اعشاری *) sgd_params->ڈیٹا;

    ثابت int64_t ne = ggml_nelements(src0);
    ثابت int64_t عدد_بلاکیں = (ne + SYCL_OPT_قدم_بلاک_سائز - 1) / SYCL_OPT_قدم_بلاک_سائز;

    stream->parallel_for(
        sycl::nd_range<1>(عدد_بلاکیں * SYCL_OPT_قدم_بلاک_سائز, SYCL_OPT_قدم_بلاک_سائز),
        [=](sycl::nd_شے<1> شے) {
            opt_قدم_sgd_f32_kernel(src0_d, src0_grad_d, sgd_params_d, ne, شے);
        });
}
