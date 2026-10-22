#অন্তর্ভাব "opt-step.hpp"

#define SYCL_OPT_STEP_BLOCK_আকার 256

template <typename T>
স্থির শূন্য opt_step_adamw_f32_kernel(
    T * __restrict__ x,
    ধ্রুবক T * __restrict__ g,
    T * __restrict__ g_m,
    T * __restrict__ g_v,
    ধ্রুবক T * __restrict__ pars,
    ধ্রুবক int64_t k,
    ধ্রুবক sycl::nd_item<1> & item) {

    ধ্রুবক int64_t i = (int64_t) item.get_global_id(0);
    যদি (i >= k) {
        ফেরত;
    }

    ধ্রুবক দশমিক alpha  = pars[0];
    ধ্রুবক দশমিক beta1  = pars[1];
    ধ্রুবক দশমিক beta2  = pars[2];
    ধ্রুবক দশমিক eps    = pars[3];
    ধ্রুবক দশমিক wd     = pars[4];
    ধ্রুবক দশমিক beta1h = pars[5];
    ধ্রুবক দশমিক beta2h = pars[6];

    ধ্রুবক দশমিক gi  = g[i];
    ধ্রুবক দশমিক gmi = g_m[i] * beta1 +      gi * (1.0f - beta1);
    ধ্রুবক দশমিক gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    ধ্রুবক দশমিক mh = gmi * beta1h;
    ধ্রুবক দশমিক vh = sycl::sqrt(gvi * beta2h) + eps;

    x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
}

template <typename T>
স্থির শূন্য opt_step_sgd_f32_kernel(
    T * __restrict__ x,
    ধ্রুবক T * __restrict__ g,
    ধ্রুবক T * __restrict__ pars,
    ধ্রুবক int64_t k,
    ধ্রুবক sycl::nd_item<1> & item) {

    ধ্রুবক int64_t i = (int64_t) item.get_global_id(0);
    যদি (i >= k) {
        ফেরত;
    }

    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

শূন্য ggml_sycl_opt_step_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    ধ্রুবক ggml_tensor * src0         = dst->src[0];
    ধ্রুবক ggml_tensor * src0_grad    = dst->src[1];
    ধ্রুবক ggml_tensor * src0_grad_m  = dst->src[2];
    ধ্রুবক ggml_tensor * src0_grad_v  = dst->src[3];
    ধ্রুবক ggml_tensor * adamw_params = dst->src[4];

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

    দশমিক       * src0_d         = (দশমিক       *) src0->তথ্য;
    ধ্রুবক দশমিক * src0_grad_d    = (ধ্রুবক দশমিক *) src0_grad->তথ্য;
    দশমিক       * src0_grad_m_d  = (দশমিক       *) src0_grad_m->তথ্য;
    দশমিক       * src0_grad_v_d  = (দশমিক       *) src0_grad_v->তথ্য;
    ধ্রুবক দশমিক * adamw_params_d = (ধ্রুবক দশমিক *) adamw_params->তথ্য;

    ধ্রুবক int64_t ne = ggml_nelements(src0);
    ধ্রুবক int64_t অঙ্ক_blocks = (ne + SYCL_OPT_STEP_BLOCK_আকার - 1) / SYCL_OPT_STEP_BLOCK_আকার;

    stream->parallel_for(
        sycl::nd_range<1>(অঙ্ক_blocks * SYCL_OPT_STEP_BLOCK_আকার, SYCL_OPT_STEP_BLOCK_আকার),
        [=](sycl::nd_item<1> item) {
            opt_step_adamw_f32_kernel(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, item);
        });
}

শূন্য ggml_sycl_opt_step_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    ধ্রুবক ggml_tensor * src0       = dst->src[0];
    ধ্রুবক ggml_tensor * src0_grad  = dst->src[1];
    ধ্রুবক ggml_tensor * sgd_params = dst->src[2];

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

    দশমিক       * src0_d      = (দশমিক       *) src0->তথ্য;
    ধ্রুবক দশমিক * src0_grad_d = (ধ্রুবক দশমিক *) src0_grad->তথ্য;
    ধ্রুবক দশমিক * sgd_params_d = (ধ্রুবক দশমিক *) sgd_params->তথ্য;

    ধ্রুবক int64_t ne = ggml_nelements(src0);
    ধ্রুবক int64_t অঙ্ক_blocks = (ne + SYCL_OPT_STEP_BLOCK_আকার - 1) / SYCL_OPT_STEP_BLOCK_আকার;

    stream->parallel_for(
        sycl::nd_range<1>(অঙ্ক_blocks * SYCL_OPT_STEP_BLOCK_আকার, SYCL_OPT_STEP_BLOCK_আকার),
        [=](sycl::nd_item<1> item) {
            opt_step_sgd_f32_kernel(src0_d, src0_grad_d, sgd_params_d, ne, item);
        });
}
