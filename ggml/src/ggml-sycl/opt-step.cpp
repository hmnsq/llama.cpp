#அந்தர்பாவ "opt-step.hpp"

#define SYCL_OPT_STEP_BLOCK_SIZE 256

template <typename T>
நிலையான வெறுமை opt_step_adamw_f32_kernel(
    T * __restrict__ x,
    மாறிலி T * __restrict__ g,
    T * __restrict__ g_m,
    T * __restrict__ g_v,
    மாறிலி T * __restrict__ pars,
    மாறிலி int64_t k,
    மாறிலி sycl::nd_item<1> & item) {

    மாறிலி int64_t i = (int64_t) item.get_global_id(0);
    எனில் (i >= k) {
        திரும்பு;
    }

    மாறிலி தசம alpha  = pars[0];
    மாறிலி தசம beta1  = pars[1];
    மாறிலி தசம beta2  = pars[2];
    மாறிலி தசம eps    = pars[3];
    மாறிலி தசம wd     = pars[4];
    மாறிலி தசம beta1h = pars[5];
    மாறிலி தசம beta2h = pars[6];

    மாறிலி தசம gi  = g[i];
    மாறிலி தசம gmi = g_m[i] * beta1 +      gi * (1.0f - beta1);
    மாறிலி தசம gvi = g_v[i] * beta2 + gi * gi * (1.0f - beta2);

    g_m[i] = gmi;
    g_v[i] = gvi;

    மாறிலி தசம mh = gmi * beta1h;
    மாறிலி தசம vh = sycl::sqrt(gvi * beta2h) + eps;

    x[i] = x[i] * (1.0f - alpha * wd) - alpha * mh / vh;
}

template <typename T>
நிலையான வெறுமை opt_step_sgd_f32_kernel(
    T * __restrict__ x,
    மாறிலி T * __restrict__ g,
    மாறிலி T * __restrict__ pars,
    மாறிலி int64_t k,
    மாறிலி sycl::nd_item<1> & item) {

    மாறிலி int64_t i = (int64_t) item.get_global_id(0);
    எனில் (i >= k) {
        திரும்பு;
    }

    x[i] = x[i] * (1.0f - pars[0] * pars[1]) - pars[0] * g[i];
}

வெறுமை ggml_sycl_opt_step_adamw(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/5);

    மாறிலி ggml_tensor * src0         = dst->src[0];
    மாறிலி ggml_tensor * src0_grad    = dst->src[1];
    மாறிலி ggml_tensor * src0_grad_m  = dst->src[2];
    மாறிலி ggml_tensor * src0_grad_v  = dst->src[3];
    மாறிலி ggml_tensor * adamw_params = dst->src[4];

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

    தசம       * src0_d         = (தசம       *) src0->data;
    மாறிலி தசம * src0_grad_d    = (மாறிலி தசம *) src0_grad->data;
    தசம       * src0_grad_m_d  = (தசம       *) src0_grad_m->data;
    தசம       * src0_grad_v_d  = (தசம       *) src0_grad_v->data;
    மாறிலி தசம * adamw_params_d = (மாறிலி தசம *) adamw_params->data;

    மாறிலி int64_t ne = ggml_nelements(src0);
    மாறிலி int64_t இலக்கம்_blocks = (ne + SYCL_OPT_STEP_BLOCK_SIZE - 1) / SYCL_OPT_STEP_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(இலக்கம்_blocks * SYCL_OPT_STEP_BLOCK_SIZE, SYCL_OPT_STEP_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            opt_step_adamw_f32_kernel(src0_d, src0_grad_d, src0_grad_m_d, src0_grad_v_d, adamw_params_d, ne, item);
        });
}

வெறுமை ggml_sycl_opt_step_sgd(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/3);

    மாறிலி ggml_tensor * src0       = dst->src[0];
    மாறிலி ggml_tensor * src0_grad  = dst->src[1];
    மாறிலி ggml_tensor * sgd_params = dst->src[2];

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

    தசம       * src0_d      = (தசம       *) src0->data;
    மாறிலி தசம * src0_grad_d = (மாறிலி தசம *) src0_grad->data;
    மாறிலி தசம * sgd_params_d = (மாறிலி தசம *) sgd_params->data;

    மாறிலி int64_t ne = ggml_nelements(src0);
    மாறிலி int64_t இலக்கம்_blocks = (ne + SYCL_OPT_STEP_BLOCK_SIZE - 1) / SYCL_OPT_STEP_BLOCK_SIZE;

    stream->parallel_for(
        sycl::nd_range<1>(இலக்கம்_blocks * SYCL_OPT_STEP_BLOCK_SIZE, SYCL_OPT_STEP_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            opt_step_sgd_f32_kernel(src0_d, src0_grad_d, sgd_params_d, ne, item);
        });
}
