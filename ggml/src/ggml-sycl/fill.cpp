#ضمن "fill.hpp"
#ضمن "common.hpp"

#تعریف_کړه SYCL_FILL_بلاک_اندازه 256

کينډۍ <ډول_نوم T>
ساكن فراغ fill_kernel(T * dst, ثابت int64_t k, ثابت T value,
                        ثابت sycl::nd_توکی<1> & توکی) {
    ثابت int64_t i = (int64_t)توکی.get_global_id(0);
    إذا (i >= k) {
        أرجع;
    }
    dst[i] = value;
}

مضمن فراغ ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::کتار_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_سېټ_device(ctx.device));

    عائم value;
    حافظه_نقل(&value, dst->op_params, حجم(عائم));

    ثابت int64_t k = ggml_nelements(dst);
    ثابت int64_t عدد_بلاکونه = (k + SYCL_FILL_بلاک_اندازه - 1) / SYCL_FILL_بلاک_اندازه;
    فراغ * dst_d = dst->معلومات;

    بدل (dst->type) {
        حالة GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(عدد_بلاکونه * SYCL_FILL_بلاک_اندازه, SYCL_FILL_بلاک_اندازه),
                [=](sycl::nd_توکی<1> توکی) {
                    fill_kernel(static_cast<عائم *>(dst_d), k, value, توکی);
                });
            اكسر;
        حالة GGML_TYPE_F16:
            {
                sycl::half h_ارزښت = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(عدد_بلاکونه * SYCL_FILL_بلاک_اندازه, SYCL_FILL_بلاک_اندازه),
                    [=](sycl::nd_توکی<1> توکی) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_ارزښت, توکی);
                    });
            }
            اكسر;
        افتراضي:
            GGML_ABORT("unsupported type");
    }
}

فراغ ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
