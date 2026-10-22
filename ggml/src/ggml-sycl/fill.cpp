#ضمن "fill.hpp"
#ضمن "common.hpp"

#تعریف_کرو SYCL_FILL_بلاک_ناپ 256

سانچا <قسم_ناں T>
ساكن فراغ fill_kernel(T * dst, ثابت int64_t k, ثابت T value,
                        ثابت sycl::nd_شے<1> & شے) {
    ثابت int64_t i = (int64_t)شے.get_global_id(0);
    إذا (i >= k) {
        أرجع;
    }
    dst[i] = value;
}

مضمن فراغ ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::قطار_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    عائم value;
    حافظہ_نقل(&value, dst->op_params, حجم(عائم));

    ثابت int64_t k = ggml_nelements(dst);
    ثابت int64_t عدد_بلاکاں = (k + SYCL_FILL_بلاک_ناپ - 1) / SYCL_FILL_بلاک_ناپ;
    فراغ * dst_d = dst->انکڑے;

    بدل (dst->type) {
        حالة GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(عدد_بلاکاں * SYCL_FILL_بلاک_ناپ, SYCL_FILL_بلاک_ناپ),
                [=](sycl::nd_شے<1> شے) {
                    fill_kernel(static_cast<عائم *>(dst_d), k, value, شے);
                });
            اكسر;
        حالة GGML_TYPE_F16:
            {
                sycl::half h_قدر = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(عدد_بلاکاں * SYCL_FILL_بلاک_ناپ, SYCL_FILL_بلاک_ناپ),
                    [=](sycl::nd_شے<1> شے) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_قدر, شے);
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
