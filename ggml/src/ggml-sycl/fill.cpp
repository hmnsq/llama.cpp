#ضمن "fill.hpp"
#ضمن "common.hpp"

#وصف_ڪريو SYCL_FILL_بلاڪ_سائيز 256

سانچو <قسم_نالو T>
ساكن فراغ fill_kernel(T * dst, ثابت int64_t k, ثابت T value,
                        ثابت sycl::nd_شي<1> & شي) {
    ثابت int64_t i = (int64_t)شي.get_global_id(0);
    إذا (i >= k) {
        أرجع;
    }
    dst[i] = value;
}

مضمن فراغ ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::قطار_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_سيٽ_device(ctx.device));

    عائم value;
    يادگيري_نقل(&value, dst->op_params, حجم(عائم));

    ثابت int64_t k = ggml_nelements(dst);
    ثابت int64_t انگ_بلاڪون = (k + SYCL_FILL_بلاڪ_سائيز - 1) / SYCL_FILL_بلاڪ_سائيز;
    فراغ * dst_d = dst->معلومات;

    بدل (dst->type) {
        حالة GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(انگ_بلاڪون * SYCL_FILL_بلاڪ_سائيز, SYCL_FILL_بلاڪ_سائيز),
                [=](sycl::nd_شي<1> شي) {
                    fill_kernel(static_cast<عائم *>(dst_d), k, value, شي);
                });
            اكسر;
        حالة GGML_TYPE_F16:
            {
                sycl::half h_قدر = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(انگ_بلاڪون * SYCL_FILL_بلاڪ_سائيز, SYCL_FILL_بلاڪ_سائيز),
                    [=](sycl::nd_شي<1> شي) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_قدر, شي);
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
