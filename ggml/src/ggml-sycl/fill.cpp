#ضمن "fill.hpp"
#ضمن "common.hpp"

#تعریف_کن SYCL_FILL_بلوک_اندازه 256

قالب <نام_نوع T>
ساكن فراغ fill_kernel(T * dst, ثابت int64_t k, ثابت T value,
                        ثابت sycl::nd_قلم<1> & قلم) {
    ثابت int64_t i = (int64_t)قلم.get_global_id(0);
    إذا (i >= k) {
        أرجع;
    }
    dst[i] = value;
}

مضمن فراغ ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::صف_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_مجموعه_device(ctx.device));

    عائم value;
    کپی_حافظه(&value, dst->op_params, حجم(عائم));

    ثابت int64_t k = ggml_nelements(dst);
    ثابت int64_t رقم_بلوکها = (k + SYCL_FILL_بلوک_اندازه - 1) / SYCL_FILL_بلوک_اندازه;
    فراغ * dst_d = dst->داده;

    بدل (dst->type) {
        حالة GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(رقم_بلوکها * SYCL_FILL_بلوک_اندازه, SYCL_FILL_بلوک_اندازه),
                [=](sycl::nd_قلم<1> قلم) {
                    fill_kernel(static_cast<عائم *>(dst_d), k, value, قلم);
                });
            اكسر;
        حالة GGML_TYPE_F16:
            {
                sycl::half h_مقدار = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(رقم_بلوکها * SYCL_FILL_بلوک_اندازه, SYCL_FILL_بلوک_اندازه),
                    [=](sycl::nd_قلم<1> قلم) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_مقدار, قلم);
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
