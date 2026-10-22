#شامل "fill.hpp"
#شامل "common.hpp"

#تعریف_کرو SYCL_FILL_بلاک_سائز 256

سانچہ <قسم_نام T>
ساکن خالی fill_kernel(T * dst, ثابت int64_t k, ثابت T value,
                        ثابت sycl::nd_شے<1> & شے) {
    ثابت int64_t i = (int64_t)شے.get_global_id(0);
    اگر (i >= k) {
        واپس;
    }
    dst[i] = value;
}

اندرونی خالی ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::قطار_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_سیٹ_device(ctx.device));

    اعشاری value;
    حافظہ_نقل(&value, dst->op_params, حجم(اعشاری));

    ثابت int64_t k = ggml_nelements(dst);
    ثابت int64_t عدد_بلاکیں = (k + SYCL_FILL_بلاک_سائز - 1) / SYCL_FILL_بلاک_سائز;
    خالی * dst_d = dst->ڈیٹا;

    انتخاب (dst->type) {
        حالت GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(عدد_بلاکیں * SYCL_FILL_بلاک_سائز, SYCL_FILL_بلاک_سائز),
                [=](sycl::nd_شے<1> شے) {
                    fill_kernel(static_cast<اعشاری *>(dst_d), k, value, شے);
                });
            توڑو;
        حالت GGML_TYPE_F16:
            {
                sycl::half h_قدر = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(عدد_بلاکیں * SYCL_FILL_بلاک_سائز, SYCL_FILL_بلاک_سائز),
                    [=](sycl::nd_شے<1> شے) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_قدر, شے);
                    });
            }
            توڑو;
        طےشدہ:
            GGML_ABORT("unsupported type");
    }
}

خالی ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
