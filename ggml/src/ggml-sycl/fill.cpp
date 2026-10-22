#ضمن "fill.hpp"
#ضمن "common.hpp"

#define SYCL_FILL_BLOCK_SIZE 256

template <typename T>
ساكن فراغ fill_kernel(T * dst, ثابت int64_t k, ثابت T value,
                        ثابت sycl::nd_item<1> & item) {
    ثابت int64_t i = (int64_t)item.get_global_id(0);
    אם (i >= k) {
        החזר;
    }
    dst[i] = value;
}

مضمن فراغ ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    عائم value;
    memcpy(&value, dst->op_params, حجم(عائم));

    ثابت int64_t k = ggml_nelements(dst);
    ثابت int64_t num_blocks = (k + SYCL_FILL_BLOCK_SIZE - 1) / SYCL_FILL_BLOCK_SIZE;
    فراغ * dst_d = dst->data;

    بدل (dst->type) {
        حالة GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(num_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<عائم *>(dst_d), k, value, item);
                });
            اكسر;
        حالة GGML_TYPE_F16:
            {
                sycl::half h_value = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(num_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_value, item);
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
