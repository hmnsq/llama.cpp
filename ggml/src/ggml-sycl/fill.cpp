#شامل "fill.hpp"
#شامل "common.hpp"

#define SYCL_FILL_BLOCK_SIZE 256

template <typename T>
ایستا تهی fill_kernel(T * dst, ثابت int64_t k, ثابت T value,
                        ثابت sycl::nd_item<1> & item) {
    ثابت int64_t i = (int64_t)item.get_global_id(0);
    اگر (i >= k) {
        بازگشت;
    }
    dst[i] = value;
}

درونخطی تهی ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    اعشاری value;
    memcpy(&value, dst->op_params, اندازه(اعشاری));

    ثابت int64_t k = ggml_nelements(dst);
    ثابت int64_t رقم_blocks = (k + SYCL_FILL_BLOCK_SIZE - 1) / SYCL_FILL_BLOCK_SIZE;
    تهی * dst_d = dst->data;

    انتخاب (dst->type) {
        حالت GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(رقم_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<اعشاری *>(dst_d), k, value, item);
                });
            بشکن;
        حالت GGML_TYPE_F16:
            {
                sycl::half h_مقدار = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(رقم_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_مقدار, item);
                    });
            }
            بشکن;
        پیشفرض:
            GGML_ABORT("unsupported type");
    }
}

تهی ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
