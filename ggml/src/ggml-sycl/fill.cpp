#அந்தர்பாவ "fill.hpp"
#அந்தர்பாவ "common.hpp"

#define SYCL_FILL_BLOCK_SIZE 256

template <typename T>
நிலையான வெறுமை fill_kernel(T * dst, மாறிலி int64_t k, மாறிலி T value,
                        மாறிலி sycl::nd_item<1> & item) {
    மாறிலி int64_t i = (int64_t)item.get_global_id(0);
    எனில் (i >= k) {
        திரும்பு;
    }
    dst[i] = value;
}

உள்வரி வெறுமை ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    தசம value;
    memcpy(&value, dst->op_params, அளவு(தசம));

    மாறிலி int64_t k = ggml_nelements(dst);
    மாறிலி int64_t இலக்கம்_blocks = (k + SYCL_FILL_BLOCK_SIZE - 1) / SYCL_FILL_BLOCK_SIZE;
    வெறுமை * dst_d = dst->data;

    தேர்வு (dst->type) {
        நிலை GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(இலக்கம்_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<தசம *>(dst_d), k, value, item);
                });
            நிறுத்து;
        நிலை GGML_TYPE_F16:
            {
                sycl::half h_மதிப்பு = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(இலக்கம்_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_மதிப்பு, item);
                    });
            }
            நிறுத்து;
        இயல்பு:
            GGML_ABORT("unsupported type");
    }
}

வெறுமை ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
