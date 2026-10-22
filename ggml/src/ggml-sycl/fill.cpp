#অন্তর্ভাব "fill.hpp"
#অন্তর্ভাব "common.hpp"

#define SYCL_FILL_BLOCK_আকার 256

template <typename T>
স্থির শূন্য fill_kernel(T * dst, ধ্রুবক int64_t k, ধ্রুবক T value,
                        ধ্রুবক sycl::nd_item<1> & item) {
    ধ্রুবক int64_t i = (int64_t)item.get_global_id(0);
    যদি (i >= k) {
        ফেরত;
    }
    dst[i] = value;
}

অন্তঃরেখ শূন্য ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    দশমিক value;
    memcpy(&value, dst->op_params, আকার(দশমিক));

    ধ্রুবক int64_t k = ggml_nelements(dst);
    ধ্রুবক int64_t অঙ্ক_blocks = (k + SYCL_FILL_BLOCK_আকার - 1) / SYCL_FILL_BLOCK_আকার;
    শূন্য * dst_d = dst->তথ্য;

    নির্বাচন (dst->type) {
        ক্ষেত্র GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(অঙ্ক_blocks * SYCL_FILL_BLOCK_আকার, SYCL_FILL_BLOCK_আকার),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<দশমিক *>(dst_d), k, value, item);
                });
            বিরতি;
        ক্ষেত্র GGML_TYPE_F16:
            {
                sycl::half h_মান = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(অঙ্ক_blocks * SYCL_FILL_BLOCK_আকার, SYCL_FILL_BLOCK_আকার),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_মান, item);
                    });
            }
            বিরতি;
        পূর্বনির্ধারিত:
            GGML_ABORT("unsupported type");
    }
}

শূন্য ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
