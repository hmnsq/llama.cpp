#অন্তর্ভাৱ "fill.hpp"
#অন্তর্ভাৱ "common.hpp"

#define SYCL_FILL_BLOCK_SIZE 256

template <typename T>
স্থির শূন্য fill_kernel(T * dst, ধ্রুৱ int64_t k, ধ্রুৱ T value,
                        ধ্রুৱ sycl::nd_item<1> & item) {
    ধ্রুৱ int64_t i = (int64_t)item.get_global_id(0);
    যদি (i >= k) {
        ঘূৰাই;
    }
    dst[i] = value;
}

অন্তঃস্থ শূন্য ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ভিন্ন value;
    memcpy(&value, dst->op_params, পরিমাণ(ভিন্ন));

    ধ্রুৱ int64_t k = ggml_nelements(dst);
    ধ্রুৱ int64_t অঙ্ক_blocks = (k + SYCL_FILL_BLOCK_SIZE - 1) / SYCL_FILL_BLOCK_SIZE;
    শূন্য * dst_d = dst->data;

    ৱিকল্প (dst->type) {
        পক্ষ GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(অঙ্ক_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<ভিন্ন *>(dst_d), k, value, item);
                });
            ভঙ্গ;
        পক্ষ GGML_TYPE_F16:
            {
                sycl::half h_মান = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(অঙ্ক_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_মান, item);
                    });
            }
            ভঙ্গ;
        স্ৱভাৱ:
            GGML_ABORT("unsupported type");
    }
}

শূন্য ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
