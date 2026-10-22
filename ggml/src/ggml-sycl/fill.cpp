#अन्तर्भाव "fill.hpp"
#अन्तर्भाव "common.hpp"

#define SYCL_FILL_BLOCK_SIZE 256

ढाँचा <typename T>
स्थिर शून्य fill_kernel(T * dst, अचर int64_t k, अचर T value,
                        अचर sycl::nd_item<1> & item) {
    अचर int64_t i = (int64_t)item.get_global_id(0);
    यदि (i >= k) {
        फर्काऊ;
    }
    dst[i] = value;
}

अन्तःस्थ शून्य ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    भिन्न value;
    स_नकल(&value, dst->op_params, परिमाण(भिन्न));

    अचर int64_t k = ggml_nelements(dst);
    अचर int64_t अङ्क_blocks = (k + SYCL_FILL_BLOCK_SIZE - 1) / SYCL_FILL_BLOCK_SIZE;
    शून्य * dst_d = dst->data;

    विकल्प (dst->type) {
        पक्ष GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(अङ्क_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<भिन्न *>(dst_d), k, value, item);
                });
            रोक;
        पक्ष GGML_TYPE_F16:
            {
                sycl::half h_मान = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(अङ्क_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_मान, item);
                    });
            }
            रोक;
        स्वभाव:
            GGML_ABORT("unsupported type");
    }
}

शून्य ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
