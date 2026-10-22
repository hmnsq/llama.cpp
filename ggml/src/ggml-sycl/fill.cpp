#ಅನ್ತರ್ಭಾವ "fill.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"

#define SYCL_FILL_BLOCK_SIZE 256

template <typename T>
ಸ್ಥಿರ ಶೂನ್ಯ fill_kernel(T * dst, ಧ್ರುವ int64_t k, ಧ್ರುವ T value,
                        ಧ್ರುವ sycl::nd_item<1> & item) {
    ಧ್ರುವ int64_t i = (int64_t)item.get_global_id(0);
    ಯದಿ (i >= k) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    dst[i] = value;
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ಭಿನ್ನ value;
    memcpy(&value, dst->op_params, ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಧ್ರುವ int64_t k = ggml_nelements(dst);
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (k + SYCL_FILL_BLOCK_SIZE - 1) / SYCL_FILL_BLOCK_SIZE;
    ಶೂನ್ಯ * dst_d = dst->data;

    ವಿಕಲ್ಪ (dst->type) {
        ಪಕ್ಷ GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(ಅಂಕೆ_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<ಭಿನ್ನ *>(dst_d), k, value, item);
                });
            ಭಙ್ಗ;
        ಪಕ್ಷ GGML_TYPE_F16:
            {
                sycl::half h_ಮೌಲ್ಯ = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(ಅಂಕೆ_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_ಮೌಲ್ಯ, item);
                    });
            }
            ಭಙ್ಗ;
        ಸ್ವಭಾವ:
            GGML_ABORT("unsupported type");
    }
}

ಶೂನ್ಯ ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
