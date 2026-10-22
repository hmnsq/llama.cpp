#അന്തര്ഭാവ "fill.hpp"
#അന്തര്ഭാവ "common.hpp"

#define SYCL_FILL_BLOCK_SIZE 256

template <typename T>
സ്ഥിതം ശൂന്യം fill_kernel(T * dst, സ്ഥിരം int64_t k, സ്ഥിരം T value,
                        സ്ഥിരം sycl::nd_item<1> & item) {
    സ്ഥിരം int64_t i = (int64_t)item.get_global_id(0);
    എങ്കിൽ (i >= k) {
        തിരികെ;
    }
    dst[i] = value;
}

വരിയിൽ ശൂന്യം ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ദശാംശം value;
    memcpy(&value, dst->op_params, വലുപ്പം(ദശാംശം));

    സ്ഥിരം int64_t k = ggml_nelements(dst);
    സ്ഥിരം int64_t അക്കം_blocks = (k + SYCL_FILL_BLOCK_SIZE - 1) / SYCL_FILL_BLOCK_SIZE;
    ശൂന്യം * dst_d = dst->data;

    തിരഞ്ഞെടുപ്പ് (dst->type) {
        അവസ്ഥ GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(അക്കം_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<ദശാംശം *>(dst_d), k, value, item);
                });
            നിർത്തുക;
        അവസ്ഥ GGML_TYPE_F16:
            {
                sycl::half h_മൂല്യം = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(അക്കം_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_മൂല്യം, item);
                    });
            }
            നിർത്തുക;
        സ്ഥിരസ്ഥിതി:
            GGML_ABORT("unsupported type");
    }
}

ശൂന്യം ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
