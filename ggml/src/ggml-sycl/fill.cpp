#ଅନ୍ତର୍ଭାଵ "fill.hpp"
#ଅନ୍ତର୍ଭାଵ "common.hpp"

#define SYCL_FILL_BLOCK_SIZE 256

template <typename T>
ସ୍ଥିର ଶୂନ୍ୟ fill_kernel(T * dst, ଅଚଳ int64_t k, ଅଚଳ T value,
                        ଅଚଳ sycl::nd_item<1> & item) {
    ଅଚଳ int64_t i = (int64_t)item.get_global_id(0);
    ଯଦି (i >= k) {
        ଫେରନ୍ତୁ;
    }
    dst[i] = value;
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ଦଶମିକ value;
    memcpy(&value, dst->op_params, ଆକାର(ଦଶମିକ));

    ଅଚଳ int64_t k = ggml_nelements(dst);
    ଅଚଳ int64_t ଅଙ୍କ_blocks = (k + SYCL_FILL_BLOCK_SIZE - 1) / SYCL_FILL_BLOCK_SIZE;
    ଶୂନ୍ୟ * dst_d = dst->data;

    ବାଛ (dst->type) {
        ସ୍ଥିତି GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(ଅଙ୍କ_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<ଦଶମିକ *>(dst_d), k, value, item);
                });
            ଭାଙ୍ଗ;
        ସ୍ଥିତି GGML_TYPE_F16:
            {
                sycl::half h_ମୂଲ୍ୟ = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(ଅଙ୍କ_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_ମୂଲ୍ୟ, item);
                    });
            }
            ଭାଙ୍ଗ;
        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            GGML_ABORT("unsupported type");
    }
}

ଶୂନ୍ୟ ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
