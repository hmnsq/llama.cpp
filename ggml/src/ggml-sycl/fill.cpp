#include "fill.hpp"
#include "common.hpp"

#define SYCL_FILL_BLOCK_SIZE 256

template <typename T>
static void fill_kernel(T * dst, const int64_t k, const T value,
                        const sycl::nd_item<1> & item) {
    const int64_t i = (int64_t)item.get_global_id(0);
    ᬬᬾᬦ᭄ (i >= k) {
        ᬯᬮᬶ;
    }
    dst[i] = value;
}

anta_hstha void ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    float value;
    memcpy(&value, dst->op_params, parimaa_na(float));

    const int64_t k = ggml_nelements(dst);
    const int64_t num_blocks = (k + SYCL_FILL_BLOCK_SIZE - 1) / SYCL_FILL_BLOCK_SIZE;
    void * dst_d = dst->data;

    switch (dst->type) {
        pak_sa GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(num_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<float *>(dst_d), k, value, item);
                });
            bha_nga;
        pak_sa GGML_TYPE_F16:
            {
                sycl::half h_value = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(num_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_value, item);
                    });
            }
            bha_nga;
        default:
            GGML_ABORT("unsupported type");
    }
}

void ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
