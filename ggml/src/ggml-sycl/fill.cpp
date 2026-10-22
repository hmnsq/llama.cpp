#અન્તર્ભાવ "fill.hpp"
#અન્તર્ભાવ "common.hpp"

#define SYCL_FILL_BLOCK_SIZE 256

template <typename T>
સ્થિર શૂન્ય fill_kernel(T * dst, અચલ int64_t k, અચલ T value,
                        અચલ sycl::nd_item<1> & item) {
    અચલ int64_t i = (int64_t)item.get_global_id(0);
    જો (i >= k) {
        પાછા;
    }
    dst[i] = value;
}

અંતઃરેખ શૂન્ય ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    દશાંશ value;
    memcpy(&value, dst->op_params, કદ(દશાંશ));

    અચલ int64_t k = ggml_nelements(dst);
    અચલ int64_t અંક_blocks = (k + SYCL_FILL_BLOCK_SIZE - 1) / SYCL_FILL_BLOCK_SIZE;
    શૂન્ય * dst_d = dst->data;

    પસંદગી (dst->type) {
        સ્થિતિ GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(અંક_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<દશાંશ *>(dst_d), k, value, item);
                });
            તોડો;
        સ્થિતિ GGML_TYPE_F16:
            {
                sycl::half h_કિંમત = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(અંક_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_કિંમત, item);
                    });
            }
            તોડો;
        મૂળભૂત:
            GGML_ABORT("unsupported type");
    }
}

શૂન્ય ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
