#अन्तर्भाव "fill.hpp"
#अन्तर्भाव "common.hpp"

#define SYCL_FILL_BLOCK_SIZE 256

ढाँचा <typename T>
स्थिर शून्य fill_kernel(T * dst, स्थिरांक int64_t k, स्थिरांक T value,
                        स्थिरांक sycl::nd_item<1> & item) {
    स्थिरांक int64_t i = (int64_t)item.get_global_id(0);
    जर (i >= k) {
        परत;
    }
    dst[i] = value;
}

ओळीत शून्य ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    दशांश value;
    स_नकल(&value, dst->op_params, आकार(दशांश));

    स्थिरांक int64_t k = ggml_nelements(dst);
    स्थिरांक int64_t अंक_blocks = (k + SYCL_FILL_BLOCK_SIZE - 1) / SYCL_FILL_BLOCK_SIZE;
    शून्य * dst_d = dst->data;

    निवड (dst->type) {
        स्थिती GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(अंक_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<दशांश *>(dst_d), k, value, item);
                });
            थांबा;
        स्थिती GGML_TYPE_F16:
            {
                sycl::half h_मूल्य = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(अंक_blocks * SYCL_FILL_BLOCK_SIZE, SYCL_FILL_BLOCK_SIZE),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_मूल्य, item);
                    });
            }
            थांबा;
        पूर्वनिर्धारित:
            GGML_ABORT("unsupported type");
    }
}

शून्य ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
