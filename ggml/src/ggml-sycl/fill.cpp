#समावेश "fill.hpp"
#समावेश "common.hpp"

#घोषणा SYCL_FILL_BLOCK_आकार 256

ढाँचा <typename T>
अटल व्योम fill_kernel(T * dst, स्थिर int64_t k, स्थिर T value,
                        स्थिर sycl::nd_item<1> & item) {
    स्थिर int64_t i = (int64_t)item.get_global_id(0);
    अगर (i >= k) {
        वापस;
    }
    dst[i] = value;
}

अंतरभूत व्योम ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    भग्न value;
    स_नकल(&value, dst->op_params, माप(भग्न));

    स्थिर int64_t k = ggml_nelements(dst);
    स्थिर int64_t अंक_blocks = (k + SYCL_FILL_BLOCK_आकार - 1) / SYCL_FILL_BLOCK_आकार;
    व्योम * dst_d = dst->तथ्य;

    चयन (dst->type) {
        हाल GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(अंक_blocks * SYCL_FILL_BLOCK_आकार, SYCL_FILL_BLOCK_आकार),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<भग्न *>(dst_d), k, value, item);
                });
            अवरोध;
        हाल GGML_TYPE_F16:
            {
                sycl::half h_मूल्य = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(अंक_blocks * SYCL_FILL_BLOCK_आकार, SYCL_FILL_BLOCK_आकार),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_मूल्य, item);
                    });
            }
            अवरोध;
        शेष:
            GGML_ABORT("unsupported type");
    }
}

व्योम ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
