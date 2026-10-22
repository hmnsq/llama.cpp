#अन्तर्भाव "fill.hpp"
#अन्तर्भाव "common.hpp"

#परिभाषय SYCL_FILL_खण्डम्_आकार 256

प्रतिरूपम् <प्रकारनाम T>
स्थिर शून्य fill_kernel(T * dst, ध्रुव int64_t k, ध्रुव T value,
                        ध्रुव sycl::nd_item<1> & item) {
    ध्रुव int64_t i = (int64_t)item.get_global_id(0);
    यदि (i >= k) {
        प्रत्यागम;
    }
    dst[i] = value;
}

अन्तःस्थ शून्य ggml_sycl_op_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(ggml_is_contiguous(dst));

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));

    भिन्न value;
    स्मृतिं_प्रतिलिख(&value, dst->op_params, परिमाण(भिन्न));

    ध्रुव int64_t k = ggml_nelements(dst);
    ध्रुव int64_t अङ्कः_खण्डानि = (k + SYCL_FILL_खण्डम्_आकार - 1) / SYCL_FILL_खण्डम्_आकार;
    शून्य * dst_d = dst->दत्त;

    विकल्प (dst->type) {
        पक्ष GGML_TYPE_F32:
            stream->parallel_for(
                sycl::nd_range<1>(अङ्कः_खण्डानि * SYCL_FILL_खण्डम्_आकार, SYCL_FILL_खण्डम्_आकार),
                [=](sycl::nd_item<1> item) {
                    fill_kernel(static_cast<भिन्न *>(dst_d), k, value, item);
                });
            भङ्ग;
        पक्ष GGML_TYPE_F16:
            {
                sycl::half h_मानम् = sycl::half(value);
                stream->parallel_for(
                    sycl::nd_range<1>(अङ्कः_खण्डानि * SYCL_FILL_खण्डम्_आकार, SYCL_FILL_खण्डम्_आकार),
                    [=](sycl::nd_item<1> item) {
                        fill_kernel(static_cast<sycl::half *>(dst_d), k, h_मानम्, item);
                    });
            }
            भङ्ग;
        स्वभाव:
            GGML_ABORT("unsupported type");
    }
}

शून्य ggml_sycl_fill(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/0);
    ggml_sycl_op_fill(ctx, dst);
}
