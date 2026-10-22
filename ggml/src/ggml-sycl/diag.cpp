#अन्तर्भाव "diag.hpp"
#अन्तर्भाव "common.hpp"

#परिभाषय SYCL_DIAG_खण्डम्_आकार 256

प्रतिरूपम् <प्रकारनाम T>
स्थिर शून्य diag_kernel(T * __restrict__ dst, ध्रुव T * __restrict__ src,
                        ध्रुव int64_t ne0, ध्रुव int64_t ne1,
                        ध्रुव int64_t ne2, ध्रुव int64_t ne3,
                        ध्रुव int64_t योग_elements,
                        ध्रुव sycl::nd_item<1> & item) {
    ध्रुव int64_t i = item.get_global_id(0);
    यदि (i >= योग_elements) {
        प्रत्यागम;
    }

    ध्रुव int64_t i0 = i % ne0;
    ध्रुव int64_t i1 = (i / ne0) % ne1;
    ध्रुव int64_t i2 = (i / (ne0 * ne1)) % ne2;
    ध्रुव int64_t i3 = i / (ne0 * ne1 * ne2);

    ध्रुव int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    यदि (i0 == i1) {
        ध्रुव int64_t batch_idx = i3 * ne2 + i2;
        dst[dst_idx] = src[batch_idx * ne0 + i0];
    } अन्यथा {
        dst[dst_idx] = T(0);
    }

    (शून्य)ne3;
}

अन्तःस्थ शून्य ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[1] == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));

    ध्रुव शून्य * src0_d = src0->दत्त;
    शून्य * dst_d = dst->दत्त;

    ध्रुव int64_t ne0 = dst->ne[0];
    ध्रुव int64_t ne1 = dst->ne[1];
    ध्रुव int64_t ne2 = dst->ne[2];
    ध्रुव int64_t ne3 = dst->ne[3];
    ध्रुव int64_t n_elems = ggml_nelements(dst);
    ध्रुव int64_t अङ्कः_खण्डानि = (n_elems + SYCL_DIAG_खण्डम्_आकार - 1) / SYCL_DIAG_खण्डम्_आकार;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    stream->parallel_for(
        sycl::nd_range<1>(अङ्कः_खण्डानि * SYCL_DIAG_खण्डम्_आकार, SYCL_DIAG_खण्डम्_आकार),
        [=](sycl::nd_item<1> item) {
            diag_kernel(static_cast<भिन्न *>(dst_d),
                        static_cast<ध्रुव भिन्न *>(src0_d),
                        ne0, ne1, ne2, ne3, n_elems, item);
        });
}

शून्य ggml_sycl_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag(ctx, dst);
}
