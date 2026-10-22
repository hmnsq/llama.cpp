#अन्तर्भाव "diag.hpp"
#अन्तर्भाव "common.hpp"

#define SYCL_DIAG_BLOCK_SIZE 256

ढाँचा <typename T>
स्थिर शून्य diag_kernel(T * __restrict__ dst, स्थिरांक T * __restrict__ src,
                        स्थिरांक int64_t ne0, स्थिरांक int64_t ne1,
                        स्थिरांक int64_t ne2, स्थिरांक int64_t ne3,
                        स्थिरांक int64_t total_elements,
                        स्थिरांक sycl::nd_item<1> & item) {
    स्थिरांक int64_t i = item.get_global_id(0);
    जर (i >= total_elements) {
        परत;
    }

    स्थिरांक int64_t i0 = i % ne0;
    स्थिरांक int64_t i1 = (i / ne0) % ne1;
    स्थिरांक int64_t i2 = (i / (ne0 * ne1)) % ne2;
    स्थिरांक int64_t i3 = i / (ne0 * ne1 * ne2);

    स्थिरांक int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    जर (i0 == i1) {
        स्थिरांक int64_t batch_idx = i3 * ne2 + i2;
        dst[dst_idx] = src[batch_idx * ne0 + i0];
    } अन्यथा {
        dst[dst_idx] = T(0);
    }

    (शून्य)ne3;
}

ओळीत शून्य ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[1] == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    स्थिरांक शून्य * src0_d = src0->data;
    शून्य * dst_d = dst->data;

    स्थिरांक int64_t ne0 = dst->ne[0];
    स्थिरांक int64_t ne1 = dst->ne[1];
    स्थिरांक int64_t ne2 = dst->ne[2];
    स्थिरांक int64_t ne3 = dst->ne[3];
    स्थिरांक int64_t n_elems = ggml_nelements(dst);
    स्थिरांक int64_t अंक_blocks = (n_elems + SYCL_DIAG_BLOCK_SIZE - 1) / SYCL_DIAG_BLOCK_SIZE;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    stream->parallel_for(
        sycl::nd_range<1>(अंक_blocks * SYCL_DIAG_BLOCK_SIZE, SYCL_DIAG_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            diag_kernel(static_cast<दशांश *>(dst_d),
                        static_cast<स्थिरांक दशांश *>(src0_d),
                        ne0, ne1, ne2, ne3, n_elems, item);
        });
}

शून्य ggml_sycl_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag(ctx, dst);
}
