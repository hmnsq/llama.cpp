#ଅନ୍ତର୍ଭାଵ "diag.hpp"
#ଅନ୍ତର୍ଭାଵ "common.hpp"

#define SYCL_DIAG_BLOCK_SIZE 256

template <typename T>
ସ୍ଥିର ଶୂନ୍ୟ diag_kernel(T * __restrict__ dst, ଅଚଳ T * __restrict__ src,
                        ଅଚଳ int64_t ne0, ଅଚଳ int64_t ne1,
                        ଅଚଳ int64_t ne2, ଅଚଳ int64_t ne3,
                        ଅଚଳ int64_t total_elements,
                        ଅଚଳ sycl::nd_item<1> & item) {
    ଅଚଳ int64_t i = item.get_global_id(0);
    ଯଦି (i >= total_elements) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ int64_t i0 = i % ne0;
    ଅଚଳ int64_t i1 = (i / ne0) % ne1;
    ଅଚଳ int64_t i2 = (i / (ne0 * ne1)) % ne2;
    ଅଚଳ int64_t i3 = i / (ne0 * ne1 * ne2);

    ଅଚଳ int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    ଯଦି (i0 == i1) {
        ଅଚଳ int64_t batch_idx = i3 * ne2 + i2;
        dst[dst_idx] = src[batch_idx * ne0 + i0];
    } ଅନ୍ୟଥା {
        dst[dst_idx] = T(0);
    }

    (ଶୂନ୍ୟ)ne3;
}

ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ଅଚଳ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[1] == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ଅଚଳ ଶୂନ୍ୟ * src0_d = src0->data;
    ଶୂନ୍ୟ * dst_d = dst->data;

    ଅଚଳ int64_t ne0 = dst->ne[0];
    ଅଚଳ int64_t ne1 = dst->ne[1];
    ଅଚଳ int64_t ne2 = dst->ne[2];
    ଅଚଳ int64_t ne3 = dst->ne[3];
    ଅଚଳ int64_t n_elems = ggml_nelements(dst);
    ଅଚଳ int64_t ଅଙ୍କ_blocks = (n_elems + SYCL_DIAG_BLOCK_SIZE - 1) / SYCL_DIAG_BLOCK_SIZE;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    stream->parallel_for(
        sycl::nd_range<1>(ଅଙ୍କ_blocks * SYCL_DIAG_BLOCK_SIZE, SYCL_DIAG_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            diag_kernel(static_cast<ଦଶମିକ *>(dst_d),
                        static_cast<ଅଚଳ ଦଶମିକ *>(src0_d),
                        ne0, ne1, ne2, ne3, n_elems, item);
        });
}

ଶୂନ୍ୟ ggml_sycl_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag(ctx, dst);
}
