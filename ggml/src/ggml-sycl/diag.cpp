#અન્તર્ભાવ "diag.hpp"
#અન્તર્ભાવ "common.hpp"

#define SYCL_DIAG_BLOCK_SIZE 256

template <typename T>
સ્થિર શૂન્ય diag_kernel(T * __restrict__ dst, અચલ T * __restrict__ src,
                        અચલ int64_t ne0, અચલ int64_t ne1,
                        અચલ int64_t ne2, અચલ int64_t ne3,
                        અચલ int64_t total_elements,
                        અચલ sycl::nd_item<1> & item) {
    અચલ int64_t i = item.get_global_id(0);
    જો (i >= total_elements) {
        પાછા;
    }

    અચલ int64_t i0 = i % ne0;
    અચલ int64_t i1 = (i / ne0) % ne1;
    અચલ int64_t i2 = (i / (ne0 * ne1)) % ne2;
    અચલ int64_t i3 = i / (ne0 * ne1 * ne2);

    અચલ int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    જો (i0 == i1) {
        અચલ int64_t batch_idx = i3 * ne2 + i2;
        dst[dst_idx] = src[batch_idx * ne0 + i0];
    } નહીંતર {
        dst[dst_idx] = T(0);
    }

    (શૂન્ય)ne3;
}

અંતઃરેખ શૂન્ય ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[1] == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    અચલ શૂન્ય * src0_d = src0->data;
    શૂન્ય * dst_d = dst->data;

    અચલ int64_t ne0 = dst->ne[0];
    અચલ int64_t ne1 = dst->ne[1];
    અચલ int64_t ne2 = dst->ne[2];
    અચલ int64_t ne3 = dst->ne[3];
    અચલ int64_t n_elems = ggml_nelements(dst);
    અચલ int64_t અંક_blocks = (n_elems + SYCL_DIAG_BLOCK_SIZE - 1) / SYCL_DIAG_BLOCK_SIZE;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    stream->parallel_for(
        sycl::nd_range<1>(અંક_blocks * SYCL_DIAG_BLOCK_SIZE, SYCL_DIAG_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            diag_kernel(static_cast<દશાંશ *>(dst_d),
                        static_cast<અચલ દશાંશ *>(src0_d),
                        ne0, ne1, ne2, ne3, n_elems, item);
        });
}

શૂન્ય ggml_sycl_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag(ctx, dst);
}
