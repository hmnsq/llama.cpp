#অন্তর্ভাৱ "diag.hpp"
#অন্তর্ভাৱ "common.hpp"

#define SYCL_DIAG_BLOCK_SIZE 256

template <typename T>
স্থির শূন্য diag_kernel(T * __restrict__ dst, ধ্রুৱ T * __restrict__ src,
                        ধ্রুৱ int64_t ne0, ধ্রুৱ int64_t ne1,
                        ধ্রুৱ int64_t ne2, ধ্রুৱ int64_t ne3,
                        ধ্রুৱ int64_t total_elements,
                        ধ্রুৱ sycl::nd_item<1> & item) {
    ধ্রুৱ int64_t i = item.get_global_id(0);
    যদি (i >= total_elements) {
        ঘূৰাই;
    }

    ধ্রুৱ int64_t i0 = i % ne0;
    ধ্রুৱ int64_t i1 = (i / ne0) % ne1;
    ধ্রুৱ int64_t i2 = (i / (ne0 * ne1)) % ne2;
    ধ্রুৱ int64_t i3 = i / (ne0 * ne1 * ne2);

    ধ্রুৱ int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    যদি (i0 == i1) {
        ধ্রুৱ int64_t batch_idx = i3 * ne2 + i2;
        dst[dst_idx] = src[batch_idx * ne0 + i0];
    } অন্যথা {
        dst[dst_idx] = T(0);
    }

    (শূন্য)ne3;
}

অন্তঃস্থ শূন্য ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুৱ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[1] == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুৱ শূন্য * src0_d = src0->data;
    শূন্য * dst_d = dst->data;

    ধ্রুৱ int64_t ne0 = dst->ne[0];
    ধ্রুৱ int64_t ne1 = dst->ne[1];
    ধ্রুৱ int64_t ne2 = dst->ne[2];
    ধ্রুৱ int64_t ne3 = dst->ne[3];
    ধ্রুৱ int64_t n_elems = ggml_nelements(dst);
    ধ্রুৱ int64_t অঙ্ক_blocks = (n_elems + SYCL_DIAG_BLOCK_SIZE - 1) / SYCL_DIAG_BLOCK_SIZE;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    stream->parallel_for(
        sycl::nd_range<1>(অঙ্ক_blocks * SYCL_DIAG_BLOCK_SIZE, SYCL_DIAG_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            diag_kernel(static_cast<ভিন্ন *>(dst_d),
                        static_cast<ধ্রুৱ ভিন্ন *>(src0_d),
                        ne0, ne1, ne2, ne3, n_elems, item);
        });
}

শূন্য ggml_sycl_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag(ctx, dst);
}
