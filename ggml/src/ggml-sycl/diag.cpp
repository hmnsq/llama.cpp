#অন্তর্ভাব "diag.hpp"
#অন্তর্ভাব "common.hpp"

#define SYCL_DIAG_BLOCK_আকার 256

template <typename T>
স্থির শূন্য diag_kernel(T * __restrict__ dst, ধ্রুবক T * __restrict__ src,
                        ধ্রুবক int64_t ne0, ধ্রুবক int64_t ne1,
                        ধ্রুবক int64_t ne2, ধ্রুবক int64_t ne3,
                        ধ্রুবক int64_t যোগ_elements,
                        ধ্রুবক sycl::nd_item<1> & item) {
    ধ্রুবক int64_t i = item.get_global_id(0);
    যদি (i >= যোগ_elements) {
        ফেরত;
    }

    ধ্রুবক int64_t i0 = i % ne0;
    ধ্রুবক int64_t i1 = (i / ne0) % ne1;
    ধ্রুবক int64_t i2 = (i / (ne0 * ne1)) % ne2;
    ধ্রুবক int64_t i3 = i / (ne0 * ne1 * ne2);

    ধ্রুবক int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    যদি (i0 == i1) {
        ধ্রুবক int64_t batch_idx = i3 * ne2 + i2;
        dst[dst_idx] = src[batch_idx * ne0 + i0];
    } অন্যথা {
        dst[dst_idx] = T(0);
    }

    (শূন্য)ne3;
}

অন্তঃরেখ শূন্য ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুবক ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[1] == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ধ্রুবক শূন্য * src0_d = src0->তথ্য;
    শূন্য * dst_d = dst->তথ্য;

    ধ্রুবক int64_t ne0 = dst->ne[0];
    ধ্রুবক int64_t ne1 = dst->ne[1];
    ধ্রুবক int64_t ne2 = dst->ne[2];
    ধ্রুবক int64_t ne3 = dst->ne[3];
    ধ্রুবক int64_t n_elems = ggml_nelements(dst);
    ধ্রুবক int64_t অঙ্ক_blocks = (n_elems + SYCL_DIAG_BLOCK_আকার - 1) / SYCL_DIAG_BLOCK_আকার;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    stream->parallel_for(
        sycl::nd_range<1>(অঙ্ক_blocks * SYCL_DIAG_BLOCK_আকার, SYCL_DIAG_BLOCK_আকার),
        [=](sycl::nd_item<1> item) {
            diag_kernel(static_cast<দশমিক *>(dst_d),
                        static_cast<ধ্রুবক দশমিক *>(src0_d),
                        ne0, ne1, ne2, ne3, n_elems, item);
        });
}

শূন্য ggml_sycl_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag(ctx, dst);
}
