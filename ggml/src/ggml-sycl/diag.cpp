#ಅನ್ತರ್ಭಾವ "diag.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"

#define SYCL_DIAG_BLOCK_SIZE 256

template <typename T>
ಸ್ಥಿರ ಶೂನ್ಯ diag_kernel(T * __restrict__ dst, ಧ್ರುವ T * __restrict__ src,
                        ಧ್ರುವ int64_t ne0, ಧ್ರುವ int64_t ne1,
                        ಧ್ರುವ int64_t ne2, ಧ್ರುವ int64_t ne3,
                        ಧ್ರುವ int64_t total_elements,
                        ಧ್ರುವ sycl::nd_item<1> & item) {
    ಧ್ರುವ int64_t i = item.get_global_id(0);
    ಯದಿ (i >= total_elements) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ int64_t i0 = i % ne0;
    ಧ್ರುವ int64_t i1 = (i / ne0) % ne1;
    ಧ್ರುವ int64_t i2 = (i / (ne0 * ne1)) % ne2;
    ಧ್ರುವ int64_t i3 = i / (ne0 * ne1 * ne2);

    ಧ್ರುವ int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    ಯದಿ (i0 == i1) {
        ಧ್ರುವ int64_t batch_idx = i3 * ne2 + i2;
        dst[dst_idx] = src[batch_idx * ne0 + i0];
    } ಅನ್ಯಥಾ {
        dst[dst_idx] = T(0);
    }

    (ಶೂನ್ಯ)ne3;
}

ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[1] == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    ಧ್ರುವ ಶೂನ್ಯ * src0_d = src0->data;
    ಶೂನ್ಯ * dst_d = dst->data;

    ಧ್ರುವ int64_t ne0 = dst->ne[0];
    ಧ್ರುವ int64_t ne1 = dst->ne[1];
    ಧ್ರುವ int64_t ne2 = dst->ne[2];
    ಧ್ರುವ int64_t ne3 = dst->ne[3];
    ಧ್ರುವ int64_t n_elems = ggml_nelements(dst);
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (n_elems + SYCL_DIAG_BLOCK_SIZE - 1) / SYCL_DIAG_BLOCK_SIZE;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    stream->parallel_for(
        sycl::nd_range<1>(ಅಂಕೆ_blocks * SYCL_DIAG_BLOCK_SIZE, SYCL_DIAG_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            diag_kernel(static_cast<ಭಿನ್ನ *>(dst_d),
                        static_cast<ಧ್ರುವ ಭಿನ್ನ *>(src0_d),
                        ne0, ne1, ne2, ne3, n_elems, item);
        });
}

ಶೂನ್ಯ ggml_sycl_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag(ctx, dst);
}
