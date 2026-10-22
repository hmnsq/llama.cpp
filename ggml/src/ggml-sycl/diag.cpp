#അന്തര്ഭാവ "diag.hpp"
#അന്തര്ഭാവ "common.hpp"

#define SYCL_DIAG_BLOCK_SIZE 256

template <typename T>
സ്ഥിതം ശൂന്യം diag_kernel(T * __restrict__ dst, സ്ഥിരം T * __restrict__ src,
                        സ്ഥിരം int64_t ne0, സ്ഥിരം int64_t ne1,
                        സ്ഥിരം int64_t ne2, സ്ഥിരം int64_t ne3,
                        സ്ഥിരം int64_t total_elements,
                        സ്ഥിരം sycl::nd_item<1> & item) {
    സ്ഥിരം int64_t i = item.get_global_id(0);
    എങ്കിൽ (i >= total_elements) {
        തിരികെ;
    }

    സ്ഥിരം int64_t i0 = i % ne0;
    സ്ഥിരം int64_t i1 = (i / ne0) % ne1;
    സ്ഥിരം int64_t i2 = (i / (ne0 * ne1)) % ne2;
    സ്ഥിരം int64_t i3 = i / (ne0 * ne1 * ne2);

    സ്ഥിരം int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    എങ്കിൽ (i0 == i1) {
        സ്ഥിരം int64_t batch_idx = i3 * ne2 + i2;
        dst[dst_idx] = src[batch_idx * ne0 + i0];
    } അല്ലെങ്കിൽ {
        dst[dst_idx] = T(0);
    }

    (ശൂന്യം)ne3;
}

വരിയിൽ ശൂന്യം ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[1] == 1);

    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    സ്ഥിരം ശൂന്യം * src0_d = src0->data;
    ശൂന്യം * dst_d = dst->data;

    സ്ഥിരം int64_t ne0 = dst->ne[0];
    സ്ഥിരം int64_t ne1 = dst->ne[1];
    സ്ഥിരം int64_t ne2 = dst->ne[2];
    സ്ഥിരം int64_t ne3 = dst->ne[3];
    സ്ഥിരം int64_t n_elems = ggml_nelements(dst);
    സ്ഥിരം int64_t അക്കം_blocks = (n_elems + SYCL_DIAG_BLOCK_SIZE - 1) / SYCL_DIAG_BLOCK_SIZE;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    stream->parallel_for(
        sycl::nd_range<1>(അക്കം_blocks * SYCL_DIAG_BLOCK_SIZE, SYCL_DIAG_BLOCK_SIZE),
        [=](sycl::nd_item<1> item) {
            diag_kernel(static_cast<ദശാംശം *>(dst_d),
                        static_cast<സ്ഥിരം ദശാംശം *>(src0_d),
                        ne0, ne1, ne2, ne3, n_elems, item);
        });
}

ശൂന്യം ggml_sycl_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag(ctx, dst);
}
