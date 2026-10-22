#ਅਨ੍ਤਰ੍ਭਾਵ "diag.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SYCL_DIAG_ਖੰਡ_ਮਾਪ 256

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ diag_kernel(T * __restrict__ dst, ਅਚਲ T * __restrict__ src,
                        ਅਚਲ int64_t ne0, ਅਚਲ int64_t ne1,
                        ਅਚਲ int64_t ne2, ਅਚਲ int64_t ne3,
                        ਅਚਲ int64_t ਕੁੱਲ_elements,
                        ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ) {
    ਅਚਲ int64_t i = ਵਸਤੂ.get_global_id(0);
    ਜੇ (i >= ਕੁੱਲ_elements) {
        ਵਾਪਸ;
    }

    ਅਚਲ int64_t i0 = i % ne0;
    ਅਚਲ int64_t i1 = (i / ne0) % ne1;
    ਅਚਲ int64_t i2 = (i / (ne0 * ne1)) % ne2;
    ਅਚਲ int64_t i3 = i / (ne0 * ne1 * ne2);

    ਅਚਲ int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    ਜੇ (i0 == i1) {
        ਅਚਲ int64_t batch_idx = i3 * ne2 + i2;
        dst[dst_idx] = src[batch_idx * ne0 + i0];
    } ਨਹੀਂਤਾਂ {
        dst[dst_idx] = T(0);
    }

    (ਖਾਲੀ)ne3;
}

ਸਤਰਅੰਦਰ ਖਾਲੀ ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[1] == 1);

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ਸੈੱਟ_device(ctx.device));

    ਅਚਲ ਖਾਲੀ * src0_d = src0->ਅੰਕੜੇ;
    ਖਾਲੀ * dst_d = dst->ਅੰਕੜੇ;

    ਅਚਲ int64_t ne0 = dst->ne[0];
    ਅਚਲ int64_t ne1 = dst->ne[1];
    ਅਚਲ int64_t ne2 = dst->ne[2];
    ਅਚਲ int64_t ne3 = dst->ne[3];
    ਅਚਲ int64_t n_elems = ggml_nelements(dst);
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (n_elems + SYCL_DIAG_ਖੰਡ_ਮਾਪ - 1) / SYCL_DIAG_ਖੰਡ_ਮਾਪ;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    stream->parallel_for(
        sycl::nd_range<1>(ਅੰਕ_ਖੰਡਾਂ * SYCL_DIAG_ਖੰਡ_ਮਾਪ, SYCL_DIAG_ਖੰਡ_ਮਾਪ),
        [=](sycl::nd_ਵਸਤੂ<1> ਵਸਤੂ) {
            diag_kernel(static_cast<ਦਸ਼ਮਲਵ *>(dst_d),
                        static_cast<ਅਚਲ ਦਸ਼ਮਲਵ *>(src0_d),
                        ne0, ne1, ne2, ne3, n_elems, ਵਸਤੂ);
        });
}

ਖਾਲੀ ggml_sycl_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag(ctx, dst);
}
