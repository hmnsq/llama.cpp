#ಅನ್ತರ್ಭಾವ "diag.hpp"
#ಅನ್ತರ್ಭಾವ "common.hpp"

#ವ್ಯಾಖ್ಯಾನಿಸು SYCL_DIAG_ಖಂಡ_ಗಾತ್ರ 256

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ diag_kernel(T * __restrict__ dst, ಸ್ಥಿರಾಂಕ T * __restrict__ src,
                        ಸ್ಥಿರಾಂಕ int64_t ne0, ಸ್ಥಿರಾಂಕ int64_t ne1,
                        ಸ್ಥಿರಾಂಕ int64_t ne2, ಸ್ಥಿರಾಂಕ int64_t ne3,
                        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elements,
                        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ) {
    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ.get_global_id(0);
    ಆದರೆ (i >= ಒಟ್ಟು_elements) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int64_t i0 = i % ne0;
    ಸ್ಥಿರಾಂಕ int64_t i1 = (i / ne0) % ne1;
    ಸ್ಥಿರಾಂಕ int64_t i2 = (i / (ne0 * ne1)) % ne2;
    ಸ್ಥಿರಾಂಕ int64_t i3 = i / (ne0 * ne1 * ne2);

    ಸ್ಥಿರಾಂಕ int64_t dst_idx = ((i3 * ne2 + i2) * ne1 + i1) * ne0 + i0;

    ಆದರೆ (i0 == i1) {
        ಸ್ಥಿರಾಂಕ int64_t batch_idx = i3 * ne2 + i2;
        dst[dst_idx] = src[batch_idx * ne0 + i0];
    } ಇಲ್ಲದಿದ್ದರೆ {
        dst[dst_idx] = T(0);
    }

    (ಶೂನ್ಯ)ne3;
}

ಸಾಲೊಳಗೆ ಶೂನ್ಯ ggml_sycl_op_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];

    GGML_ASSERT(ggml_is_contiguous(dst));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(src0->ne[1] == 1);

    dpct::ಸರತಿ_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_ಗಣ_device(ctx.device));

    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src0_d = src0->ದತ್ತಾಂಶ;
    ಶೂನ್ಯ * dst_d = dst->ದತ್ತಾಂಶ;

    ಸ್ಥಿರಾಂಕ int64_t ne0 = dst->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t ne1 = dst->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne2 = dst->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne3 = dst->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t n_elems = ggml_nelements(dst);
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (n_elems + SYCL_DIAG_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_DIAG_ಖಂಡ_ಗಾತ್ರ;

    GGML_ASSERT(dst->type == GGML_TYPE_F32);
    stream->parallel_for(
        sycl::nd_range<1>(ಅಂಕೆ_ಖಂಡಗಳು * SYCL_DIAG_ಖಂಡ_ಗಾತ್ರ, SYCL_DIAG_ಖಂಡ_ಗಾತ್ರ),
        [=](sycl::nd_ಅಂಶ<1> ಅಂಶ) {
            diag_kernel(static_cast<ದಶಮಾಂಶ *>(dst_d),
                        static_cast<ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *>(src0_d),
                        ne0, ne1, ne2, ne3, n_elems, ಅಂಶ);
        });
}

ಶೂನ್ಯ ggml_sycl_diag(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_diag(ctx, dst);
}
