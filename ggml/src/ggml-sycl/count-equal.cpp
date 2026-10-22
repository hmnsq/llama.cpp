#ಅನ್ತರ್ಭಾವ "count-equal.hpp"

#ಅನ್ತರ್ಭಾವ <cstdint>

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ ಎಣಿಕೆ_equal(ಸ್ಥಿರಾಂಕ T *__restrict__ x, ಸ್ಥಿರಾಂಕ T *__restrict__ y,
                        int64_t *__restrict__ dst, ಸ್ಥಿರಾಂಕ int64_t dk,
                        ಸ್ಥಿರಾಂಕ int64_t k) {
    ಸ್ವಯಂಚಾಲಿತ ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ int64_t i0 = (int64_t)ಅಂಶ_ct1.get_ಗುಂಪು(2) * dk;
    ಸ್ಥಿರಾಂಕ int64_t i1 = sycl::min(i0 + dk, k);

    ಪೂರ್ಣಾಂಕ nequal = 0;

    ಗಾಗಿ (int64_t i = i0 + ಅಂಶ_ct1.get_local_id(2); i < i1; i += WARP_ಗಾತ್ರ) {
        ಸ್ಥಿರಾಂಕ T xi = x[i];
        ಸ್ಥಿರಾಂಕ T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_ಮೊತ್ತ<WARP_ಗಾತ್ರ>(nequal);

    ಆದರೆ (ಅಂಶ_ct1.get_local_id(2) != 0) {
        ಹಿಂತಿರುಗಿ;
    }

    dpct::atomic_fetch_add<sycl::access::ವಿಳಾಸ_space::generic_space>(
        (ಪೂರ್ಣಾಂಕ *)dst, nequal);
}

ಶೂನ್ಯ ggml_sycl_ಎಣಿಕೆ_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->ದತ್ತಾಂಶ;

    dpct::ಸರತಿ_ptr stream = ctx.stream();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ id       = get_current_device_id();
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nsm = ggml_sycl_info().devices[id].nsm;

    ಸ್ಥಿರಾಂಕ int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    ಸ್ಥಿರಾಂಕ int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_ಎಣಿಕೆ_EQUAL_CHUNK_ಗಾತ್ರ);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dst_d, 0, ggml_nbytes(dst))));

    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_dims(WARP_ಗಾತ್ರ, 1, 1);
    ಸ್ಥಿರಾಂಕ dpct::dim3 ಖಂಡ_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_ಎಣಿಕೆ_EQUAL_CHUNK_ಗಾತ್ರ - 1) /
                                       SYCL_ಎಣಿಕೆ_EQUAL_CHUNK_ಗಾತ್ರ),
        1, 1);

    ಆಯ್ಕೆ (src0->type) {
    ಸ್ಥಿತಿ GGML_TYPE_I32: {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *src0_d = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)src0->ದತ್ತಾಂಶ;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *src1_d = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)src1->ದತ್ತಾಂಶ;
        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                ಎಣಿಕೆ_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_ಬಳಸದ(ಅಂಶ_ct1);
            });

    } ನಿಲ್ಲಿಸು;
    ಪೂರ್ವನಿಯೋಜಿತ:
        GGML_ASSERT(ಸುಳ್ಳು);
    }
}
