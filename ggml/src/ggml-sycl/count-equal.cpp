#ਅਨ੍ਤਰ੍ਭਾਵ "count-equal.hpp"

#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ ਗਿਣਤੀ_equal(ਅਚਲ T *__restrict__ x, ਅਚਲ T *__restrict__ y,
                        int64_t *__restrict__ dst, ਅਚਲ int64_t dk,
                        ਅਚਲ int64_t k) {
    ਸਵੈਚਾਲਿਤ ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ int64_t i0 = (int64_t)ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * dk;
    ਅਚਲ int64_t i1 = sycl::min(i0 + dk, k);

    ਪੂਰਨ nequal = 0;

    ਲਈ (int64_t i = i0 + ਵਸਤੂ_ct1.get_local_id(2); i < i1; i += WARP_ਮਾਪ) {
        ਅਚਲ T xi = x[i];
        ਅਚਲ T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_ਜੋੜ<WARP_ਮਾਪ>(nequal);

    ਜੇ (ਵਸਤੂ_ct1.get_local_id(2) != 0) {
        ਵਾਪਸ;
    }

    dpct::atomic_fetch_add<sycl::access::ਪਤਾ_space::generic_space>(
        (ਪੂਰਨ *)dst, nequal);
}

ਖਾਲੀ ggml_sycl_ਗਿਣਤੀ_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->ਅੰਕੜੇ;

    dpct::ਕਤਾਰ_ptr stream = ctx.stream();
    ਅਚਲ ਪੂਰਨ id       = get_current_device_id();
    ਅਚਲ ਪੂਰਨ nsm = ggml_sycl_info().devices[id].nsm;

    ਅਚਲ int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    ਅਚਲ int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_ਗਿਣਤੀ_EQUAL_CHUNK_ਮਾਪ);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dst_d, 0, ggml_nbytes(dst))));

    ਅਚਲ dpct::dim3 ਖੰਡ_dims(WARP_ਮਾਪ, 1, 1);
    ਅਚਲ dpct::dim3 ਖੰਡ_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_ਗਿਣਤੀ_EQUAL_CHUNK_ਮਾਪ - 1) /
                                       SYCL_ਗਿਣਤੀ_EQUAL_CHUNK_ਮਾਪ),
        1, 1);

    ਚੋਣ (src0->type) {
    ਹਾਲਤ GGML_TYPE_I32: {
        ਅਚਲ ਪੂਰਨ *src0_d = (ਅਚਲ ਪੂਰਨ *)src0->ਅੰਕੜੇ;
        ਅਚਲ ਪੂਰਨ *src1_d = (ਅਚਲ ਪੂਰਨ *)src1->ਅੰਕੜੇ;
        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                ਗਿਣਤੀ_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
            });

    } ਰੋਕੋ;
    ਮੂਲ:
        GGML_ASSERT(ਝੂਠ);
    }
}
