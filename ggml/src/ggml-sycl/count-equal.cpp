#ಅನ್ತರ್ಭಾವ "count-equal.hpp"

#ಅನ್ತರ್ಭಾವ <cstdint>

template <typename T>
ಸ್ಥಿರ ಶೂನ್ಯ ಎಣಿಕೆ_equal(ಧ್ರುವ T *__restrict__ x, ಧ್ರುವ T *__restrict__ y,
                        int64_t *__restrict__ dst, ಧ್ರುವ int64_t dk,
                        ಧ್ರುವ int64_t k) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ int64_t i0 = (int64_t)item_ct1.get_group(2) * dk;
    ಧ್ರುವ int64_t i1 = sycl::min(i0 + dk, k);

    ಪೂರ್ಣಾಙ್ಕ nequal = 0;

    ಪ್ರತಿ (int64_t i = i0 + item_ct1.get_local_id(2); i < i1; i += WARP_SIZE) {
        ಧ್ರುವ T xi = x[i];
        ಧ್ರುವ T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_sum<WARP_SIZE>(nequal);

    ಯದಿ (item_ct1.get_local_id(2) != 0) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        (ಪೂರ್ಣಾಙ್ಕ *)dst, nequal);
}

ಶೂನ್ಯ ggml_sycl_ಎಣಿಕೆ_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->data;

    dpct::queue_ptr stream = ctx.stream();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ id       = get_current_device_id();
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ nsm = ggml_sycl_info().devices[id].nsm;

    ಧ್ರುವ int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    ಧ್ರುವ int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_ಎಣಿಕೆ_EQUAL_CHUNK_SIZE);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dst_d, 0, ggml_nbytes(dst))));

    ಧ್ರುವ dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    ಧ್ರುವ dpct::dim3 block_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_ಎಣಿಕೆ_EQUAL_CHUNK_SIZE - 1) /
                                       SYCL_ಎಣಿಕೆ_EQUAL_CHUNK_SIZE),
        1, 1);

    ವಿಕಲ್ಪ (src0->type) {
    ಪಕ್ಷ GGML_TYPE_I32: {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *src0_d = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)src0->data;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *src1_d = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)src1->data;
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                ಎಣಿಕೆ_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_UNUSED(item_ct1);
            });

    } ಭಙ್ಗ;
    ಸ್ವಭಾವ:
        GGML_ASSERT(ಅಸತ್ಯ);
    }
}
