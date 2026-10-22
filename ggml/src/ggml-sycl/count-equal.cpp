#ضمن "count-equal.hpp"

#ضمن <cstdint>

کينډۍ <ډول_نوم T>
ساكن فراغ شمېر_equal(ثابت T *__restrict__ x, ثابت T *__restrict__ y,
                        int64_t *__restrict__ dst, ثابت int64_t dk,
                        ثابت int64_t k) {
    خپلکاری توکی_ct1 = sycl::ext::oneapi::this_work_توکی::get_nd_توکی<3>();
    ثابت int64_t i0 = (int64_t)توکی_ct1.get_ډله(2) * dk;
    ثابت int64_t i1 = sycl::min(i0 + dk, k);

    صحيح nequal = 0;

    لكل (int64_t i = i0 + توکی_ct1.get_local_id(2); i < i1; i += WARP_اندازه) {
        ثابت T xi = x[i];
        ثابت T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_مجموعه<WARP_اندازه>(nequal);

    إذا (توکی_ct1.get_local_id(2) != 0) {
        أرجع;
    }

    dpct::atomic_fetch_add<sycl::access::پته_space::generic_space>(
        (صحيح *)dst, nequal);
}

فراغ ggml_sycl_شمېر_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->معلومات;

    dpct::کتار_ptr stream = ctx.stream();
    ثابت صحيح id       = get_current_device_id();
    ثابت صحيح nsm = ggml_sycl_info().devices[id].nsm;

    ثابت int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    ثابت int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_شمېر_EQUAL_CHUNK_اندازه);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dst_d, 0, ggml_nbytes(dst))));

    ثابت dpct::dim3 بلاک_dims(WARP_اندازه, 1, 1);
    ثابت dpct::dim3 بلاک_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_شمېر_EQUAL_CHUNK_اندازه - 1) /
                                       SYCL_شمېر_EQUAL_CHUNK_اندازه),
        1, 1);

    بدل (src0->type) {
    حالة GGML_TYPE_I32: {
        ثابت صحيح *src0_d = (ثابت صحيح *)src0->معلومات;
        ثابت صحيح *src1_d = (ثابت صحيح *)src1->معلومات;
        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_توکی<3> توکی_ct1) {
                شمېر_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_ناکارول_شوی(توکی_ct1);
            });

    } اكسر;
    افتراضي:
        GGML_ASSERT(خطأ);
    }
}
