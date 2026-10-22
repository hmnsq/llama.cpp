#अन्तर्भाव "count-equal.hpp"

#अन्तर्भाव <cstdint>

प्रतिरूपम् <प्रकारनाम T>
स्थिर शून्य गणनम्_equal(ध्रुव T *__restrict__ x, ध्रुव T *__restrict__ y,
                        int64_t *__restrict__ dst, ध्रुव int64_t dk,
                        ध्रुव int64_t k) {
    स्वचालित item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ध्रुव int64_t i0 = (int64_t)item_ct1.get_गणः(2) * dk;
    ध्रुव int64_t i1 = sycl::min(i0 + dk, k);

    पूर्णाङ्क nequal = 0;

    प्रति (int64_t i = i0 + item_ct1.get_local_id(2); i < i1; i += WARP_आकार) {
        ध्रुव T xi = x[i];
        ध्रुव T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_योग<WARP_आकार>(nequal);

    यदि (item_ct1.get_local_id(2) != 0) {
        प्रत्यागम;
    }

    dpct::atomic_fetch_add<sycl::access::स्थानाङ्कः_space::generic_space>(
        (पूर्णाङ्क *)dst, nequal);
}

शून्य ggml_sycl_गणनम्_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->दत्त;

    dpct::queue_ptr stream = ctx.stream();
    ध्रुव पूर्णाङ्क id       = get_current_device_id();
    ध्रुव पूर्णाङ्क nsm = ggml_sycl_info().devices[id].nsm;

    ध्रुव int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    ध्रुव int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_गणनम्_EQUAL_CHUNK_आकार);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->स्मृतिं_पूरय(dst_d, 0, ggml_nbytes(dst))));

    ध्रुव dpct::dim3 खण्डम्_dims(WARP_आकार, 1, 1);
    ध्रुव dpct::dim3 खण्डम्_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_गणनम्_EQUAL_CHUNK_आकार - 1) /
                                       SYCL_गणनम्_EQUAL_CHUNK_आकार),
        1, 1);

    विकल्प (src0->type) {
    पक्ष GGML_TYPE_I32: {
        ध्रुव पूर्णाङ्क *src0_d = (ध्रुव पूर्णाङ्क *)src0->दत्त;
        ध्रुव पूर्णाङ्क *src1_d = (ध्रुव पूर्णाङ्क *)src1->दत्त;
        stream->parallel_for(
            sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
            [=](sycl::nd_item<3> item_ct1) {
                गणनम्_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_अप्रयुक्तम्(item_ct1);
            });

    } भङ्ग;
    स्वभाव:
        GGML_ASSERT(असत्य);
    }
}
