#अन्तर्भाव "count-equal.hpp"

#अन्तर्भाव <cstdint>

ढाँचा <typename T>
स्थिर शून्य गन्ती_equal(अचर T *__restrict__ x, अचर T *__restrict__ y,
                        int64_t *__restrict__ dst, अचर int64_t dk,
                        अचर int64_t k) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर int64_t i0 = (int64_t)item_ct1.get_group(2) * dk;
    अचर int64_t i1 = sycl::min(i0 + dk, k);

    पूर्णाङ्क nequal = 0;

    लागि (int64_t i = i0 + item_ct1.get_local_id(2); i < i1; i += WARP_SIZE) {
        अचर T xi = x[i];
        अचर T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_sum<WARP_SIZE>(nequal);

    यदि (item_ct1.get_local_id(2) != 0) {
        फर्काऊ;
    }

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        (पूर्णाङ्क *)dst, nequal);
}

शून्य ggml_sycl_गन्ती_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    अचर ggml_tensor * src0 = dst->src[0];
    अचर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->data;

    dpct::queue_ptr stream = ctx.stream();
    अचर पूर्णाङ्क id       = get_current_device_id();
    अचर पूर्णाङ्क nsm = ggml_sycl_info().devices[id].nsm;

    अचर int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    अचर int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_गन्ती_EQUAL_CHUNK_SIZE);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->स_रखो(dst_d, 0, ggml_nbytes(dst))));

    अचर dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    अचर dpct::dim3 block_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_गन्ती_EQUAL_CHUNK_SIZE - 1) /
                                       SYCL_गन्ती_EQUAL_CHUNK_SIZE),
        1, 1);

    विकल्प (src0->type) {
    पक्ष GGML_TYPE_I32: {
        अचर पूर्णाङ्क *src0_d = (अचर पूर्णाङ्क *)src0->data;
        अचर पूर्णाङ्क *src1_d = (अचर पूर्णाङ्क *)src1->data;
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                गन्ती_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_UNUSED(item_ct1);
            });

    } रोक;
    स्वभाव:
        GGML_ASSERT(असत्य);
    }
}
