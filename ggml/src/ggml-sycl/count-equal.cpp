#समावेश "count-equal.hpp"

#समावेश <cstdint>

ढाँचा <typename T>
अटल व्योम गिनती_equal(स्थिर T *__restrict__ x, स्थिर T *__restrict__ y,
                        int64_t *__restrict__ dst, स्थिर int64_t dk,
                        स्थिर int64_t k) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिर int64_t i0 = (int64_t)item_ct1.get_group(2) * dk;
    स्थिर int64_t i1 = sycl::min(i0 + dk, k);

    पूर्णांक nequal = 0;

    क्रम (int64_t i = i0 + item_ct1.get_local_id(2); i < i1; i += WARP_आकार) {
        स्थिर T xi = x[i];
        स्थिर T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_योग<WARP_आकार>(nequal);

    अगर (item_ct1.get_local_id(2) != 0) {
        वापस;
    }

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        (पूर्णांक *)dst, nequal);
}

व्योम ggml_sycl_गिनती_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    स्थिर ggml_tensor * src0 = dst->src[0];
    स्थिर ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->तथ्य;

    dpct::queue_ptr stream = ctx.stream();
    स्थिर पूर्णांक id       = get_current_device_id();
    स्थिर पूर्णांक nsm = ggml_sycl_info().devices[id].nsm;

    स्थिर int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    स्थिर int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_गिनती_EQUAL_CHUNK_आकार);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->स_रखो(dst_d, 0, ggml_nbytes(dst))));

    स्थिर dpct::dim3 block_dims(WARP_आकार, 1, 1);
    स्थिर dpct::dim3 block_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_गिनती_EQUAL_CHUNK_आकार - 1) /
                                       SYCL_गिनती_EQUAL_CHUNK_आकार),
        1, 1);

    चयन (src0->type) {
    हाल GGML_TYPE_I32: {
        स्थिर पूर्णांक *src0_d = (स्थिर पूर्णांक *)src0->तथ्य;
        स्थिर पूर्णांक *src1_d = (स्थिर पूर्णांक *)src1->तथ्य;
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                गिनती_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_UNUSED(item_ct1);
            });

    } अवरोध;
    शेष:
        GGML_ASSERT(false);
    }
}
