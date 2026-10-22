#अन्तर्भाव "count-equal.hpp"

#अन्तर्भाव <cstdint>

ढाँचा <typename T>
स्थिर शून्य मोजणी_equal(स्थिरांक T *__restrict__ x, स्थिरांक T *__restrict__ y,
                        int64_t *__restrict__ dst, स्थिरांक int64_t dk,
                        स्थिरांक int64_t k) {
    स्वतः item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिरांक int64_t i0 = (int64_t)item_ct1.get_group(2) * dk;
    स्थिरांक int64_t i1 = sycl::min(i0 + dk, k);

    पूर्णांक nequal = 0;

    साठी (int64_t i = i0 + item_ct1.get_local_id(2); i < i1; i += WARP_SIZE) {
        स्थिरांक T xi = x[i];
        स्थिरांक T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_sum<WARP_SIZE>(nequal);

    जर (item_ct1.get_local_id(2) != 0) {
        परत;
    }

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        (पूर्णांक *)dst, nequal);
}

शून्य ggml_sycl_मोजणी_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->data;

    dpct::queue_ptr stream = ctx.stream();
    स्थिरांक पूर्णांक id       = get_current_device_id();
    स्थिरांक पूर्णांक nsm = ggml_sycl_info().devices[id].nsm;

    स्थिरांक int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    स्थिरांक int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_मोजणी_EQUAL_CHUNK_SIZE);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->स_रखो(dst_d, 0, ggml_nbytes(dst))));

    स्थिरांक dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    स्थिरांक dpct::dim3 block_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_मोजणी_EQUAL_CHUNK_SIZE - 1) /
                                       SYCL_मोजणी_EQUAL_CHUNK_SIZE),
        1, 1);

    निवड (src0->type) {
    स्थिती GGML_TYPE_I32: {
        स्थिरांक पूर्णांक *src0_d = (स्थिरांक पूर्णांक *)src0->data;
        स्थिरांक पूर्णांक *src1_d = (स्थिरांक पूर्णांक *)src1->data;
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                मोजणी_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_UNUSED(item_ct1);
            });

    } थांबा;
    पूर्वनिर्धारित:
        GGML_ASSERT(असत्य);
    }
}
