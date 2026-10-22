#ଅନ୍ତର୍ଭାଵ "count-equal.hpp"

#ଅନ୍ତର୍ଭାଵ <cstdint>

template <typename T>
ସ୍ଥିର ଶୂନ୍ୟ ଗଣନା_equal(ଅଚଳ T *__restrict__ x, ଅଚଳ T *__restrict__ y,
                        int64_t *__restrict__ dst, ଅଚଳ int64_t dk,
                        ଅଚଳ int64_t k) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ଅଚଳ int64_t i0 = (int64_t)item_ct1.get_group(2) * dk;
    ଅଚଳ int64_t i1 = sycl::min(i0 + dk, k);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା nequal = 0;

    ପାଇଁ (int64_t i = i0 + item_ct1.get_local_id(2); i < i1; i += WARP_SIZE) {
        ଅଚଳ T xi = x[i];
        ଅଚଳ T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_sum<WARP_SIZE>(nequal);

    ଯଦି (item_ct1.get_local_id(2) != 0) {
        ଫେରନ୍ତୁ;
    }

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        (ପୂର୍ଣ୍ଣସଂଖ୍ୟା *)dst, nequal);
}

ଶୂନ୍ୟ ggml_sycl_ଗଣନା_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ଅଚଳ ggml_tensor * src0 = dst->src[0];
    ଅଚଳ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->data;

    dpct::queue_ptr stream = ctx.stream();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା id       = get_current_device_id();
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା nsm = ggml_sycl_info().devices[id].nsm;

    ଅଚଳ int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    ଅଚଳ int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_ଗଣନା_EQUAL_CHUNK_SIZE);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dst_d, 0, ggml_nbytes(dst))));

    ଅଚଳ dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    ଅଚଳ dpct::dim3 block_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_ଗଣନା_EQUAL_CHUNK_SIZE - 1) /
                                       SYCL_ଗଣନା_EQUAL_CHUNK_SIZE),
        1, 1);

    ବାଛ (src0->type) {
    ସ୍ଥିତି GGML_TYPE_I32: {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା *src0_d = (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା *)src0->data;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା *src1_d = (ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା *)src1->data;
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                ଗଣନା_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_UNUSED(item_ct1);
            });

    } ଭାଙ୍ଗ;
    ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
        GGML_ASSERT(ମିଥ୍ୟା);
    }
}
