#অন্তর্ভাব "count-equal.hpp"

#অন্তর্ভাব <cstdint>

template <typename T>
স্থির শূন্য গুনতি_equal(ধ্রুব T *__restrict__ x, ধ্রুব T *__restrict__ y,
                        int64_t *__restrict__ dst, ধ্রুব int64_t dk,
                        ধ্রুব int64_t k) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুব int64_t i0 = (int64_t)item_ct1.get_group(2) * dk;
    ধ্রুব int64_t i1 = sycl::min(i0 + dk, k);

    পূর্ণাঙ্ক nequal = 0;

    প্রতি (int64_t i = i0 + item_ct1.get_local_id(2); i < i1; i += WARP_SIZE) {
        ধ্রুব T xi = x[i];
        ধ্রুব T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_sum<WARP_SIZE>(nequal);

    যদি (item_ct1.get_local_id(2) != 0) {
        প্রত্যাগম;
    }

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        (পূর্ণাঙ্ক *)dst, nequal);
}

শূন্য ggml_sycl_গুনতি_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ধ্রুব ggml_tensor * src0 = dst->src[0];
    ধ্রুব ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->data;

    dpct::queue_ptr stream = ctx.stream();
    ধ্রুব পূর্ণাঙ্ক id       = get_current_device_id();
    ধ্রুব পূর্ণাঙ্ক nsm = ggml_sycl_info().devices[id].nsm;

    ধ্রুব int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    ধ্রুব int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_গুনতি_EQUAL_CHUNK_SIZE);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dst_d, 0, ggml_nbytes(dst))));

    ধ্রুব dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    ধ্রুব dpct::dim3 block_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_গুনতি_EQUAL_CHUNK_SIZE - 1) /
                                       SYCL_গুনতি_EQUAL_CHUNK_SIZE),
        1, 1);

    বিকল্প (src0->type) {
    পক্ষ GGML_TYPE_I32: {
        ধ্রুব পূর্ণাঙ্ক *src0_d = (ধ্রুব পূর্ণাঙ্ক *)src0->data;
        ধ্রুব পূর্ণাঙ্ক *src1_d = (ধ্রুব পূর্ণাঙ্ক *)src1->data;
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                গুনতি_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_UNUSED(item_ct1);
            });

    } ভঙ্গ;
    স্বভাব:
        GGML_ASSERT(অসত্য);
    }
}
