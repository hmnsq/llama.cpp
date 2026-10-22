#અન્તર્ભાવ "count-equal.hpp"

#અન્તર્ભાવ <cstdint>

template <typename T>
સ્થિર શૂન્ય ગણતરી_equal(અચલ T *__restrict__ x, અચલ T *__restrict__ y,
                        int64_t *__restrict__ dst, અચલ int64_t dk,
                        અચલ int64_t k) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    અચલ int64_t i0 = (int64_t)item_ct1.get_group(2) * dk;
    અચલ int64_t i1 = sycl::min(i0 + dk, k);

    પૂર્ણાંક nequal = 0;

    માટે (int64_t i = i0 + item_ct1.get_local_id(2); i < i1; i += WARP_SIZE) {
        અચલ T xi = x[i];
        અચલ T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_sum<WARP_SIZE>(nequal);

    જો (item_ct1.get_local_id(2) != 0) {
        પાછા;
    }

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        (પૂર્ણાંક *)dst, nequal);
}

શૂન્ય ggml_sycl_ગણતરી_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->data;

    dpct::queue_ptr stream = ctx.stream();
    અચલ પૂર્ણાંક id       = get_current_device_id();
    અચલ પૂર્ણાંક nsm = ggml_sycl_info().devices[id].nsm;

    અચલ int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    અચલ int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_ગણતરી_EQUAL_CHUNK_SIZE);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dst_d, 0, ggml_nbytes(dst))));

    અચલ dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    અચલ dpct::dim3 block_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_ગણતરી_EQUAL_CHUNK_SIZE - 1) /
                                       SYCL_ગણતરી_EQUAL_CHUNK_SIZE),
        1, 1);

    પસંદગી (src0->type) {
    સ્થિતિ GGML_TYPE_I32: {
        અચલ પૂર્ણાંક *src0_d = (અચલ પૂર્ણાંક *)src0->data;
        અચલ પૂર્ણાંક *src1_d = (અચલ પૂર્ણાંક *)src1->data;
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                ગણતરી_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_UNUSED(item_ct1);
            });

    } તોડો;
    મૂળભૂત:
        GGML_ASSERT(ખોટું);
    }
}
