#அந்தர்பாவ "count-equal.hpp"

#அந்தர்பாவ <cstdint>

template <typename T>
நிலையான வெறுமை மொத்தம்_equal(மாறிலி T *__restrict__ x, மாறிலி T *__restrict__ y,
                        int64_t *__restrict__ dst, மாறிலி int64_t dk,
                        மாறிலி int64_t k) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    மாறிலி int64_t i0 = (int64_t)item_ct1.get_group(2) * dk;
    மாறிலி int64_t i1 = sycl::min(i0 + dk, k);

    முழு nequal = 0;

    க்கு (int64_t i = i0 + item_ct1.get_local_id(2); i < i1; i += WARP_SIZE) {
        மாறிலி T xi = x[i];
        மாறிலி T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_sum<WARP_SIZE>(nequal);

    எனில் (item_ct1.get_local_id(2) != 0) {
        திரும்பு;
    }

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        (முழு *)dst, nequal);
}

வெறுமை ggml_sycl_மொத்தம்_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->data;

    dpct::queue_ptr stream = ctx.stream();
    மாறிலி முழு id       = get_current_device_id();
    மாறிலி முழு nsm = ggml_sycl_info().devices[id].nsm;

    மாறிலி int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    மாறிலி int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_மொத்தம்_EQUAL_CHUNK_SIZE);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dst_d, 0, ggml_nbytes(dst))));

    மாறிலி dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    மாறிலி dpct::dim3 block_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_மொத்தம்_EQUAL_CHUNK_SIZE - 1) /
                                       SYCL_மொத்தம்_EQUAL_CHUNK_SIZE),
        1, 1);

    தேர்வு (src0->type) {
    நிலை GGML_TYPE_I32: {
        மாறிலி முழு *src0_d = (மாறிலி முழு *)src0->data;
        மாறிலி முழு *src1_d = (மாறிலி முழு *)src1->data;
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                மொத்தம்_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_UNUSED(item_ct1);
            });

    } நிறுத்து;
    இயல்பு:
        GGML_ASSERT(பொய்);
    }
}
