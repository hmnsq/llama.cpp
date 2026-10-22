#അന്തര്ഭാവ "count-equal.hpp"

#അന്തര്ഭാവ <cstdint>

template <typename T>
സ്ഥിതം ശൂന്യം എണ്ണം_equal(സ്ഥിരം T *__restrict__ x, സ്ഥിരം T *__restrict__ y,
                        int64_t *__restrict__ dst, സ്ഥിരം int64_t dk,
                        സ്ഥിരം int64_t k) {
    auto item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    സ്ഥിരം int64_t i0 = (int64_t)item_ct1.get_group(2) * dk;
    സ്ഥിരം int64_t i1 = sycl::min(i0 + dk, k);

    പൂർണ്ണസംഖ്യ nequal = 0;

    വേണ്ടി (int64_t i = i0 + item_ct1.get_local_id(2); i < i1; i += WARP_SIZE) {
        സ്ഥിരം T xi = x[i];
        സ്ഥിരം T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_sum<WARP_SIZE>(nequal);

    എങ്കിൽ (item_ct1.get_local_id(2) != 0) {
        തിരികെ;
    }

    dpct::atomic_fetch_add<sycl::access::address_space::generic_space>(
        (പൂർണ്ണസംഖ്യ *)dst, nequal);
}

ശൂന്യം ggml_sycl_എണ്ണം_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    സ്ഥിരം ggml_tensor * src0 = dst->src[0];
    സ്ഥിരം ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->data;

    dpct::queue_ptr stream = ctx.stream();
    സ്ഥിരം പൂർണ്ണസംഖ്യ id       = get_current_device_id();
    സ്ഥിരം പൂർണ്ണസംഖ്യ nsm = ggml_sycl_info().devices[id].nsm;

    സ്ഥിരം int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    സ്ഥിരം int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_എണ്ണം_EQUAL_CHUNK_SIZE);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dst_d, 0, ggml_nbytes(dst))));

    സ്ഥിരം dpct::dim3 block_dims(WARP_SIZE, 1, 1);
    സ്ഥിരം dpct::dim3 block_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_എണ്ണം_EQUAL_CHUNK_SIZE - 1) /
                                       SYCL_എണ്ണം_EQUAL_CHUNK_SIZE),
        1, 1);

    തിരഞ്ഞെടുപ്പ് (src0->type) {
    അവസ്ഥ GGML_TYPE_I32: {
        സ്ഥിരം പൂർണ്ണസംഖ്യ *src0_d = (സ്ഥിരം പൂർണ്ണസംഖ്യ *)src0->data;
        സ്ഥിരം പൂർണ്ണസംഖ്യ *src1_d = (സ്ഥിരം പൂർണ്ണസംഖ്യ *)src1->data;
        stream->parallel_for(
            sycl::nd_range<3>(block_nums * block_dims, block_dims),
            [=](sycl::nd_item<3> item_ct1) {
                എണ്ണം_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_UNUSED(item_ct1);
            });

    } നിർത്തുക;
    സ്ഥിരസ്ഥിതി:
        GGML_ASSERT(അസത്യം);
    }
}
