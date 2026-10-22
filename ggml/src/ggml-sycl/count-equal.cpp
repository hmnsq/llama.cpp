#شامل "count-equal.hpp"

#شامل <cstdint>

سانچہ <قسم_نام T>
ساکن خالی گنتی_equal(ثابت T *__restrict__ x, ثابت T *__restrict__ y,
                        int64_t *__restrict__ dst, ثابت int64_t dk,
                        ثابت int64_t k) {
    خودکار شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت int64_t i0 = (int64_t)شے_ct1.get_گروہ(2) * dk;
    ثابت int64_t i1 = sycl::min(i0 + dk, k);

    صحیح nequal = 0;

    برائے (int64_t i = i0 + شے_ct1.get_local_id(2); i < i1; i += WARP_سائز) {
        ثابت T xi = x[i];
        ثابت T yi = y[i];
        nequal += xi == yi;
    }

    nequal = warp_reduce_مجموع<WARP_سائز>(nequal);

    اگر (شے_ct1.get_local_id(2) != 0) {
        واپس;
    }

    dpct::atomic_fetch_add<sycl::access::پتہ_space::generic_space>(
        (صحیح *)dst, nequal);
}

خالی ggml_sycl_گنتی_equal(ggml_backend_sycl_context &ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == src1->type);
    GGML_ASSERT( dst->type == GGML_TYPE_I64);

    GGML_ASSERT(ggml_are_same_shape(src0, src1));
    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    int64_t * dst_d  = (int64_t *) dst->ڈیٹا;

    dpct::قطار_ptr stream = ctx.stream();
    ثابت صحیح id       = get_current_device_id();
    ثابت صحیح nsm = ggml_sycl_info().devices[id].nsm;

    ثابت int64_t ne = ggml_nelements(src0);
    GGML_ASSERT(ne < (1 << 30) && "atomicAdd implementation only supports int");
    ثابت int64_t dne =
        GGML_PAD((ne + 4 * nsm - 1) / (4 * nsm), SYCL_گنتی_EQUAL_CHUNK_سائز);

    SYCL_CHECK(CHECK_TRY_ERROR(stream->memset(dst_d, 0, ggml_nbytes(dst))));

    ثابت dpct::dim3 بلاک_dims(WARP_سائز, 1, 1);
    ثابت dpct::dim3 بلاک_nums(
        std::min((int64_t)4 * nsm, (ne + SYCL_گنتی_EQUAL_CHUNK_سائز - 1) /
                                       SYCL_گنتی_EQUAL_CHUNK_سائز),
        1, 1);

    انتخاب (src0->type) {
    حالت GGML_TYPE_I32: {
        ثابت صحیح *src0_d = (ثابت صحیح *)src0->ڈیٹا;
        ثابت صحیح *src1_d = (ثابت صحیح *)src1->ڈیٹا;
        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) {
                گنتی_equal(src0_d, src1_d, dst_d, dne, ne);
                GGML_غیر_مستعمل(شے_ct1);
            });

    } توڑو;
    طےشدہ:
        GGML_ASSERT(جھوٹ);
    }
}
