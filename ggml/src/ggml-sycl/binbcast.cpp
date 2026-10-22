#ضمن "binbcast.hpp"

#ضمن <algorithm>
#ضمن <cstddef>
#ضمن <cstdint>
#ضمن <sycl/sycl.hpp>

#ضمن "ggml.h"

template<عائم (*bin_op)(ثابت عائم, ثابت عائم), typename src0_t, typename src1_t, typename dst_t>
ساكن فراغ k_bin_bcast(ثابت src0_t * src0, ثابت src1_t * src1, dst_t * dst,
        صحيح ne0, صحيح ne1, صحيح ne2, صحيح ne3,
        صحيح ne10, صحيح ne11, صحيح ne12, صحيح ne13,
        /*int s0, */ صحيح s1,  صحيح s2,  صحيح s3,
        صحيح s00, صحيح s01, صحيح s02, صحيح s03,
        صحيح s10, صحيح s11, صحيح s12, صحيح s13,
        ثابت sycl::nd_item<3> &item_ct1) {
    ثابت صحيح i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    ثابت صحيح i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    ثابت صحيح i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    ثابت صحيح i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    إذا (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        أرجع;
    }

    ثابت صحيح i11 = i1 % ne11;
    ثابت صحيح i12 = i2 % ne12;
    ثابت صحيح i13 = i3 % ne13;

    ثابت size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    ثابت size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    ثابت size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    ثابت src0_t * src0_row = src0 + i_src0;
    ثابت src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    لكل (صحيح i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        ثابت صحيح i10 = i0 % ne10;
        dst_row[i0] = (dst_t)bin_op(src0 ? (عائم)src0_row[i0*s00] : 0.0f, (عائم)src1_row[i10*s10]);
    }
}

template<عائم (*bin_op)(ثابت عائم, ثابت عائم), typename src0_t, typename src1_t, typename dst_t>
ساكن فراغ k_bin_bcast_unravel(ثابت src0_t * src0, ثابت src1_t * src1, dst_t * dst,
        صحيح ne0, صحيح ne1, صحيح ne2, صحيح ne3,
        صحيح ne10, صحيح ne11, صحيح ne12, صحيح ne13,
        /*int s0, */ صحيح s1,  صحيح s2,  صحيح s3,
        صحيح s00, صحيح s01, صحيح s02, صحيح s03,
        صحيح s10, صحيح s11, صحيح s12, صحيح s13,
        ثابت sycl::nd_item<3> &item_ct1) {

    ثابت صحيح i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    ثابت صحيح i3 = i/(ne2*ne1*ne0);
    ثابت صحيح i2 = (i/(ne1*ne0)) % ne2;
    ثابت صحيح i1 = (i/ne0) % ne1;
    ثابت صحيح i0 = i % ne0;

    إذا (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        أرجع;
    }

    ثابت صحيح i11 = i1 % ne11;
    ثابت صحيح i12 = i2 % ne12;
    ثابت صحيح i13 = i3 % ne13;

    ثابت size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    ثابت size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    ثابت size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    ثابت src0_t * src0_row = src0 + i_src0;
    ثابت src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    ثابت صحيح i10 = i0 % ne10;
    dst_row[i0] = (dst_t)bin_op(src0 ? (عائم)src0_row[i0*s00] : 0.0f, (عائم)src1_row[i10*s10]);
}


template<عائم (*bin_op)(ثابت عائم, ثابت عائم)>
بنية bin_bcast_sycl {
    template <typename src0_t, typename src1_t, typename dst_t>
    فراغ operator()(ثابت src0_t * src0_dd, ثابت src1_t * src1_dd, dst_t * dst_dd, ثابت int64_t ne00,
                    ثابت int64_t ne01, ثابت int64_t ne02, ثابت int64_t ne03, ثابت int64_t ne10, ثابت int64_t ne11,
                    ثابت int64_t ne12, ثابت int64_t ne13, ثابت int64_t ne0, ثابت int64_t ne1, ثابت int64_t ne2,
                    ثابت int64_t ne3, ثابت size_t nb00, ثابت size_t nb01, ثابت size_t nb02, ثابت size_t nb03,
                    ثابت size_t nb10, ثابت size_t nb11, ثابت size_t nb12, ثابت size_t nb13, ثابت size_t nb0,
                    ثابت size_t nb1, ثابت size_t nb2, ثابت size_t nb3, ثابت منطقي src0_is_contiguous,
                    ثابت منطقي src1_is_contiguous, ثابت منطقي src0_is_permuted, ثابت منطقي src1_is_permuted,
                    queue_ptr stream) {
        صحيح nr0 = ne10 / ne0;
        صحيح nr1 = ne11/ne1;
        صحيح nr2 = ne12/ne2;
        صحيح nr3 = ne13/ne3;

        صحيح nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne[] = {ne0, ne1, ne2, ne3};
        int64_t cne0[] = {ne00, ne01, ne02, ne03};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};
        size_t cnb[] = {nb0, nb1, nb2, nb3};
        size_t cnb0[] = {nb00, nb01, nb02, nb03};
        size_t cnb1[] = {nb10, nb11, nb12, nb13};
        auto collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        auto collapse_nb = [](size_t cnb[], int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        إذا (src0_is_contiguous && src1_is_contiguous && !src0_is_permuted && !src1_is_permuted) {
            لكل (صحيح i = 0; i < 4; i++) {
                إذا (nr[i] != 1) {
                    اكسر;
                }
                إذا (i > 0) {
                    collapse_nb(cnb, cne);
                    collapse_nb(cnb0, cne0);
                    collapse_nb(cnb1, cne1);
                    collapse(cne);
                    collapse(cne0);
                    collapse(cne1);
                }
            }
        }
        {
            int64_t ne0 = cne[0];
            int64_t ne1 = cne[1];
            int64_t ne2 = cne[2];
            int64_t ne3 = cne[3];

            int64_t ne10 = cne1[0];
            int64_t ne11 = cne1[1];
            int64_t ne12 = cne1[2];
            int64_t ne13 = cne1[3];

            size_t nb0 = cnb[0];
            size_t nb1 = cnb[1];
            size_t nb2 = cnb[2];
            size_t nb3 = cnb[3];

            size_t nb00 = cnb0[0];
            size_t nb01 = cnb0[1];
            size_t nb02 = cnb0[2];
            size_t nb03 = cnb0[3];

            size_t nb10 = cnb1[0];
            size_t nb11 = cnb1[1];
            size_t nb12 = cnb1[2];
            size_t nb13 = cnb1[3];

            // size_t s0 = nb0 / sizeof(dst_t);
            size_t s1 = nb1 / حجم(dst_t);
            size_t s2 = nb2 / حجم(dst_t);
            size_t s3 = nb3 / حجم(dst_t);

            size_t s10 = nb10 / حجم(src1_t);
            size_t s11 = nb11 / حجم(src1_t);
            size_t s12 = nb12 / حجم(src1_t);
            size_t s13 = nb13 / حجم(src1_t);

            size_t s00 = nb00 / حجم(src0_t);
            size_t s01 = nb01 / حجم(src0_t);
            size_t s02 = nb02 / حجم(src0_t);
            size_t s03 = nb03 / حجم(src0_t);

            GGML_UNUSED(s00);

            GGML_ASSERT(nb0 % حجم(dst_t) == 0);
            GGML_ASSERT(nb1 % حجم(dst_t) == 0);
            GGML_ASSERT(nb2 % حجم(dst_t) == 0);
            GGML_ASSERT(nb3 % حجم(dst_t) == 0);

            GGML_ASSERT(nb00 % حجم(src0_t) == 0);
            GGML_ASSERT(nb01 % حجم(src0_t) == 0);
            GGML_ASSERT(nb02 % حجم(src0_t) == 0);
            GGML_ASSERT(nb03 % حجم(src0_t) == 0);

            GGML_ASSERT(nb10 % حجم(src1_t) == 0);
            GGML_ASSERT(nb11 % حجم(src1_t) == 0);
            GGML_ASSERT(nb12 % حجم(src1_t) == 0);
            GGML_ASSERT(nb13 % حجم(src1_t) == 0);

            ثابت صحيح block_size = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            sycl::range<3> block_dims(1, 1, 1);
            block_dims[2] = std::min<unsigned صحيح>(hne0, block_size);
            block_dims[1] = std::min<unsigned صحيح>(
                ne1, block_size / (unsigned صحيح)block_dims[2]);
            block_dims[0] = std::min(
                std::min<unsigned صحيح>(
                    ne2 * ne3, block_size / (unsigned صحيح)block_dims[2] /
                                   (unsigned صحيح)block_dims[1]),
                64U);

            sycl::range<3> block_nums(
                (ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                (ne1 + block_dims[1] - 1) / block_dims[1],
                (hne0 + block_dims[2] - 1) / block_dims[2]);

            إذا (block_nums[0] > 65535) {
                // this is the maximum number of blocks in z direction, fallback to 1D grid kernel
                صحيح block_لمب = (ne0*ne1*ne2*ne3 + block_size - 1) / block_size;
                {
                    dpct::has_capability_or_fail(stream->get_device(),
                                                 {sycl::aspect::fp16});

                    stream->parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, block_لمب) *
                                              sycl::range<3>(1, 1, block_size),
                                          sycl::range<3>(1, 1, block_size)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_bin_bcast_unravel<bin_op>(
                                src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3,
                                ne10, ne11, ne12, ne13, s1, s2, s3, s00, s01, s02,
                                s03, s10, s11, s12, s13, item_ct1);
                        });
                }
            } وإلا {
                /*
                DPCT1049:16: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(block_nums * block_dims, block_dims),
                    [=](sycl::nd_item<3> item_ct1) {
                        k_bin_bcast<bin_op>(src0_dd, src1_dd, dst_dd, ne0, ne1,
                                            ne2, ne3, ne10, ne11, ne12, ne13,
                                            s1, s2, s3, s00, s01, s02, s03, s10, s11, s12, s13,
                                            item_ct1);
                    });
            }
        }
    }
};

template <صنف op>
مضمن فراغ ggml_sycl_op_bin_bcast(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1,
                                   ggml_tensor * dst) {
    dpct::queue_ptr main_stream = ctx.stream();
    GGML_TENSOR_نبيو_OP_LOCALS

    إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()((ثابت عائم *) src0->data, (ثابت عائم *) src1->data, (عائم *) dst->data, ne00, ne01, ne02, ne03, ne10,
             ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3,
             ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()((ثابت sycl::half *) src0->data, (ثابت sycl::half *) src1->data, (sycl::half *) dst->data, ne00, ne01,
             ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13,
             nb0, nb1, nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()((ثابت sycl::half *) src0->data, (ثابت عائم *) src1->data, (sycl::half *) dst->data, ne00, ne01, ne02,
             ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1,
             nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()((ثابت int32_t *) src0->data, (ثابت int32_t *) src1->data, (int32_t *) dst->data, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()((ثابت int16_t *) src0->data, (ثابت int16_t *) src1->data, (int16_t *) dst->data, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        op()((ثابت sycl::ext::oneapi::bfloat16 *) src0->data, (ثابت sycl::ext::oneapi::bfloat16 *) src1->data,
             (sycl::ext::oneapi::bfloat16 *) dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        op()((ثابت sycl::ext::oneapi::bfloat16 *) src0->data, (ثابت عائم *) src1->data,
             (sycl::ext::oneapi::bfloat16 *) dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
#endif
    } وإلا {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(dst->type),
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

مضمن فراغ ggml_sycl_op_add(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_add>>(ctx, dst->src[0], dst->src[1], dst);
}

مضمن فراغ ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_sub>>(ctx, dst->src[0], dst->src[1], dst);
}

مضمن فراغ ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(ctx, dst->src[0], dst->src[1], dst);
}

مضمن فراغ ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(ctx, dst->src[0], dst->src[1], dst);
}

مضمن فراغ ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(ctx, dst, dst->src[0], dst);
}


فراغ ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_بغ scope_dbg_بغ(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_add(ctx, dst);
}

فراغ ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_بغ scope_dbg_بغ(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_sub(ctx, dst);
}

فراغ ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_بغ scope_dbg_بغ(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_mul(ctx, dst);
}

فراغ ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_بغ scope_dbg_بغ(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_div(ctx, dst);
}

فراغ ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_بغ scope_dbg_بغ(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat(ctx, dst);
}

// fused ADD+ADD: dst = (src0 + src1) + src2. Same indexing as k_bin_bcast, so mixed
// types, broadcast, and non-contiguous layouts that add() already handles also fuse.
template<عائم (*bin_op)(ثابت عائم, ثابت عائم), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
ساكن فراغ k_bin_bcast3(ثابت src0_t * src0, ثابت src1_t * src1, ثابت src2_t * src2, dst_t * dst,
        صحيح ne0, صحيح ne1, صحيح ne2, صحيح ne3,
        صحيح ne10, صحيح ne11, صحيح ne12, صحيح ne13,
        صحيح ne20, صحيح ne21, صحيح ne22, صحيح ne23,
        صحيح s1, صحيح s2, صحيح s3,
        صحيح s00, صحيح s01, صحيح s02, صحيح s03,
        صحيح s10, صحيح s11, صحيح s12, صحيح s13,
        صحيح s20, صحيح s21, صحيح s22, صحيح s23,
        ثابت sycl::nd_item<3> & item_ct1) {
    ثابت صحيح i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    ثابت صحيح i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    ثابت صحيح i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    ثابت صحيح i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    إذا (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        أرجع;
    }

    ثابت صحيح i11 = i1 % ne11;
    ثابت صحيح i12 = i2 % ne12;
    ثابت صحيح i13 = i3 % ne13;
    ثابت صحيح i21 = i1 % ne21;
    ثابت صحيح i22 = i2 % ne22;
    ثابت صحيح i23 = i3 % ne23;

    ثابت size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    ثابت size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    ثابت size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    ثابت size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    ثابت src0_t * src0_row = src0 + i_src0;
    ثابت src1_t * src1_row = src1 + i_src1;
    ثابت src2_t * src2_row = src2 + i_src2;
    dst_t *        dst_row  = dst + i_dst;

    لكل (صحيح i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        ثابت صحيح   i10 = i0 % ne10;
        ثابت صحيح   i20 = i0 % ne20;
        ثابت عائم acc = bin_op((عائم) src0_row[i0 * s00], (عائم) src1_row[i10 * s10]);
        dst_row[i0]     = (dst_t) bin_op(acc, (عائم) src2_row[i20 * s20]);
    }
}

template<عائم (*bin_op)(ثابت عائم, ثابت عائم), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
ساكن فراغ k_bin_bcast3_unravel(ثابت src0_t * src0, ثابت src1_t * src1, ثابت src2_t * src2, dst_t * dst,
        صحيح ne0, صحيح ne1, صحيح ne2, صحيح ne3,
        صحيح ne10, صحيح ne11, صحيح ne12, صحيح ne13,
        صحيح ne20, صحيح ne21, صحيح ne22, صحيح ne23,
        صحيح s1, صحيح s2, صحيح s3,
        صحيح s00, صحيح s01, صحيح s02, صحيح s03,
        صحيح s10, صحيح s11, صحيح s12, صحيح s13,
        صحيح s20, صحيح s21, صحيح s22, صحيح s23,
        ثابت sycl::nd_item<3> & item_ct1) {
    ثابت صحيح i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    ثابت صحيح i3 = i / (ne2 * ne1 * ne0);
    ثابت صحيح i2 = (i / (ne1 * ne0)) % ne2;
    ثابت صحيح i1 = (i / ne0) % ne1;
    ثابت صحيح i0 = i % ne0;

    إذا (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        أرجع;
    }

    ثابت صحيح i11 = i1 % ne11;
    ثابت صحيح i12 = i2 % ne12;
    ثابت صحيح i13 = i3 % ne13;
    ثابت صحيح i21 = i1 % ne21;
    ثابت صحيح i22 = i2 % ne22;
    ثابت صحيح i23 = i3 % ne23;

    ثابت size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    ثابت size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    ثابت size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    ثابت size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    ثابت صحيح   i10 = i0 % ne10;
    ثابت صحيح   i20 = i0 % ne20;
    ثابت عائم acc = bin_op((عائم) src0[i_src0 + i0 * s00], (عائم) src1[i_src1 + i10 * s10]);
    dst[i_dst + i0] = (dst_t) bin_op(acc, (عائم) src2[i_src2 + i20 * s20]);
}

template<عائم (*bin_op)(ثابت عائم, ثابت عائم), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
ساكن فراغ launch_bin_bcast3(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0, ثابت ggml_tensor * src1,
                              ثابت ggml_tensor * src2, ggml_tensor * dst) {
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_TENSOR_TERNARY_OP_LOCALS

    صحيح nr1[4] = { (صحيح) (ne10 / ne0), (صحيح) (ne11 / ne1), (صحيح) (ne12 / ne2), (صحيح) (ne13 / ne3) };
    صحيح nr2[4] = { (صحيح) (ne20 / ne0), (صحيح) (ne21 / ne1), (صحيح) (ne22 / ne2), (صحيح) (ne23 / ne3) };

    int64_t cne[]  = { ne0, ne1, ne2, ne3 };
    int64_t cne0[] = { ne00, ne01, ne02, ne03 };
    int64_t cne1[] = { ne10, ne11, ne12, ne13 };
    int64_t cne2[] = { ne20, ne21, ne22, ne23 };
    size_t  cnb[]  = { nb0, nb1, nb2, nb3 };
    size_t  cnb0[] = { nb00, nb01, nb02, nb03 };
    size_t  cnb1[] = { nb10, nb11, nb12, nb13 };
    size_t  cnb2[] = { nb20, nb21, nb22, nb23 };

    auto collapse = [](int64_t cne[]) {
        cne[0] *= cne[1];
        cne[1] = cne[2];
        cne[2] = cne[3];
        cne[3] = 1;
    };

    auto collapse_nb = [](size_t cnb[], int64_t cne[]) {
        cnb[1] *= cne[1];
        cnb[2] *= cne[2];
        cnb[3] *= cne[3];
    };

    ثابت منطقي can_collapse = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(src2) &&
                              !ggml_is_permuted(src0) && !ggml_is_permuted(src1) && !ggml_is_permuted(src2);
    إذا (can_collapse) {
        لكل (صحيح i = 0; i < 4; i++) {
            إذا (nr1[i] != 1 || nr2[i] != 1) {
                اكسر;
            }
            إذا (i > 0) {
                collapse_nb(cnb, cne);
                collapse_nb(cnb0, cne0);
                collapse_nb(cnb1, cne1);
                collapse_nb(cnb2, cne2);
                collapse(cne);
                collapse(cne0);
                collapse(cne1);
                collapse(cne2);
            }
        }
    }

    {
        int64_t ne0 = cne[0];
        int64_t ne1 = cne[1];
        int64_t ne2 = cne[2];
        int64_t ne3 = cne[3];

        int64_t ne10 = cne1[0];
        int64_t ne11 = cne1[1];
        int64_t ne12 = cne1[2];
        int64_t ne13 = cne1[3];

        int64_t ne20 = cne2[0];
        int64_t ne21 = cne2[1];
        int64_t ne22 = cne2[2];
        int64_t ne23 = cne2[3];

        size_t s1 = cnb[1] / حجم(dst_t);
        size_t s2 = cnb[2] / حجم(dst_t);
        size_t s3 = cnb[3] / حجم(dst_t);

        size_t s00 = cnb0[0] / حجم(src0_t);
        size_t s01 = cnb0[1] / حجم(src0_t);
        size_t s02 = cnb0[2] / حجم(src0_t);
        size_t s03 = cnb0[3] / حجم(src0_t);

        size_t s10 = cnb1[0] / حجم(src1_t);
        size_t s11 = cnb1[1] / حجم(src1_t);
        size_t s12 = cnb1[2] / حجم(src1_t);
        size_t s13 = cnb1[3] / حجم(src1_t);

        size_t s20 = cnb2[0] / حجم(src2_t);
        size_t s21 = cnb2[1] / حجم(src2_t);
        size_t s22 = cnb2[2] / حجم(src2_t);
        size_t s23 = cnb2[3] / حجم(src2_t);

        GGML_ASSERT(cnb[0] % حجم(dst_t) == 0 && cnb[1] % حجم(dst_t) == 0 && cnb[2] % حجم(dst_t) == 0 &&
                    cnb[3] % حجم(dst_t) == 0);
        GGML_ASSERT(cnb0[0] % حجم(src0_t) == 0 && cnb0[1] % حجم(src0_t) == 0 && cnb0[2] % حجم(src0_t) == 0 &&
                    cnb0[3] % حجم(src0_t) == 0);
        GGML_ASSERT(cnb1[0] % حجم(src1_t) == 0 && cnb1[1] % حجم(src1_t) == 0 && cnb1[2] % حجم(src1_t) == 0 &&
                    cnb1[3] % حجم(src1_t) == 0);
        GGML_ASSERT(cnb2[0] % حجم(src2_t) == 0 && cnb2[1] % حجم(src2_t) == 0 && cnb2[2] % حجم(src2_t) == 0 &&
                    cnb2[3] % حجم(src2_t) == 0);

        ثابت src0_t * src0_dd = (ثابت src0_t *) src0->data;
        ثابت src1_t * src1_dd = (ثابت src1_t *) src1->data;
        ثابت src2_t * src2_dd = (ثابت src2_t *) src2->data;
        dst_t *        dst_dd  = (dst_t *) dst->data;

        ثابت صحيح block_size = 128;
        int64_t   hne0       = std::max(ne0 / 2LL, 1LL);

        sycl::range<3> block_dims(1, 1, 1);
        block_dims[2] = std::min<unsigned صحيح>(hne0, block_size);
        block_dims[1] = std::min<unsigned صحيح>(ne1, block_size / (unsigned صحيح) block_dims[2]);
        block_dims[0] = std::min(std::min<unsigned صحيح>(ne2 * ne3,
                                                        block_size / (unsigned صحيح) block_dims[2] /
                                                            (unsigned صحيح) block_dims[1]),
                                 64U);

        sycl::range<3> block_nums((ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                                  (ne1 + block_dims[1] - 1) / block_dims[1],
                                  (hne0 + block_dims[2] - 1) / block_dims[2]);

        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        إذا (block_nums[0] > 65535) {
            صحيح block_لمب = (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;
            stream->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, block_لمب) * sycl::range<3>(1, 1, block_size),
                                  sycl::range<3>(1, 1, block_size)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_bin_bcast3_unravel<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10, ne11,
                                                 ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00, s01, s02, s03,
                                                 s10, s11, s12, s13, s20, s21, s22, s23, item_ct1);
                });
        } وإلا {
            stream->parallel_for(sycl::nd_range<3>(block_nums * block_dims, block_dims),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     k_bin_bcast3<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10,
                                                          ne11, ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00,
                                                          s01, s02, s03, s10, s11, s12, s13, s20, s21, s22, s23,
                                                          item_ct1);
                                 });
        }
    }
}

فراغ ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1) {
    ثابت ggml_tensor * src0 = add0->src[0];
    ثابت ggml_tensor * src1 = add0->src[1];
    ثابت ggml_tensor * src2 = add1->src[1];
    ggml_tensor *       dst  = add1;

    GGML_ASSERT(add1->src[0] == add0);
    GGML_ASSERT(ggml_sycl_add_kernel_supports(src0->type, src1->type, add0->type));
    GGML_ASSERT(ggml_sycl_add_kernel_supports(add0->type, src2->type, dst->type));

    إذا (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        launch_bin_bcast3<op_add, عائم, عائم, عائم, عائم>(ctx, src0, src1, src2, dst);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, sycl::half, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, عائم, عائم, sycl::half>(ctx, src0, src1, src2, dst);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, sycl::half, عائم, sycl::half>(ctx, src0, src1, src2, dst);
    } وإلا إذا (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, عائم, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } وإلا إذا (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && src2->type == GGML_TYPE_I32 &&
               dst->type == GGML_TYPE_I32) {
        launch_bin_bcast3<op_add, int32_t, int32_t, int32_t, int32_t>(ctx, src0, src1, src2, dst);
    } وإلا إذا (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && src2->type == GGML_TYPE_I16 &&
               dst->type == GGML_TYPE_I16) {
        launch_bin_bcast3<op_add, int16_t, int16_t, int16_t, int16_t>(ctx, src0, src1, src2, dst);
#ifdef GGML_SYCL_HAS_BF16
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, عائم, عائم, sycl::ext::oneapi::bfloat16>(
            ctx, src0, src1, src2, dst);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, عائم,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } وإلا إذا (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, عائم, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
#endif
    } وإلا {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s, src2: %s\n", __func__,
                ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type),
                ggml_type_name(src2->type));
        GGML_ABORT("fatal error");
    }
}

