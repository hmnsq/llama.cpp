#அந்தர்பாவ "binbcast.hpp"

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cstddef>
#அந்தர்பாவ <cstdint>
#அந்தர்பாவ <sycl/sycl.hpp>

#அந்தர்பாவ "ggml.h"

template<தசம (*bin_op)(மாறிலி தசம, மாறிலி தசம), typename src0_t, typename src1_t, typename dst_t>
நிலையான வெறுமை k_bin_bcast(மாறிலி src0_t * src0, மாறிலி src1_t * src1, dst_t * dst,
        முழு ne0, முழு ne1, முழு ne2, முழு ne3,
        முழு ne10, முழு ne11, முழு ne12, முழு ne13,
        /*int s0, */ முழு s1,  முழு s2,  முழு s3,
        முழு s00, முழு s01, முழு s02, முழு s03,
        முழு s10, முழு s11, முழு s12, முழு s13,
        மாறிலி sycl::nd_item<3> &item_ct1) {
    மாறிலி முழு i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    மாறிலி முழு i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    மாறிலி முழு i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    மாறிலி முழு i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    எனில் (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        திரும்பு;
    }

    மாறிலி முழு i11 = i1 % ne11;
    மாறிலி முழு i12 = i2 % ne12;
    மாறிலி முழு i13 = i3 % ne13;

    மாறிலி size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    மாறிலி size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    மாறிலி size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    மாறிலி src0_t * src0_row = src0 + i_src0;
    மாறிலி src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    க்கு (முழு i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        மாறிலி முழு i10 = i0 % ne10;
        dst_row[i0] = (dst_t)bin_op(src0 ? (தசம)src0_row[i0*s00] : 0.0f, (தசம)src1_row[i10*s10]);
    }
}

template<தசம (*bin_op)(மாறிலி தசம, மாறிலி தசம), typename src0_t, typename src1_t, typename dst_t>
நிலையான வெறுமை k_bin_bcast_unravel(மாறிலி src0_t * src0, மாறிலி src1_t * src1, dst_t * dst,
        முழு ne0, முழு ne1, முழு ne2, முழு ne3,
        முழு ne10, முழு ne11, முழு ne12, முழு ne13,
        /*int s0, */ முழு s1,  முழு s2,  முழு s3,
        முழு s00, முழு s01, முழு s02, முழு s03,
        முழு s10, முழு s11, முழு s12, முழு s13,
        மாறிலி sycl::nd_item<3> &item_ct1) {

    மாறிலி முழு i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    மாறிலி முழு i3 = i/(ne2*ne1*ne0);
    மாறிலி முழு i2 = (i/(ne1*ne0)) % ne2;
    மாறிலி முழு i1 = (i/ne0) % ne1;
    மாறிலி முழு i0 = i % ne0;

    எனில் (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        திரும்பு;
    }

    மாறிலி முழு i11 = i1 % ne11;
    மாறிலி முழு i12 = i2 % ne12;
    மாறிலி முழு i13 = i3 % ne13;

    மாறிலி size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    மாறிலி size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    மாறிலி size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    மாறிலி src0_t * src0_row = src0 + i_src0;
    மாறிலி src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    மாறிலி முழு i10 = i0 % ne10;
    dst_row[i0] = (dst_t)bin_op(src0 ? (தசம)src0_row[i0*s00] : 0.0f, (தசம)src1_row[i10*s10]);
}


template<தசம (*bin_op)(மாறிலி தசம, மாறிலி தசம)>
கட்டமைப்பு bin_bcast_sycl {
    template <typename src0_t, typename src1_t, typename dst_t>
    வெறுமை operator()(மாறிலி src0_t * src0_dd, மாறிலி src1_t * src1_dd, dst_t * dst_dd, மாறிலி int64_t ne00,
                    மாறிலி int64_t ne01, மாறிலி int64_t ne02, மாறிலி int64_t ne03, மாறிலி int64_t ne10, மாறிலி int64_t ne11,
                    மாறிலி int64_t ne12, மாறிலி int64_t ne13, மாறிலி int64_t ne0, மாறிலி int64_t ne1, மாறிலி int64_t ne2,
                    மாறிலி int64_t ne3, மாறிலி size_t nb00, மாறிலி size_t nb01, மாறிலி size_t nb02, மாறிலி size_t nb03,
                    மாறிலி size_t nb10, மாறிலி size_t nb11, மாறிலி size_t nb12, மாறிலி size_t nb13, மாறிலி size_t nb0,
                    மாறிலி size_t nb1, மாறிலி size_t nb2, மாறிலி size_t nb3, மாறிலி தருக்க src0_is_contiguous,
                    மாறிலி தருக்க src1_is_contiguous, மாறிலி தருக்க src0_is_permuted, மாறிலி தருக்க src1_is_permuted,
                    queue_ptr stream) {
        முழு nr0 = ne10 / ne0;
        முழு nr1 = ne11/ne1;
        முழு nr2 = ne12/ne2;
        முழு nr3 = ne13/ne3;

        முழு nr[4] = { nr0, nr1, nr2, nr3 };

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

        எனில் (src0_is_contiguous && src1_is_contiguous && !src0_is_permuted && !src1_is_permuted) {
            க்கு (முழு i = 0; i < 4; i++) {
                எனில் (nr[i] != 1) {
                    நிறுத்து;
                }
                எனில் (i > 0) {
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
            size_t s1 = nb1 / அளவு(dst_t);
            size_t s2 = nb2 / அளவு(dst_t);
            size_t s3 = nb3 / அளவு(dst_t);

            size_t s10 = nb10 / அளவு(src1_t);
            size_t s11 = nb11 / அளவு(src1_t);
            size_t s12 = nb12 / அளவு(src1_t);
            size_t s13 = nb13 / அளவு(src1_t);

            size_t s00 = nb00 / அளவு(src0_t);
            size_t s01 = nb01 / அளவு(src0_t);
            size_t s02 = nb02 / அளவு(src0_t);
            size_t s03 = nb03 / அளவு(src0_t);

            GGML_UNUSED(s00);

            GGML_ASSERT(nb0 % அளவு(dst_t) == 0);
            GGML_ASSERT(nb1 % அளவு(dst_t) == 0);
            GGML_ASSERT(nb2 % அளவு(dst_t) == 0);
            GGML_ASSERT(nb3 % அளவு(dst_t) == 0);

            GGML_ASSERT(nb00 % அளவு(src0_t) == 0);
            GGML_ASSERT(nb01 % அளவு(src0_t) == 0);
            GGML_ASSERT(nb02 % அளவு(src0_t) == 0);
            GGML_ASSERT(nb03 % அளவு(src0_t) == 0);

            GGML_ASSERT(nb10 % அளவு(src1_t) == 0);
            GGML_ASSERT(nb11 % அளவு(src1_t) == 0);
            GGML_ASSERT(nb12 % அளவு(src1_t) == 0);
            GGML_ASSERT(nb13 % அளவு(src1_t) == 0);

            மாறிலி முழு block_size = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            sycl::range<3> block_dims(1, 1, 1);
            block_dims[2] = std::min<unsigned முழு>(hne0, block_size);
            block_dims[1] = std::min<unsigned முழு>(
                ne1, block_size / (unsigned முழு)block_dims[2]);
            block_dims[0] = std::min(
                std::min<unsigned முழு>(
                    ne2 * ne3, block_size / (unsigned முழு)block_dims[2] /
                                   (unsigned முழு)block_dims[1]),
                64U);

            sycl::range<3> block_nums(
                (ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                (ne1 + block_dims[1] - 1) / block_dims[1],
                (hne0 + block_dims[2] - 1) / block_dims[2]);

            எனில் (block_nums[0] > 65535) {
                // this is the maximum number of blocks in z direction, fallback to 1D grid kernel
                முழு block_இலக்கம் = (ne0*ne1*ne2*ne3 + block_size - 1) / block_size;
                {
                    dpct::has_capability_or_fail(stream->get_device(),
                                                 {sycl::aspect::fp16});

                    stream->parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, block_இலக்கம்) *
                                              sycl::range<3>(1, 1, block_size),
                                          sycl::range<3>(1, 1, block_size)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_bin_bcast_unravel<bin_op>(
                                src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3,
                                ne10, ne11, ne12, ne13, s1, s2, s3, s00, s01, s02,
                                s03, s10, s11, s12, s13, item_ct1);
                        });
                }
            } இல்லையெனில் {
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

template <வகுப்பு op>
உள்வரி வெறுமை ggml_sycl_op_bin_bcast(ggml_backend_sycl_context & ctx, மாறிலி ggml_tensor * src0, மாறிலி ggml_tensor * src1,
                                   ggml_tensor * dst) {
    dpct::queue_ptr main_stream = ctx.stream();
    GGML_TENSOR_இருமம்_OP_LOCALS

    எனில் (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()((மாறிலி தசம *) src0->data, (மாறிலி தசம *) src1->data, (தசம *) dst->data, ne00, ne01, ne02, ne03, ne10,
             ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3,
             ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()((மாறிலி sycl::half *) src0->data, (மாறிலி sycl::half *) src1->data, (sycl::half *) dst->data, ne00, ne01,
             ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13,
             nb0, nb1, nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()((மாறிலி sycl::half *) src0->data, (மாறிலி தசம *) src1->data, (sycl::half *) dst->data, ne00, ne01, ne02,
             ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1,
             nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()((மாறிலி int32_t *) src0->data, (மாறிலி int32_t *) src1->data, (int32_t *) dst->data, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()((மாறிலி int16_t *) src0->data, (மாறிலி int16_t *) src1->data, (int16_t *) dst->data, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        op()((மாறிலி sycl::ext::oneapi::bfloat16 *) src0->data, (மாறிலி sycl::ext::oneapi::bfloat16 *) src1->data,
             (sycl::ext::oneapi::bfloat16 *) dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        op()((மாறிலி sycl::ext::oneapi::bfloat16 *) src0->data, (மாறிலி தசம *) src1->data,
             (sycl::ext::oneapi::bfloat16 *) dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
#endif
    } இல்லையெனில் {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(dst->type),
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

உள்வரி வெறுமை ggml_sycl_op_add(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_add>>(ctx, dst->src[0], dst->src[1], dst);
}

உள்வரி வெறுமை ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_sub>>(ctx, dst->src[0], dst->src[1], dst);
}

உள்வரி வெறுமை ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(ctx, dst->src[0], dst->src[1], dst);
}

உள்வரி வெறுமை ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(ctx, dst->src[0], dst->src[1], dst);
}

உள்வரி வெறுமை ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(ctx, dst, dst->src[0], dst);
}


வெறுமை ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_add(ctx, dst);
}

வெறுமை ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_sub(ctx, dst);
}

வெறுமை ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_mul(ctx, dst);
}

வெறுமை ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_div(ctx, dst);
}

வெறுமை ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat(ctx, dst);
}

// fused ADD+ADD: dst = (src0 + src1) + src2. Same indexing as k_bin_bcast, so mixed
// types, broadcast, and non-contiguous layouts that add() already handles also fuse.
template<தசம (*bin_op)(மாறிலி தசம, மாறிலி தசம), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
நிலையான வெறுமை k_bin_bcast3(மாறிலி src0_t * src0, மாறிலி src1_t * src1, மாறிலி src2_t * src2, dst_t * dst,
        முழு ne0, முழு ne1, முழு ne2, முழு ne3,
        முழு ne10, முழு ne11, முழு ne12, முழு ne13,
        முழு ne20, முழு ne21, முழு ne22, முழு ne23,
        முழு s1, முழு s2, முழு s3,
        முழு s00, முழு s01, முழு s02, முழு s03,
        முழு s10, முழு s11, முழு s12, முழு s13,
        முழு s20, முழு s21, முழு s22, முழு s23,
        மாறிலி sycl::nd_item<3> & item_ct1) {
    மாறிலி முழு i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    மாறிலி முழு i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    மாறிலி முழு i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    மாறிலி முழு i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    எனில் (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        திரும்பு;
    }

    மாறிலி முழு i11 = i1 % ne11;
    மாறிலி முழு i12 = i2 % ne12;
    மாறிலி முழு i13 = i3 % ne13;
    மாறிலி முழு i21 = i1 % ne21;
    மாறிலி முழு i22 = i2 % ne22;
    மாறிலி முழு i23 = i3 % ne23;

    மாறிலி size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    மாறிலி size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    மாறிலி size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    மாறிலி size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    மாறிலி src0_t * src0_row = src0 + i_src0;
    மாறிலி src1_t * src1_row = src1 + i_src1;
    மாறிலி src2_t * src2_row = src2 + i_src2;
    dst_t *        dst_row  = dst + i_dst;

    க்கு (முழு i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        மாறிலி முழு   i10 = i0 % ne10;
        மாறிலி முழு   i20 = i0 % ne20;
        மாறிலி தசம acc = bin_op((தசம) src0_row[i0 * s00], (தசம) src1_row[i10 * s10]);
        dst_row[i0]     = (dst_t) bin_op(acc, (தசம) src2_row[i20 * s20]);
    }
}

template<தசம (*bin_op)(மாறிலி தசம, மாறிலி தசம), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
நிலையான வெறுமை k_bin_bcast3_unravel(மாறிலி src0_t * src0, மாறிலி src1_t * src1, மாறிலி src2_t * src2, dst_t * dst,
        முழு ne0, முழு ne1, முழு ne2, முழு ne3,
        முழு ne10, முழு ne11, முழு ne12, முழு ne13,
        முழு ne20, முழு ne21, முழு ne22, முழு ne23,
        முழு s1, முழு s2, முழு s3,
        முழு s00, முழு s01, முழு s02, முழு s03,
        முழு s10, முழு s11, முழு s12, முழு s13,
        முழு s20, முழு s21, முழு s22, முழு s23,
        மாறிலி sycl::nd_item<3> & item_ct1) {
    மாறிலி முழு i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    மாறிலி முழு i3 = i / (ne2 * ne1 * ne0);
    மாறிலி முழு i2 = (i / (ne1 * ne0)) % ne2;
    மாறிலி முழு i1 = (i / ne0) % ne1;
    மாறிலி முழு i0 = i % ne0;

    எனில் (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        திரும்பு;
    }

    மாறிலி முழு i11 = i1 % ne11;
    மாறிலி முழு i12 = i2 % ne12;
    மாறிலி முழு i13 = i3 % ne13;
    மாறிலி முழு i21 = i1 % ne21;
    மாறிலி முழு i22 = i2 % ne22;
    மாறிலி முழு i23 = i3 % ne23;

    மாறிலி size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    மாறிலி size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    மாறிலி size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    மாறிலி size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    மாறிலி முழு   i10 = i0 % ne10;
    மாறிலி முழு   i20 = i0 % ne20;
    மாறிலி தசம acc = bin_op((தசம) src0[i_src0 + i0 * s00], (தசம) src1[i_src1 + i10 * s10]);
    dst[i_dst + i0] = (dst_t) bin_op(acc, (தசம) src2[i_src2 + i20 * s20]);
}

template<தசம (*bin_op)(மாறிலி தசம, மாறிலி தசம), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
நிலையான வெறுமை launch_bin_bcast3(ggml_backend_sycl_context & ctx, மாறிலி ggml_tensor * src0, மாறிலி ggml_tensor * src1,
                              மாறிலி ggml_tensor * src2, ggml_tensor * dst) {
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_TENSOR_TERNARY_OP_LOCALS

    முழு nr1[4] = { (முழு) (ne10 / ne0), (முழு) (ne11 / ne1), (முழு) (ne12 / ne2), (முழு) (ne13 / ne3) };
    முழு nr2[4] = { (முழு) (ne20 / ne0), (முழு) (ne21 / ne1), (முழு) (ne22 / ne2), (முழு) (ne23 / ne3) };

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

    மாறிலி தருக்க can_collapse = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(src2) &&
                              !ggml_is_permuted(src0) && !ggml_is_permuted(src1) && !ggml_is_permuted(src2);
    எனில் (can_collapse) {
        க்கு (முழு i = 0; i < 4; i++) {
            எனில் (nr1[i] != 1 || nr2[i] != 1) {
                நிறுத்து;
            }
            எனில் (i > 0) {
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

        size_t s1 = cnb[1] / அளவு(dst_t);
        size_t s2 = cnb[2] / அளவு(dst_t);
        size_t s3 = cnb[3] / அளவு(dst_t);

        size_t s00 = cnb0[0] / அளவு(src0_t);
        size_t s01 = cnb0[1] / அளவு(src0_t);
        size_t s02 = cnb0[2] / அளவு(src0_t);
        size_t s03 = cnb0[3] / அளவு(src0_t);

        size_t s10 = cnb1[0] / அளவு(src1_t);
        size_t s11 = cnb1[1] / அளவு(src1_t);
        size_t s12 = cnb1[2] / அளவு(src1_t);
        size_t s13 = cnb1[3] / அளவு(src1_t);

        size_t s20 = cnb2[0] / அளவு(src2_t);
        size_t s21 = cnb2[1] / அளவு(src2_t);
        size_t s22 = cnb2[2] / அளவு(src2_t);
        size_t s23 = cnb2[3] / அளவு(src2_t);

        GGML_ASSERT(cnb[0] % அளவு(dst_t) == 0 && cnb[1] % அளவு(dst_t) == 0 && cnb[2] % அளவு(dst_t) == 0 &&
                    cnb[3] % அளவு(dst_t) == 0);
        GGML_ASSERT(cnb0[0] % அளவு(src0_t) == 0 && cnb0[1] % அளவு(src0_t) == 0 && cnb0[2] % அளவு(src0_t) == 0 &&
                    cnb0[3] % அளவு(src0_t) == 0);
        GGML_ASSERT(cnb1[0] % அளவு(src1_t) == 0 && cnb1[1] % அளவு(src1_t) == 0 && cnb1[2] % அளவு(src1_t) == 0 &&
                    cnb1[3] % அளவு(src1_t) == 0);
        GGML_ASSERT(cnb2[0] % அளவு(src2_t) == 0 && cnb2[1] % அளவு(src2_t) == 0 && cnb2[2] % அளவு(src2_t) == 0 &&
                    cnb2[3] % அளவு(src2_t) == 0);

        மாறிலி src0_t * src0_dd = (மாறிலி src0_t *) src0->data;
        மாறிலி src1_t * src1_dd = (மாறிலி src1_t *) src1->data;
        மாறிலி src2_t * src2_dd = (மாறிலி src2_t *) src2->data;
        dst_t *        dst_dd  = (dst_t *) dst->data;

        மாறிலி முழு block_size = 128;
        int64_t   hne0       = std::max(ne0 / 2LL, 1LL);

        sycl::range<3> block_dims(1, 1, 1);
        block_dims[2] = std::min<unsigned முழு>(hne0, block_size);
        block_dims[1] = std::min<unsigned முழு>(ne1, block_size / (unsigned முழு) block_dims[2]);
        block_dims[0] = std::min(std::min<unsigned முழு>(ne2 * ne3,
                                                        block_size / (unsigned முழு) block_dims[2] /
                                                            (unsigned முழு) block_dims[1]),
                                 64U);

        sycl::range<3> block_nums((ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                                  (ne1 + block_dims[1] - 1) / block_dims[1],
                                  (hne0 + block_dims[2] - 1) / block_dims[2]);

        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        எனில் (block_nums[0] > 65535) {
            முழு block_இலக்கம் = (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;
            stream->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, block_இலக்கம்) * sycl::range<3>(1, 1, block_size),
                                  sycl::range<3>(1, 1, block_size)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_bin_bcast3_unravel<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10, ne11,
                                                 ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00, s01, s02, s03,
                                                 s10, s11, s12, s13, s20, s21, s22, s23, item_ct1);
                });
        } இல்லையெனில் {
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

வெறுமை ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1) {
    மாறிலி ggml_tensor * src0 = add0->src[0];
    மாறிலி ggml_tensor * src1 = add0->src[1];
    மாறிலி ggml_tensor * src2 = add1->src[1];
    ggml_tensor *       dst  = add1;

    GGML_ASSERT(add1->src[0] == add0);
    GGML_ASSERT(ggml_sycl_add_kernel_supports(src0->type, src1->type, add0->type));
    GGML_ASSERT(ggml_sycl_add_kernel_supports(add0->type, src2->type, dst->type));

    எனில் (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        launch_bin_bcast3<op_add, தசம, தசம, தசம, தசம>(ctx, src0, src1, src2, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, sycl::half, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, தசம, தசம, sycl::half>(ctx, src0, src1, src2, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, sycl::half, தசம, sycl::half>(ctx, src0, src1, src2, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, தசம, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && src2->type == GGML_TYPE_I32 &&
               dst->type == GGML_TYPE_I32) {
        launch_bin_bcast3<op_add, int32_t, int32_t, int32_t, int32_t>(ctx, src0, src1, src2, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && src2->type == GGML_TYPE_I16 &&
               dst->type == GGML_TYPE_I16) {
        launch_bin_bcast3<op_add, int16_t, int16_t, int16_t, int16_t>(ctx, src0, src1, src2, dst);
#ifdef GGML_SYCL_HAS_BF16
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, தசம, தசம, sycl::ext::oneapi::bfloat16>(
            ctx, src0, src1, src2, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, தசம,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } இல்லையெனில் எனில் (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, தசம, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
#endif
    } இல்லையெனில் {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s, src2: %s\n", __func__,
                ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type),
                ggml_type_name(src2->type));
        GGML_ABORT("fatal error");
    }
}

