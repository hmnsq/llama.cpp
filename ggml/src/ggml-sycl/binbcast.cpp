#অন্তর্ভাব "binbcast.hpp"

#অন্তর্ভাব <algorithm>
#অন্তর্ভাব <cstddef>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <sycl/sycl.hpp>

#অন্তর্ভাব "ggml.h"

template<ভিন্ন (*bin_op)(ধ্রুব ভিন্ন, ধ্রুব ভিন্ন), typename src0_t, typename src1_t, typename dst_t>
স্থির শূন্য k_bin_bcast(ধ্রুব src0_t * src0, ধ্রুব src1_t * src1, dst_t * dst,
        পূর্ণাঙ্ক ne0, পূর্ণাঙ্ক ne1, পূর্ণাঙ্ক ne2, পূর্ণাঙ্ক ne3,
        পূর্ণাঙ্ক ne10, পূর্ণাঙ্ক ne11, পূর্ণাঙ্ক ne12, পূর্ণাঙ্ক ne13,
        /*int s0, */ পূর্ণাঙ্ক s1,  পূর্ণাঙ্ক s2,  পূর্ণাঙ্ক s3,
        পূর্ণাঙ্ক s00, পূর্ণাঙ্ক s01, পূর্ণাঙ্ক s02, পূর্ণাঙ্ক s03,
        পূর্ণাঙ্ক s10, পূর্ণাঙ্ক s11, পূর্ণাঙ্ক s12, পূর্ণাঙ্ক s13,
        ধ্রুব sycl::nd_item<3> &item_ct1) {
    ধ্রুব পূর্ণাঙ্ক i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    ধ্রুব পূর্ণাঙ্ক i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    ধ্রুব পূর্ণাঙ্ক i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    ধ্রুব পূর্ণাঙ্ক i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    যদি (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        প্রত্যাগম;
    }

    ধ্রুব পূর্ণাঙ্ক i11 = i1 % ne11;
    ধ্রুব পূর্ণাঙ্ক i12 = i2 % ne12;
    ধ্রুব পূর্ণাঙ্ক i13 = i3 % ne13;

    ধ্রুব size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    ধ্রুব size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    ধ্রুব size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    ধ্রুব src0_t * src0_row = src0 + i_src0;
    ধ্রুব src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    প্রতি (পূর্ণাঙ্ক i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        ধ্রুব পূর্ণাঙ্ক i10 = i0 % ne10;
        dst_row[i0] = (dst_t)bin_op(src0 ? (ভিন্ন)src0_row[i0*s00] : 0.0f, (ভিন্ন)src1_row[i10*s10]);
    }
}

template<ভিন্ন (*bin_op)(ধ্রুব ভিন্ন, ধ্রুব ভিন্ন), typename src0_t, typename src1_t, typename dst_t>
স্থির শূন্য k_bin_bcast_unravel(ধ্রুব src0_t * src0, ধ্রুব src1_t * src1, dst_t * dst,
        পূর্ণাঙ্ক ne0, পূর্ণাঙ্ক ne1, পূর্ণাঙ্ক ne2, পূর্ণাঙ্ক ne3,
        পূর্ণাঙ্ক ne10, পূর্ণাঙ্ক ne11, পূর্ণাঙ্ক ne12, পূর্ণাঙ্ক ne13,
        /*int s0, */ পূর্ণাঙ্ক s1,  পূর্ণাঙ্ক s2,  পূর্ণাঙ্ক s3,
        পূর্ণাঙ্ক s00, পূর্ণাঙ্ক s01, পূর্ণাঙ্ক s02, পূর্ণাঙ্ক s03,
        পূর্ণাঙ্ক s10, পূর্ণাঙ্ক s11, পূর্ণাঙ্ক s12, পূর্ণাঙ্ক s13,
        ধ্রুব sycl::nd_item<3> &item_ct1) {

    ধ্রুব পূর্ণাঙ্ক i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    ধ্রুব পূর্ণাঙ্ক i3 = i/(ne2*ne1*ne0);
    ধ্রুব পূর্ণাঙ্ক i2 = (i/(ne1*ne0)) % ne2;
    ধ্রুব পূর্ণাঙ্ক i1 = (i/ne0) % ne1;
    ধ্রুব পূর্ণাঙ্ক i0 = i % ne0;

    যদি (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        প্রত্যাগম;
    }

    ধ্রুব পূর্ণাঙ্ক i11 = i1 % ne11;
    ধ্রুব পূর্ণাঙ্ক i12 = i2 % ne12;
    ধ্রুব পূর্ণাঙ্ক i13 = i3 % ne13;

    ধ্রুব size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    ধ্রুব size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    ধ্রুব size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    ধ্রুব src0_t * src0_row = src0 + i_src0;
    ধ্রুব src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    ধ্রুব পূর্ণাঙ্ক i10 = i0 % ne10;
    dst_row[i0] = (dst_t)bin_op(src0 ? (ভিন্ন)src0_row[i0*s00] : 0.0f, (ভিন্ন)src1_row[i10*s10]);
}


template<ভিন্ন (*bin_op)(ধ্রুব ভিন্ন, ধ্রুব ভিন্ন)>
সংরচনা bin_bcast_sycl {
    template <typename src0_t, typename src1_t, typename dst_t>
    শূন্য operator()(ধ্রুব src0_t * src0_dd, ধ্রুব src1_t * src1_dd, dst_t * dst_dd, ধ্রুব int64_t ne00,
                    ধ্রুব int64_t ne01, ধ্রুব int64_t ne02, ধ্রুব int64_t ne03, ধ্রুব int64_t ne10, ধ্রুব int64_t ne11,
                    ধ্রুব int64_t ne12, ধ্রুব int64_t ne13, ধ্রুব int64_t ne0, ধ্রুব int64_t ne1, ধ্রুব int64_t ne2,
                    ধ্রুব int64_t ne3, ধ্রুব size_t nb00, ধ্রুব size_t nb01, ধ্রুব size_t nb02, ধ্রুব size_t nb03,
                    ধ্রুব size_t nb10, ধ্রুব size_t nb11, ধ্রুব size_t nb12, ধ্রুব size_t nb13, ধ্রুব size_t nb0,
                    ধ্রুব size_t nb1, ধ্রুব size_t nb2, ধ্রুব size_t nb3, ধ্রুব তার্কিক src0_is_contiguous,
                    ধ্রুব তার্কিক src1_is_contiguous, ধ্রুব তার্কিক src0_is_permuted, ধ্রুব তার্কিক src1_is_permuted,
                    queue_ptr stream) {
        পূর্ণাঙ্ক nr0 = ne10 / ne0;
        পূর্ণাঙ্ক nr1 = ne11/ne1;
        পূর্ণাঙ্ক nr2 = ne12/ne2;
        পূর্ণাঙ্ক nr3 = ne13/ne3;

        পূর্ণাঙ্ক nr[4] = { nr0, nr1, nr2, nr3 };

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

        যদি (src0_is_contiguous && src1_is_contiguous && !src0_is_permuted && !src1_is_permuted) {
            প্রতি (পূর্ণাঙ্ক i = 0; i < 4; i++) {
                যদি (nr[i] != 1) {
                    ভঙ্গ;
                }
                যদি (i > 0) {
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
            size_t s1 = nb1 / পরিমাণ(dst_t);
            size_t s2 = nb2 / পরিমাণ(dst_t);
            size_t s3 = nb3 / পরিমাণ(dst_t);

            size_t s10 = nb10 / পরিমাণ(src1_t);
            size_t s11 = nb11 / পরিমাণ(src1_t);
            size_t s12 = nb12 / পরিমাণ(src1_t);
            size_t s13 = nb13 / পরিমাণ(src1_t);

            size_t s00 = nb00 / পরিমাণ(src0_t);
            size_t s01 = nb01 / পরিমাণ(src0_t);
            size_t s02 = nb02 / পরিমাণ(src0_t);
            size_t s03 = nb03 / পরিমাণ(src0_t);

            GGML_UNUSED(s00);

            GGML_ASSERT(nb0 % পরিমাণ(dst_t) == 0);
            GGML_ASSERT(nb1 % পরিমাণ(dst_t) == 0);
            GGML_ASSERT(nb2 % পরিমাণ(dst_t) == 0);
            GGML_ASSERT(nb3 % পরিমাণ(dst_t) == 0);

            GGML_ASSERT(nb00 % পরিমাণ(src0_t) == 0);
            GGML_ASSERT(nb01 % পরিমাণ(src0_t) == 0);
            GGML_ASSERT(nb02 % পরিমাণ(src0_t) == 0);
            GGML_ASSERT(nb03 % পরিমাণ(src0_t) == 0);

            GGML_ASSERT(nb10 % পরিমাণ(src1_t) == 0);
            GGML_ASSERT(nb11 % পরিমাণ(src1_t) == 0);
            GGML_ASSERT(nb12 % পরিমাণ(src1_t) == 0);
            GGML_ASSERT(nb13 % পরিমাণ(src1_t) == 0);

            ধ্রুব পূর্ণাঙ্ক block_size = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            sycl::range<3> block_dims(1, 1, 1);
            block_dims[2] = std::min<unsigned পূর্ণাঙ্ক>(hne0, block_size);
            block_dims[1] = std::min<unsigned পূর্ণাঙ্ক>(
                ne1, block_size / (unsigned পূর্ণাঙ্ক)block_dims[2]);
            block_dims[0] = std::min(
                std::min<unsigned পূর্ণাঙ্ক>(
                    ne2 * ne3, block_size / (unsigned পূর্ণাঙ্ক)block_dims[2] /
                                   (unsigned পূর্ণাঙ্ক)block_dims[1]),
                64U);

            sycl::range<3> block_nums(
                (ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                (ne1 + block_dims[1] - 1) / block_dims[1],
                (hne0 + block_dims[2] - 1) / block_dims[2]);

            যদি (block_nums[0] > 65535) {
                // this is the maximum number of blocks in z direction, fallback to 1D grid kernel
                পূর্ণাঙ্ক block_অঙ্ক = (ne0*ne1*ne2*ne3 + block_size - 1) / block_size;
                {
                    dpct::has_capability_or_fail(stream->get_device(),
                                                 {sycl::aspect::fp16});

                    stream->parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, block_অঙ্ক) *
                                              sycl::range<3>(1, 1, block_size),
                                          sycl::range<3>(1, 1, block_size)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_bin_bcast_unravel<bin_op>(
                                src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3,
                                ne10, ne11, ne12, ne13, s1, s2, s3, s00, s01, s02,
                                s03, s10, s11, s12, s13, item_ct1);
                        });
                }
            } অন্যথা {
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

template <বর্গ op>
অন্তঃস্থ শূন্য ggml_sycl_op_bin_bcast(ggml_backend_sycl_context & ctx, ধ্রুব ggml_tensor * src0, ধ্রুব ggml_tensor * src1,
                                   ggml_tensor * dst) {
    dpct::queue_ptr main_stream = ctx.stream();
    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    যদি (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()((ধ্রুব ভিন্ন *) src0->data, (ধ্রুব ভিন্ন *) src1->data, (ভিন্ন *) dst->data, ne00, ne01, ne02, ne03, ne10,
             ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3,
             ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()((ধ্রুব sycl::half *) src0->data, (ধ্রুব sycl::half *) src1->data, (sycl::half *) dst->data, ne00, ne01,
             ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13,
             nb0, nb1, nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()((ধ্রুব sycl::half *) src0->data, (ধ্রুব ভিন্ন *) src1->data, (sycl::half *) dst->data, ne00, ne01, ne02,
             ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1,
             nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()((ধ্রুব int32_t *) src0->data, (ধ্রুব int32_t *) src1->data, (int32_t *) dst->data, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()((ধ্রুব int16_t *) src0->data, (ধ্রুব int16_t *) src1->data, (int16_t *) dst->data, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
#ifdef GGML_SYCL_HAS_BF16
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        op()((ধ্রুব sycl::ext::oneapi::bfloat16 *) src0->data, (ধ্রুব sycl::ext::oneapi::bfloat16 *) src1->data,
             (sycl::ext::oneapi::bfloat16 *) dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        op()((ধ্রুব sycl::ext::oneapi::bfloat16 *) src0->data, (ধ্রুব ভিন্ন *) src1->data,
             (sycl::ext::oneapi::bfloat16 *) dst->data, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
#endif
    } অন্যথা {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(dst->type),
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

অন্তঃস্থ শূন্য ggml_sycl_op_add(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_add>>(ctx, dst->src[0], dst->src[1], dst);
}

অন্তঃস্থ শূন্য ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_sub>>(ctx, dst->src[0], dst->src[1], dst);
}

অন্তঃস্থ শূন্য ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(ctx, dst->src[0], dst->src[1], dst);
}

অন্তঃস্থ শূন্য ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(ctx, dst->src[0], dst->src[1], dst);
}

অন্তঃস্থ শূন্য ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(ctx, dst, dst->src[0], dst);
}


শূন্য ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_add(ctx, dst);
}

শূন্য ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_sub(ctx, dst);
}

শূন্য ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_mul(ctx, dst);
}

শূন্য ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_div(ctx, dst);
}

শূন্য ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat(ctx, dst);
}

// fused ADD+ADD: dst = (src0 + src1) + src2. Same indexing as k_bin_bcast, so mixed
// types, broadcast, and non-contiguous layouts that add() already handles also fuse.
template<ভিন্ন (*bin_op)(ধ্রুব ভিন্ন, ধ্রুব ভিন্ন), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
স্থির শূন্য k_bin_bcast3(ধ্রুব src0_t * src0, ধ্রুব src1_t * src1, ধ্রুব src2_t * src2, dst_t * dst,
        পূর্ণাঙ্ক ne0, পূর্ণাঙ্ক ne1, পূর্ণাঙ্ক ne2, পূর্ণাঙ্ক ne3,
        পূর্ণাঙ্ক ne10, পূর্ণাঙ্ক ne11, পূর্ণাঙ্ক ne12, পূর্ণাঙ্ক ne13,
        পূর্ণাঙ্ক ne20, পূর্ণাঙ্ক ne21, পূর্ণাঙ্ক ne22, পূর্ণাঙ্ক ne23,
        পূর্ণাঙ্ক s1, পূর্ণাঙ্ক s2, পূর্ণাঙ্ক s3,
        পূর্ণাঙ্ক s00, পূর্ণাঙ্ক s01, পূর্ণাঙ্ক s02, পূর্ণাঙ্ক s03,
        পূর্ণাঙ্ক s10, পূর্ণাঙ্ক s11, পূর্ণাঙ্ক s12, পূর্ণাঙ্ক s13,
        পূর্ণাঙ্ক s20, পূর্ণাঙ্ক s21, পূর্ণাঙ্ক s22, পূর্ণাঙ্ক s23,
        ধ্রুব sycl::nd_item<3> & item_ct1) {
    ধ্রুব পূর্ণাঙ্ক i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    ধ্রুব পূর্ণাঙ্ক i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    ধ্রুব পূর্ণাঙ্ক i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    ধ্রুব পূর্ণাঙ্ক i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    যদি (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        প্রত্যাগম;
    }

    ধ্রুব পূর্ণাঙ্ক i11 = i1 % ne11;
    ধ্রুব পূর্ণাঙ্ক i12 = i2 % ne12;
    ধ্রুব পূর্ণাঙ্ক i13 = i3 % ne13;
    ধ্রুব পূর্ণাঙ্ক i21 = i1 % ne21;
    ধ্রুব পূর্ণাঙ্ক i22 = i2 % ne22;
    ধ্রুব পূর্ণাঙ্ক i23 = i3 % ne23;

    ধ্রুব size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    ধ্রুব size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    ধ্রুব size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    ধ্রুব size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    ধ্রুব src0_t * src0_row = src0 + i_src0;
    ধ্রুব src1_t * src1_row = src1 + i_src1;
    ধ্রুব src2_t * src2_row = src2 + i_src2;
    dst_t *        dst_row  = dst + i_dst;

    প্রতি (পূর্ণাঙ্ক i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        ধ্রুব পূর্ণাঙ্ক   i10 = i0 % ne10;
        ধ্রুব পূর্ণাঙ্ক   i20 = i0 % ne20;
        ধ্রুব ভিন্ন acc = bin_op((ভিন্ন) src0_row[i0 * s00], (ভিন্ন) src1_row[i10 * s10]);
        dst_row[i0]     = (dst_t) bin_op(acc, (ভিন্ন) src2_row[i20 * s20]);
    }
}

template<ভিন্ন (*bin_op)(ধ্রুব ভিন্ন, ধ্রুব ভিন্ন), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
স্থির শূন্য k_bin_bcast3_unravel(ধ্রুব src0_t * src0, ধ্রুব src1_t * src1, ধ্রুব src2_t * src2, dst_t * dst,
        পূর্ণাঙ্ক ne0, পূর্ণাঙ্ক ne1, পূর্ণাঙ্ক ne2, পূর্ণাঙ্ক ne3,
        পূর্ণাঙ্ক ne10, পূর্ণাঙ্ক ne11, পূর্ণাঙ্ক ne12, পূর্ণাঙ্ক ne13,
        পূর্ণাঙ্ক ne20, পূর্ণাঙ্ক ne21, পূর্ণাঙ্ক ne22, পূর্ণাঙ্ক ne23,
        পূর্ণাঙ্ক s1, পূর্ণাঙ্ক s2, পূর্ণাঙ্ক s3,
        পূর্ণাঙ্ক s00, পূর্ণাঙ্ক s01, পূর্ণাঙ্ক s02, পূর্ণাঙ্ক s03,
        পূর্ণাঙ্ক s10, পূর্ণাঙ্ক s11, পূর্ণাঙ্ক s12, পূর্ণাঙ্ক s13,
        পূর্ণাঙ্ক s20, পূর্ণাঙ্ক s21, পূর্ণাঙ্ক s22, পূর্ণাঙ্ক s23,
        ধ্রুব sycl::nd_item<3> & item_ct1) {
    ধ্রুব পূর্ণাঙ্ক i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    ধ্রুব পূর্ণাঙ্ক i3 = i / (ne2 * ne1 * ne0);
    ধ্রুব পূর্ণাঙ্ক i2 = (i / (ne1 * ne0)) % ne2;
    ধ্রুব পূর্ণাঙ্ক i1 = (i / ne0) % ne1;
    ধ্রুব পূর্ণাঙ্ক i0 = i % ne0;

    যদি (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        প্রত্যাগম;
    }

    ধ্রুব পূর্ণাঙ্ক i11 = i1 % ne11;
    ধ্রুব পূর্ণাঙ্ক i12 = i2 % ne12;
    ধ্রুব পূর্ণাঙ্ক i13 = i3 % ne13;
    ধ্রুব পূর্ণাঙ্ক i21 = i1 % ne21;
    ধ্রুব পূর্ণাঙ্ক i22 = i2 % ne22;
    ধ্রুব পূর্ণাঙ্ক i23 = i3 % ne23;

    ধ্রুব size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    ধ্রুব size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    ধ্রুব size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    ধ্রুব size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    ধ্রুব পূর্ণাঙ্ক   i10 = i0 % ne10;
    ধ্রুব পূর্ণাঙ্ক   i20 = i0 % ne20;
    ধ্রুব ভিন্ন acc = bin_op((ভিন্ন) src0[i_src0 + i0 * s00], (ভিন্ন) src1[i_src1 + i10 * s10]);
    dst[i_dst + i0] = (dst_t) bin_op(acc, (ভিন্ন) src2[i_src2 + i20 * s20]);
}

template<ভিন্ন (*bin_op)(ধ্রুব ভিন্ন, ধ্রুব ভিন্ন), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
স্থির শূন্য launch_bin_bcast3(ggml_backend_sycl_context & ctx, ধ্রুব ggml_tensor * src0, ধ্রুব ggml_tensor * src1,
                              ধ্রুব ggml_tensor * src2, ggml_tensor * dst) {
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_TENSOR_TERNARY_OP_LOCALS

    পূর্ণাঙ্ক nr1[4] = { (পূর্ণাঙ্ক) (ne10 / ne0), (পূর্ণাঙ্ক) (ne11 / ne1), (পূর্ণাঙ্ক) (ne12 / ne2), (পূর্ণাঙ্ক) (ne13 / ne3) };
    পূর্ণাঙ্ক nr2[4] = { (পূর্ণাঙ্ক) (ne20 / ne0), (পূর্ণাঙ্ক) (ne21 / ne1), (পূর্ণাঙ্ক) (ne22 / ne2), (পূর্ণাঙ্ক) (ne23 / ne3) };

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

    ধ্রুব তার্কিক can_collapse = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(src2) &&
                              !ggml_is_permuted(src0) && !ggml_is_permuted(src1) && !ggml_is_permuted(src2);
    যদি (can_collapse) {
        প্রতি (পূর্ণাঙ্ক i = 0; i < 4; i++) {
            যদি (nr1[i] != 1 || nr2[i] != 1) {
                ভঙ্গ;
            }
            যদি (i > 0) {
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

        size_t s1 = cnb[1] / পরিমাণ(dst_t);
        size_t s2 = cnb[2] / পরিমাণ(dst_t);
        size_t s3 = cnb[3] / পরিমাণ(dst_t);

        size_t s00 = cnb0[0] / পরিমাণ(src0_t);
        size_t s01 = cnb0[1] / পরিমাণ(src0_t);
        size_t s02 = cnb0[2] / পরিমাণ(src0_t);
        size_t s03 = cnb0[3] / পরিমাণ(src0_t);

        size_t s10 = cnb1[0] / পরিমাণ(src1_t);
        size_t s11 = cnb1[1] / পরিমাণ(src1_t);
        size_t s12 = cnb1[2] / পরিমাণ(src1_t);
        size_t s13 = cnb1[3] / পরিমাণ(src1_t);

        size_t s20 = cnb2[0] / পরিমাণ(src2_t);
        size_t s21 = cnb2[1] / পরিমাণ(src2_t);
        size_t s22 = cnb2[2] / পরিমাণ(src2_t);
        size_t s23 = cnb2[3] / পরিমাণ(src2_t);

        GGML_ASSERT(cnb[0] % পরিমাণ(dst_t) == 0 && cnb[1] % পরিমাণ(dst_t) == 0 && cnb[2] % পরিমাণ(dst_t) == 0 &&
                    cnb[3] % পরিমাণ(dst_t) == 0);
        GGML_ASSERT(cnb0[0] % পরিমাণ(src0_t) == 0 && cnb0[1] % পরিমাণ(src0_t) == 0 && cnb0[2] % পরিমাণ(src0_t) == 0 &&
                    cnb0[3] % পরিমাণ(src0_t) == 0);
        GGML_ASSERT(cnb1[0] % পরিমাণ(src1_t) == 0 && cnb1[1] % পরিমাণ(src1_t) == 0 && cnb1[2] % পরিমাণ(src1_t) == 0 &&
                    cnb1[3] % পরিমাণ(src1_t) == 0);
        GGML_ASSERT(cnb2[0] % পরিমাণ(src2_t) == 0 && cnb2[1] % পরিমাণ(src2_t) == 0 && cnb2[2] % পরিমাণ(src2_t) == 0 &&
                    cnb2[3] % পরিমাণ(src2_t) == 0);

        ধ্রুব src0_t * src0_dd = (ধ্রুব src0_t *) src0->data;
        ধ্রুব src1_t * src1_dd = (ধ্রুব src1_t *) src1->data;
        ধ্রুব src2_t * src2_dd = (ধ্রুব src2_t *) src2->data;
        dst_t *        dst_dd  = (dst_t *) dst->data;

        ধ্রুব পূর্ণাঙ্ক block_size = 128;
        int64_t   hne0       = std::max(ne0 / 2LL, 1LL);

        sycl::range<3> block_dims(1, 1, 1);
        block_dims[2] = std::min<unsigned পূর্ণাঙ্ক>(hne0, block_size);
        block_dims[1] = std::min<unsigned পূর্ণাঙ্ক>(ne1, block_size / (unsigned পূর্ণাঙ্ক) block_dims[2]);
        block_dims[0] = std::min(std::min<unsigned পূর্ণাঙ্ক>(ne2 * ne3,
                                                        block_size / (unsigned পূর্ণাঙ্ক) block_dims[2] /
                                                            (unsigned পূর্ণাঙ্ক) block_dims[1]),
                                 64U);

        sycl::range<3> block_nums((ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                                  (ne1 + block_dims[1] - 1) / block_dims[1],
                                  (hne0 + block_dims[2] - 1) / block_dims[2]);

        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        যদি (block_nums[0] > 65535) {
            পূর্ণাঙ্ক block_অঙ্ক = (ne0 * ne1 * ne2 * ne3 + block_size - 1) / block_size;
            stream->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, block_অঙ্ক) * sycl::range<3>(1, 1, block_size),
                                  sycl::range<3>(1, 1, block_size)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_bin_bcast3_unravel<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10, ne11,
                                                 ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00, s01, s02, s03,
                                                 s10, s11, s12, s13, s20, s21, s22, s23, item_ct1);
                });
        } অন্যথা {
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

শূন্য ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1) {
    ধ্রুব ggml_tensor * src0 = add0->src[0];
    ধ্রুব ggml_tensor * src1 = add0->src[1];
    ধ্রুব ggml_tensor * src2 = add1->src[1];
    ggml_tensor *       dst  = add1;

    GGML_ASSERT(add1->src[0] == add0);
    GGML_ASSERT(ggml_sycl_add_kernel_supports(src0->type, src1->type, add0->type));
    GGML_ASSERT(ggml_sycl_add_kernel_supports(add0->type, src2->type, dst->type));

    যদি (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        launch_bin_bcast3<op_add, ভিন্ন, ভিন্ন, ভিন্ন, ভিন্ন>(ctx, src0, src1, src2, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, sycl::half, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, ভিন্ন, ভিন্ন, sycl::half>(ctx, src0, src1, src2, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, sycl::half, ভিন্ন, sycl::half>(ctx, src0, src1, src2, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, ভিন্ন, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && src2->type == GGML_TYPE_I32 &&
               dst->type == GGML_TYPE_I32) {
        launch_bin_bcast3<op_add, int32_t, int32_t, int32_t, int32_t>(ctx, src0, src1, src2, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && src2->type == GGML_TYPE_I16 &&
               dst->type == GGML_TYPE_I16) {
        launch_bin_bcast3<op_add, int16_t, int16_t, int16_t, int16_t>(ctx, src0, src1, src2, dst);
#ifdef GGML_SYCL_HAS_BF16
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, ভিন্ন, ভিন্ন, sycl::ext::oneapi::bfloat16>(
            ctx, src0, src1, src2, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, ভিন্ন,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } অন্যথা যদি (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, ভিন্ন, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
#endif
    } অন্যথা {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s, src2: %s\n", __func__,
                ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type),
                ggml_type_name(src2->type));
        GGML_ABORT("fatal error");
    }
}

