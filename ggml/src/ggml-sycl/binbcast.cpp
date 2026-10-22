#अन्तर्भाव "binbcast.hpp"

#अन्तर्भाव <algorithm>
#अन्तर्भाव <cstddef>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <sycl/sycl.hpp>

#अन्तर्भाव "ggml.h"

प्रतिरूपम्<भिन्न (*bin_op)(ध्रुव भिन्न, ध्रुव भिन्न), प्रकारनाम src0_t, प्रकारनाम src1_t, प्रकारनाम dst_t>
स्थिर शून्य k_bin_bcast(ध्रुव src0_t * src0, ध्रुव src1_t * src1, dst_t * dst,
        पूर्णाङ्क ne0, पूर्णाङ्क ne1, पूर्णाङ्क ne2, पूर्णाङ्क ne3,
        पूर्णाङ्क ne10, पूर्णाङ्क ne11, पूर्णाङ्क ne12, पूर्णाङ्क ne13,
        /*int s0, */ पूर्णाङ्क s1,  पूर्णाङ्क s2,  पूर्णाङ्क s3,
        पूर्णाङ्क s00, पूर्णाङ्क s01, पूर्णाङ्क s02, पूर्णाङ्क s03,
        पूर्णाङ्क s10, पूर्णाङ्क s11, पूर्णाङ्क s12, पूर्णाङ्क s13,
        ध्रुव sycl::nd_item<3> &item_ct1) {
    ध्रुव पूर्णाङ्क i0s = item_ct1.get_local_range(2) * item_ct1.get_गणः(2) +
                    item_ct1.get_local_id(2);
    ध्रुव पूर्णाङ्क i1 = (item_ct1.get_local_range(1) * item_ct1.get_गणः(1) +
                    item_ct1.get_local_id(1));
    ध्रुव पूर्णाङ्क i2 = (item_ct1.get_local_range(0) * item_ct1.get_गणः(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    ध्रुव पूर्णाङ्क i3 = (item_ct1.get_local_range(0) * item_ct1.get_गणः(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    यदि (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क i11 = i1 % ne11;
    ध्रुव पूर्णाङ्क i12 = i2 % ne12;
    ध्रुव पूर्णाङ्क i13 = i3 % ne13;

    ध्रुव size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    ध्रुव size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    ध्रुव size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    ध्रुव src0_t * src0_row = src0 + i_src0;
    ध्रुव src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    प्रति (पूर्णाङ्क i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_गणः_range(2)) {
        ध्रुव पूर्णाङ्क i10 = i0 % ne10;
        dst_row[i0] = (dst_t)bin_op(src0 ? (भिन्न)src0_row[i0*s00] : 0.0f, (भिन्न)src1_row[i10*s10]);
    }
}

प्रतिरूपम्<भिन्न (*bin_op)(ध्रुव भिन्न, ध्रुव भिन्न), प्रकारनाम src0_t, प्रकारनाम src1_t, प्रकारनाम dst_t>
स्थिर शून्य k_bin_bcast_unravel(ध्रुव src0_t * src0, ध्रुव src1_t * src1, dst_t * dst,
        पूर्णाङ्क ne0, पूर्णाङ्क ne1, पूर्णाङ्क ne2, पूर्णाङ्क ne3,
        पूर्णाङ्क ne10, पूर्णाङ्क ne11, पूर्णाङ्क ne12, पूर्णाङ्क ne13,
        /*int s0, */ पूर्णाङ्क s1,  पूर्णाङ्क s2,  पूर्णाङ्क s3,
        पूर्णाङ्क s00, पूर्णाङ्क s01, पूर्णाङ्क s02, पूर्णाङ्क s03,
        पूर्णाङ्क s10, पूर्णाङ्क s11, पूर्णाङ्क s12, पूर्णाङ्क s13,
        ध्रुव sycl::nd_item<3> &item_ct1) {

    ध्रुव पूर्णाङ्क i = item_ct1.get_local_range(2) * item_ct1.get_गणः(2) +
                  item_ct1.get_local_id(2);

    ध्रुव पूर्णाङ्क i3 = i/(ne2*ne1*ne0);
    ध्रुव पूर्णाङ्क i2 = (i/(ne1*ne0)) % ne2;
    ध्रुव पूर्णाङ्क i1 = (i/ne0) % ne1;
    ध्रुव पूर्णाङ्क i0 = i % ne0;

    यदि (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क i11 = i1 % ne11;
    ध्रुव पूर्णाङ्क i12 = i2 % ne12;
    ध्रुव पूर्णाङ्क i13 = i3 % ne13;

    ध्रुव size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    ध्रुव size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    ध्रुव size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    ध्रुव src0_t * src0_row = src0 + i_src0;
    ध्रुव src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    ध्रुव पूर्णाङ्क i10 = i0 % ne10;
    dst_row[i0] = (dst_t)bin_op(src0 ? (भिन्न)src0_row[i0*s00] : 0.0f, (भिन्न)src1_row[i10*s10]);
}


प्रतिरूपम्<भिन्न (*bin_op)(ध्रुव भिन्न, ध्रुव भिन्न)>
संरचना bin_bcast_sycl {
    प्रतिरूपम् <प्रकारनाम src0_t, प्रकारनाम src1_t, प्रकारनाम dst_t>
    शून्य संक्रियकः()(ध्रुव src0_t * src0_dd, ध्रुव src1_t * src1_dd, dst_t * dst_dd, ध्रुव int64_t ne00,
                    ध्रुव int64_t ne01, ध्रुव int64_t ne02, ध्रुव int64_t ne03, ध्रुव int64_t ne10, ध्रुव int64_t ne11,
                    ध्रुव int64_t ne12, ध्रुव int64_t ne13, ध्रुव int64_t ne0, ध्रुव int64_t ne1, ध्रुव int64_t ne2,
                    ध्रुव int64_t ne3, ध्रुव size_t nb00, ध्रुव size_t nb01, ध्रुव size_t nb02, ध्रुव size_t nb03,
                    ध्रुव size_t nb10, ध्रुव size_t nb11, ध्रुव size_t nb12, ध्रुव size_t nb13, ध्रुव size_t nb0,
                    ध्रुव size_t nb1, ध्रुव size_t nb2, ध्रुव size_t nb3, ध्रुव तार्किक src0_is_contiguous,
                    ध्रुव तार्किक src1_is_contiguous, ध्रुव तार्किक src0_is_permuted, ध्रुव तार्किक src1_is_permuted,
                    queue_ptr stream) {
        पूर्णाङ्क nr0 = ne10 / ne0;
        पूर्णाङ्क nr1 = ne11/ne1;
        पूर्णाङ्क nr2 = ne12/ne2;
        पूर्णाङ्क nr3 = ne13/ne3;

        पूर्णाङ्क nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne[] = {ne0, ne1, ne2, ne3};
        int64_t cne0[] = {ne00, ne01, ne02, ne03};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};
        size_t cnb[] = {nb0, nb1, nb2, nb3};
        size_t cnb0[] = {nb00, nb01, nb02, nb03};
        size_t cnb1[] = {nb10, nb11, nb12, nb13};
        स्वचालित collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        स्वचालित collapse_nb = [](size_t cnb[], int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        यदि (src0_is_contiguous && src1_is_contiguous && !src0_is_permuted && !src1_is_permuted) {
            प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
                यदि (nr[i] != 1) {
                    भङ्ग;
                }
                यदि (i > 0) {
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
            size_t s1 = nb1 / परिमाण(dst_t);
            size_t s2 = nb2 / परिमाण(dst_t);
            size_t s3 = nb3 / परिमाण(dst_t);

            size_t s10 = nb10 / परिमाण(src1_t);
            size_t s11 = nb11 / परिमाण(src1_t);
            size_t s12 = nb12 / परिमाण(src1_t);
            size_t s13 = nb13 / परिमाण(src1_t);

            size_t s00 = nb00 / परिमाण(src0_t);
            size_t s01 = nb01 / परिमाण(src0_t);
            size_t s02 = nb02 / परिमाण(src0_t);
            size_t s03 = nb03 / परिमाण(src0_t);

            GGML_अप्रयुक्तम्(s00);

            GGML_ASSERT(nb0 % परिमाण(dst_t) == 0);
            GGML_ASSERT(nb1 % परिमाण(dst_t) == 0);
            GGML_ASSERT(nb2 % परिमाण(dst_t) == 0);
            GGML_ASSERT(nb3 % परिमाण(dst_t) == 0);

            GGML_ASSERT(nb00 % परिमाण(src0_t) == 0);
            GGML_ASSERT(nb01 % परिमाण(src0_t) == 0);
            GGML_ASSERT(nb02 % परिमाण(src0_t) == 0);
            GGML_ASSERT(nb03 % परिमाण(src0_t) == 0);

            GGML_ASSERT(nb10 % परिमाण(src1_t) == 0);
            GGML_ASSERT(nb11 % परिमाण(src1_t) == 0);
            GGML_ASSERT(nb12 % परिमाण(src1_t) == 0);
            GGML_ASSERT(nb13 % परिमाण(src1_t) == 0);

            ध्रुव पूर्णाङ्क खण्डम्_आकार_3 = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            sycl::range<3> खण्डम्_dims(1, 1, 1);
            खण्डम्_dims[2] = std::min<अचिह्नित पूर्णाङ्क>(hne0, खण्डम्_आकार_3);
            खण्डम्_dims[1] = std::min<अचिह्नित पूर्णाङ्क>(
                ne1, खण्डम्_आकार_3 / (अचिह्नित पूर्णाङ्क)खण्डम्_dims[2]);
            खण्डम्_dims[0] = std::min(
                std::min<अचिह्नित पूर्णाङ्क>(
                    ne2 * ne3, खण्डम्_आकार_3 / (अचिह्नित पूर्णाङ्क)खण्डम्_dims[2] /
                                   (अचिह्नित पूर्णाङ्क)खण्डम्_dims[1]),
                64U);

            sycl::range<3> खण्डम्_nums(
                (ne2 * ne3 + खण्डम्_dims[0] - 1) / खण्डम्_dims[0],
                (ne1 + खण्डम्_dims[1] - 1) / खण्डम्_dims[1],
                (hne0 + खण्डम्_dims[2] - 1) / खण्डम्_dims[2]);

            यदि (खण्डम्_nums[0] > 65535) {
                // this is the maximum number of blocks in z direction, fallback to 1D grid kernel
                पूर्णाङ्क खण्डम्_अङ्कः = (ne0*ne1*ne2*ne3 + खण्डम्_आकार_3 - 1) / खण्डम्_आकार_3;
                {
                    dpct::has_capability_or_fail(stream->get_device(),
                                                 {sycl::aspect::fp16});

                    stream->parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, खण्डम्_अङ्कः) *
                                              sycl::range<3>(1, 1, खण्डम्_आकार_3),
                                          sycl::range<3>(1, 1, खण्डम्_आकार_3)),
                        [=](sycl::nd_item<3> item_ct1) {
                            k_bin_bcast_unravel<bin_op>(
                                src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3,
                                ne10, ne11, ne12, ne13, s1, s2, s3, s00, s01, s02,
                                s03, s10, s11, s12, s13, item_ct1);
                        });
                }
            } अन्यथा {
                /*
                DPCT1049:16: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
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

प्रतिरूपम् <वर्ग op>
अन्तःस्थ शून्य ggml_sycl_op_bin_bcast(ggml_backend_sycl_context & ctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1,
                                   ggml_tensor * dst) {
    dpct::queue_ptr main_प्रवाहः = ctx.stream();
    GGML_TENSOR_द्विमान_OP_LOCALS

    यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()((ध्रुव भिन्न *) src0->दत्त, (ध्रुव भिन्न *) src1->दत्त, (भिन्न *) dst->दत्त, ne00, ne01, ne02, ne03, ne10,
             ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3,
             ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_प्रवाहः);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()((ध्रुव sycl::half *) src0->दत्त, (ध्रुव sycl::half *) src1->दत्त, (sycl::half *) dst->दत्त, ne00, ne01,
             ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13,
             nb0, nb1, nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_प्रवाहः);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()((ध्रुव sycl::half *) src0->दत्त, (ध्रुव भिन्न *) src1->दत्त, (sycl::half *) dst->दत्त, ne00, ne01, ne02,
             ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1,
             nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_प्रवाहः);
    } अन्यथा यदि (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()((ध्रुव int32_t *) src0->दत्त, (ध्रुव int32_t *) src1->दत्त, (int32_t *) dst->दत्त, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_प्रवाहः);
    } अन्यथा यदि (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()((ध्रुव int16_t *) src0->दत्त, (ध्रुव int16_t *) src1->दत्त, (int16_t *) dst->दत्त, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_प्रवाहः);
#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        op()((ध्रुव sycl::ext::oneapi::bfloat16 *) src0->दत्त, (ध्रुव sycl::ext::oneapi::bfloat16 *) src1->दत्त,
             (sycl::ext::oneapi::bfloat16 *) dst->दत्त, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_प्रवाहः);
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        op()((ध्रुव sycl::ext::oneapi::bfloat16 *) src0->दत्त, (ध्रुव भिन्न *) src1->दत्त,
             (sycl::ext::oneapi::bfloat16 *) dst->दत्त, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_प्रवाहः);
#यदि_अन्तः
    } अन्यथा {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(dst->type),
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

अन्तःस्थ शून्य ggml_sycl_op_सङ्कलय(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_सङ्कलय>>(ctx, dst->src[0], dst->src[1], dst);
}

अन्तःस्थ शून्य ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_sub>>(ctx, dst->src[0], dst->src[1], dst);
}

अन्तःस्थ शून्य ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(ctx, dst->src[0], dst->src[1], dst);
}

अन्तःस्थ शून्य ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(ctx, dst->src[0], dst->src[1], dst);
}

अन्तःस्थ शून्य ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(ctx, dst, dst->src[0], dst);
}


शून्य ggml_sycl_सङ्कलय(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_सङ्कलय(ctx, dst);
}

शून्य ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_sub(ctx, dst);
}

शून्य ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_mul(ctx, dst);
}

शून्य ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_div(ctx, dst);
}

शून्य ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat(ctx, dst);
}

// fused ADD+ADD: dst = (src0 + src1) + src2. Same indexing as k_bin_bcast, so mixed
// types, broadcast, and non-contiguous layouts that add() already handles also fuse.
प्रतिरूपम्<भिन्न (*bin_op)(ध्रुव भिन्न, ध्रुव भिन्न), प्रकारनाम src0_t, प्रकारनाम src1_t, प्रकारनाम src2_t, प्रकारनाम dst_t>
स्थिर शून्य k_bin_bcast3(ध्रुव src0_t * src0, ध्रुव src1_t * src1, ध्रुव src2_t * src2, dst_t * dst,
        पूर्णाङ्क ne0, पूर्णाङ्क ne1, पूर्णाङ्क ne2, पूर्णाङ्क ne3,
        पूर्णाङ्क ne10, पूर्णाङ्क ne11, पूर्णाङ्क ne12, पूर्णाङ्क ne13,
        पूर्णाङ्क ne20, पूर्णाङ्क ne21, पूर्णाङ्क ne22, पूर्णाङ्क ne23,
        पूर्णाङ्क s1, पूर्णाङ्क s2, पूर्णाङ्क s3,
        पूर्णाङ्क s00, पूर्णाङ्क s01, पूर्णाङ्क s02, पूर्णाङ्क s03,
        पूर्णाङ्क s10, पूर्णाङ्क s11, पूर्णाङ्क s12, पूर्णाङ्क s13,
        पूर्णाङ्क s20, पूर्णाङ्क s21, पूर्णाङ्क s22, पूर्णाङ्क s23,
        ध्रुव sycl::nd_item<3> & item_ct1) {
    ध्रुव पूर्णाङ्क i0s = item_ct1.get_local_range(2) * item_ct1.get_गणः(2) +
                    item_ct1.get_local_id(2);
    ध्रुव पूर्णाङ्क i1 = (item_ct1.get_local_range(1) * item_ct1.get_गणः(1) +
                    item_ct1.get_local_id(1));
    ध्रुव पूर्णाङ्क i2 = (item_ct1.get_local_range(0) * item_ct1.get_गणः(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    ध्रुव पूर्णाङ्क i3 = (item_ct1.get_local_range(0) * item_ct1.get_गणः(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    यदि (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क i11 = i1 % ne11;
    ध्रुव पूर्णाङ्क i12 = i2 % ne12;
    ध्रुव पूर्णाङ्क i13 = i3 % ne13;
    ध्रुव पूर्णाङ्क i21 = i1 % ne21;
    ध्रुव पूर्णाङ्क i22 = i2 % ne22;
    ध्रुव पूर्णाङ्क i23 = i3 % ne23;

    ध्रुव size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    ध्रुव size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    ध्रुव size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    ध्रुव size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    ध्रुव src0_t * src0_row = src0 + i_src0;
    ध्रुव src1_t * src1_row = src1 + i_src1;
    ध्रुव src2_t * src2_row = src2 + i_src2;
    dst_t *        dst_row  = dst + i_dst;

    प्रति (पूर्णाङ्क i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_गणः_range(2)) {
        ध्रुव पूर्णाङ्क   i10 = i0 % ne10;
        ध्रुव पूर्णाङ्क   i20 = i0 % ne20;
        ध्रुव भिन्न acc = bin_op((भिन्न) src0_row[i0 * s00], (भिन्न) src1_row[i10 * s10]);
        dst_row[i0]     = (dst_t) bin_op(acc, (भिन्न) src2_row[i20 * s20]);
    }
}

प्रतिरूपम्<भिन्न (*bin_op)(ध्रुव भिन्न, ध्रुव भिन्न), प्रकारनाम src0_t, प्रकारनाम src1_t, प्रकारनाम src2_t, प्रकारनाम dst_t>
स्थिर शून्य k_bin_bcast3_unravel(ध्रुव src0_t * src0, ध्रुव src1_t * src1, ध्रुव src2_t * src2, dst_t * dst,
        पूर्णाङ्क ne0, पूर्णाङ्क ne1, पूर्णाङ्क ne2, पूर्णाङ्क ne3,
        पूर्णाङ्क ne10, पूर्णाङ्क ne11, पूर्णाङ्क ne12, पूर्णाङ्क ne13,
        पूर्णाङ्क ne20, पूर्णाङ्क ne21, पूर्णाङ्क ne22, पूर्णाङ्क ne23,
        पूर्णाङ्क s1, पूर्णाङ्क s2, पूर्णाङ्क s3,
        पूर्णाङ्क s00, पूर्णाङ्क s01, पूर्णाङ्क s02, पूर्णाङ्क s03,
        पूर्णाङ्क s10, पूर्णाङ्क s11, पूर्णाङ्क s12, पूर्णाङ्क s13,
        पूर्णाङ्क s20, पूर्णाङ्क s21, पूर्णाङ्क s22, पूर्णाङ्क s23,
        ध्रुव sycl::nd_item<3> & item_ct1) {
    ध्रुव पूर्णाङ्क i = item_ct1.get_local_range(2) * item_ct1.get_गणः(2) +
                  item_ct1.get_local_id(2);

    ध्रुव पूर्णाङ्क i3 = i / (ne2 * ne1 * ne0);
    ध्रुव पूर्णाङ्क i2 = (i / (ne1 * ne0)) % ne2;
    ध्रुव पूर्णाङ्क i1 = (i / ne0) % ne1;
    ध्रुव पूर्णाङ्क i0 = i % ne0;

    यदि (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        प्रत्यागम;
    }

    ध्रुव पूर्णाङ्क i11 = i1 % ne11;
    ध्रुव पूर्णाङ्क i12 = i2 % ne12;
    ध्रुव पूर्णाङ्क i13 = i3 % ne13;
    ध्रुव पूर्णाङ्क i21 = i1 % ne21;
    ध्रुव पूर्णाङ्क i22 = i2 % ne22;
    ध्रुव पूर्णाङ्क i23 = i3 % ne23;

    ध्रुव size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    ध्रुव size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    ध्रुव size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    ध्रुव size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    ध्रुव पूर्णाङ्क   i10 = i0 % ne10;
    ध्रुव पूर्णाङ्क   i20 = i0 % ne20;
    ध्रुव भिन्न acc = bin_op((भिन्न) src0[i_src0 + i0 * s00], (भिन्न) src1[i_src1 + i10 * s10]);
    dst[i_dst + i0] = (dst_t) bin_op(acc, (भिन्न) src2[i_src2 + i20 * s20]);
}

प्रतिरूपम्<भिन्न (*bin_op)(ध्रुव भिन्न, ध्रुव भिन्न), प्रकारनाम src0_t, प्रकारनाम src1_t, प्रकारनाम src2_t, प्रकारनाम dst_t>
स्थिर शून्य launch_bin_bcast3(ggml_backend_sycl_context & ctx, ध्रुव ggml_tensor * src0, ध्रुव ggml_tensor * src1,
                              ध्रुव ggml_tensor * src2, ggml_tensor * dst) {
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_समुच्चयः_device(ctx.device));

    GGML_TENSOR_TERNARY_OP_LOCALS

    पूर्णाङ्क nr1[4] = { (पूर्णाङ्क) (ne10 / ne0), (पूर्णाङ्क) (ne11 / ne1), (पूर्णाङ्क) (ne12 / ne2), (पूर्णाङ्क) (ne13 / ne3) };
    पूर्णाङ्क nr2[4] = { (पूर्णाङ्क) (ne20 / ne0), (पूर्णाङ्क) (ne21 / ne1), (पूर्णाङ्क) (ne22 / ne2), (पूर्णाङ्क) (ne23 / ne3) };

    int64_t cne[]  = { ne0, ne1, ne2, ne3 };
    int64_t cne0[] = { ne00, ne01, ne02, ne03 };
    int64_t cne1[] = { ne10, ne11, ne12, ne13 };
    int64_t cne2[] = { ne20, ne21, ne22, ne23 };
    size_t  cnb[]  = { nb0, nb1, nb2, nb3 };
    size_t  cnb0[] = { nb00, nb01, nb02, nb03 };
    size_t  cnb1[] = { nb10, nb11, nb12, nb13 };
    size_t  cnb2[] = { nb20, nb21, nb22, nb23 };

    स्वचालित collapse = [](int64_t cne[]) {
        cne[0] *= cne[1];
        cne[1] = cne[2];
        cne[2] = cne[3];
        cne[3] = 1;
    };

    स्वचालित collapse_nb = [](size_t cnb[], int64_t cne[]) {
        cnb[1] *= cne[1];
        cnb[2] *= cne[2];
        cnb[3] *= cne[3];
    };

    ध्रुव तार्किक can_collapse = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(src2) &&
                              !ggml_is_permuted(src0) && !ggml_is_permuted(src1) && !ggml_is_permuted(src2);
    यदि (can_collapse) {
        प्रति (पूर्णाङ्क i = 0; i < 4; i++) {
            यदि (nr1[i] != 1 || nr2[i] != 1) {
                भङ्ग;
            }
            यदि (i > 0) {
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

        size_t s1 = cnb[1] / परिमाण(dst_t);
        size_t s2 = cnb[2] / परिमाण(dst_t);
        size_t s3 = cnb[3] / परिमाण(dst_t);

        size_t s00 = cnb0[0] / परिमाण(src0_t);
        size_t s01 = cnb0[1] / परिमाण(src0_t);
        size_t s02 = cnb0[2] / परिमाण(src0_t);
        size_t s03 = cnb0[3] / परिमाण(src0_t);

        size_t s10 = cnb1[0] / परिमाण(src1_t);
        size_t s11 = cnb1[1] / परिमाण(src1_t);
        size_t s12 = cnb1[2] / परिमाण(src1_t);
        size_t s13 = cnb1[3] / परिमाण(src1_t);

        size_t s20 = cnb2[0] / परिमाण(src2_t);
        size_t s21 = cnb2[1] / परिमाण(src2_t);
        size_t s22 = cnb2[2] / परिमाण(src2_t);
        size_t s23 = cnb2[3] / परिमाण(src2_t);

        GGML_ASSERT(cnb[0] % परिमाण(dst_t) == 0 && cnb[1] % परिमाण(dst_t) == 0 && cnb[2] % परिमाण(dst_t) == 0 &&
                    cnb[3] % परिमाण(dst_t) == 0);
        GGML_ASSERT(cnb0[0] % परिमाण(src0_t) == 0 && cnb0[1] % परिमाण(src0_t) == 0 && cnb0[2] % परिमाण(src0_t) == 0 &&
                    cnb0[3] % परिमाण(src0_t) == 0);
        GGML_ASSERT(cnb1[0] % परिमाण(src1_t) == 0 && cnb1[1] % परिमाण(src1_t) == 0 && cnb1[2] % परिमाण(src1_t) == 0 &&
                    cnb1[3] % परिमाण(src1_t) == 0);
        GGML_ASSERT(cnb2[0] % परिमाण(src2_t) == 0 && cnb2[1] % परिमाण(src2_t) == 0 && cnb2[2] % परिमाण(src2_t) == 0 &&
                    cnb2[3] % परिमाण(src2_t) == 0);

        ध्रुव src0_t * src0_dd = (ध्रुव src0_t *) src0->दत्त;
        ध्रुव src1_t * src1_dd = (ध्रुव src1_t *) src1->दत्त;
        ध्रुव src2_t * src2_dd = (ध्रुव src2_t *) src2->दत्त;
        dst_t *        dst_dd  = (dst_t *) dst->दत्त;

        ध्रुव पूर्णाङ्क खण्डम्_आकार_3 = 128;
        int64_t   hne0       = std::max(ne0 / 2LL, 1LL);

        sycl::range<3> खण्डम्_dims(1, 1, 1);
        खण्डम्_dims[2] = std::min<अचिह्नित पूर्णाङ्क>(hne0, खण्डम्_आकार_3);
        खण्डम्_dims[1] = std::min<अचिह्नित पूर्णाङ्क>(ne1, खण्डम्_आकार_3 / (अचिह्नित पूर्णाङ्क) खण्डम्_dims[2]);
        खण्डम्_dims[0] = std::min(std::min<अचिह्नित पूर्णाङ्क>(ne2 * ne3,
                                                        खण्डम्_आकार_3 / (अचिह्नित पूर्णाङ्क) खण्डम्_dims[2] /
                                                            (अचिह्नित पूर्णाङ्क) खण्डम्_dims[1]),
                                 64U);

        sycl::range<3> खण्डम्_nums((ne2 * ne3 + खण्डम्_dims[0] - 1) / खण्डम्_dims[0],
                                  (ne1 + खण्डम्_dims[1] - 1) / खण्डम्_dims[1],
                                  (hne0 + खण्डम्_dims[2] - 1) / खण्डम्_dims[2]);

        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        यदि (खण्डम्_nums[0] > 65535) {
            पूर्णाङ्क खण्डम्_अङ्कः = (ne0 * ne1 * ne2 * ne3 + खण्डम्_आकार_3 - 1) / खण्डम्_आकार_3;
            stream->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, खण्डम्_अङ्कः) * sycl::range<3>(1, 1, खण्डम्_आकार_3),
                                  sycl::range<3>(1, 1, खण्डम्_आकार_3)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_bin_bcast3_unravel<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10, ne11,
                                                 ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00, s01, s02, s03,
                                                 s10, s11, s12, s13, s20, s21, s22, s23, item_ct1);
                });
        } अन्यथा {
            stream->parallel_for(sycl::nd_range<3>(खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
                                 [=](sycl::nd_item<3> item_ct1) {
                                     k_bin_bcast3<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10,
                                                          ne11, ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00,
                                                          s01, s02, s03, s10, s11, s12, s13, s20, s21, s22, s23,
                                                          item_ct1);
                                 });
        }
    }
}

शून्य ggml_sycl_op_सङ्कलय_सङ्कलय_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1) {
    ध्रुव ggml_tensor * src0 = add0->src[0];
    ध्रुव ggml_tensor * src1 = add0->src[1];
    ध्रुव ggml_tensor * src2 = add1->src[1];
    ggml_tensor *       dst  = add1;

    GGML_ASSERT(add1->src[0] == add0);
    GGML_ASSERT(ggml_sycl_सङ्कलय_kernel_supports(src0->type, src1->type, add0->type));
    GGML_ASSERT(ggml_sycl_सङ्कलय_kernel_supports(add0->type, src2->type, dst->type));

    यदि (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        launch_bin_bcast3<op_सङ्कलय, भिन्न, भिन्न, भिन्न, भिन्न>(ctx, src0, src1, src2, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_सङ्कलय, sycl::half, sycl::half, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_सङ्कलय, sycl::half, भिन्न, भिन्न, sycl::half>(ctx, src0, src1, src2, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_सङ्कलय, sycl::half, sycl::half, भिन्न, sycl::half>(ctx, src0, src1, src2, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_सङ्कलय, sycl::half, भिन्न, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && src2->type == GGML_TYPE_I32 &&
               dst->type == GGML_TYPE_I32) {
        launch_bin_bcast3<op_सङ्कलय, int32_t, int32_t, int32_t, int32_t>(ctx, src0, src1, src2, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && src2->type == GGML_TYPE_I16 &&
               dst->type == GGML_TYPE_I16) {
        launch_bin_bcast3<op_सङ्कलय, int16_t, int16_t, int16_t, int16_t>(ctx, src0, src1, src2, dst);
#यदि_परिभाषितम् GGML_SYCL_HAS_BF16
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_सङ्कलय, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_सङ्कलय, sycl::ext::oneapi::bfloat16, भिन्न, भिन्न, sycl::ext::oneapi::bfloat16>(
            ctx, src0, src1, src2, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_सङ्कलय, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, भिन्न,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } अन्यथा यदि (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_सङ्कलय, sycl::ext::oneapi::bfloat16, भिन्न, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
#यदि_अन्तः
    } अन्यथा {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s, src2: %s\n", __func__,
                ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type),
                ggml_type_name(src2->type));
        GGML_ABORT("fatal error");
    }
}

