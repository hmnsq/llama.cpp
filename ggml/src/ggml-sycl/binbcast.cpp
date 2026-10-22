#समावेश "binbcast.hpp"

#समावेश <algorithm>
#समावेश <cstddef>
#समावेश <cstdint>
#समावेश <sycl/sycl.hpp>

#समावेश "ggml.h"

ढाँचा<भग्न (*bin_op)(स्थिर भग्न, स्थिर भग्न), typename src0_t, typename src1_t, typename dst_t>
अटल व्योम k_bin_bcast(स्थिर src0_t * src0, स्थिर src1_t * src1, dst_t * dst,
        पूर्णांक ne0, पूर्णांक ne1, पूर्णांक ne2, पूर्णांक ne3,
        पूर्णांक ne10, पूर्णांक ne11, पूर्णांक ne12, पूर्णांक ne13,
        /*int s0, */ पूर्णांक s1,  पूर्णांक s2,  पूर्णांक s3,
        पूर्णांक s00, पूर्णांक s01, पूर्णांक s02, पूर्णांक s03,
        पूर्णांक s10, पूर्णांक s11, पूर्णांक s12, पूर्णांक s13,
        स्थिर sycl::nd_item<3> &item_ct1) {
    स्थिर पूर्णांक i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    स्थिर पूर्णांक i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    स्थिर पूर्णांक i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    स्थिर पूर्णांक i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    अगर (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        वापस;
    }

    स्थिर पूर्णांक i11 = i1 % ne11;
    स्थिर पूर्णांक i12 = i2 % ne12;
    स्थिर पूर्णांक i13 = i3 % ne13;

    स्थिर माप_प्रकार i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    स्थिर माप_प्रकार i_src1 = i13*s13 + i12*s12 + i11*s11;
    स्थिर माप_प्रकार i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    स्थिर src0_t * src0_row = src0 + i_src0;
    स्थिर src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    क्रम (पूर्णांक i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        स्थिर पूर्णांक i10 = i0 % ne10;
        dst_row[i0] = (dst_t)bin_op(src0 ? (भग्न)src0_row[i0*s00] : 0.0f, (भग्न)src1_row[i10*s10]);
    }
}

ढाँचा<भग्न (*bin_op)(स्थिर भग्न, स्थिर भग्न), typename src0_t, typename src1_t, typename dst_t>
अटल व्योम k_bin_bcast_unravel(स्थिर src0_t * src0, स्थिर src1_t * src1, dst_t * dst,
        पूर्णांक ne0, पूर्णांक ne1, पूर्णांक ne2, पूर्णांक ne3,
        पूर्णांक ne10, पूर्णांक ne11, पूर्णांक ne12, पूर्णांक ne13,
        /*int s0, */ पूर्णांक s1,  पूर्णांक s2,  पूर्णांक s3,
        पूर्णांक s00, पूर्णांक s01, पूर्णांक s02, पूर्णांक s03,
        पूर्णांक s10, पूर्णांक s11, पूर्णांक s12, पूर्णांक s13,
        स्थिर sycl::nd_item<3> &item_ct1) {

    स्थिर पूर्णांक i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    स्थिर पूर्णांक i3 = i/(ne2*ne1*ne0);
    स्थिर पूर्णांक i2 = (i/(ne1*ne0)) % ne2;
    स्थिर पूर्णांक i1 = (i/ne0) % ne1;
    स्थिर पूर्णांक i0 = i % ne0;

    अगर (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        वापस;
    }

    स्थिर पूर्णांक i11 = i1 % ne11;
    स्थिर पूर्णांक i12 = i2 % ne12;
    स्थिर पूर्णांक i13 = i3 % ne13;

    स्थिर माप_प्रकार i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    स्थिर माप_प्रकार i_src1 = i13*s13 + i12*s12 + i11*s11;
    स्थिर माप_प्रकार i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    स्थिर src0_t * src0_row = src0 + i_src0;
    स्थिर src1_t * src1_row = src1 + i_src1;
    dst_t * dst_row = dst + i_dst;

    स्थिर पूर्णांक i10 = i0 % ne10;
    dst_row[i0] = (dst_t)bin_op(src0 ? (भग्न)src0_row[i0*s00] : 0.0f, (भग्न)src1_row[i10*s10]);
}


ढाँचा<भग्न (*bin_op)(स्थिर भग्न, स्थिर भग्न)>
काष्ठा bin_bcast_sycl {
    ढाँचा <typename src0_t, typename src1_t, typename dst_t>
    व्योम चालक()(स्थिर src0_t * src0_dd, स्थिर src1_t * src1_dd, dst_t * dst_dd, स्थिर int64_t ne00,
                    स्थिर int64_t ne01, स्थिर int64_t ne02, स्थिर int64_t ne03, स्थिर int64_t ne10, स्थिर int64_t ne11,
                    स्थिर int64_t ne12, स्थिर int64_t ne13, स्थिर int64_t ne0, स्थिर int64_t ne1, स्थिर int64_t ne2,
                    स्थिर int64_t ne3, स्थिर माप_प्रकार nb00, स्थिर माप_प्रकार nb01, स्थिर माप_प्रकार nb02, स्थिर माप_प्रकार nb03,
                    स्थिर माप_प्रकार nb10, स्थिर माप_प्रकार nb11, स्थिर माप_प्रकार nb12, स्थिर माप_प्रकार nb13, स्थिर माप_प्रकार nb0,
                    स्थिर माप_प्रकार nb1, स्थिर माप_प्रकार nb2, स्थिर माप_प्रकार nb3, स्थिर bool src0_is_contiguous,
                    स्थिर bool src1_is_contiguous, स्थिर bool src0_is_permuted, स्थिर bool src1_is_permuted,
                    queue_ptr stream) {
        पूर्णांक nr0 = ne10 / ne0;
        पूर्णांक nr1 = ne11/ne1;
        पूर्णांक nr2 = ne12/ne2;
        पूर्णांक nr3 = ne13/ne3;

        पूर्णांक nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne[] = {ne0, ne1, ne2, ne3};
        int64_t cne0[] = {ne00, ne01, ne02, ne03};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};
        माप_प्रकार cnb[] = {nb0, nb1, nb2, nb3};
        माप_प्रकार cnb0[] = {nb00, nb01, nb02, nb03};
        माप_प्रकार cnb1[] = {nb10, nb11, nb12, nb13};
        स्वतः collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        स्वतः collapse_nb = [](माप_प्रकार cnb[], int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        अगर (src0_is_contiguous && src1_is_contiguous && !src0_is_permuted && !src1_is_permuted) {
            क्रम (पूर्णांक i = 0; i < 4; i++) {
                अगर (nr[i] != 1) {
                    अवरोध;
                }
                अगर (i > 0) {
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

            माप_प्रकार nb0 = cnb[0];
            माप_प्रकार nb1 = cnb[1];
            माप_प्रकार nb2 = cnb[2];
            माप_प्रकार nb3 = cnb[3];

            माप_प्रकार nb00 = cnb0[0];
            माप_प्रकार nb01 = cnb0[1];
            माप_प्रकार nb02 = cnb0[2];
            माप_प्रकार nb03 = cnb0[3];

            माप_प्रकार nb10 = cnb1[0];
            माप_प्रकार nb11 = cnb1[1];
            माप_प्रकार nb12 = cnb1[2];
            माप_प्रकार nb13 = cnb1[3];

            // size_t s0 = nb0 / sizeof(dst_t);
            माप_प्रकार s1 = nb1 / माप(dst_t);
            माप_प्रकार s2 = nb2 / माप(dst_t);
            माप_प्रकार s3 = nb3 / माप(dst_t);

            माप_प्रकार s10 = nb10 / माप(src1_t);
            माप_प्रकार s11 = nb11 / माप(src1_t);
            माप_प्रकार s12 = nb12 / माप(src1_t);
            माप_प्रकार s13 = nb13 / माप(src1_t);

            माप_प्रकार s00 = nb00 / माप(src0_t);
            माप_प्रकार s01 = nb01 / माप(src0_t);
            माप_प्रकार s02 = nb02 / माप(src0_t);
            माप_प्रकार s03 = nb03 / माप(src0_t);

            GGML_UNUSED(s00);

            GGML_ASSERT(nb0 % माप(dst_t) == 0);
            GGML_ASSERT(nb1 % माप(dst_t) == 0);
            GGML_ASSERT(nb2 % माप(dst_t) == 0);
            GGML_ASSERT(nb3 % माप(dst_t) == 0);

            GGML_ASSERT(nb00 % माप(src0_t) == 0);
            GGML_ASSERT(nb01 % माप(src0_t) == 0);
            GGML_ASSERT(nb02 % माप(src0_t) == 0);
            GGML_ASSERT(nb03 % माप(src0_t) == 0);

            GGML_ASSERT(nb10 % माप(src1_t) == 0);
            GGML_ASSERT(nb11 % माप(src1_t) == 0);
            GGML_ASSERT(nb12 % माप(src1_t) == 0);
            GGML_ASSERT(nb13 % माप(src1_t) == 0);

            स्थिर पूर्णांक block_आकार_2 = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            sycl::range<3> block_dims(1, 1, 1);
            block_dims[2] = std::min<अचिन्हित पूर्णांक>(hne0, block_आकार_2);
            block_dims[1] = std::min<अचिन्हित पूर्णांक>(
                ne1, block_आकार_2 / (अचिन्हित पूर्णांक)block_dims[2]);
            block_dims[0] = std::min(
                std::min<अचिन्हित पूर्णांक>(
                    ne2 * ne3, block_आकार_2 / (अचिन्हित पूर्णांक)block_dims[2] /
                                   (अचिन्हित पूर्णांक)block_dims[1]),
                64U);

            sycl::range<3> block_nums(
                (ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                (ne1 + block_dims[1] - 1) / block_dims[1],
                (hne0 + block_dims[2] - 1) / block_dims[2]);

            अगर (block_nums[0] > 65535) {
                // this is the maximum number of blocks in z direction, fallback to 1D grid kernel
                पूर्णांक block_अंक = (ne0*ne1*ne2*ne3 + block_आकार_2 - 1) / block_आकार_2;
                {
                    dpct::has_capability_or_fail(stream->get_device(),
                                                 {sycl::aspect::fp16});

                    stream->parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, block_अंक) *
                                              sycl::range<3>(1, 1, block_आकार_2),
                                          sycl::range<3>(1, 1, block_आकार_2)),
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

ढाँचा <श्रेणी op>
अंतरभूत व्योम ggml_sycl_op_bin_bcast(ggml_backend_sycl_context & ctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1,
                                   ggml_tensor * dst) {
    dpct::queue_ptr main_stream = ctx.stream();
    GGML_TENSOR_द्विआधारी_OP_LOCALS

    अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()((स्थिर भग्न *) src0->तथ्य, (स्थिर भग्न *) src1->तथ्य, (भग्न *) dst->तथ्य, ne00, ne01, ne02, ne03, ne10,
             ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3,
             ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()((स्थिर sycl::half *) src0->तथ्य, (स्थिर sycl::half *) src1->तथ्य, (sycl::half *) dst->तथ्य, ne00, ne01,
             ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13,
             nb0, nb1, nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()((स्थिर sycl::half *) src0->तथ्य, (स्थिर भग्न *) src1->तथ्य, (sycl::half *) dst->तथ्य, ne00, ne01, ne02,
             ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1,
             nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } अन्यथा अगर (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()((स्थिर int32_t *) src0->तथ्य, (स्थिर int32_t *) src1->तथ्य, (int32_t *) dst->तथ्य, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
    } अन्यथा अगर (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()((स्थिर int16_t *) src0->तथ्य, (स्थिर int16_t *) src1->तथ्य, (int16_t *) dst->तथ्य, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_stream);
#अगर_घोषित GGML_SYCL_HAS_BF16
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        op()((स्थिर sycl::ext::oneapi::bfloat16 *) src0->तथ्य, (स्थिर sycl::ext::oneapi::bfloat16 *) src1->तथ्य,
             (sycl::ext::oneapi::bfloat16 *) dst->तथ्य, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        op()((स्थिर sycl::ext::oneapi::bfloat16 *) src0->तथ्य, (स्थिर भग्न *) src1->तथ्य,
             (sycl::ext::oneapi::bfloat16 *) dst->तथ्य, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_stream);
#पूर्ण_अगर
    } अन्यथा {
        ख_लिखो(मानक_त्रुटि, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(dst->type),
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

अंतरभूत व्योम ggml_sycl_op_add(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_add>>(ctx, dst->src[0], dst->src[1], dst);
}

अंतरभूत व्योम ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_sub>>(ctx, dst->src[0], dst->src[1], dst);
}

अंतरभूत व्योम ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(ctx, dst->src[0], dst->src[1], dst);
}

अंतरभूत व्योम ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(ctx, dst->src[0], dst->src[1], dst);
}

अंतरभूत व्योम ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(ctx, dst, dst->src[0], dst);
}


व्योम ggml_sycl_add(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_add(ctx, dst);
}

व्योम ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_sub(ctx, dst);
}

व्योम ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_mul(ctx, dst);
}

व्योम ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_div(ctx, dst);
}

व्योम ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat(ctx, dst);
}

// fused ADD+ADD: dst = (src0 + src1) + src2. Same indexing as k_bin_bcast, so mixed
// types, broadcast, and non-contiguous layouts that add() already handles also fuse.
ढाँचा<भग्न (*bin_op)(स्थिर भग्न, स्थिर भग्न), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
अटल व्योम k_bin_bcast3(स्थिर src0_t * src0, स्थिर src1_t * src1, स्थिर src2_t * src2, dst_t * dst,
        पूर्णांक ne0, पूर्णांक ne1, पूर्णांक ne2, पूर्णांक ne3,
        पूर्णांक ne10, पूर्णांक ne11, पूर्णांक ne12, पूर्णांक ne13,
        पूर्णांक ne20, पूर्णांक ne21, पूर्णांक ne22, पूर्णांक ne23,
        पूर्णांक s1, पूर्णांक s2, पूर्णांक s3,
        पूर्णांक s00, पूर्णांक s01, पूर्णांक s02, पूर्णांक s03,
        पूर्णांक s10, पूर्णांक s11, पूर्णांक s12, पूर्णांक s13,
        पूर्णांक s20, पूर्णांक s21, पूर्णांक s22, पूर्णांक s23,
        स्थिर sycl::nd_item<3> & item_ct1) {
    स्थिर पूर्णांक i0s = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                    item_ct1.get_local_id(2);
    स्थिर पूर्णांक i1 = (item_ct1.get_local_range(1) * item_ct1.get_group(1) +
                    item_ct1.get_local_id(1));
    स्थिर पूर्णांक i2 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) /
                   ne3;
    स्थिर पूर्णांक i3 = (item_ct1.get_local_range(0) * item_ct1.get_group(0) +
                    item_ct1.get_local_id(0)) %
                   ne3;

    अगर (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        वापस;
    }

    स्थिर पूर्णांक i11 = i1 % ne11;
    स्थिर पूर्णांक i12 = i2 % ne12;
    स्थिर पूर्णांक i13 = i3 % ne13;
    स्थिर पूर्णांक i21 = i1 % ne21;
    स्थिर पूर्णांक i22 = i2 % ne22;
    स्थिर पूर्णांक i23 = i3 % ne23;

    स्थिर माप_प्रकार i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    स्थिर माप_प्रकार i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    स्थिर माप_प्रकार i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    स्थिर माप_प्रकार i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    स्थिर src0_t * src0_row = src0 + i_src0;
    स्थिर src1_t * src1_row = src1 + i_src1;
    स्थिर src2_t * src2_row = src2 + i_src2;
    dst_t *        dst_row  = dst + i_dst;

    क्रम (पूर्णांक i0 = i0s; i0 < ne0;
         i0 += item_ct1.get_local_range(2) * item_ct1.get_group_range(2)) {
        स्थिर पूर्णांक   i10 = i0 % ne10;
        स्थिर पूर्णांक   i20 = i0 % ne20;
        स्थिर भग्न acc = bin_op((भग्न) src0_row[i0 * s00], (भग्न) src1_row[i10 * s10]);
        dst_row[i0]     = (dst_t) bin_op(acc, (भग्न) src2_row[i20 * s20]);
    }
}

ढाँचा<भग्न (*bin_op)(स्थिर भग्न, स्थिर भग्न), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
अटल व्योम k_bin_bcast3_unravel(स्थिर src0_t * src0, स्थिर src1_t * src1, स्थिर src2_t * src2, dst_t * dst,
        पूर्णांक ne0, पूर्णांक ne1, पूर्णांक ne2, पूर्णांक ne3,
        पूर्णांक ne10, पूर्णांक ne11, पूर्णांक ne12, पूर्णांक ne13,
        पूर्णांक ne20, पूर्णांक ne21, पूर्णांक ne22, पूर्णांक ne23,
        पूर्णांक s1, पूर्णांक s2, पूर्णांक s3,
        पूर्णांक s00, पूर्णांक s01, पूर्णांक s02, पूर्णांक s03,
        पूर्णांक s10, पूर्णांक s11, पूर्णांक s12, पूर्णांक s13,
        पूर्णांक s20, पूर्णांक s21, पूर्णांक s22, पूर्णांक s23,
        स्थिर sycl::nd_item<3> & item_ct1) {
    स्थिर पूर्णांक i = item_ct1.get_local_range(2) * item_ct1.get_group(2) +
                  item_ct1.get_local_id(2);

    स्थिर पूर्णांक i3 = i / (ne2 * ne1 * ne0);
    स्थिर पूर्णांक i2 = (i / (ne1 * ne0)) % ne2;
    स्थिर पूर्णांक i1 = (i / ne0) % ne1;
    स्थिर पूर्णांक i0 = i % ne0;

    अगर (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        वापस;
    }

    स्थिर पूर्णांक i11 = i1 % ne11;
    स्थिर पूर्णांक i12 = i2 % ne12;
    स्थिर पूर्णांक i13 = i3 % ne13;
    स्थिर पूर्णांक i21 = i1 % ne21;
    स्थिर पूर्णांक i22 = i2 % ne22;
    स्थिर पूर्णांक i23 = i3 % ne23;

    स्थिर माप_प्रकार i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    स्थिर माप_प्रकार i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    स्थिर माप_प्रकार i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    स्थिर माप_प्रकार i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    स्थिर पूर्णांक   i10 = i0 % ne10;
    स्थिर पूर्णांक   i20 = i0 % ne20;
    स्थिर भग्न acc = bin_op((भग्न) src0[i_src0 + i0 * s00], (भग्न) src1[i_src1 + i10 * s10]);
    dst[i_dst + i0] = (dst_t) bin_op(acc, (भग्न) src2[i_src2 + i20 * s20]);
}

ढाँचा<भग्न (*bin_op)(स्थिर भग्न, स्थिर भग्न), typename src0_t, typename src1_t, typename src2_t, typename dst_t>
अटल व्योम launch_bin_bcast3(ggml_backend_sycl_context & ctx, स्थिर ggml_tensor * src0, स्थिर ggml_tensor * src1,
                              स्थिर ggml_tensor * src2, ggml_tensor * dst) {
    dpct::queue_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_set_device(ctx.device));

    GGML_TENSOR_TERNARY_OP_LOCALS

    पूर्णांक nr1[4] = { (पूर्णांक) (ne10 / ne0), (पूर्णांक) (ne11 / ne1), (पूर्णांक) (ne12 / ne2), (पूर्णांक) (ne13 / ne3) };
    पूर्णांक nr2[4] = { (पूर्णांक) (ne20 / ne0), (पूर्णांक) (ne21 / ne1), (पूर्णांक) (ne22 / ne2), (पूर्णांक) (ne23 / ne3) };

    int64_t cne[]  = { ne0, ne1, ne2, ne3 };
    int64_t cne0[] = { ne00, ne01, ne02, ne03 };
    int64_t cne1[] = { ne10, ne11, ne12, ne13 };
    int64_t cne2[] = { ne20, ne21, ne22, ne23 };
    माप_प्रकार  cnb[]  = { nb0, nb1, nb2, nb3 };
    माप_प्रकार  cnb0[] = { nb00, nb01, nb02, nb03 };
    माप_प्रकार  cnb1[] = { nb10, nb11, nb12, nb13 };
    माप_प्रकार  cnb2[] = { nb20, nb21, nb22, nb23 };

    स्वतः collapse = [](int64_t cne[]) {
        cne[0] *= cne[1];
        cne[1] = cne[2];
        cne[2] = cne[3];
        cne[3] = 1;
    };

    स्वतः collapse_nb = [](माप_प्रकार cnb[], int64_t cne[]) {
        cnb[1] *= cne[1];
        cnb[2] *= cne[2];
        cnb[3] *= cne[3];
    };

    स्थिर bool can_collapse = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(src2) &&
                              !ggml_is_permuted(src0) && !ggml_is_permuted(src1) && !ggml_is_permuted(src2);
    अगर (can_collapse) {
        क्रम (पूर्णांक i = 0; i < 4; i++) {
            अगर (nr1[i] != 1 || nr2[i] != 1) {
                अवरोध;
            }
            अगर (i > 0) {
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

        माप_प्रकार s1 = cnb[1] / माप(dst_t);
        माप_प्रकार s2 = cnb[2] / माप(dst_t);
        माप_प्रकार s3 = cnb[3] / माप(dst_t);

        माप_प्रकार s00 = cnb0[0] / माप(src0_t);
        माप_प्रकार s01 = cnb0[1] / माप(src0_t);
        माप_प्रकार s02 = cnb0[2] / माप(src0_t);
        माप_प्रकार s03 = cnb0[3] / माप(src0_t);

        माप_प्रकार s10 = cnb1[0] / माप(src1_t);
        माप_प्रकार s11 = cnb1[1] / माप(src1_t);
        माप_प्रकार s12 = cnb1[2] / माप(src1_t);
        माप_प्रकार s13 = cnb1[3] / माप(src1_t);

        माप_प्रकार s20 = cnb2[0] / माप(src2_t);
        माप_प्रकार s21 = cnb2[1] / माप(src2_t);
        माप_प्रकार s22 = cnb2[2] / माप(src2_t);
        माप_प्रकार s23 = cnb2[3] / माप(src2_t);

        GGML_ASSERT(cnb[0] % माप(dst_t) == 0 && cnb[1] % माप(dst_t) == 0 && cnb[2] % माप(dst_t) == 0 &&
                    cnb[3] % माप(dst_t) == 0);
        GGML_ASSERT(cnb0[0] % माप(src0_t) == 0 && cnb0[1] % माप(src0_t) == 0 && cnb0[2] % माप(src0_t) == 0 &&
                    cnb0[3] % माप(src0_t) == 0);
        GGML_ASSERT(cnb1[0] % माप(src1_t) == 0 && cnb1[1] % माप(src1_t) == 0 && cnb1[2] % माप(src1_t) == 0 &&
                    cnb1[3] % माप(src1_t) == 0);
        GGML_ASSERT(cnb2[0] % माप(src2_t) == 0 && cnb2[1] % माप(src2_t) == 0 && cnb2[2] % माप(src2_t) == 0 &&
                    cnb2[3] % माप(src2_t) == 0);

        स्थिर src0_t * src0_dd = (स्थिर src0_t *) src0->तथ्य;
        स्थिर src1_t * src1_dd = (स्थिर src1_t *) src1->तथ्य;
        स्थिर src2_t * src2_dd = (स्थिर src2_t *) src2->तथ्य;
        dst_t *        dst_dd  = (dst_t *) dst->तथ्य;

        स्थिर पूर्णांक block_आकार_2 = 128;
        int64_t   hne0       = std::max(ne0 / 2LL, 1LL);

        sycl::range<3> block_dims(1, 1, 1);
        block_dims[2] = std::min<अचिन्हित पूर्णांक>(hne0, block_आकार_2);
        block_dims[1] = std::min<अचिन्हित पूर्णांक>(ne1, block_आकार_2 / (अचिन्हित पूर्णांक) block_dims[2]);
        block_dims[0] = std::min(std::min<अचिन्हित पूर्णांक>(ne2 * ne3,
                                                        block_आकार_2 / (अचिन्हित पूर्णांक) block_dims[2] /
                                                            (अचिन्हित पूर्णांक) block_dims[1]),
                                 64U);

        sycl::range<3> block_nums((ne2 * ne3 + block_dims[0] - 1) / block_dims[0],
                                  (ne1 + block_dims[1] - 1) / block_dims[1],
                                  (hne0 + block_dims[2] - 1) / block_dims[2]);

        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        अगर (block_nums[0] > 65535) {
            पूर्णांक block_अंक = (ne0 * ne1 * ne2 * ne3 + block_आकार_2 - 1) / block_आकार_2;
            stream->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, block_अंक) * sycl::range<3>(1, 1, block_आकार_2),
                                  sycl::range<3>(1, 1, block_आकार_2)),
                [=](sycl::nd_item<3> item_ct1) {
                    k_bin_bcast3_unravel<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10, ne11,
                                                 ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00, s01, s02, s03,
                                                 s10, s11, s12, s13, s20, s21, s22, s23, item_ct1);
                });
        } अन्यथा {
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

व्योम ggml_sycl_op_add_add_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1) {
    स्थिर ggml_tensor * src0 = add0->src[0];
    स्थिर ggml_tensor * src1 = add0->src[1];
    स्थिर ggml_tensor * src2 = add1->src[1];
    ggml_tensor *       dst  = add1;

    GGML_ASSERT(add1->src[0] == add0);
    GGML_ASSERT(ggml_sycl_add_kernel_supports(src0->type, src1->type, add0->type));
    GGML_ASSERT(ggml_sycl_add_kernel_supports(add0->type, src2->type, dst->type));

    अगर (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        launch_bin_bcast3<op_add, भग्न, भग्न, भग्न, भग्न>(ctx, src0, src1, src2, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, sycl::half, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, भग्न, भग्न, sycl::half>(ctx, src0, src1, src2, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, sycl::half, भग्न, sycl::half>(ctx, src0, src1, src2, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_add, sycl::half, भग्न, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && src2->type == GGML_TYPE_I32 &&
               dst->type == GGML_TYPE_I32) {
        launch_bin_bcast3<op_add, int32_t, int32_t, int32_t, int32_t>(ctx, src0, src1, src2, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && src2->type == GGML_TYPE_I16 &&
               dst->type == GGML_TYPE_I16) {
        launch_bin_bcast3<op_add, int16_t, int16_t, int16_t, int16_t>(ctx, src0, src1, src2, dst);
#अगर_घोषित GGML_SYCL_HAS_BF16
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, भग्न, भग्न, sycl::ext::oneapi::bfloat16>(
            ctx, src0, src1, src2, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, भग्न,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } अन्यथा अगर (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_add, sycl::ext::oneapi::bfloat16, भग्न, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
#पूर्ण_अगर
    } अन्यथा {
        ख_लिखो(मानक_त्रुटि, "%s: unsupported types: dst: %s, src0: %s, src1: %s, src2: %s\n", __func__,
                ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type),
                ggml_type_name(src2->type));
        GGML_ABORT("fatal error");
    }
}

