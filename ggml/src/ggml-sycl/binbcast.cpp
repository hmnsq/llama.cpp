#అన్తర్భావ "binbcast.hpp"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cstddef>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <sycl/sycl.hpp>

#అన్తర్భావ "ggml.h"

మూస<దశాంశం (*bin_op)(స్థిరరాశి దశాంశం, స్థిరరాశి దశాంశం), రకం_పేరు src0_t, రకం_పేరు src1_t, రకం_పేరు dst_t>
స్థిర శూన్య k_bin_bcast(స్థిరరాశి src0_t * src0, స్థిరరాశి src1_t * src1, dst_t * dst,
        పూర్ణాంకం ne0, పూర్ణాంకం ne1, పూర్ణాంకం ne2, పూర్ణాంకం ne3,
        పూర్ణాంకం ne10, పూర్ణాంకం ne11, పూర్ణాంకం ne12, పూర్ణాంకం ne13,
        /*int s0, */ పూర్ణాంకం s1,  పూర్ణాంకం s2,  పూర్ణాంకం s3,
        పూర్ణాంకం s00, పూర్ణాంకం s01, పూర్ణాంకం s02, పూర్ణాంకం s03,
        పూర్ణాంకం s10, పూర్ణాంకం s11, పూర్ణాంకం s12, పూర్ణాంకం s13,
        స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం i0s = అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం(2) +
                    అంశం_ct1.get_local_id(2);
    స్థిరరాశి పూర్ణాంకం i1 = (అంశం_ct1.get_local_range(1) * అంశం_ct1.get_సమూహం(1) +
                    అంశం_ct1.get_local_id(1));
    స్థిరరాశి పూర్ణాంకం i2 = (అంశం_ct1.get_local_range(0) * అంశం_ct1.get_సమూహం(0) +
                    అంశం_ct1.get_local_id(0)) /
                   ne3;
    స్థిరరాశి పూర్ణాంకం i3 = (అంశం_ct1.get_local_range(0) * అంశం_ct1.get_సమూహం(0) +
                    అంశం_ct1.get_local_id(0)) %
                   ne3;

    ఒకవేళ (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం i11 = i1 % ne11;
    స్థిరరాశి పూర్ణాంకం i12 = i2 % ne12;
    స్థిరరాశి పూర్ణాంకం i13 = i3 % ne13;

    స్థిరరాశి size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    స్థిరరాశి size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    స్థిరరాశి size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    స్థిరరాశి src0_t * src0_అడ్డువరుస = src0 + i_src0;
    స్థిరరాశి src1_t * src1_అడ్డువరుస = src1 + i_src1;
    dst_t * dst_అడ్డువరుస = dst + i_dst;

    కొరకు (పూర్ణాంకం i0 = i0s; i0 < ne0;
         i0 += అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం_range(2)) {
        స్థిరరాశి పూర్ణాంకం i10 = i0 % ne10;
        dst_అడ్డువరుస[i0] = (dst_t)bin_op(src0 ? (దశాంశం)src0_అడ్డువరుస[i0*s00] : 0.0f, (దశాంశం)src1_అడ్డువరుస[i10*s10]);
    }
}

మూస<దశాంశం (*bin_op)(స్థిరరాశి దశాంశం, స్థిరరాశి దశాంశం), రకం_పేరు src0_t, రకం_పేరు src1_t, రకం_పేరు dst_t>
స్థిర శూన్య k_bin_bcast_unravel(స్థిరరాశి src0_t * src0, స్థిరరాశి src1_t * src1, dst_t * dst,
        పూర్ణాంకం ne0, పూర్ణాంకం ne1, పూర్ణాంకం ne2, పూర్ణాంకం ne3,
        పూర్ణాంకం ne10, పూర్ణాంకం ne11, పూర్ణాంకం ne12, పూర్ణాంకం ne13,
        /*int s0, */ పూర్ణాంకం s1,  పూర్ణాంకం s2,  పూర్ణాంకం s3,
        పూర్ణాంకం s00, పూర్ణాంకం s01, పూర్ణాంకం s02, పూర్ణాంకం s03,
        పూర్ణాంకం s10, పూర్ణాంకం s11, పూర్ణాంకం s12, పూర్ణాంకం s13,
        స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి పూర్ణాంకం i = అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం(2) +
                  అంశం_ct1.get_local_id(2);

    స్థిరరాశి పూర్ణాంకం i3 = i/(ne2*ne1*ne0);
    స్థిరరాశి పూర్ణాంకం i2 = (i/(ne1*ne0)) % ne2;
    స్థిరరాశి పూర్ణాంకం i1 = (i/ne0) % ne1;
    స్థిరరాశి పూర్ణాంకం i0 = i % ne0;

    ఒకవేళ (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం i11 = i1 % ne11;
    స్థిరరాశి పూర్ణాంకం i12 = i2 % ne12;
    స్థిరరాశి పూర్ణాంకం i13 = i3 % ne13;

    స్థిరరాశి size_t i_src0 =  i3*s03 +  i2*s02 +  i1*s01;
    స్థిరరాశి size_t i_src1 = i13*s13 + i12*s12 + i11*s11;
    స్థిరరాశి size_t i_dst  =  i3*s3  +  i2*s2  +  i1*s1;

    స్థిరరాశి src0_t * src0_అడ్డువరుస = src0 + i_src0;
    స్థిరరాశి src1_t * src1_అడ్డువరుస = src1 + i_src1;
    dst_t * dst_అడ్డువరుస = dst + i_dst;

    స్థిరరాశి పూర్ణాంకం i10 = i0 % ne10;
    dst_అడ్డువరుస[i0] = (dst_t)bin_op(src0 ? (దశాంశం)src0_అడ్డువరుస[i0*s00] : 0.0f, (దశాంశం)src1_అడ్డువరుస[i10*s10]);
}


మూస<దశాంశం (*bin_op)(స్థిరరాశి దశాంశం, స్థిరరాశి దశాంశం)>
నిర్మాణం bin_bcast_sycl {
    మూస <రకం_పేరు src0_t, రకం_పేరు src1_t, రకం_పేరు dst_t>
    శూన్య క్రియాకారకం()(స్థిరరాశి src0_t * src0_dd, స్థిరరాశి src1_t * src1_dd, dst_t * dst_dd, స్థిరరాశి int64_t ne00,
                    స్థిరరాశి int64_t ne01, స్థిరరాశి int64_t ne02, స్థిరరాశి int64_t ne03, స్థిరరాశి int64_t ne10, స్థిరరాశి int64_t ne11,
                    స్థిరరాశి int64_t ne12, స్థిరరాశి int64_t ne13, స్థిరరాశి int64_t ne0, స్థిరరాశి int64_t ne1, స్థిరరాశి int64_t ne2,
                    స్థిరరాశి int64_t ne3, స్థిరరాశి size_t nb00, స్థిరరాశి size_t nb01, స్థిరరాశి size_t nb02, స్థిరరాశి size_t nb03,
                    స్థిరరాశి size_t nb10, స్థిరరాశి size_t nb11, స్థిరరాశి size_t nb12, స్థిరరాశి size_t nb13, స్థిరరాశి size_t nb0,
                    స్థిరరాశి size_t nb1, స్థిరరాశి size_t nb2, స్థిరరాశి size_t nb3, స్థిరరాశి తార్కిక src0_is_contiguous,
                    స్థిరరాశి తార్కిక src1_is_contiguous, స్థిరరాశి తార్కిక src0_is_permuted, స్థిరరాశి తార్కిక src1_is_permuted,
                    వరుస_ptr stream) {
        పూర్ణాంకం nr0 = ne10 / ne0;
        పూర్ణాంకం nr1 = ne11/ne1;
        పూర్ణాంకం nr2 = ne12/ne2;
        పూర్ణాంకం nr3 = ne13/ne3;

        పూర్ణాంకం nr[4] = { nr0, nr1, nr2, nr3 };

        // collapse dimensions until first broadcast dimension
        int64_t cne[] = {ne0, ne1, ne2, ne3};
        int64_t cne0[] = {ne00, ne01, ne02, ne03};
        int64_t cne1[] = {ne10, ne11, ne12, ne13};
        size_t cnb[] = {nb0, nb1, nb2, nb3};
        size_t cnb0[] = {nb00, nb01, nb02, nb03};
        size_t cnb1[] = {nb10, nb11, nb12, nb13};
        స్వయంచాలక collapse = [](int64_t cne[]) {
            cne[0] *= cne[1];
            cne[1] = cne[2];
            cne[2] = cne[3];
            cne[3] = 1;
        };

        స్వయంచాలక collapse_nb = [](size_t cnb[], int64_t cne[]) {
            cnb[1] *= cne[1];
            cnb[2] *= cne[2];
            cnb[3] *= cne[3];
        };

        ఒకవేళ (src0_is_contiguous && src1_is_contiguous && !src0_is_permuted && !src1_is_permuted) {
            కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
                ఒకవేళ (nr[i] != 1) {
                    ఆపు;
                }
                ఒకవేళ (i > 0) {
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
            size_t s1 = nb1 / పరిమాణం(dst_t);
            size_t s2 = nb2 / పరిమాణం(dst_t);
            size_t s3 = nb3 / పరిమాణం(dst_t);

            size_t s10 = nb10 / పరిమాణం(src1_t);
            size_t s11 = nb11 / పరిమాణం(src1_t);
            size_t s12 = nb12 / పరిమాణం(src1_t);
            size_t s13 = nb13 / పరిమాణం(src1_t);

            size_t s00 = nb00 / పరిమాణం(src0_t);
            size_t s01 = nb01 / పరిమాణం(src0_t);
            size_t s02 = nb02 / పరిమాణం(src0_t);
            size_t s03 = nb03 / పరిమాణం(src0_t);

            GGML_వాడని(s00);

            GGML_ASSERT(nb0 % పరిమాణం(dst_t) == 0);
            GGML_ASSERT(nb1 % పరిమాణం(dst_t) == 0);
            GGML_ASSERT(nb2 % పరిమాణం(dst_t) == 0);
            GGML_ASSERT(nb3 % పరిమాణం(dst_t) == 0);

            GGML_ASSERT(nb00 % పరిమాణం(src0_t) == 0);
            GGML_ASSERT(nb01 % పరిమాణం(src0_t) == 0);
            GGML_ASSERT(nb02 % పరిమాణం(src0_t) == 0);
            GGML_ASSERT(nb03 % పరిమాణం(src0_t) == 0);

            GGML_ASSERT(nb10 % పరిమాణం(src1_t) == 0);
            GGML_ASSERT(nb11 % పరిమాణం(src1_t) == 0);
            GGML_ASSERT(nb12 % పరిమాణం(src1_t) == 0);
            GGML_ASSERT(nb13 % పరిమాణం(src1_t) == 0);

            స్థిరరాశి పూర్ణాంకం ఖండం_కొలత_3 = 128;

            int64_t hne0 = std::max(ne0/2LL, 1LL);

            sycl::range<3> ఖండం_dims(1, 1, 1);
            ఖండం_dims[2] = std::min<గుర్తులేని పూర్ణాంకం>(hne0, ఖండం_కొలత_3);
            ఖండం_dims[1] = std::min<గుర్తులేని పూర్ణాంకం>(
                ne1, ఖండం_కొలత_3 / (గుర్తులేని పూర్ణాంకం)ఖండం_dims[2]);
            ఖండం_dims[0] = std::min(
                std::min<గుర్తులేని పూర్ణాంకం>(
                    ne2 * ne3, ఖండం_కొలత_3 / (గుర్తులేని పూర్ణాంకం)ఖండం_dims[2] /
                                   (గుర్తులేని పూర్ణాంకం)ఖండం_dims[1]),
                64U);

            sycl::range<3> ఖండం_nums(
                (ne2 * ne3 + ఖండం_dims[0] - 1) / ఖండం_dims[0],
                (ne1 + ఖండం_dims[1] - 1) / ఖండం_dims[1],
                (hne0 + ఖండం_dims[2] - 1) / ఖండం_dims[2]);

            ఒకవేళ (ఖండం_nums[0] > 65535) {
                // this is the maximum number of blocks in z direction, fallback to 1D grid kernel
                పూర్ణాంకం ఖండం_అంకె = (ne0*ne1*ne2*ne3 + ఖండం_కొలత_3 - 1) / ఖండం_కొలత_3;
                {
                    dpct::has_capability_or_fail(stream->get_device(),
                                                 {sycl::aspect::fp16});

                    stream->parallel_for(
                        sycl::nd_range<3>(sycl::range<3>(1, 1, ఖండం_అంకె) *
                                              sycl::range<3>(1, 1, ఖండం_కొలత_3),
                                          sycl::range<3>(1, 1, ఖండం_కొలత_3)),
                        [=](sycl::nd_అంశం<3> అంశం_ct1) {
                            k_bin_bcast_unravel<bin_op>(
                                src0_dd, src1_dd, dst_dd, ne0, ne1, ne2, ne3,
                                ne10, ne11, ne12, ne13, s1, s2, s3, s00, s01, s02,
                                s03, s10, s11, s12, s13, అంశం_ct1);
                        });
                }
            } లేకపోతే {
                /*
                DPCT1049:16: The work-group size passed to the SYCL kernel may
                exceed the limit. To get the device limit, query
                info::device::max_work_group_size. Adjust the work-group size if
                needed.
                */
                dpct::has_capability_or_fail(stream->get_device(),
                                             {sycl::aspect::fp16});

                stream->parallel_for(
                    sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                    [=](sycl::nd_అంశం<3> అంశం_ct1) {
                        k_bin_bcast<bin_op>(src0_dd, src1_dd, dst_dd, ne0, ne1,
                                            ne2, ne3, ne10, ne11, ne12, ne13,
                                            s1, s2, s3, s00, s01, s02, s03, s10, s11, s12, s13,
                                            అంశం_ct1);
                    });
            }
        }
    }
};

మూస <తరగతి op>
అంతఃవరుస శూన్య ggml_sycl_op_bin_bcast(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1,
                                   ggml_tensor * dst) {
    dpct::వరుస_ptr main_ప్రవాహం = ctx.stream();
    GGML_TENSOR_ద్విమాన_OP_LOCALS

    ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        op()((స్థిరరాశి దశాంశం *) src0->దత్తాంశం, (స్థిరరాశి దశాంశం *) src1->దత్తాంశం, (దశాంశం *) dst->దత్తాంశం, ne00, ne01, ne02, ne03, ne10,
             ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3,
             ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_ప్రవాహం);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && dst->type == GGML_TYPE_F16) {
        op()((స్థిరరాశి sycl::half *) src0->దత్తాంశం, (స్థిరరాశి sycl::half *) src1->దత్తాంశం, (sycl::half *) dst->దత్తాంశం, ne00, ne01,
             ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13,
             nb0, nb1, nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_ప్రవాహం);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F16) {
        op()((స్థిరరాశి sycl::half *) src0->దత్తాంశం, (స్థిరరాశి దశాంశం *) src1->దత్తాంశం, (sycl::half *) dst->దత్తాంశం, ne00, ne01, ne02,
             ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1,
             nb2, nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_ప్రవాహం);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_I32) {
        op()((స్థిరరాశి int32_t *) src0->దత్తాంశం, (స్థిరరాశి int32_t *) src1->దత్తాంశం, (int32_t *) dst->దత్తాంశం, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_ప్రవాహం);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && dst->type == GGML_TYPE_I16) {
        op()((స్థిరరాశి int16_t *) src0->దత్తాంశం, (స్థిరరాశి int16_t *) src1->దత్తాంశం, (int16_t *) dst->దత్తాంశం, ne00, ne01, ne02, ne03,
             ne10, ne11, ne12, ne13, ne0, ne1, ne2, ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2,
             nb3, ggml_is_contiguous(src0), ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1),
             main_ప్రవాహం);
#నిర్వచితమైతే GGML_SYCL_HAS_BF16
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && dst->type == GGML_TYPE_BF16) {
        op()((స్థిరరాశి sycl::ext::oneapi::bfloat16 *) src0->దత్తాంశం, (స్థిరరాశి sycl::ext::oneapi::bfloat16 *) src1->దత్తాంశం,
             (sycl::ext::oneapi::bfloat16 *) dst->దత్తాంశం, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_ప్రవాహం);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_BF16) {
        op()((స్థిరరాశి sycl::ext::oneapi::bfloat16 *) src0->దత్తాంశం, (స్థిరరాశి దశాంశం *) src1->దత్తాంశం,
             (sycl::ext::oneapi::bfloat16 *) dst->దత్తాంశం, ne00, ne01, ne02, ne03, ne10, ne11, ne12, ne13, ne0, ne1, ne2,
             ne3, nb00, nb01, nb02, nb03, nb10, nb11, nb12, nb13, nb0, nb1, nb2, nb3, ggml_is_contiguous(src0),
             ggml_is_contiguous(src1), ggml_is_permuted(src0), ggml_is_permuted(src1), main_ప్రవాహం);
#అయితే_ముగింపు
    } లేకపోతే {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s\n", __func__, ggml_type_name(dst->type),
                ggml_type_name(src0->type), ggml_type_name(src1->type));
        GGML_ABORT("fatal error");
    }
}

అంతఃవరుస శూన్య ggml_sycl_op_కూడు(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_కూడు>>(ctx, dst->src[0], dst->src[1], dst);
}

అంతఃవరుస శూన్య ggml_sycl_op_sub(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_sub>>(ctx, dst->src[0], dst->src[1], dst);
}

అంతఃవరుస శూన్య ggml_sycl_op_mul(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_mul>>(ctx, dst->src[0], dst->src[1], dst);
}

అంతఃవరుస శూన్య ggml_sycl_op_div(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {

    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_div>>(ctx, dst->src[0], dst->src[1], dst);
}

అంతఃవరుస శూన్య ggml_sycl_op_repeat(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    ggml_sycl_op_bin_bcast<bin_bcast_sycl<op_repeat>>(ctx, dst, dst->src[0], dst);
}


శూన్య ggml_sycl_కూడు(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_కూడు(ctx, dst);
}

శూన్య ggml_sycl_sub(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_sub(ctx, dst);
}

శూన్య ggml_sycl_mul(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_mul(ctx, dst);
}

శూన్య ggml_sycl_div(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ggml_sycl_op_div(ctx, dst);
}

శూన్య ggml_sycl_repeat(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_repeat(ctx, dst);
}

// fused ADD+ADD: dst = (src0 + src1) + src2. Same indexing as k_bin_bcast, so mixed
// types, broadcast, and non-contiguous layouts that add() already handles also fuse.
మూస<దశాంశం (*bin_op)(స్థిరరాశి దశాంశం, స్థిరరాశి దశాంశం), రకం_పేరు src0_t, రకం_పేరు src1_t, రకం_పేరు src2_t, రకం_పేరు dst_t>
స్థిర శూన్య k_bin_bcast3(స్థిరరాశి src0_t * src0, స్థిరరాశి src1_t * src1, స్థిరరాశి src2_t * src2, dst_t * dst,
        పూర్ణాంకం ne0, పూర్ణాంకం ne1, పూర్ణాంకం ne2, పూర్ణాంకం ne3,
        పూర్ణాంకం ne10, పూర్ణాంకం ne11, పూర్ణాంకం ne12, పూర్ణాంకం ne13,
        పూర్ణాంకం ne20, పూర్ణాంకం ne21, పూర్ణాంకం ne22, పూర్ణాంకం ne23,
        పూర్ణాంకం s1, పూర్ణాంకం s2, పూర్ణాంకం s3,
        పూర్ణాంకం s00, పూర్ణాంకం s01, పూర్ణాంకం s02, పూర్ణాంకం s03,
        పూర్ణాంకం s10, పూర్ణాంకం s11, పూర్ణాంకం s12, పూర్ణాంకం s13,
        పూర్ణాంకం s20, పూర్ణాంకం s21, పూర్ణాంకం s22, పూర్ణాంకం s23,
        స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం i0s = అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం(2) +
                    అంశం_ct1.get_local_id(2);
    స్థిరరాశి పూర్ణాంకం i1 = (అంశం_ct1.get_local_range(1) * అంశం_ct1.get_సమూహం(1) +
                    అంశం_ct1.get_local_id(1));
    స్థిరరాశి పూర్ణాంకం i2 = (అంశం_ct1.get_local_range(0) * అంశం_ct1.get_సమూహం(0) +
                    అంశం_ct1.get_local_id(0)) /
                   ne3;
    స్థిరరాశి పూర్ణాంకం i3 = (అంశం_ct1.get_local_range(0) * అంశం_ct1.get_సమూహం(0) +
                    అంశం_ct1.get_local_id(0)) %
                   ne3;

    ఒకవేళ (i0s >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం i11 = i1 % ne11;
    స్థిరరాశి పూర్ణాంకం i12 = i2 % ne12;
    స్థిరరాశి పూర్ణాంకం i13 = i3 % ne13;
    స్థిరరాశి పూర్ణాంకం i21 = i1 % ne21;
    స్థిరరాశి పూర్ణాంకం i22 = i2 % ne22;
    స్థిరరాశి పూర్ణాంకం i23 = i3 % ne23;

    స్థిరరాశి size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    స్థిరరాశి size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    స్థిరరాశి size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    స్థిరరాశి size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    స్థిరరాశి src0_t * src0_అడ్డువరుస = src0 + i_src0;
    స్థిరరాశి src1_t * src1_అడ్డువరుస = src1 + i_src1;
    స్థిరరాశి src2_t * src2_అడ్డువరుస = src2 + i_src2;
    dst_t *        dst_అడ్డువరుస  = dst + i_dst;

    కొరకు (పూర్ణాంకం i0 = i0s; i0 < ne0;
         i0 += అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం_range(2)) {
        స్థిరరాశి పూర్ణాంకం   i10 = i0 % ne10;
        స్థిరరాశి పూర్ణాంకం   i20 = i0 % ne20;
        స్థిరరాశి దశాంశం acc = bin_op((దశాంశం) src0_అడ్డువరుస[i0 * s00], (దశాంశం) src1_అడ్డువరుస[i10 * s10]);
        dst_అడ్డువరుస[i0]     = (dst_t) bin_op(acc, (దశాంశం) src2_అడ్డువరుస[i20 * s20]);
    }
}

మూస<దశాంశం (*bin_op)(స్థిరరాశి దశాంశం, స్థిరరాశి దశాంశం), రకం_పేరు src0_t, రకం_పేరు src1_t, రకం_పేరు src2_t, రకం_పేరు dst_t>
స్థిర శూన్య k_bin_bcast3_unravel(స్థిరరాశి src0_t * src0, స్థిరరాశి src1_t * src1, స్థిరరాశి src2_t * src2, dst_t * dst,
        పూర్ణాంకం ne0, పూర్ణాంకం ne1, పూర్ణాంకం ne2, పూర్ణాంకం ne3,
        పూర్ణాంకం ne10, పూర్ణాంకం ne11, పూర్ణాంకం ne12, పూర్ణాంకం ne13,
        పూర్ణాంకం ne20, పూర్ణాంకం ne21, పూర్ణాంకం ne22, పూర్ణాంకం ne23,
        పూర్ణాంకం s1, పూర్ణాంకం s2, పూర్ణాంకం s3,
        పూర్ణాంకం s00, పూర్ణాంకం s01, పూర్ణాంకం s02, పూర్ణాంకం s03,
        పూర్ణాంకం s10, పూర్ణాంకం s11, పూర్ణాంకం s12, పూర్ణాంకం s13,
        పూర్ణాంకం s20, పూర్ణాంకం s21, పూర్ణాంకం s22, పూర్ణాంకం s23,
        స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {
    స్థిరరాశి పూర్ణాంకం i = అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం(2) +
                  అంశం_ct1.get_local_id(2);

    స్థిరరాశి పూర్ణాంకం i3 = i / (ne2 * ne1 * ne0);
    స్థిరరాశి పూర్ణాంకం i2 = (i / (ne1 * ne0)) % ne2;
    స్థిరరాశి పూర్ణాంకం i1 = (i / ne0) % ne1;
    స్థిరరాశి పూర్ణాంకం i0 = i % ne0;

    ఒకవేళ (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం i11 = i1 % ne11;
    స్థిరరాశి పూర్ణాంకం i12 = i2 % ne12;
    స్థిరరాశి పూర్ణాంకం i13 = i3 % ne13;
    స్థిరరాశి పూర్ణాంకం i21 = i1 % ne21;
    స్థిరరాశి పూర్ణాంకం i22 = i2 % ne22;
    స్థిరరాశి పూర్ణాంకం i23 = i3 % ne23;

    స్థిరరాశి size_t i_src0 = i3 * s03 + i2 * s02 + i1 * s01;
    స్థిరరాశి size_t i_src1 = i13 * s13 + i12 * s12 + i11 * s11;
    స్థిరరాశి size_t i_src2 = i23 * s23 + i22 * s22 + i21 * s21;
    స్థిరరాశి size_t i_dst  = i3 * s3 + i2 * s2 + i1 * s1;

    స్థిరరాశి పూర్ణాంకం   i10 = i0 % ne10;
    స్థిరరాశి పూర్ణాంకం   i20 = i0 % ne20;
    స్థిరరాశి దశాంశం acc = bin_op((దశాంశం) src0[i_src0 + i0 * s00], (దశాంశం) src1[i_src1 + i10 * s10]);
    dst[i_dst + i0] = (dst_t) bin_op(acc, (దశాంశం) src2[i_src2 + i20 * s20]);
}

మూస<దశాంశం (*bin_op)(స్థిరరాశి దశాంశం, స్థిరరాశి దశాంశం), రకం_పేరు src0_t, రకం_పేరు src1_t, రకం_పేరు src2_t, రకం_పేరు dst_t>
స్థిర శూన్య launch_bin_bcast3(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor * src0, స్థిరరాశి ggml_tensor * src1,
                              స్థిరరాశి ggml_tensor * src2, ggml_tensor * dst) {
    dpct::వరుస_ptr stream = ctx.stream();
    SYCL_CHECK(ggml_sycl_సమితి_device(ctx.device));

    GGML_TENSOR_TERNARY_OP_LOCALS

    పూర్ణాంకం nr1[4] = { (పూర్ణాంకం) (ne10 / ne0), (పూర్ణాంకం) (ne11 / ne1), (పూర్ణాంకం) (ne12 / ne2), (పూర్ణాంకం) (ne13 / ne3) };
    పూర్ణాంకం nr2[4] = { (పూర్ణాంకం) (ne20 / ne0), (పూర్ణాంకం) (ne21 / ne1), (పూర్ణాంకం) (ne22 / ne2), (పూర్ణాంకం) (ne23 / ne3) };

    int64_t cne[]  = { ne0, ne1, ne2, ne3 };
    int64_t cne0[] = { ne00, ne01, ne02, ne03 };
    int64_t cne1[] = { ne10, ne11, ne12, ne13 };
    int64_t cne2[] = { ne20, ne21, ne22, ne23 };
    size_t  cnb[]  = { nb0, nb1, nb2, nb3 };
    size_t  cnb0[] = { nb00, nb01, nb02, nb03 };
    size_t  cnb1[] = { nb10, nb11, nb12, nb13 };
    size_t  cnb2[] = { nb20, nb21, nb22, nb23 };

    స్వయంచాలక collapse = [](int64_t cne[]) {
        cne[0] *= cne[1];
        cne[1] = cne[2];
        cne[2] = cne[3];
        cne[3] = 1;
    };

    స్వయంచాలక collapse_nb = [](size_t cnb[], int64_t cne[]) {
        cnb[1] *= cne[1];
        cnb[2] *= cne[2];
        cnb[3] *= cne[3];
    };

    స్థిరరాశి తార్కిక can_collapse = ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(src2) &&
                              !ggml_is_permuted(src0) && !ggml_is_permuted(src1) && !ggml_is_permuted(src2);
    ఒకవేళ (can_collapse) {
        కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
            ఒకవేళ (nr1[i] != 1 || nr2[i] != 1) {
                ఆపు;
            }
            ఒకవేళ (i > 0) {
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

        size_t s1 = cnb[1] / పరిమాణం(dst_t);
        size_t s2 = cnb[2] / పరిమాణం(dst_t);
        size_t s3 = cnb[3] / పరిమాణం(dst_t);

        size_t s00 = cnb0[0] / పరిమాణం(src0_t);
        size_t s01 = cnb0[1] / పరిమాణం(src0_t);
        size_t s02 = cnb0[2] / పరిమాణం(src0_t);
        size_t s03 = cnb0[3] / పరిమాణం(src0_t);

        size_t s10 = cnb1[0] / పరిమాణం(src1_t);
        size_t s11 = cnb1[1] / పరిమాణం(src1_t);
        size_t s12 = cnb1[2] / పరిమాణం(src1_t);
        size_t s13 = cnb1[3] / పరిమాణం(src1_t);

        size_t s20 = cnb2[0] / పరిమాణం(src2_t);
        size_t s21 = cnb2[1] / పరిమాణం(src2_t);
        size_t s22 = cnb2[2] / పరిమాణం(src2_t);
        size_t s23 = cnb2[3] / పరిమాణం(src2_t);

        GGML_ASSERT(cnb[0] % పరిమాణం(dst_t) == 0 && cnb[1] % పరిమాణం(dst_t) == 0 && cnb[2] % పరిమాణం(dst_t) == 0 &&
                    cnb[3] % పరిమాణం(dst_t) == 0);
        GGML_ASSERT(cnb0[0] % పరిమాణం(src0_t) == 0 && cnb0[1] % పరిమాణం(src0_t) == 0 && cnb0[2] % పరిమాణం(src0_t) == 0 &&
                    cnb0[3] % పరిమాణం(src0_t) == 0);
        GGML_ASSERT(cnb1[0] % పరిమాణం(src1_t) == 0 && cnb1[1] % పరిమాణం(src1_t) == 0 && cnb1[2] % పరిమాణం(src1_t) == 0 &&
                    cnb1[3] % పరిమాణం(src1_t) == 0);
        GGML_ASSERT(cnb2[0] % పరిమాణం(src2_t) == 0 && cnb2[1] % పరిమాణం(src2_t) == 0 && cnb2[2] % పరిమాణం(src2_t) == 0 &&
                    cnb2[3] % పరిమాణం(src2_t) == 0);

        స్థిరరాశి src0_t * src0_dd = (స్థిరరాశి src0_t *) src0->దత్తాంశం;
        స్థిరరాశి src1_t * src1_dd = (స్థిరరాశి src1_t *) src1->దత్తాంశం;
        స్థిరరాశి src2_t * src2_dd = (స్థిరరాశి src2_t *) src2->దత్తాంశం;
        dst_t *        dst_dd  = (dst_t *) dst->దత్తాంశం;

        స్థిరరాశి పూర్ణాంకం ఖండం_కొలత_3 = 128;
        int64_t   hne0       = std::max(ne0 / 2LL, 1LL);

        sycl::range<3> ఖండం_dims(1, 1, 1);
        ఖండం_dims[2] = std::min<గుర్తులేని పూర్ణాంకం>(hne0, ఖండం_కొలత_3);
        ఖండం_dims[1] = std::min<గుర్తులేని పూర్ణాంకం>(ne1, ఖండం_కొలత_3 / (గుర్తులేని పూర్ణాంకం) ఖండం_dims[2]);
        ఖండం_dims[0] = std::min(std::min<గుర్తులేని పూర్ణాంకం>(ne2 * ne3,
                                                        ఖండం_కొలత_3 / (గుర్తులేని పూర్ణాంకం) ఖండం_dims[2] /
                                                            (గుర్తులేని పూర్ణాంకం) ఖండం_dims[1]),
                                 64U);

        sycl::range<3> ఖండం_nums((ne2 * ne3 + ఖండం_dims[0] - 1) / ఖండం_dims[0],
                                  (ne1 + ఖండం_dims[1] - 1) / ఖండం_dims[1],
                                  (hne0 + ఖండం_dims[2] - 1) / ఖండం_dims[2]);

        dpct::has_capability_or_fail(stream->get_device(), { sycl::aspect::fp16 });

        ఒకవేళ (ఖండం_nums[0] > 65535) {
            పూర్ణాంకం ఖండం_అంకె = (ne0 * ne1 * ne2 * ne3 + ఖండం_కొలత_3 - 1) / ఖండం_కొలత_3;
            stream->parallel_for(
                sycl::nd_range<3>(sycl::range<3>(1, 1, ఖండం_అంకె) * sycl::range<3>(1, 1, ఖండం_కొలత_3),
                                  sycl::range<3>(1, 1, ఖండం_కొలత_3)),
                [=](sycl::nd_అంశం<3> అంశం_ct1) {
                    k_bin_bcast3_unravel<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10, ne11,
                                                 ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00, s01, s02, s03,
                                                 s10, s11, s12, s13, s20, s21, s22, s23, అంశం_ct1);
                });
        } లేకపోతే {
            stream->parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                                 [=](sycl::nd_అంశం<3> అంశం_ct1) {
                                     k_bin_bcast3<bin_op>(src0_dd, src1_dd, src2_dd, dst_dd, ne0, ne1, ne2, ne3, ne10,
                                                          ne11, ne12, ne13, ne20, ne21, ne22, ne23, s1, s2, s3, s00,
                                                          s01, s02, s03, s10, s11, s12, s13, s20, s21, s22, s23,
                                                          అంశం_ct1);
                                 });
        }
    }
}

శూన్య ggml_sycl_op_కూడు_కూడు_fused(ggml_backend_sycl_context & ctx, ggml_tensor * add0, ggml_tensor * add1) {
    స్థిరరాశి ggml_tensor * src0 = add0->src[0];
    స్థిరరాశి ggml_tensor * src1 = add0->src[1];
    స్థిరరాశి ggml_tensor * src2 = add1->src[1];
    ggml_tensor *       dst  = add1;

    GGML_ASSERT(add1->src[0] == add0);
    GGML_ASSERT(ggml_sycl_కూడు_kernel_supports(src0->type, src1->type, add0->type));
    GGML_ASSERT(ggml_sycl_కూడు_kernel_supports(add0->type, src2->type, dst->type));

    ఒకవేళ (src0->type == GGML_TYPE_F32 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
        dst->type == GGML_TYPE_F32) {
        launch_bin_bcast3<op_కూడు, దశాంశం, దశాంశం, దశాంశం, దశాంశం>(ctx, src0, src1, src2, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_కూడు, sycl::half, sycl::half, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_కూడు, sycl::half, దశాంశం, దశాంశం, sycl::half>(ctx, src0, src1, src2, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_కూడు, sycl::half, sycl::half, దశాంశం, sycl::half>(ctx, src0, src1, src2, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F16 &&
               dst->type == GGML_TYPE_F16) {
        launch_bin_bcast3<op_కూడు, sycl::half, దశాంశం, sycl::half, sycl::half>(ctx, src0, src1, src2, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_I32 && src1->type == GGML_TYPE_I32 && src2->type == GGML_TYPE_I32 &&
               dst->type == GGML_TYPE_I32) {
        launch_bin_bcast3<op_కూడు, int32_t, int32_t, int32_t, int32_t>(ctx, src0, src1, src2, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_I16 && src1->type == GGML_TYPE_I16 && src2->type == GGML_TYPE_I16 &&
               dst->type == GGML_TYPE_I16) {
        launch_bin_bcast3<op_కూడు, int16_t, int16_t, int16_t, int16_t>(ctx, src0, src1, src2, dst);
#నిర్వచితమైతే GGML_SYCL_HAS_BF16
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_కూడు, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_కూడు, sycl::ext::oneapi::bfloat16, దశాంశం, దశాంశం, sycl::ext::oneapi::bfloat16>(
            ctx, src0, src1, src2, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_BF16 && src2->type == GGML_TYPE_F32 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_కూడు, sycl::ext::oneapi::bfloat16, sycl::ext::oneapi::bfloat16, దశాంశం,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
    } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_BF16 && src1->type == GGML_TYPE_F32 && src2->type == GGML_TYPE_BF16 &&
               dst->type == GGML_TYPE_BF16) {
        launch_bin_bcast3<op_కూడు, sycl::ext::oneapi::bfloat16, దశాంశం, sycl::ext::oneapi::bfloat16,
                          sycl::ext::oneapi::bfloat16>(ctx, src0, src1, src2, dst);
#అయితే_ముగింపు
    } లేకపోతే {
        fprintf(stderr, "%s: unsupported types: dst: %s, src0: %s, src1: %s, src2: %s\n", __func__,
                ggml_type_name(dst->type), ggml_type_name(src0->type), ggml_type_name(src1->type),
                ggml_type_name(src2->type));
        GGML_ABORT("fatal error");
    }
}

