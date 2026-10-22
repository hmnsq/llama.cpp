//
// MIT license
// Copyright (C) 2026 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#అన్తర్భావ "im2col.hpp"

#నిర్వచించు MAX_GRIDDIM_Z 65535

మూస <రకం_పేరు T>
స్థిర  శూన్య im2col_kernel(
        స్థిరరాశి దశాంశం * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW,
        పూర్ణాంకం s0, పూర్ణాంకం s1, పూర్ణాంకం p0, పూర్ణాంకం p1, పూర్ణాంకం d0, పూర్ణాంకం d1) {
    స్వయంచాలక          అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి int64_t i        = అంశం_ct1.get_local_id(2) + అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
    ఒకవేళ (i >= IC_KH_KW) {
        తిరిగి;
    }

    స్థిరరాశి int64_t iic = i / (KH_KW);
    స్థిరరాశి int64_t rem = i - iic * KH_KW;
    స్థిరరాశి int64_t ikh = rem / KW;
    స్థిరరాశి int64_t ikw = rem - ikh * KW;

    స్థిరరాశి int64_t iow = అంశం_ct1.get_సమూహం(1);
    కొరకు (int64_t iz = అంశం_ct1.get_సమూహం(0); iz < N_OH; iz += MAX_GRIDDIM_Z) {
        స్థిరరాశి int64_t  లోపల = iz / OH;
        స్థిరరాశి int64_t  ioh = iz - లోపల * OH;

        స్థిరరాశి int64_t iiw = iow * s0 + ikw * d0 - p0;
        స్థిరరాశి int64_t iih = ioh * s1 + ikh * d1 - p1;

        స్థిరరాశి int64_t స్థానభ్రంశం_dst =
            ((లోపల * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

        ఒకవేళ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
            dst[స్థానభ్రంశం_dst] = 0.0f;
        } లేకపోతే {
            స్థిరరాశి int64_t స్థానభ్రంశం_src = iic * IC_IH_IW + లోపల * IH_IW;
            dst[స్థానభ్రంశం_dst] = x[స్థానభ్రంశం_src + iih * IW + iiw];
        }
    }

    GGML_వాడని(IC);
    GGML_వాడని(KH);
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
మూస <రకం_పేరు T>
స్థిర శూన్య im2col_sycl(స్థిరరాశి దశాంశం *   x,
                        T *             dst,
                        int64_t         IW,
                        int64_t         IH,
                        int64_t         OW,
                        int64_t         OH,
                        int64_t         KW,
                        int64_t         KH,
                        int64_t         IC,
                        int64_t         N,
                        int64_t         IC_IH_IW,
                        int64_t         IH_IW,
                        పూర్ణాంకం             s0,
                        పూర్ణాంకం             s1,
                        పూర్ణాంకం             p0,
                        పూర్ణాంకం             p1,
                        పూర్ణాంకం             d0,
                        పూర్ణాంకం             d1,
                        dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t IC_KH_KW = IC * KH * KW;
    స్థిరరాశి int64_t అంకె_ఖండాలు = (IC_KH_KW + SYCL_IM2_నిలువు_ఖండం_కొలత - 1) / SYCL_IM2_నిలువు_ఖండం_కొలత;
    స్థిరరాశి int64_t N_OH = N * OH;
    స్థిరరాశి int64_t KH_KW = KW*KH;
    dpct::dim3    ఖండం_nums(అంకె_ఖండాలు, OW, MIN(N_OH, MAX_GRIDDIM_Z));
    /*
    DPCT1049:73: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    stream->parallel_for(sycl::nd_range<3>(ఖండం_nums * sycl::range<3>(1, 1, MIN(IC_KH_KW, SYCL_IM2_నిలువు_ఖండం_కొలత)),
                                           sycl::range<3>(1, 1, MIN(IC_KH_KW, SYCL_IM2_నిలువు_ఖండం_కొలత))),
                         [=](sycl::nd_అంశం<3>) {
                             im2col_kernel(x, dst, IC, IW, IH, OH, OW, KW, KH, IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW,
                                           s0, s1, p0, p1, d0, d1);
                         });
}

స్థిర శూన్య im2col_sycl_f16(స్థిరరాశి దశాంశం *   x,
                            sycl::half *    dst,
                            int64_t         IW,
                            int64_t         IH,
                            int64_t         OW,
                            int64_t         OH,
                            int64_t         KW,
                            int64_t         KH,
                            int64_t         IC,
                            int64_t         N,
                            int64_t         IC_IH_IW,
                            int64_t         IH_IW,
                            పూర్ణాంకం             s0,
                            పూర్ణాంకం             s1,
                            పూర్ణాంకం             p0,
                            పూర్ణాంకం             p1,
                            పూర్ణాంకం             d0,
                            పూర్ణాంకం             d1,
                            dpct::వరుస_ptr stream) {
    im2col_sycl<sycl::half>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

స్థిర శూన్య im2col_sycl_f32(స్థిరరాశి దశాంశం *   x,
                            దశాంశం *         dst,
                            int64_t         IW,
                            int64_t         IH,
                            int64_t         OW,
                            int64_t         OH,
                            int64_t         KW,
                            int64_t         KH,
                            int64_t         IC,
                            int64_t         N,
                            int64_t         IC_IH_IW,
                            int64_t         IH_IW,
                            పూర్ణాంకం             s0,
                            పూర్ణాంకం             s1,
                            పూర్ణాంకం             p0,
                            పూర్ణాంకం             p1,
                            పూర్ణాంకం             d0,
                            పూర్ణాంకం             d1,
                            dpct::వరుస_ptr stream) {
    im2col_sycl<దశాంశం>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

శూన్య ggml_sycl_op_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    స్థిరరాశి దశాంశం * src1_d = (స్థిరరాశి దశాంశం *)src1->దత్తాంశం;
    దశాంశం * dst_d = (దశాంశం *)dst->దత్తాంశం;
    dpct::వరుస_ptr     stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    స్థిరరాశి int32_t s0 = ((స్థిరరాశి int32_t*)(dst->op_params))[0];
    స్థిరరాశి int32_t s1 = ((స్థిరరాశి int32_t*)(dst->op_params))[1];
    స్థిరరాశి int32_t p0 = ((స్థిరరాశి int32_t*)(dst->op_params))[2];
    స్థిరరాశి int32_t p1 = ((స్థిరరాశి int32_t*)(dst->op_params))[3];
    స్థిరరాశి int32_t d0 = ((స్థిరరాశి int32_t*)(dst->op_params))[4];
    స్థిరరాశి int32_t d1 = ((స్థిరరాశి int32_t*)(dst->op_params))[5];

    స్థిరరాశి తార్కిక is_2D = ((స్థిరరాశి int32_t*)(dst->op_params))[6] == 1;

    స్థిరరాశి int64_t IC = src1->ne[is_2D ? 2 : 1];
    స్థిరరాశి int64_t IH = is_2D ? src1->ne[1] : 1;
    స్థిరరాశి int64_t IW =         src1->ne[0];

    స్థిరరాశి int64_t KH = is_2D ? src0->ne[1] : 1;
    స్థిరరాశి int64_t KW =         src0->ne[0];

    స్థిరరాశి int64_t OH = is_2D ? dst->ne[2] : 1;
    స్థిరరాశి int64_t OW =         dst->ne[1];

    స్థిరరాశి int64_t IC_IH_IW = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    స్థిరరాశి int64_t N        = src1->ne[is_2D ? 3 : 2];
    స్థిరరాశి int64_t IH_IW    = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    ఒకవేళ(dst->type == GGML_TYPE_F16) {
        im2col_sycl_f16(src1_d, (sycl::half *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1,
                        d0, d1, stream);
    } లేకపోతే {
        im2col_sycl_f32(src1_d, (దశాంశం *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
మూస <రకం_పేరు T>
స్థిర  శూన్య im2col_3d_kernel(
        స్థిరరాశి దశాంశం * src, T * dst,
        int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
        int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
        int64_t OH_OW, int64_t KD_KH_KW, int64_t ID_IH_IW, int64_t KH_KW, int64_t IH_IW, int64_t IC_ID_IH_IW,
        int64_t IC_KD_KH_KW, int64_t OW_KD_KH_KW, int64_t OD_OH_OW_IC_KD_KH_KW, int64_t OH_OW_IC_KD_KH_KW,
        int64_t OW_IC_KD_KH_KW, int64_t N_OD_OH, int64_t OD_OH,
        int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
        పూర్ణాంకం s0, పూర్ణాంకం s1, పూర్ణాంకం s2, పూర్ణాంకం p0, పూర్ణాంకం p1, పూర్ణాంకం p2, పూర్ణాంకం d0, పూర్ణాంకం d1, పూర్ణాంకం d2) {
    స్వయంచాలక          అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి int64_t i        = అంశం_ct1.get_local_id(2) + అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2);
    ఒకవేళ (i >= IC_KD_KH_KW) {
        తిరిగి;
    }
    GGML_వాడని(N); GGML_వాడని(OC); GGML_వాడని(OH_OW); GGML_వాడని(OD); GGML_వాడని(OW); GGML_వాడని(KD); GGML_వాడని(KH);
    GGML_వాడని(ID_IH_IW); GGML_వాడని(IH_IW); GGML_వాడని(IC_ID_IH_IW); GGML_వాడని(OW_KD_KH_KW);

    స్థిరరాశి int64_t iic = i / KD_KH_KW;
    స్థిరరాశి int64_t ikd = (i - iic * KD_KH_KW) / KH_KW;
    స్థిరరాశి int64_t ikh = (i - iic * KD_KH_KW - ikd * KH_KW) / KW;
    స్థిరరాశి int64_t ikw = i % KW;

    స్థిరరాశి int64_t iow = అంశం_ct1.get_సమూహం(1);
    కొరకు (int64_t iz = అంశం_ct1.get_సమూహం(0); iz < N_OD_OH; iz += MAX_GRIDDIM_Z) {
        స్థిరరాశి int64_t లోపల  = iz / OD_OH;
        స్థిరరాశి int64_t iod = (iz - లోపల*OD_OH) / OH;
        స్థిరరాశి int64_t ioh = iz % OH;

        స్థిరరాశి int64_t iiw = iow * s0 + ikw * d0 - p0;
        స్థిరరాశి int64_t iih = ioh * s1 + ikh * d1 - p1;
        స్థిరరాశి int64_t iid = iod * s2 + ikd * d2 - p2;

        స్థిరరాశి int64_t స్థానభ్రంశం_dst = లోపల*OD_OH_OW_IC_KD_KH_KW + iod*OH_OW_IC_KD_KH_KW + ioh*OW_IC_KD_KH_KW + iow*IC_KD_KH_KW + iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw;

        ఒకవేళ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
            dst[స్థానభ్రంశం_dst] = 0.0f;
        } లేకపోతే {
            స్థిరరాశి int64_t స్థానభ్రంశం_src = ((లోపల * IC + iic) * stride_q) + (iid * stride_z) + (iih * stride_y) + (iiw * stride_x);
            dst[స్థానభ్రంశం_dst] = src[స్థానభ్రంశం_src];
        }
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
మూస <రకం_పేరు T>
స్థిర శూన్య im2col_3d_sycl(స్థిరరాశి దశాంశం *   src,
                           T *             dst,
                           int64_t         N,
                           int64_t         IC,
                           int64_t         ID,
                           int64_t         IH,
                           int64_t         IW,
                           int64_t         OC,
                           int64_t         KD,
                           int64_t         KH,
                           int64_t         KW,
                           int64_t         OD,
                           int64_t         OH,
                           int64_t         OW,
                           int64_t         stride_q,
                           int64_t         stride_z,
                           int64_t         stride_y,
                           int64_t         stride_x,
                           పూర్ణాంకం             s0,
                           పూర్ణాంకం             s1,
                           పూర్ణాంకం             s2,
                           పూర్ణాంకం             p0,
                           పూర్ణాంకం             p1,
                           పూర్ణాంకం             p2,
                           పూర్ణాంకం             d0,
                           పూర్ణాంకం             d1,
                           పూర్ణాంకం             d2,
                           dpct::వరుస_ptr stream) {
    స్థిరరాశి int64_t OH_OW = OH*OW;
    స్థిరరాశి int64_t KD_KH_KW = KD*KH*KW;
    స్థిరరాశి int64_t ID_IH_IW = ID*IH*IW;
    స్థిరరాశి int64_t KH_KW = KH*KW;
    స్థిరరాశి int64_t IH_IW = IH*IW;
    స్థిరరాశి int64_t IC_KD_KH_KW = IC*KD*KH*KW;
    స్థిరరాశి int64_t OW_KD_KH_KW = OW*KD*KH*KW;
    స్థిరరాశి int64_t N_OD_OH = N*OD*OH;
    స్థిరరాశి int64_t OD_OH = OD*OH;
    స్థిరరాశి int64_t IC_ID_IH_IW = IC*ID*IH*IW;
    స్థిరరాశి int64_t OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    స్థిరరాశి int64_t OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    స్థిరరాశి int64_t OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;
    స్థిరరాశి int64_t అంకె_ఖండాలు = (IC_KD_KH_KW + SYCL_IM2_నిలువు_ఖండం_కొలత - 1) / SYCL_IM2_నిలువు_ఖండం_కొలత;
    dpct::dim3    ఖండం_nums(అంకె_ఖండాలు, OW, MIN(N_OD_OH, MAX_GRIDDIM_Z));
    /*
    DPCT1049:74: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    stream->parallel_for(sycl::nd_range<3>(ఖండం_nums * sycl::range<3>(1, 1, MIN(IC_KD_KH_KW, SYCL_IM2_నిలువు_ఖండం_కొలత)),
                                           sycl::range<3>(1, 1, MIN(IC_KD_KH_KW, SYCL_IM2_నిలువు_ఖండం_కొలత))),
                         [=](sycl::nd_అంశం<3>) {
                             im2col_3d_kernel(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW, OH_OW, KD_KH_KW,
                                              ID_IH_IW, KH_KW, IH_IW, IC_ID_IH_IW, IC_KD_KH_KW, OW_KD_KH_KW,
                                              OD_OH_OW_IC_KD_KH_KW, OH_OW_IC_KD_KH_KW, OW_IC_KD_KH_KW, N_OD_OH, OD_OH,
                                              stride_q, stride_z, stride_y, stride_x, s0, s1, s2, p0, p1, p2, d0, d1,
                                              d2);
                         });
}

స్థిర శూన్య im2col_3d_sycl_f16(స్థిరరాశి దశాంశం *   src,
                               sycl::half *    dst,
                               int64_t         N,
                               int64_t         IC,
                               int64_t         ID,
                               int64_t         IH,
                               int64_t         IW,
                               int64_t         OC,
                               int64_t         KD,
                               int64_t         KH,
                               int64_t         KW,
                               int64_t         OD,
                               int64_t         OH,
                               int64_t         OW,
                               int64_t         stride_q,
                               int64_t         stride_z,
                               int64_t         stride_y,
                               int64_t         stride_x,
                               పూర్ణాంకం             s0,
                               పూర్ణాంకం             s1,
                               పూర్ణాంకం             s2,
                               పూర్ణాంకం             p0,
                               పూర్ణాంకం             p1,
                               పూర్ణాంకం             p2,
                               పూర్ణాంకం             d0,
                               పూర్ణాంకం             d1,
                               పూర్ణాంకం             d2,
                               dpct::వరుస_ptr stream) {
    im2col_3d_sycl<sycl::half>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW, stride_q, stride_z, stride_y,
                               stride_x, s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

స్థిర శూన్య im2col_3d_sycl_f32(స్థిరరాశి దశాంశం *   src,
                               దశాంశం *         dst,
                               int64_t         N,
                               int64_t         IC,
                               int64_t         ID,
                               int64_t         IH,
                               int64_t         IW,
                               int64_t         OC,
                               int64_t         KD,
                               int64_t         KH,
                               int64_t         KW,
                               int64_t         OD,
                               int64_t         OH,
                               int64_t         OW,
                               int64_t         stride_q,
                               int64_t         stride_z,
                               int64_t         stride_y,
                               int64_t         stride_x,
                               పూర్ణాంకం             s0,
                               పూర్ణాంకం             s1,
                               పూర్ణాంకం             s2,
                               పూర్ణాంకం             p0,
                               పూర్ణాంకం             p1,
                               పూర్ణాంకం             p2,
                               పూర్ణాంకం             d0,
                               పూర్ణాంకం             d1,
                               పూర్ణాంకం             d2,
                               dpct::వరుస_ptr stream) {
    im2col_3d_sycl<దశాంశం>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                          stride_q, stride_z, stride_y, stride_x,
                          s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

శూన్య ggml_sycl_op_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];
    స్థిరరాశి దశాంశం * src1_d = (స్థిరరాశి దశాంశం *)src1->దత్తాంశం;
    దశాంశం * dst_d = (దశాంశం *)dst->దత్తాంశం;
    dpct::వరుస_ptr     stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి int32_t s0 = ((స్థిరరాశి int32_t *)(dst->op_params))[0];
    స్థిరరాశి int32_t s1 = ((స్థిరరాశి int32_t *)(dst->op_params))[1];
    స్థిరరాశి int32_t s2 = ((స్థిరరాశి int32_t *)(dst->op_params))[2];
    స్థిరరాశి int32_t p0 = ((స్థిరరాశి int32_t *)(dst->op_params))[3];
    స్థిరరాశి int32_t p1 = ((స్థిరరాశి int32_t *)(dst->op_params))[4];
    స్థిరరాశి int32_t p2 = ((స్థిరరాశి int32_t *)(dst->op_params))[5];
    స్థిరరాశి int32_t d0 = ((స్థిరరాశి int32_t *)(dst->op_params))[6];
    స్థిరరాశి int32_t d1 = ((స్థిరరాశి int32_t *)(dst->op_params))[7];
    స్థిరరాశి int32_t d2 = ((స్థిరరాశి int32_t *)(dst->op_params))[8];
    స్థిరరాశి int32_t IC = ((స్థిరరాశి int32_t *)(dst->op_params))[9];

    స్థిరరాశి int64_t N  = ne13 / IC;
    స్థిరరాశి int64_t ID = ne12;
    స్థిరరాశి int64_t IH = ne11;
    స్థిరరాశి int64_t IW = ne10;

    స్థిరరాశి int64_t OC = ne03 / IC;
    స్థిరరాశి int64_t KD = ne02;
    స్థిరరాశి int64_t KH = ne01;
    స్థిరరాశి int64_t KW = ne00;

    స్థిరరాశి int64_t OD = ne3 / N;
    స్థిరరాశి int64_t OH = ne2;
    స్థిరరాశి int64_t OW = ne1;

    స్థిరరాశి size_t  es       = ggml_element_కొలత(src1);
    స్థిరరాశి int64_t stride_x = src1->nb[0] / es;
    స్థిరరాశి int64_t stride_y = src1->nb[1] / es;
    స్థిరరాశి int64_t stride_z = src1->nb[2] / es;
    స్థిరరాశి int64_t stride_q = src1->nb[3] / es;

    ఒకవేళ(dst->type == GGML_TYPE_F16) {
        im2col_3d_sycl_f16(src1_d, (sycl::half *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    } లేకపోతే {
        im2col_3d_sycl_f32(src1_d, (దశాంశం *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    }
}
