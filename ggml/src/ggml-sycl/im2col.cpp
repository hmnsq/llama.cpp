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

#ಅನ್ತರ್ಭಾವ "im2col.hpp"

#ವ್ಯಾಖ್ಯಾನಿಸು MAX_GRIDDIM_Z 65535

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ  ಶೂನ್ಯ im2col_kernel(
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW,
        ಪೂರ್ಣಾಂಕ s0, ಪೂರ್ಣಾಂಕ s1, ಪೂರ್ಣಾಂಕ p0, ಪೂರ್ಣಾಂಕ p1, ಪೂರ್ಣಾಂಕ d0, ಪೂರ್ಣಾಂಕ d1) {
    ಸ್ವಯಂಚಾಲಿತ          ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ int64_t i        = ಅಂಶ_ct1.get_local_id(2) + ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
    ಆದರೆ (i >= IC_KH_KW) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int64_t iic = i / (KH_KW);
    ಸ್ಥಿರಾಂಕ int64_t rem = i - iic * KH_KW;
    ಸ್ಥಿರಾಂಕ int64_t ikh = rem / KW;
    ಸ್ಥಿರಾಂಕ int64_t ikw = rem - ikh * KW;

    ಸ್ಥಿರಾಂಕ int64_t iow = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಗಾಗಿ (int64_t iz = ಅಂಶ_ct1.get_ಗುಂಪು(0); iz < N_OH; iz += MAX_GRIDDIM_Z) {
        ಸ್ಥಿರಾಂಕ int64_t  ಒಳ = iz / OH;
        ಸ್ಥಿರಾಂಕ int64_t  ioh = iz - ಒಳ * OH;

        ಸ್ಥಿರಾಂಕ int64_t iiw = iow * s0 + ikw * d0 - p0;
        ಸ್ಥಿರಾಂಕ int64_t iih = ioh * s1 + ikh * d1 - p1;

        ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನಾಂತರ_dst =
            ((ಒಳ * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

        ಆದರೆ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
            dst[ಸ್ಥಾನಾಂತರ_dst] = 0.0f;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನಾಂತರ_src = iic * IC_IH_IW + ಒಳ * IH_IW;
            dst[ಸ್ಥಾನಾಂತರ_dst] = x[ಸ್ಥಾನಾಂತರ_src + iih * IW + iiw];
        }
    }

    GGML_ಬಳಸದ(IC);
    GGML_ಬಳಸದ(KH);
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ im2col_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   x,
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
                        ಪೂರ್ಣಾಂಕ             s0,
                        ಪೂರ್ಣಾಂಕ             s1,
                        ಪೂರ್ಣಾಂಕ             p0,
                        ಪೂರ್ಣಾಂಕ             p1,
                        ಪೂರ್ಣಾಂಕ             d0,
                        ಪೂರ್ಣಾಂಕ             d1,
                        dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t IC_KH_KW = IC * KH * KW;
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (IC_KH_KW + SYCL_IM2_ಲಂಬಸಾಲು_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_IM2_ಲಂಬಸಾಲು_ಖಂಡ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ int64_t N_OH = N * OH;
    ಸ್ಥಿರಾಂಕ int64_t KH_KW = KW*KH;
    dpct::dim3    ಖಂಡ_nums(ಅಂಕೆ_ಖಂಡಗಳು, OW, MIN(N_OH, MAX_GRIDDIM_Z));
    /*
    DPCT1049:73: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    stream->parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * sycl::range<3>(1, 1, MIN(IC_KH_KW, SYCL_IM2_ಲಂಬಸಾಲು_ಖಂಡ_ಗಾತ್ರ)),
                                           sycl::range<3>(1, 1, MIN(IC_KH_KW, SYCL_IM2_ಲಂಬಸಾಲು_ಖಂಡ_ಗಾತ್ರ))),
                         [=](sycl::nd_ಅಂಶ<3>) {
                             im2col_kernel(x, dst, IC, IW, IH, OH, OW, KW, KH, IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW,
                                           s0, s1, p0, p1, d0, d1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_sycl_f16(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   x,
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
                            ಪೂರ್ಣಾಂಕ             s0,
                            ಪೂರ್ಣಾಂಕ             s1,
                            ಪೂರ್ಣಾಂಕ             p0,
                            ಪೂರ್ಣಾಂಕ             p1,
                            ಪೂರ್ಣಾಂಕ             d0,
                            ಪೂರ್ಣಾಂಕ             d1,
                            dpct::ಸರತಿ_ptr stream) {
    im2col_sycl<sycl::half>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_sycl_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   x,
                            ದಶಮಾಂಶ *         dst,
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
                            ಪೂರ್ಣಾಂಕ             s0,
                            ಪೂರ್ಣಾಂಕ             s1,
                            ಪೂರ್ಣಾಂಕ             p0,
                            ಪೂರ್ಣಾಂಕ             p1,
                            ಪೂರ್ಣಾಂಕ             d0,
                            ಪೂರ್ಣಾಂಕ             d1,
                            dpct::ಸರತಿ_ptr stream) {
    im2col_sycl<ದಶಮಾಂಶ>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

ಶೂನ್ಯ ggml_sycl_op_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_d = (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ;
    dpct::ಸರತಿ_ptr     stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ int32_t s0 = ((ಸ್ಥಿರಾಂಕ int32_t*)(dst->op_params))[0];
    ಸ್ಥಿರಾಂಕ int32_t s1 = ((ಸ್ಥಿರಾಂಕ int32_t*)(dst->op_params))[1];
    ಸ್ಥಿರಾಂಕ int32_t p0 = ((ಸ್ಥಿರಾಂಕ int32_t*)(dst->op_params))[2];
    ಸ್ಥಿರಾಂಕ int32_t p1 = ((ಸ್ಥಿರಾಂಕ int32_t*)(dst->op_params))[3];
    ಸ್ಥಿರಾಂಕ int32_t d0 = ((ಸ್ಥಿರಾಂಕ int32_t*)(dst->op_params))[4];
    ಸ್ಥಿರಾಂಕ int32_t d1 = ((ಸ್ಥಿರಾಂಕ int32_t*)(dst->op_params))[5];

    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ is_2D = ((ಸ್ಥಿರಾಂಕ int32_t*)(dst->op_params))[6] == 1;

    ಸ್ಥಿರಾಂಕ int64_t IC = src1->ne[is_2D ? 2 : 1];
    ಸ್ಥಿರಾಂಕ int64_t IH = is_2D ? src1->ne[1] : 1;
    ಸ್ಥಿರಾಂಕ int64_t IW =         src1->ne[0];

    ಸ್ಥಿರಾಂಕ int64_t KH = is_2D ? src0->ne[1] : 1;
    ಸ್ಥಿರಾಂಕ int64_t KW =         src0->ne[0];

    ಸ್ಥಿರಾಂಕ int64_t OH = is_2D ? dst->ne[2] : 1;
    ಸ್ಥಿರಾಂಕ int64_t OW =         dst->ne[1];

    ಸ್ಥಿರಾಂಕ int64_t IC_IH_IW = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    ಸ್ಥಿರಾಂಕ int64_t N        = src1->ne[is_2D ? 3 : 2];
    ಸ್ಥಿರಾಂಕ int64_t IH_IW    = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    ಆದರೆ(dst->type == GGML_TYPE_F16) {
        im2col_sycl_f16(src1_d, (sycl::half *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1,
                        d0, d1, stream);
    } ಇಲ್ಲದಿದ್ದರೆ {
        im2col_sycl_f32(src1_d, (ದಶಮಾಂಶ *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ  ಶೂನ್ಯ im2col_3d_kernel(
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, T * dst,
        int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
        int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
        int64_t OH_OW, int64_t KD_KH_KW, int64_t ID_IH_IW, int64_t KH_KW, int64_t IH_IW, int64_t IC_ID_IH_IW,
        int64_t IC_KD_KH_KW, int64_t OW_KD_KH_KW, int64_t OD_OH_OW_IC_KD_KH_KW, int64_t OH_OW_IC_KD_KH_KW,
        int64_t OW_IC_KD_KH_KW, int64_t N_OD_OH, int64_t OD_OH,
        int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
        ಪೂರ್ಣಾಂಕ s0, ಪೂರ್ಣಾಂಕ s1, ಪೂರ್ಣಾಂಕ s2, ಪೂರ್ಣಾಂಕ p0, ಪೂರ್ಣಾಂಕ p1, ಪೂರ್ಣಾಂಕ p2, ಪೂರ್ಣಾಂಕ d0, ಪೂರ್ಣಾಂಕ d1, ಪೂರ್ಣಾಂಕ d2) {
    ಸ್ವಯಂಚಾಲಿತ          ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ int64_t i        = ಅಂಶ_ct1.get_local_id(2) + ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
    ಆದರೆ (i >= IC_KD_KH_KW) {
        ಹಿಂತಿರುಗಿ;
    }
    GGML_ಬಳಸದ(N); GGML_ಬಳಸದ(OC); GGML_ಬಳಸದ(OH_OW); GGML_ಬಳಸದ(OD); GGML_ಬಳಸದ(OW); GGML_ಬಳಸದ(KD); GGML_ಬಳಸದ(KH);
    GGML_ಬಳಸದ(ID_IH_IW); GGML_ಬಳಸದ(IH_IW); GGML_ಬಳಸದ(IC_ID_IH_IW); GGML_ಬಳಸದ(OW_KD_KH_KW);

    ಸ್ಥಿರಾಂಕ int64_t iic = i / KD_KH_KW;
    ಸ್ಥಿರಾಂಕ int64_t ikd = (i - iic * KD_KH_KW) / KH_KW;
    ಸ್ಥಿರಾಂಕ int64_t ikh = (i - iic * KD_KH_KW - ikd * KH_KW) / KW;
    ಸ್ಥಿರಾಂಕ int64_t ikw = i % KW;

    ಸ್ಥಿರಾಂಕ int64_t iow = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಗಾಗಿ (int64_t iz = ಅಂಶ_ct1.get_ಗುಂಪು(0); iz < N_OD_OH; iz += MAX_GRIDDIM_Z) {
        ಸ್ಥಿರಾಂಕ int64_t ಒಳ  = iz / OD_OH;
        ಸ್ಥಿರಾಂಕ int64_t iod = (iz - ಒಳ*OD_OH) / OH;
        ಸ್ಥಿರಾಂಕ int64_t ioh = iz % OH;

        ಸ್ಥಿರಾಂಕ int64_t iiw = iow * s0 + ikw * d0 - p0;
        ಸ್ಥಿರಾಂಕ int64_t iih = ioh * s1 + ikh * d1 - p1;
        ಸ್ಥಿರಾಂಕ int64_t iid = iod * s2 + ikd * d2 - p2;

        ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನಾಂತರ_dst = ಒಳ*OD_OH_OW_IC_KD_KH_KW + iod*OH_OW_IC_KD_KH_KW + ioh*OW_IC_KD_KH_KW + iow*IC_KD_KH_KW + iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw;

        ಆದರೆ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
            dst[ಸ್ಥಾನಾಂತರ_dst] = 0.0f;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ int64_t ಸ್ಥಾನಾಂತರ_src = ((ಒಳ * IC + iic) * stride_q) + (iid * stride_z) + (iih * stride_y) + (iiw * stride_x);
            dst[ಸ್ಥಾನಾಂತರ_dst] = src[ಸ್ಥಾನಾಂತರ_src];
        }
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T>
ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   src,
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
                           ಪೂರ್ಣಾಂಕ             s0,
                           ಪೂರ್ಣಾಂಕ             s1,
                           ಪೂರ್ಣಾಂಕ             s2,
                           ಪೂರ್ಣಾಂಕ             p0,
                           ಪೂರ್ಣಾಂಕ             p1,
                           ಪೂರ್ಣಾಂಕ             p2,
                           ಪೂರ್ಣಾಂಕ             d0,
                           ಪೂರ್ಣಾಂಕ             d1,
                           ಪೂರ್ಣಾಂಕ             d2,
                           dpct::ಸರತಿ_ptr stream) {
    ಸ್ಥಿರಾಂಕ int64_t OH_OW = OH*OW;
    ಸ್ಥಿರಾಂಕ int64_t KD_KH_KW = KD*KH*KW;
    ಸ್ಥಿರಾಂಕ int64_t ID_IH_IW = ID*IH*IW;
    ಸ್ಥಿರಾಂಕ int64_t KH_KW = KH*KW;
    ಸ್ಥಿರಾಂಕ int64_t IH_IW = IH*IW;
    ಸ್ಥಿರಾಂಕ int64_t IC_KD_KH_KW = IC*KD*KH*KW;
    ಸ್ಥಿರಾಂಕ int64_t OW_KD_KH_KW = OW*KD*KH*KW;
    ಸ್ಥಿರಾಂಕ int64_t N_OD_OH = N*OD*OH;
    ಸ್ಥಿರಾಂಕ int64_t OD_OH = OD*OH;
    ಸ್ಥಿರಾಂಕ int64_t IC_ID_IH_IW = IC*ID*IH*IW;
    ಸ್ಥಿರಾಂಕ int64_t OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    ಸ್ಥಿರಾಂಕ int64_t OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    ಸ್ಥಿರಾಂಕ int64_t OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (IC_KD_KH_KW + SYCL_IM2_ಲಂಬಸಾಲು_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_IM2_ಲಂಬಸಾಲು_ಖಂಡ_ಗಾತ್ರ;
    dpct::dim3    ಖಂಡ_nums(ಅಂಕೆ_ಖಂಡಗಳು, OW, MIN(N_OD_OH, MAX_GRIDDIM_Z));
    /*
    DPCT1049:74: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    stream->parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * sycl::range<3>(1, 1, MIN(IC_KD_KH_KW, SYCL_IM2_ಲಂಬಸಾಲು_ಖಂಡ_ಗಾತ್ರ)),
                                           sycl::range<3>(1, 1, MIN(IC_KD_KH_KW, SYCL_IM2_ಲಂಬಸಾಲು_ಖಂಡ_ಗಾತ್ರ))),
                         [=](sycl::nd_ಅಂಶ<3>) {
                             im2col_3d_kernel(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW, OH_OW, KD_KH_KW,
                                              ID_IH_IW, KH_KW, IH_IW, IC_ID_IH_IW, IC_KD_KH_KW, OW_KD_KH_KW,
                                              OD_OH_OW_IC_KD_KH_KW, OH_OW_IC_KD_KH_KW, OW_IC_KD_KH_KW, N_OD_OH, OD_OH,
                                              stride_q, stride_z, stride_y, stride_x, s0, s1, s2, p0, p1, p2, d0, d1,
                                              d2);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_sycl_f16(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   src,
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
                               ಪೂರ್ಣಾಂಕ             s0,
                               ಪೂರ್ಣಾಂಕ             s1,
                               ಪೂರ್ಣಾಂಕ             s2,
                               ಪೂರ್ಣಾಂಕ             p0,
                               ಪೂರ್ಣಾಂಕ             p1,
                               ಪೂರ್ಣಾಂಕ             p2,
                               ಪೂರ್ಣಾಂಕ             d0,
                               ಪೂರ್ಣಾಂಕ             d1,
                               ಪೂರ್ಣಾಂಕ             d2,
                               dpct::ಸರತಿ_ptr stream) {
    im2col_3d_sycl<sycl::half>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW, stride_q, stride_z, stride_y,
                               stride_x, s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_sycl_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   src,
                               ದಶಮಾಂಶ *         dst,
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
                               ಪೂರ್ಣಾಂಕ             s0,
                               ಪೂರ್ಣಾಂಕ             s1,
                               ಪೂರ್ಣಾಂಕ             s2,
                               ಪೂರ್ಣಾಂಕ             p0,
                               ಪೂರ್ಣಾಂಕ             p1,
                               ಪೂರ್ಣಾಂಕ             p2,
                               ಪೂರ್ಣಾಂಕ             d0,
                               ಪೂರ್ಣಾಂಕ             d1,
                               ಪೂರ್ಣಾಂಕ             d2,
                               dpct::ಸರತಿ_ptr stream) {
    im2col_3d_sycl<ದಶಮಾಂಶ>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                          stride_q, stride_z, stride_y, stride_x,
                          s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

ಶೂನ್ಯ ggml_sycl_op_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_d = (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ;
    dpct::ಸರತಿ_ptr     stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ int32_t s0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[0];
    ಸ್ಥಿರಾಂಕ int32_t s1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[1];
    ಸ್ಥಿರಾಂಕ int32_t s2 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[2];
    ಸ್ಥಿರಾಂಕ int32_t p0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[3];
    ಸ್ಥಿರಾಂಕ int32_t p1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[4];
    ಸ್ಥಿರಾಂಕ int32_t p2 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[5];
    ಸ್ಥಿರಾಂಕ int32_t d0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[6];
    ಸ್ಥಿರಾಂಕ int32_t d1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[7];
    ಸ್ಥಿರಾಂಕ int32_t d2 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[8];
    ಸ್ಥಿರಾಂಕ int32_t IC = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[9];

    ಸ್ಥಿರಾಂಕ int64_t N  = ne13 / IC;
    ಸ್ಥಿರಾಂಕ int64_t ID = ne12;
    ಸ್ಥಿರಾಂಕ int64_t IH = ne11;
    ಸ್ಥಿರಾಂಕ int64_t IW = ne10;

    ಸ್ಥಿರಾಂಕ int64_t OC = ne03 / IC;
    ಸ್ಥಿರಾಂಕ int64_t KD = ne02;
    ಸ್ಥಿರಾಂಕ int64_t KH = ne01;
    ಸ್ಥಿರಾಂಕ int64_t KW = ne00;

    ಸ್ಥಿರಾಂಕ int64_t OD = ne3 / N;
    ಸ್ಥಿರಾಂಕ int64_t OH = ne2;
    ಸ್ಥಿರಾಂಕ int64_t OW = ne1;

    ಸ್ಥಿರಾಂಕ size_t  es       = ggml_element_ಗಾತ್ರ(src1);
    ಸ್ಥಿರಾಂಕ int64_t stride_x = src1->nb[0] / es;
    ಸ್ಥಿರಾಂಕ int64_t stride_y = src1->nb[1] / es;
    ಸ್ಥಿರಾಂಕ int64_t stride_z = src1->nb[2] / es;
    ಸ್ಥಿರಾಂಕ int64_t stride_q = src1->nb[3] / es;

    ಆದರೆ(dst->type == GGML_TYPE_F16) {
        im2col_3d_sycl_f16(src1_d, (sycl::half *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    } ಇಲ್ಲದಿದ್ದರೆ {
        im2col_3d_sycl_f32(src1_d, (ದಶಮಾಂಶ *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    }
}
