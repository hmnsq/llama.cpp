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

#define MAX_GRIDDIM_Z 65535

template <typename T>
ಸ್ಥಿರ  ಶೂನ್ಯ im2col_kernel(
        ಧ್ರುವ ಭಿನ್ನ * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW,
        ಪೂರ್ಣಾಙ್ಕ s0, ಪೂರ್ಣಾಙ್ಕ s1, ಪೂರ್ಣಾಙ್ಕ p0, ಪೂರ್ಣಾಙ್ಕ p1, ಪೂರ್ಣಾಙ್ಕ d0, ಪೂರ್ಣಾಙ್ಕ d1) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ int64_t i        = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ಯದಿ (i >= IC_KH_KW) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ int64_t iic = i / (KH_KW);
    ಧ್ರುವ int64_t rem = i - iic * KH_KW;
    ಧ್ರುವ int64_t ikh = rem / KW;
    ಧ್ರುವ int64_t ikw = rem - ikh * KW;

    ಧ್ರುವ int64_t iow = item_ct1.get_group(1);
    ಪ್ರತಿ (int64_t iz = item_ct1.get_group(0); iz < N_OH; iz += MAX_GRIDDIM_Z) {
        ಧ್ರುವ int64_t  in = iz / OH;
        ಧ್ರುವ int64_t  ioh = iz - in * OH;

        ಧ್ರುವ int64_t iiw = iow * s0 + ikw * d0 - p0;
        ಧ್ರುವ int64_t iih = ioh * s1 + ikh * d1 - p1;

        ಧ್ರುವ int64_t offset_dst =
            ((in * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

        ಯದಿ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
            dst[offset_dst] = 0.0f;
        } ಅನ್ಯಥಾ {
            ಧ್ರುವ int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
            dst[offset_dst] = x[offset_src + iih * IW + iiw];
        }
    }

    GGML_UNUSED(IC);
    GGML_UNUSED(KH);
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
template <typename T>
ಸ್ಥಿರ ಶೂನ್ಯ im2col_sycl(ಧ್ರುವ ಭಿನ್ನ *   x,
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
                        ಪೂರ್ಣಾಙ್ಕ             s0,
                        ಪೂರ್ಣಾಙ್ಕ             s1,
                        ಪೂರ್ಣಾಙ್ಕ             p0,
                        ಪೂರ್ಣಾಙ್ಕ             p1,
                        ಪೂರ್ಣಾಙ್ಕ             d0,
                        ಪೂರ್ಣಾಙ್ಕ             d1,
                        dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t IC_KH_KW = IC * KH * KW;
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (IC_KH_KW + SYCL_IM2COL_BLOCK_SIZE - 1) / SYCL_IM2COL_BLOCK_SIZE;
    ಧ್ರುವ int64_t N_OH = N * OH;
    ಧ್ರುವ int64_t KH_KW = KW*KH;
    dpct::dim3    block_nums(ಅಂಕೆ_blocks, OW, MIN(N_OH, MAX_GRIDDIM_Z));
    /*
    DPCT1049:73: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    stream->parallel_for(sycl::nd_range<3>(block_nums * sycl::range<3>(1, 1, MIN(IC_KH_KW, SYCL_IM2COL_BLOCK_SIZE)),
                                           sycl::range<3>(1, 1, MIN(IC_KH_KW, SYCL_IM2COL_BLOCK_SIZE))),
                         [=](sycl::nd_item<3>) {
                             im2col_kernel(x, dst, IC, IW, IH, OH, OW, KW, KH, IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW,
                                           s0, s1, p0, p1, d0, d1);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_sycl_f16(ಧ್ರುವ ಭಿನ್ನ *   x,
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
                            ಪೂರ್ಣಾಙ್ಕ             s0,
                            ಪೂರ್ಣಾಙ್ಕ             s1,
                            ಪೂರ್ಣಾಙ್ಕ             p0,
                            ಪೂರ್ಣಾಙ್ಕ             p1,
                            ಪೂರ್ಣಾಙ್ಕ             d0,
                            ಪೂರ್ಣಾಙ್ಕ             d1,
                            dpct::queue_ptr stream) {
    im2col_sycl<sycl::half>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_sycl_f32(ಧ್ರುವ ಭಿನ್ನ *   x,
                            ಭಿನ್ನ *         dst,
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
                            ಪೂರ್ಣಾಙ್ಕ             s0,
                            ಪೂರ್ಣಾಙ್ಕ             s1,
                            ಪೂರ್ಣಾಙ್ಕ             p0,
                            ಪೂರ್ಣಾಙ್ಕ             p1,
                            ಪೂರ್ಣಾಙ್ಕ             d0,
                            ಪೂರ್ಣಾಙ್ಕ             d1,
                            dpct::queue_ptr stream) {
    im2col_sycl<ಭಿನ್ನ>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

ಶೂನ್ಯ ggml_sycl_op_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];
    ಧ್ರುವ ಭಿನ್ನ * src1_d = (ಧ್ರುವ ಭಿನ್ನ *)src1->data;
    ಭಿನ್ನ * dst_d = (ಭಿನ್ನ *)dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    ಧ್ರುವ int32_t s0 = ((ಧ್ರುವ int32_t*)(dst->op_params))[0];
    ಧ್ರುವ int32_t s1 = ((ಧ್ರುವ int32_t*)(dst->op_params))[1];
    ಧ್ರುವ int32_t p0 = ((ಧ್ರುವ int32_t*)(dst->op_params))[2];
    ಧ್ರುವ int32_t p1 = ((ಧ್ರುವ int32_t*)(dst->op_params))[3];
    ಧ್ರುವ int32_t d0 = ((ಧ್ರುವ int32_t*)(dst->op_params))[4];
    ಧ್ರುವ int32_t d1 = ((ಧ್ರುವ int32_t*)(dst->op_params))[5];

    ಧ್ರುವ ತಾರ್ಕಿಕ is_2D = ((ಧ್ರುವ int32_t*)(dst->op_params))[6] == 1;

    ಧ್ರುವ int64_t IC = src1->ne[is_2D ? 2 : 1];
    ಧ್ರುವ int64_t IH = is_2D ? src1->ne[1] : 1;
    ಧ್ರುವ int64_t IW =         src1->ne[0];

    ಧ್ರುವ int64_t KH = is_2D ? src0->ne[1] : 1;
    ಧ್ರುವ int64_t KW =         src0->ne[0];

    ಧ್ರುವ int64_t OH = is_2D ? dst->ne[2] : 1;
    ಧ್ರುವ int64_t OW =         dst->ne[1];

    ಧ್ರುವ int64_t IC_IH_IW = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    ಧ್ರುವ int64_t N        = src1->ne[is_2D ? 3 : 2];
    ಧ್ರುವ int64_t IH_IW    = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    ಯದಿ(dst->type == GGML_TYPE_F16) {
        im2col_sycl_f16(src1_d, (sycl::half *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1,
                        d0, d1, stream);
    } ಅನ್ಯಥಾ {
        im2col_sycl_f32(src1_d, (ಭಿನ್ನ *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
ಸ್ಥಿರ  ಶೂನ್ಯ im2col_3d_kernel(
        ಧ್ರುವ ಭಿನ್ನ * src, T * dst,
        int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
        int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
        int64_t OH_OW, int64_t KD_KH_KW, int64_t ID_IH_IW, int64_t KH_KW, int64_t IH_IW, int64_t IC_ID_IH_IW,
        int64_t IC_KD_KH_KW, int64_t OW_KD_KH_KW, int64_t OD_OH_OW_IC_KD_KH_KW, int64_t OH_OW_IC_KD_KH_KW,
        int64_t OW_IC_KD_KH_KW, int64_t N_OD_OH, int64_t OD_OH,
        int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
        ಪೂರ್ಣಾಙ್ಕ s0, ಪೂರ್ಣಾಙ್ಕ s1, ಪೂರ್ಣಾಙ್ಕ s2, ಪೂರ್ಣಾಙ್ಕ p0, ಪೂರ್ಣಾಙ್ಕ p1, ಪೂರ್ಣಾಙ್ಕ p2, ಪೂರ್ಣಾಙ್ಕ d0, ಪೂರ್ಣಾಙ್ಕ d1, ಪೂರ್ಣಾಙ್ಕ d2) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ int64_t i        = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    ಯದಿ (i >= IC_KD_KH_KW) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    GGML_UNUSED(N); GGML_UNUSED(OC); GGML_UNUSED(OH_OW); GGML_UNUSED(OD); GGML_UNUSED(OW); GGML_UNUSED(KD); GGML_UNUSED(KH);
    GGML_UNUSED(ID_IH_IW); GGML_UNUSED(IH_IW); GGML_UNUSED(IC_ID_IH_IW); GGML_UNUSED(OW_KD_KH_KW);

    ಧ್ರುವ int64_t iic = i / KD_KH_KW;
    ಧ್ರುವ int64_t ikd = (i - iic * KD_KH_KW) / KH_KW;
    ಧ್ರುವ int64_t ikh = (i - iic * KD_KH_KW - ikd * KH_KW) / KW;
    ಧ್ರುವ int64_t ikw = i % KW;

    ಧ್ರುವ int64_t iow = item_ct1.get_group(1);
    ಪ್ರತಿ (int64_t iz = item_ct1.get_group(0); iz < N_OD_OH; iz += MAX_GRIDDIM_Z) {
        ಧ್ರುವ int64_t in  = iz / OD_OH;
        ಧ್ರುವ int64_t iod = (iz - in*OD_OH) / OH;
        ಧ್ರುವ int64_t ioh = iz % OH;

        ಧ್ರುವ int64_t iiw = iow * s0 + ikw * d0 - p0;
        ಧ್ರುವ int64_t iih = ioh * s1 + ikh * d1 - p1;
        ಧ್ರುವ int64_t iid = iod * s2 + ikd * d2 - p2;

        ಧ್ರುವ int64_t offset_dst = in*OD_OH_OW_IC_KD_KH_KW + iod*OH_OW_IC_KD_KH_KW + ioh*OW_IC_KD_KH_KW + iow*IC_KD_KH_KW + iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw;

        ಯದಿ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
            dst[offset_dst] = 0.0f;
        } ಅನ್ಯಥಾ {
            ಧ್ರುವ int64_t offset_src = ((in * IC + iic) * stride_q) + (iid * stride_z) + (iih * stride_y) + (iiw * stride_x);
            dst[offset_dst] = src[offset_src];
        }
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_sycl(ಧ್ರುವ ಭಿನ್ನ *   src,
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
                           ಪೂರ್ಣಾಙ್ಕ             s0,
                           ಪೂರ್ಣಾಙ್ಕ             s1,
                           ಪೂರ್ಣಾಙ್ಕ             s2,
                           ಪೂರ್ಣಾಙ್ಕ             p0,
                           ಪೂರ್ಣಾಙ್ಕ             p1,
                           ಪೂರ್ಣಾಙ್ಕ             p2,
                           ಪೂರ್ಣಾಙ್ಕ             d0,
                           ಪೂರ್ಣಾಙ್ಕ             d1,
                           ಪೂರ್ಣಾಙ್ಕ             d2,
                           dpct::queue_ptr stream) {
    ಧ್ರುವ int64_t OH_OW = OH*OW;
    ಧ್ರುವ int64_t KD_KH_KW = KD*KH*KW;
    ಧ್ರುವ int64_t ID_IH_IW = ID*IH*IW;
    ಧ್ರುವ int64_t KH_KW = KH*KW;
    ಧ್ರುವ int64_t IH_IW = IH*IW;
    ಧ್ರುವ int64_t IC_KD_KH_KW = IC*KD*KH*KW;
    ಧ್ರುವ int64_t OW_KD_KH_KW = OW*KD*KH*KW;
    ಧ್ರುವ int64_t N_OD_OH = N*OD*OH;
    ಧ್ರುವ int64_t OD_OH = OD*OH;
    ಧ್ರುವ int64_t IC_ID_IH_IW = IC*ID*IH*IW;
    ಧ್ರುವ int64_t OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    ಧ್ರುವ int64_t OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    ಧ್ರುವ int64_t OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;
    ಧ್ರುವ int64_t ಅಂಕೆ_blocks = (IC_KD_KH_KW + SYCL_IM2COL_BLOCK_SIZE - 1) / SYCL_IM2COL_BLOCK_SIZE;
    dpct::dim3    block_nums(ಅಂಕೆ_blocks, OW, MIN(N_OD_OH, MAX_GRIDDIM_Z));
    /*
    DPCT1049:74: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    stream->parallel_for(sycl::nd_range<3>(block_nums * sycl::range<3>(1, 1, MIN(IC_KD_KH_KW, SYCL_IM2COL_BLOCK_SIZE)),
                                           sycl::range<3>(1, 1, MIN(IC_KD_KH_KW, SYCL_IM2COL_BLOCK_SIZE))),
                         [=](sycl::nd_item<3>) {
                             im2col_3d_kernel(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW, OH_OW, KD_KH_KW,
                                              ID_IH_IW, KH_KW, IH_IW, IC_ID_IH_IW, IC_KD_KH_KW, OW_KD_KH_KW,
                                              OD_OH_OW_IC_KD_KH_KW, OH_OW_IC_KD_KH_KW, OW_IC_KD_KH_KW, N_OD_OH, OD_OH,
                                              stride_q, stride_z, stride_y, stride_x, s0, s1, s2, p0, p1, p2, d0, d1,
                                              d2);
                         });
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_sycl_f16(ಧ್ರುವ ಭಿನ್ನ *   src,
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
                               ಪೂರ್ಣಾಙ್ಕ             s0,
                               ಪೂರ್ಣಾಙ್ಕ             s1,
                               ಪೂರ್ಣಾಙ್ಕ             s2,
                               ಪೂರ್ಣಾಙ್ಕ             p0,
                               ಪೂರ್ಣಾಙ್ಕ             p1,
                               ಪೂರ್ಣಾಙ್ಕ             p2,
                               ಪೂರ್ಣಾಙ್ಕ             d0,
                               ಪೂರ್ಣಾಙ್ಕ             d1,
                               ಪೂರ್ಣಾಙ್ಕ             d2,
                               dpct::queue_ptr stream) {
    im2col_3d_sycl<sycl::half>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW, stride_q, stride_z, stride_y,
                               stride_x, s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

ಸ್ಥಿರ ಶೂನ್ಯ im2col_3d_sycl_f32(ಧ್ರುವ ಭಿನ್ನ *   src,
                               ಭಿನ್ನ *         dst,
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
                               ಪೂರ್ಣಾಙ್ಕ             s0,
                               ಪೂರ್ಣಾಙ್ಕ             s1,
                               ಪೂರ್ಣಾಙ್ಕ             s2,
                               ಪೂರ್ಣಾಙ್ಕ             p0,
                               ಪೂರ್ಣಾಙ್ಕ             p1,
                               ಪೂರ್ಣಾಙ್ಕ             p2,
                               ಪೂರ್ಣಾಙ್ಕ             d0,
                               ಪೂರ್ಣಾಙ್ಕ             d1,
                               ಪೂರ್ಣಾಙ್ಕ             d2,
                               dpct::queue_ptr stream) {
    im2col_3d_sycl<ಭಿನ್ನ>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                          stride_q, stride_z, stride_y, stride_x,
                          s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

ಶೂನ್ಯ ggml_sycl_op_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಧ್ರುವ ggml_tensor * src0 = dst->src[0];
    ಧ್ರುವ ggml_tensor * src1 = dst->src[1];
    ಧ್ರುವ ಭಿನ್ನ * src1_d = (ಧ್ರುವ ಭಿನ್ನ *)src1->data;
    ಭಿನ್ನ * dst_d = (ಭಿನ್ನ *)dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಧ್ರುವ int32_t s0 = ((ಧ್ರುವ int32_t *)(dst->op_params))[0];
    ಧ್ರುವ int32_t s1 = ((ಧ್ರುವ int32_t *)(dst->op_params))[1];
    ಧ್ರುವ int32_t s2 = ((ಧ್ರುವ int32_t *)(dst->op_params))[2];
    ಧ್ರುವ int32_t p0 = ((ಧ್ರುವ int32_t *)(dst->op_params))[3];
    ಧ್ರುವ int32_t p1 = ((ಧ್ರುವ int32_t *)(dst->op_params))[4];
    ಧ್ರುವ int32_t p2 = ((ಧ್ರುವ int32_t *)(dst->op_params))[5];
    ಧ್ರುವ int32_t d0 = ((ಧ್ರುವ int32_t *)(dst->op_params))[6];
    ಧ್ರುವ int32_t d1 = ((ಧ್ರುವ int32_t *)(dst->op_params))[7];
    ಧ್ರುವ int32_t d2 = ((ಧ್ರುವ int32_t *)(dst->op_params))[8];
    ಧ್ರುವ int32_t IC = ((ಧ್ರುವ int32_t *)(dst->op_params))[9];

    ಧ್ರುವ int64_t N  = ne13 / IC;
    ಧ್ರುವ int64_t ID = ne12;
    ಧ್ರುವ int64_t IH = ne11;
    ಧ್ರುವ int64_t IW = ne10;

    ಧ್ರುವ int64_t OC = ne03 / IC;
    ಧ್ರುವ int64_t KD = ne02;
    ಧ್ರುವ int64_t KH = ne01;
    ಧ್ರುವ int64_t KW = ne00;

    ಧ್ರುವ int64_t OD = ne3 / N;
    ಧ್ರುವ int64_t OH = ne2;
    ಧ್ರುವ int64_t OW = ne1;

    ಧ್ರುವ size_t  es       = ggml_element_size(src1);
    ಧ್ರುವ int64_t stride_x = src1->nb[0] / es;
    ಧ್ರುವ int64_t stride_y = src1->nb[1] / es;
    ಧ್ರುವ int64_t stride_z = src1->nb[2] / es;
    ಧ್ರುವ int64_t stride_q = src1->nb[3] / es;

    ಯದಿ(dst->type == GGML_TYPE_F16) {
        im2col_3d_sycl_f16(src1_d, (sycl::half *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    } ಅನ್ಯಥಾ {
        im2col_3d_sycl_f32(src1_d, (ಭಿನ್ನ *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    }
}
