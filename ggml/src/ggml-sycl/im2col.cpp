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

#অন্তর্ভাৱ "im2col.hpp"

#define MAX_GRIDDIM_Z 65535

template <typename T>
স্থির  শূন্য im2col_kernel(
        ধ্রুৱ ভিন্ন * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW,
        পূর্ণাঙ্ক s0, পূর্ণাঙ্ক s1, পূর্ণাঙ্ক p0, পূর্ণাঙ্ক p1, পূর্ণাঙ্ক d0, পূর্ণাঙ্ক d1) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুৱ int64_t i        = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    যদি (i >= IC_KH_KW) {
        ঘূৰাই;
    }

    ধ্রুৱ int64_t iic = i / (KH_KW);
    ধ্রুৱ int64_t rem = i - iic * KH_KW;
    ধ্রুৱ int64_t ikh = rem / KW;
    ধ্রুৱ int64_t ikw = rem - ikh * KW;

    ধ্রুৱ int64_t iow = item_ct1.get_group(1);
    বাবে (int64_t iz = item_ct1.get_group(0); iz < N_OH; iz += MAX_GRIDDIM_Z) {
        ধ্রুৱ int64_t  in = iz / OH;
        ধ্রুৱ int64_t  ioh = iz - in * OH;

        ধ্রুৱ int64_t iiw = iow * s0 + ikw * d0 - p0;
        ধ্রুৱ int64_t iih = ioh * s1 + ikh * d1 - p1;

        ধ্রুৱ int64_t offset_dst =
            ((in * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

        যদি (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
            dst[offset_dst] = 0.0f;
        } অন্যথা {
            ধ্রুৱ int64_t offset_src = iic * IC_IH_IW + in * IH_IW;
            dst[offset_dst] = x[offset_src + iih * IW + iiw];
        }
    }

    GGML_UNUSED(IC);
    GGML_UNUSED(KH);
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
template <typename T>
স্থির শূন্য im2col_sycl(ধ্রুৱ ভিন্ন *   x,
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
                        পূর্ণাঙ্ক             s0,
                        পূর্ণাঙ্ক             s1,
                        পূর্ণাঙ্ক             p0,
                        পূর্ণাঙ্ক             p1,
                        পূর্ণাঙ্ক             d0,
                        পূর্ণাঙ্ক             d1,
                        dpct::queue_ptr stream) {
    ধ্রুৱ int64_t IC_KH_KW = IC * KH * KW;
    ধ্রুৱ int64_t অঙ্ক_blocks = (IC_KH_KW + SYCL_IM2COL_BLOCK_SIZE - 1) / SYCL_IM2COL_BLOCK_SIZE;
    ধ্রুৱ int64_t N_OH = N * OH;
    ধ্রুৱ int64_t KH_KW = KW*KH;
    dpct::dim3    block_nums(অঙ্ক_blocks, OW, MIN(N_OH, MAX_GRIDDIM_Z));
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

স্থির শূন্য im2col_sycl_f16(ধ্রুৱ ভিন্ন *   x,
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
                            পূর্ণাঙ্ক             s0,
                            পূর্ণাঙ্ক             s1,
                            পূর্ণাঙ্ক             p0,
                            পূর্ণাঙ্ক             p1,
                            পূর্ণাঙ্ক             d0,
                            পূর্ণাঙ্ক             d1,
                            dpct::queue_ptr stream) {
    im2col_sycl<sycl::half>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

স্থির শূন্য im2col_sycl_f32(ধ্রুৱ ভিন্ন *   x,
                            ভিন্ন *         dst,
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
                            পূর্ণাঙ্ক             s0,
                            পূর্ণাঙ্ক             s1,
                            পূর্ণাঙ্ক             p0,
                            পূর্ণাঙ্ক             p1,
                            পূর্ণাঙ্ক             d0,
                            পূর্ণাঙ্ক             d1,
                            dpct::queue_ptr stream) {
    im2col_sycl<ভিন্ন>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

শূন্য ggml_sycl_op_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুৱ ggml_tensor * src0 = dst->src[0];
    ধ্রুৱ ggml_tensor * src1 = dst->src[1];
    ধ্রুৱ ভিন্ন * src1_d = (ধ্রুৱ ভিন্ন *)src1->data;
    ভিন্ন * dst_d = (ভিন্ন *)dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    ধ্রুৱ int32_t s0 = ((ধ্রুৱ int32_t*)(dst->op_params))[0];
    ধ্রুৱ int32_t s1 = ((ধ্রুৱ int32_t*)(dst->op_params))[1];
    ধ্রুৱ int32_t p0 = ((ধ্রুৱ int32_t*)(dst->op_params))[2];
    ধ্রুৱ int32_t p1 = ((ধ্রুৱ int32_t*)(dst->op_params))[3];
    ধ্রুৱ int32_t d0 = ((ধ্রুৱ int32_t*)(dst->op_params))[4];
    ধ্রুৱ int32_t d1 = ((ধ্রুৱ int32_t*)(dst->op_params))[5];

    ধ্রুৱ তার্কিক is_2D = ((ধ্রুৱ int32_t*)(dst->op_params))[6] == 1;

    ধ্রুৱ int64_t IC = src1->ne[is_2D ? 2 : 1];
    ধ্রুৱ int64_t IH = is_2D ? src1->ne[1] : 1;
    ধ্রুৱ int64_t IW =         src1->ne[0];

    ধ্রুৱ int64_t KH = is_2D ? src0->ne[1] : 1;
    ধ্রুৱ int64_t KW =         src0->ne[0];

    ধ্রুৱ int64_t OH = is_2D ? dst->ne[2] : 1;
    ধ্রুৱ int64_t OW =         dst->ne[1];

    ধ্রুৱ int64_t IC_IH_IW = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    ধ্রুৱ int64_t N        = src1->ne[is_2D ? 3 : 2];
    ধ্রুৱ int64_t IH_IW    = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    যদি(dst->type == GGML_TYPE_F16) {
        im2col_sycl_f16(src1_d, (sycl::half *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1,
                        d0, d1, stream);
    } অন্যথা {
        im2col_sycl_f32(src1_d, (ভিন্ন *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
স্থির  শূন্য im2col_3d_kernel(
        ধ্রুৱ ভিন্ন * src, T * dst,
        int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
        int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
        int64_t OH_OW, int64_t KD_KH_KW, int64_t ID_IH_IW, int64_t KH_KW, int64_t IH_IW, int64_t IC_ID_IH_IW,
        int64_t IC_KD_KH_KW, int64_t OW_KD_KH_KW, int64_t OD_OH_OW_IC_KD_KH_KW, int64_t OH_OW_IC_KD_KH_KW,
        int64_t OW_IC_KD_KH_KW, int64_t N_OD_OH, int64_t OD_OH,
        int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
        পূর্ণাঙ্ক s0, পূর্ণাঙ্ক s1, পূর্ণাঙ্ক s2, পূর্ণাঙ্ক p0, পূর্ণাঙ্ক p1, পূর্ণাঙ্ক p2, পূর্ণাঙ্ক d0, পূর্ণাঙ্ক d1, পূর্ণাঙ্ক d2) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ধ্রুৱ int64_t i        = item_ct1.get_local_id(2) + item_ct1.get_group(2) * item_ct1.get_local_range(2);
    যদি (i >= IC_KD_KH_KW) {
        ঘূৰাই;
    }
    GGML_UNUSED(N); GGML_UNUSED(OC); GGML_UNUSED(OH_OW); GGML_UNUSED(OD); GGML_UNUSED(OW); GGML_UNUSED(KD); GGML_UNUSED(KH);
    GGML_UNUSED(ID_IH_IW); GGML_UNUSED(IH_IW); GGML_UNUSED(IC_ID_IH_IW); GGML_UNUSED(OW_KD_KH_KW);

    ধ্রুৱ int64_t iic = i / KD_KH_KW;
    ধ্রুৱ int64_t ikd = (i - iic * KD_KH_KW) / KH_KW;
    ধ্রুৱ int64_t ikh = (i - iic * KD_KH_KW - ikd * KH_KW) / KW;
    ধ্রুৱ int64_t ikw = i % KW;

    ধ্রুৱ int64_t iow = item_ct1.get_group(1);
    বাবে (int64_t iz = item_ct1.get_group(0); iz < N_OD_OH; iz += MAX_GRIDDIM_Z) {
        ধ্রুৱ int64_t in  = iz / OD_OH;
        ধ্রুৱ int64_t iod = (iz - in*OD_OH) / OH;
        ধ্রুৱ int64_t ioh = iz % OH;

        ধ্রুৱ int64_t iiw = iow * s0 + ikw * d0 - p0;
        ধ্রুৱ int64_t iih = ioh * s1 + ikh * d1 - p1;
        ধ্রুৱ int64_t iid = iod * s2 + ikd * d2 - p2;

        ধ্রুৱ int64_t offset_dst = in*OD_OH_OW_IC_KD_KH_KW + iod*OH_OW_IC_KD_KH_KW + ioh*OW_IC_KD_KH_KW + iow*IC_KD_KH_KW + iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw;

        যদি (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
            dst[offset_dst] = 0.0f;
        } অন্যথা {
            ধ্রুৱ int64_t offset_src = ((in * IC + iic) * stride_q) + (iid * stride_z) + (iih * stride_y) + (iiw * stride_x);
            dst[offset_dst] = src[offset_src];
        }
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
template <typename T>
স্থির শূন্য im2col_3d_sycl(ধ্রুৱ ভিন্ন *   src,
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
                           পূর্ণাঙ্ক             s0,
                           পূর্ণাঙ্ক             s1,
                           পূর্ণাঙ্ক             s2,
                           পূর্ণাঙ্ক             p0,
                           পূর্ণাঙ্ক             p1,
                           পূর্ণাঙ্ক             p2,
                           পূর্ণাঙ্ক             d0,
                           পূর্ণাঙ্ক             d1,
                           পূর্ণাঙ্ক             d2,
                           dpct::queue_ptr stream) {
    ধ্রুৱ int64_t OH_OW = OH*OW;
    ধ্রুৱ int64_t KD_KH_KW = KD*KH*KW;
    ধ্রুৱ int64_t ID_IH_IW = ID*IH*IW;
    ধ্রুৱ int64_t KH_KW = KH*KW;
    ধ্রুৱ int64_t IH_IW = IH*IW;
    ধ্রুৱ int64_t IC_KD_KH_KW = IC*KD*KH*KW;
    ধ্রুৱ int64_t OW_KD_KH_KW = OW*KD*KH*KW;
    ধ্রুৱ int64_t N_OD_OH = N*OD*OH;
    ধ্রুৱ int64_t OD_OH = OD*OH;
    ধ্রুৱ int64_t IC_ID_IH_IW = IC*ID*IH*IW;
    ধ্রুৱ int64_t OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    ধ্রুৱ int64_t OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    ধ্রুৱ int64_t OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;
    ধ্রুৱ int64_t অঙ্ক_blocks = (IC_KD_KH_KW + SYCL_IM2COL_BLOCK_SIZE - 1) / SYCL_IM2COL_BLOCK_SIZE;
    dpct::dim3    block_nums(অঙ্ক_blocks, OW, MIN(N_OD_OH, MAX_GRIDDIM_Z));
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

স্থির শূন্য im2col_3d_sycl_f16(ধ্রুৱ ভিন্ন *   src,
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
                               পূর্ণাঙ্ক             s0,
                               পূর্ণাঙ্ক             s1,
                               পূর্ণাঙ্ক             s2,
                               পূর্ণাঙ্ক             p0,
                               পূর্ণাঙ্ক             p1,
                               পূর্ণাঙ্ক             p2,
                               পূর্ণাঙ্ক             d0,
                               পূর্ণাঙ্ক             d1,
                               পূর্ণাঙ্ক             d2,
                               dpct::queue_ptr stream) {
    im2col_3d_sycl<sycl::half>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW, stride_q, stride_z, stride_y,
                               stride_x, s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

স্থির শূন্য im2col_3d_sycl_f32(ধ্রুৱ ভিন্ন *   src,
                               ভিন্ন *         dst,
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
                               পূর্ণাঙ্ক             s0,
                               পূর্ণাঙ্ক             s1,
                               পূর্ণাঙ্ক             s2,
                               পূর্ণাঙ্ক             p0,
                               পূর্ণাঙ্ক             p1,
                               পূর্ণাঙ্ক             p2,
                               পূর্ণাঙ্ক             d0,
                               পূর্ণাঙ্ক             d1,
                               পূর্ণাঙ্ক             d2,
                               dpct::queue_ptr stream) {
    im2col_3d_sycl<ভিন্ন>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                          stride_q, stride_z, stride_y, stride_x,
                          s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

শূন্য ggml_sycl_op_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুৱ ggml_tensor * src0 = dst->src[0];
    ধ্রুৱ ggml_tensor * src1 = dst->src[1];
    ধ্রুৱ ভিন্ন * src1_d = (ধ্রুৱ ভিন্ন *)src1->data;
    ভিন্ন * dst_d = (ভিন্ন *)dst->data;
    dpct::queue_ptr     stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    GGML_TENSOR_দ্বিমিক_OP_LOCALS

    ধ্রুৱ int32_t s0 = ((ধ্রুৱ int32_t *)(dst->op_params))[0];
    ধ্রুৱ int32_t s1 = ((ধ্রুৱ int32_t *)(dst->op_params))[1];
    ধ্রুৱ int32_t s2 = ((ধ্রুৱ int32_t *)(dst->op_params))[2];
    ধ্রুৱ int32_t p0 = ((ধ্রুৱ int32_t *)(dst->op_params))[3];
    ধ্রুৱ int32_t p1 = ((ধ্রুৱ int32_t *)(dst->op_params))[4];
    ধ্রুৱ int32_t p2 = ((ধ্রুৱ int32_t *)(dst->op_params))[5];
    ধ্রুৱ int32_t d0 = ((ধ্রুৱ int32_t *)(dst->op_params))[6];
    ধ্রুৱ int32_t d1 = ((ধ্রুৱ int32_t *)(dst->op_params))[7];
    ধ্রুৱ int32_t d2 = ((ধ্রুৱ int32_t *)(dst->op_params))[8];
    ধ্রুৱ int32_t IC = ((ধ্রুৱ int32_t *)(dst->op_params))[9];

    ধ্রুৱ int64_t N  = ne13 / IC;
    ধ্রুৱ int64_t ID = ne12;
    ধ্রুৱ int64_t IH = ne11;
    ধ্রুৱ int64_t IW = ne10;

    ধ্রুৱ int64_t OC = ne03 / IC;
    ধ্রুৱ int64_t KD = ne02;
    ধ্রুৱ int64_t KH = ne01;
    ধ্রুৱ int64_t KW = ne00;

    ধ্রুৱ int64_t OD = ne3 / N;
    ধ্রুৱ int64_t OH = ne2;
    ধ্রুৱ int64_t OW = ne1;

    ধ্রুৱ size_t  es       = ggml_element_size(src1);
    ধ্রুৱ int64_t stride_x = src1->nb[0] / es;
    ধ্রুৱ int64_t stride_y = src1->nb[1] / es;
    ধ্রুৱ int64_t stride_z = src1->nb[2] / es;
    ধ্রুৱ int64_t stride_q = src1->nb[3] / es;

    যদি(dst->type == GGML_TYPE_F16) {
        im2col_3d_sycl_f16(src1_d, (sycl::half *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    } অন্যথা {
        im2col_3d_sycl_f32(src1_d, (ভিন্ন *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    }
}
