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

#ਅਨ੍ਤਰ੍ਭਾਵ "im2col.hpp"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX_GRIDDIM_Z 65535

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ  ਖਾਲੀ im2col_kernel(
        ਅਚਲ ਦਸ਼ਮਲਵ * x, T * dst,
        int64_t IC, int64_t IW, int64_t IH, int64_t OH, int64_t OW, int64_t KW, int64_t KH,
        int64_t IC_IH_IW, int64_t IH_IW, int64_t N_OH, int64_t KH_KW, int64_t IC_KH_KW,
        ਪੂਰਨ s0, ਪੂਰਨ s1, ਪੂਰਨ p0, ਪੂਰਨ p1, ਪੂਰਨ d0, ਪੂਰਨ d1) {
    ਸਵੈਚਾਲਿਤ          ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ int64_t i        = ਵਸਤੂ_ct1.get_local_id(2) + ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
    ਜੇ (i >= IC_KH_KW) {
        ਵਾਪਸ;
    }

    ਅਚਲ int64_t iic = i / (KH_KW);
    ਅਚਲ int64_t rem = i - iic * KH_KW;
    ਅਚਲ int64_t ikh = rem / KW;
    ਅਚਲ int64_t ikw = rem - ikh * KW;

    ਅਚਲ int64_t iow = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਲਈ (int64_t iz = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0); iz < N_OH; iz += MAX_GRIDDIM_Z) {
        ਅਚਲ int64_t  ਵਿੱਚ = iz / OH;
        ਅਚਲ int64_t  ioh = iz - ਵਿੱਚ * OH;

        ਅਚਲ int64_t iiw = iow * s0 + ikw * d0 - p0;
        ਅਚਲ int64_t iih = ioh * s1 + ikh * d1 - p1;

        ਅਚਲ int64_t ਵਿੱਥ_dst =
            ((ਵਿੱਚ * OH + ioh) * OW + iow) * IC_KH_KW + iic * KH_KW + ikh * KW + ikw;

        ਜੇ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW) {
            dst[ਵਿੱਥ_dst] = 0.0f;
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ int64_t ਵਿੱਥ_src = iic * IC_IH_IW + ਵਿੱਚ * IH_IW;
            dst[ਵਿੱਥ_dst] = x[ਵਿੱਥ_src + iih * IW + iiw];
        }
    }

    GGML_ਅਣਵਰਤਿਆ(IC);
    GGML_ਅਣਵਰਤਿਆ(KH);
}

// im2col: [N, IC, IH, IW] => [N, OH, OW, IC*KH*KW]
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ im2col_sycl(ਅਚਲ ਦਸ਼ਮਲਵ *   x,
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
                        ਪੂਰਨ             s0,
                        ਪੂਰਨ             s1,
                        ਪੂਰਨ             p0,
                        ਪੂਰਨ             p1,
                        ਪੂਰਨ             d0,
                        ਪੂਰਨ             d1,
                        dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t IC_KH_KW = IC * KH * KW;
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (IC_KH_KW + SYCL_IM2_ਖਾਨਾ_ਖੰਡ_ਮਾਪ - 1) / SYCL_IM2_ਖਾਨਾ_ਖੰਡ_ਮਾਪ;
    ਅਚਲ int64_t N_OH = N * OH;
    ਅਚਲ int64_t KH_KW = KW*KH;
    dpct::dim3    ਖੰਡ_nums(ਅੰਕ_ਖੰਡਾਂ, OW, MIN(N_OH, MAX_GRIDDIM_Z));
    /*
    DPCT1049:73: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    stream->parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * sycl::range<3>(1, 1, MIN(IC_KH_KW, SYCL_IM2_ਖਾਨਾ_ਖੰਡ_ਮਾਪ)),
                                           sycl::range<3>(1, 1, MIN(IC_KH_KW, SYCL_IM2_ਖਾਨਾ_ਖੰਡ_ਮਾਪ))),
                         [=](sycl::nd_ਵਸਤੂ<3>) {
                             im2col_kernel(x, dst, IC, IW, IH, OH, OW, KW, KH, IC_IH_IW, IH_IW, N_OH, KH_KW, IC_KH_KW,
                                           s0, s1, p0, p1, d0, d1);
                         });
}

ਸਥਿਰ ਖਾਲੀ im2col_sycl_f16(ਅਚਲ ਦਸ਼ਮਲਵ *   x,
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
                            ਪੂਰਨ             s0,
                            ਪੂਰਨ             s1,
                            ਪੂਰਨ             p0,
                            ਪੂਰਨ             p1,
                            ਪੂਰਨ             d0,
                            ਪੂਰਨ             d1,
                            dpct::ਕਤਾਰ_ptr stream) {
    im2col_sycl<sycl::half>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

ਸਥਿਰ ਖਾਲੀ im2col_sycl_f32(ਅਚਲ ਦਸ਼ਮਲਵ *   x,
                            ਦਸ਼ਮਲਵ *         dst,
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
                            ਪੂਰਨ             s0,
                            ਪੂਰਨ             s1,
                            ਪੂਰਨ             p0,
                            ਪੂਰਨ             p1,
                            ਪੂਰਨ             d0,
                            ਪੂਰਨ             d1,
                            dpct::ਕਤਾਰ_ptr stream) {
    im2col_sycl<ਦਸ਼ਮਲਵ>(x, dst, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
}

ਖਾਲੀ ggml_sycl_op_im2col(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅਚਲ ਦਸ਼ਮਲਵ * src1_d = (ਅਚਲ ਦਸ਼ਮਲਵ *)src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_d = (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ;
    dpct::ਕਤਾਰ_ptr     stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t*)(dst->op_params))[0];
    ਅਚਲ int32_t s1 = ((ਅਚਲ int32_t*)(dst->op_params))[1];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t*)(dst->op_params))[2];
    ਅਚਲ int32_t p1 = ((ਅਚਲ int32_t*)(dst->op_params))[3];
    ਅਚਲ int32_t d0 = ((ਅਚਲ int32_t*)(dst->op_params))[4];
    ਅਚਲ int32_t d1 = ((ਅਚਲ int32_t*)(dst->op_params))[5];

    ਅਚਲ ਤਾਰਕਿਕ is_2D = ((ਅਚਲ int32_t*)(dst->op_params))[6] == 1;

    ਅਚਲ int64_t IC = src1->ne[is_2D ? 2 : 1];
    ਅਚਲ int64_t IH = is_2D ? src1->ne[1] : 1;
    ਅਚਲ int64_t IW =         src1->ne[0];

    ਅਚਲ int64_t KH = is_2D ? src0->ne[1] : 1;
    ਅਚਲ int64_t KW =         src0->ne[0];

    ਅਚਲ int64_t OH = is_2D ? dst->ne[2] : 1;
    ਅਚਲ int64_t OW =         dst->ne[1];

    ਅਚਲ int64_t IC_IH_IW = src1->nb[is_2D ? 2 : 1] / 4; // nb is byte offset, src is type float32
    ਅਚਲ int64_t N        = src1->ne[is_2D ? 3 : 2];
    ਅਚਲ int64_t IH_IW    = src1->nb[is_2D ? 3 : 2] / 4; // nb is byte offset, src is type float32

    ਜੇ(dst->type == GGML_TYPE_F16) {
        im2col_sycl_f16(src1_d, (sycl::half *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1,
                        d0, d1, stream);
    } ਨਹੀਂਤਾਂ {
        im2col_sycl_f32(src1_d, (ਦਸ਼ਮਲਵ *) dst_d, IW, IH, OW, OH, KW, KH, IC, N, IC_IH_IW, IH_IW, s0, s1, p0, p1, d0, d1, stream);
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ  ਖਾਲੀ im2col_3d_kernel(
        ਅਚਲ ਦਸ਼ਮਲਵ * src, T * dst,
        int64_t N, int64_t IC, int64_t ID, int64_t IH, int64_t IW, int64_t OC,
        int64_t KD, int64_t KH, int64_t KW, int64_t OD, int64_t OH, int64_t OW,
        int64_t OH_OW, int64_t KD_KH_KW, int64_t ID_IH_IW, int64_t KH_KW, int64_t IH_IW, int64_t IC_ID_IH_IW,
        int64_t IC_KD_KH_KW, int64_t OW_KD_KH_KW, int64_t OD_OH_OW_IC_KD_KH_KW, int64_t OH_OW_IC_KD_KH_KW,
        int64_t OW_IC_KD_KH_KW, int64_t N_OD_OH, int64_t OD_OH,
        int64_t stride_q, int64_t stride_z, int64_t stride_y, int64_t stride_x,
        ਪੂਰਨ s0, ਪੂਰਨ s1, ਪੂਰਨ s2, ਪੂਰਨ p0, ਪੂਰਨ p1, ਪੂਰਨ p2, ਪੂਰਨ d0, ਪੂਰਨ d1, ਪੂਰਨ d2) {
    ਸਵੈਚਾਲਿਤ          ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ int64_t i        = ਵਸਤੂ_ct1.get_local_id(2) + ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
    ਜੇ (i >= IC_KD_KH_KW) {
        ਵਾਪਸ;
    }
    GGML_ਅਣਵਰਤਿਆ(N); GGML_ਅਣਵਰਤਿਆ(OC); GGML_ਅਣਵਰਤਿਆ(OH_OW); GGML_ਅਣਵਰਤਿਆ(OD); GGML_ਅਣਵਰਤਿਆ(OW); GGML_ਅਣਵਰਤਿਆ(KD); GGML_ਅਣਵਰਤਿਆ(KH);
    GGML_ਅਣਵਰਤਿਆ(ID_IH_IW); GGML_ਅਣਵਰਤਿਆ(IH_IW); GGML_ਅਣਵਰਤਿਆ(IC_ID_IH_IW); GGML_ਅਣਵਰਤਿਆ(OW_KD_KH_KW);

    ਅਚਲ int64_t iic = i / KD_KH_KW;
    ਅਚਲ int64_t ikd = (i - iic * KD_KH_KW) / KH_KW;
    ਅਚਲ int64_t ikh = (i - iic * KD_KH_KW - ikd * KH_KW) / KW;
    ਅਚਲ int64_t ikw = i % KW;

    ਅਚਲ int64_t iow = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਲਈ (int64_t iz = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0); iz < N_OD_OH; iz += MAX_GRIDDIM_Z) {
        ਅਚਲ int64_t ਵਿੱਚ  = iz / OD_OH;
        ਅਚਲ int64_t iod = (iz - ਵਿੱਚ*OD_OH) / OH;
        ਅਚਲ int64_t ioh = iz % OH;

        ਅਚਲ int64_t iiw = iow * s0 + ikw * d0 - p0;
        ਅਚਲ int64_t iih = ioh * s1 + ikh * d1 - p1;
        ਅਚਲ int64_t iid = iod * s2 + ikd * d2 - p2;

        ਅਚਲ int64_t ਵਿੱਥ_dst = ਵਿੱਚ*OD_OH_OW_IC_KD_KH_KW + iod*OH_OW_IC_KD_KH_KW + ioh*OW_IC_KD_KH_KW + iow*IC_KD_KH_KW + iic*KD_KH_KW + ikd * KH_KW + ikh*KW + ikw;

        ਜੇ (iih < 0 || iih >= IH || iiw < 0 || iiw >= IW || iid < 0 || iid >= ID) {
            dst[ਵਿੱਥ_dst] = 0.0f;
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ int64_t ਵਿੱਥ_src = ((ਵਿੱਚ * IC + iic) * stride_q) + (iid * stride_z) + (iih * stride_y) + (iiw * stride_x);
            dst[ਵਿੱਥ_dst] = src[ਵਿੱਥ_src];
        }
    }
}

// [N*IC, ID, IH, IW] => [N*OD, OH, OW, IC * KD * KH * KW]
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T>
ਸਥਿਰ ਖਾਲੀ im2col_3d_sycl(ਅਚਲ ਦਸ਼ਮਲਵ *   src,
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
                           ਪੂਰਨ             s0,
                           ਪੂਰਨ             s1,
                           ਪੂਰਨ             s2,
                           ਪੂਰਨ             p0,
                           ਪੂਰਨ             p1,
                           ਪੂਰਨ             p2,
                           ਪੂਰਨ             d0,
                           ਪੂਰਨ             d1,
                           ਪੂਰਨ             d2,
                           dpct::ਕਤਾਰ_ptr stream) {
    ਅਚਲ int64_t OH_OW = OH*OW;
    ਅਚਲ int64_t KD_KH_KW = KD*KH*KW;
    ਅਚਲ int64_t ID_IH_IW = ID*IH*IW;
    ਅਚਲ int64_t KH_KW = KH*KW;
    ਅਚਲ int64_t IH_IW = IH*IW;
    ਅਚਲ int64_t IC_KD_KH_KW = IC*KD*KH*KW;
    ਅਚਲ int64_t OW_KD_KH_KW = OW*KD*KH*KW;
    ਅਚਲ int64_t N_OD_OH = N*OD*OH;
    ਅਚਲ int64_t OD_OH = OD*OH;
    ਅਚਲ int64_t IC_ID_IH_IW = IC*ID*IH*IW;
    ਅਚਲ int64_t OD_OH_OW_IC_KD_KH_KW = OD*OH*OW*IC*KD*KH*KW;
    ਅਚਲ int64_t OH_OW_IC_KD_KH_KW = OH*OW*IC*KD*KH*KW;
    ਅਚਲ int64_t OW_IC_KD_KH_KW = OW*IC*KD*KH*KW;
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (IC_KD_KH_KW + SYCL_IM2_ਖਾਨਾ_ਖੰਡ_ਮਾਪ - 1) / SYCL_IM2_ਖਾਨਾ_ਖੰਡ_ਮਾਪ;
    dpct::dim3    ਖੰਡ_nums(ਅੰਕ_ਖੰਡਾਂ, OW, MIN(N_OD_OH, MAX_GRIDDIM_Z));
    /*
    DPCT1049:74: The work-group size passed to the SYCL kernel may exceed the limit. To get the device limit, query info::device::max_work_group_size. Adjust the work-group size if needed.
    */
    stream->parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * sycl::range<3>(1, 1, MIN(IC_KD_KH_KW, SYCL_IM2_ਖਾਨਾ_ਖੰਡ_ਮਾਪ)),
                                           sycl::range<3>(1, 1, MIN(IC_KD_KH_KW, SYCL_IM2_ਖਾਨਾ_ਖੰਡ_ਮਾਪ))),
                         [=](sycl::nd_ਵਸਤੂ<3>) {
                             im2col_3d_kernel(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW, OH_OW, KD_KH_KW,
                                              ID_IH_IW, KH_KW, IH_IW, IC_ID_IH_IW, IC_KD_KH_KW, OW_KD_KH_KW,
                                              OD_OH_OW_IC_KD_KH_KW, OH_OW_IC_KD_KH_KW, OW_IC_KD_KH_KW, N_OD_OH, OD_OH,
                                              stride_q, stride_z, stride_y, stride_x, s0, s1, s2, p0, p1, p2, d0, d1,
                                              d2);
                         });
}

ਸਥਿਰ ਖਾਲੀ im2col_3d_sycl_f16(ਅਚਲ ਦਸ਼ਮਲਵ *   src,
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
                               ਪੂਰਨ             s0,
                               ਪੂਰਨ             s1,
                               ਪੂਰਨ             s2,
                               ਪੂਰਨ             p0,
                               ਪੂਰਨ             p1,
                               ਪੂਰਨ             p2,
                               ਪੂਰਨ             d0,
                               ਪੂਰਨ             d1,
                               ਪੂਰਨ             d2,
                               dpct::ਕਤਾਰ_ptr stream) {
    im2col_3d_sycl<sycl::half>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW, stride_q, stride_z, stride_y,
                               stride_x, s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

ਸਥਿਰ ਖਾਲੀ im2col_3d_sycl_f32(ਅਚਲ ਦਸ਼ਮਲਵ *   src,
                               ਦਸ਼ਮਲਵ *         dst,
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
                               ਪੂਰਨ             s0,
                               ਪੂਰਨ             s1,
                               ਪੂਰਨ             s2,
                               ਪੂਰਨ             p0,
                               ਪੂਰਨ             p1,
                               ਪੂਰਨ             p2,
                               ਪੂਰਨ             d0,
                               ਪੂਰਨ             d1,
                               ਪੂਰਨ             d2,
                               dpct::ਕਤਾਰ_ptr stream) {
    im2col_3d_sycl<ਦਸ਼ਮਲਵ>(src, dst, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                          stride_q, stride_z, stride_y, stride_x,
                          s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
}

ਖਾਲੀ ggml_sycl_op_im2col_3d(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];
    ਅਚਲ ਦਸ਼ਮਲਵ * src1_d = (ਅਚਲ ਦਸ਼ਮਲਵ *)src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_d = (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ;
    dpct::ਕਤਾਰ_ptr     stream = ctx.stream();

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F16 || dst->type == GGML_TYPE_F32);

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ int32_t s0 = ((ਅਚਲ int32_t *)(dst->op_params))[0];
    ਅਚਲ int32_t s1 = ((ਅਚਲ int32_t *)(dst->op_params))[1];
    ਅਚਲ int32_t s2 = ((ਅਚਲ int32_t *)(dst->op_params))[2];
    ਅਚਲ int32_t p0 = ((ਅਚਲ int32_t *)(dst->op_params))[3];
    ਅਚਲ int32_t p1 = ((ਅਚਲ int32_t *)(dst->op_params))[4];
    ਅਚਲ int32_t p2 = ((ਅਚਲ int32_t *)(dst->op_params))[5];
    ਅਚਲ int32_t d0 = ((ਅਚਲ int32_t *)(dst->op_params))[6];
    ਅਚਲ int32_t d1 = ((ਅਚਲ int32_t *)(dst->op_params))[7];
    ਅਚਲ int32_t d2 = ((ਅਚਲ int32_t *)(dst->op_params))[8];
    ਅਚਲ int32_t IC = ((ਅਚਲ int32_t *)(dst->op_params))[9];

    ਅਚਲ int64_t N  = ne13 / IC;
    ਅਚਲ int64_t ID = ne12;
    ਅਚਲ int64_t IH = ne11;
    ਅਚਲ int64_t IW = ne10;

    ਅਚਲ int64_t OC = ne03 / IC;
    ਅਚਲ int64_t KD = ne02;
    ਅਚਲ int64_t KH = ne01;
    ਅਚਲ int64_t KW = ne00;

    ਅਚਲ int64_t OD = ne3 / N;
    ਅਚਲ int64_t OH = ne2;
    ਅਚਲ int64_t OW = ne1;

    ਅਚਲ size_t  es       = ggml_element_ਮਾਪ(src1);
    ਅਚਲ int64_t stride_x = src1->nb[0] / es;
    ਅਚਲ int64_t stride_y = src1->nb[1] / es;
    ਅਚਲ int64_t stride_z = src1->nb[2] / es;
    ਅਚਲ int64_t stride_q = src1->nb[3] / es;

    ਜੇ(dst->type == GGML_TYPE_F16) {
        im2col_3d_sycl_f16(src1_d, (sycl::half *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    } ਨਹੀਂਤਾਂ {
        im2col_3d_sycl_f32(src1_d, (ਦਸ਼ਮਲਵ *) dst_d, N, IC, ID, IH, IW, OC, KD, KH, KW, OD, OH, OW,
                           stride_q, stride_z, stride_y, stride_x,
                           s0, s1, s2, p0, p1, p2, d0, d1, d2, stream);
    }
}
