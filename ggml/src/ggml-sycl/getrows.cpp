//
// MIT license
// Copyright (C) 2024 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "dequantize.hpp"
#ಅನ್ತರ್ಭಾವ "getrows.hpp"


ಮಾದರಿ<ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qr, dequantize_kernel_t dequantize_kernel, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ k_get_ಅಡ್ಡಸಾಲುಗಳು(
            ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src0, ಸ್ಥಿರಾಂಕ int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1/*, size_t s13*/) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i00 = (ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2) +
                     ಅಂಶ_ct1.get_local_id(2)) *
                    2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i10 = ಅಂಶ_ct1.get_local_range(1) * ಅಂಶ_ct1.get_ಗುಂಪು(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i11 = (ಅಂಶ_ct1.get_ಗುಂಪು(0) * ಅಂಶ_ct1.get_local_range(0) +
                     ಅಂಶ_ct1.get_local_id(0)) /
                    ne12;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i12 = (ಅಂಶ_ct1.get_ಗುಂಪು(0) * ಅಂಶ_ct1.get_local_range(0) +
                     ಅಂಶ_ct1.get_local_id(0)) %
                    ne12;

    ಆದರೆ (i00 >= ne00) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_ಅಡ್ಡಸಾಲು = dst + i10*s1 + i11*s2 + i12*s3;
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src0_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib = i00/qk; // block index
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = (i00%qk)/qr; // quant index
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iybs = i00 - i00%qk; // dst block start index
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(src0_ಅಡ್ಡಸಾಲು, ib, iqs, v);

    dst_ಅಡ್ಡಸಾಲು[iybs + iqs + 0] = v.x();
    dst_ಅಡ್ಡಸಾಲು[iybs + iqs + y_ಸ್ಥಾನಾಂತರ] = v.y();
}

ಮಾದರಿ<ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qr, dequantize_kernel_f32_t dequantize_kernel, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ k_get_ಅಡ್ಡಸಾಲುಗಳು_f32(
            ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src0, ಸ್ಥಿರಾಂಕ int32_t * src1, dst_t * dst,
            int64_t ne00,
            int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i00 = (ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2) +
                     ಅಂಶ_ct1.get_local_id(2)) *
                    2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i10 = ಅಂಶ_ct1.get_local_range(1) * ಅಂಶ_ct1.get_ಗುಂಪು(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i11 = (ಅಂಶ_ct1.get_ಗುಂಪು(0) * ಅಂಶ_ct1.get_local_range(0) +
                     ಅಂಶ_ct1.get_local_id(0)) /
                    ne12;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i12 = (ಅಂಶ_ct1.get_ಗುಂಪು(0) * ಅಂಶ_ct1.get_local_range(0) +
                     ಅಂಶ_ct1.get_local_id(0)) %
                    ne12;

    ಆದರೆ (i00 >= ne00) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_ಅಡ್ಡಸಾಲು = dst + i10*s1 + i11*s2 + i12*s3;
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src0_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib = i00/qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = (i00%qk)/qr;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iybs = i00 - i00%qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = qr == 1 ? 1 : qk/2;

    ದಶಮಾಂಶ v0;
    ದಶಮಾಂಶ v1;
    dequantize_kernel(src0_ಅಡ್ಡಸಾಲು, ib, iqs, v0, v1);

    dst_ಅಡ್ಡಸಾಲು[iybs + iqs + 0] = (dst_t) v0;
    dst_ಅಡ್ಡಸಾಲು[iybs + iqs + y_ಸ್ಥಾನಾಂತರ] = (dst_t) v1;
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ k_get_ಅಡ್ಡಸಾಲುಗಳು_float(
            ಸ್ಥಿರಾಂಕ src0_t * src0, ಸ್ಥಿರಾಂಕ int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1/*, size_t s13*/) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i00 = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2) +
                    ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i10 = ಅಂಶ_ct1.get_local_range(1) * ಅಂಶ_ct1.get_ಗುಂಪು(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i11 = (ಅಂಶ_ct1.get_ಗುಂಪು(0) * ಅಂಶ_ct1.get_local_range(0) +
                     ಅಂಶ_ct1.get_local_id(0)) /
                    ne12;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i12 = (ಅಂಶ_ct1.get_ಗುಂಪು(0) * ಅಂಶ_ct1.get_local_range(0) +
                     ಅಂಶ_ct1.get_local_id(0)) %
                    ne12;

    ಆದರೆ (i00 >= ne00) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_ಅಡ್ಡಸಾಲು = dst + i10*s1 + i11*s2 + i12*s3;
    ಸ್ಥಿರಾಂಕ src0_t * src0_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ src0_t *)((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_ಅಡ್ಡಸಾಲು[i00] = src0_ಅಡ್ಡಸಾಲು[i00];
}

// src0 is in reorder layout: all qs of the tensor, then all d
ಮಾದರಿ<ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qr, dequantize_kernel_t_reorder dequantize_kernel, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ k_get_ಅಡ್ಡಸಾಲುಗಳು_reorder(
            ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src0_qs, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src0_d, ಸ್ಥಿರಾಂಕ int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t s10, size_t s11, size_t s12,
            ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i00 = (ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2) +
                     ಅಂಶ_ct1.get_local_id(2)) *
                    2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i10 = ಅಂಶ_ct1.get_local_range(1) * ಅಂಶ_ct1.get_ಗುಂಪು(1) +
                    ಅಂಶ_ct1.get_local_id(1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i11 = (ಅಂಶ_ct1.get_ಗುಂಪು(0) * ಅಂಶ_ct1.get_local_range(0) +
                     ಅಂಶ_ct1.get_local_id(0)) /
                    ne12;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i12 = (ಅಂಶ_ct1.get_ಗುಂಪು(0) * ಅಂಶ_ct1.get_local_range(0) +
                     ಅಂಶ_ct1.get_local_id(0)) %
                    ne12;

    ಆದರೆ (i00 >= ne00) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_ಅಡ್ಡಸಾಲು = dst + i10*s1 + i11*s2 + i12*s3;

    ಸ್ಥಿರಾಂಕ int64_t ib = (int64_t) i01*(ne00/qk) + i00/qk; // block index in the whole tensor
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs = (i00%qk)/qr;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iybs = i00 - i00%qk;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ y_ಸ್ಥಾನಾಂತರ = qr == 1 ? 1 : qk/2;

    dfloat2 v;
    dequantize_kernel(src0_d, ib, src0_qs, ib*(qk/qr) + iqs, v);

    dst_ಅಡ್ಡಸಾಲು[iybs + iqs + 0] = v.x();
    dst_ಅಡ್ಡಸಾಲು[iybs + iqs + y_ಸ್ಥಾನಾಂತರ] = v.y();
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qr, dequantize_kernel_t dq>
ಸ್ಥಿರ ಶೂನ್ಯ get_ಅಡ್ಡಸಾಲುಗಳು_sycl(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor *src0, ಸ್ಥಿರಾಂಕ ggml_tensor *src1,
                          ggml_tensor *dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *src0_dd,
                          ಸ್ಥಿರಾಂಕ int32_t *src1_dd, ದಶಮಾಂಶ *dst_dd,
                          ಸರತಿ_ptr stream) {

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (ne00 + 2*SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ - 1) / (2*SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(ne11 * ne12, ne10, ಖಂಡ_ಅಂಕೆ_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    ಸ್ಥಿರಾಂಕ size_t s1 = nb1 / ggml_element_ಗಾತ್ರ(dst);
    ಸ್ಥಿರಾಂಕ size_t s2 = nb2 / ggml_element_ಗಾತ್ರ(dst);
    ಸ್ಥಿರಾಂಕ size_t s3 = nb3 / ggml_element_ಗಾತ್ರ(dst);

    ಸ್ಥಿರಾಂಕ size_t s10 = nb10 / ggml_element_ಗಾತ್ರ(src1);
    ಸ್ಥಿರಾಂಕ size_t s11 = nb11 / ggml_element_ಗಾತ್ರ(src1);
    ಸ್ಥಿರಾಂಕ size_t s12 = nb12 / ggml_element_ಗಾತ್ರ(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                             k_get_ಅಡ್ಡಸಾಲುಗಳು<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, ಅಂಶ_ct1);
                         });

    GGML_ಬಳಸದ(dst);
    GGML_ಬಳಸದ(ctx);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qr, dequantize_kernel_t_reorder dq>
ಸ್ಥಿರ ಶೂನ್ಯ get_ಅಡ್ಡಸಾಲುಗಳು_sycl_reorder(ಸ್ಥಿರಾಂಕ ggml_tensor *src0, ಸ್ಥಿರಾಂಕ ggml_tensor *src1,
                                  ggml_tensor *dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *src0_dd,
                                  ಸ್ಥಿರಾಂಕ int32_t *src1_dd, ದಶಮಾಂಶ *dst_dd,
                                  ಸರತಿ_ptr stream) {

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    // reorder is done only on 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % 2 == 0);

    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (ne00 + 2*SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ - 1) / (2*SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(ne11 * ne12, ne10, ಖಂಡ_ಅಂಕೆ_x);

    ಸ್ಥಿರಾಂಕ size_t s1 = nb1 / ggml_element_ಗಾತ್ರ(dst);
    ಸ್ಥಿರಾಂಕ size_t s2 = nb2 / ggml_element_ಗಾತ್ರ(dst);
    ಸ್ಥಿರಾಂಕ size_t s3 = nb3 / ggml_element_ಗಾತ್ರ(dst);

    ಸ್ಥಿರಾಂಕ size_t s10 = nb10 / ggml_element_ಗಾತ್ರ(src1);
    ಸ್ಥಿರಾಂಕ size_t s11 = nb11 / ggml_element_ಗಾತ್ರ(src1);
    ಸ್ಥಿರಾಂಕ size_t s12 = nb12 / ggml_element_ಗಾತ್ರ(src1);

    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * src0_d = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_dd + ne00*ne01/qr;

    stream->parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                             k_get_ಅಡ್ಡಸಾಲುಗಳು_reorder<qk, qr, dq>(
                                 src0_dd, src0_d, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, s10, s11, s12, ಅಂಶ_ct1);
                         });

    GGML_ಬಳಸದ(src0);
}

// K-quants in reorder layout: one work-group dequantizes one QK_K block of one row
ಮಾದರಿ <ಪೂರ್ಣಾಂಕ dims, ಪೂರ್ಣಾಂಕ nthreads, ಪೂರ್ಣಾಂಕ nlocal, ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಶೂನ್ಯ get_ಅಡ್ಡಸಾಲುಗಳು_sycl_reorder_k(ಸ್ಥಿರಾಂಕ ggml_tensor *src0, ಸ್ಥಿರಾಂಕ ggml_tensor *src1,
                                    ggml_tensor *dst, ಸ್ಥಿರಾಂಕ int32_t *src1_dd, ದಶಮಾಂಶ *dst_dd,
                                    ಸರತಿ_ptr stream, F dequantize_ಖಂಡ) {

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    // reorder is done only on 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % QK_K == 0);

    ಸ್ಥಿರಾಂಕ int64_t nblk = ne00 / QK_K;
    ಸ್ಥಿರಾಂಕ int64_t nb = nblk * ne01; // blocks in the whole tensor
    ಸ್ಥಿರಾಂಕ int64_t ngroups = ne10 * ne11 * ne12 * nblk;

    ಸ್ಥಿರಾಂಕ size_t s1 = nb1 / ggml_element_ಗಾತ್ರ(dst);
    ಸ್ಥಿರಾಂಕ size_t s2 = nb2 / ggml_element_ಗಾತ್ರ(dst);
    ಸ್ಥಿರಾಂಕ size_t s3 = nb3 / ggml_element_ಗಾತ್ರ(dst);

    ಸ್ಥಿರಾಂಕ size_t s10 = nb10 / ggml_element_ಗಾತ್ರ(src1);
    ಸ್ಥಿರಾಂಕ size_t s11 = nb11 / ggml_element_ಗಾತ್ರ(src1);
    ಸ್ಥಿರಾಂಕ size_t s12 = nb12 / ggml_element_ಗಾತ್ರ(src1);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scales_local(sycl::range<1>(nlocal), cgh);

        ಸ್ವಯಂಚಾಲಿತ kernel = [=](sycl::nd_ಅಂಶ<dims> ಅಂಶ_ct1) {
            ಸ್ಥಿರಾಂಕ int64_t g   = ಅಂಶ_ct1.get_ಗುಂಪು(dims - 1);
            ಸ್ಥಿರಾಂಕ int64_t b   = g % nblk;
            ಸ್ಥಿರಾಂಕ int64_t r   = g / nblk;
            ಸ್ಥಿರಾಂಕ int64_t i10 = r % ne10;
            ಸ್ಥಿರಾಂಕ int64_t i11 = (r / ne10) % ne11;
            ಸ್ಥಿರಾಂಕ int64_t i12 = r / (ne10 * ne11);

            ಸ್ಥಿರಾಂಕ int64_t i01 = src1_dd[i10*s10 + i11*s11 + i12*s12];

            ದಶಮಾಂಶ * y = dst_dd + i10*s1 + i11*s2 + i12*s3 + b*QK_K;
            dequantize_ಖಂಡ(y, i01*nblk + b, get_ಸೂಚಕ(scales_local), ಅಂಶ_ct1, nb);
        };

        ಆದರೆ constexpr (dims == 1) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(ngroups * nthreads), sycl::range<1>(nthreads)), kernel);
        } ಇಲ್ಲದಿದ್ದರೆ {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ngroups * nthreads), sycl::range<3>(1, 1, nthreads)), kernel);
        }
    });

    GGML_ಬಳಸದ(src0);
}

// IQ3_S / IQ3_XXS in the persistent reorder layout: one work-group of 32 threads
// dequantizes one QK_K block of one requested row, reusing the reorder-aware
// block dequantizers that the dequantize-based mul_mat path already uses.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು F>
ಸ್ಥಿರ ಶೂನ್ಯ get_ಅಡ್ಡಸಾಲುಗಳು_sycl_reorder_iq3(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0,
                                      ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst,
                                      ಸ್ಥಿರಾಂಕ int32_t * src1_dd, ದಶಮಾಂಶ * dst_dd,
                                      F dequantize_ಖಂಡ) {
    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    // the persistent reorder is only installed for 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % QK_K == 0);

    ಸ್ಥಿರಾಂಕ int64_t nblk    = ne00 / QK_K;
    ಸ್ಥಿರಾಂಕ int64_t nb      = nblk * ne01; // blocks in the whole tensor
    ಸ್ಥಿರಾಂಕ int64_t ngroups = ne10 * ne11 * ne12 * nblk;

    ಸ್ಥಿರಾಂಕ size_t s1  = nb1 / ggml_element_ಗಾತ್ರ(dst);
    ಸ್ಥಿರಾಂಕ size_t s2  = nb2 / ggml_element_ಗಾತ್ರ(dst);
    ಸ್ಥಿರಾಂಕ size_t s3  = nb3 / ggml_element_ಗಾತ್ರ(dst);

    ಸ್ಥಿರಾಂಕ size_t s10 = nb10 / ggml_element_ಗಾತ್ರ(src1);
    ಸ್ಥಿರಾಂಕ size_t s11 = nb11 / ggml_element_ಗಾತ್ರ(src1);
    ಸ್ಥಿರಾಂಕ size_t s12 = nb12 / ggml_element_ಗಾತ್ರ(src1);

    ಸರತಿ_ptr stream = ctx.stream();
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ngroups * 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            ಸ್ಥಿರಾಂಕ int64_t g   = ಅಂಶ_ct1.get_ಗುಂಪು(2);
            ಸ್ಥಿರಾಂಕ int64_t b   = g % nblk;
            ಸ್ಥಿರಾಂಕ int64_t r   = g / nblk;
            ಸ್ಥಿರಾಂಕ int64_t i10 = r % ne10;
            ಸ್ಥಿರಾಂಕ int64_t i11 = (r / ne10) % ne11;
            ಸ್ಥಿರಾಂಕ int64_t i12 = r / (ne10 * ne11);

            ಸ್ಥಿರಾಂಕ int64_t i01 = src1_dd[i10*s10 + i11*s11 + i12*s12];

            ದಶಮಾಂಶ * y = dst_dd + i10*s1 + i11*s2 + i12*s3 + b*QK_K;
            dequantize_ಖಂಡ(y, i01*nblk + b, ಅಂಶ_ct1, nb);
        });

    GGML_ಬಳಸದ(dst);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ qk, ಪೂರ್ಣಾಂಕ qr, dequantize_kernel_f32_t dq>
ಸ್ಥಿರ ಶೂನ್ಯ get_ಅಡ್ಡಸಾಲುಗಳು_sycl_f32(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor *src0, ಸ್ಥಿರಾಂಕ ggml_tensor *src1,
                              ggml_tensor *dst, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *src0_dd,
                              ಸ್ಥಿರಾಂಕ int32_t *src1_dd, ದಶಮಾಂಶ *dst_dd,
                              ಸರತಿ_ptr stream) {

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (ne00 + 2*SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ - 1) / (2*SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(ne11 * ne12, ne10, ಖಂಡ_ಅಂಕೆ_x);

    ಸ್ಥಿರಾಂಕ size_t s1 = nb1 / ggml_element_ಗಾತ್ರ(dst);
    ಸ್ಥಿರಾಂಕ size_t s2 = nb2 / ggml_element_ಗಾತ್ರ(dst);
    ಸ್ಥಿರಾಂಕ size_t s3 = nb3 / ggml_element_ಗಾತ್ರ(dst);

    ಸ್ಥಿರಾಂಕ size_t s10 = nb10 / ggml_element_ಗಾತ್ರ(src1);
    ಸ್ಥಿರಾಂಕ size_t s11 = nb11 / ggml_element_ಗಾತ್ರ(src1);
    ಸ್ಥಿರಾಂಕ size_t s12 = nb12 / ggml_element_ಗಾತ್ರ(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                             k_get_ಅಡ್ಡಸಾಲುಗಳು_f32<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, ಅಂಶ_ct1);
                         });

    GGML_ಬಳಸದ(dst);
    GGML_ಬಳಸದ(ctx);
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು src0_t, ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ get_ಅಡ್ಡಸಾಲುಗಳು_sycl_float(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor *src0,
                                ಸ್ಥಿರಾಂಕ ggml_tensor *src1, ggml_tensor *dst,
                                ಸ್ಥಿರಾಂಕ src0_t *src0_dd, ಸ್ಥಿರಾಂಕ int32_t *src1_dd,
                                dst_t *dst_dd, ಸರತಿ_ptr stream) {

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಖಂಡ_ಅಂಕೆ_x = (ne00 + SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(ne11 * ne12, ne10, ಖಂಡ_ಅಂಕೆ_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    ಸ್ಥಿರಾಂಕ size_t s1 = nb1 / ggml_element_ಗಾತ್ರ(dst);
    ಸ್ಥಿರಾಂಕ size_t s2 = nb2 / ggml_element_ಗಾತ್ರ(dst);
    ಸ್ಥಿರಾಂಕ size_t s3 = nb3 / ggml_element_ಗಾತ್ರ(dst);

    ಸ್ಥಿರಾಂಕ size_t s10 = nb10 / ggml_element_ಗಾತ್ರ(src1);
    ಸ್ಥಿರಾಂಕ size_t s11 = nb11 / ggml_element_ಗಾತ್ರ(src1);
    ಸ್ಥಿರಾಂಕ size_t s12 = nb12 / ggml_element_ಗಾತ್ರ(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
            [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                k_get_ಅಡ್ಡಸಾಲುಗಳು_float(src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, ಅಂಶ_ct1);
            });
    }

    GGML_ಬಳಸದ(dst);
    GGML_ಬಳಸದ(ctx);
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು src0_t>
ಸ್ಥಿರ ಶೂನ್ಯ k_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ_float(ಸ್ಥಿರಾಂಕ src0_t * src0, ಸ್ಥಿರಾಂಕ int32_t * src1, ದಶಮಾಂಶ * dst,
                                  ಸ್ಥಿರಾಂಕ int64_t ncols, ಸ್ಥಿರಾಂಕ int64_t nrows_grad_10, ಸ್ಥಿರಾಂಕ int64_t nrows_grad_11, ಸ್ಥಿರಾಂಕ int64_t nrows_dst,
                                  ಸ್ಥಿರಾಂಕ size_t s01, ಸ್ಥಿರಾಂಕ size_t s02,
                                  ಸ್ಥಿರಾಂಕ size_t s10, ಸ್ಥಿರಾಂಕ size_t s11,
                                  ಸ್ಥಿರಾಂಕ size_t s1,
                                  ಸ್ಥಿರಾಂಕ int64_t ಖಂಡ_ಅಂಕೆ_y,
                                  ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ int64_t ಲಂಬಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2) + ಅಂಶ_ct1.get_local_id(2);
    ಆದರೆ (ಲಂಬಸಾಲು >= ncols) {
        ಹಿಂತಿರುಗಿ;
    }

    // block_num_y is clamped, so stride over destination rows like CUDA k_get_rows_back_float
    ಗಾಗಿ (int64_t dst_ಅಡ್ಡಸಾಲು = ಅಂಶ_ct1.get_ಗುಂಪು(1); dst_ಅಡ್ಡಸಾಲು < nrows_dst; dst_ಅಡ್ಡಸಾಲು += ಖಂಡ_ಅಂಕೆ_y) {
        ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;

        ಸ್ಥಿರಾಂಕ int64_t nrows_grad_ಒಟ್ಟು = nrows_grad_10 * nrows_grad_11;
        ಗಾಗಿ (int64_t i = 0; i < nrows_grad_ಒಟ್ಟು; ++i) {
            ಸ್ಥಿರಾಂಕ int64_t i10 = i % nrows_grad_10;
            ಸ್ಥಿರಾಂಕ int64_t i11 = i / nrows_grad_10;
            ಆದರೆ (src1[i10*s10 + i11*s11] != dst_ಅಡ್ಡಸಾಲು) {
                ಮುಂದುವರಿ;
            }
            ಮೊತ್ತ += (ದಶಮಾಂಶ) src0[ಲಂಬಸಾಲು + i10*s01 + i11*s02];
        }

        dst[ಲಂಬಸಾಲು + dst_ಅಡ್ಡಸಾಲು*s1] = ಮೊತ್ತ;
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು src0_t>
ಸ್ಥಿರ ಶೂನ್ಯ get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ_sycl_float(ggml_backend_sycl_context & ctx, ಸ್ಥಿರಾಂಕ ggml_tensor * src0,
                                     ಸ್ಥಿರಾಂಕ ggml_tensor * src1, ggml_tensor * dst,
                                     ಸ್ಥಿರಾಂಕ src0_t * src0_dd, ಸ್ಥಿರಾಂಕ int32_t * src1_dd,
                                     ದಶಮಾಂಶ * dst_dd, ಸರತಿ_ptr stream) {

    GGML_TENSOR_ದ್ವಿಮಾನ_OP_LOCALS

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);
    GGML_ASSERT(src0->nb[0] == ggml_type_ಗಾತ್ರ(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_ಗಾತ್ರ(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_ಗಾತ್ರ(dst->type));

    ಸ್ಥಿರಾಂಕ int64_t ncols = ne00;
    ಸ್ಥಿರಾಂಕ int64_t nrows_grad_10 = ne10;
    ಸ್ಥಿರಾಂಕ int64_t nrows_grad_11 = ne11;
    ಸ್ಥಿರಾಂಕ int64_t nrows_dst = ne1;

    ಸ್ಥಿರಾಂಕ size_t s01 = nb01 / ಗಾತ್ರ(src0_t);
    ಸ್ಥಿರಾಂಕ size_t s02 = nb02 / ಗಾತ್ರ(src0_t);

    ಸ್ಥಿರಾಂಕ size_t s10 = nb10 / ಗಾತ್ರ(int32_t);
    ಸ್ಥಿರಾಂಕ size_t s11 = nb11 / ಗಾತ್ರ(int32_t);

    ಸ್ಥಿರಾಂಕ size_t s1 = nb1 / ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ int64_t ಖಂಡ_ಅಂಕೆ_x = (ncols + SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_GET_ಅಡ್ಡಸಾಲುಗಳು_ಖಂಡ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ int64_t ಖಂಡ_ಅಂಕೆ_y = std::min<int64_t>(nrows_dst, (int64_t) UINT16_MAX);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, ಖಂಡ_ಅಂಕೆ_y, ಖಂಡ_ಅಂಕೆ_x);

    stream->parallel_for(sycl::nd_range<3>(ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
                         [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
                             k_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ_float(src0_dd, src1_dd, dst_dd,
                                                   ncols, nrows_grad_10, nrows_grad_11, nrows_dst,
                                                   s01, s02, s10, s11, s1,
                                                   ಖಂಡ_ಅಂಕೆ_y, ಅಂಶ_ct1);
                         });

    GGML_ಬಳಸದ(src0);
    GGML_ಬಳಸದ(src1);
    GGML_ಬಳಸದ(dst);
    GGML_ಬಳಸದ(ctx);
}

ಶೂನ್ಯ ggml_sycl_op_get_ಅಡ್ಡಸಾಲುಗಳು(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_I32 );

    GGML_ASSERT(dst->src[0]->nb[0] == ggml_type_ಗಾತ್ರ(dst->src[0]->type));
    GGML_ASSERT(dst->src[1]->nb[0] == ggml_type_ಗಾತ್ರ(dst->src[1]->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_ಗಾತ್ರ(dst->type));

    ಸ್ಥಿರಾಂಕ int32_t * src1_i32 = (ಸ್ಥಿರಾಂಕ int32_t *) dst->src[1]->ದತ್ತಾಂಶ;
    // a weight shared with MUL_MAT (e.g. tied embeddings) can be in the
    // persistent reorder layout (incl. IQ3_S / IQ3_XXS since the SYCL IQ3
    // reorder work)
    ಸ್ಥಿರಾಂಕ ತಾರ್ಕಿಕ src0_reordered = dst->src[0]->extra &&
        ((ಸ್ಥಿರಾಂಕ ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
    /* TODO: Refactor and remove duplicates */
    ಆಯ್ಕೆ (dst->src[0]->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ sycl::half *)dst->src[0]->ದತ್ತಾಂಶ,
                                src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_BF16:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ sycl::ext::oneapi::bfloat16 *)dst->src[0]->ದತ್ತಾಂಶ,
                                src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_I32:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ int32_t *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (int32_t *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q1_0:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK1_0, 1, dequantize_q1_0>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_MXFP4:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK_MXFP4, 2, dequantize_mxfp4>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_NVFP4:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK_NVFP4, 1, dequantize_nvfp4>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XXS:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK_K, 1, dequantize_iq2_xxs>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_XS:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK_K, 1, dequantize_iq2_xs>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ2_S:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK_K, 1, dequantize_iq2_s>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_XXS:
            ಆದರೆ (src0_reordered) {
                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx = dst->src[0]->ದತ್ತಾಂಶ;
                get_ಅಡ್ಡಸಾಲುಗಳು_sycl_reorder_iq3(ctx, dst->src[0], dst->src[1], dst, src1_i32, (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ,
                    [=](ದಶಮಾಂಶ * y, int64_t iblk, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1, int64_t nb) {
                        dequantize_ಖಂಡ_iq3_xxs_reorder(vx, y, iblk, ಅಂಶ_ct1, nb,
                                                         iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs);
                    });
                ನಿಲ್ಲಿಸು;
            }
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK_K, 1, dequantize_iq3_xxs>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_S:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK_K, 1, dequantize_iq1_s>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ1_M:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK_K, 1, dequantize_iq1_m>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ3_S:
            ಆದರೆ (src0_reordered) {
                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx = dst->src[0]->ದತ್ತಾಂಶ;
                get_ಅಡ್ಡಸಾಲುಗಳು_sycl_reorder_iq3(ctx, dst->src[0], dst->src[1], dst, src1_i32, (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ,
                    [=](ದಶಮಾಂಶ * y, int64_t iblk, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1, int64_t nb) {
                        dequantize_ಖಂಡ_iq3_s_reorder(vx, y, iblk, ಅಂಶ_ct1, nb,
                                                       kmask_iq2xs, iq3s_grid);
                    });
                ನಿಲ್ಲಿಸು;
            }
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK_K, 1, dequantize_iq3_s>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_NL:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK4_NL, 1, dequantize_iq4_nl>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_IQ4_XS:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK_K, 1, dequantize_iq4_xs>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q2_K:
            ಆದರೆ (src0_reordered) {
                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx = dst->src[0]->ದತ್ತಾಂಶ;
                get_ಅಡ್ಡಸಾಲುಗಳು_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream(),
                    [=](ದಶಮಾಂಶ * y, int64_t ib, uint8_t * scales_local, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1, int64_t nb) {
                        dequantize_ಖಂಡ_q2_K_reorder(vx, y, ib, ಅಂಶ_ct1, nb);
                        GGML_ಬಳಸದ(scales_local);
                    });
                ನಿಲ್ಲಿಸು;
            }
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl_f32<QK_K, 1, dequantize_q2_K_f32>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q3_K:
            ಆದರೆ (src0_reordered) {
                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx = dst->src[0]->ದತ್ತಾಂಶ;
                get_ಅಡ್ಡಸಾಲುಗಳು_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream(),
                    [=](ದಶಮಾಂಶ * y, int64_t ib, uint8_t * scales_local, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1, int64_t nb) {
                        dequantize_ಖಂಡ_q3_K_reorder(vx, y, ib, ಅಂಶ_ct1, nb);
                        GGML_ಬಳಸದ(scales_local);
                    });
                ನಿಲ್ಲಿಸು;
            }
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK_K, 1, dequantize_q3_K>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ಆದರೆ (src0_reordered) {
                get_ಅಡ್ಡಸಾಲುಗಳು_sycl_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(dst->src[0], dst->src[1], dst, dst->src[0]->ದತ್ತಾಂಶ,
                src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
                ನಿಲ್ಲಿಸು;
            }
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_1:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            ಆದರೆ (src0_reordered) {
                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx = dst->src[0]->ದತ್ತಾಂಶ;
                get_ಅಡ್ಡಸಾಲುಗಳು_sycl_reorder_k<1, 32, K_SCALE_ಗಾತ್ರ>(dst->src[0], dst->src[1], dst, src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream(),
                    [=](ದಶಮಾಂಶ * y, int64_t ib, uint8_t * scales_local, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ_ct1, int64_t nb) {
                        dequantize_ಖಂಡ_q4_K_reorder(vx, y, ib, scales_local, ಅಂಶ_ct1, nb);
                    });
                ನಿಲ್ಲಿಸು;
            }
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl_f32<QK_K, 1, dequantize_q4_K_f32>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_0:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_1:
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q5_K:
            ಆದರೆ (src0_reordered) {
                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx = dst->src[0]->ದತ್ತಾಂಶ;
                get_ಅಡ್ಡಸಾಲುಗಳು_sycl_reorder_k<3, 64, K_SCALE_ಗಾತ್ರ>(dst->src[0], dst->src[1], dst, src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream(),
                    [=](ದಶಮಾಂಶ * y, int64_t ib, uint8_t * scales_local, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1, int64_t nb) {
                        dequantize_ಖಂಡ_q5_K_reorder(vx, y, ib, scales_local, ಅಂಶ_ct1, nb);
                    });
                ನಿಲ್ಲಿಸು;
            }
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl_f32<QK_K, 1, dequantize_q5_K_f32>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q6_K:
            ಆದರೆ (src0_reordered) {
                ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx = dst->src[0]->ದತ್ತಾಂಶ;
                get_ಅಡ್ಡಸಾಲುಗಳು_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream(),
                    [=](ದಶಮಾಂಶ * y, int64_t ib, uint8_t * scales_local, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1, int64_t nb) {
                        dequantize_ಖಂಡ_q6_K_reorder(vx, y, ib, ಅಂಶ_ct1, nb);
                        GGML_ಬಳಸದ(scales_local);
                    });
                ನಿಲ್ಲಿಸು;
            }
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK_K, 1, dequantize_q6_K>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಆದರೆ (src0_reordered) {
                get_ಅಡ್ಡಸಾಲುಗಳು_sycl_reorder<QK8_0, QR8_0, dequantize_q8_0_reorder>(dst->src[0], dst->src[1], dst, dst->src[0]->ದತ್ತಾಂಶ,
                src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
                ನಿಲ್ಲಿಸು;
            }
            get_ಅಡ್ಡಸಾಲುಗಳು_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, dst->src[0], dst->src[1], dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)dst->src[0]->ದತ್ತಾಂಶ,
            src1_i32, (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ, ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            // TODO: k-quants
            GGML_LOG_ERROR("%s: unsupported type: %s\n", __func__, ggml_type_name(dst->src[0]->type));
            GGML_ABORT("fatal error");
    }
}

ಶೂನ್ಯ ggml_sycl_op_get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(dst));

    ಆಯ್ಕೆ (src0->type) {
        ಸ್ಥಿತಿ GGML_TYPE_F16:
            get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ_sycl_float(ctx, src0, src1, dst, (ಸ್ಥಿರಾಂಕ sycl::half *) src0->ದತ್ತಾಂಶ,
                                     (ಸ್ಥಿರಾಂಕ int32_t *) src1->ದತ್ತಾಂಶ, (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ,
                                     ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಸ್ಥಿತಿ GGML_TYPE_F32:
            get_ಅಡ್ಡಸಾಲುಗಳು_ಹಿಂಭಾಗ_sycl_float(ctx, src0, src1, dst, (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ,
                                     (ಸ್ಥಿರಾಂಕ int32_t *) src1->ದತ್ತಾಂಶ, (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ,
                                     ctx.stream());
            ನಿಲ್ಲಿಸು;
        ಪೂರ್ವನಿಯೋಜಿತ:
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0->type));
    }
}
