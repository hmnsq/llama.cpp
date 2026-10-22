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

#ضمن "ggml-impl.h"
#ضمن "common.hpp"
#ضمن "dequantize.hpp"
#ضمن "getrows.hpp"


کينډۍ<صحيح qk, صحيح qr, dequantize_kernel_t dequantize_kernel, ډول_نوم dst_t>
ساكن فراغ k_get_لیکې(
            ثابت فراغ * src0, ثابت int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            ثابت sycl::nd_توکی<3> &توکی_ct1/*, size_t s13*/) {

    ثابت صحيح i00 = (توکی_ct1.get_ډله(2) * توکی_ct1.get_local_range(2) +
                     توکی_ct1.get_local_id(2)) *
                    2;
    ثابت صحيح i10 = توکی_ct1.get_local_range(1) * توکی_ct1.get_ډله(1) +
                    توکی_ct1.get_local_id(1);
    ثابت صحيح i11 = (توکی_ct1.get_ډله(0) * توکی_ct1.get_local_range(0) +
                     توکی_ct1.get_local_id(0)) /
                    ne12;
    ثابت صحيح i12 = (توکی_ct1.get_ډله(0) * توکی_ct1.get_local_range(0) +
                     توکی_ct1.get_local_id(0)) %
                    ne12;

    إذا (i00 >= ne00) {
        أرجع;
    }

    ثابت صحيح i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_لیکه = dst + i10*s1 + i11*s2 + i12*s3;
    ثابت فراغ * src0_لیکه = (ثابت حرف *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    ثابت صحيح ib = i00/qk; // block index
    ثابت صحيح iqs = (i00%qk)/qr; // quant index
    ثابت صحيح iybs = i00 - i00%qk; // dst block start index
    ثابت صحيح y_دننه_واټن = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(src0_لیکه, ib, iqs, v);

    dst_لیکه[iybs + iqs + 0] = v.x();
    dst_لیکه[iybs + iqs + y_دننه_واټن] = v.y();
}

کينډۍ<صحيح qk, صحيح qr, dequantize_kernel_f32_t dequantize_kernel, ډول_نوم dst_t>
ساكن فراغ k_get_لیکې_f32(
            ثابت فراغ * src0, ثابت int32_t * src1, dst_t * dst,
            int64_t ne00,
            int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            ثابت sycl::nd_توکی<3> &توکی_ct1) {

    ثابت صحيح i00 = (توکی_ct1.get_ډله(2) * توکی_ct1.get_local_range(2) +
                     توکی_ct1.get_local_id(2)) *
                    2;
    ثابت صحيح i10 = توکی_ct1.get_local_range(1) * توکی_ct1.get_ډله(1) +
                    توکی_ct1.get_local_id(1);
    ثابت صحيح i11 = (توکی_ct1.get_ډله(0) * توکی_ct1.get_local_range(0) +
                     توکی_ct1.get_local_id(0)) /
                    ne12;
    ثابت صحيح i12 = (توکی_ct1.get_ډله(0) * توکی_ct1.get_local_range(0) +
                     توکی_ct1.get_local_id(0)) %
                    ne12;

    إذا (i00 >= ne00) {
        أرجع;
    }

    ثابت صحيح i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_لیکه = dst + i10*s1 + i11*s2 + i12*s3;
    ثابت فراغ * src0_لیکه = (ثابت حرف *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    ثابت صحيح ib = i00/qk;
    ثابت صحيح iqs = (i00%qk)/qr;
    ثابت صحيح iybs = i00 - i00%qk;
    ثابت صحيح y_دننه_واټن = qr == 1 ? 1 : qk/2;

    عائم v0;
    عائم v1;
    dequantize_kernel(src0_لیکه, ib, iqs, v0, v1);

    dst_لیکه[iybs + iqs + 0] = (dst_t) v0;
    dst_لیکه[iybs + iqs + y_دننه_واټن] = (dst_t) v1;
}

کينډۍ<ډول_نوم src0_t, ډول_نوم dst_t>
ساكن فراغ k_get_لیکې_float(
            ثابت src0_t * src0, ثابت int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            ثابت sycl::nd_توکی<3> &توکی_ct1/*, size_t s13*/) {

    ثابت صحيح i00 = توکی_ct1.get_ډله(2) * توکی_ct1.get_local_range(2) +
                    توکی_ct1.get_local_id(2);
    ثابت صحيح i10 = توکی_ct1.get_local_range(1) * توکی_ct1.get_ډله(1) +
                    توکی_ct1.get_local_id(1);
    ثابت صحيح i11 = (توکی_ct1.get_ډله(0) * توکی_ct1.get_local_range(0) +
                     توکی_ct1.get_local_id(0)) /
                    ne12;
    ثابت صحيح i12 = (توکی_ct1.get_ډله(0) * توکی_ct1.get_local_range(0) +
                     توکی_ct1.get_local_id(0)) %
                    ne12;

    إذا (i00 >= ne00) {
        أرجع;
    }

    ثابت صحيح i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_لیکه = dst + i10*s1 + i11*s2 + i12*s3;
    ثابت src0_t * src0_لیکه = (ثابت src0_t *)((ثابت حرف *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_لیکه[i00] = src0_لیکه[i00];
}

// src0 is in reorder layout: all qs of the tensor, then all d
کينډۍ<صحيح qk, صحيح qr, dequantize_kernel_t_reorder dequantize_kernel, ډول_نوم dst_t>
ساكن فراغ k_get_لیکې_reorder(
            ثابت فراغ * src0_qs, ثابت فراغ * src0_d, ثابت int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t s10, size_t s11, size_t s12,
            ثابت sycl::nd_توکی<3> &توکی_ct1) {

    ثابت صحيح i00 = (توکی_ct1.get_ډله(2) * توکی_ct1.get_local_range(2) +
                     توکی_ct1.get_local_id(2)) *
                    2;
    ثابت صحيح i10 = توکی_ct1.get_local_range(1) * توکی_ct1.get_ډله(1) +
                    توکی_ct1.get_local_id(1);
    ثابت صحيح i11 = (توکی_ct1.get_ډله(0) * توکی_ct1.get_local_range(0) +
                     توکی_ct1.get_local_id(0)) /
                    ne12;
    ثابت صحيح i12 = (توکی_ct1.get_ډله(0) * توکی_ct1.get_local_range(0) +
                     توکی_ct1.get_local_id(0)) %
                    ne12;

    إذا (i00 >= ne00) {
        أرجع;
    }

    ثابت صحيح i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_لیکه = dst + i10*s1 + i11*s2 + i12*s3;

    ثابت int64_t ib = (int64_t) i01*(ne00/qk) + i00/qk; // block index in the whole tensor
    ثابت صحيح iqs = (i00%qk)/qr;
    ثابت صحيح iybs = i00 - i00%qk;
    ثابت صحيح y_دننه_واټن = qr == 1 ? 1 : qk/2;

    dfloat2 v;
    dequantize_kernel(src0_d, ib, src0_qs, ib*(qk/qr) + iqs, v);

    dst_لیکه[iybs + iqs + 0] = v.x();
    dst_لیکه[iybs + iqs + y_دننه_واټن] = v.y();
}

کينډۍ <صحيح qk, صحيح qr, dequantize_kernel_t dq>
ساكن فراغ get_لیکې_sycl(ggml_backend_sycl_context & ctx, ثابت ggml_tensor *src0, ثابت ggml_tensor *src1,
                          ggml_tensor *dst, ثابت فراغ *src0_dd,
                          ثابت int32_t *src1_dd, عائم *dst_dd,
                          کتار_ptr stream) {

    GGML_TENSOR_دوګونی_OP_LOCALS

    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_GET_لیکې_بلاک_اندازه);
    ثابت صحيح بلاک_عدد_x = (ne00 + 2*SYCL_GET_لیکې_بلاک_اندازه - 1) / (2*SYCL_GET_لیکې_بلاک_اندازه);
    ثابت sycl::range<3> بلاک_nums(ne11 * ne12, ne10, بلاک_عدد_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    ثابت size_t s1 = nb1 / ggml_element_اندازه(dst);
    ثابت size_t s2 = nb2 / ggml_element_اندازه(dst);
    ثابت size_t s3 = nb3 / ggml_element_اندازه(dst);

    ثابت size_t s10 = nb10 / ggml_element_اندازه(src1);
    ثابت size_t s11 = nb11 / ggml_element_اندازه(src1);
    ثابت size_t s12 = nb12 / ggml_element_اندازه(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_توکی<3> توکی_ct1) {
                             k_get_لیکې<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, توکی_ct1);
                         });

    GGML_ناکارول_شوی(dst);
    GGML_ناکارول_شوی(ctx);
}

کينډۍ <صحيح qk, صحيح qr, dequantize_kernel_t_reorder dq>
ساكن فراغ get_لیکې_sycl_reorder(ثابت ggml_tensor *src0, ثابت ggml_tensor *src1,
                                  ggml_tensor *dst, ثابت فراغ *src0_dd,
                                  ثابت int32_t *src1_dd, عائم *dst_dd,
                                  کتار_ptr stream) {

    GGML_TENSOR_دوګونی_OP_LOCALS

    // reorder is done only on 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % 2 == 0);

    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_GET_لیکې_بلاک_اندازه);
    ثابت صحيح بلاک_عدد_x = (ne00 + 2*SYCL_GET_لیکې_بلاک_اندازه - 1) / (2*SYCL_GET_لیکې_بلاک_اندازه);
    ثابت sycl::range<3> بلاک_nums(ne11 * ne12, ne10, بلاک_عدد_x);

    ثابت size_t s1 = nb1 / ggml_element_اندازه(dst);
    ثابت size_t s2 = nb2 / ggml_element_اندازه(dst);
    ثابت size_t s3 = nb3 / ggml_element_اندازه(dst);

    ثابت size_t s10 = nb10 / ggml_element_اندازه(src1);
    ثابت size_t s11 = nb11 / ggml_element_اندازه(src1);
    ثابت size_t s12 = nb12 / ggml_element_اندازه(src1);

    ثابت فراغ * src0_d = (ثابت حرف *) src0_dd + ne00*ne01/qr;

    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_توکی<3> توکی_ct1) {
                             k_get_لیکې_reorder<qk, qr, dq>(
                                 src0_dd, src0_d, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, s10, s11, s12, توکی_ct1);
                         });

    GGML_ناکارول_شوی(src0);
}

// K-quants in reorder layout: one work-group dequantizes one QK_K block of one row
کينډۍ <صحيح dims, صحيح nthreads, صحيح nlocal, ډول_نوم F>
ساكن فراغ get_لیکې_sycl_reorder_k(ثابت ggml_tensor *src0, ثابت ggml_tensor *src1,
                                    ggml_tensor *dst, ثابت int32_t *src1_dd, عائم *dst_dd,
                                    کتار_ptr stream, F dequantize_بلاک) {

    GGML_TENSOR_دوګونی_OP_LOCALS

    // reorder is done only on 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % QK_K == 0);

    ثابت int64_t nblk = ne00 / QK_K;
    ثابت int64_t nb = nblk * ne01; // blocks in the whole tensor
    ثابت int64_t ngroups = ne10 * ne11 * ne12 * nblk;

    ثابت size_t s1 = nb1 / ggml_element_اندازه(dst);
    ثابت size_t s2 = nb2 / ggml_element_اندازه(dst);
    ثابت size_t s3 = nb3 / ggml_element_اندازه(dst);

    ثابت size_t s10 = nb10 / ggml_element_اندازه(src1);
    ثابت size_t s11 = nb11 / ggml_element_اندازه(src1);
    ثابت size_t s12 = nb12 / ggml_element_اندازه(src1);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scales_local(sycl::range<1>(nlocal), cgh);

        خپلکاری kernel = [=](sycl::nd_توکی<dims> توکی_ct1) {
            ثابت int64_t g   = توکی_ct1.get_ډله(dims - 1);
            ثابت int64_t b   = g % nblk;
            ثابت int64_t r   = g / nblk;
            ثابت int64_t i10 = r % ne10;
            ثابت int64_t i11 = (r / ne10) % ne11;
            ثابت int64_t i12 = r / (ne10 * ne11);

            ثابت int64_t i01 = src1_dd[i10*s10 + i11*s11 + i12*s12];

            عائم * y = dst_dd + i10*s1 + i11*s2 + i12*s3 + b*QK_K;
            dequantize_بلاک(y, i01*nblk + b, get_اشاره_ګر(scales_local), توکی_ct1, nb);
        };

        إذا constexpr (dims == 1) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(ngroups * nthreads), sycl::range<1>(nthreads)), kernel);
        } وإلا {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ngroups * nthreads), sycl::range<3>(1, 1, nthreads)), kernel);
        }
    });

    GGML_ناکارول_شوی(src0);
}

// IQ3_S / IQ3_XXS in the persistent reorder layout: one work-group of 32 threads
// dequantizes one QK_K block of one requested row, reusing the reorder-aware
// block dequantizers that the dequantize-based mul_mat path already uses.
کينډۍ <ډول_نوم F>
ساكن فراغ get_لیکې_sycl_reorder_iq3(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0,
                                      ثابت ggml_tensor * src1, ggml_tensor * dst,
                                      ثابت int32_t * src1_dd, عائم * dst_dd,
                                      F dequantize_بلاک) {
    GGML_TENSOR_دوګونی_OP_LOCALS

    // the persistent reorder is only installed for 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % QK_K == 0);

    ثابت int64_t nblk    = ne00 / QK_K;
    ثابت int64_t nb      = nblk * ne01; // blocks in the whole tensor
    ثابت int64_t ngroups = ne10 * ne11 * ne12 * nblk;

    ثابت size_t s1  = nb1 / ggml_element_اندازه(dst);
    ثابت size_t s2  = nb2 / ggml_element_اندازه(dst);
    ثابت size_t s3  = nb3 / ggml_element_اندازه(dst);

    ثابت size_t s10 = nb10 / ggml_element_اندازه(src1);
    ثابت size_t s11 = nb11 / ggml_element_اندازه(src1);
    ثابت size_t s12 = nb12 / ggml_element_اندازه(src1);

    کتار_ptr stream = ctx.stream();
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ngroups * 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_توکی<3> توکی_ct1) {
            ثابت int64_t g   = توکی_ct1.get_ډله(2);
            ثابت int64_t b   = g % nblk;
            ثابت int64_t r   = g / nblk;
            ثابت int64_t i10 = r % ne10;
            ثابت int64_t i11 = (r / ne10) % ne11;
            ثابت int64_t i12 = r / (ne10 * ne11);

            ثابت int64_t i01 = src1_dd[i10*s10 + i11*s11 + i12*s12];

            عائم * y = dst_dd + i10*s1 + i11*s2 + i12*s3 + b*QK_K;
            dequantize_بلاک(y, i01*nblk + b, توکی_ct1, nb);
        });

    GGML_ناکارول_شوی(dst);
}

کينډۍ <صحيح qk, صحيح qr, dequantize_kernel_f32_t dq>
ساكن فراغ get_لیکې_sycl_f32(ggml_backend_sycl_context & ctx, ثابت ggml_tensor *src0, ثابت ggml_tensor *src1,
                              ggml_tensor *dst, ثابت فراغ *src0_dd,
                              ثابت int32_t *src1_dd, عائم *dst_dd,
                              کتار_ptr stream) {

    GGML_TENSOR_دوګونی_OP_LOCALS

    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_GET_لیکې_بلاک_اندازه);
    ثابت صحيح بلاک_عدد_x = (ne00 + 2*SYCL_GET_لیکې_بلاک_اندازه - 1) / (2*SYCL_GET_لیکې_بلاک_اندازه);
    ثابت sycl::range<3> بلاک_nums(ne11 * ne12, ne10, بلاک_عدد_x);

    ثابت size_t s1 = nb1 / ggml_element_اندازه(dst);
    ثابت size_t s2 = nb2 / ggml_element_اندازه(dst);
    ثابت size_t s3 = nb3 / ggml_element_اندازه(dst);

    ثابت size_t s10 = nb10 / ggml_element_اندازه(src1);
    ثابت size_t s11 = nb11 / ggml_element_اندازه(src1);
    ثابت size_t s12 = nb12 / ggml_element_اندازه(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_توکی<3> توکی_ct1) {
                             k_get_لیکې_f32<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, توکی_ct1);
                         });

    GGML_ناکارول_شوی(dst);
    GGML_ناکارول_شوی(ctx);
}

کينډۍ <ډول_نوم src0_t, ډول_نوم dst_t>
ساكن فراغ get_لیکې_sycl_float(ggml_backend_sycl_context & ctx, ثابت ggml_tensor *src0,
                                ثابت ggml_tensor *src1, ggml_tensor *dst,
                                ثابت src0_t *src0_dd, ثابت int32_t *src1_dd,
                                dst_t *dst_dd, کتار_ptr stream) {

    GGML_TENSOR_دوګونی_OP_LOCALS

    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_GET_لیکې_بلاک_اندازه);
    ثابت صحيح بلاک_عدد_x = (ne00 + SYCL_GET_لیکې_بلاک_اندازه - 1) / SYCL_GET_لیکې_بلاک_اندازه;
    ثابت sycl::range<3> بلاک_nums(ne11 * ne12, ne10, بلاک_عدد_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    ثابت size_t s1 = nb1 / ggml_element_اندازه(dst);
    ثابت size_t s2 = nb2 / ggml_element_اندازه(dst);
    ثابت size_t s3 = nb3 / ggml_element_اندازه(dst);

    ثابت size_t s10 = nb10 / ggml_element_اندازه(src1);
    ثابت size_t s11 = nb11 / ggml_element_اندازه(src1);
    ثابت size_t s12 = nb12 / ggml_element_اندازه(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_توکی<3> توکی_ct1) {
                k_get_لیکې_float(src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, توکی_ct1);
            });
    }

    GGML_ناکارول_شوی(dst);
    GGML_ناکارول_شوی(ctx);
}

کينډۍ <ډول_نوم src0_t>
ساكن فراغ k_get_لیکې_شا_float(ثابت src0_t * src0, ثابت int32_t * src1, عائم * dst,
                                  ثابت int64_t ncols, ثابت int64_t nrows_grad_10, ثابت int64_t nrows_grad_11, ثابت int64_t nrows_dst,
                                  ثابت size_t s01, ثابت size_t s02,
                                  ثابت size_t s10, ثابت size_t s11,
                                  ثابت size_t s1,
                                  ثابت int64_t بلاک_عدد_y,
                                  ثابت sycl::nd_توکی<3> & توکی_ct1) {
    ثابت int64_t ستون = توکی_ct1.get_ډله(2) * توکی_ct1.get_local_range(2) + توکی_ct1.get_local_id(2);
    إذا (ستون >= ncols) {
        أرجع;
    }

    // block_num_y is clamped, so stride over destination rows like CUDA k_get_rows_back_float
    لكل (int64_t dst_لیکه = توکی_ct1.get_ډله(1); dst_لیکه < nrows_dst; dst_لیکه += بلاک_عدد_y) {
        عائم مجموعه = 0.0f;

        ثابت int64_t nrows_grad_ټول = nrows_grad_10 * nrows_grad_11;
        لكل (int64_t i = 0; i < nrows_grad_ټول; ++i) {
            ثابت int64_t i10 = i % nrows_grad_10;
            ثابت int64_t i11 = i / nrows_grad_10;
            إذا (src1[i10*s10 + i11*s11] != dst_لیکه) {
                استمر;
            }
            مجموعه += (عائم) src0[ستون + i10*s01 + i11*s02];
        }

        dst[ستون + dst_لیکه*s1] = مجموعه;
    }
}

کينډۍ <ډول_نوم src0_t>
ساكن فراغ get_لیکې_شا_sycl_float(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0,
                                     ثابت ggml_tensor * src1, ggml_tensor * dst,
                                     ثابت src0_t * src0_dd, ثابت int32_t * src1_dd,
                                     عائم * dst_dd, کتار_ptr stream) {

    GGML_TENSOR_دوګونی_OP_LOCALS

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);
    GGML_ASSERT(src0->nb[0] == ggml_type_اندازه(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_اندازه(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_اندازه(dst->type));

    ثابت int64_t ncols = ne00;
    ثابت int64_t nrows_grad_10 = ne10;
    ثابت int64_t nrows_grad_11 = ne11;
    ثابت int64_t nrows_dst = ne1;

    ثابت size_t s01 = nb01 / حجم(src0_t);
    ثابت size_t s02 = nb02 / حجم(src0_t);

    ثابت size_t s10 = nb10 / حجم(int32_t);
    ثابت size_t s11 = nb11 / حجم(int32_t);

    ثابت size_t s1 = nb1 / حجم(عائم);

    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_GET_لیکې_بلاک_اندازه);
    ثابت int64_t بلاک_عدد_x = (ncols + SYCL_GET_لیکې_بلاک_اندازه - 1) / SYCL_GET_لیکې_بلاک_اندازه;
    ثابت int64_t بلاک_عدد_y = std::min<int64_t>(nrows_dst, (int64_t) UINT16_MAX);
    ثابت sycl::range<3> بلاک_nums(1, بلاک_عدد_y, بلاک_عدد_x);

    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_توکی<3> توکی_ct1) {
                             k_get_لیکې_شا_float(src0_dd, src1_dd, dst_dd,
                                                   ncols, nrows_grad_10, nrows_grad_11, nrows_dst,
                                                   s01, s02, s10, s11, s1,
                                                   بلاک_عدد_y, توکی_ct1);
                         });

    GGML_ناکارول_شوی(src0);
    GGML_ناکارول_شوی(src1);
    GGML_ناکارول_شوی(dst);
    GGML_ناکارول_شوی(ctx);
}

فراغ ggml_sycl_op_get_لیکې(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_I32 );

    GGML_ASSERT(dst->src[0]->nb[0] == ggml_type_اندازه(dst->src[0]->type));
    GGML_ASSERT(dst->src[1]->nb[0] == ggml_type_اندازه(dst->src[1]->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_اندازه(dst->type));

    ثابت int32_t * src1_i32 = (ثابت int32_t *) dst->src[1]->معلومات;
    // a weight shared with MUL_MAT (e.g. tied embeddings) can be in the
    // persistent reorder layout (incl. IQ3_S / IQ3_XXS since the SYCL IQ3
    // reorder work)
    ثابت منطقي src0_reordered = dst->src[0]->extra &&
        ((ثابت ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
    /* TODO: Refactor and remove duplicates */
    بدل (dst->src[0]->type) {
        حالة GGML_TYPE_F16:
            get_لیکې_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ثابت sycl::half *)dst->src[0]->معلومات,
                                src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_BF16:
            get_لیکې_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ثابت sycl::ext::oneapi::bfloat16 *)dst->src[0]->معلومات,
                                src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_F32:
            get_لیکې_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_I32:
            get_لیکې_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ثابت int32_t *)dst->src[0]->معلومات,
            src1_i32, (int32_t *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_Q1_0:
            get_لیکې_sycl<QK1_0, 1, dequantize_q1_0>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_MXFP4:
            get_لیکې_sycl<QK_MXFP4, 2, dequantize_mxfp4>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_NVFP4:
            get_لیکې_sycl<QK_NVFP4, 1, dequantize_nvfp4>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_IQ2_XXS:
            get_لیکې_sycl<QK_K, 1, dequantize_iq2_xxs>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_IQ2_XS:
            get_لیکې_sycl<QK_K, 1, dequantize_iq2_xs>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_IQ2_S:
            get_لیکې_sycl<QK_K, 1, dequantize_iq2_s>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_IQ3_XXS:
            إذا (src0_reordered) {
                ثابت فراغ * vx = dst->src[0]->معلومات;
                get_لیکې_sycl_reorder_iq3(ctx, dst->src[0], dst->src[1], dst, src1_i32, (عائم *) dst->معلومات,
                    [=](عائم * y, int64_t iblk, ثابت sycl::nd_توکی<3> & توکی_ct1, int64_t nb) {
                        dequantize_بلاک_iq3_xxs_reorder(vx, y, iblk, توکی_ct1, nb,
                                                         iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs);
                    });
                اكسر;
            }
            get_لیکې_sycl<QK_K, 1, dequantize_iq3_xxs>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_IQ1_S:
            get_لیکې_sycl<QK_K, 1, dequantize_iq1_s>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_IQ1_M:
            get_لیکې_sycl<QK_K, 1, dequantize_iq1_m>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_IQ3_S:
            إذا (src0_reordered) {
                ثابت فراغ * vx = dst->src[0]->معلومات;
                get_لیکې_sycl_reorder_iq3(ctx, dst->src[0], dst->src[1], dst, src1_i32, (عائم *) dst->معلومات,
                    [=](عائم * y, int64_t iblk, ثابت sycl::nd_توکی<3> & توکی_ct1, int64_t nb) {
                        dequantize_بلاک_iq3_s_reorder(vx, y, iblk, توکی_ct1, nb,
                                                       kmask_iq2xs, iq3s_grid);
                    });
                اكسر;
            }
            get_لیکې_sycl<QK_K, 1, dequantize_iq3_s>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_IQ4_NL:
            get_لیکې_sycl<QK4_NL, 1, dequantize_iq4_nl>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_IQ4_XS:
            get_لیکې_sycl<QK_K, 1, dequantize_iq4_xs>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_Q2_K:
            إذا (src0_reordered) {
                ثابت فراغ * vx = dst->src[0]->معلومات;
                get_لیکې_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (عائم *)dst->معلومات, ctx.stream(),
                    [=](عائم * y, int64_t ib, uint8_t * scales_local, ثابت sycl::nd_توکی<3> & توکی_ct1, int64_t nb) {
                        dequantize_بلاک_q2_K_reorder(vx, y, ib, توکی_ct1, nb);
                        GGML_ناکارول_شوی(scales_local);
                    });
                اكسر;
            }
            get_لیکې_sycl_f32<QK_K, 1, dequantize_q2_K_f32>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_Q3_K:
            إذا (src0_reordered) {
                ثابت فراغ * vx = dst->src[0]->معلومات;
                get_لیکې_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (عائم *)dst->معلومات, ctx.stream(),
                    [=](عائم * y, int64_t ib, uint8_t * scales_local, ثابت sycl::nd_توکی<3> & توکی_ct1, int64_t nb) {
                        dequantize_بلاک_q3_K_reorder(vx, y, ib, توکی_ct1, nb);
                        GGML_ناکارول_شوی(scales_local);
                    });
                اكسر;
            }
            get_لیکې_sycl<QK_K, 1, dequantize_q3_K>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_Q4_0:
            إذا (src0_reordered) {
                get_لیکې_sycl_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(dst->src[0], dst->src[1], dst, dst->src[0]->معلومات,
                src1_i32, (عائم *)dst->معلومات, ctx.stream());
                اكسر;
            }
            get_لیکې_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_Q4_1:
            get_لیکې_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_Q4_K:
            إذا (src0_reordered) {
                ثابت فراغ * vx = dst->src[0]->معلومات;
                get_لیکې_sycl_reorder_k<1, 32, K_SCALE_اندازه>(dst->src[0], dst->src[1], dst, src1_i32, (عائم *)dst->معلومات, ctx.stream(),
                    [=](عائم * y, int64_t ib, uint8_t * scales_local, ثابت sycl::nd_توکی<1> & توکی_ct1, int64_t nb) {
                        dequantize_بلاک_q4_K_reorder(vx, y, ib, scales_local, توکی_ct1, nb);
                    });
                اكسر;
            }
            get_لیکې_sycl_f32<QK_K, 1, dequantize_q4_K_f32>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_Q5_0:
            get_لیکې_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_Q5_1:
            get_لیکې_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_Q5_K:
            إذا (src0_reordered) {
                ثابت فراغ * vx = dst->src[0]->معلومات;
                get_لیکې_sycl_reorder_k<3, 64, K_SCALE_اندازه>(dst->src[0], dst->src[1], dst, src1_i32, (عائم *)dst->معلومات, ctx.stream(),
                    [=](عائم * y, int64_t ib, uint8_t * scales_local, ثابت sycl::nd_توکی<3> & توکی_ct1, int64_t nb) {
                        dequantize_بلاک_q5_K_reorder(vx, y, ib, scales_local, توکی_ct1, nb);
                    });
                اكسر;
            }
            get_لیکې_sycl_f32<QK_K, 1, dequantize_q5_K_f32>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_Q6_K:
            إذا (src0_reordered) {
                ثابت فراغ * vx = dst->src[0]->معلومات;
                get_لیکې_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (عائم *)dst->معلومات, ctx.stream(),
                    [=](عائم * y, int64_t ib, uint8_t * scales_local, ثابت sycl::nd_توکی<3> & توکی_ct1, int64_t nb) {
                        dequantize_بلاک_q6_K_reorder(vx, y, ib, توکی_ct1, nb);
                        GGML_ناکارول_شوی(scales_local);
                    });
                اكسر;
            }
            get_لیکې_sycl<QK_K, 1, dequantize_q6_K>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        حالة GGML_TYPE_Q8_0:
            إذا (src0_reordered) {
                get_لیکې_sycl_reorder<QK8_0, QR8_0, dequantize_q8_0_reorder>(dst->src[0], dst->src[1], dst, dst->src[0]->معلومات,
                src1_i32, (عائم *)dst->معلومات, ctx.stream());
                اكسر;
            }
            get_لیکې_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, dst->src[0], dst->src[1], dst, (ثابت عائم *)dst->src[0]->معلومات,
            src1_i32, (عائم *)dst->معلومات, ctx.stream());
            اكسر;
        افتراضي:
            // TODO: k-quants
            GGML_LOG_ERROR("%s: unsupported type: %s\n", __func__, ggml_type_name(dst->src[0]->type));
            GGML_ABORT("fatal error");
    }
}

فراغ ggml_sycl_op_get_لیکې_شا(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(dst));

    بدل (src0->type) {
        حالة GGML_TYPE_F16:
            get_لیکې_شا_sycl_float(ctx, src0, src1, dst, (ثابت sycl::half *) src0->معلومات,
                                     (ثابت int32_t *) src1->معلومات, (عائم *) dst->معلومات,
                                     ctx.stream());
            اكسر;
        حالة GGML_TYPE_F32:
            get_لیکې_شا_sycl_float(ctx, src0, src1, dst, (ثابت عائم *) src0->معلومات,
                                     (ثابت int32_t *) src1->معلومات, (عائم *) dst->معلومات,
                                     ctx.stream());
            اكسر;
        افتراضي:
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0->type));
    }
}
