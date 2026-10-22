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

#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "common.hpp"
#అన్తర్భావ "dequantize.hpp"
#అన్తర్భావ "getrows.hpp"


మూస<పూర్ణాంకం qk, పూర్ణాంకం qr, dequantize_kernel_t dequantize_kernel, రకం_పేరు dst_t>
స్థిర శూన్య k_get_అడ్డువరుసలు(
            స్థిరరాశి శూన్య * src0, స్థిరరాశి int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1/*, size_t s13*/) {

    స్థిరరాశి పూర్ణాంకం i00 = (అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2) +
                     అంశం_ct1.get_local_id(2)) *
                    2;
    స్థిరరాశి పూర్ణాంకం i10 = అంశం_ct1.get_local_range(1) * అంశం_ct1.get_సమూహం(1) +
                    అంశం_ct1.get_local_id(1);
    స్థిరరాశి పూర్ణాంకం i11 = (అంశం_ct1.get_సమూహం(0) * అంశం_ct1.get_local_range(0) +
                     అంశం_ct1.get_local_id(0)) /
                    ne12;
    స్థిరరాశి పూర్ణాంకం i12 = (అంశం_ct1.get_సమూహం(0) * అంశం_ct1.get_local_range(0) +
                     అంశం_ct1.get_local_id(0)) %
                    ne12;

    ఒకవేళ (i00 >= ne00) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_అడ్డువరుస = dst + i10*s1 + i11*s2 + i12*s3;
    స్థిరరాశి శూన్య * src0_అడ్డువరుస = (స్థిరరాశి అక్షరం *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    స్థిరరాశి పూర్ణాంకం ib = i00/qk; // block index
    స్థిరరాశి పూర్ణాంకం iqs = (i00%qk)/qr; // quant index
    స్థిరరాశి పూర్ణాంకం iybs = i00 - i00%qk; // dst block start index
    స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(src0_అడ్డువరుస, ib, iqs, v);

    dst_అడ్డువరుస[iybs + iqs + 0] = v.x();
    dst_అడ్డువరుస[iybs + iqs + y_స్థానభ్రంశం] = v.y();
}

మూస<పూర్ణాంకం qk, పూర్ణాంకం qr, dequantize_kernel_f32_t dequantize_kernel, రకం_పేరు dst_t>
స్థిర శూన్య k_get_అడ్డువరుసలు_f32(
            స్థిరరాశి శూన్య * src0, స్థిరరాశి int32_t * src1, dst_t * dst,
            int64_t ne00,
            int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి పూర్ణాంకం i00 = (అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2) +
                     అంశం_ct1.get_local_id(2)) *
                    2;
    స్థిరరాశి పూర్ణాంకం i10 = అంశం_ct1.get_local_range(1) * అంశం_ct1.get_సమూహం(1) +
                    అంశం_ct1.get_local_id(1);
    స్థిరరాశి పూర్ణాంకం i11 = (అంశం_ct1.get_సమూహం(0) * అంశం_ct1.get_local_range(0) +
                     అంశం_ct1.get_local_id(0)) /
                    ne12;
    స్థిరరాశి పూర్ణాంకం i12 = (అంశం_ct1.get_సమూహం(0) * అంశం_ct1.get_local_range(0) +
                     అంశం_ct1.get_local_id(0)) %
                    ne12;

    ఒకవేళ (i00 >= ne00) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_అడ్డువరుస = dst + i10*s1 + i11*s2 + i12*s3;
    స్థిరరాశి శూన్య * src0_అడ్డువరుస = (స్థిరరాశి అక్షరం *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    స్థిరరాశి పూర్ణాంకం ib = i00/qk;
    స్థిరరాశి పూర్ణాంకం iqs = (i00%qk)/qr;
    స్థిరరాశి పూర్ణాంకం iybs = i00 - i00%qk;
    స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = qr == 1 ? 1 : qk/2;

    దశాంశం v0;
    దశాంశం v1;
    dequantize_kernel(src0_అడ్డువరుస, ib, iqs, v0, v1);

    dst_అడ్డువరుస[iybs + iqs + 0] = (dst_t) v0;
    dst_అడ్డువరుస[iybs + iqs + y_స్థానభ్రంశం] = (dst_t) v1;
}

మూస<రకం_పేరు src0_t, రకం_పేరు dst_t>
స్థిర శూన్య k_get_అడ్డువరుసలు_float(
            స్థిరరాశి src0_t * src0, స్థిరరాశి int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1/*, size_t s13*/) {

    స్థిరరాశి పూర్ణాంకం i00 = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2) +
                    అంశం_ct1.get_local_id(2);
    స్థిరరాశి పూర్ణాంకం i10 = అంశం_ct1.get_local_range(1) * అంశం_ct1.get_సమూహం(1) +
                    అంశం_ct1.get_local_id(1);
    స్థిరరాశి పూర్ణాంకం i11 = (అంశం_ct1.get_సమూహం(0) * అంశం_ct1.get_local_range(0) +
                     అంశం_ct1.get_local_id(0)) /
                    ne12;
    స్థిరరాశి పూర్ణాంకం i12 = (అంశం_ct1.get_సమూహం(0) * అంశం_ct1.get_local_range(0) +
                     అంశం_ct1.get_local_id(0)) %
                    ne12;

    ఒకవేళ (i00 >= ne00) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_అడ్డువరుస = dst + i10*s1 + i11*s2 + i12*s3;
    స్థిరరాశి src0_t * src0_అడ్డువరుస = (స్థిరరాశి src0_t *)((స్థిరరాశి అక్షరం *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_అడ్డువరుస[i00] = src0_అడ్డువరుస[i00];
}

// src0 is in reorder layout: all qs of the tensor, then all d
మూస<పూర్ణాంకం qk, పూర్ణాంకం qr, dequantize_kernel_t_reorder dequantize_kernel, రకం_పేరు dst_t>
స్థిర శూన్య k_get_అడ్డువరుసలు_reorder(
            స్థిరరాశి శూన్య * src0_qs, స్థిరరాశి శూన్య * src0_d, స్థిరరాశి int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t s10, size_t s11, size_t s12,
            స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి పూర్ణాంకం i00 = (అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2) +
                     అంశం_ct1.get_local_id(2)) *
                    2;
    స్థిరరాశి పూర్ణాంకం i10 = అంశం_ct1.get_local_range(1) * అంశం_ct1.get_సమూహం(1) +
                    అంశం_ct1.get_local_id(1);
    స్థిరరాశి పూర్ణాంకం i11 = (అంశం_ct1.get_సమూహం(0) * అంశం_ct1.get_local_range(0) +
                     అంశం_ct1.get_local_id(0)) /
                    ne12;
    స్థిరరాశి పూర్ణాంకం i12 = (అంశం_ct1.get_సమూహం(0) * అంశం_ct1.get_local_range(0) +
                     అంశం_ct1.get_local_id(0)) %
                    ne12;

    ఒకవేళ (i00 >= ne00) {
        తిరిగి;
    }

    స్థిరరాశి పూర్ణాంకం i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_అడ్డువరుస = dst + i10*s1 + i11*s2 + i12*s3;

    స్థిరరాశి int64_t ib = (int64_t) i01*(ne00/qk) + i00/qk; // block index in the whole tensor
    స్థిరరాశి పూర్ణాంకం iqs = (i00%qk)/qr;
    స్థిరరాశి పూర్ణాంకం iybs = i00 - i00%qk;
    స్థిరరాశి పూర్ణాంకం y_స్థానభ్రంశం = qr == 1 ? 1 : qk/2;

    dfloat2 v;
    dequantize_kernel(src0_d, ib, src0_qs, ib*(qk/qr) + iqs, v);

    dst_అడ్డువరుస[iybs + iqs + 0] = v.x();
    dst_అడ్డువరుస[iybs + iqs + y_స్థానభ్రంశం] = v.y();
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qr, dequantize_kernel_t dq>
స్థిర శూన్య get_అడ్డువరుసలు_sycl(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor *src0, స్థిరరాశి ggml_tensor *src1,
                          ggml_tensor *dst, స్థిరరాశి శూన్య *src0_dd,
                          స్థిరరాశి int32_t *src1_dd, దశాంశం *dst_dd,
                          వరుస_ptr stream) {

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (ne00 + 2*SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత - 1) / (2*SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత);
    స్థిరరాశి sycl::range<3> ఖండం_nums(ne11 * ne12, ne10, ఖండం_అంకె_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    స్థిరరాశి size_t s1 = nb1 / ggml_element_కొలత(dst);
    స్థిరరాశి size_t s2 = nb2 / ggml_element_కొలత(dst);
    స్థిరరాశి size_t s3 = nb3 / ggml_element_కొలత(dst);

    స్థిరరాశి size_t s10 = nb10 / ggml_element_కొలత(src1);
    స్థిరరాశి size_t s11 = nb11 / ggml_element_కొలత(src1);
    స్థిరరాశి size_t s12 = nb12 / ggml_element_కొలత(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> అంశం_ct1) {
                             k_get_అడ్డువరుసలు<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, అంశం_ct1);
                         });

    GGML_వాడని(dst);
    GGML_వాడని(ctx);
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qr, dequantize_kernel_t_reorder dq>
స్థిర శూన్య get_అడ్డువరుసలు_sycl_reorder(స్థిరరాశి ggml_tensor *src0, స్థిరరాశి ggml_tensor *src1,
                                  ggml_tensor *dst, స్థిరరాశి శూన్య *src0_dd,
                                  స్థిరరాశి int32_t *src1_dd, దశాంశం *dst_dd,
                                  వరుస_ptr stream) {

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    // reorder is done only on 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % 2 == 0);

    స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (ne00 + 2*SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత - 1) / (2*SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత);
    స్థిరరాశి sycl::range<3> ఖండం_nums(ne11 * ne12, ne10, ఖండం_అంకె_x);

    స్థిరరాశి size_t s1 = nb1 / ggml_element_కొలత(dst);
    స్థిరరాశి size_t s2 = nb2 / ggml_element_కొలత(dst);
    స్థిరరాశి size_t s3 = nb3 / ggml_element_కొలత(dst);

    స్థిరరాశి size_t s10 = nb10 / ggml_element_కొలత(src1);
    స్థిరరాశి size_t s11 = nb11 / ggml_element_కొలత(src1);
    స్థిరరాశి size_t s12 = nb12 / ggml_element_కొలత(src1);

    స్థిరరాశి శూన్య * src0_d = (స్థిరరాశి అక్షరం *) src0_dd + ne00*ne01/qr;

    stream->parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> అంశం_ct1) {
                             k_get_అడ్డువరుసలు_reorder<qk, qr, dq>(
                                 src0_dd, src0_d, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, s10, s11, s12, అంశం_ct1);
                         });

    GGML_వాడని(src0);
}

// K-quants in reorder layout: one work-group dequantizes one QK_K block of one row
మూస <పూర్ణాంకం dims, పూర్ణాంకం nthreads, పూర్ణాంకం nlocal, రకం_పేరు F>
స్థిర శూన్య get_అడ్డువరుసలు_sycl_reorder_k(స్థిరరాశి ggml_tensor *src0, స్థిరరాశి ggml_tensor *src1,
                                    ggml_tensor *dst, స్థిరరాశి int32_t *src1_dd, దశాంశం *dst_dd,
                                    వరుస_ptr stream, F dequantize_ఖండం) {

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    // reorder is done only on 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % QK_K == 0);

    స్థిరరాశి int64_t nblk = ne00 / QK_K;
    స్థిరరాశి int64_t nb = nblk * ne01; // blocks in the whole tensor
    స్థిరరాశి int64_t ngroups = ne10 * ne11 * ne12 * nblk;

    స్థిరరాశి size_t s1 = nb1 / ggml_element_కొలత(dst);
    స్థిరరాశి size_t s2 = nb2 / ggml_element_కొలత(dst);
    స్థిరరాశి size_t s3 = nb3 / ggml_element_కొలత(dst);

    స్థిరరాశి size_t s10 = nb10 / ggml_element_కొలత(src1);
    స్థిరరాశి size_t s11 = nb11 / ggml_element_కొలత(src1);
    స్థిరరాశి size_t s12 = nb12 / ggml_element_కొలత(src1);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scales_local(sycl::range<1>(nlocal), cgh);

        స్వయంచాలక kernel = [=](sycl::nd_అంశం<dims> అంశం_ct1) {
            స్థిరరాశి int64_t g   = అంశం_ct1.get_సమూహం(dims - 1);
            స్థిరరాశి int64_t b   = g % nblk;
            స్థిరరాశి int64_t r   = g / nblk;
            స్థిరరాశి int64_t i10 = r % ne10;
            స్థిరరాశి int64_t i11 = (r / ne10) % ne11;
            స్థిరరాశి int64_t i12 = r / (ne10 * ne11);

            స్థిరరాశి int64_t i01 = src1_dd[i10*s10 + i11*s11 + i12*s12];

            దశాంశం * y = dst_dd + i10*s1 + i11*s2 + i12*s3 + b*QK_K;
            dequantize_ఖండం(y, i01*nblk + b, get_సూచిక(scales_local), అంశం_ct1, nb);
        };

        ఒకవేళ constexpr (dims == 1) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(ngroups * nthreads), sycl::range<1>(nthreads)), kernel);
        } లేకపోతే {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ngroups * nthreads), sycl::range<3>(1, 1, nthreads)), kernel);
        }
    });

    GGML_వాడని(src0);
}

// IQ3_S / IQ3_XXS in the persistent reorder layout: one work-group of 32 threads
// dequantizes one QK_K block of one requested row, reusing the reorder-aware
// block dequantizers that the dequantize-based mul_mat path already uses.
మూస <రకం_పేరు F>
స్థిర శూన్య get_అడ్డువరుసలు_sycl_reorder_iq3(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor * src0,
                                      స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst,
                                      స్థిరరాశి int32_t * src1_dd, దశాంశం * dst_dd,
                                      F dequantize_ఖండం) {
    GGML_TENSOR_ద్విమాన_OP_LOCALS

    // the persistent reorder is only installed for 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % QK_K == 0);

    స్థిరరాశి int64_t nblk    = ne00 / QK_K;
    స్థిరరాశి int64_t nb      = nblk * ne01; // blocks in the whole tensor
    స్థిరరాశి int64_t ngroups = ne10 * ne11 * ne12 * nblk;

    స్థిరరాశి size_t s1  = nb1 / ggml_element_కొలత(dst);
    స్థిరరాశి size_t s2  = nb2 / ggml_element_కొలత(dst);
    స్థిరరాశి size_t s3  = nb3 / ggml_element_కొలత(dst);

    స్థిరరాశి size_t s10 = nb10 / ggml_element_కొలత(src1);
    స్థిరరాశి size_t s11 = nb11 / ggml_element_కొలత(src1);
    స్థిరరాశి size_t s12 = nb12 / ggml_element_కొలత(src1);

    వరుస_ptr stream = ctx.stream();
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ngroups * 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_అంశం<3> అంశం_ct1) {
            స్థిరరాశి int64_t g   = అంశం_ct1.get_సమూహం(2);
            స్థిరరాశి int64_t b   = g % nblk;
            స్థిరరాశి int64_t r   = g / nblk;
            స్థిరరాశి int64_t i10 = r % ne10;
            స్థిరరాశి int64_t i11 = (r / ne10) % ne11;
            స్థిరరాశి int64_t i12 = r / (ne10 * ne11);

            స్థిరరాశి int64_t i01 = src1_dd[i10*s10 + i11*s11 + i12*s12];

            దశాంశం * y = dst_dd + i10*s1 + i11*s2 + i12*s3 + b*QK_K;
            dequantize_ఖండం(y, i01*nblk + b, అంశం_ct1, nb);
        });

    GGML_వాడని(dst);
}

మూస <పూర్ణాంకం qk, పూర్ణాంకం qr, dequantize_kernel_f32_t dq>
స్థిర శూన్య get_అడ్డువరుసలు_sycl_f32(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor *src0, స్థిరరాశి ggml_tensor *src1,
                              ggml_tensor *dst, స్థిరరాశి శూన్య *src0_dd,
                              స్థిరరాశి int32_t *src1_dd, దశాంశం *dst_dd,
                              వరుస_ptr stream) {

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (ne00 + 2*SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత - 1) / (2*SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత);
    స్థిరరాశి sycl::range<3> ఖండం_nums(ne11 * ne12, ne10, ఖండం_అంకె_x);

    స్థిరరాశి size_t s1 = nb1 / ggml_element_కొలత(dst);
    స్థిరరాశి size_t s2 = nb2 / ggml_element_కొలత(dst);
    స్థిరరాశి size_t s3 = nb3 / ggml_element_కొలత(dst);

    స్థిరరాశి size_t s10 = nb10 / ggml_element_కొలత(src1);
    స్థిరరాశి size_t s11 = nb11 / ggml_element_కొలత(src1);
    స్థిరరాశి size_t s12 = nb12 / ggml_element_కొలత(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> అంశం_ct1) {
                             k_get_అడ్డువరుసలు_f32<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, అంశం_ct1);
                         });

    GGML_వాడని(dst);
    GGML_వాడని(ctx);
}

మూస <రకం_పేరు src0_t, రకం_పేరు dst_t>
స్థిర శూన్య get_అడ్డువరుసలు_sycl_float(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor *src0,
                                స్థిరరాశి ggml_tensor *src1, ggml_tensor *dst,
                                స్థిరరాశి src0_t *src0_dd, స్థిరరాశి int32_t *src1_dd,
                                dst_t *dst_dd, వరుస_ptr stream) {

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత);
    స్థిరరాశి పూర్ణాంకం ఖండం_అంకె_x = (ne00 + SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత - 1) / SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత;
    స్థిరరాశి sycl::range<3> ఖండం_nums(ne11 * ne12, ne10, ఖండం_అంకె_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    స్థిరరాశి size_t s1 = nb1 / ggml_element_కొలత(dst);
    స్థిరరాశి size_t s2 = nb2 / ggml_element_కొలత(dst);
    స్థిరరాశి size_t s3 = nb3 / ggml_element_కొలత(dst);

    స్థిరరాశి size_t s10 = nb10 / ggml_element_కొలత(src1);
    స్థిరరాశి size_t s11 = nb11 / ggml_element_కొలత(src1);
    స్థిరరాశి size_t s12 = nb12 / ggml_element_కొలత(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
            [=](sycl::nd_అంశం<3> అంశం_ct1) {
                k_get_అడ్డువరుసలు_float(src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, అంశం_ct1);
            });
    }

    GGML_వాడని(dst);
    GGML_వాడని(ctx);
}

మూస <రకం_పేరు src0_t>
స్థిర శూన్య k_get_అడ్డువరుసలు_వెనుక_float(స్థిరరాశి src0_t * src0, స్థిరరాశి int32_t * src1, దశాంశం * dst,
                                  స్థిరరాశి int64_t ncols, స్థిరరాశి int64_t nrows_grad_10, స్థిరరాశి int64_t nrows_grad_11, స్థిరరాశి int64_t nrows_dst,
                                  స్థిరరాశి size_t s01, స్థిరరాశి size_t s02,
                                  స్థిరరాశి size_t s10, స్థిరరాశి size_t s11,
                                  స్థిరరాశి size_t s1,
                                  స్థిరరాశి int64_t ఖండం_అంకె_y,
                                  స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {
    స్థిరరాశి int64_t నిలువు = అంశం_ct1.get_సమూహం(2) * అంశం_ct1.get_local_range(2) + అంశం_ct1.get_local_id(2);
    ఒకవేళ (నిలువు >= ncols) {
        తిరిగి;
    }

    // block_num_y is clamped, so stride over destination rows like CUDA k_get_rows_back_float
    కొరకు (int64_t dst_అడ్డువరుస = అంశం_ct1.get_సమూహం(1); dst_అడ్డువరుస < nrows_dst; dst_అడ్డువరుస += ఖండం_అంకె_y) {
        దశాంశం కూడిక = 0.0f;

        స్థిరరాశి int64_t nrows_grad_మొత్తం = nrows_grad_10 * nrows_grad_11;
        కొరకు (int64_t i = 0; i < nrows_grad_మొత్తం; ++i) {
            స్థిరరాశి int64_t i10 = i % nrows_grad_10;
            స్థిరరాశి int64_t i11 = i / nrows_grad_10;
            ఒకవేళ (src1[i10*s10 + i11*s11] != dst_అడ్డువరుస) {
                కొనసాగు;
            }
            కూడిక += (దశాంశం) src0[నిలువు + i10*s01 + i11*s02];
        }

        dst[నిలువు + dst_అడ్డువరుస*s1] = కూడిక;
    }
}

మూస <రకం_పేరు src0_t>
స్థిర శూన్య get_అడ్డువరుసలు_వెనుక_sycl_float(ggml_backend_sycl_context & ctx, స్థిరరాశి ggml_tensor * src0,
                                     స్థిరరాశి ggml_tensor * src1, ggml_tensor * dst,
                                     స్థిరరాశి src0_t * src0_dd, స్థిరరాశి int32_t * src1_dd,
                                     దశాంశం * dst_dd, వరుస_ptr stream) {

    GGML_TENSOR_ద్విమాన_OP_LOCALS

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);
    GGML_ASSERT(src0->nb[0] == ggml_type_కొలత(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_కొలత(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_కొలత(dst->type));

    స్థిరరాశి int64_t ncols = ne00;
    స్థిరరాశి int64_t nrows_grad_10 = ne10;
    స్థిరరాశి int64_t nrows_grad_11 = ne11;
    స్థిరరాశి int64_t nrows_dst = ne1;

    స్థిరరాశి size_t s01 = nb01 / పరిమాణం(src0_t);
    స్థిరరాశి size_t s02 = nb02 / పరిమాణం(src0_t);

    స్థిరరాశి size_t s10 = nb10 / పరిమాణం(int32_t);
    స్థిరరాశి size_t s11 = nb11 / పరిమాణం(int32_t);

    స్థిరరాశి size_t s1 = nb1 / పరిమాణం(దశాంశం);

    స్థిరరాశి sycl::range<3> ఖండం_dims(1, 1, SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత);
    స్థిరరాశి int64_t ఖండం_అంకె_x = (ncols + SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత - 1) / SYCL_GET_అడ్డువరుసలు_ఖండం_కొలత;
    స్థిరరాశి int64_t ఖండం_అంకె_y = std::min<int64_t>(nrows_dst, (int64_t) UINT16_MAX);
    స్థిరరాశి sycl::range<3> ఖండం_nums(1, ఖండం_అంకె_y, ఖండం_అంకె_x);

    stream->parallel_for(sycl::nd_range<3>(ఖండం_nums * ఖండం_dims, ఖండం_dims),
                         [=](sycl::nd_అంశం<3> అంశం_ct1) {
                             k_get_అడ్డువరుసలు_వెనుక_float(src0_dd, src1_dd, dst_dd,
                                                   ncols, nrows_grad_10, nrows_grad_11, nrows_dst,
                                                   s01, s02, s10, s11, s1,
                                                   ఖండం_అంకె_y, అంశం_ct1);
                         });

    GGML_వాడని(src0);
    GGML_వాడని(src1);
    GGML_వాడని(dst);
    GGML_వాడని(ctx);
}

శూన్య ggml_sycl_op_get_అడ్డువరుసలు(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_I32 );

    GGML_ASSERT(dst->src[0]->nb[0] == ggml_type_కొలత(dst->src[0]->type));
    GGML_ASSERT(dst->src[1]->nb[0] == ggml_type_కొలత(dst->src[1]->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_కొలత(dst->type));

    స్థిరరాశి int32_t * src1_i32 = (స్థిరరాశి int32_t *) dst->src[1]->దత్తాంశం;
    // a weight shared with MUL_MAT (e.g. tied embeddings) can be in the
    // persistent reorder layout (incl. IQ3_S / IQ3_XXS since the SYCL IQ3
    // reorder work)
    స్థిరరాశి తార్కిక src0_reordered = dst->src[0]->extra &&
        ((స్థిరరాశి ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
    /* TODO: Refactor and remove duplicates */
    ఎంపిక (dst->src[0]->type) {
        స్థితి GGML_TYPE_F16:
            get_అడ్డువరుసలు_sycl_float(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి sycl::half *)dst->src[0]->దత్తాంశం,
                                src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_BF16:
            get_అడ్డువరుసలు_sycl_float(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి sycl::ext::oneapi::bfloat16 *)dst->src[0]->దత్తాంశం,
                                src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_F32:
            get_అడ్డువరుసలు_sycl_float(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_I32:
            get_అడ్డువరుసలు_sycl_float(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి int32_t *)dst->src[0]->దత్తాంశం,
            src1_i32, (int32_t *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_Q1_0:
            get_అడ్డువరుసలు_sycl<QK1_0, 1, dequantize_q1_0>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_MXFP4:
            get_అడ్డువరుసలు_sycl<QK_MXFP4, 2, dequantize_mxfp4>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_NVFP4:
            get_అడ్డువరుసలు_sycl<QK_NVFP4, 1, dequantize_nvfp4>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_IQ2_XXS:
            get_అడ్డువరుసలు_sycl<QK_K, 1, dequantize_iq2_xxs>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_IQ2_XS:
            get_అడ్డువరుసలు_sycl<QK_K, 1, dequantize_iq2_xs>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_IQ2_S:
            get_అడ్డువరుసలు_sycl<QK_K, 1, dequantize_iq2_s>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_IQ3_XXS:
            ఒకవేళ (src0_reordered) {
                స్థిరరాశి శూన్య * vx = dst->src[0]->దత్తాంశం;
                get_అడ్డువరుసలు_sycl_reorder_iq3(ctx, dst->src[0], dst->src[1], dst, src1_i32, (దశాంశం *) dst->దత్తాంశం,
                    [=](దశాంశం * y, int64_t iblk, స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1, int64_t nb) {
                        dequantize_ఖండం_iq3_xxs_reorder(vx, y, iblk, అంశం_ct1, nb,
                                                         iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs);
                    });
                ఆపు;
            }
            get_అడ్డువరుసలు_sycl<QK_K, 1, dequantize_iq3_xxs>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_IQ1_S:
            get_అడ్డువరుసలు_sycl<QK_K, 1, dequantize_iq1_s>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_IQ1_M:
            get_అడ్డువరుసలు_sycl<QK_K, 1, dequantize_iq1_m>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_IQ3_S:
            ఒకవేళ (src0_reordered) {
                స్థిరరాశి శూన్య * vx = dst->src[0]->దత్తాంశం;
                get_అడ్డువరుసలు_sycl_reorder_iq3(ctx, dst->src[0], dst->src[1], dst, src1_i32, (దశాంశం *) dst->దత్తాంశం,
                    [=](దశాంశం * y, int64_t iblk, స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1, int64_t nb) {
                        dequantize_ఖండం_iq3_s_reorder(vx, y, iblk, అంశం_ct1, nb,
                                                       kmask_iq2xs, iq3s_grid);
                    });
                ఆపు;
            }
            get_అడ్డువరుసలు_sycl<QK_K, 1, dequantize_iq3_s>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_IQ4_NL:
            get_అడ్డువరుసలు_sycl<QK4_NL, 1, dequantize_iq4_nl>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_IQ4_XS:
            get_అడ్డువరుసలు_sycl<QK_K, 1, dequantize_iq4_xs>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_Q2_K:
            ఒకవేళ (src0_reordered) {
                స్థిరరాశి శూన్య * vx = dst->src[0]->దత్తాంశం;
                get_అడ్డువరుసలు_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream(),
                    [=](దశాంశం * y, int64_t ib, uint8_t * scales_local, స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1, int64_t nb) {
                        dequantize_ఖండం_q2_K_reorder(vx, y, ib, అంశం_ct1, nb);
                        GGML_వాడని(scales_local);
                    });
                ఆపు;
            }
            get_అడ్డువరుసలు_sycl_f32<QK_K, 1, dequantize_q2_K_f32>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_Q3_K:
            ఒకవేళ (src0_reordered) {
                స్థిరరాశి శూన్య * vx = dst->src[0]->దత్తాంశం;
                get_అడ్డువరుసలు_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream(),
                    [=](దశాంశం * y, int64_t ib, uint8_t * scales_local, స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1, int64_t nb) {
                        dequantize_ఖండం_q3_K_reorder(vx, y, ib, అంశం_ct1, nb);
                        GGML_వాడని(scales_local);
                    });
                ఆపు;
            }
            get_అడ్డువరుసలు_sycl<QK_K, 1, dequantize_q3_K>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_Q4_0:
            ఒకవేళ (src0_reordered) {
                get_అడ్డువరుసలు_sycl_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(dst->src[0], dst->src[1], dst, dst->src[0]->దత్తాంశం,
                src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
                ఆపు;
            }
            get_అడ్డువరుసలు_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_Q4_1:
            get_అడ్డువరుసలు_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_Q4_K:
            ఒకవేళ (src0_reordered) {
                స్థిరరాశి శూన్య * vx = dst->src[0]->దత్తాంశం;
                get_అడ్డువరుసలు_sycl_reorder_k<1, 32, K_SCALE_కొలత>(dst->src[0], dst->src[1], dst, src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream(),
                    [=](దశాంశం * y, int64_t ib, uint8_t * scales_local, స్థిరరాశి sycl::nd_అంశం<1> & అంశం_ct1, int64_t nb) {
                        dequantize_ఖండం_q4_K_reorder(vx, y, ib, scales_local, అంశం_ct1, nb);
                    });
                ఆపు;
            }
            get_అడ్డువరుసలు_sycl_f32<QK_K, 1, dequantize_q4_K_f32>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_Q5_0:
            get_అడ్డువరుసలు_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_Q5_1:
            get_అడ్డువరుసలు_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_Q5_K:
            ఒకవేళ (src0_reordered) {
                స్థిరరాశి శూన్య * vx = dst->src[0]->దత్తాంశం;
                get_అడ్డువరుసలు_sycl_reorder_k<3, 64, K_SCALE_కొలత>(dst->src[0], dst->src[1], dst, src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream(),
                    [=](దశాంశం * y, int64_t ib, uint8_t * scales_local, స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1, int64_t nb) {
                        dequantize_ఖండం_q5_K_reorder(vx, y, ib, scales_local, అంశం_ct1, nb);
                    });
                ఆపు;
            }
            get_అడ్డువరుసలు_sycl_f32<QK_K, 1, dequantize_q5_K_f32>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_Q6_K:
            ఒకవేళ (src0_reordered) {
                స్థిరరాశి శూన్య * vx = dst->src[0]->దత్తాంశం;
                get_అడ్డువరుసలు_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream(),
                    [=](దశాంశం * y, int64_t ib, uint8_t * scales_local, స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1, int64_t nb) {
                        dequantize_ఖండం_q6_K_reorder(vx, y, ib, అంశం_ct1, nb);
                        GGML_వాడని(scales_local);
                    });
                ఆపు;
            }
            get_అడ్డువరుసలు_sycl<QK_K, 1, dequantize_q6_K>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_Q8_0:
            ఒకవేళ (src0_reordered) {
                get_అడ్డువరుసలు_sycl_reorder<QK8_0, QR8_0, dequantize_q8_0_reorder>(dst->src[0], dst->src[1], dst, dst->src[0]->దత్తాంశం,
                src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
                ఆపు;
            }
            get_అడ్డువరుసలు_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, dst->src[0], dst->src[1], dst, (స్థిరరాశి దశాంశం *)dst->src[0]->దత్తాంశం,
            src1_i32, (దశాంశం *)dst->దత్తాంశం, ctx.stream());
            ఆపు;
        అప్రమేయ:
            // TODO: k-quants
            GGML_LOG_ERROR("%s: unsupported type: %s\n", __func__, ggml_type_name(dst->src[0]->type));
            GGML_ABORT("fatal error");
    }
}

శూన్య ggml_sycl_op_get_అడ్డువరుసలు_వెనుక(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    స్థిరరాశి ggml_tensor * src0 = dst->src[0];
    స్థిరరాశి ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(dst));

    ఎంపిక (src0->type) {
        స్థితి GGML_TYPE_F16:
            get_అడ్డువరుసలు_వెనుక_sycl_float(ctx, src0, src1, dst, (స్థిరరాశి sycl::half *) src0->దత్తాంశం,
                                     (స్థిరరాశి int32_t *) src1->దత్తాంశం, (దశాంశం *) dst->దత్తాంశం,
                                     ctx.stream());
            ఆపు;
        స్థితి GGML_TYPE_F32:
            get_అడ్డువరుసలు_వెనుక_sycl_float(ctx, src0, src1, dst, (స్థిరరాశి దశాంశం *) src0->దత్తాంశం,
                                     (స్థిరరాశి int32_t *) src1->దత్తాంశం, (దశాంశం *) dst->దత్తాంశం,
                                     ctx.stream());
            ఆపు;
        అప్రమేయ:
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0->type));
    }
}
