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

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "dequantize.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "getrows.hpp"


ਸਾਂਚਾ<ਪੂਰਨ qk, ਪੂਰਨ qr, dequantize_kernel_t dequantize_kernel, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ k_get_ਪੰਕਤੀਆਂ(
            ਅਚਲ ਖਾਲੀ * src0, ਅਚਲ int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1/*, size_t s13*/) {

    ਅਚਲ ਪੂਰਨ i00 = (ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2) +
                     ਵਸਤੂ_ct1.get_local_id(2)) *
                    2;
    ਅਚਲ ਪੂਰਨ i10 = ਵਸਤੂ_ct1.get_local_range(1) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਅਚਲ ਪੂਰਨ i11 = (ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ਵਸਤੂ_ct1.get_local_range(0) +
                     ਵਸਤੂ_ct1.get_local_id(0)) /
                    ne12;
    ਅਚਲ ਪੂਰਨ i12 = (ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ਵਸਤੂ_ct1.get_local_range(0) +
                     ਵਸਤੂ_ct1.get_local_id(0)) %
                    ne12;

    ਜੇ (i00 >= ne00) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_ਪੰਕਤੀ = dst + i10*s1 + i11*s2 + i12*s3;
    ਅਚਲ ਖਾਲੀ * src0_ਪੰਕਤੀ = (ਅਚਲ ਅੱਖਰ *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    ਅਚਲ ਪੂਰਨ ib = i00/qk; // block index
    ਅਚਲ ਪੂਰਨ iqs = (i00%qk)/qr; // quant index
    ਅਚਲ ਪੂਰਨ iybs = i00 - i00%qk; // dst block start index
    ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(src0_ਪੰਕਤੀ, ib, iqs, v);

    dst_ਪੰਕਤੀ[iybs + iqs + 0] = v.x();
    dst_ਪੰਕਤੀ[iybs + iqs + y_ਵਿੱਥ] = v.y();
}

ਸਾਂਚਾ<ਪੂਰਨ qk, ਪੂਰਨ qr, dequantize_kernel_f32_t dequantize_kernel, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ k_get_ਪੰਕਤੀਆਂ_f32(
            ਅਚਲ ਖਾਲੀ * src0, ਅਚਲ int32_t * src1, dst_t * dst,
            int64_t ne00,
            int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ ਪੂਰਨ i00 = (ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2) +
                     ਵਸਤੂ_ct1.get_local_id(2)) *
                    2;
    ਅਚਲ ਪੂਰਨ i10 = ਵਸਤੂ_ct1.get_local_range(1) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਅਚਲ ਪੂਰਨ i11 = (ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ਵਸਤੂ_ct1.get_local_range(0) +
                     ਵਸਤੂ_ct1.get_local_id(0)) /
                    ne12;
    ਅਚਲ ਪੂਰਨ i12 = (ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ਵਸਤੂ_ct1.get_local_range(0) +
                     ਵਸਤੂ_ct1.get_local_id(0)) %
                    ne12;

    ਜੇ (i00 >= ne00) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_ਪੰਕਤੀ = dst + i10*s1 + i11*s2 + i12*s3;
    ਅਚਲ ਖਾਲੀ * src0_ਪੰਕਤੀ = (ਅਚਲ ਅੱਖਰ *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    ਅਚਲ ਪੂਰਨ ib = i00/qk;
    ਅਚਲ ਪੂਰਨ iqs = (i00%qk)/qr;
    ਅਚਲ ਪੂਰਨ iybs = i00 - i00%qk;
    ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = qr == 1 ? 1 : qk/2;

    ਦਸ਼ਮਲਵ v0;
    ਦਸ਼ਮਲਵ v1;
    dequantize_kernel(src0_ਪੰਕਤੀ, ib, iqs, v0, v1);

    dst_ਪੰਕਤੀ[iybs + iqs + 0] = (dst_t) v0;
    dst_ਪੰਕਤੀ[iybs + iqs + y_ਵਿੱਥ] = (dst_t) v1;
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ k_get_ਪੰਕਤੀਆਂ_float(
            ਅਚਲ src0_t * src0, ਅਚਲ int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1/*, size_t s13*/) {

    ਅਚਲ ਪੂਰਨ i00 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2) +
                    ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ ਪੂਰਨ i10 = ਵਸਤੂ_ct1.get_local_range(1) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਅਚਲ ਪੂਰਨ i11 = (ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ਵਸਤੂ_ct1.get_local_range(0) +
                     ਵਸਤੂ_ct1.get_local_id(0)) /
                    ne12;
    ਅਚਲ ਪੂਰਨ i12 = (ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ਵਸਤੂ_ct1.get_local_range(0) +
                     ਵਸਤੂ_ct1.get_local_id(0)) %
                    ne12;

    ਜੇ (i00 >= ne00) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_ਪੰਕਤੀ = dst + i10*s1 + i11*s2 + i12*s3;
    ਅਚਲ src0_t * src0_ਪੰਕਤੀ = (ਅਚਲ src0_t *)((ਅਚਲ ਅੱਖਰ *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_ਪੰਕਤੀ[i00] = src0_ਪੰਕਤੀ[i00];
}

// src0 is in reorder layout: all qs of the tensor, then all d
ਸਾਂਚਾ<ਪੂਰਨ qk, ਪੂਰਨ qr, dequantize_kernel_t_reorder dequantize_kernel, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ k_get_ਪੰਕਤੀਆਂ_reorder(
            ਅਚਲ ਖਾਲੀ * src0_qs, ਅਚਲ ਖਾਲੀ * src0_d, ਅਚਲ int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t s10, size_t s11, size_t s12,
            ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ ਪੂਰਨ i00 = (ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2) +
                     ਵਸਤੂ_ct1.get_local_id(2)) *
                    2;
    ਅਚਲ ਪੂਰਨ i10 = ਵਸਤੂ_ct1.get_local_range(1) * ਵਸਤੂ_ct1.get_ਸਮੂਹ(1) +
                    ਵਸਤੂ_ct1.get_local_id(1);
    ਅਚਲ ਪੂਰਨ i11 = (ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ਵਸਤੂ_ct1.get_local_range(0) +
                     ਵਸਤੂ_ct1.get_local_id(0)) /
                    ne12;
    ਅਚਲ ਪੂਰਨ i12 = (ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) * ਵਸਤੂ_ct1.get_local_range(0) +
                     ਵਸਤੂ_ct1.get_local_id(0)) %
                    ne12;

    ਜੇ (i00 >= ne00) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਪੂਰਨ i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_ਪੰਕਤੀ = dst + i10*s1 + i11*s2 + i12*s3;

    ਅਚਲ int64_t ib = (int64_t) i01*(ne00/qk) + i00/qk; // block index in the whole tensor
    ਅਚਲ ਪੂਰਨ iqs = (i00%qk)/qr;
    ਅਚਲ ਪੂਰਨ iybs = i00 - i00%qk;
    ਅਚਲ ਪੂਰਨ y_ਵਿੱਥ = qr == 1 ? 1 : qk/2;

    dfloat2 v;
    dequantize_kernel(src0_d, ib, src0_qs, ib*(qk/qr) + iqs, v);

    dst_ਪੰਕਤੀ[iybs + iqs + 0] = v.x();
    dst_ਪੰਕਤੀ[iybs + iqs + y_ਵਿੱਥ] = v.y();
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qr, dequantize_kernel_t dq>
ਸਥਿਰ ਖਾਲੀ get_ਪੰਕਤੀਆਂ_sycl(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor *src0, ਅਚਲ ggml_tensor *src1,
                          ggml_tensor *dst, ਅਚਲ ਖਾਲੀ *src0_dd,
                          ਅਚਲ int32_t *src1_dd, ਦਸ਼ਮਲਵ *dst_dd,
                          ਕਤਾਰ_ptr stream) {

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (ne00 + 2*SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ - 1) / (2*SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(ne11 * ne12, ne10, ਖੰਡ_ਅੰਕ_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    ਅਚਲ size_t s1 = nb1 / ggml_element_ਮਾਪ(dst);
    ਅਚਲ size_t s2 = nb2 / ggml_element_ਮਾਪ(dst);
    ਅਚਲ size_t s3 = nb3 / ggml_element_ਮਾਪ(dst);

    ਅਚਲ size_t s10 = nb10 / ggml_element_ਮਾਪ(src1);
    ਅਚਲ size_t s11 = nb11 / ggml_element_ਮਾਪ(src1);
    ਅਚਲ size_t s12 = nb12 / ggml_element_ਮਾਪ(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                             k_get_ਪੰਕਤੀਆਂ<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, ਵਸਤੂ_ct1);
                         });

    GGML_ਅਣਵਰਤਿਆ(dst);
    GGML_ਅਣਵਰਤਿਆ(ctx);
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qr, dequantize_kernel_t_reorder dq>
ਸਥਿਰ ਖਾਲੀ get_ਪੰਕਤੀਆਂ_sycl_reorder(ਅਚਲ ggml_tensor *src0, ਅਚਲ ggml_tensor *src1,
                                  ggml_tensor *dst, ਅਚਲ ਖਾਲੀ *src0_dd,
                                  ਅਚਲ int32_t *src1_dd, ਦਸ਼ਮਲਵ *dst_dd,
                                  ਕਤਾਰ_ptr stream) {

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    // reorder is done only on 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % 2 == 0);

    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (ne00 + 2*SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ - 1) / (2*SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(ne11 * ne12, ne10, ਖੰਡ_ਅੰਕ_x);

    ਅਚਲ size_t s1 = nb1 / ggml_element_ਮਾਪ(dst);
    ਅਚਲ size_t s2 = nb2 / ggml_element_ਮਾਪ(dst);
    ਅਚਲ size_t s3 = nb3 / ggml_element_ਮਾਪ(dst);

    ਅਚਲ size_t s10 = nb10 / ggml_element_ਮਾਪ(src1);
    ਅਚਲ size_t s11 = nb11 / ggml_element_ਮਾਪ(src1);
    ਅਚਲ size_t s12 = nb12 / ggml_element_ਮਾਪ(src1);

    ਅਚਲ ਖਾਲੀ * src0_d = (ਅਚਲ ਅੱਖਰ *) src0_dd + ne00*ne01/qr;

    stream->parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                             k_get_ਪੰਕਤੀਆਂ_reorder<qk, qr, dq>(
                                 src0_dd, src0_d, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, s10, s11, s12, ਵਸਤੂ_ct1);
                         });

    GGML_ਅਣਵਰਤਿਆ(src0);
}

// K-quants in reorder layout: one work-group dequantizes one QK_K block of one row
ਸਾਂਚਾ <ਪੂਰਨ dims, ਪੂਰਨ nthreads, ਪੂਰਨ nlocal, ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਖਾਲੀ get_ਪੰਕਤੀਆਂ_sycl_reorder_k(ਅਚਲ ggml_tensor *src0, ਅਚਲ ggml_tensor *src1,
                                    ggml_tensor *dst, ਅਚਲ int32_t *src1_dd, ਦਸ਼ਮਲਵ *dst_dd,
                                    ਕਤਾਰ_ptr stream, F dequantize_ਖੰਡ) {

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    // reorder is done only on 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % QK_K == 0);

    ਅਚਲ int64_t nblk = ne00 / QK_K;
    ਅਚਲ int64_t nb = nblk * ne01; // blocks in the whole tensor
    ਅਚਲ int64_t ngroups = ne10 * ne11 * ne12 * nblk;

    ਅਚਲ size_t s1 = nb1 / ggml_element_ਮਾਪ(dst);
    ਅਚਲ size_t s2 = nb2 / ggml_element_ਮਾਪ(dst);
    ਅਚਲ size_t s3 = nb3 / ggml_element_ਮਾਪ(dst);

    ਅਚਲ size_t s10 = nb10 / ggml_element_ਮਾਪ(src1);
    ਅਚਲ size_t s11 = nb11 / ggml_element_ਮਾਪ(src1);
    ਅਚਲ size_t s12 = nb12 / ggml_element_ਮਾਪ(src1);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scales_local(sycl::range<1>(nlocal), cgh);

        ਸਵੈਚਾਲਿਤ kernel = [=](sycl::nd_ਵਸਤੂ<dims> ਵਸਤੂ_ct1) {
            ਅਚਲ int64_t g   = ਵਸਤੂ_ct1.get_ਸਮੂਹ(dims - 1);
            ਅਚਲ int64_t b   = g % nblk;
            ਅਚਲ int64_t r   = g / nblk;
            ਅਚਲ int64_t i10 = r % ne10;
            ਅਚਲ int64_t i11 = (r / ne10) % ne11;
            ਅਚਲ int64_t i12 = r / (ne10 * ne11);

            ਅਚਲ int64_t i01 = src1_dd[i10*s10 + i11*s11 + i12*s12];

            ਦਸ਼ਮਲਵ * y = dst_dd + i10*s1 + i11*s2 + i12*s3 + b*QK_K;
            dequantize_ਖੰਡ(y, i01*nblk + b, get_ਸੂਚਕ(scales_local), ਵਸਤੂ_ct1, nb);
        };

        ਜੇ constexpr (dims == 1) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(ngroups * nthreads), sycl::range<1>(nthreads)), kernel);
        } ਨਹੀਂਤਾਂ {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ngroups * nthreads), sycl::range<3>(1, 1, nthreads)), kernel);
        }
    });

    GGML_ਅਣਵਰਤਿਆ(src0);
}

// IQ3_S / IQ3_XXS in the persistent reorder layout: one work-group of 32 threads
// dequantizes one QK_K block of one requested row, reusing the reorder-aware
// block dequantizers that the dequantize-based mul_mat path already uses.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ F>
ਸਥਿਰ ਖਾਲੀ get_ਪੰਕਤੀਆਂ_sycl_reorder_iq3(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor * src0,
                                      ਅਚਲ ggml_tensor * src1, ggml_tensor * dst,
                                      ਅਚਲ int32_t * src1_dd, ਦਸ਼ਮਲਵ * dst_dd,
                                      F dequantize_ਖੰਡ) {
    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    // the persistent reorder is only installed for 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % QK_K == 0);

    ਅਚਲ int64_t nblk    = ne00 / QK_K;
    ਅਚਲ int64_t nb      = nblk * ne01; // blocks in the whole tensor
    ਅਚਲ int64_t ngroups = ne10 * ne11 * ne12 * nblk;

    ਅਚਲ size_t s1  = nb1 / ggml_element_ਮਾਪ(dst);
    ਅਚਲ size_t s2  = nb2 / ggml_element_ਮਾਪ(dst);
    ਅਚਲ size_t s3  = nb3 / ggml_element_ਮਾਪ(dst);

    ਅਚਲ size_t s10 = nb10 / ggml_element_ਮਾਪ(src1);
    ਅਚਲ size_t s11 = nb11 / ggml_element_ਮਾਪ(src1);
    ਅਚਲ size_t s12 = nb12 / ggml_element_ਮਾਪ(src1);

    ਕਤਾਰ_ptr stream = ctx.stream();
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ngroups * 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            ਅਚਲ int64_t g   = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
            ਅਚਲ int64_t b   = g % nblk;
            ਅਚਲ int64_t r   = g / nblk;
            ਅਚਲ int64_t i10 = r % ne10;
            ਅਚਲ int64_t i11 = (r / ne10) % ne11;
            ਅਚਲ int64_t i12 = r / (ne10 * ne11);

            ਅਚਲ int64_t i01 = src1_dd[i10*s10 + i11*s11 + i12*s12];

            ਦਸ਼ਮਲਵ * y = dst_dd + i10*s1 + i11*s2 + i12*s3 + b*QK_K;
            dequantize_ਖੰਡ(y, i01*nblk + b, ਵਸਤੂ_ct1, nb);
        });

    GGML_ਅਣਵਰਤਿਆ(dst);
}

ਸਾਂਚਾ <ਪੂਰਨ qk, ਪੂਰਨ qr, dequantize_kernel_f32_t dq>
ਸਥਿਰ ਖਾਲੀ get_ਪੰਕਤੀਆਂ_sycl_f32(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor *src0, ਅਚਲ ggml_tensor *src1,
                              ggml_tensor *dst, ਅਚਲ ਖਾਲੀ *src0_dd,
                              ਅਚਲ int32_t *src1_dd, ਦਸ਼ਮਲਵ *dst_dd,
                              ਕਤਾਰ_ptr stream) {

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (ne00 + 2*SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ - 1) / (2*SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(ne11 * ne12, ne10, ਖੰਡ_ਅੰਕ_x);

    ਅਚਲ size_t s1 = nb1 / ggml_element_ਮਾਪ(dst);
    ਅਚਲ size_t s2 = nb2 / ggml_element_ਮਾਪ(dst);
    ਅਚਲ size_t s3 = nb3 / ggml_element_ਮਾਪ(dst);

    ਅਚਲ size_t s10 = nb10 / ggml_element_ਮਾਪ(src1);
    ਅਚਲ size_t s11 = nb11 / ggml_element_ਮਾਪ(src1);
    ਅਚਲ size_t s12 = nb12 / ggml_element_ਮਾਪ(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                             k_get_ਪੰਕਤੀਆਂ_f32<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, ਵਸਤੂ_ct1);
                         });

    GGML_ਅਣਵਰਤਿਆ(dst);
    GGML_ਅਣਵਰਤਿਆ(ctx);
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ src0_t, ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ get_ਪੰਕਤੀਆਂ_sycl_float(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor *src0,
                                ਅਚਲ ggml_tensor *src1, ggml_tensor *dst,
                                ਅਚਲ src0_t *src0_dd, ਅਚਲ int32_t *src1_dd,
                                dst_t *dst_dd, ਕਤਾਰ_ptr stream) {

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ);
    ਅਚਲ ਪੂਰਨ ਖੰਡ_ਅੰਕ_x = (ne00 + SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ - 1) / SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ;
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(ne11 * ne12, ne10, ਖੰਡ_ਅੰਕ_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    ਅਚਲ size_t s1 = nb1 / ggml_element_ਮਾਪ(dst);
    ਅਚਲ size_t s2 = nb2 / ggml_element_ਮਾਪ(dst);
    ਅਚਲ size_t s3 = nb3 / ggml_element_ਮਾਪ(dst);

    ਅਚਲ size_t s10 = nb10 / ggml_element_ਮਾਪ(src1);
    ਅਚਲ size_t s11 = nb11 / ggml_element_ਮਾਪ(src1);
    ਅਚਲ size_t s12 = nb12 / ggml_element_ਮਾਪ(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
            [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                k_get_ਪੰਕਤੀਆਂ_float(src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, ਵਸਤੂ_ct1);
            });
    }

    GGML_ਅਣਵਰਤਿਆ(dst);
    GGML_ਅਣਵਰਤਿਆ(ctx);
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ src0_t>
ਸਥਿਰ ਖਾਲੀ k_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ_float(ਅਚਲ src0_t * src0, ਅਚਲ int32_t * src1, ਦਸ਼ਮਲਵ * dst,
                                  ਅਚਲ int64_t ncols, ਅਚਲ int64_t nrows_grad_10, ਅਚਲ int64_t nrows_grad_11, ਅਚਲ int64_t nrows_dst,
                                  ਅਚਲ size_t s01, ਅਚਲ size_t s02,
                                  ਅਚਲ size_t s10, ਅਚਲ size_t s11,
                                  ਅਚਲ size_t s1,
                                  ਅਚਲ int64_t ਖੰਡ_ਅੰਕ_y,
                                  ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਅਚਲ int64_t ਖਾਨਾ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2) + ਵਸਤੂ_ct1.get_local_id(2);
    ਜੇ (ਖਾਨਾ >= ncols) {
        ਵਾਪਸ;
    }

    // block_num_y is clamped, so stride over destination rows like CUDA k_get_rows_back_float
    ਲਈ (int64_t dst_ਪੰਕਤੀ = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1); dst_ਪੰਕਤੀ < nrows_dst; dst_ਪੰਕਤੀ += ਖੰਡ_ਅੰਕ_y) {
        ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;

        ਅਚਲ int64_t nrows_grad_ਕੁੱਲ = nrows_grad_10 * nrows_grad_11;
        ਲਈ (int64_t i = 0; i < nrows_grad_ਕੁੱਲ; ++i) {
            ਅਚਲ int64_t i10 = i % nrows_grad_10;
            ਅਚਲ int64_t i11 = i / nrows_grad_10;
            ਜੇ (src1[i10*s10 + i11*s11] != dst_ਪੰਕਤੀ) {
                ਜਾਰੀ;
            }
            ਜੋੜ += (ਦਸ਼ਮਲਵ) src0[ਖਾਨਾ + i10*s01 + i11*s02];
        }

        dst[ਖਾਨਾ + dst_ਪੰਕਤੀ*s1] = ਜੋੜ;
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ src0_t>
ਸਥਿਰ ਖਾਲੀ get_ਪੰਕਤੀਆਂ_ਪਿੱਠ_sycl_float(ggml_backend_sycl_context & ctx, ਅਚਲ ggml_tensor * src0,
                                     ਅਚਲ ggml_tensor * src1, ggml_tensor * dst,
                                     ਅਚਲ src0_t * src0_dd, ਅਚਲ int32_t * src1_dd,
                                     ਦਸ਼ਮਲਵ * dst_dd, ਕਤਾਰ_ptr stream) {

    GGML_TENSOR_ਦੁਆਧਾਰੀ_OP_LOCALS

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);
    GGML_ASSERT(src0->nb[0] == ggml_type_ਮਾਪ(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_ਮਾਪ(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_ਮਾਪ(dst->type));

    ਅਚਲ int64_t ncols = ne00;
    ਅਚਲ int64_t nrows_grad_10 = ne10;
    ਅਚਲ int64_t nrows_grad_11 = ne11;
    ਅਚਲ int64_t nrows_dst = ne1;

    ਅਚਲ size_t s01 = nb01 / ਆਕਾਰ(src0_t);
    ਅਚਲ size_t s02 = nb02 / ਆਕਾਰ(src0_t);

    ਅਚਲ size_t s10 = nb10 / ਆਕਾਰ(int32_t);
    ਅਚਲ size_t s11 = nb11 / ਆਕਾਰ(int32_t);

    ਅਚਲ size_t s1 = nb1 / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਅਚਲ sycl::range<3> ਖੰਡ_dims(1, 1, SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ);
    ਅਚਲ int64_t ਖੰਡ_ਅੰਕ_x = (ncols + SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ - 1) / SYCL_GET_ਪੰਕਤੀਆਂ_ਖੰਡ_ਮਾਪ;
    ਅਚਲ int64_t ਖੰਡ_ਅੰਕ_y = std::min<int64_t>(nrows_dst, (int64_t) UINT16_MAX);
    ਅਚਲ sycl::range<3> ਖੰਡ_nums(1, ਖੰਡ_ਅੰਕ_y, ਖੰਡ_ਅੰਕ_x);

    stream->parallel_for(sycl::nd_range<3>(ਖੰਡ_nums * ਖੰਡ_dims, ਖੰਡ_dims),
                         [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
                             k_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ_float(src0_dd, src1_dd, dst_dd,
                                                   ncols, nrows_grad_10, nrows_grad_11, nrows_dst,
                                                   s01, s02, s10, s11, s1,
                                                   ਖੰਡ_ਅੰਕ_y, ਵਸਤੂ_ct1);
                         });

    GGML_ਅਣਵਰਤਿਆ(src0);
    GGML_ਅਣਵਰਤਿਆ(src1);
    GGML_ਅਣਵਰਤਿਆ(dst);
    GGML_ਅਣਵਰਤਿਆ(ctx);
}

ਖਾਲੀ ggml_sycl_op_get_ਪੰਕਤੀਆਂ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_I32 );

    GGML_ASSERT(dst->src[0]->nb[0] == ggml_type_ਮਾਪ(dst->src[0]->type));
    GGML_ASSERT(dst->src[1]->nb[0] == ggml_type_ਮਾਪ(dst->src[1]->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_ਮਾਪ(dst->type));

    ਅਚਲ int32_t * src1_i32 = (ਅਚਲ int32_t *) dst->src[1]->ਅੰਕੜੇ;
    // a weight shared with MUL_MAT (e.g. tied embeddings) can be in the
    // persistent reorder layout (incl. IQ3_S / IQ3_XXS since the SYCL IQ3
    // reorder work)
    ਅਚਲ ਤਾਰਕਿਕ src0_reordered = dst->src[0]->extra &&
        ((ਅਚਲ ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
    /* TODO: Refactor and remove duplicates */
    ਚੋਣ (dst->src[0]->type) {
        ਹਾਲਤ GGML_TYPE_F16:
            get_ਪੰਕਤੀਆਂ_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ sycl::half *)dst->src[0]->ਅੰਕੜੇ,
                                src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_BF16:
            get_ਪੰਕਤੀਆਂ_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ sycl::ext::oneapi::bfloat16 *)dst->src[0]->ਅੰਕੜੇ,
                                src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            get_ਪੰਕਤੀਆਂ_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_I32:
            get_ਪੰਕਤੀਆਂ_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ int32_t *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (int32_t *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q1_0:
            get_ਪੰਕਤੀਆਂ_sycl<QK1_0, 1, dequantize_q1_0>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_MXFP4:
            get_ਪੰਕਤੀਆਂ_sycl<QK_MXFP4, 2, dequantize_mxfp4>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_NVFP4:
            get_ਪੰਕਤੀਆਂ_sycl<QK_NVFP4, 1, dequantize_nvfp4>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_XXS:
            get_ਪੰਕਤੀਆਂ_sycl<QK_K, 1, dequantize_iq2_xxs>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_XS:
            get_ਪੰਕਤੀਆਂ_sycl<QK_K, 1, dequantize_iq2_xs>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ2_S:
            get_ਪੰਕਤੀਆਂ_sycl<QK_K, 1, dequantize_iq2_s>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ3_XXS:
            ਜੇ (src0_reordered) {
                ਅਚਲ ਖਾਲੀ * vx = dst->src[0]->ਅੰਕੜੇ;
                get_ਪੰਕਤੀਆਂ_sycl_reorder_iq3(ctx, dst->src[0], dst->src[1], dst, src1_i32, (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ,
                    [=](ਦਸ਼ਮਲਵ * y, int64_t iblk, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1, int64_t nb) {
                        dequantize_ਖੰਡ_iq3_xxs_reorder(vx, y, iblk, ਵਸਤੂ_ct1, nb,
                                                         iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs);
                    });
                ਰੋਕੋ;
            }
            get_ਪੰਕਤੀਆਂ_sycl<QK_K, 1, dequantize_iq3_xxs>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ1_S:
            get_ਪੰਕਤੀਆਂ_sycl<QK_K, 1, dequantize_iq1_s>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ1_M:
            get_ਪੰਕਤੀਆਂ_sycl<QK_K, 1, dequantize_iq1_m>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ3_S:
            ਜੇ (src0_reordered) {
                ਅਚਲ ਖਾਲੀ * vx = dst->src[0]->ਅੰਕੜੇ;
                get_ਪੰਕਤੀਆਂ_sycl_reorder_iq3(ctx, dst->src[0], dst->src[1], dst, src1_i32, (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ,
                    [=](ਦਸ਼ਮਲਵ * y, int64_t iblk, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1, int64_t nb) {
                        dequantize_ਖੰਡ_iq3_s_reorder(vx, y, iblk, ਵਸਤੂ_ct1, nb,
                                                       kmask_iq2xs, iq3s_grid);
                    });
                ਰੋਕੋ;
            }
            get_ਪੰਕਤੀਆਂ_sycl<QK_K, 1, dequantize_iq3_s>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ4_NL:
            get_ਪੰਕਤੀਆਂ_sycl<QK4_NL, 1, dequantize_iq4_nl>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_IQ4_XS:
            get_ਪੰਕਤੀਆਂ_sycl<QK_K, 1, dequantize_iq4_xs>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q2_K:
            ਜੇ (src0_reordered) {
                ਅਚਲ ਖਾਲੀ * vx = dst->src[0]->ਅੰਕੜੇ;
                get_ਪੰਕਤੀਆਂ_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream(),
                    [=](ਦਸ਼ਮਲਵ * y, int64_t ib, uint8_t * scales_local, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1, int64_t nb) {
                        dequantize_ਖੰਡ_q2_K_reorder(vx, y, ib, ਵਸਤੂ_ct1, nb);
                        GGML_ਅਣਵਰਤਿਆ(scales_local);
                    });
                ਰੋਕੋ;
            }
            get_ਪੰਕਤੀਆਂ_sycl_f32<QK_K, 1, dequantize_q2_K_f32>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q3_K:
            ਜੇ (src0_reordered) {
                ਅਚਲ ਖਾਲੀ * vx = dst->src[0]->ਅੰਕੜੇ;
                get_ਪੰਕਤੀਆਂ_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream(),
                    [=](ਦਸ਼ਮਲਵ * y, int64_t ib, uint8_t * scales_local, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1, int64_t nb) {
                        dequantize_ਖੰਡ_q3_K_reorder(vx, y, ib, ਵਸਤੂ_ct1, nb);
                        GGML_ਅਣਵਰਤਿਆ(scales_local);
                    });
                ਰੋਕੋ;
            }
            get_ਪੰਕਤੀਆਂ_sycl<QK_K, 1, dequantize_q3_K>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ਜੇ (src0_reordered) {
                get_ਪੰਕਤੀਆਂ_sycl_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(dst->src[0], dst->src[1], dst, dst->src[0]->ਅੰਕੜੇ,
                src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
                ਰੋਕੋ;
            }
            get_ਪੰਕਤੀਆਂ_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_1:
            get_ਪੰਕਤੀਆਂ_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q4_K:
            ਜੇ (src0_reordered) {
                ਅਚਲ ਖਾਲੀ * vx = dst->src[0]->ਅੰਕੜੇ;
                get_ਪੰਕਤੀਆਂ_sycl_reorder_k<1, 32, K_SCALE_ਮਾਪ>(dst->src[0], dst->src[1], dst, src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream(),
                    [=](ਦਸ਼ਮਲਵ * y, int64_t ib, uint8_t * scales_local, ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ_ct1, int64_t nb) {
                        dequantize_ਖੰਡ_q4_K_reorder(vx, y, ib, scales_local, ਵਸਤੂ_ct1, nb);
                    });
                ਰੋਕੋ;
            }
            get_ਪੰਕਤੀਆਂ_sycl_f32<QK_K, 1, dequantize_q4_K_f32>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_0:
            get_ਪੰਕਤੀਆਂ_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_1:
            get_ਪੰਕਤੀਆਂ_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q5_K:
            ਜੇ (src0_reordered) {
                ਅਚਲ ਖਾਲੀ * vx = dst->src[0]->ਅੰਕੜੇ;
                get_ਪੰਕਤੀਆਂ_sycl_reorder_k<3, 64, K_SCALE_ਮਾਪ>(dst->src[0], dst->src[1], dst, src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream(),
                    [=](ਦਸ਼ਮਲਵ * y, int64_t ib, uint8_t * scales_local, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1, int64_t nb) {
                        dequantize_ਖੰਡ_q5_K_reorder(vx, y, ib, scales_local, ਵਸਤੂ_ct1, nb);
                    });
                ਰੋਕੋ;
            }
            get_ਪੰਕਤੀਆਂ_sycl_f32<QK_K, 1, dequantize_q5_K_f32>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q6_K:
            ਜੇ (src0_reordered) {
                ਅਚਲ ਖਾਲੀ * vx = dst->src[0]->ਅੰਕੜੇ;
                get_ਪੰਕਤੀਆਂ_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream(),
                    [=](ਦਸ਼ਮਲਵ * y, int64_t ib, uint8_t * scales_local, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1, int64_t nb) {
                        dequantize_ਖੰਡ_q6_K_reorder(vx, y, ib, ਵਸਤੂ_ct1, nb);
                        GGML_ਅਣਵਰਤਿਆ(scales_local);
                    });
                ਰੋਕੋ;
            }
            get_ਪੰਕਤੀਆਂ_sycl<QK_K, 1, dequantize_q6_K>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਜੇ (src0_reordered) {
                get_ਪੰਕਤੀਆਂ_sycl_reorder<QK8_0, QR8_0, dequantize_q8_0_reorder>(dst->src[0], dst->src[1], dst, dst->src[0]->ਅੰਕੜੇ,
                src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
                ਰੋਕੋ;
            }
            get_ਪੰਕਤੀਆਂ_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, dst->src[0], dst->src[1], dst, (ਅਚਲ ਦਸ਼ਮਲਵ *)dst->src[0]->ਅੰਕੜੇ,
            src1_i32, (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ, ctx.stream());
            ਰੋਕੋ;
        ਮੂਲ:
            // TODO: k-quants
            GGML_LOG_ERROR("%s: unsupported type: %s\n", __func__, ggml_type_name(dst->src[0]->type));
            GGML_ABORT("fatal error");
    }
}

ਖਾਲੀ ggml_sycl_op_get_ਪੰਕਤੀਆਂ_ਪਿੱਠ(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(dst));

    ਚੋਣ (src0->type) {
        ਹਾਲਤ GGML_TYPE_F16:
            get_ਪੰਕਤੀਆਂ_ਪਿੱਠ_sycl_float(ctx, src0, src1, dst, (ਅਚਲ sycl::half *) src0->ਅੰਕੜੇ,
                                     (ਅਚਲ int32_t *) src1->ਅੰਕੜੇ, (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ,
                                     ctx.stream());
            ਰੋਕੋ;
        ਹਾਲਤ GGML_TYPE_F32:
            get_ਪੰਕਤੀਆਂ_ਪਿੱਠ_sycl_float(ctx, src0, src1, dst, (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ,
                                     (ਅਚਲ int32_t *) src1->ਅੰਕੜੇ, (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ,
                                     ctx.stream());
            ਰੋਕੋ;
        ਮੂਲ:
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0->type));
    }
}
