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

#شامل "ggml-impl.h"
#شامل "common.hpp"
#شامل "dequantize.hpp"
#شامل "getrows.hpp"


سانچہ<صحیح qk, صحیح qr, dequantize_kernel_t dequantize_kernel, قسم_نام dst_t>
ساکن خالی k_get_صفیں(
            ثابت خالی * src0, ثابت int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            ثابت sycl::nd_شے<3> &شے_ct1/*, size_t s13*/) {

    ثابت صحیح i00 = (شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2) +
                     شے_ct1.get_local_id(2)) *
                    2;
    ثابت صحیح i10 = شے_ct1.get_local_range(1) * شے_ct1.get_گروہ(1) +
                    شے_ct1.get_local_id(1);
    ثابت صحیح i11 = (شے_ct1.get_گروہ(0) * شے_ct1.get_local_range(0) +
                     شے_ct1.get_local_id(0)) /
                    ne12;
    ثابت صحیح i12 = (شے_ct1.get_گروہ(0) * شے_ct1.get_local_range(0) +
                     شے_ct1.get_local_id(0)) %
                    ne12;

    اگر (i00 >= ne00) {
        واپس;
    }

    ثابت صحیح i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_صف = dst + i10*s1 + i11*s2 + i12*s3;
    ثابت خالی * src0_صف = (ثابت حرف *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    ثابت صحیح ib = i00/qk; // block index
    ثابت صحیح iqs = (i00%qk)/qr; // quant index
    ثابت صحیح iybs = i00 - i00%qk; // dst block start index
    ثابت صحیح y_انحراف = qr == 1 ? 1 : qk/2;

    // dequantize
    dfloat2 v;
    dequantize_kernel(src0_صف, ib, iqs, v);

    dst_صف[iybs + iqs + 0] = v.x();
    dst_صف[iybs + iqs + y_انحراف] = v.y();
}

سانچہ<صحیح qk, صحیح qr, dequantize_kernel_f32_t dequantize_kernel, قسم_نام dst_t>
ساکن خالی k_get_صفیں_f32(
            ثابت خالی * src0, ثابت int32_t * src1, dst_t * dst,
            int64_t ne00,
            int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت صحیح i00 = (شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2) +
                     شے_ct1.get_local_id(2)) *
                    2;
    ثابت صحیح i10 = شے_ct1.get_local_range(1) * شے_ct1.get_گروہ(1) +
                    شے_ct1.get_local_id(1);
    ثابت صحیح i11 = (شے_ct1.get_گروہ(0) * شے_ct1.get_local_range(0) +
                     شے_ct1.get_local_id(0)) /
                    ne12;
    ثابت صحیح i12 = (شے_ct1.get_گروہ(0) * شے_ct1.get_local_range(0) +
                     شے_ct1.get_local_id(0)) %
                    ne12;

    اگر (i00 >= ne00) {
        واپس;
    }

    ثابت صحیح i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_صف = dst + i10*s1 + i11*s2 + i12*s3;
    ثابت خالی * src0_صف = (ثابت حرف *)src0 + i01*nb01 + i11*nb02 + i12*nb03;

    ثابت صحیح ib = i00/qk;
    ثابت صحیح iqs = (i00%qk)/qr;
    ثابت صحیح iybs = i00 - i00%qk;
    ثابت صحیح y_انحراف = qr == 1 ? 1 : qk/2;

    اعشاری v0;
    اعشاری v1;
    dequantize_kernel(src0_صف, ib, iqs, v0, v1);

    dst_صف[iybs + iqs + 0] = (dst_t) v0;
    dst_صف[iybs + iqs + y_انحراف] = (dst_t) v1;
}

سانچہ<قسم_نام src0_t, قسم_نام dst_t>
ساکن خالی k_get_صفیں_float(
            ثابت src0_t * src0, ثابت int32_t * src1, dst_t * dst,
            int64_t ne00, /*int64_t ne01, int64_t ne02, int64_t ne03,*/
            /*int64_t ne10, int64_t ne11,*/ int64_t ne12, /*int64_t ne13,*/
            /*size_t s0,*/ size_t s1, size_t s2, size_t s3,
            /*size_t nb00,*/ size_t nb01, size_t nb02, size_t nb03,
            size_t s10, size_t s11, size_t s12,
            ثابت sycl::nd_شے<3> &شے_ct1/*, size_t s13*/) {

    ثابت صحیح i00 = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2) +
                    شے_ct1.get_local_id(2);
    ثابت صحیح i10 = شے_ct1.get_local_range(1) * شے_ct1.get_گروہ(1) +
                    شے_ct1.get_local_id(1);
    ثابت صحیح i11 = (شے_ct1.get_گروہ(0) * شے_ct1.get_local_range(0) +
                     شے_ct1.get_local_id(0)) /
                    ne12;
    ثابت صحیح i12 = (شے_ct1.get_گروہ(0) * شے_ct1.get_local_range(0) +
                     شے_ct1.get_local_id(0)) %
                    ne12;

    اگر (i00 >= ne00) {
        واپس;
    }

    ثابت صحیح i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_صف = dst + i10*s1 + i11*s2 + i12*s3;
    ثابت src0_t * src0_صف = (ثابت src0_t *)((ثابت حرف *)src0 + i01*nb01 + i11*nb02 + i12*nb03);

    dst_صف[i00] = src0_صف[i00];
}

// src0 is in reorder layout: all qs of the tensor, then all d
سانچہ<صحیح qk, صحیح qr, dequantize_kernel_t_reorder dequantize_kernel, قسم_نام dst_t>
ساکن خالی k_get_صفیں_reorder(
            ثابت خالی * src0_qs, ثابت خالی * src0_d, ثابت int32_t * src1, dst_t * dst,
            int64_t ne00, int64_t ne12,
            size_t s1, size_t s2, size_t s3,
            size_t s10, size_t s11, size_t s12,
            ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت صحیح i00 = (شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2) +
                     شے_ct1.get_local_id(2)) *
                    2;
    ثابت صحیح i10 = شے_ct1.get_local_range(1) * شے_ct1.get_گروہ(1) +
                    شے_ct1.get_local_id(1);
    ثابت صحیح i11 = (شے_ct1.get_گروہ(0) * شے_ct1.get_local_range(0) +
                     شے_ct1.get_local_id(0)) /
                    ne12;
    ثابت صحیح i12 = (شے_ct1.get_گروہ(0) * شے_ct1.get_local_range(0) +
                     شے_ct1.get_local_id(0)) %
                    ne12;

    اگر (i00 >= ne00) {
        واپس;
    }

    ثابت صحیح i01 = src1[i10*s10 + i11*s11 + i12*s12];

    dst_t * dst_صف = dst + i10*s1 + i11*s2 + i12*s3;

    ثابت int64_t ib = (int64_t) i01*(ne00/qk) + i00/qk; // block index in the whole tensor
    ثابت صحیح iqs = (i00%qk)/qr;
    ثابت صحیح iybs = i00 - i00%qk;
    ثابت صحیح y_انحراف = qr == 1 ? 1 : qk/2;

    dfloat2 v;
    dequantize_kernel(src0_d, ib, src0_qs, ib*(qk/qr) + iqs, v);

    dst_صف[iybs + iqs + 0] = v.x();
    dst_صف[iybs + iqs + y_انحراف] = v.y();
}

سانچہ <صحیح qk, صحیح qr, dequantize_kernel_t dq>
ساکن خالی get_صفیں_sycl(ggml_backend_sycl_context & ctx, ثابت ggml_tensor *src0, ثابت ggml_tensor *src1,
                          ggml_tensor *dst, ثابت خالی *src0_dd,
                          ثابت int32_t *src1_dd, اعشاری *dst_dd,
                          قطار_ptr stream) {

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_GET_صفیں_بلاک_سائز);
    ثابت صحیح بلاک_عدد_x = (ne00 + 2*SYCL_GET_صفیں_بلاک_سائز - 1) / (2*SYCL_GET_صفیں_بلاک_سائز);
    ثابت sycl::range<3> بلاک_nums(ne11 * ne12, ne10, بلاک_عدد_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    ثابت size_t s1 = nb1 / ggml_element_سائز(dst);
    ثابت size_t s2 = nb2 / ggml_element_سائز(dst);
    ثابت size_t s3 = nb3 / ggml_element_سائز(dst);

    ثابت size_t s10 = nb10 / ggml_element_سائز(src1);
    ثابت size_t s11 = nb11 / ggml_element_سائز(src1);
    ثابت size_t s12 = nb12 / ggml_element_سائز(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> شے_ct1) {
                             k_get_صفیں<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, شے_ct1);
                         });

    GGML_غیر_مستعمل(dst);
    GGML_غیر_مستعمل(ctx);
}

سانچہ <صحیح qk, صحیح qr, dequantize_kernel_t_reorder dq>
ساکن خالی get_صفیں_sycl_reorder(ثابت ggml_tensor *src0, ثابت ggml_tensor *src1,
                                  ggml_tensor *dst, ثابت خالی *src0_dd,
                                  ثابت int32_t *src1_dd, اعشاری *dst_dd,
                                  قطار_ptr stream) {

    GGML_TENSOR_ثنائی_OP_LOCALS

    // reorder is done only on 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % 2 == 0);

    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_GET_صفیں_بلاک_سائز);
    ثابت صحیح بلاک_عدد_x = (ne00 + 2*SYCL_GET_صفیں_بلاک_سائز - 1) / (2*SYCL_GET_صفیں_بلاک_سائز);
    ثابت sycl::range<3> بلاک_nums(ne11 * ne12, ne10, بلاک_عدد_x);

    ثابت size_t s1 = nb1 / ggml_element_سائز(dst);
    ثابت size_t s2 = nb2 / ggml_element_سائز(dst);
    ثابت size_t s3 = nb3 / ggml_element_سائز(dst);

    ثابت size_t s10 = nb10 / ggml_element_سائز(src1);
    ثابت size_t s11 = nb11 / ggml_element_سائز(src1);
    ثابت size_t s12 = nb12 / ggml_element_سائز(src1);

    ثابت خالی * src0_d = (ثابت حرف *) src0_dd + ne00*ne01/qr;

    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> شے_ct1) {
                             k_get_صفیں_reorder<qk, qr, dq>(
                                 src0_dd, src0_d, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, s10, s11, s12, شے_ct1);
                         });

    GGML_غیر_مستعمل(src0);
}

// K-quants in reorder layout: one work-group dequantizes one QK_K block of one row
سانچہ <صحیح dims, صحیح nthreads, صحیح nlocal, قسم_نام F>
ساکن خالی get_صفیں_sycl_reorder_k(ثابت ggml_tensor *src0, ثابت ggml_tensor *src1,
                                    ggml_tensor *dst, ثابت int32_t *src1_dd, اعشاری *dst_dd,
                                    قطار_ptr stream, F dequantize_بلاک) {

    GGML_TENSOR_ثنائی_OP_LOCALS

    // reorder is done only on 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % QK_K == 0);

    ثابت int64_t nblk = ne00 / QK_K;
    ثابت int64_t nb = nblk * ne01; // blocks in the whole tensor
    ثابت int64_t ngroups = ne10 * ne11 * ne12 * nblk;

    ثابت size_t s1 = nb1 / ggml_element_سائز(dst);
    ثابت size_t s2 = nb2 / ggml_element_سائز(dst);
    ثابت size_t s3 = nb3 / ggml_element_سائز(dst);

    ثابت size_t s10 = nb10 / ggml_element_سائز(src1);
    ثابت size_t s11 = nb11 / ggml_element_سائز(src1);
    ثابت size_t s12 = nb12 / ggml_element_سائز(src1);

    stream->submit([&](sycl::handler & cgh) {
        sycl::local_accessor<uint8_t, 1> scales_local(sycl::range<1>(nlocal), cgh);

        خودکار kernel = [=](sycl::nd_شے<dims> شے_ct1) {
            ثابت int64_t g   = شے_ct1.get_گروہ(dims - 1);
            ثابت int64_t b   = g % nblk;
            ثابت int64_t r   = g / nblk;
            ثابت int64_t i10 = r % ne10;
            ثابت int64_t i11 = (r / ne10) % ne11;
            ثابت int64_t i12 = r / (ne10 * ne11);

            ثابت int64_t i01 = src1_dd[i10*s10 + i11*s11 + i12*s12];

            اعشاری * y = dst_dd + i10*s1 + i11*s2 + i12*s3 + b*QK_K;
            dequantize_بلاک(y, i01*nblk + b, get_اشارہ_گر(scales_local), شے_ct1, nb);
        };

        اگر constexpr (dims == 1) {
            cgh.parallel_for(sycl::nd_range<1>(sycl::range<1>(ngroups * nthreads), sycl::range<1>(nthreads)), kernel);
        } ورنہ {
            cgh.parallel_for(sycl::nd_range<3>(sycl::range<3>(1, 1, ngroups * nthreads), sycl::range<3>(1, 1, nthreads)), kernel);
        }
    });

    GGML_غیر_مستعمل(src0);
}

// IQ3_S / IQ3_XXS in the persistent reorder layout: one work-group of 32 threads
// dequantizes one QK_K block of one requested row, reusing the reorder-aware
// block dequantizers that the dequantize-based mul_mat path already uses.
سانچہ <قسم_نام F>
ساکن خالی get_صفیں_sycl_reorder_iq3(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0,
                                      ثابت ggml_tensor * src1, ggml_tensor * dst,
                                      ثابت int32_t * src1_dd, اعشاری * dst_dd,
                                      F dequantize_بلاک) {
    GGML_TENSOR_ثنائی_OP_LOCALS

    // the persistent reorder is only installed for 2D weights
    GGML_ASSERT(ne02 == 1 && ne03 == 1);
    GGML_ASSERT(ne00 % QK_K == 0);

    ثابت int64_t nblk    = ne00 / QK_K;
    ثابت int64_t nb      = nblk * ne01; // blocks in the whole tensor
    ثابت int64_t ngroups = ne10 * ne11 * ne12 * nblk;

    ثابت size_t s1  = nb1 / ggml_element_سائز(dst);
    ثابت size_t s2  = nb2 / ggml_element_سائز(dst);
    ثابت size_t s3  = nb3 / ggml_element_سائز(dst);

    ثابت size_t s10 = nb10 / ggml_element_سائز(src1);
    ثابت size_t s11 = nb11 / ggml_element_سائز(src1);
    ثابت size_t s12 = nb12 / ggml_element_سائز(src1);

    قطار_ptr stream = ctx.stream();
    stream->parallel_for(
        sycl::nd_range<3>(sycl::range<3>(1, 1, ngroups * 32), sycl::range<3>(1, 1, 32)),
        [=](sycl::nd_شے<3> شے_ct1) {
            ثابت int64_t g   = شے_ct1.get_گروہ(2);
            ثابت int64_t b   = g % nblk;
            ثابت int64_t r   = g / nblk;
            ثابت int64_t i10 = r % ne10;
            ثابت int64_t i11 = (r / ne10) % ne11;
            ثابت int64_t i12 = r / (ne10 * ne11);

            ثابت int64_t i01 = src1_dd[i10*s10 + i11*s11 + i12*s12];

            اعشاری * y = dst_dd + i10*s1 + i11*s2 + i12*s3 + b*QK_K;
            dequantize_بلاک(y, i01*nblk + b, شے_ct1, nb);
        });

    GGML_غیر_مستعمل(dst);
}

سانچہ <صحیح qk, صحیح qr, dequantize_kernel_f32_t dq>
ساکن خالی get_صفیں_sycl_f32(ggml_backend_sycl_context & ctx, ثابت ggml_tensor *src0, ثابت ggml_tensor *src1,
                              ggml_tensor *dst, ثابت خالی *src0_dd,
                              ثابت int32_t *src1_dd, اعشاری *dst_dd,
                              قطار_ptr stream) {

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_GET_صفیں_بلاک_سائز);
    ثابت صحیح بلاک_عدد_x = (ne00 + 2*SYCL_GET_صفیں_بلاک_سائز - 1) / (2*SYCL_GET_صفیں_بلاک_سائز);
    ثابت sycl::range<3> بلاک_nums(ne11 * ne12, ne10, بلاک_عدد_x);

    ثابت size_t s1 = nb1 / ggml_element_سائز(dst);
    ثابت size_t s2 = nb2 / ggml_element_سائز(dst);
    ثابت size_t s3 = nb3 / ggml_element_سائز(dst);

    ثابت size_t s10 = nb10 / ggml_element_سائز(src1);
    ثابت size_t s11 = nb11 / ggml_element_سائز(src1);
    ثابت size_t s12 = nb12 / ggml_element_سائز(src1);

    GGML_ASSERT(ne00 % 2 == 0);

    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> شے_ct1) {
                             k_get_صفیں_f32<qk, qr, dq>(
                                 src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, شے_ct1);
                         });

    GGML_غیر_مستعمل(dst);
    GGML_غیر_مستعمل(ctx);
}

سانچہ <قسم_نام src0_t, قسم_نام dst_t>
ساکن خالی get_صفیں_sycl_float(ggml_backend_sycl_context & ctx, ثابت ggml_tensor *src0,
                                ثابت ggml_tensor *src1, ggml_tensor *dst,
                                ثابت src0_t *src0_dd, ثابت int32_t *src1_dd,
                                dst_t *dst_dd, قطار_ptr stream) {

    GGML_TENSOR_ثنائی_OP_LOCALS

    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_GET_صفیں_بلاک_سائز);
    ثابت صحیح بلاک_عدد_x = (ne00 + SYCL_GET_صفیں_بلاک_سائز - 1) / SYCL_GET_صفیں_بلاک_سائز;
    ثابت sycl::range<3> بلاک_nums(ne11 * ne12, ne10, بلاک_عدد_x);

    // strides in elements
    //const size_t s0 = nb0 / ggml_element_size(dst);
    ثابت size_t s1 = nb1 / ggml_element_سائز(dst);
    ثابت size_t s2 = nb2 / ggml_element_سائز(dst);
    ثابت size_t s3 = nb3 / ggml_element_سائز(dst);

    ثابت size_t s10 = nb10 / ggml_element_سائز(src1);
    ثابت size_t s11 = nb11 / ggml_element_سائز(src1);
    ثابت size_t s12 = nb12 / ggml_element_سائز(src1);
    //const size_t s13 = nb13 / ggml_element_size(src1);

    {
        dpct::has_capability_or_fail(stream->get_device(),
                                     {sycl::aspect::fp16});

        stream->parallel_for(
            sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
            [=](sycl::nd_شے<3> شے_ct1) {
                k_get_صفیں_float(src0_dd, src1_dd, dst_dd, ne00, ne12, s1, s2,
                                 s3, nb01, nb02, nb03, s10, s11, s12, شے_ct1);
            });
    }

    GGML_غیر_مستعمل(dst);
    GGML_غیر_مستعمل(ctx);
}

سانچہ <قسم_نام src0_t>
ساکن خالی k_get_صفیں_عقب_float(ثابت src0_t * src0, ثابت int32_t * src1, اعشاری * dst,
                                  ثابت int64_t ncols, ثابت int64_t nrows_grad_10, ثابت int64_t nrows_grad_11, ثابت int64_t nrows_dst,
                                  ثابت size_t s01, ثابت size_t s02,
                                  ثابت size_t s10, ثابت size_t s11,
                                  ثابت size_t s1,
                                  ثابت int64_t بلاک_عدد_y,
                                  ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت int64_t ستون = شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2) + شے_ct1.get_local_id(2);
    اگر (ستون >= ncols) {
        واپس;
    }

    // block_num_y is clamped, so stride over destination rows like CUDA k_get_rows_back_float
    برائے (int64_t dst_صف = شے_ct1.get_گروہ(1); dst_صف < nrows_dst; dst_صف += بلاک_عدد_y) {
        اعشاری مجموع = 0.0f;

        ثابت int64_t nrows_grad_کل = nrows_grad_10 * nrows_grad_11;
        برائے (int64_t i = 0; i < nrows_grad_کل; ++i) {
            ثابت int64_t i10 = i % nrows_grad_10;
            ثابت int64_t i11 = i / nrows_grad_10;
            اگر (src1[i10*s10 + i11*s11] != dst_صف) {
                جاری;
            }
            مجموع += (اعشاری) src0[ستون + i10*s01 + i11*s02];
        }

        dst[ستون + dst_صف*s1] = مجموع;
    }
}

سانچہ <قسم_نام src0_t>
ساکن خالی get_صفیں_عقب_sycl_float(ggml_backend_sycl_context & ctx, ثابت ggml_tensor * src0,
                                     ثابت ggml_tensor * src1, ggml_tensor * dst,
                                     ثابت src0_t * src0_dd, ثابت int32_t * src1_dd,
                                     اعشاری * dst_dd, قطار_ptr stream) {

    GGML_TENSOR_ثنائی_OP_LOCALS

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);
    GGML_ASSERT(src0->nb[0] == ggml_type_سائز(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_سائز(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_سائز(dst->type));

    ثابت int64_t ncols = ne00;
    ثابت int64_t nrows_grad_10 = ne10;
    ثابت int64_t nrows_grad_11 = ne11;
    ثابت int64_t nrows_dst = ne1;

    ثابت size_t s01 = nb01 / حجم(src0_t);
    ثابت size_t s02 = nb02 / حجم(src0_t);

    ثابت size_t s10 = nb10 / حجم(int32_t);
    ثابت size_t s11 = nb11 / حجم(int32_t);

    ثابت size_t s1 = nb1 / حجم(اعشاری);

    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_GET_صفیں_بلاک_سائز);
    ثابت int64_t بلاک_عدد_x = (ncols + SYCL_GET_صفیں_بلاک_سائز - 1) / SYCL_GET_صفیں_بلاک_سائز;
    ثابت int64_t بلاک_عدد_y = std::min<int64_t>(nrows_dst, (int64_t) UINT16_MAX);
    ثابت sycl::range<3> بلاک_nums(1, بلاک_عدد_y, بلاک_عدد_x);

    stream->parallel_for(sycl::nd_range<3>(بلاک_nums * بلاک_dims, بلاک_dims),
                         [=](sycl::nd_شے<3> شے_ct1) {
                             k_get_صفیں_عقب_float(src0_dd, src1_dd, dst_dd,
                                                   ncols, nrows_grad_10, nrows_grad_11, nrows_dst,
                                                   s01, s02, s10, s11, s1,
                                                   بلاک_عدد_y, شے_ct1);
                         });

    GGML_غیر_مستعمل(src0);
    GGML_غیر_مستعمل(src1);
    GGML_غیر_مستعمل(dst);
    GGML_غیر_مستعمل(ctx);
}

خالی ggml_sycl_op_get_صفیں(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    GGML_ASSERT(dst->src[1]->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32 || dst->type == GGML_TYPE_I32 );

    GGML_ASSERT(dst->src[0]->nb[0] == ggml_type_سائز(dst->src[0]->type));
    GGML_ASSERT(dst->src[1]->nb[0] == ggml_type_سائز(dst->src[1]->type));
    GGML_ASSERT(dst->nb[0] == ggml_type_سائز(dst->type));

    ثابت int32_t * src1_i32 = (ثابت int32_t *) dst->src[1]->ڈیٹا;
    // a weight shared with MUL_MAT (e.g. tied embeddings) can be in the
    // persistent reorder layout (incl. IQ3_S / IQ3_XXS since the SYCL IQ3
    // reorder work)
    ثابت منطقی src0_reordered = dst->src[0]->extra &&
        ((ثابت ggml_tensor_extra_gpu *) dst->src[0]->extra)->optimized_feature.reorder;
    /* TODO: Refactor and remove duplicates */
    انتخاب (dst->src[0]->type) {
        حالت GGML_TYPE_F16:
            get_صفیں_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ثابت sycl::half *)dst->src[0]->ڈیٹا,
                                src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_BF16:
            get_صفیں_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ثابت sycl::ext::oneapi::bfloat16 *)dst->src[0]->ڈیٹا,
                                src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_F32:
            get_صفیں_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_I32:
            get_صفیں_sycl_float(ctx, dst->src[0], dst->src[1], dst, (ثابت int32_t *)dst->src[0]->ڈیٹا,
            src1_i32, (int32_t *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_Q1_0:
            get_صفیں_sycl<QK1_0, 1, dequantize_q1_0>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_MXFP4:
            get_صفیں_sycl<QK_MXFP4, 2, dequantize_mxfp4>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_NVFP4:
            get_صفیں_sycl<QK_NVFP4, 1, dequantize_nvfp4>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_IQ2_XXS:
            get_صفیں_sycl<QK_K, 1, dequantize_iq2_xxs>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_IQ2_XS:
            get_صفیں_sycl<QK_K, 1, dequantize_iq2_xs>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_IQ2_S:
            get_صفیں_sycl<QK_K, 1, dequantize_iq2_s>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_IQ3_XXS:
            اگر (src0_reordered) {
                ثابت خالی * vx = dst->src[0]->ڈیٹا;
                get_صفیں_sycl_reorder_iq3(ctx, dst->src[0], dst->src[1], dst, src1_i32, (اعشاری *) dst->ڈیٹا,
                    [=](اعشاری * y, int64_t iblk, ثابت sycl::nd_شے<3> & شے_ct1, int64_t nb) {
                        dequantize_بلاک_iq3_xxs_reorder(vx, y, iblk, شے_ct1, nb,
                                                         iq3xxs_grid, ksigns_iq2xs, kmask_iq2xs);
                    });
                توڑو;
            }
            get_صفیں_sycl<QK_K, 1, dequantize_iq3_xxs>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_IQ1_S:
            get_صفیں_sycl<QK_K, 1, dequantize_iq1_s>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_IQ1_M:
            get_صفیں_sycl<QK_K, 1, dequantize_iq1_m>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_IQ3_S:
            اگر (src0_reordered) {
                ثابت خالی * vx = dst->src[0]->ڈیٹا;
                get_صفیں_sycl_reorder_iq3(ctx, dst->src[0], dst->src[1], dst, src1_i32, (اعشاری *) dst->ڈیٹا,
                    [=](اعشاری * y, int64_t iblk, ثابت sycl::nd_شے<3> & شے_ct1, int64_t nb) {
                        dequantize_بلاک_iq3_s_reorder(vx, y, iblk, شے_ct1, nb,
                                                       kmask_iq2xs, iq3s_grid);
                    });
                توڑو;
            }
            get_صفیں_sycl<QK_K, 1, dequantize_iq3_s>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_IQ4_NL:
            get_صفیں_sycl<QK4_NL, 1, dequantize_iq4_nl>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_IQ4_XS:
            get_صفیں_sycl<QK_K, 1, dequantize_iq4_xs>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_Q2_K:
            اگر (src0_reordered) {
                ثابت خالی * vx = dst->src[0]->ڈیٹا;
                get_صفیں_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream(),
                    [=](اعشاری * y, int64_t ib, uint8_t * scales_local, ثابت sycl::nd_شے<3> & شے_ct1, int64_t nb) {
                        dequantize_بلاک_q2_K_reorder(vx, y, ib, شے_ct1, nb);
                        GGML_غیر_مستعمل(scales_local);
                    });
                توڑو;
            }
            get_صفیں_sycl_f32<QK_K, 1, dequantize_q2_K_f32>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_Q3_K:
            اگر (src0_reordered) {
                ثابت خالی * vx = dst->src[0]->ڈیٹا;
                get_صفیں_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream(),
                    [=](اعشاری * y, int64_t ib, uint8_t * scales_local, ثابت sycl::nd_شے<3> & شے_ct1, int64_t nb) {
                        dequantize_بلاک_q3_K_reorder(vx, y, ib, شے_ct1, nb);
                        GGML_غیر_مستعمل(scales_local);
                    });
                توڑو;
            }
            get_صفیں_sycl<QK_K, 1, dequantize_q3_K>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_Q4_0:
            اگر (src0_reordered) {
                get_صفیں_sycl_reorder<QK4_0, QR4_0, dequantize_q4_0_reorder>(dst->src[0], dst->src[1], dst, dst->src[0]->ڈیٹا,
                src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
                توڑو;
            }
            get_صفیں_sycl<QK4_0, QR4_0, dequantize_q4_0>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_Q4_1:
            get_صفیں_sycl<QK4_1, QR4_1, dequantize_q4_1>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_Q4_K:
            اگر (src0_reordered) {
                ثابت خالی * vx = dst->src[0]->ڈیٹا;
                get_صفیں_sycl_reorder_k<1, 32, K_SCALE_سائز>(dst->src[0], dst->src[1], dst, src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream(),
                    [=](اعشاری * y, int64_t ib, uint8_t * scales_local, ثابت sycl::nd_شے<1> & شے_ct1, int64_t nb) {
                        dequantize_بلاک_q4_K_reorder(vx, y, ib, scales_local, شے_ct1, nb);
                    });
                توڑو;
            }
            get_صفیں_sycl_f32<QK_K, 1, dequantize_q4_K_f32>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_Q5_0:
            get_صفیں_sycl<QK5_0, QR5_0, dequantize_q5_0>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_Q5_1:
            get_صفیں_sycl<QK5_1, QR5_1, dequantize_q5_1>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_Q5_K:
            اگر (src0_reordered) {
                ثابت خالی * vx = dst->src[0]->ڈیٹا;
                get_صفیں_sycl_reorder_k<3, 64, K_SCALE_سائز>(dst->src[0], dst->src[1], dst, src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream(),
                    [=](اعشاری * y, int64_t ib, uint8_t * scales_local, ثابت sycl::nd_شے<3> & شے_ct1, int64_t nb) {
                        dequantize_بلاک_q5_K_reorder(vx, y, ib, scales_local, شے_ct1, nb);
                    });
                توڑو;
            }
            get_صفیں_sycl_f32<QK_K, 1, dequantize_q5_K_f32>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_Q6_K:
            اگر (src0_reordered) {
                ثابت خالی * vx = dst->src[0]->ڈیٹا;
                get_صفیں_sycl_reorder_k<3, 64, 1>(dst->src[0], dst->src[1], dst, src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream(),
                    [=](اعشاری * y, int64_t ib, uint8_t * scales_local, ثابت sycl::nd_شے<3> & شے_ct1, int64_t nb) {
                        dequantize_بلاک_q6_K_reorder(vx, y, ib, شے_ct1, nb);
                        GGML_غیر_مستعمل(scales_local);
                    });
                توڑو;
            }
            get_صفیں_sycl<QK_K, 1, dequantize_q6_K>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        حالت GGML_TYPE_Q8_0:
            اگر (src0_reordered) {
                get_صفیں_sycl_reorder<QK8_0, QR8_0, dequantize_q8_0_reorder>(dst->src[0], dst->src[1], dst, dst->src[0]->ڈیٹا,
                src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
                توڑو;
            }
            get_صفیں_sycl<QK8_0, QR8_0, dequantize_q8_0>(ctx, dst->src[0], dst->src[1], dst, (ثابت اعشاری *)dst->src[0]->ڈیٹا,
            src1_i32, (اعشاری *)dst->ڈیٹا, ctx.stream());
            توڑو;
        طےشدہ:
            // TODO: k-quants
            GGML_LOG_ERROR("%s: unsupported type: %s\n", __func__, ggml_type_name(dst->src[0]->type));
            GGML_ABORT("fatal error");
    }
}

خالی ggml_sycl_op_get_صفیں_عقب(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت ggml_tensor * src1 = dst->src[1];

    GGML_ASSERT(src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(dst));

    انتخاب (src0->type) {
        حالت GGML_TYPE_F16:
            get_صفیں_عقب_sycl_float(ctx, src0, src1, dst, (ثابت sycl::half *) src0->ڈیٹا,
                                     (ثابت int32_t *) src1->ڈیٹا, (اعشاری *) dst->ڈیٹا,
                                     ctx.stream());
            توڑو;
        حالت GGML_TYPE_F32:
            get_صفیں_عقب_sycl_float(ctx, src0, src1, dst, (ثابت اعشاری *) src0->ڈیٹا,
                                     (ثابت int32_t *) src1->ڈیٹا, (اعشاری *) dst->ڈیٹا,
                                     ctx.stream());
            توڑو;
        طےشدہ:
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0->type));
    }
}
