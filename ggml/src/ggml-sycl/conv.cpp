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

#ضمن "conv.hpp"

ساكن  فراغ conv_transpose_1d_kernel(
        ثابت صحيح s0, ثابت صحيح output_اندازه,
        ثابت صحيح src0_ne0, ثابت صحيح src0_ne1, ثابت صحيح src0_ne2,
        ثابت صحيح src1_ne0, ثابت صحيح dst_ne0,
        ثابت عائم * src0, ثابت عائم * src1,  عائم * dst,
        ثابت sycl::nd_توکی<3> &توکی_ct1) {
    صحيح global_شاخص = توکی_ct1.get_local_id(2) +
                       توکی_ct1.get_ډله(2) * توکی_ct1.get_local_range(2);
    إذا (global_شاخص >= output_اندازه) {
        أرجع;
    }

    صحيح بهر_شاخص = global_شاخص / dst_ne0;

    عائم راټولونکی = 0;

    لكل (صحيح c = 0; c < src0_ne2; c++) {
        صحيح idx = global_شاخص % dst_ne0;

        صحيح kernel_دننه_واټن = (src0_ne0 * src0_ne1 * c) + (بهر_شاخص * src0_ne0);
        صحيح ننوت_دننه_واټن = src1_ne0 * c;

        لكل (صحيح i = 0; i < src1_ne0; i++) {
            إذا (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                استمر;
            }
            صحيح وزن_idx = idx - i*s0;

            عائم kernel_وزن = src0[kernel_دننه_واټن + وزن_idx];
            عائم ننوت_ارزښت =  src1[ننوت_دننه_واټن+i];

            راټولونکی += kernel_وزن * ننوت_ارزښت;
        }
    }
    dst[global_شاخص] = راټولونکی;
}

ساكن فراغ conv_transpose_1d_f32_f32_sycl(
    ثابت صحيح s0, ثابت صحيح output_اندازه,
    ثابت صحيح src0_ne0, ثابت صحيح src0_ne1, ثابت صحيح src0_ne2,
    ثابت صحيح src1_ne0, ثابت صحيح dst_ne0,
    ثابت عائم *src0, ثابت عائم *src1, عائم *dst,
    ثابت کتار_ptr& stream) {

    ثابت صحيح عدد_بلاکونه = (output_اندازه + SYCL_CONV_TRANPOSE_1_D_بلاک_اندازه - 1) / SYCL_CONV_TRANPOSE_1_D_بلاک_اندازه;
    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_CONV_TRANPOSE_1_D_بلاک_اندازه);
    ثابت sycl::range<3> بلاک_nums(1, 1, عدد_بلاکونه);
    stream->parallel_for(
        sycl::nd_range<3>(
            بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_توکی<3> توکی_ct1) {
            conv_transpose_1d_kernel(
                s0, output_اندازه,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, توکی_ct1);
        });
}

فراغ ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor *src0 = dst->src[0];
    ثابت ggml_tensor *src1 = dst->src[1];
    ثابت عائم * src0_d = (ثابت عائم *)src0->معلومات;
    ثابت عائم * src1_d = (ثابت عائم *)src1->معلومات;

    عائم * dst_d = (عائم *)dst->معلومات;
    dpct::کتار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;

    ثابت صحيح s0 = opts[0];

    ثابت int64_t output_اندازه = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_اندازه,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

