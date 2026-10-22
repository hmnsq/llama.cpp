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
        ثابت صحيح s0, ثابت صحيح output_سائيز,
        ثابت صحيح src0_ne0, ثابت صحيح src0_ne1, ثابت صحيح src0_ne2,
        ثابت صحيح src1_ne0, ثابت صحيح dst_ne0,
        ثابت عائم * src0, ثابت عائم * src1,  عائم * dst,
        ثابت sycl::nd_شي<3> &شي_ct1) {
    صحيح global_اشاريو = شي_ct1.get_local_id(2) +
                       شي_ct1.get_گروهه(2) * شي_ct1.get_local_range(2);
    إذا (global_اشاريو >= output_سائيز) {
        أرجع;
    }

    صحيح ٻاهريون_اشاريو = global_اشاريو / dst_ne0;

    عائم گڏ_ڪندڙ = 0;

    لكل (صحيح c = 0; c < src0_ne2; c++) {
        صحيح idx = global_اشاريو % dst_ne0;

        صحيح kernel_وٿي = (src0_ne0 * src0_ne1 * c) + (ٻاهريون_اشاريو * src0_ne0);
        صحيح داخلا_وٿي = src1_ne0 * c;

        لكل (صحيح i = 0; i < src1_ne0; i++) {
            إذا (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                استمر;
            }
            صحيح وزن_idx = idx - i*s0;

            عائم kernel_وزن = src0[kernel_وٿي + وزن_idx];
            عائم داخلا_قدر =  src1[داخلا_وٿي+i];

            گڏ_ڪندڙ += kernel_وزن * داخلا_قدر;
        }
    }
    dst[global_اشاريو] = گڏ_ڪندڙ;
}

ساكن فراغ conv_transpose_1d_f32_f32_sycl(
    ثابت صحيح s0, ثابت صحيح output_سائيز,
    ثابت صحيح src0_ne0, ثابت صحيح src0_ne1, ثابت صحيح src0_ne2,
    ثابت صحيح src1_ne0, ثابت صحيح dst_ne0,
    ثابت عائم *src0, ثابت عائم *src1, عائم *dst,
    ثابت قطار_ptr& stream) {

    ثابت صحيح انگ_بلاڪون = (output_سائيز + SYCL_CONV_TRANPOSE_1_D_بلاڪ_سائيز - 1) / SYCL_CONV_TRANPOSE_1_D_بلاڪ_سائيز;
    ثابت sycl::range<3> بلاڪ_dims(1, 1, SYCL_CONV_TRANPOSE_1_D_بلاڪ_سائيز);
    ثابت sycl::range<3> بلاڪ_nums(1, 1, انگ_بلاڪون);
    stream->parallel_for(
        sycl::nd_range<3>(
            بلاڪ_nums * بلاڪ_dims, بلاڪ_dims),
        [=](sycl::nd_شي<3> شي_ct1) {
            conv_transpose_1d_kernel(
                s0, output_سائيز,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, شي_ct1);
        });
}

فراغ ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor *src0 = dst->src[0];
    ثابت ggml_tensor *src1 = dst->src[1];
    ثابت عائم * src0_d = (ثابت عائم *)src0->معلومات;
    ثابت عائم * src1_d = (ثابت عائم *)src1->معلومات;

    عائم * dst_d = (عائم *)dst->معلومات;
    dpct::قطار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;

    ثابت صحيح s0 = opts[0];

    ثابت int64_t output_سائيز = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_سائيز,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

