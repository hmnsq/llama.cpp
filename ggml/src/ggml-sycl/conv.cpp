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

#شامل "conv.hpp"

ساکن  خالی conv_transpose_1d_kernel(
        ثابت صحیح s0, ثابت صحیح output_سائز,
        ثابت صحیح src0_ne0, ثابت صحیح src0_ne1, ثابت صحیح src0_ne2,
        ثابت صحیح src1_ne0, ثابت صحیح dst_ne0,
        ثابت اعشاری * src0, ثابت اعشاری * src1,  اعشاری * dst,
        ثابت sycl::nd_شے<3> &شے_ct1) {
    صحیح global_اشاریہ = شے_ct1.get_local_id(2) +
                       شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2);
    اگر (global_اشاریہ >= output_سائز) {
        واپس;
    }

    صحیح برآمد_اشاریہ = global_اشاریہ / dst_ne0;

    اعشاری مجمع = 0;

    برائے (صحیح c = 0; c < src0_ne2; c++) {
        صحیح idx = global_اشاریہ % dst_ne0;

        صحیح kernel_انحراف = (src0_ne0 * src0_ne1 * c) + (برآمد_اشاریہ * src0_ne0);
        صحیح داخلہ_انحراف = src1_ne0 * c;

        برائے (صحیح i = 0; i < src1_ne0; i++) {
            اگر (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                جاری;
            }
            صحیح وزن_idx = idx - i*s0;

            اعشاری kernel_وزن = src0[kernel_انحراف + وزن_idx];
            اعشاری داخلہ_قدر =  src1[داخلہ_انحراف+i];

            مجمع += kernel_وزن * داخلہ_قدر;
        }
    }
    dst[global_اشاریہ] = مجمع;
}

ساکن خالی conv_transpose_1d_f32_f32_sycl(
    ثابت صحیح s0, ثابت صحیح output_سائز,
    ثابت صحیح src0_ne0, ثابت صحیح src0_ne1, ثابت صحیح src0_ne2,
    ثابت صحیح src1_ne0, ثابت صحیح dst_ne0,
    ثابت اعشاری *src0, ثابت اعشاری *src1, اعشاری *dst,
    ثابت قطار_ptr& stream) {

    ثابت صحیح عدد_بلاکیں = (output_سائز + SYCL_CONV_TRANPOSE_1_D_بلاک_سائز - 1) / SYCL_CONV_TRANPOSE_1_D_بلاک_سائز;
    ثابت sycl::range<3> بلاک_dims(1, 1, SYCL_CONV_TRANPOSE_1_D_بلاک_سائز);
    ثابت sycl::range<3> بلاک_nums(1, 1, عدد_بلاکیں);
    stream->parallel_for(
        sycl::nd_range<3>(
            بلاک_nums * بلاک_dims, بلاک_dims),
        [=](sycl::nd_شے<3> شے_ct1) {
            conv_transpose_1d_kernel(
                s0, output_سائز,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, شے_ct1);
        });
}

خالی ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ثابت ggml_tensor *src0 = dst->src[0];
    ثابت ggml_tensor *src1 = dst->src[1];
    ثابت اعشاری * src0_d = (ثابت اعشاری *)src0->ڈیٹا;
    ثابت اعشاری * src1_d = (ثابت اعشاری *)src1->ڈیٹا;

    اعشاری * dst_d = (اعشاری *)dst->ڈیٹا;
    dpct::قطار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ثابت int32_t * opts = (ثابت int32_t *)dst->op_params;

    ثابت صحیح s0 = opts[0];

    ثابت int64_t output_سائز = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_سائز,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

