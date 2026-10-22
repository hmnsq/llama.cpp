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

#अन्तर्भाव "conv.hpp"

स्थिर  शून्य conv_transpose_1d_kernel(
        ध्रुव पूर्णाङ्क s0, ध्रुव पूर्णाङ्क output_आकार,
        ध्रुव पूर्णाङ्क src0_ne0, ध्रुव पूर्णाङ्क src0_ne1, ध्रुव पूर्णाङ्क src0_ne2,
        ध्रुव पूर्णाङ्क src1_ne0, ध्रुव पूर्णाङ्क dst_ne0,
        ध्रुव भिन्न * src0, ध्रुव भिन्न * src1,  भिन्न * dst,
        ध्रुव sycl::nd_item<3> &item_ct1) {
    पूर्णाङ्क global_सूचकांक = item_ct1.get_local_id(2) +
                       item_ct1.get_गणः(2) * item_ct1.get_local_range(2);
    यदि (global_सूचकांक >= output_आकार) {
        प्रत्यागम;
    }

    पूर्णाङ्क बहिर्गमनम्_सूचकांक = global_सूचकांक / dst_ne0;

    भिन्न सञ्चायकः = 0;

    प्रति (पूर्णाङ्क c = 0; c < src0_ne2; c++) {
        पूर्णाङ्क idx = global_सूचकांक % dst_ne0;

        पूर्णाङ्क kernel_विस्थापनम् = (src0_ne0 * src0_ne1 * c) + (बहिर्गमनम्_सूचकांक * src0_ne0);
        पूर्णाङ्क निवेशः_विस्थापनम् = src1_ne0 * c;

        प्रति (पूर्णाङ्क i = 0; i < src1_ne0; i++) {
            यदि (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                अनुवर्त;
            }
            पूर्णाङ्क weight_idx = idx - i*s0;

            भिन्न kernel_weight = src0[kernel_विस्थापनम् + weight_idx];
            भिन्न निवेशः_मानम् =  src1[निवेशः_विस्थापनम्+i];

            सञ्चायकः += kernel_weight * निवेशः_मानम्;
        }
    }
    dst[global_सूचकांक] = सञ्चायकः;
}

स्थिर शून्य conv_transpose_1d_f32_f32_sycl(
    ध्रुव पूर्णाङ्क s0, ध्रुव पूर्णाङ्क output_आकार,
    ध्रुव पूर्णाङ्क src0_ne0, ध्रुव पूर्णाङ्क src0_ne1, ध्रुव पूर्णाङ्क src0_ne2,
    ध्रुव पूर्णाङ्क src1_ne0, ध्रुव पूर्णाङ्क dst_ne0,
    ध्रुव भिन्न *src0, ध्रुव भिन्न *src1, भिन्न *dst,
    ध्रुव queue_ptr& stream) {

    ध्रुव पूर्णाङ्क अङ्कः_खण्डानि = (output_आकार + SYCL_CONV_TRANPOSE_1_D_खण्डम्_आकार - 1) / SYCL_CONV_TRANPOSE_1_D_खण्डम्_आकार;
    ध्रुव sycl::range<3> खण्डम्_dims(1, 1, SYCL_CONV_TRANPOSE_1_D_खण्डम्_आकार);
    ध्रुव sycl::range<3> खण्डम्_nums(1, 1, अङ्कः_खण्डानि);
    stream->parallel_for(
        sycl::nd_range<3>(
            खण्डम्_nums * खण्डम्_dims, खण्डम्_dims),
        [=](sycl::nd_item<3> item_ct1) {
            conv_transpose_1d_kernel(
                s0, output_आकार,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, item_ct1);
        });
}

शून्य ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ध्रुव ggml_tensor *src0 = dst->src[0];
    ध्रुव ggml_tensor *src1 = dst->src[1];
    ध्रुव भिन्न * src0_d = (ध्रुव भिन्न *)src0->दत्त;
    ध्रुव भिन्न * src1_d = (ध्रुव भिन्न *)src1->दत्त;

    भिन्न * dst_d = (भिन्न *)dst->दत्त;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ध्रुव int32_t * opts = (ध्रुव int32_t *)dst->op_params;

    ध्रुव पूर्णाङ्क s0 = opts[0];

    ध्रुव int64_t output_आकार = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_आकार,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

