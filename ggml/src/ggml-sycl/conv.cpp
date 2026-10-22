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

#ಅನ್ತರ್ಭಾವ "conv.hpp"

ಸ್ಥಿರ  ಶೂನ್ಯ conv_transpose_1d_kernel(
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ output_ಗಾತ್ರ,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src0_ne0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src0_ne1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src0_ne2,
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src1_ne0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dst_ne0,
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1,  ದಶಮಾಂಶ * dst,
        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಪೂರ್ಣಾಂಕ global_ಸೂಚ್ಯಂಕ = ಅಂಶ_ct1.get_local_id(2) +
                       ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
    ಆದರೆ (global_ಸೂಚ್ಯಂಕ >= output_ಗಾತ್ರ) {
        ಹಿಂತಿರುಗಿ;
    }

    ಪೂರ್ಣಾಂಕ ಹೊರಹೋಗು_ಸೂಚ್ಯಂಕ = global_ಸೂಚ್ಯಂಕ / dst_ne0;

    ದಶಮಾಂಶ ಸಂಚಯಕ = 0;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = 0; c < src0_ne2; c++) {
        ಪೂರ್ಣಾಂಕ idx = global_ಸೂಚ್ಯಂಕ % dst_ne0;

        ಪೂರ್ಣಾಂಕ kernel_ಸ್ಥಾನಾಂತರ = (src0_ne0 * src0_ne1 * c) + (ಹೊರಹೋಗು_ಸೂಚ್ಯಂಕ * src0_ne0);
        ಪೂರ್ಣಾಂಕ ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ = src1_ne0 * c;

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < src1_ne0; i++) {
            ಆದರೆ (!(idx >= i*s0 && idx < i*s0 + src0_ne0)) {
                ಮುಂದುವರಿ;
            }
            ಪೂರ್ಣಾಂಕ ತೂಕ_idx = idx - i*s0;

            ದಶಮಾಂಶ kernel_ತೂಕ = src0[kernel_ಸ್ಥಾನಾಂತರ + ತೂಕ_idx];
            ದಶಮಾಂಶ ಒಳಹರಿವು_ಮೌಲ್ಯ =  src1[ಒಳಹರಿವು_ಸ್ಥಾನಾಂತರ+i];

            ಸಂಚಯಕ += kernel_ತೂಕ * ಒಳಹರಿವು_ಮೌಲ್ಯ;
        }
    }
    dst[global_ಸೂಚ್ಯಂಕ] = ಸಂಚಯಕ;
}

ಸ್ಥಿರ ಶೂನ್ಯ conv_transpose_1d_f32_f32_sycl(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ output_ಗಾತ್ರ,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src0_ne0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src0_ne1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src0_ne2,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ src1_ne0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dst_ne0,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src0, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *src1, ದಶಮಾಂಶ *dst,
    ಸ್ಥಿರಾಂಕ ಸರತಿ_ptr& stream) {

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = (output_ಗಾತ್ರ + SYCL_CONV_TRANPOSE_1_D_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_CONV_TRANPOSE_1_D_ಖಂಡ_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_dims(1, 1, SYCL_CONV_TRANPOSE_1_D_ಖಂಡ_ಗಾತ್ರ);
    ಸ್ಥಿರಾಂಕ sycl::range<3> ಖಂಡ_nums(1, 1, ಅಂಕೆ_ಖಂಡಗಳು);
    stream->parallel_for(
        sycl::nd_range<3>(
            ಖಂಡ_nums * ಖಂಡ_dims, ಖಂಡ_dims),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            conv_transpose_1d_kernel(
                s0, output_ಗಾತ್ರ,
                src0_ne0, src0_ne1, src0_ne2,
                src1_ne0, dst_ne0,
                src0, src1, dst, ಅಂಶ_ct1);
        });
}

ಶೂನ್ಯ ggml_sycl_op_conv_transpose_1d(ggml_backend_sycl_context & ctx, ggml_tensor *dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/2);
    ಸ್ಥಿರಾಂಕ ggml_tensor *src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ggml_tensor *src1 = dst->src[1];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src1_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src1->ದತ್ತಾಂಶ;

    ದಶಮಾಂಶ * dst_d = (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ;
    dpct::ಸರತಿ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT( dst->type == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));

    ಸ್ಥಿರಾಂಕ int32_t * opts = (ಸ್ಥಿರಾಂಕ int32_t *)dst->op_params;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ s0 = opts[0];

    ಸ್ಥಿರಾಂಕ int64_t output_ಗಾತ್ರ = ggml_nelements(dst);

    conv_transpose_1d_f32_f32_sycl(s0, output_ಗಾತ್ರ,
        src0->ne[0], src0->ne[1], src0->ne[2],
        src1->ne[0], dst->ne[0],
        src0_d, src1_d, dst_d, stream);
}

