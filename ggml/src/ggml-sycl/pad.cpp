//
// MIT license
// Copyright (C) 2025 Intel Corporation
// SPDX-License-Identifier: MIT
//

//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//

//#include "common.hpp"
#अन्तर्भाव "pad.hpp"

स्थिर शून्य pad_f32(ध्रुव भिन्न * src, size_t s00, size_t s01, size_t s02, size_t s03,
                    भिन्न * dst,
                    ध्रुव पूर्णाङ्क lp0, ध्रुव पूर्णाङ्क rp0, ध्रुव पूर्णाङ्क lp1, ध्रुव पूर्णाङ्क rp1,
                    ध्रुव पूर्णाङ्क lp2, ध्रुव पूर्णाङ्क rp2, ध्रुव पूर्णाङ्क lp3, ध्रुव पूर्णाङ्क rp3,
                    ध्रुव पूर्णाङ्क ne0, ध्रुव पूर्णाङ्क ne1, ध्रुव पूर्णाङ्क ne2, ध्रुव पूर्णाङ्क ne3,
                    sycl::nd_item<3> item_ct1) {
    पूर्णाङ्क i0 = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
    पूर्णाङ्क i1 = item_ct1.get_group(1);
    पूर्णाङ्क i2 = item_ct1.get_group(0) % ne2;
    पूर्णाङ्क i3 = item_ct1.get_group(0) / ne2;
    जइ (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        प्रत्यागम;
    }

    ध्रुव int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
    जइ ((i0 >= lp0 && i0 < ne0 - rp0) &&
        (i1 >= lp1 && i1 < ne1 - rp1) &&
        (i2 >= lp2 && i2 < ne2 - rp2) &&
        (i3 >= lp3 && i3 < ne3 - rp3)) {
        ध्रुव int64_t i00 = i0 - lp0;
        ध्रुव int64_t i01 = i1 - lp1;
        ध्रुव int64_t i02 = i2 - lp2;
        ध्रुव int64_t i03 = i3 - lp3;

        ध्रुव int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

        dst[dst_idx] = src[src_idx];
    } अण्णहा {
        dst[dst_idx] = 0.0f;
    }
}

स्थिर शून्य pad_f32_sycl(ध्रुव भिन्न * src, size_t s00, size_t s01, size_t s02, size_t s03,
                         भिन्न * dst, ध्रुव पूर्णाङ्क lp0, ध्रुव पूर्णाङ्क rp0, ध्रुव पूर्णाङ्क lp1, ध्रुव पूर्णाङ्क rp1,
                         ध्रुव पूर्णाङ्क lp2, ध्रुव पूर्णाङ्क rp2, ध्रुव पूर्णाङ्क lp3, ध्रुव पूर्णाङ्क rp3,
                         ध्रुव पूर्णाङ्क ne0, ध्रुव पूर्णाङ्क ne1, ध्रुव पूर्णाङ्क ne2, ध्रुव पूर्णाङ्क ne3,
                         dpct::queue_ptr stream) {
    पूर्णाङ्क अङ्कः_blocks = (ne0 + SYCL_PAD_BLOCK_SIZE - 1) / SYCL_PAD_BLOCK_SIZE;
    sycl::range<3> grid(ne2 * ne3, ne1, अङ्कः_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pad_f32(src, s00, s01, s02, s03, dst, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                    ne0, ne1, ne2, ne3, item_ct1);
        });
}

शून्य ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ध्रुव ggml_tensor * src0 = dst->src[0];
    ध्रुव भिन्न * src0_d = (ध्रुव भिन्न *)src0->data;
    भिन्न * dst_d = (भिन्न *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ध्रुव size_t ts = ggml_type_size(src0->type);
    ध्रुव size_t s00 = src0->nb[0] / ts;
    ध्रुव size_t s01 = src0->nb[1] / ts;
    ध्रुव size_t s02 = src0->nb[2] / ts;
    ध्रुव size_t s03 = src0->nb[3] / ts;

    ध्रुव int32_t lp0 = ((ध्रुव int32_t *)(dst->op_params))[0];
    ध्रुव int32_t rp0 = ((ध्रुव int32_t *)(dst->op_params))[1];
    ध्रुव int32_t lp1 = ((ध्रुव int32_t *)(dst->op_params))[2];
    ध्रुव int32_t rp1 = ((ध्रुव int32_t *)(dst->op_params))[3];
    ध्रुव int32_t lp2 = ((ध्रुव int32_t *)(dst->op_params))[4];
    ध्रुव int32_t rp2 = ((ध्रुव int32_t *)(dst->op_params))[5];
    ध्रुव int32_t lp3 = ((ध्रुव int32_t *)(dst->op_params))[6];
    ध्रुव int32_t rp3 = ((ध्रुव int32_t *)(dst->op_params))[7];

    pad_f32_sycl(src0_d, s00, s01, s02, s03, dst_d,
                 lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                 dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
}

शून्य ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pad(ctx, dst);
}
