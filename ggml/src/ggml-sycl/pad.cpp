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

स्थिर शून्य pad_f32(स्थिरांक दशांश * src, माप_प्रकार s00, माप_प्रकार s01, माप_प्रकार s02, माप_प्रकार s03,
                    दशांश * dst,
                    स्थिरांक पूर्णांक lp0, स्थिरांक पूर्णांक rp0, स्थिरांक पूर्णांक lp1, स्थिरांक पूर्णांक rp1,
                    स्थिरांक पूर्णांक lp2, स्थिरांक पूर्णांक rp2, स्थिरांक पूर्णांक lp3, स्थिरांक पूर्णांक rp3,
                    स्थिरांक पूर्णांक ne0, स्थिरांक पूर्णांक ne1, स्थिरांक पूर्णांक ne2, स्थिरांक पूर्णांक ne3,
                    sycl::nd_item<3> item_ct1) {
    पूर्णांक i0 = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
    पूर्णांक i1 = item_ct1.get_group(1);
    पूर्णांक i2 = item_ct1.get_group(0) % ne2;
    पूर्णांक i3 = item_ct1.get_group(0) / ne2;
    जर (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        परत;
    }

    स्थिरांक int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
    जर ((i0 >= lp0 && i0 < ne0 - rp0) &&
        (i1 >= lp1 && i1 < ne1 - rp1) &&
        (i2 >= lp2 && i2 < ne2 - rp2) &&
        (i3 >= lp3 && i3 < ne3 - rp3)) {
        स्थिरांक int64_t i00 = i0 - lp0;
        स्थिरांक int64_t i01 = i1 - lp1;
        स्थिरांक int64_t i02 = i2 - lp2;
        स्थिरांक int64_t i03 = i3 - lp3;

        स्थिरांक int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

        dst[dst_idx] = src[src_idx];
    } अन्यथा {
        dst[dst_idx] = 0.0f;
    }
}

स्थिर शून्य pad_f32_sycl(स्थिरांक दशांश * src, माप_प्रकार s00, माप_प्रकार s01, माप_प्रकार s02, माप_प्रकार s03,
                         दशांश * dst, स्थिरांक पूर्णांक lp0, स्थिरांक पूर्णांक rp0, स्थिरांक पूर्णांक lp1, स्थिरांक पूर्णांक rp1,
                         स्थिरांक पूर्णांक lp2, स्थिरांक पूर्णांक rp2, स्थिरांक पूर्णांक lp3, स्थिरांक पूर्णांक rp3,
                         स्थिरांक पूर्णांक ne0, स्थिरांक पूर्णांक ne1, स्थिरांक पूर्णांक ne2, स्थिरांक पूर्णांक ne3,
                         dpct::queue_ptr stream) {
    पूर्णांक अंक_blocks = (ne0 + SYCL_PAD_BLOCK_SIZE - 1) / SYCL_PAD_BLOCK_SIZE;
    sycl::range<3> grid(ne2 * ne3, ne1, अंक_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pad_f32(src, s00, s01, s02, s03, dst, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                    ne0, ne1, ne2, ne3, item_ct1);
        });
}

शून्य ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    स्थिरांक ggml_tensor * src0 = dst->src[0];
    स्थिरांक दशांश * src0_d = (स्थिरांक दशांश *)src0->data;
    दशांश * dst_d = (दशांश *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    स्थिरांक माप_प्रकार ts = ggml_type_size(src0->type);
    स्थिरांक माप_प्रकार s00 = src0->nb[0] / ts;
    स्थिरांक माप_प्रकार s01 = src0->nb[1] / ts;
    स्थिरांक माप_प्रकार s02 = src0->nb[2] / ts;
    स्थिरांक माप_प्रकार s03 = src0->nb[3] / ts;

    स्थिरांक int32_t lp0 = ((स्थिरांक int32_t *)(dst->op_params))[0];
    स्थिरांक int32_t rp0 = ((स्थिरांक int32_t *)(dst->op_params))[1];
    स्थिरांक int32_t lp1 = ((स्थिरांक int32_t *)(dst->op_params))[2];
    स्थिरांक int32_t rp1 = ((स्थिरांक int32_t *)(dst->op_params))[3];
    स्थिरांक int32_t lp2 = ((स्थिरांक int32_t *)(dst->op_params))[4];
    स्थिरांक int32_t rp2 = ((स्थिरांक int32_t *)(dst->op_params))[5];
    स्थिरांक int32_t lp3 = ((स्थिरांक int32_t *)(dst->op_params))[6];
    स्थिरांक int32_t rp3 = ((स्थिरांक int32_t *)(dst->op_params))[7];

    pad_f32_sycl(src0_d, s00, s01, s02, s03, dst_d,
                 lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                 dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
}

शून्य ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pad(ctx, dst);
}
