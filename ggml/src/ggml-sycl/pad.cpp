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
#অন্তর্ভাব "pad.hpp"

স্থির শূন্য pad_f32(ধ্রুবক দশমিক * src, size_t s00, size_t s01, size_t s02, size_t s03,
                    দশমিক * dst,
                    ধ্রুবক পূর্ণসংখ্যা lp0, ধ্রুবক পূর্ণসংখ্যা rp0, ধ্রুবক পূর্ণসংখ্যা lp1, ধ্রুবক পূর্ণসংখ্যা rp1,
                    ধ্রুবক পূর্ণসংখ্যা lp2, ধ্রুবক পূর্ণসংখ্যা rp2, ধ্রুবক পূর্ণসংখ্যা lp3, ধ্রুবক পূর্ণসংখ্যা rp3,
                    ধ্রুবক পূর্ণসংখ্যা ne0, ধ্রুবক পূর্ণসংখ্যা ne1, ধ্রুবক পূর্ণসংখ্যা ne2, ধ্রুবক পূর্ণসংখ্যা ne3,
                    sycl::nd_item<3> item_ct1) {
    পূর্ণসংখ্যা i0 = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
    পূর্ণসংখ্যা i1 = item_ct1.get_group(1);
    পূর্ণসংখ্যা i2 = item_ct1.get_group(0) % ne2;
    পূর্ণসংখ্যা i3 = item_ct1.get_group(0) / ne2;
    যদি (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        ফেরত;
    }

    ধ্রুবক int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
    যদি ((i0 >= lp0 && i0 < ne0 - rp0) &&
        (i1 >= lp1 && i1 < ne1 - rp1) &&
        (i2 >= lp2 && i2 < ne2 - rp2) &&
        (i3 >= lp3 && i3 < ne3 - rp3)) {
        ধ্রুবক int64_t i00 = i0 - lp0;
        ধ্রুবক int64_t i01 = i1 - lp1;
        ধ্রুবক int64_t i02 = i2 - lp2;
        ধ্রুবক int64_t i03 = i3 - lp3;

        ধ্রুবক int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

        dst[dst_idx] = src[src_idx];
    } অন্যথা {
        dst[dst_idx] = 0.0f;
    }
}

স্থির শূন্য pad_f32_sycl(ধ্রুবক দশমিক * src, size_t s00, size_t s01, size_t s02, size_t s03,
                         দশমিক * dst, ধ্রুবক পূর্ণসংখ্যা lp0, ধ্রুবক পূর্ণসংখ্যা rp0, ধ্রুবক পূর্ণসংখ্যা lp1, ধ্রুবক পূর্ণসংখ্যা rp1,
                         ধ্রুবক পূর্ণসংখ্যা lp2, ধ্রুবক পূর্ণসংখ্যা rp2, ধ্রুবক পূর্ণসংখ্যা lp3, ধ্রুবক পূর্ণসংখ্যা rp3,
                         ধ্রুবক পূর্ণসংখ্যা ne0, ধ্রুবক পূর্ণসংখ্যা ne1, ধ্রুবক পূর্ণসংখ্যা ne2, ধ্রুবক পূর্ণসংখ্যা ne3,
                         dpct::queue_ptr stream) {
    পূর্ণসংখ্যা অঙ্ক_blocks = (ne0 + SYCL_PAD_BLOCK_আকার - 1) / SYCL_PAD_BLOCK_আকার;
    sycl::range<3> grid(ne2 * ne3, ne1, অঙ্ক_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, SYCL_PAD_BLOCK_আকার),
                          sycl::range<3>(1, 1, SYCL_PAD_BLOCK_আকার)),
        [=](sycl::nd_item<3> item_ct1) {
            pad_f32(src, s00, s01, s02, s03, dst, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                    ne0, ne1, ne2, ne3, item_ct1);
        });
}

শূন্য ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ধ্রুবক ggml_tensor * src0 = dst->src[0];
    ধ্রুবক দশমিক * src0_d = (ধ্রুবক দশমিক *)src0->তথ্য;
    দশমিক * dst_d = (দশমিক *)dst->তথ্য;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ধ্রুবক size_t ts = ggml_type_আকার(src0->type);
    ধ্রুবক size_t s00 = src0->nb[0] / ts;
    ধ্রুবক size_t s01 = src0->nb[1] / ts;
    ধ্রুবক size_t s02 = src0->nb[2] / ts;
    ধ্রুবক size_t s03 = src0->nb[3] / ts;

    ধ্রুবক int32_t lp0 = ((ধ্রুবক int32_t *)(dst->op_params))[0];
    ধ্রুবক int32_t rp0 = ((ধ্রুবক int32_t *)(dst->op_params))[1];
    ধ্রুবক int32_t lp1 = ((ধ্রুবক int32_t *)(dst->op_params))[2];
    ধ্রুবক int32_t rp1 = ((ধ্রুবক int32_t *)(dst->op_params))[3];
    ধ্রুবক int32_t lp2 = ((ধ্রুবক int32_t *)(dst->op_params))[4];
    ধ্রুবক int32_t rp2 = ((ধ্রুবক int32_t *)(dst->op_params))[5];
    ধ্রুবক int32_t lp3 = ((ধ্রুবক int32_t *)(dst->op_params))[6];
    ধ্রুবক int32_t rp3 = ((ধ্রুবক int32_t *)(dst->op_params))[7];

    pad_f32_sycl(src0_d, s00, s01, s02, s03, dst_d,
                 lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                 dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
}

শূন্য ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pad(ctx, dst);
}
