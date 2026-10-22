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
#அந்தர்பாவ "pad.hpp"

நிலையான வெறுமை pad_f32(மாறிலி தசம * src, size_t s00, size_t s01, size_t s02, size_t s03,
                    தசம * dst,
                    மாறிலி முழு lp0, மாறிலி முழு rp0, மாறிலி முழு lp1, மாறிலி முழு rp1,
                    மாறிலி முழு lp2, மாறிலி முழு rp2, மாறிலி முழு lp3, மாறிலி முழு rp3,
                    மாறிலி முழு ne0, மாறிலி முழு ne1, மாறிலி முழு ne2, மாறிலி முழு ne3,
                    sycl::nd_item<3> item_ct1) {
    முழு i0 = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
    முழு i1 = item_ct1.get_group(1);
    முழு i2 = item_ct1.get_group(0) % ne2;
    முழு i3 = item_ct1.get_group(0) / ne2;
    எனில் (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        திரும்பு;
    }

    மாறிலி int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
    எனில் ((i0 >= lp0 && i0 < ne0 - rp0) &&
        (i1 >= lp1 && i1 < ne1 - rp1) &&
        (i2 >= lp2 && i2 < ne2 - rp2) &&
        (i3 >= lp3 && i3 < ne3 - rp3)) {
        மாறிலி int64_t i00 = i0 - lp0;
        மாறிலி int64_t i01 = i1 - lp1;
        மாறிலி int64_t i02 = i2 - lp2;
        மாறிலி int64_t i03 = i3 - lp3;

        மாறிலி int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

        dst[dst_idx] = src[src_idx];
    } இல்லையெனில் {
        dst[dst_idx] = 0.0f;
    }
}

நிலையான வெறுமை pad_f32_sycl(மாறிலி தசம * src, size_t s00, size_t s01, size_t s02, size_t s03,
                         தசம * dst, மாறிலி முழு lp0, மாறிலி முழு rp0, மாறிலி முழு lp1, மாறிலி முழு rp1,
                         மாறிலி முழு lp2, மாறிலி முழு rp2, மாறிலி முழு lp3, மாறிலி முழு rp3,
                         மாறிலி முழு ne0, மாறிலி முழு ne1, மாறிலி முழு ne2, மாறிலி முழு ne3,
                         dpct::queue_ptr stream) {
    முழு இலக்கம்_blocks = (ne0 + SYCL_PAD_BLOCK_SIZE - 1) / SYCL_PAD_BLOCK_SIZE;
    sycl::range<3> grid(ne2 * ne3, ne1, இலக்கம்_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pad_f32(src, s00, s01, s02, s03, dst, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                    ne0, ne1, ne2, ne3, item_ct1);
        });
}

வெறுமை ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    மாறிலி ggml_tensor * src0 = dst->src[0];
    மாறிலி தசம * src0_d = (மாறிலி தசம *)src0->data;
    தசம * dst_d = (தசம *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    மாறிலி size_t ts = ggml_type_size(src0->type);
    மாறிலி size_t s00 = src0->nb[0] / ts;
    மாறிலி size_t s01 = src0->nb[1] / ts;
    மாறிலி size_t s02 = src0->nb[2] / ts;
    மாறிலி size_t s03 = src0->nb[3] / ts;

    மாறிலி int32_t lp0 = ((மாறிலி int32_t *)(dst->op_params))[0];
    மாறிலி int32_t rp0 = ((மாறிலி int32_t *)(dst->op_params))[1];
    மாறிலி int32_t lp1 = ((மாறிலி int32_t *)(dst->op_params))[2];
    மாறிலி int32_t rp1 = ((மாறிலி int32_t *)(dst->op_params))[3];
    மாறிலி int32_t lp2 = ((மாறிலி int32_t *)(dst->op_params))[4];
    மாறிலி int32_t rp2 = ((மாறிலி int32_t *)(dst->op_params))[5];
    மாறிலி int32_t lp3 = ((மாறிலி int32_t *)(dst->op_params))[6];
    மாறிலி int32_t rp3 = ((மாறிலி int32_t *)(dst->op_params))[7];

    pad_f32_sycl(src0_d, s00, s01, s02, s03, dst_d,
                 lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                 dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
}

வெறுமை ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pad(ctx, dst);
}
