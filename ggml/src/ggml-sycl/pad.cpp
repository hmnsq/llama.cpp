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
#અન્તર્ભાવ "pad.hpp"

સ્થિર શૂન્ય pad_f32(અચલ દશાંશ * src, size_t s00, size_t s01, size_t s02, size_t s03,
                    દશાંશ * dst,
                    અચલ પૂર્ણાંક lp0, અચલ પૂર્ણાંક rp0, અચલ પૂર્ણાંક lp1, અચલ પૂર્ણાંક rp1,
                    અચલ પૂર્ણાંક lp2, અચલ પૂર્ણાંક rp2, અચલ પૂર્ણાંક lp3, અચલ પૂર્ણાંક rp3,
                    અચલ પૂર્ણાંક ne0, અચલ પૂર્ણાંક ne1, અચલ પૂર્ણાંક ne2, અચલ પૂર્ણાંક ne3,
                    sycl::nd_item<3> item_ct1) {
    પૂર્ણાંક i0 = item_ct1.get_local_id(2) +
             item_ct1.get_group(2) * item_ct1.get_local_range(2);
    પૂર્ણાંક i1 = item_ct1.get_group(1);
    પૂર્ણાંક i2 = item_ct1.get_group(0) % ne2;
    પૂર્ણાંક i3 = item_ct1.get_group(0) / ne2;
    જો (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        પાછા;
    }

    અચલ int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
    જો ((i0 >= lp0 && i0 < ne0 - rp0) &&
        (i1 >= lp1 && i1 < ne1 - rp1) &&
        (i2 >= lp2 && i2 < ne2 - rp2) &&
        (i3 >= lp3 && i3 < ne3 - rp3)) {
        અચલ int64_t i00 = i0 - lp0;
        અચલ int64_t i01 = i1 - lp1;
        અચલ int64_t i02 = i2 - lp2;
        અચલ int64_t i03 = i3 - lp3;

        અચલ int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

        dst[dst_idx] = src[src_idx];
    } નહીંતર {
        dst[dst_idx] = 0.0f;
    }
}

સ્થિર શૂન્ય pad_f32_sycl(અચલ દશાંશ * src, size_t s00, size_t s01, size_t s02, size_t s03,
                         દશાંશ * dst, અચલ પૂર્ણાંક lp0, અચલ પૂર્ણાંક rp0, અચલ પૂર્ણાંક lp1, અચલ પૂર્ણાંક rp1,
                         અચલ પૂર્ણાંક lp2, અચલ પૂર્ણાંક rp2, અચલ પૂર્ણાંક lp3, અચલ પૂર્ણાંક rp3,
                         અચલ પૂર્ણાંક ne0, અચલ પૂર્ણાંક ne1, અચલ પૂર્ણાંક ne2, અચલ પૂર્ણાંક ne3,
                         dpct::queue_ptr stream) {
    પૂર્ણાંક અંક_blocks = (ne0 + SYCL_PAD_BLOCK_SIZE - 1) / SYCL_PAD_BLOCK_SIZE;
    sycl::range<3> grid(ne2 * ne3, ne1, અંક_blocks);
    stream->parallel_for(
        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE),
                          sycl::range<3>(1, 1, SYCL_PAD_BLOCK_SIZE)),
        [=](sycl::nd_item<3> item_ct1) {
            pad_f32(src, s00, s01, s02, s03, dst, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                    ne0, ne1, ne2, ne3, item_ct1);
        });
}

શૂન્ય ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    અચલ ggml_tensor * src0 = dst->src[0];
    અચલ દશાંશ * src0_d = (અચલ દશાંશ *)src0->data;
    દશાંશ * dst_d = (દશાંશ *)dst->data;
    dpct::queue_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    અચલ size_t ts = ggml_type_size(src0->type);
    અચલ size_t s00 = src0->nb[0] / ts;
    અચલ size_t s01 = src0->nb[1] / ts;
    અચલ size_t s02 = src0->nb[2] / ts;
    અચલ size_t s03 = src0->nb[3] / ts;

    અચલ int32_t lp0 = ((અચલ int32_t *)(dst->op_params))[0];
    અચલ int32_t rp0 = ((અચલ int32_t *)(dst->op_params))[1];
    અચલ int32_t lp1 = ((અચલ int32_t *)(dst->op_params))[2];
    અચલ int32_t rp1 = ((અચલ int32_t *)(dst->op_params))[3];
    અચલ int32_t lp2 = ((અચલ int32_t *)(dst->op_params))[4];
    અચલ int32_t rp2 = ((અચલ int32_t *)(dst->op_params))[5];
    અચલ int32_t lp3 = ((અચલ int32_t *)(dst->op_params))[6];
    અચલ int32_t rp3 = ((અચલ int32_t *)(dst->op_params))[7];

    pad_f32_sycl(src0_d, s00, s01, s02, s03, dst_d,
                 lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                 dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
}

શૂન્ય ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pad(ctx, dst);
}
