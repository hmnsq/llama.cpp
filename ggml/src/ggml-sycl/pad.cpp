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
#ضمن "pad.hpp"

ساكن فراغ pad_f32(ثابت عائم * src, size_t s00, size_t s01, size_t s02, size_t s03,
                    عائم * dst,
                    ثابت صحيح lp0, ثابت صحيح rp0, ثابت صحيح lp1, ثابت صحيح rp1,
                    ثابت صحيح lp2, ثابت صحيح rp2, ثابت صحيح lp3, ثابت صحيح rp3,
                    ثابت صحيح ne0, ثابت صحيح ne1, ثابت صحيح ne2, ثابت صحيح ne3,
                    sycl::nd_شي<3> شي_ct1) {
    صحيح i0 = شي_ct1.get_local_id(2) +
             شي_ct1.get_گروهه(2) * شي_ct1.get_local_range(2);
    صحيح i1 = شي_ct1.get_گروهه(1);
    صحيح i2 = شي_ct1.get_گروهه(0) % ne2;
    صحيح i3 = شي_ct1.get_گروهه(0) / ne2;
    إذا (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        أرجع;
    }

    ثابت int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
    إذا ((i0 >= lp0 && i0 < ne0 - rp0) &&
        (i1 >= lp1 && i1 < ne1 - rp1) &&
        (i2 >= lp2 && i2 < ne2 - rp2) &&
        (i3 >= lp3 && i3 < ne3 - rp3)) {
        ثابت int64_t i00 = i0 - lp0;
        ثابت int64_t i01 = i1 - lp1;
        ثابت int64_t i02 = i2 - lp2;
        ثابت int64_t i03 = i3 - lp3;

        ثابت int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

        dst[dst_idx] = src[src_idx];
    } وإلا {
        dst[dst_idx] = 0.0f;
    }
}

ساكن فراغ pad_f32_sycl(ثابت عائم * src, size_t s00, size_t s01, size_t s02, size_t s03,
                         عائم * dst, ثابت صحيح lp0, ثابت صحيح rp0, ثابت صحيح lp1, ثابت صحيح rp1,
                         ثابت صحيح lp2, ثابت صحيح rp2, ثابت صحيح lp3, ثابت صحيح rp3,
                         ثابت صحيح ne0, ثابت صحيح ne1, ثابت صحيح ne2, ثابت صحيح ne3,
                         dpct::قطار_ptr stream) {
    صحيح انگ_بلاڪون = (ne0 + SYCL_PAD_بلاڪ_سائيز - 1) / SYCL_PAD_بلاڪ_سائيز;
    sycl::range<3> grid(ne2 * ne3, ne1, انگ_بلاڪون);
    stream->parallel_for(
        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, SYCL_PAD_بلاڪ_سائيز),
                          sycl::range<3>(1, 1, SYCL_PAD_بلاڪ_سائيز)),
        [=](sycl::nd_شي<3> شي_ct1) {
            pad_f32(src, s00, s01, s02, s03, dst, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                    ne0, ne1, ne2, ne3, شي_ct1);
        });
}

فراغ ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت عائم * src0_d = (ثابت عائم *)src0->معلومات;
    عائم * dst_d = (عائم *)dst->معلومات;
    dpct::قطار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت size_t ts = ggml_type_سائيز(src0->type);
    ثابت size_t s00 = src0->nb[0] / ts;
    ثابت size_t s01 = src0->nb[1] / ts;
    ثابت size_t s02 = src0->nb[2] / ts;
    ثابت size_t s03 = src0->nb[3] / ts;

    ثابت int32_t lp0 = ((ثابت int32_t *)(dst->op_params))[0];
    ثابت int32_t rp0 = ((ثابت int32_t *)(dst->op_params))[1];
    ثابت int32_t lp1 = ((ثابت int32_t *)(dst->op_params))[2];
    ثابت int32_t rp1 = ((ثابت int32_t *)(dst->op_params))[3];
    ثابت int32_t lp2 = ((ثابت int32_t *)(dst->op_params))[4];
    ثابت int32_t rp2 = ((ثابت int32_t *)(dst->op_params))[5];
    ثابت int32_t lp3 = ((ثابت int32_t *)(dst->op_params))[6];
    ثابت int32_t rp3 = ((ثابت int32_t *)(dst->op_params))[7];

    pad_f32_sycl(src0_d, s00, s01, s02, s03, dst_d,
                 lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                 dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
}

فراغ ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pad(ctx, dst);
}
