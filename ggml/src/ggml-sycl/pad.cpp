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
#شامل "pad.hpp"

ساکن خالی pad_f32(ثابت اعشاری * src, size_t s00, size_t s01, size_t s02, size_t s03,
                    اعشاری * dst,
                    ثابت صحیح lp0, ثابت صحیح rp0, ثابت صحیح lp1, ثابت صحیح rp1,
                    ثابت صحیح lp2, ثابت صحیح rp2, ثابت صحیح lp3, ثابت صحیح rp3,
                    ثابت صحیح ne0, ثابت صحیح ne1, ثابت صحیح ne2, ثابت صحیح ne3,
                    sycl::nd_شے<3> شے_ct1) {
    صحیح i0 = شے_ct1.get_local_id(2) +
             شے_ct1.get_گروہ(2) * شے_ct1.get_local_range(2);
    صحیح i1 = شے_ct1.get_گروہ(1);
    صحیح i2 = شے_ct1.get_گروہ(0) % ne2;
    صحیح i3 = شے_ct1.get_گروہ(0) / ne2;
    اگر (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        واپس;
    }

    ثابت int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
    اگر ((i0 >= lp0 && i0 < ne0 - rp0) &&
        (i1 >= lp1 && i1 < ne1 - rp1) &&
        (i2 >= lp2 && i2 < ne2 - rp2) &&
        (i3 >= lp3 && i3 < ne3 - rp3)) {
        ثابت int64_t i00 = i0 - lp0;
        ثابت int64_t i01 = i1 - lp1;
        ثابت int64_t i02 = i2 - lp2;
        ثابت int64_t i03 = i3 - lp3;

        ثابت int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

        dst[dst_idx] = src[src_idx];
    } ورنہ {
        dst[dst_idx] = 0.0f;
    }
}

ساکن خالی pad_f32_sycl(ثابت اعشاری * src, size_t s00, size_t s01, size_t s02, size_t s03,
                         اعشاری * dst, ثابت صحیح lp0, ثابت صحیح rp0, ثابت صحیح lp1, ثابت صحیح rp1,
                         ثابت صحیح lp2, ثابت صحیح rp2, ثابت صحیح lp3, ثابت صحیح rp3,
                         ثابت صحیح ne0, ثابت صحیح ne1, ثابت صحیح ne2, ثابت صحیح ne3,
                         dpct::قطار_ptr stream) {
    صحیح عدد_بلاکیں = (ne0 + SYCL_PAD_بلاک_سائز - 1) / SYCL_PAD_بلاک_سائز;
    sycl::range<3> grid(ne2 * ne3, ne1, عدد_بلاکیں);
    stream->parallel_for(
        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, SYCL_PAD_بلاک_سائز),
                          sycl::range<3>(1, 1, SYCL_PAD_بلاک_سائز)),
        [=](sycl::nd_شے<3> شے_ct1) {
            pad_f32(src, s00, s01, s02, s03, dst, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                    ne0, ne1, ne2, ne3, شے_ct1);
        });
}

خالی ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ثابت ggml_tensor * src0 = dst->src[0];
    ثابت اعشاری * src0_d = (ثابت اعشاری *)src0->ڈیٹا;
    اعشاری * dst_d = (اعشاری *)dst->ڈیٹا;
    dpct::قطار_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ثابت size_t ts = ggml_type_سائز(src0->type);
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

خالی ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pad(ctx, dst);
}
