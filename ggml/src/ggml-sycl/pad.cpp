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
#ਅਨ੍ਤਰ੍ਭਾਵ "pad.hpp"

ਸਥਿਰ ਖਾਲੀ pad_f32(ਅਚਲ ਦਸ਼ਮਲਵ * src, size_t s00, size_t s01, size_t s02, size_t s03,
                    ਦਸ਼ਮਲਵ * dst,
                    ਅਚਲ ਪੂਰਨ lp0, ਅਚਲ ਪੂਰਨ rp0, ਅਚਲ ਪੂਰਨ lp1, ਅਚਲ ਪੂਰਨ rp1,
                    ਅਚਲ ਪੂਰਨ lp2, ਅਚਲ ਪੂਰਨ rp2, ਅਚਲ ਪੂਰਨ lp3, ਅਚਲ ਪੂਰਨ rp3,
                    ਅਚਲ ਪੂਰਨ ne0, ਅਚਲ ਪੂਰਨ ne1, ਅਚਲ ਪੂਰਨ ne2, ਅਚਲ ਪੂਰਨ ne3,
                    sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
    ਪੂਰਨ i0 = ਵਸਤੂ_ct1.get_local_id(2) +
             ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * ਵਸਤੂ_ct1.get_local_range(2);
    ਪੂਰਨ i1 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(1);
    ਪੂਰਨ i2 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) % ne2;
    ਪੂਰਨ i3 = ਵਸਤੂ_ct1.get_ਸਮੂਹ(0) / ne2;
    ਜੇ (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        ਵਾਪਸ;
    }

    ਅਚਲ int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
    ਜੇ ((i0 >= lp0 && i0 < ne0 - rp0) &&
        (i1 >= lp1 && i1 < ne1 - rp1) &&
        (i2 >= lp2 && i2 < ne2 - rp2) &&
        (i3 >= lp3 && i3 < ne3 - rp3)) {
        ਅਚਲ int64_t i00 = i0 - lp0;
        ਅਚਲ int64_t i01 = i1 - lp1;
        ਅਚਲ int64_t i02 = i2 - lp2;
        ਅਚਲ int64_t i03 = i3 - lp3;

        ਅਚਲ int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

        dst[dst_idx] = src[src_idx];
    } ਨਹੀਂਤਾਂ {
        dst[dst_idx] = 0.0f;
    }
}

ਸਥਿਰ ਖਾਲੀ pad_f32_sycl(ਅਚਲ ਦਸ਼ਮਲਵ * src, size_t s00, size_t s01, size_t s02, size_t s03,
                         ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਪੂਰਨ lp0, ਅਚਲ ਪੂਰਨ rp0, ਅਚਲ ਪੂਰਨ lp1, ਅਚਲ ਪੂਰਨ rp1,
                         ਅਚਲ ਪੂਰਨ lp2, ਅਚਲ ਪੂਰਨ rp2, ਅਚਲ ਪੂਰਨ lp3, ਅਚਲ ਪੂਰਨ rp3,
                         ਅਚਲ ਪੂਰਨ ne0, ਅਚਲ ਪੂਰਨ ne1, ਅਚਲ ਪੂਰਨ ne2, ਅਚਲ ਪੂਰਨ ne3,
                         dpct::ਕਤਾਰ_ptr stream) {
    ਪੂਰਨ ਅੰਕ_ਖੰਡਾਂ = (ne0 + SYCL_PAD_ਖੰਡ_ਮਾਪ - 1) / SYCL_PAD_ਖੰਡ_ਮਾਪ;
    sycl::range<3> grid(ne2 * ne3, ne1, ਅੰਕ_ਖੰਡਾਂ);
    stream->parallel_for(
        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, SYCL_PAD_ਖੰਡ_ਮਾਪ),
                          sycl::range<3>(1, 1, SYCL_PAD_ਖੰਡ_ਮਾਪ)),
        [=](sycl::nd_ਵਸਤੂ<3> ਵਸਤੂ_ct1) {
            pad_f32(src, s00, s01, s02, s03, dst, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                    ne0, ne1, ne2, ne3, ਵਸਤੂ_ct1);
        });
}

ਖਾਲੀ ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ਅਚਲ ggml_tensor * src0 = dst->src[0];
    ਅਚਲ ਦਸ਼ਮਲਵ * src0_d = (ਅਚਲ ਦਸ਼ਮਲਵ *)src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_d = (ਦਸ਼ਮਲਵ *)dst->ਅੰਕੜੇ;
    dpct::ਕਤਾਰ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ਅਚਲ size_t ts = ggml_type_ਮਾਪ(src0->type);
    ਅਚਲ size_t s00 = src0->nb[0] / ts;
    ਅਚਲ size_t s01 = src0->nb[1] / ts;
    ਅਚਲ size_t s02 = src0->nb[2] / ts;
    ਅਚਲ size_t s03 = src0->nb[3] / ts;

    ਅਚਲ int32_t lp0 = ((ਅਚਲ int32_t *)(dst->op_params))[0];
    ਅਚਲ int32_t rp0 = ((ਅਚਲ int32_t *)(dst->op_params))[1];
    ਅਚਲ int32_t lp1 = ((ਅਚਲ int32_t *)(dst->op_params))[2];
    ਅਚਲ int32_t rp1 = ((ਅਚਲ int32_t *)(dst->op_params))[3];
    ਅਚਲ int32_t lp2 = ((ਅਚਲ int32_t *)(dst->op_params))[4];
    ਅਚਲ int32_t rp2 = ((ਅਚਲ int32_t *)(dst->op_params))[5];
    ਅਚਲ int32_t lp3 = ((ਅਚਲ int32_t *)(dst->op_params))[6];
    ਅਚਲ int32_t rp3 = ((ਅਚਲ int32_t *)(dst->op_params))[7];

    pad_f32_sycl(src0_d, s00, s01, s02, s03, dst_d,
                 lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                 dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
}

ਖਾਲੀ ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pad(ctx, dst);
}
