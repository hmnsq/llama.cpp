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
#ಅನ್ತರ್ಭಾವ "pad.hpp"

ಸ್ಥಿರ ಶೂನ್ಯ pad_f32(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, size_t s00, size_t s01, size_t s02, size_t s03,
                    ದಶಮಾಂಶ * dst,
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lp0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ rp0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lp1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ rp1,
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lp2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ rp2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lp3, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ rp3,
                    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne3,
                    sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
    ಪೂರ್ಣಾಂಕ i0 = ಅಂಶ_ct1.get_local_id(2) +
             ಅಂಶ_ct1.get_ಗುಂಪು(2) * ಅಂಶ_ct1.get_local_range(2);
    ಪೂರ್ಣಾಂಕ i1 = ಅಂಶ_ct1.get_ಗುಂಪು(1);
    ಪೂರ್ಣಾಂಕ i2 = ಅಂಶ_ct1.get_ಗುಂಪು(0) % ne2;
    ಪೂರ್ಣಾಂಕ i3 = ಅಂಶ_ct1.get_ಗುಂಪು(0) / ne2;
    ಆದರೆ (i0 >= ne0 || i1 >= ne1 || i2 >= ne2 || i3 >= ne3) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int64_t dst_idx = i3*(ne0*ne1*ne2) + i2*(ne0*ne1) + i1*ne0 + i0;
    ಆದರೆ ((i0 >= lp0 && i0 < ne0 - rp0) &&
        (i1 >= lp1 && i1 < ne1 - rp1) &&
        (i2 >= lp2 && i2 < ne2 - rp2) &&
        (i3 >= lp3 && i3 < ne3 - rp3)) {
        ಸ್ಥಿರಾಂಕ int64_t i00 = i0 - lp0;
        ಸ್ಥಿರಾಂಕ int64_t i01 = i1 - lp1;
        ಸ್ಥಿರಾಂಕ int64_t i02 = i2 - lp2;
        ಸ್ಥಿರಾಂಕ int64_t i03 = i3 - lp3;

        ಸ್ಥಿರಾಂಕ int64_t src_idx = i03 * s03 + i02 * s02 + i01 * s01 + i00 * s00;

        dst[dst_idx] = src[src_idx];
    } ಇಲ್ಲದಿದ್ದರೆ {
        dst[dst_idx] = 0.0f;
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ pad_f32_sycl(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, size_t s00, size_t s01, size_t s02, size_t s03,
                         ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lp0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ rp0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lp1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ rp1,
                         ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lp2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ rp2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ lp3, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ rp3,
                         ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne2, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ne3,
                         dpct::ಸರತಿ_ptr stream) {
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಖಂಡಗಳು = (ne0 + SYCL_PAD_ಖಂಡ_ಗಾತ್ರ - 1) / SYCL_PAD_ಖಂಡ_ಗಾತ್ರ;
    sycl::range<3> grid(ne2 * ne3, ne1, ಅಂಕೆ_ಖಂಡಗಳು);
    stream->parallel_for(
        sycl::nd_range<3>(grid * sycl::range<3>(1, 1, SYCL_PAD_ಖಂಡ_ಗಾತ್ರ),
                          sycl::range<3>(1, 1, SYCL_PAD_ಖಂಡ_ಗಾತ್ರ)),
        [=](sycl::nd_ಅಂಶ<3> ಅಂಶ_ct1) {
            pad_f32(src, s00, s01, s02, s03, dst, lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                    ne0, ne1, ne2, ne3, ಅಂಶ_ct1);
        });
}

ಶೂನ್ಯ ggml_sycl_op_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    ಸ್ಥಿರಾಂಕ ggml_tensor * src0 = dst->src[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src0_d = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_d = (ದಶಮಾಂಶ *)dst->ದತ್ತಾಂಶ;
    dpct::ಸರತಿ_ptr stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type == GGML_TYPE_F32);

    ಸ್ಥಿರಾಂಕ size_t ts = ggml_type_ಗಾತ್ರ(src0->type);
    ಸ್ಥಿರಾಂಕ size_t s00 = src0->nb[0] / ts;
    ಸ್ಥಿರಾಂಕ size_t s01 = src0->nb[1] / ts;
    ಸ್ಥಿರಾಂಕ size_t s02 = src0->nb[2] / ts;
    ಸ್ಥಿರಾಂಕ size_t s03 = src0->nb[3] / ts;

    ಸ್ಥಿರಾಂಕ int32_t lp0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[0];
    ಸ್ಥಿರಾಂಕ int32_t rp0 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[1];
    ಸ್ಥಿರಾಂಕ int32_t lp1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[2];
    ಸ್ಥಿರಾಂಕ int32_t rp1 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[3];
    ಸ್ಥಿರಾಂಕ int32_t lp2 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[4];
    ಸ್ಥಿರಾಂಕ int32_t rp2 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[5];
    ಸ್ಥಿರಾಂಕ int32_t lp3 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[6];
    ಸ್ಥಿರಾಂಕ int32_t rp3 = ((ಸ್ಥಿರಾಂಕ int32_t *)(dst->op_params))[7];

    pad_f32_sycl(src0_d, s00, s01, s02, s03, dst_d,
                 lp0, rp0, lp1, rp1, lp2, rp2, lp3, rp3,
                 dst->ne[0], dst->ne[1], dst->ne[2], dst->ne[3], stream);
}

ಶೂನ್ಯ ggml_sycl_pad(ggml_backend_sycl_context & ctx, ggml_tensor * dst) {
    scope_op_debug_print scope_dbg_print(__func__, dst, /*num_src=*/1);
    ggml_sycl_op_pad(ctx, dst);
}
