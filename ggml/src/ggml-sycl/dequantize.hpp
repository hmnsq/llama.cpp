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

#ifndef GGML_SYCL_DEQUANTIZE_HPP
#define GGML_SYCL_DEQUANTIZE_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "convert.hpp"

ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ (*dequantize_kernel_t)(ಧ್ರುವ ಶೂನ್ಯ * vx, ಧ್ರುವ int64_t ib, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 & v);
ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ (*dequantize_kernel_t_reorder)(ಧ್ರುವ ಶೂನ್ಯ *d, ಧ್ರುವ int64_t ib, ಧ್ರುವ ಶೂನ್ಯ *qs,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v);
ಪ್ರಕಾರನಿರ್ದೇಶ ಶೂನ್ಯ (*dequantize_kernel_f32_t)(ಧ್ರುವ ಶೂನ್ಯ * vx, ಧ್ರುವ int64_t ib, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, ಭಿನ್ನ & v0, ಭಿನ್ನ & v1);

#ಯದಿ QK_K == 256
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ get_scale_min_k4(ಪೂರ್ಣಾಙ್ಕ j, ಧ್ರುವ uint8_t * q, uint8_t & d, uint8_t & m);
#endif

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q2_0(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
    ಧ್ರುವ block_q2_0 * x = (ಧ್ರುವ block_q2_0 *) vx;

    ಧ್ರುವ dfloat d = x[ib].d;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ byte_idx = iqs / 4;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ shift = (iqs % 4) * 2;
    ಧ್ರುವ uint8_t vui = x[ib].qs[byte_idx];

    v.x() = (vui >> shift) & 3;
    v.y() = (vui >> (shift + 2)) & 3;

#ifdef GGML_SYCL_F16
    v.s0() = ((dfloat)v.s0() - 1.0f) * d;
    v.s1() = ((dfloat)v.s1() - 1.0f) * d;
#ಅನ್ಯಥಾ
    v.x() = ((dfloat)v.x() - 1.0f) * d;
    v.y() = ((dfloat)v.y() - 1.0f) * d;
#endif // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q4_0(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
    ಧ್ರುವ block_q4_0 * x = (ಧ್ರುವ block_q4_0 *) vx;

    ಧ್ರುವ dfloat d = x[ib].d;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ifdef GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#ಅನ್ಯಥಾ
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#endif // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q4_0_reorder(ಧ್ರುವ ಶೂನ್ಯ *d_ptr, ಧ್ರುವ int64_t ib, ಧ್ರುವ ಶೂನ್ಯ *qs,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
    // const block_q4_0 * x = (const block_q4_0 *) vx;

    ಧ್ರುವ dfloat d = (ಧ್ರುವ dfloat)*((ಧ್ರುವ sycl::half*)d_ptr+ib);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vui = *((ಧ್ರುವ uint8_t *)qs+iqs);

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ifdef GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#ಅನ್ಯಥಾ
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#endif // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q1_0_reorder(ಧ್ರುವ ಶೂನ್ಯ *d_ptr, ಧ್ರುವ int64_t ib, ಧ್ರುವ ಶೂನ್ಯ *qs,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
    // Q1_0 reorder layout: scale values followed by quantized bits
    ಧ್ರುವ dfloat d = (ಧ್ರುವ dfloat)*((ಧ್ರುವ sycl::half*)d_ptr+ib);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bit_index_0 = iqs + 0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bit_index_1 = iqs + 1;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bit_0 = (*((ಧ್ರುವ uint8_t *)qs + bit_index_0 / 8) >> (bit_index_0 % 8)) & 1;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bit_1 = (*((ಧ್ರುವ uint8_t *)qs + bit_index_1 / 8) >> (bit_index_1 % 8)) & 1;

    v.x() = (2 * bit_0 - 1) * d;
    v.y() = (2 * bit_1 - 1) * d;
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q1_0(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
    ಧ್ರುವ block_q1_0 * x = (ಧ್ರುವ block_q1_0 *) vx;
    ಧ್ರುವ dfloat d = x[ib].d;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bit_index_0 = iqs + 0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bit_index_1 = iqs + 1;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bit_0 = (x[ib].qs[bit_index_0 / 8] >> (bit_index_0 % 8)) & 1;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bit_1 = (x[ib].qs[bit_index_1 / 8] >> (bit_index_1 % 8)) & 1;

    v.x() = (2 * bit_0 - 1) * d;
    v.y() = (2 * bit_1 - 1) * d;
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q4_1(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
    ಧ್ರುವ block_q4_1 * x = (ಧ್ರುವ block_q4_1 *) vx;

    ಧ್ರುವ dfloat d = x[ib].dm[0];
    ಧ್ರುವ dfloat m = x[ib].dm[1];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ifdef GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = sycl::fma(v.s0(), d, m);
    v.s1() = sycl::fma(v.s1(), d, m);

#ಅನ್ಯಥಾ
    v.x() = sycl::fma(v.x(), d, m);
    v.y() = sycl::fma(v.y(), d, m);
#endif // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q4_K(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_q4_K * x = (ಧ್ರುವ block_q4_K *) vx;
    ಧ್ರುವ sycl::half2 dm = x[ib].dm;
    ಧ್ರುವ ಭಿನ್ನ dall = dm[0];
    ಧ್ರುವ ಭಿನ್ನ dmin = dm[1];

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ il = idx / 64;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ in = idx % 64;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ is = 2 * il + (in >= 32 ? 1 : 0);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ off = in & 31;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ qsi = 32 * il + off;

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ಧ್ರುವ uint8_t q = x[ib].qs[qsi];
        ಧ್ರುವ uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);
        ಪ್ರತ್ಯಾಗಮ sycl::fma((dfloat) qv, (dfloat) (dall * sc), (dfloat) (-dmin * m));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("Q4_K dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q4_K_f32(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, ಭಿನ್ನ &v0, ಭಿನ್ನ &v1) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_q4_K * x = (ಧ್ರುವ block_q4_K *) vx;
    ಧ್ರುವ sycl::half2 dm = x[ib].dm;
    ಧ್ರುವ ಭಿನ್ನ dall = dm[0];
    ಧ್ರುವ ಭಿನ್ನ dmin = dm[1];

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> ಭಿನ್ನ {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ il = idx / 64;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ in = idx % 64;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ is = 2 * il + (in >= 32 ? 1 : 0);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ qsi = 32 * il + (in & 31);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ಧ್ರುವ ಭಿನ್ನ d = dall * sc;
        ಧ್ರುವ ಭಿನ್ನ mn = dmin * m;
        ಧ್ರುವ uint8_t q = x[ib].qs[qsi];
        ಧ್ರುವ uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);

        ಪ್ರತ್ಯಾಗಮ d * qv - mn;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("Q4_K dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q2_K(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_q2_K * x = (ಧ್ರುವ block_q2_K *) vx;
    ಧ್ರುವ ಭಿನ್ನ dall = x[ib].dm[0];
    ಧ್ರುವ ಭಿನ್ನ dmin = x[ib].dm[1];

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n = idx / 128;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ r = idx % 128;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ g = r / 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ l = r % 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ is = 8 * n + l / 16;

        ಧ್ರುವ uint8_t q = x[ib].qs[32 * n + l];
        ಧ್ರುವ uint8_t sc = x[ib].scales[is + 2 * g];
        ಧ್ರುವ ಭಿನ್ನ d = dall * (sc & 0xF);
        ಧ್ರುವ ಭಿನ್ನ m = dmin * (sc >> 4);

        ಪ್ರತ್ಯಾಗಮ (dfloat) d * (dfloat) ((q >> (2 * g)) & 3) - (dfloat) m;
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("Q2_K dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q2_K_f32(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, ಭಿನ್ನ &v0, ಭಿನ್ನ &v1) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_q2_K * x = (ಧ್ರುವ block_q2_K *) vx;
    ಧ್ರುವ ಭಿನ್ನ dall = x[ib].dm[0];
    ಧ್ರುವ ಭಿನ್ನ dmin = x[ib].dm[1];

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> ಭಿನ್ನ {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n = idx / 128;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ r = idx % 128;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ g = r / 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ l = r % 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ is = 8 * n + l / 16;

        ಧ್ರುವ uint8_t q = x[ib].qs[32 * n + l];
        ಧ್ರುವ uint8_t sc = x[ib].scales[is + 2 * g];
        ಧ್ರುವ ಭಿನ್ನ d = dall * (sc & 0xF);
        ಧ್ರುವ ಭಿನ್ನ m = dmin * (sc >> 4);

        ಪ್ರತ್ಯಾಗಮ d * ((q >> (2 * g)) & 3) - m;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("Q2_K dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q3_K(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_q3_K * x = (ಧ್ರುವ block_q3_K *) vx;
    ಧ್ರುವ ಭಿನ್ನ d_all = x[ib].d;

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ n = idx / 128;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ r = idx % 128;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = r / 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ l = r % 32;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ is0 = l / 16;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ is = 8 * n + 2 * j + is0;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ shift = 2 * j;
        ಧ್ರುವ uint8_t m = 1 << (4 * n + j);

        ಧ್ರುವ int8_t us = is <  4 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 8] >> 0) & 3) << 4) :
                         is <  8 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 4] >> 2) & 3) << 4) :
                         is < 12 ? (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is + 0] >> 4) & 3) << 4) :
                                   (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is - 4] >> 6) & 3) << 4);

        ಧ್ರುವ ಭಿನ್ನ dl = d_all * (us - 32);
        ಧ್ರುವ uint8_t q = x[ib].qs[32 * n + l];
        ಧ್ರುವ uint8_t h = x[ib].hmask[l];
        ಧ್ರುವ int8_t qv = ((q >> shift) & 3) - ((h & m) ? 0 : 4);

        ಪ್ರತ್ಯಾಗಮ (dfloat) (dl * qv);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("Q3_K dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q5_K(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_q5_K * x = (ಧ್ರುವ block_q5_K *) vx;
    ಧ್ರುವ ಭಿನ್ನ dall = x[ib].dm[0];
    ಧ್ರುವ ಭಿನ್ನ dmin = x[ib].dm[1];

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ il = idx / 64;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ in = idx % 64;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ is = 2 * il + (in >= 32 ? 1 : 0);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ir = (in & 31) / 2;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iq = in & 1;

        ಧ್ರುವ uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        ಧ್ರುವ uint8_t h = x[ib].qh[2 * ir + iq];
        ಧ್ರುವ uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ಧ್ರುವ ಭಿನ್ನ d = dall * sc;
        ಧ್ರುವ ಭಿನ್ನ mn = dmin * m;
        ಧ್ರುವ uint8_t hm = 1 << (2 * il + (in >= 32 ? 1 : 0));

        ಪ್ರತ್ಯಾಗಮ sycl::fma((dfloat) (qv + ((h & hm) ? 16 : 0)), (dfloat) d, (dfloat) (-mn));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("Q5_K dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q5_K_f32(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, ಭಿನ್ನ &v0, ಭಿನ್ನ &v1) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_q5_K * x = (ಧ್ರುವ block_q5_K *) vx;
    ಧ್ರುವ ಭಿನ್ನ dall = x[ib].dm[0];
    ಧ್ರುವ ಭಿನ್ನ dmin = x[ib].dm[1];

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> ಭಿನ್ನ {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ il = idx / 64;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ in = idx % 64;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ is = 2 * il + (in >= 32 ? 1 : 0);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ir = (in & 31) / 2;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iq = in & 1;

        ಧ್ರುವ uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        ಧ್ರುವ uint8_t h = x[ib].qh[2 * ir + iq];
        ಧ್ರುವ uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ಧ್ರುವ ಭಿನ್ನ d = dall * sc;
        ಧ್ರುವ ಭಿನ್ನ mn = dmin * m;
        ಧ್ರುವ uint8_t hm = 1 << (2 * il + (in >= 32 ? 1 : 0));

        ಪ್ರತ್ಯಾಗಮ (qv + ((h & hm) ? 16 : 0)) * d - mn;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("Q5_K dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q6_K(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_q6_K * x = (ಧ್ರುವ block_q6_K *) vx;
    ಧ್ರುವ ಭಿನ್ನ d = x[ib].d;

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ip = idx / 128;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ in = idx % 128;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ il = in & 31;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ig = in / 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ is = 8 * ip + il / 16;

        ಧ್ರುವ uint8_t ql0 = x[ib].ql[64 * ip + il];
        ಧ್ರುವ uint8_t ql1 = x[ib].ql[64 * ip + il + 32];
        ಧ್ರುವ uint8_t qh = x[ib].qh[32 * ip + il];
        ಧ್ರುವ int8_t * sc = x[ib].scales + is;

        uint8_t qv;
        int8_t scale;
        ಯದಿ (ig == 0) {
            qv = (ql0 & 0xF) | (((qh >> 0) & 3) << 4);
            scale = sc[0];
        } ಅನ್ಯಥಾ ಯದಿ (ig == 1) {
            qv = (ql1 & 0xF) | (((qh >> 2) & 3) << 4);
            scale = sc[2];
        } ಅನ್ಯಥಾ ಯದಿ (ig == 2) {
            qv = (ql0 >> 4) | (((qh >> 4) & 3) << 4);
            scale = sc[4];
        } ಅನ್ಯಥಾ {
            qv = (ql1 >> 4) | (((qh >> 6) & 3) << 4);
            scale = sc[6];
        }

        ಪ್ರತ್ಯಾಗಮ (dfloat) (d * scale * ((int8_t) qv - 32));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("Q6_K dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_mxfp4(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
    ಧ್ರುವ block_mxfp4 * x = (ಧ್ರುವ block_mxfp4 *) vx;
    ಧ್ರುವ ಭಿನ್ನ d = ggml_sycl_e8m0_to_fp32(x[ib].e);
    ಧ್ರುವ uint8_t q = x[ib].qs[iqs];

    v.x() = d * kvalues_mxfp4[q & 0xF] * 0.5f;
    v.y() = d * kvalues_mxfp4[q >> 4] * 0.5f;
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_nvfp4(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
    ಧ್ರುವ block_nvfp4 & xb = ((ಧ್ರುವ block_nvfp4 *) vx)[ib];

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sub = idx / QK_NVFP4_SUB;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = idx % QK_NVFP4_SUB;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ jh = j % (QK_NVFP4_SUB / 2);

        ಧ್ರುವ ಭಿನ್ನ d = ggml_sycl_ue4m3_to_fp32(xb.d[sub]);
        ಧ್ರುವ uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + jh];
        ಧ್ರುವ uint8_t qv = (j < (QK_NVFP4_SUB / 2)) ? (q & 0x0F) : (q >> 4);

        ಪ್ರತ್ಯಾಗಮ d * kvalues_mxfp4[qv];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq2_xxs(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                               ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq2_xxs * x = (ಧ್ರುವ block_iq2_xxs *) vx;

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib8 = idx / 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ r = idx % 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ il = r / 8;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = r % 8;

        ಧ್ರುವ uint16_t * q2 = x[ib].qs + 4 * ib8;
        ಧ್ರುವ uint8_t * aux8 = (ಧ್ರುವ uint8_t *) q2;
        ಧ್ರುವ uint8_t * grid = (ಧ್ರುವ uint8_t *) (iq2xxs_grid + aux8[il]);
        ಧ್ರುವ uint32_t aux32 = q2[2] | (q2[3] << 16);
        ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ) x[ib].d * (0.5f + (aux32 >> 28)) * 0.25f;
        ಧ್ರುವ uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        ಪ್ರತ್ಯಾಗಮ d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("IQ2_XXS dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq2_xs(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                              ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq2_xs * x = (ಧ್ರುವ block_iq2_xs *) vx;

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib8 = idx / 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ r = idx % 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ il = r / 8;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = r % 8;

        ಧ್ರುವ uint16_t * q2 = x[ib].qs + 4 * ib8;
        ಧ್ರುವ uint8_t * grid = (ಧ್ರುವ uint8_t *) (iq2xs_grid + (q2[il] & 511));
        ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ಧ್ರುವ uint8_t signs = ksigns_iq2xs[q2[il] >> 9];

        ಪ್ರತ್ಯಾಗಮ d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("IQ2_XS dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq2_s(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq2_s * x = (ಧ್ರುವ block_iq2_s *) vx;

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib8 = idx / 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ r = idx % 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ il = r / 8;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = r % 8;

        ಧ್ರುವ uint16_t grid_id = x[ib].qs[4 * ib8 + il] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 0x300);
        ಧ್ರುವ uint8_t * grid = (ಧ್ರುವ uint8_t *) (iq2s_grid + grid_id);
        ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ಧ್ರುವ uint8_t signs = x[ib].qs[QK_K / 8 + 4 * ib8 + il];

        ಪ್ರತ್ಯಾಗಮ d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("IQ2_S dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq3_xxs(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                               ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq3_xxs * x = (ಧ್ರುವ block_iq3_xxs *) vx;

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib8 = idx / 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ r = idx % 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ il = r / 8;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = r % 8;

        ಧ್ರುವ uint8_t * q3 = x[ib].qs + 8 * ib8;
        ಧ್ರುವ uint16_t * gas = (ಧ್ರುವ uint16_t *) (x[ib].qs + QK_K / 4) + 2 * ib8;
        ಧ್ರುವ uint8_t * grid1 = (ಧ್ರುವ uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        ಧ್ರುವ uint8_t * grid2 = (ಧ್ರುವ uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        ಧ್ರುವ uint32_t aux32 = gas[0] | (gas[1] << 16);
        ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ) x[ib].d * (0.5f + (aux32 >> 28)) * 0.5f;
        ಧ್ರುವ uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        ಯದಿ (j < 4) {
            ಪ್ರತ್ಯಾಗಮ d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        ಪ್ರತ್ಯಾಗಮ d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("IQ3_XXS dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq3_s(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq3_s * x = (ಧ್ರುವ block_iq3_s *) vx;

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib8 = idx / 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ r = idx % 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ il = r / 8;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = r % 8;

        ಧ್ರುವ uint8_t * qs = x[ib].qs + 8 * ib8;
        ಧ್ರುವ uint16_t grid1_id = qs[2 * il + 0] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 256);
        ಧ್ರುವ uint16_t grid2_id = qs[2 * il + 1] | ((x[ib].qh[ib8] << (7 - 2 * il)) & 256);
        ಧ್ರುವ uint8_t * grid1 = (ಧ್ರುವ uint8_t *) (iq3s_grid + grid1_id);
        ಧ್ರುವ uint8_t * grid2 = (ಧ್ರುವ uint8_t *) (iq3s_grid + grid2_id);
        ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ) x[ib].d * (1 + 2 * ((x[ib].scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
        ಧ್ರುವ uint8_t signs = x[ib].signs[4 * ib8 + il];

        ಯದಿ (j < 4) {
            ಪ್ರತ್ಯಾಗಮ d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        ಪ್ರತ್ಯಾಗಮ d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("IQ3_S dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq1_s(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq1_s * x = (ಧ್ರುವ block_iq1_s *) vx;

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib8 = idx / 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ r = idx % 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ il = r / 8;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = r % 8;

        ಧ್ರುವ ಭಿನ್ನ delta = (x[ib].qh[ib8] & 0x8000) ? (-1.f - IQ1S_DELTA) : (-1.f + IQ1S_DELTA);
        ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ) x[ib].d * (2 * ((x[ib].qh[ib8] >> 12) & 7) + 1);
        ಧ್ರುವ uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((x[ib].qh[ib8] >> (3 * il)) & 7) << 8);
        ಧ್ರುವ uint32_t g = iq1s_grid_gpu[grid_id];
        ಧ್ರುವ int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        ಪ್ರತ್ಯಾಗಮ d * (qv + delta);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("IQ1_S dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq1_m(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq1_m * x = (ಧ್ರುವ block_iq1_m *) vx;

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib8 = idx / 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ r = idx % 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ il = r / 8;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = r % 8;

        ಧ್ರುವ uint16_t * sc = (ಧ್ರುವ uint16_t *) x[ib].scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib16 = 2 * ib8 + il / 2;
        ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);

        ಧ್ರುವ uint8_t qh = x[ib].qh[2 * ib8 + il / 2];
        ಧ್ರುವ ಭಿನ್ನ delta = (qh & (0x08 << (4 * (il % 2)))) ? (-1.f - IQ1M_DELTA) : (-1.f + IQ1M_DELTA);

        ಧ್ರುವ uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((qh >> (4 * (il % 2))) & 7) << 8);
        ಧ್ರುವ uint32_t g = iq1s_grid_gpu[grid_id];
        ಧ್ರುವ int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        ಪ್ರತ್ಯಾಗಮ d * (qv + delta);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("IQ1_M dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq4_nl(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                              ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
    ಧ್ರುವ block_iq4_nl * x = (ಧ್ರುವ block_iq4_nl *) vx;
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ) x[ib].d;

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಯದಿ (idx < 16) {
            ಪ್ರತ್ಯಾಗಮ d * kvalues_iq4nl[x[ib].qs[idx] & 0xF];
        }
        ಪ್ರತ್ಯಾಗಮ d * kvalues_iq4nl[x[ib].qs[idx - 16] >> 4];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq4_xs(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                              ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq4_xs * x = (ಧ್ರುವ block_iq4_xs *) vx;

    auto dequantize_one = [&](ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ idx) -> dfloat {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib8 = idx / 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ r = idx % 32;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ byte_idx = (r < 16) ? r : (r - 16);
        ಧ್ರುವ uint8_t q = x[ib].qs[16 * ib8 + byte_idx];
        ಧ್ರುವ uint8_t qv = (r < 16) ? (q & 0x0F) : (q >> 4);

        ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ) x[ib].d * ((((x[ib].scales_l[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf) |
                        (((x[ib].scales_h >> (2 * ib8)) & 3) << 4)) - 32);
        ಪ್ರತ್ಯಾಗಮ d * kvalues_iq4nl[qv];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಅನ್ಯಥಾ
    GGML_ABORT("IQ4_XS dequantize not supported for QK_K != 256");
#endif
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q5_0(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
    ಧ್ರುವ block_q5_0 * x = (ಧ್ರುವ block_q5_0 *) vx;

    ಧ್ರುವ dfloat d = x[ib].d;

    uint32_t qh;
    memcpy(&qh, x[ib].qh, ಪರಿಮಾಣ(qh));

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#ifdef GGML_SYCL_F16
    // v = v - {16.0f, 16.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 16.0f) * d;
    v.s1() = (v.s1() - 16.0f) * d;

#ಅನ್ಯಥಾ
    v.x() = (v.x() - 16.0f) * d;
    v.y() = (v.y() - 16.0f) * d;
#endif // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q5_1(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
    ಧ್ರುವ block_q5_1 * x = (ಧ್ರುವ block_q5_1 *) vx;

    ಧ್ರುವ dfloat d = x[ib].dm[0];
    ಧ್ರುವ dfloat m = x[ib].dm[1];

    uint32_t qh;
    memcpy(&qh, x[ib].qh, ಪರಿಮಾಣ(qh));

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#ifdef GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = sycl::fma(v.s0(), d, m);
    v.s1() = sycl::fma(v.s1(), d, m);
#ಅನ್ಯಥಾ
    v.x() = sycl::fma(v.x(), d, m);
    v.y() = sycl::fma(v.y(), d, m);
#endif // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q8_0_reorder(ಧ್ರುವ ಶೂನ್ಯ *d_ptr, ಧ್ರುವ int64_t ib, ಧ್ರುವ ಶೂನ್ಯ *qs,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
    ಧ್ರುವ dfloat d = (ಧ್ರುವ dfloat)*((ಧ್ರುವ sycl::half*)d_ptr + ib);

    v.x() = ((ಧ್ರುವ int8_t *)qs)[iqs + 0];
    v.y() = ((ಧ್ರುವ int8_t *)qs)[iqs + 1];

#ifdef GGML_SYCL_F16
    v.s0() *= d;
    v.s1() *= d;
#ಅನ್ಯಥಾ
    v.x() *= d;
    v.y() *= d;
#endif // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q8_0(ಧ್ರುವ ಶೂನ್ಯ *vx, ಧ್ರುವ int64_t ib,
                                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs, dfloat2 &v) {
    ಧ್ರುವ block_q8_0 * x = (ಧ್ರುವ block_q8_0 *) vx;

    ಧ್ರುವ dfloat d = x[ib].d;

    v.x() = x[ib].qs[iqs + 0];
    v.y() = x[ib].qs[iqs + 1];

#ifdef GGML_SYCL_F16
    // v = v * {d, d};
    v.s0() *= d;
    v.s1() *= d;
#ಅನ್ಯಥಾ
    v.x() *= d;
    v.y() *= d;
#endif // GGML_SYCL_F16
}

template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q4_0(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  ಧ್ರುವ sycl::nd_item<3> &item_ct1) {

    ಧ್ರುವ int64_t i = item_ct1.get_group(2);

    // assume 32 threads
    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
    ಧ್ರುವ int64_t il  = tid/8;
    ಧ್ರುವ int64_t ir  = tid%8;
    ಧ್ರುವ int64_t ib = 8*i + ir;
    ಯದಿ (ib >= nb32) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    ಧ್ರುವ block_q4_0 * x = (ಧ್ರುವ block_q4_0 *)vx + ib;
    ಧ್ರುವ ಭಿನ್ನ d = sycl::vec<sycl::half, 1>(x->d)
                        .convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>()[0];
    ಧ್ರುವ ಭಿನ್ನ dm = -8*d;

    ಧ್ರುವ uint8_t * q = x->qs + 4*il;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q4_0_reorder(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                  ಧ್ರುವ sycl::nd_item<3> &item_ct1) {
    // one thread per qs byte, like dequantize_block<dequantize_q4_0>; the reorder layout makes the byte reads contiguous
    // grid-stride loop because the launcher may shrink the work-group to keep the global range below INT_MAX
    // float math, not dequantize_q4_0_reorder: that helper computes in half with GGML_SYCL_F16, which changes f32 output
    ಧ್ರುವ sycl::half * d_ptr  = (ಧ್ರುವ sycl::half *) ((ಧ್ರುವ uint8_t *) vx + k/2);
    ಧ್ರುವ int64_t      stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    ಪ್ರತಿ (int64_t iq = item_ct1.get_global_id(2); iq < k/2; iq += stride) {
        ಧ್ರುವ int64_t ib  = iq / (QK4_0/2);
        ಧ್ರುವ int64_t iqs = iq % (QK4_0/2);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     vq  = ((ಧ್ರುವ uint8_t *) vx)[iq];
        ಧ್ರುವ ಭಿನ್ನ   d   = ಭಿನ್ನ(d_ptr[ib]);

        dst_t * y = yy + ib*QK4_0 + iqs;
        y[0]       = d * ((vq & 0xF) - 8);
        y[QK4_0/2] = d * ((vq >>  4) - 8);
    }
}

// Dequantize Q8_0 from reorder layout: [all qs (k bytes)][all d values]
template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q8_0_reorder(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                  ಧ್ರುವ sycl::nd_item<3> &item_ct1) {
    // two values per thread, like dequantize_block<dequantize_q8_0>; the reorder layout keeps all qs contiguous
    // grid-stride loop because the launcher may shrink the work-group to keep the global range below INT_MAX
    // float math, not dequantize_q8_0_reorder: that helper computes in half with GGML_SYCL_F16, which changes f32 output
    ಧ್ರುವ int8_t *     qs     = (ಧ್ರುವ int8_t *) vx;
    ಧ್ರುವ sycl::half * d_ptr  = (ಧ್ರುವ sycl::half *) ((ಧ್ರುವ uint8_t *) vx + k);
    ಧ್ರುವ int64_t      stride = 2 * item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    ಪ್ರತಿ (int64_t i = 2 * item_ct1.get_global_id(2); i < k; i += stride) {
        ಧ್ರುವ ಭಿನ್ನ d = ಭಿನ್ನ(d_ptr[i / QK8_0]);

        yy[i + 0] = d * qs[i + 0];
        yy[i + 1] = d * qs[i + 1];
    }
}

template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q8_0_reorder_wide(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                               ಧ್ರುವ sycl::id<1> idx) {
    ಧ್ರುವ int64_t i = 4 * (int64_t) idx[0];

    ಧ್ರುವ int8_t *     qs    = (ಧ್ರುವ int8_t *) vx;
    ಧ್ರುವ sycl::half * d_ptr = (ಧ್ರುವ sycl::half *) ((ಧ್ರುವ uint8_t *) vx + k);

    ಧ್ರುವ ಭಿನ್ನ d0 = ಭಿನ್ನ(d_ptr[i / QK8_0]);
    ಧ್ರುವ ಭಿನ್ನ d1 = ಭಿನ್ನ(d_ptr[(i + 2) / QK8_0]);

    sycl::vec<ಭಿನ್ನ, 4> p;
    p[0] = d0 * (ಭಿನ್ನ) qs[i + 0];
    p[1] = d0 * (ಭಿನ್ನ) qs[i + 1];
    p[2] = d1 * (ಭಿನ್ನ) qs[i + 2];
    p[3] = d1 * (ಭಿನ್ನ) qs[i + 3];

    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yy + i) = p.convert<dst_t, sycl::rounding_mode::automatic>();
}

template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q4_1(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  ಧ್ರುವ sycl::nd_item<3> &item_ct1) {

    ಧ್ರುವ int64_t i = item_ct1.get_group(2);

    // assume 32 threads
    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
    ಧ್ರುವ int64_t il  = tid/8;
    ಧ್ರುವ int64_t ir  = tid%8;
    ಧ್ರುವ int64_t ib = 8*i + ir;
    ಯದಿ (ib >= nb32) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    ಧ್ರುವ block_q4_1 * x = (ಧ್ರುವ block_q4_1 *)vx + ib;
    ಧ್ರುವ sycl::float2 d =
        x->dm.convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();

    ಧ್ರುವ uint8_t * q = x->qs + 4*il;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 4; ++l) {
        y[l + 0] = d.x() * (q[l] & 0xF) + d.y();
        y[l + 16] = d.x() * (q[l] >> 4) + d.y();
    }
}


//================================== k-quants

template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q2_K(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy,
                                  ಧ್ರುವ sycl::nd_item<3> &item_ct1) {

    ಧ್ರುವ int64_t i = item_ct1.get_group(2);
    ಧ್ರುವ block_q2_K * x = (ಧ್ರುವ block_q2_K *) vx;

    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
#ಯದಿ QK_K == 256
    ಧ್ರುವ int64_t n   = tid/32;
    ಧ್ರುವ int64_t l   = tid - 32*n;
    ಧ್ರುವ int64_t is  = 8*n + l/16;

    ಧ್ರುವ uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    ಭಿನ್ನ dall = x[i].dm[0];
    ಭಿನ್ನ dmin = x[i].dm[1];
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#ಅನ್ಯಥಾ
    ಧ್ರುವ int64_t is = tid/16;  // 0 or 1
    ಧ್ರುವ int64_t il = tid%16;  // 0...15
    ಧ್ರುವ uint8_t q = x[i].qs[il] >> (2*is);
    dst_t * y = yy + i*QK_K + 16*is + il;

    ಭಿನ್ನ dall = x[i].dm[0];
    ಭಿನ್ನ dmin = x[i].dm[1];
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#endif

}

template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q2_K_reorder(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, ಧ್ರುವ int64_t i,
                                          ಧ್ರುವ sycl::nd_item<3> & item_ct1, int64_t n_blocks) {
#ಯದಿ QK_K == 256
    ಯದಿ (i >= n_blocks) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ uint8_t * base          = static_cast<ಧ್ರುವ uint8_t *>(vx);
    ಧ್ರುವ size_t    qs_offset     = i * (QK_K / 4);
    ಧ್ರುವ size_t    scales_offset = n_blocks * (QK_K / 4) + i * (QK_K / 16);
    ಧ್ರುವ size_t    dm_offset     = n_blocks * (QK_K / 4) + n_blocks * (QK_K / 16) + i * ಪರಿಮಾಣ(ggml_half2);

    ಧ್ರುವ uint8_t *     qs     = base + qs_offset;
    ಧ್ರುವ uint8_t *     scales = base + scales_offset;
    ಧ್ರುವ ggml_half2 * dm     = reinterpret_cast<ಧ್ರುವ ggml_half2 *>(base + dm_offset);

    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
    ಧ್ರುವ int64_t n    = tid / 32;
    ಧ್ರುವ int64_t l    = tid - 32 * n;
    ಧ್ರುವ int64_t is   = 8 * n + l / 16;

    ಧ್ರುವ uint8_t q = qs[32 * n + l];
    dst_t * y = yy + 128 * n;

    ಧ್ರುವ ಭಿನ್ನ dall = (*dm)[0];
    ಧ್ರುವ ಭಿನ್ನ dmin = (*dm)[1];
    y[l+ 0] = dall * (scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (scales[is+0] >> 4);
    y[l+32] = dall * (scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (scales[is+2] >> 4);
    y[l+64] = dall * (scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (scales[is+4] >> 4);
    y[l+96] = dall * (scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (scales[is+6] >> 4);
#ಅನ್ಯಥಾ
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(i);
    GGML_UNUSED(item_ct1);
    GGML_UNUSED(n_blocks);
    GGML_ABORT("Q2_K reorder dequantize not supported for QK_K != 256");
#endif
}

template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q3_K(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy,
                                  ಧ್ರುವ sycl::nd_item<3> &item_ct1) {

    ಧ್ರುವ int64_t i = item_ct1.get_group(2);
    ಧ್ರುವ block_q3_K * x = (ಧ್ರುವ block_q3_K *) vx;

#ಯದಿ QK_K == 256
    ಧ್ರುವ int64_t r = item_ct1.get_local_id(2) / 4;
    ಧ್ರುವ int64_t tid = r/2;
    ಧ್ರುವ int64_t is0 = r%2;
    ಧ್ರುವ int64_t l0 = 16 * is0 + 4 * (item_ct1.get_local_id(2) % 4);
    ಧ್ರುವ int64_t n = tid / 4;
    ಧ್ರುವ int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    ಪೂರ್ಣಾಙ್ಕ shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    ಭಿನ್ನ d_all = x[i].d;
    ಭಿನ್ನ dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    ಧ್ರುವ uint8_t * q = x[i].qs + 32*n;
    ಧ್ರುವ uint8_t * hm = x[i].hmask;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
#ಅನ್ಯಥಾ
    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
    ಧ್ರುವ int64_t is  = tid/16;  // 0 or 1
    ಧ್ರುವ int64_t il  = tid%16;  // 0...15
    ಧ್ರುವ int64_t im  = il/8;    // 0...1
    ಧ್ರುವ int64_t in  = il%8;    // 0...7

    dst_t * y = yy + i*QK_K + 16*is + il;

    ಧ್ರುವ uint8_t q = x[i].qs[il] >> (2*is);
    ಧ್ರುವ uint8_t h = x[i].hmask[in] >> (2*is + im);
    ಧ್ರುವ ಭಿನ್ನ   d = (ಭಿನ್ನ)x[i].d;

    ಯದಿ (is == 0) {
        y[ 0] = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    } ಅನ್ಯಥಾ {
        y[ 0] = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    }
#endif

}

template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q3_K_reorder(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, ಧ್ರುವ int64_t i,
                                          ಧ್ರುವ sycl::nd_item<3> & item_ct1, int64_t n_blocks) {
#ಯದಿ QK_K == 256
    ಯದಿ (i >= n_blocks) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ uint8_t * base          = static_cast<ಧ್ರುವ uint8_t *>(vx);
    ಧ್ರುವ size_t    qs_offset     = i * (QK_K / 4);
    ಧ್ರುವ size_t    hmask_offset  = n_blocks * (QK_K / 4) + i * (QK_K / 8);
    ಧ್ರುವ size_t    scales_offset = n_blocks * (QK_K / 4) + n_blocks * (QK_K / 8) + i * 12;
    ಧ್ರುವ size_t    d_offset      = n_blocks * (QK_K / 4) + n_blocks * (QK_K / 8) + n_blocks * 12 +
                                 i * ಪರಿಮಾಣ(ggml_half);

    ಧ್ರುವ uint8_t * qs     = base + qs_offset;
    ಧ್ರುವ uint8_t * hmask  = base + hmask_offset;
    ಧ್ರುವ uint8_t * scales = base + scales_offset;
    ಧ್ರುವ ಭಿನ್ನ     d_all  = static_cast<ಭಿನ್ನ>(*reinterpret_cast<ಧ್ರುವ ggml_half *>(base + d_offset));

    ಧ್ರುವ int64_t r    = item_ct1.get_local_id(2) / 4;
    ಧ್ರುವ int64_t tid  = r / 2;
    ಧ್ರುವ int64_t is0  = r % 2;
    ಧ್ರುವ int64_t l0   = 16 * is0 + 4 * (item_ct1.get_local_id(2) % 4);
    ಧ್ರುವ int64_t n    = tid / 4;
    ಧ್ರುವ int64_t j    = tid - 4 * n;
    ಧ್ರುವ int64_t is   = 8 * n + 2 * j + is0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     shift = 2 * j;
    uint8_t       m    = 1 << (4 * n + j);

    uint8_t us = is < 4
        ? (scales[is - 0] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4)
        : is < 8
            ? (scales[is - 0] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4)
            : is < 12
                ? (scales[is - 8] >> 4) | (((scales[is + 0] >> 4) & 3) << 4)
                : (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);

    ಧ್ರುವ ಭಿನ್ನ dl = d_all * (us - 32);

    dst_t * y = yy + 128 * n + 32 * j;
    ಧ್ರುವ uint8_t * q  = qs + 32 * n;
    ಧ್ರುವ uint8_t * hm = hmask;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = l0; l < l0 + 4; ++l) {
        y[l] = dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
    }
#ಅನ್ಯಥಾ
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(i);
    GGML_UNUSED(item_ct1);
    GGML_UNUSED(n_blocks);
    GGML_ABORT("Q3_K reorder dequantize not supported for QK_K != 256");
#endif
}

#ಯದಿ QK_K == 256
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ get_scale_min_k4(ಪೂರ್ಣಾಙ್ಕ j, ಧ್ರುವ uint8_t * q, uint8_t & d, uint8_t & m) {
    ಯದಿ (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } ಅನ್ಯಥಾ {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#endif

// Work-items that cover one 256-weight K-quant block: four 64-element chunks, 8 work-items
// per chunk, 4 qs bytes per work-item.
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಙ್ಕ SYCL_DEQK_WI_PER_BLOCK = 32;

template <typename dst_t>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ dequantize_q4_K_wide_one(dst_t * __restrict__ y, ಧ್ರುವ uint8_t * __restrict__ qs,
                                            ಧ್ರುವ uint8_t * __restrict__ scales, ಧ್ರುವ ಭಿನ್ನ dall,
                                            ಧ್ರುವ ಭಿನ್ನ dmin, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tid) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ il = tid >> 3;        // 64-element chunk, 0..3
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ir = (tid & 7) * 4;   // 4 qs bytes within the chunk

    uint8_t sc, m;
    get_scale_min_k4(2 * il + 0, scales, sc, m);
    ಧ್ರುವ ಭಿನ್ನ d1 = dall * sc;
    ಧ್ರುವ ಭಿನ್ನ m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, scales, sc, m);
    ಧ್ರುವ ಭಿನ್ನ d2 = dall * sc;
    ಧ್ರುವ ಭಿನ್ನ m2 = dmin * m;

    ಧ್ರುವ sycl::vec<uint8_t, 4> q = vec_aligned_load<uint8_t, 4>(qs + 32 * il + ir);

    sycl::vec<dst_t, 4> lo, hi;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 4; ++l) {
        lo[l] = d1 * (q[l] & 0xF) - m1;
        hi[l] = d2 * (q[l] >> 4)  - m2;
    }
    dst_t * yb = y + 64 * il + ir;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb +  0) = lo;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb + 32) = hi;
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q4_K_wide(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb,
                                       ಧ್ರುವ sycl::nd_item<1> & it) {
    static_assert(QK_K == 256, "wide K-quant dequant assumes QK_K == 256");
    ಧ್ರುವ int64_t g   = it.get_global_id(0);
    ಧ್ರುವ int64_t b   = g / SYCL_DEQK_WI_PER_BLOCK;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     tid = (ಪೂರ್ಣಾಙ್ಕ) (g % SYCL_DEQK_WI_PER_BLOCK);
    ಯದಿ (b >= nb) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಧ್ರುವ block_q4_K * x = (ಧ್ರುವ block_q4_K *) vx + b;
    dequantize_q4_K_wide_one(yy + b * QK_K, x->qs, x->scales, (ಭಿನ್ನ) x->dm[0], (ಭಿನ್ನ) x->dm[1], tid);
}


template <typename dst_t>
ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ dequantize_q4_K_common(dst_t * __restrict__ y, ಧ್ರುವ uint8_t * __restrict__ qs_ptr, ಧ್ರುವ ಭಿನ್ನ dall,
                                   ಧ್ರುವ ಭಿನ್ನ dmin, uint8_t * __restrict__ scales_local, ಪೂರ್ಣಾಙ್ಕ il, ಪೂರ್ಣಾಙ್ಕ ir) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ is = 2 * il;
    constexpr ಪೂರ್ಣಾಙ್ಕ n  = 4;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    ಧ್ರುವ ಭಿನ್ನ d1 = dall * sc;
    ಧ್ರುವ ಭಿನ್ನ m1 = dmin * m;

    get_scale_min_k4(is + 1, scales_local, sc, m);
    ಧ್ರುವ ಭಿನ್ನ d2 = dall * sc;
    ಧ್ರುವ ಭಿನ್ನ m2 = dmin * m;

    sycl::vec<uint8_t, n> q_vec = vec_aligned_load<uint8_t, n>(qs_ptr + 32 * il + n * ir);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < n; ++l) {
        y[l + 0]  = d1 * (q_vec[l] & 0xF) - m1;
        y[l + 32] = d2 * (q_vec[l] >> 4) - m2;
    }
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q4_K_reorder(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, ಧ್ರುವ int64_t i,
                                          uint8_t * scales_local, ಧ್ರುವ sycl::nd_item<1> & item_ct1, int64_t nb) {
    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(0);  // thread index within block
    ಧ್ರುವ int64_t il  = tid / 8;
    ಧ್ರುವ int64_t ir  = tid % 8;

    dst_t * y = yy + 64 * il + 4 * ir;

    ಧ್ರುವ uint8_t * base          = static_cast<ಧ್ರುವ uint8_t *>(vx);
    ಧ್ರುವ size_t    qs_offset     = i * (QK_K / 2);
    ಧ್ರುವ size_t    scales_offset = nb * (QK_K / 2) + i * K_SCALE_SIZE;
    ಧ್ರುವ size_t    dm_offset     = nb * (QK_K / 2) + nb * K_SCALE_SIZE + i * ಪರಿಮಾಣ(ggml_half2);

    ಧ್ರುವ uint8_t *    qs_ptr     = base + qs_offset;
    ಧ್ರುವ uint8_t *    scales_ptr = base + scales_offset;
    ggml_half2         dm_ಮೌಲ್ಯಗಳು  = *reinterpret_cast<ಧ್ರುವ ggml_half2 *>(base + dm_offset);

    ಧ್ರುವ ಭಿನ್ನ dall = dm_ಮೌಲ್ಯಗಳು.x();
    ಧ್ರುವ ಭಿನ್ನ dmin = dm_ಮೌಲ್ಯಗಳು.y();

    ಯದಿ (tid < 12) {
        scales_local[tid] = scales_ptr[tid];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
    dequantize_q4_K_common(y, qs_ptr, dall, dmin, scales_local, il, ir);
}

template <typename dst_t>
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ dequantize_q5_K_wide_one(dst_t * __restrict__ y, ಧ್ರುವ uint8_t * __restrict__ qs,
                                            ಧ್ರುವ uint8_t * __restrict__ qh_ptr,
                                            ಧ್ರುವ uint8_t * __restrict__ scales, ಧ್ರುವ ಭಿನ್ನ dall,
                                            ಧ್ರುವ ಭಿನ್ನ dmin, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ tid) {
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ il = tid >> 3;        // 64-element chunk, 0..3
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ir = (tid & 7) * 4;   // 4 qs bytes within the chunk; qh holds the 5th bit

    uint8_t sc, m;
    get_scale_min_k4(2 * il + 0, scales, sc, m);
    ಧ್ರುವ ಭಿನ್ನ d1 = dall * sc;
    ಧ್ರುವ ಭಿನ್ನ m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, scales, sc, m);
    ಧ್ರುವ ಭಿನ್ನ d2 = dall * sc;
    ಧ್ರುವ ಭಿನ್ನ m2 = dmin * m;

    ಧ್ರುವ sycl::vec<uint8_t, 4> q = vec_aligned_load<uint8_t, 4>(qs + 32 * il + ir);
    ಧ್ರುವ sycl::vec<uint8_t, 4> h = vec_aligned_load<uint8_t, 4>(qh_ptr + ir);

    ಧ್ರುವ uint8_t hm1 = 1 << (2 * il);
    ಧ್ರುವ uint8_t hm2 = hm1 << 1;

    sycl::vec<dst_t, 4> lo, hi;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 4; ++l) {
        lo[l] = d1 * ((q[l] & 0xF) + ((h[l] & hm1) ? 16 : 0)) - m1;
        hi[l] = d2 * ((q[l] >>  4) + ((h[l] & hm2) ? 16 : 0)) - m2;
    }
    dst_t * yb = y + 64 * il + ir;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb +  0) = lo;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb + 32) = hi;
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q5_K_wide(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb,
                                       ಧ್ರುವ sycl::nd_item<1> & it) {
    static_assert(QK_K == 256, "wide K-quant dequant assumes QK_K == 256");
    ಧ್ರುವ int64_t g   = it.get_global_id(0);
    ಧ್ರುವ int64_t b   = g / SYCL_DEQK_WI_PER_BLOCK;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     tid = (ಪೂರ್ಣಾಙ್ಕ) (g % SYCL_DEQK_WI_PER_BLOCK);
    ಯದಿ (b >= nb) {
        ಪ್ರತ್ಯಾಗಮ;
    }
    ಧ್ರುವ block_q5_K * x = (ಧ್ರುವ block_q5_K *) vx + b;
    dequantize_q5_K_wide_one(yy + b * QK_K, x->qs, x->qh, x->scales, (ಭಿನ್ನ) x->dm[0], (ಭಿನ್ನ) x->dm[1], tid);
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q5_K_reorder(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, ಧ್ರುವ int64_t ib,
                                          uint8_t * scales_local, ಧ್ರುವ sycl::nd_item<3> & item_ct1, int64_t n_blocks) {

#ಯದಿ QK_K == 256
    // assume 64 threads
    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
    ಧ್ರುವ int64_t il  = tid / 16;   // 0...3
    ಧ್ರುವ int64_t ir  = tid % 16;   // 0...15
    ಧ್ರುವ int64_t is  = 2 * il;

    dst_t * y = yy + 64 * il + 2 * ir;

    ಧ್ರುವ uint8_t * base = static_cast<ಧ್ರುವ uint8_t *>(vx);

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales (K_SCALE_SIZE per block)] [dm (half2 per block)]
    ಧ್ರುವ size_t qs_offset     = ib * (QK_K / 2);
    ಧ್ರುವ size_t qh_offset     = n_blocks * (QK_K / 2) + ib * (QK_K / 8);
    ಧ್ರುವ size_t scales_offset = n_blocks * (QK_K / 2) + n_blocks * (QK_K / 8) + ib * K_SCALE_SIZE;
    ಧ್ರುವ size_t dm_offset     = n_blocks * (QK_K / 2) + n_blocks * (QK_K / 8) + n_blocks * K_SCALE_SIZE + ib * ಪರಿಮಾಣ(ggml_half2);

    ಧ್ರುವ uint8_t *  qs_ptr     = base + qs_offset;
    ಧ್ರುವ uint8_t *  qh_ptr     = base + qh_offset;
    ಧ್ರುವ uint8_t *  scales_ptr = base + scales_offset;
    ಧ್ರುವ ggml_half2 dm_ಮೌಲ್ಯಗಳು  = *reinterpret_cast<ಧ್ರುವ ggml_half2 *>(base + dm_offset);

    ಧ್ರುವ ಭಿನ್ನ dall = dm_ಮೌಲ್ಯಗಳು.x();
    ಧ್ರುವ ಭಿನ್ನ dmin = dm_ಮೌಲ್ಯಗಳು.y();

    ಧ್ರುವ uint8_t * ql = qs_ptr + 32 * il + 2 * ir;
    ಧ್ರುವ uint8_t * qh = qh_ptr + 2 * ir;

    ಯದಿ (tid < K_SCALE_SIZE) {
        scales_local[tid] = scales_ptr[tid];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    ಧ್ರುವ ಭಿನ್ನ d1 = dall * sc; ಧ್ರುವ ಭಿನ್ನ m1 = dmin * m;
    get_scale_min_k4(is + 1, scales_local, sc, m);
    ಧ್ರುವ ಭಿನ್ನ d2 = dall * sc; ಧ್ರುವ ಭಿನ್ನ m2 = dmin * m;

    uint8_t hm  = 1 << (2 * il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
#ಅನ್ಯಥಾ
    GGML_UNUSED(ib); GGML_UNUSED(tid); GGML_UNUSED(yy); GGML_UNUSED(scales_local); GGML_UNUSED(n_blocks);
    GGML_ABORT("Q5_K reorder dequantize not supported for QK_K != 256");
#endif
}

template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q6_K(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy,
                                  ಧ್ರುವ sycl::nd_item<3> &item_ct1) {
    ಧ್ರುವ block_q6_K * x = (ಧ್ರುವ block_q6_K *) vx;

    ಧ್ರುವ int64_t i = item_ct1.get_group(2);
#ಯದಿ QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
    ಧ್ರುವ int64_t ip  = tid/32;   // ip is 0 or 1
    ಧ್ರುವ int64_t il  = tid - 32*ip; // 0...32
    ಧ್ರುವ int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    ಧ್ರುವ ಭಿನ್ನ d = x[i].d;

    ಧ್ರುವ uint8_t * ql = x[i].ql + 64*ip + il;
    ಧ್ರುವ uint8_t   qh = x[i].qh[32*ip + il];
    ಧ್ರುವ int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#ಅನ್ಯಥಾ

    // assume 32 threads
    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
    ಧ್ರುವ int64_t ip  = tid/16;         // 0 or 1
    ಧ್ರುವ int64_t il  = tid - 16*ip;    // 0...15

    dst_t * y = yy + i*QK_K + 16*ip + il;

    ಧ್ರುವ ಭಿನ್ನ d = x[i].d;

    ಧ್ರುವ uint8_t   ql = x[i].ql[16*ip + il];
    ಧ್ರುವ uint8_t   qh = x[i].qh[il] >> (2*ip);
    ಧ್ರುವ int8_t  * sc = x[i].scales;

    y[ 0] = d * sc[ip+0] * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[ip+2] * ((int8_t)((ql  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
#endif
}

template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_q6_K_reorder(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, ಧ್ರುವ int64_t ib,
                                          ಧ್ರುವ sycl::nd_item<3> & item_ct1, int64_t n_blocks) {

    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
    ಧ್ರುವ int64_t ip  = tid / 32;       // ip is 0 or 1
    ಧ್ರುವ int64_t il  = tid - 32 * ip;  // 0...32
    ಧ್ರುವ int64_t is  = 8 * ip + il / 16;

    ಧ್ರುವ uint8_t *   base_ptr           = static_cast<ಧ್ರುವ uint8_t *>(vx);
    ಧ್ರುವ auto        ql_offset          = ib * (QK_K / 2);
    ಧ್ರುವ auto        qh_offset          = (QK_K / 2) * n_blocks + (QK_K / 4) * ib;
    ಧ್ರುವ auto        base_scales_offset = (QK_K / 2) * n_blocks + (QK_K / 4) * n_blocks + (QK_K / 16) * ib;
    ಧ್ರುವ auto        base_d_offset      = ((QK_K / 2) + (QK_K / 4) + (QK_K / 16)) * n_blocks;
    ಧ್ರುವ uint8_t *   ql_ptr             = base_ptr + ql_offset;
    ಧ್ರುವ uint8_t *   qh_ptr             = base_ptr + qh_offset;
    ಧ್ರುವ uint8_t *   scales_ptr         = base_ptr + base_scales_offset;
    ಧ್ರುವ ggml_half * d                  = (ಧ್ರುವ ggml_half *) (base_ptr + base_d_offset) + ib;

    dst_t * y = yy + 128 * ip + il;

    ಧ್ರುವ uint8_t * ql = ql_ptr + 64 * ip + il;
    ಧ್ರುವ uint8_t   qh = *(qh_ptr + 32 * ip + il);
    ಧ್ರುವ int8_t *  sc = reinterpret_cast<ಧ್ರುವ int8_t *>(scales_ptr + is);

    y[0]  = *d * sc[0] * ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = *d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = *d * sc[4] * ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = *d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_iq2_xxs(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy,
                                     ಧ್ರುವ sycl::nd_item<3> &item_ct1,
                                     ಧ್ರುವ uint64_t *iq2xxs_grid_ptr,
                                     ಧ್ರುವ uint8_t *ksigns_iq2xs_ptr,
                                     ಧ್ರುವ uint8_t *kmask_iq2xs_ptr) {

    ಧ್ರುವ int64_t i = item_ct1.get_group(2);
    ಧ್ರುವ block_iq2_xxs * x = (ಧ್ರುವ block_iq2_xxs  *) vx;

    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
#ಯದಿ QK_K == 256
    ಧ್ರುವ int64_t il = tid/8; // 0...3
    ಧ್ರುವ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ಧ್ರುವ uint16_t * q2 = x[i].qs + 4*ib;
    ಧ್ರುವ uint8_t  * aux8 = (ಧ್ರುವ uint8_t *)q2;
    ಧ್ರುವ uint8_t  * grid = (ಧ್ರುವ uint8_t *)(iq2xxs_grid_ptr + aux8[il]);
    ಧ್ರುವ uint32_t aux32 = q2[2] | (q2[3] << 16);
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    ಧ್ರುವ uint8_t signs = ksigns_iq2xs_ptr[(aux32 >> 7*il) & 127];
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs_ptr[j] ? -1.f : 1.f);
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
#endif

}

template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_iq2_xs(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy,
                                    ಧ್ರುವ sycl::nd_item<3> &item_ct1,
                                    ಧ್ರುವ uint64_t *iq2xs_grid,
                                    ಧ್ರುವ uint8_t *ksigns_iq2xs,
                                    ಧ್ರುವ uint8_t *kmask_iq2xs) {

    ಧ್ರುವ int64_t i = item_ct1.get_group(2);
    ಧ್ರುವ block_iq2_xs * x = (ಧ್ರುವ block_iq2_xs *) vx;

    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
#ಯದಿ QK_K == 256
    ಧ್ರುವ int64_t il = tid/8; // 0...3
    ಧ್ರುವ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ಧ್ರುವ uint16_t * q2 = x[i].qs + 4*ib;
    ಧ್ರುವ uint8_t  * grid = (ಧ್ರುವ uint8_t *)(iq2xs_grid + (q2[il] & 511));
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    ಧ್ರುವ uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
#endif

}

template <typename dst_t>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
dequantize_block_iq2_s(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vx, dst_t *__restrict__ yy,
                       ಧ್ರುವ sycl::nd_item<3> &item_ct1) {

    ಧ್ರುವ int64_t i = item_ct1.get_group(2);
    ಧ್ರುವ block_iq2_s * x = (ಧ್ರುವ block_iq2_s *) vx;

    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
#ಯದಿ QK_K == 256
    ಧ್ರುವ int64_t il = tid/8; // 0...3
    ಧ್ರುವ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ಧ್ರುವ uint8_t * grid = (ಧ್ರುವ uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    ಧ್ರುವ uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; ++j)
        y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);

#endif

}

template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_iq3_xxs(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy,
                                     ಧ್ರುವ sycl::nd_item<3> &item_ct1,
                                     ಧ್ರುವ uint32_t *iq3xxs_grid,
                                     ಧ್ರುವ uint8_t *ksigns_iq2xs,
                                     ಧ್ರುವ uint8_t *kmask_iq2xs) {

    ಧ್ರುವ int64_t i = item_ct1.get_group(2);
    ಧ್ರುವ block_iq3_xxs * x = (ಧ್ರುವ block_iq3_xxs  *) vx;

    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
#ಯದಿ QK_K == 256
    ಧ್ರುವ int64_t il = tid/8; // 0...3
    ಧ್ರುವ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ಧ್ರುವ uint8_t  * q3 = x[i].qs + 8*ib;
    ಧ್ರುವ uint16_t * gas = (ಧ್ರುವ uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    ಧ್ರುವ uint8_t  * grid1 = (ಧ್ರುವ uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    ಧ್ರುವ uint8_t  * grid2 = (ಧ್ರುವ uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    ಧ್ರುವ uint32_t aux32 = gas[0] | (gas[1] << 16);
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    ಧ್ರುವ uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
#endif

}

template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_iq3_xxs_reorder(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, ಧ್ರುವ int64_t i,
                                             ಧ್ರುವ sycl::nd_item<3> & item_ct1, ಧ್ರುವ int64_t nb,
                                             ಧ್ರುವ uint32_t * iq3xxs_grid,
                                             ಧ್ರುವ uint8_t * ksigns_iq2xs,
                                             ಧ್ರುವ uint8_t * kmask_iq2xs) {
    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
#ಯದಿ QK_K == 256
    ಧ್ರುವ int64_t il = tid / 8;
    ಧ್ರುವ int64_t ib = tid % 8;
    ಧ್ರುವ uint8_t * base = static_cast<ಧ್ರುವ uint8_t *>(vx);
    ಧ್ರುವ uint8_t * qs = base + i * (3 * QK_K / 8);
    ಧ್ರುವ ggml_half * d = reinterpret_cast<ಧ್ರುವ ggml_half *>(base + nb * (3 * QK_K / 8));
    ಧ್ರುವ uint8_t * q3 = qs + 8 * ib;
    ಧ್ರುವ uint16_t * gas = reinterpret_cast<ಧ್ರುವ uint16_t *>(qs + QK_K / 4) + 2 * ib;
    ಧ್ರುವ uint8_t * grid1 = reinterpret_cast<ಧ್ರುವ uint8_t *>(iq3xxs_grid + q3[2*il+0]);
    ಧ್ರುವ uint8_t * grid2 = reinterpret_cast<ಧ್ರುವ uint8_t *>(iq3xxs_grid + q3[2*il+1]);
    ಧ್ರುವ uint32_t aux32 = gas[0] | (gas[1] << 16);
    ಧ್ರುವ ಭಿನ್ನ dl = (ಭಿನ್ನ) d[i] * (0.5f + (aux32 >> 28)) * 0.5f;
    ಧ್ರುವ uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
    dst_t * y = yy + 32 * ib + 8 * il;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; ++j) {
        y[j+0] = dl * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = dl * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ಅನ್ಯಥಾ
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(i);
    GGML_UNUSED(item_ct1);
    GGML_UNUSED(nb);
    GGML_UNUSED(iq3xxs_grid);
    GGML_UNUSED(ksigns_iq2xs);
    GGML_UNUSED(kmask_iq2xs);
    GGML_ABORT("IQ3_XXS reorder dequantization requires QK_K == 256");
#endif
}

template <typename dst_t>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
dequantize_block_iq3_s(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vx, dst_t *__restrict__ yy,
                       ಧ್ರುವ sycl::nd_item<3> &item_ct1,
                       ಧ್ರುವ uint8_t *kmask_iq2xs, ಧ್ರುವ uint32_t *iq3s_grid) {

    ಧ್ರುವ int64_t i = item_ct1.get_group(2);
    ಧ್ರುವ block_iq3_s * x = (ಧ್ರುವ block_iq3_s *) vx;

    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
#ಯದಿ QK_K == 256
    ಧ್ರುವ int64_t il = tid/8; // 0...3
    ಧ್ರುವ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ಧ್ರುವ uint8_t * qs = x[i].qs + 8*ib;
    ಧ್ರುವ uint8_t * grid1 = (ಧ್ರುವ uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    ಧ್ರುವ uint8_t * grid2 = (ಧ್ರುವ uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    ಧ್ರುವ uint8_t signs = x[i].signs[4*ib + il];
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
#endif

}

template <typename dst_t>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
dequantize_block_iq3_s_reorder(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, ಧ್ರುವ int64_t i,
                               ಧ್ರುವ sycl::nd_item<3> & item_ct1, ಧ್ರುವ int64_t nb,
                               ಧ್ರುವ uint8_t * kmask_iq2xs, ಧ್ರುವ uint32_t * iq3s_grid) {
    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
#ಯದಿ QK_K == 256
    ಧ್ರುವ int64_t il = tid / 8;
    ಧ್ರುವ int64_t ib = tid % 8;
    ಧ್ರುವ uint8_t * base = static_cast<ಧ್ರುವ uint8_t *>(vx);
    ಧ್ರುವ uint8_t * qs = base + i * (QK_K / 4);
    ಧ್ರುವ uint8_t * qh = base + nb * (QK_K / 4) + i * (QK_K / 32);
    ಧ್ರುವ uint8_t * signs_base = base + nb * (QK_K / 4 + QK_K / 32) + i * (QK_K / 8);
    ಧ್ರುವ uint8_t * metadata = base + nb * (QK_K / 4 + QK_K / 32 + QK_K / 8) +
                               i * (ಪರಿಮಾಣ(ggml_half) + IQ3S_N_SCALE);
    ಧ್ರುವ ggml_half d = *reinterpret_cast<ಧ್ರುವ ggml_half *>(metadata);
    ಧ್ರುವ uint8_t * scales = metadata + ಪರಿಮಾಣ(ggml_half);
    ಧ್ರುವ uint8_t * qs_ib = qs + 8 * ib;
    ಧ್ರುವ uint8_t * grid1 = reinterpret_cast<ಧ್ರುವ uint8_t *>(
        iq3s_grid + (qs_ib[2*il+0] | ((qh[ib] << (8 - 2*il)) & 256)));
    ಧ್ರುವ uint8_t * grid2 = reinterpret_cast<ಧ್ರುವ uint8_t *>(
        iq3s_grid + (qs_ib[2*il+1] | ((qh[ib] << (7 - 2*il)) & 256)));
    ಧ್ರುವ ಭಿನ್ನ dl = (ಭಿನ್ನ) d * (1 + 2 * ((scales[ib/2] >> (4 * (ib%2))) & 0xf));
    ಧ್ರುವ uint8_t signs = signs_base[4 * ib + il];
    dst_t * y = yy + 32 * ib + 8 * il;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; ++j) {
        y[j+0] = dl * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = dl * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ಅನ್ಯಥಾ
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(i);
    GGML_UNUSED(item_ct1);
    GGML_UNUSED(nb);
    GGML_UNUSED(kmask_iq2xs);
    GGML_UNUSED(iq3s_grid);
    GGML_ABORT("IQ3_S reorder dequantization requires QK_K == 256");
#endif
}

template <typename dst_t>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
dequantize_block_iq1_s(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vx, dst_t *__restrict__ yy,
                       ಧ್ರುವ sycl::nd_item<3> &item_ct1,
                       ಧ್ರುವ uint32_t *iq1s_grid_gpu) {

    ಧ್ರುವ int64_t i = item_ct1.get_group(2);
    ಧ್ರುವ block_iq1_s * x = (ಧ್ರುವ block_iq1_s  *) vx;

    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
#ಯದಿ QK_K == 256
    ಧ್ರುವ int64_t il = tid/8; // 0...3
    ಧ್ರುವ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ಧ್ರುವ ಭಿನ್ನ delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; ಧ್ರುವ int8_t * q = (ಧ್ರುವ int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
#endif

}

template <typename dst_t>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
dequantize_block_iq1_m(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vx, dst_t *__restrict__ yy,
                       ಧ್ರುವ sycl::nd_item<3> &item_ct1,
                       ಧ್ರುವ uint32_t *iq1s_grid_gpu) {

    ಧ್ರುವ int64_t i = item_ct1.get_group(2);
    ಧ್ರುವ block_iq1_m * x = (ಧ್ರುವ block_iq1_m  *) vx;

    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
#ಯದಿ QK_K == 256
    ಧ್ರುವ int64_t il = tid/8; // 0...3
    ಧ್ರುವ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ಧ್ರುವ uint16_t * sc = (ಧ್ರುವ uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    ಧ್ರುವ ಭಿನ್ನ delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; ಧ್ರುವ int8_t * q = (ಧ್ರುವ int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
#endif

}

template <typename dst_t>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
dequantize_block_iq4_nl(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vx, dst_t *__restrict__ yy,
                        ಧ್ರುವ sycl::nd_item<3> &item_ct1) {

    ಧ್ರುವ int64_t i = item_ct1.get_group(2);
    ಧ್ರುವ block_iq4_nl * x = (ಧ್ರುವ block_iq4_nl *) vx + i*(QK_K/QK4_NL);

    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
    ಧ್ರುವ int64_t il = tid/8; // 0...3
    ಧ್ರುವ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    ಧ್ರುವ uint8_t  * q4 = x[ib].qs + 4*il;
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)x[ib].d;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }

}


template <typename dst_t>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
dequantize_block_iq4_xs(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vx, dst_t *__restrict__ yy,
                        ಧ್ರುವ sycl::nd_item<3> &item_ct1) {
    ಧ್ರುವ int64_t i = item_ct1.get_group(2);
    ಧ್ರುವ block_iq4_xs * x = (ಧ್ರುವ block_iq4_xs *)vx;

    ಧ್ರುವ int64_t tid = item_ct1.get_local_id(2);
    ಧ್ರುವ int64_t il = tid/8; // 0...3
    ಧ್ರುವ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    ಧ್ರುವ uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

template<typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_mxfp4(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy,
                                   ಧ್ರುವ sycl::nd_item<3> &item_ct1) {
    // auto                item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ int64_t       i        = item_ct1.get_group(2);
    ಧ್ರುವ block_mxfp4 * x = (ಧ್ರುವ block_mxfp4 *) vx + i*(QK_K/QK_MXFP4);

    ಧ್ರುವ int64_t    tid = item_ct1.get_local_id(2);
    ಧ್ರುವ int64_t il = tid/8; // 0...3
    ಧ್ರುವ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    ಧ್ರುವ uint8_t  * q4 = x[ib].qs + 4*il;
    ಧ್ರುವ ಭಿನ್ನ d = ggml_sycl_e8m0_to_fp32(x[ib].e);
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_mxfp4[q4[j] & 0xf]*0.5f;
        y[j+16] = d * kvalues_mxfp4[q4[j] >>  4]*0.5f;
    }
}

// Reordered MXFP4 ([qs...][e...], see ggml_sycl_reordered::block_q_t<MXFP4>): one work-item per block.
template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_mxfp4_reorder(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                           ಧ್ರುವ sycl::nd_item<3> & item_ct1) {
    ಧ್ರುವ int64_t ib = (int64_t) item_ct1.get_group(2) * WARP_SIZE + item_ct1.get_local_id(2);
    ಯದಿ (ib >= k / QK_MXFP4) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ uint8_t * qs = (ಧ್ರುವ uint8_t *) vx + ib * (QK_MXFP4 / 2);
    ಧ್ರುವ ಭಿನ್ನ     d  = ggml_sycl_e8m0_to_fp32(((ಧ್ರುವ uint8_t *) vx)[k / 2 + ib]) * 0.5f;
    dst_t *         y  = yy + ib * QK_MXFP4;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < QK_MXFP4 / 2; ++j) {
        y[j]                = d * kvalues_mxfp4[qs[j] & 0xf];
        y[j + QK_MXFP4 / 2] = d * kvalues_mxfp4[qs[j] >> 4];
    }
}


template <typename dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_block_nvfp4(
        ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vx,
        dst_t * __restrict__ yy,
        ಧ್ರುವ int64_t ne) {
    auto          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಧ್ರುವ int64_t i        = item_ct1.get_group(2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ     tid      = item_ct1.get_local_id(2);

    ಧ್ರುವ int64_t base = i * QK_NVFP4;
    ಯದಿ (base >= ne) {
        ಪ್ರತ್ಯಾಗಮ;
    }

    ಧ್ರುವ block_nvfp4 * x = (ಧ್ರುವ block_nvfp4 *) vx;
    ಧ್ರುವ block_nvfp4 & xb = x[i];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sub = tid / (QK_NVFP4_SUB / 2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = tid % (QK_NVFP4_SUB / 2);

    ಧ್ರುವ ಭಿನ್ನ d = ggml_sycl_ue4m3_to_fp32(xb.d[sub]);
    ಧ್ರುವ uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    ಧ್ರುವ int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    ಧ್ರುವ int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_sycl_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_sycl_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}


#endif // GGML_SYCL_DEQUANTIZE_HPP
