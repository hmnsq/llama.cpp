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

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_DEQUANTIZE_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_DEQUANTIZE_HPP

#ಅನ್ತರ್ಭಾವ "common.hpp"
#ಅನ್ತರ್ಭಾವ "convert.hpp"

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*dequantize_kernel_t)(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ int64_t ib, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 & v);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*dequantize_kernel_t_reorder)(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *d, ಸ್ಥಿರಾಂಕ int64_t ib, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *qs,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v);
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಶೂನ್ಯ (*dequantize_kernel_f32_t)(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * vx, ಸ್ಥಿರಾಂಕ int64_t ib, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, ದಶಮಾಂಶ & v0, ದಶಮಾಂಶ & v1);

#ಆದರೆ QK_K == 256
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ get_scale_min_k4(ಪೂರ್ಣಾಂಕ j, ಸ್ಥಿರಾಂಕ uint8_t * q, uint8_t & d, uint8_t & m);
#ಆದರೆ_ಅಂತ್ಯ

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q2_0(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_0 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_0 *) vx;

    ಸ್ಥಿರಾಂಕ dfloat d = x[ib].d;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ byte_idx = iqs / 4;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸರಿಸು = (iqs % 4) * 2;
    ಸ್ಥಿರಾಂಕ uint8_t vui = x[ib].qs[byte_idx];

    v.x() = (vui >> ಸರಿಸು) & 3;
    v.y() = (vui >> (ಸರಿಸು + 2)) & 3;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    v.s0() = ((dfloat)v.s0() - 1.0f) * d;
    v.s1() = ((dfloat)v.s1() - 1.0f) * d;
#ಇಲ್ಲದಿದ್ದರೆ
    v.x() = ((dfloat)v.x() - 1.0f) * d;
    v.y() = ((dfloat)v.y() - 1.0f) * d;
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q4_0(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) vx;

    ಸ್ಥಿರಾಂಕ dfloat d = x[ib].d;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#ಇಲ್ಲದಿದ್ದರೆ
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q4_0_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *d_ptr, ಸ್ಥಿರಾಂಕ int64_t ib, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *qs,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
    // const block_q4_0 * x = (const block_q4_0 *) vx;

    ಸ್ಥಿರಾಂಕ dfloat d = (ಸ್ಥಿರಾಂಕ dfloat)*((ಸ್ಥಿರಾಂಕ sycl::half*)d_ptr+ib);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vui = *((ಸ್ಥಿರಾಂಕ uint8_t *)qs+iqs);

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#ಇಲ್ಲದಿದ್ದರೆ
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q1_0_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *d_ptr, ಸ್ಥಿರಾಂಕ int64_t ib, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *qs,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
    // Q1_0 reorder layout: scale values followed by quantized bits
    ಸ್ಥಿರಾಂಕ dfloat d = (ಸ್ಥಿರಾಂಕ dfloat)*((ಸ್ಥಿರಾಂಕ sycl::half*)d_ptr+ib);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಬಿಟ್_ಸೂಚ್ಯಂಕ_0 = iqs + 0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಬಿಟ್_ಸೂಚ್ಯಂಕ_1 = iqs + 1;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಬಿಟ್_0 = (*((ಸ್ಥಿರಾಂಕ uint8_t *)qs + ಬಿಟ್_ಸೂಚ್ಯಂಕ_0 / 8) >> (ಬಿಟ್_ಸೂಚ್ಯಂಕ_0 % 8)) & 1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಬಿಟ್_1 = (*((ಸ್ಥಿರಾಂಕ uint8_t *)qs + ಬಿಟ್_ಸೂಚ್ಯಂಕ_1 / 8) >> (ಬಿಟ್_ಸೂಚ್ಯಂಕ_1 % 8)) & 1;

    v.x() = (2 * ಬಿಟ್_0 - 1) * d;
    v.y() = (2 * ಬಿಟ್_1 - 1) * d;
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q1_0(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q1_0 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q1_0 *) vx;
    ಸ್ಥಿರಾಂಕ dfloat d = x[ib].d;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಬಿಟ್_ಸೂಚ್ಯಂಕ_0 = iqs + 0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಬಿಟ್_ಸೂಚ್ಯಂಕ_1 = iqs + 1;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಬಿಟ್_0 = (x[ib].qs[ಬಿಟ್_ಸೂಚ್ಯಂಕ_0 / 8] >> (ಬಿಟ್_ಸೂಚ್ಯಂಕ_0 % 8)) & 1;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಬಿಟ್_1 = (x[ib].qs[ಬಿಟ್_ಸೂಚ್ಯಂಕ_1 / 8] >> (ಬಿಟ್_ಸೂಚ್ಯಂಕ_1 % 8)) & 1;

    v.x() = (2 * ಬಿಟ್_0 - 1) * d;
    v.y() = (2 * ಬಿಟ್_1 - 1) * d;
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q4_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 *) vx;

    ಸ್ಥಿರಾಂಕ dfloat d = x[ib].dm[0];
    ಸ್ಥಿರಾಂಕ dfloat m = x[ib].dm[1];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = sycl::fma(v.s0(), d, m);
    v.s1() = sycl::fma(v.s1(), d, m);

#ಇಲ್ಲದಿದ್ದರೆ
    v.x() = sycl::fma(v.x(), d, m);
    v.y() = sycl::fma(v.y(), d, m);
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q4_K(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K *) vx;
    ಸ್ಥಿರಾಂಕ sycl::half2 dm = x[ib].dm;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = dm[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = dm[1];

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il = idx / 64;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = idx % 64;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is = 2 * il + (ಒಳ >= 32 ? 1 : 0);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ off = ಒಳ & 31;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qsi = 32 * il + off;

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ಸ್ಥಿರಾಂಕ uint8_t q = x[ib].qs[qsi];
        ಸ್ಥಿರಾಂಕ uint8_t qv = (ಒಳ >= 32) ? (q >> 4) : (q & 0xF);
        ಹಿಂತಿರುಗಿ sycl::fma((dfloat) qv, (dfloat) (dall * sc), (dfloat) (-dmin * m));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("Q4_K dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q4_K_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, ದಶಮಾಂಶ &v0, ದಶಮಾಂಶ &v1) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K *) vx;
    ಸ್ಥಿರಾಂಕ sycl::half2 dm = x[ib].dm;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = dm[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = dm[1];

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> ದಶಮಾಂಶ {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il = idx / 64;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = idx % 64;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is = 2 * il + (ಒಳ >= 32 ? 1 : 0);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ qsi = 32 * il + (ಒಳ & 31);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = dall * sc;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ mn = dmin * m;
        ಸ್ಥಿರಾಂಕ uint8_t q = x[ib].qs[qsi];
        ಸ್ಥಿರಾಂಕ uint8_t qv = (ಒಳ >= 32) ? (q >> 4) : (q & 0xF);

        ಹಿಂತಿರುಗಿ d * qv - mn;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("Q4_K dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q2_K(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K *) vx;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = x[ib].dm[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = x[ib].dm[1];

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n = idx / 128;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = idx % 128;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ g = r / 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l = r % 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is = 8 * n + l / 16;

        ಸ್ಥಿರಾಂಕ uint8_t q = x[ib].qs[32 * n + l];
        ಸ್ಥಿರಾಂಕ uint8_t sc = x[ib].scales[is + 2 * g];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = dall * (sc & 0xF);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m = dmin * (sc >> 4);

        ಹಿಂತಿರುಗಿ (dfloat) d * (dfloat) ((q >> (2 * g)) & 3) - (dfloat) m;
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("Q2_K dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q2_K_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, ದಶಮಾಂಶ &v0, ದಶಮಾಂಶ &v1) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K *) vx;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = x[ib].dm[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = x[ib].dm[1];

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> ದಶಮಾಂಶ {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n = idx / 128;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = idx % 128;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ g = r / 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l = r % 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is = 8 * n + l / 16;

        ಸ್ಥಿರಾಂಕ uint8_t q = x[ib].qs[32 * n + l];
        ಸ್ಥಿರಾಂಕ uint8_t sc = x[ib].scales[is + 2 * g];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = dall * (sc & 0xF);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m = dmin * (sc >> 4);

        ಹಿಂತಿರುಗಿ d * ((q >> (2 * g)) & 3) - m;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("Q2_K dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q3_K(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K *) vx;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d_all = x[ib].d;

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ n = idx / 128;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = idx % 128;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = r / 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ l = r % 32;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is0 = l / 16;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is = 8 * n + 2 * j + is0;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸರಿಸು = 2 * j;
        ಸ್ಥಿರಾಂಕ uint8_t m = 1 << (4 * n + j);

        ಸ್ಥಿರಾಂಕ int8_t us = is <  4 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 8] >> 0) & 3) << 4) :
                         is <  8 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 4] >> 2) & 3) << 4) :
                         is < 12 ? (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is + 0] >> 4) & 3) << 4) :
                                   (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is - 4] >> 6) & 3) << 4);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dl = d_all * (us - 32);
        ಸ್ಥಿರಾಂಕ uint8_t q = x[ib].qs[32 * n + l];
        ಸ್ಥಿರಾಂಕ uint8_t h = x[ib].hmask[l];
        ಸ್ಥಿರಾಂಕ int8_t qv = ((q >> ಸರಿಸು) & 3) - ((h & m) ? 0 : 4);

        ಹಿಂತಿರುಗಿ (dfloat) (dl * qv);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("Q3_K dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q5_K(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K *) vx;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = x[ib].dm[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = x[ib].dm[1];

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il = idx / 64;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = idx % 64;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is = 2 * il + (ಒಳ >= 32 ? 1 : 0);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir = (ಒಳ & 31) / 2;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq = ಒಳ & 1;

        ಸ್ಥಿರಾಂಕ uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        ಸ್ಥಿರಾಂಕ uint8_t h = x[ib].qh[2 * ir + iq];
        ಸ್ಥಿರಾಂಕ uint8_t qv = (ಒಳ >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = dall * sc;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ mn = dmin * m;
        ಸ್ಥಿರಾಂಕ uint8_t hm = 1 << (2 * il + (ಒಳ >= 32 ? 1 : 0));

        ಹಿಂತಿರುಗಿ sycl::fma((dfloat) (qv + ((h & hm) ? 16 : 0)), (dfloat) d, (dfloat) (-mn));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("Q5_K dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q5_K_f32(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, ದಶಮಾಂಶ &v0, ದಶಮಾಂಶ &v1) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K *) vx;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = x[ib].dm[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = x[ib].dm[1];

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> ದಶಮಾಂಶ {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il = idx / 64;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = idx % 64;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is = 2 * il + (ಒಳ >= 32 ? 1 : 0);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir = (ಒಳ & 31) / 2;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iq = ಒಳ & 1;

        ಸ್ಥಿರಾಂಕ uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        ಸ್ಥಿರಾಂಕ uint8_t h = x[ib].qh[2 * ir + iq];
        ಸ್ಥಿರಾಂಕ uint8_t qv = (ಒಳ >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = dall * sc;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ mn = dmin * m;
        ಸ್ಥಿರಾಂಕ uint8_t hm = 1 << (2 * il + (ಒಳ >= 32 ? 1 : 0));

        ಹಿಂತಿರುಗಿ (qv + ((h & hm) ? 16 : 0)) * d - mn;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("Q5_K dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q6_K(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K *) vx;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = x[ib].d;

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ip = idx / 128;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = idx % 128;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il = ಒಳ & 31;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ig = ಒಳ / 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is = 8 * ip + il / 16;

        ಸ್ಥಿರಾಂಕ uint8_t ql0 = x[ib].ql[64 * ip + il];
        ಸ್ಥಿರಾಂಕ uint8_t ql1 = x[ib].ql[64 * ip + il + 32];
        ಸ್ಥಿರಾಂಕ uint8_t qh = x[ib].qh[32 * ip + il];
        ಸ್ಥಿರಾಂಕ int8_t * sc = x[ib].scales + is;

        uint8_t qv;
        int8_t scale;
        ಆದರೆ (ig == 0) {
            qv = (ql0 & 0xF) | (((qh >> 0) & 3) << 4);
            scale = sc[0];
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ig == 1) {
            qv = (ql1 & 0xF) | (((qh >> 2) & 3) << 4);
            scale = sc[2];
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (ig == 2) {
            qv = (ql0 >> 4) | (((qh >> 4) & 3) << 4);
            scale = sc[4];
        } ಇಲ್ಲದಿದ್ದರೆ {
            qv = (ql1 >> 4) | (((qh >> 6) & 3) << 4);
            scale = sc[6];
        }

        ಹಿಂತಿರುಗಿ (dfloat) (d * scale * ((int8_t) qv - 32));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("Q6_K dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_mxfp4(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 *) vx;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = ggml_sycl_e8m0_ಗಮ್ಯ_fp32(x[ib].e);
    ಸ್ಥಿರಾಂಕ uint8_t q = x[ib].qs[iqs];

    v.x() = d * kvalues_mxfp4[q & 0xF] * 0.5f;
    v.y() = d * kvalues_mxfp4[q >> 4] * 0.5f;
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_nvfp4(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_nvfp4 & xb = ((ಸ್ಥಿರಾಂಕ ಖಂಡ_nvfp4 *) vx)[ib];

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sub = idx / QK_NVFP4_SUB;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = idx % QK_NVFP4_SUB;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ jh = j % (QK_NVFP4_SUB / 2);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = ggml_sycl_ue4m3_ಗಮ್ಯ_fp32(xb.d[sub]);
        ಸ್ಥಿರಾಂಕ uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + jh];
        ಸ್ಥಿರಾಂಕ uint8_t qv = (j < (QK_NVFP4_SUB / 2)) ? (q & 0x0F) : (q >> 4);

        ಹಿಂತಿರುಗಿ d * kvalues_mxfp4[qv];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq2_xxs(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xxs * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xxs *) vx;

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib8 = idx / 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = idx % 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il = r / 8;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = r % 8;

        ಸ್ಥಿರಾಂಕ uint16_t * q2 = x[ib].qs + 4 * ib8;
        ಸ್ಥಿರಾಂಕ uint8_t * aux8 = (ಸ್ಥಿರಾಂಕ uint8_t *) q2;
        ಸ್ಥಿರಾಂಕ uint8_t * grid = (ಸ್ಥಿರಾಂಕ uint8_t *) (iq2xxs_grid + aux8[il]);
        ಸ್ಥಿರಾಂಕ uint32_t aux32 = q2[2] | (q2[3] << 16);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ) x[ib].d * (0.5f + (aux32 >> 28)) * 0.25f;
        ಸ್ಥಿರಾಂಕ uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        ಹಿಂತಿರುಗಿ d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("IQ2_XXS dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq2_xs(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                              ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xs * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xs *) vx;

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib8 = idx / 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = idx % 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il = r / 8;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = r % 8;

        ಸ್ಥಿರಾಂಕ uint16_t * q2 = x[ib].qs + 4 * ib8;
        ಸ್ಥಿರಾಂಕ uint8_t * grid = (ಸ್ಥಿರಾಂಕ uint8_t *) (iq2xs_grid + (q2[il] & 511));
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ಸ್ಥಿರಾಂಕ uint8_t signs = ksigns_iq2xs[q2[il] >> 9];

        ಹಿಂತಿರುಗಿ d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("IQ2_XS dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq2_s(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_s * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_s *) vx;

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib8 = idx / 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = idx % 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il = r / 8;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = r % 8;

        ಸ್ಥಿರಾಂಕ uint16_t grid_id = x[ib].qs[4 * ib8 + il] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 0x300);
        ಸ್ಥಿರಾಂಕ uint8_t * grid = (ಸ್ಥಿರಾಂಕ uint8_t *) (iq2s_grid + grid_id);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ಸ್ಥಿರಾಂಕ uint8_t signs = x[ib].qs[QK_K / 8 + 4 * ib8 + il];

        ಹಿಂತಿರುಗಿ d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("IQ2_S dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq3_xxs(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs *) vx;

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib8 = idx / 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = idx % 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il = r / 8;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = r % 8;

        ಸ್ಥಿರಾಂಕ uint8_t * q3 = x[ib].qs + 8 * ib8;
        ಸ್ಥಿರಾಂಕ uint16_t * gas = (ಸ್ಥಿರಾಂಕ uint16_t *) (x[ib].qs + QK_K / 4) + 2 * ib8;
        ಸ್ಥಿರಾಂಕ uint8_t * grid1 = (ಸ್ಥಿರಾಂಕ uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        ಸ್ಥಿರಾಂಕ uint8_t * grid2 = (ಸ್ಥಿರಾಂಕ uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        ಸ್ಥಿರಾಂಕ uint32_t aux32 = gas[0] | (gas[1] << 16);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ) x[ib].d * (0.5f + (aux32 >> 28)) * 0.5f;
        ಸ್ಥಿರಾಂಕ uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        ಆದರೆ (j < 4) {
            ಹಿಂತಿರುಗಿ d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        ಹಿಂತಿರುಗಿ d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("IQ3_XXS dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq3_s(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s *) vx;

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib8 = idx / 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = idx % 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il = r / 8;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = r % 8;

        ಸ್ಥಿರಾಂಕ uint8_t * qs = x[ib].qs + 8 * ib8;
        ಸ್ಥಿರಾಂಕ uint16_t grid1_id = qs[2 * il + 0] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 256);
        ಸ್ಥಿರಾಂಕ uint16_t grid2_id = qs[2 * il + 1] | ((x[ib].qh[ib8] << (7 - 2 * il)) & 256);
        ಸ್ಥಿರಾಂಕ uint8_t * grid1 = (ಸ್ಥಿರಾಂಕ uint8_t *) (iq3s_grid + grid1_id);
        ಸ್ಥಿರಾಂಕ uint8_t * grid2 = (ಸ್ಥಿರಾಂಕ uint8_t *) (iq3s_grid + grid2_id);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ) x[ib].d * (1 + 2 * ((x[ib].scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
        ಸ್ಥಿರಾಂಕ uint8_t signs = x[ib].signs[4 * ib8 + il];

        ಆದರೆ (j < 4) {
            ಹಿಂತಿರುಗಿ d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        ಹಿಂತಿರುಗಿ d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("IQ3_S dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq1_s(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_s * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_s *) vx;

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib8 = idx / 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = idx % 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il = r / 8;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = r % 8;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ delta = (x[ib].qh[ib8] & 0x8000) ? (-1.f - IQ1S_DELTA) : (-1.f + IQ1S_DELTA);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ) x[ib].d * (2 * ((x[ib].qh[ib8] >> 12) & 7) + 1);
        ಸ್ಥಿರಾಂಕ uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((x[ib].qh[ib8] >> (3 * il)) & 7) << 8);
        ಸ್ಥಿರಾಂಕ uint32_t g = iq1s_grid_gpu[grid_id];
        ಸ್ಥಿರಾಂಕ int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        ಹಿಂತಿರುಗಿ d * (qv + delta);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("IQ1_S dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq1_m(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_m * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_m *) vx;

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib8 = idx / 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = idx % 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il = r / 8;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = r % 8;

        ಸ್ಥಿರಾಂಕ uint16_t * sc = (ಸ್ಥಿರಾಂಕ uint16_t *) x[ib].scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib16 = 2 * ib8 + il / 2;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);

        ಸ್ಥಿರಾಂಕ uint8_t qh = x[ib].qh[2 * ib8 + il / 2];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ delta = (qh & (0x08 << (4 * (il % 2)))) ? (-1.f - IQ1M_DELTA) : (-1.f + IQ1M_DELTA);

        ಸ್ಥಿರಾಂಕ uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((qh >> (4 * (il % 2))) & 7) << 8);
        ಸ್ಥಿರಾಂಕ uint32_t g = iq1s_grid_gpu[grid_id];
        ಸ್ಥಿರಾಂಕ int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        ಹಿಂತಿರುಗಿ d * (qv + delta);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("IQ1_M dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq4_nl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                              ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl *) vx;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ) x[ib].d;

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಆದರೆ (idx < 16) {
            ಹಿಂತಿರುಗಿ d * kvalues_iq4nl[x[ib].qs[idx] & 0xF];
        }
        ಹಿಂತಿರುಗಿ d * kvalues_iq4nl[x[ib].qs[idx - 16] >> 4];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_iq4_xs(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                              ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs *) vx;

    ಸ್ವಯಂಚಾಲಿತ dequantize_one = [&](ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ idx) -> dfloat {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib8 = idx / 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = idx % 32;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ byte_idx = (r < 16) ? r : (r - 16);
        ಸ್ಥಿರಾಂಕ uint8_t q = x[ib].qs[16 * ib8 + byte_idx];
        ಸ್ಥಿರಾಂಕ uint8_t qv = (r < 16) ? (q & 0x0F) : (q >> 4);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ) x[ib].d * ((((x[ib].scales_l[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf) |
                        (((x[ib].scales_h >> (2 * ib8)) & 3) << 4)) - 32);
        ಹಿಂತಿರುಗಿ d * kvalues_iq4nl[qv];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ABORT("IQ4_XS dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q5_0(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 *) vx;

    ಸ್ಥಿರಾಂಕ dfloat d = x[ib].d;

    uint32_t qh;
    ಸ್ಮರಣೆ_ನಕಲು(&qh, x[ib].qh, ಗಾತ್ರ(qh));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    // v = v - {16.0f, 16.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 16.0f) * d;
    v.s1() = (v.s1() - 16.0f) * d;

#ಇಲ್ಲದಿದ್ದರೆ
    v.x() = (v.x() - 16.0f) * d;
    v.y() = (v.y() - 16.0f) * d;
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q5_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 *) vx;

    ಸ್ಥಿರಾಂಕ dfloat d = x[ib].dm[0];
    ಸ್ಥಿರಾಂಕ dfloat m = x[ib].dm[1];

    uint32_t qh;
    ಸ್ಮರಣೆ_ನಕಲು(&qh, x[ib].qh, ಗಾತ್ರ(qh));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = sycl::fma(v.s0(), d, m);
    v.s1() = sycl::fma(v.s1(), d, m);
#ಇಲ್ಲದಿದ್ದರೆ
    v.x() = sycl::fma(v.x(), d, m);
    v.y() = sycl::fma(v.y(), d, m);
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q8_0_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *d_ptr, ಸ್ಥಿರಾಂಕ int64_t ib, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *qs,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
    ಸ್ಥಿರಾಂಕ dfloat d = (ಸ್ಥಿರಾಂಕ dfloat)*((ಸ್ಥಿರಾಂಕ sycl::half*)d_ptr + ib);

    v.x() = ((ಸ್ಥಿರಾಂಕ int8_t *)qs)[iqs + 0];
    v.y() = ((ಸ್ಥಿರಾಂಕ int8_t *)qs)[iqs + 1];

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    v.s0() *= d;
    v.s1() *= d;
#ಇಲ್ಲದಿದ್ದರೆ
    v.x() *= d;
    v.y() *= d;
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ dequantize_q8_0(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *vx, ಸ್ಥಿರಾಂಕ int64_t ib,
                                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs, dfloat2 &v) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) vx;

    ಸ್ಥಿರಾಂಕ dfloat d = x[ib].d;

    v.x() = x[ib].qs[iqs + 0];
    v.y() = x[ib].qs[iqs + 1];

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    // v = v * {d, d};
    v.s0() *= d;
    v.s1() *= d;
#ಇಲ್ಲದಿದ್ದರೆ
    v.x() *= d;
    v.y() *= d;
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q4_0(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);

    // assume 32 threads
    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ int64_t il  = tid/8;
    ಸ್ಥಿರಾಂಕ int64_t ir  = tid%8;
    ಸ್ಥಿರಾಂಕ int64_t ib = 8*i + ir;
    ಆದರೆ (ib >= nb32) {
        ಹಿಂತಿರುಗಿ;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *)vx + ib;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = sycl::vec<sycl::half, 1>(x->d)
                        .convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>()[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dm = -8*d;

    ಸ್ಥಿರಾಂಕ uint8_t * q = x->qs + 4*il;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q4_0_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                  ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    // one thread per qs byte, like dequantize_block<dequantize_q4_0>; the reorder layout makes the byte reads contiguous
    // grid-stride loop because the launcher may shrink the work-group to keep the global range below INT_MAX
    // float math, not dequantize_q4_0_reorder: that helper computes in half with GGML_SYCL_F16, which changes f32 output
    ಸ್ಥಿರಾಂಕ sycl::half * d_ptr  = (ಸ್ಥಿರಾಂಕ sycl::half *) ((ಸ್ಥಿರಾಂಕ uint8_t *) vx + k/2);
    ಸ್ಥಿರಾಂಕ int64_t      stride = ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು_range(2);

    ಗಾಗಿ (int64_t iq = ಅಂಶ_ct1.get_global_id(2); iq < k/2; iq += stride) {
        ಸ್ಥಿರಾಂಕ int64_t ib  = iq / (QK4_0/2);
        ಸ್ಥಿರಾಂಕ int64_t iqs = iq % (QK4_0/2);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     vq  = ((ಸ್ಥಿರಾಂಕ uint8_t *) vx)[iq];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   d   = ದಶಮಾಂಶ(d_ptr[ib]);

        dst_t * y = yy + ib*QK4_0 + iqs;
        y[0]       = d * ((vq & 0xF) - 8);
        y[QK4_0/2] = d * ((vq >>  4) - 8);
    }
}

// Dequantize Q8_0 from reorder layout: [all qs (k bytes)][all d values]
ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q8_0_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                  ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    // two values per thread, like dequantize_block<dequantize_q8_0>; the reorder layout keeps all qs contiguous
    // grid-stride loop because the launcher may shrink the work-group to keep the global range below INT_MAX
    // float math, not dequantize_q8_0_reorder: that helper computes in half with GGML_SYCL_F16, which changes f32 output
    ಸ್ಥಿರಾಂಕ int8_t *     qs     = (ಸ್ಥಿರಾಂಕ int8_t *) vx;
    ಸ್ಥಿರಾಂಕ sycl::half * d_ptr  = (ಸ್ಥಿರಾಂಕ sycl::half *) ((ಸ್ಥಿರಾಂಕ uint8_t *) vx + k);
    ಸ್ಥಿರಾಂಕ int64_t      stride = 2 * ಅಂಶ_ct1.get_local_range(2) * ಅಂಶ_ct1.get_ಗುಂಪು_range(2);

    ಗಾಗಿ (int64_t i = 2 * ಅಂಶ_ct1.get_global_id(2); i < k; i += stride) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = ದಶಮಾಂಶ(d_ptr[i / QK8_0]);

        yy[i + 0] = d * qs[i + 0];
        yy[i + 1] = d * qs[i + 1];
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q8_0_reorder_wide(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                               ಸ್ಥಿರಾಂಕ sycl::id<1> idx) {
    ಸ್ಥಿರಾಂಕ int64_t i = 4 * (int64_t) idx[0];

    ಸ್ಥಿರಾಂಕ int8_t *     qs    = (ಸ್ಥಿರಾಂಕ int8_t *) vx;
    ಸ್ಥಿರಾಂಕ sycl::half * d_ptr = (ಸ್ಥಿರಾಂಕ sycl::half *) ((ಸ್ಥಿರಾಂಕ uint8_t *) vx + k);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d0 = ದಶಮಾಂಶ(d_ptr[i / QK8_0]);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1 = ದಶಮಾಂಶ(d_ptr[(i + 2) / QK8_0]);

    sycl::vec<ದಶಮಾಂಶ, 4> p;
    p[0] = d0 * (ದಶಮಾಂಶ) qs[i + 0];
    p[1] = d0 * (ದಶಮಾಂಶ) qs[i + 1];
    p[2] = d1 * (ದಶಮಾಂಶ) qs[i + 2];
    p[3] = d1 * (ದಶಮಾಂಶ) qs[i + 3];

    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yy + i) = p.convert<dst_t, sycl::rounding_ಬಹುಲಕ::automatic>();
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q4_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);

    // assume 32 threads
    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ int64_t il  = tid/8;
    ಸ್ಥಿರಾಂಕ int64_t ir  = tid%8;
    ಸ್ಥಿರಾಂಕ int64_t ib = 8*i + ir;
    ಆದರೆ (ib >= nb32) {
        ಹಿಂತಿರುಗಿ;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 *)vx + ib;
    ಸ್ಥಿರಾಂಕ sycl::float2 d =
        x->dm.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

    ಸ್ಥಿರಾಂಕ uint8_t * q = x->qs + 4*il;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
        y[l + 0] = d.x() * (q[l] & 0xF) + d.y();
        y[l + 16] = d.x() * (q[l] >> 4) + d.y();
    }
}


//================================== k-quants

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q2_K(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy,
                                  ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K *) vx;

    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ int64_t n   = tid/32;
    ಸ್ಥಿರಾಂಕ int64_t l   = tid - 32*n;
    ಸ್ಥಿರಾಂಕ int64_t is  = 8*n + l/16;

    ಸ್ಥಿರಾಂಕ uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    ದಶಮಾಂಶ dall = x[i].dm[0];
    ದಶಮಾಂಶ dmin = x[i].dm[1];
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ int64_t is = tid/16;  // 0 or 1
    ಸ್ಥಿರಾಂಕ int64_t il = tid%16;  // 0...15
    ಸ್ಥಿರಾಂಕ uint8_t q = x[i].qs[il] >> (2*is);
    dst_t * y = yy + i*QK_K + 16*is + il;

    ದಶಮಾಂಶ dall = x[i].dm[0];
    ದಶಮಾಂಶ dmin = x[i].dm[1];
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#ಆದರೆ_ಅಂತ್ಯ

}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q2_K_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, ಸ್ಥಿರಾಂಕ int64_t i,
                                          ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1, int64_t n_ಖಂಡಗಳು) {
#ಆದರೆ QK_K == 256
    ಆದರೆ (i >= n_ಖಂಡಗಳು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint8_t * ತಳ          = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vx);
    ಸ್ಥಿರಾಂಕ size_t    qs_ಸ್ಥಾನಾಂತರ     = i * (QK_K / 4);
    ಸ್ಥಿರಾಂಕ size_t    scales_ಸ್ಥಾನಾಂತರ = n_ಖಂಡಗಳು * (QK_K / 4) + i * (QK_K / 16);
    ಸ್ಥಿರಾಂಕ size_t    dm_ಸ್ಥಾನಾಂತರ     = n_ಖಂಡಗಳು * (QK_K / 4) + n_ಖಂಡಗಳು * (QK_K / 16) + i * ಗಾತ್ರ(ggml_half2);

    ಸ್ಥಿರಾಂಕ uint8_t *     qs     = ತಳ + qs_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ uint8_t *     scales = ತಳ + scales_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ ggml_half2 * dm     = reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half2 *>(ತಳ + dm_ಸ್ಥಾನಾಂತರ);

    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ int64_t n    = tid / 32;
    ಸ್ಥಿರಾಂಕ int64_t l    = tid - 32 * n;
    ಸ್ಥಿರಾಂಕ int64_t is   = 8 * n + l / 16;

    ಸ್ಥಿರಾಂಕ uint8_t q = qs[32 * n + l];
    dst_t * y = yy + 128 * n;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = (*dm)[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = (*dm)[1];
    y[l+ 0] = dall * (scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (scales[is+0] >> 4);
    y[l+32] = dall * (scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (scales[is+2] >> 4);
    y[l+64] = dall * (scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (scales[is+4] >> 4);
    y[l+96] = dall * (scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (scales[is+6] >> 4);
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(vx);
    GGML_ಬಳಸದ(yy);
    GGML_ಬಳಸದ(i);
    GGML_ಬಳಸದ(ಅಂಶ_ct1);
    GGML_ಬಳಸದ(n_ಖಂಡಗಳು);
    GGML_ABORT("Q2_K reorder dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q3_K(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy,
                                  ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K *) vx;

#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ int64_t r = ಅಂಶ_ct1.get_local_id(2) / 4;
    ಸ್ಥಿರಾಂಕ int64_t tid = r/2;
    ಸ್ಥಿರಾಂಕ int64_t is0 = r%2;
    ಸ್ಥಿರಾಂಕ int64_t l0 = 16 * is0 + 4 * (ಅಂಶ_ct1.get_local_id(2) % 4);
    ಸ್ಥಿರಾಂಕ int64_t n = tid / 4;
    ಸ್ಥಿರಾಂಕ int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    ಪೂರ್ಣಾಂಕ ಸರಿಸು = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    ದಶಮಾಂಶ d_all = x[i].d;
    ದಶಮಾಂಶ dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    ಸ್ಥಿರಾಂಕ uint8_t * q = x[i].qs + 32*n;
    ಸ್ಥಿರಾಂಕ uint8_t * hm = x[i].hmask;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> ಸರಿಸು) & 3) - ((hm[l] & m) ? 0 : 4));
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ int64_t is  = tid/16;  // 0 or 1
    ಸ್ಥಿರಾಂಕ int64_t il  = tid%16;  // 0...15
    ಸ್ಥಿರಾಂಕ int64_t im  = il/8;    // 0...1
    ಸ್ಥಿರಾಂಕ int64_t ಒಳ  = il%8;    // 0...7

    dst_t * y = yy + i*QK_K + 16*is + il;

    ಸ್ಥಿರಾಂಕ uint8_t q = x[i].qs[il] >> (2*is);
    ಸ್ಥಿರಾಂಕ uint8_t h = x[i].hmask[ಒಳ] >> (2*is + im);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   d = (ದಶಮಾಂಶ)x[i].d;

    ಆದರೆ (is == 0) {
        y[ 0] = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    } ಇಲ್ಲದಿದ್ದರೆ {
        y[ 0] = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    }
#ಆದರೆ_ಅಂತ್ಯ

}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q3_K_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, ಸ್ಥಿರಾಂಕ int64_t i,
                                          ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1, int64_t n_ಖಂಡಗಳು) {
#ಆದರೆ QK_K == 256
    ಆದರೆ (i >= n_ಖಂಡಗಳು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint8_t * ತಳ          = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vx);
    ಸ್ಥಿರಾಂಕ size_t    qs_ಸ್ಥಾನಾಂತರ     = i * (QK_K / 4);
    ಸ್ಥಿರಾಂಕ size_t    hmask_ಸ್ಥಾನಾಂತರ  = n_ಖಂಡಗಳು * (QK_K / 4) + i * (QK_K / 8);
    ಸ್ಥಿರಾಂಕ size_t    scales_ಸ್ಥಾನಾಂತರ = n_ಖಂಡಗಳು * (QK_K / 4) + n_ಖಂಡಗಳು * (QK_K / 8) + i * 12;
    ಸ್ಥಿರಾಂಕ size_t    d_ಸ್ಥಾನಾಂತರ      = n_ಖಂಡಗಳು * (QK_K / 4) + n_ಖಂಡಗಳು * (QK_K / 8) + n_ಖಂಡಗಳು * 12 +
                                 i * ಗಾತ್ರ(ggml_half);

    ಸ್ಥಿರಾಂಕ uint8_t * qs     = ತಳ + qs_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ uint8_t * hmask  = ತಳ + hmask_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ uint8_t * scales = ತಳ + scales_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     d_all  = static_cast<ದಶಮಾಂಶ>(*reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half *>(ತಳ + d_ಸ್ಥಾನಾಂತರ));

    ಸ್ಥಿರಾಂಕ int64_t r    = ಅಂಶ_ct1.get_local_id(2) / 4;
    ಸ್ಥಿರಾಂಕ int64_t tid  = r / 2;
    ಸ್ಥಿರಾಂಕ int64_t is0  = r % 2;
    ಸ್ಥಿರಾಂಕ int64_t l0   = 16 * is0 + 4 * (ಅಂಶ_ct1.get_local_id(2) % 4);
    ಸ್ಥಿರಾಂಕ int64_t n    = tid / 4;
    ಸ್ಥಿರಾಂಕ int64_t j    = tid - 4 * n;
    ಸ್ಥಿರಾಂಕ int64_t is   = 8 * n + 2 * j + is0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     ಸರಿಸು = 2 * j;
    uint8_t       m    = 1 << (4 * n + j);

    uint8_t us = is < 4
        ? (scales[is - 0] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4)
        : is < 8
            ? (scales[is - 0] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4)
            : is < 12
                ? (scales[is - 8] >> 4) | (((scales[is + 0] >> 4) & 3) << 4)
                : (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dl = d_all * (us - 32);

    dst_t * y = yy + 128 * n + 32 * j;
    ಸ್ಥಿರಾಂಕ uint8_t * q  = qs + 32 * n;
    ಸ್ಥಿರಾಂಕ uint8_t * hm = hmask;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = l0; l < l0 + 4; ++l) {
        y[l] = dl * ((int8_t) ((q[l] >> ಸರಿಸು) & 3) - ((hm[l] & m) ? 0 : 4));
    }
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(vx);
    GGML_ಬಳಸದ(yy);
    GGML_ಬಳಸದ(i);
    GGML_ಬಳಸದ(ಅಂಶ_ct1);
    GGML_ಬಳಸದ(n_ಖಂಡಗಳು);
    GGML_ABORT("Q3_K reorder dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ QK_K == 256
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ get_scale_min_k4(ಪೂರ್ಣಾಂಕ j, ಸ್ಥಿರಾಂಕ uint8_t * q, uint8_t & d, uint8_t & m) {
    ಆದರೆ (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } ಇಲ್ಲದಿದ್ದರೆ {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#ಆದರೆ_ಅಂತ್ಯ

// Work-items that cover one 256-weight K-quant block: four 64-element chunks, 8 work-items
// per chunk, 4 qs bytes per work-item.
ಸ್ಥಿರ constexpr ಪೂರ್ಣಾಂಕ SYCL_DEQK_WI_ಪ್ರತಿ_ಖಂಡ = 32;

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dequantize_q4_K_wide_one(dst_t * __restrict__ y, ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ qs,
                                            ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ scales, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall,
                                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il = tid >> 3;        // 64-element chunk, 0..3
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir = (tid & 7) * 4;   // 4 qs bytes within the chunk

    uint8_t sc, m;
    get_scale_min_k4(2 * il + 0, scales, sc, m);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1 = dall * sc;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, scales, sc, m);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d2 = dall * sc;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m2 = dmin * m;

    ಸ್ಥಿರಾಂಕ sycl::vec<uint8_t, 4> q = vec_aligned_ತುಂಬು<uint8_t, 4>(qs + 32 * il + ir);

    sycl::vec<dst_t, 4> lo, hi;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
        lo[l] = d1 * (q[l] & 0xF) - m1;
        hi[l] = d2 * (q[l] >> 4)  - m2;
    }
    dst_t * yb = y + 64 * il + ir;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb +  0) = lo;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb + 32) = hi;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q4_K_wide(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb,
                                       ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & it) {
    static_assert(QK_K == 256, "wide K-quant dequant assumes QK_K == 256");
    ಸ್ಥಿರಾಂಕ int64_t g   = it.get_global_id(0);
    ಸ್ಥಿರಾಂಕ int64_t b   = g / SYCL_DEQK_WI_ಪ್ರತಿ_ಖಂಡ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     tid = (ಪೂರ್ಣಾಂಕ) (g % SYCL_DEQK_WI_ಪ್ರತಿ_ಖಂಡ);
    ಆದರೆ (b >= nb) {
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K *) vx + b;
    dequantize_q4_K_wide_one(yy + b * QK_K, x->qs, x->scales, (ದಶಮಾಂಶ) x->dm[0], (ದಶಮಾಂಶ) x->dm[1], tid);
}


ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸಾಲೊಳಗೆ ಶೂನ್ಯ dequantize_q4_K_common(dst_t * __restrict__ y, ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ qs_ptr, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall,
                                   ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin, uint8_t * __restrict__ scales_local, ಪೂರ್ಣಾಂಕ il, ಪೂರ್ಣಾಂಕ ir) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is = 2 * il;
    constexpr ಪೂರ್ಣಾಂಕ n  = 4;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1 = dall * sc;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = dmin * m;

    get_scale_min_k4(is + 1, scales_local, sc, m);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d2 = dall * sc;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m2 = dmin * m;

    sycl::vec<uint8_t, n> q_vec = vec_aligned_ತುಂಬು<uint8_t, n>(qs_ptr + 32 * il + n * ir);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < n; ++l) {
        y[l + 0]  = d1 * (q_vec[l] & 0xF) - m1;
        y[l + 32] = d2 * (q_vec[l] >> 4) - m2;
    }
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q4_K_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, ಸ್ಥಿರಾಂಕ int64_t i,
                                          uint8_t * scales_local, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & ಅಂಶ_ct1, int64_t nb) {
    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(0);  // thread index within block
    ಸ್ಥಿರಾಂಕ int64_t il  = tid / 8;
    ಸ್ಥಿರಾಂಕ int64_t ir  = tid % 8;

    dst_t * y = yy + 64 * il + 4 * ir;

    ಸ್ಥಿರಾಂಕ uint8_t * ತಳ          = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vx);
    ಸ್ಥಿರಾಂಕ size_t    qs_ಸ್ಥಾನಾಂತರ     = i * (QK_K / 2);
    ಸ್ಥಿರಾಂಕ size_t    scales_ಸ್ಥಾನಾಂತರ = nb * (QK_K / 2) + i * K_SCALE_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t    dm_ಸ್ಥಾನಾಂತರ     = nb * (QK_K / 2) + nb * K_SCALE_ಗಾತ್ರ + i * ಗಾತ್ರ(ggml_half2);

    ಸ್ಥಿರಾಂಕ uint8_t *    qs_ptr     = ತಳ + qs_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ uint8_t *    scales_ptr = ತಳ + scales_ಸ್ಥಾನಾಂತರ;
    ggml_half2         dm_ಮೌಲ್ಯಗಳು  = *reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half2 *>(ತಳ + dm_ಸ್ಥಾನಾಂತರ);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = dm_ಮೌಲ್ಯಗಳು.x();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = dm_ಮೌಲ್ಯಗಳು.y();

    ಆದರೆ (tid < 12) {
        scales_local[tid] = scales_ptr[tid];
    }

    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);
    dequantize_q4_K_common(y, qs_ptr, dall, dmin, scales_local, il, ir);
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ dequantize_q5_K_wide_one(dst_t * __restrict__ y, ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ qs,
                                            ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ qh_ptr,
                                            ಸ್ಥಿರಾಂಕ uint8_t * __restrict__ scales, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall,
                                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ tid) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ il = tid >> 3;        // 64-element chunk, 0..3
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ir = (tid & 7) * 4;   // 4 qs bytes within the chunk; qh holds the 5th bit

    uint8_t sc, m;
    get_scale_min_k4(2 * il + 0, scales, sc, m);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1 = dall * sc;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, scales, sc, m);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d2 = dall * sc;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m2 = dmin * m;

    ಸ್ಥಿರಾಂಕ sycl::vec<uint8_t, 4> q = vec_aligned_ತುಂಬು<uint8_t, 4>(qs + 32 * il + ir);
    ಸ್ಥಿರಾಂಕ sycl::vec<uint8_t, 4> h = vec_aligned_ತುಂಬು<uint8_t, 4>(qh_ptr + ir);

    ಸ್ಥಿರಾಂಕ uint8_t hm1 = 1 << (2 * il);
    ಸ್ಥಿರಾಂಕ uint8_t hm2 = hm1 << 1;

    sycl::vec<dst_t, 4> lo, hi;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
        lo[l] = d1 * ((q[l] & 0xF) + ((h[l] & hm1) ? 16 : 0)) - m1;
        hi[l] = d2 * ((q[l] >>  4) + ((h[l] & hm2) ? 16 : 0)) - m2;
    }
    dst_t * yb = y + 64 * il + ir;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb +  0) = lo;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb + 32) = hi;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q5_K_wide(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb,
                                       ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<1> & it) {
    static_assert(QK_K == 256, "wide K-quant dequant assumes QK_K == 256");
    ಸ್ಥಿರಾಂಕ int64_t g   = it.get_global_id(0);
    ಸ್ಥಿರಾಂಕ int64_t b   = g / SYCL_DEQK_WI_ಪ್ರತಿ_ಖಂಡ;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     tid = (ಪೂರ್ಣಾಂಕ) (g % SYCL_DEQK_WI_ಪ್ರತಿ_ಖಂಡ);
    ಆದರೆ (b >= nb) {
        ಹಿಂತಿರುಗಿ;
    }
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K *) vx + b;
    dequantize_q5_K_wide_one(yy + b * QK_K, x->qs, x->qh, x->scales, (ದಶಮಾಂಶ) x->dm[0], (ದಶಮಾಂಶ) x->dm[1], tid);
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q5_K_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, ಸ್ಥಿರಾಂಕ int64_t ib,
                                          uint8_t * scales_local, ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1, int64_t n_ಖಂಡಗಳು) {

#ಆದರೆ QK_K == 256
    // assume 64 threads
    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ int64_t il  = tid / 16;   // 0...3
    ಸ್ಥಿರಾಂಕ int64_t ir  = tid % 16;   // 0...15
    ಸ್ಥಿರಾಂಕ int64_t is  = 2 * il;

    dst_t * y = yy + 64 * il + 2 * ir;

    ಸ್ಥಿರಾಂಕ uint8_t * ತಳ = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vx);

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales (K_SCALE_SIZE per block)] [dm (half2 per block)]
    ಸ್ಥಿರಾಂಕ size_t qs_ಸ್ಥಾನಾಂತರ     = ib * (QK_K / 2);
    ಸ್ಥಿರಾಂಕ size_t qh_ಸ್ಥಾನಾಂತರ     = n_ಖಂಡಗಳು * (QK_K / 2) + ib * (QK_K / 8);
    ಸ್ಥಿರಾಂಕ size_t scales_ಸ್ಥಾನಾಂತರ = n_ಖಂಡಗಳು * (QK_K / 2) + n_ಖಂಡಗಳು * (QK_K / 8) + ib * K_SCALE_ಗಾತ್ರ;
    ಸ್ಥಿರಾಂಕ size_t dm_ಸ್ಥಾನಾಂತರ     = n_ಖಂಡಗಳು * (QK_K / 2) + n_ಖಂಡಗಳು * (QK_K / 8) + n_ಖಂಡಗಳು * K_SCALE_ಗಾತ್ರ + ib * ಗಾತ್ರ(ggml_half2);

    ಸ್ಥಿರಾಂಕ uint8_t *  qs_ptr     = ತಳ + qs_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ uint8_t *  qh_ptr     = ತಳ + qh_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ uint8_t *  scales_ptr = ತಳ + scales_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ ggml_half2 dm_ಮೌಲ್ಯಗಳು  = *reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half2 *>(ತಳ + dm_ಸ್ಥಾನಾಂತರ);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = dm_ಮೌಲ್ಯಗಳು.x();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = dm_ಮೌಲ್ಯಗಳು.y();

    ಸ್ಥಿರಾಂಕ uint8_t * ql = qs_ptr + 32 * il + 2 * ir;
    ಸ್ಥಿರಾಂಕ uint8_t * qh = qh_ptr + 2 * ir;

    ಆದರೆ (tid < K_SCALE_ಗಾತ್ರ) {
        scales_local[tid] = scales_ptr[tid];
    }

    ಅಂಶ_ct1.barrier(sycl::access::fence_space::local_space);

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1 = dall * sc; ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = dmin * m;
    get_scale_min_k4(is + 1, scales_local, sc, m);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d2 = dall * sc; ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m2 = dmin * m;

    uint8_t hm  = 1 << (2 * il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(ib); GGML_ಬಳಸದ(tid); GGML_ಬಳಸದ(yy); GGML_ಬಳಸದ(scales_local); GGML_ಬಳಸದ(n_ಖಂಡಗಳು);
    GGML_ABORT("Q5_K reorder dequantize not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q6_K(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy,
                                  ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K *) vx;

    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);
#ಆದರೆ QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ int64_t ip  = tid/32;   // ip is 0 or 1
    ಸ್ಥಿರಾಂಕ int64_t il  = tid - 32*ip; // 0...32
    ಸ್ಥಿರಾಂಕ int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = x[i].d;

    ಸ್ಥಿರಾಂಕ uint8_t * ql = x[i].ql + 64*ip + il;
    ಸ್ಥಿರಾಂಕ uint8_t   qh = x[i].qh[32*ip + il];
    ಸ್ಥಿರಾಂಕ int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#ಇಲ್ಲದಿದ್ದರೆ

    // assume 32 threads
    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ int64_t ip  = tid/16;         // 0 or 1
    ಸ್ಥಿರಾಂಕ int64_t il  = tid - 16*ip;    // 0...15

    dst_t * y = yy + i*QK_K + 16*ip + il;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = x[i].d;

    ಸ್ಥಿರಾಂಕ uint8_t   ql = x[i].ql[16*ip + il];
    ಸ್ಥಿರಾಂಕ uint8_t   qh = x[i].qh[il] >> (2*ip);
    ಸ್ಥಿರಾಂಕ int8_t  * sc = x[i].scales;

    y[ 0] = d * sc[ip+0] * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[ip+2] * ((int8_t)((ql  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_q6_K_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, ಸ್ಥಿರಾಂಕ int64_t ib,
                                          ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1, int64_t n_ಖಂಡಗಳು) {

    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ int64_t ip  = tid / 32;       // ip is 0 or 1
    ಸ್ಥಿರಾಂಕ int64_t il  = tid - 32 * ip;  // 0...32
    ಸ್ಥಿರಾಂಕ int64_t is  = 8 * ip + il / 16;

    ಸ್ಥಿರಾಂಕ uint8_t *   ತಳ_ptr           = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vx);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ        ql_ಸ್ಥಾನಾಂತರ          = ib * (QK_K / 2);
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ        qh_ಸ್ಥಾನಾಂತರ          = (QK_K / 2) * n_ಖಂಡಗಳು + (QK_K / 4) * ib;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ        ತಳ_scales_ಸ್ಥಾನಾಂತರ = (QK_K / 2) * n_ಖಂಡಗಳು + (QK_K / 4) * n_ಖಂಡಗಳು + (QK_K / 16) * ib;
    ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ        ತಳ_d_ಸ್ಥಾನಾಂತರ      = ((QK_K / 2) + (QK_K / 4) + (QK_K / 16)) * n_ಖಂಡಗಳು;
    ಸ್ಥಿರಾಂಕ uint8_t *   ql_ptr             = ತಳ_ptr + ql_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ uint8_t *   qh_ptr             = ತಳ_ptr + qh_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ uint8_t *   scales_ptr         = ತಳ_ptr + ತಳ_scales_ಸ್ಥಾನಾಂತರ;
    ಸ್ಥಿರಾಂಕ ggml_half * d                  = (ಸ್ಥಿರಾಂಕ ggml_half *) (ತಳ_ptr + ತಳ_d_ಸ್ಥಾನಾಂತರ) + ib;

    dst_t * y = yy + 128 * ip + il;

    ಸ್ಥಿರಾಂಕ uint8_t * ql = ql_ptr + 64 * ip + il;
    ಸ್ಥಿರಾಂಕ uint8_t   qh = *(qh_ptr + 32 * ip + il);
    ಸ್ಥಿರಾಂಕ int8_t *  sc = reinterpret_cast<ಸ್ಥಿರಾಂಕ int8_t *>(scales_ptr + is);

    y[0]  = *d * sc[0] * ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = *d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = *d * sc[4] * ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = *d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_iq2_xxs(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy,
                                     ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1,
                                     ಸ್ಥಿರಾಂಕ uint64_t *iq2xxs_grid_ptr,
                                     ಸ್ಥಿರಾಂಕ uint8_t *ksigns_iq2xs_ptr,
                                     ಸ್ಥಿರಾಂಕ uint8_t *kmask_iq2xs_ptr) {

    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xxs * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xxs  *) vx;

    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ int64_t il = tid/8; // 0...3
    ಸ್ಥಿರಾಂಕ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ಸ್ಥಿರಾಂಕ uint16_t * q2 = x[i].qs + 4*ib;
    ಸ್ಥಿರಾಂಕ uint8_t  * aux8 = (ಸ್ಥಿರಾಂಕ uint8_t *)q2;
    ಸ್ಥಿರಾಂಕ uint8_t  * grid = (ಸ್ಥಿರಾಂಕ uint8_t *)(iq2xxs_grid_ptr + aux8[il]);
    ಸ್ಥಿರಾಂಕ uint32_t aux32 = q2[2] | (q2[3] << 16);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    ಸ್ಥಿರಾಂಕ uint8_t signs = ksigns_iq2xs_ptr[(aux32 >> 7*il) & 127];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs_ptr[j] ? -1.f : 1.f);
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
#ಆದರೆ_ಅಂತ್ಯ

}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_iq2_xs(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy,
                                    ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1,
                                    ಸ್ಥಿರಾಂಕ uint64_t *iq2xs_grid,
                                    ಸ್ಥಿರಾಂಕ uint8_t *ksigns_iq2xs,
                                    ಸ್ಥಿರಾಂಕ uint8_t *kmask_iq2xs) {

    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xs * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xs *) vx;

    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ int64_t il = tid/8; // 0...3
    ಸ್ಥಿರಾಂಕ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ಸ್ಥಿರಾಂಕ uint16_t * q2 = x[i].qs + 4*ib;
    ಸ್ಥಿರಾಂಕ uint8_t  * grid = (ಸ್ಥಿರಾಂಕ uint8_t *)(iq2xs_grid + (q2[il] & 511));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    ಸ್ಥಿರಾಂಕ uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
#ಆದರೆ_ಅಂತ್ಯ

}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
dequantize_ಖಂಡ_iq2_s(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, dst_t *__restrict__ yy,
                       ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_s * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_s *) vx;

    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ int64_t il = tid/8; // 0...3
    ಸ್ಥಿರಾಂಕ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ಸ್ಥಿರಾಂಕ uint8_t * grid = (ಸ್ಥಿರಾಂಕ uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    ಸ್ಥಿರಾಂಕ uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j)
        y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);

#ಆದರೆ_ಅಂತ್ಯ

}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_iq3_xxs(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy,
                                     ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1,
                                     ಸ್ಥಿರಾಂಕ uint32_t *iq3xxs_grid,
                                     ಸ್ಥಿರಾಂಕ uint8_t *ksigns_iq2xs,
                                     ಸ್ಥಿರಾಂಕ uint8_t *kmask_iq2xs) {

    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs  *) vx;

    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ int64_t il = tid/8; // 0...3
    ಸ್ಥಿರಾಂಕ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ಸ್ಥಿರಾಂಕ uint8_t  * q3 = x[i].qs + 8*ib;
    ಸ್ಥಿರಾಂಕ uint16_t * gas = (ಸ್ಥಿರಾಂಕ uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    ಸ್ಥಿರಾಂಕ uint8_t  * grid1 = (ಸ್ಥಿರಾಂಕ uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    ಸ್ಥಿರಾಂಕ uint8_t  * grid2 = (ಸ್ಥಿರಾಂಕ uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    ಸ್ಥಿರಾಂಕ uint32_t aux32 = gas[0] | (gas[1] << 16);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    ಸ್ಥಿರಾಂಕ uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
#ಆದರೆ_ಅಂತ್ಯ

}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_iq3_xxs_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, ಸ್ಥಿರಾಂಕ int64_t i,
                                             ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1, ಸ್ಥಿರಾಂಕ int64_t nb,
                                             ಸ್ಥಿರಾಂಕ uint32_t * iq3xxs_grid,
                                             ಸ್ಥಿರಾಂಕ uint8_t * ksigns_iq2xs,
                                             ಸ್ಥಿರಾಂಕ uint8_t * kmask_iq2xs) {
    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ int64_t il = tid / 8;
    ಸ್ಥಿರಾಂಕ int64_t ib = tid % 8;
    ಸ್ಥಿರಾಂಕ uint8_t * ತಳ = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vx);
    ಸ್ಥಿರಾಂಕ uint8_t * qs = ತಳ + i * (3 * QK_K / 8);
    ಸ್ಥಿರಾಂಕ ggml_half * d = reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half *>(ತಳ + nb * (3 * QK_K / 8));
    ಸ್ಥಿರಾಂಕ uint8_t * q3 = qs + 8 * ib;
    ಸ್ಥಿರಾಂಕ uint16_t * gas = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint16_t *>(qs + QK_K / 4) + 2 * ib;
    ಸ್ಥಿರಾಂಕ uint8_t * grid1 = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(iq3xxs_grid + q3[2*il+0]);
    ಸ್ಥಿರಾಂಕ uint8_t * grid2 = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(iq3xxs_grid + q3[2*il+1]);
    ಸ್ಥಿರಾಂಕ uint32_t aux32 = gas[0] | (gas[1] << 16);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dl = (ದಶಮಾಂಶ) d[i] * (0.5f + (aux32 >> 28)) * 0.5f;
    ಸ್ಥಿರಾಂಕ uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
    dst_t * y = yy + 32 * ib + 8 * il;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; ++j) {
        y[j+0] = dl * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = dl * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(vx);
    GGML_ಬಳಸದ(yy);
    GGML_ಬಳಸದ(i);
    GGML_ಬಳಸದ(ಅಂಶ_ct1);
    GGML_ಬಳಸದ(nb);
    GGML_ಬಳಸದ(iq3xxs_grid);
    GGML_ಬಳಸದ(ksigns_iq2xs);
    GGML_ಬಳಸದ(kmask_iq2xs);
    GGML_ABORT("IQ3_XXS reorder dequantization requires QK_K == 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
dequantize_ಖಂಡ_iq3_s(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, dst_t *__restrict__ yy,
                       ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1,
                       ಸ್ಥಿರಾಂಕ uint8_t *kmask_iq2xs, ಸ್ಥಿರಾಂಕ uint32_t *iq3s_grid) {

    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s *) vx;

    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ int64_t il = tid/8; // 0...3
    ಸ್ಥಿರಾಂಕ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ಸ್ಥಿರಾಂಕ uint8_t * qs = x[i].qs + 8*ib;
    ಸ್ಥಿರಾಂಕ uint8_t * grid1 = (ಸ್ಥಿರಾಂಕ uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    ಸ್ಥಿರಾಂಕ uint8_t * grid2 = (ಸ್ಥಿರಾಂಕ uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    ಸ್ಥಿರಾಂಕ uint8_t signs = x[i].signs[4*ib + il];
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
#ಆದರೆ_ಅಂತ್ಯ

}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
dequantize_ಖಂಡ_iq3_s_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, ಸ್ಥಿರಾಂಕ int64_t i,
                               ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1, ಸ್ಥಿರಾಂಕ int64_t nb,
                               ಸ್ಥಿರಾಂಕ uint8_t * kmask_iq2xs, ಸ್ಥಿರಾಂಕ uint32_t * iq3s_grid) {
    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ int64_t il = tid / 8;
    ಸ್ಥಿರಾಂಕ int64_t ib = tid % 8;
    ಸ್ಥಿರಾಂಕ uint8_t * ತಳ = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vx);
    ಸ್ಥಿರಾಂಕ uint8_t * qs = ತಳ + i * (QK_K / 4);
    ಸ್ಥಿರಾಂಕ uint8_t * qh = ತಳ + nb * (QK_K / 4) + i * (QK_K / 32);
    ಸ್ಥಿರಾಂಕ uint8_t * signs_ತಳ = ತಳ + nb * (QK_K / 4 + QK_K / 32) + i * (QK_K / 8);
    ಸ್ಥಿರಾಂಕ uint8_t * metadata = ತಳ + nb * (QK_K / 4 + QK_K / 32 + QK_K / 8) +
                               i * (ಗಾತ್ರ(ggml_half) + IQ3S_N_SCALE);
    ಸ್ಥಿರಾಂಕ ggml_half d = *reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half *>(metadata);
    ಸ್ಥಿರಾಂಕ uint8_t * scales = metadata + ಗಾತ್ರ(ggml_half);
    ಸ್ಥಿರಾಂಕ uint8_t * qs_ib = qs + 8 * ib;
    ಸ್ಥಿರಾಂಕ uint8_t * grid1 = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(
        iq3s_grid + (qs_ib[2*il+0] | ((qh[ib] << (8 - 2*il)) & 256)));
    ಸ್ಥಿರಾಂಕ uint8_t * grid2 = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(
        iq3s_grid + (qs_ib[2*il+1] | ((qh[ib] << (7 - 2*il)) & 256)));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dl = (ದಶಮಾಂಶ) d * (1 + 2 * ((scales[ib/2] >> (4 * (ib%2))) & 0xf));
    ಸ್ಥಿರಾಂಕ uint8_t signs = signs_ತಳ[4 * ib + il];
    dst_t * y = yy + 32 * ib + 8 * il;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; ++j) {
        y[j+0] = dl * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = dl * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    GGML_ಬಳಸದ(vx);
    GGML_ಬಳಸದ(yy);
    GGML_ಬಳಸದ(i);
    GGML_ಬಳಸದ(ಅಂಶ_ct1);
    GGML_ಬಳಸದ(nb);
    GGML_ಬಳಸದ(kmask_iq2xs);
    GGML_ಬಳಸದ(iq3s_grid);
    GGML_ABORT("IQ3_S reorder dequantization requires QK_K == 256");
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
dequantize_ಖಂಡ_iq1_s(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, dst_t *__restrict__ yy,
                       ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1,
                       ಸ್ಥಿರಾಂಕ uint32_t *iq1s_grid_gpu) {

    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_s * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_s  *) vx;

    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ int64_t il = tid/8; // 0...3
    ಸ್ಥಿರಾಂಕ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; ಸ್ಥಿರಾಂಕ int8_t * q = (ಸ್ಥಿರಾಂಕ int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
#ಆದರೆ_ಅಂತ್ಯ

}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
dequantize_ಖಂಡ_iq1_m(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, dst_t *__restrict__ yy,
                       ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1,
                       ಸ್ಥಿರಾಂಕ uint32_t *iq1s_grid_gpu) {

    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_m * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_m  *) vx;

    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ int64_t il = tid/8; // 0...3
    ಸ್ಥಿರಾಂಕ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ಸ್ಥಿರಾಂಕ uint16_t * sc = (ಸ್ಥಿರಾಂಕ uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; ಸ್ಥಿರಾಂಕ int8_t * q = (ಸ್ಥಿರಾಂಕ int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
#ಆದರೆ_ಅಂತ್ಯ

}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
dequantize_ಖಂಡ_iq4_nl(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, dst_t *__restrict__ yy,
                        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {

    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl *) vx + i*(QK_K/QK4_NL);

    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ int64_t il = tid/8; // 0...3
    ಸ್ಥಿರಾಂಕ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    ಸ್ಥಿರಾಂಕ uint8_t  * q4 = x[ib].qs + 4*il;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)x[ib].d;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }

}


ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
__dpct_inline__ ಸ್ಥಿರ ಶೂನ್ಯ
dequantize_ಖಂಡ_iq4_xs(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vx, dst_t *__restrict__ yy,
                        ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ int64_t i = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs *)vx;

    ಸ್ಥಿರಾಂಕ int64_t tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ int64_t il = tid/8; // 0...3
    ಸ್ಥಿರಾಂಕ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    ಸ್ಥಿರಾಂಕ uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

ಮಾದರಿ<ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_mxfp4(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy,
                                   ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> &ಅಂಶ_ct1) {
    // auto                item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ಸ್ಥಿರಾಂಕ int64_t       i        = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 *) vx + i*(QK_K/QK_MXFP4);

    ಸ್ಥಿರಾಂಕ int64_t    tid = ಅಂಶ_ct1.get_local_id(2);
    ಸ್ಥಿರಾಂಕ int64_t il = tid/8; // 0...3
    ಸ್ಥಿರಾಂಕ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    ಸ್ಥಿರಾಂಕ uint8_t  * q4 = x[ib].qs + 4*il;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = ggml_sycl_e8m0_ಗಮ್ಯ_fp32(x[ib].e);
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_mxfp4[q4[j] & 0xf]*0.5f;
        y[j+16] = d * kvalues_mxfp4[q4[j] >>  4]*0.5f;
    }
}

// Reordered MXFP4 ([qs...][e...], see ggml_sycl_reordered::block_q_t<MXFP4>): one work-item per block.
ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_mxfp4_reorder(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                           ಸ್ಥಿರಾಂಕ sycl::nd_ಅಂಶ<3> & ಅಂಶ_ct1) {
    ಸ್ಥಿರಾಂಕ int64_t ib = (int64_t) ಅಂಶ_ct1.get_ಗುಂಪು(2) * WARP_ಗಾತ್ರ + ಅಂಶ_ct1.get_local_id(2);
    ಆದರೆ (ib >= k / QK_MXFP4) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ uint8_t * qs = (ಸ್ಥಿರಾಂಕ uint8_t *) vx + ib * (QK_MXFP4 / 2);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ     d  = ggml_sycl_e8m0_ಗಮ್ಯ_fp32(((ಸ್ಥಿರಾಂಕ uint8_t *) vx)[k / 2 + ib]) * 0.5f;
    dst_t *         y  = yy + ib * QK_MXFP4;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_MXFP4 / 2; ++j) {
        y[j]                = d * kvalues_mxfp4[qs[j] & 0xf];
        y[j + QK_MXFP4 / 2] = d * kvalues_mxfp4[qs[j] >> 4];
    }
}


ಮಾದರಿ <ಬಗೆ_ಹೆಸರು dst_t>
ಸ್ಥಿರ ಶೂನ್ಯ dequantize_ಖಂಡ_nvfp4(
        ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vx,
        dst_t * __restrict__ yy,
        ಸ್ಥಿರಾಂಕ int64_t ne) {
    ಸ್ವಯಂಚಾಲಿತ          ಅಂಶ_ct1 = sycl::ext::oneapi::this_work_ಅಂಶ::get_nd_ಅಂಶ<3>();
    ಸ್ಥಿರಾಂಕ int64_t i        = ಅಂಶ_ct1.get_ಗುಂಪು(2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     tid      = ಅಂಶ_ct1.get_local_id(2);

    ಸ್ಥಿರಾಂಕ int64_t ತಳ = i * QK_NVFP4;
    ಆದರೆ (ತಳ >= ne) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ ಖಂಡ_nvfp4 * x = (ಸ್ಥಿರಾಂಕ ಖಂಡ_nvfp4 *) vx;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_nvfp4 & xb = x[i];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sub = tid / (QK_NVFP4_SUB / 2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = tid % (QK_NVFP4_SUB / 2);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = ggml_sycl_ue4m3_ಗಮ್ಯ_fp32(xb.d[sub]);
    ಸ್ಥಿರಾಂಕ uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    ಸ್ಥಿರಾಂಕ int64_t y0 = ತಳ + sub * QK_NVFP4_SUB + j;
    ಸ್ಥಿರಾಂಕ int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_sycl_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_sycl_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}


#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_DEQUANTIZE_HPP
