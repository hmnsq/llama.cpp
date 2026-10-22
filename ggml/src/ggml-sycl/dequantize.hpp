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

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_DEQUANTIZE_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_DEQUANTIZE_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "common.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "convert.hpp"

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*dequantize_kernel_t)(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ int64_t ib, ਅਚਲ ਪੂਰਨ iqs, dfloat2 & v);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*dequantize_kernel_t_reorder)(ਅਚਲ ਖਾਲੀ *d, ਅਚਲ int64_t ib, ਅਚਲ ਖਾਲੀ *qs,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v);
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਖਾਲੀ (*dequantize_kernel_f32_t)(ਅਚਲ ਖਾਲੀ * vx, ਅਚਲ int64_t ib, ਅਚਲ ਪੂਰਨ iqs, ਦਸ਼ਮਲਵ & v0, ਦਸ਼ਮਲਵ & v1);

#ਜੇ QK_K == 256
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ get_scale_min_k4(ਪੂਰਨ j, ਅਚਲ uint8_t * q, uint8_t & d, uint8_t & m);
#ਜੇ_ਅੰਤ

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q2_0(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
    ਅਚਲ ਖੰਡ_q2_0 * x = (ਅਚਲ ਖੰਡ_q2_0 *) vx;

    ਅਚਲ dfloat d = x[ib].d;

    ਅਚਲ ਪੂਰਨ byte_idx = iqs / 4;
    ਅਚਲ ਪੂਰਨ ਖਿਸਕਾਓ = (iqs % 4) * 2;
    ਅਚਲ uint8_t vui = x[ib].qs[byte_idx];

    v.x() = (vui >> ਖਿਸਕਾਓ) & 3;
    v.y() = (vui >> (ਖਿਸਕਾਓ + 2)) & 3;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    v.s0() = ((dfloat)v.s0() - 1.0f) * d;
    v.s1() = ((dfloat)v.s1() - 1.0f) * d;
#ਨਹੀਂਤਾਂ
    v.x() = ((dfloat)v.x() - 1.0f) * d;
    v.y() = ((dfloat)v.y() - 1.0f) * d;
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q4_0(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
    ਅਚਲ ਖੰਡ_q4_0 * x = (ਅਚਲ ਖੰਡ_q4_0 *) vx;

    ਅਚਲ dfloat d = x[ib].d;

    ਅਚਲ ਪੂਰਨ vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#ਨਹੀਂਤਾਂ
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q4_0_reorder(ਅਚਲ ਖਾਲੀ *d_ptr, ਅਚਲ int64_t ib, ਅਚਲ ਖਾਲੀ *qs,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
    // const block_q4_0 * x = (const block_q4_0 *) vx;

    ਅਚਲ dfloat d = (ਅਚਲ dfloat)*((ਅਚਲ sycl::half*)d_ptr+ib);

    ਅਚਲ ਪੂਰਨ vui = *((ਅਚਲ uint8_t *)qs+iqs);

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#ਨਹੀਂਤਾਂ
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q1_0_reorder(ਅਚਲ ਖਾਲੀ *d_ptr, ਅਚਲ int64_t ib, ਅਚਲ ਖਾਲੀ *qs,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
    // Q1_0 reorder layout: scale values followed by quantized bits
    ਅਚਲ dfloat d = (ਅਚਲ dfloat)*((ਅਚਲ sycl::half*)d_ptr+ib);

    ਅਚਲ ਪੂਰਨ ਦੋਆਧਾਰੀ_ਅੰਕ_ਸੂਚਕਾਂਕ_0 = iqs + 0;
    ਅਚਲ ਪੂਰਨ ਦੋਆਧਾਰੀ_ਅੰਕ_ਸੂਚਕਾਂਕ_1 = iqs + 1;

    ਅਚਲ ਪੂਰਨ ਦੋਆਧਾਰੀ_ਅੰਕ_0 = (*((ਅਚਲ uint8_t *)qs + ਦੋਆਧਾਰੀ_ਅੰਕ_ਸੂਚਕਾਂਕ_0 / 8) >> (ਦੋਆਧਾਰੀ_ਅੰਕ_ਸੂਚਕਾਂਕ_0 % 8)) & 1;
    ਅਚਲ ਪੂਰਨ ਦੋਆਧਾਰੀ_ਅੰਕ_1 = (*((ਅਚਲ uint8_t *)qs + ਦੋਆਧਾਰੀ_ਅੰਕ_ਸੂਚਕਾਂਕ_1 / 8) >> (ਦੋਆਧਾਰੀ_ਅੰਕ_ਸੂਚਕਾਂਕ_1 % 8)) & 1;

    v.x() = (2 * ਦੋਆਧਾਰੀ_ਅੰਕ_0 - 1) * d;
    v.y() = (2 * ਦੋਆਧਾਰੀ_ਅੰਕ_1 - 1) * d;
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q1_0(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
    ਅਚਲ ਖੰਡ_q1_0 * x = (ਅਚਲ ਖੰਡ_q1_0 *) vx;
    ਅਚਲ dfloat d = x[ib].d;

    ਅਚਲ ਪੂਰਨ ਦੋਆਧਾਰੀ_ਅੰਕ_ਸੂਚਕਾਂਕ_0 = iqs + 0;
    ਅਚਲ ਪੂਰਨ ਦੋਆਧਾਰੀ_ਅੰਕ_ਸੂਚਕਾਂਕ_1 = iqs + 1;

    ਅਚਲ ਪੂਰਨ ਦੋਆਧਾਰੀ_ਅੰਕ_0 = (x[ib].qs[ਦੋਆਧਾਰੀ_ਅੰਕ_ਸੂਚਕਾਂਕ_0 / 8] >> (ਦੋਆਧਾਰੀ_ਅੰਕ_ਸੂਚਕਾਂਕ_0 % 8)) & 1;
    ਅਚਲ ਪੂਰਨ ਦੋਆਧਾਰੀ_ਅੰਕ_1 = (x[ib].qs[ਦੋਆਧਾਰੀ_ਅੰਕ_ਸੂਚਕਾਂਕ_1 / 8] >> (ਦੋਆਧਾਰੀ_ਅੰਕ_ਸੂਚਕਾਂਕ_1 % 8)) & 1;

    v.x() = (2 * ਦੋਆਧਾਰੀ_ਅੰਕ_0 - 1) * d;
    v.y() = (2 * ਦੋਆਧਾਰੀ_ਅੰਕ_1 - 1) * d;
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q4_1(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
    ਅਚਲ ਖੰਡ_q4_1 * x = (ਅਚਲ ਖੰਡ_q4_1 *) vx;

    ਅਚਲ dfloat d = x[ib].dm[0];
    ਅਚਲ dfloat m = x[ib].dm[1];

    ਅਚਲ ਪੂਰਨ vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = sycl::fma(v.s0(), d, m);
    v.s1() = sycl::fma(v.s1(), d, m);

#ਨਹੀਂਤਾਂ
    v.x() = sycl::fma(v.x(), d, m);
    v.y() = sycl::fma(v.y(), d, m);
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q4_K(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_q4_K * x = (ਅਚਲ ਖੰਡ_q4_K *) vx;
    ਅਚਲ sycl::half2 dm = x[ib].dm;
    ਅਚਲ ਦਸ਼ਮਲਵ dall = dm[0];
    ਅਚਲ ਦਸ਼ਮਲਵ dmin = dm[1];

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਅਚਲ ਪੂਰਨ il = idx / 64;
        ਅਚਲ ਪੂਰਨ ਵਿੱਚ = idx % 64;
        ਅਚਲ ਪੂਰਨ is = 2 * il + (ਵਿੱਚ >= 32 ? 1 : 0);
        ਅਚਲ ਪੂਰਨ off = ਵਿੱਚ & 31;
        ਅਚਲ ਪੂਰਨ qsi = 32 * il + off;

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ਅਚਲ uint8_t q = x[ib].qs[qsi];
        ਅਚਲ uint8_t qv = (ਵਿੱਚ >= 32) ? (q >> 4) : (q & 0xF);
        ਵਾਪਸ sycl::fma((dfloat) qv, (dfloat) (dall * sc), (dfloat) (-dmin * m));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("Q4_K dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q4_K_f32(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                                ਅਚਲ ਪੂਰਨ iqs, ਦਸ਼ਮਲਵ &v0, ਦਸ਼ਮਲਵ &v1) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_q4_K * x = (ਅਚਲ ਖੰਡ_q4_K *) vx;
    ਅਚਲ sycl::half2 dm = x[ib].dm;
    ਅਚਲ ਦਸ਼ਮਲਵ dall = dm[0];
    ਅਚਲ ਦਸ਼ਮਲਵ dmin = dm[1];

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> ਦਸ਼ਮਲਵ {
        ਅਚਲ ਪੂਰਨ il = idx / 64;
        ਅਚਲ ਪੂਰਨ ਵਿੱਚ = idx % 64;
        ਅਚਲ ਪੂਰਨ is = 2 * il + (ਵਿੱਚ >= 32 ? 1 : 0);
        ਅਚਲ ਪੂਰਨ qsi = 32 * il + (ਵਿੱਚ & 31);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ਅਚਲ ਦਸ਼ਮਲਵ d = dall * sc;
        ਅਚਲ ਦਸ਼ਮਲਵ mn = dmin * m;
        ਅਚਲ uint8_t q = x[ib].qs[qsi];
        ਅਚਲ uint8_t qv = (ਵਿੱਚ >= 32) ? (q >> 4) : (q & 0xF);

        ਵਾਪਸ d * qv - mn;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("Q4_K dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q2_K(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_q2_K * x = (ਅਚਲ ਖੰਡ_q2_K *) vx;
    ਅਚਲ ਦਸ਼ਮਲਵ dall = x[ib].dm[0];
    ਅਚਲ ਦਸ਼ਮਲਵ dmin = x[ib].dm[1];

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਅਚਲ ਪੂਰਨ n = idx / 128;
        ਅਚਲ ਪੂਰਨ r = idx % 128;
        ਅਚਲ ਪੂਰਨ g = r / 32;
        ਅਚਲ ਪੂਰਨ l = r % 32;
        ਅਚਲ ਪੂਰਨ is = 8 * n + l / 16;

        ਅਚਲ uint8_t q = x[ib].qs[32 * n + l];
        ਅਚਲ uint8_t sc = x[ib].scales[is + 2 * g];
        ਅਚਲ ਦਸ਼ਮਲਵ d = dall * (sc & 0xF);
        ਅਚਲ ਦਸ਼ਮਲਵ m = dmin * (sc >> 4);

        ਵਾਪਸ (dfloat) d * (dfloat) ((q >> (2 * g)) & 3) - (dfloat) m;
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("Q2_K dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q2_K_f32(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                                ਅਚਲ ਪੂਰਨ iqs, ਦਸ਼ਮਲਵ &v0, ਦਸ਼ਮਲਵ &v1) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_q2_K * x = (ਅਚਲ ਖੰਡ_q2_K *) vx;
    ਅਚਲ ਦਸ਼ਮਲਵ dall = x[ib].dm[0];
    ਅਚਲ ਦਸ਼ਮਲਵ dmin = x[ib].dm[1];

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> ਦਸ਼ਮਲਵ {
        ਅਚਲ ਪੂਰਨ n = idx / 128;
        ਅਚਲ ਪੂਰਨ r = idx % 128;
        ਅਚਲ ਪੂਰਨ g = r / 32;
        ਅਚਲ ਪੂਰਨ l = r % 32;
        ਅਚਲ ਪੂਰਨ is = 8 * n + l / 16;

        ਅਚਲ uint8_t q = x[ib].qs[32 * n + l];
        ਅਚਲ uint8_t sc = x[ib].scales[is + 2 * g];
        ਅਚਲ ਦਸ਼ਮਲਵ d = dall * (sc & 0xF);
        ਅਚਲ ਦਸ਼ਮਲਵ m = dmin * (sc >> 4);

        ਵਾਪਸ d * ((q >> (2 * g)) & 3) - m;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("Q2_K dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q3_K(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_q3_K * x = (ਅਚਲ ਖੰਡ_q3_K *) vx;
    ਅਚਲ ਦਸ਼ਮਲਵ d_all = x[ib].d;

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਅਚਲ ਪੂਰਨ n = idx / 128;
        ਅਚਲ ਪੂਰਨ r = idx % 128;
        ਅਚਲ ਪੂਰਨ j = r / 32;
        ਅਚਲ ਪੂਰਨ l = r % 32;

        ਅਚਲ ਪੂਰਨ is0 = l / 16;
        ਅਚਲ ਪੂਰਨ is = 8 * n + 2 * j + is0;
        ਅਚਲ ਪੂਰਨ ਖਿਸਕਾਓ = 2 * j;
        ਅਚਲ uint8_t m = 1 << (4 * n + j);

        ਅਚਲ int8_t us = is <  4 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 8] >> 0) & 3) << 4) :
                         is <  8 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 4] >> 2) & 3) << 4) :
                         is < 12 ? (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is + 0] >> 4) & 3) << 4) :
                                   (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is - 4] >> 6) & 3) << 4);

        ਅਚਲ ਦਸ਼ਮਲਵ dl = d_all * (us - 32);
        ਅਚਲ uint8_t q = x[ib].qs[32 * n + l];
        ਅਚਲ uint8_t h = x[ib].hmask[l];
        ਅਚਲ int8_t qv = ((q >> ਖਿਸਕਾਓ) & 3) - ((h & m) ? 0 : 4);

        ਵਾਪਸ (dfloat) (dl * qv);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("Q3_K dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q5_K(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_q5_K * x = (ਅਚਲ ਖੰਡ_q5_K *) vx;
    ਅਚਲ ਦਸ਼ਮਲਵ dall = x[ib].dm[0];
    ਅਚਲ ਦਸ਼ਮਲਵ dmin = x[ib].dm[1];

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਅਚਲ ਪੂਰਨ il = idx / 64;
        ਅਚਲ ਪੂਰਨ ਵਿੱਚ = idx % 64;
        ਅਚਲ ਪੂਰਨ is = 2 * il + (ਵਿੱਚ >= 32 ? 1 : 0);
        ਅਚਲ ਪੂਰਨ ir = (ਵਿੱਚ & 31) / 2;
        ਅਚਲ ਪੂਰਨ iq = ਵਿੱਚ & 1;

        ਅਚਲ uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        ਅਚਲ uint8_t h = x[ib].qh[2 * ir + iq];
        ਅਚਲ uint8_t qv = (ਵਿੱਚ >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ਅਚਲ ਦਸ਼ਮਲਵ d = dall * sc;
        ਅਚਲ ਦਸ਼ਮਲਵ mn = dmin * m;
        ਅਚਲ uint8_t hm = 1 << (2 * il + (ਵਿੱਚ >= 32 ? 1 : 0));

        ਵਾਪਸ sycl::fma((dfloat) (qv + ((h & hm) ? 16 : 0)), (dfloat) d, (dfloat) (-mn));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("Q5_K dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q5_K_f32(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                                ਅਚਲ ਪੂਰਨ iqs, ਦਸ਼ਮਲਵ &v0, ਦਸ਼ਮਲਵ &v1) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_q5_K * x = (ਅਚਲ ਖੰਡ_q5_K *) vx;
    ਅਚਲ ਦਸ਼ਮਲਵ dall = x[ib].dm[0];
    ਅਚਲ ਦਸ਼ਮਲਵ dmin = x[ib].dm[1];

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> ਦਸ਼ਮਲਵ {
        ਅਚਲ ਪੂਰਨ il = idx / 64;
        ਅਚਲ ਪੂਰਨ ਵਿੱਚ = idx % 64;
        ਅਚਲ ਪੂਰਨ is = 2 * il + (ਵਿੱਚ >= 32 ? 1 : 0);
        ਅਚਲ ਪੂਰਨ ir = (ਵਿੱਚ & 31) / 2;
        ਅਚਲ ਪੂਰਨ iq = ਵਿੱਚ & 1;

        ਅਚਲ uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        ਅਚਲ uint8_t h = x[ib].qh[2 * ir + iq];
        ਅਚਲ uint8_t qv = (ਵਿੱਚ >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ਅਚਲ ਦਸ਼ਮਲਵ d = dall * sc;
        ਅਚਲ ਦਸ਼ਮਲਵ mn = dmin * m;
        ਅਚਲ uint8_t hm = 1 << (2 * il + (ਵਿੱਚ >= 32 ? 1 : 0));

        ਵਾਪਸ (qv + ((h & hm) ? 16 : 0)) * d - mn;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("Q5_K dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q6_K(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_q6_K * x = (ਅਚਲ ਖੰਡ_q6_K *) vx;
    ਅਚਲ ਦਸ਼ਮਲਵ d = x[ib].d;

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਅਚਲ ਪੂਰਨ ip = idx / 128;
        ਅਚਲ ਪੂਰਨ ਵਿੱਚ = idx % 128;
        ਅਚਲ ਪੂਰਨ il = ਵਿੱਚ & 31;
        ਅਚਲ ਪੂਰਨ ig = ਵਿੱਚ / 32;
        ਅਚਲ ਪੂਰਨ is = 8 * ip + il / 16;

        ਅਚਲ uint8_t ql0 = x[ib].ql[64 * ip + il];
        ਅਚਲ uint8_t ql1 = x[ib].ql[64 * ip + il + 32];
        ਅਚਲ uint8_t qh = x[ib].qh[32 * ip + il];
        ਅਚਲ int8_t * sc = x[ib].scales + is;

        uint8_t qv;
        int8_t scale;
        ਜੇ (ig == 0) {
            qv = (ql0 & 0xF) | (((qh >> 0) & 3) << 4);
            scale = sc[0];
        } ਨਹੀਂਤਾਂ ਜੇ (ig == 1) {
            qv = (ql1 & 0xF) | (((qh >> 2) & 3) << 4);
            scale = sc[2];
        } ਨਹੀਂਤਾਂ ਜੇ (ig == 2) {
            qv = (ql0 >> 4) | (((qh >> 4) & 3) << 4);
            scale = sc[4];
        } ਨਹੀਂਤਾਂ {
            qv = (ql1 >> 4) | (((qh >> 6) & 3) << 4);
            scale = sc[6];
        }

        ਵਾਪਸ (dfloat) (d * scale * ((int8_t) qv - 32));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("Q6_K dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_mxfp4(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                             ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
    ਅਚਲ ਖੰਡ_mxfp4 * x = (ਅਚਲ ਖੰਡ_mxfp4 *) vx;
    ਅਚਲ ਦਸ਼ਮਲਵ d = ggml_sycl_e8m0_ਮੰਜ਼ਿਲ_fp32(x[ib].e);
    ਅਚਲ uint8_t q = x[ib].qs[iqs];

    v.x() = d * kvalues_mxfp4[q & 0xF] * 0.5f;
    v.y() = d * kvalues_mxfp4[q >> 4] * 0.5f;
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_nvfp4(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                             ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
    ਅਚਲ ਖੰਡ_nvfp4 & xb = ((ਅਚਲ ਖੰਡ_nvfp4 *) vx)[ib];

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਅਚਲ ਪੂਰਨ sub = idx / QK_NVFP4_SUB;
        ਅਚਲ ਪੂਰਨ j = idx % QK_NVFP4_SUB;
        ਅਚਲ ਪੂਰਨ jh = j % (QK_NVFP4_SUB / 2);

        ਅਚਲ ਦਸ਼ਮਲਵ d = ggml_sycl_ue4m3_ਮੰਜ਼ਿਲ_fp32(xb.d[sub]);
        ਅਚਲ uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + jh];
        ਅਚਲ uint8_t qv = (j < (QK_NVFP4_SUB / 2)) ? (q & 0x0F) : (q >> 4);

        ਵਾਪਸ d * kvalues_mxfp4[qv];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_iq2_xxs(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                               ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq2_xxs * x = (ਅਚਲ ਖੰਡ_iq2_xxs *) vx;

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਅਚਲ ਪੂਰਨ ib8 = idx / 32;
        ਅਚਲ ਪੂਰਨ r = idx % 32;
        ਅਚਲ ਪੂਰਨ il = r / 8;
        ਅਚਲ ਪੂਰਨ j = r % 8;

        ਅਚਲ uint16_t * q2 = x[ib].qs + 4 * ib8;
        ਅਚਲ uint8_t * aux8 = (ਅਚਲ uint8_t *) q2;
        ਅਚਲ uint8_t * grid = (ਅਚਲ uint8_t *) (iq2xxs_grid + aux8[il]);
        ਅਚਲ uint32_t aux32 = q2[2] | (q2[3] << 16);
        ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ) x[ib].d * (0.5f + (aux32 >> 28)) * 0.25f;
        ਅਚਲ uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        ਵਾਪਸ d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("IQ2_XXS dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_iq2_xs(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                              ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq2_xs * x = (ਅਚਲ ਖੰਡ_iq2_xs *) vx;

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਅਚਲ ਪੂਰਨ ib8 = idx / 32;
        ਅਚਲ ਪੂਰਨ r = idx % 32;
        ਅਚਲ ਪੂਰਨ il = r / 8;
        ਅਚਲ ਪੂਰਨ j = r % 8;

        ਅਚਲ uint16_t * q2 = x[ib].qs + 4 * ib8;
        ਅਚਲ uint8_t * grid = (ਅਚਲ uint8_t *) (iq2xs_grid + (q2[il] & 511));
        ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ਅਚਲ uint8_t signs = ksigns_iq2xs[q2[il] >> 9];

        ਵਾਪਸ d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("IQ2_XS dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_iq2_s(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                             ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq2_s * x = (ਅਚਲ ਖੰਡ_iq2_s *) vx;

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਅਚਲ ਪੂਰਨ ib8 = idx / 32;
        ਅਚਲ ਪੂਰਨ r = idx % 32;
        ਅਚਲ ਪੂਰਨ il = r / 8;
        ਅਚਲ ਪੂਰਨ j = r % 8;

        ਅਚਲ uint16_t grid_id = x[ib].qs[4 * ib8 + il] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 0x300);
        ਅਚਲ uint8_t * grid = (ਅਚਲ uint8_t *) (iq2s_grid + grid_id);
        ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ਅਚਲ uint8_t signs = x[ib].qs[QK_K / 8 + 4 * ib8 + il];

        ਵਾਪਸ d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("IQ2_S dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_iq3_xxs(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                               ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq3_xxs * x = (ਅਚਲ ਖੰਡ_iq3_xxs *) vx;

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਅਚਲ ਪੂਰਨ ib8 = idx / 32;
        ਅਚਲ ਪੂਰਨ r = idx % 32;
        ਅਚਲ ਪੂਰਨ il = r / 8;
        ਅਚਲ ਪੂਰਨ j = r % 8;

        ਅਚਲ uint8_t * q3 = x[ib].qs + 8 * ib8;
        ਅਚਲ uint16_t * gas = (ਅਚਲ uint16_t *) (x[ib].qs + QK_K / 4) + 2 * ib8;
        ਅਚਲ uint8_t * grid1 = (ਅਚਲ uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        ਅਚਲ uint8_t * grid2 = (ਅਚਲ uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        ਅਚਲ uint32_t aux32 = gas[0] | (gas[1] << 16);
        ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ) x[ib].d * (0.5f + (aux32 >> 28)) * 0.5f;
        ਅਚਲ uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        ਜੇ (j < 4) {
            ਵਾਪਸ d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        ਵਾਪਸ d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("IQ3_XXS dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_iq3_s(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                             ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq3_s * x = (ਅਚਲ ਖੰਡ_iq3_s *) vx;

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਅਚਲ ਪੂਰਨ ib8 = idx / 32;
        ਅਚਲ ਪੂਰਨ r = idx % 32;
        ਅਚਲ ਪੂਰਨ il = r / 8;
        ਅਚਲ ਪੂਰਨ j = r % 8;

        ਅਚਲ uint8_t * qs = x[ib].qs + 8 * ib8;
        ਅਚਲ uint16_t grid1_id = qs[2 * il + 0] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 256);
        ਅਚਲ uint16_t grid2_id = qs[2 * il + 1] | ((x[ib].qh[ib8] << (7 - 2 * il)) & 256);
        ਅਚਲ uint8_t * grid1 = (ਅਚਲ uint8_t *) (iq3s_grid + grid1_id);
        ਅਚਲ uint8_t * grid2 = (ਅਚਲ uint8_t *) (iq3s_grid + grid2_id);
        ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ) x[ib].d * (1 + 2 * ((x[ib].scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
        ਅਚਲ uint8_t signs = x[ib].signs[4 * ib8 + il];

        ਜੇ (j < 4) {
            ਵਾਪਸ d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        ਵਾਪਸ d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("IQ3_S dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_iq1_s(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                             ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq1_s * x = (ਅਚਲ ਖੰਡ_iq1_s *) vx;

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਅਚਲ ਪੂਰਨ ib8 = idx / 32;
        ਅਚਲ ਪੂਰਨ r = idx % 32;
        ਅਚਲ ਪੂਰਨ il = r / 8;
        ਅਚਲ ਪੂਰਨ j = r % 8;

        ਅਚਲ ਦਸ਼ਮਲਵ delta = (x[ib].qh[ib8] & 0x8000) ? (-1.f - IQ1S_DELTA) : (-1.f + IQ1S_DELTA);
        ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ) x[ib].d * (2 * ((x[ib].qh[ib8] >> 12) & 7) + 1);
        ਅਚਲ uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((x[ib].qh[ib8] >> (3 * il)) & 7) << 8);
        ਅਚਲ uint32_t g = iq1s_grid_gpu[grid_id];
        ਅਚਲ int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        ਵਾਪਸ d * (qv + delta);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("IQ1_S dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_iq1_m(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                             ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq1_m * x = (ਅਚਲ ਖੰਡ_iq1_m *) vx;

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਅਚਲ ਪੂਰਨ ib8 = idx / 32;
        ਅਚਲ ਪੂਰਨ r = idx % 32;
        ਅਚਲ ਪੂਰਨ il = r / 8;
        ਅਚਲ ਪੂਰਨ j = r % 8;

        ਅਚਲ uint16_t * sc = (ਅਚਲ uint16_t *) x[ib].scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        ਅਚਲ ਪੂਰਨ ib16 = 2 * ib8 + il / 2;
        ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);

        ਅਚਲ uint8_t qh = x[ib].qh[2 * ib8 + il / 2];
        ਅਚਲ ਦਸ਼ਮਲਵ delta = (qh & (0x08 << (4 * (il % 2)))) ? (-1.f - IQ1M_DELTA) : (-1.f + IQ1M_DELTA);

        ਅਚਲ uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((qh >> (4 * (il % 2))) & 7) << 8);
        ਅਚਲ uint32_t g = iq1s_grid_gpu[grid_id];
        ਅਚਲ int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        ਵਾਪਸ d * (qv + delta);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("IQ1_M dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_iq4_nl(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                              ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
    ਅਚਲ ਖੰਡ_iq4_nl * x = (ਅਚਲ ਖੰਡ_iq4_nl *) vx;
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ) x[ib].d;

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਜੇ (idx < 16) {
            ਵਾਪਸ d * kvalues_iq4nl[x[ib].qs[idx] & 0xF];
        }
        ਵਾਪਸ d * kvalues_iq4nl[x[ib].qs[idx - 16] >> 4];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_iq4_xs(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                              ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq4_xs * x = (ਅਚਲ ਖੰਡ_iq4_xs *) vx;

    ਸਵੈਚਾਲਿਤ dequantize_one = [&](ਅਚਲ ਪੂਰਨ idx) -> dfloat {
        ਅਚਲ ਪੂਰਨ ib8 = idx / 32;
        ਅਚਲ ਪੂਰਨ r = idx % 32;
        ਅਚਲ ਪੂਰਨ byte_idx = (r < 16) ? r : (r - 16);
        ਅਚਲ uint8_t q = x[ib].qs[16 * ib8 + byte_idx];
        ਅਚਲ uint8_t qv = (r < 16) ? (q & 0x0F) : (q >> 4);

        ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ) x[ib].d * ((((x[ib].scales_l[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf) |
                        (((x[ib].scales_h >> (2 * ib8)) & 3) << 4)) - 32);
        ਵਾਪਸ d * kvalues_iq4nl[qv];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ਨਹੀਂਤਾਂ
    GGML_ABORT("IQ4_XS dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q5_0(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
    ਅਚਲ ਖੰਡ_q5_0 * x = (ਅਚਲ ਖੰਡ_q5_0 *) vx;

    ਅਚਲ dfloat d = x[ib].d;

    uint32_t qh;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&qh, x[ib].qh, ਆਕਾਰ(qh));

    ਅਚਲ ਪੂਰਨ xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    ਅਚਲ ਪੂਰਨ xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    // v = v - {16.0f, 16.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 16.0f) * d;
    v.s1() = (v.s1() - 16.0f) * d;

#ਨਹੀਂਤਾਂ
    v.x() = (v.x() - 16.0f) * d;
    v.y() = (v.y() - 16.0f) * d;
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q5_1(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
    ਅਚਲ ਖੰਡ_q5_1 * x = (ਅਚਲ ਖੰਡ_q5_1 *) vx;

    ਅਚਲ dfloat d = x[ib].dm[0];
    ਅਚਲ dfloat m = x[ib].dm[1];

    uint32_t qh;
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&qh, x[ib].qh, ਆਕਾਰ(qh));

    ਅਚਲ ਪੂਰਨ xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    ਅਚਲ ਪੂਰਨ xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = sycl::fma(v.s0(), d, m);
    v.s1() = sycl::fma(v.s1(), d, m);
#ਨਹੀਂਤਾਂ
    v.x() = sycl::fma(v.x(), d, m);
    v.y() = sycl::fma(v.y(), d, m);
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q8_0_reorder(ਅਚਲ ਖਾਲੀ *d_ptr, ਅਚਲ int64_t ib, ਅਚਲ ਖਾਲੀ *qs,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
    ਅਚਲ dfloat d = (ਅਚਲ dfloat)*((ਅਚਲ sycl::half*)d_ptr + ib);

    v.x() = ((ਅਚਲ int8_t *)qs)[iqs + 0];
    v.y() = ((ਅਚਲ int8_t *)qs)[iqs + 1];

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    v.s0() *= d;
    v.s1() *= d;
#ਨਹੀਂਤਾਂ
    v.x() *= d;
    v.y() *= d;
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ dequantize_q8_0(ਅਚਲ ਖਾਲੀ *vx, ਅਚਲ int64_t ib,
                                            ਅਚਲ ਪੂਰਨ iqs, dfloat2 &v) {
    ਅਚਲ ਖੰਡ_q8_0 * x = (ਅਚਲ ਖੰਡ_q8_0 *) vx;

    ਅਚਲ dfloat d = x[ib].d;

    v.x() = x[ib].qs[iqs + 0];
    v.y() = x[ib].qs[iqs + 1];

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    // v = v * {d, d};
    v.s0() *= d;
    v.s1() *= d;
#ਨਹੀਂਤਾਂ
    v.x() *= d;
    v.y() *= d;
#ਜੇ_ਅੰਤ // GGML_SYCL_F16
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q4_0(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);

    // assume 32 threads
    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ int64_t il  = tid/8;
    ਅਚਲ int64_t ir  = tid%8;
    ਅਚਲ int64_t ib = 8*i + ir;
    ਜੇ (ib >= nb32) {
        ਵਾਪਸ;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    ਅਚਲ ਖੰਡ_q4_0 * x = (ਅਚਲ ਖੰਡ_q4_0 *)vx + ib;
    ਅਚਲ ਦਸ਼ਮਲਵ d = sycl::vec<sycl::half, 1>(x->d)
                        .convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>()[0];
    ਅਚਲ ਦਸ਼ਮਲਵ dm = -8*d;

    ਅਚਲ uint8_t * q = x->qs + 4*il;

    ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q4_0_reorder(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                  ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    // one thread per qs byte, like dequantize_block<dequantize_q4_0>; the reorder layout makes the byte reads contiguous
    // grid-stride loop because the launcher may shrink the work-group to keep the global range below INT_MAX
    // float math, not dequantize_q4_0_reorder: that helper computes in half with GGML_SYCL_F16, which changes f32 output
    ਅਚਲ sycl::half * d_ptr  = (ਅਚਲ sycl::half *) ((ਅਚਲ uint8_t *) vx + k/2);
    ਅਚਲ int64_t      stride = ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2);

    ਲਈ (int64_t iq = ਵਸਤੂ_ct1.get_global_id(2); iq < k/2; iq += stride) {
        ਅਚਲ int64_t ib  = iq / (QK4_0/2);
        ਅਚਲ int64_t iqs = iq % (QK4_0/2);
        ਅਚਲ ਪੂਰਨ     vq  = ((ਅਚਲ uint8_t *) vx)[iq];
        ਅਚਲ ਦਸ਼ਮਲਵ   d   = ਦਸ਼ਮਲਵ(d_ptr[ib]);

        dst_t * y = yy + ib*QK4_0 + iqs;
        y[0]       = d * ((vq & 0xF) - 8);
        y[QK4_0/2] = d * ((vq >>  4) - 8);
    }
}

// Dequantize Q8_0 from reorder layout: [all qs (k bytes)][all d values]
ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q8_0_reorder(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                  ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    // two values per thread, like dequantize_block<dequantize_q8_0>; the reorder layout keeps all qs contiguous
    // grid-stride loop because the launcher may shrink the work-group to keep the global range below INT_MAX
    // float math, not dequantize_q8_0_reorder: that helper computes in half with GGML_SYCL_F16, which changes f32 output
    ਅਚਲ int8_t *     qs     = (ਅਚਲ int8_t *) vx;
    ਅਚਲ sycl::half * d_ptr  = (ਅਚਲ sycl::half *) ((ਅਚਲ uint8_t *) vx + k);
    ਅਚਲ int64_t      stride = 2 * ਵਸਤੂ_ct1.get_local_range(2) * ਵਸਤੂ_ct1.get_ਸਮੂਹ_range(2);

    ਲਈ (int64_t i = 2 * ਵਸਤੂ_ct1.get_global_id(2); i < k; i += stride) {
        ਅਚਲ ਦਸ਼ਮਲਵ d = ਦਸ਼ਮਲਵ(d_ptr[i / QK8_0]);

        yy[i + 0] = d * qs[i + 0];
        yy[i + 1] = d * qs[i + 1];
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q8_0_reorder_wide(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                               ਅਚਲ sycl::id<1> idx) {
    ਅਚਲ int64_t i = 4 * (int64_t) idx[0];

    ਅਚਲ int8_t *     qs    = (ਅਚਲ int8_t *) vx;
    ਅਚਲ sycl::half * d_ptr = (ਅਚਲ sycl::half *) ((ਅਚਲ uint8_t *) vx + k);

    ਅਚਲ ਦਸ਼ਮਲਵ d0 = ਦਸ਼ਮਲਵ(d_ptr[i / QK8_0]);
    ਅਚਲ ਦਸ਼ਮਲਵ d1 = ਦਸ਼ਮਲਵ(d_ptr[(i + 2) / QK8_0]);

    sycl::vec<ਦਸ਼ਮਲਵ, 4> p;
    p[0] = d0 * (ਦਸ਼ਮਲਵ) qs[i + 0];
    p[1] = d0 * (ਦਸ਼ਮਲਵ) qs[i + 1];
    p[2] = d1 * (ਦਸ਼ਮਲਵ) qs[i + 2];
    p[3] = d1 * (ਦਸ਼ਮਲਵ) qs[i + 3];

    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yy + i) = p.convert<dst_t, sycl::rounding_ਬਹੁਲਕ::automatic>();
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q4_1(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);

    // assume 32 threads
    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ int64_t il  = tid/8;
    ਅਚਲ int64_t ir  = tid%8;
    ਅਚਲ int64_t ib = 8*i + ir;
    ਜੇ (ib >= nb32) {
        ਵਾਪਸ;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    ਅਚਲ ਖੰਡ_q4_1 * x = (ਅਚਲ ਖੰਡ_q4_1 *)vx + ib;
    ਅਚਲ sycl::float2 d =
        x->dm.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

    ਅਚਲ uint8_t * q = x->qs + 4*il;

    ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
        y[l + 0] = d.x() * (q[l] & 0xF) + d.y();
        y[l + 16] = d.x() * (q[l] >> 4) + d.y();
    }
}


//================================== k-quants

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q2_K(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy,
                                  ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਖੰਡ_q2_K * x = (ਅਚਲ ਖੰਡ_q2_K *) vx;

    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
#ਜੇ QK_K == 256
    ਅਚਲ int64_t n   = tid/32;
    ਅਚਲ int64_t l   = tid - 32*n;
    ਅਚਲ int64_t is  = 8*n + l/16;

    ਅਚਲ uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    ਦਸ਼ਮਲਵ dall = x[i].dm[0];
    ਦਸ਼ਮਲਵ dmin = x[i].dm[1];
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#ਨਹੀਂਤਾਂ
    ਅਚਲ int64_t is = tid/16;  // 0 or 1
    ਅਚਲ int64_t il = tid%16;  // 0...15
    ਅਚਲ uint8_t q = x[i].qs[il] >> (2*is);
    dst_t * y = yy + i*QK_K + 16*is + il;

    ਦਸ਼ਮਲਵ dall = x[i].dm[0];
    ਦਸ਼ਮਲਵ dmin = x[i].dm[1];
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#ਜੇ_ਅੰਤ

}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q2_K_reorder(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, ਅਚਲ int64_t i,
                                          ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1, int64_t n_ਖੰਡਾਂ) {
#ਜੇ QK_K == 256
    ਜੇ (i >= n_ਖੰਡਾਂ) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint8_t * ਅਧਾਰ          = static_cast<ਅਚਲ uint8_t *>(vx);
    ਅਚਲ size_t    qs_ਵਿੱਥ     = i * (QK_K / 4);
    ਅਚਲ size_t    scales_ਵਿੱਥ = n_ਖੰਡਾਂ * (QK_K / 4) + i * (QK_K / 16);
    ਅਚਲ size_t    dm_ਵਿੱਥ     = n_ਖੰਡਾਂ * (QK_K / 4) + n_ਖੰਡਾਂ * (QK_K / 16) + i * ਆਕਾਰ(ggml_half2);

    ਅਚਲ uint8_t *     qs     = ਅਧਾਰ + qs_ਵਿੱਥ;
    ਅਚਲ uint8_t *     scales = ਅਧਾਰ + scales_ਵਿੱਥ;
    ਅਚਲ ggml_half2 * dm     = reinterpret_cast<ਅਚਲ ggml_half2 *>(ਅਧਾਰ + dm_ਵਿੱਥ);

    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ int64_t n    = tid / 32;
    ਅਚਲ int64_t l    = tid - 32 * n;
    ਅਚਲ int64_t is   = 8 * n + l / 16;

    ਅਚਲ uint8_t q = qs[32 * n + l];
    dst_t * y = yy + 128 * n;

    ਅਚਲ ਦਸ਼ਮਲਵ dall = (*dm)[0];
    ਅਚਲ ਦਸ਼ਮਲਵ dmin = (*dm)[1];
    y[l+ 0] = dall * (scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (scales[is+0] >> 4);
    y[l+32] = dall * (scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (scales[is+2] >> 4);
    y[l+64] = dall * (scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (scales[is+4] >> 4);
    y[l+96] = dall * (scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (scales[is+6] >> 4);
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(vx);
    GGML_ਅਣਵਰਤਿਆ(yy);
    GGML_ਅਣਵਰਤਿਆ(i);
    GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
    GGML_ਅਣਵਰਤਿਆ(n_ਖੰਡਾਂ);
    GGML_ABORT("Q2_K reorder dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q3_K(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy,
                                  ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਖੰਡ_q3_K * x = (ਅਚਲ ਖੰਡ_q3_K *) vx;

#ਜੇ QK_K == 256
    ਅਚਲ int64_t r = ਵਸਤੂ_ct1.get_local_id(2) / 4;
    ਅਚਲ int64_t tid = r/2;
    ਅਚਲ int64_t is0 = r%2;
    ਅਚਲ int64_t l0 = 16 * is0 + 4 * (ਵਸਤੂ_ct1.get_local_id(2) % 4);
    ਅਚਲ int64_t n = tid / 4;
    ਅਚਲ int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    ਪੂਰਨ ਖਿਸਕਾਓ = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    ਦਸ਼ਮਲਵ d_all = x[i].d;
    ਦਸ਼ਮਲਵ dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    ਅਚਲ uint8_t * q = x[i].qs + 32*n;
    ਅਚਲ uint8_t * hm = x[i].hmask;

    ਲਈ (ਪੂਰਨ l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> ਖਿਸਕਾਓ) & 3) - ((hm[l] & m) ? 0 : 4));
#ਨਹੀਂਤਾਂ
    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ int64_t is  = tid/16;  // 0 or 1
    ਅਚਲ int64_t il  = tid%16;  // 0...15
    ਅਚਲ int64_t im  = il/8;    // 0...1
    ਅਚਲ int64_t ਵਿੱਚ  = il%8;    // 0...7

    dst_t * y = yy + i*QK_K + 16*is + il;

    ਅਚਲ uint8_t q = x[i].qs[il] >> (2*is);
    ਅਚਲ uint8_t h = x[i].hmask[ਵਿੱਚ] >> (2*is + im);
    ਅਚਲ ਦਸ਼ਮਲਵ   d = (ਦਸ਼ਮਲਵ)x[i].d;

    ਜੇ (is == 0) {
        y[ 0] = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    } ਨਹੀਂਤਾਂ {
        y[ 0] = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    }
#ਜੇ_ਅੰਤ

}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q3_K_reorder(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, ਅਚਲ int64_t i,
                                          ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1, int64_t n_ਖੰਡਾਂ) {
#ਜੇ QK_K == 256
    ਜੇ (i >= n_ਖੰਡਾਂ) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint8_t * ਅਧਾਰ          = static_cast<ਅਚਲ uint8_t *>(vx);
    ਅਚਲ size_t    qs_ਵਿੱਥ     = i * (QK_K / 4);
    ਅਚਲ size_t    hmask_ਵਿੱਥ  = n_ਖੰਡਾਂ * (QK_K / 4) + i * (QK_K / 8);
    ਅਚਲ size_t    scales_ਵਿੱਥ = n_ਖੰਡਾਂ * (QK_K / 4) + n_ਖੰਡਾਂ * (QK_K / 8) + i * 12;
    ਅਚਲ size_t    d_ਵਿੱਥ      = n_ਖੰਡਾਂ * (QK_K / 4) + n_ਖੰਡਾਂ * (QK_K / 8) + n_ਖੰਡਾਂ * 12 +
                                 i * ਆਕਾਰ(ggml_half);

    ਅਚਲ uint8_t * qs     = ਅਧਾਰ + qs_ਵਿੱਥ;
    ਅਚਲ uint8_t * hmask  = ਅਧਾਰ + hmask_ਵਿੱਥ;
    ਅਚਲ uint8_t * scales = ਅਧਾਰ + scales_ਵਿੱਥ;
    ਅਚਲ ਦਸ਼ਮਲਵ     d_all  = static_cast<ਦਸ਼ਮਲਵ>(*reinterpret_cast<ਅਚਲ ggml_half *>(ਅਧਾਰ + d_ਵਿੱਥ));

    ਅਚਲ int64_t r    = ਵਸਤੂ_ct1.get_local_id(2) / 4;
    ਅਚਲ int64_t tid  = r / 2;
    ਅਚਲ int64_t is0  = r % 2;
    ਅਚਲ int64_t l0   = 16 * is0 + 4 * (ਵਸਤੂ_ct1.get_local_id(2) % 4);
    ਅਚਲ int64_t n    = tid / 4;
    ਅਚਲ int64_t j    = tid - 4 * n;
    ਅਚਲ int64_t is   = 8 * n + 2 * j + is0;
    ਅਚਲ ਪੂਰਨ     ਖਿਸਕਾਓ = 2 * j;
    uint8_t       m    = 1 << (4 * n + j);

    uint8_t us = is < 4
        ? (scales[is - 0] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4)
        : is < 8
            ? (scales[is - 0] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4)
            : is < 12
                ? (scales[is - 8] >> 4) | (((scales[is + 0] >> 4) & 3) << 4)
                : (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);

    ਅਚਲ ਦਸ਼ਮਲਵ dl = d_all * (us - 32);

    dst_t * y = yy + 128 * n + 32 * j;
    ਅਚਲ uint8_t * q  = qs + 32 * n;
    ਅਚਲ uint8_t * hm = hmask;

    ਲਈ (ਪੂਰਨ l = l0; l < l0 + 4; ++l) {
        y[l] = dl * ((int8_t) ((q[l] >> ਖਿਸਕਾਓ) & 3) - ((hm[l] & m) ? 0 : 4));
    }
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(vx);
    GGML_ਅਣਵਰਤਿਆ(yy);
    GGML_ਅਣਵਰਤਿਆ(i);
    GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
    GGML_ਅਣਵਰਤਿਆ(n_ਖੰਡਾਂ);
    GGML_ABORT("Q3_K reorder dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

#ਜੇ QK_K == 256
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ get_scale_min_k4(ਪੂਰਨ j, ਅਚਲ uint8_t * q, uint8_t & d, uint8_t & m) {
    ਜੇ (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } ਨਹੀਂਤਾਂ {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#ਜੇ_ਅੰਤ

// Work-items that cover one 256-weight K-quant block: four 64-element chunks, 8 work-items
// per chunk, 4 qs bytes per work-item.
ਸਥਿਰ constexpr ਪੂਰਨ SYCL_DEQK_WI_ਪ੍ਰਤੀ_ਖੰਡ = 32;

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dequantize_q4_K_wide_one(dst_t * __restrict__ y, ਅਚਲ uint8_t * __restrict__ qs,
                                            ਅਚਲ uint8_t * __restrict__ scales, ਅਚਲ ਦਸ਼ਮਲਵ dall,
                                            ਅਚਲ ਦਸ਼ਮਲਵ dmin, ਅਚਲ ਪੂਰਨ tid) {
    ਅਚਲ ਪੂਰਨ il = tid >> 3;        // 64-element chunk, 0..3
    ਅਚਲ ਪੂਰਨ ir = (tid & 7) * 4;   // 4 qs bytes within the chunk

    uint8_t sc, m;
    get_scale_min_k4(2 * il + 0, scales, sc, m);
    ਅਚਲ ਦਸ਼ਮਲਵ d1 = dall * sc;
    ਅਚਲ ਦਸ਼ਮਲਵ m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, scales, sc, m);
    ਅਚਲ ਦਸ਼ਮਲਵ d2 = dall * sc;
    ਅਚਲ ਦਸ਼ਮਲਵ m2 = dmin * m;

    ਅਚਲ sycl::vec<uint8_t, 4> q = vec_aligned_ਲੱਦੋ<uint8_t, 4>(qs + 32 * il + ir);

    sycl::vec<dst_t, 4> lo, hi;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
        lo[l] = d1 * (q[l] & 0xF) - m1;
        hi[l] = d2 * (q[l] >> 4)  - m2;
    }
    dst_t * yb = y + 64 * il + ir;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb +  0) = lo;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb + 32) = hi;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q4_K_wide(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb,
                                       ਅਚਲ sycl::nd_ਵਸਤੂ<1> & it) {
    static_assert(QK_K == 256, "wide K-quant dequant assumes QK_K == 256");
    ਅਚਲ int64_t g   = it.get_global_id(0);
    ਅਚਲ int64_t b   = g / SYCL_DEQK_WI_ਪ੍ਰਤੀ_ਖੰਡ;
    ਅਚਲ ਪੂਰਨ     tid = (ਪੂਰਨ) (g % SYCL_DEQK_WI_ਪ੍ਰਤੀ_ਖੰਡ);
    ਜੇ (b >= nb) {
        ਵਾਪਸ;
    }
    ਅਚਲ ਖੰਡ_q4_K * x = (ਅਚਲ ਖੰਡ_q4_K *) vx + b;
    dequantize_q4_K_wide_one(yy + b * QK_K, x->qs, x->scales, (ਦਸ਼ਮਲਵ) x->dm[0], (ਦਸ਼ਮਲਵ) x->dm[1], tid);
}


ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਤਰਅੰਦਰ ਖਾਲੀ dequantize_q4_K_common(dst_t * __restrict__ y, ਅਚਲ uint8_t * __restrict__ qs_ptr, ਅਚਲ ਦਸ਼ਮਲਵ dall,
                                   ਅਚਲ ਦਸ਼ਮਲਵ dmin, uint8_t * __restrict__ scales_local, ਪੂਰਨ il, ਪੂਰਨ ir) {
    ਅਚਲ ਪੂਰਨ is = 2 * il;
    constexpr ਪੂਰਨ n  = 4;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    ਅਚਲ ਦਸ਼ਮਲਵ d1 = dall * sc;
    ਅਚਲ ਦਸ਼ਮਲਵ m1 = dmin * m;

    get_scale_min_k4(is + 1, scales_local, sc, m);
    ਅਚਲ ਦਸ਼ਮਲਵ d2 = dall * sc;
    ਅਚਲ ਦਸ਼ਮਲਵ m2 = dmin * m;

    sycl::vec<uint8_t, n> q_vec = vec_aligned_ਲੱਦੋ<uint8_t, n>(qs_ptr + 32 * il + n * ir);
    ਲਈ (ਪੂਰਨ l = 0; l < n; ++l) {
        y[l + 0]  = d1 * (q_vec[l] & 0xF) - m1;
        y[l + 32] = d2 * (q_vec[l] >> 4) - m2;
    }
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q4_K_reorder(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, ਅਚਲ int64_t i,
                                          uint8_t * scales_local, ਅਚਲ sycl::nd_ਵਸਤੂ<1> & ਵਸਤੂ_ct1, int64_t nb) {
    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(0);  // thread index within block
    ਅਚਲ int64_t il  = tid / 8;
    ਅਚਲ int64_t ir  = tid % 8;

    dst_t * y = yy + 64 * il + 4 * ir;

    ਅਚਲ uint8_t * ਅਧਾਰ          = static_cast<ਅਚਲ uint8_t *>(vx);
    ਅਚਲ size_t    qs_ਵਿੱਥ     = i * (QK_K / 2);
    ਅਚਲ size_t    scales_ਵਿੱਥ = nb * (QK_K / 2) + i * K_SCALE_ਮਾਪ;
    ਅਚਲ size_t    dm_ਵਿੱਥ     = nb * (QK_K / 2) + nb * K_SCALE_ਮਾਪ + i * ਆਕਾਰ(ggml_half2);

    ਅਚਲ uint8_t *    qs_ptr     = ਅਧਾਰ + qs_ਵਿੱਥ;
    ਅਚਲ uint8_t *    scales_ptr = ਅਧਾਰ + scales_ਵਿੱਥ;
    ggml_half2         dm_ਮੁੱਲਾਂ  = *reinterpret_cast<ਅਚਲ ggml_half2 *>(ਅਧਾਰ + dm_ਵਿੱਥ);

    ਅਚਲ ਦਸ਼ਮਲਵ dall = dm_ਮੁੱਲਾਂ.x();
    ਅਚਲ ਦਸ਼ਮਲਵ dmin = dm_ਮੁੱਲਾਂ.y();

    ਜੇ (tid < 12) {
        scales_local[tid] = scales_ptr[tid];
    }

    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);
    dequantize_q4_K_common(y, qs_ptr, dall, dmin, scales_local, il, ir);
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ dequantize_q5_K_wide_one(dst_t * __restrict__ y, ਅਚਲ uint8_t * __restrict__ qs,
                                            ਅਚਲ uint8_t * __restrict__ qh_ptr,
                                            ਅਚਲ uint8_t * __restrict__ scales, ਅਚਲ ਦਸ਼ਮਲਵ dall,
                                            ਅਚਲ ਦਸ਼ਮਲਵ dmin, ਅਚਲ ਪੂਰਨ tid) {
    ਅਚਲ ਪੂਰਨ il = tid >> 3;        // 64-element chunk, 0..3
    ਅਚਲ ਪੂਰਨ ir = (tid & 7) * 4;   // 4 qs bytes within the chunk; qh holds the 5th bit

    uint8_t sc, m;
    get_scale_min_k4(2 * il + 0, scales, sc, m);
    ਅਚਲ ਦਸ਼ਮਲਵ d1 = dall * sc;
    ਅਚਲ ਦਸ਼ਮਲਵ m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, scales, sc, m);
    ਅਚਲ ਦਸ਼ਮਲਵ d2 = dall * sc;
    ਅਚਲ ਦਸ਼ਮਲਵ m2 = dmin * m;

    ਅਚਲ sycl::vec<uint8_t, 4> q = vec_aligned_ਲੱਦੋ<uint8_t, 4>(qs + 32 * il + ir);
    ਅਚਲ sycl::vec<uint8_t, 4> h = vec_aligned_ਲੱਦੋ<uint8_t, 4>(qh_ptr + ir);

    ਅਚਲ uint8_t hm1 = 1 << (2 * il);
    ਅਚਲ uint8_t hm2 = hm1 << 1;

    sycl::vec<dst_t, 4> lo, hi;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
        lo[l] = d1 * ((q[l] & 0xF) + ((h[l] & hm1) ? 16 : 0)) - m1;
        hi[l] = d2 * ((q[l] >>  4) + ((h[l] & hm2) ? 16 : 0)) - m2;
    }
    dst_t * yb = y + 64 * il + ir;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb +  0) = lo;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb + 32) = hi;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q5_K_wide(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb,
                                       ਅਚਲ sycl::nd_ਵਸਤੂ<1> & it) {
    static_assert(QK_K == 256, "wide K-quant dequant assumes QK_K == 256");
    ਅਚਲ int64_t g   = it.get_global_id(0);
    ਅਚਲ int64_t b   = g / SYCL_DEQK_WI_ਪ੍ਰਤੀ_ਖੰਡ;
    ਅਚਲ ਪੂਰਨ     tid = (ਪੂਰਨ) (g % SYCL_DEQK_WI_ਪ੍ਰਤੀ_ਖੰਡ);
    ਜੇ (b >= nb) {
        ਵਾਪਸ;
    }
    ਅਚਲ ਖੰਡ_q5_K * x = (ਅਚਲ ਖੰਡ_q5_K *) vx + b;
    dequantize_q5_K_wide_one(yy + b * QK_K, x->qs, x->qh, x->scales, (ਦਸ਼ਮਲਵ) x->dm[0], (ਦਸ਼ਮਲਵ) x->dm[1], tid);
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q5_K_reorder(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, ਅਚਲ int64_t ib,
                                          uint8_t * scales_local, ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1, int64_t n_ਖੰਡਾਂ) {

#ਜੇ QK_K == 256
    // assume 64 threads
    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ int64_t il  = tid / 16;   // 0...3
    ਅਚਲ int64_t ir  = tid % 16;   // 0...15
    ਅਚਲ int64_t is  = 2 * il;

    dst_t * y = yy + 64 * il + 2 * ir;

    ਅਚਲ uint8_t * ਅਧਾਰ = static_cast<ਅਚਲ uint8_t *>(vx);

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales (K_SCALE_SIZE per block)] [dm (half2 per block)]
    ਅਚਲ size_t qs_ਵਿੱਥ     = ib * (QK_K / 2);
    ਅਚਲ size_t qh_ਵਿੱਥ     = n_ਖੰਡਾਂ * (QK_K / 2) + ib * (QK_K / 8);
    ਅਚਲ size_t scales_ਵਿੱਥ = n_ਖੰਡਾਂ * (QK_K / 2) + n_ਖੰਡਾਂ * (QK_K / 8) + ib * K_SCALE_ਮਾਪ;
    ਅਚਲ size_t dm_ਵਿੱਥ     = n_ਖੰਡਾਂ * (QK_K / 2) + n_ਖੰਡਾਂ * (QK_K / 8) + n_ਖੰਡਾਂ * K_SCALE_ਮਾਪ + ib * ਆਕਾਰ(ggml_half2);

    ਅਚਲ uint8_t *  qs_ptr     = ਅਧਾਰ + qs_ਵਿੱਥ;
    ਅਚਲ uint8_t *  qh_ptr     = ਅਧਾਰ + qh_ਵਿੱਥ;
    ਅਚਲ uint8_t *  scales_ptr = ਅਧਾਰ + scales_ਵਿੱਥ;
    ਅਚਲ ggml_half2 dm_ਮੁੱਲਾਂ  = *reinterpret_cast<ਅਚਲ ggml_half2 *>(ਅਧਾਰ + dm_ਵਿੱਥ);

    ਅਚਲ ਦਸ਼ਮਲਵ dall = dm_ਮੁੱਲਾਂ.x();
    ਅਚਲ ਦਸ਼ਮਲਵ dmin = dm_ਮੁੱਲਾਂ.y();

    ਅਚਲ uint8_t * ql = qs_ptr + 32 * il + 2 * ir;
    ਅਚਲ uint8_t * qh = qh_ptr + 2 * ir;

    ਜੇ (tid < K_SCALE_ਮਾਪ) {
        scales_local[tid] = scales_ptr[tid];
    }

    ਵਸਤੂ_ct1.barrier(sycl::access::fence_space::local_space);

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    ਅਚਲ ਦਸ਼ਮਲਵ d1 = dall * sc; ਅਚਲ ਦਸ਼ਮਲਵ m1 = dmin * m;
    get_scale_min_k4(is + 1, scales_local, sc, m);
    ਅਚਲ ਦਸ਼ਮਲਵ d2 = dall * sc; ਅਚਲ ਦਸ਼ਮਲਵ m2 = dmin * m;

    uint8_t hm  = 1 << (2 * il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(ib); GGML_ਅਣਵਰਤਿਆ(tid); GGML_ਅਣਵਰਤਿਆ(yy); GGML_ਅਣਵਰਤਿਆ(scales_local); GGML_ਅਣਵਰਤਿਆ(n_ਖੰਡਾਂ);
    GGML_ABORT("Q5_K reorder dequantize not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q6_K(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy,
                                  ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ ਖੰਡ_q6_K * x = (ਅਚਲ ਖੰਡ_q6_K *) vx;

    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
#ਜੇ QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ int64_t ip  = tid/32;   // ip is 0 or 1
    ਅਚਲ int64_t il  = tid - 32*ip; // 0...32
    ਅਚਲ int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    ਅਚਲ ਦਸ਼ਮਲਵ d = x[i].d;

    ਅਚਲ uint8_t * ql = x[i].ql + 64*ip + il;
    ਅਚਲ uint8_t   qh = x[i].qh[32*ip + il];
    ਅਚਲ int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#ਨਹੀਂਤਾਂ

    // assume 32 threads
    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ int64_t ip  = tid/16;         // 0 or 1
    ਅਚਲ int64_t il  = tid - 16*ip;    // 0...15

    dst_t * y = yy + i*QK_K + 16*ip + il;

    ਅਚਲ ਦਸ਼ਮਲਵ d = x[i].d;

    ਅਚਲ uint8_t   ql = x[i].ql[16*ip + il];
    ਅਚਲ uint8_t   qh = x[i].qh[il] >> (2*ip);
    ਅਚਲ int8_t  * sc = x[i].scales;

    y[ 0] = d * sc[ip+0] * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[ip+2] * ((int8_t)((ql  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_q6_K_reorder(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, ਅਚਲ int64_t ib,
                                          ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1, int64_t n_ਖੰਡਾਂ) {

    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ int64_t ip  = tid / 32;       // ip is 0 or 1
    ਅਚਲ int64_t il  = tid - 32 * ip;  // 0...32
    ਅਚਲ int64_t is  = 8 * ip + il / 16;

    ਅਚਲ uint8_t *   ਅਧਾਰ_ptr           = static_cast<ਅਚਲ uint8_t *>(vx);
    ਅਚਲ ਸਵੈਚਾਲਿਤ        ql_ਵਿੱਥ          = ib * (QK_K / 2);
    ਅਚਲ ਸਵੈਚਾਲਿਤ        qh_ਵਿੱਥ          = (QK_K / 2) * n_ਖੰਡਾਂ + (QK_K / 4) * ib;
    ਅਚਲ ਸਵੈਚਾਲਿਤ        ਅਧਾਰ_scales_ਵਿੱਥ = (QK_K / 2) * n_ਖੰਡਾਂ + (QK_K / 4) * n_ਖੰਡਾਂ + (QK_K / 16) * ib;
    ਅਚਲ ਸਵੈਚਾਲਿਤ        ਅਧਾਰ_d_ਵਿੱਥ      = ((QK_K / 2) + (QK_K / 4) + (QK_K / 16)) * n_ਖੰਡਾਂ;
    ਅਚਲ uint8_t *   ql_ptr             = ਅਧਾਰ_ptr + ql_ਵਿੱਥ;
    ਅਚਲ uint8_t *   qh_ptr             = ਅਧਾਰ_ptr + qh_ਵਿੱਥ;
    ਅਚਲ uint8_t *   scales_ptr         = ਅਧਾਰ_ptr + ਅਧਾਰ_scales_ਵਿੱਥ;
    ਅਚਲ ggml_half * d                  = (ਅਚਲ ggml_half *) (ਅਧਾਰ_ptr + ਅਧਾਰ_d_ਵਿੱਥ) + ib;

    dst_t * y = yy + 128 * ip + il;

    ਅਚਲ uint8_t * ql = ql_ptr + 64 * ip + il;
    ਅਚਲ uint8_t   qh = *(qh_ptr + 32 * ip + il);
    ਅਚਲ int8_t *  sc = reinterpret_cast<ਅਚਲ int8_t *>(scales_ptr + is);

    y[0]  = *d * sc[0] * ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = *d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = *d * sc[4] * ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = *d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_iq2_xxs(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy,
                                     ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1,
                                     ਅਚਲ uint64_t *iq2xxs_grid_ptr,
                                     ਅਚਲ uint8_t *ksigns_iq2xs_ptr,
                                     ਅਚਲ uint8_t *kmask_iq2xs_ptr) {

    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਖੰਡ_iq2_xxs * x = (ਅਚਲ ਖੰਡ_iq2_xxs  *) vx;

    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
#ਜੇ QK_K == 256
    ਅਚਲ int64_t il = tid/8; // 0...3
    ਅਚਲ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ਅਚਲ uint16_t * q2 = x[i].qs + 4*ib;
    ਅਚਲ uint8_t  * aux8 = (ਅਚਲ uint8_t *)q2;
    ਅਚਲ uint8_t  * grid = (ਅਚਲ uint8_t *)(iq2xxs_grid_ptr + aux8[il]);
    ਅਚਲ uint32_t aux32 = q2[2] | (q2[3] << 16);
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    ਅਚਲ uint8_t signs = ksigns_iq2xs_ptr[(aux32 >> 7*il) & 127];
    ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs_ptr[j] ? -1.f : 1.f);
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
#ਜੇ_ਅੰਤ

}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_iq2_xs(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy,
                                    ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1,
                                    ਅਚਲ uint64_t *iq2xs_grid,
                                    ਅਚਲ uint8_t *ksigns_iq2xs,
                                    ਅਚਲ uint8_t *kmask_iq2xs) {

    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਖੰਡ_iq2_xs * x = (ਅਚਲ ਖੰਡ_iq2_xs *) vx;

    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
#ਜੇ QK_K == 256
    ਅਚਲ int64_t il = tid/8; // 0...3
    ਅਚਲ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ਅਚਲ uint16_t * q2 = x[i].qs + 4*ib;
    ਅਚਲ uint8_t  * grid = (ਅਚਲ uint8_t *)(iq2xs_grid + (q2[il] & 511));
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    ਅਚਲ uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
#ਜੇ_ਅੰਤ

}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
__dpct_inline__ ਸਥਿਰ ਖਾਲੀ
dequantize_ਖੰਡ_iq2_s(ਅਚਲ ਖਾਲੀ *__restrict__ vx, dst_t *__restrict__ yy,
                       ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਖੰਡ_iq2_s * x = (ਅਚਲ ਖੰਡ_iq2_s *) vx;

    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
#ਜੇ QK_K == 256
    ਅਚਲ int64_t il = tid/8; // 0...3
    ਅਚਲ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ਅਚਲ uint8_t * grid = (ਅਚਲ uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    ਅਚਲ uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j)
        y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);

#ਜੇ_ਅੰਤ

}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_iq3_xxs(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy,
                                     ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1,
                                     ਅਚਲ uint32_t *iq3xxs_grid,
                                     ਅਚਲ uint8_t *ksigns_iq2xs,
                                     ਅਚਲ uint8_t *kmask_iq2xs) {

    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਖੰਡ_iq3_xxs * x = (ਅਚਲ ਖੰਡ_iq3_xxs  *) vx;

    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
#ਜੇ QK_K == 256
    ਅਚਲ int64_t il = tid/8; // 0...3
    ਅਚਲ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ਅਚਲ uint8_t  * q3 = x[i].qs + 8*ib;
    ਅਚਲ uint16_t * gas = (ਅਚਲ uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    ਅਚਲ uint8_t  * grid1 = (ਅਚਲ uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    ਅਚਲ uint8_t  * grid2 = (ਅਚਲ uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    ਅਚਲ uint32_t aux32 = gas[0] | (gas[1] << 16);
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    ਅਚਲ uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
#ਜੇ_ਅੰਤ

}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_iq3_xxs_reorder(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, ਅਚਲ int64_t i,
                                             ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1, ਅਚਲ int64_t nb,
                                             ਅਚਲ uint32_t * iq3xxs_grid,
                                             ਅਚਲ uint8_t * ksigns_iq2xs,
                                             ਅਚਲ uint8_t * kmask_iq2xs) {
    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
#ਜੇ QK_K == 256
    ਅਚਲ int64_t il = tid / 8;
    ਅਚਲ int64_t ib = tid % 8;
    ਅਚਲ uint8_t * ਅਧਾਰ = static_cast<ਅਚਲ uint8_t *>(vx);
    ਅਚਲ uint8_t * qs = ਅਧਾਰ + i * (3 * QK_K / 8);
    ਅਚਲ ggml_half * d = reinterpret_cast<ਅਚਲ ggml_half *>(ਅਧਾਰ + nb * (3 * QK_K / 8));
    ਅਚਲ uint8_t * q3 = qs + 8 * ib;
    ਅਚਲ uint16_t * gas = reinterpret_cast<ਅਚਲ uint16_t *>(qs + QK_K / 4) + 2 * ib;
    ਅਚਲ uint8_t * grid1 = reinterpret_cast<ਅਚਲ uint8_t *>(iq3xxs_grid + q3[2*il+0]);
    ਅਚਲ uint8_t * grid2 = reinterpret_cast<ਅਚਲ uint8_t *>(iq3xxs_grid + q3[2*il+1]);
    ਅਚਲ uint32_t aux32 = gas[0] | (gas[1] << 16);
    ਅਚਲ ਦਸ਼ਮਲਵ dl = (ਦਸ਼ਮਲਵ) d[i] * (0.5f + (aux32 >> 28)) * 0.5f;
    ਅਚਲ uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
    dst_t * y = yy + 32 * ib + 8 * il;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
        y[j+0] = dl * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = dl * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(vx);
    GGML_ਅਣਵਰਤਿਆ(yy);
    GGML_ਅਣਵਰਤਿਆ(i);
    GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
    GGML_ਅਣਵਰਤਿਆ(nb);
    GGML_ਅਣਵਰਤਿਆ(iq3xxs_grid);
    GGML_ਅਣਵਰਤਿਆ(ksigns_iq2xs);
    GGML_ਅਣਵਰਤਿਆ(kmask_iq2xs);
    GGML_ABORT("IQ3_XXS reorder dequantization requires QK_K == 256");
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
__dpct_inline__ ਸਥਿਰ ਖਾਲੀ
dequantize_ਖੰਡ_iq3_s(ਅਚਲ ਖਾਲੀ *__restrict__ vx, dst_t *__restrict__ yy,
                       ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1,
                       ਅਚਲ uint8_t *kmask_iq2xs, ਅਚਲ uint32_t *iq3s_grid) {

    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਖੰਡ_iq3_s * x = (ਅਚਲ ਖੰਡ_iq3_s *) vx;

    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
#ਜੇ QK_K == 256
    ਅਚਲ int64_t il = tid/8; // 0...3
    ਅਚਲ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ਅਚਲ uint8_t * qs = x[i].qs + 8*ib;
    ਅਚਲ uint8_t * grid1 = (ਅਚਲ uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    ਅਚਲ uint8_t * grid2 = (ਅਚਲ uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    ਅਚਲ uint8_t signs = x[i].signs[4*ib + il];
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
#ਜੇ_ਅੰਤ

}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
__dpct_inline__ ਸਥਿਰ ਖਾਲੀ
dequantize_ਖੰਡ_iq3_s_reorder(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, ਅਚਲ int64_t i,
                               ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1, ਅਚਲ int64_t nb,
                               ਅਚਲ uint8_t * kmask_iq2xs, ਅਚਲ uint32_t * iq3s_grid) {
    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
#ਜੇ QK_K == 256
    ਅਚਲ int64_t il = tid / 8;
    ਅਚਲ int64_t ib = tid % 8;
    ਅਚਲ uint8_t * ਅਧਾਰ = static_cast<ਅਚਲ uint8_t *>(vx);
    ਅਚਲ uint8_t * qs = ਅਧਾਰ + i * (QK_K / 4);
    ਅਚਲ uint8_t * qh = ਅਧਾਰ + nb * (QK_K / 4) + i * (QK_K / 32);
    ਅਚਲ uint8_t * signs_ਅਧਾਰ = ਅਧਾਰ + nb * (QK_K / 4 + QK_K / 32) + i * (QK_K / 8);
    ਅਚਲ uint8_t * metadata = ਅਧਾਰ + nb * (QK_K / 4 + QK_K / 32 + QK_K / 8) +
                               i * (ਆਕਾਰ(ggml_half) + IQ3S_N_SCALE);
    ਅਚਲ ggml_half d = *reinterpret_cast<ਅਚਲ ggml_half *>(metadata);
    ਅਚਲ uint8_t * scales = metadata + ਆਕਾਰ(ggml_half);
    ਅਚਲ uint8_t * qs_ib = qs + 8 * ib;
    ਅਚਲ uint8_t * grid1 = reinterpret_cast<ਅਚਲ uint8_t *>(
        iq3s_grid + (qs_ib[2*il+0] | ((qh[ib] << (8 - 2*il)) & 256)));
    ਅਚਲ uint8_t * grid2 = reinterpret_cast<ਅਚਲ uint8_t *>(
        iq3s_grid + (qs_ib[2*il+1] | ((qh[ib] << (7 - 2*il)) & 256)));
    ਅਚਲ ਦਸ਼ਮਲਵ dl = (ਦਸ਼ਮਲਵ) d * (1 + 2 * ((scales[ib/2] >> (4 * (ib%2))) & 0xf));
    ਅਚਲ uint8_t signs = signs_ਅਧਾਰ[4 * ib + il];
    dst_t * y = yy + 32 * ib + 8 * il;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
        y[j+0] = dl * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = dl * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ਨਹੀਂਤਾਂ
    GGML_ਅਣਵਰਤਿਆ(vx);
    GGML_ਅਣਵਰਤਿਆ(yy);
    GGML_ਅਣਵਰਤਿਆ(i);
    GGML_ਅਣਵਰਤਿਆ(ਵਸਤੂ_ct1);
    GGML_ਅਣਵਰਤਿਆ(nb);
    GGML_ਅਣਵਰਤਿਆ(kmask_iq2xs);
    GGML_ਅਣਵਰਤਿਆ(iq3s_grid);
    GGML_ABORT("IQ3_S reorder dequantization requires QK_K == 256");
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
__dpct_inline__ ਸਥਿਰ ਖਾਲੀ
dequantize_ਖੰਡ_iq1_s(ਅਚਲ ਖਾਲੀ *__restrict__ vx, dst_t *__restrict__ yy,
                       ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1,
                       ਅਚਲ uint32_t *iq1s_grid_gpu) {

    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਖੰਡ_iq1_s * x = (ਅਚਲ ਖੰਡ_iq1_s  *) vx;

    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
#ਜੇ QK_K == 256
    ਅਚਲ int64_t il = tid/8; // 0...3
    ਅਚਲ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ਅਚਲ ਦਸ਼ਮਲਵ delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; ਅਚਲ int8_t * q = (ਅਚਲ int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
#ਜੇ_ਅੰਤ

}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
__dpct_inline__ ਸਥਿਰ ਖਾਲੀ
dequantize_ਖੰਡ_iq1_m(ਅਚਲ ਖਾਲੀ *__restrict__ vx, dst_t *__restrict__ yy,
                       ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1,
                       ਅਚਲ uint32_t *iq1s_grid_gpu) {

    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਖੰਡ_iq1_m * x = (ਅਚਲ ਖੰਡ_iq1_m  *) vx;

    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
#ਜੇ QK_K == 256
    ਅਚਲ int64_t il = tid/8; // 0...3
    ਅਚਲ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ਅਚਲ uint16_t * sc = (ਅਚਲ uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    ਅਚਲ ਪੂਰਨ ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    ਅਚਲ ਦਸ਼ਮਲਵ delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; ਅਚਲ int8_t * q = (ਅਚਲ int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
#ਜੇ_ਅੰਤ

}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
__dpct_inline__ ਸਥਿਰ ਖਾਲੀ
dequantize_ਖੰਡ_iq4_nl(ਅਚਲ ਖਾਲੀ *__restrict__ vx, dst_t *__restrict__ yy,
                        ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {

    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਖੰਡ_iq4_nl * x = (ਅਚਲ ਖੰਡ_iq4_nl *) vx + i*(QK_K/QK4_NL);

    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ int64_t il = tid/8; // 0...3
    ਅਚਲ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    ਅਚਲ uint8_t  * q4 = x[ib].qs + 4*il;
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)x[ib].d;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }

}


ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
__dpct_inline__ ਸਥਿਰ ਖਾਲੀ
dequantize_ਖੰਡ_iq4_xs(ਅਚਲ ਖਾਲੀ *__restrict__ vx, dst_t *__restrict__ yy,
                        ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    ਅਚਲ int64_t i = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਖੰਡ_iq4_xs * x = (ਅਚਲ ਖੰਡ_iq4_xs *)vx;

    ਅਚਲ int64_t tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ int64_t il = tid/8; // 0...3
    ਅਚਲ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    ਅਚਲ uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

ਸਾਂਚਾ<ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_mxfp4(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy,
                                   ਅਚਲ sycl::nd_ਵਸਤੂ<3> &ਵਸਤੂ_ct1) {
    // auto                item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ਅਚਲ int64_t       i        = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਖੰਡ_mxfp4 * x = (ਅਚਲ ਖੰਡ_mxfp4 *) vx + i*(QK_K/QK_MXFP4);

    ਅਚਲ int64_t    tid = ਵਸਤੂ_ct1.get_local_id(2);
    ਅਚਲ int64_t il = tid/8; // 0...3
    ਅਚਲ int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    ਅਚਲ uint8_t  * q4 = x[ib].qs + 4*il;
    ਅਚਲ ਦਸ਼ਮਲਵ d = ggml_sycl_e8m0_ਮੰਜ਼ਿਲ_fp32(x[ib].e);
    ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_mxfp4[q4[j] & 0xf]*0.5f;
        y[j+16] = d * kvalues_mxfp4[q4[j] >>  4]*0.5f;
    }
}

// Reordered MXFP4 ([qs...][e...], see ggml_sycl_reordered::block_q_t<MXFP4>): one work-item per block.
ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_mxfp4_reorder(ਅਚਲ ਖਾਲੀ * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                           ਅਚਲ sycl::nd_ਵਸਤੂ<3> & ਵਸਤੂ_ct1) {
    ਅਚਲ int64_t ib = (int64_t) ਵਸਤੂ_ct1.get_ਸਮੂਹ(2) * WARP_ਮਾਪ + ਵਸਤੂ_ct1.get_local_id(2);
    ਜੇ (ib >= k / QK_MXFP4) {
        ਵਾਪਸ;
    }

    ਅਚਲ uint8_t * qs = (ਅਚਲ uint8_t *) vx + ib * (QK_MXFP4 / 2);
    ਅਚਲ ਦਸ਼ਮਲਵ     d  = ggml_sycl_e8m0_ਮੰਜ਼ਿਲ_fp32(((ਅਚਲ uint8_t *) vx)[k / 2 + ib]) * 0.5f;
    dst_t *         y  = yy + ib * QK_MXFP4;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < QK_MXFP4 / 2; ++j) {
        y[j]                = d * kvalues_mxfp4[qs[j] & 0xf];
        y[j + QK_MXFP4 / 2] = d * kvalues_mxfp4[qs[j] >> 4];
    }
}


ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ dst_t>
ਸਥਿਰ ਖਾਲੀ dequantize_ਖੰਡ_nvfp4(
        ਅਚਲ ਖਾਲੀ * __restrict__ vx,
        dst_t * __restrict__ yy,
        ਅਚਲ int64_t ne) {
    ਸਵੈਚਾਲਿਤ          ਵਸਤੂ_ct1 = sycl::ext::oneapi::this_work_ਵਸਤੂ::get_nd_ਵਸਤੂ<3>();
    ਅਚਲ int64_t i        = ਵਸਤੂ_ct1.get_ਸਮੂਹ(2);
    ਅਚਲ ਪੂਰਨ     tid      = ਵਸਤੂ_ct1.get_local_id(2);

    ਅਚਲ int64_t ਅਧਾਰ = i * QK_NVFP4;
    ਜੇ (ਅਧਾਰ >= ne) {
        ਵਾਪਸ;
    }

    ਅਚਲ ਖੰਡ_nvfp4 * x = (ਅਚਲ ਖੰਡ_nvfp4 *) vx;
    ਅਚਲ ਖੰਡ_nvfp4 & xb = x[i];

    ਅਚਲ ਪੂਰਨ sub = tid / (QK_NVFP4_SUB / 2);
    ਅਚਲ ਪੂਰਨ j = tid % (QK_NVFP4_SUB / 2);

    ਅਚਲ ਦਸ਼ਮਲਵ d = ggml_sycl_ue4m3_ਮੰਜ਼ਿਲ_fp32(xb.d[sub]);
    ਅਚਲ uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    ਅਚਲ int64_t y0 = ਅਧਾਰ + sub * QK_NVFP4_SUB + j;
    ਅਚਲ int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_sycl_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_sycl_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}


#ਜੇ_ਅੰਤ // GGML_SYCL_DEQUANTIZE_HPP
