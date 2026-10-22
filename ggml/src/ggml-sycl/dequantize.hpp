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

#నిర్వచితం_కాకపోతే GGML_SYCL_DEQUANTIZE_HPP
#నిర్వచించు GGML_SYCL_DEQUANTIZE_HPP

#అన్తర్భావ "common.hpp"
#అన్తర్భావ "convert.hpp"

రకనిర్వచనం శూన్య (*dequantize_kernel_t)(స్థిరరాశి శూన్య * vx, స్థిరరాశి int64_t ib, స్థిరరాశి పూర్ణాంకం iqs, dfloat2 & v);
రకనిర్వచనం శూన్య (*dequantize_kernel_t_reorder)(స్థిరరాశి శూన్య *d, స్థిరరాశి int64_t ib, స్థిరరాశి శూన్య *qs,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v);
రకనిర్వచనం శూన్య (*dequantize_kernel_f32_t)(స్థిరరాశి శూన్య * vx, స్థిరరాశి int64_t ib, స్థిరరాశి పూర్ణాంకం iqs, దశాంశం & v0, దశాంశం & v1);

#ఒకవేళ QK_K == 256
స్థిర అంతఃవరుస శూన్య get_scale_min_k4(పూర్ణాంకం j, స్థిరరాశి uint8_t * q, uint8_t & d, uint8_t & m);
#అయితే_ముగింపు

స్థిర __dpct_inline__ శూన్య dequantize_q2_0(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
    స్థిరరాశి ఖండం_q2_0 * x = (స్థిరరాశి ఖండం_q2_0 *) vx;

    స్థిరరాశి dfloat d = x[ib].d;

    స్థిరరాశి పూర్ణాంకం byte_idx = iqs / 4;
    స్థిరరాశి పూర్ణాంకం జరుపు = (iqs % 4) * 2;
    స్థిరరాశి uint8_t vui = x[ib].qs[byte_idx];

    v.x() = (vui >> జరుపు) & 3;
    v.y() = (vui >> (జరుపు + 2)) & 3;

#నిర్వచితమైతే GGML_SYCL_F16
    v.s0() = ((dfloat)v.s0() - 1.0f) * d;
    v.s1() = ((dfloat)v.s1() - 1.0f) * d;
#లేకపోతే
    v.x() = ((dfloat)v.x() - 1.0f) * d;
    v.y() = ((dfloat)v.y() - 1.0f) * d;
#అయితే_ముగింపు // GGML_SYCL_F16
}

స్థిర __dpct_inline__ శూన్య dequantize_q4_0(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
    స్థిరరాశి ఖండం_q4_0 * x = (స్థిరరాశి ఖండం_q4_0 *) vx;

    స్థిరరాశి dfloat d = x[ib].d;

    స్థిరరాశి పూర్ణాంకం vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#నిర్వచితమైతే GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#లేకపోతే
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#అయితే_ముగింపు // GGML_SYCL_F16
}

స్థిర __dpct_inline__ శూన్య dequantize_q4_0_reorder(స్థిరరాశి శూన్య *d_ptr, స్థిరరాశి int64_t ib, స్థిరరాశి శూన్య *qs,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
    // const block_q4_0 * x = (const block_q4_0 *) vx;

    స్థిరరాశి dfloat d = (స్థిరరాశి dfloat)*((స్థిరరాశి sycl::half*)d_ptr+ib);

    స్థిరరాశి పూర్ణాంకం vui = *((స్థిరరాశి uint8_t *)qs+iqs);

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#నిర్వచితమైతే GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#లేకపోతే
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#అయితే_ముగింపు // GGML_SYCL_F16
}

స్థిర __dpct_inline__ శూన్య dequantize_q1_0_reorder(స్థిరరాశి శూన్య *d_ptr, స్థిరరాశి int64_t ib, స్థిరరాశి శూన్య *qs,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
    // Q1_0 reorder layout: scale values followed by quantized bits
    స్థిరరాశి dfloat d = (స్థిరరాశి dfloat)*((స్థిరరాశి sycl::half*)d_ptr+ib);

    స్థిరరాశి పూర్ణాంకం బిట్టు_సూచీ_0 = iqs + 0;
    స్థిరరాశి పూర్ణాంకం బిట్టు_సూచీ_1 = iqs + 1;

    స్థిరరాశి పూర్ణాంకం బిట్టు_0 = (*((స్థిరరాశి uint8_t *)qs + బిట్టు_సూచీ_0 / 8) >> (బిట్టు_సూచీ_0 % 8)) & 1;
    స్థిరరాశి పూర్ణాంకం బిట్టు_1 = (*((స్థిరరాశి uint8_t *)qs + బిట్టు_సూచీ_1 / 8) >> (బిట్టు_సూచీ_1 % 8)) & 1;

    v.x() = (2 * బిట్టు_0 - 1) * d;
    v.y() = (2 * బిట్టు_1 - 1) * d;
}

స్థిర __dpct_inline__ శూన్య dequantize_q1_0(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
    స్థిరరాశి ఖండం_q1_0 * x = (స్థిరరాశి ఖండం_q1_0 *) vx;
    స్థిరరాశి dfloat d = x[ib].d;

    స్థిరరాశి పూర్ణాంకం బిట్టు_సూచీ_0 = iqs + 0;
    స్థిరరాశి పూర్ణాంకం బిట్టు_సూచీ_1 = iqs + 1;

    స్థిరరాశి పూర్ణాంకం బిట్టు_0 = (x[ib].qs[బిట్టు_సూచీ_0 / 8] >> (బిట్టు_సూచీ_0 % 8)) & 1;
    స్థిరరాశి పూర్ణాంకం బిట్టు_1 = (x[ib].qs[బిట్టు_సూచీ_1 / 8] >> (బిట్టు_సూచీ_1 % 8)) & 1;

    v.x() = (2 * బిట్టు_0 - 1) * d;
    v.y() = (2 * బిట్టు_1 - 1) * d;
}

స్థిర __dpct_inline__ శూన్య dequantize_q4_1(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
    స్థిరరాశి ఖండం_q4_1 * x = (స్థిరరాశి ఖండం_q4_1 *) vx;

    స్థిరరాశి dfloat d = x[ib].dm[0];
    స్థిరరాశి dfloat m = x[ib].dm[1];

    స్థిరరాశి పూర్ణాంకం vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#నిర్వచితమైతే GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = sycl::fma(v.s0(), d, m);
    v.s1() = sycl::fma(v.s1(), d, m);

#లేకపోతే
    v.x() = sycl::fma(v.x(), d, m);
    v.y() = sycl::fma(v.y(), d, m);
#అయితే_ముగింపు // GGML_SYCL_F16
}

స్థిర __dpct_inline__ శూన్య dequantize_q4_K(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_q4_K * x = (స్థిరరాశి ఖండం_q4_K *) vx;
    స్థిరరాశి sycl::half2 dm = x[ib].dm;
    స్థిరరాశి దశాంశం dall = dm[0];
    స్థిరరాశి దశాంశం dmin = dm[1];

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        స్థిరరాశి పూర్ణాంకం il = idx / 64;
        స్థిరరాశి పూర్ణాంకం లోపల = idx % 64;
        స్థిరరాశి పూర్ణాంకం is = 2 * il + (లోపల >= 32 ? 1 : 0);
        స్థిరరాశి పూర్ణాంకం off = లోపల & 31;
        స్థిరరాశి పూర్ణాంకం qsi = 32 * il + off;

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        స్థిరరాశి uint8_t q = x[ib].qs[qsi];
        స్థిరరాశి uint8_t qv = (లోపల >= 32) ? (q >> 4) : (q & 0xF);
        తిరిగి sycl::fma((dfloat) qv, (dfloat) (dall * sc), (dfloat) (-dmin * m));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("Q4_K dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_q4_K_f32(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                                స్థిరరాశి పూర్ణాంకం iqs, దశాంశం &v0, దశాంశం &v1) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_q4_K * x = (స్థిరరాశి ఖండం_q4_K *) vx;
    స్థిరరాశి sycl::half2 dm = x[ib].dm;
    స్థిరరాశి దశాంశం dall = dm[0];
    స్థిరరాశి దశాంశం dmin = dm[1];

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> దశాంశం {
        స్థిరరాశి పూర్ణాంకం il = idx / 64;
        స్థిరరాశి పూర్ణాంకం లోపల = idx % 64;
        స్థిరరాశి పూర్ణాంకం is = 2 * il + (లోపల >= 32 ? 1 : 0);
        స్థిరరాశి పూర్ణాంకం qsi = 32 * il + (లోపల & 31);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        స్థిరరాశి దశాంశం d = dall * sc;
        స్థిరరాశి దశాంశం mn = dmin * m;
        స్థిరరాశి uint8_t q = x[ib].qs[qsi];
        స్థిరరాశి uint8_t qv = (లోపల >= 32) ? (q >> 4) : (q & 0xF);

        తిరిగి d * qv - mn;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("Q4_K dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_q2_K(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_q2_K * x = (స్థిరరాశి ఖండం_q2_K *) vx;
    స్థిరరాశి దశాంశం dall = x[ib].dm[0];
    స్థిరరాశి దశాంశం dmin = x[ib].dm[1];

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        స్థిరరాశి పూర్ణాంకం n = idx / 128;
        స్థిరరాశి పూర్ణాంకం r = idx % 128;
        స్థిరరాశి పూర్ణాంకం g = r / 32;
        స్థిరరాశి పూర్ణాంకం l = r % 32;
        స్థిరరాశి పూర్ణాంకం is = 8 * n + l / 16;

        స్థిరరాశి uint8_t q = x[ib].qs[32 * n + l];
        స్థిరరాశి uint8_t sc = x[ib].scales[is + 2 * g];
        స్థిరరాశి దశాంశం d = dall * (sc & 0xF);
        స్థిరరాశి దశాంశం m = dmin * (sc >> 4);

        తిరిగి (dfloat) d * (dfloat) ((q >> (2 * g)) & 3) - (dfloat) m;
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("Q2_K dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_q2_K_f32(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                                స్థిరరాశి పూర్ణాంకం iqs, దశాంశం &v0, దశాంశం &v1) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_q2_K * x = (స్థిరరాశి ఖండం_q2_K *) vx;
    స్థిరరాశి దశాంశం dall = x[ib].dm[0];
    స్థిరరాశి దశాంశం dmin = x[ib].dm[1];

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> దశాంశం {
        స్థిరరాశి పూర్ణాంకం n = idx / 128;
        స్థిరరాశి పూర్ణాంకం r = idx % 128;
        స్థిరరాశి పూర్ణాంకం g = r / 32;
        స్థిరరాశి పూర్ణాంకం l = r % 32;
        స్థిరరాశి పూర్ణాంకం is = 8 * n + l / 16;

        స్థిరరాశి uint8_t q = x[ib].qs[32 * n + l];
        స్థిరరాశి uint8_t sc = x[ib].scales[is + 2 * g];
        స్థిరరాశి దశాంశం d = dall * (sc & 0xF);
        స్థిరరాశి దశాంశం m = dmin * (sc >> 4);

        తిరిగి d * ((q >> (2 * g)) & 3) - m;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("Q2_K dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_q3_K(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_q3_K * x = (స్థిరరాశి ఖండం_q3_K *) vx;
    స్థిరరాశి దశాంశం d_all = x[ib].d;

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        స్థిరరాశి పూర్ణాంకం n = idx / 128;
        స్థిరరాశి పూర్ణాంకం r = idx % 128;
        స్థిరరాశి పూర్ణాంకం j = r / 32;
        స్థిరరాశి పూర్ణాంకం l = r % 32;

        స్థిరరాశి పూర్ణాంకం is0 = l / 16;
        స్థిరరాశి పూర్ణాంకం is = 8 * n + 2 * j + is0;
        స్థిరరాశి పూర్ణాంకం జరుపు = 2 * j;
        స్థిరరాశి uint8_t m = 1 << (4 * n + j);

        స్థిరరాశి int8_t us = is <  4 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 8] >> 0) & 3) << 4) :
                         is <  8 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 4] >> 2) & 3) << 4) :
                         is < 12 ? (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is + 0] >> 4) & 3) << 4) :
                                   (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is - 4] >> 6) & 3) << 4);

        స్థిరరాశి దశాంశం dl = d_all * (us - 32);
        స్థిరరాశి uint8_t q = x[ib].qs[32 * n + l];
        స్థిరరాశి uint8_t h = x[ib].hmask[l];
        స్థిరరాశి int8_t qv = ((q >> జరుపు) & 3) - ((h & m) ? 0 : 4);

        తిరిగి (dfloat) (dl * qv);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("Q3_K dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_q5_K(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_q5_K * x = (స్థిరరాశి ఖండం_q5_K *) vx;
    స్థిరరాశి దశాంశం dall = x[ib].dm[0];
    స్థిరరాశి దశాంశం dmin = x[ib].dm[1];

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        స్థిరరాశి పూర్ణాంకం il = idx / 64;
        స్థిరరాశి పూర్ణాంకం లోపల = idx % 64;
        స్థిరరాశి పూర్ణాంకం is = 2 * il + (లోపల >= 32 ? 1 : 0);
        స్థిరరాశి పూర్ణాంకం ir = (లోపల & 31) / 2;
        స్థిరరాశి పూర్ణాంకం iq = లోపల & 1;

        స్థిరరాశి uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        స్థిరరాశి uint8_t h = x[ib].qh[2 * ir + iq];
        స్థిరరాశి uint8_t qv = (లోపల >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        స్థిరరాశి దశాంశం d = dall * sc;
        స్థిరరాశి దశాంశం mn = dmin * m;
        స్థిరరాశి uint8_t hm = 1 << (2 * il + (లోపల >= 32 ? 1 : 0));

        తిరిగి sycl::fma((dfloat) (qv + ((h & hm) ? 16 : 0)), (dfloat) d, (dfloat) (-mn));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("Q5_K dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_q5_K_f32(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                                స్థిరరాశి పూర్ణాంకం iqs, దశాంశం &v0, దశాంశం &v1) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_q5_K * x = (స్థిరరాశి ఖండం_q5_K *) vx;
    స్థిరరాశి దశాంశం dall = x[ib].dm[0];
    స్థిరరాశి దశాంశం dmin = x[ib].dm[1];

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> దశాంశం {
        స్థిరరాశి పూర్ణాంకం il = idx / 64;
        స్థిరరాశి పూర్ణాంకం లోపల = idx % 64;
        స్థిరరాశి పూర్ణాంకం is = 2 * il + (లోపల >= 32 ? 1 : 0);
        స్థిరరాశి పూర్ణాంకం ir = (లోపల & 31) / 2;
        స్థిరరాశి పూర్ణాంకం iq = లోపల & 1;

        స్థిరరాశి uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        స్థిరరాశి uint8_t h = x[ib].qh[2 * ir + iq];
        స్థిరరాశి uint8_t qv = (లోపల >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        స్థిరరాశి దశాంశం d = dall * sc;
        స్థిరరాశి దశాంశం mn = dmin * m;
        స్థిరరాశి uint8_t hm = 1 << (2 * il + (లోపల >= 32 ? 1 : 0));

        తిరిగి (qv + ((h & hm) ? 16 : 0)) * d - mn;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("Q5_K dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_q6_K(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_q6_K * x = (స్థిరరాశి ఖండం_q6_K *) vx;
    స్థిరరాశి దశాంశం d = x[ib].d;

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        స్థిరరాశి పూర్ణాంకం ip = idx / 128;
        స్థిరరాశి పూర్ణాంకం లోపల = idx % 128;
        స్థిరరాశి పూర్ణాంకం il = లోపల & 31;
        స్థిరరాశి పూర్ణాంకం ig = లోపల / 32;
        స్థిరరాశి పూర్ణాంకం is = 8 * ip + il / 16;

        స్థిరరాశి uint8_t ql0 = x[ib].ql[64 * ip + il];
        స్థిరరాశి uint8_t ql1 = x[ib].ql[64 * ip + il + 32];
        స్థిరరాశి uint8_t qh = x[ib].qh[32 * ip + il];
        స్థిరరాశి int8_t * sc = x[ib].scales + is;

        uint8_t qv;
        int8_t scale;
        ఒకవేళ (ig == 0) {
            qv = (ql0 & 0xF) | (((qh >> 0) & 3) << 4);
            scale = sc[0];
        } లేకపోతే ఒకవేళ (ig == 1) {
            qv = (ql1 & 0xF) | (((qh >> 2) & 3) << 4);
            scale = sc[2];
        } లేకపోతే ఒకవేళ (ig == 2) {
            qv = (ql0 >> 4) | (((qh >> 4) & 3) << 4);
            scale = sc[4];
        } లేకపోతే {
            qv = (ql1 >> 4) | (((qh >> 6) & 3) << 4);
            scale = sc[6];
        }

        తిరిగి (dfloat) (d * scale * ((int8_t) qv - 32));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("Q6_K dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_mxfp4(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                             స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
    స్థిరరాశి ఖండం_mxfp4 * x = (స్థిరరాశి ఖండం_mxfp4 *) vx;
    స్థిరరాశి దశాంశం d = ggml_sycl_e8m0_గమ్యం_fp32(x[ib].e);
    స్థిరరాశి uint8_t q = x[ib].qs[iqs];

    v.x() = d * kvalues_mxfp4[q & 0xF] * 0.5f;
    v.y() = d * kvalues_mxfp4[q >> 4] * 0.5f;
}

స్థిర __dpct_inline__ శూన్య dequantize_nvfp4(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                             స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
    స్థిరరాశి ఖండం_nvfp4 & xb = ((స్థిరరాశి ఖండం_nvfp4 *) vx)[ib];

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        స్థిరరాశి పూర్ణాంకం sub = idx / QK_NVFP4_SUB;
        స్థిరరాశి పూర్ణాంకం j = idx % QK_NVFP4_SUB;
        స్థిరరాశి పూర్ణాంకం jh = j % (QK_NVFP4_SUB / 2);

        స్థిరరాశి దశాంశం d = ggml_sycl_ue4m3_గమ్యం_fp32(xb.d[sub]);
        స్థిరరాశి uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + jh];
        స్థిరరాశి uint8_t qv = (j < (QK_NVFP4_SUB / 2)) ? (q & 0x0F) : (q >> 4);

        తిరిగి d * kvalues_mxfp4[qv];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
}

స్థిర __dpct_inline__ శూన్య dequantize_iq2_xxs(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                               స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq2_xxs * x = (స్థిరరాశి ఖండం_iq2_xxs *) vx;

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        స్థిరరాశి పూర్ణాంకం ib8 = idx / 32;
        స్థిరరాశి పూర్ణాంకం r = idx % 32;
        స్థిరరాశి పూర్ణాంకం il = r / 8;
        స్థిరరాశి పూర్ణాంకం j = r % 8;

        స్థిరరాశి uint16_t * q2 = x[ib].qs + 4 * ib8;
        స్థిరరాశి uint8_t * aux8 = (స్థిరరాశి uint8_t *) q2;
        స్థిరరాశి uint8_t * grid = (స్థిరరాశి uint8_t *) (iq2xxs_grid + aux8[il]);
        స్థిరరాశి uint32_t aux32 = q2[2] | (q2[3] << 16);
        స్థిరరాశి దశాంశం d = (దశాంశం) x[ib].d * (0.5f + (aux32 >> 28)) * 0.25f;
        స్థిరరాశి uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        తిరిగి d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("IQ2_XXS dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_iq2_xs(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                              స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq2_xs * x = (స్థిరరాశి ఖండం_iq2_xs *) vx;

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        స్థిరరాశి పూర్ణాంకం ib8 = idx / 32;
        స్థిరరాశి పూర్ణాంకం r = idx % 32;
        స్థిరరాశి పూర్ణాంకం il = r / 8;
        స్థిరరాశి పూర్ణాంకం j = r % 8;

        స్థిరరాశి uint16_t * q2 = x[ib].qs + 4 * ib8;
        స్థిరరాశి uint8_t * grid = (స్థిరరాశి uint8_t *) (iq2xs_grid + (q2[il] & 511));
        స్థిరరాశి దశాంశం d = (దశాంశం) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        స్థిరరాశి uint8_t signs = ksigns_iq2xs[q2[il] >> 9];

        తిరిగి d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("IQ2_XS dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_iq2_s(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                             స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq2_s * x = (స్థిరరాశి ఖండం_iq2_s *) vx;

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        స్థిరరాశి పూర్ణాంకం ib8 = idx / 32;
        స్థిరరాశి పూర్ణాంకం r = idx % 32;
        స్థిరరాశి పూర్ణాంకం il = r / 8;
        స్థిరరాశి పూర్ణాంకం j = r % 8;

        స్థిరరాశి uint16_t grid_id = x[ib].qs[4 * ib8 + il] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 0x300);
        స్థిరరాశి uint8_t * grid = (స్థిరరాశి uint8_t *) (iq2s_grid + grid_id);
        స్థిరరాశి దశాంశం d = (దశాంశం) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        స్థిరరాశి uint8_t signs = x[ib].qs[QK_K / 8 + 4 * ib8 + il];

        తిరిగి d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("IQ2_S dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_iq3_xxs(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                               స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq3_xxs * x = (స్థిరరాశి ఖండం_iq3_xxs *) vx;

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        స్థిరరాశి పూర్ణాంకం ib8 = idx / 32;
        స్థిరరాశి పూర్ణాంకం r = idx % 32;
        స్థిరరాశి పూర్ణాంకం il = r / 8;
        స్థిరరాశి పూర్ణాంకం j = r % 8;

        స్థిరరాశి uint8_t * q3 = x[ib].qs + 8 * ib8;
        స్థిరరాశి uint16_t * gas = (స్థిరరాశి uint16_t *) (x[ib].qs + QK_K / 4) + 2 * ib8;
        స్థిరరాశి uint8_t * grid1 = (స్థిరరాశి uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        స్థిరరాశి uint8_t * grid2 = (స్థిరరాశి uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        స్థిరరాశి uint32_t aux32 = gas[0] | (gas[1] << 16);
        స్థిరరాశి దశాంశం d = (దశాంశం) x[ib].d * (0.5f + (aux32 >> 28)) * 0.5f;
        స్థిరరాశి uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        ఒకవేళ (j < 4) {
            తిరిగి d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        తిరిగి d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("IQ3_XXS dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_iq3_s(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                             స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq3_s * x = (స్థిరరాశి ఖండం_iq3_s *) vx;

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        స్థిరరాశి పూర్ణాంకం ib8 = idx / 32;
        స్థిరరాశి పూర్ణాంకం r = idx % 32;
        స్థిరరాశి పూర్ణాంకం il = r / 8;
        స్థిరరాశి పూర్ణాంకం j = r % 8;

        స్థిరరాశి uint8_t * qs = x[ib].qs + 8 * ib8;
        స్థిరరాశి uint16_t grid1_id = qs[2 * il + 0] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 256);
        స్థిరరాశి uint16_t grid2_id = qs[2 * il + 1] | ((x[ib].qh[ib8] << (7 - 2 * il)) & 256);
        స్థిరరాశి uint8_t * grid1 = (స్థిరరాశి uint8_t *) (iq3s_grid + grid1_id);
        స్థిరరాశి uint8_t * grid2 = (స్థిరరాశి uint8_t *) (iq3s_grid + grid2_id);
        స్థిరరాశి దశాంశం d = (దశాంశం) x[ib].d * (1 + 2 * ((x[ib].scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
        స్థిరరాశి uint8_t signs = x[ib].signs[4 * ib8 + il];

        ఒకవేళ (j < 4) {
            తిరిగి d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        తిరిగి d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("IQ3_S dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_iq1_s(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                             స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq1_s * x = (స్థిరరాశి ఖండం_iq1_s *) vx;

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        స్థిరరాశి పూర్ణాంకం ib8 = idx / 32;
        స్థిరరాశి పూర్ణాంకం r = idx % 32;
        స్థిరరాశి పూర్ణాంకం il = r / 8;
        స్థిరరాశి పూర్ణాంకం j = r % 8;

        స్థిరరాశి దశాంశం delta = (x[ib].qh[ib8] & 0x8000) ? (-1.f - IQ1S_DELTA) : (-1.f + IQ1S_DELTA);
        స్థిరరాశి దశాంశం d = (దశాంశం) x[ib].d * (2 * ((x[ib].qh[ib8] >> 12) & 7) + 1);
        స్థిరరాశి uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((x[ib].qh[ib8] >> (3 * il)) & 7) << 8);
        స్థిరరాశి uint32_t g = iq1s_grid_gpu[grid_id];
        స్థిరరాశి int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        తిరిగి d * (qv + delta);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("IQ1_S dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_iq1_m(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                             స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq1_m * x = (స్థిరరాశి ఖండం_iq1_m *) vx;

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        స్థిరరాశి పూర్ణాంకం ib8 = idx / 32;
        స్థిరరాశి పూర్ణాంకం r = idx % 32;
        స్థిరరాశి పూర్ణాంకం il = r / 8;
        స్థిరరాశి పూర్ణాంకం j = r % 8;

        స్థిరరాశి uint16_t * sc = (స్థిరరాశి uint16_t *) x[ib].scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        స్థిరరాశి పూర్ణాంకం ib16 = 2 * ib8 + il / 2;
        స్థిరరాశి దశాంశం d = (దశాంశం) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);

        స్థిరరాశి uint8_t qh = x[ib].qh[2 * ib8 + il / 2];
        స్థిరరాశి దశాంశం delta = (qh & (0x08 << (4 * (il % 2)))) ? (-1.f - IQ1M_DELTA) : (-1.f + IQ1M_DELTA);

        స్థిరరాశి uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((qh >> (4 * (il % 2))) & 7) << 8);
        స్థిరరాశి uint32_t g = iq1s_grid_gpu[grid_id];
        స్థిరరాశి int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        తిరిగి d * (qv + delta);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("IQ1_M dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_iq4_nl(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                              స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
    స్థిరరాశి ఖండం_iq4_nl * x = (స్థిరరాశి ఖండం_iq4_nl *) vx;
    స్థిరరాశి దశాంశం d = (దశాంశం) x[ib].d;

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        ఒకవేళ (idx < 16) {
            తిరిగి d * kvalues_iq4nl[x[ib].qs[idx] & 0xF];
        }
        తిరిగి d * kvalues_iq4nl[x[ib].qs[idx - 16] >> 4];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
}

స్థిర __dpct_inline__ శూన్య dequantize_iq4_xs(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                              స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq4_xs * x = (స్థిరరాశి ఖండం_iq4_xs *) vx;

    స్వయంచాలక dequantize_one = [&](స్థిరరాశి పూర్ణాంకం idx) -> dfloat {
        స్థిరరాశి పూర్ణాంకం ib8 = idx / 32;
        స్థిరరాశి పూర్ణాంకం r = idx % 32;
        స్థిరరాశి పూర్ణాంకం byte_idx = (r < 16) ? r : (r - 16);
        స్థిరరాశి uint8_t q = x[ib].qs[16 * ib8 + byte_idx];
        స్థిరరాశి uint8_t qv = (r < 16) ? (q & 0x0F) : (q >> 4);

        స్థిరరాశి దశాంశం d = (దశాంశం) x[ib].d * ((((x[ib].scales_l[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf) |
                        (((x[ib].scales_h >> (2 * ib8)) & 3) << 4)) - 32);
        తిరిగి d * kvalues_iq4nl[qv];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#లేకపోతే
    GGML_ABORT("IQ4_XS dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

స్థిర __dpct_inline__ శూన్య dequantize_q5_0(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
    స్థిరరాశి ఖండం_q5_0 * x = (స్థిరరాశి ఖండం_q5_0 *) vx;

    స్థిరరాశి dfloat d = x[ib].d;

    uint32_t qh;
    జ్ఞాపకం_నకలు(&qh, x[ib].qh, పరిమాణం(qh));

    స్థిరరాశి పూర్ణాంకం xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    స్థిరరాశి పూర్ణాంకం xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#నిర్వచితమైతే GGML_SYCL_F16
    // v = v - {16.0f, 16.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 16.0f) * d;
    v.s1() = (v.s1() - 16.0f) * d;

#లేకపోతే
    v.x() = (v.x() - 16.0f) * d;
    v.y() = (v.y() - 16.0f) * d;
#అయితే_ముగింపు // GGML_SYCL_F16
}

స్థిర __dpct_inline__ శూన్య dequantize_q5_1(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
    స్థిరరాశి ఖండం_q5_1 * x = (స్థిరరాశి ఖండం_q5_1 *) vx;

    స్థిరరాశి dfloat d = x[ib].dm[0];
    స్థిరరాశి dfloat m = x[ib].dm[1];

    uint32_t qh;
    జ్ఞాపకం_నకలు(&qh, x[ib].qh, పరిమాణం(qh));

    స్థిరరాశి పూర్ణాంకం xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    స్థిరరాశి పూర్ణాంకం xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#నిర్వచితమైతే GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = sycl::fma(v.s0(), d, m);
    v.s1() = sycl::fma(v.s1(), d, m);
#లేకపోతే
    v.x() = sycl::fma(v.x(), d, m);
    v.y() = sycl::fma(v.y(), d, m);
#అయితే_ముగింపు // GGML_SYCL_F16
}

స్థిర __dpct_inline__ శూన్య dequantize_q8_0_reorder(స్థిరరాశి శూన్య *d_ptr, స్థిరరాశి int64_t ib, స్థిరరాశి శూన్య *qs,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
    స్థిరరాశి dfloat d = (స్థిరరాశి dfloat)*((స్థిరరాశి sycl::half*)d_ptr + ib);

    v.x() = ((స్థిరరాశి int8_t *)qs)[iqs + 0];
    v.y() = ((స్థిరరాశి int8_t *)qs)[iqs + 1];

#నిర్వచితమైతే GGML_SYCL_F16
    v.s0() *= d;
    v.s1() *= d;
#లేకపోతే
    v.x() *= d;
    v.y() *= d;
#అయితే_ముగింపు // GGML_SYCL_F16
}

స్థిర __dpct_inline__ శూన్య dequantize_q8_0(స్థిరరాశి శూన్య *vx, స్థిరరాశి int64_t ib,
                                            స్థిరరాశి పూర్ణాంకం iqs, dfloat2 &v) {
    స్థిరరాశి ఖండం_q8_0 * x = (స్థిరరాశి ఖండం_q8_0 *) vx;

    స్థిరరాశి dfloat d = x[ib].d;

    v.x() = x[ib].qs[iqs + 0];
    v.y() = x[ib].qs[iqs + 1];

#నిర్వచితమైతే GGML_SYCL_F16
    // v = v * {d, d};
    v.s0() *= d;
    v.s1() *= d;
#లేకపోతే
    v.x() *= d;
    v.y() *= d;
#అయితే_ముగింపు // GGML_SYCL_F16
}

మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q4_0(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి int64_t i = అంశం_ct1.get_సమూహం(2);

    // assume 32 threads
    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి int64_t il  = tid/8;
    స్థిరరాశి int64_t ir  = tid%8;
    స్థిరరాశి int64_t ib = 8*i + ir;
    ఒకవేళ (ib >= nb32) {
        తిరిగి;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    స్థిరరాశి ఖండం_q4_0 * x = (స్థిరరాశి ఖండం_q4_0 *)vx + ib;
    స్థిరరాశి దశాంశం d = sycl::vec<sycl::half, 1>(x->d)
                        .convert<దశాంశం, sycl::rounding_బహుళకం::automatic>()[0];
    స్థిరరాశి దశాంశం dm = -8*d;

    స్థిరరాశి uint8_t * q = x->qs + 4*il;

    కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q4_0_reorder(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                  స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    // one thread per qs byte, like dequantize_block<dequantize_q4_0>; the reorder layout makes the byte reads contiguous
    // grid-stride loop because the launcher may shrink the work-group to keep the global range below INT_MAX
    // float math, not dequantize_q4_0_reorder: that helper computes in half with GGML_SYCL_F16, which changes f32 output
    స్థిరరాశి sycl::half * d_ptr  = (స్థిరరాశి sycl::half *) ((స్థిరరాశి uint8_t *) vx + k/2);
    స్థిరరాశి int64_t      stride = అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం_range(2);

    కొరకు (int64_t iq = అంశం_ct1.get_global_id(2); iq < k/2; iq += stride) {
        స్థిరరాశి int64_t ib  = iq / (QK4_0/2);
        స్థిరరాశి int64_t iqs = iq % (QK4_0/2);
        స్థిరరాశి పూర్ణాంకం     vq  = ((స్థిరరాశి uint8_t *) vx)[iq];
        స్థిరరాశి దశాంశం   d   = దశాంశం(d_ptr[ib]);

        dst_t * y = yy + ib*QK4_0 + iqs;
        y[0]       = d * ((vq & 0xF) - 8);
        y[QK4_0/2] = d * ((vq >>  4) - 8);
    }
}

// Dequantize Q8_0 from reorder layout: [all qs (k bytes)][all d values]
మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q8_0_reorder(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                  స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    // two values per thread, like dequantize_block<dequantize_q8_0>; the reorder layout keeps all qs contiguous
    // grid-stride loop because the launcher may shrink the work-group to keep the global range below INT_MAX
    // float math, not dequantize_q8_0_reorder: that helper computes in half with GGML_SYCL_F16, which changes f32 output
    స్థిరరాశి int8_t *     qs     = (స్థిరరాశి int8_t *) vx;
    స్థిరరాశి sycl::half * d_ptr  = (స్థిరరాశి sycl::half *) ((స్థిరరాశి uint8_t *) vx + k);
    స్థిరరాశి int64_t      stride = 2 * అంశం_ct1.get_local_range(2) * అంశం_ct1.get_సమూహం_range(2);

    కొరకు (int64_t i = 2 * అంశం_ct1.get_global_id(2); i < k; i += stride) {
        స్థిరరాశి దశాంశం d = దశాంశం(d_ptr[i / QK8_0]);

        yy[i + 0] = d * qs[i + 0];
        yy[i + 1] = d * qs[i + 1];
    }
}

మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q8_0_reorder_wide(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                               స్థిరరాశి sycl::id<1> idx) {
    స్థిరరాశి int64_t i = 4 * (int64_t) idx[0];

    స్థిరరాశి int8_t *     qs    = (స్థిరరాశి int8_t *) vx;
    స్థిరరాశి sycl::half * d_ptr = (స్థిరరాశి sycl::half *) ((స్థిరరాశి uint8_t *) vx + k);

    స్థిరరాశి దశాంశం d0 = దశాంశం(d_ptr[i / QK8_0]);
    స్థిరరాశి దశాంశం d1 = దశాంశం(d_ptr[(i + 2) / QK8_0]);

    sycl::vec<దశాంశం, 4> p;
    p[0] = d0 * (దశాంశం) qs[i + 0];
    p[1] = d0 * (దశాంశం) qs[i + 1];
    p[2] = d1 * (దశాంశం) qs[i + 2];
    p[3] = d1 * (దశాంశం) qs[i + 3];

    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yy + i) = p.convert<dst_t, sycl::rounding_బహుళకం::automatic>();
}

మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q4_1(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి int64_t i = అంశం_ct1.get_సమూహం(2);

    // assume 32 threads
    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి int64_t il  = tid/8;
    స్థిరరాశి int64_t ir  = tid%8;
    స్థిరరాశి int64_t ib = 8*i + ir;
    ఒకవేళ (ib >= nb32) {
        తిరిగి;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    స్థిరరాశి ఖండం_q4_1 * x = (స్థిరరాశి ఖండం_q4_1 *)vx + ib;
    స్థిరరాశి sycl::float2 d =
        x->dm.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

    స్థిరరాశి uint8_t * q = x->qs + 4*il;

    కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
        y[l + 0] = d.x() * (q[l] & 0xF) + d.y();
        y[l + 16] = d.x() * (q[l] >> 4) + d.y();
    }
}


//================================== k-quants

మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q2_K(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy,
                                  స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి int64_t i = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి ఖండం_q2_K * x = (స్థిరరాశి ఖండం_q2_K *) vx;

    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
#ఒకవేళ QK_K == 256
    స్థిరరాశి int64_t n   = tid/32;
    స్థిరరాశి int64_t l   = tid - 32*n;
    స్థిరరాశి int64_t is  = 8*n + l/16;

    స్థిరరాశి uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    దశాంశం dall = x[i].dm[0];
    దశాంశం dmin = x[i].dm[1];
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#లేకపోతే
    స్థిరరాశి int64_t is = tid/16;  // 0 or 1
    స్థిరరాశి int64_t il = tid%16;  // 0...15
    స్థిరరాశి uint8_t q = x[i].qs[il] >> (2*is);
    dst_t * y = yy + i*QK_K + 16*is + il;

    దశాంశం dall = x[i].dm[0];
    దశాంశం dmin = x[i].dm[1];
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#అయితే_ముగింపు

}

మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q2_K_reorder(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, స్థిరరాశి int64_t i,
                                          స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1, int64_t n_ఖండాలు) {
#ఒకవేళ QK_K == 256
    ఒకవేళ (i >= n_ఖండాలు) {
        తిరిగి;
    }

    స్థిరరాశి uint8_t * పునాది          = static_cast<స్థిరరాశి uint8_t *>(vx);
    స్థిరరాశి size_t    qs_స్థానభ్రంశం     = i * (QK_K / 4);
    స్థిరరాశి size_t    scales_స్థానభ్రంశం = n_ఖండాలు * (QK_K / 4) + i * (QK_K / 16);
    స్థిరరాశి size_t    dm_స్థానభ్రంశం     = n_ఖండాలు * (QK_K / 4) + n_ఖండాలు * (QK_K / 16) + i * పరిమాణం(ggml_half2);

    స్థిరరాశి uint8_t *     qs     = పునాది + qs_స్థానభ్రంశం;
    స్థిరరాశి uint8_t *     scales = పునాది + scales_స్థానభ్రంశం;
    స్థిరరాశి ggml_half2 * dm     = reinterpret_cast<స్థిరరాశి ggml_half2 *>(పునాది + dm_స్థానభ్రంశం);

    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి int64_t n    = tid / 32;
    స్థిరరాశి int64_t l    = tid - 32 * n;
    స్థిరరాశి int64_t is   = 8 * n + l / 16;

    స్థిరరాశి uint8_t q = qs[32 * n + l];
    dst_t * y = yy + 128 * n;

    స్థిరరాశి దశాంశం dall = (*dm)[0];
    స్థిరరాశి దశాంశం dmin = (*dm)[1];
    y[l+ 0] = dall * (scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (scales[is+0] >> 4);
    y[l+32] = dall * (scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (scales[is+2] >> 4);
    y[l+64] = dall * (scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (scales[is+4] >> 4);
    y[l+96] = dall * (scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (scales[is+6] >> 4);
#లేకపోతే
    GGML_వాడని(vx);
    GGML_వాడని(yy);
    GGML_వాడని(i);
    GGML_వాడని(అంశం_ct1);
    GGML_వాడని(n_ఖండాలు);
    GGML_ABORT("Q2_K reorder dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q3_K(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy,
                                  స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి int64_t i = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి ఖండం_q3_K * x = (స్థిరరాశి ఖండం_q3_K *) vx;

#ఒకవేళ QK_K == 256
    స్థిరరాశి int64_t r = అంశం_ct1.get_local_id(2) / 4;
    స్థిరరాశి int64_t tid = r/2;
    స్థిరరాశి int64_t is0 = r%2;
    స్థిరరాశి int64_t l0 = 16 * is0 + 4 * (అంశం_ct1.get_local_id(2) % 4);
    స్థిరరాశి int64_t n = tid / 4;
    స్థిరరాశి int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    పూర్ణాంకం జరుపు = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    దశాంశం d_all = x[i].d;
    దశాంశం dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    స్థిరరాశి uint8_t * q = x[i].qs + 32*n;
    స్థిరరాశి uint8_t * hm = x[i].hmask;

    కొరకు (పూర్ణాంకం l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> జరుపు) & 3) - ((hm[l] & m) ? 0 : 4));
#లేకపోతే
    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి int64_t is  = tid/16;  // 0 or 1
    స్థిరరాశి int64_t il  = tid%16;  // 0...15
    స్థిరరాశి int64_t im  = il/8;    // 0...1
    స్థిరరాశి int64_t లోపల  = il%8;    // 0...7

    dst_t * y = yy + i*QK_K + 16*is + il;

    స్థిరరాశి uint8_t q = x[i].qs[il] >> (2*is);
    స్థిరరాశి uint8_t h = x[i].hmask[లోపల] >> (2*is + im);
    స్థిరరాశి దశాంశం   d = (దశాంశం)x[i].d;

    ఒకవేళ (is == 0) {
        y[ 0] = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    } లేకపోతే {
        y[ 0] = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    }
#అయితే_ముగింపు

}

మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q3_K_reorder(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, స్థిరరాశి int64_t i,
                                          స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1, int64_t n_ఖండాలు) {
#ఒకవేళ QK_K == 256
    ఒకవేళ (i >= n_ఖండాలు) {
        తిరిగి;
    }

    స్థిరరాశి uint8_t * పునాది          = static_cast<స్థిరరాశి uint8_t *>(vx);
    స్థిరరాశి size_t    qs_స్థానభ్రంశం     = i * (QK_K / 4);
    స్థిరరాశి size_t    hmask_స్థానభ్రంశం  = n_ఖండాలు * (QK_K / 4) + i * (QK_K / 8);
    స్థిరరాశి size_t    scales_స్థానభ్రంశం = n_ఖండాలు * (QK_K / 4) + n_ఖండాలు * (QK_K / 8) + i * 12;
    స్థిరరాశి size_t    d_స్థానభ్రంశం      = n_ఖండాలు * (QK_K / 4) + n_ఖండాలు * (QK_K / 8) + n_ఖండాలు * 12 +
                                 i * పరిమాణం(ggml_half);

    స్థిరరాశి uint8_t * qs     = పునాది + qs_స్థానభ్రంశం;
    స్థిరరాశి uint8_t * hmask  = పునాది + hmask_స్థానభ్రంశం;
    స్థిరరాశి uint8_t * scales = పునాది + scales_స్థానభ్రంశం;
    స్థిరరాశి దశాంశం     d_all  = static_cast<దశాంశం>(*reinterpret_cast<స్థిరరాశి ggml_half *>(పునాది + d_స్థానభ్రంశం));

    స్థిరరాశి int64_t r    = అంశం_ct1.get_local_id(2) / 4;
    స్థిరరాశి int64_t tid  = r / 2;
    స్థిరరాశి int64_t is0  = r % 2;
    స్థిరరాశి int64_t l0   = 16 * is0 + 4 * (అంశం_ct1.get_local_id(2) % 4);
    స్థిరరాశి int64_t n    = tid / 4;
    స్థిరరాశి int64_t j    = tid - 4 * n;
    స్థిరరాశి int64_t is   = 8 * n + 2 * j + is0;
    స్థిరరాశి పూర్ణాంకం     జరుపు = 2 * j;
    uint8_t       m    = 1 << (4 * n + j);

    uint8_t us = is < 4
        ? (scales[is - 0] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4)
        : is < 8
            ? (scales[is - 0] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4)
            : is < 12
                ? (scales[is - 8] >> 4) | (((scales[is + 0] >> 4) & 3) << 4)
                : (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);

    స్థిరరాశి దశాంశం dl = d_all * (us - 32);

    dst_t * y = yy + 128 * n + 32 * j;
    స్థిరరాశి uint8_t * q  = qs + 32 * n;
    స్థిరరాశి uint8_t * hm = hmask;

    కొరకు (పూర్ణాంకం l = l0; l < l0 + 4; ++l) {
        y[l] = dl * ((int8_t) ((q[l] >> జరుపు) & 3) - ((hm[l] & m) ? 0 : 4));
    }
#లేకపోతే
    GGML_వాడని(vx);
    GGML_వాడని(yy);
    GGML_వాడని(i);
    GGML_వాడని(అంశం_ct1);
    GGML_వాడని(n_ఖండాలు);
    GGML_ABORT("Q3_K reorder dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

#ఒకవేళ QK_K == 256
స్థిర అంతఃవరుస శూన్య get_scale_min_k4(పూర్ణాంకం j, స్థిరరాశి uint8_t * q, uint8_t & d, uint8_t & m) {
    ఒకవేళ (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } లేకపోతే {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#అయితే_ముగింపు

// Work-items that cover one 256-weight K-quant block: four 64-element chunks, 8 work-items
// per chunk, 4 qs bytes per work-item.
స్థిర constexpr పూర్ణాంకం SYCL_DEQK_WI_చొప్పున_ఖండం = 32;

మూస <రకం_పేరు dst_t>
స్థిర అంతఃవరుస శూన్య dequantize_q4_K_wide_one(dst_t * __restrict__ y, స్థిరరాశి uint8_t * __restrict__ qs,
                                            స్థిరరాశి uint8_t * __restrict__ scales, స్థిరరాశి దశాంశం dall,
                                            స్థిరరాశి దశాంశం dmin, స్థిరరాశి పూర్ణాంకం tid) {
    స్థిరరాశి పూర్ణాంకం il = tid >> 3;        // 64-element chunk, 0..3
    స్థిరరాశి పూర్ణాంకం ir = (tid & 7) * 4;   // 4 qs bytes within the chunk

    uint8_t sc, m;
    get_scale_min_k4(2 * il + 0, scales, sc, m);
    స్థిరరాశి దశాంశం d1 = dall * sc;
    స్థిరరాశి దశాంశం m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, scales, sc, m);
    స్థిరరాశి దశాంశం d2 = dall * sc;
    స్థిరరాశి దశాంశం m2 = dmin * m;

    స్థిరరాశి sycl::vec<uint8_t, 4> q = vec_aligned_ఎక్కించు<uint8_t, 4>(qs + 32 * il + ir);

    sycl::vec<dst_t, 4> lo, hi;
#సూచన unroll
    కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
        lo[l] = d1 * (q[l] & 0xF) - m1;
        hi[l] = d2 * (q[l] >> 4)  - m2;
    }
    dst_t * yb = y + 64 * il + ir;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb +  0) = lo;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb + 32) = hi;
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q4_K_wide(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb,
                                       స్థిరరాశి sycl::nd_అంశం<1> & it) {
    static_assert(QK_K == 256, "wide K-quant dequant assumes QK_K == 256");
    స్థిరరాశి int64_t g   = it.get_global_id(0);
    స్థిరరాశి int64_t b   = g / SYCL_DEQK_WI_చొప్పున_ఖండం;
    స్థిరరాశి పూర్ణాంకం     tid = (పూర్ణాంకం) (g % SYCL_DEQK_WI_చొప్పున_ఖండం);
    ఒకవేళ (b >= nb) {
        తిరిగి;
    }
    స్థిరరాశి ఖండం_q4_K * x = (స్థిరరాశి ఖండం_q4_K *) vx + b;
    dequantize_q4_K_wide_one(yy + b * QK_K, x->qs, x->scales, (దశాంశం) x->dm[0], (దశాంశం) x->dm[1], tid);
}


మూస <రకం_పేరు dst_t>
అంతఃవరుస శూన్య dequantize_q4_K_common(dst_t * __restrict__ y, స్థిరరాశి uint8_t * __restrict__ qs_ptr, స్థిరరాశి దశాంశం dall,
                                   స్థిరరాశి దశాంశం dmin, uint8_t * __restrict__ scales_local, పూర్ణాంకం il, పూర్ణాంకం ir) {
    స్థిరరాశి పూర్ణాంకం is = 2 * il;
    constexpr పూర్ణాంకం n  = 4;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    స్థిరరాశి దశాంశం d1 = dall * sc;
    స్థిరరాశి దశాంశం m1 = dmin * m;

    get_scale_min_k4(is + 1, scales_local, sc, m);
    స్థిరరాశి దశాంశం d2 = dall * sc;
    స్థిరరాశి దశాంశం m2 = dmin * m;

    sycl::vec<uint8_t, n> q_vec = vec_aligned_ఎక్కించు<uint8_t, n>(qs_ptr + 32 * il + n * ir);
    కొరకు (పూర్ణాంకం l = 0; l < n; ++l) {
        y[l + 0]  = d1 * (q_vec[l] & 0xF) - m1;
        y[l + 32] = d2 * (q_vec[l] >> 4) - m2;
    }
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q4_K_reorder(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, స్థిరరాశి int64_t i,
                                          uint8_t * scales_local, స్థిరరాశి sycl::nd_అంశం<1> & అంశం_ct1, int64_t nb) {
    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(0);  // thread index within block
    స్థిరరాశి int64_t il  = tid / 8;
    స్థిరరాశి int64_t ir  = tid % 8;

    dst_t * y = yy + 64 * il + 4 * ir;

    స్థిరరాశి uint8_t * పునాది          = static_cast<స్థిరరాశి uint8_t *>(vx);
    స్థిరరాశి size_t    qs_స్థానభ్రంశం     = i * (QK_K / 2);
    స్థిరరాశి size_t    scales_స్థానభ్రంశం = nb * (QK_K / 2) + i * K_SCALE_కొలత;
    స్థిరరాశి size_t    dm_స్థానభ్రంశం     = nb * (QK_K / 2) + nb * K_SCALE_కొలత + i * పరిమాణం(ggml_half2);

    స్థిరరాశి uint8_t *    qs_ptr     = పునాది + qs_స్థానభ్రంశం;
    స్థిరరాశి uint8_t *    scales_ptr = పునాది + scales_స్థానభ్రంశం;
    ggml_half2         dm_విలువలు  = *reinterpret_cast<స్థిరరాశి ggml_half2 *>(పునాది + dm_స్థానభ్రంశం);

    స్థిరరాశి దశాంశం dall = dm_విలువలు.x();
    స్థిరరాశి దశాంశం dmin = dm_విలువలు.y();

    ఒకవేళ (tid < 12) {
        scales_local[tid] = scales_ptr[tid];
    }

    అంశం_ct1.barrier(sycl::access::fence_space::local_space);
    dequantize_q4_K_common(y, qs_ptr, dall, dmin, scales_local, il, ir);
}

మూస <రకం_పేరు dst_t>
స్థిర అంతఃవరుస శూన్య dequantize_q5_K_wide_one(dst_t * __restrict__ y, స్థిరరాశి uint8_t * __restrict__ qs,
                                            స్థిరరాశి uint8_t * __restrict__ qh_ptr,
                                            స్థిరరాశి uint8_t * __restrict__ scales, స్థిరరాశి దశాంశం dall,
                                            స్థిరరాశి దశాంశం dmin, స్థిరరాశి పూర్ణాంకం tid) {
    స్థిరరాశి పూర్ణాంకం il = tid >> 3;        // 64-element chunk, 0..3
    స్థిరరాశి పూర్ణాంకం ir = (tid & 7) * 4;   // 4 qs bytes within the chunk; qh holds the 5th bit

    uint8_t sc, m;
    get_scale_min_k4(2 * il + 0, scales, sc, m);
    స్థిరరాశి దశాంశం d1 = dall * sc;
    స్థిరరాశి దశాంశం m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, scales, sc, m);
    స్థిరరాశి దశాంశం d2 = dall * sc;
    స్థిరరాశి దశాంశం m2 = dmin * m;

    స్థిరరాశి sycl::vec<uint8_t, 4> q = vec_aligned_ఎక్కించు<uint8_t, 4>(qs + 32 * il + ir);
    స్థిరరాశి sycl::vec<uint8_t, 4> h = vec_aligned_ఎక్కించు<uint8_t, 4>(qh_ptr + ir);

    స్థిరరాశి uint8_t hm1 = 1 << (2 * il);
    స్థిరరాశి uint8_t hm2 = hm1 << 1;

    sycl::vec<dst_t, 4> lo, hi;
#సూచన unroll
    కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
        lo[l] = d1 * ((q[l] & 0xF) + ((h[l] & hm1) ? 16 : 0)) - m1;
        hi[l] = d2 * ((q[l] >>  4) + ((h[l] & hm2) ? 16 : 0)) - m2;
    }
    dst_t * yb = y + 64 * il + ir;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb +  0) = lo;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb + 32) = hi;
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q5_K_wide(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb,
                                       స్థిరరాశి sycl::nd_అంశం<1> & it) {
    static_assert(QK_K == 256, "wide K-quant dequant assumes QK_K == 256");
    స్థిరరాశి int64_t g   = it.get_global_id(0);
    స్థిరరాశి int64_t b   = g / SYCL_DEQK_WI_చొప్పున_ఖండం;
    స్థిరరాశి పూర్ణాంకం     tid = (పూర్ణాంకం) (g % SYCL_DEQK_WI_చొప్పున_ఖండం);
    ఒకవేళ (b >= nb) {
        తిరిగి;
    }
    స్థిరరాశి ఖండం_q5_K * x = (స్థిరరాశి ఖండం_q5_K *) vx + b;
    dequantize_q5_K_wide_one(yy + b * QK_K, x->qs, x->qh, x->scales, (దశాంశం) x->dm[0], (దశాంశం) x->dm[1], tid);
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q5_K_reorder(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, స్థిరరాశి int64_t ib,
                                          uint8_t * scales_local, స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1, int64_t n_ఖండాలు) {

#ఒకవేళ QK_K == 256
    // assume 64 threads
    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి int64_t il  = tid / 16;   // 0...3
    స్థిరరాశి int64_t ir  = tid % 16;   // 0...15
    స్థిరరాశి int64_t is  = 2 * il;

    dst_t * y = yy + 64 * il + 2 * ir;

    స్థిరరాశి uint8_t * పునాది = static_cast<స్థిరరాశి uint8_t *>(vx);

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales (K_SCALE_SIZE per block)] [dm (half2 per block)]
    స్థిరరాశి size_t qs_స్థానభ్రంశం     = ib * (QK_K / 2);
    స్థిరరాశి size_t qh_స్థానభ్రంశం     = n_ఖండాలు * (QK_K / 2) + ib * (QK_K / 8);
    స్థిరరాశి size_t scales_స్థానభ్రంశం = n_ఖండాలు * (QK_K / 2) + n_ఖండాలు * (QK_K / 8) + ib * K_SCALE_కొలత;
    స్థిరరాశి size_t dm_స్థానభ్రంశం     = n_ఖండాలు * (QK_K / 2) + n_ఖండాలు * (QK_K / 8) + n_ఖండాలు * K_SCALE_కొలత + ib * పరిమాణం(ggml_half2);

    స్థిరరాశి uint8_t *  qs_ptr     = పునాది + qs_స్థానభ్రంశం;
    స్థిరరాశి uint8_t *  qh_ptr     = పునాది + qh_స్థానభ్రంశం;
    స్థిరరాశి uint8_t *  scales_ptr = పునాది + scales_స్థానభ్రంశం;
    స్థిరరాశి ggml_half2 dm_విలువలు  = *reinterpret_cast<స్థిరరాశి ggml_half2 *>(పునాది + dm_స్థానభ్రంశం);

    స్థిరరాశి దశాంశం dall = dm_విలువలు.x();
    స్థిరరాశి దశాంశం dmin = dm_విలువలు.y();

    స్థిరరాశి uint8_t * ql = qs_ptr + 32 * il + 2 * ir;
    స్థిరరాశి uint8_t * qh = qh_ptr + 2 * ir;

    ఒకవేళ (tid < K_SCALE_కొలత) {
        scales_local[tid] = scales_ptr[tid];
    }

    అంశం_ct1.barrier(sycl::access::fence_space::local_space);

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    స్థిరరాశి దశాంశం d1 = dall * sc; స్థిరరాశి దశాంశం m1 = dmin * m;
    get_scale_min_k4(is + 1, scales_local, sc, m);
    స్థిరరాశి దశాంశం d2 = dall * sc; స్థిరరాశి దశాంశం m2 = dmin * m;

    uint8_t hm  = 1 << (2 * il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
#లేకపోతే
    GGML_వాడని(ib); GGML_వాడని(tid); GGML_వాడని(yy); GGML_వాడని(scales_local); GGML_వాడని(n_ఖండాలు);
    GGML_ABORT("Q5_K reorder dequantize not supported for QK_K != 256");
#అయితే_ముగింపు
}

మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q6_K(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy,
                                  స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి ఖండం_q6_K * x = (స్థిరరాశి ఖండం_q6_K *) vx;

    స్థిరరాశి int64_t i = అంశం_ct1.get_సమూహం(2);
#ఒకవేళ QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి int64_t ip  = tid/32;   // ip is 0 or 1
    స్థిరరాశి int64_t il  = tid - 32*ip; // 0...32
    స్థిరరాశి int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    స్థిరరాశి దశాంశం d = x[i].d;

    స్థిరరాశి uint8_t * ql = x[i].ql + 64*ip + il;
    స్థిరరాశి uint8_t   qh = x[i].qh[32*ip + il];
    స్థిరరాశి int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#లేకపోతే

    // assume 32 threads
    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి int64_t ip  = tid/16;         // 0 or 1
    స్థిరరాశి int64_t il  = tid - 16*ip;    // 0...15

    dst_t * y = yy + i*QK_K + 16*ip + il;

    స్థిరరాశి దశాంశం d = x[i].d;

    స్థిరరాశి uint8_t   ql = x[i].ql[16*ip + il];
    స్థిరరాశి uint8_t   qh = x[i].qh[il] >> (2*ip);
    స్థిరరాశి int8_t  * sc = x[i].scales;

    y[ 0] = d * sc[ip+0] * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[ip+2] * ((int8_t)((ql  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
#అయితే_ముగింపు
}

మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_q6_K_reorder(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, స్థిరరాశి int64_t ib,
                                          స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1, int64_t n_ఖండాలు) {

    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి int64_t ip  = tid / 32;       // ip is 0 or 1
    స్థిరరాశి int64_t il  = tid - 32 * ip;  // 0...32
    స్థిరరాశి int64_t is  = 8 * ip + il / 16;

    స్థిరరాశి uint8_t *   పునాది_ptr           = static_cast<స్థిరరాశి uint8_t *>(vx);
    స్థిరరాశి స్వయంచాలక        ql_స్థానభ్రంశం          = ib * (QK_K / 2);
    స్థిరరాశి స్వయంచాలక        qh_స్థానభ్రంశం          = (QK_K / 2) * n_ఖండాలు + (QK_K / 4) * ib;
    స్థిరరాశి స్వయంచాలక        పునాది_scales_స్థానభ్రంశం = (QK_K / 2) * n_ఖండాలు + (QK_K / 4) * n_ఖండాలు + (QK_K / 16) * ib;
    స్థిరరాశి స్వయంచాలక        పునాది_d_స్థానభ్రంశం      = ((QK_K / 2) + (QK_K / 4) + (QK_K / 16)) * n_ఖండాలు;
    స్థిరరాశి uint8_t *   ql_ptr             = పునాది_ptr + ql_స్థానభ్రంశం;
    స్థిరరాశి uint8_t *   qh_ptr             = పునాది_ptr + qh_స్థానభ్రంశం;
    స్థిరరాశి uint8_t *   scales_ptr         = పునాది_ptr + పునాది_scales_స్థానభ్రంశం;
    స్థిరరాశి ggml_half * d                  = (స్థిరరాశి ggml_half *) (పునాది_ptr + పునాది_d_స్థానభ్రంశం) + ib;

    dst_t * y = yy + 128 * ip + il;

    స్థిరరాశి uint8_t * ql = ql_ptr + 64 * ip + il;
    స్థిరరాశి uint8_t   qh = *(qh_ptr + 32 * ip + il);
    స్థిరరాశి int8_t *  sc = reinterpret_cast<స్థిరరాశి int8_t *>(scales_ptr + is);

    y[0]  = *d * sc[0] * ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = *d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = *d * sc[4] * ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = *d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_iq2_xxs(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy,
                                     స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1,
                                     స్థిరరాశి uint64_t *iq2xxs_grid_ptr,
                                     స్థిరరాశి uint8_t *ksigns_iq2xs_ptr,
                                     స్థిరరాశి uint8_t *kmask_iq2xs_ptr) {

    స్థిరరాశి int64_t i = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి ఖండం_iq2_xxs * x = (స్థిరరాశి ఖండం_iq2_xxs  *) vx;

    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
#ఒకవేళ QK_K == 256
    స్థిరరాశి int64_t il = tid/8; // 0...3
    స్థిరరాశి int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    స్థిరరాశి uint16_t * q2 = x[i].qs + 4*ib;
    స్థిరరాశి uint8_t  * aux8 = (స్థిరరాశి uint8_t *)q2;
    స్థిరరాశి uint8_t  * grid = (స్థిరరాశి uint8_t *)(iq2xxs_grid_ptr + aux8[il]);
    స్థిరరాశి uint32_t aux32 = q2[2] | (q2[3] << 16);
    స్థిరరాశి దశాంశం d = (దశాంశం)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    స్థిరరాశి uint8_t signs = ksigns_iq2xs_ptr[(aux32 >> 7*il) & 127];
    కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs_ptr[j] ? -1.f : 1.f);
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
#అయితే_ముగింపు

}

మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_iq2_xs(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy,
                                    స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1,
                                    స్థిరరాశి uint64_t *iq2xs_grid,
                                    స్థిరరాశి uint8_t *ksigns_iq2xs,
                                    స్థిరరాశి uint8_t *kmask_iq2xs) {

    స్థిరరాశి int64_t i = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి ఖండం_iq2_xs * x = (స్థిరరాశి ఖండం_iq2_xs *) vx;

    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
#ఒకవేళ QK_K == 256
    స్థిరరాశి int64_t il = tid/8; // 0...3
    స్థిరరాశి int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    స్థిరరాశి uint16_t * q2 = x[i].qs + 4*ib;
    స్థిరరాశి uint8_t  * grid = (స్థిరరాశి uint8_t *)(iq2xs_grid + (q2[il] & 511));
    స్థిరరాశి దశాంశం d = (దశాంశం)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    స్థిరరాశి uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
#అయితే_ముగింపు

}

మూస <రకం_పేరు dst_t>
__dpct_inline__ స్థిర శూన్య
dequantize_ఖండం_iq2_s(స్థిరరాశి శూన్య *__restrict__ vx, dst_t *__restrict__ yy,
                       స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి int64_t i = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి ఖండం_iq2_s * x = (స్థిరరాశి ఖండం_iq2_s *) vx;

    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
#ఒకవేళ QK_K == 256
    స్థిరరాశి int64_t il = tid/8; // 0...3
    స్థిరరాశి int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    స్థిరరాశి uint8_t * grid = (స్థిరరాశి uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    స్థిరరాశి దశాంశం d = (దశాంశం)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    స్థిరరాశి uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 8; ++j)
        y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);

#అయితే_ముగింపు

}

మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_iq3_xxs(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy,
                                     స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1,
                                     స్థిరరాశి uint32_t *iq3xxs_grid,
                                     స్థిరరాశి uint8_t *ksigns_iq2xs,
                                     స్థిరరాశి uint8_t *kmask_iq2xs) {

    స్థిరరాశి int64_t i = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి ఖండం_iq3_xxs * x = (స్థిరరాశి ఖండం_iq3_xxs  *) vx;

    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
#ఒకవేళ QK_K == 256
    స్థిరరాశి int64_t il = tid/8; // 0...3
    స్థిరరాశి int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    స్థిరరాశి uint8_t  * q3 = x[i].qs + 8*ib;
    స్థిరరాశి uint16_t * gas = (స్థిరరాశి uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    స్థిరరాశి uint8_t  * grid1 = (స్థిరరాశి uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    స్థిరరాశి uint8_t  * grid2 = (స్థిరరాశి uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    స్థిరరాశి uint32_t aux32 = gas[0] | (gas[1] << 16);
    స్థిరరాశి దశాంశం d = (దశాంశం)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    స్థిరరాశి uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    కొరకు (పూర్ణాంకం j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
#అయితే_ముగింపు

}

మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_iq3_xxs_reorder(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, స్థిరరాశి int64_t i,
                                             స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1, స్థిరరాశి int64_t nb,
                                             స్థిరరాశి uint32_t * iq3xxs_grid,
                                             స్థిరరాశి uint8_t * ksigns_iq2xs,
                                             స్థిరరాశి uint8_t * kmask_iq2xs) {
    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
#ఒకవేళ QK_K == 256
    స్థిరరాశి int64_t il = tid / 8;
    స్థిరరాశి int64_t ib = tid % 8;
    స్థిరరాశి uint8_t * పునాది = static_cast<స్థిరరాశి uint8_t *>(vx);
    స్థిరరాశి uint8_t * qs = పునాది + i * (3 * QK_K / 8);
    స్థిరరాశి ggml_half * d = reinterpret_cast<స్థిరరాశి ggml_half *>(పునాది + nb * (3 * QK_K / 8));
    స్థిరరాశి uint8_t * q3 = qs + 8 * ib;
    స్థిరరాశి uint16_t * gas = reinterpret_cast<స్థిరరాశి uint16_t *>(qs + QK_K / 4) + 2 * ib;
    స్థిరరాశి uint8_t * grid1 = reinterpret_cast<స్థిరరాశి uint8_t *>(iq3xxs_grid + q3[2*il+0]);
    స్థిరరాశి uint8_t * grid2 = reinterpret_cast<స్థిరరాశి uint8_t *>(iq3xxs_grid + q3[2*il+1]);
    స్థిరరాశి uint32_t aux32 = gas[0] | (gas[1] << 16);
    స్థిరరాశి దశాంశం dl = (దశాంశం) d[i] * (0.5f + (aux32 >> 28)) * 0.5f;
    స్థిరరాశి uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
    dst_t * y = yy + 32 * ib + 8 * il;
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 4; ++j) {
        y[j+0] = dl * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = dl * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#లేకపోతే
    GGML_వాడని(vx);
    GGML_వాడని(yy);
    GGML_వాడని(i);
    GGML_వాడని(అంశం_ct1);
    GGML_వాడని(nb);
    GGML_వాడని(iq3xxs_grid);
    GGML_వాడని(ksigns_iq2xs);
    GGML_వాడని(kmask_iq2xs);
    GGML_ABORT("IQ3_XXS reorder dequantization requires QK_K == 256");
#అయితే_ముగింపు
}

మూస <రకం_పేరు dst_t>
__dpct_inline__ స్థిర శూన్య
dequantize_ఖండం_iq3_s(స్థిరరాశి శూన్య *__restrict__ vx, dst_t *__restrict__ yy,
                       స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1,
                       స్థిరరాశి uint8_t *kmask_iq2xs, స్థిరరాశి uint32_t *iq3s_grid) {

    స్థిరరాశి int64_t i = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి ఖండం_iq3_s * x = (స్థిరరాశి ఖండం_iq3_s *) vx;

    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
#ఒకవేళ QK_K == 256
    స్థిరరాశి int64_t il = tid/8; // 0...3
    స్థిరరాశి int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    స్థిరరాశి uint8_t * qs = x[i].qs + 8*ib;
    స్థిరరాశి uint8_t * grid1 = (స్థిరరాశి uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    స్థిరరాశి uint8_t * grid2 = (స్థిరరాశి uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    స్థిరరాశి దశాంశం d = (దశాంశం)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    స్థిరరాశి uint8_t signs = x[i].signs[4*ib + il];
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
#అయితే_ముగింపు

}

మూస <రకం_పేరు dst_t>
__dpct_inline__ స్థిర శూన్య
dequantize_ఖండం_iq3_s_reorder(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, స్థిరరాశి int64_t i,
                               స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1, స్థిరరాశి int64_t nb,
                               స్థిరరాశి uint8_t * kmask_iq2xs, స్థిరరాశి uint32_t * iq3s_grid) {
    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
#ఒకవేళ QK_K == 256
    స్థిరరాశి int64_t il = tid / 8;
    స్థిరరాశి int64_t ib = tid % 8;
    స్థిరరాశి uint8_t * పునాది = static_cast<స్థిరరాశి uint8_t *>(vx);
    స్థిరరాశి uint8_t * qs = పునాది + i * (QK_K / 4);
    స్థిరరాశి uint8_t * qh = పునాది + nb * (QK_K / 4) + i * (QK_K / 32);
    స్థిరరాశి uint8_t * signs_పునాది = పునాది + nb * (QK_K / 4 + QK_K / 32) + i * (QK_K / 8);
    స్థిరరాశి uint8_t * metadata = పునాది + nb * (QK_K / 4 + QK_K / 32 + QK_K / 8) +
                               i * (పరిమాణం(ggml_half) + IQ3S_N_SCALE);
    స్థిరరాశి ggml_half d = *reinterpret_cast<స్థిరరాశి ggml_half *>(metadata);
    స్థిరరాశి uint8_t * scales = metadata + పరిమాణం(ggml_half);
    స్థిరరాశి uint8_t * qs_ib = qs + 8 * ib;
    స్థిరరాశి uint8_t * grid1 = reinterpret_cast<స్థిరరాశి uint8_t *>(
        iq3s_grid + (qs_ib[2*il+0] | ((qh[ib] << (8 - 2*il)) & 256)));
    స్థిరరాశి uint8_t * grid2 = reinterpret_cast<స్థిరరాశి uint8_t *>(
        iq3s_grid + (qs_ib[2*il+1] | ((qh[ib] << (7 - 2*il)) & 256)));
    స్థిరరాశి దశాంశం dl = (దశాంశం) d * (1 + 2 * ((scales[ib/2] >> (4 * (ib%2))) & 0xf));
    స్థిరరాశి uint8_t signs = signs_పునాది[4 * ib + il];
    dst_t * y = yy + 32 * ib + 8 * il;
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 4; ++j) {
        y[j+0] = dl * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = dl * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#లేకపోతే
    GGML_వాడని(vx);
    GGML_వాడని(yy);
    GGML_వాడని(i);
    GGML_వాడని(అంశం_ct1);
    GGML_వాడని(nb);
    GGML_వాడని(kmask_iq2xs);
    GGML_వాడని(iq3s_grid);
    GGML_ABORT("IQ3_S reorder dequantization requires QK_K == 256");
#అయితే_ముగింపు
}

మూస <రకం_పేరు dst_t>
__dpct_inline__ స్థిర శూన్య
dequantize_ఖండం_iq1_s(స్థిరరాశి శూన్య *__restrict__ vx, dst_t *__restrict__ yy,
                       స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1,
                       స్థిరరాశి uint32_t *iq1s_grid_gpu) {

    స్థిరరాశి int64_t i = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి ఖండం_iq1_s * x = (స్థిరరాశి ఖండం_iq1_s  *) vx;

    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
#ఒకవేళ QK_K == 256
    స్థిరరాశి int64_t il = tid/8; // 0...3
    స్థిరరాశి int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    స్థిరరాశి దశాంశం delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    స్థిరరాశి దశాంశం d = (దశాంశం)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; స్థిరరాశి int8_t * q = (స్థిరరాశి int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
#అయితే_ముగింపు

}

మూస <రకం_పేరు dst_t>
__dpct_inline__ స్థిర శూన్య
dequantize_ఖండం_iq1_m(స్థిరరాశి శూన్య *__restrict__ vx, dst_t *__restrict__ yy,
                       స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1,
                       స్థిరరాశి uint32_t *iq1s_grid_gpu) {

    స్థిరరాశి int64_t i = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి ఖండం_iq1_m * x = (స్థిరరాశి ఖండం_iq1_m  *) vx;

    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
#ఒకవేళ QK_K == 256
    స్థిరరాశి int64_t il = tid/8; // 0...3
    స్థిరరాశి int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    స్థిరరాశి uint16_t * sc = (స్థిరరాశి uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    స్థిరరాశి పూర్ణాంకం ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    స్థిరరాశి దశాంశం d = (దశాంశం)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    స్థిరరాశి దశాంశం delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; స్థిరరాశి int8_t * q = (స్థిరరాశి int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
#అయితే_ముగింపు

}

మూస <రకం_పేరు dst_t>
__dpct_inline__ స్థిర శూన్య
dequantize_ఖండం_iq4_nl(స్థిరరాశి శూన్య *__restrict__ vx, dst_t *__restrict__ yy,
                        స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {

    స్థిరరాశి int64_t i = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి ఖండం_iq4_nl * x = (స్థిరరాశి ఖండం_iq4_nl *) vx + i*(QK_K/QK4_NL);

    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి int64_t il = tid/8; // 0...3
    స్థిరరాశి int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    స్థిరరాశి uint8_t  * q4 = x[ib].qs + 4*il;
    స్థిరరాశి దశాంశం d = (దశాంశం)x[ib].d;
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }

}


మూస <రకం_పేరు dst_t>
__dpct_inline__ స్థిర శూన్య
dequantize_ఖండం_iq4_xs(స్థిరరాశి శూన్య *__restrict__ vx, dst_t *__restrict__ yy,
                        స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    స్థిరరాశి int64_t i = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి ఖండం_iq4_xs * x = (స్థిరరాశి ఖండం_iq4_xs *)vx;

    స్థిరరాశి int64_t tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి int64_t il = tid/8; // 0...3
    స్థిరరాశి int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    స్థిరరాశి uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    స్థిరరాశి దశాంశం d = (దశాంశం)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

మూస<రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_mxfp4(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy,
                                   స్థిరరాశి sycl::nd_అంశం<3> &అంశం_ct1) {
    // auto                item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    స్థిరరాశి int64_t       i        = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి ఖండం_mxfp4 * x = (స్థిరరాశి ఖండం_mxfp4 *) vx + i*(QK_K/QK_MXFP4);

    స్థిరరాశి int64_t    tid = అంశం_ct1.get_local_id(2);
    స్థిరరాశి int64_t il = tid/8; // 0...3
    స్థిరరాశి int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    స్థిరరాశి uint8_t  * q4 = x[ib].qs + 4*il;
    స్థిరరాశి దశాంశం d = ggml_sycl_e8m0_గమ్యం_fp32(x[ib].e);
    కొరకు (పూర్ణాంకం j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_mxfp4[q4[j] & 0xf]*0.5f;
        y[j+16] = d * kvalues_mxfp4[q4[j] >>  4]*0.5f;
    }
}

// Reordered MXFP4 ([qs...][e...], see ggml_sycl_reordered::block_q_t<MXFP4>): one work-item per block.
మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_mxfp4_reorder(స్థిరరాశి శూన్య * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                           స్థిరరాశి sycl::nd_అంశం<3> & అంశం_ct1) {
    స్థిరరాశి int64_t ib = (int64_t) అంశం_ct1.get_సమూహం(2) * WARP_కొలత + అంశం_ct1.get_local_id(2);
    ఒకవేళ (ib >= k / QK_MXFP4) {
        తిరిగి;
    }

    స్థిరరాశి uint8_t * qs = (స్థిరరాశి uint8_t *) vx + ib * (QK_MXFP4 / 2);
    స్థిరరాశి దశాంశం     d  = ggml_sycl_e8m0_గమ్యం_fp32(((స్థిరరాశి uint8_t *) vx)[k / 2 + ib]) * 0.5f;
    dst_t *         y  = yy + ib * QK_MXFP4;

#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < QK_MXFP4 / 2; ++j) {
        y[j]                = d * kvalues_mxfp4[qs[j] & 0xf];
        y[j + QK_MXFP4 / 2] = d * kvalues_mxfp4[qs[j] >> 4];
    }
}


మూస <రకం_పేరు dst_t>
స్థిర శూన్య dequantize_ఖండం_nvfp4(
        స్థిరరాశి శూన్య * __restrict__ vx,
        dst_t * __restrict__ yy,
        స్థిరరాశి int64_t ne) {
    స్వయంచాలక          అంశం_ct1 = sycl::ext::oneapi::this_work_అంశం::get_nd_అంశం<3>();
    స్థిరరాశి int64_t i        = అంశం_ct1.get_సమూహం(2);
    స్థిరరాశి పూర్ణాంకం     tid      = అంశం_ct1.get_local_id(2);

    స్థిరరాశి int64_t పునాది = i * QK_NVFP4;
    ఒకవేళ (పునాది >= ne) {
        తిరిగి;
    }

    స్థిరరాశి ఖండం_nvfp4 * x = (స్థిరరాశి ఖండం_nvfp4 *) vx;
    స్థిరరాశి ఖండం_nvfp4 & xb = x[i];

    స్థిరరాశి పూర్ణాంకం sub = tid / (QK_NVFP4_SUB / 2);
    స్థిరరాశి పూర్ణాంకం j = tid % (QK_NVFP4_SUB / 2);

    స్థిరరాశి దశాంశం d = ggml_sycl_ue4m3_గమ్యం_fp32(xb.d[sub]);
    స్థిరరాశి uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    స్థిరరాశి int64_t y0 = పునాది + sub * QK_NVFP4_SUB + j;
    స్థిరరాశి int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_sycl_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_sycl_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}


#అయితే_ముగింపు // GGML_SYCL_DEQUANTIZE_HPP
