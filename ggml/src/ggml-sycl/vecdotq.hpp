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

#నిర్వచితం_కాకపోతే GGML_SYCL_VECDOTQ_HPP
#నిర్వచించు GGML_SYCL_VECDOTQ_HPP

#అన్తర్భావ "dpct/helper.hpp"
#అన్తర్భావ "ggml.h"
#అన్తర్భావ "type.hpp"
#అన్తర్భావ "quants.hpp"

రకనిర్వచనం దశాంశం (*vec_dot_q_sycl_t)(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి ఖండం_q8_1 * __restrict__ bq8_1,
                                  స్థిరరాశి పూర్ణాంకం & iqs);

స్థిర __dpct_inline__ పూర్ణాంకం get_int_b1(స్థిరరాశి శూన్య * x, స్థిరరాశి పూర్ణాంకం & i32) {
    స్థిరరాశి uint8_t * x8 = (స్థిరరాశి uint8_t *) x;

    పూర్ణాంకం x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    తిరిగి x32;
}

స్థిర __dpct_inline__ పూర్ణాంకం get_int_b2(స్థిరరాశి శూన్య * x, స్థిరరాశి పూర్ణాంకం & i32) {
    స్థిరరాశి uint16_t * x16 = (స్థిరరాశి uint16_t *) x; // assume at least 2 byte alignment

    పూర్ణాంకం x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    తిరిగి x32;
}

స్థిర __dpct_inline__ పూర్ణాంకం get_int_b4(స్థిరరాశి శూన్య * x, స్థిరరాశి పూర్ణాంకం & i32) {
    తిరిగి ((స్థిరరాశి పూర్ణాంకం *) x)[i32]; // assume at least 4 byte alignment
}

స్థిర __dpct_inline__ పూర్ణాంకం get_int_from_int8(స్థిరరాశి int8_t* x8, స్థిరరాశి పూర్ణాంకం& i32) {
  స్థిరరాశి uint16_t* x16 =
      (స్థిరరాశి uint16_t*)(x8 + పరిమాణం(పూర్ణాంకం) * i32); // assume at least 2 byte
                                                 // alignment

  పూర్ణాంకం x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  తిరిగి x32;
}

స్థిర __dpct_inline__ పూర్ణాంకం get_int_from_uint8(
    స్థిరరాశి uint8_t* x8,
    స్థిరరాశి పూర్ణాంకం& i32) {
  స్థిరరాశి uint16_t* x16 =
      (స్థిరరాశి uint16_t*)(x8 + పరిమాణం(పూర్ణాంకం) * i32); // assume at least 2 byte
                                                 // alignment

  పూర్ణాంకం x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  తిరిగి x32;
}

స్థిర __dpct_inline__ పూర్ణాంకం get_int_from_int8_aligned(
    స్థిరరాశి int8_t* x8,
    స్థిరరాశి పూర్ణాంకం& i32) {
  తిరిగి *(
      (స్థిరరాశి పూర్ణాంకం*)(x8 + పరిమాణం(పూర్ణాంకం) * i32)); // assume at least 4 byte alignment
}

స్థిర __dpct_inline__ పూర్ణాంకం get_int_from_uint8_aligned(
    స్థిరరాశి uint8_t* x8,
    స్థిరరాశి పూర్ణాంకం& i32) {
  తిరిగి *(
      (స్థిరరాశి పూర్ణాంకం*)(x8 + పరిమాణం(పూర్ణాంకం) * i32)); // assume at least 4 byte alignment
}

స్థిర __dpct_inline__ పూర్ణాంకం byte_sub_4(స్థిరరాశి పూర్ణాంకం a, స్థిరరాశి పూర్ణాంకం b) {
  స్థిరరాశి uint32_t ua = static_cast<uint32_t>(a);
  స్థిరరాశి uint32_t ub = static_cast<uint32_t>(b);
  తిరిగి static_cast<పూర్ణాంకం>(((ua | 0x80808080u) - ub) ^ 0x80808080u);
}

స్థిర __dpct_inline__ దశాంశం vec_dot_q6_K_q8_1_impl_mmvq_scalar(
    స్థిరరాశి పూర్ణాంకం vl, స్థిరరాశి పూర్ణాంకం vh, స్థిరరాశి పూర్ణాంకం u0, స్థిరరాశి పూర్ణాంకం u1, స్థిరరాశి int8_t sc0,
    స్థిరరాశి int8_t sc1, స్థిరరాశి దశాంశం d, స్థిరరాశి దశాంశం d80, స్థిరరాశి దశాంశం d81) {
    static_assert(QR6_K == 2, "q6_K MMVQ scalar fast path assumes QR6_K == 2");

    స్థిరరాశి పూర్ణాంకం vil0 = (vl >> 0) & 0x0F0F0F0F;
    స్థిరరాశి పూర్ణాంకం vih0 = ((vh >> 0) << 4) & 0x30303030;
    స్థిరరాశి పూర్ణాంకం vi0 = byte_sub_4(vil0 | vih0, 0x20202020);

    స్థిరరాశి పూర్ణాంకం vil1 = (vl >> 4) & 0x0F0F0F0F;
    స్థిరరాశి పూర్ణాంకం vih1 = ((vh >> 4) << 4) & 0x30303030;
    స్థిరరాశి పూర్ణాంకం vi1 = byte_sub_4(vil1 | vih1, 0x20202020);

    స్థిరరాశి దశాంశం sumf =
        d80 * (dpct::dp4a(vi0, u0, 0) * sc0) +
        d81 * (dpct::dp4a(vi1, u1, 0) * sc1);

    తిరిగి d * sumf;
}

స్థిర __dpct_inline__ శూన్య get_int_from_పట్టిక_16(స్థిరరాశి uint32_t &q4,
                                                  స్థిరరాశి uint8_t *విలువలు,
                                                  పూర్ణాంకం &val1, పూర్ణాంకం &val2) {

    uint32_t aux32; స్థిరరాశి uint8_t * q8 = (స్థిరరాశి uint8_t *)&aux32;
    aux32 = q4 & 0x0f0f0f0f;
    uint16_t v1 = విలువలు[q8[0]] | (విలువలు[q8[1]] << 8);
    uint16_t v2 = విలువలు[q8[2]] | (విలువలు[q8[3]] << 8);
    val1 = v1 | (v2 << 16);
    aux32 = (q4 >> 4) & 0x0f0f0f0f;
    v1 = విలువలు[q8[0]] | (విలువలు[q8[1]] << 8);
    v2 = విలువలు[q8[2]] | (విలువలు[q8[3]] << 8);
    val2 = v1 | (v2 << 16);
}

స్థిర __dpct_inline__ sycl::int2 get_int_from_పట్టిక_16(
    స్థిరరాశి పూర్ణాంకం& q4, స్థిరరాశి int8_t* పట్టిక) {
  స్థిరరాశి uint32_t* table32 = (స్థిరరాశి uint32_t*)పట్టిక;
  uint32_t tmp[2];
  స్థిరరాశి uint32_t దిగువ_ఎగువ_selection_indices =
      (0x32103210 | ((q4 & 0x88888888) >> 1));
#సూచన unroll
  కొరకు (uint32_t i = 0; i < 2; ++i) {
    స్థిరరాశి uint32_t జరుపు = 16 * i;

    స్థిరరాశి uint32_t దిగువ =
        dpct::byte_level_క్రమమార్పు(table32[0], table32[1], q4 >> జరుపు);
    స్థిరరాశి uint32_t ఎగువ =
        dpct::byte_level_క్రమమార్పు(table32[2], table32[3], q4 >> జరుపు);
    tmp[i] = dpct::byte_level_క్రమమార్పు(
        దిగువ, ఎగువ, దిగువ_ఎగువ_selection_indices >> జరుపు);
  }
  తిరిగి sycl::int2(
      dpct::byte_level_క్రమమార్పు(tmp[0], tmp[1], 0x6420),
      dpct::byte_level_క్రమమార్పు(tmp[0], tmp[1], 0x7531));
}

// Four E2M1 codes (one per byte, bits 0..3) to their kvalues_mxfp4 int8 values. SWAR arithmetic
// replaces get_int_from_table_16 for MXFP4: dpct::byte_level_permute is emulated with 64-bit shifts,
// eight per int, which made the MXFP4 GEMV compute-bound on Intel GPUs.
// Magnitudes 0,1,2,3,4,6,8,12 = m + [m>=5] + [m>=6] + 3*[m>=7]; each byte stays below 256, so the
// byte-wise adds never carry. -0 (code 8) is left as 0 so the two's-complement +1 cannot carry either.
స్థిర __dpct_inline__ పూర్ణాంకం mxfp4_codes_గమ్యం_int8(స్థిరరాశి uint32_t x) {
    స్థిరరాశి uint32_t m   = x & 0x07070707u;
    స్థిరరాశి uint32_t ge5 = ((m + 0x03030303u) >> 3) & 0x01010101u;
    స్థిరరాశి uint32_t ge6 = ((m + 0x02020202u) >> 3) & 0x01010101u;
    స్థిరరాశి uint32_t ge7 = ((m + 0x01010101u) >> 3) & 0x01010101u;
    స్థిరరాశి uint32_t mag = m + ge5 + ge6 + 3u * ge7;
    స్థిరరాశి uint32_t nz  = ((mag + 0x7f7f7f7fu) >> 7) & 0x01010101u;
    స్థిరరాశి uint32_t neg = (x >> 3) & nz & 0x01010101u;
    తిరిగి (పూర్ణాంకం) ((mag ^ (neg * 0xffu)) + neg);
}

// Same result as get_int_from_table_16(q4, kvalues_mxfp4): x = low nibbles, y = high nibbles.
స్థిర __dpct_inline__ sycl::int2 get_int_from_mxfp4(స్థిరరాశి పూర్ణాంకం q4) {
    తిరిగి sycl::int2(mxfp4_codes_గమ్యం_int8((uint32_t) q4 & 0x0f0f0f0fu),
                      mxfp4_codes_గమ్యం_int8(((uint32_t) q4 >> 4) & 0x0f0f0f0fu));
}

#నిర్వచించు VDR_Q2_K_Q8_1_MMVQ 1

// contiguous v/x values
స్థిర __dpct_inline__ దశాంశం vec_dot_q2_K_q8_1_impl_mmvq(
    స్థిరరాశి పూర్ణాంకం &v, స్థిరరాశి పూర్ణాంకం *__restrict__ u, స్థిరరాశి uint8_t *__restrict__ scales,
    స్థిరరాశి sycl::half2 &dm2, స్థిరరాశి దశాంశం *__restrict__ d8) {

    దశాంశం sumf_d = 0.0f;
    దశాంశం sumf_m = 0.0f;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < QR2_K; ++i) {
        స్థిరరాశి పూర్ణాంకం sc = scales[2*i];

        స్థిరరాశి పూర్ణాంకం vi = (v >> (2*i)) & 0x03030303;

        sumf_d +=
            d8[i] * (dpct::dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        పూర్ణాంకం m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] *
                  dpct::dp4a(
                      m, u[i],
                      0); // multiply constant q2_K part with sum of q8_1 values
    }

    స్థిరరాశి sycl::float2 dm2f =
        dm2.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

    తిరిగి dm2f.x() * sumf_d - dm2f.y() * sumf_m;
}


#నిర్వచించు VDR_Q3_K_Q8_1_MMVQ 1

// contiguous v/x values
స్థిర __dpct_inline__ దశాంశం vec_dot_q3_K_q8_1_impl_mmvq(
    స్థిరరాశి పూర్ణాంకం &vl, స్థిరరాశి పూర్ణాంకం &vh, స్థిరరాశి పూర్ణాంకం *__restrict__ u,
    స్థిరరాశి uint8_t *__restrict__ scales, స్థిరరాశి పూర్ణాంకం &scale_స్థానభ్రంశం,
    స్థిరరాశి దశాంశం &d3, స్థిరరాశి దశాంశం *__restrict__ d8) {

    దశాంశం sumf = 0.0f;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < QR3_K; ++i) {
        స్థిరరాశి పూర్ణాంకం isc = scale_స్థానభ్రంశం + 2*i;

        స్థిరరాశి పూర్ణాంకం isc_దిగువ = isc % (QK_K/32);
        స్థిరరాశి పూర్ణాంకం sc_జరుపు_దిగువ = 4 * (isc / (QK_K/32));
        స్థిరరాశి పూర్ణాంకం sc_దిగువ  = (scales[isc_దిగువ] >> sc_జరుపు_దిగువ) & 0xF;

        స్థిరరాశి పూర్ణాంకం isc_ఎగువ = isc % (QK_K/64);
        స్థిరరాశి పూర్ణాంకం sc_జరుపు_ఎగువ = 2 * (isc / (QK_K/64));
        స్థిరరాశి పూర్ణాంకం sc_ఎగువ = ((scales[(QK_K/32) + isc_ఎగువ] >> sc_జరుపు_ఎగువ) & 3) << 4;

        స్థిరరాశి పూర్ణాంకం sc = (sc_దిగువ | sc_ఎగువ) - 32;

        స్థిరరాశి పూర్ణాంకం vil = (vl >> (2*i)) & 0x03030303;

        స్థిరరాశి పూర్ణాంకం vih = ((vh >> i) << 2) & 0x04040404;

        స్థిరరాశి పూర్ణాంకం vi =
            dpct::vectorized_ద్విమాన<sycl::char4>(vil, vih, dpct::sub_sat());

        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    తిరిగి d3 * sumf;
}

#నిర్వచించు VDR_Q4_K_Q8_1_MMVQ 2

// contiguous v/x values
స్థిర __dpct_inline__ దశాంశం vec_dot_q4_K_q8_1_impl_vmmq(
    స్థిరరాశి పూర్ణాంకం *__restrict__ v, స్థిరరాశి పూర్ణాంకం *__restrict__ u,
    స్థిరరాశి uint8_t *__restrict__ sc, స్థిరరాశి uint8_t *__restrict__ m,
    స్థిరరాశి sycl::half2 &dm4, స్థిరరాశి దశాంశం *__restrict__ d8) {

    దశాంశం sumf_d = 0.0f;
    దశాంశం sumf_m = 0.0f;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < QR4_K; ++i) {
        స్థిరరాశి పూర్ణాంకం v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        స్థిరరాశి పూర్ణాంకం v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        స్థిరరాశి పూర్ణాంకం dot1 =
            dpct::dp4a(v1i, u[2 * i + 1],
                       dpct::dp4a(v0i, u[2 * i + 0], 0)); // SIMD dot product
        స్థిరరాశి పూర్ణాంకం dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 1],
                       dpct::dp4a(0x01010101, u[2 * i + 0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    స్థిరరాశి sycl::float2 dm4f =
        dm4.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

    తిరిగి dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


#నిర్వచించు VDR_Q5_K_Q8_1_MMVQ 2

// contiguous v/x values
స్థిర __dpct_inline__ దశాంశం vec_dot_q5_K_q8_1_impl_vmmq(
    స్థిరరాశి పూర్ణాంకం *__restrict__ vl, స్థిరరాశి పూర్ణాంకం *__restrict__ vh,
    స్థిరరాశి పూర్ణాంకం *__restrict__ u, స్థిరరాశి uint8_t *__restrict__ sc,
    స్థిరరాశి uint8_t *__restrict__ m, స్థిరరాశి sycl::half2 &dm5,
    స్థిరరాశి దశాంశం *__restrict__ d8) {

    దశాంశం sumf_d = 0.0f;
    దశాంశం sumf_m = 0.0f;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < QR5_K; ++i) {
        స్థిరరాశి పూర్ణాంకం vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        స్థిరరాశి పూర్ణాంకం vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        స్థిరరాశి పూర్ణాంకం vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        స్థిరరాశి పూర్ణాంకం vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        స్థిరరాశి పూర్ణాంకం v0i = vl0i | vh0i;
        స్థిరరాశి పూర్ణాంకం v1i = vl1i | vh1i;

        స్థిరరాశి పూర్ణాంకం dot1 =
            dpct::dp4a(v0i, u[2 * i + 0],
                       dpct::dp4a(v1i, u[2 * i + 1], 0)); // SIMD dot product
        స్థిరరాశి పూర్ణాంకం dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 0],
                       dpct::dp4a(0x01010101, u[2 * i + 1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    స్థిరరాశి sycl::float2 dm5f =
        dm5.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

    తిరిగి dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}


#నిర్వచించు VDR_Q6_K_Q8_1_MMVQ 1

// contiguous v/x values
స్థిర __dpct_inline__ దశాంశం
vec_dot_q6_K_q8_1_impl_mmvq(స్థిరరాశి పూర్ణాంకం &vl, స్థిరరాశి పూర్ణాంకం &vh,
                            స్థిరరాశి పూర్ణాంకం *__restrict__ u,
                            స్థిరరాశి int8_t *__restrict__ scales, స్థిరరాశి దశాంశం &d,
                            స్థిరరాశి దశాంశం *__restrict__ d8) {
    తిరిగి vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
}

#నిర్వచించు VDR_Q1_0_Q8_1_MMVQ 1
#నిర్వచించు VDR_Q1_0_Q8_1_MMQ  4

స్థిర __dpct_inline__ దశాంశం
vec_dot_q1_0_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                  స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {

    స్థిరరాశి ఖండం_q1_0 * bq1_0 = (స్థిరరాశి ఖండం_q1_0 *) vbq;

    స్థిరరాశి ఖండం_q8_1 * bq8_1_chunk = bq8_1 + iqs;
    స్థిరరాశి దశాంశం        d1          = bq1_0->d;
    స్థిరరాశి పూర్ణాంకం          v           = get_int_from_uint8_aligned(bq1_0->qs, iqs);

    పూర్ణాంకం vi_బైట్లు[8];
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) {
        స్థిరరాశి పూర్ణాంకం జరుపు = j * 4;
        స్థిరరాశి పూర్ణాంకం bits4 = (v >> జరుపు) & 0x0F;
        స్థిరరాశి పూర్ణాంకం b0    = (bits4 & 0x01) ? 1 : -1;
        స్థిరరాశి పూర్ణాంకం b1    = (bits4 & 0x02) ? 1 : -1;
        స్థిరరాశి పూర్ణాంకం b2    = (bits4 & 0x04) ? 1 : -1;
        స్థిరరాశి పూర్ణాంకం b3    = (bits4 & 0x08) ? 1 : -1;
        vi_బైట్లు[j]     = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }

    పూర్ణాంకం sumi = 0;
#సూచన unroll
    కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) {
        స్థిరరాశి పూర్ణాంకం u = get_int_from_int8_aligned(bq8_1_chunk->qs, j);
        sumi        = ggml_sycl_dp4a(vi_బైట్లు[j], u, sumi);
    }

    తిరిగి d1 * bq8_1_chunk->ds[0] * sumi;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

మూస <ggml_type T> నిర్మాణం reorder_vec_dot_q_sycl {
    static_assert(T != T, "ggml_type for reorder vecdot not implemented");
};

// For some types the weight side of the dot product does not depend on the destination column, so a
// multi-column mul_mat_vec can unpack it once per block instead of once per column. Such a type adds
// load() and dot() next to operator() and opts in here. See reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>.
మూస <ggml_type T> నిర్మాణం reorder_vec_dot_shared_weights {
    స్థిర constexpr తార్కిక value = అబద్ధం;
};

మూస <> నిర్మాణం reorder_vec_dot_shared_weights<GGML_TYPE_Q4_K> {
    స్థిర constexpr తార్కిక value = నిజం;
};

మూస <> నిర్మాణం reorder_vec_dot_shared_weights<GGML_TYPE_Q5_K> {
    స్థిర constexpr తార్కిక value = నిజం;
};

మూస <ggml_type T> నిర్మాణం reorder_vec_dot_shared_activations {
    స్థిర constexpr తార్కిక value = అబద్ధం;
};

మూస <> నిర్మాణం reorder_vec_dot_shared_activations<GGML_TYPE_Q4_K> {
    స్థిర constexpr తార్కిక value = నిజం;
};

మూస <> నిర్మాణం reorder_vec_dot_shared_activations<GGML_TYPE_Q5_K> {
    స్థిర constexpr తార్కిక value = నిజం;
};

మూస <> నిర్మాణం reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0> {
    స్థిర constexpr ggml_type gtype = GGML_TYPE_Q4_0;

    ఉపయోగించి q4_0_ఖండం  = ggml_sycl_reordered::ఖండం_q_t<GGML_TYPE_Q4_0>;
    ఉపయోగించి q4_0_traits = రకం_పేరు q4_0_ఖండం::traits;

    __dpct_inline__ దశాంశం vec_dot_q4_0_q8_1_impl(స్థిరరాశి పూర్ణాంకం * v, స్థిరరాశి పూర్ణాంకం * u, స్థిరరాశి దశాంశం & d4, స్థిరరాశి sycl::half2 & ds8) {
        పూర్ణాంకం sumi = 0;

#సూచన unroll
        కొరకు (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            స్థిరరాశి పూర్ణాంకం vi0 = (v[i] >> 0) & 0x0F0F0F0F;
            స్థిరరాశి పూర్ణాంకం vi1 = (v[i] >> 4) & 0x0F0F0F0F;

            // SIMD dot product of quantized values
            sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
            sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
        }

        స్థిరరాశి sycl::float2 ds8f = ds8.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

        // second part effectively subtracts 8 from each quant value
        తిరిగి d4 * (sumi * ds8f.x() - (8 * q4_0_traits::vdr_mmvq / q4_0_traits::qi) * ds8f.y());
    }

    __dpct_inline__ దశాంశం క్రియాకారకం()(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> ibx_స్థానభ్రంశం,
                                     స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> d_స్థానభ్రంశం, స్థిరరాశి int8_t * q8_1_quant_ptr,
                                     స్థిరరాశి sycl::half2 * q8_1_ds, స్థిరరాశి పూర్ణాంకం & iqs) {
        స్థిరరాశి uint8_t * bq4_0 = static_cast<స్థిరరాశి uint8_t *>(vbq) + ibx_స్థానభ్రంశం.మొదటి;
        స్థిరరాశి ggml_half d = *(reinterpret_cast<స్థిరరాశి ggml_half *>(static_cast<స్థిరరాశి uint8_t *>(vbq) + d_స్థానభ్రంశం.మొదటి));
        పూర్ణాంకం             v[q4_0_traits::vdr_mmvq];
        పూర్ణాంకం             u[2 * q4_0_traits::vdr_mmvq];


#సూచన unroll
        కొరకు (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            v[i]         = get_int_from_uint8(bq4_0, iqs + i);
            u[2 * i + 0] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
            u[2 * i + 1] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i + q4_0_traits::qi);
        }

        తిరిగి vec_dot_q4_0_q8_1_impl(v, u, d, *q8_1_ds);
    }
};

// Load four contiguous dwords per operand instead of loading each value separately.
నిర్మాణం reorder_vec_dot_q8_0_wide {
    స్థిర constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    ఉపయోగించి q8_0_ఖండం  = ggml_sycl_reordered::ఖండం_q_t<GGML_TYPE_Q8_0>;
    ఉపయోగించి q8_0_traits = రకం_పేరు q8_0_ఖండం::traits;

    __dpct_inline__ దశాంశం క్రియాకారకం()(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> ibx_స్థానభ్రంశం,
                                     స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> d_స్థానభ్రంశం, స్థిరరాశి int8_t * q8_1_quant_ptr,
                                     స్థిరరాశి sycl::half2 * q8_1_ds, స్థిరరాశి పూర్ణాంకం & iqs) {
        static_assert(q8_0_traits::vdr_mmvq == 4, "the wide load moves exactly four dwords");

        స్థిరరాశి uint8_t * పునాది = static_cast<స్థిరరాశి uint8_t *>(vbq);
        స్థిరరాశి int8_t *  qs   = reinterpret_cast<స్థిరరాశి int8_t *>(పునాది + ibx_స్థానభ్రంశం.మొదటి);
        స్థిరరాశి ggml_half d    = *reinterpret_cast<స్థిరరాశి ggml_half *>(పునాది + d_స్థానభ్రంశం.మొదటి);

        స్థిరరాశి sycl::int4 v = *reinterpret_cast<స్థిరరాశి sycl::int4 *>(qs + పరిమాణం(పూర్ణాంకం) * iqs);
        స్థిరరాశి sycl::int4 u = *reinterpret_cast<స్థిరరాశి sycl::int4 *>(q8_1_quant_ptr + పరిమాణం(పూర్ణాంకం) * iqs);

        పూర్ణాంకం sumi = 0;
#సూచన unroll
        కొరకు (పూర్ణాంకం i = 0; i < 4; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        స్థిరరాశి sycl::half2 ds_విలువలు = *q8_1_ds;
        తిరిగి static_cast<దశాంశం>(d) * static_cast<దశాంశం>(ds_విలువలు[0]) * sumi;
    }
};

మూస <> నిర్మాణం reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0> {
    స్థిర constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    ఉపయోగించి q8_0_ఖండం  = ggml_sycl_reordered::ఖండం_q_t<GGML_TYPE_Q8_0>;
    ఉపయోగించి q8_0_traits = రకం_పేరు q8_0_ఖండం::traits;

    __dpct_inline__ దశాంశం క్రియాకారకం()(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> ibx_స్థానభ్రంశం,
                                     స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> d_స్థానభ్రంశం, స్థిరరాశి int8_t * q8_1_quant_ptr,
                                     స్థిరరాశి sycl::half2 * q8_1_ds, స్థిరరాశి పూర్ణాంకం & iqs,
                                     స్థిరరాశి పూర్ణాంకం vdr = q8_0_traits::vdr_mmvq) {
        స్థిరరాశి uint8_t * పునాది = static_cast<స్థిరరాశి uint8_t *>(vbq);
        స్థిరరాశి int8_t *  qs   = reinterpret_cast<స్థిరరాశి int8_t *>(పునాది + ibx_స్థానభ్రంశం.మొదటి);
        స్థిరరాశి ggml_half  d   = *reinterpret_cast<స్థిరరాశి ggml_half *>(పునాది + d_స్థానభ్రంశం.మొదటి);

        పూర్ణాంకం v[q8_0_traits::vdr_mmvq];
        పూర్ణాంకం u[q8_0_traits::vdr_mmvq];

#సూచన unroll
        కొరకు (పూర్ణాంకం i = 0; i < vdr; ++i) {
            v[i] = get_int_from_int8(qs, iqs + i);
            u[i] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
        }

        పూర్ణాంకం sumi = 0;
#సూచన unroll
        కొరకు (పూర్ణాంకం i = 0; i < vdr; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        స్థిరరాశి sycl::half2 ds_విలువలు = *q8_1_ds;
        తిరిగి static_cast<దశాంశం>(d) * static_cast<దశాంశం>(ds_విలువలు[0]) * sumi;
    }
};

మూస <> నిర్మాణం reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K> {
    స్థిర constexpr ggml_type gtype = GGML_TYPE_Q2_K;

    ఉపయోగించి q2_k_ఖండం  = ggml_sycl_reordered::ఖండం_q_t<GGML_TYPE_Q2_K>;
    ఉపయోగించి q2_k_traits = రకం_పేరు q2_k_ఖండం::traits;

    __dpct_inline__ దశాంశం క్రియాకారకం()(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> ibx_స్థానభ్రంశం,
                                     స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> d_స్థానభ్రంశం, స్థిరరాశి int8_t * q8_1_quant_ptr,
                                     స్థిరరాశి sycl::half2 * q8_1_ds, స్థిరరాశి పూర్ణాంకం & iqs) {
        స్థిరరాశి uint8_t *    పునాది   = static_cast<స్థిరరాశి uint8_t *>(vbq);
        స్థిరరాశి uint8_t *    qs     = పునాది + ibx_స్థానభ్రంశం.మొదటి;
        స్థిరరాశి uint8_t *    scales = పునాది + d_స్థానభ్రంశం.మొదటి;
        స్థిరరాశి ggml_half2 * dm     = reinterpret_cast<స్థిరరాశి ggml_half2 *>(పునాది + d_స్థానభ్రంశం.second);

        స్థిరరాశి పూర్ణాంకం bq8_స్థానభ్రంశం   = QR2_K * (iqs / QI8_1);
        స్థిరరాశి పూర్ణాంకం scale_స్థానభ్రంశం = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        స్థిరరాశి పూర్ణాంకం v = get_int_from_uint8_aligned(qs, iqs);

        పూర్ణాంకం   u[QR2_K];
        దశాంశం d8[QR2_K];

#సూచన unroll
        కొరకు (పూర్ణాంకం i = 0; i < QR2_K; ++i) {
            స్థిరరాశి int8_t * quant_పునాది_ptr = q8_1_quant_ptr + (bq8_స్థానభ్రంశం + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_పునాది_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_స్థానభ్రంశం + i))[0];
        }

        తిరిగి vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales + scale_స్థానభ్రంశం, *dm, d8);
    }
};

మూస <> నిర్మాణం reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K> {
    స్థిర constexpr ggml_type gtype = GGML_TYPE_Q3_K;

    ఉపయోగించి q3_k_ఖండం  = ggml_sycl_reordered::ఖండం_q_t<GGML_TYPE_Q3_K>;
    ఉపయోగించి q3_k_traits = రకం_పేరు q3_k_ఖండం::traits;

    __dpct_inline__ దశాంశం క్రియాకారకం()(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> ibx_స్థానభ్రంశం,
                                     స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> d_స్థానభ్రంశం, స్థిరరాశి int8_t * q8_1_quant_ptr,
                                     స్థిరరాశి sycl::half2 * q8_1_ds, స్థిరరాశి పూర్ణాంకం & iqs) {
        స్థిరరాశి uint8_t *  పునాది   = static_cast<స్థిరరాశి uint8_t *>(vbq);
        స్థిరరాశి uint8_t *  qs     = పునాది + ibx_స్థానభ్రంశం.మొదటి;
        స్థిరరాశి uint8_t *  hmask  = పునాది + ibx_స్థానభ్రంశం.second;
        స్థిరరాశి uint8_t *  scales = పునాది + d_స్థానభ్రంశం.మొదటి;
        స్థిరరాశి ggml_half  d      = *reinterpret_cast<స్థిరరాశి ggml_half *>(పునాది + d_స్థానభ్రంశం.second);

        స్థిరరాశి పూర్ణాంకం bq8_స్థానభ్రంశం   = QR3_K * (iqs / (QI3_K / 2));
        స్థిరరాశి పూర్ణాంకం scale_స్థానభ్రంశం = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        స్థిరరాశి పూర్ణాంకం vl = get_int_from_uint8(qs, iqs);
        స్థిరరాశి పూర్ణాంకం vh = ~get_int_from_uint8(hmask, iqs % (QI3_K / 2)) >> bq8_స్థానభ్రంశం;

        పూర్ణాంకం   u[QR3_K];
        దశాంశం d8[QR3_K];

#సూచన unroll
        కొరకు (పూర్ణాంకం i = 0; i < QR3_K; ++i) {
            స్థిరరాశి int8_t * quant_పునాది_ptr = q8_1_quant_ptr + (bq8_స్థానభ్రంశం + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_పునాది_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_స్థానభ్రంశం + i))[0];
        }

        తిరిగి vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, scales, scale_స్థానభ్రంశం, static_cast<దశాంశం>(d), d8);
    }
};

స్థిర అంతఃవరుస దశాంశం vec_dot_q4_K_q8_1_common(స్థిరరాశి పూర్ణాంకం * __restrict__ q4, స్థిరరాశి uint16_t * __restrict__ scales,
                                             స్థిరరాశి ggml_half2 & dm, స్థిరరాశి ఖండం_q8_1 * __restrict__ bq8_1,
                                             స్థిరరాశి పూర్ణాంకం &        iqs) {
    పూర్ణాంకం   v[2];
    పూర్ణాంకం   u[2 * QR4_K];
    దశాంశం d8[QR4_K];

    v[0] = q4[0];
    v[1] = q4[4];

    uint16_t  aux[2];
    స్థిరరాశి పూర్ణాంకం j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
    ఒకవేళ (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } లేకపోతే {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }

    స్థిరరాశి uint8_t * sc = (స్థిరరాశి uint8_t *) aux;
    స్థిరరాశి uint8_t * m  = sc + 2;

    స్థిరరాశి పూర్ణాంకం bq8_స్థానభ్రంశం = QR4_K * ((iqs / 2) / (QI8_1 / 2));

    కొరకు (పూర్ణాంకం i = 0; i < QR4_K; ++i) {
        స్థిరరాశి ఖండం_q8_1 * bq8i = bq8_1 + bq8_స్థానభ్రంశం + i;
        d8[i]                   = bq8i->ds[0];

        స్థిరరాశి పూర్ణాంకం * q8 = (స్థిరరాశి పూర్ణాంకం *) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0]   = q8[0];
        u[2 * i + 1]   = q8[4];
    }

    తిరిగి vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
}

మూస <> నిర్మాణం reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K> {
    స్థిర constexpr ggml_type gtype = GGML_TYPE_Q4_K;

    ఉపయోగించి q4_k_ఖండం  = ggml_sycl_reordered::ఖండం_q_t<GGML_TYPE_Q4_K>;
    ఉపయోగించి q4_k_traits = రకం_పేరు q4_k_ఖండం::traits;

    నిర్మాణం weights {
        పూర్ణాంకం        v[2];
        uint16_t   aux[2];
        ggml_half2 dm;
        పూర్ణాంకం        bq8_స్థానభ్రంశం;
    };

    నిర్మాణం activations {
        పూర్ణాంకం   u[2 * QR4_K];
        దశాంశం d8[QR4_K];
    };

    __dpct_inline__ స్థిర weights load(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> ibx_స్థానభ్రంశం,
                                        స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> d_స్థానభ్రంశం, స్థిరరాశి పూర్ణాంకం & iqs) {
        స్థిరరాశి uint8_t *    పునాది = static_cast<స్థిరరాశి uint8_t *>(vbq);
        స్థిరరాశి uint8_t *    qs   = పునాది + ibx_స్థానభ్రంశం.మొదటి;
        స్థిరరాశి uint8_t *    scs  = పునాది + d_స్థానభ్రంశం.మొదటి;
        స్థిరరాశి ggml_half2 * dms  = reinterpret_cast<స్థిరరాశి ggml_half2 *>(పునాది + d_స్థానభ్రంశం.second);

        weights w;
        w.bq8_స్థానభ్రంశం = QR4_K * ((iqs / 2) / (QI8_1 / 2));

        స్థిరరాశి పూర్ణాంకం *      q4     = (స్థిరరాశి పూర్ణాంకం *) (qs + 16 * w.bq8_స్థానభ్రంశం + 4 * ((iqs / 2) % 4));
        స్థిరరాశి uint16_t * scales = (స్థిరరాశి uint16_t *) scs;

        w.v[0] = q4[0];
        w.v[1] = q4[4];

        స్థిరరాశి పూర్ణాంకం j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        ఒకవేళ (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } లేకపోతే {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        తిరిగి w;
    }

    __dpct_inline__ స్థిర activations ఎక్కించు_activations(స్థిరరాశి int8_t * q8_1_quant_ptr,
                                                        స్థిరరాశి sycl::half2 * q8_1_ds, స్థిరరాశి పూర్ణాంకం & iqs) {
        activations a;
        స్థిరరాశి పూర్ణాంకం bq8_స్థానభ్రంశం = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        కొరకు (పూర్ణాంకం i = 0; i < QR4_K; ++i) {
            స్థిరరాశి int8_t * quant_పునాది_ptr = q8_1_quant_ptr + (bq8_స్థానభ్రంశం + i) * QK8_1;
            sycl::half2    ds_విలువలు      = *(q8_1_ds + bq8_స్థానభ్రంశం + i);

            a.d8[i] = ds_విలువలు[0];

            స్థిరరాశి పూర్ణాంకం * q8 = (స్థిరరాశి పూర్ణాంకం *) quant_పునాది_ptr + ((iqs / 2) % 4);
            a.u[2 * i + 0] = q8[0];
            a.u[2 * i + 1] = q8[4];
        }

        తిరిగి a;
    }

    __dpct_inline__ స్థిర దశాంశం apply(స్థిరరాశి weights & w, స్థిరరాశి activations & a) {
        స్థిరరాశి uint8_t * sc = (స్థిరరాశి uint8_t *) w.aux;
        స్థిరరాశి uint8_t * m  = sc + 2;

        తిరిగి vec_dot_q4_K_q8_1_impl_vmmq(w.v, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ స్థిర దశాంశం dot(స్థిరరాశి weights & w, స్థిరరాశి int8_t * q8_1_quant_ptr,
                                     స్థిరరాశి sycl::half2 * q8_1_ds, స్థిరరాశి పూర్ణాంకం & iqs) {
        స్థిరరాశి స్వయంచాలక a = ఎక్కించు_activations(q8_1_quant_ptr, q8_1_ds, iqs);

        తిరిగి apply(w, a);
    }

    __dpct_inline__ దశాంశం క్రియాకారకం()(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> ibx_స్థానభ్రంశం,
                                     స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> d_స్థానభ్రంశం, స్థిరరాశి int8_t * q8_1_quant_ptr,
                                     స్థిరరాశి sycl::half2 * q8_1_ds, స్థిరరాశి పూర్ణాంకం & iqs) {
        తిరిగి dot(load(vbq, ibx_స్థానభ్రంశం, d_స్థానభ్రంశం, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

మూస <> నిర్మాణం reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K> {
    స్థిర constexpr ggml_type gtype = GGML_TYPE_Q5_K;

    ఉపయోగించి q5_k_ఖండం  = ggml_sycl_reordered::ఖండం_q_t<GGML_TYPE_Q5_K>;
    ఉపయోగించి q5_k_traits = రకం_పేరు q5_k_ఖండం::traits;

    నిర్మాణం weights {
        పూర్ణాంకం        vl[2];
        పూర్ణాంకం        vh[2];
        uint16_t   aux[2];
        ggml_half2 dm;
    };

    // same activation layout as Q4_K
    static_assert(QR5_K == QR4_K);
    ఉపయోగించి activations = reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::activations;

    __dpct_inline__ స్థిర weights load(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> ibx_స్థానభ్రంశం,
                                        స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> d_స్థానభ్రంశం, స్థిరరాశి పూర్ణాంకం & iqs) {
        స్థిరరాశి uint8_t *    పునాది    = static_cast<స్థిరరాశి uint8_t *>(vbq);
        స్థిరరాశి uint8_t *    qs      = పునాది + ibx_స్థానభ్రంశం.మొదటి;   // low 4 bits
        స్థిరరాశి uint8_t *    qh_పునాది = పునాది + ibx_స్థానభ్రంశం.second;  // high bit
        స్థిరరాశి uint8_t *    scs     = పునాది + d_స్థానభ్రంశం.మొదటి;
        స్థిరరాశి ggml_half2 * dms     = reinterpret_cast<స్థిరరాశి ggml_half2 *>(పునాది + d_స్థానభ్రంశం.second);

        స్థిరరాశి పూర్ణాంకం        bq8_స్థానభ్రంశం = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        స్థిరరాశి పూర్ణాంకం *      ql_ptr     = (స్థిరరాశి పూర్ణాంకం *) (qs + 16 * bq8_స్థానభ్రంశం + 4 * ((iqs / 2) % 4));
        స్థిరరాశి పూర్ణాంకం *      qh_ptr     = (స్థిరరాశి పూర్ణాంకం *) (qh_పునాది + 4 * ((iqs / 2) % 4));
        స్థిరరాశి uint16_t * scales     = (స్థిరరాశి uint16_t *) scs;

        weights w;
        w.vl[0] = ql_ptr[0];
        w.vl[1] = ql_ptr[4];

        w.vh[0] = qh_ptr[0] >> bq8_స్థానభ్రంశం;
        w.vh[1] = qh_ptr[4] >> bq8_స్థానభ్రంశం;

        స్థిరరాశి పూర్ణాంకం j = (QR5_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        ఒకవేళ (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } లేకపోతే {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        తిరిగి w;
    }

    __dpct_inline__ స్థిర activations ఎక్కించు_activations(స్థిరరాశి int8_t * q8_1_quant_ptr,
                                                        స్థిరరాశి sycl::half2 * q8_1_ds, స్థిరరాశి పూర్ణాంకం & iqs) {
        తిరిగి reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::ఎక్కించు_activations(q8_1_quant_ptr, q8_1_ds, iqs);
    }

    __dpct_inline__ స్థిర దశాంశం apply(స్థిరరాశి weights & w, స్థిరరాశి activations & a) {
        స్థిరరాశి uint8_t * sc = (స్థిరరాశి uint8_t *) w.aux;
        స్థిరరాశి uint8_t * m  = sc + 2;

        తిరిగి vec_dot_q5_K_q8_1_impl_vmmq(w.vl, w.vh, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ స్థిర దశాంశం dot(స్థిరరాశి weights & w, స్థిరరాశి int8_t * q8_1_quant_ptr,
                                     స్థిరరాశి sycl::half2 * q8_1_ds, స్థిరరాశి పూర్ణాంకం & iqs) {
        తిరిగి apply(w, ఎక్కించు_activations(q8_1_quant_ptr, q8_1_ds, iqs));
    }

    __dpct_inline__ దశాంశం క్రియాకారకం()(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> ibx_స్థానభ్రంశం,
                                     స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> d_స్థానభ్రంశం, స్థిరరాశి int8_t * q8_1_quant_ptr,
                                     స్థిరరాశి sycl::half2 * q8_1_ds, స్థిరరాశి పూర్ణాంకం & iqs) {
        తిరిగి dot(load(vbq, ibx_స్థానభ్రంశం, d_స్థానభ్రంశం, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

మూస <> నిర్మాణం reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K> {
    స్థిర constexpr ggml_type gtype = GGML_TYPE_Q6_K;

    ఉపయోగించి q6_k_ఖండం  = ggml_sycl_reordered::ఖండం_q_t<GGML_TYPE_Q6_K>;
    ఉపయోగించి q6_k_traits = రకం_పేరు q6_k_ఖండం::traits;

    __dpct_inline__ దశాంశం vec_dot_q6_K_q8_1_impl_mmvq(స్థిరరాశి పూర్ణాంకం vl, స్థిరరాశి పూర్ణాంకం vh, స్థిరరాశి పూర్ణాంకం * __restrict__ u,
                                                      స్థిరరాశి int8_t * __restrict__ scales, స్థిరరాశి దశాంశం d,
                                                      స్థిరరాశి దశాంశం * __restrict__ d8) {
        తిరిగి vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
    }

    __dpct_inline__ దశాంశం క్రియాకారకం()(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> ibx_స్థానభ్రంశం,
                     స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> d_స్థానభ్రంశం, స్థిరరాశి int8_t * q8_1_quant_ptr, స్థిరరాశి sycl::half2 * q8_1_ds,
                     స్థిరరాశి పూర్ణాంకం iqs) {
        స్థిరరాశి uint8_t *   పునాది   = static_cast<స్థిరరాశి uint8_t *>(vbq);
        స్థిరరాశి uint8_t *   ql     = పునాది + ibx_స్థానభ్రంశం.మొదటి;
        స్థిరరాశి uint8_t *   qh     = పునాది + ibx_స్థానభ్రంశం.second;
        స్థిరరాశి int8_t *    scales = reinterpret_cast<స్థిరరాశి int8_t *>(పునాది + d_స్థానభ్రంశం.మొదటి);
        స్థిరరాశి ggml_half * d      = (స్థిరరాశి ggml_half *) (పునాది + d_స్థానభ్రంశం.second);

        స్థిరరాశి పూర్ణాంకం bq8_స్థానభ్రంశం   = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        స్థిరరాశి పూర్ణాంకం scale_స్థానభ్రంశం = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        స్థిరరాశి పూర్ణాంకం vh_జరుపు     = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

        స్థిరరాశి పూర్ణాంకం vl = get_int_from_uint8(ql, iqs);
        స్థిరరాశి పూర్ణాంకం vh = get_int_from_uint8(qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_జరుపు;

        స్థిరరాశి int8_t * scs = scales + scale_స్థానభ్రంశం;

        స్థిరరాశి పూర్ణాంకం u0 = get_int_from_int8_aligned(
            q8_1_quant_ptr + bq8_స్థానభ్రంశం * QK8_1, iqs % QI8_1);
        స్థిరరాశి పూర్ణాంకం u1 = get_int_from_int8_aligned(
            q8_1_quant_ptr + (bq8_స్థానభ్రంశం + 2) * QK8_1, iqs % QI8_1);
        స్థిరరాశి దశాంశం d80 = (*(q8_1_ds + bq8_స్థానభ్రంశం + 0))[0];
        స్థిరరాశి దశాంశం d81 = (*(q8_1_ds + bq8_స్థానభ్రంశం + 2))[0];

        తిరిగి vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u0, u1, scs[0], scs[4], *d, d80, d81);
    }
};

మూస <> నిర్మాణం reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4> {
    స్థిర constexpr ggml_type gtype = GGML_TYPE_MXFP4;

    ఉపయోగించి mxfp4_ఖండం  = ggml_sycl_reordered::ఖండం_q_t<GGML_TYPE_MXFP4>;
    ఉపయోగించి mxfp4_traits = రకం_పేరు mxfp4_ఖండం::traits;

    __dpct_inline__ దశాంశం క్రియాకారకం()(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> ibx_స్థానభ్రంశం,
                                     స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> d_స్థానభ్రంశం, స్థిరరాశి int8_t * q8_1_quant_ptr,
                                     స్థిరరాశి sycl::half2 * q8_1_ds, స్థిరరాశి పూర్ణాంకం & iqs) {
        static_assert(mxfp4_traits::vdr_mmvq == 2, "vector load assumes vdr_mmvq == 2");
        స్థిరరాశి uint8_t * పునాది = static_cast<స్థిరరాశి uint8_t *>(vbq);

        // Reordered nibble blocks are 16 contiguous bytes and iqs is 0 or 2, so each lane's two
        // weight ints are one aligned 8-byte load (the AoS layout needed eight byte loads).
        స్థిరరాశి sycl::int2 q4 = *reinterpret_cast<స్థిరరాశి sycl::int2 *>(పునాది + ibx_స్థానభ్రంశం.మొదటి + పరిమాణం(పూర్ణాంకం) * iqs);
        స్థిరరాశి uint8_t    e  = పునాది[d_స్థానభ్రంశం.మొదటి];

        // Low nibbles pair with q8_1 ints iqs..iqs+1, high nibbles with iqs+4..iqs+5.
        స్థిరరాశి sycl::int2 u_lo = *reinterpret_cast<స్థిరరాశి sycl::int2 *>(q8_1_quant_ptr + పరిమాణం(పూర్ణాంకం) * iqs);
        స్థిరరాశి sycl::int2 u_hi = *reinterpret_cast<స్థిరరాశి sycl::int2 *>(q8_1_quant_ptr + పరిమాణం(పూర్ణాంకం) * (iqs + 4));

        స్థిరరాశి sycl::int2 v0 = get_int_from_mxfp4(q4.x());
        స్థిరరాశి sycl::int2 v1 = get_int_from_mxfp4(q4.y());

        పూర్ణాంకం sumi = 0;
        sumi = ggml_sycl_dp4a(v0.x(), u_lo.x(), sumi);
        sumi = ggml_sycl_dp4a(v0.y(), u_hi.x(), sumi);
        sumi = ggml_sycl_dp4a(v1.x(), u_lo.y(), sumi);
        sumi = ggml_sycl_dp4a(v1.y(), u_hi.y(), sumi);

        స్థిరరాశి దశాంశం d = ggml_sycl_e8m0_గమ్యం_fp32(e) * 0.5f * static_cast<దశాంశం>((*q8_1_ds)[0]);
        తిరిగి d * sumi;
    }
};
#నిర్వచించు VDR_Q4_0_Q8_1_MMVQ 2
#నిర్వచించు VDR_Q4_0_Q8_1_MMQ  4

#నిర్వచించు VDR_Q2_0_Q8_1_MMVQ 1

మూస <పూర్ణాంకం vdr>
స్థిర __dpct_inline__ దశాంశం vec_dot_q2_0_q8_1_impl(
    స్థిరరాశి పూర్ణాంకం * v,
    స్థిరరాశి పూర్ణాంకం * u,
    స్థిరరాశి దశాంశం & d2,
    స్థిరరాశి sycl::half2 & ds8) {
    పూర్ణాంకం sumi = 0;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < vdr; ++i) {
#సూచన unroll
        కొరకు (పూర్ణాంకం j = 0; j < 4; ++j) {
            స్థిరరాశి uint8_t q = (uint8_t) ((uint32_t) v[i] >> (8 * j));

            // unpack 2-bit values to byte lanes (0..3), then apply zero-point
            // correction with ds8f.y() below, mirroring the q4_0 style.
            పూర్ణాంకం vi = 0;
            vi |= (((q >> 0) & 0x3) & 0xFF) << 0;
            vi |= (((q >> 2) & 0x3) & 0xFF) << 8;
            vi |= (((q >> 4) & 0x3) & 0xFF) << 16;
            vi |= (((q >> 6) & 0x3) & 0xFF) << 24;

            sumi = dpct::dp4a(vi, u[4 * i + j], sumi);
        }
    }

    స్థిరరాశి sycl::float2 ds8f = ds8.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
    // q2_0 has zero-point 1. Scale ds8f.y() by processed-lane ratio,
    // consistent with q4_0's explicit zero-point subtraction style.
    తిరిగి d2 * (sumi * ds8f.x() - ((దశాంశం) vdr / (దశాంశం) QI2_0) * ds8f.y());
}

మూస <పూర్ణాంకం vdr>
స్థిర __dpct_inline__ దశాంశం vec_dot_q4_0_q8_1_impl(స్థిరరాశి పూర్ణాంకం * v, స్థిరరాశి పూర్ణాంకం * u, స్థిరరాశి దశాంశం & d4,
                                                    స్థిరరాశి sycl::half2 & ds8) {
    పూర్ణాంకం sumi = 0;
#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < vdr; ++i) {
        స్థిరరాశి పూర్ణాంకం vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        స్థిరరాశి పూర్ణాంకం vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

    స్థిరరాశి sycl::float2 ds8f = ds8.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

    // second part effectively subtracts 8 from each quant value
    తిరిగి d4 * (sumi * ds8f.x() - (8 * vdr / QI4_0) * ds8f.y());
}

#నిర్వచించు VDR_Q4_1_Q8_1_MMVQ 2
#నిర్వచించు VDR_Q4_1_Q8_1_MMQ  4

మూస <పూర్ణాంకం vdr>
స్థిర __dpct_inline__ దశాంశం vec_dot_q4_1_q8_1_impl(స్థిరరాశి పూర్ణాంకం *v, స్థిరరాశి పూర్ణాంకం *u,
                                                    స్థిరరాశి sycl::half2 &dm4,
                                                    స్థిరరాశి sycl::half2 &ds8) {

    పూర్ణాంకం sumi = 0;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < vdr; ++i) {
        స్థిరరాశి పూర్ణాంకం vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        స్థిరరాశి పూర్ణాంకం vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

#నిర్వచితమైతే GGML_SYCL_F16
    స్థిరరాశి sycl::float2 tmp =
        (dm4 * ds8).convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
    స్థిరరాశి దశాంశం d4d8 = tmp.x();
    స్థిరరాశి దశాంశం m4s8 = tmp.y();
#లేకపోతే
    స్థిరరాశి sycl::float2 dm4f =
        dm4.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
    స్థిరరాశి sycl::float2 ds8f =
        ds8.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
    స్థిరరాశి దశాంశం d4d8 = dm4f.x() * ds8f.x();
    స్థిరరాశి దశాంశం m4s8 = dm4f.y() * ds8f.y();
#అయితే_ముగింపు // GGML_SYCL_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    తిరిగి sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#నిర్వచించు VDR_Q5_0_Q8_1_MMVQ 2
#నిర్వచించు VDR_Q5_0_Q8_1_MMQ  4

మూస <పూర్ణాంకం vdr>
స్థిర __dpct_inline__ దశాంశం
vec_dot_q5_0_q8_1_impl(స్థిరరాశి పూర్ణాంకం *vl, స్థిరరాశి పూర్ణాంకం *vh, స్థిరరాశి పూర్ణాంకం *u,
                       స్థిరరాశి దశాంశం &d5, స్థిరరాశి sycl::half2 &ds8) {
    పూర్ణాంకం sumi = 0;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < vdr; ++i) {
        పూర్ణాంకం vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        పూర్ణాంకం vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

    స్థిరరాశి sycl::float2 ds8f =
        ds8.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();

    // second part effectively subtracts 16 from each quant value
    తిరిగి d5 * (sumi * ds8f.x() - (16 * vdr / QI5_0) * ds8f.y());
}

#నిర్వచించు VDR_Q5_1_Q8_1_MMVQ 2
#నిర్వచించు VDR_Q5_1_Q8_1_MMQ  4

మూస <పూర్ణాంకం vdr>
స్థిర __dpct_inline__ దశాంశం
vec_dot_q5_1_q8_1_impl(స్థిరరాశి పూర్ణాంకం *vl, స్థిరరాశి పూర్ణాంకం *vh, స్థిరరాశి పూర్ణాంకం *u,
                       స్థిరరాశి sycl::half2 &dm5, స్థిరరాశి sycl::half2 &ds8) {

    పూర్ణాంకం sumi = 0;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < vdr; ++i) {
        పూర్ణాంకం vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        పూర్ణాంకం vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

#నిర్వచితమైతే GGML_SYCL_F16
     స్థిరరాశి sycl::float2 tmp =
        (dm5 * ds8).convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
    స్థిరరాశి దశాంశం d5d8 = tmp.x();
    స్థిరరాశి దశాంశం m5s8 = tmp.y();


#లేకపోతే
    స్థిరరాశి sycl::float2 dm5f =
        dm5.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
    స్థిరరాశి sycl::float2 ds8f =
        ds8.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
    స్థిరరాశి దశాంశం d5d8 = dm5f.x() * ds8f.x();
    స్థిరరాశి దశాంశం m5s8 = dm5f.y() * ds8f.y();
#అయితే_ముగింపు // GGML_SYCL_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    తిరిగి sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#నిర్వచించు VDR_Q8_0_Q8_1_MMVQ 2
#నిర్వచించు VDR_Q8_0_Q8_1_MMQ 8

మూస <పూర్ణాంకం vdr>
స్థిర __dpct_inline__ దశాంశం vec_dot_q8_0_q8_1_impl(స్థిరరాశి పూర్ణాంకం *v, స్థిరరాశి పూర్ణాంకం *u,
                                                    స్థిరరాశి దశాంశం &d8_0,
                                                    స్థిరరాశి దశాంశం &d8_1) {

    పూర్ణాంకం sumi = 0;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

    తిరిగి d8_0*d8_1 * sumi;
}

మూస <రకం_పేరు T, పూర్ణాంకం vdr>
స్థిర __dpct_inline__ T vec_dot_q8_0_q8_1_impl(స్థిరరాశి పూర్ణాంకం * v, స్థిరరాశి పూర్ణాంకం * u, స్థిరరాశి T & d8_0, స్థిరరాశి T & d8_1) {
    పూర్ణాంకం sumi = 0;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_sycl_dp4a(v[i], u[i], sumi);
    }

    తిరిగి d8_0*d8_1 * ((T) sumi);
}

మూస <పూర్ణాంకం vdr>
స్థిర __dpct_inline__ దశాంశం vec_dot_q8_1_q8_1_impl(స్థిరరాశి పూర్ణాంకం *v, స్థిరరాశి పూర్ణాంకం *u,
                                                    స్థిరరాశి sycl::half2 &dm8,
                                                    స్థిరరాశి sycl::half2 &ds8) {

    పూర్ణాంకం sumi = 0;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

#నిర్వచితమైతే GGML_SYCL_F16
    స్థిరరాశి sycl::float2 tmp =
        (dm8 * ds8).convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
    స్థిరరాశి దశాంశం d8d8 = tmp.x();
    స్థిరరాశి దశాంశం m8s8 = tmp.y();
#లేకపోతే
    స్థిరరాశి sycl::float2 dm8f =
        dm8.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
    స్థిరరాశి sycl::float2 ds8f =
        ds8.convert<దశాంశం, sycl::rounding_బహుళకం::automatic>();
    స్థిరరాశి దశాంశం d8d8 = dm8f.x() * ds8f.x();
    స్థిరరాశి దశాంశం m8s8 = dm8f.y() * ds8f.y();
#అయితే_ముగింపు // GGML_SYCL_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    తిరిగి sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

స్థిర __dpct_inline__ దశాంశం
vec_dot_q4_0_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                  స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {

    స్థిరరాశి ఖండం_q4_0 * bq4_0 = (స్థిరరాశి ఖండం_q4_0 *) vbq;

    పూర్ణాంకం v[VDR_Q4_0_Q8_1_MMVQ];
    పూర్ణాంకం u[2 * VDR_Q4_0_Q8_1_MMVQ];

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]         = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    తిరిగి vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

స్థిర __dpct_inline__ దశాంశం
vec_dot_q2_0_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                  స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {

    స్థిరరాశి ఖండం_q2_0 * bq2_0 = (స్థిరరాశి ఖండం_q2_0 *) vbq;

    పూర్ణాంకం v[2 * VDR_Q2_0_Q8_1_MMVQ];
    పూర్ణాంకం u[8 * VDR_Q2_0_Q8_1_MMVQ];

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < VDR_Q2_0_Q8_1_MMVQ; ++i) {
        స్థిరరాశి పూర్ణాంకం పునాది = 4 * (iqs + i);

        // Q2_0 has QK2_0 = 64 and uses 2 x QK8_1 blocks on the RHS.
        v[2 * i + 0] = get_int_from_uint8(bq2_0->qs, iqs + i);
        v[2 * i + 1] = get_int_from_uint8(bq2_0->qs, iqs + i + QI2_0);

        u[8 * i + 0] = get_int_from_int8_aligned(bq8_1[0].qs, పునాది + 0);
        u[8 * i + 1] = get_int_from_int8_aligned(bq8_1[0].qs, పునాది + 1);
        u[8 * i + 2] = get_int_from_int8_aligned(bq8_1[0].qs, పునాది + 2);
        u[8 * i + 3] = get_int_from_int8_aligned(bq8_1[0].qs, పునాది + 3);

        u[8 * i + 4] = get_int_from_int8_aligned(bq8_1[1].qs, పునాది + 0);
        u[8 * i + 5] = get_int_from_int8_aligned(bq8_1[1].qs, పునాది + 1);
        u[8 * i + 6] = get_int_from_int8_aligned(bq8_1[1].qs, పునాది + 2);
        u[8 * i + 7] = get_int_from_int8_aligned(bq8_1[1].qs, పునాది + 3);
    }

    స్థిరరాశి దశాంశం sum0 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + 0, u + 0, bq2_0->d, bq8_1[0].ds);
    స్థిరరాశి దశాంశం sum1 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + VDR_Q2_0_Q8_1_MMVQ, u + 4 * VDR_Q2_0_Q8_1_MMVQ, bq2_0->d, bq8_1[1].ds);
    తిరిగి sum0 + sum1;
}

స్థిర __dpct_inline__ దశాంశం
vec_dot_q4_1_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                  స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {

    స్థిరరాశి ఖండం_q4_1 * bq4_1 = (స్థిరరాశి ఖండం_q4_1 *) vbq;

    పూర్ణాంకం v[VDR_Q4_1_Q8_1_MMVQ];
    పూర్ణాంకం u[2*VDR_Q4_1_Q8_1_MMVQ];

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    తిరిగి vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

#నిర్వచించు VDR_MXFP4_Q8_1_MMVQ 2
#నిర్వచించు VDR_MXFP4_Q8_1_MMQ  4

స్థిర __dpct_inline__ దశాంశం vec_dot_mxfp4_q8_1(స్థిరరాశి శూన్య * __restrict__ vbq,
                                                స్థిరరాశి ఖండం_q8_1 * __restrict__ bq8_1,
                                                స్థిరరాశి పూర్ణాంకం & iqs) {
    స్థిరరాశి ఖండం_mxfp4 * bq4 = (స్థిరరాశి ఖండం_mxfp4 *) vbq;

    స్థిరరాశి పూర్ణాంకం * q8 = (స్థిరరాశి పూర్ణాంకం *) bq8_1->qs + iqs;

    పూర్ణాంకం sumi = 0;
#సూచన unroll
    కొరకు (పూర్ణాంకం l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        స్థిరరాశి పూర్ణాంకం aux_q4 = get_int_b1(bq4->qs, iqs + l);
        స్థిరరాశి sycl::int2 v      = get_int_from_mxfp4(aux_q4);
        sumi = ggml_sycl_dp4a(v.x(), q8[l + 0], sumi);
        sumi = ggml_sycl_dp4a(v.y(), q8[l + 4], sumi);
    }

    స్థిరరాశి దశాంశం d = ggml_sycl_e8m0_గమ్యం_fp32(bq4->e) * 0.5f * (bq8_1->ds)[0];
    తిరిగి d * sumi;
}

#నిర్వచించు VDR_NVFP4_Q8_1_MMVQ 4
#నిర్వచించు VDR_NVFP4_Q8_1_MMQ  8

స్థిర __dpct_inline__ దశాంశం vec_dot_nvfp4_q8_1(స్థిరరాశి శూన్య * __restrict__ vbq,
                                                స్థిరరాశి ఖండం_q8_1 * __restrict__ bq8_1,
                                                స్థిరరాశి int32_t & iqs) {
    స్థిరరాశి ఖండం_nvfp4 * bq4 = (స్థిరరాశి ఖండం_nvfp4 *) vbq;
    దశాంశం కూడిక = 0.0f;
#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; i++) {
        స్థిరరాశి int32_t iqs0 = iqs + 2*i;
        స్థిరరాశి int32_t iqs1 = iqs0 + 1;
        స్థిరరాశి int32_t is = iqs0 >> 1;
        స్థిరరాశి sycl::int2   v0   = get_int_from_పట్టిక_16(get_int_b4(bq4->qs, iqs0), kvalues_mxfp4);
        స్థిరరాశి sycl::int2   v1   = get_int_from_పట్టిక_16(get_int_b4(bq4->qs, iqs1), kvalues_mxfp4);
        స్థిరరాశి ఖండం_q8_1 * bq8 = bq8_1 + (is >> 1);
        స్థిరరాశి int32_t i8 = ((is & 1) << 2);

        పూర్ణాంకం sumi = ggml_sycl_dp4a(v0.x(), get_int_b4(bq8->qs, i8 + 0), 0);
        sumi     = ggml_sycl_dp4a(v0.y(), get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi     = ggml_sycl_dp4a(v1.x(), get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi     = ggml_sycl_dp4a(v1.y(), get_int_b4(bq8->qs, i8 + 3), sumi);

        స్థిరరాశి దశాంశం d = ggml_sycl_ue4m3_గమ్యం_fp32(bq4->d[is]) * (bq8->ds)[0];
        కూడిక += d * దశాంశం(sumi);
    }

    తిరిగి కూడిక;
}

స్థిర __dpct_inline__ దశాంశం
vec_dot_q5_0_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                  స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {

    స్థిరరాశి ఖండం_q5_0 * bq5_0 = (స్థిరరాశి ఖండం_q5_0 *) vbq;

    పూర్ణాంకం vl[VDR_Q5_0_Q8_1_MMVQ];
    పూర్ణాంకం vh[VDR_Q5_0_Q8_1_MMVQ];
    పూర్ణాంకం  u[2*VDR_Q5_0_Q8_1_MMVQ];

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    తిరిగి vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

స్థిర __dpct_inline__ దశాంశం
vec_dot_q5_1_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                  స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {

    స్థిరరాశి ఖండం_q5_1 * bq5_1 = (స్థిరరాశి ఖండం_q5_1 *) vbq;

    పూర్ణాంకం vl[VDR_Q5_1_Q8_1_MMVQ];
    పూర్ణాంకం vh[VDR_Q5_1_Q8_1_MMVQ];
    పూర్ణాంకం  u[2*VDR_Q5_1_Q8_1_MMVQ];

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    తిరిగి vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

స్థిర __dpct_inline__ దశాంశం
vec_dot_q8_0_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                  స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {

    స్థిరరాశి ఖండం_q8_0 * bq8_0 = (స్థిరరాశి ఖండం_q8_0 *) vbq;

    పూర్ణాంకం v[VDR_Q8_0_Q8_1_MMVQ];
    పూర్ణాంకం u[VDR_Q8_0_Q8_1_MMVQ];

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    తిరిగి vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                      bq8_1->ds[0]);
}

స్థిర __dpct_inline__ దశాంశం
vec_dot_q2_K_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                  స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {

    స్థిరరాశి ఖండం_q2_K * bq2_K = (స్థిరరాశి ఖండం_q2_K *) vbq;

    స్థిరరాశి పూర్ణాంకం bq8_స్థానభ్రంశం = QR2_K * (iqs / QI8_1);
    స్థిరరాశి పూర్ణాంకం scale_స్థానభ్రంశం = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    స్థిరరాశి uint8_t * scales = bq2_K->scales + scale_స్థానభ్రంశం;

    స్థిరరాశి పూర్ణాంకం v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    పూర్ణాంకం    u[QR2_K];
    దశాంశం d8[QR2_K];

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_స్థానభ్రంశం + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_స్థానభ్రంశం + i].ds[0];
    }

    తిరిగి vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

స్థిర __dpct_inline__ దశాంశం
vec_dot_q3_K_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                  స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {

    స్థిరరాశి ఖండం_q3_K * bq3_K = (స్థిరరాశి ఖండం_q3_K *) vbq;

    స్థిరరాశి పూర్ణాంకం bq8_స్థానభ్రంశం = QR3_K * (iqs / (QI3_K/2));
    స్థిరరాశి పూర్ణాంకం scale_స్థానభ్రంశం = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    స్థిరరాశి దశాంశం d = bq3_K->d;

    స్థిరరాశి పూర్ణాంకం vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    స్థిరరాశి పూర్ణాంకం vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_స్థానభ్రంశం;

    పూర్ణాంకం    u[QR3_K];
    దశాంశం d8[QR3_K];

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_స్థానభ్రంశం + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_స్థానభ్రంశం + i].ds[0];
    }

    తిరిగి vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_స్థానభ్రంశం, d, d8);
}

స్థిర __dpct_inline__ దశాంశం vec_dot_q4_K_q8_1(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి ఖండం_q8_1 * __restrict__ bq8_1,
                                               స్థిరరాశి పూర్ణాంకం & iqs) {
#నిర్వచితం_కాకపోతే GGML_QKK_64

    స్థిరరాశి ఖండం_q4_K * bq4_K = (స్థిరరాశి ఖండం_q4_K *) vbq;

    స్థిరరాశి పూర్ణాంకం        bq8_స్థానభ్రంశం = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    స్థిరరాశి పూర్ణాంకం *      q4         = (స్థిరరాశి పూర్ణాంకం *) (bq4_K->qs + 16 * bq8_స్థానభ్రంశం + 4 * ((iqs / 2) % 4));
    స్థిరరాశి uint16_t * scales     = (స్థిరరాశి uint16_t *) bq4_K->scales;

    తిరిగి vec_dot_q4_K_q8_1_common(q4, scales, bq4_K->dm, bq8_1, iqs);

#లేకపోతే

#ఒకవేళ __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    స్థిరరాశి ఖండం_q4_K * bq4_K = (స్థిరరాశి ఖండం_q4_K *) vbq;

    దశాంశం sumf_d = 0.0f;
    దశాంశం sumf_m = 0.0f;

    uint16_t aux16[2];
    స్థిరరాశి uint8_t * s = (స్థిరరాశి uint8_t *)aux16;

    స్థిరరాశి uint16_t * a = (స్థిరరాశి uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    స్థిరరాశి దశాంశం dall = bq4_K->dm[0];
    స్థిరరాశి దశాంశం dmin = bq4_K->dm[1];

    స్థిరరాశి దశాంశం d8_1 = bq8_1[0].ds[0];
    స్థిరరాశి దశాంశం d8_2 = bq8_1[1].ds[1];

    స్థిరరాశి పూర్ణాంకం ui1 = *((స్థిరరాశి పూర్ణాంకం *)bq8_1[0].qs + (iqs/2));
    స్థిరరాశి పూర్ణాంకం ui2 = *((స్థిరరాశి పూర్ణాంకం *)bq8_1[0].qs + (iqs/2) + 4);
    స్థిరరాశి పూర్ణాంకం ui3 = *((స్థిరరాశి పూర్ణాంకం *)bq8_1[1].qs + (iqs/2));
    స్థిరరాశి పూర్ణాంకం ui4 = *((స్థిరరాశి పూర్ణాంకం *)bq8_1[1].qs + (iqs/2) + 4);

    స్థిరరాశి పూర్ణాంకం * q4 = (స్థిరరాశి పూర్ణాంకం *)bq4_K->qs + (iqs/2);
    స్థిరరాశి పూర్ణాంకం v1 = q4[0];
    స్థిరరాశి పూర్ణాంకం v2 = q4[4];

    స్థిరరాశి పూర్ణాంకం dot1 = dpct::dp4a(ui2, v2 & 0x0f0f0f0f, dpct::dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    స్థిరరాశి పూర్ణాంకం dot2 = dpct::dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, dpct::dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    స్థిరరాశి పూర్ణాంకం dot3 = dpct::dp4a(0x01010101, ui2, dpct::dp4a(0x01010101, ui1, 0));
    స్థిరరాశి పూర్ణాంకం dot4 = dpct::dp4a(0x01010101, ui4, dpct::dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    తిరిగి dall * sumf_d - dmin * sumf_m;

#లేకపోతే
    bad_arch();
#అయితే_ముగింపు // __SYCL_ARCH__ >= VER_4VEC

#అయితే_ముగింపు
}

స్థిర __dpct_inline__ దశాంశం
vec_dot_q5_K_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                  స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {

#నిర్వచితం_కాకపోతే GGML_QKK_64
    స్థిరరాశి ఖండం_q5_K * bq5_K = (స్థిరరాశి ఖండం_q5_K *) vbq;

    పూర్ణాంకం   vl[2];
    పూర్ణాంకం   vh[2];
    పూర్ణాంకం    u[2*QR5_K];
    దశాంశం d8[QR5_K];

    స్థిరరాశి పూర్ణాంకం bq8_స్థానభ్రంశం = QR5_K * ((iqs/2) / (QI8_1/2));
    స్థిరరాశి పూర్ణాంకం * ql = (స్థిరరాశి పూర్ణాంకం *)(bq5_K->qs + 16 * bq8_స్థానభ్రంశం + 4 * ((iqs/2)%4));
    స్థిరరాశి పూర్ణాంకం * qh = (స్థిరరాశి పూర్ణాంకం *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_స్థానభ్రంశం;
    vh[1] = qh[4] >> bq8_స్థానభ్రంశం;

    స్థిరరాశి uint16_t * scales = (స్థిరరాశి uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    స్థిరరాశి పూర్ణాంకం j = bq8_స్థానభ్రంశం/2;
    ఒకవేళ (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } లేకపోతే {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    స్థిరరాశి uint8_t * sc = (స్థిరరాశి uint8_t *)aux;
    స్థిరరాశి uint8_t * m  = sc + 2;

#సూచన unroll
    కొరకు (పూర్ణాంకం i = 0; i < QR5_K; ++i) {
        స్థిరరాశి ఖండం_q8_1 * bq8i = bq8_1 + bq8_స్థానభ్రంశం + i;
        d8[i] = bq8i->ds[0];

        స్థిరరాశి పూర్ణాంకం * q8 = (స్థిరరాశి పూర్ణాంకం *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    తిరిగి vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#లేకపోతే

#ఒకవేళ __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    స్థిరరాశి ఖండం_q5_K * bq5_K = (స్థిరరాశి ఖండం_q5_K *) vbq;

    స్థిరరాశి int8_t * s = bq5_K->scales;

    స్థిరరాశి దశాంశం d = bq5_K->d;

    స్థిరరాశి దశాంశం d8_1 = bq8_1[0].ds[0];
    స్థిరరాశి దశాంశం d8_2 = bq8_1[1].ds[1];

    స్థిరరాశి పూర్ణాంకం ui1 = *((స్థిరరాశి పూర్ణాంకం *)bq8_1[0].qs + (iqs/2));
    స్థిరరాశి పూర్ణాంకం ui2 = *((స్థిరరాశి పూర్ణాంకం *)bq8_1[0].qs + (iqs/2) + 4);
    స్థిరరాశి పూర్ణాంకం ui3 = *((స్థిరరాశి పూర్ణాంకం *)bq8_1[1].qs + (iqs/2));
    స్థిరరాశి పూర్ణాంకం ui4 = *((స్థిరరాశి పూర్ణాంకం *)bq8_1[1].qs + (iqs/2) + 4);

    స్థిరరాశి పూర్ణాంకం * ql = (స్థిరరాశి పూర్ణాంకం *)bq5_K->qs + (iqs/2);
    స్థిరరాశి పూర్ణాంకం vl1 = ql[0];
    స్థిరరాశి పూర్ణాంకం vl2 = ql[4];

    స్థిరరాశి పూర్ణాంకం అడుగు = 4 * (iqs/2); // 0, 4, 8, 12
    స్థిరరాశి పూర్ణాంకం im = అడుగు/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    స్థిరరాశి పూర్ణాంకం లోపల = అడుగు%8; // 0, 4, 0, 4
    స్థిరరాశి పూర్ణాంకం vh = (*((స్థిరరాశి పూర్ణాంకం *)(bq5_K->qh + లోపల))) >> im;

    స్థిరరాశి పూర్ణాంకం v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    స్థిరరాశి పూర్ణాంకం v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    స్థిరరాశి పూర్ణాంకం v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    స్థిరరాశి పూర్ణాంకం v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    స్థిరరాశి దశాంశం sumf_d = d8_1 * (dpct::dp4a(ui1, v1, 0) * s[0] + dpct::dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (dpct::dp4a(ui3, v3, 0) * s[2] + dpct::dp4a(ui4, v4, 0) * s[3]);

    తిరిగి d * sumf_d;

#లేకపోతే
    bad_arch();
#అయితే_ముగింపు // __SYCL_ARCH__ >= VER_4VEC

#అయితే_ముగింపు
}

స్థిర __dpct_inline__ దశాంశం
vec_dot_q6_K_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                  స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {

    స్థిరరాశి ఖండం_q6_K * bq6_K = (స్థిరరాశి ఖండం_q6_K *) vbq;

    స్థిరరాశి పూర్ణాంకం bq8_స్థానభ్రంశం = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    స్థిరరాశి పూర్ణాంకం scale_స్థానభ్రంశం = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    స్థిరరాశి పూర్ణాంకం vh_జరుపు = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    స్థిరరాశి పూర్ణాంకం vl = get_int_from_uint8(bq6_K->ql, iqs);
    స్థిరరాశి పూర్ణాంకం vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_జరుపు;

    స్థిరరాశి int8_t * scales = bq6_K->scales + scale_స్థానభ్రంశం;

    స్థిరరాశి పూర్ణాంకం u0 = get_int_from_int8_aligned(
        bq8_1[bq8_స్థానభ్రంశం + 0].qs, iqs % QI8_1);
    స్థిరరాశి పూర్ణాంకం u1 = get_int_from_int8_aligned(
        bq8_1[bq8_స్థానభ్రంశం + 2].qs, iqs % QI8_1);
    స్థిరరాశి దశాంశం d80 = bq8_1[bq8_స్థానభ్రంశం + 0].ds[0];
    స్థిరరాశి దశాంశం d81 = bq8_1[bq8_స్థానభ్రంశం + 2].ds[0];

    తిరిగి vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u0, u1, scales[0], scales[4], bq6_K->d, d80, d81);
}


// NOTE: the VDR_IQ*_Q8_1_MMVQ values deliberately differ from the identically named CUDA constants
// (vecdotq.cuh): the SYCL kernels pair them with a halved qi (e.g. QI3_S/2), so the values are not
// interchangeable and must not be copied across backends.
#నిర్వచించు VDR_IQ2_XXS_Q8_1_MMVQ 1

స్థిర __dpct_inline__ uint32_t iq_sign_ముసుగు(స్థిరరాశి uint32_t signs) {
    స్థిరరాశి uint32_t బిట్లు =
        (signs & 0x01) | ((signs & 0x02) << 7) |
        ((signs & 0x04) << 14) | ((signs & 0x08) << 21);

    తిరిగి బిట్లు * 0xff;
}

స్థిర __dpct_inline__ uint32_t iq_sign_బిట్లు(స్థిరరాశి uint32_t encoded) {
    స్థిరరాశి uint32_t signs = encoded & 127;

    తిరిగి signs | ((sycl::popcount(signs) & 1) << 7);
}

స్థిర __dpct_inline__ దశాంశం
vec_dot_iq2_xxs_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                     స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs,
                     స్థిరరాశి uint64_t *iq2xxs_grid, స్థిరరాశి uint8_t *ksigns_iq2xs,
                     స్థిరరాశి uint8_t *kmask_iq2xs) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq2_xxs * bq2 = (స్థిరరాశి ఖండం_iq2_xxs *) vbq;

    స్థిరరాశి పూర్ణాంకం ib32 = iqs;
    స్థిరరాశి uint16_t * q2 = bq2->qs + 4*ib32;
    స్థిరరాశి uint8_t  * aux8 = (స్థిరరాశి uint8_t *)q2;
    స్థిరరాశి int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = q2[2] | (q2[3] << 16);
    పూర్ణాంకం sumi = 0;
    కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
        స్థిరరాశి uint8_t * grid = (స్థిరరాశి uint8_t *)(iq2xxs_grid + aux8[l]);
        స్థిరరాశి uint8_t  signs = ksigns_iq2xs[aux32 & 127];
        కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) {
            sumi += q8[j] * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
        aux32 >>= 7;
    }
    స్థిరరాశి దశాంశం d = (దశాంశం)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.25f;
    తిరిగి d * sumi;
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
    తిరిగి 0.f;
#అయితే_ముగింపు
}

#నిర్వచించు VDR_IQ2_XS_Q8_1_MMVQ 1

స్థిర __dpct_inline__ దశాంశం
vec_dot_iq2_xs_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                    స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs,
                    స్థిరరాశి uint64_t *iq2xs_grid) {
#ఒకవేళ DPCT_COMPATIBILITY_తాత్కాలిక >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq2_xs * bq2 = (స్థిరరాశి ఖండం_iq2_xs *) vbq;

    స్థిరరాశి పూర్ణాంకం ib32 = iqs;
    స్థిరరాశి uint16_t * q2 = bq2->qs + 4*ib32;
    స్థిరరాశి int8_t   * q8 = bq8_1[ib32].qs;
    స్థిరరాశి uint8_t ls1 = bq2->scales[ib32] & 0xf;
    స్థిరరాశి uint8_t ls2 = bq2->scales[ib32] >>  4;
    పూర్ణాంకం sumi1 = 0;
#సూచన unroll
    కొరకు (పూర్ణాంకం l = 0; l < 2; ++l) {
        స్థిరరాశి uint64_t grid = iq2xs_grid[q2[l] & 511];
        స్థిరరాశి uint32_t sign_బిట్లు = iq_sign_బిట్లు(q2[l] >> 9);
        స్థిరరాశి uint32_t signs0 = iq_sign_ముసుగు(sign_బిట్లు);
        స్థిరరాశి uint32_t signs1 = iq_sign_ముసుగు(sign_బిట్లు >> 4);
        స్థిరరాశి పూర్ణాంకం grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        స్థిరరాశి పూర్ణాంకం grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi1 = dpct::dp4a(grid_l, *((స్థిరరాశి పూర్ణాంకం *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((స్థిరరాశి పూర్ణాంకం *)q8 + 1), sumi1);
        q8 += 8;
    }
    పూర్ణాంకం sumi2 = 0;
#సూచన unroll
    కొరకు (పూర్ణాంకం l = 2; l < 4; ++l) {
        స్థిరరాశి uint64_t grid = iq2xs_grid[q2[l] & 511];
        స్థిరరాశి uint32_t sign_బిట్లు = iq_sign_బిట్లు(q2[l] >> 9);
        స్థిరరాశి uint32_t signs0 = iq_sign_ముసుగు(sign_బిట్లు);
        స్థిరరాశి uint32_t signs1 = iq_sign_ముసుగు(sign_బిట్లు >> 4);
        స్థిరరాశి పూర్ణాంకం grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        స్థిరరాశి పూర్ణాంకం grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi2 = dpct::dp4a(grid_l, *((స్థిరరాశి పూర్ణాంకం *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((స్థిరరాశి పూర్ణాంకం *)q8 + 1), sumi2);
        q8 += 8;
    }
    స్థిరరాశి దశాంశం d = (దశాంశం)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    తిరిగి d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
    తిరిగి 0.f;
#అయితే_ముగింపు
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
    తిరిగి 0.f;
#అయితే_ముగింపు
}

#నిర్వచించు VDR_IQ2_S_Q8_1_MMVQ 1

స్థిర __dpct_inline__ దశాంశం
vec_dot_iq2_s_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                   స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq2_s * bq2 = (స్థిరరాశి ఖండం_iq2_s *) vbq;

    స్థిరరాశి పూర్ణాంకం ib32 = iqs;
    స్థిరరాశి int8_t  * q8 = bq8_1[ib32].qs;
    స్థిరరాశి uint8_t * signs = bq2->qs + QK_K/8 + 4*ib32;
    స్థిరరాశి uint8_t ls1 = bq2->scales[ib32] & 0xf;
    స్థిరరాశి uint8_t ls2 = bq2->scales[ib32] >>  4;
    పూర్ణాంకం sumi1 = 0;
    కొరకు (పూర్ణాంకం l = 0; l < 2; ++l) {
        స్థిరరాశి uint32_t * grid = (స్థిరరాశి uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        స్థిరరాశి uint32_t signs0 = dpct::vectorized_ద్విమాన<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        స్థిరరాశి uint32_t signs1 = dpct::vectorized_ద్విమాన<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        స్థిరరాశి పూర్ణాంకం grid_l = dpct::vectorized_ద్విమాన<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        స్థిరరాశి పూర్ణాంకం grid_h = dpct::vectorized_ద్విమాన<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi1 = dpct::dp4a(grid_l, *((స్థిరరాశి పూర్ణాంకం *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((స్థిరరాశి పూర్ణాంకం *)q8 + 1), sumi1);
        q8 += 8;
    }
    పూర్ణాంకం sumi2 = 0;
    కొరకు (పూర్ణాంకం l = 2; l < 4; ++l) {
        స్థిరరాశి uint32_t * grid = (స్థిరరాశి uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        స్థిరరాశి uint32_t signs0 = dpct::vectorized_ద్విమాన<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        స్థిరరాశి uint32_t signs1 = dpct::vectorized_ద్విమాన<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        స్థిరరాశి పూర్ణాంకం grid_l = dpct::vectorized_ద్విమాన<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        స్థిరరాశి పూర్ణాంకం grid_h = dpct::vectorized_ద్విమాన<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi2 = dpct::dp4a(grid_l, *((స్థిరరాశి పూర్ణాంకం *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((స్థిరరాశి పూర్ణాంకం *)q8 + 1), sumi2);
        q8 += 8;
    }
    స్థిరరాశి దశాంశం d = (దశాంశం)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    తిరిగి d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
#అయితే_ముగింపు
}

#నిర్వచించు VDR_IQ3_XXS_Q8_1_MMVQ 1

స్థిర __dpct_inline__ దశాంశం
vec_dot_iq3_xxs_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                     స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs,
                     స్థిరరాశి uint32_t *iq3xxs_grid) {
#ఒకవేళ DPCT_COMPATIBILITY_తాత్కాలిక >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq3_xxs * bq2 = (స్థిరరాశి ఖండం_iq3_xxs *) vbq;

    స్థిరరాశి పూర్ణాంకం ib32 = iqs;
    స్థిరరాశి uint8_t  * q3 = bq2->qs + 8*ib32;
    స్థిరరాశి uint16_t * gas = (స్థిరరాశి uint16_t *)(bq2->qs + QK_K/4) + 2*ib32;
    స్థిరరాశి int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = gas[0] | (gas[1] << 16);
    పూర్ణాంకం sumi = 0;
#సూచన unroll
    కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
        స్థిరరాశి uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
        స్థిరరాశి uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
        స్థిరరాశి uint32_t sign_బిట్లు = iq_sign_బిట్లు(aux32);
        స్థిరరాశి uint32_t signs0 = iq_sign_ముసుగు(sign_బిట్లు);
        స్థిరరాశి uint32_t signs1 = iq_sign_ముసుగు(sign_బిట్లు >> 4);
        స్థిరరాశి పూర్ణాంకం grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        స్థిరరాశి పూర్ణాంకం grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((స్థిరరాశి పూర్ణాంకం *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((స్థిరరాశి పూర్ణాంకం *)q8 + 1), sumi);
        q8 += 8;
        aux32 >>= 7;
    }
    స్థిరరాశి దశాంశం d = (దశాంశం)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.5f;
    తిరిగి d * sumi;
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
    తిరిగి 0.f;
#అయితే_ముగింపు
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
    తిరిగి 0.f;
#అయితే_ముగింపు
}

మూస <> నిర్మాణం reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_XXS> {
    స్థిర constexpr ggml_type gtype = GGML_TYPE_IQ3_XXS;

    __dpct_inline__ దశాంశం క్రియాకారకం()(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> ibx_స్థానభ్రంశం,
                                     స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> d_స్థానభ్రంశం, స్థిరరాశి int8_t * q8_1_quant_ptr,
                                     స్థిరరాశి sycl::half2 * q8_1_ds, స్థిరరాశి పూర్ణాంకం & iqs) {
#ఒకవేళ DPCT_COMPATIBILITY_తాత్కాలిక >= MIN_CC_DP4A && QK_K == 256
        స్థిరరాశి uint8_t * పునాది = static_cast<స్థిరరాశి uint8_t *>(vbq);
        స్థిరరాశి uint8_t * qs   = పునాది + ibx_స్థానభ్రంశం.మొదటి;
        స్థిరరాశి uint8_t * q3   = qs + 8 * iqs;
        స్థిరరాశి uint16_t * gas = reinterpret_cast<స్థిరరాశి uint16_t *>(qs + QK_K / 4) + 2 * iqs;
        స్థిరరాశి int8_t * q8    = q8_1_quant_ptr + iqs * QK8_1;
        uint32_t aux32       = gas[0] | (gas[1] << 16);

        పూర్ణాంకం sumi = 0;
#సూచన unroll
        కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
            స్థిరరాశి uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
            స్థిరరాశి uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
            స్థిరరాశి uint32_t sign_బిట్లు = iq_sign_బిట్లు(aux32);
            స్థిరరాశి uint32_t signs0 = iq_sign_ముసుగు(sign_బిట్లు);
            స్థిరరాశి uint32_t signs1 = iq_sign_ముసుగు(sign_బిట్లు >> 4);
            స్థిరరాశి పూర్ణాంకం grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            స్థిరరాశి పూర్ణాంకం grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((స్థిరరాశి పూర్ణాంకం *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((స్థిరరాశి పూర్ణాంకం *) q8 + 1), sumi);
            q8 += 8;
            aux32 >>= 7;
        }

        స్థిరరాశి ggml_half d = *reinterpret_cast<స్థిరరాశి ggml_half *>(పునాది + d_స్థానభ్రంశం.మొదటి);
        తిరిగి (దశాంశం) d * (0.5f + aux32) * q8_1_ds[iqs][0] * 0.5f * sumi;
#లేకపోతే
        GGML_వాడని(vbq);
        GGML_వాడని(ibx_స్థానభ్రంశం);
        GGML_వాడని(d_స్థానభ్రంశం);
        GGML_వాడని(q8_1_quant_ptr);
        GGML_వాడని(q8_1_ds);
        GGML_వాడని(iqs);
        GGML_ABORT("IQ3_XXS reorder vec dot requires dp4a and QK_K == 256");
#అయితే_ముగింపు
    }
};

#నిర్వచించు VDR_IQ3_S_Q8_1_MMVQ 1

స్థిర __dpct_inline__ దశాంశం
vec_dot_iq3_s_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                   స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs,
                   స్థిరరాశి uint32_t *iq3s_grid) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq3_s * bq2 = (స్థిరరాశి ఖండం_iq3_s *) vbq;

    స్థిరరాశి పూర్ణాంకం ib32 = iqs;
    స్థిరరాశి uint8_t  * qs = bq2->qs + 8*ib32;
    స్థిరరాశి int8_t   * q8 = bq8_1[ib32].qs;
    పూర్ణాంకం sumi = 0;
    కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
        స్థిరరాశి uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((bq2->qh[ib32] << (8 - 2*l)) & 256)];
        స్థిరరాశి uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((bq2->qh[ib32] << (7 - 2*l)) & 256)];
        స్థిరరాశి uint32_t signs = bq2->signs[4 * ib32 + l];
        స్థిరరాశి uint32_t signs0 = iq_sign_ముసుగు(signs);
        స్థిరరాశి uint32_t signs1 = iq_sign_ముసుగు(signs >> 4);
        స్థిరరాశి పూర్ణాంకం grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        స్థిరరాశి పూర్ణాంకం grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((స్థిరరాశి పూర్ణాంకం *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((స్థిరరాశి పూర్ణాంకం *)q8 + 1), sumi);
        q8 += 8;
    }
    స్థిరరాశి దశాంశం d =
        (దశాంశం)bq2->d *
        (1 + 2 * ((bq2->scales[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf)) *
        bq8_1[ib32].ds[0];
    తిరిగి d * sumi;
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
#అయితే_ముగింపు
}

మూస <> నిర్మాణం reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_S> {
    స్థిర constexpr ggml_type gtype = GGML_TYPE_IQ3_S;

    __dpct_inline__ దశాంశం క్రియాకారకం()(స్థిరరాశి శూన్య * __restrict__ vbq, స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> ibx_స్థానభ్రంశం,
                                     స్థిరరాశి std::pair<పూర్ణాంకం, పూర్ణాంకం> d_స్థానభ్రంశం, స్థిరరాశి int8_t * q8_1_quant_ptr,
                                     స్థిరరాశి sycl::half2 * q8_1_ds, స్థిరరాశి పూర్ణాంకం & iqs) {
#ఒకవేళ QK_K == 256
        స్థిరరాశి uint8_t * పునాది     = static_cast<స్థిరరాశి uint8_t *>(vbq);
        స్థిరరాశి uint8_t * qs       = పునాది + ibx_స్థానభ్రంశం.మొదటి + 8 * iqs;
        స్థిరరాశి uint8_t * qh       = పునాది + ibx_స్థానభ్రంశం.second;
        స్థిరరాశి uint8_t * signs    = పునాది + d_స్థానభ్రంశం.మొదటి + 4 * iqs;
        స్థిరరాశి uint8_t * metadata = పునాది + d_స్థానభ్రంశం.second;
        స్థిరరాశి ggml_half d        = *reinterpret_cast<స్థిరరాశి ggml_half *>(metadata);
        స్థిరరాశి uint8_t * scales   = metadata + పరిమాణం(ggml_half);
        స్థిరరాశి int8_t * q8        = q8_1_quant_ptr + iqs * QK8_1;

        పూర్ణాంకం sumi = 0;
#సూచన unroll
        కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
            స్థిరరాశి uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((qh[iqs] << (8 - 2*l)) & 256)];
            స్థిరరాశి uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((qh[iqs] << (7 - 2*l)) & 256)];
            స్థిరరాశి uint32_t signs0 = iq_sign_ముసుగు(signs[l]);
            స్థిరరాశి uint32_t signs1 = iq_sign_ముసుగు(signs[l] >> 4);
            స్థిరరాశి పూర్ణాంకం grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            స్థిరరాశి పూర్ణాంకం grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((స్థిరరాశి పూర్ణాంకం *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((స్థిరరాశి పూర్ణాంకం *) q8 + 1), sumi);
            q8 += 8;
        }

        స్థిరరాశి దశాంశం scale = 1 + 2 * ((scales[iqs / 2] >> (4 * (iqs % 2))) & 0xf);
        తిరిగి (దశాంశం) d * scale * q8_1_ds[iqs][0] * sumi;
#లేకపోతే
        GGML_వాడని(vbq);
        GGML_వాడని(ibx_స్థానభ్రంశం);
        GGML_వాడని(d_స్థానభ్రంశం);
        GGML_వాడని(q8_1_quant_ptr);
        GGML_వాడని(q8_1_ds);
        GGML_వాడని(iqs);
        GGML_ABORT("IQ3_S reorder vec dot not supported for QK_K != 256");
#అయితే_ముగింపు
    }
};

#నిర్వచించు VDR_IQ1_S_Q8_1_MMVQ 1

స్థిర __dpct_inline__ దశాంశం
vec_dot_iq1_s_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                   స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs,
                   స్థిరరాశి uint32_t *iq1s_grid_gpu) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq1_s * bq1 = (స్థిరరాశి ఖండం_iq1_s *) vbq;

    స్థిరరాశి పూర్ణాంకం ib32 = iqs;
    పూర్ణాంకం sumi = 0;
    స్థిరరాశి పూర్ణాంకం * q8 = (స్థిరరాశి పూర్ణాంకం *)bq8_1[ib32].qs;
    కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
        స్థిరరాశి పూర్ణాంకం * grid = (స్థిరరాశి పూర్ణాంకం *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[ib32] >> 3*l) & 7) << 8)));
        పూర్ణాంకం grid0 = grid[0] & 0x0f0f0f0f;
        పూర్ణాంకం grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi = dpct::dp4a(q8[2 * l + 1], grid1,
                          dpct::dp4a(q8[2 * l + 0], grid0, sumi));
    }

    స్థిరరాశి దశాంశం delta = bq1->qh[ib32] & 0x8000 ? -1-IQ1S_DELTA : -1+IQ1S_DELTA;
    స్థిరరాశి దశాంశం d1q = (దశాంశం)bq1->d * (2*((bq1->qh[ib32] >> 12) & 7) + 1);
    స్థిరరాశి దశాంశం d = d1q * bq8_1[ib32].ds[0];
    స్థిరరాశి దశాంశం m = d1q * bq8_1[ib32].ds[1];
    తిరిగి d * sumi + m * delta;
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
#అయితే_ముగింపు
}

#నిర్వచించు VDR_IQ1_M_Q8_1_MMVQ 1

స్థిర __dpct_inline__ దశాంశం
vec_dot_iq1_m_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                   స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {
#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq1_m * bq1 = (స్థిరరాశి ఖండం_iq1_m *) vbq;

    స్థిరరాశి పూర్ణాంకం ib32 = iqs;
    పూర్ణాంకం   sumi[2] = {0, 0};
    దశాంశం sumf[2] = {0.f, 0.f};

    స్థిరరాశి పూర్ణాంకం * q8 = (స్థిరరాశి పూర్ణాంకం *)bq8_1[ib32].qs;
    కొరకు (పూర్ణాంకం l = 0; l < 4; ++l) {
        స్థిరరాశి పూర్ణాంకం * grid = (స్థిరరాశి పూర్ణాంకం *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 7) << 8)));
        పూర్ణాంకం grid0 = grid[0] & 0x0f0f0f0f;
        పూర్ణాంకం grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi[l / 2] = dpct::dp4a(q8[2 * l + 1], grid1,
                                 dpct::dp4a(q8[2 * l + 0], grid0, sumi[l / 2]));
        స్థిరరాశి దశాంశం delta = (bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 0x08 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA;
        స్థిరరాశి పూర్ణాంకం sumy = dpct::dp4a(q8[2 * l + 1], 0x01010101,
                                    dpct::dp4a(q8[2 * l + 0], 0x01010101, 0));
        sumf[l/2] += delta*sumy;
    }

    iq1m_scale_t scale;
    స్థిరరాశి uint16_t * sc = (స్థిరరాశి uint16_t *)bq1->scales;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    స్థిరరాశి దశాంశం d = (దశాంశం)scale.f16 * bq8_1[ib32].ds[0];
    తిరిగి d * ((sumi[0] + sumf[0]) * (2*((sc[ib32/2] >> 6*(ib32%2)) & 0x7) + 1) + (sumi[1] + sumf[1]) * (2*((sc[ib32/2] >> (6*(ib32%2)+3)) & 0x7) + 1));
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
#అయితే_ముగింపు
}


#నిర్వచించు VDR_IQ4_NL_Q8_1_MMVQ 2

స్థిర __dpct_inline__ దశాంశం
vec_dot_iq4_nl_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                    స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {

    స్థిరరాశి ఖండం_iq4_nl * bq = (స్థిరరాశి ఖండం_iq4_nl *) vbq;

    స్థిరరాశి uint16_t * q4 = (స్థిరరాశి uint16_t *)bq->qs + 2*iqs;
    స్థిరరాశి int32_t  * q8 = (స్థిరరాశి int32_t  *)bq8_1->qs + iqs;

    స్థిరరాశి uint8_t * విలువలు = (స్థిరరాశి uint8_t *)kvalues_iq4nl;

    పూర్ణాంకం v1, v2;
    పూర్ణాంకం sumi1 = 0, sumi2 = 0;
    కొరకు (పూర్ణాంకం l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        స్థిరరాశి uint32_t aux = q4[2*l] | (q4[2*l+1] << 16);
        get_int_from_పట్టిక_16(aux, విలువలు, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[l + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[l + 4], sumi2);
    }

    స్థిరరాశి దశాంశం d = (దశాంశం)bq->d * bq8_1->ds[0];
    తిరిగి d * (sumi1 + sumi2);
}


#నిర్వచించు VDR_IQ4_XS_Q8_1_MMVQ 1

స్థిర __dpct_inline__ దశాంశం
vec_dot_iq4_xs_q8_1(స్థిరరాశి శూన్య *__restrict__ vbq,
                    స్థిరరాశి ఖండం_q8_1 *__restrict__ bq8_1, స్థిరరాశి పూర్ణాంకం &iqs) {

#ఒకవేళ QK_K == 256
    స్థిరరాశి ఖండం_iq4_xs * bq4 = (స్థిరరాశి ఖండం_iq4_xs *) vbq;
    స్థిరరాశి uint8_t * విలువలు = (స్థిరరాశి uint8_t *)kvalues_iq4nl;

    // iqs is 0...7
    స్థిరరాశి పూర్ణాంకం ib32 = iqs;
    స్థిరరాశి int32_t  * q8 = (స్థిరరాశి పూర్ణాంకం *)bq8_1[ib32].qs;
    స్థిరరాశి uint32_t * q4 = (స్థిరరాశి uint32_t *)bq4->qs + 4*ib32;
    స్థిరరాశి int8_t ls = ((bq4->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((bq4->scales_h >> 2*ib32) & 3) << 4);
    స్థిరరాశి దశాంశం d = (దశాంశం)bq4->d * (ls - 32) * bq8_1[ib32].ds[0];
    పూర్ణాంకం v1, v2;
    పూర్ణాంకం sumi1 = 0, sumi2 = 0;
    కొరకు (పూర్ణాంకం j = 0; j < 4; ++j) {
        get_int_from_పట్టిక_16(q4[j], విలువలు, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[j + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[j + 4], sumi2);
    }
    తిరిగి d * (sumi1 + sumi2);
#లేకపోతే
    ప్రతిజ్ఞా(అబద్ధం);
#అయితే_ముగింపు
}

#అయితే_ముగింపు // GGML_SYCL_VECDOTQ_HPP
