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

#ifndef GGML_SYCL_VECDOTQ_HPP
#define GGML_SYCL_VECDOTQ_HPP

#ಅನ್ತರ್ಭಾವ "dpct/helper.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "type.hpp"
#ಅನ್ತರ್ಭಾವ "quants.hpp"

ಪ್ರಕಾರನಿರ್ದೇಶ ಭಿನ್ನ (*vec_dot_q_sycl_t)(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ block_q8_1 * __restrict__ bq8_1,
                                  ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs);

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಙ್ಕ get_int_b1(ಧ್ರುವ ಶೂನ್ಯ * x, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & i32) {
    ಧ್ರುವ uint8_t * x8 = (ಧ್ರುವ uint8_t *) x;

    ಪೂರ್ಣಾಙ್ಕ x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    ಪ್ರತ್ಯಾಗಮ x32;
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಙ್ಕ get_int_b2(ಧ್ರುವ ಶೂನ್ಯ * x, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & i32) {
    ಧ್ರುವ uint16_t * x16 = (ಧ್ರುವ uint16_t *) x; // assume at least 2 byte alignment

    ಪೂರ್ಣಾಙ್ಕ x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    ಪ್ರತ್ಯಾಗಮ x32;
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಙ್ಕ get_int_b4(ಧ್ರುವ ಶೂನ್ಯ * x, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & i32) {
    ಪ್ರತ್ಯಾಗಮ ((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *) x)[i32]; // assume at least 4 byte alignment
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಙ್ಕ get_int_from_int8(ಧ್ರುವ int8_t* x8, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ& i32) {
  ಧ್ರುವ uint16_t* x16 =
      (ಧ್ರುವ uint16_t*)(x8 + ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * i32); // assume at least 2 byte
                                                 // alignment

  ಪೂರ್ಣಾಙ್ಕ x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  ಪ್ರತ್ಯಾಗಮ x32;
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಙ್ಕ get_int_from_uint8(
    ಧ್ರುವ uint8_t* x8,
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ& i32) {
  ಧ್ರುವ uint16_t* x16 =
      (ಧ್ರುವ uint16_t*)(x8 + ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * i32); // assume at least 2 byte
                                                 // alignment

  ಪೂರ್ಣಾಙ್ಕ x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  ಪ್ರತ್ಯಾಗಮ x32;
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಙ್ಕ get_int_from_int8_aligned(
    ಧ್ರುವ int8_t* x8,
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ& i32) {
  ಪ್ರತ್ಯಾಗಮ *(
      (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ*)(x8 + ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * i32)); // assume at least 4 byte alignment
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಙ್ಕ get_int_from_uint8_aligned(
    ಧ್ರುವ uint8_t* x8,
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ& i32) {
  ಪ್ರತ್ಯಾಗಮ *(
      (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ*)(x8 + ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * i32)); // assume at least 4 byte alignment
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಙ್ಕ byte_sub_4(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ a, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ b) {
  ಧ್ರುವ uint32_t ua = static_cast<uint32_t>(a);
  ಧ್ರುವ uint32_t ub = static_cast<uint32_t>(b);
  ಪ್ರತ್ಯಾಗಮ static_cast<ಪೂರ್ಣಾಙ್ಕ>(((ua | 0x80808080u) - ub) ^ 0x80808080u);
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vl, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vh, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ u0, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ u1, ಧ್ರುವ int8_t sc0,
    ಧ್ರುವ int8_t sc1, ಧ್ರುವ ಭಿನ್ನ d, ಧ್ರುವ ಭಿನ್ನ d80, ಧ್ರುವ ಭಿನ್ನ d81) {
    static_assert(QR6_K == 2, "q6_K MMVQ scalar fast path assumes QR6_K == 2");

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vil0 = (vl >> 0) & 0x0F0F0F0F;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vih0 = ((vh >> 0) << 4) & 0x30303030;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vi0 = byte_sub_4(vil0 | vih0, 0x20202020);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vil1 = (vl >> 4) & 0x0F0F0F0F;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vih1 = ((vh >> 4) << 4) & 0x30303030;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vi1 = byte_sub_4(vil1 | vih1, 0x20202020);

    ಧ್ರುವ ಭಿನ್ನ sumf =
        d80 * (dpct::dp4a(vi0, u0, 0) * sc0) +
        d81 * (dpct::dp4a(vi1, u1, 0) * sc1);

    ಪ್ರತ್ಯಾಗಮ d * sumf;
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ get_int_from_table_16(ಧ್ರುವ uint32_t &q4,
                                                  ಧ್ರುವ uint8_t *ಮೌಲ್ಯಗಳು,
                                                  ಪೂರ್ಣಾಙ್ಕ &val1, ಪೂರ್ಣಾಙ್ಕ &val2) {

    uint32_t aux32; ಧ್ರುವ uint8_t * q8 = (ಧ್ರುವ uint8_t *)&aux32;
    aux32 = q4 & 0x0f0f0f0f;
    uint16_t v1 = ಮೌಲ್ಯಗಳು[q8[0]] | (ಮೌಲ್ಯಗಳು[q8[1]] << 8);
    uint16_t v2 = ಮೌಲ್ಯಗಳು[q8[2]] | (ಮೌಲ್ಯಗಳು[q8[3]] << 8);
    val1 = v1 | (v2 << 16);
    aux32 = (q4 >> 4) & 0x0f0f0f0f;
    v1 = ಮೌಲ್ಯಗಳು[q8[0]] | (ಮೌಲ್ಯಗಳು[q8[1]] << 8);
    v2 = ಮೌಲ್ಯಗಳು[q8[2]] | (ಮೌಲ್ಯಗಳು[q8[3]] << 8);
    val2 = v1 | (v2 << 16);
}

ಸ್ಥಿರ __dpct_inline__ sycl::int2 get_int_from_table_16(
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ& q4, ಧ್ರುವ int8_t* table) {
  ಧ್ರುವ uint32_t* table32 = (ಧ್ರುವ uint32_t*)table;
  uint32_t tmp[2];
  ಧ್ರುವ uint32_t ಕೆಳಗಿನ_ಮೇಲಿನ_selection_indices =
      (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
  ಪ್ರತಿ (uint32_t i = 0; i < 2; ++i) {
    ಧ್ರುವ uint32_t shift = 16 * i;

    ಧ್ರುವ uint32_t ಕೆಳಗಿನ =
        dpct::byte_level_permute(table32[0], table32[1], q4 >> shift);
    ಧ್ರುವ uint32_t ಮೇಲಿನ =
        dpct::byte_level_permute(table32[2], table32[3], q4 >> shift);
    tmp[i] = dpct::byte_level_permute(
        ಕೆಳಗಿನ, ಮೇಲಿನ, ಕೆಳಗಿನ_ಮೇಲಿನ_selection_indices >> shift);
  }
  ಪ್ರತ್ಯಾಗಮ sycl::int2(
      dpct::byte_level_permute(tmp[0], tmp[1], 0x6420),
      dpct::byte_level_permute(tmp[0], tmp[1], 0x7531));
}

// Four E2M1 codes (one per byte, bits 0..3) to their kvalues_mxfp4 int8 values. SWAR arithmetic
// replaces get_int_from_table_16 for MXFP4: dpct::byte_level_permute is emulated with 64-bit shifts,
// eight per int, which made the MXFP4 GEMV compute-bound on Intel GPUs.
// Magnitudes 0,1,2,3,4,6,8,12 = m + [m>=5] + [m>=6] + 3*[m>=7]; each byte stays below 256, so the
// byte-wise adds never carry. -0 (code 8) is left as 0 so the two's-complement +1 cannot carry either.
ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಙ್ಕ mxfp4_codes_to_int8(ಧ್ರುವ uint32_t x) {
    ಧ್ರುವ uint32_t m   = x & 0x07070707u;
    ಧ್ರುವ uint32_t ge5 = ((m + 0x03030303u) >> 3) & 0x01010101u;
    ಧ್ರುವ uint32_t ge6 = ((m + 0x02020202u) >> 3) & 0x01010101u;
    ಧ್ರುವ uint32_t ge7 = ((m + 0x01010101u) >> 3) & 0x01010101u;
    ಧ್ರುವ uint32_t mag = m + ge5 + ge6 + 3u * ge7;
    ಧ್ರುವ uint32_t nz  = ((mag + 0x7f7f7f7fu) >> 7) & 0x01010101u;
    ಧ್ರುವ uint32_t neg = (x >> 3) & nz & 0x01010101u;
    ಪ್ರತ್ಯಾಗಮ (ಪೂರ್ಣಾಙ್ಕ) ((mag ^ (neg * 0xffu)) + neg);
}

// Same result as get_int_from_table_16(q4, kvalues_mxfp4): x = low nibbles, y = high nibbles.
ಸ್ಥಿರ __dpct_inline__ sycl::int2 get_int_from_mxfp4(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ q4) {
    ಪ್ರತ್ಯಾಗಮ sycl::int2(mxfp4_codes_to_int8((uint32_t) q4 & 0x0f0f0f0fu),
                      mxfp4_codes_to_int8(((uint32_t) q4 >> 4) & 0x0f0f0f0fu));
}

#define VDR_Q2_K_Q8_1_MMVQ 1

// contiguous v/x values
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_q2_K_q8_1_impl_mmvq(
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &v, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *__restrict__ u, ಧ್ರುವ uint8_t *__restrict__ scales,
    ಧ್ರುವ sycl::half2 &dm2, ಧ್ರುವ ಭಿನ್ನ *__restrict__ d8) {

    ಭಿನ್ನ sumf_d = 0.0f;
    ಭಿನ್ನ sumf_m = 0.0f;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < QR2_K; ++i) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sc = scales[2*i];

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vi = (v >> (2*i)) & 0x03030303;

        sumf_d +=
            d8[i] * (dpct::dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        ಪೂರ್ಣಾಙ್ಕ m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] *
                  dpct::dp4a(
                      m, u[i],
                      0); // multiply constant q2_K part with sum of q8_1 values
    }

    ಧ್ರುವ sycl::float2 dm2f =
        dm2.convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();

    ಪ್ರತ್ಯಾಗಮ dm2f.x() * sumf_d - dm2f.y() * sumf_m;
}


#define VDR_Q3_K_Q8_1_MMVQ 1

// contiguous v/x values
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_q3_K_q8_1_impl_mmvq(
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &vl, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &vh, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *__restrict__ u,
    ಧ್ರುವ uint8_t *__restrict__ scales, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &scale_offset,
    ಧ್ರುವ ಭಿನ್ನ &d3, ಧ್ರುವ ಭಿನ್ನ *__restrict__ d8) {

    ಭಿನ್ನ sumf = 0.0f;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < QR3_K; ++i) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ isc = scale_offset + 2*i;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ isc_ಕೆಳಗಿನ = isc % (QK_K/32);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sc_shift_ಕೆಳಗಿನ = 4 * (isc / (QK_K/32));
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sc_ಕೆಳಗಿನ  = (scales[isc_ಕೆಳಗಿನ] >> sc_shift_ಕೆಳಗಿನ) & 0xF;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ isc_ಮೇಲಿನ = isc % (QK_K/64);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sc_shift_ಮೇಲಿನ = 2 * (isc / (QK_K/64));
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sc_ಮೇಲಿನ = ((scales[(QK_K/32) + isc_ಮೇಲಿನ] >> sc_shift_ಮೇಲಿನ) & 3) << 4;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sc = (sc_ಕೆಳಗಿನ | sc_ಮೇಲಿನ) - 32;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vil = (vl >> (2*i)) & 0x03030303;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vih = ((vh >> i) << 2) & 0x04040404;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vi =
            dpct::vectorized_ದ್ವಿಮಾನ<sycl::char4>(vil, vih, dpct::sub_sat());

        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    ಪ್ರತ್ಯಾಗಮ d3 * sumf;
}

#define VDR_Q4_K_Q8_1_MMVQ 2

// contiguous v/x values
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_q4_K_q8_1_impl_vmmq(
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *__restrict__ v, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *__restrict__ u,
    ಧ್ರುವ uint8_t *__restrict__ sc, ಧ್ರುವ uint8_t *__restrict__ m,
    ಧ್ರುವ sycl::half2 &dm4, ಧ್ರುವ ಭಿನ್ನ *__restrict__ d8) {

    ಭಿನ್ನ sumf_d = 0.0f;
    ಭಿನ್ನ sumf_m = 0.0f;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < QR4_K; ++i) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dot1 =
            dpct::dp4a(v1i, u[2 * i + 1],
                       dpct::dp4a(v0i, u[2 * i + 0], 0)); // SIMD dot product
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 1],
                       dpct::dp4a(0x01010101, u[2 * i + 0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    ಧ್ರುವ sycl::float2 dm4f =
        dm4.convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();

    ಪ್ರತ್ಯಾಗಮ dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


#define VDR_Q5_K_Q8_1_MMVQ 2

// contiguous v/x values
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_q5_K_q8_1_impl_vmmq(
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *__restrict__ vl, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *__restrict__ vh,
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *__restrict__ u, ಧ್ರುವ uint8_t *__restrict__ sc,
    ಧ್ರುವ uint8_t *__restrict__ m, ಧ್ರುವ sycl::half2 &dm5,
    ಧ್ರುವ ಭಿನ್ನ *__restrict__ d8) {

    ಭಿನ್ನ sumf_d = 0.0f;
    ಭಿನ್ನ sumf_m = 0.0f;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < QR5_K; ++i) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ v0i = vl0i | vh0i;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ v1i = vl1i | vh1i;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dot1 =
            dpct::dp4a(v0i, u[2 * i + 0],
                       dpct::dp4a(v1i, u[2 * i + 1], 0)); // SIMD dot product
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 0],
                       dpct::dp4a(0x01010101, u[2 * i + 1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    ಧ್ರುವ sycl::float2 dm5f =
        dm5.convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();

    ಪ್ರತ್ಯಾಗಮ dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}


#define VDR_Q6_K_Q8_1_MMVQ 1

// contiguous v/x values
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_q6_K_q8_1_impl_mmvq(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &vl, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &vh,
                            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *__restrict__ u,
                            ಧ್ರುವ int8_t *__restrict__ scales, ಧ್ರುವ ಭಿನ್ನ &d,
                            ಧ್ರುವ ಭಿನ್ನ *__restrict__ d8) {
    ಪ್ರತ್ಯಾಗಮ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
}

#define VDR_Q1_0_Q8_1_MMVQ 1
#define VDR_Q1_0_Q8_1_MMQ  4

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_q1_0_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                  ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {

    ಧ್ರುವ block_q1_0 * bq1_0 = (ಧ್ರುವ block_q1_0 *) vbq;

    ಧ್ರುವ block_q8_1 * bq8_1_chunk = bq8_1 + iqs;
    ಧ್ರುವ ಭಿನ್ನ        d1          = bq1_0->d;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ          v           = get_int_from_uint8_aligned(bq1_0->qs, iqs);

    ಪೂರ್ಣಾಙ್ಕ vi_bytes[8];
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; ++j) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ shift = j * 4;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bits4 = (v >> shift) & 0x0F;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ b0    = (bits4 & 0x01) ? 1 : -1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ b1    = (bits4 & 0x02) ? 1 : -1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ b2    = (bits4 & 0x04) ? 1 : -1;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ b3    = (bits4 & 0x08) ? 1 : -1;
        vi_bytes[j]     = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }

    ಪೂರ್ಣಾಙ್ಕ sumi = 0;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; ++j) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ u = get_int_from_int8_aligned(bq8_1_chunk->qs, j);
        sumi        = ggml_sycl_dp4a(vi_bytes[j], u, sumi);
    }

    ಪ್ರತ್ಯಾಗಮ d1 * bq8_1_chunk->ds[0] * sumi;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

template <ggml_type T> ಸಂರಚನಾ reorder_vec_dot_q_sycl {
    static_assert(T != T, "ggml_type for reorder vecdot not implemented");
};

// For some types the weight side of the dot product does not depend on the destination column, so a
// multi-column mul_mat_vec can unpack it once per block instead of once per column. Such a type adds
// load() and dot() next to operator() and opts in here. See reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>.
template <ggml_type T> ಸಂರಚನಾ reorder_vec_dot_shared_weights {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ value = ಅಸತ್ಯ;
};

template <> ಸಂರಚನಾ reorder_vec_dot_shared_weights<GGML_TYPE_Q4_K> {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ value = ಸತ್ಯ;
};

template <> ಸಂರಚನಾ reorder_vec_dot_shared_weights<GGML_TYPE_Q5_K> {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ value = ಸತ್ಯ;
};

template <ggml_type T> ಸಂರಚನಾ reorder_vec_dot_shared_activations {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ value = ಅಸತ್ಯ;
};

template <> ಸಂರಚನಾ reorder_vec_dot_shared_activations<GGML_TYPE_Q4_K> {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ value = ಸತ್ಯ;
};

template <> ಸಂರಚನಾ reorder_vec_dot_shared_activations<GGML_TYPE_Q5_K> {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ value = ಸತ್ಯ;
};

template <> ಸಂರಚನಾ reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q4_0;

    using q4_0_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_0>;
    using q4_0_traits = typename q4_0_block::traits;

    __dpct_inline__ ಭಿನ್ನ vec_dot_q4_0_q8_1_impl(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * v, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * u, ಧ್ರುವ ಭಿನ್ನ & d4, ಧ್ರುವ sycl::half2 & ds8) {
        ಪೂರ್ಣಾಙ್ಕ sumi = 0;

#pragma unroll
        ಪ್ರತಿ (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vi0 = (v[i] >> 0) & 0x0F0F0F0F;
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vi1 = (v[i] >> 4) & 0x0F0F0F0F;

            // SIMD dot product of quantized values
            sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
            sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
        }

        ಧ್ರುವ sycl::float2 ds8f = ds8.convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();

        // second part effectively subtracts 8 from each quant value
        ಪ್ರತ್ಯಾಗಮ d4 * (sumi * ds8f.x() - (8 * q4_0_traits::vdr_mmvq / q4_0_traits::qi) * ds8f.y());
    }

    __dpct_inline__ ಭಿನ್ನ operator()(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> ibx_offset,
                                     ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> d_offset, ಧ್ರುವ int8_t * q8_1_quant_ptr,
                                     ಧ್ರುವ sycl::half2 * q8_1_ds, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
        ಧ್ರುವ uint8_t * bq4_0 = static_cast<ಧ್ರುವ uint8_t *>(vbq) + ibx_offset.first;
        ಧ್ರುವ ggml_half d = *(reinterpret_cast<ಧ್ರುವ ggml_half *>(static_cast<ಧ್ರುವ uint8_t *>(vbq) + d_offset.first));
        ಪೂರ್ಣಾಙ್ಕ             v[q4_0_traits::vdr_mmvq];
        ಪೂರ್ಣಾಙ್ಕ             u[2 * q4_0_traits::vdr_mmvq];


#pragma unroll
        ಪ್ರತಿ (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            v[i]         = get_int_from_uint8(bq4_0, iqs + i);
            u[2 * i + 0] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
            u[2 * i + 1] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i + q4_0_traits::qi);
        }

        ಪ್ರತ್ಯಾಗಮ vec_dot_q4_0_q8_1_impl(v, u, d, *q8_1_ds);
    }
};

// Load four contiguous dwords per operand instead of loading each value separately.
ಸಂರಚನಾ reorder_vec_dot_q8_0_wide {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    using q8_0_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_0>;
    using q8_0_traits = typename q8_0_block::traits;

    __dpct_inline__ ಭಿನ್ನ operator()(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> ibx_offset,
                                     ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> d_offset, ಧ್ರುವ int8_t * q8_1_quant_ptr,
                                     ಧ್ರುವ sycl::half2 * q8_1_ds, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
        static_assert(q8_0_traits::vdr_mmvq == 4, "the wide load moves exactly four dwords");

        ಧ್ರುವ uint8_t * base = static_cast<ಧ್ರುವ uint8_t *>(vbq);
        ಧ್ರುವ int8_t *  qs   = reinterpret_cast<ಧ್ರುವ int8_t *>(base + ibx_offset.first);
        ಧ್ರುವ ggml_half d    = *reinterpret_cast<ಧ್ರುವ ggml_half *>(base + d_offset.first);

        ಧ್ರುವ sycl::int4 v = *reinterpret_cast<ಧ್ರುವ sycl::int4 *>(qs + ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * iqs);
        ಧ್ರುವ sycl::int4 u = *reinterpret_cast<ಧ್ರುವ sycl::int4 *>(q8_1_quant_ptr + ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * iqs);

        ಪೂರ್ಣಾಙ್ಕ sumi = 0;
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < 4; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        ಧ್ರುವ sycl::half2 ds_ಮೌಲ್ಯಗಳು = *q8_1_ds;
        ಪ್ರತ್ಯಾಗಮ static_cast<ಭಿನ್ನ>(d) * static_cast<ಭಿನ್ನ>(ds_ಮೌಲ್ಯಗಳು[0]) * sumi;
    }
};

template <> ಸಂರಚನಾ reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    using q8_0_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_0>;
    using q8_0_traits = typename q8_0_block::traits;

    __dpct_inline__ ಭಿನ್ನ operator()(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> ibx_offset,
                                     ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> d_offset, ಧ್ರುವ int8_t * q8_1_quant_ptr,
                                     ಧ್ರುವ sycl::half2 * q8_1_ds, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs,
                                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vdr = q8_0_traits::vdr_mmvq) {
        ಧ್ರುವ uint8_t * base = static_cast<ಧ್ರುವ uint8_t *>(vbq);
        ಧ್ರುವ int8_t *  qs   = reinterpret_cast<ಧ್ರುವ int8_t *>(base + ibx_offset.first);
        ಧ್ರುವ ggml_half  d   = *reinterpret_cast<ಧ್ರುವ ggml_half *>(base + d_offset.first);

        ಪೂರ್ಣಾಙ್ಕ v[q8_0_traits::vdr_mmvq];
        ಪೂರ್ಣಾಙ್ಕ u[q8_0_traits::vdr_mmvq];

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < vdr; ++i) {
            v[i] = get_int_from_int8(qs, iqs + i);
            u[i] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
        }

        ಪೂರ್ಣಾಙ್ಕ sumi = 0;
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < vdr; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        ಧ್ರುವ sycl::half2 ds_ಮೌಲ್ಯಗಳು = *q8_1_ds;
        ಪ್ರತ್ಯಾಗಮ static_cast<ಭಿನ್ನ>(d) * static_cast<ಭಿನ್ನ>(ds_ಮೌಲ್ಯಗಳು[0]) * sumi;
    }
};

template <> ಸಂರಚನಾ reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q2_K;

    using q2_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q2_K>;
    using q2_k_traits = typename q2_k_block::traits;

    __dpct_inline__ ಭಿನ್ನ operator()(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> ibx_offset,
                                     ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> d_offset, ಧ್ರುವ int8_t * q8_1_quant_ptr,
                                     ಧ್ರುವ sycl::half2 * q8_1_ds, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
        ಧ್ರುವ uint8_t *    base   = static_cast<ಧ್ರುವ uint8_t *>(vbq);
        ಧ್ರುವ uint8_t *    qs     = base + ibx_offset.first;
        ಧ್ರುವ uint8_t *    scales = base + d_offset.first;
        ಧ್ರುವ ggml_half2 * dm     = reinterpret_cast<ಧ್ರುವ ggml_half2 *>(base + d_offset.second);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bq8_offset   = QR2_K * (iqs / QI8_1);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ v = get_int_from_uint8_aligned(qs, iqs);

        ಪೂರ್ಣಾಙ್ಕ   u[QR2_K];
        ಭಿನ್ನ d8[QR2_K];

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < QR2_K; ++i) {
            ಧ್ರುವ int8_t * quant_base_ptr = q8_1_quant_ptr + (bq8_offset + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_base_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_offset + i))[0];
        }

        ಪ್ರತ್ಯಾಗಮ vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales + scale_offset, *dm, d8);
    }
};

template <> ಸಂರಚನಾ reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q3_K;

    using q3_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q3_K>;
    using q3_k_traits = typename q3_k_block::traits;

    __dpct_inline__ ಭಿನ್ನ operator()(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> ibx_offset,
                                     ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> d_offset, ಧ್ರುವ int8_t * q8_1_quant_ptr,
                                     ಧ್ರುವ sycl::half2 * q8_1_ds, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
        ಧ್ರುವ uint8_t *  base   = static_cast<ಧ್ರುವ uint8_t *>(vbq);
        ಧ್ರುವ uint8_t *  qs     = base + ibx_offset.first;
        ಧ್ರುವ uint8_t *  hmask  = base + ibx_offset.second;
        ಧ್ರುವ uint8_t *  scales = base + d_offset.first;
        ಧ್ರುವ ggml_half  d      = *reinterpret_cast<ಧ್ರುವ ggml_half *>(base + d_offset.second);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bq8_offset   = QR3_K * (iqs / (QI3_K / 2));
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vl = get_int_from_uint8(qs, iqs);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vh = ~get_int_from_uint8(hmask, iqs % (QI3_K / 2)) >> bq8_offset;

        ಪೂರ್ಣಾಙ್ಕ   u[QR3_K];
        ಭಿನ್ನ d8[QR3_K];

#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < QR3_K; ++i) {
            ಧ್ರುವ int8_t * quant_base_ptr = q8_1_quant_ptr + (bq8_offset + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_base_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_offset + i))[0];
        }

        ಪ್ರತ್ಯಾಗಮ vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, scales, scale_offset, static_cast<ಭಿನ್ನ>(d), d8);
    }
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ vec_dot_q4_K_q8_1_common(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * __restrict__ q4, ಧ್ರುವ uint16_t * __restrict__ scales,
                                             ಧ್ರುವ ggml_half2 & dm, ಧ್ರುವ block_q8_1 * __restrict__ bq8_1,
                                             ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &        iqs) {
    ಪೂರ್ಣಾಙ್ಕ   v[2];
    ಪೂರ್ಣಾಙ್ಕ   u[2 * QR4_K];
    ಭಿನ್ನ d8[QR4_K];

    v[0] = q4[0];
    v[1] = q4[4];

    uint16_t  aux[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
    ಯದಿ (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } ಅನ್ಯಥಾ {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }

    ಧ್ರುವ uint8_t * sc = (ಧ್ರುವ uint8_t *) aux;
    ಧ್ರುವ uint8_t * m  = sc + 2;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < QR4_K; ++i) {
        ಧ್ರುವ block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i]                   = bq8i->ds[0];

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * q8 = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0]   = q8[0];
        u[2 * i + 1]   = q8[4];
    }

    ಪ್ರತ್ಯಾಗಮ vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
}

template <> ಸಂರಚನಾ reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q4_K;

    using q4_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;
    using q4_k_traits = typename q4_k_block::traits;

    ಸಂರಚನಾ weights {
        ಪೂರ್ಣಾಙ್ಕ        v[2];
        uint16_t   aux[2];
        ggml_half2 dm;
        ಪೂರ್ಣಾಙ್ಕ        bq8_offset;
    };

    ಸಂರಚನಾ activations {
        ಪೂರ್ಣಾಙ್ಕ   u[2 * QR4_K];
        ಭಿನ್ನ d8[QR4_K];
    };

    __dpct_inline__ ಸ್ಥಿರ weights load(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> ibx_offset,
                                        ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> d_offset, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
        ಧ್ರುವ uint8_t *    base = static_cast<ಧ್ರುವ uint8_t *>(vbq);
        ಧ್ರುವ uint8_t *    qs   = base + ibx_offset.first;
        ಧ್ರುವ uint8_t *    scs  = base + d_offset.first;
        ಧ್ರುವ ggml_half2 * dms  = reinterpret_cast<ಧ್ರುವ ggml_half2 *>(base + d_offset.second);

        weights w;
        w.bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *      q4     = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *) (qs + 16 * w.bq8_offset + 4 * ((iqs / 2) % 4));
        ಧ್ರುವ uint16_t * scales = (ಧ್ರುವ uint16_t *) scs;

        w.v[0] = q4[0];
        w.v[1] = q4[4];

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        ಯದಿ (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } ಅನ್ಯಥಾ {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        ಪ್ರತ್ಯಾಗಮ w;
    }

    __dpct_inline__ ಸ್ಥಿರ activations load_activations(ಧ್ರುವ int8_t * q8_1_quant_ptr,
                                                        ಧ್ರುವ sycl::half2 * q8_1_ds, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
        activations a;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < QR4_K; ++i) {
            ಧ್ರುವ int8_t * quant_base_ptr = q8_1_quant_ptr + (bq8_offset + i) * QK8_1;
            sycl::half2    ds_ಮೌಲ್ಯಗಳು      = *(q8_1_ds + bq8_offset + i);

            a.d8[i] = ds_ಮೌಲ್ಯಗಳು[0];

            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * q8 = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *) quant_base_ptr + ((iqs / 2) % 4);
            a.u[2 * i + 0] = q8[0];
            a.u[2 * i + 1] = q8[4];
        }

        ಪ್ರತ್ಯಾಗಮ a;
    }

    __dpct_inline__ ಸ್ಥಿರ ಭಿನ್ನ apply(ಧ್ರುವ weights & w, ಧ್ರುವ activations & a) {
        ಧ್ರುವ uint8_t * sc = (ಧ್ರುವ uint8_t *) w.aux;
        ಧ್ರುವ uint8_t * m  = sc + 2;

        ಪ್ರತ್ಯಾಗಮ vec_dot_q4_K_q8_1_impl_vmmq(w.v, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ ಸ್ಥಿರ ಭಿನ್ನ dot(ಧ್ರುವ weights & w, ಧ್ರುವ int8_t * q8_1_quant_ptr,
                                     ಧ್ರುವ sycl::half2 * q8_1_ds, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
        ಧ್ರುವ auto a = load_activations(q8_1_quant_ptr, q8_1_ds, iqs);

        ಪ್ರತ್ಯಾಗಮ apply(w, a);
    }

    __dpct_inline__ ಭಿನ್ನ operator()(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> ibx_offset,
                                     ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> d_offset, ಧ್ರುವ int8_t * q8_1_quant_ptr,
                                     ಧ್ರುವ sycl::half2 * q8_1_ds, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
        ಪ್ರತ್ಯಾಗಮ dot(load(vbq, ibx_offset, d_offset, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

template <> ಸಂರಚನಾ reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q5_K;

    using q5_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q5_K>;
    using q5_k_traits = typename q5_k_block::traits;

    ಸಂರಚನಾ weights {
        ಪೂರ್ಣಾಙ್ಕ        vl[2];
        ಪೂರ್ಣಾಙ್ಕ        vh[2];
        uint16_t   aux[2];
        ggml_half2 dm;
    };

    // same activation layout as Q4_K
    static_assert(QR5_K == QR4_K);
    using activations = reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::activations;

    __dpct_inline__ ಸ್ಥಿರ weights load(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> ibx_offset,
                                        ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> d_offset, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
        ಧ್ರುವ uint8_t *    base    = static_cast<ಧ್ರುವ uint8_t *>(vbq);
        ಧ್ರುವ uint8_t *    qs      = base + ibx_offset.first;   // low 4 bits
        ಧ್ರುವ uint8_t *    qh_base = base + ibx_offset.second;  // high bit
        ಧ್ರುವ uint8_t *    scs     = base + d_offset.first;
        ಧ್ರುವ ggml_half2 * dms     = reinterpret_cast<ಧ್ರುವ ggml_half2 *>(base + d_offset.second);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ        bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *      ql_ptr     = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *) (qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *      qh_ptr     = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *) (qh_base + 4 * ((iqs / 2) % 4));
        ಧ್ರುವ uint16_t * scales     = (ಧ್ರುವ uint16_t *) scs;

        weights w;
        w.vl[0] = ql_ptr[0];
        w.vl[1] = ql_ptr[4];

        w.vh[0] = qh_ptr[0] >> bq8_offset;
        w.vh[1] = qh_ptr[4] >> bq8_offset;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = (QR5_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        ಯದಿ (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } ಅನ್ಯಥಾ {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        ಪ್ರತ್ಯಾಗಮ w;
    }

    __dpct_inline__ ಸ್ಥಿರ activations load_activations(ಧ್ರುವ int8_t * q8_1_quant_ptr,
                                                        ಧ್ರುವ sycl::half2 * q8_1_ds, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
        ಪ್ರತ್ಯಾಗಮ reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::load_activations(q8_1_quant_ptr, q8_1_ds, iqs);
    }

    __dpct_inline__ ಸ್ಥಿರ ಭಿನ್ನ apply(ಧ್ರುವ weights & w, ಧ್ರುವ activations & a) {
        ಧ್ರುವ uint8_t * sc = (ಧ್ರುವ uint8_t *) w.aux;
        ಧ್ರುವ uint8_t * m  = sc + 2;

        ಪ್ರತ್ಯಾಗಮ vec_dot_q5_K_q8_1_impl_vmmq(w.vl, w.vh, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ ಸ್ಥಿರ ಭಿನ್ನ dot(ಧ್ರುವ weights & w, ಧ್ರುವ int8_t * q8_1_quant_ptr,
                                     ಧ್ರುವ sycl::half2 * q8_1_ds, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
        ಪ್ರತ್ಯಾಗಮ apply(w, load_activations(q8_1_quant_ptr, q8_1_ds, iqs));
    }

    __dpct_inline__ ಭಿನ್ನ operator()(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> ibx_offset,
                                     ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> d_offset, ಧ್ರುವ int8_t * q8_1_quant_ptr,
                                     ಧ್ರುವ sycl::half2 * q8_1_ds, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
        ಪ್ರತ್ಯಾಗಮ dot(load(vbq, ibx_offset, d_offset, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

template <> ಸಂರಚನಾ reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q6_K;

    using q6_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q6_K>;
    using q6_k_traits = typename q6_k_block::traits;

    __dpct_inline__ ಭಿನ್ನ vec_dot_q6_K_q8_1_impl_mmvq(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vl, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vh, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * __restrict__ u,
                                                      ಧ್ರುವ int8_t * __restrict__ scales, ಧ್ರುವ ಭಿನ್ನ d,
                                                      ಧ್ರುವ ಭಿನ್ನ * __restrict__ d8) {
        ಪ್ರತ್ಯಾಗಮ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
    }

    __dpct_inline__ ಭಿನ್ನ operator()(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> ibx_offset,
                     ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> d_offset, ಧ್ರುವ int8_t * q8_1_quant_ptr, ಧ್ರುವ sycl::half2 * q8_1_ds,
                     ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ iqs) {
        ಧ್ರುವ uint8_t *   base   = static_cast<ಧ್ರುವ uint8_t *>(vbq);
        ಧ್ರುವ uint8_t *   ql     = base + ibx_offset.first;
        ಧ್ರುವ uint8_t *   qh     = base + ibx_offset.second;
        ಧ್ರುವ int8_t *    scales = reinterpret_cast<ಧ್ರುವ int8_t *>(base + d_offset.first);
        ಧ್ರುವ ggml_half * d      = (ಧ್ರುವ ggml_half *) (base + d_offset.second);

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bq8_offset   = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vh_shift     = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vl = get_int_from_uint8(ql, iqs);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vh = get_int_from_uint8(qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;

        ಧ್ರುವ int8_t * scs = scales + scale_offset;

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ u0 = get_int_from_int8_aligned(
            q8_1_quant_ptr + bq8_offset * QK8_1, iqs % QI8_1);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ u1 = get_int_from_int8_aligned(
            q8_1_quant_ptr + (bq8_offset + 2) * QK8_1, iqs % QI8_1);
        ಧ್ರುವ ಭಿನ್ನ d80 = (*(q8_1_ds + bq8_offset + 0))[0];
        ಧ್ರುವ ಭಿನ್ನ d81 = (*(q8_1_ds + bq8_offset + 2))[0];

        ಪ್ರತ್ಯಾಗಮ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u0, u1, scs[0], scs[4], *d, d80, d81);
    }
};

template <> ಸಂರಚನಾ reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_MXFP4;

    using mxfp4_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_MXFP4>;
    using mxfp4_traits = typename mxfp4_block::traits;

    __dpct_inline__ ಭಿನ್ನ operator()(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> ibx_offset,
                                     ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> d_offset, ಧ್ರುವ int8_t * q8_1_quant_ptr,
                                     ಧ್ರುವ sycl::half2 * q8_1_ds, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
        static_assert(mxfp4_traits::vdr_mmvq == 2, "vector load assumes vdr_mmvq == 2");
        ಧ್ರುವ uint8_t * base = static_cast<ಧ್ರುವ uint8_t *>(vbq);

        // Reordered nibble blocks are 16 contiguous bytes and iqs is 0 or 2, so each lane's two
        // weight ints are one aligned 8-byte load (the AoS layout needed eight byte loads).
        ಧ್ರುವ sycl::int2 q4 = *reinterpret_cast<ಧ್ರುವ sycl::int2 *>(base + ibx_offset.first + ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * iqs);
        ಧ್ರುವ uint8_t    e  = base[d_offset.first];

        // Low nibbles pair with q8_1 ints iqs..iqs+1, high nibbles with iqs+4..iqs+5.
        ಧ್ರುವ sycl::int2 u_lo = *reinterpret_cast<ಧ್ರುವ sycl::int2 *>(q8_1_quant_ptr + ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * iqs);
        ಧ್ರುವ sycl::int2 u_hi = *reinterpret_cast<ಧ್ರುವ sycl::int2 *>(q8_1_quant_ptr + ಪರಿಮಾಣ(ಪೂರ್ಣಾಙ್ಕ) * (iqs + 4));

        ಧ್ರುವ sycl::int2 v0 = get_int_from_mxfp4(q4.x());
        ಧ್ರುವ sycl::int2 v1 = get_int_from_mxfp4(q4.y());

        ಪೂರ್ಣಾಙ್ಕ sumi = 0;
        sumi = ggml_sycl_dp4a(v0.x(), u_lo.x(), sumi);
        sumi = ggml_sycl_dp4a(v0.y(), u_hi.x(), sumi);
        sumi = ggml_sycl_dp4a(v1.x(), u_lo.y(), sumi);
        sumi = ggml_sycl_dp4a(v1.y(), u_hi.y(), sumi);

        ಧ್ರುವ ಭಿನ್ನ d = ggml_sycl_e8m0_to_fp32(e) * 0.5f * static_cast<ಭಿನ್ನ>((*q8_1_ds)[0]);
        ಪ್ರತ್ಯಾಗಮ d * sumi;
    }
};
#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

#define VDR_Q2_0_Q8_1_MMVQ 1

template <ಪೂರ್ಣಾಙ್ಕ vdr>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_q2_0_q8_1_impl(
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * v,
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * u,
    ಧ್ರುವ ಭಿನ್ನ & d2,
    ಧ್ರುವ sycl::half2 & ds8) {
    ಪೂರ್ಣಾಙ್ಕ sumi = 0;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < vdr; ++i) {
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; ++j) {
            ಧ್ರುವ uint8_t q = (uint8_t) ((uint32_t) v[i] >> (8 * j));

            // unpack 2-bit values to byte lanes (0..3), then apply zero-point
            // correction with ds8f.y() below, mirroring the q4_0 style.
            ಪೂರ್ಣಾಙ್ಕ vi = 0;
            vi |= (((q >> 0) & 0x3) & 0xFF) << 0;
            vi |= (((q >> 2) & 0x3) & 0xFF) << 8;
            vi |= (((q >> 4) & 0x3) & 0xFF) << 16;
            vi |= (((q >> 6) & 0x3) & 0xFF) << 24;

            sumi = dpct::dp4a(vi, u[4 * i + j], sumi);
        }
    }

    ಧ್ರುವ sycl::float2 ds8f = ds8.convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
    // q2_0 has zero-point 1. Scale ds8f.y() by processed-lane ratio,
    // consistent with q4_0's explicit zero-point subtraction style.
    ಪ್ರತ್ಯಾಗಮ d2 * (sumi * ds8f.x() - ((ಭಿನ್ನ) vdr / (ಭಿನ್ನ) QI2_0) * ds8f.y());
}

template <ಪೂರ್ಣಾಙ್ಕ vdr>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_q4_0_q8_1_impl(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * v, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * u, ಧ್ರುವ ಭಿನ್ನ & d4,
                                                    ಧ್ರುವ sycl::half2 & ds8) {
    ಪೂರ್ಣಾಙ್ಕ sumi = 0;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < vdr; ++i) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

    ಧ್ರುವ sycl::float2 ds8f = ds8.convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();

    // second part effectively subtracts 8 from each quant value
    ಪ್ರತ್ಯಾಗಮ d4 * (sumi * ds8f.x() - (8 * vdr / QI4_0) * ds8f.y());
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

template <ಪೂರ್ಣಾಙ್ಕ vdr>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_q4_1_q8_1_impl(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *v, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *u,
                                                    ಧ್ರುವ sycl::half2 &dm4,
                                                    ಧ್ರುವ sycl::half2 &ds8) {

    ಪೂರ್ಣಾಙ್ಕ sumi = 0;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < vdr; ++i) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

#ifdef GGML_SYCL_F16
    ಧ್ರುವ sycl::float2 tmp =
        (dm4 * ds8).convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
    ಧ್ರುವ ಭಿನ್ನ d4d8 = tmp.x();
    ಧ್ರುವ ಭಿನ್ನ m4s8 = tmp.y();
#ಅನ್ಯಥಾ
    ಧ್ರುವ sycl::float2 dm4f =
        dm4.convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
    ಧ್ರುವ sycl::float2 ds8f =
        ds8.convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
    ಧ್ರುವ ಭಿನ್ನ d4d8 = dm4f.x() * ds8f.x();
    ಧ್ರುವ ಭಿನ್ನ m4s8 = dm4f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    ಪ್ರತ್ಯಾಗಮ sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <ಪೂರ್ಣಾಙ್ಕ vdr>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_q5_0_q8_1_impl(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *vl, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *vh, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *u,
                       ಧ್ರುವ ಭಿನ್ನ &d5, ಧ್ರುವ sycl::half2 &ds8) {
    ಪೂರ್ಣಾಙ್ಕ sumi = 0;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < vdr; ++i) {
        ಪೂರ್ಣಾಙ್ಕ vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        ಪೂರ್ಣಾಙ್ಕ vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

    ಧ್ರುವ sycl::float2 ds8f =
        ds8.convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();

    // second part effectively subtracts 16 from each quant value
    ಪ್ರತ್ಯಾಗಮ d5 * (sumi * ds8f.x() - (16 * vdr / QI5_0) * ds8f.y());
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <ಪೂರ್ಣಾಙ್ಕ vdr>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_q5_1_q8_1_impl(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *vl, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *vh, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *u,
                       ಧ್ರುವ sycl::half2 &dm5, ಧ್ರುವ sycl::half2 &ds8) {

    ಪೂರ್ಣಾಙ್ಕ sumi = 0;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < vdr; ++i) {
        ಪೂರ್ಣಾಙ್ಕ vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        ಪೂರ್ಣಾಙ್ಕ vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

#ifdef GGML_SYCL_F16
     ಧ್ರುವ sycl::float2 tmp =
        (dm5 * ds8).convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
    ಧ್ರುವ ಭಿನ್ನ d5d8 = tmp.x();
    ಧ್ರುವ ಭಿನ್ನ m5s8 = tmp.y();


#ಅನ್ಯಥಾ
    ಧ್ರುವ sycl::float2 dm5f =
        dm5.convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
    ಧ್ರುವ sycl::float2 ds8f =
        ds8.convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
    ಧ್ರುವ ಭಿನ್ನ d5d8 = dm5f.x() * ds8f.x();
    ಧ್ರುವ ಭಿನ್ನ m5s8 = dm5f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    ಪ್ರತ್ಯಾಗಮ sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <ಪೂರ್ಣಾಙ್ಕ vdr>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_q8_0_q8_1_impl(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *v, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *u,
                                                    ಧ್ರುವ ಭಿನ್ನ &d8_0,
                                                    ಧ್ರುವ ಭಿನ್ನ &d8_1) {

    ಪೂರ್ಣಾಙ್ಕ sumi = 0;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

    ಪ್ರತ್ಯಾಗಮ d8_0*d8_1 * sumi;
}

template <typename T, ಪೂರ್ಣಾಙ್ಕ vdr>
ಸ್ಥಿರ __dpct_inline__ T vec_dot_q8_0_q8_1_impl(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * v, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * u, ಧ್ರುವ T & d8_0, ಧ್ರುವ T & d8_1) {
    ಪೂರ್ಣಾಙ್ಕ sumi = 0;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_sycl_dp4a(v[i], u[i], sumi);
    }

    ಪ್ರತ್ಯಾಗಮ d8_0*d8_1 * ((T) sumi);
}

template <ಪೂರ್ಣಾಙ್ಕ vdr>
ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_q8_1_q8_1_impl(ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *v, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *u,
                                                    ಧ್ರುವ sycl::half2 &dm8,
                                                    ಧ್ರುವ sycl::half2 &ds8) {

    ಪೂರ್ಣಾಙ್ಕ sumi = 0;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

#ifdef GGML_SYCL_F16
    ಧ್ರುವ sycl::float2 tmp =
        (dm8 * ds8).convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
    ಧ್ರುವ ಭಿನ್ನ d8d8 = tmp.x();
    ಧ್ರುವ ಭಿನ್ನ m8s8 = tmp.y();
#ಅನ್ಯಥಾ
    ಧ್ರುವ sycl::float2 dm8f =
        dm8.convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
    ಧ್ರುವ sycl::float2 ds8f =
        ds8.convert<ಭಿನ್ನ, sycl::rounding_mode::automatic>();
    ಧ್ರುವ ಭಿನ್ನ d8d8 = dm8f.x() * ds8f.x();
    ಧ್ರುವ ಭಿನ್ನ m8s8 = dm8f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    ಪ್ರತ್ಯಾಗಮ sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_q4_0_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                  ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {

    ಧ್ರುವ block_q4_0 * bq4_0 = (ಧ್ರುವ block_q4_0 *) vbq;

    ಪೂರ್ಣಾಙ್ಕ v[VDR_Q4_0_Q8_1_MMVQ];
    ಪೂರ್ಣಾಙ್ಕ u[2 * VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]         = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    ಪ್ರತ್ಯಾಗಮ vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_q2_0_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                  ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {

    ಧ್ರುವ block_q2_0 * bq2_0 = (ಧ್ರುವ block_q2_0 *) vbq;

    ಪೂರ್ಣಾಙ್ಕ v[2 * VDR_Q2_0_Q8_1_MMVQ];
    ಪೂರ್ಣಾಙ್ಕ u[8 * VDR_Q2_0_Q8_1_MMVQ];

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < VDR_Q2_0_Q8_1_MMVQ; ++i) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ base = 4 * (iqs + i);

        // Q2_0 has QK2_0 = 64 and uses 2 x QK8_1 blocks on the RHS.
        v[2 * i + 0] = get_int_from_uint8(bq2_0->qs, iqs + i);
        v[2 * i + 1] = get_int_from_uint8(bq2_0->qs, iqs + i + QI2_0);

        u[8 * i + 0] = get_int_from_int8_aligned(bq8_1[0].qs, base + 0);
        u[8 * i + 1] = get_int_from_int8_aligned(bq8_1[0].qs, base + 1);
        u[8 * i + 2] = get_int_from_int8_aligned(bq8_1[0].qs, base + 2);
        u[8 * i + 3] = get_int_from_int8_aligned(bq8_1[0].qs, base + 3);

        u[8 * i + 4] = get_int_from_int8_aligned(bq8_1[1].qs, base + 0);
        u[8 * i + 5] = get_int_from_int8_aligned(bq8_1[1].qs, base + 1);
        u[8 * i + 6] = get_int_from_int8_aligned(bq8_1[1].qs, base + 2);
        u[8 * i + 7] = get_int_from_int8_aligned(bq8_1[1].qs, base + 3);
    }

    ಧ್ರುವ ಭಿನ್ನ sum0 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + 0, u + 0, bq2_0->d, bq8_1[0].ds);
    ಧ್ರುವ ಭಿನ್ನ sum1 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + VDR_Q2_0_Q8_1_MMVQ, u + 4 * VDR_Q2_0_Q8_1_MMVQ, bq2_0->d, bq8_1[1].ds);
    ಪ್ರತ್ಯಾಗಮ sum0 + sum1;
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_q4_1_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                  ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {

    ಧ್ರುವ block_q4_1 * bq4_1 = (ಧ್ರುವ block_q4_1 *) vbq;

    ಪೂರ್ಣಾಙ್ಕ v[VDR_Q4_1_Q8_1_MMVQ];
    ಪೂರ್ಣಾಙ್ಕ u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    ಪ್ರತ್ಯಾಗಮ vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

#define VDR_MXFP4_Q8_1_MMVQ 2
#define VDR_MXFP4_Q8_1_MMQ  4

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_mxfp4_q8_1(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq,
                                                ಧ್ರುವ block_q8_1 * __restrict__ bq8_1,
                                                ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
    ಧ್ರುವ block_mxfp4 * bq4 = (ಧ್ರುವ block_mxfp4 *) vbq;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * q8 = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *) bq8_1->qs + iqs;

    ಪೂರ್ಣಾಙ್ಕ sumi = 0;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ aux_q4 = get_int_b1(bq4->qs, iqs + l);
        ಧ್ರುವ sycl::int2 v      = get_int_from_mxfp4(aux_q4);
        sumi = ggml_sycl_dp4a(v.x(), q8[l + 0], sumi);
        sumi = ggml_sycl_dp4a(v.y(), q8[l + 4], sumi);
    }

    ಧ್ರುವ ಭಿನ್ನ d = ggml_sycl_e8m0_to_fp32(bq4->e) * 0.5f * (bq8_1->ds)[0];
    ಪ್ರತ್ಯಾಗಮ d * sumi;
}

#define VDR_NVFP4_Q8_1_MMVQ 4
#define VDR_NVFP4_Q8_1_MMQ  8

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_nvfp4_q8_1(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq,
                                                ಧ್ರುವ block_q8_1 * __restrict__ bq8_1,
                                                ಧ್ರುವ int32_t & iqs) {
    ಧ್ರುವ block_nvfp4 * bq4 = (ಧ್ರುವ block_nvfp4 *) vbq;
    ಭಿನ್ನ sum = 0.0f;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; i++) {
        ಧ್ರುವ int32_t iqs0 = iqs + 2*i;
        ಧ್ರುವ int32_t iqs1 = iqs0 + 1;
        ಧ್ರುವ int32_t is = iqs0 >> 1;
        ಧ್ರುವ sycl::int2   v0   = get_int_from_table_16(get_int_b4(bq4->qs, iqs0), kvalues_mxfp4);
        ಧ್ರುವ sycl::int2   v1   = get_int_from_table_16(get_int_b4(bq4->qs, iqs1), kvalues_mxfp4);
        ಧ್ರುವ block_q8_1 * bq8 = bq8_1 + (is >> 1);
        ಧ್ರುವ int32_t i8 = ((is & 1) << 2);

        ಪೂರ್ಣಾಙ್ಕ sumi = ggml_sycl_dp4a(v0.x(), get_int_b4(bq8->qs, i8 + 0), 0);
        sumi     = ggml_sycl_dp4a(v0.y(), get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi     = ggml_sycl_dp4a(v1.x(), get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi     = ggml_sycl_dp4a(v1.y(), get_int_b4(bq8->qs, i8 + 3), sumi);

        ಧ್ರುವ ಭಿನ್ನ d = ggml_sycl_ue4m3_to_fp32(bq4->d[is]) * (bq8->ds)[0];
        sum += d * ಭಿನ್ನ(sumi);
    }

    ಪ್ರತ್ಯಾಗಮ sum;
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_q5_0_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                  ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {

    ಧ್ರುವ block_q5_0 * bq5_0 = (ಧ್ರುವ block_q5_0 *) vbq;

    ಪೂರ್ಣಾಙ್ಕ vl[VDR_Q5_0_Q8_1_MMVQ];
    ಪೂರ್ಣಾಙ್ಕ vh[VDR_Q5_0_Q8_1_MMVQ];
    ಪೂರ್ಣಾಙ್ಕ  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    ಪ್ರತ್ಯಾಗಮ vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_q5_1_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                  ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {

    ಧ್ರುವ block_q5_1 * bq5_1 = (ಧ್ರುವ block_q5_1 *) vbq;

    ಪೂರ್ಣಾಙ್ಕ vl[VDR_Q5_1_Q8_1_MMVQ];
    ಪೂರ್ಣಾಙ್ಕ vh[VDR_Q5_1_Q8_1_MMVQ];
    ಪೂರ್ಣಾಙ್ಕ  u[2*VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    ಪ್ರತ್ಯಾಗಮ vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_q8_0_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                  ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {

    ಧ್ರುವ block_q8_0 * bq8_0 = (ಧ್ರುವ block_q8_0 *) vbq;

    ಪೂರ್ಣಾಙ್ಕ v[VDR_Q8_0_Q8_1_MMVQ];
    ಪೂರ್ಣಾಙ್ಕ u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    ಪ್ರತ್ಯಾಗಮ vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                      bq8_1->ds[0]);
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_q2_K_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                  ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {

    ಧ್ರುವ block_q2_K * bq2_K = (ಧ್ರುವ block_q2_K *) vbq;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bq8_offset = QR2_K * (iqs / QI8_1);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    ಧ್ರುವ uint8_t * scales = bq2_K->scales + scale_offset;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    ಪೂರ್ಣಾಙ್ಕ    u[QR2_K];
    ಭಿನ್ನ d8[QR2_K];

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + i].ds[0];
    }

    ಪ್ರತ್ಯಾಗಮ vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_q3_K_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                  ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {

    ಧ್ರುವ block_q3_K * bq3_K = (ಧ್ರುವ block_q3_K *) vbq;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bq8_offset = QR3_K * (iqs / (QI3_K/2));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    ಧ್ರುವ ಭಿನ್ನ d = bq3_K->d;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    ಪೂರ್ಣಾಙ್ಕ    u[QR3_K];
    ಭಿನ್ನ d8[QR3_K];

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + i].ds[0];
    }

    ಪ್ರತ್ಯಾಗಮ vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ vec_dot_q4_K_q8_1(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ block_q8_1 * __restrict__ bq8_1,
                                               ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
#ifndef GGML_QKK_64

    ಧ್ರುವ block_q4_K * bq4_K = (ಧ್ರುವ block_q4_K *) vbq;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ        bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *      q4         = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *) (bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    ಧ್ರುವ uint16_t * scales     = (ಧ್ರುವ uint16_t *) bq4_K->scales;

    ಪ್ರತ್ಯಾಗಮ vec_dot_q4_K_q8_1_common(q4, scales, bq4_K->dm, bq8_1, iqs);

#ಅನ್ಯಥಾ

#ಯದಿ __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    ಧ್ರುವ block_q4_K * bq4_K = (ಧ್ರುವ block_q4_K *) vbq;

    ಭಿನ್ನ sumf_d = 0.0f;
    ಭಿನ್ನ sumf_m = 0.0f;

    uint16_t aux16[2];
    ಧ್ರುವ uint8_t * s = (ಧ್ರುವ uint8_t *)aux16;

    ಧ್ರುವ uint16_t * a = (ಧ್ರುವ uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    ಧ್ರುವ ಭಿನ್ನ dall = bq4_K->dm[0];
    ಧ್ರುವ ಭಿನ್ನ dmin = bq4_K->dm[1];

    ಧ್ರುವ ಭಿನ್ನ d8_1 = bq8_1[0].ds[0];
    ಧ್ರುವ ಭಿನ್ನ d8_2 = bq8_1[1].ds[1];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ui1 = *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)bq8_1[0].qs + (iqs/2));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ui2 = *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)bq8_1[0].qs + (iqs/2) + 4);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ui3 = *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)bq8_1[1].qs + (iqs/2));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ui4 = *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)bq8_1[1].qs + (iqs/2) + 4);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * q4 = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)bq4_K->qs + (iqs/2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ v1 = q4[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ v2 = q4[4];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dot1 = dpct::dp4a(ui2, v2 & 0x0f0f0f0f, dpct::dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dot2 = dpct::dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, dpct::dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dot3 = dpct::dp4a(0x01010101, ui2, dpct::dp4a(0x01010101, ui1, 0));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ dot4 = dpct::dp4a(0x01010101, ui4, dpct::dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    ಪ್ರತ್ಯಾಗಮ dall * sumf_d - dmin * sumf_m;

#ಅನ್ಯಥಾ
    bad_arch();
#endif // __SYCL_ARCH__ >= VER_4VEC

#endif
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_q5_K_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                  ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {

#ifndef GGML_QKK_64
    ಧ್ರುವ block_q5_K * bq5_K = (ಧ್ರುವ block_q5_K *) vbq;

    ಪೂರ್ಣಾಙ್ಕ   vl[2];
    ಪೂರ್ಣಾಙ್ಕ   vh[2];
    ಪೂರ್ಣಾಙ್ಕ    u[2*QR5_K];
    ಭಿನ್ನ d8[QR5_K];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * ql = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * qh = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    ಧ್ರುವ uint16_t * scales = (ಧ್ರುವ uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ j = bq8_offset/2;
    ಯದಿ (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } ಅನ್ಯಥಾ {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    ಧ್ರುವ uint8_t * sc = (ಧ್ರುವ uint8_t *)aux;
    ಧ್ರುವ uint8_t * m  = sc + 2;

#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ i = 0; i < QR5_K; ++i) {
        ಧ್ರುವ block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = bq8i->ds[0];

        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * q8 = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    ಪ್ರತ್ಯಾಗಮ vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#ಅನ್ಯಥಾ

#ಯದಿ __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    ಧ್ರುವ block_q5_K * bq5_K = (ಧ್ರುವ block_q5_K *) vbq;

    ಧ್ರುವ int8_t * s = bq5_K->scales;

    ಧ್ರುವ ಭಿನ್ನ d = bq5_K->d;

    ಧ್ರುವ ಭಿನ್ನ d8_1 = bq8_1[0].ds[0];
    ಧ್ರುವ ಭಿನ್ನ d8_2 = bq8_1[1].ds[1];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ui1 = *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)bq8_1[0].qs + (iqs/2));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ui2 = *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)bq8_1[0].qs + (iqs/2) + 4);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ui3 = *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)bq8_1[1].qs + (iqs/2));
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ui4 = *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)bq8_1[1].qs + (iqs/2) + 4);

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * ql = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)bq5_K->qs + (iqs/2);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vl1 = ql[0];
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vl2 = ql[4];

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ step = 4 * (iqs/2); // 0, 4, 8, 12
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ im = step/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ in = step%8; // 0, 4, 0, 4
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vh = (*((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)(bq5_K->qh + in))) >> im;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    ಧ್ರುವ ಭಿನ್ನ sumf_d = d8_1 * (dpct::dp4a(ui1, v1, 0) * s[0] + dpct::dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (dpct::dp4a(ui3, v3, 0) * s[2] + dpct::dp4a(ui4, v4, 0) * s[3]);

    ಪ್ರತ್ಯಾಗಮ d * sumf_d;

#ಅನ್ಯಥಾ
    bad_arch();
#endif // __SYCL_ARCH__ >= VER_4VEC

#endif
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_q6_K_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                  ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {

    ಧ್ರುವ block_q6_K * bq6_K = (ಧ್ರುವ block_q6_K *) vbq;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vl = get_int_from_uint8(bq6_K->ql, iqs);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    ಧ್ರುವ int8_t * scales = bq6_K->scales + scale_offset;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ u0 = get_int_from_int8_aligned(
        bq8_1[bq8_offset + 0].qs, iqs % QI8_1);
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ u1 = get_int_from_int8_aligned(
        bq8_1[bq8_offset + 2].qs, iqs % QI8_1);
    ಧ್ರುವ ಭಿನ್ನ d80 = bq8_1[bq8_offset + 0].ds[0];
    ಧ್ರುವ ಭಿನ್ನ d81 = bq8_1[bq8_offset + 2].ds[0];

    ಪ್ರತ್ಯಾಗಮ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u0, u1, scales[0], scales[4], bq6_K->d, d80, d81);
}


// NOTE: the VDR_IQ*_Q8_1_MMVQ values deliberately differ from the identically named CUDA constants
// (vecdotq.cuh): the SYCL kernels pair them with a halved qi (e.g. QI3_S/2), so the values are not
// interchangeable and must not be copied across backends.
#define VDR_IQ2_XXS_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ uint32_t iq_sign_mask(ಧ್ರುವ uint32_t signs) {
    ಧ್ರುವ uint32_t bits =
        (signs & 0x01) | ((signs & 0x02) << 7) |
        ((signs & 0x04) << 14) | ((signs & 0x08) << 21);

    ಪ್ರತ್ಯಾಗಮ bits * 0xff;
}

ಸ್ಥಿರ __dpct_inline__ uint32_t iq_sign_bits(ಧ್ರುವ uint32_t encoded) {
    ಧ್ರುವ uint32_t signs = encoded & 127;

    ಪ್ರತ್ಯಾಗಮ signs | ((sycl::popcount(signs) & 1) << 7);
}

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_iq2_xxs_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                     ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs,
                     ಧ್ರುವ uint64_t *iq2xxs_grid, ಧ್ರುವ uint8_t *ksigns_iq2xs,
                     ಧ್ರುವ uint8_t *kmask_iq2xs) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq2_xxs * bq2 = (ಧ್ರುವ block_iq2_xxs *) vbq;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib32 = iqs;
    ಧ್ರುವ uint16_t * q2 = bq2->qs + 4*ib32;
    ಧ್ರುವ uint8_t  * aux8 = (ಧ್ರುವ uint8_t *)q2;
    ಧ್ರುವ int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = q2[2] | (q2[3] << 16);
    ಪೂರ್ಣಾಙ್ಕ sumi = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 4; ++l) {
        ಧ್ರುವ uint8_t * grid = (ಧ್ರುವ uint8_t *)(iq2xxs_grid + aux8[l]);
        ಧ್ರುವ uint8_t  signs = ksigns_iq2xs[aux32 & 127];
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 8; ++j) {
            sumi += q8[j] * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
        aux32 >>= 7;
    }
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.25f;
    ಪ್ರತ್ಯಾಗಮ d * sumi;
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
    ಪ್ರತ್ಯಾಗಮ 0.f;
#endif
}

#define VDR_IQ2_XS_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_iq2_xs_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                    ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs,
                    ಧ್ರುವ uint64_t *iq2xs_grid) {
#ಯದಿ DPCT_COMPATIBILITY_TEMP >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq2_xs * bq2 = (ಧ್ರುವ block_iq2_xs *) vbq;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib32 = iqs;
    ಧ್ರುವ uint16_t * q2 = bq2->qs + 4*ib32;
    ಧ್ರುವ int8_t   * q8 = bq8_1[ib32].qs;
    ಧ್ರುವ uint8_t ls1 = bq2->scales[ib32] & 0xf;
    ಧ್ರುವ uint8_t ls2 = bq2->scales[ib32] >>  4;
    ಪೂರ್ಣಾಙ್ಕ sumi1 = 0;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 2; ++l) {
        ಧ್ರುವ uint64_t grid = iq2xs_grid[q2[l] & 511];
        ಧ್ರುವ uint32_t sign_bits = iq_sign_bits(q2[l] >> 9);
        ಧ್ರುವ uint32_t signs0 = iq_sign_mask(sign_bits);
        ಧ್ರುವ uint32_t signs1 = iq_sign_mask(sign_bits >> 4);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi1 = dpct::dp4a(grid_l, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)q8 + 1), sumi1);
        q8 += 8;
    }
    ಪೂರ್ಣಾಙ್ಕ sumi2 = 0;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 2; l < 4; ++l) {
        ಧ್ರುವ uint64_t grid = iq2xs_grid[q2[l] & 511];
        ಧ್ರುವ uint32_t sign_bits = iq_sign_bits(q2[l] >> 9);
        ಧ್ರುವ uint32_t signs0 = iq_sign_mask(sign_bits);
        ಧ್ರುವ uint32_t signs1 = iq_sign_mask(sign_bits >> 4);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi2 = dpct::dp4a(grid_l, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)q8 + 1), sumi2);
        q8 += 8;
    }
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    ಪ್ರತ್ಯಾಗಮ d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
    ಪ್ರತ್ಯಾಗಮ 0.f;
#endif
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
    ಪ್ರತ್ಯಾಗಮ 0.f;
#endif
}

#define VDR_IQ2_S_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_iq2_s_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                   ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq2_s * bq2 = (ಧ್ರುವ block_iq2_s *) vbq;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib32 = iqs;
    ಧ್ರುವ int8_t  * q8 = bq8_1[ib32].qs;
    ಧ್ರುವ uint8_t * signs = bq2->qs + QK_K/8 + 4*ib32;
    ಧ್ರುವ uint8_t ls1 = bq2->scales[ib32] & 0xf;
    ಧ್ರುವ uint8_t ls2 = bq2->scales[ib32] >>  4;
    ಪೂರ್ಣಾಙ್ಕ sumi1 = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 2; ++l) {
        ಧ್ರುವ uint32_t * grid = (ಧ್ರುವ uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        ಧ್ರುವ uint32_t signs0 = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ಧ್ರುವ uint32_t signs1 = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_l = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_h = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi1 = dpct::dp4a(grid_l, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)q8 + 1), sumi1);
        q8 += 8;
    }
    ಪೂರ್ಣಾಙ್ಕ sumi2 = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 2; l < 4; ++l) {
        ಧ್ರುವ uint32_t * grid = (ಧ್ರುವ uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        ಧ್ರುವ uint32_t signs0 = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ಧ್ರುವ uint32_t signs1 = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_l = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_h = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi2 = dpct::dp4a(grid_l, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)q8 + 1), sumi2);
        q8 += 8;
    }
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    ಪ್ರತ್ಯಾಗಮ d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
#endif
}

#define VDR_IQ3_XXS_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_iq3_xxs_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                     ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs,
                     ಧ್ರುವ uint32_t *iq3xxs_grid) {
#ಯದಿ DPCT_COMPATIBILITY_TEMP >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq3_xxs * bq2 = (ಧ್ರುವ block_iq3_xxs *) vbq;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib32 = iqs;
    ಧ್ರುವ uint8_t  * q3 = bq2->qs + 8*ib32;
    ಧ್ರುವ uint16_t * gas = (ಧ್ರುವ uint16_t *)(bq2->qs + QK_K/4) + 2*ib32;
    ಧ್ರುವ int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = gas[0] | (gas[1] << 16);
    ಪೂರ್ಣಾಙ್ಕ sumi = 0;
#pragma unroll
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 4; ++l) {
        ಧ್ರುವ uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
        ಧ್ರುವ uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
        ಧ್ರುವ uint32_t sign_bits = iq_sign_bits(aux32);
        ಧ್ರುವ uint32_t signs0 = iq_sign_mask(sign_bits);
        ಧ್ರುವ uint32_t signs1 = iq_sign_mask(sign_bits >> 4);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)q8 + 1), sumi);
        q8 += 8;
        aux32 >>= 7;
    }
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.5f;
    ಪ್ರತ್ಯಾಗಮ d * sumi;
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
    ಪ್ರತ್ಯಾಗಮ 0.f;
#endif
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
    ಪ್ರತ್ಯಾಗಮ 0.f;
#endif
}

template <> ಸಂರಚನಾ reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_XXS> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_IQ3_XXS;

    __dpct_inline__ ಭಿನ್ನ operator()(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> ibx_offset,
                                     ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> d_offset, ಧ್ರುವ int8_t * q8_1_quant_ptr,
                                     ಧ್ರುವ sycl::half2 * q8_1_ds, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
#ಯದಿ DPCT_COMPATIBILITY_TEMP >= MIN_CC_DP4A && QK_K == 256
        ಧ್ರುವ uint8_t * base = static_cast<ಧ್ರುವ uint8_t *>(vbq);
        ಧ್ರುವ uint8_t * qs   = base + ibx_offset.first;
        ಧ್ರುವ uint8_t * q3   = qs + 8 * iqs;
        ಧ್ರುವ uint16_t * gas = reinterpret_cast<ಧ್ರುವ uint16_t *>(qs + QK_K / 4) + 2 * iqs;
        ಧ್ರುವ int8_t * q8    = q8_1_quant_ptr + iqs * QK8_1;
        uint32_t aux32       = gas[0] | (gas[1] << 16);

        ಪೂರ್ಣಾಙ್ಕ sumi = 0;
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 4; ++l) {
            ಧ್ರುವ uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
            ಧ್ರುವ uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
            ಧ್ರುವ uint32_t sign_bits = iq_sign_bits(aux32);
            ಧ್ರುವ uint32_t signs0 = iq_sign_mask(sign_bits);
            ಧ್ರುವ uint32_t signs1 = iq_sign_mask(sign_bits >> 4);
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *) q8 + 1), sumi);
            q8 += 8;
            aux32 >>= 7;
        }

        ಧ್ರುವ ggml_half d = *reinterpret_cast<ಧ್ರುವ ggml_half *>(base + d_offset.first);
        ಪ್ರತ್ಯಾಗಮ (ಭಿನ್ನ) d * (0.5f + aux32) * q8_1_ds[iqs][0] * 0.5f * sumi;
#ಅನ್ಯಥಾ
        GGML_UNUSED(vbq);
        GGML_UNUSED(ibx_offset);
        GGML_UNUSED(d_offset);
        GGML_UNUSED(q8_1_quant_ptr);
        GGML_UNUSED(q8_1_ds);
        GGML_UNUSED(iqs);
        GGML_ABORT("IQ3_XXS reorder vec dot requires dp4a and QK_K == 256");
#endif
    }
};

#define VDR_IQ3_S_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_iq3_s_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                   ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs,
                   ಧ್ರುವ uint32_t *iq3s_grid) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq3_s * bq2 = (ಧ್ರುವ block_iq3_s *) vbq;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib32 = iqs;
    ಧ್ರುವ uint8_t  * qs = bq2->qs + 8*ib32;
    ಧ್ರುವ int8_t   * q8 = bq8_1[ib32].qs;
    ಪೂರ್ಣಾಙ್ಕ sumi = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 4; ++l) {
        ಧ್ರುವ uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((bq2->qh[ib32] << (8 - 2*l)) & 256)];
        ಧ್ರುವ uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((bq2->qh[ib32] << (7 - 2*l)) & 256)];
        ಧ್ರುವ uint32_t signs = bq2->signs[4 * ib32 + l];
        ಧ್ರುವ uint32_t signs0 = iq_sign_mask(signs);
        ಧ್ರುವ uint32_t signs1 = iq_sign_mask(signs >> 4);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)q8 + 1), sumi);
        q8 += 8;
    }
    ಧ್ರುವ ಭಿನ್ನ d =
        (ಭಿನ್ನ)bq2->d *
        (1 + 2 * ((bq2->scales[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf)) *
        bq8_1[ib32].ds[0];
    ಪ್ರತ್ಯಾಗಮ d * sumi;
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
#endif
}

template <> ಸಂರಚನಾ reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_S> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_IQ3_S;

    __dpct_inline__ ಭಿನ್ನ operator()(ಧ್ರುವ ಶೂನ್ಯ * __restrict__ vbq, ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> ibx_offset,
                                     ಧ್ರುವ std::pair<ಪೂರ್ಣಾಙ್ಕ, ಪೂರ್ಣಾಙ್ಕ> d_offset, ಧ್ರುವ int8_t * q8_1_quant_ptr,
                                     ಧ್ರುವ sycl::half2 * q8_1_ds, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ & iqs) {
#ಯದಿ QK_K == 256
        ಧ್ರುವ uint8_t * base     = static_cast<ಧ್ರುವ uint8_t *>(vbq);
        ಧ್ರುವ uint8_t * qs       = base + ibx_offset.first + 8 * iqs;
        ಧ್ರುವ uint8_t * qh       = base + ibx_offset.second;
        ಧ್ರುವ uint8_t * signs    = base + d_offset.first + 4 * iqs;
        ಧ್ರುವ uint8_t * metadata = base + d_offset.second;
        ಧ್ರುವ ggml_half d        = *reinterpret_cast<ಧ್ರುವ ggml_half *>(metadata);
        ಧ್ರುವ uint8_t * scales   = metadata + ಪರಿಮಾಣ(ggml_half);
        ಧ್ರುವ int8_t * q8        = q8_1_quant_ptr + iqs * QK8_1;

        ಪೂರ್ಣಾಙ್ಕ sumi = 0;
#pragma unroll
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 4; ++l) {
            ಧ್ರುವ uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((qh[iqs] << (8 - 2*l)) & 256)];
            ಧ್ರುವ uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((qh[iqs] << (7 - 2*l)) & 256)];
            ಧ್ರುವ uint32_t signs0 = iq_sign_mask(signs[l]);
            ಧ್ರುವ uint32_t signs1 = iq_sign_mask(signs[l] >> 4);
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *) q8 + 1), sumi);
            q8 += 8;
        }

        ಧ್ರುವ ಭಿನ್ನ scale = 1 + 2 * ((scales[iqs / 2] >> (4 * (iqs % 2))) & 0xf);
        ಪ್ರತ್ಯಾಗಮ (ಭಿನ್ನ) d * scale * q8_1_ds[iqs][0] * sumi;
#ಅನ್ಯಥಾ
        GGML_UNUSED(vbq);
        GGML_UNUSED(ibx_offset);
        GGML_UNUSED(d_offset);
        GGML_UNUSED(q8_1_quant_ptr);
        GGML_UNUSED(q8_1_ds);
        GGML_UNUSED(iqs);
        GGML_ABORT("IQ3_S reorder vec dot not supported for QK_K != 256");
#endif
    }
};

#define VDR_IQ1_S_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_iq1_s_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                   ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs,
                   ಧ್ರುವ uint32_t *iq1s_grid_gpu) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq1_s * bq1 = (ಧ್ರುವ block_iq1_s *) vbq;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib32 = iqs;
    ಪೂರ್ಣಾಙ್ಕ sumi = 0;
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * q8 = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)bq8_1[ib32].qs;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 4; ++l) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * grid = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[ib32] >> 3*l) & 7) << 8)));
        ಪೂರ್ಣಾಙ್ಕ grid0 = grid[0] & 0x0f0f0f0f;
        ಪೂರ್ಣಾಙ್ಕ grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi = dpct::dp4a(q8[2 * l + 1], grid1,
                          dpct::dp4a(q8[2 * l + 0], grid0, sumi));
    }

    ಧ್ರುವ ಭಿನ್ನ delta = bq1->qh[ib32] & 0x8000 ? -1-IQ1S_DELTA : -1+IQ1S_DELTA;
    ಧ್ರುವ ಭಿನ್ನ d1q = (ಭಿನ್ನ)bq1->d * (2*((bq1->qh[ib32] >> 12) & 7) + 1);
    ಧ್ರುವ ಭಿನ್ನ d = d1q * bq8_1[ib32].ds[0];
    ಧ್ರುವ ಭಿನ್ನ m = d1q * bq8_1[ib32].ds[1];
    ಪ್ರತ್ಯಾಗಮ d * sumi + m * delta;
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
#endif
}

#define VDR_IQ1_M_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_iq1_m_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                   ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {
#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq1_m * bq1 = (ಧ್ರುವ block_iq1_m *) vbq;

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib32 = iqs;
    ಪೂರ್ಣಾಙ್ಕ   sumi[2] = {0, 0};
    ಭಿನ್ನ sumf[2] = {0.f, 0.f};

    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * q8 = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)bq8_1[ib32].qs;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < 4; ++l) {
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ * grid = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 7) << 8)));
        ಪೂರ್ಣಾಙ್ಕ grid0 = grid[0] & 0x0f0f0f0f;
        ಪೂರ್ಣಾಙ್ಕ grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi[l / 2] = dpct::dp4a(q8[2 * l + 1], grid1,
                                 dpct::dp4a(q8[2 * l + 0], grid0, sumi[l / 2]));
        ಧ್ರುವ ಭಿನ್ನ delta = (bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 0x08 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA;
        ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ sumy = dpct::dp4a(q8[2 * l + 1], 0x01010101,
                                    dpct::dp4a(q8[2 * l + 0], 0x01010101, 0));
        sumf[l/2] += delta*sumy;
    }

    iq1m_scale_t scale;
    ಧ್ರುವ uint16_t * sc = (ಧ್ರುವ uint16_t *)bq1->scales;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)scale.f16 * bq8_1[ib32].ds[0];
    ಪ್ರತ್ಯಾಗಮ d * ((sumi[0] + sumf[0]) * (2*((sc[ib32/2] >> 6*(ib32%2)) & 0x7) + 1) + (sumi[1] + sumf[1]) * (2*((sc[ib32/2] >> (6*(ib32%2)+3)) & 0x7) + 1));
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
#endif
}


#define VDR_IQ4_NL_Q8_1_MMVQ 2

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_iq4_nl_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                    ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {

    ಧ್ರುವ block_iq4_nl * bq = (ಧ್ರುವ block_iq4_nl *) vbq;

    ಧ್ರುವ uint16_t * q4 = (ಧ್ರುವ uint16_t *)bq->qs + 2*iqs;
    ಧ್ರುವ int32_t  * q8 = (ಧ್ರುವ int32_t  *)bq8_1->qs + iqs;

    ಧ್ರುವ uint8_t * ಮೌಲ್ಯಗಳು = (ಧ್ರುವ uint8_t *)kvalues_iq4nl;

    ಪೂರ್ಣಾಙ್ಕ v1, v2;
    ಪೂರ್ಣಾಙ್ಕ sumi1 = 0, sumi2 = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        ಧ್ರುವ uint32_t aux = q4[2*l] | (q4[2*l+1] << 16);
        get_int_from_table_16(aux, ಮೌಲ್ಯಗಳು, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[l + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[l + 4], sumi2);
    }

    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)bq->d * bq8_1->ds[0];
    ಪ್ರತ್ಯಾಗಮ d * (sumi1 + sumi2);
}


#define VDR_IQ4_XS_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ ಭಿನ್ನ
vec_dot_iq4_xs_q8_1(ಧ್ರುವ ಶೂನ್ಯ *__restrict__ vbq,
                    ಧ್ರುವ block_q8_1 *__restrict__ bq8_1, ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ &iqs) {

#ಯದಿ QK_K == 256
    ಧ್ರುವ block_iq4_xs * bq4 = (ಧ್ರುವ block_iq4_xs *) vbq;
    ಧ್ರುವ uint8_t * ಮೌಲ್ಯಗಳು = (ಧ್ರುವ uint8_t *)kvalues_iq4nl;

    // iqs is 0...7
    ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ ib32 = iqs;
    ಧ್ರುವ int32_t  * q8 = (ಧ್ರುವ ಪೂರ್ಣಾಙ್ಕ *)bq8_1[ib32].qs;
    ಧ್ರುವ uint32_t * q4 = (ಧ್ರುವ uint32_t *)bq4->qs + 4*ib32;
    ಧ್ರುವ int8_t ls = ((bq4->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((bq4->scales_h >> 2*ib32) & 3) << 4);
    ಧ್ರುವ ಭಿನ್ನ d = (ಭಿನ್ನ)bq4->d * (ls - 32) * bq8_1[ib32].ds[0];
    ಪೂರ್ಣಾಙ್ಕ v1, v2;
    ಪೂರ್ಣಾಙ್ಕ sumi1 = 0, sumi2 = 0;
    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ j = 0; j < 4; ++j) {
        get_int_from_table_16(q4[j], ಮೌಲ್ಯಗಳು, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[j + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[j + 4], sumi2);
    }
    ಪ್ರತ್ಯಾಗಮ d * (sumi1 + sumi2);
#ಅನ್ಯಥಾ
    ಪ್ರತಿಜ್ಞಾ(ಅಸತ್ಯ);
#endif
}

#endif // GGML_SYCL_VECDOTQ_HPP
