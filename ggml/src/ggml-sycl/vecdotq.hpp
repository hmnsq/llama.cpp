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

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_SYCL_VECDOTQ_HPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_SYCL_VECDOTQ_HPP

#ಅನ್ತರ್ಭಾವ "dpct/helper.hpp"
#ಅನ್ತರ್ಭಾವ "ggml.h"
#ಅನ್ತರ್ಭಾವ "type.hpp"
#ಅನ್ತರ್ಭಾವ "quants.hpp"

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ದಶಮಾಂಶ (*vec_dot_q_sycl_t)(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * __restrict__ bq8_1,
                                  ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs);

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಂಕ get_int_b1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & i32) {
    ಸ್ಥಿರಾಂಕ uint8_t * x8 = (ಸ್ಥಿರಾಂಕ uint8_t *) x;

    ಪೂರ್ಣಾಂಕ x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    ಹಿಂತಿರುಗಿ x32;
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಂಕ get_int_b2(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & i32) {
    ಸ್ಥಿರಾಂಕ uint16_t * x16 = (ಸ್ಥಿರಾಂಕ uint16_t *) x; // assume at least 2 byte alignment

    ಪೂರ್ಣಾಂಕ x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    ಹಿಂತಿರುಗಿ x32;
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಂಕ get_int_b4(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * x, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & i32) {
    ಹಿಂತಿರುಗಿ ((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) x)[i32]; // assume at least 4 byte alignment
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಂಕ get_int_from_int8(ಸ್ಥಿರಾಂಕ int8_t* x8, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ& i32) {
  ಸ್ಥಿರಾಂಕ uint16_t* x16 =
      (ಸ್ಥಿರಾಂಕ uint16_t*)(x8 + ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * i32); // assume at least 2 byte
                                                 // alignment

  ಪೂರ್ಣಾಂಕ x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  ಹಿಂತಿರುಗಿ x32;
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಂಕ get_int_from_uint8(
    ಸ್ಥಿರಾಂಕ uint8_t* x8,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ& i32) {
  ಸ್ಥಿರಾಂಕ uint16_t* x16 =
      (ಸ್ಥಿರಾಂಕ uint16_t*)(x8 + ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * i32); // assume at least 2 byte
                                                 // alignment

  ಪೂರ್ಣಾಂಕ x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  ಹಿಂತಿರುಗಿ x32;
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಂಕ get_int_from_int8_aligned(
    ಸ್ಥಿರಾಂಕ int8_t* x8,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ& i32) {
  ಹಿಂತಿರುಗಿ *(
      (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ*)(x8 + ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * i32)); // assume at least 4 byte alignment
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಂಕ get_int_from_uint8_aligned(
    ಸ್ಥಿರಾಂಕ uint8_t* x8,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ& i32) {
  ಹಿಂತಿರುಗಿ *(
      (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ*)(x8 + ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * i32)); // assume at least 4 byte alignment
}

ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಂಕ byte_sub_4(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ a, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b) {
  ಸ್ಥಿರಾಂಕ uint32_t ua = static_cast<uint32_t>(a);
  ಸ್ಥಿರಾಂಕ uint32_t ub = static_cast<uint32_t>(b);
  ಹಿಂತಿರುಗಿ static_cast<ಪೂರ್ಣಾಂಕ>(((ua | 0x80808080u) - ub) ^ 0x80808080u);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ u0, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ u1, ಸ್ಥಿರಾಂಕ int8_t sc0,
    ಸ್ಥಿರಾಂಕ int8_t sc1, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d80, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d81) {
    static_assert(QR6_K == 2, "q6_K MMVQ scalar fast path assumes QR6_K == 2");

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vil0 = (vl >> 0) & 0x0F0F0F0F;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vih0 = ((vh >> 0) << 4) & 0x30303030;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vi0 = byte_sub_4(vil0 | vih0, 0x20202020);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vil1 = (vl >> 4) & 0x0F0F0F0F;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vih1 = ((vh >> 4) << 4) & 0x30303030;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vi1 = byte_sub_4(vil1 | vih1, 0x20202020);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sumf =
        d80 * (dpct::dp4a(vi0, u0, 0) * sc0) +
        d81 * (dpct::dp4a(vi1, u1, 0) * sc1);

    ಹಿಂತಿರುಗಿ d * sumf;
}

ಸ್ಥಿರ __dpct_inline__ ಶೂನ್ಯ get_int_from_ಕೋಷ್ಟಕ_16(ಸ್ಥಿರಾಂಕ uint32_t &q4,
                                                  ಸ್ಥಿರಾಂಕ uint8_t *ಮೌಲ್ಯಗಳು,
                                                  ಪೂರ್ಣಾಂಕ &val1, ಪೂರ್ಣಾಂಕ &val2) {

    uint32_t aux32; ಸ್ಥಿರಾಂಕ uint8_t * q8 = (ಸ್ಥಿರಾಂಕ uint8_t *)&aux32;
    aux32 = q4 & 0x0f0f0f0f;
    uint16_t v1 = ಮೌಲ್ಯಗಳು[q8[0]] | (ಮೌಲ್ಯಗಳು[q8[1]] << 8);
    uint16_t v2 = ಮೌಲ್ಯಗಳು[q8[2]] | (ಮೌಲ್ಯಗಳು[q8[3]] << 8);
    val1 = v1 | (v2 << 16);
    aux32 = (q4 >> 4) & 0x0f0f0f0f;
    v1 = ಮೌಲ್ಯಗಳು[q8[0]] | (ಮೌಲ್ಯಗಳು[q8[1]] << 8);
    v2 = ಮೌಲ್ಯಗಳು[q8[2]] | (ಮೌಲ್ಯಗಳು[q8[3]] << 8);
    val2 = v1 | (v2 << 16);
}

ಸ್ಥಿರ __dpct_inline__ sycl::int2 get_int_from_ಕೋಷ್ಟಕ_16(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ& q4, ಸ್ಥಿರಾಂಕ int8_t* ಕೋಷ್ಟಕ) {
  ಸ್ಥಿರಾಂಕ uint32_t* table32 = (ಸ್ಥಿರಾಂಕ uint32_t*)ಕೋಷ್ಟಕ;
  uint32_t tmp[2];
  ಸ್ಥಿರಾಂಕ uint32_t ಕೆಳಗಿನ_ಮೇಲಿನ_selection_indices =
      (0x32103210 | ((q4 & 0x88888888) >> 1));
#ಸೂಚನೆ unroll
  ಗಾಗಿ (uint32_t i = 0; i < 2; ++i) {
    ಸ್ಥಿರಾಂಕ uint32_t ಸರಿಸು = 16 * i;

    ಸ್ಥಿರಾಂಕ uint32_t ಕೆಳಗಿನ =
        dpct::byte_level_ಕ್ರಮಬದಲಾವಣೆ(table32[0], table32[1], q4 >> ಸರಿಸು);
    ಸ್ಥಿರಾಂಕ uint32_t ಮೇಲಿನ =
        dpct::byte_level_ಕ್ರಮಬದಲಾವಣೆ(table32[2], table32[3], q4 >> ಸರಿಸು);
    tmp[i] = dpct::byte_level_ಕ್ರಮಬದಲಾವಣೆ(
        ಕೆಳಗಿನ, ಮೇಲಿನ, ಕೆಳಗಿನ_ಮೇಲಿನ_selection_indices >> ಸರಿಸು);
  }
  ಹಿಂತಿರುಗಿ sycl::int2(
      dpct::byte_level_ಕ್ರಮಬದಲಾವಣೆ(tmp[0], tmp[1], 0x6420),
      dpct::byte_level_ಕ್ರಮಬದಲಾವಣೆ(tmp[0], tmp[1], 0x7531));
}

// Four E2M1 codes (one per byte, bits 0..3) to their kvalues_mxfp4 int8 values. SWAR arithmetic
// replaces get_int_from_table_16 for MXFP4: dpct::byte_level_permute is emulated with 64-bit shifts,
// eight per int, which made the MXFP4 GEMV compute-bound on Intel GPUs.
// Magnitudes 0,1,2,3,4,6,8,12 = m + [m>=5] + [m>=6] + 3*[m>=7]; each byte stays below 256, so the
// byte-wise adds never carry. -0 (code 8) is left as 0 so the two's-complement +1 cannot carry either.
ಸ್ಥಿರ __dpct_inline__ ಪೂರ್ಣಾಂಕ mxfp4_codes_ಗಮ್ಯ_int8(ಸ್ಥಿರಾಂಕ uint32_t x) {
    ಸ್ಥಿರಾಂಕ uint32_t m   = x & 0x07070707u;
    ಸ್ಥಿರಾಂಕ uint32_t ge5 = ((m + 0x03030303u) >> 3) & 0x01010101u;
    ಸ್ಥಿರಾಂಕ uint32_t ge6 = ((m + 0x02020202u) >> 3) & 0x01010101u;
    ಸ್ಥಿರಾಂಕ uint32_t ge7 = ((m + 0x01010101u) >> 3) & 0x01010101u;
    ಸ್ಥಿರಾಂಕ uint32_t mag = m + ge5 + ge6 + 3u * ge7;
    ಸ್ಥಿರಾಂಕ uint32_t nz  = ((mag + 0x7f7f7f7fu) >> 7) & 0x01010101u;
    ಸ್ಥಿರಾಂಕ uint32_t neg = (x >> 3) & nz & 0x01010101u;
    ಹಿಂತಿರುಗಿ (ಪೂರ್ಣಾಂಕ) ((mag ^ (neg * 0xffu)) + neg);
}

// Same result as get_int_from_table_16(q4, kvalues_mxfp4): x = low nibbles, y = high nibbles.
ಸ್ಥಿರ __dpct_inline__ sycl::int2 get_int_from_mxfp4(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ q4) {
    ಹಿಂತಿರುಗಿ sycl::int2(mxfp4_codes_ಗಮ್ಯ_int8((uint32_t) q4 & 0x0f0f0f0fu),
                      mxfp4_codes_ಗಮ್ಯ_int8(((uint32_t) q4 >> 4) & 0x0f0f0f0fu));
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q2_K_Q8_1_MMVQ 1

// contiguous v/x values
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q2_K_q8_1_impl_mmvq(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &v, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ u, ಸ್ಥಿರಾಂಕ uint8_t *__restrict__ scales,
    ಸ್ಥಿರಾಂಕ sycl::half2 &dm2, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict__ d8) {

    ದಶಮಾಂಶ sumf_d = 0.0f;
    ದಶಮಾಂಶ sumf_m = 0.0f;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QR2_K; ++i) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sc = scales[2*i];

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vi = (v >> (2*i)) & 0x03030303;

        sumf_d +=
            d8[i] * (dpct::dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        ಪೂರ್ಣಾಂಕ m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] *
                  dpct::dp4a(
                      m, u[i],
                      0); // multiply constant q2_K part with sum of q8_1 values
    }

    ಸ್ಥಿರಾಂಕ sycl::float2 dm2f =
        dm2.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

    ಹಿಂತಿರುಗಿ dm2f.x() * sumf_d - dm2f.y() * sumf_m;
}


#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q3_K_Q8_1_MMVQ 1

// contiguous v/x values
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q3_K_q8_1_impl_mmvq(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &vl, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &vh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ u,
    ಸ್ಥಿರಾಂಕ uint8_t *__restrict__ scales, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &scale_ಸ್ಥಾನಾಂತರ,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ &d3, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict__ d8) {

    ದಶಮಾಂಶ sumf = 0.0f;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QR3_K; ++i) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ isc = scale_ಸ್ಥಾನಾಂತರ + 2*i;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ isc_ಕೆಳಗಿನ = isc % (QK_K/32);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sc_ಸರಿಸು_ಕೆಳಗಿನ = 4 * (isc / (QK_K/32));
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sc_ಕೆಳಗಿನ  = (scales[isc_ಕೆಳಗಿನ] >> sc_ಸರಿಸು_ಕೆಳಗಿನ) & 0xF;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ isc_ಮೇಲಿನ = isc % (QK_K/64);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sc_ಸರಿಸು_ಮೇಲಿನ = 2 * (isc / (QK_K/64));
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sc_ಮೇಲಿನ = ((scales[(QK_K/32) + isc_ಮೇಲಿನ] >> sc_ಸರಿಸು_ಮೇಲಿನ) & 3) << 4;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sc = (sc_ಕೆಳಗಿನ | sc_ಮೇಲಿನ) - 32;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vil = (vl >> (2*i)) & 0x03030303;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vih = ((vh >> i) << 2) & 0x04040404;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vi =
            dpct::vectorized_ದ್ವಿಮಾನ<sycl::char4>(vil, vih, dpct::sub_sat());

        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    ಹಿಂತಿರುಗಿ d3 * sumf;
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q4_K_Q8_1_MMVQ 2

// contiguous v/x values
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q4_K_q8_1_impl_vmmq(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ v, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ u,
    ಸ್ಥಿರಾಂಕ uint8_t *__restrict__ sc, ಸ್ಥಿರಾಂಕ uint8_t *__restrict__ m,
    ಸ್ಥಿರಾಂಕ sycl::half2 &dm4, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict__ d8) {

    ದಶಮಾಂಶ sumf_d = 0.0f;
    ದಶಮಾಂಶ sumf_m = 0.0f;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QR4_K; ++i) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dot1 =
            dpct::dp4a(v1i, u[2 * i + 1],
                       dpct::dp4a(v0i, u[2 * i + 0], 0)); // SIMD dot product
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 1],
                       dpct::dp4a(0x01010101, u[2 * i + 0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    ಸ್ಥಿರಾಂಕ sycl::float2 dm4f =
        dm4.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

    ಹಿಂತಿರುಗಿ dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q5_K_Q8_1_MMVQ 2

// contiguous v/x values
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q5_K_q8_1_impl_vmmq(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ vl, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ vh,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ u, ಸ್ಥಿರಾಂಕ uint8_t *__restrict__ sc,
    ಸ್ಥಿರಾಂಕ uint8_t *__restrict__ m, ಸ್ಥಿರಾಂಕ sycl::half2 &dm5,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict__ d8) {

    ದಶಮಾಂಶ sumf_d = 0.0f;
    ದಶಮಾಂಶ sumf_m = 0.0f;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QR5_K; ++i) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ v0i = vl0i | vh0i;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ v1i = vl1i | vh1i;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dot1 =
            dpct::dp4a(v0i, u[2 * i + 0],
                       dpct::dp4a(v1i, u[2 * i + 1], 0)); // SIMD dot product
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 0],
                       dpct::dp4a(0x01010101, u[2 * i + 1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    ಸ್ಥಿರಾಂಕ sycl::float2 dm5f =
        dm5.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

    ಹಿಂತಿರುಗಿ dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}


#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q6_K_Q8_1_MMVQ 1

// contiguous v/x values
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q6_K_q8_1_impl_mmvq(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &vl, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &vh,
                            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *__restrict__ u,
                            ಸ್ಥಿರಾಂಕ int8_t *__restrict__ scales, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ &d,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *__restrict__ d8) {
    ಹಿಂತಿರುಗಿ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q1_0_Q8_1_MMVQ 1
#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q1_0_Q8_1_MMQ  4

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q1_0_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                  ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q1_0 * bq1_0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q1_0 *) vbq;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * bq8_1_chunk = bq8_1 + iqs;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ        d1          = bq1_0->d;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ          v           = get_int_from_uint8_aligned(bq1_0->qs, iqs);

    ಪೂರ್ಣಾಂಕ vi_ಬೈಟುಗಳು[8];
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಸರಿಸು = j * 4;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bits4 = (v >> ಸರಿಸು) & 0x0F;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b0    = (bits4 & 0x01) ? 1 : -1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b1    = (bits4 & 0x02) ? 1 : -1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b2    = (bits4 & 0x04) ? 1 : -1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ b3    = (bits4 & 0x08) ? 1 : -1;
        vi_ಬೈಟುಗಳು[j]     = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }

    ಪೂರ್ಣಾಂಕ sumi = 0;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ u = get_int_from_int8_aligned(bq8_1_chunk->qs, j);
        sumi        = ggml_sycl_dp4a(vi_ಬೈಟುಗಳು[j], u, sumi);
    }

    ಹಿಂತಿರುಗಿ d1 * bq8_1_chunk->ds[0] * sumi;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

ಮಾದರಿ <ggml_type T> ರಚನೆ reorder_vec_dot_q_sycl {
    static_assert(T != T, "ggml_type for reorder vecdot not implemented");
};

// For some types the weight side of the dot product does not depend on the destination column, so a
// multi-column mul_mat_vec can unpack it once per block instead of once per column. Such a type adds
// load() and dot() next to operator() and opts in here. See reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>.
ಮಾದರಿ <ggml_type T> ರಚನೆ reorder_vec_dot_shared_weights {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ value = ಸುಳ್ಳು;
};

ಮಾದರಿ <> ರಚನೆ reorder_vec_dot_shared_weights<GGML_TYPE_Q4_K> {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ value = ನಿಜ;
};

ಮಾದರಿ <> ರಚನೆ reorder_vec_dot_shared_weights<GGML_TYPE_Q5_K> {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ value = ನಿಜ;
};

ಮಾದರಿ <ggml_type T> ರಚನೆ reorder_vec_dot_shared_activations {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ value = ಸುಳ್ಳು;
};

ಮಾದರಿ <> ರಚನೆ reorder_vec_dot_shared_activations<GGML_TYPE_Q4_K> {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ value = ನಿಜ;
};

ಮಾದರಿ <> ರಚನೆ reorder_vec_dot_shared_activations<GGML_TYPE_Q5_K> {
    ಸ್ಥಿರ constexpr ತಾರ್ಕಿಕ value = ನಿಜ;
};

ಮಾದರಿ <> ರಚನೆ reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q4_0;

    ಬಳಸಿ q4_0_ಖಂಡ  = ggml_sycl_reordered::ಖಂಡ_q_t<GGML_TYPE_Q4_0>;
    ಬಳಸಿ q4_0_traits = ಬಗೆ_ಹೆಸರು q4_0_ಖಂಡ::traits;

    __dpct_inline__ ದಶಮಾಂಶ vec_dot_q4_0_q8_1_impl(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * v, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * u, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ & d4, ಸ್ಥಿರಾಂಕ sycl::half2 & ds8) {
        ಪೂರ್ಣಾಂಕ sumi = 0;

#ಸೂಚನೆ unroll
        ಗಾಗಿ (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vi0 = (v[i] >> 0) & 0x0F0F0F0F;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vi1 = (v[i] >> 4) & 0x0F0F0F0F;

            // SIMD dot product of quantized values
            sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
            sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
        }

        ಸ್ಥಿರಾಂಕ sycl::float2 ds8f = ds8.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

        // second part effectively subtracts 8 from each quant value
        ಹಿಂತಿರುಗಿ d4 * (sumi * ds8f.x() - (8 * q4_0_traits::vdr_mmvq / q4_0_traits::qi) * ds8f.y());
    }

    __dpct_inline__ ದಶಮಾಂಶ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> ibx_ಸ್ಥಾನಾಂತರ,
                                     ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> d_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr,
                                     ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
        ಸ್ಥಿರಾಂಕ uint8_t * bq4_0 = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vbq) + ibx_ಸ್ಥಾನಾಂತರ.ಮೊದಲ;
        ಸ್ಥಿರಾಂಕ ggml_half d = *(reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half *>(static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vbq) + d_ಸ್ಥಾನಾಂತರ.ಮೊದಲ));
        ಪೂರ್ಣಾಂಕ             v[q4_0_traits::vdr_mmvq];
        ಪೂರ್ಣಾಂಕ             u[2 * q4_0_traits::vdr_mmvq];


#ಸೂಚನೆ unroll
        ಗಾಗಿ (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            v[i]         = get_int_from_uint8(bq4_0, iqs + i);
            u[2 * i + 0] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
            u[2 * i + 1] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i + q4_0_traits::qi);
        }

        ಹಿಂತಿರುಗಿ vec_dot_q4_0_q8_1_impl(v, u, d, *q8_1_ds);
    }
};

// Load four contiguous dwords per operand instead of loading each value separately.
ರಚನೆ reorder_vec_dot_q8_0_wide {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    ಬಳಸಿ q8_0_ಖಂಡ  = ggml_sycl_reordered::ಖಂಡ_q_t<GGML_TYPE_Q8_0>;
    ಬಳಸಿ q8_0_traits = ಬಗೆ_ಹೆಸರು q8_0_ಖಂಡ::traits;

    __dpct_inline__ ದಶಮಾಂಶ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> ibx_ಸ್ಥಾನಾಂತರ,
                                     ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> d_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr,
                                     ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
        static_assert(q8_0_traits::vdr_mmvq == 4, "the wide load moves exactly four dwords");

        ಸ್ಥಿರಾಂಕ uint8_t * ತಳ = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vbq);
        ಸ್ಥಿರಾಂಕ int8_t *  qs   = reinterpret_cast<ಸ್ಥಿರಾಂಕ int8_t *>(ತಳ + ibx_ಸ್ಥಾನಾಂತರ.ಮೊದಲ);
        ಸ್ಥಿರಾಂಕ ggml_half d    = *reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half *>(ತಳ + d_ಸ್ಥಾನಾಂತರ.ಮೊದಲ);

        ಸ್ಥಿರಾಂಕ sycl::int4 v = *reinterpret_cast<ಸ್ಥಿರಾಂಕ sycl::int4 *>(qs + ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * iqs);
        ಸ್ಥಿರಾಂಕ sycl::int4 u = *reinterpret_cast<ಸ್ಥಿರಾಂಕ sycl::int4 *>(q8_1_quant_ptr + ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * iqs);

        ಪೂರ್ಣಾಂಕ sumi = 0;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        ಸ್ಥಿರಾಂಕ sycl::half2 ds_ಮೌಲ್ಯಗಳು = *q8_1_ds;
        ಹಿಂತಿರುಗಿ static_cast<ದಶಮಾಂಶ>(d) * static_cast<ದಶಮಾಂಶ>(ds_ಮೌಲ್ಯಗಳು[0]) * sumi;
    }
};

ಮಾದರಿ <> ರಚನೆ reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    ಬಳಸಿ q8_0_ಖಂಡ  = ggml_sycl_reordered::ಖಂಡ_q_t<GGML_TYPE_Q8_0>;
    ಬಳಸಿ q8_0_traits = ಬಗೆ_ಹೆಸರು q8_0_ಖಂಡ::traits;

    __dpct_inline__ ದಶಮಾಂಶ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> ibx_ಸ್ಥಾನಾಂತರ,
                                     ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> d_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr,
                                     ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vdr = q8_0_traits::vdr_mmvq) {
        ಸ್ಥಿರಾಂಕ uint8_t * ತಳ = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vbq);
        ಸ್ಥಿರಾಂಕ int8_t *  qs   = reinterpret_cast<ಸ್ಥಿರಾಂಕ int8_t *>(ತಳ + ibx_ಸ್ಥಾನಾಂತರ.ಮೊದಲ);
        ಸ್ಥಿರಾಂಕ ggml_half  d   = *reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half *>(ತಳ + d_ಸ್ಥಾನಾಂತರ.ಮೊದಲ);

        ಪೂರ್ಣಾಂಕ v[q8_0_traits::vdr_mmvq];
        ಪೂರ್ಣಾಂಕ u[q8_0_traits::vdr_mmvq];

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < vdr; ++i) {
            v[i] = get_int_from_int8(qs, iqs + i);
            u[i] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
        }

        ಪೂರ್ಣಾಂಕ sumi = 0;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < vdr; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        ಸ್ಥಿರಾಂಕ sycl::half2 ds_ಮೌಲ್ಯಗಳು = *q8_1_ds;
        ಹಿಂತಿರುಗಿ static_cast<ದಶಮಾಂಶ>(d) * static_cast<ದಶಮಾಂಶ>(ds_ಮೌಲ್ಯಗಳು[0]) * sumi;
    }
};

ಮಾದರಿ <> ರಚನೆ reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q2_K;

    ಬಳಸಿ q2_k_ಖಂಡ  = ggml_sycl_reordered::ಖಂಡ_q_t<GGML_TYPE_Q2_K>;
    ಬಳಸಿ q2_k_traits = ಬಗೆ_ಹೆಸರು q2_k_ಖಂಡ::traits;

    __dpct_inline__ ದಶಮಾಂಶ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> ibx_ಸ್ಥಾನಾಂತರ,
                                     ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> d_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr,
                                     ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
        ಸ್ಥಿರಾಂಕ uint8_t *    ತಳ   = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vbq);
        ಸ್ಥಿರಾಂಕ uint8_t *    qs     = ತಳ + ibx_ಸ್ಥಾನಾಂತರ.ಮೊದಲ;
        ಸ್ಥಿರಾಂಕ uint8_t *    scales = ತಳ + d_ಸ್ಥಾನಾಂತರ.ಮೊದಲ;
        ಸ್ಥಿರಾಂಕ ggml_half2 * dm     = reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half2 *>(ತಳ + d_ಸ್ಥಾನಾಂತರ.second);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bq8_ಸ್ಥಾನಾಂತರ   = QR2_K * (iqs / QI8_1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ scale_ಸ್ಥಾನಾಂತರ = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ v = get_int_from_uint8_aligned(qs, iqs);

        ಪೂರ್ಣಾಂಕ   u[QR2_K];
        ದಶಮಾಂಶ d8[QR2_K];

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QR2_K; ++i) {
            ಸ್ಥಿರಾಂಕ int8_t * quant_ತಳ_ptr = q8_1_quant_ptr + (bq8_ಸ್ಥಾನಾಂತರ + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_ತಳ_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_ಸ್ಥಾನಾಂತರ + i))[0];
        }

        ಹಿಂತಿರುಗಿ vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales + scale_ಸ್ಥಾನಾಂತರ, *dm, d8);
    }
};

ಮಾದರಿ <> ರಚನೆ reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q3_K;

    ಬಳಸಿ q3_k_ಖಂಡ  = ggml_sycl_reordered::ಖಂಡ_q_t<GGML_TYPE_Q3_K>;
    ಬಳಸಿ q3_k_traits = ಬಗೆ_ಹೆಸರು q3_k_ಖಂಡ::traits;

    __dpct_inline__ ದಶಮಾಂಶ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> ibx_ಸ್ಥಾನಾಂತರ,
                                     ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> d_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr,
                                     ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
        ಸ್ಥಿರಾಂಕ uint8_t *  ತಳ   = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vbq);
        ಸ್ಥಿರಾಂಕ uint8_t *  qs     = ತಳ + ibx_ಸ್ಥಾನಾಂತರ.ಮೊದಲ;
        ಸ್ಥಿರಾಂಕ uint8_t *  hmask  = ತಳ + ibx_ಸ್ಥಾನಾಂತರ.second;
        ಸ್ಥಿರಾಂಕ uint8_t *  scales = ತಳ + d_ಸ್ಥಾನಾಂತರ.ಮೊದಲ;
        ಸ್ಥಿರಾಂಕ ggml_half  d      = *reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half *>(ತಳ + d_ಸ್ಥಾನಾಂತರ.second);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bq8_ಸ್ಥಾನಾಂತರ   = QR3_K * (iqs / (QI3_K / 2));
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ scale_ಸ್ಥಾನಾಂತರ = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl = get_int_from_uint8(qs, iqs);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vh = ~get_int_from_uint8(hmask, iqs % (QI3_K / 2)) >> bq8_ಸ್ಥಾನಾಂತರ;

        ಪೂರ್ಣಾಂಕ   u[QR3_K];
        ದಶಮಾಂಶ d8[QR3_K];

#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QR3_K; ++i) {
            ಸ್ಥಿರಾಂಕ int8_t * quant_ತಳ_ptr = q8_1_quant_ptr + (bq8_ಸ್ಥಾನಾಂತರ + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_ತಳ_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_ಸ್ಥಾನಾಂತರ + i))[0];
        }

        ಹಿಂತಿರುಗಿ vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, scales, scale_ಸ್ಥಾನಾಂತರ, static_cast<ದಶಮಾಂಶ>(d), d8);
    }
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ vec_dot_q4_K_q8_1_common(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * __restrict__ q4, ಸ್ಥಿರಾಂಕ uint16_t * __restrict__ scales,
                                             ಸ್ಥಿರಾಂಕ ggml_half2 & dm, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * __restrict__ bq8_1,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &        iqs) {
    ಪೂರ್ಣಾಂಕ   v[2];
    ಪೂರ್ಣಾಂಕ   u[2 * QR4_K];
    ದಶಮಾಂಶ d8[QR4_K];

    v[0] = q4[0];
    v[1] = q4[4];

    uint16_t  aux[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
    ಆದರೆ (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } ಇಲ್ಲದಿದ್ದರೆ {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }

    ಸ್ಥಿರಾಂಕ uint8_t * sc = (ಸ್ಥಿರಾಂಕ uint8_t *) aux;
    ಸ್ಥಿರಾಂಕ uint8_t * m  = sc + 2;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bq8_ಸ್ಥಾನಾಂತರ = QR4_K * ((iqs / 2) / (QI8_1 / 2));

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QR4_K; ++i) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * bq8i = bq8_1 + bq8_ಸ್ಥಾನಾಂತರ + i;
        d8[i]                   = bq8i->ds[0];

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * q8 = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0]   = q8[0];
        u[2 * i + 1]   = q8[4];
    }

    ಹಿಂತಿರುಗಿ vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
}

ಮಾದರಿ <> ರಚನೆ reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q4_K;

    ಬಳಸಿ q4_k_ಖಂಡ  = ggml_sycl_reordered::ಖಂಡ_q_t<GGML_TYPE_Q4_K>;
    ಬಳಸಿ q4_k_traits = ಬಗೆ_ಹೆಸರು q4_k_ಖಂಡ::traits;

    ರಚನೆ weights {
        ಪೂರ್ಣಾಂಕ        v[2];
        uint16_t   aux[2];
        ggml_half2 dm;
        ಪೂರ್ಣಾಂಕ        bq8_ಸ್ಥಾನಾಂತರ;
    };

    ರಚನೆ activations {
        ಪೂರ್ಣಾಂಕ   u[2 * QR4_K];
        ದಶಮಾಂಶ d8[QR4_K];
    };

    __dpct_inline__ ಸ್ಥಿರ weights load(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> ibx_ಸ್ಥಾನಾಂತರ,
                                        ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> d_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
        ಸ್ಥಿರಾಂಕ uint8_t *    ತಳ = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vbq);
        ಸ್ಥಿರಾಂಕ uint8_t *    qs   = ತಳ + ibx_ಸ್ಥಾನಾಂತರ.ಮೊದಲ;
        ಸ್ಥಿರಾಂಕ uint8_t *    scs  = ತಳ + d_ಸ್ಥಾನಾಂತರ.ಮೊದಲ;
        ಸ್ಥಿರಾಂಕ ggml_half2 * dms  = reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half2 *>(ತಳ + d_ಸ್ಥಾನಾಂತರ.second);

        weights w;
        w.bq8_ಸ್ಥಾನಾಂತರ = QR4_K * ((iqs / 2) / (QI8_1 / 2));

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *      q4     = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) (qs + 16 * w.bq8_ಸ್ಥಾನಾಂತರ + 4 * ((iqs / 2) % 4));
        ಸ್ಥಿರಾಂಕ uint16_t * scales = (ಸ್ಥಿರಾಂಕ uint16_t *) scs;

        w.v[0] = q4[0];
        w.v[1] = q4[4];

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        ಆದರೆ (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } ಇಲ್ಲದಿದ್ದರೆ {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        ಹಿಂತಿರುಗಿ w;
    }

    __dpct_inline__ ಸ್ಥಿರ activations ತುಂಬು_activations(ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr,
                                                        ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
        activations a;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bq8_ಸ್ಥಾನಾಂತರ = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QR4_K; ++i) {
            ಸ್ಥಿರಾಂಕ int8_t * quant_ತಳ_ptr = q8_1_quant_ptr + (bq8_ಸ್ಥಾನಾಂತರ + i) * QK8_1;
            sycl::half2    ds_ಮೌಲ್ಯಗಳು      = *(q8_1_ds + bq8_ಸ್ಥಾನಾಂತರ + i);

            a.d8[i] = ds_ಮೌಲ್ಯಗಳು[0];

            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * q8 = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) quant_ತಳ_ptr + ((iqs / 2) % 4);
            a.u[2 * i + 0] = q8[0];
            a.u[2 * i + 1] = q8[4];
        }

        ಹಿಂತಿರುಗಿ a;
    }

    __dpct_inline__ ಸ್ಥಿರ ದಶಮಾಂಶ apply(ಸ್ಥಿರಾಂಕ weights & w, ಸ್ಥಿರಾಂಕ activations & a) {
        ಸ್ಥಿರಾಂಕ uint8_t * sc = (ಸ್ಥಿರಾಂಕ uint8_t *) w.aux;
        ಸ್ಥಿರಾಂಕ uint8_t * m  = sc + 2;

        ಹಿಂತಿರುಗಿ vec_dot_q4_K_q8_1_impl_vmmq(w.v, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ ಸ್ಥಿರ ದಶಮಾಂಶ dot(ಸ್ಥಿರಾಂಕ weights & w, ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr,
                                     ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
        ಸ್ಥಿರಾಂಕ ಸ್ವಯಂಚಾಲಿತ a = ತುಂಬು_activations(q8_1_quant_ptr, q8_1_ds, iqs);

        ಹಿಂತಿರುಗಿ apply(w, a);
    }

    __dpct_inline__ ದಶಮಾಂಶ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> ibx_ಸ್ಥಾನಾಂತರ,
                                     ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> d_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr,
                                     ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
        ಹಿಂತಿರುಗಿ dot(load(vbq, ibx_ಸ್ಥಾನಾಂತರ, d_ಸ್ಥಾನಾಂತರ, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

ಮಾದರಿ <> ರಚನೆ reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q5_K;

    ಬಳಸಿ q5_k_ಖಂಡ  = ggml_sycl_reordered::ಖಂಡ_q_t<GGML_TYPE_Q5_K>;
    ಬಳಸಿ q5_k_traits = ಬಗೆ_ಹೆಸರು q5_k_ಖಂಡ::traits;

    ರಚನೆ weights {
        ಪೂರ್ಣಾಂಕ        vl[2];
        ಪೂರ್ಣಾಂಕ        vh[2];
        uint16_t   aux[2];
        ggml_half2 dm;
    };

    // same activation layout as Q4_K
    static_assert(QR5_K == QR4_K);
    ಬಳಸಿ activations = reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::activations;

    __dpct_inline__ ಸ್ಥಿರ weights load(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> ibx_ಸ್ಥಾನಾಂತರ,
                                        ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> d_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
        ಸ್ಥಿರಾಂಕ uint8_t *    ತಳ    = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vbq);
        ಸ್ಥಿರಾಂಕ uint8_t *    qs      = ತಳ + ibx_ಸ್ಥಾನಾಂತರ.ಮೊದಲ;   // low 4 bits
        ಸ್ಥಿರಾಂಕ uint8_t *    qh_ತಳ = ತಳ + ibx_ಸ್ಥಾನಾಂತರ.second;  // high bit
        ಸ್ಥಿರಾಂಕ uint8_t *    scs     = ತಳ + d_ಸ್ಥಾನಾಂತರ.ಮೊದಲ;
        ಸ್ಥಿರಾಂಕ ggml_half2 * dms     = reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half2 *>(ತಳ + d_ಸ್ಥಾನಾಂತರ.second);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ        bq8_ಸ್ಥಾನಾಂತರ = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *      ql_ptr     = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) (qs + 16 * bq8_ಸ್ಥಾನಾಂತರ + 4 * ((iqs / 2) % 4));
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *      qh_ptr     = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) (qh_ತಳ + 4 * ((iqs / 2) % 4));
        ಸ್ಥಿರಾಂಕ uint16_t * scales     = (ಸ್ಥಿರಾಂಕ uint16_t *) scs;

        weights w;
        w.vl[0] = ql_ptr[0];
        w.vl[1] = ql_ptr[4];

        w.vh[0] = qh_ptr[0] >> bq8_ಸ್ಥಾನಾಂತರ;
        w.vh[1] = qh_ptr[4] >> bq8_ಸ್ಥಾನಾಂತರ;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = (QR5_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        ಆದರೆ (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } ಇಲ್ಲದಿದ್ದರೆ {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        ಹಿಂತಿರುಗಿ w;
    }

    __dpct_inline__ ಸ್ಥಿರ activations ತುಂಬು_activations(ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr,
                                                        ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
        ಹಿಂತಿರುಗಿ reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::ತುಂಬು_activations(q8_1_quant_ptr, q8_1_ds, iqs);
    }

    __dpct_inline__ ಸ್ಥಿರ ದಶಮಾಂಶ apply(ಸ್ಥಿರಾಂಕ weights & w, ಸ್ಥಿರಾಂಕ activations & a) {
        ಸ್ಥಿರಾಂಕ uint8_t * sc = (ಸ್ಥಿರಾಂಕ uint8_t *) w.aux;
        ಸ್ಥಿರಾಂಕ uint8_t * m  = sc + 2;

        ಹಿಂತಿರುಗಿ vec_dot_q5_K_q8_1_impl_vmmq(w.vl, w.vh, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ ಸ್ಥಿರ ದಶಮಾಂಶ dot(ಸ್ಥಿರಾಂಕ weights & w, ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr,
                                     ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
        ಹಿಂತಿರುಗಿ apply(w, ತುಂಬು_activations(q8_1_quant_ptr, q8_1_ds, iqs));
    }

    __dpct_inline__ ದಶಮಾಂಶ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> ibx_ಸ್ಥಾನಾಂತರ,
                                     ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> d_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr,
                                     ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
        ಹಿಂತಿರುಗಿ dot(load(vbq, ibx_ಸ್ಥಾನಾಂತರ, d_ಸ್ಥಾನಾಂತರ, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

ಮಾದರಿ <> ರಚನೆ reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_Q6_K;

    ಬಳಸಿ q6_k_ಖಂಡ  = ggml_sycl_reordered::ಖಂಡ_q_t<GGML_TYPE_Q6_K>;
    ಬಳಸಿ q6_k_traits = ಬಗೆ_ಹೆಸರು q6_k_ಖಂಡ::traits;

    __dpct_inline__ ದಶಮಾಂಶ vec_dot_q6_K_q8_1_impl_mmvq(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * __restrict__ u,
                                                      ಸ್ಥಿರಾಂಕ int8_t * __restrict__ scales, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d,
                                                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * __restrict__ d8) {
        ಹಿಂತಿರುಗಿ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
    }

    __dpct_inline__ ದಶಮಾಂಶ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> ibx_ಸ್ಥಾನಾಂತರ,
                     ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> d_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr, ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds,
                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ iqs) {
        ಸ್ಥಿರಾಂಕ uint8_t *   ತಳ   = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vbq);
        ಸ್ಥಿರಾಂಕ uint8_t *   ql     = ತಳ + ibx_ಸ್ಥಾನಾಂತರ.ಮೊದಲ;
        ಸ್ಥಿರಾಂಕ uint8_t *   qh     = ತಳ + ibx_ಸ್ಥಾನಾಂತರ.second;
        ಸ್ಥಿರಾಂಕ int8_t *    scales = reinterpret_cast<ಸ್ಥಿರಾಂಕ int8_t *>(ತಳ + d_ಸ್ಥಾನಾಂತರ.ಮೊದಲ);
        ಸ್ಥಿರಾಂಕ ggml_half * d      = (ಸ್ಥಿರಾಂಕ ggml_half *) (ತಳ + d_ಸ್ಥಾನಾಂತರ.second);

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bq8_ಸ್ಥಾನಾಂತರ   = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ scale_ಸ್ಥಾನಾಂತರ = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vh_ಸರಿಸು     = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl = get_int_from_uint8(ql, iqs);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vh = get_int_from_uint8(qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_ಸರಿಸು;

        ಸ್ಥಿರಾಂಕ int8_t * scs = scales + scale_ಸ್ಥಾನಾಂತರ;

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ u0 = get_int_from_int8_aligned(
            q8_1_quant_ptr + bq8_ಸ್ಥಾನಾಂತರ * QK8_1, iqs % QI8_1);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ u1 = get_int_from_int8_aligned(
            q8_1_quant_ptr + (bq8_ಸ್ಥಾನಾಂತರ + 2) * QK8_1, iqs % QI8_1);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d80 = (*(q8_1_ds + bq8_ಸ್ಥಾನಾಂತರ + 0))[0];
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d81 = (*(q8_1_ds + bq8_ಸ್ಥಾನಾಂತರ + 2))[0];

        ಹಿಂತಿರುಗಿ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u0, u1, scs[0], scs[4], *d, d80, d81);
    }
};

ಮಾದರಿ <> ರಚನೆ reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_MXFP4;

    ಬಳಸಿ mxfp4_ಖಂಡ  = ggml_sycl_reordered::ಖಂಡ_q_t<GGML_TYPE_MXFP4>;
    ಬಳಸಿ mxfp4_traits = ಬಗೆ_ಹೆಸರು mxfp4_ಖಂಡ::traits;

    __dpct_inline__ ದಶಮಾಂಶ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> ibx_ಸ್ಥಾನಾಂತರ,
                                     ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> d_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr,
                                     ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
        static_assert(mxfp4_traits::vdr_mmvq == 2, "vector load assumes vdr_mmvq == 2");
        ಸ್ಥಿರಾಂಕ uint8_t * ತಳ = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vbq);

        // Reordered nibble blocks are 16 contiguous bytes and iqs is 0 or 2, so each lane's two
        // weight ints are one aligned 8-byte load (the AoS layout needed eight byte loads).
        ಸ್ಥಿರಾಂಕ sycl::int2 q4 = *reinterpret_cast<ಸ್ಥಿರಾಂಕ sycl::int2 *>(ತಳ + ibx_ಸ್ಥಾನಾಂತರ.ಮೊದಲ + ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * iqs);
        ಸ್ಥಿರಾಂಕ uint8_t    e  = ತಳ[d_ಸ್ಥಾನಾಂತರ.ಮೊದಲ];

        // Low nibbles pair with q8_1 ints iqs..iqs+1, high nibbles with iqs+4..iqs+5.
        ಸ್ಥಿರಾಂಕ sycl::int2 u_lo = *reinterpret_cast<ಸ್ಥಿರಾಂಕ sycl::int2 *>(q8_1_quant_ptr + ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * iqs);
        ಸ್ಥಿರಾಂಕ sycl::int2 u_hi = *reinterpret_cast<ಸ್ಥಿರಾಂಕ sycl::int2 *>(q8_1_quant_ptr + ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * (iqs + 4));

        ಸ್ಥಿರಾಂಕ sycl::int2 v0 = get_int_from_mxfp4(q4.x());
        ಸ್ಥಿರಾಂಕ sycl::int2 v1 = get_int_from_mxfp4(q4.y());

        ಪೂರ್ಣಾಂಕ sumi = 0;
        sumi = ggml_sycl_dp4a(v0.x(), u_lo.x(), sumi);
        sumi = ggml_sycl_dp4a(v0.y(), u_hi.x(), sumi);
        sumi = ggml_sycl_dp4a(v1.x(), u_lo.y(), sumi);
        sumi = ggml_sycl_dp4a(v1.y(), u_hi.y(), sumi);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = ggml_sycl_e8m0_ಗಮ್ಯ_fp32(e) * 0.5f * static_cast<ದಶಮಾಂಶ>((*q8_1_ds)[0]);
        ಹಿಂತಿರುಗಿ d * sumi;
    }
};
#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q4_0_Q8_1_MMVQ 2
#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q4_0_Q8_1_MMQ  4

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q2_0_Q8_1_MMVQ 1

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q2_0_q8_1_impl(
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * v,
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * u,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ & d2,
    ಸ್ಥಿರಾಂಕ sycl::half2 & ds8) {
    ಪೂರ್ಣಾಂಕ sumi = 0;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < vdr; ++i) {
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; ++j) {
            ಸ್ಥಿರಾಂಕ uint8_t q = (uint8_t) ((uint32_t) v[i] >> (8 * j));

            // unpack 2-bit values to byte lanes (0..3), then apply zero-point
            // correction with ds8f.y() below, mirroring the q4_0 style.
            ಪೂರ್ಣಾಂಕ vi = 0;
            vi |= (((q >> 0) & 0x3) & 0xFF) << 0;
            vi |= (((q >> 2) & 0x3) & 0xFF) << 8;
            vi |= (((q >> 4) & 0x3) & 0xFF) << 16;
            vi |= (((q >> 6) & 0x3) & 0xFF) << 24;

            sumi = dpct::dp4a(vi, u[4 * i + j], sumi);
        }
    }

    ಸ್ಥಿರಾಂಕ sycl::float2 ds8f = ds8.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
    // q2_0 has zero-point 1. Scale ds8f.y() by processed-lane ratio,
    // consistent with q4_0's explicit zero-point subtraction style.
    ಹಿಂತಿರುಗಿ d2 * (sumi * ds8f.x() - ((ದಶಮಾಂಶ) vdr / (ದಶಮಾಂಶ) QI2_0) * ds8f.y());
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q4_0_q8_1_impl(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * v, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * u, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ & d4,
                                                    ಸ್ಥಿರಾಂಕ sycl::half2 & ds8) {
    ಪೂರ್ಣಾಂಕ sumi = 0;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < vdr; ++i) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

    ಸ್ಥಿರಾಂಕ sycl::float2 ds8f = ds8.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

    // second part effectively subtracts 8 from each quant value
    ಹಿಂತಿರುಗಿ d4 * (sumi * ds8f.x() - (8 * vdr / QI4_0) * ds8f.y());
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q4_1_Q8_1_MMVQ 2
#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q4_1_Q8_1_MMQ  4

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q4_1_q8_1_impl(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *v, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *u,
                                                    ಸ್ಥಿರಾಂಕ sycl::half2 &dm4,
                                                    ಸ್ಥಿರಾಂಕ sycl::half2 &ds8) {

    ಪೂರ್ಣಾಂಕ sumi = 0;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < vdr; ++i) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    ಸ್ಥಿರಾಂಕ sycl::float2 tmp =
        (dm4 * ds8).convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d4d8 = tmp.x();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m4s8 = tmp.y();
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ sycl::float2 dm4f =
        dm4.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
    ಸ್ಥಿರಾಂಕ sycl::float2 ds8f =
        ds8.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d4d8 = dm4f.x() * ds8f.x();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m4s8 = dm4f.y() * ds8f.y();
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    ಹಿಂತಿರುಗಿ sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q5_0_Q8_1_MMVQ 2
#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q5_0_Q8_1_MMQ  4

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q5_0_q8_1_impl(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *vl, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *vh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *u,
                       ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ &d5, ಸ್ಥಿರಾಂಕ sycl::half2 &ds8) {
    ಪೂರ್ಣಾಂಕ sumi = 0;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < vdr; ++i) {
        ಪೂರ್ಣಾಂಕ vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        ಪೂರ್ಣಾಂಕ vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

    ಸ್ಥಿರಾಂಕ sycl::float2 ds8f =
        ds8.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();

    // second part effectively subtracts 16 from each quant value
    ಹಿಂತಿರುಗಿ d5 * (sumi * ds8f.x() - (16 * vdr / QI5_0) * ds8f.y());
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q5_1_Q8_1_MMVQ 2
#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q5_1_Q8_1_MMQ  4

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q5_1_q8_1_impl(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *vl, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *vh, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *u,
                       ಸ್ಥಿರಾಂಕ sycl::half2 &dm5, ಸ್ಥಿರಾಂಕ sycl::half2 &ds8) {

    ಪೂರ್ಣಾಂಕ sumi = 0;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < vdr; ++i) {
        ಪೂರ್ಣಾಂಕ vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        ಪೂರ್ಣಾಂಕ vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
     ಸ್ಥಿರಾಂಕ sycl::float2 tmp =
        (dm5 * ds8).convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d5d8 = tmp.x();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m5s8 = tmp.y();


#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ sycl::float2 dm5f =
        dm5.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
    ಸ್ಥಿರಾಂಕ sycl::float2 ds8f =
        ds8.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d5d8 = dm5f.x() * ds8f.x();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m5s8 = dm5f.y() * ds8f.y();
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    ಹಿಂತಿರುಗಿ sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q8_0_Q8_1_MMVQ 2
#ವ್ಯಾಖ್ಯಾನಿಸು VDR_Q8_0_Q8_1_MMQ 8

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q8_0_q8_1_impl(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *v, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *u,
                                                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ &d8_0,
                                                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ &d8_1) {

    ಪೂರ್ಣಾಂಕ sumi = 0;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

    ಹಿಂತಿರುಗಿ d8_0*d8_1 * sumi;
}

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು T, ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ __dpct_inline__ T vec_dot_q8_0_q8_1_impl(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * v, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * u, ಸ್ಥಿರಾಂಕ T & d8_0, ಸ್ಥಿರಾಂಕ T & d8_1) {
    ಪೂರ್ಣಾಂಕ sumi = 0;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_sycl_dp4a(v[i], u[i], sumi);
    }

    ಹಿಂತಿರುಗಿ d8_0*d8_1 * ((T) sumi);
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ vdr>
ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q8_1_q8_1_impl(ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *v, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *u,
                                                    ಸ್ಥಿರಾಂಕ sycl::half2 &dm8,
                                                    ಸ್ಥಿರಾಂಕ sycl::half2 &ds8) {

    ಪೂರ್ಣಾಂಕ sumi = 0;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_SYCL_F16
    ಸ್ಥಿರಾಂಕ sycl::float2 tmp =
        (dm8 * ds8).convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d8d8 = tmp.x();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m8s8 = tmp.y();
#ಇಲ್ಲದಿದ್ದರೆ
    ಸ್ಥಿರಾಂಕ sycl::float2 dm8f =
        dm8.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
    ಸ್ಥಿರಾಂಕ sycl::float2 ds8f =
        ds8.convert<ದಶಮಾಂಶ, sycl::rounding_ಬಹುಲಕ::automatic>();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d8d8 = dm8f.x() * ds8f.x();
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m8s8 = dm8f.y() * ds8f.y();
#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    ಹಿಂತಿರುಗಿ sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q4_0_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                  ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * bq4_0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) vbq;

    ಪೂರ್ಣಾಂಕ v[VDR_Q4_0_Q8_1_MMVQ];
    ಪೂರ್ಣಾಂಕ u[2 * VDR_Q4_0_Q8_1_MMVQ];

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]         = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    ಹಿಂತಿರುಗಿ vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q2_0_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                  ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_0 * bq2_0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_0 *) vbq;

    ಪೂರ್ಣಾಂಕ v[2 * VDR_Q2_0_Q8_1_MMVQ];
    ಪೂರ್ಣಾಂಕ u[8 * VDR_Q2_0_Q8_1_MMVQ];

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < VDR_Q2_0_Q8_1_MMVQ; ++i) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ತಳ = 4 * (iqs + i);

        // Q2_0 has QK2_0 = 64 and uses 2 x QK8_1 blocks on the RHS.
        v[2 * i + 0] = get_int_from_uint8(bq2_0->qs, iqs + i);
        v[2 * i + 1] = get_int_from_uint8(bq2_0->qs, iqs + i + QI2_0);

        u[8 * i + 0] = get_int_from_int8_aligned(bq8_1[0].qs, ತಳ + 0);
        u[8 * i + 1] = get_int_from_int8_aligned(bq8_1[0].qs, ತಳ + 1);
        u[8 * i + 2] = get_int_from_int8_aligned(bq8_1[0].qs, ತಳ + 2);
        u[8 * i + 3] = get_int_from_int8_aligned(bq8_1[0].qs, ತಳ + 3);

        u[8 * i + 4] = get_int_from_int8_aligned(bq8_1[1].qs, ತಳ + 0);
        u[8 * i + 5] = get_int_from_int8_aligned(bq8_1[1].qs, ತಳ + 1);
        u[8 * i + 6] = get_int_from_int8_aligned(bq8_1[1].qs, ತಳ + 2);
        u[8 * i + 7] = get_int_from_int8_aligned(bq8_1[1].qs, ತಳ + 3);
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sum0 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + 0, u + 0, bq2_0->d, bq8_1[0].ds);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sum1 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + VDR_Q2_0_Q8_1_MMVQ, u + 4 * VDR_Q2_0_Q8_1_MMVQ, bq2_0->d, bq8_1[1].ds);
    ಹಿಂತಿರುಗಿ sum0 + sum1;
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q4_1_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                  ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * bq4_1 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 *) vbq;

    ಪೂರ್ಣಾಂಕ v[VDR_Q4_1_Q8_1_MMVQ];
    ಪೂರ್ಣಾಂಕ u[2*VDR_Q4_1_Q8_1_MMVQ];

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    ಹಿಂತಿರುಗಿ vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_MXFP4_Q8_1_MMVQ 2
#ವ್ಯಾಖ್ಯಾನಿಸು VDR_MXFP4_Q8_1_MMQ  4

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_mxfp4_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq,
                                                ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * __restrict__ bq8_1,
                                                ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 * bq4 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 *) vbq;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * q8 = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) bq8_1->qs + iqs;

    ಪೂರ್ಣಾಂಕ sumi = 0;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ aux_q4 = get_int_b1(bq4->qs, iqs + l);
        ಸ್ಥಿರಾಂಕ sycl::int2 v      = get_int_from_mxfp4(aux_q4);
        sumi = ggml_sycl_dp4a(v.x(), q8[l + 0], sumi);
        sumi = ggml_sycl_dp4a(v.y(), q8[l + 4], sumi);
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = ggml_sycl_e8m0_ಗಮ್ಯ_fp32(bq4->e) * 0.5f * (bq8_1->ds)[0];
    ಹಿಂತಿರುಗಿ d * sumi;
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_NVFP4_Q8_1_MMVQ 4
#ವ್ಯಾಖ್ಯಾನಿಸು VDR_NVFP4_Q8_1_MMQ  8

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_nvfp4_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq,
                                                ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * __restrict__ bq8_1,
                                                ಸ್ಥಿರಾಂಕ int32_t & iqs) {
    ಸ್ಥಿರಾಂಕ ಖಂಡ_nvfp4 * bq4 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_nvfp4 *) vbq;
    ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; i++) {
        ಸ್ಥಿರಾಂಕ int32_t iqs0 = iqs + 2*i;
        ಸ್ಥಿರಾಂಕ int32_t iqs1 = iqs0 + 1;
        ಸ್ಥಿರಾಂಕ int32_t is = iqs0 >> 1;
        ಸ್ಥಿರಾಂಕ sycl::int2   v0   = get_int_from_ಕೋಷ್ಟಕ_16(get_int_b4(bq4->qs, iqs0), kvalues_mxfp4);
        ಸ್ಥಿರಾಂಕ sycl::int2   v1   = get_int_from_ಕೋಷ್ಟಕ_16(get_int_b4(bq4->qs, iqs1), kvalues_mxfp4);
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * bq8 = bq8_1 + (is >> 1);
        ಸ್ಥಿರಾಂಕ int32_t i8 = ((is & 1) << 2);

        ಪೂರ್ಣಾಂಕ sumi = ggml_sycl_dp4a(v0.x(), get_int_b4(bq8->qs, i8 + 0), 0);
        sumi     = ggml_sycl_dp4a(v0.y(), get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi     = ggml_sycl_dp4a(v1.x(), get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi     = ggml_sycl_dp4a(v1.y(), get_int_b4(bq8->qs, i8 + 3), sumi);

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = ggml_sycl_ue4m3_ಗಮ್ಯ_fp32(bq4->d[is]) * (bq8->ds)[0];
        ಮೊತ್ತ += d * ದಶಮಾಂಶ(sumi);
    }

    ಹಿಂತಿರುಗಿ ಮೊತ್ತ;
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q5_0_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                  ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 * bq5_0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 *) vbq;

    ಪೂರ್ಣಾಂಕ vl[VDR_Q5_0_Q8_1_MMVQ];
    ಪೂರ್ಣಾಂಕ vh[VDR_Q5_0_Q8_1_MMVQ];
    ಪೂರ್ಣಾಂಕ  u[2*VDR_Q5_0_Q8_1_MMVQ];

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    ಹಿಂತಿರುಗಿ vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q5_1_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                  ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 * bq5_1 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 *) vbq;

    ಪೂರ್ಣಾಂಕ vl[VDR_Q5_1_Q8_1_MMVQ];
    ಪೂರ್ಣಾಂಕ vh[VDR_Q5_1_Q8_1_MMVQ];
    ಪೂರ್ಣಾಂಕ  u[2*VDR_Q5_1_Q8_1_MMVQ];

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    ಹಿಂತಿರುಗಿ vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q8_0_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                  ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * bq8_0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) vbq;

    ಪೂರ್ಣಾಂಕ v[VDR_Q8_0_Q8_1_MMVQ];
    ಪೂರ್ಣಾಂಕ u[VDR_Q8_0_Q8_1_MMVQ];

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    ಹಿಂತಿರುಗಿ vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                      bq8_1->ds[0]);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q2_K_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                  ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * bq2_K = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K *) vbq;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bq8_ಸ್ಥಾನಾಂತರ = QR2_K * (iqs / QI8_1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ scale_ಸ್ಥಾನಾಂತರ = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    ಸ್ಥಿರಾಂಕ uint8_t * scales = bq2_K->scales + scale_ಸ್ಥಾನಾಂತರ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    ಪೂರ್ಣಾಂಕ    u[QR2_K];
    ದಶಮಾಂಶ d8[QR2_K];

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_ಸ್ಥಾನಾಂತರ + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_ಸ್ಥಾನಾಂತರ + i].ds[0];
    }

    ಹಿಂತಿರುಗಿ vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q3_K_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                  ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * bq3_K = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K *) vbq;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bq8_ಸ್ಥಾನಾಂತರ = QR3_K * (iqs / (QI3_K/2));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ scale_ಸ್ಥಾನಾಂತರ = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = bq3_K->d;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_ಸ್ಥಾನಾಂತರ;

    ಪೂರ್ಣಾಂಕ    u[QR3_K];
    ದಶಮಾಂಶ d8[QR3_K];

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_ಸ್ಥಾನಾಂತರ + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_ಸ್ಥಾನಾಂತರ + i].ds[0];
    }

    ಹಿಂತಿರುಗಿ vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_ಸ್ಥಾನಾಂತರ, d, d8);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ vec_dot_q4_K_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * __restrict__ bq8_1,
                                               ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_QKK_64

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * bq4_K = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K *) vbq;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ        bq8_ಸ್ಥಾನಾಂತರ = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *      q4         = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) (bq4_K->qs + 16 * bq8_ಸ್ಥಾನಾಂತರ + 4 * ((iqs / 2) % 4));
    ಸ್ಥಿರಾಂಕ uint16_t * scales     = (ಸ್ಥಿರಾಂಕ uint16_t *) bq4_K->scales;

    ಹಿಂತಿರುಗಿ vec_dot_q4_K_q8_1_common(q4, scales, bq4_K->dm, bq8_1, iqs);

#ಇಲ್ಲದಿದ್ದರೆ

#ಆದರೆ __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * bq4_K = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K *) vbq;

    ದಶಮಾಂಶ sumf_d = 0.0f;
    ದಶಮಾಂಶ sumf_m = 0.0f;

    uint16_t aux16[2];
    ಸ್ಥಿರಾಂಕ uint8_t * s = (ಸ್ಥಿರಾಂಕ uint8_t *)aux16;

    ಸ್ಥಿರಾಂಕ uint16_t * a = (ಸ್ಥಿರಾಂಕ uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dall = bq4_K->dm[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ dmin = bq4_K->dm[1];

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d8_1 = bq8_1[0].ds[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d8_2 = bq8_1[1].ds[1];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ui1 = *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)bq8_1[0].qs + (iqs/2));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ui2 = *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)bq8_1[0].qs + (iqs/2) + 4);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ui3 = *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)bq8_1[1].qs + (iqs/2));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ui4 = *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)bq8_1[1].qs + (iqs/2) + 4);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * q4 = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)bq4_K->qs + (iqs/2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ v1 = q4[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ v2 = q4[4];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dot1 = dpct::dp4a(ui2, v2 & 0x0f0f0f0f, dpct::dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dot2 = dpct::dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, dpct::dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dot3 = dpct::dp4a(0x01010101, ui2, dpct::dp4a(0x01010101, ui1, 0));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ dot4 = dpct::dp4a(0x01010101, ui4, dpct::dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    ಹಿಂತಿರುಗಿ dall * sumf_d - dmin * sumf_m;

#ಇಲ್ಲದಿದ್ದರೆ
    bad_arch();
#ಆದರೆ_ಅಂತ್ಯ // __SYCL_ARCH__ >= VER_4VEC

#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q5_K_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                  ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ GGML_QKK_64
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * bq5_K = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K *) vbq;

    ಪೂರ್ಣಾಂಕ   vl[2];
    ಪೂರ್ಣಾಂಕ   vh[2];
    ಪೂರ್ಣಾಂಕ    u[2*QR5_K];
    ದಶಮಾಂಶ d8[QR5_K];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bq8_ಸ್ಥಾನಾಂತರ = QR5_K * ((iqs/2) / (QI8_1/2));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ql = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)(bq5_K->qs + 16 * bq8_ಸ್ಥಾನಾಂತರ + 4 * ((iqs/2)%4));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * qh = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_ಸ್ಥಾನಾಂತರ;
    vh[1] = qh[4] >> bq8_ಸ್ಥಾನಾಂತರ;

    ಸ್ಥಿರಾಂಕ uint16_t * scales = (ಸ್ಥಿರಾಂಕ uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ j = bq8_ಸ್ಥಾನಾಂತರ/2;
    ಆದರೆ (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } ಇಲ್ಲದಿದ್ದರೆ {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    ಸ್ಥಿರಾಂಕ uint8_t * sc = (ಸ್ಥಿರಾಂಕ uint8_t *)aux;
    ಸ್ಥಿರಾಂಕ uint8_t * m  = sc + 2;

#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < QR5_K; ++i) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 * bq8i = bq8_1 + bq8_ಸ್ಥಾನಾಂತರ + i;
        d8[i] = bq8i->ds[0];

        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * q8 = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    ಹಿಂತಿರುಗಿ vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#ಇಲ್ಲದಿದ್ದರೆ

#ಆದರೆ __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K * bq5_K = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K *) vbq;

    ಸ್ಥಿರಾಂಕ int8_t * s = bq5_K->scales;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = bq5_K->d;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d8_1 = bq8_1[0].ds[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d8_2 = bq8_1[1].ds[1];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ui1 = *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)bq8_1[0].qs + (iqs/2));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ui2 = *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)bq8_1[0].qs + (iqs/2) + 4);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ui3 = *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)bq8_1[1].qs + (iqs/2));
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ui4 = *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)bq8_1[1].qs + (iqs/2) + 4);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * ql = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)bq5_K->qs + (iqs/2);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl1 = ql[0];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl2 = ql[4];

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಹೆಜ್ಜೆ = 4 * (iqs/2); // 0, 4, 8, 12
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ im = ಹೆಜ್ಜೆ/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಒಳ = ಹೆಜ್ಜೆ%8; // 0, 4, 0, 4
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vh = (*((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)(bq5_K->qh + ಒಳ))) >> im;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sumf_d = d8_1 * (dpct::dp4a(ui1, v1, 0) * s[0] + dpct::dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (dpct::dp4a(ui3, v3, 0) * s[2] + dpct::dp4a(ui4, v4, 0) * s[3]);

    ಹಿಂತಿರುಗಿ d * sumf_d;

#ಇಲ್ಲದಿದ್ದರೆ
    bad_arch();
#ಆದರೆ_ಅಂತ್ಯ // __SYCL_ARCH__ >= VER_4VEC

#ಆದರೆ_ಅಂತ್ಯ
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_q6_K_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                  ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * bq6_K = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K *) vbq;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ bq8_ಸ್ಥಾನಾಂತರ = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ scale_ಸ್ಥಾನಾಂತರ = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vh_ಸರಿಸು = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vl = get_int_from_uint8(bq6_K->ql, iqs);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_ಸರಿಸು;

    ಸ್ಥಿರಾಂಕ int8_t * scales = bq6_K->scales + scale_ಸ್ಥಾನಾಂತರ;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ u0 = get_int_from_int8_aligned(
        bq8_1[bq8_ಸ್ಥಾನಾಂತರ + 0].qs, iqs % QI8_1);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ u1 = get_int_from_int8_aligned(
        bq8_1[bq8_ಸ್ಥಾನಾಂತರ + 2].qs, iqs % QI8_1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d80 = bq8_1[bq8_ಸ್ಥಾನಾಂತರ + 0].ds[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d81 = bq8_1[bq8_ಸ್ಥಾನಾಂತರ + 2].ds[0];

    ಹಿಂತಿರುಗಿ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u0, u1, scales[0], scales[4], bq6_K->d, d80, d81);
}


// NOTE: the VDR_IQ*_Q8_1_MMVQ values deliberately differ from the identically named CUDA constants
// (vecdotq.cuh): the SYCL kernels pair them with a halved qi (e.g. QI3_S/2), so the values are not
// interchangeable and must not be copied across backends.
#ವ್ಯಾಖ್ಯಾನಿಸು VDR_IQ2_XXS_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ uint32_t iq_sign_ಮುಖವಾಡ(ಸ್ಥಿರಾಂಕ uint32_t signs) {
    ಸ್ಥಿರಾಂಕ uint32_t ಬಿಟ್ಟುಗಳು =
        (signs & 0x01) | ((signs & 0x02) << 7) |
        ((signs & 0x04) << 14) | ((signs & 0x08) << 21);

    ಹಿಂತಿರುಗಿ ಬಿಟ್ಟುಗಳು * 0xff;
}

ಸ್ಥಿರ __dpct_inline__ uint32_t iq_sign_ಬಿಟ್ಟುಗಳು(ಸ್ಥಿರಾಂಕ uint32_t encoded) {
    ಸ್ಥಿರಾಂಕ uint32_t signs = encoded & 127;

    ಹಿಂತಿರುಗಿ signs | ((sycl::popcount(signs) & 1) << 7);
}

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_iq2_xxs_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                     ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs,
                     ಸ್ಥಿರಾಂಕ uint64_t *iq2xxs_grid, ಸ್ಥಿರಾಂಕ uint8_t *ksigns_iq2xs,
                     ಸ್ಥಿರಾಂಕ uint8_t *kmask_iq2xs) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xxs * bq2 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xxs *) vbq;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib32 = iqs;
    ಸ್ಥಿರಾಂಕ uint16_t * q2 = bq2->qs + 4*ib32;
    ಸ್ಥಿರಾಂಕ uint8_t  * aux8 = (ಸ್ಥಿರಾಂಕ uint8_t *)q2;
    ಸ್ಥಿರಾಂಕ int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = q2[2] | (q2[3] << 16);
    ಪೂರ್ಣಾಂಕ sumi = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
        ಸ್ಥಿರಾಂಕ uint8_t * grid = (ಸ್ಥಿರಾಂಕ uint8_t *)(iq2xxs_grid + aux8[l]);
        ಸ್ಥಿರಾಂಕ uint8_t  signs = ksigns_iq2xs[aux32 & 127];
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
            sumi += q8[j] * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
        aux32 >>= 7;
    }
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.25f;
    ಹಿಂತಿರುಗಿ d * sumi;
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
    ಹಿಂತಿರುಗಿ 0.f;
#ಆದರೆ_ಅಂತ್ಯ
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_IQ2_XS_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_iq2_xs_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs,
                    ಸ್ಥಿರಾಂಕ uint64_t *iq2xs_grid) {
#ಆದರೆ DPCT_COMPATIBILITY_ತಾತ್ಕಾಲಿಕ >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xs * bq2 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_xs *) vbq;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib32 = iqs;
    ಸ್ಥಿರಾಂಕ uint16_t * q2 = bq2->qs + 4*ib32;
    ಸ್ಥಿರಾಂಕ int8_t   * q8 = bq8_1[ib32].qs;
    ಸ್ಥಿರಾಂಕ uint8_t ls1 = bq2->scales[ib32] & 0xf;
    ಸ್ಥಿರಾಂಕ uint8_t ls2 = bq2->scales[ib32] >>  4;
    ಪೂರ್ಣಾಂಕ sumi1 = 0;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 2; ++l) {
        ಸ್ಥಿರಾಂಕ uint64_t grid = iq2xs_grid[q2[l] & 511];
        ಸ್ಥಿರಾಂಕ uint32_t sign_ಬಿಟ್ಟುಗಳು = iq_sign_ಬಿಟ್ಟುಗಳು(q2[l] >> 9);
        ಸ್ಥಿರಾಂಕ uint32_t signs0 = iq_sign_ಮುಖವಾಡ(sign_ಬಿಟ್ಟುಗಳು);
        ಸ್ಥಿರಾಂಕ uint32_t signs1 = iq_sign_ಮುಖವಾಡ(sign_ಬಿಟ್ಟುಗಳು >> 4);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi1 = dpct::dp4a(grid_l, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)q8 + 1), sumi1);
        q8 += 8;
    }
    ಪೂರ್ಣಾಂಕ sumi2 = 0;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 2; l < 4; ++l) {
        ಸ್ಥಿರಾಂಕ uint64_t grid = iq2xs_grid[q2[l] & 511];
        ಸ್ಥಿರಾಂಕ uint32_t sign_ಬಿಟ್ಟುಗಳು = iq_sign_ಬಿಟ್ಟುಗಳು(q2[l] >> 9);
        ಸ್ಥಿರಾಂಕ uint32_t signs0 = iq_sign_ಮುಖವಾಡ(sign_ಬಿಟ್ಟುಗಳು);
        ಸ್ಥಿರಾಂಕ uint32_t signs1 = iq_sign_ಮುಖವಾಡ(sign_ಬಿಟ್ಟುಗಳು >> 4);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi2 = dpct::dp4a(grid_l, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)q8 + 1), sumi2);
        q8 += 8;
    }
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    ಹಿಂತಿರುಗಿ d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
    ಹಿಂತಿರುಗಿ 0.f;
#ಆದರೆ_ಅಂತ್ಯ
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
    ಹಿಂತಿರುಗಿ 0.f;
#ಆದರೆ_ಅಂತ್ಯ
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_IQ2_S_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_iq2_s_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                   ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_s * bq2 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq2_s *) vbq;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib32 = iqs;
    ಸ್ಥಿರಾಂಕ int8_t  * q8 = bq8_1[ib32].qs;
    ಸ್ಥಿರಾಂಕ uint8_t * signs = bq2->qs + QK_K/8 + 4*ib32;
    ಸ್ಥಿರಾಂಕ uint8_t ls1 = bq2->scales[ib32] & 0xf;
    ಸ್ಥಿರಾಂಕ uint8_t ls2 = bq2->scales[ib32] >>  4;
    ಪೂರ್ಣಾಂಕ sumi1 = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 2; ++l) {
        ಸ್ಥಿರಾಂಕ uint32_t * grid = (ಸ್ಥಿರಾಂಕ uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        ಸ್ಥಿರಾಂಕ uint32_t signs0 = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ಸ್ಥಿರಾಂಕ uint32_t signs1 = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_l = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_h = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi1 = dpct::dp4a(grid_l, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)q8 + 1), sumi1);
        q8 += 8;
    }
    ಪೂರ್ಣಾಂಕ sumi2 = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 2; l < 4; ++l) {
        ಸ್ಥಿರಾಂಕ uint32_t * grid = (ಸ್ಥಿರಾಂಕ uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        ಸ್ಥಿರಾಂಕ uint32_t signs0 = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ಸ್ಥಿರಾಂಕ uint32_t signs1 = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_l = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_h = dpct::vectorized_ದ್ವಿಮಾನ<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi2 = dpct::dp4a(grid_l, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)q8 + 1), sumi2);
        q8 += 8;
    }
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    ಹಿಂತಿರುಗಿ d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
#ಆದರೆ_ಅಂತ್ಯ
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_IQ3_XXS_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_iq3_xxs_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                     ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs,
                     ಸ್ಥಿರಾಂಕ uint32_t *iq3xxs_grid) {
#ಆದರೆ DPCT_COMPATIBILITY_ತಾತ್ಕಾಲಿಕ >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs * bq2 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_xxs *) vbq;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib32 = iqs;
    ಸ್ಥಿರಾಂಕ uint8_t  * q3 = bq2->qs + 8*ib32;
    ಸ್ಥಿರಾಂಕ uint16_t * gas = (ಸ್ಥಿರಾಂಕ uint16_t *)(bq2->qs + QK_K/4) + 2*ib32;
    ಸ್ಥಿರಾಂಕ int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = gas[0] | (gas[1] << 16);
    ಪೂರ್ಣಾಂಕ sumi = 0;
#ಸೂಚನೆ unroll
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
        ಸ್ಥಿರಾಂಕ uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
        ಸ್ಥಿರಾಂಕ uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
        ಸ್ಥಿರಾಂಕ uint32_t sign_ಬಿಟ್ಟುಗಳು = iq_sign_ಬಿಟ್ಟುಗಳು(aux32);
        ಸ್ಥಿರಾಂಕ uint32_t signs0 = iq_sign_ಮುಖವಾಡ(sign_ಬಿಟ್ಟುಗಳು);
        ಸ್ಥಿರಾಂಕ uint32_t signs1 = iq_sign_ಮುಖವಾಡ(sign_ಬಿಟ್ಟುಗಳು >> 4);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)q8 + 1), sumi);
        q8 += 8;
        aux32 >>= 7;
    }
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.5f;
    ಹಿಂತಿರುಗಿ d * sumi;
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
    ಹಿಂತಿರುಗಿ 0.f;
#ಆದರೆ_ಅಂತ್ಯ
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
    ಹಿಂತಿರುಗಿ 0.f;
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <> ರಚನೆ reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_XXS> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_IQ3_XXS;

    __dpct_inline__ ದಶಮಾಂಶ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> ibx_ಸ್ಥಾನಾಂತರ,
                                     ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> d_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr,
                                     ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
#ಆದರೆ DPCT_COMPATIBILITY_ತಾತ್ಕಾಲಿಕ >= MIN_CC_DP4A && QK_K == 256
        ಸ್ಥಿರಾಂಕ uint8_t * ತಳ = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vbq);
        ಸ್ಥಿರಾಂಕ uint8_t * qs   = ತಳ + ibx_ಸ್ಥಾನಾಂತರ.ಮೊದಲ;
        ಸ್ಥಿರಾಂಕ uint8_t * q3   = qs + 8 * iqs;
        ಸ್ಥಿರಾಂಕ uint16_t * gas = reinterpret_cast<ಸ್ಥಿರಾಂಕ uint16_t *>(qs + QK_K / 4) + 2 * iqs;
        ಸ್ಥಿರಾಂಕ int8_t * q8    = q8_1_quant_ptr + iqs * QK8_1;
        uint32_t aux32       = gas[0] | (gas[1] << 16);

        ಪೂರ್ಣಾಂಕ sumi = 0;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
            ಸ್ಥಿರಾಂಕ uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
            ಸ್ಥಿರಾಂಕ uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
            ಸ್ಥಿರಾಂಕ uint32_t sign_ಬಿಟ್ಟುಗಳು = iq_sign_ಬಿಟ್ಟುಗಳು(aux32);
            ಸ್ಥಿರಾಂಕ uint32_t signs0 = iq_sign_ಮುಖವಾಡ(sign_ಬಿಟ್ಟುಗಳು);
            ಸ್ಥಿರಾಂಕ uint32_t signs1 = iq_sign_ಮುಖವಾಡ(sign_ಬಿಟ್ಟುಗಳು >> 4);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) q8 + 1), sumi);
            q8 += 8;
            aux32 >>= 7;
        }

        ಸ್ಥಿರಾಂಕ ggml_half d = *reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half *>(ತಳ + d_ಸ್ಥಾನಾಂತರ.ಮೊದಲ);
        ಹಿಂತಿರುಗಿ (ದಶಮಾಂಶ) d * (0.5f + aux32) * q8_1_ds[iqs][0] * 0.5f * sumi;
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_ಬಳಸದ(vbq);
        GGML_ಬಳಸದ(ibx_ಸ್ಥಾನಾಂತರ);
        GGML_ಬಳಸದ(d_ಸ್ಥಾನಾಂತರ);
        GGML_ಬಳಸದ(q8_1_quant_ptr);
        GGML_ಬಳಸದ(q8_1_ds);
        GGML_ಬಳಸದ(iqs);
        GGML_ABORT("IQ3_XXS reorder vec dot requires dp4a and QK_K == 256");
#ಆದರೆ_ಅಂತ್ಯ
    }
};

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_IQ3_S_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_iq3_s_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                   ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs,
                   ಸ್ಥಿರಾಂಕ uint32_t *iq3s_grid) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s * bq2 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq3_s *) vbq;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib32 = iqs;
    ಸ್ಥಿರಾಂಕ uint8_t  * qs = bq2->qs + 8*ib32;
    ಸ್ಥಿರಾಂಕ int8_t   * q8 = bq8_1[ib32].qs;
    ಪೂರ್ಣಾಂಕ sumi = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
        ಸ್ಥಿರಾಂಕ uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((bq2->qh[ib32] << (8 - 2*l)) & 256)];
        ಸ್ಥಿರಾಂಕ uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((bq2->qh[ib32] << (7 - 2*l)) & 256)];
        ಸ್ಥಿರಾಂಕ uint32_t signs = bq2->signs[4 * ib32 + l];
        ಸ್ಥಿರಾಂಕ uint32_t signs0 = iq_sign_ಮುಖವಾಡ(signs);
        ಸ್ಥಿರಾಂಕ uint32_t signs1 = iq_sign_ಮುಖವಾಡ(signs >> 4);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)q8 + 1), sumi);
        q8 += 8;
    }
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d =
        (ದಶಮಾಂಶ)bq2->d *
        (1 + 2 * ((bq2->scales[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf)) *
        bq8_1[ib32].ds[0];
    ಹಿಂತಿರುಗಿ d * sumi;
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <> ರಚನೆ reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_S> {
    ಸ್ಥಿರ constexpr ggml_type gtype = GGML_TYPE_IQ3_S;

    __dpct_inline__ ದಶಮಾಂಶ ಕ್ರಿಯಾಕಾರಕ()(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * __restrict__ vbq, ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> ibx_ಸ್ಥಾನಾಂತರ,
                                     ಸ್ಥಿರಾಂಕ std::pair<ಪೂರ್ಣಾಂಕ, ಪೂರ್ಣಾಂಕ> d_ಸ್ಥಾನಾಂತರ, ಸ್ಥಿರಾಂಕ int8_t * q8_1_quant_ptr,
                                     ಸ್ಥಿರಾಂಕ sycl::half2 * q8_1_ds, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ & iqs) {
#ಆದರೆ QK_K == 256
        ಸ್ಥಿರಾಂಕ uint8_t * ತಳ     = static_cast<ಸ್ಥಿರಾಂಕ uint8_t *>(vbq);
        ಸ್ಥಿರಾಂಕ uint8_t * qs       = ತಳ + ibx_ಸ್ಥಾನಾಂತರ.ಮೊದಲ + 8 * iqs;
        ಸ್ಥಿರಾಂಕ uint8_t * qh       = ತಳ + ibx_ಸ್ಥಾನಾಂತರ.second;
        ಸ್ಥಿರಾಂಕ uint8_t * signs    = ತಳ + d_ಸ್ಥಾನಾಂತರ.ಮೊದಲ + 4 * iqs;
        ಸ್ಥಿರಾಂಕ uint8_t * metadata = ತಳ + d_ಸ್ಥಾನಾಂತರ.second;
        ಸ್ಥಿರಾಂಕ ggml_half d        = *reinterpret_cast<ಸ್ಥಿರಾಂಕ ggml_half *>(metadata);
        ಸ್ಥಿರಾಂಕ uint8_t * scales   = metadata + ಗಾತ್ರ(ggml_half);
        ಸ್ಥಿರಾಂಕ int8_t * q8        = q8_1_quant_ptr + iqs * QK8_1;

        ಪೂರ್ಣಾಂಕ sumi = 0;
#ಸೂಚನೆ unroll
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
            ಸ್ಥಿರಾಂಕ uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((qh[iqs] << (8 - 2*l)) & 256)];
            ಸ್ಥಿರಾಂಕ uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((qh[iqs] << (7 - 2*l)) & 256)];
            ಸ್ಥಿರಾಂಕ uint32_t signs0 = iq_sign_ಮುಖವಾಡ(signs[l]);
            ಸ್ಥಿರಾಂಕ uint32_t signs1 = iq_sign_ಮುಖವಾಡ(signs[l] >> 4);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *) q8 + 1), sumi);
            q8 += 8;
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ scale = 1 + 2 * ((scales[iqs / 2] >> (4 * (iqs % 2))) & 0xf);
        ಹಿಂತಿರುಗಿ (ದಶಮಾಂಶ) d * scale * q8_1_ds[iqs][0] * sumi;
#ಇಲ್ಲದಿದ್ದರೆ
        GGML_ಬಳಸದ(vbq);
        GGML_ಬಳಸದ(ibx_ಸ್ಥಾನಾಂತರ);
        GGML_ಬಳಸದ(d_ಸ್ಥಾನಾಂತರ);
        GGML_ಬಳಸದ(q8_1_quant_ptr);
        GGML_ಬಳಸದ(q8_1_ds);
        GGML_ಬಳಸದ(iqs);
        GGML_ABORT("IQ3_S reorder vec dot not supported for QK_K != 256");
#ಆದರೆ_ಅಂತ್ಯ
    }
};

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_IQ1_S_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_iq1_s_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                   ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs,
                   ಸ್ಥಿರಾಂಕ uint32_t *iq1s_grid_gpu) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_s * bq1 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_s *) vbq;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib32 = iqs;
    ಪೂರ್ಣಾಂಕ sumi = 0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * q8 = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)bq8_1[ib32].qs;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * grid = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[ib32] >> 3*l) & 7) << 8)));
        ಪೂರ್ಣಾಂಕ grid0 = grid[0] & 0x0f0f0f0f;
        ಪೂರ್ಣಾಂಕ grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi = dpct::dp4a(q8[2 * l + 1], grid1,
                          dpct::dp4a(q8[2 * l + 0], grid0, sumi));
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ delta = bq1->qh[ib32] & 0x8000 ? -1-IQ1S_DELTA : -1+IQ1S_DELTA;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1q = (ದಶಮಾಂಶ)bq1->d * (2*((bq1->qh[ib32] >> 12) & 7) + 1);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = d1q * bq8_1[ib32].ds[0];
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m = d1q * bq8_1[ib32].ds[1];
    ಹಿಂತಿರುಗಿ d * sumi + m * delta;
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
#ಆದರೆ_ಅಂತ್ಯ
}

#ವ್ಯಾಖ್ಯಾನಿಸು VDR_IQ1_M_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_iq1_m_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                   ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {
#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_m * bq1 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq1_m *) vbq;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib32 = iqs;
    ಪೂರ್ಣಾಂಕ   sumi[2] = {0, 0};
    ದಶಮಾಂಶ sumf[2] = {0.f, 0.f};

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * q8 = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)bq8_1[ib32].qs;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 4; ++l) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ * grid = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 7) << 8)));
        ಪೂರ್ಣಾಂಕ grid0 = grid[0] & 0x0f0f0f0f;
        ಪೂರ್ಣಾಂಕ grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi[l / 2] = dpct::dp4a(q8[2 * l + 1], grid1,
                                 dpct::dp4a(q8[2 * l + 0], grid0, sumi[l / 2]));
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ delta = (bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 0x08 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sumy = dpct::dp4a(q8[2 * l + 1], 0x01010101,
                                    dpct::dp4a(q8[2 * l + 0], 0x01010101, 0));
        sumf[l/2] += delta*sumy;
    }

    iq1m_scale_t scale;
    ಸ್ಥಿರಾಂಕ uint16_t * sc = (ಸ್ಥಿರಾಂಕ uint16_t *)bq1->scales;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)scale.f16 * bq8_1[ib32].ds[0];
    ಹಿಂತಿರುಗಿ d * ((sumi[0] + sumf[0]) * (2*((sc[ib32/2] >> 6*(ib32%2)) & 0x7) + 1) + (sumi[1] + sumf[1]) * (2*((sc[ib32/2] >> (6*(ib32%2)+3)) & 0x7) + 1));
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
#ಆದರೆ_ಅಂತ್ಯ
}


#ವ್ಯಾಖ್ಯಾನಿಸು VDR_IQ4_NL_Q8_1_MMVQ 2

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_iq4_nl_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {

    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl * bq = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_nl *) vbq;

    ಸ್ಥಿರಾಂಕ uint16_t * q4 = (ಸ್ಥಿರಾಂಕ uint16_t *)bq->qs + 2*iqs;
    ಸ್ಥಿರಾಂಕ int32_t  * q8 = (ಸ್ಥಿರಾಂಕ int32_t  *)bq8_1->qs + iqs;

    ಸ್ಥಿರಾಂಕ uint8_t * ಮೌಲ್ಯಗಳು = (ಸ್ಥಿರಾಂಕ uint8_t *)kvalues_iq4nl;

    ಪೂರ್ಣಾಂಕ v1, v2;
    ಪೂರ್ಣಾಂಕ sumi1 = 0, sumi2 = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        ಸ್ಥಿರಾಂಕ uint32_t aux = q4[2*l] | (q4[2*l+1] << 16);
        get_int_from_ಕೋಷ್ಟಕ_16(aux, ಮೌಲ್ಯಗಳು, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[l + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[l + 4], sumi2);
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)bq->d * bq8_1->ds[0];
    ಹಿಂತಿರುಗಿ d * (sumi1 + sumi2);
}


#ವ್ಯಾಖ್ಯಾನಿಸು VDR_IQ4_XS_Q8_1_MMVQ 1

ಸ್ಥಿರ __dpct_inline__ ದಶಮಾಂಶ
vec_dot_iq4_xs_q8_1(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *__restrict__ vbq,
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_1 *__restrict__ bq8_1, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ &iqs) {

#ಆದರೆ QK_K == 256
    ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs * bq4 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_iq4_xs *) vbq;
    ಸ್ಥಿರಾಂಕ uint8_t * ಮೌಲ್ಯಗಳು = (ಸ್ಥಿರಾಂಕ uint8_t *)kvalues_iq4nl;

    // iqs is 0...7
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ib32 = iqs;
    ಸ್ಥಿರಾಂಕ int32_t  * q8 = (ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ *)bq8_1[ib32].qs;
    ಸ್ಥಿರಾಂಕ uint32_t * q4 = (ಸ್ಥಿರಾಂಕ uint32_t *)bq4->qs + 4*ib32;
    ಸ್ಥಿರಾಂಕ int8_t ls = ((bq4->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((bq4->scales_h >> 2*ib32) & 3) << 4);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = (ದಶಮಾಂಶ)bq4->d * (ls - 32) * bq8_1[ib32].ds[0];
    ಪೂರ್ಣಾಂಕ v1, v2;
    ಪೂರ್ಣಾಂಕ sumi1 = 0, sumi2 = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; ++j) {
        get_int_from_ಕೋಷ್ಟಕ_16(q4[j], ಮೌಲ್ಯಗಳು, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[j + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[j + 4], sumi2);
    }
    ಹಿಂತಿರುಗಿ d * (sumi1 + sumi2);
#ಇಲ್ಲದಿದ್ದರೆ
    ಪ್ರತಿಜ್ಞಾ(ಸುಳ್ಳು);
#ಆದರೆ_ಅಂತ್ಯ
}

#ಆದರೆ_ಅಂತ್ಯ // GGML_SYCL_VECDOTQ_HPP
