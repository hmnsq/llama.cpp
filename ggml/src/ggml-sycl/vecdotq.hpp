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

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_VECDOTQ_HPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_SYCL_VECDOTQ_HPP

#ਅਨ੍ਤਰ੍ਭਾਵ "dpct/helper.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "type.hpp"
#ਅਨ੍ਤਰ੍ਭਾਵ "quants.hpp"

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਦਸ਼ਮਲਵ (*vec_dot_q_sycl_t)(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ ਖੰਡ_q8_1 * __restrict__ bq8_1,
                                  ਅਚਲ ਪੂਰਨ & iqs);

ਸਥਿਰ __dpct_inline__ ਪੂਰਨ get_int_b1(ਅਚਲ ਖਾਲੀ * x, ਅਚਲ ਪੂਰਨ & i32) {
    ਅਚਲ uint8_t * x8 = (ਅਚਲ uint8_t *) x;

    ਪੂਰਨ x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    ਵਾਪਸ x32;
}

ਸਥਿਰ __dpct_inline__ ਪੂਰਨ get_int_b2(ਅਚਲ ਖਾਲੀ * x, ਅਚਲ ਪੂਰਨ & i32) {
    ਅਚਲ uint16_t * x16 = (ਅਚਲ uint16_t *) x; // assume at least 2 byte alignment

    ਪੂਰਨ x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    ਵਾਪਸ x32;
}

ਸਥਿਰ __dpct_inline__ ਪੂਰਨ get_int_b4(ਅਚਲ ਖਾਲੀ * x, ਅਚਲ ਪੂਰਨ & i32) {
    ਵਾਪਸ ((ਅਚਲ ਪੂਰਨ *) x)[i32]; // assume at least 4 byte alignment
}

ਸਥਿਰ __dpct_inline__ ਪੂਰਨ get_int_from_int8(ਅਚਲ int8_t* x8, ਅਚਲ ਪੂਰਨ& i32) {
  ਅਚਲ uint16_t* x16 =
      (ਅਚਲ uint16_t*)(x8 + ਆਕਾਰ(ਪੂਰਨ) * i32); // assume at least 2 byte
                                                 // alignment

  ਪੂਰਨ x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  ਵਾਪਸ x32;
}

ਸਥਿਰ __dpct_inline__ ਪੂਰਨ get_int_from_uint8(
    ਅਚਲ uint8_t* x8,
    ਅਚਲ ਪੂਰਨ& i32) {
  ਅਚਲ uint16_t* x16 =
      (ਅਚਲ uint16_t*)(x8 + ਆਕਾਰ(ਪੂਰਨ) * i32); // assume at least 2 byte
                                                 // alignment

  ਪੂਰਨ x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  ਵਾਪਸ x32;
}

ਸਥਿਰ __dpct_inline__ ਪੂਰਨ get_int_from_int8_aligned(
    ਅਚਲ int8_t* x8,
    ਅਚਲ ਪੂਰਨ& i32) {
  ਵਾਪਸ *(
      (ਅਚਲ ਪੂਰਨ*)(x8 + ਆਕਾਰ(ਪੂਰਨ) * i32)); // assume at least 4 byte alignment
}

ਸਥਿਰ __dpct_inline__ ਪੂਰਨ get_int_from_uint8_aligned(
    ਅਚਲ uint8_t* x8,
    ਅਚਲ ਪੂਰਨ& i32) {
  ਵਾਪਸ *(
      (ਅਚਲ ਪੂਰਨ*)(x8 + ਆਕਾਰ(ਪੂਰਨ) * i32)); // assume at least 4 byte alignment
}

ਸਥਿਰ __dpct_inline__ ਪੂਰਨ byte_sub_4(ਅਚਲ ਪੂਰਨ a, ਅਚਲ ਪੂਰਨ b) {
  ਅਚਲ uint32_t ua = static_cast<uint32_t>(a);
  ਅਚਲ uint32_t ub = static_cast<uint32_t>(b);
  ਵਾਪਸ static_cast<ਪੂਰਨ>(((ua | 0x80808080u) - ub) ^ 0x80808080u);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
    ਅਚਲ ਪੂਰਨ vl, ਅਚਲ ਪੂਰਨ vh, ਅਚਲ ਪੂਰਨ u0, ਅਚਲ ਪੂਰਨ u1, ਅਚਲ int8_t sc0,
    ਅਚਲ int8_t sc1, ਅਚਲ ਦਸ਼ਮਲਵ d, ਅਚਲ ਦਸ਼ਮਲਵ d80, ਅਚਲ ਦਸ਼ਮਲਵ d81) {
    static_assert(QR6_K == 2, "q6_K MMVQ scalar fast path assumes QR6_K == 2");

    ਅਚਲ ਪੂਰਨ vil0 = (vl >> 0) & 0x0F0F0F0F;
    ਅਚਲ ਪੂਰਨ vih0 = ((vh >> 0) << 4) & 0x30303030;
    ਅਚਲ ਪੂਰਨ vi0 = byte_sub_4(vil0 | vih0, 0x20202020);

    ਅਚਲ ਪੂਰਨ vil1 = (vl >> 4) & 0x0F0F0F0F;
    ਅਚਲ ਪੂਰਨ vih1 = ((vh >> 4) << 4) & 0x30303030;
    ਅਚਲ ਪੂਰਨ vi1 = byte_sub_4(vil1 | vih1, 0x20202020);

    ਅਚਲ ਦਸ਼ਮਲਵ sumf =
        d80 * (dpct::dp4a(vi0, u0, 0) * sc0) +
        d81 * (dpct::dp4a(vi1, u1, 0) * sc1);

    ਵਾਪਸ d * sumf;
}

ਸਥਿਰ __dpct_inline__ ਖਾਲੀ get_int_from_ਸਾਰਣੀ_16(ਅਚਲ uint32_t &q4,
                                                  ਅਚਲ uint8_t *ਮੁੱਲਾਂ,
                                                  ਪੂਰਨ &val1, ਪੂਰਨ &val2) {

    uint32_t aux32; ਅਚਲ uint8_t * q8 = (ਅਚਲ uint8_t *)&aux32;
    aux32 = q4 & 0x0f0f0f0f;
    uint16_t v1 = ਮੁੱਲਾਂ[q8[0]] | (ਮੁੱਲਾਂ[q8[1]] << 8);
    uint16_t v2 = ਮੁੱਲਾਂ[q8[2]] | (ਮੁੱਲਾਂ[q8[3]] << 8);
    val1 = v1 | (v2 << 16);
    aux32 = (q4 >> 4) & 0x0f0f0f0f;
    v1 = ਮੁੱਲਾਂ[q8[0]] | (ਮੁੱਲਾਂ[q8[1]] << 8);
    v2 = ਮੁੱਲਾਂ[q8[2]] | (ਮੁੱਲਾਂ[q8[3]] << 8);
    val2 = v1 | (v2 << 16);
}

ਸਥਿਰ __dpct_inline__ sycl::int2 get_int_from_ਸਾਰਣੀ_16(
    ਅਚਲ ਪੂਰਨ& q4, ਅਚਲ int8_t* ਸਾਰਣੀ) {
  ਅਚਲ uint32_t* table32 = (ਅਚਲ uint32_t*)ਸਾਰਣੀ;
  uint32_t tmp[2];
  ਅਚਲ uint32_t ਨੀਵਾਂ_ਉੱਚਾ_selection_indices =
      (0x32103210 | ((q4 & 0x88888888) >> 1));
#ਹਦਾਇਤ unroll
  ਲਈ (uint32_t i = 0; i < 2; ++i) {
    ਅਚਲ uint32_t ਖਿਸਕਾਓ = 16 * i;

    ਅਚਲ uint32_t ਨੀਵਾਂ =
        dpct::byte_level_ਕ੍ਰਮ_ਬਦਲੋ(table32[0], table32[1], q4 >> ਖਿਸਕਾਓ);
    ਅਚਲ uint32_t ਉੱਚਾ =
        dpct::byte_level_ਕ੍ਰਮ_ਬਦਲੋ(table32[2], table32[3], q4 >> ਖਿਸਕਾਓ);
    tmp[i] = dpct::byte_level_ਕ੍ਰਮ_ਬਦਲੋ(
        ਨੀਵਾਂ, ਉੱਚਾ, ਨੀਵਾਂ_ਉੱਚਾ_selection_indices >> ਖਿਸਕਾਓ);
  }
  ਵਾਪਸ sycl::int2(
      dpct::byte_level_ਕ੍ਰਮ_ਬਦਲੋ(tmp[0], tmp[1], 0x6420),
      dpct::byte_level_ਕ੍ਰਮ_ਬਦਲੋ(tmp[0], tmp[1], 0x7531));
}

// Four E2M1 codes (one per byte, bits 0..3) to their kvalues_mxfp4 int8 values. SWAR arithmetic
// replaces get_int_from_table_16 for MXFP4: dpct::byte_level_permute is emulated with 64-bit shifts,
// eight per int, which made the MXFP4 GEMV compute-bound on Intel GPUs.
// Magnitudes 0,1,2,3,4,6,8,12 = m + [m>=5] + [m>=6] + 3*[m>=7]; each byte stays below 256, so the
// byte-wise adds never carry. -0 (code 8) is left as 0 so the two's-complement +1 cannot carry either.
ਸਥਿਰ __dpct_inline__ ਪੂਰਨ mxfp4_codes_ਮੰਜ਼ਿਲ_int8(ਅਚਲ uint32_t x) {
    ਅਚਲ uint32_t m   = x & 0x07070707u;
    ਅਚਲ uint32_t ge5 = ((m + 0x03030303u) >> 3) & 0x01010101u;
    ਅਚਲ uint32_t ge6 = ((m + 0x02020202u) >> 3) & 0x01010101u;
    ਅਚਲ uint32_t ge7 = ((m + 0x01010101u) >> 3) & 0x01010101u;
    ਅਚਲ uint32_t mag = m + ge5 + ge6 + 3u * ge7;
    ਅਚਲ uint32_t nz  = ((mag + 0x7f7f7f7fu) >> 7) & 0x01010101u;
    ਅਚਲ uint32_t neg = (x >> 3) & nz & 0x01010101u;
    ਵਾਪਸ (ਪੂਰਨ) ((mag ^ (neg * 0xffu)) + neg);
}

// Same result as get_int_from_table_16(q4, kvalues_mxfp4): x = low nibbles, y = high nibbles.
ਸਥਿਰ __dpct_inline__ sycl::int2 get_int_from_mxfp4(ਅਚਲ ਪੂਰਨ q4) {
    ਵਾਪਸ sycl::int2(mxfp4_codes_ਮੰਜ਼ਿਲ_int8((uint32_t) q4 & 0x0f0f0f0fu),
                      mxfp4_codes_ਮੰਜ਼ਿਲ_int8(((uint32_t) q4 >> 4) & 0x0f0f0f0fu));
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q2_K_Q8_1_MMVQ 1

// contiguous v/x values
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q2_K_q8_1_impl_mmvq(
    ਅਚਲ ਪੂਰਨ &v, ਅਚਲ ਪੂਰਨ *__restrict__ u, ਅਚਲ uint8_t *__restrict__ scales,
    ਅਚਲ sycl::half2 &dm2, ਅਚਲ ਦਸ਼ਮਲਵ *__restrict__ d8) {

    ਦਸ਼ਮਲਵ sumf_d = 0.0f;
    ਦਸ਼ਮਲਵ sumf_m = 0.0f;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < QR2_K; ++i) {
        ਅਚਲ ਪੂਰਨ sc = scales[2*i];

        ਅਚਲ ਪੂਰਨ vi = (v >> (2*i)) & 0x03030303;

        sumf_d +=
            d8[i] * (dpct::dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        ਪੂਰਨ m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] *
                  dpct::dp4a(
                      m, u[i],
                      0); // multiply constant q2_K part with sum of q8_1 values
    }

    ਅਚਲ sycl::float2 dm2f =
        dm2.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

    ਵਾਪਸ dm2f.x() * sumf_d - dm2f.y() * sumf_m;
}


#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q3_K_Q8_1_MMVQ 1

// contiguous v/x values
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q3_K_q8_1_impl_mmvq(
    ਅਚਲ ਪੂਰਨ &vl, ਅਚਲ ਪੂਰਨ &vh, ਅਚਲ ਪੂਰਨ *__restrict__ u,
    ਅਚਲ uint8_t *__restrict__ scales, ਅਚਲ ਪੂਰਨ &scale_ਵਿੱਥ,
    ਅਚਲ ਦਸ਼ਮਲਵ &d3, ਅਚਲ ਦਸ਼ਮਲਵ *__restrict__ d8) {

    ਦਸ਼ਮਲਵ sumf = 0.0f;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < QR3_K; ++i) {
        ਅਚਲ ਪੂਰਨ isc = scale_ਵਿੱਥ + 2*i;

        ਅਚਲ ਪੂਰਨ isc_ਨੀਵਾਂ = isc % (QK_K/32);
        ਅਚਲ ਪੂਰਨ sc_ਖਿਸਕਾਓ_ਨੀਵਾਂ = 4 * (isc / (QK_K/32));
        ਅਚਲ ਪੂਰਨ sc_ਨੀਵਾਂ  = (scales[isc_ਨੀਵਾਂ] >> sc_ਖਿਸਕਾਓ_ਨੀਵਾਂ) & 0xF;

        ਅਚਲ ਪੂਰਨ isc_ਉੱਚਾ = isc % (QK_K/64);
        ਅਚਲ ਪੂਰਨ sc_ਖਿਸਕਾਓ_ਉੱਚਾ = 2 * (isc / (QK_K/64));
        ਅਚਲ ਪੂਰਨ sc_ਉੱਚਾ = ((scales[(QK_K/32) + isc_ਉੱਚਾ] >> sc_ਖਿਸਕਾਓ_ਉੱਚਾ) & 3) << 4;

        ਅਚਲ ਪੂਰਨ sc = (sc_ਨੀਵਾਂ | sc_ਉੱਚਾ) - 32;

        ਅਚਲ ਪੂਰਨ vil = (vl >> (2*i)) & 0x03030303;

        ਅਚਲ ਪੂਰਨ vih = ((vh >> i) << 2) & 0x04040404;

        ਅਚਲ ਪੂਰਨ vi =
            dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::char4>(vil, vih, dpct::sub_sat());

        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    ਵਾਪਸ d3 * sumf;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q4_K_Q8_1_MMVQ 2

// contiguous v/x values
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q4_K_q8_1_impl_vmmq(
    ਅਚਲ ਪੂਰਨ *__restrict__ v, ਅਚਲ ਪੂਰਨ *__restrict__ u,
    ਅਚਲ uint8_t *__restrict__ sc, ਅਚਲ uint8_t *__restrict__ m,
    ਅਚਲ sycl::half2 &dm4, ਅਚਲ ਦਸ਼ਮਲਵ *__restrict__ d8) {

    ਦਸ਼ਮਲਵ sumf_d = 0.0f;
    ਦਸ਼ਮਲਵ sumf_m = 0.0f;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < QR4_K; ++i) {
        ਅਚਲ ਪੂਰਨ v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        ਅਚਲ ਪੂਰਨ v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        ਅਚਲ ਪੂਰਨ dot1 =
            dpct::dp4a(v1i, u[2 * i + 1],
                       dpct::dp4a(v0i, u[2 * i + 0], 0)); // SIMD dot product
        ਅਚਲ ਪੂਰਨ dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 1],
                       dpct::dp4a(0x01010101, u[2 * i + 0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    ਅਚਲ sycl::float2 dm4f =
        dm4.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

    ਵਾਪਸ dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q5_K_Q8_1_MMVQ 2

// contiguous v/x values
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q5_K_q8_1_impl_vmmq(
    ਅਚਲ ਪੂਰਨ *__restrict__ vl, ਅਚਲ ਪੂਰਨ *__restrict__ vh,
    ਅਚਲ ਪੂਰਨ *__restrict__ u, ਅਚਲ uint8_t *__restrict__ sc,
    ਅਚਲ uint8_t *__restrict__ m, ਅਚਲ sycl::half2 &dm5,
    ਅਚਲ ਦਸ਼ਮਲਵ *__restrict__ d8) {

    ਦਸ਼ਮਲਵ sumf_d = 0.0f;
    ਦਸ਼ਮਲਵ sumf_m = 0.0f;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < QR5_K; ++i) {
        ਅਚਲ ਪੂਰਨ vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        ਅਚਲ ਪੂਰਨ vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        ਅਚਲ ਪੂਰਨ vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        ਅਚਲ ਪੂਰਨ vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        ਅਚਲ ਪੂਰਨ v0i = vl0i | vh0i;
        ਅਚਲ ਪੂਰਨ v1i = vl1i | vh1i;

        ਅਚਲ ਪੂਰਨ dot1 =
            dpct::dp4a(v0i, u[2 * i + 0],
                       dpct::dp4a(v1i, u[2 * i + 1], 0)); // SIMD dot product
        ਅਚਲ ਪੂਰਨ dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 0],
                       dpct::dp4a(0x01010101, u[2 * i + 1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    ਅਚਲ sycl::float2 dm5f =
        dm5.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

    ਵਾਪਸ dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}


#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q6_K_Q8_1_MMVQ 1

// contiguous v/x values
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q6_K_q8_1_impl_mmvq(ਅਚਲ ਪੂਰਨ &vl, ਅਚਲ ਪੂਰਨ &vh,
                            ਅਚਲ ਪੂਰਨ *__restrict__ u,
                            ਅਚਲ int8_t *__restrict__ scales, ਅਚਲ ਦਸ਼ਮਲਵ &d,
                            ਅਚਲ ਦਸ਼ਮਲਵ *__restrict__ d8) {
    ਵਾਪਸ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q1_0_Q8_1_MMVQ 1
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q1_0_Q8_1_MMQ  4

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q1_0_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                  ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {

    ਅਚਲ ਖੰਡ_q1_0 * bq1_0 = (ਅਚਲ ਖੰਡ_q1_0 *) vbq;

    ਅਚਲ ਖੰਡ_q8_1 * bq8_1_chunk = bq8_1 + iqs;
    ਅਚਲ ਦਸ਼ਮਲਵ        d1          = bq1_0->d;
    ਅਚਲ ਪੂਰਨ          v           = get_int_from_uint8_aligned(bq1_0->qs, iqs);

    ਪੂਰਨ vi_ਬਾਈਟਾਂ[8];
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
        ਅਚਲ ਪੂਰਨ ਖਿਸਕਾਓ = j * 4;
        ਅਚਲ ਪੂਰਨ bits4 = (v >> ਖਿਸਕਾਓ) & 0x0F;
        ਅਚਲ ਪੂਰਨ b0    = (bits4 & 0x01) ? 1 : -1;
        ਅਚਲ ਪੂਰਨ b1    = (bits4 & 0x02) ? 1 : -1;
        ਅਚਲ ਪੂਰਨ b2    = (bits4 & 0x04) ? 1 : -1;
        ਅਚਲ ਪੂਰਨ b3    = (bits4 & 0x08) ? 1 : -1;
        vi_ਬਾਈਟਾਂ[j]     = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }

    ਪੂਰਨ sumi = 0;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
        ਅਚਲ ਪੂਰਨ u = get_int_from_int8_aligned(bq8_1_chunk->qs, j);
        sumi        = ggml_sycl_dp4a(vi_ਬਾਈਟਾਂ[j], u, sumi);
    }

    ਵਾਪਸ d1 * bq8_1_chunk->ds[0] * sumi;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

ਸਾਂਚਾ <ggml_type T> ਬਣਤਰ reorder_vec_dot_q_sycl {
    static_assert(T != T, "ggml_type for reorder vecdot not implemented");
};

// For some types the weight side of the dot product does not depend on the destination column, so a
// multi-column mul_mat_vec can unpack it once per block instead of once per column. Such a type adds
// load() and dot() next to operator() and opts in here. See reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>.
ਸਾਂਚਾ <ggml_type T> ਬਣਤਰ reorder_vec_dot_shared_weights {
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ value = ਝੂਠ;
};

ਸਾਂਚਾ <> ਬਣਤਰ reorder_vec_dot_shared_weights<GGML_TYPE_Q4_K> {
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ value = ਸੱਚ;
};

ਸਾਂਚਾ <> ਬਣਤਰ reorder_vec_dot_shared_weights<GGML_TYPE_Q5_K> {
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ value = ਸੱਚ;
};

ਸਾਂਚਾ <ggml_type T> ਬਣਤਰ reorder_vec_dot_shared_activations {
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ value = ਝੂਠ;
};

ਸਾਂਚਾ <> ਬਣਤਰ reorder_vec_dot_shared_activations<GGML_TYPE_Q4_K> {
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ value = ਸੱਚ;
};

ਸਾਂਚਾ <> ਬਣਤਰ reorder_vec_dot_shared_activations<GGML_TYPE_Q5_K> {
    ਸਥਿਰ constexpr ਤਾਰਕਿਕ value = ਸੱਚ;
};

ਸਾਂਚਾ <> ਬਣਤਰ reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0> {
    ਸਥਿਰ constexpr ggml_type gtype = GGML_TYPE_Q4_0;

    ਵਰਤੋਂ q4_0_ਖੰਡ  = ggml_sycl_reordered::ਖੰਡ_q_t<GGML_TYPE_Q4_0>;
    ਵਰਤੋਂ q4_0_traits = ਕਿਸਮ_ਨਾਂ q4_0_ਖੰਡ::traits;

    __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q4_0_q8_1_impl(ਅਚਲ ਪੂਰਨ * v, ਅਚਲ ਪੂਰਨ * u, ਅਚਲ ਦਸ਼ਮਲਵ & d4, ਅਚਲ sycl::half2 & ds8) {
        ਪੂਰਨ sumi = 0;

#ਹਦਾਇਤ unroll
        ਲਈ (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            ਅਚਲ ਪੂਰਨ vi0 = (v[i] >> 0) & 0x0F0F0F0F;
            ਅਚਲ ਪੂਰਨ vi1 = (v[i] >> 4) & 0x0F0F0F0F;

            // SIMD dot product of quantized values
            sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
            sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
        }

        ਅਚਲ sycl::float2 ds8f = ds8.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

        // second part effectively subtracts 8 from each quant value
        ਵਾਪਸ d4 * (sumi * ds8f.x() - (8 * q4_0_traits::vdr_mmvq / q4_0_traits::qi) * ds8f.y());
    }

    __dpct_inline__ ਦਸ਼ਮਲਵ ਸੰਚਾਲਕ()(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> ibx_ਵਿੱਥ,
                                     ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> d_ਵਿੱਥ, ਅਚਲ int8_t * q8_1_quant_ptr,
                                     ਅਚਲ sycl::half2 * q8_1_ds, ਅਚਲ ਪੂਰਨ & iqs) {
        ਅਚਲ uint8_t * bq4_0 = static_cast<ਅਚਲ uint8_t *>(vbq) + ibx_ਵਿੱਥ.ਪਹਿਲਾ;
        ਅਚਲ ggml_half d = *(reinterpret_cast<ਅਚਲ ggml_half *>(static_cast<ਅਚਲ uint8_t *>(vbq) + d_ਵਿੱਥ.ਪਹਿਲਾ));
        ਪੂਰਨ             v[q4_0_traits::vdr_mmvq];
        ਪੂਰਨ             u[2 * q4_0_traits::vdr_mmvq];


#ਹਦਾਇਤ unroll
        ਲਈ (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            v[i]         = get_int_from_uint8(bq4_0, iqs + i);
            u[2 * i + 0] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
            u[2 * i + 1] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i + q4_0_traits::qi);
        }

        ਵਾਪਸ vec_dot_q4_0_q8_1_impl(v, u, d, *q8_1_ds);
    }
};

// Load four contiguous dwords per operand instead of loading each value separately.
ਬਣਤਰ reorder_vec_dot_q8_0_wide {
    ਸਥਿਰ constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    ਵਰਤੋਂ q8_0_ਖੰਡ  = ggml_sycl_reordered::ਖੰਡ_q_t<GGML_TYPE_Q8_0>;
    ਵਰਤੋਂ q8_0_traits = ਕਿਸਮ_ਨਾਂ q8_0_ਖੰਡ::traits;

    __dpct_inline__ ਦਸ਼ਮਲਵ ਸੰਚਾਲਕ()(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> ibx_ਵਿੱਥ,
                                     ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> d_ਵਿੱਥ, ਅਚਲ int8_t * q8_1_quant_ptr,
                                     ਅਚਲ sycl::half2 * q8_1_ds, ਅਚਲ ਪੂਰਨ & iqs) {
        static_assert(q8_0_traits::vdr_mmvq == 4, "the wide load moves exactly four dwords");

        ਅਚਲ uint8_t * ਅਧਾਰ = static_cast<ਅਚਲ uint8_t *>(vbq);
        ਅਚਲ int8_t *  qs   = reinterpret_cast<ਅਚਲ int8_t *>(ਅਧਾਰ + ibx_ਵਿੱਥ.ਪਹਿਲਾ);
        ਅਚਲ ggml_half d    = *reinterpret_cast<ਅਚਲ ggml_half *>(ਅਧਾਰ + d_ਵਿੱਥ.ਪਹਿਲਾ);

        ਅਚਲ sycl::int4 v = *reinterpret_cast<ਅਚਲ sycl::int4 *>(qs + ਆਕਾਰ(ਪੂਰਨ) * iqs);
        ਅਚਲ sycl::int4 u = *reinterpret_cast<ਅਚਲ sycl::int4 *>(q8_1_quant_ptr + ਆਕਾਰ(ਪੂਰਨ) * iqs);

        ਪੂਰਨ sumi = 0;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i = 0; i < 4; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        ਅਚਲ sycl::half2 ds_ਮੁੱਲਾਂ = *q8_1_ds;
        ਵਾਪਸ static_cast<ਦਸ਼ਮਲਵ>(d) * static_cast<ਦਸ਼ਮਲਵ>(ds_ਮੁੱਲਾਂ[0]) * sumi;
    }
};

ਸਾਂਚਾ <> ਬਣਤਰ reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0> {
    ਸਥਿਰ constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    ਵਰਤੋਂ q8_0_ਖੰਡ  = ggml_sycl_reordered::ਖੰਡ_q_t<GGML_TYPE_Q8_0>;
    ਵਰਤੋਂ q8_0_traits = ਕਿਸਮ_ਨਾਂ q8_0_ਖੰਡ::traits;

    __dpct_inline__ ਦਸ਼ਮਲਵ ਸੰਚਾਲਕ()(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> ibx_ਵਿੱਥ,
                                     ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> d_ਵਿੱਥ, ਅਚਲ int8_t * q8_1_quant_ptr,
                                     ਅਚਲ sycl::half2 * q8_1_ds, ਅਚਲ ਪੂਰਨ & iqs,
                                     ਅਚਲ ਪੂਰਨ vdr = q8_0_traits::vdr_mmvq) {
        ਅਚਲ uint8_t * ਅਧਾਰ = static_cast<ਅਚਲ uint8_t *>(vbq);
        ਅਚਲ int8_t *  qs   = reinterpret_cast<ਅਚਲ int8_t *>(ਅਧਾਰ + ibx_ਵਿੱਥ.ਪਹਿਲਾ);
        ਅਚਲ ggml_half  d   = *reinterpret_cast<ਅਚਲ ggml_half *>(ਅਧਾਰ + d_ਵਿੱਥ.ਪਹਿਲਾ);

        ਪੂਰਨ v[q8_0_traits::vdr_mmvq];
        ਪੂਰਨ u[q8_0_traits::vdr_mmvq];

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i = 0; i < vdr; ++i) {
            v[i] = get_int_from_int8(qs, iqs + i);
            u[i] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
        }

        ਪੂਰਨ sumi = 0;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i = 0; i < vdr; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        ਅਚਲ sycl::half2 ds_ਮੁੱਲਾਂ = *q8_1_ds;
        ਵਾਪਸ static_cast<ਦਸ਼ਮਲਵ>(d) * static_cast<ਦਸ਼ਮਲਵ>(ds_ਮੁੱਲਾਂ[0]) * sumi;
    }
};

ਸਾਂਚਾ <> ਬਣਤਰ reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K> {
    ਸਥਿਰ constexpr ggml_type gtype = GGML_TYPE_Q2_K;

    ਵਰਤੋਂ q2_k_ਖੰਡ  = ggml_sycl_reordered::ਖੰਡ_q_t<GGML_TYPE_Q2_K>;
    ਵਰਤੋਂ q2_k_traits = ਕਿਸਮ_ਨਾਂ q2_k_ਖੰਡ::traits;

    __dpct_inline__ ਦਸ਼ਮਲਵ ਸੰਚਾਲਕ()(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> ibx_ਵਿੱਥ,
                                     ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> d_ਵਿੱਥ, ਅਚਲ int8_t * q8_1_quant_ptr,
                                     ਅਚਲ sycl::half2 * q8_1_ds, ਅਚਲ ਪੂਰਨ & iqs) {
        ਅਚਲ uint8_t *    ਅਧਾਰ   = static_cast<ਅਚਲ uint8_t *>(vbq);
        ਅਚਲ uint8_t *    qs     = ਅਧਾਰ + ibx_ਵਿੱਥ.ਪਹਿਲਾ;
        ਅਚਲ uint8_t *    scales = ਅਧਾਰ + d_ਵਿੱਥ.ਪਹਿਲਾ;
        ਅਚਲ ggml_half2 * dm     = reinterpret_cast<ਅਚਲ ggml_half2 *>(ਅਧਾਰ + d_ਵਿੱਥ.second);

        ਅਚਲ ਪੂਰਨ bq8_ਵਿੱਥ   = QR2_K * (iqs / QI8_1);
        ਅਚਲ ਪੂਰਨ scale_ਵਿੱਥ = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        ਅਚਲ ਪੂਰਨ v = get_int_from_uint8_aligned(qs, iqs);

        ਪੂਰਨ   u[QR2_K];
        ਦਸ਼ਮਲਵ d8[QR2_K];

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i = 0; i < QR2_K; ++i) {
            ਅਚਲ int8_t * quant_ਅਧਾਰ_ptr = q8_1_quant_ptr + (bq8_ਵਿੱਥ + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_ਅਧਾਰ_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_ਵਿੱਥ + i))[0];
        }

        ਵਾਪਸ vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales + scale_ਵਿੱਥ, *dm, d8);
    }
};

ਸਾਂਚਾ <> ਬਣਤਰ reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K> {
    ਸਥਿਰ constexpr ggml_type gtype = GGML_TYPE_Q3_K;

    ਵਰਤੋਂ q3_k_ਖੰਡ  = ggml_sycl_reordered::ਖੰਡ_q_t<GGML_TYPE_Q3_K>;
    ਵਰਤੋਂ q3_k_traits = ਕਿਸਮ_ਨਾਂ q3_k_ਖੰਡ::traits;

    __dpct_inline__ ਦਸ਼ਮਲਵ ਸੰਚਾਲਕ()(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> ibx_ਵਿੱਥ,
                                     ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> d_ਵਿੱਥ, ਅਚਲ int8_t * q8_1_quant_ptr,
                                     ਅਚਲ sycl::half2 * q8_1_ds, ਅਚਲ ਪੂਰਨ & iqs) {
        ਅਚਲ uint8_t *  ਅਧਾਰ   = static_cast<ਅਚਲ uint8_t *>(vbq);
        ਅਚਲ uint8_t *  qs     = ਅਧਾਰ + ibx_ਵਿੱਥ.ਪਹਿਲਾ;
        ਅਚਲ uint8_t *  hmask  = ਅਧਾਰ + ibx_ਵਿੱਥ.second;
        ਅਚਲ uint8_t *  scales = ਅਧਾਰ + d_ਵਿੱਥ.ਪਹਿਲਾ;
        ਅਚਲ ggml_half  d      = *reinterpret_cast<ਅਚਲ ggml_half *>(ਅਧਾਰ + d_ਵਿੱਥ.second);

        ਅਚਲ ਪੂਰਨ bq8_ਵਿੱਥ   = QR3_K * (iqs / (QI3_K / 2));
        ਅਚਲ ਪੂਰਨ scale_ਵਿੱਥ = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        ਅਚਲ ਪੂਰਨ vl = get_int_from_uint8(qs, iqs);
        ਅਚਲ ਪੂਰਨ vh = ~get_int_from_uint8(hmask, iqs % (QI3_K / 2)) >> bq8_ਵਿੱਥ;

        ਪੂਰਨ   u[QR3_K];
        ਦਸ਼ਮਲਵ d8[QR3_K];

#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ i = 0; i < QR3_K; ++i) {
            ਅਚਲ int8_t * quant_ਅਧਾਰ_ptr = q8_1_quant_ptr + (bq8_ਵਿੱਥ + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_ਅਧਾਰ_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_ਵਿੱਥ + i))[0];
        }

        ਵਾਪਸ vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, scales, scale_ਵਿੱਥ, static_cast<ਦਸ਼ਮਲਵ>(d), d8);
    }
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ vec_dot_q4_K_q8_1_common(ਅਚਲ ਪੂਰਨ * __restrict__ q4, ਅਚਲ uint16_t * __restrict__ scales,
                                             ਅਚਲ ggml_half2 & dm, ਅਚਲ ਖੰਡ_q8_1 * __restrict__ bq8_1,
                                             ਅਚਲ ਪੂਰਨ &        iqs) {
    ਪੂਰਨ   v[2];
    ਪੂਰਨ   u[2 * QR4_K];
    ਦਸ਼ਮਲਵ d8[QR4_K];

    v[0] = q4[0];
    v[1] = q4[4];

    uint16_t  aux[2];
    ਅਚਲ ਪੂਰਨ j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
    ਜੇ (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } ਨਹੀਂਤਾਂ {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }

    ਅਚਲ uint8_t * sc = (ਅਚਲ uint8_t *) aux;
    ਅਚਲ uint8_t * m  = sc + 2;

    ਅਚਲ ਪੂਰਨ bq8_ਵਿੱਥ = QR4_K * ((iqs / 2) / (QI8_1 / 2));

    ਲਈ (ਪੂਰਨ i = 0; i < QR4_K; ++i) {
        ਅਚਲ ਖੰਡ_q8_1 * bq8i = bq8_1 + bq8_ਵਿੱਥ + i;
        d8[i]                   = bq8i->ds[0];

        ਅਚਲ ਪੂਰਨ * q8 = (ਅਚਲ ਪੂਰਨ *) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0]   = q8[0];
        u[2 * i + 1]   = q8[4];
    }

    ਵਾਪਸ vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
}

ਸਾਂਚਾ <> ਬਣਤਰ reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K> {
    ਸਥਿਰ constexpr ggml_type gtype = GGML_TYPE_Q4_K;

    ਵਰਤੋਂ q4_k_ਖੰਡ  = ggml_sycl_reordered::ਖੰਡ_q_t<GGML_TYPE_Q4_K>;
    ਵਰਤੋਂ q4_k_traits = ਕਿਸਮ_ਨਾਂ q4_k_ਖੰਡ::traits;

    ਬਣਤਰ weights {
        ਪੂਰਨ        v[2];
        uint16_t   aux[2];
        ggml_half2 dm;
        ਪੂਰਨ        bq8_ਵਿੱਥ;
    };

    ਬਣਤਰ activations {
        ਪੂਰਨ   u[2 * QR4_K];
        ਦਸ਼ਮਲਵ d8[QR4_K];
    };

    __dpct_inline__ ਸਥਿਰ weights load(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> ibx_ਵਿੱਥ,
                                        ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> d_ਵਿੱਥ, ਅਚਲ ਪੂਰਨ & iqs) {
        ਅਚਲ uint8_t *    ਅਧਾਰ = static_cast<ਅਚਲ uint8_t *>(vbq);
        ਅਚਲ uint8_t *    qs   = ਅਧਾਰ + ibx_ਵਿੱਥ.ਪਹਿਲਾ;
        ਅਚਲ uint8_t *    scs  = ਅਧਾਰ + d_ਵਿੱਥ.ਪਹਿਲਾ;
        ਅਚਲ ggml_half2 * dms  = reinterpret_cast<ਅਚਲ ggml_half2 *>(ਅਧਾਰ + d_ਵਿੱਥ.second);

        weights w;
        w.bq8_ਵਿੱਥ = QR4_K * ((iqs / 2) / (QI8_1 / 2));

        ਅਚਲ ਪੂਰਨ *      q4     = (ਅਚਲ ਪੂਰਨ *) (qs + 16 * w.bq8_ਵਿੱਥ + 4 * ((iqs / 2) % 4));
        ਅਚਲ uint16_t * scales = (ਅਚਲ uint16_t *) scs;

        w.v[0] = q4[0];
        w.v[1] = q4[4];

        ਅਚਲ ਪੂਰਨ j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        ਜੇ (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } ਨਹੀਂਤਾਂ {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        ਵਾਪਸ w;
    }

    __dpct_inline__ ਸਥਿਰ activations ਲੱਦੋ_activations(ਅਚਲ int8_t * q8_1_quant_ptr,
                                                        ਅਚਲ sycl::half2 * q8_1_ds, ਅਚਲ ਪੂਰਨ & iqs) {
        activations a;
        ਅਚਲ ਪੂਰਨ bq8_ਵਿੱਥ = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        ਲਈ (ਪੂਰਨ i = 0; i < QR4_K; ++i) {
            ਅਚਲ int8_t * quant_ਅਧਾਰ_ptr = q8_1_quant_ptr + (bq8_ਵਿੱਥ + i) * QK8_1;
            sycl::half2    ds_ਮੁੱਲਾਂ      = *(q8_1_ds + bq8_ਵਿੱਥ + i);

            a.d8[i] = ds_ਮੁੱਲਾਂ[0];

            ਅਚਲ ਪੂਰਨ * q8 = (ਅਚਲ ਪੂਰਨ *) quant_ਅਧਾਰ_ptr + ((iqs / 2) % 4);
            a.u[2 * i + 0] = q8[0];
            a.u[2 * i + 1] = q8[4];
        }

        ਵਾਪਸ a;
    }

    __dpct_inline__ ਸਥਿਰ ਦਸ਼ਮਲਵ apply(ਅਚਲ weights & w, ਅਚਲ activations & a) {
        ਅਚਲ uint8_t * sc = (ਅਚਲ uint8_t *) w.aux;
        ਅਚਲ uint8_t * m  = sc + 2;

        ਵਾਪਸ vec_dot_q4_K_q8_1_impl_vmmq(w.v, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ ਸਥਿਰ ਦਸ਼ਮਲਵ dot(ਅਚਲ weights & w, ਅਚਲ int8_t * q8_1_quant_ptr,
                                     ਅਚਲ sycl::half2 * q8_1_ds, ਅਚਲ ਪੂਰਨ & iqs) {
        ਅਚਲ ਸਵੈਚਾਲਿਤ a = ਲੱਦੋ_activations(q8_1_quant_ptr, q8_1_ds, iqs);

        ਵਾਪਸ apply(w, a);
    }

    __dpct_inline__ ਦਸ਼ਮਲਵ ਸੰਚਾਲਕ()(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> ibx_ਵਿੱਥ,
                                     ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> d_ਵਿੱਥ, ਅਚਲ int8_t * q8_1_quant_ptr,
                                     ਅਚਲ sycl::half2 * q8_1_ds, ਅਚਲ ਪੂਰਨ & iqs) {
        ਵਾਪਸ dot(load(vbq, ibx_ਵਿੱਥ, d_ਵਿੱਥ, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

ਸਾਂਚਾ <> ਬਣਤਰ reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K> {
    ਸਥਿਰ constexpr ggml_type gtype = GGML_TYPE_Q5_K;

    ਵਰਤੋਂ q5_k_ਖੰਡ  = ggml_sycl_reordered::ਖੰਡ_q_t<GGML_TYPE_Q5_K>;
    ਵਰਤੋਂ q5_k_traits = ਕਿਸਮ_ਨਾਂ q5_k_ਖੰਡ::traits;

    ਬਣਤਰ weights {
        ਪੂਰਨ        vl[2];
        ਪੂਰਨ        vh[2];
        uint16_t   aux[2];
        ggml_half2 dm;
    };

    // same activation layout as Q4_K
    static_assert(QR5_K == QR4_K);
    ਵਰਤੋਂ activations = reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::activations;

    __dpct_inline__ ਸਥਿਰ weights load(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> ibx_ਵਿੱਥ,
                                        ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> d_ਵਿੱਥ, ਅਚਲ ਪੂਰਨ & iqs) {
        ਅਚਲ uint8_t *    ਅਧਾਰ    = static_cast<ਅਚਲ uint8_t *>(vbq);
        ਅਚਲ uint8_t *    qs      = ਅਧਾਰ + ibx_ਵਿੱਥ.ਪਹਿਲਾ;   // low 4 bits
        ਅਚਲ uint8_t *    qh_ਅਧਾਰ = ਅਧਾਰ + ibx_ਵਿੱਥ.second;  // high bit
        ਅਚਲ uint8_t *    scs     = ਅਧਾਰ + d_ਵਿੱਥ.ਪਹਿਲਾ;
        ਅਚਲ ggml_half2 * dms     = reinterpret_cast<ਅਚਲ ggml_half2 *>(ਅਧਾਰ + d_ਵਿੱਥ.second);

        ਅਚਲ ਪੂਰਨ        bq8_ਵਿੱਥ = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        ਅਚਲ ਪੂਰਨ *      ql_ptr     = (ਅਚਲ ਪੂਰਨ *) (qs + 16 * bq8_ਵਿੱਥ + 4 * ((iqs / 2) % 4));
        ਅਚਲ ਪੂਰਨ *      qh_ptr     = (ਅਚਲ ਪੂਰਨ *) (qh_ਅਧਾਰ + 4 * ((iqs / 2) % 4));
        ਅਚਲ uint16_t * scales     = (ਅਚਲ uint16_t *) scs;

        weights w;
        w.vl[0] = ql_ptr[0];
        w.vl[1] = ql_ptr[4];

        w.vh[0] = qh_ptr[0] >> bq8_ਵਿੱਥ;
        w.vh[1] = qh_ptr[4] >> bq8_ਵਿੱਥ;

        ਅਚਲ ਪੂਰਨ j = (QR5_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        ਜੇ (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } ਨਹੀਂਤਾਂ {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        ਵਾਪਸ w;
    }

    __dpct_inline__ ਸਥਿਰ activations ਲੱਦੋ_activations(ਅਚਲ int8_t * q8_1_quant_ptr,
                                                        ਅਚਲ sycl::half2 * q8_1_ds, ਅਚਲ ਪੂਰਨ & iqs) {
        ਵਾਪਸ reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::ਲੱਦੋ_activations(q8_1_quant_ptr, q8_1_ds, iqs);
    }

    __dpct_inline__ ਸਥਿਰ ਦਸ਼ਮਲਵ apply(ਅਚਲ weights & w, ਅਚਲ activations & a) {
        ਅਚਲ uint8_t * sc = (ਅਚਲ uint8_t *) w.aux;
        ਅਚਲ uint8_t * m  = sc + 2;

        ਵਾਪਸ vec_dot_q5_K_q8_1_impl_vmmq(w.vl, w.vh, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ ਸਥਿਰ ਦਸ਼ਮਲਵ dot(ਅਚਲ weights & w, ਅਚਲ int8_t * q8_1_quant_ptr,
                                     ਅਚਲ sycl::half2 * q8_1_ds, ਅਚਲ ਪੂਰਨ & iqs) {
        ਵਾਪਸ apply(w, ਲੱਦੋ_activations(q8_1_quant_ptr, q8_1_ds, iqs));
    }

    __dpct_inline__ ਦਸ਼ਮਲਵ ਸੰਚਾਲਕ()(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> ibx_ਵਿੱਥ,
                                     ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> d_ਵਿੱਥ, ਅਚਲ int8_t * q8_1_quant_ptr,
                                     ਅਚਲ sycl::half2 * q8_1_ds, ਅਚਲ ਪੂਰਨ & iqs) {
        ਵਾਪਸ dot(load(vbq, ibx_ਵਿੱਥ, d_ਵਿੱਥ, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

ਸਾਂਚਾ <> ਬਣਤਰ reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K> {
    ਸਥਿਰ constexpr ggml_type gtype = GGML_TYPE_Q6_K;

    ਵਰਤੋਂ q6_k_ਖੰਡ  = ggml_sycl_reordered::ਖੰਡ_q_t<GGML_TYPE_Q6_K>;
    ਵਰਤੋਂ q6_k_traits = ਕਿਸਮ_ਨਾਂ q6_k_ਖੰਡ::traits;

    __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q6_K_q8_1_impl_mmvq(ਅਚਲ ਪੂਰਨ vl, ਅਚਲ ਪੂਰਨ vh, ਅਚਲ ਪੂਰਨ * __restrict__ u,
                                                      ਅਚਲ int8_t * __restrict__ scales, ਅਚਲ ਦਸ਼ਮਲਵ d,
                                                      ਅਚਲ ਦਸ਼ਮਲਵ * __restrict__ d8) {
        ਵਾਪਸ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
    }

    __dpct_inline__ ਦਸ਼ਮਲਵ ਸੰਚਾਲਕ()(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> ibx_ਵਿੱਥ,
                     ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> d_ਵਿੱਥ, ਅਚਲ int8_t * q8_1_quant_ptr, ਅਚਲ sycl::half2 * q8_1_ds,
                     ਅਚਲ ਪੂਰਨ iqs) {
        ਅਚਲ uint8_t *   ਅਧਾਰ   = static_cast<ਅਚਲ uint8_t *>(vbq);
        ਅਚਲ uint8_t *   ql     = ਅਧਾਰ + ibx_ਵਿੱਥ.ਪਹਿਲਾ;
        ਅਚਲ uint8_t *   qh     = ਅਧਾਰ + ibx_ਵਿੱਥ.second;
        ਅਚਲ int8_t *    scales = reinterpret_cast<ਅਚਲ int8_t *>(ਅਧਾਰ + d_ਵਿੱਥ.ਪਹਿਲਾ);
        ਅਚਲ ggml_half * d      = (ਅਚਲ ggml_half *) (ਅਧਾਰ + d_ਵਿੱਥ.second);

        ਅਚਲ ਪੂਰਨ bq8_ਵਿੱਥ   = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        ਅਚਲ ਪੂਰਨ scale_ਵਿੱਥ = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        ਅਚਲ ਪੂਰਨ vh_ਖਿਸਕਾਓ     = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

        ਅਚਲ ਪੂਰਨ vl = get_int_from_uint8(ql, iqs);
        ਅਚਲ ਪੂਰਨ vh = get_int_from_uint8(qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_ਖਿਸਕਾਓ;

        ਅਚਲ int8_t * scs = scales + scale_ਵਿੱਥ;

        ਅਚਲ ਪੂਰਨ u0 = get_int_from_int8_aligned(
            q8_1_quant_ptr + bq8_ਵਿੱਥ * QK8_1, iqs % QI8_1);
        ਅਚਲ ਪੂਰਨ u1 = get_int_from_int8_aligned(
            q8_1_quant_ptr + (bq8_ਵਿੱਥ + 2) * QK8_1, iqs % QI8_1);
        ਅਚਲ ਦਸ਼ਮਲਵ d80 = (*(q8_1_ds + bq8_ਵਿੱਥ + 0))[0];
        ਅਚਲ ਦਸ਼ਮਲਵ d81 = (*(q8_1_ds + bq8_ਵਿੱਥ + 2))[0];

        ਵਾਪਸ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u0, u1, scs[0], scs[4], *d, d80, d81);
    }
};

ਸਾਂਚਾ <> ਬਣਤਰ reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4> {
    ਸਥਿਰ constexpr ggml_type gtype = GGML_TYPE_MXFP4;

    ਵਰਤੋਂ mxfp4_ਖੰਡ  = ggml_sycl_reordered::ਖੰਡ_q_t<GGML_TYPE_MXFP4>;
    ਵਰਤੋਂ mxfp4_traits = ਕਿਸਮ_ਨਾਂ mxfp4_ਖੰਡ::traits;

    __dpct_inline__ ਦਸ਼ਮਲਵ ਸੰਚਾਲਕ()(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> ibx_ਵਿੱਥ,
                                     ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> d_ਵਿੱਥ, ਅਚਲ int8_t * q8_1_quant_ptr,
                                     ਅਚਲ sycl::half2 * q8_1_ds, ਅਚਲ ਪੂਰਨ & iqs) {
        static_assert(mxfp4_traits::vdr_mmvq == 2, "vector load assumes vdr_mmvq == 2");
        ਅਚਲ uint8_t * ਅਧਾਰ = static_cast<ਅਚਲ uint8_t *>(vbq);

        // Reordered nibble blocks are 16 contiguous bytes and iqs is 0 or 2, so each lane's two
        // weight ints are one aligned 8-byte load (the AoS layout needed eight byte loads).
        ਅਚਲ sycl::int2 q4 = *reinterpret_cast<ਅਚਲ sycl::int2 *>(ਅਧਾਰ + ibx_ਵਿੱਥ.ਪਹਿਲਾ + ਆਕਾਰ(ਪੂਰਨ) * iqs);
        ਅਚਲ uint8_t    e  = ਅਧਾਰ[d_ਵਿੱਥ.ਪਹਿਲਾ];

        // Low nibbles pair with q8_1 ints iqs..iqs+1, high nibbles with iqs+4..iqs+5.
        ਅਚਲ sycl::int2 u_lo = *reinterpret_cast<ਅਚਲ sycl::int2 *>(q8_1_quant_ptr + ਆਕਾਰ(ਪੂਰਨ) * iqs);
        ਅਚਲ sycl::int2 u_hi = *reinterpret_cast<ਅਚਲ sycl::int2 *>(q8_1_quant_ptr + ਆਕਾਰ(ਪੂਰਨ) * (iqs + 4));

        ਅਚਲ sycl::int2 v0 = get_int_from_mxfp4(q4.x());
        ਅਚਲ sycl::int2 v1 = get_int_from_mxfp4(q4.y());

        ਪੂਰਨ sumi = 0;
        sumi = ggml_sycl_dp4a(v0.x(), u_lo.x(), sumi);
        sumi = ggml_sycl_dp4a(v0.y(), u_hi.x(), sumi);
        sumi = ggml_sycl_dp4a(v1.x(), u_lo.y(), sumi);
        sumi = ggml_sycl_dp4a(v1.y(), u_hi.y(), sumi);

        ਅਚਲ ਦਸ਼ਮਲਵ d = ggml_sycl_e8m0_ਮੰਜ਼ਿਲ_fp32(e) * 0.5f * static_cast<ਦਸ਼ਮਲਵ>((*q8_1_ds)[0]);
        ਵਾਪਸ d * sumi;
    }
};
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q4_0_Q8_1_MMVQ 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q4_0_Q8_1_MMQ  4

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q2_0_Q8_1_MMVQ 1

ਸਾਂਚਾ <ਪੂਰਨ vdr>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q2_0_q8_1_impl(
    ਅਚਲ ਪੂਰਨ * v,
    ਅਚਲ ਪੂਰਨ * u,
    ਅਚਲ ਦਸ਼ਮਲਵ & d2,
    ਅਚਲ sycl::half2 & ds8) {
    ਪੂਰਨ sumi = 0;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < vdr; ++i) {
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
            ਅਚਲ uint8_t q = (uint8_t) ((uint32_t) v[i] >> (8 * j));

            // unpack 2-bit values to byte lanes (0..3), then apply zero-point
            // correction with ds8f.y() below, mirroring the q4_0 style.
            ਪੂਰਨ vi = 0;
            vi |= (((q >> 0) & 0x3) & 0xFF) << 0;
            vi |= (((q >> 2) & 0x3) & 0xFF) << 8;
            vi |= (((q >> 4) & 0x3) & 0xFF) << 16;
            vi |= (((q >> 6) & 0x3) & 0xFF) << 24;

            sumi = dpct::dp4a(vi, u[4 * i + j], sumi);
        }
    }

    ਅਚਲ sycl::float2 ds8f = ds8.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
    // q2_0 has zero-point 1. Scale ds8f.y() by processed-lane ratio,
    // consistent with q4_0's explicit zero-point subtraction style.
    ਵਾਪਸ d2 * (sumi * ds8f.x() - ((ਦਸ਼ਮਲਵ) vdr / (ਦਸ਼ਮਲਵ) QI2_0) * ds8f.y());
}

ਸਾਂਚਾ <ਪੂਰਨ vdr>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q4_0_q8_1_impl(ਅਚਲ ਪੂਰਨ * v, ਅਚਲ ਪੂਰਨ * u, ਅਚਲ ਦਸ਼ਮਲਵ & d4,
                                                    ਅਚਲ sycl::half2 & ds8) {
    ਪੂਰਨ sumi = 0;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < vdr; ++i) {
        ਅਚਲ ਪੂਰਨ vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        ਅਚਲ ਪੂਰਨ vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

    ਅਚਲ sycl::float2 ds8f = ds8.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

    // second part effectively subtracts 8 from each quant value
    ਵਾਪਸ d4 * (sumi * ds8f.x() - (8 * vdr / QI4_0) * ds8f.y());
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q4_1_Q8_1_MMVQ 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q4_1_Q8_1_MMQ  4

ਸਾਂਚਾ <ਪੂਰਨ vdr>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q4_1_q8_1_impl(ਅਚਲ ਪੂਰਨ *v, ਅਚਲ ਪੂਰਨ *u,
                                                    ਅਚਲ sycl::half2 &dm4,
                                                    ਅਚਲ sycl::half2 &ds8) {

    ਪੂਰਨ sumi = 0;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < vdr; ++i) {
        ਅਚਲ ਪੂਰਨ vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        ਅਚਲ ਪੂਰਨ vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    ਅਚਲ sycl::float2 tmp =
        (dm4 * ds8).convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
    ਅਚਲ ਦਸ਼ਮਲਵ d4d8 = tmp.x();
    ਅਚਲ ਦਸ਼ਮਲਵ m4s8 = tmp.y();
#ਨਹੀਂਤਾਂ
    ਅਚਲ sycl::float2 dm4f =
        dm4.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
    ਅਚਲ sycl::float2 ds8f =
        ds8.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
    ਅਚਲ ਦਸ਼ਮਲਵ d4d8 = dm4f.x() * ds8f.x();
    ਅਚਲ ਦਸ਼ਮਲਵ m4s8 = dm4f.y() * ds8f.y();
#ਜੇ_ਅੰਤ // GGML_SYCL_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    ਵਾਪਸ sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q5_0_Q8_1_MMVQ 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q5_0_Q8_1_MMQ  4

ਸਾਂਚਾ <ਪੂਰਨ vdr>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q5_0_q8_1_impl(ਅਚਲ ਪੂਰਨ *vl, ਅਚਲ ਪੂਰਨ *vh, ਅਚਲ ਪੂਰਨ *u,
                       ਅਚਲ ਦਸ਼ਮਲਵ &d5, ਅਚਲ sycl::half2 &ds8) {
    ਪੂਰਨ sumi = 0;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < vdr; ++i) {
        ਪੂਰਨ vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        ਪੂਰਨ vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

    ਅਚਲ sycl::float2 ds8f =
        ds8.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();

    // second part effectively subtracts 16 from each quant value
    ਵਾਪਸ d5 * (sumi * ds8f.x() - (16 * vdr / QI5_0) * ds8f.y());
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q5_1_Q8_1_MMVQ 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q5_1_Q8_1_MMQ  4

ਸਾਂਚਾ <ਪੂਰਨ vdr>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q5_1_q8_1_impl(ਅਚਲ ਪੂਰਨ *vl, ਅਚਲ ਪੂਰਨ *vh, ਅਚਲ ਪੂਰਨ *u,
                       ਅਚਲ sycl::half2 &dm5, ਅਚਲ sycl::half2 &ds8) {

    ਪੂਰਨ sumi = 0;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < vdr; ++i) {
        ਪੂਰਨ vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        ਪੂਰਨ vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
     ਅਚਲ sycl::float2 tmp =
        (dm5 * ds8).convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
    ਅਚਲ ਦਸ਼ਮਲਵ d5d8 = tmp.x();
    ਅਚਲ ਦਸ਼ਮਲਵ m5s8 = tmp.y();


#ਨਹੀਂਤਾਂ
    ਅਚਲ sycl::float2 dm5f =
        dm5.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
    ਅਚਲ sycl::float2 ds8f =
        ds8.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
    ਅਚਲ ਦਸ਼ਮਲਵ d5d8 = dm5f.x() * ds8f.x();
    ਅਚਲ ਦਸ਼ਮਲਵ m5s8 = dm5f.y() * ds8f.y();
#ਜੇ_ਅੰਤ // GGML_SYCL_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    ਵਾਪਸ sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q8_0_Q8_1_MMVQ 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_Q8_0_Q8_1_MMQ 8

ਸਾਂਚਾ <ਪੂਰਨ vdr>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q8_0_q8_1_impl(ਅਚਲ ਪੂਰਨ *v, ਅਚਲ ਪੂਰਨ *u,
                                                    ਅਚਲ ਦਸ਼ਮਲਵ &d8_0,
                                                    ਅਚਲ ਦਸ਼ਮਲਵ &d8_1) {

    ਪੂਰਨ sumi = 0;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

    ਵਾਪਸ d8_0*d8_1 * sumi;
}

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ T, ਪੂਰਨ vdr>
ਸਥਿਰ __dpct_inline__ T vec_dot_q8_0_q8_1_impl(ਅਚਲ ਪੂਰਨ * v, ਅਚਲ ਪੂਰਨ * u, ਅਚਲ T & d8_0, ਅਚਲ T & d8_1) {
    ਪੂਰਨ sumi = 0;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_sycl_dp4a(v[i], u[i], sumi);
    }

    ਵਾਪਸ d8_0*d8_1 * ((T) sumi);
}

ਸਾਂਚਾ <ਪੂਰਨ vdr>
ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q8_1_q8_1_impl(ਅਚਲ ਪੂਰਨ *v, ਅਚਲ ਪੂਰਨ *u,
                                                    ਅਚਲ sycl::half2 &dm8,
                                                    ਅਚਲ sycl::half2 &ds8) {

    ਪੂਰਨ sumi = 0;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_SYCL_F16
    ਅਚਲ sycl::float2 tmp =
        (dm8 * ds8).convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
    ਅਚਲ ਦਸ਼ਮਲਵ d8d8 = tmp.x();
    ਅਚਲ ਦਸ਼ਮਲਵ m8s8 = tmp.y();
#ਨਹੀਂਤਾਂ
    ਅਚਲ sycl::float2 dm8f =
        dm8.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
    ਅਚਲ sycl::float2 ds8f =
        ds8.convert<ਦਸ਼ਮਲਵ, sycl::rounding_ਬਹੁਲਕ::automatic>();
    ਅਚਲ ਦਸ਼ਮਲਵ d8d8 = dm8f.x() * ds8f.x();
    ਅਚਲ ਦਸ਼ਮਲਵ m8s8 = dm8f.y() * ds8f.y();
#ਜੇ_ਅੰਤ // GGML_SYCL_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    ਵਾਪਸ sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q4_0_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                  ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {

    ਅਚਲ ਖੰਡ_q4_0 * bq4_0 = (ਅਚਲ ਖੰਡ_q4_0 *) vbq;

    ਪੂਰਨ v[VDR_Q4_0_Q8_1_MMVQ];
    ਪੂਰਨ u[2 * VDR_Q4_0_Q8_1_MMVQ];

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]         = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    ਵਾਪਸ vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q2_0_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                  ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {

    ਅਚਲ ਖੰਡ_q2_0 * bq2_0 = (ਅਚਲ ਖੰਡ_q2_0 *) vbq;

    ਪੂਰਨ v[2 * VDR_Q2_0_Q8_1_MMVQ];
    ਪੂਰਨ u[8 * VDR_Q2_0_Q8_1_MMVQ];

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < VDR_Q2_0_Q8_1_MMVQ; ++i) {
        ਅਚਲ ਪੂਰਨ ਅਧਾਰ = 4 * (iqs + i);

        // Q2_0 has QK2_0 = 64 and uses 2 x QK8_1 blocks on the RHS.
        v[2 * i + 0] = get_int_from_uint8(bq2_0->qs, iqs + i);
        v[2 * i + 1] = get_int_from_uint8(bq2_0->qs, iqs + i + QI2_0);

        u[8 * i + 0] = get_int_from_int8_aligned(bq8_1[0].qs, ਅਧਾਰ + 0);
        u[8 * i + 1] = get_int_from_int8_aligned(bq8_1[0].qs, ਅਧਾਰ + 1);
        u[8 * i + 2] = get_int_from_int8_aligned(bq8_1[0].qs, ਅਧਾਰ + 2);
        u[8 * i + 3] = get_int_from_int8_aligned(bq8_1[0].qs, ਅਧਾਰ + 3);

        u[8 * i + 4] = get_int_from_int8_aligned(bq8_1[1].qs, ਅਧਾਰ + 0);
        u[8 * i + 5] = get_int_from_int8_aligned(bq8_1[1].qs, ਅਧਾਰ + 1);
        u[8 * i + 6] = get_int_from_int8_aligned(bq8_1[1].qs, ਅਧਾਰ + 2);
        u[8 * i + 7] = get_int_from_int8_aligned(bq8_1[1].qs, ਅਧਾਰ + 3);
    }

    ਅਚਲ ਦਸ਼ਮਲਵ sum0 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + 0, u + 0, bq2_0->d, bq8_1[0].ds);
    ਅਚਲ ਦਸ਼ਮਲਵ sum1 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + VDR_Q2_0_Q8_1_MMVQ, u + 4 * VDR_Q2_0_Q8_1_MMVQ, bq2_0->d, bq8_1[1].ds);
    ਵਾਪਸ sum0 + sum1;
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q4_1_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                  ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {

    ਅਚਲ ਖੰਡ_q4_1 * bq4_1 = (ਅਚਲ ਖੰਡ_q4_1 *) vbq;

    ਪੂਰਨ v[VDR_Q4_1_Q8_1_MMVQ];
    ਪੂਰਨ u[2*VDR_Q4_1_Q8_1_MMVQ];

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    ਵਾਪਸ vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_MXFP4_Q8_1_MMVQ 2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_MXFP4_Q8_1_MMQ  4

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_mxfp4_q8_1(ਅਚਲ ਖਾਲੀ * __restrict__ vbq,
                                                ਅਚਲ ਖੰਡ_q8_1 * __restrict__ bq8_1,
                                                ਅਚਲ ਪੂਰਨ & iqs) {
    ਅਚਲ ਖੰਡ_mxfp4 * bq4 = (ਅਚਲ ਖੰਡ_mxfp4 *) vbq;

    ਅਚਲ ਪੂਰਨ * q8 = (ਅਚਲ ਪੂਰਨ *) bq8_1->qs + iqs;

    ਪੂਰਨ sumi = 0;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        ਅਚਲ ਪੂਰਨ aux_q4 = get_int_b1(bq4->qs, iqs + l);
        ਅਚਲ sycl::int2 v      = get_int_from_mxfp4(aux_q4);
        sumi = ggml_sycl_dp4a(v.x(), q8[l + 0], sumi);
        sumi = ggml_sycl_dp4a(v.y(), q8[l + 4], sumi);
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d = ggml_sycl_e8m0_ਮੰਜ਼ਿਲ_fp32(bq4->e) * 0.5f * (bq8_1->ds)[0];
    ਵਾਪਸ d * sumi;
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_NVFP4_Q8_1_MMVQ 4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_NVFP4_Q8_1_MMQ  8

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_nvfp4_q8_1(ਅਚਲ ਖਾਲੀ * __restrict__ vbq,
                                                ਅਚਲ ਖੰਡ_q8_1 * __restrict__ bq8_1,
                                                ਅਚਲ int32_t & iqs) {
    ਅਚਲ ਖੰਡ_nvfp4 * bq4 = (ਅਚਲ ਖੰਡ_nvfp4 *) vbq;
    ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; i++) {
        ਅਚਲ int32_t iqs0 = iqs + 2*i;
        ਅਚਲ int32_t iqs1 = iqs0 + 1;
        ਅਚਲ int32_t is = iqs0 >> 1;
        ਅਚਲ sycl::int2   v0   = get_int_from_ਸਾਰਣੀ_16(get_int_b4(bq4->qs, iqs0), kvalues_mxfp4);
        ਅਚਲ sycl::int2   v1   = get_int_from_ਸਾਰਣੀ_16(get_int_b4(bq4->qs, iqs1), kvalues_mxfp4);
        ਅਚਲ ਖੰਡ_q8_1 * bq8 = bq8_1 + (is >> 1);
        ਅਚਲ int32_t i8 = ((is & 1) << 2);

        ਪੂਰਨ sumi = ggml_sycl_dp4a(v0.x(), get_int_b4(bq8->qs, i8 + 0), 0);
        sumi     = ggml_sycl_dp4a(v0.y(), get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi     = ggml_sycl_dp4a(v1.x(), get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi     = ggml_sycl_dp4a(v1.y(), get_int_b4(bq8->qs, i8 + 3), sumi);

        ਅਚਲ ਦਸ਼ਮਲਵ d = ggml_sycl_ue4m3_ਮੰਜ਼ਿਲ_fp32(bq4->d[is]) * (bq8->ds)[0];
        ਜੋੜ += d * ਦਸ਼ਮਲਵ(sumi);
    }

    ਵਾਪਸ ਜੋੜ;
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q5_0_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                  ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {

    ਅਚਲ ਖੰਡ_q5_0 * bq5_0 = (ਅਚਲ ਖੰਡ_q5_0 *) vbq;

    ਪੂਰਨ vl[VDR_Q5_0_Q8_1_MMVQ];
    ਪੂਰਨ vh[VDR_Q5_0_Q8_1_MMVQ];
    ਪੂਰਨ  u[2*VDR_Q5_0_Q8_1_MMVQ];

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    ਵਾਪਸ vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q5_1_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                  ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {

    ਅਚਲ ਖੰਡ_q5_1 * bq5_1 = (ਅਚਲ ਖੰਡ_q5_1 *) vbq;

    ਪੂਰਨ vl[VDR_Q5_1_Q8_1_MMVQ];
    ਪੂਰਨ vh[VDR_Q5_1_Q8_1_MMVQ];
    ਪੂਰਨ  u[2*VDR_Q5_1_Q8_1_MMVQ];

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    ਵਾਪਸ vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q8_0_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                  ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {

    ਅਚਲ ਖੰਡ_q8_0 * bq8_0 = (ਅਚਲ ਖੰਡ_q8_0 *) vbq;

    ਪੂਰਨ v[VDR_Q8_0_Q8_1_MMVQ];
    ਪੂਰਨ u[VDR_Q8_0_Q8_1_MMVQ];

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    ਵਾਪਸ vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                      bq8_1->ds[0]);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q2_K_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                  ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {

    ਅਚਲ ਖੰਡ_q2_K * bq2_K = (ਅਚਲ ਖੰਡ_q2_K *) vbq;

    ਅਚਲ ਪੂਰਨ bq8_ਵਿੱਥ = QR2_K * (iqs / QI8_1);
    ਅਚਲ ਪੂਰਨ scale_ਵਿੱਥ = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    ਅਚਲ uint8_t * scales = bq2_K->scales + scale_ਵਿੱਥ;

    ਅਚਲ ਪੂਰਨ v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    ਪੂਰਨ    u[QR2_K];
    ਦਸ਼ਮਲਵ d8[QR2_K];

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_ਵਿੱਥ + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_ਵਿੱਥ + i].ds[0];
    }

    ਵਾਪਸ vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q3_K_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                  ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {

    ਅਚਲ ਖੰਡ_q3_K * bq3_K = (ਅਚਲ ਖੰਡ_q3_K *) vbq;

    ਅਚਲ ਪੂਰਨ bq8_ਵਿੱਥ = QR3_K * (iqs / (QI3_K/2));
    ਅਚਲ ਪੂਰਨ scale_ਵਿੱਥ = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    ਅਚਲ ਦਸ਼ਮਲਵ d = bq3_K->d;

    ਅਚਲ ਪੂਰਨ vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    ਅਚਲ ਪੂਰਨ vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_ਵਿੱਥ;

    ਪੂਰਨ    u[QR3_K];
    ਦਸ਼ਮਲਵ d8[QR3_K];

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_ਵਿੱਥ + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_ਵਿੱਥ + i].ds[0];
    }

    ਵਾਪਸ vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_ਵਿੱਥ, d, d8);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ vec_dot_q4_K_q8_1(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ ਖੰਡ_q8_1 * __restrict__ bq8_1,
                                               ਅਚਲ ਪੂਰਨ & iqs) {
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_QKK_64

    ਅਚਲ ਖੰਡ_q4_K * bq4_K = (ਅਚਲ ਖੰਡ_q4_K *) vbq;

    ਅਚਲ ਪੂਰਨ        bq8_ਵਿੱਥ = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    ਅਚਲ ਪੂਰਨ *      q4         = (ਅਚਲ ਪੂਰਨ *) (bq4_K->qs + 16 * bq8_ਵਿੱਥ + 4 * ((iqs / 2) % 4));
    ਅਚਲ uint16_t * scales     = (ਅਚਲ uint16_t *) bq4_K->scales;

    ਵਾਪਸ vec_dot_q4_K_q8_1_common(q4, scales, bq4_K->dm, bq8_1, iqs);

#ਨਹੀਂਤਾਂ

#ਜੇ __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    ਅਚਲ ਖੰਡ_q4_K * bq4_K = (ਅਚਲ ਖੰਡ_q4_K *) vbq;

    ਦਸ਼ਮਲਵ sumf_d = 0.0f;
    ਦਸ਼ਮਲਵ sumf_m = 0.0f;

    uint16_t aux16[2];
    ਅਚਲ uint8_t * s = (ਅਚਲ uint8_t *)aux16;

    ਅਚਲ uint16_t * a = (ਅਚਲ uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    ਅਚਲ ਦਸ਼ਮਲਵ dall = bq4_K->dm[0];
    ਅਚਲ ਦਸ਼ਮਲਵ dmin = bq4_K->dm[1];

    ਅਚਲ ਦਸ਼ਮਲਵ d8_1 = bq8_1[0].ds[0];
    ਅਚਲ ਦਸ਼ਮਲਵ d8_2 = bq8_1[1].ds[1];

    ਅਚਲ ਪੂਰਨ ui1 = *((ਅਚਲ ਪੂਰਨ *)bq8_1[0].qs + (iqs/2));
    ਅਚਲ ਪੂਰਨ ui2 = *((ਅਚਲ ਪੂਰਨ *)bq8_1[0].qs + (iqs/2) + 4);
    ਅਚਲ ਪੂਰਨ ui3 = *((ਅਚਲ ਪੂਰਨ *)bq8_1[1].qs + (iqs/2));
    ਅਚਲ ਪੂਰਨ ui4 = *((ਅਚਲ ਪੂਰਨ *)bq8_1[1].qs + (iqs/2) + 4);

    ਅਚਲ ਪੂਰਨ * q4 = (ਅਚਲ ਪੂਰਨ *)bq4_K->qs + (iqs/2);
    ਅਚਲ ਪੂਰਨ v1 = q4[0];
    ਅਚਲ ਪੂਰਨ v2 = q4[4];

    ਅਚਲ ਪੂਰਨ dot1 = dpct::dp4a(ui2, v2 & 0x0f0f0f0f, dpct::dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    ਅਚਲ ਪੂਰਨ dot2 = dpct::dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, dpct::dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    ਅਚਲ ਪੂਰਨ dot3 = dpct::dp4a(0x01010101, ui2, dpct::dp4a(0x01010101, ui1, 0));
    ਅਚਲ ਪੂਰਨ dot4 = dpct::dp4a(0x01010101, ui4, dpct::dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    ਵਾਪਸ dall * sumf_d - dmin * sumf_m;

#ਨਹੀਂਤਾਂ
    bad_arch();
#ਜੇ_ਅੰਤ // __SYCL_ARCH__ >= VER_4VEC

#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q5_K_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                  ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ GGML_QKK_64
    ਅਚਲ ਖੰਡ_q5_K * bq5_K = (ਅਚਲ ਖੰਡ_q5_K *) vbq;

    ਪੂਰਨ   vl[2];
    ਪੂਰਨ   vh[2];
    ਪੂਰਨ    u[2*QR5_K];
    ਦਸ਼ਮਲਵ d8[QR5_K];

    ਅਚਲ ਪੂਰਨ bq8_ਵਿੱਥ = QR5_K * ((iqs/2) / (QI8_1/2));
    ਅਚਲ ਪੂਰਨ * ql = (ਅਚਲ ਪੂਰਨ *)(bq5_K->qs + 16 * bq8_ਵਿੱਥ + 4 * ((iqs/2)%4));
    ਅਚਲ ਪੂਰਨ * qh = (ਅਚਲ ਪੂਰਨ *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_ਵਿੱਥ;
    vh[1] = qh[4] >> bq8_ਵਿੱਥ;

    ਅਚਲ uint16_t * scales = (ਅਚਲ uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    ਅਚਲ ਪੂਰਨ j = bq8_ਵਿੱਥ/2;
    ਜੇ (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } ਨਹੀਂਤਾਂ {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    ਅਚਲ uint8_t * sc = (ਅਚਲ uint8_t *)aux;
    ਅਚਲ uint8_t * m  = sc + 2;

#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ i = 0; i < QR5_K; ++i) {
        ਅਚਲ ਖੰਡ_q8_1 * bq8i = bq8_1 + bq8_ਵਿੱਥ + i;
        d8[i] = bq8i->ds[0];

        ਅਚਲ ਪੂਰਨ * q8 = (ਅਚਲ ਪੂਰਨ *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    ਵਾਪਸ vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#ਨਹੀਂਤਾਂ

#ਜੇ __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    ਅਚਲ ਖੰਡ_q5_K * bq5_K = (ਅਚਲ ਖੰਡ_q5_K *) vbq;

    ਅਚਲ int8_t * s = bq5_K->scales;

    ਅਚਲ ਦਸ਼ਮਲਵ d = bq5_K->d;

    ਅਚਲ ਦਸ਼ਮਲਵ d8_1 = bq8_1[0].ds[0];
    ਅਚਲ ਦਸ਼ਮਲਵ d8_2 = bq8_1[1].ds[1];

    ਅਚਲ ਪੂਰਨ ui1 = *((ਅਚਲ ਪੂਰਨ *)bq8_1[0].qs + (iqs/2));
    ਅਚਲ ਪੂਰਨ ui2 = *((ਅਚਲ ਪੂਰਨ *)bq8_1[0].qs + (iqs/2) + 4);
    ਅਚਲ ਪੂਰਨ ui3 = *((ਅਚਲ ਪੂਰਨ *)bq8_1[1].qs + (iqs/2));
    ਅਚਲ ਪੂਰਨ ui4 = *((ਅਚਲ ਪੂਰਨ *)bq8_1[1].qs + (iqs/2) + 4);

    ਅਚਲ ਪੂਰਨ * ql = (ਅਚਲ ਪੂਰਨ *)bq5_K->qs + (iqs/2);
    ਅਚਲ ਪੂਰਨ vl1 = ql[0];
    ਅਚਲ ਪੂਰਨ vl2 = ql[4];

    ਅਚਲ ਪੂਰਨ ਕਦਮ = 4 * (iqs/2); // 0, 4, 8, 12
    ਅਚਲ ਪੂਰਨ im = ਕਦਮ/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    ਅਚਲ ਪੂਰਨ ਵਿੱਚ = ਕਦਮ%8; // 0, 4, 0, 4
    ਅਚਲ ਪੂਰਨ vh = (*((ਅਚਲ ਪੂਰਨ *)(bq5_K->qh + ਵਿੱਚ))) >> im;

    ਅਚਲ ਪੂਰਨ v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    ਅਚਲ ਪੂਰਨ v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    ਅਚਲ ਪੂਰਨ v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    ਅਚਲ ਪੂਰਨ v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    ਅਚਲ ਦਸ਼ਮਲਵ sumf_d = d8_1 * (dpct::dp4a(ui1, v1, 0) * s[0] + dpct::dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (dpct::dp4a(ui3, v3, 0) * s[2] + dpct::dp4a(ui4, v4, 0) * s[3]);

    ਵਾਪਸ d * sumf_d;

#ਨਹੀਂਤਾਂ
    bad_arch();
#ਜੇ_ਅੰਤ // __SYCL_ARCH__ >= VER_4VEC

#ਜੇ_ਅੰਤ
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_q6_K_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                  ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {

    ਅਚਲ ਖੰਡ_q6_K * bq6_K = (ਅਚਲ ਖੰਡ_q6_K *) vbq;

    ਅਚਲ ਪੂਰਨ bq8_ਵਿੱਥ = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    ਅਚਲ ਪੂਰਨ scale_ਵਿੱਥ = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    ਅਚਲ ਪੂਰਨ vh_ਖਿਸਕਾਓ = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    ਅਚਲ ਪੂਰਨ vl = get_int_from_uint8(bq6_K->ql, iqs);
    ਅਚਲ ਪੂਰਨ vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_ਖਿਸਕਾਓ;

    ਅਚਲ int8_t * scales = bq6_K->scales + scale_ਵਿੱਥ;

    ਅਚਲ ਪੂਰਨ u0 = get_int_from_int8_aligned(
        bq8_1[bq8_ਵਿੱਥ + 0].qs, iqs % QI8_1);
    ਅਚਲ ਪੂਰਨ u1 = get_int_from_int8_aligned(
        bq8_1[bq8_ਵਿੱਥ + 2].qs, iqs % QI8_1);
    ਅਚਲ ਦਸ਼ਮਲਵ d80 = bq8_1[bq8_ਵਿੱਥ + 0].ds[0];
    ਅਚਲ ਦਸ਼ਮਲਵ d81 = bq8_1[bq8_ਵਿੱਥ + 2].ds[0];

    ਵਾਪਸ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u0, u1, scales[0], scales[4], bq6_K->d, d80, d81);
}


// NOTE: the VDR_IQ*_Q8_1_MMVQ values deliberately differ from the identically named CUDA constants
// (vecdotq.cuh): the SYCL kernels pair them with a halved qi (e.g. QI3_S/2), so the values are not
// interchangeable and must not be copied across backends.
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_IQ2_XXS_Q8_1_MMVQ 1

ਸਥਿਰ __dpct_inline__ uint32_t iq_sign_ਨਕਾਬ(ਅਚਲ uint32_t signs) {
    ਅਚਲ uint32_t ਬਿੱਟ =
        (signs & 0x01) | ((signs & 0x02) << 7) |
        ((signs & 0x04) << 14) | ((signs & 0x08) << 21);

    ਵਾਪਸ ਬਿੱਟ * 0xff;
}

ਸਥਿਰ __dpct_inline__ uint32_t iq_sign_ਬਿੱਟ(ਅਚਲ uint32_t encoded) {
    ਅਚਲ uint32_t signs = encoded & 127;

    ਵਾਪਸ signs | ((sycl::popcount(signs) & 1) << 7);
}

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_iq2_xxs_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                     ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs,
                     ਅਚਲ uint64_t *iq2xxs_grid, ਅਚਲ uint8_t *ksigns_iq2xs,
                     ਅਚਲ uint8_t *kmask_iq2xs) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq2_xxs * bq2 = (ਅਚਲ ਖੰਡ_iq2_xxs *) vbq;

    ਅਚਲ ਪੂਰਨ ib32 = iqs;
    ਅਚਲ uint16_t * q2 = bq2->qs + 4*ib32;
    ਅਚਲ uint8_t  * aux8 = (ਅਚਲ uint8_t *)q2;
    ਅਚਲ int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = q2[2] | (q2[3] << 16);
    ਪੂਰਨ sumi = 0;
    ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
        ਅਚਲ uint8_t * grid = (ਅਚਲ uint8_t *)(iq2xxs_grid + aux8[l]);
        ਅਚਲ uint8_t  signs = ksigns_iq2xs[aux32 & 127];
        ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
            sumi += q8[j] * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
        aux32 >>= 7;
    }
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.25f;
    ਵਾਪਸ d * sumi;
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
    ਵਾਪਸ 0.f;
#ਜੇ_ਅੰਤ
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_IQ2_XS_Q8_1_MMVQ 1

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_iq2_xs_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                    ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs,
                    ਅਚਲ uint64_t *iq2xs_grid) {
#ਜੇ DPCT_COMPATIBILITY_ਆਰਜ਼ੀ >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq2_xs * bq2 = (ਅਚਲ ਖੰਡ_iq2_xs *) vbq;

    ਅਚਲ ਪੂਰਨ ib32 = iqs;
    ਅਚਲ uint16_t * q2 = bq2->qs + 4*ib32;
    ਅਚਲ int8_t   * q8 = bq8_1[ib32].qs;
    ਅਚਲ uint8_t ls1 = bq2->scales[ib32] & 0xf;
    ਅਚਲ uint8_t ls2 = bq2->scales[ib32] >>  4;
    ਪੂਰਨ sumi1 = 0;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ l = 0; l < 2; ++l) {
        ਅਚਲ uint64_t grid = iq2xs_grid[q2[l] & 511];
        ਅਚਲ uint32_t sign_ਬਿੱਟ = iq_sign_ਬਿੱਟ(q2[l] >> 9);
        ਅਚਲ uint32_t signs0 = iq_sign_ਨਕਾਬ(sign_ਬਿੱਟ);
        ਅਚਲ uint32_t signs1 = iq_sign_ਨਕਾਬ(sign_ਬਿੱਟ >> 4);
        ਅਚਲ ਪੂਰਨ grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        ਅਚਲ ਪੂਰਨ grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi1 = dpct::dp4a(grid_l, *((ਅਚਲ ਪੂਰਨ *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((ਅਚਲ ਪੂਰਨ *)q8 + 1), sumi1);
        q8 += 8;
    }
    ਪੂਰਨ sumi2 = 0;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ l = 2; l < 4; ++l) {
        ਅਚਲ uint64_t grid = iq2xs_grid[q2[l] & 511];
        ਅਚਲ uint32_t sign_ਬਿੱਟ = iq_sign_ਬਿੱਟ(q2[l] >> 9);
        ਅਚਲ uint32_t signs0 = iq_sign_ਨਕਾਬ(sign_ਬਿੱਟ);
        ਅਚਲ uint32_t signs1 = iq_sign_ਨਕਾਬ(sign_ਬਿੱਟ >> 4);
        ਅਚਲ ਪੂਰਨ grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        ਅਚਲ ਪੂਰਨ grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi2 = dpct::dp4a(grid_l, *((ਅਚਲ ਪੂਰਨ *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((ਅਚਲ ਪੂਰਨ *)q8 + 1), sumi2);
        q8 += 8;
    }
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    ਵਾਪਸ d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
    ਵਾਪਸ 0.f;
#ਜੇ_ਅੰਤ
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
    ਵਾਪਸ 0.f;
#ਜੇ_ਅੰਤ
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_IQ2_S_Q8_1_MMVQ 1

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_iq2_s_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                   ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq2_s * bq2 = (ਅਚਲ ਖੰਡ_iq2_s *) vbq;

    ਅਚਲ ਪੂਰਨ ib32 = iqs;
    ਅਚਲ int8_t  * q8 = bq8_1[ib32].qs;
    ਅਚਲ uint8_t * signs = bq2->qs + QK_K/8 + 4*ib32;
    ਅਚਲ uint8_t ls1 = bq2->scales[ib32] & 0xf;
    ਅਚਲ uint8_t ls2 = bq2->scales[ib32] >>  4;
    ਪੂਰਨ sumi1 = 0;
    ਲਈ (ਪੂਰਨ l = 0; l < 2; ++l) {
        ਅਚਲ uint32_t * grid = (ਅਚਲ uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        ਅਚਲ uint32_t signs0 = dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ਅਚਲ uint32_t signs1 = dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ਅਚਲ ਪੂਰਨ grid_l = dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        ਅਚਲ ਪੂਰਨ grid_h = dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi1 = dpct::dp4a(grid_l, *((ਅਚਲ ਪੂਰਨ *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((ਅਚਲ ਪੂਰਨ *)q8 + 1), sumi1);
        q8 += 8;
    }
    ਪੂਰਨ sumi2 = 0;
    ਲਈ (ਪੂਰਨ l = 2; l < 4; ++l) {
        ਅਚਲ uint32_t * grid = (ਅਚਲ uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        ਅਚਲ uint32_t signs0 = dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ਅਚਲ uint32_t signs1 = dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ਅਚਲ ਪੂਰਨ grid_l = dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        ਅਚਲ ਪੂਰਨ grid_h = dpct::vectorized_ਦੁਆਧਾਰੀ<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi2 = dpct::dp4a(grid_l, *((ਅਚਲ ਪੂਰਨ *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((ਅਚਲ ਪੂਰਨ *)q8 + 1), sumi2);
        q8 += 8;
    }
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    ਵਾਪਸ d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
#ਜੇ_ਅੰਤ
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_IQ3_XXS_Q8_1_MMVQ 1

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_iq3_xxs_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                     ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs,
                     ਅਚਲ uint32_t *iq3xxs_grid) {
#ਜੇ DPCT_COMPATIBILITY_ਆਰਜ਼ੀ >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq3_xxs * bq2 = (ਅਚਲ ਖੰਡ_iq3_xxs *) vbq;

    ਅਚਲ ਪੂਰਨ ib32 = iqs;
    ਅਚਲ uint8_t  * q3 = bq2->qs + 8*ib32;
    ਅਚਲ uint16_t * gas = (ਅਚਲ uint16_t *)(bq2->qs + QK_K/4) + 2*ib32;
    ਅਚਲ int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = gas[0] | (gas[1] << 16);
    ਪੂਰਨ sumi = 0;
#ਹਦਾਇਤ unroll
    ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
        ਅਚਲ uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
        ਅਚਲ uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
        ਅਚਲ uint32_t sign_ਬਿੱਟ = iq_sign_ਬਿੱਟ(aux32);
        ਅਚਲ uint32_t signs0 = iq_sign_ਨਕਾਬ(sign_ਬਿੱਟ);
        ਅਚਲ uint32_t signs1 = iq_sign_ਨਕਾਬ(sign_ਬਿੱਟ >> 4);
        ਅਚਲ ਪੂਰਨ grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        ਅਚਲ ਪੂਰਨ grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((ਅਚਲ ਪੂਰਨ *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((ਅਚਲ ਪੂਰਨ *)q8 + 1), sumi);
        q8 += 8;
        aux32 >>= 7;
    }
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.5f;
    ਵਾਪਸ d * sumi;
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
    ਵਾਪਸ 0.f;
#ਜੇ_ਅੰਤ
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
    ਵਾਪਸ 0.f;
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <> ਬਣਤਰ reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_XXS> {
    ਸਥਿਰ constexpr ggml_type gtype = GGML_TYPE_IQ3_XXS;

    __dpct_inline__ ਦਸ਼ਮਲਵ ਸੰਚਾਲਕ()(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> ibx_ਵਿੱਥ,
                                     ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> d_ਵਿੱਥ, ਅਚਲ int8_t * q8_1_quant_ptr,
                                     ਅਚਲ sycl::half2 * q8_1_ds, ਅਚਲ ਪੂਰਨ & iqs) {
#ਜੇ DPCT_COMPATIBILITY_ਆਰਜ਼ੀ >= MIN_CC_DP4A && QK_K == 256
        ਅਚਲ uint8_t * ਅਧਾਰ = static_cast<ਅਚਲ uint8_t *>(vbq);
        ਅਚਲ uint8_t * qs   = ਅਧਾਰ + ibx_ਵਿੱਥ.ਪਹਿਲਾ;
        ਅਚਲ uint8_t * q3   = qs + 8 * iqs;
        ਅਚਲ uint16_t * gas = reinterpret_cast<ਅਚਲ uint16_t *>(qs + QK_K / 4) + 2 * iqs;
        ਅਚਲ int8_t * q8    = q8_1_quant_ptr + iqs * QK8_1;
        uint32_t aux32       = gas[0] | (gas[1] << 16);

        ਪੂਰਨ sumi = 0;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
            ਅਚਲ uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
            ਅਚਲ uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
            ਅਚਲ uint32_t sign_ਬਿੱਟ = iq_sign_ਬਿੱਟ(aux32);
            ਅਚਲ uint32_t signs0 = iq_sign_ਨਕਾਬ(sign_ਬਿੱਟ);
            ਅਚਲ uint32_t signs1 = iq_sign_ਨਕਾਬ(sign_ਬਿੱਟ >> 4);
            ਅਚਲ ਪੂਰਨ grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            ਅਚਲ ਪੂਰਨ grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((ਅਚਲ ਪੂਰਨ *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((ਅਚਲ ਪੂਰਨ *) q8 + 1), sumi);
            q8 += 8;
            aux32 >>= 7;
        }

        ਅਚਲ ggml_half d = *reinterpret_cast<ਅਚਲ ggml_half *>(ਅਧਾਰ + d_ਵਿੱਥ.ਪਹਿਲਾ);
        ਵਾਪਸ (ਦਸ਼ਮਲਵ) d * (0.5f + aux32) * q8_1_ds[iqs][0] * 0.5f * sumi;
#ਨਹੀਂਤਾਂ
        GGML_ਅਣਵਰਤਿਆ(vbq);
        GGML_ਅਣਵਰਤਿਆ(ibx_ਵਿੱਥ);
        GGML_ਅਣਵਰਤਿਆ(d_ਵਿੱਥ);
        GGML_ਅਣਵਰਤਿਆ(q8_1_quant_ptr);
        GGML_ਅਣਵਰਤਿਆ(q8_1_ds);
        GGML_ਅਣਵਰਤਿਆ(iqs);
        GGML_ABORT("IQ3_XXS reorder vec dot requires dp4a and QK_K == 256");
#ਜੇ_ਅੰਤ
    }
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_IQ3_S_Q8_1_MMVQ 1

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_iq3_s_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                   ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs,
                   ਅਚਲ uint32_t *iq3s_grid) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq3_s * bq2 = (ਅਚਲ ਖੰਡ_iq3_s *) vbq;

    ਅਚਲ ਪੂਰਨ ib32 = iqs;
    ਅਚਲ uint8_t  * qs = bq2->qs + 8*ib32;
    ਅਚਲ int8_t   * q8 = bq8_1[ib32].qs;
    ਪੂਰਨ sumi = 0;
    ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
        ਅਚਲ uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((bq2->qh[ib32] << (8 - 2*l)) & 256)];
        ਅਚਲ uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((bq2->qh[ib32] << (7 - 2*l)) & 256)];
        ਅਚਲ uint32_t signs = bq2->signs[4 * ib32 + l];
        ਅਚਲ uint32_t signs0 = iq_sign_ਨਕਾਬ(signs);
        ਅਚਲ uint32_t signs1 = iq_sign_ਨਕਾਬ(signs >> 4);
        ਅਚਲ ਪੂਰਨ grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        ਅਚਲ ਪੂਰਨ grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((ਅਚਲ ਪੂਰਨ *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((ਅਚਲ ਪੂਰਨ *)q8 + 1), sumi);
        q8 += 8;
    }
    ਅਚਲ ਦਸ਼ਮਲਵ d =
        (ਦਸ਼ਮਲਵ)bq2->d *
        (1 + 2 * ((bq2->scales[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf)) *
        bq8_1[ib32].ds[0];
    ਵਾਪਸ d * sumi;
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <> ਬਣਤਰ reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_S> {
    ਸਥਿਰ constexpr ggml_type gtype = GGML_TYPE_IQ3_S;

    __dpct_inline__ ਦਸ਼ਮਲਵ ਸੰਚਾਲਕ()(ਅਚਲ ਖਾਲੀ * __restrict__ vbq, ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> ibx_ਵਿੱਥ,
                                     ਅਚਲ std::pair<ਪੂਰਨ, ਪੂਰਨ> d_ਵਿੱਥ, ਅਚਲ int8_t * q8_1_quant_ptr,
                                     ਅਚਲ sycl::half2 * q8_1_ds, ਅਚਲ ਪੂਰਨ & iqs) {
#ਜੇ QK_K == 256
        ਅਚਲ uint8_t * ਅਧਾਰ     = static_cast<ਅਚਲ uint8_t *>(vbq);
        ਅਚਲ uint8_t * qs       = ਅਧਾਰ + ibx_ਵਿੱਥ.ਪਹਿਲਾ + 8 * iqs;
        ਅਚਲ uint8_t * qh       = ਅਧਾਰ + ibx_ਵਿੱਥ.second;
        ਅਚਲ uint8_t * signs    = ਅਧਾਰ + d_ਵਿੱਥ.ਪਹਿਲਾ + 4 * iqs;
        ਅਚਲ uint8_t * metadata = ਅਧਾਰ + d_ਵਿੱਥ.second;
        ਅਚਲ ggml_half d        = *reinterpret_cast<ਅਚਲ ggml_half *>(metadata);
        ਅਚਲ uint8_t * scales   = metadata + ਆਕਾਰ(ggml_half);
        ਅਚਲ int8_t * q8        = q8_1_quant_ptr + iqs * QK8_1;

        ਪੂਰਨ sumi = 0;
#ਹਦਾਇਤ unroll
        ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
            ਅਚਲ uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((qh[iqs] << (8 - 2*l)) & 256)];
            ਅਚਲ uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((qh[iqs] << (7 - 2*l)) & 256)];
            ਅਚਲ uint32_t signs0 = iq_sign_ਨਕਾਬ(signs[l]);
            ਅਚਲ uint32_t signs1 = iq_sign_ਨਕਾਬ(signs[l] >> 4);
            ਅਚਲ ਪੂਰਨ grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            ਅਚਲ ਪੂਰਨ grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((ਅਚਲ ਪੂਰਨ *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((ਅਚਲ ਪੂਰਨ *) q8 + 1), sumi);
            q8 += 8;
        }

        ਅਚਲ ਦਸ਼ਮਲਵ scale = 1 + 2 * ((scales[iqs / 2] >> (4 * (iqs % 2))) & 0xf);
        ਵਾਪਸ (ਦਸ਼ਮਲਵ) d * scale * q8_1_ds[iqs][0] * sumi;
#ਨਹੀਂਤਾਂ
        GGML_ਅਣਵਰਤਿਆ(vbq);
        GGML_ਅਣਵਰਤਿਆ(ibx_ਵਿੱਥ);
        GGML_ਅਣਵਰਤਿਆ(d_ਵਿੱਥ);
        GGML_ਅਣਵਰਤਿਆ(q8_1_quant_ptr);
        GGML_ਅਣਵਰਤਿਆ(q8_1_ds);
        GGML_ਅਣਵਰਤਿਆ(iqs);
        GGML_ABORT("IQ3_S reorder vec dot not supported for QK_K != 256");
#ਜੇ_ਅੰਤ
    }
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_IQ1_S_Q8_1_MMVQ 1

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_iq1_s_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                   ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs,
                   ਅਚਲ uint32_t *iq1s_grid_gpu) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq1_s * bq1 = (ਅਚਲ ਖੰਡ_iq1_s *) vbq;

    ਅਚਲ ਪੂਰਨ ib32 = iqs;
    ਪੂਰਨ sumi = 0;
    ਅਚਲ ਪੂਰਨ * q8 = (ਅਚਲ ਪੂਰਨ *)bq8_1[ib32].qs;
    ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
        ਅਚਲ ਪੂਰਨ * grid = (ਅਚਲ ਪੂਰਨ *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[ib32] >> 3*l) & 7) << 8)));
        ਪੂਰਨ grid0 = grid[0] & 0x0f0f0f0f;
        ਪੂਰਨ grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi = dpct::dp4a(q8[2 * l + 1], grid1,
                          dpct::dp4a(q8[2 * l + 0], grid0, sumi));
    }

    ਅਚਲ ਦਸ਼ਮਲਵ delta = bq1->qh[ib32] & 0x8000 ? -1-IQ1S_DELTA : -1+IQ1S_DELTA;
    ਅਚਲ ਦਸ਼ਮਲਵ d1q = (ਦਸ਼ਮਲਵ)bq1->d * (2*((bq1->qh[ib32] >> 12) & 7) + 1);
    ਅਚਲ ਦਸ਼ਮਲਵ d = d1q * bq8_1[ib32].ds[0];
    ਅਚਲ ਦਸ਼ਮਲਵ m = d1q * bq8_1[ib32].ds[1];
    ਵਾਪਸ d * sumi + m * delta;
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
#ਜੇ_ਅੰਤ
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_IQ1_M_Q8_1_MMVQ 1

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_iq1_m_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                   ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {
#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq1_m * bq1 = (ਅਚਲ ਖੰਡ_iq1_m *) vbq;

    ਅਚਲ ਪੂਰਨ ib32 = iqs;
    ਪੂਰਨ   sumi[2] = {0, 0};
    ਦਸ਼ਮਲਵ sumf[2] = {0.f, 0.f};

    ਅਚਲ ਪੂਰਨ * q8 = (ਅਚਲ ਪੂਰਨ *)bq8_1[ib32].qs;
    ਲਈ (ਪੂਰਨ l = 0; l < 4; ++l) {
        ਅਚਲ ਪੂਰਨ * grid = (ਅਚਲ ਪੂਰਨ *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 7) << 8)));
        ਪੂਰਨ grid0 = grid[0] & 0x0f0f0f0f;
        ਪੂਰਨ grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi[l / 2] = dpct::dp4a(q8[2 * l + 1], grid1,
                                 dpct::dp4a(q8[2 * l + 0], grid0, sumi[l / 2]));
        ਅਚਲ ਦਸ਼ਮਲਵ delta = (bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 0x08 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA;
        ਅਚਲ ਪੂਰਨ sumy = dpct::dp4a(q8[2 * l + 1], 0x01010101,
                                    dpct::dp4a(q8[2 * l + 0], 0x01010101, 0));
        sumf[l/2] += delta*sumy;
    }

    iq1m_scale_t scale;
    ਅਚਲ uint16_t * sc = (ਅਚਲ uint16_t *)bq1->scales;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)scale.f16 * bq8_1[ib32].ds[0];
    ਵਾਪਸ d * ((sumi[0] + sumf[0]) * (2*((sc[ib32/2] >> 6*(ib32%2)) & 0x7) + 1) + (sumi[1] + sumf[1]) * (2*((sc[ib32/2] >> (6*(ib32%2)+3)) & 0x7) + 1));
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
#ਜੇ_ਅੰਤ
}


#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_IQ4_NL_Q8_1_MMVQ 2

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_iq4_nl_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                    ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {

    ਅਚਲ ਖੰਡ_iq4_nl * bq = (ਅਚਲ ਖੰਡ_iq4_nl *) vbq;

    ਅਚਲ uint16_t * q4 = (ਅਚਲ uint16_t *)bq->qs + 2*iqs;
    ਅਚਲ int32_t  * q8 = (ਅਚਲ int32_t  *)bq8_1->qs + iqs;

    ਅਚਲ uint8_t * ਮੁੱਲਾਂ = (ਅਚਲ uint8_t *)kvalues_iq4nl;

    ਪੂਰਨ v1, v2;
    ਪੂਰਨ sumi1 = 0, sumi2 = 0;
    ਲਈ (ਪੂਰਨ l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        ਅਚਲ uint32_t aux = q4[2*l] | (q4[2*l+1] << 16);
        get_int_from_ਸਾਰਣੀ_16(aux, ਮੁੱਲਾਂ, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[l + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[l + 4], sumi2);
    }

    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)bq->d * bq8_1->ds[0];
    ਵਾਪਸ d * (sumi1 + sumi2);
}


#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ VDR_IQ4_XS_Q8_1_MMVQ 1

ਸਥਿਰ __dpct_inline__ ਦਸ਼ਮਲਵ
vec_dot_iq4_xs_q8_1(ਅਚਲ ਖਾਲੀ *__restrict__ vbq,
                    ਅਚਲ ਖੰਡ_q8_1 *__restrict__ bq8_1, ਅਚਲ ਪੂਰਨ &iqs) {

#ਜੇ QK_K == 256
    ਅਚਲ ਖੰਡ_iq4_xs * bq4 = (ਅਚਲ ਖੰਡ_iq4_xs *) vbq;
    ਅਚਲ uint8_t * ਮੁੱਲਾਂ = (ਅਚਲ uint8_t *)kvalues_iq4nl;

    // iqs is 0...7
    ਅਚਲ ਪੂਰਨ ib32 = iqs;
    ਅਚਲ int32_t  * q8 = (ਅਚਲ ਪੂਰਨ *)bq8_1[ib32].qs;
    ਅਚਲ uint32_t * q4 = (ਅਚਲ uint32_t *)bq4->qs + 4*ib32;
    ਅਚਲ int8_t ls = ((bq4->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((bq4->scales_h >> 2*ib32) & 3) << 4);
    ਅਚਲ ਦਸ਼ਮਲਵ d = (ਦਸ਼ਮਲਵ)bq4->d * (ls - 32) * bq8_1[ib32].ds[0];
    ਪੂਰਨ v1, v2;
    ਪੂਰਨ sumi1 = 0, sumi2 = 0;
    ਲਈ (ਪੂਰਨ j = 0; j < 4; ++j) {
        get_int_from_ਸਾਰਣੀ_16(q4[j], ਮੁੱਲਾਂ, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[j + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[j + 4], sumi2);
    }
    ਵਾਪਸ d * (sumi1 + sumi2);
#ਨਹੀਂਤਾਂ
    ਪ੍ਰਤਿਜ੍ਞਾ(ਝੂਠ);
#ਜੇ_ਅੰਤ
}

#ਜੇ_ਅੰਤ // GGML_SYCL_VECDOTQ_HPP
