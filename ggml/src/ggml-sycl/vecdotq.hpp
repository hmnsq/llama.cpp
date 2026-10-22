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

#અન્તર્ભાવ "dpct/helper.hpp"
#અન્તર્ભાવ "ggml.h"
#અન્તર્ભાવ "type.hpp"
#અન્તર્ભાવ "quants.hpp"

પ્રકારવ્યાખ્યા દશાંશ (*vec_dot_q_sycl_t)(અચલ શૂન્ય * __restrict__ vbq, અચલ block_q8_1 * __restrict__ bq8_1,
                                  અચલ પૂર્ણાંક & iqs);

સ્થિર __dpct_inline__ પૂર્ણાંક get_int_b1(અચલ શૂન્ય * x, અચલ પૂર્ણાંક & i32) {
    અચલ uint8_t * x8 = (અચલ uint8_t *) x;

    પૂર્ણાંક x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    પાછા x32;
}

સ્થિર __dpct_inline__ પૂર્ણાંક get_int_b2(અચલ શૂન્ય * x, અચલ પૂર્ણાંક & i32) {
    અચલ uint16_t * x16 = (અચલ uint16_t *) x; // assume at least 2 byte alignment

    પૂર્ણાંક x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    પાછા x32;
}

સ્થિર __dpct_inline__ પૂર્ણાંક get_int_b4(અચલ શૂન્ય * x, અચલ પૂર્ણાંક & i32) {
    પાછા ((અચલ પૂર્ણાંક *) x)[i32]; // assume at least 4 byte alignment
}

સ્થિર __dpct_inline__ પૂર્ણાંક get_int_from_int8(અચલ int8_t* x8, અચલ પૂર્ણાંક& i32) {
  અચલ uint16_t* x16 =
      (અચલ uint16_t*)(x8 + કદ(પૂર્ણાંક) * i32); // assume at least 2 byte
                                                 // alignment

  પૂર્ણાંક x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  પાછા x32;
}

સ્થિર __dpct_inline__ પૂર્ણાંક get_int_from_uint8(
    અચલ uint8_t* x8,
    અચલ પૂર્ણાંક& i32) {
  અચલ uint16_t* x16 =
      (અચલ uint16_t*)(x8 + કદ(પૂર્ણાંક) * i32); // assume at least 2 byte
                                                 // alignment

  પૂર્ણાંક x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  પાછા x32;
}

સ્થિર __dpct_inline__ પૂર્ણાંક get_int_from_int8_aligned(
    અચલ int8_t* x8,
    અચલ પૂર્ણાંક& i32) {
  પાછા *(
      (અચલ પૂર્ણાંક*)(x8 + કદ(પૂર્ણાંક) * i32)); // assume at least 4 byte alignment
}

સ્થિર __dpct_inline__ પૂર્ણાંક get_int_from_uint8_aligned(
    અચલ uint8_t* x8,
    અચલ પૂર્ણાંક& i32) {
  પાછા *(
      (અચલ પૂર્ણાંક*)(x8 + કદ(પૂર્ણાંક) * i32)); // assume at least 4 byte alignment
}

સ્થિર __dpct_inline__ પૂર્ણાંક byte_sub_4(અચલ પૂર્ણાંક a, અચલ પૂર્ણાંક b) {
  અચલ uint32_t ua = static_cast<uint32_t>(a);
  અચલ uint32_t ub = static_cast<uint32_t>(b);
  પાછા static_cast<પૂર્ણાંક>(((ua | 0x80808080u) - ub) ^ 0x80808080u);
}

સ્થિર __dpct_inline__ દશાંશ vec_dot_q6_K_q8_1_impl_mmvq_scalar(
    અચલ પૂર્ણાંક vl, અચલ પૂર્ણાંક vh, અચલ પૂર્ણાંક u0, અચલ પૂર્ણાંક u1, અચલ int8_t sc0,
    અચલ int8_t sc1, અચલ દશાંશ d, અચલ દશાંશ d80, અચલ દશાંશ d81) {
    static_assert(QR6_K == 2, "q6_K MMVQ scalar fast path assumes QR6_K == 2");

    અચલ પૂર્ણાંક vil0 = (vl >> 0) & 0x0F0F0F0F;
    અચલ પૂર્ણાંક vih0 = ((vh >> 0) << 4) & 0x30303030;
    અચલ પૂર્ણાંક vi0 = byte_sub_4(vil0 | vih0, 0x20202020);

    અચલ પૂર્ણાંક vil1 = (vl >> 4) & 0x0F0F0F0F;
    અચલ પૂર્ણાંક vih1 = ((vh >> 4) << 4) & 0x30303030;
    અચલ પૂર્ણાંક vi1 = byte_sub_4(vil1 | vih1, 0x20202020);

    અચલ દશાંશ sumf =
        d80 * (dpct::dp4a(vi0, u0, 0) * sc0) +
        d81 * (dpct::dp4a(vi1, u1, 0) * sc1);

    પાછા d * sumf;
}

સ્થિર __dpct_inline__ શૂન્ય get_int_from_table_16(અચલ uint32_t &q4,
                                                  અચલ uint8_t *કિંમતો,
                                                  પૂર્ણાંક &val1, પૂર્ણાંક &val2) {

    uint32_t aux32; અચલ uint8_t * q8 = (અચલ uint8_t *)&aux32;
    aux32 = q4 & 0x0f0f0f0f;
    uint16_t v1 = કિંમતો[q8[0]] | (કિંમતો[q8[1]] << 8);
    uint16_t v2 = કિંમતો[q8[2]] | (કિંમતો[q8[3]] << 8);
    val1 = v1 | (v2 << 16);
    aux32 = (q4 >> 4) & 0x0f0f0f0f;
    v1 = કિંમતો[q8[0]] | (કિંમતો[q8[1]] << 8);
    v2 = કિંમતો[q8[2]] | (કિંમતો[q8[3]] << 8);
    val2 = v1 | (v2 << 16);
}

સ્થિર __dpct_inline__ sycl::int2 get_int_from_table_16(
    અચલ પૂર્ણાંક& q4, અચલ int8_t* table) {
  અચલ uint32_t* table32 = (અચલ uint32_t*)table;
  uint32_t tmp[2];
  અચલ uint32_t નીચલું_ઉપલું_selection_indices =
      (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
  માટે (uint32_t i = 0; i < 2; ++i) {
    અચલ uint32_t shift = 16 * i;

    અચલ uint32_t નીચલું =
        dpct::byte_level_permute(table32[0], table32[1], q4 >> shift);
    અચલ uint32_t ઉપલું =
        dpct::byte_level_permute(table32[2], table32[3], q4 >> shift);
    tmp[i] = dpct::byte_level_permute(
        નીચલું, ઉપલું, નીચલું_ઉપલું_selection_indices >> shift);
  }
  પાછા sycl::int2(
      dpct::byte_level_permute(tmp[0], tmp[1], 0x6420),
      dpct::byte_level_permute(tmp[0], tmp[1], 0x7531));
}

// Four E2M1 codes (one per byte, bits 0..3) to their kvalues_mxfp4 int8 values. SWAR arithmetic
// replaces get_int_from_table_16 for MXFP4: dpct::byte_level_permute is emulated with 64-bit shifts,
// eight per int, which made the MXFP4 GEMV compute-bound on Intel GPUs.
// Magnitudes 0,1,2,3,4,6,8,12 = m + [m>=5] + [m>=6] + 3*[m>=7]; each byte stays below 256, so the
// byte-wise adds never carry. -0 (code 8) is left as 0 so the two's-complement +1 cannot carry either.
સ્થિર __dpct_inline__ પૂર્ણાંક mxfp4_codes_to_int8(અચલ uint32_t x) {
    અચલ uint32_t m   = x & 0x07070707u;
    અચલ uint32_t ge5 = ((m + 0x03030303u) >> 3) & 0x01010101u;
    અચલ uint32_t ge6 = ((m + 0x02020202u) >> 3) & 0x01010101u;
    અચલ uint32_t ge7 = ((m + 0x01010101u) >> 3) & 0x01010101u;
    અચલ uint32_t mag = m + ge5 + ge6 + 3u * ge7;
    અચલ uint32_t nz  = ((mag + 0x7f7f7f7fu) >> 7) & 0x01010101u;
    અચલ uint32_t neg = (x >> 3) & nz & 0x01010101u;
    પાછા (પૂર્ણાંક) ((mag ^ (neg * 0xffu)) + neg);
}

// Same result as get_int_from_table_16(q4, kvalues_mxfp4): x = low nibbles, y = high nibbles.
સ્થિર __dpct_inline__ sycl::int2 get_int_from_mxfp4(અચલ પૂર્ણાંક q4) {
    પાછા sycl::int2(mxfp4_codes_to_int8((uint32_t) q4 & 0x0f0f0f0fu),
                      mxfp4_codes_to_int8(((uint32_t) q4 >> 4) & 0x0f0f0f0fu));
}

#define VDR_Q2_K_Q8_1_MMVQ 1

// contiguous v/x values
સ્થિર __dpct_inline__ દશાંશ vec_dot_q2_K_q8_1_impl_mmvq(
    અચલ પૂર્ણાંક &v, અચલ પૂર્ણાંક *__restrict__ u, અચલ uint8_t *__restrict__ scales,
    અચલ sycl::half2 &dm2, અચલ દશાંશ *__restrict__ d8) {

    દશાંશ sumf_d = 0.0f;
    દશાંશ sumf_m = 0.0f;

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < QR2_K; ++i) {
        અચલ પૂર્ણાંક sc = scales[2*i];

        અચલ પૂર્ણાંક vi = (v >> (2*i)) & 0x03030303;

        sumf_d +=
            d8[i] * (dpct::dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        પૂર્ણાંક m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] *
                  dpct::dp4a(
                      m, u[i],
                      0); // multiply constant q2_K part with sum of q8_1 values
    }

    અચલ sycl::float2 dm2f =
        dm2.convert<દશાંશ, sycl::rounding_mode::automatic>();

    પાછા dm2f.x() * sumf_d - dm2f.y() * sumf_m;
}


#define VDR_Q3_K_Q8_1_MMVQ 1

// contiguous v/x values
સ્થિર __dpct_inline__ દશાંશ vec_dot_q3_K_q8_1_impl_mmvq(
    અચલ પૂર્ણાંક &vl, અચલ પૂર્ણાંક &vh, અચલ પૂર્ણાંક *__restrict__ u,
    અચલ uint8_t *__restrict__ scales, અચલ પૂર્ણાંક &scale_offset,
    અચલ દશાંશ &d3, અચલ દશાંશ *__restrict__ d8) {

    દશાંશ sumf = 0.0f;

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < QR3_K; ++i) {
        અચલ પૂર્ણાંક isc = scale_offset + 2*i;

        અચલ પૂર્ણાંક isc_નીચલું = isc % (QK_K/32);
        અચલ પૂર્ણાંક sc_shift_નીચલું = 4 * (isc / (QK_K/32));
        અચલ પૂર્ણાંક sc_નીચલું  = (scales[isc_નીચલું] >> sc_shift_નીચલું) & 0xF;

        અચલ પૂર્ણાંક isc_ઉપલું = isc % (QK_K/64);
        અચલ પૂર્ણાંક sc_shift_ઉપલું = 2 * (isc / (QK_K/64));
        અચલ પૂર્ણાંક sc_ઉપલું = ((scales[(QK_K/32) + isc_ઉપલું] >> sc_shift_ઉપલું) & 3) << 4;

        અચલ પૂર્ણાંક sc = (sc_નીચલું | sc_ઉપલું) - 32;

        અચલ પૂર્ણાંક vil = (vl >> (2*i)) & 0x03030303;

        અચલ પૂર્ણાંક vih = ((vh >> i) << 2) & 0x04040404;

        અચલ પૂર્ણાંક vi =
            dpct::vectorized_દ્વિઅંકી<sycl::char4>(vil, vih, dpct::sub_sat());

        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    પાછા d3 * sumf;
}

#define VDR_Q4_K_Q8_1_MMVQ 2

// contiguous v/x values
સ્થિર __dpct_inline__ દશાંશ vec_dot_q4_K_q8_1_impl_vmmq(
    અચલ પૂર્ણાંક *__restrict__ v, અચલ પૂર્ણાંક *__restrict__ u,
    અચલ uint8_t *__restrict__ sc, અચલ uint8_t *__restrict__ m,
    અચલ sycl::half2 &dm4, અચલ દશાંશ *__restrict__ d8) {

    દશાંશ sumf_d = 0.0f;
    દશાંશ sumf_m = 0.0f;

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < QR4_K; ++i) {
        અચલ પૂર્ણાંક v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        અચલ પૂર્ણાંક v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        અચલ પૂર્ણાંક dot1 =
            dpct::dp4a(v1i, u[2 * i + 1],
                       dpct::dp4a(v0i, u[2 * i + 0], 0)); // SIMD dot product
        અચલ પૂર્ણાંક dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 1],
                       dpct::dp4a(0x01010101, u[2 * i + 0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    અચલ sycl::float2 dm4f =
        dm4.convert<દશાંશ, sycl::rounding_mode::automatic>();

    પાછા dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


#define VDR_Q5_K_Q8_1_MMVQ 2

// contiguous v/x values
સ્થિર __dpct_inline__ દશાંશ vec_dot_q5_K_q8_1_impl_vmmq(
    અચલ પૂર્ણાંક *__restrict__ vl, અચલ પૂર્ણાંક *__restrict__ vh,
    અચલ પૂર્ણાંક *__restrict__ u, અચલ uint8_t *__restrict__ sc,
    અચલ uint8_t *__restrict__ m, અચલ sycl::half2 &dm5,
    અચલ દશાંશ *__restrict__ d8) {

    દશાંશ sumf_d = 0.0f;
    દશાંશ sumf_m = 0.0f;

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < QR5_K; ++i) {
        અચલ પૂર્ણાંક vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        અચલ પૂર્ણાંક vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        અચલ પૂર્ણાંક vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        અચલ પૂર્ણાંક vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        અચલ પૂર્ણાંક v0i = vl0i | vh0i;
        અચલ પૂર્ણાંક v1i = vl1i | vh1i;

        અચલ પૂર્ણાંક dot1 =
            dpct::dp4a(v0i, u[2 * i + 0],
                       dpct::dp4a(v1i, u[2 * i + 1], 0)); // SIMD dot product
        અચલ પૂર્ણાંક dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 0],
                       dpct::dp4a(0x01010101, u[2 * i + 1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    અચલ sycl::float2 dm5f =
        dm5.convert<દશાંશ, sycl::rounding_mode::automatic>();

    પાછા dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}


#define VDR_Q6_K_Q8_1_MMVQ 1

// contiguous v/x values
સ્થિર __dpct_inline__ દશાંશ
vec_dot_q6_K_q8_1_impl_mmvq(અચલ પૂર્ણાંક &vl, અચલ પૂર્ણાંક &vh,
                            અચલ પૂર્ણાંક *__restrict__ u,
                            અચલ int8_t *__restrict__ scales, અચલ દશાંશ &d,
                            અચલ દશાંશ *__restrict__ d8) {
    પાછા vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
}

#define VDR_Q1_0_Q8_1_MMVQ 1
#define VDR_Q1_0_Q8_1_MMQ  4

સ્થિર __dpct_inline__ દશાંશ
vec_dot_q1_0_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                  અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {

    અચલ block_q1_0 * bq1_0 = (અચલ block_q1_0 *) vbq;

    અચલ block_q8_1 * bq8_1_chunk = bq8_1 + iqs;
    અચલ દશાંશ        d1          = bq1_0->d;
    અચલ પૂર્ણાંક          v           = get_int_from_uint8_aligned(bq1_0->qs, iqs);

    પૂર્ણાંક vi_bytes[8];
#pragma unroll
    માટે (પૂર્ણાંક j = 0; j < 8; ++j) {
        અચલ પૂર્ણાંક shift = j * 4;
        અચલ પૂર્ણાંક bits4 = (v >> shift) & 0x0F;
        અચલ પૂર્ણાંક b0    = (bits4 & 0x01) ? 1 : -1;
        અચલ પૂર્ણાંક b1    = (bits4 & 0x02) ? 1 : -1;
        અચલ પૂર્ણાંક b2    = (bits4 & 0x04) ? 1 : -1;
        અચલ પૂર્ણાંક b3    = (bits4 & 0x08) ? 1 : -1;
        vi_bytes[j]     = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }

    પૂર્ણાંક sumi = 0;
#pragma unroll
    માટે (પૂર્ણાંક j = 0; j < 8; ++j) {
        અચલ પૂર્ણાંક u = get_int_from_int8_aligned(bq8_1_chunk->qs, j);
        sumi        = ggml_sycl_dp4a(vi_bytes[j], u, sumi);
    }

    પાછા d1 * bq8_1_chunk->ds[0] * sumi;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

template <ggml_type T> સંરચના reorder_vec_dot_q_sycl {
    static_assert(T != T, "ggml_type for reorder vecdot not implemented");
};

// For some types the weight side of the dot product does not depend on the destination column, so a
// multi-column mul_mat_vec can unpack it once per block instead of once per column. Such a type adds
// load() and dot() next to operator() and opts in here. See reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>.
template <ggml_type T> સંરચના reorder_vec_dot_shared_weights {
    સ્થિર constexpr તાર્કિક value = ખોટું;
};

template <> સંરચના reorder_vec_dot_shared_weights<GGML_TYPE_Q4_K> {
    સ્થિર constexpr તાર્કિક value = સાચું;
};

template <> સંરચના reorder_vec_dot_shared_weights<GGML_TYPE_Q5_K> {
    સ્થિર constexpr તાર્કિક value = સાચું;
};

template <ggml_type T> સંરચના reorder_vec_dot_shared_activations {
    સ્થિર constexpr તાર્કિક value = ખોટું;
};

template <> સંરચના reorder_vec_dot_shared_activations<GGML_TYPE_Q4_K> {
    સ્થિર constexpr તાર્કિક value = સાચું;
};

template <> સંરચના reorder_vec_dot_shared_activations<GGML_TYPE_Q5_K> {
    સ્થિર constexpr તાર્કિક value = સાચું;
};

template <> સંરચના reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0> {
    સ્થિર constexpr ggml_type gtype = GGML_TYPE_Q4_0;

    using q4_0_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_0>;
    using q4_0_traits = typename q4_0_block::traits;

    __dpct_inline__ દશાંશ vec_dot_q4_0_q8_1_impl(અચલ પૂર્ણાંક * v, અચલ પૂર્ણાંક * u, અચલ દશાંશ & d4, અચલ sycl::half2 & ds8) {
        પૂર્ણાંક sumi = 0;

#pragma unroll
        માટે (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            અચલ પૂર્ણાંક vi0 = (v[i] >> 0) & 0x0F0F0F0F;
            અચલ પૂર્ણાંક vi1 = (v[i] >> 4) & 0x0F0F0F0F;

            // SIMD dot product of quantized values
            sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
            sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
        }

        અચલ sycl::float2 ds8f = ds8.convert<દશાંશ, sycl::rounding_mode::automatic>();

        // second part effectively subtracts 8 from each quant value
        પાછા d4 * (sumi * ds8f.x() - (8 * q4_0_traits::vdr_mmvq / q4_0_traits::qi) * ds8f.y());
    }

    __dpct_inline__ દશાંશ operator()(અચલ શૂન્ય * __restrict__ vbq, અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> ibx_offset,
                                     અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> d_offset, અચલ int8_t * q8_1_quant_ptr,
                                     અચલ sycl::half2 * q8_1_ds, અચલ પૂર્ણાંક & iqs) {
        અચલ uint8_t * bq4_0 = static_cast<અચલ uint8_t *>(vbq) + ibx_offset.first;
        અચલ ggml_half d = *(reinterpret_cast<અચલ ggml_half *>(static_cast<અચલ uint8_t *>(vbq) + d_offset.first));
        પૂર્ણાંક             v[q4_0_traits::vdr_mmvq];
        પૂર્ણાંક             u[2 * q4_0_traits::vdr_mmvq];


#pragma unroll
        માટે (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            v[i]         = get_int_from_uint8(bq4_0, iqs + i);
            u[2 * i + 0] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
            u[2 * i + 1] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i + q4_0_traits::qi);
        }

        પાછા vec_dot_q4_0_q8_1_impl(v, u, d, *q8_1_ds);
    }
};

// Load four contiguous dwords per operand instead of loading each value separately.
સંરચના reorder_vec_dot_q8_0_wide {
    સ્થિર constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    using q8_0_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_0>;
    using q8_0_traits = typename q8_0_block::traits;

    __dpct_inline__ દશાંશ operator()(અચલ શૂન્ય * __restrict__ vbq, અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> ibx_offset,
                                     અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> d_offset, અચલ int8_t * q8_1_quant_ptr,
                                     અચલ sycl::half2 * q8_1_ds, અચલ પૂર્ણાંક & iqs) {
        static_assert(q8_0_traits::vdr_mmvq == 4, "the wide load moves exactly four dwords");

        અચલ uint8_t * base = static_cast<અચલ uint8_t *>(vbq);
        અચલ int8_t *  qs   = reinterpret_cast<અચલ int8_t *>(base + ibx_offset.first);
        અચલ ggml_half d    = *reinterpret_cast<અચલ ggml_half *>(base + d_offset.first);

        અચલ sycl::int4 v = *reinterpret_cast<અચલ sycl::int4 *>(qs + કદ(પૂર્ણાંક) * iqs);
        અચલ sycl::int4 u = *reinterpret_cast<અચલ sycl::int4 *>(q8_1_quant_ptr + કદ(પૂર્ણાંક) * iqs);

        પૂર્ણાંક sumi = 0;
#pragma unroll
        માટે (પૂર્ણાંક i = 0; i < 4; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        અચલ sycl::half2 ds_કિંમતો = *q8_1_ds;
        પાછા static_cast<દશાંશ>(d) * static_cast<દશાંશ>(ds_કિંમતો[0]) * sumi;
    }
};

template <> સંરચના reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0> {
    સ્થિર constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    using q8_0_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_0>;
    using q8_0_traits = typename q8_0_block::traits;

    __dpct_inline__ દશાંશ operator()(અચલ શૂન્ય * __restrict__ vbq, અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> ibx_offset,
                                     અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> d_offset, અચલ int8_t * q8_1_quant_ptr,
                                     અચલ sycl::half2 * q8_1_ds, અચલ પૂર્ણાંક & iqs,
                                     અચલ પૂર્ણાંક vdr = q8_0_traits::vdr_mmvq) {
        અચલ uint8_t * base = static_cast<અચલ uint8_t *>(vbq);
        અચલ int8_t *  qs   = reinterpret_cast<અચલ int8_t *>(base + ibx_offset.first);
        અચલ ggml_half  d   = *reinterpret_cast<અચલ ggml_half *>(base + d_offset.first);

        પૂર્ણાંક v[q8_0_traits::vdr_mmvq];
        પૂર્ણાંક u[q8_0_traits::vdr_mmvq];

#pragma unroll
        માટે (પૂર્ણાંક i = 0; i < vdr; ++i) {
            v[i] = get_int_from_int8(qs, iqs + i);
            u[i] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
        }

        પૂર્ણાંક sumi = 0;
#pragma unroll
        માટે (પૂર્ણાંક i = 0; i < vdr; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        અચલ sycl::half2 ds_કિંમતો = *q8_1_ds;
        પાછા static_cast<દશાંશ>(d) * static_cast<દશાંશ>(ds_કિંમતો[0]) * sumi;
    }
};

template <> સંરચના reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K> {
    સ્થિર constexpr ggml_type gtype = GGML_TYPE_Q2_K;

    using q2_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q2_K>;
    using q2_k_traits = typename q2_k_block::traits;

    __dpct_inline__ દશાંશ operator()(અચલ શૂન્ય * __restrict__ vbq, અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> ibx_offset,
                                     અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> d_offset, અચલ int8_t * q8_1_quant_ptr,
                                     અચલ sycl::half2 * q8_1_ds, અચલ પૂર્ણાંક & iqs) {
        અચલ uint8_t *    base   = static_cast<અચલ uint8_t *>(vbq);
        અચલ uint8_t *    qs     = base + ibx_offset.first;
        અચલ uint8_t *    scales = base + d_offset.first;
        અચલ ggml_half2 * dm     = reinterpret_cast<અચલ ggml_half2 *>(base + d_offset.second);

        અચલ પૂર્ણાંક bq8_offset   = QR2_K * (iqs / QI8_1);
        અચલ પૂર્ણાંક scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        અચલ પૂર્ણાંક v = get_int_from_uint8_aligned(qs, iqs);

        પૂર્ણાંક   u[QR2_K];
        દશાંશ d8[QR2_K];

#pragma unroll
        માટે (પૂર્ણાંક i = 0; i < QR2_K; ++i) {
            અચલ int8_t * quant_base_ptr = q8_1_quant_ptr + (bq8_offset + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_base_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_offset + i))[0];
        }

        પાછા vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales + scale_offset, *dm, d8);
    }
};

template <> સંરચના reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K> {
    સ્થિર constexpr ggml_type gtype = GGML_TYPE_Q3_K;

    using q3_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q3_K>;
    using q3_k_traits = typename q3_k_block::traits;

    __dpct_inline__ દશાંશ operator()(અચલ શૂન્ય * __restrict__ vbq, અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> ibx_offset,
                                     અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> d_offset, અચલ int8_t * q8_1_quant_ptr,
                                     અચલ sycl::half2 * q8_1_ds, અચલ પૂર્ણાંક & iqs) {
        અચલ uint8_t *  base   = static_cast<અચલ uint8_t *>(vbq);
        અચલ uint8_t *  qs     = base + ibx_offset.first;
        અચલ uint8_t *  hmask  = base + ibx_offset.second;
        અચલ uint8_t *  scales = base + d_offset.first;
        અચલ ggml_half  d      = *reinterpret_cast<અચલ ggml_half *>(base + d_offset.second);

        અચલ પૂર્ણાંક bq8_offset   = QR3_K * (iqs / (QI3_K / 2));
        અચલ પૂર્ણાંક scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        અચલ પૂર્ણાંક vl = get_int_from_uint8(qs, iqs);
        અચલ પૂર્ણાંક vh = ~get_int_from_uint8(hmask, iqs % (QI3_K / 2)) >> bq8_offset;

        પૂર્ણાંક   u[QR3_K];
        દશાંશ d8[QR3_K];

#pragma unroll
        માટે (પૂર્ણાંક i = 0; i < QR3_K; ++i) {
            અચલ int8_t * quant_base_ptr = q8_1_quant_ptr + (bq8_offset + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_base_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_offset + i))[0];
        }

        પાછા vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, scales, scale_offset, static_cast<દશાંશ>(d), d8);
    }
};

સ્થિર અંતઃરેખ દશાંશ vec_dot_q4_K_q8_1_common(અચલ પૂર્ણાંક * __restrict__ q4, અચલ uint16_t * __restrict__ scales,
                                             અચલ ggml_half2 & dm, અચલ block_q8_1 * __restrict__ bq8_1,
                                             અચલ પૂર્ણાંક &        iqs) {
    પૂર્ણાંક   v[2];
    પૂર્ણાંક   u[2 * QR4_K];
    દશાંશ d8[QR4_K];

    v[0] = q4[0];
    v[1] = q4[4];

    uint16_t  aux[2];
    અચલ પૂર્ણાંક j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
    જો (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } નહીંતર {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }

    અચલ uint8_t * sc = (અચલ uint8_t *) aux;
    અચલ uint8_t * m  = sc + 2;

    અચલ પૂર્ણાંક bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));

    માટે (પૂર્ણાંક i = 0; i < QR4_K; ++i) {
        અચલ block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i]                   = bq8i->ds[0];

        અચલ પૂર્ણાંક * q8 = (અચલ પૂર્ણાંક *) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0]   = q8[0];
        u[2 * i + 1]   = q8[4];
    }

    પાછા vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
}

template <> સંરચના reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K> {
    સ્થિર constexpr ggml_type gtype = GGML_TYPE_Q4_K;

    using q4_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;
    using q4_k_traits = typename q4_k_block::traits;

    સંરચના weights {
        પૂર્ણાંક        v[2];
        uint16_t   aux[2];
        ggml_half2 dm;
        પૂર્ણાંક        bq8_offset;
    };

    સંરચના activations {
        પૂર્ણાંક   u[2 * QR4_K];
        દશાંશ d8[QR4_K];
    };

    __dpct_inline__ સ્થિર weights load(અચલ શૂન્ય * __restrict__ vbq, અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> ibx_offset,
                                        અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> d_offset, અચલ પૂર્ણાંક & iqs) {
        અચલ uint8_t *    base = static_cast<અચલ uint8_t *>(vbq);
        અચલ uint8_t *    qs   = base + ibx_offset.first;
        અચલ uint8_t *    scs  = base + d_offset.first;
        અચલ ggml_half2 * dms  = reinterpret_cast<અચલ ggml_half2 *>(base + d_offset.second);

        weights w;
        w.bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));

        અચલ પૂર્ણાંક *      q4     = (અચલ પૂર્ણાંક *) (qs + 16 * w.bq8_offset + 4 * ((iqs / 2) % 4));
        અચલ uint16_t * scales = (અચલ uint16_t *) scs;

        w.v[0] = q4[0];
        w.v[1] = q4[4];

        અચલ પૂર્ણાંક j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        જો (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } નહીંતર {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        પાછા w;
    }

    __dpct_inline__ સ્થિર activations load_activations(અચલ int8_t * q8_1_quant_ptr,
                                                        અચલ sycl::half2 * q8_1_ds, અચલ પૂર્ણાંક & iqs) {
        activations a;
        અચલ પૂર્ણાંક bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        માટે (પૂર્ણાંક i = 0; i < QR4_K; ++i) {
            અચલ int8_t * quant_base_ptr = q8_1_quant_ptr + (bq8_offset + i) * QK8_1;
            sycl::half2    ds_કિંમતો      = *(q8_1_ds + bq8_offset + i);

            a.d8[i] = ds_કિંમતો[0];

            અચલ પૂર્ણાંક * q8 = (અચલ પૂર્ણાંક *) quant_base_ptr + ((iqs / 2) % 4);
            a.u[2 * i + 0] = q8[0];
            a.u[2 * i + 1] = q8[4];
        }

        પાછા a;
    }

    __dpct_inline__ સ્થિર દશાંશ apply(અચલ weights & w, અચલ activations & a) {
        અચલ uint8_t * sc = (અચલ uint8_t *) w.aux;
        અચલ uint8_t * m  = sc + 2;

        પાછા vec_dot_q4_K_q8_1_impl_vmmq(w.v, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ સ્થિર દશાંશ dot(અચલ weights & w, અચલ int8_t * q8_1_quant_ptr,
                                     અચલ sycl::half2 * q8_1_ds, અચલ પૂર્ણાંક & iqs) {
        અચલ auto a = load_activations(q8_1_quant_ptr, q8_1_ds, iqs);

        પાછા apply(w, a);
    }

    __dpct_inline__ દશાંશ operator()(અચલ શૂન્ય * __restrict__ vbq, અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> ibx_offset,
                                     અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> d_offset, અચલ int8_t * q8_1_quant_ptr,
                                     અચલ sycl::half2 * q8_1_ds, અચલ પૂર્ણાંક & iqs) {
        પાછા dot(load(vbq, ibx_offset, d_offset, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

template <> સંરચના reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K> {
    સ્થિર constexpr ggml_type gtype = GGML_TYPE_Q5_K;

    using q5_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q5_K>;
    using q5_k_traits = typename q5_k_block::traits;

    સંરચના weights {
        પૂર્ણાંક        vl[2];
        પૂર્ણાંક        vh[2];
        uint16_t   aux[2];
        ggml_half2 dm;
    };

    // same activation layout as Q4_K
    static_assert(QR5_K == QR4_K);
    using activations = reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::activations;

    __dpct_inline__ સ્થિર weights load(અચલ શૂન્ય * __restrict__ vbq, અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> ibx_offset,
                                        અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> d_offset, અચલ પૂર્ણાંક & iqs) {
        અચલ uint8_t *    base    = static_cast<અચલ uint8_t *>(vbq);
        અચલ uint8_t *    qs      = base + ibx_offset.first;   // low 4 bits
        અચલ uint8_t *    qh_base = base + ibx_offset.second;  // high bit
        અચલ uint8_t *    scs     = base + d_offset.first;
        અચલ ggml_half2 * dms     = reinterpret_cast<અચલ ggml_half2 *>(base + d_offset.second);

        અચલ પૂર્ણાંક        bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        અચલ પૂર્ણાંક *      ql_ptr     = (અચલ પૂર્ણાંક *) (qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        અચલ પૂર્ણાંક *      qh_ptr     = (અચલ પૂર્ણાંક *) (qh_base + 4 * ((iqs / 2) % 4));
        અચલ uint16_t * scales     = (અચલ uint16_t *) scs;

        weights w;
        w.vl[0] = ql_ptr[0];
        w.vl[1] = ql_ptr[4];

        w.vh[0] = qh_ptr[0] >> bq8_offset;
        w.vh[1] = qh_ptr[4] >> bq8_offset;

        અચલ પૂર્ણાંક j = (QR5_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        જો (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } નહીંતર {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        પાછા w;
    }

    __dpct_inline__ સ્થિર activations load_activations(અચલ int8_t * q8_1_quant_ptr,
                                                        અચલ sycl::half2 * q8_1_ds, અચલ પૂર્ણાંક & iqs) {
        પાછા reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::load_activations(q8_1_quant_ptr, q8_1_ds, iqs);
    }

    __dpct_inline__ સ્થિર દશાંશ apply(અચલ weights & w, અચલ activations & a) {
        અચલ uint8_t * sc = (અચલ uint8_t *) w.aux;
        અચલ uint8_t * m  = sc + 2;

        પાછા vec_dot_q5_K_q8_1_impl_vmmq(w.vl, w.vh, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ સ્થિર દશાંશ dot(અચલ weights & w, અચલ int8_t * q8_1_quant_ptr,
                                     અચલ sycl::half2 * q8_1_ds, અચલ પૂર્ણાંક & iqs) {
        પાછા apply(w, load_activations(q8_1_quant_ptr, q8_1_ds, iqs));
    }

    __dpct_inline__ દશાંશ operator()(અચલ શૂન્ય * __restrict__ vbq, અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> ibx_offset,
                                     અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> d_offset, અચલ int8_t * q8_1_quant_ptr,
                                     અચલ sycl::half2 * q8_1_ds, અચલ પૂર્ણાંક & iqs) {
        પાછા dot(load(vbq, ibx_offset, d_offset, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

template <> સંરચના reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K> {
    સ્થિર constexpr ggml_type gtype = GGML_TYPE_Q6_K;

    using q6_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q6_K>;
    using q6_k_traits = typename q6_k_block::traits;

    __dpct_inline__ દશાંશ vec_dot_q6_K_q8_1_impl_mmvq(અચલ પૂર્ણાંક vl, અચલ પૂર્ણાંક vh, અચલ પૂર્ણાંક * __restrict__ u,
                                                      અચલ int8_t * __restrict__ scales, અચલ દશાંશ d,
                                                      અચલ દશાંશ * __restrict__ d8) {
        પાછા vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
    }

    __dpct_inline__ દશાંશ operator()(અચલ શૂન્ય * __restrict__ vbq, અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> ibx_offset,
                     અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> d_offset, અચલ int8_t * q8_1_quant_ptr, અચલ sycl::half2 * q8_1_ds,
                     અચલ પૂર્ણાંક iqs) {
        અચલ uint8_t *   base   = static_cast<અચલ uint8_t *>(vbq);
        અચલ uint8_t *   ql     = base + ibx_offset.first;
        અચલ uint8_t *   qh     = base + ibx_offset.second;
        અચલ int8_t *    scales = reinterpret_cast<અચલ int8_t *>(base + d_offset.first);
        અચલ ggml_half * d      = (અચલ ggml_half *) (base + d_offset.second);

        અચલ પૂર્ણાંક bq8_offset   = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        અચલ પૂર્ણાંક scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        અચલ પૂર્ણાંક vh_shift     = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

        અચલ પૂર્ણાંક vl = get_int_from_uint8(ql, iqs);
        અચલ પૂર્ણાંક vh = get_int_from_uint8(qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;

        અચલ int8_t * scs = scales + scale_offset;

        અચલ પૂર્ણાંક u0 = get_int_from_int8_aligned(
            q8_1_quant_ptr + bq8_offset * QK8_1, iqs % QI8_1);
        અચલ પૂર્ણાંક u1 = get_int_from_int8_aligned(
            q8_1_quant_ptr + (bq8_offset + 2) * QK8_1, iqs % QI8_1);
        અચલ દશાંશ d80 = (*(q8_1_ds + bq8_offset + 0))[0];
        અચલ દશાંશ d81 = (*(q8_1_ds + bq8_offset + 2))[0];

        પાછા vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u0, u1, scs[0], scs[4], *d, d80, d81);
    }
};

template <> સંરચના reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4> {
    સ્થિર constexpr ggml_type gtype = GGML_TYPE_MXFP4;

    using mxfp4_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_MXFP4>;
    using mxfp4_traits = typename mxfp4_block::traits;

    __dpct_inline__ દશાંશ operator()(અચલ શૂન્ય * __restrict__ vbq, અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> ibx_offset,
                                     અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> d_offset, અચલ int8_t * q8_1_quant_ptr,
                                     અચલ sycl::half2 * q8_1_ds, અચલ પૂર્ણાંક & iqs) {
        static_assert(mxfp4_traits::vdr_mmvq == 2, "vector load assumes vdr_mmvq == 2");
        અચલ uint8_t * base = static_cast<અચલ uint8_t *>(vbq);

        // Reordered nibble blocks are 16 contiguous bytes and iqs is 0 or 2, so each lane's two
        // weight ints are one aligned 8-byte load (the AoS layout needed eight byte loads).
        અચલ sycl::int2 q4 = *reinterpret_cast<અચલ sycl::int2 *>(base + ibx_offset.first + કદ(પૂર્ણાંક) * iqs);
        અચલ uint8_t    e  = base[d_offset.first];

        // Low nibbles pair with q8_1 ints iqs..iqs+1, high nibbles with iqs+4..iqs+5.
        અચલ sycl::int2 u_lo = *reinterpret_cast<અચલ sycl::int2 *>(q8_1_quant_ptr + કદ(પૂર્ણાંક) * iqs);
        અચલ sycl::int2 u_hi = *reinterpret_cast<અચલ sycl::int2 *>(q8_1_quant_ptr + કદ(પૂર્ણાંક) * (iqs + 4));

        અચલ sycl::int2 v0 = get_int_from_mxfp4(q4.x());
        અચલ sycl::int2 v1 = get_int_from_mxfp4(q4.y());

        પૂર્ણાંક sumi = 0;
        sumi = ggml_sycl_dp4a(v0.x(), u_lo.x(), sumi);
        sumi = ggml_sycl_dp4a(v0.y(), u_hi.x(), sumi);
        sumi = ggml_sycl_dp4a(v1.x(), u_lo.y(), sumi);
        sumi = ggml_sycl_dp4a(v1.y(), u_hi.y(), sumi);

        અચલ દશાંશ d = ggml_sycl_e8m0_to_fp32(e) * 0.5f * static_cast<દશાંશ>((*q8_1_ds)[0]);
        પાછા d * sumi;
    }
};
#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

#define VDR_Q2_0_Q8_1_MMVQ 1

template <પૂર્ણાંક vdr>
સ્થિર __dpct_inline__ દશાંશ vec_dot_q2_0_q8_1_impl(
    અચલ પૂર્ણાંક * v,
    અચલ પૂર્ણાંક * u,
    અચલ દશાંશ & d2,
    અચલ sycl::half2 & ds8) {
    પૂર્ણાંક sumi = 0;

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < vdr; ++i) {
#pragma unroll
        માટે (પૂર્ણાંક j = 0; j < 4; ++j) {
            અચલ uint8_t q = (uint8_t) ((uint32_t) v[i] >> (8 * j));

            // unpack 2-bit values to byte lanes (0..3), then apply zero-point
            // correction with ds8f.y() below, mirroring the q4_0 style.
            પૂર્ણાંક vi = 0;
            vi |= (((q >> 0) & 0x3) & 0xFF) << 0;
            vi |= (((q >> 2) & 0x3) & 0xFF) << 8;
            vi |= (((q >> 4) & 0x3) & 0xFF) << 16;
            vi |= (((q >> 6) & 0x3) & 0xFF) << 24;

            sumi = dpct::dp4a(vi, u[4 * i + j], sumi);
        }
    }

    અચલ sycl::float2 ds8f = ds8.convert<દશાંશ, sycl::rounding_mode::automatic>();
    // q2_0 has zero-point 1. Scale ds8f.y() by processed-lane ratio,
    // consistent with q4_0's explicit zero-point subtraction style.
    પાછા d2 * (sumi * ds8f.x() - ((દશાંશ) vdr / (દશાંશ) QI2_0) * ds8f.y());
}

template <પૂર્ણાંક vdr>
સ્થિર __dpct_inline__ દશાંશ vec_dot_q4_0_q8_1_impl(અચલ પૂર્ણાંક * v, અચલ પૂર્ણાંક * u, અચલ દશાંશ & d4,
                                                    અચલ sycl::half2 & ds8) {
    પૂર્ણાંક sumi = 0;
#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < vdr; ++i) {
        અચલ પૂર્ણાંક vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        અચલ પૂર્ણાંક vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

    અચલ sycl::float2 ds8f = ds8.convert<દશાંશ, sycl::rounding_mode::automatic>();

    // second part effectively subtracts 8 from each quant value
    પાછા d4 * (sumi * ds8f.x() - (8 * vdr / QI4_0) * ds8f.y());
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

template <પૂર્ણાંક vdr>
સ્થિર __dpct_inline__ દશાંશ vec_dot_q4_1_q8_1_impl(અચલ પૂર્ણાંક *v, અચલ પૂર્ણાંક *u,
                                                    અચલ sycl::half2 &dm4,
                                                    અચલ sycl::half2 &ds8) {

    પૂર્ણાંક sumi = 0;

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < vdr; ++i) {
        અચલ પૂર્ણાંક vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        અચલ પૂર્ણાંક vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

#ifdef GGML_SYCL_F16
    અચલ sycl::float2 tmp =
        (dm4 * ds8).convert<દશાંશ, sycl::rounding_mode::automatic>();
    અચલ દશાંશ d4d8 = tmp.x();
    અચલ દશાંશ m4s8 = tmp.y();
#નહીંતર
    અચલ sycl::float2 dm4f =
        dm4.convert<દશાંશ, sycl::rounding_mode::automatic>();
    અચલ sycl::float2 ds8f =
        ds8.convert<દશાંશ, sycl::rounding_mode::automatic>();
    અચલ દશાંશ d4d8 = dm4f.x() * ds8f.x();
    અચલ દશાંશ m4s8 = dm4f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    પાછા sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <પૂર્ણાંક vdr>
સ્થિર __dpct_inline__ દશાંશ
vec_dot_q5_0_q8_1_impl(અચલ પૂર્ણાંક *vl, અચલ પૂર્ણાંક *vh, અચલ પૂર્ણાંક *u,
                       અચલ દશાંશ &d5, અચલ sycl::half2 &ds8) {
    પૂર્ણાંક sumi = 0;

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < vdr; ++i) {
        પૂર્ણાંક vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        પૂર્ણાંક vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

    અચલ sycl::float2 ds8f =
        ds8.convert<દશાંશ, sycl::rounding_mode::automatic>();

    // second part effectively subtracts 16 from each quant value
    પાછા d5 * (sumi * ds8f.x() - (16 * vdr / QI5_0) * ds8f.y());
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <પૂર્ણાંક vdr>
સ્થિર __dpct_inline__ દશાંશ
vec_dot_q5_1_q8_1_impl(અચલ પૂર્ણાંક *vl, અચલ પૂર્ણાંક *vh, અચલ પૂર્ણાંક *u,
                       અચલ sycl::half2 &dm5, અચલ sycl::half2 &ds8) {

    પૂર્ણાંક sumi = 0;

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < vdr; ++i) {
        પૂર્ણાંક vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        પૂર્ણાંક vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

#ifdef GGML_SYCL_F16
     અચલ sycl::float2 tmp =
        (dm5 * ds8).convert<દશાંશ, sycl::rounding_mode::automatic>();
    અચલ દશાંશ d5d8 = tmp.x();
    અચલ દશાંશ m5s8 = tmp.y();


#નહીંતર
    અચલ sycl::float2 dm5f =
        dm5.convert<દશાંશ, sycl::rounding_mode::automatic>();
    અચલ sycl::float2 ds8f =
        ds8.convert<દશાંશ, sycl::rounding_mode::automatic>();
    અચલ દશાંશ d5d8 = dm5f.x() * ds8f.x();
    અચલ દશાંશ m5s8 = dm5f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    પાછા sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <પૂર્ણાંક vdr>
સ્થિર __dpct_inline__ દશાંશ vec_dot_q8_0_q8_1_impl(અચલ પૂર્ણાંક *v, અચલ પૂર્ણાંક *u,
                                                    અચલ દશાંશ &d8_0,
                                                    અચલ દશાંશ &d8_1) {

    પૂર્ણાંક sumi = 0;

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

    પાછા d8_0*d8_1 * sumi;
}

template <typename T, પૂર્ણાંક vdr>
સ્થિર __dpct_inline__ T vec_dot_q8_0_q8_1_impl(અચલ પૂર્ણાંક * v, અચલ પૂર્ણાંક * u, અચલ T & d8_0, અચલ T & d8_1) {
    પૂર્ણાંક sumi = 0;

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_sycl_dp4a(v[i], u[i], sumi);
    }

    પાછા d8_0*d8_1 * ((T) sumi);
}

template <પૂર્ણાંક vdr>
સ્થિર __dpct_inline__ દશાંશ vec_dot_q8_1_q8_1_impl(અચલ પૂર્ણાંક *v, અચલ પૂર્ણાંક *u,
                                                    અચલ sycl::half2 &dm8,
                                                    અચલ sycl::half2 &ds8) {

    પૂર્ણાંક sumi = 0;

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

#ifdef GGML_SYCL_F16
    અચલ sycl::float2 tmp =
        (dm8 * ds8).convert<દશાંશ, sycl::rounding_mode::automatic>();
    અચલ દશાંશ d8d8 = tmp.x();
    અચલ દશાંશ m8s8 = tmp.y();
#નહીંતર
    અચલ sycl::float2 dm8f =
        dm8.convert<દશાંશ, sycl::rounding_mode::automatic>();
    અચલ sycl::float2 ds8f =
        ds8.convert<દશાંશ, sycl::rounding_mode::automatic>();
    અચલ દશાંશ d8d8 = dm8f.x() * ds8f.x();
    અચલ દશાંશ m8s8 = dm8f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    પાછા sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

સ્થિર __dpct_inline__ દશાંશ
vec_dot_q4_0_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                  અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {

    અચલ block_q4_0 * bq4_0 = (અચલ block_q4_0 *) vbq;

    પૂર્ણાંક v[VDR_Q4_0_Q8_1_MMVQ];
    પૂર્ણાંક u[2 * VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]         = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    પાછા vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

સ્થિર __dpct_inline__ દશાંશ
vec_dot_q2_0_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                  અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {

    અચલ block_q2_0 * bq2_0 = (અચલ block_q2_0 *) vbq;

    પૂર્ણાંક v[2 * VDR_Q2_0_Q8_1_MMVQ];
    પૂર્ણાંક u[8 * VDR_Q2_0_Q8_1_MMVQ];

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < VDR_Q2_0_Q8_1_MMVQ; ++i) {
        અચલ પૂર્ણાંક base = 4 * (iqs + i);

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

    અચલ દશાંશ sum0 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + 0, u + 0, bq2_0->d, bq8_1[0].ds);
    અચલ દશાંશ sum1 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + VDR_Q2_0_Q8_1_MMVQ, u + 4 * VDR_Q2_0_Q8_1_MMVQ, bq2_0->d, bq8_1[1].ds);
    પાછા sum0 + sum1;
}

સ્થિર __dpct_inline__ દશાંશ
vec_dot_q4_1_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                  અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {

    અચલ block_q4_1 * bq4_1 = (અચલ block_q4_1 *) vbq;

    પૂર્ણાંક v[VDR_Q4_1_Q8_1_MMVQ];
    પૂર્ણાંક u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    પાછા vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

#define VDR_MXFP4_Q8_1_MMVQ 2
#define VDR_MXFP4_Q8_1_MMQ  4

સ્થિર __dpct_inline__ દશાંશ vec_dot_mxfp4_q8_1(અચલ શૂન્ય * __restrict__ vbq,
                                                અચલ block_q8_1 * __restrict__ bq8_1,
                                                અચલ પૂર્ણાંક & iqs) {
    અચલ block_mxfp4 * bq4 = (અચલ block_mxfp4 *) vbq;

    અચલ પૂર્ણાંક * q8 = (અચલ પૂર્ણાંક *) bq8_1->qs + iqs;

    પૂર્ણાંક sumi = 0;
#pragma unroll
    માટે (પૂર્ણાંક l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        અચલ પૂર્ણાંક aux_q4 = get_int_b1(bq4->qs, iqs + l);
        અચલ sycl::int2 v      = get_int_from_mxfp4(aux_q4);
        sumi = ggml_sycl_dp4a(v.x(), q8[l + 0], sumi);
        sumi = ggml_sycl_dp4a(v.y(), q8[l + 4], sumi);
    }

    અચલ દશાંશ d = ggml_sycl_e8m0_to_fp32(bq4->e) * 0.5f * (bq8_1->ds)[0];
    પાછા d * sumi;
}

#define VDR_NVFP4_Q8_1_MMVQ 4
#define VDR_NVFP4_Q8_1_MMQ  8

સ્થિર __dpct_inline__ દશાંશ vec_dot_nvfp4_q8_1(અચલ શૂન્ય * __restrict__ vbq,
                                                અચલ block_q8_1 * __restrict__ bq8_1,
                                                અચલ int32_t & iqs) {
    અચલ block_nvfp4 * bq4 = (અચલ block_nvfp4 *) vbq;
    દશાંશ sum = 0.0f;
#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; i++) {
        અચલ int32_t iqs0 = iqs + 2*i;
        અચલ int32_t iqs1 = iqs0 + 1;
        અચલ int32_t is = iqs0 >> 1;
        અચલ sycl::int2   v0   = get_int_from_table_16(get_int_b4(bq4->qs, iqs0), kvalues_mxfp4);
        અચલ sycl::int2   v1   = get_int_from_table_16(get_int_b4(bq4->qs, iqs1), kvalues_mxfp4);
        અચલ block_q8_1 * bq8 = bq8_1 + (is >> 1);
        અચલ int32_t i8 = ((is & 1) << 2);

        પૂર્ણાંક sumi = ggml_sycl_dp4a(v0.x(), get_int_b4(bq8->qs, i8 + 0), 0);
        sumi     = ggml_sycl_dp4a(v0.y(), get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi     = ggml_sycl_dp4a(v1.x(), get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi     = ggml_sycl_dp4a(v1.y(), get_int_b4(bq8->qs, i8 + 3), sumi);

        અચલ દશાંશ d = ggml_sycl_ue4m3_to_fp32(bq4->d[is]) * (bq8->ds)[0];
        sum += d * દશાંશ(sumi);
    }

    પાછા sum;
}

સ્થિર __dpct_inline__ દશાંશ
vec_dot_q5_0_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                  અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {

    અચલ block_q5_0 * bq5_0 = (અચલ block_q5_0 *) vbq;

    પૂર્ણાંક vl[VDR_Q5_0_Q8_1_MMVQ];
    પૂર્ણાંક vh[VDR_Q5_0_Q8_1_MMVQ];
    પૂર્ણાંક  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    પાછા vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

સ્થિર __dpct_inline__ દશાંશ
vec_dot_q5_1_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                  અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {

    અચલ block_q5_1 * bq5_1 = (અચલ block_q5_1 *) vbq;

    પૂર્ણાંક vl[VDR_Q5_1_Q8_1_MMVQ];
    પૂર્ણાંક vh[VDR_Q5_1_Q8_1_MMVQ];
    પૂર્ણાંક  u[2*VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    પાછા vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

સ્થિર __dpct_inline__ દશાંશ
vec_dot_q8_0_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                  અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {

    અચલ block_q8_0 * bq8_0 = (અચલ block_q8_0 *) vbq;

    પૂર્ણાંક v[VDR_Q8_0_Q8_1_MMVQ];
    પૂર્ણાંક u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    પાછા vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                      bq8_1->ds[0]);
}

સ્થિર __dpct_inline__ દશાંશ
vec_dot_q2_K_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                  અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {

    અચલ block_q2_K * bq2_K = (અચલ block_q2_K *) vbq;

    અચલ પૂર્ણાંક bq8_offset = QR2_K * (iqs / QI8_1);
    અચલ પૂર્ણાંક scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    અચલ uint8_t * scales = bq2_K->scales + scale_offset;

    અચલ પૂર્ણાંક v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    પૂર્ણાંક    u[QR2_K];
    દશાંશ d8[QR2_K];

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + i].ds[0];
    }

    પાછા vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

સ્થિર __dpct_inline__ દશાંશ
vec_dot_q3_K_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                  અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {

    અચલ block_q3_K * bq3_K = (અચલ block_q3_K *) vbq;

    અચલ પૂર્ણાંક bq8_offset = QR3_K * (iqs / (QI3_K/2));
    અચલ પૂર્ણાંક scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    અચલ દશાંશ d = bq3_K->d;

    અચલ પૂર્ણાંક vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    અચલ પૂર્ણાંક vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    પૂર્ણાંક    u[QR3_K];
    દશાંશ d8[QR3_K];

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + i].ds[0];
    }

    પાછા vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

સ્થિર __dpct_inline__ દશાંશ vec_dot_q4_K_q8_1(અચલ શૂન્ય * __restrict__ vbq, અચલ block_q8_1 * __restrict__ bq8_1,
                                               અચલ પૂર્ણાંક & iqs) {
#ifndef GGML_QKK_64

    અચલ block_q4_K * bq4_K = (અચલ block_q4_K *) vbq;

    અચલ પૂર્ણાંક        bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    અચલ પૂર્ણાંક *      q4         = (અચલ પૂર્ણાંક *) (bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    અચલ uint16_t * scales     = (અચલ uint16_t *) bq4_K->scales;

    પાછા vec_dot_q4_K_q8_1_common(q4, scales, bq4_K->dm, bq8_1, iqs);

#નહીંતર

#જો __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    અચલ block_q4_K * bq4_K = (અચલ block_q4_K *) vbq;

    દશાંશ sumf_d = 0.0f;
    દશાંશ sumf_m = 0.0f;

    uint16_t aux16[2];
    અચલ uint8_t * s = (અચલ uint8_t *)aux16;

    અચલ uint16_t * a = (અચલ uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    અચલ દશાંશ dall = bq4_K->dm[0];
    અચલ દશાંશ dmin = bq4_K->dm[1];

    અચલ દશાંશ d8_1 = bq8_1[0].ds[0];
    અચલ દશાંશ d8_2 = bq8_1[1].ds[1];

    અચલ પૂર્ણાંક ui1 = *((અચલ પૂર્ણાંક *)bq8_1[0].qs + (iqs/2));
    અચલ પૂર્ણાંક ui2 = *((અચલ પૂર્ણાંક *)bq8_1[0].qs + (iqs/2) + 4);
    અચલ પૂર્ણાંક ui3 = *((અચલ પૂર્ણાંક *)bq8_1[1].qs + (iqs/2));
    અચલ પૂર્ણાંક ui4 = *((અચલ પૂર્ણાંક *)bq8_1[1].qs + (iqs/2) + 4);

    અચલ પૂર્ણાંક * q4 = (અચલ પૂર્ણાંક *)bq4_K->qs + (iqs/2);
    અચલ પૂર્ણાંક v1 = q4[0];
    અચલ પૂર્ણાંક v2 = q4[4];

    અચલ પૂર્ણાંક dot1 = dpct::dp4a(ui2, v2 & 0x0f0f0f0f, dpct::dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    અચલ પૂર્ણાંક dot2 = dpct::dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, dpct::dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    અચલ પૂર્ણાંક dot3 = dpct::dp4a(0x01010101, ui2, dpct::dp4a(0x01010101, ui1, 0));
    અચલ પૂર્ણાંક dot4 = dpct::dp4a(0x01010101, ui4, dpct::dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    પાછા dall * sumf_d - dmin * sumf_m;

#નહીંતર
    bad_arch();
#endif // __SYCL_ARCH__ >= VER_4VEC

#endif
}

સ્થિર __dpct_inline__ દશાંશ
vec_dot_q5_K_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                  અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {

#ifndef GGML_QKK_64
    અચલ block_q5_K * bq5_K = (અચલ block_q5_K *) vbq;

    પૂર્ણાંક   vl[2];
    પૂર્ણાંક   vh[2];
    પૂર્ણાંક    u[2*QR5_K];
    દશાંશ d8[QR5_K];

    અચલ પૂર્ણાંક bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    અચલ પૂર્ણાંક * ql = (અચલ પૂર્ણાંક *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    અચલ પૂર્ણાંક * qh = (અચલ પૂર્ણાંક *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    અચલ uint16_t * scales = (અચલ uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    અચલ પૂર્ણાંક j = bq8_offset/2;
    જો (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } નહીંતર {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    અચલ uint8_t * sc = (અચલ uint8_t *)aux;
    અચલ uint8_t * m  = sc + 2;

#pragma unroll
    માટે (પૂર્ણાંક i = 0; i < QR5_K; ++i) {
        અચલ block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = bq8i->ds[0];

        અચલ પૂર્ણાંક * q8 = (અચલ પૂર્ણાંક *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    પાછા vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#નહીંતર

#જો __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    અચલ block_q5_K * bq5_K = (અચલ block_q5_K *) vbq;

    અચલ int8_t * s = bq5_K->scales;

    અચલ દશાંશ d = bq5_K->d;

    અચલ દશાંશ d8_1 = bq8_1[0].ds[0];
    અચલ દશાંશ d8_2 = bq8_1[1].ds[1];

    અચલ પૂર્ણાંક ui1 = *((અચલ પૂર્ણાંક *)bq8_1[0].qs + (iqs/2));
    અચલ પૂર્ણાંક ui2 = *((અચલ પૂર્ણાંક *)bq8_1[0].qs + (iqs/2) + 4);
    અચલ પૂર્ણાંક ui3 = *((અચલ પૂર્ણાંક *)bq8_1[1].qs + (iqs/2));
    અચલ પૂર્ણાંક ui4 = *((અચલ પૂર્ણાંક *)bq8_1[1].qs + (iqs/2) + 4);

    અચલ પૂર્ણાંક * ql = (અચલ પૂર્ણાંક *)bq5_K->qs + (iqs/2);
    અચલ પૂર્ણાંક vl1 = ql[0];
    અચલ પૂર્ણાંક vl2 = ql[4];

    અચલ પૂર્ણાંક step = 4 * (iqs/2); // 0, 4, 8, 12
    અચલ પૂર્ણાંક im = step/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    અચલ પૂર્ણાંક in = step%8; // 0, 4, 0, 4
    અચલ પૂર્ણાંક vh = (*((અચલ પૂર્ણાંક *)(bq5_K->qh + in))) >> im;

    અચલ પૂર્ણાંક v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    અચલ પૂર્ણાંક v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    અચલ પૂર્ણાંક v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    અચલ પૂર્ણાંક v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    અચલ દશાંશ sumf_d = d8_1 * (dpct::dp4a(ui1, v1, 0) * s[0] + dpct::dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (dpct::dp4a(ui3, v3, 0) * s[2] + dpct::dp4a(ui4, v4, 0) * s[3]);

    પાછા d * sumf_d;

#નહીંતર
    bad_arch();
#endif // __SYCL_ARCH__ >= VER_4VEC

#endif
}

સ્થિર __dpct_inline__ દશાંશ
vec_dot_q6_K_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                  અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {

    અચલ block_q6_K * bq6_K = (અચલ block_q6_K *) vbq;

    અચલ પૂર્ણાંક bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    અચલ પૂર્ણાંક scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    અચલ પૂર્ણાંક vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    અચલ પૂર્ણાંક vl = get_int_from_uint8(bq6_K->ql, iqs);
    અચલ પૂર્ણાંક vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    અચલ int8_t * scales = bq6_K->scales + scale_offset;

    અચલ પૂર્ણાંક u0 = get_int_from_int8_aligned(
        bq8_1[bq8_offset + 0].qs, iqs % QI8_1);
    અચલ પૂર્ણાંક u1 = get_int_from_int8_aligned(
        bq8_1[bq8_offset + 2].qs, iqs % QI8_1);
    અચલ દશાંશ d80 = bq8_1[bq8_offset + 0].ds[0];
    અચલ દશાંશ d81 = bq8_1[bq8_offset + 2].ds[0];

    પાછા vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u0, u1, scales[0], scales[4], bq6_K->d, d80, d81);
}


// NOTE: the VDR_IQ*_Q8_1_MMVQ values deliberately differ from the identically named CUDA constants
// (vecdotq.cuh): the SYCL kernels pair them with a halved qi (e.g. QI3_S/2), so the values are not
// interchangeable and must not be copied across backends.
#define VDR_IQ2_XXS_Q8_1_MMVQ 1

સ્થિર __dpct_inline__ uint32_t iq_sign_mask(અચલ uint32_t signs) {
    અચલ uint32_t bits =
        (signs & 0x01) | ((signs & 0x02) << 7) |
        ((signs & 0x04) << 14) | ((signs & 0x08) << 21);

    પાછા bits * 0xff;
}

સ્થિર __dpct_inline__ uint32_t iq_sign_bits(અચલ uint32_t encoded) {
    અચલ uint32_t signs = encoded & 127;

    પાછા signs | ((sycl::popcount(signs) & 1) << 7);
}

સ્થિર __dpct_inline__ દશાંશ
vec_dot_iq2_xxs_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                     અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs,
                     અચલ uint64_t *iq2xxs_grid, અચલ uint8_t *ksigns_iq2xs,
                     અચલ uint8_t *kmask_iq2xs) {
#જો QK_K == 256
    અચલ block_iq2_xxs * bq2 = (અચલ block_iq2_xxs *) vbq;

    અચલ પૂર્ણાંક ib32 = iqs;
    અચલ uint16_t * q2 = bq2->qs + 4*ib32;
    અચલ uint8_t  * aux8 = (અચલ uint8_t *)q2;
    અચલ int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = q2[2] | (q2[3] << 16);
    પૂર્ણાંક sumi = 0;
    માટે (પૂર્ણાંક l = 0; l < 4; ++l) {
        અચલ uint8_t * grid = (અચલ uint8_t *)(iq2xxs_grid + aux8[l]);
        અચલ uint8_t  signs = ksigns_iq2xs[aux32 & 127];
        માટે (પૂર્ણાંક j = 0; j < 8; ++j) {
            sumi += q8[j] * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
        aux32 >>= 7;
    }
    અચલ દશાંશ d = (દશાંશ)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.25f;
    પાછા d * sumi;
#નહીંતર
    પ્રતિજ્ઞા(ખોટું);
    પાછા 0.f;
#endif
}

#define VDR_IQ2_XS_Q8_1_MMVQ 1

સ્થિર __dpct_inline__ દશાંશ
vec_dot_iq2_xs_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                    અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs,
                    અચલ uint64_t *iq2xs_grid) {
#જો DPCT_COMPATIBILITY_TEMP >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#જો QK_K == 256
    અચલ block_iq2_xs * bq2 = (અચલ block_iq2_xs *) vbq;

    અચલ પૂર્ણાંક ib32 = iqs;
    અચલ uint16_t * q2 = bq2->qs + 4*ib32;
    અચલ int8_t   * q8 = bq8_1[ib32].qs;
    અચલ uint8_t ls1 = bq2->scales[ib32] & 0xf;
    અચલ uint8_t ls2 = bq2->scales[ib32] >>  4;
    પૂર્ણાંક sumi1 = 0;
#pragma unroll
    માટે (પૂર્ણાંક l = 0; l < 2; ++l) {
        અચલ uint64_t grid = iq2xs_grid[q2[l] & 511];
        અચલ uint32_t sign_bits = iq_sign_bits(q2[l] >> 9);
        અચલ uint32_t signs0 = iq_sign_mask(sign_bits);
        અચલ uint32_t signs1 = iq_sign_mask(sign_bits >> 4);
        અચલ પૂર્ણાંક grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        અચલ પૂર્ણાંક grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi1 = dpct::dp4a(grid_l, *((અચલ પૂર્ણાંક *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((અચલ પૂર્ણાંક *)q8 + 1), sumi1);
        q8 += 8;
    }
    પૂર્ણાંક sumi2 = 0;
#pragma unroll
    માટે (પૂર્ણાંક l = 2; l < 4; ++l) {
        અચલ uint64_t grid = iq2xs_grid[q2[l] & 511];
        અચલ uint32_t sign_bits = iq_sign_bits(q2[l] >> 9);
        અચલ uint32_t signs0 = iq_sign_mask(sign_bits);
        અચલ uint32_t signs1 = iq_sign_mask(sign_bits >> 4);
        અચલ પૂર્ણાંક grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        અચલ પૂર્ણાંક grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi2 = dpct::dp4a(grid_l, *((અચલ પૂર્ણાંક *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((અચલ પૂર્ણાંક *)q8 + 1), sumi2);
        q8 += 8;
    }
    અચલ દશાંશ d = (દશાંશ)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    પાછા d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#નહીંતર
    પ્રતિજ્ઞા(ખોટું);
    પાછા 0.f;
#endif
#નહીંતર
    પ્રતિજ્ઞા(ખોટું);
    પાછા 0.f;
#endif
}

#define VDR_IQ2_S_Q8_1_MMVQ 1

સ્થિર __dpct_inline__ દશાંશ
vec_dot_iq2_s_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                   અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {
#જો QK_K == 256
    અચલ block_iq2_s * bq2 = (અચલ block_iq2_s *) vbq;

    અચલ પૂર્ણાંક ib32 = iqs;
    અચલ int8_t  * q8 = bq8_1[ib32].qs;
    અચલ uint8_t * signs = bq2->qs + QK_K/8 + 4*ib32;
    અચલ uint8_t ls1 = bq2->scales[ib32] & 0xf;
    અચલ uint8_t ls2 = bq2->scales[ib32] >>  4;
    પૂર્ણાંક sumi1 = 0;
    માટે (પૂર્ણાંક l = 0; l < 2; ++l) {
        અચલ uint32_t * grid = (અચલ uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        અચલ uint32_t signs0 = dpct::vectorized_દ્વિઅંકી<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        અચલ uint32_t signs1 = dpct::vectorized_દ્વિઅંકી<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        અચલ પૂર્ણાંક grid_l = dpct::vectorized_દ્વિઅંકી<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        અચલ પૂર્ણાંક grid_h = dpct::vectorized_દ્વિઅંકી<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi1 = dpct::dp4a(grid_l, *((અચલ પૂર્ણાંક *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((અચલ પૂર્ણાંક *)q8 + 1), sumi1);
        q8 += 8;
    }
    પૂર્ણાંક sumi2 = 0;
    માટે (પૂર્ણાંક l = 2; l < 4; ++l) {
        અચલ uint32_t * grid = (અચલ uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        અચલ uint32_t signs0 = dpct::vectorized_દ્વિઅંકી<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        અચલ uint32_t signs1 = dpct::vectorized_દ્વિઅંકી<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        અચલ પૂર્ણાંક grid_l = dpct::vectorized_દ્વિઅંકી<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        અચલ પૂર્ણાંક grid_h = dpct::vectorized_દ્વિઅંકી<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi2 = dpct::dp4a(grid_l, *((અચલ પૂર્ણાંક *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((અચલ પૂર્ણાંક *)q8 + 1), sumi2);
        q8 += 8;
    }
    અચલ દશાંશ d = (દશાંશ)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    પાછા d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#નહીંતર
    પ્રતિજ્ઞા(ખોટું);
#endif
}

#define VDR_IQ3_XXS_Q8_1_MMVQ 1

સ્થિર __dpct_inline__ દશાંશ
vec_dot_iq3_xxs_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                     અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs,
                     અચલ uint32_t *iq3xxs_grid) {
#જો DPCT_COMPATIBILITY_TEMP >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#જો QK_K == 256
    અચલ block_iq3_xxs * bq2 = (અચલ block_iq3_xxs *) vbq;

    અચલ પૂર્ણાંક ib32 = iqs;
    અચલ uint8_t  * q3 = bq2->qs + 8*ib32;
    અચલ uint16_t * gas = (અચલ uint16_t *)(bq2->qs + QK_K/4) + 2*ib32;
    અચલ int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = gas[0] | (gas[1] << 16);
    પૂર્ણાંક sumi = 0;
#pragma unroll
    માટે (પૂર્ણાંક l = 0; l < 4; ++l) {
        અચલ uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
        અચલ uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
        અચલ uint32_t sign_bits = iq_sign_bits(aux32);
        અચલ uint32_t signs0 = iq_sign_mask(sign_bits);
        અચલ uint32_t signs1 = iq_sign_mask(sign_bits >> 4);
        અચલ પૂર્ણાંક grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        અચલ પૂર્ણાંક grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((અચલ પૂર્ણાંક *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((અચલ પૂર્ણાંક *)q8 + 1), sumi);
        q8 += 8;
        aux32 >>= 7;
    }
    અચલ દશાંશ d = (દશાંશ)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.5f;
    પાછા d * sumi;
#નહીંતર
    પ્રતિજ્ઞા(ખોટું);
    પાછા 0.f;
#endif
#નહીંતર
    પ્રતિજ્ઞા(ખોટું);
    પાછા 0.f;
#endif
}

template <> સંરચના reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_XXS> {
    સ્થિર constexpr ggml_type gtype = GGML_TYPE_IQ3_XXS;

    __dpct_inline__ દશાંશ operator()(અચલ શૂન્ય * __restrict__ vbq, અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> ibx_offset,
                                     અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> d_offset, અચલ int8_t * q8_1_quant_ptr,
                                     અચલ sycl::half2 * q8_1_ds, અચલ પૂર્ણાંક & iqs) {
#જો DPCT_COMPATIBILITY_TEMP >= MIN_CC_DP4A && QK_K == 256
        અચલ uint8_t * base = static_cast<અચલ uint8_t *>(vbq);
        અચલ uint8_t * qs   = base + ibx_offset.first;
        અચલ uint8_t * q3   = qs + 8 * iqs;
        અચલ uint16_t * gas = reinterpret_cast<અચલ uint16_t *>(qs + QK_K / 4) + 2 * iqs;
        અચલ int8_t * q8    = q8_1_quant_ptr + iqs * QK8_1;
        uint32_t aux32       = gas[0] | (gas[1] << 16);

        પૂર્ણાંક sumi = 0;
#pragma unroll
        માટે (પૂર્ણાંક l = 0; l < 4; ++l) {
            અચલ uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
            અચલ uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
            અચલ uint32_t sign_bits = iq_sign_bits(aux32);
            અચલ uint32_t signs0 = iq_sign_mask(sign_bits);
            અચલ uint32_t signs1 = iq_sign_mask(sign_bits >> 4);
            અચલ પૂર્ણાંક grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            અચલ પૂર્ણાંક grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((અચલ પૂર્ણાંક *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((અચલ પૂર્ણાંક *) q8 + 1), sumi);
            q8 += 8;
            aux32 >>= 7;
        }

        અચલ ggml_half d = *reinterpret_cast<અચલ ggml_half *>(base + d_offset.first);
        પાછા (દશાંશ) d * (0.5f + aux32) * q8_1_ds[iqs][0] * 0.5f * sumi;
#નહીંતર
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

સ્થિર __dpct_inline__ દશાંશ
vec_dot_iq3_s_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                   અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs,
                   અચલ uint32_t *iq3s_grid) {
#જો QK_K == 256
    અચલ block_iq3_s * bq2 = (અચલ block_iq3_s *) vbq;

    અચલ પૂર્ણાંક ib32 = iqs;
    અચલ uint8_t  * qs = bq2->qs + 8*ib32;
    અચલ int8_t   * q8 = bq8_1[ib32].qs;
    પૂર્ણાંક sumi = 0;
    માટે (પૂર્ણાંક l = 0; l < 4; ++l) {
        અચલ uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((bq2->qh[ib32] << (8 - 2*l)) & 256)];
        અચલ uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((bq2->qh[ib32] << (7 - 2*l)) & 256)];
        અચલ uint32_t signs = bq2->signs[4 * ib32 + l];
        અચલ uint32_t signs0 = iq_sign_mask(signs);
        અચલ uint32_t signs1 = iq_sign_mask(signs >> 4);
        અચલ પૂર્ણાંક grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        અચલ પૂર્ણાંક grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((અચલ પૂર્ણાંક *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((અચલ પૂર્ણાંક *)q8 + 1), sumi);
        q8 += 8;
    }
    અચલ દશાંશ d =
        (દશાંશ)bq2->d *
        (1 + 2 * ((bq2->scales[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf)) *
        bq8_1[ib32].ds[0];
    પાછા d * sumi;
#નહીંતર
    પ્રતિજ્ઞા(ખોટું);
#endif
}

template <> સંરચના reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_S> {
    સ્થિર constexpr ggml_type gtype = GGML_TYPE_IQ3_S;

    __dpct_inline__ દશાંશ operator()(અચલ શૂન્ય * __restrict__ vbq, અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> ibx_offset,
                                     અચલ std::pair<પૂર્ણાંક, પૂર્ણાંક> d_offset, અચલ int8_t * q8_1_quant_ptr,
                                     અચલ sycl::half2 * q8_1_ds, અચલ પૂર્ણાંક & iqs) {
#જો QK_K == 256
        અચલ uint8_t * base     = static_cast<અચલ uint8_t *>(vbq);
        અચલ uint8_t * qs       = base + ibx_offset.first + 8 * iqs;
        અચલ uint8_t * qh       = base + ibx_offset.second;
        અચલ uint8_t * signs    = base + d_offset.first + 4 * iqs;
        અચલ uint8_t * metadata = base + d_offset.second;
        અચલ ggml_half d        = *reinterpret_cast<અચલ ggml_half *>(metadata);
        અચલ uint8_t * scales   = metadata + કદ(ggml_half);
        અચલ int8_t * q8        = q8_1_quant_ptr + iqs * QK8_1;

        પૂર્ણાંક sumi = 0;
#pragma unroll
        માટે (પૂર્ણાંક l = 0; l < 4; ++l) {
            અચલ uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((qh[iqs] << (8 - 2*l)) & 256)];
            અચલ uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((qh[iqs] << (7 - 2*l)) & 256)];
            અચલ uint32_t signs0 = iq_sign_mask(signs[l]);
            અચલ uint32_t signs1 = iq_sign_mask(signs[l] >> 4);
            અચલ પૂર્ણાંક grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            અચલ પૂર્ણાંક grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((અચલ પૂર્ણાંક *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((અચલ પૂર્ણાંક *) q8 + 1), sumi);
            q8 += 8;
        }

        અચલ દશાંશ scale = 1 + 2 * ((scales[iqs / 2] >> (4 * (iqs % 2))) & 0xf);
        પાછા (દશાંશ) d * scale * q8_1_ds[iqs][0] * sumi;
#નહીંતર
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

સ્થિર __dpct_inline__ દશાંશ
vec_dot_iq1_s_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                   અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs,
                   અચલ uint32_t *iq1s_grid_gpu) {
#જો QK_K == 256
    અચલ block_iq1_s * bq1 = (અચલ block_iq1_s *) vbq;

    અચલ પૂર્ણાંક ib32 = iqs;
    પૂર્ણાંક sumi = 0;
    અચલ પૂર્ણાંક * q8 = (અચલ પૂર્ણાંક *)bq8_1[ib32].qs;
    માટે (પૂર્ણાંક l = 0; l < 4; ++l) {
        અચલ પૂર્ણાંક * grid = (અચલ પૂર્ણાંક *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[ib32] >> 3*l) & 7) << 8)));
        પૂર્ણાંક grid0 = grid[0] & 0x0f0f0f0f;
        પૂર્ણાંક grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi = dpct::dp4a(q8[2 * l + 1], grid1,
                          dpct::dp4a(q8[2 * l + 0], grid0, sumi));
    }

    અચલ દશાંશ delta = bq1->qh[ib32] & 0x8000 ? -1-IQ1S_DELTA : -1+IQ1S_DELTA;
    અચલ દશાંશ d1q = (દશાંશ)bq1->d * (2*((bq1->qh[ib32] >> 12) & 7) + 1);
    અચલ દશાંશ d = d1q * bq8_1[ib32].ds[0];
    અચલ દશાંશ m = d1q * bq8_1[ib32].ds[1];
    પાછા d * sumi + m * delta;
#નહીંતર
    પ્રતિજ્ઞા(ખોટું);
#endif
}

#define VDR_IQ1_M_Q8_1_MMVQ 1

સ્થિર __dpct_inline__ દશાંશ
vec_dot_iq1_m_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                   અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {
#જો QK_K == 256
    અચલ block_iq1_m * bq1 = (અચલ block_iq1_m *) vbq;

    અચલ પૂર્ણાંક ib32 = iqs;
    પૂર્ણાંક   sumi[2] = {0, 0};
    દશાંશ sumf[2] = {0.f, 0.f};

    અચલ પૂર્ણાંક * q8 = (અચલ પૂર્ણાંક *)bq8_1[ib32].qs;
    માટે (પૂર્ણાંક l = 0; l < 4; ++l) {
        અચલ પૂર્ણાંક * grid = (અચલ પૂર્ણાંક *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 7) << 8)));
        પૂર્ણાંક grid0 = grid[0] & 0x0f0f0f0f;
        પૂર્ણાંક grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi[l / 2] = dpct::dp4a(q8[2 * l + 1], grid1,
                                 dpct::dp4a(q8[2 * l + 0], grid0, sumi[l / 2]));
        અચલ દશાંશ delta = (bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 0x08 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA;
        અચલ પૂર્ણાંક sumy = dpct::dp4a(q8[2 * l + 1], 0x01010101,
                                    dpct::dp4a(q8[2 * l + 0], 0x01010101, 0));
        sumf[l/2] += delta*sumy;
    }

    iq1m_scale_t scale;
    અચલ uint16_t * sc = (અચલ uint16_t *)bq1->scales;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    અચલ દશાંશ d = (દશાંશ)scale.f16 * bq8_1[ib32].ds[0];
    પાછા d * ((sumi[0] + sumf[0]) * (2*((sc[ib32/2] >> 6*(ib32%2)) & 0x7) + 1) + (sumi[1] + sumf[1]) * (2*((sc[ib32/2] >> (6*(ib32%2)+3)) & 0x7) + 1));
#નહીંતર
    પ્રતિજ્ઞા(ખોટું);
#endif
}


#define VDR_IQ4_NL_Q8_1_MMVQ 2

સ્થિર __dpct_inline__ દશાંશ
vec_dot_iq4_nl_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                    અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {

    અચલ block_iq4_nl * bq = (અચલ block_iq4_nl *) vbq;

    અચલ uint16_t * q4 = (અચલ uint16_t *)bq->qs + 2*iqs;
    અચલ int32_t  * q8 = (અચલ int32_t  *)bq8_1->qs + iqs;

    અચલ uint8_t * કિંમતો = (અચલ uint8_t *)kvalues_iq4nl;

    પૂર્ણાંક v1, v2;
    પૂર્ણાંક sumi1 = 0, sumi2 = 0;
    માટે (પૂર્ણાંક l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        અચલ uint32_t aux = q4[2*l] | (q4[2*l+1] << 16);
        get_int_from_table_16(aux, કિંમતો, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[l + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[l + 4], sumi2);
    }

    અચલ દશાંશ d = (દશાંશ)bq->d * bq8_1->ds[0];
    પાછા d * (sumi1 + sumi2);
}


#define VDR_IQ4_XS_Q8_1_MMVQ 1

સ્થિર __dpct_inline__ દશાંશ
vec_dot_iq4_xs_q8_1(અચલ શૂન્ય *__restrict__ vbq,
                    અચલ block_q8_1 *__restrict__ bq8_1, અચલ પૂર્ણાંક &iqs) {

#જો QK_K == 256
    અચલ block_iq4_xs * bq4 = (અચલ block_iq4_xs *) vbq;
    અચલ uint8_t * કિંમતો = (અચલ uint8_t *)kvalues_iq4nl;

    // iqs is 0...7
    અચલ પૂર્ણાંક ib32 = iqs;
    અચલ int32_t  * q8 = (અચલ પૂર્ણાંક *)bq8_1[ib32].qs;
    અચલ uint32_t * q4 = (અચલ uint32_t *)bq4->qs + 4*ib32;
    અચલ int8_t ls = ((bq4->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((bq4->scales_h >> 2*ib32) & 3) << 4);
    અચલ દશાંશ d = (દશાંશ)bq4->d * (ls - 32) * bq8_1[ib32].ds[0];
    પૂર્ણાંક v1, v2;
    પૂર્ણાંક sumi1 = 0, sumi2 = 0;
    માટે (પૂર્ણાંક j = 0; j < 4; ++j) {
        get_int_from_table_16(q4[j], કિંમતો, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[j + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[j + 4], sumi2);
    }
    પાછા d * (sumi1 + sumi2);
#નહીંતર
    પ્રતિજ્ઞા(ખોટું);
#endif
}

#endif // GGML_SYCL_VECDOTQ_HPP
