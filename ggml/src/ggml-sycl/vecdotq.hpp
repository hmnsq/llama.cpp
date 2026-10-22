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

#यदि_अपरिभाषितम् GGML_SYCL_VECDOTQ_HPP
#परिभाषय GGML_SYCL_VECDOTQ_HPP

#अन्तर्भाव "dpct/helper.hpp"
#अन्तर्भाव "ggml.h"
#अन्तर्भाव "type.hpp"
#अन्तर्भाव "quants.hpp"

प्रकारनिर्देश भिन्न (*vec_dot_q_sycl_t)(ध्रुव शून्य * __restrict__ vbq, ध्रुव खण्डम्_q8_1 * __restrict__ bq8_1,
                                  ध्रुव पूर्णाङ्क & iqs);

स्थिर __dpct_inline__ पूर्णाङ्क get_int_b1(ध्रुव शून्य * x, ध्रुव पूर्णाङ्क & i32) {
    ध्रुव uint8_t * x8 = (ध्रुव uint8_t *) x;

    पूर्णाङ्क x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    प्रत्यागम x32;
}

स्थिर __dpct_inline__ पूर्णाङ्क get_int_b2(ध्रुव शून्य * x, ध्रुव पूर्णाङ्क & i32) {
    ध्रुव uint16_t * x16 = (ध्रुव uint16_t *) x; // assume at least 2 byte alignment

    पूर्णाङ्क x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    प्रत्यागम x32;
}

स्थिर __dpct_inline__ पूर्णाङ्क get_int_b4(ध्रुव शून्य * x, ध्रुव पूर्णाङ्क & i32) {
    प्रत्यागम ((ध्रुव पूर्णाङ्क *) x)[i32]; // assume at least 4 byte alignment
}

स्थिर __dpct_inline__ पूर्णाङ्क get_int_from_int8(ध्रुव int8_t* x8, ध्रुव पूर्णाङ्क& i32) {
  ध्रुव uint16_t* x16 =
      (ध्रुव uint16_t*)(x8 + परिमाण(पूर्णाङ्क) * i32); // assume at least 2 byte
                                                 // alignment

  पूर्णाङ्क x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  प्रत्यागम x32;
}

स्थिर __dpct_inline__ पूर्णाङ्क get_int_from_uint8(
    ध्रुव uint8_t* x8,
    ध्रुव पूर्णाङ्क& i32) {
  ध्रुव uint16_t* x16 =
      (ध्रुव uint16_t*)(x8 + परिमाण(पूर्णाङ्क) * i32); // assume at least 2 byte
                                                 // alignment

  पूर्णाङ्क x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  प्रत्यागम x32;
}

स्थिर __dpct_inline__ पूर्णाङ्क get_int_from_int8_aligned(
    ध्रुव int8_t* x8,
    ध्रुव पूर्णाङ्क& i32) {
  प्रत्यागम *(
      (ध्रुव पूर्णाङ्क*)(x8 + परिमाण(पूर्णाङ्क) * i32)); // assume at least 4 byte alignment
}

स्थिर __dpct_inline__ पूर्णाङ्क get_int_from_uint8_aligned(
    ध्रुव uint8_t* x8,
    ध्रुव पूर्णाङ्क& i32) {
  प्रत्यागम *(
      (ध्रुव पूर्णाङ्क*)(x8 + परिमाण(पूर्णाङ्क) * i32)); // assume at least 4 byte alignment
}

स्थिर __dpct_inline__ पूर्णाङ्क byte_sub_4(ध्रुव पूर्णाङ्क a, ध्रुव पूर्णाङ्क b) {
  ध्रुव uint32_t ua = static_cast<uint32_t>(a);
  ध्रुव uint32_t ub = static_cast<uint32_t>(b);
  प्रत्यागम static_cast<पूर्णाङ्क>(((ua | 0x80808080u) - ub) ^ 0x80808080u);
}

स्थिर __dpct_inline__ भिन्न vec_dot_q6_K_q8_1_impl_mmvq_scalar(
    ध्रुव पूर्णाङ्क vl, ध्रुव पूर्णाङ्क vh, ध्रुव पूर्णाङ्क u0, ध्रुव पूर्णाङ्क u1, ध्रुव int8_t sc0,
    ध्रुव int8_t sc1, ध्रुव भिन्न d, ध्रुव भिन्न d80, ध्रुव भिन्न d81) {
    static_assert(QR6_K == 2, "q6_K MMVQ scalar fast path assumes QR6_K == 2");

    ध्रुव पूर्णाङ्क vil0 = (vl >> 0) & 0x0F0F0F0F;
    ध्रुव पूर्णाङ्क vih0 = ((vh >> 0) << 4) & 0x30303030;
    ध्रुव पूर्णाङ्क vi0 = byte_sub_4(vil0 | vih0, 0x20202020);

    ध्रुव पूर्णाङ्क vil1 = (vl >> 4) & 0x0F0F0F0F;
    ध्रुव पूर्णाङ्क vih1 = ((vh >> 4) << 4) & 0x30303030;
    ध्रुव पूर्णाङ्क vi1 = byte_sub_4(vil1 | vih1, 0x20202020);

    ध्रुव भिन्न sumf =
        d80 * (dpct::dp4a(vi0, u0, 0) * sc0) +
        d81 * (dpct::dp4a(vi1, u1, 0) * sc1);

    प्रत्यागम d * sumf;
}

स्थिर __dpct_inline__ शून्य get_int_from_सारणी_16(ध्रुव uint32_t &q4,
                                                  ध्रुव uint8_t *मानानि,
                                                  पूर्णाङ्क &val1, पूर्णाङ्क &val2) {

    uint32_t aux32; ध्रुव uint8_t * q8 = (ध्रुव uint8_t *)&aux32;
    aux32 = q4 & 0x0f0f0f0f;
    uint16_t v1 = मानानि[q8[0]] | (मानानि[q8[1]] << 8);
    uint16_t v2 = मानानि[q8[2]] | (मानानि[q8[3]] << 8);
    val1 = v1 | (v2 << 16);
    aux32 = (q4 >> 4) & 0x0f0f0f0f;
    v1 = मानानि[q8[0]] | (मानानि[q8[1]] << 8);
    v2 = मानानि[q8[2]] | (मानानि[q8[3]] << 8);
    val2 = v1 | (v2 << 16);
}

स्थिर __dpct_inline__ sycl::int2 get_int_from_सारणी_16(
    ध्रुव पूर्णाङ्क& q4, ध्रुव int8_t* सारणी) {
  ध्रुव uint32_t* table32 = (ध्रुव uint32_t*)सारणी;
  uint32_t tmp[2];
  ध्रुव uint32_t नीचः_उच्चः_selection_indices =
      (0x32103210 | ((q4 & 0x88888888) >> 1));
#निर्देशः unroll
  प्रति (uint32_t i = 0; i < 2; ++i) {
    ध्रुव uint32_t सरणम् = 16 * i;

    ध्रुव uint32_t नीचः =
        dpct::byte_level_permute(table32[0], table32[1], q4 >> सरणम्);
    ध्रुव uint32_t उच्चः =
        dpct::byte_level_permute(table32[2], table32[3], q4 >> सरणम्);
    tmp[i] = dpct::byte_level_permute(
        नीचः, उच्चः, नीचः_उच्चः_selection_indices >> सरणम्);
  }
  प्रत्यागम sycl::int2(
      dpct::byte_level_permute(tmp[0], tmp[1], 0x6420),
      dpct::byte_level_permute(tmp[0], tmp[1], 0x7531));
}

// Four E2M1 codes (one per byte, bits 0..3) to their kvalues_mxfp4 int8 values. SWAR arithmetic
// replaces get_int_from_table_16 for MXFP4: dpct::byte_level_permute is emulated with 64-bit shifts,
// eight per int, which made the MXFP4 GEMV compute-bound on Intel GPUs.
// Magnitudes 0,1,2,3,4,6,8,12 = m + [m>=5] + [m>=6] + 3*[m>=7]; each byte stays below 256, so the
// byte-wise adds never carry. -0 (code 8) is left as 0 so the two's-complement +1 cannot carry either.
स्थिर __dpct_inline__ पूर्णाङ्क mxfp4_codes_गन्तव्यम्_int8(ध्रुव uint32_t x) {
    ध्रुव uint32_t m   = x & 0x07070707u;
    ध्रुव uint32_t ge5 = ((m + 0x03030303u) >> 3) & 0x01010101u;
    ध्रुव uint32_t ge6 = ((m + 0x02020202u) >> 3) & 0x01010101u;
    ध्रुव uint32_t ge7 = ((m + 0x01010101u) >> 3) & 0x01010101u;
    ध्रुव uint32_t mag = m + ge5 + ge6 + 3u * ge7;
    ध्रुव uint32_t nz  = ((mag + 0x7f7f7f7fu) >> 7) & 0x01010101u;
    ध्रुव uint32_t neg = (x >> 3) & nz & 0x01010101u;
    प्रत्यागम (पूर्णाङ्क) ((mag ^ (neg * 0xffu)) + neg);
}

// Same result as get_int_from_table_16(q4, kvalues_mxfp4): x = low nibbles, y = high nibbles.
स्थिर __dpct_inline__ sycl::int2 get_int_from_mxfp4(ध्रुव पूर्णाङ्क q4) {
    प्रत्यागम sycl::int2(mxfp4_codes_गन्तव्यम्_int8((uint32_t) q4 & 0x0f0f0f0fu),
                      mxfp4_codes_गन्तव्यम्_int8(((uint32_t) q4 >> 4) & 0x0f0f0f0fu));
}

#परिभाषय VDR_Q2_K_Q8_1_MMVQ 1

// contiguous v/x values
स्थिर __dpct_inline__ भिन्न vec_dot_q2_K_q8_1_impl_mmvq(
    ध्रुव पूर्णाङ्क &v, ध्रुव पूर्णाङ्क *__restrict__ u, ध्रुव uint8_t *__restrict__ scales,
    ध्रुव sycl::half2 &dm2, ध्रुव भिन्न *__restrict__ d8) {

    भिन्न sumf_d = 0.0f;
    भिन्न sumf_m = 0.0f;

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < QR2_K; ++i) {
        ध्रुव पूर्णाङ्क sc = scales[2*i];

        ध्रुव पूर्णाङ्क vi = (v >> (2*i)) & 0x03030303;

        sumf_d +=
            d8[i] * (dpct::dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        पूर्णाङ्क m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] *
                  dpct::dp4a(
                      m, u[i],
                      0); // multiply constant q2_K part with sum of q8_1 values
    }

    ध्रुव sycl::float2 dm2f =
        dm2.convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();

    प्रत्यागम dm2f.x() * sumf_d - dm2f.y() * sumf_m;
}


#परिभाषय VDR_Q3_K_Q8_1_MMVQ 1

// contiguous v/x values
स्थिर __dpct_inline__ भिन्न vec_dot_q3_K_q8_1_impl_mmvq(
    ध्रुव पूर्णाङ्क &vl, ध्रुव पूर्णाङ्क &vh, ध्रुव पूर्णाङ्क *__restrict__ u,
    ध्रुव uint8_t *__restrict__ scales, ध्रुव पूर्णाङ्क &scale_विस्थापनम्,
    ध्रुव भिन्न &d3, ध्रुव भिन्न *__restrict__ d8) {

    भिन्न sumf = 0.0f;

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < QR3_K; ++i) {
        ध्रुव पूर्णाङ्क isc = scale_विस्थापनम् + 2*i;

        ध्रुव पूर्णाङ्क isc_नीचः = isc % (QK_K/32);
        ध्रुव पूर्णाङ्क sc_सरणम्_नीचः = 4 * (isc / (QK_K/32));
        ध्रुव पूर्णाङ्क sc_नीचः  = (scales[isc_नीचः] >> sc_सरणम्_नीचः) & 0xF;

        ध्रुव पूर्णाङ्क isc_उच्चः = isc % (QK_K/64);
        ध्रुव पूर्णाङ्क sc_सरणम्_उच्चः = 2 * (isc / (QK_K/64));
        ध्रुव पूर्णाङ्क sc_उच्चः = ((scales[(QK_K/32) + isc_उच्चः] >> sc_सरणम्_उच्चः) & 3) << 4;

        ध्रुव पूर्णाङ्क sc = (sc_नीचः | sc_उच्चः) - 32;

        ध्रुव पूर्णाङ्क vil = (vl >> (2*i)) & 0x03030303;

        ध्रुव पूर्णाङ्क vih = ((vh >> i) << 2) & 0x04040404;

        ध्रुव पूर्णाङ्क vi =
            dpct::vectorized_द्विमान<sycl::char4>(vil, vih, dpct::sub_sat());

        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    प्रत्यागम d3 * sumf;
}

#परिभाषय VDR_Q4_K_Q8_1_MMVQ 2

// contiguous v/x values
स्थिर __dpct_inline__ भिन्न vec_dot_q4_K_q8_1_impl_vmmq(
    ध्रुव पूर्णाङ्क *__restrict__ v, ध्रुव पूर्णाङ्क *__restrict__ u,
    ध्रुव uint8_t *__restrict__ sc, ध्रुव uint8_t *__restrict__ m,
    ध्रुव sycl::half2 &dm4, ध्रुव भिन्न *__restrict__ d8) {

    भिन्न sumf_d = 0.0f;
    भिन्न sumf_m = 0.0f;

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < QR4_K; ++i) {
        ध्रुव पूर्णाङ्क v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        ध्रुव पूर्णाङ्क v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        ध्रुव पूर्णाङ्क dot1 =
            dpct::dp4a(v1i, u[2 * i + 1],
                       dpct::dp4a(v0i, u[2 * i + 0], 0)); // SIMD dot product
        ध्रुव पूर्णाङ्क dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 1],
                       dpct::dp4a(0x01010101, u[2 * i + 0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    ध्रुव sycl::float2 dm4f =
        dm4.convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();

    प्रत्यागम dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


#परिभाषय VDR_Q5_K_Q8_1_MMVQ 2

// contiguous v/x values
स्थिर __dpct_inline__ भिन्न vec_dot_q5_K_q8_1_impl_vmmq(
    ध्रुव पूर्णाङ्क *__restrict__ vl, ध्रुव पूर्णाङ्क *__restrict__ vh,
    ध्रुव पूर्णाङ्क *__restrict__ u, ध्रुव uint8_t *__restrict__ sc,
    ध्रुव uint8_t *__restrict__ m, ध्रुव sycl::half2 &dm5,
    ध्रुव भिन्न *__restrict__ d8) {

    भिन्न sumf_d = 0.0f;
    भिन्न sumf_m = 0.0f;

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < QR5_K; ++i) {
        ध्रुव पूर्णाङ्क vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        ध्रुव पूर्णाङ्क vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        ध्रुव पूर्णाङ्क vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        ध्रुव पूर्णाङ्क vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        ध्रुव पूर्णाङ्क v0i = vl0i | vh0i;
        ध्रुव पूर्णाङ्क v1i = vl1i | vh1i;

        ध्रुव पूर्णाङ्क dot1 =
            dpct::dp4a(v0i, u[2 * i + 0],
                       dpct::dp4a(v1i, u[2 * i + 1], 0)); // SIMD dot product
        ध्रुव पूर्णाङ्क dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 0],
                       dpct::dp4a(0x01010101, u[2 * i + 1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    ध्रुव sycl::float2 dm5f =
        dm5.convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();

    प्रत्यागम dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}


#परिभाषय VDR_Q6_K_Q8_1_MMVQ 1

// contiguous v/x values
स्थिर __dpct_inline__ भिन्न
vec_dot_q6_K_q8_1_impl_mmvq(ध्रुव पूर्णाङ्क &vl, ध्रुव पूर्णाङ्क &vh,
                            ध्रुव पूर्णाङ्क *__restrict__ u,
                            ध्रुव int8_t *__restrict__ scales, ध्रुव भिन्न &d,
                            ध्रुव भिन्न *__restrict__ d8) {
    प्रत्यागम vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
}

#परिभाषय VDR_Q1_0_Q8_1_MMVQ 1
#परिभाषय VDR_Q1_0_Q8_1_MMQ  4

स्थिर __dpct_inline__ भिन्न
vec_dot_q1_0_q8_1(ध्रुव शून्य *__restrict__ vbq,
                  ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {

    ध्रुव खण्डम्_q1_0 * bq1_0 = (ध्रुव खण्डम्_q1_0 *) vbq;

    ध्रुव खण्डम्_q8_1 * bq8_1_chunk = bq8_1 + iqs;
    ध्रुव भिन्न        d1          = bq1_0->d;
    ध्रुव पूर्णाङ्क          v           = get_int_from_uint8_aligned(bq1_0->qs, iqs);

    पूर्णाङ्क vi_अष्टकानि[8];
#निर्देशः unroll
    प्रति (पूर्णाङ्क j = 0; j < 8; ++j) {
        ध्रुव पूर्णाङ्क सरणम् = j * 4;
        ध्रुव पूर्णाङ्क bits4 = (v >> सरणम्) & 0x0F;
        ध्रुव पूर्णाङ्क b0    = (bits4 & 0x01) ? 1 : -1;
        ध्रुव पूर्णाङ्क b1    = (bits4 & 0x02) ? 1 : -1;
        ध्रुव पूर्णाङ्क b2    = (bits4 & 0x04) ? 1 : -1;
        ध्रुव पूर्णाङ्क b3    = (bits4 & 0x08) ? 1 : -1;
        vi_अष्टकानि[j]     = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }

    पूर्णाङ्क sumi = 0;
#निर्देशः unroll
    प्रति (पूर्णाङ्क j = 0; j < 8; ++j) {
        ध्रुव पूर्णाङ्क u = get_int_from_int8_aligned(bq8_1_chunk->qs, j);
        sumi        = ggml_sycl_dp4a(vi_अष्टकानि[j], u, sumi);
    }

    प्रत्यागम d1 * bq8_1_chunk->ds[0] * sumi;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

प्रतिरूपम् <ggml_type T> संरचना reorder_vec_dot_q_sycl {
    static_assert(T != T, "ggml_type for reorder vecdot not implemented");
};

// For some types the weight side of the dot product does not depend on the destination column, so a
// multi-column mul_mat_vec can unpack it once per block instead of once per column. Such a type adds
// load() and dot() next to operator() and opts in here. See reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>.
प्रतिरूपम् <ggml_type T> संरचना reorder_vec_dot_shared_weights {
    स्थिर constexpr तार्किक value = असत्य;
};

प्रतिरूपम् <> संरचना reorder_vec_dot_shared_weights<GGML_TYPE_Q4_K> {
    स्थिर constexpr तार्किक value = सत्य;
};

प्रतिरूपम् <> संरचना reorder_vec_dot_shared_weights<GGML_TYPE_Q5_K> {
    स्थिर constexpr तार्किक value = सत्य;
};

प्रतिरूपम् <ggml_type T> संरचना reorder_vec_dot_shared_activations {
    स्थिर constexpr तार्किक value = असत्य;
};

प्रतिरूपम् <> संरचना reorder_vec_dot_shared_activations<GGML_TYPE_Q4_K> {
    स्थिर constexpr तार्किक value = सत्य;
};

प्रतिरूपम् <> संरचना reorder_vec_dot_shared_activations<GGML_TYPE_Q5_K> {
    स्थिर constexpr तार्किक value = सत्य;
};

प्रतिरूपम् <> संरचना reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0> {
    स्थिर constexpr ggml_type gtype = GGML_TYPE_Q4_0;

    उपयुज्य q4_0_खण्डम्  = ggml_sycl_reordered::खण्डम्_q_t<GGML_TYPE_Q4_0>;
    उपयुज्य q4_0_traits = प्रकारनाम q4_0_खण्डम्::traits;

    __dpct_inline__ भिन्न vec_dot_q4_0_q8_1_impl(ध्रुव पूर्णाङ्क * v, ध्रुव पूर्णाङ्क * u, ध्रुव भिन्न & d4, ध्रुव sycl::half2 & ds8) {
        पूर्णाङ्क sumi = 0;

#निर्देशः unroll
        प्रति (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            ध्रुव पूर्णाङ्क vi0 = (v[i] >> 0) & 0x0F0F0F0F;
            ध्रुव पूर्णाङ्क vi1 = (v[i] >> 4) & 0x0F0F0F0F;

            // SIMD dot product of quantized values
            sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
            sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
        }

        ध्रुव sycl::float2 ds8f = ds8.convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();

        // second part effectively subtracts 8 from each quant value
        प्रत्यागम d4 * (sumi * ds8f.x() - (8 * q4_0_traits::vdr_mmvq / q4_0_traits::qi) * ds8f.y());
    }

    __dpct_inline__ भिन्न संक्रियकः()(ध्रुव शून्य * __restrict__ vbq, ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> ibx_विस्थापनम्,
                                     ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> d_विस्थापनम्, ध्रुव int8_t * q8_1_quant_ptr,
                                     ध्रुव sycl::half2 * q8_1_ds, ध्रुव पूर्णाङ्क & iqs) {
        ध्रुव uint8_t * bq4_0 = static_cast<ध्रुव uint8_t *>(vbq) + ibx_विस्थापनम्.first;
        ध्रुव ggml_half d = *(reinterpret_cast<ध्रुव ggml_half *>(static_cast<ध्रुव uint8_t *>(vbq) + d_विस्थापनम्.first));
        पूर्णाङ्क             v[q4_0_traits::vdr_mmvq];
        पूर्णाङ्क             u[2 * q4_0_traits::vdr_mmvq];


#निर्देशः unroll
        प्रति (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            v[i]         = get_int_from_uint8(bq4_0, iqs + i);
            u[2 * i + 0] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
            u[2 * i + 1] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i + q4_0_traits::qi);
        }

        प्रत्यागम vec_dot_q4_0_q8_1_impl(v, u, d, *q8_1_ds);
    }
};

// Load four contiguous dwords per operand instead of loading each value separately.
संरचना reorder_vec_dot_q8_0_wide {
    स्थिर constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    उपयुज्य q8_0_खण्डम्  = ggml_sycl_reordered::खण्डम्_q_t<GGML_TYPE_Q8_0>;
    उपयुज्य q8_0_traits = प्रकारनाम q8_0_खण्डम्::traits;

    __dpct_inline__ भिन्न संक्रियकः()(ध्रुव शून्य * __restrict__ vbq, ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> ibx_विस्थापनम्,
                                     ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> d_विस्थापनम्, ध्रुव int8_t * q8_1_quant_ptr,
                                     ध्रुव sycl::half2 * q8_1_ds, ध्रुव पूर्णाङ्क & iqs) {
        static_assert(q8_0_traits::vdr_mmvq == 4, "the wide load moves exactly four dwords");

        ध्रुव uint8_t * आधारः = static_cast<ध्रुव uint8_t *>(vbq);
        ध्रुव int8_t *  qs   = reinterpret_cast<ध्रुव int8_t *>(आधारः + ibx_विस्थापनम्.first);
        ध्रुव ggml_half d    = *reinterpret_cast<ध्रुव ggml_half *>(आधारः + d_विस्थापनम्.first);

        ध्रुव sycl::int4 v = *reinterpret_cast<ध्रुव sycl::int4 *>(qs + परिमाण(पूर्णाङ्क) * iqs);
        ध्रुव sycl::int4 u = *reinterpret_cast<ध्रुव sycl::int4 *>(q8_1_quant_ptr + परिमाण(पूर्णाङ्क) * iqs);

        पूर्णाङ्क sumi = 0;
#निर्देशः unroll
        प्रति (पूर्णाङ्क i = 0; i < 4; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        ध्रुव sycl::half2 ds_मानानि = *q8_1_ds;
        प्रत्यागम static_cast<भिन्न>(d) * static_cast<भिन्न>(ds_मानानि[0]) * sumi;
    }
};

प्रतिरूपम् <> संरचना reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0> {
    स्थिर constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    उपयुज्य q8_0_खण्डम्  = ggml_sycl_reordered::खण्डम्_q_t<GGML_TYPE_Q8_0>;
    उपयुज्य q8_0_traits = प्रकारनाम q8_0_खण्डम्::traits;

    __dpct_inline__ भिन्न संक्रियकः()(ध्रुव शून्य * __restrict__ vbq, ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> ibx_विस्थापनम्,
                                     ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> d_विस्थापनम्, ध्रुव int8_t * q8_1_quant_ptr,
                                     ध्रुव sycl::half2 * q8_1_ds, ध्रुव पूर्णाङ्क & iqs,
                                     ध्रुव पूर्णाङ्क vdr = q8_0_traits::vdr_mmvq) {
        ध्रुव uint8_t * आधारः = static_cast<ध्रुव uint8_t *>(vbq);
        ध्रुव int8_t *  qs   = reinterpret_cast<ध्रुव int8_t *>(आधारः + ibx_विस्थापनम्.first);
        ध्रुव ggml_half  d   = *reinterpret_cast<ध्रुव ggml_half *>(आधारः + d_विस्थापनम्.first);

        पूर्णाङ्क v[q8_0_traits::vdr_mmvq];
        पूर्णाङ्क u[q8_0_traits::vdr_mmvq];

#निर्देशः unroll
        प्रति (पूर्णाङ्क i = 0; i < vdr; ++i) {
            v[i] = get_int_from_int8(qs, iqs + i);
            u[i] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
        }

        पूर्णाङ्क sumi = 0;
#निर्देशः unroll
        प्रति (पूर्णाङ्क i = 0; i < vdr; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        ध्रुव sycl::half2 ds_मानानि = *q8_1_ds;
        प्रत्यागम static_cast<भिन्न>(d) * static_cast<भिन्न>(ds_मानानि[0]) * sumi;
    }
};

प्रतिरूपम् <> संरचना reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K> {
    स्थिर constexpr ggml_type gtype = GGML_TYPE_Q2_K;

    उपयुज्य q2_k_खण्डम्  = ggml_sycl_reordered::खण्डम्_q_t<GGML_TYPE_Q2_K>;
    उपयुज्य q2_k_traits = प्रकारनाम q2_k_खण्डम्::traits;

    __dpct_inline__ भिन्न संक्रियकः()(ध्रुव शून्य * __restrict__ vbq, ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> ibx_विस्थापनम्,
                                     ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> d_विस्थापनम्, ध्रुव int8_t * q8_1_quant_ptr,
                                     ध्रुव sycl::half2 * q8_1_ds, ध्रुव पूर्णाङ्क & iqs) {
        ध्रुव uint8_t *    आधारः   = static_cast<ध्रुव uint8_t *>(vbq);
        ध्रुव uint8_t *    qs     = आधारः + ibx_विस्थापनम्.first;
        ध्रुव uint8_t *    scales = आधारः + d_विस्थापनम्.first;
        ध्रुव ggml_half2 * dm     = reinterpret_cast<ध्रुव ggml_half2 *>(आधारः + d_विस्थापनम्.second);

        ध्रुव पूर्णाङ्क bq8_विस्थापनम्   = QR2_K * (iqs / QI8_1);
        ध्रुव पूर्णाङ्क scale_विस्थापनम् = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        ध्रुव पूर्णाङ्क v = get_int_from_uint8_aligned(qs, iqs);

        पूर्णाङ्क   u[QR2_K];
        भिन्न d8[QR2_K];

#निर्देशः unroll
        प्रति (पूर्णाङ्क i = 0; i < QR2_K; ++i) {
            ध्रुव int8_t * quant_आधारः_ptr = q8_1_quant_ptr + (bq8_विस्थापनम् + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_आधारः_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_विस्थापनम् + i))[0];
        }

        प्रत्यागम vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales + scale_विस्थापनम्, *dm, d8);
    }
};

प्रतिरूपम् <> संरचना reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K> {
    स्थिर constexpr ggml_type gtype = GGML_TYPE_Q3_K;

    उपयुज्य q3_k_खण्डम्  = ggml_sycl_reordered::खण्डम्_q_t<GGML_TYPE_Q3_K>;
    उपयुज्य q3_k_traits = प्रकारनाम q3_k_खण्डम्::traits;

    __dpct_inline__ भिन्न संक्रियकः()(ध्रुव शून्य * __restrict__ vbq, ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> ibx_विस्थापनम्,
                                     ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> d_विस्थापनम्, ध्रुव int8_t * q8_1_quant_ptr,
                                     ध्रुव sycl::half2 * q8_1_ds, ध्रुव पूर्णाङ्क & iqs) {
        ध्रुव uint8_t *  आधारः   = static_cast<ध्रुव uint8_t *>(vbq);
        ध्रुव uint8_t *  qs     = आधारः + ibx_विस्थापनम्.first;
        ध्रुव uint8_t *  hmask  = आधारः + ibx_विस्थापनम्.second;
        ध्रुव uint8_t *  scales = आधारः + d_विस्थापनम्.first;
        ध्रुव ggml_half  d      = *reinterpret_cast<ध्रुव ggml_half *>(आधारः + d_विस्थापनम्.second);

        ध्रुव पूर्णाङ्क bq8_विस्थापनम्   = QR3_K * (iqs / (QI3_K / 2));
        ध्रुव पूर्णाङ्क scale_विस्थापनम् = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        ध्रुव पूर्णाङ्क vl = get_int_from_uint8(qs, iqs);
        ध्रुव पूर्णाङ्क vh = ~get_int_from_uint8(hmask, iqs % (QI3_K / 2)) >> bq8_विस्थापनम्;

        पूर्णाङ्क   u[QR3_K];
        भिन्न d8[QR3_K];

#निर्देशः unroll
        प्रति (पूर्णाङ्क i = 0; i < QR3_K; ++i) {
            ध्रुव int8_t * quant_आधारः_ptr = q8_1_quant_ptr + (bq8_विस्थापनम् + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_आधारः_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_विस्थापनम् + i))[0];
        }

        प्रत्यागम vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, scales, scale_विस्थापनम्, static_cast<भिन्न>(d), d8);
    }
};

स्थिर अन्तःस्थ भिन्न vec_dot_q4_K_q8_1_common(ध्रुव पूर्णाङ्क * __restrict__ q4, ध्रुव uint16_t * __restrict__ scales,
                                             ध्रुव ggml_half2 & dm, ध्रुव खण्डम्_q8_1 * __restrict__ bq8_1,
                                             ध्रुव पूर्णाङ्क &        iqs) {
    पूर्णाङ्क   v[2];
    पूर्णाङ्क   u[2 * QR4_K];
    भिन्न d8[QR4_K];

    v[0] = q4[0];
    v[1] = q4[4];

    uint16_t  aux[2];
    ध्रुव पूर्णाङ्क j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
    यदि (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } अन्यथा {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }

    ध्रुव uint8_t * sc = (ध्रुव uint8_t *) aux;
    ध्रुव uint8_t * m  = sc + 2;

    ध्रुव पूर्णाङ्क bq8_विस्थापनम् = QR4_K * ((iqs / 2) / (QI8_1 / 2));

    प्रति (पूर्णाङ्क i = 0; i < QR4_K; ++i) {
        ध्रुव खण्डम्_q8_1 * bq8i = bq8_1 + bq8_विस्थापनम् + i;
        d8[i]                   = bq8i->ds[0];

        ध्रुव पूर्णाङ्क * q8 = (ध्रुव पूर्णाङ्क *) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0]   = q8[0];
        u[2 * i + 1]   = q8[4];
    }

    प्रत्यागम vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
}

प्रतिरूपम् <> संरचना reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K> {
    स्थिर constexpr ggml_type gtype = GGML_TYPE_Q4_K;

    उपयुज्य q4_k_खण्डम्  = ggml_sycl_reordered::खण्डम्_q_t<GGML_TYPE_Q4_K>;
    उपयुज्य q4_k_traits = प्रकारनाम q4_k_खण्डम्::traits;

    संरचना weights {
        पूर्णाङ्क        v[2];
        uint16_t   aux[2];
        ggml_half2 dm;
        पूर्णाङ्क        bq8_विस्थापनम्;
    };

    संरचना activations {
        पूर्णाङ्क   u[2 * QR4_K];
        भिन्न d8[QR4_K];
    };

    __dpct_inline__ स्थिर weights load(ध्रुव शून्य * __restrict__ vbq, ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> ibx_विस्थापनम्,
                                        ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> d_विस्थापनम्, ध्रुव पूर्णाङ्क & iqs) {
        ध्रुव uint8_t *    आधारः = static_cast<ध्रुव uint8_t *>(vbq);
        ध्रुव uint8_t *    qs   = आधारः + ibx_विस्थापनम्.first;
        ध्रुव uint8_t *    scs  = आधारः + d_विस्थापनम्.first;
        ध्रुव ggml_half2 * dms  = reinterpret_cast<ध्रुव ggml_half2 *>(आधारः + d_विस्थापनम्.second);

        weights w;
        w.bq8_विस्थापनम् = QR4_K * ((iqs / 2) / (QI8_1 / 2));

        ध्रुव पूर्णाङ्क *      q4     = (ध्रुव पूर्णाङ्क *) (qs + 16 * w.bq8_विस्थापनम् + 4 * ((iqs / 2) % 4));
        ध्रुव uint16_t * scales = (ध्रुव uint16_t *) scs;

        w.v[0] = q4[0];
        w.v[1] = q4[4];

        ध्रुव पूर्णाङ्क j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        यदि (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } अन्यथा {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        प्रत्यागम w;
    }

    __dpct_inline__ स्थिर activations आरोपय_activations(ध्रुव int8_t * q8_1_quant_ptr,
                                                        ध्रुव sycl::half2 * q8_1_ds, ध्रुव पूर्णाङ्क & iqs) {
        activations a;
        ध्रुव पूर्णाङ्क bq8_विस्थापनम् = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        प्रति (पूर्णाङ्क i = 0; i < QR4_K; ++i) {
            ध्रुव int8_t * quant_आधारः_ptr = q8_1_quant_ptr + (bq8_विस्थापनम् + i) * QK8_1;
            sycl::half2    ds_मानानि      = *(q8_1_ds + bq8_विस्थापनम् + i);

            a.d8[i] = ds_मानानि[0];

            ध्रुव पूर्णाङ्क * q8 = (ध्रुव पूर्णाङ्क *) quant_आधारः_ptr + ((iqs / 2) % 4);
            a.u[2 * i + 0] = q8[0];
            a.u[2 * i + 1] = q8[4];
        }

        प्रत्यागम a;
    }

    __dpct_inline__ स्थिर भिन्न apply(ध्रुव weights & w, ध्रुव activations & a) {
        ध्रुव uint8_t * sc = (ध्रुव uint8_t *) w.aux;
        ध्रुव uint8_t * m  = sc + 2;

        प्रत्यागम vec_dot_q4_K_q8_1_impl_vmmq(w.v, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ स्थिर भिन्न dot(ध्रुव weights & w, ध्रुव int8_t * q8_1_quant_ptr,
                                     ध्रुव sycl::half2 * q8_1_ds, ध्रुव पूर्णाङ्क & iqs) {
        ध्रुव स्वचालित a = आरोपय_activations(q8_1_quant_ptr, q8_1_ds, iqs);

        प्रत्यागम apply(w, a);
    }

    __dpct_inline__ भिन्न संक्रियकः()(ध्रुव शून्य * __restrict__ vbq, ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> ibx_विस्थापनम्,
                                     ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> d_विस्थापनम्, ध्रुव int8_t * q8_1_quant_ptr,
                                     ध्रुव sycl::half2 * q8_1_ds, ध्रुव पूर्णाङ्क & iqs) {
        प्रत्यागम dot(load(vbq, ibx_विस्थापनम्, d_विस्थापनम्, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

प्रतिरूपम् <> संरचना reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K> {
    स्थिर constexpr ggml_type gtype = GGML_TYPE_Q5_K;

    उपयुज्य q5_k_खण्डम्  = ggml_sycl_reordered::खण्डम्_q_t<GGML_TYPE_Q5_K>;
    उपयुज्य q5_k_traits = प्रकारनाम q5_k_खण्डम्::traits;

    संरचना weights {
        पूर्णाङ्क        vl[2];
        पूर्णाङ्क        vh[2];
        uint16_t   aux[2];
        ggml_half2 dm;
    };

    // same activation layout as Q4_K
    static_assert(QR5_K == QR4_K);
    उपयुज्य activations = reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::activations;

    __dpct_inline__ स्थिर weights load(ध्रुव शून्य * __restrict__ vbq, ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> ibx_विस्थापनम्,
                                        ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> d_विस्थापनम्, ध्रुव पूर्णाङ्क & iqs) {
        ध्रुव uint8_t *    आधारः    = static_cast<ध्रुव uint8_t *>(vbq);
        ध्रुव uint8_t *    qs      = आधारः + ibx_विस्थापनम्.first;   // low 4 bits
        ध्रुव uint8_t *    qh_आधारः = आधारः + ibx_विस्थापनम्.second;  // high bit
        ध्रुव uint8_t *    scs     = आधारः + d_विस्थापनम्.first;
        ध्रुव ggml_half2 * dms     = reinterpret_cast<ध्रुव ggml_half2 *>(आधारः + d_विस्थापनम्.second);

        ध्रुव पूर्णाङ्क        bq8_विस्थापनम् = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        ध्रुव पूर्णाङ्क *      ql_ptr     = (ध्रुव पूर्णाङ्क *) (qs + 16 * bq8_विस्थापनम् + 4 * ((iqs / 2) % 4));
        ध्रुव पूर्णाङ्क *      qh_ptr     = (ध्रुव पूर्णाङ्क *) (qh_आधारः + 4 * ((iqs / 2) % 4));
        ध्रुव uint16_t * scales     = (ध्रुव uint16_t *) scs;

        weights w;
        w.vl[0] = ql_ptr[0];
        w.vl[1] = ql_ptr[4];

        w.vh[0] = qh_ptr[0] >> bq8_विस्थापनम्;
        w.vh[1] = qh_ptr[4] >> bq8_विस्थापनम्;

        ध्रुव पूर्णाङ्क j = (QR5_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        यदि (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } अन्यथा {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        प्रत्यागम w;
    }

    __dpct_inline__ स्थिर activations आरोपय_activations(ध्रुव int8_t * q8_1_quant_ptr,
                                                        ध्रुव sycl::half2 * q8_1_ds, ध्रुव पूर्णाङ्क & iqs) {
        प्रत्यागम reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::आरोपय_activations(q8_1_quant_ptr, q8_1_ds, iqs);
    }

    __dpct_inline__ स्थिर भिन्न apply(ध्रुव weights & w, ध्रुव activations & a) {
        ध्रुव uint8_t * sc = (ध्रुव uint8_t *) w.aux;
        ध्रुव uint8_t * m  = sc + 2;

        प्रत्यागम vec_dot_q5_K_q8_1_impl_vmmq(w.vl, w.vh, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ स्थिर भिन्न dot(ध्रुव weights & w, ध्रुव int8_t * q8_1_quant_ptr,
                                     ध्रुव sycl::half2 * q8_1_ds, ध्रुव पूर्णाङ्क & iqs) {
        प्रत्यागम apply(w, आरोपय_activations(q8_1_quant_ptr, q8_1_ds, iqs));
    }

    __dpct_inline__ भिन्न संक्रियकः()(ध्रुव शून्य * __restrict__ vbq, ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> ibx_विस्थापनम्,
                                     ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> d_विस्थापनम्, ध्रुव int8_t * q8_1_quant_ptr,
                                     ध्रुव sycl::half2 * q8_1_ds, ध्रुव पूर्णाङ्क & iqs) {
        प्रत्यागम dot(load(vbq, ibx_विस्थापनम्, d_विस्थापनम्, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

प्रतिरूपम् <> संरचना reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K> {
    स्थिर constexpr ggml_type gtype = GGML_TYPE_Q6_K;

    उपयुज्य q6_k_खण्डम्  = ggml_sycl_reordered::खण्डम्_q_t<GGML_TYPE_Q6_K>;
    उपयुज्य q6_k_traits = प्रकारनाम q6_k_खण्डम्::traits;

    __dpct_inline__ भिन्न vec_dot_q6_K_q8_1_impl_mmvq(ध्रुव पूर्णाङ्क vl, ध्रुव पूर्णाङ्क vh, ध्रुव पूर्णाङ्क * __restrict__ u,
                                                      ध्रुव int8_t * __restrict__ scales, ध्रुव भिन्न d,
                                                      ध्रुव भिन्न * __restrict__ d8) {
        प्रत्यागम vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
    }

    __dpct_inline__ भिन्न संक्रियकः()(ध्रुव शून्य * __restrict__ vbq, ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> ibx_विस्थापनम्,
                     ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> d_विस्थापनम्, ध्रुव int8_t * q8_1_quant_ptr, ध्रुव sycl::half2 * q8_1_ds,
                     ध्रुव पूर्णाङ्क iqs) {
        ध्रुव uint8_t *   आधारः   = static_cast<ध्रुव uint8_t *>(vbq);
        ध्रुव uint8_t *   ql     = आधारः + ibx_विस्थापनम्.first;
        ध्रुव uint8_t *   qh     = आधारः + ibx_विस्थापनम्.second;
        ध्रुव int8_t *    scales = reinterpret_cast<ध्रुव int8_t *>(आधारः + d_विस्थापनम्.first);
        ध्रुव ggml_half * d      = (ध्रुव ggml_half *) (आधारः + d_विस्थापनम्.second);

        ध्रुव पूर्णाङ्क bq8_विस्थापनम्   = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        ध्रुव पूर्णाङ्क scale_विस्थापनम् = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        ध्रुव पूर्णाङ्क vh_सरणम्     = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

        ध्रुव पूर्णाङ्क vl = get_int_from_uint8(ql, iqs);
        ध्रुव पूर्णाङ्क vh = get_int_from_uint8(qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_सरणम्;

        ध्रुव int8_t * scs = scales + scale_विस्थापनम्;

        ध्रुव पूर्णाङ्क u0 = get_int_from_int8_aligned(
            q8_1_quant_ptr + bq8_विस्थापनम् * QK8_1, iqs % QI8_1);
        ध्रुव पूर्णाङ्क u1 = get_int_from_int8_aligned(
            q8_1_quant_ptr + (bq8_विस्थापनम् + 2) * QK8_1, iqs % QI8_1);
        ध्रुव भिन्न d80 = (*(q8_1_ds + bq8_विस्थापनम् + 0))[0];
        ध्रुव भिन्न d81 = (*(q8_1_ds + bq8_विस्थापनम् + 2))[0];

        प्रत्यागम vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u0, u1, scs[0], scs[4], *d, d80, d81);
    }
};

प्रतिरूपम् <> संरचना reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4> {
    स्थिर constexpr ggml_type gtype = GGML_TYPE_MXFP4;

    उपयुज्य mxfp4_खण्डम्  = ggml_sycl_reordered::खण्डम्_q_t<GGML_TYPE_MXFP4>;
    उपयुज्य mxfp4_traits = प्रकारनाम mxfp4_खण्डम्::traits;

    __dpct_inline__ भिन्न संक्रियकः()(ध्रुव शून्य * __restrict__ vbq, ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> ibx_विस्थापनम्,
                                     ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> d_विस्थापनम्, ध्रुव int8_t * q8_1_quant_ptr,
                                     ध्रुव sycl::half2 * q8_1_ds, ध्रुव पूर्णाङ्क & iqs) {
        static_assert(mxfp4_traits::vdr_mmvq == 2, "vector load assumes vdr_mmvq == 2");
        ध्रुव uint8_t * आधारः = static_cast<ध्रुव uint8_t *>(vbq);

        // Reordered nibble blocks are 16 contiguous bytes and iqs is 0 or 2, so each lane's two
        // weight ints are one aligned 8-byte load (the AoS layout needed eight byte loads).
        ध्रुव sycl::int2 q4 = *reinterpret_cast<ध्रुव sycl::int2 *>(आधारः + ibx_विस्थापनम्.first + परिमाण(पूर्णाङ्क) * iqs);
        ध्रुव uint8_t    e  = आधारः[d_विस्थापनम्.first];

        // Low nibbles pair with q8_1 ints iqs..iqs+1, high nibbles with iqs+4..iqs+5.
        ध्रुव sycl::int2 u_lo = *reinterpret_cast<ध्रुव sycl::int2 *>(q8_1_quant_ptr + परिमाण(पूर्णाङ्क) * iqs);
        ध्रुव sycl::int2 u_hi = *reinterpret_cast<ध्रुव sycl::int2 *>(q8_1_quant_ptr + परिमाण(पूर्णाङ्क) * (iqs + 4));

        ध्रुव sycl::int2 v0 = get_int_from_mxfp4(q4.x());
        ध्रुव sycl::int2 v1 = get_int_from_mxfp4(q4.y());

        पूर्णाङ्क sumi = 0;
        sumi = ggml_sycl_dp4a(v0.x(), u_lo.x(), sumi);
        sumi = ggml_sycl_dp4a(v0.y(), u_hi.x(), sumi);
        sumi = ggml_sycl_dp4a(v1.x(), u_lo.y(), sumi);
        sumi = ggml_sycl_dp4a(v1.y(), u_hi.y(), sumi);

        ध्रुव भिन्न d = ggml_sycl_e8m0_गन्तव्यम्_fp32(e) * 0.5f * static_cast<भिन्न>((*q8_1_ds)[0]);
        प्रत्यागम d * sumi;
    }
};
#परिभाषय VDR_Q4_0_Q8_1_MMVQ 2
#परिभाषय VDR_Q4_0_Q8_1_MMQ  4

#परिभाषय VDR_Q2_0_Q8_1_MMVQ 1

प्रतिरूपम् <पूर्णाङ्क vdr>
स्थिर __dpct_inline__ भिन्न vec_dot_q2_0_q8_1_impl(
    ध्रुव पूर्णाङ्क * v,
    ध्रुव पूर्णाङ्क * u,
    ध्रुव भिन्न & d2,
    ध्रुव sycl::half2 & ds8) {
    पूर्णाङ्क sumi = 0;

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < vdr; ++i) {
#निर्देशः unroll
        प्रति (पूर्णाङ्क j = 0; j < 4; ++j) {
            ध्रुव uint8_t q = (uint8_t) ((uint32_t) v[i] >> (8 * j));

            // unpack 2-bit values to byte lanes (0..3), then apply zero-point
            // correction with ds8f.y() below, mirroring the q4_0 style.
            पूर्णाङ्क vi = 0;
            vi |= (((q >> 0) & 0x3) & 0xFF) << 0;
            vi |= (((q >> 2) & 0x3) & 0xFF) << 8;
            vi |= (((q >> 4) & 0x3) & 0xFF) << 16;
            vi |= (((q >> 6) & 0x3) & 0xFF) << 24;

            sumi = dpct::dp4a(vi, u[4 * i + j], sumi);
        }
    }

    ध्रुव sycl::float2 ds8f = ds8.convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();
    // q2_0 has zero-point 1. Scale ds8f.y() by processed-lane ratio,
    // consistent with q4_0's explicit zero-point subtraction style.
    प्रत्यागम d2 * (sumi * ds8f.x() - ((भिन्न) vdr / (भिन्न) QI2_0) * ds8f.y());
}

प्रतिरूपम् <पूर्णाङ्क vdr>
स्थिर __dpct_inline__ भिन्न vec_dot_q4_0_q8_1_impl(ध्रुव पूर्णाङ्क * v, ध्रुव पूर्णाङ्क * u, ध्रुव भिन्न & d4,
                                                    ध्रुव sycl::half2 & ds8) {
    पूर्णाङ्क sumi = 0;
#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < vdr; ++i) {
        ध्रुव पूर्णाङ्क vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        ध्रुव पूर्णाङ्क vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

    ध्रुव sycl::float2 ds8f = ds8.convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();

    // second part effectively subtracts 8 from each quant value
    प्रत्यागम d4 * (sumi * ds8f.x() - (8 * vdr / QI4_0) * ds8f.y());
}

#परिभाषय VDR_Q4_1_Q8_1_MMVQ 2
#परिभाषय VDR_Q4_1_Q8_1_MMQ  4

प्रतिरूपम् <पूर्णाङ्क vdr>
स्थिर __dpct_inline__ भिन्न vec_dot_q4_1_q8_1_impl(ध्रुव पूर्णाङ्क *v, ध्रुव पूर्णाङ्क *u,
                                                    ध्रुव sycl::half2 &dm4,
                                                    ध्रुव sycl::half2 &ds8) {

    पूर्णाङ्क sumi = 0;

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < vdr; ++i) {
        ध्रुव पूर्णाङ्क vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        ध्रुव पूर्णाङ्क vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

#यदि_परिभाषितम् GGML_SYCL_F16
    ध्रुव sycl::float2 tmp =
        (dm4 * ds8).convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();
    ध्रुव भिन्न d4d8 = tmp.x();
    ध्रुव भिन्न m4s8 = tmp.y();
#अन्यथा
    ध्रुव sycl::float2 dm4f =
        dm4.convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();
    ध्रुव sycl::float2 ds8f =
        ds8.convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();
    ध्रुव भिन्न d4d8 = dm4f.x() * ds8f.x();
    ध्रुव भिन्न m4s8 = dm4f.y() * ds8f.y();
#यदि_अन्तः // GGML_SYCL_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    प्रत्यागम sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#परिभाषय VDR_Q5_0_Q8_1_MMVQ 2
#परिभाषय VDR_Q5_0_Q8_1_MMQ  4

प्रतिरूपम् <पूर्णाङ्क vdr>
स्थिर __dpct_inline__ भिन्न
vec_dot_q5_0_q8_1_impl(ध्रुव पूर्णाङ्क *vl, ध्रुव पूर्णाङ्क *vh, ध्रुव पूर्णाङ्क *u,
                       ध्रुव भिन्न &d5, ध्रुव sycl::half2 &ds8) {
    पूर्णाङ्क sumi = 0;

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < vdr; ++i) {
        पूर्णाङ्क vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        पूर्णाङ्क vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

    ध्रुव sycl::float2 ds8f =
        ds8.convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();

    // second part effectively subtracts 16 from each quant value
    प्रत्यागम d5 * (sumi * ds8f.x() - (16 * vdr / QI5_0) * ds8f.y());
}

#परिभाषय VDR_Q5_1_Q8_1_MMVQ 2
#परिभाषय VDR_Q5_1_Q8_1_MMQ  4

प्रतिरूपम् <पूर्णाङ्क vdr>
स्थिर __dpct_inline__ भिन्न
vec_dot_q5_1_q8_1_impl(ध्रुव पूर्णाङ्क *vl, ध्रुव पूर्णाङ्क *vh, ध्रुव पूर्णाङ्क *u,
                       ध्रुव sycl::half2 &dm5, ध्रुव sycl::half2 &ds8) {

    पूर्णाङ्क sumi = 0;

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < vdr; ++i) {
        पूर्णाङ्क vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        पूर्णाङ्क vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

#यदि_परिभाषितम् GGML_SYCL_F16
     ध्रुव sycl::float2 tmp =
        (dm5 * ds8).convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();
    ध्रुव भिन्न d5d8 = tmp.x();
    ध्रुव भिन्न m5s8 = tmp.y();


#अन्यथा
    ध्रुव sycl::float2 dm5f =
        dm5.convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();
    ध्रुव sycl::float2 ds8f =
        ds8.convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();
    ध्रुव भिन्न d5d8 = dm5f.x() * ds8f.x();
    ध्रुव भिन्न m5s8 = dm5f.y() * ds8f.y();
#यदि_अन्तः // GGML_SYCL_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    प्रत्यागम sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#परिभाषय VDR_Q8_0_Q8_1_MMVQ 2
#परिभाषय VDR_Q8_0_Q8_1_MMQ 8

प्रतिरूपम् <पूर्णाङ्क vdr>
स्थिर __dpct_inline__ भिन्न vec_dot_q8_0_q8_1_impl(ध्रुव पूर्णाङ्क *v, ध्रुव पूर्णाङ्क *u,
                                                    ध्रुव भिन्न &d8_0,
                                                    ध्रुव भिन्न &d8_1) {

    पूर्णाङ्क sumi = 0;

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

    प्रत्यागम d8_0*d8_1 * sumi;
}

प्रतिरूपम् <प्रकारनाम T, पूर्णाङ्क vdr>
स्थिर __dpct_inline__ T vec_dot_q8_0_q8_1_impl(ध्रुव पूर्णाङ्क * v, ध्रुव पूर्णाङ्क * u, ध्रुव T & d8_0, ध्रुव T & d8_1) {
    पूर्णाङ्क sumi = 0;

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_sycl_dp4a(v[i], u[i], sumi);
    }

    प्रत्यागम d8_0*d8_1 * ((T) sumi);
}

प्रतिरूपम् <पूर्णाङ्क vdr>
स्थिर __dpct_inline__ भिन्न vec_dot_q8_1_q8_1_impl(ध्रुव पूर्णाङ्क *v, ध्रुव पूर्णाङ्क *u,
                                                    ध्रुव sycl::half2 &dm8,
                                                    ध्रुव sycl::half2 &ds8) {

    पूर्णाङ्क sumi = 0;

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

#यदि_परिभाषितम् GGML_SYCL_F16
    ध्रुव sycl::float2 tmp =
        (dm8 * ds8).convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();
    ध्रुव भिन्न d8d8 = tmp.x();
    ध्रुव भिन्न m8s8 = tmp.y();
#अन्यथा
    ध्रुव sycl::float2 dm8f =
        dm8.convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();
    ध्रुव sycl::float2 ds8f =
        ds8.convert<भिन्न, sycl::rounding_बहुलकम्::automatic>();
    ध्रुव भिन्न d8d8 = dm8f.x() * ds8f.x();
    ध्रुव भिन्न m8s8 = dm8f.y() * ds8f.y();
#यदि_अन्तः // GGML_SYCL_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    प्रत्यागम sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

स्थिर __dpct_inline__ भिन्न
vec_dot_q4_0_q8_1(ध्रुव शून्य *__restrict__ vbq,
                  ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {

    ध्रुव खण्डम्_q4_0 * bq4_0 = (ध्रुव खण्डम्_q4_0 *) vbq;

    पूर्णाङ्क v[VDR_Q4_0_Q8_1_MMVQ];
    पूर्णाङ्क u[2 * VDR_Q4_0_Q8_1_MMVQ];

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]         = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    प्रत्यागम vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

स्थिर __dpct_inline__ भिन्न
vec_dot_q2_0_q8_1(ध्रुव शून्य *__restrict__ vbq,
                  ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {

    ध्रुव खण्डम्_q2_0 * bq2_0 = (ध्रुव खण्डम्_q2_0 *) vbq;

    पूर्णाङ्क v[2 * VDR_Q2_0_Q8_1_MMVQ];
    पूर्णाङ्क u[8 * VDR_Q2_0_Q8_1_MMVQ];

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < VDR_Q2_0_Q8_1_MMVQ; ++i) {
        ध्रुव पूर्णाङ्क आधारः = 4 * (iqs + i);

        // Q2_0 has QK2_0 = 64 and uses 2 x QK8_1 blocks on the RHS.
        v[2 * i + 0] = get_int_from_uint8(bq2_0->qs, iqs + i);
        v[2 * i + 1] = get_int_from_uint8(bq2_0->qs, iqs + i + QI2_0);

        u[8 * i + 0] = get_int_from_int8_aligned(bq8_1[0].qs, आधारः + 0);
        u[8 * i + 1] = get_int_from_int8_aligned(bq8_1[0].qs, आधारः + 1);
        u[8 * i + 2] = get_int_from_int8_aligned(bq8_1[0].qs, आधारः + 2);
        u[8 * i + 3] = get_int_from_int8_aligned(bq8_1[0].qs, आधारः + 3);

        u[8 * i + 4] = get_int_from_int8_aligned(bq8_1[1].qs, आधारः + 0);
        u[8 * i + 5] = get_int_from_int8_aligned(bq8_1[1].qs, आधारः + 1);
        u[8 * i + 6] = get_int_from_int8_aligned(bq8_1[1].qs, आधारः + 2);
        u[8 * i + 7] = get_int_from_int8_aligned(bq8_1[1].qs, आधारः + 3);
    }

    ध्रुव भिन्न sum0 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + 0, u + 0, bq2_0->d, bq8_1[0].ds);
    ध्रुव भिन्न sum1 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + VDR_Q2_0_Q8_1_MMVQ, u + 4 * VDR_Q2_0_Q8_1_MMVQ, bq2_0->d, bq8_1[1].ds);
    प्रत्यागम sum0 + sum1;
}

स्थिर __dpct_inline__ भिन्न
vec_dot_q4_1_q8_1(ध्रुव शून्य *__restrict__ vbq,
                  ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {

    ध्रुव खण्डम्_q4_1 * bq4_1 = (ध्रुव खण्डम्_q4_1 *) vbq;

    पूर्णाङ्क v[VDR_Q4_1_Q8_1_MMVQ];
    पूर्णाङ्क u[2*VDR_Q4_1_Q8_1_MMVQ];

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    प्रत्यागम vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

#परिभाषय VDR_MXFP4_Q8_1_MMVQ 2
#परिभाषय VDR_MXFP4_Q8_1_MMQ  4

स्थिर __dpct_inline__ भिन्न vec_dot_mxfp4_q8_1(ध्रुव शून्य * __restrict__ vbq,
                                                ध्रुव खण्डम्_q8_1 * __restrict__ bq8_1,
                                                ध्रुव पूर्णाङ्क & iqs) {
    ध्रुव खण्डम्_mxfp4 * bq4 = (ध्रुव खण्डम्_mxfp4 *) vbq;

    ध्रुव पूर्णाङ्क * q8 = (ध्रुव पूर्णाङ्क *) bq8_1->qs + iqs;

    पूर्णाङ्क sumi = 0;
#निर्देशः unroll
    प्रति (पूर्णाङ्क l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        ध्रुव पूर्णाङ्क aux_q4 = get_int_b1(bq4->qs, iqs + l);
        ध्रुव sycl::int2 v      = get_int_from_mxfp4(aux_q4);
        sumi = ggml_sycl_dp4a(v.x(), q8[l + 0], sumi);
        sumi = ggml_sycl_dp4a(v.y(), q8[l + 4], sumi);
    }

    ध्रुव भिन्न d = ggml_sycl_e8m0_गन्तव्यम्_fp32(bq4->e) * 0.5f * (bq8_1->ds)[0];
    प्रत्यागम d * sumi;
}

#परिभाषय VDR_NVFP4_Q8_1_MMVQ 4
#परिभाषय VDR_NVFP4_Q8_1_MMQ  8

स्थिर __dpct_inline__ भिन्न vec_dot_nvfp4_q8_1(ध्रुव शून्य * __restrict__ vbq,
                                                ध्रुव खण्डम्_q8_1 * __restrict__ bq8_1,
                                                ध्रुव int32_t & iqs) {
    ध्रुव खण्डम्_nvfp4 * bq4 = (ध्रुव खण्डम्_nvfp4 *) vbq;
    भिन्न योग_2 = 0.0f;
#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; i++) {
        ध्रुव int32_t iqs0 = iqs + 2*i;
        ध्रुव int32_t iqs1 = iqs0 + 1;
        ध्रुव int32_t is = iqs0 >> 1;
        ध्रुव sycl::int2   v0   = get_int_from_सारणी_16(get_int_b4(bq4->qs, iqs0), kvalues_mxfp4);
        ध्रुव sycl::int2   v1   = get_int_from_सारणी_16(get_int_b4(bq4->qs, iqs1), kvalues_mxfp4);
        ध्रुव खण्डम्_q8_1 * bq8 = bq8_1 + (is >> 1);
        ध्रुव int32_t i8 = ((is & 1) << 2);

        पूर्णाङ्क sumi = ggml_sycl_dp4a(v0.x(), get_int_b4(bq8->qs, i8 + 0), 0);
        sumi     = ggml_sycl_dp4a(v0.y(), get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi     = ggml_sycl_dp4a(v1.x(), get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi     = ggml_sycl_dp4a(v1.y(), get_int_b4(bq8->qs, i8 + 3), sumi);

        ध्रुव भिन्न d = ggml_sycl_ue4m3_गन्तव्यम्_fp32(bq4->d[is]) * (bq8->ds)[0];
        योग_2 += d * भिन्न(sumi);
    }

    प्रत्यागम योग_2;
}

स्थिर __dpct_inline__ भिन्न
vec_dot_q5_0_q8_1(ध्रुव शून्य *__restrict__ vbq,
                  ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {

    ध्रुव खण्डम्_q5_0 * bq5_0 = (ध्रुव खण्डम्_q5_0 *) vbq;

    पूर्णाङ्क vl[VDR_Q5_0_Q8_1_MMVQ];
    पूर्णाङ्क vh[VDR_Q5_0_Q8_1_MMVQ];
    पूर्णाङ्क  u[2*VDR_Q5_0_Q8_1_MMVQ];

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    प्रत्यागम vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

स्थिर __dpct_inline__ भिन्न
vec_dot_q5_1_q8_1(ध्रुव शून्य *__restrict__ vbq,
                  ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {

    ध्रुव खण्डम्_q5_1 * bq5_1 = (ध्रुव खण्डम्_q5_1 *) vbq;

    पूर्णाङ्क vl[VDR_Q5_1_Q8_1_MMVQ];
    पूर्णाङ्क vh[VDR_Q5_1_Q8_1_MMVQ];
    पूर्णाङ्क  u[2*VDR_Q5_1_Q8_1_MMVQ];

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    प्रत्यागम vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

स्थिर __dpct_inline__ भिन्न
vec_dot_q8_0_q8_1(ध्रुव शून्य *__restrict__ vbq,
                  ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {

    ध्रुव खण्डम्_q8_0 * bq8_0 = (ध्रुव खण्डम्_q8_0 *) vbq;

    पूर्णाङ्क v[VDR_Q8_0_Q8_1_MMVQ];
    पूर्णाङ्क u[VDR_Q8_0_Q8_1_MMVQ];

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    प्रत्यागम vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                      bq8_1->ds[0]);
}

स्थिर __dpct_inline__ भिन्न
vec_dot_q2_K_q8_1(ध्रुव शून्य *__restrict__ vbq,
                  ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {

    ध्रुव खण्डम्_q2_K * bq2_K = (ध्रुव खण्डम्_q2_K *) vbq;

    ध्रुव पूर्णाङ्क bq8_विस्थापनम् = QR2_K * (iqs / QI8_1);
    ध्रुव पूर्णाङ्क scale_विस्थापनम् = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    ध्रुव uint8_t * scales = bq2_K->scales + scale_विस्थापनम्;

    ध्रुव पूर्णाङ्क v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    पूर्णाङ्क    u[QR2_K];
    भिन्न d8[QR2_K];

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_विस्थापनम् + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_विस्थापनम् + i].ds[0];
    }

    प्रत्यागम vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

स्थिर __dpct_inline__ भिन्न
vec_dot_q3_K_q8_1(ध्रुव शून्य *__restrict__ vbq,
                  ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {

    ध्रुव खण्डम्_q3_K * bq3_K = (ध्रुव खण्डम्_q3_K *) vbq;

    ध्रुव पूर्णाङ्क bq8_विस्थापनम् = QR3_K * (iqs / (QI3_K/2));
    ध्रुव पूर्णाङ्क scale_विस्थापनम् = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    ध्रुव भिन्न d = bq3_K->d;

    ध्रुव पूर्णाङ्क vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    ध्रुव पूर्णाङ्क vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_विस्थापनम्;

    पूर्णाङ्क    u[QR3_K];
    भिन्न d8[QR3_K];

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_विस्थापनम् + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_विस्थापनम् + i].ds[0];
    }

    प्रत्यागम vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_विस्थापनम्, d, d8);
}

स्थिर __dpct_inline__ भिन्न vec_dot_q4_K_q8_1(ध्रुव शून्य * __restrict__ vbq, ध्रुव खण्डम्_q8_1 * __restrict__ bq8_1,
                                               ध्रुव पूर्णाङ्क & iqs) {
#यदि_अपरिभाषितम् GGML_QKK_64

    ध्रुव खण्डम्_q4_K * bq4_K = (ध्रुव खण्डम्_q4_K *) vbq;

    ध्रुव पूर्णाङ्क        bq8_विस्थापनम् = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    ध्रुव पूर्णाङ्क *      q4         = (ध्रुव पूर्णाङ्क *) (bq4_K->qs + 16 * bq8_विस्थापनम् + 4 * ((iqs / 2) % 4));
    ध्रुव uint16_t * scales     = (ध्रुव uint16_t *) bq4_K->scales;

    प्रत्यागम vec_dot_q4_K_q8_1_common(q4, scales, bq4_K->dm, bq8_1, iqs);

#अन्यथा

#यदि __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    ध्रुव खण्डम्_q4_K * bq4_K = (ध्रुव खण्डम्_q4_K *) vbq;

    भिन्न sumf_d = 0.0f;
    भिन्न sumf_m = 0.0f;

    uint16_t aux16[2];
    ध्रुव uint8_t * s = (ध्रुव uint8_t *)aux16;

    ध्रुव uint16_t * a = (ध्रुव uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    ध्रुव भिन्न dall = bq4_K->dm[0];
    ध्रुव भिन्न dmin = bq4_K->dm[1];

    ध्रुव भिन्न d8_1 = bq8_1[0].ds[0];
    ध्रुव भिन्न d8_2 = bq8_1[1].ds[1];

    ध्रुव पूर्णाङ्क ui1 = *((ध्रुव पूर्णाङ्क *)bq8_1[0].qs + (iqs/2));
    ध्रुव पूर्णाङ्क ui2 = *((ध्रुव पूर्णाङ्क *)bq8_1[0].qs + (iqs/2) + 4);
    ध्रुव पूर्णाङ्क ui3 = *((ध्रुव पूर्णाङ्क *)bq8_1[1].qs + (iqs/2));
    ध्रुव पूर्णाङ्क ui4 = *((ध्रुव पूर्णाङ्क *)bq8_1[1].qs + (iqs/2) + 4);

    ध्रुव पूर्णाङ्क * q4 = (ध्रुव पूर्णाङ्क *)bq4_K->qs + (iqs/2);
    ध्रुव पूर्णाङ्क v1 = q4[0];
    ध्रुव पूर्णाङ्क v2 = q4[4];

    ध्रुव पूर्णाङ्क dot1 = dpct::dp4a(ui2, v2 & 0x0f0f0f0f, dpct::dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    ध्रुव पूर्णाङ्क dot2 = dpct::dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, dpct::dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    ध्रुव पूर्णाङ्क dot3 = dpct::dp4a(0x01010101, ui2, dpct::dp4a(0x01010101, ui1, 0));
    ध्रुव पूर्णाङ्क dot4 = dpct::dp4a(0x01010101, ui4, dpct::dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    प्रत्यागम dall * sumf_d - dmin * sumf_m;

#अन्यथा
    bad_arch();
#यदि_अन्तः // __SYCL_ARCH__ >= VER_4VEC

#यदि_अन्तः
}

स्थिर __dpct_inline__ भिन्न
vec_dot_q5_K_q8_1(ध्रुव शून्य *__restrict__ vbq,
                  ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {

#यदि_अपरिभाषितम् GGML_QKK_64
    ध्रुव खण्डम्_q5_K * bq5_K = (ध्रुव खण्डम्_q5_K *) vbq;

    पूर्णाङ्क   vl[2];
    पूर्णाङ्क   vh[2];
    पूर्णाङ्क    u[2*QR5_K];
    भिन्न d8[QR5_K];

    ध्रुव पूर्णाङ्क bq8_विस्थापनम् = QR5_K * ((iqs/2) / (QI8_1/2));
    ध्रुव पूर्णाङ्क * ql = (ध्रुव पूर्णाङ्क *)(bq5_K->qs + 16 * bq8_विस्थापनम् + 4 * ((iqs/2)%4));
    ध्रुव पूर्णाङ्क * qh = (ध्रुव पूर्णाङ्क *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_विस्थापनम्;
    vh[1] = qh[4] >> bq8_विस्थापनम्;

    ध्रुव uint16_t * scales = (ध्रुव uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    ध्रुव पूर्णाङ्क j = bq8_विस्थापनम्/2;
    यदि (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } अन्यथा {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    ध्रुव uint8_t * sc = (ध्रुव uint8_t *)aux;
    ध्रुव uint8_t * m  = sc + 2;

#निर्देशः unroll
    प्रति (पूर्णाङ्क i = 0; i < QR5_K; ++i) {
        ध्रुव खण्डम्_q8_1 * bq8i = bq8_1 + bq8_विस्थापनम् + i;
        d8[i] = bq8i->ds[0];

        ध्रुव पूर्णाङ्क * q8 = (ध्रुव पूर्णाङ्क *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    प्रत्यागम vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#अन्यथा

#यदि __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    ध्रुव खण्डम्_q5_K * bq5_K = (ध्रुव खण्डम्_q5_K *) vbq;

    ध्रुव int8_t * s = bq5_K->scales;

    ध्रुव भिन्न d = bq5_K->d;

    ध्रुव भिन्न d8_1 = bq8_1[0].ds[0];
    ध्रुव भिन्न d8_2 = bq8_1[1].ds[1];

    ध्रुव पूर्णाङ्क ui1 = *((ध्रुव पूर्णाङ्क *)bq8_1[0].qs + (iqs/2));
    ध्रुव पूर्णाङ्क ui2 = *((ध्रुव पूर्णाङ्क *)bq8_1[0].qs + (iqs/2) + 4);
    ध्रुव पूर्णाङ्क ui3 = *((ध्रुव पूर्णाङ्क *)bq8_1[1].qs + (iqs/2));
    ध्रुव पूर्णाङ्क ui4 = *((ध्रुव पूर्णाङ्क *)bq8_1[1].qs + (iqs/2) + 4);

    ध्रुव पूर्णाङ्क * ql = (ध्रुव पूर्णाङ्क *)bq5_K->qs + (iqs/2);
    ध्रुव पूर्णाङ्क vl1 = ql[0];
    ध्रुव पूर्णाङ्क vl2 = ql[4];

    ध्रुव पूर्णाङ्क चरणः = 4 * (iqs/2); // 0, 4, 8, 12
    ध्रुव पूर्णाङ्क im = चरणः/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    ध्रुव पूर्णाङ्क अन्तर् = चरणः%8; // 0, 4, 0, 4
    ध्रुव पूर्णाङ्क vh = (*((ध्रुव पूर्णाङ्क *)(bq5_K->qh + अन्तर्))) >> im;

    ध्रुव पूर्णाङ्क v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    ध्रुव पूर्णाङ्क v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    ध्रुव पूर्णाङ्क v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    ध्रुव पूर्णाङ्क v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    ध्रुव भिन्न sumf_d = d8_1 * (dpct::dp4a(ui1, v1, 0) * s[0] + dpct::dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (dpct::dp4a(ui3, v3, 0) * s[2] + dpct::dp4a(ui4, v4, 0) * s[3]);

    प्रत्यागम d * sumf_d;

#अन्यथा
    bad_arch();
#यदि_अन्तः // __SYCL_ARCH__ >= VER_4VEC

#यदि_अन्तः
}

स्थिर __dpct_inline__ भिन्न
vec_dot_q6_K_q8_1(ध्रुव शून्य *__restrict__ vbq,
                  ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {

    ध्रुव खण्डम्_q6_K * bq6_K = (ध्रुव खण्डम्_q6_K *) vbq;

    ध्रुव पूर्णाङ्क bq8_विस्थापनम् = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    ध्रुव पूर्णाङ्क scale_विस्थापनम् = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    ध्रुव पूर्णाङ्क vh_सरणम् = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    ध्रुव पूर्णाङ्क vl = get_int_from_uint8(bq6_K->ql, iqs);
    ध्रुव पूर्णाङ्क vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_सरणम्;

    ध्रुव int8_t * scales = bq6_K->scales + scale_विस्थापनम्;

    ध्रुव पूर्णाङ्क u0 = get_int_from_int8_aligned(
        bq8_1[bq8_विस्थापनम् + 0].qs, iqs % QI8_1);
    ध्रुव पूर्णाङ्क u1 = get_int_from_int8_aligned(
        bq8_1[bq8_विस्थापनम् + 2].qs, iqs % QI8_1);
    ध्रुव भिन्न d80 = bq8_1[bq8_विस्थापनम् + 0].ds[0];
    ध्रुव भिन्न d81 = bq8_1[bq8_विस्थापनम् + 2].ds[0];

    प्रत्यागम vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u0, u1, scales[0], scales[4], bq6_K->d, d80, d81);
}


// NOTE: the VDR_IQ*_Q8_1_MMVQ values deliberately differ from the identically named CUDA constants
// (vecdotq.cuh): the SYCL kernels pair them with a halved qi (e.g. QI3_S/2), so the values are not
// interchangeable and must not be copied across backends.
#परिभाषय VDR_IQ2_XXS_Q8_1_MMVQ 1

स्थिर __dpct_inline__ uint32_t iq_sign_आवरणम्(ध्रुव uint32_t signs) {
    ध्रुव uint32_t द्वयङ्काः =
        (signs & 0x01) | ((signs & 0x02) << 7) |
        ((signs & 0x04) << 14) | ((signs & 0x08) << 21);

    प्रत्यागम द्वयङ्काः * 0xff;
}

स्थिर __dpct_inline__ uint32_t iq_sign_द्वयङ्काः(ध्रुव uint32_t encoded) {
    ध्रुव uint32_t signs = encoded & 127;

    प्रत्यागम signs | ((sycl::popcount(signs) & 1) << 7);
}

स्थिर __dpct_inline__ भिन्न
vec_dot_iq2_xxs_q8_1(ध्रुव शून्य *__restrict__ vbq,
                     ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs,
                     ध्रुव uint64_t *iq2xxs_grid, ध्रुव uint8_t *ksigns_iq2xs,
                     ध्रुव uint8_t *kmask_iq2xs) {
#यदि QK_K == 256
    ध्रुव खण्डम्_iq2_xxs * bq2 = (ध्रुव खण्डम्_iq2_xxs *) vbq;

    ध्रुव पूर्णाङ्क ib32 = iqs;
    ध्रुव uint16_t * q2 = bq2->qs + 4*ib32;
    ध्रुव uint8_t  * aux8 = (ध्रुव uint8_t *)q2;
    ध्रुव int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = q2[2] | (q2[3] << 16);
    पूर्णाङ्क sumi = 0;
    प्रति (पूर्णाङ्क l = 0; l < 4; ++l) {
        ध्रुव uint8_t * grid = (ध्रुव uint8_t *)(iq2xxs_grid + aux8[l]);
        ध्रुव uint8_t  signs = ksigns_iq2xs[aux32 & 127];
        प्रति (पूर्णाङ्क j = 0; j < 8; ++j) {
            sumi += q8[j] * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
        aux32 >>= 7;
    }
    ध्रुव भिन्न d = (भिन्न)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.25f;
    प्रत्यागम d * sumi;
#अन्यथा
    प्रतिज्ञा(असत्य);
    प्रत्यागम 0.f;
#यदि_अन्तः
}

#परिभाषय VDR_IQ2_XS_Q8_1_MMVQ 1

स्थिर __dpct_inline__ भिन्न
vec_dot_iq2_xs_q8_1(ध्रुव शून्य *__restrict__ vbq,
                    ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs,
                    ध्रुव uint64_t *iq2xs_grid) {
#यदि DPCT_COMPATIBILITY_अस्थायी >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#यदि QK_K == 256
    ध्रुव खण्डम्_iq2_xs * bq2 = (ध्रुव खण्डम्_iq2_xs *) vbq;

    ध्रुव पूर्णाङ्क ib32 = iqs;
    ध्रुव uint16_t * q2 = bq2->qs + 4*ib32;
    ध्रुव int8_t   * q8 = bq8_1[ib32].qs;
    ध्रुव uint8_t ls1 = bq2->scales[ib32] & 0xf;
    ध्रुव uint8_t ls2 = bq2->scales[ib32] >>  4;
    पूर्णाङ्क sumi1 = 0;
#निर्देशः unroll
    प्रति (पूर्णाङ्क l = 0; l < 2; ++l) {
        ध्रुव uint64_t grid = iq2xs_grid[q2[l] & 511];
        ध्रुव uint32_t sign_द्वयङ्काः = iq_sign_द्वयङ्काः(q2[l] >> 9);
        ध्रुव uint32_t signs0 = iq_sign_आवरणम्(sign_द्वयङ्काः);
        ध्रुव uint32_t signs1 = iq_sign_आवरणम्(sign_द्वयङ्काः >> 4);
        ध्रुव पूर्णाङ्क grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        ध्रुव पूर्णाङ्क grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi1 = dpct::dp4a(grid_l, *((ध्रुव पूर्णाङ्क *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((ध्रुव पूर्णाङ्क *)q8 + 1), sumi1);
        q8 += 8;
    }
    पूर्णाङ्क sumi2 = 0;
#निर्देशः unroll
    प्रति (पूर्णाङ्क l = 2; l < 4; ++l) {
        ध्रुव uint64_t grid = iq2xs_grid[q2[l] & 511];
        ध्रुव uint32_t sign_द्वयङ्काः = iq_sign_द्वयङ्काः(q2[l] >> 9);
        ध्रुव uint32_t signs0 = iq_sign_आवरणम्(sign_द्वयङ्काः);
        ध्रुव uint32_t signs1 = iq_sign_आवरणम्(sign_द्वयङ्काः >> 4);
        ध्रुव पूर्णाङ्क grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        ध्रुव पूर्णाङ्क grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi2 = dpct::dp4a(grid_l, *((ध्रुव पूर्णाङ्क *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((ध्रुव पूर्णाङ्क *)q8 + 1), sumi2);
        q8 += 8;
    }
    ध्रुव भिन्न d = (भिन्न)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    प्रत्यागम d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#अन्यथा
    प्रतिज्ञा(असत्य);
    प्रत्यागम 0.f;
#यदि_अन्तः
#अन्यथा
    प्रतिज्ञा(असत्य);
    प्रत्यागम 0.f;
#यदि_अन्तः
}

#परिभाषय VDR_IQ2_S_Q8_1_MMVQ 1

स्थिर __dpct_inline__ भिन्न
vec_dot_iq2_s_q8_1(ध्रुव शून्य *__restrict__ vbq,
                   ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {
#यदि QK_K == 256
    ध्रुव खण्डम्_iq2_s * bq2 = (ध्रुव खण्डम्_iq2_s *) vbq;

    ध्रुव पूर्णाङ्क ib32 = iqs;
    ध्रुव int8_t  * q8 = bq8_1[ib32].qs;
    ध्रुव uint8_t * signs = bq2->qs + QK_K/8 + 4*ib32;
    ध्रुव uint8_t ls1 = bq2->scales[ib32] & 0xf;
    ध्रुव uint8_t ls2 = bq2->scales[ib32] >>  4;
    पूर्णाङ्क sumi1 = 0;
    प्रति (पूर्णाङ्क l = 0; l < 2; ++l) {
        ध्रुव uint32_t * grid = (ध्रुव uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        ध्रुव uint32_t signs0 = dpct::vectorized_द्विमान<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ध्रुव uint32_t signs1 = dpct::vectorized_द्विमान<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ध्रुव पूर्णाङ्क grid_l = dpct::vectorized_द्विमान<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        ध्रुव पूर्णाङ्क grid_h = dpct::vectorized_द्विमान<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi1 = dpct::dp4a(grid_l, *((ध्रुव पूर्णाङ्क *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((ध्रुव पूर्णाङ्क *)q8 + 1), sumi1);
        q8 += 8;
    }
    पूर्णाङ्क sumi2 = 0;
    प्रति (पूर्णाङ्क l = 2; l < 4; ++l) {
        ध्रुव uint32_t * grid = (ध्रुव uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        ध्रुव uint32_t signs0 = dpct::vectorized_द्विमान<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ध्रुव uint32_t signs1 = dpct::vectorized_द्विमान<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ध्रुव पूर्णाङ्क grid_l = dpct::vectorized_द्विमान<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        ध्रुव पूर्णाङ्क grid_h = dpct::vectorized_द्विमान<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi2 = dpct::dp4a(grid_l, *((ध्रुव पूर्णाङ्क *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((ध्रुव पूर्णाङ्क *)q8 + 1), sumi2);
        q8 += 8;
    }
    ध्रुव भिन्न d = (भिन्न)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    प्रत्यागम d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#अन्यथा
    प्रतिज्ञा(असत्य);
#यदि_अन्तः
}

#परिभाषय VDR_IQ3_XXS_Q8_1_MMVQ 1

स्थिर __dpct_inline__ भिन्न
vec_dot_iq3_xxs_q8_1(ध्रुव शून्य *__restrict__ vbq,
                     ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs,
                     ध्रुव uint32_t *iq3xxs_grid) {
#यदि DPCT_COMPATIBILITY_अस्थायी >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#यदि QK_K == 256
    ध्रुव खण्डम्_iq3_xxs * bq2 = (ध्रुव खण्डम्_iq3_xxs *) vbq;

    ध्रुव पूर्णाङ्क ib32 = iqs;
    ध्रुव uint8_t  * q3 = bq2->qs + 8*ib32;
    ध्रुव uint16_t * gas = (ध्रुव uint16_t *)(bq2->qs + QK_K/4) + 2*ib32;
    ध्रुव int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = gas[0] | (gas[1] << 16);
    पूर्णाङ्क sumi = 0;
#निर्देशः unroll
    प्रति (पूर्णाङ्क l = 0; l < 4; ++l) {
        ध्रुव uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
        ध्रुव uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
        ध्रुव uint32_t sign_द्वयङ्काः = iq_sign_द्वयङ्काः(aux32);
        ध्रुव uint32_t signs0 = iq_sign_आवरणम्(sign_द्वयङ्काः);
        ध्रुव uint32_t signs1 = iq_sign_आवरणम्(sign_द्वयङ्काः >> 4);
        ध्रुव पूर्णाङ्क grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        ध्रुव पूर्णाङ्क grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((ध्रुव पूर्णाङ्क *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((ध्रुव पूर्णाङ्क *)q8 + 1), sumi);
        q8 += 8;
        aux32 >>= 7;
    }
    ध्रुव भिन्न d = (भिन्न)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.5f;
    प्रत्यागम d * sumi;
#अन्यथा
    प्रतिज्ञा(असत्य);
    प्रत्यागम 0.f;
#यदि_अन्तः
#अन्यथा
    प्रतिज्ञा(असत्य);
    प्रत्यागम 0.f;
#यदि_अन्तः
}

प्रतिरूपम् <> संरचना reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_XXS> {
    स्थिर constexpr ggml_type gtype = GGML_TYPE_IQ3_XXS;

    __dpct_inline__ भिन्न संक्रियकः()(ध्रुव शून्य * __restrict__ vbq, ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> ibx_विस्थापनम्,
                                     ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> d_विस्थापनम्, ध्रुव int8_t * q8_1_quant_ptr,
                                     ध्रुव sycl::half2 * q8_1_ds, ध्रुव पूर्णाङ्क & iqs) {
#यदि DPCT_COMPATIBILITY_अस्थायी >= MIN_CC_DP4A && QK_K == 256
        ध्रुव uint8_t * आधारः = static_cast<ध्रुव uint8_t *>(vbq);
        ध्रुव uint8_t * qs   = आधारः + ibx_विस्थापनम्.first;
        ध्रुव uint8_t * q3   = qs + 8 * iqs;
        ध्रुव uint16_t * gas = reinterpret_cast<ध्रुव uint16_t *>(qs + QK_K / 4) + 2 * iqs;
        ध्रुव int8_t * q8    = q8_1_quant_ptr + iqs * QK8_1;
        uint32_t aux32       = gas[0] | (gas[1] << 16);

        पूर्णाङ्क sumi = 0;
#निर्देशः unroll
        प्रति (पूर्णाङ्क l = 0; l < 4; ++l) {
            ध्रुव uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
            ध्रुव uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
            ध्रुव uint32_t sign_द्वयङ्काः = iq_sign_द्वयङ्काः(aux32);
            ध्रुव uint32_t signs0 = iq_sign_आवरणम्(sign_द्वयङ्काः);
            ध्रुव uint32_t signs1 = iq_sign_आवरणम्(sign_द्वयङ्काः >> 4);
            ध्रुव पूर्णाङ्क grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            ध्रुव पूर्णाङ्क grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((ध्रुव पूर्णाङ्क *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((ध्रुव पूर्णाङ्क *) q8 + 1), sumi);
            q8 += 8;
            aux32 >>= 7;
        }

        ध्रुव ggml_half d = *reinterpret_cast<ध्रुव ggml_half *>(आधारः + d_विस्थापनम्.first);
        प्रत्यागम (भिन्न) d * (0.5f + aux32) * q8_1_ds[iqs][0] * 0.5f * sumi;
#अन्यथा
        GGML_अप्रयुक्तम्(vbq);
        GGML_अप्रयुक्तम्(ibx_विस्थापनम्);
        GGML_अप्रयुक्तम्(d_विस्थापनम्);
        GGML_अप्रयुक्तम्(q8_1_quant_ptr);
        GGML_अप्रयुक्तम्(q8_1_ds);
        GGML_अप्रयुक्तम्(iqs);
        GGML_ABORT("IQ3_XXS reorder vec dot requires dp4a and QK_K == 256");
#यदि_अन्तः
    }
};

#परिभाषय VDR_IQ3_S_Q8_1_MMVQ 1

स्थिर __dpct_inline__ भिन्न
vec_dot_iq3_s_q8_1(ध्रुव शून्य *__restrict__ vbq,
                   ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs,
                   ध्रुव uint32_t *iq3s_grid) {
#यदि QK_K == 256
    ध्रुव खण्डम्_iq3_s * bq2 = (ध्रुव खण्डम्_iq3_s *) vbq;

    ध्रुव पूर्णाङ्क ib32 = iqs;
    ध्रुव uint8_t  * qs = bq2->qs + 8*ib32;
    ध्रुव int8_t   * q8 = bq8_1[ib32].qs;
    पूर्णाङ्क sumi = 0;
    प्रति (पूर्णाङ्क l = 0; l < 4; ++l) {
        ध्रुव uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((bq2->qh[ib32] << (8 - 2*l)) & 256)];
        ध्रुव uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((bq2->qh[ib32] << (7 - 2*l)) & 256)];
        ध्रुव uint32_t signs = bq2->signs[4 * ib32 + l];
        ध्रुव uint32_t signs0 = iq_sign_आवरणम्(signs);
        ध्रुव uint32_t signs1 = iq_sign_आवरणम्(signs >> 4);
        ध्रुव पूर्णाङ्क grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        ध्रुव पूर्णाङ्क grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((ध्रुव पूर्णाङ्क *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((ध्रुव पूर्णाङ्क *)q8 + 1), sumi);
        q8 += 8;
    }
    ध्रुव भिन्न d =
        (भिन्न)bq2->d *
        (1 + 2 * ((bq2->scales[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf)) *
        bq8_1[ib32].ds[0];
    प्रत्यागम d * sumi;
#अन्यथा
    प्रतिज्ञा(असत्य);
#यदि_अन्तः
}

प्रतिरूपम् <> संरचना reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_S> {
    स्थिर constexpr ggml_type gtype = GGML_TYPE_IQ3_S;

    __dpct_inline__ भिन्न संक्रियकः()(ध्रुव शून्य * __restrict__ vbq, ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> ibx_विस्थापनम्,
                                     ध्रुव std::pair<पूर्णाङ्क, पूर्णाङ्क> d_विस्थापनम्, ध्रुव int8_t * q8_1_quant_ptr,
                                     ध्रुव sycl::half2 * q8_1_ds, ध्रुव पूर्णाङ्क & iqs) {
#यदि QK_K == 256
        ध्रुव uint8_t * आधारः     = static_cast<ध्रुव uint8_t *>(vbq);
        ध्रुव uint8_t * qs       = आधारः + ibx_विस्थापनम्.first + 8 * iqs;
        ध्रुव uint8_t * qh       = आधारः + ibx_विस्थापनम्.second;
        ध्रुव uint8_t * signs    = आधारः + d_विस्थापनम्.first + 4 * iqs;
        ध्रुव uint8_t * metadata = आधारः + d_विस्थापनम्.second;
        ध्रुव ggml_half d        = *reinterpret_cast<ध्रुव ggml_half *>(metadata);
        ध्रुव uint8_t * scales   = metadata + परिमाण(ggml_half);
        ध्रुव int8_t * q8        = q8_1_quant_ptr + iqs * QK8_1;

        पूर्णाङ्क sumi = 0;
#निर्देशः unroll
        प्रति (पूर्णाङ्क l = 0; l < 4; ++l) {
            ध्रुव uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((qh[iqs] << (8 - 2*l)) & 256)];
            ध्रुव uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((qh[iqs] << (7 - 2*l)) & 256)];
            ध्रुव uint32_t signs0 = iq_sign_आवरणम्(signs[l]);
            ध्रुव uint32_t signs1 = iq_sign_आवरणम्(signs[l] >> 4);
            ध्रुव पूर्णाङ्क grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            ध्रुव पूर्णाङ्क grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((ध्रुव पूर्णाङ्क *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((ध्रुव पूर्णाङ्क *) q8 + 1), sumi);
            q8 += 8;
        }

        ध्रुव भिन्न scale = 1 + 2 * ((scales[iqs / 2] >> (4 * (iqs % 2))) & 0xf);
        प्रत्यागम (भिन्न) d * scale * q8_1_ds[iqs][0] * sumi;
#अन्यथा
        GGML_अप्रयुक्तम्(vbq);
        GGML_अप्रयुक्तम्(ibx_विस्थापनम्);
        GGML_अप्रयुक्तम्(d_विस्थापनम्);
        GGML_अप्रयुक्तम्(q8_1_quant_ptr);
        GGML_अप्रयुक्तम्(q8_1_ds);
        GGML_अप्रयुक्तम्(iqs);
        GGML_ABORT("IQ3_S reorder vec dot not supported for QK_K != 256");
#यदि_अन्तः
    }
};

#परिभाषय VDR_IQ1_S_Q8_1_MMVQ 1

स्थिर __dpct_inline__ भिन्न
vec_dot_iq1_s_q8_1(ध्रुव शून्य *__restrict__ vbq,
                   ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs,
                   ध्रुव uint32_t *iq1s_grid_gpu) {
#यदि QK_K == 256
    ध्रुव खण्डम्_iq1_s * bq1 = (ध्रुव खण्डम्_iq1_s *) vbq;

    ध्रुव पूर्णाङ्क ib32 = iqs;
    पूर्णाङ्क sumi = 0;
    ध्रुव पूर्णाङ्क * q8 = (ध्रुव पूर्णाङ्क *)bq8_1[ib32].qs;
    प्रति (पूर्णाङ्क l = 0; l < 4; ++l) {
        ध्रुव पूर्णाङ्क * grid = (ध्रुव पूर्णाङ्क *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[ib32] >> 3*l) & 7) << 8)));
        पूर्णाङ्क grid0 = grid[0] & 0x0f0f0f0f;
        पूर्णाङ्क grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi = dpct::dp4a(q8[2 * l + 1], grid1,
                          dpct::dp4a(q8[2 * l + 0], grid0, sumi));
    }

    ध्रुव भिन्न delta = bq1->qh[ib32] & 0x8000 ? -1-IQ1S_DELTA : -1+IQ1S_DELTA;
    ध्रुव भिन्न d1q = (भिन्न)bq1->d * (2*((bq1->qh[ib32] >> 12) & 7) + 1);
    ध्रुव भिन्न d = d1q * bq8_1[ib32].ds[0];
    ध्रुव भिन्न m = d1q * bq8_1[ib32].ds[1];
    प्रत्यागम d * sumi + m * delta;
#अन्यथा
    प्रतिज्ञा(असत्य);
#यदि_अन्तः
}

#परिभाषय VDR_IQ1_M_Q8_1_MMVQ 1

स्थिर __dpct_inline__ भिन्न
vec_dot_iq1_m_q8_1(ध्रुव शून्य *__restrict__ vbq,
                   ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {
#यदि QK_K == 256
    ध्रुव खण्डम्_iq1_m * bq1 = (ध्रुव खण्डम्_iq1_m *) vbq;

    ध्रुव पूर्णाङ्क ib32 = iqs;
    पूर्णाङ्क   sumi[2] = {0, 0};
    भिन्न sumf[2] = {0.f, 0.f};

    ध्रुव पूर्णाङ्क * q8 = (ध्रुव पूर्णाङ्क *)bq8_1[ib32].qs;
    प्रति (पूर्णाङ्क l = 0; l < 4; ++l) {
        ध्रुव पूर्णाङ्क * grid = (ध्रुव पूर्णाङ्क *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 7) << 8)));
        पूर्णाङ्क grid0 = grid[0] & 0x0f0f0f0f;
        पूर्णाङ्क grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi[l / 2] = dpct::dp4a(q8[2 * l + 1], grid1,
                                 dpct::dp4a(q8[2 * l + 0], grid0, sumi[l / 2]));
        ध्रुव भिन्न delta = (bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 0x08 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA;
        ध्रुव पूर्णाङ्क sumy = dpct::dp4a(q8[2 * l + 1], 0x01010101,
                                    dpct::dp4a(q8[2 * l + 0], 0x01010101, 0));
        sumf[l/2] += delta*sumy;
    }

    iq1m_scale_t scale;
    ध्रुव uint16_t * sc = (ध्रुव uint16_t *)bq1->scales;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    ध्रुव भिन्न d = (भिन्न)scale.f16 * bq8_1[ib32].ds[0];
    प्रत्यागम d * ((sumi[0] + sumf[0]) * (2*((sc[ib32/2] >> 6*(ib32%2)) & 0x7) + 1) + (sumi[1] + sumf[1]) * (2*((sc[ib32/2] >> (6*(ib32%2)+3)) & 0x7) + 1));
#अन्यथा
    प्रतिज्ञा(असत्य);
#यदि_अन्तः
}


#परिभाषय VDR_IQ4_NL_Q8_1_MMVQ 2

स्थिर __dpct_inline__ भिन्न
vec_dot_iq4_nl_q8_1(ध्रुव शून्य *__restrict__ vbq,
                    ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {

    ध्रुव खण्डम्_iq4_nl * bq = (ध्रुव खण्डम्_iq4_nl *) vbq;

    ध्रुव uint16_t * q4 = (ध्रुव uint16_t *)bq->qs + 2*iqs;
    ध्रुव int32_t  * q8 = (ध्रुव int32_t  *)bq8_1->qs + iqs;

    ध्रुव uint8_t * मानानि = (ध्रुव uint8_t *)kvalues_iq4nl;

    पूर्णाङ्क v1, v2;
    पूर्णाङ्क sumi1 = 0, sumi2 = 0;
    प्रति (पूर्णाङ्क l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        ध्रुव uint32_t aux = q4[2*l] | (q4[2*l+1] << 16);
        get_int_from_सारणी_16(aux, मानानि, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[l + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[l + 4], sumi2);
    }

    ध्रुव भिन्न d = (भिन्न)bq->d * bq8_1->ds[0];
    प्रत्यागम d * (sumi1 + sumi2);
}


#परिभाषय VDR_IQ4_XS_Q8_1_MMVQ 1

स्थिर __dpct_inline__ भिन्न
vec_dot_iq4_xs_q8_1(ध्रुव शून्य *__restrict__ vbq,
                    ध्रुव खण्डम्_q8_1 *__restrict__ bq8_1, ध्रुव पूर्णाङ्क &iqs) {

#यदि QK_K == 256
    ध्रुव खण्डम्_iq4_xs * bq4 = (ध्रुव खण्डम्_iq4_xs *) vbq;
    ध्रुव uint8_t * मानानि = (ध्रुव uint8_t *)kvalues_iq4nl;

    // iqs is 0...7
    ध्रुव पूर्णाङ्क ib32 = iqs;
    ध्रुव int32_t  * q8 = (ध्रुव पूर्णाङ्क *)bq8_1[ib32].qs;
    ध्रुव uint32_t * q4 = (ध्रुव uint32_t *)bq4->qs + 4*ib32;
    ध्रुव int8_t ls = ((bq4->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((bq4->scales_h >> 2*ib32) & 3) << 4);
    ध्रुव भिन्न d = (भिन्न)bq4->d * (ls - 32) * bq8_1[ib32].ds[0];
    पूर्णाङ्क v1, v2;
    पूर्णाङ्क sumi1 = 0, sumi2 = 0;
    प्रति (पूर्णाङ्क j = 0; j < 4; ++j) {
        get_int_from_सारणी_16(q4[j], मानानि, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[j + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[j + 4], sumi2);
    }
    प्रत्यागम d * (sumi1 + sumi2);
#अन्यथा
    प्रतिज्ञा(असत्य);
#यदि_अन्तः
}

#यदि_अन्तः // GGML_SYCL_VECDOTQ_HPP
