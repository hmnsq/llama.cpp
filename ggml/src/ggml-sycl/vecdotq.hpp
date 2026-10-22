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

#அந்தர்பாவ "dpct/helper.hpp"
#அந்தர்பாவ "ggml.h"
#அந்தர்பாவ "type.hpp"
#அந்தர்பாவ "quants.hpp"

வகைவரை தசம (*vec_dot_q_sycl_t)(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி block_q8_1 * __restrict__ bq8_1,
                                  மாறிலி முழு & iqs);

நிலையான __dpct_inline__ முழு get_int_b1(மாறிலி வெறுமை * x, மாறிலி முழு & i32) {
    மாறிலி uint8_t * x8 = (மாறிலி uint8_t *) x;

    முழு x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    திரும்பு x32;
}

நிலையான __dpct_inline__ முழு get_int_b2(மாறிலி வெறுமை * x, மாறிலி முழு & i32) {
    மாறிலி uint16_t * x16 = (மாறிலி uint16_t *) x; // assume at least 2 byte alignment

    முழு x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    திரும்பு x32;
}

நிலையான __dpct_inline__ முழு get_int_b4(மாறிலி வெறுமை * x, மாறிலி முழு & i32) {
    திரும்பு ((மாறிலி முழு *) x)[i32]; // assume at least 4 byte alignment
}

நிலையான __dpct_inline__ முழு get_int_from_int8(மாறிலி int8_t* x8, மாறிலி முழு& i32) {
  மாறிலி uint16_t* x16 =
      (மாறிலி uint16_t*)(x8 + அளவு(முழு) * i32); // assume at least 2 byte
                                                 // alignment

  முழு x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  திரும்பு x32;
}

நிலையான __dpct_inline__ முழு get_int_from_uint8(
    மாறிலி uint8_t* x8,
    மாறிலி முழு& i32) {
  மாறிலி uint16_t* x16 =
      (மாறிலி uint16_t*)(x8 + அளவு(முழு) * i32); // assume at least 2 byte
                                                 // alignment

  முழு x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  திரும்பு x32;
}

நிலையான __dpct_inline__ முழு get_int_from_int8_aligned(
    மாறிலி int8_t* x8,
    மாறிலி முழு& i32) {
  திரும்பு *(
      (மாறிலி முழு*)(x8 + அளவு(முழு) * i32)); // assume at least 4 byte alignment
}

நிலையான __dpct_inline__ முழு get_int_from_uint8_aligned(
    மாறிலி uint8_t* x8,
    மாறிலி முழு& i32) {
  திரும்பு *(
      (மாறிலி முழு*)(x8 + அளவு(முழு) * i32)); // assume at least 4 byte alignment
}

நிலையான __dpct_inline__ முழு byte_sub_4(மாறிலி முழு a, மாறிலி முழு b) {
  மாறிலி uint32_t ua = static_cast<uint32_t>(a);
  மாறிலி uint32_t ub = static_cast<uint32_t>(b);
  திரும்பு static_cast<முழு>(((ua | 0x80808080u) - ub) ^ 0x80808080u);
}

நிலையான __dpct_inline__ தசம vec_dot_q6_K_q8_1_impl_mmvq_scalar(
    மாறிலி முழு vl, மாறிலி முழு vh, மாறிலி முழு u0, மாறிலி முழு u1, மாறிலி int8_t sc0,
    மாறிலி int8_t sc1, மாறிலி தசம d, மாறிலி தசம d80, மாறிலி தசம d81) {
    static_assert(QR6_K == 2, "q6_K MMVQ scalar fast path assumes QR6_K == 2");

    மாறிலி முழு vil0 = (vl >> 0) & 0x0F0F0F0F;
    மாறிலி முழு vih0 = ((vh >> 0) << 4) & 0x30303030;
    மாறிலி முழு vi0 = byte_sub_4(vil0 | vih0, 0x20202020);

    மாறிலி முழு vil1 = (vl >> 4) & 0x0F0F0F0F;
    மாறிலி முழு vih1 = ((vh >> 4) << 4) & 0x30303030;
    மாறிலி முழு vi1 = byte_sub_4(vil1 | vih1, 0x20202020);

    மாறிலி தசம sumf =
        d80 * (dpct::dp4a(vi0, u0, 0) * sc0) +
        d81 * (dpct::dp4a(vi1, u1, 0) * sc1);

    திரும்பு d * sumf;
}

நிலையான __dpct_inline__ வெறுமை get_int_from_table_16(மாறிலி uint32_t &q4,
                                                  மாறிலி uint8_t *மதிப்புகள்,
                                                  முழு &val1, முழு &val2) {

    uint32_t aux32; மாறிலி uint8_t * q8 = (மாறிலி uint8_t *)&aux32;
    aux32 = q4 & 0x0f0f0f0f;
    uint16_t v1 = மதிப்புகள்[q8[0]] | (மதிப்புகள்[q8[1]] << 8);
    uint16_t v2 = மதிப்புகள்[q8[2]] | (மதிப்புகள்[q8[3]] << 8);
    val1 = v1 | (v2 << 16);
    aux32 = (q4 >> 4) & 0x0f0f0f0f;
    v1 = மதிப்புகள்[q8[0]] | (மதிப்புகள்[q8[1]] << 8);
    v2 = மதிப்புகள்[q8[2]] | (மதிப்புகள்[q8[3]] << 8);
    val2 = v1 | (v2 << 16);
}

நிலையான __dpct_inline__ sycl::int2 get_int_from_table_16(
    மாறிலி முழு& q4, மாறிலி int8_t* table) {
  மாறிலி uint32_t* table32 = (மாறிலி uint32_t*)table;
  uint32_t tmp[2];
  மாறிலி uint32_t கீழ்_உயர்வு_selection_indices =
      (0x32103210 | ((q4 & 0x88888888) >> 1));
#pragma unroll
  க்கு (uint32_t i = 0; i < 2; ++i) {
    மாறிலி uint32_t shift = 16 * i;

    மாறிலி uint32_t கீழ் =
        dpct::byte_level_permute(table32[0], table32[1], q4 >> shift);
    மாறிலி uint32_t உயர்வு =
        dpct::byte_level_permute(table32[2], table32[3], q4 >> shift);
    tmp[i] = dpct::byte_level_permute(
        கீழ், உயர்வு, கீழ்_உயர்வு_selection_indices >> shift);
  }
  திரும்பு sycl::int2(
      dpct::byte_level_permute(tmp[0], tmp[1], 0x6420),
      dpct::byte_level_permute(tmp[0], tmp[1], 0x7531));
}

// Four E2M1 codes (one per byte, bits 0..3) to their kvalues_mxfp4 int8 values. SWAR arithmetic
// replaces get_int_from_table_16 for MXFP4: dpct::byte_level_permute is emulated with 64-bit shifts,
// eight per int, which made the MXFP4 GEMV compute-bound on Intel GPUs.
// Magnitudes 0,1,2,3,4,6,8,12 = m + [m>=5] + [m>=6] + 3*[m>=7]; each byte stays below 256, so the
// byte-wise adds never carry. -0 (code 8) is left as 0 so the two's-complement +1 cannot carry either.
நிலையான __dpct_inline__ முழு mxfp4_codes_to_int8(மாறிலி uint32_t x) {
    மாறிலி uint32_t m   = x & 0x07070707u;
    மாறிலி uint32_t ge5 = ((m + 0x03030303u) >> 3) & 0x01010101u;
    மாறிலி uint32_t ge6 = ((m + 0x02020202u) >> 3) & 0x01010101u;
    மாறிலி uint32_t ge7 = ((m + 0x01010101u) >> 3) & 0x01010101u;
    மாறிலி uint32_t mag = m + ge5 + ge6 + 3u * ge7;
    மாறிலி uint32_t nz  = ((mag + 0x7f7f7f7fu) >> 7) & 0x01010101u;
    மாறிலி uint32_t neg = (x >> 3) & nz & 0x01010101u;
    திரும்பு (முழு) ((mag ^ (neg * 0xffu)) + neg);
}

// Same result as get_int_from_table_16(q4, kvalues_mxfp4): x = low nibbles, y = high nibbles.
நிலையான __dpct_inline__ sycl::int2 get_int_from_mxfp4(மாறிலி முழு q4) {
    திரும்பு sycl::int2(mxfp4_codes_to_int8((uint32_t) q4 & 0x0f0f0f0fu),
                      mxfp4_codes_to_int8(((uint32_t) q4 >> 4) & 0x0f0f0f0fu));
}

#define VDR_Q2_K_Q8_1_MMVQ 1

// contiguous v/x values
நிலையான __dpct_inline__ தசம vec_dot_q2_K_q8_1_impl_mmvq(
    மாறிலி முழு &v, மாறிலி முழு *__restrict__ u, மாறிலி uint8_t *__restrict__ scales,
    மாறிலி sycl::half2 &dm2, மாறிலி தசம *__restrict__ d8) {

    தசம sumf_d = 0.0f;
    தசம sumf_m = 0.0f;

#pragma unroll
    க்கு (முழு i = 0; i < QR2_K; ++i) {
        மாறிலி முழு sc = scales[2*i];

        மாறிலி முழு vi = (v >> (2*i)) & 0x03030303;

        sumf_d +=
            d8[i] * (dpct::dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        முழு m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] *
                  dpct::dp4a(
                      m, u[i],
                      0); // multiply constant q2_K part with sum of q8_1 values
    }

    மாறிலி sycl::float2 dm2f =
        dm2.convert<தசம, sycl::rounding_mode::automatic>();

    திரும்பு dm2f.x() * sumf_d - dm2f.y() * sumf_m;
}


#define VDR_Q3_K_Q8_1_MMVQ 1

// contiguous v/x values
நிலையான __dpct_inline__ தசம vec_dot_q3_K_q8_1_impl_mmvq(
    மாறிலி முழு &vl, மாறிலி முழு &vh, மாறிலி முழு *__restrict__ u,
    மாறிலி uint8_t *__restrict__ scales, மாறிலி முழு &scale_offset,
    மாறிலி தசம &d3, மாறிலி தசம *__restrict__ d8) {

    தசம sumf = 0.0f;

#pragma unroll
    க்கு (முழு i = 0; i < QR3_K; ++i) {
        மாறிலி முழு isc = scale_offset + 2*i;

        மாறிலி முழு isc_கீழ் = isc % (QK_K/32);
        மாறிலி முழு sc_shift_கீழ் = 4 * (isc / (QK_K/32));
        மாறிலி முழு sc_கீழ்  = (scales[isc_கீழ்] >> sc_shift_கீழ்) & 0xF;

        மாறிலி முழு isc_உயர்வு = isc % (QK_K/64);
        மாறிலி முழு sc_shift_உயர்வு = 2 * (isc / (QK_K/64));
        மாறிலி முழு sc_உயர்வு = ((scales[(QK_K/32) + isc_உயர்வு] >> sc_shift_உயர்வு) & 3) << 4;

        மாறிலி முழு sc = (sc_கீழ் | sc_உயர்வு) - 32;

        மாறிலி முழு vil = (vl >> (2*i)) & 0x03030303;

        மாறிலி முழு vih = ((vh >> i) << 2) & 0x04040404;

        மாறிலி முழு vi =
            dpct::vectorized_இருமம்<sycl::char4>(vil, vih, dpct::sub_sat());

        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    திரும்பு d3 * sumf;
}

#define VDR_Q4_K_Q8_1_MMVQ 2

// contiguous v/x values
நிலையான __dpct_inline__ தசம vec_dot_q4_K_q8_1_impl_vmmq(
    மாறிலி முழு *__restrict__ v, மாறிலி முழு *__restrict__ u,
    மாறிலி uint8_t *__restrict__ sc, மாறிலி uint8_t *__restrict__ m,
    மாறிலி sycl::half2 &dm4, மாறிலி தசம *__restrict__ d8) {

    தசம sumf_d = 0.0f;
    தசம sumf_m = 0.0f;

#pragma unroll
    க்கு (முழு i = 0; i < QR4_K; ++i) {
        மாறிலி முழு v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        மாறிலி முழு v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        மாறிலி முழு dot1 =
            dpct::dp4a(v1i, u[2 * i + 1],
                       dpct::dp4a(v0i, u[2 * i + 0], 0)); // SIMD dot product
        மாறிலி முழு dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 1],
                       dpct::dp4a(0x01010101, u[2 * i + 0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    மாறிலி sycl::float2 dm4f =
        dm4.convert<தசம, sycl::rounding_mode::automatic>();

    திரும்பு dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


#define VDR_Q5_K_Q8_1_MMVQ 2

// contiguous v/x values
நிலையான __dpct_inline__ தசம vec_dot_q5_K_q8_1_impl_vmmq(
    மாறிலி முழு *__restrict__ vl, மாறிலி முழு *__restrict__ vh,
    மாறிலி முழு *__restrict__ u, மாறிலி uint8_t *__restrict__ sc,
    மாறிலி uint8_t *__restrict__ m, மாறிலி sycl::half2 &dm5,
    மாறிலி தசம *__restrict__ d8) {

    தசம sumf_d = 0.0f;
    தசம sumf_m = 0.0f;

#pragma unroll
    க்கு (முழு i = 0; i < QR5_K; ++i) {
        மாறிலி முழு vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        மாறிலி முழு vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        மாறிலி முழு vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        மாறிலி முழு vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        மாறிலி முழு v0i = vl0i | vh0i;
        மாறிலி முழு v1i = vl1i | vh1i;

        மாறிலி முழு dot1 =
            dpct::dp4a(v0i, u[2 * i + 0],
                       dpct::dp4a(v1i, u[2 * i + 1], 0)); // SIMD dot product
        மாறிலி முழு dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 0],
                       dpct::dp4a(0x01010101, u[2 * i + 1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    மாறிலி sycl::float2 dm5f =
        dm5.convert<தசம, sycl::rounding_mode::automatic>();

    திரும்பு dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}


#define VDR_Q6_K_Q8_1_MMVQ 1

// contiguous v/x values
நிலையான __dpct_inline__ தசம
vec_dot_q6_K_q8_1_impl_mmvq(மாறிலி முழு &vl, மாறிலி முழு &vh,
                            மாறிலி முழு *__restrict__ u,
                            மாறிலி int8_t *__restrict__ scales, மாறிலி தசம &d,
                            மாறிலி தசம *__restrict__ d8) {
    திரும்பு vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
}

#define VDR_Q1_0_Q8_1_MMVQ 1
#define VDR_Q1_0_Q8_1_MMQ  4

நிலையான __dpct_inline__ தசம
vec_dot_q1_0_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                  மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {

    மாறிலி block_q1_0 * bq1_0 = (மாறிலி block_q1_0 *) vbq;

    மாறிலி block_q8_1 * bq8_1_chunk = bq8_1 + iqs;
    மாறிலி தசம        d1          = bq1_0->d;
    மாறிலி முழு          v           = get_int_from_uint8_aligned(bq1_0->qs, iqs);

    முழு vi_bytes[8];
#pragma unroll
    க்கு (முழு j = 0; j < 8; ++j) {
        மாறிலி முழு shift = j * 4;
        மாறிலி முழு bits4 = (v >> shift) & 0x0F;
        மாறிலி முழு b0    = (bits4 & 0x01) ? 1 : -1;
        மாறிலி முழு b1    = (bits4 & 0x02) ? 1 : -1;
        மாறிலி முழு b2    = (bits4 & 0x04) ? 1 : -1;
        மாறிலி முழு b3    = (bits4 & 0x08) ? 1 : -1;
        vi_bytes[j]     = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }

    முழு sumi = 0;
#pragma unroll
    க்கு (முழு j = 0; j < 8; ++j) {
        மாறிலி முழு u = get_int_from_int8_aligned(bq8_1_chunk->qs, j);
        sumi        = ggml_sycl_dp4a(vi_bytes[j], u, sumi);
    }

    திரும்பு d1 * bq8_1_chunk->ds[0] * sumi;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

template <ggml_type T> கட்டமைப்பு reorder_vec_dot_q_sycl {
    static_assert(T != T, "ggml_type for reorder vecdot not implemented");
};

// For some types the weight side of the dot product does not depend on the destination column, so a
// multi-column mul_mat_vec can unpack it once per block instead of once per column. Such a type adds
// load() and dot() next to operator() and opts in here. See reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>.
template <ggml_type T> கட்டமைப்பு reorder_vec_dot_shared_weights {
    நிலையான constexpr தருக்க value = பொய்;
};

template <> கட்டமைப்பு reorder_vec_dot_shared_weights<GGML_TYPE_Q4_K> {
    நிலையான constexpr தருக்க value = உண்மை;
};

template <> கட்டமைப்பு reorder_vec_dot_shared_weights<GGML_TYPE_Q5_K> {
    நிலையான constexpr தருக்க value = உண்மை;
};

template <ggml_type T> கட்டமைப்பு reorder_vec_dot_shared_activations {
    நிலையான constexpr தருக்க value = பொய்;
};

template <> கட்டமைப்பு reorder_vec_dot_shared_activations<GGML_TYPE_Q4_K> {
    நிலையான constexpr தருக்க value = உண்மை;
};

template <> கட்டமைப்பு reorder_vec_dot_shared_activations<GGML_TYPE_Q5_K> {
    நிலையான constexpr தருக்க value = உண்மை;
};

template <> கட்டமைப்பு reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0> {
    நிலையான constexpr ggml_type gtype = GGML_TYPE_Q4_0;

    using q4_0_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_0>;
    using q4_0_traits = typename q4_0_block::traits;

    __dpct_inline__ தசம vec_dot_q4_0_q8_1_impl(மாறிலி முழு * v, மாறிலி முழு * u, மாறிலி தசம & d4, மாறிலி sycl::half2 & ds8) {
        முழு sumi = 0;

#pragma unroll
        க்கு (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            மாறிலி முழு vi0 = (v[i] >> 0) & 0x0F0F0F0F;
            மாறிலி முழு vi1 = (v[i] >> 4) & 0x0F0F0F0F;

            // SIMD dot product of quantized values
            sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
            sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
        }

        மாறிலி sycl::float2 ds8f = ds8.convert<தசம, sycl::rounding_mode::automatic>();

        // second part effectively subtracts 8 from each quant value
        திரும்பு d4 * (sumi * ds8f.x() - (8 * q4_0_traits::vdr_mmvq / q4_0_traits::qi) * ds8f.y());
    }

    __dpct_inline__ தசம operator()(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி std::pair<முழு, முழு> ibx_offset,
                                     மாறிலி std::pair<முழு, முழு> d_offset, மாறிலி int8_t * q8_1_quant_ptr,
                                     மாறிலி sycl::half2 * q8_1_ds, மாறிலி முழு & iqs) {
        மாறிலி uint8_t * bq4_0 = static_cast<மாறிலி uint8_t *>(vbq) + ibx_offset.first;
        மாறிலி ggml_half d = *(reinterpret_cast<மாறிலி ggml_half *>(static_cast<மாறிலி uint8_t *>(vbq) + d_offset.first));
        முழு             v[q4_0_traits::vdr_mmvq];
        முழு             u[2 * q4_0_traits::vdr_mmvq];


#pragma unroll
        க்கு (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            v[i]         = get_int_from_uint8(bq4_0, iqs + i);
            u[2 * i + 0] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
            u[2 * i + 1] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i + q4_0_traits::qi);
        }

        திரும்பு vec_dot_q4_0_q8_1_impl(v, u, d, *q8_1_ds);
    }
};

// Load four contiguous dwords per operand instead of loading each value separately.
கட்டமைப்பு reorder_vec_dot_q8_0_wide {
    நிலையான constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    using q8_0_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_0>;
    using q8_0_traits = typename q8_0_block::traits;

    __dpct_inline__ தசம operator()(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி std::pair<முழு, முழு> ibx_offset,
                                     மாறிலி std::pair<முழு, முழு> d_offset, மாறிலி int8_t * q8_1_quant_ptr,
                                     மாறிலி sycl::half2 * q8_1_ds, மாறிலி முழு & iqs) {
        static_assert(q8_0_traits::vdr_mmvq == 4, "the wide load moves exactly four dwords");

        மாறிலி uint8_t * base = static_cast<மாறிலி uint8_t *>(vbq);
        மாறிலி int8_t *  qs   = reinterpret_cast<மாறிலி int8_t *>(base + ibx_offset.first);
        மாறிலி ggml_half d    = *reinterpret_cast<மாறிலி ggml_half *>(base + d_offset.first);

        மாறிலி sycl::int4 v = *reinterpret_cast<மாறிலி sycl::int4 *>(qs + அளவு(முழு) * iqs);
        மாறிலி sycl::int4 u = *reinterpret_cast<மாறிலி sycl::int4 *>(q8_1_quant_ptr + அளவு(முழு) * iqs);

        முழு sumi = 0;
#pragma unroll
        க்கு (முழு i = 0; i < 4; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        மாறிலி sycl::half2 ds_மதிப்புகள் = *q8_1_ds;
        திரும்பு static_cast<தசம>(d) * static_cast<தசம>(ds_மதிப்புகள்[0]) * sumi;
    }
};

template <> கட்டமைப்பு reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0> {
    நிலையான constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    using q8_0_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q8_0>;
    using q8_0_traits = typename q8_0_block::traits;

    __dpct_inline__ தசம operator()(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி std::pair<முழு, முழு> ibx_offset,
                                     மாறிலி std::pair<முழு, முழு> d_offset, மாறிலி int8_t * q8_1_quant_ptr,
                                     மாறிலி sycl::half2 * q8_1_ds, மாறிலி முழு & iqs,
                                     மாறிலி முழு vdr = q8_0_traits::vdr_mmvq) {
        மாறிலி uint8_t * base = static_cast<மாறிலி uint8_t *>(vbq);
        மாறிலி int8_t *  qs   = reinterpret_cast<மாறிலி int8_t *>(base + ibx_offset.first);
        மாறிலி ggml_half  d   = *reinterpret_cast<மாறிலி ggml_half *>(base + d_offset.first);

        முழு v[q8_0_traits::vdr_mmvq];
        முழு u[q8_0_traits::vdr_mmvq];

#pragma unroll
        க்கு (முழு i = 0; i < vdr; ++i) {
            v[i] = get_int_from_int8(qs, iqs + i);
            u[i] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
        }

        முழு sumi = 0;
#pragma unroll
        க்கு (முழு i = 0; i < vdr; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        மாறிலி sycl::half2 ds_மதிப்புகள் = *q8_1_ds;
        திரும்பு static_cast<தசம>(d) * static_cast<தசம>(ds_மதிப்புகள்[0]) * sumi;
    }
};

template <> கட்டமைப்பு reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K> {
    நிலையான constexpr ggml_type gtype = GGML_TYPE_Q2_K;

    using q2_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q2_K>;
    using q2_k_traits = typename q2_k_block::traits;

    __dpct_inline__ தசம operator()(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி std::pair<முழு, முழு> ibx_offset,
                                     மாறிலி std::pair<முழு, முழு> d_offset, மாறிலி int8_t * q8_1_quant_ptr,
                                     மாறிலி sycl::half2 * q8_1_ds, மாறிலி முழு & iqs) {
        மாறிலி uint8_t *    base   = static_cast<மாறிலி uint8_t *>(vbq);
        மாறிலி uint8_t *    qs     = base + ibx_offset.first;
        மாறிலி uint8_t *    scales = base + d_offset.first;
        மாறிலி ggml_half2 * dm     = reinterpret_cast<மாறிலி ggml_half2 *>(base + d_offset.second);

        மாறிலி முழு bq8_offset   = QR2_K * (iqs / QI8_1);
        மாறிலி முழு scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        மாறிலி முழு v = get_int_from_uint8_aligned(qs, iqs);

        முழு   u[QR2_K];
        தசம d8[QR2_K];

#pragma unroll
        க்கு (முழு i = 0; i < QR2_K; ++i) {
            மாறிலி int8_t * quant_base_ptr = q8_1_quant_ptr + (bq8_offset + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_base_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_offset + i))[0];
        }

        திரும்பு vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales + scale_offset, *dm, d8);
    }
};

template <> கட்டமைப்பு reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K> {
    நிலையான constexpr ggml_type gtype = GGML_TYPE_Q3_K;

    using q3_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q3_K>;
    using q3_k_traits = typename q3_k_block::traits;

    __dpct_inline__ தசம operator()(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி std::pair<முழு, முழு> ibx_offset,
                                     மாறிலி std::pair<முழு, முழு> d_offset, மாறிலி int8_t * q8_1_quant_ptr,
                                     மாறிலி sycl::half2 * q8_1_ds, மாறிலி முழு & iqs) {
        மாறிலி uint8_t *  base   = static_cast<மாறிலி uint8_t *>(vbq);
        மாறிலி uint8_t *  qs     = base + ibx_offset.first;
        மாறிலி uint8_t *  hmask  = base + ibx_offset.second;
        மாறிலி uint8_t *  scales = base + d_offset.first;
        மாறிலி ggml_half  d      = *reinterpret_cast<மாறிலி ggml_half *>(base + d_offset.second);

        மாறிலி முழு bq8_offset   = QR3_K * (iqs / (QI3_K / 2));
        மாறிலி முழு scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        மாறிலி முழு vl = get_int_from_uint8(qs, iqs);
        மாறிலி முழு vh = ~get_int_from_uint8(hmask, iqs % (QI3_K / 2)) >> bq8_offset;

        முழு   u[QR3_K];
        தசம d8[QR3_K];

#pragma unroll
        க்கு (முழு i = 0; i < QR3_K; ++i) {
            மாறிலி int8_t * quant_base_ptr = q8_1_quant_ptr + (bq8_offset + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_base_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_offset + i))[0];
        }

        திரும்பு vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, scales, scale_offset, static_cast<தசம>(d), d8);
    }
};

நிலையான உள்வரி தசம vec_dot_q4_K_q8_1_common(மாறிலி முழு * __restrict__ q4, மாறிலி uint16_t * __restrict__ scales,
                                             மாறிலி ggml_half2 & dm, மாறிலி block_q8_1 * __restrict__ bq8_1,
                                             மாறிலி முழு &        iqs) {
    முழு   v[2];
    முழு   u[2 * QR4_K];
    தசம d8[QR4_K];

    v[0] = q4[0];
    v[1] = q4[4];

    uint16_t  aux[2];
    மாறிலி முழு j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
    எனில் (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } இல்லையெனில் {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }

    மாறிலி uint8_t * sc = (மாறிலி uint8_t *) aux;
    மாறிலி uint8_t * m  = sc + 2;

    மாறிலி முழு bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));

    க்கு (முழு i = 0; i < QR4_K; ++i) {
        மாறிலி block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i]                   = bq8i->ds[0];

        மாறிலி முழு * q8 = (மாறிலி முழு *) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0]   = q8[0];
        u[2 * i + 1]   = q8[4];
    }

    திரும்பு vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
}

template <> கட்டமைப்பு reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K> {
    நிலையான constexpr ggml_type gtype = GGML_TYPE_Q4_K;

    using q4_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q4_K>;
    using q4_k_traits = typename q4_k_block::traits;

    கட்டமைப்பு weights {
        முழு        v[2];
        uint16_t   aux[2];
        ggml_half2 dm;
        முழு        bq8_offset;
    };

    கட்டமைப்பு activations {
        முழு   u[2 * QR4_K];
        தசம d8[QR4_K];
    };

    __dpct_inline__ நிலையான weights load(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி std::pair<முழு, முழு> ibx_offset,
                                        மாறிலி std::pair<முழு, முழு> d_offset, மாறிலி முழு & iqs) {
        மாறிலி uint8_t *    base = static_cast<மாறிலி uint8_t *>(vbq);
        மாறிலி uint8_t *    qs   = base + ibx_offset.first;
        மாறிலி uint8_t *    scs  = base + d_offset.first;
        மாறிலி ggml_half2 * dms  = reinterpret_cast<மாறிலி ggml_half2 *>(base + d_offset.second);

        weights w;
        w.bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));

        மாறிலி முழு *      q4     = (மாறிலி முழு *) (qs + 16 * w.bq8_offset + 4 * ((iqs / 2) % 4));
        மாறிலி uint16_t * scales = (மாறிலி uint16_t *) scs;

        w.v[0] = q4[0];
        w.v[1] = q4[4];

        மாறிலி முழு j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        எனில் (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } இல்லையெனில் {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        திரும்பு w;
    }

    __dpct_inline__ நிலையான activations load_activations(மாறிலி int8_t * q8_1_quant_ptr,
                                                        மாறிலி sycl::half2 * q8_1_ds, மாறிலி முழு & iqs) {
        activations a;
        மாறிலி முழு bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        க்கு (முழு i = 0; i < QR4_K; ++i) {
            மாறிலி int8_t * quant_base_ptr = q8_1_quant_ptr + (bq8_offset + i) * QK8_1;
            sycl::half2    ds_மதிப்புகள்      = *(q8_1_ds + bq8_offset + i);

            a.d8[i] = ds_மதிப்புகள்[0];

            மாறிலி முழு * q8 = (மாறிலி முழு *) quant_base_ptr + ((iqs / 2) % 4);
            a.u[2 * i + 0] = q8[0];
            a.u[2 * i + 1] = q8[4];
        }

        திரும்பு a;
    }

    __dpct_inline__ நிலையான தசம apply(மாறிலி weights & w, மாறிலி activations & a) {
        மாறிலி uint8_t * sc = (மாறிலி uint8_t *) w.aux;
        மாறிலி uint8_t * m  = sc + 2;

        திரும்பு vec_dot_q4_K_q8_1_impl_vmmq(w.v, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ நிலையான தசம dot(மாறிலி weights & w, மாறிலி int8_t * q8_1_quant_ptr,
                                     மாறிலி sycl::half2 * q8_1_ds, மாறிலி முழு & iqs) {
        மாறிலி auto a = load_activations(q8_1_quant_ptr, q8_1_ds, iqs);

        திரும்பு apply(w, a);
    }

    __dpct_inline__ தசம operator()(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி std::pair<முழு, முழு> ibx_offset,
                                     மாறிலி std::pair<முழு, முழு> d_offset, மாறிலி int8_t * q8_1_quant_ptr,
                                     மாறிலி sycl::half2 * q8_1_ds, மாறிலி முழு & iqs) {
        திரும்பு dot(load(vbq, ibx_offset, d_offset, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

template <> கட்டமைப்பு reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K> {
    நிலையான constexpr ggml_type gtype = GGML_TYPE_Q5_K;

    using q5_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q5_K>;
    using q5_k_traits = typename q5_k_block::traits;

    கட்டமைப்பு weights {
        முழு        vl[2];
        முழு        vh[2];
        uint16_t   aux[2];
        ggml_half2 dm;
    };

    // same activation layout as Q4_K
    static_assert(QR5_K == QR4_K);
    using activations = reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::activations;

    __dpct_inline__ நிலையான weights load(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி std::pair<முழு, முழு> ibx_offset,
                                        மாறிலி std::pair<முழு, முழு> d_offset, மாறிலி முழு & iqs) {
        மாறிலி uint8_t *    base    = static_cast<மாறிலி uint8_t *>(vbq);
        மாறிலி uint8_t *    qs      = base + ibx_offset.first;   // low 4 bits
        மாறிலி uint8_t *    qh_base = base + ibx_offset.second;  // high bit
        மாறிலி uint8_t *    scs     = base + d_offset.first;
        மாறிலி ggml_half2 * dms     = reinterpret_cast<மாறிலி ggml_half2 *>(base + d_offset.second);

        மாறிலி முழு        bq8_offset = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        மாறிலி முழு *      ql_ptr     = (மாறிலி முழு *) (qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
        மாறிலி முழு *      qh_ptr     = (மாறிலி முழு *) (qh_base + 4 * ((iqs / 2) % 4));
        மாறிலி uint16_t * scales     = (மாறிலி uint16_t *) scs;

        weights w;
        w.vl[0] = ql_ptr[0];
        w.vl[1] = ql_ptr[4];

        w.vh[0] = qh_ptr[0] >> bq8_offset;
        w.vh[1] = qh_ptr[4] >> bq8_offset;

        மாறிலி முழு j = (QR5_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        எனில் (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } இல்லையெனில் {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        திரும்பு w;
    }

    __dpct_inline__ நிலையான activations load_activations(மாறிலி int8_t * q8_1_quant_ptr,
                                                        மாறிலி sycl::half2 * q8_1_ds, மாறிலி முழு & iqs) {
        திரும்பு reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::load_activations(q8_1_quant_ptr, q8_1_ds, iqs);
    }

    __dpct_inline__ நிலையான தசம apply(மாறிலி weights & w, மாறிலி activations & a) {
        மாறிலி uint8_t * sc = (மாறிலி uint8_t *) w.aux;
        மாறிலி uint8_t * m  = sc + 2;

        திரும்பு vec_dot_q5_K_q8_1_impl_vmmq(w.vl, w.vh, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ நிலையான தசம dot(மாறிலி weights & w, மாறிலி int8_t * q8_1_quant_ptr,
                                     மாறிலி sycl::half2 * q8_1_ds, மாறிலி முழு & iqs) {
        திரும்பு apply(w, load_activations(q8_1_quant_ptr, q8_1_ds, iqs));
    }

    __dpct_inline__ தசம operator()(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி std::pair<முழு, முழு> ibx_offset,
                                     மாறிலி std::pair<முழு, முழு> d_offset, மாறிலி int8_t * q8_1_quant_ptr,
                                     மாறிலி sycl::half2 * q8_1_ds, மாறிலி முழு & iqs) {
        திரும்பு dot(load(vbq, ibx_offset, d_offset, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

template <> கட்டமைப்பு reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K> {
    நிலையான constexpr ggml_type gtype = GGML_TYPE_Q6_K;

    using q6_k_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_Q6_K>;
    using q6_k_traits = typename q6_k_block::traits;

    __dpct_inline__ தசம vec_dot_q6_K_q8_1_impl_mmvq(மாறிலி முழு vl, மாறிலி முழு vh, மாறிலி முழு * __restrict__ u,
                                                      மாறிலி int8_t * __restrict__ scales, மாறிலி தசம d,
                                                      மாறிலி தசம * __restrict__ d8) {
        திரும்பு vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
    }

    __dpct_inline__ தசம operator()(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி std::pair<முழு, முழு> ibx_offset,
                     மாறிலி std::pair<முழு, முழு> d_offset, மாறிலி int8_t * q8_1_quant_ptr, மாறிலி sycl::half2 * q8_1_ds,
                     மாறிலி முழு iqs) {
        மாறிலி uint8_t *   base   = static_cast<மாறிலி uint8_t *>(vbq);
        மாறிலி uint8_t *   ql     = base + ibx_offset.first;
        மாறிலி uint8_t *   qh     = base + ibx_offset.second;
        மாறிலி int8_t *    scales = reinterpret_cast<மாறிலி int8_t *>(base + d_offset.first);
        மாறிலி ggml_half * d      = (மாறிலி ggml_half *) (base + d_offset.second);

        மாறிலி முழு bq8_offset   = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        மாறிலி முழு scale_offset = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        மாறிலி முழு vh_shift     = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

        மாறிலி முழு vl = get_int_from_uint8(ql, iqs);
        மாறிலி முழு vh = get_int_from_uint8(qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_shift;

        மாறிலி int8_t * scs = scales + scale_offset;

        மாறிலி முழு u0 = get_int_from_int8_aligned(
            q8_1_quant_ptr + bq8_offset * QK8_1, iqs % QI8_1);
        மாறிலி முழு u1 = get_int_from_int8_aligned(
            q8_1_quant_ptr + (bq8_offset + 2) * QK8_1, iqs % QI8_1);
        மாறிலி தசம d80 = (*(q8_1_ds + bq8_offset + 0))[0];
        மாறிலி தசம d81 = (*(q8_1_ds + bq8_offset + 2))[0];

        திரும்பு vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u0, u1, scs[0], scs[4], *d, d80, d81);
    }
};

template <> கட்டமைப்பு reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4> {
    நிலையான constexpr ggml_type gtype = GGML_TYPE_MXFP4;

    using mxfp4_block  = ggml_sycl_reordered::block_q_t<GGML_TYPE_MXFP4>;
    using mxfp4_traits = typename mxfp4_block::traits;

    __dpct_inline__ தசம operator()(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி std::pair<முழு, முழு> ibx_offset,
                                     மாறிலி std::pair<முழு, முழு> d_offset, மாறிலி int8_t * q8_1_quant_ptr,
                                     மாறிலி sycl::half2 * q8_1_ds, மாறிலி முழு & iqs) {
        static_assert(mxfp4_traits::vdr_mmvq == 2, "vector load assumes vdr_mmvq == 2");
        மாறிலி uint8_t * base = static_cast<மாறிலி uint8_t *>(vbq);

        // Reordered nibble blocks are 16 contiguous bytes and iqs is 0 or 2, so each lane's two
        // weight ints are one aligned 8-byte load (the AoS layout needed eight byte loads).
        மாறிலி sycl::int2 q4 = *reinterpret_cast<மாறிலி sycl::int2 *>(base + ibx_offset.first + அளவு(முழு) * iqs);
        மாறிலி uint8_t    e  = base[d_offset.first];

        // Low nibbles pair with q8_1 ints iqs..iqs+1, high nibbles with iqs+4..iqs+5.
        மாறிலி sycl::int2 u_lo = *reinterpret_cast<மாறிலி sycl::int2 *>(q8_1_quant_ptr + அளவு(முழு) * iqs);
        மாறிலி sycl::int2 u_hi = *reinterpret_cast<மாறிலி sycl::int2 *>(q8_1_quant_ptr + அளவு(முழு) * (iqs + 4));

        மாறிலி sycl::int2 v0 = get_int_from_mxfp4(q4.x());
        மாறிலி sycl::int2 v1 = get_int_from_mxfp4(q4.y());

        முழு sumi = 0;
        sumi = ggml_sycl_dp4a(v0.x(), u_lo.x(), sumi);
        sumi = ggml_sycl_dp4a(v0.y(), u_hi.x(), sumi);
        sumi = ggml_sycl_dp4a(v1.x(), u_lo.y(), sumi);
        sumi = ggml_sycl_dp4a(v1.y(), u_hi.y(), sumi);

        மாறிலி தசம d = ggml_sycl_e8m0_to_fp32(e) * 0.5f * static_cast<தசம>((*q8_1_ds)[0]);
        திரும்பு d * sumi;
    }
};
#define VDR_Q4_0_Q8_1_MMVQ 2
#define VDR_Q4_0_Q8_1_MMQ  4

#define VDR_Q2_0_Q8_1_MMVQ 1

template <முழு vdr>
நிலையான __dpct_inline__ தசம vec_dot_q2_0_q8_1_impl(
    மாறிலி முழு * v,
    மாறிலி முழு * u,
    மாறிலி தசம & d2,
    மாறிலி sycl::half2 & ds8) {
    முழு sumi = 0;

#pragma unroll
    க்கு (முழு i = 0; i < vdr; ++i) {
#pragma unroll
        க்கு (முழு j = 0; j < 4; ++j) {
            மாறிலி uint8_t q = (uint8_t) ((uint32_t) v[i] >> (8 * j));

            // unpack 2-bit values to byte lanes (0..3), then apply zero-point
            // correction with ds8f.y() below, mirroring the q4_0 style.
            முழு vi = 0;
            vi |= (((q >> 0) & 0x3) & 0xFF) << 0;
            vi |= (((q >> 2) & 0x3) & 0xFF) << 8;
            vi |= (((q >> 4) & 0x3) & 0xFF) << 16;
            vi |= (((q >> 6) & 0x3) & 0xFF) << 24;

            sumi = dpct::dp4a(vi, u[4 * i + j], sumi);
        }
    }

    மாறிலி sycl::float2 ds8f = ds8.convert<தசம, sycl::rounding_mode::automatic>();
    // q2_0 has zero-point 1. Scale ds8f.y() by processed-lane ratio,
    // consistent with q4_0's explicit zero-point subtraction style.
    திரும்பு d2 * (sumi * ds8f.x() - ((தசம) vdr / (தசம) QI2_0) * ds8f.y());
}

template <முழு vdr>
நிலையான __dpct_inline__ தசம vec_dot_q4_0_q8_1_impl(மாறிலி முழு * v, மாறிலி முழு * u, மாறிலி தசம & d4,
                                                    மாறிலி sycl::half2 & ds8) {
    முழு sumi = 0;
#pragma unroll
    க்கு (முழு i = 0; i < vdr; ++i) {
        மாறிலி முழு vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        மாறிலி முழு vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

    மாறிலி sycl::float2 ds8f = ds8.convert<தசம, sycl::rounding_mode::automatic>();

    // second part effectively subtracts 8 from each quant value
    திரும்பு d4 * (sumi * ds8f.x() - (8 * vdr / QI4_0) * ds8f.y());
}

#define VDR_Q4_1_Q8_1_MMVQ 2
#define VDR_Q4_1_Q8_1_MMQ  4

template <முழு vdr>
நிலையான __dpct_inline__ தசம vec_dot_q4_1_q8_1_impl(மாறிலி முழு *v, மாறிலி முழு *u,
                                                    மாறிலி sycl::half2 &dm4,
                                                    மாறிலி sycl::half2 &ds8) {

    முழு sumi = 0;

#pragma unroll
    க்கு (முழு i = 0; i < vdr; ++i) {
        மாறிலி முழு vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        மாறிலி முழு vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

#ifdef GGML_SYCL_F16
    மாறிலி sycl::float2 tmp =
        (dm4 * ds8).convert<தசம, sycl::rounding_mode::automatic>();
    மாறிலி தசம d4d8 = tmp.x();
    மாறிலி தசம m4s8 = tmp.y();
#இல்லையெனில்
    மாறிலி sycl::float2 dm4f =
        dm4.convert<தசம, sycl::rounding_mode::automatic>();
    மாறிலி sycl::float2 ds8f =
        ds8.convert<தசம, sycl::rounding_mode::automatic>();
    மாறிலி தசம d4d8 = dm4f.x() * ds8f.x();
    மாறிலி தசம m4s8 = dm4f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    திரும்பு sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#define VDR_Q5_0_Q8_1_MMVQ 2
#define VDR_Q5_0_Q8_1_MMQ  4

template <முழு vdr>
நிலையான __dpct_inline__ தசம
vec_dot_q5_0_q8_1_impl(மாறிலி முழு *vl, மாறிலி முழு *vh, மாறிலி முழு *u,
                       மாறிலி தசம &d5, மாறிலி sycl::half2 &ds8) {
    முழு sumi = 0;

#pragma unroll
    க்கு (முழு i = 0; i < vdr; ++i) {
        முழு vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        முழு vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

    மாறிலி sycl::float2 ds8f =
        ds8.convert<தசம, sycl::rounding_mode::automatic>();

    // second part effectively subtracts 16 from each quant value
    திரும்பு d5 * (sumi * ds8f.x() - (16 * vdr / QI5_0) * ds8f.y());
}

#define VDR_Q5_1_Q8_1_MMVQ 2
#define VDR_Q5_1_Q8_1_MMQ  4

template <முழு vdr>
நிலையான __dpct_inline__ தசம
vec_dot_q5_1_q8_1_impl(மாறிலி முழு *vl, மாறிலி முழு *vh, மாறிலி முழு *u,
                       மாறிலி sycl::half2 &dm5, மாறிலி sycl::half2 &ds8) {

    முழு sumi = 0;

#pragma unroll
    க்கு (முழு i = 0; i < vdr; ++i) {
        முழு vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        முழு vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

#ifdef GGML_SYCL_F16
     மாறிலி sycl::float2 tmp =
        (dm5 * ds8).convert<தசம, sycl::rounding_mode::automatic>();
    மாறிலி தசம d5d8 = tmp.x();
    மாறிலி தசம m5s8 = tmp.y();


#இல்லையெனில்
    மாறிலி sycl::float2 dm5f =
        dm5.convert<தசம, sycl::rounding_mode::automatic>();
    மாறிலி sycl::float2 ds8f =
        ds8.convert<தசம, sycl::rounding_mode::automatic>();
    மாறிலி தசம d5d8 = dm5f.x() * ds8f.x();
    மாறிலி தசம m5s8 = dm5f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    திரும்பு sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#define VDR_Q8_0_Q8_1_MMVQ 2
#define VDR_Q8_0_Q8_1_MMQ 8

template <முழு vdr>
நிலையான __dpct_inline__ தசம vec_dot_q8_0_q8_1_impl(மாறிலி முழு *v, மாறிலி முழு *u,
                                                    மாறிலி தசம &d8_0,
                                                    மாறிலி தசம &d8_1) {

    முழு sumi = 0;

#pragma unroll
    க்கு (முழு i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

    திரும்பு d8_0*d8_1 * sumi;
}

template <typename T, முழு vdr>
நிலையான __dpct_inline__ T vec_dot_q8_0_q8_1_impl(மாறிலி முழு * v, மாறிலி முழு * u, மாறிலி T & d8_0, மாறிலி T & d8_1) {
    முழு sumi = 0;

#pragma unroll
    க்கு (முழு i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_sycl_dp4a(v[i], u[i], sumi);
    }

    திரும்பு d8_0*d8_1 * ((T) sumi);
}

template <முழு vdr>
நிலையான __dpct_inline__ தசம vec_dot_q8_1_q8_1_impl(மாறிலி முழு *v, மாறிலி முழு *u,
                                                    மாறிலி sycl::half2 &dm8,
                                                    மாறிலி sycl::half2 &ds8) {

    முழு sumi = 0;

#pragma unroll
    க்கு (முழு i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

#ifdef GGML_SYCL_F16
    மாறிலி sycl::float2 tmp =
        (dm8 * ds8).convert<தசம, sycl::rounding_mode::automatic>();
    மாறிலி தசம d8d8 = tmp.x();
    மாறிலி தசம m8s8 = tmp.y();
#இல்லையெனில்
    மாறிலி sycl::float2 dm8f =
        dm8.convert<தசம, sycl::rounding_mode::automatic>();
    மாறிலி sycl::float2 ds8f =
        ds8.convert<தசம, sycl::rounding_mode::automatic>();
    மாறிலி தசம d8d8 = dm8f.x() * ds8f.x();
    மாறிலி தசம m8s8 = dm8f.y() * ds8f.y();
#endif // GGML_SYCL_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    திரும்பு sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

நிலையான __dpct_inline__ தசம
vec_dot_q4_0_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                  மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {

    மாறிலி block_q4_0 * bq4_0 = (மாறிலி block_q4_0 *) vbq;

    முழு v[VDR_Q4_0_Q8_1_MMVQ];
    முழு u[2 * VDR_Q4_0_Q8_1_MMVQ];

#pragma unroll
    க்கு (முழு i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]         = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    திரும்பு vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

நிலையான __dpct_inline__ தசம
vec_dot_q2_0_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                  மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {

    மாறிலி block_q2_0 * bq2_0 = (மாறிலி block_q2_0 *) vbq;

    முழு v[2 * VDR_Q2_0_Q8_1_MMVQ];
    முழு u[8 * VDR_Q2_0_Q8_1_MMVQ];

#pragma unroll
    க்கு (முழு i = 0; i < VDR_Q2_0_Q8_1_MMVQ; ++i) {
        மாறிலி முழு base = 4 * (iqs + i);

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

    மாறிலி தசம sum0 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + 0, u + 0, bq2_0->d, bq8_1[0].ds);
    மாறிலி தசம sum1 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + VDR_Q2_0_Q8_1_MMVQ, u + 4 * VDR_Q2_0_Q8_1_MMVQ, bq2_0->d, bq8_1[1].ds);
    திரும்பு sum0 + sum1;
}

நிலையான __dpct_inline__ தசம
vec_dot_q4_1_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                  மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {

    மாறிலி block_q4_1 * bq4_1 = (மாறிலி block_q4_1 *) vbq;

    முழு v[VDR_Q4_1_Q8_1_MMVQ];
    முழு u[2*VDR_Q4_1_Q8_1_MMVQ];

#pragma unroll
    க்கு (முழு i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    திரும்பு vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

#define VDR_MXFP4_Q8_1_MMVQ 2
#define VDR_MXFP4_Q8_1_MMQ  4

நிலையான __dpct_inline__ தசம vec_dot_mxfp4_q8_1(மாறிலி வெறுமை * __restrict__ vbq,
                                                மாறிலி block_q8_1 * __restrict__ bq8_1,
                                                மாறிலி முழு & iqs) {
    மாறிலி block_mxfp4 * bq4 = (மாறிலி block_mxfp4 *) vbq;

    மாறிலி முழு * q8 = (மாறிலி முழு *) bq8_1->qs + iqs;

    முழு sumi = 0;
#pragma unroll
    க்கு (முழு l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        மாறிலி முழு aux_q4 = get_int_b1(bq4->qs, iqs + l);
        மாறிலி sycl::int2 v      = get_int_from_mxfp4(aux_q4);
        sumi = ggml_sycl_dp4a(v.x(), q8[l + 0], sumi);
        sumi = ggml_sycl_dp4a(v.y(), q8[l + 4], sumi);
    }

    மாறிலி தசம d = ggml_sycl_e8m0_to_fp32(bq4->e) * 0.5f * (bq8_1->ds)[0];
    திரும்பு d * sumi;
}

#define VDR_NVFP4_Q8_1_MMVQ 4
#define VDR_NVFP4_Q8_1_MMQ  8

நிலையான __dpct_inline__ தசம vec_dot_nvfp4_q8_1(மாறிலி வெறுமை * __restrict__ vbq,
                                                மாறிலி block_q8_1 * __restrict__ bq8_1,
                                                மாறிலி int32_t & iqs) {
    மாறிலி block_nvfp4 * bq4 = (மாறிலி block_nvfp4 *) vbq;
    தசம sum = 0.0f;
#pragma unroll
    க்கு (முழு i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; i++) {
        மாறிலி int32_t iqs0 = iqs + 2*i;
        மாறிலி int32_t iqs1 = iqs0 + 1;
        மாறிலி int32_t is = iqs0 >> 1;
        மாறிலி sycl::int2   v0   = get_int_from_table_16(get_int_b4(bq4->qs, iqs0), kvalues_mxfp4);
        மாறிலி sycl::int2   v1   = get_int_from_table_16(get_int_b4(bq4->qs, iqs1), kvalues_mxfp4);
        மாறிலி block_q8_1 * bq8 = bq8_1 + (is >> 1);
        மாறிலி int32_t i8 = ((is & 1) << 2);

        முழு sumi = ggml_sycl_dp4a(v0.x(), get_int_b4(bq8->qs, i8 + 0), 0);
        sumi     = ggml_sycl_dp4a(v0.y(), get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi     = ggml_sycl_dp4a(v1.x(), get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi     = ggml_sycl_dp4a(v1.y(), get_int_b4(bq8->qs, i8 + 3), sumi);

        மாறிலி தசம d = ggml_sycl_ue4m3_to_fp32(bq4->d[is]) * (bq8->ds)[0];
        sum += d * தசம(sumi);
    }

    திரும்பு sum;
}

நிலையான __dpct_inline__ தசம
vec_dot_q5_0_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                  மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {

    மாறிலி block_q5_0 * bq5_0 = (மாறிலி block_q5_0 *) vbq;

    முழு vl[VDR_Q5_0_Q8_1_MMVQ];
    முழு vh[VDR_Q5_0_Q8_1_MMVQ];
    முழு  u[2*VDR_Q5_0_Q8_1_MMVQ];

#pragma unroll
    க்கு (முழு i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    திரும்பு vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

நிலையான __dpct_inline__ தசம
vec_dot_q5_1_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                  மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {

    மாறிலி block_q5_1 * bq5_1 = (மாறிலி block_q5_1 *) vbq;

    முழு vl[VDR_Q5_1_Q8_1_MMVQ];
    முழு vh[VDR_Q5_1_Q8_1_MMVQ];
    முழு  u[2*VDR_Q5_1_Q8_1_MMVQ];

#pragma unroll
    க்கு (முழு i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    திரும்பு vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

நிலையான __dpct_inline__ தசம
vec_dot_q8_0_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                  மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {

    மாறிலி block_q8_0 * bq8_0 = (மாறிலி block_q8_0 *) vbq;

    முழு v[VDR_Q8_0_Q8_1_MMVQ];
    முழு u[VDR_Q8_0_Q8_1_MMVQ];

#pragma unroll
    க்கு (முழு i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    திரும்பு vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                      bq8_1->ds[0]);
}

நிலையான __dpct_inline__ தசம
vec_dot_q2_K_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                  மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {

    மாறிலி block_q2_K * bq2_K = (மாறிலி block_q2_K *) vbq;

    மாறிலி முழு bq8_offset = QR2_K * (iqs / QI8_1);
    மாறிலி முழு scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    மாறிலி uint8_t * scales = bq2_K->scales + scale_offset;

    மாறிலி முழு v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    முழு    u[QR2_K];
    தசம d8[QR2_K];

#pragma unroll
    க்கு (முழு i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + i].ds[0];
    }

    திரும்பு vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

நிலையான __dpct_inline__ தசம
vec_dot_q3_K_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                  மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {

    மாறிலி block_q3_K * bq3_K = (மாறிலி block_q3_K *) vbq;

    மாறிலி முழு bq8_offset = QR3_K * (iqs / (QI3_K/2));
    மாறிலி முழு scale_offset = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    மாறிலி தசம d = bq3_K->d;

    மாறிலி முழு vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    மாறிலி முழு vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_offset;

    முழு    u[QR3_K];
    தசம d8[QR3_K];

#pragma unroll
    க்கு (முழு i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_offset + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_offset + i].ds[0];
    }

    திரும்பு vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_offset, d, d8);
}

நிலையான __dpct_inline__ தசம vec_dot_q4_K_q8_1(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி block_q8_1 * __restrict__ bq8_1,
                                               மாறிலி முழு & iqs) {
#ifndef GGML_QKK_64

    மாறிலி block_q4_K * bq4_K = (மாறிலி block_q4_K *) vbq;

    மாறிலி முழு        bq8_offset = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    மாறிலி முழு *      q4         = (மாறிலி முழு *) (bq4_K->qs + 16 * bq8_offset + 4 * ((iqs / 2) % 4));
    மாறிலி uint16_t * scales     = (மாறிலி uint16_t *) bq4_K->scales;

    திரும்பு vec_dot_q4_K_q8_1_common(q4, scales, bq4_K->dm, bq8_1, iqs);

#இல்லையெனில்

#எனில் __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    மாறிலி block_q4_K * bq4_K = (மாறிலி block_q4_K *) vbq;

    தசம sumf_d = 0.0f;
    தசம sumf_m = 0.0f;

    uint16_t aux16[2];
    மாறிலி uint8_t * s = (மாறிலி uint8_t *)aux16;

    மாறிலி uint16_t * a = (மாறிலி uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    மாறிலி தசம dall = bq4_K->dm[0];
    மாறிலி தசம dmin = bq4_K->dm[1];

    மாறிலி தசம d8_1 = bq8_1[0].ds[0];
    மாறிலி தசம d8_2 = bq8_1[1].ds[1];

    மாறிலி முழு ui1 = *((மாறிலி முழு *)bq8_1[0].qs + (iqs/2));
    மாறிலி முழு ui2 = *((மாறிலி முழு *)bq8_1[0].qs + (iqs/2) + 4);
    மாறிலி முழு ui3 = *((மாறிலி முழு *)bq8_1[1].qs + (iqs/2));
    மாறிலி முழு ui4 = *((மாறிலி முழு *)bq8_1[1].qs + (iqs/2) + 4);

    மாறிலி முழு * q4 = (மாறிலி முழு *)bq4_K->qs + (iqs/2);
    மாறிலி முழு v1 = q4[0];
    மாறிலி முழு v2 = q4[4];

    மாறிலி முழு dot1 = dpct::dp4a(ui2, v2 & 0x0f0f0f0f, dpct::dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    மாறிலி முழு dot2 = dpct::dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, dpct::dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    மாறிலி முழு dot3 = dpct::dp4a(0x01010101, ui2, dpct::dp4a(0x01010101, ui1, 0));
    மாறிலி முழு dot4 = dpct::dp4a(0x01010101, ui4, dpct::dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    திரும்பு dall * sumf_d - dmin * sumf_m;

#இல்லையெனில்
    bad_arch();
#endif // __SYCL_ARCH__ >= VER_4VEC

#endif
}

நிலையான __dpct_inline__ தசம
vec_dot_q5_K_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                  மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {

#ifndef GGML_QKK_64
    மாறிலி block_q5_K * bq5_K = (மாறிலி block_q5_K *) vbq;

    முழு   vl[2];
    முழு   vh[2];
    முழு    u[2*QR5_K];
    தசம d8[QR5_K];

    மாறிலி முழு bq8_offset = QR5_K * ((iqs/2) / (QI8_1/2));
    மாறிலி முழு * ql = (மாறிலி முழு *)(bq5_K->qs + 16 * bq8_offset + 4 * ((iqs/2)%4));
    மாறிலி முழு * qh = (மாறிலி முழு *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_offset;
    vh[1] = qh[4] >> bq8_offset;

    மாறிலி uint16_t * scales = (மாறிலி uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    மாறிலி முழு j = bq8_offset/2;
    எனில் (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } இல்லையெனில் {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    மாறிலி uint8_t * sc = (மாறிலி uint8_t *)aux;
    மாறிலி uint8_t * m  = sc + 2;

#pragma unroll
    க்கு (முழு i = 0; i < QR5_K; ++i) {
        மாறிலி block_q8_1 * bq8i = bq8_1 + bq8_offset + i;
        d8[i] = bq8i->ds[0];

        மாறிலி முழு * q8 = (மாறிலி முழு *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    திரும்பு vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#இல்லையெனில்

#எனில் __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    மாறிலி block_q5_K * bq5_K = (மாறிலி block_q5_K *) vbq;

    மாறிலி int8_t * s = bq5_K->scales;

    மாறிலி தசம d = bq5_K->d;

    மாறிலி தசம d8_1 = bq8_1[0].ds[0];
    மாறிலி தசம d8_2 = bq8_1[1].ds[1];

    மாறிலி முழு ui1 = *((மாறிலி முழு *)bq8_1[0].qs + (iqs/2));
    மாறிலி முழு ui2 = *((மாறிலி முழு *)bq8_1[0].qs + (iqs/2) + 4);
    மாறிலி முழு ui3 = *((மாறிலி முழு *)bq8_1[1].qs + (iqs/2));
    மாறிலி முழு ui4 = *((மாறிலி முழு *)bq8_1[1].qs + (iqs/2) + 4);

    மாறிலி முழு * ql = (மாறிலி முழு *)bq5_K->qs + (iqs/2);
    மாறிலி முழு vl1 = ql[0];
    மாறிலி முழு vl2 = ql[4];

    மாறிலி முழு step = 4 * (iqs/2); // 0, 4, 8, 12
    மாறிலி முழு im = step/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    மாறிலி முழு in = step%8; // 0, 4, 0, 4
    மாறிலி முழு vh = (*((மாறிலி முழு *)(bq5_K->qh + in))) >> im;

    மாறிலி முழு v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    மாறிலி முழு v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    மாறிலி முழு v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    மாறிலி முழு v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    மாறிலி தசம sumf_d = d8_1 * (dpct::dp4a(ui1, v1, 0) * s[0] + dpct::dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (dpct::dp4a(ui3, v3, 0) * s[2] + dpct::dp4a(ui4, v4, 0) * s[3]);

    திரும்பு d * sumf_d;

#இல்லையெனில்
    bad_arch();
#endif // __SYCL_ARCH__ >= VER_4VEC

#endif
}

நிலையான __dpct_inline__ தசம
vec_dot_q6_K_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                  மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {

    மாறிலி block_q6_K * bq6_K = (மாறிலி block_q6_K *) vbq;

    மாறிலி முழு bq8_offset = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    மாறிலி முழு scale_offset = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    மாறிலி முழு vh_shift = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    மாறிலி முழு vl = get_int_from_uint8(bq6_K->ql, iqs);
    மாறிலி முழு vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_shift;

    மாறிலி int8_t * scales = bq6_K->scales + scale_offset;

    மாறிலி முழு u0 = get_int_from_int8_aligned(
        bq8_1[bq8_offset + 0].qs, iqs % QI8_1);
    மாறிலி முழு u1 = get_int_from_int8_aligned(
        bq8_1[bq8_offset + 2].qs, iqs % QI8_1);
    மாறிலி தசம d80 = bq8_1[bq8_offset + 0].ds[0];
    மாறிலி தசம d81 = bq8_1[bq8_offset + 2].ds[0];

    திரும்பு vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u0, u1, scales[0], scales[4], bq6_K->d, d80, d81);
}


// NOTE: the VDR_IQ*_Q8_1_MMVQ values deliberately differ from the identically named CUDA constants
// (vecdotq.cuh): the SYCL kernels pair them with a halved qi (e.g. QI3_S/2), so the values are not
// interchangeable and must not be copied across backends.
#define VDR_IQ2_XXS_Q8_1_MMVQ 1

நிலையான __dpct_inline__ uint32_t iq_sign_mask(மாறிலி uint32_t signs) {
    மாறிலி uint32_t bits =
        (signs & 0x01) | ((signs & 0x02) << 7) |
        ((signs & 0x04) << 14) | ((signs & 0x08) << 21);

    திரும்பு bits * 0xff;
}

நிலையான __dpct_inline__ uint32_t iq_sign_bits(மாறிலி uint32_t encoded) {
    மாறிலி uint32_t signs = encoded & 127;

    திரும்பு signs | ((sycl::popcount(signs) & 1) << 7);
}

நிலையான __dpct_inline__ தசம
vec_dot_iq2_xxs_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                     மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs,
                     மாறிலி uint64_t *iq2xxs_grid, மாறிலி uint8_t *ksigns_iq2xs,
                     மாறிலி uint8_t *kmask_iq2xs) {
#எனில் QK_K == 256
    மாறிலி block_iq2_xxs * bq2 = (மாறிலி block_iq2_xxs *) vbq;

    மாறிலி முழு ib32 = iqs;
    மாறிலி uint16_t * q2 = bq2->qs + 4*ib32;
    மாறிலி uint8_t  * aux8 = (மாறிலி uint8_t *)q2;
    மாறிலி int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = q2[2] | (q2[3] << 16);
    முழு sumi = 0;
    க்கு (முழு l = 0; l < 4; ++l) {
        மாறிலி uint8_t * grid = (மாறிலி uint8_t *)(iq2xxs_grid + aux8[l]);
        மாறிலி uint8_t  signs = ksigns_iq2xs[aux32 & 127];
        க்கு (முழு j = 0; j < 8; ++j) {
            sumi += q8[j] * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
        aux32 >>= 7;
    }
    மாறிலி தசம d = (தசம)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.25f;
    திரும்பு d * sumi;
#இல்லையெனில்
    ப்ரதிஜ்ஞா(பொய்);
    திரும்பு 0.f;
#endif
}

#define VDR_IQ2_XS_Q8_1_MMVQ 1

நிலையான __dpct_inline__ தசம
vec_dot_iq2_xs_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                    மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs,
                    மாறிலி uint64_t *iq2xs_grid) {
#எனில் DPCT_COMPATIBILITY_TEMP >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#எனில் QK_K == 256
    மாறிலி block_iq2_xs * bq2 = (மாறிலி block_iq2_xs *) vbq;

    மாறிலி முழு ib32 = iqs;
    மாறிலி uint16_t * q2 = bq2->qs + 4*ib32;
    மாறிலி int8_t   * q8 = bq8_1[ib32].qs;
    மாறிலி uint8_t ls1 = bq2->scales[ib32] & 0xf;
    மாறிலி uint8_t ls2 = bq2->scales[ib32] >>  4;
    முழு sumi1 = 0;
#pragma unroll
    க்கு (முழு l = 0; l < 2; ++l) {
        மாறிலி uint64_t grid = iq2xs_grid[q2[l] & 511];
        மாறிலி uint32_t sign_bits = iq_sign_bits(q2[l] >> 9);
        மாறிலி uint32_t signs0 = iq_sign_mask(sign_bits);
        மாறிலி uint32_t signs1 = iq_sign_mask(sign_bits >> 4);
        மாறிலி முழு grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        மாறிலி முழு grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi1 = dpct::dp4a(grid_l, *((மாறிலி முழு *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((மாறிலி முழு *)q8 + 1), sumi1);
        q8 += 8;
    }
    முழு sumi2 = 0;
#pragma unroll
    க்கு (முழு l = 2; l < 4; ++l) {
        மாறிலி uint64_t grid = iq2xs_grid[q2[l] & 511];
        மாறிலி uint32_t sign_bits = iq_sign_bits(q2[l] >> 9);
        மாறிலி uint32_t signs0 = iq_sign_mask(sign_bits);
        மாறிலி uint32_t signs1 = iq_sign_mask(sign_bits >> 4);
        மாறிலி முழு grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        மாறிலி முழு grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi2 = dpct::dp4a(grid_l, *((மாறிலி முழு *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((மாறிலி முழு *)q8 + 1), sumi2);
        q8 += 8;
    }
    மாறிலி தசம d = (தசம)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    திரும்பு d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#இல்லையெனில்
    ப்ரதிஜ்ஞா(பொய்);
    திரும்பு 0.f;
#endif
#இல்லையெனில்
    ப்ரதிஜ்ஞா(பொய்);
    திரும்பு 0.f;
#endif
}

#define VDR_IQ2_S_Q8_1_MMVQ 1

நிலையான __dpct_inline__ தசம
vec_dot_iq2_s_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                   மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {
#எனில் QK_K == 256
    மாறிலி block_iq2_s * bq2 = (மாறிலி block_iq2_s *) vbq;

    மாறிலி முழு ib32 = iqs;
    மாறிலி int8_t  * q8 = bq8_1[ib32].qs;
    மாறிலி uint8_t * signs = bq2->qs + QK_K/8 + 4*ib32;
    மாறிலி uint8_t ls1 = bq2->scales[ib32] & 0xf;
    மாறிலி uint8_t ls2 = bq2->scales[ib32] >>  4;
    முழு sumi1 = 0;
    க்கு (முழு l = 0; l < 2; ++l) {
        மாறிலி uint32_t * grid = (மாறிலி uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        மாறிலி uint32_t signs0 = dpct::vectorized_இருமம்<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        மாறிலி uint32_t signs1 = dpct::vectorized_இருமம்<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        மாறிலி முழு grid_l = dpct::vectorized_இருமம்<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        மாறிலி முழு grid_h = dpct::vectorized_இருமம்<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi1 = dpct::dp4a(grid_l, *((மாறிலி முழு *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((மாறிலி முழு *)q8 + 1), sumi1);
        q8 += 8;
    }
    முழு sumi2 = 0;
    க்கு (முழு l = 2; l < 4; ++l) {
        மாறிலி uint32_t * grid = (மாறிலி uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        மாறிலி uint32_t signs0 = dpct::vectorized_இருமம்<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        மாறிலி uint32_t signs1 = dpct::vectorized_இருமம்<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        மாறிலி முழு grid_l = dpct::vectorized_இருமம்<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        மாறிலி முழு grid_h = dpct::vectorized_இருமம்<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi2 = dpct::dp4a(grid_l, *((மாறிலி முழு *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((மாறிலி முழு *)q8 + 1), sumi2);
        q8 += 8;
    }
    மாறிலி தசம d = (தசம)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    திரும்பு d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#இல்லையெனில்
    ப்ரதிஜ்ஞா(பொய்);
#endif
}

#define VDR_IQ3_XXS_Q8_1_MMVQ 1

நிலையான __dpct_inline__ தசம
vec_dot_iq3_xxs_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                     மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs,
                     மாறிலி uint32_t *iq3xxs_grid) {
#எனில் DPCT_COMPATIBILITY_TEMP >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#எனில் QK_K == 256
    மாறிலி block_iq3_xxs * bq2 = (மாறிலி block_iq3_xxs *) vbq;

    மாறிலி முழு ib32 = iqs;
    மாறிலி uint8_t  * q3 = bq2->qs + 8*ib32;
    மாறிலி uint16_t * gas = (மாறிலி uint16_t *)(bq2->qs + QK_K/4) + 2*ib32;
    மாறிலி int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = gas[0] | (gas[1] << 16);
    முழு sumi = 0;
#pragma unroll
    க்கு (முழு l = 0; l < 4; ++l) {
        மாறிலி uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
        மாறிலி uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
        மாறிலி uint32_t sign_bits = iq_sign_bits(aux32);
        மாறிலி uint32_t signs0 = iq_sign_mask(sign_bits);
        மாறிலி uint32_t signs1 = iq_sign_mask(sign_bits >> 4);
        மாறிலி முழு grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        மாறிலி முழு grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((மாறிலி முழு *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((மாறிலி முழு *)q8 + 1), sumi);
        q8 += 8;
        aux32 >>= 7;
    }
    மாறிலி தசம d = (தசம)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.5f;
    திரும்பு d * sumi;
#இல்லையெனில்
    ப்ரதிஜ்ஞா(பொய்);
    திரும்பு 0.f;
#endif
#இல்லையெனில்
    ப்ரதிஜ்ஞா(பொய்);
    திரும்பு 0.f;
#endif
}

template <> கட்டமைப்பு reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_XXS> {
    நிலையான constexpr ggml_type gtype = GGML_TYPE_IQ3_XXS;

    __dpct_inline__ தசம operator()(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி std::pair<முழு, முழு> ibx_offset,
                                     மாறிலி std::pair<முழு, முழு> d_offset, மாறிலி int8_t * q8_1_quant_ptr,
                                     மாறிலி sycl::half2 * q8_1_ds, மாறிலி முழு & iqs) {
#எனில் DPCT_COMPATIBILITY_TEMP >= MIN_CC_DP4A && QK_K == 256
        மாறிலி uint8_t * base = static_cast<மாறிலி uint8_t *>(vbq);
        மாறிலி uint8_t * qs   = base + ibx_offset.first;
        மாறிலி uint8_t * q3   = qs + 8 * iqs;
        மாறிலி uint16_t * gas = reinterpret_cast<மாறிலி uint16_t *>(qs + QK_K / 4) + 2 * iqs;
        மாறிலி int8_t * q8    = q8_1_quant_ptr + iqs * QK8_1;
        uint32_t aux32       = gas[0] | (gas[1] << 16);

        முழு sumi = 0;
#pragma unroll
        க்கு (முழு l = 0; l < 4; ++l) {
            மாறிலி uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
            மாறிலி uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
            மாறிலி uint32_t sign_bits = iq_sign_bits(aux32);
            மாறிலி uint32_t signs0 = iq_sign_mask(sign_bits);
            மாறிலி uint32_t signs1 = iq_sign_mask(sign_bits >> 4);
            மாறிலி முழு grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            மாறிலி முழு grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((மாறிலி முழு *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((மாறிலி முழு *) q8 + 1), sumi);
            q8 += 8;
            aux32 >>= 7;
        }

        மாறிலி ggml_half d = *reinterpret_cast<மாறிலி ggml_half *>(base + d_offset.first);
        திரும்பு (தசம) d * (0.5f + aux32) * q8_1_ds[iqs][0] * 0.5f * sumi;
#இல்லையெனில்
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

நிலையான __dpct_inline__ தசம
vec_dot_iq3_s_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                   மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs,
                   மாறிலி uint32_t *iq3s_grid) {
#எனில் QK_K == 256
    மாறிலி block_iq3_s * bq2 = (மாறிலி block_iq3_s *) vbq;

    மாறிலி முழு ib32 = iqs;
    மாறிலி uint8_t  * qs = bq2->qs + 8*ib32;
    மாறிலி int8_t   * q8 = bq8_1[ib32].qs;
    முழு sumi = 0;
    க்கு (முழு l = 0; l < 4; ++l) {
        மாறிலி uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((bq2->qh[ib32] << (8 - 2*l)) & 256)];
        மாறிலி uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((bq2->qh[ib32] << (7 - 2*l)) & 256)];
        மாறிலி uint32_t signs = bq2->signs[4 * ib32 + l];
        மாறிலி uint32_t signs0 = iq_sign_mask(signs);
        மாறிலி uint32_t signs1 = iq_sign_mask(signs >> 4);
        மாறிலி முழு grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        மாறிலி முழு grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((மாறிலி முழு *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((மாறிலி முழு *)q8 + 1), sumi);
        q8 += 8;
    }
    மாறிலி தசம d =
        (தசம)bq2->d *
        (1 + 2 * ((bq2->scales[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf)) *
        bq8_1[ib32].ds[0];
    திரும்பு d * sumi;
#இல்லையெனில்
    ப்ரதிஜ்ஞா(பொய்);
#endif
}

template <> கட்டமைப்பு reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_S> {
    நிலையான constexpr ggml_type gtype = GGML_TYPE_IQ3_S;

    __dpct_inline__ தசம operator()(மாறிலி வெறுமை * __restrict__ vbq, மாறிலி std::pair<முழு, முழு> ibx_offset,
                                     மாறிலி std::pair<முழு, முழு> d_offset, மாறிலி int8_t * q8_1_quant_ptr,
                                     மாறிலி sycl::half2 * q8_1_ds, மாறிலி முழு & iqs) {
#எனில் QK_K == 256
        மாறிலி uint8_t * base     = static_cast<மாறிலி uint8_t *>(vbq);
        மாறிலி uint8_t * qs       = base + ibx_offset.first + 8 * iqs;
        மாறிலி uint8_t * qh       = base + ibx_offset.second;
        மாறிலி uint8_t * signs    = base + d_offset.first + 4 * iqs;
        மாறிலி uint8_t * metadata = base + d_offset.second;
        மாறிலி ggml_half d        = *reinterpret_cast<மாறிலி ggml_half *>(metadata);
        மாறிலி uint8_t * scales   = metadata + அளவு(ggml_half);
        மாறிலி int8_t * q8        = q8_1_quant_ptr + iqs * QK8_1;

        முழு sumi = 0;
#pragma unroll
        க்கு (முழு l = 0; l < 4; ++l) {
            மாறிலி uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((qh[iqs] << (8 - 2*l)) & 256)];
            மாறிலி uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((qh[iqs] << (7 - 2*l)) & 256)];
            மாறிலி uint32_t signs0 = iq_sign_mask(signs[l]);
            மாறிலி uint32_t signs1 = iq_sign_mask(signs[l] >> 4);
            மாறிலி முழு grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            மாறிலி முழு grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((மாறிலி முழு *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((மாறிலி முழு *) q8 + 1), sumi);
            q8 += 8;
        }

        மாறிலி தசம scale = 1 + 2 * ((scales[iqs / 2] >> (4 * (iqs % 2))) & 0xf);
        திரும்பு (தசம) d * scale * q8_1_ds[iqs][0] * sumi;
#இல்லையெனில்
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

நிலையான __dpct_inline__ தசம
vec_dot_iq1_s_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                   மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs,
                   மாறிலி uint32_t *iq1s_grid_gpu) {
#எனில் QK_K == 256
    மாறிலி block_iq1_s * bq1 = (மாறிலி block_iq1_s *) vbq;

    மாறிலி முழு ib32 = iqs;
    முழு sumi = 0;
    மாறிலி முழு * q8 = (மாறிலி முழு *)bq8_1[ib32].qs;
    க்கு (முழு l = 0; l < 4; ++l) {
        மாறிலி முழு * grid = (மாறிலி முழு *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[ib32] >> 3*l) & 7) << 8)));
        முழு grid0 = grid[0] & 0x0f0f0f0f;
        முழு grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi = dpct::dp4a(q8[2 * l + 1], grid1,
                          dpct::dp4a(q8[2 * l + 0], grid0, sumi));
    }

    மாறிலி தசம delta = bq1->qh[ib32] & 0x8000 ? -1-IQ1S_DELTA : -1+IQ1S_DELTA;
    மாறிலி தசம d1q = (தசம)bq1->d * (2*((bq1->qh[ib32] >> 12) & 7) + 1);
    மாறிலி தசம d = d1q * bq8_1[ib32].ds[0];
    மாறிலி தசம m = d1q * bq8_1[ib32].ds[1];
    திரும்பு d * sumi + m * delta;
#இல்லையெனில்
    ப்ரதிஜ்ஞா(பொய்);
#endif
}

#define VDR_IQ1_M_Q8_1_MMVQ 1

நிலையான __dpct_inline__ தசம
vec_dot_iq1_m_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                   மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {
#எனில் QK_K == 256
    மாறிலி block_iq1_m * bq1 = (மாறிலி block_iq1_m *) vbq;

    மாறிலி முழு ib32 = iqs;
    முழு   sumi[2] = {0, 0};
    தசம sumf[2] = {0.f, 0.f};

    மாறிலி முழு * q8 = (மாறிலி முழு *)bq8_1[ib32].qs;
    க்கு (முழு l = 0; l < 4; ++l) {
        மாறிலி முழு * grid = (மாறிலி முழு *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 7) << 8)));
        முழு grid0 = grid[0] & 0x0f0f0f0f;
        முழு grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi[l / 2] = dpct::dp4a(q8[2 * l + 1], grid1,
                                 dpct::dp4a(q8[2 * l + 0], grid0, sumi[l / 2]));
        மாறிலி தசம delta = (bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 0x08 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA;
        மாறிலி முழு sumy = dpct::dp4a(q8[2 * l + 1], 0x01010101,
                                    dpct::dp4a(q8[2 * l + 0], 0x01010101, 0));
        sumf[l/2] += delta*sumy;
    }

    iq1m_scale_t scale;
    மாறிலி uint16_t * sc = (மாறிலி uint16_t *)bq1->scales;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    மாறிலி தசம d = (தசம)scale.f16 * bq8_1[ib32].ds[0];
    திரும்பு d * ((sumi[0] + sumf[0]) * (2*((sc[ib32/2] >> 6*(ib32%2)) & 0x7) + 1) + (sumi[1] + sumf[1]) * (2*((sc[ib32/2] >> (6*(ib32%2)+3)) & 0x7) + 1));
#இல்லையெனில்
    ப்ரதிஜ்ஞா(பொய்);
#endif
}


#define VDR_IQ4_NL_Q8_1_MMVQ 2

நிலையான __dpct_inline__ தசம
vec_dot_iq4_nl_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                    மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {

    மாறிலி block_iq4_nl * bq = (மாறிலி block_iq4_nl *) vbq;

    மாறிலி uint16_t * q4 = (மாறிலி uint16_t *)bq->qs + 2*iqs;
    மாறிலி int32_t  * q8 = (மாறிலி int32_t  *)bq8_1->qs + iqs;

    மாறிலி uint8_t * மதிப்புகள் = (மாறிலி uint8_t *)kvalues_iq4nl;

    முழு v1, v2;
    முழு sumi1 = 0, sumi2 = 0;
    க்கு (முழு l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        மாறிலி uint32_t aux = q4[2*l] | (q4[2*l+1] << 16);
        get_int_from_table_16(aux, மதிப்புகள், v1, v2);
        sumi1 = dpct::dp4a(v1, q8[l + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[l + 4], sumi2);
    }

    மாறிலி தசம d = (தசம)bq->d * bq8_1->ds[0];
    திரும்பு d * (sumi1 + sumi2);
}


#define VDR_IQ4_XS_Q8_1_MMVQ 1

நிலையான __dpct_inline__ தசம
vec_dot_iq4_xs_q8_1(மாறிலி வெறுமை *__restrict__ vbq,
                    மாறிலி block_q8_1 *__restrict__ bq8_1, மாறிலி முழு &iqs) {

#எனில் QK_K == 256
    மாறிலி block_iq4_xs * bq4 = (மாறிலி block_iq4_xs *) vbq;
    மாறிலி uint8_t * மதிப்புகள் = (மாறிலி uint8_t *)kvalues_iq4nl;

    // iqs is 0...7
    மாறிலி முழு ib32 = iqs;
    மாறிலி int32_t  * q8 = (மாறிலி முழு *)bq8_1[ib32].qs;
    மாறிலி uint32_t * q4 = (மாறிலி uint32_t *)bq4->qs + 4*ib32;
    மாறிலி int8_t ls = ((bq4->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((bq4->scales_h >> 2*ib32) & 3) << 4);
    மாறிலி தசம d = (தசம)bq4->d * (ls - 32) * bq8_1[ib32].ds[0];
    முழு v1, v2;
    முழு sumi1 = 0, sumi2 = 0;
    க்கு (முழு j = 0; j < 4; ++j) {
        get_int_from_table_16(q4[j], மதிப்புகள், v1, v2);
        sumi1 = dpct::dp4a(v1, q8[j + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[j + 4], sumi2);
    }
    திரும்பு d * (sumi1 + sumi2);
#இல்லையெனில்
    ப்ரதிஜ்ஞா(பொய்);
#endif
}

#endif // GGML_SYCL_VECDOTQ_HPP
