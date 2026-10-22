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

#اگر_تعریف_نشده GGML_SYCL_VECDOTQ_HPP
#تعریف_کن GGML_SYCL_VECDOTQ_HPP

#ضمن "dpct/helper.hpp"
#ضمن "ggml.h"
#ضمن "type.hpp"
#ضمن "quants.hpp"

تعريفنوع عائم (*vec_dot_q_sycl_t)(ثابت فراغ * __restrict__ vbq, ثابت بلوک_q8_1 * __restrict__ bq8_1,
                                  ثابت صحيح & iqs);

ساكن __dpct_inline__ صحيح get_int_b1(ثابت فراغ * x, ثابت صحيح & i32) {
    ثابت uint8_t * x8 = (ثابت uint8_t *) x;

    صحيح x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    أرجع x32;
}

ساكن __dpct_inline__ صحيح get_int_b2(ثابت فراغ * x, ثابت صحيح & i32) {
    ثابت uint16_t * x16 = (ثابت uint16_t *) x; // assume at least 2 byte alignment

    صحيح x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    أرجع x32;
}

ساكن __dpct_inline__ صحيح get_int_b4(ثابت فراغ * x, ثابت صحيح & i32) {
    أرجع ((ثابت صحيح *) x)[i32]; // assume at least 4 byte alignment
}

ساكن __dpct_inline__ صحيح get_int_from_int8(ثابت int8_t* x8, ثابت صحيح& i32) {
  ثابت uint16_t* x16 =
      (ثابت uint16_t*)(x8 + حجم(صحيح) * i32); // assume at least 2 byte
                                                 // alignment

  صحيح x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  أرجع x32;
}

ساكن __dpct_inline__ صحيح get_int_from_uint8(
    ثابت uint8_t* x8,
    ثابت صحيح& i32) {
  ثابت uint16_t* x16 =
      (ثابت uint16_t*)(x8 + حجم(صحيح) * i32); // assume at least 2 byte
                                                 // alignment

  صحيح x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  أرجع x32;
}

ساكن __dpct_inline__ صحيح get_int_from_int8_aligned(
    ثابت int8_t* x8,
    ثابت صحيح& i32) {
  أرجع *(
      (ثابت صحيح*)(x8 + حجم(صحيح) * i32)); // assume at least 4 byte alignment
}

ساكن __dpct_inline__ صحيح get_int_from_uint8_aligned(
    ثابت uint8_t* x8,
    ثابت صحيح& i32) {
  أرجع *(
      (ثابت صحيح*)(x8 + حجم(صحيح) * i32)); // assume at least 4 byte alignment
}

ساكن __dpct_inline__ صحيح byte_sub_4(ثابت صحيح a, ثابت صحيح b) {
  ثابت uint32_t ua = static_cast<uint32_t>(a);
  ثابت uint32_t ub = static_cast<uint32_t>(b);
  أرجع static_cast<صحيح>(((ua | 0x80808080u) - ub) ^ 0x80808080u);
}

ساكن __dpct_inline__ عائم vec_dot_q6_K_q8_1_impl_mmvq_scalar(
    ثابت صحيح vl, ثابت صحيح vh, ثابت صحيح u0, ثابت صحيح u1, ثابت int8_t sc0,
    ثابت int8_t sc1, ثابت عائم d, ثابت عائم d80, ثابت عائم d81) {
    static_assert(QR6_K == 2, "q6_K MMVQ scalar fast path assumes QR6_K == 2");

    ثابت صحيح vil0 = (vl >> 0) & 0x0F0F0F0F;
    ثابت صحيح vih0 = ((vh >> 0) << 4) & 0x30303030;
    ثابت صحيح vi0 = byte_sub_4(vil0 | vih0, 0x20202020);

    ثابت صحيح vil1 = (vl >> 4) & 0x0F0F0F0F;
    ثابت صحيح vih1 = ((vh >> 4) << 4) & 0x30303030;
    ثابت صحيح vi1 = byte_sub_4(vil1 | vih1, 0x20202020);

    ثابت عائم sumf =
        d80 * (dpct::dp4a(vi0, u0, 0) * sc0) +
        d81 * (dpct::dp4a(vi1, u1, 0) * sc1);

    أرجع d * sumf;
}

ساكن __dpct_inline__ فراغ get_int_from_جدول_16(ثابت uint32_t &q4,
                                                  ثابت uint8_t *مقادیر,
                                                  صحيح &val1, صحيح &val2) {

    uint32_t aux32; ثابت uint8_t * q8 = (ثابت uint8_t *)&aux32;
    aux32 = q4 & 0x0f0f0f0f;
    uint16_t v1 = مقادیر[q8[0]] | (مقادیر[q8[1]] << 8);
    uint16_t v2 = مقادیر[q8[2]] | (مقادیر[q8[3]] << 8);
    val1 = v1 | (v2 << 16);
    aux32 = (q4 >> 4) & 0x0f0f0f0f;
    v1 = مقادیر[q8[0]] | (مقادیر[q8[1]] << 8);
    v2 = مقادیر[q8[2]] | (مقادیر[q8[3]] << 8);
    val2 = v1 | (v2 << 16);
}

ساكن __dpct_inline__ sycl::int2 get_int_from_جدول_16(
    ثابت صحيح& q4, ثابت int8_t* جدول) {
  ثابت uint32_t* table32 = (ثابت uint32_t*)جدول;
  uint32_t tmp[2];
  ثابت uint32_t کم_بالا_selection_indices =
      (0x32103210 | ((q4 & 0x88888888) >> 1));
#رهنمود unroll
  لكل (uint32_t i = 0; i < 2; ++i) {
    ثابت uint32_t لغزش = 16 * i;

    ثابت uint32_t کم =
        dpct::byte_level_جایگشت(table32[0], table32[1], q4 >> لغزش);
    ثابت uint32_t بالا =
        dpct::byte_level_جایگشت(table32[2], table32[3], q4 >> لغزش);
    tmp[i] = dpct::byte_level_جایگشت(
        کم, بالا, کم_بالا_selection_indices >> لغزش);
  }
  أرجع sycl::int2(
      dpct::byte_level_جایگشت(tmp[0], tmp[1], 0x6420),
      dpct::byte_level_جایگشت(tmp[0], tmp[1], 0x7531));
}

// Four E2M1 codes (one per byte, bits 0..3) to their kvalues_mxfp4 int8 values. SWAR arithmetic
// replaces get_int_from_table_16 for MXFP4: dpct::byte_level_permute is emulated with 64-bit shifts,
// eight per int, which made the MXFP4 GEMV compute-bound on Intel GPUs.
// Magnitudes 0,1,2,3,4,6,8,12 = m + [m>=5] + [m>=6] + 3*[m>=7]; each byte stays below 256, so the
// byte-wise adds never carry. -0 (code 8) is left as 0 so the two's-complement +1 cannot carry either.
ساكن __dpct_inline__ صحيح mxfp4_codes_مقصد_int8(ثابت uint32_t x) {
    ثابت uint32_t m   = x & 0x07070707u;
    ثابت uint32_t ge5 = ((m + 0x03030303u) >> 3) & 0x01010101u;
    ثابت uint32_t ge6 = ((m + 0x02020202u) >> 3) & 0x01010101u;
    ثابت uint32_t ge7 = ((m + 0x01010101u) >> 3) & 0x01010101u;
    ثابت uint32_t mag = m + ge5 + ge6 + 3u * ge7;
    ثابت uint32_t nz  = ((mag + 0x7f7f7f7fu) >> 7) & 0x01010101u;
    ثابت uint32_t neg = (x >> 3) & nz & 0x01010101u;
    أرجع (صحيح) ((mag ^ (neg * 0xffu)) + neg);
}

// Same result as get_int_from_table_16(q4, kvalues_mxfp4): x = low nibbles, y = high nibbles.
ساكن __dpct_inline__ sycl::int2 get_int_from_mxfp4(ثابت صحيح q4) {
    أرجع sycl::int2(mxfp4_codes_مقصد_int8((uint32_t) q4 & 0x0f0f0f0fu),
                      mxfp4_codes_مقصد_int8(((uint32_t) q4 >> 4) & 0x0f0f0f0fu));
}

#تعریف_کن VDR_Q2_K_Q8_1_MMVQ 1

// contiguous v/x values
ساكن __dpct_inline__ عائم vec_dot_q2_K_q8_1_impl_mmvq(
    ثابت صحيح &v, ثابت صحيح *__restrict__ u, ثابت uint8_t *__restrict__ scales,
    ثابت sycl::half2 &dm2, ثابت عائم *__restrict__ d8) {

    عائم sumf_d = 0.0f;
    عائم sumf_m = 0.0f;

#رهنمود unroll
    لكل (صحيح i = 0; i < QR2_K; ++i) {
        ثابت صحيح sc = scales[2*i];

        ثابت صحيح vi = (v >> (2*i)) & 0x03030303;

        sumf_d +=
            d8[i] * (dpct::dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        صحيح m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] *
                  dpct::dp4a(
                      m, u[i],
                      0); // multiply constant q2_K part with sum of q8_1 values
    }

    ثابت sycl::float2 dm2f =
        dm2.convert<عائم, sycl::rounding_نما::automatic>();

    أرجع dm2f.x() * sumf_d - dm2f.y() * sumf_m;
}


#تعریف_کن VDR_Q3_K_Q8_1_MMVQ 1

// contiguous v/x values
ساكن __dpct_inline__ عائم vec_dot_q3_K_q8_1_impl_mmvq(
    ثابت صحيح &vl, ثابت صحيح &vh, ثابت صحيح *__restrict__ u,
    ثابت uint8_t *__restrict__ scales, ثابت صحيح &scale_جابه_جایی,
    ثابت عائم &d3, ثابت عائم *__restrict__ d8) {

    عائم sumf = 0.0f;

#رهنمود unroll
    لكل (صحيح i = 0; i < QR3_K; ++i) {
        ثابت صحيح isc = scale_جابه_جایی + 2*i;

        ثابت صحيح isc_کم = isc % (QK_K/32);
        ثابت صحيح sc_لغزش_کم = 4 * (isc / (QK_K/32));
        ثابت صحيح sc_کم  = (scales[isc_کم] >> sc_لغزش_کم) & 0xF;

        ثابت صحيح isc_بالا = isc % (QK_K/64);
        ثابت صحيح sc_لغزش_بالا = 2 * (isc / (QK_K/64));
        ثابت صحيح sc_بالا = ((scales[(QK_K/32) + isc_بالا] >> sc_لغزش_بالا) & 3) << 4;

        ثابت صحيح sc = (sc_کم | sc_بالا) - 32;

        ثابت صحيح vil = (vl >> (2*i)) & 0x03030303;

        ثابت صحيح vih = ((vh >> i) << 2) & 0x04040404;

        ثابت صحيح vi =
            dpct::vectorized_دودویی<sycl::char4>(vil, vih, dpct::sub_sat());

        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    أرجع d3 * sumf;
}

#تعریف_کن VDR_Q4_K_Q8_1_MMVQ 2

// contiguous v/x values
ساكن __dpct_inline__ عائم vec_dot_q4_K_q8_1_impl_vmmq(
    ثابت صحيح *__restrict__ v, ثابت صحيح *__restrict__ u,
    ثابت uint8_t *__restrict__ sc, ثابت uint8_t *__restrict__ m,
    ثابت sycl::half2 &dm4, ثابت عائم *__restrict__ d8) {

    عائم sumf_d = 0.0f;
    عائم sumf_m = 0.0f;

#رهنمود unroll
    لكل (صحيح i = 0; i < QR4_K; ++i) {
        ثابت صحيح v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        ثابت صحيح v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        ثابت صحيح dot1 =
            dpct::dp4a(v1i, u[2 * i + 1],
                       dpct::dp4a(v0i, u[2 * i + 0], 0)); // SIMD dot product
        ثابت صحيح dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 1],
                       dpct::dp4a(0x01010101, u[2 * i + 0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    ثابت sycl::float2 dm4f =
        dm4.convert<عائم, sycl::rounding_نما::automatic>();

    أرجع dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


#تعریف_کن VDR_Q5_K_Q8_1_MMVQ 2

// contiguous v/x values
ساكن __dpct_inline__ عائم vec_dot_q5_K_q8_1_impl_vmmq(
    ثابت صحيح *__restrict__ vl, ثابت صحيح *__restrict__ vh,
    ثابت صحيح *__restrict__ u, ثابت uint8_t *__restrict__ sc,
    ثابت uint8_t *__restrict__ m, ثابت sycl::half2 &dm5,
    ثابت عائم *__restrict__ d8) {

    عائم sumf_d = 0.0f;
    عائم sumf_m = 0.0f;

#رهنمود unroll
    لكل (صحيح i = 0; i < QR5_K; ++i) {
        ثابت صحيح vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        ثابت صحيح vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        ثابت صحيح vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        ثابت صحيح vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        ثابت صحيح v0i = vl0i | vh0i;
        ثابت صحيح v1i = vl1i | vh1i;

        ثابت صحيح dot1 =
            dpct::dp4a(v0i, u[2 * i + 0],
                       dpct::dp4a(v1i, u[2 * i + 1], 0)); // SIMD dot product
        ثابت صحيح dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 0],
                       dpct::dp4a(0x01010101, u[2 * i + 1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    ثابت sycl::float2 dm5f =
        dm5.convert<عائم, sycl::rounding_نما::automatic>();

    أرجع dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}


#تعریف_کن VDR_Q6_K_Q8_1_MMVQ 1

// contiguous v/x values
ساكن __dpct_inline__ عائم
vec_dot_q6_K_q8_1_impl_mmvq(ثابت صحيح &vl, ثابت صحيح &vh,
                            ثابت صحيح *__restrict__ u,
                            ثابت int8_t *__restrict__ scales, ثابت عائم &d,
                            ثابت عائم *__restrict__ d8) {
    أرجع vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
}

#تعریف_کن VDR_Q1_0_Q8_1_MMVQ 1
#تعریف_کن VDR_Q1_0_Q8_1_MMQ  4

ساكن __dpct_inline__ عائم
vec_dot_q1_0_q8_1(ثابت فراغ *__restrict__ vbq,
                  ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {

    ثابت بلوک_q1_0 * bq1_0 = (ثابت بلوک_q1_0 *) vbq;

    ثابت بلوک_q8_1 * bq8_1_chunk = bq8_1 + iqs;
    ثابت عائم        d1          = bq1_0->d;
    ثابت صحيح          v           = get_int_from_uint8_aligned(bq1_0->qs, iqs);

    صحيح vi_بایتها[8];
#رهنمود unroll
    لكل (صحيح j = 0; j < 8; ++j) {
        ثابت صحيح لغزش = j * 4;
        ثابت صحيح bits4 = (v >> لغزش) & 0x0F;
        ثابت صحيح b0    = (bits4 & 0x01) ? 1 : -1;
        ثابت صحيح b1    = (bits4 & 0x02) ? 1 : -1;
        ثابت صحيح b2    = (bits4 & 0x04) ? 1 : -1;
        ثابت صحيح b3    = (bits4 & 0x08) ? 1 : -1;
        vi_بایتها[j]     = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }

    صحيح sumi = 0;
#رهنمود unroll
    لكل (صحيح j = 0; j < 8; ++j) {
        ثابت صحيح u = get_int_from_int8_aligned(bq8_1_chunk->qs, j);
        sumi        = ggml_sycl_dp4a(vi_بایتها[j], u, sumi);
    }

    أرجع d1 * bq8_1_chunk->ds[0] * sumi;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

قالب <ggml_type T> بنية reorder_vec_dot_q_sycl {
    static_assert(T != T, "ggml_type for reorder vecdot not implemented");
};

// For some types the weight side of the dot product does not depend on the destination column, so a
// multi-column mul_mat_vec can unpack it once per block instead of once per column. Such a type adds
// load() and dot() next to operator() and opts in here. See reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>.
قالب <ggml_type T> بنية reorder_vec_dot_shared_weights {
    ساكن constexpr منطقي value = خطأ;
};

قالب <> بنية reorder_vec_dot_shared_weights<GGML_TYPE_Q4_K> {
    ساكن constexpr منطقي value = درست;
};

قالب <> بنية reorder_vec_dot_shared_weights<GGML_TYPE_Q5_K> {
    ساكن constexpr منطقي value = درست;
};

قالب <ggml_type T> بنية reorder_vec_dot_shared_activations {
    ساكن constexpr منطقي value = خطأ;
};

قالب <> بنية reorder_vec_dot_shared_activations<GGML_TYPE_Q4_K> {
    ساكن constexpr منطقي value = درست;
};

قالب <> بنية reorder_vec_dot_shared_activations<GGML_TYPE_Q5_K> {
    ساكن constexpr منطقي value = درست;
};

قالب <> بنية reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0> {
    ساكن constexpr ggml_type gtype = GGML_TYPE_Q4_0;

    استفاده q4_0_بلوک  = ggml_sycl_reordered::بلوک_q_t<GGML_TYPE_Q4_0>;
    استفاده q4_0_traits = نام_نوع q4_0_بلوک::traits;

    __dpct_inline__ عائم vec_dot_q4_0_q8_1_impl(ثابت صحيح * v, ثابت صحيح * u, ثابت عائم & d4, ثابت sycl::half2 & ds8) {
        صحيح sumi = 0;

#رهنمود unroll
        لكل (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            ثابت صحيح vi0 = (v[i] >> 0) & 0x0F0F0F0F;
            ثابت صحيح vi1 = (v[i] >> 4) & 0x0F0F0F0F;

            // SIMD dot product of quantized values
            sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
            sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
        }

        ثابت sycl::float2 ds8f = ds8.convert<عائم, sycl::rounding_نما::automatic>();

        // second part effectively subtracts 8 from each quant value
        أرجع d4 * (sumi * ds8f.x() - (8 * q4_0_traits::vdr_mmvq / q4_0_traits::qi) * ds8f.y());
    }

    __dpct_inline__ عائم عملگر()(ثابت فراغ * __restrict__ vbq, ثابت std::pair<صحيح, صحيح> ibx_جابه_جایی,
                                     ثابت std::pair<صحيح, صحيح> d_جابه_جایی, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحيح & iqs) {
        ثابت uint8_t * bq4_0 = static_cast<ثابت uint8_t *>(vbq) + ibx_جابه_جایی.اولی;
        ثابت ggml_half d = *(reinterpret_cast<ثابت ggml_half *>(static_cast<ثابت uint8_t *>(vbq) + d_جابه_جایی.اولی));
        صحيح             v[q4_0_traits::vdr_mmvq];
        صحيح             u[2 * q4_0_traits::vdr_mmvq];


#رهنمود unroll
        لكل (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            v[i]         = get_int_from_uint8(bq4_0, iqs + i);
            u[2 * i + 0] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
            u[2 * i + 1] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i + q4_0_traits::qi);
        }

        أرجع vec_dot_q4_0_q8_1_impl(v, u, d, *q8_1_ds);
    }
};

// Load four contiguous dwords per operand instead of loading each value separately.
بنية reorder_vec_dot_q8_0_wide {
    ساكن constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    استفاده q8_0_بلوک  = ggml_sycl_reordered::بلوک_q_t<GGML_TYPE_Q8_0>;
    استفاده q8_0_traits = نام_نوع q8_0_بلوک::traits;

    __dpct_inline__ عائم عملگر()(ثابت فراغ * __restrict__ vbq, ثابت std::pair<صحيح, صحيح> ibx_جابه_جایی,
                                     ثابت std::pair<صحيح, صحيح> d_جابه_جایی, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحيح & iqs) {
        static_assert(q8_0_traits::vdr_mmvq == 4, "the wide load moves exactly four dwords");

        ثابت uint8_t * پایه = static_cast<ثابت uint8_t *>(vbq);
        ثابت int8_t *  qs   = reinterpret_cast<ثابت int8_t *>(پایه + ibx_جابه_جایی.اولی);
        ثابت ggml_half d    = *reinterpret_cast<ثابت ggml_half *>(پایه + d_جابه_جایی.اولی);

        ثابت sycl::int4 v = *reinterpret_cast<ثابت sycl::int4 *>(qs + حجم(صحيح) * iqs);
        ثابت sycl::int4 u = *reinterpret_cast<ثابت sycl::int4 *>(q8_1_quant_ptr + حجم(صحيح) * iqs);

        صحيح sumi = 0;
#رهنمود unroll
        لكل (صحيح i = 0; i < 4; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        ثابت sycl::half2 ds_مقادیر = *q8_1_ds;
        أرجع static_cast<عائم>(d) * static_cast<عائم>(ds_مقادیر[0]) * sumi;
    }
};

قالب <> بنية reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0> {
    ساكن constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    استفاده q8_0_بلوک  = ggml_sycl_reordered::بلوک_q_t<GGML_TYPE_Q8_0>;
    استفاده q8_0_traits = نام_نوع q8_0_بلوک::traits;

    __dpct_inline__ عائم عملگر()(ثابت فراغ * __restrict__ vbq, ثابت std::pair<صحيح, صحيح> ibx_جابه_جایی,
                                     ثابت std::pair<صحيح, صحيح> d_جابه_جایی, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحيح & iqs,
                                     ثابت صحيح vdr = q8_0_traits::vdr_mmvq) {
        ثابت uint8_t * پایه = static_cast<ثابت uint8_t *>(vbq);
        ثابت int8_t *  qs   = reinterpret_cast<ثابت int8_t *>(پایه + ibx_جابه_جایی.اولی);
        ثابت ggml_half  d   = *reinterpret_cast<ثابت ggml_half *>(پایه + d_جابه_جایی.اولی);

        صحيح v[q8_0_traits::vdr_mmvq];
        صحيح u[q8_0_traits::vdr_mmvq];

#رهنمود unroll
        لكل (صحيح i = 0; i < vdr; ++i) {
            v[i] = get_int_from_int8(qs, iqs + i);
            u[i] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
        }

        صحيح sumi = 0;
#رهنمود unroll
        لكل (صحيح i = 0; i < vdr; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        ثابت sycl::half2 ds_مقادیر = *q8_1_ds;
        أرجع static_cast<عائم>(d) * static_cast<عائم>(ds_مقادیر[0]) * sumi;
    }
};

قالب <> بنية reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K> {
    ساكن constexpr ggml_type gtype = GGML_TYPE_Q2_K;

    استفاده q2_k_بلوک  = ggml_sycl_reordered::بلوک_q_t<GGML_TYPE_Q2_K>;
    استفاده q2_k_traits = نام_نوع q2_k_بلوک::traits;

    __dpct_inline__ عائم عملگر()(ثابت فراغ * __restrict__ vbq, ثابت std::pair<صحيح, صحيح> ibx_جابه_جایی,
                                     ثابت std::pair<صحيح, صحيح> d_جابه_جایی, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحيح & iqs) {
        ثابت uint8_t *    پایه   = static_cast<ثابت uint8_t *>(vbq);
        ثابت uint8_t *    qs     = پایه + ibx_جابه_جایی.اولی;
        ثابت uint8_t *    scales = پایه + d_جابه_جایی.اولی;
        ثابت ggml_half2 * dm     = reinterpret_cast<ثابت ggml_half2 *>(پایه + d_جابه_جایی.second);

        ثابت صحيح bq8_جابه_جایی   = QR2_K * (iqs / QI8_1);
        ثابت صحيح scale_جابه_جایی = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        ثابت صحيح v = get_int_from_uint8_aligned(qs, iqs);

        صحيح   u[QR2_K];
        عائم d8[QR2_K];

#رهنمود unroll
        لكل (صحيح i = 0; i < QR2_K; ++i) {
            ثابت int8_t * quant_پایه_ptr = q8_1_quant_ptr + (bq8_جابه_جایی + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_پایه_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_جابه_جایی + i))[0];
        }

        أرجع vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales + scale_جابه_جایی, *dm, d8);
    }
};

قالب <> بنية reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K> {
    ساكن constexpr ggml_type gtype = GGML_TYPE_Q3_K;

    استفاده q3_k_بلوک  = ggml_sycl_reordered::بلوک_q_t<GGML_TYPE_Q3_K>;
    استفاده q3_k_traits = نام_نوع q3_k_بلوک::traits;

    __dpct_inline__ عائم عملگر()(ثابت فراغ * __restrict__ vbq, ثابت std::pair<صحيح, صحيح> ibx_جابه_جایی,
                                     ثابت std::pair<صحيح, صحيح> d_جابه_جایی, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحيح & iqs) {
        ثابت uint8_t *  پایه   = static_cast<ثابت uint8_t *>(vbq);
        ثابت uint8_t *  qs     = پایه + ibx_جابه_جایی.اولی;
        ثابت uint8_t *  hmask  = پایه + ibx_جابه_جایی.second;
        ثابت uint8_t *  scales = پایه + d_جابه_جایی.اولی;
        ثابت ggml_half  d      = *reinterpret_cast<ثابت ggml_half *>(پایه + d_جابه_جایی.second);

        ثابت صحيح bq8_جابه_جایی   = QR3_K * (iqs / (QI3_K / 2));
        ثابت صحيح scale_جابه_جایی = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        ثابت صحيح vl = get_int_from_uint8(qs, iqs);
        ثابت صحيح vh = ~get_int_from_uint8(hmask, iqs % (QI3_K / 2)) >> bq8_جابه_جایی;

        صحيح   u[QR3_K];
        عائم d8[QR3_K];

#رهنمود unroll
        لكل (صحيح i = 0; i < QR3_K; ++i) {
            ثابت int8_t * quant_پایه_ptr = q8_1_quant_ptr + (bq8_جابه_جایی + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_پایه_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_جابه_جایی + i))[0];
        }

        أرجع vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, scales, scale_جابه_جایی, static_cast<عائم>(d), d8);
    }
};

ساكن مضمن عائم vec_dot_q4_K_q8_1_common(ثابت صحيح * __restrict__ q4, ثابت uint16_t * __restrict__ scales,
                                             ثابت ggml_half2 & dm, ثابت بلوک_q8_1 * __restrict__ bq8_1,
                                             ثابت صحيح &        iqs) {
    صحيح   v[2];
    صحيح   u[2 * QR4_K];
    عائم d8[QR4_K];

    v[0] = q4[0];
    v[1] = q4[4];

    uint16_t  aux[2];
    ثابت صحيح j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
    إذا (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } وإلا {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }

    ثابت uint8_t * sc = (ثابت uint8_t *) aux;
    ثابت uint8_t * m  = sc + 2;

    ثابت صحيح bq8_جابه_جایی = QR4_K * ((iqs / 2) / (QI8_1 / 2));

    لكل (صحيح i = 0; i < QR4_K; ++i) {
        ثابت بلوک_q8_1 * bq8i = bq8_1 + bq8_جابه_جایی + i;
        d8[i]                   = bq8i->ds[0];

        ثابت صحيح * q8 = (ثابت صحيح *) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0]   = q8[0];
        u[2 * i + 1]   = q8[4];
    }

    أرجع vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
}

قالب <> بنية reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K> {
    ساكن constexpr ggml_type gtype = GGML_TYPE_Q4_K;

    استفاده q4_k_بلوک  = ggml_sycl_reordered::بلوک_q_t<GGML_TYPE_Q4_K>;
    استفاده q4_k_traits = نام_نوع q4_k_بلوک::traits;

    بنية weights {
        صحيح        v[2];
        uint16_t   aux[2];
        ggml_half2 dm;
        صحيح        bq8_جابه_جایی;
    };

    بنية activations {
        صحيح   u[2 * QR4_K];
        عائم d8[QR4_K];
    };

    __dpct_inline__ ساكن weights load(ثابت فراغ * __restrict__ vbq, ثابت std::pair<صحيح, صحيح> ibx_جابه_جایی,
                                        ثابت std::pair<صحيح, صحيح> d_جابه_جایی, ثابت صحيح & iqs) {
        ثابت uint8_t *    پایه = static_cast<ثابت uint8_t *>(vbq);
        ثابت uint8_t *    qs   = پایه + ibx_جابه_جایی.اولی;
        ثابت uint8_t *    scs  = پایه + d_جابه_جایی.اولی;
        ثابت ggml_half2 * dms  = reinterpret_cast<ثابت ggml_half2 *>(پایه + d_جابه_جایی.second);

        weights w;
        w.bq8_جابه_جایی = QR4_K * ((iqs / 2) / (QI8_1 / 2));

        ثابت صحيح *      q4     = (ثابت صحيح *) (qs + 16 * w.bq8_جابه_جایی + 4 * ((iqs / 2) % 4));
        ثابت uint16_t * scales = (ثابت uint16_t *) scs;

        w.v[0] = q4[0];
        w.v[1] = q4[4];

        ثابت صحيح j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        إذا (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } وإلا {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        أرجع w;
    }

    __dpct_inline__ ساكن activations بارگیری_activations(ثابت int8_t * q8_1_quant_ptr,
                                                        ثابت sycl::half2 * q8_1_ds, ثابت صحيح & iqs) {
        activations a;
        ثابت صحيح bq8_جابه_جایی = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        لكل (صحيح i = 0; i < QR4_K; ++i) {
            ثابت int8_t * quant_پایه_ptr = q8_1_quant_ptr + (bq8_جابه_جایی + i) * QK8_1;
            sycl::half2    ds_مقادیر      = *(q8_1_ds + bq8_جابه_جایی + i);

            a.d8[i] = ds_مقادیر[0];

            ثابت صحيح * q8 = (ثابت صحيح *) quant_پایه_ptr + ((iqs / 2) % 4);
            a.u[2 * i + 0] = q8[0];
            a.u[2 * i + 1] = q8[4];
        }

        أرجع a;
    }

    __dpct_inline__ ساكن عائم apply(ثابت weights & w, ثابت activations & a) {
        ثابت uint8_t * sc = (ثابت uint8_t *) w.aux;
        ثابت uint8_t * m  = sc + 2;

        أرجع vec_dot_q4_K_q8_1_impl_vmmq(w.v, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ ساكن عائم dot(ثابت weights & w, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحيح & iqs) {
        ثابت خودکار a = بارگیری_activations(q8_1_quant_ptr, q8_1_ds, iqs);

        أرجع apply(w, a);
    }

    __dpct_inline__ عائم عملگر()(ثابت فراغ * __restrict__ vbq, ثابت std::pair<صحيح, صحيح> ibx_جابه_جایی,
                                     ثابت std::pair<صحيح, صحيح> d_جابه_جایی, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحيح & iqs) {
        أرجع dot(load(vbq, ibx_جابه_جایی, d_جابه_جایی, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

قالب <> بنية reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K> {
    ساكن constexpr ggml_type gtype = GGML_TYPE_Q5_K;

    استفاده q5_k_بلوک  = ggml_sycl_reordered::بلوک_q_t<GGML_TYPE_Q5_K>;
    استفاده q5_k_traits = نام_نوع q5_k_بلوک::traits;

    بنية weights {
        صحيح        vl[2];
        صحيح        vh[2];
        uint16_t   aux[2];
        ggml_half2 dm;
    };

    // same activation layout as Q4_K
    static_assert(QR5_K == QR4_K);
    استفاده activations = reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::activations;

    __dpct_inline__ ساكن weights load(ثابت فراغ * __restrict__ vbq, ثابت std::pair<صحيح, صحيح> ibx_جابه_جایی,
                                        ثابت std::pair<صحيح, صحيح> d_جابه_جایی, ثابت صحيح & iqs) {
        ثابت uint8_t *    پایه    = static_cast<ثابت uint8_t *>(vbq);
        ثابت uint8_t *    qs      = پایه + ibx_جابه_جایی.اولی;   // low 4 bits
        ثابت uint8_t *    qh_پایه = پایه + ibx_جابه_جایی.second;  // high bit
        ثابت uint8_t *    scs     = پایه + d_جابه_جایی.اولی;
        ثابت ggml_half2 * dms     = reinterpret_cast<ثابت ggml_half2 *>(پایه + d_جابه_جایی.second);

        ثابت صحيح        bq8_جابه_جایی = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        ثابت صحيح *      ql_ptr     = (ثابت صحيح *) (qs + 16 * bq8_جابه_جایی + 4 * ((iqs / 2) % 4));
        ثابت صحيح *      qh_ptr     = (ثابت صحيح *) (qh_پایه + 4 * ((iqs / 2) % 4));
        ثابت uint16_t * scales     = (ثابت uint16_t *) scs;

        weights w;
        w.vl[0] = ql_ptr[0];
        w.vl[1] = ql_ptr[4];

        w.vh[0] = qh_ptr[0] >> bq8_جابه_جایی;
        w.vh[1] = qh_ptr[4] >> bq8_جابه_جایی;

        ثابت صحيح j = (QR5_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        إذا (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } وإلا {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        أرجع w;
    }

    __dpct_inline__ ساكن activations بارگیری_activations(ثابت int8_t * q8_1_quant_ptr,
                                                        ثابت sycl::half2 * q8_1_ds, ثابت صحيح & iqs) {
        أرجع reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::بارگیری_activations(q8_1_quant_ptr, q8_1_ds, iqs);
    }

    __dpct_inline__ ساكن عائم apply(ثابت weights & w, ثابت activations & a) {
        ثابت uint8_t * sc = (ثابت uint8_t *) w.aux;
        ثابت uint8_t * m  = sc + 2;

        أرجع vec_dot_q5_K_q8_1_impl_vmmq(w.vl, w.vh, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ ساكن عائم dot(ثابت weights & w, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحيح & iqs) {
        أرجع apply(w, بارگیری_activations(q8_1_quant_ptr, q8_1_ds, iqs));
    }

    __dpct_inline__ عائم عملگر()(ثابت فراغ * __restrict__ vbq, ثابت std::pair<صحيح, صحيح> ibx_جابه_جایی,
                                     ثابت std::pair<صحيح, صحيح> d_جابه_جایی, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحيح & iqs) {
        أرجع dot(load(vbq, ibx_جابه_جایی, d_جابه_جایی, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

قالب <> بنية reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K> {
    ساكن constexpr ggml_type gtype = GGML_TYPE_Q6_K;

    استفاده q6_k_بلوک  = ggml_sycl_reordered::بلوک_q_t<GGML_TYPE_Q6_K>;
    استفاده q6_k_traits = نام_نوع q6_k_بلوک::traits;

    __dpct_inline__ عائم vec_dot_q6_K_q8_1_impl_mmvq(ثابت صحيح vl, ثابت صحيح vh, ثابت صحيح * __restrict__ u,
                                                      ثابت int8_t * __restrict__ scales, ثابت عائم d,
                                                      ثابت عائم * __restrict__ d8) {
        أرجع vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
    }

    __dpct_inline__ عائم عملگر()(ثابت فراغ * __restrict__ vbq, ثابت std::pair<صحيح, صحيح> ibx_جابه_جایی,
                     ثابت std::pair<صحيح, صحيح> d_جابه_جایی, ثابت int8_t * q8_1_quant_ptr, ثابت sycl::half2 * q8_1_ds,
                     ثابت صحيح iqs) {
        ثابت uint8_t *   پایه   = static_cast<ثابت uint8_t *>(vbq);
        ثابت uint8_t *   ql     = پایه + ibx_جابه_جایی.اولی;
        ثابت uint8_t *   qh     = پایه + ibx_جابه_جایی.second;
        ثابت int8_t *    scales = reinterpret_cast<ثابت int8_t *>(پایه + d_جابه_جایی.اولی);
        ثابت ggml_half * d      = (ثابت ggml_half *) (پایه + d_جابه_جایی.second);

        ثابت صحيح bq8_جابه_جایی   = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        ثابت صحيح scale_جابه_جایی = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        ثابت صحيح vh_لغزش     = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

        ثابت صحيح vl = get_int_from_uint8(ql, iqs);
        ثابت صحيح vh = get_int_from_uint8(qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_لغزش;

        ثابت int8_t * scs = scales + scale_جابه_جایی;

        ثابت صحيح u0 = get_int_from_int8_aligned(
            q8_1_quant_ptr + bq8_جابه_جایی * QK8_1, iqs % QI8_1);
        ثابت صحيح u1 = get_int_from_int8_aligned(
            q8_1_quant_ptr + (bq8_جابه_جایی + 2) * QK8_1, iqs % QI8_1);
        ثابت عائم d80 = (*(q8_1_ds + bq8_جابه_جایی + 0))[0];
        ثابت عائم d81 = (*(q8_1_ds + bq8_جابه_جایی + 2))[0];

        أرجع vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u0, u1, scs[0], scs[4], *d, d80, d81);
    }
};

قالب <> بنية reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4> {
    ساكن constexpr ggml_type gtype = GGML_TYPE_MXFP4;

    استفاده mxfp4_بلوک  = ggml_sycl_reordered::بلوک_q_t<GGML_TYPE_MXFP4>;
    استفاده mxfp4_traits = نام_نوع mxfp4_بلوک::traits;

    __dpct_inline__ عائم عملگر()(ثابت فراغ * __restrict__ vbq, ثابت std::pair<صحيح, صحيح> ibx_جابه_جایی,
                                     ثابت std::pair<صحيح, صحيح> d_جابه_جایی, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحيح & iqs) {
        static_assert(mxfp4_traits::vdr_mmvq == 2, "vector load assumes vdr_mmvq == 2");
        ثابت uint8_t * پایه = static_cast<ثابت uint8_t *>(vbq);

        // Reordered nibble blocks are 16 contiguous bytes and iqs is 0 or 2, so each lane's two
        // weight ints are one aligned 8-byte load (the AoS layout needed eight byte loads).
        ثابت sycl::int2 q4 = *reinterpret_cast<ثابت sycl::int2 *>(پایه + ibx_جابه_جایی.اولی + حجم(صحيح) * iqs);
        ثابت uint8_t    e  = پایه[d_جابه_جایی.اولی];

        // Low nibbles pair with q8_1 ints iqs..iqs+1, high nibbles with iqs+4..iqs+5.
        ثابت sycl::int2 u_lo = *reinterpret_cast<ثابت sycl::int2 *>(q8_1_quant_ptr + حجم(صحيح) * iqs);
        ثابت sycl::int2 u_hi = *reinterpret_cast<ثابت sycl::int2 *>(q8_1_quant_ptr + حجم(صحيح) * (iqs + 4));

        ثابت sycl::int2 v0 = get_int_from_mxfp4(q4.x());
        ثابت sycl::int2 v1 = get_int_from_mxfp4(q4.y());

        صحيح sumi = 0;
        sumi = ggml_sycl_dp4a(v0.x(), u_lo.x(), sumi);
        sumi = ggml_sycl_dp4a(v0.y(), u_hi.x(), sumi);
        sumi = ggml_sycl_dp4a(v1.x(), u_lo.y(), sumi);
        sumi = ggml_sycl_dp4a(v1.y(), u_hi.y(), sumi);

        ثابت عائم d = ggml_sycl_e8m0_مقصد_fp32(e) * 0.5f * static_cast<عائم>((*q8_1_ds)[0]);
        أرجع d * sumi;
    }
};
#تعریف_کن VDR_Q4_0_Q8_1_MMVQ 2
#تعریف_کن VDR_Q4_0_Q8_1_MMQ  4

#تعریف_کن VDR_Q2_0_Q8_1_MMVQ 1

قالب <صحيح vdr>
ساكن __dpct_inline__ عائم vec_dot_q2_0_q8_1_impl(
    ثابت صحيح * v,
    ثابت صحيح * u,
    ثابت عائم & d2,
    ثابت sycl::half2 & ds8) {
    صحيح sumi = 0;

#رهنمود unroll
    لكل (صحيح i = 0; i < vdr; ++i) {
#رهنمود unroll
        لكل (صحيح j = 0; j < 4; ++j) {
            ثابت uint8_t q = (uint8_t) ((uint32_t) v[i] >> (8 * j));

            // unpack 2-bit values to byte lanes (0..3), then apply zero-point
            // correction with ds8f.y() below, mirroring the q4_0 style.
            صحيح vi = 0;
            vi |= (((q >> 0) & 0x3) & 0xFF) << 0;
            vi |= (((q >> 2) & 0x3) & 0xFF) << 8;
            vi |= (((q >> 4) & 0x3) & 0xFF) << 16;
            vi |= (((q >> 6) & 0x3) & 0xFF) << 24;

            sumi = dpct::dp4a(vi, u[4 * i + j], sumi);
        }
    }

    ثابت sycl::float2 ds8f = ds8.convert<عائم, sycl::rounding_نما::automatic>();
    // q2_0 has zero-point 1. Scale ds8f.y() by processed-lane ratio,
    // consistent with q4_0's explicit zero-point subtraction style.
    أرجع d2 * (sumi * ds8f.x() - ((عائم) vdr / (عائم) QI2_0) * ds8f.y());
}

قالب <صحيح vdr>
ساكن __dpct_inline__ عائم vec_dot_q4_0_q8_1_impl(ثابت صحيح * v, ثابت صحيح * u, ثابت عائم & d4,
                                                    ثابت sycl::half2 & ds8) {
    صحيح sumi = 0;
#رهنمود unroll
    لكل (صحيح i = 0; i < vdr; ++i) {
        ثابت صحيح vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        ثابت صحيح vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

    ثابت sycl::float2 ds8f = ds8.convert<عائم, sycl::rounding_نما::automatic>();

    // second part effectively subtracts 8 from each quant value
    أرجع d4 * (sumi * ds8f.x() - (8 * vdr / QI4_0) * ds8f.y());
}

#تعریف_کن VDR_Q4_1_Q8_1_MMVQ 2
#تعریف_کن VDR_Q4_1_Q8_1_MMQ  4

قالب <صحيح vdr>
ساكن __dpct_inline__ عائم vec_dot_q4_1_q8_1_impl(ثابت صحيح *v, ثابت صحيح *u,
                                                    ثابت sycl::half2 &dm4,
                                                    ثابت sycl::half2 &ds8) {

    صحيح sumi = 0;

#رهنمود unroll
    لكل (صحيح i = 0; i < vdr; ++i) {
        ثابت صحيح vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        ثابت صحيح vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

#اگر_تعریف_شده GGML_SYCL_F16
    ثابت sycl::float2 tmp =
        (dm4 * ds8).convert<عائم, sycl::rounding_نما::automatic>();
    ثابت عائم d4d8 = tmp.x();
    ثابت عائم m4s8 = tmp.y();
#وإلا
    ثابت sycl::float2 dm4f =
        dm4.convert<عائم, sycl::rounding_نما::automatic>();
    ثابت sycl::float2 ds8f =
        ds8.convert<عائم, sycl::rounding_نما::automatic>();
    ثابت عائم d4d8 = dm4f.x() * ds8f.x();
    ثابت عائم m4s8 = dm4f.y() * ds8f.y();
#پایان_اگر // GGML_SYCL_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    أرجع sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#تعریف_کن VDR_Q5_0_Q8_1_MMVQ 2
#تعریف_کن VDR_Q5_0_Q8_1_MMQ  4

قالب <صحيح vdr>
ساكن __dpct_inline__ عائم
vec_dot_q5_0_q8_1_impl(ثابت صحيح *vl, ثابت صحيح *vh, ثابت صحيح *u,
                       ثابت عائم &d5, ثابت sycl::half2 &ds8) {
    صحيح sumi = 0;

#رهنمود unroll
    لكل (صحيح i = 0; i < vdr; ++i) {
        صحيح vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        صحيح vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

    ثابت sycl::float2 ds8f =
        ds8.convert<عائم, sycl::rounding_نما::automatic>();

    // second part effectively subtracts 16 from each quant value
    أرجع d5 * (sumi * ds8f.x() - (16 * vdr / QI5_0) * ds8f.y());
}

#تعریف_کن VDR_Q5_1_Q8_1_MMVQ 2
#تعریف_کن VDR_Q5_1_Q8_1_MMQ  4

قالب <صحيح vdr>
ساكن __dpct_inline__ عائم
vec_dot_q5_1_q8_1_impl(ثابت صحيح *vl, ثابت صحيح *vh, ثابت صحيح *u,
                       ثابت sycl::half2 &dm5, ثابت sycl::half2 &ds8) {

    صحيح sumi = 0;

#رهنمود unroll
    لكل (صحيح i = 0; i < vdr; ++i) {
        صحيح vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        صحيح vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

#اگر_تعریف_شده GGML_SYCL_F16
     ثابت sycl::float2 tmp =
        (dm5 * ds8).convert<عائم, sycl::rounding_نما::automatic>();
    ثابت عائم d5d8 = tmp.x();
    ثابت عائم m5s8 = tmp.y();


#وإلا
    ثابت sycl::float2 dm5f =
        dm5.convert<عائم, sycl::rounding_نما::automatic>();
    ثابت sycl::float2 ds8f =
        ds8.convert<عائم, sycl::rounding_نما::automatic>();
    ثابت عائم d5d8 = dm5f.x() * ds8f.x();
    ثابت عائم m5s8 = dm5f.y() * ds8f.y();
#پایان_اگر // GGML_SYCL_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    أرجع sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#تعریف_کن VDR_Q8_0_Q8_1_MMVQ 2
#تعریف_کن VDR_Q8_0_Q8_1_MMQ 8

قالب <صحيح vdr>
ساكن __dpct_inline__ عائم vec_dot_q8_0_q8_1_impl(ثابت صحيح *v, ثابت صحيح *u,
                                                    ثابت عائم &d8_0,
                                                    ثابت عائم &d8_1) {

    صحيح sumi = 0;

#رهنمود unroll
    لكل (صحيح i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

    أرجع d8_0*d8_1 * sumi;
}

قالب <نام_نوع T, صحيح vdr>
ساكن __dpct_inline__ T vec_dot_q8_0_q8_1_impl(ثابت صحيح * v, ثابت صحيح * u, ثابت T & d8_0, ثابت T & d8_1) {
    صحيح sumi = 0;

#رهنمود unroll
    لكل (صحيح i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_sycl_dp4a(v[i], u[i], sumi);
    }

    أرجع d8_0*d8_1 * ((T) sumi);
}

قالب <صحيح vdr>
ساكن __dpct_inline__ عائم vec_dot_q8_1_q8_1_impl(ثابت صحيح *v, ثابت صحيح *u,
                                                    ثابت sycl::half2 &dm8,
                                                    ثابت sycl::half2 &ds8) {

    صحيح sumi = 0;

#رهنمود unroll
    لكل (صحيح i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

#اگر_تعریف_شده GGML_SYCL_F16
    ثابت sycl::float2 tmp =
        (dm8 * ds8).convert<عائم, sycl::rounding_نما::automatic>();
    ثابت عائم d8d8 = tmp.x();
    ثابت عائم m8s8 = tmp.y();
#وإلا
    ثابت sycl::float2 dm8f =
        dm8.convert<عائم, sycl::rounding_نما::automatic>();
    ثابت sycl::float2 ds8f =
        ds8.convert<عائم, sycl::rounding_نما::automatic>();
    ثابت عائم d8d8 = dm8f.x() * ds8f.x();
    ثابت عائم m8s8 = dm8f.y() * ds8f.y();
#پایان_اگر // GGML_SYCL_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    أرجع sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

ساكن __dpct_inline__ عائم
vec_dot_q4_0_q8_1(ثابت فراغ *__restrict__ vbq,
                  ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {

    ثابت بلوک_q4_0 * bq4_0 = (ثابت بلوک_q4_0 *) vbq;

    صحيح v[VDR_Q4_0_Q8_1_MMVQ];
    صحيح u[2 * VDR_Q4_0_Q8_1_MMVQ];

#رهنمود unroll
    لكل (صحيح i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]         = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    أرجع vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

ساكن __dpct_inline__ عائم
vec_dot_q2_0_q8_1(ثابت فراغ *__restrict__ vbq,
                  ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {

    ثابت بلوک_q2_0 * bq2_0 = (ثابت بلوک_q2_0 *) vbq;

    صحيح v[2 * VDR_Q2_0_Q8_1_MMVQ];
    صحيح u[8 * VDR_Q2_0_Q8_1_MMVQ];

#رهنمود unroll
    لكل (صحيح i = 0; i < VDR_Q2_0_Q8_1_MMVQ; ++i) {
        ثابت صحيح پایه = 4 * (iqs + i);

        // Q2_0 has QK2_0 = 64 and uses 2 x QK8_1 blocks on the RHS.
        v[2 * i + 0] = get_int_from_uint8(bq2_0->qs, iqs + i);
        v[2 * i + 1] = get_int_from_uint8(bq2_0->qs, iqs + i + QI2_0);

        u[8 * i + 0] = get_int_from_int8_aligned(bq8_1[0].qs, پایه + 0);
        u[8 * i + 1] = get_int_from_int8_aligned(bq8_1[0].qs, پایه + 1);
        u[8 * i + 2] = get_int_from_int8_aligned(bq8_1[0].qs, پایه + 2);
        u[8 * i + 3] = get_int_from_int8_aligned(bq8_1[0].qs, پایه + 3);

        u[8 * i + 4] = get_int_from_int8_aligned(bq8_1[1].qs, پایه + 0);
        u[8 * i + 5] = get_int_from_int8_aligned(bq8_1[1].qs, پایه + 1);
        u[8 * i + 6] = get_int_from_int8_aligned(bq8_1[1].qs, پایه + 2);
        u[8 * i + 7] = get_int_from_int8_aligned(bq8_1[1].qs, پایه + 3);
    }

    ثابت عائم sum0 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + 0, u + 0, bq2_0->d, bq8_1[0].ds);
    ثابت عائم sum1 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + VDR_Q2_0_Q8_1_MMVQ, u + 4 * VDR_Q2_0_Q8_1_MMVQ, bq2_0->d, bq8_1[1].ds);
    أرجع sum0 + sum1;
}

ساكن __dpct_inline__ عائم
vec_dot_q4_1_q8_1(ثابت فراغ *__restrict__ vbq,
                  ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {

    ثابت بلوک_q4_1 * bq4_1 = (ثابت بلوک_q4_1 *) vbq;

    صحيح v[VDR_Q4_1_Q8_1_MMVQ];
    صحيح u[2*VDR_Q4_1_Q8_1_MMVQ];

#رهنمود unroll
    لكل (صحيح i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    أرجع vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

#تعریف_کن VDR_MXFP4_Q8_1_MMVQ 2
#تعریف_کن VDR_MXFP4_Q8_1_MMQ  4

ساكن __dpct_inline__ عائم vec_dot_mxfp4_q8_1(ثابت فراغ * __restrict__ vbq,
                                                ثابت بلوک_q8_1 * __restrict__ bq8_1,
                                                ثابت صحيح & iqs) {
    ثابت بلوک_mxfp4 * bq4 = (ثابت بلوک_mxfp4 *) vbq;

    ثابت صحيح * q8 = (ثابت صحيح *) bq8_1->qs + iqs;

    صحيح sumi = 0;
#رهنمود unroll
    لكل (صحيح l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        ثابت صحيح aux_q4 = get_int_b1(bq4->qs, iqs + l);
        ثابت sycl::int2 v      = get_int_from_mxfp4(aux_q4);
        sumi = ggml_sycl_dp4a(v.x(), q8[l + 0], sumi);
        sumi = ggml_sycl_dp4a(v.y(), q8[l + 4], sumi);
    }

    ثابت عائم d = ggml_sycl_e8m0_مقصد_fp32(bq4->e) * 0.5f * (bq8_1->ds)[0];
    أرجع d * sumi;
}

#تعریف_کن VDR_NVFP4_Q8_1_MMVQ 4
#تعریف_کن VDR_NVFP4_Q8_1_MMQ  8

ساكن __dpct_inline__ عائم vec_dot_nvfp4_q8_1(ثابت فراغ * __restrict__ vbq,
                                                ثابت بلوک_q8_1 * __restrict__ bq8_1,
                                                ثابت int32_t & iqs) {
    ثابت بلوک_nvfp4 * bq4 = (ثابت بلوک_nvfp4 *) vbq;
    عائم جمع = 0.0f;
#رهنمود unroll
    لكل (صحيح i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; i++) {
        ثابت int32_t iqs0 = iqs + 2*i;
        ثابت int32_t iqs1 = iqs0 + 1;
        ثابت int32_t is = iqs0 >> 1;
        ثابت sycl::int2   v0   = get_int_from_جدول_16(get_int_b4(bq4->qs, iqs0), kvalues_mxfp4);
        ثابت sycl::int2   v1   = get_int_from_جدول_16(get_int_b4(bq4->qs, iqs1), kvalues_mxfp4);
        ثابت بلوک_q8_1 * bq8 = bq8_1 + (is >> 1);
        ثابت int32_t i8 = ((is & 1) << 2);

        صحيح sumi = ggml_sycl_dp4a(v0.x(), get_int_b4(bq8->qs, i8 + 0), 0);
        sumi     = ggml_sycl_dp4a(v0.y(), get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi     = ggml_sycl_dp4a(v1.x(), get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi     = ggml_sycl_dp4a(v1.y(), get_int_b4(bq8->qs, i8 + 3), sumi);

        ثابت عائم d = ggml_sycl_ue4m3_مقصد_fp32(bq4->d[is]) * (bq8->ds)[0];
        جمع += d * عائم(sumi);
    }

    أرجع جمع;
}

ساكن __dpct_inline__ عائم
vec_dot_q5_0_q8_1(ثابت فراغ *__restrict__ vbq,
                  ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {

    ثابت بلوک_q5_0 * bq5_0 = (ثابت بلوک_q5_0 *) vbq;

    صحيح vl[VDR_Q5_0_Q8_1_MMVQ];
    صحيح vh[VDR_Q5_0_Q8_1_MMVQ];
    صحيح  u[2*VDR_Q5_0_Q8_1_MMVQ];

#رهنمود unroll
    لكل (صحيح i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    أرجع vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

ساكن __dpct_inline__ عائم
vec_dot_q5_1_q8_1(ثابت فراغ *__restrict__ vbq,
                  ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {

    ثابت بلوک_q5_1 * bq5_1 = (ثابت بلوک_q5_1 *) vbq;

    صحيح vl[VDR_Q5_1_Q8_1_MMVQ];
    صحيح vh[VDR_Q5_1_Q8_1_MMVQ];
    صحيح  u[2*VDR_Q5_1_Q8_1_MMVQ];

#رهنمود unroll
    لكل (صحيح i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    أرجع vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

ساكن __dpct_inline__ عائم
vec_dot_q8_0_q8_1(ثابت فراغ *__restrict__ vbq,
                  ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {

    ثابت بلوک_q8_0 * bq8_0 = (ثابت بلوک_q8_0 *) vbq;

    صحيح v[VDR_Q8_0_Q8_1_MMVQ];
    صحيح u[VDR_Q8_0_Q8_1_MMVQ];

#رهنمود unroll
    لكل (صحيح i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    أرجع vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                      bq8_1->ds[0]);
}

ساكن __dpct_inline__ عائم
vec_dot_q2_K_q8_1(ثابت فراغ *__restrict__ vbq,
                  ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {

    ثابت بلوک_q2_K * bq2_K = (ثابت بلوک_q2_K *) vbq;

    ثابت صحيح bq8_جابه_جایی = QR2_K * (iqs / QI8_1);
    ثابت صحيح scale_جابه_جایی = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    ثابت uint8_t * scales = bq2_K->scales + scale_جابه_جایی;

    ثابت صحيح v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    صحيح    u[QR2_K];
    عائم d8[QR2_K];

#رهنمود unroll
    لكل (صحيح i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_جابه_جایی + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_جابه_جایی + i].ds[0];
    }

    أرجع vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

ساكن __dpct_inline__ عائم
vec_dot_q3_K_q8_1(ثابت فراغ *__restrict__ vbq,
                  ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {

    ثابت بلوک_q3_K * bq3_K = (ثابت بلوک_q3_K *) vbq;

    ثابت صحيح bq8_جابه_جایی = QR3_K * (iqs / (QI3_K/2));
    ثابت صحيح scale_جابه_جایی = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    ثابت عائم d = bq3_K->d;

    ثابت صحيح vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    ثابت صحيح vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_جابه_جایی;

    صحيح    u[QR3_K];
    عائم d8[QR3_K];

#رهنمود unroll
    لكل (صحيح i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_جابه_جایی + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_جابه_جایی + i].ds[0];
    }

    أرجع vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_جابه_جایی, d, d8);
}

ساكن __dpct_inline__ عائم vec_dot_q4_K_q8_1(ثابت فراغ * __restrict__ vbq, ثابت بلوک_q8_1 * __restrict__ bq8_1,
                                               ثابت صحيح & iqs) {
#اگر_تعریف_نشده GGML_QKK_64

    ثابت بلوک_q4_K * bq4_K = (ثابت بلوک_q4_K *) vbq;

    ثابت صحيح        bq8_جابه_جایی = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    ثابت صحيح *      q4         = (ثابت صحيح *) (bq4_K->qs + 16 * bq8_جابه_جایی + 4 * ((iqs / 2) % 4));
    ثابت uint16_t * scales     = (ثابت uint16_t *) bq4_K->scales;

    أرجع vec_dot_q4_K_q8_1_common(q4, scales, bq4_K->dm, bq8_1, iqs);

#وإلا

#إذا __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    ثابت بلوک_q4_K * bq4_K = (ثابت بلوک_q4_K *) vbq;

    عائم sumf_d = 0.0f;
    عائم sumf_m = 0.0f;

    uint16_t aux16[2];
    ثابت uint8_t * s = (ثابت uint8_t *)aux16;

    ثابت uint16_t * a = (ثابت uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    ثابت عائم dall = bq4_K->dm[0];
    ثابت عائم dmin = bq4_K->dm[1];

    ثابت عائم d8_1 = bq8_1[0].ds[0];
    ثابت عائم d8_2 = bq8_1[1].ds[1];

    ثابت صحيح ui1 = *((ثابت صحيح *)bq8_1[0].qs + (iqs/2));
    ثابت صحيح ui2 = *((ثابت صحيح *)bq8_1[0].qs + (iqs/2) + 4);
    ثابت صحيح ui3 = *((ثابت صحيح *)bq8_1[1].qs + (iqs/2));
    ثابت صحيح ui4 = *((ثابت صحيح *)bq8_1[1].qs + (iqs/2) + 4);

    ثابت صحيح * q4 = (ثابت صحيح *)bq4_K->qs + (iqs/2);
    ثابت صحيح v1 = q4[0];
    ثابت صحيح v2 = q4[4];

    ثابت صحيح dot1 = dpct::dp4a(ui2, v2 & 0x0f0f0f0f, dpct::dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    ثابت صحيح dot2 = dpct::dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, dpct::dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    ثابت صحيح dot3 = dpct::dp4a(0x01010101, ui2, dpct::dp4a(0x01010101, ui1, 0));
    ثابت صحيح dot4 = dpct::dp4a(0x01010101, ui4, dpct::dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    أرجع dall * sumf_d - dmin * sumf_m;

#وإلا
    bad_arch();
#پایان_اگر // __SYCL_ARCH__ >= VER_4VEC

#پایان_اگر
}

ساكن __dpct_inline__ عائم
vec_dot_q5_K_q8_1(ثابت فراغ *__restrict__ vbq,
                  ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {

#اگر_تعریف_نشده GGML_QKK_64
    ثابت بلوک_q5_K * bq5_K = (ثابت بلوک_q5_K *) vbq;

    صحيح   vl[2];
    صحيح   vh[2];
    صحيح    u[2*QR5_K];
    عائم d8[QR5_K];

    ثابت صحيح bq8_جابه_جایی = QR5_K * ((iqs/2) / (QI8_1/2));
    ثابت صحيح * ql = (ثابت صحيح *)(bq5_K->qs + 16 * bq8_جابه_جایی + 4 * ((iqs/2)%4));
    ثابت صحيح * qh = (ثابت صحيح *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_جابه_جایی;
    vh[1] = qh[4] >> bq8_جابه_جایی;

    ثابت uint16_t * scales = (ثابت uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    ثابت صحيح j = bq8_جابه_جایی/2;
    إذا (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } وإلا {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    ثابت uint8_t * sc = (ثابت uint8_t *)aux;
    ثابت uint8_t * m  = sc + 2;

#رهنمود unroll
    لكل (صحيح i = 0; i < QR5_K; ++i) {
        ثابت بلوک_q8_1 * bq8i = bq8_1 + bq8_جابه_جایی + i;
        d8[i] = bq8i->ds[0];

        ثابت صحيح * q8 = (ثابت صحيح *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    أرجع vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#وإلا

#إذا __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    ثابت بلوک_q5_K * bq5_K = (ثابت بلوک_q5_K *) vbq;

    ثابت int8_t * s = bq5_K->scales;

    ثابت عائم d = bq5_K->d;

    ثابت عائم d8_1 = bq8_1[0].ds[0];
    ثابت عائم d8_2 = bq8_1[1].ds[1];

    ثابت صحيح ui1 = *((ثابت صحيح *)bq8_1[0].qs + (iqs/2));
    ثابت صحيح ui2 = *((ثابت صحيح *)bq8_1[0].qs + (iqs/2) + 4);
    ثابت صحيح ui3 = *((ثابت صحيح *)bq8_1[1].qs + (iqs/2));
    ثابت صحيح ui4 = *((ثابت صحيح *)bq8_1[1].qs + (iqs/2) + 4);

    ثابت صحيح * ql = (ثابت صحيح *)bq5_K->qs + (iqs/2);
    ثابت صحيح vl1 = ql[0];
    ثابت صحيح vl2 = ql[4];

    ثابت صحيح گام = 4 * (iqs/2); // 0, 4, 8, 12
    ثابت صحيح im = گام/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    ثابت صحيح درون = گام%8; // 0, 4, 0, 4
    ثابت صحيح vh = (*((ثابت صحيح *)(bq5_K->qh + درون))) >> im;

    ثابت صحيح v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    ثابت صحيح v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    ثابت صحيح v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    ثابت صحيح v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    ثابت عائم sumf_d = d8_1 * (dpct::dp4a(ui1, v1, 0) * s[0] + dpct::dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (dpct::dp4a(ui3, v3, 0) * s[2] + dpct::dp4a(ui4, v4, 0) * s[3]);

    أرجع d * sumf_d;

#وإلا
    bad_arch();
#پایان_اگر // __SYCL_ARCH__ >= VER_4VEC

#پایان_اگر
}

ساكن __dpct_inline__ عائم
vec_dot_q6_K_q8_1(ثابت فراغ *__restrict__ vbq,
                  ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {

    ثابت بلوک_q6_K * bq6_K = (ثابت بلوک_q6_K *) vbq;

    ثابت صحيح bq8_جابه_جایی = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    ثابت صحيح scale_جابه_جایی = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    ثابت صحيح vh_لغزش = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    ثابت صحيح vl = get_int_from_uint8(bq6_K->ql, iqs);
    ثابت صحيح vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_لغزش;

    ثابت int8_t * scales = bq6_K->scales + scale_جابه_جایی;

    ثابت صحيح u0 = get_int_from_int8_aligned(
        bq8_1[bq8_جابه_جایی + 0].qs, iqs % QI8_1);
    ثابت صحيح u1 = get_int_from_int8_aligned(
        bq8_1[bq8_جابه_جایی + 2].qs, iqs % QI8_1);
    ثابت عائم d80 = bq8_1[bq8_جابه_جایی + 0].ds[0];
    ثابت عائم d81 = bq8_1[bq8_جابه_جایی + 2].ds[0];

    أرجع vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u0, u1, scales[0], scales[4], bq6_K->d, d80, d81);
}


// NOTE: the VDR_IQ*_Q8_1_MMVQ values deliberately differ from the identically named CUDA constants
// (vecdotq.cuh): the SYCL kernels pair them with a halved qi (e.g. QI3_S/2), so the values are not
// interchangeable and must not be copied across backends.
#تعریف_کن VDR_IQ2_XXS_Q8_1_MMVQ 1

ساكن __dpct_inline__ uint32_t iq_sign_ماسک(ثابت uint32_t signs) {
    ثابت uint32_t بیتها =
        (signs & 0x01) | ((signs & 0x02) << 7) |
        ((signs & 0x04) << 14) | ((signs & 0x08) << 21);

    أرجع بیتها * 0xff;
}

ساكن __dpct_inline__ uint32_t iq_sign_بیتها(ثابت uint32_t encoded) {
    ثابت uint32_t signs = encoded & 127;

    أرجع signs | ((sycl::popcount(signs) & 1) << 7);
}

ساكن __dpct_inline__ عائم
vec_dot_iq2_xxs_q8_1(ثابت فراغ *__restrict__ vbq,
                     ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs,
                     ثابت uint64_t *iq2xxs_grid, ثابت uint8_t *ksigns_iq2xs,
                     ثابت uint8_t *kmask_iq2xs) {
#إذا QK_K == 256
    ثابت بلوک_iq2_xxs * bq2 = (ثابت بلوک_iq2_xxs *) vbq;

    ثابت صحيح ib32 = iqs;
    ثابت uint16_t * q2 = bq2->qs + 4*ib32;
    ثابت uint8_t  * aux8 = (ثابت uint8_t *)q2;
    ثابت int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = q2[2] | (q2[3] << 16);
    صحيح sumi = 0;
    لكل (صحيح l = 0; l < 4; ++l) {
        ثابت uint8_t * grid = (ثابت uint8_t *)(iq2xxs_grid + aux8[l]);
        ثابت uint8_t  signs = ksigns_iq2xs[aux32 & 127];
        لكل (صحيح j = 0; j < 8; ++j) {
            sumi += q8[j] * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
        aux32 >>= 7;
    }
    ثابت عائم d = (عائم)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.25f;
    أرجع d * sumi;
#وإلا
    أكد(خطأ);
    أرجع 0.f;
#پایان_اگر
}

#تعریف_کن VDR_IQ2_XS_Q8_1_MMVQ 1

ساكن __dpct_inline__ عائم
vec_dot_iq2_xs_q8_1(ثابت فراغ *__restrict__ vbq,
                    ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs,
                    ثابت uint64_t *iq2xs_grid) {
#إذا DPCT_COMPATIBILITY_موقت >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#إذا QK_K == 256
    ثابت بلوک_iq2_xs * bq2 = (ثابت بلوک_iq2_xs *) vbq;

    ثابت صحيح ib32 = iqs;
    ثابت uint16_t * q2 = bq2->qs + 4*ib32;
    ثابت int8_t   * q8 = bq8_1[ib32].qs;
    ثابت uint8_t ls1 = bq2->scales[ib32] & 0xf;
    ثابت uint8_t ls2 = bq2->scales[ib32] >>  4;
    صحيح sumi1 = 0;
#رهنمود unroll
    لكل (صحيح l = 0; l < 2; ++l) {
        ثابت uint64_t grid = iq2xs_grid[q2[l] & 511];
        ثابت uint32_t sign_بیتها = iq_sign_بیتها(q2[l] >> 9);
        ثابت uint32_t signs0 = iq_sign_ماسک(sign_بیتها);
        ثابت uint32_t signs1 = iq_sign_ماسک(sign_بیتها >> 4);
        ثابت صحيح grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        ثابت صحيح grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi1 = dpct::dp4a(grid_l, *((ثابت صحيح *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((ثابت صحيح *)q8 + 1), sumi1);
        q8 += 8;
    }
    صحيح sumi2 = 0;
#رهنمود unroll
    لكل (صحيح l = 2; l < 4; ++l) {
        ثابت uint64_t grid = iq2xs_grid[q2[l] & 511];
        ثابت uint32_t sign_بیتها = iq_sign_بیتها(q2[l] >> 9);
        ثابت uint32_t signs0 = iq_sign_ماسک(sign_بیتها);
        ثابت uint32_t signs1 = iq_sign_ماسک(sign_بیتها >> 4);
        ثابت صحيح grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        ثابت صحيح grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi2 = dpct::dp4a(grid_l, *((ثابت صحيح *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((ثابت صحيح *)q8 + 1), sumi2);
        q8 += 8;
    }
    ثابت عائم d = (عائم)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    أرجع d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#وإلا
    أكد(خطأ);
    أرجع 0.f;
#پایان_اگر
#وإلا
    أكد(خطأ);
    أرجع 0.f;
#پایان_اگر
}

#تعریف_کن VDR_IQ2_S_Q8_1_MMVQ 1

ساكن __dpct_inline__ عائم
vec_dot_iq2_s_q8_1(ثابت فراغ *__restrict__ vbq,
                   ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {
#إذا QK_K == 256
    ثابت بلوک_iq2_s * bq2 = (ثابت بلوک_iq2_s *) vbq;

    ثابت صحيح ib32 = iqs;
    ثابت int8_t  * q8 = bq8_1[ib32].qs;
    ثابت uint8_t * signs = bq2->qs + QK_K/8 + 4*ib32;
    ثابت uint8_t ls1 = bq2->scales[ib32] & 0xf;
    ثابت uint8_t ls2 = bq2->scales[ib32] >>  4;
    صحيح sumi1 = 0;
    لكل (صحيح l = 0; l < 2; ++l) {
        ثابت uint32_t * grid = (ثابت uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        ثابت uint32_t signs0 = dpct::vectorized_دودویی<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ثابت uint32_t signs1 = dpct::vectorized_دودویی<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ثابت صحيح grid_l = dpct::vectorized_دودویی<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        ثابت صحيح grid_h = dpct::vectorized_دودویی<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi1 = dpct::dp4a(grid_l, *((ثابت صحيح *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((ثابت صحيح *)q8 + 1), sumi1);
        q8 += 8;
    }
    صحيح sumi2 = 0;
    لكل (صحيح l = 2; l < 4; ++l) {
        ثابت uint32_t * grid = (ثابت uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        ثابت uint32_t signs0 = dpct::vectorized_دودویی<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ثابت uint32_t signs1 = dpct::vectorized_دودویی<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ثابت صحيح grid_l = dpct::vectorized_دودویی<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        ثابت صحيح grid_h = dpct::vectorized_دودویی<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi2 = dpct::dp4a(grid_l, *((ثابت صحيح *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((ثابت صحيح *)q8 + 1), sumi2);
        q8 += 8;
    }
    ثابت عائم d = (عائم)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    أرجع d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#وإلا
    أكد(خطأ);
#پایان_اگر
}

#تعریف_کن VDR_IQ3_XXS_Q8_1_MMVQ 1

ساكن __dpct_inline__ عائم
vec_dot_iq3_xxs_q8_1(ثابت فراغ *__restrict__ vbq,
                     ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs,
                     ثابت uint32_t *iq3xxs_grid) {
#إذا DPCT_COMPATIBILITY_موقت >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#إذا QK_K == 256
    ثابت بلوک_iq3_xxs * bq2 = (ثابت بلوک_iq3_xxs *) vbq;

    ثابت صحيح ib32 = iqs;
    ثابت uint8_t  * q3 = bq2->qs + 8*ib32;
    ثابت uint16_t * gas = (ثابت uint16_t *)(bq2->qs + QK_K/4) + 2*ib32;
    ثابت int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = gas[0] | (gas[1] << 16);
    صحيح sumi = 0;
#رهنمود unroll
    لكل (صحيح l = 0; l < 4; ++l) {
        ثابت uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
        ثابت uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
        ثابت uint32_t sign_بیتها = iq_sign_بیتها(aux32);
        ثابت uint32_t signs0 = iq_sign_ماسک(sign_بیتها);
        ثابت uint32_t signs1 = iq_sign_ماسک(sign_بیتها >> 4);
        ثابت صحيح grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        ثابت صحيح grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((ثابت صحيح *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((ثابت صحيح *)q8 + 1), sumi);
        q8 += 8;
        aux32 >>= 7;
    }
    ثابت عائم d = (عائم)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.5f;
    أرجع d * sumi;
#وإلا
    أكد(خطأ);
    أرجع 0.f;
#پایان_اگر
#وإلا
    أكد(خطأ);
    أرجع 0.f;
#پایان_اگر
}

قالب <> بنية reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_XXS> {
    ساكن constexpr ggml_type gtype = GGML_TYPE_IQ3_XXS;

    __dpct_inline__ عائم عملگر()(ثابت فراغ * __restrict__ vbq, ثابت std::pair<صحيح, صحيح> ibx_جابه_جایی,
                                     ثابت std::pair<صحيح, صحيح> d_جابه_جایی, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحيح & iqs) {
#إذا DPCT_COMPATIBILITY_موقت >= MIN_CC_DP4A && QK_K == 256
        ثابت uint8_t * پایه = static_cast<ثابت uint8_t *>(vbq);
        ثابت uint8_t * qs   = پایه + ibx_جابه_جایی.اولی;
        ثابت uint8_t * q3   = qs + 8 * iqs;
        ثابت uint16_t * gas = reinterpret_cast<ثابت uint16_t *>(qs + QK_K / 4) + 2 * iqs;
        ثابت int8_t * q8    = q8_1_quant_ptr + iqs * QK8_1;
        uint32_t aux32       = gas[0] | (gas[1] << 16);

        صحيح sumi = 0;
#رهنمود unroll
        لكل (صحيح l = 0; l < 4; ++l) {
            ثابت uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
            ثابت uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
            ثابت uint32_t sign_بیتها = iq_sign_بیتها(aux32);
            ثابت uint32_t signs0 = iq_sign_ماسک(sign_بیتها);
            ثابت uint32_t signs1 = iq_sign_ماسک(sign_بیتها >> 4);
            ثابت صحيح grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            ثابت صحيح grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((ثابت صحيح *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((ثابت صحيح *) q8 + 1), sumi);
            q8 += 8;
            aux32 >>= 7;
        }

        ثابت ggml_half d = *reinterpret_cast<ثابت ggml_half *>(پایه + d_جابه_جایی.اولی);
        أرجع (عائم) d * (0.5f + aux32) * q8_1_ds[iqs][0] * 0.5f * sumi;
#وإلا
        GGML_استفادهنشده(vbq);
        GGML_استفادهنشده(ibx_جابه_جایی);
        GGML_استفادهنشده(d_جابه_جایی);
        GGML_استفادهنشده(q8_1_quant_ptr);
        GGML_استفادهنشده(q8_1_ds);
        GGML_استفادهنشده(iqs);
        GGML_ABORT("IQ3_XXS reorder vec dot requires dp4a and QK_K == 256");
#پایان_اگر
    }
};

#تعریف_کن VDR_IQ3_S_Q8_1_MMVQ 1

ساكن __dpct_inline__ عائم
vec_dot_iq3_s_q8_1(ثابت فراغ *__restrict__ vbq,
                   ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs,
                   ثابت uint32_t *iq3s_grid) {
#إذا QK_K == 256
    ثابت بلوک_iq3_s * bq2 = (ثابت بلوک_iq3_s *) vbq;

    ثابت صحيح ib32 = iqs;
    ثابت uint8_t  * qs = bq2->qs + 8*ib32;
    ثابت int8_t   * q8 = bq8_1[ib32].qs;
    صحيح sumi = 0;
    لكل (صحيح l = 0; l < 4; ++l) {
        ثابت uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((bq2->qh[ib32] << (8 - 2*l)) & 256)];
        ثابت uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((bq2->qh[ib32] << (7 - 2*l)) & 256)];
        ثابت uint32_t signs = bq2->signs[4 * ib32 + l];
        ثابت uint32_t signs0 = iq_sign_ماسک(signs);
        ثابت uint32_t signs1 = iq_sign_ماسک(signs >> 4);
        ثابت صحيح grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        ثابت صحيح grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((ثابت صحيح *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((ثابت صحيح *)q8 + 1), sumi);
        q8 += 8;
    }
    ثابت عائم d =
        (عائم)bq2->d *
        (1 + 2 * ((bq2->scales[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf)) *
        bq8_1[ib32].ds[0];
    أرجع d * sumi;
#وإلا
    أكد(خطأ);
#پایان_اگر
}

قالب <> بنية reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_S> {
    ساكن constexpr ggml_type gtype = GGML_TYPE_IQ3_S;

    __dpct_inline__ عائم عملگر()(ثابت فراغ * __restrict__ vbq, ثابت std::pair<صحيح, صحيح> ibx_جابه_جایی,
                                     ثابت std::pair<صحيح, صحيح> d_جابه_جایی, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحيح & iqs) {
#إذا QK_K == 256
        ثابت uint8_t * پایه     = static_cast<ثابت uint8_t *>(vbq);
        ثابت uint8_t * qs       = پایه + ibx_جابه_جایی.اولی + 8 * iqs;
        ثابت uint8_t * qh       = پایه + ibx_جابه_جایی.second;
        ثابت uint8_t * signs    = پایه + d_جابه_جایی.اولی + 4 * iqs;
        ثابت uint8_t * metadata = پایه + d_جابه_جایی.second;
        ثابت ggml_half d        = *reinterpret_cast<ثابت ggml_half *>(metadata);
        ثابت uint8_t * scales   = metadata + حجم(ggml_half);
        ثابت int8_t * q8        = q8_1_quant_ptr + iqs * QK8_1;

        صحيح sumi = 0;
#رهنمود unroll
        لكل (صحيح l = 0; l < 4; ++l) {
            ثابت uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((qh[iqs] << (8 - 2*l)) & 256)];
            ثابت uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((qh[iqs] << (7 - 2*l)) & 256)];
            ثابت uint32_t signs0 = iq_sign_ماسک(signs[l]);
            ثابت uint32_t signs1 = iq_sign_ماسک(signs[l] >> 4);
            ثابت صحيح grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            ثابت صحيح grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((ثابت صحيح *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((ثابت صحيح *) q8 + 1), sumi);
            q8 += 8;
        }

        ثابت عائم scale = 1 + 2 * ((scales[iqs / 2] >> (4 * (iqs % 2))) & 0xf);
        أرجع (عائم) d * scale * q8_1_ds[iqs][0] * sumi;
#وإلا
        GGML_استفادهنشده(vbq);
        GGML_استفادهنشده(ibx_جابه_جایی);
        GGML_استفادهنشده(d_جابه_جایی);
        GGML_استفادهنشده(q8_1_quant_ptr);
        GGML_استفادهنشده(q8_1_ds);
        GGML_استفادهنشده(iqs);
        GGML_ABORT("IQ3_S reorder vec dot not supported for QK_K != 256");
#پایان_اگر
    }
};

#تعریف_کن VDR_IQ1_S_Q8_1_MMVQ 1

ساكن __dpct_inline__ عائم
vec_dot_iq1_s_q8_1(ثابت فراغ *__restrict__ vbq,
                   ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs,
                   ثابت uint32_t *iq1s_grid_gpu) {
#إذا QK_K == 256
    ثابت بلوک_iq1_s * bq1 = (ثابت بلوک_iq1_s *) vbq;

    ثابت صحيح ib32 = iqs;
    صحيح sumi = 0;
    ثابت صحيح * q8 = (ثابت صحيح *)bq8_1[ib32].qs;
    لكل (صحيح l = 0; l < 4; ++l) {
        ثابت صحيح * grid = (ثابت صحيح *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[ib32] >> 3*l) & 7) << 8)));
        صحيح grid0 = grid[0] & 0x0f0f0f0f;
        صحيح grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi = dpct::dp4a(q8[2 * l + 1], grid1,
                          dpct::dp4a(q8[2 * l + 0], grid0, sumi));
    }

    ثابت عائم delta = bq1->qh[ib32] & 0x8000 ? -1-IQ1S_DELTA : -1+IQ1S_DELTA;
    ثابت عائم d1q = (عائم)bq1->d * (2*((bq1->qh[ib32] >> 12) & 7) + 1);
    ثابت عائم d = d1q * bq8_1[ib32].ds[0];
    ثابت عائم m = d1q * bq8_1[ib32].ds[1];
    أرجع d * sumi + m * delta;
#وإلا
    أكد(خطأ);
#پایان_اگر
}

#تعریف_کن VDR_IQ1_M_Q8_1_MMVQ 1

ساكن __dpct_inline__ عائم
vec_dot_iq1_m_q8_1(ثابت فراغ *__restrict__ vbq,
                   ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {
#إذا QK_K == 256
    ثابت بلوک_iq1_m * bq1 = (ثابت بلوک_iq1_m *) vbq;

    ثابت صحيح ib32 = iqs;
    صحيح   sumi[2] = {0, 0};
    عائم sumf[2] = {0.f, 0.f};

    ثابت صحيح * q8 = (ثابت صحيح *)bq8_1[ib32].qs;
    لكل (صحيح l = 0; l < 4; ++l) {
        ثابت صحيح * grid = (ثابت صحيح *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 7) << 8)));
        صحيح grid0 = grid[0] & 0x0f0f0f0f;
        صحيح grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi[l / 2] = dpct::dp4a(q8[2 * l + 1], grid1,
                                 dpct::dp4a(q8[2 * l + 0], grid0, sumi[l / 2]));
        ثابت عائم delta = (bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 0x08 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA;
        ثابت صحيح sumy = dpct::dp4a(q8[2 * l + 1], 0x01010101,
                                    dpct::dp4a(q8[2 * l + 0], 0x01010101, 0));
        sumf[l/2] += delta*sumy;
    }

    iq1m_scale_t scale;
    ثابت uint16_t * sc = (ثابت uint16_t *)bq1->scales;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    ثابت عائم d = (عائم)scale.f16 * bq8_1[ib32].ds[0];
    أرجع d * ((sumi[0] + sumf[0]) * (2*((sc[ib32/2] >> 6*(ib32%2)) & 0x7) + 1) + (sumi[1] + sumf[1]) * (2*((sc[ib32/2] >> (6*(ib32%2)+3)) & 0x7) + 1));
#وإلا
    أكد(خطأ);
#پایان_اگر
}


#تعریف_کن VDR_IQ4_NL_Q8_1_MMVQ 2

ساكن __dpct_inline__ عائم
vec_dot_iq4_nl_q8_1(ثابت فراغ *__restrict__ vbq,
                    ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {

    ثابت بلوک_iq4_nl * bq = (ثابت بلوک_iq4_nl *) vbq;

    ثابت uint16_t * q4 = (ثابت uint16_t *)bq->qs + 2*iqs;
    ثابت int32_t  * q8 = (ثابت int32_t  *)bq8_1->qs + iqs;

    ثابت uint8_t * مقادیر = (ثابت uint8_t *)kvalues_iq4nl;

    صحيح v1, v2;
    صحيح sumi1 = 0, sumi2 = 0;
    لكل (صحيح l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        ثابت uint32_t aux = q4[2*l] | (q4[2*l+1] << 16);
        get_int_from_جدول_16(aux, مقادیر, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[l + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[l + 4], sumi2);
    }

    ثابت عائم d = (عائم)bq->d * bq8_1->ds[0];
    أرجع d * (sumi1 + sumi2);
}


#تعریف_کن VDR_IQ4_XS_Q8_1_MMVQ 1

ساكن __dpct_inline__ عائم
vec_dot_iq4_xs_q8_1(ثابت فراغ *__restrict__ vbq,
                    ثابت بلوک_q8_1 *__restrict__ bq8_1, ثابت صحيح &iqs) {

#إذا QK_K == 256
    ثابت بلوک_iq4_xs * bq4 = (ثابت بلوک_iq4_xs *) vbq;
    ثابت uint8_t * مقادیر = (ثابت uint8_t *)kvalues_iq4nl;

    // iqs is 0...7
    ثابت صحيح ib32 = iqs;
    ثابت int32_t  * q8 = (ثابت صحيح *)bq8_1[ib32].qs;
    ثابت uint32_t * q4 = (ثابت uint32_t *)bq4->qs + 4*ib32;
    ثابت int8_t ls = ((bq4->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((bq4->scales_h >> 2*ib32) & 3) << 4);
    ثابت عائم d = (عائم)bq4->d * (ls - 32) * bq8_1[ib32].ds[0];
    صحيح v1, v2;
    صحيح sumi1 = 0, sumi2 = 0;
    لكل (صحيح j = 0; j < 4; ++j) {
        get_int_from_جدول_16(q4[j], مقادیر, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[j + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[j + 4], sumi2);
    }
    أرجع d * (sumi1 + sumi2);
#وإلا
    أكد(خطأ);
#پایان_اگر
}

#پایان_اگر // GGML_SYCL_VECDOTQ_HPP
