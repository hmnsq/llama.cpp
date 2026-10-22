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

#اگر_غیر_تعریف_شدہ GGML_SYCL_VECDOTQ_HPP
#تعریف_کرو GGML_SYCL_VECDOTQ_HPP

#شامل "dpct/helper.hpp"
#شامل "ggml.h"
#شامل "type.hpp"
#شامل "quants.hpp"

قسمتعریف اعشاری (*vec_dot_q_sycl_t)(ثابت خالی * __restrict__ vbq, ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                  ثابت صحیح & iqs);

ساکن __dpct_inline__ صحیح get_int_b1(ثابت خالی * x, ثابت صحیح & i32) {
    ثابت uint8_t * x8 = (ثابت uint8_t *) x;

    صحیح x32  = x8[4*i32 + 0] <<  0;
    x32     |= x8[4*i32 + 1] <<  8;
    x32     |= x8[4*i32 + 2] << 16;
    x32     |= x8[4*i32 + 3] << 24;

    واپس x32;
}

ساکن __dpct_inline__ صحیح get_int_b2(ثابت خالی * x, ثابت صحیح & i32) {
    ثابت uint16_t * x16 = (ثابت uint16_t *) x; // assume at least 2 byte alignment

    صحیح x32  = x16[2*i32 + 0] <<  0;
    x32     |= x16[2*i32 + 1] << 16;

    واپس x32;
}

ساکن __dpct_inline__ صحیح get_int_b4(ثابت خالی * x, ثابت صحیح & i32) {
    واپس ((ثابت صحیح *) x)[i32]; // assume at least 4 byte alignment
}

ساکن __dpct_inline__ صحیح get_int_from_int8(ثابت int8_t* x8, ثابت صحیح& i32) {
  ثابت uint16_t* x16 =
      (ثابت uint16_t*)(x8 + حجم(صحیح) * i32); // assume at least 2 byte
                                                 // alignment

  صحیح x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  واپس x32;
}

ساکن __dpct_inline__ صحیح get_int_from_uint8(
    ثابت uint8_t* x8,
    ثابت صحیح& i32) {
  ثابت uint16_t* x16 =
      (ثابت uint16_t*)(x8 + حجم(صحیح) * i32); // assume at least 2 byte
                                                 // alignment

  صحیح x32 = 0;
  x32 |= x16[0] << 0;
  x32 |= x16[1] << 16;

  واپس x32;
}

ساکن __dpct_inline__ صحیح get_int_from_int8_aligned(
    ثابت int8_t* x8,
    ثابت صحیح& i32) {
  واپس *(
      (ثابت صحیح*)(x8 + حجم(صحیح) * i32)); // assume at least 4 byte alignment
}

ساکن __dpct_inline__ صحیح get_int_from_uint8_aligned(
    ثابت uint8_t* x8,
    ثابت صحیح& i32) {
  واپس *(
      (ثابت صحیح*)(x8 + حجم(صحیح) * i32)); // assume at least 4 byte alignment
}

ساکن __dpct_inline__ صحیح byte_sub_4(ثابت صحیح a, ثابت صحیح b) {
  ثابت uint32_t ua = static_cast<uint32_t>(a);
  ثابت uint32_t ub = static_cast<uint32_t>(b);
  واپس static_cast<صحیح>(((ua | 0x80808080u) - ub) ^ 0x80808080u);
}

ساکن __dpct_inline__ اعشاری vec_dot_q6_K_q8_1_impl_mmvq_scalar(
    ثابت صحیح vl, ثابت صحیح vh, ثابت صحیح u0, ثابت صحیح u1, ثابت int8_t sc0,
    ثابت int8_t sc1, ثابت اعشاری d, ثابت اعشاری d80, ثابت اعشاری d81) {
    static_assert(QR6_K == 2, "q6_K MMVQ scalar fast path assumes QR6_K == 2");

    ثابت صحیح vil0 = (vl >> 0) & 0x0F0F0F0F;
    ثابت صحیح vih0 = ((vh >> 0) << 4) & 0x30303030;
    ثابت صحیح vi0 = byte_sub_4(vil0 | vih0, 0x20202020);

    ثابت صحیح vil1 = (vl >> 4) & 0x0F0F0F0F;
    ثابت صحیح vih1 = ((vh >> 4) << 4) & 0x30303030;
    ثابت صحیح vi1 = byte_sub_4(vil1 | vih1, 0x20202020);

    ثابت اعشاری sumf =
        d80 * (dpct::dp4a(vi0, u0, 0) * sc0) +
        d81 * (dpct::dp4a(vi1, u1, 0) * sc1);

    واپس d * sumf;
}

ساکن __dpct_inline__ خالی get_int_from_جدول_16(ثابت uint32_t &q4,
                                                  ثابت uint8_t *اقدار,
                                                  صحیح &val1, صحیح &val2) {

    uint32_t aux32; ثابت uint8_t * q8 = (ثابت uint8_t *)&aux32;
    aux32 = q4 & 0x0f0f0f0f;
    uint16_t v1 = اقدار[q8[0]] | (اقدار[q8[1]] << 8);
    uint16_t v2 = اقدار[q8[2]] | (اقدار[q8[3]] << 8);
    val1 = v1 | (v2 << 16);
    aux32 = (q4 >> 4) & 0x0f0f0f0f;
    v1 = اقدار[q8[0]] | (اقدار[q8[1]] << 8);
    v2 = اقدار[q8[2]] | (اقدار[q8[3]] << 8);
    val2 = v1 | (v2 << 16);
}

ساکن __dpct_inline__ sycl::int2 get_int_from_جدول_16(
    ثابت صحیح& q4, ثابت int8_t* جدول) {
  ثابت uint32_t* table32 = (ثابت uint32_t*)جدول;
  uint32_t tmp[2];
  ثابت uint32_t کم_بلند_selection_indices =
      (0x32103210 | ((q4 & 0x88888888) >> 1));
#ہدایت unroll
  برائے (uint32_t i = 0; i < 2; ++i) {
    ثابت uint32_t سرکاؤ = 16 * i;

    ثابت uint32_t کم =
        dpct::byte_level_ترتیب_بدلو(table32[0], table32[1], q4 >> سرکاؤ);
    ثابت uint32_t بلند =
        dpct::byte_level_ترتیب_بدلو(table32[2], table32[3], q4 >> سرکاؤ);
    tmp[i] = dpct::byte_level_ترتیب_بدلو(
        کم, بلند, کم_بلند_selection_indices >> سرکاؤ);
  }
  واپس sycl::int2(
      dpct::byte_level_ترتیب_بدلو(tmp[0], tmp[1], 0x6420),
      dpct::byte_level_ترتیب_بدلو(tmp[0], tmp[1], 0x7531));
}

// Four E2M1 codes (one per byte, bits 0..3) to their kvalues_mxfp4 int8 values. SWAR arithmetic
// replaces get_int_from_table_16 for MXFP4: dpct::byte_level_permute is emulated with 64-bit shifts,
// eight per int, which made the MXFP4 GEMV compute-bound on Intel GPUs.
// Magnitudes 0,1,2,3,4,6,8,12 = m + [m>=5] + [m>=6] + 3*[m>=7]; each byte stays below 256, so the
// byte-wise adds never carry. -0 (code 8) is left as 0 so the two's-complement +1 cannot carry either.
ساکن __dpct_inline__ صحیح mxfp4_codes_مقصد_int8(ثابت uint32_t x) {
    ثابت uint32_t m   = x & 0x07070707u;
    ثابت uint32_t ge5 = ((m + 0x03030303u) >> 3) & 0x01010101u;
    ثابت uint32_t ge6 = ((m + 0x02020202u) >> 3) & 0x01010101u;
    ثابت uint32_t ge7 = ((m + 0x01010101u) >> 3) & 0x01010101u;
    ثابت uint32_t mag = m + ge5 + ge6 + 3u * ge7;
    ثابت uint32_t nz  = ((mag + 0x7f7f7f7fu) >> 7) & 0x01010101u;
    ثابت uint32_t neg = (x >> 3) & nz & 0x01010101u;
    واپس (صحیح) ((mag ^ (neg * 0xffu)) + neg);
}

// Same result as get_int_from_table_16(q4, kvalues_mxfp4): x = low nibbles, y = high nibbles.
ساکن __dpct_inline__ sycl::int2 get_int_from_mxfp4(ثابت صحیح q4) {
    واپس sycl::int2(mxfp4_codes_مقصد_int8((uint32_t) q4 & 0x0f0f0f0fu),
                      mxfp4_codes_مقصد_int8(((uint32_t) q4 >> 4) & 0x0f0f0f0fu));
}

#تعریف_کرو VDR_Q2_K_Q8_1_MMVQ 1

// contiguous v/x values
ساکن __dpct_inline__ اعشاری vec_dot_q2_K_q8_1_impl_mmvq(
    ثابت صحیح &v, ثابت صحیح *__restrict__ u, ثابت uint8_t *__restrict__ scales,
    ثابت sycl::half2 &dm2, ثابت اعشاری *__restrict__ d8) {

    اعشاری sumf_d = 0.0f;
    اعشاری sumf_m = 0.0f;

#ہدایت unroll
    برائے (صحیح i = 0; i < QR2_K; ++i) {
        ثابت صحیح sc = scales[2*i];

        ثابت صحیح vi = (v >> (2*i)) & 0x03030303;

        sumf_d +=
            d8[i] * (dpct::dp4a(vi, u[i], 0) * (sc & 0xF)); // SIMD dot product

        // fill int with 4x m
        صحیح m = sc >> 4;
        m |= m <<  8;
        m |= m << 16;
        sumf_m += d8[i] *
                  dpct::dp4a(
                      m, u[i],
                      0); // multiply constant q2_K part with sum of q8_1 values
    }

    ثابت sycl::float2 dm2f =
        dm2.convert<اعشاری, sycl::rounding_منوال::automatic>();

    واپس dm2f.x() * sumf_d - dm2f.y() * sumf_m;
}


#تعریف_کرو VDR_Q3_K_Q8_1_MMVQ 1

// contiguous v/x values
ساکن __dpct_inline__ اعشاری vec_dot_q3_K_q8_1_impl_mmvq(
    ثابت صحیح &vl, ثابت صحیح &vh, ثابت صحیح *__restrict__ u,
    ثابت uint8_t *__restrict__ scales, ثابت صحیح &scale_انحراف,
    ثابت اعشاری &d3, ثابت اعشاری *__restrict__ d8) {

    اعشاری sumf = 0.0f;

#ہدایت unroll
    برائے (صحیح i = 0; i < QR3_K; ++i) {
        ثابت صحیح isc = scale_انحراف + 2*i;

        ثابت صحیح isc_کم = isc % (QK_K/32);
        ثابت صحیح sc_سرکاؤ_کم = 4 * (isc / (QK_K/32));
        ثابت صحیح sc_کم  = (scales[isc_کم] >> sc_سرکاؤ_کم) & 0xF;

        ثابت صحیح isc_بلند = isc % (QK_K/64);
        ثابت صحیح sc_سرکاؤ_بلند = 2 * (isc / (QK_K/64));
        ثابت صحیح sc_بلند = ((scales[(QK_K/32) + isc_بلند] >> sc_سرکاؤ_بلند) & 3) << 4;

        ثابت صحیح sc = (sc_کم | sc_بلند) - 32;

        ثابت صحیح vil = (vl >> (2*i)) & 0x03030303;

        ثابت صحیح vih = ((vh >> i) << 2) & 0x04040404;

        ثابت صحیح vi =
            dpct::vectorized_ثنائی<sycl::char4>(vil, vih, dpct::sub_sat());

        sumf += d8[i] * (dpct::dp4a(vi, u[i], 0) * sc); // SIMD dot product
    }

    واپس d3 * sumf;
}

#تعریف_کرو VDR_Q4_K_Q8_1_MMVQ 2

// contiguous v/x values
ساکن __dpct_inline__ اعشاری vec_dot_q4_K_q8_1_impl_vmmq(
    ثابت صحیح *__restrict__ v, ثابت صحیح *__restrict__ u,
    ثابت uint8_t *__restrict__ sc, ثابت uint8_t *__restrict__ m,
    ثابت sycl::half2 &dm4, ثابت اعشاری *__restrict__ d8) {

    اعشاری sumf_d = 0.0f;
    اعشاری sumf_m = 0.0f;

#ہدایت unroll
    برائے (صحیح i = 0; i < QR4_K; ++i) {
        ثابت صحیح v0i = (v[0] >> (4*i)) & 0x0F0F0F0F;
        ثابت صحیح v1i = (v[1] >> (4*i)) & 0x0F0F0F0F;

        ثابت صحیح dot1 =
            dpct::dp4a(v1i, u[2 * i + 1],
                       dpct::dp4a(v0i, u[2 * i + 0], 0)); // SIMD dot product
        ثابت صحیح dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 1],
                       dpct::dp4a(0x01010101, u[2 * i + 0], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);  // multiply constant part of q4_K with sum of q8_1 values
    }

    ثابت sycl::float2 dm4f =
        dm4.convert<اعشاری, sycl::rounding_منوال::automatic>();

    واپس dm4f.x() * sumf_d - dm4f.y() * sumf_m;
}


#تعریف_کرو VDR_Q5_K_Q8_1_MMVQ 2

// contiguous v/x values
ساکن __dpct_inline__ اعشاری vec_dot_q5_K_q8_1_impl_vmmq(
    ثابت صحیح *__restrict__ vl, ثابت صحیح *__restrict__ vh,
    ثابت صحیح *__restrict__ u, ثابت uint8_t *__restrict__ sc,
    ثابت uint8_t *__restrict__ m, ثابت sycl::half2 &dm5,
    ثابت اعشاری *__restrict__ d8) {

    اعشاری sumf_d = 0.0f;
    اعشاری sumf_m = 0.0f;

#ہدایت unroll
    برائے (صحیح i = 0; i < QR5_K; ++i) {
        ثابت صحیح vl0i = (vl[0] >> (4*i)) & 0x0F0F0F0F;
        ثابت صحیح vl1i = (vl[1] >> (4*i)) & 0x0F0F0F0F;

        ثابت صحیح vh0i = ((vh[0] >> i) << 4) & 0x10101010;
        ثابت صحیح vh1i = ((vh[1] >> i) << 4) & 0x10101010;

        ثابت صحیح v0i = vl0i | vh0i;
        ثابت صحیح v1i = vl1i | vh1i;

        ثابت صحیح dot1 =
            dpct::dp4a(v0i, u[2 * i + 0],
                       dpct::dp4a(v1i, u[2 * i + 1], 0)); // SIMD dot product
        ثابت صحیح dot2 =
            dpct::dp4a(0x01010101, u[2 * i + 0],
                       dpct::dp4a(0x01010101, u[2 * i + 1], 0)); // sum of u

        sumf_d += d8[i] * (dot1 * sc[i]);
        sumf_m += d8[i] * (dot2 * m[i]);

    }

    ثابت sycl::float2 dm5f =
        dm5.convert<اعشاری, sycl::rounding_منوال::automatic>();

    واپس dm5f.x() * sumf_d - dm5f.y() * sumf_m;
}


#تعریف_کرو VDR_Q6_K_Q8_1_MMVQ 1

// contiguous v/x values
ساکن __dpct_inline__ اعشاری
vec_dot_q6_K_q8_1_impl_mmvq(ثابت صحیح &vl, ثابت صحیح &vh,
                            ثابت صحیح *__restrict__ u,
                            ثابت int8_t *__restrict__ scales, ثابت اعشاری &d,
                            ثابت اعشاری *__restrict__ d8) {
    واپس vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
}

#تعریف_کرو VDR_Q1_0_Q8_1_MMVQ 1
#تعریف_کرو VDR_Q1_0_Q8_1_MMQ  4

ساکن __dpct_inline__ اعشاری
vec_dot_q1_0_q8_1(ثابت خالی *__restrict__ vbq,
                  ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {

    ثابت بلاک_q1_0 * bq1_0 = (ثابت بلاک_q1_0 *) vbq;

    ثابت بلاک_q8_1 * bq8_1_chunk = bq8_1 + iqs;
    ثابت اعشاری        d1          = bq1_0->d;
    ثابت صحیح          v           = get_int_from_uint8_aligned(bq1_0->qs, iqs);

    صحیح vi_بائٹیں[8];
#ہدایت unroll
    برائے (صحیح j = 0; j < 8; ++j) {
        ثابت صحیح سرکاؤ = j * 4;
        ثابت صحیح bits4 = (v >> سرکاؤ) & 0x0F;
        ثابت صحیح b0    = (bits4 & 0x01) ? 1 : -1;
        ثابت صحیح b1    = (bits4 & 0x02) ? 1 : -1;
        ثابت صحیح b2    = (bits4 & 0x04) ? 1 : -1;
        ثابت صحیح b3    = (bits4 & 0x08) ? 1 : -1;
        vi_بائٹیں[j]     = (b0 & 0xFF) | ((b1 & 0xFF) << 8) | ((b2 & 0xFF) << 16) | ((b3 & 0xFF) << 24);
    }

    صحیح sumi = 0;
#ہدایت unroll
    برائے (صحیح j = 0; j < 8; ++j) {
        ثابت صحیح u = get_int_from_int8_aligned(bq8_1_chunk->qs, j);
        sumi        = ggml_sycl_dp4a(vi_بائٹیں[j], u, sumi);
    }

    واپس d1 * bq8_1_chunk->ds[0] * sumi;
}

// VDR = vec dot ratio, how many contiguous integers each thread processes when the vec dot kernel is called
// MMVQ = mul_mat_vec_q, MMQ = mul_mat_q

سانچہ <ggml_type T> ساخت reorder_vec_dot_q_sycl {
    static_assert(T != T, "ggml_type for reorder vecdot not implemented");
};

// For some types the weight side of the dot product does not depend on the destination column, so a
// multi-column mul_mat_vec can unpack it once per block instead of once per column. Such a type adds
// load() and dot() next to operator() and opts in here. See reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>.
سانچہ <ggml_type T> ساخت reorder_vec_dot_shared_weights {
    ساکن constexpr منطقی value = جھوٹ;
};

سانچہ <> ساخت reorder_vec_dot_shared_weights<GGML_TYPE_Q4_K> {
    ساکن constexpr منطقی value = سچ;
};

سانچہ <> ساخت reorder_vec_dot_shared_weights<GGML_TYPE_Q5_K> {
    ساکن constexpr منطقی value = سچ;
};

سانچہ <ggml_type T> ساخت reorder_vec_dot_shared_activations {
    ساکن constexpr منطقی value = جھوٹ;
};

سانچہ <> ساخت reorder_vec_dot_shared_activations<GGML_TYPE_Q4_K> {
    ساکن constexpr منطقی value = سچ;
};

سانچہ <> ساخت reorder_vec_dot_shared_activations<GGML_TYPE_Q5_K> {
    ساکن constexpr منطقی value = سچ;
};

سانچہ <> ساخت reorder_vec_dot_q_sycl<GGML_TYPE_Q4_0> {
    ساکن constexpr ggml_type gtype = GGML_TYPE_Q4_0;

    استعمال q4_0_بلاک  = ggml_sycl_reordered::بلاک_q_t<GGML_TYPE_Q4_0>;
    استعمال q4_0_traits = قسم_نام q4_0_بلاک::traits;

    __dpct_inline__ اعشاری vec_dot_q4_0_q8_1_impl(ثابت صحیح * v, ثابت صحیح * u, ثابت اعشاری & d4, ثابت sycl::half2 & ds8) {
        صحیح sumi = 0;

#ہدایت unroll
        برائے (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            ثابت صحیح vi0 = (v[i] >> 0) & 0x0F0F0F0F;
            ثابت صحیح vi1 = (v[i] >> 4) & 0x0F0F0F0F;

            // SIMD dot product of quantized values
            sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
            sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
        }

        ثابت sycl::float2 ds8f = ds8.convert<اعشاری, sycl::rounding_منوال::automatic>();

        // second part effectively subtracts 8 from each quant value
        واپس d4 * (sumi * ds8f.x() - (8 * q4_0_traits::vdr_mmvq / q4_0_traits::qi) * ds8f.y());
    }

    __dpct_inline__ اعشاری عامل()(ثابت خالی * __restrict__ vbq, ثابت std::pair<صحیح, صحیح> ibx_انحراف,
                                     ثابت std::pair<صحیح, صحیح> d_انحراف, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحیح & iqs) {
        ثابت uint8_t * bq4_0 = static_cast<ثابت uint8_t *>(vbq) + ibx_انحراف.پہلا;
        ثابت ggml_half d = *(reinterpret_cast<ثابت ggml_half *>(static_cast<ثابت uint8_t *>(vbq) + d_انحراف.پہلا));
        صحیح             v[q4_0_traits::vdr_mmvq];
        صحیح             u[2 * q4_0_traits::vdr_mmvq];


#ہدایت unroll
        برائے (size_t i = 0; i < q4_0_traits::vdr_mmvq; ++i) {
            v[i]         = get_int_from_uint8(bq4_0, iqs + i);
            u[2 * i + 0] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
            u[2 * i + 1] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i + q4_0_traits::qi);
        }

        واپس vec_dot_q4_0_q8_1_impl(v, u, d, *q8_1_ds);
    }
};

// Load four contiguous dwords per operand instead of loading each value separately.
ساخت reorder_vec_dot_q8_0_wide {
    ساکن constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    استعمال q8_0_بلاک  = ggml_sycl_reordered::بلاک_q_t<GGML_TYPE_Q8_0>;
    استعمال q8_0_traits = قسم_نام q8_0_بلاک::traits;

    __dpct_inline__ اعشاری عامل()(ثابت خالی * __restrict__ vbq, ثابت std::pair<صحیح, صحیح> ibx_انحراف,
                                     ثابت std::pair<صحیح, صحیح> d_انحراف, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحیح & iqs) {
        static_assert(q8_0_traits::vdr_mmvq == 4, "the wide load moves exactly four dwords");

        ثابت uint8_t * بنیاد = static_cast<ثابت uint8_t *>(vbq);
        ثابت int8_t *  qs   = reinterpret_cast<ثابت int8_t *>(بنیاد + ibx_انحراف.پہلا);
        ثابت ggml_half d    = *reinterpret_cast<ثابت ggml_half *>(بنیاد + d_انحراف.پہلا);

        ثابت sycl::int4 v = *reinterpret_cast<ثابت sycl::int4 *>(qs + حجم(صحیح) * iqs);
        ثابت sycl::int4 u = *reinterpret_cast<ثابت sycl::int4 *>(q8_1_quant_ptr + حجم(صحیح) * iqs);

        صحیح sumi = 0;
#ہدایت unroll
        برائے (صحیح i = 0; i < 4; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        ثابت sycl::half2 ds_اقدار = *q8_1_ds;
        واپس static_cast<اعشاری>(d) * static_cast<اعشاری>(ds_اقدار[0]) * sumi;
    }
};

سانچہ <> ساخت reorder_vec_dot_q_sycl<GGML_TYPE_Q8_0> {
    ساکن constexpr ggml_type gtype = GGML_TYPE_Q8_0;

    استعمال q8_0_بلاک  = ggml_sycl_reordered::بلاک_q_t<GGML_TYPE_Q8_0>;
    استعمال q8_0_traits = قسم_نام q8_0_بلاک::traits;

    __dpct_inline__ اعشاری عامل()(ثابت خالی * __restrict__ vbq, ثابت std::pair<صحیح, صحیح> ibx_انحراف,
                                     ثابت std::pair<صحیح, صحیح> d_انحراف, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحیح & iqs,
                                     ثابت صحیح vdr = q8_0_traits::vdr_mmvq) {
        ثابت uint8_t * بنیاد = static_cast<ثابت uint8_t *>(vbq);
        ثابت int8_t *  qs   = reinterpret_cast<ثابت int8_t *>(بنیاد + ibx_انحراف.پہلا);
        ثابت ggml_half  d   = *reinterpret_cast<ثابت ggml_half *>(بنیاد + d_انحراف.پہلا);

        صحیح v[q8_0_traits::vdr_mmvq];
        صحیح u[q8_0_traits::vdr_mmvq];

#ہدایت unroll
        برائے (صحیح i = 0; i < vdr; ++i) {
            v[i] = get_int_from_int8(qs, iqs + i);
            u[i] = get_int_from_int8_aligned(q8_1_quant_ptr, iqs + i);
        }

        صحیح sumi = 0;
#ہدایت unroll
        برائے (صحیح i = 0; i < vdr; ++i) {
            sumi = dpct::dp4a(v[i], u[i], sumi);
        }

        ثابت sycl::half2 ds_اقدار = *q8_1_ds;
        واپس static_cast<اعشاری>(d) * static_cast<اعشاری>(ds_اقدار[0]) * sumi;
    }
};

سانچہ <> ساخت reorder_vec_dot_q_sycl<GGML_TYPE_Q2_K> {
    ساکن constexpr ggml_type gtype = GGML_TYPE_Q2_K;

    استعمال q2_k_بلاک  = ggml_sycl_reordered::بلاک_q_t<GGML_TYPE_Q2_K>;
    استعمال q2_k_traits = قسم_نام q2_k_بلاک::traits;

    __dpct_inline__ اعشاری عامل()(ثابت خالی * __restrict__ vbq, ثابت std::pair<صحیح, صحیح> ibx_انحراف,
                                     ثابت std::pair<صحیح, صحیح> d_انحراف, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحیح & iqs) {
        ثابت uint8_t *    بنیاد   = static_cast<ثابت uint8_t *>(vbq);
        ثابت uint8_t *    qs     = بنیاد + ibx_انحراف.پہلا;
        ثابت uint8_t *    scales = بنیاد + d_انحراف.پہلا;
        ثابت ggml_half2 * dm     = reinterpret_cast<ثابت ggml_half2 *>(بنیاد + d_انحراف.second);

        ثابت صحیح bq8_انحراف   = QR2_K * (iqs / QI8_1);
        ثابت صحیح scale_انحراف = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        ثابت صحیح v = get_int_from_uint8_aligned(qs, iqs);

        صحیح   u[QR2_K];
        اعشاری d8[QR2_K];

#ہدایت unroll
        برائے (صحیح i = 0; i < QR2_K; ++i) {
            ثابت int8_t * quant_بنیاد_ptr = q8_1_quant_ptr + (bq8_انحراف + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_بنیاد_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_انحراف + i))[0];
        }

        واپس vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales + scale_انحراف, *dm, d8);
    }
};

سانچہ <> ساخت reorder_vec_dot_q_sycl<GGML_TYPE_Q3_K> {
    ساکن constexpr ggml_type gtype = GGML_TYPE_Q3_K;

    استعمال q3_k_بلاک  = ggml_sycl_reordered::بلاک_q_t<GGML_TYPE_Q3_K>;
    استعمال q3_k_traits = قسم_نام q3_k_بلاک::traits;

    __dpct_inline__ اعشاری عامل()(ثابت خالی * __restrict__ vbq, ثابت std::pair<صحیح, صحیح> ibx_انحراف,
                                     ثابت std::pair<صحیح, صحیح> d_انحراف, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحیح & iqs) {
        ثابت uint8_t *  بنیاد   = static_cast<ثابت uint8_t *>(vbq);
        ثابت uint8_t *  qs     = بنیاد + ibx_انحراف.پہلا;
        ثابت uint8_t *  hmask  = بنیاد + ibx_انحراف.second;
        ثابت uint8_t *  scales = بنیاد + d_انحراف.پہلا;
        ثابت ggml_half  d      = *reinterpret_cast<ثابت ggml_half *>(بنیاد + d_انحراف.second);

        ثابت صحیح bq8_انحراف   = QR3_K * (iqs / (QI3_K / 2));
        ثابت صحیح scale_انحراف = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1 / 2);

        ثابت صحیح vl = get_int_from_uint8(qs, iqs);
        ثابت صحیح vh = ~get_int_from_uint8(hmask, iqs % (QI3_K / 2)) >> bq8_انحراف;

        صحیح   u[QR3_K];
        اعشاری d8[QR3_K];

#ہدایت unroll
        برائے (صحیح i = 0; i < QR3_K; ++i) {
            ثابت int8_t * quant_بنیاد_ptr = q8_1_quant_ptr + (bq8_انحراف + i) * QK8_1;
            u[i]                          = get_int_from_int8_aligned(quant_بنیاد_ptr, iqs % QI8_1);
            d8[i]                         = (*(q8_1_ds + bq8_انحراف + i))[0];
        }

        واپس vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, scales, scale_انحراف, static_cast<اعشاری>(d), d8);
    }
};

ساکن اندرونی اعشاری vec_dot_q4_K_q8_1_common(ثابت صحیح * __restrict__ q4, ثابت uint16_t * __restrict__ scales,
                                             ثابت ggml_half2 & dm, ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                             ثابت صحیح &        iqs) {
    صحیح   v[2];
    صحیح   u[2 * QR4_K];
    اعشاری d8[QR4_K];

    v[0] = q4[0];
    v[1] = q4[4];

    uint16_t  aux[2];
    ثابت صحیح j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
    اگر (j < 2) {
        aux[0] = scales[j + 0] & 0x3f3f;
        aux[1] = scales[j + 2] & 0x3f3f;
    } ورنہ {
        aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
    }

    ثابت uint8_t * sc = (ثابت uint8_t *) aux;
    ثابت uint8_t * m  = sc + 2;

    ثابت صحیح bq8_انحراف = QR4_K * ((iqs / 2) / (QI8_1 / 2));

    برائے (صحیح i = 0; i < QR4_K; ++i) {
        ثابت بلاک_q8_1 * bq8i = bq8_1 + bq8_انحراف + i;
        d8[i]                   = bq8i->ds[0];

        ثابت صحیح * q8 = (ثابت صحیح *) bq8i->qs + ((iqs / 2) % 4);
        u[2 * i + 0]   = q8[0];
        u[2 * i + 1]   = q8[4];
    }

    واپس vec_dot_q4_K_q8_1_impl_vmmq(v, u, sc, m, dm, d8);
}

سانچہ <> ساخت reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K> {
    ساکن constexpr ggml_type gtype = GGML_TYPE_Q4_K;

    استعمال q4_k_بلاک  = ggml_sycl_reordered::بلاک_q_t<GGML_TYPE_Q4_K>;
    استعمال q4_k_traits = قسم_نام q4_k_بلاک::traits;

    ساخت weights {
        صحیح        v[2];
        uint16_t   aux[2];
        ggml_half2 dm;
        صحیح        bq8_انحراف;
    };

    ساخت activations {
        صحیح   u[2 * QR4_K];
        اعشاری d8[QR4_K];
    };

    __dpct_inline__ ساکن weights load(ثابت خالی * __restrict__ vbq, ثابت std::pair<صحیح, صحیح> ibx_انحراف,
                                        ثابت std::pair<صحیح, صحیح> d_انحراف, ثابت صحیح & iqs) {
        ثابت uint8_t *    بنیاد = static_cast<ثابت uint8_t *>(vbq);
        ثابت uint8_t *    qs   = بنیاد + ibx_انحراف.پہلا;
        ثابت uint8_t *    scs  = بنیاد + d_انحراف.پہلا;
        ثابت ggml_half2 * dms  = reinterpret_cast<ثابت ggml_half2 *>(بنیاد + d_انحراف.second);

        weights w;
        w.bq8_انحراف = QR4_K * ((iqs / 2) / (QI8_1 / 2));

        ثابت صحیح *      q4     = (ثابت صحیح *) (qs + 16 * w.bq8_انحراف + 4 * ((iqs / 2) % 4));
        ثابت uint16_t * scales = (ثابت uint16_t *) scs;

        w.v[0] = q4[0];
        w.v[1] = q4[4];

        ثابت صحیح j = (QR4_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        اگر (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } ورنہ {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        واپس w;
    }

    __dpct_inline__ ساکن activations بھرو_activations(ثابت int8_t * q8_1_quant_ptr,
                                                        ثابت sycl::half2 * q8_1_ds, ثابت صحیح & iqs) {
        activations a;
        ثابت صحیح bq8_انحراف = QR4_K * ((iqs / 2) / (QI8_1 / 2));
        برائے (صحیح i = 0; i < QR4_K; ++i) {
            ثابت int8_t * quant_بنیاد_ptr = q8_1_quant_ptr + (bq8_انحراف + i) * QK8_1;
            sycl::half2    ds_اقدار      = *(q8_1_ds + bq8_انحراف + i);

            a.d8[i] = ds_اقدار[0];

            ثابت صحیح * q8 = (ثابت صحیح *) quant_بنیاد_ptr + ((iqs / 2) % 4);
            a.u[2 * i + 0] = q8[0];
            a.u[2 * i + 1] = q8[4];
        }

        واپس a;
    }

    __dpct_inline__ ساکن اعشاری apply(ثابت weights & w, ثابت activations & a) {
        ثابت uint8_t * sc = (ثابت uint8_t *) w.aux;
        ثابت uint8_t * m  = sc + 2;

        واپس vec_dot_q4_K_q8_1_impl_vmmq(w.v, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ ساکن اعشاری dot(ثابت weights & w, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحیح & iqs) {
        ثابت خودکار a = بھرو_activations(q8_1_quant_ptr, q8_1_ds, iqs);

        واپس apply(w, a);
    }

    __dpct_inline__ اعشاری عامل()(ثابت خالی * __restrict__ vbq, ثابت std::pair<صحیح, صحیح> ibx_انحراف,
                                     ثابت std::pair<صحیح, صحیح> d_انحراف, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحیح & iqs) {
        واپس dot(load(vbq, ibx_انحراف, d_انحراف, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

سانچہ <> ساخت reorder_vec_dot_q_sycl<GGML_TYPE_Q5_K> {
    ساکن constexpr ggml_type gtype = GGML_TYPE_Q5_K;

    استعمال q5_k_بلاک  = ggml_sycl_reordered::بلاک_q_t<GGML_TYPE_Q5_K>;
    استعمال q5_k_traits = قسم_نام q5_k_بلاک::traits;

    ساخت weights {
        صحیح        vl[2];
        صحیح        vh[2];
        uint16_t   aux[2];
        ggml_half2 dm;
    };

    // same activation layout as Q4_K
    static_assert(QR5_K == QR4_K);
    استعمال activations = reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::activations;

    __dpct_inline__ ساکن weights load(ثابت خالی * __restrict__ vbq, ثابت std::pair<صحیح, صحیح> ibx_انحراف,
                                        ثابت std::pair<صحیح, صحیح> d_انحراف, ثابت صحیح & iqs) {
        ثابت uint8_t *    بنیاد    = static_cast<ثابت uint8_t *>(vbq);
        ثابت uint8_t *    qs      = بنیاد + ibx_انحراف.پہلا;   // low 4 bits
        ثابت uint8_t *    qh_بنیاد = بنیاد + ibx_انحراف.second;  // high bit
        ثابت uint8_t *    scs     = بنیاد + d_انحراف.پہلا;
        ثابت ggml_half2 * dms     = reinterpret_cast<ثابت ggml_half2 *>(بنیاد + d_انحراف.second);

        ثابت صحیح        bq8_انحراف = QR5_K * ((iqs / 2) / (QI8_1 / 2));
        ثابت صحیح *      ql_ptr     = (ثابت صحیح *) (qs + 16 * bq8_انحراف + 4 * ((iqs / 2) % 4));
        ثابت صحیح *      qh_ptr     = (ثابت صحیح *) (qh_بنیاد + 4 * ((iqs / 2) % 4));
        ثابت uint16_t * scales     = (ثابت uint16_t *) scs;

        weights w;
        w.vl[0] = ql_ptr[0];
        w.vl[1] = ql_ptr[4];

        w.vh[0] = qh_ptr[0] >> bq8_انحراف;
        w.vh[1] = qh_ptr[4] >> bq8_انحراف;

        ثابت صحیح j = (QR5_K * ((iqs / 2) / (QI8_1 / 2))) / 2;
        اگر (j < 2) {
            w.aux[0] = scales[j + 0] & 0x3f3f;
            w.aux[1] = scales[j + 2] & 0x3f3f;
        } ورنہ {
            w.aux[0] = ((scales[j + 2] >> 0) & 0x0f0f) | ((scales[j - 2] & 0xc0c0) >> 2);
            w.aux[1] = ((scales[j + 2] >> 4) & 0x0f0f) | ((scales[j - 0] & 0xc0c0) >> 2);
        }

        w.dm = *dms;

        واپس w;
    }

    __dpct_inline__ ساکن activations بھرو_activations(ثابت int8_t * q8_1_quant_ptr,
                                                        ثابت sycl::half2 * q8_1_ds, ثابت صحیح & iqs) {
        واپس reorder_vec_dot_q_sycl<GGML_TYPE_Q4_K>::بھرو_activations(q8_1_quant_ptr, q8_1_ds, iqs);
    }

    __dpct_inline__ ساکن اعشاری apply(ثابت weights & w, ثابت activations & a) {
        ثابت uint8_t * sc = (ثابت uint8_t *) w.aux;
        ثابت uint8_t * m  = sc + 2;

        واپس vec_dot_q5_K_q8_1_impl_vmmq(w.vl, w.vh, a.u, sc, m, w.dm, a.d8);
    }

    __dpct_inline__ ساکن اعشاری dot(ثابت weights & w, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحیح & iqs) {
        واپس apply(w, بھرو_activations(q8_1_quant_ptr, q8_1_ds, iqs));
    }

    __dpct_inline__ اعشاری عامل()(ثابت خالی * __restrict__ vbq, ثابت std::pair<صحیح, صحیح> ibx_انحراف,
                                     ثابت std::pair<صحیح, صحیح> d_انحراف, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحیح & iqs) {
        واپس dot(load(vbq, ibx_انحراف, d_انحراف, iqs), q8_1_quant_ptr, q8_1_ds, iqs);
    }
};

سانچہ <> ساخت reorder_vec_dot_q_sycl<GGML_TYPE_Q6_K> {
    ساکن constexpr ggml_type gtype = GGML_TYPE_Q6_K;

    استعمال q6_k_بلاک  = ggml_sycl_reordered::بلاک_q_t<GGML_TYPE_Q6_K>;
    استعمال q6_k_traits = قسم_نام q6_k_بلاک::traits;

    __dpct_inline__ اعشاری vec_dot_q6_K_q8_1_impl_mmvq(ثابت صحیح vl, ثابت صحیح vh, ثابت صحیح * __restrict__ u,
                                                      ثابت int8_t * __restrict__ scales, ثابت اعشاری d,
                                                      ثابت اعشاری * __restrict__ d8) {
        واپس vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u[0], u[1], scales[0], scales[4], d, d8[0], d8[1]);
    }

    __dpct_inline__ اعشاری عامل()(ثابت خالی * __restrict__ vbq, ثابت std::pair<صحیح, صحیح> ibx_انحراف,
                     ثابت std::pair<صحیح, صحیح> d_انحراف, ثابت int8_t * q8_1_quant_ptr, ثابت sycl::half2 * q8_1_ds,
                     ثابت صحیح iqs) {
        ثابت uint8_t *   بنیاد   = static_cast<ثابت uint8_t *>(vbq);
        ثابت uint8_t *   ql     = بنیاد + ibx_انحراف.پہلا;
        ثابت uint8_t *   qh     = بنیاد + ibx_انحراف.second;
        ثابت int8_t *    scales = reinterpret_cast<ثابت int8_t *>(بنیاد + d_انحراف.پہلا);
        ثابت ggml_half * d      = (ثابت ggml_half *) (بنیاد + d_انحراف.second);

        ثابت صحیح bq8_انحراف   = 2 * QR6_K * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 4);
        ثابت صحیح scale_انحراف = (QI6_K / 4) * (iqs / (QI6_K / 2)) + (iqs % (QI6_K / 2)) / (QI6_K / 8);
        ثابت صحیح vh_سرکاؤ     = 2 * ((iqs % (QI6_K / 2)) / (QI6_K / 4));

        ثابت صحیح vl = get_int_from_uint8(ql, iqs);
        ثابت صحیح vh = get_int_from_uint8(qh, (QI6_K / 4) * (iqs / (QI6_K / 2)) + iqs % (QI6_K / 4)) >> vh_سرکاؤ;

        ثابت int8_t * scs = scales + scale_انحراف;

        ثابت صحیح u0 = get_int_from_int8_aligned(
            q8_1_quant_ptr + bq8_انحراف * QK8_1, iqs % QI8_1);
        ثابت صحیح u1 = get_int_from_int8_aligned(
            q8_1_quant_ptr + (bq8_انحراف + 2) * QK8_1, iqs % QI8_1);
        ثابت اعشاری d80 = (*(q8_1_ds + bq8_انحراف + 0))[0];
        ثابت اعشاری d81 = (*(q8_1_ds + bq8_انحراف + 2))[0];

        واپس vec_dot_q6_K_q8_1_impl_mmvq_scalar(
            vl, vh, u0, u1, scs[0], scs[4], *d, d80, d81);
    }
};

سانچہ <> ساخت reorder_vec_dot_q_sycl<GGML_TYPE_MXFP4> {
    ساکن constexpr ggml_type gtype = GGML_TYPE_MXFP4;

    استعمال mxfp4_بلاک  = ggml_sycl_reordered::بلاک_q_t<GGML_TYPE_MXFP4>;
    استعمال mxfp4_traits = قسم_نام mxfp4_بلاک::traits;

    __dpct_inline__ اعشاری عامل()(ثابت خالی * __restrict__ vbq, ثابت std::pair<صحیح, صحیح> ibx_انحراف,
                                     ثابت std::pair<صحیح, صحیح> d_انحراف, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحیح & iqs) {
        static_assert(mxfp4_traits::vdr_mmvq == 2, "vector load assumes vdr_mmvq == 2");
        ثابت uint8_t * بنیاد = static_cast<ثابت uint8_t *>(vbq);

        // Reordered nibble blocks are 16 contiguous bytes and iqs is 0 or 2, so each lane's two
        // weight ints are one aligned 8-byte load (the AoS layout needed eight byte loads).
        ثابت sycl::int2 q4 = *reinterpret_cast<ثابت sycl::int2 *>(بنیاد + ibx_انحراف.پہلا + حجم(صحیح) * iqs);
        ثابت uint8_t    e  = بنیاد[d_انحراف.پہلا];

        // Low nibbles pair with q8_1 ints iqs..iqs+1, high nibbles with iqs+4..iqs+5.
        ثابت sycl::int2 u_lo = *reinterpret_cast<ثابت sycl::int2 *>(q8_1_quant_ptr + حجم(صحیح) * iqs);
        ثابت sycl::int2 u_hi = *reinterpret_cast<ثابت sycl::int2 *>(q8_1_quant_ptr + حجم(صحیح) * (iqs + 4));

        ثابت sycl::int2 v0 = get_int_from_mxfp4(q4.x());
        ثابت sycl::int2 v1 = get_int_from_mxfp4(q4.y());

        صحیح sumi = 0;
        sumi = ggml_sycl_dp4a(v0.x(), u_lo.x(), sumi);
        sumi = ggml_sycl_dp4a(v0.y(), u_hi.x(), sumi);
        sumi = ggml_sycl_dp4a(v1.x(), u_lo.y(), sumi);
        sumi = ggml_sycl_dp4a(v1.y(), u_hi.y(), sumi);

        ثابت اعشاری d = ggml_sycl_e8m0_مقصد_fp32(e) * 0.5f * static_cast<اعشاری>((*q8_1_ds)[0]);
        واپس d * sumi;
    }
};
#تعریف_کرو VDR_Q4_0_Q8_1_MMVQ 2
#تعریف_کرو VDR_Q4_0_Q8_1_MMQ  4

#تعریف_کرو VDR_Q2_0_Q8_1_MMVQ 1

سانچہ <صحیح vdr>
ساکن __dpct_inline__ اعشاری vec_dot_q2_0_q8_1_impl(
    ثابت صحیح * v,
    ثابت صحیح * u,
    ثابت اعشاری & d2,
    ثابت sycl::half2 & ds8) {
    صحیح sumi = 0;

#ہدایت unroll
    برائے (صحیح i = 0; i < vdr; ++i) {
#ہدایت unroll
        برائے (صحیح j = 0; j < 4; ++j) {
            ثابت uint8_t q = (uint8_t) ((uint32_t) v[i] >> (8 * j));

            // unpack 2-bit values to byte lanes (0..3), then apply zero-point
            // correction with ds8f.y() below, mirroring the q4_0 style.
            صحیح vi = 0;
            vi |= (((q >> 0) & 0x3) & 0xFF) << 0;
            vi |= (((q >> 2) & 0x3) & 0xFF) << 8;
            vi |= (((q >> 4) & 0x3) & 0xFF) << 16;
            vi |= (((q >> 6) & 0x3) & 0xFF) << 24;

            sumi = dpct::dp4a(vi, u[4 * i + j], sumi);
        }
    }

    ثابت sycl::float2 ds8f = ds8.convert<اعشاری, sycl::rounding_منوال::automatic>();
    // q2_0 has zero-point 1. Scale ds8f.y() by processed-lane ratio,
    // consistent with q4_0's explicit zero-point subtraction style.
    واپس d2 * (sumi * ds8f.x() - ((اعشاری) vdr / (اعشاری) QI2_0) * ds8f.y());
}

سانچہ <صحیح vdr>
ساکن __dpct_inline__ اعشاری vec_dot_q4_0_q8_1_impl(ثابت صحیح * v, ثابت صحیح * u, ثابت اعشاری & d4,
                                                    ثابت sycl::half2 & ds8) {
    صحیح sumi = 0;
#ہدایت unroll
    برائے (صحیح i = 0; i < vdr; ++i) {
        ثابت صحیح vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        ثابت صحیح vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

    ثابت sycl::float2 ds8f = ds8.convert<اعشاری, sycl::rounding_منوال::automatic>();

    // second part effectively subtracts 8 from each quant value
    واپس d4 * (sumi * ds8f.x() - (8 * vdr / QI4_0) * ds8f.y());
}

#تعریف_کرو VDR_Q4_1_Q8_1_MMVQ 2
#تعریف_کرو VDR_Q4_1_Q8_1_MMQ  4

سانچہ <صحیح vdr>
ساکن __dpct_inline__ اعشاری vec_dot_q4_1_q8_1_impl(ثابت صحیح *v, ثابت صحیح *u,
                                                    ثابت sycl::half2 &dm4,
                                                    ثابت sycl::half2 &ds8) {

    صحیح sumi = 0;

#ہدایت unroll
    برائے (صحیح i = 0; i < vdr; ++i) {
        ثابت صحیح vi0 = (v[i] >> 0) & 0x0F0F0F0F;
        ثابت صحیح vi1 = (v[i] >> 4) & 0x0F0F0F0F;

        // SIMD dot product of quantized values
        sumi = dpct::dp4a(vi0, u[2 * i + 0], sumi);
        sumi = dpct::dp4a(vi1, u[2 * i + 1], sumi);
    }

#اگر_تعریف_شدہ GGML_SYCL_F16
    ثابت sycl::float2 tmp =
        (dm4 * ds8).convert<اعشاری, sycl::rounding_منوال::automatic>();
    ثابت اعشاری d4d8 = tmp.x();
    ثابت اعشاری m4s8 = tmp.y();
#ورنہ
    ثابت sycl::float2 dm4f =
        dm4.convert<اعشاری, sycl::rounding_منوال::automatic>();
    ثابت sycl::float2 ds8f =
        ds8.convert<اعشاری, sycl::rounding_منوال::automatic>();
    ثابت اعشاری d4d8 = dm4f.x() * ds8f.x();
    ثابت اعشاری m4s8 = dm4f.y() * ds8f.y();
#اگر_ختم // GGML_SYCL_F16

    // scale second part of sum by QI8_1/(vdr * QR4_1) to compensate for multiple threads adding it
    واپس sumi * d4d8 + m4s8 / (QI8_1 / (vdr * QR4_1));
}

#تعریف_کرو VDR_Q5_0_Q8_1_MMVQ 2
#تعریف_کرو VDR_Q5_0_Q8_1_MMQ  4

سانچہ <صحیح vdr>
ساکن __dpct_inline__ اعشاری
vec_dot_q5_0_q8_1_impl(ثابت صحیح *vl, ثابت صحیح *vh, ثابت صحیح *u,
                       ثابت اعشاری &d5, ثابت sycl::half2 &ds8) {
    صحیح sumi = 0;

#ہدایت unroll
    برائے (صحیح i = 0; i < vdr; ++i) {
        صحیح vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        صحیح vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

    ثابت sycl::float2 ds8f =
        ds8.convert<اعشاری, sycl::rounding_منوال::automatic>();

    // second part effectively subtracts 16 from each quant value
    واپس d5 * (sumi * ds8f.x() - (16 * vdr / QI5_0) * ds8f.y());
}

#تعریف_کرو VDR_Q5_1_Q8_1_MMVQ 2
#تعریف_کرو VDR_Q5_1_Q8_1_MMQ  4

سانچہ <صحیح vdr>
ساکن __dpct_inline__ اعشاری
vec_dot_q5_1_q8_1_impl(ثابت صحیح *vl, ثابت صحیح *vh, ثابت صحیح *u,
                       ثابت sycl::half2 &dm5, ثابت sycl::half2 &ds8) {

    صحیح sumi = 0;

#ہدایت unroll
    برائے (صحیح i = 0; i < vdr; ++i) {
        صحیح vi0 = (vl[i] >>  0) & 0x0F0F0F0F; // lower 4 qs bits, still need qh as 5th bits
        vi0    |= (vh[i] <<  4) & 0x00000010; // 0 ->  4
        vi0    |= (vh[i] << 11) & 0x00001000; // 1 -> 12
        vi0    |= (vh[i] << 18) & 0x00100000; // 2 -> 20
        vi0    |= (vh[i] << 25) & 0x10000000; // 3 -> 28
        sumi = dpct::dp4a(vi0, u[2 * i + 0],
                          sumi); // SIMD dot product of quantized values

        صحیح vi1 = (vl[i] >>  4) & 0x0F0F0F0F; // upper 4 qs bits, still need qh as 5th bits
        vi1    |= (vh[i] >> 12) & 0x00000010; // 16 ->  4
        vi1    |= (vh[i] >>  5) & 0x00001000; // 17 -> 12
        vi1    |= (vh[i] <<  2) & 0x00100000; // 18 -> 20
        vi1    |= (vh[i] <<  9) & 0x10000000; // 19 -> 28
        sumi = dpct::dp4a(vi1, u[2 * i + 1],
                          sumi); // SIMD dot product of quantized values
    }

#اگر_تعریف_شدہ GGML_SYCL_F16
     ثابت sycl::float2 tmp =
        (dm5 * ds8).convert<اعشاری, sycl::rounding_منوال::automatic>();
    ثابت اعشاری d5d8 = tmp.x();
    ثابت اعشاری m5s8 = tmp.y();


#ورنہ
    ثابت sycl::float2 dm5f =
        dm5.convert<اعشاری, sycl::rounding_منوال::automatic>();
    ثابت sycl::float2 ds8f =
        ds8.convert<اعشاری, sycl::rounding_منوال::automatic>();
    ثابت اعشاری d5d8 = dm5f.x() * ds8f.x();
    ثابت اعشاری m5s8 = dm5f.y() * ds8f.y();
#اگر_ختم // GGML_SYCL_F16

    // scale second part of sum by QI5_1 / vdr to compensate for multiple threads adding it
    واپس sumi*d5d8 + m5s8 / (QI5_1 / vdr);
}

#تعریف_کرو VDR_Q8_0_Q8_1_MMVQ 2
#تعریف_کرو VDR_Q8_0_Q8_1_MMQ 8

سانچہ <صحیح vdr>
ساکن __dpct_inline__ اعشاری vec_dot_q8_0_q8_1_impl(ثابت صحیح *v, ثابت صحیح *u,
                                                    ثابت اعشاری &d8_0,
                                                    ثابت اعشاری &d8_1) {

    صحیح sumi = 0;

#ہدایت unroll
    برائے (صحیح i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

    واپس d8_0*d8_1 * sumi;
}

سانچہ <قسم_نام T, صحیح vdr>
ساکن __dpct_inline__ T vec_dot_q8_0_q8_1_impl(ثابت صحیح * v, ثابت صحیح * u, ثابت T & d8_0, ثابت T & d8_1) {
    صحیح sumi = 0;

#ہدایت unroll
    برائے (صحیح i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = ggml_sycl_dp4a(v[i], u[i], sumi);
    }

    واپس d8_0*d8_1 * ((T) sumi);
}

سانچہ <صحیح vdr>
ساکن __dpct_inline__ اعشاری vec_dot_q8_1_q8_1_impl(ثابت صحیح *v, ثابت صحیح *u,
                                                    ثابت sycl::half2 &dm8,
                                                    ثابت sycl::half2 &ds8) {

    صحیح sumi = 0;

#ہدایت unroll
    برائے (صحیح i = 0; i < vdr; ++i) {
        // SIMD dot product of quantized values
        sumi = dpct::dp4a(v[i], u[i], sumi);
    }

#اگر_تعریف_شدہ GGML_SYCL_F16
    ثابت sycl::float2 tmp =
        (dm8 * ds8).convert<اعشاری, sycl::rounding_منوال::automatic>();
    ثابت اعشاری d8d8 = tmp.x();
    ثابت اعشاری m8s8 = tmp.y();
#ورنہ
    ثابت sycl::float2 dm8f =
        dm8.convert<اعشاری, sycl::rounding_منوال::automatic>();
    ثابت sycl::float2 ds8f =
        ds8.convert<اعشاری, sycl::rounding_منوال::automatic>();
    ثابت اعشاری d8d8 = dm8f.x() * ds8f.x();
    ثابت اعشاری m8s8 = dm8f.y() * ds8f.y();
#اگر_ختم // GGML_SYCL_F16

    // scale second part of sum by QI8_1/ vdr to compensate for multiple threads adding it
    واپس sumi*d8d8 + m8s8 / (QI8_1 / vdr);
}

ساکن __dpct_inline__ اعشاری
vec_dot_q4_0_q8_1(ثابت خالی *__restrict__ vbq,
                  ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {

    ثابت بلاک_q4_0 * bq4_0 = (ثابت بلاک_q4_0 *) vbq;

    صحیح v[VDR_Q4_0_Q8_1_MMVQ];
    صحیح u[2 * VDR_Q4_0_Q8_1_MMVQ];

#ہدایت unroll
    برائے (صحیح i = 0; i < VDR_Q4_0_Q8_1_MMVQ; ++i) {
        v[i]         = get_int_from_uint8(bq4_0->qs, iqs + i);
        u[2 * i + 0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2 * i + 1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_0);
    }

    واپس vec_dot_q4_0_q8_1_impl<VDR_Q4_0_Q8_1_MMVQ>(v, u, bq4_0->d, bq8_1->ds);
}

ساکن __dpct_inline__ اعشاری
vec_dot_q2_0_q8_1(ثابت خالی *__restrict__ vbq,
                  ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {

    ثابت بلاک_q2_0 * bq2_0 = (ثابت بلاک_q2_0 *) vbq;

    صحیح v[2 * VDR_Q2_0_Q8_1_MMVQ];
    صحیح u[8 * VDR_Q2_0_Q8_1_MMVQ];

#ہدایت unroll
    برائے (صحیح i = 0; i < VDR_Q2_0_Q8_1_MMVQ; ++i) {
        ثابت صحیح بنیاد = 4 * (iqs + i);

        // Q2_0 has QK2_0 = 64 and uses 2 x QK8_1 blocks on the RHS.
        v[2 * i + 0] = get_int_from_uint8(bq2_0->qs, iqs + i);
        v[2 * i + 1] = get_int_from_uint8(bq2_0->qs, iqs + i + QI2_0);

        u[8 * i + 0] = get_int_from_int8_aligned(bq8_1[0].qs, بنیاد + 0);
        u[8 * i + 1] = get_int_from_int8_aligned(bq8_1[0].qs, بنیاد + 1);
        u[8 * i + 2] = get_int_from_int8_aligned(bq8_1[0].qs, بنیاد + 2);
        u[8 * i + 3] = get_int_from_int8_aligned(bq8_1[0].qs, بنیاد + 3);

        u[8 * i + 4] = get_int_from_int8_aligned(bq8_1[1].qs, بنیاد + 0);
        u[8 * i + 5] = get_int_from_int8_aligned(bq8_1[1].qs, بنیاد + 1);
        u[8 * i + 6] = get_int_from_int8_aligned(bq8_1[1].qs, بنیاد + 2);
        u[8 * i + 7] = get_int_from_int8_aligned(bq8_1[1].qs, بنیاد + 3);
    }

    ثابت اعشاری sum0 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + 0, u + 0, bq2_0->d, bq8_1[0].ds);
    ثابت اعشاری sum1 = vec_dot_q2_0_q8_1_impl<VDR_Q2_0_Q8_1_MMVQ>(
        v + VDR_Q2_0_Q8_1_MMVQ, u + 4 * VDR_Q2_0_Q8_1_MMVQ, bq2_0->d, bq8_1[1].ds);
    واپس sum0 + sum1;
}

ساکن __dpct_inline__ اعشاری
vec_dot_q4_1_q8_1(ثابت خالی *__restrict__ vbq,
                  ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {

    ثابت بلاک_q4_1 * bq4_1 = (ثابت بلاک_q4_1 *) vbq;

    صحیح v[VDR_Q4_1_Q8_1_MMVQ];
    صحیح u[2*VDR_Q4_1_Q8_1_MMVQ];

#ہدایت unroll
    برائے (صحیح i = 0; i < VDR_Q4_1_Q8_1_MMVQ; ++i) {
        v[i]    = get_int_from_uint8_aligned(bq4_1->qs, iqs + i);
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI4_1);
    }

    واپس vec_dot_q4_1_q8_1_impl<VDR_Q4_1_Q8_1_MMVQ>(v, u, bq4_1->dm, bq8_1->ds);
}

#تعریف_کرو VDR_MXFP4_Q8_1_MMVQ 2
#تعریف_کرو VDR_MXFP4_Q8_1_MMQ  4

ساکن __dpct_inline__ اعشاری vec_dot_mxfp4_q8_1(ثابت خالی * __restrict__ vbq,
                                                ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                                ثابت صحیح & iqs) {
    ثابت بلاک_mxfp4 * bq4 = (ثابت بلاک_mxfp4 *) vbq;

    ثابت صحیح * q8 = (ثابت صحیح *) bq8_1->qs + iqs;

    صحیح sumi = 0;
#ہدایت unroll
    برائے (صحیح l = 0; l < VDR_MXFP4_Q8_1_MMVQ; ++l) {
        ثابت صحیح aux_q4 = get_int_b1(bq4->qs, iqs + l);
        ثابت sycl::int2 v      = get_int_from_mxfp4(aux_q4);
        sumi = ggml_sycl_dp4a(v.x(), q8[l + 0], sumi);
        sumi = ggml_sycl_dp4a(v.y(), q8[l + 4], sumi);
    }

    ثابت اعشاری d = ggml_sycl_e8m0_مقصد_fp32(bq4->e) * 0.5f * (bq8_1->ds)[0];
    واپس d * sumi;
}

#تعریف_کرو VDR_NVFP4_Q8_1_MMVQ 4
#تعریف_کرو VDR_NVFP4_Q8_1_MMQ  8

ساکن __dpct_inline__ اعشاری vec_dot_nvfp4_q8_1(ثابت خالی * __restrict__ vbq,
                                                ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                                ثابت int32_t & iqs) {
    ثابت بلاک_nvfp4 * bq4 = (ثابت بلاک_nvfp4 *) vbq;
    اعشاری مجموع = 0.0f;
#ہدایت unroll
    برائے (صحیح i = 0; i < VDR_NVFP4_Q8_1_MMVQ/2; i++) {
        ثابت int32_t iqs0 = iqs + 2*i;
        ثابت int32_t iqs1 = iqs0 + 1;
        ثابت int32_t is = iqs0 >> 1;
        ثابت sycl::int2   v0   = get_int_from_جدول_16(get_int_b4(bq4->qs, iqs0), kvalues_mxfp4);
        ثابت sycl::int2   v1   = get_int_from_جدول_16(get_int_b4(bq4->qs, iqs1), kvalues_mxfp4);
        ثابت بلاک_q8_1 * bq8 = bq8_1 + (is >> 1);
        ثابت int32_t i8 = ((is & 1) << 2);

        صحیح sumi = ggml_sycl_dp4a(v0.x(), get_int_b4(bq8->qs, i8 + 0), 0);
        sumi     = ggml_sycl_dp4a(v0.y(), get_int_b4(bq8->qs, i8 + 2), sumi);
        sumi     = ggml_sycl_dp4a(v1.x(), get_int_b4(bq8->qs, i8 + 1), sumi);
        sumi     = ggml_sycl_dp4a(v1.y(), get_int_b4(bq8->qs, i8 + 3), sumi);

        ثابت اعشاری d = ggml_sycl_ue4m3_مقصد_fp32(bq4->d[is]) * (bq8->ds)[0];
        مجموع += d * اعشاری(sumi);
    }

    واپس مجموع;
}

ساکن __dpct_inline__ اعشاری
vec_dot_q5_0_q8_1(ثابت خالی *__restrict__ vbq,
                  ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {

    ثابت بلاک_q5_0 * bq5_0 = (ثابت بلاک_q5_0 *) vbq;

    صحیح vl[VDR_Q5_0_Q8_1_MMVQ];
    صحیح vh[VDR_Q5_0_Q8_1_MMVQ];
    صحیح  u[2*VDR_Q5_0_Q8_1_MMVQ];

#ہدایت unroll
    برائے (صحیح i = 0; i < VDR_Q5_0_Q8_1_MMVQ; ++i) {
        vl[i]    = get_int_from_uint8(bq5_0->qs, iqs + i);
        vh[i]    = get_int_from_uint8(bq5_0->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_0);
    }

    واپس vec_dot_q5_0_q8_1_impl<VDR_Q5_0_Q8_1_MMVQ>(vl, vh, u, bq5_0->d, bq8_1->ds);
}

ساکن __dpct_inline__ اعشاری
vec_dot_q5_1_q8_1(ثابت خالی *__restrict__ vbq,
                  ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {

    ثابت بلاک_q5_1 * bq5_1 = (ثابت بلاک_q5_1 *) vbq;

    صحیح vl[VDR_Q5_1_Q8_1_MMVQ];
    صحیح vh[VDR_Q5_1_Q8_1_MMVQ];
    صحیح  u[2*VDR_Q5_1_Q8_1_MMVQ];

#ہدایت unroll
    برائے (صحیح i = 0; i < VDR_Q5_1_Q8_1_MMVQ; ++i) {
        vl[i]   = get_int_from_uint8_aligned(bq5_1->qs, iqs + i);
        vh[i]   = get_int_from_uint8_aligned(bq5_1->qh, 0) >> (4 * (iqs + i));
        u[2*i+0] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
        u[2*i+1] = get_int_from_int8_aligned(bq8_1->qs, iqs + i + QI5_1);
    }

    واپس vec_dot_q5_1_q8_1_impl<VDR_Q5_1_Q8_1_MMVQ>(vl, vh, u, bq5_1->dm, bq8_1->ds);
}

ساکن __dpct_inline__ اعشاری
vec_dot_q8_0_q8_1(ثابت خالی *__restrict__ vbq,
                  ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {

    ثابت بلاک_q8_0 * bq8_0 = (ثابت بلاک_q8_0 *) vbq;

    صحیح v[VDR_Q8_0_Q8_1_MMVQ];
    صحیح u[VDR_Q8_0_Q8_1_MMVQ];

#ہدایت unroll
    برائے (صحیح i = 0; i < VDR_Q8_0_Q8_1_MMVQ; ++i) {
        v[i] = get_int_from_int8(bq8_0->qs, iqs + i);
        u[i] = get_int_from_int8_aligned(bq8_1->qs, iqs + i);
    }

    واپس vec_dot_q8_0_q8_1_impl<VDR_Q8_0_Q8_1_MMVQ>(v, u, bq8_0->d,
                                                      bq8_1->ds[0]);
}

ساکن __dpct_inline__ اعشاری
vec_dot_q2_K_q8_1(ثابت خالی *__restrict__ vbq,
                  ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {

    ثابت بلاک_q2_K * bq2_K = (ثابت بلاک_q2_K *) vbq;

    ثابت صحیح bq8_انحراف = QR2_K * (iqs / QI8_1);
    ثابت صحیح scale_انحراف = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    ثابت uint8_t * scales = bq2_K->scales + scale_انحراف;

    ثابت صحیح v = get_int_from_uint8_aligned(bq2_K->qs, iqs);
    صحیح    u[QR2_K];
    اعشاری d8[QR2_K];

#ہدایت unroll
    برائے (صحیح i = 0; i < QR2_K; ++ i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_انحراف + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_انحراف + i].ds[0];
    }

    واپس vec_dot_q2_K_q8_1_impl_mmvq(v, u, scales, bq2_K->dm, d8);
}

ساکن __dpct_inline__ اعشاری
vec_dot_q3_K_q8_1(ثابت خالی *__restrict__ vbq,
                  ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {

    ثابت بلاک_q3_K * bq3_K = (ثابت بلاک_q3_K *) vbq;

    ثابت صحیح bq8_انحراف = QR3_K * (iqs / (QI3_K/2));
    ثابت صحیح scale_انحراف = iqs - iqs % QI8_1 + (iqs % QI8_1) / (QI8_1/2);

    ثابت اعشاری d = bq3_K->d;

    ثابت صحیح vl = get_int_from_uint8(bq3_K->qs, iqs);

    // invert the mask with ~ so that a 0/1 results in 4/0 being subtracted
    ثابت صحیح vh = ~get_int_from_uint8(bq3_K->hmask, iqs % (QI3_K/2)) >> bq8_انحراف;

    صحیح    u[QR3_K];
    اعشاری d8[QR3_K];

#ہدایت unroll
    برائے (صحیح i = 0; i < QR3_K; ++i) {
        u[i]  = get_int_from_int8_aligned(bq8_1[bq8_انحراف + i].qs, iqs % QI8_1);
        d8[i] = bq8_1[bq8_انحراف + i].ds[0];
    }

    واپس vec_dot_q3_K_q8_1_impl_mmvq(vl, vh, u, bq3_K->scales, scale_انحراف, d, d8);
}

ساکن __dpct_inline__ اعشاری vec_dot_q4_K_q8_1(ثابت خالی * __restrict__ vbq, ثابت بلاک_q8_1 * __restrict__ bq8_1,
                                               ثابت صحیح & iqs) {
#اگر_غیر_تعریف_شدہ GGML_QKK_64

    ثابت بلاک_q4_K * bq4_K = (ثابت بلاک_q4_K *) vbq;

    ثابت صحیح        bq8_انحراف = QR4_K * ((iqs / 2) / (QI8_1 / 2));
    ثابت صحیح *      q4         = (ثابت صحیح *) (bq4_K->qs + 16 * bq8_انحراف + 4 * ((iqs / 2) % 4));
    ثابت uint16_t * scales     = (ثابت uint16_t *) bq4_K->scales;

    واپس vec_dot_q4_K_q8_1_common(q4, scales, bq4_K->dm, bq8_1, iqs);

#ورنہ

#اگر __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    ثابت بلاک_q4_K * bq4_K = (ثابت بلاک_q4_K *) vbq;

    اعشاری sumf_d = 0.0f;
    اعشاری sumf_m = 0.0f;

    uint16_t aux16[2];
    ثابت uint8_t * s = (ثابت uint8_t *)aux16;

    ثابت uint16_t * a = (ثابت uint16_t *)bq4_K->scales;
    aux16[0] = a[0] & 0x0f0f;
    aux16[1] = (a[0] >> 4) & 0x0f0f;

    ثابت اعشاری dall = bq4_K->dm[0];
    ثابت اعشاری dmin = bq4_K->dm[1];

    ثابت اعشاری d8_1 = bq8_1[0].ds[0];
    ثابت اعشاری d8_2 = bq8_1[1].ds[1];

    ثابت صحیح ui1 = *((ثابت صحیح *)bq8_1[0].qs + (iqs/2));
    ثابت صحیح ui2 = *((ثابت صحیح *)bq8_1[0].qs + (iqs/2) + 4);
    ثابت صحیح ui3 = *((ثابت صحیح *)bq8_1[1].qs + (iqs/2));
    ثابت صحیح ui4 = *((ثابت صحیح *)bq8_1[1].qs + (iqs/2) + 4);

    ثابت صحیح * q4 = (ثابت صحیح *)bq4_K->qs + (iqs/2);
    ثابت صحیح v1 = q4[0];
    ثابت صحیح v2 = q4[4];

    ثابت صحیح dot1 = dpct::dp4a(ui2, v2 & 0x0f0f0f0f, dpct::dp4a(ui1, v1 & 0x0f0f0f0f, 0));
    ثابت صحیح dot2 = dpct::dp4a(ui4, (v2 >> 4) & 0x0f0f0f0f, dpct::dp4a(ui3, (v1 >> 4) & 0x0f0f0f0f, 0));
    ثابت صحیح dot3 = dpct::dp4a(0x01010101, ui2, dpct::dp4a(0x01010101, ui1, 0));
    ثابت صحیح dot4 = dpct::dp4a(0x01010101, ui4, dpct::dp4a(0x01010101, ui3, 0));

    sumf_d += d8_1 * (dot1 * s[0]) + d8_2 * (dot2 * s[1]);
    sumf_m += d8_1 * (dot3 * s[2]) + d8_2 * (dot4 * s[3]);

    واپس dall * sumf_d - dmin * sumf_m;

#ورنہ
    bad_arch();
#اگر_ختم // __SYCL_ARCH__ >= VER_4VEC

#اگر_ختم
}

ساکن __dpct_inline__ اعشاری
vec_dot_q5_K_q8_1(ثابت خالی *__restrict__ vbq,
                  ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {

#اگر_غیر_تعریف_شدہ GGML_QKK_64
    ثابت بلاک_q5_K * bq5_K = (ثابت بلاک_q5_K *) vbq;

    صحیح   vl[2];
    صحیح   vh[2];
    صحیح    u[2*QR5_K];
    اعشاری d8[QR5_K];

    ثابت صحیح bq8_انحراف = QR5_K * ((iqs/2) / (QI8_1/2));
    ثابت صحیح * ql = (ثابت صحیح *)(bq5_K->qs + 16 * bq8_انحراف + 4 * ((iqs/2)%4));
    ثابت صحیح * qh = (ثابت صحیح *)(bq5_K->qh + 4 * ((iqs/2)%4));

    vl[0] = ql[0];
    vl[1] = ql[4];

    vh[0] = qh[0] >> bq8_انحراف;
    vh[1] = qh[4] >> bq8_انحراف;

    ثابت uint16_t * scales = (ثابت uint16_t *)bq5_K->scales;
    uint16_t aux[2];
    ثابت صحیح j = bq8_انحراف/2;
    اگر (j < 2) {
        aux[0] = scales[j+0] & 0x3f3f;
        aux[1] = scales[j+2] & 0x3f3f;
    } ورنہ {
        aux[0] = ((scales[j+2] >> 0) & 0x0f0f) | ((scales[j-2] & 0xc0c0) >> 2);
        aux[1] = ((scales[j+2] >> 4) & 0x0f0f) | ((scales[j-0] & 0xc0c0) >> 2);
    }
    ثابت uint8_t * sc = (ثابت uint8_t *)aux;
    ثابت uint8_t * m  = sc + 2;

#ہدایت unroll
    برائے (صحیح i = 0; i < QR5_K; ++i) {
        ثابت بلاک_q8_1 * bq8i = bq8_1 + bq8_انحراف + i;
        d8[i] = bq8i->ds[0];

        ثابت صحیح * q8 = (ثابت صحیح *)bq8i->qs + ((iqs/2)%4);
        u[2*i+0] = q8[0];
        u[2*i+1] = q8[4];
    }

    واپس vec_dot_q5_K_q8_1_impl_vmmq(vl, vh, u, sc, m, bq5_K->dm, d8);

#ورنہ

#اگر __SYCL_ARCH__ >= VER_4VEC // lowest compute capability for integer intrinsics
    ثابت بلاک_q5_K * bq5_K = (ثابت بلاک_q5_K *) vbq;

    ثابت int8_t * s = bq5_K->scales;

    ثابت اعشاری d = bq5_K->d;

    ثابت اعشاری d8_1 = bq8_1[0].ds[0];
    ثابت اعشاری d8_2 = bq8_1[1].ds[1];

    ثابت صحیح ui1 = *((ثابت صحیح *)bq8_1[0].qs + (iqs/2));
    ثابت صحیح ui2 = *((ثابت صحیح *)bq8_1[0].qs + (iqs/2) + 4);
    ثابت صحیح ui3 = *((ثابت صحیح *)bq8_1[1].qs + (iqs/2));
    ثابت صحیح ui4 = *((ثابت صحیح *)bq8_1[1].qs + (iqs/2) + 4);

    ثابت صحیح * ql = (ثابت صحیح *)bq5_K->qs + (iqs/2);
    ثابت صحیح vl1 = ql[0];
    ثابت صحیح vl2 = ql[4];

    ثابت صحیح قدم = 4 * (iqs/2); // 0, 4, 8, 12
    ثابت صحیح im = قدم/8; // = 0 for iqs = 0, 2, = 1 for iqs = 4, 6
    ثابت صحیح اندر = قدم%8; // 0, 4, 0, 4
    ثابت صحیح vh = (*((ثابت صحیح *)(bq5_K->qh + اندر))) >> im;

    ثابت صحیح v1 = (((vh << 4) & 0x10101010) ^ 0x10101010) | ((vl1 >> 0) & 0x0f0f0f0f);
    ثابت صحیح v2 = (((vh << 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 0) & 0x0f0f0f0f);
    ثابت صحیح v3 = (((vh >> 0) & 0x10101010) ^ 0x10101010) | ((vl1 >> 4) & 0x0f0f0f0f);
    ثابت صحیح v4 = (((vh >> 2) & 0x10101010) ^ 0x10101010) | ((vl2 >> 4) & 0x0f0f0f0f);

    ثابت اعشاری sumf_d = d8_1 * (dpct::dp4a(ui1, v1, 0) * s[0] + dpct::dp4a(ui2, v2, 0) * s[1])
                       + d8_2 * (dpct::dp4a(ui3, v3, 0) * s[2] + dpct::dp4a(ui4, v4, 0) * s[3]);

    واپس d * sumf_d;

#ورنہ
    bad_arch();
#اگر_ختم // __SYCL_ARCH__ >= VER_4VEC

#اگر_ختم
}

ساکن __dpct_inline__ اعشاری
vec_dot_q6_K_q8_1(ثابت خالی *__restrict__ vbq,
                  ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {

    ثابت بلاک_q6_K * bq6_K = (ثابت بلاک_q6_K *) vbq;

    ثابت صحیح bq8_انحراف = 2 * QR6_K * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/4);
    ثابت صحیح scale_انحراف = (QI6_K/4) * (iqs / (QI6_K/2)) + (iqs % (QI6_K/2)) / (QI6_K/8);
    ثابت صحیح vh_سرکاؤ = 2 * ((iqs % (QI6_K/2)) / (QI6_K/4));

    ثابت صحیح vl = get_int_from_uint8(bq6_K->ql, iqs);
    ثابت صحیح vh = get_int_from_uint8(bq6_K->qh, (QI6_K/4) * (iqs / (QI6_K/2)) + iqs % (QI6_K/4)) >> vh_سرکاؤ;

    ثابت int8_t * scales = bq6_K->scales + scale_انحراف;

    ثابت صحیح u0 = get_int_from_int8_aligned(
        bq8_1[bq8_انحراف + 0].qs, iqs % QI8_1);
    ثابت صحیح u1 = get_int_from_int8_aligned(
        bq8_1[bq8_انحراف + 2].qs, iqs % QI8_1);
    ثابت اعشاری d80 = bq8_1[bq8_انحراف + 0].ds[0];
    ثابت اعشاری d81 = bq8_1[bq8_انحراف + 2].ds[0];

    واپس vec_dot_q6_K_q8_1_impl_mmvq_scalar(
        vl, vh, u0, u1, scales[0], scales[4], bq6_K->d, d80, d81);
}


// NOTE: the VDR_IQ*_Q8_1_MMVQ values deliberately differ from the identically named CUDA constants
// (vecdotq.cuh): the SYCL kernels pair them with a halved qi (e.g. QI3_S/2), so the values are not
// interchangeable and must not be copied across backends.
#تعریف_کرو VDR_IQ2_XXS_Q8_1_MMVQ 1

ساکن __dpct_inline__ uint32_t iq_sign_نقاب(ثابت uint32_t signs) {
    ثابت uint32_t بٹ =
        (signs & 0x01) | ((signs & 0x02) << 7) |
        ((signs & 0x04) << 14) | ((signs & 0x08) << 21);

    واپس بٹ * 0xff;
}

ساکن __dpct_inline__ uint32_t iq_sign_بٹ(ثابت uint32_t encoded) {
    ثابت uint32_t signs = encoded & 127;

    واپس signs | ((sycl::popcount(signs) & 1) << 7);
}

ساکن __dpct_inline__ اعشاری
vec_dot_iq2_xxs_q8_1(ثابت خالی *__restrict__ vbq,
                     ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs,
                     ثابت uint64_t *iq2xxs_grid, ثابت uint8_t *ksigns_iq2xs,
                     ثابت uint8_t *kmask_iq2xs) {
#اگر QK_K == 256
    ثابت بلاک_iq2_xxs * bq2 = (ثابت بلاک_iq2_xxs *) vbq;

    ثابت صحیح ib32 = iqs;
    ثابت uint16_t * q2 = bq2->qs + 4*ib32;
    ثابت uint8_t  * aux8 = (ثابت uint8_t *)q2;
    ثابت int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = q2[2] | (q2[3] << 16);
    صحیح sumi = 0;
    برائے (صحیح l = 0; l < 4; ++l) {
        ثابت uint8_t * grid = (ثابت uint8_t *)(iq2xxs_grid + aux8[l]);
        ثابت uint8_t  signs = ksigns_iq2xs[aux32 & 127];
        برائے (صحیح j = 0; j < 8; ++j) {
            sumi += q8[j] * grid[j] * (signs & kmask_iq2xs[j] ? -1 : 1);
        }
        q8 += 8;
        aux32 >>= 7;
    }
    ثابت اعشاری d = (اعشاری)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.25f;
    واپس d * sumi;
#ورنہ
    أكد(جھوٹ);
    واپس 0.f;
#اگر_ختم
}

#تعریف_کرو VDR_IQ2_XS_Q8_1_MMVQ 1

ساکن __dpct_inline__ اعشاری
vec_dot_iq2_xs_q8_1(ثابت خالی *__restrict__ vbq,
                    ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs,
                    ثابت uint64_t *iq2xs_grid) {
#اگر DPCT_COMPATIBILITY_عارضی >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#اگر QK_K == 256
    ثابت بلاک_iq2_xs * bq2 = (ثابت بلاک_iq2_xs *) vbq;

    ثابت صحیح ib32 = iqs;
    ثابت uint16_t * q2 = bq2->qs + 4*ib32;
    ثابت int8_t   * q8 = bq8_1[ib32].qs;
    ثابت uint8_t ls1 = bq2->scales[ib32] & 0xf;
    ثابت uint8_t ls2 = bq2->scales[ib32] >>  4;
    صحیح sumi1 = 0;
#ہدایت unroll
    برائے (صحیح l = 0; l < 2; ++l) {
        ثابت uint64_t grid = iq2xs_grid[q2[l] & 511];
        ثابت uint32_t sign_بٹ = iq_sign_بٹ(q2[l] >> 9);
        ثابت uint32_t signs0 = iq_sign_نقاب(sign_بٹ);
        ثابت uint32_t signs1 = iq_sign_نقاب(sign_بٹ >> 4);
        ثابت صحیح grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        ثابت صحیح grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi1 = dpct::dp4a(grid_l, *((ثابت صحیح *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((ثابت صحیح *)q8 + 1), sumi1);
        q8 += 8;
    }
    صحیح sumi2 = 0;
#ہدایت unroll
    برائے (صحیح l = 2; l < 4; ++l) {
        ثابت uint64_t grid = iq2xs_grid[q2[l] & 511];
        ثابت uint32_t sign_بٹ = iq_sign_بٹ(q2[l] >> 9);
        ثابت uint32_t signs0 = iq_sign_نقاب(sign_بٹ);
        ثابت uint32_t signs1 = iq_sign_نقاب(sign_بٹ >> 4);
        ثابت صحیح grid_l = ((uint32_t)grid ^ signs0) + (signs0 & 0x01010101);
        ثابت صحیح grid_h = ((uint32_t)(grid >> 32) ^ signs1) + (signs1 & 0x01010101);
        sumi2 = dpct::dp4a(grid_l, *((ثابت صحیح *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((ثابت صحیح *)q8 + 1), sumi2);
        q8 += 8;
    }
    ثابت اعشاری d = (اعشاری)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    واپس d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#ورنہ
    أكد(جھوٹ);
    واپس 0.f;
#اگر_ختم
#ورنہ
    أكد(جھوٹ);
    واپس 0.f;
#اگر_ختم
}

#تعریف_کرو VDR_IQ2_S_Q8_1_MMVQ 1

ساکن __dpct_inline__ اعشاری
vec_dot_iq2_s_q8_1(ثابت خالی *__restrict__ vbq,
                   ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {
#اگر QK_K == 256
    ثابت بلاک_iq2_s * bq2 = (ثابت بلاک_iq2_s *) vbq;

    ثابت صحیح ib32 = iqs;
    ثابت int8_t  * q8 = bq8_1[ib32].qs;
    ثابت uint8_t * signs = bq2->qs + QK_K/8 + 4*ib32;
    ثابت uint8_t ls1 = bq2->scales[ib32] & 0xf;
    ثابت uint8_t ls2 = bq2->scales[ib32] >>  4;
    صحیح sumi1 = 0;
    برائے (صحیح l = 0; l < 2; ++l) {
        ثابت uint32_t * grid = (ثابت uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        ثابت uint32_t signs0 = dpct::vectorized_ثنائی<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ثابت uint32_t signs1 = dpct::vectorized_ثنائی<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ثابت صحیح grid_l = dpct::vectorized_ثنائی<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        ثابت صحیح grid_h = dpct::vectorized_ثنائی<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi1 = dpct::dp4a(grid_l, *((ثابت صحیح *)q8 + 0), sumi1);
        sumi1 = dpct::dp4a(grid_h, *((ثابت صحیح *)q8 + 1), sumi1);
        q8 += 8;
    }
    صحیح sumi2 = 0;
    برائے (صحیح l = 2; l < 4; ++l) {
        ثابت uint32_t * grid = (ثابت uint32_t *)(iq2s_grid + (bq2->qs[4*ib32+l] | ((bq2->qh[ib32] << (8-2*l)) & 0x300)));
        ثابت uint32_t signs0 = dpct::vectorized_ثنائی<sycl::uchar4>(
            ((signs[l] & 0xf) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ثابت uint32_t signs1 = dpct::vectorized_ثنائی<sycl::uchar4>(
            ((signs[l] >> 4) * 0x01010101) & 0x08040201, 0x08040201,
            std::equal_to<>());
        ثابت صحیح grid_l = dpct::vectorized_ثنائی<sycl::uchar4>(
            grid[0] ^ signs0, signs0, std::minus<>());
        ثابت صحیح grid_h = dpct::vectorized_ثنائی<sycl::uchar4>(
            grid[1] ^ signs1, signs1, std::minus<>());
        sumi2 = dpct::dp4a(grid_l, *((ثابت صحیح *)q8 + 0), sumi2);
        sumi2 = dpct::dp4a(grid_h, *((ثابت صحیح *)q8 + 1), sumi2);
        q8 += 8;
    }
    ثابت اعشاری d = (اعشاری)bq2->d * bq8_1[ib32].ds[0] * 0.25f;
    واپس d * ((0.5f + ls1) * sumi1 + (0.5f + ls2) * sumi2);
#ورنہ
    أكد(جھوٹ);
#اگر_ختم
}

#تعریف_کرو VDR_IQ3_XXS_Q8_1_MMVQ 1

ساکن __dpct_inline__ اعشاری
vec_dot_iq3_xxs_q8_1(ثابت خالی *__restrict__ vbq,
                     ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs,
                     ثابت uint32_t *iq3xxs_grid) {
#اگر DPCT_COMPATIBILITY_عارضی >=                                                 \
    MIN_CC_DP4A // lowest compute capability for integer intrinsics
#اگر QK_K == 256
    ثابت بلاک_iq3_xxs * bq2 = (ثابت بلاک_iq3_xxs *) vbq;

    ثابت صحیح ib32 = iqs;
    ثابت uint8_t  * q3 = bq2->qs + 8*ib32;
    ثابت uint16_t * gas = (ثابت uint16_t *)(bq2->qs + QK_K/4) + 2*ib32;
    ثابت int8_t   * q8 = bq8_1[ib32].qs;
    uint32_t aux32 = gas[0] | (gas[1] << 16);
    صحیح sumi = 0;
#ہدایت unroll
    برائے (صحیح l = 0; l < 4; ++l) {
        ثابت uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
        ثابت uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
        ثابت uint32_t sign_بٹ = iq_sign_بٹ(aux32);
        ثابت uint32_t signs0 = iq_sign_نقاب(sign_بٹ);
        ثابت uint32_t signs1 = iq_sign_نقاب(sign_بٹ >> 4);
        ثابت صحیح grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        ثابت صحیح grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((ثابت صحیح *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((ثابت صحیح *)q8 + 1), sumi);
        q8 += 8;
        aux32 >>= 7;
    }
    ثابت اعشاری d = (اعشاری)bq2->d * (0.5f + aux32) * bq8_1[ib32].ds[0] * 0.5f;
    واپس d * sumi;
#ورنہ
    أكد(جھوٹ);
    واپس 0.f;
#اگر_ختم
#ورنہ
    أكد(جھوٹ);
    واپس 0.f;
#اگر_ختم
}

سانچہ <> ساخت reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_XXS> {
    ساکن constexpr ggml_type gtype = GGML_TYPE_IQ3_XXS;

    __dpct_inline__ اعشاری عامل()(ثابت خالی * __restrict__ vbq, ثابت std::pair<صحیح, صحیح> ibx_انحراف,
                                     ثابت std::pair<صحیح, صحیح> d_انحراف, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحیح & iqs) {
#اگر DPCT_COMPATIBILITY_عارضی >= MIN_CC_DP4A && QK_K == 256
        ثابت uint8_t * بنیاد = static_cast<ثابت uint8_t *>(vbq);
        ثابت uint8_t * qs   = بنیاد + ibx_انحراف.پہلا;
        ثابت uint8_t * q3   = qs + 8 * iqs;
        ثابت uint16_t * gas = reinterpret_cast<ثابت uint16_t *>(qs + QK_K / 4) + 2 * iqs;
        ثابت int8_t * q8    = q8_1_quant_ptr + iqs * QK8_1;
        uint32_t aux32       = gas[0] | (gas[1] << 16);

        صحیح sumi = 0;
#ہدایت unroll
        برائے (صحیح l = 0; l < 4; ++l) {
            ثابت uint32_t grid1 = iq3xxs_grid[q3[2*l+0]];
            ثابت uint32_t grid2 = iq3xxs_grid[q3[2*l+1]];
            ثابت uint32_t sign_بٹ = iq_sign_بٹ(aux32);
            ثابت uint32_t signs0 = iq_sign_نقاب(sign_بٹ);
            ثابت uint32_t signs1 = iq_sign_نقاب(sign_بٹ >> 4);
            ثابت صحیح grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            ثابت صحیح grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((ثابت صحیح *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((ثابت صحیح *) q8 + 1), sumi);
            q8 += 8;
            aux32 >>= 7;
        }

        ثابت ggml_half d = *reinterpret_cast<ثابت ggml_half *>(بنیاد + d_انحراف.پہلا);
        واپس (اعشاری) d * (0.5f + aux32) * q8_1_ds[iqs][0] * 0.5f * sumi;
#ورنہ
        GGML_غیر_مستعمل(vbq);
        GGML_غیر_مستعمل(ibx_انحراف);
        GGML_غیر_مستعمل(d_انحراف);
        GGML_غیر_مستعمل(q8_1_quant_ptr);
        GGML_غیر_مستعمل(q8_1_ds);
        GGML_غیر_مستعمل(iqs);
        GGML_ABORT("IQ3_XXS reorder vec dot requires dp4a and QK_K == 256");
#اگر_ختم
    }
};

#تعریف_کرو VDR_IQ3_S_Q8_1_MMVQ 1

ساکن __dpct_inline__ اعشاری
vec_dot_iq3_s_q8_1(ثابت خالی *__restrict__ vbq,
                   ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs,
                   ثابت uint32_t *iq3s_grid) {
#اگر QK_K == 256
    ثابت بلاک_iq3_s * bq2 = (ثابت بلاک_iq3_s *) vbq;

    ثابت صحیح ib32 = iqs;
    ثابت uint8_t  * qs = bq2->qs + 8*ib32;
    ثابت int8_t   * q8 = bq8_1[ib32].qs;
    صحیح sumi = 0;
    برائے (صحیح l = 0; l < 4; ++l) {
        ثابت uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((bq2->qh[ib32] << (8 - 2*l)) & 256)];
        ثابت uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((bq2->qh[ib32] << (7 - 2*l)) & 256)];
        ثابت uint32_t signs = bq2->signs[4 * ib32 + l];
        ثابت uint32_t signs0 = iq_sign_نقاب(signs);
        ثابت uint32_t signs1 = iq_sign_نقاب(signs >> 4);
        ثابت صحیح grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
        ثابت صحیح grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
        sumi = dpct::dp4a(grid_l, *((ثابت صحیح *)q8 + 0), sumi);
        sumi = dpct::dp4a(grid_h, *((ثابت صحیح *)q8 + 1), sumi);
        q8 += 8;
    }
    ثابت اعشاری d =
        (اعشاری)bq2->d *
        (1 + 2 * ((bq2->scales[ib32 / 2] >> 4 * (ib32 % 2)) & 0xf)) *
        bq8_1[ib32].ds[0];
    واپس d * sumi;
#ورنہ
    أكد(جھوٹ);
#اگر_ختم
}

سانچہ <> ساخت reorder_vec_dot_q_sycl<GGML_TYPE_IQ3_S> {
    ساکن constexpr ggml_type gtype = GGML_TYPE_IQ3_S;

    __dpct_inline__ اعشاری عامل()(ثابت خالی * __restrict__ vbq, ثابت std::pair<صحیح, صحیح> ibx_انحراف,
                                     ثابت std::pair<صحیح, صحیح> d_انحراف, ثابت int8_t * q8_1_quant_ptr,
                                     ثابت sycl::half2 * q8_1_ds, ثابت صحیح & iqs) {
#اگر QK_K == 256
        ثابت uint8_t * بنیاد     = static_cast<ثابت uint8_t *>(vbq);
        ثابت uint8_t * qs       = بنیاد + ibx_انحراف.پہلا + 8 * iqs;
        ثابت uint8_t * qh       = بنیاد + ibx_انحراف.second;
        ثابت uint8_t * signs    = بنیاد + d_انحراف.پہلا + 4 * iqs;
        ثابت uint8_t * metadata = بنیاد + d_انحراف.second;
        ثابت ggml_half d        = *reinterpret_cast<ثابت ggml_half *>(metadata);
        ثابت uint8_t * scales   = metadata + حجم(ggml_half);
        ثابت int8_t * q8        = q8_1_quant_ptr + iqs * QK8_1;

        صحیح sumi = 0;
#ہدایت unroll
        برائے (صحیح l = 0; l < 4; ++l) {
            ثابت uint32_t grid1 = iq3s_grid[qs[2*l+0] | ((qh[iqs] << (8 - 2*l)) & 256)];
            ثابت uint32_t grid2 = iq3s_grid[qs[2*l+1] | ((qh[iqs] << (7 - 2*l)) & 256)];
            ثابت uint32_t signs0 = iq_sign_نقاب(signs[l]);
            ثابت uint32_t signs1 = iq_sign_نقاب(signs[l] >> 4);
            ثابت صحیح grid_l = (grid1 ^ signs0) + (signs0 & 0x01010101);
            ثابت صحیح grid_h = (grid2 ^ signs1) + (signs1 & 0x01010101);
            sumi = dpct::dp4a(grid_l, *((ثابت صحیح *) q8 + 0), sumi);
            sumi = dpct::dp4a(grid_h, *((ثابت صحیح *) q8 + 1), sumi);
            q8 += 8;
        }

        ثابت اعشاری scale = 1 + 2 * ((scales[iqs / 2] >> (4 * (iqs % 2))) & 0xf);
        واپس (اعشاری) d * scale * q8_1_ds[iqs][0] * sumi;
#ورنہ
        GGML_غیر_مستعمل(vbq);
        GGML_غیر_مستعمل(ibx_انحراف);
        GGML_غیر_مستعمل(d_انحراف);
        GGML_غیر_مستعمل(q8_1_quant_ptr);
        GGML_غیر_مستعمل(q8_1_ds);
        GGML_غیر_مستعمل(iqs);
        GGML_ABORT("IQ3_S reorder vec dot not supported for QK_K != 256");
#اگر_ختم
    }
};

#تعریف_کرو VDR_IQ1_S_Q8_1_MMVQ 1

ساکن __dpct_inline__ اعشاری
vec_dot_iq1_s_q8_1(ثابت خالی *__restrict__ vbq,
                   ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs,
                   ثابت uint32_t *iq1s_grid_gpu) {
#اگر QK_K == 256
    ثابت بلاک_iq1_s * bq1 = (ثابت بلاک_iq1_s *) vbq;

    ثابت صحیح ib32 = iqs;
    صحیح sumi = 0;
    ثابت صحیح * q8 = (ثابت صحیح *)bq8_1[ib32].qs;
    برائے (صحیح l = 0; l < 4; ++l) {
        ثابت صحیح * grid = (ثابت صحیح *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[ib32] >> 3*l) & 7) << 8)));
        صحیح grid0 = grid[0] & 0x0f0f0f0f;
        صحیح grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi = dpct::dp4a(q8[2 * l + 1], grid1,
                          dpct::dp4a(q8[2 * l + 0], grid0, sumi));
    }

    ثابت اعشاری delta = bq1->qh[ib32] & 0x8000 ? -1-IQ1S_DELTA : -1+IQ1S_DELTA;
    ثابت اعشاری d1q = (اعشاری)bq1->d * (2*((bq1->qh[ib32] >> 12) & 7) + 1);
    ثابت اعشاری d = d1q * bq8_1[ib32].ds[0];
    ثابت اعشاری m = d1q * bq8_1[ib32].ds[1];
    واپس d * sumi + m * delta;
#ورنہ
    أكد(جھوٹ);
#اگر_ختم
}

#تعریف_کرو VDR_IQ1_M_Q8_1_MMVQ 1

ساکن __dpct_inline__ اعشاری
vec_dot_iq1_m_q8_1(ثابت خالی *__restrict__ vbq,
                   ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {
#اگر QK_K == 256
    ثابت بلاک_iq1_m * bq1 = (ثابت بلاک_iq1_m *) vbq;

    ثابت صحیح ib32 = iqs;
    صحیح   sumi[2] = {0, 0};
    اعشاری sumf[2] = {0.f, 0.f};

    ثابت صحیح * q8 = (ثابت صحیح *)bq8_1[ib32].qs;
    برائے (صحیح l = 0; l < 4; ++l) {
        ثابت صحیح * grid = (ثابت صحیح *)(iq1s_grid_gpu + (bq1->qs[4*ib32+l] | (((bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 7) << 8)));
        صحیح grid0 = grid[0] & 0x0f0f0f0f;
        صحیح grid1 = (grid[0] >> 4) & 0x0f0f0f0f;
        sumi[l / 2] = dpct::dp4a(q8[2 * l + 1], grid1,
                                 dpct::dp4a(q8[2 * l + 0], grid0, sumi[l / 2]));
        ثابت اعشاری delta = (bq1->qh[2*ib32+l/2] >> 4*(l%2)) & 0x08 ? -1-IQ1M_DELTA : -1+IQ1M_DELTA;
        ثابت صحیح sumy = dpct::dp4a(q8[2 * l + 1], 0x01010101,
                                    dpct::dp4a(q8[2 * l + 0], 0x01010101, 0));
        sumf[l/2] += delta*sumy;
    }

    iq1m_scale_t scale;
    ثابت uint16_t * sc = (ثابت uint16_t *)bq1->scales;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    ثابت اعشاری d = (اعشاری)scale.f16 * bq8_1[ib32].ds[0];
    واپس d * ((sumi[0] + sumf[0]) * (2*((sc[ib32/2] >> 6*(ib32%2)) & 0x7) + 1) + (sumi[1] + sumf[1]) * (2*((sc[ib32/2] >> (6*(ib32%2)+3)) & 0x7) + 1));
#ورنہ
    أكد(جھوٹ);
#اگر_ختم
}


#تعریف_کرو VDR_IQ4_NL_Q8_1_MMVQ 2

ساکن __dpct_inline__ اعشاری
vec_dot_iq4_nl_q8_1(ثابت خالی *__restrict__ vbq,
                    ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {

    ثابت بلاک_iq4_nl * bq = (ثابت بلاک_iq4_nl *) vbq;

    ثابت uint16_t * q4 = (ثابت uint16_t *)bq->qs + 2*iqs;
    ثابت int32_t  * q8 = (ثابت int32_t  *)bq8_1->qs + iqs;

    ثابت uint8_t * اقدار = (ثابت uint8_t *)kvalues_iq4nl;

    صحیح v1, v2;
    صحیح sumi1 = 0, sumi2 = 0;
    برائے (صحیح l = 0; l < VDR_Q4_0_Q8_1_MMVQ; ++l) {
        ثابت uint32_t aux = q4[2*l] | (q4[2*l+1] << 16);
        get_int_from_جدول_16(aux, اقدار, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[l + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[l + 4], sumi2);
    }

    ثابت اعشاری d = (اعشاری)bq->d * bq8_1->ds[0];
    واپس d * (sumi1 + sumi2);
}


#تعریف_کرو VDR_IQ4_XS_Q8_1_MMVQ 1

ساکن __dpct_inline__ اعشاری
vec_dot_iq4_xs_q8_1(ثابت خالی *__restrict__ vbq,
                    ثابت بلاک_q8_1 *__restrict__ bq8_1, ثابت صحیح &iqs) {

#اگر QK_K == 256
    ثابت بلاک_iq4_xs * bq4 = (ثابت بلاک_iq4_xs *) vbq;
    ثابت uint8_t * اقدار = (ثابت uint8_t *)kvalues_iq4nl;

    // iqs is 0...7
    ثابت صحیح ib32 = iqs;
    ثابت int32_t  * q8 = (ثابت صحیح *)bq8_1[ib32].qs;
    ثابت uint32_t * q4 = (ثابت uint32_t *)bq4->qs + 4*ib32;
    ثابت int8_t ls = ((bq4->scales_l[ib32/2] >> 4*(ib32%2)) & 0xf) | (((bq4->scales_h >> 2*ib32) & 3) << 4);
    ثابت اعشاری d = (اعشاری)bq4->d * (ls - 32) * bq8_1[ib32].ds[0];
    صحیح v1, v2;
    صحیح sumi1 = 0, sumi2 = 0;
    برائے (صحیح j = 0; j < 4; ++j) {
        get_int_from_جدول_16(q4[j], اقدار, v1, v2);
        sumi1 = dpct::dp4a(v1, q8[j + 0], sumi1);
        sumi2 = dpct::dp4a(v2, q8[j + 4], sumi2);
    }
    واپس d * (sumi1 + sumi2);
#ورنہ
    أكد(جھوٹ);
#اگر_ختم
}

#اگر_ختم // GGML_SYCL_VECDOTQ_HPP
