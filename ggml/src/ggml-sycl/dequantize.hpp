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

#اگر_غیر_تعریف_شدہ GGML_SYCL_DEQUANTIZE_HPP
#تعریف_کرو GGML_SYCL_DEQUANTIZE_HPP

#شامل "common.hpp"
#شامل "convert.hpp"

قسمتعریف خالی (*dequantize_kernel_t)(ثابت خالی * vx, ثابت int64_t ib, ثابت صحیح iqs, dfloat2 & v);
قسمتعریف خالی (*dequantize_kernel_t_reorder)(ثابت خالی *d, ثابت int64_t ib, ثابت خالی *qs,
                                            ثابت صحیح iqs, dfloat2 &v);
قسمتعریف خالی (*dequantize_kernel_f32_t)(ثابت خالی * vx, ثابت int64_t ib, ثابت صحیح iqs, اعشاری & v0, اعشاری & v1);

#اگر QK_K == 256
ساکن اندرونی خالی get_scale_min_k4(صحیح j, ثابت uint8_t * q, uint8_t & d, uint8_t & m);
#اگر_ختم

ساکن __dpct_inline__ خالی dequantize_q2_0(ثابت خالی *vx, ثابت int64_t ib,
                                            ثابت صحیح iqs, dfloat2 &v) {
    ثابت بلاک_q2_0 * x = (ثابت بلاک_q2_0 *) vx;

    ثابت dfloat d = x[ib].d;

    ثابت صحیح byte_idx = iqs / 4;
    ثابت صحیح سرکاؤ = (iqs % 4) * 2;
    ثابت uint8_t vui = x[ib].qs[byte_idx];

    v.x() = (vui >> سرکاؤ) & 3;
    v.y() = (vui >> (سرکاؤ + 2)) & 3;

#اگر_تعریف_شدہ GGML_SYCL_F16
    v.s0() = ((dfloat)v.s0() - 1.0f) * d;
    v.s1() = ((dfloat)v.s1() - 1.0f) * d;
#ورنہ
    v.x() = ((dfloat)v.x() - 1.0f) * d;
    v.y() = ((dfloat)v.y() - 1.0f) * d;
#اگر_ختم // GGML_SYCL_F16
}

ساکن __dpct_inline__ خالی dequantize_q4_0(ثابت خالی *vx, ثابت int64_t ib,
                                            ثابت صحیح iqs, dfloat2 &v) {
    ثابت بلاک_q4_0 * x = (ثابت بلاک_q4_0 *) vx;

    ثابت dfloat d = x[ib].d;

    ثابت صحیح vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#اگر_تعریف_شدہ GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#ورنہ
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#اگر_ختم // GGML_SYCL_F16
}

ساکن __dpct_inline__ خالی dequantize_q4_0_reorder(ثابت خالی *d_ptr, ثابت int64_t ib, ثابت خالی *qs,
                                            ثابت صحیح iqs, dfloat2 &v) {
    // const block_q4_0 * x = (const block_q4_0 *) vx;

    ثابت dfloat d = (ثابت dfloat)*((ثابت sycl::half*)d_ptr+ib);

    ثابت صحیح vui = *((ثابت uint8_t *)qs+iqs);

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#اگر_تعریف_شدہ GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#ورنہ
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#اگر_ختم // GGML_SYCL_F16
}

ساکن __dpct_inline__ خالی dequantize_q1_0_reorder(ثابت خالی *d_ptr, ثابت int64_t ib, ثابت خالی *qs,
                                            ثابت صحیح iqs, dfloat2 &v) {
    // Q1_0 reorder layout: scale values followed by quantized bits
    ثابت dfloat d = (ثابت dfloat)*((ثابت sycl::half*)d_ptr+ib);

    ثابت صحیح ثنائی_ہندسہ_اشاریہ_0 = iqs + 0;
    ثابت صحیح ثنائی_ہندسہ_اشاریہ_1 = iqs + 1;

    ثابت صحیح ثنائی_ہندسہ_0 = (*((ثابت uint8_t *)qs + ثنائی_ہندسہ_اشاریہ_0 / 8) >> (ثنائی_ہندسہ_اشاریہ_0 % 8)) & 1;
    ثابت صحیح ثنائی_ہندسہ_1 = (*((ثابت uint8_t *)qs + ثنائی_ہندسہ_اشاریہ_1 / 8) >> (ثنائی_ہندسہ_اشاریہ_1 % 8)) & 1;

    v.x() = (2 * ثنائی_ہندسہ_0 - 1) * d;
    v.y() = (2 * ثنائی_ہندسہ_1 - 1) * d;
}

ساکن __dpct_inline__ خالی dequantize_q1_0(ثابت خالی *vx, ثابت int64_t ib,
                                            ثابت صحیح iqs, dfloat2 &v) {
    ثابت بلاک_q1_0 * x = (ثابت بلاک_q1_0 *) vx;
    ثابت dfloat d = x[ib].d;

    ثابت صحیح ثنائی_ہندسہ_اشاریہ_0 = iqs + 0;
    ثابت صحیح ثنائی_ہندسہ_اشاریہ_1 = iqs + 1;

    ثابت صحیح ثنائی_ہندسہ_0 = (x[ib].qs[ثنائی_ہندسہ_اشاریہ_0 / 8] >> (ثنائی_ہندسہ_اشاریہ_0 % 8)) & 1;
    ثابت صحیح ثنائی_ہندسہ_1 = (x[ib].qs[ثنائی_ہندسہ_اشاریہ_1 / 8] >> (ثنائی_ہندسہ_اشاریہ_1 % 8)) & 1;

    v.x() = (2 * ثنائی_ہندسہ_0 - 1) * d;
    v.y() = (2 * ثنائی_ہندسہ_1 - 1) * d;
}

ساکن __dpct_inline__ خالی dequantize_q4_1(ثابت خالی *vx, ثابت int64_t ib,
                                            ثابت صحیح iqs, dfloat2 &v) {
    ثابت بلاک_q4_1 * x = (ثابت بلاک_q4_1 *) vx;

    ثابت dfloat d = x[ib].dm[0];
    ثابت dfloat m = x[ib].dm[1];

    ثابت صحیح vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#اگر_تعریف_شدہ GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = sycl::fma(v.s0(), d, m);
    v.s1() = sycl::fma(v.s1(), d, m);

#ورنہ
    v.x() = sycl::fma(v.x(), d, m);
    v.y() = sycl::fma(v.y(), d, m);
#اگر_ختم // GGML_SYCL_F16
}

ساکن __dpct_inline__ خالی dequantize_q4_K(ثابت خالی *vx, ثابت int64_t ib,
                                            ثابت صحیح iqs, dfloat2 &v) {
#اگر QK_K == 256
    ثابت بلاک_q4_K * x = (ثابت بلاک_q4_K *) vx;
    ثابت sycl::half2 dm = x[ib].dm;
    ثابت اعشاری dall = dm[0];
    ثابت اعشاری dmin = dm[1];

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        ثابت صحیح il = idx / 64;
        ثابت صحیح اندر = idx % 64;
        ثابت صحیح is = 2 * il + (اندر >= 32 ? 1 : 0);
        ثابت صحیح off = اندر & 31;
        ثابت صحیح qsi = 32 * il + off;

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ثابت uint8_t q = x[ib].qs[qsi];
        ثابت uint8_t qv = (اندر >= 32) ? (q >> 4) : (q & 0xF);
        واپس sycl::fma((dfloat) qv, (dfloat) (dall * sc), (dfloat) (-dmin * m));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("Q4_K dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_q4_K_f32(ثابت خالی *vx, ثابت int64_t ib,
                                                ثابت صحیح iqs, اعشاری &v0, اعشاری &v1) {
#اگر QK_K == 256
    ثابت بلاک_q4_K * x = (ثابت بلاک_q4_K *) vx;
    ثابت sycl::half2 dm = x[ib].dm;
    ثابت اعشاری dall = dm[0];
    ثابت اعشاری dmin = dm[1];

    خودکار dequantize_one = [&](ثابت صحیح idx) -> اعشاری {
        ثابت صحیح il = idx / 64;
        ثابت صحیح اندر = idx % 64;
        ثابت صحیح is = 2 * il + (اندر >= 32 ? 1 : 0);
        ثابت صحیح qsi = 32 * il + (اندر & 31);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ثابت اعشاری d = dall * sc;
        ثابت اعشاری mn = dmin * m;
        ثابت uint8_t q = x[ib].qs[qsi];
        ثابت uint8_t qv = (اندر >= 32) ? (q >> 4) : (q & 0xF);

        واپس d * qv - mn;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("Q4_K dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_q2_K(ثابت خالی *vx, ثابت int64_t ib,
                                            ثابت صحیح iqs, dfloat2 &v) {
#اگر QK_K == 256
    ثابت بلاک_q2_K * x = (ثابت بلاک_q2_K *) vx;
    ثابت اعشاری dall = x[ib].dm[0];
    ثابت اعشاری dmin = x[ib].dm[1];

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        ثابت صحیح n = idx / 128;
        ثابت صحیح r = idx % 128;
        ثابت صحیح g = r / 32;
        ثابت صحیح l = r % 32;
        ثابت صحیح is = 8 * n + l / 16;

        ثابت uint8_t q = x[ib].qs[32 * n + l];
        ثابت uint8_t sc = x[ib].scales[is + 2 * g];
        ثابت اعشاری d = dall * (sc & 0xF);
        ثابت اعشاری m = dmin * (sc >> 4);

        واپس (dfloat) d * (dfloat) ((q >> (2 * g)) & 3) - (dfloat) m;
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("Q2_K dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_q2_K_f32(ثابت خالی *vx, ثابت int64_t ib,
                                                ثابت صحیح iqs, اعشاری &v0, اعشاری &v1) {
#اگر QK_K == 256
    ثابت بلاک_q2_K * x = (ثابت بلاک_q2_K *) vx;
    ثابت اعشاری dall = x[ib].dm[0];
    ثابت اعشاری dmin = x[ib].dm[1];

    خودکار dequantize_one = [&](ثابت صحیح idx) -> اعشاری {
        ثابت صحیح n = idx / 128;
        ثابت صحیح r = idx % 128;
        ثابت صحیح g = r / 32;
        ثابت صحیح l = r % 32;
        ثابت صحیح is = 8 * n + l / 16;

        ثابت uint8_t q = x[ib].qs[32 * n + l];
        ثابت uint8_t sc = x[ib].scales[is + 2 * g];
        ثابت اعشاری d = dall * (sc & 0xF);
        ثابت اعشاری m = dmin * (sc >> 4);

        واپس d * ((q >> (2 * g)) & 3) - m;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("Q2_K dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_q3_K(ثابت خالی *vx, ثابت int64_t ib,
                                            ثابت صحیح iqs, dfloat2 &v) {
#اگر QK_K == 256
    ثابت بلاک_q3_K * x = (ثابت بلاک_q3_K *) vx;
    ثابت اعشاری d_all = x[ib].d;

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        ثابت صحیح n = idx / 128;
        ثابت صحیح r = idx % 128;
        ثابت صحیح j = r / 32;
        ثابت صحیح l = r % 32;

        ثابت صحیح is0 = l / 16;
        ثابت صحیح is = 8 * n + 2 * j + is0;
        ثابت صحیح سرکاؤ = 2 * j;
        ثابت uint8_t m = 1 << (4 * n + j);

        ثابت int8_t us = is <  4 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 8] >> 0) & 3) << 4) :
                         is <  8 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 4] >> 2) & 3) << 4) :
                         is < 12 ? (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is + 0] >> 4) & 3) << 4) :
                                   (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is - 4] >> 6) & 3) << 4);

        ثابت اعشاری dl = d_all * (us - 32);
        ثابت uint8_t q = x[ib].qs[32 * n + l];
        ثابت uint8_t h = x[ib].hmask[l];
        ثابت int8_t qv = ((q >> سرکاؤ) & 3) - ((h & m) ? 0 : 4);

        واپس (dfloat) (dl * qv);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("Q3_K dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_q5_K(ثابت خالی *vx, ثابت int64_t ib,
                                            ثابت صحیح iqs, dfloat2 &v) {
#اگر QK_K == 256
    ثابت بلاک_q5_K * x = (ثابت بلاک_q5_K *) vx;
    ثابت اعشاری dall = x[ib].dm[0];
    ثابت اعشاری dmin = x[ib].dm[1];

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        ثابت صحیح il = idx / 64;
        ثابت صحیح اندر = idx % 64;
        ثابت صحیح is = 2 * il + (اندر >= 32 ? 1 : 0);
        ثابت صحیح ir = (اندر & 31) / 2;
        ثابت صحیح iq = اندر & 1;

        ثابت uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        ثابت uint8_t h = x[ib].qh[2 * ir + iq];
        ثابت uint8_t qv = (اندر >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ثابت اعشاری d = dall * sc;
        ثابت اعشاری mn = dmin * m;
        ثابت uint8_t hm = 1 << (2 * il + (اندر >= 32 ? 1 : 0));

        واپس sycl::fma((dfloat) (qv + ((h & hm) ? 16 : 0)), (dfloat) d, (dfloat) (-mn));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("Q5_K dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_q5_K_f32(ثابت خالی *vx, ثابت int64_t ib,
                                                ثابت صحیح iqs, اعشاری &v0, اعشاری &v1) {
#اگر QK_K == 256
    ثابت بلاک_q5_K * x = (ثابت بلاک_q5_K *) vx;
    ثابت اعشاری dall = x[ib].dm[0];
    ثابت اعشاری dmin = x[ib].dm[1];

    خودکار dequantize_one = [&](ثابت صحیح idx) -> اعشاری {
        ثابت صحیح il = idx / 64;
        ثابت صحیح اندر = idx % 64;
        ثابت صحیح is = 2 * il + (اندر >= 32 ? 1 : 0);
        ثابت صحیح ir = (اندر & 31) / 2;
        ثابت صحیح iq = اندر & 1;

        ثابت uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        ثابت uint8_t h = x[ib].qh[2 * ir + iq];
        ثابت uint8_t qv = (اندر >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        ثابت اعشاری d = dall * sc;
        ثابت اعشاری mn = dmin * m;
        ثابت uint8_t hm = 1 << (2 * il + (اندر >= 32 ? 1 : 0));

        واپس (qv + ((h & hm) ? 16 : 0)) * d - mn;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("Q5_K dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_q6_K(ثابت خالی *vx, ثابت int64_t ib,
                                            ثابت صحیح iqs, dfloat2 &v) {
#اگر QK_K == 256
    ثابت بلاک_q6_K * x = (ثابت بلاک_q6_K *) vx;
    ثابت اعشاری d = x[ib].d;

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        ثابت صحیح ip = idx / 128;
        ثابت صحیح اندر = idx % 128;
        ثابت صحیح il = اندر & 31;
        ثابت صحیح ig = اندر / 32;
        ثابت صحیح is = 8 * ip + il / 16;

        ثابت uint8_t ql0 = x[ib].ql[64 * ip + il];
        ثابت uint8_t ql1 = x[ib].ql[64 * ip + il + 32];
        ثابت uint8_t qh = x[ib].qh[32 * ip + il];
        ثابت int8_t * sc = x[ib].scales + is;

        uint8_t qv;
        int8_t scale;
        اگر (ig == 0) {
            qv = (ql0 & 0xF) | (((qh >> 0) & 3) << 4);
            scale = sc[0];
        } ورنہ اگر (ig == 1) {
            qv = (ql1 & 0xF) | (((qh >> 2) & 3) << 4);
            scale = sc[2];
        } ورنہ اگر (ig == 2) {
            qv = (ql0 >> 4) | (((qh >> 4) & 3) << 4);
            scale = sc[4];
        } ورنہ {
            qv = (ql1 >> 4) | (((qh >> 6) & 3) << 4);
            scale = sc[6];
        }

        واپس (dfloat) (d * scale * ((int8_t) qv - 32));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("Q6_K dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_mxfp4(ثابت خالی *vx, ثابت int64_t ib,
                                             ثابت صحیح iqs, dfloat2 &v) {
    ثابت بلاک_mxfp4 * x = (ثابت بلاک_mxfp4 *) vx;
    ثابت اعشاری d = ggml_sycl_e8m0_مقصد_fp32(x[ib].e);
    ثابت uint8_t q = x[ib].qs[iqs];

    v.x() = d * kvalues_mxfp4[q & 0xF] * 0.5f;
    v.y() = d * kvalues_mxfp4[q >> 4] * 0.5f;
}

ساکن __dpct_inline__ خالی dequantize_nvfp4(ثابت خالی *vx, ثابت int64_t ib,
                                             ثابت صحیح iqs, dfloat2 &v) {
    ثابت بلاک_nvfp4 & xb = ((ثابت بلاک_nvfp4 *) vx)[ib];

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        ثابت صحیح sub = idx / QK_NVFP4_SUB;
        ثابت صحیح j = idx % QK_NVFP4_SUB;
        ثابت صحیح jh = j % (QK_NVFP4_SUB / 2);

        ثابت اعشاری d = ggml_sycl_ue4m3_مقصد_fp32(xb.d[sub]);
        ثابت uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + jh];
        ثابت uint8_t qv = (j < (QK_NVFP4_SUB / 2)) ? (q & 0x0F) : (q >> 4);

        واپس d * kvalues_mxfp4[qv];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
}

ساکن __dpct_inline__ خالی dequantize_iq2_xxs(ثابت خالی *vx, ثابت int64_t ib,
                                               ثابت صحیح iqs, dfloat2 &v) {
#اگر QK_K == 256
    ثابت بلاک_iq2_xxs * x = (ثابت بلاک_iq2_xxs *) vx;

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        ثابت صحیح ib8 = idx / 32;
        ثابت صحیح r = idx % 32;
        ثابت صحیح il = r / 8;
        ثابت صحیح j = r % 8;

        ثابت uint16_t * q2 = x[ib].qs + 4 * ib8;
        ثابت uint8_t * aux8 = (ثابت uint8_t *) q2;
        ثابت uint8_t * grid = (ثابت uint8_t *) (iq2xxs_grid + aux8[il]);
        ثابت uint32_t aux32 = q2[2] | (q2[3] << 16);
        ثابت اعشاری d = (اعشاری) x[ib].d * (0.5f + (aux32 >> 28)) * 0.25f;
        ثابت uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        واپس d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("IQ2_XXS dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_iq2_xs(ثابت خالی *vx, ثابت int64_t ib,
                                              ثابت صحیح iqs, dfloat2 &v) {
#اگر QK_K == 256
    ثابت بلاک_iq2_xs * x = (ثابت بلاک_iq2_xs *) vx;

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        ثابت صحیح ib8 = idx / 32;
        ثابت صحیح r = idx % 32;
        ثابت صحیح il = r / 8;
        ثابت صحیح j = r % 8;

        ثابت uint16_t * q2 = x[ib].qs + 4 * ib8;
        ثابت uint8_t * grid = (ثابت uint8_t *) (iq2xs_grid + (q2[il] & 511));
        ثابت اعشاری d = (اعشاری) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ثابت uint8_t signs = ksigns_iq2xs[q2[il] >> 9];

        واپس d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("IQ2_XS dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_iq2_s(ثابت خالی *vx, ثابت int64_t ib,
                                             ثابت صحیح iqs, dfloat2 &v) {
#اگر QK_K == 256
    ثابت بلاک_iq2_s * x = (ثابت بلاک_iq2_s *) vx;

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        ثابت صحیح ib8 = idx / 32;
        ثابت صحیح r = idx % 32;
        ثابت صحیح il = r / 8;
        ثابت صحیح j = r % 8;

        ثابت uint16_t grid_id = x[ib].qs[4 * ib8 + il] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 0x300);
        ثابت uint8_t * grid = (ثابت uint8_t *) (iq2s_grid + grid_id);
        ثابت اعشاری d = (اعشاری) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        ثابت uint8_t signs = x[ib].qs[QK_K / 8 + 4 * ib8 + il];

        واپس d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("IQ2_S dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_iq3_xxs(ثابت خالی *vx, ثابت int64_t ib,
                                               ثابت صحیح iqs, dfloat2 &v) {
#اگر QK_K == 256
    ثابت بلاک_iq3_xxs * x = (ثابت بلاک_iq3_xxs *) vx;

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        ثابت صحیح ib8 = idx / 32;
        ثابت صحیح r = idx % 32;
        ثابت صحیح il = r / 8;
        ثابت صحیح j = r % 8;

        ثابت uint8_t * q3 = x[ib].qs + 8 * ib8;
        ثابت uint16_t * gas = (ثابت uint16_t *) (x[ib].qs + QK_K / 4) + 2 * ib8;
        ثابت uint8_t * grid1 = (ثابت uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        ثابت uint8_t * grid2 = (ثابت uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        ثابت uint32_t aux32 = gas[0] | (gas[1] << 16);
        ثابت اعشاری d = (اعشاری) x[ib].d * (0.5f + (aux32 >> 28)) * 0.5f;
        ثابت uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        اگر (j < 4) {
            واپس d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        واپس d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("IQ3_XXS dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_iq3_s(ثابت خالی *vx, ثابت int64_t ib,
                                             ثابت صحیح iqs, dfloat2 &v) {
#اگر QK_K == 256
    ثابت بلاک_iq3_s * x = (ثابت بلاک_iq3_s *) vx;

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        ثابت صحیح ib8 = idx / 32;
        ثابت صحیح r = idx % 32;
        ثابت صحیح il = r / 8;
        ثابت صحیح j = r % 8;

        ثابت uint8_t * qs = x[ib].qs + 8 * ib8;
        ثابت uint16_t grid1_id = qs[2 * il + 0] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 256);
        ثابت uint16_t grid2_id = qs[2 * il + 1] | ((x[ib].qh[ib8] << (7 - 2 * il)) & 256);
        ثابت uint8_t * grid1 = (ثابت uint8_t *) (iq3s_grid + grid1_id);
        ثابت uint8_t * grid2 = (ثابت uint8_t *) (iq3s_grid + grid2_id);
        ثابت اعشاری d = (اعشاری) x[ib].d * (1 + 2 * ((x[ib].scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
        ثابت uint8_t signs = x[ib].signs[4 * ib8 + il];

        اگر (j < 4) {
            واپس d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        واپس d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("IQ3_S dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_iq1_s(ثابت خالی *vx, ثابت int64_t ib,
                                             ثابت صحیح iqs, dfloat2 &v) {
#اگر QK_K == 256
    ثابت بلاک_iq1_s * x = (ثابت بلاک_iq1_s *) vx;

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        ثابت صحیح ib8 = idx / 32;
        ثابت صحیح r = idx % 32;
        ثابت صحیح il = r / 8;
        ثابت صحیح j = r % 8;

        ثابت اعشاری delta = (x[ib].qh[ib8] & 0x8000) ? (-1.f - IQ1S_DELTA) : (-1.f + IQ1S_DELTA);
        ثابت اعشاری d = (اعشاری) x[ib].d * (2 * ((x[ib].qh[ib8] >> 12) & 7) + 1);
        ثابت uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((x[ib].qh[ib8] >> (3 * il)) & 7) << 8);
        ثابت uint32_t g = iq1s_grid_gpu[grid_id];
        ثابت int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        واپس d * (qv + delta);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("IQ1_S dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_iq1_m(ثابت خالی *vx, ثابت int64_t ib,
                                             ثابت صحیح iqs, dfloat2 &v) {
#اگر QK_K == 256
    ثابت بلاک_iq1_m * x = (ثابت بلاک_iq1_m *) vx;

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        ثابت صحیح ib8 = idx / 32;
        ثابت صحیح r = idx % 32;
        ثابت صحیح il = r / 8;
        ثابت صحیح j = r % 8;

        ثابت uint16_t * sc = (ثابت uint16_t *) x[ib].scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        ثابت صحیح ib16 = 2 * ib8 + il / 2;
        ثابت اعشاری d = (اعشاری) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);

        ثابت uint8_t qh = x[ib].qh[2 * ib8 + il / 2];
        ثابت اعشاری delta = (qh & (0x08 << (4 * (il % 2)))) ? (-1.f - IQ1M_DELTA) : (-1.f + IQ1M_DELTA);

        ثابت uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((qh >> (4 * (il % 2))) & 7) << 8);
        ثابت uint32_t g = iq1s_grid_gpu[grid_id];
        ثابت int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        واپس d * (qv + delta);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("IQ1_M dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_iq4_nl(ثابت خالی *vx, ثابت int64_t ib,
                                              ثابت صحیح iqs, dfloat2 &v) {
    ثابت بلاک_iq4_nl * x = (ثابت بلاک_iq4_nl *) vx;
    ثابت اعشاری d = (اعشاری) x[ib].d;

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        اگر (idx < 16) {
            واپس d * kvalues_iq4nl[x[ib].qs[idx] & 0xF];
        }
        واپس d * kvalues_iq4nl[x[ib].qs[idx - 16] >> 4];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
}

ساکن __dpct_inline__ خالی dequantize_iq4_xs(ثابت خالی *vx, ثابت int64_t ib,
                                              ثابت صحیح iqs, dfloat2 &v) {
#اگر QK_K == 256
    ثابت بلاک_iq4_xs * x = (ثابت بلاک_iq4_xs *) vx;

    خودکار dequantize_one = [&](ثابت صحیح idx) -> dfloat {
        ثابت صحیح ib8 = idx / 32;
        ثابت صحیح r = idx % 32;
        ثابت صحیح byte_idx = (r < 16) ? r : (r - 16);
        ثابت uint8_t q = x[ib].qs[16 * ib8 + byte_idx];
        ثابت uint8_t qv = (r < 16) ? (q & 0x0F) : (q >> 4);

        ثابت اعشاری d = (اعشاری) x[ib].d * ((((x[ib].scales_l[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf) |
                        (((x[ib].scales_h >> (2 * ib8)) & 3) << 4)) - 32);
        واپس d * kvalues_iq4nl[qv];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#ورنہ
    GGML_ABORT("IQ4_XS dequantize not supported for QK_K != 256");
#اگر_ختم
}

ساکن __dpct_inline__ خالی dequantize_q5_0(ثابت خالی *vx, ثابت int64_t ib,
                                            ثابت صحیح iqs, dfloat2 &v) {
    ثابت بلاک_q5_0 * x = (ثابت بلاک_q5_0 *) vx;

    ثابت dfloat d = x[ib].d;

    uint32_t qh;
    حافظہ_نقل(&qh, x[ib].qh, حجم(qh));

    ثابت صحیح xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    ثابت صحیح xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#اگر_تعریف_شدہ GGML_SYCL_F16
    // v = v - {16.0f, 16.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 16.0f) * d;
    v.s1() = (v.s1() - 16.0f) * d;

#ورنہ
    v.x() = (v.x() - 16.0f) * d;
    v.y() = (v.y() - 16.0f) * d;
#اگر_ختم // GGML_SYCL_F16
}

ساکن __dpct_inline__ خالی dequantize_q5_1(ثابت خالی *vx, ثابت int64_t ib,
                                            ثابت صحیح iqs, dfloat2 &v) {
    ثابت بلاک_q5_1 * x = (ثابت بلاک_q5_1 *) vx;

    ثابت dfloat d = x[ib].dm[0];
    ثابت dfloat m = x[ib].dm[1];

    uint32_t qh;
    حافظہ_نقل(&qh, x[ib].qh, حجم(qh));

    ثابت صحیح xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    ثابت صحیح xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#اگر_تعریف_شدہ GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = sycl::fma(v.s0(), d, m);
    v.s1() = sycl::fma(v.s1(), d, m);
#ورنہ
    v.x() = sycl::fma(v.x(), d, m);
    v.y() = sycl::fma(v.y(), d, m);
#اگر_ختم // GGML_SYCL_F16
}

ساکن __dpct_inline__ خالی dequantize_q8_0_reorder(ثابت خالی *d_ptr, ثابت int64_t ib, ثابت خالی *qs,
                                            ثابت صحیح iqs, dfloat2 &v) {
    ثابت dfloat d = (ثابت dfloat)*((ثابت sycl::half*)d_ptr + ib);

    v.x() = ((ثابت int8_t *)qs)[iqs + 0];
    v.y() = ((ثابت int8_t *)qs)[iqs + 1];

#اگر_تعریف_شدہ GGML_SYCL_F16
    v.s0() *= d;
    v.s1() *= d;
#ورنہ
    v.x() *= d;
    v.y() *= d;
#اگر_ختم // GGML_SYCL_F16
}

ساکن __dpct_inline__ خالی dequantize_q8_0(ثابت خالی *vx, ثابت int64_t ib,
                                            ثابت صحیح iqs, dfloat2 &v) {
    ثابت بلاک_q8_0 * x = (ثابت بلاک_q8_0 *) vx;

    ثابت dfloat d = x[ib].d;

    v.x() = x[ib].qs[iqs + 0];
    v.y() = x[ib].qs[iqs + 1];

#اگر_تعریف_شدہ GGML_SYCL_F16
    // v = v * {d, d};
    v.s0() *= d;
    v.s1() *= d;
#ورنہ
    v.x() *= d;
    v.y() *= d;
#اگر_ختم // GGML_SYCL_F16
}

سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q4_0(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت int64_t i = شے_ct1.get_گروہ(2);

    // assume 32 threads
    ثابت int64_t tid = شے_ct1.get_local_id(2);
    ثابت int64_t il  = tid/8;
    ثابت int64_t ir  = tid%8;
    ثابت int64_t ib = 8*i + ir;
    اگر (ib >= nb32) {
        واپس;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    ثابت بلاک_q4_0 * x = (ثابت بلاک_q4_0 *)vx + ib;
    ثابت اعشاری d = sycl::vec<sycl::half, 1>(x->d)
                        .convert<اعشاری, sycl::rounding_منوال::automatic>()[0];
    ثابت اعشاری dm = -8*d;

    ثابت uint8_t * q = x->qs + 4*il;

    برائے (صحیح l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q4_0_reorder(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                  ثابت sycl::nd_شے<3> &شے_ct1) {
    // one thread per qs byte, like dequantize_block<dequantize_q4_0>; the reorder layout makes the byte reads contiguous
    // grid-stride loop because the launcher may shrink the work-group to keep the global range below INT_MAX
    // float math, not dequantize_q4_0_reorder: that helper computes in half with GGML_SYCL_F16, which changes f32 output
    ثابت sycl::half * d_ptr  = (ثابت sycl::half *) ((ثابت uint8_t *) vx + k/2);
    ثابت int64_t      stride = شے_ct1.get_local_range(2) * شے_ct1.get_گروہ_range(2);

    برائے (int64_t iq = شے_ct1.get_global_id(2); iq < k/2; iq += stride) {
        ثابت int64_t ib  = iq / (QK4_0/2);
        ثابت int64_t iqs = iq % (QK4_0/2);
        ثابت صحیح     vq  = ((ثابت uint8_t *) vx)[iq];
        ثابت اعشاری   d   = اعشاری(d_ptr[ib]);

        dst_t * y = yy + ib*QK4_0 + iqs;
        y[0]       = d * ((vq & 0xF) - 8);
        y[QK4_0/2] = d * ((vq >>  4) - 8);
    }
}

// Dequantize Q8_0 from reorder layout: [all qs (k bytes)][all d values]
سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q8_0_reorder(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                  ثابت sycl::nd_شے<3> &شے_ct1) {
    // two values per thread, like dequantize_block<dequantize_q8_0>; the reorder layout keeps all qs contiguous
    // grid-stride loop because the launcher may shrink the work-group to keep the global range below INT_MAX
    // float math, not dequantize_q8_0_reorder: that helper computes in half with GGML_SYCL_F16, which changes f32 output
    ثابت int8_t *     qs     = (ثابت int8_t *) vx;
    ثابت sycl::half * d_ptr  = (ثابت sycl::half *) ((ثابت uint8_t *) vx + k);
    ثابت int64_t      stride = 2 * شے_ct1.get_local_range(2) * شے_ct1.get_گروہ_range(2);

    برائے (int64_t i = 2 * شے_ct1.get_global_id(2); i < k; i += stride) {
        ثابت اعشاری d = اعشاری(d_ptr[i / QK8_0]);

        yy[i + 0] = d * qs[i + 0];
        yy[i + 1] = d * qs[i + 1];
    }
}

سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q8_0_reorder_wide(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                               ثابت sycl::id<1> idx) {
    ثابت int64_t i = 4 * (int64_t) idx[0];

    ثابت int8_t *     qs    = (ثابت int8_t *) vx;
    ثابت sycl::half * d_ptr = (ثابت sycl::half *) ((ثابت uint8_t *) vx + k);

    ثابت اعشاری d0 = اعشاری(d_ptr[i / QK8_0]);
    ثابت اعشاری d1 = اعشاری(d_ptr[(i + 2) / QK8_0]);

    sycl::vec<اعشاری, 4> p;
    p[0] = d0 * (اعشاری) qs[i + 0];
    p[1] = d0 * (اعشاری) qs[i + 1];
    p[2] = d1 * (اعشاری) qs[i + 2];
    p[3] = d1 * (اعشاری) qs[i + 3];

    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yy + i) = p.convert<dst_t, sycl::rounding_منوال::automatic>();
}

سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q4_1(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت int64_t i = شے_ct1.get_گروہ(2);

    // assume 32 threads
    ثابت int64_t tid = شے_ct1.get_local_id(2);
    ثابت int64_t il  = tid/8;
    ثابت int64_t ir  = tid%8;
    ثابت int64_t ib = 8*i + ir;
    اگر (ib >= nb32) {
        واپس;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    ثابت بلاک_q4_1 * x = (ثابت بلاک_q4_1 *)vx + ib;
    ثابت sycl::float2 d =
        x->dm.convert<اعشاری, sycl::rounding_منوال::automatic>();

    ثابت uint8_t * q = x->qs + 4*il;

    برائے (صحیح l = 0; l < 4; ++l) {
        y[l + 0] = d.x() * (q[l] & 0xF) + d.y();
        y[l + 16] = d.x() * (q[l] >> 4) + d.y();
    }
}


//================================== k-quants

سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q2_K(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy,
                                  ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت int64_t i = شے_ct1.get_گروہ(2);
    ثابت بلاک_q2_K * x = (ثابت بلاک_q2_K *) vx;

    ثابت int64_t tid = شے_ct1.get_local_id(2);
#اگر QK_K == 256
    ثابت int64_t n   = tid/32;
    ثابت int64_t l   = tid - 32*n;
    ثابت int64_t is  = 8*n + l/16;

    ثابت uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    اعشاری dall = x[i].dm[0];
    اعشاری dmin = x[i].dm[1];
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#ورنہ
    ثابت int64_t is = tid/16;  // 0 or 1
    ثابت int64_t il = tid%16;  // 0...15
    ثابت uint8_t q = x[i].qs[il] >> (2*is);
    dst_t * y = yy + i*QK_K + 16*is + il;

    اعشاری dall = x[i].dm[0];
    اعشاری dmin = x[i].dm[1];
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#اگر_ختم

}

سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q2_K_reorder(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, ثابت int64_t i,
                                          ثابت sycl::nd_شے<3> & شے_ct1, int64_t n_بلاکیں) {
#اگر QK_K == 256
    اگر (i >= n_بلاکیں) {
        واپس;
    }

    ثابت uint8_t * بنیاد          = static_cast<ثابت uint8_t *>(vx);
    ثابت size_t    qs_انحراف     = i * (QK_K / 4);
    ثابت size_t    scales_انحراف = n_بلاکیں * (QK_K / 4) + i * (QK_K / 16);
    ثابت size_t    dm_انحراف     = n_بلاکیں * (QK_K / 4) + n_بلاکیں * (QK_K / 16) + i * حجم(ggml_half2);

    ثابت uint8_t *     qs     = بنیاد + qs_انحراف;
    ثابت uint8_t *     scales = بنیاد + scales_انحراف;
    ثابت ggml_half2 * dm     = reinterpret_cast<ثابت ggml_half2 *>(بنیاد + dm_انحراف);

    ثابت int64_t tid = شے_ct1.get_local_id(2);
    ثابت int64_t n    = tid / 32;
    ثابت int64_t l    = tid - 32 * n;
    ثابت int64_t is   = 8 * n + l / 16;

    ثابت uint8_t q = qs[32 * n + l];
    dst_t * y = yy + 128 * n;

    ثابت اعشاری dall = (*dm)[0];
    ثابت اعشاری dmin = (*dm)[1];
    y[l+ 0] = dall * (scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (scales[is+0] >> 4);
    y[l+32] = dall * (scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (scales[is+2] >> 4);
    y[l+64] = dall * (scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (scales[is+4] >> 4);
    y[l+96] = dall * (scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (scales[is+6] >> 4);
#ورنہ
    GGML_غیر_مستعمل(vx);
    GGML_غیر_مستعمل(yy);
    GGML_غیر_مستعمل(i);
    GGML_غیر_مستعمل(شے_ct1);
    GGML_غیر_مستعمل(n_بلاکیں);
    GGML_ABORT("Q2_K reorder dequantize not supported for QK_K != 256");
#اگر_ختم
}

سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q3_K(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy,
                                  ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت int64_t i = شے_ct1.get_گروہ(2);
    ثابت بلاک_q3_K * x = (ثابت بلاک_q3_K *) vx;

#اگر QK_K == 256
    ثابت int64_t r = شے_ct1.get_local_id(2) / 4;
    ثابت int64_t tid = r/2;
    ثابت int64_t is0 = r%2;
    ثابت int64_t l0 = 16 * is0 + 4 * (شے_ct1.get_local_id(2) % 4);
    ثابت int64_t n = tid / 4;
    ثابت int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    صحیح سرکاؤ = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    اعشاری d_all = x[i].d;
    اعشاری dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    ثابت uint8_t * q = x[i].qs + 32*n;
    ثابت uint8_t * hm = x[i].hmask;

    برائے (صحیح l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> سرکاؤ) & 3) - ((hm[l] & m) ? 0 : 4));
#ورنہ
    ثابت int64_t tid = شے_ct1.get_local_id(2);
    ثابت int64_t is  = tid/16;  // 0 or 1
    ثابت int64_t il  = tid%16;  // 0...15
    ثابت int64_t im  = il/8;    // 0...1
    ثابت int64_t اندر  = il%8;    // 0...7

    dst_t * y = yy + i*QK_K + 16*is + il;

    ثابت uint8_t q = x[i].qs[il] >> (2*is);
    ثابت uint8_t h = x[i].hmask[اندر] >> (2*is + im);
    ثابت اعشاری   d = (اعشاری)x[i].d;

    اگر (is == 0) {
        y[ 0] = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    } ورنہ {
        y[ 0] = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    }
#اگر_ختم

}

سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q3_K_reorder(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, ثابت int64_t i,
                                          ثابت sycl::nd_شے<3> & شے_ct1, int64_t n_بلاکیں) {
#اگر QK_K == 256
    اگر (i >= n_بلاکیں) {
        واپس;
    }

    ثابت uint8_t * بنیاد          = static_cast<ثابت uint8_t *>(vx);
    ثابت size_t    qs_انحراف     = i * (QK_K / 4);
    ثابت size_t    hmask_انحراف  = n_بلاکیں * (QK_K / 4) + i * (QK_K / 8);
    ثابت size_t    scales_انحراف = n_بلاکیں * (QK_K / 4) + n_بلاکیں * (QK_K / 8) + i * 12;
    ثابت size_t    d_انحراف      = n_بلاکیں * (QK_K / 4) + n_بلاکیں * (QK_K / 8) + n_بلاکیں * 12 +
                                 i * حجم(ggml_half);

    ثابت uint8_t * qs     = بنیاد + qs_انحراف;
    ثابت uint8_t * hmask  = بنیاد + hmask_انحراف;
    ثابت uint8_t * scales = بنیاد + scales_انحراف;
    ثابت اعشاری     d_all  = static_cast<اعشاری>(*reinterpret_cast<ثابت ggml_half *>(بنیاد + d_انحراف));

    ثابت int64_t r    = شے_ct1.get_local_id(2) / 4;
    ثابت int64_t tid  = r / 2;
    ثابت int64_t is0  = r % 2;
    ثابت int64_t l0   = 16 * is0 + 4 * (شے_ct1.get_local_id(2) % 4);
    ثابت int64_t n    = tid / 4;
    ثابت int64_t j    = tid - 4 * n;
    ثابت int64_t is   = 8 * n + 2 * j + is0;
    ثابت صحیح     سرکاؤ = 2 * j;
    uint8_t       m    = 1 << (4 * n + j);

    uint8_t us = is < 4
        ? (scales[is - 0] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4)
        : is < 8
            ? (scales[is - 0] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4)
            : is < 12
                ? (scales[is - 8] >> 4) | (((scales[is + 0] >> 4) & 3) << 4)
                : (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);

    ثابت اعشاری dl = d_all * (us - 32);

    dst_t * y = yy + 128 * n + 32 * j;
    ثابت uint8_t * q  = qs + 32 * n;
    ثابت uint8_t * hm = hmask;

    برائے (صحیح l = l0; l < l0 + 4; ++l) {
        y[l] = dl * ((int8_t) ((q[l] >> سرکاؤ) & 3) - ((hm[l] & m) ? 0 : 4));
    }
#ورنہ
    GGML_غیر_مستعمل(vx);
    GGML_غیر_مستعمل(yy);
    GGML_غیر_مستعمل(i);
    GGML_غیر_مستعمل(شے_ct1);
    GGML_غیر_مستعمل(n_بلاکیں);
    GGML_ABORT("Q3_K reorder dequantize not supported for QK_K != 256");
#اگر_ختم
}

#اگر QK_K == 256
ساکن اندرونی خالی get_scale_min_k4(صحیح j, ثابت uint8_t * q, uint8_t & d, uint8_t & m) {
    اگر (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } ورنہ {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#اگر_ختم

// Work-items that cover one 256-weight K-quant block: four 64-element chunks, 8 work-items
// per chunk, 4 qs bytes per work-item.
ساکن constexpr صحیح SYCL_DEQK_WI_فی_بلاک = 32;

سانچہ <قسم_نام dst_t>
ساکن اندرونی خالی dequantize_q4_K_wide_one(dst_t * __restrict__ y, ثابت uint8_t * __restrict__ qs,
                                            ثابت uint8_t * __restrict__ scales, ثابت اعشاری dall,
                                            ثابت اعشاری dmin, ثابت صحیح tid) {
    ثابت صحیح il = tid >> 3;        // 64-element chunk, 0..3
    ثابت صحیح ir = (tid & 7) * 4;   // 4 qs bytes within the chunk

    uint8_t sc, m;
    get_scale_min_k4(2 * il + 0, scales, sc, m);
    ثابت اعشاری d1 = dall * sc;
    ثابت اعشاری m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, scales, sc, m);
    ثابت اعشاری d2 = dall * sc;
    ثابت اعشاری m2 = dmin * m;

    ثابت sycl::vec<uint8_t, 4> q = vec_aligned_بھرو<uint8_t, 4>(qs + 32 * il + ir);

    sycl::vec<dst_t, 4> lo, hi;
#ہدایت unroll
    برائے (صحیح l = 0; l < 4; ++l) {
        lo[l] = d1 * (q[l] & 0xF) - m1;
        hi[l] = d2 * (q[l] >> 4)  - m2;
    }
    dst_t * yb = y + 64 * il + ir;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb +  0) = lo;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb + 32) = hi;
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q4_K_wide(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb,
                                       ثابت sycl::nd_شے<1> & it) {
    static_assert(QK_K == 256, "wide K-quant dequant assumes QK_K == 256");
    ثابت int64_t g   = it.get_global_id(0);
    ثابت int64_t b   = g / SYCL_DEQK_WI_فی_بلاک;
    ثابت صحیح     tid = (صحیح) (g % SYCL_DEQK_WI_فی_بلاک);
    اگر (b >= nb) {
        واپس;
    }
    ثابت بلاک_q4_K * x = (ثابت بلاک_q4_K *) vx + b;
    dequantize_q4_K_wide_one(yy + b * QK_K, x->qs, x->scales, (اعشاری) x->dm[0], (اعشاری) x->dm[1], tid);
}


سانچہ <قسم_نام dst_t>
اندرونی خالی dequantize_q4_K_common(dst_t * __restrict__ y, ثابت uint8_t * __restrict__ qs_ptr, ثابت اعشاری dall,
                                   ثابت اعشاری dmin, uint8_t * __restrict__ scales_local, صحیح il, صحیح ir) {
    ثابت صحیح is = 2 * il;
    constexpr صحیح n  = 4;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    ثابت اعشاری d1 = dall * sc;
    ثابت اعشاری m1 = dmin * m;

    get_scale_min_k4(is + 1, scales_local, sc, m);
    ثابت اعشاری d2 = dall * sc;
    ثابت اعشاری m2 = dmin * m;

    sycl::vec<uint8_t, n> q_vec = vec_aligned_بھرو<uint8_t, n>(qs_ptr + 32 * il + n * ir);
    برائے (صحیح l = 0; l < n; ++l) {
        y[l + 0]  = d1 * (q_vec[l] & 0xF) - m1;
        y[l + 32] = d2 * (q_vec[l] >> 4) - m2;
    }
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q4_K_reorder(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, ثابت int64_t i,
                                          uint8_t * scales_local, ثابت sycl::nd_شے<1> & شے_ct1, int64_t nb) {
    ثابت int64_t tid = شے_ct1.get_local_id(0);  // thread index within block
    ثابت int64_t il  = tid / 8;
    ثابت int64_t ir  = tid % 8;

    dst_t * y = yy + 64 * il + 4 * ir;

    ثابت uint8_t * بنیاد          = static_cast<ثابت uint8_t *>(vx);
    ثابت size_t    qs_انحراف     = i * (QK_K / 2);
    ثابت size_t    scales_انحراف = nb * (QK_K / 2) + i * K_SCALE_سائز;
    ثابت size_t    dm_انحراف     = nb * (QK_K / 2) + nb * K_SCALE_سائز + i * حجم(ggml_half2);

    ثابت uint8_t *    qs_ptr     = بنیاد + qs_انحراف;
    ثابت uint8_t *    scales_ptr = بنیاد + scales_انحراف;
    ggml_half2         dm_اقدار  = *reinterpret_cast<ثابت ggml_half2 *>(بنیاد + dm_انحراف);

    ثابت اعشاری dall = dm_اقدار.x();
    ثابت اعشاری dmin = dm_اقدار.y();

    اگر (tid < 12) {
        scales_local[tid] = scales_ptr[tid];
    }

    شے_ct1.barrier(sycl::access::fence_space::local_space);
    dequantize_q4_K_common(y, qs_ptr, dall, dmin, scales_local, il, ir);
}

سانچہ <قسم_نام dst_t>
ساکن اندرونی خالی dequantize_q5_K_wide_one(dst_t * __restrict__ y, ثابت uint8_t * __restrict__ qs,
                                            ثابت uint8_t * __restrict__ qh_ptr,
                                            ثابت uint8_t * __restrict__ scales, ثابت اعشاری dall,
                                            ثابت اعشاری dmin, ثابت صحیح tid) {
    ثابت صحیح il = tid >> 3;        // 64-element chunk, 0..3
    ثابت صحیح ir = (tid & 7) * 4;   // 4 qs bytes within the chunk; qh holds the 5th bit

    uint8_t sc, m;
    get_scale_min_k4(2 * il + 0, scales, sc, m);
    ثابت اعشاری d1 = dall * sc;
    ثابت اعشاری m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, scales, sc, m);
    ثابت اعشاری d2 = dall * sc;
    ثابت اعشاری m2 = dmin * m;

    ثابت sycl::vec<uint8_t, 4> q = vec_aligned_بھرو<uint8_t, 4>(qs + 32 * il + ir);
    ثابت sycl::vec<uint8_t, 4> h = vec_aligned_بھرو<uint8_t, 4>(qh_ptr + ir);

    ثابت uint8_t hm1 = 1 << (2 * il);
    ثابت uint8_t hm2 = hm1 << 1;

    sycl::vec<dst_t, 4> lo, hi;
#ہدایت unroll
    برائے (صحیح l = 0; l < 4; ++l) {
        lo[l] = d1 * ((q[l] & 0xF) + ((h[l] & hm1) ? 16 : 0)) - m1;
        hi[l] = d2 * ((q[l] >>  4) + ((h[l] & hm2) ? 16 : 0)) - m2;
    }
    dst_t * yb = y + 64 * il + ir;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb +  0) = lo;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb + 32) = hi;
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q5_K_wide(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb,
                                       ثابت sycl::nd_شے<1> & it) {
    static_assert(QK_K == 256, "wide K-quant dequant assumes QK_K == 256");
    ثابت int64_t g   = it.get_global_id(0);
    ثابت int64_t b   = g / SYCL_DEQK_WI_فی_بلاک;
    ثابت صحیح     tid = (صحیح) (g % SYCL_DEQK_WI_فی_بلاک);
    اگر (b >= nb) {
        واپس;
    }
    ثابت بلاک_q5_K * x = (ثابت بلاک_q5_K *) vx + b;
    dequantize_q5_K_wide_one(yy + b * QK_K, x->qs, x->qh, x->scales, (اعشاری) x->dm[0], (اعشاری) x->dm[1], tid);
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q5_K_reorder(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, ثابت int64_t ib,
                                          uint8_t * scales_local, ثابت sycl::nd_شے<3> & شے_ct1, int64_t n_بلاکیں) {

#اگر QK_K == 256
    // assume 64 threads
    ثابت int64_t tid = شے_ct1.get_local_id(2);
    ثابت int64_t il  = tid / 16;   // 0...3
    ثابت int64_t ir  = tid % 16;   // 0...15
    ثابت int64_t is  = 2 * il;

    dst_t * y = yy + 64 * il + 2 * ir;

    ثابت uint8_t * بنیاد = static_cast<ثابت uint8_t *>(vx);

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales (K_SCALE_SIZE per block)] [dm (half2 per block)]
    ثابت size_t qs_انحراف     = ib * (QK_K / 2);
    ثابت size_t qh_انحراف     = n_بلاکیں * (QK_K / 2) + ib * (QK_K / 8);
    ثابت size_t scales_انحراف = n_بلاکیں * (QK_K / 2) + n_بلاکیں * (QK_K / 8) + ib * K_SCALE_سائز;
    ثابت size_t dm_انحراف     = n_بلاکیں * (QK_K / 2) + n_بلاکیں * (QK_K / 8) + n_بلاکیں * K_SCALE_سائز + ib * حجم(ggml_half2);

    ثابت uint8_t *  qs_ptr     = بنیاد + qs_انحراف;
    ثابت uint8_t *  qh_ptr     = بنیاد + qh_انحراف;
    ثابت uint8_t *  scales_ptr = بنیاد + scales_انحراف;
    ثابت ggml_half2 dm_اقدار  = *reinterpret_cast<ثابت ggml_half2 *>(بنیاد + dm_انحراف);

    ثابت اعشاری dall = dm_اقدار.x();
    ثابت اعشاری dmin = dm_اقدار.y();

    ثابت uint8_t * ql = qs_ptr + 32 * il + 2 * ir;
    ثابت uint8_t * qh = qh_ptr + 2 * ir;

    اگر (tid < K_SCALE_سائز) {
        scales_local[tid] = scales_ptr[tid];
    }

    شے_ct1.barrier(sycl::access::fence_space::local_space);

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    ثابت اعشاری d1 = dall * sc; ثابت اعشاری m1 = dmin * m;
    get_scale_min_k4(is + 1, scales_local, sc, m);
    ثابت اعشاری d2 = dall * sc; ثابت اعشاری m2 = dmin * m;

    uint8_t hm  = 1 << (2 * il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
#ورنہ
    GGML_غیر_مستعمل(ib); GGML_غیر_مستعمل(tid); GGML_غیر_مستعمل(yy); GGML_غیر_مستعمل(scales_local); GGML_غیر_مستعمل(n_بلاکیں);
    GGML_ABORT("Q5_K reorder dequantize not supported for QK_K != 256");
#اگر_ختم
}

سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q6_K(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy,
                                  ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت بلاک_q6_K * x = (ثابت بلاک_q6_K *) vx;

    ثابت int64_t i = شے_ct1.get_گروہ(2);
#اگر QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    ثابت int64_t tid = شے_ct1.get_local_id(2);
    ثابت int64_t ip  = tid/32;   // ip is 0 or 1
    ثابت int64_t il  = tid - 32*ip; // 0...32
    ثابت int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    ثابت اعشاری d = x[i].d;

    ثابت uint8_t * ql = x[i].ql + 64*ip + il;
    ثابت uint8_t   qh = x[i].qh[32*ip + il];
    ثابت int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#ورنہ

    // assume 32 threads
    ثابت int64_t tid = شے_ct1.get_local_id(2);
    ثابت int64_t ip  = tid/16;         // 0 or 1
    ثابت int64_t il  = tid - 16*ip;    // 0...15

    dst_t * y = yy + i*QK_K + 16*ip + il;

    ثابت اعشاری d = x[i].d;

    ثابت uint8_t   ql = x[i].ql[16*ip + il];
    ثابت uint8_t   qh = x[i].qh[il] >> (2*ip);
    ثابت int8_t  * sc = x[i].scales;

    y[ 0] = d * sc[ip+0] * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[ip+2] * ((int8_t)((ql  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
#اگر_ختم
}

سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_بلاک_q6_K_reorder(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, ثابت int64_t ib,
                                          ثابت sycl::nd_شے<3> & شے_ct1, int64_t n_بلاکیں) {

    ثابت int64_t tid = شے_ct1.get_local_id(2);
    ثابت int64_t ip  = tid / 32;       // ip is 0 or 1
    ثابت int64_t il  = tid - 32 * ip;  // 0...32
    ثابت int64_t is  = 8 * ip + il / 16;

    ثابت uint8_t *   بنیاد_ptr           = static_cast<ثابت uint8_t *>(vx);
    ثابت خودکار        ql_انحراف          = ib * (QK_K / 2);
    ثابت خودکار        qh_انحراف          = (QK_K / 2) * n_بلاکیں + (QK_K / 4) * ib;
    ثابت خودکار        بنیاد_scales_انحراف = (QK_K / 2) * n_بلاکیں + (QK_K / 4) * n_بلاکیں + (QK_K / 16) * ib;
    ثابت خودکار        بنیاد_d_انحراف      = ((QK_K / 2) + (QK_K / 4) + (QK_K / 16)) * n_بلاکیں;
    ثابت uint8_t *   ql_ptr             = بنیاد_ptr + ql_انحراف;
    ثابت uint8_t *   qh_ptr             = بنیاد_ptr + qh_انحراف;
    ثابت uint8_t *   scales_ptr         = بنیاد_ptr + بنیاد_scales_انحراف;
    ثابت ggml_half * d                  = (ثابت ggml_half *) (بنیاد_ptr + بنیاد_d_انحراف) + ib;

    dst_t * y = yy + 128 * ip + il;

    ثابت uint8_t * ql = ql_ptr + 64 * ip + il;
    ثابت uint8_t   qh = *(qh_ptr + 32 * ip + il);
    ثابت int8_t *  sc = reinterpret_cast<ثابت int8_t *>(scales_ptr + is);

    y[0]  = *d * sc[0] * ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = *d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = *d * sc[4] * ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = *d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_iq2_xxs(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy,
                                     ثابت sycl::nd_شے<3> &شے_ct1,
                                     ثابت uint64_t *iq2xxs_grid_ptr,
                                     ثابت uint8_t *ksigns_iq2xs_ptr,
                                     ثابت uint8_t *kmask_iq2xs_ptr) {

    ثابت int64_t i = شے_ct1.get_گروہ(2);
    ثابت بلاک_iq2_xxs * x = (ثابت بلاک_iq2_xxs  *) vx;

    ثابت int64_t tid = شے_ct1.get_local_id(2);
#اگر QK_K == 256
    ثابت int64_t il = tid/8; // 0...3
    ثابت int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ثابت uint16_t * q2 = x[i].qs + 4*ib;
    ثابت uint8_t  * aux8 = (ثابت uint8_t *)q2;
    ثابت uint8_t  * grid = (ثابت uint8_t *)(iq2xxs_grid_ptr + aux8[il]);
    ثابت uint32_t aux32 = q2[2] | (q2[3] << 16);
    ثابت اعشاری d = (اعشاری)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    ثابت uint8_t signs = ksigns_iq2xs_ptr[(aux32 >> 7*il) & 127];
    برائے (صحیح j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs_ptr[j] ? -1.f : 1.f);
#ورنہ
    أكد(جھوٹ);
#اگر_ختم

}

سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_iq2_xs(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy,
                                    ثابت sycl::nd_شے<3> &شے_ct1,
                                    ثابت uint64_t *iq2xs_grid,
                                    ثابت uint8_t *ksigns_iq2xs,
                                    ثابت uint8_t *kmask_iq2xs) {

    ثابت int64_t i = شے_ct1.get_گروہ(2);
    ثابت بلاک_iq2_xs * x = (ثابت بلاک_iq2_xs *) vx;

    ثابت int64_t tid = شے_ct1.get_local_id(2);
#اگر QK_K == 256
    ثابت int64_t il = tid/8; // 0...3
    ثابت int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ثابت uint16_t * q2 = x[i].qs + 4*ib;
    ثابت uint8_t  * grid = (ثابت uint8_t *)(iq2xs_grid + (q2[il] & 511));
    ثابت اعشاری d = (اعشاری)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    ثابت uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    برائے (صحیح j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#ورنہ
    أكد(جھوٹ);
#اگر_ختم

}

سانچہ <قسم_نام dst_t>
__dpct_inline__ ساکن خالی
dequantize_بلاک_iq2_s(ثابت خالی *__restrict__ vx, dst_t *__restrict__ yy,
                       ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت int64_t i = شے_ct1.get_گروہ(2);
    ثابت بلاک_iq2_s * x = (ثابت بلاک_iq2_s *) vx;

    ثابت int64_t tid = شے_ct1.get_local_id(2);
#اگر QK_K == 256
    ثابت int64_t il = tid/8; // 0...3
    ثابت int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ثابت uint8_t * grid = (ثابت uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    ثابت اعشاری d = (اعشاری)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    ثابت uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
#ہدایت unroll
    برائے (صحیح j = 0; j < 8; ++j)
        y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#ورنہ
    أكد(جھوٹ);

#اگر_ختم

}

سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_iq3_xxs(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy,
                                     ثابت sycl::nd_شے<3> &شے_ct1,
                                     ثابت uint32_t *iq3xxs_grid,
                                     ثابت uint8_t *ksigns_iq2xs,
                                     ثابت uint8_t *kmask_iq2xs) {

    ثابت int64_t i = شے_ct1.get_گروہ(2);
    ثابت بلاک_iq3_xxs * x = (ثابت بلاک_iq3_xxs  *) vx;

    ثابت int64_t tid = شے_ct1.get_local_id(2);
#اگر QK_K == 256
    ثابت int64_t il = tid/8; // 0...3
    ثابت int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ثابت uint8_t  * q3 = x[i].qs + 8*ib;
    ثابت uint16_t * gas = (ثابت uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    ثابت uint8_t  * grid1 = (ثابت uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    ثابت uint8_t  * grid2 = (ثابت uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    ثابت uint32_t aux32 = gas[0] | (gas[1] << 16);
    ثابت اعشاری d = (اعشاری)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    ثابت uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    برائے (صحیح j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ورنہ
    أكد(جھوٹ);
#اگر_ختم

}

سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_iq3_xxs_reorder(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, ثابت int64_t i,
                                             ثابت sycl::nd_شے<3> & شے_ct1, ثابت int64_t nb,
                                             ثابت uint32_t * iq3xxs_grid,
                                             ثابت uint8_t * ksigns_iq2xs,
                                             ثابت uint8_t * kmask_iq2xs) {
    ثابت int64_t tid = شے_ct1.get_local_id(2);
#اگر QK_K == 256
    ثابت int64_t il = tid / 8;
    ثابت int64_t ib = tid % 8;
    ثابت uint8_t * بنیاد = static_cast<ثابت uint8_t *>(vx);
    ثابت uint8_t * qs = بنیاد + i * (3 * QK_K / 8);
    ثابت ggml_half * d = reinterpret_cast<ثابت ggml_half *>(بنیاد + nb * (3 * QK_K / 8));
    ثابت uint8_t * q3 = qs + 8 * ib;
    ثابت uint16_t * gas = reinterpret_cast<ثابت uint16_t *>(qs + QK_K / 4) + 2 * ib;
    ثابت uint8_t * grid1 = reinterpret_cast<ثابت uint8_t *>(iq3xxs_grid + q3[2*il+0]);
    ثابت uint8_t * grid2 = reinterpret_cast<ثابت uint8_t *>(iq3xxs_grid + q3[2*il+1]);
    ثابت uint32_t aux32 = gas[0] | (gas[1] << 16);
    ثابت اعشاری dl = (اعشاری) d[i] * (0.5f + (aux32 >> 28)) * 0.5f;
    ثابت uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
    dst_t * y = yy + 32 * ib + 8 * il;
#ہدایت unroll
    برائے (صحیح j = 0; j < 4; ++j) {
        y[j+0] = dl * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = dl * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ورنہ
    GGML_غیر_مستعمل(vx);
    GGML_غیر_مستعمل(yy);
    GGML_غیر_مستعمل(i);
    GGML_غیر_مستعمل(شے_ct1);
    GGML_غیر_مستعمل(nb);
    GGML_غیر_مستعمل(iq3xxs_grid);
    GGML_غیر_مستعمل(ksigns_iq2xs);
    GGML_غیر_مستعمل(kmask_iq2xs);
    GGML_ABORT("IQ3_XXS reorder dequantization requires QK_K == 256");
#اگر_ختم
}

سانچہ <قسم_نام dst_t>
__dpct_inline__ ساکن خالی
dequantize_بلاک_iq3_s(ثابت خالی *__restrict__ vx, dst_t *__restrict__ yy,
                       ثابت sycl::nd_شے<3> &شے_ct1,
                       ثابت uint8_t *kmask_iq2xs, ثابت uint32_t *iq3s_grid) {

    ثابت int64_t i = شے_ct1.get_گروہ(2);
    ثابت بلاک_iq3_s * x = (ثابت بلاک_iq3_s *) vx;

    ثابت int64_t tid = شے_ct1.get_local_id(2);
#اگر QK_K == 256
    ثابت int64_t il = tid/8; // 0...3
    ثابت int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ثابت uint8_t * qs = x[i].qs + 8*ib;
    ثابت uint8_t * grid1 = (ثابت uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    ثابت uint8_t * grid2 = (ثابت uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    ثابت اعشاری d = (اعشاری)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    ثابت uint8_t signs = x[i].signs[4*ib + il];
#ہدایت unroll
    برائے (صحیح j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ورنہ
    أكد(جھوٹ);
#اگر_ختم

}

سانچہ <قسم_نام dst_t>
__dpct_inline__ ساکن خالی
dequantize_بلاک_iq3_s_reorder(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, ثابت int64_t i,
                               ثابت sycl::nd_شے<3> & شے_ct1, ثابت int64_t nb,
                               ثابت uint8_t * kmask_iq2xs, ثابت uint32_t * iq3s_grid) {
    ثابت int64_t tid = شے_ct1.get_local_id(2);
#اگر QK_K == 256
    ثابت int64_t il = tid / 8;
    ثابت int64_t ib = tid % 8;
    ثابت uint8_t * بنیاد = static_cast<ثابت uint8_t *>(vx);
    ثابت uint8_t * qs = بنیاد + i * (QK_K / 4);
    ثابت uint8_t * qh = بنیاد + nb * (QK_K / 4) + i * (QK_K / 32);
    ثابت uint8_t * signs_بنیاد = بنیاد + nb * (QK_K / 4 + QK_K / 32) + i * (QK_K / 8);
    ثابت uint8_t * metadata = بنیاد + nb * (QK_K / 4 + QK_K / 32 + QK_K / 8) +
                               i * (حجم(ggml_half) + IQ3S_N_SCALE);
    ثابت ggml_half d = *reinterpret_cast<ثابت ggml_half *>(metadata);
    ثابت uint8_t * scales = metadata + حجم(ggml_half);
    ثابت uint8_t * qs_ib = qs + 8 * ib;
    ثابت uint8_t * grid1 = reinterpret_cast<ثابت uint8_t *>(
        iq3s_grid + (qs_ib[2*il+0] | ((qh[ib] << (8 - 2*il)) & 256)));
    ثابت uint8_t * grid2 = reinterpret_cast<ثابت uint8_t *>(
        iq3s_grid + (qs_ib[2*il+1] | ((qh[ib] << (7 - 2*il)) & 256)));
    ثابت اعشاری dl = (اعشاری) d * (1 + 2 * ((scales[ib/2] >> (4 * (ib%2))) & 0xf));
    ثابت uint8_t signs = signs_بنیاد[4 * ib + il];
    dst_t * y = yy + 32 * ib + 8 * il;
#ہدایت unroll
    برائے (صحیح j = 0; j < 4; ++j) {
        y[j+0] = dl * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = dl * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#ورنہ
    GGML_غیر_مستعمل(vx);
    GGML_غیر_مستعمل(yy);
    GGML_غیر_مستعمل(i);
    GGML_غیر_مستعمل(شے_ct1);
    GGML_غیر_مستعمل(nb);
    GGML_غیر_مستعمل(kmask_iq2xs);
    GGML_غیر_مستعمل(iq3s_grid);
    GGML_ABORT("IQ3_S reorder dequantization requires QK_K == 256");
#اگر_ختم
}

سانچہ <قسم_نام dst_t>
__dpct_inline__ ساکن خالی
dequantize_بلاک_iq1_s(ثابت خالی *__restrict__ vx, dst_t *__restrict__ yy,
                       ثابت sycl::nd_شے<3> &شے_ct1,
                       ثابت uint32_t *iq1s_grid_gpu) {

    ثابت int64_t i = شے_ct1.get_گروہ(2);
    ثابت بلاک_iq1_s * x = (ثابت بلاک_iq1_s  *) vx;

    ثابت int64_t tid = شے_ct1.get_local_id(2);
#اگر QK_K == 256
    ثابت int64_t il = tid/8; // 0...3
    ثابت int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ثابت اعشاری delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    ثابت اعشاری d = (اعشاری)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; ثابت int8_t * q = (ثابت int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#ہدایت unroll
    برائے (صحیح j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#ورنہ
    أكد(جھوٹ);
#اگر_ختم

}

سانچہ <قسم_نام dst_t>
__dpct_inline__ ساکن خالی
dequantize_بلاک_iq1_m(ثابت خالی *__restrict__ vx, dst_t *__restrict__ yy,
                       ثابت sycl::nd_شے<3> &شے_ct1,
                       ثابت uint32_t *iq1s_grid_gpu) {

    ثابت int64_t i = شے_ct1.get_گروہ(2);
    ثابت بلاک_iq1_m * x = (ثابت بلاک_iq1_m  *) vx;

    ثابت int64_t tid = شے_ct1.get_local_id(2);
#اگر QK_K == 256
    ثابت int64_t il = tid/8; // 0...3
    ثابت int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    ثابت uint16_t * sc = (ثابت uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    ثابت صحیح ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    ثابت اعشاری d = (اعشاری)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    ثابت اعشاری delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; ثابت int8_t * q = (ثابت int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#ہدایت unroll
    برائے (صحیح j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#ورنہ
    أكد(جھوٹ);
#اگر_ختم

}

سانچہ <قسم_نام dst_t>
__dpct_inline__ ساکن خالی
dequantize_بلاک_iq4_nl(ثابت خالی *__restrict__ vx, dst_t *__restrict__ yy,
                        ثابت sycl::nd_شے<3> &شے_ct1) {

    ثابت int64_t i = شے_ct1.get_گروہ(2);
    ثابت بلاک_iq4_nl * x = (ثابت بلاک_iq4_nl *) vx + i*(QK_K/QK4_NL);

    ثابت int64_t tid = شے_ct1.get_local_id(2);
    ثابت int64_t il = tid/8; // 0...3
    ثابت int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    ثابت uint8_t  * q4 = x[ib].qs + 4*il;
    ثابت اعشاری d = (اعشاری)x[ib].d;
#ہدایت unroll
    برائے (صحیح j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }

}


سانچہ <قسم_نام dst_t>
__dpct_inline__ ساکن خالی
dequantize_بلاک_iq4_xs(ثابت خالی *__restrict__ vx, dst_t *__restrict__ yy,
                        ثابت sycl::nd_شے<3> &شے_ct1) {
    ثابت int64_t i = شے_ct1.get_گروہ(2);
    ثابت بلاک_iq4_xs * x = (ثابت بلاک_iq4_xs *)vx;

    ثابت int64_t tid = شے_ct1.get_local_id(2);
    ثابت int64_t il = tid/8; // 0...3
    ثابت int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    ثابت uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    ثابت اعشاری d = (اعشاری)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
#ہدایت unroll
    برائے (صحیح j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

سانچہ<قسم_نام dst_t>
ساکن خالی dequantize_بلاک_mxfp4(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy,
                                   ثابت sycl::nd_شے<3> &شے_ct1) {
    // auto                item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    ثابت int64_t       i        = شے_ct1.get_گروہ(2);
    ثابت بلاک_mxfp4 * x = (ثابت بلاک_mxfp4 *) vx + i*(QK_K/QK_MXFP4);

    ثابت int64_t    tid = شے_ct1.get_local_id(2);
    ثابت int64_t il = tid/8; // 0...3
    ثابت int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    ثابت uint8_t  * q4 = x[ib].qs + 4*il;
    ثابت اعشاری d = ggml_sycl_e8m0_مقصد_fp32(x[ib].e);
    برائے (صحیح j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_mxfp4[q4[j] & 0xf]*0.5f;
        y[j+16] = d * kvalues_mxfp4[q4[j] >>  4]*0.5f;
    }
}

// Reordered MXFP4 ([qs...][e...], see ggml_sycl_reordered::block_q_t<MXFP4>): one work-item per block.
سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_بلاک_mxfp4_reorder(ثابت خالی * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                           ثابت sycl::nd_شے<3> & شے_ct1) {
    ثابت int64_t ib = (int64_t) شے_ct1.get_گروہ(2) * WARP_سائز + شے_ct1.get_local_id(2);
    اگر (ib >= k / QK_MXFP4) {
        واپس;
    }

    ثابت uint8_t * qs = (ثابت uint8_t *) vx + ib * (QK_MXFP4 / 2);
    ثابت اعشاری     d  = ggml_sycl_e8m0_مقصد_fp32(((ثابت uint8_t *) vx)[k / 2 + ib]) * 0.5f;
    dst_t *         y  = yy + ib * QK_MXFP4;

#ہدایت unroll
    برائے (صحیح j = 0; j < QK_MXFP4 / 2; ++j) {
        y[j]                = d * kvalues_mxfp4[qs[j] & 0xf];
        y[j + QK_MXFP4 / 2] = d * kvalues_mxfp4[qs[j] >> 4];
    }
}


سانچہ <قسم_نام dst_t>
ساکن خالی dequantize_بلاک_nvfp4(
        ثابت خالی * __restrict__ vx,
        dst_t * __restrict__ yy,
        ثابت int64_t ne) {
    خودکار          شے_ct1 = sycl::ext::oneapi::this_work_شے::get_nd_شے<3>();
    ثابت int64_t i        = شے_ct1.get_گروہ(2);
    ثابت صحیح     tid      = شے_ct1.get_local_id(2);

    ثابت int64_t بنیاد = i * QK_NVFP4;
    اگر (بنیاد >= ne) {
        واپس;
    }

    ثابت بلاک_nvfp4 * x = (ثابت بلاک_nvfp4 *) vx;
    ثابت بلاک_nvfp4 & xb = x[i];

    ثابت صحیح sub = tid / (QK_NVFP4_SUB / 2);
    ثابت صحیح j = tid % (QK_NVFP4_SUB / 2);

    ثابت اعشاری d = ggml_sycl_ue4m3_مقصد_fp32(xb.d[sub]);
    ثابت uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    ثابت int64_t y0 = بنیاد + sub * QK_NVFP4_SUB + j;
    ثابت int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_sycl_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_sycl_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}


#اگر_ختم // GGML_SYCL_DEQUANTIZE_HPP
