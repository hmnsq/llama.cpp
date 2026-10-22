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

#ifndef GGML_SYCL_DEQUANTIZE_HPP
#define GGML_SYCL_DEQUANTIZE_HPP

#अन्तर्भाव "common.hpp"
#अन्तर्भाव "convert.hpp"

प्रकारव्याख्या शून्य (*dequantize_kernel_t)(स्थिरांक शून्य * vx, स्थिरांक int64_t ib, स्थिरांक पूर्णांक iqs, dfloat2 & v);
प्रकारव्याख्या शून्य (*dequantize_kernel_t_reorder)(स्थिरांक शून्य *d, स्थिरांक int64_t ib, स्थिरांक शून्य *qs,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v);
प्रकारव्याख्या शून्य (*dequantize_kernel_f32_t)(स्थिरांक शून्य * vx, स्थिरांक int64_t ib, स्थिरांक पूर्णांक iqs, दशांश & v0, दशांश & v1);

#जर QK_K == 256
स्थिर ओळीत शून्य get_scale_min_k4(पूर्णांक j, स्थिरांक uint8_t * q, uint8_t & d, uint8_t & m);
#endif

स्थिर __dpct_inline__ शून्य dequantize_q2_0(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
    स्थिरांक block_q2_0 * x = (स्थिरांक block_q2_0 *) vx;

    स्थिरांक dfloat d = x[ib].d;

    स्थिरांक पूर्णांक byte_idx = iqs / 4;
    स्थिरांक पूर्णांक shift = (iqs % 4) * 2;
    स्थिरांक uint8_t vui = x[ib].qs[byte_idx];

    v.x() = (vui >> shift) & 3;
    v.y() = (vui >> (shift + 2)) & 3;

#ifdef GGML_SYCL_F16
    v.s0() = ((dfloat)v.s0() - 1.0f) * d;
    v.s1() = ((dfloat)v.s1() - 1.0f) * d;
#अन्यथा
    v.x() = ((dfloat)v.x() - 1.0f) * d;
    v.y() = ((dfloat)v.y() - 1.0f) * d;
#endif // GGML_SYCL_F16
}

स्थिर __dpct_inline__ शून्य dequantize_q4_0(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
    स्थिरांक block_q4_0 * x = (स्थिरांक block_q4_0 *) vx;

    स्थिरांक dfloat d = x[ib].d;

    स्थिरांक पूर्णांक vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ifdef GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#अन्यथा
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#endif // GGML_SYCL_F16
}

स्थिर __dpct_inline__ शून्य dequantize_q4_0_reorder(स्थिरांक शून्य *d_ptr, स्थिरांक int64_t ib, स्थिरांक शून्य *qs,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
    // const block_q4_0 * x = (const block_q4_0 *) vx;

    स्थिरांक dfloat d = (स्थिरांक dfloat)*((स्थिरांक sycl::half*)d_ptr+ib);

    स्थिरांक पूर्णांक vui = *((स्थिरांक uint8_t *)qs+iqs);

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ifdef GGML_SYCL_F16
    // v = v - {8.0f, 8.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 8.0f) * d;
    v.s1() = (v.s1() - 8.0f) * d;

#अन्यथा
    v.x() = (v.x() - 8.0f) * d;
    v.y() = (v.y() - 8.0f) * d;
#endif // GGML_SYCL_F16
}

स्थिर __dpct_inline__ शून्य dequantize_q1_0_reorder(स्थिरांक शून्य *d_ptr, स्थिरांक int64_t ib, स्थिरांक शून्य *qs,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
    // Q1_0 reorder layout: scale values followed by quantized bits
    स्थिरांक dfloat d = (स्थिरांक dfloat)*((स्थिरांक sycl::half*)d_ptr+ib);

    स्थिरांक पूर्णांक bit_index_0 = iqs + 0;
    स्थिरांक पूर्णांक bit_index_1 = iqs + 1;

    स्थिरांक पूर्णांक bit_0 = (*((स्थिरांक uint8_t *)qs + bit_index_0 / 8) >> (bit_index_0 % 8)) & 1;
    स्थिरांक पूर्णांक bit_1 = (*((स्थिरांक uint8_t *)qs + bit_index_1 / 8) >> (bit_index_1 % 8)) & 1;

    v.x() = (2 * bit_0 - 1) * d;
    v.y() = (2 * bit_1 - 1) * d;
}

स्थिर __dpct_inline__ शून्य dequantize_q1_0(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
    स्थिरांक block_q1_0 * x = (स्थिरांक block_q1_0 *) vx;
    स्थिरांक dfloat d = x[ib].d;

    स्थिरांक पूर्णांक bit_index_0 = iqs + 0;
    स्थिरांक पूर्णांक bit_index_1 = iqs + 1;

    स्थिरांक पूर्णांक bit_0 = (x[ib].qs[bit_index_0 / 8] >> (bit_index_0 % 8)) & 1;
    स्थिरांक पूर्णांक bit_1 = (x[ib].qs[bit_index_1 / 8] >> (bit_index_1 % 8)) & 1;

    v.x() = (2 * bit_0 - 1) * d;
    v.y() = (2 * bit_1 - 1) * d;
}

स्थिर __dpct_inline__ शून्य dequantize_q4_1(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
    स्थिरांक block_q4_1 * x = (स्थिरांक block_q4_1 *) vx;

    स्थिरांक dfloat d = x[ib].dm[0];
    स्थिरांक dfloat m = x[ib].dm[1];

    स्थिरांक पूर्णांक vui = x[ib].qs[iqs];

    v.x() = vui & 0xF;
    v.y() = vui >> 4;

#ifdef GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = sycl::fma(v.s0(), d, m);
    v.s1() = sycl::fma(v.s1(), d, m);

#अन्यथा
    v.x() = sycl::fma(v.x(), d, m);
    v.y() = sycl::fma(v.y(), d, m);
#endif // GGML_SYCL_F16
}

स्थिर __dpct_inline__ शून्य dequantize_q4_K(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
#जर QK_K == 256
    स्थिरांक block_q4_K * x = (स्थिरांक block_q4_K *) vx;
    स्थिरांक sycl::half2 dm = x[ib].dm;
    स्थिरांक दशांश dall = dm[0];
    स्थिरांक दशांश dmin = dm[1];

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        स्थिरांक पूर्णांक il = idx / 64;
        स्थिरांक पूर्णांक in = idx % 64;
        स्थिरांक पूर्णांक is = 2 * il + (in >= 32 ? 1 : 0);
        स्थिरांक पूर्णांक off = in & 31;
        स्थिरांक पूर्णांक qsi = 32 * il + off;

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        स्थिरांक uint8_t q = x[ib].qs[qsi];
        स्थिरांक uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);
        परत sycl::fma((dfloat) qv, (dfloat) (dall * sc), (dfloat) (-dmin * m));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q4_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q4_K_f32(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                                स्थिरांक पूर्णांक iqs, दशांश &v0, दशांश &v1) {
#जर QK_K == 256
    स्थिरांक block_q4_K * x = (स्थिरांक block_q4_K *) vx;
    स्थिरांक sycl::half2 dm = x[ib].dm;
    स्थिरांक दशांश dall = dm[0];
    स्थिरांक दशांश dmin = dm[1];

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> दशांश {
        स्थिरांक पूर्णांक il = idx / 64;
        स्थिरांक पूर्णांक in = idx % 64;
        स्थिरांक पूर्णांक is = 2 * il + (in >= 32 ? 1 : 0);
        स्थिरांक पूर्णांक qsi = 32 * il + (in & 31);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        स्थिरांक दशांश d = dall * sc;
        स्थिरांक दशांश mn = dmin * m;
        स्थिरांक uint8_t q = x[ib].qs[qsi];
        स्थिरांक uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);

        परत d * qv - mn;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q4_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q2_K(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
#जर QK_K == 256
    स्थिरांक block_q2_K * x = (स्थिरांक block_q2_K *) vx;
    स्थिरांक दशांश dall = x[ib].dm[0];
    स्थिरांक दशांश dmin = x[ib].dm[1];

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        स्थिरांक पूर्णांक n = idx / 128;
        स्थिरांक पूर्णांक r = idx % 128;
        स्थिरांक पूर्णांक g = r / 32;
        स्थिरांक पूर्णांक l = r % 32;
        स्थिरांक पूर्णांक is = 8 * n + l / 16;

        स्थिरांक uint8_t q = x[ib].qs[32 * n + l];
        स्थिरांक uint8_t sc = x[ib].scales[is + 2 * g];
        स्थिरांक दशांश d = dall * (sc & 0xF);
        स्थिरांक दशांश m = dmin * (sc >> 4);

        परत (dfloat) d * (dfloat) ((q >> (2 * g)) & 3) - (dfloat) m;
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q2_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q2_K_f32(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                                स्थिरांक पूर्णांक iqs, दशांश &v0, दशांश &v1) {
#जर QK_K == 256
    स्थिरांक block_q2_K * x = (स्थिरांक block_q2_K *) vx;
    स्थिरांक दशांश dall = x[ib].dm[0];
    स्थिरांक दशांश dmin = x[ib].dm[1];

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> दशांश {
        स्थिरांक पूर्णांक n = idx / 128;
        स्थिरांक पूर्णांक r = idx % 128;
        स्थिरांक पूर्णांक g = r / 32;
        स्थिरांक पूर्णांक l = r % 32;
        स्थिरांक पूर्णांक is = 8 * n + l / 16;

        स्थिरांक uint8_t q = x[ib].qs[32 * n + l];
        स्थिरांक uint8_t sc = x[ib].scales[is + 2 * g];
        स्थिरांक दशांश d = dall * (sc & 0xF);
        स्थिरांक दशांश m = dmin * (sc >> 4);

        परत d * ((q >> (2 * g)) & 3) - m;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q2_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q3_K(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
#जर QK_K == 256
    स्थिरांक block_q3_K * x = (स्थिरांक block_q3_K *) vx;
    स्थिरांक दशांश d_all = x[ib].d;

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        स्थिरांक पूर्णांक n = idx / 128;
        स्थिरांक पूर्णांक r = idx % 128;
        स्थिरांक पूर्णांक j = r / 32;
        स्थिरांक पूर्णांक l = r % 32;

        स्थिरांक पूर्णांक is0 = l / 16;
        स्थिरांक पूर्णांक is = 8 * n + 2 * j + is0;
        स्थिरांक पूर्णांक shift = 2 * j;
        स्थिरांक uint8_t m = 1 << (4 * n + j);

        स्थिरांक int8_t us = is <  4 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 8] >> 0) & 3) << 4) :
                         is <  8 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 4] >> 2) & 3) << 4) :
                         is < 12 ? (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is + 0] >> 4) & 3) << 4) :
                                   (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is - 4] >> 6) & 3) << 4);

        स्थिरांक दशांश dl = d_all * (us - 32);
        स्थिरांक uint8_t q = x[ib].qs[32 * n + l];
        स्थिरांक uint8_t h = x[ib].hmask[l];
        स्थिरांक int8_t qv = ((q >> shift) & 3) - ((h & m) ? 0 : 4);

        परत (dfloat) (dl * qv);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q3_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q5_K(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
#जर QK_K == 256
    स्थिरांक block_q5_K * x = (स्थिरांक block_q5_K *) vx;
    स्थिरांक दशांश dall = x[ib].dm[0];
    स्थिरांक दशांश dmin = x[ib].dm[1];

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        स्थिरांक पूर्णांक il = idx / 64;
        स्थिरांक पूर्णांक in = idx % 64;
        स्थिरांक पूर्णांक is = 2 * il + (in >= 32 ? 1 : 0);
        स्थिरांक पूर्णांक ir = (in & 31) / 2;
        स्थिरांक पूर्णांक iq = in & 1;

        स्थिरांक uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        स्थिरांक uint8_t h = x[ib].qh[2 * ir + iq];
        स्थिरांक uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        स्थिरांक दशांश d = dall * sc;
        स्थिरांक दशांश mn = dmin * m;
        स्थिरांक uint8_t hm = 1 << (2 * il + (in >= 32 ? 1 : 0));

        परत sycl::fma((dfloat) (qv + ((h & hm) ? 16 : 0)), (dfloat) d, (dfloat) (-mn));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q5_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q5_K_f32(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                                स्थिरांक पूर्णांक iqs, दशांश &v0, दशांश &v1) {
#जर QK_K == 256
    स्थिरांक block_q5_K * x = (स्थिरांक block_q5_K *) vx;
    स्थिरांक दशांश dall = x[ib].dm[0];
    स्थिरांक दशांश dmin = x[ib].dm[1];

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> दशांश {
        स्थिरांक पूर्णांक il = idx / 64;
        स्थिरांक पूर्णांक in = idx % 64;
        स्थिरांक पूर्णांक is = 2 * il + (in >= 32 ? 1 : 0);
        स्थिरांक पूर्णांक ir = (in & 31) / 2;
        स्थिरांक पूर्णांक iq = in & 1;

        स्थिरांक uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        स्थिरांक uint8_t h = x[ib].qh[2 * ir + iq];
        स्थिरांक uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        स्थिरांक दशांश d = dall * sc;
        स्थिरांक दशांश mn = dmin * m;
        स्थिरांक uint8_t hm = 1 << (2 * il + (in >= 32 ? 1 : 0));

        परत (qv + ((h & hm) ? 16 : 0)) * d - mn;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q5_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q6_K(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
#जर QK_K == 256
    स्थिरांक block_q6_K * x = (स्थिरांक block_q6_K *) vx;
    स्थिरांक दशांश d = x[ib].d;

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        स्थिरांक पूर्णांक ip = idx / 128;
        स्थिरांक पूर्णांक in = idx % 128;
        स्थिरांक पूर्णांक il = in & 31;
        स्थिरांक पूर्णांक ig = in / 32;
        स्थिरांक पूर्णांक is = 8 * ip + il / 16;

        स्थिरांक uint8_t ql0 = x[ib].ql[64 * ip + il];
        स्थिरांक uint8_t ql1 = x[ib].ql[64 * ip + il + 32];
        स्थिरांक uint8_t qh = x[ib].qh[32 * ip + il];
        स्थिरांक int8_t * sc = x[ib].scales + is;

        uint8_t qv;
        int8_t scale;
        जर (ig == 0) {
            qv = (ql0 & 0xF) | (((qh >> 0) & 3) << 4);
            scale = sc[0];
        } अन्यथा जर (ig == 1) {
            qv = (ql1 & 0xF) | (((qh >> 2) & 3) << 4);
            scale = sc[2];
        } अन्यथा जर (ig == 2) {
            qv = (ql0 >> 4) | (((qh >> 4) & 3) << 4);
            scale = sc[4];
        } अन्यथा {
            qv = (ql1 >> 4) | (((qh >> 6) & 3) << 4);
            scale = sc[6];
        }

        परत (dfloat) (d * scale * ((int8_t) qv - 32));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q6_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_mxfp4(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                             स्थिरांक पूर्णांक iqs, dfloat2 &v) {
    स्थिरांक block_mxfp4 * x = (स्थिरांक block_mxfp4 *) vx;
    स्थिरांक दशांश d = ggml_sycl_e8m0_to_fp32(x[ib].e);
    स्थिरांक uint8_t q = x[ib].qs[iqs];

    v.x() = d * kvalues_mxfp4[q & 0xF] * 0.5f;
    v.y() = d * kvalues_mxfp4[q >> 4] * 0.5f;
}

स्थिर __dpct_inline__ शून्य dequantize_nvfp4(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                             स्थिरांक पूर्णांक iqs, dfloat2 &v) {
    स्थिरांक block_nvfp4 & xb = ((स्थिरांक block_nvfp4 *) vx)[ib];

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        स्थिरांक पूर्णांक sub = idx / QK_NVFP4_SUB;
        स्थिरांक पूर्णांक j = idx % QK_NVFP4_SUB;
        स्थिरांक पूर्णांक jh = j % (QK_NVFP4_SUB / 2);

        स्थिरांक दशांश d = ggml_sycl_ue4m3_to_fp32(xb.d[sub]);
        स्थिरांक uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + jh];
        स्थिरांक uint8_t qv = (j < (QK_NVFP4_SUB / 2)) ? (q & 0x0F) : (q >> 4);

        परत d * kvalues_mxfp4[qv];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
}

स्थिर __dpct_inline__ शून्य dequantize_iq2_xxs(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                               स्थिरांक पूर्णांक iqs, dfloat2 &v) {
#जर QK_K == 256
    स्थिरांक block_iq2_xxs * x = (स्थिरांक block_iq2_xxs *) vx;

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        स्थिरांक पूर्णांक ib8 = idx / 32;
        स्थिरांक पूर्णांक r = idx % 32;
        स्थिरांक पूर्णांक il = r / 8;
        स्थिरांक पूर्णांक j = r % 8;

        स्थिरांक uint16_t * q2 = x[ib].qs + 4 * ib8;
        स्थिरांक uint8_t * aux8 = (स्थिरांक uint8_t *) q2;
        स्थिरांक uint8_t * grid = (स्थिरांक uint8_t *) (iq2xxs_grid + aux8[il]);
        स्थिरांक uint32_t aux32 = q2[2] | (q2[3] << 16);
        स्थिरांक दशांश d = (दशांश) x[ib].d * (0.5f + (aux32 >> 28)) * 0.25f;
        स्थिरांक uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        परत d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ2_XXS dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_iq2_xs(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                              स्थिरांक पूर्णांक iqs, dfloat2 &v) {
#जर QK_K == 256
    स्थिरांक block_iq2_xs * x = (स्थिरांक block_iq2_xs *) vx;

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        स्थिरांक पूर्णांक ib8 = idx / 32;
        स्थिरांक पूर्णांक r = idx % 32;
        स्थिरांक पूर्णांक il = r / 8;
        स्थिरांक पूर्णांक j = r % 8;

        स्थिरांक uint16_t * q2 = x[ib].qs + 4 * ib8;
        स्थिरांक uint8_t * grid = (स्थिरांक uint8_t *) (iq2xs_grid + (q2[il] & 511));
        स्थिरांक दशांश d = (दशांश) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        स्थिरांक uint8_t signs = ksigns_iq2xs[q2[il] >> 9];

        परत d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ2_XS dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_iq2_s(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                             स्थिरांक पूर्णांक iqs, dfloat2 &v) {
#जर QK_K == 256
    स्थिरांक block_iq2_s * x = (स्थिरांक block_iq2_s *) vx;

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        स्थिरांक पूर्णांक ib8 = idx / 32;
        स्थिरांक पूर्णांक r = idx % 32;
        स्थिरांक पूर्णांक il = r / 8;
        स्थिरांक पूर्णांक j = r % 8;

        स्थिरांक uint16_t grid_id = x[ib].qs[4 * ib8 + il] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 0x300);
        स्थिरांक uint8_t * grid = (स्थिरांक uint8_t *) (iq2s_grid + grid_id);
        स्थिरांक दशांश d = (दशांश) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        स्थिरांक uint8_t signs = x[ib].qs[QK_K / 8 + 4 * ib8 + il];

        परत d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ2_S dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_iq3_xxs(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                               स्थिरांक पूर्णांक iqs, dfloat2 &v) {
#जर QK_K == 256
    स्थिरांक block_iq3_xxs * x = (स्थिरांक block_iq3_xxs *) vx;

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        स्थिरांक पूर्णांक ib8 = idx / 32;
        स्थिरांक पूर्णांक r = idx % 32;
        स्थिरांक पूर्णांक il = r / 8;
        स्थिरांक पूर्णांक j = r % 8;

        स्थिरांक uint8_t * q3 = x[ib].qs + 8 * ib8;
        स्थिरांक uint16_t * gas = (स्थिरांक uint16_t *) (x[ib].qs + QK_K / 4) + 2 * ib8;
        स्थिरांक uint8_t * grid1 = (स्थिरांक uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        स्थिरांक uint8_t * grid2 = (स्थिरांक uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        स्थिरांक uint32_t aux32 = gas[0] | (gas[1] << 16);
        स्थिरांक दशांश d = (दशांश) x[ib].d * (0.5f + (aux32 >> 28)) * 0.5f;
        स्थिरांक uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        जर (j < 4) {
            परत d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        परत d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ3_XXS dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_iq3_s(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                             स्थिरांक पूर्णांक iqs, dfloat2 &v) {
#जर QK_K == 256
    स्थिरांक block_iq3_s * x = (स्थिरांक block_iq3_s *) vx;

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        स्थिरांक पूर्णांक ib8 = idx / 32;
        स्थिरांक पूर्णांक r = idx % 32;
        स्थिरांक पूर्णांक il = r / 8;
        स्थिरांक पूर्णांक j = r % 8;

        स्थिरांक uint8_t * qs = x[ib].qs + 8 * ib8;
        स्थिरांक uint16_t grid1_id = qs[2 * il + 0] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 256);
        स्थिरांक uint16_t grid2_id = qs[2 * il + 1] | ((x[ib].qh[ib8] << (7 - 2 * il)) & 256);
        स्थिरांक uint8_t * grid1 = (स्थिरांक uint8_t *) (iq3s_grid + grid1_id);
        स्थिरांक uint8_t * grid2 = (स्थिरांक uint8_t *) (iq3s_grid + grid2_id);
        स्थिरांक दशांश d = (दशांश) x[ib].d * (1 + 2 * ((x[ib].scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
        स्थिरांक uint8_t signs = x[ib].signs[4 * ib8 + il];

        जर (j < 4) {
            परत d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        परत d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ3_S dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_iq1_s(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                             स्थिरांक पूर्णांक iqs, dfloat2 &v) {
#जर QK_K == 256
    स्थिरांक block_iq1_s * x = (स्थिरांक block_iq1_s *) vx;

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        स्थिरांक पूर्णांक ib8 = idx / 32;
        स्थिरांक पूर्णांक r = idx % 32;
        स्थिरांक पूर्णांक il = r / 8;
        स्थिरांक पूर्णांक j = r % 8;

        स्थिरांक दशांश delta = (x[ib].qh[ib8] & 0x8000) ? (-1.f - IQ1S_DELTA) : (-1.f + IQ1S_DELTA);
        स्थिरांक दशांश d = (दशांश) x[ib].d * (2 * ((x[ib].qh[ib8] >> 12) & 7) + 1);
        स्थिरांक uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((x[ib].qh[ib8] >> (3 * il)) & 7) << 8);
        स्थिरांक uint32_t g = iq1s_grid_gpu[grid_id];
        स्थिरांक int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        परत d * (qv + delta);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ1_S dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_iq1_m(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                             स्थिरांक पूर्णांक iqs, dfloat2 &v) {
#जर QK_K == 256
    स्थिरांक block_iq1_m * x = (स्थिरांक block_iq1_m *) vx;

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        स्थिरांक पूर्णांक ib8 = idx / 32;
        स्थिरांक पूर्णांक r = idx % 32;
        स्थिरांक पूर्णांक il = r / 8;
        स्थिरांक पूर्णांक j = r % 8;

        स्थिरांक uint16_t * sc = (स्थिरांक uint16_t *) x[ib].scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        स्थिरांक पूर्णांक ib16 = 2 * ib8 + il / 2;
        स्थिरांक दशांश d = (दशांश) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);

        स्थिरांक uint8_t qh = x[ib].qh[2 * ib8 + il / 2];
        स्थिरांक दशांश delta = (qh & (0x08 << (4 * (il % 2)))) ? (-1.f - IQ1M_DELTA) : (-1.f + IQ1M_DELTA);

        स्थिरांक uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((qh >> (4 * (il % 2))) & 7) << 8);
        स्थिरांक uint32_t g = iq1s_grid_gpu[grid_id];
        स्थिरांक int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        परत d * (qv + delta);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ1_M dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_iq4_nl(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                              स्थिरांक पूर्णांक iqs, dfloat2 &v) {
    स्थिरांक block_iq4_nl * x = (स्थिरांक block_iq4_nl *) vx;
    स्थिरांक दशांश d = (दशांश) x[ib].d;

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        जर (idx < 16) {
            परत d * kvalues_iq4nl[x[ib].qs[idx] & 0xF];
        }
        परत d * kvalues_iq4nl[x[ib].qs[idx - 16] >> 4];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
}

स्थिर __dpct_inline__ शून्य dequantize_iq4_xs(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                              स्थिरांक पूर्णांक iqs, dfloat2 &v) {
#जर QK_K == 256
    स्थिरांक block_iq4_xs * x = (स्थिरांक block_iq4_xs *) vx;

    स्वतः dequantize_one = [&](स्थिरांक पूर्णांक idx) -> dfloat {
        स्थिरांक पूर्णांक ib8 = idx / 32;
        स्थिरांक पूर्णांक r = idx % 32;
        स्थिरांक पूर्णांक byte_idx = (r < 16) ? r : (r - 16);
        स्थिरांक uint8_t q = x[ib].qs[16 * ib8 + byte_idx];
        स्थिरांक uint8_t qv = (r < 16) ? (q & 0x0F) : (q >> 4);

        स्थिरांक दशांश d = (दशांश) x[ib].d * ((((x[ib].scales_l[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf) |
                        (((x[ib].scales_h >> (2 * ib8)) & 3) << 4)) - 32);
        परत d * kvalues_iq4nl[qv];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ4_XS dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q5_0(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
    स्थिरांक block_q5_0 * x = (स्थिरांक block_q5_0 *) vx;

    स्थिरांक dfloat d = x[ib].d;

    uint32_t qh;
    स_नकल(&qh, x[ib].qh, आकार(qh));

    स्थिरांक पूर्णांक xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    स्थिरांक पूर्णांक xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#ifdef GGML_SYCL_F16
    // v = v - {16.0f, 16.0f};
    // v = v * {d, d};
    v.s0() = (v.s0() - 16.0f) * d;
    v.s1() = (v.s1() - 16.0f) * d;

#अन्यथा
    v.x() = (v.x() - 16.0f) * d;
    v.y() = (v.y() - 16.0f) * d;
#endif // GGML_SYCL_F16
}

स्थिर __dpct_inline__ शून्य dequantize_q5_1(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
    स्थिरांक block_q5_1 * x = (स्थिरांक block_q5_1 *) vx;

    स्थिरांक dfloat d = x[ib].dm[0];
    स्थिरांक dfloat m = x[ib].dm[1];

    uint32_t qh;
    स_नकल(&qh, x[ib].qh, आकार(qh));

    स्थिरांक पूर्णांक xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    स्थिरांक पूर्णांक xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

    v.x() = ((x[ib].qs[iqs] & 0xf) | xh_0);
    v.y() = ((x[ib].qs[iqs] >> 4) | xh_1);

#ifdef GGML_SYCL_F16
    // v = v * {d, d};
    // v = v + {m, m};
    v.s0() = sycl::fma(v.s0(), d, m);
    v.s1() = sycl::fma(v.s1(), d, m);
#अन्यथा
    v.x() = sycl::fma(v.x(), d, m);
    v.y() = sycl::fma(v.y(), d, m);
#endif // GGML_SYCL_F16
}

स्थिर __dpct_inline__ शून्य dequantize_q8_0_reorder(स्थिरांक शून्य *d_ptr, स्थिरांक int64_t ib, स्थिरांक शून्य *qs,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
    स्थिरांक dfloat d = (स्थिरांक dfloat)*((स्थिरांक sycl::half*)d_ptr + ib);

    v.x() = ((स्थिरांक int8_t *)qs)[iqs + 0];
    v.y() = ((स्थिरांक int8_t *)qs)[iqs + 1];

#ifdef GGML_SYCL_F16
    v.s0() *= d;
    v.s1() *= d;
#अन्यथा
    v.x() *= d;
    v.y() *= d;
#endif // GGML_SYCL_F16
}

स्थिर __dpct_inline__ शून्य dequantize_q8_0(स्थिरांक शून्य *vx, स्थिरांक int64_t ib,
                                            स्थिरांक पूर्णांक iqs, dfloat2 &v) {
    स्थिरांक block_q8_0 * x = (स्थिरांक block_q8_0 *) vx;

    स्थिरांक dfloat d = x[ib].d;

    v.x() = x[ib].qs[iqs + 0];
    v.y() = x[ib].qs[iqs + 1];

#ifdef GGML_SYCL_F16
    // v = v * {d, d};
    v.s0() *= d;
    v.s1() *= d;
#अन्यथा
    v.x() *= d;
    v.y() *= d;
#endif // GGML_SYCL_F16
}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q4_0(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  स्थिरांक sycl::nd_item<3> &item_ct1) {

    स्थिरांक int64_t i = item_ct1.get_group(2);

    // assume 32 threads
    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
    स्थिरांक int64_t il  = tid/8;
    स्थिरांक int64_t ir  = tid%8;
    स्थिरांक int64_t ib = 8*i + ir;
    जर (ib >= nb32) {
        परत;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    स्थिरांक block_q4_0 * x = (स्थिरांक block_q4_0 *)vx + ib;
    स्थिरांक दशांश d = sycl::vec<sycl::half, 1>(x->d)
                        .convert<दशांश, sycl::rounding_mode::automatic>()[0];
    स्थिरांक दशांश dm = -8*d;

    स्थिरांक uint8_t * q = x->qs + 4*il;

    साठी (पूर्णांक l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q4_0_reorder(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                  स्थिरांक sycl::nd_item<3> &item_ct1) {
    // one thread per qs byte, like dequantize_block<dequantize_q4_0>; the reorder layout makes the byte reads contiguous
    // grid-stride loop because the launcher may shrink the work-group to keep the global range below INT_MAX
    // float math, not dequantize_q4_0_reorder: that helper computes in half with GGML_SYCL_F16, which changes f32 output
    स्थिरांक sycl::half * d_ptr  = (स्थिरांक sycl::half *) ((स्थिरांक uint8_t *) vx + k/2);
    स्थिरांक int64_t      stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    साठी (int64_t iq = item_ct1.get_global_id(2); iq < k/2; iq += stride) {
        स्थिरांक int64_t ib  = iq / (QK4_0/2);
        स्थिरांक int64_t iqs = iq % (QK4_0/2);
        स्थिरांक पूर्णांक     vq  = ((स्थिरांक uint8_t *) vx)[iq];
        स्थिरांक दशांश   d   = दशांश(d_ptr[ib]);

        dst_t * y = yy + ib*QK4_0 + iqs;
        y[0]       = d * ((vq & 0xF) - 8);
        y[QK4_0/2] = d * ((vq >>  4) - 8);
    }
}

// Dequantize Q8_0 from reorder layout: [all qs (k bytes)][all d values]
ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q8_0_reorder(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                  स्थिरांक sycl::nd_item<3> &item_ct1) {
    // two values per thread, like dequantize_block<dequantize_q8_0>; the reorder layout keeps all qs contiguous
    // grid-stride loop because the launcher may shrink the work-group to keep the global range below INT_MAX
    // float math, not dequantize_q8_0_reorder: that helper computes in half with GGML_SYCL_F16, which changes f32 output
    स्थिरांक int8_t *     qs     = (स्थिरांक int8_t *) vx;
    स्थिरांक sycl::half * d_ptr  = (स्थिरांक sycl::half *) ((स्थिरांक uint8_t *) vx + k);
    स्थिरांक int64_t      stride = 2 * item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    साठी (int64_t i = 2 * item_ct1.get_global_id(2); i < k; i += stride) {
        स्थिरांक दशांश d = दशांश(d_ptr[i / QK8_0]);

        yy[i + 0] = d * qs[i + 0];
        yy[i + 1] = d * qs[i + 1];
    }
}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q8_0_reorder_wide(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                               स्थिरांक sycl::id<1> idx) {
    स्थिरांक int64_t i = 4 * (int64_t) idx[0];

    स्थिरांक int8_t *     qs    = (स्थिरांक int8_t *) vx;
    स्थिरांक sycl::half * d_ptr = (स्थिरांक sycl::half *) ((स्थिरांक uint8_t *) vx + k);

    स्थिरांक दशांश d0 = दशांश(d_ptr[i / QK8_0]);
    स्थिरांक दशांश d1 = दशांश(d_ptr[(i + 2) / QK8_0]);

    sycl::vec<दशांश, 4> p;
    p[0] = d0 * (दशांश) qs[i + 0];
    p[1] = d0 * (दशांश) qs[i + 1];
    p[2] = d1 * (दशांश) qs[i + 2];
    p[3] = d1 * (दशांश) qs[i + 3];

    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yy + i) = p.convert<dst_t, sycl::rounding_mode::automatic>();
}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q4_1(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  स्थिरांक sycl::nd_item<3> &item_ct1) {

    स्थिरांक int64_t i = item_ct1.get_group(2);

    // assume 32 threads
    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
    स्थिरांक int64_t il  = tid/8;
    स्थिरांक int64_t ir  = tid%8;
    स्थिरांक int64_t ib = 8*i + ir;
    जर (ib >= nb32) {
        परत;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    स्थिरांक block_q4_1 * x = (स्थिरांक block_q4_1 *)vx + ib;
    स्थिरांक sycl::float2 d =
        x->dm.convert<दशांश, sycl::rounding_mode::automatic>();

    स्थिरांक uint8_t * q = x->qs + 4*il;

    साठी (पूर्णांक l = 0; l < 4; ++l) {
        y[l + 0] = d.x() * (q[l] & 0xF) + d.y();
        y[l + 16] = d.x() * (q[l] >> 4) + d.y();
    }
}


//================================== k-quants

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q2_K(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy,
                                  स्थिरांक sycl::nd_item<3> &item_ct1) {

    स्थिरांक int64_t i = item_ct1.get_group(2);
    स्थिरांक block_q2_K * x = (स्थिरांक block_q2_K *) vx;

    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
#जर QK_K == 256
    स्थिरांक int64_t n   = tid/32;
    स्थिरांक int64_t l   = tid - 32*n;
    स्थिरांक int64_t is  = 8*n + l/16;

    स्थिरांक uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    दशांश dall = x[i].dm[0];
    दशांश dmin = x[i].dm[1];
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#अन्यथा
    स्थिरांक int64_t is = tid/16;  // 0 or 1
    स्थिरांक int64_t il = tid%16;  // 0...15
    स्थिरांक uint8_t q = x[i].qs[il] >> (2*is);
    dst_t * y = yy + i*QK_K + 16*is + il;

    दशांश dall = x[i].dm[0];
    दशांश dmin = x[i].dm[1];
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#endif

}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q2_K_reorder(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, स्थिरांक int64_t i,
                                          स्थिरांक sycl::nd_item<3> & item_ct1, int64_t n_blocks) {
#जर QK_K == 256
    जर (i >= n_blocks) {
        परत;
    }

    स्थिरांक uint8_t * base          = static_cast<स्थिरांक uint8_t *>(vx);
    स्थिरांक माप_प्रकार    qs_offset     = i * (QK_K / 4);
    स्थिरांक माप_प्रकार    scales_offset = n_blocks * (QK_K / 4) + i * (QK_K / 16);
    स्थिरांक माप_प्रकार    dm_offset     = n_blocks * (QK_K / 4) + n_blocks * (QK_K / 16) + i * आकार(ggml_half2);

    स्थिरांक uint8_t *     qs     = base + qs_offset;
    स्थिरांक uint8_t *     scales = base + scales_offset;
    स्थिरांक ggml_half2 * dm     = reinterpret_cast<स्थिरांक ggml_half2 *>(base + dm_offset);

    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
    स्थिरांक int64_t n    = tid / 32;
    स्थिरांक int64_t l    = tid - 32 * n;
    स्थिरांक int64_t is   = 8 * n + l / 16;

    स्थिरांक uint8_t q = qs[32 * n + l];
    dst_t * y = yy + 128 * n;

    स्थिरांक दशांश dall = (*dm)[0];
    स्थिरांक दशांश dmin = (*dm)[1];
    y[l+ 0] = dall * (scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (scales[is+0] >> 4);
    y[l+32] = dall * (scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (scales[is+2] >> 4);
    y[l+64] = dall * (scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (scales[is+4] >> 4);
    y[l+96] = dall * (scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (scales[is+6] >> 4);
#अन्यथा
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(i);
    GGML_UNUSED(item_ct1);
    GGML_UNUSED(n_blocks);
    GGML_ABORT("Q2_K reorder dequantize not supported for QK_K != 256");
#endif
}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q3_K(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy,
                                  स्थिरांक sycl::nd_item<3> &item_ct1) {

    स्थिरांक int64_t i = item_ct1.get_group(2);
    स्थिरांक block_q3_K * x = (स्थिरांक block_q3_K *) vx;

#जर QK_K == 256
    स्थिरांक int64_t r = item_ct1.get_local_id(2) / 4;
    स्थिरांक int64_t tid = r/2;
    स्थिरांक int64_t is0 = r%2;
    स्थिरांक int64_t l0 = 16 * is0 + 4 * (item_ct1.get_local_id(2) % 4);
    स्थिरांक int64_t n = tid / 4;
    स्थिरांक int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    पूर्णांक shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    दशांश d_all = x[i].d;
    दशांश dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    स्थिरांक uint8_t * q = x[i].qs + 32*n;
    स्थिरांक uint8_t * hm = x[i].hmask;

    साठी (पूर्णांक l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
#अन्यथा
    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
    स्थिरांक int64_t is  = tid/16;  // 0 or 1
    स्थिरांक int64_t il  = tid%16;  // 0...15
    स्थिरांक int64_t im  = il/8;    // 0...1
    स्थिरांक int64_t in  = il%8;    // 0...7

    dst_t * y = yy + i*QK_K + 16*is + il;

    स्थिरांक uint8_t q = x[i].qs[il] >> (2*is);
    स्थिरांक uint8_t h = x[i].hmask[in] >> (2*is + im);
    स्थिरांक दशांश   d = (दशांश)x[i].d;

    जर (is == 0) {
        y[ 0] = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    } अन्यथा {
        y[ 0] = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    }
#endif

}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q3_K_reorder(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, स्थिरांक int64_t i,
                                          स्थिरांक sycl::nd_item<3> & item_ct1, int64_t n_blocks) {
#जर QK_K == 256
    जर (i >= n_blocks) {
        परत;
    }

    स्थिरांक uint8_t * base          = static_cast<स्थिरांक uint8_t *>(vx);
    स्थिरांक माप_प्रकार    qs_offset     = i * (QK_K / 4);
    स्थिरांक माप_प्रकार    hmask_offset  = n_blocks * (QK_K / 4) + i * (QK_K / 8);
    स्थिरांक माप_प्रकार    scales_offset = n_blocks * (QK_K / 4) + n_blocks * (QK_K / 8) + i * 12;
    स्थिरांक माप_प्रकार    d_offset      = n_blocks * (QK_K / 4) + n_blocks * (QK_K / 8) + n_blocks * 12 +
                                 i * आकार(ggml_half);

    स्थिरांक uint8_t * qs     = base + qs_offset;
    स्थिरांक uint8_t * hmask  = base + hmask_offset;
    स्थिरांक uint8_t * scales = base + scales_offset;
    स्थिरांक दशांश     d_all  = static_cast<दशांश>(*reinterpret_cast<स्थिरांक ggml_half *>(base + d_offset));

    स्थिरांक int64_t r    = item_ct1.get_local_id(2) / 4;
    स्थिरांक int64_t tid  = r / 2;
    स्थिरांक int64_t is0  = r % 2;
    स्थिरांक int64_t l0   = 16 * is0 + 4 * (item_ct1.get_local_id(2) % 4);
    स्थिरांक int64_t n    = tid / 4;
    स्थिरांक int64_t j    = tid - 4 * n;
    स्थिरांक int64_t is   = 8 * n + 2 * j + is0;
    स्थिरांक पूर्णांक     shift = 2 * j;
    uint8_t       m    = 1 << (4 * n + j);

    uint8_t us = is < 4
        ? (scales[is - 0] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4)
        : is < 8
            ? (scales[is - 0] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4)
            : is < 12
                ? (scales[is - 8] >> 4) | (((scales[is + 0] >> 4) & 3) << 4)
                : (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);

    स्थिरांक दशांश dl = d_all * (us - 32);

    dst_t * y = yy + 128 * n + 32 * j;
    स्थिरांक uint8_t * q  = qs + 32 * n;
    स्थिरांक uint8_t * hm = hmask;

    साठी (पूर्णांक l = l0; l < l0 + 4; ++l) {
        y[l] = dl * ((int8_t) ((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
    }
#अन्यथा
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(i);
    GGML_UNUSED(item_ct1);
    GGML_UNUSED(n_blocks);
    GGML_ABORT("Q3_K reorder dequantize not supported for QK_K != 256");
#endif
}

#जर QK_K == 256
स्थिर ओळीत शून्य get_scale_min_k4(पूर्णांक j, स्थिरांक uint8_t * q, uint8_t & d, uint8_t & m) {
    जर (j < 4) {
        d = q[j] & 63;
        m = q[j + 4] & 63;
    } अन्यथा {
        d = (q[j+4] & 0xF) | ((q[j-4] >> 6) << 4);
        m = (q[j+4] >>  4) | ((q[j-0] >> 6) << 4);
    }
}
#endif

// Work-items that cover one 256-weight K-quant block: four 64-element chunks, 8 work-items
// per chunk, 4 qs bytes per work-item.
स्थिर constexpr पूर्णांक SYCL_DEQK_WI_PER_BLOCK = 32;

ढाँचा <typename dst_t>
स्थिर ओळीत शून्य dequantize_q4_K_wide_one(dst_t * __restrict__ y, स्थिरांक uint8_t * __restrict__ qs,
                                            स्थिरांक uint8_t * __restrict__ scales, स्थिरांक दशांश dall,
                                            स्थिरांक दशांश dmin, स्थिरांक पूर्णांक tid) {
    स्थिरांक पूर्णांक il = tid >> 3;        // 64-element chunk, 0..3
    स्थिरांक पूर्णांक ir = (tid & 7) * 4;   // 4 qs bytes within the chunk

    uint8_t sc, m;
    get_scale_min_k4(2 * il + 0, scales, sc, m);
    स्थिरांक दशांश d1 = dall * sc;
    स्थिरांक दशांश m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, scales, sc, m);
    स्थिरांक दशांश d2 = dall * sc;
    स्थिरांक दशांश m2 = dmin * m;

    स्थिरांक sycl::vec<uint8_t, 4> q = vec_aligned_load<uint8_t, 4>(qs + 32 * il + ir);

    sycl::vec<dst_t, 4> lo, hi;
#pragma unroll
    साठी (पूर्णांक l = 0; l < 4; ++l) {
        lo[l] = d1 * (q[l] & 0xF) - m1;
        hi[l] = d2 * (q[l] >> 4)  - m2;
    }
    dst_t * yb = y + 64 * il + ir;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb +  0) = lo;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb + 32) = hi;
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_block_q4_K_wide(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb,
                                       स्थिरांक sycl::nd_item<1> & it) {
    static_assert(QK_K == 256, "wide K-quant dequant assumes QK_K == 256");
    स्थिरांक int64_t g   = it.get_global_id(0);
    स्थिरांक int64_t b   = g / SYCL_DEQK_WI_PER_BLOCK;
    स्थिरांक पूर्णांक     tid = (पूर्णांक) (g % SYCL_DEQK_WI_PER_BLOCK);
    जर (b >= nb) {
        परत;
    }
    स्थिरांक block_q4_K * x = (स्थिरांक block_q4_K *) vx + b;
    dequantize_q4_K_wide_one(yy + b * QK_K, x->qs, x->scales, (दशांश) x->dm[0], (दशांश) x->dm[1], tid);
}


ढाँचा <typename dst_t>
ओळीत शून्य dequantize_q4_K_common(dst_t * __restrict__ y, स्थिरांक uint8_t * __restrict__ qs_ptr, स्थिरांक दशांश dall,
                                   स्थिरांक दशांश dmin, uint8_t * __restrict__ scales_local, पूर्णांक il, पूर्णांक ir) {
    स्थिरांक पूर्णांक is = 2 * il;
    constexpr पूर्णांक n  = 4;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    स्थिरांक दशांश d1 = dall * sc;
    स्थिरांक दशांश m1 = dmin * m;

    get_scale_min_k4(is + 1, scales_local, sc, m);
    स्थिरांक दशांश d2 = dall * sc;
    स्थिरांक दशांश m2 = dmin * m;

    sycl::vec<uint8_t, n> q_vec = vec_aligned_load<uint8_t, n>(qs_ptr + 32 * il + n * ir);
    साठी (पूर्णांक l = 0; l < n; ++l) {
        y[l + 0]  = d1 * (q_vec[l] & 0xF) - m1;
        y[l + 32] = d2 * (q_vec[l] >> 4) - m2;
    }
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_block_q4_K_reorder(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, स्थिरांक int64_t i,
                                          uint8_t * scales_local, स्थिरांक sycl::nd_item<1> & item_ct1, int64_t nb) {
    स्थिरांक int64_t tid = item_ct1.get_local_id(0);  // thread index within block
    स्थिरांक int64_t il  = tid / 8;
    स्थिरांक int64_t ir  = tid % 8;

    dst_t * y = yy + 64 * il + 4 * ir;

    स्थिरांक uint8_t * base          = static_cast<स्थिरांक uint8_t *>(vx);
    स्थिरांक माप_प्रकार    qs_offset     = i * (QK_K / 2);
    स्थिरांक माप_प्रकार    scales_offset = nb * (QK_K / 2) + i * K_SCALE_SIZE;
    स्थिरांक माप_प्रकार    dm_offset     = nb * (QK_K / 2) + nb * K_SCALE_SIZE + i * आकार(ggml_half2);

    स्थिरांक uint8_t *    qs_ptr     = base + qs_offset;
    स्थिरांक uint8_t *    scales_ptr = base + scales_offset;
    ggml_half2         dm_मूल्ये  = *reinterpret_cast<स्थिरांक ggml_half2 *>(base + dm_offset);

    स्थिरांक दशांश dall = dm_मूल्ये.x();
    स्थिरांक दशांश dmin = dm_मूल्ये.y();

    जर (tid < 12) {
        scales_local[tid] = scales_ptr[tid];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
    dequantize_q4_K_common(y, qs_ptr, dall, dmin, scales_local, il, ir);
}

ढाँचा <typename dst_t>
स्थिर ओळीत शून्य dequantize_q5_K_wide_one(dst_t * __restrict__ y, स्थिरांक uint8_t * __restrict__ qs,
                                            स्थिरांक uint8_t * __restrict__ qh_ptr,
                                            स्थिरांक uint8_t * __restrict__ scales, स्थिरांक दशांश dall,
                                            स्थिरांक दशांश dmin, स्थिरांक पूर्णांक tid) {
    स्थिरांक पूर्णांक il = tid >> 3;        // 64-element chunk, 0..3
    स्थिरांक पूर्णांक ir = (tid & 7) * 4;   // 4 qs bytes within the chunk; qh holds the 5th bit

    uint8_t sc, m;
    get_scale_min_k4(2 * il + 0, scales, sc, m);
    स्थिरांक दशांश d1 = dall * sc;
    स्थिरांक दशांश m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, scales, sc, m);
    स्थिरांक दशांश d2 = dall * sc;
    स्थिरांक दशांश m2 = dmin * m;

    स्थिरांक sycl::vec<uint8_t, 4> q = vec_aligned_load<uint8_t, 4>(qs + 32 * il + ir);
    स्थिरांक sycl::vec<uint8_t, 4> h = vec_aligned_load<uint8_t, 4>(qh_ptr + ir);

    स्थिरांक uint8_t hm1 = 1 << (2 * il);
    स्थिरांक uint8_t hm2 = hm1 << 1;

    sycl::vec<dst_t, 4> lo, hi;
#pragma unroll
    साठी (पूर्णांक l = 0; l < 4; ++l) {
        lo[l] = d1 * ((q[l] & 0xF) + ((h[l] & hm1) ? 16 : 0)) - m1;
        hi[l] = d2 * ((q[l] >>  4) + ((h[l] & hm2) ? 16 : 0)) - m2;
    }
    dst_t * yb = y + 64 * il + ir;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb +  0) = lo;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb + 32) = hi;
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_block_q5_K_wide(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb,
                                       स्थिरांक sycl::nd_item<1> & it) {
    static_assert(QK_K == 256, "wide K-quant dequant assumes QK_K == 256");
    स्थिरांक int64_t g   = it.get_global_id(0);
    स्थिरांक int64_t b   = g / SYCL_DEQK_WI_PER_BLOCK;
    स्थिरांक पूर्णांक     tid = (पूर्णांक) (g % SYCL_DEQK_WI_PER_BLOCK);
    जर (b >= nb) {
        परत;
    }
    स्थिरांक block_q5_K * x = (स्थिरांक block_q5_K *) vx + b;
    dequantize_q5_K_wide_one(yy + b * QK_K, x->qs, x->qh, x->scales, (दशांश) x->dm[0], (दशांश) x->dm[1], tid);
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_block_q5_K_reorder(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, स्थिरांक int64_t ib,
                                          uint8_t * scales_local, स्थिरांक sycl::nd_item<3> & item_ct1, int64_t n_blocks) {

#जर QK_K == 256
    // assume 64 threads
    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
    स्थिरांक int64_t il  = tid / 16;   // 0...3
    स्थिरांक int64_t ir  = tid % 16;   // 0...15
    स्थिरांक int64_t is  = 2 * il;

    dst_t * y = yy + 64 * il + 2 * ir;

    स्थिरांक uint8_t * base = static_cast<स्थिरांक uint8_t *>(vx);

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales (K_SCALE_SIZE per block)] [dm (half2 per block)]
    स्थिरांक माप_प्रकार qs_offset     = ib * (QK_K / 2);
    स्थिरांक माप_प्रकार qh_offset     = n_blocks * (QK_K / 2) + ib * (QK_K / 8);
    स्थिरांक माप_प्रकार scales_offset = n_blocks * (QK_K / 2) + n_blocks * (QK_K / 8) + ib * K_SCALE_SIZE;
    स्थिरांक माप_प्रकार dm_offset     = n_blocks * (QK_K / 2) + n_blocks * (QK_K / 8) + n_blocks * K_SCALE_SIZE + ib * आकार(ggml_half2);

    स्थिरांक uint8_t *  qs_ptr     = base + qs_offset;
    स्थिरांक uint8_t *  qh_ptr     = base + qh_offset;
    स्थिरांक uint8_t *  scales_ptr = base + scales_offset;
    स्थिरांक ggml_half2 dm_मूल्ये  = *reinterpret_cast<स्थिरांक ggml_half2 *>(base + dm_offset);

    स्थिरांक दशांश dall = dm_मूल्ये.x();
    स्थिरांक दशांश dmin = dm_मूल्ये.y();

    स्थिरांक uint8_t * ql = qs_ptr + 32 * il + 2 * ir;
    स्थिरांक uint8_t * qh = qh_ptr + 2 * ir;

    जर (tid < K_SCALE_SIZE) {
        scales_local[tid] = scales_ptr[tid];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    स्थिरांक दशांश d1 = dall * sc; स्थिरांक दशांश m1 = dmin * m;
    get_scale_min_k4(is + 1, scales_local, sc, m);
    स्थिरांक दशांश d2 = dall * sc; स्थिरांक दशांश m2 = dmin * m;

    uint8_t hm  = 1 << (2 * il);
    y[ 0] = d1 * ((ql[ 0] & 0xF) + (qh[ 0] & hm ? 16 : 0)) - m1;
    y[ 1] = d1 * ((ql[ 1] & 0xF) + (qh[ 1] & hm ? 16 : 0)) - m1;
    hm <<= 1;
    y[32] = d2 * ((ql[ 0] >>  4) + (qh[ 0] & hm ? 16 : 0)) - m2;
    y[33] = d2 * ((ql[ 1] >>  4) + (qh[ 1] & hm ? 16 : 0)) - m2;
#अन्यथा
    GGML_UNUSED(ib); GGML_UNUSED(tid); GGML_UNUSED(yy); GGML_UNUSED(scales_local); GGML_UNUSED(n_blocks);
    GGML_ABORT("Q5_K reorder dequantize not supported for QK_K != 256");
#endif
}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q6_K(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy,
                                  स्थिरांक sycl::nd_item<3> &item_ct1) {
    स्थिरांक block_q6_K * x = (स्थिरांक block_q6_K *) vx;

    स्थिरांक int64_t i = item_ct1.get_group(2);
#जर QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
    स्थिरांक int64_t ip  = tid/32;   // ip is 0 or 1
    स्थिरांक int64_t il  = tid - 32*ip; // 0...32
    स्थिरांक int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    स्थिरांक दशांश d = x[i].d;

    स्थिरांक uint8_t * ql = x[i].ql + 64*ip + il;
    स्थिरांक uint8_t   qh = x[i].qh[32*ip + il];
    स्थिरांक int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#अन्यथा

    // assume 32 threads
    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
    स्थिरांक int64_t ip  = tid/16;         // 0 or 1
    स्थिरांक int64_t il  = tid - 16*ip;    // 0...15

    dst_t * y = yy + i*QK_K + 16*ip + il;

    स्थिरांक दशांश d = x[i].d;

    स्थिरांक uint8_t   ql = x[i].ql[16*ip + il];
    स्थिरांक uint8_t   qh = x[i].qh[il] >> (2*ip);
    स्थिरांक int8_t  * sc = x[i].scales;

    y[ 0] = d * sc[ip+0] * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[ip+2] * ((int8_t)((ql  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
#endif
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_block_q6_K_reorder(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, स्थिरांक int64_t ib,
                                          स्थिरांक sycl::nd_item<3> & item_ct1, int64_t n_blocks) {

    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
    स्थिरांक int64_t ip  = tid / 32;       // ip is 0 or 1
    स्थिरांक int64_t il  = tid - 32 * ip;  // 0...32
    स्थिरांक int64_t is  = 8 * ip + il / 16;

    स्थिरांक uint8_t *   base_ptr           = static_cast<स्थिरांक uint8_t *>(vx);
    स्थिरांक स्वतः        ql_offset          = ib * (QK_K / 2);
    स्थिरांक स्वतः        qh_offset          = (QK_K / 2) * n_blocks + (QK_K / 4) * ib;
    स्थिरांक स्वतः        base_scales_offset = (QK_K / 2) * n_blocks + (QK_K / 4) * n_blocks + (QK_K / 16) * ib;
    स्थिरांक स्वतः        base_d_offset      = ((QK_K / 2) + (QK_K / 4) + (QK_K / 16)) * n_blocks;
    स्थिरांक uint8_t *   ql_ptr             = base_ptr + ql_offset;
    स्थिरांक uint8_t *   qh_ptr             = base_ptr + qh_offset;
    स्थिरांक uint8_t *   scales_ptr         = base_ptr + base_scales_offset;
    स्थिरांक ggml_half * d                  = (स्थिरांक ggml_half *) (base_ptr + base_d_offset) + ib;

    dst_t * y = yy + 128 * ip + il;

    स्थिरांक uint8_t * ql = ql_ptr + 64 * ip + il;
    स्थिरांक uint8_t   qh = *(qh_ptr + 32 * ip + il);
    स्थिरांक int8_t *  sc = reinterpret_cast<स्थिरांक int8_t *>(scales_ptr + is);

    y[0]  = *d * sc[0] * ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = *d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = *d * sc[4] * ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = *d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_iq2_xxs(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy,
                                     स्थिरांक sycl::nd_item<3> &item_ct1,
                                     स्थिरांक uint64_t *iq2xxs_grid_ptr,
                                     स्थिरांक uint8_t *ksigns_iq2xs_ptr,
                                     स्थिरांक uint8_t *kmask_iq2xs_ptr) {

    स्थिरांक int64_t i = item_ct1.get_group(2);
    स्थिरांक block_iq2_xxs * x = (स्थिरांक block_iq2_xxs  *) vx;

    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
#जर QK_K == 256
    स्थिरांक int64_t il = tid/8; // 0...3
    स्थिरांक int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    स्थिरांक uint16_t * q2 = x[i].qs + 4*ib;
    स्थिरांक uint8_t  * aux8 = (स्थिरांक uint8_t *)q2;
    स्थिरांक uint8_t  * grid = (स्थिरांक uint8_t *)(iq2xxs_grid_ptr + aux8[il]);
    स्थिरांक uint32_t aux32 = q2[2] | (q2[3] << 16);
    स्थिरांक दशांश d = (दशांश)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    स्थिरांक uint8_t signs = ksigns_iq2xs_ptr[(aux32 >> 7*il) & 127];
    साठी (पूर्णांक j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs_ptr[j] ? -1.f : 1.f);
#अन्यथा
    प्रतिज्ञा(असत्य);
#endif

}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_iq2_xs(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy,
                                    स्थिरांक sycl::nd_item<3> &item_ct1,
                                    स्थिरांक uint64_t *iq2xs_grid,
                                    स्थिरांक uint8_t *ksigns_iq2xs,
                                    स्थिरांक uint8_t *kmask_iq2xs) {

    स्थिरांक int64_t i = item_ct1.get_group(2);
    स्थिरांक block_iq2_xs * x = (स्थिरांक block_iq2_xs *) vx;

    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
#जर QK_K == 256
    स्थिरांक int64_t il = tid/8; // 0...3
    स्थिरांक int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    स्थिरांक uint16_t * q2 = x[i].qs + 4*ib;
    स्थिरांक uint8_t  * grid = (स्थिरांक uint8_t *)(iq2xs_grid + (q2[il] & 511));
    स्थिरांक दशांश d = (दशांश)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    स्थिरांक uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    साठी (पूर्णांक j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#अन्यथा
    प्रतिज्ञा(असत्य);
#endif

}

ढाँचा <typename dst_t>
__dpct_inline__ स्थिर शून्य
dequantize_block_iq2_s(स्थिरांक शून्य *__restrict__ vx, dst_t *__restrict__ yy,
                       स्थिरांक sycl::nd_item<3> &item_ct1) {

    स्थिरांक int64_t i = item_ct1.get_group(2);
    स्थिरांक block_iq2_s * x = (स्थिरांक block_iq2_s *) vx;

    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
#जर QK_K == 256
    स्थिरांक int64_t il = tid/8; // 0...3
    स्थिरांक int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    स्थिरांक uint8_t * grid = (स्थिरांक uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    स्थिरांक दशांश d = (दशांश)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    स्थिरांक uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
#pragma unroll
    साठी (पूर्णांक j = 0; j < 8; ++j)
        y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#अन्यथा
    प्रतिज्ञा(असत्य);

#endif

}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_iq3_xxs(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy,
                                     स्थिरांक sycl::nd_item<3> &item_ct1,
                                     स्थिरांक uint32_t *iq3xxs_grid,
                                     स्थिरांक uint8_t *ksigns_iq2xs,
                                     स्थिरांक uint8_t *kmask_iq2xs) {

    स्थिरांक int64_t i = item_ct1.get_group(2);
    स्थिरांक block_iq3_xxs * x = (स्थिरांक block_iq3_xxs  *) vx;

    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
#जर QK_K == 256
    स्थिरांक int64_t il = tid/8; // 0...3
    स्थिरांक int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    स्थिरांक uint8_t  * q3 = x[i].qs + 8*ib;
    स्थिरांक uint16_t * gas = (स्थिरांक uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    स्थिरांक uint8_t  * grid1 = (स्थिरांक uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    स्थिरांक uint8_t  * grid2 = (स्थिरांक uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    स्थिरांक uint32_t aux32 = gas[0] | (gas[1] << 16);
    स्थिरांक दशांश d = (दशांश)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    स्थिरांक uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    साठी (पूर्णांक j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#अन्यथा
    प्रतिज्ञा(असत्य);
#endif

}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_iq3_xxs_reorder(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, स्थिरांक int64_t i,
                                             स्थिरांक sycl::nd_item<3> & item_ct1, स्थिरांक int64_t nb,
                                             स्थिरांक uint32_t * iq3xxs_grid,
                                             स्थिरांक uint8_t * ksigns_iq2xs,
                                             स्थिरांक uint8_t * kmask_iq2xs) {
    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
#जर QK_K == 256
    स्थिरांक int64_t il = tid / 8;
    स्थिरांक int64_t ib = tid % 8;
    स्थिरांक uint8_t * base = static_cast<स्थिरांक uint8_t *>(vx);
    स्थिरांक uint8_t * qs = base + i * (3 * QK_K / 8);
    स्थिरांक ggml_half * d = reinterpret_cast<स्थिरांक ggml_half *>(base + nb * (3 * QK_K / 8));
    स्थिरांक uint8_t * q3 = qs + 8 * ib;
    स्थिरांक uint16_t * gas = reinterpret_cast<स्थिरांक uint16_t *>(qs + QK_K / 4) + 2 * ib;
    स्थिरांक uint8_t * grid1 = reinterpret_cast<स्थिरांक uint8_t *>(iq3xxs_grid + q3[2*il+0]);
    स्थिरांक uint8_t * grid2 = reinterpret_cast<स्थिरांक uint8_t *>(iq3xxs_grid + q3[2*il+1]);
    स्थिरांक uint32_t aux32 = gas[0] | (gas[1] << 16);
    स्थिरांक दशांश dl = (दशांश) d[i] * (0.5f + (aux32 >> 28)) * 0.5f;
    स्थिरांक uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
    dst_t * y = yy + 32 * ib + 8 * il;
#pragma unroll
    साठी (पूर्णांक j = 0; j < 4; ++j) {
        y[j+0] = dl * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = dl * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#अन्यथा
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(i);
    GGML_UNUSED(item_ct1);
    GGML_UNUSED(nb);
    GGML_UNUSED(iq3xxs_grid);
    GGML_UNUSED(ksigns_iq2xs);
    GGML_UNUSED(kmask_iq2xs);
    GGML_ABORT("IQ3_XXS reorder dequantization requires QK_K == 256");
#endif
}

ढाँचा <typename dst_t>
__dpct_inline__ स्थिर शून्य
dequantize_block_iq3_s(स्थिरांक शून्य *__restrict__ vx, dst_t *__restrict__ yy,
                       स्थिरांक sycl::nd_item<3> &item_ct1,
                       स्थिरांक uint8_t *kmask_iq2xs, स्थिरांक uint32_t *iq3s_grid) {

    स्थिरांक int64_t i = item_ct1.get_group(2);
    स्थिरांक block_iq3_s * x = (स्थिरांक block_iq3_s *) vx;

    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
#जर QK_K == 256
    स्थिरांक int64_t il = tid/8; // 0...3
    स्थिरांक int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    स्थिरांक uint8_t * qs = x[i].qs + 8*ib;
    स्थिरांक uint8_t * grid1 = (स्थिरांक uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    स्थिरांक uint8_t * grid2 = (स्थिरांक uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    स्थिरांक दशांश d = (दशांश)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    स्थिरांक uint8_t signs = x[i].signs[4*ib + il];
#pragma unroll
    साठी (पूर्णांक j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#अन्यथा
    प्रतिज्ञा(असत्य);
#endif

}

ढाँचा <typename dst_t>
__dpct_inline__ स्थिर शून्य
dequantize_block_iq3_s_reorder(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, स्थिरांक int64_t i,
                               स्थिरांक sycl::nd_item<3> & item_ct1, स्थिरांक int64_t nb,
                               स्थिरांक uint8_t * kmask_iq2xs, स्थिरांक uint32_t * iq3s_grid) {
    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
#जर QK_K == 256
    स्थिरांक int64_t il = tid / 8;
    स्थिरांक int64_t ib = tid % 8;
    स्थिरांक uint8_t * base = static_cast<स्थिरांक uint8_t *>(vx);
    स्थिरांक uint8_t * qs = base + i * (QK_K / 4);
    स्थिरांक uint8_t * qh = base + nb * (QK_K / 4) + i * (QK_K / 32);
    स्थिरांक uint8_t * signs_base = base + nb * (QK_K / 4 + QK_K / 32) + i * (QK_K / 8);
    स्थिरांक uint8_t * metadata = base + nb * (QK_K / 4 + QK_K / 32 + QK_K / 8) +
                               i * (आकार(ggml_half) + IQ3S_N_SCALE);
    स्थिरांक ggml_half d = *reinterpret_cast<स्थिरांक ggml_half *>(metadata);
    स्थिरांक uint8_t * scales = metadata + आकार(ggml_half);
    स्थिरांक uint8_t * qs_ib = qs + 8 * ib;
    स्थिरांक uint8_t * grid1 = reinterpret_cast<स्थिरांक uint8_t *>(
        iq3s_grid + (qs_ib[2*il+0] | ((qh[ib] << (8 - 2*il)) & 256)));
    स्थिरांक uint8_t * grid2 = reinterpret_cast<स्थिरांक uint8_t *>(
        iq3s_grid + (qs_ib[2*il+1] | ((qh[ib] << (7 - 2*il)) & 256)));
    स्थिरांक दशांश dl = (दशांश) d * (1 + 2 * ((scales[ib/2] >> (4 * (ib%2))) & 0xf));
    स्थिरांक uint8_t signs = signs_base[4 * ib + il];
    dst_t * y = yy + 32 * ib + 8 * il;
#pragma unroll
    साठी (पूर्णांक j = 0; j < 4; ++j) {
        y[j+0] = dl * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = dl * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#अन्यथा
    GGML_UNUSED(vx);
    GGML_UNUSED(yy);
    GGML_UNUSED(i);
    GGML_UNUSED(item_ct1);
    GGML_UNUSED(nb);
    GGML_UNUSED(kmask_iq2xs);
    GGML_UNUSED(iq3s_grid);
    GGML_ABORT("IQ3_S reorder dequantization requires QK_K == 256");
#endif
}

ढाँचा <typename dst_t>
__dpct_inline__ स्थिर शून्य
dequantize_block_iq1_s(स्थिरांक शून्य *__restrict__ vx, dst_t *__restrict__ yy,
                       स्थिरांक sycl::nd_item<3> &item_ct1,
                       स्थिरांक uint32_t *iq1s_grid_gpu) {

    स्थिरांक int64_t i = item_ct1.get_group(2);
    स्थिरांक block_iq1_s * x = (स्थिरांक block_iq1_s  *) vx;

    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
#जर QK_K == 256
    स्थिरांक int64_t il = tid/8; // 0...3
    स्थिरांक int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    स्थिरांक दशांश delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    स्थिरांक दशांश d = (दशांश)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; स्थिरांक int8_t * q = (स्थिरांक int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#pragma unroll
    साठी (पूर्णांक j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#अन्यथा
    प्रतिज्ञा(असत्य);
#endif

}

ढाँचा <typename dst_t>
__dpct_inline__ स्थिर शून्य
dequantize_block_iq1_m(स्थिरांक शून्य *__restrict__ vx, dst_t *__restrict__ yy,
                       स्थिरांक sycl::nd_item<3> &item_ct1,
                       स्थिरांक uint32_t *iq1s_grid_gpu) {

    स्थिरांक int64_t i = item_ct1.get_group(2);
    स्थिरांक block_iq1_m * x = (स्थिरांक block_iq1_m  *) vx;

    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
#जर QK_K == 256
    स्थिरांक int64_t il = tid/8; // 0...3
    स्थिरांक int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    स्थिरांक uint16_t * sc = (स्थिरांक uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    स्थिरांक पूर्णांक ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    स्थिरांक दशांश d = (दशांश)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    स्थिरांक दशांश delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; स्थिरांक int8_t * q = (स्थिरांक int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#pragma unroll
    साठी (पूर्णांक j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#अन्यथा
    प्रतिज्ञा(असत्य);
#endif

}

ढाँचा <typename dst_t>
__dpct_inline__ स्थिर शून्य
dequantize_block_iq4_nl(स्थिरांक शून्य *__restrict__ vx, dst_t *__restrict__ yy,
                        स्थिरांक sycl::nd_item<3> &item_ct1) {

    स्थिरांक int64_t i = item_ct1.get_group(2);
    स्थिरांक block_iq4_nl * x = (स्थिरांक block_iq4_nl *) vx + i*(QK_K/QK4_NL);

    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
    स्थिरांक int64_t il = tid/8; // 0...3
    स्थिरांक int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    स्थिरांक uint8_t  * q4 = x[ib].qs + 4*il;
    स्थिरांक दशांश d = (दशांश)x[ib].d;
#pragma unroll
    साठी (पूर्णांक j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }

}


ढाँचा <typename dst_t>
__dpct_inline__ स्थिर शून्य
dequantize_block_iq4_xs(स्थिरांक शून्य *__restrict__ vx, dst_t *__restrict__ yy,
                        स्थिरांक sycl::nd_item<3> &item_ct1) {
    स्थिरांक int64_t i = item_ct1.get_group(2);
    स्थिरांक block_iq4_xs * x = (स्थिरांक block_iq4_xs *)vx;

    स्थिरांक int64_t tid = item_ct1.get_local_id(2);
    स्थिरांक int64_t il = tid/8; // 0...3
    स्थिरांक int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    स्थिरांक uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    स्थिरांक दशांश d = (दशांश)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
#pragma unroll
    साठी (पूर्णांक j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_mxfp4(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy,
                                   स्थिरांक sycl::nd_item<3> &item_ct1) {
    // auto                item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिरांक int64_t       i        = item_ct1.get_group(2);
    स्थिरांक block_mxfp4 * x = (स्थिरांक block_mxfp4 *) vx + i*(QK_K/QK_MXFP4);

    स्थिरांक int64_t    tid = item_ct1.get_local_id(2);
    स्थिरांक int64_t il = tid/8; // 0...3
    स्थिरांक int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    स्थिरांक uint8_t  * q4 = x[ib].qs + 4*il;
    स्थिरांक दशांश d = ggml_sycl_e8m0_to_fp32(x[ib].e);
    साठी (पूर्णांक j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_mxfp4[q4[j] & 0xf]*0.5f;
        y[j+16] = d * kvalues_mxfp4[q4[j] >>  4]*0.5f;
    }
}

// Reordered MXFP4 ([qs...][e...], see ggml_sycl_reordered::block_q_t<MXFP4>): one work-item per block.
ढाँचा <typename dst_t>
स्थिर शून्य dequantize_block_mxfp4_reorder(स्थिरांक शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                           स्थिरांक sycl::nd_item<3> & item_ct1) {
    स्थिरांक int64_t ib = (int64_t) item_ct1.get_group(2) * WARP_SIZE + item_ct1.get_local_id(2);
    जर (ib >= k / QK_MXFP4) {
        परत;
    }

    स्थिरांक uint8_t * qs = (स्थिरांक uint8_t *) vx + ib * (QK_MXFP4 / 2);
    स्थिरांक दशांश     d  = ggml_sycl_e8m0_to_fp32(((स्थिरांक uint8_t *) vx)[k / 2 + ib]) * 0.5f;
    dst_t *         y  = yy + ib * QK_MXFP4;

#pragma unroll
    साठी (पूर्णांक j = 0; j < QK_MXFP4 / 2; ++j) {
        y[j]                = d * kvalues_mxfp4[qs[j] & 0xf];
        y[j + QK_MXFP4 / 2] = d * kvalues_mxfp4[qs[j] >> 4];
    }
}


ढाँचा <typename dst_t>
स्थिर शून्य dequantize_block_nvfp4(
        स्थिरांक शून्य * __restrict__ vx,
        dst_t * __restrict__ yy,
        स्थिरांक int64_t ne) {
    स्वतः          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    स्थिरांक int64_t i        = item_ct1.get_group(2);
    स्थिरांक पूर्णांक     tid      = item_ct1.get_local_id(2);

    स्थिरांक int64_t base = i * QK_NVFP4;
    जर (base >= ne) {
        परत;
    }

    स्थिरांक block_nvfp4 * x = (स्थिरांक block_nvfp4 *) vx;
    स्थिरांक block_nvfp4 & xb = x[i];

    स्थिरांक पूर्णांक sub = tid / (QK_NVFP4_SUB / 2);
    स्थिरांक पूर्णांक j = tid % (QK_NVFP4_SUB / 2);

    स्थिरांक दशांश d = ggml_sycl_ue4m3_to_fp32(xb.d[sub]);
    स्थिरांक uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    स्थिरांक int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    स्थिरांक int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_sycl_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_sycl_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}


#endif // GGML_SYCL_DEQUANTIZE_HPP
