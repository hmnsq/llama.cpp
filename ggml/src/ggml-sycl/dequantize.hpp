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

प्रकारनिर्देश शून्य (*dequantize_kernel_t)(अचर शून्य * vx, अचर int64_t ib, अचर पूर्णाङ्क iqs, dfloat2 & v);
प्रकारनिर्देश शून्य (*dequantize_kernel_t_reorder)(अचर शून्य *d, अचर int64_t ib, अचर शून्य *qs,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v);
प्रकारनिर्देश शून्य (*dequantize_kernel_f32_t)(अचर शून्य * vx, अचर int64_t ib, अचर पूर्णाङ्क iqs, भिन्न & v0, भिन्न & v1);

#यदि QK_K == 256
स्थिर अन्तःस्थ शून्य get_scale_min_k4(पूर्णाङ्क j, अचर uint8_t * q, uint8_t & d, uint8_t & m);
#endif

स्थिर __dpct_inline__ शून्य dequantize_q2_0(अचर शून्य *vx, अचर int64_t ib,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
    अचर block_q2_0 * x = (अचर block_q2_0 *) vx;

    अचर dfloat d = x[ib].d;

    अचर पूर्णाङ्क byte_idx = iqs / 4;
    अचर पूर्णाङ्क shift = (iqs % 4) * 2;
    अचर uint8_t vui = x[ib].qs[byte_idx];

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

स्थिर __dpct_inline__ शून्य dequantize_q4_0(अचर शून्य *vx, अचर int64_t ib,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
    अचर block_q4_0 * x = (अचर block_q4_0 *) vx;

    अचर dfloat d = x[ib].d;

    अचर पूर्णाङ्क vui = x[ib].qs[iqs];

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

स्थिर __dpct_inline__ शून्य dequantize_q4_0_reorder(अचर शून्य *d_ptr, अचर int64_t ib, अचर शून्य *qs,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
    // const block_q4_0 * x = (const block_q4_0 *) vx;

    अचर dfloat d = (अचर dfloat)*((अचर sycl::half*)d_ptr+ib);

    अचर पूर्णाङ्क vui = *((अचर uint8_t *)qs+iqs);

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

स्थिर __dpct_inline__ शून्य dequantize_q1_0_reorder(अचर शून्य *d_ptr, अचर int64_t ib, अचर शून्य *qs,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
    // Q1_0 reorder layout: scale values followed by quantized bits
    अचर dfloat d = (अचर dfloat)*((अचर sycl::half*)d_ptr+ib);

    अचर पूर्णाङ्क bit_index_0 = iqs + 0;
    अचर पूर्णाङ्क bit_index_1 = iqs + 1;

    अचर पूर्णाङ्क bit_0 = (*((अचर uint8_t *)qs + bit_index_0 / 8) >> (bit_index_0 % 8)) & 1;
    अचर पूर्णाङ्क bit_1 = (*((अचर uint8_t *)qs + bit_index_1 / 8) >> (bit_index_1 % 8)) & 1;

    v.x() = (2 * bit_0 - 1) * d;
    v.y() = (2 * bit_1 - 1) * d;
}

स्थिर __dpct_inline__ शून्य dequantize_q1_0(अचर शून्य *vx, अचर int64_t ib,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
    अचर block_q1_0 * x = (अचर block_q1_0 *) vx;
    अचर dfloat d = x[ib].d;

    अचर पूर्णाङ्क bit_index_0 = iqs + 0;
    अचर पूर्णाङ्क bit_index_1 = iqs + 1;

    अचर पूर्णाङ्क bit_0 = (x[ib].qs[bit_index_0 / 8] >> (bit_index_0 % 8)) & 1;
    अचर पूर्णाङ्क bit_1 = (x[ib].qs[bit_index_1 / 8] >> (bit_index_1 % 8)) & 1;

    v.x() = (2 * bit_0 - 1) * d;
    v.y() = (2 * bit_1 - 1) * d;
}

स्थिर __dpct_inline__ शून्य dequantize_q4_1(अचर शून्य *vx, अचर int64_t ib,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
    अचर block_q4_1 * x = (अचर block_q4_1 *) vx;

    अचर dfloat d = x[ib].dm[0];
    अचर dfloat m = x[ib].dm[1];

    अचर पूर्णाङ्क vui = x[ib].qs[iqs];

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

स्थिर __dpct_inline__ शून्य dequantize_q4_K(अचर शून्य *vx, अचर int64_t ib,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
#यदि QK_K == 256
    अचर block_q4_K * x = (अचर block_q4_K *) vx;
    अचर sycl::half2 dm = x[ib].dm;
    अचर भिन्न dall = dm[0];
    अचर भिन्न dmin = dm[1];

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        अचर पूर्णाङ्क il = idx / 64;
        अचर पूर्णाङ्क in = idx % 64;
        अचर पूर्णाङ्क is = 2 * il + (in >= 32 ? 1 : 0);
        अचर पूर्णाङ्क off = in & 31;
        अचर पूर्णाङ्क qsi = 32 * il + off;

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        अचर uint8_t q = x[ib].qs[qsi];
        अचर uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);
        फर्काऊ sycl::fma((dfloat) qv, (dfloat) (dall * sc), (dfloat) (-dmin * m));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q4_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q4_K_f32(अचर शून्य *vx, अचर int64_t ib,
                                                अचर पूर्णाङ्क iqs, भिन्न &v0, भिन्न &v1) {
#यदि QK_K == 256
    अचर block_q4_K * x = (अचर block_q4_K *) vx;
    अचर sycl::half2 dm = x[ib].dm;
    अचर भिन्न dall = dm[0];
    अचर भिन्न dmin = dm[1];

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> भिन्न {
        अचर पूर्णाङ्क il = idx / 64;
        अचर पूर्णाङ्क in = idx % 64;
        अचर पूर्णाङ्क is = 2 * il + (in >= 32 ? 1 : 0);
        अचर पूर्णाङ्क qsi = 32 * il + (in & 31);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        अचर भिन्न d = dall * sc;
        अचर भिन्न mn = dmin * m;
        अचर uint8_t q = x[ib].qs[qsi];
        अचर uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);

        फर्काऊ d * qv - mn;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q4_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q2_K(अचर शून्य *vx, अचर int64_t ib,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
#यदि QK_K == 256
    अचर block_q2_K * x = (अचर block_q2_K *) vx;
    अचर भिन्न dall = x[ib].dm[0];
    अचर भिन्न dmin = x[ib].dm[1];

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        अचर पूर्णाङ्क n = idx / 128;
        अचर पूर्णाङ्क r = idx % 128;
        अचर पूर्णाङ्क g = r / 32;
        अचर पूर्णाङ्क l = r % 32;
        अचर पूर्णाङ्क is = 8 * n + l / 16;

        अचर uint8_t q = x[ib].qs[32 * n + l];
        अचर uint8_t sc = x[ib].scales[is + 2 * g];
        अचर भिन्न d = dall * (sc & 0xF);
        अचर भिन्न m = dmin * (sc >> 4);

        फर्काऊ (dfloat) d * (dfloat) ((q >> (2 * g)) & 3) - (dfloat) m;
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q2_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q2_K_f32(अचर शून्य *vx, अचर int64_t ib,
                                                अचर पूर्णाङ्क iqs, भिन्न &v0, भिन्न &v1) {
#यदि QK_K == 256
    अचर block_q2_K * x = (अचर block_q2_K *) vx;
    अचर भिन्न dall = x[ib].dm[0];
    अचर भिन्न dmin = x[ib].dm[1];

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> भिन्न {
        अचर पूर्णाङ्क n = idx / 128;
        अचर पूर्णाङ्क r = idx % 128;
        अचर पूर्णाङ्क g = r / 32;
        अचर पूर्णाङ्क l = r % 32;
        अचर पूर्णाङ्क is = 8 * n + l / 16;

        अचर uint8_t q = x[ib].qs[32 * n + l];
        अचर uint8_t sc = x[ib].scales[is + 2 * g];
        अचर भिन्न d = dall * (sc & 0xF);
        अचर भिन्न m = dmin * (sc >> 4);

        फर्काऊ d * ((q >> (2 * g)) & 3) - m;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q2_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q3_K(अचर शून्य *vx, अचर int64_t ib,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
#यदि QK_K == 256
    अचर block_q3_K * x = (अचर block_q3_K *) vx;
    अचर भिन्न d_all = x[ib].d;

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        अचर पूर्णाङ्क n = idx / 128;
        अचर पूर्णाङ्क r = idx % 128;
        अचर पूर्णाङ्क j = r / 32;
        अचर पूर्णाङ्क l = r % 32;

        अचर पूर्णाङ्क is0 = l / 16;
        अचर पूर्णाङ्क is = 8 * n + 2 * j + is0;
        अचर पूर्णाङ्क shift = 2 * j;
        अचर uint8_t m = 1 << (4 * n + j);

        अचर int8_t us = is <  4 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 8] >> 0) & 3) << 4) :
                         is <  8 ? (x[ib].scales[is - 0] & 0xF) | (((x[ib].scales[is + 4] >> 2) & 3) << 4) :
                         is < 12 ? (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is + 0] >> 4) & 3) << 4) :
                                   (x[ib].scales[is - 8] >> 4)  | (((x[ib].scales[is - 4] >> 6) & 3) << 4);

        अचर भिन्न dl = d_all * (us - 32);
        अचर uint8_t q = x[ib].qs[32 * n + l];
        अचर uint8_t h = x[ib].hmask[l];
        अचर int8_t qv = ((q >> shift) & 3) - ((h & m) ? 0 : 4);

        फर्काऊ (dfloat) (dl * qv);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q3_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q5_K(अचर शून्य *vx, अचर int64_t ib,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
#यदि QK_K == 256
    अचर block_q5_K * x = (अचर block_q5_K *) vx;
    अचर भिन्न dall = x[ib].dm[0];
    अचर भिन्न dmin = x[ib].dm[1];

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        अचर पूर्णाङ्क il = idx / 64;
        अचर पूर्णाङ्क in = idx % 64;
        अचर पूर्णाङ्क is = 2 * il + (in >= 32 ? 1 : 0);
        अचर पूर्णाङ्क ir = (in & 31) / 2;
        अचर पूर्णाङ्क iq = in & 1;

        अचर uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        अचर uint8_t h = x[ib].qh[2 * ir + iq];
        अचर uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        अचर भिन्न d = dall * sc;
        अचर भिन्न mn = dmin * m;
        अचर uint8_t hm = 1 << (2 * il + (in >= 32 ? 1 : 0));

        फर्काऊ sycl::fma((dfloat) (qv + ((h & hm) ? 16 : 0)), (dfloat) d, (dfloat) (-mn));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q5_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q5_K_f32(अचर शून्य *vx, अचर int64_t ib,
                                                अचर पूर्णाङ्क iqs, भिन्न &v0, भिन्न &v1) {
#यदि QK_K == 256
    अचर block_q5_K * x = (अचर block_q5_K *) vx;
    अचर भिन्न dall = x[ib].dm[0];
    अचर भिन्न dmin = x[ib].dm[1];

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> भिन्न {
        अचर पूर्णाङ्क il = idx / 64;
        अचर पूर्णाङ्क in = idx % 64;
        अचर पूर्णाङ्क is = 2 * il + (in >= 32 ? 1 : 0);
        अचर पूर्णाङ्क ir = (in & 31) / 2;
        अचर पूर्णाङ्क iq = in & 1;

        अचर uint8_t q = x[ib].qs[32 * il + 2 * ir + iq];
        अचर uint8_t h = x[ib].qh[2 * ir + iq];
        अचर uint8_t qv = (in >= 32) ? (q >> 4) : (q & 0xF);

        uint8_t sc;
        uint8_t m;
        get_scale_min_k4(is, x[ib].scales, sc, m);

        अचर भिन्न d = dall * sc;
        अचर भिन्न mn = dmin * m;
        अचर uint8_t hm = 1 << (2 * il + (in >= 32 ? 1 : 0));

        फर्काऊ (qv + ((h & hm) ? 16 : 0)) * d - mn;
    };

    v0 = dequantize_one(iqs + 0);
    v1 = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q5_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q6_K(अचर शून्य *vx, अचर int64_t ib,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
#यदि QK_K == 256
    अचर block_q6_K * x = (अचर block_q6_K *) vx;
    अचर भिन्न d = x[ib].d;

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        अचर पूर्णाङ्क ip = idx / 128;
        अचर पूर्णाङ्क in = idx % 128;
        अचर पूर्णाङ्क il = in & 31;
        अचर पूर्णाङ्क ig = in / 32;
        अचर पूर्णाङ्क is = 8 * ip + il / 16;

        अचर uint8_t ql0 = x[ib].ql[64 * ip + il];
        अचर uint8_t ql1 = x[ib].ql[64 * ip + il + 32];
        अचर uint8_t qh = x[ib].qh[32 * ip + il];
        अचर int8_t * sc = x[ib].scales + is;

        uint8_t qv;
        int8_t scale;
        यदि (ig == 0) {
            qv = (ql0 & 0xF) | (((qh >> 0) & 3) << 4);
            scale = sc[0];
        } अन्यथा यदि (ig == 1) {
            qv = (ql1 & 0xF) | (((qh >> 2) & 3) << 4);
            scale = sc[2];
        } अन्यथा यदि (ig == 2) {
            qv = (ql0 >> 4) | (((qh >> 4) & 3) << 4);
            scale = sc[4];
        } अन्यथा {
            qv = (ql1 >> 4) | (((qh >> 6) & 3) << 4);
            scale = sc[6];
        }

        फर्काऊ (dfloat) (d * scale * ((int8_t) qv - 32));
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("Q6_K dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_mxfp4(अचर शून्य *vx, अचर int64_t ib,
                                             अचर पूर्णाङ्क iqs, dfloat2 &v) {
    अचर block_mxfp4 * x = (अचर block_mxfp4 *) vx;
    अचर भिन्न d = ggml_sycl_e8m0_to_fp32(x[ib].e);
    अचर uint8_t q = x[ib].qs[iqs];

    v.x() = d * kvalues_mxfp4[q & 0xF] * 0.5f;
    v.y() = d * kvalues_mxfp4[q >> 4] * 0.5f;
}

स्थिर __dpct_inline__ शून्य dequantize_nvfp4(अचर शून्य *vx, अचर int64_t ib,
                                             अचर पूर्णाङ्क iqs, dfloat2 &v) {
    अचर block_nvfp4 & xb = ((अचर block_nvfp4 *) vx)[ib];

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        अचर पूर्णाङ्क sub = idx / QK_NVFP4_SUB;
        अचर पूर्णाङ्क j = idx % QK_NVFP4_SUB;
        अचर पूर्णाङ्क jh = j % (QK_NVFP4_SUB / 2);

        अचर भिन्न d = ggml_sycl_ue4m3_to_fp32(xb.d[sub]);
        अचर uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + jh];
        अचर uint8_t qv = (j < (QK_NVFP4_SUB / 2)) ? (q & 0x0F) : (q >> 4);

        फर्काऊ d * kvalues_mxfp4[qv];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
}

स्थिर __dpct_inline__ शून्य dequantize_iq2_xxs(अचर शून्य *vx, अचर int64_t ib,
                                               अचर पूर्णाङ्क iqs, dfloat2 &v) {
#यदि QK_K == 256
    अचर block_iq2_xxs * x = (अचर block_iq2_xxs *) vx;

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        अचर पूर्णाङ्क ib8 = idx / 32;
        अचर पूर्णाङ्क r = idx % 32;
        अचर पूर्णाङ्क il = r / 8;
        अचर पूर्णाङ्क j = r % 8;

        अचर uint16_t * q2 = x[ib].qs + 4 * ib8;
        अचर uint8_t * aux8 = (अचर uint8_t *) q2;
        अचर uint8_t * grid = (अचर uint8_t *) (iq2xxs_grid + aux8[il]);
        अचर uint32_t aux32 = q2[2] | (q2[3] << 16);
        अचर भिन्न d = (भिन्न) x[ib].d * (0.5f + (aux32 >> 28)) * 0.25f;
        अचर uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        फर्काऊ d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ2_XXS dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_iq2_xs(अचर शून्य *vx, अचर int64_t ib,
                                              अचर पूर्णाङ्क iqs, dfloat2 &v) {
#यदि QK_K == 256
    अचर block_iq2_xs * x = (अचर block_iq2_xs *) vx;

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        अचर पूर्णाङ्क ib8 = idx / 32;
        अचर पूर्णाङ्क r = idx % 32;
        अचर पूर्णाङ्क il = r / 8;
        अचर पूर्णाङ्क j = r % 8;

        अचर uint16_t * q2 = x[ib].qs + 4 * ib8;
        अचर uint8_t * grid = (अचर uint8_t *) (iq2xs_grid + (q2[il] & 511));
        अचर भिन्न d = (भिन्न) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        अचर uint8_t signs = ksigns_iq2xs[q2[il] >> 9];

        फर्काऊ d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ2_XS dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_iq2_s(अचर शून्य *vx, अचर int64_t ib,
                                             अचर पूर्णाङ्क iqs, dfloat2 &v) {
#यदि QK_K == 256
    अचर block_iq2_s * x = (अचर block_iq2_s *) vx;

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        अचर पूर्णाङ्क ib8 = idx / 32;
        अचर पूर्णाङ्क r = idx % 32;
        अचर पूर्णाङ्क il = r / 8;
        अचर पूर्णाङ्क j = r % 8;

        अचर uint16_t grid_id = x[ib].qs[4 * ib8 + il] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 0x300);
        अचर uint8_t * grid = (अचर uint8_t *) (iq2s_grid + grid_id);
        अचर भिन्न d = (भिन्न) x[ib].d * (0.5f + ((x[ib].scales[ib8] >> (4 * (il / 2))) & 0xf)) * 0.25f;
        अचर uint8_t signs = x[ib].qs[QK_K / 8 + 4 * ib8 + il];

        फर्काऊ d * grid[j] * ((signs & kmask_iq2xs[j]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ2_S dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_iq3_xxs(अचर शून्य *vx, अचर int64_t ib,
                                               अचर पूर्णाङ्क iqs, dfloat2 &v) {
#यदि QK_K == 256
    अचर block_iq3_xxs * x = (अचर block_iq3_xxs *) vx;

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        अचर पूर्णाङ्क ib8 = idx / 32;
        अचर पूर्णाङ्क r = idx % 32;
        अचर पूर्णाङ्क il = r / 8;
        अचर पूर्णाङ्क j = r % 8;

        अचर uint8_t * q3 = x[ib].qs + 8 * ib8;
        अचर uint16_t * gas = (अचर uint16_t *) (x[ib].qs + QK_K / 4) + 2 * ib8;
        अचर uint8_t * grid1 = (अचर uint8_t *) (iq3xxs_grid + q3[2 * il + 0]);
        अचर uint8_t * grid2 = (अचर uint8_t *) (iq3xxs_grid + q3[2 * il + 1]);
        अचर uint32_t aux32 = gas[0] | (gas[1] << 16);
        अचर भिन्न d = (भिन्न) x[ib].d * (0.5f + (aux32 >> 28)) * 0.5f;
        अचर uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];

        यदि (j < 4) {
            फर्काऊ d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        फर्काऊ d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ3_XXS dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_iq3_s(अचर शून्य *vx, अचर int64_t ib,
                                             अचर पूर्णाङ्क iqs, dfloat2 &v) {
#यदि QK_K == 256
    अचर block_iq3_s * x = (अचर block_iq3_s *) vx;

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        अचर पूर्णाङ्क ib8 = idx / 32;
        अचर पूर्णाङ्क r = idx % 32;
        अचर पूर्णाङ्क il = r / 8;
        अचर पूर्णाङ्क j = r % 8;

        अचर uint8_t * qs = x[ib].qs + 8 * ib8;
        अचर uint16_t grid1_id = qs[2 * il + 0] | ((x[ib].qh[ib8] << (8 - 2 * il)) & 256);
        अचर uint16_t grid2_id = qs[2 * il + 1] | ((x[ib].qh[ib8] << (7 - 2 * il)) & 256);
        अचर uint8_t * grid1 = (अचर uint8_t *) (iq3s_grid + grid1_id);
        अचर uint8_t * grid2 = (अचर uint8_t *) (iq3s_grid + grid2_id);
        अचर भिन्न d = (भिन्न) x[ib].d * (1 + 2 * ((x[ib].scales[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf));
        अचर uint8_t signs = x[ib].signs[4 * ib8 + il];

        यदि (j < 4) {
            फर्काऊ d * grid1[j] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
        }
        फर्काऊ d * grid2[j - 4] * ((signs & kmask_iq2xs[j + 0]) ? -1.f : 1.f);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ3_S dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_iq1_s(अचर शून्य *vx, अचर int64_t ib,
                                             अचर पूर्णाङ्क iqs, dfloat2 &v) {
#यदि QK_K == 256
    अचर block_iq1_s * x = (अचर block_iq1_s *) vx;

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        अचर पूर्णाङ्क ib8 = idx / 32;
        अचर पूर्णाङ्क r = idx % 32;
        अचर पूर्णाङ्क il = r / 8;
        अचर पूर्णाङ्क j = r % 8;

        अचर भिन्न delta = (x[ib].qh[ib8] & 0x8000) ? (-1.f - IQ1S_DELTA) : (-1.f + IQ1S_DELTA);
        अचर भिन्न d = (भिन्न) x[ib].d * (2 * ((x[ib].qh[ib8] >> 12) & 7) + 1);
        अचर uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((x[ib].qh[ib8] >> (3 * il)) & 7) << 8);
        अचर uint32_t g = iq1s_grid_gpu[grid_id];
        अचर int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        फर्काऊ d * (qv + delta);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ1_S dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_iq1_m(अचर शून्य *vx, अचर int64_t ib,
                                             अचर पूर्णाङ्क iqs, dfloat2 &v) {
#यदि QK_K == 256
    अचर block_iq1_m * x = (अचर block_iq1_m *) vx;

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        अचर पूर्णाङ्क ib8 = idx / 32;
        अचर पूर्णाङ्क r = idx % 32;
        अचर पूर्णाङ्क il = r / 8;
        अचर पूर्णाङ्क j = r % 8;

        अचर uint16_t * sc = (अचर uint16_t *) x[ib].scales;
        iq1m_scale_t scale;
        scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);

        अचर पूर्णाङ्क ib16 = 2 * ib8 + il / 2;
        अचर भिन्न d = (भिन्न) scale.f16 * (2 * ((sc[ib16 / 4] >> (3 * (ib16 % 4))) & 0x7) + 1);

        अचर uint8_t qh = x[ib].qh[2 * ib8 + il / 2];
        अचर भिन्न delta = (qh & (0x08 << (4 * (il % 2)))) ? (-1.f - IQ1M_DELTA) : (-1.f + IQ1M_DELTA);

        अचर uint16_t grid_id = x[ib].qs[4 * ib8 + il] | (((qh >> (4 * (il % 2))) & 7) << 8);
        अचर uint32_t g = iq1s_grid_gpu[grid_id];
        अचर int8_t qv = (j < 4) ? ((g >> (8 * j)) & 0x0F) : ((g >> (8 * (j - 4) + 4)) & 0x0F);

        फर्काऊ d * (qv + delta);
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ1_M dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_iq4_nl(अचर शून्य *vx, अचर int64_t ib,
                                              अचर पूर्णाङ्क iqs, dfloat2 &v) {
    अचर block_iq4_nl * x = (अचर block_iq4_nl *) vx;
    अचर भिन्न d = (भिन्न) x[ib].d;

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        यदि (idx < 16) {
            फर्काऊ d * kvalues_iq4nl[x[ib].qs[idx] & 0xF];
        }
        फर्काऊ d * kvalues_iq4nl[x[ib].qs[idx - 16] >> 4];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
}

स्थिर __dpct_inline__ शून्य dequantize_iq4_xs(अचर शून्य *vx, अचर int64_t ib,
                                              अचर पूर्णाङ्क iqs, dfloat2 &v) {
#यदि QK_K == 256
    अचर block_iq4_xs * x = (अचर block_iq4_xs *) vx;

    स्वतः dequantize_one = [&](अचर पूर्णाङ्क idx) -> dfloat {
        अचर पूर्णाङ्क ib8 = idx / 32;
        अचर पूर्णाङ्क r = idx % 32;
        अचर पूर्णाङ्क byte_idx = (r < 16) ? r : (r - 16);
        अचर uint8_t q = x[ib].qs[16 * ib8 + byte_idx];
        अचर uint8_t qv = (r < 16) ? (q & 0x0F) : (q >> 4);

        अचर भिन्न d = (भिन्न) x[ib].d * ((((x[ib].scales_l[ib8 / 2] >> (4 * (ib8 % 2))) & 0xf) |
                        (((x[ib].scales_h >> (2 * ib8)) & 3) << 4)) - 32);
        फर्काऊ d * kvalues_iq4nl[qv];
    };

    v.x() = dequantize_one(iqs + 0);
    v.y() = dequantize_one(iqs + 1);
#अन्यथा
    GGML_ABORT("IQ4_XS dequantize not supported for QK_K != 256");
#endif
}

स्थिर __dpct_inline__ शून्य dequantize_q5_0(अचर शून्य *vx, अचर int64_t ib,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
    अचर block_q5_0 * x = (अचर block_q5_0 *) vx;

    अचर dfloat d = x[ib].d;

    uint32_t qh;
    स_नकल(&qh, x[ib].qh, परिमाण(qh));

    अचर पूर्णाङ्क xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    अचर पूर्णाङ्क xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

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

स्थिर __dpct_inline__ शून्य dequantize_q5_1(अचर शून्य *vx, अचर int64_t ib,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
    अचर block_q5_1 * x = (अचर block_q5_1 *) vx;

    अचर dfloat d = x[ib].dm[0];
    अचर dfloat m = x[ib].dm[1];

    uint32_t qh;
    स_नकल(&qh, x[ib].qh, परिमाण(qh));

    अचर पूर्णाङ्क xh_0 = ((qh >> (iqs +  0)) << 4) & 0x10;
    अचर पूर्णाङ्क xh_1 = ((qh >> (iqs + 12))     ) & 0x10;

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

स्थिर __dpct_inline__ शून्य dequantize_q8_0_reorder(अचर शून्य *d_ptr, अचर int64_t ib, अचर शून्य *qs,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
    अचर dfloat d = (अचर dfloat)*((अचर sycl::half*)d_ptr + ib);

    v.x() = ((अचर int8_t *)qs)[iqs + 0];
    v.y() = ((अचर int8_t *)qs)[iqs + 1];

#ifdef GGML_SYCL_F16
    v.s0() *= d;
    v.s1() *= d;
#अन्यथा
    v.x() *= d;
    v.y() *= d;
#endif // GGML_SYCL_F16
}

स्थिर __dpct_inline__ शून्य dequantize_q8_0(अचर शून्य *vx, अचर int64_t ib,
                                            अचर पूर्णाङ्क iqs, dfloat2 &v) {
    अचर block_q8_0 * x = (अचर block_q8_0 *) vx;

    अचर dfloat d = x[ib].d;

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
स्थिर शून्य dequantize_block_q4_0(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  अचर sycl::nd_item<3> &item_ct1) {

    अचर int64_t i = item_ct1.get_group(2);

    // assume 32 threads
    अचर int64_t tid = item_ct1.get_local_id(2);
    अचर int64_t il  = tid/8;
    अचर int64_t ir  = tid%8;
    अचर int64_t ib = 8*i + ir;
    यदि (ib >= nb32) {
        फर्काऊ;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    अचर block_q4_0 * x = (अचर block_q4_0 *)vx + ib;
    अचर भिन्न d = sycl::vec<sycl::half, 1>(x->d)
                        .convert<भिन्न, sycl::rounding_mode::automatic>()[0];
    अचर भिन्न dm = -8*d;

    अचर uint8_t * q = x->qs + 4*il;

    लागि (पूर्णाङ्क l = 0; l < 4; ++l) {
        y[l+ 0] = d * (q[l] & 0xF) + dm;
        y[l+16] = d * (q[l] >>  4) + dm;
    }
}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q4_0_reorder(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                  अचर sycl::nd_item<3> &item_ct1) {
    // one thread per qs byte, like dequantize_block<dequantize_q4_0>; the reorder layout makes the byte reads contiguous
    // grid-stride loop because the launcher may shrink the work-group to keep the global range below INT_MAX
    // float math, not dequantize_q4_0_reorder: that helper computes in half with GGML_SYCL_F16, which changes f32 output
    अचर sycl::half * d_ptr  = (अचर sycl::half *) ((अचर uint8_t *) vx + k/2);
    अचर int64_t      stride = item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    लागि (int64_t iq = item_ct1.get_global_id(2); iq < k/2; iq += stride) {
        अचर int64_t ib  = iq / (QK4_0/2);
        अचर int64_t iqs = iq % (QK4_0/2);
        अचर पूर्णाङ्क     vq  = ((अचर uint8_t *) vx)[iq];
        अचर भिन्न   d   = भिन्न(d_ptr[ib]);

        dst_t * y = yy + ib*QK4_0 + iqs;
        y[0]       = d * ((vq & 0xF) - 8);
        y[QK4_0/2] = d * ((vq >>  4) - 8);
    }
}

// Dequantize Q8_0 from reorder layout: [all qs (k bytes)][all d values]
ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q8_0_reorder(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                  अचर sycl::nd_item<3> &item_ct1) {
    // two values per thread, like dequantize_block<dequantize_q8_0>; the reorder layout keeps all qs contiguous
    // grid-stride loop because the launcher may shrink the work-group to keep the global range below INT_MAX
    // float math, not dequantize_q8_0_reorder: that helper computes in half with GGML_SYCL_F16, which changes f32 output
    अचर int8_t *     qs     = (अचर int8_t *) vx;
    अचर sycl::half * d_ptr  = (अचर sycl::half *) ((अचर uint8_t *) vx + k);
    अचर int64_t      stride = 2 * item_ct1.get_local_range(2) * item_ct1.get_group_range(2);

    लागि (int64_t i = 2 * item_ct1.get_global_id(2); i < k; i += stride) {
        अचर भिन्न d = भिन्न(d_ptr[i / QK8_0]);

        yy[i + 0] = d * qs[i + 0];
        yy[i + 1] = d * qs[i + 1];
    }
}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q8_0_reorder_wide(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                               अचर sycl::id<1> idx) {
    अचर int64_t i = 4 * (int64_t) idx[0];

    अचर int8_t *     qs    = (अचर int8_t *) vx;
    अचर sycl::half * d_ptr = (अचर sycl::half *) ((अचर uint8_t *) vx + k);

    अचर भिन्न d0 = भिन्न(d_ptr[i / QK8_0]);
    अचर भिन्न d1 = भिन्न(d_ptr[(i + 2) / QK8_0]);

    sycl::vec<भिन्न, 4> p;
    p[0] = d0 * (भिन्न) qs[i + 0];
    p[1] = d0 * (भिन्न) qs[i + 1];
    p[2] = d1 * (भिन्न) qs[i + 2];
    p[3] = d1 * (भिन्न) qs[i + 3];

    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yy + i) = p.convert<dst_t, sycl::rounding_mode::automatic>();
}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q4_1(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb32,
                                  अचर sycl::nd_item<3> &item_ct1) {

    अचर int64_t i = item_ct1.get_group(2);

    // assume 32 threads
    अचर int64_t tid = item_ct1.get_local_id(2);
    अचर int64_t il  = tid/8;
    अचर int64_t ir  = tid%8;
    अचर int64_t ib = 8*i + ir;
    यदि (ib >= nb32) {
        फर्काऊ;
    }

    dst_t * y = yy + 256*i + 32*ir + 4*il;

    अचर block_q4_1 * x = (अचर block_q4_1 *)vx + ib;
    अचर sycl::float2 d =
        x->dm.convert<भिन्न, sycl::rounding_mode::automatic>();

    अचर uint8_t * q = x->qs + 4*il;

    लागि (पूर्णाङ्क l = 0; l < 4; ++l) {
        y[l + 0] = d.x() * (q[l] & 0xF) + d.y();
        y[l + 16] = d.x() * (q[l] >> 4) + d.y();
    }
}


//================================== k-quants

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q2_K(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy,
                                  अचर sycl::nd_item<3> &item_ct1) {

    अचर int64_t i = item_ct1.get_group(2);
    अचर block_q2_K * x = (अचर block_q2_K *) vx;

    अचर int64_t tid = item_ct1.get_local_id(2);
#यदि QK_K == 256
    अचर int64_t n   = tid/32;
    अचर int64_t l   = tid - 32*n;
    अचर int64_t is  = 8*n + l/16;

    अचर uint8_t q = x[i].qs[32*n + l];
    dst_t * y = yy + i*QK_K + 128*n;

    भिन्न dall = x[i].dm[0];
    भिन्न dmin = x[i].dm[1];
    y[l+ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[l+32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 2) & 3) - dmin * (x[i].scales[is+2] >> 4);
    y[l+64] = dall * (x[i].scales[is+4] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+4] >> 4);
    y[l+96] = dall * (x[i].scales[is+6] & 0xF) * ((q >> 6) & 3) - dmin * (x[i].scales[is+6] >> 4);
#अन्यथा
    अचर int64_t is = tid/16;  // 0 or 1
    अचर int64_t il = tid%16;  // 0...15
    अचर uint8_t q = x[i].qs[il] >> (2*is);
    dst_t * y = yy + i*QK_K + 16*is + il;

    भिन्न dall = x[i].dm[0];
    भिन्न dmin = x[i].dm[1];
    y[ 0] = dall * (x[i].scales[is+0] & 0xF) * ((q >> 0) & 3) - dmin * (x[i].scales[is+0] >> 4);
    y[32] = dall * (x[i].scales[is+2] & 0xF) * ((q >> 4) & 3) - dmin * (x[i].scales[is+2] >> 4);
#endif

}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q2_K_reorder(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, अचर int64_t i,
                                          अचर sycl::nd_item<3> & item_ct1, int64_t n_blocks) {
#यदि QK_K == 256
    यदि (i >= n_blocks) {
        फर्काऊ;
    }

    अचर uint8_t * base          = static_cast<अचर uint8_t *>(vx);
    अचर माप_प्रकार    qs_offset     = i * (QK_K / 4);
    अचर माप_प्रकार    scales_offset = n_blocks * (QK_K / 4) + i * (QK_K / 16);
    अचर माप_प्रकार    dm_offset     = n_blocks * (QK_K / 4) + n_blocks * (QK_K / 16) + i * परिमाण(ggml_half2);

    अचर uint8_t *     qs     = base + qs_offset;
    अचर uint8_t *     scales = base + scales_offset;
    अचर ggml_half2 * dm     = reinterpret_cast<अचर ggml_half2 *>(base + dm_offset);

    अचर int64_t tid = item_ct1.get_local_id(2);
    अचर int64_t n    = tid / 32;
    अचर int64_t l    = tid - 32 * n;
    अचर int64_t is   = 8 * n + l / 16;

    अचर uint8_t q = qs[32 * n + l];
    dst_t * y = yy + 128 * n;

    अचर भिन्न dall = (*dm)[0];
    अचर भिन्न dmin = (*dm)[1];
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
स्थिर शून्य dequantize_block_q3_K(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy,
                                  अचर sycl::nd_item<3> &item_ct1) {

    अचर int64_t i = item_ct1.get_group(2);
    अचर block_q3_K * x = (अचर block_q3_K *) vx;

#यदि QK_K == 256
    अचर int64_t r = item_ct1.get_local_id(2) / 4;
    अचर int64_t tid = r/2;
    अचर int64_t is0 = r%2;
    अचर int64_t l0 = 16 * is0 + 4 * (item_ct1.get_local_id(2) % 4);
    अचर int64_t n = tid / 4;
    अचर int64_t j = tid - 4*n;

    uint8_t m = 1 << (4*n + j);
    int64_t is = 8*n + 2*j + is0;
    पूर्णाङ्क shift = 2*j;

    int8_t us = is <  4 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+8] >> 0) & 3) << 4) :
                is <  8 ? (x[i].scales[is-0] & 0xF) | (((x[i].scales[is+4] >> 2) & 3) << 4) :
                is < 12 ? (x[i].scales[is-8] >>  4) | (((x[i].scales[is+0] >> 4) & 3) << 4) :
                          (x[i].scales[is-8] >>  4) | (((x[i].scales[is-4] >> 6) & 3) << 4);
    भिन्न d_all = x[i].d;
    भिन्न dl = d_all * (us - 32);

    dst_t * y = yy + i*QK_K + 128*n + 32*j;
    अचर uint8_t * q = x[i].qs + 32*n;
    अचर uint8_t * hm = x[i].hmask;

    लागि (पूर्णाङ्क l = l0; l < l0+4; ++l) y[l] = dl * ((int8_t)((q[l] >> shift) & 3) - ((hm[l] & m) ? 0 : 4));
#अन्यथा
    अचर int64_t tid = item_ct1.get_local_id(2);
    अचर int64_t is  = tid/16;  // 0 or 1
    अचर int64_t il  = tid%16;  // 0...15
    अचर int64_t im  = il/8;    // 0...1
    अचर int64_t in  = il%8;    // 0...7

    dst_t * y = yy + i*QK_K + 16*is + il;

    अचर uint8_t q = x[i].qs[il] >> (2*is);
    अचर uint8_t h = x[i].hmask[in] >> (2*is + im);
    अचर भिन्न   d = (भिन्न)x[i].d;

    यदि (is == 0) {
        y[ 0] = d * ((x[i].scales[0] & 0xF) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] & 0xF) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    } अन्यथा {
        y[ 0] = d * ((x[i].scales[0] >>  4) - 8) * ((int8_t)((q >> 0) & 3) - ((h >> 0) & 1 ? 0 : 4));
        y[32] = d * ((x[i].scales[1] >>  4) - 8) * ((int8_t)((q >> 4) & 3) - ((h >> 4) & 1 ? 0 : 4));
    }
#endif

}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_q3_K_reorder(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, अचर int64_t i,
                                          अचर sycl::nd_item<3> & item_ct1, int64_t n_blocks) {
#यदि QK_K == 256
    यदि (i >= n_blocks) {
        फर्काऊ;
    }

    अचर uint8_t * base          = static_cast<अचर uint8_t *>(vx);
    अचर माप_प्रकार    qs_offset     = i * (QK_K / 4);
    अचर माप_प्रकार    hmask_offset  = n_blocks * (QK_K / 4) + i * (QK_K / 8);
    अचर माप_प्रकार    scales_offset = n_blocks * (QK_K / 4) + n_blocks * (QK_K / 8) + i * 12;
    अचर माप_प्रकार    d_offset      = n_blocks * (QK_K / 4) + n_blocks * (QK_K / 8) + n_blocks * 12 +
                                 i * परिमाण(ggml_half);

    अचर uint8_t * qs     = base + qs_offset;
    अचर uint8_t * hmask  = base + hmask_offset;
    अचर uint8_t * scales = base + scales_offset;
    अचर भिन्न     d_all  = static_cast<भिन्न>(*reinterpret_cast<अचर ggml_half *>(base + d_offset));

    अचर int64_t r    = item_ct1.get_local_id(2) / 4;
    अचर int64_t tid  = r / 2;
    अचर int64_t is0  = r % 2;
    अचर int64_t l0   = 16 * is0 + 4 * (item_ct1.get_local_id(2) % 4);
    अचर int64_t n    = tid / 4;
    अचर int64_t j    = tid - 4 * n;
    अचर int64_t is   = 8 * n + 2 * j + is0;
    अचर पूर्णाङ्क     shift = 2 * j;
    uint8_t       m    = 1 << (4 * n + j);

    uint8_t us = is < 4
        ? (scales[is - 0] & 0xF) | (((scales[is + 8] >> 0) & 3) << 4)
        : is < 8
            ? (scales[is - 0] & 0xF) | (((scales[is + 4] >> 2) & 3) << 4)
            : is < 12
                ? (scales[is - 8] >> 4) | (((scales[is + 0] >> 4) & 3) << 4)
                : (scales[is - 8] >> 4) | (((scales[is - 4] >> 6) & 3) << 4);

    अचर भिन्न dl = d_all * (us - 32);

    dst_t * y = yy + 128 * n + 32 * j;
    अचर uint8_t * q  = qs + 32 * n;
    अचर uint8_t * hm = hmask;

    लागि (पूर्णाङ्क l = l0; l < l0 + 4; ++l) {
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

#यदि QK_K == 256
स्थिर अन्तःस्थ शून्य get_scale_min_k4(पूर्णाङ्क j, अचर uint8_t * q, uint8_t & d, uint8_t & m) {
    यदि (j < 4) {
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
स्थिर constexpr पूर्णाङ्क SYCL_DEQK_WI_PER_BLOCK = 32;

ढाँचा <typename dst_t>
स्थिर अन्तःस्थ शून्य dequantize_q4_K_wide_one(dst_t * __restrict__ y, अचर uint8_t * __restrict__ qs,
                                            अचर uint8_t * __restrict__ scales, अचर भिन्न dall,
                                            अचर भिन्न dmin, अचर पूर्णाङ्क tid) {
    अचर पूर्णाङ्क il = tid >> 3;        // 64-element chunk, 0..3
    अचर पूर्णाङ्क ir = (tid & 7) * 4;   // 4 qs bytes within the chunk

    uint8_t sc, m;
    get_scale_min_k4(2 * il + 0, scales, sc, m);
    अचर भिन्न d1 = dall * sc;
    अचर भिन्न m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, scales, sc, m);
    अचर भिन्न d2 = dall * sc;
    अचर भिन्न m2 = dmin * m;

    अचर sycl::vec<uint8_t, 4> q = vec_aligned_load<uint8_t, 4>(qs + 32 * il + ir);

    sycl::vec<dst_t, 4> lo, hi;
#pragma unroll
    लागि (पूर्णाङ्क l = 0; l < 4; ++l) {
        lo[l] = d1 * (q[l] & 0xF) - m1;
        hi[l] = d2 * (q[l] >> 4)  - m2;
    }
    dst_t * yb = y + 64 * il + ir;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb +  0) = lo;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb + 32) = hi;
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_block_q4_K_wide(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb,
                                       अचर sycl::nd_item<1> & it) {
    static_assert(QK_K == 256, "wide K-quant dequant assumes QK_K == 256");
    अचर int64_t g   = it.get_global_id(0);
    अचर int64_t b   = g / SYCL_DEQK_WI_PER_BLOCK;
    अचर पूर्णाङ्क     tid = (पूर्णाङ्क) (g % SYCL_DEQK_WI_PER_BLOCK);
    यदि (b >= nb) {
        फर्काऊ;
    }
    अचर block_q4_K * x = (अचर block_q4_K *) vx + b;
    dequantize_q4_K_wide_one(yy + b * QK_K, x->qs, x->scales, (भिन्न) x->dm[0], (भिन्न) x->dm[1], tid);
}


ढाँचा <typename dst_t>
अन्तःस्थ शून्य dequantize_q4_K_common(dst_t * __restrict__ y, अचर uint8_t * __restrict__ qs_ptr, अचर भिन्न dall,
                                   अचर भिन्न dmin, uint8_t * __restrict__ scales_local, पूर्णाङ्क il, पूर्णाङ्क ir) {
    अचर पूर्णाङ्क is = 2 * il;
    constexpr पूर्णाङ्क n  = 4;

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    अचर भिन्न d1 = dall * sc;
    अचर भिन्न m1 = dmin * m;

    get_scale_min_k4(is + 1, scales_local, sc, m);
    अचर भिन्न d2 = dall * sc;
    अचर भिन्न m2 = dmin * m;

    sycl::vec<uint8_t, n> q_vec = vec_aligned_load<uint8_t, n>(qs_ptr + 32 * il + n * ir);
    लागि (पूर्णाङ्क l = 0; l < n; ++l) {
        y[l + 0]  = d1 * (q_vec[l] & 0xF) - m1;
        y[l + 32] = d2 * (q_vec[l] >> 4) - m2;
    }
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_block_q4_K_reorder(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, अचर int64_t i,
                                          uint8_t * scales_local, अचर sycl::nd_item<1> & item_ct1, int64_t nb) {
    अचर int64_t tid = item_ct1.get_local_id(0);  // thread index within block
    अचर int64_t il  = tid / 8;
    अचर int64_t ir  = tid % 8;

    dst_t * y = yy + 64 * il + 4 * ir;

    अचर uint8_t * base          = static_cast<अचर uint8_t *>(vx);
    अचर माप_प्रकार    qs_offset     = i * (QK_K / 2);
    अचर माप_प्रकार    scales_offset = nb * (QK_K / 2) + i * K_SCALE_SIZE;
    अचर माप_प्रकार    dm_offset     = nb * (QK_K / 2) + nb * K_SCALE_SIZE + i * परिमाण(ggml_half2);

    अचर uint8_t *    qs_ptr     = base + qs_offset;
    अचर uint8_t *    scales_ptr = base + scales_offset;
    ggml_half2         dm_मानहरू  = *reinterpret_cast<अचर ggml_half2 *>(base + dm_offset);

    अचर भिन्न dall = dm_मानहरू.x();
    अचर भिन्न dmin = dm_मानहरू.y();

    यदि (tid < 12) {
        scales_local[tid] = scales_ptr[tid];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);
    dequantize_q4_K_common(y, qs_ptr, dall, dmin, scales_local, il, ir);
}

ढाँचा <typename dst_t>
स्थिर अन्तःस्थ शून्य dequantize_q5_K_wide_one(dst_t * __restrict__ y, अचर uint8_t * __restrict__ qs,
                                            अचर uint8_t * __restrict__ qh_ptr,
                                            अचर uint8_t * __restrict__ scales, अचर भिन्न dall,
                                            अचर भिन्न dmin, अचर पूर्णाङ्क tid) {
    अचर पूर्णाङ्क il = tid >> 3;        // 64-element chunk, 0..3
    अचर पूर्णाङ्क ir = (tid & 7) * 4;   // 4 qs bytes within the chunk; qh holds the 5th bit

    uint8_t sc, m;
    get_scale_min_k4(2 * il + 0, scales, sc, m);
    अचर भिन्न d1 = dall * sc;
    अचर भिन्न m1 = dmin * m;
    get_scale_min_k4(2 * il + 1, scales, sc, m);
    अचर भिन्न d2 = dall * sc;
    अचर भिन्न m2 = dmin * m;

    अचर sycl::vec<uint8_t, 4> q = vec_aligned_load<uint8_t, 4>(qs + 32 * il + ir);
    अचर sycl::vec<uint8_t, 4> h = vec_aligned_load<uint8_t, 4>(qh_ptr + ir);

    अचर uint8_t hm1 = 1 << (2 * il);
    अचर uint8_t hm2 = hm1 << 1;

    sycl::vec<dst_t, 4> lo, hi;
#pragma unroll
    लागि (पूर्णाङ्क l = 0; l < 4; ++l) {
        lo[l] = d1 * ((q[l] & 0xF) + ((h[l] & hm1) ? 16 : 0)) - m1;
        hi[l] = d2 * ((q[l] >>  4) + ((h[l] & hm2) ? 16 : 0)) - m2;
    }
    dst_t * yb = y + 64 * il + ir;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb +  0) = lo;
    *reinterpret_cast<sycl::vec<dst_t, 4> *>(yb + 32) = hi;
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_block_q5_K_wide(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t nb,
                                       अचर sycl::nd_item<1> & it) {
    static_assert(QK_K == 256, "wide K-quant dequant assumes QK_K == 256");
    अचर int64_t g   = it.get_global_id(0);
    अचर int64_t b   = g / SYCL_DEQK_WI_PER_BLOCK;
    अचर पूर्णाङ्क     tid = (पूर्णाङ्क) (g % SYCL_DEQK_WI_PER_BLOCK);
    यदि (b >= nb) {
        फर्काऊ;
    }
    अचर block_q5_K * x = (अचर block_q5_K *) vx + b;
    dequantize_q5_K_wide_one(yy + b * QK_K, x->qs, x->qh, x->scales, (भिन्न) x->dm[0], (भिन्न) x->dm[1], tid);
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_block_q5_K_reorder(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, अचर int64_t ib,
                                          uint8_t * scales_local, अचर sycl::nd_item<3> & item_ct1, int64_t n_blocks) {

#यदि QK_K == 256
    // assume 64 threads
    अचर int64_t tid = item_ct1.get_local_id(2);
    अचर int64_t il  = tid / 16;   // 0...3
    अचर int64_t ir  = tid % 16;   // 0...15
    अचर int64_t is  = 2 * il;

    dst_t * y = yy + 64 * il + 2 * ir;

    अचर uint8_t * base = static_cast<अचर uint8_t *>(vx);

    // Reordered layout: [qs (QK_K/2 per block)] [qh (QK_K/8 per block)] [scales (K_SCALE_SIZE per block)] [dm (half2 per block)]
    अचर माप_प्रकार qs_offset     = ib * (QK_K / 2);
    अचर माप_प्रकार qh_offset     = n_blocks * (QK_K / 2) + ib * (QK_K / 8);
    अचर माप_प्रकार scales_offset = n_blocks * (QK_K / 2) + n_blocks * (QK_K / 8) + ib * K_SCALE_SIZE;
    अचर माप_प्रकार dm_offset     = n_blocks * (QK_K / 2) + n_blocks * (QK_K / 8) + n_blocks * K_SCALE_SIZE + ib * परिमाण(ggml_half2);

    अचर uint8_t *  qs_ptr     = base + qs_offset;
    अचर uint8_t *  qh_ptr     = base + qh_offset;
    अचर uint8_t *  scales_ptr = base + scales_offset;
    अचर ggml_half2 dm_मानहरू  = *reinterpret_cast<अचर ggml_half2 *>(base + dm_offset);

    अचर भिन्न dall = dm_मानहरू.x();
    अचर भिन्न dmin = dm_मानहरू.y();

    अचर uint8_t * ql = qs_ptr + 32 * il + 2 * ir;
    अचर uint8_t * qh = qh_ptr + 2 * ir;

    यदि (tid < K_SCALE_SIZE) {
        scales_local[tid] = scales_ptr[tid];
    }

    item_ct1.barrier(sycl::access::fence_space::local_space);

    uint8_t sc, m;
    get_scale_min_k4(is + 0, scales_local, sc, m);
    अचर भिन्न d1 = dall * sc; अचर भिन्न m1 = dmin * m;
    get_scale_min_k4(is + 1, scales_local, sc, m);
    अचर भिन्न d2 = dall * sc; अचर भिन्न m2 = dmin * m;

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
स्थिर शून्य dequantize_block_q6_K(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy,
                                  अचर sycl::nd_item<3> &item_ct1) {
    अचर block_q6_K * x = (अचर block_q6_K *) vx;

    अचर int64_t i = item_ct1.get_group(2);
#यदि QK_K == 256

    // assume 64 threads - this is very slightly better than the one below
    अचर int64_t tid = item_ct1.get_local_id(2);
    अचर int64_t ip  = tid/32;   // ip is 0 or 1
    अचर int64_t il  = tid - 32*ip; // 0...32
    अचर int64_t is  = 8*ip + il/16;

    dst_t * y = yy + i*QK_K + 128*ip + il;

    अचर भिन्न d = x[i].d;

    अचर uint8_t * ql = x[i].ql + 64*ip + il;
    अचर uint8_t   qh = x[i].qh[32*ip + il];
    अचर int8_t  * sc = x[i].scales + is;

    y[ 0] = d * sc[0] * ((int8_t)((ql[ 0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[2] * ((int8_t)((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = d * sc[4] * ((int8_t)((ql[ 0]  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = d * sc[6] * ((int8_t)((ql[32]  >> 4) | (((qh >> 6) & 3) << 4)) - 32);
#अन्यथा

    // assume 32 threads
    अचर int64_t tid = item_ct1.get_local_id(2);
    अचर int64_t ip  = tid/16;         // 0 or 1
    अचर int64_t il  = tid - 16*ip;    // 0...15

    dst_t * y = yy + i*QK_K + 16*ip + il;

    अचर भिन्न d = x[i].d;

    अचर uint8_t   ql = x[i].ql[16*ip + il];
    अचर uint8_t   qh = x[i].qh[il] >> (2*ip);
    अचर int8_t  * sc = x[i].scales;

    y[ 0] = d * sc[ip+0] * ((int8_t)((ql & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = d * sc[ip+2] * ((int8_t)((ql  >> 4) | (((qh >> 4) & 3) << 4)) - 32);
#endif
}

ढाँचा <typename dst_t>
स्थिर शून्य dequantize_block_q6_K_reorder(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, अचर int64_t ib,
                                          अचर sycl::nd_item<3> & item_ct1, int64_t n_blocks) {

    अचर int64_t tid = item_ct1.get_local_id(2);
    अचर int64_t ip  = tid / 32;       // ip is 0 or 1
    अचर int64_t il  = tid - 32 * ip;  // 0...32
    अचर int64_t is  = 8 * ip + il / 16;

    अचर uint8_t *   base_ptr           = static_cast<अचर uint8_t *>(vx);
    अचर स्वतः        ql_offset          = ib * (QK_K / 2);
    अचर स्वतः        qh_offset          = (QK_K / 2) * n_blocks + (QK_K / 4) * ib;
    अचर स्वतः        base_scales_offset = (QK_K / 2) * n_blocks + (QK_K / 4) * n_blocks + (QK_K / 16) * ib;
    अचर स्वतः        base_d_offset      = ((QK_K / 2) + (QK_K / 4) + (QK_K / 16)) * n_blocks;
    अचर uint8_t *   ql_ptr             = base_ptr + ql_offset;
    अचर uint8_t *   qh_ptr             = base_ptr + qh_offset;
    अचर uint8_t *   scales_ptr         = base_ptr + base_scales_offset;
    अचर ggml_half * d                  = (अचर ggml_half *) (base_ptr + base_d_offset) + ib;

    dst_t * y = yy + 128 * ip + il;

    अचर uint8_t * ql = ql_ptr + 64 * ip + il;
    अचर uint8_t   qh = *(qh_ptr + 32 * ip + il);
    अचर int8_t *  sc = reinterpret_cast<अचर int8_t *>(scales_ptr + is);

    y[0]  = *d * sc[0] * ((int8_t) ((ql[0] & 0xF) | (((qh >> 0) & 3) << 4)) - 32);
    y[32] = *d * sc[2] * ((int8_t) ((ql[32] & 0xF) | (((qh >> 2) & 3) << 4)) - 32);
    y[64] = *d * sc[4] * ((int8_t) ((ql[0] >> 4) | (((qh >> 4) & 3) << 4)) - 32);
    y[96] = *d * sc[6] * ((int8_t) ((ql[32] >> 4) | (((qh >> 6) & 3) << 4)) - 32);
}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_iq2_xxs(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy,
                                     अचर sycl::nd_item<3> &item_ct1,
                                     अचर uint64_t *iq2xxs_grid_ptr,
                                     अचर uint8_t *ksigns_iq2xs_ptr,
                                     अचर uint8_t *kmask_iq2xs_ptr) {

    अचर int64_t i = item_ct1.get_group(2);
    अचर block_iq2_xxs * x = (अचर block_iq2_xxs  *) vx;

    अचर int64_t tid = item_ct1.get_local_id(2);
#यदि QK_K == 256
    अचर int64_t il = tid/8; // 0...3
    अचर int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    अचर uint16_t * q2 = x[i].qs + 4*ib;
    अचर uint8_t  * aux8 = (अचर uint8_t *)q2;
    अचर uint8_t  * grid = (अचर uint8_t *)(iq2xxs_grid_ptr + aux8[il]);
    अचर uint32_t aux32 = q2[2] | (q2[3] << 16);
    अचर भिन्न d = (भिन्न)x[i].d * (0.5f + (aux32 >> 28)) * 0.25f;
    अचर uint8_t signs = ksigns_iq2xs_ptr[(aux32 >> 7*il) & 127];
    लागि (पूर्णाङ्क j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs_ptr[j] ? -1.f : 1.f);
#अन्यथा
    प्रतिज्ञा(असत्य);
#endif

}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_iq2_xs(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy,
                                    अचर sycl::nd_item<3> &item_ct1,
                                    अचर uint64_t *iq2xs_grid,
                                    अचर uint8_t *ksigns_iq2xs,
                                    अचर uint8_t *kmask_iq2xs) {

    अचर int64_t i = item_ct1.get_group(2);
    अचर block_iq2_xs * x = (अचर block_iq2_xs *) vx;

    अचर int64_t tid = item_ct1.get_local_id(2);
#यदि QK_K == 256
    अचर int64_t il = tid/8; // 0...3
    अचर int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    अचर uint16_t * q2 = x[i].qs + 4*ib;
    अचर uint8_t  * grid = (अचर uint8_t *)(iq2xs_grid + (q2[il] & 511));
    अचर भिन्न d = (भिन्न)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    अचर uint8_t signs = ksigns_iq2xs[q2[il] >> 9];
    लागि (पूर्णाङ्क j = 0; j < 8; ++j) y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#अन्यथा
    प्रतिज्ञा(असत्य);
#endif

}

ढाँचा <typename dst_t>
__dpct_inline__ स्थिर शून्य
dequantize_block_iq2_s(अचर शून्य *__restrict__ vx, dst_t *__restrict__ yy,
                       अचर sycl::nd_item<3> &item_ct1) {

    अचर int64_t i = item_ct1.get_group(2);
    अचर block_iq2_s * x = (अचर block_iq2_s *) vx;

    अचर int64_t tid = item_ct1.get_local_id(2);
#यदि QK_K == 256
    अचर int64_t il = tid/8; // 0...3
    अचर int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    अचर uint8_t * grid = (अचर uint8_t *)(iq2s_grid + (x[i].qs[4*ib+il] | ((x[i].qh[ib] << (8-2*il)) & 0x300)));
    अचर भिन्न d = (भिन्न)x[i].d * (0.5f + ((x[i].scales[ib] >> 4*(il/2)) & 0xf)) * 0.25f;
    अचर uint8_t signs = x[i].qs[QK_K/8+4*ib+il];
#pragma unroll
    लागि (पूर्णाङ्क j = 0; j < 8; ++j)
        y[j] = d * grid[j] * (signs & kmask_iq2xs[j] ? -1.f : 1.f);
#अन्यथा
    प्रतिज्ञा(असत्य);

#endif

}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_iq3_xxs(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy,
                                     अचर sycl::nd_item<3> &item_ct1,
                                     अचर uint32_t *iq3xxs_grid,
                                     अचर uint8_t *ksigns_iq2xs,
                                     अचर uint8_t *kmask_iq2xs) {

    अचर int64_t i = item_ct1.get_group(2);
    अचर block_iq3_xxs * x = (अचर block_iq3_xxs  *) vx;

    अचर int64_t tid = item_ct1.get_local_id(2);
#यदि QK_K == 256
    अचर int64_t il = tid/8; // 0...3
    अचर int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    अचर uint8_t  * q3 = x[i].qs + 8*ib;
    अचर uint16_t * gas = (अचर uint16_t *)(x[i].qs + QK_K/4) + 2*ib;
    अचर uint8_t  * grid1 = (अचर uint8_t *)(iq3xxs_grid + q3[2*il+0]);
    अचर uint8_t  * grid2 = (अचर uint8_t *)(iq3xxs_grid + q3[2*il+1]);
    अचर uint32_t aux32 = gas[0] | (gas[1] << 16);
    अचर भिन्न d = (भिन्न)x[i].d * (0.5f + (aux32 >> 28)) * 0.5f;
    अचर uint8_t signs = ksigns_iq2xs[(aux32 >> 7*il) & 127];
    लागि (पूर्णाङ्क j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#अन्यथा
    प्रतिज्ञा(असत्य);
#endif

}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_iq3_xxs_reorder(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, अचर int64_t i,
                                             अचर sycl::nd_item<3> & item_ct1, अचर int64_t nb,
                                             अचर uint32_t * iq3xxs_grid,
                                             अचर uint8_t * ksigns_iq2xs,
                                             अचर uint8_t * kmask_iq2xs) {
    अचर int64_t tid = item_ct1.get_local_id(2);
#यदि QK_K == 256
    अचर int64_t il = tid / 8;
    अचर int64_t ib = tid % 8;
    अचर uint8_t * base = static_cast<अचर uint8_t *>(vx);
    अचर uint8_t * qs = base + i * (3 * QK_K / 8);
    अचर ggml_half * d = reinterpret_cast<अचर ggml_half *>(base + nb * (3 * QK_K / 8));
    अचर uint8_t * q3 = qs + 8 * ib;
    अचर uint16_t * gas = reinterpret_cast<अचर uint16_t *>(qs + QK_K / 4) + 2 * ib;
    अचर uint8_t * grid1 = reinterpret_cast<अचर uint8_t *>(iq3xxs_grid + q3[2*il+0]);
    अचर uint8_t * grid2 = reinterpret_cast<अचर uint8_t *>(iq3xxs_grid + q3[2*il+1]);
    अचर uint32_t aux32 = gas[0] | (gas[1] << 16);
    अचर भिन्न dl = (भिन्न) d[i] * (0.5f + (aux32 >> 28)) * 0.5f;
    अचर uint8_t signs = ksigns_iq2xs[(aux32 >> (7 * il)) & 127];
    dst_t * y = yy + 32 * ib + 8 * il;
#pragma unroll
    लागि (पूर्णाङ्क j = 0; j < 4; ++j) {
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
dequantize_block_iq3_s(अचर शून्य *__restrict__ vx, dst_t *__restrict__ yy,
                       अचर sycl::nd_item<3> &item_ct1,
                       अचर uint8_t *kmask_iq2xs, अचर uint32_t *iq3s_grid) {

    अचर int64_t i = item_ct1.get_group(2);
    अचर block_iq3_s * x = (अचर block_iq3_s *) vx;

    अचर int64_t tid = item_ct1.get_local_id(2);
#यदि QK_K == 256
    अचर int64_t il = tid/8; // 0...3
    अचर int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    अचर uint8_t * qs = x[i].qs + 8*ib;
    अचर uint8_t * grid1 = (अचर uint8_t *)(iq3s_grid + (qs[2*il+0] | ((x[i].qh[ib] << (8-2*il)) & 256)));
    अचर uint8_t * grid2 = (अचर uint8_t *)(iq3s_grid + (qs[2*il+1] | ((x[i].qh[ib] << (7-2*il)) & 256)));
    अचर भिन्न d = (भिन्न)x[i].d * (1 + 2*((x[i].scales[ib/2] >> 4*(ib%2)) & 0xf));
    अचर uint8_t signs = x[i].signs[4*ib + il];
#pragma unroll
    लागि (पूर्णाङ्क j = 0; j < 4; ++j) {
        y[j+0] = d * grid1[j] * (signs & kmask_iq2xs[j+0] ? -1.f : 1.f);
        y[j+4] = d * grid2[j] * (signs & kmask_iq2xs[j+4] ? -1.f : 1.f);
    }
#अन्यथा
    प्रतिज्ञा(असत्य);
#endif

}

ढाँचा <typename dst_t>
__dpct_inline__ स्थिर शून्य
dequantize_block_iq3_s_reorder(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, अचर int64_t i,
                               अचर sycl::nd_item<3> & item_ct1, अचर int64_t nb,
                               अचर uint8_t * kmask_iq2xs, अचर uint32_t * iq3s_grid) {
    अचर int64_t tid = item_ct1.get_local_id(2);
#यदि QK_K == 256
    अचर int64_t il = tid / 8;
    अचर int64_t ib = tid % 8;
    अचर uint8_t * base = static_cast<अचर uint8_t *>(vx);
    अचर uint8_t * qs = base + i * (QK_K / 4);
    अचर uint8_t * qh = base + nb * (QK_K / 4) + i * (QK_K / 32);
    अचर uint8_t * signs_base = base + nb * (QK_K / 4 + QK_K / 32) + i * (QK_K / 8);
    अचर uint8_t * metadata = base + nb * (QK_K / 4 + QK_K / 32 + QK_K / 8) +
                               i * (परिमाण(ggml_half) + IQ3S_N_SCALE);
    अचर ggml_half d = *reinterpret_cast<अचर ggml_half *>(metadata);
    अचर uint8_t * scales = metadata + परिमाण(ggml_half);
    अचर uint8_t * qs_ib = qs + 8 * ib;
    अचर uint8_t * grid1 = reinterpret_cast<अचर uint8_t *>(
        iq3s_grid + (qs_ib[2*il+0] | ((qh[ib] << (8 - 2*il)) & 256)));
    अचर uint8_t * grid2 = reinterpret_cast<अचर uint8_t *>(
        iq3s_grid + (qs_ib[2*il+1] | ((qh[ib] << (7 - 2*il)) & 256)));
    अचर भिन्न dl = (भिन्न) d * (1 + 2 * ((scales[ib/2] >> (4 * (ib%2))) & 0xf));
    अचर uint8_t signs = signs_base[4 * ib + il];
    dst_t * y = yy + 32 * ib + 8 * il;
#pragma unroll
    लागि (पूर्णाङ्क j = 0; j < 4; ++j) {
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
dequantize_block_iq1_s(अचर शून्य *__restrict__ vx, dst_t *__restrict__ yy,
                       अचर sycl::nd_item<3> &item_ct1,
                       अचर uint32_t *iq1s_grid_gpu) {

    अचर int64_t i = item_ct1.get_group(2);
    अचर block_iq1_s * x = (अचर block_iq1_s  *) vx;

    अचर int64_t tid = item_ct1.get_local_id(2);
#यदि QK_K == 256
    अचर int64_t il = tid/8; // 0...3
    अचर int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    अचर भिन्न delta = x[i].qh[ib] & 0x8000 ? -1 - IQ1S_DELTA : -1 + IQ1S_DELTA;
    अचर भिन्न d = (भिन्न)x[i].d * (2*((x[i].qh[ib] >> 12) & 7) + 1);
    uint32_t grid32[2]; अचर int8_t * q = (अचर int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[ib] >> 3*il) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#pragma unroll
    लागि (पूर्णाङ्क j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#अन्यथा
    प्रतिज्ञा(असत्य);
#endif

}

ढाँचा <typename dst_t>
__dpct_inline__ स्थिर शून्य
dequantize_block_iq1_m(अचर शून्य *__restrict__ vx, dst_t *__restrict__ yy,
                       अचर sycl::nd_item<3> &item_ct1,
                       अचर uint32_t *iq1s_grid_gpu) {

    अचर int64_t i = item_ct1.get_group(2);
    अचर block_iq1_m * x = (अचर block_iq1_m  *) vx;

    अचर int64_t tid = item_ct1.get_local_id(2);
#यदि QK_K == 256
    अचर int64_t il = tid/8; // 0...3
    अचर int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 8*il;
    अचर uint16_t * sc = (अचर uint16_t *)x[i].scales;
    iq1m_scale_t scale;
    scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
    अचर पूर्णाङ्क ib16 = 2*ib + il/2; // sc[ib16/4] >> 3*(ib16%4) -> sc[ib/2] >> 3*((2*ib+il/2)%4);
    अचर भिन्न d = (भिन्न)scale.f16 * (2*((sc[ib16/4] >> 3*(ib16%4)) & 0x7) + 1);
    अचर भिन्न delta = x[i].qh[2*ib+il/2] & (0x08 << 4*(il%2)) ? -1 - IQ1M_DELTA : -1 + IQ1M_DELTA;
    uint32_t grid32[2]; अचर int8_t * q = (अचर int8_t *)grid32;
    grid32[0] = iq1s_grid_gpu[x[i].qs[4*ib+il] | (((x[i].qh[2*ib+il/2] >> 4*(il%2)) & 7) << 8)];
    grid32[1] = (grid32[0] >> 4) & 0x0f0f0f0f;
    grid32[0] &= 0x0f0f0f0f;
#pragma unroll
    लागि (पूर्णाङ्क j = 0; j < 8; ++j) {
        y[j] = d * (q[j] + delta);
    }
#अन्यथा
    प्रतिज्ञा(असत्य);
#endif

}

ढाँचा <typename dst_t>
__dpct_inline__ स्थिर शून्य
dequantize_block_iq4_nl(अचर शून्य *__restrict__ vx, dst_t *__restrict__ yy,
                        अचर sycl::nd_item<3> &item_ct1) {

    अचर int64_t i = item_ct1.get_group(2);
    अचर block_iq4_nl * x = (अचर block_iq4_nl *) vx + i*(QK_K/QK4_NL);

    अचर int64_t tid = item_ct1.get_local_id(2);
    अचर int64_t il = tid/8; // 0...3
    अचर int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    अचर uint8_t  * q4 = x[ib].qs + 4*il;
    अचर भिन्न d = (भिन्न)x[ib].d;
#pragma unroll
    लागि (पूर्णाङ्क j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }

}


ढाँचा <typename dst_t>
__dpct_inline__ स्थिर शून्य
dequantize_block_iq4_xs(अचर शून्य *__restrict__ vx, dst_t *__restrict__ yy,
                        अचर sycl::nd_item<3> &item_ct1) {
    अचर int64_t i = item_ct1.get_group(2);
    अचर block_iq4_xs * x = (अचर block_iq4_xs *)vx;

    अचर int64_t tid = item_ct1.get_local_id(2);
    अचर int64_t il = tid/8; // 0...3
    अचर int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    अचर uint8_t  * q4 = x[i].qs + 16*ib + 4*il;
    अचर भिन्न d = (भिन्न)x[i].d * ((((x[i].scales_l[ib/2] >> 4*(ib%2)) & 0xf) | (((x[i].scales_h >> 2*ib) & 3) << 4)) - 32);
#pragma unroll
    लागि (पूर्णाङ्क j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_iq4nl[q4[j] & 0xf];
        y[j+16] = d * kvalues_iq4nl[q4[j] >>  4];
    }
}

ढाँचा<typename dst_t>
स्थिर शून्य dequantize_block_mxfp4(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy,
                                   अचर sycl::nd_item<3> &item_ct1) {
    // auto                item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर int64_t       i        = item_ct1.get_group(2);
    अचर block_mxfp4 * x = (अचर block_mxfp4 *) vx + i*(QK_K/QK_MXFP4);

    अचर int64_t    tid = item_ct1.get_local_id(2);
    अचर int64_t il = tid/8; // 0...3
    अचर int64_t ib = tid%8; // 0...7
    dst_t * y = yy + i*QK_K + 32*ib + 4*il;
    अचर uint8_t  * q4 = x[ib].qs + 4*il;
    अचर भिन्न d = ggml_sycl_e8m0_to_fp32(x[ib].e);
    लागि (पूर्णाङ्क j = 0; j < 4; ++j) {
        y[j+ 0] = d * kvalues_mxfp4[q4[j] & 0xf]*0.5f;
        y[j+16] = d * kvalues_mxfp4[q4[j] >>  4]*0.5f;
    }
}

// Reordered MXFP4 ([qs...][e...], see ggml_sycl_reordered::block_q_t<MXFP4>): one work-item per block.
ढाँचा <typename dst_t>
स्थिर शून्य dequantize_block_mxfp4_reorder(अचर शून्य * __restrict__ vx, dst_t * __restrict__ yy, int64_t k,
                                           अचर sycl::nd_item<3> & item_ct1) {
    अचर int64_t ib = (int64_t) item_ct1.get_group(2) * WARP_SIZE + item_ct1.get_local_id(2);
    यदि (ib >= k / QK_MXFP4) {
        फर्काऊ;
    }

    अचर uint8_t * qs = (अचर uint8_t *) vx + ib * (QK_MXFP4 / 2);
    अचर भिन्न     d  = ggml_sycl_e8m0_to_fp32(((अचर uint8_t *) vx)[k / 2 + ib]) * 0.5f;
    dst_t *         y  = yy + ib * QK_MXFP4;

#pragma unroll
    लागि (पूर्णाङ्क j = 0; j < QK_MXFP4 / 2; ++j) {
        y[j]                = d * kvalues_mxfp4[qs[j] & 0xf];
        y[j + QK_MXFP4 / 2] = d * kvalues_mxfp4[qs[j] >> 4];
    }
}


ढाँचा <typename dst_t>
स्थिर शून्य dequantize_block_nvfp4(
        अचर शून्य * __restrict__ vx,
        dst_t * __restrict__ yy,
        अचर int64_t ne) {
    स्वतः          item_ct1 = sycl::ext::oneapi::this_work_item::get_nd_item<3>();
    अचर int64_t i        = item_ct1.get_group(2);
    अचर पूर्णाङ्क     tid      = item_ct1.get_local_id(2);

    अचर int64_t base = i * QK_NVFP4;
    यदि (base >= ne) {
        फर्काऊ;
    }

    अचर block_nvfp4 * x = (अचर block_nvfp4 *) vx;
    अचर block_nvfp4 & xb = x[i];

    अचर पूर्णाङ्क sub = tid / (QK_NVFP4_SUB / 2);
    अचर पूर्णाङ्क j = tid % (QK_NVFP4_SUB / 2);

    अचर भिन्न d = ggml_sycl_ue4m3_to_fp32(xb.d[sub]);
    अचर uint8_t q = xb.qs[sub * (QK_NVFP4_SUB / 2) + j];

    अचर int64_t y0 = base + sub * QK_NVFP4_SUB + j;
    अचर int64_t y1 = y0 + QK_NVFP4_SUB / 2;

    yy[y0] = ggml_sycl_cast<dst_t>(d * kvalues_mxfp4[q & 0x0F]);
    yy[y1] = ggml_sycl_cast<dst_t>(d * kvalues_mxfp4[q >> 4]);
}


#endif // GGML_SYCL_DEQUANTIZE_HPP
