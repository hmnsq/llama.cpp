#تعریف_کن GGML_COMMON_IMPL_CPP
#تعریف_کن GGML_COMMON_DECL_CPP

#ضمن "repack.h"

#ضمن "ggml-common.h"
#ضمن "ggml-cpu.h"
#ضمن "ggml-impl.h"
#ضمن "ime_kernels.h"

#ضمن <algorithm>
#ضمن <cassert>
#ضمن <cmath>
#ضمن <cstring>

// clang-format off
#إذا defined(__riscv)

#إذا !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#خطا "riscv v extension or v_intrinsic not enabled"
#وإلا
#ضمن <riscv_vector.h>
#پایان_اگر

#إذا !defined(__riscv_zfh)
#خطا "riscv zfh extension not enabled"
#پایان_اگر

#وإلا
#خطا "riscv not enabled in this build"
#پایان_اگر

#إذا defined(__GNUC__)
#رهنمود GCC diagnostic ignored "-Wcast-qual"
#رهنمود GCC diagnostic ignored "-Wunused-parameter"
#پایان_اگر

// clang-format on

قالب <صحيح K> constexpr صحيح QK_0() {
    إذا constexpr (K == 4) {
        أرجع QK4_0;
    }
    إذا constexpr (K == 8) {
        أرجع QK8_0;
    }
    أرجع -1;
}

قالب <صحيح K, صحيح N> بنية بلوک {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    uint8_t   qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

قالب <صحيح K, صحيح N> بنية بلوک_with_zp {
    ggml_half d[N];                         // deltas for N qK_1 blocks
    uint8_t   zp[N];                        // zero points for N qK_1 blocks
    uint8_t   qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_1 blocks
};

// control size
static_assert(حجم(بلوک<4, 16>) == 16 * حجم(ggml_half) + QK4_0 * 8, "wrong block<4,16> size/padding");
static_assert(حجم(بلوک_with_zp<4, 16>) == 16 * حجم(ggml_half) + QK4_0 * 8 + 16 * حجم(uint8_t),
              "wrong block_with_zp<4,16> size/padding");

static_assert(حجم(بلوک<8, 16>) == 16 * حجم(ggml_half) + QK4_0 * 16, "wrong block<8,16> size/padding");

static_assert(حجم(بلوک<4, 32>) == 32 * حجم(ggml_half) + QK4_0 * 16, "wrong block<4,32> size/padding");
static_assert(حجم(بلوک_with_zp<4, 32>) == 32 * حجم(ggml_half) + QK4_0 * 16 + 32 * حجم(uint8_t),
              "wrong block_with_zp<4,32> size/padding");

استفاده بلوک_q4_0x16 = بلوک<4, 16>;
استفاده بلوک_q4_1x16 = بلوک_with_zp<4, 16>;
استفاده بلوک_q8_0x16 = بلوک<8, 16>;

استفاده بلوک_q4_0x32 = بلوک<4, 32>;
استفاده بلوک_q4_1x32 = بلوک_with_zp<4, 32>;
استفاده بلوک_q8_0x32 = بلوک<8, 32>;

بنية بلوک_q4_0x32x256 {
    بلوک_q4_0x32 بلوکها[8];  // [f16 * 32 | i4 * 32 * 32] * 8
};

بنية بلوک_q4_1x32x256 {
    بلوک_q4_0x32 بلوکها[8];
    uint8_t       zps[32 * 8];
};

ساكن بلوک_q4_0x16 make_بلوک_q4_0x16(بلوک_q4_0 * درون, بیعلامت صحيح blck_اندازه_interleave) {
    بلوک_q4_0x16 بیرون;
    GGML_ASSERT(QK4_0 / blck_اندازه_interleave == 2);

    لكل (صحيح i = 0; i < 16; i++) {
        بیرون.d[i] = درون[i].d;
    }

    لكل (صحيح i = 0; i < 16; i++) {
        // [0, 15], in.d & 0x0F
        لكل (صحيح j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b8] ......... [b7 b15]
            بیرون.qs[i * QK4_0 / 4 + j] = (درون[i].qs[j] & 0x0F) | ((درون[i].qs[j + QK4_0 / 4] & 0x0F) << 4);
        }
    }

    لكل (صحيح i = 0; i < 16; i++) {
        // [16, 31], in.d & 0xF0
        لكل (صحيح j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            بیرون.qs[4 * QK4_0 + i * QK4_0 / 4 + j] = ((درون[i].qs[j] & 0xF0) >> 4) | (درون[i].qs[j + QK4_0 / 4] & 0xF0);
        }
    }

    أرجع بیرون;
}

ساكن بلوک_q4_1x16 make_بلوک_q4_1x16(بلوک_q4_1 * درون, بیعلامت صحيح blck_اندازه_interleave) {
    بلوک_q4_1x16 بیرون;
    GGML_ASSERT(QK4_1 / blck_اندازه_interleave == 2);

    لكل (صحيح i = 0; i < 16; i++) {
        عائم d   = GGML_FP16_مقصد_FP32(درون[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        عائم m   = GGML_FP16_مقصد_FP32(درون[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
        عائم mid = -std::nearbyintf(m / d);
        mid       = std::min(15.0f, std::max(0.0f, mid));
        بیرون.d[i]  = GGML_FP32_مقصد_FP16(d);
        بیرون.zp[i] = static_cast<uint8_t>(mid);
    }

    لكل (صحيح i = 0; i < 16; i++) {
        // [0, 15], in.d & 0x0F
        لكل (صحيح j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b8] ......... [b7 b15]
            بیرون.qs[i * QK4_1 / 4 + j] = (درون[i].qs[j] & 0x0F) | ((درون[i].qs[j + QK4_1 / 4] & 0x0F) << 4);
        }
    }

    لكل (صحيح i = 0; i < 16; i++) {
        // [16, 31], in.d & 0xF0
        لكل (صحيح j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            بیرون.qs[4 * QK4_1 + i * QK4_1 / 4 + j] = ((درون[i].qs[j] & 0xF0) >> 4) | (درون[i].qs[j + QK4_1 / 4] & 0xF0);
        }
    }

    أرجع بیرون;
}

ساكن صحيح repack_q4_0_مقصد_q4_0_16_bl(ggml_tensor *              t,
                                     صحيح                        interleave_بلوک,
                                     ثابت فراغ * GGML_RESTRICT داده,
                                     size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_بلوک == 16);

    constexpr صحيح nrows_interleaved = 16;

    بلوک_q4_0x16 *    dst = (بلوک_q4_0x16 *) t->داده;
    ثابت بلوک_q4_0 * src = (ثابت بلوک_q4_0 *) داده;
    بلوک_q4_0         dst_tmp[16];
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q4_0));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلوک_q4_0x16(dst_tmp, interleave_بلوک);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

ساكن صحيح repack_q4_1_مقصد_q4_1_16_bl(ggml_tensor *              t,
                                     صحيح                        interleave_بلوک,
                                     ثابت فراغ * GGML_RESTRICT داده,
                                     size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_بلوک == 16);

    constexpr صحيح nrows_interleaved = 16;

    بلوک_q4_1x16 *    dst = (بلوک_q4_1x16 *) t->داده;
    ثابت بلوک_q4_1 * src = (ثابت بلوک_q4_1 *) داده;
    بلوک_q4_1         dst_tmp[16];
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q4_1));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلوک_q4_1x16(dst_tmp, interleave_بلوک);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

ساكن مضمن فراغ get_scale_min_k4(صحيح                           j,
                                    ثابت uint8_t * GGML_RESTRICT q,
                                    uint8_t * GGML_RESTRICT       d,
                                    uint8_t * GGML_RESTRICT       m) {
    إذا (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } وإلا {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

ساكن صحيح repack_q4_k_مقصد_q4_1_16_bl(ggml_tensor *              t,
                                     صحيح                        interleave_بلوک,
                                     ثابت فراغ * GGML_RESTRICT داده,
                                     size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_بلوک == 16);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr صحيح nrows_interleaved = 16;

    بلوک_q4_1x16 *    dst = (بلوک_q4_1x16 *) t->داده;
    ثابت بلوک_q4_K * src = (ثابت بلوک_q4_K *) داده;
    بلوک_q4_1         dst_tmp[16];
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK_K;

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح j = 0; j < 8; j++) {
                لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    ثابت عائم d = GGML_FP16_مقصد_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    ثابت عائم min =
                        GGML_FP16_مقصد_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);
                    ثابت عائم d1 = d * sc;
                    ثابت عائم m1 = min * m;

                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_مقصد_FP16(d1);
                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_مقصد_FP16(-m1);
                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    ثابت uint8_t * q                                  = src[x + i * nblocks].qs + (j / 2) * QK4_1;
                    إذا (j % 2 == 0) {
                        لكل (صحيح ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } وإلا {
                        لكل (صحيح ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }
                }
                *dst++ = make_بلوک_q4_1x16(dst_tmp, interleave_بلوک);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

ساكن بلوک_q4_0x32 make_بلوک_q4_0x32(بلوک_q4_0 * درون, بیعلامت صحيح blck_اندازه_interleave) {
    بلوک_q4_0x32 بیرون;
    أكد(QK4_0 / blck_اندازه_interleave == 1);
    GGML_استفادهنشده(blck_اندازه_interleave);

    لكل (صحيح i = 0; i < 32; i++) {
        بیرون.d[i] = درون[i].d;
    }

    لكل (صحيح i = 0; i < 32; i++) {
        // [0, 15], in.d & 0x0F
        لكل (صحيح j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b1] .........  [b14 b15]
            بیرون.qs[i * QK4_0 / 2 + j] = (درون[i].qs[j * 2] & 0x0F) | ((درون[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
    }

    لكل (صحيح i = 0; i < 32; i++) {
        // [16, 31], in.d & 0xF0
        لكل (صحيح j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b17] ......... [b30 b31]
            بیرون.qs[i * QK4_0 / 2 + QK4_0 / 4 + j] = ((درون[i].qs[j * 2] & 0xF0) >> 4) | (درون[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    أرجع بیرون;
}

ساكن بلوک_q4_1x32 make_بلوک_q4_1x32(بلوک_q4_1 * درون, بیعلامت صحيح blck_اندازه_interleave) {
    بلوک_q4_1x32 بیرون;
    GGML_ASSERT(QK4_1 / blck_اندازه_interleave == 1);
    GGML_استفادهنشده(blck_اندازه_interleave);

    لكل (صحيح i = 0; i < 32; i++) {
        عائم d   = GGML_FP16_مقصد_FP32(درون[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        عائم m   = GGML_FP16_مقصد_FP32(درون[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
        عائم mid = -std::nearbyintf(m / d);
        mid       = std::min(15.0f, std::max(0.0f, mid));
        بیرون.d[i]  = GGML_FP32_مقصد_FP16(d);
        بیرون.zp[i] = static_cast<uint8_t>(mid);
    }

    لكل (صحيح i = 0; i < 32; i++) {
        // [0, 15], in.d & 0x0F
        لكل (صحيح j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b1] ......... [b14 b15]
            بیرون.qs[i * QK4_1 / 2 + j] = (درون[i].qs[j * 2] & 0x0F) | ((درون[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
    }

    لكل (صحيح i = 0; i < 32; i++) {
        // [16, 31], in.d & 0xF0
        لكل (صحيح j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            بیرون.qs[i * QK4_1 / 2 + QK4_1 / 4 + j] = ((درون[i].qs[j * 2] & 0xF0) >> 4) | (درون[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    أرجع بیرون;
}

ساكن بلوک_q8_0x32 make_بلوک_q8_0x32(بلوک_q8_0 * درون, بیعلامت صحيح blck_اندازه_interleave) {
    بلوک_q8_0x32 بیرون;
    GGML_ASSERT(QK8_0 / blck_اندازه_interleave == 1);
    GGML_استفادهنشده(blck_اندازه_interleave);

    لكل (صحيح i = 0; i < 32; i++) {
        بیرون.d[i] = درون[i].d;
    }

    لكل (صحيح i = 0; i < 32; i++) {
        کپی_حافظه(بیرون.qs + i * QK8_0, درون[i].qs, QK8_0);
    }

    أرجع بیرون;
}

// IME1: interleave 16 q8_0 rows so a plain vle8 sequence in the i8i8 kernel lands weights in the
// vmadot group/parity layout. Mirrors make_block_q4_0x16 but stores full int8 (no nibble packing).
// qs (512B) = [INNER step 0: reg0..reg7][INNER step 1: reg0..reg7], each reg 32B holding 4 columns
// x (even=K-first-half / odd=K-second-half). Column col -> acc=col/4, cgrp=col%4.
ساكن بلوک_q8_0x16 make_بلوک_q8_0x16(بلوک_q8_0 * درون, بیعلامت صحيح blck_اندازه_interleave) {
    بلوک_q8_0x16 بیرون;
    GGML_ASSERT(QK8_0 / blck_اندازه_interleave == 2);
    GGML_استفادهنشده(blck_اندازه_interleave);

    لكل (صحيح i = 0; i < 16; i++) {
        بیرون.d[i] = درون[i].d;
    }

    memset(بیرون.qs, 0, حجم(بیرون.qs));
    لكل (صحيح ستون = 0; ستون < 16; ستون++) {
        ثابت صحيح      acc  = ستون / 4;
        ثابت صحيح      cgrp = ستون % 4;
        ثابت int8_t * q    = درون[ستون].qs;
        لكل (صحيح s = 0; s < 2; s++) {
            ثابت صحيح پایه   = s * 16;
            uint8_t * reg_lo = بیرون.qs + (s * 8 + acc)     * 32;
            uint8_t * reg_hi = بیرون.qs + (s * 8 + acc + 4) * 32;
            لكل (صحيح i = 0; i < 4; i++) {
                reg_lo[(2 * cgrp)     * 4 + i] = (uint8_t) q[پایه + 0  + i];
                reg_lo[(2 * cgrp + 1) * 4 + i] = (uint8_t) q[پایه + 4  + i];
                reg_hi[(2 * cgrp)     * 4 + i] = (uint8_t) q[پایه + 8  + i];
                reg_hi[(2 * cgrp + 1) * 4 + i] = (uint8_t) q[پایه + 12 + i];
            }
        }
    }
    أرجع بیرون;
}

ساكن صحيح repack_q8_0_مقصد_q8_0_16_bl_ref(ggml_tensor *              t,
                                         صحيح                        interleave_بلوک,
                                         ثابت فراغ * GGML_RESTRICT داده,
                                         size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_بلوک == 16);

    constexpr صحيح nrows_interleaved = 16;

    بلوک_q8_0x16 *    dst = (بلوک_q8_0x16 *) t->داده;
    ثابت بلوک_q8_0 * src = (ثابت بلوک_q8_0 *) داده;
    بلوک_q8_0         dst_tmp[16];
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q8_0));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK8_0 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلوک_q8_0x16(dst_tmp, interleave_بلوک);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

ساكن صحيح repack_q2_k_مقصد_q2_k_32_bl(ggml_tensor *              t,
                                     صحيح                        interleave_بلوک,
                                     ثابت فراغ * GGML_RESTRICT داده,
                                     size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_بلوک == 32);
    GGML_ASSERT(QK_K == 256);

    constexpr صحيح nrows_interleaved = 32;

    ثابت بلوک_q2_K * src = (ثابت بلوک_q2_K *) داده;

    خودکار * dst = (spacemit_kernels::nrow_بلوک_q2_k<32> *) t->داده;

    صحيح nrow    = ggml_nrows(t);
    صحيح nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q2_K));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        أرجع -1;
    }

    uint8_t qs_aux[256] = { 0 };
    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                ثابت بلوک_q2_K * src_بلوک = &src[(b + i) * nblocks + x];

                // scale for [16, N]
                لكل (صحيح j = 0; j < 16; j++) {
                    خودکار zp_aux = (dst->scales[j * nrows_interleaved + i]) & 0xF0;

                    dst->scales[j * nrows_interleaved + i] = (src_بلوک->scales[j] & 0x0F) | zp_aux;
                }

                // zp for [N, 16]
                لكل (صحيح j = 0; j < 16; j++) {
                    خودکار scale_aux = (dst->scales[16 * i + j]) & 0x0F;

                    dst->scales[16 * i + j] = (src_بلوک->scales[j] & 0xF0) | scale_aux;
                }

                لكل (صحيح k = 0; k < 4; k++) {
                    لكل (صحيح j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j] = (src_بلوک->qs[j] >> (2 * k)) & 0x03;
                    }
                }

                لكل (صحيح k = 0; k < 4; k++) {
                    لكل (صحيح j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j + 128] = (src_بلوک->qs[j + 32] >> (2 * k)) & 0x03;
                    }
                }

                // from nrows_interleaved * [2 * 32byte]
                // to 4 * [nrows_interleaved * 16byte]
                لكل (صحيح k = 0; k < 4; k++) {
                    لكل (صحيح j = 0; j < 16; j++) {
                        uint8_t qs0  = qs_aux[j + k * 64];
                        uint8_t qs16 = qs_aux[j + 16 + k * 64];
                        uint8_t qs32 = qs_aux[j + 32 + k * 64];
                        uint8_t qs48 = qs_aux[j + 48 + k * 64];

                        dst->qs[(k * nrows_interleaved + i) * 16 + j] =
                            (qs0 & 0x03) | ((qs16 & 0x03) << 2) | ((qs32 & 0x03) << 4) | ((qs48 & 0x03) << 6);
                    }
                }

                dst->scales16[i] = src_بلوک->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
                dst->zeros16[i]  = src_بلوک->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
            }
            dst++;
        }
    }

    أرجع 0;
}

ساكن صحيح repack_q3_k_مقصد_q3_k_32_bl(ggml_tensor *              t,
                                     صحيح                        interleave_بلوک,
                                     ثابت فراغ * GGML_RESTRICT داده,
                                     size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q3_K);
    GGML_ASSERT(interleave_بلوک == 32);
    GGML_ASSERT(QK_K == 256);

    constexpr صحيح nrows_interleaved = 32;

    ثابت uint32_t kmask1 = 0x03030303;
    ثابت uint32_t kmask2 = 0x0f0f0f0f;

    ثابت بلوک_q3_K * src = (ثابت بلوک_q3_K *) داده;

    خودکار * dst = (spacemit_kernels::nrow_بلوک_q3_k<32> *) t->داده;

    صحيح nrow    = ggml_nrows(t);
    صحيح nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q3_K));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        أرجع -1;
    }

    uint32_t b_scale_aux[4] = { 0 };
    uint8_t  qs_aux[256]    = { 0 };

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                ثابت بلوک_q3_K * src_بلوک = &src[(b + i) * nblocks + x];

                uint32_t * auxs  = b_scale_aux;
                int8_t *   scale = (int8_t *) auxs;
                کپی_حافظه(auxs, src_بلوک->scales, 12);

                uint32_t tmp = auxs[2];
                auxs[2]      = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                auxs[3]      = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                auxs[0]      = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                auxs[1]      = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

                لكل (صحيح j = 0; j < 16; j++) {
                    dst->scales[j * nrows_interleaved + i] = scale[j] - 32;
                }

                لكل (صحيح k = 0; k < 4; k++) {
                    لكل (صحيح j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j] = (src_بلوک->qs[j] >> (2 * k)) & 0x03;
                    }
                }

                لكل (صحيح k = 0; k < 4; k++) {
                    لكل (صحيح j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j + 128] = (src_بلوک->qs[j + 32] >> (2 * k)) & 0x03;
                    }
                }

                // from nrows_interleaved * [2 * 32byte]
                // to 4 * [nrows_interleaved * 16byte]
                لكل (صحيح k = 0; k < 4; k++) {
                    لكل (صحيح j = 0; j < 16; j++) {
                        uint8_t qs0  = qs_aux[j + k * 64];
                        uint8_t qs16 = qs_aux[j + 16 + k * 64];
                        uint8_t qs32 = qs_aux[j + 32 + k * 64];
                        uint8_t qs48 = qs_aux[j + 48 + k * 64];

                        dst->qs[(k * nrows_interleaved + i) * 16 + j] =
                            (qs0 & 0x03) | ((qs16 & 0x03) << 2) | ((qs32 & 0x03) << 4) | ((qs48 & 0x03) << 6);
                    }
                }

                //memcpy(dst->hmask + i * 32, src_block->hmask, 32);

                // from nrows_interleaved * [32byte]
                // to 16 * [nrows_interleaved * uint16_t]
                uint16_t * dst_ماسک = ((uint16_t *) dst->hmask) + i;
                لكل (صحيح j = 0; j < 16; j++, dst_ماسک += nrows_interleaved) {
                    uint8_t   b_لغزش    = j / 2;
                    uint8_t * b_ماسک_ستون = (uint8_t *) (src_بلوک->hmask + (j % 2) * 16);
                    // b0 - b15
                    uint16_t  msk_بیرون_0  = 0;

                    لكل (صحيح k = 0; k < 8; k++) {
                        msk_بیرون_0 |= (uint16_t) ((b_ماسک_ستون[k] >> b_لغزش) & 0x01) << k;
                    }
                    لكل (صحيح k = 8; k < 16; k++) {
                        msk_بیرون_0 |= (uint16_t) ((b_ماسک_ستون[k] >> b_لغزش) & 0x01) << k;
                    }

                    dst_ماسک[0] = msk_بیرون_0;
                }

                dst->scales16[i] = src_بلوک->d;
            }

            dst++;
        }
    }

    أرجع 0;
}

ساكن صحيح repack_q4_0_مقصد_q4_0_32_bl_ref(ggml_tensor *              t,
                                         صحيح                        interleave_بلوک,
                                         ثابت فراغ * GGML_RESTRICT داده,
                                         size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_بلوک == 32);  //استفادهنشده

    constexpr صحيح nrows_interleaved = 32;

    بلوک_q4_0x32 *    dst = (بلوک_q4_0x32 *) t->داده;
    ثابت بلوک_q4_0 * src = (ثابت بلوک_q4_0 *) داده;
    بلوک_q4_0         dst_tmp[32];
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q4_0));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلوک_q4_0x32(dst_tmp, interleave_بلوک);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

ساكن صحيح repack_q4_0_مقصد_q4_0_256_32_bl_ref(ggml_tensor *              t,
                                             صحيح                        interleave_بلوک,
                                             ثابت فراغ * GGML_RESTRICT داده,
                                             size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_بلوک == 32);  //استفادهنشده

    constexpr صحيح nrows_interleaved = 32;

    بلوک_q4_0x32x256 * dst = (بلوک_q4_0x32x256 *) t->داده;
    ثابت بلوک_q4_0 *  src = (ثابت بلوک_q4_0 *) داده;
    بلوک_q4_0          dst_tmp[32];
    صحيح                 nrow    = ggml_nrows(t);
    صحيح                 nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q4_0));
    GGML_ASSERT(nblocks % 8 == 0);  // for 256-block interleaving
    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x += 8) {
            لكل (صحيح j = 0; j < 8; j++) {
                لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                    dst_tmp[i] = src[x + j + i * nblocks];
                }
                dst->بلوکها[j] = make_بلوک_q4_0x32(dst_tmp, interleave_بلوک);
            }
            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

ساكن صحيح repack_q4_0_مقصد_q4_1_256_32_bl_ref(ggml_tensor *              t,
                                             صحيح                        interleave_بلوک,
                                             ثابت فراغ * GGML_RESTRICT داده,
                                             size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_بلوک == 32);  //استفادهنشده

    constexpr صحيح nrows_interleaved = 32;

    بلوک_q4_1x32x256 * dst = (بلوک_q4_1x32x256 *) t->داده;
    ثابت بلوک_q4_1 *  src = (ثابت بلوک_q4_1 *) داده;
    بلوک_q4_1          dst_tmp[32];
    صحيح                 nrow    = ggml_nrows(t);
    صحيح                 nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q4_1));
    GGML_ASSERT(nblocks % 8 == 0);  // for 256-block interleaving
    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x += 8) {
            لكل (صحيح j = 0; j < 8; j++) {
                لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                    dst_tmp[i] = src[x + j + i * nblocks];
                }

                بلوک_q4_0x32 * dst_بلوک = &dst->بلوکها[j];
                uint8_t *       dst_zp    = dst->zps + j * nrows_interleaved;

                لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                    عائم d   = GGML_FP16_مقصد_FP32(dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    عائم m   = GGML_FP16_مقصد_FP32(dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
                    عائم mid = -std::nearbyintf(m / d);
                    mid       = std::min(15.0f, std::max(0.0f, mid));

                    dst_بلوک->d[i] = GGML_FP32_مقصد_FP16(d);
                    dst_zp[i]       = static_cast<uint8_t>(mid);
                }

                لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                    لكل (صحيح k = 0; k < QK4_1 / 4; k++) {
                        dst_بلوک->qs[i * QK4_1 / 2 + k] =
                            (dst_tmp[i].qs[k * 2] & 0x0F) | ((dst_tmp[i].qs[k * 2 + 1] & 0x0F) << 4);
                    }
                }

                لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                    لكل (صحيح k = 0; k < QK4_1 / 4; k++) {
                        dst_بلوک->qs[i * QK4_1 / 2 + QK4_1 / 4 + k] =
                            ((dst_tmp[i].qs[k * 2] & 0xF0) >> 4) | (dst_tmp[i].qs[k * 2 + 1] & 0xF0);
                    }
                }
            }
            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

// RVV optimized version of repack_q4_0_to_q4_0_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes nibble repack.
ساكن صحيح repack_q4_0_مقصد_q4_0_32_bl(ggml_tensor *              t,
                                     صحيح                        interleave_بلوک,
                                     ثابت فراغ * GGML_RESTRICT داده,
                                     size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_بلوک == 32);

    constexpr صحيح nrows_interleaved = 32;
    constexpr صحيح qs_بایتها          = QK4_0 / 2;  // 16

    بلوک_q4_0x32 *    dst     = (بلوک_q4_0x32 *) t->داده;
    ثابت بلوک_q4_0 * src     = (ثابت بلوک_q4_0 *) داده;
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q4_0));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        أرجع -1;
    }

    ثابت ptrdiff_t ردیف_stride = (ptrdiff_t) nblocks * حجم(بلوک_q4_0);

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            ثابت بلوک_q4_0 * ستون_src = src + x;

            // --- 1) Gather 32 scale values (ggml_half d) with stride load ---
            // d is at offset 0 of each block_q4_0, stride between rows = row_stride
            {
                ثابت uint8_t * d_پایه    = (ثابت uint8_t *) &ستون_src->d;
                ggml_half *     d_dst     = dst->d;
                size_t          مانده = 32;
                size_t          جابه_جایی    = 0;
                طالما (مانده > 0) {
                    size_t      vl = __riscv_vsetvl_e16m1(مانده);
                    vuint16m1_t vd =
                        __riscv_vlse16_v_u16m1((ثابت uint16_t *) (d_پایه + جابه_جایی * ردیف_stride), ردیف_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + جابه_جایی), vd, vl);
                    جابه_جایی += vl;
                    مانده -= vl;
                }
            }

            // --- 2) Nibble repack qs for each of the 32 rows ---
            // For each row i:
            //   src qs[16]: [b0|b16] [b1|b17] ... [b15|b31]  (lo nibble = b_j, hi nibble = b_{j+16})
            //   dst qs low  8B: (qs[2j] & 0x0F) | ((qs[2j+1] & 0x0F) << 4)  for j=0..7
            //   dst qs high 8B: ((qs[2j] >> 4))  | (qs[2j+1] & 0xF0)         for j=0..7
            {
                ثابت size_t vl8 = __riscv_vsetvl_e8m1(8);
                لكل (صحيح i = 0; i < 32; i++) {
                    ثابت uint8_t * sq = ستون_src[i * nblocks].qs;
                    uint8_t *       dq = dst->qs + i * qs_بایتها;

                    // stride-2 load to separate even/odd bytes
                    vuint8m1_t v_even = __riscv_vlse8_v_u8m1(sq, 2, vl8);      // qs[0], qs[2], ..., qs[14]
                    vuint8m1_t v_odd  = __riscv_vlse8_v_u8m1(sq + 1, 2, vl8);  // qs[1], qs[3], ..., qs[15]

                    // low nibble part: (even & 0x0F) | ((odd & 0x0F) << 4)
                    vuint8m1_t v_even_lo = __riscv_vand_vx_u8m1(v_even, 0x0F, vl8);
                    vuint8m1_t v_odd_lo  = __riscv_vand_vx_u8m1(v_odd, 0x0F, vl8);
                    vuint8m1_t v_lo      = __riscv_vor_vv_u8m1(v_even_lo, __riscv_vsll_vx_u8m1(v_odd_lo, 4, vl8), vl8);

                    // high nibble part: (even >> 4) | (odd & 0xF0)
                    vuint8m1_t v_even_hi = __riscv_vsrl_vx_u8m1(v_even, 4, vl8);
                    vuint8m1_t v_odd_hi  = __riscv_vand_vx_u8m1(v_odd, 0xF0, vl8);
                    vuint8m1_t v_hi      = __riscv_vor_vv_u8m1(v_even_hi, v_odd_hi, vl8);

                    __riscv_vse8_v_u8m1(dq, v_lo, vl8);
                    __riscv_vse8_v_u8m1(dq + 8, v_hi, vl8);
                }
            }

            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

ساكن صحيح repack_q4_1_مقصد_q4_1_32_bl_ref(ggml_tensor *              t,
                                         صحيح                        interleave_بلوک,
                                         ثابت فراغ * GGML_RESTRICT داده,
                                         size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_بلوک == 32);  //استفادهنشده

    constexpr صحيح nrows_interleaved = 32;

    بلوک_q4_1x32 *    dst = (بلوک_q4_1x32 *) t->داده;
    ثابت بلوک_q4_1 * src = (ثابت بلوک_q4_1 *) داده;
    بلوک_q4_1         dst_tmp[32];
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q4_1));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_بلوک_q4_1x32(dst_tmp, interleave_بلوک);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

// RVV optimized version of repack_q4_1_to_q4_1_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes nibble repack + zp computation.
ساكن صحيح repack_q4_1_مقصد_q4_1_32_bl(ggml_tensor *              t,
                                     صحيح                        interleave_بلوک,
                                     ثابت فراغ * GGML_RESTRICT داده,
                                     size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_بلوک == 32);

    constexpr صحيح nrows_interleaved = 32;
    constexpr صحيح qs_بایتها          = QK4_1 / 2;  // 16

    بلوک_q4_1x32 *    dst     = (بلوک_q4_1x32 *) t->داده;
    ثابت بلوک_q4_1 * src     = (ثابت بلوک_q4_1 *) داده;
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q4_1));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        أرجع -1;
    }

    ثابت ptrdiff_t ردیف_stride = (ptrdiff_t) nblocks * حجم(بلوک_q4_1);

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            ثابت بلوک_q4_1 * ستون_src = src + x;

            // --- 1) Gather d and m, compute zp = clamp(nearbyint(-m/d), 0, 15) ---
            // block_q4_1 layout: [d(f16), m(f16), qs[16]]
            // d is at byte offset 0, m is at byte offset 2 from each block start
            {
                ثابت uint8_t * dm_پایه   = (ثابت uint8_t *) &ستون_src->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
                ggml_half *     d_dst     = dst->d;
                uint8_t *       zp_dst    = dst->zp;
                size_t          مانده = 32;
                size_t          جابه_جایی    = 0;
                طالما (مانده > 0) {
                    size_t vl = __riscv_vsetvl_e16m1(مانده);

                    // stride load d (f16) from each row
                    vuint16m1_t vd_raw =
                        __riscv_vlse16_v_u16m1((ثابت uint16_t *) (dm_پایه + جابه_جایی * ردیف_stride), ردیف_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + جابه_جایی), vd_raw, vl);

                    // stride load m (f16) from each row (offset +2 bytes from d)
                    vuint16m1_t vm_raw =
                        __riscv_vlse16_v_u16m1((ثابت uint16_t *) (dm_پایه + 2 + جابه_جایی * ردیف_stride), ردیف_stride, vl);

                    // convert to f32 for zp computation: zp = nearbyint(-m / d)
                    vfloat16m1_t vd_f16 = __riscv_vreinterpret_v_u16m1_f16m1(vd_raw);
                    vfloat16m1_t vm_f16 = __riscv_vreinterpret_v_u16m1_f16m1(vm_raw);

                    // -m / d in f16 directly (SpaceMIT X60 supports f16 arithmetic)
                    vfloat16m1_t v_neg_m = __riscv_vfneg_v_f16m1(vm_f16, vl);
                    vfloat16m1_t v_نسبت = __riscv_vfdiv_vv_f16m1(v_neg_m, vd_f16, vl);

                    // Convert to f32 for nearbyint, then clamp
                    vfloat32m2_t v_نسبت_f32 = __riscv_vfwcvt_f_f_v_f32m2(v_نسبت, vl);

                    // Use integer rounding: convert f32 -> int (rounds to nearest)
                    vint32m2_t v_zp_i32 = __riscv_vfcvt_x_f_v_i32m2(v_نسبت_f32, vl);

                    // clamp to [0, 15]
                    v_zp_i32 = __riscv_vmax_vx_i32m2(v_zp_i32, 0, vl);
                    v_zp_i32 = __riscv_vmin_vx_i32m2(v_zp_i32, 15, vl);

                    // narrow i32 -> u8
                    vint16m1_t  v_zp_i16 = __riscv_vncvt_x_x_w_i16m1(v_zp_i32, vl);
                    vint8mf2_t  v_zp_i8  = __riscv_vncvt_x_x_w_i8mf2(v_zp_i16, vl);
                    vuint8mf2_t v_zp_u8  = __riscv_vreinterpret_v_i8mf2_u8mf2(v_zp_i8);
                    __riscv_vse8_v_u8mf2(zp_dst + جابه_جایی, v_zp_u8, vl);

                    جابه_جایی += vl;
                    مانده -= vl;
                }
            }

            // --- 2) Nibble repack qs for each of the 32 rows ---
            {
                ثابت size_t vl8 = __riscv_vsetvl_e8m1(8);
                لكل (صحيح i = 0; i < 32; i++) {
                    ثابت uint8_t * sq = ستون_src[i * nblocks].qs;
                    uint8_t *       dq = dst->qs + i * qs_بایتها;

                    // stride-2 load to separate even/odd bytes
                    vuint8m1_t v_even = __riscv_vlse8_v_u8m1(sq, 2, vl8);
                    vuint8m1_t v_odd  = __riscv_vlse8_v_u8m1(sq + 1, 2, vl8);

                    // low nibble part: (even & 0x0F) | ((odd & 0x0F) << 4)
                    vuint8m1_t v_even_lo = __riscv_vand_vx_u8m1(v_even, 0x0F, vl8);
                    vuint8m1_t v_odd_lo  = __riscv_vand_vx_u8m1(v_odd, 0x0F, vl8);
                    vuint8m1_t v_lo      = __riscv_vor_vv_u8m1(v_even_lo, __riscv_vsll_vx_u8m1(v_odd_lo, 4, vl8), vl8);

                    // high nibble part: (even >> 4) | (odd & 0xF0)
                    vuint8m1_t v_even_hi = __riscv_vsrl_vx_u8m1(v_even, 4, vl8);
                    vuint8m1_t v_odd_hi  = __riscv_vand_vx_u8m1(v_odd, 0xF0, vl8);
                    vuint8m1_t v_hi      = __riscv_vor_vv_u8m1(v_even_hi, v_odd_hi, vl8);

                    __riscv_vse8_v_u8m1(dq, v_lo, vl8);
                    __riscv_vse8_v_u8m1(dq + 8, v_hi, vl8);
                }
            }

            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

ساكن صحيح repack_q4_k_مقصد_q4_1_32_bl(ggml_tensor *              t,
                                     صحيح                        interleave_بلوک,
                                     ثابت فراغ * GGML_RESTRICT داده,
                                     size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_بلوک == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr صحيح nrows_interleaved = 32;

    بلوک_q4_1x32 *    dst = (بلوک_q4_1x32 *) t->داده;
    ثابت بلوک_q4_K * src = (ثابت بلوک_q4_K *) داده;
    بلوک_q4_1         dst_tmp[32];
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK_K;

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح j = 0; j < 8; j++) {
                لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    ثابت عائم d = GGML_FP16_مقصد_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    ثابت عائم min =
                        GGML_FP16_مقصد_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);
                    ثابت عائم d1 = d * sc;
                    ثابت عائم m1 = min * m;

                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_مقصد_FP16(d1);
                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_مقصد_FP16(-m1);
                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    ثابت uint8_t * q                                  = src[x + i * nblocks].qs + (j / 2) * QK4_1;
                    إذا (j % 2 == 0) {
                        لكل (صحيح ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } وإلا {
                        لكل (صحيح ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }
                }
                *dst++ = make_بلوک_q4_1x32(dst_tmp, interleave_بلوک);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

ساكن صحيح repack_q6_k_مقصد_q8_0_32_bl_ref(ggml_tensor *              t,
                                         صحيح                        interleave_بلوک,
                                         ثابت فراغ * GGML_RESTRICT داده,
                                         size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_بلوک == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr صحيح nrows_interleaved = 32;

    بلوک_q8_0x32 *    dst = (بلوک_q8_0x32 *) t->داده;
    ثابت بلوک_q6_K * src = (ثابت بلوک_q6_K *) داده;
    بلوک_q8_0         dst_tmp[32];
    int8_t             aux8[QK4_1];
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK_K;

    إذا (t->ne[0] % QK_K != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        int64_t nrow_real = std::min((int64_t) nrow - b, (int64_t) nrows_interleaved);
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح bi = 0; bi < 8; bi++) {
                صحيح i = 0;
                لكل (; i < nrow_real; i++) {
                    ثابت uint8_t * q4     = src[x + i * nblocks].ql;
                    ثابت uint8_t * qh     = src[x + i * nblocks].qh;
                    ثابت int8_t *  scales = src[x + i * nblocks].scales;
                    عائم           d      = GGML_FP16_مقصد_FP32(src[x + i * nblocks].d);

                    q4 += 64 * (bi / 4);
                    qh += 32 * (bi / 4);
                    int8_t * GGML_RESTRICT a = aux8;

                    int8_t bi_idx = bi % 4;

                    إذا (bi_idx == 0) {
                        لكل (صحيح l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        }
                    } وإلا إذا (bi_idx == 1) {
                        لكل (صحيح l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        }
                    } وإلا إذا (bi_idx == 2) {
                        لكل (صحيح l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        }
                    } وإلا إذا (bi_idx == 3) {
                        لكل (صحيح l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        }
                    }
                    a = aux8;

                    عائم a_max_abs = 0.0f;
                    عائم scale_0   = scales[bi * 2 + 0] * d;
                    عائم scale_1   = scales[bi * 2 + 1] * d;
                    لكل (صحيح l = 0; l < 16; ++l) {
                        a_max_abs = std::max(a_max_abs, std::قدر_مطلق(a[l] * scale_0));
                    }

                    لكل (صحيح l = 16; l < 32; ++l) {
                        a_max_abs = std::max(a_max_abs, std::قدر_مطلق(a[l] * scale_1));
                    }

                    عائم reflect_scale   = a_max_abs / ((1 << 7) - 1);
                    عائم reflect_scale_0 = scale_0 / reflect_scale;
                    عائم reflect_scale_1 = scale_1 / reflect_scale;

                    لكل (صحيح l = 0; l < 16; ++l) {
                        عائم a_موقت = std::clamp(std::nearbyintf(a[l] * reflect_scale_0), -128.0f, 127.0f);
                        a[l]         = (int8_t) (a_موقت);
                    }

                    لكل (صحيح l = 16; l < 32; ++l) {
                        عائم a_موقت = std::clamp(std::nearbyintf(a[l] * reflect_scale_1), -128.0f, 127.0f);
                        a[l]         = (int8_t) (a_موقت);
                    }

                    dst_tmp[i].d = GGML_FP32_مقصد_FP16(reflect_scale);

                    کپی_حافظه(dst_tmp[i].qs, a, 32 * حجم(int8_t));
                }

                لكل (; i < nrows_interleaved; i++) {
                    memset(&dst_tmp[i], 0, حجم(بلوک_q8_0));
                }

                *dst++ = make_بلوک_q8_0x32(dst_tmp, interleave_بلوک);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

// RVV optimized version of repack_q6_k_to_q8_0_32_bl
// Vectorizes the Q6_K dequant -> requant pipeline using RVV intrinsics.
// For each sub-block (bi), dequant 32 Q6_K values to int6 -> apply two sub-block scales ->
// find max abs -> compute reflect_scale -> requant to int8 -> gather d with stride load.
ساكن صحيح repack_q6_k_مقصد_q8_0_32_bl(ggml_tensor *              t,
                                     صحيح                        interleave_بلوک,
                                     ثابت فراغ * GGML_RESTRICT داده,
                                     size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_بلوک == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr صحيح nrows_interleaved = 32;

    بلوک_q8_0x32 *    dst     = (بلوک_q8_0x32 *) t->داده;
    ثابت بلوک_q6_K * src     = (ثابت بلوک_q6_K *) داده;
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK_K;

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        أرجع -1;
    }

    ثابت ptrdiff_t ردیف_stride = (ptrdiff_t) nblocks * حجم(بلوک_q6_K);

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح bi = 0; bi < 8; bi++) {
                // --- 1) Gather 32 d values with stride load ---
                // We need to compute reflect_scale per row first, so gather d later.
                // Process each row: dequant Q6_K sub-block -> requant to Q8_0
                لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                    ثابت بلوک_q6_K * src_blk = &src[x + i * nblocks];
                    ثابت uint8_t *    q4      = src_blk->ql + 64 * (bi / 4);
                    ثابت uint8_t *    qh      = src_blk->qh + 32 * (bi / 4);
                    ثابت int8_t *     scales  = src_blk->scales;
                    عائم              d       = GGML_FP16_مقصد_FP32(src_blk->d);

                    int8_t bi_idx = bi % 4;

                    // --- Dequant 32 Q6_K values to int6 (range [-32, 31]) using RVV ---
                    // vl = 32 for e8m2 (VLEN=256) or loop for smaller VLEN
                    ثابت size_t vl16 = __riscv_vsetvl_e8m1(16);

                    vint8m1_t va_lo, va_hi;  // 16 elements each

                    إذا (bi_idx == 0) {
                        // a[l] = (q4[l] & 0xF) | (((qh[l] >> 0) & 3) << 4) - 32
                        vuint8m1_t vq4_lo = __riscv_vle8_v_u8m1(q4, vl16);
                        vuint8m1_t vq4_hi = __riscv_vle8_v_u8m1(q4 + 16, vl16);
                        vuint8m1_t vqh_lo = __riscv_vle8_v_u8m1(qh, vl16);
                        vuint8m1_t vqh_hi = __riscv_vle8_v_u8m1(qh + 16, vl16);

                        vuint8m1_t vlo4_lo = __riscv_vand_vx_u8m1(vq4_lo, 0x0F, vl16);
                        vuint8m1_t vlo4_hi = __riscv_vand_vx_u8m1(vq4_hi, 0x0F, vl16);
                        vuint8m1_t vh_lo   = __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(vqh_lo, 0x03, vl16), 4, vl16);
                        vuint8m1_t vh_hi   = __riscv_vsll_vx_u8m1(__riscv_vand_vx_u8m1(vqh_hi, 0x03, vl16), 4, vl16);

                        vuint8m1_t vcomb_lo = __riscv_vor_vv_u8m1(vlo4_lo, vh_lo, vl16);
                        vuint8m1_t vcomb_hi = __riscv_vor_vv_u8m1(vlo4_hi, vh_hi, vl16);

                        va_lo = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_lo), 32, vl16);
                        va_hi = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_hi), 32, vl16);
                    } وإلا إذا (bi_idx == 1) {
                        // a[l] = (q4[l+32] & 0xF) | (((qh[l] >> 2) & 3) << 4) - 32
                        vuint8m1_t vq4_lo = __riscv_vle8_v_u8m1(q4 + 32, vl16);
                        vuint8m1_t vq4_hi = __riscv_vle8_v_u8m1(q4 + 48, vl16);
                        vuint8m1_t vqh_lo = __riscv_vle8_v_u8m1(qh, vl16);
                        vuint8m1_t vqh_hi = __riscv_vle8_v_u8m1(qh + 16, vl16);

                        vuint8m1_t vlo4_lo = __riscv_vand_vx_u8m1(vq4_lo, 0x0F, vl16);
                        vuint8m1_t vlo4_hi = __riscv_vand_vx_u8m1(vq4_hi, 0x0F, vl16);
                        vuint8m1_t vh_lo   = __riscv_vsll_vx_u8m1(
                            __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh_lo, 2, vl16), 0x03, vl16), 4, vl16);
                        vuint8m1_t vh_hi = __riscv_vsll_vx_u8m1(
                            __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh_hi, 2, vl16), 0x03, vl16), 4, vl16);

                        vuint8m1_t vcomb_lo = __riscv_vor_vv_u8m1(vlo4_lo, vh_lo, vl16);
                        vuint8m1_t vcomb_hi = __riscv_vor_vv_u8m1(vlo4_hi, vh_hi, vl16);

                        va_lo = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_lo), 32, vl16);
                        va_hi = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_hi), 32, vl16);
                    } وإلا إذا (bi_idx == 2) {
                        // a[l] = (q4[l] >> 4) | (((qh[l] >> 4) & 3) << 4) - 32
                        vuint8m1_t vq4_lo = __riscv_vle8_v_u8m1(q4, vl16);
                        vuint8m1_t vq4_hi = __riscv_vle8_v_u8m1(q4 + 16, vl16);
                        vuint8m1_t vqh_lo = __riscv_vle8_v_u8m1(qh, vl16);
                        vuint8m1_t vqh_hi = __riscv_vle8_v_u8m1(qh + 16, vl16);

                        vuint8m1_t vhi4_lo = __riscv_vsrl_vx_u8m1(vq4_lo, 4, vl16);
                        vuint8m1_t vhi4_hi = __riscv_vsrl_vx_u8m1(vq4_hi, 4, vl16);
                        vuint8m1_t vh_lo   = __riscv_vsll_vx_u8m1(
                            __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh_lo, 4, vl16), 0x03, vl16), 4, vl16);
                        vuint8m1_t vh_hi = __riscv_vsll_vx_u8m1(
                            __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh_hi, 4, vl16), 0x03, vl16), 4, vl16);

                        vuint8m1_t vcomb_lo = __riscv_vor_vv_u8m1(vhi4_lo, vh_lo, vl16);
                        vuint8m1_t vcomb_hi = __riscv_vor_vv_u8m1(vhi4_hi, vh_hi, vl16);

                        va_lo = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_lo), 32, vl16);
                        va_hi = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_hi), 32, vl16);
                    } وإلا {  // bi_idx == 3
                        // a[l] = (q4[l+32] >> 4) | (((qh[l] >> 6) & 3) << 4) - 32
                        vuint8m1_t vq4_lo = __riscv_vle8_v_u8m1(q4 + 32, vl16);
                        vuint8m1_t vq4_hi = __riscv_vle8_v_u8m1(q4 + 48, vl16);
                        vuint8m1_t vqh_lo = __riscv_vle8_v_u8m1(qh, vl16);
                        vuint8m1_t vqh_hi = __riscv_vle8_v_u8m1(qh + 16, vl16);

                        vuint8m1_t vhi4_lo = __riscv_vsrl_vx_u8m1(vq4_lo, 4, vl16);
                        vuint8m1_t vhi4_hi = __riscv_vsrl_vx_u8m1(vq4_hi, 4, vl16);
                        vuint8m1_t vh_lo   = __riscv_vsll_vx_u8m1(
                            __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh_lo, 6, vl16), 0x03, vl16), 4, vl16);
                        vuint8m1_t vh_hi = __riscv_vsll_vx_u8m1(
                            __riscv_vand_vx_u8m1(__riscv_vsrl_vx_u8m1(vqh_hi, 6, vl16), 0x03, vl16), 4, vl16);

                        vuint8m1_t vcomb_lo = __riscv_vor_vv_u8m1(vhi4_lo, vh_lo, vl16);
                        vuint8m1_t vcomb_hi = __riscv_vor_vv_u8m1(vhi4_hi, vh_hi, vl16);

                        va_lo = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_lo), 32, vl16);
                        va_hi = __riscv_vsub_vx_i8m1(__riscv_vreinterpret_v_u8m1_i8m1(vcomb_hi), 32, vl16);
                    }

                    // --- Widen to i16 for scaled abs computation ---
                    عائم scale_0 = scales[bi * 2 + 0] * d;
                    عائم scale_1 = scales[bi * 2 + 1] * d;

                    // Widen i8 -> i16 -> f32 for abs*scale computation
                    vint16m2_t va_lo_w = __riscv_vsext_vf2_i16m2(va_lo, vl16);
                    vint16m2_t va_hi_w = __riscv_vsext_vf2_i16m2(va_hi, vl16);

                    // Compute |a[l] * scale_0| for lo half, |a[l] * scale_1| for hi half
                    vfloat32m4_t vf_lo = __riscv_vfcvt_f_x_v_f32m4(__riscv_vsext_vf2_i32m4(va_lo_w, vl16), vl16);
                    vfloat32m4_t vf_hi = __riscv_vfcvt_f_x_v_f32m4(__riscv_vsext_vf2_i32m4(va_hi_w, vl16), vl16);

                    vfloat32m4_t vabs_lo = __riscv_vfabs_v_f32m4(__riscv_vfmul_vf_f32m4(vf_lo, scale_0, vl16), vl16);
                    vfloat32m4_t vabs_hi = __riscv_vfabs_v_f32m4(__riscv_vfmul_vf_f32m4(vf_hi, scale_1, vl16), vl16);

                    // Find max abs across both halves
                    vfloat32m4_t vabs_max = __riscv_vfmax_vv_f32m4(vabs_lo, vabs_hi, vl16);

                    // Reduce to scalar max
                    vfloat32m1_t vzero     = __riscv_vfmv_v_f_f32m1(0.0f, 1);
                    vfloat32m1_t vmax_red  = __riscv_vfredmax_vs_f32m4_f32m1(vabs_max, vzero, vl16);
                    عائم        a_max_abs = __riscv_vfmv_f_s_f32m1_f32(vmax_red);

                    عائم reflect_scale   = a_max_abs / 127.0f;
                    عائم reflect_scale_0 = scale_0 / reflect_scale;
                    عائم reflect_scale_1 = scale_1 / reflect_scale;

                    // --- Requant: a[l] = clamp(nearbyint(a[l] * reflect_scale_x), -128, 127) ---
                    vfloat32m4_t vscaled_lo = __riscv_vfmul_vf_f32m4(vf_lo, reflect_scale_0, vl16);
                    vfloat32m4_t vscaled_hi = __riscv_vfmul_vf_f32m4(vf_hi, reflect_scale_1, vl16);

                    // fcvt.x rounds to nearest (using current rounding mode)
                    vint32m4_t vi_lo = __riscv_vfcvt_x_f_v_i32m4(vscaled_lo, vl16);
                    vint32m4_t vi_hi = __riscv_vfcvt_x_f_v_i32m4(vscaled_hi, vl16);

                    // Clamp to [-128, 127]
                    vi_lo = __riscv_vmax_vx_i32m4(vi_lo, -128, vl16);
                    vi_lo = __riscv_vmin_vx_i32m4(vi_lo, 127, vl16);
                    vi_hi = __riscv_vmax_vx_i32m4(vi_hi, -128, vl16);
                    vi_hi = __riscv_vmin_vx_i32m4(vi_hi, 127, vl16);

                    // Narrow i32 -> i16 -> i8
                    vint16m2_t vi16_lo = __riscv_vncvt_x_x_w_i16m2(vi_lo, vl16);
                    vint16m2_t vi16_hi = __riscv_vncvt_x_x_w_i16m2(vi_hi, vl16);
                    vint8m1_t  vi8_lo  = __riscv_vncvt_x_x_w_i8m1(vi16_lo, vl16);
                    vint8m1_t  vi8_hi  = __riscv_vncvt_x_x_w_i8m1(vi16_hi, vl16);

                    // Store d and qs directly into dst block
                    dst->d[i]   = GGML_FP32_مقصد_FP16(reflect_scale);
                    int8_t * dq = (int8_t *) dst->qs + i * QK8_0;
                    __riscv_vse8_v_i8m1(dq, vi8_lo, vl16);
                    __riscv_vse8_v_i8m1(dq + 16, vi8_hi, vl16);
                }
                dst++;
            }
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

ساكن صحيح repack_q8_0_مقصد_q8_0_32_bl_ref(ggml_tensor *              t,
                                         صحيح                        interleave_بلوک,
                                         ثابت فراغ * GGML_RESTRICT داده,
                                         size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_بلوک == 32);  //استفادهنشده

    constexpr صحيح nrows_interleaved = 32;

    بلوک_q8_0x32 *    dst = (بلوک_q8_0x32 *) t->داده;
    ثابت بلوک_q8_0 * src = (ثابت بلوک_q8_0 *) داده;
    بلوک_q8_0         dst_tmp[32];
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q8_0));

    إذا (t->ne[0] % QK8_0 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        int64_t nrows_real = std::min((int64_t) nrow - b, (int64_t) nrows_interleaved);
        لكل (int64_t x = 0; x < nblocks; x++) {
            صحيح i = 0;
            لكل (; i < nrows_real; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            لكل (; i < nrows_interleaved; i++) {
                memset(&dst_tmp[i], 0, حجم(بلوک_q8_0));
            }
            *dst++ = make_بلوک_q8_0x32(dst_tmp, interleave_بلوک);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

// RVV optimized version of repack_q8_0_to_q8_0_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes scale gather + qs copy.
ساكن صحيح repack_q8_0_مقصد_q8_0_32_bl(ggml_tensor *              t,
                                     صحيح                        interleave_بلوک,
                                     ثابت فراغ * GGML_RESTRICT داده,
                                     size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_بلوک == 32);

    constexpr صحيح nrows_interleaved = 32;

    بلوک_q8_0x32 *    dst     = (بلوک_q8_0x32 *) t->داده;
    ثابت بلوک_q8_0 * src     = (ثابت بلوک_q8_0 *) داده;
    صحيح                nrow    = ggml_nrows(t);
    صحيح                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q8_0));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK8_0 != 0) {
        أرجع -1;
    }

    ثابت ptrdiff_t ردیف_stride = (ptrdiff_t) nblocks * حجم(بلوک_q8_0);

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            ثابت بلوک_q8_0 * ستون_src = src + x;

            // --- 1) Gather 32 scale values (ggml_half d) with stride load ---
            {
                ثابت uint8_t * d_پایه    = (ثابت uint8_t *) &ستون_src->d;
                ggml_half *     d_dst     = dst->d;
                size_t          مانده = 32;
                size_t          جابه_جایی    = 0;
                طالما (مانده > 0) {
                    size_t      vl = __riscv_vsetvl_e16m1(مانده);
                    vuint16m1_t vd =
                        __riscv_vlse16_v_u16m1((ثابت uint16_t *) (d_پایه + جابه_جایی * ردیف_stride), ردیف_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + جابه_جایی), vd, vl);
                    جابه_جایی += vl;
                    مانده -= vl;
                }
            }

            // --- 2) Copy qs for each of the 32 rows (32 bytes per row) ---
            {
                لكل (صحيح i = 0; i < 32; i++) {
                    ثابت int8_t * sq = ستون_src[i * nblocks].qs;
                    int8_t *       dq = (int8_t *) dst->qs + i * QK8_0;

                    size_t len = QK8_0;
                    size_t idx = 0;
                    طالما (len > 0) {
                        size_t    vl = __riscv_vsetvl_e8m2(len);
                        vint8m2_t vs = __riscv_vle8_v_i8m2(sq + idx, vl);
                        __riscv_vse8_v_i8m2(dq + idx, vs, vl);
                        idx += vl;
                        len -= vl;
                    }
                }
            }

            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;

    GGML_استفادهنشده(داده_اندازه);
}

ساكن فراغ convert_mxfp4_مقصد_5bit(ثابت بلوک_mxfp4 & src, spacemit_kernels::nrow_بلوک_mxfp4<1> & dst) {
    dst.e[0] = src.e;

    // Decode all 32 mxfp4 values to signed integers via kvalues_mxfp4
    int8_t vals[32];
    لكل (صحيح j = 0; j < QK_MXFP4 / 2; j++) {
        vals[j]                = kvalues_mxfp4[src.qs[j] & 0xF];
        vals[j + QK_MXFP4 / 2] = kvalues_mxfp4[src.qs[j] >> 4];
    }

    // vals [b0, b1, b2, b3, ..., b30, b31]
    // Pack abs into qs with reorder: [b0,b1]..[b14,b15]..[b30,b31]
    لكل (صحيح j = 0; j < QK_MXFP4 / 2; j++) {
        uint8_t lo0 = static_cast<uint8_t>(std::قدر_مطلق(vals[j * 2]));
        uint8_t lo1 = static_cast<uint8_t>(std::قدر_مطلق(vals[j * 2 + 1]));
        dst.qs[j]   = (lo0 & 0x0F) | ((lo1 & 0x0F) << 4);
    }

    // Pack sign bits into qh[4] (32 bits total, 1 bit per weight)
    // reorder: [0,1,2,...,15,16,17,...,31] after the qs reorder above
    uint32_t sign_بیتها = 0;
    لكل (صحيح j = 0; j < 32; j++) {
        إذا (vals[j] < 0) {
            sign_بیتها |= (1u << j);
        }
    }
    کپی_حافظه(dst.qh, &sign_بیتها, 4);
}

ساكن spacemit_kernels::nrow_بلوک_mxfp4<32> make_بلوک_mxfp4x32(spacemit_kernels::nrow_بلوک_mxfp4<1> * درون,
                                                                  بیعلامت صحيح blck_اندازه_interleave) {
    spacemit_kernels::nrow_بلوک_mxfp4<32> بیرون;
    GGML_ASSERT(QK_MXFP4 / blck_اندازه_interleave == 1);
    GGML_استفادهنشده(blck_اندازه_interleave);

    لكل (صحيح i = 0; i < 32; i++) {
        بیرون.e[i] = درون[i].e[0];
    }

    // qs: copy per-row 16 bytes
    لكل (صحيح i = 0; i < 32; i++) {
        کپی_حافظه(بیرون.qs + i * 16, درون[i].qs, 16);
    }

    // qh: copy per-row 4 bytes
    لكل (صحيح i = 0; i < 32; i++) {
        کپی_حافظه(بیرون.qh + i * 4, درون[i].qh, 4);
    }

    أرجع بیرون;
}

ساكن صحيح repack_mxfp4_مقصد_mxfp4_32_bl(ggml_tensor *              t,
                                       صحيح                        interleave_بلوک,
                                       ثابت فراغ * GGML_RESTRICT داده,
                                       size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_بلوک == 32);

    constexpr صحيح nrows_interleaved = 32;

    spacemit_kernels::nrow_بلوک_mxfp4<32> * dst = (spacemit_kernels::nrow_بلوک_mxfp4<32> *) t->داده;
    ثابت بلوک_mxfp4 *                      src = (ثابت بلوک_mxfp4 *) داده;
    spacemit_kernels::nrow_بلوک_mxfp4<1>    dst_tmp[32];
    صحيح                                      nrow    = ggml_nrows(t);
    صحيح                                      nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_mxfp4));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_MXFP4 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                convert_mxfp4_مقصد_5bit(src[x + i * nblocks], dst_tmp[i]);
            }
            *dst++ = make_بلوک_mxfp4x32(dst_tmp, interleave_بلوک);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;
}

ساكن spacemit_kernels::nrow_بلوک_q5_1<32> make_بلوک_q5_1x32(spacemit_kernels::nrow_بلوک_q5_1<1> * درون,
                                                                بیعلامت صحيح blck_اندازه_interleave) {
    spacemit_kernels::nrow_بلوک_q5_1<32> بیرون;
    GGML_ASSERT(QK5_1 / blck_اندازه_interleave == 1);
    GGML_استفادهنشده(blck_اندازه_interleave);

    لكل (صحيح i = 0; i < 32; i++) {
        بیرون.scales16[i] = درون[i].scales16[0];
        بیرون.zp[i]       = درون[i].zp[0];
    }

    // qs: low 4 bits, reorder from [b0,b16],[b1,b17]... to [b0,b1]...[b14,b15] and [b16,b17]...[b30,b31]
    لكل (صحيح i = 0; i < 32; i++) {
        // low half [0..15]
        لكل (صحيح j = 0; j < QK5_1 / 4; j++) {
            بیرون.qs[i * QK5_1 / 2 + j] = (درون[i].qs[j * 2] & 0x0F) | ((درون[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
        // high half [16..31]
        لكل (صحيح j = 0; j < QK5_1 / 4; j++) {
            بیرون.qs[i * QK5_1 / 2 + QK5_1 / 4 + j] = ((درون[i].qs[j * 2] & 0xF0) >> 4) | (درون[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    // qh: 5th bit, copy directly
    لكل (صحيح i = 0; i < 32; i++) {
        لكل (صحيح j = 0; j < 4; j++) {
            بیرون.qh[i * 4 + j] = درون[i].qh[j];
        }
    }

    أرجع بیرون;
}

ساكن spacemit_kernels::nrow_بلوک_q5_0<32> make_بلوک_q5_0x32(spacemit_kernels::nrow_بلوک_q5_0<1> * درون,
                                                                بیعلامت صحيح blck_اندازه_interleave) {
    spacemit_kernels::nrow_بلوک_q5_0<32> بیرون;
    GGML_ASSERT(QK5_0 / blck_اندازه_interleave == 1);
    GGML_استفادهنشده(blck_اندازه_interleave);

    لكل (صحيح i = 0; i < 32; i++) {
        بیرون.scales16[i] = درون[i].scales16[0];
    }

    // qs: low 4 bits, reorder from [b0,b16],[b1,b17]... to [b0,b1]...[b14,b15] and [b16,b17]...[b30,b31]
    لكل (صحيح i = 0; i < 32; i++) {
        // low half [0..15]
        لكل (صحيح j = 0; j < QK5_0 / 4; j++) {
            بیرون.qs[i * QK5_0 / 2 + j] = (درون[i].qs[j * 2] & 0x0F) | ((درون[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
        // high half [16..31]
        لكل (صحيح j = 0; j < QK5_0 / 4; j++) {
            بیرون.qs[i * QK5_0 / 2 + QK5_0 / 4 + j] = ((درون[i].qs[j * 2] & 0xF0) >> 4) | (درون[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    // qh: 5th bit, copy directly
    لكل (صحيح i = 0; i < 32; i++) {
        لكل (صحيح j = 0; j < 4; j++) {
            بیرون.qh[i * 4 + j] = درون[i].qh[j];
        }
    }

    أرجع بیرون;
}

ساكن صحيح repack_q5_0_مقصد_q5_0_32_bl(ggml_tensor *              t,
                                     صحيح                        interleave_بلوک,
                                     ثابت فراغ * GGML_RESTRICT داده,
                                     size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(interleave_بلوک == 32);  //استفادهنشده

    constexpr صحيح nrows_interleaved = 32;

    spacemit_kernels::nrow_بلوک_q5_0<32> * dst = (spacemit_kernels::nrow_بلوک_q5_0<32> *) t->داده;
    ثابت بلوک_q5_0 *                      src = (ثابت بلوک_q5_0 *) داده;
    spacemit_kernels::nrow_بلوک_q5_0<1>    dst_tmp[32];
    صحيح                                     nrow    = ggml_nrows(t);
    صحيح                                     nblocks = t->ne[0] / QK5_0;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q5_0));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK5_0 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                ثابت بلوک_q5_0 & s = src[x + i * nblocks];

                dst_tmp[i].scales16[0] = s.d;
                کپی_حافظه(dst_tmp[i].qs, s.qs, حجم(dst_tmp[i].qs));
                کپی_حافظه(dst_tmp[i].qh, s.qh, حجم(dst_tmp[i].qh));
            }
            *dst++ = make_بلوک_q5_0x32(dst_tmp, interleave_بلوک);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;
}

ساكن صحيح repack_q5_1_مقصد_q5_1_32_bl(ggml_tensor *              t,
                                     صحيح                        interleave_بلوک,
                                     ثابت فراغ * GGML_RESTRICT داده,
                                     size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(interleave_بلوک == 32);  //استفادهنشده

    constexpr صحيح nrows_interleaved = 32;

    spacemit_kernels::nrow_بلوک_q5_1<32> * dst = (spacemit_kernels::nrow_بلوک_q5_1<32> *) t->داده;
    ثابت بلوک_q5_1 *                      src = (ثابت بلوک_q5_1 *) داده;
    spacemit_kernels::nrow_بلوک_q5_1<1>    dst_tmp[32];
    صحيح                                     nrow    = ggml_nrows(t);
    صحيح                                     nblocks = t->ne[0] / QK5_1;

    GGML_ASSERT(داده_اندازه == nrow * nblocks * حجم(بلوک_q5_1));

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK5_1 != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                ثابت بلوک_q5_1 & s = src[x + i * nblocks];

                عائم d = GGML_FP16_مقصد_FP32(s.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                عائم m = GGML_FP16_مقصد_FP32(s.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);

                إذا (d == 0.0f) {
                    dst_tmp[i].scales16[0] = GGML_FP32_مقصد_FP16(std::fabs(m));
                    dst_tmp[i].zp[0]       = m < 0.0f ? 1 : 0;
                    memset(dst_tmp[i].qh, 0, حجم(dst_tmp[i].qh));
                    memset(dst_tmp[i].qs, m > 0.0f ? 0x11 : 0x00, حجم(dst_tmp[i].qs));
                    استمر;
                }

                عائم mid = std::nearbyintf(-m / d);
                mid       = std::min(31.0f, std::max(0.0f, mid));

                dst_tmp[i].scales16[0] = GGML_FP32_مقصد_FP16(d);
                dst_tmp[i].zp[0]       = static_cast<uint8_t>(mid);

                // qs: copy low 4 bits directly (same nibble packing)
                کپی_حافظه(dst_tmp[i].qs, s.qs, QK5_1 / 2);

                // qh: copy 5th bit directly
                کپی_حافظه(dst_tmp[i].qh, s.qh, 4);
            }
            *dst++ = make_بلوک_q5_1x32(dst_tmp, interleave_بلوک);
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;
}

ساكن صحيح repack_q5_k_مقصد_q5_1_32_bl(ggml_tensor *              t,
                                     صحيح                        interleave_بلوک,
                                     ثابت فراغ * GGML_RESTRICT داده,
                                     size_t                     داده_اندازه) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_بلوک == 32);
    GGML_ASSERT(QK_K / QK5_1 == 8);

    constexpr صحيح nrows_interleaved = 32;

    spacemit_kernels::nrow_بلوک_q5_1<32> * dst = (spacemit_kernels::nrow_بلوک_q5_1<32> *) t->داده;
    ثابت بلوک_q5_K *                      src = (ثابت بلوک_q5_K *) داده;
    spacemit_kernels::nrow_بلوک_q5_1<1>    dst_tmp[32];
    صحيح                                     nrow    = ggml_nrows(t);
    صحيح                                     nblocks = t->ne[0] / QK_K;

    إذا (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        أرجع -1;
    }

    لكل (صحيح b = 0; b < nrow; b += nrows_interleaved) {
        لكل (int64_t x = 0; x < nblocks; x++) {
            لكل (صحيح j = 0; j < 8; j++) {
                لكل (صحيح i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    ثابت عائم d = GGML_FP16_مقصد_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    ثابت عائم min =
                        GGML_FP16_مقصد_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);

                    عائم d1 = d * sc;
                    عائم m1 = min * m;

                    عائم mid              = std::nearbyintf(m1 / d1);
                    mid                    = std::min(31.0f, std::max(0.0f, mid));
                    dst_tmp[i].scales16[0] = GGML_FP32_مقصد_FP16(d1);
                    dst_tmp[i].zp[0]       = static_cast<uint8_t>(mid);

                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    ثابت uint8_t * q = src[x + i * nblocks].qs + (j / 2) * QK5_1;
                    إذا (j % 2 == 0) {
                        لكل (صحيح ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } وإلا {
                        لكل (صحيح ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }

                    // Extract the 5th bit (qh) for this sub-block
                    // block_q5_K.qh[32]: for sub-block j, the 5th bit is at bit position j in qh[l]
                    // qs was reordered: dst_qs maps to src weights [0,16,1,17,...,15,31]
                    // So qh must follow the same reorder to stay aligned with qs
                    // dst qh[4] = 32 bits for 32 weights in the reordered layout:
                    //   byte 0: weights 0..7   (from src_qh[0..7])
                    //   byte 1: weights 8..15  (from src_qh[8..15])
                    //   byte 2: weights 16..23 (from src_qh[16..23])
                    //   byte 3: weights 24..31 (from src_qh[24..31])
                    ثابت uint8_t * src_qh = src[x + i * nblocks].qh;
                    لكل (صحيح bi = 0; bi < 4; bi++) {
                        uint8_t qh_byte = 0;
                        لكل (صحيح k = 0; k < 8; k++) {
                            صحيح src_idx = bi * 8 + k;
                            qh_byte |= ((src_qh[src_idx] >> j) & 1) << k;
                        }
                        dst_tmp[i].qh[bi] = qh_byte;
                    }
                }
                *dst++ = make_بلوک_q5_1x32(dst_tmp, interleave_بلوک);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    أرجع 0;
}

نطاق ggml::cpu::riscv64_spacemit {

قالب <نام_نوع BLOC_TYPE, int64_t INTER_اندازه, int64_t NB_تعداد_ستون> صحيح repack(ggml_tensor *, ثابت فراغ *, size_t);

قالب <> صحيح repack<بلوک_q4_0, 32, 16>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
    أرجع repack_q4_0_مقصد_q4_0_16_bl(t, 16, داده, داده_اندازه);
}

قالب <> صحيح repack<بلوک_q4_1, 32, 16>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
    أرجع repack_q4_1_مقصد_q4_1_16_bl(t, 16, داده, داده_اندازه);
}

قالب <> صحيح repack<بلوک_q4_K, 32, 16>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
    أرجع repack_q4_k_مقصد_q4_1_16_bl(t, 16, داده, داده_اندازه);
}

قالب <> صحيح repack<بلوک_q2_K, 256, 32>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
    أرجع repack_q2_k_مقصد_q2_k_32_bl(t, 32, داده, داده_اندازه);
}

قالب <> صحيح repack<بلوک_q3_K, 256, 32>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
    أرجع repack_q3_k_مقصد_q3_k_32_bl(t, 32, داده, داده_اندازه);
}

قالب <> صحيح repack<بلوک_q4_0, 32, 32>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
#إذا 0
    أرجع repack_q4_0_مقصد_q4_0_32_bl_ref(t, 32, داده, داده_اندازه);
#وإلا
    أرجع repack_q4_0_مقصد_q4_0_32_bl(t, 32, داده, داده_اندازه);
#پایان_اگر
}

قالب <> صحيح repack<بلوک_q4_0, 256, 32>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
#إذا 1
    أرجع repack_q4_0_مقصد_q4_0_256_32_bl_ref(t, 32, داده, داده_اندازه);
#وإلا
    //return repack_q4_0_to_q4_0_256_32_bl(t, 32, data, data_size);
#پایان_اگر
}

قالب <> صحيح repack<بلوک_q4_1, 32, 32>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
#إذا 0
    أرجع repack_q4_1_مقصد_q4_1_32_bl_ref(t, 32, داده, داده_اندازه);
#وإلا
    أرجع repack_q4_1_مقصد_q4_1_32_bl(t, 32, داده, داده_اندازه);
#پایان_اگر
}

قالب <> صحيح repack<بلوک_q4_1, 256, 32>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
#إذا 1
    أرجع repack_q4_0_مقصد_q4_1_256_32_bl_ref(t, 32, داده, داده_اندازه);
#وإلا
    أرجع repack_q4_1_مقصد_q4_1_256_32_bl(t, 32, داده, داده_اندازه);
#پایان_اگر
}

قالب <> صحيح repack<بلوک_q4_K, 32, 32>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
    أرجع repack_q4_k_مقصد_q4_1_32_bl(t, 32, داده, داده_اندازه);
}

قالب <> صحيح repack<بلوک_q6_K, 32, 32>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
#إذا 1
    أرجع repack_q6_k_مقصد_q8_0_32_bl_ref(t, 32, داده, داده_اندازه);
#وإلا
    أرجع repack_q6_k_مقصد_q8_0_32_bl(t, 32, داده, داده_اندازه);
#پایان_اگر
}

قالب <> صحيح repack<بلوک_q8_0, 32, 16>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
    أرجع repack_q8_0_مقصد_q8_0_16_bl_ref(t, 16, داده, داده_اندازه);
}

قالب <> صحيح repack<بلوک_q8_0, 32, 32>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
#إذا 1
    أرجع repack_q8_0_مقصد_q8_0_32_bl_ref(t, 32, داده, داده_اندازه);
#وإلا
    أرجع repack_q8_0_مقصد_q8_0_32_bl(t, 32, داده, داده_اندازه);
#پایان_اگر
}

قالب <> صحيح repack<بلوک_mxfp4, 32, 32>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
    أرجع repack_mxfp4_مقصد_mxfp4_32_bl(t, 32, داده, داده_اندازه);
}

قالب <> صحيح repack<بلوک_q5_0, 32, 32>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
    أرجع repack_q5_0_مقصد_q5_0_32_bl(t, 32, داده, داده_اندازه);
}

قالب <> صحيح repack<بلوک_q5_1, 32, 32>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
    أرجع repack_q5_1_مقصد_q5_1_32_bl(t, 32, داده, داده_اندازه);
}

قالب <> صحيح repack<بلوک_q5_K, 32, 32>(ggml_tensor * t, ثابت فراغ * داده, size_t داده_اندازه) {
    أرجع repack_q5_k_مقصد_q5_1_32_bl(t, 32, داده, داده_اندازه);
}

}  // namespace ggml::cpu::riscv64_spacemit
