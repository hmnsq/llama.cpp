#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_IMPL_CPP
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_CPP

#ಅನ್ತರ್ಭಾವ "repack.h"

#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "ggml-cpu.h"
#ಅನ್ತರ್ಭಾವ "ggml-impl.h"
#ಅನ್ತರ್ಭಾವ "ime_kernels.h"

#ಅನ್ತರ್ಭಾವ <algorithm>
#ಅನ್ತರ್ಭಾವ <cassert>
#ಅನ್ತರ್ಭಾವ <cmath>
#ಅನ್ತರ್ಭಾವ <cstring>

// clang-format off
#ಆದರೆ defined(__riscv)

#ಆದರೆ !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#ದೋಷ "riscv v extension or v_intrinsic not enabled"
#ಇಲ್ಲದಿದ್ದರೆ
#ಅನ್ತರ್ಭಾವ <riscv_vector.h>
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ !defined(__riscv_zfh)
#ದೋಷ "riscv zfh extension not enabled"
#ಆದರೆ_ಅಂತ್ಯ

#ಇಲ್ಲದಿದ್ದರೆ
#ದೋಷ "riscv not enabled in this build"
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(__GNUC__)
#ಸೂಚನೆ GCC diagnostic ignored "-Wcast-qual"
#ಸೂಚನೆ GCC diagnostic ignored "-Wunused-parameter"
#ಆದರೆ_ಅಂತ್ಯ

// clang-format on

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ K> constexpr ಪೂರ್ಣಾಂಕ QK_0() {
    ಆದರೆ constexpr (K == 4) {
        ಹಿಂತಿರುಗಿ QK4_0;
    }
    ಆದರೆ constexpr (K == 8) {
        ಹಿಂತಿರುಗಿ QK8_0;
    }
    ಹಿಂತಿರುಗಿ -1;
}

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ K, ಪೂರ್ಣಾಂಕ N> ರಚನೆ ಖಂಡ {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    uint8_t   qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

ಮಾದರಿ <ಪೂರ್ಣಾಂಕ K, ಪೂರ್ಣಾಂಕ N> ರಚನೆ ಖಂಡ_with_zp {
    ggml_half d[N];                         // deltas for N qK_1 blocks
    uint8_t   zp[N];                        // zero points for N qK_1 blocks
    uint8_t   qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_1 blocks
};

// control size
static_assert(ಗಾತ್ರ(ಖಂಡ<4, 16>) == 16 * ಗಾತ್ರ(ggml_half) + QK4_0 * 8, "wrong block<4,16> size/padding");
static_assert(ಗಾತ್ರ(ಖಂಡ_with_zp<4, 16>) == 16 * ಗಾತ್ರ(ggml_half) + QK4_0 * 8 + 16 * ಗಾತ್ರ(uint8_t),
              "wrong block_with_zp<4,16> size/padding");

static_assert(ಗಾತ್ರ(ಖಂಡ<8, 16>) == 16 * ಗಾತ್ರ(ggml_half) + QK4_0 * 16, "wrong block<8,16> size/padding");

static_assert(ಗಾತ್ರ(ಖಂಡ<4, 32>) == 32 * ಗಾತ್ರ(ggml_half) + QK4_0 * 16, "wrong block<4,32> size/padding");
static_assert(ಗಾತ್ರ(ಖಂಡ_with_zp<4, 32>) == 32 * ಗಾತ್ರ(ggml_half) + QK4_0 * 16 + 32 * ಗಾತ್ರ(uint8_t),
              "wrong block_with_zp<4,32> size/padding");

ಬಳಸಿ ಖಂಡ_q4_0x16 = ಖಂಡ<4, 16>;
ಬಳಸಿ ಖಂಡ_q4_1x16 = ಖಂಡ_with_zp<4, 16>;
ಬಳಸಿ ಖಂಡ_q8_0x16 = ಖಂಡ<8, 16>;

ಬಳಸಿ ಖಂಡ_q4_0x32 = ಖಂಡ<4, 32>;
ಬಳಸಿ ಖಂಡ_q4_1x32 = ಖಂಡ_with_zp<4, 32>;
ಬಳಸಿ ಖಂಡ_q8_0x32 = ಖಂಡ<8, 32>;

ರಚನೆ ಖಂಡ_q4_0x32x256 {
    ಖಂಡ_q4_0x32 ಖಂಡಗಳು[8];  // [f16 * 32 | i4 * 32 * 32] * 8
};

ರಚನೆ ಖಂಡ_q4_1x32x256 {
    ಖಂಡ_q4_0x32 ಖಂಡಗಳು[8];
    uint8_t       zps[32 * 8];
};

ಸ್ಥಿರ ಖಂಡ_q4_0x16 make_ಖಂಡ_q4_0x16(ಖಂಡ_q4_0 * ಒಳ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ_interleave) {
    ಖಂಡ_q4_0x16 ಹೊರಹೋಗು;
    GGML_ASSERT(QK4_0 / blck_ಗಾತ್ರ_interleave == 2);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; i++) {
        ಹೊರಹೋಗು.d[i] = ಒಳ[i].d;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; i++) {
        // [0, 15], in.d & 0x0F
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b8] ......... [b7 b15]
            ಹೊರಹೋಗು.qs[i * QK4_0 / 4 + j] = (ಒಳ[i].qs[j] & 0x0F) | ((ಒಳ[i].qs[j + QK4_0 / 4] & 0x0F) << 4);
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; i++) {
        // [16, 31], in.d & 0xF0
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            ಹೊರಹೋಗು.qs[4 * QK4_0 + i * QK4_0 / 4 + j] = ((ಒಳ[i].qs[j] & 0xF0) >> 4) | (ಒಳ[i].qs[j + QK4_0 / 4] & 0xF0);
        }
    }

    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು;
}

ಸ್ಥಿರ ಖಂಡ_q4_1x16 make_ಖಂಡ_q4_1x16(ಖಂಡ_q4_1 * ಒಳ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ_interleave) {
    ಖಂಡ_q4_1x16 ಹೊರಹೋಗು;
    GGML_ASSERT(QK4_1 / blck_ಗಾತ್ರ_interleave == 2);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; i++) {
        ದಶಮಾಂಶ d   = GGML_FP16_ಗಮ್ಯ_FP32(ಒಳ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        ದಶಮಾಂಶ m   = GGML_FP16_ಗಮ್ಯ_FP32(ಒಳ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
        ದಶಮಾಂಶ mid = -std::nearbyintf(m / d);
        mid       = std::min(15.0f, std::max(0.0f, mid));
        ಹೊರಹೋಗು.d[i]  = GGML_FP32_ಗಮ್ಯ_FP16(d);
        ಹೊರಹೋಗು.zp[i] = static_cast<uint8_t>(mid);
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; i++) {
        // [0, 15], in.d & 0x0F
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b8] ......... [b7 b15]
            ಹೊರಹೋಗು.qs[i * QK4_1 / 4 + j] = (ಒಳ[i].qs[j] & 0x0F) | ((ಒಳ[i].qs[j + QK4_1 / 4] & 0x0F) << 4);
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; i++) {
        // [16, 31], in.d & 0xF0
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            ಹೊರಹೋಗು.qs[4 * QK4_1 + i * QK4_1 / 4 + j] = ((ಒಳ[i].qs[j] & 0xF0) >> 4) | (ಒಳ[i].qs[j + QK4_1 / 4] & 0xF0);
        }
    }

    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q4_0_ಗಮ್ಯ_q4_0_16_bl(ggml_tensor *              t,
                                     ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                     size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ಖಂಡ == 16);

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 16;

    ಖಂಡ_q4_0x16 *    dst = (ಖಂಡ_q4_0x16 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) ದತ್ತಾಂಶ;
    ಖಂಡ_q4_0         dst_tmp[16];
    ಪೂರ್ಣಾಂಕ                nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q4_0));

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ಖಂಡ_q4_0x16(dst_tmp, interleave_ಖಂಡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q4_1_ಗಮ್ಯ_q4_1_16_bl(ggml_tensor *              t,
                                     ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                     size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_ಖಂಡ == 16);

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 16;

    ಖಂಡ_q4_1x16 *    dst = (ಖಂಡ_q4_1x16 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 *) ದತ್ತಾಂಶ;
    ಖಂಡ_q4_1         dst_tmp[16];
    ಪೂರ್ಣಾಂಕ                nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q4_1));

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ಖಂಡ_q4_1x16(dst_tmp, interleave_ಖಂಡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ get_scale_min_k4(ಪೂರ್ಣಾಂಕ                           j,
                                    ಸ್ಥಿರಾಂಕ uint8_t * GGML_RESTRICT q,
                                    uint8_t * GGML_RESTRICT       d,
                                    uint8_t * GGML_RESTRICT       m) {
    ಆದರೆ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ಇಲ್ಲದಿದ್ದರೆ {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q4_k_ಗಮ್ಯ_q4_1_16_bl(ggml_tensor *              t,
                                     ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                     size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_ಖಂಡ == 16);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 16;

    ಖಂಡ_q4_1x16 *    dst = (ಖಂಡ_q4_1x16 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K *) ದತ್ತಾಂಶ;
    ಖಂಡ_q4_1         dst_tmp[16];
    ಪೂರ್ಣಾಂಕ                nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                nblocks = t->ne[0] / QK_K;

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_FP16_ಗಮ್ಯ_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ min =
                        GGML_FP16_ಗಮ್ಯ_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1 = d * sc;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = min * m;

                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_ಗಮ್ಯ_FP16(d1);
                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_ಗಮ್ಯ_FP16(-m1);
                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    ಸ್ಥಿರಾಂಕ uint8_t * q                                  = src[x + i * nblocks].qs + (j / 2) * QK4_1;
                    ಆದರೆ (j % 2 == 0) {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }
                }
                *dst++ = make_ಖಂಡ_q4_1x16(dst_tmp, interleave_ಖಂಡ);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಖಂಡ_q4_0x32 make_ಖಂಡ_q4_0x32(ಖಂಡ_q4_0 * ಒಳ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ_interleave) {
    ಖಂಡ_q4_0x32 ಹೊರಹೋಗು;
    ಪ್ರತಿಜ್ಞಾ(QK4_0 / blck_ಗಾತ್ರ_interleave == 1);
    GGML_ಬಳಸದ(blck_ಗಾತ್ರ_interleave);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        ಹೊರಹೋಗು.d[i] = ಒಳ[i].d;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        // [0, 15], in.d & 0x0F
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b1] .........  [b14 b15]
            ಹೊರಹೋಗು.qs[i * QK4_0 / 2 + j] = (ಒಳ[i].qs[j * 2] & 0x0F) | ((ಒಳ[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        // [16, 31], in.d & 0xF0
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b17] ......... [b30 b31]
            ಹೊರಹೋಗು.qs[i * QK4_0 / 2 + QK4_0 / 4 + j] = ((ಒಳ[i].qs[j * 2] & 0xF0) >> 4) | (ಒಳ[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು;
}

ಸ್ಥಿರ ಖಂಡ_q4_1x32 make_ಖಂಡ_q4_1x32(ಖಂಡ_q4_1 * ಒಳ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ_interleave) {
    ಖಂಡ_q4_1x32 ಹೊರಹೋಗು;
    GGML_ASSERT(QK4_1 / blck_ಗಾತ್ರ_interleave == 1);
    GGML_ಬಳಸದ(blck_ಗಾತ್ರ_interleave);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        ದಶಮಾಂಶ d   = GGML_FP16_ಗಮ್ಯ_FP32(ಒಳ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        ದಶಮಾಂಶ m   = GGML_FP16_ಗಮ್ಯ_FP32(ಒಳ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
        ದಶಮಾಂಶ mid = -std::nearbyintf(m / d);
        mid       = std::min(15.0f, std::max(0.0f, mid));
        ಹೊರಹೋಗು.d[i]  = GGML_FP32_ಗಮ್ಯ_FP16(d);
        ಹೊರಹೋಗು.zp[i] = static_cast<uint8_t>(mid);
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        // [0, 15], in.d & 0x0F
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b1] ......... [b14 b15]
            ಹೊರಹೋಗು.qs[i * QK4_1 / 2 + j] = (ಒಳ[i].qs[j * 2] & 0x0F) | ((ಒಳ[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        // [16, 31], in.d & 0xF0
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            ಹೊರಹೋಗು.qs[i * QK4_1 / 2 + QK4_1 / 4 + j] = ((ಒಳ[i].qs[j * 2] & 0xF0) >> 4) | (ಒಳ[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು;
}

ಸ್ಥಿರ ಖಂಡ_q8_0x32 make_ಖಂಡ_q8_0x32(ಖಂಡ_q8_0 * ಒಳ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ_interleave) {
    ಖಂಡ_q8_0x32 ಹೊರಹೋಗು;
    GGML_ASSERT(QK8_0 / blck_ಗಾತ್ರ_interleave == 1);
    GGML_ಬಳಸದ(blck_ಗಾತ್ರ_interleave);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        ಹೊರಹೋಗು.d[i] = ಒಳ[i].d;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        ಸ್ಮರಣೆ_ನಕಲು(ಹೊರಹೋಗು.qs + i * QK8_0, ಒಳ[i].qs, QK8_0);
    }

    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು;
}

// IME1: interleave 16 q8_0 rows so a plain vle8 sequence in the i8i8 kernel lands weights in the
// vmadot group/parity layout. Mirrors make_block_q4_0x16 but stores full int8 (no nibble packing).
// qs (512B) = [INNER step 0: reg0..reg7][INNER step 1: reg0..reg7], each reg 32B holding 4 columns
// x (even=K-first-half / odd=K-second-half). Column col -> acc=col/4, cgrp=col%4.
ಸ್ಥಿರ ಖಂಡ_q8_0x16 make_ಖಂಡ_q8_0x16(ಖಂಡ_q8_0 * ಒಳ, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ_interleave) {
    ಖಂಡ_q8_0x16 ಹೊರಹೋಗು;
    GGML_ASSERT(QK8_0 / blck_ಗಾತ್ರ_interleave == 2);
    GGML_ಬಳಸದ(blck_ಗಾತ್ರ_interleave);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 16; i++) {
        ಹೊರಹೋಗು.d[i] = ಒಳ[i].d;
    }

    memset(ಹೊರಹೋಗು.qs, 0, ಗಾತ್ರ(ಹೊರಹೋಗು.qs));
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ಲಂಬಸಾಲು = 0; ಲಂಬಸಾಲು < 16; ಲಂಬಸಾಲು++) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ      acc  = ಲಂಬಸಾಲು / 4;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ      cgrp = ಲಂಬಸಾಲು % 4;
        ಸ್ಥಿರಾಂಕ int8_t * q    = ಒಳ[ಲಂಬಸಾಲು].qs;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 0; s < 2; s++) {
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ತಳ   = s * 16;
            uint8_t * reg_lo = ಹೊರಹೋಗು.qs + (s * 8 + acc)     * 32;
            uint8_t * reg_hi = ಹೊರಹೋಗು.qs + (s * 8 + acc + 4) * 32;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
                reg_lo[(2 * cgrp)     * 4 + i] = (uint8_t) q[ತಳ + 0  + i];
                reg_lo[(2 * cgrp + 1) * 4 + i] = (uint8_t) q[ತಳ + 4  + i];
                reg_hi[(2 * cgrp)     * 4 + i] = (uint8_t) q[ತಳ + 8  + i];
                reg_hi[(2 * cgrp + 1) * 4 + i] = (uint8_t) q[ತಳ + 12 + i];
            }
        }
    }
    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q8_0_ಗಮ್ಯ_q8_0_16_bl_ref(ggml_tensor *              t,
                                         ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                         size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_ಖಂಡ == 16);

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 16;

    ಖಂಡ_q8_0x16 *    dst = (ಖಂಡ_q8_0x16 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) ದತ್ತಾಂಶ;
    ಖಂಡ_q8_0         dst_tmp[16];
    ಪೂರ್ಣಾಂಕ                nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q8_0));

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK8_0 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ಖಂಡ_q8_0x16(dst_tmp, interleave_ಖಂಡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q2_k_ಗಮ್ಯ_q2_k_32_bl(ggml_tensor *              t,
                                     ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                     size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_ಖಂಡ == 32);
    GGML_ASSERT(QK_K == 256);

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K *) ದತ್ತಾಂಶ;

    ಸ್ವಯಂಚಾಲಿತ * dst = (spacemit_kernels::nrow_ಖಂಡ_q2_k<32> *) t->ದತ್ತಾಂಶ;

    ಪೂರ್ಣಾಂಕ nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q2_K));

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    uint8_t qs_aux[256] = { 0 };
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                ಸ್ಥಿರಾಂಕ ಖಂಡ_q2_K * src_ಖಂಡ = &src[(b + i) * nblocks + x];

                // scale for [16, N]
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 16; j++) {
                    ಸ್ವಯಂಚಾಲಿತ zp_aux = (dst->scales[j * nrows_interleaved + i]) & 0xF0;

                    dst->scales[j * nrows_interleaved + i] = (src_ಖಂಡ->scales[j] & 0x0F) | zp_aux;
                }

                // zp for [N, 16]
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 16; j++) {
                    ಸ್ವಯಂಚಾಲಿತ scale_aux = (dst->scales[16 * i + j]) & 0x0F;

                    dst->scales[16 * i + j] = (src_ಖಂಡ->scales[j] & 0xF0) | scale_aux;
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; k++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j] = (src_ಖಂಡ->qs[j] >> (2 * k)) & 0x03;
                    }
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; k++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j + 128] = (src_ಖಂಡ->qs[j + 32] >> (2 * k)) & 0x03;
                    }
                }

                // from nrows_interleaved * [2 * 32byte]
                // to 4 * [nrows_interleaved * 16byte]
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; k++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 16; j++) {
                        uint8_t qs0  = qs_aux[j + k * 64];
                        uint8_t qs16 = qs_aux[j + 16 + k * 64];
                        uint8_t qs32 = qs_aux[j + 32 + k * 64];
                        uint8_t qs48 = qs_aux[j + 48 + k * 64];

                        dst->qs[(k * nrows_interleaved + i) * 16 + j] =
                            (qs0 & 0x03) | ((qs16 & 0x03) << 2) | ((qs32 & 0x03) << 4) | ((qs48 & 0x03) << 6);
                    }
                }

                dst->scales16[i] = src_ಖಂಡ->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
                dst->zeros16[i]  = src_ಖಂಡ->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
            }
            dst++;
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q3_k_ಗಮ್ಯ_q3_k_32_bl(ggml_tensor *              t,
                                     ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                     size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q3_K);
    GGML_ASSERT(interleave_ಖಂಡ == 32);
    GGML_ASSERT(QK_K == 256);

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    ಸ್ಥಿರಾಂಕ uint32_t kmask1 = 0x03030303;
    ಸ್ಥಿರಾಂಕ uint32_t kmask2 = 0x0f0f0f0f;

    ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K *) ದತ್ತಾಂಶ;

    ಸ್ವಯಂಚಾಲಿತ * dst = (spacemit_kernels::nrow_ಖಂಡ_q3_k<32> *) t->ದತ್ತಾಂಶ;

    ಪೂರ್ಣಾಂಕ nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q3_K));

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    uint32_t b_scale_aux[4] = { 0 };
    uint8_t  qs_aux[256]    = { 0 };

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                ಸ್ಥಿರಾಂಕ ಖಂಡ_q3_K * src_ಖಂಡ = &src[(b + i) * nblocks + x];

                uint32_t * auxs  = b_scale_aux;
                int8_t *   scale = (int8_t *) auxs;
                ಸ್ಮರಣೆ_ನಕಲು(auxs, src_ಖಂಡ->scales, 12);

                uint32_t tmp = auxs[2];
                auxs[2]      = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                auxs[3]      = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                auxs[0]      = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                auxs[1]      = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 16; j++) {
                    dst->scales[j * nrows_interleaved + i] = scale[j] - 32;
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; k++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j] = (src_ಖಂಡ->qs[j] >> (2 * k)) & 0x03;
                    }
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; k++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j + 128] = (src_ಖಂಡ->qs[j + 32] >> (2 * k)) & 0x03;
                    }
                }

                // from nrows_interleaved * [2 * 32byte]
                // to 4 * [nrows_interleaved * 16byte]
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; k++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 16; j++) {
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
                uint16_t * dst_ಮುಖವಾಡ = ((uint16_t *) dst->hmask) + i;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 16; j++, dst_ಮುಖವಾಡ += nrows_interleaved) {
                    uint8_t   b_ಸರಿಸು    = j / 2;
                    uint8_t * b_ಮುಖವಾಡ_ಲಂಬಸಾಲು = (uint8_t *) (src_ಖಂಡ->hmask + (j % 2) * 16);
                    // b0 - b15
                    uint16_t  msk_ಹೊರಹೋಗು_0  = 0;

                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; k++) {
                        msk_ಹೊರಹೋಗು_0 |= (uint16_t) ((b_ಮುಖವಾಡ_ಲಂಬಸಾಲು[k] >> b_ಸರಿಸು) & 0x01) << k;
                    }
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 8; k < 16; k++) {
                        msk_ಹೊರಹೋಗು_0 |= (uint16_t) ((b_ಮುಖವಾಡ_ಲಂಬಸಾಲು[k] >> b_ಸರಿಸು) & 0x01) << k;
                    }

                    dst_ಮುಖವಾಡ[0] = msk_ಹೊರಹೋಗು_0;
                }

                dst->scales16[i] = src_ಖಂಡ->d;
            }

            dst++;
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q4_0_ಗಮ್ಯ_q4_0_32_bl_ref(ggml_tensor *              t,
                                         ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                         size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ಖಂಡ == 32);  //ಬಳಸದ

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    ಖಂಡ_q4_0x32 *    dst = (ಖಂಡ_q4_0x32 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) ದತ್ತಾಂಶ;
    ಖಂಡ_q4_0         dst_tmp[32];
    ಪೂರ್ಣಾಂಕ                nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q4_0));

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ಖಂಡ_q4_0x32(dst_tmp, interleave_ಖಂಡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q4_0_ಗಮ್ಯ_q4_0_256_32_bl_ref(ggml_tensor *              t,
                                             ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                             ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                             size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ಖಂಡ == 32);  //ಬಳಸದ

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    ಖಂಡ_q4_0x32x256 * dst = (ಖಂಡ_q4_0x32x256 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *  src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) ದತ್ತಾಂಶ;
    ಖಂಡ_q4_0          dst_tmp[32];
    ಪೂರ್ಣಾಂಕ                 nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                 nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q4_0));
    GGML_ASSERT(nblocks % 8 == 0);  // for 256-block interleaving
    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x += 8) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                    dst_tmp[i] = src[x + j + i * nblocks];
                }
                dst->ಖಂಡಗಳು[j] = make_ಖಂಡ_q4_0x32(dst_tmp, interleave_ಖಂಡ);
            }
            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q4_0_ಗಮ್ಯ_q4_1_256_32_bl_ref(ggml_tensor *              t,
                                             ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                             ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                             size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_ಖಂಡ == 32);  //ಬಳಸದ

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    ಖಂಡ_q4_1x32x256 * dst = (ಖಂಡ_q4_1x32x256 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 *  src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 *) ದತ್ತಾಂಶ;
    ಖಂಡ_q4_1          dst_tmp[32];
    ಪೂರ್ಣಾಂಕ                 nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                 nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q4_1));
    GGML_ASSERT(nblocks % 8 == 0);  // for 256-block interleaving
    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x += 8) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                    dst_tmp[i] = src[x + j + i * nblocks];
                }

                ಖಂಡ_q4_0x32 * dst_ಖಂಡ = &dst->ಖಂಡಗಳು[j];
                uint8_t *       dst_zp    = dst->zps + j * nrows_interleaved;

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                    ದಶಮಾಂಶ d   = GGML_FP16_ಗಮ್ಯ_FP32(dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    ದಶಮಾಂಶ m   = GGML_FP16_ಗಮ್ಯ_FP32(dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
                    ದಶಮಾಂಶ mid = -std::nearbyintf(m / d);
                    mid       = std::min(15.0f, std::max(0.0f, mid));

                    dst_ಖಂಡ->d[i] = GGML_FP32_ಗಮ್ಯ_FP16(d);
                    dst_zp[i]       = static_cast<uint8_t>(mid);
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < QK4_1 / 4; k++) {
                        dst_ಖಂಡ->qs[i * QK4_1 / 2 + k] =
                            (dst_tmp[i].qs[k * 2] & 0x0F) | ((dst_tmp[i].qs[k * 2 + 1] & 0x0F) << 4);
                    }
                }

                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < QK4_1 / 4; k++) {
                        dst_ಖಂಡ->qs[i * QK4_1 / 2 + QK4_1 / 4 + k] =
                            ((dst_tmp[i].qs[k * 2] & 0xF0) >> 4) | (dst_tmp[i].qs[k * 2 + 1] & 0xF0);
                    }
                }
            }
            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

// RVV optimized version of repack_q4_0_to_q4_0_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes nibble repack.
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q4_0_ಗಮ್ಯ_q4_0_32_bl(ggml_tensor *              t,
                                     ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                     size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ಖಂಡ == 32);

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;
    constexpr ಪೂರ್ಣಾಂಕ qs_ಬೈಟುಗಳು          = QK4_0 / 2;  // 16

    ಖಂಡ_q4_0x32 *    dst     = (ಖಂಡ_q4_0x32 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * src     = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) ದತ್ತಾಂಶ;
    ಪೂರ್ಣಾಂಕ                nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q4_0));

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ptrdiff_t ಅಡ್ಡಸಾಲು_stride = (ptrdiff_t) nblocks * ಗಾತ್ರ(ಖಂಡ_q4_0);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * ಲಂಬಸಾಲು_src = src + x;

            // --- 1) Gather 32 scale values (ggml_half d) with stride load ---
            // d is at offset 0 of each block_q4_0, stride between rows = row_stride
            {
                ಸ್ಥಿರಾಂಕ uint8_t * d_ತಳ    = (ಸ್ಥಿರಾಂಕ uint8_t *) &ಲಂಬಸಾಲು_src->d;
                ggml_half *     d_dst     = dst->d;
                size_t          ಉಳಿದ = 32;
                size_t          ಸ್ಥಾನಾಂತರ    = 0;
                ಆಗ (ಉಳಿದ > 0) {
                    size_t      vl = __riscv_vsetvl_e16m1(ಉಳಿದ);
                    vuint16m1_t vd =
                        __riscv_vlse16_v_u16m1((ಸ್ಥಿರಾಂಕ uint16_t *) (d_ತಳ + ಸ್ಥಾನಾಂತರ * ಅಡ್ಡಸಾಲು_stride), ಅಡ್ಡಸಾಲು_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + ಸ್ಥಾನಾಂತರ), vd, vl);
                    ಸ್ಥಾನಾಂತರ += vl;
                    ಉಳಿದ -= vl;
                }
            }

            // --- 2) Nibble repack qs for each of the 32 rows ---
            // For each row i:
            //   src qs[16]: [b0|b16] [b1|b17] ... [b15|b31]  (lo nibble = b_j, hi nibble = b_{j+16})
            //   dst qs low  8B: (qs[2j] & 0x0F) | ((qs[2j+1] & 0x0F) << 4)  for j=0..7
            //   dst qs high 8B: ((qs[2j] >> 4))  | (qs[2j+1] & 0xF0)         for j=0..7
            {
                ಸ್ಥಿರಾಂಕ size_t vl8 = __riscv_vsetvl_e8m1(8);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
                    ಸ್ಥಿರಾಂಕ uint8_t * sq = ಲಂಬಸಾಲು_src[i * nblocks].qs;
                    uint8_t *       dq = dst->qs + i * qs_ಬೈಟುಗಳು;

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
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q4_1_ಗಮ್ಯ_q4_1_32_bl_ref(ggml_tensor *              t,
                                         ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                         size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_ಖಂಡ == 32);  //ಬಳಸದ

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    ಖಂಡ_q4_1x32 *    dst = (ಖಂಡ_q4_1x32 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 *) ದತ್ತಾಂಶ;
    ಖಂಡ_q4_1         dst_tmp[32];
    ಪೂರ್ಣಾಂಕ                nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q4_1));

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ಖಂಡ_q4_1x32(dst_tmp, interleave_ಖಂಡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

// RVV optimized version of repack_q4_1_to_q4_1_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes nibble repack + zp computation.
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q4_1_ಗಮ್ಯ_q4_1_32_bl(ggml_tensor *              t,
                                     ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                     size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_ಖಂಡ == 32);

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;
    constexpr ಪೂರ್ಣಾಂಕ qs_ಬೈಟುಗಳು          = QK4_1 / 2;  // 16

    ಖಂಡ_q4_1x32 *    dst     = (ಖಂಡ_q4_1x32 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * src     = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 *) ದತ್ತಾಂಶ;
    ಪೂರ್ಣಾಂಕ                nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q4_1));

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ptrdiff_t ಅಡ್ಡಸಾಲು_stride = (ptrdiff_t) nblocks * ಗಾತ್ರ(ಖಂಡ_q4_1);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_1 * ಲಂಬಸಾಲು_src = src + x;

            // --- 1) Gather d and m, compute zp = clamp(nearbyint(-m/d), 0, 15) ---
            // block_q4_1 layout: [d(f16), m(f16), qs[16]]
            // d is at byte offset 0, m is at byte offset 2 from each block start
            {
                ಸ್ಥಿರಾಂಕ uint8_t * dm_ತಳ   = (ಸ್ಥಿರಾಂಕ uint8_t *) &ಲಂಬಸಾಲು_src->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
                ggml_half *     d_dst     = dst->d;
                uint8_t *       zp_dst    = dst->zp;
                size_t          ಉಳಿದ = 32;
                size_t          ಸ್ಥಾನಾಂತರ    = 0;
                ಆಗ (ಉಳಿದ > 0) {
                    size_t vl = __riscv_vsetvl_e16m1(ಉಳಿದ);

                    // stride load d (f16) from each row
                    vuint16m1_t vd_raw =
                        __riscv_vlse16_v_u16m1((ಸ್ಥಿರಾಂಕ uint16_t *) (dm_ತಳ + ಸ್ಥಾನಾಂತರ * ಅಡ್ಡಸಾಲು_stride), ಅಡ್ಡಸಾಲು_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + ಸ್ಥಾನಾಂತರ), vd_raw, vl);

                    // stride load m (f16) from each row (offset +2 bytes from d)
                    vuint16m1_t vm_raw =
                        __riscv_vlse16_v_u16m1((ಸ್ಥಿರಾಂಕ uint16_t *) (dm_ತಳ + 2 + ಸ್ಥಾನಾಂತರ * ಅಡ್ಡಸಾಲು_stride), ಅಡ್ಡಸಾಲು_stride, vl);

                    // convert to f32 for zp computation: zp = nearbyint(-m / d)
                    vfloat16m1_t vd_f16 = __riscv_vreinterpret_v_u16m1_f16m1(vd_raw);
                    vfloat16m1_t vm_f16 = __riscv_vreinterpret_v_u16m1_f16m1(vm_raw);

                    // -m / d in f16 directly (SpaceMIT X60 supports f16 arithmetic)
                    vfloat16m1_t v_neg_m = __riscv_vfneg_v_f16m1(vm_f16, vl);
                    vfloat16m1_t v_ಅನುಪಾತ = __riscv_vfdiv_vv_f16m1(v_neg_m, vd_f16, vl);

                    // Convert to f32 for nearbyint, then clamp
                    vfloat32m2_t v_ಅನುಪಾತ_f32 = __riscv_vfwcvt_f_f_v_f32m2(v_ಅನುಪಾತ, vl);

                    // Use integer rounding: convert f32 -> int (rounds to nearest)
                    vint32m2_t v_zp_i32 = __riscv_vfcvt_x_f_v_i32m2(v_ಅನುಪಾತ_f32, vl);

                    // clamp to [0, 15]
                    v_zp_i32 = __riscv_vmax_vx_i32m2(v_zp_i32, 0, vl);
                    v_zp_i32 = __riscv_vmin_vx_i32m2(v_zp_i32, 15, vl);

                    // narrow i32 -> u8
                    vint16m1_t  v_zp_i16 = __riscv_vncvt_x_x_w_i16m1(v_zp_i32, vl);
                    vint8mf2_t  v_zp_i8  = __riscv_vncvt_x_x_w_i8mf2(v_zp_i16, vl);
                    vuint8mf2_t v_zp_u8  = __riscv_vreinterpret_v_i8mf2_u8mf2(v_zp_i8);
                    __riscv_vse8_v_u8mf2(zp_dst + ಸ್ಥಾನಾಂತರ, v_zp_u8, vl);

                    ಸ್ಥಾನಾಂತರ += vl;
                    ಉಳಿದ -= vl;
                }
            }

            // --- 2) Nibble repack qs for each of the 32 rows ---
            {
                ಸ್ಥಿರಾಂಕ size_t vl8 = __riscv_vsetvl_e8m1(8);
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
                    ಸ್ಥಿರಾಂಕ uint8_t * sq = ಲಂಬಸಾಲು_src[i * nblocks].qs;
                    uint8_t *       dq = dst->qs + i * qs_ಬೈಟುಗಳು;

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
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q4_k_ಗಮ್ಯ_q4_1_32_bl(ggml_tensor *              t,
                                     ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                     size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_ಖಂಡ == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    ಖಂಡ_q4_1x32 *    dst = (ಖಂಡ_q4_1x32 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K *) ದತ್ತಾಂಶ;
    ಖಂಡ_q4_1         dst_tmp[32];
    ಪೂರ್ಣಾಂಕ                nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                nblocks = t->ne[0] / QK_K;

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_FP16_ಗಮ್ಯ_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ min =
                        GGML_FP16_ಗಮ್ಯ_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1 = d * sc;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ m1 = min * m;

                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_ಗಮ್ಯ_FP16(d1);
                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_ಗಮ್ಯ_FP16(-m1);
                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    ಸ್ಥಿರಾಂಕ uint8_t * q                                  = src[x + i * nblocks].qs + (j / 2) * QK4_1;
                    ಆದರೆ (j % 2 == 0) {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }
                }
                *dst++ = make_ಖಂಡ_q4_1x32(dst_tmp, interleave_ಖಂಡ);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q6_k_ಗಮ್ಯ_q8_0_32_bl_ref(ggml_tensor *              t,
                                         ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                         size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_ಖಂಡ == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    ಖಂಡ_q8_0x32 *    dst = (ಖಂಡ_q8_0x32 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K *) ದತ್ತಾಂಶ;
    ಖಂಡ_q8_0         dst_tmp[32];
    int8_t             aux8[QK4_1];
    ಪೂರ್ಣಾಂಕ                nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                nblocks = t->ne[0] / QK_K;

    ಆದರೆ (t->ne[0] % QK_K != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        int64_t nrow_real = std::min((int64_t) nrow - b, (int64_t) nrows_interleaved);
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ bi = 0; bi < 8; bi++) {
                ಪೂರ್ಣಾಂಕ i = 0;
                ಗಾಗಿ (; i < nrow_real; i++) {
                    ಸ್ಥಿರಾಂಕ uint8_t * q4     = src[x + i * nblocks].ql;
                    ಸ್ಥಿರಾಂಕ uint8_t * qh     = src[x + i * nblocks].qh;
                    ಸ್ಥಿರಾಂಕ int8_t *  scales = src[x + i * nblocks].scales;
                    ದಶಮಾಂಶ           d      = GGML_FP16_ಗಮ್ಯ_FP32(src[x + i * nblocks].d);

                    q4 += 64 * (bi / 4);
                    qh += 32 * (bi / 4);
                    int8_t * GGML_RESTRICT a = aux8;

                    int8_t bi_idx = bi % 4;

                    ಆದರೆ (bi_idx == 0) {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (bi_idx == 1) {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (bi_idx == 2) {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (bi_idx == 3) {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        }
                    }
                    a = aux8;

                    ದಶಮಾಂಶ a_max_abs = 0.0f;
                    ದಶಮಾಂಶ scale_0   = scales[bi * 2 + 0] * d;
                    ದಶಮಾಂಶ scale_1   = scales[bi * 2 + 1] * d;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 16; ++l) {
                        a_max_abs = std::max(a_max_abs, std::ನಿರಪೇಕ್ಷ(a[l] * scale_0));
                    }

                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 16; l < 32; ++l) {
                        a_max_abs = std::max(a_max_abs, std::ನಿರಪೇಕ್ಷ(a[l] * scale_1));
                    }

                    ದಶಮಾಂಶ reflect_scale   = a_max_abs / ((1 << 7) - 1);
                    ದಶಮಾಂಶ reflect_scale_0 = scale_0 / reflect_scale;
                    ದಶಮಾಂಶ reflect_scale_1 = scale_1 / reflect_scale;

                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < 16; ++l) {
                        ದಶಮಾಂಶ a_ತಾತ್ಕಾಲಿಕ = std::clamp(std::nearbyintf(a[l] * reflect_scale_0), -128.0f, 127.0f);
                        a[l]         = (int8_t) (a_ತಾತ್ಕಾಲಿಕ);
                    }

                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 16; l < 32; ++l) {
                        ದಶಮಾಂಶ a_ತಾತ್ಕಾಲಿಕ = std::clamp(std::nearbyintf(a[l] * reflect_scale_1), -128.0f, 127.0f);
                        a[l]         = (int8_t) (a_ತಾತ್ಕಾಲಿಕ);
                    }

                    dst_tmp[i].d = GGML_FP32_ಗಮ್ಯ_FP16(reflect_scale);

                    ಸ್ಮರಣೆ_ನಕಲು(dst_tmp[i].qs, a, 32 * ಗಾತ್ರ(int8_t));
                }

                ಗಾಗಿ (; i < nrows_interleaved; i++) {
                    memset(&dst_tmp[i], 0, ಗಾತ್ರ(ಖಂಡ_q8_0));
                }

                *dst++ = make_ಖಂಡ_q8_0x32(dst_tmp, interleave_ಖಂಡ);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

// RVV optimized version of repack_q6_k_to_q8_0_32_bl
// Vectorizes the Q6_K dequant -> requant pipeline using RVV intrinsics.
// For each sub-block (bi), dequant 32 Q6_K values to int6 -> apply two sub-block scales ->
// find max abs -> compute reflect_scale -> requant to int8 -> gather d with stride load.
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q6_k_ಗಮ್ಯ_q8_0_32_bl(ggml_tensor *              t,
                                     ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                     size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_ಖಂಡ == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    ಖಂಡ_q8_0x32 *    dst     = (ಖಂಡ_q8_0x32 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * src     = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K *) ದತ್ತಾಂಶ;
    ಪೂರ್ಣಾಂಕ                nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                nblocks = t->ne[0] / QK_K;

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ptrdiff_t ಅಡ್ಡಸಾಲು_stride = (ptrdiff_t) nblocks * ಗಾತ್ರ(ಖಂಡ_q6_K);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ bi = 0; bi < 8; bi++) {
                // --- 1) Gather 32 d values with stride load ---
                // We need to compute reflect_scale per row first, so gather d later.
                // Process each row: dequant Q6_K sub-block -> requant to Q8_0
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                    ಸ್ಥಿರಾಂಕ ಖಂಡ_q6_K * src_blk = &src[x + i * nblocks];
                    ಸ್ಥಿರಾಂಕ uint8_t *    q4      = src_blk->ql + 64 * (bi / 4);
                    ಸ್ಥಿರಾಂಕ uint8_t *    qh      = src_blk->qh + 32 * (bi / 4);
                    ಸ್ಥಿರಾಂಕ int8_t *     scales  = src_blk->scales;
                    ದಶಮಾಂಶ              d       = GGML_FP16_ಗಮ್ಯ_FP32(src_blk->d);

                    int8_t bi_idx = bi % 4;

                    // --- Dequant 32 Q6_K values to int6 (range [-32, 31]) using RVV ---
                    // vl = 32 for e8m2 (VLEN=256) or loop for smaller VLEN
                    ಸ್ಥಿರಾಂಕ size_t vl16 = __riscv_vsetvl_e8m1(16);

                    vint8m1_t va_lo, va_hi;  // 16 elements each

                    ಆದರೆ (bi_idx == 0) {
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
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (bi_idx == 1) {
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
                    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (bi_idx == 2) {
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
                    } ಇಲ್ಲದಿದ್ದರೆ {  // bi_idx == 3
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
                    ದಶಮಾಂಶ scale_0 = scales[bi * 2 + 0] * d;
                    ದಶಮಾಂಶ scale_1 = scales[bi * 2 + 1] * d;

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
                    ದಶಮಾಂಶ        a_max_abs = __riscv_vfmv_f_s_f32m1_f32(vmax_red);

                    ದಶಮಾಂಶ reflect_scale   = a_max_abs / 127.0f;
                    ದಶಮಾಂಶ reflect_scale_0 = scale_0 / reflect_scale;
                    ದಶಮಾಂಶ reflect_scale_1 = scale_1 / reflect_scale;

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
                    dst->d[i]   = GGML_FP32_ಗಮ್ಯ_FP16(reflect_scale);
                    int8_t * dq = (int8_t *) dst->qs + i * QK8_0;
                    __riscv_vse8_v_i8m1(dq, vi8_lo, vl16);
                    __riscv_vse8_v_i8m1(dq + 16, vi8_hi, vl16);
                }
                dst++;
            }
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q8_0_ಗಮ್ಯ_q8_0_32_bl_ref(ggml_tensor *              t,
                                         ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                         ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                         size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_ಖಂಡ == 32);  //ಬಳಸದ

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    ಖಂಡ_q8_0x32 *    dst = (ಖಂಡ_q8_0x32 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) ದತ್ತಾಂಶ;
    ಖಂಡ_q8_0         dst_tmp[32];
    ಪೂರ್ಣಾಂಕ                nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q8_0));

    ಆದರೆ (t->ne[0] % QK8_0 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        int64_t nrows_real = std::min((int64_t) nrow - b, (int64_t) nrows_interleaved);
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಪೂರ್ಣಾಂಕ i = 0;
            ಗಾಗಿ (; i < nrows_real; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            ಗಾಗಿ (; i < nrows_interleaved; i++) {
                memset(&dst_tmp[i], 0, ಗಾತ್ರ(ಖಂಡ_q8_0));
            }
            *dst++ = make_ಖಂಡ_q8_0x32(dst_tmp, interleave_ಖಂಡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

// RVV optimized version of repack_q8_0_to_q8_0_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes scale gather + qs copy.
ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q8_0_ಗಮ್ಯ_q8_0_32_bl(ggml_tensor *              t,
                                     ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                     size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_ಖಂಡ == 32);

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    ಖಂಡ_q8_0x32 *    dst     = (ಖಂಡ_q8_0x32 *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * src     = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) ದತ್ತಾಂಶ;
    ಪೂರ್ಣಾಂಕ                nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q8_0));

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK8_0 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ptrdiff_t ಅಡ್ಡಸಾಲು_stride = (ptrdiff_t) nblocks * ಗಾತ್ರ(ಖಂಡ_q8_0);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * ಲಂಬಸಾಲು_src = src + x;

            // --- 1) Gather 32 scale values (ggml_half d) with stride load ---
            {
                ಸ್ಥಿರಾಂಕ uint8_t * d_ತಳ    = (ಸ್ಥಿರಾಂಕ uint8_t *) &ಲಂಬಸಾಲು_src->d;
                ggml_half *     d_dst     = dst->d;
                size_t          ಉಳಿದ = 32;
                size_t          ಸ್ಥಾನಾಂತರ    = 0;
                ಆಗ (ಉಳಿದ > 0) {
                    size_t      vl = __riscv_vsetvl_e16m1(ಉಳಿದ);
                    vuint16m1_t vd =
                        __riscv_vlse16_v_u16m1((ಸ್ಥಿರಾಂಕ uint16_t *) (d_ತಳ + ಸ್ಥಾನಾಂತರ * ಅಡ್ಡಸಾಲು_stride), ಅಡ್ಡಸಾಲು_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + ಸ್ಥಾನಾಂತರ), vd, vl);
                    ಸ್ಥಾನಾಂತರ += vl;
                    ಉಳಿದ -= vl;
                }
            }

            // --- 2) Copy qs for each of the 32 rows (32 bytes per row) ---
            {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
                    ಸ್ಥಿರಾಂಕ int8_t * sq = ಲಂಬಸಾಲು_src[i * nblocks].qs;
                    int8_t *       dq = (int8_t *) dst->qs + i * QK8_0;

                    size_t len = QK8_0;
                    size_t idx = 0;
                    ಆಗ (len > 0) {
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
    ಹಿಂತಿರುಗಿ 0;

    GGML_ಬಳಸದ(ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಸ್ಥಿರ ಶೂನ್ಯ convert_mxfp4_ಗಮ್ಯ_5bit(ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 & src, spacemit_kernels::nrow_ಖಂಡ_mxfp4<1> & dst) {
    dst.e[0] = src.e;

    // Decode all 32 mxfp4 values to signed integers via kvalues_mxfp4
    int8_t vals[32];
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_MXFP4 / 2; j++) {
        vals[j]                = kvalues_mxfp4[src.qs[j] & 0xF];
        vals[j + QK_MXFP4 / 2] = kvalues_mxfp4[src.qs[j] >> 4];
    }

    // vals [b0, b1, b2, b3, ..., b30, b31]
    // Pack abs into qs with reorder: [b0,b1]..[b14,b15]..[b30,b31]
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_MXFP4 / 2; j++) {
        uint8_t lo0 = static_cast<uint8_t>(std::ನಿರಪೇಕ್ಷ(vals[j * 2]));
        uint8_t lo1 = static_cast<uint8_t>(std::ನಿರಪೇಕ್ಷ(vals[j * 2 + 1]));
        dst.qs[j]   = (lo0 & 0x0F) | ((lo1 & 0x0F) << 4);
    }

    // Pack sign bits into qh[4] (32 bits total, 1 bit per weight)
    // reorder: [0,1,2,...,15,16,17,...,31] after the qs reorder above
    uint32_t sign_ಬಿಟ್ಟುಗಳು = 0;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 32; j++) {
        ಆದರೆ (vals[j] < 0) {
            sign_ಬಿಟ್ಟುಗಳು |= (1u << j);
        }
    }
    ಸ್ಮರಣೆ_ನಕಲು(dst.qh, &sign_ಬಿಟ್ಟುಗಳು, 4);
}

ಸ್ಥಿರ spacemit_kernels::nrow_ಖಂಡ_mxfp4<32> make_ಖಂಡ_mxfp4x32(spacemit_kernels::nrow_ಖಂಡ_mxfp4<1> * ಒಳ,
                                                                  ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ_interleave) {
    spacemit_kernels::nrow_ಖಂಡ_mxfp4<32> ಹೊರಹೋಗು;
    GGML_ASSERT(QK_MXFP4 / blck_ಗಾತ್ರ_interleave == 1);
    GGML_ಬಳಸದ(blck_ಗಾತ್ರ_interleave);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        ಹೊರಹೋಗು.e[i] = ಒಳ[i].e[0];
    }

    // qs: copy per-row 16 bytes
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        ಸ್ಮರಣೆ_ನಕಲು(ಹೊರಹೋಗು.qs + i * 16, ಒಳ[i].qs, 16);
    }

    // qh: copy per-row 4 bytes
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        ಸ್ಮರಣೆ_ನಕಲು(ಹೊರಹೋಗು.qh + i * 4, ಒಳ[i].qh, 4);
    }

    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_mxfp4_ಗಮ್ಯ_mxfp4_32_bl(ggml_tensor *              t,
                                       ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                       ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                       size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_ಖಂಡ == 32);

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    spacemit_kernels::nrow_ಖಂಡ_mxfp4<32> * dst = (spacemit_kernels::nrow_ಖಂಡ_mxfp4<32> *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 *                      src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_mxfp4 *) ದತ್ತಾಂಶ;
    spacemit_kernels::nrow_ಖಂಡ_mxfp4<1>    dst_tmp[32];
    ಪೂರ್ಣಾಂಕ                                      nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                                      nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_mxfp4));

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_MXFP4 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                convert_mxfp4_ಗಮ್ಯ_5bit(src[x + i * nblocks], dst_tmp[i]);
            }
            *dst++ = make_ಖಂಡ_mxfp4x32(dst_tmp, interleave_ಖಂಡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ spacemit_kernels::nrow_ಖಂಡ_q5_1<32> make_ಖಂಡ_q5_1x32(spacemit_kernels::nrow_ಖಂಡ_q5_1<1> * ಒಳ,
                                                                ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ_interleave) {
    spacemit_kernels::nrow_ಖಂಡ_q5_1<32> ಹೊರಹೋಗು;
    GGML_ASSERT(QK5_1 / blck_ಗಾತ್ರ_interleave == 1);
    GGML_ಬಳಸದ(blck_ಗಾತ್ರ_interleave);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        ಹೊರಹೋಗು.scales16[i] = ಒಳ[i].scales16[0];
        ಹೊರಹೋಗು.zp[i]       = ಒಳ[i].zp[0];
    }

    // qs: low 4 bits, reorder from [b0,b16],[b1,b17]... to [b0,b1]...[b14,b15] and [b16,b17]...[b30,b31]
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        // low half [0..15]
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK5_1 / 4; j++) {
            ಹೊರಹೋಗು.qs[i * QK5_1 / 2 + j] = (ಒಳ[i].qs[j * 2] & 0x0F) | ((ಒಳ[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
        // high half [16..31]
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK5_1 / 4; j++) {
            ಹೊರಹೋಗು.qs[i * QK5_1 / 2 + QK5_1 / 4 + j] = ((ಒಳ[i].qs[j * 2] & 0xF0) >> 4) | (ಒಳ[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    // qh: 5th bit, copy directly
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; j++) {
            ಹೊರಹೋಗು.qh[i * 4 + j] = ಒಳ[i].qh[j];
        }
    }

    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು;
}

ಸ್ಥಿರ spacemit_kernels::nrow_ಖಂಡ_q5_0<32> make_ಖಂಡ_q5_0x32(spacemit_kernels::nrow_ಖಂಡ_q5_0<1> * ಒಳ,
                                                                ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ blck_ಗಾತ್ರ_interleave) {
    spacemit_kernels::nrow_ಖಂಡ_q5_0<32> ಹೊರಹೋಗು;
    GGML_ASSERT(QK5_0 / blck_ಗಾತ್ರ_interleave == 1);
    GGML_ಬಳಸದ(blck_ಗಾತ್ರ_interleave);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        ಹೊರಹೋಗು.scales16[i] = ಒಳ[i].scales16[0];
    }

    // qs: low 4 bits, reorder from [b0,b16],[b1,b17]... to [b0,b1]...[b14,b15] and [b16,b17]...[b30,b31]
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        // low half [0..15]
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK5_0 / 4; j++) {
            ಹೊರಹೋಗು.qs[i * QK5_0 / 2 + j] = (ಒಳ[i].qs[j * 2] & 0x0F) | ((ಒಳ[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
        // high half [16..31]
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK5_0 / 4; j++) {
            ಹೊರಹೋಗು.qs[i * QK5_0 / 2 + QK5_0 / 4 + j] = ((ಒಳ[i].qs[j * 2] & 0xF0) >> 4) | (ಒಳ[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    // qh: 5th bit, copy directly
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 32; i++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 4; j++) {
            ಹೊರಹೋಗು.qh[i * 4 + j] = ಒಳ[i].qh[j];
        }
    }

    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q5_0_ಗಮ್ಯ_q5_0_32_bl(ggml_tensor *              t,
                                     ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                     size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(interleave_ಖಂಡ == 32);  //ಬಳಸದ

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    spacemit_kernels::nrow_ಖಂಡ_q5_0<32> * dst = (spacemit_kernels::nrow_ಖಂಡ_q5_0<32> *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 *                      src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 *) ದತ್ತಾಂಶ;
    spacemit_kernels::nrow_ಖಂಡ_q5_0<1>    dst_tmp[32];
    ಪೂರ್ಣಾಂಕ                                     nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                                     nblocks = t->ne[0] / QK5_0;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q5_0));

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK5_0 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_0 & s = src[x + i * nblocks];

                dst_tmp[i].scales16[0] = s.d;
                ಸ್ಮರಣೆ_ನಕಲು(dst_tmp[i].qs, s.qs, ಗಾತ್ರ(dst_tmp[i].qs));
                ಸ್ಮರಣೆ_ನಕಲು(dst_tmp[i].qh, s.qh, ಗಾತ್ರ(dst_tmp[i].qh));
            }
            *dst++ = make_ಖಂಡ_q5_0x32(dst_tmp, interleave_ಖಂಡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q5_1_ಗಮ್ಯ_q5_1_32_bl(ggml_tensor *              t,
                                     ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                     size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(interleave_ಖಂಡ == 32);  //ಬಳಸದ

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    spacemit_kernels::nrow_ಖಂಡ_q5_1<32> * dst = (spacemit_kernels::nrow_ಖಂಡ_q5_1<32> *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 *                      src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 *) ದತ್ತಾಂಶ;
    spacemit_kernels::nrow_ಖಂಡ_q5_1<1>    dst_tmp[32];
    ಪೂರ್ಣಾಂಕ                                     nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                                     nblocks = t->ne[0] / QK5_1;

    GGML_ASSERT(ದತ್ತಾಂಶ_ಗಾತ್ರ == nrow * nblocks * ಗಾತ್ರ(ಖಂಡ_q5_1));

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK5_1 != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_1 & s = src[x + i * nblocks];

                ದಶಮಾಂಶ d = GGML_FP16_ಗಮ್ಯ_FP32(s.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                ದಶಮಾಂಶ m = GGML_FP16_ಗಮ್ಯ_FP32(s.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);

                ಆದರೆ (d == 0.0f) {
                    dst_tmp[i].scales16[0] = GGML_FP32_ಗಮ್ಯ_FP16(std::fabs(m));
                    dst_tmp[i].zp[0]       = m < 0.0f ? 1 : 0;
                    memset(dst_tmp[i].qh, 0, ಗಾತ್ರ(dst_tmp[i].qh));
                    memset(dst_tmp[i].qs, m > 0.0f ? 0x11 : 0x00, ಗಾತ್ರ(dst_tmp[i].qs));
                    ಮುಂದುವರಿ;
                }

                ದಶಮಾಂಶ mid = std::nearbyintf(-m / d);
                mid       = std::min(31.0f, std::max(0.0f, mid));

                dst_tmp[i].scales16[0] = GGML_FP32_ಗಮ್ಯ_FP16(d);
                dst_tmp[i].zp[0]       = static_cast<uint8_t>(mid);

                // qs: copy low 4 bits directly (same nibble packing)
                ಸ್ಮರಣೆ_ನಕಲು(dst_tmp[i].qs, s.qs, QK5_1 / 2);

                // qh: copy 5th bit directly
                ಸ್ಮರಣೆ_ನಕಲು(dst_tmp[i].qh, s.qh, 4);
            }
            *dst++ = make_ಖಂಡ_q5_1x32(dst_tmp, interleave_ಖಂಡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ repack_q5_k_ಗಮ್ಯ_q5_1_32_bl(ggml_tensor *              t,
                                     ಪೂರ್ಣಾಂಕ                        interleave_ಖಂಡ,
                                     ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * GGML_RESTRICT ದತ್ತಾಂಶ,
                                     size_t                     ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_ಖಂಡ == 32);
    GGML_ASSERT(QK_K / QK5_1 == 8);

    constexpr ಪೂರ್ಣಾಂಕ nrows_interleaved = 32;

    spacemit_kernels::nrow_ಖಂಡ_q5_1<32> * dst = (spacemit_kernels::nrow_ಖಂಡ_q5_1<32> *) t->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K *                      src = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q5_K *) ದತ್ತಾಂಶ;
    spacemit_kernels::nrow_ಖಂಡ_q5_1<1>    dst_tmp[32];
    ಪೂರ್ಣಾಂಕ                                     nrow    = ggml_nrows(t);
    ಪೂರ್ಣಾಂಕ                                     nblocks = t->ne[0] / QK_K;

    ಆದರೆ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ b = 0; b < nrow; b += nrows_interleaved) {
        ಗಾಗಿ (int64_t x = 0; x < nblocks; x++) {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) {
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d = GGML_FP16_ಗಮ್ಯ_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ min =
                        GGML_FP16_ಗಮ್ಯ_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);

                    ದಶಮಾಂಶ d1 = d * sc;
                    ದಶಮಾಂಶ m1 = min * m;

                    ದಶಮಾಂಶ mid              = std::nearbyintf(m1 / d1);
                    mid                    = std::min(31.0f, std::max(0.0f, mid));
                    dst_tmp[i].scales16[0] = GGML_FP32_ಗಮ್ಯ_FP16(d1);
                    dst_tmp[i].zp[0]       = static_cast<uint8_t>(mid);

                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    ಸ್ಥಿರಾಂಕ uint8_t * q = src[x + i * nblocks].qs + (j / 2) * QK5_1;
                    ಆದರೆ (j % 2 == 0) {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ ii = 0; ii < 16; ii++) {
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
                    ಸ್ಥಿರಾಂಕ uint8_t * src_qh = src[x + i * nblocks].qh;
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ bi = 0; bi < 4; bi++) {
                        uint8_t qh_byte = 0;
                        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; k++) {
                            ಪೂರ್ಣಾಂಕ src_idx = bi * 8 + k;
                            qh_byte |= ((src_qh[src_idx] >> j) & 1) << k;
                        }
                        dst_tmp[i].qh[bi] = qh_byte;
                    }
                }
                *dst++ = make_ಖಂಡ_q5_1x32(dst_tmp, interleave_ಖಂಡ);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    ಹಿಂತಿರುಗಿ 0;
}

ನಾಮಾಕಾಶ ggml::cpu::riscv64_spacemit {

ಮಾದರಿ <ಬಗೆ_ಹೆಸರು BLOC_TYPE, int64_t INTER_ಗಾತ್ರ, int64_t NB_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ> ಪೂರ್ಣಾಂಕ repack(ggml_tensor *, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *, size_t);

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q4_0, 32, 16>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ repack_q4_0_ಗಮ್ಯ_q4_0_16_bl(t, 16, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q4_1, 32, 16>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ repack_q4_1_ಗಮ್ಯ_q4_1_16_bl(t, 16, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q4_K, 32, 16>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ repack_q4_k_ಗಮ್ಯ_q4_1_16_bl(t, 16, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q2_K, 256, 32>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ repack_q2_k_ಗಮ್ಯ_q2_k_32_bl(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q3_K, 256, 32>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ repack_q3_k_ಗಮ್ಯ_q3_k_32_bl(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q4_0, 32, 32>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
#ಆದರೆ 0
    ಹಿಂತಿರುಗಿ repack_q4_0_ಗಮ್ಯ_q4_0_32_bl_ref(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ repack_q4_0_ಗಮ್ಯ_q4_0_32_bl(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q4_0, 256, 32>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
#ಆದರೆ 1
    ಹಿಂತಿರುಗಿ repack_q4_0_ಗಮ್ಯ_q4_0_256_32_bl_ref(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
#ಇಲ್ಲದಿದ್ದರೆ
    //return repack_q4_0_to_q4_0_256_32_bl(t, 32, data, data_size);
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q4_1, 32, 32>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
#ಆದರೆ 0
    ಹಿಂತಿರುಗಿ repack_q4_1_ಗಮ್ಯ_q4_1_32_bl_ref(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ repack_q4_1_ಗಮ್ಯ_q4_1_32_bl(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q4_1, 256, 32>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
#ಆದರೆ 1
    ಹಿಂತಿರುಗಿ repack_q4_0_ಗಮ್ಯ_q4_1_256_32_bl_ref(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ repack_q4_1_ಗಮ್ಯ_q4_1_256_32_bl(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q4_K, 32, 32>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ repack_q4_k_ಗಮ್ಯ_q4_1_32_bl(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q6_K, 32, 32>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
#ಆದರೆ 1
    ಹಿಂತಿರುಗಿ repack_q6_k_ಗಮ್ಯ_q8_0_32_bl_ref(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ repack_q6_k_ಗಮ್ಯ_q8_0_32_bl(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q8_0, 32, 16>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ repack_q8_0_ಗಮ್ಯ_q8_0_16_bl_ref(t, 16, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q8_0, 32, 32>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
#ಆದರೆ 1
    ಹಿಂತಿರುಗಿ repack_q8_0_ಗಮ್ಯ_q8_0_32_bl_ref(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
#ಇಲ್ಲದಿದ್ದರೆ
    ಹಿಂತಿರುಗಿ repack_q8_0_ಗಮ್ಯ_q8_0_32_bl(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
#ಆದರೆ_ಅಂತ್ಯ
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_mxfp4, 32, 32>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ repack_mxfp4_ಗಮ್ಯ_mxfp4_32_bl(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q5_0, 32, 32>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ repack_q5_0_ಗಮ್ಯ_q5_0_32_bl(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q5_1, 32, 32>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ repack_q5_1_ಗಮ್ಯ_q5_1_32_bl(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

ಮಾದರಿ <> ಪೂರ್ಣಾಂಕ repack<ಖಂಡ_q5_K, 32, 32>(ggml_tensor * t, ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ದತ್ತಾಂಶ, size_t ದತ್ತಾಂಶ_ಗಾತ್ರ) {
    ಹಿಂತಿರುಗಿ repack_q5_k_ಗಮ್ಯ_q5_1_32_bl(t, 32, ದತ್ತಾಂಶ, ದತ್ತಾಂಶ_ಗಾತ್ರ);
}

}  // namespace ggml::cpu::riscv64_spacemit
