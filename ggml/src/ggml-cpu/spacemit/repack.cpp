#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_IMPL_CPP
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_CPP

#ਅਨ੍ਤਰ੍ਭਾਵ "repack.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-cpu.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-impl.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ime_kernels.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <algorithm>
#ਅਨ੍ਤਰ੍ਭਾਵ <cassert>
#ਅਨ੍ਤਰ੍ਭਾਵ <cmath>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstring>

// clang-format off
#ਜੇ defined(__riscv)

#ਜੇ !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#ਗਲਤੀ "riscv v extension or v_intrinsic not enabled"
#ਨਹੀਂਤਾਂ
#ਅਨ੍ਤਰ੍ਭਾਵ <riscv_vector.h>
#ਜੇ_ਅੰਤ

#ਜੇ !defined(__riscv_zfh)
#ਗਲਤੀ "riscv zfh extension not enabled"
#ਜੇ_ਅੰਤ

#ਨਹੀਂਤਾਂ
#ਗਲਤੀ "riscv not enabled in this build"
#ਜੇ_ਅੰਤ

#ਜੇ defined(__GNUC__)
#ਹਦਾਇਤ GCC diagnostic ignored "-Wcast-qual"
#ਹਦਾਇਤ GCC diagnostic ignored "-Wunused-parameter"
#ਜੇ_ਅੰਤ

// clang-format on

ਸਾਂਚਾ <ਪੂਰਨ K> constexpr ਪੂਰਨ QK_0() {
    ਜੇ constexpr (K == 4) {
        ਵਾਪਸ QK4_0;
    }
    ਜੇ constexpr (K == 8) {
        ਵਾਪਸ QK8_0;
    }
    ਵਾਪਸ -1;
}

ਸਾਂਚਾ <ਪੂਰਨ K, ਪੂਰਨ N> ਬਣਤਰ ਖੰਡ {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    uint8_t   qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

ਸਾਂਚਾ <ਪੂਰਨ K, ਪੂਰਨ N> ਬਣਤਰ ਖੰਡ_with_zp {
    ggml_half d[N];                         // deltas for N qK_1 blocks
    uint8_t   zp[N];                        // zero points for N qK_1 blocks
    uint8_t   qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_1 blocks
};

// control size
static_assert(ਆਕਾਰ(ਖੰਡ<4, 16>) == 16 * ਆਕਾਰ(ggml_half) + QK4_0 * 8, "wrong block<4,16> size/padding");
static_assert(ਆਕਾਰ(ਖੰਡ_with_zp<4, 16>) == 16 * ਆਕਾਰ(ggml_half) + QK4_0 * 8 + 16 * ਆਕਾਰ(uint8_t),
              "wrong block_with_zp<4,16> size/padding");

static_assert(ਆਕਾਰ(ਖੰਡ<8, 16>) == 16 * ਆਕਾਰ(ggml_half) + QK4_0 * 16, "wrong block<8,16> size/padding");

static_assert(ਆਕਾਰ(ਖੰਡ<4, 32>) == 32 * ਆਕਾਰ(ggml_half) + QK4_0 * 16, "wrong block<4,32> size/padding");
static_assert(ਆਕਾਰ(ਖੰਡ_with_zp<4, 32>) == 32 * ਆਕਾਰ(ggml_half) + QK4_0 * 16 + 32 * ਆਕਾਰ(uint8_t),
              "wrong block_with_zp<4,32> size/padding");

ਵਰਤੋਂ ਖੰਡ_q4_0x16 = ਖੰਡ<4, 16>;
ਵਰਤੋਂ ਖੰਡ_q4_1x16 = ਖੰਡ_with_zp<4, 16>;
ਵਰਤੋਂ ਖੰਡ_q8_0x16 = ਖੰਡ<8, 16>;

ਵਰਤੋਂ ਖੰਡ_q4_0x32 = ਖੰਡ<4, 32>;
ਵਰਤੋਂ ਖੰਡ_q4_1x32 = ਖੰਡ_with_zp<4, 32>;
ਵਰਤੋਂ ਖੰਡ_q8_0x32 = ਖੰਡ<8, 32>;

ਬਣਤਰ ਖੰਡ_q4_0x32x256 {
    ਖੰਡ_q4_0x32 ਖੰਡਾਂ[8];  // [f16 * 32 | i4 * 32 * 32] * 8
};

ਬਣਤਰ ਖੰਡ_q4_1x32x256 {
    ਖੰਡ_q4_0x32 ਖੰਡਾਂ[8];
    uint8_t       zps[32 * 8];
};

ਸਥਿਰ ਖੰਡ_q4_0x16 make_ਖੰਡ_q4_0x16(ਖੰਡ_q4_0 * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q4_0x16 ਬਾਹਰਲਾ;
    GGML_ASSERT(QK4_0 / blck_ਮਾਪ_interleave == 2);

    ਲਈ (ਪੂਰਨ i = 0; i < 16; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].d;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 16; i++) {
        // [0, 15], in.d & 0x0F
        ਲਈ (ਪੂਰਨ j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b8] ......... [b7 b15]
            ਬਾਹਰਲਾ.qs[i * QK4_0 / 4 + j] = (ਵਿੱਚ[i].qs[j] & 0x0F) | ((ਵਿੱਚ[i].qs[j + QK4_0 / 4] & 0x0F) << 4);
        }
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 16; i++) {
        // [16, 31], in.d & 0xF0
        ਲਈ (ਪੂਰਨ j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            ਬਾਹਰਲਾ.qs[4 * QK4_0 + i * QK4_0 / 4 + j] = ((ਵਿੱਚ[i].qs[j] & 0xF0) >> 4) | (ਵਿੱਚ[i].qs[j + QK4_0 / 4] & 0xF0);
        }
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਖੰਡ_q4_1x16 make_ਖੰਡ_q4_1x16(ਖੰਡ_q4_1 * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q4_1x16 ਬਾਹਰਲਾ;
    GGML_ASSERT(QK4_1 / blck_ਮਾਪ_interleave == 2);

    ਲਈ (ਪੂਰਨ i = 0; i < 16; i++) {
        ਦਸ਼ਮਲਵ d   = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(ਵਿੱਚ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        ਦਸ਼ਮਲਵ m   = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(ਵਿੱਚ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
        ਦਸ਼ਮਲਵ mid = -std::nearbyintf(m / d);
        mid       = std::min(15.0f, std::max(0.0f, mid));
        ਬਾਹਰਲਾ.d[i]  = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(d);
        ਬਾਹਰਲਾ.zp[i] = static_cast<uint8_t>(mid);
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 16; i++) {
        // [0, 15], in.d & 0x0F
        ਲਈ (ਪੂਰਨ j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b8] ......... [b7 b15]
            ਬਾਹਰਲਾ.qs[i * QK4_1 / 4 + j] = (ਵਿੱਚ[i].qs[j] & 0x0F) | ((ਵਿੱਚ[i].qs[j + QK4_1 / 4] & 0x0F) << 4);
        }
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 16; i++) {
        // [16, 31], in.d & 0xF0
        ਲਈ (ਪੂਰਨ j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            ਬਾਹਰਲਾ.qs[4 * QK4_1 + i * QK4_1 / 4 + j] = ((ਵਿੱਚ[i].qs[j] & 0xF0) >> 4) | (ਵਿੱਚ[i].qs[j + QK4_1 / 4] & 0xF0);
        }
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਪੂਰਨ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_16_bl(ggml_tensor *              t,
                                     ਪੂਰਨ                        interleave_ਖੰਡ,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                     size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ਖੰਡ == 16);

    constexpr ਪੂਰਨ nrows_interleaved = 16;

    ਖੰਡ_q4_0x16 *    dst = (ਖੰਡ_q4_0x16 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_0 * src = (ਅਚਲ ਖੰਡ_q4_0 *) ਅੰਕੜੇ;
    ਖੰਡ_q4_0         dst_tmp[16];
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q4_0));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q4_0x16(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q4_1_ਮੰਜ਼ਿਲ_q4_1_16_bl(ggml_tensor *              t,
                                     ਪੂਰਨ                        interleave_ਖੰਡ,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                     size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_ਖੰਡ == 16);

    constexpr ਪੂਰਨ nrows_interleaved = 16;

    ਖੰਡ_q4_1x16 *    dst = (ਖੰਡ_q4_1x16 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_1 * src = (ਅਚਲ ਖੰਡ_q4_1 *) ਅੰਕੜੇ;
    ਖੰਡ_q4_1         dst_tmp[16];
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q4_1));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q4_1x16(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ get_scale_min_k4(ਪੂਰਨ                           j,
                                    ਅਚਲ uint8_t * GGML_RESTRICT q,
                                    uint8_t * GGML_RESTRICT       d,
                                    uint8_t * GGML_RESTRICT       m) {
    ਜੇ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } ਨਹੀਂਤਾਂ {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

ਸਥਿਰ ਪੂਰਨ repack_q4_k_ਮੰਜ਼ਿਲ_q4_1_16_bl(ggml_tensor *              t,
                                     ਪੂਰਨ                        interleave_ਖੰਡ,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                     size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_ਖੰਡ == 16);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr ਪੂਰਨ nrows_interleaved = 16;

    ਖੰਡ_q4_1x16 *    dst = (ਖੰਡ_q4_1x16 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_K * src = (ਅਚਲ ਖੰਡ_q4_K *) ਅੰਕੜੇ;
    ਖੰਡ_q4_1         dst_tmp[16];
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK_K;

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
                ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    ਅਚਲ ਦਸ਼ਮਲਵ min =
                        GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);
                    ਅਚਲ ਦਸ਼ਮਲਵ d1 = d * sc;
                    ਅਚਲ ਦਸ਼ਮਲਵ m1 = min * m;

                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(d1);
                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(-m1);
                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    ਅਚਲ uint8_t * q                                  = src[x + i * nblocks].qs + (j / 2) * QK4_1;
                    ਜੇ (j % 2 == 0) {
                        ਲਈ (ਪੂਰਨ ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } ਨਹੀਂਤਾਂ {
                        ਲਈ (ਪੂਰਨ ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }
                }
                *dst++ = make_ਖੰਡ_q4_1x16(dst_tmp, interleave_ਖੰਡ);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਖੰਡ_q4_0x32 make_ਖੰਡ_q4_0x32(ਖੰਡ_q4_0 * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q4_0x32 ਬਾਹਰਲਾ;
    ਪ੍ਰਤਿਜ੍ਞਾ(QK4_0 / blck_ਮਾਪ_interleave == 1);
    GGML_ਅਣਵਰਤਿਆ(blck_ਮਾਪ_interleave);

    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].d;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        // [0, 15], in.d & 0x0F
        ਲਈ (ਪੂਰਨ j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b1] .........  [b14 b15]
            ਬਾਹਰਲਾ.qs[i * QK4_0 / 2 + j] = (ਵਿੱਚ[i].qs[j * 2] & 0x0F) | ((ਵਿੱਚ[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        // [16, 31], in.d & 0xF0
        ਲਈ (ਪੂਰਨ j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b17] ......... [b30 b31]
            ਬਾਹਰਲਾ.qs[i * QK4_0 / 2 + QK4_0 / 4 + j] = ((ਵਿੱਚ[i].qs[j * 2] & 0xF0) >> 4) | (ਵਿੱਚ[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਖੰਡ_q4_1x32 make_ਖੰਡ_q4_1x32(ਖੰਡ_q4_1 * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q4_1x32 ਬਾਹਰਲਾ;
    GGML_ASSERT(QK4_1 / blck_ਮਾਪ_interleave == 1);
    GGML_ਅਣਵਰਤਿਆ(blck_ਮਾਪ_interleave);

    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        ਦਸ਼ਮਲਵ d   = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(ਵਿੱਚ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        ਦਸ਼ਮਲਵ m   = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(ਵਿੱਚ[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
        ਦਸ਼ਮਲਵ mid = -std::nearbyintf(m / d);
        mid       = std::min(15.0f, std::max(0.0f, mid));
        ਬਾਹਰਲਾ.d[i]  = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(d);
        ਬਾਹਰਲਾ.zp[i] = static_cast<uint8_t>(mid);
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        // [0, 15], in.d & 0x0F
        ਲਈ (ਪੂਰਨ j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b1] ......... [b14 b15]
            ਬਾਹਰਲਾ.qs[i * QK4_1 / 2 + j] = (ਵਿੱਚ[i].qs[j * 2] & 0x0F) | ((ਵਿੱਚ[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        // [16, 31], in.d & 0xF0
        ਲਈ (ਪੂਰਨ j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            ਬਾਹਰਲਾ.qs[i * QK4_1 / 2 + QK4_1 / 4 + j] = ((ਵਿੱਚ[i].qs[j * 2] & 0xF0) >> 4) | (ਵਿੱਚ[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਖੰਡ_q8_0x32 make_ਖੰਡ_q8_0x32(ਖੰਡ_q8_0 * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q8_0x32 ਬਾਹਰਲਾ;
    GGML_ASSERT(QK8_0 / blck_ਮਾਪ_interleave == 1);
    GGML_ਅਣਵਰਤਿਆ(blck_ਮਾਪ_interleave);

    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].d;
    }

    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਬਾਹਰਲਾ.qs + i * QK8_0, ਵਿੱਚ[i].qs, QK8_0);
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

// IME1: interleave 16 q8_0 rows so a plain vle8 sequence in the i8i8 kernel lands weights in the
// vmadot group/parity layout. Mirrors make_block_q4_0x16 but stores full int8 (no nibble packing).
// qs (512B) = [INNER step 0: reg0..reg7][INNER step 1: reg0..reg7], each reg 32B holding 4 columns
// x (even=K-first-half / odd=K-second-half). Column col -> acc=col/4, cgrp=col%4.
ਸਥਿਰ ਖੰਡ_q8_0x16 make_ਖੰਡ_q8_0x16(ਖੰਡ_q8_0 * ਵਿੱਚ, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    ਖੰਡ_q8_0x16 ਬਾਹਰਲਾ;
    GGML_ASSERT(QK8_0 / blck_ਮਾਪ_interleave == 2);
    GGML_ਅਣਵਰਤਿਆ(blck_ਮਾਪ_interleave);

    ਲਈ (ਪੂਰਨ i = 0; i < 16; i++) {
        ਬਾਹਰਲਾ.d[i] = ਵਿੱਚ[i].d;
    }

    memset(ਬਾਹਰਲਾ.qs, 0, ਆਕਾਰ(ਬਾਹਰਲਾ.qs));
    ਲਈ (ਪੂਰਨ ਖਾਨਾ = 0; ਖਾਨਾ < 16; ਖਾਨਾ++) {
        ਅਚਲ ਪੂਰਨ      acc  = ਖਾਨਾ / 4;
        ਅਚਲ ਪੂਰਨ      cgrp = ਖਾਨਾ % 4;
        ਅਚਲ int8_t * q    = ਵਿੱਚ[ਖਾਨਾ].qs;
        ਲਈ (ਪੂਰਨ s = 0; s < 2; s++) {
            ਅਚਲ ਪੂਰਨ ਅਧਾਰ   = s * 16;
            uint8_t * reg_lo = ਬਾਹਰਲਾ.qs + (s * 8 + acc)     * 32;
            uint8_t * reg_hi = ਬਾਹਰਲਾ.qs + (s * 8 + acc + 4) * 32;
            ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
                reg_lo[(2 * cgrp)     * 4 + i] = (uint8_t) q[ਅਧਾਰ + 0  + i];
                reg_lo[(2 * cgrp + 1) * 4 + i] = (uint8_t) q[ਅਧਾਰ + 4  + i];
                reg_hi[(2 * cgrp)     * 4 + i] = (uint8_t) q[ਅਧਾਰ + 8  + i];
                reg_hi[(2 * cgrp + 1) * 4 + i] = (uint8_t) q[ਅਧਾਰ + 12 + i];
            }
        }
    }
    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਪੂਰਨ repack_q8_0_ਮੰਜ਼ਿਲ_q8_0_16_bl_ref(ggml_tensor *              t,
                                         ਪੂਰਨ                        interleave_ਖੰਡ,
                                         ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                         size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_ਖੰਡ == 16);

    constexpr ਪੂਰਨ nrows_interleaved = 16;

    ਖੰਡ_q8_0x16 *    dst = (ਖੰਡ_q8_0x16 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q8_0 * src = (ਅਚਲ ਖੰਡ_q8_0 *) ਅੰਕੜੇ;
    ਖੰਡ_q8_0         dst_tmp[16];
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q8_0));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK8_0 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q8_0x16(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q2_k_ਮੰਜ਼ਿਲ_q2_k_32_bl(ggml_tensor *              t,
                                     ਪੂਰਨ                        interleave_ਖੰਡ,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                     size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_ਖੰਡ == 32);
    GGML_ASSERT(QK_K == 256);

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    ਅਚਲ ਖੰਡ_q2_K * src = (ਅਚਲ ਖੰਡ_q2_K *) ਅੰਕੜੇ;

    ਸਵੈਚਾਲਿਤ * dst = (spacemit_kernels::nrow_ਖੰਡ_q2_k<32> *) t->ਅੰਕੜੇ;

    ਪੂਰਨ nrow    = ggml_nrows(t);
    ਪੂਰਨ nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q2_K));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        ਵਾਪਸ -1;
    }

    uint8_t qs_aux[256] = { 0 };
    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                ਅਚਲ ਖੰਡ_q2_K * src_ਖੰਡ = &src[(b + i) * nblocks + x];

                // scale for [16, N]
                ਲਈ (ਪੂਰਨ j = 0; j < 16; j++) {
                    ਸਵੈਚਾਲਿਤ zp_aux = (dst->scales[j * nrows_interleaved + i]) & 0xF0;

                    dst->scales[j * nrows_interleaved + i] = (src_ਖੰਡ->scales[j] & 0x0F) | zp_aux;
                }

                // zp for [N, 16]
                ਲਈ (ਪੂਰਨ j = 0; j < 16; j++) {
                    ਸਵੈਚਾਲਿਤ scale_aux = (dst->scales[16 * i + j]) & 0x0F;

                    dst->scales[16 * i + j] = (src_ਖੰਡ->scales[j] & 0xF0) | scale_aux;
                }

                ਲਈ (ਪੂਰਨ k = 0; k < 4; k++) {
                    ਲਈ (ਪੂਰਨ j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j] = (src_ਖੰਡ->qs[j] >> (2 * k)) & 0x03;
                    }
                }

                ਲਈ (ਪੂਰਨ k = 0; k < 4; k++) {
                    ਲਈ (ਪੂਰਨ j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j + 128] = (src_ਖੰਡ->qs[j + 32] >> (2 * k)) & 0x03;
                    }
                }

                // from nrows_interleaved * [2 * 32byte]
                // to 4 * [nrows_interleaved * 16byte]
                ਲਈ (ਪੂਰਨ k = 0; k < 4; k++) {
                    ਲਈ (ਪੂਰਨ j = 0; j < 16; j++) {
                        uint8_t qs0  = qs_aux[j + k * 64];
                        uint8_t qs16 = qs_aux[j + 16 + k * 64];
                        uint8_t qs32 = qs_aux[j + 32 + k * 64];
                        uint8_t qs48 = qs_aux[j + 48 + k * 64];

                        dst->qs[(k * nrows_interleaved + i) * 16 + j] =
                            (qs0 & 0x03) | ((qs16 & 0x03) << 2) | ((qs32 & 0x03) << 4) | ((qs48 & 0x03) << 6);
                    }
                }

                dst->scales16[i] = src_ਖੰਡ->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
                dst->zeros16[i]  = src_ਖੰਡ->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
            }
            dst++;
        }
    }

    ਵਾਪਸ 0;
}

ਸਥਿਰ ਪੂਰਨ repack_q3_k_ਮੰਜ਼ਿਲ_q3_k_32_bl(ggml_tensor *              t,
                                     ਪੂਰਨ                        interleave_ਖੰਡ,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                     size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q3_K);
    GGML_ASSERT(interleave_ਖੰਡ == 32);
    GGML_ASSERT(QK_K == 256);

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    ਅਚਲ uint32_t kmask1 = 0x03030303;
    ਅਚਲ uint32_t kmask2 = 0x0f0f0f0f;

    ਅਚਲ ਖੰਡ_q3_K * src = (ਅਚਲ ਖੰਡ_q3_K *) ਅੰਕੜੇ;

    ਸਵੈਚਾਲਿਤ * dst = (spacemit_kernels::nrow_ਖੰਡ_q3_k<32> *) t->ਅੰਕੜੇ;

    ਪੂਰਨ nrow    = ggml_nrows(t);
    ਪੂਰਨ nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q3_K));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        ਵਾਪਸ -1;
    }

    uint32_t b_scale_aux[4] = { 0 };
    uint8_t  qs_aux[256]    = { 0 };

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                ਅਚਲ ਖੰਡ_q3_K * src_ਖੰਡ = &src[(b + i) * nblocks + x];

                uint32_t * auxs  = b_scale_aux;
                int8_t *   scale = (int8_t *) auxs;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(auxs, src_ਖੰਡ->scales, 12);

                uint32_t tmp = auxs[2];
                auxs[2]      = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                auxs[3]      = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                auxs[0]      = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                auxs[1]      = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

                ਲਈ (ਪੂਰਨ j = 0; j < 16; j++) {
                    dst->scales[j * nrows_interleaved + i] = scale[j] - 32;
                }

                ਲਈ (ਪੂਰਨ k = 0; k < 4; k++) {
                    ਲਈ (ਪੂਰਨ j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j] = (src_ਖੰਡ->qs[j] >> (2 * k)) & 0x03;
                    }
                }

                ਲਈ (ਪੂਰਨ k = 0; k < 4; k++) {
                    ਲਈ (ਪੂਰਨ j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j + 128] = (src_ਖੰਡ->qs[j + 32] >> (2 * k)) & 0x03;
                    }
                }

                // from nrows_interleaved * [2 * 32byte]
                // to 4 * [nrows_interleaved * 16byte]
                ਲਈ (ਪੂਰਨ k = 0; k < 4; k++) {
                    ਲਈ (ਪੂਰਨ j = 0; j < 16; j++) {
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
                uint16_t * dst_ਨਕਾਬ = ((uint16_t *) dst->hmask) + i;
                ਲਈ (ਪੂਰਨ j = 0; j < 16; j++, dst_ਨਕਾਬ += nrows_interleaved) {
                    uint8_t   b_ਖਿਸਕਾਓ    = j / 2;
                    uint8_t * b_ਨਕਾਬ_ਖਾਨਾ = (uint8_t *) (src_ਖੰਡ->hmask + (j % 2) * 16);
                    // b0 - b15
                    uint16_t  msk_ਬਾਹਰਲਾ_0  = 0;

                    ਲਈ (ਪੂਰਨ k = 0; k < 8; k++) {
                        msk_ਬਾਹਰਲਾ_0 |= (uint16_t) ((b_ਨਕਾਬ_ਖਾਨਾ[k] >> b_ਖਿਸਕਾਓ) & 0x01) << k;
                    }
                    ਲਈ (ਪੂਰਨ k = 8; k < 16; k++) {
                        msk_ਬਾਹਰਲਾ_0 |= (uint16_t) ((b_ਨਕਾਬ_ਖਾਨਾ[k] >> b_ਖਿਸਕਾਓ) & 0x01) << k;
                    }

                    dst_ਨਕਾਬ[0] = msk_ਬਾਹਰਲਾ_0;
                }

                dst->scales16[i] = src_ਖੰਡ->d;
            }

            dst++;
        }
    }

    ਵਾਪਸ 0;
}

ਸਥਿਰ ਪੂਰਨ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_32_bl_ref(ggml_tensor *              t,
                                         ਪੂਰਨ                        interleave_ਖੰਡ,
                                         ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                         size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ਖੰਡ == 32);  //ਅਣਵਰਤਿਆ

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    ਖੰਡ_q4_0x32 *    dst = (ਖੰਡ_q4_0x32 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_0 * src = (ਅਚਲ ਖੰਡ_q4_0 *) ਅੰਕੜੇ;
    ਖੰਡ_q4_0         dst_tmp[32];
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q4_0));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q4_0x32(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_256_32_bl_ref(ggml_tensor *              t,
                                             ਪੂਰਨ                        interleave_ਖੰਡ,
                                             ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                             size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ਖੰਡ == 32);  //ਅਣਵਰਤਿਆ

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    ਖੰਡ_q4_0x32x256 * dst = (ਖੰਡ_q4_0x32x256 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_0 *  src = (ਅਚਲ ਖੰਡ_q4_0 *) ਅੰਕੜੇ;
    ਖੰਡ_q4_0          dst_tmp[32];
    ਪੂਰਨ                 nrow    = ggml_nrows(t);
    ਪੂਰਨ                 nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q4_0));
    GGML_ASSERT(nblocks % 8 == 0);  // for 256-block interleaving
    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x += 8) {
            ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
                ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                    dst_tmp[i] = src[x + j + i * nblocks];
                }
                dst->ਖੰਡਾਂ[j] = make_ਖੰਡ_q4_0x32(dst_tmp, interleave_ਖੰਡ);
            }
            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q4_0_ਮੰਜ਼ਿਲ_q4_1_256_32_bl_ref(ggml_tensor *              t,
                                             ਪੂਰਨ                        interleave_ਖੰਡ,
                                             ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                             size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_ਖੰਡ == 32);  //ਅਣਵਰਤਿਆ

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    ਖੰਡ_q4_1x32x256 * dst = (ਖੰਡ_q4_1x32x256 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_1 *  src = (ਅਚਲ ਖੰਡ_q4_1 *) ਅੰਕੜੇ;
    ਖੰਡ_q4_1          dst_tmp[32];
    ਪੂਰਨ                 nrow    = ggml_nrows(t);
    ਪੂਰਨ                 nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q4_1));
    GGML_ASSERT(nblocks % 8 == 0);  // for 256-block interleaving
    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x += 8) {
            ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
                ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                    dst_tmp[i] = src[x + j + i * nblocks];
                }

                ਖੰਡ_q4_0x32 * dst_ਖੰਡ = &dst->ਖੰਡਾਂ[j];
                uint8_t *       dst_zp    = dst->zps + j * nrows_interleaved;

                ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                    ਦਸ਼ਮਲਵ d   = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    ਦਸ਼ਮਲਵ m   = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
                    ਦਸ਼ਮਲਵ mid = -std::nearbyintf(m / d);
                    mid       = std::min(15.0f, std::max(0.0f, mid));

                    dst_ਖੰਡ->d[i] = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(d);
                    dst_zp[i]       = static_cast<uint8_t>(mid);
                }

                ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                    ਲਈ (ਪੂਰਨ k = 0; k < QK4_1 / 4; k++) {
                        dst_ਖੰਡ->qs[i * QK4_1 / 2 + k] =
                            (dst_tmp[i].qs[k * 2] & 0x0F) | ((dst_tmp[i].qs[k * 2 + 1] & 0x0F) << 4);
                    }
                }

                ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                    ਲਈ (ਪੂਰਨ k = 0; k < QK4_1 / 4; k++) {
                        dst_ਖੰਡ->qs[i * QK4_1 / 2 + QK4_1 / 4 + k] =
                            ((dst_tmp[i].qs[k * 2] & 0xF0) >> 4) | (dst_tmp[i].qs[k * 2 + 1] & 0xF0);
                    }
                }
            }
            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

// RVV optimized version of repack_q4_0_to_q4_0_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes nibble repack.
ਸਥਿਰ ਪੂਰਨ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_32_bl(ggml_tensor *              t,
                                     ਪੂਰਨ                        interleave_ਖੰਡ,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                     size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ਖੰਡ == 32);

    constexpr ਪੂਰਨ nrows_interleaved = 32;
    constexpr ਪੂਰਨ qs_ਬਾਈਟਾਂ          = QK4_0 / 2;  // 16

    ਖੰਡ_q4_0x32 *    dst     = (ਖੰਡ_q4_0x32 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_0 * src     = (ਅਚਲ ਖੰਡ_q4_0 *) ਅੰਕੜੇ;
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q4_0));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ptrdiff_t ਪੰਕਤੀ_stride = (ptrdiff_t) nblocks * ਆਕਾਰ(ਖੰਡ_q4_0);

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਅਚਲ ਖੰਡ_q4_0 * ਖਾਨਾ_src = src + x;

            // --- 1) Gather 32 scale values (ggml_half d) with stride load ---
            // d is at offset 0 of each block_q4_0, stride between rows = row_stride
            {
                ਅਚਲ uint8_t * d_ਅਧਾਰ    = (ਅਚਲ uint8_t *) &ਖਾਨਾ_src->d;
                ggml_half *     d_dst     = dst->d;
                size_t          ਬਚਿਆ = 32;
                size_t          ਵਿੱਥ    = 0;
                ਜਦੋਂ (ਬਚਿਆ > 0) {
                    size_t      vl = __riscv_vsetvl_e16m1(ਬਚਿਆ);
                    vuint16m1_t vd =
                        __riscv_vlse16_v_u16m1((ਅਚਲ uint16_t *) (d_ਅਧਾਰ + ਵਿੱਥ * ਪੰਕਤੀ_stride), ਪੰਕਤੀ_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + ਵਿੱਥ), vd, vl);
                    ਵਿੱਥ += vl;
                    ਬਚਿਆ -= vl;
                }
            }

            // --- 2) Nibble repack qs for each of the 32 rows ---
            // For each row i:
            //   src qs[16]: [b0|b16] [b1|b17] ... [b15|b31]  (lo nibble = b_j, hi nibble = b_{j+16})
            //   dst qs low  8B: (qs[2j] & 0x0F) | ((qs[2j+1] & 0x0F) << 4)  for j=0..7
            //   dst qs high 8B: ((qs[2j] >> 4))  | (qs[2j+1] & 0xF0)         for j=0..7
            {
                ਅਚਲ size_t vl8 = __riscv_vsetvl_e8m1(8);
                ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
                    ਅਚਲ uint8_t * sq = ਖਾਨਾ_src[i * nblocks].qs;
                    uint8_t *       dq = dst->qs + i * qs_ਬਾਈਟਾਂ;

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
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q4_1_ਮੰਜ਼ਿਲ_q4_1_32_bl_ref(ggml_tensor *              t,
                                         ਪੂਰਨ                        interleave_ਖੰਡ,
                                         ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                         size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_ਖੰਡ == 32);  //ਅਣਵਰਤਿਆ

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    ਖੰਡ_q4_1x32 *    dst = (ਖੰਡ_q4_1x32 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_1 * src = (ਅਚਲ ਖੰਡ_q4_1 *) ਅੰਕੜੇ;
    ਖੰਡ_q4_1         dst_tmp[32];
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q4_1));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ਖੰਡ_q4_1x32(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

// RVV optimized version of repack_q4_1_to_q4_1_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes nibble repack + zp computation.
ਸਥਿਰ ਪੂਰਨ repack_q4_1_ਮੰਜ਼ਿਲ_q4_1_32_bl(ggml_tensor *              t,
                                     ਪੂਰਨ                        interleave_ਖੰਡ,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                     size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_ਖੰਡ == 32);

    constexpr ਪੂਰਨ nrows_interleaved = 32;
    constexpr ਪੂਰਨ qs_ਬਾਈਟਾਂ          = QK4_1 / 2;  // 16

    ਖੰਡ_q4_1x32 *    dst     = (ਖੰਡ_q4_1x32 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_1 * src     = (ਅਚਲ ਖੰਡ_q4_1 *) ਅੰਕੜੇ;
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q4_1));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ptrdiff_t ਪੰਕਤੀ_stride = (ptrdiff_t) nblocks * ਆਕਾਰ(ਖੰਡ_q4_1);

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਅਚਲ ਖੰਡ_q4_1 * ਖਾਨਾ_src = src + x;

            // --- 1) Gather d and m, compute zp = clamp(nearbyint(-m/d), 0, 15) ---
            // block_q4_1 layout: [d(f16), m(f16), qs[16]]
            // d is at byte offset 0, m is at byte offset 2 from each block start
            {
                ਅਚਲ uint8_t * dm_ਅਧਾਰ   = (ਅਚਲ uint8_t *) &ਖਾਨਾ_src->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
                ggml_half *     d_dst     = dst->d;
                uint8_t *       zp_dst    = dst->zp;
                size_t          ਬਚਿਆ = 32;
                size_t          ਵਿੱਥ    = 0;
                ਜਦੋਂ (ਬਚਿਆ > 0) {
                    size_t vl = __riscv_vsetvl_e16m1(ਬਚਿਆ);

                    // stride load d (f16) from each row
                    vuint16m1_t vd_raw =
                        __riscv_vlse16_v_u16m1((ਅਚਲ uint16_t *) (dm_ਅਧਾਰ + ਵਿੱਥ * ਪੰਕਤੀ_stride), ਪੰਕਤੀ_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + ਵਿੱਥ), vd_raw, vl);

                    // stride load m (f16) from each row (offset +2 bytes from d)
                    vuint16m1_t vm_raw =
                        __riscv_vlse16_v_u16m1((ਅਚਲ uint16_t *) (dm_ਅਧਾਰ + 2 + ਵਿੱਥ * ਪੰਕਤੀ_stride), ਪੰਕਤੀ_stride, vl);

                    // convert to f32 for zp computation: zp = nearbyint(-m / d)
                    vfloat16m1_t vd_f16 = __riscv_vreinterpret_v_u16m1_f16m1(vd_raw);
                    vfloat16m1_t vm_f16 = __riscv_vreinterpret_v_u16m1_f16m1(vm_raw);

                    // -m / d in f16 directly (SpaceMIT X60 supports f16 arithmetic)
                    vfloat16m1_t v_neg_m = __riscv_vfneg_v_f16m1(vm_f16, vl);
                    vfloat16m1_t v_ਅਨੁਪਾਤ = __riscv_vfdiv_vv_f16m1(v_neg_m, vd_f16, vl);

                    // Convert to f32 for nearbyint, then clamp
                    vfloat32m2_t v_ਅਨੁਪਾਤ_f32 = __riscv_vfwcvt_f_f_v_f32m2(v_ਅਨੁਪਾਤ, vl);

                    // Use integer rounding: convert f32 -> int (rounds to nearest)
                    vint32m2_t v_zp_i32 = __riscv_vfcvt_x_f_v_i32m2(v_ਅਨੁਪਾਤ_f32, vl);

                    // clamp to [0, 15]
                    v_zp_i32 = __riscv_vmax_vx_i32m2(v_zp_i32, 0, vl);
                    v_zp_i32 = __riscv_vmin_vx_i32m2(v_zp_i32, 15, vl);

                    // narrow i32 -> u8
                    vint16m1_t  v_zp_i16 = __riscv_vncvt_x_x_w_i16m1(v_zp_i32, vl);
                    vint8mf2_t  v_zp_i8  = __riscv_vncvt_x_x_w_i8mf2(v_zp_i16, vl);
                    vuint8mf2_t v_zp_u8  = __riscv_vreinterpret_v_i8mf2_u8mf2(v_zp_i8);
                    __riscv_vse8_v_u8mf2(zp_dst + ਵਿੱਥ, v_zp_u8, vl);

                    ਵਿੱਥ += vl;
                    ਬਚਿਆ -= vl;
                }
            }

            // --- 2) Nibble repack qs for each of the 32 rows ---
            {
                ਅਚਲ size_t vl8 = __riscv_vsetvl_e8m1(8);
                ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
                    ਅਚਲ uint8_t * sq = ਖਾਨਾ_src[i * nblocks].qs;
                    uint8_t *       dq = dst->qs + i * qs_ਬਾਈਟਾਂ;

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
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q4_k_ਮੰਜ਼ਿਲ_q4_1_32_bl(ggml_tensor *              t,
                                     ਪੂਰਨ                        interleave_ਖੰਡ,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                     size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_ਖੰਡ == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    ਖੰਡ_q4_1x32 *    dst = (ਖੰਡ_q4_1x32 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q4_K * src = (ਅਚਲ ਖੰਡ_q4_K *) ਅੰਕੜੇ;
    ਖੰਡ_q4_1         dst_tmp[32];
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK_K;

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
                ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    ਅਚਲ ਦਸ਼ਮਲਵ min =
                        GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);
                    ਅਚਲ ਦਸ਼ਮਲਵ d1 = d * sc;
                    ਅਚਲ ਦਸ਼ਮਲਵ m1 = min * m;

                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(d1);
                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(-m1);
                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    ਅਚਲ uint8_t * q                                  = src[x + i * nblocks].qs + (j / 2) * QK4_1;
                    ਜੇ (j % 2 == 0) {
                        ਲਈ (ਪੂਰਨ ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } ਨਹੀਂਤਾਂ {
                        ਲਈ (ਪੂਰਨ ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }
                }
                *dst++ = make_ਖੰਡ_q4_1x32(dst_tmp, interleave_ਖੰਡ);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q6_k_ਮੰਜ਼ਿਲ_q8_0_32_bl_ref(ggml_tensor *              t,
                                         ਪੂਰਨ                        interleave_ਖੰਡ,
                                         ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                         size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_ਖੰਡ == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    ਖੰਡ_q8_0x32 *    dst = (ਖੰਡ_q8_0x32 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q6_K * src = (ਅਚਲ ਖੰਡ_q6_K *) ਅੰਕੜੇ;
    ਖੰਡ_q8_0         dst_tmp[32];
    int8_t             aux8[QK4_1];
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK_K;

    ਜੇ (t->ne[0] % QK_K != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        int64_t nrow_real = std::min((int64_t) nrow - b, (int64_t) nrows_interleaved);
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ bi = 0; bi < 8; bi++) {
                ਪੂਰਨ i = 0;
                ਲਈ (; i < nrow_real; i++) {
                    ਅਚਲ uint8_t * q4     = src[x + i * nblocks].ql;
                    ਅਚਲ uint8_t * qh     = src[x + i * nblocks].qh;
                    ਅਚਲ int8_t *  scales = src[x + i * nblocks].scales;
                    ਦਸ਼ਮਲਵ           d      = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src[x + i * nblocks].d);

                    q4 += 64 * (bi / 4);
                    qh += 32 * (bi / 4);
                    int8_t * GGML_RESTRICT a = aux8;

                    int8_t bi_idx = bi % 4;

                    ਜੇ (bi_idx == 0) {
                        ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        }
                    } ਨਹੀਂਤਾਂ ਜੇ (bi_idx == 1) {
                        ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        }
                    } ਨਹੀਂਤਾਂ ਜੇ (bi_idx == 2) {
                        ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        }
                    } ਨਹੀਂਤਾਂ ਜੇ (bi_idx == 3) {
                        ਲਈ (ਪੂਰਨ l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        }
                    }
                    a = aux8;

                    ਦਸ਼ਮਲਵ a_max_abs = 0.0f;
                    ਦਸ਼ਮਲਵ scale_0   = scales[bi * 2 + 0] * d;
                    ਦਸ਼ਮਲਵ scale_1   = scales[bi * 2 + 1] * d;
                    ਲਈ (ਪੂਰਨ l = 0; l < 16; ++l) {
                        a_max_abs = std::max(a_max_abs, std::ਨਿਰਪੇਖ(a[l] * scale_0));
                    }

                    ਲਈ (ਪੂਰਨ l = 16; l < 32; ++l) {
                        a_max_abs = std::max(a_max_abs, std::ਨਿਰਪੇਖ(a[l] * scale_1));
                    }

                    ਦਸ਼ਮਲਵ reflect_scale   = a_max_abs / ((1 << 7) - 1);
                    ਦਸ਼ਮਲਵ reflect_scale_0 = scale_0 / reflect_scale;
                    ਦਸ਼ਮਲਵ reflect_scale_1 = scale_1 / reflect_scale;

                    ਲਈ (ਪੂਰਨ l = 0; l < 16; ++l) {
                        ਦਸ਼ਮਲਵ a_ਆਰਜ਼ੀ = std::clamp(std::nearbyintf(a[l] * reflect_scale_0), -128.0f, 127.0f);
                        a[l]         = (int8_t) (a_ਆਰਜ਼ੀ);
                    }

                    ਲਈ (ਪੂਰਨ l = 16; l < 32; ++l) {
                        ਦਸ਼ਮਲਵ a_ਆਰਜ਼ੀ = std::clamp(std::nearbyintf(a[l] * reflect_scale_1), -128.0f, 127.0f);
                        a[l]         = (int8_t) (a_ਆਰਜ਼ੀ);
                    }

                    dst_tmp[i].d = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(reflect_scale);

                    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_tmp[i].qs, a, 32 * ਆਕਾਰ(int8_t));
                }

                ਲਈ (; i < nrows_interleaved; i++) {
                    memset(&dst_tmp[i], 0, ਆਕਾਰ(ਖੰਡ_q8_0));
                }

                *dst++ = make_ਖੰਡ_q8_0x32(dst_tmp, interleave_ਖੰਡ);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

// RVV optimized version of repack_q6_k_to_q8_0_32_bl
// Vectorizes the Q6_K dequant -> requant pipeline using RVV intrinsics.
// For each sub-block (bi), dequant 32 Q6_K values to int6 -> apply two sub-block scales ->
// find max abs -> compute reflect_scale -> requant to int8 -> gather d with stride load.
ਸਥਿਰ ਪੂਰਨ repack_q6_k_ਮੰਜ਼ਿਲ_q8_0_32_bl(ggml_tensor *              t,
                                     ਪੂਰਨ                        interleave_ਖੰਡ,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                     size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_ਖੰਡ == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    ਖੰਡ_q8_0x32 *    dst     = (ਖੰਡ_q8_0x32 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q6_K * src     = (ਅਚਲ ਖੰਡ_q6_K *) ਅੰਕੜੇ;
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK_K;

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ptrdiff_t ਪੰਕਤੀ_stride = (ptrdiff_t) nblocks * ਆਕਾਰ(ਖੰਡ_q6_K);

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ bi = 0; bi < 8; bi++) {
                // --- 1) Gather 32 d values with stride load ---
                // We need to compute reflect_scale per row first, so gather d later.
                // Process each row: dequant Q6_K sub-block -> requant to Q8_0
                ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                    ਅਚਲ ਖੰਡ_q6_K * src_blk = &src[x + i * nblocks];
                    ਅਚਲ uint8_t *    q4      = src_blk->ql + 64 * (bi / 4);
                    ਅਚਲ uint8_t *    qh      = src_blk->qh + 32 * (bi / 4);
                    ਅਚਲ int8_t *     scales  = src_blk->scales;
                    ਦਸ਼ਮਲਵ              d       = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src_blk->d);

                    int8_t bi_idx = bi % 4;

                    // --- Dequant 32 Q6_K values to int6 (range [-32, 31]) using RVV ---
                    // vl = 32 for e8m2 (VLEN=256) or loop for smaller VLEN
                    ਅਚਲ size_t vl16 = __riscv_vsetvl_e8m1(16);

                    vint8m1_t va_lo, va_hi;  // 16 elements each

                    ਜੇ (bi_idx == 0) {
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
                    } ਨਹੀਂਤਾਂ ਜੇ (bi_idx == 1) {
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
                    } ਨਹੀਂਤਾਂ ਜੇ (bi_idx == 2) {
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
                    } ਨਹੀਂਤਾਂ {  // bi_idx == 3
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
                    ਦਸ਼ਮਲਵ scale_0 = scales[bi * 2 + 0] * d;
                    ਦਸ਼ਮਲਵ scale_1 = scales[bi * 2 + 1] * d;

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
                    ਦਸ਼ਮਲਵ        a_max_abs = __riscv_vfmv_f_s_f32m1_f32(vmax_red);

                    ਦਸ਼ਮਲਵ reflect_scale   = a_max_abs / 127.0f;
                    ਦਸ਼ਮਲਵ reflect_scale_0 = scale_0 / reflect_scale;
                    ਦਸ਼ਮਲਵ reflect_scale_1 = scale_1 / reflect_scale;

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
                    dst->d[i]   = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(reflect_scale);
                    int8_t * dq = (int8_t *) dst->qs + i * QK8_0;
                    __riscv_vse8_v_i8m1(dq, vi8_lo, vl16);
                    __riscv_vse8_v_i8m1(dq + 16, vi8_hi, vl16);
                }
                dst++;
            }
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਪੂਰਨ repack_q8_0_ਮੰਜ਼ਿਲ_q8_0_32_bl_ref(ggml_tensor *              t,
                                         ਪੂਰਨ                        interleave_ਖੰਡ,
                                         ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                         size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_ਖੰਡ == 32);  //ਅਣਵਰਤਿਆ

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    ਖੰਡ_q8_0x32 *    dst = (ਖੰਡ_q8_0x32 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q8_0 * src = (ਅਚਲ ਖੰਡ_q8_0 *) ਅੰਕੜੇ;
    ਖੰਡ_q8_0         dst_tmp[32];
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q8_0));

    ਜੇ (t->ne[0] % QK8_0 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        int64_t nrows_real = std::min((int64_t) nrow - b, (int64_t) nrows_interleaved);
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਪੂਰਨ i = 0;
            ਲਈ (; i < nrows_real; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            ਲਈ (; i < nrows_interleaved; i++) {
                memset(&dst_tmp[i], 0, ਆਕਾਰ(ਖੰਡ_q8_0));
            }
            *dst++ = make_ਖੰਡ_q8_0x32(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

// RVV optimized version of repack_q8_0_to_q8_0_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes scale gather + qs copy.
ਸਥਿਰ ਪੂਰਨ repack_q8_0_ਮੰਜ਼ਿਲ_q8_0_32_bl(ggml_tensor *              t,
                                     ਪੂਰਨ                        interleave_ਖੰਡ,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                     size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_ਖੰਡ == 32);

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    ਖੰਡ_q8_0x32 *    dst     = (ਖੰਡ_q8_0x32 *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q8_0 * src     = (ਅਚਲ ਖੰਡ_q8_0 *) ਅੰਕੜੇ;
    ਪੂਰਨ                nrow    = ggml_nrows(t);
    ਪੂਰਨ                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q8_0));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK8_0 != 0) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ptrdiff_t ਪੰਕਤੀ_stride = (ptrdiff_t) nblocks * ਆਕਾਰ(ਖੰਡ_q8_0);

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਅਚਲ ਖੰਡ_q8_0 * ਖਾਨਾ_src = src + x;

            // --- 1) Gather 32 scale values (ggml_half d) with stride load ---
            {
                ਅਚਲ uint8_t * d_ਅਧਾਰ    = (ਅਚਲ uint8_t *) &ਖਾਨਾ_src->d;
                ggml_half *     d_dst     = dst->d;
                size_t          ਬਚਿਆ = 32;
                size_t          ਵਿੱਥ    = 0;
                ਜਦੋਂ (ਬਚਿਆ > 0) {
                    size_t      vl = __riscv_vsetvl_e16m1(ਬਚਿਆ);
                    vuint16m1_t vd =
                        __riscv_vlse16_v_u16m1((ਅਚਲ uint16_t *) (d_ਅਧਾਰ + ਵਿੱਥ * ਪੰਕਤੀ_stride), ਪੰਕਤੀ_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + ਵਿੱਥ), vd, vl);
                    ਵਿੱਥ += vl;
                    ਬਚਿਆ -= vl;
                }
            }

            // --- 2) Copy qs for each of the 32 rows (32 bytes per row) ---
            {
                ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
                    ਅਚਲ int8_t * sq = ਖਾਨਾ_src[i * nblocks].qs;
                    int8_t *       dq = (int8_t *) dst->qs + i * QK8_0;

                    size_t len = QK8_0;
                    size_t idx = 0;
                    ਜਦੋਂ (len > 0) {
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
    ਵਾਪਸ 0;

    GGML_ਅਣਵਰਤਿਆ(ਅੰਕੜੇ_ਮਾਪ);
}

ਸਥਿਰ ਖਾਲੀ convert_mxfp4_ਮੰਜ਼ਿਲ_5bit(ਅਚਲ ਖੰਡ_mxfp4 & src, spacemit_kernels::nrow_ਖੰਡ_mxfp4<1> & dst) {
    dst.e[0] = src.e;

    // Decode all 32 mxfp4 values to signed integers via kvalues_mxfp4
    int8_t vals[32];
    ਲਈ (ਪੂਰਨ j = 0; j < QK_MXFP4 / 2; j++) {
        vals[j]                = kvalues_mxfp4[src.qs[j] & 0xF];
        vals[j + QK_MXFP4 / 2] = kvalues_mxfp4[src.qs[j] >> 4];
    }

    // vals [b0, b1, b2, b3, ..., b30, b31]
    // Pack abs into qs with reorder: [b0,b1]..[b14,b15]..[b30,b31]
    ਲਈ (ਪੂਰਨ j = 0; j < QK_MXFP4 / 2; j++) {
        uint8_t lo0 = static_cast<uint8_t>(std::ਨਿਰਪੇਖ(vals[j * 2]));
        uint8_t lo1 = static_cast<uint8_t>(std::ਨਿਰਪੇਖ(vals[j * 2 + 1]));
        dst.qs[j]   = (lo0 & 0x0F) | ((lo1 & 0x0F) << 4);
    }

    // Pack sign bits into qh[4] (32 bits total, 1 bit per weight)
    // reorder: [0,1,2,...,15,16,17,...,31] after the qs reorder above
    uint32_t sign_ਬਿੱਟ = 0;
    ਲਈ (ਪੂਰਨ j = 0; j < 32; j++) {
        ਜੇ (vals[j] < 0) {
            sign_ਬਿੱਟ |= (1u << j);
        }
    }
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst.qh, &sign_ਬਿੱਟ, 4);
}

ਸਥਿਰ spacemit_kernels::nrow_ਖੰਡ_mxfp4<32> make_ਖੰਡ_mxfp4x32(spacemit_kernels::nrow_ਖੰਡ_mxfp4<1> * ਵਿੱਚ,
                                                                  ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    spacemit_kernels::nrow_ਖੰਡ_mxfp4<32> ਬਾਹਰਲਾ;
    GGML_ASSERT(QK_MXFP4 / blck_ਮਾਪ_interleave == 1);
    GGML_ਅਣਵਰਤਿਆ(blck_ਮਾਪ_interleave);

    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        ਬਾਹਰਲਾ.e[i] = ਵਿੱਚ[i].e[0];
    }

    // qs: copy per-row 16 bytes
    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਬਾਹਰਲਾ.qs + i * 16, ਵਿੱਚ[i].qs, 16);
    }

    // qh: copy per-row 4 bytes
    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(ਬਾਹਰਲਾ.qh + i * 4, ਵਿੱਚ[i].qh, 4);
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਪੂਰਨ repack_mxfp4_ਮੰਜ਼ਿਲ_mxfp4_32_bl(ggml_tensor *              t,
                                       ਪੂਰਨ                        interleave_ਖੰਡ,
                                       ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                       size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_ਖੰਡ == 32);

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    spacemit_kernels::nrow_ਖੰਡ_mxfp4<32> * dst = (spacemit_kernels::nrow_ਖੰਡ_mxfp4<32> *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_mxfp4 *                      src = (ਅਚਲ ਖੰਡ_mxfp4 *) ਅੰਕੜੇ;
    spacemit_kernels::nrow_ਖੰਡ_mxfp4<1>    dst_tmp[32];
    ਪੂਰਨ                                      nrow    = ggml_nrows(t);
    ਪੂਰਨ                                      nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_mxfp4));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_MXFP4 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                convert_mxfp4_ਮੰਜ਼ਿਲ_5bit(src[x + i * nblocks], dst_tmp[i]);
            }
            *dst++ = make_ਖੰਡ_mxfp4x32(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;
}

ਸਥਿਰ spacemit_kernels::nrow_ਖੰਡ_q5_1<32> make_ਖੰਡ_q5_1x32(spacemit_kernels::nrow_ਖੰਡ_q5_1<1> * ਵਿੱਚ,
                                                                ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    spacemit_kernels::nrow_ਖੰਡ_q5_1<32> ਬਾਹਰਲਾ;
    GGML_ASSERT(QK5_1 / blck_ਮਾਪ_interleave == 1);
    GGML_ਅਣਵਰਤਿਆ(blck_ਮਾਪ_interleave);

    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        ਬਾਹਰਲਾ.scales16[i] = ਵਿੱਚ[i].scales16[0];
        ਬਾਹਰਲਾ.zp[i]       = ਵਿੱਚ[i].zp[0];
    }

    // qs: low 4 bits, reorder from [b0,b16],[b1,b17]... to [b0,b1]...[b14,b15] and [b16,b17]...[b30,b31]
    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        // low half [0..15]
        ਲਈ (ਪੂਰਨ j = 0; j < QK5_1 / 4; j++) {
            ਬਾਹਰਲਾ.qs[i * QK5_1 / 2 + j] = (ਵਿੱਚ[i].qs[j * 2] & 0x0F) | ((ਵਿੱਚ[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
        // high half [16..31]
        ਲਈ (ਪੂਰਨ j = 0; j < QK5_1 / 4; j++) {
            ਬਾਹਰਲਾ.qs[i * QK5_1 / 2 + QK5_1 / 4 + j] = ((ਵਿੱਚ[i].qs[j * 2] & 0xF0) >> 4) | (ਵਿੱਚ[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    // qh: 5th bit, copy directly
    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        ਲਈ (ਪੂਰਨ j = 0; j < 4; j++) {
            ਬਾਹਰਲਾ.qh[i * 4 + j] = ਵਿੱਚ[i].qh[j];
        }
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ spacemit_kernels::nrow_ਖੰਡ_q5_0<32> make_ਖੰਡ_q5_0x32(spacemit_kernels::nrow_ਖੰਡ_q5_0<1> * ਵਿੱਚ,
                                                                ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ blck_ਮਾਪ_interleave) {
    spacemit_kernels::nrow_ਖੰਡ_q5_0<32> ਬਾਹਰਲਾ;
    GGML_ASSERT(QK5_0 / blck_ਮਾਪ_interleave == 1);
    GGML_ਅਣਵਰਤਿਆ(blck_ਮਾਪ_interleave);

    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        ਬਾਹਰਲਾ.scales16[i] = ਵਿੱਚ[i].scales16[0];
    }

    // qs: low 4 bits, reorder from [b0,b16],[b1,b17]... to [b0,b1]...[b14,b15] and [b16,b17]...[b30,b31]
    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        // low half [0..15]
        ਲਈ (ਪੂਰਨ j = 0; j < QK5_0 / 4; j++) {
            ਬਾਹਰਲਾ.qs[i * QK5_0 / 2 + j] = (ਵਿੱਚ[i].qs[j * 2] & 0x0F) | ((ਵਿੱਚ[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
        // high half [16..31]
        ਲਈ (ਪੂਰਨ j = 0; j < QK5_0 / 4; j++) {
            ਬਾਹਰਲਾ.qs[i * QK5_0 / 2 + QK5_0 / 4 + j] = ((ਵਿੱਚ[i].qs[j * 2] & 0xF0) >> 4) | (ਵਿੱਚ[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    // qh: 5th bit, copy directly
    ਲਈ (ਪੂਰਨ i = 0; i < 32; i++) {
        ਲਈ (ਪੂਰਨ j = 0; j < 4; j++) {
            ਬਾਹਰਲਾ.qh[i * 4 + j] = ਵਿੱਚ[i].qh[j];
        }
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

ਸਥਿਰ ਪੂਰਨ repack_q5_0_ਮੰਜ਼ਿਲ_q5_0_32_bl(ggml_tensor *              t,
                                     ਪੂਰਨ                        interleave_ਖੰਡ,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                     size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(interleave_ਖੰਡ == 32);  //ਅਣਵਰਤਿਆ

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    spacemit_kernels::nrow_ਖੰਡ_q5_0<32> * dst = (spacemit_kernels::nrow_ਖੰਡ_q5_0<32> *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q5_0 *                      src = (ਅਚਲ ਖੰਡ_q5_0 *) ਅੰਕੜੇ;
    spacemit_kernels::nrow_ਖੰਡ_q5_0<1>    dst_tmp[32];
    ਪੂਰਨ                                     nrow    = ggml_nrows(t);
    ਪੂਰਨ                                     nblocks = t->ne[0] / QK5_0;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q5_0));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK5_0 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                ਅਚਲ ਖੰਡ_q5_0 & s = src[x + i * nblocks];

                dst_tmp[i].scales16[0] = s.d;
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_tmp[i].qs, s.qs, ਆਕਾਰ(dst_tmp[i].qs));
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_tmp[i].qh, s.qh, ਆਕਾਰ(dst_tmp[i].qh));
            }
            *dst++ = make_ਖੰਡ_q5_0x32(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;
}

ਸਥਿਰ ਪੂਰਨ repack_q5_1_ਮੰਜ਼ਿਲ_q5_1_32_bl(ggml_tensor *              t,
                                     ਪੂਰਨ                        interleave_ਖੰਡ,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                     size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(interleave_ਖੰਡ == 32);  //ਅਣਵਰਤਿਆ

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    spacemit_kernels::nrow_ਖੰਡ_q5_1<32> * dst = (spacemit_kernels::nrow_ਖੰਡ_q5_1<32> *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q5_1 *                      src = (ਅਚਲ ਖੰਡ_q5_1 *) ਅੰਕੜੇ;
    spacemit_kernels::nrow_ਖੰਡ_q5_1<1>    dst_tmp[32];
    ਪੂਰਨ                                     nrow    = ggml_nrows(t);
    ਪੂਰਨ                                     nblocks = t->ne[0] / QK5_1;

    GGML_ASSERT(ਅੰਕੜੇ_ਮਾਪ == nrow * nblocks * ਆਕਾਰ(ਖੰਡ_q5_1));

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK5_1 != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                ਅਚਲ ਖੰਡ_q5_1 & s = src[x + i * nblocks];

                ਦਸ਼ਮਲਵ d = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(s.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                ਦਸ਼ਮਲਵ m = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(s.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);

                ਜੇ (d == 0.0f) {
                    dst_tmp[i].scales16[0] = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(std::fabs(m));
                    dst_tmp[i].zp[0]       = m < 0.0f ? 1 : 0;
                    memset(dst_tmp[i].qh, 0, ਆਕਾਰ(dst_tmp[i].qh));
                    memset(dst_tmp[i].qs, m > 0.0f ? 0x11 : 0x00, ਆਕਾਰ(dst_tmp[i].qs));
                    ਜਾਰੀ;
                }

                ਦਸ਼ਮਲਵ mid = std::nearbyintf(-m / d);
                mid       = std::min(31.0f, std::max(0.0f, mid));

                dst_tmp[i].scales16[0] = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(d);
                dst_tmp[i].zp[0]       = static_cast<uint8_t>(mid);

                // qs: copy low 4 bits directly (same nibble packing)
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_tmp[i].qs, s.qs, QK5_1 / 2);

                // qh: copy 5th bit directly
                ਸਮ੍ਰਿਤੀ_ਨਕਲ(dst_tmp[i].qh, s.qh, 4);
            }
            *dst++ = make_ਖੰਡ_q5_1x32(dst_tmp, interleave_ਖੰਡ);
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;
}

ਸਥਿਰ ਪੂਰਨ repack_q5_k_ਮੰਜ਼ਿਲ_q5_1_32_bl(ggml_tensor *              t,
                                     ਪੂਰਨ                        interleave_ਖੰਡ,
                                     ਅਚਲ ਖਾਲੀ * GGML_RESTRICT ਅੰਕੜੇ,
                                     size_t                     ਅੰਕੜੇ_ਮਾਪ) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_ਖੰਡ == 32);
    GGML_ASSERT(QK_K / QK5_1 == 8);

    constexpr ਪੂਰਨ nrows_interleaved = 32;

    spacemit_kernels::nrow_ਖੰਡ_q5_1<32> * dst = (spacemit_kernels::nrow_ਖੰਡ_q5_1<32> *) t->ਅੰਕੜੇ;
    ਅਚਲ ਖੰਡ_q5_K *                      src = (ਅਚਲ ਖੰਡ_q5_K *) ਅੰਕੜੇ;
    spacemit_kernels::nrow_ਖੰਡ_q5_1<1>    dst_tmp[32];
    ਪੂਰਨ                                     nrow    = ggml_nrows(t);
    ਪੂਰਨ                                     nblocks = t->ne[0] / QK_K;

    ਜੇ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        ਵਾਪਸ -1;
    }

    ਲਈ (ਪੂਰਨ b = 0; b < nrow; b += nrows_interleaved) {
        ਲਈ (int64_t x = 0; x < nblocks; x++) {
            ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
                ਲਈ (ਪੂਰਨ i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    ਅਚਲ ਦਸ਼ਮਲਵ d = GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    ਅਚਲ ਦਸ਼ਮਲਵ min =
                        GGML_FP16_ਮੰਜ਼ਿਲ_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);

                    ਦਸ਼ਮਲਵ d1 = d * sc;
                    ਦਸ਼ਮਲਵ m1 = min * m;

                    ਦਸ਼ਮਲਵ mid              = std::nearbyintf(m1 / d1);
                    mid                    = std::min(31.0f, std::max(0.0f, mid));
                    dst_tmp[i].scales16[0] = GGML_FP32_ਮੰਜ਼ਿਲ_FP16(d1);
                    dst_tmp[i].zp[0]       = static_cast<uint8_t>(mid);

                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    ਅਚਲ uint8_t * q = src[x + i * nblocks].qs + (j / 2) * QK5_1;
                    ਜੇ (j % 2 == 0) {
                        ਲਈ (ਪੂਰਨ ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } ਨਹੀਂਤਾਂ {
                        ਲਈ (ਪੂਰਨ ii = 0; ii < 16; ii++) {
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
                    ਅਚਲ uint8_t * src_qh = src[x + i * nblocks].qh;
                    ਲਈ (ਪੂਰਨ bi = 0; bi < 4; bi++) {
                        uint8_t qh_byte = 0;
                        ਲਈ (ਪੂਰਨ k = 0; k < 8; k++) {
                            ਪੂਰਨ src_idx = bi * 8 + k;
                            qh_byte |= ((src_qh[src_idx] >> j) & 1) << k;
                        }
                        dst_tmp[i].qh[bi] = qh_byte;
                    }
                }
                *dst++ = make_ਖੰਡ_q5_1x32(dst_tmp, interleave_ਖੰਡ);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    ਵਾਪਸ 0;
}

ਨਾਮਾਕਾਸ਼ ggml::cpu::riscv64_spacemit {

ਸਾਂਚਾ <ਕਿਸਮ_ਨਾਂ BLOC_TYPE, int64_t INTER_ਮਾਪ, int64_t NB_ਖਾਨੇ_ਗਿਣਤੀ> ਪੂਰਨ repack(ggml_tensor *, ਅਚਲ ਖਾਲੀ *, size_t);

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_0, 32, 16>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_16_bl(t, 16, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_1, 32, 16>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q4_1_ਮੰਜ਼ਿਲ_q4_1_16_bl(t, 16, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_K, 32, 16>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q4_k_ਮੰਜ਼ਿਲ_q4_1_16_bl(t, 16, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q2_K, 256, 32>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q2_k_ਮੰਜ਼ਿਲ_q2_k_32_bl(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q3_K, 256, 32>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q3_k_ਮੰਜ਼ਿਲ_q3_k_32_bl(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_0, 32, 32>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
#ਜੇ 0
    ਵਾਪਸ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_32_bl_ref(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_32_bl(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_0, 256, 32>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
#ਜੇ 1
    ਵਾਪਸ repack_q4_0_ਮੰਜ਼ਿਲ_q4_0_256_32_bl_ref(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
#ਨਹੀਂਤਾਂ
    //return repack_q4_0_to_q4_0_256_32_bl(t, 32, data, data_size);
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_1, 32, 32>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
#ਜੇ 0
    ਵਾਪਸ repack_q4_1_ਮੰਜ਼ਿਲ_q4_1_32_bl_ref(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ repack_q4_1_ਮੰਜ਼ਿਲ_q4_1_32_bl(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_1, 256, 32>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
#ਜੇ 1
    ਵਾਪਸ repack_q4_0_ਮੰਜ਼ਿਲ_q4_1_256_32_bl_ref(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ repack_q4_1_ਮੰਜ਼ਿਲ_q4_1_256_32_bl(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q4_K, 32, 32>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q4_k_ਮੰਜ਼ਿਲ_q4_1_32_bl(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q6_K, 32, 32>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
#ਜੇ 1
    ਵਾਪਸ repack_q6_k_ਮੰਜ਼ਿਲ_q8_0_32_bl_ref(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ repack_q6_k_ਮੰਜ਼ਿਲ_q8_0_32_bl(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q8_0, 32, 16>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q8_0_ਮੰਜ਼ਿਲ_q8_0_16_bl_ref(t, 16, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q8_0, 32, 32>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
#ਜੇ 1
    ਵਾਪਸ repack_q8_0_ਮੰਜ਼ਿਲ_q8_0_32_bl_ref(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
#ਨਹੀਂਤਾਂ
    ਵਾਪਸ repack_q8_0_ਮੰਜ਼ਿਲ_q8_0_32_bl(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
#ਜੇ_ਅੰਤ
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_mxfp4, 32, 32>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_mxfp4_ਮੰਜ਼ਿਲ_mxfp4_32_bl(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q5_0, 32, 32>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q5_0_ਮੰਜ਼ਿਲ_q5_0_32_bl(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q5_1, 32, 32>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q5_1_ਮੰਜ਼ਿਲ_q5_1_32_bl(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

ਸਾਂਚਾ <> ਪੂਰਨ repack<ਖੰਡ_q5_K, 32, 32>(ggml_tensor * t, ਅਚਲ ਖਾਲੀ * ਅੰਕੜੇ, size_t ਅੰਕੜੇ_ਮਾਪ) {
    ਵਾਪਸ repack_q5_k_ਮੰਜ਼ਿਲ_q5_1_32_bl(t, 32, ਅੰਕੜੇ, ਅੰਕੜੇ_ਮਾਪ);
}

}  // namespace ggml::cpu::riscv64_spacemit
