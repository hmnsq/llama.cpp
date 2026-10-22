#define GGML_COMMON_IMPL_CPP
#define GGML_COMMON_DECL_CPP

#அந்தர்பாவ "repack.h"

#அந்தர்பாவ "ggml-common.h"
#அந்தர்பாவ "ggml-cpu.h"
#அந்தர்பாவ "ggml-impl.h"
#அந்தர்பாவ "ime_kernels.h"

#அந்தர்பாவ <algorithm>
#அந்தர்பாவ <cassert>
#அந்தர்பாவ <cmath>
#அந்தர்பாவ <cstring>

// clang-format off
#எனில் defined(__riscv)

#எனில் !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#error "riscv v extension or v_intrinsic not enabled"
#இல்லையெனில்
#அந்தர்பாவ <riscv_vector.h>
#endif

#எனில் !defined(__riscv_zfh)
#error "riscv zfh extension not enabled"
#endif

#இல்லையெனில்
#error "riscv not enabled in this build"
#endif

#எனில் defined(__GNUC__)
#pragma GCC diagnostic ignored "-Wcast-qual"
#pragma GCC diagnostic ignored "-Wunused-parameter"
#endif

// clang-format on

template <முழு K> constexpr முழு QK_0() {
    எனில் constexpr (K == 4) {
        திரும்பு QK4_0;
    }
    எனில் constexpr (K == 8) {
        திரும்பு QK8_0;
    }
    திரும்பு -1;
}

template <முழு K, முழு N> கட்டமைப்பு block {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    uint8_t   qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

template <முழு K, முழு N> கட்டமைப்பு block_with_zp {
    ggml_half d[N];                         // deltas for N qK_1 blocks
    uint8_t   zp[N];                        // zero points for N qK_1 blocks
    uint8_t   qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_1 blocks
};

// control size
static_assert(அளவு(block<4, 16>) == 16 * அளவு(ggml_half) + QK4_0 * 8, "wrong block<4,16> size/padding");
static_assert(அளவு(block_with_zp<4, 16>) == 16 * அளவு(ggml_half) + QK4_0 * 8 + 16 * அளவு(uint8_t),
              "wrong block_with_zp<4,16> size/padding");

static_assert(அளவு(block<8, 16>) == 16 * அளவு(ggml_half) + QK4_0 * 16, "wrong block<8,16> size/padding");

static_assert(அளவு(block<4, 32>) == 32 * அளவு(ggml_half) + QK4_0 * 16, "wrong block<4,32> size/padding");
static_assert(அளவு(block_with_zp<4, 32>) == 32 * அளவு(ggml_half) + QK4_0 * 16 + 32 * அளவு(uint8_t),
              "wrong block_with_zp<4,32> size/padding");

using block_q4_0x16 = block<4, 16>;
using block_q4_1x16 = block_with_zp<4, 16>;
using block_q8_0x16 = block<8, 16>;

using block_q4_0x32 = block<4, 32>;
using block_q4_1x32 = block_with_zp<4, 32>;
using block_q8_0x32 = block<8, 32>;

கட்டமைப்பு block_q4_0x32x256 {
    block_q4_0x32 blocks[8];  // [f16 * 32 | i4 * 32 * 32] * 8
};

கட்டமைப்பு block_q4_1x32x256 {
    block_q4_0x32 blocks[8];
    uint8_t       zps[32 * 8];
};

நிலையான block_q4_0x16 make_block_q4_0x16(block_q4_0 * in, unsigned முழு blck_size_interleave) {
    block_q4_0x16 out;
    GGML_ASSERT(QK4_0 / blck_size_interleave == 2);

    க்கு (முழு i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    க்கு (முழு i = 0; i < 16; i++) {
        // [0, 15], in.d & 0x0F
        க்கு (முழு j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b8] ......... [b7 b15]
            out.qs[i * QK4_0 / 4 + j] = (in[i].qs[j] & 0x0F) | ((in[i].qs[j + QK4_0 / 4] & 0x0F) << 4);
        }
    }

    க்கு (முழு i = 0; i < 16; i++) {
        // [16, 31], in.d & 0xF0
        க்கு (முழு j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            out.qs[4 * QK4_0 + i * QK4_0 / 4 + j] = ((in[i].qs[j] & 0xF0) >> 4) | (in[i].qs[j + QK4_0 / 4] & 0xF0);
        }
    }

    திரும்பு out;
}

நிலையான block_q4_1x16 make_block_q4_1x16(block_q4_1 * in, unsigned முழு blck_size_interleave) {
    block_q4_1x16 out;
    GGML_ASSERT(QK4_1 / blck_size_interleave == 2);

    க்கு (முழு i = 0; i < 16; i++) {
        தசம d   = GGML_FP16_TO_FP32(in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        தசம m   = GGML_FP16_TO_FP32(in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
        தசம mid = -std::nearbyintf(m / d);
        mid       = std::min(15.0f, std::max(0.0f, mid));
        out.d[i]  = GGML_FP32_TO_FP16(d);
        out.zp[i] = static_cast<uint8_t>(mid);
    }

    க்கு (முழு i = 0; i < 16; i++) {
        // [0, 15], in.d & 0x0F
        க்கு (முழு j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b8] ......... [b7 b15]
            out.qs[i * QK4_1 / 4 + j] = (in[i].qs[j] & 0x0F) | ((in[i].qs[j + QK4_1 / 4] & 0x0F) << 4);
        }
    }

    க்கு (முழு i = 0; i < 16; i++) {
        // [16, 31], in.d & 0xF0
        க்கு (முழு j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            out.qs[4 * QK4_1 + i * QK4_1 / 4 + j] = ((in[i].qs[j] & 0xF0) >> 4) | (in[i].qs[j + QK4_1 / 4] & 0xF0);
        }
    }

    திரும்பு out;
}

நிலையான முழு repack_q4_0_to_q4_0_16_bl(ggml_tensor *              t,
                                     முழு                        interleave_block,
                                     மாறிலி வெறுமை * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 16);

    constexpr முழு nrows_interleaved = 16;

    block_q4_0x16 *    dst = (block_q4_0x16 *) t->data;
    மாறிலி block_q4_0 * src = (மாறிலி block_q4_0 *) data;
    block_q4_0         dst_tmp[16];
    முழு                nrow    = ggml_nrows(t);
    முழு                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q4_0));

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

நிலையான முழு repack_q4_1_to_q4_1_16_bl(ggml_tensor *              t,
                                     முழு                        interleave_block,
                                     மாறிலி வெறுமை * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_block == 16);

    constexpr முழு nrows_interleaved = 16;

    block_q4_1x16 *    dst = (block_q4_1x16 *) t->data;
    மாறிலி block_q4_1 * src = (மாறிலி block_q4_1 *) data;
    block_q4_1         dst_tmp[16];
    முழு                nrow    = ggml_nrows(t);
    முழு                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q4_1));

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_1x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

நிலையான உள்வரி வெறுமை get_scale_min_k4(முழு                           j,
                                    மாறிலி uint8_t * GGML_RESTRICT q,
                                    uint8_t * GGML_RESTRICT       d,
                                    uint8_t * GGML_RESTRICT       m) {
    எனில் (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } இல்லையெனில் {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

நிலையான முழு repack_q4_k_to_q4_1_16_bl(ggml_tensor *              t,
                                     முழு                        interleave_block,
                                     மாறிலி வெறுமை * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_block == 16);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr முழு nrows_interleaved = 16;

    block_q4_1x16 *    dst = (block_q4_1x16 *) t->data;
    மாறிலி block_q4_K * src = (மாறிலி block_q4_K *) data;
    block_q4_1         dst_tmp[16];
    முழு                nrow    = ggml_nrows(t);
    முழு                nblocks = t->ne[0] / QK_K;

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு j = 0; j < 8; j++) {
                க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    மாறிலி தசம d = GGML_FP16_TO_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    மாறிலி தசம min =
                        GGML_FP16_TO_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);
                    மாறிலி தசம d1 = d * sc;
                    மாறிலி தசம m1 = min * m;

                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_TO_FP16(d1);
                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_TO_FP16(-m1);
                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    மாறிலி uint8_t * q                                  = src[x + i * nblocks].qs + (j / 2) * QK4_1;
                    எனில் (j % 2 == 0) {
                        க்கு (முழு ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } இல்லையெனில் {
                        க்கு (முழு ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }
                }
                *dst++ = make_block_q4_1x16(dst_tmp, interleave_block);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

நிலையான block_q4_0x32 make_block_q4_0x32(block_q4_0 * in, unsigned முழு blck_size_interleave) {
    block_q4_0x32 out;
    ப்ரதிஜ்ஞா(QK4_0 / blck_size_interleave == 1);
    GGML_UNUSED(blck_size_interleave);

    க்கு (முழு i = 0; i < 32; i++) {
        out.d[i] = in[i].d;
    }

    க்கு (முழு i = 0; i < 32; i++) {
        // [0, 15], in.d & 0x0F
        க்கு (முழு j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b1] .........  [b14 b15]
            out.qs[i * QK4_0 / 2 + j] = (in[i].qs[j * 2] & 0x0F) | ((in[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
    }

    க்கு (முழு i = 0; i < 32; i++) {
        // [16, 31], in.d & 0xF0
        க்கு (முழு j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b17] ......... [b30 b31]
            out.qs[i * QK4_0 / 2 + QK4_0 / 4 + j] = ((in[i].qs[j * 2] & 0xF0) >> 4) | (in[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    திரும்பு out;
}

நிலையான block_q4_1x32 make_block_q4_1x32(block_q4_1 * in, unsigned முழு blck_size_interleave) {
    block_q4_1x32 out;
    GGML_ASSERT(QK4_1 / blck_size_interleave == 1);
    GGML_UNUSED(blck_size_interleave);

    க்கு (முழு i = 0; i < 32; i++) {
        தசம d   = GGML_FP16_TO_FP32(in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        தசம m   = GGML_FP16_TO_FP32(in[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
        தசம mid = -std::nearbyintf(m / d);
        mid       = std::min(15.0f, std::max(0.0f, mid));
        out.d[i]  = GGML_FP32_TO_FP16(d);
        out.zp[i] = static_cast<uint8_t>(mid);
    }

    க்கு (முழு i = 0; i < 32; i++) {
        // [0, 15], in.d & 0x0F
        க்கு (முழு j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b1] ......... [b14 b15]
            out.qs[i * QK4_1 / 2 + j] = (in[i].qs[j * 2] & 0x0F) | ((in[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
    }

    க்கு (முழு i = 0; i < 32; i++) {
        // [16, 31], in.d & 0xF0
        க்கு (முழு j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            out.qs[i * QK4_1 / 2 + QK4_1 / 4 + j] = ((in[i].qs[j * 2] & 0xF0) >> 4) | (in[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    திரும்பு out;
}

நிலையான block_q8_0x32 make_block_q8_0x32(block_q8_0 * in, unsigned முழு blck_size_interleave) {
    block_q8_0x32 out;
    GGML_ASSERT(QK8_0 / blck_size_interleave == 1);
    GGML_UNUSED(blck_size_interleave);

    க்கு (முழு i = 0; i < 32; i++) {
        out.d[i] = in[i].d;
    }

    க்கு (முழு i = 0; i < 32; i++) {
        memcpy(out.qs + i * QK8_0, in[i].qs, QK8_0);
    }

    திரும்பு out;
}

// IME1: interleave 16 q8_0 rows so a plain vle8 sequence in the i8i8 kernel lands weights in the
// vmadot group/parity layout. Mirrors make_block_q4_0x16 but stores full int8 (no nibble packing).
// qs (512B) = [INNER step 0: reg0..reg7][INNER step 1: reg0..reg7], each reg 32B holding 4 columns
// x (even=K-first-half / odd=K-second-half). Column col -> acc=col/4, cgrp=col%4.
நிலையான block_q8_0x16 make_block_q8_0x16(block_q8_0 * in, unsigned முழு blck_size_interleave) {
    block_q8_0x16 out;
    GGML_ASSERT(QK8_0 / blck_size_interleave == 2);
    GGML_UNUSED(blck_size_interleave);

    க்கு (முழு i = 0; i < 16; i++) {
        out.d[i] = in[i].d;
    }

    memset(out.qs, 0, அளவு(out.qs));
    க்கு (முழு col = 0; col < 16; col++) {
        மாறிலி முழு      acc  = col / 4;
        மாறிலி முழு      cgrp = col % 4;
        மாறிலி int8_t * q    = in[col].qs;
        க்கு (முழு s = 0; s < 2; s++) {
            மாறிலி முழு base   = s * 16;
            uint8_t * reg_lo = out.qs + (s * 8 + acc)     * 32;
            uint8_t * reg_hi = out.qs + (s * 8 + acc + 4) * 32;
            க்கு (முழு i = 0; i < 4; i++) {
                reg_lo[(2 * cgrp)     * 4 + i] = (uint8_t) q[base + 0  + i];
                reg_lo[(2 * cgrp + 1) * 4 + i] = (uint8_t) q[base + 4  + i];
                reg_hi[(2 * cgrp)     * 4 + i] = (uint8_t) q[base + 8  + i];
                reg_hi[(2 * cgrp + 1) * 4 + i] = (uint8_t) q[base + 12 + i];
            }
        }
    }
    திரும்பு out;
}

நிலையான முழு repack_q8_0_to_q8_0_16_bl_ref(ggml_tensor *              t,
                                         முழு                        interleave_block,
                                         மாறிலி வெறுமை * GGML_RESTRICT data,
                                         size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_block == 16);

    constexpr முழு nrows_interleaved = 16;

    block_q8_0x16 *    dst = (block_q8_0x16 *) t->data;
    மாறிலி block_q8_0 * src = (மாறிலி block_q8_0 *) data;
    block_q8_0         dst_tmp[16];
    முழு                nrow    = ggml_nrows(t);
    முழு                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q8_0));

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK8_0 != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q8_0x16(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

நிலையான முழு repack_q2_k_to_q2_k_32_bl(ggml_tensor *              t,
                                     முழு                        interleave_block,
                                     மாறிலி வெறுமை * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_block == 32);
    GGML_ASSERT(QK_K == 256);

    constexpr முழு nrows_interleaved = 32;

    மாறிலி block_q2_K * src = (மாறிலி block_q2_K *) data;

    auto * dst = (spacemit_kernels::nrow_block_q2_k<32> *) t->data;

    முழு nrow    = ggml_nrows(t);
    முழு nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q2_K));

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        திரும்பு -1;
    }

    uint8_t qs_aux[256] = { 0 };
    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                மாறிலி block_q2_K * src_block = &src[(b + i) * nblocks + x];

                // scale for [16, N]
                க்கு (முழு j = 0; j < 16; j++) {
                    auto zp_aux = (dst->scales[j * nrows_interleaved + i]) & 0xF0;

                    dst->scales[j * nrows_interleaved + i] = (src_block->scales[j] & 0x0F) | zp_aux;
                }

                // zp for [N, 16]
                க்கு (முழு j = 0; j < 16; j++) {
                    auto scale_aux = (dst->scales[16 * i + j]) & 0x0F;

                    dst->scales[16 * i + j] = (src_block->scales[j] & 0xF0) | scale_aux;
                }

                க்கு (முழு k = 0; k < 4; k++) {
                    க்கு (முழு j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j] = (src_block->qs[j] >> (2 * k)) & 0x03;
                    }
                }

                க்கு (முழு k = 0; k < 4; k++) {
                    க்கு (முழு j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j + 128] = (src_block->qs[j + 32] >> (2 * k)) & 0x03;
                    }
                }

                // from nrows_interleaved * [2 * 32byte]
                // to 4 * [nrows_interleaved * 16byte]
                க்கு (முழு k = 0; k < 4; k++) {
                    க்கு (முழு j = 0; j < 16; j++) {
                        uint8_t qs0  = qs_aux[j + k * 64];
                        uint8_t qs16 = qs_aux[j + 16 + k * 64];
                        uint8_t qs32 = qs_aux[j + 32 + k * 64];
                        uint8_t qs48 = qs_aux[j + 48 + k * 64];

                        dst->qs[(k * nrows_interleaved + i) * 16 + j] =
                            (qs0 & 0x03) | ((qs16 & 0x03) << 2) | ((qs32 & 0x03) << 4) | ((qs48 & 0x03) << 6);
                    }
                }

                dst->scales16[i] = src_block->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
                dst->zeros16[i]  = src_block->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
            }
            dst++;
        }
    }

    திரும்பு 0;
}

நிலையான முழு repack_q3_k_to_q3_k_32_bl(ggml_tensor *              t,
                                     முழு                        interleave_block,
                                     மாறிலி வெறுமை * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q3_K);
    GGML_ASSERT(interleave_block == 32);
    GGML_ASSERT(QK_K == 256);

    constexpr முழு nrows_interleaved = 32;

    மாறிலி uint32_t kmask1 = 0x03030303;
    மாறிலி uint32_t kmask2 = 0x0f0f0f0f;

    மாறிலி block_q3_K * src = (மாறிலி block_q3_K *) data;

    auto * dst = (spacemit_kernels::nrow_block_q3_k<32> *) t->data;

    முழு nrow    = ggml_nrows(t);
    முழு nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q3_K));

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        திரும்பு -1;
    }

    uint32_t b_scale_aux[4] = { 0 };
    uint8_t  qs_aux[256]    = { 0 };

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                மாறிலி block_q3_K * src_block = &src[(b + i) * nblocks + x];

                uint32_t * auxs  = b_scale_aux;
                int8_t *   scale = (int8_t *) auxs;
                memcpy(auxs, src_block->scales, 12);

                uint32_t tmp = auxs[2];
                auxs[2]      = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                auxs[3]      = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                auxs[0]      = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                auxs[1]      = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

                க்கு (முழு j = 0; j < 16; j++) {
                    dst->scales[j * nrows_interleaved + i] = scale[j] - 32;
                }

                க்கு (முழு k = 0; k < 4; k++) {
                    க்கு (முழு j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j] = (src_block->qs[j] >> (2 * k)) & 0x03;
                    }
                }

                க்கு (முழு k = 0; k < 4; k++) {
                    க்கு (முழு j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j + 128] = (src_block->qs[j + 32] >> (2 * k)) & 0x03;
                    }
                }

                // from nrows_interleaved * [2 * 32byte]
                // to 4 * [nrows_interleaved * 16byte]
                க்கு (முழு k = 0; k < 4; k++) {
                    க்கு (முழு j = 0; j < 16; j++) {
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
                uint16_t * dst_mask = ((uint16_t *) dst->hmask) + i;
                க்கு (முழு j = 0; j < 16; j++, dst_mask += nrows_interleaved) {
                    uint8_t   b_shift    = j / 2;
                    uint8_t * b_mask_col = (uint8_t *) (src_block->hmask + (j % 2) * 16);
                    // b0 - b15
                    uint16_t  msk_out_0  = 0;

                    க்கு (முழு k = 0; k < 8; k++) {
                        msk_out_0 |= (uint16_t) ((b_mask_col[k] >> b_shift) & 0x01) << k;
                    }
                    க்கு (முழு k = 8; k < 16; k++) {
                        msk_out_0 |= (uint16_t) ((b_mask_col[k] >> b_shift) & 0x01) << k;
                    }

                    dst_mask[0] = msk_out_0;
                }

                dst->scales16[i] = src_block->d;
            }

            dst++;
        }
    }

    திரும்பு 0;
}

நிலையான முழு repack_q4_0_to_q4_0_32_bl_ref(ggml_tensor *              t,
                                         முழு                        interleave_block,
                                         மாறிலி வெறுமை * GGML_RESTRICT data,
                                         size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr முழு nrows_interleaved = 32;

    block_q4_0x32 *    dst = (block_q4_0x32 *) t->data;
    மாறிலி block_q4_0 * src = (மாறிலி block_q4_0 *) data;
    block_q4_0         dst_tmp[32];
    முழு                nrow    = ggml_nrows(t);
    முழு                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q4_0));

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_0x32(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

நிலையான முழு repack_q4_0_to_q4_0_256_32_bl_ref(ggml_tensor *              t,
                                             முழு                        interleave_block,
                                             மாறிலி வெறுமை * GGML_RESTRICT data,
                                             size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr முழு nrows_interleaved = 32;

    block_q4_0x32x256 * dst = (block_q4_0x32x256 *) t->data;
    மாறிலி block_q4_0 *  src = (மாறிலி block_q4_0 *) data;
    block_q4_0          dst_tmp[32];
    முழு                 nrow    = ggml_nrows(t);
    முழு                 nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q4_0));
    GGML_ASSERT(nblocks % 8 == 0);  // for 256-block interleaving
    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x += 8) {
            க்கு (முழு j = 0; j < 8; j++) {
                க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                    dst_tmp[i] = src[x + j + i * nblocks];
                }
                dst->blocks[j] = make_block_q4_0x32(dst_tmp, interleave_block);
            }
            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

நிலையான முழு repack_q4_0_to_q4_1_256_32_bl_ref(ggml_tensor *              t,
                                             முழு                        interleave_block,
                                             மாறிலி வெறுமை * GGML_RESTRICT data,
                                             size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr முழு nrows_interleaved = 32;

    block_q4_1x32x256 * dst = (block_q4_1x32x256 *) t->data;
    மாறிலி block_q4_1 *  src = (மாறிலி block_q4_1 *) data;
    block_q4_1          dst_tmp[32];
    முழு                 nrow    = ggml_nrows(t);
    முழு                 nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q4_1));
    GGML_ASSERT(nblocks % 8 == 0);  // for 256-block interleaving
    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x += 8) {
            க்கு (முழு j = 0; j < 8; j++) {
                க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                    dst_tmp[i] = src[x + j + i * nblocks];
                }

                block_q4_0x32 * dst_block = &dst->blocks[j];
                uint8_t *       dst_zp    = dst->zps + j * nrows_interleaved;

                க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                    தசம d   = GGML_FP16_TO_FP32(dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    தசம m   = GGML_FP16_TO_FP32(dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
                    தசம mid = -std::nearbyintf(m / d);
                    mid       = std::min(15.0f, std::max(0.0f, mid));

                    dst_block->d[i] = GGML_FP32_TO_FP16(d);
                    dst_zp[i]       = static_cast<uint8_t>(mid);
                }

                க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                    க்கு (முழு k = 0; k < QK4_1 / 4; k++) {
                        dst_block->qs[i * QK4_1 / 2 + k] =
                            (dst_tmp[i].qs[k * 2] & 0x0F) | ((dst_tmp[i].qs[k * 2 + 1] & 0x0F) << 4);
                    }
                }

                க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                    க்கு (முழு k = 0; k < QK4_1 / 4; k++) {
                        dst_block->qs[i * QK4_1 / 2 + QK4_1 / 4 + k] =
                            ((dst_tmp[i].qs[k * 2] & 0xF0) >> 4) | (dst_tmp[i].qs[k * 2 + 1] & 0xF0);
                    }
                }
            }
            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

// RVV optimized version of repack_q4_0_to_q4_0_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes nibble repack.
நிலையான முழு repack_q4_0_to_q4_0_32_bl(ggml_tensor *              t,
                                     முழு                        interleave_block,
                                     மாறிலி வெறுமை * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_block == 32);

    constexpr முழு nrows_interleaved = 32;
    constexpr முழு qs_bytes          = QK4_0 / 2;  // 16

    block_q4_0x32 *    dst     = (block_q4_0x32 *) t->data;
    மாறிலி block_q4_0 * src     = (மாறிலி block_q4_0 *) data;
    முழு                nrow    = ggml_nrows(t);
    முழு                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q4_0));

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        திரும்பு -1;
    }

    மாறிலி ptrdiff_t row_stride = (ptrdiff_t) nblocks * அளவு(block_q4_0);

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            மாறிலி block_q4_0 * col_src = src + x;

            // --- 1) Gather 32 scale values (ggml_half d) with stride load ---
            // d is at offset 0 of each block_q4_0, stride between rows = row_stride
            {
                மாறிலி uint8_t * d_base    = (மாறிலி uint8_t *) &col_src->d;
                ggml_half *     d_dst     = dst->d;
                size_t          remaining = 32;
                size_t          offset    = 0;
                போது (remaining > 0) {
                    size_t      vl = __riscv_vsetvl_e16m1(remaining);
                    vuint16m1_t vd =
                        __riscv_vlse16_v_u16m1((மாறிலி uint16_t *) (d_base + offset * row_stride), row_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + offset), vd, vl);
                    offset += vl;
                    remaining -= vl;
                }
            }

            // --- 2) Nibble repack qs for each of the 32 rows ---
            // For each row i:
            //   src qs[16]: [b0|b16] [b1|b17] ... [b15|b31]  (lo nibble = b_j, hi nibble = b_{j+16})
            //   dst qs low  8B: (qs[2j] & 0x0F) | ((qs[2j+1] & 0x0F) << 4)  for j=0..7
            //   dst qs high 8B: ((qs[2j] >> 4))  | (qs[2j+1] & 0xF0)         for j=0..7
            {
                மாறிலி size_t vl8 = __riscv_vsetvl_e8m1(8);
                க்கு (முழு i = 0; i < 32; i++) {
                    மாறிலி uint8_t * sq = col_src[i * nblocks].qs;
                    uint8_t *       dq = dst->qs + i * qs_bytes;

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
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

நிலையான முழு repack_q4_1_to_q4_1_32_bl_ref(ggml_tensor *              t,
                                         முழு                        interleave_block,
                                         மாறிலி வெறுமை * GGML_RESTRICT data,
                                         size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr முழு nrows_interleaved = 32;

    block_q4_1x32 *    dst = (block_q4_1x32 *) t->data;
    மாறிலி block_q4_1 * src = (மாறிலி block_q4_1 *) data;
    block_q4_1         dst_tmp[32];
    முழு                nrow    = ggml_nrows(t);
    முழு                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q4_1));

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_block_q4_1x32(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

// RVV optimized version of repack_q4_1_to_q4_1_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes nibble repack + zp computation.
நிலையான முழு repack_q4_1_to_q4_1_32_bl(ggml_tensor *              t,
                                     முழு                        interleave_block,
                                     மாறிலி வெறுமை * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_block == 32);

    constexpr முழு nrows_interleaved = 32;
    constexpr முழு qs_bytes          = QK4_1 / 2;  // 16

    block_q4_1x32 *    dst     = (block_q4_1x32 *) t->data;
    மாறிலி block_q4_1 * src     = (மாறிலி block_q4_1 *) data;
    முழு                nrow    = ggml_nrows(t);
    முழு                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q4_1));

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        திரும்பு -1;
    }

    மாறிலி ptrdiff_t row_stride = (ptrdiff_t) nblocks * அளவு(block_q4_1);

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            மாறிலி block_q4_1 * col_src = src + x;

            // --- 1) Gather d and m, compute zp = clamp(nearbyint(-m/d), 0, 15) ---
            // block_q4_1 layout: [d(f16), m(f16), qs[16]]
            // d is at byte offset 0, m is at byte offset 2 from each block start
            {
                மாறிலி uint8_t * dm_base   = (மாறிலி uint8_t *) &col_src->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
                ggml_half *     d_dst     = dst->d;
                uint8_t *       zp_dst    = dst->zp;
                size_t          remaining = 32;
                size_t          offset    = 0;
                போது (remaining > 0) {
                    size_t vl = __riscv_vsetvl_e16m1(remaining);

                    // stride load d (f16) from each row
                    vuint16m1_t vd_raw =
                        __riscv_vlse16_v_u16m1((மாறிலி uint16_t *) (dm_base + offset * row_stride), row_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + offset), vd_raw, vl);

                    // stride load m (f16) from each row (offset +2 bytes from d)
                    vuint16m1_t vm_raw =
                        __riscv_vlse16_v_u16m1((மாறிலி uint16_t *) (dm_base + 2 + offset * row_stride), row_stride, vl);

                    // convert to f32 for zp computation: zp = nearbyint(-m / d)
                    vfloat16m1_t vd_f16 = __riscv_vreinterpret_v_u16m1_f16m1(vd_raw);
                    vfloat16m1_t vm_f16 = __riscv_vreinterpret_v_u16m1_f16m1(vm_raw);

                    // -m / d in f16 directly (SpaceMIT X60 supports f16 arithmetic)
                    vfloat16m1_t v_neg_m = __riscv_vfneg_v_f16m1(vm_f16, vl);
                    vfloat16m1_t v_ratio = __riscv_vfdiv_vv_f16m1(v_neg_m, vd_f16, vl);

                    // Convert to f32 for nearbyint, then clamp
                    vfloat32m2_t v_ratio_f32 = __riscv_vfwcvt_f_f_v_f32m2(v_ratio, vl);

                    // Use integer rounding: convert f32 -> int (rounds to nearest)
                    vint32m2_t v_zp_i32 = __riscv_vfcvt_x_f_v_i32m2(v_ratio_f32, vl);

                    // clamp to [0, 15]
                    v_zp_i32 = __riscv_vmax_vx_i32m2(v_zp_i32, 0, vl);
                    v_zp_i32 = __riscv_vmin_vx_i32m2(v_zp_i32, 15, vl);

                    // narrow i32 -> u8
                    vint16m1_t  v_zp_i16 = __riscv_vncvt_x_x_w_i16m1(v_zp_i32, vl);
                    vint8mf2_t  v_zp_i8  = __riscv_vncvt_x_x_w_i8mf2(v_zp_i16, vl);
                    vuint8mf2_t v_zp_u8  = __riscv_vreinterpret_v_i8mf2_u8mf2(v_zp_i8);
                    __riscv_vse8_v_u8mf2(zp_dst + offset, v_zp_u8, vl);

                    offset += vl;
                    remaining -= vl;
                }
            }

            // --- 2) Nibble repack qs for each of the 32 rows ---
            {
                மாறிலி size_t vl8 = __riscv_vsetvl_e8m1(8);
                க்கு (முழு i = 0; i < 32; i++) {
                    மாறிலி uint8_t * sq = col_src[i * nblocks].qs;
                    uint8_t *       dq = dst->qs + i * qs_bytes;

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
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

நிலையான முழு repack_q4_k_to_q4_1_32_bl(ggml_tensor *              t,
                                     முழு                        interleave_block,
                                     மாறிலி வெறுமை * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_block == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr முழு nrows_interleaved = 32;

    block_q4_1x32 *    dst = (block_q4_1x32 *) t->data;
    மாறிலி block_q4_K * src = (மாறிலி block_q4_K *) data;
    block_q4_1         dst_tmp[32];
    முழு                nrow    = ggml_nrows(t);
    முழு                nblocks = t->ne[0] / QK_K;

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு j = 0; j < 8; j++) {
                க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    மாறிலி தசம d = GGML_FP16_TO_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    மாறிலி தசம min =
                        GGML_FP16_TO_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);
                    மாறிலி தசம d1 = d * sc;
                    மாறிலி தசம m1 = min * m;

                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_TO_FP16(d1);
                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_TO_FP16(-m1);
                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    மாறிலி uint8_t * q                                  = src[x + i * nblocks].qs + (j / 2) * QK4_1;
                    எனில் (j % 2 == 0) {
                        க்கு (முழு ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } இல்லையெனில் {
                        க்கு (முழு ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }
                }
                *dst++ = make_block_q4_1x32(dst_tmp, interleave_block);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

நிலையான முழு repack_q6_k_to_q8_0_32_bl_ref(ggml_tensor *              t,
                                         முழு                        interleave_block,
                                         மாறிலி வெறுமை * GGML_RESTRICT data,
                                         size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_block == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr முழு nrows_interleaved = 32;

    block_q8_0x32 *    dst = (block_q8_0x32 *) t->data;
    மாறிலி block_q6_K * src = (மாறிலி block_q6_K *) data;
    block_q8_0         dst_tmp[32];
    int8_t             aux8[QK4_1];
    முழு                nrow    = ggml_nrows(t);
    முழு                nblocks = t->ne[0] / QK_K;

    எனில் (t->ne[0] % QK_K != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        int64_t nrow_real = std::min((int64_t) nrow - b, (int64_t) nrows_interleaved);
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு bi = 0; bi < 8; bi++) {
                முழு i = 0;
                க்கு (; i < nrow_real; i++) {
                    மாறிலி uint8_t * q4     = src[x + i * nblocks].ql;
                    மாறிலி uint8_t * qh     = src[x + i * nblocks].qh;
                    மாறிலி int8_t *  scales = src[x + i * nblocks].scales;
                    தசம           d      = GGML_FP16_TO_FP32(src[x + i * nblocks].d);

                    q4 += 64 * (bi / 4);
                    qh += 32 * (bi / 4);
                    int8_t * GGML_RESTRICT a = aux8;

                    int8_t bi_idx = bi % 4;

                    எனில் (bi_idx == 0) {
                        க்கு (முழு l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        }
                    } இல்லையெனில் எனில் (bi_idx == 1) {
                        க்கு (முழு l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        }
                    } இல்லையெனில் எனில் (bi_idx == 2) {
                        க்கு (முழு l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        }
                    } இல்லையெனில் எனில் (bi_idx == 3) {
                        க்கு (முழு l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        }
                    }
                    a = aux8;

                    தசம a_max_abs = 0.0f;
                    தசம scale_0   = scales[bi * 2 + 0] * d;
                    தசம scale_1   = scales[bi * 2 + 1] * d;
                    க்கு (முழு l = 0; l < 16; ++l) {
                        a_max_abs = std::max(a_max_abs, std::abs(a[l] * scale_0));
                    }

                    க்கு (முழு l = 16; l < 32; ++l) {
                        a_max_abs = std::max(a_max_abs, std::abs(a[l] * scale_1));
                    }

                    தசம reflect_scale   = a_max_abs / ((1 << 7) - 1);
                    தசம reflect_scale_0 = scale_0 / reflect_scale;
                    தசம reflect_scale_1 = scale_1 / reflect_scale;

                    க்கு (முழு l = 0; l < 16; ++l) {
                        தசம a_temp = std::clamp(std::nearbyintf(a[l] * reflect_scale_0), -128.0f, 127.0f);
                        a[l]         = (int8_t) (a_temp);
                    }

                    க்கு (முழு l = 16; l < 32; ++l) {
                        தசம a_temp = std::clamp(std::nearbyintf(a[l] * reflect_scale_1), -128.0f, 127.0f);
                        a[l]         = (int8_t) (a_temp);
                    }

                    dst_tmp[i].d = GGML_FP32_TO_FP16(reflect_scale);

                    memcpy(dst_tmp[i].qs, a, 32 * அளவு(int8_t));
                }

                க்கு (; i < nrows_interleaved; i++) {
                    memset(&dst_tmp[i], 0, அளவு(block_q8_0));
                }

                *dst++ = make_block_q8_0x32(dst_tmp, interleave_block);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

// RVV optimized version of repack_q6_k_to_q8_0_32_bl
// Vectorizes the Q6_K dequant -> requant pipeline using RVV intrinsics.
// For each sub-block (bi), dequant 32 Q6_K values to int6 -> apply two sub-block scales ->
// find max abs -> compute reflect_scale -> requant to int8 -> gather d with stride load.
நிலையான முழு repack_q6_k_to_q8_0_32_bl(ggml_tensor *              t,
                                     முழு                        interleave_block,
                                     மாறிலி வெறுமை * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_block == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr முழு nrows_interleaved = 32;

    block_q8_0x32 *    dst     = (block_q8_0x32 *) t->data;
    மாறிலி block_q6_K * src     = (மாறிலி block_q6_K *) data;
    முழு                nrow    = ggml_nrows(t);
    முழு                nblocks = t->ne[0] / QK_K;

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        திரும்பு -1;
    }

    மாறிலி ptrdiff_t row_stride = (ptrdiff_t) nblocks * அளவு(block_q6_K);

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு bi = 0; bi < 8; bi++) {
                // --- 1) Gather 32 d values with stride load ---
                // We need to compute reflect_scale per row first, so gather d later.
                // Process each row: dequant Q6_K sub-block -> requant to Q8_0
                க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                    மாறிலி block_q6_K * src_blk = &src[x + i * nblocks];
                    மாறிலி uint8_t *    q4      = src_blk->ql + 64 * (bi / 4);
                    மாறிலி uint8_t *    qh      = src_blk->qh + 32 * (bi / 4);
                    மாறிலி int8_t *     scales  = src_blk->scales;
                    தசம              d       = GGML_FP16_TO_FP32(src_blk->d);

                    int8_t bi_idx = bi % 4;

                    // --- Dequant 32 Q6_K values to int6 (range [-32, 31]) using RVV ---
                    // vl = 32 for e8m2 (VLEN=256) or loop for smaller VLEN
                    மாறிலி size_t vl16 = __riscv_vsetvl_e8m1(16);

                    vint8m1_t va_lo, va_hi;  // 16 elements each

                    எனில் (bi_idx == 0) {
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
                    } இல்லையெனில் எனில் (bi_idx == 1) {
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
                    } இல்லையெனில் எனில் (bi_idx == 2) {
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
                    } இல்லையெனில் {  // bi_idx == 3
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
                    தசம scale_0 = scales[bi * 2 + 0] * d;
                    தசம scale_1 = scales[bi * 2 + 1] * d;

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
                    தசம        a_max_abs = __riscv_vfmv_f_s_f32m1_f32(vmax_red);

                    தசம reflect_scale   = a_max_abs / 127.0f;
                    தசம reflect_scale_0 = scale_0 / reflect_scale;
                    தசம reflect_scale_1 = scale_1 / reflect_scale;

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
                    dst->d[i]   = GGML_FP32_TO_FP16(reflect_scale);
                    int8_t * dq = (int8_t *) dst->qs + i * QK8_0;
                    __riscv_vse8_v_i8m1(dq, vi8_lo, vl16);
                    __riscv_vse8_v_i8m1(dq + 16, vi8_hi, vl16);
                }
                dst++;
            }
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

நிலையான முழு repack_q8_0_to_q8_0_32_bl_ref(ggml_tensor *              t,
                                         முழு                        interleave_block,
                                         மாறிலி வெறுமை * GGML_RESTRICT data,
                                         size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr முழு nrows_interleaved = 32;

    block_q8_0x32 *    dst = (block_q8_0x32 *) t->data;
    மாறிலி block_q8_0 * src = (மாறிலி block_q8_0 *) data;
    block_q8_0         dst_tmp[32];
    முழு                nrow    = ggml_nrows(t);
    முழு                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q8_0));

    எனில் (t->ne[0] % QK8_0 != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        int64_t nrows_real = std::min((int64_t) nrow - b, (int64_t) nrows_interleaved);
        க்கு (int64_t x = 0; x < nblocks; x++) {
            முழு i = 0;
            க்கு (; i < nrows_real; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            க்கு (; i < nrows_interleaved; i++) {
                memset(&dst_tmp[i], 0, அளவு(block_q8_0));
            }
            *dst++ = make_block_q8_0x32(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

// RVV optimized version of repack_q8_0_to_q8_0_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes scale gather + qs copy.
நிலையான முழு repack_q8_0_to_q8_0_32_bl(ggml_tensor *              t,
                                     முழு                        interleave_block,
                                     மாறிலி வெறுமை * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_block == 32);

    constexpr முழு nrows_interleaved = 32;

    block_q8_0x32 *    dst     = (block_q8_0x32 *) t->data;
    மாறிலி block_q8_0 * src     = (மாறிலி block_q8_0 *) data;
    முழு                nrow    = ggml_nrows(t);
    முழு                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q8_0));

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK8_0 != 0) {
        திரும்பு -1;
    }

    மாறிலி ptrdiff_t row_stride = (ptrdiff_t) nblocks * அளவு(block_q8_0);

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            மாறிலி block_q8_0 * col_src = src + x;

            // --- 1) Gather 32 scale values (ggml_half d) with stride load ---
            {
                மாறிலி uint8_t * d_base    = (மாறிலி uint8_t *) &col_src->d;
                ggml_half *     d_dst     = dst->d;
                size_t          remaining = 32;
                size_t          offset    = 0;
                போது (remaining > 0) {
                    size_t      vl = __riscv_vsetvl_e16m1(remaining);
                    vuint16m1_t vd =
                        __riscv_vlse16_v_u16m1((மாறிலி uint16_t *) (d_base + offset * row_stride), row_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + offset), vd, vl);
                    offset += vl;
                    remaining -= vl;
                }
            }

            // --- 2) Copy qs for each of the 32 rows (32 bytes per row) ---
            {
                க்கு (முழு i = 0; i < 32; i++) {
                    மாறிலி int8_t * sq = col_src[i * nblocks].qs;
                    int8_t *       dq = (int8_t *) dst->qs + i * QK8_0;

                    size_t len = QK8_0;
                    size_t idx = 0;
                    போது (len > 0) {
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
    திரும்பு 0;

    GGML_UNUSED(data_size);
}

நிலையான வெறுமை convert_mxfp4_to_5bit(மாறிலி block_mxfp4 & src, spacemit_kernels::nrow_block_mxfp4<1> & dst) {
    dst.e[0] = src.e;

    // Decode all 32 mxfp4 values to signed integers via kvalues_mxfp4
    int8_t vals[32];
    க்கு (முழு j = 0; j < QK_MXFP4 / 2; j++) {
        vals[j]                = kvalues_mxfp4[src.qs[j] & 0xF];
        vals[j + QK_MXFP4 / 2] = kvalues_mxfp4[src.qs[j] >> 4];
    }

    // vals [b0, b1, b2, b3, ..., b30, b31]
    // Pack abs into qs with reorder: [b0,b1]..[b14,b15]..[b30,b31]
    க்கு (முழு j = 0; j < QK_MXFP4 / 2; j++) {
        uint8_t lo0 = static_cast<uint8_t>(std::abs(vals[j * 2]));
        uint8_t lo1 = static_cast<uint8_t>(std::abs(vals[j * 2 + 1]));
        dst.qs[j]   = (lo0 & 0x0F) | ((lo1 & 0x0F) << 4);
    }

    // Pack sign bits into qh[4] (32 bits total, 1 bit per weight)
    // reorder: [0,1,2,...,15,16,17,...,31] after the qs reorder above
    uint32_t sign_bits = 0;
    க்கு (முழு j = 0; j < 32; j++) {
        எனில் (vals[j] < 0) {
            sign_bits |= (1u << j);
        }
    }
    memcpy(dst.qh, &sign_bits, 4);
}

நிலையான spacemit_kernels::nrow_block_mxfp4<32> make_block_mxfp4x32(spacemit_kernels::nrow_block_mxfp4<1> * in,
                                                                  unsigned முழு blck_size_interleave) {
    spacemit_kernels::nrow_block_mxfp4<32> out;
    GGML_ASSERT(QK_MXFP4 / blck_size_interleave == 1);
    GGML_UNUSED(blck_size_interleave);

    க்கு (முழு i = 0; i < 32; i++) {
        out.e[i] = in[i].e[0];
    }

    // qs: copy per-row 16 bytes
    க்கு (முழு i = 0; i < 32; i++) {
        memcpy(out.qs + i * 16, in[i].qs, 16);
    }

    // qh: copy per-row 4 bytes
    க்கு (முழு i = 0; i < 32; i++) {
        memcpy(out.qh + i * 4, in[i].qh, 4);
    }

    திரும்பு out;
}

நிலையான முழு repack_mxfp4_to_mxfp4_32_bl(ggml_tensor *              t,
                                       முழு                        interleave_block,
                                       மாறிலி வெறுமை * GGML_RESTRICT data,
                                       size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_block == 32);

    constexpr முழு nrows_interleaved = 32;

    spacemit_kernels::nrow_block_mxfp4<32> * dst = (spacemit_kernels::nrow_block_mxfp4<32> *) t->data;
    மாறிலி block_mxfp4 *                      src = (மாறிலி block_mxfp4 *) data;
    spacemit_kernels::nrow_block_mxfp4<1>    dst_tmp[32];
    முழு                                      nrow    = ggml_nrows(t);
    முழு                                      nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_mxfp4));

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_MXFP4 != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                convert_mxfp4_to_5bit(src[x + i * nblocks], dst_tmp[i]);
            }
            *dst++ = make_block_mxfp4x32(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;
}

நிலையான spacemit_kernels::nrow_block_q5_1<32> make_block_q5_1x32(spacemit_kernels::nrow_block_q5_1<1> * in,
                                                                unsigned முழு blck_size_interleave) {
    spacemit_kernels::nrow_block_q5_1<32> out;
    GGML_ASSERT(QK5_1 / blck_size_interleave == 1);
    GGML_UNUSED(blck_size_interleave);

    க்கு (முழு i = 0; i < 32; i++) {
        out.scales16[i] = in[i].scales16[0];
        out.zp[i]       = in[i].zp[0];
    }

    // qs: low 4 bits, reorder from [b0,b16],[b1,b17]... to [b0,b1]...[b14,b15] and [b16,b17]...[b30,b31]
    க்கு (முழு i = 0; i < 32; i++) {
        // low half [0..15]
        க்கு (முழு j = 0; j < QK5_1 / 4; j++) {
            out.qs[i * QK5_1 / 2 + j] = (in[i].qs[j * 2] & 0x0F) | ((in[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
        // high half [16..31]
        க்கு (முழு j = 0; j < QK5_1 / 4; j++) {
            out.qs[i * QK5_1 / 2 + QK5_1 / 4 + j] = ((in[i].qs[j * 2] & 0xF0) >> 4) | (in[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    // qh: 5th bit, copy directly
    க்கு (முழு i = 0; i < 32; i++) {
        க்கு (முழு j = 0; j < 4; j++) {
            out.qh[i * 4 + j] = in[i].qh[j];
        }
    }

    திரும்பு out;
}

நிலையான spacemit_kernels::nrow_block_q5_0<32> make_block_q5_0x32(spacemit_kernels::nrow_block_q5_0<1> * in,
                                                                unsigned முழு blck_size_interleave) {
    spacemit_kernels::nrow_block_q5_0<32> out;
    GGML_ASSERT(QK5_0 / blck_size_interleave == 1);
    GGML_UNUSED(blck_size_interleave);

    க்கு (முழு i = 0; i < 32; i++) {
        out.scales16[i] = in[i].scales16[0];
    }

    // qs: low 4 bits, reorder from [b0,b16],[b1,b17]... to [b0,b1]...[b14,b15] and [b16,b17]...[b30,b31]
    க்கு (முழு i = 0; i < 32; i++) {
        // low half [0..15]
        க்கு (முழு j = 0; j < QK5_0 / 4; j++) {
            out.qs[i * QK5_0 / 2 + j] = (in[i].qs[j * 2] & 0x0F) | ((in[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
        // high half [16..31]
        க்கு (முழு j = 0; j < QK5_0 / 4; j++) {
            out.qs[i * QK5_0 / 2 + QK5_0 / 4 + j] = ((in[i].qs[j * 2] & 0xF0) >> 4) | (in[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    // qh: 5th bit, copy directly
    க்கு (முழு i = 0; i < 32; i++) {
        க்கு (முழு j = 0; j < 4; j++) {
            out.qh[i * 4 + j] = in[i].qh[j];
        }
    }

    திரும்பு out;
}

நிலையான முழு repack_q5_0_to_q5_0_32_bl(ggml_tensor *              t,
                                     முழு                        interleave_block,
                                     மாறிலி வெறுமை * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr முழு nrows_interleaved = 32;

    spacemit_kernels::nrow_block_q5_0<32> * dst = (spacemit_kernels::nrow_block_q5_0<32> *) t->data;
    மாறிலி block_q5_0 *                      src = (மாறிலி block_q5_0 *) data;
    spacemit_kernels::nrow_block_q5_0<1>    dst_tmp[32];
    முழு                                     nrow    = ggml_nrows(t);
    முழு                                     nblocks = t->ne[0] / QK5_0;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q5_0));

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK5_0 != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                மாறிலி block_q5_0 & s = src[x + i * nblocks];

                dst_tmp[i].scales16[0] = s.d;
                memcpy(dst_tmp[i].qs, s.qs, அளவு(dst_tmp[i].qs));
                memcpy(dst_tmp[i].qh, s.qh, அளவு(dst_tmp[i].qh));
            }
            *dst++ = make_block_q5_0x32(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;
}

நிலையான முழு repack_q5_1_to_q5_1_32_bl(ggml_tensor *              t,
                                     முழு                        interleave_block,
                                     மாறிலி வெறுமை * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(interleave_block == 32);  // unused

    constexpr முழு nrows_interleaved = 32;

    spacemit_kernels::nrow_block_q5_1<32> * dst = (spacemit_kernels::nrow_block_q5_1<32> *) t->data;
    மாறிலி block_q5_1 *                      src = (மாறிலி block_q5_1 *) data;
    spacemit_kernels::nrow_block_q5_1<1>    dst_tmp[32];
    முழு                                     nrow    = ggml_nrows(t);
    முழு                                     nblocks = t->ne[0] / QK5_1;

    GGML_ASSERT(data_size == nrow * nblocks * அளவு(block_q5_1));

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK5_1 != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                மாறிலி block_q5_1 & s = src[x + i * nblocks];

                தசம d = GGML_FP16_TO_FP32(s.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                தசம m = GGML_FP16_TO_FP32(s.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);

                எனில் (d == 0.0f) {
                    dst_tmp[i].scales16[0] = GGML_FP32_TO_FP16(std::fabs(m));
                    dst_tmp[i].zp[0]       = m < 0.0f ? 1 : 0;
                    memset(dst_tmp[i].qh, 0, அளவு(dst_tmp[i].qh));
                    memset(dst_tmp[i].qs, m > 0.0f ? 0x11 : 0x00, அளவு(dst_tmp[i].qs));
                    தொடர்;
                }

                தசம mid = std::nearbyintf(-m / d);
                mid       = std::min(31.0f, std::max(0.0f, mid));

                dst_tmp[i].scales16[0] = GGML_FP32_TO_FP16(d);
                dst_tmp[i].zp[0]       = static_cast<uint8_t>(mid);

                // qs: copy low 4 bits directly (same nibble packing)
                memcpy(dst_tmp[i].qs, s.qs, QK5_1 / 2);

                // qh: copy 5th bit directly
                memcpy(dst_tmp[i].qh, s.qh, 4);
            }
            *dst++ = make_block_q5_1x32(dst_tmp, interleave_block);
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;
}

நிலையான முழு repack_q5_k_to_q5_1_32_bl(ggml_tensor *              t,
                                     முழு                        interleave_block,
                                     மாறிலி வெறுமை * GGML_RESTRICT data,
                                     size_t                     data_size) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_block == 32);
    GGML_ASSERT(QK_K / QK5_1 == 8);

    constexpr முழு nrows_interleaved = 32;

    spacemit_kernels::nrow_block_q5_1<32> * dst = (spacemit_kernels::nrow_block_q5_1<32> *) t->data;
    மாறிலி block_q5_K *                      src = (மாறிலி block_q5_K *) data;
    spacemit_kernels::nrow_block_q5_1<1>    dst_tmp[32];
    முழு                                     nrow    = ggml_nrows(t);
    முழு                                     nblocks = t->ne[0] / QK_K;

    எனில் (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        திரும்பு -1;
    }

    க்கு (முழு b = 0; b < nrow; b += nrows_interleaved) {
        க்கு (int64_t x = 0; x < nblocks; x++) {
            க்கு (முழு j = 0; j < 8; j++) {
                க்கு (முழு i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    மாறிலி தசம d = GGML_FP16_TO_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    மாறிலி தசம min =
                        GGML_FP16_TO_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);

                    தசம d1 = d * sc;
                    தசம m1 = min * m;

                    தசம mid              = std::nearbyintf(m1 / d1);
                    mid                    = std::min(31.0f, std::max(0.0f, mid));
                    dst_tmp[i].scales16[0] = GGML_FP32_TO_FP16(d1);
                    dst_tmp[i].zp[0]       = static_cast<uint8_t>(mid);

                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    மாறிலி uint8_t * q = src[x + i * nblocks].qs + (j / 2) * QK5_1;
                    எனில் (j % 2 == 0) {
                        க்கு (முழு ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } இல்லையெனில் {
                        க்கு (முழு ii = 0; ii < 16; ii++) {
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
                    மாறிலி uint8_t * src_qh = src[x + i * nblocks].qh;
                    க்கு (முழு bi = 0; bi < 4; bi++) {
                        uint8_t qh_byte = 0;
                        க்கு (முழு k = 0; k < 8; k++) {
                            முழு src_idx = bi * 8 + k;
                            qh_byte |= ((src_qh[src_idx] >> j) & 1) << k;
                        }
                        dst_tmp[i].qh[bi] = qh_byte;
                    }
                }
                *dst++ = make_block_q5_1x32(dst_tmp, interleave_block);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    திரும்பு 0;
}

நாமாகாஶ ggml::cpu::riscv64_spacemit {

template <typename BLOC_TYPE, int64_t INTER_SIZE, int64_t NB_COLS> முழு repack(ggml_tensor *, மாறிலி வெறுமை *, size_t);

template <> முழு repack<block_q4_0, 32, 16>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
    திரும்பு repack_q4_0_to_q4_0_16_bl(t, 16, data, data_size);
}

template <> முழு repack<block_q4_1, 32, 16>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
    திரும்பு repack_q4_1_to_q4_1_16_bl(t, 16, data, data_size);
}

template <> முழு repack<block_q4_K, 32, 16>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
    திரும்பு repack_q4_k_to_q4_1_16_bl(t, 16, data, data_size);
}

template <> முழு repack<block_q2_K, 256, 32>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
    திரும்பு repack_q2_k_to_q2_k_32_bl(t, 32, data, data_size);
}

template <> முழு repack<block_q3_K, 256, 32>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
    திரும்பு repack_q3_k_to_q3_k_32_bl(t, 32, data, data_size);
}

template <> முழு repack<block_q4_0, 32, 32>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
#எனில் 0
    திரும்பு repack_q4_0_to_q4_0_32_bl_ref(t, 32, data, data_size);
#இல்லையெனில்
    திரும்பு repack_q4_0_to_q4_0_32_bl(t, 32, data, data_size);
#endif
}

template <> முழு repack<block_q4_0, 256, 32>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
#எனில் 1
    திரும்பு repack_q4_0_to_q4_0_256_32_bl_ref(t, 32, data, data_size);
#இல்லையெனில்
    //return repack_q4_0_to_q4_0_256_32_bl(t, 32, data, data_size);
#endif
}

template <> முழு repack<block_q4_1, 32, 32>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
#எனில் 0
    திரும்பு repack_q4_1_to_q4_1_32_bl_ref(t, 32, data, data_size);
#இல்லையெனில்
    திரும்பு repack_q4_1_to_q4_1_32_bl(t, 32, data, data_size);
#endif
}

template <> முழு repack<block_q4_1, 256, 32>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
#எனில் 1
    திரும்பு repack_q4_0_to_q4_1_256_32_bl_ref(t, 32, data, data_size);
#இல்லையெனில்
    திரும்பு repack_q4_1_to_q4_1_256_32_bl(t, 32, data, data_size);
#endif
}

template <> முழு repack<block_q4_K, 32, 32>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
    திரும்பு repack_q4_k_to_q4_1_32_bl(t, 32, data, data_size);
}

template <> முழு repack<block_q6_K, 32, 32>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
#எனில் 1
    திரும்பு repack_q6_k_to_q8_0_32_bl_ref(t, 32, data, data_size);
#இல்லையெனில்
    திரும்பு repack_q6_k_to_q8_0_32_bl(t, 32, data, data_size);
#endif
}

template <> முழு repack<block_q8_0, 32, 16>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
    திரும்பு repack_q8_0_to_q8_0_16_bl_ref(t, 16, data, data_size);
}

template <> முழு repack<block_q8_0, 32, 32>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
#எனில் 1
    திரும்பு repack_q8_0_to_q8_0_32_bl_ref(t, 32, data, data_size);
#இல்லையெனில்
    திரும்பு repack_q8_0_to_q8_0_32_bl(t, 32, data, data_size);
#endif
}

template <> முழு repack<block_mxfp4, 32, 32>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
    திரும்பு repack_mxfp4_to_mxfp4_32_bl(t, 32, data, data_size);
}

template <> முழு repack<block_q5_0, 32, 32>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
    திரும்பு repack_q5_0_to_q5_0_32_bl(t, 32, data, data_size);
}

template <> முழு repack<block_q5_1, 32, 32>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
    திரும்பு repack_q5_1_to_q5_1_32_bl(t, 32, data, data_size);
}

template <> முழு repack<block_q5_K, 32, 32>(ggml_tensor * t, மாறிலி வெறுமை * data, size_t data_size) {
    திரும்பு repack_q5_k_to_q5_1_32_bl(t, 32, data, data_size);
}

}  // namespace ggml::cpu::riscv64_spacemit
