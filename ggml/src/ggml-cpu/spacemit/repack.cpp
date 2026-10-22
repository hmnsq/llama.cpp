#నిర్వచించు GGML_COMMON_IMPL_CPP
#నిర్వచించు GGML_COMMON_DECL_CPP

#అన్తర్భావ "repack.h"

#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "ggml-cpu.h"
#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ime_kernels.h"

#అన్తర్భావ <algorithm>
#అన్తర్భావ <cassert>
#అన్తర్భావ <cmath>
#అన్తర్భావ <cstring>

// clang-format off
#ఒకవేళ defined(__riscv)

#ఒకవేళ !defined(__riscv_v) || !defined(__riscv_v_intrinsic)
#దోషం "riscv v extension or v_intrinsic not enabled"
#లేకపోతే
#అన్తర్భావ <riscv_vector.h>
#అయితే_ముగింపు

#ఒకవేళ !defined(__riscv_zfh)
#దోషం "riscv zfh extension not enabled"
#అయితే_ముగింపు

#లేకపోతే
#దోషం "riscv not enabled in this build"
#అయితే_ముగింపు

#ఒకవేళ defined(__GNUC__)
#సూచన GCC diagnostic ignored "-Wcast-qual"
#సూచన GCC diagnostic ignored "-Wunused-parameter"
#అయితే_ముగింపు

// clang-format on

మూస <పూర్ణాంకం K> constexpr పూర్ణాంకం QK_0() {
    ఒకవేళ constexpr (K == 4) {
        తిరిగి QK4_0;
    }
    ఒకవేళ constexpr (K == 8) {
        తిరిగి QK8_0;
    }
    తిరిగి -1;
}

మూస <పూర్ణాంకం K, పూర్ణాంకం N> నిర్మాణం ఖండం {
    ggml_half d[N];                         // deltas for N qK_0 blocks
    uint8_t   qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_0 blocks
};

మూస <పూర్ణాంకం K, పూర్ణాంకం N> నిర్మాణం ఖండం_with_zp {
    ggml_half d[N];                         // deltas for N qK_1 blocks
    uint8_t   zp[N];                        // zero points for N qK_1 blocks
    uint8_t   qs[(QK_0<K>() * N * K) / 8];  // quants for N qK_1 blocks
};

// control size
static_assert(పరిమాణం(ఖండం<4, 16>) == 16 * పరిమాణం(ggml_half) + QK4_0 * 8, "wrong block<4,16> size/padding");
static_assert(పరిమాణం(ఖండం_with_zp<4, 16>) == 16 * పరిమాణం(ggml_half) + QK4_0 * 8 + 16 * పరిమాణం(uint8_t),
              "wrong block_with_zp<4,16> size/padding");

static_assert(పరిమాణం(ఖండం<8, 16>) == 16 * పరిమాణం(ggml_half) + QK4_0 * 16, "wrong block<8,16> size/padding");

static_assert(పరిమాణం(ఖండం<4, 32>) == 32 * పరిమాణం(ggml_half) + QK4_0 * 16, "wrong block<4,32> size/padding");
static_assert(పరిమాణం(ఖండం_with_zp<4, 32>) == 32 * పరిమాణం(ggml_half) + QK4_0 * 16 + 32 * పరిమాణం(uint8_t),
              "wrong block_with_zp<4,32> size/padding");

ఉపయోగించి ఖండం_q4_0x16 = ఖండం<4, 16>;
ఉపయోగించి ఖండం_q4_1x16 = ఖండం_with_zp<4, 16>;
ఉపయోగించి ఖండం_q8_0x16 = ఖండం<8, 16>;

ఉపయోగించి ఖండం_q4_0x32 = ఖండం<4, 32>;
ఉపయోగించి ఖండం_q4_1x32 = ఖండం_with_zp<4, 32>;
ఉపయోగించి ఖండం_q8_0x32 = ఖండం<8, 32>;

నిర్మాణం ఖండం_q4_0x32x256 {
    ఖండం_q4_0x32 ఖండాలు[8];  // [f16 * 32 | i4 * 32 * 32] * 8
};

నిర్మాణం ఖండం_q4_1x32x256 {
    ఖండం_q4_0x32 ఖండాలు[8];
    uint8_t       zps[32 * 8];
};

స్థిర ఖండం_q4_0x16 make_ఖండం_q4_0x16(ఖండం_q4_0 * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q4_0x16 బహిర్గతం;
    GGML_ASSERT(QK4_0 / blck_కొలత_interleave == 2);

    కొరకు (పూర్ణాంకం i = 0; i < 16; i++) {
        బహిర్గతం.d[i] = లోపల[i].d;
    }

    కొరకు (పూర్ణాంకం i = 0; i < 16; i++) {
        // [0, 15], in.d & 0x0F
        కొరకు (పూర్ణాంకం j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b8] ......... [b7 b15]
            బహిర్గతం.qs[i * QK4_0 / 4 + j] = (లోపల[i].qs[j] & 0x0F) | ((లోపల[i].qs[j + QK4_0 / 4] & 0x0F) << 4);
        }
    }

    కొరకు (పూర్ణాంకం i = 0; i < 16; i++) {
        // [16, 31], in.d & 0xF0
        కొరకు (పూర్ణాంకం j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            బహిర్గతం.qs[4 * QK4_0 + i * QK4_0 / 4 + j] = ((లోపల[i].qs[j] & 0xF0) >> 4) | (లోపల[i].qs[j + QK4_0 / 4] & 0xF0);
        }
    }

    తిరిగి బహిర్గతం;
}

స్థిర ఖండం_q4_1x16 make_ఖండం_q4_1x16(ఖండం_q4_1 * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q4_1x16 బహిర్గతం;
    GGML_ASSERT(QK4_1 / blck_కొలత_interleave == 2);

    కొరకు (పూర్ణాంకం i = 0; i < 16; i++) {
        దశాంశం d   = GGML_FP16_గమ్యం_FP32(లోపల[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        దశాంశం m   = GGML_FP16_గమ్యం_FP32(లోపల[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
        దశాంశం mid = -std::nearbyintf(m / d);
        mid       = std::min(15.0f, std::max(0.0f, mid));
        బహిర్గతం.d[i]  = GGML_FP32_గమ్యం_FP16(d);
        బహిర్గతం.zp[i] = static_cast<uint8_t>(mid);
    }

    కొరకు (పూర్ణాంకం i = 0; i < 16; i++) {
        // [0, 15], in.d & 0x0F
        కొరకు (పూర్ణాంకం j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b8] ......... [b7 b15]
            బహిర్గతం.qs[i * QK4_1 / 4 + j] = (లోపల[i].qs[j] & 0x0F) | ((లోపల[i].qs[j + QK4_1 / 4] & 0x0F) << 4);
        }
    }

    కొరకు (పూర్ణాంకం i = 0; i < 16; i++) {
        // [16, 31], in.d & 0xF0
        కొరకు (పూర్ణాంకం j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            బహిర్గతం.qs[4 * QK4_1 + i * QK4_1 / 4 + j] = ((లోపల[i].qs[j] & 0xF0) >> 4) | (లోపల[i].qs[j + QK4_1 / 4] & 0xF0);
        }
    }

    తిరిగి బహిర్గతం;
}

స్థిర పూర్ణాంకం repack_q4_0_గమ్యం_q4_0_16_bl(ggml_tensor *              t,
                                     పూర్ణాంకం                        interleave_ఖండం,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                     size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ఖండం == 16);

    constexpr పూర్ణాంకం nrows_interleaved = 16;

    ఖండం_q4_0x16 *    dst = (ఖండం_q4_0x16 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_0 * src = (స్థిరరాశి ఖండం_q4_0 *) దత్తాంశం;
    ఖండం_q4_0         dst_tmp[16];
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q4_0));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q4_0x16(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q4_1_గమ్యం_q4_1_16_bl(ggml_tensor *              t,
                                     పూర్ణాంకం                        interleave_ఖండం,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                     size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_ఖండం == 16);

    constexpr పూర్ణాంకం nrows_interleaved = 16;

    ఖండం_q4_1x16 *    dst = (ఖండం_q4_1x16 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_1 * src = (స్థిరరాశి ఖండం_q4_1 *) దత్తాంశం;
    ఖండం_q4_1         dst_tmp[16];
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q4_1));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q4_1x16(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర అంతఃవరుస శూన్య get_scale_min_k4(పూర్ణాంకం                           j,
                                    స్థిరరాశి uint8_t * GGML_RESTRICT q,
                                    uint8_t * GGML_RESTRICT       d,
                                    uint8_t * GGML_RESTRICT       m) {
    ఒకవేళ (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } లేకపోతే {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

స్థిర పూర్ణాంకం repack_q4_k_గమ్యం_q4_1_16_bl(ggml_tensor *              t,
                                     పూర్ణాంకం                        interleave_ఖండం,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                     size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_ఖండం == 16);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr పూర్ణాంకం nrows_interleaved = 16;

    ఖండం_q4_1x16 *    dst = (ఖండం_q4_1x16 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_K * src = (స్థిరరాశి ఖండం_q4_K *) దత్తాంశం;
    ఖండం_q4_1         dst_tmp[16];
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK_K;

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
                కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    స్థిరరాశి దశాంశం d = GGML_FP16_గమ్యం_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    స్థిరరాశి దశాంశం min =
                        GGML_FP16_గమ్యం_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);
                    స్థిరరాశి దశాంశం d1 = d * sc;
                    స్థిరరాశి దశాంశం m1 = min * m;

                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_గమ్యం_FP16(d1);
                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_గమ్యం_FP16(-m1);
                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    స్థిరరాశి uint8_t * q                                  = src[x + i * nblocks].qs + (j / 2) * QK4_1;
                    ఒకవేళ (j % 2 == 0) {
                        కొరకు (పూర్ణాంకం ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } లేకపోతే {
                        కొరకు (పూర్ణాంకం ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }
                }
                *dst++ = make_ఖండం_q4_1x16(dst_tmp, interleave_ఖండం);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర ఖండం_q4_0x32 make_ఖండం_q4_0x32(ఖండం_q4_0 * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q4_0x32 బహిర్గతం;
    ప్రతిజ్ఞా(QK4_0 / blck_కొలత_interleave == 1);
    GGML_వాడని(blck_కొలత_interleave);

    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        బహిర్గతం.d[i] = లోపల[i].d;
    }

    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        // [0, 15], in.d & 0x0F
        కొరకు (పూర్ణాంకం j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b1] .........  [b14 b15]
            బహిర్గతం.qs[i * QK4_0 / 2 + j] = (లోపల[i].qs[j * 2] & 0x0F) | ((లోపల[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
    }

    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        // [16, 31], in.d & 0xF0
        కొరకు (పూర్ణాంకం j = 0; j < QK4_0 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b17] ......... [b30 b31]
            బహిర్గతం.qs[i * QK4_0 / 2 + QK4_0 / 4 + j] = ((లోపల[i].qs[j * 2] & 0xF0) >> 4) | (లోపల[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    తిరిగి బహిర్గతం;
}

స్థిర ఖండం_q4_1x32 make_ఖండం_q4_1x32(ఖండం_q4_1 * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q4_1x32 బహిర్గతం;
    GGML_ASSERT(QK4_1 / blck_కొలత_interleave == 1);
    GGML_వాడని(blck_కొలత_interleave);

    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        దశాంశం d   = GGML_FP16_గమ్యం_FP32(లోపల[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
        దశాంశం m   = GGML_FP16_గమ్యం_FP32(లోపల[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
        దశాంశం mid = -std::nearbyintf(m / d);
        mid       = std::min(15.0f, std::max(0.0f, mid));
        బహిర్గతం.d[i]  = GGML_FP32_గమ్యం_FP16(d);
        బహిర్గతం.zp[i] = static_cast<uint8_t>(mid);
    }

    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        // [0, 15], in.d & 0x0F
        కొరకు (పూర్ణాంకం j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b0 b1] ......... [b14 b15]
            బహిర్గతం.qs[i * QK4_1 / 2 + j] = (లోపల[i].qs[j * 2] & 0x0F) | ((లోపల[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
    }

    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        // [16, 31], in.d & 0xF0
        కొరకు (పూర్ణాంకం j = 0; j < QK4_1 / 4; j++) {
            //src [b0 b16] ......... [b8 b24] ......... [b15 b31]
            //dst [b16 b24] ......... [b23 b31]
            బహిర్గతం.qs[i * QK4_1 / 2 + QK4_1 / 4 + j] = ((లోపల[i].qs[j * 2] & 0xF0) >> 4) | (లోపల[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    తిరిగి బహిర్గతం;
}

స్థిర ఖండం_q8_0x32 make_ఖండం_q8_0x32(ఖండం_q8_0 * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q8_0x32 బహిర్గతం;
    GGML_ASSERT(QK8_0 / blck_కొలత_interleave == 1);
    GGML_వాడని(blck_కొలత_interleave);

    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        బహిర్గతం.d[i] = లోపల[i].d;
    }

    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        జ్ఞాపకం_నకలు(బహిర్గతం.qs + i * QK8_0, లోపల[i].qs, QK8_0);
    }

    తిరిగి బహిర్గతం;
}

// IME1: interleave 16 q8_0 rows so a plain vle8 sequence in the i8i8 kernel lands weights in the
// vmadot group/parity layout. Mirrors make_block_q4_0x16 but stores full int8 (no nibble packing).
// qs (512B) = [INNER step 0: reg0..reg7][INNER step 1: reg0..reg7], each reg 32B holding 4 columns
// x (even=K-first-half / odd=K-second-half). Column col -> acc=col/4, cgrp=col%4.
స్థిర ఖండం_q8_0x16 make_ఖండం_q8_0x16(ఖండం_q8_0 * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q8_0x16 బహిర్గతం;
    GGML_ASSERT(QK8_0 / blck_కొలత_interleave == 2);
    GGML_వాడని(blck_కొలత_interleave);

    కొరకు (పూర్ణాంకం i = 0; i < 16; i++) {
        బహిర్గతం.d[i] = లోపల[i].d;
    }

    memset(బహిర్గతం.qs, 0, పరిమాణం(బహిర్గతం.qs));
    కొరకు (పూర్ణాంకం నిలువు = 0; నిలువు < 16; నిలువు++) {
        స్థిరరాశి పూర్ణాంకం      acc  = నిలువు / 4;
        స్థిరరాశి పూర్ణాంకం      cgrp = నిలువు % 4;
        స్థిరరాశి int8_t * q    = లోపల[నిలువు].qs;
        కొరకు (పూర్ణాంకం s = 0; s < 2; s++) {
            స్థిరరాశి పూర్ణాంకం పునాది   = s * 16;
            uint8_t * reg_lo = బహిర్గతం.qs + (s * 8 + acc)     * 32;
            uint8_t * reg_hi = బహిర్గతం.qs + (s * 8 + acc + 4) * 32;
            కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
                reg_lo[(2 * cgrp)     * 4 + i] = (uint8_t) q[పునాది + 0  + i];
                reg_lo[(2 * cgrp + 1) * 4 + i] = (uint8_t) q[పునాది + 4  + i];
                reg_hi[(2 * cgrp)     * 4 + i] = (uint8_t) q[పునాది + 8  + i];
                reg_hi[(2 * cgrp + 1) * 4 + i] = (uint8_t) q[పునాది + 12 + i];
            }
        }
    }
    తిరిగి బహిర్గతం;
}

స్థిర పూర్ణాంకం repack_q8_0_గమ్యం_q8_0_16_bl_ref(ggml_tensor *              t,
                                         పూర్ణాంకం                        interleave_ఖండం,
                                         స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                         size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_ఖండం == 16);

    constexpr పూర్ణాంకం nrows_interleaved = 16;

    ఖండం_q8_0x16 *    dst = (ఖండం_q8_0x16 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q8_0 * src = (స్థిరరాశి ఖండం_q8_0 *) దత్తాంశం;
    ఖండం_q8_0         dst_tmp[16];
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q8_0));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK8_0 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q8_0x16(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q2_k_గమ్యం_q2_k_32_bl(ggml_tensor *              t,
                                     పూర్ణాంకం                        interleave_ఖండం,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                     size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_ఖండం == 32);
    GGML_ASSERT(QK_K == 256);

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    స్థిరరాశి ఖండం_q2_K * src = (స్థిరరాశి ఖండం_q2_K *) దత్తాంశం;

    స్వయంచాలక * dst = (spacemit_kernels::nrow_ఖండం_q2_k<32> *) t->దత్తాంశం;

    పూర్ణాంకం nrow    = ggml_nrows(t);
    పూర్ణాంకం nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q2_K));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        తిరిగి -1;
    }

    uint8_t qs_aux[256] = { 0 };
    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                స్థిరరాశి ఖండం_q2_K * src_ఖండం = &src[(b + i) * nblocks + x];

                // scale for [16, N]
                కొరకు (పూర్ణాంకం j = 0; j < 16; j++) {
                    స్వయంచాలక zp_aux = (dst->scales[j * nrows_interleaved + i]) & 0xF0;

                    dst->scales[j * nrows_interleaved + i] = (src_ఖండం->scales[j] & 0x0F) | zp_aux;
                }

                // zp for [N, 16]
                కొరకు (పూర్ణాంకం j = 0; j < 16; j++) {
                    స్వయంచాలక scale_aux = (dst->scales[16 * i + j]) & 0x0F;

                    dst->scales[16 * i + j] = (src_ఖండం->scales[j] & 0xF0) | scale_aux;
                }

                కొరకు (పూర్ణాంకం k = 0; k < 4; k++) {
                    కొరకు (పూర్ణాంకం j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j] = (src_ఖండం->qs[j] >> (2 * k)) & 0x03;
                    }
                }

                కొరకు (పూర్ణాంకం k = 0; k < 4; k++) {
                    కొరకు (పూర్ణాంకం j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j + 128] = (src_ఖండం->qs[j + 32] >> (2 * k)) & 0x03;
                    }
                }

                // from nrows_interleaved * [2 * 32byte]
                // to 4 * [nrows_interleaved * 16byte]
                కొరకు (పూర్ణాంకం k = 0; k < 4; k++) {
                    కొరకు (పూర్ణాంకం j = 0; j < 16; j++) {
                        uint8_t qs0  = qs_aux[j + k * 64];
                        uint8_t qs16 = qs_aux[j + 16 + k * 64];
                        uint8_t qs32 = qs_aux[j + 32 + k * 64];
                        uint8_t qs48 = qs_aux[j + 48 + k * 64];

                        dst->qs[(k * nrows_interleaved + i) * 16 + j] =
                            (qs0 & 0x03) | ((qs16 & 0x03) << 2) | ((qs32 & 0x03) << 4) | ((qs48 & 0x03) << 6);
                    }
                }

                dst->scales16[i] = src_ఖండం->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
                dst->zeros16[i]  = src_ఖండం->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
            }
            dst++;
        }
    }

    తిరిగి 0;
}

స్థిర పూర్ణాంకం repack_q3_k_గమ్యం_q3_k_32_bl(ggml_tensor *              t,
                                     పూర్ణాంకం                        interleave_ఖండం,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                     size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q3_K);
    GGML_ASSERT(interleave_ఖండం == 32);
    GGML_ASSERT(QK_K == 256);

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    స్థిరరాశి uint32_t kmask1 = 0x03030303;
    స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;

    స్థిరరాశి ఖండం_q3_K * src = (స్థిరరాశి ఖండం_q3_K *) దత్తాంశం;

    స్వయంచాలక * dst = (spacemit_kernels::nrow_ఖండం_q3_k<32> *) t->దత్తాంశం;

    పూర్ణాంకం nrow    = ggml_nrows(t);
    పూర్ణాంకం nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q3_K));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        తిరిగి -1;
    }

    uint32_t b_scale_aux[4] = { 0 };
    uint8_t  qs_aux[256]    = { 0 };

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                స్థిరరాశి ఖండం_q3_K * src_ఖండం = &src[(b + i) * nblocks + x];

                uint32_t * auxs  = b_scale_aux;
                int8_t *   scale = (int8_t *) auxs;
                జ్ఞాపకం_నకలు(auxs, src_ఖండం->scales, 12);

                uint32_t tmp = auxs[2];
                auxs[2]      = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
                auxs[3]      = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
                auxs[0]      = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
                auxs[1]      = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);

                కొరకు (పూర్ణాంకం j = 0; j < 16; j++) {
                    dst->scales[j * nrows_interleaved + i] = scale[j] - 32;
                }

                కొరకు (పూర్ణాంకం k = 0; k < 4; k++) {
                    కొరకు (పూర్ణాంకం j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j] = (src_ఖండం->qs[j] >> (2 * k)) & 0x03;
                    }
                }

                కొరకు (పూర్ణాంకం k = 0; k < 4; k++) {
                    కొరకు (పూర్ణాంకం j = 0; j < 32; j++) {
                        qs_aux[k * 32 + j + 128] = (src_ఖండం->qs[j + 32] >> (2 * k)) & 0x03;
                    }
                }

                // from nrows_interleaved * [2 * 32byte]
                // to 4 * [nrows_interleaved * 16byte]
                కొరకు (పూర్ణాంకం k = 0; k < 4; k++) {
                    కొరకు (పూర్ణాంకం j = 0; j < 16; j++) {
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
                uint16_t * dst_ముసుగు = ((uint16_t *) dst->hmask) + i;
                కొరకు (పూర్ణాంకం j = 0; j < 16; j++, dst_ముసుగు += nrows_interleaved) {
                    uint8_t   b_జరుపు    = j / 2;
                    uint8_t * b_ముసుగు_నిలువు = (uint8_t *) (src_ఖండం->hmask + (j % 2) * 16);
                    // b0 - b15
                    uint16_t  msk_బహిర్గతం_0  = 0;

                    కొరకు (పూర్ణాంకం k = 0; k < 8; k++) {
                        msk_బహిర్గతం_0 |= (uint16_t) ((b_ముసుగు_నిలువు[k] >> b_జరుపు) & 0x01) << k;
                    }
                    కొరకు (పూర్ణాంకం k = 8; k < 16; k++) {
                        msk_బహిర్గతం_0 |= (uint16_t) ((b_ముసుగు_నిలువు[k] >> b_జరుపు) & 0x01) << k;
                    }

                    dst_ముసుగు[0] = msk_బహిర్గతం_0;
                }

                dst->scales16[i] = src_ఖండం->d;
            }

            dst++;
        }
    }

    తిరిగి 0;
}

స్థిర పూర్ణాంకం repack_q4_0_గమ్యం_q4_0_32_bl_ref(ggml_tensor *              t,
                                         పూర్ణాంకం                        interleave_ఖండం,
                                         స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                         size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ఖండం == 32);  //వాడని

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    ఖండం_q4_0x32 *    dst = (ఖండం_q4_0x32 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_0 * src = (స్థిరరాశి ఖండం_q4_0 *) దత్తాంశం;
    ఖండం_q4_0         dst_tmp[32];
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q4_0));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q4_0x32(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q4_0_గమ్యం_q4_0_256_32_bl_ref(ggml_tensor *              t,
                                             పూర్ణాంకం                        interleave_ఖండం,
                                             స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                             size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ఖండం == 32);  //వాడని

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    ఖండం_q4_0x32x256 * dst = (ఖండం_q4_0x32x256 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_0 *  src = (స్థిరరాశి ఖండం_q4_0 *) దత్తాంశం;
    ఖండం_q4_0          dst_tmp[32];
    పూర్ణాంకం                 nrow    = ggml_nrows(t);
    పూర్ణాంకం                 nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q4_0));
    GGML_ASSERT(nblocks % 8 == 0);  // for 256-block interleaving
    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x += 8) {
            కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
                కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                    dst_tmp[i] = src[x + j + i * nblocks];
                }
                dst->ఖండాలు[j] = make_ఖండం_q4_0x32(dst_tmp, interleave_ఖండం);
            }
            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q4_0_గమ్యం_q4_1_256_32_bl_ref(ggml_tensor *              t,
                                             పూర్ణాంకం                        interleave_ఖండం,
                                             స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                             size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_ఖండం == 32);  //వాడని

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    ఖండం_q4_1x32x256 * dst = (ఖండం_q4_1x32x256 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_1 *  src = (స్థిరరాశి ఖండం_q4_1 *) దత్తాంశం;
    ఖండం_q4_1          dst_tmp[32];
    పూర్ణాంకం                 nrow    = ggml_nrows(t);
    పూర్ణాంకం                 nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q4_1));
    GGML_ASSERT(nblocks % 8 == 0);  // for 256-block interleaving
    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x += 8) {
            కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
                కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                    dst_tmp[i] = src[x + j + i * nblocks];
                }

                ఖండం_q4_0x32 * dst_ఖండం = &dst->ఖండాలు[j];
                uint8_t *       dst_zp    = dst->zps + j * nrows_interleaved;

                కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                    దశాంశం d   = GGML_FP16_గమ్యం_FP32(dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    దశాంశం m   = GGML_FP16_గమ్యం_FP32(dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);
                    దశాంశం mid = -std::nearbyintf(m / d);
                    mid       = std::min(15.0f, std::max(0.0f, mid));

                    dst_ఖండం->d[i] = GGML_FP32_గమ్యం_FP16(d);
                    dst_zp[i]       = static_cast<uint8_t>(mid);
                }

                కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                    కొరకు (పూర్ణాంకం k = 0; k < QK4_1 / 4; k++) {
                        dst_ఖండం->qs[i * QK4_1 / 2 + k] =
                            (dst_tmp[i].qs[k * 2] & 0x0F) | ((dst_tmp[i].qs[k * 2 + 1] & 0x0F) << 4);
                    }
                }

                కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                    కొరకు (పూర్ణాంకం k = 0; k < QK4_1 / 4; k++) {
                        dst_ఖండం->qs[i * QK4_1 / 2 + QK4_1 / 4 + k] =
                            ((dst_tmp[i].qs[k * 2] & 0xF0) >> 4) | (dst_tmp[i].qs[k * 2 + 1] & 0xF0);
                    }
                }
            }
            dst++;
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

// RVV optimized version of repack_q4_0_to_q4_0_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes nibble repack.
స్థిర పూర్ణాంకం repack_q4_0_గమ్యం_q4_0_32_bl(ggml_tensor *              t,
                                     పూర్ణాంకం                        interleave_ఖండం,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                     size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ఖండం == 32);

    constexpr పూర్ణాంకం nrows_interleaved = 32;
    constexpr పూర్ణాంకం qs_బైట్లు          = QK4_0 / 2;  // 16

    ఖండం_q4_0x32 *    dst     = (ఖండం_q4_0x32 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_0 * src     = (స్థిరరాశి ఖండం_q4_0 *) దత్తాంశం;
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q4_0));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_0 != 0) {
        తిరిగి -1;
    }

    స్థిరరాశి ptrdiff_t అడ్డువరుస_stride = (ptrdiff_t) nblocks * పరిమాణం(ఖండం_q4_0);

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            స్థిరరాశి ఖండం_q4_0 * నిలువు_src = src + x;

            // --- 1) Gather 32 scale values (ggml_half d) with stride load ---
            // d is at offset 0 of each block_q4_0, stride between rows = row_stride
            {
                స్థిరరాశి uint8_t * d_పునాది    = (స్థిరరాశి uint8_t *) &నిలువు_src->d;
                ggml_half *     d_dst     = dst->d;
                size_t          మిగిలిన = 32;
                size_t          స్థానభ్రంశం    = 0;
                అయితే (మిగిలిన > 0) {
                    size_t      vl = __riscv_vsetvl_e16m1(మిగిలిన);
                    vuint16m1_t vd =
                        __riscv_vlse16_v_u16m1((స్థిరరాశి uint16_t *) (d_పునాది + స్థానభ్రంశం * అడ్డువరుస_stride), అడ్డువరుస_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + స్థానభ్రంశం), vd, vl);
                    స్థానభ్రంశం += vl;
                    మిగిలిన -= vl;
                }
            }

            // --- 2) Nibble repack qs for each of the 32 rows ---
            // For each row i:
            //   src qs[16]: [b0|b16] [b1|b17] ... [b15|b31]  (lo nibble = b_j, hi nibble = b_{j+16})
            //   dst qs low  8B: (qs[2j] & 0x0F) | ((qs[2j+1] & 0x0F) << 4)  for j=0..7
            //   dst qs high 8B: ((qs[2j] >> 4))  | (qs[2j+1] & 0xF0)         for j=0..7
            {
                స్థిరరాశి size_t vl8 = __riscv_vsetvl_e8m1(8);
                కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
                    స్థిరరాశి uint8_t * sq = నిలువు_src[i * nblocks].qs;
                    uint8_t *       dq = dst->qs + i * qs_బైట్లు;

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
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q4_1_గమ్యం_q4_1_32_bl_ref(ggml_tensor *              t,
                                         పూర్ణాంకం                        interleave_ఖండం,
                                         స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                         size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_ఖండం == 32);  //వాడని

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    ఖండం_q4_1x32 *    dst = (ఖండం_q4_1x32 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_1 * src = (స్థిరరాశి ఖండం_q4_1 *) దత్తాంశం;
    ఖండం_q4_1         dst_tmp[32];
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q4_1));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q4_1x32(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

// RVV optimized version of repack_q4_1_to_q4_1_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes nibble repack + zp computation.
స్థిర పూర్ణాంకం repack_q4_1_గమ్యం_q4_1_32_bl(ggml_tensor *              t,
                                     పూర్ణాంకం                        interleave_ఖండం,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                     size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_1);
    GGML_ASSERT(interleave_ఖండం == 32);

    constexpr పూర్ణాంకం nrows_interleaved = 32;
    constexpr పూర్ణాంకం qs_బైట్లు          = QK4_1 / 2;  // 16

    ఖండం_q4_1x32 *    dst     = (ఖండం_q4_1x32 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_1 * src     = (స్థిరరాశి ఖండం_q4_1 *) దత్తాంశం;
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK4_1;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q4_1));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK4_1 != 0) {
        తిరిగి -1;
    }

    స్థిరరాశి ptrdiff_t అడ్డువరుస_stride = (ptrdiff_t) nblocks * పరిమాణం(ఖండం_q4_1);

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            స్థిరరాశి ఖండం_q4_1 * నిలువు_src = src + x;

            // --- 1) Gather d and m, compute zp = clamp(nearbyint(-m/d), 0, 15) ---
            // block_q4_1 layout: [d(f16), m(f16), qs[16]]
            // d is at byte offset 0, m is at byte offset 2 from each block start
            {
                స్థిరరాశి uint8_t * dm_పునాది   = (స్థిరరాశి uint8_t *) &నిలువు_src->GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
                ggml_half *     d_dst     = dst->d;
                uint8_t *       zp_dst    = dst->zp;
                size_t          మిగిలిన = 32;
                size_t          స్థానభ్రంశం    = 0;
                అయితే (మిగిలిన > 0) {
                    size_t vl = __riscv_vsetvl_e16m1(మిగిలిన);

                    // stride load d (f16) from each row
                    vuint16m1_t vd_raw =
                        __riscv_vlse16_v_u16m1((స్థిరరాశి uint16_t *) (dm_పునాది + స్థానభ్రంశం * అడ్డువరుస_stride), అడ్డువరుస_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + స్థానభ్రంశం), vd_raw, vl);

                    // stride load m (f16) from each row (offset +2 bytes from d)
                    vuint16m1_t vm_raw =
                        __riscv_vlse16_v_u16m1((స్థిరరాశి uint16_t *) (dm_పునాది + 2 + స్థానభ్రంశం * అడ్డువరుస_stride), అడ్డువరుస_stride, vl);

                    // convert to f32 for zp computation: zp = nearbyint(-m / d)
                    vfloat16m1_t vd_f16 = __riscv_vreinterpret_v_u16m1_f16m1(vd_raw);
                    vfloat16m1_t vm_f16 = __riscv_vreinterpret_v_u16m1_f16m1(vm_raw);

                    // -m / d in f16 directly (SpaceMIT X60 supports f16 arithmetic)
                    vfloat16m1_t v_neg_m = __riscv_vfneg_v_f16m1(vm_f16, vl);
                    vfloat16m1_t v_నిష్పత్తి = __riscv_vfdiv_vv_f16m1(v_neg_m, vd_f16, vl);

                    // Convert to f32 for nearbyint, then clamp
                    vfloat32m2_t v_నిష్పత్తి_f32 = __riscv_vfwcvt_f_f_v_f32m2(v_నిష్పత్తి, vl);

                    // Use integer rounding: convert f32 -> int (rounds to nearest)
                    vint32m2_t v_zp_i32 = __riscv_vfcvt_x_f_v_i32m2(v_నిష్పత్తి_f32, vl);

                    // clamp to [0, 15]
                    v_zp_i32 = __riscv_vmax_vx_i32m2(v_zp_i32, 0, vl);
                    v_zp_i32 = __riscv_vmin_vx_i32m2(v_zp_i32, 15, vl);

                    // narrow i32 -> u8
                    vint16m1_t  v_zp_i16 = __riscv_vncvt_x_x_w_i16m1(v_zp_i32, vl);
                    vint8mf2_t  v_zp_i8  = __riscv_vncvt_x_x_w_i8mf2(v_zp_i16, vl);
                    vuint8mf2_t v_zp_u8  = __riscv_vreinterpret_v_i8mf2_u8mf2(v_zp_i8);
                    __riscv_vse8_v_u8mf2(zp_dst + స్థానభ్రంశం, v_zp_u8, vl);

                    స్థానభ్రంశం += vl;
                    మిగిలిన -= vl;
                }
            }

            // --- 2) Nibble repack qs for each of the 32 rows ---
            {
                స్థిరరాశి size_t vl8 = __riscv_vsetvl_e8m1(8);
                కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
                    స్థిరరాశి uint8_t * sq = నిలువు_src[i * nblocks].qs;
                    uint8_t *       dq = dst->qs + i * qs_బైట్లు;

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
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q4_k_గమ్యం_q4_1_32_bl(ggml_tensor *              t,
                                     పూర్ణాంకం                        interleave_ఖండం,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                     size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_ఖండం == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    ఖండం_q4_1x32 *    dst = (ఖండం_q4_1x32 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_K * src = (స్థిరరాశి ఖండం_q4_K *) దత్తాంశం;
    ఖండం_q4_1         dst_tmp[32];
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK_K;

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
                కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    స్థిరరాశి దశాంశం d = GGML_FP16_గమ్యం_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    స్థిరరాశి దశాంశం min =
                        GGML_FP16_గమ్యం_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);
                    స్థిరరాశి దశాంశం d1 = d * sc;
                    స్థిరరాశి దశాంశం m1 = min * m;

                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d = GGML_FP32_గమ్యం_FP16(d1);
                    dst_tmp[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m = GGML_FP32_గమ్యం_FP16(-m1);
                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    స్థిరరాశి uint8_t * q                                  = src[x + i * nblocks].qs + (j / 2) * QK4_1;
                    ఒకవేళ (j % 2 == 0) {
                        కొరకు (పూర్ణాంకం ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } లేకపోతే {
                        కొరకు (పూర్ణాంకం ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = ((q[ii] & 0xF0) >> 4) | (q[ii + 16] & 0xF0);
                        }
                    }
                }
                *dst++ = make_ఖండం_q4_1x32(dst_tmp, interleave_ఖండం);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q6_k_గమ్యం_q8_0_32_bl_ref(ggml_tensor *              t,
                                         పూర్ణాంకం                        interleave_ఖండం,
                                         స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                         size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_ఖండం == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    ఖండం_q8_0x32 *    dst = (ఖండం_q8_0x32 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q6_K * src = (స్థిరరాశి ఖండం_q6_K *) దత్తాంశం;
    ఖండం_q8_0         dst_tmp[32];
    int8_t             aux8[QK4_1];
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK_K;

    ఒకవేళ (t->ne[0] % QK_K != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        int64_t nrow_real = std::min((int64_t) nrow - b, (int64_t) nrows_interleaved);
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం bi = 0; bi < 8; bi++) {
                పూర్ణాంకం i = 0;
                కొరకు (; i < nrow_real; i++) {
                    స్థిరరాశి uint8_t * q4     = src[x + i * nblocks].ql;
                    స్థిరరాశి uint8_t * qh     = src[x + i * nblocks].qh;
                    స్థిరరాశి int8_t *  scales = src[x + i * nblocks].scales;
                    దశాంశం           d      = GGML_FP16_గమ్యం_FP32(src[x + i * nblocks].d);

                    q4 += 64 * (bi / 4);
                    qh += 32 * (bi / 4);
                    int8_t * GGML_RESTRICT a = aux8;

                    int8_t bi_idx = bi % 4;

                    ఒకవేళ (bi_idx == 0) {
                        కొరకు (పూర్ణాంకం l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l] & 0xF) | (((qh[l] >> 0) & 3) << 4)) - 32;
                        }
                    } లేకపోతే ఒకవేళ (bi_idx == 1) {
                        కొరకు (పూర్ణాంకం l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 32] & 0xF) | (((qh[l] >> 2) & 3) << 4)) - 32;
                        }
                    } లేకపోతే ఒకవేళ (bi_idx == 2) {
                        కొరకు (పూర్ణాంకం l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 0] >> 4) | (((qh[l] >> 4) & 3) << 4)) - 32;
                        }
                    } లేకపోతే ఒకవేళ (bi_idx == 3) {
                        కొరకు (పూర్ణాంకం l = 0; l < 32; ++l) {
                            a[l] = (int8_t) ((q4[l + 32] >> 4) | (((qh[l] >> 6) & 3) << 4)) - 32;
                        }
                    }
                    a = aux8;

                    దశాంశం a_max_abs = 0.0f;
                    దశాంశం scale_0   = scales[bi * 2 + 0] * d;
                    దశాంశం scale_1   = scales[bi * 2 + 1] * d;
                    కొరకు (పూర్ణాంకం l = 0; l < 16; ++l) {
                        a_max_abs = std::max(a_max_abs, std::నిరపేక్ష(a[l] * scale_0));
                    }

                    కొరకు (పూర్ణాంకం l = 16; l < 32; ++l) {
                        a_max_abs = std::max(a_max_abs, std::నిరపేక్ష(a[l] * scale_1));
                    }

                    దశాంశం reflect_scale   = a_max_abs / ((1 << 7) - 1);
                    దశాంశం reflect_scale_0 = scale_0 / reflect_scale;
                    దశాంశం reflect_scale_1 = scale_1 / reflect_scale;

                    కొరకు (పూర్ణాంకం l = 0; l < 16; ++l) {
                        దశాంశం a_తాత్కాలిక = std::clamp(std::nearbyintf(a[l] * reflect_scale_0), -128.0f, 127.0f);
                        a[l]         = (int8_t) (a_తాత్కాలిక);
                    }

                    కొరకు (పూర్ణాంకం l = 16; l < 32; ++l) {
                        దశాంశం a_తాత్కాలిక = std::clamp(std::nearbyintf(a[l] * reflect_scale_1), -128.0f, 127.0f);
                        a[l]         = (int8_t) (a_తాత్కాలిక);
                    }

                    dst_tmp[i].d = GGML_FP32_గమ్యం_FP16(reflect_scale);

                    జ్ఞాపకం_నకలు(dst_tmp[i].qs, a, 32 * పరిమాణం(int8_t));
                }

                కొరకు (; i < nrows_interleaved; i++) {
                    memset(&dst_tmp[i], 0, పరిమాణం(ఖండం_q8_0));
                }

                *dst++ = make_ఖండం_q8_0x32(dst_tmp, interleave_ఖండం);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

// RVV optimized version of repack_q6_k_to_q8_0_32_bl
// Vectorizes the Q6_K dequant -> requant pipeline using RVV intrinsics.
// For each sub-block (bi), dequant 32 Q6_K values to int6 -> apply two sub-block scales ->
// find max abs -> compute reflect_scale -> requant to int8 -> gather d with stride load.
స్థిర పూర్ణాంకం repack_q6_k_గమ్యం_q8_0_32_bl(ggml_tensor *              t,
                                     పూర్ణాంకం                        interleave_ఖండం,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                     size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_ఖండం == 32);
    GGML_ASSERT(QK_K / QK4_1 == 8);

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    ఖండం_q8_0x32 *    dst     = (ఖండం_q8_0x32 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q6_K * src     = (స్థిరరాశి ఖండం_q6_K *) దత్తాంశం;
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK_K;

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        తిరిగి -1;
    }

    స్థిరరాశి ptrdiff_t అడ్డువరుస_stride = (ptrdiff_t) nblocks * పరిమాణం(ఖండం_q6_K);

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం bi = 0; bi < 8; bi++) {
                // --- 1) Gather 32 d values with stride load ---
                // We need to compute reflect_scale per row first, so gather d later.
                // Process each row: dequant Q6_K sub-block -> requant to Q8_0
                కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                    స్థిరరాశి ఖండం_q6_K * src_blk = &src[x + i * nblocks];
                    స్థిరరాశి uint8_t *    q4      = src_blk->ql + 64 * (bi / 4);
                    స్థిరరాశి uint8_t *    qh      = src_blk->qh + 32 * (bi / 4);
                    స్థిరరాశి int8_t *     scales  = src_blk->scales;
                    దశాంశం              d       = GGML_FP16_గమ్యం_FP32(src_blk->d);

                    int8_t bi_idx = bi % 4;

                    // --- Dequant 32 Q6_K values to int6 (range [-32, 31]) using RVV ---
                    // vl = 32 for e8m2 (VLEN=256) or loop for smaller VLEN
                    స్థిరరాశి size_t vl16 = __riscv_vsetvl_e8m1(16);

                    vint8m1_t va_lo, va_hi;  // 16 elements each

                    ఒకవేళ (bi_idx == 0) {
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
                    } లేకపోతే ఒకవేళ (bi_idx == 1) {
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
                    } లేకపోతే ఒకవేళ (bi_idx == 2) {
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
                    } లేకపోతే {  // bi_idx == 3
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
                    దశాంశం scale_0 = scales[bi * 2 + 0] * d;
                    దశాంశం scale_1 = scales[bi * 2 + 1] * d;

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
                    దశాంశం        a_max_abs = __riscv_vfmv_f_s_f32m1_f32(vmax_red);

                    దశాంశం reflect_scale   = a_max_abs / 127.0f;
                    దశాంశం reflect_scale_0 = scale_0 / reflect_scale;
                    దశాంశం reflect_scale_1 = scale_1 / reflect_scale;

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
                    dst->d[i]   = GGML_FP32_గమ్యం_FP16(reflect_scale);
                    int8_t * dq = (int8_t *) dst->qs + i * QK8_0;
                    __riscv_vse8_v_i8m1(dq, vi8_lo, vl16);
                    __riscv_vse8_v_i8m1(dq + 16, vi8_hi, vl16);
                }
                dst++;
            }
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q8_0_గమ్యం_q8_0_32_bl_ref(ggml_tensor *              t,
                                         పూర్ణాంకం                        interleave_ఖండం,
                                         స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                         size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_ఖండం == 32);  //వాడని

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    ఖండం_q8_0x32 *    dst = (ఖండం_q8_0x32 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q8_0 * src = (స్థిరరాశి ఖండం_q8_0 *) దత్తాంశం;
    ఖండం_q8_0         dst_tmp[32];
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q8_0));

    ఒకవేళ (t->ne[0] % QK8_0 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        int64_t nrows_real = std::min((int64_t) nrow - b, (int64_t) nrows_interleaved);
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            పూర్ణాంకం i = 0;
            కొరకు (; i < nrows_real; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            కొరకు (; i < nrows_interleaved; i++) {
                memset(&dst_tmp[i], 0, పరిమాణం(ఖండం_q8_0));
            }
            *dst++ = make_ఖండం_q8_0x32(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

// RVV optimized version of repack_q8_0_to_q8_0_32_bl
// Eliminates the intermediate dst_tmp buffer and vectorizes scale gather + qs copy.
స్థిర పూర్ణాంకం repack_q8_0_గమ్యం_q8_0_32_bl(ggml_tensor *              t,
                                     పూర్ణాంకం                        interleave_ఖండం,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                     size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_ఖండం == 32);

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    ఖండం_q8_0x32 *    dst     = (ఖండం_q8_0x32 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q8_0 * src     = (స్థిరరాశి ఖండం_q8_0 *) దత్తాంశం;
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q8_0));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK8_0 != 0) {
        తిరిగి -1;
    }

    స్థిరరాశి ptrdiff_t అడ్డువరుస_stride = (ptrdiff_t) nblocks * పరిమాణం(ఖండం_q8_0);

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            స్థిరరాశి ఖండం_q8_0 * నిలువు_src = src + x;

            // --- 1) Gather 32 scale values (ggml_half d) with stride load ---
            {
                స్థిరరాశి uint8_t * d_పునాది    = (స్థిరరాశి uint8_t *) &నిలువు_src->d;
                ggml_half *     d_dst     = dst->d;
                size_t          మిగిలిన = 32;
                size_t          స్థానభ్రంశం    = 0;
                అయితే (మిగిలిన > 0) {
                    size_t      vl = __riscv_vsetvl_e16m1(మిగిలిన);
                    vuint16m1_t vd =
                        __riscv_vlse16_v_u16m1((స్థిరరాశి uint16_t *) (d_పునాది + స్థానభ్రంశం * అడ్డువరుస_stride), అడ్డువరుస_stride, vl);
                    __riscv_vse16_v_u16m1((uint16_t *) (d_dst + స్థానభ్రంశం), vd, vl);
                    స్థానభ్రంశం += vl;
                    మిగిలిన -= vl;
                }
            }

            // --- 2) Copy qs for each of the 32 rows (32 bytes per row) ---
            {
                కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
                    స్థిరరాశి int8_t * sq = నిలువు_src[i * nblocks].qs;
                    int8_t *       dq = (int8_t *) dst->qs + i * QK8_0;

                    size_t len = QK8_0;
                    size_t idx = 0;
                    అయితే (len > 0) {
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
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర శూన్య convert_mxfp4_గమ్యం_5bit(స్థిరరాశి ఖండం_mxfp4 & src, spacemit_kernels::nrow_ఖండం_mxfp4<1> & dst) {
    dst.e[0] = src.e;

    // Decode all 32 mxfp4 values to signed integers via kvalues_mxfp4
    int8_t vals[32];
    కొరకు (పూర్ణాంకం j = 0; j < QK_MXFP4 / 2; j++) {
        vals[j]                = kvalues_mxfp4[src.qs[j] & 0xF];
        vals[j + QK_MXFP4 / 2] = kvalues_mxfp4[src.qs[j] >> 4];
    }

    // vals [b0, b1, b2, b3, ..., b30, b31]
    // Pack abs into qs with reorder: [b0,b1]..[b14,b15]..[b30,b31]
    కొరకు (పూర్ణాంకం j = 0; j < QK_MXFP4 / 2; j++) {
        uint8_t lo0 = static_cast<uint8_t>(std::నిరపేక్ష(vals[j * 2]));
        uint8_t lo1 = static_cast<uint8_t>(std::నిరపేక్ష(vals[j * 2 + 1]));
        dst.qs[j]   = (lo0 & 0x0F) | ((lo1 & 0x0F) << 4);
    }

    // Pack sign bits into qh[4] (32 bits total, 1 bit per weight)
    // reorder: [0,1,2,...,15,16,17,...,31] after the qs reorder above
    uint32_t sign_బిట్లు = 0;
    కొరకు (పూర్ణాంకం j = 0; j < 32; j++) {
        ఒకవేళ (vals[j] < 0) {
            sign_బిట్లు |= (1u << j);
        }
    }
    జ్ఞాపకం_నకలు(dst.qh, &sign_బిట్లు, 4);
}

స్థిర spacemit_kernels::nrow_ఖండం_mxfp4<32> make_ఖండం_mxfp4x32(spacemit_kernels::nrow_ఖండం_mxfp4<1> * లోపల,
                                                                  గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    spacemit_kernels::nrow_ఖండం_mxfp4<32> బహిర్గతం;
    GGML_ASSERT(QK_MXFP4 / blck_కొలత_interleave == 1);
    GGML_వాడని(blck_కొలత_interleave);

    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        బహిర్గతం.e[i] = లోపల[i].e[0];
    }

    // qs: copy per-row 16 bytes
    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        జ్ఞాపకం_నకలు(బహిర్గతం.qs + i * 16, లోపల[i].qs, 16);
    }

    // qh: copy per-row 4 bytes
    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        జ్ఞాపకం_నకలు(బహిర్గతం.qh + i * 4, లోపల[i].qh, 4);
    }

    తిరిగి బహిర్గతం;
}

స్థిర పూర్ణాంకం repack_mxfp4_గమ్యం_mxfp4_32_bl(ggml_tensor *              t,
                                       పూర్ణాంకం                        interleave_ఖండం,
                                       స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                       size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_ఖండం == 32);

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    spacemit_kernels::nrow_ఖండం_mxfp4<32> * dst = (spacemit_kernels::nrow_ఖండం_mxfp4<32> *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_mxfp4 *                      src = (స్థిరరాశి ఖండం_mxfp4 *) దత్తాంశం;
    spacemit_kernels::nrow_ఖండం_mxfp4<1>    dst_tmp[32];
    పూర్ణాంకం                                      nrow    = ggml_nrows(t);
    పూర్ణాంకం                                      nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_mxfp4));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_MXFP4 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                convert_mxfp4_గమ్యం_5bit(src[x + i * nblocks], dst_tmp[i]);
            }
            *dst++ = make_ఖండం_mxfp4x32(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;
}

స్థిర spacemit_kernels::nrow_ఖండం_q5_1<32> make_ఖండం_q5_1x32(spacemit_kernels::nrow_ఖండం_q5_1<1> * లోపల,
                                                                గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    spacemit_kernels::nrow_ఖండం_q5_1<32> బహిర్గతం;
    GGML_ASSERT(QK5_1 / blck_కొలత_interleave == 1);
    GGML_వాడని(blck_కొలత_interleave);

    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        బహిర్గతం.scales16[i] = లోపల[i].scales16[0];
        బహిర్గతం.zp[i]       = లోపల[i].zp[0];
    }

    // qs: low 4 bits, reorder from [b0,b16],[b1,b17]... to [b0,b1]...[b14,b15] and [b16,b17]...[b30,b31]
    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        // low half [0..15]
        కొరకు (పూర్ణాంకం j = 0; j < QK5_1 / 4; j++) {
            బహిర్గతం.qs[i * QK5_1 / 2 + j] = (లోపల[i].qs[j * 2] & 0x0F) | ((లోపల[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
        // high half [16..31]
        కొరకు (పూర్ణాంకం j = 0; j < QK5_1 / 4; j++) {
            బహిర్గతం.qs[i * QK5_1 / 2 + QK5_1 / 4 + j] = ((లోపల[i].qs[j * 2] & 0xF0) >> 4) | (లోపల[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    // qh: 5th bit, copy directly
    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        కొరకు (పూర్ణాంకం j = 0; j < 4; j++) {
            బహిర్గతం.qh[i * 4 + j] = లోపల[i].qh[j];
        }
    }

    తిరిగి బహిర్గతం;
}

స్థిర spacemit_kernels::nrow_ఖండం_q5_0<32> make_ఖండం_q5_0x32(spacemit_kernels::nrow_ఖండం_q5_0<1> * లోపల,
                                                                గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    spacemit_kernels::nrow_ఖండం_q5_0<32> బహిర్గతం;
    GGML_ASSERT(QK5_0 / blck_కొలత_interleave == 1);
    GGML_వాడని(blck_కొలత_interleave);

    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        బహిర్గతం.scales16[i] = లోపల[i].scales16[0];
    }

    // qs: low 4 bits, reorder from [b0,b16],[b1,b17]... to [b0,b1]...[b14,b15] and [b16,b17]...[b30,b31]
    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        // low half [0..15]
        కొరకు (పూర్ణాంకం j = 0; j < QK5_0 / 4; j++) {
            బహిర్గతం.qs[i * QK5_0 / 2 + j] = (లోపల[i].qs[j * 2] & 0x0F) | ((లోపల[i].qs[j * 2 + 1] & 0x0F) << 4);
        }
        // high half [16..31]
        కొరకు (పూర్ణాంకం j = 0; j < QK5_0 / 4; j++) {
            బహిర్గతం.qs[i * QK5_0 / 2 + QK5_0 / 4 + j] = ((లోపల[i].qs[j * 2] & 0xF0) >> 4) | (లోపల[i].qs[j * 2 + 1] & 0xF0);
        }
    }

    // qh: 5th bit, copy directly
    కొరకు (పూర్ణాంకం i = 0; i < 32; i++) {
        కొరకు (పూర్ణాంకం j = 0; j < 4; j++) {
            బహిర్గతం.qh[i * 4 + j] = లోపల[i].qh[j];
        }
    }

    తిరిగి బహిర్గతం;
}

స్థిర పూర్ణాంకం repack_q5_0_గమ్యం_q5_0_32_bl(ggml_tensor *              t,
                                     పూర్ణాంకం                        interleave_ఖండం,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                     size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_0);
    GGML_ASSERT(interleave_ఖండం == 32);  //వాడని

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    spacemit_kernels::nrow_ఖండం_q5_0<32> * dst = (spacemit_kernels::nrow_ఖండం_q5_0<32> *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q5_0 *                      src = (స్థిరరాశి ఖండం_q5_0 *) దత్తాంశం;
    spacemit_kernels::nrow_ఖండం_q5_0<1>    dst_tmp[32];
    పూర్ణాంకం                                     nrow    = ggml_nrows(t);
    పూర్ణాంకం                                     nblocks = t->ne[0] / QK5_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q5_0));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK5_0 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                స్థిరరాశి ఖండం_q5_0 & s = src[x + i * nblocks];

                dst_tmp[i].scales16[0] = s.d;
                జ్ఞాపకం_నకలు(dst_tmp[i].qs, s.qs, పరిమాణం(dst_tmp[i].qs));
                జ్ఞాపకం_నకలు(dst_tmp[i].qh, s.qh, పరిమాణం(dst_tmp[i].qh));
            }
            *dst++ = make_ఖండం_q5_0x32(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;
}

స్థిర పూర్ణాంకం repack_q5_1_గమ్యం_q5_1_32_bl(ggml_tensor *              t,
                                     పూర్ణాంకం                        interleave_ఖండం,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                     size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_1);
    GGML_ASSERT(interleave_ఖండం == 32);  //వాడని

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    spacemit_kernels::nrow_ఖండం_q5_1<32> * dst = (spacemit_kernels::nrow_ఖండం_q5_1<32> *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q5_1 *                      src = (స్థిరరాశి ఖండం_q5_1 *) దత్తాంశం;
    spacemit_kernels::nrow_ఖండం_q5_1<1>    dst_tmp[32];
    పూర్ణాంకం                                     nrow    = ggml_nrows(t);
    పూర్ణాంకం                                     nblocks = t->ne[0] / QK5_1;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q5_1));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK5_1 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                స్థిరరాశి ఖండం_q5_1 & s = src[x + i * nblocks];

                దశాంశం d = GGML_FP16_గమ్యం_FP32(s.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                దశాంశం m = GGML_FP16_గమ్యం_FP32(s.GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.m);

                ఒకవేళ (d == 0.0f) {
                    dst_tmp[i].scales16[0] = GGML_FP32_గమ్యం_FP16(std::fabs(m));
                    dst_tmp[i].zp[0]       = m < 0.0f ? 1 : 0;
                    memset(dst_tmp[i].qh, 0, పరిమాణం(dst_tmp[i].qh));
                    memset(dst_tmp[i].qs, m > 0.0f ? 0x11 : 0x00, పరిమాణం(dst_tmp[i].qs));
                    కొనసాగు;
                }

                దశాంశం mid = std::nearbyintf(-m / d);
                mid       = std::min(31.0f, std::max(0.0f, mid));

                dst_tmp[i].scales16[0] = GGML_FP32_గమ్యం_FP16(d);
                dst_tmp[i].zp[0]       = static_cast<uint8_t>(mid);

                // qs: copy low 4 bits directly (same nibble packing)
                జ్ఞాపకం_నకలు(dst_tmp[i].qs, s.qs, QK5_1 / 2);

                // qh: copy 5th bit directly
                జ్ఞాపకం_నకలు(dst_tmp[i].qh, s.qh, 4);
            }
            *dst++ = make_ఖండం_q5_1x32(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;
}

స్థిర పూర్ణాంకం repack_q5_k_గమ్యం_q5_1_32_bl(ggml_tensor *              t,
                                     పూర్ణాంకం                        interleave_ఖండం,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                     size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_ఖండం == 32);
    GGML_ASSERT(QK_K / QK5_1 == 8);

    constexpr పూర్ణాంకం nrows_interleaved = 32;

    spacemit_kernels::nrow_ఖండం_q5_1<32> * dst = (spacemit_kernels::nrow_ఖండం_q5_1<32> *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q5_K *                      src = (స్థిరరాశి ఖండం_q5_K *) దత్తాంశం;
    spacemit_kernels::nrow_ఖండం_q5_1<1>    dst_tmp[32];
    పూర్ణాంకం                                     nrow    = ggml_nrows(t);
    పూర్ణాంకం                                     nblocks = t->ne[0] / QK_K;

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % QK_K != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
                కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                    uint8_t     sc, m;
                    స్థిరరాశి దశాంశం d = GGML_FP16_గమ్యం_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d);
                    స్థిరరాశి దశాంశం min =
                        GGML_FP16_గమ్యం_FP32(src[x + i * nblocks].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin);
                    get_scale_min_k4(j, src[x + i * nblocks].scales, &sc, &m);

                    దశాంశం d1 = d * sc;
                    దశాంశం m1 = min * m;

                    దశాంశం mid              = std::nearbyintf(m1 / d1);
                    mid                    = std::min(31.0f, std::max(0.0f, mid));
                    dst_tmp[i].scales16[0] = GGML_FP32_గమ్యం_FP16(d1);
                    dst_tmp[i].zp[0]       = static_cast<uint8_t>(mid);

                    // src -> [b0, b32] [b1, b33] ... [b31, b63]
                    // dst -> [b0, b16] [b1, b17] ... [b15, b31] [b32, b48] [b33, b49] ... [b47, b63]
                    స్థిరరాశి uint8_t * q = src[x + i * nblocks].qs + (j / 2) * QK5_1;
                    ఒకవేళ (j % 2 == 0) {
                        కొరకు (పూర్ణాంకం ii = 0; ii < 16; ii++) {
                            dst_tmp[i].qs[ii] = (q[ii] & 0x0F) | ((q[ii + 16] & 0x0F) << 4);
                        }
                    } లేకపోతే {
                        కొరకు (పూర్ణాంకం ii = 0; ii < 16; ii++) {
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
                    స్థిరరాశి uint8_t * src_qh = src[x + i * nblocks].qh;
                    కొరకు (పూర్ణాంకం bi = 0; bi < 4; bi++) {
                        uint8_t qh_byte = 0;
                        కొరకు (పూర్ణాంకం k = 0; k < 8; k++) {
                            పూర్ణాంకం src_idx = bi * 8 + k;
                            qh_byte |= ((src_qh[src_idx] >> j) & 1) << k;
                        }
                        dst_tmp[i].qh[bi] = qh_byte;
                    }
                }
                *dst++ = make_ఖండం_q5_1x32(dst_tmp, interleave_ఖండం);
            }
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;
}

నామాకాశ ggml::cpu::riscv64_spacemit {

మూస <రకం_పేరు BLOC_TYPE, int64_t INTER_కొలత, int64_t NB_నిలువుల_సంఖ్య> పూర్ణాంకం repack(ggml_tensor *, స్థిరరాశి శూన్య *, size_t);

మూస <> పూర్ణాంకం repack<ఖండం_q4_0, 32, 16>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q4_0_గమ్యం_q4_0_16_bl(t, 16, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q4_1, 32, 16>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q4_1_గమ్యం_q4_1_16_bl(t, 16, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q4_K, 32, 16>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q4_k_గమ్యం_q4_1_16_bl(t, 16, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q2_K, 256, 32>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q2_k_గమ్యం_q2_k_32_bl(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q3_K, 256, 32>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q3_k_గమ్యం_q3_k_32_bl(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q4_0, 32, 32>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
#ఒకవేళ 0
    తిరిగి repack_q4_0_గమ్యం_q4_0_32_bl_ref(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
#లేకపోతే
    తిరిగి repack_q4_0_గమ్యం_q4_0_32_bl(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
#అయితే_ముగింపు
}

మూస <> పూర్ణాంకం repack<ఖండం_q4_0, 256, 32>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
#ఒకవేళ 1
    తిరిగి repack_q4_0_గమ్యం_q4_0_256_32_bl_ref(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
#లేకపోతే
    //return repack_q4_0_to_q4_0_256_32_bl(t, 32, data, data_size);
#అయితే_ముగింపు
}

మూస <> పూర్ణాంకం repack<ఖండం_q4_1, 32, 32>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
#ఒకవేళ 0
    తిరిగి repack_q4_1_గమ్యం_q4_1_32_bl_ref(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
#లేకపోతే
    తిరిగి repack_q4_1_గమ్యం_q4_1_32_bl(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
#అయితే_ముగింపు
}

మూస <> పూర్ణాంకం repack<ఖండం_q4_1, 256, 32>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
#ఒకవేళ 1
    తిరిగి repack_q4_0_గమ్యం_q4_1_256_32_bl_ref(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
#లేకపోతే
    తిరిగి repack_q4_1_గమ్యం_q4_1_256_32_bl(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
#అయితే_ముగింపు
}

మూస <> పూర్ణాంకం repack<ఖండం_q4_K, 32, 32>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q4_k_గమ్యం_q4_1_32_bl(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q6_K, 32, 32>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
#ఒకవేళ 1
    తిరిగి repack_q6_k_గమ్యం_q8_0_32_bl_ref(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
#లేకపోతే
    తిరిగి repack_q6_k_గమ్యం_q8_0_32_bl(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
#అయితే_ముగింపు
}

మూస <> పూర్ణాంకం repack<ఖండం_q8_0, 32, 16>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q8_0_గమ్యం_q8_0_16_bl_ref(t, 16, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q8_0, 32, 32>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
#ఒకవేళ 1
    తిరిగి repack_q8_0_గమ్యం_q8_0_32_bl_ref(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
#లేకపోతే
    తిరిగి repack_q8_0_గమ్యం_q8_0_32_bl(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
#అయితే_ముగింపు
}

మూస <> పూర్ణాంకం repack<ఖండం_mxfp4, 32, 32>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_mxfp4_గమ్యం_mxfp4_32_bl(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q5_0, 32, 32>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q5_0_గమ్యం_q5_0_32_bl(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q5_1, 32, 32>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q5_1_గమ్యం_q5_1_32_bl(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q5_K, 32, 32>(ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q5_k_గమ్యం_q5_1_32_bl(t, 32, దత్తాంశం, దత్తాంశం_కొలత);
}

}  // namespace ggml::cpu::riscv64_spacemit
