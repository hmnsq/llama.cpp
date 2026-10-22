#నిర్వచించు GGML_COMMON_IMPL_CPP
#నిర్వచించు GGML_COMMON_DECL_CPP
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "ggml-backend-impl.h"

#అన్తర్భావ "ggml-impl.h"
#అన్తర్భావ "ggml-cpu.h"
#అన్తర్భావ "ggml-cpu-impl.h"
#అన్తర్భావ "simd-mappings.h"
#అన్తర్భావ "traits.h"

#అన్తర్భావ "arch-fallback.h"

#అన్తర్భావ <cmath>
#అన్తర్భావ <cstring>
#అన్తర్భావ <cassert>
#అన్తర్భావ <cstdio>  // for GGML_ASSERT

#అన్తర్భావ "repack.h"

#ఒకవేళ defined(__GNUC__)
#సూచన GCC diagnostic ignored "-Woverlength-strings"
#అయితే_ముగింపు

#నిర్వచించు వాడని GGML_వాడని

స్థిర అంతఃవరుస పూర్ణాంకం సమీప_int(దశాంశం fval) {
    ప్రతిజ్ఞా(fabsf(fval) <= 4194303.f);
    దశాంశం val = fval + 12582912.f;
    పూర్ణాంకం i; జ్ఞాపకం_నకలు(&i, &val, పరిమాణం(పూర్ణాంకం));
    తిరిగి (i & 0x007fffff) - 0x00400000;
}

// Functions to create the interleaved data layout formats

// interleave 4 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x4
// in the interleaved block_q4_0x4, place deltas for 4 block_q4_0 blocks
// first, then interleave quants from 4 block_q4_0s in blocks of blck_size_interleave
//
// - in                  : an array of block_q4_0 pointers
// - blck_size_interleave : the block_q4_0 quants bytes are interleaved in blocks of
//                         blck_size_interleave bytes
// - xor_mask            : the mask to convert the nibbles in block_q4_0 quants bytes
//                         from bias offset form to pure sign form (this saves subtract
//                         operations durin unpacking)
//

బాహ్య "C" {

#ఒకవేళ defined __riscv_zvfh
శూన్య ggml_quantize_mat_q8_0_4x1_generic(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t k) {
    ప్రతిజ్ఞా(QK8_0 == 32);
    ప్రతిజ్ఞా(k % QK8_0 == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / QK8_0;

    ఖండం_q8_0x4 * GGML_RESTRICT y = (ఖండం_q8_0x4 *) vy;

    // scalar
    స్థిరరాశి పూర్ణాంకం blck_కొలత_interleave = 1;
    దశాంశం srcv[4][QK8_0];
    దశాంశం id[4];

    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        కొరకు (పూర్ణాంకం అడ్డువరుస_iter = 0; అడ్డువరుస_iter < 4; అడ్డువరుస_iter++) {
            దశాంశం amax = 0.0f; // absolute max

            కొరకు (పూర్ణాంకం j = 0; j < QK8_0; j++) {
                srcv[అడ్డువరుస_iter][j] = x[అడ్డువరుస_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[అడ్డువరుస_iter][j]));
            }

            స్థిరరాశి దశాంశం d = amax / ((1 << 7) - 1);
            id[అడ్డువరుస_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[అడ్డువరుస_iter] = GGML_CPU_FP32_గమ్యం_FP16(d);
        }

        కొరకు (పూర్ణాంకం j = 0; j < QK8_0 * 4; j++) {
            పూర్ణాంకం src_స్థానభ్రంశం = (j / (4 * blck_కొలత_interleave)) * blck_కొలత_interleave;
            పూర్ణాంకం src_id = (j % (4 * blck_కొలత_interleave)) / blck_కొలత_interleave;
            src_స్థానభ్రంశం += (j % blck_కొలత_interleave);

            దశాంశం x0 = srcv[src_id][src_స్థానభ్రంశం] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

శూన్య ggml_quantize_mat_q8_K_4x1_generic(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t k) {
    ప్రతిజ్ఞా(QK_K == 256);
    ప్రతిజ్ఞా(k % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / QK_K;

    ఖండం_q8_Kx4 * GGML_RESTRICT y = (ఖండం_q8_Kx4 *) vy;

    స్థిరరాశి పూర్ణాంకం blck_కొలత_interleave = 1;
    దశాంశం srcv[4][QK_K];
    దశాంశం iscale[4];

    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        కొరకు (పూర్ణాంకం అడ్డువరుస_iter = 0; అడ్డువరుస_iter < 4; అడ్డువరుస_iter++) {
            దశాంశం amax = 0.0f; // absolute max
            దశాంశం max = 0;

            కొరకు (పూర్ణాంకం j = 0; j < QK_K; j++) {
                srcv[అడ్డువరుస_iter][j] = x[అడ్డువరుస_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                ఒకవేళ(amax < fabsf(srcv[అడ్డువరుస_iter][j])) {
                    amax = fabsf(srcv[అడ్డువరుస_iter][j]);
                    max = srcv[అడ్డువరుస_iter][j];
                }
            }

            iscale[అడ్డువరుస_iter] = amax ? -127.f/max : 0;
            y[i].d[అడ్డువరుస_iter] = amax ? 1/iscale[అడ్డువరుస_iter] : 0;
        }

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }
        కొరకు (పూర్ణాంకం j = 0; j < QK_K * 4; j++) {
            పూర్ణాంకం src_id = j % 4;
            పూర్ణాంకం src_స్థానభ్రంశం = j / 4;
            పూర్ణాంకం సూచీ = ((j >> 6) << 2) + (j & 3);

            దశాంశం x0 = srcv[src_id][src_స్థానభ్రంశం] * iscale[src_id];
            y[i].qs[j] = సమీప_int(x0);
            y[i].bsums[సూచీ] += y[i].qs[j];
        }
    }
}
#అయితే_ముగింపు

శూన్య ggml_quantize_mat_q8_0_4x4_generic(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t k) {
    ప్రతిజ్ఞా(QK8_0 == 32);
    ప్రతిజ్ఞా(k % QK8_0 == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / QK8_0;

    ఖండం_q8_0x4 * GGML_RESTRICT y = (ఖండం_q8_0x4 *) vy;

    // scalar
    స్థిరరాశి పూర్ణాంకం blck_కొలత_interleave = 4;
    దశాంశం srcv[4][QK8_0];
    దశాంశం id[4];

    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        కొరకు (పూర్ణాంకం అడ్డువరుస_iter = 0; అడ్డువరుస_iter < 4; అడ్డువరుస_iter++) {
            దశాంశం amax = 0.0f; // absolute max

            కొరకు (పూర్ణాంకం j = 0; j < QK8_0; j++) {
                srcv[అడ్డువరుస_iter][j] = x[అడ్డువరుస_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[అడ్డువరుస_iter][j]));
            }

            స్థిరరాశి దశాంశం d = amax / ((1 << 7) - 1);
            id[అడ్డువరుస_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[అడ్డువరుస_iter] = GGML_CPU_FP32_గమ్యం_FP16(d);
        }

        కొరకు (పూర్ణాంకం j = 0; j < QK8_0 * 4; j++) {
            పూర్ణాంకం src_స్థానభ్రంశం = (j / (4 * blck_కొలత_interleave)) * blck_కొలత_interleave;
            పూర్ణాంకం src_id = (j % (4 * blck_కొలత_interleave)) / blck_కొలత_interleave;
            src_స్థానభ్రంశం += (j % blck_కొలత_interleave);

            దశాంశం x0 = srcv[src_id][src_స్థానభ్రంశం] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

శూన్య ggml_quantize_mat_q8_0_4x8_generic(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t k) {
    ప్రతిజ్ఞా(QK8_0 == 32);
    ప్రతిజ్ఞా(k % QK8_0 == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / QK8_0;

    ఖండం_q8_0x4 * GGML_RESTRICT y = (ఖండం_q8_0x4 *) vy;

    // scalar
    స్థిరరాశి పూర్ణాంకం blck_కొలత_interleave = 8;
    దశాంశం srcv[4][QK8_0];
    దశాంశం id[4];

    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        కొరకు (పూర్ణాంకం అడ్డువరుస_iter = 0; అడ్డువరుస_iter < 4; అడ్డువరుస_iter++) {
            దశాంశం amax = 0.0f; // absolute max

            కొరకు (పూర్ణాంకం j = 0; j < QK8_0; j++) {
                srcv[అడ్డువరుస_iter][j] = x[అడ్డువరుస_iter * k + i * QK8_0 + j];
                amax = MAX(amax, fabsf(srcv[అడ్డువరుస_iter][j]));
            }

            స్థిరరాశి దశాంశం d = amax / ((1 << 7) - 1);
            id[అడ్డువరుస_iter] = d ? 1.0f / d : 0.0f;

            y[i].d[అడ్డువరుస_iter] = GGML_CPU_FP32_గమ్యం_FP16(d);
        }

        కొరకు (పూర్ణాంకం j = 0; j < QK8_0 * 4; j++) {
            పూర్ణాంకం src_స్థానభ్రంశం = (j / (4 * blck_కొలత_interleave)) * blck_కొలత_interleave;
            పూర్ణాంకం src_id = (j % (4 * blck_కొలత_interleave)) / blck_కొలత_interleave;
            src_స్థానభ్రంశం += (j % blck_కొలత_interleave);

            దశాంశం x0 = srcv[src_id][src_స్థానభ్రంశం] * id[src_id];
            y[i].qs[j] = roundf(x0);
        }
    }
}

శూన్య ggml_quantize_mat_q8_K_4x4_generic(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t k) {
    ప్రతిజ్ఞా(QK_K == 256);
    ప్రతిజ్ఞా(k % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / QK_K;

    ఖండం_q8_Kx4 * GGML_RESTRICT y = (ఖండం_q8_Kx4 *) vy;

    // scalar
    స్థిరరాశి పూర్ణాంకం blck_కొలత_interleave = 4;
    దశాంశం srcv[4][QK_K];
    దశాంశం iscale[4];

    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        కొరకు (పూర్ణాంకం అడ్డువరుస_iter = 0; అడ్డువరుస_iter < 4; అడ్డువరుస_iter++) {
            దశాంశం amax = 0.0f; // absolute max
            దశాంశం max = 0;

            కొరకు (పూర్ణాంకం j = 0; j < QK_K; j++) {
                srcv[అడ్డువరుస_iter][j] = x[అడ్డువరుస_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                ఒకవేళ(amax < fabsf(srcv[అడ్డువరుస_iter][j])) {
                    amax = fabsf(srcv[అడ్డువరుస_iter][j]);
                    max = srcv[అడ్డువరుస_iter][j];
                }
            }

            iscale[అడ్డువరుస_iter] = amax ? -127.f/max : 0;

            y[i].d[అడ్డువరుస_iter] = amax ? 1/iscale[అడ్డువరుస_iter] : 0;
        }

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of four bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        కొరకు (పూర్ణాంకం j = 0; j < QK_K * 4; j++) {
            పూర్ణాంకం src_స్థానభ్రంశం = (j / (4 * blck_కొలత_interleave)) * blck_కొలత_interleave;
            పూర్ణాంకం src_id     = (j % (4 * blck_కొలత_interleave)) / blck_కొలత_interleave;
            src_స్థానభ్రంశం += (j % blck_కొలత_interleave);
            పూర్ణాంకం సూచీ = (((j & 15) >> 2) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            దశాంశం x0 = srcv[src_id][src_స్థానభ్రంశం] * iscale[src_id];
            y[i].qs[j] = సమీప_int(x0);
            y[i].bsums[సూచీ] += y[i].qs[j];
        }
    }
}

శూన్య ggml_quantize_mat_q8_K_4x8_generic(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t k) {
    ప్రతిజ్ఞా(QK_K == 256);
    ప్రతిజ్ఞా(k % QK_K == 0);
    స్థిరరాశి పూర్ణాంకం nb = k / QK_K;

    ఖండం_q8_Kx4 * GGML_RESTRICT y = (ఖండం_q8_Kx4 *) vy;

    // scalar
    స్థిరరాశి పూర్ణాంకం blck_కొలత_interleave = 8;
    దశాంశం srcv[4][QK_K];
    దశాంశం iscale[4];

    కొరకు (పూర్ణాంకం i = 0; i < nb; i++) {
        కొరకు (పూర్ణాంకం అడ్డువరుస_iter = 0; అడ్డువరుస_iter < 4; అడ్డువరుస_iter++) {
            దశాంశం amax = 0.0f; // absolute max
            దశాంశం max = 0;

            కొరకు (పూర్ణాంకం j = 0; j < QK_K; j++) {
                srcv[అడ్డువరుస_iter][j] = x[అడ్డువరుస_iter * k + i * QK_K + j];
                // Update the maximum value of the corresponding super block
                ఒకవేళ(amax < fabsf(srcv[అడ్డువరుస_iter][j])) {
                    amax = fabsf(srcv[అడ్డువరుస_iter][j]);
                    max = srcv[అడ్డువరుస_iter][j];
                }
            }

            iscale[అడ్డువరుస_iter] = amax ? -127.f/max : 0;

            y[i].d[అడ్డువరుస_iter] = amax ? 1/iscale[అడ్డువరుస_iter] : 0;
        }

        కొరకు (పూర్ణాంకం j = 0; j < QK_K / 4; j++) {
            y[i].bsums[j] = 0;
        }

        // Quants values are interleaved in sequence of eight bytes from corresponding super blocks
        // Bsums values are interleaved in sequence of four bsums from each super block taken for interleaving
        // i.e first four bsums from the first super block, followed by first four bsums from second super block and so on
        కొరకు (పూర్ణాంకం j = 0; j < QK_K * 4; j++) {
            పూర్ణాంకం src_స్థానభ్రంశం = (j / (4 * blck_కొలత_interleave)) * blck_కొలత_interleave;
            పూర్ణాంకం src_id     = (j % (4 * blck_కొలత_interleave)) / blck_కొలత_interleave;
            src_స్థానభ్రంశం += (j % blck_కొలత_interleave);
            పూర్ణాంకం సూచీ = (((j & 31) >> 3) << 2) + ((j >> 8) << 4) + ((j >> 6) & 3);

            దశాంశం x0 = srcv[src_id][src_స్థానభ్రంశం] * iscale[src_id];
            y[i].qs[j] = సమీప_int(x0);
            y[i].bsums[సూచీ] += y[i].qs[j];
        }
    }
}

} // extern "C"

మూస <int64_t INTER_కొలత, ggml_type PARAM_TYPE>
శూన్య ggml_quantize_mat_t(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t nrow, int64_t n_చొప్పున_అడ్డువరుస);

మూస <> శూన్య ggml_quantize_mat_t<4, GGML_TYPE_Q8_0>(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t nrow, int64_t n_చొప్పున_అడ్డువరుస) {
    ప్రతిజ్ఞా(nrow == 4);
    వాడని(nrow);
    ggml_quantize_mat_q8_0_4x4(x, vy, n_చొప్పున_అడ్డువరుస);
}

మూస <> శూన్య ggml_quantize_mat_t<8, GGML_TYPE_Q8_0>(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t nrow, int64_t n_చొప్పున_అడ్డువరుస) {
    ప్రతిజ్ఞా(nrow == 4);
    వాడని(nrow);
    ggml_quantize_mat_q8_0_4x8(x, vy, n_చొప్పున_అడ్డువరుస);
}

మూస <> శూన్య ggml_quantize_mat_t<4, GGML_TYPE_Q8_K>(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t nrow, int64_t n_చొప్పున_అడ్డువరుస) {
    ప్రతిజ్ఞా(nrow == 4);
    వాడని(nrow);
    ggml_quantize_mat_q8_K_4x4(x, vy, n_చొప్పున_అడ్డువరుస);
}

మూస <> శూన్య ggml_quantize_mat_t<8, GGML_TYPE_Q8_K>(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t nrow, int64_t n_చొప్పున_అడ్డువరుస) {
    ప్రతిజ్ఞా(nrow == 4);
    వాడని(nrow);
    ggml_quantize_mat_q8_K_4x8(x, vy, n_చొప్పున_అడ్డువరుస);
}

#ఒకవేళ defined __riscv_zvfh
మూస <> శూన్య ggml_quantize_mat_t<1, GGML_TYPE_Q8_0>(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t nrow, int64_t n_చొప్పున_అడ్డువరుస) {
    ప్రతిజ్ఞా(nrow == 4);
    వాడని(nrow);
    ggml_quantize_mat_q8_0_4x1(x, vy, n_చొప్పున_అడ్డువరుస);
}

మూస <> శూన్య ggml_quantize_mat_t<1, GGML_TYPE_Q8_K>(స్థిరరాశి దశాంశం * GGML_RESTRICT x, శూన్య * GGML_RESTRICT vy, int64_t nrow, int64_t n_చొప్పున_అడ్డువరుస) {
    ప్రతిజ్ఞా(nrow == 4);
    వాడని(nrow);
    ggml_quantize_mat_q8_K_4x1(x, vy, n_చొప్పున_అడ్డువరుస);
}
#అయితే_ముగింపు

మూస <పూర్ణాంకం M, పూర్ణాంకం N>
స్థిర శూన్య ggml_gemv_q6_K_NxM_q8_K_generic_impl(పూర్ణాంకం                        n,
                                                 దశాంశం * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 స్థిరరాశి శూన్య * GGML_RESTRICT vx,
                                                 స్థిరరాశి శూన్య * GGML_RESTRICT vy,
                                                 పూర్ణాంకం                        nr,
                                                 పూర్ణాంకం                        nc) {
    constexpr పూర్ణాంకం blocklen          = M;
    constexpr పూర్ణాంకం ncols_interleaved = N;
    స్థిరరాశి పూర్ణాంకం     qk                = QK_K;
    స్థిరరాశి పూర్ణాంకం     nb                = n / qk;
    స్థిరరాశి పూర్ణాంకం     ఖండాలు_చొప్పున_half   = 64 / blocklen;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

    దశాంశం sumf[8];

    స్థిరరాశి ఖండం_q8_K * a_ptr = (స్థిరరాశి ఖండం_q8_K *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q6_Kx8 * b_ptr = (స్థిరరాశి ఖండం_q6_Kx8 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                స్థిరరాశి పూర్ణాంకం పునాది_l = (k / ఖండాలు_చొప్పున_half) * 128 + (k % ఖండాలు_చొప్పున_half) * blocklen;
                స్థిరరాశి పూర్ణాంకం పునాది_h = పునాది_l + 64;

                స్థిరరాశి పూర్ణాంకం scale_idx_l = పునాది_l / 16;
                స్థిరరాశి పూర్ణాంకం scale_idx_h = పునాది_h / 16;

                స్థిరరాశి పూర్ణాంకం qh_జరుపు_l = ((పునాది_l % 128) / 32) * 2;
                స్థిరరాశి పూర్ణాంకం qh_జరుపు_h = ((పునాది_h % 128) / 32) * 2;

                స్థిరరాశి పూర్ణాంకం qh_half_l = (పునాది_l / 128) * 32;
                స్థిరరాశి పూర్ణాంకం qh_half_h = (పునాది_h / 128) * 32;

                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    స్థిరరాశి int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                    స్థిరరాశి int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                    పూర్ణాంకం sumi_l = 0;
                    పూర్ణాంకం sumi_h = 0;

                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; i++) {
                        స్థిరరాశి పూర్ణాంకం ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                        స్థిరరాశి పూర్ణాంకం l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                        స్థిరరాశి పూర్ణాంకం hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                        స్థిరరాశి పూర్ణాంకం qh_idx_l    = qh_half_l + ((పునాది_l + i) % 32);
                        స్థిరరాశి పూర్ణాంకం qh_chunk_l  = qh_idx_l / blocklen;
                        స్థిరరాశి పూర్ణాంకం qh_pos_l    = qh_idx_l % blocklen;
                        స్థిరరాశి పూర్ణాంకం qh_స్థానభ్రంశం_l = qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                        స్థిరరాశి పూర్ణాంకం hi_2_l      = (b_ptr[l].qh[qh_స్థానభ్రంశం_l] >> qh_జరుపు_l) & 0x3;

                        స్థిరరాశి పూర్ణాంకం qh_idx_h    = qh_half_h + ((పునాది_h + i) % 32);
                        స్థిరరాశి పూర్ణాంకం qh_chunk_h  = qh_idx_h / blocklen;
                        స్థిరరాశి పూర్ణాంకం qh_pos_h    = qh_idx_h % blocklen;
                        స్థిరరాశి పూర్ణాంకం qh_స్థానభ్రంశం_h = qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                        స్థిరరాశి పూర్ణాంకం hi_2_h      = (b_ptr[l].qh[qh_స్థానభ్రంశం_h] >> qh_జరుపు_h) & 0x3;

                        స్థిరరాశి పూర్ణాంకం q_l = ((hi_2_l << 4) | l_4) - 32;
                        స్థిరరాశి పూర్ణాంకం q_h = ((hi_2_h << 4) | hi_4) - 32;

                        స్థిరరాశి int8_t a_l = a_ptr[l].qs[పునాది_l + i];
                        స్థిరరాశి int8_t a_h = a_ptr[l].qs[పునాది_h + i];

                        sumi_l += q_l * a_l;
                        sumi_h += q_h * a_h;
                    }

                    sumf[j] +=
                        (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
        }

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

మూస <పూర్ణాంకం M, పూర్ణాంకం N>
స్థిర శూన్య ggml_gemm_q6_K_NxM_q8_K_generic_impl(పూర్ణాంకం                        n,
                                                 దశాంశం * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 స్థిరరాశి శూన్య * GGML_RESTRICT vx,
                                                 స్థిరరాశి శూన్య * GGML_RESTRICT vy,
                                                 పూర్ణాంకం                        nr,
                                                 పూర్ణాంకం                        nc) {
    constexpr పూర్ణాంకం blocklen          = M;
    constexpr పూర్ణాంకం ncols_interleaved = N;
    స్థిరరాశి పూర్ణాంకం     qk                = QK_K;
    స్థిరరాశి పూర్ణాంకం     nb                = n / qk;
    స్థిరరాశి పూర్ణాంకం     ఖండాలు_చొప్పున_half   = 64 / blocklen;
    స్థిరరాశి పూర్ణాంకం     q8_half_stride    = 512;
    స్థిరరాశి పూర్ణాంకం     q8_దిగువ_ఎగువ_అడుగు  = 256;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nr % 4 == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(bs);

    దశాంశం sumf[4][8];

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_Kx4 * a_ptr = (స్థిరరాశి ఖండం_q8_Kx4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q6_Kx8 * b_ptr = (స్థిరరాశి ఖండం_q6_Kx8 *) vx + (x * nb);

            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                    స్థిరరాశి పూర్ణాంకం పునాది_l = (k / ఖండాలు_చొప్పున_half) * 128 + (k % ఖండాలు_చొప్పున_half) * blocklen;
                    స్థిరరాశి పూర్ణాంకం పునాది_h = పునాది_l + 64;

                    స్థిరరాశి పూర్ణాంకం scale_idx_l = పునాది_l / 16;
                    స్థిరరాశి పూర్ణాంకం scale_idx_h = పునాది_h / 16;

                    స్థిరరాశి పూర్ణాంకం qh_జరుపు_l = ((పునాది_l % 128) / 32) * 2;
                    స్థిరరాశి పూర్ణాంకం qh_జరుపు_h = ((పునాది_h % 128) / 32) * 2;

                    స్థిరరాశి పూర్ణాంకం qh_half_l = (పునాది_l / 128) * 32;
                    స్థిరరాశి పూర్ణాంకం qh_half_h = (పునాది_h / 128) * 32;

                    స్థిరరాశి పూర్ణాంకం q8_పునాది = (k / ఖండాలు_చొప్పున_half) * q8_half_stride + (k % ఖండాలు_చొప్పున_half) * (blocklen * 4);

                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            స్థిరరాశి int8_t scale_l = b_ptr[l].scales[scale_idx_l * ncols_interleaved + j];
                            స్థిరరాశి int8_t scale_h = b_ptr[l].scales[scale_idx_h * ncols_interleaved + j];

                            పూర్ణాంకం sumi_l = 0;
                            పూర్ణాంకం sumi_h = 0;

                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; i++) {
                                స్థిరరాశి పూర్ణాంకం ql_pos = k * ncols_interleaved * blocklen + j * blocklen + i;
                                స్థిరరాశి పూర్ణాంకం l_4    = b_ptr[l].ql[ql_pos] & 0xF;
                                స్థిరరాశి పూర్ణాంకం hi_4   = (b_ptr[l].ql[ql_pos] >> 4) & 0xF;

                                స్థిరరాశి పూర్ణాంకం qh_idx_l   = qh_half_l + ((పునాది_l + i) % 32);
                                స్థిరరాశి పూర్ణాంకం qh_chunk_l = qh_idx_l / blocklen;
                                స్థిరరాశి పూర్ణాంకం qh_pos_l   = qh_idx_l % blocklen;
                                స్థిరరాశి పూర్ణాంకం qh_స్థానభ్రంశం_l =
                                    qh_chunk_l * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_l;
                                స్థిరరాశి పూర్ణాంకం hi_2_l = (b_ptr[l].qh[qh_స్థానభ్రంశం_l] >> qh_జరుపు_l) & 0x3;

                                స్థిరరాశి పూర్ణాంకం qh_idx_h   = qh_half_h + ((పునాది_h + i) % 32);
                                స్థిరరాశి పూర్ణాంకం qh_chunk_h = qh_idx_h / blocklen;
                                స్థిరరాశి పూర్ణాంకం qh_pos_h   = qh_idx_h % blocklen;
                                స్థిరరాశి పూర్ణాంకం qh_స్థానభ్రంశం_h =
                                    qh_chunk_h * (blocklen * ncols_interleaved) + j * blocklen + qh_pos_h;
                                స్థిరరాశి పూర్ణాంకం hi_2_h = (b_ptr[l].qh[qh_స్థానభ్రంశం_h] >> qh_జరుపు_h) & 0x3;

                                స్థిరరాశి పూర్ణాంకం q_l = ((hi_2_l << 4) | l_4) - 32;
                                స్థిరరాశి పూర్ణాంకం q_h = ((hi_2_h << 4) | hi_4) - 32;

                                స్థిరరాశి int8_t q8_l = a_ptr[l].qs[q8_పునాది + m * blocklen + i];
                                స్థిరరాశి int8_t q8_h = a_ptr[l].qs[q8_పునాది + m * blocklen + i + q8_దిగువ_ఎగువ_అడుగు];

                                sumi_l += q_l * q8_l;
                                sumi_h += q_h * q8_h;
                            }

                            sumf[m][j] += (sumi_l * scale_l + sumi_h * scale_h) * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) *
                                          a_ptr[l].d[m];
                        }
                    }
                }
            }

            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

మూస <పూర్ణాంకం M, పూర్ణాంకం N>
స్థిర శూన్య ggml_gemv_q5_K_NxM_q8_K_generic_impl(పూర్ణాంకం                        n,
                                                 దశాంశం * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 స్థిరరాశి శూన్య * GGML_RESTRICT vx,
                                                 స్థిరరాశి శూన్య * GGML_RESTRICT vy,
                                                 పూర్ణాంకం                        nr,
                                                 పూర్ణాంకం                        nc) {
    constexpr పూర్ణాంకం         blocklen          = M;
    constexpr పూర్ణాంకం         ncols_interleaved = N;
    స్థిరరాశి పూర్ణాంకం             qk                = QK_K;
    స్థిరరాశి పూర్ణాంకం             nb                = n / qk;
    స్థిర స్థిరరాశి uint32_t kmask1            = 0x3f3f3f3f;
    స్థిర స్థిరరాశి uint32_t kmask2            = 0x0f0f0f0f;
    స్థిర స్థిరరాశి uint32_t kmask3            = 0x03030303;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

    దశాంశం    sumf[ncols_interleaved];
    దశాంశం    కూడిక_minf[ncols_interleaved];
    uint32_t utmp[32];
    పూర్ణాంకం      sumi1;
    పూర్ణాంకం      sumi2;
    పూర్ణాంకం      sumi;

    స్థిరరాశి ఖండం_q8_K * a_ptr = (స్థిరరాశి ఖండం_q8_K *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q5_Kx8 * b_ptr = (స్థిరరాశి ఖండం_q5_Kx8 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            sumf[j]     = 0.0;
            కూడిక_minf[j] = 0.0;
        }
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb++) {
                జ్ఞాపకం_నకలు(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_కొలత, K_SCALE_కొలత);
                utmp[sb * 4 + 3]      = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                స్థిరరాశి uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2]      = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                constexpr పూర్ణాంకం scale_stride = 32;
                uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                స్థిరరాశి పూర్ణాంకం qh_జరుపు = (k / (32 / blocklen)) * 2;
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi  = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం b_qs_స్థానభ్రంశం = k * ncols_interleaved * blocklen + j * blocklen + i;

                        స్థిరరాశి పూర్ణాంకం qh_idx      = (k * blocklen + i) % 32;
                        స్థిరరాశి పూర్ణాంకం qh_chunk    = qh_idx / blocklen;
                        స్థిరరాశి పూర్ణాంకం qh_pos      = qh_idx % blocklen;
                        స్థిరరాశి పూర్ణాంకం b_qh_స్థానభ్రంశం = qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                        స్థిరరాశి uint8_t qh_val = b_ptr[l].qh[b_qh_స్థానభ్రంశం];
                        స్థిరరాశి uint8_t h0     = (qh_val >> qh_జరుపు) & 1;
                        స్థిరరాశి uint8_t h1     = (qh_val >> (qh_జరుపు + 1)) & 1;

                        స్థిరరాశి పూర్ణాంకం v0 = (int8_t) ((b_ptr[l].qs[b_qs_స్థానభ్రంశం] & 0xF) | (h0 << 4));
                        స్థిరరాశి పూర్ణాంకం v1 = (int8_t) ((b_ptr[l].qs[b_qs_స్థానభ్రంశం] >> 4) | (h1 << 4));

                        స్థిరరాశి పూర్ణాంకం q8_స్థానభ్రంశం = (k / (32 / blocklen)) * 64 + (k % (32 / blocklen)) * blocklen + i;

                        sumi1 = (v0 * a_ptr[l].qs[q8_స్థానభ్రంశం]);
                        sumi2 = (v1 * a_ptr[l].qs[q8_స్థానభ్రంశం + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    కూడిక_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) *
                                   GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - కూడిక_minf[j];
        }
    }
}

మూస <పూర్ణాంకం M, పూర్ణాంకం N>
స్థిర శూన్య ggml_gemm_q5_K_NxM_q8_K_generic_impl(పూర్ణాంకం                        n,
                                                 దశాంశం * GGML_RESTRICT      s,
                                                 size_t                     bs,
                                                 స్థిరరాశి శూన్య * GGML_RESTRICT vx,
                                                 స్థిరరాశి శూన్య * GGML_RESTRICT vy,
                                                 పూర్ణాంకం                        nr,
                                                 పూర్ణాంకం                        nc) {
    constexpr పూర్ణాంకం         blocklen          = M;
    constexpr పూర్ణాంకం         ncols_interleaved = N;
    స్థిరరాశి పూర్ణాంకం             qk                = QK_K;
    స్థిరరాశి పూర్ణాంకం             nb                = n / qk;
    స్థిర స్థిరరాశి uint32_t kmask1            = 0x3f3f3f3f;
    స్థిర స్థిరరాశి uint32_t kmask2            = 0x0f0f0f0f;
    స్థిర స్థిరరాశి uint32_t kmask3            = 0x03030303;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nr % 4 == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    దశాంశం    sumf[4][ncols_interleaved];
    దశాంశం    కూడిక_minf[4][ncols_interleaved];
    uint32_t utmp[32];
    పూర్ణాంకం      sumi1;
    పూర్ణాంకం      sumi2;
    పూర్ణాంకం      sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_Kx4 * a_ptr = (స్థిరరాశి ఖండం_q8_Kx4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q5_Kx8 * b_ptr = (స్థిరరాశి ఖండం_q5_Kx8 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j]     = 0.0;
                    కూడిక_minf[m][j] = 0.0;
                }
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb++) {
                    జ్ఞాపకం_నకలు(utmp + sb * 4, b_ptr[l].scales + sb * K_SCALE_కొలత, K_SCALE_కొలత);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    స్థిరరాశి uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1]      = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2]      = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                    constexpr పూర్ణాంకం scale_stride = 32;
                    uint8_t *     scales_0     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride;
                    uint8_t *     scales_1     = (uint8_t *) utmp + (k / (32 / blocklen)) * scale_stride + 16;

                    స్థిరరాశి పూర్ణాంకం qh_జరుపు = (k / (32 / blocklen)) * 2;
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi  = 0;
                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                స్థిరరాశి పూర్ణాంకం b_qs_స్థానభ్రంశం = k * ncols_interleaved * blocklen + j * blocklen + i;

                                స్థిరరాశి పూర్ణాంకం qh_idx   = (k * blocklen + i) % 32;
                                స్థిరరాశి పూర్ణాంకం qh_chunk = qh_idx / blocklen;
                                స్థిరరాశి పూర్ణాంకం qh_pos   = qh_idx % blocklen;
                                స్థిరరాశి పూర్ణాంకం b_qh_స్థానభ్రంశం =
                                    qh_chunk * (blocklen * ncols_interleaved) + j * blocklen + qh_pos;

                                స్థిరరాశి uint8_t qh_val = b_ptr[l].qh[b_qh_స్థానభ్రంశం];
                                స్థిరరాశి uint8_t h0     = (qh_val >> qh_జరుపు) & 1;
                                స్థిరరాశి uint8_t h1     = (qh_val >> (qh_జరుపు + 1)) & 1;

                                స్థిరరాశి పూర్ణాంకం v0 = (int8_t) ((b_ptr[l].qs[b_qs_స్థానభ్రంశం] & 0xF) | (h0 << 4));
                                స్థిరరాశి పూర్ణాంకం v1 = (int8_t) ((b_ptr[l].qs[b_qs_స్థానభ్రంశం] >> 4) | (h1 << 4));

                                స్థిరరాశి పూర్ణాంకం q8_స్థానభ్రంశం = (k / (32 / blocklen)) * 256 +
                                                      (k % (32 / blocklen)) * 4 * blocklen + m * blocklen + i;

                                sumi1 = (v0 * a_ptr[l].qs[q8_స్థానభ్రంశం]);
                                sumi2 = (v1 * a_ptr[l].qs[q8_స్థానభ్రంశం + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        స్థిరరాశి int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            కూడిక_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) *
                                              GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - కూడిక_minf[m][j];
                }
            }
        }
    }
}

బాహ్య "C" {

శూన్య ggml_gemv_q4_0_4x4_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;
    స్థిరరాశి పూర్ణాంకం blocklen = 4;

    ప్రతిజ్ఞా(nr == 1);
    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(s);
    వాడని(bs);
    వాడని(vx);
    వాడని(vy);
    వాడని(nr);
    వాడని(nc);
    వాడని(nb);
    వాడని(ncols_interleaved);
    వాడని(blocklen);

    దశాంశం sumf[4];
    పూర్ణాంకం sumi;

    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q4_0x4 * b_ptr = (స్థిరరాశి ఖండం_q4_0x4 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        స్థిరరాశి పూర్ణాంకం v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d);
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

శూన్య ggml_gemv_q4_0_4x8_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;
    స్థిరరాశి పూర్ణాంకం blocklen = 8;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(s);
    వాడని(bs);
    వాడని(vx);
    వాడని(vy);
    వాడని(nr);
    వాడని(nc);
    వాడని(nb);
    వాడని(ncols_interleaved);
    వాడని(blocklen);

    దశాంశం sumf[4];
    పూర్ణాంకం sumi;

    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q4_0x4 * b_ptr = (స్థిరరాశి ఖండం_q4_0x4 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        స్థిరరాశి పూర్ణాంకం v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d);
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

శూన్య ggml_gemv_q4_0_8x8_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 8;
    స్థిరరాశి పూర్ణాంకం blocklen = 8;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(s);
    వాడని(bs);
    వాడని(vx);
    వాడని(vy);
    వాడని(nr);
    వాడని(nc);
    వాడని(nb);
    వాడని(ncols_interleaved);
    వాడని(blocklen);

    దశాంశం sumf[8];
    పూర్ణాంకం sumi;

    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q4_0x8 * b_ptr = (స్థిరరాశి ఖండం_q4_0x8 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        స్థిరరాశి పూర్ణాంకం v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d);
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

శూన్య ggml_gemv_q4_K_8x4_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK_K;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 8;
    స్థిరరాశి పూర్ణాంకం blocklen = 4;
    స్థిర స్థిరరాశి uint32_t kmask1 = 0x3f3f3f3f;
    స్థిర స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;
    స్థిర స్థిరరాశి uint32_t kmask3 = 0x03030303;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

    దశాంశం sumf[8];
    దశాంశం కూడిక_minf[8];
    uint32_t utmp[32];
    పూర్ణాంకం sumi1;
    పూర్ణాంకం sumi2;
    పూర్ణాంకం sumi;

    స్థిరరాశి ఖండం_q8_K * a_ptr = (స్థిరరాశి ఖండం_q8_K *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q4_Kx8 * b_ptr = (స్థిరరాశి ఖండం_q4_Kx8 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            కూడిక_minf[j] = 0.0;
        }
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb++) {
                జ్ఞాపకం_నకలు(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                స్థిరరాశి uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        స్థిరరాశి పూర్ణాంకం v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 64 + (k % 8) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb++) {
                uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    కూడిక_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - కూడిక_minf[j];
        }
    }
}

శూన్య ggml_gemv_q4_K_8x8_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK_K;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 8;
    స్థిరరాశి పూర్ణాంకం blocklen = 8;
    స్థిర స్థిరరాశి uint32_t kmask1 = 0x3f3f3f3f;
    స్థిర స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;
    స్థిర స్థిరరాశి uint32_t kmask3 = 0x03030303;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

    దశాంశం sumf[8];
    దశాంశం కూడిక_minf[8];
    uint32_t utmp[32];
    పూర్ణాంకం sumi1;
    పూర్ణాంకం sumi2;
    పూర్ణాంకం sumi;

    స్థిరరాశి ఖండం_q8_K * a_ptr = (స్థిరరాశి ఖండం_q8_K *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q4_Kx8 * b_ptr = (స్థిరరాశి ఖండం_q4_Kx8 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            కూడిక_minf[j] = 0.0;
        }
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb++) {
                జ్ఞాపకం_నకలు(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                స్థిరరాశి uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                utmp[sb * 4 + 2] = uaux_0;
                utmp[sb * 4 + 0] &= kmask1;
            }
            కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                        స్థిరరాశి పూర్ణాంకం v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 64 + (k % 4) * blocklen + i + 32]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb++) {
                uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    కూడిక_minf[j] += mins[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - కూడిక_minf[j];
        }
    }
}

శూన్య ggml_gemv_q2_K_8x8_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK_K;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 8;
    స్థిరరాశి పూర్ణాంకం blocklen = 8;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(s);
    వాడని(bs);
    వాడని(vx);
    వాడని(vy);
    వాడని(nr);
    వాడని(nc);
    వాడని(nb);
    వాడని(ncols_interleaved);
    వాడని(blocklen);

    దశాంశం sumf[8];
    దశాంశం కూడిక_minf[8];
    పూర్ణాంకం sumi1,sumi2,sumi3,sumi4;
    పూర్ణాంకం sumi;

    స్థిరరాశి ఖండం_q8_K * a_ptr = (స్థిరరాశి ఖండం_q8_K *)vy;
    కొరకు(పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q2_Kx8 * b_ptr = (స్థిరరాశి ఖండం_q2_Kx8 *) vx + (x * nb);
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
            కూడిక_minf[j] = 0.0;
        }
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం k = 0; k < (qk / (4 * blocklen)); k++) {
                స్థిరరాశి uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                స్థిరరాశి uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                స్థిరరాశి uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                స్థిరరాశి uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi1 = 0;
                    sumi2 = 0;
                    sumi3 = 0;
                    sumi4 = 0;
                    sumi = 0;
                    పూర్ణాంకం స్థానభ్రంశం = ((k / 2) % 2) + j * 2;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i){
                        స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                        స్థిరరాశి పూర్ణాంకం v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                        స్థిరరాశి పూర్ణాంకం v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                        స్థిరరాశి పూర్ణాంకం v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
                        sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i]);
                        sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 32]);
                        sumi3 = (v2 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 64]);
                        sumi4 = (v3 * a_ptr[l].qs[(k >> 2) * 128 + (k % 4) * blocklen + i + 96]);

                        sumi1 = sumi1 * (scales_0[స్థానభ్రంశం] & 0xF);
                        sumi2 = sumi2 * (scales_1[స్థానభ్రంశం] & 0xF);
                        sumi3 = sumi3 * (scales_2[స్థానభ్రంశం] & 0xF);
                        sumi4 = sumi4 * (scales_3[స్థానభ్రంశం] & 0xF);
                        sumi += sumi1 + sumi2 + sumi3 + sumi4;
                    }
                    sumf[j] += sumi * GGML_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                }
            }
            కొరకు(పూర్ణాంకం sb = 0; sb < 8; sb++) {
                స్థిరరాశి uint8_t *mins = b_ptr[l].scales + sb * 16;
                కొరకు(పూర్ణాంకం j = 0; j < ncols_interleaved; j++){
                    కూడిక_minf[j] += ((mins[j * 2] >> 4) * a_ptr[l].bsums[sb * 2] + (mins[(j * 2)+ 1] >> 4) * a_ptr[l].bsums[sb * 2 + 1]) * GGML_FP16_గమ్యం_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - కూడిక_minf[j];
        }
    }
}

శూన్య ggml_gemv_q5_K_8x4_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

శూన్య ggml_gemv_q5_K_8x8_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}


శూన్య ggml_gemv_q6_K_8x4_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

శూన్య ggml_gemv_q6_K_8x8_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

శూన్య ggml_gemv_iq4_nl_4x4_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;
    స్థిరరాశి పూర్ణాంకం blocklen = 4;

    ప్రతిజ్ఞా(nr == 1);
    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

    దశాంశం sumf[4];
    పూర్ణాంకం sumi;

    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_iq4_nlx4 * b_ptr = (స్థిరరాశి ఖండం_iq4_nlx4 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        స్థిరరాశి పూర్ణాంకం v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d);
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

శూన్య ggml_gemv_iq4_nl_8x8_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 8;
    స్థిరరాశి పూర్ణాంకం blocklen = 8;

    ప్రతిజ్ఞా(nr == 1);
    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

    దశాంశం sumf[8];
    పూర్ణాంకం sumi;

    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_iq4_nlx8 * b_ptr = (స్థిరరాశి ఖండం_iq4_nlx8 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        స్థిరరాశి పూర్ణాంకం v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d);
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

శూన్య ggml_gemv_mxfp4_4x4_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;
    స్థిరరాశి పూర్ణాంకం blocklen = 4;

    ప్రతిజ్ఞా(nr == 1);
    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

    దశాంశం sumf[4];
    పూర్ణాంకం sumi;

    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_mxfp4x4 * b_ptr = (స్థిరరాశి ఖండం_mxfp4x4 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        స్థిరరాశి పూర్ణాంకం v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8_M0_గమ్యం_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d);
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

శూన్య ggml_gemv_mxfp4_8x8_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 8;
    స్థిరరాశి పూర్ణాంకం blocklen = 8;

    ప్రతిజ్ఞా(nr == 1);
    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

    దశాంశం sumf[8];
    పూర్ణాంకం sumi;

    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_mxfp4x8 * b_ptr = (స్థిరరాశి ఖండం_mxfp4x8 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        స్థిరరాశి పూర్ణాంకం v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_E8_M0_గమ్యం_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d);
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

శూన్య ggml_gemv_q8_0_4x4_q8_0_generic(పూర్ణాంకం                        n,
                                     దశాంశం * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vx,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vy,
                                     పూర్ణాంకం                        nr,
                                     పూర్ణాంకం                        nc) {
    స్థిరరాశి పూర్ణాంకం qk                = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb                = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;
    స్థిరరాశి పూర్ణాంకం blocklen          = 4;

    ప్రతిజ్ఞా(nr == 1);
    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

    దశాంశం sumf[4];
    పూర్ణాంకం   sumi;

    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q8_0x4 * b_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం k = 0; k < (qk / blocklen); k++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d);
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

శూన్య ggml_gemv_q8_0_4x8_q8_0_generic(పూర్ణాంకం                        n,
                                     దశాంశం * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vx,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vy,
                                     పూర్ణాంకం                        nr,
                                     పూర్ణాంకం                        nc) {
    స్థిరరాశి పూర్ణాంకం qk                = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb                = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;
    స్థిరరాశి పూర్ణాంకం blocklen          = 8;

    ప్రతిజ్ఞా(nr == 1);
    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

    దశాంశం sumf[4];
    పూర్ణాంకం   sumi;

    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q8_0x4 * b_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం k = 0; k < (qk / blocklen); k++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d);
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

శూన్య ggml_gemv_q1_0_4x4_q8_0_generic(పూర్ణాంకం                        n,
                                     దశాంశం * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vx,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vy,
                                     పూర్ణాంకం                        nr,
                                     పూర్ణాంకం                        nc) {
    స్థిరరాశి పూర్ణాంకం qk                = QK1_0;
    స్థిరరాశి పూర్ణాంకం nb                = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;

    ప్రతిజ్ఞా(nr == 1);
    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

    దశాంశం sumf[4];

    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q1_0x4 * b_ptr = (స్థిరరాశి ఖండం_q1_0x4 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }

        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            స్థిరరాశి దశాంశం d0[4] = {
                GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[0]),
                GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[1]),
                GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[2]),
                GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[3]),
            };

            కొరకు (పూర్ణాంకం k = 0; k < QK1_0 / QK8_0; ++k) {
                స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT a_blk = a_ptr + l * (QK1_0 / QK8_0) + k;
                స్థిరరాశి దశాంశం d1 = GGML_CPU_FP16_గమ్యం_FP32(a_blk->d);
                స్థిరరాశి దశాంశం scale[4] = { d0[0] * d1, d0[1] * d1, d0[2] * d1, d0[3] * d1 };

                కొరకు (పూర్ణాంకం tile = 0; tile < QK8_0 / 4; ++tile) {
                    స్థిరరాశి uint8_t బిట్లు_lo = b_ptr[l].qs[k * 16 + 2 * tile + 0];
                    స్థిరరాశి uint8_t బిట్లు_hi = b_ptr[l].qs[k * 16 + 2 * tile + 1];

                    కొరకు (పూర్ణాంకం p = 0; p < 4; ++p) {
                        స్థిరరాశి దశాంశం q = (దశాంశం) a_blk->qs[tile * 4 + p];

                        sumf[0] += ((బిట్లు_lo & (1u << p))       ? scale[0] : -scale[0]) * q;
                        sumf[1] += ((బిట్లు_lo & (1u << (4 + p))) ? scale[1] : -scale[1]) * q;
                        sumf[2] += ((బిట్లు_hi & (1u << p))       ? scale[2] : -scale[2]) * q;
                        sumf[3] += ((బిట్లు_hi & (1u << (4 + p))) ? scale[3] : -scale[3]) * q;
                    }
                }
            }
        }

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

శూన్య ggml_gemv_q1_0_4x8_q8_0_generic(పూర్ణాంకం                        n,
                                     దశాంశం * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vx,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vy,
                                     పూర్ణాంకం                        nr,
                                     పూర్ణాంకం                        nc) {
    స్థిరరాశి పూర్ణాంకం qk                = QK1_0;
    స్థిరరాశి పూర్ణాంకం nb                = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;
    స్థిరరాశి పూర్ణాంకం blocklen          = 8;

    ప్రతిజ్ఞా(nr == 1);
    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

    దశాంశం sumf[4];

    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q1_0x4 * b_ptr = (స్థిరరాశి ఖండం_q1_0x4 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
        }

        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            స్థిరరాశి దశాంశం d0[4] = {
                GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[0]),
                GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[1]),
                GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[2]),
                GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[3]),
            };

            కొరకు (పూర్ణాంకం k = 0; k < qk / blocklen; ++k) {
                స్థిరరాశి ఖండం_q8_0 * GGML_RESTRICT a_blk = a_ptr + l * (qk / QK8_0) + k / (QK8_0 / blocklen);
                స్థిరరాశి దశాంశం d1 = GGML_CPU_FP16_గమ్యం_FP32(a_blk->d);
                స్థిరరాశి దశాంశం scale[4] = { d0[0] * d1, d0[1] * d1, d0[2] * d1, d0[3] * d1 };
                స్థిరరాశి uint8_t bits0 = b_ptr[l].qs[k * ncols_interleaved + 0];
                స్థిరరాశి uint8_t bits1 = b_ptr[l].qs[k * ncols_interleaved + 1];
                స్థిరరాశి uint8_t bits2 = b_ptr[l].qs[k * ncols_interleaved + 2];
                స్థిరరాశి uint8_t bits3 = b_ptr[l].qs[k * ncols_interleaved + 3];
                స్థిరరాశి పూర్ణాంకం q_స్థానభ్రంశం = (k % (QK8_0 / blocklen)) * blocklen;

                కొరకు (పూర్ణాంకం p = 0; p < blocklen; ++p) {
                    స్థిరరాశి దశాంశం q = (దశాంశం) a_blk->qs[q_స్థానభ్రంశం + p];

                    sumf[0] += ((bits0 & (1u << p)) ? scale[0] : -scale[0]) * q;
                    sumf[1] += ((bits1 & (1u << p)) ? scale[1] : -scale[1]) * q;
                    sumf[2] += ((bits2 & (1u << p)) ? scale[2] : -scale[2]) * q;
                    sumf[3] += ((bits3 & (1u << p)) ? scale[3] : -scale[3]) * q;
                }
            }
        }

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

// Only enable these for RISC-V.
#ఒకవేళ defined __riscv_zvfh
శూన్య ggml_gemv_q4_0_16x1_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 16;
    స్థిరరాశి పూర్ణాంకం blocklen = 1;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(s);
    వాడని(bs);
    వాడని(vx);
    వాడని(vy);
    వాడని(nr);
    వాడని(nc);
    వాడని(nb);
    వాడని(ncols_interleaved);
    వాడని(blocklen);

    దశాంశం sumf[16];
    పూర్ణాంకం sumi;

    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q4_0x16 * b_ptr = (స్థిరరాశి ఖండం_q4_0x16 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                        స్థిరరాశి పూర్ణాంకం v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2])) >> 4;
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d);
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

శూన్య ggml_gemv_q4_K_16x1_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK_K;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 16;
    స్థిరరాశి పూర్ణాంకం blocklen = 1;
    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);
    వాడని(s);
    వాడని(bs);
    వాడని(vx);
    వాడని(vy);
    వాడని(nr);
    వాడని(nc);
    వాడని(nb);
    వాడని(ncols_interleaved);
    వాడని(blocklen);
    దశాంశం sumf[16];
    దశాంశం కూడిక_minf[16];
    uint8_t scales[128];
    uint8_t mins[128];
    పూర్ణాంకం sumi1;
    పూర్ణాంకం sumi2;
    పూర్ణాంకం sumi;
    స్థిరరాశి ఖండం_q8_K * a_ptr = (స్థిరరాశి ఖండం_q8_K *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q4_Kx16 * b_ptr = (స్థిరరాశి ఖండం_q4_Kx16 *) vx + (x * nb);
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0f;
            కూడిక_minf[j] = 0.0f;
        }
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం i = 0; i < 128; i++) {
                scales[i] = b_ptr[l].scales[i] & 0x0F;
                mins[i] = b_ptr[l].scales[i] >> 4;
            }
            కొరకు (పూర్ణాంకం i = 0; i < 64; i++) {
                scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
            }
            కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb++) {
                uint8_t *min = &mins[sb * 16];
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    కూడిక_minf[j] += min[j] * (a_ptr[l].bsums[sb * 2] + a_ptr[l].bsums[sb * 2 + 1]) * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d;
                }
            }
            కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb += 2) {
                uint8_t *scales_0 = &scales[sb * 16];
                uint8_t *scales_1 = &scales[(sb + 1) * 16];
                కొరకు (పూర్ణాంకం i = 0; i < QK4_0; i++) {
                    కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                        sumi1 = 0;
                        sumi2 = 0;
                        sumi = 0;
                        స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                        స్థిరరాశి పూర్ణాంకం v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                        sumi1 = (v0 * a_ptr[l].qs[sb * 32 + i]);
                        sumi2 = (v1 * a_ptr[l].qs[sb * 32 + 32 + i]);
                        sumi1 = sumi1 * scales_0[j];
                        sumi2 = sumi2 * scales_1[j];
                        sumi += sumi1 + sumi2;
                        sumf[j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * a_ptr[l].d;
                    }
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j] - కూడిక_minf[j];
        }
    }
}

శూన్య ggml_gemv_iq4_nl_16x1_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 16;
    స్థిరరాశి పూర్ణాంకం blocklen = 1;

    ప్రతిజ్ఞా(nr == 1);
    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

    దశాంశం sumf[16];
    పూర్ణాంకం sumi;

    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_iq4_nlx16 * b_ptr = (స్థిరరాశి ఖండం_iq4_nlx16 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[j] = 0.0;
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                        స్థిరరాశి పూర్ణాంకం v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                        sumi += ((v0 * a_ptr[l].qs[k * blocklen + i]) + (v1 * a_ptr[l].qs[k * blocklen + i + qk / 2]));
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d);
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) s[x * ncols_interleaved + j] = sumf[j];
    }
}

శూన్య ggml_gemv_q8_0_16x1_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk                = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb                = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 16;
    స్థిరరాశి పూర్ణాంకం blocklen          = 1;

    ప్రతిజ్ఞా(nr == 1);
    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    వాడని(bs);
    వాడని(nr);

    దశాంశం sumf[16];
    పూర్ణాంకం   sumi;

    స్థిరరాశి ఖండం_q8_0 * a_ptr = (స్థిరరాశి ఖండం_q8_0 *) vy;
    కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
        స్థిరరాశి ఖండం_q8_0x16 * b_ptr = (స్థిరరాశి ఖండం_q8_0x16 *) vx + (x * nb);

        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            sumf[j] = 0.0;
        }
        కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
            కొరకు (పూర్ణాంకం k = 0; k < (qk / blocklen); k++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumi = 0;
                    కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                        స్థిరరాశి పూర్ణాంకం v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                        sumi += v0 * a_ptr[l].qs[k * blocklen + i];
                    }
                    sumf[j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d);
                }
            }
        }
        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
            s[x * ncols_interleaved + j] = sumf[j];
        }
    }
}

శూన్య ggml_gemv_q2_K_16x1_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nr == 1);
    ప్రతిజ్ఞా(nc % 16 == 0);

    వాడని(bs);
    వాడని(nr);

    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;
    స్థిరరాశి ఖండం_q2_Kx16 * x = (స్థిరరాశి ఖండం_q2_Kx16 *)vx;
    స్థిరరాశి ఖండం_q8_K    * y = (స్థిరరాశి ఖండం_q8_K *)vy;

    // Layout: Even-Low(0,2,4,6), Odd-Low(1,3,5,7), Even-High(8...), Odd-High(9...)
    స్థిరరాశి పూర్ణాంకం sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,  // 0-7
        8, 12, 9, 13, 10, 14, 11, 15 // 8-15
    };

    కొరకు (పూర్ణాంకం నిలువు_tile = 0; నిలువు_tile < nc; నిలువు_tile += 16) {
        స్థిరరాశి ఖండం_q2_Kx16 * x_ptr = x + (నిలువు_tile / 16) * nb;
        స్థిరరాశి ఖండం_q8_K    * y_ptr = y;

        దశాంశం sumf[16] = {0};

        // Loop over K-blocks
        కొరకు (పూర్ణాంకం k_ఖండం = 0; k_ఖండం < nb; ++k_ఖండం) {
            int32_t isum[16]  = {0};
            int32_t summs[16] = {0};

            స్థిరరాశి uint8_t * qs_rhs = x_ptr[k_ఖండం].qs;
            స్థిరరాశి uint8_t * sc_rhs = x_ptr[k_ఖండం].scales;
            స్థిరరాశి int8_t  * qs_lhs = y_ptr[k_ఖండం].qs;
            స్థిరరాశి int16_t * bs_lhs = y_ptr[k_ఖండం].bsums;

            // Iterate over sub-blocks 0..15
            కొరకు (పూర్ణాంకం sb = 0; sb < 16; ++sb) {
                // Correction Term
                int16_t bsum = bs_lhs[sb];
                పూర్ణాంకం scale_స్థానభ్రంశం = sb_perm[sb] * 16;

                కొరకు (పూర్ణాంకం నిలువు = 0; నిలువు < 16; ++నిలువు) {
                    uint8_t sc_val = sc_rhs[scale_స్థానభ్రంశం + నిలువు];
                    summs[నిలువు] += bsum * (sc_val >> 4); // Min is high 4 bits
                }

                // Main Dot Product
                // Calculate base offsets for Q2 unpacking based on SB
                పూర్ణాంకం byte_పునాది;
                ఒకవేళ (sb < 8) byte_పునాది = (sb % 2 == 0) ? 0 : 16;
                లేకపోతే        byte_పునాది = (sb % 2 == 0) ? 32 : 48;

                పూర్ణాంకం జరుపు = ((sb / 2) % 4) * 2;

                కొరకు (పూర్ణాంకం నిలువు = 0; నిలువు < 16; ++నిలువు) {
                    uint8_t sc_val = sc_rhs[scale_స్థానభ్రంశం + నిలువు];
                    int32_t d_sb = sc_val & 0xF; // Scale is low 4 bits

                    // Process 16 elements (l=0..15)
                    కొరకు (పూర్ణాంకం l = 0; l < 16; ++l) {
                        // Q2: Interleaved by column. Byte `l` contains 4 k-values.
                        పూర్ణాంకం qs_idx = (byte_పునాది + l) * 16 + నిలువు;
                        uint8_t q2_val = (qs_rhs[qs_idx] >> జరుపు) & 3;

                        // Q8: Linear access
                        పూర్ణాంకం k = sb * 16 + l;
                        int8_t q8_val = qs_lhs[k];

                        isum[నిలువు] += q8_val * q2_val * d_sb;
                    }
                }
            }

            // Finalize K-Block
            కొరకు (పూర్ణాంకం నిలువు = 0; నిలువు < 16; ++నిలువు) {
                దశాంశం d_lhs = y_ptr[k_ఖండం].d;
                దశాంశం d_rhs = GGML_FP16_గమ్యం_FP32(x_ptr[k_ఖండం].d[నిలువు]);
                దశాంశం dm_rhs = GGML_FP16_గమ్యం_FP32(x_ptr[k_ఖండం].dmin[నిలువు]);

                దశాంశం d_all = d_lhs * d_rhs;
                దశాంశం d_min = d_lhs * dm_rhs;

                sumf[నిలువు] += (isum[నిలువు] * d_all) - (summs[నిలువు] * d_min);
            }
        }

        కొరకు (పూర్ణాంకం నిలువు = 0; నిలువు < 16; ++నిలువు) {
            s[నిలువు_tile + నిలువు] = sumf[నిలువు];
        }
    }
}
#అయితే_ముగింపు

శూన్య ggml_gemm_q4_0_4x4_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;
    స్థిరరాశి పూర్ణాంకం blocklen = 4;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nr % 4 == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(s);
    వాడని(bs);
    వాడని(vx);
    వాడని(vy);
    వాడని(nr);
    వాడని(nc);
    వాడని(nb);
    వాడని(ncols_interleaved);
    వాడని(blocklen);

    {
        దశాంశం sumf[4][4];
        పూర్ణాంకం sumi;

        కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
            స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (y * nb);
            కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
                స్థిరరాశి ఖండం_q4_0x4 * b_ptr = (స్థిరరాశి ఖండం_q4_0x4 *) vx + (x * nb);
                కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                    కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                    కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                        కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                            కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                    స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                    స్థిరరాశి పూర్ణాంకం v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                    కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

శూన్య ggml_gemm_q4_0_4x8_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;
    స్థిరరాశి పూర్ణాంకం blocklen = 8;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nr % 4 == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(s);
    వాడని(bs);
    వాడని(vx);
    వాడని(vy);
    వాడని(nr);
    వాడని(nc);
    వాడని(nb);
    వాడని(ncols_interleaved);
    వాడని(blocklen);

    దశాంశం sumf[4][4];
    పూర్ణాంకం sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q4_0x4 * b_ptr = (స్థిరరాశి ఖండం_q4_0x4 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                స్థిరరాశి పూర్ణాంకం v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                        (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

శూన్య ggml_gemm_q4_0_8x8_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 8;
    స్థిరరాశి పూర్ణాంకం blocklen = 8;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nr % 4 == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(s);
    వాడని(bs);
    వాడని(vx);
    వాడని(vy);
    వాడని(nr);
    వాడని(nc);
    వాడని(nb);
    వాడని(ncols_interleaved);
    వాడని(blocklen);

    దశాంశం sumf[4][8];
    పూర్ణాంకం sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q4_0x8 * b_ptr = (స్థిరరాశి ఖండం_q4_0x8 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                స్థిరరాశి పూర్ణాంకం v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

శూన్య ggml_gemm_q4_K_8x4_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK_K;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 8;
    స్థిరరాశి పూర్ణాంకం blocklen = 4;
    స్థిర స్థిరరాశి uint32_t kmask1 = 0x3f3f3f3f;
    స్థిర స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;
    స్థిర స్థిరరాశి uint32_t kmask3 = 0x03030303;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nr % 4 == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(nb);
    వాడని(ncols_interleaved);
    వాడని(blocklen);

    దశాంశం sumf[4][8];
    దశాంశం కూడిక_minf[4][8];
    uint32_t utmp[32];
    పూర్ణాంకం sumi1;
    పూర్ణాంకం sumi2;
    పూర్ణాంకం sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_Kx4 * a_ptr = (స్థిరరాశి ఖండం_q8_Kx4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q4_Kx8 * b_ptr = (స్థిరరాశి ఖండం_q4_Kx8 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    కూడిక_minf[m][j] = 0.0;
                }
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb++) {
                    జ్ఞాపకం_నకలు(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    స్థిరరాశి uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t * scales_0 = (uint8_t *) utmp + (k / 8) * 32;
                    uint8_t * scales_1 = (uint8_t *) utmp + (k / 8) * 32 + 16;
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                స్థిరరాశి పూర్ణాంకం v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k / 8) * 256 + (k % 8) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb++) {
                    uint8_t * mins = (uint8_t *) utmp + 8 + sb * 16;
                    కొరకు(పూర్ణాంకం m = 0; m < 4; m++) {
                        స్థిరరాశి int16_t * bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        కొరకు(పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            కూడిక_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - కూడిక_minf[m][j];
                }
            }
        }
    }
}

శూన్య ggml_gemm_q4_K_8x8_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK_K;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 8;
    స్థిరరాశి పూర్ణాంకం blocklen = 8;
    స్థిర స్థిరరాశి uint32_t kmask1 = 0x3f3f3f3f;
    స్థిర స్థిరరాశి uint32_t kmask2 = 0x0f0f0f0f;
    స్థిర స్థిరరాశి uint32_t kmask3 = 0x03030303;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nr % 4 == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(bs);

    దశాంశం sumf[4][8];
    దశాంశం కూడిక_minf[4][8];
    uint32_t utmp[32];
    పూర్ణాంకం sumi1;
    పూర్ణాంకం sumi2;
    పూర్ణాంకం sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_Kx4 * a_ptr = (స్థిరరాశి ఖండం_q8_Kx4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q4_Kx8 * b_ptr = (స్థిరరాశి ఖండం_q4_Kx8 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    కూడిక_minf[m][j] = 0.0;
                }
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb++) {
                    జ్ఞాపకం_నకలు(utmp + sb * 4, b_ptr[l].scales + sb * 12, 12);
                    utmp[sb * 4 + 3] = ((utmp[sb * 4 + 2] >> 4) & kmask2) | (((utmp[sb * 4 + 1] >> 6) & kmask3) << 4);
                    స్థిరరాశి uint32_t uaux_0 = utmp[sb * 4 + 1] & kmask1;
                    utmp[sb * 4 + 1] = (utmp[sb * 4 + 2] & kmask2) | (((utmp[sb * 4 + 0] >> 6) & kmask3) << 4);
                    utmp[sb * 4 + 2] = uaux_0;
                    utmp[sb * 4 + 0] &= kmask1;
                }
                కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                    uint8_t *scales_0 = (uint8_t*) utmp + (k / 4) * 32;
                    uint8_t *scales_1 = (uint8_t*) utmp + (k / 4) * 32 + 16;
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi = 0;
                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF);
                                స్థిరరాశి పూర్ణాంకం v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 256 + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb++) {
                    uint8_t *mins = (uint8_t*) utmp + 8 + sb * 16;
                    కొరకు(పూర్ణాంకం m = 0; m < 4; m++) {
                        స్థిరరాశి int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) * 6);
                        కొరకు(పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            కూడిక_minf[m][j] += mins[j] * (bsums[0] + bsums[1]) * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - కూడిక_minf[m][j];
                }
            }
        }
    }
}

శూన్య ggml_gemm_q2_K_8x8_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK_K;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 8;
    స్థిరరాశి పూర్ణాంకం blocklen = 8;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nr % 4 == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(s);
    వాడని(bs);
    వాడని(vx);
    వాడని(vy);
    వాడని(nr);
    వాడని(nc);
    వాడని(nb);
    వాడని(ncols_interleaved);
    వాడని(blocklen);

    దశాంశం sumf[4][8];
    దశాంశం కూడిక_minf[4][8];
    పూర్ణాంకం sumi1, sumi2, sumi3, sumi4;
    పూర్ణాంకం sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_Kx4 * a_ptr = (స్థిరరాశి ఖండం_q8_Kx4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q2_Kx8 * b_ptr = (స్థిరరాశి ఖండం_q2_Kx8 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    కూడిక_minf[m][j] = 0.0;
                }
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం k = 0; k < (qk / (4 * blocklen)); k++) {

                    స్థిరరాశి uint8_t *scales_0 = b_ptr[l].scales + (k / 4) * 64 ;
                    స్థిరరాశి uint8_t *scales_1 = b_ptr[l].scales + (k / 4) * 64 + 16;
                    స్థిరరాశి uint8_t *scales_2 = b_ptr[l].scales + (k / 4) * 64 + 32;
                    స్థిరరాశి uint8_t *scales_3 = b_ptr[l].scales + (k / 4) * 64 + 48;
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            sumi1 = 0;
                            sumi2 = 0;
                            sumi3 = 0;
                            sumi4 = 0;
                            sumi = 0;
                            పూర్ణాంకం స్థానభ్రంశం = ((k / 2) % 2) + j * 2;
                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i){
                                స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 3);
                                స్థిరరాశి పూర్ణాంకం v1 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 2 ) & 3);
                                స్థిరరాశి పూర్ణాంకం v2 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4 ) & 3);
                                స్థిరరాశి పూర్ణాంకం v3 = (int8_t) ((b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 6 ) & 3);
                                sumi1 = (v0 * a_ptr[l].qs[(k >> 2) * 512 + (k % 4) * 4 * blocklen + m * blocklen + i]);
                                sumi2 = (v1 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 128]);
                                sumi3 = (v2 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 256]);
                                sumi4 = (v3 * a_ptr[l].qs[(k >> 2) * 512  + (k % 4) * 4 * blocklen + m * blocklen + i + 384]);
                                sumi1 = sumi1 * (scales_0[స్థానభ్రంశం] & 0xF);
                                sumi2 = sumi2 * (scales_1[స్థానభ్రంశం] & 0xF);
                                sumi3 = sumi3 * (scales_2[స్థానభ్రంశం] & 0xF);
                                sumi4 = sumi4 * (scales_3[స్థానభ్రంశం] & 0xF);
                                sumi += sumi1 + sumi2 + sumi3 + sumi4;
                            }
                            sumf[m][j] += sumi * GGML_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                        }
                    }
                }
                కొరకు(పూర్ణాంకం sb = 0; sb < 8; sb++) {
                    స్థిరరాశి uint8_t *mins = b_ptr[l].scales + sb * 16;
                    కొరకు(పూర్ణాంకం m = 0; m < 4; m++) {
                        స్థిరరాశి int16_t *bsums = a_ptr[l].bsums + (sb * 8) + (m * 4) - ((sb % 2) *  6);
                        కొరకు(పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            పూర్ణాంకం mins_prod = ((mins[j * 2] >> 4) * bsums[0] + (mins[(j * 2)+ 1] >> 4) * bsums[1]);
                            కూడిక_minf[m][j] += (mins_prod) * GGML_FP16_గమ్యం_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }
            }

            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - కూడిక_minf[m][j];
                }
            }
        }
    }
}

శూన్య ggml_gemm_q5_K_8x4_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

శూన్య ggml_gemm_q5_K_8x8_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q5_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

శూన్య ggml_gemm_q6_K_8x4_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q6_K_NxM_q8_K_generic_impl<4, 8>(n, s, bs, vx, vy, nr, nc);
}

శూన్య ggml_gemm_q6_K_8x8_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
   ggml_gemm_q6_K_NxM_q8_K_generic_impl<8, 8>(n, s, bs, vx, vy, nr, nc);
}

శూన్య ggml_gemm_iq4_nl_4x4_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;
    స్థిరరాశి పూర్ణాంకం blocklen = 4;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nr % 4 == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(s);
    వాడని(bs);
    వాడని(vx);
    వాడని(vy);
    వాడని(nr);
    వాడని(nc);
    వాడని(nb);
    వాడని(ncols_interleaved);
    వాడని(blocklen);

    {
        దశాంశం sumf[4][4];
        పూర్ణాంకం sumi;

        కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
            స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (y * nb);
            కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
                స్థిరరాశి ఖండం_iq4_nlx4 * b_ptr = (స్థిరరాశి ఖండం_iq4_nlx4 *) vx + (x * nb);
                కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                    కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
                }
                కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                    కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                        కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                            కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                                sumi = 0;
                                కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                    స్థిరరాశి పూర్ణాంకం v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                    స్థిరరాశి పూర్ణాంకం v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                    sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                            (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                                }
                                sumf[m][j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d[m]);
                            }
                        }
                    }
                }
                కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                    కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++)
                        s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

శూన్య ggml_gemm_iq4_nl_8x8_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 8;
    స్థిరరాశి పూర్ణాంకం blocklen = 8;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nr % 4 == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    దశాంశం sumf[4][8];
    పూర్ణాంకం sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_iq4_nlx8 * b_ptr = (స్థిరరాశి ఖండం_iq4_nlx8 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                స్థిరరాశి పూర్ణాంకం v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                స్థిరరాశి పూర్ణాంకం v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

శూన్య ggml_gemm_mxfp4_4x4_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;
    స్థిరరాశి పూర్ణాంకం blocklen = 4;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nr % 4 == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    దశాంశం sumf[4][4];
    పూర్ణాంకం sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_mxfp4x4 * b_ptr = (స్థిరరాశి ఖండం_mxfp4x4 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                స్థిరరాశి పూర్ణాంకం v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                స్థిరరాశి పూర్ణాంకం v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8_M0_గమ్యం_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

శూన్య ggml_gemm_mxfp4_8x8_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 8;
    స్థిరరాశి పూర్ణాంకం blocklen = 8;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nr % 4 == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    దశాంశం sumf[4][8];
    పూర్ణాంకం sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_mxfp4x8 * b_ptr = (స్థిరరాశి ఖండం_mxfp4x8 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                స్థిరరాశి పూర్ణాంకం v0 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                స్థిరరాశి పూర్ణాంకం v1 = kvalues_mxfp4[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_E8_M0_గమ్యం_FP32_HALF(b_ptr[l].e[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

శూన్య ggml_gemm_q8_0_4x4_q8_0_generic(పూర్ణాంకం                        n,
                                     దశాంశం * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vx,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vy,
                                     పూర్ణాంకం                        nr,
                                     పూర్ణాంకం                        nc) {
    స్థిరరాశి పూర్ణాంకం qk                = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb                = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;
    స్థిరరాశి పూర్ణాంకం blocklen          = 4;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nr % 4 == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    దశాంశం sumf[4][4];
    పూర్ణాంకం   sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q8_0x4 * b_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం k = 0; k < (qk / blocklen); k++) {
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                స్థిరరాశి పూర్ణాంకం v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}



శూన్య ggml_gemm_q8_0_4x8_q8_0_generic(పూర్ణాంకం                        n,
                                     దశాంశం * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vx,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vy,
                                     పూర్ణాంకం                        nr,
                                     పూర్ణాంకం                        nc) {
    స్థిరరాశి పూర్ణాంకం qk                = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb                = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;
    స్థిరరాశి పూర్ణాంకం blocklen          = 8;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nr % 4 == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    దశాంశం sumf[4][4];
    పూర్ణాంకం   sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q8_0x4 * b_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం k = 0; k < (qk / blocklen); k++) {
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                స్థిరరాశి పూర్ణాంకం v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

శూన్య ggml_gemm_q1_0_4x4_q8_0_generic(పూర్ణాంకం                        n,
                                     దశాంశం * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vx,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vy,
                                     పూర్ణాంకం                        nr,
                                     పూర్ణాంకం                        nc) {
    స్థిరరాశి పూర్ణాంకం qk                = QK1_0;
    స్థిరరాశి పూర్ణాంకం nb                = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nr % 4 == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    దశాంశం sumf[4][4];

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (4 * y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q1_0x4 * b_ptr = (స్థిరరాశి ఖండం_q1_0x4 *) vx + (x * nb);

            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }

            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                స్థిరరాశి దశాంశం d0[4] = {
                    GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[3]),
                };

                కొరకు (పూర్ణాంకం k = 0; k < QK1_0 / QK8_0; ++k) {
                    స్థిరరాశి ఖండం_q8_0x4 * GGML_RESTRICT a_blk = a_ptr + 4 * l + k;
                    స్థిరరాశి దశాంశం a_d[4] = {
                        GGML_CPU_FP16_గమ్యం_FP32(a_blk->d[0]),
                        GGML_CPU_FP16_గమ్యం_FP32(a_blk->d[1]),
                        GGML_CPU_FP16_గమ్యం_FP32(a_blk->d[2]),
                        GGML_CPU_FP16_గమ్యం_FP32(a_blk->d[3]),
                    };

                    కొరకు (పూర్ణాంకం tile = 0; tile < QK8_0 / 4; ++tile) {
                        స్థిరరాశి uint8_t బిట్లు_lo = b_ptr[l].qs[k * 16 + 2 * tile + 0];
                        స్థిరరాశి uint8_t బిట్లు_hi = b_ptr[l].qs[k * 16 + 2 * tile + 1];
                        స్థిరరాశి పూర్ణాంకం tile_స్థానభ్రంశం = tile * 16;

                        కొరకు (పూర్ణాంకం p = 0; p < 4; ++p) {
                            స్థిరరాశి int8_t q_అడ్డువరుస[4] = {
                                a_blk->qs[tile_స్థానభ్రంశం + 0 * 4 + p],
                                a_blk->qs[tile_స్థానభ్రంశం + 1 * 4 + p],
                                a_blk->qs[tile_స్థానభ్రంశం + 2 * 4 + p],
                                a_blk->qs[tile_స్థానభ్రంశం + 3 * 4 + p],
                            };
                            స్థిరరాశి పూర్ణాంకం sign[4] = {
                                (బిట్లు_lo & (1u << p))       ? 1 : -1,
                                (బిట్లు_lo & (1u << (4 + p))) ? 1 : -1,
                                (బిట్లు_hi & (1u << p))       ? 1 : -1,
                                (బిట్లు_hi & (1u << (4 + p))) ? 1 : -1,
                            };

                            కొరకు (పూర్ణాంకం m = 0; m < 4; ++m) {
                                స్థిరరాశి దశాంశం అడ్డువరుస_scale = a_d[m];
                                sumf[m][0] += sign[0] * q_అడ్డువరుస[m] * d0[0] * అడ్డువరుస_scale;
                                sumf[m][1] += sign[1] * q_అడ్డువరుస[m] * d0[1] * అడ్డువరుస_scale;
                                sumf[m][2] += sign[2] * q_అడ్డువరుస[m] * d0[2] * అడ్డువరుస_scale;
                                sumf[m][3] += sign[3] * q_అడ్డువరుస[m] * d0[3] * అడ్డువరుస_scale;
                            }
                        }
                    }
                }
            }

            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

శూన్య ggml_gemm_q1_0_4x8_q8_0_generic(పూర్ణాంకం                        n,
                                     దశాంశం * GGML_RESTRICT      s,
                                     size_t                     bs,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vx,
                                     స్థిరరాశి శూన్య * GGML_RESTRICT vy,
                                     పూర్ణాంకం                        nr,
                                     పూర్ణాంకం                        nc) {
    స్థిరరాశి పూర్ణాంకం qk                = QK1_0;
    స్థిరరాశి పూర్ణాంకం nb                = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 4;
    స్థిరరాశి పూర్ణాంకం blocklen          = 8;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nr % 4 == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    దశాంశం sumf[4][4];

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (4 * y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q1_0x4 * b_ptr = (స్థిరరాశి ఖండం_q1_0x4 *) vx + (x * nb);

            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0f;
                }
            }

            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                స్థిరరాశి దశాంశం d0[4] = {
                    GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[0]),
                    GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[1]),
                    GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[2]),
                    GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[3]),
                };

                కొరకు (పూర్ణాంకం k = 0; k < qk / blocklen; ++k) {
                    స్థిరరాశి ఖండం_q8_0x4 * GGML_RESTRICT a_blk = a_ptr + 4 * l + k / (QK8_0 / blocklen);
                    స్థిరరాశి దశాంశం a_d[4] = {
                        GGML_CPU_FP16_గమ్యం_FP32(a_blk->d[0]),
                        GGML_CPU_FP16_గమ్యం_FP32(a_blk->d[1]),
                        GGML_CPU_FP16_గమ్యం_FP32(a_blk->d[2]),
                        GGML_CPU_FP16_గమ్యం_FP32(a_blk->d[3]),
                    };
                    స్థిరరాశి uint8_t bits0 = b_ptr[l].qs[k * ncols_interleaved + 0];
                    స్థిరరాశి uint8_t bits1 = b_ptr[l].qs[k * ncols_interleaved + 1];
                    స్థిరరాశి uint8_t bits2 = b_ptr[l].qs[k * ncols_interleaved + 2];
                    స్థిరరాశి uint8_t bits3 = b_ptr[l].qs[k * ncols_interleaved + 3];
                    స్థిరరాశి పూర్ణాంకం q_స్థానభ్రంశం = (k % (QK8_0 / blocklen)) * 4 * blocklen;

                    కొరకు (పూర్ణాంకం p = 0; p < blocklen; ++p) {
                        స్థిరరాశి int8_t q_అడ్డువరుస[4] = {
                            a_blk->qs[q_స్థానభ్రంశం + 0 * blocklen + p],
                            a_blk->qs[q_స్థానభ్రంశం + 1 * blocklen + p],
                            a_blk->qs[q_స్థానభ్రంశం + 2 * blocklen + p],
                            a_blk->qs[q_స్థానభ్రంశం + 3 * blocklen + p],
                        };
                        స్థిరరాశి పూర్ణాంకం sign[4] = {
                            (bits0 & (1u << p)) ? 1 : -1,
                            (bits1 & (1u << p)) ? 1 : -1,
                            (bits2 & (1u << p)) ? 1 : -1,
                            (bits3 & (1u << p)) ? 1 : -1,
                        };

                        కొరకు (పూర్ణాంకం m = 0; m < 4; ++m) {
                            స్థిరరాశి దశాంశం అడ్డువరుస_scale = a_d[m];
                            sumf[m][0] += sign[0] * q_అడ్డువరుస[m] * d0[0] * అడ్డువరుస_scale;
                            sumf[m][1] += sign[1] * q_అడ్డువరుస[m] * d0[1] * అడ్డువరుస_scale;
                            sumf[m][2] += sign[2] * q_అడ్డువరుస[m] * d0[2] * అడ్డువరుస_scale;
                            sumf[m][3] += sign[3] * q_అడ్డువరుస[m] * d0[3] * అడ్డువరుస_scale;
                        }
                    }
                }
            }

            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}

// Only enable these for RISC-V.
#ఒకవేళ defined __riscv_zvfh
శూన్య ggml_gemm_q4_0_16x1_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 16;
    స్థిరరాశి పూర్ణాంకం blocklen = 1;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nr % 4 == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(s);
    వాడని(bs);
    వాడని(vx);
    వాడని(vy);
    వాడని(nr);
    వాడని(nc);
    వాడని(nb);
    వాడని(ncols_interleaved);
    వాడని(blocklen);

    దశాంశం sumf[4][16];
    పూర్ణాంకం sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q4_0x16 * b_ptr = (స్థిరరాశి ఖండం_q4_0x16 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] << 4);
                                స్థిరరాశి పూర్ణాంకం v1 = (int8_t) (b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0xF0);
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + qk / 2 * 4])) >> 4;
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

శూన్య ggml_gemm_q4_K_16x1_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK_K;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 16;
    స్థిరరాశి పూర్ణాంకం blocklen = 1;

    ప్రతిజ్ఞా (n % qk == 0);
    ప్రతిజ్ఞా (nr % 4 == 0);
    ప్రతిజ్ఞా (nc % ncols_interleaved == 0);

    వాడని(s);
    వాడని(bs);
    వాడని(vx);
    వాడని(vy);
    వాడని(nr);
    వాడని(nc);
    వాడని(nb);
    వాడని(ncols_interleaved);
    వాడని(blocklen);

    దశాంశం sumf[4][16];
    దశాంశం కూడిక_minf[4][16];
    uint8_t scales[128];
    uint8_t mins[128];
    పూర్ణాంకం sumi1;
    పూర్ణాంకం sumi2;
    పూర్ణాంకం sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_Kx4 * a_ptr = (స్థిరరాశి ఖండం_q8_Kx4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q4_Kx16 * b_ptr = (స్థిరరాశి ఖండం_q4_Kx16 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                    కూడిక_minf[m][j] = 0.0;
                }
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం i = 0; i < 128; i++) {
                    scales[i] = b_ptr[l].scales[i] & 0x0F;
                    mins[i] = b_ptr[l].scales[i] >> 4;
                }
                కొరకు (పూర్ణాంకం i = 0; i < 64; i++) {
                    scales[i] |= (b_ptr[l].scales[128 + i] & 0x03) << 4;
                    mins[i] |= (b_ptr[l].scales[128 + i] & 0x0C) << 2;
                    scales[i + 64] |= (b_ptr[l].scales[128 + i] & 0x30);
                    mins[i + 64] |= (b_ptr[l].scales[128 + i] & 0xC0) >> 2;
                }

                కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb++) {
                    uint8_t *min = &mins[sb * 16];
                    కొరకు(పూర్ణాంకం m = 0; m < 4; m++) {
                        స్థిరరాశి int16_t bsums = a_ptr[l].bsums[sb * 8 + m] + a_ptr[l].bsums[sb * 8 + m + 4];
                        కొరకు(పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            కూడిక_minf[m][j] += min[j] * bsums * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].dmin[j]) * a_ptr[l].d[m];
                        }
                    }
                }

                కొరకు (పూర్ణాంకం sb = 0; sb < 8; sb += 2) {
                    uint8_t *scales_0 = &scales[sb * 16];
                    uint8_t *scales_1 = &scales[(sb + 1) * 16];

                    కొరకు (పూర్ణాంకం i = 0; i < QK4_0; i++) {
                        కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                            కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                                sumi1 = 0;
                                sumi2 = 0;
                                sumi = 0;

                                స్థిరరాశి పూర్ణాంకం v0 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] & 0xF);
                                స్థిరరాశి పూర్ణాంకం v1 = (int8_t) (b_ptr[l].qs[sb * 256 + i * 16 + j] >> 4);
                                sumi1 = (v0 * a_ptr[l].qs[sb * 4 * 32 + i * 4 + m]);
                                sumi2 = (v1 * a_ptr[l].qs[sb * 4 * 32 + 32 * 4 + i * 4 + m]);
                                sumi1 = sumi1 * scales_0[j];
                                sumi2 = sumi2 * scales_1[j];
                                sumi += sumi1 + sumi2;

                                sumf[m][j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * a_ptr[l].d[m];
                            }
                        }
                    }
                }
            }
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j] - కూడిక_minf[m][j];
                }
            }
        }
    }
}

శూన్య ggml_gemm_iq4_nl_16x1_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 16;
    స్థిరరాశి పూర్ణాంకం blocklen = 1;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nr % 4 == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    దశాంశం sumf[4][16];
    పూర్ణాంకం sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_iq4_nlx16 * b_ptr = (స్థిరరాశి ఖండం_iq4_nlx16 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) sumf[m][j] = 0.0;
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం k = 0; k < (qk / (2 * blocklen)); k++) {
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                స్థిరరాశి పూర్ణాంకం v0 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] & 0x0F];
                                స్థిరరాశి పూర్ణాంకం v1 = kvalues_iq4nl[b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i] >> 4];
                                sumi += ((v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i]) +
                                         (v1 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i + (qk / 2) * 4]));
                            }
                            sumf[m][j] += sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++)
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
            }
        }
    }
}

శూన్య ggml_gemm_q8_0_16x1_q8_0_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    స్థిరరాశి పూర్ణాంకం qk                = QK8_0;
    స్థిరరాశి పూర్ణాంకం nb                = n / qk;
    స్థిరరాశి పూర్ణాంకం ncols_interleaved = 16;
    స్థిరరాశి పూర్ణాంకం blocklen          = 1;

    ప్రతిజ్ఞా(n % qk == 0);
    ప్రతిజ్ఞా(nr % 4 == 0);
    ప్రతిజ్ఞా(nc % ncols_interleaved == 0);

    దశాంశం sumf[4][16];
    పూర్ణాంకం   sumi;

    కొరకు (పూర్ణాంకం y = 0; y < nr / 4; y++) {
        స్థిరరాశి ఖండం_q8_0x4 * a_ptr = (స్థిరరాశి ఖండం_q8_0x4 *) vy + (y * nb);
        కొరకు (పూర్ణాంకం x = 0; x < nc / ncols_interleaved; x++) {
            స్థిరరాశి ఖండం_q8_0x16 * b_ptr = (స్థిరరాశి ఖండం_q8_0x16 *) vx + (x * nb);
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    sumf[m][j] = 0.0;
                }
            }
            కొరకు (పూర్ణాంకం l = 0; l < nb; l++) {
                కొరకు (పూర్ణాంకం k = 0; k < (qk / blocklen); k++) {
                    కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                        కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                            sumi = 0;
                            కొరకు (పూర్ణాంకం i = 0; i < blocklen; ++i) {
                                స్థిరరాశి పూర్ణాంకం v0 = b_ptr[l].qs[k * ncols_interleaved * blocklen + j * blocklen + i];
                                sumi += v0 * a_ptr[l].qs[k * 4 * blocklen + m * blocklen + i];
                            }
                            sumf[m][j] +=
                                sumi * GGML_CPU_FP16_గమ్యం_FP32(b_ptr[l].d[j]) * GGML_CPU_FP16_గమ్యం_FP32(a_ptr[l].d[m]);
                        }
                    }
                }
            }
            కొరకు (పూర్ణాంకం m = 0; m < 4; m++) {
                కొరకు (పూర్ణాంకం j = 0; j < ncols_interleaved; j++) {
                    s[(y * 4 + m) * bs + x * ncols_interleaved + j] = sumf[m][j];
                }
            }
        }
    }
}


శూన్య ggml_gemm_q2_K_16x1_q8_K_generic(పూర్ణాంకం n, దశాంశం * GGML_RESTRICT s, size_t bs, స్థిరరాశి శూన్య * GGML_RESTRICT vx, స్థిరరాశి శూన్య * GGML_RESTRICT vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ప్రతిజ్ఞా(n % QK_K == 0);
    ప్రతిజ్ఞా(nr % 4 == 0);
    ప్రతిజ్ఞా(nc % 16 == 0);
    స్థిరరాశి పూర్ణాంకం nb = n / QK_K;
    స్థిరరాశి ఖండం_q2_Kx16 * x = (స్థిరరాశి ఖండం_q2_Kx16 *)vx;
    స్థిరరాశి ఖండం_q8_Kx4  * y = (స్థిరరాశి ఖండం_q8_Kx4 *)vy;

    స్థిరరాశి పూర్ణాంకం sb_perm[16] = {
        0, 4, 1, 5, 2, 6, 3, 7,
        8, 12, 9, 13, 10, 14, 11, 15
    };

    // Iterate Rows in tiles of 4
    కొరకు (పూర్ణాంకం అడ్డువరుస_tile = 0; అడ్డువరుస_tile < nr; అడ్డువరుస_tile += 4) {
        // Iterate Columns in tiles of 16
        కొరకు (పూర్ణాంకం నిలువు_tile = 0; నిలువు_tile < nc; నిలువు_tile += 16) {

            స్థిరరాశి ఖండం_q2_Kx16 * x_ptr = x + (నిలువు_tile / 16) * nb;
            స్థిరరాశి ఖండం_q8_Kx4  * y_ptr = y + (అడ్డువరుస_tile / 4) * nb;

            దశాంశం sumf[4][16];
            memset(sumf, 0, పరిమాణం(sumf));

            కొరకు (పూర్ణాంకం k_ఖండం = 0; k_ఖండం < nb; ++k_ఖండం) {
                int32_t isum[4][16];
                int32_t summs[4][16];
                memset(isum, 0, పరిమాణం(isum));
                memset(summs, 0, పరిమాణం(summs));

                స్థిరరాశి uint8_t * qs_rhs = x_ptr[k_ఖండం].qs;
                స్థిరరాశి uint8_t * sc_rhs = x_ptr[k_ఖండం].scales;
                స్థిరరాశి int8_t  * qs_lhs = y_ptr[k_ఖండం].qs;
                స్థిరరాశి int16_t * bs_lhs = y_ptr[k_ఖండం].bsums;

                కొరకు (పూర్ణాంకం sb = 0; sb < 16; ++sb) {
                    పూర్ణాంకం scale_స్థానభ్రంశం = sb_perm[sb] * 16;

                    పూర్ణాంకం byte_పునాది;
                    ఒకవేళ (sb < 8) byte_పునాది = (sb % 2 == 0) ? 0 : 16;
                    లేకపోతే        byte_పునాది = (sb % 2 == 0) ? 32 : 48;
                    పూర్ణాంకం జరుపు = ((sb / 2) % 4) * 2;

                    కొరకు (పూర్ణాంకం నిలువు = 0; నిలువు < 16; ++నిలువు) {
                        uint8_t sc_val = sc_rhs[scale_స్థానభ్రంశం + నిలువు];
                        int32_t d_sb = sc_val & 0xF;
                        int32_t m_sb = sc_val >> 4;

                        // Correction Term
                        కొరకు (పూర్ణాంకం r = 0; r < 4; ++r) {
                            పూర్ణాంకం bsum_idx = (sb / 4) * 16 + r * 4 + (sb % 4);
                            summs[r][నిలువు] += bs_lhs[bsum_idx] * m_sb;
                        }

                        // Main Dot Product
                        కొరకు (పూర్ణాంకం l = 0; l < 16; ++l) {
                            పూర్ణాంకం qs_idx = (byte_పునాది + l) * 16 + నిలువు;
                            uint8_t q2_val = (qs_rhs[qs_idx] >> జరుపు) & 3;

                            // Calculate Q8 index for this specific k and row
                            పూర్ణాంకం k = sb * 16 + l;
                            పూర్ణాంకం q8_idx = (k / 4) * 16 + (k % 4);

                            కొరకు (పూర్ణాంకం r = 0; r < 4; ++r) {
                                // Add r*4 to jump to the correct row within the 4x4 chunk
                                int8_t q8_val = qs_lhs[q8_idx + r * 4];
                                isum[r][నిలువు] += q8_val * q2_val * d_sb;
                            }
                        }
                    }
                }

                // Finalize K-Block
                కొరకు (పూర్ణాంకం నిలువు = 0; నిలువు < 16; ++నిలువు) {
                    దశాంశం d_rhs = GGML_FP16_గమ్యం_FP32(x_ptr[k_ఖండం].d[నిలువు]);
                    దశాంశం dm_rhs = GGML_FP16_గమ్యం_FP32(x_ptr[k_ఖండం].dmin[నిలువు]);

                    కొరకు (పూర్ణాంకం r = 0; r < 4; ++r) {
                        దశాంశం d_lhs = y_ptr[k_ఖండం].d[r];
                        దశాంశం d_all = d_lhs * d_rhs;
                        దశాంశం d_min = d_lhs * dm_rhs;
                        sumf[r][నిలువు] += (isum[r][నిలువు] * d_all) - (summs[r][నిలువు] * d_min);
                    }
                }
            }

            కొరకు (పూర్ణాంకం r = 0; r < 4; ++r) {
                కొరకు (పూర్ణాంకం నిలువు = 0; నిలువు < 16; ++నిలువు) {
                    s[(అడ్డువరుస_tile + r) * bs + (నిలువు_tile + నిలువు)] = sumf[r][నిలువు];
                }
            }
        }
    }
}
#అయితే_ముగింపు

} // extern "C"

స్థిర ఖండం_q8_0x4 make_ఖండం_q8_0x4(ఖండం_q8_0 * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q8_0x4 బహిర్గతం;

    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        బహిర్గతం.d[i] = లోపల[i].d;
    }

    స్థిరరాశి పూర్ణాంకం end = QK8_0 * 4 / blck_కొలత_interleave;
    కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
        పూర్ణాంకం src_id     = i % 4;
        పూర్ణాంకం src_స్థానభ్రంశం = (i / 4) * blck_కొలత_interleave;
        పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;
        జ్ఞాపకం_నకలు(&బహిర్గతం.qs[dst_స్థానభ్రంశం], &లోపల[src_id].qs[src_స్థానభ్రంశం], blck_కొలత_interleave);
    }
    తిరిగి బహిర్గతం;
}

స్థిర ఖండం_q1_0x4 make_ఖండం_q1_0x4(ఖండం_q1_0 * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q1_0x4 బహిర్గతం;

    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        బహిర్గతం.d[i] = లోపల[i].d;
    }

    GGML_ASSERT(blck_కొలత_interleave == 4 || blck_కొలత_interleave == 8);

    ఒకవేళ (blck_కొలత_interleave == 4) {
        కొరకు (పూర్ణాంకం k = 0; k < QK1_0 / QK8_0; ++k) {
            కొరకు (పూర్ణాంకం tile = 0; tile < QK8_0 / 4; ++tile) {
                uint8_t packed_lo = 0;
                uint8_t packed_hi = 0;

                స్థిరరాశి పూర్ణాంకం బరువు_పునాది = k * QK8_0 + tile * 4;
                కొరకు (పూర్ణాంకం pos = 0; pos < 4; ++pos) {
                    స్థిరరాశి పూర్ణాంకం బరువు_idx = బరువు_పునాది + pos;
                    స్థిరరాశి పూర్ణాంకం byte_idx   = బరువు_idx / 8;
                    స్థిరరాశి పూర్ణాంకం బిట్టు_idx    = బరువు_idx % 8;

                    packed_lo |= ((లోపల[0].qs[byte_idx] >> బిట్టు_idx) & 1u) << pos;
                    packed_lo |= ((లోపల[1].qs[byte_idx] >> బిట్టు_idx) & 1u) << (4 + pos);
                    packed_hi |= ((లోపల[2].qs[byte_idx] >> బిట్టు_idx) & 1u) << pos;
                    packed_hi |= ((లోపల[3].qs[byte_idx] >> బిట్టు_idx) & 1u) << (4 + pos);
                }

                బహిర్గతం.qs[k * 16 + 2 * tile + 0] = packed_lo;
                బహిర్గతం.qs[k * 16 + 2 * tile + 1] = packed_hi;
            }
        }
        తిరిగి బహిర్గతం;
    }

    కొరకు (పూర్ణాంకం byte_idx = 0; byte_idx < QK1_0 / 8; ++byte_idx) {
        బహిర్గతం.qs[byte_idx * 4 + 0] = లోపల[0].qs[byte_idx];
        బహిర్గతం.qs[byte_idx * 4 + 1] = లోపల[1].qs[byte_idx];
        బహిర్గతం.qs[byte_idx * 4 + 2] = లోపల[2].qs[byte_idx];
        బహిర్గతం.qs[byte_idx * 4 + 3] = లోపల[3].qs[byte_idx];
    }

    తిరిగి బహిర్గతం;
}

స్థిర ఖండం_q4_0x4 make_ఖండం_q4_0x4(ఖండం_q4_0 * లోపల, పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q4_0x4 బహిర్గతం;

    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        బహిర్గతం.d[i] = లోపల[i].d;
    }

    స్థిరరాశి పూర్ణాంకం end = QK4_0 * 2 / blck_కొలత_interleave;

    ఒకవేళ (blck_కొలత_interleave == 8) {
        స్థిరరాశి uint64_t xor_ముసుగు = 0x8888888888888888ULL;
        కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
            పూర్ణాంకం src_id = i % 4;
            పూర్ణాంకం src_స్థానభ్రంశం = (i / 4) * blck_కొలత_interleave;
            పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;

            uint64_t elems;
            // Using memcpy to avoid unaligned memory accesses
            జ్ఞాపకం_నకలు(&elems, &లోపల[src_id].qs[src_స్థానభ్రంశం], పరిమాణం(uint64_t));
            elems ^= xor_ముసుగు;
            జ్ఞాపకం_నకలు(&బహిర్గతం.qs[dst_స్థానభ్రంశం], &elems, పరిమాణం(uint64_t));
        }
    } లేకపోతే ఒకవేళ (blck_కొలత_interleave == 4) {
        స్థిరరాశి uint32_t xor_ముసుగు = 0x88888888;
        కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
            పూర్ణాంకం src_id = i % 4;
            పూర్ణాంకం src_స్థానభ్రంశం = (i / 4) * blck_కొలత_interleave;
            పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;

            uint32_t elems;
            జ్ఞాపకం_నకలు(&elems, &లోపల[src_id].qs[src_స్థానభ్రంశం], పరిమాణం(uint32_t));
            elems ^= xor_ముసుగు;
            జ్ఞాపకం_నకలు(&బహిర్గతం.qs[dst_స్థానభ్రంశం], &elems, పరిమాణం(uint32_t));
        }
    } లేకపోతే {
        GGML_ASSERT(అబద్ధం);
    }

    తిరిగి బహిర్గతం;
}

// interleave 8 block_q4_0s in blocks of blck_size_interleave
// returns an interleaved block_q4_0x8
// in the interleaved block_q4_0x8, place deltas for 8 block_q4_0 blocks
// first, then interleave quants from 8 block_q4_0s in blocks of blck_size_interleave
స్థిర ఖండం_q4_0x8 make_ఖండం_q4_0x8(ఖండం_q4_0 * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q4_0x8 బహిర్గతం;

    కొరకు (పూర్ణాంకం i = 0; i < 8; i++) {
        బహిర్గతం.d[i] = లోపల[i].d;
    }

    స్థిరరాశి పూర్ణాంకం end = QK4_0 * 4 / blck_కొలత_interleave;
    స్థిరరాశి uint64_t xor_ముసుగు = 0x8888888888888888ULL;

    కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
        పూర్ణాంకం src_id = i % 8;
        పూర్ణాంకం src_స్థానభ్రంశం = (i / 8) * blck_కొలత_interleave;
        పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;

        uint64_t elems;
        జ్ఞాపకం_నకలు(&elems, &లోపల[src_id].qs[src_స్థానభ్రంశం], పరిమాణం(uint64_t));
        elems ^= xor_ముసుగు;
        జ్ఞాపకం_నకలు(&బహిర్గతం.qs[dst_స్థానభ్రంశం], &elems, పరిమాణం(uint64_t));
    }

    తిరిగి బహిర్గతం;
}

స్థిర ఖండం_q4_0x16 make_ఖండం_q4_0x16(ఖండం_q4_0 * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q4_0x16 బహిర్గతం;

    కొరకు (పూర్ణాంకం i = 0; i < 16; i++) {
        బహిర్గతం.d[i] = లోపల[i].d;
    }

    స్థిరరాశి పూర్ణాంకం end = QK4_0 * 8 / blck_కొలత_interleave;

    ఒకవేళ (blck_కొలత_interleave == 1) {
        స్థిరరాశి uint8_t xor_ముసుగు = 0x88;
        కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
            పూర్ణాంకం src_id = i % 16;
            పూర్ణాంకం src_స్థానభ్రంశం = i / 16;
            పూర్ణాంకం dst_స్థానభ్రంశం = i;

            బహిర్గతం.qs[dst_స్థానభ్రంశం] = లోపల[src_id].qs[src_స్థానభ్రంశం] ^ xor_ముసుగు;
        }
    } లేకపోతే {
        GGML_ASSERT(అబద్ధం);
    }

    తిరిగి బహిర్గతం;
}

స్థిర ఖండం_q4_Kx8 make_ఖండం_q4_Kx8(ఖండం_q4_K * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q4_Kx8 బహిర్గతం;
    //Delta(scale) and dmin values of the eight Q4_K structures are copied onto the output interleaved structure
    కొరకు (పూర్ణాంకం i = 0; i < 8; i++) {
        బహిర్గతం.d[i] = లోపల[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    కొరకు (పూర్ణాంకం i = 0; i < 8; i++) {
        బహిర్గతం.dmin[i] = లోపల[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    స్థిరరాశి పూర్ణాంకం end = QK_K * 4 / blck_కొలత_interleave;

    // Interleave Q4_K quants by taking 8 bytes at a time
    కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
        పూర్ణాంకం src_id = i % 8;
        పూర్ణాంకం src_స్థానభ్రంశం = (i / 8) * blck_కొలత_interleave;
        పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;

        // buffer large enough for the max interleave block size (8 bytes)
        uint64_t elems;
        జ్ఞాపకం_నకలు(&elems, &లోపల[src_id].qs[src_స్థానభ్రంశం], blck_కొలత_interleave);
        జ్ఞాపకం_నకలు(&బహిర్గతం.qs[dst_స్థానభ్రంశం], &elems, blck_కొలత_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q4_K
    // Currently the Q4_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q4_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q4_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q4_K structures
    uint8_t s[8], m[8];

    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
            s[j] = లోపల[j].scales[i] & 63;
            m[j] = లోపల[j].scales[i + 4] & 63;
        }

        బహిర్గతం.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        బహిర్గతం.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        బహిర్గతం.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        బహిర్గతం.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
            s[j] = ((లోపల[j].scales[i] & 192) >> 2) | (లోపల[j].scales[i+8] & 15);
            m[j] = ((లోపల[j].scales[i + 4] & 192) >> 2) | ((లోపల[j].scales[i+8] & 240) >> 4);
        }

        బహిర్గతం.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        బహిర్గతం.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        బహిర్గతం.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        బహిర్గతం.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);

    }

    తిరిగి బహిర్గతం;
}

స్థిర ఖండం_q4_Kx16 make_ఖండం_q4_Kx16(ఖండం_q4_K * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q4_Kx16 బహిర్గతం;
    //Delta(scale) and dmin values of the 16 Q4_K structures are copied onto the output interleaved structure
    కొరకు (పూర్ణాంకం i = 0; i < 16; i++) {
        బహిర్గతం.d[i] = లోపల[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    కొరకు (పూర్ణాంకం i = 0; i < 16; i++) {
        బహిర్గతం.dmin[i] = లోపల[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    స్థిరరాశి పూర్ణాంకం end = QK_K * 8 / blck_కొలత_interleave;

    ఒకవేళ (blck_కొలత_interleave == 1) {
        కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
            పూర్ణాంకం src_id = i % 16;
            పూర్ణాంకం src_స్థానభ్రంశం = i / 16;
            పూర్ణాంకం dst_స్థానభ్రంశం = i;

            బహిర్గతం.qs[dst_స్థానభ్రంశం] = లోపల[src_id].qs[src_స్థానభ్రంశం];
        }

        // RVV repacking.
        //
        // Extract sums and mins for all 8 sub-blocks for each block of Q4_K.
        uint8_t s[128], m[128];
        కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
            కొరకు (పూర్ణాంకం j = 0; j < 16; j++) {
                s[i * 16 + j] = లోపల[j].scales[i] & 63;
                m[i * 16 + j] = లోపల[j].scales[i + 4] & 63;
            }
        }
        కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
            కొరకు (పూర్ణాంకం j = 0; j < 16; j++) {
                s[64 + i * 16 + j] = ((లోపల[j].scales[i] & 192) >> 2) | (లోపల[j].scales[i+8] & 15);
                m[64 + i * 16 + j] = ((లోపల[j].scales[i + 4] & 192) >> 2) | ((లోపల[j].scales[i+8] & 240) >> 4);
            }
        }

        కొరకు (పూర్ణాంకం i = 0; i < 128; i++) {
            బహిర్గతం.scales[i] = (s[i] & 15) | ((m[i] & 15) << 4);
        }
        కొరకు (పూర్ణాంకం i = 0; i < 64; i++) {
            బహిర్గతం.scales[128 + i] = ((s[i] & 48) >> 4) | ((m[i] & 48) >> 2) | (s[64 + i] & 48) | ((m[64 + i] & 48) << 2);
        }
    } లేకపోతే {
        GGML_ASSERT(అబద్ధం);
    }

    తిరిగి బహిర్గతం;
}

స్థిర ఖండం_q2_Kx8 make_ఖండం_q2_Kx8(ఖండం_q2_K * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q2_Kx8 బహిర్గతం;

    // Delta(scale) and dmin values of the eight Q2_K structures are copied onto the output interleaved structure
    కొరకు (పూర్ణాంకం i = 0; i < 8; i++) {
        బహిర్గతం.d[i] = లోపల[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    కొరకు (పూర్ణాంకం i = 0; i < 8; i++) {
        బహిర్గతం.dmin[i] = లోపల[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    స్థిరరాశి పూర్ణాంకం end = QK_K * 2 / blck_కొలత_interleave;

    // Interleave Q2_K quants by taking 8 bytes at a time
    కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
        పూర్ణాంకం src_id = i % 8;
        పూర్ణాంకం src_స్థానభ్రంశం = (i / 8) * blck_కొలత_interleave;
        పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;

        uint64_t elems;
        జ్ఞాపకం_నకలు(&elems, &లోపల[src_id].qs[src_స్థానభ్రంశం], పరిమాణం(uint64_t));
        జ్ఞాపకం_నకలు(&బహిర్గతం.qs[dst_స్థానభ్రంశం], &elems, పరిమాణం(uint64_t));
    }

    // The below logic is designed so as to unpack and rearrange scales and mins values in Q2_K
    // Currently the Q2_K structure has 16 scales and 16 mins packed in 16 bytes ( 4 bits for each value)
    // The output Q2_Kx8 structure has 128 bytes for storing scales and mins
    // Every 16 byte is packed such that it contains scales and mins for corresponding sub blocks from Q2_K structure
    // For eg - First 16 bytes contains 16 scales and 16 mins - each of first and second sub blocks from different Q2_K structures

    కొరకు (పూర్ణాంకం i = 0; i < 128; i++) {
        // Index for selecting which q2k super block
        పూర్ణాంకం src1 = (i % 16) / 2;
        // Index for selecting scale
        పూర్ణాంకం src2 = ((i / 16) * 2) + (i % 2);

        బహిర్గతం.scales[i] = లోపల[src1].scales[src2];
    }
    తిరిగి బహిర్గతం;
}

స్థిర ఖండం_q5_Kx8 make_ఖండం_q5_Kx8(ఖండం_q5_K * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q5_Kx8 బహిర్గతం;
    //Delta(scale) and dmin values of the eight Q5_K structures are copied onto the output interleaved structure
    కొరకు (పూర్ణాంకం i = 0; i < 8; i++) {
        బహిర్గతం.d[i] = లోపల[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
    }

    కొరకు (పూర్ణాంకం i = 0; i < 8; i++) {
        బహిర్గతం.dmin[i] = లోపల[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    స్థిరరాశి పూర్ణాంకం end = QK_K * 4 / blck_కొలత_interleave;

    // Interleave Q5_K quants by taking blck_size_interleave bytes at a time
    కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
        పూర్ణాంకం src_id     = i % 8;
        పూర్ణాంకం src_స్థానభ్రంశం = (i / 8) * blck_కొలత_interleave;
        పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;

        జ్ఞాపకం_నకలు(&బహిర్గతం.qs[dst_స్థానభ్రంశం], &లోపల[src_id].qs[src_స్థానభ్రంశం], blck_కొలత_interleave);
    }

    // Repeat for high bits with the same chunk size, since
    // the high bits are interleaved in Q5_K and the index is
    // qh_idx = (qs_idx % 32);
    // qh_val = qh[qh_idx] >> (qs_idx / 32);
    కొరకు (పూర్ణాంకం i = 0; i < end / 4; ++i) {
        పూర్ణాంకం src_id     = i % 8;
        పూర్ణాంకం src_స్థానభ్రంశం = (i / 8) * blck_కొలత_interleave;
        పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;

        జ్ఞాపకం_నకలు(&బహిర్గతం.qh[dst_స్థానభ్రంశం], &లోపల[src_id].qh[src_స్థానభ్రంశం], blck_కొలత_interleave);
    }

    // The below logic is copied over from Q4_K
    // The point is to unpack all the scales and mins for each sub block every time we load 12 bytes.
    // Currently the Q5_K structure has 8 scales and 8 mins packed in 12 bytes ( 6 bits for each value)
    // The output Q5_Kx8 structure has 96 bytes
    // Every 12 byte is packed such that it contains scales and mins for corresponding sub blocks from Q5_K structure
    // For eg - First 12 bytes contains 8 scales and 8 mins - each of first sub block from different Q5_K structures
    uint8_t s[8], m[8];

    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
            s[j] = లోపల[j].scales[i] & 63;
            m[j] = లోపల[j].scales[i + 4] & 63;
        }

        బహిర్గతం.scales[i * 12]      = (s[0] & 63) + ((s[4] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 1]  = (s[1] & 63) + ((s[5] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 2]  = (s[2] & 63) + ((s[6] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 3]  = (s[3] & 63) + ((s[7] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 4]  = (m[0] & 63) + ((m[4] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 5]  = (m[1] & 63) + ((m[5] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 6]  = (m[2] & 63) + ((m[6] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 7]  = (m[3] & 63) + ((m[7] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 8]  = (s[4] & 15) + ((m[4] & 15) << 4);
        బహిర్గతం.scales[i * 12 + 9]  = (s[5] & 15) + ((m[5] & 15) << 4);
        బహిర్గతం.scales[i * 12 + 10] = (s[6] & 15) + ((m[6] & 15) << 4);
        బహిర్గతం.scales[i * 12 + 11] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
            s[j] = ((లోపల[j].scales[i] & 192) >> 2) | (లోపల[j].scales[i + 8] & 15);
            m[j] = ((లోపల[j].scales[i + 4] & 192) >> 2) | ((లోపల[j].scales[i + 8] & 240) >> 4);
        }

        బహిర్గతం.scales[i * 12 + 48] = (s[0] & 63) + ((s[4] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 49] = (s[1] & 63) + ((s[5] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 50] = (s[2] & 63) + ((s[6] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 51] = (s[3] & 63) + ((s[7] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 52] = (m[0] & 63) + ((m[4] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 53] = (m[1] & 63) + ((m[5] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 54] = (m[2] & 63) + ((m[6] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 55] = (m[3] & 63) + ((m[7] & 48) << 2);
        బహిర్గతం.scales[i * 12 + 56] = (s[4] & 15) + ((m[4] & 15) << 4);
        బహిర్గతం.scales[i * 12 + 57] = (s[5] & 15) + ((m[5] & 15) << 4);
        బహిర్గతం.scales[i * 12 + 58] = (s[6] & 15) + ((m[6] & 15) << 4);
        బహిర్గతం.scales[i * 12 + 59] = (s[7] & 15) + ((m[7] & 15) << 4);
    }

    తిరిగి బహిర్గతం;
}

స్థిర ఖండం_q6_Kx8 make_ఖండం_q6_Kx8(ఖండం_q6_K * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q6_Kx8  బహిర్గతం;
    constexpr పూర్ణాంకం n_ఖండాలు = 8;  // Kx8
    కొరకు (పూర్ణాంకం i = 0; i < n_ఖండాలు; i++) {
        బహిర్గతం.d[i] = లోపల[i].d;
    }

    స్థిరరాశి పూర్ణాంకం చివర_ls = QK_K * 4 / blck_కొలత_interleave;
    // Interleave Q6_K quants by taking blck_size_interleave bytes at a time
    కొరకు (పూర్ణాంకం i = 0; i < చివర_ls; ++i) {
        పూర్ణాంకం src_id     = i % n_ఖండాలు;
        పూర్ణాంకం src_స్థానభ్రంశం = (i / n_ఖండాలు) * blck_కొలత_interleave;
        పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;

        uint64_t elem_ls;
        జ్ఞాపకం_నకలు(&elem_ls, &లోపల[src_id].ql[src_స్థానభ్రంశం], blck_కొలత_interleave);
        జ్ఞాపకం_నకలు(&బహిర్గతం.ql[dst_స్థానభ్రంశం], &elem_ls, blck_కొలత_interleave);
    }

    // Interleave high bits using same chunk size as low bits
    స్థిరరాశి పూర్ణాంకం చివర_hs = చివర_ls / 2;
    కొరకు (పూర్ణాంకం i = 0; i < చివర_hs; ++i) {
        పూర్ణాంకం src_id     = i % n_ఖండాలు;
        పూర్ణాంకం src_స్థానభ్రంశం = (i / n_ఖండాలు) * blck_కొలత_interleave;
        పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;

        uint64_t elem_hs;
        జ్ఞాపకం_నకలు(&elem_hs, &లోపల[src_id].qh[src_స్థానభ్రంశం], blck_కొలత_interleave);
        జ్ఞాపకం_నకలు(&బహిర్గతం.qh[dst_స్థానభ్రంశం], &elem_hs, blck_కొలత_interleave);
    }

    // The below logic is designed so as to unpack and rearrange scales in Q6_K
    // The output Q6_Kx8 structure interleaves the 8 bit scales in the same fashion as the quants
    // Q6_K structure has an 8-bit scale per 16 elements -> 16 scales
    // scales: [0 bl0 0 bl1 ... 0 bl7][1 bl0 ... 1 bl7] ... [15 bl0 ... 15 bl7]  (bl = block)
    constexpr పూర్ణాంకం n_scales = QK_K / 16;

    కొరకు (పూర్ణాంకం i = 0; i < n_ఖండాలు; i++) {
        కొరకు (పూర్ణాంకం j = 0; j < n_scales; j++) {
            బహిర్గతం.scales[j * n_ఖండాలు + i] = లోపల[i].scales[j];
        }
    }

    తిరిగి బహిర్గతం;
}

స్థిర ఖండం_q2_Kx16 make_ఖండం_q2_Kx16(స్థిరరాశి ఖండం_q2_K * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q2_Kx16 బహిర్గతం;
    constexpr పూర్ణాంకం N_నిలువుల_సంఖ్య = 16;

    // 1. Copy Super-Scales (d) and Super-Mins (dmin)
    కొరకు (పూర్ణాంకం i = 0; i < N_నిలువుల_సంఖ్య; i++) {
        బహిర్గతం.d[i]    = లోపల[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.d;
        బహిర్గతం.dmin[i] = లోపల[i].GGML_COMMON_AGGR_U.GGML_COMMON_AGGR_S.dmin;
    }

    // 2. Interleave Q2_K Data
    స్థిరరాశి పూర్ణాంకం బైట్లు_చొప్పున_నిలువు = 64;
    స్థిరరాశి పూర్ణాంకం మొత్తం_బైట్లు = N_నిలువుల_సంఖ్య * బైట్లు_చొప్పున_నిలువు;
    స్థిరరాశి పూర్ణాంకం end = మొత్తం_బైట్లు / blck_కొలత_interleave;

    కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
        పూర్ణాంకం src_నిలువు_id = i % N_నిలువుల_సంఖ్య;
        పూర్ణాంకం src_స్థానభ్రంశం = (i / N_నిలువుల_సంఖ్య) * blck_కొలత_interleave;
        పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;
        జ్ఞాపకం_నకలు(&బహిర్గతం.qs[dst_స్థానభ్రంశం], &లోపల[src_నిలువు_id].qs[src_స్థానభ్రంశం], blck_కొలత_interleave);
    }

    // 3. Repack Scales into the Optimized "Sequential-Parallel" Layout
    పూర్ణాంకం బహిర్గతం_idx = 0;

    // Arrays define the sub-block order for each group
    స్థిరరాశి పూర్ణాంకం even_దిగువ_sbs[]  = {0, 2, 4, 6};
    స్థిరరాశి పూర్ణాంకం odd_దిగువ_sbs[]   = {1, 3, 5, 7};
    స్థిరరాశి పూర్ణాంకం even_ఎగువ_sbs[] = {8, 10, 12, 14};
    స్థిరరాశి పూర్ణాంకం odd_ఎగువ_sbs[]  = {9, 11, 13, 15};

    // Pack Group 1: Even-Low
    కొరకు (పూర్ణాంకం sb : even_దిగువ_sbs) {
        కొరకు (పూర్ణాంకం నిలువు = 0; నిలువు < N_నిలువుల_సంఖ్య; నిలువు++) {
            బహిర్గతం.scales[బహిర్గతం_idx++] = లోపల[నిలువు].scales[sb];
        }
    }

    // Pack Group 2: Odd-Low
    కొరకు (పూర్ణాంకం sb : odd_దిగువ_sbs) {
        కొరకు (పూర్ణాంకం నిలువు = 0; నిలువు < N_నిలువుల_సంఖ్య; నిలువు++) {
            బహిర్గతం.scales[బహిర్గతం_idx++] = లోపల[నిలువు].scales[sb];
        }
    }

    // Pack Group 3: Even-High
    కొరకు (పూర్ణాంకం sb : even_ఎగువ_sbs) {
        కొరకు (పూర్ణాంకం నిలువు = 0; నిలువు < N_నిలువుల_సంఖ్య; నిలువు++) {
            బహిర్గతం.scales[బహిర్గతం_idx++] = లోపల[నిలువు].scales[sb];
        }
    }

    // Pack Group 4: Odd-High
    కొరకు (పూర్ణాంకం sb : odd_ఎగువ_sbs) {
        కొరకు (పూర్ణాంకం నిలువు = 0; నిలువు < N_నిలువుల_సంఖ్య; నిలువు++) {
            బహిర్గతం.scales[బహిర్గతం_idx++] = లోపల[నిలువు].scales[sb];
        }
    }

    తిరిగి బహిర్గతం;
}

స్థిర పూర్ణాంకం repack_q4_0_గమ్యం_q4_0_4_bl(నిర్మాణం ggml_tensor * t, పూర్ణాంకం interleave_ఖండం, స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం, size_t దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ఖండం == 4 || interleave_ఖండం == 8);
    constexpr పూర్ణాంకం nrows_interleaved = 4;

    ఖండం_q4_0x4 * dst = (ఖండం_q4_0x4 *)t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_0 * src = (స్థిరరాశి ఖండం_q4_0 *)దత్తాంశం;
    ఖండం_q4_0 dst_tmp[4];
    పూర్ణాంకం nrow = ggml_nrows(t);
    పూర్ణాంకం nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q4_0));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q4_0x4(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q4_K_గమ్యం_q4_K_8_bl(నిర్మాణం ggml_tensor * t, పూర్ణాంకం interleave_ఖండం, స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం, size_t దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    GGML_ASSERT(interleave_ఖండం == 8 || interleave_ఖండం == 4);
    constexpr పూర్ణాంకం nrows_interleaved = 8;

    ఖండం_q4_Kx8 * dst = (ఖండం_q4_Kx8*)t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_K * src = (స్థిరరాశి ఖండం_q4_K*) దత్తాంశం;
    ఖండం_q4_K dst_tmp[8];
    పూర్ణాంకం nrow = ggml_nrows(t);
    పూర్ణాంకం nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q4_K));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q4_Kx8(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q4_K_గమ్యం_q4_K_16_bl(నిర్మాణం ggml_tensor * t, పూర్ణాంకం interleave_ఖండం, స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం, size_t దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_K);
    constexpr పూర్ణాంకం nrows_interleaved = 16;

    ఖండం_q4_Kx16 * dst = (ఖండం_q4_Kx16*)t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_K * src = (స్థిరరాశి ఖండం_q4_K*) దత్తాంశం;
    ఖండం_q4_K dst_tmp[16];
    పూర్ణాంకం nrow = ggml_nrows(t);
    పూర్ణాంకం nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q4_K));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q4_Kx16(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q2_K_గమ్యం_q2_K_8_bl(నిర్మాణం ggml_tensor * t, పూర్ణాంకం interleave_ఖండం, స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం, size_t దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    GGML_ASSERT(interleave_ఖండం == 8);
    constexpr పూర్ణాంకం nrows_interleaved = 8;

    ఖండం_q2_Kx8 * dst = (ఖండం_q2_Kx8*)t->దత్తాంశం;
    స్థిరరాశి ఖండం_q2_K * src = (స్థిరరాశి ఖండం_q2_K*) దత్తాంశం;
    ఖండం_q2_K dst_tmp[8];
    పూర్ణాంకం nrow = ggml_nrows(t);
    పూర్ణాంకం nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q2_K));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q2_Kx8(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q2_K_గమ్యం_q2_K_16_bl(నిర్మాణం ggml_tensor * t, పూర్ణాంకం interleave_ఖండం, స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం, size_t దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q2_K);
    constexpr పూర్ణాంకం nrows_interleaved = 16;

    ఖండం_q2_Kx16 * dst = (ఖండం_q2_Kx16*)t->దత్తాంశం;
    స్థిరరాశి ఖండం_q2_K * src = (స్థిరరాశి ఖండం_q2_K*) దత్తాంశం;

    ఖండం_q2_K dst_tmp[nrows_interleaved];

    పూర్ణాంకం nrow = ggml_nrows(t);
    పూర్ణాంకం nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q2_K));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            // This loop gathers 16 separate blocks (one from each column)
            // that correspond to the same K-dimension chunk.
            కొరకు (పూర్ణాంకం i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }

            *dst++ = make_ఖండం_q2_Kx16(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q4_0_గమ్యం_q4_0_16_bl(నిర్మాణం ggml_tensor * t, పూర్ణాంకం interleave_ఖండం, స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం, size_t దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    constexpr పూర్ణాంకం nrows_interleaved = 16;

    ఖండం_q4_0x16 * dst = (ఖండం_q4_0x16*)t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_0 * src = (స్థిరరాశి ఖండం_q4_0*) దత్తాంశం;
    ఖండం_q4_0 dst_tmp[16];
    పూర్ణాంకం nrow = ggml_nrows(t);
    పూర్ణాంకం nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q4_0));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q4_0x16(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q5_K_గమ్యం_q5_K_8_bl(నిర్మాణం ggml_tensor *       t,
                                    పూర్ణాంకం                        interleave_ఖండం,
                                    స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                    size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q5_K);
    GGML_ASSERT(interleave_ఖండం == 4 || interleave_ఖండం == 8);
    constexpr పూర్ణాంకం nrows_interleaved = 8;

    ఖండం_q5_Kx8 *     dst = (ఖండం_q5_Kx8 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q5_K * src = (స్థిరరాశి ఖండం_q5_K *) దత్తాంశం;
    ఖండం_q5_K         dst_tmp[8];
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q5_K));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q5_Kx8(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;
}

స్థిర పూర్ణాంకం repack_q6_K_గమ్యం_q6_K_8_bl(నిర్మాణం ggml_tensor * t, పూర్ణాంకం interleave_ఖండం, స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం, size_t దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q6_K);
    GGML_ASSERT(interleave_ఖండం == 4 || interleave_ఖండం == 8);
    constexpr పూర్ణాంకం nrows_interleaved = 8;

    ఖండం_q6_Kx8 * dst = (ఖండం_q6_Kx8 *)t->దత్తాంశం;
    స్థిరరాశి ఖండం_q6_K * src = (స్థిరరాశి ఖండం_q6_K *) దత్తాంశం;
    ఖండం_q6_K dst_tmp[8];
    పూర్ణాంకం nrow = ggml_nrows(t);
    పూర్ణాంకం nblocks = t->ne[0] / QK_K;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q6_K));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q6_Kx8(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;
}

స్థిర పూర్ణాంకం repack_q4_0_గమ్యం_q4_0_8_bl(నిర్మాణం ggml_tensor * t, పూర్ణాంకం interleave_ఖండం, స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం, size_t దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q4_0);
    GGML_ASSERT(interleave_ఖండం == 8);
    constexpr పూర్ణాంకం nrows_interleaved = 8;

    ఖండం_q4_0x8 * dst = (ఖండం_q4_0x8*)t->దత్తాంశం;
    స్థిరరాశి ఖండం_q4_0 * src = (స్థిరరాశి ఖండం_q4_0*) దత్తాంశం;
    ఖండం_q4_0 dst_tmp[8];
    పూర్ణాంకం nrow = ggml_nrows(t);
    పూర్ణాంకం nblocks = t->ne[0] / QK4_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q4_0));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i  = 0; i < nrows_interleaved; i++ ) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q4_0x8(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర పూర్ణాంకం repack_q8_0_గమ్యం_q8_0_4_bl(నిర్మాణం ggml_tensor *       t,
                                    పూర్ణాంకం                        interleave_ఖండం,
                                    స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                    size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    GGML_ASSERT(interleave_ఖండం == 4 || interleave_ఖండం == 8);
    constexpr పూర్ణాంకం nrows_interleaved = 4;

    ఖండం_q8_0x4 *     dst = (ఖండం_q8_0x4 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q8_0 * src = (స్థిరరాశి ఖండం_q8_0 *) దత్తాంశం;
    ఖండం_q8_0         dst_tmp[4];
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q8_0));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q8_0x4(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;
}

స్థిర పూర్ణాంకం repack_q1_0_గమ్యం_q1_0_4_bl(నిర్మాణం ggml_tensor *       t,
                                    పూర్ణాంకం                        interleave_ఖండం,
                                    స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                    size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q1_0);
    GGML_ASSERT(interleave_ఖండం == 4 || interleave_ఖండం == 8);
    constexpr పూర్ణాంకం nrows_interleaved = 4;

    ఖండం_q1_0x4 *     dst = (ఖండం_q1_0x4 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q1_0 * src = (స్థిరరాశి ఖండం_q1_0 *) దత్తాంశం;
    ఖండం_q1_0         dst_tmp[4];
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK1_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q1_0));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_q1_0x4(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;
}

స్థిర ఖండం_q8_0x16 make_ఖండం_q8_0x16(ఖండం_q8_0 * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_q8_0x16 బహిర్గతం;

    కొరకు (పూర్ణాంకం i = 0; i < 16; i++) {
        బహిర్గతం.d[i] = లోపల[i].d;
    }

    స్థిరరాశి పూర్ణాంకం end = QK8_0 * 16 / blck_కొలత_interleave;

    ఒకవేళ (blck_కొలత_interleave == 1) {
        కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
            పూర్ణాంకం src_id     = i % 16;
            పూర్ణాంకం src_స్థానభ్రంశం = i / 16;
            పూర్ణాంకం dst_స్థానభ్రంశం = i;
            బహిర్గతం.qs[dst_స్థానభ్రంశం] = లోపల[src_id].qs[src_స్థానభ్రంశం];
        }
    } లేకపోతే {
        GGML_ASSERT(అబద్ధం);
    }

    తిరిగి బహిర్గతం;
}

స్థిర పూర్ణాంకం repack_q8_0_గమ్యం_q8_0_16_bl(నిర్మాణం ggml_tensor *       t,
                                    పూర్ణాంకం                        interleave_ఖండం,
                                    స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం,
                                    size_t                     దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_Q8_0);
    constexpr పూర్ణాంకం nrows_interleaved = 16;

    ఖండం_q8_0x16 *     dst = (ఖండం_q8_0x16 *) t->దత్తాంశం;
    స్థిరరాశి ఖండం_q8_0 * src = (స్థిరరాశి ఖండం_q8_0 *) దత్తాంశం;
    ఖండం_q8_0         dst_tmp[16];
    పూర్ణాంకం                nrow    = ggml_nrows(t);
    పూర్ణాంకం                nblocks = t->ne[0] / QK8_0;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_q8_0));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
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
}

స్థిర ఖండం_iq4_nlx4 make_ఖండం_iq4_nlx4(ఖండం_iq4_nl * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_iq4_nlx4 బహిర్గతం;

    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        బహిర్గతం.d[i] = లోపల[i].d;
    }

    స్థిరరాశి పూర్ణాంకం end = QK4_NL * 2 / blck_కొలత_interleave;

    // TODO: this branch seems wrong
    //if (blck_size_interleave == 8) {
    //    for (int i = 0; i < end; ++i) {
    //        int src_id = i % 4;
    //        int src_offset = (i / 4) * blck_size_interleave;
    //        int dst_offset = i * blck_size_interleave;

    //        // Using memcpy to avoid unaligned memory accesses
    //        memcpy(&out.qs[dst_offset], &in[src_id].qs[src_offset], sizeof(uint64_t));
    //    }
    //} else
    ఒకవేళ (blck_కొలత_interleave == 4) {
        కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
            పూర్ణాంకం src_id = i % 4;
            పూర్ణాంకం src_స్థానభ్రంశం = (i / 4) * blck_కొలత_interleave;
            పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;

            జ్ఞాపకం_నకలు(&బహిర్గతం.qs[dst_స్థానభ్రంశం], &లోపల[src_id].qs[src_స్థానభ్రంశం], పరిమాణం(uint32_t));
        }
    } లేకపోతే {
        GGML_ASSERT(అబద్ధం);
    }

    తిరిగి బహిర్గతం;
}

స్థిర పూర్ణాంకం repack_iq4_nl_గమ్యం_iq4_nl_4_bl(నిర్మాణం ggml_tensor * t, పూర్ణాంకం interleave_ఖండం, స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం, size_t దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_ఖండం == 4);

    స్థిరరాశి ఖండం_iq4_nl   * src = (స్థిరరాశి ఖండం_iq4_nl   *)దత్తాంశం;
          ఖండం_iq4_nlx4 * dst = (      ఖండం_iq4_nlx4 *)t->దత్తాంశం;

    ఖండం_iq4_nl dst_tmp[4];

    పూర్ణాంకం nrow = ggml_nrows(t);
    పూర్ణాంకం nrows_interleaved = 4;
    పూర్ణాంకం nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_iq4_nl));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_iq4_nlx4(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర ఖండం_iq4_nlx8 make_ఖండం_iq4_nlx8(ఖండం_iq4_nl * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_iq4_nlx8 బహిర్గతం;

    కొరకు (పూర్ణాంకం i = 0; i < 8; i++) {
        బహిర్గతం.d[i] = లోపల[i].d;
    }

    స్థిరరాశి పూర్ణాంకం end = QK4_NL * 4 / blck_కొలత_interleave;

    ఒకవేళ (blck_కొలత_interleave == 8) {
        కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
            పూర్ణాంకం src_id = i % 8;
            పూర్ణాంకం src_స్థానభ్రంశం = (i / 8) * blck_కొలత_interleave;
            పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;

            జ్ఞాపకం_నకలు(&బహిర్గతం.qs[dst_స్థానభ్రంశం], &లోపల[src_id].qs[src_స్థానభ్రంశం], పరిమాణం(uint64_t));
        }
    } లేకపోతే {
        GGML_ASSERT(అబద్ధం);
    }

    తిరిగి బహిర్గతం;
}

స్థిర పూర్ణాంకం repack_iq4_nl_గమ్యం_iq4_nl_8_bl(నిర్మాణం ggml_tensor * t, పూర్ణాంకం interleave_ఖండం, స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం, size_t దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_ఖండం == 8);

    స్థిరరాశి ఖండం_iq4_nl   * src = (స్థిరరాశి ఖండం_iq4_nl   *)దత్తాంశం;
          ఖండం_iq4_nlx8 * dst = (      ఖండం_iq4_nlx8 *)t->దత్తాంశం;

    ఖండం_iq4_nl dst_tmp[8];

    పూర్ణాంకం nrow = ggml_nrows(t);
    పూర్ణాంకం nrows_interleaved = 8;
    పూర్ణాంకం nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_iq4_nl));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_iq4_nlx8(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర ఖండం_iq4_nlx16 make_ఖండం_iq4_nlx16(ఖండం_iq4_nl * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_iq4_nlx16 బహిర్గతం;

    కొరకు (పూర్ణాంకం i = 0; i < 16; i++) {
        బహిర్గతం.d[i] = లోపల[i].d;
    }

    స్థిరరాశి పూర్ణాంకం end = QK4_NL * 8 / blck_కొలత_interleave;

    ఒకవేళ (blck_కొలత_interleave == 1) {
        కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
            పూర్ణాంకం src_id = i % 16;
            పూర్ణాంకం src_స్థానభ్రంశం = i / 16;
            పూర్ణాంకం dst_స్థానభ్రంశం = i;

            బహిర్గతం.qs[dst_స్థానభ్రంశం] = లోపల[src_id].qs[src_స్థానభ్రంశం];
        }
    } లేకపోతే {
        GGML_ASSERT(అబద్ధం);
    }

    తిరిగి బహిర్గతం;
}

స్థిర పూర్ణాంకం repack_iq4_nl_గమ్యం_iq4_nl_16_bl(నిర్మాణం ggml_tensor * t, పూర్ణాంకం interleave_ఖండం, స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం, size_t దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_IQ4_NL);
    GGML_ASSERT(interleave_ఖండం == 1);

    స్థిరరాశి ఖండం_iq4_nl    * src = (స్థిరరాశి ఖండం_iq4_nl   *)దత్తాంశం;
          ఖండం_iq4_nlx16 * dst = (      ఖండం_iq4_nlx16 *)t->దత్తాంశం;

    ఖండం_iq4_nl dst_tmp[16];

    పూర్ణాంకం nrow = ggml_nrows(t);
    పూర్ణాంకం nrows_interleaved = 16;
    పూర్ణాంకం nblocks = t->ne[0] / QK4_NL;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_iq4_nl));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_iq4_nlx16(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర ఖండం_mxfp4x4 make_ఖండం_mxfp4x4(ఖండం_mxfp4 * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_mxfp4x4 బహిర్గతం;

    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        బహిర్గతం.e[i] = లోపల[i].e;
    }

    స్థిరరాశి పూర్ణాంకం end = QK_MXFP4 * 2 / blck_కొలత_interleave;

    ఒకవేళ (blck_కొలత_interleave == 4) {
        కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
            పూర్ణాంకం src_id = i % 4;
            పూర్ణాంకం src_స్థానభ్రంశం = (i / 4) * blck_కొలత_interleave;
            పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;

            జ్ఞాపకం_నకలు(&బహిర్గతం.qs[dst_స్థానభ్రంశం], &లోపల[src_id].qs[src_స్థానభ్రంశం], పరిమాణం(uint32_t));
        }
    } లేకపోతే {
        GGML_ASSERT(అబద్ధం);
    }

    తిరిగి బహిర్గతం;
}

స్థిర పూర్ణాంకం repack_mxfp4_గమ్యం_mxfp4_4_bl(నిర్మాణం ggml_tensor * t, పూర్ణాంకం interleave_ఖండం, స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం, size_t దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_ఖండం == 4);

    స్థిరరాశి ఖండం_mxfp4   * src = (స్థిరరాశి ఖండం_mxfp4   *)దత్తాంశం;
          ఖండం_mxfp4x4 * dst = (      ఖండం_mxfp4x4 *)t->దత్తాంశం;

    ఖండం_mxfp4 dst_tmp[4];

    పూర్ణాంకం nrow = ggml_nrows(t);
    పూర్ణాంకం nrows_interleaved = 4;
    పూర్ణాంకం nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_mxfp4));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0 || t->ne[0] % 8 != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_mxfp4x4(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

స్థిర ఖండం_mxfp4x8 make_ఖండం_mxfp4x8(ఖండం_mxfp4 * లోపల, గుర్తులేని పూర్ణాంకం blck_కొలత_interleave) {
    ఖండం_mxfp4x8 బహిర్గతం;

    కొరకు (పూర్ణాంకం i = 0; i < 8; i++) {
        బహిర్గతం.e[i] = లోపల[i].e;
    }

    స్థిరరాశి పూర్ణాంకం end = QK_MXFP4 * 4 / blck_కొలత_interleave;

    ఒకవేళ (blck_కొలత_interleave == 8) {
        కొరకు (పూర్ణాంకం i = 0; i < end; ++i) {
            పూర్ణాంకం src_id = i % 8;
            పూర్ణాంకం src_స్థానభ్రంశం = (i / 8) * blck_కొలత_interleave;
            పూర్ణాంకం dst_స్థానభ్రంశం = i * blck_కొలత_interleave;

            జ్ఞాపకం_నకలు(&బహిర్గతం.qs[dst_స్థానభ్రంశం], &లోపల[src_id].qs[src_స్థానభ్రంశం], పరిమాణం(uint64_t));
        }
    } లేకపోతే {
        GGML_ASSERT(అబద్ధం);
    }

    తిరిగి బహిర్గతం;
}

స్థిర పూర్ణాంకం repack_mxfp4_గమ్యం_mxfp4_8_bl(నిర్మాణం ggml_tensor * t, పూర్ణాంకం interleave_ఖండం, స్థిరరాశి శూన్య * GGML_RESTRICT దత్తాంశం, size_t దత్తాంశం_కొలత) {
    GGML_ASSERT(t->type == GGML_TYPE_MXFP4);
    GGML_ASSERT(interleave_ఖండం == 8);

    స్థిరరాశి ఖండం_mxfp4   * src = (స్థిరరాశి ఖండం_mxfp4   *)దత్తాంశం;
          ఖండం_mxfp4x8 * dst = (      ఖండం_mxfp4x8 *)t->దత్తాంశం;

    ఖండం_mxfp4 dst_tmp[8];

    పూర్ణాంకం nrow = ggml_nrows(t);
    పూర్ణాంకం nrows_interleaved = 8;
    పూర్ణాంకం nblocks = t->ne[0] / QK_MXFP4;

    GGML_ASSERT(దత్తాంశం_కొలత == nrow * nblocks * పరిమాణం(ఖండం_mxfp4));

    ఒకవేళ (t->ne[1] % nrows_interleaved != 0) {
        తిరిగి -1;
    }

    కొరకు (పూర్ణాంకం b = 0; b < nrow; b += nrows_interleaved) {
        కొరకు (int64_t x = 0; x < nblocks; x++) {
            కొరకు (పూర్ణాంకం i = 0; i < nrows_interleaved; i++) {
                dst_tmp[i] = src[x + i * nblocks];
            }
            *dst++ = make_ఖండం_mxfp4x8(dst_tmp, interleave_ఖండం);
        }
        src += nrows_interleaved * nblocks;
    }
    తిరిగి 0;

    GGML_వాడని(దత్తాంశం_కొలత);
}

నామాకాశ ggml::cpu::repack {
// repack
మూస <రకం_పేరు BLOC_TYPE, int64_t INTER_కొలత, int64_t NB_నిలువుల_సంఖ్య>
పూర్ణాంకం repack(నిర్మాణం ggml_tensor *, స్థిరరాశి శూన్య *, size_t);

// TODO: generalise.
మూస <> పూర్ణాంకం repack<ఖండం_q1_0, 4, 4>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q1_0_గమ్యం_q1_0_4_bl(t, 4, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q1_0, 8, 4>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q1_0_గమ్యం_q1_0_4_bl(t, 8, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q4_0, 4, 4>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q4_0_గమ్యం_q4_0_4_bl(t, 4, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q4_0, 8, 4>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q4_0_గమ్యం_q4_0_4_bl(t, 8, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q4_0, 8, 8>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q4_0_గమ్యం_q4_0_8_bl(t, 8, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q4_K, 8, 8>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q4_K_గమ్యం_q4_K_8_bl(t, 8, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q4_K, 4, 8>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q4_K_గమ్యం_q4_K_8_bl(t, 4, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q2_K, 8, 8>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q2_K_గమ్యం_q2_K_8_bl(t, 8, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q5_K, 4, 8>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q5_K_గమ్యం_q5_K_8_bl(t, 4, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q5_K, 8, 8>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q5_K_గమ్యం_q5_K_8_bl(t, 8, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q6_K, 4, 8>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q6_K_గమ్యం_q6_K_8_bl(t, 4, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q6_K, 8, 8>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q6_K_గమ్యం_q6_K_8_bl(t, 8, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_iq4_nl, 4, 4>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_iq4_nl_గమ్యం_iq4_nl_4_bl(t, 4, దత్తాంశం, దత్తాంశం_కొలత);
}

// TODO: needs to be revisited
//template <> int repack<block_iq4_nl, 8, 4>(struct ggml_tensor * t, const void * data, size_t data_size) {
//    return repack_iq4_nl_to_iq4_nl_4_bl(t, 8, data, data_size);
//}

మూస <> పూర్ణాంకం repack<ఖండం_iq4_nl, 8, 8>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_iq4_nl_గమ్యం_iq4_nl_8_bl(t, 8, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_mxfp4, 4, 4>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_mxfp4_గమ్యం_mxfp4_4_bl(t, 4, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_mxfp4, 8, 8>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_mxfp4_గమ్యం_mxfp4_8_bl(t, 8, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q8_0, 4, 4>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q8_0_గమ్యం_q8_0_4_bl(t, 4, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q8_0, 8, 4>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q8_0_గమ్యం_q8_0_4_bl(t, 8, దత్తాంశం, దత్తాంశం_కొలత);
}

#ఒకవేళ defined __riscv_zvfh
మూస <> పూర్ణాంకం repack<ఖండం_q4_0, 1, 16>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q4_0_గమ్యం_q4_0_16_bl(t, 1, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q4_K, 1, 16>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q4_K_గమ్యం_q4_K_16_bl(t, 1, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_iq4_nl, 1, 16>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_iq4_nl_గమ్యం_iq4_nl_16_bl(t, 1, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q8_0, 1, 16>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q8_0_గమ్యం_q8_0_16_bl(t, 1, దత్తాంశం, దత్తాంశం_కొలత);
}

మూస <> పూర్ణాంకం repack<ఖండం_q2_K, 1, 16>(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) {
    తిరిగి repack_q2_K_గమ్యం_q2_K_16_bl(t, 1, దత్తాంశం, దత్తాంశం_కొలత);
}
#అయితే_ముగింపు

// gemv
మూస <రకం_పేరు BLOC_TYPE, int64_t INTER_కొలత, int64_t NB_నిలువుల_సంఖ్య, ggml_type PARAM_TYPE>
శూన్య gemv(పూర్ణాంకం, దశాంశం *, size_t, స్థిరరాశి శూన్య *, స్థిరరాశి శూన్య *, పూర్ణాంకం, పూర్ణాంకం);

మూస <> శూన్య gemv<ఖండం_q1_0, 4, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q1_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q1_0, 8, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q1_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q4_0, 4, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q4_0, 8, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q4_0, 8, 8, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <>
శూన్య gemv<ఖండం_q2_K, 8, 8, GGML_TYPE_Q8_K>(పూర్ణాంకం          n,
                                            దశాంశం *      s,
                                            size_t       bs,
                                            స్థిరరాశి శూన్య * vx,
                                            స్థిరరాశి శూన్య * vy,
                                            పూర్ణాంకం          nr,
                                            పూర్ణాంకం          nc) {
    ggml_gemv_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q4_K, 4, 8, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q4_K, 8, 8, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q5_K, 4, 8, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q5_K, 8, 8, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q6_K, 4, 8, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q6_K, 8, 8, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_mxfp4, 4, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_mxfp4, 8, 8, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q8_0, 4, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q8_0, 8, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#ఒకవేళ defined __riscv_zvfh
మూస <> శూన్య gemv<ఖండం_q4_0, 1, 16, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q4_K, 1, 16, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q8_0, 1, 16, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemv<ఖండం_q2_K, 1, 16, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemv_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#అయితే_ముగింపు

// gemm
మూస <రకం_పేరు BLOC_TYPE, int64_t INTER_కొలత, int64_t NB_నిలువుల_సంఖ్య, ggml_type PARAM_TYPE>
శూన్య gemm(పూర్ణాంకం, దశాంశం *, size_t, స్థిరరాశి శూన్య *, స్థిరరాశి శూన్య *, పూర్ణాంకం, పూర్ణాంకం);

మూస <> శూన్య gemm<ఖండం_q1_0, 4, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q1_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q1_0, 8, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q1_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q4_0, 4, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q4_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q4_0, 8, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q4_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <>
శూన్య gemm<ఖండం_q4_0, 8, 8, GGML_TYPE_Q8_0>(పూర్ణాంకం          n,
                                            దశాంశం *      s,
                                            size_t       bs,
                                            స్థిరరాశి శూన్య * vx,
                                            స్థిరరాశి శూన్య * vy,
                                            పూర్ణాంకం          nr,
                                            పూర్ణాంకం          nc) {
    ggml_gemm_q4_0_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q2_K, 8, 8, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q2_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q4_K, 4, 8, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q4_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q4_K, 8, 8, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q4_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q5_K, 4, 8, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q5_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q5_K, 8, 8, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q5_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q6_K, 4, 8, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q6_K_8x4_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q6_K, 8, 8, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q6_K_8x8_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_iq4_nl, 4, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_iq4_nl_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_iq4_nl, 8, 8, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_iq4_nl_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_mxfp4, 4, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_mxfp4_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_mxfp4, 8, 8, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_mxfp4_8x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q8_0, 4, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q8_0_4x4_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q8_0, 8, 4, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q8_0_4x8_q8_0(n, s, bs, vx, vy, nr, nc);
}

#ఒకవేళ defined __riscv_zvfh
మూస <> శూన్య gemm<ఖండం_q4_0, 1, 16, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q4_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q4_K, 1, 16, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q4_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_iq4_nl, 1, 16, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_iq4_nl_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q8_0, 1, 16, GGML_TYPE_Q8_0>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q8_0_16x1_q8_0(n, s, bs, vx, vy, nr, nc);
}

మూస <> శూన్య gemm<ఖండం_q2_K, 1, 16, GGML_TYPE_Q8_K>(పూర్ణాంకం n, దశాంశం * s, size_t bs, స్థిరరాశి శూన్య * vx, స్థిరరాశి శూన్య * vy, పూర్ణాంకం nr, పూర్ణాంకం nc) {
    ggml_gemm_q2_K_16x1_q8_K(n, s, bs, vx, vy, nr, nc);
}
#అయితే_ముగింపు

తరగతి tensor_traits_పునాది : బహిరంగ ggml::cpu::tensor_traits {
  బహిరంగ:
    మాయా పూర్ణాంకం repack(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) = 0;
};

మూస <రకం_పేరు BLOC_TYPE, int64_t INTER_కొలత, int64_t NB_నిలువుల_సంఖ్య, ggml_type PARAM_TYPE> తరగతి tensor_traits : బహిరంగ tensor_traits_పునాది {

    తార్కిక work_కొలత(పూర్ణాంకం /* n_threads */, స్థిరరాశి నిర్మాణం ggml_tensor * op, size_t & size) override {
        // not realy a GGML_TYPE_Q8_0 but same size.
        ఎంపిక (op->op) {
            స్థితి GGML_OP_MUL_MAT:
                {
                    size = ggml_అడ్డువరుస_కొలత(PARAM_TYPE, ggml_nelements(op->src[1]));
                    తిరిగి నిజం;
                }
            స్థితి GGML_OP_MUL_MAT_ID:
                {
                    size = ggml_అడ్డువరుస_కొలత(PARAM_TYPE, ggml_nelements(op->src[1]));
                    size = GGML_PAD(size, పరిమాణం(int64_t)); // + padding for next block.

                    స్థిరరాశి int64_t ne02 = op->src[0]->ne[2]; // n_as, n_expert
                    స్థిరరాశి int64_t ne12 = op->src[1]->ne[2]; // n_tokens

                    స్థిరరాశి size_t sizeof_mmid_అడ్డువరుస_mapping = పరిమాణం(int64_t);

                    size += sizeof_mmid_అడ్డువరుస_mapping*ne02*(ne12 + 1);

                    తిరిగి నిజం;
                }
            అప్రమేయ:
                // GGML_ABORT("fatal error");
                ఆపు;
        }
        తిరిగి అబద్ధం;
    }

    తార్కిక compute_forward(నిర్మాణం ggml_compute_params * params, నిర్మాణం ggml_tensor * op) override {
        ఎంపిక (op->op) {
            స్థితి GGML_OP_MUL_MAT:
                forward_mul_mat(params, op);
                తిరిగి నిజం;
            స్థితి GGML_OP_MUL_MAT_ID:
                forward_mul_mat_id(params, op);
                తిరిగి నిజం;
            అప్రమేయ:
                // GGML_ABORT("fatal error");
                ఆపు;
        }
        తిరిగి అబద్ధం;
    }

    శూన్య forward_mul_mat_one_chunk(ggml_compute_params * params,
                                   ggml_tensor *         op,
                                   int64_t               src0_ప్రారంభం,
                                   int64_t               src0_చివర,
                                   int64_t               src1_ప్రారంభం,
                                   int64_t               src1_చివర) {
        స్థిరరాశి ggml_tensor * src0 = op->src[0];
        స్థిరరాశి ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ద్విమాన_OP_LOCALS

        స్థిరరాశి size_t src1_నిలువు_stride = ggml_అడ్డువరుస_కొలత(PARAM_TYPE, ne10);

        GGML_ASSERT(ne03 == 1 && ne13 == 1);
        GGML_ASSERT(ne12 % ne02 == 0);
        స్థిరరాశి int64_t r2 = ne12 / ne02;

        స్థిరరాశి int64_t i12 = src1_ప్రారంభం / ne1;
        స్థిరరాశి int64_t i11 = src1_ప్రారంభం - i12 * ne1;

        // Determine batch index
        స్థిరరాశి int64_t i02 = i12 / r2;

        స్థిరరాశి int64_t i1 = i11;
        స్థిరరాశి int64_t i2 = i12;

        స్థిరరాశి అక్షరం * src0_ptr = (స్థిరరాశి అక్షరం *) src0->దత్తాంశం + i02 * nb02;
        స్థిరరాశి అక్షరం * src1_ptr = (స్థిరరాశి అక్షరం *) params->wdata + (i11 + i12 * ne11) * src1_నిలువు_stride;
        అక్షరం *       dst_ptr  = ((అక్షరం *) dst->దత్తాంశం + (i1 * nb1 + i2 * nb2));

        స్థిరరాశి int64_t nrows = src1_చివర - src1_ప్రారంభం;
        స్థిరరాశి int64_t ncols = src0_చివర - src0_ప్రారంభం;

        GGML_ASSERT(src1_ptr + src1_నిలువు_stride * nrows <= (స్థిరరాశి అక్షరం *) params->wdata + params->wsize);

        // If there are more than three rows in src1, use gemm; otherwise, use gemv.
        ఒకవేళ (nrows > 3) {
            gemm<BLOC_TYPE, INTER_కొలత, NB_నిలువుల_సంఖ్య, PARAM_TYPE>(ne00, (దశాంశం *) (dst_ptr) + src0_ప్రారంభం, nb1 / nb0,
                                                             src0_ptr + src0_ప్రారంభం * nb01, src1_ptr,
                                                             nrows - (nrows % 4), ncols);
        }
        కొరకు (పూర్ణాంకం iter = nrows - (nrows % 4); iter < nrows; iter++) {
            gemv<BLOC_TYPE, INTER_కొలత, NB_నిలువుల_సంఖ్య, PARAM_TYPE>(ne00, (దశాంశం *) (dst_ptr + (iter * nb1)) + src0_ప్రారంభం,
                                                             ne01, src0_ptr + src0_ప్రారంభం * nb01,
                                                             src1_ptr + (src1_నిలువు_stride * iter), 1 /* nrows */, ncols);
        }
    }

    శూన్య forward_mul_mat(ggml_compute_params * params, ggml_tensor * op) {
        స్థిరరాశి ggml_tensor * src0 = op->src[0];
        స్థిరరాశి ggml_tensor * src1 = op->src[1];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ద్విమాన_OP_LOCALS

        స్థిరరాశి పూర్ణాంకం ith = params->ith;
        స్థిరరాశి పూర్ణాంకం nth = params->nth;

        GGML_ASSERT(ne0 == ne01);
        GGML_ASSERT(ne1 == ne11);
        GGML_ASSERT(ne2 == ne12);
        GGML_ASSERT(ne3 == ne13);

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        // TODO: General batched mul mat for 4D tensors
        // Currently only supports 3D tensors
        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3 == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        GGML_ASSERT(ggml_n_dims(op->src[0]) == 2);
        // GGML_ASSERT(ggml_n_dims(op->src[1]) == 2);

        అక్షరం *       wdata = static_cast<అక్షరం *>(params->wdata);
        స్థిరరాశి size_t nbw1  = ggml_అడ్డువరుస_కొలత(PARAM_TYPE, ne10);
        స్థిరరాశి size_t nbw2  = nbw1 * ne11;

        ప్రతిజ్ఞా(params->wsize >= nbw2 * ne12);

        స్థిరరాశి ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // INFO: Quantization is done in planes to avoid extra complexity in chunking.
        // Flattening dimensions not multiple of INTER_SIZE would require extra handling depending on how
        // the planes are broadcast.
        కొరకు (int64_t i12 = 0; i12 < ne12; i12++) {
            అక్షరం * దత్తాంశం_ptr  = (అక్షరం *) src1->దత్తాంశం + i12 * nb12;
            అక్షరం * wdata_ptr = wdata + i12 * nbw2;

            కొరకు (int64_t i11 = ith * 4; i11 < ne11 - ne11 % 4; i11 += nth * 4) {
                ggml_quantize_mat_t<INTER_కొలత, PARAM_TYPE>((దశాంశం *) (దత్తాంశం_ptr + i11 * nb11),
                                                            (శూన్య *) (wdata_ptr + i11 * nbw1), 4, ne10);
            }

            స్థిరరాశి int64_t i11_processed = ne11 - ne11 % 4;
            కొరకు (int64_t i11 = i11_processed + ith; i11 < ne11; i11 += nth) {
                from_float((దశాంశం *) (దత్తాంశం_ptr + i11 * nb11), (శూన్య *) (wdata_ptr + i11 * nbw1), ne10);
            }
        }

        // disable for NUMA
        స్థిరరాశి తార్కిక disable_chunking = ggml_is_numa();

        // 4x chunks per thread
        స్థిరరాశి int64_t nr0 = ggml_nrows(op->src[0]);

        పూర్ణాంకం     nth_scaled  = nth * 4;
        int64_t chunk_size0 = (nr0 + nth_scaled - 1) / nth_scaled;
        int64_t nchunk0     = (nr0 + chunk_size0 - 1) / chunk_size0;

        // src1 is chunked only by full planes.
        // When we flatten we need to address dimensions not multiple of the q8 INTER_SIZE
        // to route them thorugh GEMV.
        // nchunk1 = ne12 also avoids messing the chunking for models with no 3d tensors
        // to avoid affecting their performance
        int64_t nchunk1 = ne12;

        // Ensure minimum chunk size to avoid alignment issues with high thread counts
        // Minimum chunk size should be at least NB_COLS to prevent overlapping chunks after alignment
        స్థిరరాశి int64_t min_chunk_కొలత = NB_నిలువుల_సంఖ్య;
        ఒకవేళ (nchunk0 > 0 && (nr0 / nchunk0) < min_chunk_కొలత && nr0 >= min_chunk_కొలత) {
            nchunk0 = (nr0 + min_chunk_కొలత - 1) / min_chunk_కొలత;
        }

        int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        // Only increase nchunk0 to nth if it won't make chunks too small
        ఒకవేళ (nth == 1 || ((nchunk0 < nth || disable_chunking) && (nr0 + nth - 1) / nth >= min_chunk_కొలత)) {
            nchunk0 = nth;
            dr0 = (nr0 + nchunk0 - 1) / nchunk0;
        }

        // Ensure nchunk doesn't exceed the number of rows divided by minimum chunk size
        // This prevents creating too many tiny chunks that could overlap after alignment
        స్థిరరాశి int64_t max_nchunk = (nr0 + min_chunk_కొలత - 1) / min_chunk_కొలత;
        nchunk0                  = MIN(nchunk0, max_nchunk);

        ఒకవేళ (ith == 0) {
            // Every thread starts at ith, so the first unprocessed chunk is nth.  This save a bit of coordination right at the start.
            ggml_threadpool_chunk_సమితి(params->threadpool, nth);
        }

        ggml_barrier(params->threadpool);

        // The first chunk comes from our thread_id, the rest will get auto-assigned.
        పూర్ణాంకం current_chunk = ith;

        అయితే (current_chunk < nchunk0 * nchunk1) {
            స్థిరరాశి int64_t ith0 = current_chunk % nchunk0;
            స్థిరరాశి int64_t ith1 = current_chunk / nchunk0;

            int64_t src0_ప్రారంభం = dr0 * ith0;
            int64_t src0_చివర   = MIN(src0_ప్రారంభం + dr0, nr0);

            // full-plane range for src1
            int64_t src1_ప్రారంభం = ith1 * ne11;
            int64_t src1_చివర = (ith1 + 1) * ne11;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            // The chunk size limiting above ensures chunks are large enough to prevent overlaps
            src0_ప్రారంభం = (src0_ప్రారంభం % NB_నిలువుల_సంఖ్య) ? src0_ప్రారంభం + NB_నిలువుల_సంఖ్య - (src0_ప్రారంభం % NB_నిలువుల_సంఖ్య) : src0_ప్రారంభం;
            src0_చివర   = (src0_చివర % NB_నిలువుల_సంఖ్య) ? src0_చివర + NB_నిలువుల_సంఖ్య - (src0_చివర % NB_నిలువుల_సంఖ్య) : src0_చివర;
            src0_చివర   = MIN(src0_చివర, ne01);

            // Make sure current plane is the last one before exiting
            ఒకవేళ (src0_ప్రారంభం >= src0_చివర) {
                current_chunk = ggml_threadpool_chunk_కూడు(params->threadpool, 1);
                కొనసాగు;
            }

            forward_mul_mat_one_chunk(params, dst, src0_ప్రారంభం, src0_చివర, src1_ప్రారంభం, src1_చివర);

            current_chunk = ggml_threadpool_chunk_కూడు(params->threadpool, 1);
        }
    }

    శూన్య forward_mul_mat_id(ggml_compute_params * params, ggml_tensor * op) {
        స్థిరరాశి ggml_tensor * src0 = op->src[0];
        స్థిరరాశి ggml_tensor * src1 = op->src[1];
        స్థిరరాశి ggml_tensor * ids  = op->src[2];
        ggml_tensor *       dst  = op;

        GGML_TENSOR_ద్విమాన_OP_LOCALS

        స్థిరరాశి పూర్ణాంకం ith = params->ith;
        స్థిరరాశి పూర్ణాంకం nth = params->nth;

        స్థిరరాశి ggml_from_float_t from_float = ggml_get_type_traits_cpu(PARAM_TYPE)->from_float;

        // we don't support permuted src0 or src1
        GGML_ASSERT(nb00 == ggml_type_కొలత(src0->type));
        GGML_ASSERT(nb10 == ggml_type_కొలత(src1->type));

        // dst cannot be transposed or permuted
        GGML_ASSERT(nb0 == పరిమాణం(దశాంశం));
        GGML_ASSERT(nb0 <= nb1);
        GGML_ASSERT(nb1 <= nb2);
        GGML_ASSERT(nb2 <= nb3);

        GGML_ASSERT(ne03 == 1);
        GGML_ASSERT(ne13 == 1);
        GGML_ASSERT(ne3  == 1);

        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        // row groups
        స్థిరరాశి పూర్ణాంకం n_ids = ids->ne[0]; // n_expert_used
        స్థిరరాశి పూర్ణాంకం n_as  = ne02;       // n_expert

        స్థిరరాశి size_t nbw1 = ggml_అడ్డువరుస_కొలత(PARAM_TYPE, ne10);
        స్థిరరాశి size_t nbw2 = nbw1*ne11;
        స్థిరరాశి size_t nbw3 = nbw2*ne12;

        నిర్మాణం mmid_అడ్డువరుస_mapping {
            int32_t i1;
            int32_t i2;
        };

        GGML_ASSERT(params->wsize >=
                (GGML_PAD(nbw3, పరిమాణం(int64_t)) +
                 n_as*(ne12 + 1)*పరిమాణం(mmid_అడ్డువరుస_mapping))
                );

        స్వయంచాలక * wdata          = (అక్షరం *)params->wdata;
        స్వయంచాలక * wdata_src1_చివర = (అక్షరం *)wdata + GGML_PAD(nbw3, పరిమాణం(int64_t));

        // total of [n_as][ne12 + 1] elements of type mmid_row_mapping (2*int32_t = int64_t)
        స్వయంచాలక * మాతృక_అడ్డువరుస_counts = (int64_t *) (wdata_src1_చివర);                                        // [n_as]
        నిర్మాణం mmid_అడ్డువరుస_mapping * మాతృక_అడ్డువరుసలు = (నిర్మాణం mmid_అడ్డువరుస_mapping *) (మాతృక_అడ్డువరుస_counts + n_as); // [n_as][ne12]

        // src1: float32 => param type
        కొరకు (int64_t i12 = 0; i12 < ne12; ++i12) {
            కొరకు (int64_t i11 = ith; i11 < ne11; i11 += nth) {
                from_float((దశాంశం *)((అక్షరం *) src1->దత్తాంశం + i12 * nb12 + i11 * nb11),
                           (శూన్య *)               (wdata + i12 * nbw2 + i11 * nbw1),
                           ne10);
            }
        }

#నిర్వచించు MMID_మాతృక_అడ్డువరుస(అడ్డువరుస_id, i1) మాతృక_అడ్డువరుసలు[(అడ్డువరుస_id) * ne12 + (i1)]

        ఒకవేళ (ith == 0) {
            // initialize matrix_row_counts
            memset(మాతృక_అడ్డువరుస_counts, 0, n_as * పరిమాణం(int64_t));

            // group rows by src0 matrix
            కొరకు (int32_t iid1 = 0; iid1 < ids->ne[1]; ++iid1) {
                కొరకు (int32_t id = 0; id < n_ids; ++id) {
                    స్థిరరాశి int32_t i02 =
                        *(స్థిరరాశి int32_t *) ((స్థిరరాశి అక్షరం *) ids->దత్తాంశం + iid1 * ids->nb[1] + id * ids->nb[0]);

                    GGML_ASSERT(i02 >= 0 && i02 < n_as);

                    MMID_మాతృక_అడ్డువరుస(i02, మాతృక_అడ్డువరుస_counts[i02]) = { id, iid1 };
                    మాతృక_అడ్డువరుస_counts[i02] += 1;
                }
            }
        }

        ggml_barrier(params->threadpool);

        // compute each matrix multiplication in sequence
        కొరకు (పూర్ణాంకం cur_a = 0; cur_a < n_as; ++cur_a) {
            స్థిరరాశి int64_t cne1 = మాతృక_అడ్డువరుస_counts[cur_a];

            ఒకవేళ (cne1 == 0) {
                కొనసాగు;
            }

            స్థిరరాశి స్వయంచాలక * src0_cur = (స్థిరరాశి అక్షరం *) src0->దత్తాంశం + cur_a*nb02;

            //const int64_t nr0 = ne01; // src0 rows
            స్థిరరాశి int64_t nr1 = cne1; // src1 rows

            int64_t src0_cur_ప్రారంభం = (ith * ne01) / nth;
            int64_t src0_cur_చివర   = ((ith + 1) * ne01) / nth;

            // Align boundaries to NB_COLS - round up to ensure all data is included
            src0_cur_ప్రారంభం = (src0_cur_ప్రారంభం % NB_నిలువుల_సంఖ్య) ? src0_cur_ప్రారంభం + NB_నిలువుల_సంఖ్య - (src0_cur_ప్రారంభం % NB_నిలువుల_సంఖ్య) : src0_cur_ప్రారంభం;
            src0_cur_చివర   = (src0_cur_చివర   % NB_నిలువుల_సంఖ్య) ? src0_cur_చివర   + NB_నిలువుల_సంఖ్య - (src0_cur_చివర   % NB_నిలువుల_సంఖ్య) : src0_cur_చివర;
            ఒకవేళ (src0_cur_చివర > ne01) {
                src0_cur_చివర = ne01;
            }

            ఒకవేళ (src0_cur_ప్రారంభం >= src0_cur_చివర) {
                తిరిగి;
            }

            కొరకు (పూర్ణాంకం ir1 = 0; ir1 < nr1; ir1++) {
                నిర్మాణం mmid_అడ్డువరుస_mapping అడ్డువరుస_mapping = MMID_మాతృక_అడ్డువరుస(cur_a, ir1);

                స్థిరరాశి పూర్ణాంకం id = అడ్డువరుస_mapping.i1;  // selected expert index

                స్థిరరాశి int64_t i11 = id % ne11;
                స్థిరరాశి int64_t i12 = అడ్డువరుస_mapping.i2;  // row index in src1

                స్థిరరాశి int64_t i1 = id;               // selected expert index
                స్థిరరాశి int64_t i2 = i12;              //అడ్డువరుస

                స్థిరరాశి స్వయంచాలక * src1_నిలువు = (స్థిరరాశి అక్షరం *) wdata + (i11 * nbw1 + i12 * nbw2);

                gemv<BLOC_TYPE, INTER_కొలత, NB_నిలువుల_సంఖ్య, PARAM_TYPE>(
                    ne00, (దశాంశం *) ((అక్షరం *) dst->దత్తాంశం + (i1 * nb1 + i2 * nb2)) + src0_cur_ప్రారంభం, ne01,
                    src0_cur + src0_cur_ప్రారంభం * nb01, src1_నిలువు, 1, src0_cur_చివర - src0_cur_ప్రారంభం);
            }
        }
#నిర్వచనం_రద్దు MMID_మాతృక_అడ్డువరుస
    }

    పూర్ణాంకం repack(నిర్మాణం ggml_tensor * t, స్థిరరాశి శూన్య * దత్తాంశం, size_t దత్తాంశం_కొలత) override {
        GGML_LOG_DEBUG("%s: repack tensor %s with %s_%dx%d\n", __func__, t->name, ggml_type_name(t->type),
                       (పూర్ణాంకం) NB_నిలువుల_సంఖ్య, (పూర్ణాంకం) INTER_కొలత);
        తిరిగి ggml::cpu::repack::repack<BLOC_TYPE, INTER_కొలత, NB_నిలువుల_సంఖ్య>(t, దత్తాంశం, దత్తాంశం_కొలత);
    }
};

}  // namespace ggml::cpu::repack

స్థిర స్థిరరాశి ggml::cpu::tensor_traits * ggml_repack_get_optimal_repack_type(స్థిరరాశి నిర్మాణం ggml_tensor * cur) {
    // instance for Q1_0
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q1_0, 4, 4, GGML_TYPE_Q8_0> q1_0_4x4_q8_0;
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q1_0, 8, 4, GGML_TYPE_Q8_0> q1_0_4x8_q8_0;

    // instance for Q4
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q4_0, 4, 4, GGML_TYPE_Q8_0> q4_0_4x4_q8_0;
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q4_0, 8, 4, GGML_TYPE_Q8_0> q4_0_4x8_q8_0;
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q4_0, 8, 8, GGML_TYPE_Q8_0> q4_0_8x8_q8_0;

    // instance for Q4_K
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q4_K, 4, 8, GGML_TYPE_Q8_K> q4_K_8x4_q8_K;
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q4_K, 8, 8, GGML_TYPE_Q8_K> q4_K_8x8_q8_K;

    // instance for Q5_K
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q5_K, 4, 8, GGML_TYPE_Q8_K> q5_K_8x4_q8_K;
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q5_K, 8, 8, GGML_TYPE_Q8_K> q5_K_8x8_q8_K;

    // instance for Q6_K
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q6_K, 4, 8, GGML_TYPE_Q8_K> q6_K_8x4_q8_K;
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q6_K, 8, 8, GGML_TYPE_Q8_K> q6_K_8x8_q8_K;

    // instance for Q2
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q2_K, 8, 8, GGML_TYPE_Q8_K> q2_K_8x8_q8_K;

    // instance for IQ4
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_iq4_nl, 4, 4, GGML_TYPE_Q8_0> iq4_nl_4x4_q8_0;
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_iq4_nl, 8, 8, GGML_TYPE_Q8_0> iq4_nl_8x8_q8_0;

    // instance for MXFP4
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_mxfp4, 4, 4, GGML_TYPE_Q8_0> mxfp4_4x4_q8_0;
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_mxfp4, 8, 8, GGML_TYPE_Q8_0> mxfp4_8x8_q8_0;

    // instance for Q8_0
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q8_0, 4, 4, GGML_TYPE_Q8_0> q8_0_4x4_q8_0;
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q8_0, 8, 4, GGML_TYPE_Q8_0> q8_0_4x8_q8_0;

    // instances for RISC-V
    //
    // These implement outer-product style matrix multiplication kernels with
    // an interleave of 1.
#ఒకవేళ defined __riscv_zvfh
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q4_0, 1, 16, GGML_TYPE_Q8_0> q4_0_16x1_q8_0;
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q4_K, 1, 16, GGML_TYPE_Q8_K> q4_K_16x1_q8_K;
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_iq4_nl, 1, 16, GGML_TYPE_Q8_0> iq4_nl_16x1_q8_0;
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q8_0, 1, 16, GGML_TYPE_Q8_0> q8_0_16x1_q8_0;
    స్థిర స్థిరరాశి ggml::cpu::repack::tensor_traits<ఖండం_q2_K, 1, 16, GGML_TYPE_Q8_K> q2_K_16x1_q8_K;
#అయితే_ముగింపు

    ఒకవేళ (cur->type == GGML_TYPE_Q4_0) {
        ఒకవేళ (ggml_cpu_has_avx2() || (ggml_cpu_has_sve() && ggml_cpu_has_matmul_int8() && ggml_cpu_get_sve_cnt() == QK8_0)) {
            ఒకవేళ (cur->ne[1] % 8 == 0) {
                తిరిగి &q4_0_8x8_q8_0;
            }
        }
        ఒకవేళ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            ఒకవేళ (cur->ne[1] % 4 == 0) {
                తిరిగి &q4_0_4x8_q8_0;
            }
        }
        ఒకవేళ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ఒకవేళ (cur->ne[1] % 4 == 0) {
                తిరిగి &q4_0_4x4_q8_0;
            }
        }
        ఒకవేళ (ggml_cpu_has_vxe()) {
            ఒకవేళ (cur->ne[1] % 4 == 0) {
                తిరిగి &q4_0_4x4_q8_0;
            }
        }
        ఒకవేళ (ggml_cpu_has_riscv_v()) {
            #ఒకవేళ defined __riscv_zvfh
            ఎంపిక (__riscv_vlenb() * 8) {
                స్థితి 128:  { ఆపు; } // TODO
                స్థితి 256:  { ఒకవేళ (cur->ne[1] % 16 == 0) { తిరిగి &q4_0_16x1_q8_0; } ఆపు; }
                స్థితి 512:  { ఆపు; } // TODO
                స్థితి 1024: { ఆపు; } // TODO
                అప్రమేయ:   { తిరిగి శూన్యసూచిక; }
            }
            #అయితే_ముగింపు
        }
    } లేకపోతే ఒకవేళ (cur->type == GGML_TYPE_Q4_K) {
        ఒకవేళ (ggml_cpu_has_avx2()) {
            ఒకవేళ (cur->ne[1] % 8 == 0) {
                తిరిగి &q4_K_8x8_q8_K;
            }
        }
        ఒకవేళ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            ఒకవేళ (cur->ne[1] % 8 == 0) {
                తిరిగి &q4_K_8x8_q8_K;
            }
        }
        ఒకవేళ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ఒకవేళ (cur->ne[1] % 8 == 0) {
                తిరిగి &q4_K_8x4_q8_K;
            }
        }
        ఒకవేళ (ggml_cpu_has_riscv_v()) {
            #ఒకవేళ defined __riscv_zvfh
            ఎంపిక (__riscv_vlenb() * 8) {
                స్థితి 128:  { ఆపు; } // TODO
                స్థితి 256:  { ఒకవేళ (cur->ne[1] % 16 == 0) { తిరిగి &q4_K_16x1_q8_K; } ఆపు; }
                స్థితి 512:  { ఆపు; } // TODO
                స్థితి 1024: { ఆపు; } // TODO
                అప్రమేయ:   { తిరిగి శూన్యసూచిక; }
            }
            #అయితే_ముగింపు
        }
    } లేకపోతే ఒకవేళ (cur->type == GGML_TYPE_Q2_K) {
        ఒకవేళ (ggml_cpu_has_avx512()) {
            ఒకవేళ (cur->ne[1] % 8 == 0) {
                తిరిగి &q2_K_8x8_q8_K;
            }
        }
        ఒకవేళ (ggml_cpu_has_riscv_v()) {
            #ఒకవేళ defined __riscv_zvfh
            ఎంపిక (__riscv_vlenb() * 8) {
                స్థితి 128:  { ఆపు; } // TODO
                స్థితి 256:  { ఒకవేళ (cur->ne[1] % 16 == 0) { తిరిగి &q2_K_16x1_q8_K; } ఆపు; }
                స్థితి 512:  { ఆపు; } // TODO
                స్థితి 1024: { ఆపు; } // TODO
                అప్రమేయ:   { తిరిగి శూన్యసూచిక; }
            }
            #అయితే_ముగింపు
        }
    } లేకపోతే ఒకవేళ (cur->type == GGML_TYPE_Q5_K) {
        ఒకవేళ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            ఒకవేళ (cur->ne[1] % 8 == 0) {
                తిరిగి &q5_K_8x8_q8_K;
            }
        }
        ఒకవేళ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ఒకవేళ (cur->ne[1] % 8 == 0) {
                తిరిగి &q5_K_8x4_q8_K;
            }
        }
    } లేకపోతే ఒకవేళ (cur->type == GGML_TYPE_Q6_K) {
        ఒకవేళ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            ఒకవేళ (cur->ne[1] % 8 == 0) {
                తిరిగి &q6_K_8x8_q8_K;
            }
        }
        ఒకవేళ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ఒకవేళ (cur->ne[1] % 8 == 0) {
                తిరిగి &q6_K_8x4_q8_K;
            }
        }
    } లేకపోతే ఒకవేళ (cur->type == GGML_TYPE_IQ4_NL) {
        ఒకవేళ (ggml_cpu_has_avx2()) {
            ఒకవేళ (cur->ne[1] % 8 == 0) {
                తిరిగి &iq4_nl_8x8_q8_0;
            }
        }
        ఒకవేళ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ఒకవేళ (cur->ne[1] % 4 == 0) {
                తిరిగి &iq4_nl_4x4_q8_0;
            }
        }
        ఒకవేళ (ggml_cpu_has_riscv_v()) {
            #ఒకవేళ defined __riscv_zvfh
            ఎంపిక (__riscv_vlenb() * 8) {
                స్థితి 128:  { ఆపు; } // TODO
                స్థితి 256:  { ఒకవేళ (cur->ne[1] % 16 == 0) { తిరిగి &iq4_nl_16x1_q8_0; } ఆపు; }
                స్థితి 512:  { ఆపు; } // TODO
                స్థితి 1024: { ఆపు; } // TODO
                అప్రమేయ:   { తిరిగి శూన్యసూచిక; }
            }
            #అయితే_ముగింపు
        }
    } లేకపోతే ఒకవేళ (cur->type == GGML_TYPE_MXFP4) {
        ఒకవేళ (ggml_cpu_has_avx2()) {
            ఒకవేళ (cur->ne[1] % 8 == 0) {
                తిరిగి &mxfp4_8x8_q8_0;
            }
        }
        ఒకవేళ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ఒకవేళ (cur->ne[1] % 4 == 0) {
                తిరిగి &mxfp4_4x4_q8_0;
            }
        }
    } లేకపోతే ఒకవేళ (cur->type == GGML_TYPE_Q8_0) {
        ఒకవేళ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            ఒకవేళ (cur->ne[1] % 4 == 0) {
                తిరిగి &q8_0_4x8_q8_0;
            }
        }
        ఒకవేళ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ఒకవేళ (cur->ne[1] % 4 == 0) {
                తిరిగి &q8_0_4x4_q8_0;
            }
        }
        ఒకవేళ (ggml_cpu_has_riscv_v()) {
            #ఒకవేళ defined __riscv_zvfh
            ఎంపిక (__riscv_vlenb() * 8) {
                స్థితి 128:  { ఆపు; } // TODO
                స్థితి 256:  { ఒకవేళ (cur->ne[1] % 16 == 0) { తిరిగి &q8_0_16x1_q8_0; } ఆపు; }
                స్థితి 512:  { ఆపు; } // TODO
                స్థితి 1024: { ఆపు; } // TODO
                అప్రమేయ:   { తిరిగి శూన్యసూచిక; }
            }
            #అయితే_ముగింపు
        }
    } లేకపోతే ఒకవేళ (cur->type == GGML_TYPE_Q1_0) {
        ఒకవేళ (ggml_cpu_has_neon() && ggml_cpu_has_matmul_int8()) {
            ఒకవేళ (cur->ne[1] % 4 == 0) {
                తిరిగి &q1_0_4x8_q8_0;
            }
        }
        ఒకవేళ (ggml_cpu_has_neon() && ggml_cpu_has_dotprod()) {
            ఒకవేళ (cur->ne[1] % 4 == 0) {
                తిరిగి &q1_0_4x4_q8_0;
            }
        }
    }

    తిరిగి శూన్యసూచిక;
}

స్థిర గణన ggml_స్థితి ggml_backend_cpu_repack_బఫరు_init_tensor(ggml_backend_బఫరు_t బఫరు, నిర్మాణం ggml_tensor * tensor) {
    tensor->extra = (శూన్య *) const_cast<ggml::cpu::tensor_traits *>(ggml_repack_get_optimal_repack_type(tensor));

    GGML_వాడని(బఫరు);
    తిరిగి GGML_స్థితి_SUCCESS;
}

స్థిర శూన్య ggml_backend_cpu_repack_బఫరు_సమితి_tensor(ggml_backend_బఫరు_t బఫరు, నిర్మాణం ggml_tensor * tensor,
                                                       స్థిరరాశి శూన్య * దత్తాంశం, size_t స్థానభ్రంశం, size_t size) {
    GGML_ASSERT(స్థానభ్రంశం == 0);
    GGML_ASSERT(size == ggml_nbytes(tensor));

    స్వయంచాలక tensor_traits = (ggml::cpu::repack::tensor_traits_పునాది *) tensor->extra;
    స్వయంచాలక OK            = tensor_traits->repack(tensor, దత్తాంశం, size);

    GGML_ASSERT(OK == 0);
    GGML_వాడని(బఫరు);
}

స్థిర స్థిరరాశి అక్షరం * ggml_backend_cpu_repack_బఫరు_type_get_name(ggml_backend_బఫరు_type_t buft) {
    తిరిగి "CPU_REPACK";

    GGML_వాడని(buft);
}

స్థిర ggml_backend_బఫరు_t ggml_backend_cpu_repack_బఫరు_type_alloc_బఫరు(ggml_backend_బఫరు_type_t buft, size_t size) {
    ggml_backend_బఫరు_t బఫరు = ggml_backend_buft_alloc_బఫరు(ggml_backend_cpu_బఫరు_type(), size);

    ఒకవేళ (బఫరు == శూన్యసూచిక) {
        తిరిగి శూన్యసూచిక;
    }

    బఫరు->buft              = buft;
    బఫరు->iface.init_tensor = ggml_backend_cpu_repack_బఫరు_init_tensor;
    బఫరు->iface.సమితి_tensor  = ggml_backend_cpu_repack_బఫరు_సమితి_tensor;
    బఫరు->iface.get_tensor  = శూన్యసూచిక;
    బఫరు->iface.cpy_tensor  = శూన్యసూచిక;
    తిరిగి బఫరు;
}

స్థిర size_t ggml_backend_cpu_repack_బఫరు_type_get_alignment(ggml_backend_బఫరు_type_t buft) {
    తిరిగి TENSOR_ALIGNMENT;

    GGML_వాడని(buft);
}

నామాకాశ ggml::cpu::repack {
తరగతి extra_బఫరు_type : ggml::cpu::extra_బఫరు_type {
    తార్కిక supports_op(ggml_backend_dev_t, స్థిరరాశి నిర్మాణం ggml_tensor * op) override {
        ఒకవేళ (    op->op == GGML_OP_MUL_MAT &&
                op->src[0]->బఫరు &&
                (ggml_n_dims(op->src[0]) == 2) &&
                op->src[0]->బఫరు->buft == ggml_backend_cpu_repack_బఫరు_type() &&
                ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            ఒకవేళ (op->src[1]->బఫరు && !ggml_backend_buft_is_host(op->src[1]->బఫరు->buft)) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (op->src[1]->type == GGML_TYPE_F32) {
                తిరిగి నిజం;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
            // may be possible if Q8_0 packed...
        } లేకపోతే ఒకవేళ (op->op == GGML_OP_MUL_MAT_ID
                && op->src[0]->బఫరు
                && (ggml_n_dims(op->src[0]) == 3)
                && op->src[0]->బఫరు->buft == ggml_backend_cpu_repack_బఫరు_type()
                && ggml_repack_get_optimal_repack_type(op->src[0])
                ) {
            ఒకవేళ (op->src[1]->బఫరు && !ggml_backend_buft_is_host(op->src[1]->బఫరు->buft)) {
                తిరిగి అబద్ధం;
            }
            ఒకవేళ (op->src[1]->type == GGML_TYPE_F32) {
                తిరిగి నిజం;
            }
            //if (op->src[1]->type == GGML_TYPE_Q8_0) {
            //    return true;
            //}
        }
        తిరిగి అబద్ధం;
    }

    ggml::cpu::tensor_traits * get_tensor_traits(స్థిరరాశి నిర్మాణం ggml_tensor * op) override {
        ఒకవేళ (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID) {
            ఒకవేళ (op->src[0]->బఫరు && op->src[0]->బఫరు->buft == ggml_backend_cpu_repack_బఫరు_type()) {
                తిరిగి (ggml::cpu::tensor_traits *) op->src[0]->extra;
            }
        }
        తిరిగి శూన్యసూచిక;
    }
};
}  // namespace ggml::cpu::repack

ggml_backend_బఫరు_type_t ggml_backend_cpu_repack_బఫరు_type(శూన్య) {
    స్థిర నిర్మాణం ggml_backend_బఫరు_type ggml_backend_cpu_బఫరు_type_repack = {
        /* .iface    = */ {
                           /* .get_name             = */ ggml_backend_cpu_repack_బఫరు_type_get_name,
                           /* .alloc_buffer         = */ ggml_backend_cpu_repack_బఫరు_type_alloc_బఫరు,
                           /* .alloc_buffer_n       = */ శూన్యసూచిక,
                           /* .get_alignment        = */ ggml_backend_cpu_repack_బఫరు_type_get_alignment,
                           /* .get_max_size         = */ శూన్యసూచిక,  // defaults to SIZE_MAX
                           /* .get_alloc_size       = */ శూన్యసూచిక,  // defaults to ggml_nbytes
                           /* .get_alloc_size_n     = */ ఖాళీ,
                           /* .is_host              = */ శూన్యసూచిక,
                           },
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cpu_reg(), 0),
        /* .context = */ కొత్త ggml::cpu::repack::extra_బఫరు_type(),
    };

    తిరిగి &ggml_backend_cpu_బఫరు_type_repack;
}
