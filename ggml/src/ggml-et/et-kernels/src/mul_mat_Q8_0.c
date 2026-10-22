//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#అన్తర్భావ "block_ops.h"
#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"
#అన్తర్భావ "quants.h"

#అన్తర్భావ <stdint.h>

#నిర్వచించు STRIDE_M                   2048 /* 32 shires x 32 minions x 2 harts */
#నిర్వచించు STRIDE_M_KSPLIT            1024 /* 32 shires x 32 minions (both harts share rows) */
#నిర్వచించు KSPLIT_MIN_K_ఖండాలు        256  /* K >= 8192 elements */
#నిర్వచించు KSPLIT_SMALL_అడ్డువరుసలు_K_ఖండాలు 64   /* K >= 2048 elements for very small M */
#నిర్వచించు KSPLIT_MAX_అడ్డువరుసలు            8    /* max rows per minion for K-split */
#నిర్వచించు TILE_KB                    256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */
#నిర్వచించు KSPLIT_సమూహం_అడ్డువరుసలు          4
#నిర్వచించు SIMPLE_X2_అడ్డువరుసలు             2

స్థిర అంతఃవరుస size_t tensor_బైట్లు(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    తిరిగి (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_mm_q8_params * params, శూన్య * env) {
    uint64_t hart_id = get_hart_id();

    // Matrix dimensions
    స్థిరరాశి int64_t K    = params->src0.ne[0];
    స్థిరరాశి int64_t M    = params->src0.ne[1];
    స్థిరరాశి int64_t N    = params->src1.ne[1];
    స్థిరరాశి int64_t ne02 = params->src0.ne[2];
    స్థిరరాశి int64_t ne03 = params->src0.ne[3];
    స్థిరరాశి int64_t ne12 = params->src1.ne[2];
    స్థిరరాశి int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    స్థిరరాశి size_t nb01 = params->src0.nb[1];
    స్థిరరాశి size_t nb02 = params->src0.nb[2];
    స్థిరరాశి size_t nb03 = params->src0.nb[3];

    స్థిరరాశి size_t nb11 = params->src1.nb[1];
    స్థిరరాశి size_t nb12 = params->src1.nb[2];
    స్థిరరాశి size_t nb13 = params->src1.nb[3];

    స్థిరరాశి size_t nbd1 = params->dst.nb[1];
    స్థిరరాశి size_t nbd2 = params->dst.nb[2];
    స్థిరరాశి size_t nbd3 = params->dst.nb[3];

    // Optional residual bias
    స్థిరరాశి అక్షరం * bias_పునాది = (స్థిరరాశి అక్షరం *) params->bias.దత్తాంశం;
    స్థిరరాశి size_t nbb1      = params->bias.nb[1];
    స్థిరరాశి size_t nbb2      = params->bias.nb[2];
    స్థిరరాశి size_t nbb3      = params->bias.nb[3];

    // Q8_0 block size is 32
    స్థిరరాశి int64_t K_ఖండాలు      = K / 32;
    స్థిరరాశి పూర్ణాంకం     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    స్థిరరాశి int64_t r2 = ne12 / ne02;
    స్థిరరాశి int64_t r3 = ne13 / ne03;

    // K-split decision
    స్థిరరాశి int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    స్థిరరాశి int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    స్థిరరాశి పూర్ణాంకం     is_hart1              = hart_id & 1;
    స్థిరరాశి int64_t అడ్డువరుసలు_చొప్పున_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    స్థిరరాశి int64_t k_half                = K_ఖండాలు / 2;
    స్థిరరాశి పూర్ణాంకం     use_ksplit_small_అడ్డువరుసలు = (అడ్డువరుసలు_చొప్పున_minion <= 2) && (K_ఖండాలు >= KSPLIT_SMALL_అడ్డువరుసలు_K_ఖండాలు);
    /*
     * K-split when K is large enough to benefit, and either:
     *   - few rows (≤4): always safe, proven working
     *   - more rows (5-8): only if each hart's half fits in one tile,
     *     otherwise L1 thrashing from 2 harts × 8 rows kills performance
     *
     * Also allow K-split earlier for the low-M regime (≤2 rows/minion). In
     * that case the simple row-striped path leaves half the machine idle, so
     * using both harts on each row pays off even for moderate K.
     */
    స్థిరరాశి పూర్ణాంకం     use_ksplit            = ((K_ఖండాలు >= KSPLIT_MIN_K_ఖండాలు) && (అడ్డువరుసలు_చొప్పున_minion <= KSPLIT_MAX_అడ్డువరుసలు) &&
                                           (అడ్డువరుసలు_చొప్పున_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_అడ్డువరుసలు;
    స్థిరరాశి పూర్ణాంకం     use_ksplit_సమూహం      = !use_ksplit && (K_ఖండాలు >= KSPLIT_MIN_K_ఖండాలు) && (అడ్డువరుసలు_చొప్పున_minion > 4) &&
                                          (అడ్డువరుసలు_చొప్పున_minion <= KSPLIT_MAX_అడ్డువరుసలు);

    evict_region_past_l2(params->src1.దత్తాంశం, tensor_బైట్లు(&params->src1));
    ఒకవేళ (params->bias.దత్తాంశం) {
        evict_region_past_l2(params->bias.దత్తాంశం, tensor_బైట్లు(&params->bias));
    }

    ఒకవేళ (use_ksplit) {
        /* Each hart processes half the K dimension */
        స్థిరరాశి int64_t k_ప్రారంభం = is_hart1 ? k_half : 0;
        స్థిరరాశి int64_t k_len   = is_hart1 ? (K_ఖండాలు - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        అస్థిర దశాంశం * l2scp_స్థానం = (అస్థిర దశాంశం *) et_shire_l2scp_local(local_minion * 64);

        కొరకు (int64_t i3 = 0; i3 < ne13; i3++) {
            స్థిరరాశి int64_t i03       = i3 / r3;
            స్థిరరాశి అక్షరం *  src0_ptr3 = (స్థిరరాశి అక్షరం *) params->src0.దత్తాంశం + i03 * nb03;
            స్థిరరాశి అక్షరం *  src1_ptr3 = (స్థిరరాశి అక్షరం *) params->src1.దత్తాంశం + i3 * nb13;
            అక్షరం *        dst_ptr3  = (అక్షరం *) params->dst.దత్తాంశం + i3 * nbd3;
            స్థిరరాశి అక్షరం *  bias_ptr3 = bias_పునాది ? bias_పునాది + i3 * nbb3 : (స్థిరరాశి అక్షరం *) 0;

            కొరకు (int64_t i2 = 0; i2 < ne12; i2++) {
                స్థిరరాశి int64_t i02       = i2 / r2;
                స్థిరరాశి అక్షరం *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                స్థిరరాశి అక్షరం *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                అక్షరం *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                స్థిరరాశి అక్షరం *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (స్థిరరాశి అక్షరం *) 0;

                కొరకు (int64_t n = 0; n < N; n++) {
                    స్థిరరాశి దశాంశం * b_నిలువు_పునాది = (స్థిరరాశి దశాంశం *) (src1_ptr2 + n * nb11);
                    స్థిరరాశి దశాంశం * bias_n     = bias_ptr2 ? (స్థిరరాశి దశాంశం *) (bias_ptr2 + n * nbb1) : (స్థిరరాశి దశాంశం *) 0;

                    కొరకు (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        స్థిరరాశి ఖండం_q8_0 * q_అడ్డువరుస = (స్థిరరాశి ఖండం_q8_0 *) (src0_ptr2 + m * nb01);

                        దశాంశం partial = compute_అడ్డువరుస_dot_q8_0(q_అడ్డువరుస + k_ప్రారంభం, b_నిలువు_పునాది + k_ప్రారంభం * 32, k_len);

                        ఒకవేళ (is_hart1) {
                            *l2scp_స్థానం = partial;
                            FENCE;
                            flush_గమ్యం_l2((స్థిరరాశి శూన్య *) l2scp_స్థానం, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } లేకపోతే {
                            et_sem_wait(ET_BARRIER_MINION);
                            దశాంశం other = *l2scp_స్థానం;
                            et_sem_post(ET_BARRIER_MINION);

                            దశాంశం * dst_నమోదు = (దశాంశం *) (dst_ptr2 + n * nbd1 + m * పరిమాణం(దశాంశం));
                            దశాంశం   కూడిక       = partial + other;
                            ఒకవేళ (bias_n) {
                                కూడిక += bias_n[m];
                            }
                            atomic_store_f32((అస్థిర దశాంశం *) dst_నమోదు, కూడిక);
                        }
                    }
                }
            }
        }
    } లేకపోతే ఒకవేళ (use_ksplit_సమూహం) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime.
         *
         * Both harts process the same 4-row group, each on half of K, and
         * exchange 4 partial sums once per group instead of once per row.
         * This keeps the K-split bandwidth benefit while cutting semaphore
         * traffic by 4x relative to the old per-row exchange.
         */
        స్థిరరాశి int64_t    k_ప్రారంభం    = is_hart1 ? k_half : 0;
        స్థిరరాశి int64_t    k_len      = is_hart1 ? (K_ఖండాలు - k_half) : k_half;
        అస్థిర దశాంశం * l2scp_స్థానం = (అస్థిర దశాంశం *) et_shire_l2scp_local(local_minion * 64);

        కొరకు (int64_t i3 = 0; i3 < ne13; i3++) {
            స్థిరరాశి int64_t i03       = i3 / r3;
            స్థిరరాశి అక్షరం *  src0_ptr3 = (స్థిరరాశి అక్షరం *) params->src0.దత్తాంశం + i03 * nb03;
            స్థిరరాశి అక్షరం *  src1_ptr3 = (స్థిరరాశి అక్షరం *) params->src1.దత్తాంశం + i3 * nb13;
            అక్షరం *        dst_ptr3  = (అక్షరం *) params->dst.దత్తాంశం + i3 * nbd3;
            స్థిరరాశి అక్షరం *  bias_ptr3 = bias_పునాది ? bias_పునాది + i3 * nbb3 : (స్థిరరాశి అక్షరం *) 0;

            కొరకు (int64_t i2 = 0; i2 < ne12; i2++) {
                స్థిరరాశి int64_t i02       = i2 / r2;
                స్థిరరాశి అక్షరం *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                స్థిరరాశి అక్షరం *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                అక్షరం *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                స్థిరరాశి అక్షరం *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (స్థిరరాశి అక్షరం *) 0;

                కొరకు (int64_t n = 0; n < N; n++) {
                    స్థిరరాశి దశాంశం * b_నిలువు_పునాది = (స్థిరరాశి దశాంశం *) (src1_ptr2 + n * nb11);
                    స్థిరరాశి దశాంశం * bias_n     = bias_ptr2 ? (స్థిరరాశి దశాంశం *) (bias_ptr2 + n * nbb1) : (స్థిరరాశి దశాంశం *) 0;

                    కొరకు (int64_t m_పునాది = minion_id; m_పునాది < M; m_పునాది += STRIDE_M_KSPLIT * KSPLIT_సమూహం_అడ్డువరుసలు) {
                        స్థిరరాశి int64_t m0 = m_పునాది;
                        స్థిరరాశి int64_t m1 = m0 + STRIDE_M_KSPLIT;
                        స్థిరరాశి int64_t m2 = m1 + STRIDE_M_KSPLIT;
                        స్థిరరాశి int64_t m3 = m2 + STRIDE_M_KSPLIT;

                        దశాంశం s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        కొరకు (int64_t kb = 0; kb < K_ఖండాలు; kb += TILE_KB) {
                            int64_t tile_len = k_len - kb;
                            ఒకవేళ (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            ఒకవేళ (tile_len <= 0) {
                                ఆపు;
                            }
                            స్థిరరాశి దశాంశం * b_tile = b_నిలువు_పునాది + (k_ప్రారంభం + kb) * 32;
                            స్థిరరాశి int64_t అడ్డువరుస_kb = k_ప్రారంభం + kb;

                            ఒకవేళ (m0 < M) {
                                s0 += compute_అడ్డువరుస_dot_q8_0((స్థిరరాశి ఖండం_q8_0 *) (src0_ptr2 + m0 * nb01) + అడ్డువరుస_kb,
                                                           b_tile, tile_len);
                            }
                            ఒకవేళ (m1 < M) {
                                s1 += compute_అడ్డువరుస_dot_q8_0((స్థిరరాశి ఖండం_q8_0 *) (src0_ptr2 + m1 * nb01) + అడ్డువరుస_kb,
                                                           b_tile, tile_len);
                            }
                            ఒకవేళ (m2 < M) {
                                s2 += compute_అడ్డువరుస_dot_q8_0((స్థిరరాశి ఖండం_q8_0 *) (src0_ptr2 + m2 * nb01) + అడ్డువరుస_kb,
                                                           b_tile, tile_len);
                            }
                            ఒకవేళ (m3 < M) {
                                s3 += compute_అడ్డువరుస_dot_q8_0((స్థిరరాశి ఖండం_q8_0 *) (src0_ptr2 + m3 * nb01) + అడ్డువరుస_kb,
                                                           b_tile, tile_len);
                            }
                        }

                        ఒకవేళ (is_hart1) {
                            l2scp_స్థానం[0] = s0;
                            l2scp_స్థానం[1] = s1;
                            l2scp_స్థానం[2] = s2;
                            l2scp_స్థానం[3] = s3;
                            FENCE;
                            flush_గమ్యం_l2((స్థిరరాశి శూన్య *) l2scp_స్థానం, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } లేకపోతే {
                            et_sem_wait(ET_BARRIER_MINION);
                            స్థిరరాశి దశాంశం p0 = l2scp_స్థానం[0];
                            స్థిరరాశి దశాంశం p1 = l2scp_స్థానం[1];
                            స్థిరరాశి దశాంశం p2 = l2scp_స్థానం[2];
                            స్థిరరాశి దశాంశం p3 = l2scp_స్థానం[3];
                            et_sem_post(ET_BARRIER_MINION);

                            దశాంశం *     c_పునాది = (దశాంశం *) (dst_ptr2 + n * nbd1);
                            స్థిరరాశి దశాంశం b0     = bias_n ? bias_n[m0] : 0.0f;
                            స్థిరరాశి దశాంశం b1     = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                            స్థిరరాశి దశాంశం b2     = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                            స్థిరరాశి దశాంశం b3     = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                            ఒకవేళ (m0 < M) {
                                atomic_store_f32((అస్థిర దశాంశం *) (c_పునాది + m0), s0 + p0 + b0);
                            }
                            ఒకవేళ (m1 < M) {
                                atomic_store_f32((అస్థిర దశాంశం *) (c_పునాది + m1), s1 + p1 + b1);
                            }
                            ఒకవేళ (m2 < M) {
                                atomic_store_f32((అస్థిర దశాంశం *) (c_పునాది + m2), s2 + p2 + b2);
                            }
                            ఒకవేళ (m3 < M) {
                                atomic_store_f32((అస్థిర దశాంశం *) (c_పునాది + m3), s3 + p3 + b3);
                            }
                        }
                    }
                }
            }
        }
    } లేకపోతే ఒకవేళ (K_ఖండాలు > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        కొరకు (int64_t i3 = 0; i3 < ne13; i3++) {
            స్థిరరాశి int64_t i03       = i3 / r3;
            స్థిరరాశి అక్షరం *  src0_ptr3 = (స్థిరరాశి అక్షరం *) params->src0.దత్తాంశం + i03 * nb03;
            స్థిరరాశి అక్షరం *  src1_ptr3 = (స్థిరరాశి అక్షరం *) params->src1.దత్తాంశం + i3 * nb13;
            అక్షరం *        dst_ptr3  = (అక్షరం *) params->dst.దత్తాంశం + i3 * nbd3;
            స్థిరరాశి అక్షరం *  bias_ptr3 = bias_పునాది ? bias_పునాది + i3 * nbb3 : (స్థిరరాశి అక్షరం *) 0;

            కొరకు (int64_t i2 = 0; i2 < ne12; i2++) {
                స్థిరరాశి int64_t i02       = i2 / r2;
                స్థిరరాశి అక్షరం *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                స్థిరరాశి అక్షరం *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                అక్షరం *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                స్థిరరాశి అక్షరం *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (స్థిరరాశి అక్షరం *) 0;

                కొరకు (int64_t n = 0; n < N; n++) {
                    స్థిరరాశి దశాంశం * b_నిలువు_పునాది = (స్థిరరాశి దశాంశం *) (src1_ptr2 + n * nb11);
                    స్థిరరాశి దశాంశం * bias_n     = bias_ptr2 ? (స్థిరరాశి దశాంశం *) (bias_ptr2 + n * nbb1) : (స్థిరరాశి దశాంశం *) 0;

                    కొరకు (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        స్థిరరాశి int64_t m1 = m0 + STRIDE_M;
                        స్థిరరాశి int64_t m2 = m0 + STRIDE_M * 2;
                        స్థిరరాశి int64_t m3 = m0 + STRIDE_M * 3;

                        దశాంశం s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        కొరకు (int64_t kb = 0; kb < K_ఖండాలు; kb += TILE_KB) {
                            int64_t tile_len = K_ఖండాలు - kb;
                            ఒకవేళ (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            స్థిరరాశి దశాంశం * b_tile = b_నిలువు_పునాది + kb * 32;

                            s0 += compute_అడ్డువరుస_dot_q8_0((స్థిరరాశి ఖండం_q8_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            ఒకవేళ (m1 < M) {
                                s1 += compute_అడ్డువరుస_dot_q8_0((స్థిరరాశి ఖండం_q8_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            ఒకవేళ (m2 < M) {
                                s2 += compute_అడ్డువరుస_dot_q8_0((స్థిరరాశి ఖండం_q8_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            ఒకవేళ (m3 < M) {
                                s3 += compute_అడ్డువరుస_dot_q8_0((స్థిరరాశి ఖండం_q8_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        దశాంశం *     dst_పునాది = (దశాంశం *) (dst_ptr2 + n * nbd1);
                        స్థిరరాశి దశాంశం b0       = bias_n ? bias_n[m0] : 0.0f;
                        స్థిరరాశి దశాంశం b1       = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                        స్థిరరాశి దశాంశం b2       = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                        స్థిరరాశి దశాంశం b3       = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                        atomic_store_f32((అస్థిర దశాంశం *) (dst_పునాది + m0), s0 + b0);
                        ఒకవేళ (m1 < M) {
                            atomic_store_f32((అస్థిర దశాంశం *) (dst_పునాది + m1), s1 + b1);
                        }
                        ఒకవేళ (m2 < M) {
                            atomic_store_f32((అస్థిర దశాంశం *) (dst_పునాది + m2), s2 + b2);
                        }
                        ఒకవేళ (m3 < M) {
                            atomic_store_f32((అస్థిర దశాంశం *) (dst_పునాది + m3), s3 + b3);
                        }
                    }
                }
            }
        }
    } లేకపోతే {
        /*
         * Simple path for small K.
         *
         * When `nb01` is 32-byte aligned, every row has the same block-alignment
         * pattern. That lets us compute two rows together and reuse each loaded
         * B chunk across both rows instead of reloading it in a second dot call.
         */
        కొరకు (int64_t i3 = 0; i3 < ne13; i3++) {
            స్థిరరాశి int64_t i03       = i3 / r3;
            స్థిరరాశి అక్షరం *  src0_ptr3 = (స్థిరరాశి అక్షరం *) params->src0.దత్తాంశం + i03 * nb03;
            స్థిరరాశి అక్షరం *  src1_ptr3 = (స్థిరరాశి అక్షరం *) params->src1.దత్తాంశం + i3 * nb13;
            అక్షరం *        dst_ptr3  = (అక్షరం *) params->dst.దత్తాంశం + i3 * nbd3;
            స్థిరరాశి అక్షరం *  bias_ptr3 = bias_పునాది ? bias_పునాది + i3 * nbb3 : (స్థిరరాశి అక్షరం *) 0;

            కొరకు (int64_t i2 = 0; i2 < ne12; i2++) {
                స్థిరరాశి int64_t i02       = i2 / r2;
                స్థిరరాశి అక్షరం *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                స్థిరరాశి అక్షరం *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                అక్షరం *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                స్థిరరాశి అక్షరం *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (స్థిరరాశి అక్షరం *) 0;

                కొరకు (int64_t n = 0; n < N; n++) {
                    స్థిరరాశి దశాంశం * b_నిలువు_పునాది = (స్థిరరాశి దశాంశం *) (src1_ptr2 + n * nb11);
                    స్థిరరాశి దశాంశం * bias_n     = bias_ptr2 ? (స్థిరరాశి దశాంశం *) (bias_ptr2 + n * nbb1) : (స్థిరరాశి దశాంశం *) 0;
                    q8_dot_state  q8_state;
                    q8_dot_begin(&q8_state);

                    ఒకవేళ (use_simple_x2) {
                        కొరకు (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_అడ్డువరుసలు) {
                            స్థిరరాశి int64_t      m1     = m0 + STRIDE_M;
                            స్థిరరాశి ఖండం_q8_0 * q_row0 = (స్థిరరాశి ఖండం_q8_0 *) (src0_ptr2 + m0 * nb01);

                            ఒకవేళ (m1 < M) {
                                స్థిరరాశి ఖండం_q8_0 * q_row1 = (స్థిరరాశి ఖండం_q8_0 *) (src0_ptr2 + m1 * nb01);
                                దశాంశం              s0, s1;
                                q8_dot_compute_x2_aligned(q_row0, q_row1, b_నిలువు_పునాది, K_ఖండాలు, &s0, &s1);

                                దశాంశం * dst0 = (దశాంశం *) (dst_ptr2 + n * nbd1 + m0 * పరిమాణం(దశాంశం));
                                దశాంశం * dst1 = (దశాంశం *) (dst_ptr2 + n * nbd1 + m1 * పరిమాణం(దశాంశం));
                                ఒకవేళ (bias_n) {
                                    s0 += bias_n[m0];
                                    s1 += bias_n[m1];
                                }
                                atomic_store_f32((అస్థిర దశాంశం *) dst0, s0);
                                atomic_store_f32((అస్థిర దశాంశం *) dst1, s1);
                            } లేకపోతే {
                                దశాంశం   కూడిక = q8_dot_compute(q_row0, b_నిలువు_పునాది, K_ఖండాలు);
                                దశాంశం * dst = (దశాంశం *) (dst_ptr2 + n * nbd1 + m0 * పరిమాణం(దశాంశం));
                                ఒకవేళ (bias_n) {
                                    కూడిక += bias_n[m0];
                                }
                                atomic_store_f32((అస్థిర దశాంశం *) dst, కూడిక);
                            }
                        }
                    } లేకపోతే {
                        కొరకు (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            స్థిరరాశి ఖండం_q8_0 * q_అడ్డువరుస = (స్థిరరాశి ఖండం_q8_0 *) (src0_ptr2 + m * nb01);

                            దశాంశం కూడిక = q8_dot_compute(q_అడ్డువరుస, b_నిలువు_పునాది, K_ఖండాలు);

                            దశాంశం * dst_నమోదు = (దశాంశం *) (dst_ptr2 + n * nbd1 + m * పరిమాణం(దశాంశం));
                            ఒకవేళ (bias_n) {
                                కూడిక += bias_n[m];
                            }
                            atomic_store_f32((అస్థిర దశాంశం *) dst_నమోదు, కూడిక);
                        }
                    }

                    q8_dot_చివర(&q8_state);
                }
            }
        }
    }

#నిర్వచితమైతే ET_UBERKERNEL
    FENCE;
    evict_region_past_l2(params->dst.దత్తాంశం, tensor_బైట్లు(&params->dst));
    WAIT_CACHEOPS;
    FENCE;
#అయితే_ముగింపు
    తిరిగి 0;
}
