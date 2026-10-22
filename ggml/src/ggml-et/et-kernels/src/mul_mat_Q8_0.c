//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#अन्तर्भाव "block_ops.h"
#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"
#अन्तर्भाव "quants.h"

#अन्तर्भाव <stdint.h>

#परिभाषय STRIDE_M                   2048 /* 32 shires x 32 minions x 2 harts */
#परिभाषय STRIDE_M_KSPLIT            1024 /* 32 shires x 32 minions (both harts share rows) */
#परिभाषय KSPLIT_MIN_K_खण्डानि        256  /* K >= 8192 elements */
#परिभाषय KSPLIT_SMALL_पङ्क्तिकाः_K_खण्डानि 64   /* K >= 2048 elements for very small M */
#परिभाषय KSPLIT_MAX_पङ्क्तिकाः            8    /* max rows per minion for K-split */
#परिभाषय TILE_KB                    256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */
#परिभाषय KSPLIT_गणः_पङ्क्तिकाः          4
#परिभाषय SIMPLE_X2_पङ्क्तिकाः             2

स्थिर अन्तःस्थ size_t tensor_अष्टकानि(ध्रुव संरचना ggml_tensor * t) {
    प्रत्यागम (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_mm_q8_params * params, शून्य * env) {
    uint64_t hart_id = get_hart_id();

    // Matrix dimensions
    ध्रुव int64_t K    = params->src0.ne[0];
    ध्रुव int64_t M    = params->src0.ne[1];
    ध्रुव int64_t N    = params->src1.ne[1];
    ध्रुव int64_t ne02 = params->src0.ne[2];
    ध्रुव int64_t ne03 = params->src0.ne[3];
    ध्रुव int64_t ne12 = params->src1.ne[2];
    ध्रुव int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    ध्रुव size_t nb01 = params->src0.nb[1];
    ध्रुव size_t nb02 = params->src0.nb[2];
    ध्रुव size_t nb03 = params->src0.nb[3];

    ध्रुव size_t nb11 = params->src1.nb[1];
    ध्रुव size_t nb12 = params->src1.nb[2];
    ध्रुव size_t nb13 = params->src1.nb[3];

    ध्रुव size_t nbd1 = params->dst.nb[1];
    ध्रुव size_t nbd2 = params->dst.nb[2];
    ध्रुव size_t nbd3 = params->dst.nb[3];

    // Optional residual bias
    ध्रुव वर्ण * bias_आधारः = (ध्रुव वर्ण *) params->bias.दत्त;
    ध्रुव size_t nbb1      = params->bias.nb[1];
    ध्रुव size_t nbb2      = params->bias.nb[2];
    ध्रुव size_t nbb3      = params->bias.nb[3];

    // Q8_0 block size is 32
    ध्रुव int64_t K_खण्डानि      = K / 32;
    ध्रुव पूर्णाङ्क     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    ध्रुव int64_t r2 = ne12 / ne02;
    ध्रुव int64_t r3 = ne13 / ne03;

    // K-split decision
    ध्रुव int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    ध्रुव int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    ध्रुव पूर्णाङ्क     is_hart1              = hart_id & 1;
    ध्रुव int64_t पङ्क्तिकाः_प्रत्येकम्_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    ध्रुव int64_t k_half                = K_खण्डानि / 2;
    ध्रुव पूर्णाङ्क     use_ksplit_small_पङ्क्तिकाः = (पङ्क्तिकाः_प्रत्येकम्_minion <= 2) && (K_खण्डानि >= KSPLIT_SMALL_पङ्क्तिकाः_K_खण्डानि);
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
    ध्रुव पूर्णाङ्क     use_ksplit            = ((K_खण्डानि >= KSPLIT_MIN_K_खण्डानि) && (पङ्क्तिकाः_प्रत्येकम्_minion <= KSPLIT_MAX_पङ्क्तिकाः) &&
                                           (पङ्क्तिकाः_प्रत्येकम्_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_पङ्क्तिकाः;
    ध्रुव पूर्णाङ्क     use_ksplit_गणः      = !use_ksplit && (K_खण्डानि >= KSPLIT_MIN_K_खण्डानि) && (पङ्क्तिकाः_प्रत्येकम्_minion > 4) &&
                                          (पङ्क्तिकाः_प्रत्येकम्_minion <= KSPLIT_MAX_पङ्क्तिकाः);

    evict_region_past_l2(params->src1.दत्त, tensor_अष्टकानि(&params->src1));
    यदि (params->bias.दत्त) {
        evict_region_past_l2(params->bias.दत्त, tensor_अष्टकानि(&params->bias));
    }

    यदि (use_ksplit) {
        /* Each hart processes half the K dimension */
        ध्रुव int64_t k_start = is_hart1 ? k_half : 0;
        ध्रुव int64_t k_len   = is_hart1 ? (K_खण्डानि - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        चञ्चल भिन्न * l2scp_slot = (चञ्चल भिन्न *) et_shire_l2scp_local(local_minion * 64);

        प्रति (int64_t i3 = 0; i3 < ne13; i3++) {
            ध्रुव int64_t i03       = i3 / r3;
            ध्रुव वर्ण *  src0_ptr3 = (ध्रुव वर्ण *) params->src0.दत्त + i03 * nb03;
            ध्रुव वर्ण *  src1_ptr3 = (ध्रुव वर्ण *) params->src1.दत्त + i3 * nb13;
            वर्ण *        dst_ptr3  = (वर्ण *) params->dst.दत्त + i3 * nbd3;
            ध्रुव वर्ण *  bias_ptr3 = bias_आधारः ? bias_आधारः + i3 * nbb3 : (ध्रुव वर्ण *) 0;

            प्रति (int64_t i2 = 0; i2 < ne12; i2++) {
                ध्रुव int64_t i02       = i2 / r2;
                ध्रुव वर्ण *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ध्रुव वर्ण *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                वर्ण *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ध्रुव वर्ण *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ध्रुव वर्ण *) 0;

                प्रति (int64_t n = 0; n < N; n++) {
                    ध्रुव भिन्न * b_col_आधारः = (ध्रुव भिन्न *) (src1_ptr2 + n * nb11);
                    ध्रुव भिन्न * bias_n     = bias_ptr2 ? (ध्रुव भिन्न *) (bias_ptr2 + n * nbb1) : (ध्रुव भिन्न *) 0;

                    प्रति (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        ध्रुव खण्डम्_q8_0 * q_row = (ध्रुव खण्डम्_q8_0 *) (src0_ptr2 + m * nb01);

                        भिन्न partial = compute_row_dot_q8_0(q_row + k_start, b_col_आधारः + k_start * 32, k_len);

                        यदि (is_hart1) {
                            *l2scp_slot = partial;
                            FENCE;
                            flush_गन्तव्यम्_l2((ध्रुव शून्य *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } अन्यथा {
                            et_sem_wait(ET_BARRIER_MINION);
                            भिन्न other = *l2scp_slot;
                            et_sem_post(ET_BARRIER_MINION);

                            भिन्न * dst_प्रविष्टिः = (भिन्न *) (dst_ptr2 + n * nbd1 + m * परिमाण(भिन्न));
                            भिन्न   योग_2       = partial + other;
                            यदि (bias_n) {
                                योग_2 += bias_n[m];
                            }
                            atomic_store_f32((चञ्चल भिन्न *) dst_प्रविष्टिः, योग_2);
                        }
                    }
                }
            }
        }
    } अन्यथा यदि (use_ksplit_गणः) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime.
         *
         * Both harts process the same 4-row group, each on half of K, and
         * exchange 4 partial sums once per group instead of once per row.
         * This keeps the K-split bandwidth benefit while cutting semaphore
         * traffic by 4x relative to the old per-row exchange.
         */
        ध्रुव int64_t    k_start    = is_hart1 ? k_half : 0;
        ध्रुव int64_t    k_len      = is_hart1 ? (K_खण्डानि - k_half) : k_half;
        चञ्चल भिन्न * l2scp_slot = (चञ्चल भिन्न *) et_shire_l2scp_local(local_minion * 64);

        प्रति (int64_t i3 = 0; i3 < ne13; i3++) {
            ध्रुव int64_t i03       = i3 / r3;
            ध्रुव वर्ण *  src0_ptr3 = (ध्रुव वर्ण *) params->src0.दत्त + i03 * nb03;
            ध्रुव वर्ण *  src1_ptr3 = (ध्रुव वर्ण *) params->src1.दत्त + i3 * nb13;
            वर्ण *        dst_ptr3  = (वर्ण *) params->dst.दत्त + i3 * nbd3;
            ध्रुव वर्ण *  bias_ptr3 = bias_आधारः ? bias_आधारः + i3 * nbb3 : (ध्रुव वर्ण *) 0;

            प्रति (int64_t i2 = 0; i2 < ne12; i2++) {
                ध्रुव int64_t i02       = i2 / r2;
                ध्रुव वर्ण *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ध्रुव वर्ण *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                वर्ण *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ध्रुव वर्ण *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ध्रुव वर्ण *) 0;

                प्रति (int64_t n = 0; n < N; n++) {
                    ध्रुव भिन्न * b_col_आधारः = (ध्रुव भिन्न *) (src1_ptr2 + n * nb11);
                    ध्रुव भिन्न * bias_n     = bias_ptr2 ? (ध्रुव भिन्न *) (bias_ptr2 + n * nbb1) : (ध्रुव भिन्न *) 0;

                    प्रति (int64_t m_आधारः = minion_id; m_आधारः < M; m_आधारः += STRIDE_M_KSPLIT * KSPLIT_गणः_पङ्क्तिकाः) {
                        ध्रुव int64_t m0 = m_आधारः;
                        ध्रुव int64_t m1 = m0 + STRIDE_M_KSPLIT;
                        ध्रुव int64_t m2 = m1 + STRIDE_M_KSPLIT;
                        ध्रुव int64_t m3 = m2 + STRIDE_M_KSPLIT;

                        भिन्न s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        प्रति (int64_t kb = 0; kb < K_खण्डानि; kb += TILE_KB) {
                            int64_t tile_len = k_len - kb;
                            यदि (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            यदि (tile_len <= 0) {
                                भङ्ग;
                            }
                            ध्रुव भिन्न * b_tile = b_col_आधारः + (k_start + kb) * 32;
                            ध्रुव int64_t row_kb = k_start + kb;

                            यदि (m0 < M) {
                                s0 += compute_row_dot_q8_0((ध्रुव खण्डम्_q8_0 *) (src0_ptr2 + m0 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            यदि (m1 < M) {
                                s1 += compute_row_dot_q8_0((ध्रुव खण्डम्_q8_0 *) (src0_ptr2 + m1 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            यदि (m2 < M) {
                                s2 += compute_row_dot_q8_0((ध्रुव खण्डम्_q8_0 *) (src0_ptr2 + m2 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            यदि (m3 < M) {
                                s3 += compute_row_dot_q8_0((ध्रुव खण्डम्_q8_0 *) (src0_ptr2 + m3 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                        }

                        यदि (is_hart1) {
                            l2scp_slot[0] = s0;
                            l2scp_slot[1] = s1;
                            l2scp_slot[2] = s2;
                            l2scp_slot[3] = s3;
                            FENCE;
                            flush_गन्तव्यम्_l2((ध्रुव शून्य *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } अन्यथा {
                            et_sem_wait(ET_BARRIER_MINION);
                            ध्रुव भिन्न p0 = l2scp_slot[0];
                            ध्रुव भिन्न p1 = l2scp_slot[1];
                            ध्रुव भिन्न p2 = l2scp_slot[2];
                            ध्रुव भिन्न p3 = l2scp_slot[3];
                            et_sem_post(ET_BARRIER_MINION);

                            भिन्न *     c_आधारः = (भिन्न *) (dst_ptr2 + n * nbd1);
                            ध्रुव भिन्न b0     = bias_n ? bias_n[m0] : 0.0f;
                            ध्रुव भिन्न b1     = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                            ध्रुव भिन्न b2     = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                            ध्रुव भिन्न b3     = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                            यदि (m0 < M) {
                                atomic_store_f32((चञ्चल भिन्न *) (c_आधारः + m0), s0 + p0 + b0);
                            }
                            यदि (m1 < M) {
                                atomic_store_f32((चञ्चल भिन्न *) (c_आधारः + m1), s1 + p1 + b1);
                            }
                            यदि (m2 < M) {
                                atomic_store_f32((चञ्चल भिन्न *) (c_आधारः + m2), s2 + p2 + b2);
                            }
                            यदि (m3 < M) {
                                atomic_store_f32((चञ्चल भिन्न *) (c_आधारः + m3), s3 + p3 + b3);
                            }
                        }
                    }
                }
            }
        }
    } अन्यथा यदि (K_खण्डानि > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        प्रति (int64_t i3 = 0; i3 < ne13; i3++) {
            ध्रुव int64_t i03       = i3 / r3;
            ध्रुव वर्ण *  src0_ptr3 = (ध्रुव वर्ण *) params->src0.दत्त + i03 * nb03;
            ध्रुव वर्ण *  src1_ptr3 = (ध्रुव वर्ण *) params->src1.दत्त + i3 * nb13;
            वर्ण *        dst_ptr3  = (वर्ण *) params->dst.दत्त + i3 * nbd3;
            ध्रुव वर्ण *  bias_ptr3 = bias_आधारः ? bias_आधारः + i3 * nbb3 : (ध्रुव वर्ण *) 0;

            प्रति (int64_t i2 = 0; i2 < ne12; i2++) {
                ध्रुव int64_t i02       = i2 / r2;
                ध्रुव वर्ण *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ध्रुव वर्ण *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                वर्ण *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ध्रुव वर्ण *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ध्रुव वर्ण *) 0;

                प्रति (int64_t n = 0; n < N; n++) {
                    ध्रुव भिन्न * b_col_आधारः = (ध्रुव भिन्न *) (src1_ptr2 + n * nb11);
                    ध्रुव भिन्न * bias_n     = bias_ptr2 ? (ध्रुव भिन्न *) (bias_ptr2 + n * nbb1) : (ध्रुव भिन्न *) 0;

                    प्रति (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        ध्रुव int64_t m1 = m0 + STRIDE_M;
                        ध्रुव int64_t m2 = m0 + STRIDE_M * 2;
                        ध्रुव int64_t m3 = m0 + STRIDE_M * 3;

                        भिन्न s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        प्रति (int64_t kb = 0; kb < K_खण्डानि; kb += TILE_KB) {
                            int64_t tile_len = K_खण्डानि - kb;
                            यदि (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            ध्रुव भिन्न * b_tile = b_col_आधारः + kb * 32;

                            s0 += compute_row_dot_q8_0((ध्रुव खण्डम्_q8_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            यदि (m1 < M) {
                                s1 += compute_row_dot_q8_0((ध्रुव खण्डम्_q8_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            यदि (m2 < M) {
                                s2 += compute_row_dot_q8_0((ध्रुव खण्डम्_q8_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            यदि (m3 < M) {
                                s3 += compute_row_dot_q8_0((ध्रुव खण्डम्_q8_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        भिन्न *     dst_आधारः = (भिन्न *) (dst_ptr2 + n * nbd1);
                        ध्रुव भिन्न b0       = bias_n ? bias_n[m0] : 0.0f;
                        ध्रुव भिन्न b1       = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                        ध्रुव भिन्न b2       = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                        ध्रुव भिन्न b3       = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                        atomic_store_f32((चञ्चल भिन्न *) (dst_आधारः + m0), s0 + b0);
                        यदि (m1 < M) {
                            atomic_store_f32((चञ्चल भिन्न *) (dst_आधारः + m1), s1 + b1);
                        }
                        यदि (m2 < M) {
                            atomic_store_f32((चञ्चल भिन्न *) (dst_आधारः + m2), s2 + b2);
                        }
                        यदि (m3 < M) {
                            atomic_store_f32((चञ्चल भिन्न *) (dst_आधारः + m3), s3 + b3);
                        }
                    }
                }
            }
        }
    } अन्यथा {
        /*
         * Simple path for small K.
         *
         * When `nb01` is 32-byte aligned, every row has the same block-alignment
         * pattern. That lets us compute two rows together and reuse each loaded
         * B chunk across both rows instead of reloading it in a second dot call.
         */
        प्रति (int64_t i3 = 0; i3 < ne13; i3++) {
            ध्रुव int64_t i03       = i3 / r3;
            ध्रुव वर्ण *  src0_ptr3 = (ध्रुव वर्ण *) params->src0.दत्त + i03 * nb03;
            ध्रुव वर्ण *  src1_ptr3 = (ध्रुव वर्ण *) params->src1.दत्त + i3 * nb13;
            वर्ण *        dst_ptr3  = (वर्ण *) params->dst.दत्त + i3 * nbd3;
            ध्रुव वर्ण *  bias_ptr3 = bias_आधारः ? bias_आधारः + i3 * nbb3 : (ध्रुव वर्ण *) 0;

            प्रति (int64_t i2 = 0; i2 < ne12; i2++) {
                ध्रुव int64_t i02       = i2 / r2;
                ध्रुव वर्ण *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ध्रुव वर्ण *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                वर्ण *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ध्रुव वर्ण *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ध्रुव वर्ण *) 0;

                प्रति (int64_t n = 0; n < N; n++) {
                    ध्रुव भिन्न * b_col_आधारः = (ध्रुव भिन्न *) (src1_ptr2 + n * nb11);
                    ध्रुव भिन्न * bias_n     = bias_ptr2 ? (ध्रुव भिन्न *) (bias_ptr2 + n * nbb1) : (ध्रुव भिन्न *) 0;
                    q8_dot_state  q8_state;
                    q8_dot_begin(&q8_state);

                    यदि (use_simple_x2) {
                        प्रति (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_पङ्क्तिकाः) {
                            ध्रुव int64_t      m1     = m0 + STRIDE_M;
                            ध्रुव खण्डम्_q8_0 * q_row0 = (ध्रुव खण्डम्_q8_0 *) (src0_ptr2 + m0 * nb01);

                            यदि (m1 < M) {
                                ध्रुव खण्डम्_q8_0 * q_row1 = (ध्रुव खण्डम्_q8_0 *) (src0_ptr2 + m1 * nb01);
                                भिन्न              s0, s1;
                                q8_dot_compute_x2_aligned(q_row0, q_row1, b_col_आधारः, K_खण्डानि, &s0, &s1);

                                भिन्न * dst0 = (भिन्न *) (dst_ptr2 + n * nbd1 + m0 * परिमाण(भिन्न));
                                भिन्न * dst1 = (भिन्न *) (dst_ptr2 + n * nbd1 + m1 * परिमाण(भिन्न));
                                यदि (bias_n) {
                                    s0 += bias_n[m0];
                                    s1 += bias_n[m1];
                                }
                                atomic_store_f32((चञ्चल भिन्न *) dst0, s0);
                                atomic_store_f32((चञ्चल भिन्न *) dst1, s1);
                            } अन्यथा {
                                भिन्न   योग_2 = q8_dot_compute(q_row0, b_col_आधारः, K_खण्डानि);
                                भिन्न * dst = (भिन्न *) (dst_ptr2 + n * nbd1 + m0 * परिमाण(भिन्न));
                                यदि (bias_n) {
                                    योग_2 += bias_n[m0];
                                }
                                atomic_store_f32((चञ्चल भिन्न *) dst, योग_2);
                            }
                        }
                    } अन्यथा {
                        प्रति (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            ध्रुव खण्डम्_q8_0 * q_row = (ध्रुव खण्डम्_q8_0 *) (src0_ptr2 + m * nb01);

                            भिन्न योग_2 = q8_dot_compute(q_row, b_col_आधारः, K_खण्डानि);

                            भिन्न * dst_प्रविष्टिः = (भिन्न *) (dst_ptr2 + n * nbd1 + m * परिमाण(भिन्न));
                            यदि (bias_n) {
                                योग_2 += bias_n[m];
                            }
                            atomic_store_f32((चञ्चल भिन्न *) dst_प्रविष्टिः, योग_2);
                        }
                    }

                    q8_dot_अन्तः(&q8_state);
                }
            }
        }
    }

#यदि_परिभाषितम् ET_UBERKERNEL
    FENCE;
    evict_region_past_l2(params->dst.दत्त, tensor_अष्टकानि(&params->dst));
    WAIT_CACHEOPS;
    FENCE;
#यदि_अन्तः
    प्रत्यागम 0;
}
