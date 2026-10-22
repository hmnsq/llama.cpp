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

#define STRIDE_M                   2048 /* 32 shires x 32 minions x 2 harts */
#define STRIDE_M_KSPLIT            1024 /* 32 shires x 32 minions (both harts share rows) */
#define KSPLIT_MIN_K_BLOCKS        256  /* K >= 8192 elements */
#define KSPLIT_SMALL_ROWS_K_BLOCKS 64   /* K >= 2048 elements for very small M */
#define KSPLIT_MAX_ROWS            8    /* max rows per minion for K-split */
#define TILE_KB                    256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */
#define KSPLIT_GROUP_ROWS          4
#define SIMPLE_X2_ROWS             2

स्थिर अन्तःस्थ माप_प्रकार tensor_bytes(अचर संरचना ggml_tensor * t) {
    फर्काऊ (माप_प्रकार) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

पूर्णाङ्क entry_point(संरचना ggml_et_mm_q8_params * params, शून्य * env) {
    uint64_t hart_id = get_hart_id();

    // Matrix dimensions
    अचर int64_t K    = params->src0.ne[0];
    अचर int64_t M    = params->src0.ne[1];
    अचर int64_t N    = params->src1.ne[1];
    अचर int64_t ne02 = params->src0.ne[2];
    अचर int64_t ne03 = params->src0.ne[3];
    अचर int64_t ne12 = params->src1.ne[2];
    अचर int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    अचर माप_प्रकार nb01 = params->src0.nb[1];
    अचर माप_प्रकार nb02 = params->src0.nb[2];
    अचर माप_प्रकार nb03 = params->src0.nb[3];

    अचर माप_प्रकार nb11 = params->src1.nb[1];
    अचर माप_प्रकार nb12 = params->src1.nb[2];
    अचर माप_प्रकार nb13 = params->src1.nb[3];

    अचर माप_प्रकार nbd1 = params->dst.nb[1];
    अचर माप_प्रकार nbd2 = params->dst.nb[2];
    अचर माप_प्रकार nbd3 = params->dst.nb[3];

    // Optional residual bias
    अचर वर्ण * bias_base = (अचर वर्ण *) params->bias.data;
    अचर माप_प्रकार nbb1      = params->bias.nb[1];
    अचर माप_प्रकार nbb2      = params->bias.nb[2];
    अचर माप_प्रकार nbb3      = params->bias.nb[3];

    // Q8_0 block size is 32
    अचर int64_t K_blocks      = K / 32;
    अचर पूर्णाङ्क     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    अचर int64_t r2 = ne12 / ne02;
    अचर int64_t r3 = ne13 / ne03;

    // K-split decision
    अचर int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    अचर int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    अचर पूर्णाङ्क     is_hart1              = hart_id & 1;
    अचर int64_t rows_per_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    अचर int64_t k_half                = K_blocks / 2;
    अचर पूर्णाङ्क     use_ksplit_small_rows = (rows_per_minion <= 2) && (K_blocks >= KSPLIT_SMALL_ROWS_K_BLOCKS);
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
    अचर पूर्णाङ्क     use_ksplit            = ((K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion <= KSPLIT_MAX_ROWS) &&
                                           (rows_per_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_rows;
    अचर पूर्णाङ्क     use_ksplit_group      = !use_ksplit && (K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion > 4) &&
                                          (rows_per_minion <= KSPLIT_MAX_ROWS);

    evict_region_past_l2(params->src1.data, tensor_bytes(&params->src1));
    यदि (params->bias.data) {
        evict_region_past_l2(params->bias.data, tensor_bytes(&params->bias));
    }

    यदि (use_ksplit) {
        /* Each hart processes half the K dimension */
        अचर int64_t k_start = is_hart1 ? k_half : 0;
        अचर int64_t k_len   = is_hart1 ? (K_blocks - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        चञ्चल भिन्न * l2scp_slot = (चञ्चल भिन्न *) et_shire_l2scp_local(local_minion * 64);

        लागि (int64_t i3 = 0; i3 < ne13; i3++) {
            अचर int64_t i03       = i3 / r3;
            अचर वर्ण *  src0_ptr3 = (अचर वर्ण *) params->src0.data + i03 * nb03;
            अचर वर्ण *  src1_ptr3 = (अचर वर्ण *) params->src1.data + i3 * nb13;
            वर्ण *        dst_ptr3  = (वर्ण *) params->dst.data + i3 * nbd3;
            अचर वर्ण *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (अचर वर्ण *) 0;

            लागि (int64_t i2 = 0; i2 < ne12; i2++) {
                अचर int64_t i02       = i2 / r2;
                अचर वर्ण *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                अचर वर्ण *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                वर्ण *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                अचर वर्ण *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (अचर वर्ण *) 0;

                लागि (int64_t n = 0; n < N; n++) {
                    अचर भिन्न * b_col_base = (अचर भिन्न *) (src1_ptr2 + n * nb11);
                    अचर भिन्न * bias_n     = bias_ptr2 ? (अचर भिन्न *) (bias_ptr2 + n * nbb1) : (अचर भिन्न *) 0;

                    लागि (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        अचर block_q8_0 * q_row = (अचर block_q8_0 *) (src0_ptr2 + m * nb01);

                        भिन्न partial = compute_row_dot_q8_0(q_row + k_start, b_col_base + k_start * 32, k_len);

                        यदि (is_hart1) {
                            *l2scp_slot = partial;
                            FENCE;
                            flush_to_l2((अचर शून्य *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } अन्यथा {
                            et_sem_wait(ET_BARRIER_MINION);
                            भिन्न other = *l2scp_slot;
                            et_sem_post(ET_BARRIER_MINION);

                            भिन्न * dst_entry = (भिन्न *) (dst_ptr2 + n * nbd1 + m * परिमाण(भिन्न));
                            भिन्न   sum       = partial + other;
                            यदि (bias_n) {
                                sum += bias_n[m];
                            }
                            atomic_store_f32((चञ्चल भिन्न *) dst_entry, sum);
                        }
                    }
                }
            }
        }
    } अन्यथा यदि (use_ksplit_group) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime.
         *
         * Both harts process the same 4-row group, each on half of K, and
         * exchange 4 partial sums once per group instead of once per row.
         * This keeps the K-split bandwidth benefit while cutting semaphore
         * traffic by 4x relative to the old per-row exchange.
         */
        अचर int64_t    k_start    = is_hart1 ? k_half : 0;
        अचर int64_t    k_len      = is_hart1 ? (K_blocks - k_half) : k_half;
        चञ्चल भिन्न * l2scp_slot = (चञ्चल भिन्न *) et_shire_l2scp_local(local_minion * 64);

        लागि (int64_t i3 = 0; i3 < ne13; i3++) {
            अचर int64_t i03       = i3 / r3;
            अचर वर्ण *  src0_ptr3 = (अचर वर्ण *) params->src0.data + i03 * nb03;
            अचर वर्ण *  src1_ptr3 = (अचर वर्ण *) params->src1.data + i3 * nb13;
            वर्ण *        dst_ptr3  = (वर्ण *) params->dst.data + i3 * nbd3;
            अचर वर्ण *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (अचर वर्ण *) 0;

            लागि (int64_t i2 = 0; i2 < ne12; i2++) {
                अचर int64_t i02       = i2 / r2;
                अचर वर्ण *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                अचर वर्ण *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                वर्ण *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                अचर वर्ण *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (अचर वर्ण *) 0;

                लागि (int64_t n = 0; n < N; n++) {
                    अचर भिन्न * b_col_base = (अचर भिन्न *) (src1_ptr2 + n * nb11);
                    अचर भिन्न * bias_n     = bias_ptr2 ? (अचर भिन्न *) (bias_ptr2 + n * nbb1) : (अचर भिन्न *) 0;

                    लागि (int64_t m_base = minion_id; m_base < M; m_base += STRIDE_M_KSPLIT * KSPLIT_GROUP_ROWS) {
                        अचर int64_t m0 = m_base;
                        अचर int64_t m1 = m0 + STRIDE_M_KSPLIT;
                        अचर int64_t m2 = m1 + STRIDE_M_KSPLIT;
                        अचर int64_t m3 = m2 + STRIDE_M_KSPLIT;

                        भिन्न s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        लागि (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = k_len - kb;
                            यदि (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            यदि (tile_len <= 0) {
                                रोक;
                            }
                            अचर भिन्न * b_tile = b_col_base + (k_start + kb) * 32;
                            अचर int64_t row_kb = k_start + kb;

                            यदि (m0 < M) {
                                s0 += compute_row_dot_q8_0((अचर block_q8_0 *) (src0_ptr2 + m0 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            यदि (m1 < M) {
                                s1 += compute_row_dot_q8_0((अचर block_q8_0 *) (src0_ptr2 + m1 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            यदि (m2 < M) {
                                s2 += compute_row_dot_q8_0((अचर block_q8_0 *) (src0_ptr2 + m2 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            यदि (m3 < M) {
                                s3 += compute_row_dot_q8_0((अचर block_q8_0 *) (src0_ptr2 + m3 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                        }

                        यदि (is_hart1) {
                            l2scp_slot[0] = s0;
                            l2scp_slot[1] = s1;
                            l2scp_slot[2] = s2;
                            l2scp_slot[3] = s3;
                            FENCE;
                            flush_to_l2((अचर शून्य *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } अन्यथा {
                            et_sem_wait(ET_BARRIER_MINION);
                            अचर भिन्न p0 = l2scp_slot[0];
                            अचर भिन्न p1 = l2scp_slot[1];
                            अचर भिन्न p2 = l2scp_slot[2];
                            अचर भिन्न p3 = l2scp_slot[3];
                            et_sem_post(ET_BARRIER_MINION);

                            भिन्न *     c_base = (भिन्न *) (dst_ptr2 + n * nbd1);
                            अचर भिन्न b0     = bias_n ? bias_n[m0] : 0.0f;
                            अचर भिन्न b1     = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                            अचर भिन्न b2     = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                            अचर भिन्न b3     = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                            यदि (m0 < M) {
                                atomic_store_f32((चञ्चल भिन्न *) (c_base + m0), s0 + p0 + b0);
                            }
                            यदि (m1 < M) {
                                atomic_store_f32((चञ्चल भिन्न *) (c_base + m1), s1 + p1 + b1);
                            }
                            यदि (m2 < M) {
                                atomic_store_f32((चञ्चल भिन्न *) (c_base + m2), s2 + p2 + b2);
                            }
                            यदि (m3 < M) {
                                atomic_store_f32((चञ्चल भिन्न *) (c_base + m3), s3 + p3 + b3);
                            }
                        }
                    }
                }
            }
        }
    } अन्यथा यदि (K_blocks > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        लागि (int64_t i3 = 0; i3 < ne13; i3++) {
            अचर int64_t i03       = i3 / r3;
            अचर वर्ण *  src0_ptr3 = (अचर वर्ण *) params->src0.data + i03 * nb03;
            अचर वर्ण *  src1_ptr3 = (अचर वर्ण *) params->src1.data + i3 * nb13;
            वर्ण *        dst_ptr3  = (वर्ण *) params->dst.data + i3 * nbd3;
            अचर वर्ण *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (अचर वर्ण *) 0;

            लागि (int64_t i2 = 0; i2 < ne12; i2++) {
                अचर int64_t i02       = i2 / r2;
                अचर वर्ण *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                अचर वर्ण *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                वर्ण *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                अचर वर्ण *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (अचर वर्ण *) 0;

                लागि (int64_t n = 0; n < N; n++) {
                    अचर भिन्न * b_col_base = (अचर भिन्न *) (src1_ptr2 + n * nb11);
                    अचर भिन्न * bias_n     = bias_ptr2 ? (अचर भिन्न *) (bias_ptr2 + n * nbb1) : (अचर भिन्न *) 0;

                    लागि (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        अचर int64_t m1 = m0 + STRIDE_M;
                        अचर int64_t m2 = m0 + STRIDE_M * 2;
                        अचर int64_t m3 = m0 + STRIDE_M * 3;

                        भिन्न s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        लागि (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = K_blocks - kb;
                            यदि (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            अचर भिन्न * b_tile = b_col_base + kb * 32;

                            s0 += compute_row_dot_q8_0((अचर block_q8_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            यदि (m1 < M) {
                                s1 += compute_row_dot_q8_0((अचर block_q8_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            यदि (m2 < M) {
                                s2 += compute_row_dot_q8_0((अचर block_q8_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            यदि (m3 < M) {
                                s3 += compute_row_dot_q8_0((अचर block_q8_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        भिन्न *     dst_base = (भिन्न *) (dst_ptr2 + n * nbd1);
                        अचर भिन्न b0       = bias_n ? bias_n[m0] : 0.0f;
                        अचर भिन्न b1       = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                        अचर भिन्न b2       = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                        अचर भिन्न b3       = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                        atomic_store_f32((चञ्चल भिन्न *) (dst_base + m0), s0 + b0);
                        यदि (m1 < M) {
                            atomic_store_f32((चञ्चल भिन्न *) (dst_base + m1), s1 + b1);
                        }
                        यदि (m2 < M) {
                            atomic_store_f32((चञ्चल भिन्न *) (dst_base + m2), s2 + b2);
                        }
                        यदि (m3 < M) {
                            atomic_store_f32((चञ्चल भिन्न *) (dst_base + m3), s3 + b3);
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
        लागि (int64_t i3 = 0; i3 < ne13; i3++) {
            अचर int64_t i03       = i3 / r3;
            अचर वर्ण *  src0_ptr3 = (अचर वर्ण *) params->src0.data + i03 * nb03;
            अचर वर्ण *  src1_ptr3 = (अचर वर्ण *) params->src1.data + i3 * nb13;
            वर्ण *        dst_ptr3  = (वर्ण *) params->dst.data + i3 * nbd3;
            अचर वर्ण *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (अचर वर्ण *) 0;

            लागि (int64_t i2 = 0; i2 < ne12; i2++) {
                अचर int64_t i02       = i2 / r2;
                अचर वर्ण *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                अचर वर्ण *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                वर्ण *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                अचर वर्ण *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (अचर वर्ण *) 0;

                लागि (int64_t n = 0; n < N; n++) {
                    अचर भिन्न * b_col_base = (अचर भिन्न *) (src1_ptr2 + n * nb11);
                    अचर भिन्न * bias_n     = bias_ptr2 ? (अचर भिन्न *) (bias_ptr2 + n * nbb1) : (अचर भिन्न *) 0;
                    q8_dot_state  q8_state;
                    q8_dot_begin(&q8_state);

                    यदि (use_simple_x2) {
                        लागि (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_ROWS) {
                            अचर int64_t      m1     = m0 + STRIDE_M;
                            अचर block_q8_0 * q_row0 = (अचर block_q8_0 *) (src0_ptr2 + m0 * nb01);

                            यदि (m1 < M) {
                                अचर block_q8_0 * q_row1 = (अचर block_q8_0 *) (src0_ptr2 + m1 * nb01);
                                भिन्न              s0, s1;
                                q8_dot_compute_x2_aligned(q_row0, q_row1, b_col_base, K_blocks, &s0, &s1);

                                भिन्न * dst0 = (भिन्न *) (dst_ptr2 + n * nbd1 + m0 * परिमाण(भिन्न));
                                भिन्न * dst1 = (भिन्न *) (dst_ptr2 + n * nbd1 + m1 * परिमाण(भिन्न));
                                यदि (bias_n) {
                                    s0 += bias_n[m0];
                                    s1 += bias_n[m1];
                                }
                                atomic_store_f32((चञ्चल भिन्न *) dst0, s0);
                                atomic_store_f32((चञ्चल भिन्न *) dst1, s1);
                            } अन्यथा {
                                भिन्न   sum = q8_dot_compute(q_row0, b_col_base, K_blocks);
                                भिन्न * dst = (भिन्न *) (dst_ptr2 + n * nbd1 + m0 * परिमाण(भिन्न));
                                यदि (bias_n) {
                                    sum += bias_n[m0];
                                }
                                atomic_store_f32((चञ्चल भिन्न *) dst, sum);
                            }
                        }
                    } अन्यथा {
                        लागि (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            अचर block_q8_0 * q_row = (अचर block_q8_0 *) (src0_ptr2 + m * nb01);

                            भिन्न sum = q8_dot_compute(q_row, b_col_base, K_blocks);

                            भिन्न * dst_entry = (भिन्न *) (dst_ptr2 + n * nbd1 + m * परिमाण(भिन्न));
                            यदि (bias_n) {
                                sum += bias_n[m];
                            }
                            atomic_store_f32((चञ्चल भिन्न *) dst_entry, sum);
                        }
                    }

                    q8_dot_end(&q8_state);
                }
            }
        }
    }

#ifdef ET_UBERKERNEL
    FENCE;
    evict_region_past_l2(params->dst.data, tensor_bytes(&params->dst));
    WAIT_CACHEOPS;
    FENCE;
#endif
    फर्काऊ 0;
}
