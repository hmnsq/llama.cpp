//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#समावेश "block_ops.h"
#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"
#समावेश "quants.h"

#समावेश <stdint.h>

#घोषणा STRIDE_M                   2048 /* 32 shires x 32 minions x 2 harts */
#घोषणा STRIDE_M_KSPLIT            1024 /* 32 shires x 32 minions (both harts share rows) */
#घोषणा KSPLIT_MIN_K_BLOCKS        256  /* K >= 8192 elements */
#घोषणा KSPLIT_SMALL_ROWS_K_BLOCKS 64   /* K >= 2048 elements for very small M */
#घोषणा KSPLIT_MAX_ROWS            8    /* max rows per minion for K-split */
#घोषणा TILE_KB                    256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */
#घोषणा KSPLIT_GROUP_ROWS          4
#घोषणा SIMPLE_X2_ROWS             2

अटल अंतरभूत माप_प्रकार tensor_bytes(स्थिर काष्ठा ggml_tensor * t) {
    वापस (माप_प्रकार) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

पूर्णांक entry_point(काष्ठा ggml_et_mm_q8_params * params, व्योम * env) {
    uint64_t hart_id = get_hart_id();

    // Matrix dimensions
    स्थिर int64_t K    = params->src0.ne[0];
    स्थिर int64_t M    = params->src0.ne[1];
    स्थिर int64_t N    = params->src1.ne[1];
    स्थिर int64_t ne02 = params->src0.ne[2];
    स्थिर int64_t ne03 = params->src0.ne[3];
    स्थिर int64_t ne12 = params->src1.ne[2];
    स्थिर int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    स्थिर माप_प्रकार nb01 = params->src0.nb[1];
    स्थिर माप_प्रकार nb02 = params->src0.nb[2];
    स्थिर माप_प्रकार nb03 = params->src0.nb[3];

    स्थिर माप_प्रकार nb11 = params->src1.nb[1];
    स्थिर माप_प्रकार nb12 = params->src1.nb[2];
    स्थिर माप_प्रकार nb13 = params->src1.nb[3];

    स्थिर माप_प्रकार nbd1 = params->dst.nb[1];
    स्थिर माप_प्रकार nbd2 = params->dst.nb[2];
    स्थिर माप_प्रकार nbd3 = params->dst.nb[3];

    // Optional residual bias
    स्थिर अक्षर * bias_base = (स्थिर अक्षर *) params->bias.तथ्य;
    स्थिर माप_प्रकार nbb1      = params->bias.nb[1];
    स्थिर माप_प्रकार nbb2      = params->bias.nb[2];
    स्थिर माप_प्रकार nbb3      = params->bias.nb[3];

    // Q8_0 block size is 32
    स्थिर int64_t K_blocks      = K / 32;
    स्थिर पूर्णांक     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    स्थिर int64_t r2 = ne12 / ne02;
    स्थिर int64_t r3 = ne13 / ne03;

    // K-split decision
    स्थिर int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    स्थिर int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    स्थिर पूर्णांक     is_hart1              = hart_id & 1;
    स्थिर int64_t rows_per_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    स्थिर int64_t k_half                = K_blocks / 2;
    स्थिर पूर्णांक     use_ksplit_small_rows = (rows_per_minion <= 2) && (K_blocks >= KSPLIT_SMALL_ROWS_K_BLOCKS);
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
    स्थिर पूर्णांक     use_ksplit            = ((K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion <= KSPLIT_MAX_ROWS) &&
                                           (rows_per_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_rows;
    स्थिर पूर्णांक     use_ksplit_group      = !use_ksplit && (K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion > 4) &&
                                          (rows_per_minion <= KSPLIT_MAX_ROWS);

    evict_region_past_l2(params->src1.तथ्य, tensor_bytes(&params->src1));
    अगर (params->bias.तथ्य) {
        evict_region_past_l2(params->bias.तथ्य, tensor_bytes(&params->bias));
    }

    अगर (use_ksplit) {
        /* Each hart processes half the K dimension */
        स्थिर int64_t k_start = is_hart1 ? k_half : 0;
        स्थिर int64_t k_len   = is_hart1 ? (K_blocks - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        अस्थिर भग्न * l2scp_slot = (अस्थिर भग्न *) et_shire_l2scp_local(local_minion * 64);

        क्रम (int64_t i3 = 0; i3 < ne13; i3++) {
            स्थिर int64_t i03       = i3 / r3;
            स्थिर अक्षर *  src0_ptr3 = (स्थिर अक्षर *) params->src0.तथ्य + i03 * nb03;
            स्थिर अक्षर *  src1_ptr3 = (स्थिर अक्षर *) params->src1.तथ्य + i3 * nb13;
            अक्षर *        dst_ptr3  = (अक्षर *) params->dst.तथ्य + i3 * nbd3;
            स्थिर अक्षर *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (स्थिर अक्षर *) 0;

            क्रम (int64_t i2 = 0; i2 < ne12; i2++) {
                स्थिर int64_t i02       = i2 / r2;
                स्थिर अक्षर *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                स्थिर अक्षर *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                अक्षर *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                स्थिर अक्षर *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (स्थिर अक्षर *) 0;

                क्रम (int64_t n = 0; n < N; n++) {
                    स्थिर भग्न * b_col_base = (स्थिर भग्न *) (src1_ptr2 + n * nb11);
                    स्थिर भग्न * bias_n     = bias_ptr2 ? (स्थिर भग्न *) (bias_ptr2 + n * nbb1) : (स्थिर भग्न *) 0;

                    क्रम (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        स्थिर block_q8_0 * q_row = (स्थिर block_q8_0 *) (src0_ptr2 + m * nb01);

                        भग्न partial = compute_row_dot_q8_0(q_row + k_start, b_col_base + k_start * 32, k_len);

                        अगर (is_hart1) {
                            *l2scp_slot = partial;
                            FENCE;
                            flush_to_l2((स्थिर व्योम *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } अन्यथा {
                            et_sem_wait(ET_BARRIER_MINION);
                            भग्न other = *l2scp_slot;
                            et_sem_post(ET_BARRIER_MINION);

                            भग्न * dst_entry = (भग्न *) (dst_ptr2 + n * nbd1 + m * माप(भग्न));
                            भग्न   योग_2       = partial + other;
                            अगर (bias_n) {
                                योग_2 += bias_n[m];
                            }
                            atomic_store_f32((अस्थिर भग्न *) dst_entry, योग_2);
                        }
                    }
                }
            }
        }
    } अन्यथा अगर (use_ksplit_group) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime.
         *
         * Both harts process the same 4-row group, each on half of K, and
         * exchange 4 partial sums once per group instead of once per row.
         * This keeps the K-split bandwidth benefit while cutting semaphore
         * traffic by 4x relative to the old per-row exchange.
         */
        स्थिर int64_t    k_start    = is_hart1 ? k_half : 0;
        स्थिर int64_t    k_len      = is_hart1 ? (K_blocks - k_half) : k_half;
        अस्थिर भग्न * l2scp_slot = (अस्थिर भग्न *) et_shire_l2scp_local(local_minion * 64);

        क्रम (int64_t i3 = 0; i3 < ne13; i3++) {
            स्थिर int64_t i03       = i3 / r3;
            स्थिर अक्षर *  src0_ptr3 = (स्थिर अक्षर *) params->src0.तथ्य + i03 * nb03;
            स्थिर अक्षर *  src1_ptr3 = (स्थिर अक्षर *) params->src1.तथ्य + i3 * nb13;
            अक्षर *        dst_ptr3  = (अक्षर *) params->dst.तथ्य + i3 * nbd3;
            स्थिर अक्षर *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (स्थिर अक्षर *) 0;

            क्रम (int64_t i2 = 0; i2 < ne12; i2++) {
                स्थिर int64_t i02       = i2 / r2;
                स्थिर अक्षर *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                स्थिर अक्षर *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                अक्षर *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                स्थिर अक्षर *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (स्थिर अक्षर *) 0;

                क्रम (int64_t n = 0; n < N; n++) {
                    स्थिर भग्न * b_col_base = (स्थिर भग्न *) (src1_ptr2 + n * nb11);
                    स्थिर भग्न * bias_n     = bias_ptr2 ? (स्थिर भग्न *) (bias_ptr2 + n * nbb1) : (स्थिर भग्न *) 0;

                    क्रम (int64_t m_base = minion_id; m_base < M; m_base += STRIDE_M_KSPLIT * KSPLIT_GROUP_ROWS) {
                        स्थिर int64_t m0 = m_base;
                        स्थिर int64_t m1 = m0 + STRIDE_M_KSPLIT;
                        स्थिर int64_t m2 = m1 + STRIDE_M_KSPLIT;
                        स्थिर int64_t m3 = m2 + STRIDE_M_KSPLIT;

                        भग्न s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        क्रम (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = k_len - kb;
                            अगर (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            अगर (tile_len <= 0) {
                                अवरोध;
                            }
                            स्थिर भग्न * b_tile = b_col_base + (k_start + kb) * 32;
                            स्थिर int64_t row_kb = k_start + kb;

                            अगर (m0 < M) {
                                s0 += compute_row_dot_q8_0((स्थिर block_q8_0 *) (src0_ptr2 + m0 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            अगर (m1 < M) {
                                s1 += compute_row_dot_q8_0((स्थिर block_q8_0 *) (src0_ptr2 + m1 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            अगर (m2 < M) {
                                s2 += compute_row_dot_q8_0((स्थिर block_q8_0 *) (src0_ptr2 + m2 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            अगर (m3 < M) {
                                s3 += compute_row_dot_q8_0((स्थिर block_q8_0 *) (src0_ptr2 + m3 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                        }

                        अगर (is_hart1) {
                            l2scp_slot[0] = s0;
                            l2scp_slot[1] = s1;
                            l2scp_slot[2] = s2;
                            l2scp_slot[3] = s3;
                            FENCE;
                            flush_to_l2((स्थिर व्योम *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } अन्यथा {
                            et_sem_wait(ET_BARRIER_MINION);
                            स्थिर भग्न p0 = l2scp_slot[0];
                            स्थिर भग्न p1 = l2scp_slot[1];
                            स्थिर भग्न p2 = l2scp_slot[2];
                            स्थिर भग्न p3 = l2scp_slot[3];
                            et_sem_post(ET_BARRIER_MINION);

                            भग्न *     c_base = (भग्न *) (dst_ptr2 + n * nbd1);
                            स्थिर भग्न b0     = bias_n ? bias_n[m0] : 0.0f;
                            स्थिर भग्न b1     = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                            स्थिर भग्न b2     = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                            स्थिर भग्न b3     = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                            अगर (m0 < M) {
                                atomic_store_f32((अस्थिर भग्न *) (c_base + m0), s0 + p0 + b0);
                            }
                            अगर (m1 < M) {
                                atomic_store_f32((अस्थिर भग्न *) (c_base + m1), s1 + p1 + b1);
                            }
                            अगर (m2 < M) {
                                atomic_store_f32((अस्थिर भग्न *) (c_base + m2), s2 + p2 + b2);
                            }
                            अगर (m3 < M) {
                                atomic_store_f32((अस्थिर भग्न *) (c_base + m3), s3 + p3 + b3);
                            }
                        }
                    }
                }
            }
        }
    } अन्यथा अगर (K_blocks > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        क्रम (int64_t i3 = 0; i3 < ne13; i3++) {
            स्थिर int64_t i03       = i3 / r3;
            स्थिर अक्षर *  src0_ptr3 = (स्थिर अक्षर *) params->src0.तथ्य + i03 * nb03;
            स्थिर अक्षर *  src1_ptr3 = (स्थिर अक्षर *) params->src1.तथ्य + i3 * nb13;
            अक्षर *        dst_ptr3  = (अक्षर *) params->dst.तथ्य + i3 * nbd3;
            स्थिर अक्षर *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (स्थिर अक्षर *) 0;

            क्रम (int64_t i2 = 0; i2 < ne12; i2++) {
                स्थिर int64_t i02       = i2 / r2;
                स्थिर अक्षर *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                स्थिर अक्षर *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                अक्षर *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                स्थिर अक्षर *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (स्थिर अक्षर *) 0;

                क्रम (int64_t n = 0; n < N; n++) {
                    स्थिर भग्न * b_col_base = (स्थिर भग्न *) (src1_ptr2 + n * nb11);
                    स्थिर भग्न * bias_n     = bias_ptr2 ? (स्थिर भग्न *) (bias_ptr2 + n * nbb1) : (स्थिर भग्न *) 0;

                    क्रम (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        स्थिर int64_t m1 = m0 + STRIDE_M;
                        स्थिर int64_t m2 = m0 + STRIDE_M * 2;
                        स्थिर int64_t m3 = m0 + STRIDE_M * 3;

                        भग्न s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        क्रम (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = K_blocks - kb;
                            अगर (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            स्थिर भग्न * b_tile = b_col_base + kb * 32;

                            s0 += compute_row_dot_q8_0((स्थिर block_q8_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            अगर (m1 < M) {
                                s1 += compute_row_dot_q8_0((स्थिर block_q8_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            अगर (m2 < M) {
                                s2 += compute_row_dot_q8_0((स्थिर block_q8_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            अगर (m3 < M) {
                                s3 += compute_row_dot_q8_0((स्थिर block_q8_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        भग्न *     dst_base = (भग्न *) (dst_ptr2 + n * nbd1);
                        स्थिर भग्न b0       = bias_n ? bias_n[m0] : 0.0f;
                        स्थिर भग्न b1       = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                        स्थिर भग्न b2       = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                        स्थिर भग्न b3       = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                        atomic_store_f32((अस्थिर भग्न *) (dst_base + m0), s0 + b0);
                        अगर (m1 < M) {
                            atomic_store_f32((अस्थिर भग्न *) (dst_base + m1), s1 + b1);
                        }
                        अगर (m2 < M) {
                            atomic_store_f32((अस्थिर भग्न *) (dst_base + m2), s2 + b2);
                        }
                        अगर (m3 < M) {
                            atomic_store_f32((अस्थिर भग्न *) (dst_base + m3), s3 + b3);
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
        क्रम (int64_t i3 = 0; i3 < ne13; i3++) {
            स्थिर int64_t i03       = i3 / r3;
            स्थिर अक्षर *  src0_ptr3 = (स्थिर अक्षर *) params->src0.तथ्य + i03 * nb03;
            स्थिर अक्षर *  src1_ptr3 = (स्थिर अक्षर *) params->src1.तथ्य + i3 * nb13;
            अक्षर *        dst_ptr3  = (अक्षर *) params->dst.तथ्य + i3 * nbd3;
            स्थिर अक्षर *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (स्थिर अक्षर *) 0;

            क्रम (int64_t i2 = 0; i2 < ne12; i2++) {
                स्थिर int64_t i02       = i2 / r2;
                स्थिर अक्षर *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                स्थिर अक्षर *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                अक्षर *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                स्थिर अक्षर *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (स्थिर अक्षर *) 0;

                क्रम (int64_t n = 0; n < N; n++) {
                    स्थिर भग्न * b_col_base = (स्थिर भग्न *) (src1_ptr2 + n * nb11);
                    स्थिर भग्न * bias_n     = bias_ptr2 ? (स्थिर भग्न *) (bias_ptr2 + n * nbb1) : (स्थिर भग्न *) 0;
                    q8_dot_state  q8_state;
                    q8_dot_begin(&q8_state);

                    अगर (use_simple_x2) {
                        क्रम (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_ROWS) {
                            स्थिर int64_t      m1     = m0 + STRIDE_M;
                            स्थिर block_q8_0 * q_row0 = (स्थिर block_q8_0 *) (src0_ptr2 + m0 * nb01);

                            अगर (m1 < M) {
                                स्थिर block_q8_0 * q_row1 = (स्थिर block_q8_0 *) (src0_ptr2 + m1 * nb01);
                                भग्न              s0, s1;
                                q8_dot_compute_x2_aligned(q_row0, q_row1, b_col_base, K_blocks, &s0, &s1);

                                भग्न * dst0 = (भग्न *) (dst_ptr2 + n * nbd1 + m0 * माप(भग्न));
                                भग्न * dst1 = (भग्न *) (dst_ptr2 + n * nbd1 + m1 * माप(भग्न));
                                अगर (bias_n) {
                                    s0 += bias_n[m0];
                                    s1 += bias_n[m1];
                                }
                                atomic_store_f32((अस्थिर भग्न *) dst0, s0);
                                atomic_store_f32((अस्थिर भग्न *) dst1, s1);
                            } अन्यथा {
                                भग्न   योग_2 = q8_dot_compute(q_row0, b_col_base, K_blocks);
                                भग्न * dst = (भग्न *) (dst_ptr2 + n * nbd1 + m0 * माप(भग्न));
                                अगर (bias_n) {
                                    योग_2 += bias_n[m0];
                                }
                                atomic_store_f32((अस्थिर भग्न *) dst, योग_2);
                            }
                        }
                    } अन्यथा {
                        क्रम (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            स्थिर block_q8_0 * q_row = (स्थिर block_q8_0 *) (src0_ptr2 + m * nb01);

                            भग्न योग_2 = q8_dot_compute(q_row, b_col_base, K_blocks);

                            भग्न * dst_entry = (भग्न *) (dst_ptr2 + n * nbd1 + m * माप(भग्न));
                            अगर (bias_n) {
                                योग_2 += bias_n[m];
                            }
                            atomic_store_f32((अस्थिर भग्न *) dst_entry, योग_2);
                        }
                    }

                    q8_dot_end(&q8_state);
                }
            }
        }
    }

#अगर_घोषित ET_UBERKERNEL
    FENCE;
    evict_region_past_l2(params->dst.तथ्य, tensor_bytes(&params->dst));
    WAIT_CACHEOPS;
    FENCE;
#पूर्ण_अगर
    वापस 0;
}
