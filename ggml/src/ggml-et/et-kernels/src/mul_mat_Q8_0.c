//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#અન્તર્ભાવ "block_ops.h"
#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "math_fp.h"
#અન્તર્ભાવ "platform.h"
#અન્તર્ભાવ "quants.h"

#અન્તર્ભાવ <stdint.h>

#define STRIDE_M                   2048 /* 32 shires x 32 minions x 2 harts */
#define STRIDE_M_KSPLIT            1024 /* 32 shires x 32 minions (both harts share rows) */
#define KSPLIT_MIN_K_BLOCKS        256  /* K >= 8192 elements */
#define KSPLIT_SMALL_ROWS_K_BLOCKS 64   /* K >= 2048 elements for very small M */
#define KSPLIT_MAX_ROWS            8    /* max rows per minion for K-split */
#define TILE_KB                    256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */
#define KSPLIT_GROUP_ROWS          4
#define SIMPLE_X2_ROWS             2

સ્થિર અંતઃરેખ size_t tensor_bytes(અચલ સંરચના ggml_tensor * t) {
    પાછા (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

પૂર્ણાંક entry_point(સંરચના ggml_et_mm_q8_params * params, શૂન્ય * env) {
    uint64_t hart_id = get_hart_id();

    // Matrix dimensions
    અચલ int64_t K    = params->src0.ne[0];
    અચલ int64_t M    = params->src0.ne[1];
    અચલ int64_t N    = params->src1.ne[1];
    અચલ int64_t ne02 = params->src0.ne[2];
    અચલ int64_t ne03 = params->src0.ne[3];
    અચલ int64_t ne12 = params->src1.ne[2];
    અચલ int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    અચલ size_t nb01 = params->src0.nb[1];
    અચલ size_t nb02 = params->src0.nb[2];
    અચલ size_t nb03 = params->src0.nb[3];

    અચલ size_t nb11 = params->src1.nb[1];
    અચલ size_t nb12 = params->src1.nb[2];
    અચલ size_t nb13 = params->src1.nb[3];

    અચલ size_t nbd1 = params->dst.nb[1];
    અચલ size_t nbd2 = params->dst.nb[2];
    અચલ size_t nbd3 = params->dst.nb[3];

    // Optional residual bias
    અચલ અક્ષર * bias_base = (અચલ અક્ષર *) params->bias.data;
    અચલ size_t nbb1      = params->bias.nb[1];
    અચલ size_t nbb2      = params->bias.nb[2];
    અચલ size_t nbb3      = params->bias.nb[3];

    // Q8_0 block size is 32
    અચલ int64_t K_blocks      = K / 32;
    અચલ પૂર્ણાંક     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    અચલ int64_t r2 = ne12 / ne02;
    અચલ int64_t r3 = ne13 / ne03;

    // K-split decision
    અચલ int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    અચલ int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    અચલ પૂર્ણાંક     is_hart1              = hart_id & 1;
    અચલ int64_t rows_per_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    અચલ int64_t k_half                = K_blocks / 2;
    અચલ પૂર્ણાંક     use_ksplit_small_rows = (rows_per_minion <= 2) && (K_blocks >= KSPLIT_SMALL_ROWS_K_BLOCKS);
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
    અચલ પૂર્ણાંક     use_ksplit            = ((K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion <= KSPLIT_MAX_ROWS) &&
                                           (rows_per_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_rows;
    અચલ પૂર્ણાંક     use_ksplit_group      = !use_ksplit && (K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion > 4) &&
                                          (rows_per_minion <= KSPLIT_MAX_ROWS);

    evict_region_past_l2(params->src1.data, tensor_bytes(&params->src1));
    જો (params->bias.data) {
        evict_region_past_l2(params->bias.data, tensor_bytes(&params->bias));
    }

    જો (use_ksplit) {
        /* Each hart processes half the K dimension */
        અચલ int64_t k_start = is_hart1 ? k_half : 0;
        અચલ int64_t k_len   = is_hart1 ? (K_blocks - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        અસ્થિર દશાંશ * l2scp_slot = (અસ્થિર દશાંશ *) et_shire_l2scp_local(local_minion * 64);

        માટે (int64_t i3 = 0; i3 < ne13; i3++) {
            અચલ int64_t i03       = i3 / r3;
            અચલ અક્ષર *  src0_ptr3 = (અચલ અક્ષર *) params->src0.data + i03 * nb03;
            અચલ અક્ષર *  src1_ptr3 = (અચલ અક્ષર *) params->src1.data + i3 * nb13;
            અક્ષર *        dst_ptr3  = (અક્ષર *) params->dst.data + i3 * nbd3;
            અચલ અક્ષર *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (અચલ અક્ષર *) 0;

            માટે (int64_t i2 = 0; i2 < ne12; i2++) {
                અચલ int64_t i02       = i2 / r2;
                અચલ અક્ષર *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                અચલ અક્ષર *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                અક્ષર *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                અચલ અક્ષર *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (અચલ અક્ષર *) 0;

                માટે (int64_t n = 0; n < N; n++) {
                    અચલ દશાંશ * b_col_base = (અચલ દશાંશ *) (src1_ptr2 + n * nb11);
                    અચલ દશાંશ * bias_n     = bias_ptr2 ? (અચલ દશાંશ *) (bias_ptr2 + n * nbb1) : (અચલ દશાંશ *) 0;

                    માટે (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        અચલ block_q8_0 * q_row = (અચલ block_q8_0 *) (src0_ptr2 + m * nb01);

                        દશાંશ partial = compute_row_dot_q8_0(q_row + k_start, b_col_base + k_start * 32, k_len);

                        જો (is_hart1) {
                            *l2scp_slot = partial;
                            FENCE;
                            flush_to_l2((અચલ શૂન્ય *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } નહીંતર {
                            et_sem_wait(ET_BARRIER_MINION);
                            દશાંશ other = *l2scp_slot;
                            et_sem_post(ET_BARRIER_MINION);

                            દશાંશ * dst_entry = (દશાંશ *) (dst_ptr2 + n * nbd1 + m * કદ(દશાંશ));
                            દશાંશ   sum       = partial + other;
                            જો (bias_n) {
                                sum += bias_n[m];
                            }
                            atomic_store_f32((અસ્થિર દશાંશ *) dst_entry, sum);
                        }
                    }
                }
            }
        }
    } નહીંતર જો (use_ksplit_group) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime.
         *
         * Both harts process the same 4-row group, each on half of K, and
         * exchange 4 partial sums once per group instead of once per row.
         * This keeps the K-split bandwidth benefit while cutting semaphore
         * traffic by 4x relative to the old per-row exchange.
         */
        અચલ int64_t    k_start    = is_hart1 ? k_half : 0;
        અચલ int64_t    k_len      = is_hart1 ? (K_blocks - k_half) : k_half;
        અસ્થિર દશાંશ * l2scp_slot = (અસ્થિર દશાંશ *) et_shire_l2scp_local(local_minion * 64);

        માટે (int64_t i3 = 0; i3 < ne13; i3++) {
            અચલ int64_t i03       = i3 / r3;
            અચલ અક્ષર *  src0_ptr3 = (અચલ અક્ષર *) params->src0.data + i03 * nb03;
            અચલ અક્ષર *  src1_ptr3 = (અચલ અક્ષર *) params->src1.data + i3 * nb13;
            અક્ષર *        dst_ptr3  = (અક્ષર *) params->dst.data + i3 * nbd3;
            અચલ અક્ષર *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (અચલ અક્ષર *) 0;

            માટે (int64_t i2 = 0; i2 < ne12; i2++) {
                અચલ int64_t i02       = i2 / r2;
                અચલ અક્ષર *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                અચલ અક્ષર *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                અક્ષર *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                અચલ અક્ષર *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (અચલ અક્ષર *) 0;

                માટે (int64_t n = 0; n < N; n++) {
                    અચલ દશાંશ * b_col_base = (અચલ દશાંશ *) (src1_ptr2 + n * nb11);
                    અચલ દશાંશ * bias_n     = bias_ptr2 ? (અચલ દશાંશ *) (bias_ptr2 + n * nbb1) : (અચલ દશાંશ *) 0;

                    માટે (int64_t m_base = minion_id; m_base < M; m_base += STRIDE_M_KSPLIT * KSPLIT_GROUP_ROWS) {
                        અચલ int64_t m0 = m_base;
                        અચલ int64_t m1 = m0 + STRIDE_M_KSPLIT;
                        અચલ int64_t m2 = m1 + STRIDE_M_KSPLIT;
                        અચલ int64_t m3 = m2 + STRIDE_M_KSPLIT;

                        દશાંશ s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        માટે (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = k_len - kb;
                            જો (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            જો (tile_len <= 0) {
                                તોડો;
                            }
                            અચલ દશાંશ * b_tile = b_col_base + (k_start + kb) * 32;
                            અચલ int64_t row_kb = k_start + kb;

                            જો (m0 < M) {
                                s0 += compute_row_dot_q8_0((અચલ block_q8_0 *) (src0_ptr2 + m0 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            જો (m1 < M) {
                                s1 += compute_row_dot_q8_0((અચલ block_q8_0 *) (src0_ptr2 + m1 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            જો (m2 < M) {
                                s2 += compute_row_dot_q8_0((અચલ block_q8_0 *) (src0_ptr2 + m2 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            જો (m3 < M) {
                                s3 += compute_row_dot_q8_0((અચલ block_q8_0 *) (src0_ptr2 + m3 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                        }

                        જો (is_hart1) {
                            l2scp_slot[0] = s0;
                            l2scp_slot[1] = s1;
                            l2scp_slot[2] = s2;
                            l2scp_slot[3] = s3;
                            FENCE;
                            flush_to_l2((અચલ શૂન્ય *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } નહીંતર {
                            et_sem_wait(ET_BARRIER_MINION);
                            અચલ દશાંશ p0 = l2scp_slot[0];
                            અચલ દશાંશ p1 = l2scp_slot[1];
                            અચલ દશાંશ p2 = l2scp_slot[2];
                            અચલ દશાંશ p3 = l2scp_slot[3];
                            et_sem_post(ET_BARRIER_MINION);

                            દશાંશ *     c_base = (દશાંશ *) (dst_ptr2 + n * nbd1);
                            અચલ દશાંશ b0     = bias_n ? bias_n[m0] : 0.0f;
                            અચલ દશાંશ b1     = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                            અચલ દશાંશ b2     = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                            અચલ દશાંશ b3     = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                            જો (m0 < M) {
                                atomic_store_f32((અસ્થિર દશાંશ *) (c_base + m0), s0 + p0 + b0);
                            }
                            જો (m1 < M) {
                                atomic_store_f32((અસ્થિર દશાંશ *) (c_base + m1), s1 + p1 + b1);
                            }
                            જો (m2 < M) {
                                atomic_store_f32((અસ્થિર દશાંશ *) (c_base + m2), s2 + p2 + b2);
                            }
                            જો (m3 < M) {
                                atomic_store_f32((અસ્થિર દશાંશ *) (c_base + m3), s3 + p3 + b3);
                            }
                        }
                    }
                }
            }
        }
    } નહીંતર જો (K_blocks > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        માટે (int64_t i3 = 0; i3 < ne13; i3++) {
            અચલ int64_t i03       = i3 / r3;
            અચલ અક્ષર *  src0_ptr3 = (અચલ અક્ષર *) params->src0.data + i03 * nb03;
            અચલ અક્ષર *  src1_ptr3 = (અચલ અક્ષર *) params->src1.data + i3 * nb13;
            અક્ષર *        dst_ptr3  = (અક્ષર *) params->dst.data + i3 * nbd3;
            અચલ અક્ષર *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (અચલ અક્ષર *) 0;

            માટે (int64_t i2 = 0; i2 < ne12; i2++) {
                અચલ int64_t i02       = i2 / r2;
                અચલ અક્ષર *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                અચલ અક્ષર *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                અક્ષર *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                અચલ અક્ષર *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (અચલ અક્ષર *) 0;

                માટે (int64_t n = 0; n < N; n++) {
                    અચલ દશાંશ * b_col_base = (અચલ દશાંશ *) (src1_ptr2 + n * nb11);
                    અચલ દશાંશ * bias_n     = bias_ptr2 ? (અચલ દશાંશ *) (bias_ptr2 + n * nbb1) : (અચલ દશાંશ *) 0;

                    માટે (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        અચલ int64_t m1 = m0 + STRIDE_M;
                        અચલ int64_t m2 = m0 + STRIDE_M * 2;
                        અચલ int64_t m3 = m0 + STRIDE_M * 3;

                        દશાંશ s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        માટે (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = K_blocks - kb;
                            જો (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            અચલ દશાંશ * b_tile = b_col_base + kb * 32;

                            s0 += compute_row_dot_q8_0((અચલ block_q8_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            જો (m1 < M) {
                                s1 += compute_row_dot_q8_0((અચલ block_q8_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            જો (m2 < M) {
                                s2 += compute_row_dot_q8_0((અચલ block_q8_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            જો (m3 < M) {
                                s3 += compute_row_dot_q8_0((અચલ block_q8_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        દશાંશ *     dst_base = (દશાંશ *) (dst_ptr2 + n * nbd1);
                        અચલ દશાંશ b0       = bias_n ? bias_n[m0] : 0.0f;
                        અચલ દશાંશ b1       = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                        અચલ દશાંશ b2       = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                        અચલ દશાંશ b3       = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                        atomic_store_f32((અસ્થિર દશાંશ *) (dst_base + m0), s0 + b0);
                        જો (m1 < M) {
                            atomic_store_f32((અસ્થિર દશાંશ *) (dst_base + m1), s1 + b1);
                        }
                        જો (m2 < M) {
                            atomic_store_f32((અસ્થિર દશાંશ *) (dst_base + m2), s2 + b2);
                        }
                        જો (m3 < M) {
                            atomic_store_f32((અસ્થિર દશાંશ *) (dst_base + m3), s3 + b3);
                        }
                    }
                }
            }
        }
    } નહીંતર {
        /*
         * Simple path for small K.
         *
         * When `nb01` is 32-byte aligned, every row has the same block-alignment
         * pattern. That lets us compute two rows together and reuse each loaded
         * B chunk across both rows instead of reloading it in a second dot call.
         */
        માટે (int64_t i3 = 0; i3 < ne13; i3++) {
            અચલ int64_t i03       = i3 / r3;
            અચલ અક્ષર *  src0_ptr3 = (અચલ અક્ષર *) params->src0.data + i03 * nb03;
            અચલ અક્ષર *  src1_ptr3 = (અચલ અક્ષર *) params->src1.data + i3 * nb13;
            અક્ષર *        dst_ptr3  = (અક્ષર *) params->dst.data + i3 * nbd3;
            અચલ અક્ષર *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (અચલ અક્ષર *) 0;

            માટે (int64_t i2 = 0; i2 < ne12; i2++) {
                અચલ int64_t i02       = i2 / r2;
                અચલ અક્ષર *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                અચલ અક્ષર *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                અક્ષર *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                અચલ અક્ષર *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (અચલ અક્ષર *) 0;

                માટે (int64_t n = 0; n < N; n++) {
                    અચલ દશાંશ * b_col_base = (અચલ દશાંશ *) (src1_ptr2 + n * nb11);
                    અચલ દશાંશ * bias_n     = bias_ptr2 ? (અચલ દશાંશ *) (bias_ptr2 + n * nbb1) : (અચલ દશાંશ *) 0;
                    q8_dot_state  q8_state;
                    q8_dot_begin(&q8_state);

                    જો (use_simple_x2) {
                        માટે (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_ROWS) {
                            અચલ int64_t      m1     = m0 + STRIDE_M;
                            અચલ block_q8_0 * q_row0 = (અચલ block_q8_0 *) (src0_ptr2 + m0 * nb01);

                            જો (m1 < M) {
                                અચલ block_q8_0 * q_row1 = (અચલ block_q8_0 *) (src0_ptr2 + m1 * nb01);
                                દશાંશ              s0, s1;
                                q8_dot_compute_x2_aligned(q_row0, q_row1, b_col_base, K_blocks, &s0, &s1);

                                દશાંશ * dst0 = (દશાંશ *) (dst_ptr2 + n * nbd1 + m0 * કદ(દશાંશ));
                                દશાંશ * dst1 = (દશાંશ *) (dst_ptr2 + n * nbd1 + m1 * કદ(દશાંશ));
                                જો (bias_n) {
                                    s0 += bias_n[m0];
                                    s1 += bias_n[m1];
                                }
                                atomic_store_f32((અસ્થિર દશાંશ *) dst0, s0);
                                atomic_store_f32((અસ્થિર દશાંશ *) dst1, s1);
                            } નહીંતર {
                                દશાંશ   sum = q8_dot_compute(q_row0, b_col_base, K_blocks);
                                દશાંશ * dst = (દશાંશ *) (dst_ptr2 + n * nbd1 + m0 * કદ(દશાંશ));
                                જો (bias_n) {
                                    sum += bias_n[m0];
                                }
                                atomic_store_f32((અસ્થિર દશાંશ *) dst, sum);
                            }
                        }
                    } નહીંતર {
                        માટે (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            અચલ block_q8_0 * q_row = (અચલ block_q8_0 *) (src0_ptr2 + m * nb01);

                            દશાંશ sum = q8_dot_compute(q_row, b_col_base, K_blocks);

                            દશાંશ * dst_entry = (દશાંશ *) (dst_ptr2 + n * nbd1 + m * કદ(દશાંશ));
                            જો (bias_n) {
                                sum += bias_n[m];
                            }
                            atomic_store_f32((અસ્થિર દશાંશ *) dst_entry, sum);
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
    પાછા 0;
}
