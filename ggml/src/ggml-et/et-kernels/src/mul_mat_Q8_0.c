//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#অন্তর্ভাব "block_ops.h"
#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"
#অন্তর্ভাব "quants.h"

#অন্তর্ভাব <stdint.h>

#define STRIDE_M                   2048 /* 32 shires x 32 minions x 2 harts */
#define STRIDE_M_KSPLIT            1024 /* 32 shires x 32 minions (both harts share rows) */
#define KSPLIT_MIN_K_BLOCKS        256  /* K >= 8192 elements */
#define KSPLIT_SMALL_ROWS_K_BLOCKS 64   /* K >= 2048 elements for very small M */
#define KSPLIT_MAX_ROWS            8    /* max rows per minion for K-split */
#define TILE_KB                    256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */
#define KSPLIT_GROUP_ROWS          4
#define SIMPLE_X2_ROWS             2

স্থির অন্তঃস্থ size_t tensor_bytes(ধ্রুব সংরচনা ggml_tensor * t) {
    প্রত্যাগম (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_mm_q8_params * params, শূন্য * env) {
    uint64_t hart_id = get_hart_id();

    // Matrix dimensions
    ধ্রুব int64_t K    = params->src0.ne[0];
    ধ্রুব int64_t M    = params->src0.ne[1];
    ধ্রুব int64_t N    = params->src1.ne[1];
    ধ্রুব int64_t ne02 = params->src0.ne[2];
    ধ্রুব int64_t ne03 = params->src0.ne[3];
    ধ্রুব int64_t ne12 = params->src1.ne[2];
    ধ্রুব int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    ধ্রুব size_t nb01 = params->src0.nb[1];
    ধ্রুব size_t nb02 = params->src0.nb[2];
    ধ্রুব size_t nb03 = params->src0.nb[3];

    ধ্রুব size_t nb11 = params->src1.nb[1];
    ধ্রুব size_t nb12 = params->src1.nb[2];
    ধ্রুব size_t nb13 = params->src1.nb[3];

    ধ্রুব size_t nbd1 = params->dst.nb[1];
    ধ্রুব size_t nbd2 = params->dst.nb[2];
    ধ্রুব size_t nbd3 = params->dst.nb[3];

    // Optional residual bias
    ধ্রুব বর্ণ * bias_base = (ধ্রুব বর্ণ *) params->bias.data;
    ধ্রুব size_t nbb1      = params->bias.nb[1];
    ধ্রুব size_t nbb2      = params->bias.nb[2];
    ধ্রুব size_t nbb3      = params->bias.nb[3];

    // Q8_0 block size is 32
    ধ্রুব int64_t K_blocks      = K / 32;
    ধ্রুব পূর্ণাঙ্ক     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    ধ্রুব int64_t r2 = ne12 / ne02;
    ধ্রুব int64_t r3 = ne13 / ne03;

    // K-split decision
    ধ্রুব int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    ধ্রুব int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    ধ্রুব পূর্ণাঙ্ক     is_hart1              = hart_id & 1;
    ধ্রুব int64_t rows_per_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    ধ্রুব int64_t k_half                = K_blocks / 2;
    ধ্রুব পূর্ণাঙ্ক     use_ksplit_small_rows = (rows_per_minion <= 2) && (K_blocks >= KSPLIT_SMALL_ROWS_K_BLOCKS);
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
    ধ্রুব পূর্ণাঙ্ক     use_ksplit            = ((K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion <= KSPLIT_MAX_ROWS) &&
                                           (rows_per_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_rows;
    ধ্রুব পূর্ণাঙ্ক     use_ksplit_group      = !use_ksplit && (K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion > 4) &&
                                          (rows_per_minion <= KSPLIT_MAX_ROWS);

    evict_region_past_l2(params->src1.data, tensor_bytes(&params->src1));
    যদি (params->bias.data) {
        evict_region_past_l2(params->bias.data, tensor_bytes(&params->bias));
    }

    যদি (use_ksplit) {
        /* Each hart processes half the K dimension */
        ধ্রুব int64_t k_start = is_hart1 ? k_half : 0;
        ধ্রুব int64_t k_len   = is_hart1 ? (K_blocks - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        চঞ্চল ভিন্ন * l2scp_slot = (চঞ্চল ভিন্ন *) et_shire_l2scp_local(local_minion * 64);

        প্রতি (int64_t i3 = 0; i3 < ne13; i3++) {
            ধ্রুব int64_t i03       = i3 / r3;
            ধ্রুব বর্ণ *  src0_ptr3 = (ধ্রুব বর্ণ *) params->src0.data + i03 * nb03;
            ধ্রুব বর্ণ *  src1_ptr3 = (ধ্রুব বর্ণ *) params->src1.data + i3 * nb13;
            বর্ণ *        dst_ptr3  = (বর্ণ *) params->dst.data + i3 * nbd3;
            ধ্রুব বর্ণ *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (ধ্রুব বর্ণ *) 0;

            প্রতি (int64_t i2 = 0; i2 < ne12; i2++) {
                ধ্রুব int64_t i02       = i2 / r2;
                ধ্রুব বর্ণ *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ধ্রুব বর্ণ *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                বর্ণ *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ধ্রুব বর্ণ *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ধ্রুব বর্ণ *) 0;

                প্রতি (int64_t n = 0; n < N; n++) {
                    ধ্রুব ভিন্ন * b_col_base = (ধ্রুব ভিন্ন *) (src1_ptr2 + n * nb11);
                    ধ্রুব ভিন্ন * bias_n     = bias_ptr2 ? (ধ্রুব ভিন্ন *) (bias_ptr2 + n * nbb1) : (ধ্রুব ভিন্ন *) 0;

                    প্রতি (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        ধ্রুব block_q8_0 * q_row = (ধ্রুব block_q8_0 *) (src0_ptr2 + m * nb01);

                        ভিন্ন partial = compute_row_dot_q8_0(q_row + k_start, b_col_base + k_start * 32, k_len);

                        যদি (is_hart1) {
                            *l2scp_slot = partial;
                            FENCE;
                            flush_to_l2((ধ্রুব শূন্য *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } অন্যথা {
                            et_sem_wait(ET_BARRIER_MINION);
                            ভিন্ন other = *l2scp_slot;
                            et_sem_post(ET_BARRIER_MINION);

                            ভিন্ন * dst_entry = (ভিন্ন *) (dst_ptr2 + n * nbd1 + m * পরিমাণ(ভিন্ন));
                            ভিন্ন   sum       = partial + other;
                            যদি (bias_n) {
                                sum += bias_n[m];
                            }
                            atomic_store_f32((চঞ্চল ভিন্ন *) dst_entry, sum);
                        }
                    }
                }
            }
        }
    } অন্যথা যদি (use_ksplit_group) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime.
         *
         * Both harts process the same 4-row group, each on half of K, and
         * exchange 4 partial sums once per group instead of once per row.
         * This keeps the K-split bandwidth benefit while cutting semaphore
         * traffic by 4x relative to the old per-row exchange.
         */
        ধ্রুব int64_t    k_start    = is_hart1 ? k_half : 0;
        ধ্রুব int64_t    k_len      = is_hart1 ? (K_blocks - k_half) : k_half;
        চঞ্চল ভিন্ন * l2scp_slot = (চঞ্চল ভিন্ন *) et_shire_l2scp_local(local_minion * 64);

        প্রতি (int64_t i3 = 0; i3 < ne13; i3++) {
            ধ্রুব int64_t i03       = i3 / r3;
            ধ্রুব বর্ণ *  src0_ptr3 = (ধ্রুব বর্ণ *) params->src0.data + i03 * nb03;
            ধ্রুব বর্ণ *  src1_ptr3 = (ধ্রুব বর্ণ *) params->src1.data + i3 * nb13;
            বর্ণ *        dst_ptr3  = (বর্ণ *) params->dst.data + i3 * nbd3;
            ধ্রুব বর্ণ *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (ধ্রুব বর্ণ *) 0;

            প্রতি (int64_t i2 = 0; i2 < ne12; i2++) {
                ধ্রুব int64_t i02       = i2 / r2;
                ধ্রুব বর্ণ *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ধ্রুব বর্ণ *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                বর্ণ *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ধ্রুব বর্ণ *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ধ্রুব বর্ণ *) 0;

                প্রতি (int64_t n = 0; n < N; n++) {
                    ধ্রুব ভিন্ন * b_col_base = (ধ্রুব ভিন্ন *) (src1_ptr2 + n * nb11);
                    ধ্রুব ভিন্ন * bias_n     = bias_ptr2 ? (ধ্রুব ভিন্ন *) (bias_ptr2 + n * nbb1) : (ধ্রুব ভিন্ন *) 0;

                    প্রতি (int64_t m_base = minion_id; m_base < M; m_base += STRIDE_M_KSPLIT * KSPLIT_GROUP_ROWS) {
                        ধ্রুব int64_t m0 = m_base;
                        ধ্রুব int64_t m1 = m0 + STRIDE_M_KSPLIT;
                        ধ্রুব int64_t m2 = m1 + STRIDE_M_KSPLIT;
                        ধ্রুব int64_t m3 = m2 + STRIDE_M_KSPLIT;

                        ভিন্ন s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        প্রতি (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = k_len - kb;
                            যদি (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            যদি (tile_len <= 0) {
                                ভঙ্গ;
                            }
                            ধ্রুব ভিন্ন * b_tile = b_col_base + (k_start + kb) * 32;
                            ধ্রুব int64_t row_kb = k_start + kb;

                            যদি (m0 < M) {
                                s0 += compute_row_dot_q8_0((ধ্রুব block_q8_0 *) (src0_ptr2 + m0 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            যদি (m1 < M) {
                                s1 += compute_row_dot_q8_0((ধ্রুব block_q8_0 *) (src0_ptr2 + m1 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            যদি (m2 < M) {
                                s2 += compute_row_dot_q8_0((ধ্রুব block_q8_0 *) (src0_ptr2 + m2 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            যদি (m3 < M) {
                                s3 += compute_row_dot_q8_0((ধ্রুব block_q8_0 *) (src0_ptr2 + m3 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                        }

                        যদি (is_hart1) {
                            l2scp_slot[0] = s0;
                            l2scp_slot[1] = s1;
                            l2scp_slot[2] = s2;
                            l2scp_slot[3] = s3;
                            FENCE;
                            flush_to_l2((ধ্রুব শূন্য *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } অন্যথা {
                            et_sem_wait(ET_BARRIER_MINION);
                            ধ্রুব ভিন্ন p0 = l2scp_slot[0];
                            ধ্রুব ভিন্ন p1 = l2scp_slot[1];
                            ধ্রুব ভিন্ন p2 = l2scp_slot[2];
                            ধ্রুব ভিন্ন p3 = l2scp_slot[3];
                            et_sem_post(ET_BARRIER_MINION);

                            ভিন্ন *     c_base = (ভিন্ন *) (dst_ptr2 + n * nbd1);
                            ধ্রুব ভিন্ন b0     = bias_n ? bias_n[m0] : 0.0f;
                            ধ্রুব ভিন্ন b1     = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                            ধ্রুব ভিন্ন b2     = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                            ধ্রুব ভিন্ন b3     = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                            যদি (m0 < M) {
                                atomic_store_f32((চঞ্চল ভিন্ন *) (c_base + m0), s0 + p0 + b0);
                            }
                            যদি (m1 < M) {
                                atomic_store_f32((চঞ্চল ভিন্ন *) (c_base + m1), s1 + p1 + b1);
                            }
                            যদি (m2 < M) {
                                atomic_store_f32((চঞ্চল ভিন্ন *) (c_base + m2), s2 + p2 + b2);
                            }
                            যদি (m3 < M) {
                                atomic_store_f32((চঞ্চল ভিন্ন *) (c_base + m3), s3 + p3 + b3);
                            }
                        }
                    }
                }
            }
        }
    } অন্যথা যদি (K_blocks > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        প্রতি (int64_t i3 = 0; i3 < ne13; i3++) {
            ধ্রুব int64_t i03       = i3 / r3;
            ধ্রুব বর্ণ *  src0_ptr3 = (ধ্রুব বর্ণ *) params->src0.data + i03 * nb03;
            ধ্রুব বর্ণ *  src1_ptr3 = (ধ্রুব বর্ণ *) params->src1.data + i3 * nb13;
            বর্ণ *        dst_ptr3  = (বর্ণ *) params->dst.data + i3 * nbd3;
            ধ্রুব বর্ণ *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (ধ্রুব বর্ণ *) 0;

            প্রতি (int64_t i2 = 0; i2 < ne12; i2++) {
                ধ্রুব int64_t i02       = i2 / r2;
                ধ্রুব বর্ণ *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ধ্রুব বর্ণ *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                বর্ণ *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ধ্রুব বর্ণ *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ধ্রুব বর্ণ *) 0;

                প্রতি (int64_t n = 0; n < N; n++) {
                    ধ্রুব ভিন্ন * b_col_base = (ধ্রুব ভিন্ন *) (src1_ptr2 + n * nb11);
                    ধ্রুব ভিন্ন * bias_n     = bias_ptr2 ? (ধ্রুব ভিন্ন *) (bias_ptr2 + n * nbb1) : (ধ্রুব ভিন্ন *) 0;

                    প্রতি (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        ধ্রুব int64_t m1 = m0 + STRIDE_M;
                        ধ্রুব int64_t m2 = m0 + STRIDE_M * 2;
                        ধ্রুব int64_t m3 = m0 + STRIDE_M * 3;

                        ভিন্ন s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        প্রতি (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = K_blocks - kb;
                            যদি (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            ধ্রুব ভিন্ন * b_tile = b_col_base + kb * 32;

                            s0 += compute_row_dot_q8_0((ধ্রুব block_q8_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            যদি (m1 < M) {
                                s1 += compute_row_dot_q8_0((ধ্রুব block_q8_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            যদি (m2 < M) {
                                s2 += compute_row_dot_q8_0((ধ্রুব block_q8_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            যদি (m3 < M) {
                                s3 += compute_row_dot_q8_0((ধ্রুব block_q8_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        ভিন্ন *     dst_base = (ভিন্ন *) (dst_ptr2 + n * nbd1);
                        ধ্রুব ভিন্ন b0       = bias_n ? bias_n[m0] : 0.0f;
                        ধ্রুব ভিন্ন b1       = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                        ধ্রুব ভিন্ন b2       = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                        ধ্রুব ভিন্ন b3       = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                        atomic_store_f32((চঞ্চল ভিন্ন *) (dst_base + m0), s0 + b0);
                        যদি (m1 < M) {
                            atomic_store_f32((চঞ্চল ভিন্ন *) (dst_base + m1), s1 + b1);
                        }
                        যদি (m2 < M) {
                            atomic_store_f32((চঞ্চল ভিন্ন *) (dst_base + m2), s2 + b2);
                        }
                        যদি (m3 < M) {
                            atomic_store_f32((চঞ্চল ভিন্ন *) (dst_base + m3), s3 + b3);
                        }
                    }
                }
            }
        }
    } অন্যথা {
        /*
         * Simple path for small K.
         *
         * When `nb01` is 32-byte aligned, every row has the same block-alignment
         * pattern. That lets us compute two rows together and reuse each loaded
         * B chunk across both rows instead of reloading it in a second dot call.
         */
        প্রতি (int64_t i3 = 0; i3 < ne13; i3++) {
            ধ্রুব int64_t i03       = i3 / r3;
            ধ্রুব বর্ণ *  src0_ptr3 = (ধ্রুব বর্ণ *) params->src0.data + i03 * nb03;
            ধ্রুব বর্ণ *  src1_ptr3 = (ধ্রুব বর্ণ *) params->src1.data + i3 * nb13;
            বর্ণ *        dst_ptr3  = (বর্ণ *) params->dst.data + i3 * nbd3;
            ধ্রুব বর্ণ *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (ধ্রুব বর্ণ *) 0;

            প্রতি (int64_t i2 = 0; i2 < ne12; i2++) {
                ধ্রুব int64_t i02       = i2 / r2;
                ধ্রুব বর্ণ *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ধ্রুব বর্ণ *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                বর্ণ *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ধ্রুব বর্ণ *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ধ্রুব বর্ণ *) 0;

                প্রতি (int64_t n = 0; n < N; n++) {
                    ধ্রুব ভিন্ন * b_col_base = (ধ্রুব ভিন্ন *) (src1_ptr2 + n * nb11);
                    ধ্রুব ভিন্ন * bias_n     = bias_ptr2 ? (ধ্রুব ভিন্ন *) (bias_ptr2 + n * nbb1) : (ধ্রুব ভিন্ন *) 0;
                    q8_dot_state  q8_state;
                    q8_dot_begin(&q8_state);

                    যদি (use_simple_x2) {
                        প্রতি (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_ROWS) {
                            ধ্রুব int64_t      m1     = m0 + STRIDE_M;
                            ধ্রুব block_q8_0 * q_row0 = (ধ্রুব block_q8_0 *) (src0_ptr2 + m0 * nb01);

                            যদি (m1 < M) {
                                ধ্রুব block_q8_0 * q_row1 = (ধ্রুব block_q8_0 *) (src0_ptr2 + m1 * nb01);
                                ভিন্ন              s0, s1;
                                q8_dot_compute_x2_aligned(q_row0, q_row1, b_col_base, K_blocks, &s0, &s1);

                                ভিন্ন * dst0 = (ভিন্ন *) (dst_ptr2 + n * nbd1 + m0 * পরিমাণ(ভিন্ন));
                                ভিন্ন * dst1 = (ভিন্ন *) (dst_ptr2 + n * nbd1 + m1 * পরিমাণ(ভিন্ন));
                                যদি (bias_n) {
                                    s0 += bias_n[m0];
                                    s1 += bias_n[m1];
                                }
                                atomic_store_f32((চঞ্চল ভিন্ন *) dst0, s0);
                                atomic_store_f32((চঞ্চল ভিন্ন *) dst1, s1);
                            } অন্যথা {
                                ভিন্ন   sum = q8_dot_compute(q_row0, b_col_base, K_blocks);
                                ভিন্ন * dst = (ভিন্ন *) (dst_ptr2 + n * nbd1 + m0 * পরিমাণ(ভিন্ন));
                                যদি (bias_n) {
                                    sum += bias_n[m0];
                                }
                                atomic_store_f32((চঞ্চল ভিন্ন *) dst, sum);
                            }
                        }
                    } অন্যথা {
                        প্রতি (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            ধ্রুব block_q8_0 * q_row = (ধ্রুব block_q8_0 *) (src0_ptr2 + m * nb01);

                            ভিন্ন sum = q8_dot_compute(q_row, b_col_base, K_blocks);

                            ভিন্ন * dst_entry = (ভিন্ন *) (dst_ptr2 + n * nbd1 + m * পরিমাণ(ভিন্ন));
                            যদি (bias_n) {
                                sum += bias_n[m];
                            }
                            atomic_store_f32((চঞ্চল ভিন্ন *) dst_entry, sum);
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
    প্রত্যাগম 0;
}
