//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#ضمن "block_ops.h"
#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"
#ضمن "quants.h"

#ضمن <stdint.h>

#define STRIDE_M                   2048 /* 32 shires x 32 minions x 2 harts */
#define STRIDE_M_KSPLIT            1024 /* 32 shires x 32 minions (both harts share rows) */
#define KSPLIT_MIN_K_BLOCKS        256  /* K >= 8192 elements */
#define KSPLIT_SMALL_ROWS_K_BLOCKS 64   /* K >= 2048 elements for very small M */
#define KSPLIT_MAX_ROWS            8    /* max rows per minion for K-split */
#define TILE_KB                    256  /* K-tile size in Q4_0 blocks (8192 elems, 32KB B data) */
#define KSPLIT_GROUP_ROWS          4
#define SIMPLE_X2_ROWS             2

صحيح entry_point(بنية ggml_et_نبيو_params * params, فراغ * env) {
    uint64_t hart_id = get_hart_id();

    // Matrix dimensions
    ثابت int64_t K    = params->src0.ne[0];
    ثابت int64_t M    = params->src0.ne[1];
    ثابت int64_t N    = params->src1.ne[1];
    ثابت int64_t ne02 = params->src0.ne[2];
    ثابت int64_t ne03 = params->src0.ne[3];
    ثابت int64_t ne12 = params->src1.ne[2];
    ثابت int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    ثابت size_t nb01 = params->src0.nb[1];
    ثابت size_t nb02 = params->src0.nb[2];
    ثابت size_t nb03 = params->src0.nb[3];

    ثابت size_t nb11 = params->src1.nb[1];
    ثابت size_t nb12 = params->src1.nb[2];
    ثابت size_t nb13 = params->src1.nb[3];

    ثابت size_t nbd1 = params->dst.nb[1];
    ثابت size_t nbd2 = params->dst.nb[2];
    ثابت size_t nbd3 = params->dst.nb[3];

    // Q4_0 block size is 32
    ثابت int64_t K_blocks      = K / 32;
    ثابت صحيح     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    ثابت int64_t r2 = ne12 / ne02;
    ثابت int64_t r3 = ne13 / ne03;

    // K-split decision
    ثابت int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    ثابت int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    ثابت صحيح     is_hart1              = hart_id & 1;
    ثابت int64_t rows_per_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    ثابت int64_t k_half                = K_blocks / 2;
    ثابت صحيح     use_ksplit_small_rows = (rows_per_minion <= 2) && (K_blocks >= KSPLIT_SMALL_ROWS_K_BLOCKS);
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
    ثابت صحيح     use_ksplit            = ((K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion <= KSPLIT_MAX_ROWS) &&
                                           (rows_per_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_rows;
    ثابت صحيح     use_ksplit_group      = !use_ksplit && (K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion > 4) &&
                                          (rows_per_minion <= KSPLIT_MAX_ROWS);

    إذا (use_ksplit) {
        /* Each hart processes half the K dimension */
        ثابت int64_t k_start = is_hart1 ? k_half : 0;
        ثابت int64_t k_len   = is_hart1 ? (K_blocks - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        متطاير عائم * l2scp_slot = (متطاير عائم *) et_shire_l2scp_local(local_minion * 64);

        لكل (int64_t i3 = 0; i3 < ne13; i3++) {
            ثابت int64_t i03       = i3 / r3;
            ثابت حرف *  src0_ptr3 = (ثابت حرف *) params->src0.data + i03 * nb03;
            ثابت حرف *  src1_ptr3 = (ثابت حرف *) params->src1.data + i3 * nb13;
            حرف *        dst_ptr3  = (حرف *) params->dst.data + i3 * nbd3;

            لكل (int64_t i2 = 0; i2 < ne12; i2++) {
                ثابت int64_t i02       = i2 / r2;
                ثابت حرف *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ثابت حرف *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                حرف *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                لكل (int64_t n = 0; n < N; n++) {
                    ثابت عائم * b_col_base = (ثابت عائم *) (src1_ptr2 + n * nb11);

                    لكل (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        ثابت block_q4_0 * q_row = (ثابت block_q4_0 *) (src0_ptr2 + m * nb01);

                        عائم partial = compute_row_dot_q4_0(q_row + k_start, b_col_base + k_start * 32, k_len);

                        إذا (is_hart1) {
                            *l2scp_slot = partial;
                            FENCE;
                            flush_to_l2((ثابت فراغ *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } وإلا {
                            et_sem_wait(ET_BARRIER_MINION);
                            عائم other = *l2scp_slot;
                            et_sem_post(ET_BARRIER_MINION);

                            عائم * dst_entry = (عائم *) (dst_ptr2 + n * nbd1 + m * حجم(عائم));
                            atomic_store_f32((متطاير عائم *) dst_entry, partial + other);
                        }
                    }
                }
            }
        }
    } وإلا إذا (use_ksplit_group) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime.
         *
         * Both harts process the same 4-row group, each on half of K, and
         * exchange 4 partial sums once per group instead of once per row.
         * This keeps the K-split bandwidth benefit while cutting semaphore
         * traffic by 4x relative to the old per-row exchange.
         */
        ثابت int64_t    k_start    = is_hart1 ? k_half : 0;
        ثابت int64_t    k_len      = is_hart1 ? (K_blocks - k_half) : k_half;
        متطاير عائم * l2scp_slot = (متطاير عائم *) et_shire_l2scp_local(local_minion * 64);

        لكل (int64_t i3 = 0; i3 < ne13; i3++) {
            ثابت int64_t i03       = i3 / r3;
            ثابت حرف *  src0_ptr3 = (ثابت حرف *) params->src0.data + i03 * nb03;
            ثابت حرف *  src1_ptr3 = (ثابت حرف *) params->src1.data + i3 * nb13;
            حرف *        dst_ptr3  = (حرف *) params->dst.data + i3 * nbd3;

            لكل (int64_t i2 = 0; i2 < ne12; i2++) {
                ثابت int64_t i02       = i2 / r2;
                ثابت حرف *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ثابت حرف *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                حرف *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                لكل (int64_t n = 0; n < N; n++) {
                    ثابت عائم * b_col_base = (ثابت عائم *) (src1_ptr2 + n * nb11);

                    لكل (int64_t m_base = minion_id; m_base < M; m_base += STRIDE_M_KSPLIT * KSPLIT_GROUP_ROWS) {
                        ثابت int64_t m0 = m_base;
                        ثابت int64_t m1 = m0 + STRIDE_M_KSPLIT;
                        ثابت int64_t m2 = m1 + STRIDE_M_KSPLIT;
                        ثابت int64_t m3 = m2 + STRIDE_M_KSPLIT;

                        عائم s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        لكل (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = k_len - kb;
                            إذا (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            إذا (tile_len <= 0) {
                                اكسر;
                            }
                            ثابت عائم * b_tile = b_col_base + (k_start + kb) * 32;
                            ثابت int64_t row_kb = k_start + kb;

                            إذا (m0 < M) {
                                s0 += compute_row_dot_q4_0((ثابت block_q4_0 *) (src0_ptr2 + m0 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            إذا (m1 < M) {
                                s1 += compute_row_dot_q4_0((ثابت block_q4_0 *) (src0_ptr2 + m1 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            إذا (m2 < M) {
                                s2 += compute_row_dot_q4_0((ثابت block_q4_0 *) (src0_ptr2 + m2 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            إذا (m3 < M) {
                                s3 += compute_row_dot_q4_0((ثابت block_q4_0 *) (src0_ptr2 + m3 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                        }

                        إذا (is_hart1) {
                            l2scp_slot[0] = s0;
                            l2scp_slot[1] = s1;
                            l2scp_slot[2] = s2;
                            l2scp_slot[3] = s3;
                            FENCE;
                            flush_to_l2((ثابت فراغ *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } وإلا {
                            et_sem_wait(ET_BARRIER_MINION);
                            ثابت عائم p0 = l2scp_slot[0];
                            ثابت عائم p1 = l2scp_slot[1];
                            ثابت عائم p2 = l2scp_slot[2];
                            ثابت عائم p3 = l2scp_slot[3];
                            et_sem_post(ET_BARRIER_MINION);

                            عائم * c_base = (عائم *) (dst_ptr2 + n * nbd1);
                            إذا (m0 < M) {
                                atomic_store_f32((متطاير عائم *) (c_base + m0), s0 + p0);
                            }
                            إذا (m1 < M) {
                                atomic_store_f32((متطاير عائم *) (c_base + m1), s1 + p1);
                            }
                            إذا (m2 < M) {
                                atomic_store_f32((متطاير عائم *) (c_base + m2), s2 + p2);
                            }
                            إذا (m3 < M) {
                                atomic_store_f32((متطاير عائم *) (c_base + m3), s3 + p3);
                            }
                        }
                    }
                }
            }
        }
    } وإلا إذا (K_blocks > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        لكل (int64_t i3 = 0; i3 < ne13; i3++) {
            ثابت int64_t i03       = i3 / r3;
            ثابت حرف *  src0_ptr3 = (ثابت حرف *) params->src0.data + i03 * nb03;
            ثابت حرف *  src1_ptr3 = (ثابت حرف *) params->src1.data + i3 * nb13;
            حرف *        dst_ptr3  = (حرف *) params->dst.data + i3 * nbd3;

            لكل (int64_t i2 = 0; i2 < ne12; i2++) {
                ثابت int64_t i02       = i2 / r2;
                ثابت حرف *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ثابت حرف *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                حرف *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                لكل (int64_t n = 0; n < N; n++) {
                    ثابت عائم * b_col_base = (ثابت عائم *) (src1_ptr2 + n * nb11);

                    لكل (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        ثابت int64_t m1 = m0 + STRIDE_M;
                        ثابت int64_t m2 = m0 + STRIDE_M * 2;
                        ثابت int64_t m3 = m0 + STRIDE_M * 3;

                        عائم s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        لكل (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = K_blocks - kb;
                            إذا (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            ثابت عائم * b_tile = b_col_base + kb * 32;

                            s0 += compute_row_dot_q4_0((ثابت block_q4_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            إذا (m1 < M) {
                                s1 += compute_row_dot_q4_0((ثابت block_q4_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            إذا (m2 < M) {
                                s2 += compute_row_dot_q4_0((ثابت block_q4_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            إذا (m3 < M) {
                                s3 += compute_row_dot_q4_0((ثابت block_q4_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        عائم * dst_base = (عائم *) (dst_ptr2 + n * nbd1);
                        atomic_store_f32((متطاير عائم *) (dst_base + m0), s0);
                        إذا (m1 < M) {
                            atomic_store_f32((متطاير عائم *) (dst_base + m1), s1);
                        }
                        إذا (m2 < M) {
                            atomic_store_f32((متطاير عائم *) (dst_base + m2), s2);
                        }
                        إذا (m3 < M) {
                            atomic_store_f32((متطاير عائم *) (dst_base + m3), s3);
                        }
                    }
                }
            }
        }
    } وإلا {
        /*
         * Simple path for small K.
         *
         * When `nb01` is 32-byte aligned, every row has the same block-alignment
         * pattern. That lets us compute two rows together and reuse each loaded
         * B chunk across both rows instead of reloading it in a second dot call.
         */
        لكل (int64_t i3 = 0; i3 < ne13; i3++) {
            ثابت int64_t i03       = i3 / r3;
            ثابت حرف *  src0_ptr3 = (ثابت حرف *) params->src0.data + i03 * nb03;
            ثابت حرف *  src1_ptr3 = (ثابت حرف *) params->src1.data + i3 * nb13;
            حرف *        dst_ptr3  = (حرف *) params->dst.data + i3 * nbd3;

            لكل (int64_t i2 = 0; i2 < ne12; i2++) {
                ثابت int64_t i02       = i2 / r2;
                ثابت حرف *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ثابت حرف *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                حرف *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                لكل (int64_t n = 0; n < N; n++) {
                    ثابت عائم * b_col_base = (ثابت عائم *) (src1_ptr2 + n * nb11);
                    q4_dot_state  q4_state;
                    q4_dot_begin(&q4_state);

                    إذا (use_simple_x2) {
                        لكل (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_ROWS) {
                            ثابت int64_t      m1     = m0 + STRIDE_M;
                            ثابت block_q4_0 * q_row0 = (ثابت block_q4_0 *) (src0_ptr2 + m0 * nb01);

                            إذا (m1 < M) {
                                ثابت block_q4_0 * q_row1 = (ثابت block_q4_0 *) (src0_ptr2 + m1 * nb01);
                                عائم              s0, s1;
                                q4_dot_compute_x2_aligned(q_row0, q_row1, b_col_base, K_blocks, &s0, &s1);

                                عائم * dst0 = (عائم *) (dst_ptr2 + n * nbd1 + m0 * حجم(عائم));
                                عائم * dst1 = (عائم *) (dst_ptr2 + n * nbd1 + m1 * حجم(عائم));
                                atomic_store_f32((متطاير عائم *) dst0, s0);
                                atomic_store_f32((متطاير عائم *) dst1, s1);
                            } وإلا {
                                عائم   sum = q4_dot_compute(q_row0, b_col_base, K_blocks);
                                عائم * dst = (عائم *) (dst_ptr2 + n * nbd1 + m0 * حجم(عائم));
                                atomic_store_f32((متطاير عائم *) dst, sum);
                            }
                        }
                    } وإلا {
                        لكل (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            ثابت block_q4_0 * q_row = (ثابت block_q4_0 *) (src0_ptr2 + m * nb01);

                            عائم sum = q4_dot_compute(q_row, b_col_base, K_blocks);

                            عائم * dst_entry = (عائم *) (dst_ptr2 + n * nbd1 + m * حجم(عائم));
                            atomic_store_f32((متطاير عائم *) dst_entry, sum);
                        }
                    }

                    q4_dot_end(&q4_state);
                }
            }
        }
    }

    أرجع 0;
}
