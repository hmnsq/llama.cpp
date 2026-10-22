//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#അന്തര്ഭാവ "block_ops.h"
#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "math_fp.h"
#അന്തര്ഭാവ "platform.h"
#അന്തര്ഭാവ "quants.h"

#അന്തര്ഭാവ <stdint.h>

#define STRIDE_M                   2048 /* 32 shires x 32 minions x 2 harts */
#define STRIDE_M_KSPLIT            1024 /* 32 shires x 32 minions (both harts share rows) */
#define KSPLIT_MIN_K_BLOCKS        256  /* K >= 8192 elements */
#define KSPLIT_SMALL_ROWS_K_BLOCKS 64   /* K >= 2048 elements for very small M */
#define KSPLIT_MAX_ROWS            8    /* max rows per minion for K-split */
#define TILE_KB                    256  /* K-tile size in Q4_0 blocks (8192 elems, 32KB B data) */
#define KSPLIT_GROUP_ROWS          4
#define SIMPLE_X2_ROWS             2

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_ദ്വിമാന_params * params, ശൂന്യം * env) {
    uint64_t hart_id = get_hart_id();

    // Matrix dimensions
    സ്ഥിരം int64_t K    = params->src0.ne[0];
    സ്ഥിരം int64_t M    = params->src0.ne[1];
    സ്ഥിരം int64_t N    = params->src1.ne[1];
    സ്ഥിരം int64_t ne02 = params->src0.ne[2];
    സ്ഥിരം int64_t ne03 = params->src0.ne[3];
    സ്ഥിരം int64_t ne12 = params->src1.ne[2];
    സ്ഥിരം int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    സ്ഥിരം size_t nb01 = params->src0.nb[1];
    സ്ഥിരം size_t nb02 = params->src0.nb[2];
    സ്ഥിരം size_t nb03 = params->src0.nb[3];

    സ്ഥിരം size_t nb11 = params->src1.nb[1];
    സ്ഥിരം size_t nb12 = params->src1.nb[2];
    സ്ഥിരം size_t nb13 = params->src1.nb[3];

    സ്ഥിരം size_t nbd1 = params->dst.nb[1];
    സ്ഥിരം size_t nbd2 = params->dst.nb[2];
    സ്ഥിരം size_t nbd3 = params->dst.nb[3];

    // Q4_0 block size is 32
    സ്ഥിരം int64_t K_blocks      = K / 32;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    സ്ഥിരം int64_t r2 = ne12 / ne02;
    സ്ഥിരം int64_t r3 = ne13 / ne03;

    // K-split decision
    സ്ഥിരം int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    സ്ഥിരം int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    സ്ഥിരം പൂർണ്ണസംഖ്യ     is_hart1              = hart_id & 1;
    സ്ഥിരം int64_t rows_per_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    സ്ഥിരം int64_t k_half                = K_blocks / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     use_ksplit_small_rows = (rows_per_minion <= 2) && (K_blocks >= KSPLIT_SMALL_ROWS_K_BLOCKS);
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
    സ്ഥിരം പൂർണ്ണസംഖ്യ     use_ksplit            = ((K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion <= KSPLIT_MAX_ROWS) &&
                                           (rows_per_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_rows;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     use_ksplit_group      = !use_ksplit && (K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion > 4) &&
                                          (rows_per_minion <= KSPLIT_MAX_ROWS);

    എങ്കിൽ (use_ksplit) {
        /* Each hart processes half the K dimension */
        സ്ഥിരം int64_t k_start = is_hart1 ? k_half : 0;
        സ്ഥിരം int64_t k_len   = is_hart1 ? (K_blocks - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        അസ്ഥിര ദശാംശം * l2scp_slot = (അസ്ഥിര ദശാംശം *) et_shire_l2scp_local(local_minion * 64);

        വേണ്ടി (int64_t i3 = 0; i3 < ne13; i3++) {
            സ്ഥിരം int64_t i03       = i3 / r3;
            സ്ഥിരം അക്ഷരം *  src0_ptr3 = (സ്ഥിരം അക്ഷരം *) params->src0.data + i03 * nb03;
            സ്ഥിരം അക്ഷരം *  src1_ptr3 = (സ്ഥിരം അക്ഷരം *) params->src1.data + i3 * nb13;
            അക്ഷരം *        dst_ptr3  = (അക്ഷരം *) params->dst.data + i3 * nbd3;

            വേണ്ടി (int64_t i2 = 0; i2 < ne12; i2++) {
                സ്ഥിരം int64_t i02       = i2 / r2;
                സ്ഥിരം അക്ഷരം *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                സ്ഥിരം അക്ഷരം *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                അക്ഷരം *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                വേണ്ടി (int64_t n = 0; n < N; n++) {
                    സ്ഥിരം ദശാംശം * b_col_base = (സ്ഥിരം ദശാംശം *) (src1_ptr2 + n * nb11);

                    വേണ്ടി (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        സ്ഥിരം block_q4_0 * q_row = (സ്ഥിരം block_q4_0 *) (src0_ptr2 + m * nb01);

                        ദശാംശം partial = compute_row_dot_q4_0(q_row + k_start, b_col_base + k_start * 32, k_len);

                        എങ്കിൽ (is_hart1) {
                            *l2scp_slot = partial;
                            FENCE;
                            flush_to_l2((സ്ഥിരം ശൂന്യം *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } അല്ലെങ്കിൽ {
                            et_sem_wait(ET_BARRIER_MINION);
                            ദശാംശം other = *l2scp_slot;
                            et_sem_post(ET_BARRIER_MINION);

                            ദശാംശം * dst_entry = (ദശാംശം *) (dst_ptr2 + n * nbd1 + m * വലുപ്പം(ദശാംശം));
                            atomic_store_f32((അസ്ഥിര ദശാംശം *) dst_entry, partial + other);
                        }
                    }
                }
            }
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (use_ksplit_group) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime.
         *
         * Both harts process the same 4-row group, each on half of K, and
         * exchange 4 partial sums once per group instead of once per row.
         * This keeps the K-split bandwidth benefit while cutting semaphore
         * traffic by 4x relative to the old per-row exchange.
         */
        സ്ഥിരം int64_t    k_start    = is_hart1 ? k_half : 0;
        സ്ഥിരം int64_t    k_len      = is_hart1 ? (K_blocks - k_half) : k_half;
        അസ്ഥിര ദശാംശം * l2scp_slot = (അസ്ഥിര ദശാംശം *) et_shire_l2scp_local(local_minion * 64);

        വേണ്ടി (int64_t i3 = 0; i3 < ne13; i3++) {
            സ്ഥിരം int64_t i03       = i3 / r3;
            സ്ഥിരം അക്ഷരം *  src0_ptr3 = (സ്ഥിരം അക്ഷരം *) params->src0.data + i03 * nb03;
            സ്ഥിരം അക്ഷരം *  src1_ptr3 = (സ്ഥിരം അക്ഷരം *) params->src1.data + i3 * nb13;
            അക്ഷരം *        dst_ptr3  = (അക്ഷരം *) params->dst.data + i3 * nbd3;

            വേണ്ടി (int64_t i2 = 0; i2 < ne12; i2++) {
                സ്ഥിരം int64_t i02       = i2 / r2;
                സ്ഥിരം അക്ഷരം *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                സ്ഥിരം അക്ഷരം *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                അക്ഷരം *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                വേണ്ടി (int64_t n = 0; n < N; n++) {
                    സ്ഥിരം ദശാംശം * b_col_base = (സ്ഥിരം ദശാംശം *) (src1_ptr2 + n * nb11);

                    വേണ്ടി (int64_t m_base = minion_id; m_base < M; m_base += STRIDE_M_KSPLIT * KSPLIT_GROUP_ROWS) {
                        സ്ഥിരം int64_t m0 = m_base;
                        സ്ഥിരം int64_t m1 = m0 + STRIDE_M_KSPLIT;
                        സ്ഥിരം int64_t m2 = m1 + STRIDE_M_KSPLIT;
                        സ്ഥിരം int64_t m3 = m2 + STRIDE_M_KSPLIT;

                        ദശാംശം s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        വേണ്ടി (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = k_len - kb;
                            എങ്കിൽ (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            എങ്കിൽ (tile_len <= 0) {
                                നിർത്തുക;
                            }
                            സ്ഥിരം ദശാംശം * b_tile = b_col_base + (k_start + kb) * 32;
                            സ്ഥിരം int64_t row_kb = k_start + kb;

                            എങ്കിൽ (m0 < M) {
                                s0 += compute_row_dot_q4_0((സ്ഥിരം block_q4_0 *) (src0_ptr2 + m0 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            എങ്കിൽ (m1 < M) {
                                s1 += compute_row_dot_q4_0((സ്ഥിരം block_q4_0 *) (src0_ptr2 + m1 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            എങ്കിൽ (m2 < M) {
                                s2 += compute_row_dot_q4_0((സ്ഥിരം block_q4_0 *) (src0_ptr2 + m2 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            എങ്കിൽ (m3 < M) {
                                s3 += compute_row_dot_q4_0((സ്ഥിരം block_q4_0 *) (src0_ptr2 + m3 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                        }

                        എങ്കിൽ (is_hart1) {
                            l2scp_slot[0] = s0;
                            l2scp_slot[1] = s1;
                            l2scp_slot[2] = s2;
                            l2scp_slot[3] = s3;
                            FENCE;
                            flush_to_l2((സ്ഥിരം ശൂന്യം *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } അല്ലെങ്കിൽ {
                            et_sem_wait(ET_BARRIER_MINION);
                            സ്ഥിരം ദശാംശം p0 = l2scp_slot[0];
                            സ്ഥിരം ദശാംശം p1 = l2scp_slot[1];
                            സ്ഥിരം ദശാംശം p2 = l2scp_slot[2];
                            സ്ഥിരം ദശാംശം p3 = l2scp_slot[3];
                            et_sem_post(ET_BARRIER_MINION);

                            ദശാംശം * c_base = (ദശാംശം *) (dst_ptr2 + n * nbd1);
                            എങ്കിൽ (m0 < M) {
                                atomic_store_f32((അസ്ഥിര ദശാംശം *) (c_base + m0), s0 + p0);
                            }
                            എങ്കിൽ (m1 < M) {
                                atomic_store_f32((അസ്ഥിര ദശാംശം *) (c_base + m1), s1 + p1);
                            }
                            എങ്കിൽ (m2 < M) {
                                atomic_store_f32((അസ്ഥിര ദശാംശം *) (c_base + m2), s2 + p2);
                            }
                            എങ്കിൽ (m3 < M) {
                                atomic_store_f32((അസ്ഥിര ദശാംശം *) (c_base + m3), s3 + p3);
                            }
                        }
                    }
                }
            }
        }
    } അല്ലെങ്കിൽ എങ്കിൽ (K_blocks > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        വേണ്ടി (int64_t i3 = 0; i3 < ne13; i3++) {
            സ്ഥിരം int64_t i03       = i3 / r3;
            സ്ഥിരം അക്ഷരം *  src0_ptr3 = (സ്ഥിരം അക്ഷരം *) params->src0.data + i03 * nb03;
            സ്ഥിരം അക്ഷരം *  src1_ptr3 = (സ്ഥിരം അക്ഷരം *) params->src1.data + i3 * nb13;
            അക്ഷരം *        dst_ptr3  = (അക്ഷരം *) params->dst.data + i3 * nbd3;

            വേണ്ടി (int64_t i2 = 0; i2 < ne12; i2++) {
                സ്ഥിരം int64_t i02       = i2 / r2;
                സ്ഥിരം അക്ഷരം *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                സ്ഥിരം അക്ഷരം *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                അക്ഷരം *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                വേണ്ടി (int64_t n = 0; n < N; n++) {
                    സ്ഥിരം ദശാംശം * b_col_base = (സ്ഥിരം ദശാംശം *) (src1_ptr2 + n * nb11);

                    വേണ്ടി (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        സ്ഥിരം int64_t m1 = m0 + STRIDE_M;
                        സ്ഥിരം int64_t m2 = m0 + STRIDE_M * 2;
                        സ്ഥിരം int64_t m3 = m0 + STRIDE_M * 3;

                        ദശാംശം s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        വേണ്ടി (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = K_blocks - kb;
                            എങ്കിൽ (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            സ്ഥിരം ദശാംശം * b_tile = b_col_base + kb * 32;

                            s0 += compute_row_dot_q4_0((സ്ഥിരം block_q4_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            എങ്കിൽ (m1 < M) {
                                s1 += compute_row_dot_q4_0((സ്ഥിരം block_q4_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            എങ്കിൽ (m2 < M) {
                                s2 += compute_row_dot_q4_0((സ്ഥിരം block_q4_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            എങ്കിൽ (m3 < M) {
                                s3 += compute_row_dot_q4_0((സ്ഥിരം block_q4_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        ദശാംശം * dst_base = (ദശാംശം *) (dst_ptr2 + n * nbd1);
                        atomic_store_f32((അസ്ഥിര ദശാംശം *) (dst_base + m0), s0);
                        എങ്കിൽ (m1 < M) {
                            atomic_store_f32((അസ്ഥിര ദശാംശം *) (dst_base + m1), s1);
                        }
                        എങ്കിൽ (m2 < M) {
                            atomic_store_f32((അസ്ഥിര ദശാംശം *) (dst_base + m2), s2);
                        }
                        എങ്കിൽ (m3 < M) {
                            atomic_store_f32((അസ്ഥിര ദശാംശം *) (dst_base + m3), s3);
                        }
                    }
                }
            }
        }
    } അല്ലെങ്കിൽ {
        /*
         * Simple path for small K.
         *
         * When `nb01` is 32-byte aligned, every row has the same block-alignment
         * pattern. That lets us compute two rows together and reuse each loaded
         * B chunk across both rows instead of reloading it in a second dot call.
         */
        വേണ്ടി (int64_t i3 = 0; i3 < ne13; i3++) {
            സ്ഥിരം int64_t i03       = i3 / r3;
            സ്ഥിരം അക്ഷരം *  src0_ptr3 = (സ്ഥിരം അക്ഷരം *) params->src0.data + i03 * nb03;
            സ്ഥിരം അക്ഷരം *  src1_ptr3 = (സ്ഥിരം അക്ഷരം *) params->src1.data + i3 * nb13;
            അക്ഷരം *        dst_ptr3  = (അക്ഷരം *) params->dst.data + i3 * nbd3;

            വേണ്ടി (int64_t i2 = 0; i2 < ne12; i2++) {
                സ്ഥിരം int64_t i02       = i2 / r2;
                സ്ഥിരം അക്ഷരം *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                സ്ഥിരം അക്ഷരം *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                അക്ഷരം *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                വേണ്ടി (int64_t n = 0; n < N; n++) {
                    സ്ഥിരം ദശാംശം * b_col_base = (സ്ഥിരം ദശാംശം *) (src1_ptr2 + n * nb11);
                    q4_dot_state  q4_state;
                    q4_dot_begin(&q4_state);

                    എങ്കിൽ (use_simple_x2) {
                        വേണ്ടി (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_ROWS) {
                            സ്ഥിരം int64_t      m1     = m0 + STRIDE_M;
                            സ്ഥിരം block_q4_0 * q_row0 = (സ്ഥിരം block_q4_0 *) (src0_ptr2 + m0 * nb01);

                            എങ്കിൽ (m1 < M) {
                                സ്ഥിരം block_q4_0 * q_row1 = (സ്ഥിരം block_q4_0 *) (src0_ptr2 + m1 * nb01);
                                ദശാംശം              s0, s1;
                                q4_dot_compute_x2_aligned(q_row0, q_row1, b_col_base, K_blocks, &s0, &s1);

                                ദശാംശം * dst0 = (ദശാംശം *) (dst_ptr2 + n * nbd1 + m0 * വലുപ്പം(ദശാംശം));
                                ദശാംശം * dst1 = (ദശാംശം *) (dst_ptr2 + n * nbd1 + m1 * വലുപ്പം(ദശാംശം));
                                atomic_store_f32((അസ്ഥിര ദശാംശം *) dst0, s0);
                                atomic_store_f32((അസ്ഥിര ദശാംശം *) dst1, s1);
                            } അല്ലെങ്കിൽ {
                                ദശാംശം   sum = q4_dot_compute(q_row0, b_col_base, K_blocks);
                                ദശാംശം * dst = (ദശാംശം *) (dst_ptr2 + n * nbd1 + m0 * വലുപ്പം(ദശാംശം));
                                atomic_store_f32((അസ്ഥിര ദശാംശം *) dst, sum);
                            }
                        }
                    } അല്ലെങ്കിൽ {
                        വേണ്ടി (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            സ്ഥിരം block_q4_0 * q_row = (സ്ഥിരം block_q4_0 *) (src0_ptr2 + m * nb01);

                            ദശാംശം sum = q4_dot_compute(q_row, b_col_base, K_blocks);

                            ദശാംശം * dst_entry = (ദശാംശം *) (dst_ptr2 + n * nbd1 + m * വലുപ്പം(ദശാംശം));
                            atomic_store_f32((അസ്ഥിര ദശാംശം *) dst_entry, sum);
                        }
                    }

                    q4_dot_end(&q4_state);
                }
            }
        }
    }

    തിരികെ 0;
}
