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
#define TILE_KB                    256  /* K-tile size in Q4_0 blocks (8192 elems, 32KB B data) */
#define KSPLIT_GROUP_ROWS          4
#define SIMPLE_X2_ROWS             2

पूर्णांक entry_point(संरचना ggml_et_द्विमान_params * params, शून्य * env) {
    uint64_t hart_id = get_hart_id();

    // Matrix dimensions
    स्थिरांक int64_t K    = params->src0.ne[0];
    स्थिरांक int64_t M    = params->src0.ne[1];
    स्थिरांक int64_t N    = params->src1.ne[1];
    स्थिरांक int64_t ne02 = params->src0.ne[2];
    स्थिरांक int64_t ne03 = params->src0.ne[3];
    स्थिरांक int64_t ne12 = params->src1.ne[2];
    स्थिरांक int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    स्थिरांक माप_प्रकार nb01 = params->src0.nb[1];
    स्थिरांक माप_प्रकार nb02 = params->src0.nb[2];
    स्थिरांक माप_प्रकार nb03 = params->src0.nb[3];

    स्थिरांक माप_प्रकार nb11 = params->src1.nb[1];
    स्थिरांक माप_प्रकार nb12 = params->src1.nb[2];
    स्थिरांक माप_प्रकार nb13 = params->src1.nb[3];

    स्थिरांक माप_प्रकार nbd1 = params->dst.nb[1];
    स्थिरांक माप_प्रकार nbd2 = params->dst.nb[2];
    स्थिरांक माप_प्रकार nbd3 = params->dst.nb[3];

    // Q4_0 block size is 32
    स्थिरांक int64_t K_blocks      = K / 32;
    स्थिरांक पूर्णांक     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    स्थिरांक int64_t r2 = ne12 / ne02;
    स्थिरांक int64_t r3 = ne13 / ne03;

    // K-split decision
    स्थिरांक int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    स्थिरांक int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    स्थिरांक पूर्णांक     is_hart1              = hart_id & 1;
    स्थिरांक int64_t rows_per_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    स्थिरांक int64_t k_half                = K_blocks / 2;
    स्थिरांक पूर्णांक     use_ksplit_small_rows = (rows_per_minion <= 2) && (K_blocks >= KSPLIT_SMALL_ROWS_K_BLOCKS);
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
    स्थिरांक पूर्णांक     use_ksplit            = ((K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion <= KSPLIT_MAX_ROWS) &&
                                           (rows_per_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_rows;
    स्थिरांक पूर्णांक     use_ksplit_group      = !use_ksplit && (K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion > 4) &&
                                          (rows_per_minion <= KSPLIT_MAX_ROWS);

    जर (use_ksplit) {
        /* Each hart processes half the K dimension */
        स्थिरांक int64_t k_start = is_hart1 ? k_half : 0;
        स्थिरांक int64_t k_len   = is_hart1 ? (K_blocks - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        अस्थिर दशांश * l2scp_slot = (अस्थिर दशांश *) et_shire_l2scp_local(local_minion * 64);

        साठी (int64_t i3 = 0; i3 < ne13; i3++) {
            स्थिरांक int64_t i03       = i3 / r3;
            स्थिरांक अक्षर *  src0_ptr3 = (स्थिरांक अक्षर *) params->src0.data + i03 * nb03;
            स्थिरांक अक्षर *  src1_ptr3 = (स्थिरांक अक्षर *) params->src1.data + i3 * nb13;
            अक्षर *        dst_ptr3  = (अक्षर *) params->dst.data + i3 * nbd3;

            साठी (int64_t i2 = 0; i2 < ne12; i2++) {
                स्थिरांक int64_t i02       = i2 / r2;
                स्थिरांक अक्षर *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                स्थिरांक अक्षर *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                अक्षर *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                साठी (int64_t n = 0; n < N; n++) {
                    स्थिरांक दशांश * b_col_base = (स्थिरांक दशांश *) (src1_ptr2 + n * nb11);

                    साठी (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        स्थिरांक block_q4_0 * q_row = (स्थिरांक block_q4_0 *) (src0_ptr2 + m * nb01);

                        दशांश partial = compute_row_dot_q4_0(q_row + k_start, b_col_base + k_start * 32, k_len);

                        जर (is_hart1) {
                            *l2scp_slot = partial;
                            FENCE;
                            flush_to_l2((स्थिरांक शून्य *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } अन्यथा {
                            et_sem_wait(ET_BARRIER_MINION);
                            दशांश other = *l2scp_slot;
                            et_sem_post(ET_BARRIER_MINION);

                            दशांश * dst_entry = (दशांश *) (dst_ptr2 + n * nbd1 + m * आकार(दशांश));
                            atomic_store_f32((अस्थिर दशांश *) dst_entry, partial + other);
                        }
                    }
                }
            }
        }
    } अन्यथा जर (use_ksplit_group) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime.
         *
         * Both harts process the same 4-row group, each on half of K, and
         * exchange 4 partial sums once per group instead of once per row.
         * This keeps the K-split bandwidth benefit while cutting semaphore
         * traffic by 4x relative to the old per-row exchange.
         */
        स्थिरांक int64_t    k_start    = is_hart1 ? k_half : 0;
        स्थिरांक int64_t    k_len      = is_hart1 ? (K_blocks - k_half) : k_half;
        अस्थिर दशांश * l2scp_slot = (अस्थिर दशांश *) et_shire_l2scp_local(local_minion * 64);

        साठी (int64_t i3 = 0; i3 < ne13; i3++) {
            स्थिरांक int64_t i03       = i3 / r3;
            स्थिरांक अक्षर *  src0_ptr3 = (स्थिरांक अक्षर *) params->src0.data + i03 * nb03;
            स्थिरांक अक्षर *  src1_ptr3 = (स्थिरांक अक्षर *) params->src1.data + i3 * nb13;
            अक्षर *        dst_ptr3  = (अक्षर *) params->dst.data + i3 * nbd3;

            साठी (int64_t i2 = 0; i2 < ne12; i2++) {
                स्थिरांक int64_t i02       = i2 / r2;
                स्थिरांक अक्षर *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                स्थिरांक अक्षर *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                अक्षर *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                साठी (int64_t n = 0; n < N; n++) {
                    स्थिरांक दशांश * b_col_base = (स्थिरांक दशांश *) (src1_ptr2 + n * nb11);

                    साठी (int64_t m_base = minion_id; m_base < M; m_base += STRIDE_M_KSPLIT * KSPLIT_GROUP_ROWS) {
                        स्थिरांक int64_t m0 = m_base;
                        स्थिरांक int64_t m1 = m0 + STRIDE_M_KSPLIT;
                        स्थिरांक int64_t m2 = m1 + STRIDE_M_KSPLIT;
                        स्थिरांक int64_t m3 = m2 + STRIDE_M_KSPLIT;

                        दशांश s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        साठी (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = k_len - kb;
                            जर (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            जर (tile_len <= 0) {
                                थांबा;
                            }
                            स्थिरांक दशांश * b_tile = b_col_base + (k_start + kb) * 32;
                            स्थिरांक int64_t row_kb = k_start + kb;

                            जर (m0 < M) {
                                s0 += compute_row_dot_q4_0((स्थिरांक block_q4_0 *) (src0_ptr2 + m0 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            जर (m1 < M) {
                                s1 += compute_row_dot_q4_0((स्थिरांक block_q4_0 *) (src0_ptr2 + m1 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            जर (m2 < M) {
                                s2 += compute_row_dot_q4_0((स्थिरांक block_q4_0 *) (src0_ptr2 + m2 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            जर (m3 < M) {
                                s3 += compute_row_dot_q4_0((स्थिरांक block_q4_0 *) (src0_ptr2 + m3 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                        }

                        जर (is_hart1) {
                            l2scp_slot[0] = s0;
                            l2scp_slot[1] = s1;
                            l2scp_slot[2] = s2;
                            l2scp_slot[3] = s3;
                            FENCE;
                            flush_to_l2((स्थिरांक शून्य *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } अन्यथा {
                            et_sem_wait(ET_BARRIER_MINION);
                            स्थिरांक दशांश p0 = l2scp_slot[0];
                            स्थिरांक दशांश p1 = l2scp_slot[1];
                            स्थिरांक दशांश p2 = l2scp_slot[2];
                            स्थिरांक दशांश p3 = l2scp_slot[3];
                            et_sem_post(ET_BARRIER_MINION);

                            दशांश * c_base = (दशांश *) (dst_ptr2 + n * nbd1);
                            जर (m0 < M) {
                                atomic_store_f32((अस्थिर दशांश *) (c_base + m0), s0 + p0);
                            }
                            जर (m1 < M) {
                                atomic_store_f32((अस्थिर दशांश *) (c_base + m1), s1 + p1);
                            }
                            जर (m2 < M) {
                                atomic_store_f32((अस्थिर दशांश *) (c_base + m2), s2 + p2);
                            }
                            जर (m3 < M) {
                                atomic_store_f32((अस्थिर दशांश *) (c_base + m3), s3 + p3);
                            }
                        }
                    }
                }
            }
        }
    } अन्यथा जर (K_blocks > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        साठी (int64_t i3 = 0; i3 < ne13; i3++) {
            स्थिरांक int64_t i03       = i3 / r3;
            स्थिरांक अक्षर *  src0_ptr3 = (स्थिरांक अक्षर *) params->src0.data + i03 * nb03;
            स्थिरांक अक्षर *  src1_ptr3 = (स्थिरांक अक्षर *) params->src1.data + i3 * nb13;
            अक्षर *        dst_ptr3  = (अक्षर *) params->dst.data + i3 * nbd3;

            साठी (int64_t i2 = 0; i2 < ne12; i2++) {
                स्थिरांक int64_t i02       = i2 / r2;
                स्थिरांक अक्षर *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                स्थिरांक अक्षर *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                अक्षर *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                साठी (int64_t n = 0; n < N; n++) {
                    स्थिरांक दशांश * b_col_base = (स्थिरांक दशांश *) (src1_ptr2 + n * nb11);

                    साठी (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        स्थिरांक int64_t m1 = m0 + STRIDE_M;
                        स्थिरांक int64_t m2 = m0 + STRIDE_M * 2;
                        स्थिरांक int64_t m3 = m0 + STRIDE_M * 3;

                        दशांश s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        साठी (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = K_blocks - kb;
                            जर (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            स्थिरांक दशांश * b_tile = b_col_base + kb * 32;

                            s0 += compute_row_dot_q4_0((स्थिरांक block_q4_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            जर (m1 < M) {
                                s1 += compute_row_dot_q4_0((स्थिरांक block_q4_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            जर (m2 < M) {
                                s2 += compute_row_dot_q4_0((स्थिरांक block_q4_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            जर (m3 < M) {
                                s3 += compute_row_dot_q4_0((स्थिरांक block_q4_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        दशांश * dst_base = (दशांश *) (dst_ptr2 + n * nbd1);
                        atomic_store_f32((अस्थिर दशांश *) (dst_base + m0), s0);
                        जर (m1 < M) {
                            atomic_store_f32((अस्थिर दशांश *) (dst_base + m1), s1);
                        }
                        जर (m2 < M) {
                            atomic_store_f32((अस्थिर दशांश *) (dst_base + m2), s2);
                        }
                        जर (m3 < M) {
                            atomic_store_f32((अस्थिर दशांश *) (dst_base + m3), s3);
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
        साठी (int64_t i3 = 0; i3 < ne13; i3++) {
            स्थिरांक int64_t i03       = i3 / r3;
            स्थिरांक अक्षर *  src0_ptr3 = (स्थिरांक अक्षर *) params->src0.data + i03 * nb03;
            स्थिरांक अक्षर *  src1_ptr3 = (स्थिरांक अक्षर *) params->src1.data + i3 * nb13;
            अक्षर *        dst_ptr3  = (अक्षर *) params->dst.data + i3 * nbd3;

            साठी (int64_t i2 = 0; i2 < ne12; i2++) {
                स्थिरांक int64_t i02       = i2 / r2;
                स्थिरांक अक्षर *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                स्थिरांक अक्षर *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                अक्षर *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                साठी (int64_t n = 0; n < N; n++) {
                    स्थिरांक दशांश * b_col_base = (स्थिरांक दशांश *) (src1_ptr2 + n * nb11);
                    q4_dot_state  q4_state;
                    q4_dot_begin(&q4_state);

                    जर (use_simple_x2) {
                        साठी (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_ROWS) {
                            स्थिरांक int64_t      m1     = m0 + STRIDE_M;
                            स्थिरांक block_q4_0 * q_row0 = (स्थिरांक block_q4_0 *) (src0_ptr2 + m0 * nb01);

                            जर (m1 < M) {
                                स्थिरांक block_q4_0 * q_row1 = (स्थिरांक block_q4_0 *) (src0_ptr2 + m1 * nb01);
                                दशांश              s0, s1;
                                q4_dot_compute_x2_aligned(q_row0, q_row1, b_col_base, K_blocks, &s0, &s1);

                                दशांश * dst0 = (दशांश *) (dst_ptr2 + n * nbd1 + m0 * आकार(दशांश));
                                दशांश * dst1 = (दशांश *) (dst_ptr2 + n * nbd1 + m1 * आकार(दशांश));
                                atomic_store_f32((अस्थिर दशांश *) dst0, s0);
                                atomic_store_f32((अस्थिर दशांश *) dst1, s1);
                            } अन्यथा {
                                दशांश   sum = q4_dot_compute(q_row0, b_col_base, K_blocks);
                                दशांश * dst = (दशांश *) (dst_ptr2 + n * nbd1 + m0 * आकार(दशांश));
                                atomic_store_f32((अस्थिर दशांश *) dst, sum);
                            }
                        }
                    } अन्यथा {
                        साठी (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            स्थिरांक block_q4_0 * q_row = (स्थिरांक block_q4_0 *) (src0_ptr2 + m * nb01);

                            दशांश sum = q4_dot_compute(q_row, b_col_base, K_blocks);

                            दशांश * dst_entry = (दशांश *) (dst_ptr2 + n * nbd1 + m * आकार(दशांश));
                            atomic_store_f32((अस्थिर दशांश *) dst_entry, sum);
                        }
                    }

                    q4_dot_end(&q4_state);
                }
            }
        }
    }

    परत 0;
}
