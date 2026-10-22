//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#அந்தர்பாவ "block_ops.h"
#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "math_fp.h"
#அந்தர்பாவ "platform.h"
#அந்தர்பாவ "quants.h"

#அந்தர்பாவ <stdint.h>

#define STRIDE_M                   2048 /* 32 shires x 32 minions x 2 harts */
#define STRIDE_M_KSPLIT            1024 /* 32 shires x 32 minions (both harts share rows) */
#define KSPLIT_MIN_K_BLOCKS        256  /* K >= 8192 elements */
#define KSPLIT_SMALL_ROWS_K_BLOCKS 64   /* K >= 2048 elements for very small M */
#define KSPLIT_MAX_ROWS            8    /* max rows per minion for K-split */
#define TILE_KB                    256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */
#define KSPLIT_GROUP_ROWS          4
#define SIMPLE_X2_ROWS             2

நிலையான உள்வரி size_t tensor_bytes(மாறிலி கட்டமைப்பு ggml_tensor * t) {
    திரும்பு (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

முழு entry_point(கட்டமைப்பு ggml_et_mm_q8_params * params, வெறுமை * env) {
    uint64_t hart_id = get_hart_id();

    // Matrix dimensions
    மாறிலி int64_t K    = params->src0.ne[0];
    மாறிலி int64_t M    = params->src0.ne[1];
    மாறிலி int64_t N    = params->src1.ne[1];
    மாறிலி int64_t ne02 = params->src0.ne[2];
    மாறிலி int64_t ne03 = params->src0.ne[3];
    மாறிலி int64_t ne12 = params->src1.ne[2];
    மாறிலி int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    மாறிலி size_t nb01 = params->src0.nb[1];
    மாறிலி size_t nb02 = params->src0.nb[2];
    மாறிலி size_t nb03 = params->src0.nb[3];

    மாறிலி size_t nb11 = params->src1.nb[1];
    மாறிலி size_t nb12 = params->src1.nb[2];
    மாறிலி size_t nb13 = params->src1.nb[3];

    மாறிலி size_t nbd1 = params->dst.nb[1];
    மாறிலி size_t nbd2 = params->dst.nb[2];
    மாறிலி size_t nbd3 = params->dst.nb[3];

    // Optional residual bias
    மாறிலி எழுத்து * bias_base = (மாறிலி எழுத்து *) params->bias.data;
    மாறிலி size_t nbb1      = params->bias.nb[1];
    மாறிலி size_t nbb2      = params->bias.nb[2];
    மாறிலி size_t nbb3      = params->bias.nb[3];

    // Q8_0 block size is 32
    மாறிலி int64_t K_blocks      = K / 32;
    மாறிலி முழு     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    மாறிலி int64_t r2 = ne12 / ne02;
    மாறிலி int64_t r3 = ne13 / ne03;

    // K-split decision
    மாறிலி int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    மாறிலி int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    மாறிலி முழு     is_hart1              = hart_id & 1;
    மாறிலி int64_t rows_per_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    மாறிலி int64_t k_half                = K_blocks / 2;
    மாறிலி முழு     use_ksplit_small_rows = (rows_per_minion <= 2) && (K_blocks >= KSPLIT_SMALL_ROWS_K_BLOCKS);
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
    மாறிலி முழு     use_ksplit            = ((K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion <= KSPLIT_MAX_ROWS) &&
                                           (rows_per_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_rows;
    மாறிலி முழு     use_ksplit_group      = !use_ksplit && (K_blocks >= KSPLIT_MIN_K_BLOCKS) && (rows_per_minion > 4) &&
                                          (rows_per_minion <= KSPLIT_MAX_ROWS);

    evict_region_past_l2(params->src1.data, tensor_bytes(&params->src1));
    எனில் (params->bias.data) {
        evict_region_past_l2(params->bias.data, tensor_bytes(&params->bias));
    }

    எனில் (use_ksplit) {
        /* Each hart processes half the K dimension */
        மாறிலி int64_t k_start = is_hart1 ? k_half : 0;
        மாறிலி int64_t k_len   = is_hart1 ? (K_blocks - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        நிலையற்ற தசம * l2scp_slot = (நிலையற்ற தசம *) et_shire_l2scp_local(local_minion * 64);

        க்கு (int64_t i3 = 0; i3 < ne13; i3++) {
            மாறிலி int64_t i03       = i3 / r3;
            மாறிலி எழுத்து *  src0_ptr3 = (மாறிலி எழுத்து *) params->src0.data + i03 * nb03;
            மாறிலி எழுத்து *  src1_ptr3 = (மாறிலி எழுத்து *) params->src1.data + i3 * nb13;
            எழுத்து *        dst_ptr3  = (எழுத்து *) params->dst.data + i3 * nbd3;
            மாறிலி எழுத்து *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (மாறிலி எழுத்து *) 0;

            க்கு (int64_t i2 = 0; i2 < ne12; i2++) {
                மாறிலி int64_t i02       = i2 / r2;
                மாறிலி எழுத்து *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                மாறிலி எழுத்து *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                எழுத்து *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                மாறிலி எழுத்து *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (மாறிலி எழுத்து *) 0;

                க்கு (int64_t n = 0; n < N; n++) {
                    மாறிலி தசம * b_col_base = (மாறிலி தசம *) (src1_ptr2 + n * nb11);
                    மாறிலி தசம * bias_n     = bias_ptr2 ? (மாறிலி தசம *) (bias_ptr2 + n * nbb1) : (மாறிலி தசம *) 0;

                    க்கு (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        மாறிலி block_q8_0 * q_row = (மாறிலி block_q8_0 *) (src0_ptr2 + m * nb01);

                        தசம partial = compute_row_dot_q8_0(q_row + k_start, b_col_base + k_start * 32, k_len);

                        எனில் (is_hart1) {
                            *l2scp_slot = partial;
                            FENCE;
                            flush_to_l2((மாறிலி வெறுமை *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } இல்லையெனில் {
                            et_sem_wait(ET_BARRIER_MINION);
                            தசம other = *l2scp_slot;
                            et_sem_post(ET_BARRIER_MINION);

                            தசம * dst_entry = (தசம *) (dst_ptr2 + n * nbd1 + m * அளவு(தசம));
                            தசம   sum       = partial + other;
                            எனில் (bias_n) {
                                sum += bias_n[m];
                            }
                            atomic_store_f32((நிலையற்ற தசம *) dst_entry, sum);
                        }
                    }
                }
            }
        }
    } இல்லையெனில் எனில் (use_ksplit_group) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime.
         *
         * Both harts process the same 4-row group, each on half of K, and
         * exchange 4 partial sums once per group instead of once per row.
         * This keeps the K-split bandwidth benefit while cutting semaphore
         * traffic by 4x relative to the old per-row exchange.
         */
        மாறிலி int64_t    k_start    = is_hart1 ? k_half : 0;
        மாறிலி int64_t    k_len      = is_hart1 ? (K_blocks - k_half) : k_half;
        நிலையற்ற தசம * l2scp_slot = (நிலையற்ற தசம *) et_shire_l2scp_local(local_minion * 64);

        க்கு (int64_t i3 = 0; i3 < ne13; i3++) {
            மாறிலி int64_t i03       = i3 / r3;
            மாறிலி எழுத்து *  src0_ptr3 = (மாறிலி எழுத்து *) params->src0.data + i03 * nb03;
            மாறிலி எழுத்து *  src1_ptr3 = (மாறிலி எழுத்து *) params->src1.data + i3 * nb13;
            எழுத்து *        dst_ptr3  = (எழுத்து *) params->dst.data + i3 * nbd3;
            மாறிலி எழுத்து *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (மாறிலி எழுத்து *) 0;

            க்கு (int64_t i2 = 0; i2 < ne12; i2++) {
                மாறிலி int64_t i02       = i2 / r2;
                மாறிலி எழுத்து *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                மாறிலி எழுத்து *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                எழுத்து *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                மாறிலி எழுத்து *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (மாறிலி எழுத்து *) 0;

                க்கு (int64_t n = 0; n < N; n++) {
                    மாறிலி தசம * b_col_base = (மாறிலி தசம *) (src1_ptr2 + n * nb11);
                    மாறிலி தசம * bias_n     = bias_ptr2 ? (மாறிலி தசம *) (bias_ptr2 + n * nbb1) : (மாறிலி தசம *) 0;

                    க்கு (int64_t m_base = minion_id; m_base < M; m_base += STRIDE_M_KSPLIT * KSPLIT_GROUP_ROWS) {
                        மாறிலி int64_t m0 = m_base;
                        மாறிலி int64_t m1 = m0 + STRIDE_M_KSPLIT;
                        மாறிலி int64_t m2 = m1 + STRIDE_M_KSPLIT;
                        மாறிலி int64_t m3 = m2 + STRIDE_M_KSPLIT;

                        தசம s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        க்கு (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = k_len - kb;
                            எனில் (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            எனில் (tile_len <= 0) {
                                நிறுத்து;
                            }
                            மாறிலி தசம * b_tile = b_col_base + (k_start + kb) * 32;
                            மாறிலி int64_t row_kb = k_start + kb;

                            எனில் (m0 < M) {
                                s0 += compute_row_dot_q8_0((மாறிலி block_q8_0 *) (src0_ptr2 + m0 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            எனில் (m1 < M) {
                                s1 += compute_row_dot_q8_0((மாறிலி block_q8_0 *) (src0_ptr2 + m1 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            எனில் (m2 < M) {
                                s2 += compute_row_dot_q8_0((மாறிலி block_q8_0 *) (src0_ptr2 + m2 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                            எனில் (m3 < M) {
                                s3 += compute_row_dot_q8_0((மாறிலி block_q8_0 *) (src0_ptr2 + m3 * nb01) + row_kb,
                                                           b_tile, tile_len);
                            }
                        }

                        எனில் (is_hart1) {
                            l2scp_slot[0] = s0;
                            l2scp_slot[1] = s1;
                            l2scp_slot[2] = s2;
                            l2scp_slot[3] = s3;
                            FENCE;
                            flush_to_l2((மாறிலி வெறுமை *) l2scp_slot, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } இல்லையெனில் {
                            et_sem_wait(ET_BARRIER_MINION);
                            மாறிலி தசம p0 = l2scp_slot[0];
                            மாறிலி தசம p1 = l2scp_slot[1];
                            மாறிலி தசம p2 = l2scp_slot[2];
                            மாறிலி தசம p3 = l2scp_slot[3];
                            et_sem_post(ET_BARRIER_MINION);

                            தசம *     c_base = (தசம *) (dst_ptr2 + n * nbd1);
                            மாறிலி தசம b0     = bias_n ? bias_n[m0] : 0.0f;
                            மாறிலி தசம b1     = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                            மாறிலி தசம b2     = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                            மாறிலி தசம b3     = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                            எனில் (m0 < M) {
                                atomic_store_f32((நிலையற்ற தசம *) (c_base + m0), s0 + p0 + b0);
                            }
                            எனில் (m1 < M) {
                                atomic_store_f32((நிலையற்ற தசம *) (c_base + m1), s1 + p1 + b1);
                            }
                            எனில் (m2 < M) {
                                atomic_store_f32((நிலையற்ற தசம *) (c_base + m2), s2 + p2 + b2);
                            }
                            எனில் (m3 < M) {
                                atomic_store_f32((நிலையற்ற தசம *) (c_base + m3), s3 + p3 + b3);
                            }
                        }
                    }
                }
            }
        }
    } இல்லையெனில் எனில் (K_blocks > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        க்கு (int64_t i3 = 0; i3 < ne13; i3++) {
            மாறிலி int64_t i03       = i3 / r3;
            மாறிலி எழுத்து *  src0_ptr3 = (மாறிலி எழுத்து *) params->src0.data + i03 * nb03;
            மாறிலி எழுத்து *  src1_ptr3 = (மாறிலி எழுத்து *) params->src1.data + i3 * nb13;
            எழுத்து *        dst_ptr3  = (எழுத்து *) params->dst.data + i3 * nbd3;
            மாறிலி எழுத்து *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (மாறிலி எழுத்து *) 0;

            க்கு (int64_t i2 = 0; i2 < ne12; i2++) {
                மாறிலி int64_t i02       = i2 / r2;
                மாறிலி எழுத்து *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                மாறிலி எழுத்து *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                எழுத்து *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                மாறிலி எழுத்து *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (மாறிலி எழுத்து *) 0;

                க்கு (int64_t n = 0; n < N; n++) {
                    மாறிலி தசம * b_col_base = (மாறிலி தசம *) (src1_ptr2 + n * nb11);
                    மாறிலி தசம * bias_n     = bias_ptr2 ? (மாறிலி தசம *) (bias_ptr2 + n * nbb1) : (மாறிலி தசம *) 0;

                    க்கு (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        மாறிலி int64_t m1 = m0 + STRIDE_M;
                        மாறிலி int64_t m2 = m0 + STRIDE_M * 2;
                        மாறிலி int64_t m3 = m0 + STRIDE_M * 3;

                        தசம s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        க்கு (int64_t kb = 0; kb < K_blocks; kb += TILE_KB) {
                            int64_t tile_len = K_blocks - kb;
                            எனில் (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            மாறிலி தசம * b_tile = b_col_base + kb * 32;

                            s0 += compute_row_dot_q8_0((மாறிலி block_q8_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            எனில் (m1 < M) {
                                s1 += compute_row_dot_q8_0((மாறிலி block_q8_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            எனில் (m2 < M) {
                                s2 += compute_row_dot_q8_0((மாறிலி block_q8_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            எனில் (m3 < M) {
                                s3 += compute_row_dot_q8_0((மாறிலி block_q8_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        தசம *     dst_base = (தசம *) (dst_ptr2 + n * nbd1);
                        மாறிலி தசம b0       = bias_n ? bias_n[m0] : 0.0f;
                        மாறிலி தசம b1       = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                        மாறிலி தசம b2       = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                        மாறிலி தசம b3       = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                        atomic_store_f32((நிலையற்ற தசம *) (dst_base + m0), s0 + b0);
                        எனில் (m1 < M) {
                            atomic_store_f32((நிலையற்ற தசம *) (dst_base + m1), s1 + b1);
                        }
                        எனில் (m2 < M) {
                            atomic_store_f32((நிலையற்ற தசம *) (dst_base + m2), s2 + b2);
                        }
                        எனில் (m3 < M) {
                            atomic_store_f32((நிலையற்ற தசம *) (dst_base + m3), s3 + b3);
                        }
                    }
                }
            }
        }
    } இல்லையெனில் {
        /*
         * Simple path for small K.
         *
         * When `nb01` is 32-byte aligned, every row has the same block-alignment
         * pattern. That lets us compute two rows together and reuse each loaded
         * B chunk across both rows instead of reloading it in a second dot call.
         */
        க்கு (int64_t i3 = 0; i3 < ne13; i3++) {
            மாறிலி int64_t i03       = i3 / r3;
            மாறிலி எழுத்து *  src0_ptr3 = (மாறிலி எழுத்து *) params->src0.data + i03 * nb03;
            மாறிலி எழுத்து *  src1_ptr3 = (மாறிலி எழுத்து *) params->src1.data + i3 * nb13;
            எழுத்து *        dst_ptr3  = (எழுத்து *) params->dst.data + i3 * nbd3;
            மாறிலி எழுத்து *  bias_ptr3 = bias_base ? bias_base + i3 * nbb3 : (மாறிலி எழுத்து *) 0;

            க்கு (int64_t i2 = 0; i2 < ne12; i2++) {
                மாறிலி int64_t i02       = i2 / r2;
                மாறிலி எழுத்து *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                மாறிலி எழுத்து *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                எழுத்து *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                மாறிலி எழுத்து *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (மாறிலி எழுத்து *) 0;

                க்கு (int64_t n = 0; n < N; n++) {
                    மாறிலி தசம * b_col_base = (மாறிலி தசம *) (src1_ptr2 + n * nb11);
                    மாறிலி தசம * bias_n     = bias_ptr2 ? (மாறிலி தசம *) (bias_ptr2 + n * nbb1) : (மாறிலி தசம *) 0;
                    q8_dot_state  q8_state;
                    q8_dot_begin(&q8_state);

                    எனில் (use_simple_x2) {
                        க்கு (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_ROWS) {
                            மாறிலி int64_t      m1     = m0 + STRIDE_M;
                            மாறிலி block_q8_0 * q_row0 = (மாறிலி block_q8_0 *) (src0_ptr2 + m0 * nb01);

                            எனில் (m1 < M) {
                                மாறிலி block_q8_0 * q_row1 = (மாறிலி block_q8_0 *) (src0_ptr2 + m1 * nb01);
                                தசம              s0, s1;
                                q8_dot_compute_x2_aligned(q_row0, q_row1, b_col_base, K_blocks, &s0, &s1);

                                தசம * dst0 = (தசம *) (dst_ptr2 + n * nbd1 + m0 * அளவு(தசம));
                                தசம * dst1 = (தசம *) (dst_ptr2 + n * nbd1 + m1 * அளவு(தசம));
                                எனில் (bias_n) {
                                    s0 += bias_n[m0];
                                    s1 += bias_n[m1];
                                }
                                atomic_store_f32((நிலையற்ற தசம *) dst0, s0);
                                atomic_store_f32((நிலையற்ற தசம *) dst1, s1);
                            } இல்லையெனில் {
                                தசம   sum = q8_dot_compute(q_row0, b_col_base, K_blocks);
                                தசம * dst = (தசம *) (dst_ptr2 + n * nbd1 + m0 * அளவு(தசம));
                                எனில் (bias_n) {
                                    sum += bias_n[m0];
                                }
                                atomic_store_f32((நிலையற்ற தசம *) dst, sum);
                            }
                        }
                    } இல்லையெனில் {
                        க்கு (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            மாறிலி block_q8_0 * q_row = (மாறிலி block_q8_0 *) (src0_ptr2 + m * nb01);

                            தசம sum = q8_dot_compute(q_row, b_col_base, K_blocks);

                            தசம * dst_entry = (தசம *) (dst_ptr2 + n * nbd1 + m * அளவு(தசம));
                            எனில் (bias_n) {
                                sum += bias_n[m];
                            }
                            atomic_store_f32((நிலையற்ற தசம *) dst_entry, sum);
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
    திரும்பு 0;
}
