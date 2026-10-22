//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#شامل "block_ops.h"
#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"
#شامل "quants.h"

#شامل <stdint.h>

#تعریف_کرو STRIDE_M                   2048 /* 32 shires x 32 minions x 2 harts */
#تعریف_کرو STRIDE_M_KSPLIT            1024 /* 32 shires x 32 minions (both harts share rows) */
#تعریف_کرو KSPLIT_MIN_K_بلاکیں        256  /* K >= 8192 elements */
#تعریف_کرو KSPLIT_SMALL_صفیں_K_بلاکیں 64   /* K >= 2048 elements for very small M */
#تعریف_کرو KSPLIT_MAX_صفیں            8    /* max rows per minion for K-split */
#تعریف_کرو TILE_KB                    256  /* K-tile size in Q4_0 blocks (8192 elems, 32KB B data) */
#تعریف_کرو KSPLIT_گروہ_صفیں          4
#تعریف_کرو SIMPLE_X2_صفیں             2

صحیح اندراج_نقطہ(ساخت ggml_et_ثنائی_params * params, خالی * env) {
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
    ثابت int64_t K_بلاکیں      = K / 32;
    ثابت صحیح     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    ثابت int64_t r2 = ne12 / ne02;
    ثابت int64_t r3 = ne13 / ne03;

    // K-split decision
    ثابت int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    ثابت int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    ثابت صحیح     is_hart1              = hart_id & 1;
    ثابت int64_t صفیں_فی_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    ثابت int64_t k_half                = K_بلاکیں / 2;
    ثابت صحیح     use_ksplit_small_صفیں = (صفیں_فی_minion <= 2) && (K_بلاکیں >= KSPLIT_SMALL_صفیں_K_بلاکیں);
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
    ثابت صحیح     use_ksplit            = ((K_بلاکیں >= KSPLIT_MIN_K_بلاکیں) && (صفیں_فی_minion <= KSPLIT_MAX_صفیں) &&
                                           (صفیں_فی_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_صفیں;
    ثابت صحیح     use_ksplit_گروہ      = !use_ksplit && (K_بلاکیں >= KSPLIT_MIN_K_بلاکیں) && (صفیں_فی_minion > 4) &&
                                          (صفیں_فی_minion <= KSPLIT_MAX_صفیں);

    اگر (use_ksplit) {
        /* Each hart processes half the K dimension */
        ثابت int64_t k_آغاز = is_hart1 ? k_half : 0;
        ثابت int64_t k_len   = is_hart1 ? (K_بلاکیں - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        متلون اعشاری * l2scp_خانہ = (متلون اعشاری *) et_shire_l2scp_local(local_minion * 64);

        برائے (int64_t i3 = 0; i3 < ne13; i3++) {
            ثابت int64_t i03       = i3 / r3;
            ثابت حرف *  src0_ptr3 = (ثابت حرف *) params->src0.ڈیٹا + i03 * nb03;
            ثابت حرف *  src1_ptr3 = (ثابت حرف *) params->src1.ڈیٹا + i3 * nb13;
            حرف *        dst_ptr3  = (حرف *) params->dst.ڈیٹا + i3 * nbd3;

            برائے (int64_t i2 = 0; i2 < ne12; i2++) {
                ثابت int64_t i02       = i2 / r2;
                ثابت حرف *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ثابت حرف *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                حرف *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                برائے (int64_t n = 0; n < N; n++) {
                    ثابت اعشاری * b_ستون_بنیاد = (ثابت اعشاری *) (src1_ptr2 + n * nb11);

                    برائے (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        ثابت بلاک_q4_0 * q_صف = (ثابت بلاک_q4_0 *) (src0_ptr2 + m * nb01);

                        اعشاری partial = compute_صف_dot_q4_0(q_صف + k_آغاز, b_ستون_بنیاد + k_آغاز * 32, k_len);

                        اگر (is_hart1) {
                            *l2scp_خانہ = partial;
                            FENCE;
                            flush_مقصد_l2((ثابت خالی *) l2scp_خانہ, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } ورنہ {
                            et_sem_wait(ET_BARRIER_MINION);
                            اعشاری other = *l2scp_خانہ;
                            et_sem_post(ET_BARRIER_MINION);

                            اعشاری * dst_اندراج = (اعشاری *) (dst_ptr2 + n * nbd1 + m * حجم(اعشاری));
                            atomic_store_f32((متلون اعشاری *) dst_اندراج, partial + other);
                        }
                    }
                }
            }
        }
    } ورنہ اگر (use_ksplit_گروہ) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime.
         *
         * Both harts process the same 4-row group, each on half of K, and
         * exchange 4 partial sums once per group instead of once per row.
         * This keeps the K-split bandwidth benefit while cutting semaphore
         * traffic by 4x relative to the old per-row exchange.
         */
        ثابت int64_t    k_آغاز    = is_hart1 ? k_half : 0;
        ثابت int64_t    k_len      = is_hart1 ? (K_بلاکیں - k_half) : k_half;
        متلون اعشاری * l2scp_خانہ = (متلون اعشاری *) et_shire_l2scp_local(local_minion * 64);

        برائے (int64_t i3 = 0; i3 < ne13; i3++) {
            ثابت int64_t i03       = i3 / r3;
            ثابت حرف *  src0_ptr3 = (ثابت حرف *) params->src0.ڈیٹا + i03 * nb03;
            ثابت حرف *  src1_ptr3 = (ثابت حرف *) params->src1.ڈیٹا + i3 * nb13;
            حرف *        dst_ptr3  = (حرف *) params->dst.ڈیٹا + i3 * nbd3;

            برائے (int64_t i2 = 0; i2 < ne12; i2++) {
                ثابت int64_t i02       = i2 / r2;
                ثابت حرف *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ثابت حرف *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                حرف *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                برائے (int64_t n = 0; n < N; n++) {
                    ثابت اعشاری * b_ستون_بنیاد = (ثابت اعشاری *) (src1_ptr2 + n * nb11);

                    برائے (int64_t m_بنیاد = minion_id; m_بنیاد < M; m_بنیاد += STRIDE_M_KSPLIT * KSPLIT_گروہ_صفیں) {
                        ثابت int64_t m0 = m_بنیاد;
                        ثابت int64_t m1 = m0 + STRIDE_M_KSPLIT;
                        ثابت int64_t m2 = m1 + STRIDE_M_KSPLIT;
                        ثابت int64_t m3 = m2 + STRIDE_M_KSPLIT;

                        اعشاری s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        برائے (int64_t kb = 0; kb < K_بلاکیں; kb += TILE_KB) {
                            int64_t tile_len = k_len - kb;
                            اگر (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            اگر (tile_len <= 0) {
                                توڑو;
                            }
                            ثابت اعشاری * b_tile = b_ستون_بنیاد + (k_آغاز + kb) * 32;
                            ثابت int64_t صف_kb = k_آغاز + kb;

                            اگر (m0 < M) {
                                s0 += compute_صف_dot_q4_0((ثابت بلاک_q4_0 *) (src0_ptr2 + m0 * nb01) + صف_kb,
                                                           b_tile, tile_len);
                            }
                            اگر (m1 < M) {
                                s1 += compute_صف_dot_q4_0((ثابت بلاک_q4_0 *) (src0_ptr2 + m1 * nb01) + صف_kb,
                                                           b_tile, tile_len);
                            }
                            اگر (m2 < M) {
                                s2 += compute_صف_dot_q4_0((ثابت بلاک_q4_0 *) (src0_ptr2 + m2 * nb01) + صف_kb,
                                                           b_tile, tile_len);
                            }
                            اگر (m3 < M) {
                                s3 += compute_صف_dot_q4_0((ثابت بلاک_q4_0 *) (src0_ptr2 + m3 * nb01) + صف_kb,
                                                           b_tile, tile_len);
                            }
                        }

                        اگر (is_hart1) {
                            l2scp_خانہ[0] = s0;
                            l2scp_خانہ[1] = s1;
                            l2scp_خانہ[2] = s2;
                            l2scp_خانہ[3] = s3;
                            FENCE;
                            flush_مقصد_l2((ثابت خالی *) l2scp_خانہ, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } ورنہ {
                            et_sem_wait(ET_BARRIER_MINION);
                            ثابت اعشاری p0 = l2scp_خانہ[0];
                            ثابت اعشاری p1 = l2scp_خانہ[1];
                            ثابت اعشاری p2 = l2scp_خانہ[2];
                            ثابت اعشاری p3 = l2scp_خانہ[3];
                            et_sem_post(ET_BARRIER_MINION);

                            اعشاری * c_بنیاد = (اعشاری *) (dst_ptr2 + n * nbd1);
                            اگر (m0 < M) {
                                atomic_store_f32((متلون اعشاری *) (c_بنیاد + m0), s0 + p0);
                            }
                            اگر (m1 < M) {
                                atomic_store_f32((متلون اعشاری *) (c_بنیاد + m1), s1 + p1);
                            }
                            اگر (m2 < M) {
                                atomic_store_f32((متلون اعشاری *) (c_بنیاد + m2), s2 + p2);
                            }
                            اگر (m3 < M) {
                                atomic_store_f32((متلون اعشاری *) (c_بنیاد + m3), s3 + p3);
                            }
                        }
                    }
                }
            }
        }
    } ورنہ اگر (K_بلاکیں > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        برائے (int64_t i3 = 0; i3 < ne13; i3++) {
            ثابت int64_t i03       = i3 / r3;
            ثابت حرف *  src0_ptr3 = (ثابت حرف *) params->src0.ڈیٹا + i03 * nb03;
            ثابت حرف *  src1_ptr3 = (ثابت حرف *) params->src1.ڈیٹا + i3 * nb13;
            حرف *        dst_ptr3  = (حرف *) params->dst.ڈیٹا + i3 * nbd3;

            برائے (int64_t i2 = 0; i2 < ne12; i2++) {
                ثابت int64_t i02       = i2 / r2;
                ثابت حرف *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ثابت حرف *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                حرف *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                برائے (int64_t n = 0; n < N; n++) {
                    ثابت اعشاری * b_ستون_بنیاد = (ثابت اعشاری *) (src1_ptr2 + n * nb11);

                    برائے (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        ثابت int64_t m1 = m0 + STRIDE_M;
                        ثابت int64_t m2 = m0 + STRIDE_M * 2;
                        ثابت int64_t m3 = m0 + STRIDE_M * 3;

                        اعشاری s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        برائے (int64_t kb = 0; kb < K_بلاکیں; kb += TILE_KB) {
                            int64_t tile_len = K_بلاکیں - kb;
                            اگر (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            ثابت اعشاری * b_tile = b_ستون_بنیاد + kb * 32;

                            s0 += compute_صف_dot_q4_0((ثابت بلاک_q4_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            اگر (m1 < M) {
                                s1 += compute_صف_dot_q4_0((ثابت بلاک_q4_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            اگر (m2 < M) {
                                s2 += compute_صف_dot_q4_0((ثابت بلاک_q4_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            اگر (m3 < M) {
                                s3 += compute_صف_dot_q4_0((ثابت بلاک_q4_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        اعشاری * dst_بنیاد = (اعشاری *) (dst_ptr2 + n * nbd1);
                        atomic_store_f32((متلون اعشاری *) (dst_بنیاد + m0), s0);
                        اگر (m1 < M) {
                            atomic_store_f32((متلون اعشاری *) (dst_بنیاد + m1), s1);
                        }
                        اگر (m2 < M) {
                            atomic_store_f32((متلون اعشاری *) (dst_بنیاد + m2), s2);
                        }
                        اگر (m3 < M) {
                            atomic_store_f32((متلون اعشاری *) (dst_بنیاد + m3), s3);
                        }
                    }
                }
            }
        }
    } ورنہ {
        /*
         * Simple path for small K.
         *
         * When `nb01` is 32-byte aligned, every row has the same block-alignment
         * pattern. That lets us compute two rows together and reuse each loaded
         * B chunk across both rows instead of reloading it in a second dot call.
         */
        برائے (int64_t i3 = 0; i3 < ne13; i3++) {
            ثابت int64_t i03       = i3 / r3;
            ثابت حرف *  src0_ptr3 = (ثابت حرف *) params->src0.ڈیٹا + i03 * nb03;
            ثابت حرف *  src1_ptr3 = (ثابت حرف *) params->src1.ڈیٹا + i3 * nb13;
            حرف *        dst_ptr3  = (حرف *) params->dst.ڈیٹا + i3 * nbd3;

            برائے (int64_t i2 = 0; i2 < ne12; i2++) {
                ثابت int64_t i02       = i2 / r2;
                ثابت حرف *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ثابت حرف *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                حرف *        dst_ptr2  = dst_ptr3 + i2 * nbd2;

                برائے (int64_t n = 0; n < N; n++) {
                    ثابت اعشاری * b_ستون_بنیاد = (ثابت اعشاری *) (src1_ptr2 + n * nb11);
                    q4_dot_state  q4_state;
                    q4_dot_begin(&q4_state);

                    اگر (use_simple_x2) {
                        برائے (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_صفیں) {
                            ثابت int64_t      m1     = m0 + STRIDE_M;
                            ثابت بلاک_q4_0 * q_row0 = (ثابت بلاک_q4_0 *) (src0_ptr2 + m0 * nb01);

                            اگر (m1 < M) {
                                ثابت بلاک_q4_0 * q_row1 = (ثابت بلاک_q4_0 *) (src0_ptr2 + m1 * nb01);
                                اعشاری              s0, s1;
                                q4_dot_compute_x2_aligned(q_row0, q_row1, b_ستون_بنیاد, K_بلاکیں, &s0, &s1);

                                اعشاری * dst0 = (اعشاری *) (dst_ptr2 + n * nbd1 + m0 * حجم(اعشاری));
                                اعشاری * dst1 = (اعشاری *) (dst_ptr2 + n * nbd1 + m1 * حجم(اعشاری));
                                atomic_store_f32((متلون اعشاری *) dst0, s0);
                                atomic_store_f32((متلون اعشاری *) dst1, s1);
                            } ورنہ {
                                اعشاری   مجموع = q4_dot_compute(q_row0, b_ستون_بنیاد, K_بلاکیں);
                                اعشاری * dst = (اعشاری *) (dst_ptr2 + n * nbd1 + m0 * حجم(اعشاری));
                                atomic_store_f32((متلون اعشاری *) dst, مجموع);
                            }
                        }
                    } ورنہ {
                        برائے (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            ثابت بلاک_q4_0 * q_صف = (ثابت بلاک_q4_0 *) (src0_ptr2 + m * nb01);

                            اعشاری مجموع = q4_dot_compute(q_صف, b_ستون_بنیاد, K_بلاکیں);

                            اعشاری * dst_اندراج = (اعشاری *) (dst_ptr2 + n * nbd1 + m * حجم(اعشاری));
                            atomic_store_f32((متلون اعشاری *) dst_اندراج, مجموع);
                        }
                    }

                    q4_dot_انتہا(&q4_state);
                }
            }
        }
    }

    واپس 0;
}
