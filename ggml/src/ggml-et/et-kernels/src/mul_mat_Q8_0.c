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

#وصف_ڪريو STRIDE_M                   2048 /* 32 shires x 32 minions x 2 harts */
#وصف_ڪريو STRIDE_M_KSPLIT            1024 /* 32 shires x 32 minions (both harts share rows) */
#وصف_ڪريو KSPLIT_MIN_K_بلاڪون        256  /* K >= 8192 elements */
#وصف_ڪريو KSPLIT_SMALL_صفون_K_بلاڪون 64   /* K >= 2048 elements for very small M */
#وصف_ڪريو KSPLIT_MAX_صفون            8    /* max rows per minion for K-split */
#وصف_ڪريو TILE_KB                    256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */
#وصف_ڪريو KSPLIT_گروهه_صفون          4
#وصف_ڪريو SIMPLE_X2_صفون             2

ساكن مضمن size_t tensor_بائيٽون(ثابت بنية ggml_tensor * t) {
    أرجع (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

صحيح داخلا_نقطو(بنية ggml_et_mm_q8_params * params, فراغ * env) {
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

    // Optional residual bias
    ثابت حرف * bias_بنياد = (ثابت حرف *) params->bias.معلومات;
    ثابت size_t nbb1      = params->bias.nb[1];
    ثابت size_t nbb2      = params->bias.nb[2];
    ثابت size_t nbb3      = params->bias.nb[3];

    // Q8_0 block size is 32
    ثابت int64_t K_بلاڪون      = K / 32;
    ثابت صحيح     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    ثابت int64_t r2 = ne12 / ne02;
    ثابت int64_t r3 = ne13 / ne03;

    // K-split decision
    ثابت int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    ثابت int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    ثابت صحيح     is_hart1              = hart_id & 1;
    ثابت int64_t صفون_في_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    ثابت int64_t k_half                = K_بلاڪون / 2;
    ثابت صحيح     use_ksplit_small_صفون = (صفون_في_minion <= 2) && (K_بلاڪون >= KSPLIT_SMALL_صفون_K_بلاڪون);
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
    ثابت صحيح     use_ksplit            = ((K_بلاڪون >= KSPLIT_MIN_K_بلاڪون) && (صفون_في_minion <= KSPLIT_MAX_صفون) &&
                                           (صفون_في_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_صفون;
    ثابت صحيح     use_ksplit_گروهه      = !use_ksplit && (K_بلاڪون >= KSPLIT_MIN_K_بلاڪون) && (صفون_في_minion > 4) &&
                                          (صفون_في_minion <= KSPLIT_MAX_صفون);

    evict_region_past_l2(params->src1.معلومات, tensor_بائيٽون(&params->src1));
    إذا (params->bias.معلومات) {
        evict_region_past_l2(params->bias.معلومات, tensor_بائيٽون(&params->bias));
    }

    إذا (use_ksplit) {
        /* Each hart processes half the K dimension */
        ثابت int64_t k_شروعات = is_hart1 ? k_half : 0;
        ثابت int64_t k_len   = is_hart1 ? (K_بلاڪون - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        متطاير عائم * l2scp_جڳهه = (متطاير عائم *) et_shire_l2scp_local(local_minion * 64);

        لكل (int64_t i3 = 0; i3 < ne13; i3++) {
            ثابت int64_t i03       = i3 / r3;
            ثابت حرف *  src0_ptr3 = (ثابت حرف *) params->src0.معلومات + i03 * nb03;
            ثابت حرف *  src1_ptr3 = (ثابت حرف *) params->src1.معلومات + i3 * nb13;
            حرف *        dst_ptr3  = (حرف *) params->dst.معلومات + i3 * nbd3;
            ثابت حرف *  bias_ptr3 = bias_بنياد ? bias_بنياد + i3 * nbb3 : (ثابت حرف *) 0;

            لكل (int64_t i2 = 0; i2 < ne12; i2++) {
                ثابت int64_t i02       = i2 / r2;
                ثابت حرف *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ثابت حرف *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                حرف *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ثابت حرف *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ثابت حرف *) 0;

                لكل (int64_t n = 0; n < N; n++) {
                    ثابت عائم * b_کانو_بنياد = (ثابت عائم *) (src1_ptr2 + n * nb11);
                    ثابت عائم * bias_n     = bias_ptr2 ? (ثابت عائم *) (bias_ptr2 + n * nbb1) : (ثابت عائم *) 0;

                    لكل (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        ثابت بلاڪ_q8_0 * q_صف = (ثابت بلاڪ_q8_0 *) (src0_ptr2 + m * nb01);

                        عائم partial = compute_صف_dot_q8_0(q_صف + k_شروعات, b_کانو_بنياد + k_شروعات * 32, k_len);

                        إذا (is_hart1) {
                            *l2scp_جڳهه = partial;
                            FENCE;
                            flush_منزل_l2((ثابت فراغ *) l2scp_جڳهه, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } وإلا {
                            et_sem_wait(ET_BARRIER_MINION);
                            عائم other = *l2scp_جڳهه;
                            et_sem_post(ET_BARRIER_MINION);

                            عائم * dst_داخلا = (عائم *) (dst_ptr2 + n * nbd1 + m * حجم(عائم));
                            عائم   جوڙ       = partial + other;
                            إذا (bias_n) {
                                جوڙ += bias_n[m];
                            }
                            atomic_store_f32((متطاير عائم *) dst_داخلا, جوڙ);
                        }
                    }
                }
            }
        }
    } وإلا إذا (use_ksplit_گروهه) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime.
         *
         * Both harts process the same 4-row group, each on half of K, and
         * exchange 4 partial sums once per group instead of once per row.
         * This keeps the K-split bandwidth benefit while cutting semaphore
         * traffic by 4x relative to the old per-row exchange.
         */
        ثابت int64_t    k_شروعات    = is_hart1 ? k_half : 0;
        ثابت int64_t    k_len      = is_hart1 ? (K_بلاڪون - k_half) : k_half;
        متطاير عائم * l2scp_جڳهه = (متطاير عائم *) et_shire_l2scp_local(local_minion * 64);

        لكل (int64_t i3 = 0; i3 < ne13; i3++) {
            ثابت int64_t i03       = i3 / r3;
            ثابت حرف *  src0_ptr3 = (ثابت حرف *) params->src0.معلومات + i03 * nb03;
            ثابت حرف *  src1_ptr3 = (ثابت حرف *) params->src1.معلومات + i3 * nb13;
            حرف *        dst_ptr3  = (حرف *) params->dst.معلومات + i3 * nbd3;
            ثابت حرف *  bias_ptr3 = bias_بنياد ? bias_بنياد + i3 * nbb3 : (ثابت حرف *) 0;

            لكل (int64_t i2 = 0; i2 < ne12; i2++) {
                ثابت int64_t i02       = i2 / r2;
                ثابت حرف *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ثابت حرف *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                حرف *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ثابت حرف *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ثابت حرف *) 0;

                لكل (int64_t n = 0; n < N; n++) {
                    ثابت عائم * b_کانو_بنياد = (ثابت عائم *) (src1_ptr2 + n * nb11);
                    ثابت عائم * bias_n     = bias_ptr2 ? (ثابت عائم *) (bias_ptr2 + n * nbb1) : (ثابت عائم *) 0;

                    لكل (int64_t m_بنياد = minion_id; m_بنياد < M; m_بنياد += STRIDE_M_KSPLIT * KSPLIT_گروهه_صفون) {
                        ثابت int64_t m0 = m_بنياد;
                        ثابت int64_t m1 = m0 + STRIDE_M_KSPLIT;
                        ثابت int64_t m2 = m1 + STRIDE_M_KSPLIT;
                        ثابت int64_t m3 = m2 + STRIDE_M_KSPLIT;

                        عائم s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        لكل (int64_t kb = 0; kb < K_بلاڪون; kb += TILE_KB) {
                            int64_t tile_len = k_len - kb;
                            إذا (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            إذا (tile_len <= 0) {
                                اكسر;
                            }
                            ثابت عائم * b_tile = b_کانو_بنياد + (k_شروعات + kb) * 32;
                            ثابت int64_t صف_kb = k_شروعات + kb;

                            إذا (m0 < M) {
                                s0 += compute_صف_dot_q8_0((ثابت بلاڪ_q8_0 *) (src0_ptr2 + m0 * nb01) + صف_kb,
                                                           b_tile, tile_len);
                            }
                            إذا (m1 < M) {
                                s1 += compute_صف_dot_q8_0((ثابت بلاڪ_q8_0 *) (src0_ptr2 + m1 * nb01) + صف_kb,
                                                           b_tile, tile_len);
                            }
                            إذا (m2 < M) {
                                s2 += compute_صف_dot_q8_0((ثابت بلاڪ_q8_0 *) (src0_ptr2 + m2 * nb01) + صف_kb,
                                                           b_tile, tile_len);
                            }
                            إذا (m3 < M) {
                                s3 += compute_صف_dot_q8_0((ثابت بلاڪ_q8_0 *) (src0_ptr2 + m3 * nb01) + صف_kb,
                                                           b_tile, tile_len);
                            }
                        }

                        إذا (is_hart1) {
                            l2scp_جڳهه[0] = s0;
                            l2scp_جڳهه[1] = s1;
                            l2scp_جڳهه[2] = s2;
                            l2scp_جڳهه[3] = s3;
                            FENCE;
                            flush_منزل_l2((ثابت فراغ *) l2scp_جڳهه, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } وإلا {
                            et_sem_wait(ET_BARRIER_MINION);
                            ثابت عائم p0 = l2scp_جڳهه[0];
                            ثابت عائم p1 = l2scp_جڳهه[1];
                            ثابت عائم p2 = l2scp_جڳهه[2];
                            ثابت عائم p3 = l2scp_جڳهه[3];
                            et_sem_post(ET_BARRIER_MINION);

                            عائم *     c_بنياد = (عائم *) (dst_ptr2 + n * nbd1);
                            ثابت عائم b0     = bias_n ? bias_n[m0] : 0.0f;
                            ثابت عائم b1     = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                            ثابت عائم b2     = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                            ثابت عائم b3     = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                            إذا (m0 < M) {
                                atomic_store_f32((متطاير عائم *) (c_بنياد + m0), s0 + p0 + b0);
                            }
                            إذا (m1 < M) {
                                atomic_store_f32((متطاير عائم *) (c_بنياد + m1), s1 + p1 + b1);
                            }
                            إذا (m2 < M) {
                                atomic_store_f32((متطاير عائم *) (c_بنياد + m2), s2 + p2 + b2);
                            }
                            إذا (m3 < M) {
                                atomic_store_f32((متطاير عائم *) (c_بنياد + m3), s3 + p3 + b3);
                            }
                        }
                    }
                }
            }
        }
    } وإلا إذا (K_بلاڪون > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        لكل (int64_t i3 = 0; i3 < ne13; i3++) {
            ثابت int64_t i03       = i3 / r3;
            ثابت حرف *  src0_ptr3 = (ثابت حرف *) params->src0.معلومات + i03 * nb03;
            ثابت حرف *  src1_ptr3 = (ثابت حرف *) params->src1.معلومات + i3 * nb13;
            حرف *        dst_ptr3  = (حرف *) params->dst.معلومات + i3 * nbd3;
            ثابت حرف *  bias_ptr3 = bias_بنياد ? bias_بنياد + i3 * nbb3 : (ثابت حرف *) 0;

            لكل (int64_t i2 = 0; i2 < ne12; i2++) {
                ثابت int64_t i02       = i2 / r2;
                ثابت حرف *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ثابت حرف *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                حرف *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ثابت حرف *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ثابت حرف *) 0;

                لكل (int64_t n = 0; n < N; n++) {
                    ثابت عائم * b_کانو_بنياد = (ثابت عائم *) (src1_ptr2 + n * nb11);
                    ثابت عائم * bias_n     = bias_ptr2 ? (ثابت عائم *) (bias_ptr2 + n * nbb1) : (ثابت عائم *) 0;

                    لكل (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        ثابت int64_t m1 = m0 + STRIDE_M;
                        ثابت int64_t m2 = m0 + STRIDE_M * 2;
                        ثابت int64_t m3 = m0 + STRIDE_M * 3;

                        عائم s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        لكل (int64_t kb = 0; kb < K_بلاڪون; kb += TILE_KB) {
                            int64_t tile_len = K_بلاڪون - kb;
                            إذا (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            ثابت عائم * b_tile = b_کانو_بنياد + kb * 32;

                            s0 += compute_صف_dot_q8_0((ثابت بلاڪ_q8_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            إذا (m1 < M) {
                                s1 += compute_صف_dot_q8_0((ثابت بلاڪ_q8_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            إذا (m2 < M) {
                                s2 += compute_صف_dot_q8_0((ثابت بلاڪ_q8_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            إذا (m3 < M) {
                                s3 += compute_صف_dot_q8_0((ثابت بلاڪ_q8_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        عائم *     dst_بنياد = (عائم *) (dst_ptr2 + n * nbd1);
                        ثابت عائم b0       = bias_n ? bias_n[m0] : 0.0f;
                        ثابت عائم b1       = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                        ثابت عائم b2       = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                        ثابت عائم b3       = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                        atomic_store_f32((متطاير عائم *) (dst_بنياد + m0), s0 + b0);
                        إذا (m1 < M) {
                            atomic_store_f32((متطاير عائم *) (dst_بنياد + m1), s1 + b1);
                        }
                        إذا (m2 < M) {
                            atomic_store_f32((متطاير عائم *) (dst_بنياد + m2), s2 + b2);
                        }
                        إذا (m3 < M) {
                            atomic_store_f32((متطاير عائم *) (dst_بنياد + m3), s3 + b3);
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
            ثابت حرف *  src0_ptr3 = (ثابت حرف *) params->src0.معلومات + i03 * nb03;
            ثابت حرف *  src1_ptr3 = (ثابت حرف *) params->src1.معلومات + i3 * nb13;
            حرف *        dst_ptr3  = (حرف *) params->dst.معلومات + i3 * nbd3;
            ثابت حرف *  bias_ptr3 = bias_بنياد ? bias_بنياد + i3 * nbb3 : (ثابت حرف *) 0;

            لكل (int64_t i2 = 0; i2 < ne12; i2++) {
                ثابت int64_t i02       = i2 / r2;
                ثابت حرف *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ثابت حرف *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                حرف *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ثابت حرف *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ثابت حرف *) 0;

                لكل (int64_t n = 0; n < N; n++) {
                    ثابت عائم * b_کانو_بنياد = (ثابت عائم *) (src1_ptr2 + n * nb11);
                    ثابت عائم * bias_n     = bias_ptr2 ? (ثابت عائم *) (bias_ptr2 + n * nbb1) : (ثابت عائم *) 0;
                    q8_dot_state  q8_state;
                    q8_dot_begin(&q8_state);

                    إذا (use_simple_x2) {
                        لكل (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_صفون) {
                            ثابت int64_t      m1     = m0 + STRIDE_M;
                            ثابت بلاڪ_q8_0 * q_row0 = (ثابت بلاڪ_q8_0 *) (src0_ptr2 + m0 * nb01);

                            إذا (m1 < M) {
                                ثابت بلاڪ_q8_0 * q_row1 = (ثابت بلاڪ_q8_0 *) (src0_ptr2 + m1 * nb01);
                                عائم              s0, s1;
                                q8_dot_compute_x2_aligned(q_row0, q_row1, b_کانو_بنياد, K_بلاڪون, &s0, &s1);

                                عائم * dst0 = (عائم *) (dst_ptr2 + n * nbd1 + m0 * حجم(عائم));
                                عائم * dst1 = (عائم *) (dst_ptr2 + n * nbd1 + m1 * حجم(عائم));
                                إذا (bias_n) {
                                    s0 += bias_n[m0];
                                    s1 += bias_n[m1];
                                }
                                atomic_store_f32((متطاير عائم *) dst0, s0);
                                atomic_store_f32((متطاير عائم *) dst1, s1);
                            } وإلا {
                                عائم   جوڙ = q8_dot_compute(q_row0, b_کانو_بنياد, K_بلاڪون);
                                عائم * dst = (عائم *) (dst_ptr2 + n * nbd1 + m0 * حجم(عائم));
                                إذا (bias_n) {
                                    جوڙ += bias_n[m0];
                                }
                                atomic_store_f32((متطاير عائم *) dst, جوڙ);
                            }
                        }
                    } وإلا {
                        لكل (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            ثابت بلاڪ_q8_0 * q_صف = (ثابت بلاڪ_q8_0 *) (src0_ptr2 + m * nb01);

                            عائم جوڙ = q8_dot_compute(q_صف, b_کانو_بنياد, K_بلاڪون);

                            عائم * dst_داخلا = (عائم *) (dst_ptr2 + n * nbd1 + m * حجم(عائم));
                            إذا (bias_n) {
                                جوڙ += bias_n[m];
                            }
                            atomic_store_f32((متطاير عائم *) dst_داخلا, جوڙ);
                        }
                    }

                    q8_dot_پڇاڙي(&q8_state);
                }
            }
        }
    }

#جيڪڏهن_وصف_ٿيل ET_UBERKERNEL
    FENCE;
    evict_region_past_l2(params->dst.معلومات, tensor_بائيٽون(&params->dst));
    WAIT_CACHEOPS;
    FENCE;
#جيڪڏهن_پورو
    أرجع 0;
}
