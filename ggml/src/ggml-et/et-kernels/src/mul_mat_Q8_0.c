//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "block_ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "quants.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ STRIDE_M                   2048 /* 32 shires x 32 minions x 2 harts */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ STRIDE_M_KSPLIT            1024 /* 32 shires x 32 minions (both harts share rows) */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ KSPLIT_MIN_K_ਖੰਡਾਂ        256  /* K >= 8192 elements */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ KSPLIT_SMALL_ਪੰਕਤੀਆਂ_K_ਖੰਡਾਂ 64   /* K >= 2048 elements for very small M */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ KSPLIT_MAX_ਪੰਕਤੀਆਂ            8    /* max rows per minion for K-split */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE_KB                    256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ KSPLIT_ਸਮੂਹ_ਪੰਕਤੀਆਂ          4
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SIMPLE_X2_ਪੰਕਤੀਆਂ             2

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t tensor_ਬਾਈਟਾਂ(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਵਾਪਸ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_mm_q8_params * params, ਖਾਲੀ * env) {
    uint64_t hart_id = get_hart_id();

    // Matrix dimensions
    ਅਚਲ int64_t K    = params->src0.ne[0];
    ਅਚਲ int64_t M    = params->src0.ne[1];
    ਅਚਲ int64_t N    = params->src1.ne[1];
    ਅਚਲ int64_t ne02 = params->src0.ne[2];
    ਅਚਲ int64_t ne03 = params->src0.ne[3];
    ਅਚਲ int64_t ne12 = params->src1.ne[2];
    ਅਚਲ int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    ਅਚਲ size_t nb01 = params->src0.nb[1];
    ਅਚਲ size_t nb02 = params->src0.nb[2];
    ਅਚਲ size_t nb03 = params->src0.nb[3];

    ਅਚਲ size_t nb11 = params->src1.nb[1];
    ਅਚਲ size_t nb12 = params->src1.nb[2];
    ਅਚਲ size_t nb13 = params->src1.nb[3];

    ਅਚਲ size_t nbd1 = params->dst.nb[1];
    ਅਚਲ size_t nbd2 = params->dst.nb[2];
    ਅਚਲ size_t nbd3 = params->dst.nb[3];

    // Optional residual bias
    ਅਚਲ ਅੱਖਰ * bias_ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) params->bias.ਅੰਕੜੇ;
    ਅਚਲ size_t nbb1      = params->bias.nb[1];
    ਅਚਲ size_t nbb2      = params->bias.nb[2];
    ਅਚਲ size_t nbb3      = params->bias.nb[3];

    // Q8_0 block size is 32
    ਅਚਲ int64_t K_ਖੰਡਾਂ      = K / 32;
    ਅਚਲ ਪੂਰਨ     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    ਅਚਲ int64_t r2 = ne12 / ne02;
    ਅਚਲ int64_t r3 = ne13 / ne03;

    // K-split decision
    ਅਚਲ int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    ਅਚਲ int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    ਅਚਲ ਪੂਰਨ     is_hart1              = hart_id & 1;
    ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    ਅਚਲ int64_t k_half                = K_ਖੰਡਾਂ / 2;
    ਅਚਲ ਪੂਰਨ     use_ksplit_small_ਪੰਕਤੀਆਂ = (ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_minion <= 2) && (K_ਖੰਡਾਂ >= KSPLIT_SMALL_ਪੰਕਤੀਆਂ_K_ਖੰਡਾਂ);
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
    ਅਚਲ ਪੂਰਨ     use_ksplit            = ((K_ਖੰਡਾਂ >= KSPLIT_MIN_K_ਖੰਡਾਂ) && (ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_minion <= KSPLIT_MAX_ਪੰਕਤੀਆਂ) &&
                                           (ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_ਪੰਕਤੀਆਂ;
    ਅਚਲ ਪੂਰਨ     use_ksplit_ਸਮੂਹ      = !use_ksplit && (K_ਖੰਡਾਂ >= KSPLIT_MIN_K_ਖੰਡਾਂ) && (ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_minion > 4) &&
                                          (ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_minion <= KSPLIT_MAX_ਪੰਕਤੀਆਂ);

    evict_region_past_l2(params->src1.ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ(&params->src1));
    ਜੇ (params->bias.ਅੰਕੜੇ) {
        evict_region_past_l2(params->bias.ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ(&params->bias));
    }

    ਜੇ (use_ksplit) {
        /* Each hart processes half the K dimension */
        ਅਚਲ int64_t k_ਸ਼ੁਰੂ = is_hart1 ? k_half : 0;
        ਅਚਲ int64_t k_len   = is_hart1 ? (K_ਖੰਡਾਂ - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        ਅਸਥਿਰ ਦਸ਼ਮਲਵ * l2scp_ਥਾਂ = (ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) et_shire_l2scp_local(local_minion * 64);

        ਲਈ (int64_t i3 = 0; i3 < ne13; i3++) {
            ਅਚਲ int64_t i03       = i3 / r3;
            ਅਚਲ ਅੱਖਰ *  src0_ptr3 = (ਅਚਲ ਅੱਖਰ *) params->src0.ਅੰਕੜੇ + i03 * nb03;
            ਅਚਲ ਅੱਖਰ *  src1_ptr3 = (ਅਚਲ ਅੱਖਰ *) params->src1.ਅੰਕੜੇ + i3 * nb13;
            ਅੱਖਰ *        dst_ptr3  = (ਅੱਖਰ *) params->dst.ਅੰਕੜੇ + i3 * nbd3;
            ਅਚਲ ਅੱਖਰ *  bias_ptr3 = bias_ਅਧਾਰ ? bias_ਅਧਾਰ + i3 * nbb3 : (ਅਚਲ ਅੱਖਰ *) 0;

            ਲਈ (int64_t i2 = 0; i2 < ne12; i2++) {
                ਅਚਲ int64_t i02       = i2 / r2;
                ਅਚਲ ਅੱਖਰ *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ਅਚਲ ਅੱਖਰ *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                ਅੱਖਰ *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ਅਚਲ ਅੱਖਰ *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ਅਚਲ ਅੱਖਰ *) 0;

                ਲਈ (int64_t n = 0; n < N; n++) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ਅਧਾਰ = (ਅਚਲ ਦਸ਼ਮਲਵ *) (src1_ptr2 + n * nb11);
                    ਅਚਲ ਦਸ਼ਮਲਵ * bias_n     = bias_ptr2 ? (ਅਚਲ ਦਸ਼ਮਲਵ *) (bias_ptr2 + n * nbb1) : (ਅਚਲ ਦਸ਼ਮਲਵ *) 0;

                    ਲਈ (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        ਅਚਲ ਖੰਡ_q8_0 * q_ਪੰਕਤੀ = (ਅਚਲ ਖੰਡ_q8_0 *) (src0_ptr2 + m * nb01);

                        ਦਸ਼ਮਲਵ partial = compute_ਪੰਕਤੀ_dot_q8_0(q_ਪੰਕਤੀ + k_ਸ਼ੁਰੂ, b_ਖਾਨਾ_ਅਧਾਰ + k_ਸ਼ੁਰੂ * 32, k_len);

                        ਜੇ (is_hart1) {
                            *l2scp_ਥਾਂ = partial;
                            FENCE;
                            flush_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) l2scp_ਥਾਂ, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } ਨਹੀਂਤਾਂ {
                            et_sem_wait(ET_BARRIER_MINION);
                            ਦਸ਼ਮਲਵ other = *l2scp_ਥਾਂ;
                            et_sem_post(ET_BARRIER_MINION);

                            ਦਸ਼ਮਲਵ * dst_ਇੰਦਰਾਜ = (ਦਸ਼ਮਲਵ *) (dst_ptr2 + n * nbd1 + m * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                            ਦਸ਼ਮਲਵ   ਜੋੜ       = partial + other;
                            ਜੇ (bias_n) {
                                ਜੋੜ += bias_n[m];
                            }
                            atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) dst_ਇੰਦਰਾਜ, ਜੋੜ);
                        }
                    }
                }
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (use_ksplit_ਸਮੂਹ) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime.
         *
         * Both harts process the same 4-row group, each on half of K, and
         * exchange 4 partial sums once per group instead of once per row.
         * This keeps the K-split bandwidth benefit while cutting semaphore
         * traffic by 4x relative to the old per-row exchange.
         */
        ਅਚਲ int64_t    k_ਸ਼ੁਰੂ    = is_hart1 ? k_half : 0;
        ਅਚਲ int64_t    k_len      = is_hart1 ? (K_ਖੰਡਾਂ - k_half) : k_half;
        ਅਸਥਿਰ ਦਸ਼ਮਲਵ * l2scp_ਥਾਂ = (ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) et_shire_l2scp_local(local_minion * 64);

        ਲਈ (int64_t i3 = 0; i3 < ne13; i3++) {
            ਅਚਲ int64_t i03       = i3 / r3;
            ਅਚਲ ਅੱਖਰ *  src0_ptr3 = (ਅਚਲ ਅੱਖਰ *) params->src0.ਅੰਕੜੇ + i03 * nb03;
            ਅਚਲ ਅੱਖਰ *  src1_ptr3 = (ਅਚਲ ਅੱਖਰ *) params->src1.ਅੰਕੜੇ + i3 * nb13;
            ਅੱਖਰ *        dst_ptr3  = (ਅੱਖਰ *) params->dst.ਅੰਕੜੇ + i3 * nbd3;
            ਅਚਲ ਅੱਖਰ *  bias_ptr3 = bias_ਅਧਾਰ ? bias_ਅਧਾਰ + i3 * nbb3 : (ਅਚਲ ਅੱਖਰ *) 0;

            ਲਈ (int64_t i2 = 0; i2 < ne12; i2++) {
                ਅਚਲ int64_t i02       = i2 / r2;
                ਅਚਲ ਅੱਖਰ *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ਅਚਲ ਅੱਖਰ *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                ਅੱਖਰ *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ਅਚਲ ਅੱਖਰ *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ਅਚਲ ਅੱਖਰ *) 0;

                ਲਈ (int64_t n = 0; n < N; n++) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ਅਧਾਰ = (ਅਚਲ ਦਸ਼ਮਲਵ *) (src1_ptr2 + n * nb11);
                    ਅਚਲ ਦਸ਼ਮਲਵ * bias_n     = bias_ptr2 ? (ਅਚਲ ਦਸ਼ਮਲਵ *) (bias_ptr2 + n * nbb1) : (ਅਚਲ ਦਸ਼ਮਲਵ *) 0;

                    ਲਈ (int64_t m_ਅਧਾਰ = minion_id; m_ਅਧਾਰ < M; m_ਅਧਾਰ += STRIDE_M_KSPLIT * KSPLIT_ਸਮੂਹ_ਪੰਕਤੀਆਂ) {
                        ਅਚਲ int64_t m0 = m_ਅਧਾਰ;
                        ਅਚਲ int64_t m1 = m0 + STRIDE_M_KSPLIT;
                        ਅਚਲ int64_t m2 = m1 + STRIDE_M_KSPLIT;
                        ਅਚਲ int64_t m3 = m2 + STRIDE_M_KSPLIT;

                        ਦਸ਼ਮਲਵ s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        ਲਈ (int64_t kb = 0; kb < K_ਖੰਡਾਂ; kb += TILE_KB) {
                            int64_t tile_len = k_len - kb;
                            ਜੇ (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            ਜੇ (tile_len <= 0) {
                                ਰੋਕੋ;
                            }
                            ਅਚਲ ਦਸ਼ਮਲਵ * b_tile = b_ਖਾਨਾ_ਅਧਾਰ + (k_ਸ਼ੁਰੂ + kb) * 32;
                            ਅਚਲ int64_t ਪੰਕਤੀ_kb = k_ਸ਼ੁਰੂ + kb;

                            ਜੇ (m0 < M) {
                                s0 += compute_ਪੰਕਤੀ_dot_q8_0((ਅਚਲ ਖੰਡ_q8_0 *) (src0_ptr2 + m0 * nb01) + ਪੰਕਤੀ_kb,
                                                           b_tile, tile_len);
                            }
                            ਜੇ (m1 < M) {
                                s1 += compute_ਪੰਕਤੀ_dot_q8_0((ਅਚਲ ਖੰਡ_q8_0 *) (src0_ptr2 + m1 * nb01) + ਪੰਕਤੀ_kb,
                                                           b_tile, tile_len);
                            }
                            ਜੇ (m2 < M) {
                                s2 += compute_ਪੰਕਤੀ_dot_q8_0((ਅਚਲ ਖੰਡ_q8_0 *) (src0_ptr2 + m2 * nb01) + ਪੰਕਤੀ_kb,
                                                           b_tile, tile_len);
                            }
                            ਜੇ (m3 < M) {
                                s3 += compute_ਪੰਕਤੀ_dot_q8_0((ਅਚਲ ਖੰਡ_q8_0 *) (src0_ptr2 + m3 * nb01) + ਪੰਕਤੀ_kb,
                                                           b_tile, tile_len);
                            }
                        }

                        ਜੇ (is_hart1) {
                            l2scp_ਥਾਂ[0] = s0;
                            l2scp_ਥਾਂ[1] = s1;
                            l2scp_ਥਾਂ[2] = s2;
                            l2scp_ਥਾਂ[3] = s3;
                            FENCE;
                            flush_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) l2scp_ਥਾਂ, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } ਨਹੀਂਤਾਂ {
                            et_sem_wait(ET_BARRIER_MINION);
                            ਅਚਲ ਦਸ਼ਮਲਵ p0 = l2scp_ਥਾਂ[0];
                            ਅਚਲ ਦਸ਼ਮਲਵ p1 = l2scp_ਥਾਂ[1];
                            ਅਚਲ ਦਸ਼ਮਲਵ p2 = l2scp_ਥਾਂ[2];
                            ਅਚਲ ਦਸ਼ਮਲਵ p3 = l2scp_ਥਾਂ[3];
                            et_sem_post(ET_BARRIER_MINION);

                            ਦਸ਼ਮਲਵ *     c_ਅਧਾਰ = (ਦਸ਼ਮਲਵ *) (dst_ptr2 + n * nbd1);
                            ਅਚਲ ਦਸ਼ਮਲਵ b0     = bias_n ? bias_n[m0] : 0.0f;
                            ਅਚਲ ਦਸ਼ਮਲਵ b1     = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                            ਅਚਲ ਦਸ਼ਮਲਵ b2     = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                            ਅਚਲ ਦਸ਼ਮਲਵ b3     = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                            ਜੇ (m0 < M) {
                                atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) (c_ਅਧਾਰ + m0), s0 + p0 + b0);
                            }
                            ਜੇ (m1 < M) {
                                atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) (c_ਅਧਾਰ + m1), s1 + p1 + b1);
                            }
                            ਜੇ (m2 < M) {
                                atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) (c_ਅਧਾਰ + m2), s2 + p2 + b2);
                            }
                            ਜੇ (m3 < M) {
                                atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) (c_ਅਧਾਰ + m3), s3 + p3 + b3);
                            }
                        }
                    }
                }
            }
        }
    } ਨਹੀਂਤਾਂ ਜੇ (K_ਖੰਡਾਂ > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        ਲਈ (int64_t i3 = 0; i3 < ne13; i3++) {
            ਅਚਲ int64_t i03       = i3 / r3;
            ਅਚਲ ਅੱਖਰ *  src0_ptr3 = (ਅਚਲ ਅੱਖਰ *) params->src0.ਅੰਕੜੇ + i03 * nb03;
            ਅਚਲ ਅੱਖਰ *  src1_ptr3 = (ਅਚਲ ਅੱਖਰ *) params->src1.ਅੰਕੜੇ + i3 * nb13;
            ਅੱਖਰ *        dst_ptr3  = (ਅੱਖਰ *) params->dst.ਅੰਕੜੇ + i3 * nbd3;
            ਅਚਲ ਅੱਖਰ *  bias_ptr3 = bias_ਅਧਾਰ ? bias_ਅਧਾਰ + i3 * nbb3 : (ਅਚਲ ਅੱਖਰ *) 0;

            ਲਈ (int64_t i2 = 0; i2 < ne12; i2++) {
                ਅਚਲ int64_t i02       = i2 / r2;
                ਅਚਲ ਅੱਖਰ *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ਅਚਲ ਅੱਖਰ *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                ਅੱਖਰ *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ਅਚਲ ਅੱਖਰ *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ਅਚਲ ਅੱਖਰ *) 0;

                ਲਈ (int64_t n = 0; n < N; n++) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ਅਧਾਰ = (ਅਚਲ ਦਸ਼ਮਲਵ *) (src1_ptr2 + n * nb11);
                    ਅਚਲ ਦਸ਼ਮਲਵ * bias_n     = bias_ptr2 ? (ਅਚਲ ਦਸ਼ਮਲਵ *) (bias_ptr2 + n * nbb1) : (ਅਚਲ ਦਸ਼ਮਲਵ *) 0;

                    ਲਈ (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        ਅਚਲ int64_t m1 = m0 + STRIDE_M;
                        ਅਚਲ int64_t m2 = m0 + STRIDE_M * 2;
                        ਅਚਲ int64_t m3 = m0 + STRIDE_M * 3;

                        ਦਸ਼ਮਲਵ s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        ਲਈ (int64_t kb = 0; kb < K_ਖੰਡਾਂ; kb += TILE_KB) {
                            int64_t tile_len = K_ਖੰਡਾਂ - kb;
                            ਜੇ (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            ਅਚਲ ਦਸ਼ਮਲਵ * b_tile = b_ਖਾਨਾ_ਅਧਾਰ + kb * 32;

                            s0 += compute_ਪੰਕਤੀ_dot_q8_0((ਅਚਲ ਖੰਡ_q8_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            ਜੇ (m1 < M) {
                                s1 += compute_ਪੰਕਤੀ_dot_q8_0((ਅਚਲ ਖੰਡ_q8_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            ਜੇ (m2 < M) {
                                s2 += compute_ਪੰਕਤੀ_dot_q8_0((ਅਚਲ ਖੰਡ_q8_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            ਜੇ (m3 < M) {
                                s3 += compute_ਪੰਕਤੀ_dot_q8_0((ਅਚਲ ਖੰਡ_q8_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        ਦਸ਼ਮਲਵ *     dst_ਅਧਾਰ = (ਦਸ਼ਮਲਵ *) (dst_ptr2 + n * nbd1);
                        ਅਚਲ ਦਸ਼ਮਲਵ b0       = bias_n ? bias_n[m0] : 0.0f;
                        ਅਚਲ ਦਸ਼ਮਲਵ b1       = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                        ਅਚਲ ਦਸ਼ਮਲਵ b2       = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                        ਅਚਲ ਦਸ਼ਮਲਵ b3       = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                        atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) (dst_ਅਧਾਰ + m0), s0 + b0);
                        ਜੇ (m1 < M) {
                            atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) (dst_ਅਧਾਰ + m1), s1 + b1);
                        }
                        ਜੇ (m2 < M) {
                            atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) (dst_ਅਧਾਰ + m2), s2 + b2);
                        }
                        ਜੇ (m3 < M) {
                            atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) (dst_ਅਧਾਰ + m3), s3 + b3);
                        }
                    }
                }
            }
        }
    } ਨਹੀਂਤਾਂ {
        /*
         * Simple path for small K.
         *
         * When `nb01` is 32-byte aligned, every row has the same block-alignment
         * pattern. That lets us compute two rows together and reuse each loaded
         * B chunk across both rows instead of reloading it in a second dot call.
         */
        ਲਈ (int64_t i3 = 0; i3 < ne13; i3++) {
            ਅਚਲ int64_t i03       = i3 / r3;
            ਅਚਲ ਅੱਖਰ *  src0_ptr3 = (ਅਚਲ ਅੱਖਰ *) params->src0.ਅੰਕੜੇ + i03 * nb03;
            ਅਚਲ ਅੱਖਰ *  src1_ptr3 = (ਅਚਲ ਅੱਖਰ *) params->src1.ਅੰਕੜੇ + i3 * nb13;
            ਅੱਖਰ *        dst_ptr3  = (ਅੱਖਰ *) params->dst.ਅੰਕੜੇ + i3 * nbd3;
            ਅਚਲ ਅੱਖਰ *  bias_ptr3 = bias_ਅਧਾਰ ? bias_ਅਧਾਰ + i3 * nbb3 : (ਅਚਲ ਅੱਖਰ *) 0;

            ਲਈ (int64_t i2 = 0; i2 < ne12; i2++) {
                ਅਚਲ int64_t i02       = i2 / r2;
                ਅਚਲ ਅੱਖਰ *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ਅਚਲ ਅੱਖਰ *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                ਅੱਖਰ *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ਅਚਲ ਅੱਖਰ *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ਅਚਲ ਅੱਖਰ *) 0;

                ਲਈ (int64_t n = 0; n < N; n++) {
                    ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ਅਧਾਰ = (ਅਚਲ ਦਸ਼ਮਲਵ *) (src1_ptr2 + n * nb11);
                    ਅਚਲ ਦਸ਼ਮਲਵ * bias_n     = bias_ptr2 ? (ਅਚਲ ਦਸ਼ਮਲਵ *) (bias_ptr2 + n * nbb1) : (ਅਚਲ ਦਸ਼ਮਲਵ *) 0;
                    q8_dot_state  q8_state;
                    q8_dot_begin(&q8_state);

                    ਜੇ (use_simple_x2) {
                        ਲਈ (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_ਪੰਕਤੀਆਂ) {
                            ਅਚਲ int64_t      m1     = m0 + STRIDE_M;
                            ਅਚਲ ਖੰਡ_q8_0 * q_row0 = (ਅਚਲ ਖੰਡ_q8_0 *) (src0_ptr2 + m0 * nb01);

                            ਜੇ (m1 < M) {
                                ਅਚਲ ਖੰਡ_q8_0 * q_row1 = (ਅਚਲ ਖੰਡ_q8_0 *) (src0_ptr2 + m1 * nb01);
                                ਦਸ਼ਮਲਵ              s0, s1;
                                q8_dot_compute_x2_aligned(q_row0, q_row1, b_ਖਾਨਾ_ਅਧਾਰ, K_ਖੰਡਾਂ, &s0, &s1);

                                ਦਸ਼ਮਲਵ * dst0 = (ਦਸ਼ਮਲਵ *) (dst_ptr2 + n * nbd1 + m0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                                ਦਸ਼ਮਲਵ * dst1 = (ਦਸ਼ਮਲਵ *) (dst_ptr2 + n * nbd1 + m1 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                                ਜੇ (bias_n) {
                                    s0 += bias_n[m0];
                                    s1 += bias_n[m1];
                                }
                                atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) dst0, s0);
                                atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) dst1, s1);
                            } ਨਹੀਂਤਾਂ {
                                ਦਸ਼ਮਲਵ   ਜੋੜ = q8_dot_compute(q_row0, b_ਖਾਨਾ_ਅਧਾਰ, K_ਖੰਡਾਂ);
                                ਦਸ਼ਮਲਵ * dst = (ਦਸ਼ਮਲਵ *) (dst_ptr2 + n * nbd1 + m0 * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                                ਜੇ (bias_n) {
                                    ਜੋੜ += bias_n[m0];
                                }
                                atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) dst, ਜੋੜ);
                            }
                        }
                    } ਨਹੀਂਤਾਂ {
                        ਲਈ (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            ਅਚਲ ਖੰਡ_q8_0 * q_ਪੰਕਤੀ = (ਅਚਲ ਖੰਡ_q8_0 *) (src0_ptr2 + m * nb01);

                            ਦਸ਼ਮਲਵ ਜੋੜ = q8_dot_compute(q_ਪੰਕਤੀ, b_ਖਾਨਾ_ਅਧਾਰ, K_ਖੰਡਾਂ);

                            ਦਸ਼ਮਲਵ * dst_ਇੰਦਰਾਜ = (ਦਸ਼ਮਲਵ *) (dst_ptr2 + n * nbd1 + m * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                            ਜੇ (bias_n) {
                                ਜੋੜ += bias_n[m];
                            }
                            atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) dst_ਇੰਦਰਾਜ, ਜੋੜ);
                        }
                    }

                    q8_dot_ਸਿਰਾ(&q8_state);
                }
            }
        }
    }

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ET_UBERKERNEL
    FENCE;
    evict_region_past_l2(params->dst.ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ(&params->dst));
    WAIT_CACHEOPS;
    FENCE;
#ਜੇ_ਅੰਤ
    ਵਾਪਸ 0;
}
