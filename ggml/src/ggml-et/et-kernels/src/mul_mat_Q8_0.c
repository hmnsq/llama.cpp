//******************************************************************************
// MUL_MAT Kernel
// Matrix multiplication: C[M,N] = A[M,K] * B[K,N]
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "block_ops.h"
#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"
#ಅನ್ತರ್ಭಾವ "quants.h"

#ಅನ್ತರ್ಭಾವ <stdint.h>

#ವ್ಯಾಖ್ಯಾನಿಸು STRIDE_M                   2048 /* 32 shires x 32 minions x 2 harts */
#ವ್ಯಾಖ್ಯಾನಿಸು STRIDE_M_KSPLIT            1024 /* 32 shires x 32 minions (both harts share rows) */
#ವ್ಯಾಖ್ಯಾನಿಸು KSPLIT_MIN_K_ಖಂಡಗಳು        256  /* K >= 8192 elements */
#ವ್ಯಾಖ್ಯಾನಿಸು KSPLIT_SMALL_ಅಡ್ಡಸಾಲುಗಳು_K_ಖಂಡಗಳು 64   /* K >= 2048 elements for very small M */
#ವ್ಯಾಖ್ಯಾನಿಸು KSPLIT_MAX_ಅಡ್ಡಸಾಲುಗಳು            8    /* max rows per minion for K-split */
#ವ್ಯಾಖ್ಯಾನಿಸು TILE_KB                    256  /* K-tile size in Q8_0 blocks (8192 elems, 32KB B data) */
#ವ್ಯಾಖ್ಯಾನಿಸು KSPLIT_ಗುಂಪು_ಅಡ್ಡಸಾಲುಗಳು          4
#ವ್ಯಾಖ್ಯಾನಿಸು SIMPLE_X2_ಅಡ್ಡಸಾಲುಗಳು             2

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t tensor_ಬೈಟುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_mm_q8_params * params, ಶೂನ್ಯ * env) {
    uint64_t hart_id = get_hart_id();

    // Matrix dimensions
    ಸ್ಥಿರಾಂಕ int64_t K    = params->src0.ne[0];
    ಸ್ಥಿರಾಂಕ int64_t M    = params->src0.ne[1];
    ಸ್ಥಿರಾಂಕ int64_t N    = params->src1.ne[1];
    ಸ್ಥಿರಾಂಕ int64_t ne02 = params->src0.ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne03 = params->src0.ne[3];
    ಸ್ಥಿರಾಂಕ int64_t ne12 = params->src1.ne[2];
    ಸ್ಥಿರಾಂಕ int64_t ne13 = params->src1.ne[3];

    // Strides (in bytes)
    ಸ್ಥಿರಾಂಕ size_t nb01 = params->src0.nb[1];
    ಸ್ಥಿರಾಂಕ size_t nb02 = params->src0.nb[2];
    ಸ್ಥಿರಾಂಕ size_t nb03 = params->src0.nb[3];

    ಸ್ಥಿರಾಂಕ size_t nb11 = params->src1.nb[1];
    ಸ್ಥಿರಾಂಕ size_t nb12 = params->src1.nb[2];
    ಸ್ಥಿರಾಂಕ size_t nb13 = params->src1.nb[3];

    ಸ್ಥಿರಾಂಕ size_t nbd1 = params->dst.nb[1];
    ಸ್ಥಿರಾಂಕ size_t nbd2 = params->dst.nb[2];
    ಸ್ಥಿರಾಂಕ size_t nbd3 = params->dst.nb[3];

    // Optional residual bias
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * bias_ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) params->bias.ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ size_t nbb1      = params->bias.nb[1];
    ಸ್ಥಿರಾಂಕ size_t nbb2      = params->bias.nb[2];
    ಸ್ಥಿರಾಂಕ size_t nbb3      = params->bias.nb[3];

    // Q8_0 block size is 32
    ಸ್ಥಿರಾಂಕ int64_t K_ಖಂಡಗಳು      = K / 32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     use_simple_x2 = ((nb01 & 31) == 0);

    // Broadcasting ratios
    ಸ್ಥಿರಾಂಕ int64_t r2 = ne12 / ne02;
    ಸ್ಥಿರಾಂಕ int64_t r3 = ne13 / ne03;

    // K-split decision
    ಸ್ಥಿರಾಂಕ int64_t minion_id             = hart_id >> 1;          /* 0..1023 global */
    ಸ್ಥಿರಾಂಕ int64_t local_minion          = (hart_id >> 1) & 0x1F; /* 0..31 within shire */
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     is_hart1              = hart_id & 1;
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_minion       = (M + STRIDE_M_KSPLIT - 1) / STRIDE_M_KSPLIT;
    ಸ್ಥಿರಾಂಕ int64_t k_half                = K_ಖಂಡಗಳು / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     use_ksplit_small_ಅಡ್ಡಸಾಲುಗಳು = (ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_minion <= 2) && (K_ಖಂಡಗಳು >= KSPLIT_SMALL_ಅಡ್ಡಸಾಲುಗಳು_K_ಖಂಡಗಳು);
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
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     use_ksplit            = ((K_ಖಂಡಗಳು >= KSPLIT_MIN_K_ಖಂಡಗಳು) && (ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_minion <= KSPLIT_MAX_ಅಡ್ಡಸಾಲುಗಳು) &&
                                           (ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_minion <= 4 || k_half <= TILE_KB)) ||
                                          use_ksplit_small_ಅಡ್ಡಸಾಲುಗಳು;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     use_ksplit_ಗುಂಪು      = !use_ksplit && (K_ಖಂಡಗಳು >= KSPLIT_MIN_K_ಖಂಡಗಳು) && (ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_minion > 4) &&
                                          (ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_minion <= KSPLIT_MAX_ಅಡ್ಡಸಾಲುಗಳು);

    evict_region_past_l2(params->src1.ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು(&params->src1));
    ಆದರೆ (params->bias.ದತ್ತಾಂಶ) {
        evict_region_past_l2(params->bias.ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು(&params->bias));
    }

    ಆದರೆ (use_ksplit) {
        /* Each hart processes half the K dimension */
        ಸ್ಥಿರಾಂಕ int64_t k_ಆರಂಭ = is_hart1 ? k_half : 0;
        ಸ್ಥಿರಾಂಕ int64_t k_len   = is_hart1 ? (K_ಖಂಡಗಳು - k_half) : k_half;

        /* One cache-line-aligned L2SCP slot per minion for exchange */
        ಅಸ್ಥಿರ ದಶಮಾಂಶ * l2scp_ಸ್ಥಾನ = (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) et_shire_l2scp_local(local_minion * 64);

        ಗಾಗಿ (int64_t i3 = 0; i3 < ne13; i3++) {
            ಸ್ಥಿರಾಂಕ int64_t i03       = i3 / r3;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src0_ptr3 = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) params->src0.ದತ್ತಾಂಶ + i03 * nb03;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src1_ptr3 = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) params->src1.ದತ್ತಾಂಶ + i3 * nb13;
            ಅಕ್ಷರ *        dst_ptr3  = (ಅಕ್ಷರ *) params->dst.ದತ್ತಾಂಶ + i3 * nbd3;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  bias_ptr3 = bias_ತಳ ? bias_ತಳ + i3 * nbb3 : (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) 0;

            ಗಾಗಿ (int64_t i2 = 0; i2 < ne12; i2++) {
                ಸ್ಥಿರಾಂಕ int64_t i02       = i2 / r2;
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                ಅಕ್ಷರ *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) 0;

                ಗಾಗಿ (int64_t n = 0; n < N; n++) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_ಲಂಬಸಾಲು_ತಳ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (src1_ptr2 + n * nb11);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * bias_n     = bias_ptr2 ? (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (bias_ptr2 + n * nbb1) : (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) 0;

                    ಗಾಗಿ (int64_t m = minion_id; m < M; m += STRIDE_M_KSPLIT) {
                        ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * q_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) (src0_ptr2 + m * nb01);

                        ದಶಮಾಂಶ partial = compute_ಅಡ್ಡಸಾಲು_dot_q8_0(q_ಅಡ್ಡಸಾಲು + k_ಆರಂಭ, b_ಲಂಬಸಾಲು_ತಳ + k_ಆರಂಭ * 32, k_len);

                        ಆದರೆ (is_hart1) {
                            *l2scp_ಸ್ಥಾನ = partial;
                            FENCE;
                            flush_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) l2scp_ಸ್ಥಾನ, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            et_sem_wait(ET_BARRIER_MINION);
                            ದಶಮಾಂಶ other = *l2scp_ಸ್ಥಾನ;
                            et_sem_post(ET_BARRIER_MINION);

                            ದಶಮಾಂಶ * dst_ನಮೂದು = (ದಶಮಾಂಶ *) (dst_ptr2 + n * nbd1 + m * ಗಾತ್ರ(ದಶಮಾಂಶ));
                            ದಶಮಾಂಶ   ಮೊತ್ತ       = partial + other;
                            ಆದರೆ (bias_n) {
                                ಮೊತ್ತ += bias_n[m];
                            }
                            atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) dst_ನಮೂದು, ಮೊತ್ತ);
                        }
                    }
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (use_ksplit_ಗುಂಪು) {
        /*
         * Grouped K-split for the 5-8 rows/minion regime.
         *
         * Both harts process the same 4-row group, each on half of K, and
         * exchange 4 partial sums once per group instead of once per row.
         * This keeps the K-split bandwidth benefit while cutting semaphore
         * traffic by 4x relative to the old per-row exchange.
         */
        ಸ್ಥಿರಾಂಕ int64_t    k_ಆರಂಭ    = is_hart1 ? k_half : 0;
        ಸ್ಥಿರಾಂಕ int64_t    k_len      = is_hart1 ? (K_ಖಂಡಗಳು - k_half) : k_half;
        ಅಸ್ಥಿರ ದಶಮಾಂಶ * l2scp_ಸ್ಥಾನ = (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) et_shire_l2scp_local(local_minion * 64);

        ಗಾಗಿ (int64_t i3 = 0; i3 < ne13; i3++) {
            ಸ್ಥಿರಾಂಕ int64_t i03       = i3 / r3;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src0_ptr3 = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) params->src0.ದತ್ತಾಂಶ + i03 * nb03;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src1_ptr3 = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) params->src1.ದತ್ತಾಂಶ + i3 * nb13;
            ಅಕ್ಷರ *        dst_ptr3  = (ಅಕ್ಷರ *) params->dst.ದತ್ತಾಂಶ + i3 * nbd3;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  bias_ptr3 = bias_ತಳ ? bias_ತಳ + i3 * nbb3 : (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) 0;

            ಗಾಗಿ (int64_t i2 = 0; i2 < ne12; i2++) {
                ಸ್ಥಿರಾಂಕ int64_t i02       = i2 / r2;
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                ಅಕ್ಷರ *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) 0;

                ಗಾಗಿ (int64_t n = 0; n < N; n++) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_ಲಂಬಸಾಲು_ತಳ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (src1_ptr2 + n * nb11);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * bias_n     = bias_ptr2 ? (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (bias_ptr2 + n * nbb1) : (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) 0;

                    ಗಾಗಿ (int64_t m_ತಳ = minion_id; m_ತಳ < M; m_ತಳ += STRIDE_M_KSPLIT * KSPLIT_ಗುಂಪು_ಅಡ್ಡಸಾಲುಗಳು) {
                        ಸ್ಥಿರಾಂಕ int64_t m0 = m_ತಳ;
                        ಸ್ಥಿರಾಂಕ int64_t m1 = m0 + STRIDE_M_KSPLIT;
                        ಸ್ಥಿರಾಂಕ int64_t m2 = m1 + STRIDE_M_KSPLIT;
                        ಸ್ಥಿರಾಂಕ int64_t m3 = m2 + STRIDE_M_KSPLIT;

                        ದಶಮಾಂಶ s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        ಗಾಗಿ (int64_t kb = 0; kb < K_ಖಂಡಗಳು; kb += TILE_KB) {
                            int64_t tile_len = k_len - kb;
                            ಆದರೆ (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            ಆದರೆ (tile_len <= 0) {
                                ನಿಲ್ಲಿಸು;
                            }
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_tile = b_ಲಂಬಸಾಲು_ತಳ + (k_ಆರಂಭ + kb) * 32;
                            ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_kb = k_ಆರಂಭ + kb;

                            ಆದರೆ (m0 < M) {
                                s0 += compute_ಅಡ್ಡಸಾಲು_dot_q8_0((ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) (src0_ptr2 + m0 * nb01) + ಅಡ್ಡಸಾಲು_kb,
                                                           b_tile, tile_len);
                            }
                            ಆದರೆ (m1 < M) {
                                s1 += compute_ಅಡ್ಡಸಾಲು_dot_q8_0((ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) (src0_ptr2 + m1 * nb01) + ಅಡ್ಡಸಾಲು_kb,
                                                           b_tile, tile_len);
                            }
                            ಆದರೆ (m2 < M) {
                                s2 += compute_ಅಡ್ಡಸಾಲು_dot_q8_0((ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) (src0_ptr2 + m2 * nb01) + ಅಡ್ಡಸಾಲು_kb,
                                                           b_tile, tile_len);
                            }
                            ಆದರೆ (m3 < M) {
                                s3 += compute_ಅಡ್ಡಸಾಲು_dot_q8_0((ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) (src0_ptr2 + m3 * nb01) + ಅಡ್ಡಸಾಲು_kb,
                                                           b_tile, tile_len);
                            }
                        }

                        ಆದರೆ (is_hart1) {
                            l2scp_ಸ್ಥಾನ[0] = s0;
                            l2scp_ಸ್ಥಾನ[1] = s1;
                            l2scp_ಸ್ಥಾನ[2] = s2;
                            l2scp_ಸ್ಥಾನ[3] = s3;
                            FENCE;
                            flush_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) l2scp_ಸ್ಥಾನ, 1, 64);
                            WAIT_CACHEOPS;
                            et_sem_post(ET_BARRIER_MINION);
                            et_sem_wait(ET_BARRIER_MINION);
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            et_sem_wait(ET_BARRIER_MINION);
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ p0 = l2scp_ಸ್ಥಾನ[0];
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ p1 = l2scp_ಸ್ಥಾನ[1];
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ p2 = l2scp_ಸ್ಥಾನ[2];
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ p3 = l2scp_ಸ್ಥಾನ[3];
                            et_sem_post(ET_BARRIER_MINION);

                            ದಶಮಾಂಶ *     c_ತಳ = (ದಶಮಾಂಶ *) (dst_ptr2 + n * nbd1);
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b0     = bias_n ? bias_n[m0] : 0.0f;
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b1     = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b2     = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b3     = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                            ಆದರೆ (m0 < M) {
                                atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) (c_ತಳ + m0), s0 + p0 + b0);
                            }
                            ಆದರೆ (m1 < M) {
                                atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) (c_ತಳ + m1), s1 + p1 + b1);
                            }
                            ಆದರೆ (m2 < M) {
                                atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) (c_ತಳ + m2), s2 + p2 + b2);
                            }
                            ಆದರೆ (m3 < M) {
                                atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) (c_ತಳ + m3), s3 + p3 + b3);
                            }
                        }
                    }
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (K_ಖಂಡಗಳು > TILE_KB) {
        /*
         * Tile-outer with scalar row groups: process up to 4 rows per
         * hart sharing each B tile before advancing to the next tile.
         * Uses scalar float variables (not an array) to accumulate across
         * tiles — avoids the flw/fadd.s/fsw stack ops that corrupt vector
         * register state on ET-SoC-1's MMX-style shared FP file.
         */
        ಗಾಗಿ (int64_t i3 = 0; i3 < ne13; i3++) {
            ಸ್ಥಿರಾಂಕ int64_t i03       = i3 / r3;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src0_ptr3 = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) params->src0.ದತ್ತಾಂಶ + i03 * nb03;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src1_ptr3 = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) params->src1.ದತ್ತಾಂಶ + i3 * nb13;
            ಅಕ್ಷರ *        dst_ptr3  = (ಅಕ್ಷರ *) params->dst.ದತ್ತಾಂಶ + i3 * nbd3;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  bias_ptr3 = bias_ತಳ ? bias_ತಳ + i3 * nbb3 : (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) 0;

            ಗಾಗಿ (int64_t i2 = 0; i2 < ne12; i2++) {
                ಸ್ಥಿರಾಂಕ int64_t i02       = i2 / r2;
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                ಅಕ್ಷರ *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) 0;

                ಗಾಗಿ (int64_t n = 0; n < N; n++) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_ಲಂಬಸಾಲು_ತಳ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (src1_ptr2 + n * nb11);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * bias_n     = bias_ptr2 ? (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (bias_ptr2 + n * nbb1) : (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) 0;

                    ಗಾಗಿ (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * 4) {
                        ಸ್ಥಿರಾಂಕ int64_t m1 = m0 + STRIDE_M;
                        ಸ್ಥಿರಾಂಕ int64_t m2 = m0 + STRIDE_M * 2;
                        ಸ್ಥಿರಾಂಕ int64_t m3 = m0 + STRIDE_M * 3;

                        ದಶಮಾಂಶ s0 = 0.0f, s1 = 0.0f, s2 = 0.0f, s3 = 0.0f;

                        ಗಾಗಿ (int64_t kb = 0; kb < K_ಖಂಡಗಳು; kb += TILE_KB) {
                            int64_t tile_len = K_ಖಂಡಗಳು - kb;
                            ಆದರೆ (tile_len > TILE_KB) {
                                tile_len = TILE_KB;
                            }
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_tile = b_ಲಂಬಸಾಲು_ತಳ + kb * 32;

                            s0 += compute_ಅಡ್ಡಸಾಲು_dot_q8_0((ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) (src0_ptr2 + m0 * nb01) + kb, b_tile,
                                                       tile_len);
                            ಆದರೆ (m1 < M) {
                                s1 += compute_ಅಡ್ಡಸಾಲು_dot_q8_0((ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) (src0_ptr2 + m1 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            ಆದರೆ (m2 < M) {
                                s2 += compute_ಅಡ್ಡಸಾಲು_dot_q8_0((ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) (src0_ptr2 + m2 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                            ಆದರೆ (m3 < M) {
                                s3 += compute_ಅಡ್ಡಸಾಲು_dot_q8_0((ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) (src0_ptr2 + m3 * nb01) + kb, b_tile,
                                                           tile_len);
                            }
                        }

                        ದಶಮಾಂಶ *     dst_ತಳ = (ದಶಮಾಂಶ *) (dst_ptr2 + n * nbd1);
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b0       = bias_n ? bias_n[m0] : 0.0f;
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b1       = (bias_n && m1 < M) ? bias_n[m1] : 0.0f;
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b2       = (bias_n && m2 < M) ? bias_n[m2] : 0.0f;
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ b3       = (bias_n && m3 < M) ? bias_n[m3] : 0.0f;
                        atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) (dst_ತಳ + m0), s0 + b0);
                        ಆದರೆ (m1 < M) {
                            atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) (dst_ತಳ + m1), s1 + b1);
                        }
                        ಆದರೆ (m2 < M) {
                            atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) (dst_ತಳ + m2), s2 + b2);
                        }
                        ಆದರೆ (m3 < M) {
                            atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) (dst_ತಳ + m3), s3 + b3);
                        }
                    }
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        /*
         * Simple path for small K.
         *
         * When `nb01` is 32-byte aligned, every row has the same block-alignment
         * pattern. That lets us compute two rows together and reuse each loaded
         * B chunk across both rows instead of reloading it in a second dot call.
         */
        ಗಾಗಿ (int64_t i3 = 0; i3 < ne13; i3++) {
            ಸ್ಥಿರಾಂಕ int64_t i03       = i3 / r3;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src0_ptr3 = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) params->src0.ದತ್ತಾಂಶ + i03 * nb03;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src1_ptr3 = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) params->src1.ದತ್ತಾಂಶ + i3 * nb13;
            ಅಕ್ಷರ *        dst_ptr3  = (ಅಕ್ಷರ *) params->dst.ದತ್ತಾಂಶ + i3 * nbd3;
            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  bias_ptr3 = bias_ತಳ ? bias_ತಳ + i3 * nbb3 : (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) 0;

            ಗಾಗಿ (int64_t i2 = 0; i2 < ne12; i2++) {
                ಸ್ಥಿರಾಂಕ int64_t i02       = i2 / r2;
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src0_ptr2 = src0_ptr3 + i02 * nb02;
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src1_ptr2 = src1_ptr3 + i2 * nb12;
                ಅಕ್ಷರ *        dst_ptr2  = dst_ptr3 + i2 * nbd2;
                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  bias_ptr2 = bias_ptr3 ? bias_ptr3 + i2 * nbb2 : (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) 0;

                ಗಾಗಿ (int64_t n = 0; n < N; n++) {
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * b_ಲಂಬಸಾಲು_ತಳ = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (src1_ptr2 + n * nb11);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * bias_n     = bias_ptr2 ? (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (bias_ptr2 + n * nbb1) : (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) 0;
                    q8_dot_state  q8_state;
                    q8_dot_begin(&q8_state);

                    ಆದರೆ (use_simple_x2) {
                        ಗಾಗಿ (int64_t m0 = hart_id; m0 < M; m0 += STRIDE_M * SIMPLE_X2_ಅಡ್ಡಸಾಲುಗಳು) {
                            ಸ್ಥಿರಾಂಕ int64_t      m1     = m0 + STRIDE_M;
                            ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * q_row0 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) (src0_ptr2 + m0 * nb01);

                            ಆದರೆ (m1 < M) {
                                ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * q_row1 = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) (src0_ptr2 + m1 * nb01);
                                ದಶಮಾಂಶ              s0, s1;
                                q8_dot_compute_x2_aligned(q_row0, q_row1, b_ಲಂಬಸಾಲು_ತಳ, K_ಖಂಡಗಳು, &s0, &s1);

                                ದಶಮಾಂಶ * dst0 = (ದಶಮಾಂಶ *) (dst_ptr2 + n * nbd1 + m0 * ಗಾತ್ರ(ದಶಮಾಂಶ));
                                ದಶಮಾಂಶ * dst1 = (ದಶಮಾಂಶ *) (dst_ptr2 + n * nbd1 + m1 * ಗಾತ್ರ(ದಶಮಾಂಶ));
                                ಆದರೆ (bias_n) {
                                    s0 += bias_n[m0];
                                    s1 += bias_n[m1];
                                }
                                atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) dst0, s0);
                                atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) dst1, s1);
                            } ಇಲ್ಲದಿದ್ದರೆ {
                                ದಶಮಾಂಶ   ಮೊತ್ತ = q8_dot_compute(q_row0, b_ಲಂಬಸಾಲು_ತಳ, K_ಖಂಡಗಳು);
                                ದಶಮಾಂಶ * dst = (ದಶಮಾಂಶ *) (dst_ptr2 + n * nbd1 + m0 * ಗಾತ್ರ(ದಶಮಾಂಶ));
                                ಆದರೆ (bias_n) {
                                    ಮೊತ್ತ += bias_n[m0];
                                }
                                atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) dst, ಮೊತ್ತ);
                            }
                        }
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        ಗಾಗಿ (int64_t m = hart_id; m < M; m += STRIDE_M) {
                            ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * q_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) (src0_ptr2 + m * nb01);

                            ದಶಮಾಂಶ ಮೊತ್ತ = q8_dot_compute(q_ಅಡ್ಡಸಾಲು, b_ಲಂಬಸಾಲು_ತಳ, K_ಖಂಡಗಳು);

                            ದಶಮಾಂಶ * dst_ನಮೂದು = (ದಶಮಾಂಶ *) (dst_ptr2 + n * nbd1 + m * ಗಾತ್ರ(ದಶಮಾಂಶ));
                            ಆದರೆ (bias_n) {
                                ಮೊತ್ತ += bias_n[m];
                            }
                            atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) dst_ನಮೂದು, ಮೊತ್ತ);
                        }
                    }

                    q8_dot_ಕೊನೆ(&q8_state);
                }
            }
        }
    }

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ET_UBERKERNEL
    FENCE;
    evict_region_past_l2(params->dst.ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು(&params->dst));
    WAIT_CACHEOPS;
    FENCE;
#ಆದರೆ_ಅಂತ್ಯ
    ಹಿಂತಿರುಗಿ 0;
}
