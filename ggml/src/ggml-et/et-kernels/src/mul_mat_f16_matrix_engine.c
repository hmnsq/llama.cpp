#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"
#ಅನ್ತರ್ಭಾವ "tensor.h"

#ಅನ್ತರ್ಭಾವ <etsoc/common/utils.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

// FP16 x FP16 -> FP32 MUL_MAT with hart 1 B-panel packing
//
// Hart 0: tensor engine (load A, load B from SCP, FMA, reduce, store)
// Hart 1: pack B into double-buffered L2 SCP panels, flush for tensor_load
//
// Sync: monotonic counters in L2 SCP with evict-based coherency.
// Double-buffered bpanel allows pack/FMA overlap.
//
#ವ್ಯಾಖ್ಯಾನಿಸು ಅಂಕೆ_COMPUTE_SHIRES 32
#ವ್ಯಾಖ್ಯಾನಿಸು MINIONS_ಪ್ರತಿ_SHIRE  32

#ವ್ಯಾಖ್ಯಾನಿಸು TILE_M 16
#ವ್ಯಾಖ್ಯಾನಿಸು TILE_N 16
#ವ್ಯಾಖ್ಯಾನಿಸು TILE_K 32

#ವ್ಯಾಖ್ಯಾನಿಸು CACHEOP_MAX 0
#ವ್ಯಾಖ್ಯಾನಿಸು REP_ದರ    0

#ವ್ಯಾಖ್ಯಾನಿಸು A_L1_ಆರಂಭ 0   // SCP lines  0..15 for A
#ವ್ಯಾಖ್ಯಾನಿಸು B_L1_ಆರಂಭ 16  // SCP lines 16..31 for B

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uint16_t et_fp16_t;

// L2 SCP layout per minion (double-buffered bpanel + sync counters)
//   [0..1023]    bpanel buffer 0 (16 lines x 64 bytes)
//   [1024..2047] bpanel buffer 1
//   [2048..2111] ready counter   (hart1 -> hart0, own cache line)
//   [2112..2175] consumed counter (hart0 -> hart1, own cache line)
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_BPANEL_ಗಾತ್ರ  (16 * 32 * ಗಾತ್ರ(et_fp16_t))  // 1024 bytes
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_READY_OFF    (2 * SCP_BPANEL_ಗಾತ್ರ)          // 2048
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_CONSUMED_OFF (SCP_READY_OFF + 64)           // 2112
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_ಪ್ರತಿ_MINION   (SCP_CONSUMED_OFF + 64)        // 2176

// Signal a counter value to the other hart via L2 SCP.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __attribute__((always_inline)) scp_signal(ಅಸ್ಥಿರ uint32_t * flag, uint32_t ಮೌಲ್ಯ) {
    *flag = ಮೌಲ್ಯ;
    FENCE;
    evict_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) flag, 1, 64);
    WAIT_CACHEOPS;
}

// Wait for a counter in L2 SCP to reach the expected value.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __attribute__((always_inline)) scp_wait(ಅಸ್ಥಿರ uint32_t * flag, uint32_t expected) {
    ಆಗ (1) {
        evict_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) flag, 1, 64);
        WAIT_CACHEOPS;
        ಆದರೆ (*flag >= expected) {
            ಹಿಂತಿರುಗಿ;
        }
    }
}

/**
 * Build the interleaved B panel that TensorFMA16A32 expects (vectorized).
 *
 * Output: 16 lines x 32 fp16 = 1024 bytes, 64-byte aligned.
 *   out[l][j*2+0] = src0[mb + j][kb + 2*l]
 *   out[l][j*2+1] = src0[mb + j][kb + 2*l + 1]
 *
 * Uses fsch.ps scatter store: load 8 pairs per row, scatter to 8 output lines.
 */
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __attribute__((always_inline)) pack_b_interleaved(et_fp16_t *  ಹೊರಹೋಗು,
                                                                     ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_batch,
                                                                     int64_t      mb,
                                                                     int64_t      kb,
                                                                     int64_t      nb1_0) {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ int32_t __attribute__((aligned(32))) scatter_idx[8] = { 0, 64, 128, 192, 256, 320, 384, 448 };

    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ old_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "flw.ps    f1, 0(%[idx])    \n\t"
        : [ms] "=&r"(old_ಮುಖವಾಡ)
        : [idx] "r"(scatter_idx)
        : "f1");

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < TILE_M; ++j) {
        ಸ್ಥಿರಾಂಕ et_fp16_t * ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ et_fp16_t *) (src0_batch + (mb + j) * nb1_0) + kb;
        ಅಕ್ಷರ *            dst = (ಅಕ್ಷರ *) ಹೊರಹೋಗು + j * 4;

        __asm__ ಅಸ್ಥಿರ(
            "flw.ps    f2, 0(%[src])    \n\t"
            "flw.ps    f3, 32(%[src])   \n\t"
            "fscw.ps   f2, f1(%[d0])    \n\t"
            "fscw.ps   f3, f1(%[d1])    \n\t"
            :
            : [src] "r"(ಅಡ್ಡಸಾಲು), [d0] "r"(dst), [d1] "r"(dst + 512)
            : "f2", "f3", "memory");
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_ಮುಖವಾಡ));
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_ದ್ವಿಮಾನ_params * params, ಶೂನ್ಯ * env) {
    (ಶೂನ್ಯ) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    ಆದರೆ (shire_id >= ಅಂಕೆ_COMPUTE_SHIRES) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    // Dimensions (both harts need these for tile assignment)
    ಸ್ಥಿರಾಂಕ int64_t K = params->src0.ne[0];
    ಸ್ಥಿರಾಂಕ int64_t M = params->src0.ne[1];
    ಸ್ಥಿರಾಂಕ int64_t N = params->src1.ne[1];

    ಸ್ಥಿರಾಂಕ int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    ಸ್ಥಿರಾಂಕ int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    ಸ್ಥಿರಾಂಕ int64_t nb1_0 = params->src0.nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];

    ಸ್ಥಿರಾಂಕ int64_t nb1_1 = params->src1.nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];

    ಸ್ಥಿರಾಂಕ int64_t nb1_d = params->dst.nb[1];
    ಸ್ಥಿರಾಂಕ int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) params->src0.ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src1_ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) params->src1.ದತ್ತಾಂಶ;
    ಅಕ್ಷರ *       dst_ತಳ  = (ಅಕ್ಷರ *) params->dst.ದತ್ತಾಂಶ;

    ಆದರೆ ((M % TILE_M) != 0) {
        ಹಿಂತಿರುಗಿ 0;
    }
    ಆದರೆ ((K % TILE_K) != 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ int64_t m_tiles     = M / TILE_M;
    ಸ್ಥಿರಾಂಕ int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    ಸ್ಥಿರಾಂಕ int64_t batch_ಎಣಿಕೆ = ne2_1 * ne3_1;
    ಸ್ಥಿರಾಂಕ int64_t ತಳ_tiles  = m_tiles * n_tiles * batch_ಎಣಿಕೆ;

    ಸ್ಥಿರಾಂಕ int64_t r2 = ne2_1 / ne2_0;
    ಸ್ಥಿರಾಂಕ int64_t r3 = ne3_1 / ne3_0;

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_harts = ಅಂಕೆ_COMPUTE_SHIRES * MINIONS_ಪ್ರತಿ_SHIRE;
    ಸ್ಥಿರಾಂಕ int64_t k_ಹೆಜ್ಜೆಗಳು     = K / TILE_K;

    int64_t k_splits = 1;
    ಆದರೆ (ತಳ_tiles < ಒಟ್ಟು_harts) {
        k_splits   = (ಒಟ್ಟು_harts + ತಳ_tiles - 1) / ತಳ_tiles;
        int64_t ks = 1;
        ಆಗ (ks * 2 <= k_splits && ks * 2 <= 32 && k_ಹೆಜ್ಜೆಗಳು % (ks * 2) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ಸ್ಥಿರಾಂಕ int64_t tiles_ಪ್ರತಿ_shire = MINIONS_ಪ್ರತಿ_SHIRE / k_splits;
    ಸ್ಥಿರಾಂಕ int64_t k_ವಿಭಜಿಸು         = local_minion % k_splits;
    ಸ್ಥಿರಾಂಕ int64_t local_tile_idx  = local_minion / k_splits;
    ಸ್ಥಿರಾಂಕ int64_t tiles_stride    = (int64_t) ಅಂಕೆ_COMPUTE_SHIRES * tiles_ಪ್ರತಿ_shire;

    ಸ್ಥಿರಾಂಕ int64_t k_ಹೆಜ್ಜೆಗಳು_ಪ್ರತಿ_ವಿಭಜಿಸು = k_ಹೆಜ್ಜೆಗಳು / k_splits;
    ಸ್ಥಿರಾಂಕ int64_t k_ಆರಂಭ           = k_ವಿಭಜಿಸು * k_ಹೆಜ್ಜೆಗಳು_ಪ್ರತಿ_ವಿಭಜಿಸು * TILE_K;
    ಸ್ಥಿರಾಂಕ int64_t k_ಕೊನೆ             = k_ಆರಂಭ + k_ಹೆಜ್ಜೆಗಳು_ಪ್ರತಿ_ವಿಭಜಿಸು * TILE_K;

    // L2 SCP pointers for this minion's double-buffered panels + sync
    uint64_t    scp_ತಳ  = local_minion * SCP_ಪ್ರತಿ_MINION;
    et_fp16_t * scp_bp[2] = {
        (et_fp16_t *) et_shire_l2scp_local(scp_ತಳ),
        (et_fp16_t *) et_shire_l2scp_local(scp_ತಳ + SCP_BPANEL_ಗಾತ್ರ),
    };
    ಅಸ್ಥಿರ uint32_t * ready_ctr    = (ಅಸ್ಥಿರ uint32_t *) et_shire_l2scp_local(scp_ತಳ + SCP_READY_OFF);
    ಅಸ್ಥಿರ uint32_t * consumed_ctr = (ಅಸ್ಥಿರ uint32_t *) et_shire_l2scp_local(scp_ತಳ + SCP_CONSUMED_OFF);

    // ================================================================
    // Hart 1: B-panel packer
    // ================================================================
    ಆದರೆ (is_hart1) {
        // Initialize sync counters
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);

        uint32_t chunk_id = 0;

        ಗಾಗಿ (int64_t tile = (int64_t) shire_id + local_tile_idx * ಅಂಕೆ_COMPUTE_SHIRES; tile < ತಳ_tiles;
             tile += tiles_stride) {
            ಸ್ಥಿರಾಂಕ int64_t tiles_ಪ್ರತಿ_batch = m_tiles * n_tiles;
            ಸ್ಥಿರಾಂಕ int64_t batch_idx       = tile / tiles_ಪ್ರತಿ_batch;
            ಸ್ಥಿರಾಂಕ int64_t tile_ಒಳ_batch   = tile % tiles_ಪ್ರತಿ_batch;

            ಸ್ಥಿರಾಂಕ int64_t mb_idx = tile_ಒಳ_batch % m_tiles;

            ಸ್ಥಿರಾಂಕ int64_t i3   = batch_idx / ne2_1;
            ಸ್ಥಿರಾಂಕ int64_t i2   = batch_idx % ne2_1;
            ಸ್ಥಿರಾಂಕ int64_t i2_0 = i2 / r2;
            ಸ್ಥಿರಾಂಕ int64_t i3_0 = i3 / r3;

            ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *  src0_batch = src0_ತಳ + i3_0 * nb3_0 + i2_0 * nb2_0;
            ಸ್ಥಿರಾಂಕ int64_t mb         = mb_idx * TILE_M;

            ಗಾಗಿ (int64_t kb = k_ಆರಂಭ; kb < k_ಕೊನೆ; kb += TILE_K) {
                ಪೂರ್ಣಾಂಕ buf = chunk_id & 1;

                // Back-pressure: wait for hart 0 to finish with this buffer
                ಆದರೆ (chunk_id >= 2) {
                    scp_wait(consumed_ctr, chunk_id - 1);
                }

                pack_b_interleaved(scp_bp[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_ಗಮ್ಯ_l2(scp_bp[buf], 16, 64);
                WAIT_CACHEOPS;

                chunk_id++;
                scp_signal(ready_ctr, chunk_id);
            }
        }

        FENCE;
        ಹಿಂತಿರುಗಿ 0;
    }

    // ================================================================
    // Hart 0: tensor engine compute
    // ================================================================
    uint64_t       my_minion_id      = get_minion_id();
    ಸ್ಥಿರಾಂಕ uint64_t ಗುಂಪು_ತಳ_global = my_minion_id - k_ವಿಭಜಿಸು;

    ಸಿದ್ಧತೆ_cache_scp();
#ಆದರೆ CACHEOP_MAX > 0 || REP_ದರ > 0
    ucache_control(1, REP_ದರ, CACHEOP_MAX);
#ಆದರೆ_ಅಂತ್ಯ
    CLEAR_TENSOR_ERROR;

    // Evict any stale L1D copies of sync counters
    evict_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) ready_ctr, 1, 64);
    WAIT_CACHEOPS;
    evict_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) consumed_ctr, 1, 64);
    WAIT_CACHEOPS;

    uint32_t chunk_id = 0;

    ಗಾಗಿ (int64_t tile = (int64_t) shire_id + local_tile_idx * ಅಂಕೆ_COMPUTE_SHIRES; tile < ತಳ_tiles;
         tile += tiles_stride) {
        ಸ್ಥಿರಾಂಕ int64_t tiles_ಪ್ರತಿ_batch = m_tiles * n_tiles;
        ಸ್ಥಿರಾಂಕ int64_t batch_idx       = tile / tiles_ಪ್ರತಿ_batch;
        ಸ್ಥಿರಾಂಕ int64_t tile_ಒಳ_batch   = tile % tiles_ಪ್ರತಿ_batch;

        ಸ್ಥಿರಾಂಕ int64_t nb_idx = tile_ಒಳ_batch / m_tiles;
        ಸ್ಥಿರಾಂಕ int64_t mb_idx = tile_ಒಳ_batch % m_tiles;

        ಸ್ಥಿರಾಂಕ int64_t i3 = batch_idx / ne2_1;
        ಸ್ಥಿರಾಂಕ int64_t i2 = batch_idx % ne2_1;

        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src1_batch = src1_ತಳ + i3 * nb3_1 + i2 * nb2_1;
        ಅಕ್ಷರ *       dst_batch  = dst_ತಳ + i3 * nb3_d + i2 * nb2_d;

        ಸ್ಥಿರಾಂಕ int64_t mb    = mb_idx * TILE_M;
        ಸ್ಥಿರಾಂಕ int64_t nb    = nb_idx * TILE_N;
        ಸ್ಥಿರಾಂಕ int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        // Set tensor_mask for partial N tiles
        ಆದರೆ (n_cur < TILE_N) {
            uint64_t ಮುಖವಾಡ = (1ULL << n_cur) - 1;
            __asm__ __volatile__("csrw 0x805, %0" : : "r"(ಮುಖವಾಡ));
        }

        ಗಾಗಿ (int64_t kb = k_ಆರಂಭ; kb < k_ಕೊನೆ; kb += TILE_K) {
            ಪೂರ್ಣಾಂಕ buf = chunk_id & 1;

            // Start loading A from DRAM (overlaps with waiting for hart 1)
            tensor_ತುಂಬು((n_cur < TILE_N), false, A_L1_ಆರಂಭ, TENSOR_ತುಂಬು_PLAIN, 0,
                        (uint64_t) (src1_batch + nb * nb1_1 + kb * (int64_t) ಗಾತ್ರ(et_fp16_t)), 0, n_cur - 1,
                        (uint64_t) nb1_1, 0);

            // Wait for hart 1 to finish packing this chunk
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            // Load B from L2 SCP (hart 1 already flushed it)
            tensor_ತುಂಬು(false, false, B_L1_ಆರಂಭ, TENSOR_ತುಂಬು_PLAIN, 0, (uint64_t) scp_bp[buf], 0, 15, 64, 1);

            tensor_wait(TENSOR_ತುಂಬು_WAIT_0);
            tensor_wait(TENSOR_ತುಂಬು_WAIT_1);

            // TensorFMA16A32
            tensor_fma((n_cur < TILE_N), 3, n_cur - 1, 15, 0, false, false, false, false, B_L1_ಆರಂಭ, A_L1_ಆರಂಭ,
                       TENSOR_FMA_OP_FP16, (kb == k_ಆರಂಭ));

            tensor_wait(TENSOR_FMA_WAIT);

            // Signal that this buffer is free for hart 1 to reuse
            scp_signal(consumed_ctr, chunk_id);
        }

        // K-split ring reduce
        ಆದರೆ (k_splits > 1) {
            ಸ್ಥಿರಾಂಕ uint64_t ಅಂಕೆ_regs = (uint64_t) n_cur * 2;

            ಆದರೆ (k_ವಿಭಜಿಸು > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, ಅಂಕೆ_regs, ಗುಂಪು_ತಳ_global + k_ವಿಭಜಿಸು - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }

            ಆದರೆ (k_ವಿಭಜಿಸು < k_splits - 1) {
                tensor_reduce_send(0, ಅಂಕೆ_regs, ಗುಂಪು_ತಳ_global + k_ವಿಭಜಿಸು + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        // Store FP32 result tile
        ಆದರೆ (k_ವಿಭಜಿಸು == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    ಹಿಂತಿರುಗಿ 0;
}
