#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "tensor.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <etsoc/common/utils.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

// FP16 x FP16 -> FP32 MUL_MAT with hart 1 B-panel packing
//
// Hart 0: tensor engine (load A, load B from SCP, FMA, reduce, store)
// Hart 1: pack B into double-buffered L2 SCP panels, flush for tensor_load
//
// Sync: monotonic counters in L2 SCP with evict-based coherency.
// Double-buffered bpanel allows pack/FMA overlap.
//
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅੰਕ_COMPUTE_SHIRES 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MINIONS_ਪ੍ਰਤੀ_SHIRE  32

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE_M 16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE_N 16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE_K 32

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CACHEOP_MAX 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ REP_ਦਰ    0

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ A_L1_ਸ਼ੁਰੂ 0   // SCP lines  0..15 for A
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ B_L1_ਸ਼ੁਰੂ 16  // SCP lines 16..31 for B

ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint16_t et_fp16_t;

// L2 SCP layout per minion (double-buffered bpanel + sync counters)
//   [0..1023]    bpanel buffer 0 (16 lines x 64 bytes)
//   [1024..2047] bpanel buffer 1
//   [2048..2111] ready counter   (hart1 -> hart0, own cache line)
//   [2112..2175] consumed counter (hart0 -> hart1, own cache line)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_BPANEL_ਮਾਪ  (16 * 32 * ਆਕਾਰ(et_fp16_t))  // 1024 bytes
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_READY_OFF    (2 * SCP_BPANEL_ਮਾਪ)          // 2048
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_CONSUMED_OFF (SCP_READY_OFF + 64)           // 2112
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_ਪ੍ਰਤੀ_MINION   (SCP_CONSUMED_OFF + 64)        // 2176

// Signal a counter value to the other hart via L2 SCP.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) scp_signal(ਅਸਥਿਰ uint32_t * flag, uint32_t ਮੁੱਲ) {
    *flag = ਮੁੱਲ;
    FENCE;
    evict_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) flag, 1, 64);
    WAIT_CACHEOPS;
}

// Wait for a counter in L2 SCP to reach the expected value.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) scp_wait(ਅਸਥਿਰ uint32_t * flag, uint32_t expected) {
    ਜਦੋਂ (1) {
        evict_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) flag, 1, 64);
        WAIT_CACHEOPS;
        ਜੇ (*flag >= expected) {
            ਵਾਪਸ;
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
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) pack_b_interleaved(et_fp16_t *  ਬਾਹਰਲਾ,
                                                                     ਅਚਲ ਅੱਖਰ * src0_batch,
                                                                     int64_t      mb,
                                                                     int64_t      kb,
                                                                     int64_t      nb1_0) {
    ਸਥਿਰ ਅਚਲ int32_t __attribute__((aligned(32))) scatter_idx[8] = { 0, 64, 128, 192, 256, 320, 384, 448 };

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ old_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "flw.ps    f1, 0(%[idx])    \n\t"
        : [ms] "=&r"(old_ਨਕਾਬ)
        : [idx] "r"(scatter_idx)
        : "f1");

    ਲਈ (ਪੂਰਨ j = 0; j < TILE_M; ++j) {
        ਅਚਲ et_fp16_t * ਪੰਕਤੀ = (ਅਚਲ et_fp16_t *) (src0_batch + (mb + j) * nb1_0) + kb;
        ਅੱਖਰ *            dst = (ਅੱਖਰ *) ਬਾਹਰਲਾ + j * 4;

        __asm__ ਅਸਥਿਰ(
            "flw.ps    f2, 0(%[src])    \n\t"
            "flw.ps    f3, 32(%[src])   \n\t"
            "fscw.ps   f2, f1(%[d0])    \n\t"
            "fscw.ps   f3, f1(%[d1])    \n\t"
            :
            : [src] "r"(ਪੰਕਤੀ), [d0] "r"(dst), [d1] "r"(dst + 512)
            : "f2", "f3", "memory");
    }

    __asm__ ਅਸਥਿਰ("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_ਨਕਾਬ));
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_ਦੁਆਧਾਰੀ_params * params, ਖਾਲੀ * env) {
    (ਖਾਲੀ) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    ਜੇ (shire_id >= ਅੰਕ_COMPUTE_SHIRES) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ ਪੂਰਨ is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    // Dimensions (both harts need these for tile assignment)
    ਅਚਲ int64_t K = params->src0.ne[0];
    ਅਚਲ int64_t M = params->src0.ne[1];
    ਅਚਲ int64_t N = params->src1.ne[1];

    ਅਚਲ int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    ਅਚਲ int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    ਅਚਲ int64_t nb1_0 = params->src0.nb[1];
    ਅਚਲ int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];

    ਅਚਲ int64_t nb1_1 = params->src1.nb[1];
    ਅਚਲ int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];

    ਅਚਲ int64_t nb1_d = params->dst.nb[1];
    ਅਚਲ int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    ਅਚਲ ਅੱਖਰ * src0_ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) params->src0.ਅੰਕੜੇ;
    ਅਚਲ ਅੱਖਰ * src1_ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) params->src1.ਅੰਕੜੇ;
    ਅੱਖਰ *       dst_ਅਧਾਰ  = (ਅੱਖਰ *) params->dst.ਅੰਕੜੇ;

    ਜੇ ((M % TILE_M) != 0) {
        ਵਾਪਸ 0;
    }
    ਜੇ ((K % TILE_K) != 0) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ int64_t m_tiles     = M / TILE_M;
    ਅਚਲ int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    ਅਚਲ int64_t batch_ਗਿਣਤੀ = ne2_1 * ne3_1;
    ਅਚਲ int64_t ਅਧਾਰ_tiles  = m_tiles * n_tiles * batch_ਗਿਣਤੀ;

    ਅਚਲ int64_t r2 = ne2_1 / ne2_0;
    ਅਚਲ int64_t r3 = ne3_1 / ne3_0;

    ਅਚਲ int64_t ਕੁੱਲ_harts = ਅੰਕ_COMPUTE_SHIRES * MINIONS_ਪ੍ਰਤੀ_SHIRE;
    ਅਚਲ int64_t k_ਕਦਮਾਂ     = K / TILE_K;

    int64_t k_splits = 1;
    ਜੇ (ਅਧਾਰ_tiles < ਕੁੱਲ_harts) {
        k_splits   = (ਕੁੱਲ_harts + ਅਧਾਰ_tiles - 1) / ਅਧਾਰ_tiles;
        int64_t ks = 1;
        ਜਦੋਂ (ks * 2 <= k_splits && ks * 2 <= 32 && k_ਕਦਮਾਂ % (ks * 2) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ਅਚਲ int64_t tiles_ਪ੍ਰਤੀ_shire = MINIONS_ਪ੍ਰਤੀ_SHIRE / k_splits;
    ਅਚਲ int64_t k_ਵੰਡੋ         = local_minion % k_splits;
    ਅਚਲ int64_t local_tile_idx  = local_minion / k_splits;
    ਅਚਲ int64_t tiles_stride    = (int64_t) ਅੰਕ_COMPUTE_SHIRES * tiles_ਪ੍ਰਤੀ_shire;

    ਅਚਲ int64_t k_ਕਦਮਾਂ_ਪ੍ਰਤੀ_ਵੰਡੋ = k_ਕਦਮਾਂ / k_splits;
    ਅਚਲ int64_t k_ਸ਼ੁਰੂ           = k_ਵੰਡੋ * k_ਕਦਮਾਂ_ਪ੍ਰਤੀ_ਵੰਡੋ * TILE_K;
    ਅਚਲ int64_t k_ਸਿਰਾ             = k_ਸ਼ੁਰੂ + k_ਕਦਮਾਂ_ਪ੍ਰਤੀ_ਵੰਡੋ * TILE_K;

    // L2 SCP pointers for this minion's double-buffered panels + sync
    uint64_t    scp_ਅਧਾਰ  = local_minion * SCP_ਪ੍ਰਤੀ_MINION;
    et_fp16_t * scp_bp[2] = {
        (et_fp16_t *) et_shire_l2scp_local(scp_ਅਧਾਰ),
        (et_fp16_t *) et_shire_l2scp_local(scp_ਅਧਾਰ + SCP_BPANEL_ਮਾਪ),
    };
    ਅਸਥਿਰ uint32_t * ready_ctr    = (ਅਸਥਿਰ uint32_t *) et_shire_l2scp_local(scp_ਅਧਾਰ + SCP_READY_OFF);
    ਅਸਥਿਰ uint32_t * consumed_ctr = (ਅਸਥਿਰ uint32_t *) et_shire_l2scp_local(scp_ਅਧਾਰ + SCP_CONSUMED_OFF);

    // ================================================================
    // Hart 1: B-panel packer
    // ================================================================
    ਜੇ (is_hart1) {
        // Initialize sync counters
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);

        uint32_t chunk_id = 0;

        ਲਈ (int64_t tile = (int64_t) shire_id + local_tile_idx * ਅੰਕ_COMPUTE_SHIRES; tile < ਅਧਾਰ_tiles;
             tile += tiles_stride) {
            ਅਚਲ int64_t tiles_ਪ੍ਰਤੀ_batch = m_tiles * n_tiles;
            ਅਚਲ int64_t batch_idx       = tile / tiles_ਪ੍ਰਤੀ_batch;
            ਅਚਲ int64_t tile_ਵਿੱਚ_batch   = tile % tiles_ਪ੍ਰਤੀ_batch;

            ਅਚਲ int64_t mb_idx = tile_ਵਿੱਚ_batch % m_tiles;

            ਅਚਲ int64_t i3   = batch_idx / ne2_1;
            ਅਚਲ int64_t i2   = batch_idx % ne2_1;
            ਅਚਲ int64_t i2_0 = i2 / r2;
            ਅਚਲ int64_t i3_0 = i3 / r3;

            ਅਚਲ ਅੱਖਰ *  src0_batch = src0_ਅਧਾਰ + i3_0 * nb3_0 + i2_0 * nb2_0;
            ਅਚਲ int64_t mb         = mb_idx * TILE_M;

            ਲਈ (int64_t kb = k_ਸ਼ੁਰੂ; kb < k_ਸਿਰਾ; kb += TILE_K) {
                ਪੂਰਨ buf = chunk_id & 1;

                // Back-pressure: wait for hart 0 to finish with this buffer
                ਜੇ (chunk_id >= 2) {
                    scp_wait(consumed_ctr, chunk_id - 1);
                }

                pack_b_interleaved(scp_bp[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_ਮੰਜ਼ਿਲ_l2(scp_bp[buf], 16, 64);
                WAIT_CACHEOPS;

                chunk_id++;
                scp_signal(ready_ctr, chunk_id);
            }
        }

        FENCE;
        ਵਾਪਸ 0;
    }

    // ================================================================
    // Hart 0: tensor engine compute
    // ================================================================
    uint64_t       my_minion_id      = get_minion_id();
    ਅਚਲ uint64_t ਸਮੂਹ_ਅਧਾਰ_global = my_minion_id - k_ਵੰਡੋ;

    ਤਿਆਰੀ_cache_scp();
#ਜੇ CACHEOP_MAX > 0 || REP_ਦਰ > 0
    ucache_control(1, REP_ਦਰ, CACHEOP_MAX);
#ਜੇ_ਅੰਤ
    CLEAR_TENSOR_ERROR;

    // Evict any stale L1D copies of sync counters
    evict_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) ready_ctr, 1, 64);
    WAIT_CACHEOPS;
    evict_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) consumed_ctr, 1, 64);
    WAIT_CACHEOPS;

    uint32_t chunk_id = 0;

    ਲਈ (int64_t tile = (int64_t) shire_id + local_tile_idx * ਅੰਕ_COMPUTE_SHIRES; tile < ਅਧਾਰ_tiles;
         tile += tiles_stride) {
        ਅਚਲ int64_t tiles_ਪ੍ਰਤੀ_batch = m_tiles * n_tiles;
        ਅਚਲ int64_t batch_idx       = tile / tiles_ਪ੍ਰਤੀ_batch;
        ਅਚਲ int64_t tile_ਵਿੱਚ_batch   = tile % tiles_ਪ੍ਰਤੀ_batch;

        ਅਚਲ int64_t nb_idx = tile_ਵਿੱਚ_batch / m_tiles;
        ਅਚਲ int64_t mb_idx = tile_ਵਿੱਚ_batch % m_tiles;

        ਅਚਲ int64_t i3 = batch_idx / ne2_1;
        ਅਚਲ int64_t i2 = batch_idx % ne2_1;

        ਅਚਲ ਅੱਖਰ * src1_batch = src1_ਅਧਾਰ + i3 * nb3_1 + i2 * nb2_1;
        ਅੱਖਰ *       dst_batch  = dst_ਅਧਾਰ + i3 * nb3_d + i2 * nb2_d;

        ਅਚਲ int64_t mb    = mb_idx * TILE_M;
        ਅਚਲ int64_t nb    = nb_idx * TILE_N;
        ਅਚਲ int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        // Set tensor_mask for partial N tiles
        ਜੇ (n_cur < TILE_N) {
            uint64_t ਨਕਾਬ = (1ULL << n_cur) - 1;
            __asm__ __volatile__("csrw 0x805, %0" : : "r"(ਨਕਾਬ));
        }

        ਲਈ (int64_t kb = k_ਸ਼ੁਰੂ; kb < k_ਸਿਰਾ; kb += TILE_K) {
            ਪੂਰਨ buf = chunk_id & 1;

            // Start loading A from DRAM (overlaps with waiting for hart 1)
            tensor_ਲੱਦੋ((n_cur < TILE_N), false, A_L1_ਸ਼ੁਰੂ, TENSOR_ਲੱਦੋ_PLAIN, 0,
                        (uint64_t) (src1_batch + nb * nb1_1 + kb * (int64_t) ਆਕਾਰ(et_fp16_t)), 0, n_cur - 1,
                        (uint64_t) nb1_1, 0);

            // Wait for hart 1 to finish packing this chunk
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            // Load B from L2 SCP (hart 1 already flushed it)
            tensor_ਲੱਦੋ(false, false, B_L1_ਸ਼ੁਰੂ, TENSOR_ਲੱਦੋ_PLAIN, 0, (uint64_t) scp_bp[buf], 0, 15, 64, 1);

            tensor_wait(TENSOR_ਲੱਦੋ_WAIT_0);
            tensor_wait(TENSOR_ਲੱਦੋ_WAIT_1);

            // TensorFMA16A32
            tensor_fma((n_cur < TILE_N), 3, n_cur - 1, 15, 0, false, false, false, false, B_L1_ਸ਼ੁਰੂ, A_L1_ਸ਼ੁਰੂ,
                       TENSOR_FMA_OP_FP16, (kb == k_ਸ਼ੁਰੂ));

            tensor_wait(TENSOR_FMA_WAIT);

            // Signal that this buffer is free for hart 1 to reuse
            scp_signal(consumed_ctr, chunk_id);
        }

        // K-split ring reduce
        ਜੇ (k_splits > 1) {
            ਅਚਲ uint64_t ਅੰਕ_regs = (uint64_t) n_cur * 2;

            ਜੇ (k_ਵੰਡੋ > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, ਅੰਕ_regs, ਸਮੂਹ_ਅਧਾਰ_global + k_ਵੰਡੋ - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }

            ਜੇ (k_ਵੰਡੋ < k_splits - 1) {
                tensor_reduce_send(0, ਅੰਕ_regs, ਸਮੂਹ_ਅਧਾਰ_global + k_ਵੰਡੋ + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        // Store FP32 result tile
        ਜੇ (k_ਵੰਡੋ == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    ਵਾਪਸ 0;
}
