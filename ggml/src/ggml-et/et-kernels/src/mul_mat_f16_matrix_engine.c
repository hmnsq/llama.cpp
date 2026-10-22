#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"
#অন্তর্ভাব "tensor.h"

#অন্তর্ভাব <etsoc/common/utils.h>
#অন্তর্ভাব <stdint.h>

// FP16 x FP16 -> FP32 MUL_MAT with hart 1 B-panel packing
//
// Hart 0: tensor engine (load A, load B from SCP, FMA, reduce, store)
// Hart 1: pack B into double-buffered L2 SCP panels, flush for tensor_load
//
// Sync: monotonic counters in L2 SCP with evict-based coherency.
// Double-buffered bpanel allows pack/FMA overlap.
//
#define অঙ্ক_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32

#define TILE_M 16
#define TILE_N 16
#define TILE_K 32

#define CACHEOP_MAX 0
#define REP_RATE    0

#define A_L1_START 0   // SCP lines  0..15 for A
#define B_L1_START 16  // SCP lines 16..31 for B

প্রকারলেখ uint16_t et_fp16_t;

// L2 SCP layout per minion (double-buffered bpanel + sync counters)
//   [0..1023]    bpanel buffer 0 (16 lines x 64 bytes)
//   [1024..2047] bpanel buffer 1
//   [2048..2111] ready counter   (hart1 -> hart0, own cache line)
//   [2112..2175] consumed counter (hart0 -> hart1, own cache line)
#define SCP_BPANEL_আকার  (16 * 32 * আকার(et_fp16_t))  // 1024 bytes
#define SCP_READY_OFF    (2 * SCP_BPANEL_আকার)          // 2048
#define SCP_CONSUMED_OFF (SCP_READY_OFF + 64)           // 2112
#define SCP_PER_MINION   (SCP_CONSUMED_OFF + 64)        // 2176

// Signal a counter value to the other hart via L2 SCP.
স্থির অন্তঃরেখ শূন্য __attribute__((always_inline)) scp_signal(অস্থির uint32_t * flag, uint32_t মান) {
    *flag = মান;
    FENCE;
    evict_to_l2((ধ্রুবক শূন্য *) flag, 1, 64);
    WAIT_CACHEOPS;
}

// Wait for a counter in L2 SCP to reach the expected value.
স্থির অন্তঃরেখ শূন্য __attribute__((always_inline)) scp_wait(অস্থির uint32_t * flag, uint32_t expected) {
    যখন (1) {
        evict_to_l2((ধ্রুবক শূন্য *) flag, 1, 64);
        WAIT_CACHEOPS;
        যদি (*flag >= expected) {
            ফেরত;
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
স্থির অন্তঃরেখ শূন্য __attribute__((always_inline)) pack_b_interleaved(et_fp16_t *  out,
                                                                     ধ্রুবক অক্ষর * src0_batch,
                                                                     int64_t      mb,
                                                                     int64_t      kb,
                                                                     int64_t      nb1_0) {
    স্থির ধ্রুবক int32_t __attribute__((aligned(32))) scatter_idx[8] = { 0, 64, 128, 192, 256, 320, 384, 448 };

    unsigned long old_mask;
    __asm__ অস্থির(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "flw.ps    f1, 0(%[idx])    \n\t"
        : [ms] "=&r"(old_mask)
        : [idx] "r"(scatter_idx)
        : "f1");

    জন্য (পূর্ণসংখ্যা j = 0; j < TILE_M; ++j) {
        ধ্রুবক et_fp16_t * row = (ধ্রুবক et_fp16_t *) (src0_batch + (mb + j) * nb1_0) + kb;
        অক্ষর *            dst = (অক্ষর *) out + j * 4;

        __asm__ অস্থির(
            "flw.ps    f2, 0(%[src])    \n\t"
            "flw.ps    f3, 32(%[src])   \n\t"
            "fscw.ps   f2, f1(%[d0])    \n\t"
            "fscw.ps   f3, f1(%[d1])    \n\t"
            :
            : [src] "r"(row), [d0] "r"(dst), [d1] "r"(dst + 512)
            : "f2", "f3", "memory");
    }

    __asm__ অস্থির("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_mask));
}

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_দ্বিমিক_params * params, শূন্য * env) {
    (শূন্য) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    যদি (shire_id >= অঙ্ক_COMPUTE_SHIRES) {
        ফেরত 0;
    }

    ধ্রুবক পূর্ণসংখ্যা is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    // Dimensions (both harts need these for tile assignment)
    ধ্রুবক int64_t K = params->src0.ne[0];
    ধ্রুবক int64_t M = params->src0.ne[1];
    ধ্রুবক int64_t N = params->src1.ne[1];

    ধ্রুবক int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    ধ্রুবক int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    ধ্রুবক int64_t nb1_0 = params->src0.nb[1];
    ধ্রুবক int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];

    ধ্রুবক int64_t nb1_1 = params->src1.nb[1];
    ধ্রুবক int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];

    ধ্রুবক int64_t nb1_d = params->dst.nb[1];
    ধ্রুবক int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    ধ্রুবক অক্ষর * src0_base = (ধ্রুবক অক্ষর *) params->src0.তথ্য;
    ধ্রুবক অক্ষর * src1_base = (ধ্রুবক অক্ষর *) params->src1.তথ্য;
    অক্ষর *       dst_base  = (অক্ষর *) params->dst.তথ্য;

    যদি ((M % TILE_M) != 0) {
        ফেরত 0;
    }
    যদি ((K % TILE_K) != 0) {
        ফেরত 0;
    }

    ধ্রুবক int64_t m_tiles     = M / TILE_M;
    ধ্রুবক int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    ধ্রুবক int64_t batch_গুনতি = ne2_1 * ne3_1;
    ধ্রুবক int64_t base_tiles  = m_tiles * n_tiles * batch_গুনতি;

    ধ্রুবক int64_t r2 = ne2_1 / ne2_0;
    ধ্রুবক int64_t r3 = ne3_1 / ne3_0;

    ধ্রুবক int64_t যোগ_harts = অঙ্ক_COMPUTE_SHIRES * MINIONS_PER_SHIRE;
    ধ্রুবক int64_t k_steps     = K / TILE_K;

    int64_t k_splits = 1;
    যদি (base_tiles < যোগ_harts) {
        k_splits   = (যোগ_harts + base_tiles - 1) / base_tiles;
        int64_t ks = 1;
        যখন (ks * 2 <= k_splits && ks * 2 <= 32 && k_steps % (ks * 2) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ধ্রুবক int64_t tiles_per_shire = MINIONS_PER_SHIRE / k_splits;
    ধ্রুবক int64_t k_split         = local_minion % k_splits;
    ধ্রুবক int64_t local_tile_idx  = local_minion / k_splits;
    ধ্রুবক int64_t tiles_stride    = (int64_t) অঙ্ক_COMPUTE_SHIRES * tiles_per_shire;

    ধ্রুবক int64_t k_steps_per_split = k_steps / k_splits;
    ধ্রুবক int64_t k_start           = k_split * k_steps_per_split * TILE_K;
    ধ্রুবক int64_t k_end             = k_start + k_steps_per_split * TILE_K;

    // L2 SCP pointers for this minion's double-buffered panels + sync
    uint64_t    scp_base  = local_minion * SCP_PER_MINION;
    et_fp16_t * scp_bp[2] = {
        (et_fp16_t *) et_shire_l2scp_local(scp_base),
        (et_fp16_t *) et_shire_l2scp_local(scp_base + SCP_BPANEL_আকার),
    };
    অস্থির uint32_t * ready_ctr    = (অস্থির uint32_t *) et_shire_l2scp_local(scp_base + SCP_READY_OFF);
    অস্থির uint32_t * consumed_ctr = (অস্থির uint32_t *) et_shire_l2scp_local(scp_base + SCP_CONSUMED_OFF);

    // ================================================================
    // Hart 1: B-panel packer
    // ================================================================
    যদি (is_hart1) {
        // Initialize sync counters
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);

        uint32_t chunk_id = 0;

        জন্য (int64_t tile = (int64_t) shire_id + local_tile_idx * অঙ্ক_COMPUTE_SHIRES; tile < base_tiles;
             tile += tiles_stride) {
            ধ্রুবক int64_t tiles_per_batch = m_tiles * n_tiles;
            ধ্রুবক int64_t batch_idx       = tile / tiles_per_batch;
            ধ্রুবক int64_t tile_in_batch   = tile % tiles_per_batch;

            ধ্রুবক int64_t mb_idx = tile_in_batch % m_tiles;

            ধ্রুবক int64_t i3   = batch_idx / ne2_1;
            ধ্রুবক int64_t i2   = batch_idx % ne2_1;
            ধ্রুবক int64_t i2_0 = i2 / r2;
            ধ্রুবক int64_t i3_0 = i3 / r3;

            ধ্রুবক অক্ষর *  src0_batch = src0_base + i3_0 * nb3_0 + i2_0 * nb2_0;
            ধ্রুবক int64_t mb         = mb_idx * TILE_M;

            জন্য (int64_t kb = k_start; kb < k_end; kb += TILE_K) {
                পূর্ণসংখ্যা buf = chunk_id & 1;

                // Back-pressure: wait for hart 0 to finish with this buffer
                যদি (chunk_id >= 2) {
                    scp_wait(consumed_ctr, chunk_id - 1);
                }

                pack_b_interleaved(scp_bp[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_to_l2(scp_bp[buf], 16, 64);
                WAIT_CACHEOPS;

                chunk_id++;
                scp_signal(ready_ctr, chunk_id);
            }
        }

        FENCE;
        ফেরত 0;
    }

    // ================================================================
    // Hart 0: tensor engine compute
    // ================================================================
    uint64_t       my_minion_id      = get_minion_id();
    ধ্রুবক uint64_t group_base_global = my_minion_id - k_split;

    setup_cache_scp();
#যদি CACHEOP_MAX > 0 || REP_RATE > 0
    ucache_control(1, REP_RATE, CACHEOP_MAX);
#endif
    CLEAR_TENSOR_ERROR;

    // Evict any stale L1D copies of sync counters
    evict_to_l2((ধ্রুবক শূন্য *) ready_ctr, 1, 64);
    WAIT_CACHEOPS;
    evict_to_l2((ধ্রুবক শূন্য *) consumed_ctr, 1, 64);
    WAIT_CACHEOPS;

    uint32_t chunk_id = 0;

    জন্য (int64_t tile = (int64_t) shire_id + local_tile_idx * অঙ্ক_COMPUTE_SHIRES; tile < base_tiles;
         tile += tiles_stride) {
        ধ্রুবক int64_t tiles_per_batch = m_tiles * n_tiles;
        ধ্রুবক int64_t batch_idx       = tile / tiles_per_batch;
        ধ্রুবক int64_t tile_in_batch   = tile % tiles_per_batch;

        ধ্রুবক int64_t nb_idx = tile_in_batch / m_tiles;
        ধ্রুবক int64_t mb_idx = tile_in_batch % m_tiles;

        ধ্রুবক int64_t i3 = batch_idx / ne2_1;
        ধ্রুবক int64_t i2 = batch_idx % ne2_1;

        ধ্রুবক অক্ষর * src1_batch = src1_base + i3 * nb3_1 + i2 * nb2_1;
        অক্ষর *       dst_batch  = dst_base + i3 * nb3_d + i2 * nb2_d;

        ধ্রুবক int64_t mb    = mb_idx * TILE_M;
        ধ্রুবক int64_t nb    = nb_idx * TILE_N;
        ধ্রুবক int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        // Set tensor_mask for partial N tiles
        যদি (n_cur < TILE_N) {
            uint64_t mask = (1ULL << n_cur) - 1;
            __asm__ __volatile__("csrw 0x805, %0" : : "r"(mask));
        }

        জন্য (int64_t kb = k_start; kb < k_end; kb += TILE_K) {
            পূর্ণসংখ্যা buf = chunk_id & 1;

            // Start loading A from DRAM (overlaps with waiting for hart 1)
            tensor_load((n_cur < TILE_N), false, A_L1_START, TENSOR_LOAD_PLAIN, 0,
                        (uint64_t) (src1_batch + nb * nb1_1 + kb * (int64_t) আকার(et_fp16_t)), 0, n_cur - 1,
                        (uint64_t) nb1_1, 0);

            // Wait for hart 1 to finish packing this chunk
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            // Load B from L2 SCP (hart 1 already flushed it)
            tensor_load(false, false, B_L1_START, TENSOR_LOAD_PLAIN, 0, (uint64_t) scp_bp[buf], 0, 15, 64, 1);

            tensor_wait(TENSOR_LOAD_WAIT_0);
            tensor_wait(TENSOR_LOAD_WAIT_1);

            // TensorFMA16A32
            tensor_fma((n_cur < TILE_N), 3, n_cur - 1, 15, 0, false, false, false, false, B_L1_START, A_L1_START,
                       TENSOR_FMA_OP_FP16, (kb == k_start));

            tensor_wait(TENSOR_FMA_WAIT);

            // Signal that this buffer is free for hart 1 to reuse
            scp_signal(consumed_ctr, chunk_id);
        }

        // K-split ring reduce
        যদি (k_splits > 1) {
            ধ্রুবক uint64_t অঙ্ক_regs = (uint64_t) n_cur * 2;

            যদি (k_split > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, অঙ্ক_regs, group_base_global + k_split - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }

            যদি (k_split < k_splits - 1) {
                tensor_reduce_send(0, অঙ্ক_regs, group_base_global + k_split + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        // Store FP32 result tile
        যদি (k_split == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * (int64_t) আকার(দশমিক)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    ফেরত 0;
}
