#ضمن "ggml_tensor.h"
#ضمن "platform.h"
#ضمن "tensor.h"

#ضمن <etsoc/common/utils.h>
#ضمن <stdint.h>

// FP16 x FP16 -> FP32 MUL_MAT with hart 1 B-panel packing
//
// Hart 0: tensor engine (load A, load B from SCP, FMA, reduce, store)
// Hart 1: pack B into double-buffered L2 SCP panels, flush for tensor_load
//
// Sync: monotonic counters in L2 SCP with evict-based coherency.
// Double-buffered bpanel allows pack/FMA overlap.
//
#تعریف_کرو عدد_COMPUTE_SHIRES 32
#تعریف_کرو MINIONS_فی_SHIRE  32

#تعریف_کرو TILE_M 16
#تعریف_کرو TILE_N 16
#تعریف_کرو TILE_K 32

#تعریف_کرو CACHEOP_MAX 0
#تعریف_کرو REP_شرح    0

#تعریف_کرو A_L1_شروع 0   // SCP lines  0..15 for A
#تعریف_کرو B_L1_شروع 16  // SCP lines 16..31 for B

تعريفنوع uint16_t et_fp16_t;

// L2 SCP layout per minion (double-buffered bpanel + sync counters)
//   [0..1023]    bpanel buffer 0 (16 lines x 64 bytes)
//   [1024..2047] bpanel buffer 1
//   [2048..2111] ready counter   (hart1 -> hart0, own cache line)
//   [2112..2175] consumed counter (hart0 -> hart1, own cache line)
#تعریف_کرو SCP_BPANEL_ناپ  (16 * 32 * حجم(et_fp16_t))  // 1024 bytes
#تعریف_کرو SCP_READY_OFF    (2 * SCP_BPANEL_ناپ)          // 2048
#تعریف_کرو SCP_CONSUMED_OFF (SCP_READY_OFF + 64)           // 2112
#تعریف_کرو SCP_فی_MINION   (SCP_CONSUMED_OFF + 64)        // 2176

// Signal a counter value to the other hart via L2 SCP.
ساكن مضمن فراغ __attribute__((always_inline)) scp_signal(متطاير uint32_t * flag, uint32_t قدر) {
    *flag = قدر;
    FENCE;
    evict_منزل_l2((ثابت فراغ *) flag, 1, 64);
    WAIT_CACHEOPS;
}

// Wait for a counter in L2 SCP to reach the expected value.
ساكن مضمن فراغ __attribute__((always_inline)) scp_wait(متطاير uint32_t * flag, uint32_t expected) {
    طالما (1) {
        evict_منزل_l2((ثابت فراغ *) flag, 1, 64);
        WAIT_CACHEOPS;
        إذا (*flag >= expected) {
            أرجع;
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
ساكن مضمن فراغ __attribute__((always_inline)) pack_b_interleaved(et_fp16_t *  باہرلا,
                                                                     ثابت حرف * src0_batch,
                                                                     int64_t      mb,
                                                                     int64_t      kb,
                                                                     int64_t      nb1_0) {
    ساكن ثابت int32_t __attribute__((aligned(32))) scatter_idx[8] = { 0, 64, 128, 192, 256, 320, 384, 448 };

    بنانشانی لما old_نقاب;
    __asm__ متطاير(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "flw.ps    f1, 0(%[idx])    \n\t"
        : [ms] "=&r"(old_نقاب)
        : [idx] "r"(scatter_idx)
        : "f1");

    لكل (صحيح j = 0; j < TILE_M; ++j) {
        ثابت et_fp16_t * صف = (ثابت et_fp16_t *) (src0_batch + (mb + j) * nb1_0) + kb;
        حرف *            dst = (حرف *) باہرلا + j * 4;

        __asm__ متطاير(
            "flw.ps    f2, 0(%[src])    \n\t"
            "flw.ps    f3, 32(%[src])   \n\t"
            "fscw.ps   f2, f1(%[d0])    \n\t"
            "fscw.ps   f3, f1(%[d1])    \n\t"
            :
            : [src] "r"(صف), [d0] "r"(dst), [d1] "r"(dst + 512)
            : "f2", "f3", "memory");
    }

    __asm__ متطاير("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_نقاب));
}

صحيح اندراج_نقطہ(بنية ggml_et_ثنائی_params * params, فراغ * env) {
    (فراغ) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    إذا (shire_id >= عدد_COMPUTE_SHIRES) {
        أرجع 0;
    }

    ثابت صحيح is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    // Dimensions (both harts need these for tile assignment)
    ثابت int64_t K = params->src0.ne[0];
    ثابت int64_t M = params->src0.ne[1];
    ثابت int64_t N = params->src1.ne[1];

    ثابت int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    ثابت int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    ثابت int64_t nb1_0 = params->src0.nb[1];
    ثابت int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];

    ثابت int64_t nb1_1 = params->src1.nb[1];
    ثابت int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];

    ثابت int64_t nb1_d = params->dst.nb[1];
    ثابت int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    ثابت حرف * src0_ادھار = (ثابت حرف *) params->src0.انکڑے;
    ثابت حرف * src1_ادھار = (ثابت حرف *) params->src1.انکڑے;
    حرف *       dst_ادھار  = (حرف *) params->dst.انکڑے;

    إذا ((M % TILE_M) != 0) {
        أرجع 0;
    }
    إذا ((K % TILE_K) != 0) {
        أرجع 0;
    }

    ثابت int64_t m_tiles     = M / TILE_M;
    ثابت int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    ثابت int64_t batch_گنتی = ne2_1 * ne3_1;
    ثابت int64_t ادھار_tiles  = m_tiles * n_tiles * batch_گنتی;

    ثابت int64_t r2 = ne2_1 / ne2_0;
    ثابت int64_t r3 = ne3_1 / ne3_0;

    ثابت int64_t کل_harts = عدد_COMPUTE_SHIRES * MINIONS_فی_SHIRE;
    ثابت int64_t k_قدماں     = K / TILE_K;

    int64_t k_splits = 1;
    إذا (ادھار_tiles < کل_harts) {
        k_splits   = (کل_harts + ادھار_tiles - 1) / ادھار_tiles;
        int64_t ks = 1;
        طالما (ks * 2 <= k_splits && ks * 2 <= 32 && k_قدماں % (ks * 2) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ثابت int64_t tiles_فی_shire = MINIONS_فی_SHIRE / k_splits;
    ثابت int64_t k_ونڈو         = local_minion % k_splits;
    ثابت int64_t local_tile_idx  = local_minion / k_splits;
    ثابت int64_t tiles_stride    = (int64_t) عدد_COMPUTE_SHIRES * tiles_فی_shire;

    ثابت int64_t k_قدماں_فی_ونڈو = k_قدماں / k_splits;
    ثابت int64_t k_شروع           = k_ونڈو * k_قدماں_فی_ونڈو * TILE_K;
    ثابت int64_t k_سرا             = k_شروع + k_قدماں_فی_ونڈو * TILE_K;

    // L2 SCP pointers for this minion's double-buffered panels + sync
    uint64_t    scp_ادھار  = local_minion * SCP_فی_MINION;
    et_fp16_t * scp_bp[2] = {
        (et_fp16_t *) et_shire_l2scp_local(scp_ادھار),
        (et_fp16_t *) et_shire_l2scp_local(scp_ادھار + SCP_BPANEL_ناپ),
    };
    متطاير uint32_t * ready_ctr    = (متطاير uint32_t *) et_shire_l2scp_local(scp_ادھار + SCP_READY_OFF);
    متطاير uint32_t * consumed_ctr = (متطاير uint32_t *) et_shire_l2scp_local(scp_ادھار + SCP_CONSUMED_OFF);

    // ================================================================
    // Hart 1: B-panel packer
    // ================================================================
    إذا (is_hart1) {
        // Initialize sync counters
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);

        uint32_t chunk_id = 0;

        لكل (int64_t tile = (int64_t) shire_id + local_tile_idx * عدد_COMPUTE_SHIRES; tile < ادھار_tiles;
             tile += tiles_stride) {
            ثابت int64_t tiles_فی_batch = m_tiles * n_tiles;
            ثابت int64_t batch_idx       = tile / tiles_فی_batch;
            ثابت int64_t tile_وچ_batch   = tile % tiles_فی_batch;

            ثابت int64_t mb_idx = tile_وچ_batch % m_tiles;

            ثابت int64_t i3   = batch_idx / ne2_1;
            ثابت int64_t i2   = batch_idx % ne2_1;
            ثابت int64_t i2_0 = i2 / r2;
            ثابت int64_t i3_0 = i3 / r3;

            ثابت حرف *  src0_batch = src0_ادھار + i3_0 * nb3_0 + i2_0 * nb2_0;
            ثابت int64_t mb         = mb_idx * TILE_M;

            لكل (int64_t kb = k_شروع; kb < k_سرا; kb += TILE_K) {
                صحيح buf = chunk_id & 1;

                // Back-pressure: wait for hart 0 to finish with this buffer
                إذا (chunk_id >= 2) {
                    scp_wait(consumed_ctr, chunk_id - 1);
                }

                pack_b_interleaved(scp_bp[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_منزل_l2(scp_bp[buf], 16, 64);
                WAIT_CACHEOPS;

                chunk_id++;
                scp_signal(ready_ctr, chunk_id);
            }
        }

        FENCE;
        أرجع 0;
    }

    // ================================================================
    // Hart 0: tensor engine compute
    // ================================================================
    uint64_t       my_minion_id      = get_minion_id();
    ثابت uint64_t گروہ_ادھار_global = my_minion_id - k_ونڈو;

    تیاری_cache_scp();
#إذا CACHEOP_MAX > 0 || REP_شرح > 0
    ucache_control(1, REP_شرح, CACHEOP_MAX);
#جے_انت
    CLEAR_TENSOR_ERROR;

    // Evict any stale L1D copies of sync counters
    evict_منزل_l2((ثابت فراغ *) ready_ctr, 1, 64);
    WAIT_CACHEOPS;
    evict_منزل_l2((ثابت فراغ *) consumed_ctr, 1, 64);
    WAIT_CACHEOPS;

    uint32_t chunk_id = 0;

    لكل (int64_t tile = (int64_t) shire_id + local_tile_idx * عدد_COMPUTE_SHIRES; tile < ادھار_tiles;
         tile += tiles_stride) {
        ثابت int64_t tiles_فی_batch = m_tiles * n_tiles;
        ثابت int64_t batch_idx       = tile / tiles_فی_batch;
        ثابت int64_t tile_وچ_batch   = tile % tiles_فی_batch;

        ثابت int64_t nb_idx = tile_وچ_batch / m_tiles;
        ثابت int64_t mb_idx = tile_وچ_batch % m_tiles;

        ثابت int64_t i3 = batch_idx / ne2_1;
        ثابت int64_t i2 = batch_idx % ne2_1;

        ثابت حرف * src1_batch = src1_ادھار + i3 * nb3_1 + i2 * nb2_1;
        حرف *       dst_batch  = dst_ادھار + i3 * nb3_d + i2 * nb2_d;

        ثابت int64_t mb    = mb_idx * TILE_M;
        ثابت int64_t nb    = nb_idx * TILE_N;
        ثابت int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        // Set tensor_mask for partial N tiles
        إذا (n_cur < TILE_N) {
            uint64_t نقاب = (1ULL << n_cur) - 1;
            __asm__ __volatile__("csrw 0x805, %0" : : "r"(نقاب));
        }

        لكل (int64_t kb = k_شروع; kb < k_سرا; kb += TILE_K) {
            صحيح buf = chunk_id & 1;

            // Start loading A from DRAM (overlaps with waiting for hart 1)
            tensor_لدو((n_cur < TILE_N), false, A_L1_شروع, TENSOR_لدو_PLAIN, 0,
                        (uint64_t) (src1_batch + nb * nb1_1 + kb * (int64_t) حجم(et_fp16_t)), 0, n_cur - 1,
                        (uint64_t) nb1_1, 0);

            // Wait for hart 1 to finish packing this chunk
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            // Load B from L2 SCP (hart 1 already flushed it)
            tensor_لدو(false, false, B_L1_شروع, TENSOR_لدو_PLAIN, 0, (uint64_t) scp_bp[buf], 0, 15, 64, 1);

            tensor_wait(TENSOR_لدو_WAIT_0);
            tensor_wait(TENSOR_لدو_WAIT_1);

            // TensorFMA16A32
            tensor_fma((n_cur < TILE_N), 3, n_cur - 1, 15, 0, false, false, false, false, B_L1_شروع, A_L1_شروع,
                       TENSOR_FMA_OP_FP16, (kb == k_شروع));

            tensor_wait(TENSOR_FMA_WAIT);

            // Signal that this buffer is free for hart 1 to reuse
            scp_signal(consumed_ctr, chunk_id);
        }

        // K-split ring reduce
        إذا (k_splits > 1) {
            ثابت uint64_t عدد_regs = (uint64_t) n_cur * 2;

            إذا (k_ونڈو > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, عدد_regs, گروہ_ادھار_global + k_ونڈو - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }

            إذا (k_ونڈو < k_splits - 1) {
                tensor_reduce_send(0, عدد_regs, گروہ_ادھار_global + k_ونڈو + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        // Store FP32 result tile
        إذا (k_ونڈو == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * (int64_t) حجم(عائم)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    أرجع 0;
}
