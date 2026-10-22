#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"
#అన్తర్భావ "tensor.h"

#అన్తర్భావ <etsoc/common/utils.h>
#అన్తర్భావ <stdint.h>

// FP16 x FP16 -> FP32 MUL_MAT with hart 1 B-panel packing
//
// Hart 0: tensor engine (load A, load B from SCP, FMA, reduce, store)
// Hart 1: pack B into double-buffered L2 SCP panels, flush for tensor_load
//
// Sync: monotonic counters in L2 SCP with evict-based coherency.
// Double-buffered bpanel allows pack/FMA overlap.
//
#నిర్వచించు అంకె_COMPUTE_SHIRES 32
#నిర్వచించు MINIONS_చొప్పున_SHIRE  32

#నిర్వచించు TILE_M 16
#నిర్వచించు TILE_N 16
#నిర్వచించు TILE_K 32

#నిర్వచించు CACHEOP_MAX 0
#నిర్వచించు REP_రేటు    0

#నిర్వచించు A_L1_ప్రారంభం 0   // SCP lines  0..15 for A
#నిర్వచించు B_L1_ప్రారంభం 16  // SCP lines 16..31 for B

రకనిర్వచనం uint16_t et_fp16_t;

// L2 SCP layout per minion (double-buffered bpanel + sync counters)
//   [0..1023]    bpanel buffer 0 (16 lines x 64 bytes)
//   [1024..2047] bpanel buffer 1
//   [2048..2111] ready counter   (hart1 -> hart0, own cache line)
//   [2112..2175] consumed counter (hart0 -> hart1, own cache line)
#నిర్వచించు SCP_BPANEL_కొలత  (16 * 32 * పరిమాణం(et_fp16_t))  // 1024 bytes
#నిర్వచించు SCP_READY_OFF    (2 * SCP_BPANEL_కొలత)          // 2048
#నిర్వచించు SCP_CONSUMED_OFF (SCP_READY_OFF + 64)           // 2112
#నిర్వచించు SCP_చొప్పున_MINION   (SCP_CONSUMED_OFF + 64)        // 2176

// Signal a counter value to the other hart via L2 SCP.
స్థిర అంతఃవరుస శూన్య __attribute__((always_inline)) scp_signal(అస్థిర uint32_t * flag, uint32_t విలువ) {
    *flag = విలువ;
    FENCE;
    evict_గమ్యం_l2((స్థిరరాశి శూన్య *) flag, 1, 64);
    WAIT_CACHEOPS;
}

// Wait for a counter in L2 SCP to reach the expected value.
స్థిర అంతఃవరుస శూన్య __attribute__((always_inline)) scp_wait(అస్థిర uint32_t * flag, uint32_t expected) {
    అయితే (1) {
        evict_గమ్యం_l2((స్థిరరాశి శూన్య *) flag, 1, 64);
        WAIT_CACHEOPS;
        ఒకవేళ (*flag >= expected) {
            తిరిగి;
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
స్థిర అంతఃవరుస శూన్య __attribute__((always_inline)) pack_b_interleaved(et_fp16_t *  బహిర్గతం,
                                                                     స్థిరరాశి అక్షరం * src0_batch,
                                                                     int64_t      mb,
                                                                     int64_t      kb,
                                                                     int64_t      nb1_0) {
    స్థిర స్థిరరాశి int32_t __attribute__((aligned(32))) scatter_idx[8] = { 0, 64, 128, 192, 256, 320, 384, 448 };

    గుర్తులేని దీర్ఘ old_ముసుగు;
    __asm__ అస్థిర(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        "flw.ps    f1, 0(%[idx])    \n\t"
        : [ms] "=&r"(old_ముసుగు)
        : [idx] "r"(scatter_idx)
        : "f1");

    కొరకు (పూర్ణాంకం j = 0; j < TILE_M; ++j) {
        స్థిరరాశి et_fp16_t * అడ్డువరుస = (స్థిరరాశి et_fp16_t *) (src0_batch + (mb + j) * nb1_0) + kb;
        అక్షరం *            dst = (అక్షరం *) బహిర్గతం + j * 4;

        __asm__ అస్థిర(
            "flw.ps    f2, 0(%[src])    \n\t"
            "flw.ps    f3, 32(%[src])   \n\t"
            "fscw.ps   f2, f1(%[d0])    \n\t"
            "fscw.ps   f3, f1(%[d1])    \n\t"
            :
            : [src] "r"(అడ్డువరుస), [d0] "r"(dst), [d1] "r"(dst + 512)
            : "f2", "f3", "memory");
    }

    __asm__ అస్థిర("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_ముసుగు));
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_ద్విమాన_params * params, శూన్య * env) {
    (శూన్య) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    ఒకవేళ (shire_id >= అంకె_COMPUTE_SHIRES) {
        తిరిగి 0;
    }

    స్థిరరాశి పూర్ణాంకం is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    // Dimensions (both harts need these for tile assignment)
    స్థిరరాశి int64_t K = params->src0.ne[0];
    స్థిరరాశి int64_t M = params->src0.ne[1];
    స్థిరరాశి int64_t N = params->src1.ne[1];

    స్థిరరాశి int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    స్థిరరాశి int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    స్థిరరాశి int64_t nb1_0 = params->src0.nb[1];
    స్థిరరాశి int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];

    స్థిరరాశి int64_t nb1_1 = params->src1.nb[1];
    స్థిరరాశి int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];

    స్థిరరాశి int64_t nb1_d = params->dst.nb[1];
    స్థిరరాశి int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    స్థిరరాశి అక్షరం * src0_పునాది = (స్థిరరాశి అక్షరం *) params->src0.దత్తాంశం;
    స్థిరరాశి అక్షరం * src1_పునాది = (స్థిరరాశి అక్షరం *) params->src1.దత్తాంశం;
    అక్షరం *       dst_పునాది  = (అక్షరం *) params->dst.దత్తాంశం;

    ఒకవేళ ((M % TILE_M) != 0) {
        తిరిగి 0;
    }
    ఒకవేళ ((K % TILE_K) != 0) {
        తిరిగి 0;
    }

    స్థిరరాశి int64_t m_tiles     = M / TILE_M;
    స్థిరరాశి int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    స్థిరరాశి int64_t batch_లెక్క = ne2_1 * ne3_1;
    స్థిరరాశి int64_t పునాది_tiles  = m_tiles * n_tiles * batch_లెక్క;

    స్థిరరాశి int64_t r2 = ne2_1 / ne2_0;
    స్థిరరాశి int64_t r3 = ne3_1 / ne3_0;

    స్థిరరాశి int64_t మొత్తం_harts = అంకె_COMPUTE_SHIRES * MINIONS_చొప్పున_SHIRE;
    స్థిరరాశి int64_t k_అడుగులు     = K / TILE_K;

    int64_t k_splits = 1;
    ఒకవేళ (పునాది_tiles < మొత్తం_harts) {
        k_splits   = (మొత్తం_harts + పునాది_tiles - 1) / పునాది_tiles;
        int64_t ks = 1;
        అయితే (ks * 2 <= k_splits && ks * 2 <= 32 && k_అడుగులు % (ks * 2) == 0) {
            ks *= 2;
        }
        k_splits = ks;
    }

    స్థిరరాశి int64_t tiles_చొప్పున_shire = MINIONS_చొప్పున_SHIRE / k_splits;
    స్థిరరాశి int64_t k_విభజించు         = local_minion % k_splits;
    స్థిరరాశి int64_t local_tile_idx  = local_minion / k_splits;
    స్థిరరాశి int64_t tiles_stride    = (int64_t) అంకె_COMPUTE_SHIRES * tiles_చొప్పున_shire;

    స్థిరరాశి int64_t k_అడుగులు_చొప్పున_విభజించు = k_అడుగులు / k_splits;
    స్థిరరాశి int64_t k_ప్రారంభం           = k_విభజించు * k_అడుగులు_చొప్పున_విభజించు * TILE_K;
    స్థిరరాశి int64_t k_చివర             = k_ప్రారంభం + k_అడుగులు_చొప్పున_విభజించు * TILE_K;

    // L2 SCP pointers for this minion's double-buffered panels + sync
    uint64_t    scp_పునాది  = local_minion * SCP_చొప్పున_MINION;
    et_fp16_t * scp_bp[2] = {
        (et_fp16_t *) et_shire_l2scp_local(scp_పునాది),
        (et_fp16_t *) et_shire_l2scp_local(scp_పునాది + SCP_BPANEL_కొలత),
    };
    అస్థిర uint32_t * ready_ctr    = (అస్థిర uint32_t *) et_shire_l2scp_local(scp_పునాది + SCP_READY_OFF);
    అస్థిర uint32_t * consumed_ctr = (అస్థిర uint32_t *) et_shire_l2scp_local(scp_పునాది + SCP_CONSUMED_OFF);

    // ================================================================
    // Hart 1: B-panel packer
    // ================================================================
    ఒకవేళ (is_hart1) {
        // Initialize sync counters
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);

        uint32_t chunk_id = 0;

        కొరకు (int64_t tile = (int64_t) shire_id + local_tile_idx * అంకె_COMPUTE_SHIRES; tile < పునాది_tiles;
             tile += tiles_stride) {
            స్థిరరాశి int64_t tiles_చొప్పున_batch = m_tiles * n_tiles;
            స్థిరరాశి int64_t batch_idx       = tile / tiles_చొప్పున_batch;
            స్థిరరాశి int64_t tile_లోపల_batch   = tile % tiles_చొప్పున_batch;

            స్థిరరాశి int64_t mb_idx = tile_లోపల_batch % m_tiles;

            స్థిరరాశి int64_t i3   = batch_idx / ne2_1;
            స్థిరరాశి int64_t i2   = batch_idx % ne2_1;
            స్థిరరాశి int64_t i2_0 = i2 / r2;
            స్థిరరాశి int64_t i3_0 = i3 / r3;

            స్థిరరాశి అక్షరం *  src0_batch = src0_పునాది + i3_0 * nb3_0 + i2_0 * nb2_0;
            స్థిరరాశి int64_t mb         = mb_idx * TILE_M;

            కొరకు (int64_t kb = k_ప్రారంభం; kb < k_చివర; kb += TILE_K) {
                పూర్ణాంకం buf = chunk_id & 1;

                // Back-pressure: wait for hart 0 to finish with this buffer
                ఒకవేళ (chunk_id >= 2) {
                    scp_wait(consumed_ctr, chunk_id - 1);
                }

                pack_b_interleaved(scp_bp[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_గమ్యం_l2(scp_bp[buf], 16, 64);
                WAIT_CACHEOPS;

                chunk_id++;
                scp_signal(ready_ctr, chunk_id);
            }
        }

        FENCE;
        తిరిగి 0;
    }

    // ================================================================
    // Hart 0: tensor engine compute
    // ================================================================
    uint64_t       my_minion_id      = get_minion_id();
    స్థిరరాశి uint64_t సమూహం_పునాది_global = my_minion_id - k_విభజించు;

    అమరిక_cache_scp();
#ఒకవేళ CACHEOP_MAX > 0 || REP_రేటు > 0
    ucache_control(1, REP_రేటు, CACHEOP_MAX);
#అయితే_ముగింపు
    CLEAR_TENSOR_ERROR;

    // Evict any stale L1D copies of sync counters
    evict_గమ్యం_l2((స్థిరరాశి శూన్య *) ready_ctr, 1, 64);
    WAIT_CACHEOPS;
    evict_గమ్యం_l2((స్థిరరాశి శూన్య *) consumed_ctr, 1, 64);
    WAIT_CACHEOPS;

    uint32_t chunk_id = 0;

    కొరకు (int64_t tile = (int64_t) shire_id + local_tile_idx * అంకె_COMPUTE_SHIRES; tile < పునాది_tiles;
         tile += tiles_stride) {
        స్థిరరాశి int64_t tiles_చొప్పున_batch = m_tiles * n_tiles;
        స్థిరరాశి int64_t batch_idx       = tile / tiles_చొప్పున_batch;
        స్థిరరాశి int64_t tile_లోపల_batch   = tile % tiles_చొప్పున_batch;

        స్థిరరాశి int64_t nb_idx = tile_లోపల_batch / m_tiles;
        స్థిరరాశి int64_t mb_idx = tile_లోపల_batch % m_tiles;

        స్థిరరాశి int64_t i3 = batch_idx / ne2_1;
        స్థిరరాశి int64_t i2 = batch_idx % ne2_1;

        స్థిరరాశి అక్షరం * src1_batch = src1_పునాది + i3 * nb3_1 + i2 * nb2_1;
        అక్షరం *       dst_batch  = dst_పునాది + i3 * nb3_d + i2 * nb2_d;

        స్థిరరాశి int64_t mb    = mb_idx * TILE_M;
        స్థిరరాశి int64_t nb    = nb_idx * TILE_N;
        స్థిరరాశి int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        // Set tensor_mask for partial N tiles
        ఒకవేళ (n_cur < TILE_N) {
            uint64_t ముసుగు = (1ULL << n_cur) - 1;
            __asm__ __volatile__("csrw 0x805, %0" : : "r"(ముసుగు));
        }

        కొరకు (int64_t kb = k_ప్రారంభం; kb < k_చివర; kb += TILE_K) {
            పూర్ణాంకం buf = chunk_id & 1;

            // Start loading A from DRAM (overlaps with waiting for hart 1)
            tensor_ఎక్కించు((n_cur < TILE_N), false, A_L1_ప్రారంభం, TENSOR_ఎక్కించు_PLAIN, 0,
                        (uint64_t) (src1_batch + nb * nb1_1 + kb * (int64_t) పరిమాణం(et_fp16_t)), 0, n_cur - 1,
                        (uint64_t) nb1_1, 0);

            // Wait for hart 1 to finish packing this chunk
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            // Load B from L2 SCP (hart 1 already flushed it)
            tensor_ఎక్కించు(false, false, B_L1_ప్రారంభం, TENSOR_ఎక్కించు_PLAIN, 0, (uint64_t) scp_bp[buf], 0, 15, 64, 1);

            tensor_wait(TENSOR_ఎక్కించు_WAIT_0);
            tensor_wait(TENSOR_ఎక్కించు_WAIT_1);

            // TensorFMA16A32
            tensor_fma((n_cur < TILE_N), 3, n_cur - 1, 15, 0, false, false, false, false, B_L1_ప్రారంభం, A_L1_ప్రారంభం,
                       TENSOR_FMA_OP_FP16, (kb == k_ప్రారంభం));

            tensor_wait(TENSOR_FMA_WAIT);

            // Signal that this buffer is free for hart 1 to reuse
            scp_signal(consumed_ctr, chunk_id);
        }

        // K-split ring reduce
        ఒకవేళ (k_splits > 1) {
            స్థిరరాశి uint64_t అంకె_regs = (uint64_t) n_cur * 2;

            ఒకవేళ (k_విభజించు > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, అంకె_regs, సమూహం_పునాది_global + k_విభజించు - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }

            ఒకవేళ (k_విభజించు < k_splits - 1) {
                tensor_reduce_send(0, అంకె_regs, సమూహం_పునాది_global + k_విభజించు + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        // Store FP32 result tile
        ఒకవేళ (k_విభజించు == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * (int64_t) పరిమాణం(దశాంశం)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    తిరిగి 0;
}
