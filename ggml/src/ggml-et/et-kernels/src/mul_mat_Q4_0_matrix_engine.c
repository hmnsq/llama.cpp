#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"
#అన్తర్భావ "quants.h"
#అన్తర్భావ "tensor.h"

#అన్తర్భావ <etsoc/common/utils.h>
#అన్తర్భావ <stdint.h>

// Q4_0 x F32 -> F32 MUL_MAT on the tensor (matrix) engine, TensorFMA32.
// Hart 1: dequantize Q4_0 weights to FP32 into double-buffered L2 SCP.
// Hart 0: tensor engine compute (FMA, reduce, store).

#నిర్వచించు అంకె_COMPUTE_SHIRES 32
#నిర్వచించు MINIONS_చొప్పున_SHIRE  32

#నిర్వచించు TILE_M  16
#నిర్వచించు TILE_N  16
#నిర్వచించు ఖండం_K QK4_0  // 32 elements per Q4_0 block
#నిర్వచించు FMA_K   16     // tensor FMA k-width for FP32 (a_num_cols = FMA_K-1)

#నిర్వచించు CACHEOP_MAX 0
#నిర్వచించు REP_రేటు    0

#నిర్వచించు A_L1_ప్రారంభం 0   // L1 SCP lines  0..15 for A (activations)
#నిర్వచించు B_L1_ప్రారంభం 16  // L1 SCP lines 16..31 for B (dequantized weights)

// L2 SCP layout per minion (double-buffered dequant panel + sync counters).
// panel = BLOCK_K k-lines x TILE_M m (FP32) = 32 * 64 = 2048 bytes, in TenB
// [k][m] order: panel[k*TILE_M + m].
#నిర్వచించు SCP_PANEL_కొలత   (ఖండం_K * TILE_M * (uint64_t) పరిమాణం(దశాంశం))  // 2048
#నిర్వచించు SCP_READY_OFF    (2 * SCP_PANEL_కొలత)                           // 4096
#నిర్వచించు SCP_CONSUMED_OFF (SCP_READY_OFF + 64)                           // 4160
#నిర్వచించు SCP_చొప్పున_MINION   (SCP_CONSUMED_OFF + 64)                        // 4224

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

// Dequantize one 32-element Q4_0 block of TILE_M weight rows into the FP32
// panel, written directly in TenB [k][m] order: panel[k*TILE_M + m].
//   Low  nibble of byte i -> k = i
//   High nibble of byte i -> k = i + 16
//   value = d * (nibble - 8)
//
// Vectorized: for each weight row m we gather 8 packed bytes at a time, expand
// the low/high nibbles to FP32 (nibble-8), scale by the block's fp16 d, and
// fscw.ps-scatter the 8 values down 8 panel lines (stride 64B) at column m.
// 4 groups of 8 cover the 32 k-values (low 0..15, high 16..31).
స్థిర అంతఃవరుస శూన్య __attribute__((always_inline)) dequant_q4_0_panel(దశాంశం *      panel,
                                                                     స్థిరరాశి అక్షరం * src0_batch,
                                                                     int64_t      mb,
                                                                     int64_t      kb_ఖండం,
                                                                     int64_t      nb1_0) {
    స్థిర స్థిరరాశి int32_t __attribute__((aligned(32))) scatter_idx[8] = {
        0, 64, 128, 192, 256, 320, 384, 448  // byte offsets: 8 lines apart
    };
    స్థిర స్థిరరాశి int32_t __attribute__((aligned(32))) gather_idx[8] = {
        0, 1, 2, 3, 4, 5, 6, 7  // 8 consecutive bytes
    };

    గుర్తులేని దీర్ఘ old_ముసుగు;
    __asm__ అస్థిర(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"  // all 8 lanes active
        "flw.ps    f1, (%[sidx])    \n\t"  // f1 = scatter offsets
        "flw.ps    f2, (%[gidx])    \n\t"  // f2 = gather offsets
        : [ms] "=&r"(old_ముసుగు)
        : [sidx] "r"(scatter_idx), [gidx] "r"(gather_idx)
        : "f1", "f2");

    అక్షరం * pbase = (అక్షరం *) panel;
    కొరకు (పూర్ణాంకం j = 0; j < TILE_M; ++j) {
        స్థిరరాశి ఖండం_q4_0 * blk       = (స్థిరరాశి ఖండం_q4_0 *) (src0_batch + (mb + j) * nb1_0) + kb_ఖండం;
        uint32_t           scale_raw = (uint32_t) blk->d;
        స్థిరరాశి uint8_t *    qs        = blk->qs;
        అక్షరం *             నిలువు       = pbase + j * 4;  // column m=j of the panel

        __asm__ అస్థిర(
            "fbcx.ps     f3, %[sb]      \n\t"  // broadcast fp16 scale bits
            "fcvt.ps.f16 f3, f3         \n\t"  // -> d in all 8 lanes (fp32)

            "fgb.ps      f4, f2(%[qs0]) \n\t"  // gather qs[0..7]
            "fandi.pi    f5, f4, 15     \n\t"  // low nibble
            "faddi.pi    f5, f5, -8     \n\t"
            "fcvt.ps.pw  f5, f5, rne    \n\t"
            "fmul.ps     f5, f5, f3     \n\t"
            "fscw.ps     f5, f1(%[c0])  \n\t"  // k=0..7   -> lines 0..7
            "fsrli.pi    f6, f4, 4      \n\t"  // high nibble
            "fandi.pi    f6, f6, 15     \n\t"
            "faddi.pi    f6, f6, -8     \n\t"
            "fcvt.ps.pw  f6, f6, rne    \n\t"
            "fmul.ps     f6, f6, f3     \n\t"
            "fscw.ps     f6, f1(%[c16]) \n\t"  // k=16..23 -> lines 16..23

            "fgb.ps      f4, f2(%[qs8]) \n\t"  // gather qs[8..15]
            "fandi.pi    f5, f4, 15     \n\t"
            "faddi.pi    f5, f5, -8     \n\t"
            "fcvt.ps.pw  f5, f5, rne    \n\t"
            "fmul.ps     f5, f5, f3     \n\t"
            "fscw.ps     f5, f1(%[c8])  \n\t"  // k=8..15  -> lines 8..15
            "fsrli.pi    f6, f4, 4      \n\t"
            "fandi.pi    f6, f6, 15     \n\t"
            "faddi.pi    f6, f6, -8     \n\t"
            "fcvt.ps.pw  f6, f6, rne    \n\t"
            "fmul.ps     f6, f6, f3     \n\t"
            "fscw.ps     f6, f1(%[c24]) \n\t"  // k=24..31 -> lines 24..31
            :
            : [sb] "r"(scale_raw), [qs0] "r"(qs), [qs8] "r"(qs + 8), [c0] "r"(నిలువు), [c8] "r"(నిలువు + 8 * 64),
              [c16] "r"(నిలువు + 16 * 64), [c24] "r"(నిలువు + 24 * 64)
            : "f3", "f4", "f5", "f6", "memory");
    }

    __asm__ అస్థిర("mova.m.x %0" ::"r"(old_ముసుగు));
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

    ఒకవేళ ((M % TILE_M) != 0) {
        తిరిగి 0;
    }
    ఒకవేళ ((K % ఖండం_K) != 0) {
        తిరిగి 0;
    }

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

    స్థిరరాశి int64_t m_tiles     = M / TILE_M;
    స్థిరరాశి int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    స్థిరరాశి int64_t batch_లెక్క = ne2_1 * ne3_1;
    స్థిరరాశి int64_t పునాది_tiles  = m_tiles * n_tiles * batch_లెక్క;

    స్థిరరాశి int64_t r2 = ne2_1 / ne2_0;
    స్థిరరాశి int64_t r3 = ne3_1 / ne3_0;

    స్థిరరాశి int64_t k_అడుగులు = K / ఖండం_K;  // number of Q4_0 blocks

    // Force a single K-split.
    స్థిరరాశి int64_t k_splits = 1;

    స్థిరరాశి int64_t tiles_చొప్పున_shire = MINIONS_చొప్పున_SHIRE / k_splits;
    స్థిరరాశి int64_t k_విభజించు         = local_minion % k_splits;
    స్థిరరాశి int64_t local_tile_idx  = local_minion / k_splits;
    స్థిరరాశి int64_t tiles_stride    = (int64_t) అంకె_COMPUTE_SHIRES * tiles_చొప్పున_shire;

    స్థిరరాశి int64_t k_అడుగులు_చొప్పున_విభజించు = k_అడుగులు / k_splits;
    స్థిరరాశి int64_t kb_ప్రారంభం          = k_విభజించు * k_అడుగులు_చొప్పున_విభజించు;   // first block
    స్థిరరాశి int64_t kb_చివర            = kb_ప్రారంభం + k_అడుగులు_చొప్పున_విభజించు;  // one past last

    // L2 SCP pointers for this minion's double-buffered panels + sync.
    uint64_t scp_పునాది     = local_minion * SCP_చొప్పున_MINION;
    దశాంశం *  scp_panel[2] = {
        (దశాంశం *) et_shire_l2scp_local(scp_పునాది),
        (దశాంశం *) et_shire_l2scp_local(scp_పునాది + SCP_PANEL_కొలత),
    };
    అస్థిర uint32_t * ready_ctr    = (అస్థిర uint32_t *) et_shire_l2scp_local(scp_పునాది + SCP_READY_OFF);
    అస్థిర uint32_t * consumed_ctr = (అస్థిర uint32_t *) et_shire_l2scp_local(scp_పునాది + SCP_CONSUMED_OFF);

    // ================================================================
    // Hart 1: Q4_0 weight dequant producer
    // ================================================================
    ఒకవేళ (is_hart1) {
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

            కొరకు (int64_t kb = kb_ప్రారంభం; kb < kb_చివర; ++kb) {
                పూర్ణాంకం buf = chunk_id & 1;

                // Back-pressure: wait for hart 0 to finish with this buffer.
                ఒకవేళ (chunk_id >= 2) {
                    scp_wait(consumed_ctr, chunk_id - 1);
                }

                dequant_q4_0_panel(scp_panel[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_గమ్యం_l2(scp_panel[buf], ఖండం_K, 64);
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

        // Partial-N tiles run TensorFMA32 with a_num_rows = n_cur-1.
        // Errata Type D workaround for n_cur == 4 (AROWS==3): pad A to AROWS==4.
        స్థిరరాశి int64_t arows_fma = (n_cur == 4) ? 4 : (n_cur - 1);

        ఒకవేళ (n_cur == 4) {
            // Zero the padded 5th A row (line A_L1_START+4) once; the per-pass A
            // load only writes lines A_L1_START..+3, so this persists.
            స్థిర స్థిరరాశి దశాంశం __attribute__((aligned(64))) zero_పంక్తి[16] = { 0 };
            tensor_ఎక్కించు(false, false, A_L1_ప్రారంభం + 4, TENSOR_ఎక్కించు_PLAIN, 0, (uint64_t) zero_పంక్తి, 0,
                        0,  // 1 line
                        64, 0);
            tensor_wait(TENSOR_ఎక్కించు_WAIT_0);
        }

        పూర్ణాంకం మొదటి = 1;  // first_pass=1 only for the very first FMA of the tile

        కొరకు (int64_t kb = kb_ప్రారంభం; kb < kb_చివర; ++kb) {
            పూర్ణాంకం buf = chunk_id & 1;

            // Wait for hart 1 to finish dequantizing this block.
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            // Two FMA passes over the 32-wide block (16 K-cols each).
            కొరకు (పూర్ణాంకం half = 0; half < 2; ++half) {
                స్థిరరాశి int64_t k_elem = kb * ఖండం_K + half * FMA_K;

                // Load A (activations) for this 16-K sub-tile, PLAIN.
                tensor_ఎక్కించు(false, false, A_L1_ప్రారంభం, TENSOR_ఎక్కించు_PLAIN, 0,
                            (uint64_t) (src1_batch + nb * nb1_1 + k_elem * (int64_t) పరిమాణం(దశాంశం)), 0, n_cur - 1,
                            (uint64_t) nb1_1, 0);

                // Load B (dequantized weights) half from L2 SCP panel, PLAIN.
                tensor_ఎక్కించు(false, false, B_L1_ప్రారంభం, TENSOR_ఎక్కించు_PLAIN, 0,
                            (uint64_t) (scp_panel[buf] + (int64_t) half * FMA_K * TILE_M), 0, FMA_K - 1, 64, 1);

                tensor_wait(TENSOR_ఎక్కించు_WAIT_0);
                tensor_wait(TENSOR_ఎక్కించు_WAIT_1);

                tensor_fma(false,
                           3,          // b_num_col: (16/4)-1
                           arows_fma,  // a_num_rows (n_cur-1, or 4 for the n_cur==4 errata pad)
                           FMA_K - 1,  // a_num_cols
                           0, false, false, false, false, B_L1_ప్రారంభం, A_L1_ప్రారంభం, TENSOR_FMA_OP_FP32, మొదటి);

                tensor_wait(TENSOR_FMA_WAIT);
                మొదటి = 0;
            }

            // Signal that this buffer is free for hart 1 to reuse.
            scp_signal(consumed_ctr, chunk_id);
        }

        // K-split ring reduce.
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

        // Store FP32 result tile (only the last k-split owns the final sum).
        ఒకవేళ (k_విభజించు == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * (int64_t) పరిమాణం(దశాంశం)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    తిరిగి 0;
}
