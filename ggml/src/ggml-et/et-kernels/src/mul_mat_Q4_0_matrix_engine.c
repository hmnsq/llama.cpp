#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"
#ضمن "quants.h"
#ضمن "tensor.h"

#ضمن <etsoc/common/utils.h>
#ضمن <stdint.h>

// Q4_0 x F32 -> F32 MUL_MAT on the tensor (matrix) engine, TensorFMA32.
// Hart 1: dequantize Q4_0 weights to FP32 into double-buffered L2 SCP.
// Hart 0: tensor engine compute (FMA, reduce, store).

#define عدد_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32

#define TILE_M  16
#define TILE_N  16
#define BLOCK_K QK4_0  // 32 elements per Q4_0 block
#define FMA_K   16     // tensor FMA k-width for FP32 (a_num_cols = FMA_K-1)

#define CACHEOP_MAX 0
#define REP_RATE    0

#define A_L1_START 0   // L1 SCP lines  0..15 for A (activations)
#define B_L1_START 16  // L1 SCP lines 16..31 for B (dequantized weights)

// L2 SCP layout per minion (double-buffered dequant panel + sync counters).
// panel = BLOCK_K k-lines x TILE_M m (FP32) = 32 * 64 = 2048 bytes, in TenB
// [k][m] order: panel[k*TILE_M + m].
#define SCP_PANEL_SIZE   (BLOCK_K * TILE_M * (uint64_t) حجم(عائم))  // 2048
#define SCP_READY_OFF    (2 * SCP_PANEL_SIZE)                           // 4096
#define SCP_CONSUMED_OFF (SCP_READY_OFF + 64)                           // 4160
#define SCP_PER_MINION   (SCP_CONSUMED_OFF + 64)                        // 4224

// Signal a counter value to the other hart via L2 SCP.
ساكن مضمن فراغ __attribute__((always_inline)) scp_signal(متطاير uint32_t * flag, uint32_t قدر) {
    *flag = قدر;
    FENCE;
    evict_to_l2((ثابت فراغ *) flag, 1, 64);
    WAIT_CACHEOPS;
}

// Wait for a counter in L2 SCP to reach the expected value.
ساكن مضمن فراغ __attribute__((always_inline)) scp_wait(متطاير uint32_t * flag, uint32_t expected) {
    طالما (1) {
        evict_to_l2((ثابت فراغ *) flag, 1, 64);
        WAIT_CACHEOPS;
        إذا (*flag >= expected) {
            أرجع;
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
ساكن مضمن فراغ __attribute__((always_inline)) dequant_q4_0_panel(عائم *      panel,
                                                                     ثابت حرف * src0_batch,
                                                                     int64_t      mb,
                                                                     int64_t      kb_block,
                                                                     int64_t      nb1_0) {
    ساكن ثابت int32_t __attribute__((aligned(32))) scatter_idx[8] = {
        0, 64, 128, 192, 256, 320, 384, 448  // byte offsets: 8 lines apart
    };
    ساكن ثابت int32_t __attribute__((aligned(32))) gather_idx[8] = {
        0, 1, 2, 3, 4, 5, 6, 7  // 8 consecutive bytes
    };

    unsigned long old_mask;
    __asm__ متطاير(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"  // all 8 lanes active
        "flw.ps    f1, (%[sidx])    \n\t"  // f1 = scatter offsets
        "flw.ps    f2, (%[gidx])    \n\t"  // f2 = gather offsets
        : [ms] "=&r"(old_mask)
        : [sidx] "r"(scatter_idx), [gidx] "r"(gather_idx)
        : "f1", "f2");

    حرف * pbase = (حرف *) panel;
    لكل (صحيح j = 0; j < TILE_M; ++j) {
        ثابت block_q4_0 * blk       = (ثابت block_q4_0 *) (src0_batch + (mb + j) * nb1_0) + kb_block;
        uint32_t           scale_raw = (uint32_t) blk->d;
        ثابت uint8_t *    qs        = blk->qs;
        حرف *             col       = pbase + j * 4;  // column m=j of the panel

        __asm__ متطاير(
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
            : [sb] "r"(scale_raw), [qs0] "r"(qs), [qs8] "r"(qs + 8), [c0] "r"(col), [c8] "r"(col + 8 * 64),
              [c16] "r"(col + 16 * 64), [c24] "r"(col + 24 * 64)
            : "f3", "f4", "f5", "f6", "memory");
    }

    __asm__ متطاير("mova.m.x %0" ::"r"(old_mask));
}

صحيح entry_point(بنية ggml_et_ثنائی_params * params, فراغ * env) {
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

    إذا ((M % TILE_M) != 0) {
        أرجع 0;
    }
    إذا ((K % BLOCK_K) != 0) {
        أرجع 0;
    }

    ثابت int64_t ne2_0 = params->src0.ne[2], ne3_0 = params->src0.ne[3];
    ثابت int64_t ne2_1 = params->src1.ne[2], ne3_1 = params->src1.ne[3];

    ثابت int64_t nb1_0 = params->src0.nb[1];
    ثابت int64_t nb2_0 = params->src0.nb[2], nb3_0 = params->src0.nb[3];

    ثابت int64_t nb1_1 = params->src1.nb[1];
    ثابت int64_t nb2_1 = params->src1.nb[2], nb3_1 = params->src1.nb[3];

    ثابت int64_t nb1_d = params->dst.nb[1];
    ثابت int64_t nb2_d = params->dst.nb[2], nb3_d = params->dst.nb[3];

    ثابت حرف * src0_base = (ثابت حرف *) params->src0.data;
    ثابت حرف * src1_base = (ثابت حرف *) params->src1.data;
    حرف *       dst_base  = (حرف *) params->dst.data;

    ثابت int64_t m_tiles     = M / TILE_M;
    ثابت int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    ثابت int64_t batch_گنتی = ne2_1 * ne3_1;
    ثابت int64_t base_tiles  = m_tiles * n_tiles * batch_گنتی;

    ثابت int64_t r2 = ne2_1 / ne2_0;
    ثابت int64_t r3 = ne3_1 / ne3_0;

    ثابت int64_t k_steps = K / BLOCK_K;  // number of Q4_0 blocks

    // Force a single K-split.
    ثابت int64_t k_splits = 1;

    ثابت int64_t tiles_per_shire = MINIONS_PER_SHIRE / k_splits;
    ثابت int64_t k_split         = local_minion % k_splits;
    ثابت int64_t local_tile_idx  = local_minion / k_splits;
    ثابت int64_t tiles_stride    = (int64_t) عدد_COMPUTE_SHIRES * tiles_per_shire;

    ثابت int64_t k_steps_per_split = k_steps / k_splits;
    ثابت int64_t kb_start          = k_split * k_steps_per_split;   // first block
    ثابت int64_t kb_end            = kb_start + k_steps_per_split;  // one past last

    // L2 SCP pointers for this minion's double-buffered panels + sync.
    uint64_t scp_base     = local_minion * SCP_PER_MINION;
    عائم *  scp_panel[2] = {
        (عائم *) et_shire_l2scp_local(scp_base),
        (عائم *) et_shire_l2scp_local(scp_base + SCP_PANEL_SIZE),
    };
    متطاير uint32_t * ready_ctr    = (متطاير uint32_t *) et_shire_l2scp_local(scp_base + SCP_READY_OFF);
    متطاير uint32_t * consumed_ctr = (متطاير uint32_t *) et_shire_l2scp_local(scp_base + SCP_CONSUMED_OFF);

    // ================================================================
    // Hart 1: Q4_0 weight dequant producer
    // ================================================================
    إذا (is_hart1) {
        scp_signal(ready_ctr, 0);
        scp_signal(consumed_ctr, 0);

        uint32_t chunk_id = 0;

        لكل (int64_t tile = (int64_t) shire_id + local_tile_idx * عدد_COMPUTE_SHIRES; tile < base_tiles;
             tile += tiles_stride) {
            ثابت int64_t tiles_per_batch = m_tiles * n_tiles;
            ثابت int64_t batch_idx       = tile / tiles_per_batch;
            ثابت int64_t tile_in_batch   = tile % tiles_per_batch;

            ثابت int64_t mb_idx = tile_in_batch % m_tiles;

            ثابت int64_t i3   = batch_idx / ne2_1;
            ثابت int64_t i2   = batch_idx % ne2_1;
            ثابت int64_t i2_0 = i2 / r2;
            ثابت int64_t i3_0 = i3 / r3;

            ثابت حرف *  src0_batch = src0_base + i3_0 * nb3_0 + i2_0 * nb2_0;
            ثابت int64_t mb         = mb_idx * TILE_M;

            لكل (int64_t kb = kb_start; kb < kb_end; ++kb) {
                صحيح buf = chunk_id & 1;

                // Back-pressure: wait for hart 0 to finish with this buffer.
                إذا (chunk_id >= 2) {
                    scp_wait(consumed_ctr, chunk_id - 1);
                }

                dequant_q4_0_panel(scp_panel[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_to_l2(scp_panel[buf], BLOCK_K, 64);
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
    ثابت uint64_t group_base_global = my_minion_id - k_split;

    setup_cache_scp();
#إذا CACHEOP_MAX > 0 || REP_RATE > 0
    ucache_control(1, REP_RATE, CACHEOP_MAX);
#endif
    CLEAR_TENSOR_ERROR;

    evict_to_l2((ثابت فراغ *) ready_ctr, 1, 64);
    WAIT_CACHEOPS;
    evict_to_l2((ثابت فراغ *) consumed_ctr, 1, 64);
    WAIT_CACHEOPS;

    uint32_t chunk_id = 0;

    لكل (int64_t tile = (int64_t) shire_id + local_tile_idx * عدد_COMPUTE_SHIRES; tile < base_tiles;
         tile += tiles_stride) {
        ثابت int64_t tiles_per_batch = m_tiles * n_tiles;
        ثابت int64_t batch_idx       = tile / tiles_per_batch;
        ثابت int64_t tile_in_batch   = tile % tiles_per_batch;

        ثابت int64_t nb_idx = tile_in_batch / m_tiles;
        ثابت int64_t mb_idx = tile_in_batch % m_tiles;

        ثابت int64_t i3 = batch_idx / ne2_1;
        ثابت int64_t i2 = batch_idx % ne2_1;

        ثابت حرف * src1_batch = src1_base + i3 * nb3_1 + i2 * nb2_1;
        حرف *       dst_batch  = dst_base + i3 * nb3_d + i2 * nb2_d;

        ثابت int64_t mb    = mb_idx * TILE_M;
        ثابت int64_t nb    = nb_idx * TILE_N;
        ثابت int64_t n_cur = (nb + TILE_N <= N) ? TILE_N : (N - nb);

        // Partial-N tiles run TensorFMA32 with a_num_rows = n_cur-1.
        // Errata Type D workaround for n_cur == 4 (AROWS==3): pad A to AROWS==4.
        ثابت int64_t arows_fma = (n_cur == 4) ? 4 : (n_cur - 1);

        إذا (n_cur == 4) {
            // Zero the padded 5th A row (line A_L1_START+4) once; the per-pass A
            // load only writes lines A_L1_START..+3, so this persists.
            ساكن ثابت عائم __attribute__((aligned(64))) zero_سطر[16] = { 0 };
            tensor_load(false, false, A_L1_START + 4, TENSOR_LOAD_PLAIN, 0, (uint64_t) zero_سطر, 0,
                        0,  // 1 line
                        64, 0);
            tensor_wait(TENSOR_LOAD_WAIT_0);
        }

        صحيح first = 1;  // first_pass=1 only for the very first FMA of the tile

        لكل (int64_t kb = kb_start; kb < kb_end; ++kb) {
            صحيح buf = chunk_id & 1;

            // Wait for hart 1 to finish dequantizing this block.
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            // Two FMA passes over the 32-wide block (16 K-cols each).
            لكل (صحيح half = 0; half < 2; ++half) {
                ثابت int64_t k_elem = kb * BLOCK_K + half * FMA_K;

                // Load A (activations) for this 16-K sub-tile, PLAIN.
                tensor_load(false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0,
                            (uint64_t) (src1_batch + nb * nb1_1 + k_elem * (int64_t) حجم(عائم)), 0, n_cur - 1,
                            (uint64_t) nb1_1, 0);

                // Load B (dequantized weights) half from L2 SCP panel, PLAIN.
                tensor_load(false, false, B_L1_START, TENSOR_LOAD_PLAIN, 0,
                            (uint64_t) (scp_panel[buf] + (int64_t) half * FMA_K * TILE_M), 0, FMA_K - 1, 64, 1);

                tensor_wait(TENSOR_LOAD_WAIT_0);
                tensor_wait(TENSOR_LOAD_WAIT_1);

                tensor_fma(false,
                           3,          // b_num_col: (16/4)-1
                           arows_fma,  // a_num_rows (n_cur-1, or 4 for the n_cur==4 errata pad)
                           FMA_K - 1,  // a_num_cols
                           0, false, false, false, false, B_L1_START, A_L1_START, TENSOR_FMA_OP_FP32, first);

                tensor_wait(TENSOR_FMA_WAIT);
                first = 0;
            }

            // Signal that this buffer is free for hart 1 to reuse.
            scp_signal(consumed_ctr, chunk_id);
        }

        // K-split ring reduce.
        إذا (k_splits > 1) {
            ثابت uint64_t عدد_regs = (uint64_t) n_cur * 2;

            إذا (k_split > 0) {
                tensor_reduce_recv(0, TENSOR_REDUCE_OP_FADD, عدد_regs, group_base_global + k_split - 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }

            إذا (k_split < k_splits - 1) {
                tensor_reduce_send(0, عدد_regs, group_base_global + k_split + 1);
                tensor_wait(TENSOR_REDUCE_WAIT);
            }
        }

        // Store FP32 result tile (only the last k-split owns the final sum).
        إذا (k_split == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * (int64_t) حجم(عائم)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    أرجع 0;
}
