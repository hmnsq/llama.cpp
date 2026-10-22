#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "quants.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "tensor.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <etsoc/common/utils.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

// Q4_0 x F32 -> F32 MUL_MAT on the tensor (matrix) engine, TensorFMA32.
// Hart 1: dequantize Q4_0 weights to FP32 into double-buffered L2 SCP.
// Hart 0: tensor engine compute (FMA, reduce, store).

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅੰਕ_COMPUTE_SHIRES 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MINIONS_ਪ੍ਰਤੀ_SHIRE  32

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE_M  16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE_N  16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਖੰਡ_K QK4_0  // 32 elements per Q4_0 block
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FMA_K   16     // tensor FMA k-width for FP32 (a_num_cols = FMA_K-1)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CACHEOP_MAX 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ REP_ਦਰ    0

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ A_L1_ਸ਼ੁਰੂ 0   // L1 SCP lines  0..15 for A (activations)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ B_L1_ਸ਼ੁਰੂ 16  // L1 SCP lines 16..31 for B (dequantized weights)

// L2 SCP layout per minion (double-buffered dequant panel + sync counters).
// panel = BLOCK_K k-lines x TILE_M m (FP32) = 32 * 64 = 2048 bytes, in TenB
// [k][m] order: panel[k*TILE_M + m].
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_PANEL_ਮਾਪ   (ਖੰਡ_K * TILE_M * (uint64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ))  // 2048
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_READY_OFF    (2 * SCP_PANEL_ਮਾਪ)                           // 4096
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_CONSUMED_OFF (SCP_READY_OFF + 64)                           // 4160
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_ਪ੍ਰਤੀ_MINION   (SCP_CONSUMED_OFF + 64)                        // 4224

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
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) dequant_q4_0_panel(ਦਸ਼ਮਲਵ *      panel,
                                                                     ਅਚਲ ਅੱਖਰ * src0_batch,
                                                                     int64_t      mb,
                                                                     int64_t      kb_ਖੰਡ,
                                                                     int64_t      nb1_0) {
    ਸਥਿਰ ਅਚਲ int32_t __attribute__((aligned(32))) scatter_idx[8] = {
        0, 64, 128, 192, 256, 320, 384, 448  // byte offsets: 8 lines apart
    };
    ਸਥਿਰ ਅਚਲ int32_t __attribute__((aligned(32))) gather_idx[8] = {
        0, 1, 2, 3, 4, 5, 6, 7  // 8 consecutive bytes
    };

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ old_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"  // all 8 lanes active
        "flw.ps    f1, (%[sidx])    \n\t"  // f1 = scatter offsets
        "flw.ps    f2, (%[gidx])    \n\t"  // f2 = gather offsets
        : [ms] "=&r"(old_ਨਕਾਬ)
        : [sidx] "r"(scatter_idx), [gidx] "r"(gather_idx)
        : "f1", "f2");

    ਅੱਖਰ * pbase = (ਅੱਖਰ *) panel;
    ਲਈ (ਪੂਰਨ j = 0; j < TILE_M; ++j) {
        ਅਚਲ ਖੰਡ_q4_0 * blk       = (ਅਚਲ ਖੰਡ_q4_0 *) (src0_batch + (mb + j) * nb1_0) + kb_ਖੰਡ;
        uint32_t           scale_raw = (uint32_t) blk->d;
        ਅਚਲ uint8_t *    qs        = blk->qs;
        ਅੱਖਰ *             ਖਾਨਾ       = pbase + j * 4;  // column m=j of the panel

        __asm__ ਅਸਥਿਰ(
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
            : [sb] "r"(scale_raw), [qs0] "r"(qs), [qs8] "r"(qs + 8), [c0] "r"(ਖਾਨਾ), [c8] "r"(ਖਾਨਾ + 8 * 64),
              [c16] "r"(ਖਾਨਾ + 16 * 64), [c24] "r"(ਖਾਨਾ + 24 * 64)
            : "f3", "f4", "f5", "f6", "memory");
    }

    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(old_ਨਕਾਬ));
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

    ਜੇ ((M % TILE_M) != 0) {
        ਵਾਪਸ 0;
    }
    ਜੇ ((K % ਖੰਡ_K) != 0) {
        ਵਾਪਸ 0;
    }

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

    ਅਚਲ int64_t m_tiles     = M / TILE_M;
    ਅਚਲ int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    ਅਚਲ int64_t batch_ਗਿਣਤੀ = ne2_1 * ne3_1;
    ਅਚਲ int64_t ਅਧਾਰ_tiles  = m_tiles * n_tiles * batch_ਗਿਣਤੀ;

    ਅਚਲ int64_t r2 = ne2_1 / ne2_0;
    ਅਚਲ int64_t r3 = ne3_1 / ne3_0;

    ਅਚਲ int64_t k_ਕਦਮਾਂ = K / ਖੰਡ_K;  // number of Q4_0 blocks

    // Force a single K-split.
    ਅਚਲ int64_t k_splits = 1;

    ਅਚਲ int64_t tiles_ਪ੍ਰਤੀ_shire = MINIONS_ਪ੍ਰਤੀ_SHIRE / k_splits;
    ਅਚਲ int64_t k_ਵੰਡੋ         = local_minion % k_splits;
    ਅਚਲ int64_t local_tile_idx  = local_minion / k_splits;
    ਅਚਲ int64_t tiles_stride    = (int64_t) ਅੰਕ_COMPUTE_SHIRES * tiles_ਪ੍ਰਤੀ_shire;

    ਅਚਲ int64_t k_ਕਦਮਾਂ_ਪ੍ਰਤੀ_ਵੰਡੋ = k_ਕਦਮਾਂ / k_splits;
    ਅਚਲ int64_t kb_ਸ਼ੁਰੂ          = k_ਵੰਡੋ * k_ਕਦਮਾਂ_ਪ੍ਰਤੀ_ਵੰਡੋ;   // first block
    ਅਚਲ int64_t kb_ਸਿਰਾ            = kb_ਸ਼ੁਰੂ + k_ਕਦਮਾਂ_ਪ੍ਰਤੀ_ਵੰਡੋ;  // one past last

    // L2 SCP pointers for this minion's double-buffered panels + sync.
    uint64_t scp_ਅਧਾਰ     = local_minion * SCP_ਪ੍ਰਤੀ_MINION;
    ਦਸ਼ਮਲਵ *  scp_panel[2] = {
        (ਦਸ਼ਮਲਵ *) et_shire_l2scp_local(scp_ਅਧਾਰ),
        (ਦਸ਼ਮਲਵ *) et_shire_l2scp_local(scp_ਅਧਾਰ + SCP_PANEL_ਮਾਪ),
    };
    ਅਸਥਿਰ uint32_t * ready_ctr    = (ਅਸਥਿਰ uint32_t *) et_shire_l2scp_local(scp_ਅਧਾਰ + SCP_READY_OFF);
    ਅਸਥਿਰ uint32_t * consumed_ctr = (ਅਸਥਿਰ uint32_t *) et_shire_l2scp_local(scp_ਅਧਾਰ + SCP_CONSUMED_OFF);

    // ================================================================
    // Hart 1: Q4_0 weight dequant producer
    // ================================================================
    ਜੇ (is_hart1) {
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

            ਲਈ (int64_t kb = kb_ਸ਼ੁਰੂ; kb < kb_ਸਿਰਾ; ++kb) {
                ਪੂਰਨ buf = chunk_id & 1;

                // Back-pressure: wait for hart 0 to finish with this buffer.
                ਜੇ (chunk_id >= 2) {
                    scp_wait(consumed_ctr, chunk_id - 1);
                }

                dequant_q4_0_panel(scp_panel[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_ਮੰਜ਼ਿਲ_l2(scp_panel[buf], ਖੰਡ_K, 64);
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

        // Partial-N tiles run TensorFMA32 with a_num_rows = n_cur-1.
        // Errata Type D workaround for n_cur == 4 (AROWS==3): pad A to AROWS==4.
        ਅਚਲ int64_t arows_fma = (n_cur == 4) ? 4 : (n_cur - 1);

        ਜੇ (n_cur == 4) {
            // Zero the padded 5th A row (line A_L1_START+4) once; the per-pass A
            // load only writes lines A_L1_START..+3, so this persists.
            ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ __attribute__((aligned(64))) zero_ਸਤਰ[16] = { 0 };
            tensor_ਲੱਦੋ(false, false, A_L1_ਸ਼ੁਰੂ + 4, TENSOR_ਲੱਦੋ_PLAIN, 0, (uint64_t) zero_ਸਤਰ, 0,
                        0,  // 1 line
                        64, 0);
            tensor_wait(TENSOR_ਲੱਦੋ_WAIT_0);
        }

        ਪੂਰਨ ਪਹਿਲਾ = 1;  // first_pass=1 only for the very first FMA of the tile

        ਲਈ (int64_t kb = kb_ਸ਼ੁਰੂ; kb < kb_ਸਿਰਾ; ++kb) {
            ਪੂਰਨ buf = chunk_id & 1;

            // Wait for hart 1 to finish dequantizing this block.
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            // Two FMA passes over the 32-wide block (16 K-cols each).
            ਲਈ (ਪੂਰਨ half = 0; half < 2; ++half) {
                ਅਚਲ int64_t k_elem = kb * ਖੰਡ_K + half * FMA_K;

                // Load A (activations) for this 16-K sub-tile, PLAIN.
                tensor_ਲੱਦੋ(false, false, A_L1_ਸ਼ੁਰੂ, TENSOR_ਲੱਦੋ_PLAIN, 0,
                            (uint64_t) (src1_batch + nb * nb1_1 + k_elem * (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ)), 0, n_cur - 1,
                            (uint64_t) nb1_1, 0);

                // Load B (dequantized weights) half from L2 SCP panel, PLAIN.
                tensor_ਲੱਦੋ(false, false, B_L1_ਸ਼ੁਰੂ, TENSOR_ਲੱਦੋ_PLAIN, 0,
                            (uint64_t) (scp_panel[buf] + (int64_t) half * FMA_K * TILE_M), 0, FMA_K - 1, 64, 1);

                tensor_wait(TENSOR_ਲੱਦੋ_WAIT_0);
                tensor_wait(TENSOR_ਲੱਦੋ_WAIT_1);

                tensor_fma(false,
                           3,          // b_num_col: (16/4)-1
                           arows_fma,  // a_num_rows (n_cur-1, or 4 for the n_cur==4 errata pad)
                           FMA_K - 1,  // a_num_cols
                           0, false, false, false, false, B_L1_ਸ਼ੁਰੂ, A_L1_ਸ਼ੁਰੂ, TENSOR_FMA_OP_FP32, ਪਹਿਲਾ);

                tensor_wait(TENSOR_FMA_WAIT);
                ਪਹਿਲਾ = 0;
            }

            // Signal that this buffer is free for hart 1 to reuse.
            scp_signal(consumed_ctr, chunk_id);
        }

        // K-split ring reduce.
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

        // Store FP32 result tile (only the last k-split owns the final sum).
        ਜੇ (k_ਵੰਡੋ == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    ਵਾਪਸ 0;
}
