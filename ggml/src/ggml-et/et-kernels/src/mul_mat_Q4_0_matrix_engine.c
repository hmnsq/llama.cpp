#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"
#ಅನ್ತರ್ಭಾವ "quants.h"
#ಅನ್ತರ್ಭಾವ "tensor.h"

#ಅನ್ತರ್ಭಾವ <etsoc/common/utils.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

// Q4_0 x F32 -> F32 MUL_MAT on the tensor (matrix) engine, TensorFMA32.
// Hart 1: dequantize Q4_0 weights to FP32 into double-buffered L2 SCP.
// Hart 0: tensor engine compute (FMA, reduce, store).

#ವ್ಯಾಖ್ಯಾನಿಸು ಅಂಕೆ_COMPUTE_SHIRES 32
#ವ್ಯಾಖ್ಯಾನಿಸು MINIONS_ಪ್ರತಿ_SHIRE  32

#ವ್ಯಾಖ್ಯಾನಿಸು TILE_M  16
#ವ್ಯಾಖ್ಯಾನಿಸು TILE_N  16
#ವ್ಯಾಖ್ಯಾನಿಸು ಖಂಡ_K QK4_0  // 32 elements per Q4_0 block
#ವ್ಯಾಖ್ಯಾನಿಸು FMA_K   16     // tensor FMA k-width for FP32 (a_num_cols = FMA_K-1)

#ವ್ಯಾಖ್ಯಾನಿಸು CACHEOP_MAX 0
#ವ್ಯಾಖ್ಯಾನಿಸು REP_ದರ    0

#ವ್ಯಾಖ್ಯಾನಿಸು A_L1_ಆರಂಭ 0   // L1 SCP lines  0..15 for A (activations)
#ವ್ಯಾಖ್ಯಾನಿಸು B_L1_ಆರಂಭ 16  // L1 SCP lines 16..31 for B (dequantized weights)

// L2 SCP layout per minion (double-buffered dequant panel + sync counters).
// panel = BLOCK_K k-lines x TILE_M m (FP32) = 32 * 64 = 2048 bytes, in TenB
// [k][m] order: panel[k*TILE_M + m].
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_PANEL_ಗಾತ್ರ   (ಖಂಡ_K * TILE_M * (uint64_t) ಗಾತ್ರ(ದಶಮಾಂಶ))  // 2048
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_READY_OFF    (2 * SCP_PANEL_ಗಾತ್ರ)                           // 4096
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_CONSUMED_OFF (SCP_READY_OFF + 64)                           // 4160
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_ಪ್ರತಿ_MINION   (SCP_CONSUMED_OFF + 64)                        // 4224

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
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __attribute__((always_inline)) dequant_q4_0_panel(ದಶಮಾಂಶ *      panel,
                                                                     ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * src0_batch,
                                                                     int64_t      mb,
                                                                     int64_t      kb_ಖಂಡ,
                                                                     int64_t      nb1_0) {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ int32_t __attribute__((aligned(32))) scatter_idx[8] = {
        0, 64, 128, 192, 256, 320, 384, 448  // byte offsets: 8 lines apart
    };
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ int32_t __attribute__((aligned(32))) gather_idx[8] = {
        0, 1, 2, 3, 4, 5, 6, 7  // 8 consecutive bytes
    };

    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ old_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"  // all 8 lanes active
        "flw.ps    f1, (%[sidx])    \n\t"  // f1 = scatter offsets
        "flw.ps    f2, (%[gidx])    \n\t"  // f2 = gather offsets
        : [ms] "=&r"(old_ಮುಖವಾಡ)
        : [sidx] "r"(scatter_idx), [gidx] "r"(gather_idx)
        : "f1", "f2");

    ಅಕ್ಷರ * pbase = (ಅಕ್ಷರ *) panel;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < TILE_M; ++j) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * blk       = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) (src0_batch + (mb + j) * nb1_0) + kb_ಖಂಡ;
        uint32_t           scale_raw = (uint32_t) blk->d;
        ಸ್ಥಿರಾಂಕ uint8_t *    qs        = blk->qs;
        ಅಕ್ಷರ *             ಲಂಬಸಾಲು       = pbase + j * 4;  // column m=j of the panel

        __asm__ ಅಸ್ಥಿರ(
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
            : [sb] "r"(scale_raw), [qs0] "r"(qs), [qs8] "r"(qs + 8), [c0] "r"(ಲಂಬಸಾಲು), [c8] "r"(ಲಂಬಸಾಲು + 8 * 64),
              [c16] "r"(ಲಂಬಸಾಲು + 16 * 64), [c24] "r"(ಲಂಬಸಾಲು + 24 * 64)
            : "f3", "f4", "f5", "f6", "memory");
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(old_ಮುಖವಾಡ));
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

    ಆದರೆ ((M % TILE_M) != 0) {
        ಹಿಂತಿರುಗಿ 0;
    }
    ಆದರೆ ((K % ಖಂಡ_K) != 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

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

    ಸ್ಥಿರಾಂಕ int64_t m_tiles     = M / TILE_M;
    ಸ್ಥಿರಾಂಕ int64_t n_tiles     = (N + TILE_N - 1) / TILE_N;
    ಸ್ಥಿರಾಂಕ int64_t batch_ಎಣಿಕೆ = ne2_1 * ne3_1;
    ಸ್ಥಿರಾಂಕ int64_t ತಳ_tiles  = m_tiles * n_tiles * batch_ಎಣಿಕೆ;

    ಸ್ಥಿರಾಂಕ int64_t r2 = ne2_1 / ne2_0;
    ಸ್ಥಿರಾಂಕ int64_t r3 = ne3_1 / ne3_0;

    ಸ್ಥಿರಾಂಕ int64_t k_ಹೆಜ್ಜೆಗಳು = K / ಖಂಡ_K;  // number of Q4_0 blocks

    // Force a single K-split.
    ಸ್ಥಿರಾಂಕ int64_t k_splits = 1;

    ಸ್ಥಿರಾಂಕ int64_t tiles_ಪ್ರತಿ_shire = MINIONS_ಪ್ರತಿ_SHIRE / k_splits;
    ಸ್ಥಿರಾಂಕ int64_t k_ವಿಭಜಿಸು         = local_minion % k_splits;
    ಸ್ಥಿರಾಂಕ int64_t local_tile_idx  = local_minion / k_splits;
    ಸ್ಥಿರಾಂಕ int64_t tiles_stride    = (int64_t) ಅಂಕೆ_COMPUTE_SHIRES * tiles_ಪ್ರತಿ_shire;

    ಸ್ಥಿರಾಂಕ int64_t k_ಹೆಜ್ಜೆಗಳು_ಪ್ರತಿ_ವಿಭಜಿಸು = k_ಹೆಜ್ಜೆಗಳು / k_splits;
    ಸ್ಥಿರಾಂಕ int64_t kb_ಆರಂಭ          = k_ವಿಭಜಿಸು * k_ಹೆಜ್ಜೆಗಳು_ಪ್ರತಿ_ವಿಭಜಿಸು;   // first block
    ಸ್ಥಿರಾಂಕ int64_t kb_ಕೊನೆ            = kb_ಆರಂಭ + k_ಹೆಜ್ಜೆಗಳು_ಪ್ರತಿ_ವಿಭಜಿಸು;  // one past last

    // L2 SCP pointers for this minion's double-buffered panels + sync.
    uint64_t scp_ತಳ     = local_minion * SCP_ಪ್ರತಿ_MINION;
    ದಶಮಾಂಶ *  scp_panel[2] = {
        (ದಶಮಾಂಶ *) et_shire_l2scp_local(scp_ತಳ),
        (ದಶಮಾಂಶ *) et_shire_l2scp_local(scp_ತಳ + SCP_PANEL_ಗಾತ್ರ),
    };
    ಅಸ್ಥಿರ uint32_t * ready_ctr    = (ಅಸ್ಥಿರ uint32_t *) et_shire_l2scp_local(scp_ತಳ + SCP_READY_OFF);
    ಅಸ್ಥಿರ uint32_t * consumed_ctr = (ಅಸ್ಥಿರ uint32_t *) et_shire_l2scp_local(scp_ತಳ + SCP_CONSUMED_OFF);

    // ================================================================
    // Hart 1: Q4_0 weight dequant producer
    // ================================================================
    ಆದರೆ (is_hart1) {
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

            ಗಾಗಿ (int64_t kb = kb_ಆರಂಭ; kb < kb_ಕೊನೆ; ++kb) {
                ಪೂರ್ಣಾಂಕ buf = chunk_id & 1;

                // Back-pressure: wait for hart 0 to finish with this buffer.
                ಆದರೆ (chunk_id >= 2) {
                    scp_wait(consumed_ctr, chunk_id - 1);
                }

                dequant_q4_0_panel(scp_panel[buf], src0_batch, mb, kb, nb1_0);

                FENCE;
                flush_ಗಮ್ಯ_l2(scp_panel[buf], ಖಂಡ_K, 64);
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

        // Partial-N tiles run TensorFMA32 with a_num_rows = n_cur-1.
        // Errata Type D workaround for n_cur == 4 (AROWS==3): pad A to AROWS==4.
        ಸ್ಥಿರಾಂಕ int64_t arows_fma = (n_cur == 4) ? 4 : (n_cur - 1);

        ಆದರೆ (n_cur == 4) {
            // Zero the padded 5th A row (line A_L1_START+4) once; the per-pass A
            // load only writes lines A_L1_START..+3, so this persists.
            ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ __attribute__((aligned(64))) zero_ಸಾಲು[16] = { 0 };
            tensor_ತುಂಬು(false, false, A_L1_ಆರಂಭ + 4, TENSOR_ತುಂಬು_PLAIN, 0, (uint64_t) zero_ಸಾಲು, 0,
                        0,  // 1 line
                        64, 0);
            tensor_wait(TENSOR_ತುಂಬು_WAIT_0);
        }

        ಪೂರ್ಣಾಂಕ ಮೊದಲ = 1;  // first_pass=1 only for the very first FMA of the tile

        ಗಾಗಿ (int64_t kb = kb_ಆರಂಭ; kb < kb_ಕೊನೆ; ++kb) {
            ಪೂರ್ಣಾಂಕ buf = chunk_id & 1;

            // Wait for hart 1 to finish dequantizing this block.
            chunk_id++;
            scp_wait(ready_ctr, chunk_id);

            // Two FMA passes over the 32-wide block (16 K-cols each).
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ half = 0; half < 2; ++half) {
                ಸ್ಥಿರಾಂಕ int64_t k_elem = kb * ಖಂಡ_K + half * FMA_K;

                // Load A (activations) for this 16-K sub-tile, PLAIN.
                tensor_ತುಂಬು(false, false, A_L1_ಆರಂಭ, TENSOR_ತುಂಬು_PLAIN, 0,
                            (uint64_t) (src1_batch + nb * nb1_1 + k_elem * (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ)), 0, n_cur - 1,
                            (uint64_t) nb1_1, 0);

                // Load B (dequantized weights) half from L2 SCP panel, PLAIN.
                tensor_ತುಂಬು(false, false, B_L1_ಆರಂಭ, TENSOR_ತುಂಬು_PLAIN, 0,
                            (uint64_t) (scp_panel[buf] + (int64_t) half * FMA_K * TILE_M), 0, FMA_K - 1, 64, 1);

                tensor_wait(TENSOR_ತುಂಬು_WAIT_0);
                tensor_wait(TENSOR_ತುಂಬು_WAIT_1);

                tensor_fma(false,
                           3,          // b_num_col: (16/4)-1
                           arows_fma,  // a_num_rows (n_cur-1, or 4 for the n_cur==4 errata pad)
                           FMA_K - 1,  // a_num_cols
                           0, false, false, false, false, B_L1_ಆರಂಭ, A_L1_ಆರಂಭ, TENSOR_FMA_OP_FP32, ಮೊದಲ);

                tensor_wait(TENSOR_FMA_WAIT);
                ಮೊದಲ = 0;
            }

            // Signal that this buffer is free for hart 1 to reuse.
            scp_signal(consumed_ctr, chunk_id);
        }

        // K-split ring reduce.
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

        // Store FP32 result tile (only the last k-split owns the final sum).
        ಆದರೆ (k_ವಿಭಜಿಸು == k_splits - 1) {
            tensor_store(0, 0, 3, n_cur - 1, (uint64_t) (dst_batch + nb * nb1_d + mb * (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ)), 0,
                         (uint64_t) nb1_d);
            tensor_wait(TENSOR_STORE_WAIT);
        }
    }

    FENCE;
    ಹಿಂತಿರುಗಿ 0;
}
