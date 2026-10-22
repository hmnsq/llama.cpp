//******************************************************************************
// Flash Attention with TensorFMA16A32 for QK^T
//
// Uses the matrix engine for the QK^T dot products (F16×F16→F32),
// scalar code for online softmax and V accumulation.
//
// Hart 0: tensor engine (Q load, K load from SCP, FMA, softmax, V accum)
// Hart 1: pack K into double-buffered L2 SCP panels, flush for tensor_load
//
// Requirements:
//   - Q: F32 (converted to F16 internally)
//   - K, V: F16
//   - dk must be a multiple of 32 (TensorFMA16A32 K-tile)
//   - dv ≤ 512 (accumulator in shire-local L2 SCP)
//
// Parallelization: each minion independently processes one (qpos, head, batch)
// row, round-robin across all minion hart-0s. Hart 1 assists with K packing.
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"
#ಅನ್ತರ್ಭಾವ "tensor.h"

#ಅನ್ತರ್ಭಾವ <etsoc/common/utils.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <string.h>

#ವ್ಯಾಖ್ಯಾನಿಸು ಅಂಕೆ_COMPUTE_SHIRES 32
#ವ್ಯಾಖ್ಯಾನಿಸು MINIONS_ಪ್ರತಿ_SHIRE  32

// QK^T tiles: 16 KV positions at a time, K in chunks of 32 F16
#ವ್ಯಾಖ್ಯಾನಿಸು TILE_KV 16
#ವ್ಯಾಖ್ಯಾನಿಸು TILE_K  32

// L1 scratchpad layout: A (Q) in lines 0-15, B (K interleaved) in lines 16-31
#ವ್ಯಾಖ್ಯಾನಿಸು A_L1_ಆರಂಭ 0
#ವ್ಯಾಖ್ಯಾನಿಸು B_L1_ಆರಂಭ 16

// Max head dimensions
#ವ್ಯಾಖ್ಯಾನಿಸು FA_DV_MAX 512  // max value head dim (dv)
#ವ್ಯಾಖ್ಯಾನಿಸು FA_DK_MAX 512  // max key head dim (dk) - some models use hsk > hsv

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ uint16_t et_fp16_t;

#ವ್ಯಾಖ್ಯಾನಿಸು ET_NEG_INF_F (-3.402823466e+38f)

// L2 SCP layout per minion:
//   [0..2047]       accumulator (FA_DV_MAX * sizeof(float))
//   [2048..4095]    kpanel buffer 0 (32 × 32 × 2 = 2048 bytes)
//   [4096..6143]    kpanel buffer 1 (2048 bytes)
//   [6144..6207]    stats line - (M_p at +0, S_p at +4), own cache line
// Double-buffering ensures hart 0 finishes buf[N%2] before hart 1
// overwrites it at chunk N+2.
//
// The stats line reserves a cache-line-aligned slot for split-KV softmax
// partials (M_p, S_p). With k_splits=1 the slot is currently unused; step 2
// will populate it and use peer minions' slots during the reduction.
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_ACC_OFF     0
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_ACC_STRIDE  (FA_DV_MAX * ಗಾತ್ರ(ದಶಮಾಂಶ))       // 2048
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_KPANEL_ಗಾತ್ರ (32 * 32 * ಗಾತ್ರ(et_fp16_t))     // 2048
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_KP0_OFF     SCP_ACC_STRIDE                    // 2048
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_KP1_OFF     (SCP_KP0_OFF + SCP_KPANEL_ಗಾತ್ರ)   // 4096
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_STATS_OFF   (SCP_KP1_OFF + SCP_KPANEL_ಗಾತ್ರ)   // 6144
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_STATS_ಗಾತ್ರ  64                                // own cache line
#ವ್ಯಾಖ್ಯಾನಿಸು SCP_ಪ್ರತಿ_MINION  (SCP_STATS_OFF + SCP_STATS_ಗಾತ್ರ)  // 6208

ರಚನೆ ggml_et_flash_attn_ext_params {
    ರಚನೆ ggml_tensor src0;  // Q (F32)
    ರಚನೆ ggml_tensor src1;  // K (F16)
    ರಚನೆ ggml_tensor src2;  // V (F16)
    ರಚನೆ ggml_tensor ಮುಖವಾಡ;  // mask (F16 or F32), zeroed when absent
    ರಚನೆ ggml_tensor dst;   // Output (F32)
    ದಶಮಾಂಶ              scale;
    int32_t            has_ಮುಖವಾಡ;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ get_ಮುಖವಾಡ_val(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಮುಖವಾಡ, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ತಳ = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ಮುಖವಾಡ->ದತ್ತಾಂಶ + iq1 * ಮುಖವಾಡ->nb[1] + (iq2 % ಮುಖವಾಡ->ne[2]) * ಮುಖವಾಡ->nb[2] +
                        (iq3 % ಮುಖವಾಡ->ne[3]) * ಮುಖವಾಡ->nb[3];

    ಆದರೆ (ಮುಖವಾಡ->type == GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (ತಳ + ik1 * ಮುಖವಾಡ->nb[0]);
    }
    ಹಿಂತಿರುಗಿ fp16_ಗಮ್ಯ_fp32(*(ಸ್ಥಿರಾಂಕ uint16_t *) (ತಳ + ik1 * ಮುಖವಾಡ->nb[0]));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * get_ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು_ತಳ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಮುಖವಾಡ, int64_t iq1, int64_t iq2, int64_t iq3) {
    ಹಿಂತಿರುಗಿ (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) ಮುಖವಾಡ->ದತ್ತಾಂಶ + iq1 * ಮುಖವಾಡ->nb[1] + (iq2 % ಮುಖವಾಡ->ne[2]) * ಮುಖವಾಡ->nb[2] +
           (iq3 % ಮುಖವಾಡ->ne[3]) * ಮುಖವಾಡ->nb[3];
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ get_ಮುಖವಾಡ_val_from_ತಳ(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * ಮುಖವಾಡ, ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ತಳ, int64_t ik1) {
    ಆದರೆ (ಮುಖವಾಡ->type == GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ *(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (ತಳ + ik1 * ಮುಖವಾಡ->nb[0]);
    }
    ಹಿಂತಿರುಗಿ fp16_ಗಮ್ಯ_fp32(*(ಸ್ಥಿರಾಂಕ uint16_t *) (ತಳ + ik1 * ಮುಖವಾಡ->nb[0]));
}

// Pack K rows for TensorLoadTranspose16 (even/odd deinterleave)
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __attribute__((always_inline)) pack_k_for_transpose16(et_fp16_t *  ಹೊರಹೋಗು,
                                                                         ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * k_ತಳ,
                                                                         int64_t      kv_ಆರಂಭ,
                                                                         int64_t      dk_ಆರಂಭ,
                                                                         int64_t      kv_ಎಣಿಕೆ,
                                                                         int64_t      nb1_k) {
    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ old_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        : [ms] "=&r"(old_ಮುಖವಾಡ)
        :
        :);

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < (ಪೂರ್ಣಾಂಕ) kv_ಎಣಿಕೆ; ++j) {
        ಸ್ಥಿರಾಂಕ et_fp16_t * k_ಅಡ್ಡಸಾಲು    = (ಸ್ಥಿರಾಂಕ et_fp16_t *) (k_ತಳ + (kv_ಆರಂಭ + j) * nb1_k) + dk_ಆರಂಭ;
        et_fp16_t *       even_ಅಡ್ಡಸಾಲು = ಹೊರಹೋಗು + (j * 2) * 32;
        et_fp16_t *       odd_ಅಡ್ಡಸಾಲು  = ಹೊರಹೋಗು + (j * 2 + 1) * 32;
        __asm__ ಅಸ್ಥಿರ(
            "flw.ps    f2, 0(%[src0])  \n\t"  // load row[0..15]
            "flw.ps    f3, 0(%[src1])  \n\t"  // load row[16..31]
            "fpackreph.pi f4, f2       \n\t"  // even_lo from src0
            "fpackreph.pi f6, f3       \n\t"  // even_lo from src1 (interleaved)
            "fsrli.pi  f5, f2, 16      \n\t"  // shift src0 for odd
            "fsrli.pi  f7, f3, 16      \n\t"  // shift src1 for odd (interleaved)
            "fpackreph.pi f5, f5       \n\t"  // odd from src0
            "fpackreph.pi f7, f7       \n\t"  // odd from src1
            "mov.m.x   m0, x0, 0x0F   \n\t"
            "fcmovm.ps f4, f4, f6      \n\t"  // merge even halves
            "fcmovm.ps f5, f5, f7      \n\t"  // merge odd halves
            "mov.m.x   m0, x0, 0xFF   \n\t"
            "fsw.ps    f4, 0(%[even])  \n\t"
            "fsw.ps    f5, 0(%[odd])   \n\t"
            :
            : [src0] "r"(k_ಅಡ್ಡಸಾಲು), [src1] "r"(k_ಅಡ್ಡಸಾಲು + 16), [even] "r"(even_ಅಡ್ಡಸಾಲು), [odd] "r"(odd_ಅಡ್ಡಸಾಲು)
            : "f2", "f3", "f4", "f5", "f6", "f7", "memory");
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_ಮುಖವಾಡ));

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = (ಪೂರ್ಣಾಂಕ) kv_ಎಣಿಕೆ; j < TILE_KV; ++j) {
        et_fp16_t * even_ಅಡ್ಡಸಾಲು = ಹೊರಹೋಗು + (j * 2) * 32;
        et_fp16_t * odd_ಅಡ್ಡಸಾಲು  = ಹೊರಹೋಗು + (j * 2 + 1) * 32;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ l = 0; l < TILE_K / 2; ++l) {
            even_ಅಡ್ಡಸಾಲು[l] = 0;
            odd_ಅಡ್ಡಸಾಲು[l]  = 0;
        }
    }
}

// Build interleaved B panel for TensorFMA16A32 (weights @ V).
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __attribute__((always_inline)) pack_v_interleaved(et_fp16_t *  ಹೊರಹೋಗು,
                                                                     ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * v_ತಲೆ,
                                                                     int64_t      kv_ತಳ,
                                                                     int64_t      dv_ಆರಂಭ,
                                                                     int64_t      kv_ಎಣಿಕೆ,
                                                                     int64_t      nb1_v) {
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < TILE_KV; ++k) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ         l   = k >> 1;
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ         r   = k & 1;
        et_fp16_t * ಸ್ಥಿರಾಂಕ dst = ಹೊರಹೋಗು + l * 32 + r;
        ಆದರೆ (k < (ಪೂರ್ಣಾಂಕ) kv_ಎಣಿಕೆ) {
            ಸ್ಥಿರಾಂಕ et_fp16_t * v_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ et_fp16_t *) (v_ತಲೆ + (kv_ತಳ + k) * nb1_v) + dv_ಆರಂಭ;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 16; ++n) {
                dst[n * 2] = v_ಅಡ್ಡಸಾಲು[n];
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ n = 0; n < 16; ++n) {
                dst[n * 2] = 0;
            }
        }
    }
}

// Prefetch KV rows for one chunk into L2.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __attribute__((always_inline)) prefetch_kv_ಗಮ್ಯ_l2(ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ತಲೆ,
                                                                    int64_t      kv_ಆರಂಭ,
                                                                    int64_t      d_ಆರಂಭ,
                                                                    int64_t      kv_ಎಣಿಕೆ,
                                                                    int64_t      nb1) {
    ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * ತಳ = (ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) (ತಲೆ + kv_ಆರಂಭ * nb1 + d_ಆರಂಭ * 2);
    l2_prefetch(ತಳ, (uint64_t) kv_ಎಣಿಕೆ, (uint64_t) nb1);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __attribute__((always_inline)) convert_q_ಅಡ್ಡಸಾಲು_f32_ಗಮ್ಯ_f16(et_fp16_t *   dst,
                                                                           ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src,
                                                                           int64_t       n) {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ int32_t __attribute__((aligned(32))) offsets[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };

    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ old_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 0xFF      \n\t"
        "flw.ps    f1, 0(%[offs])    \n\t"
        : [ms] "=&r"(old_ಮುಖವಾಡ)
        : [offs] "r"(offsets)
        : "f1");

    ಗಾಗಿ (int64_t d = 0; d < n; d += 8) {
        __asm__ ಅಸ್ಥಿರ(
            "flw.ps      f2, 0(%[src])    \n\t"
            "fcvt.f16.ps f3, f2           \n\t"
            "fsch.ps     f3, f1(%[dst])   \n\t"
            :
            : [src] "r"(src + d), [dst] "r"(dst + d)
            : "f2", "f3", "memory");
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x  %[ms]             \n\t" : : [ms] "r"(old_ಮುಖವಾಡ));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __attribute__((always_inline)) zero_acc_vec(ದಶಮಾಂಶ * acc, int64_t dv) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   zero = 0.0f;
    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ old_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(old_ಮುಖವಾಡ));
    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");
    __asm__ ಅಸ್ಥಿರ("fbc.ps  f2, 0(%[z])" ::[z] "r"(&zero) : "f2");

    ಗಾಗಿ (int64_t d = 0; d < dv; d += 8) {
        __asm__ ಅಸ್ಥಿರ("fsw.ps  f2, 0(%[a])     \n\t" ::[a] "r"(acc + d) : "f2", "memory");
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(old_ಮುಖವಾಡ));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __attribute__((always_inline)) scale_acc_vec(ದಶಮಾಂಶ * acc, int64_t dv, ದಶಮಾಂಶ scale) {
    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ old_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(old_ಮುಖವಾಡ));
    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");

    ಗಾಗಿ (int64_t d = 0; d < dv; d += 8) {
        __asm__ ಅಸ್ಥಿರ(
            "fbc.ps    f2, 0(%[s])    \n\t"
            "flw.ps    f3, 0(%[a])    \n\t"
            "fmul.ps   f3, f3, f2     \n\t"
            "fsw.ps    f3, 0(%[a])    \n\t"
            :
            : [s] "r"(&scale), [a] "r"(acc + d)
            : "f2", "f3", "memory");
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(old_ಮುಖವಾಡ));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __attribute__((always_inline)) normalize_store_vec(ದಶಮಾಂಶ * ಹೊರಹೋಗು,
                                                                      ದಶಮಾಂಶ * acc,
                                                                      int64_t dv,
                                                                      ದಶಮಾಂಶ   inv,
                                                                      ಪೂರ್ಣಾಂಕ     use_fast_store) {
    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ old_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(old_ಮುಖವಾಡ));
    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");

    ಗಾಗಿ (int64_t d = 0; d < dv; d += 8) {
        __asm__ ಅಸ್ಥಿರ(
            "fbc.ps    f2, 0(%[inv])   \n\t"
            "flw.ps    f3, 0(%[a])     \n\t"
            "fmul.ps   f3, f3, f2      \n\t"
            "fsw.ps    f3, 0(%[a])     \n\t"
            :
            : [inv] "r"(&inv), [a] "r"(acc + d)
            : "f2", "f3", "memory");
        ಆದರೆ (use_fast_store) {
            __asm__ ಅಸ್ಥಿರ(
                "flw.ps  f4, 0(%[a])     \n\t"
                "fsw.ps  f4, 0(%[o])     \n\t"
                :
                : [a] "r"(acc + d), [o] "r"(ಹೊರಹೋಗು + d)
                : "f4", "memory");
        } ಇಲ್ಲದಿದ್ದರೆ {
            atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) &ಹೊರಹೋಗು[d + 0], acc[d + 0]);
            atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) &ಹೊರಹೋಗು[d + 1], acc[d + 1]);
            atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) &ಹೊರಹೋಗು[d + 2], acc[d + 2]);
            atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) &ಹೊರಹೋಗು[d + 3], acc[d + 3]);
            atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) &ಹೊರಹೋಗು[d + 4], acc[d + 4]);
            atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) &ಹೊರಹೋಗು[d + 5], acc[d + 5]);
            atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) &ಹೊರಹೋಗು[d + 6], acc[d + 6]);
            atomic_store_f32((ಅಸ್ಥಿರ ದಶಮಾಂಶ *) &ಹೊರಹೋಗು[d + 7], acc[d + 7]);
        }
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(old_ಮುಖವಾಡ));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t tensor_ಬೈಟುಗಳು_fa(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

// Evict a byte range from L1D to L2 SCP, splitting into batches of ≤16
// cache lines (the hw limit for evict_to_l2). Use before a barrier when
// another minion in the shire needs to read the region, or after a barrier
// on the reader side to drop stale L1D copies before reading peer data.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __attribute__((always_inline)) evict_range_ಗಮ್ಯ_l2(ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ * addr, int64_t ಬೈಟುಗಳು) {
    ಆದರೆ (ಬೈಟುಗಳು <= 0) {
        ಹಿಂತಿರುಗಿ;
    }
    int64_t      ಸಾಲುಗಳು = (ಬೈಟುಗಳು + 63) / 64;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * p     = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) addr;
    ಆಗ (ಸಾಲುಗಳು > 0) {
        int64_t batch = ಸಾಲುಗಳು > 16 ? 16 : ಸಾಲುಗಳು;
        evict_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) p, (uint64_t) batch, 64);
        p += batch * 64;
        ಸಾಲುಗಳು -= batch;
    }
}

// Split-KV online merge inner loop:
//
//   for d in [0, dv) step 8:
//       acc[d..d+8] = alpha_own * acc[d..d+8] + alpha_peer * peer_acc[d..d+8]
//
// Runs on the reducer (k_split == 0) after all tensor_fma ops for the row are
// complete, so f0..f31 are dead at entry. We still bracket the loop in inline
// asm with explicit f2/f3/f4/f5 clobbers to lock register usage down — per the
// MM register lifetime rule, never let the compiler mingle FP ops into code
// that sits anywhere near a tensor engine output window.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ __attribute__((always_inline)) ವಿಲೀನ_rescale_ಕೂಡಿಸು_asm(ದಶಮಾಂಶ *       acc,
                                                                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * peer_acc,
                                                                        int64_t       dv,
                                                                        ದಶಮಾಂಶ         alpha_own,
                                                                        ದಶಮಾಂಶ         alpha_peer) {
    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ old_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ(
        "mova.x.m  %[ms]              \n\t"
        "mov.m.x   m0, x0, 0xFF       \n\t"
        "fbc.ps    f4, 0(%[ao])       \n\t"  // broadcast alpha_own
        "fbc.ps    f5, 0(%[ap])       \n\t"  // broadcast alpha_peer
        : [ms] "=&r"(old_ಮುಖವಾಡ)
        : [ao] "r"(&alpha_own), [ap] "r"(&alpha_peer)
        : "f4", "f5");

    ಗಾಗಿ (int64_t d = 0; d < dv; d += 8) {
        __asm__ ಅಸ್ಥಿರ(
            "flw.ps    f2, 0(%[a])      \n\t"  // own
            "flw.ps    f3, 0(%[p])      \n\t"  // peer
            "fmul.ps   f2, f2, f4       \n\t"  // own *= alpha_own
            "fmul.ps   f3, f3, f5       \n\t"  // peer *= alpha_peer
            "fadd.ps   f2, f2, f3       \n\t"
            "fsw.ps    f2, 0(%[a])      \n\t"
            :
            : [a] "r"(acc + d), [p] "r"(peer_acc + d)
            : "f2", "f3", "memory");
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(old_ಮುಖವಾಡ));
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_flash_attn_ext_params * params, ಶೂನ್ಯ * env) {
    (ಶೂನ್ಯ) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    ಆದರೆ (shire_id >= ಅಂಕೆ_COMPUTE_SHIRES) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    ರಚನೆ ggml_tensor * q        = &params->src0;
    ರಚನೆ ggml_tensor * k        = &params->src1;
    ರಚನೆ ggml_tensor * v        = &params->src2;
    ರಚನೆ ggml_tensor * dst      = &params->dst;
    ಸ್ಥಿರಾಂಕ int32_t        has_ಮುಖವಾಡ = params->has_ಮುಖವಾಡ;
    ರಚನೆ ggml_tensor * ಮುಖವಾಡ     = has_ಮುಖವಾಡ ? &params->ಮುಖವಾಡ : (ರಚನೆ ggml_tensor *) 0;

    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * q_ದತ್ತಾಂಶ   = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) q->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * k_ದತ್ತಾಂಶ   = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) k->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * v_ದತ್ತಾಂಶ   = (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) v->ದತ್ತಾಂಶ;
    ಅಕ್ಷರ *       dst_ದತ್ತಾಂಶ = (ಅಕ್ಷರ *) dst->ದತ್ತಾಂಶ;

    // et_barrier(ET_BARRIER_GLOBAL);
    evict_region_past_l2(q->ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು_fa(q));
    evict_region_past_l2(k->ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು_fa(k));
    evict_region_past_l2(v->ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು_fa(v));
    ಆದರೆ (ಮುಖವಾಡ) {
        evict_region_past_l2(ಮುಖವಾಡ->ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು_fa(ಮುಖವಾಡ));
    }
    et_barrier(ET_BARRIER_GLOBAL);

    ಸ್ಥಿರಾಂಕ int64_t dk  = q->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t nq  = q->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t nhq = q->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t no  = q->ne[3];
    ಸ್ಥಿರಾಂಕ int64_t nk  = k->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t nhk = k->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t dv  = v->ne[0];

    ಆದರೆ (dv > FA_DV_MAX || dk > FA_DK_MAX) {
        ಹಿಂತಿರುಗಿ -1;
    }
    ಆದರೆ (k->nb[0] != 2 || v->nb[0] != 2) {
        ಹಿಂತಿರುಗಿ -1;
    }
    ಆದರೆ ((dk % 8) != 0 || (dv % 16) != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ int64_t gqa_ಅನುಪಾತ      = nhq / nhk;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು     = nq * nhq * no;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   scale          = params->scale;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     use_fast_store = (dv % 16 == 0);

    // Split-KV team layout (mirrors mul_mat_f16_matrix_engine.c)
    //
    // When total_rows is small compared to the total minion count (typical
    // for decode: nq=1, nhq small), we group k_splits minions within the
    // same shire into a team that cooperates on one row by splitting the
    // KV dimension. Each team member computes a partial (M_p, S_p, acc_p)
    // over its KV slab; the k_split==0 member merges the partials with the
    // softmax combine rule.
    //
    // k_splits is a power of two, capped at MINIONS_PER_SHIRE (so a team
    // never spans shires — L2 SCP is shire-local) and at nk_tiles (so each
    // team member gets at least one KV tile).
    ಸ್ಥಿರಾಂಕ int64_t nk_tiles      = (nk + TILE_KV - 1) / TILE_KV;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_minions = 2 * ಅಂಕೆ_COMPUTE_SHIRES * MINIONS_ಪ್ರತಿ_SHIRE;
    int64_t       k_splits      = 1;
    ಆದರೆ (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು < ಒಟ್ಟು_minions) {
        int64_t ಗುರಿ = ಒಟ್ಟು_minions / ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
        int64_t ks     = 1;
        ಆಗ (ks * 2 <= ಗುರಿ && ks * 2 <= MINIONS_ಪ್ರತಿ_SHIRE && ks * 2 <= nk_tiles) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ಸ್ಥಿರಾಂಕ int64_t tiles_ಪ್ರತಿ_shire = MINIONS_ಪ್ರತಿ_SHIRE / k_splits;
    ಸ್ಥಿರಾಂಕ int64_t k_ವಿಭಜಿಸು         = (int64_t) local_minion % k_splits;
    ಸ್ಥಿರಾಂಕ int64_t local_tile_idx  = (int64_t) local_minion / k_splits;
    ಸ್ಥಿರಾಂಕ int64_t tiles_stride    = (int64_t) ಅಂಕೆ_COMPUTE_SHIRES * tiles_ಪ್ರತಿ_shire;

    // KV slab for this k_split. With k_splits=1 this is the full range.
    ಸ್ಥಿರಾಂಕ int64_t tiles_ಪ್ರತಿ_ವಿಭಜಿಸು_rounded = (nk_tiles + k_splits - 1) / k_splits;
    ಸ್ಥಿರಾಂಕ int64_t tile_ಆರಂಭ              = k_ವಿಭಜಿಸು * tiles_ಪ್ರತಿ_ವಿಭಜಿಸು_rounded;
    int64_t       tile_ಕೊನೆ                = tile_ಆರಂಭ + tiles_ಪ್ರತಿ_ವಿಭಜಿಸು_rounded;
    ಆದರೆ (tile_ಕೊನೆ > nk_tiles) {
        tile_ಕೊನೆ = nk_tiles;
    }
    ಸ್ಥಿರಾಂಕ int64_t kv_ಆರಂಭ = tile_ಆರಂಭ * TILE_KV;
    int64_t       kv_ಕೊನೆ   = tile_ಕೊನೆ * TILE_KV;
    ಆದರೆ (kv_ಕೊನೆ > nk) {
        kv_ಕೊನೆ = nk;
    }

    // L2 SCP pointers for this minion
    uint64_t    scp_ತಳ  = local_minion * SCP_ಪ್ರತಿ_MINION;
    et_fp16_t * scp_kp[2] = {
        (et_fp16_t *) et_shire_l2scp_local(scp_ತಳ + SCP_KP0_OFF),
        (et_fp16_t *) et_shire_l2scp_local(scp_ತಳ + SCP_KP1_OFF),
    };

    // Hart 1 does K-panel packing
    //
    // When k_splits > 1, hart 1 must also participate in the two shire
    // barriers that bracket the merge phase (one before and one after, so
    // the reducer can read peer partials safely and the writers know when
    // their acc/stats slab is free to reuse). Hart 1 has no useful work
    // between those barriers.
    //
    // All teams in a shire must iterate the same number of times so the
    // per-iter shire barriers stay balanced. Teams whose assigned row is
    // past total_rows still call the barriers but skip the packing work.
    et_barrier(ET_BARRIER_SHIRE);
    // et_barrier(ET_BARRIER_GLOBAL);
    ಆದರೆ (is_hart1) {
        uint32_t      chunk_id = 0;
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ತಳ = (int64_t) shire_id + local_tile_idx * ಅಂಕೆ_COMPUTE_SHIRES;

        int64_t max_iters;
        ಆದರೆ (k_splits > 1) {
            max_iters = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು + tiles_stride - 1) / tiles_stride;
        } ಇಲ್ಲದಿದ್ದರೆ {
            max_iters = (ಅಡ್ಡಸಾಲು_ತಳ >= ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) ? 0 : ((ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು - ಅಡ್ಡಸಾಲು_ತಳ - 1) / tiles_stride + 1);
        }

        ಗಾಗಿ (int64_t iter = 0; iter < max_iters; iter++) {
            ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು      = ಅಡ್ಡಸಾಲು_ತಳ + iter * tiles_stride;
            ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     has_work = (ಅಡ್ಡಸಾಲು < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು);

            ಆದರೆ (has_work) {
                ಸ್ಥಿರಾಂಕ int64_t iq3 = ಅಡ್ಡಸಾಲು / (nhq * nq);
                ಸ್ಥಿರಾಂಕ int64_t rem = ಅಡ್ಡಸಾಲು % (nhq * nq);
                ಸ್ಥಿರಾಂಕ int64_t iq2 = rem / nq;
                ಸ್ಥಿರಾಂಕ int64_t ik2 = iq2 / gqa_ಅನುಪಾತ;

                ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * k_ತಲೆ = k_ದತ್ತಾಂಶ + ik2 * k->nb[2] + iq3 * k->nb[3];

                ಗಾಗಿ (int64_t kv_ತಳ = kv_ಆರಂಭ; kv_ತಳ < kv_ಕೊನೆ; kv_ತಳ += TILE_KV) {
                    ಸ್ಥಿರಾಂಕ int64_t kv_ಎಣಿಕೆ = (kv_ತಳ + TILE_KV <= nk) ? TILE_KV : (nk - kv_ತಳ);

                    ಗಾಗಿ (int64_t dk_chunk = 0; dk_chunk < dk; dk_chunk += TILE_K) {
                        ಪೂರ್ಣಾಂಕ buf = chunk_id & 1;

                        // Back-pressure: before overwriting buf[buf] on chunk N
                        // (which will displace chunk N-2), wait for hart 0 to
                        // post that it's done with chunk N-2. Gates both
                        // directions of double-buffering.
                        //
                        // NOTE: we use et_sem_* (FCC 0 only) rather than
                        // et_barrier(ET_BARRIER_MINION) here because the
                        // minion barrier for minion 0 shares FLB 0 with
                        // ET_BARRIER_SHIRE. Mixing them deadlocks. See
                        // feedback_flb_collision.
                        ಆದರೆ (chunk_id >= 2) {
                            et_sem_wait(ET_BARRIER_MINION);
                        }

                        // Prefetch K data for this chunk
                        prefetch_kv_ಗಮ್ಯ_l2(k_ತಲೆ, kv_ತಳ, dk_chunk, kv_ಎಣಿಕೆ, k->nb[1]);

                        pack_k_for_transpose16(scp_kp[buf], k_ತಲೆ, kv_ತಳ, dk_chunk, kv_ಎಣಿಕೆ, k->nb[1]);

                        FENCE;
                        flush_ಗಮ್ಯ_l2(scp_kp[buf], 16, 64);
                        flush_ಗಮ್ಯ_l2((et_fp16_t *) ((ಅಕ್ಷರ *) scp_kp[buf] + 1024), 16, 64);
                        WAIT_CACHEOPS;

                        // Signal: this buf is ready for hart 0 to consume.
                        et_sem_post(ET_BARRIER_MINION);

                        chunk_id++;
                    }
                }
            }

            // Shire barriers for split-KV merge (hart 1 is a passive arrival).
            ಆದರೆ (k_splits > 1) {
                et_barrier(ET_BARRIER_SHIRE);  // A: team has written its partial
                et_barrier(ET_BARRIER_SHIRE);  // B: reducer has finished merge
            }
        }

        // Self-drain phantom FCC 0 credits left by the wait-skip on the
        // first 2 chunks. Hart 1 issued chunk_id posts but only
        // (chunk_id - 2) waits (when chunk_id >= 2), so hart 1's FCC 0
        // carries +min(chunk_id,2) credits from hart 0's matching posts
        // that hart 1 never consumed.
        uint32_t drain = (chunk_id < 2) ? chunk_id : 2;
        ಗಾಗಿ (uint32_t d = 0; d < drain; d++) {
            et_sem_wait(ET_BARRIER_MINION);
        }

        // FENCE;
        // et_barrier(ET_BARRIER_GLOBAL);
        ಹಿಂತಿರುಗಿ 0;
    }

    // Hart 0: tensor engine compute
#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ UBERKERNEL_SUPPRESS_SCP_ಸಿದ್ಧತೆ
    ಸಿದ್ಧತೆ_cache_scp();
#ಆದರೆ_ಅಂತ್ಯ
    CLEAR_TENSOR_ERROR;

    // Q converted to F16 (one row at a time)
    et_fp16_t q_f16[FA_DK_MAX] __attribute__((aligned(64)));

    // Score buffer for QK^T output (16 scores per KV tile)
    ದಶಮಾಂಶ scores[TILE_KV] __attribute__((aligned(64)));

    // Small buffers for V accumulation
    et_fp16_t w_f16_buf[32] __attribute__((aligned(64)));       // 64 bytes
    et_fp16_t vpanel_buf[8 * 32] __attribute__((aligned(64)));  // 512 bytes

    ದಶಮಾಂಶ * acc = (ದಶಮಾಂಶ *) et_shire_l2scp_local(scp_ತಳ + SCP_ACC_OFF);

    uint32_t chunk_id = 0;

    // Iter-based outer loop (matches hart 1). When k_splits > 1 all teams
    // in a shire iterate the same number of times so the per-row shire
    // barriers stay balanced; iterations with row >= total_rows skip the
    // compute but still participate in the barriers.
    ಸ್ಥಿರಾಂಕ int64_t hart0_ಅಡ್ಡಸಾಲು_ತಳ = (int64_t) shire_id + local_tile_idx * ಅಂಕೆ_COMPUTE_SHIRES;
    int64_t       hart0_max_iters;
    ಆದರೆ (k_splits > 1) {
        hart0_max_iters = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು + tiles_stride - 1) / tiles_stride;
    } ಇಲ್ಲದಿದ್ದರೆ {
        hart0_max_iters = (hart0_ಅಡ್ಡಸಾಲು_ತಳ >= ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) ? 0 : ((ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು - hart0_ಅಡ್ಡಸಾಲು_ತಳ - 1) / tiles_stride + 1);
    }

    ಗಾಗಿ (int64_t iter = 0; iter < hart0_max_iters; iter++) {
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು = hart0_ಅಡ್ಡಸಾಲು_ತಳ + iter * tiles_stride;
        ಆದರೆ (ಅಡ್ಡಸಾಲು >= ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
            // No-work iteration: only participate in barriers (k_splits > 1).
            ಆದರೆ (k_splits > 1) {
                et_barrier(ET_BARRIER_SHIRE);  // A
                et_barrier(ET_BARRIER_SHIRE);  // B
            }
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ int64_t iq3 = ಅಡ್ಡಸಾಲು / (nhq * nq);
        ಸ್ಥಿರಾಂಕ int64_t rem = ಅಡ್ಡಸಾಲು % (nhq * nq);
        ಸ್ಥಿರಾಂಕ int64_t iq2 = rem / nq;
        ಸ್ಥಿರಾಂಕ int64_t iq1 = rem % nq;
        ಸ್ಥಿರಾಂಕ int64_t ik2 = iq2 / gqa_ಅನುಪಾತ;

        // Read Q row (F32) and convert to F16
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * pq = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (q_ದತ್ತಾಂಶ + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);
        convert_q_ಅಡ್ಡಸಾಲು_f32_ಗಮ್ಯ_f16(q_f16, pq, dk);

        // V base for this head + batch (K packing handled by hart 1)
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * v_ತಲೆ = v_ದತ್ತಾಂಶ + ik2 * v->nb[2] + iq3 * v->nb[3];

        // Output pointer
        ದಶಮಾಂಶ * ಹೊರಹೋಗು = (ದಶಮಾಂಶ *) (dst_ದತ್ತಾಂಶ + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        zero_acc_vec(acc, dv);
        ದಶಮಾಂಶ        M         = ET_NEG_INF_F;
        ದಶಮಾಂಶ        S         = 0.0f;
        ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ * ಮುಖವಾಡ_ತಳ = has_ಮುಖವಾಡ ? get_ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು_ತಳ(ಮುಖವಾಡ, iq1, iq2, iq3) : (ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) 0;

        // Flush Q_f16 to L2 so tensor_load can see it
        FENCE;
        flush_ಗಮ್ಯ_l2(q_f16, (dk * 2 + 63) / 64, 64);
        WAIT_CACHEOPS;

        ಗಾಗಿ (int64_t kv_ತಳ = kv_ಆರಂಭ; kv_ತಳ < kv_ಕೊನೆ; kv_ತಳ += TILE_KV) {
            ಸ್ಥಿರಾಂಕ int64_t kv_ಎಣಿಕೆ = (kv_ತಳ + TILE_KV <= nk) ? TILE_KV : (nk - kv_ತಳ);

            // Set tensor_mask for partial tiles
            ಆದರೆ (kv_ಎಣಿಕೆ < TILE_KV) {
                uint64_t tmask = (1ULL << kv_ಎಣಿಕೆ) - 1;
                __asm__ __volatile__("csrw 0x805, %0" : : "r"(tmask));
            }

            // ============================================================
            // QK^T via TensorFMA16A32
            // ============================================================

            // Pipelined QK^T:
            //   - Q for the whole row is preloaded once into A_L1[0..n-1].
            //     Each FMA picks its chunk via scp_loc_a = chunk_idx.
            //   - K is double-buffered in L1: K_BUFS[0]=lines 16..31,
            //     K_BUFS[1]=lines 32..47.
            //   - In iteration i (1..N-1), the K[i] load runs concurrently
            //     with the FMA on chunk i-1: they touch disjoint L1 regions
            //     (FMA reads K_BUFS[(i-1)&1], load writes K_BUFS[i&1]; FMA
            //     reads A_L1[i-1], load doesn't touch A_L1).
            //
            // L1 footprint: max dk=512 → Q uses 16 lines (0..15), K uses 32
            // lines (16..47). Within ET-SoC-1 L1 SCP (≥128 lines per minion).
            ಸ್ಥಿರಾಂಕ int64_t  n_dk_chunks = dk / TILE_K;
            ಸ್ಥಿರಾಂಕ uint64_t K_BUFS[2]   = {
                (uint64_t) B_L1_ಆರಂಭ,         // 16..31
                (uint64_t) (B_L1_ಆರಂಭ + 16),  // 32..47
            };

            // Preload entire Q row into A_L1[0..n_dk_chunks-1] (one tensor_load,
            // one wait, regardless of dk).
            tensor_ತುಂಬು(false, false, A_L1_ಆರಂಭ, TENSOR_ತುಂಬು_PLAIN, 0, (uint64_t) q_f16, 0,
                        (uint64_t) (n_dk_chunks - 1), 64, 0);

            // Prologue: wait hart 1's K[0], issue K[0] load, wait both loads.
            {
                ಪೂರ್ಣಾಂಕ buf = chunk_id & 1;
                et_sem_wait(ET_BARRIER_MINION);
                tensor_ತುಂಬು(false, false, K_BUFS[0], TENSOR_ತುಂಬು_TRANSPOSE16, 0, (uint64_t) scp_kp[buf], 0, 15, 64, 1);
                tensor_wait(TENSOR_ತುಂಬು_WAIT_0);  // Q row complete
                tensor_wait(TENSOR_ತುಂಬು_WAIT_1);  // K[0] complete
                et_sem_post(ET_BARRIER_MINION);
                chunk_id++;
            }

            // Main loop: in iter i, issue K[i] load and FMA chunk i-1 in
            // parallel. The matrix engine is busy on FMA[i-1] while the
            // load unit fetches K[i] from L2 SCP.
            //
            // Order of waits matters: wait K[i] load first, then sem_post
            // immediately (frees scp_kp[buf] for hart 1 to refill chunk i+2),
            // then wait FMA. Putting sem_post after FMA wait would stall
            // hart 1 by a full FMA latency — defeating the producer pipeline.
            ಗಾಗಿ (int64_t i = 1; i < n_dk_chunks; i++) {
                ಪೂರ್ಣಾಂಕ buf         = chunk_id & 1;
                ಪೂರ್ಣಾಂಕ k_ಸ್ಥಾನ_prev = (ಪೂರ್ಣಾಂಕ) ((i - 1) & 1);
                ಪೂರ್ಣಾಂಕ k_ಸ್ಥಾನ      = (ಪೂರ್ಣಾಂಕ) (i & 1);

                et_sem_wait(ET_BARRIER_MINION);
                tensor_ತುಂಬು(false, false, K_BUFS[k_ಸ್ಥಾನ], TENSOR_ತುಂಬು_TRANSPOSE16, 0, (uint64_t) scp_kp[buf], 0, 15, 64,
                            1);

                tensor_fma((kv_ಎಣಿಕೆ < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_ಸ್ಥಾನ_prev],
                           (uint64_t) (i - 1), TENSOR_FMA_OP_FP16, (i == 1));

                tensor_wait(TENSOR_ತುಂಬು_WAIT_1);  // K[i] in L1
                et_sem_post(ET_BARRIER_MINION);   // release scp_kp[buf] EARLY
                tensor_wait(TENSOR_FMA_WAIT);     // then wait FMA[i-1]
                chunk_id++;
            }

            // Epilogue: FMA on the last chunk (no overlapping load).
            {
                ಪೂರ್ಣಾಂಕ k_ಸ್ಥಾನ_ಕೊನೆಯ = (ಪೂರ್ಣಾಂಕ) ((n_dk_chunks - 1) & 1);
                tensor_fma((kv_ಎಣಿಕೆ < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_ಸ್ಥಾನ_ಕೊನೆಯ],
                           (uint64_t) (n_dk_chunks - 1), TENSOR_FMA_OP_FP16, (n_dk_chunks == 1));
                tensor_wait(TENSOR_FMA_WAIT);
            }

            // Prefetch V rows for this tile.
            // Only useful for the partial-tile path below
            ಆದರೆ (kv_ಎಣಿಕೆ < TILE_KV) {
                ಗಾಗಿ (int64_t d = 0; d < dv; d += 32) {
                    prefetch_kv_ಗಮ್ಯ_l2(v_ತಲೆ, kv_ತಳ, d, kv_ಎಣಿಕೆ, v->nb[1]);
                }
            }

            // Extract QK^T scores from vector register file
            __asm__ ಅಸ್ಥಿರ("" ::: "f0", "f1");
            {
                ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ _ms;
                __asm__ ಅಸ್ಥಿರ(
                    "mova.x.m  %[ms]                \n\t"
                    "mov.m.x   m0, x0, 0xFF         \n\t"
                    "fbc.ps    f2, 0(%[p_scale])    \n\t"
                    "fmul.ps   f0, f0, f2           \n\t"
                    "fmul.ps   f1, f1, f2           \n\t"
                    "fsw.ps    f0, 0(%[dst])        \n\t"
                    "fsw.ps    f1, 32(%[dst])       \n\t"
                    "mova.m.x  %[ms]                \n\t"
                    : [ms] "=&r"(_ms)
                    : [dst] "r"(scores), [p_scale] "r"(&scale)
                    : "f0", "f1", "f2", "memory");
            }

            // ============================================================
            // Two-phase softmax + V accumulation
            // ============================================================

            ದಶಮಾಂಶ weights[TILE_KV] __attribute__((aligned(64)));
            {
                // A1: apply mask to scores, pad unused slots
                ಗಾಗಿ (int64_t j = 0; j < kv_ಎಣಿಕೆ; ++j) {
                    ಆದರೆ (has_ಮುಖವಾಡ) {
                        ದಶಮಾಂಶ mv = get_ಮುಖವಾಡ_val_from_ತಳ(ಮುಖವಾಡ, ಮುಖವಾಡ_ತಳ, kv_ತಳ + j);
                        ಆದರೆ (mv == ET_NEG_INF_F || mv != mv) {
                            scores[j] = ET_NEG_INF_F;
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            scores[j] += mv;
                        }
                    }
                }
                ಗಾಗಿ (int64_t j = kv_ಎಣಿಕೆ; j < TILE_KV; ++j) {
                    scores[j] = ET_NEG_INF_F;
                }

                // A1b: SIMD horizontal max across all 16 scores
                ದಶಮಾಂಶ tile_max;
                {
                    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ _ms;
                    __asm__ ಅಸ್ಥಿರ(
                        "mova.x.m  %[ms]              \n\t"
                        "mov.m.x   m0, x0, 0xFF       \n\t"
                        "flw.ps    f2, 0(%[sc])       \n\t"
                        "flw.ps    f3, 32(%[sc])      \n\t"
                        "fmax.ps   f2, f2, f3         \n\t"
                        "fswizz.ps f3, f2, 0xB1       \n\t"
                        "fmax.ps   f2, f2, f3         \n\t"
                        "fswizz.ps f3, f2, 0x4E       \n\t"
                        "fmax.ps   f2, f2, f3         \n\t"
                        "fmvz.x.ps t0, f2, 4          \n\t"
                        "fbcx.ps   f3, t0             \n\t"
                        "fmax.ps   %[tm], f2, f3      \n\t"
                        "mova.m.x  %[ms]              \n\t"
                        : [ms] "=&r"(_ms), [tm] "=f"(tile_max)
                        : [sc] "r"(scores)
                        : "f2", "f3", "t0", "memory");
                }

                ಆದರೆ (tile_max > ET_NEG_INF_F) {
                    // A2: rescale accumulator if this tile has a new global max
                    ಆದರೆ (tile_max > M) {
                        ದಶಮಾಂಶ rescale = et_exp2f((M - tile_max) * 1.4426950408889634f);
                        scale_acc_vec(acc, dv, rescale);
                        S *= rescale;
                        M = tile_max;
                    }

                    // A3: SIMD exp2 + horizontal sum
                    // Interleaved: f2/f3 chains alternate to hide ALU latency.
                    // fexp.ps has multi-cycle latency — the two independent
                    // exp2 calls naturally pipeline.
                    {
                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   log2e = 1.4426950408889634f;
                        ದಶಮಾಂಶ         S_tile;
                        ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ _ms;
                        __asm__ ಅಸ್ಥಿರ(
                            "mova.x.m  %[ms]              \n\t"
                            "mov.m.x   m0, x0, 0xFF       \n\t"
                            "flw.ps    f2, 0(%[sc])       \n\t"
                            "fbc.ps    f4, 0(%[pM])       \n\t"
                            "flw.ps    f3, 32(%[sc])      \n\t"
                            "fbc.ps    f5, 0(%[pL])       \n\t"
                            "fsub.ps   f2, f2, f4         \n\t"
                            "fsub.ps   f3, f3, f4         \n\t"
                            "fmul.ps   f2, f2, f5         \n\t"
                            "fmul.ps   f3, f3, f5         \n\t"
                            "fexp.ps   f2, f2             \n\t"
                            "fexp.ps   f3, f3             \n\t"
                            "fsw.ps    f2, 0(%[wt])       \n\t"
                            "fsw.ps    f3, 32(%[wt])      \n\t"
                            "fadd.ps   f2, f2, f3, rne    \n\t"
                            "fswizz.ps f3, f2, 0xB1       \n\t"
                            "fadd.ps   f2, f2, f3, rne    \n\t"
                            "fswizz.ps f3, f2, 0x4E       \n\t"
                            "fadd.ps   f2, f2, f3, rne    \n\t"
                            "fmvz.x.ps t0, f2, 4          \n\t"
                            "fbcx.ps   f3, t0             \n\t"
                            "fadd.ps   %[st], f2, f3, rne \n\t"
                            "mova.m.x  %[ms]              \n\t"
                            : [ms] "=&r"(_ms), [st] "=f"(S_tile)
                            : [pM] "r"(&M), [pL] "r"(&log2e), [sc] "r"(scores), [wt] "r"(weights)
                            : "f2", "f3", "f4", "f5", "t0", "memory");
                        S += S_tile;
                    }

                    // Phase B: weights @ V via TensorFMA16A32
                    {
                        // B1: convert weights F32 → F16
                        convert_q_ಅಡ್ಡಸಾಲು_f32_ಗಮ್ಯ_f16(w_f16_buf, weights, TILE_KV);

                        FENCE;
                        flush_ಗಮ್ಯ_l2(w_f16_buf, 1, 64);
                        WAIT_CACHEOPS;

                        // Issue weights load (wait_id=0) and the first V chunk
                        // load (wait_id=1) concurrently. Weights comes from
                        // L2 SCP (just flushed); V[0] comes from DRAM via
                        // INTERLEAVE16 — running them in parallel hides the
                        // shorter load behind the longer one. For partial
                        // tiles, V is software-packed below — we only kick
                        // off the early V load on the full-tile fast path.
                        tensor_ತುಂಬು(false, false, A_L1_ಆರಂಭ, TENSOR_ತುಂಬು_PLAIN, 0, (uint64_t) w_f16_buf, 0, 0, 64, 0);

                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ       v_ತುಂಬಿದ_tile = (kv_ಎಣಿಕೆ == TILE_KV);
                        ಸ್ಥಿರಾಂಕ uintptr_t v_ತಳ      = (uintptr_t) v_ತಲೆ + kv_ತಳ * v->nb[1];
                        ಸ್ಥಿರಾಂಕ uint64_t  nb1_v       = (uint64_t) v->nb[1];
                        uint64_t        b_cur       = 8;

                        ಆದರೆ (v_ತುಂಬಿದ_tile) {
                            tensor_ತುಂಬು(false, false, b_cur, TENSOR_ತುಂಬು_INTERLEAVE16, 0, (uint64_t) v_ತಳ, 0, 7,
                                        nb1_v, 1);
                        }

                        tensor_wait(TENSOR_ತುಂಬು_WAIT_0);      // weights in A_L1
                        ಆದರೆ (v_ತುಂಬಿದ_tile) {
                            tensor_wait(TENSOR_ತುಂಬು_WAIT_1);  // V[0] in b_cur
                        }

                        // B2: process dv in chunks of 16
                        ಆದರೆ (v_ತುಂಬಿದ_tile) {
                            ಗಾಗಿ (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                ಸ್ಥಿರಾಂಕ uint64_t b_nxt = b_cur ^ 24;

                                ಆದರೆ (dv_off + 16 < dv) {
                                    tensor_ತುಂಬು(false, false, b_nxt, TENSOR_ತುಂಬು_INTERLEAVE16, 0,
                                                (uint64_t) (v_ತಳ + (dv_off + 16) * 2), 0, 7, nb1_v, 1);
                                }

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, b_cur, A_L1_ಆರಂಭ,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ ಅಸ್ಥಿರ("" ::: "f0", "f1");
                                {
                                    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ _ms;
                                    __asm__ ಅಸ್ಥಿರ(
                                        "mova.x.m  %[ms]            \n\t"
                                        "mov.m.x   m0, x0, 0xFF     \n\t"
                                        "flw.ps    f2, 0(%[pa])     \n\t"
                                        "flw.ps    f3, 32(%[pa])    \n\t"
                                        "fadd.ps   f0, f0, f2       \n\t"
                                        "fadd.ps   f1, f1, f3       \n\t"
                                        "fsw.ps    f0, 0(%[pa])     \n\t"
                                        "fsw.ps    f1, 32(%[pa])    \n\t"
                                        "mova.m.x  %[ms]            \n\t"
                                        : [ms] "=&r"(_ms)
                                        : [pa] "r"(acc + dv_off)
                                        : "f0", "f1", "f2", "f3", "memory");
                                }

                                ಆದರೆ (dv_off + 16 < dv) {
                                    tensor_wait(TENSOR_ತುಂಬು_WAIT_1);
                                    b_cur = b_nxt;
                                }
                            }
                        } ಇಲ್ಲದಿದ್ದರೆ {
                            // Partial tile: software pack, no pipeline
                            ಗಾಗಿ (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                pack_v_interleaved(vpanel_buf, v_ತಲೆ, kv_ತಳ, dv_off, kv_ಎಣಿಕೆ, v->nb[1]);
                                FENCE;
                                flush_ಗಮ್ಯ_l2(vpanel_buf, 8, 64);
                                WAIT_CACHEOPS;
                                tensor_ತುಂಬು(false, false, B_L1_ಆರಂಭ, TENSOR_ತುಂಬು_PLAIN, 0, (uint64_t) vpanel_buf, 0, 7,
                                            64, 0);
                                tensor_wait(TENSOR_ತುಂಬು_WAIT_0);

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, B_L1_ಆರಂಭ, A_L1_ಆರಂಭ,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ ಅಸ್ಥಿರ("" ::: "f0", "f1");
                                {
                                    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ _ms;
                                    __asm__ ಅಸ್ಥಿರ(
                                        "mova.x.m  %[ms]            \n\t"
                                        "mov.m.x   m0, x0, 0xFF     \n\t"
                                        "flw.ps    f2, 0(%[pa])     \n\t"
                                        "flw.ps    f3, 32(%[pa])    \n\t"
                                        "fadd.ps   f0, f0, f2       \n\t"
                                        "fadd.ps   f1, f1, f3       \n\t"
                                        "fsw.ps    f0, 0(%[pa])     \n\t"
                                        "fsw.ps    f1, 32(%[pa])    \n\t"
                                        "mova.m.x  %[ms]            \n\t"
                                        : [ms] "=&r"(_ms)
                                        : [pa] "r"(acc + dv_off)
                                        : "f0", "f1", "f2", "f3", "memory");
                                }
                            }
                        }
                    }
                }
            }
        }

        // Finalize row
        //
        // k_splits == 1: this minion computed the full row. Normalize in
        //                place and store to DRAM.
        //
        // k_splits  > 1: this minion computed a KV slab. Publish the
        //                partial (M, S, acc) to L2 SCP, sync with the
        //                team, and let the k_split==0 member do the
        //                softmax combine and the final store. All tensor
        //                engine ops are complete before this block, so
        //                f0..f31 are free to use.
        ಆದರೆ (k_splits > 1) {
            // Publish our partial.
            ಅಸ್ಥಿರ ದಶಮಾಂಶ * my_stats = (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) et_shire_l2scp_local(scp_ತಳ + SCP_STATS_OFF);
            my_stats[0]               = M;
            my_stats[1]               = S;
            FENCE;
            evict_range_ಗಮ್ಯ_l2(acc, (int64_t) dv * (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ));
            evict_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) my_stats, 1, 64);
            WAIT_CACHEOPS;

            // A: team members have all written their partials.
            et_barrier(ET_BARRIER_SHIRE);

            ಆದರೆ (k_ವಿಭಜಿಸು == 0) {
                // Online softmax merge: fold peers 1..k_splits-1 into our
                // own (M_running, S_running, acc). For each peer p:
                //   M_new    = max(M_running, M_p)
                //   α_own    = exp2((M_running - M_new) * log2e)
                //   α_p      = exp2((M_p     - M_new) * log2e)
                //   acc[d]   = α_own * acc[d] + α_p * peer_acc[d]
                //   S_running = α_own * S_running + α_p * S_p
                ದಶಮಾಂಶ       M_running = M;
                ದಶಮಾಂಶ       S_running = S;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ log2e     = 1.4426950408889634f;

                ಗಾಗಿ (int64_t p = 1; p < k_splits; p++) {
                    uint64_t         peer_scp   = (local_tile_idx * k_splits + p) * SCP_ಪ್ರತಿ_MINION;
                    ಅಸ್ಥಿರ ದಶಮಾಂಶ * peer_stats = (ಅಸ್ಥಿರ ದಶಮಾಂಶ *) et_shire_l2scp_local(peer_scp + SCP_STATS_OFF);
                    ದಶಮಾಂಶ *          peer_acc   = (ದಶಮಾಂಶ *) et_shire_l2scp_local(peer_scp + SCP_ACC_OFF);

                    // Drop stale L1D copies before reading peer's data.
                    evict_ಗಮ್ಯ_l2((ಸ್ಥಿರಾಂಕ ಶೂನ್ಯ *) peer_stats, 1, 64);
                    evict_range_ಗಮ್ಯ_l2(peer_acc, (int64_t) dv * (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ));
                    WAIT_CACHEOPS;

                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ M_p = peer_stats[0];
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ S_p = peer_stats[1];

                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ M_new     = (M_p > M_running) ? M_p : M_running;
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha_own = (M_running == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_running - M_new) * log2e);
                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ alpha_p   = (M_p == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_p - M_new) * log2e);

                    ವಿಲೀನ_rescale_ಕೂಡಿಸು_asm(acc, peer_acc, dv, alpha_own, alpha_p);

                    S_running = alpha_own * S_running + alpha_p * S_p;
                    M_running = M_new;
                }

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ S_inv = (S_running == 0.0f) ? 0.0f : et_fdiv(1.0f, S_running);
                normalize_store_vec(ಹೊರಹೋಗು, acc, dv, S_inv, use_fast_store);
            }

            // B: reducer is done, team may reuse its acc/stats slabs.
            et_barrier(ET_BARRIER_SHIRE);
        } ಇಲ್ಲದಿದ್ದರೆ {
            // k_splits == 1 fast path — this minion owns the full row.
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
            normalize_store_vec(ಹೊರಹೋಗು, acc, dv, S_inv, use_fast_store);
        }
    }

    FENCE;
    ಹಿಂತಿರುಗಿ 0;
}
