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

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "tensor.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <etsoc/common/utils.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅੰਕ_COMPUTE_SHIRES 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MINIONS_ਪ੍ਰਤੀ_SHIRE  32

// QK^T tiles: 16 KV positions at a time, K in chunks of 32 F16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE_KV 16
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ TILE_K  32

// L1 scratchpad layout: A (Q) in lines 0-15, B (K interleaved) in lines 16-31
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ A_L1_ਸ਼ੁਰੂ 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ B_L1_ਸ਼ੁਰੂ 16

// Max head dimensions
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FA_DV_MAX 512  // max value head dim (dv)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FA_DK_MAX 512  // max key head dim (dk) - some models use hsk > hsv

ਕਿਸਮਪਰਿਭਾਸ਼ਾ uint16_t et_fp16_t;

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ET_NEG_INF_F (-3.402823466e+38f)

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
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_ACC_OFF     0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_ACC_STRIDE  (FA_DV_MAX * ਆਕਾਰ(ਦਸ਼ਮਲਵ))       // 2048
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_KPANEL_ਮਾਪ (32 * 32 * ਆਕਾਰ(et_fp16_t))     // 2048
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_KP0_OFF     SCP_ACC_STRIDE                    // 2048
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_KP1_OFF     (SCP_KP0_OFF + SCP_KPANEL_ਮਾਪ)   // 4096
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_STATS_OFF   (SCP_KP1_OFF + SCP_KPANEL_ਮਾਪ)   // 6144
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_STATS_ਮਾਪ  64                                // own cache line
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ SCP_ਪ੍ਰਤੀ_MINION  (SCP_STATS_OFF + SCP_STATS_ਮਾਪ)  // 6208

ਬਣਤਰ ggml_et_flash_attn_ext_params {
    ਬਣਤਰ ggml_tensor src0;  // Q (F32)
    ਬਣਤਰ ggml_tensor src1;  // K (F16)
    ਬਣਤਰ ggml_tensor src2;  // V (F16)
    ਬਣਤਰ ggml_tensor ਨਕਾਬ;  // mask (F16 or F32), zeroed when absent
    ਬਣਤਰ ggml_tensor dst;   // Output (F32)
    ਦਸ਼ਮਲਵ              scale;
    int32_t            has_ਨਕਾਬ;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ get_ਨਕਾਬ_val(ਅਚਲ ਬਣਤਰ ggml_tensor * ਨਕਾਬ, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    ਅਚਲ ਅੱਖਰ * ਅਧਾਰ = (ਅਚਲ ਅੱਖਰ *) ਨਕਾਬ->ਅੰਕੜੇ + iq1 * ਨਕਾਬ->nb[1] + (iq2 % ਨਕਾਬ->ne[2]) * ਨਕਾਬ->nb[2] +
                        (iq3 % ਨਕਾਬ->ne[3]) * ਨਕਾਬ->nb[3];

    ਜੇ (ਨਕਾਬ->type == GGML_TYPE_F32) {
        ਵਾਪਸ *(ਅਚਲ ਦਸ਼ਮਲਵ *) (ਅਧਾਰ + ik1 * ਨਕਾਬ->nb[0]);
    }
    ਵਾਪਸ fp16_ਮੰਜ਼ਿਲ_fp32(*(ਅਚਲ uint16_t *) (ਅਧਾਰ + ik1 * ਨਕਾਬ->nb[0]));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਅਚਲ ਅੱਖਰ * get_ਨਕਾਬ_ਪੰਕਤੀ_ਅਧਾਰ(ਅਚਲ ਬਣਤਰ ggml_tensor * ਨਕਾਬ, int64_t iq1, int64_t iq2, int64_t iq3) {
    ਵਾਪਸ (ਅਚਲ ਅੱਖਰ *) ਨਕਾਬ->ਅੰਕੜੇ + iq1 * ਨਕਾਬ->nb[1] + (iq2 % ਨਕਾਬ->ne[2]) * ਨਕਾਬ->nb[2] +
           (iq3 % ਨਕਾਬ->ne[3]) * ਨਕਾਬ->nb[3];
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ get_ਨਕਾਬ_val_from_ਅਧਾਰ(ਅਚਲ ਬਣਤਰ ggml_tensor * ਨਕਾਬ, ਅਚਲ ਅੱਖਰ * ਅਧਾਰ, int64_t ik1) {
    ਜੇ (ਨਕਾਬ->type == GGML_TYPE_F32) {
        ਵਾਪਸ *(ਅਚਲ ਦਸ਼ਮਲਵ *) (ਅਧਾਰ + ik1 * ਨਕਾਬ->nb[0]);
    }
    ਵਾਪਸ fp16_ਮੰਜ਼ਿਲ_fp32(*(ਅਚਲ uint16_t *) (ਅਧਾਰ + ik1 * ਨਕਾਬ->nb[0]));
}

// Pack K rows for TensorLoadTranspose16 (even/odd deinterleave)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) pack_k_for_transpose16(et_fp16_t *  ਬਾਹਰਲਾ,
                                                                         ਅਚਲ ਅੱਖਰ * k_ਅਧਾਰ,
                                                                         int64_t      kv_ਸ਼ੁਰੂ,
                                                                         int64_t      dk_ਸ਼ੁਰੂ,
                                                                         int64_t      kv_ਗਿਣਤੀ,
                                                                         int64_t      nb1_k) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ old_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        : [ms] "=&r"(old_ਨਕਾਬ)
        :
        :);

    ਲਈ (ਪੂਰਨ j = 0; j < (ਪੂਰਨ) kv_ਗਿਣਤੀ; ++j) {
        ਅਚਲ et_fp16_t * k_ਪੰਕਤੀ    = (ਅਚਲ et_fp16_t *) (k_ਅਧਾਰ + (kv_ਸ਼ੁਰੂ + j) * nb1_k) + dk_ਸ਼ੁਰੂ;
        et_fp16_t *       even_ਪੰਕਤੀ = ਬਾਹਰਲਾ + (j * 2) * 32;
        et_fp16_t *       odd_ਪੰਕਤੀ  = ਬਾਹਰਲਾ + (j * 2 + 1) * 32;
        __asm__ ਅਸਥਿਰ(
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
            : [src0] "r"(k_ਪੰਕਤੀ), [src1] "r"(k_ਪੰਕਤੀ + 16), [even] "r"(even_ਪੰਕਤੀ), [odd] "r"(odd_ਪੰਕਤੀ)
            : "f2", "f3", "f4", "f5", "f6", "f7", "memory");
    }

    __asm__ ਅਸਥਿਰ("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_ਨਕਾਬ));

    ਲਈ (ਪੂਰਨ j = (ਪੂਰਨ) kv_ਗਿਣਤੀ; j < TILE_KV; ++j) {
        et_fp16_t * even_ਪੰਕਤੀ = ਬਾਹਰਲਾ + (j * 2) * 32;
        et_fp16_t * odd_ਪੰਕਤੀ  = ਬਾਹਰਲਾ + (j * 2 + 1) * 32;
        ਲਈ (ਪੂਰਨ l = 0; l < TILE_K / 2; ++l) {
            even_ਪੰਕਤੀ[l] = 0;
            odd_ਪੰਕਤੀ[l]  = 0;
        }
    }
}

// Build interleaved B panel for TensorFMA16A32 (weights @ V).
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) pack_v_interleaved(et_fp16_t *  ਬਾਹਰਲਾ,
                                                                     ਅਚਲ ਅੱਖਰ * v_ਸਿਰ,
                                                                     int64_t      kv_ਅਧਾਰ,
                                                                     int64_t      dv_ਸ਼ੁਰੂ,
                                                                     int64_t      kv_ਗਿਣਤੀ,
                                                                     int64_t      nb1_v) {
    ਲਈ (ਪੂਰਨ k = 0; k < TILE_KV; ++k) {
        ਅਚਲ ਪੂਰਨ         l   = k >> 1;
        ਅਚਲ ਪੂਰਨ         r   = k & 1;
        et_fp16_t * ਅਚਲ dst = ਬਾਹਰਲਾ + l * 32 + r;
        ਜੇ (k < (ਪੂਰਨ) kv_ਗਿਣਤੀ) {
            ਅਚਲ et_fp16_t * v_ਪੰਕਤੀ = (ਅਚਲ et_fp16_t *) (v_ਸਿਰ + (kv_ਅਧਾਰ + k) * nb1_v) + dv_ਸ਼ੁਰੂ;
            ਲਈ (ਪੂਰਨ n = 0; n < 16; ++n) {
                dst[n * 2] = v_ਪੰਕਤੀ[n];
            }
        } ਨਹੀਂਤਾਂ {
            ਲਈ (ਪੂਰਨ n = 0; n < 16; ++n) {
                dst[n * 2] = 0;
            }
        }
    }
}

// Prefetch KV rows for one chunk into L2.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) prefetch_kv_ਮੰਜ਼ਿਲ_l2(ਅਚਲ ਅੱਖਰ * ਸਿਰ,
                                                                    int64_t      kv_ਸ਼ੁਰੂ,
                                                                    int64_t      d_ਸ਼ੁਰੂ,
                                                                    int64_t      kv_ਗਿਣਤੀ,
                                                                    int64_t      nb1) {
    ਅਚਲ ਖਾਲੀ * ਅਧਾਰ = (ਅਚਲ ਖਾਲੀ *) (ਸਿਰ + kv_ਸ਼ੁਰੂ * nb1 + d_ਸ਼ੁਰੂ * 2);
    l2_prefetch(ਅਧਾਰ, (uint64_t) kv_ਗਿਣਤੀ, (uint64_t) nb1);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) convert_q_ਪੰਕਤੀ_f32_ਮੰਜ਼ਿਲ_f16(et_fp16_t *   dst,
                                                                           ਅਚਲ ਦਸ਼ਮਲਵ * src,
                                                                           int64_t       n) {
    ਸਥਿਰ ਅਚਲ int32_t __attribute__((aligned(32))) offsets[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ old_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 0xFF      \n\t"
        "flw.ps    f1, 0(%[offs])    \n\t"
        : [ms] "=&r"(old_ਨਕਾਬ)
        : [offs] "r"(offsets)
        : "f1");

    ਲਈ (int64_t d = 0; d < n; d += 8) {
        __asm__ ਅਸਥਿਰ(
            "flw.ps      f2, 0(%[src])    \n\t"
            "fcvt.f16.ps f3, f2           \n\t"
            "fsch.ps     f3, f1(%[dst])   \n\t"
            :
            : [src] "r"(src + d), [dst] "r"(dst + d)
            : "f2", "f3", "memory");
    }

    __asm__ ਅਸਥਿਰ("mova.m.x  %[ms]             \n\t" : : [ms] "r"(old_ਨਕਾਬ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) zero_acc_vec(ਦਸ਼ਮਲਵ * acc, int64_t dv) {
    ਅਚਲ ਦਸ਼ਮਲਵ   zero = 0.0f;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ old_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(old_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");
    __asm__ ਅਸਥਿਰ("fbc.ps  f2, 0(%[z])" ::[z] "r"(&zero) : "f2");

    ਲਈ (int64_t d = 0; d < dv; d += 8) {
        __asm__ ਅਸਥਿਰ("fsw.ps  f2, 0(%[a])     \n\t" ::[a] "r"(acc + d) : "f2", "memory");
    }

    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(old_ਨਕਾਬ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) scale_acc_vec(ਦਸ਼ਮਲਵ * acc, int64_t dv, ਦਸ਼ਮਲਵ scale) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ old_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(old_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

    ਲਈ (int64_t d = 0; d < dv; d += 8) {
        __asm__ ਅਸਥਿਰ(
            "fbc.ps    f2, 0(%[s])    \n\t"
            "flw.ps    f3, 0(%[a])    \n\t"
            "fmul.ps   f3, f3, f2     \n\t"
            "fsw.ps    f3, 0(%[a])    \n\t"
            :
            : [s] "r"(&scale), [a] "r"(acc + d)
            : "f2", "f3", "memory");
    }

    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(old_ਨਕਾਬ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) normalize_store_vec(ਦਸ਼ਮਲਵ * ਬਾਹਰਲਾ,
                                                                      ਦਸ਼ਮਲਵ * acc,
                                                                      int64_t dv,
                                                                      ਦਸ਼ਮਲਵ   inv,
                                                                      ਪੂਰਨ     use_fast_store) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ old_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(old_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

    ਲਈ (int64_t d = 0; d < dv; d += 8) {
        __asm__ ਅਸਥਿਰ(
            "fbc.ps    f2, 0(%[inv])   \n\t"
            "flw.ps    f3, 0(%[a])     \n\t"
            "fmul.ps   f3, f3, f2      \n\t"
            "fsw.ps    f3, 0(%[a])     \n\t"
            :
            : [inv] "r"(&inv), [a] "r"(acc + d)
            : "f2", "f3", "memory");
        ਜੇ (use_fast_store) {
            __asm__ ਅਸਥਿਰ(
                "flw.ps  f4, 0(%[a])     \n\t"
                "fsw.ps  f4, 0(%[o])     \n\t"
                :
                : [a] "r"(acc + d), [o] "r"(ਬਾਹਰਲਾ + d)
                : "f4", "memory");
        } ਨਹੀਂਤਾਂ {
            atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) &ਬਾਹਰਲਾ[d + 0], acc[d + 0]);
            atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) &ਬਾਹਰਲਾ[d + 1], acc[d + 1]);
            atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) &ਬਾਹਰਲਾ[d + 2], acc[d + 2]);
            atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) &ਬਾਹਰਲਾ[d + 3], acc[d + 3]);
            atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) &ਬਾਹਰਲਾ[d + 4], acc[d + 4]);
            atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) &ਬਾਹਰਲਾ[d + 5], acc[d + 5]);
            atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) &ਬਾਹਰਲਾ[d + 6], acc[d + 6]);
            atomic_store_f32((ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) &ਬਾਹਰਲਾ[d + 7], acc[d + 7]);
        }
    }

    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(old_ਨਕਾਬ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t tensor_ਬਾਈਟਾਂ_fa(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਵਾਪਸ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

// Evict a byte range from L1D to L2 SCP, splitting into batches of ≤16
// cache lines (the hw limit for evict_to_l2). Use before a barrier when
// another minion in the shire needs to read the region, or after a barrier
// on the reader side to drop stale L1D copies before reading peer data.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) evict_range_ਮੰਜ਼ਿਲ_l2(ਅਚਲ ਖਾਲੀ * addr, int64_t ਬਾਈਟਾਂ) {
    ਜੇ (ਬਾਈਟਾਂ <= 0) {
        ਵਾਪਸ;
    }
    int64_t      ਸਤਰਾਂ = (ਬਾਈਟਾਂ + 63) / 64;
    ਅਚਲ ਅੱਖਰ * p     = (ਅਚਲ ਅੱਖਰ *) addr;
    ਜਦੋਂ (ਸਤਰਾਂ > 0) {
        int64_t batch = ਸਤਰਾਂ > 16 ? 16 : ਸਤਰਾਂ;
        evict_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) p, (uint64_t) batch, 64);
        p += batch * 64;
        ਸਤਰਾਂ -= batch;
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
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) ਮਿਲਾਓ_rescale_ਜੋੜੋ_asm(ਦਸ਼ਮਲਵ *       acc,
                                                                        ਅਚਲ ਦਸ਼ਮਲਵ * peer_acc,
                                                                        int64_t       dv,
                                                                        ਦਸ਼ਮਲਵ         alpha_own,
                                                                        ਦਸ਼ਮਲਵ         alpha_peer) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ old_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[ms]              \n\t"
        "mov.m.x   m0, x0, 0xFF       \n\t"
        "fbc.ps    f4, 0(%[ao])       \n\t"  // broadcast alpha_own
        "fbc.ps    f5, 0(%[ap])       \n\t"  // broadcast alpha_peer
        : [ms] "=&r"(old_ਨਕਾਬ)
        : [ao] "r"(&alpha_own), [ap] "r"(&alpha_peer)
        : "f4", "f5");

    ਲਈ (int64_t d = 0; d < dv; d += 8) {
        __asm__ ਅਸਥਿਰ(
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

    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(old_ਨਕਾਬ));
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_flash_attn_ext_params * params, ਖਾਲੀ * env) {
    (ਖਾਲੀ) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    ਜੇ (shire_id >= ਅੰਕ_COMPUTE_SHIRES) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ ਪੂਰਨ is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    ਬਣਤਰ ggml_tensor * q        = &params->src0;
    ਬਣਤਰ ggml_tensor * k        = &params->src1;
    ਬਣਤਰ ggml_tensor * v        = &params->src2;
    ਬਣਤਰ ggml_tensor * dst      = &params->dst;
    ਅਚਲ int32_t        has_ਨਕਾਬ = params->has_ਨਕਾਬ;
    ਬਣਤਰ ggml_tensor * ਨਕਾਬ     = has_ਨਕਾਬ ? &params->ਨਕਾਬ : (ਬਣਤਰ ggml_tensor *) 0;

    ਅਚਲ ਅੱਖਰ * q_ਅੰਕੜੇ   = (ਅਚਲ ਅੱਖਰ *) q->ਅੰਕੜੇ;
    ਅਚਲ ਅੱਖਰ * k_ਅੰਕੜੇ   = (ਅਚਲ ਅੱਖਰ *) k->ਅੰਕੜੇ;
    ਅਚਲ ਅੱਖਰ * v_ਅੰਕੜੇ   = (ਅਚਲ ਅੱਖਰ *) v->ਅੰਕੜੇ;
    ਅੱਖਰ *       dst_ਅੰਕੜੇ = (ਅੱਖਰ *) dst->ਅੰਕੜੇ;

    // et_barrier(ET_BARRIER_GLOBAL);
    evict_region_past_l2(q->ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ_fa(q));
    evict_region_past_l2(k->ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ_fa(k));
    evict_region_past_l2(v->ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ_fa(v));
    ਜੇ (ਨਕਾਬ) {
        evict_region_past_l2(ਨਕਾਬ->ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ_fa(ਨਕਾਬ));
    }
    et_barrier(ET_BARRIER_GLOBAL);

    ਅਚਲ int64_t dk  = q->ne[0];
    ਅਚਲ int64_t nq  = q->ne[1];
    ਅਚਲ int64_t nhq = q->ne[2];
    ਅਚਲ int64_t no  = q->ne[3];
    ਅਚਲ int64_t nk  = k->ne[1];
    ਅਚਲ int64_t nhk = k->ne[2];
    ਅਚਲ int64_t dv  = v->ne[0];

    ਜੇ (dv > FA_DV_MAX || dk > FA_DK_MAX) {
        ਵਾਪਸ -1;
    }
    ਜੇ (k->nb[0] != 2 || v->nb[0] != 2) {
        ਵਾਪਸ -1;
    }
    ਜੇ ((dk % 8) != 0 || (dv % 16) != 0) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ int64_t gqa_ਅਨੁਪਾਤ      = nhq / nhk;
    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ     = nq * nhq * no;
    ਅਚਲ ਦਸ਼ਮਲਵ   scale          = params->scale;
    ਅਚਲ ਪੂਰਨ     use_fast_store = (dv % 16 == 0);

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
    ਅਚਲ int64_t nk_tiles      = (nk + TILE_KV - 1) / TILE_KV;
    ਅਚਲ int64_t ਕੁੱਲ_minions = 2 * ਅੰਕ_COMPUTE_SHIRES * MINIONS_ਪ੍ਰਤੀ_SHIRE;
    int64_t       k_splits      = 1;
    ਜੇ (ਕੁੱਲ_ਪੰਕਤੀਆਂ < ਕੁੱਲ_minions) {
        int64_t ਨਿਸ਼ਾਨਾ = ਕੁੱਲ_minions / ਕੁੱਲ_ਪੰਕਤੀਆਂ;
        int64_t ks     = 1;
        ਜਦੋਂ (ks * 2 <= ਨਿਸ਼ਾਨਾ && ks * 2 <= MINIONS_ਪ੍ਰਤੀ_SHIRE && ks * 2 <= nk_tiles) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ਅਚਲ int64_t tiles_ਪ੍ਰਤੀ_shire = MINIONS_ਪ੍ਰਤੀ_SHIRE / k_splits;
    ਅਚਲ int64_t k_ਵੰਡੋ         = (int64_t) local_minion % k_splits;
    ਅਚਲ int64_t local_tile_idx  = (int64_t) local_minion / k_splits;
    ਅਚਲ int64_t tiles_stride    = (int64_t) ਅੰਕ_COMPUTE_SHIRES * tiles_ਪ੍ਰਤੀ_shire;

    // KV slab for this k_split. With k_splits=1 this is the full range.
    ਅਚਲ int64_t tiles_ਪ੍ਰਤੀ_ਵੰਡੋ_rounded = (nk_tiles + k_splits - 1) / k_splits;
    ਅਚਲ int64_t tile_ਸ਼ੁਰੂ              = k_ਵੰਡੋ * tiles_ਪ੍ਰਤੀ_ਵੰਡੋ_rounded;
    int64_t       tile_ਸਿਰਾ                = tile_ਸ਼ੁਰੂ + tiles_ਪ੍ਰਤੀ_ਵੰਡੋ_rounded;
    ਜੇ (tile_ਸਿਰਾ > nk_tiles) {
        tile_ਸਿਰਾ = nk_tiles;
    }
    ਅਚਲ int64_t kv_ਸ਼ੁਰੂ = tile_ਸ਼ੁਰੂ * TILE_KV;
    int64_t       kv_ਸਿਰਾ   = tile_ਸਿਰਾ * TILE_KV;
    ਜੇ (kv_ਸਿਰਾ > nk) {
        kv_ਸਿਰਾ = nk;
    }

    // L2 SCP pointers for this minion
    uint64_t    scp_ਅਧਾਰ  = local_minion * SCP_ਪ੍ਰਤੀ_MINION;
    et_fp16_t * scp_kp[2] = {
        (et_fp16_t *) et_shire_l2scp_local(scp_ਅਧਾਰ + SCP_KP0_OFF),
        (et_fp16_t *) et_shire_l2scp_local(scp_ਅਧਾਰ + SCP_KP1_OFF),
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
    ਜੇ (is_hart1) {
        uint32_t      chunk_id = 0;
        ਅਚਲ int64_t ਪੰਕਤੀ_ਅਧਾਰ = (int64_t) shire_id + local_tile_idx * ਅੰਕ_COMPUTE_SHIRES;

        int64_t max_iters;
        ਜੇ (k_splits > 1) {
            max_iters = (ਕੁੱਲ_ਪੰਕਤੀਆਂ + tiles_stride - 1) / tiles_stride;
        } ਨਹੀਂਤਾਂ {
            max_iters = (ਪੰਕਤੀ_ਅਧਾਰ >= ਕੁੱਲ_ਪੰਕਤੀਆਂ) ? 0 : ((ਕੁੱਲ_ਪੰਕਤੀਆਂ - ਪੰਕਤੀ_ਅਧਾਰ - 1) / tiles_stride + 1);
        }

        ਲਈ (int64_t iter = 0; iter < max_iters; iter++) {
            ਅਚਲ int64_t ਪੰਕਤੀ      = ਪੰਕਤੀ_ਅਧਾਰ + iter * tiles_stride;
            ਅਚਲ ਪੂਰਨ     has_work = (ਪੰਕਤੀ < ਕੁੱਲ_ਪੰਕਤੀਆਂ);

            ਜੇ (has_work) {
                ਅਚਲ int64_t iq3 = ਪੰਕਤੀ / (nhq * nq);
                ਅਚਲ int64_t rem = ਪੰਕਤੀ % (nhq * nq);
                ਅਚਲ int64_t iq2 = rem / nq;
                ਅਚਲ int64_t ik2 = iq2 / gqa_ਅਨੁਪਾਤ;

                ਅਚਲ ਅੱਖਰ * k_ਸਿਰ = k_ਅੰਕੜੇ + ik2 * k->nb[2] + iq3 * k->nb[3];

                ਲਈ (int64_t kv_ਅਧਾਰ = kv_ਸ਼ੁਰੂ; kv_ਅਧਾਰ < kv_ਸਿਰਾ; kv_ਅਧਾਰ += TILE_KV) {
                    ਅਚਲ int64_t kv_ਗਿਣਤੀ = (kv_ਅਧਾਰ + TILE_KV <= nk) ? TILE_KV : (nk - kv_ਅਧਾਰ);

                    ਲਈ (int64_t dk_chunk = 0; dk_chunk < dk; dk_chunk += TILE_K) {
                        ਪੂਰਨ buf = chunk_id & 1;

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
                        ਜੇ (chunk_id >= 2) {
                            et_sem_wait(ET_BARRIER_MINION);
                        }

                        // Prefetch K data for this chunk
                        prefetch_kv_ਮੰਜ਼ਿਲ_l2(k_ਸਿਰ, kv_ਅਧਾਰ, dk_chunk, kv_ਗਿਣਤੀ, k->nb[1]);

                        pack_k_for_transpose16(scp_kp[buf], k_ਸਿਰ, kv_ਅਧਾਰ, dk_chunk, kv_ਗਿਣਤੀ, k->nb[1]);

                        FENCE;
                        flush_ਮੰਜ਼ਿਲ_l2(scp_kp[buf], 16, 64);
                        flush_ਮੰਜ਼ਿਲ_l2((et_fp16_t *) ((ਅੱਖਰ *) scp_kp[buf] + 1024), 16, 64);
                        WAIT_CACHEOPS;

                        // Signal: this buf is ready for hart 0 to consume.
                        et_sem_post(ET_BARRIER_MINION);

                        chunk_id++;
                    }
                }
            }

            // Shire barriers for split-KV merge (hart 1 is a passive arrival).
            ਜੇ (k_splits > 1) {
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
        ਲਈ (uint32_t d = 0; d < drain; d++) {
            et_sem_wait(ET_BARRIER_MINION);
        }

        // FENCE;
        // et_barrier(ET_BARRIER_GLOBAL);
        ਵਾਪਸ 0;
    }

    // Hart 0: tensor engine compute
#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ UBERKERNEL_SUPPRESS_SCP_ਤਿਆਰੀ
    ਤਿਆਰੀ_cache_scp();
#ਜੇ_ਅੰਤ
    CLEAR_TENSOR_ERROR;

    // Q converted to F16 (one row at a time)
    et_fp16_t q_f16[FA_DK_MAX] __attribute__((aligned(64)));

    // Score buffer for QK^T output (16 scores per KV tile)
    ਦਸ਼ਮਲਵ scores[TILE_KV] __attribute__((aligned(64)));

    // Small buffers for V accumulation
    et_fp16_t w_f16_buf[32] __attribute__((aligned(64)));       // 64 bytes
    et_fp16_t vpanel_buf[8 * 32] __attribute__((aligned(64)));  // 512 bytes

    ਦਸ਼ਮਲਵ * acc = (ਦਸ਼ਮਲਵ *) et_shire_l2scp_local(scp_ਅਧਾਰ + SCP_ACC_OFF);

    uint32_t chunk_id = 0;

    // Iter-based outer loop (matches hart 1). When k_splits > 1 all teams
    // in a shire iterate the same number of times so the per-row shire
    // barriers stay balanced; iterations with row >= total_rows skip the
    // compute but still participate in the barriers.
    ਅਚਲ int64_t hart0_ਪੰਕਤੀ_ਅਧਾਰ = (int64_t) shire_id + local_tile_idx * ਅੰਕ_COMPUTE_SHIRES;
    int64_t       hart0_max_iters;
    ਜੇ (k_splits > 1) {
        hart0_max_iters = (ਕੁੱਲ_ਪੰਕਤੀਆਂ + tiles_stride - 1) / tiles_stride;
    } ਨਹੀਂਤਾਂ {
        hart0_max_iters = (hart0_ਪੰਕਤੀ_ਅਧਾਰ >= ਕੁੱਲ_ਪੰਕਤੀਆਂ) ? 0 : ((ਕੁੱਲ_ਪੰਕਤੀਆਂ - hart0_ਪੰਕਤੀ_ਅਧਾਰ - 1) / tiles_stride + 1);
    }

    ਲਈ (int64_t iter = 0; iter < hart0_max_iters; iter++) {
        ਅਚਲ int64_t ਪੰਕਤੀ = hart0_ਪੰਕਤੀ_ਅਧਾਰ + iter * tiles_stride;
        ਜੇ (ਪੰਕਤੀ >= ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
            // No-work iteration: only participate in barriers (k_splits > 1).
            ਜੇ (k_splits > 1) {
                et_barrier(ET_BARRIER_SHIRE);  // A
                et_barrier(ET_BARRIER_SHIRE);  // B
            }
            ਜਾਰੀ;
        }

        ਅਚਲ int64_t iq3 = ਪੰਕਤੀ / (nhq * nq);
        ਅਚਲ int64_t rem = ਪੰਕਤੀ % (nhq * nq);
        ਅਚਲ int64_t iq2 = rem / nq;
        ਅਚਲ int64_t iq1 = rem % nq;
        ਅਚਲ int64_t ik2 = iq2 / gqa_ਅਨੁਪਾਤ;

        // Read Q row (F32) and convert to F16
        ਅਚਲ ਦਸ਼ਮਲਵ * pq = (ਅਚਲ ਦਸ਼ਮਲਵ *) (q_ਅੰਕੜੇ + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);
        convert_q_ਪੰਕਤੀ_f32_ਮੰਜ਼ਿਲ_f16(q_f16, pq, dk);

        // V base for this head + batch (K packing handled by hart 1)
        ਅਚਲ ਅੱਖਰ * v_ਸਿਰ = v_ਅੰਕੜੇ + ik2 * v->nb[2] + iq3 * v->nb[3];

        // Output pointer
        ਦਸ਼ਮਲਵ * ਬਾਹਰਲਾ = (ਦਸ਼ਮਲਵ *) (dst_ਅੰਕੜੇ + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        zero_acc_vec(acc, dv);
        ਦਸ਼ਮਲਵ        M         = ET_NEG_INF_F;
        ਦਸ਼ਮਲਵ        S         = 0.0f;
        ਅਚਲ ਅੱਖਰ * ਨਕਾਬ_ਅਧਾਰ = has_ਨਕਾਬ ? get_ਨਕਾਬ_ਪੰਕਤੀ_ਅਧਾਰ(ਨਕਾਬ, iq1, iq2, iq3) : (ਅਚਲ ਅੱਖਰ *) 0;

        // Flush Q_f16 to L2 so tensor_load can see it
        FENCE;
        flush_ਮੰਜ਼ਿਲ_l2(q_f16, (dk * 2 + 63) / 64, 64);
        WAIT_CACHEOPS;

        ਲਈ (int64_t kv_ਅਧਾਰ = kv_ਸ਼ੁਰੂ; kv_ਅਧਾਰ < kv_ਸਿਰਾ; kv_ਅਧਾਰ += TILE_KV) {
            ਅਚਲ int64_t kv_ਗਿਣਤੀ = (kv_ਅਧਾਰ + TILE_KV <= nk) ? TILE_KV : (nk - kv_ਅਧਾਰ);

            // Set tensor_mask for partial tiles
            ਜੇ (kv_ਗਿਣਤੀ < TILE_KV) {
                uint64_t tmask = (1ULL << kv_ਗਿਣਤੀ) - 1;
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
            ਅਚਲ int64_t  n_dk_chunks = dk / TILE_K;
            ਅਚਲ uint64_t K_BUFS[2]   = {
                (uint64_t) B_L1_ਸ਼ੁਰੂ,         // 16..31
                (uint64_t) (B_L1_ਸ਼ੁਰੂ + 16),  // 32..47
            };

            // Preload entire Q row into A_L1[0..n_dk_chunks-1] (one tensor_load,
            // one wait, regardless of dk).
            tensor_ਲੱਦੋ(false, false, A_L1_ਸ਼ੁਰੂ, TENSOR_ਲੱਦੋ_PLAIN, 0, (uint64_t) q_f16, 0,
                        (uint64_t) (n_dk_chunks - 1), 64, 0);

            // Prologue: wait hart 1's K[0], issue K[0] load, wait both loads.
            {
                ਪੂਰਨ buf = chunk_id & 1;
                et_sem_wait(ET_BARRIER_MINION);
                tensor_ਲੱਦੋ(false, false, K_BUFS[0], TENSOR_ਲੱਦੋ_TRANSPOSE16, 0, (uint64_t) scp_kp[buf], 0, 15, 64, 1);
                tensor_wait(TENSOR_ਲੱਦੋ_WAIT_0);  // Q row complete
                tensor_wait(TENSOR_ਲੱਦੋ_WAIT_1);  // K[0] complete
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
            ਲਈ (int64_t i = 1; i < n_dk_chunks; i++) {
                ਪੂਰਨ buf         = chunk_id & 1;
                ਪੂਰਨ k_ਥਾਂ_prev = (ਪੂਰਨ) ((i - 1) & 1);
                ਪੂਰਨ k_ਥਾਂ      = (ਪੂਰਨ) (i & 1);

                et_sem_wait(ET_BARRIER_MINION);
                tensor_ਲੱਦੋ(false, false, K_BUFS[k_ਥਾਂ], TENSOR_ਲੱਦੋ_TRANSPOSE16, 0, (uint64_t) scp_kp[buf], 0, 15, 64,
                            1);

                tensor_fma((kv_ਗਿਣਤੀ < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_ਥਾਂ_prev],
                           (uint64_t) (i - 1), TENSOR_FMA_OP_FP16, (i == 1));

                tensor_wait(TENSOR_ਲੱਦੋ_WAIT_1);  // K[i] in L1
                et_sem_post(ET_BARRIER_MINION);   // release scp_kp[buf] EARLY
                tensor_wait(TENSOR_FMA_WAIT);     // then wait FMA[i-1]
                chunk_id++;
            }

            // Epilogue: FMA on the last chunk (no overlapping load).
            {
                ਪੂਰਨ k_ਥਾਂ_ਆਖਰੀ = (ਪੂਰਨ) ((n_dk_chunks - 1) & 1);
                tensor_fma((kv_ਗਿਣਤੀ < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_ਥਾਂ_ਆਖਰੀ],
                           (uint64_t) (n_dk_chunks - 1), TENSOR_FMA_OP_FP16, (n_dk_chunks == 1));
                tensor_wait(TENSOR_FMA_WAIT);
            }

            // Prefetch V rows for this tile.
            // Only useful for the partial-tile path below
            ਜੇ (kv_ਗਿਣਤੀ < TILE_KV) {
                ਲਈ (int64_t d = 0; d < dv; d += 32) {
                    prefetch_kv_ਮੰਜ਼ਿਲ_l2(v_ਸਿਰ, kv_ਅਧਾਰ, d, kv_ਗਿਣਤੀ, v->nb[1]);
                }
            }

            // Extract QK^T scores from vector register file
            __asm__ ਅਸਥਿਰ("" ::: "f0", "f1");
            {
                ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ _ms;
                __asm__ ਅਸਥਿਰ(
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

            ਦਸ਼ਮਲਵ weights[TILE_KV] __attribute__((aligned(64)));
            {
                // A1: apply mask to scores, pad unused slots
                ਲਈ (int64_t j = 0; j < kv_ਗਿਣਤੀ; ++j) {
                    ਜੇ (has_ਨਕਾਬ) {
                        ਦਸ਼ਮਲਵ mv = get_ਨਕਾਬ_val_from_ਅਧਾਰ(ਨਕਾਬ, ਨਕਾਬ_ਅਧਾਰ, kv_ਅਧਾਰ + j);
                        ਜੇ (mv == ET_NEG_INF_F || mv != mv) {
                            scores[j] = ET_NEG_INF_F;
                        } ਨਹੀਂਤਾਂ {
                            scores[j] += mv;
                        }
                    }
                }
                ਲਈ (int64_t j = kv_ਗਿਣਤੀ; j < TILE_KV; ++j) {
                    scores[j] = ET_NEG_INF_F;
                }

                // A1b: SIMD horizontal max across all 16 scores
                ਦਸ਼ਮਲਵ tile_max;
                {
                    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ _ms;
                    __asm__ ਅਸਥਿਰ(
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

                ਜੇ (tile_max > ET_NEG_INF_F) {
                    // A2: rescale accumulator if this tile has a new global max
                    ਜੇ (tile_max > M) {
                        ਦਸ਼ਮਲਵ rescale = et_exp2f((M - tile_max) * 1.4426950408889634f);
                        scale_acc_vec(acc, dv, rescale);
                        S *= rescale;
                        M = tile_max;
                    }

                    // A3: SIMD exp2 + horizontal sum
                    // Interleaved: f2/f3 chains alternate to hide ALU latency.
                    // fexp.ps has multi-cycle latency — the two independent
                    // exp2 calls naturally pipeline.
                    {
                        ਅਚਲ ਦਸ਼ਮਲਵ   log2e = 1.4426950408889634f;
                        ਦਸ਼ਮਲਵ         S_tile;
                        ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ _ms;
                        __asm__ ਅਸਥਿਰ(
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
                        convert_q_ਪੰਕਤੀ_f32_ਮੰਜ਼ਿਲ_f16(w_f16_buf, weights, TILE_KV);

                        FENCE;
                        flush_ਮੰਜ਼ਿਲ_l2(w_f16_buf, 1, 64);
                        WAIT_CACHEOPS;

                        // Issue weights load (wait_id=0) and the first V chunk
                        // load (wait_id=1) concurrently. Weights comes from
                        // L2 SCP (just flushed); V[0] comes from DRAM via
                        // INTERLEAVE16 — running them in parallel hides the
                        // shorter load behind the longer one. For partial
                        // tiles, V is software-packed below — we only kick
                        // off the early V load on the full-tile fast path.
                        tensor_ਲੱਦੋ(false, false, A_L1_ਸ਼ੁਰੂ, TENSOR_ਲੱਦੋ_PLAIN, 0, (uint64_t) w_f16_buf, 0, 0, 64, 0);

                        ਅਚਲ ਪੂਰਨ       v_ਭਰਿਆ_tile = (kv_ਗਿਣਤੀ == TILE_KV);
                        ਅਚਲ uintptr_t v_ਅਧਾਰ      = (uintptr_t) v_ਸਿਰ + kv_ਅਧਾਰ * v->nb[1];
                        ਅਚਲ uint64_t  nb1_v       = (uint64_t) v->nb[1];
                        uint64_t        b_cur       = 8;

                        ਜੇ (v_ਭਰਿਆ_tile) {
                            tensor_ਲੱਦੋ(false, false, b_cur, TENSOR_ਲੱਦੋ_INTERLEAVE16, 0, (uint64_t) v_ਅਧਾਰ, 0, 7,
                                        nb1_v, 1);
                        }

                        tensor_wait(TENSOR_ਲੱਦੋ_WAIT_0);      // weights in A_L1
                        ਜੇ (v_ਭਰਿਆ_tile) {
                            tensor_wait(TENSOR_ਲੱਦੋ_WAIT_1);  // V[0] in b_cur
                        }

                        // B2: process dv in chunks of 16
                        ਜੇ (v_ਭਰਿਆ_tile) {
                            ਲਈ (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                ਅਚਲ uint64_t b_nxt = b_cur ^ 24;

                                ਜੇ (dv_off + 16 < dv) {
                                    tensor_ਲੱਦੋ(false, false, b_nxt, TENSOR_ਲੱਦੋ_INTERLEAVE16, 0,
                                                (uint64_t) (v_ਅਧਾਰ + (dv_off + 16) * 2), 0, 7, nb1_v, 1);
                                }

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, b_cur, A_L1_ਸ਼ੁਰੂ,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ ਅਸਥਿਰ("" ::: "f0", "f1");
                                {
                                    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ _ms;
                                    __asm__ ਅਸਥਿਰ(
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

                                ਜੇ (dv_off + 16 < dv) {
                                    tensor_wait(TENSOR_ਲੱਦੋ_WAIT_1);
                                    b_cur = b_nxt;
                                }
                            }
                        } ਨਹੀਂਤਾਂ {
                            // Partial tile: software pack, no pipeline
                            ਲਈ (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                pack_v_interleaved(vpanel_buf, v_ਸਿਰ, kv_ਅਧਾਰ, dv_off, kv_ਗਿਣਤੀ, v->nb[1]);
                                FENCE;
                                flush_ਮੰਜ਼ਿਲ_l2(vpanel_buf, 8, 64);
                                WAIT_CACHEOPS;
                                tensor_ਲੱਦੋ(false, false, B_L1_ਸ਼ੁਰੂ, TENSOR_ਲੱਦੋ_PLAIN, 0, (uint64_t) vpanel_buf, 0, 7,
                                            64, 0);
                                tensor_wait(TENSOR_ਲੱਦੋ_WAIT_0);

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, B_L1_ਸ਼ੁਰੂ, A_L1_ਸ਼ੁਰੂ,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ ਅਸਥਿਰ("" ::: "f0", "f1");
                                {
                                    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ _ms;
                                    __asm__ ਅਸਥਿਰ(
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
        ਜੇ (k_splits > 1) {
            // Publish our partial.
            ਅਸਥਿਰ ਦਸ਼ਮਲਵ * my_stats = (ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) et_shire_l2scp_local(scp_ਅਧਾਰ + SCP_STATS_OFF);
            my_stats[0]               = M;
            my_stats[1]               = S;
            FENCE;
            evict_range_ਮੰਜ਼ਿਲ_l2(acc, (int64_t) dv * (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ));
            evict_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) my_stats, 1, 64);
            WAIT_CACHEOPS;

            // A: team members have all written their partials.
            et_barrier(ET_BARRIER_SHIRE);

            ਜੇ (k_ਵੰਡੋ == 0) {
                // Online softmax merge: fold peers 1..k_splits-1 into our
                // own (M_running, S_running, acc). For each peer p:
                //   M_new    = max(M_running, M_p)
                //   α_own    = exp2((M_running - M_new) * log2e)
                //   α_p      = exp2((M_p     - M_new) * log2e)
                //   acc[d]   = α_own * acc[d] + α_p * peer_acc[d]
                //   S_running = α_own * S_running + α_p * S_p
                ਦਸ਼ਮਲਵ       M_running = M;
                ਦਸ਼ਮਲਵ       S_running = S;
                ਅਚਲ ਦਸ਼ਮਲਵ log2e     = 1.4426950408889634f;

                ਲਈ (int64_t p = 1; p < k_splits; p++) {
                    uint64_t         peer_scp   = (local_tile_idx * k_splits + p) * SCP_ਪ੍ਰਤੀ_MINION;
                    ਅਸਥਿਰ ਦਸ਼ਮਲਵ * peer_stats = (ਅਸਥਿਰ ਦਸ਼ਮਲਵ *) et_shire_l2scp_local(peer_scp + SCP_STATS_OFF);
                    ਦਸ਼ਮਲਵ *          peer_acc   = (ਦਸ਼ਮਲਵ *) et_shire_l2scp_local(peer_scp + SCP_ACC_OFF);

                    // Drop stale L1D copies before reading peer's data.
                    evict_ਮੰਜ਼ਿਲ_l2((ਅਚਲ ਖਾਲੀ *) peer_stats, 1, 64);
                    evict_range_ਮੰਜ਼ਿਲ_l2(peer_acc, (int64_t) dv * (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                    WAIT_CACHEOPS;

                    ਅਚਲ ਦਸ਼ਮਲਵ M_p = peer_stats[0];
                    ਅਚਲ ਦਸ਼ਮਲਵ S_p = peer_stats[1];

                    ਅਚਲ ਦਸ਼ਮਲਵ M_new     = (M_p > M_running) ? M_p : M_running;
                    ਅਚਲ ਦਸ਼ਮਲਵ alpha_own = (M_running == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_running - M_new) * log2e);
                    ਅਚਲ ਦਸ਼ਮਲਵ alpha_p   = (M_p == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_p - M_new) * log2e);

                    ਮਿਲਾਓ_rescale_ਜੋੜੋ_asm(acc, peer_acc, dv, alpha_own, alpha_p);

                    S_running = alpha_own * S_running + alpha_p * S_p;
                    M_running = M_new;
                }

                ਅਚਲ ਦਸ਼ਮਲਵ S_inv = (S_running == 0.0f) ? 0.0f : et_fdiv(1.0f, S_running);
                normalize_store_vec(ਬਾਹਰਲਾ, acc, dv, S_inv, use_fast_store);
            }

            // B: reducer is done, team may reuse its acc/stats slabs.
            et_barrier(ET_BARRIER_SHIRE);
        } ਨਹੀਂਤਾਂ {
            // k_splits == 1 fast path — this minion owns the full row.
            ਅਚਲ ਦਸ਼ਮਲਵ S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
            normalize_store_vec(ਬਾਹਰਲਾ, acc, dv, S_inv, use_fast_store);
        }
    }

    FENCE;
    ਵਾਪਸ 0;
}
