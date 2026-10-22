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

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"
#అన్తర్భావ "tensor.h"

#అన్తర్భావ <etsoc/common/utils.h>
#అన్తర్భావ <stdint.h>
#అన్తర్భావ <string.h>

#నిర్వచించు అంకె_COMPUTE_SHIRES 32
#నిర్వచించు MINIONS_చొప్పున_SHIRE  32

// QK^T tiles: 16 KV positions at a time, K in chunks of 32 F16
#నిర్వచించు TILE_KV 16
#నిర్వచించు TILE_K  32

// L1 scratchpad layout: A (Q) in lines 0-15, B (K interleaved) in lines 16-31
#నిర్వచించు A_L1_ప్రారంభం 0
#నిర్వచించు B_L1_ప్రారంభం 16

// Max head dimensions
#నిర్వచించు FA_DV_MAX 512  // max value head dim (dv)
#నిర్వచించు FA_DK_MAX 512  // max key head dim (dk) - some models use hsk > hsv

రకనిర్వచనం uint16_t et_fp16_t;

#నిర్వచించు ET_NEG_INF_F (-3.402823466e+38f)

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
#నిర్వచించు SCP_ACC_OFF     0
#నిర్వచించు SCP_ACC_STRIDE  (FA_DV_MAX * పరిమాణం(దశాంశం))       // 2048
#నిర్వచించు SCP_KPANEL_కొలత (32 * 32 * పరిమాణం(et_fp16_t))     // 2048
#నిర్వచించు SCP_KP0_OFF     SCP_ACC_STRIDE                    // 2048
#నిర్వచించు SCP_KP1_OFF     (SCP_KP0_OFF + SCP_KPANEL_కొలత)   // 4096
#నిర్వచించు SCP_STATS_OFF   (SCP_KP1_OFF + SCP_KPANEL_కొలత)   // 6144
#నిర్వచించు SCP_STATS_కొలత  64                                // own cache line
#నిర్వచించు SCP_చొప్పున_MINION  (SCP_STATS_OFF + SCP_STATS_కొలత)  // 6208

నిర్మాణం ggml_et_flash_attn_ext_params {
    నిర్మాణం ggml_tensor src0;  // Q (F32)
    నిర్మాణం ggml_tensor src1;  // K (F16)
    నిర్మాణం ggml_tensor src2;  // V (F16)
    నిర్మాణం ggml_tensor ముసుగు;  // mask (F16 or F32), zeroed when absent
    నిర్మాణం ggml_tensor dst;   // Output (F32)
    దశాంశం              scale;
    int32_t            has_ముసుగు;
};

స్థిర అంతఃవరుస దశాంశం get_ముసుగు_val(స్థిరరాశి నిర్మాణం ggml_tensor * ముసుగు, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    స్థిరరాశి అక్షరం * పునాది = (స్థిరరాశి అక్షరం *) ముసుగు->దత్తాంశం + iq1 * ముసుగు->nb[1] + (iq2 % ముసుగు->ne[2]) * ముసుగు->nb[2] +
                        (iq3 % ముసుగు->ne[3]) * ముసుగు->nb[3];

    ఒకవేళ (ముసుగు->type == GGML_TYPE_F32) {
        తిరిగి *(స్థిరరాశి దశాంశం *) (పునాది + ik1 * ముసుగు->nb[0]);
    }
    తిరిగి fp16_గమ్యం_fp32(*(స్థిరరాశి uint16_t *) (పునాది + ik1 * ముసుగు->nb[0]));
}

స్థిర అంతఃవరుస స్థిరరాశి అక్షరం * get_ముసుగు_అడ్డువరుస_పునాది(స్థిరరాశి నిర్మాణం ggml_tensor * ముసుగు, int64_t iq1, int64_t iq2, int64_t iq3) {
    తిరిగి (స్థిరరాశి అక్షరం *) ముసుగు->దత్తాంశం + iq1 * ముసుగు->nb[1] + (iq2 % ముసుగు->ne[2]) * ముసుగు->nb[2] +
           (iq3 % ముసుగు->ne[3]) * ముసుగు->nb[3];
}

స్థిర అంతఃవరుస దశాంశం get_ముసుగు_val_from_పునాది(స్థిరరాశి నిర్మాణం ggml_tensor * ముసుగు, స్థిరరాశి అక్షరం * పునాది, int64_t ik1) {
    ఒకవేళ (ముసుగు->type == GGML_TYPE_F32) {
        తిరిగి *(స్థిరరాశి దశాంశం *) (పునాది + ik1 * ముసుగు->nb[0]);
    }
    తిరిగి fp16_గమ్యం_fp32(*(స్థిరరాశి uint16_t *) (పునాది + ik1 * ముసుగు->nb[0]));
}

// Pack K rows for TensorLoadTranspose16 (even/odd deinterleave)
స్థిర అంతఃవరుస శూన్య __attribute__((always_inline)) pack_k_for_transpose16(et_fp16_t *  బహిర్గతం,
                                                                         స్థిరరాశి అక్షరం * k_పునాది,
                                                                         int64_t      kv_ప్రారంభం,
                                                                         int64_t      dk_ప్రారంభం,
                                                                         int64_t      kv_లెక్క,
                                                                         int64_t      nb1_k) {
    గుర్తులేని దీర్ఘ old_ముసుగు;
    __asm__ అస్థిర(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        : [ms] "=&r"(old_ముసుగు)
        :
        :);

    కొరకు (పూర్ణాంకం j = 0; j < (పూర్ణాంకం) kv_లెక్క; ++j) {
        స్థిరరాశి et_fp16_t * k_అడ్డువరుస    = (స్థిరరాశి et_fp16_t *) (k_పునాది + (kv_ప్రారంభం + j) * nb1_k) + dk_ప్రారంభం;
        et_fp16_t *       even_అడ్డువరుస = బహిర్గతం + (j * 2) * 32;
        et_fp16_t *       odd_అడ్డువరుస  = బహిర్గతం + (j * 2 + 1) * 32;
        __asm__ అస్థిర(
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
            : [src0] "r"(k_అడ్డువరుస), [src1] "r"(k_అడ్డువరుస + 16), [even] "r"(even_అడ్డువరుస), [odd] "r"(odd_అడ్డువరుస)
            : "f2", "f3", "f4", "f5", "f6", "f7", "memory");
    }

    __asm__ అస్థిర("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_ముసుగు));

    కొరకు (పూర్ణాంకం j = (పూర్ణాంకం) kv_లెక్క; j < TILE_KV; ++j) {
        et_fp16_t * even_అడ్డువరుస = బహిర్గతం + (j * 2) * 32;
        et_fp16_t * odd_అడ్డువరుస  = బహిర్గతం + (j * 2 + 1) * 32;
        కొరకు (పూర్ణాంకం l = 0; l < TILE_K / 2; ++l) {
            even_అడ్డువరుస[l] = 0;
            odd_అడ్డువరుస[l]  = 0;
        }
    }
}

// Build interleaved B panel for TensorFMA16A32 (weights @ V).
స్థిర అంతఃవరుస శూన్య __attribute__((always_inline)) pack_v_interleaved(et_fp16_t *  బహిర్గతం,
                                                                     స్థిరరాశి అక్షరం * v_తల,
                                                                     int64_t      kv_పునాది,
                                                                     int64_t      dv_ప్రారంభం,
                                                                     int64_t      kv_లెక్క,
                                                                     int64_t      nb1_v) {
    కొరకు (పూర్ణాంకం k = 0; k < TILE_KV; ++k) {
        స్థిరరాశి పూర్ణాంకం         l   = k >> 1;
        స్థిరరాశి పూర్ణాంకం         r   = k & 1;
        et_fp16_t * స్థిరరాశి dst = బహిర్గతం + l * 32 + r;
        ఒకవేళ (k < (పూర్ణాంకం) kv_లెక్క) {
            స్థిరరాశి et_fp16_t * v_అడ్డువరుస = (స్థిరరాశి et_fp16_t *) (v_తల + (kv_పునాది + k) * nb1_v) + dv_ప్రారంభం;
            కొరకు (పూర్ణాంకం n = 0; n < 16; ++n) {
                dst[n * 2] = v_అడ్డువరుస[n];
            }
        } లేకపోతే {
            కొరకు (పూర్ణాంకం n = 0; n < 16; ++n) {
                dst[n * 2] = 0;
            }
        }
    }
}

// Prefetch KV rows for one chunk into L2.
స్థిర అంతఃవరుస శూన్య __attribute__((always_inline)) prefetch_kv_గమ్యం_l2(స్థిరరాశి అక్షరం * తల,
                                                                    int64_t      kv_ప్రారంభం,
                                                                    int64_t      d_ప్రారంభం,
                                                                    int64_t      kv_లెక్క,
                                                                    int64_t      nb1) {
    స్థిరరాశి శూన్య * పునాది = (స్థిరరాశి శూన్య *) (తల + kv_ప్రారంభం * nb1 + d_ప్రారంభం * 2);
    l2_prefetch(పునాది, (uint64_t) kv_లెక్క, (uint64_t) nb1);
}

స్థిర అంతఃవరుస శూన్య __attribute__((always_inline)) convert_q_అడ్డువరుస_f32_గమ్యం_f16(et_fp16_t *   dst,
                                                                           స్థిరరాశి దశాంశం * src,
                                                                           int64_t       n) {
    స్థిర స్థిరరాశి int32_t __attribute__((aligned(32))) offsets[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };

    గుర్తులేని దీర్ఘ old_ముసుగు;
    __asm__ అస్థిర(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 0xFF      \n\t"
        "flw.ps    f1, 0(%[offs])    \n\t"
        : [ms] "=&r"(old_ముసుగు)
        : [offs] "r"(offsets)
        : "f1");

    కొరకు (int64_t d = 0; d < n; d += 8) {
        __asm__ అస్థిర(
            "flw.ps      f2, 0(%[src])    \n\t"
            "fcvt.f16.ps f3, f2           \n\t"
            "fsch.ps     f3, f1(%[dst])   \n\t"
            :
            : [src] "r"(src + d), [dst] "r"(dst + d)
            : "f2", "f3", "memory");
    }

    __asm__ అస్థిర("mova.m.x  %[ms]             \n\t" : : [ms] "r"(old_ముసుగు));
}

స్థిర అంతఃవరుస శూన్య __attribute__((always_inline)) zero_acc_vec(దశాంశం * acc, int64_t dv) {
    స్థిరరాశి దశాంశం   zero = 0.0f;
    గుర్తులేని దీర్ఘ old_ముసుగు;
    __asm__ అస్థిర("mova.x.m %0" : "=r"(old_ముసుగు));
    __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");
    __asm__ అస్థిర("fbc.ps  f2, 0(%[z])" ::[z] "r"(&zero) : "f2");

    కొరకు (int64_t d = 0; d < dv; d += 8) {
        __asm__ అస్థిర("fsw.ps  f2, 0(%[a])     \n\t" ::[a] "r"(acc + d) : "f2", "memory");
    }

    __asm__ అస్థిర("mova.m.x %0" ::"r"(old_ముసుగు));
}

స్థిర అంతఃవరుస శూన్య __attribute__((always_inline)) scale_acc_vec(దశాంశం * acc, int64_t dv, దశాంశం scale) {
    గుర్తులేని దీర్ఘ old_ముసుగు;
    __asm__ అస్థిర("mova.x.m %0" : "=r"(old_ముసుగు));
    __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");

    కొరకు (int64_t d = 0; d < dv; d += 8) {
        __asm__ అస్థిర(
            "fbc.ps    f2, 0(%[s])    \n\t"
            "flw.ps    f3, 0(%[a])    \n\t"
            "fmul.ps   f3, f3, f2     \n\t"
            "fsw.ps    f3, 0(%[a])    \n\t"
            :
            : [s] "r"(&scale), [a] "r"(acc + d)
            : "f2", "f3", "memory");
    }

    __asm__ అస్థిర("mova.m.x %0" ::"r"(old_ముసుగు));
}

స్థిర అంతఃవరుస శూన్య __attribute__((always_inline)) normalize_store_vec(దశాంశం * బహిర్గతం,
                                                                      దశాంశం * acc,
                                                                      int64_t dv,
                                                                      దశాంశం   inv,
                                                                      పూర్ణాంకం     use_fast_store) {
    గుర్తులేని దీర్ఘ old_ముసుగు;
    __asm__ అస్థిర("mova.x.m %0" : "=r"(old_ముసుగు));
    __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");

    కొరకు (int64_t d = 0; d < dv; d += 8) {
        __asm__ అస్థిర(
            "fbc.ps    f2, 0(%[inv])   \n\t"
            "flw.ps    f3, 0(%[a])     \n\t"
            "fmul.ps   f3, f3, f2      \n\t"
            "fsw.ps    f3, 0(%[a])     \n\t"
            :
            : [inv] "r"(&inv), [a] "r"(acc + d)
            : "f2", "f3", "memory");
        ఒకవేళ (use_fast_store) {
            __asm__ అస్థిర(
                "flw.ps  f4, 0(%[a])     \n\t"
                "fsw.ps  f4, 0(%[o])     \n\t"
                :
                : [a] "r"(acc + d), [o] "r"(బహిర్గతం + d)
                : "f4", "memory");
        } లేకపోతే {
            atomic_store_f32((అస్థిర దశాంశం *) &బహిర్గతం[d + 0], acc[d + 0]);
            atomic_store_f32((అస్థిర దశాంశం *) &బహిర్గతం[d + 1], acc[d + 1]);
            atomic_store_f32((అస్థిర దశాంశం *) &బహిర్గతం[d + 2], acc[d + 2]);
            atomic_store_f32((అస్థిర దశాంశం *) &బహిర్గతం[d + 3], acc[d + 3]);
            atomic_store_f32((అస్థిర దశాంశం *) &బహిర్గతం[d + 4], acc[d + 4]);
            atomic_store_f32((అస్థిర దశాంశం *) &బహిర్గతం[d + 5], acc[d + 5]);
            atomic_store_f32((అస్థిర దశాంశం *) &బహిర్గతం[d + 6], acc[d + 6]);
            atomic_store_f32((అస్థిర దశాంశం *) &బహిర్గతం[d + 7], acc[d + 7]);
        }
    }

    __asm__ అస్థిర("mova.m.x %0" ::"r"(old_ముసుగు));
}

స్థిర అంతఃవరుస size_t tensor_బైట్లు_fa(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    తిరిగి (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

// Evict a byte range from L1D to L2 SCP, splitting into batches of ≤16
// cache lines (the hw limit for evict_to_l2). Use before a barrier when
// another minion in the shire needs to read the region, or after a barrier
// on the reader side to drop stale L1D copies before reading peer data.
స్థిర అంతఃవరుస శూన్య __attribute__((always_inline)) evict_range_గమ్యం_l2(స్థిరరాశి శూన్య * addr, int64_t బైట్లు) {
    ఒకవేళ (బైట్లు <= 0) {
        తిరిగి;
    }
    int64_t      పంక్తులు = (బైట్లు + 63) / 64;
    స్థిరరాశి అక్షరం * p     = (స్థిరరాశి అక్షరం *) addr;
    అయితే (పంక్తులు > 0) {
        int64_t batch = పంక్తులు > 16 ? 16 : పంక్తులు;
        evict_గమ్యం_l2((స్థిరరాశి శూన్య *) p, (uint64_t) batch, 64);
        p += batch * 64;
        పంక్తులు -= batch;
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
స్థిర అంతఃవరుస శూన్య __attribute__((always_inline)) విలీనం_rescale_కూడు_asm(దశాంశం *       acc,
                                                                        స్థిరరాశి దశాంశం * peer_acc,
                                                                        int64_t       dv,
                                                                        దశాంశం         alpha_own,
                                                                        దశాంశం         alpha_peer) {
    గుర్తులేని దీర్ఘ old_ముసుగు;
    __asm__ అస్థిర(
        "mova.x.m  %[ms]              \n\t"
        "mov.m.x   m0, x0, 0xFF       \n\t"
        "fbc.ps    f4, 0(%[ao])       \n\t"  // broadcast alpha_own
        "fbc.ps    f5, 0(%[ap])       \n\t"  // broadcast alpha_peer
        : [ms] "=&r"(old_ముసుగు)
        : [ao] "r"(&alpha_own), [ap] "r"(&alpha_peer)
        : "f4", "f5");

    కొరకు (int64_t d = 0; d < dv; d += 8) {
        __asm__ అస్థిర(
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

    __asm__ అస్థిర("mova.m.x %0" ::"r"(old_ముసుగు));
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_flash_attn_ext_params * params, శూన్య * env) {
    (శూన్య) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    ఒకవేళ (shire_id >= అంకె_COMPUTE_SHIRES) {
        తిరిగి 0;
    }

    స్థిరరాశి పూర్ణాంకం is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    నిర్మాణం ggml_tensor * q        = &params->src0;
    నిర్మాణం ggml_tensor * k        = &params->src1;
    నిర్మాణం ggml_tensor * v        = &params->src2;
    నిర్మాణం ggml_tensor * dst      = &params->dst;
    స్థిరరాశి int32_t        has_ముసుగు = params->has_ముసుగు;
    నిర్మాణం ggml_tensor * ముసుగు     = has_ముసుగు ? &params->ముసుగు : (నిర్మాణం ggml_tensor *) 0;

    స్థిరరాశి అక్షరం * q_దత్తాంశం   = (స్థిరరాశి అక్షరం *) q->దత్తాంశం;
    స్థిరరాశి అక్షరం * k_దత్తాంశం   = (స్థిరరాశి అక్షరం *) k->దత్తాంశం;
    స్థిరరాశి అక్షరం * v_దత్తాంశం   = (స్థిరరాశి అక్షరం *) v->దత్తాంశం;
    అక్షరం *       dst_దత్తాంశం = (అక్షరం *) dst->దత్తాంశం;

    // et_barrier(ET_BARRIER_GLOBAL);
    evict_region_past_l2(q->దత్తాంశం, tensor_బైట్లు_fa(q));
    evict_region_past_l2(k->దత్తాంశం, tensor_బైట్లు_fa(k));
    evict_region_past_l2(v->దత్తాంశం, tensor_బైట్లు_fa(v));
    ఒకవేళ (ముసుగు) {
        evict_region_past_l2(ముసుగు->దత్తాంశం, tensor_బైట్లు_fa(ముసుగు));
    }
    et_barrier(ET_BARRIER_GLOBAL);

    స్థిరరాశి int64_t dk  = q->ne[0];
    స్థిరరాశి int64_t nq  = q->ne[1];
    స్థిరరాశి int64_t nhq = q->ne[2];
    స్థిరరాశి int64_t no  = q->ne[3];
    స్థిరరాశి int64_t nk  = k->ne[1];
    స్థిరరాశి int64_t nhk = k->ne[2];
    స్థిరరాశి int64_t dv  = v->ne[0];

    ఒకవేళ (dv > FA_DV_MAX || dk > FA_DK_MAX) {
        తిరిగి -1;
    }
    ఒకవేళ (k->nb[0] != 2 || v->nb[0] != 2) {
        తిరిగి -1;
    }
    ఒకవేళ ((dk % 8) != 0 || (dv % 16) != 0) {
        తిరిగి -1;
    }

    స్థిరరాశి int64_t gqa_నిష్పత్తి      = nhq / nhk;
    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు     = nq * nhq * no;
    స్థిరరాశి దశాంశం   scale          = params->scale;
    స్థిరరాశి పూర్ణాంకం     use_fast_store = (dv % 16 == 0);

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
    స్థిరరాశి int64_t nk_tiles      = (nk + TILE_KV - 1) / TILE_KV;
    స్థిరరాశి int64_t మొత్తం_minions = 2 * అంకె_COMPUTE_SHIRES * MINIONS_చొప్పున_SHIRE;
    int64_t       k_splits      = 1;
    ఒకవేళ (మొత్తం_అడ్డువరుసలు < మొత్తం_minions) {
        int64_t లక్ష్యం = మొత్తం_minions / మొత్తం_అడ్డువరుసలు;
        int64_t ks     = 1;
        అయితే (ks * 2 <= లక్ష్యం && ks * 2 <= MINIONS_చొప్పున_SHIRE && ks * 2 <= nk_tiles) {
            ks *= 2;
        }
        k_splits = ks;
    }

    స్థిరరాశి int64_t tiles_చొప్పున_shire = MINIONS_చొప్పున_SHIRE / k_splits;
    స్థిరరాశి int64_t k_విభజించు         = (int64_t) local_minion % k_splits;
    స్థిరరాశి int64_t local_tile_idx  = (int64_t) local_minion / k_splits;
    స్థిరరాశి int64_t tiles_stride    = (int64_t) అంకె_COMPUTE_SHIRES * tiles_చొప్పున_shire;

    // KV slab for this k_split. With k_splits=1 this is the full range.
    స్థిరరాశి int64_t tiles_చొప్పున_విభజించు_rounded = (nk_tiles + k_splits - 1) / k_splits;
    స్థిరరాశి int64_t tile_ప్రారంభం              = k_విభజించు * tiles_చొప్పున_విభజించు_rounded;
    int64_t       tile_చివర                = tile_ప్రారంభం + tiles_చొప్పున_విభజించు_rounded;
    ఒకవేళ (tile_చివర > nk_tiles) {
        tile_చివర = nk_tiles;
    }
    స్థిరరాశి int64_t kv_ప్రారంభం = tile_ప్రారంభం * TILE_KV;
    int64_t       kv_చివర   = tile_చివర * TILE_KV;
    ఒకవేళ (kv_చివర > nk) {
        kv_చివర = nk;
    }

    // L2 SCP pointers for this minion
    uint64_t    scp_పునాది  = local_minion * SCP_చొప్పున_MINION;
    et_fp16_t * scp_kp[2] = {
        (et_fp16_t *) et_shire_l2scp_local(scp_పునాది + SCP_KP0_OFF),
        (et_fp16_t *) et_shire_l2scp_local(scp_పునాది + SCP_KP1_OFF),
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
    ఒకవేళ (is_hart1) {
        uint32_t      chunk_id = 0;
        స్థిరరాశి int64_t అడ్డువరుస_పునాది = (int64_t) shire_id + local_tile_idx * అంకె_COMPUTE_SHIRES;

        int64_t max_iters;
        ఒకవేళ (k_splits > 1) {
            max_iters = (మొత్తం_అడ్డువరుసలు + tiles_stride - 1) / tiles_stride;
        } లేకపోతే {
            max_iters = (అడ్డువరుస_పునాది >= మొత్తం_అడ్డువరుసలు) ? 0 : ((మొత్తం_అడ్డువరుసలు - అడ్డువరుస_పునాది - 1) / tiles_stride + 1);
        }

        కొరకు (int64_t iter = 0; iter < max_iters; iter++) {
            స్థిరరాశి int64_t అడ్డువరుస      = అడ్డువరుస_పునాది + iter * tiles_stride;
            స్థిరరాశి పూర్ణాంకం     has_work = (అడ్డువరుస < మొత్తం_అడ్డువరుసలు);

            ఒకవేళ (has_work) {
                స్థిరరాశి int64_t iq3 = అడ్డువరుస / (nhq * nq);
                స్థిరరాశి int64_t rem = అడ్డువరుస % (nhq * nq);
                స్థిరరాశి int64_t iq2 = rem / nq;
                స్థిరరాశి int64_t ik2 = iq2 / gqa_నిష్పత్తి;

                స్థిరరాశి అక్షరం * k_తల = k_దత్తాంశం + ik2 * k->nb[2] + iq3 * k->nb[3];

                కొరకు (int64_t kv_పునాది = kv_ప్రారంభం; kv_పునాది < kv_చివర; kv_పునాది += TILE_KV) {
                    స్థిరరాశి int64_t kv_లెక్క = (kv_పునాది + TILE_KV <= nk) ? TILE_KV : (nk - kv_పునాది);

                    కొరకు (int64_t dk_chunk = 0; dk_chunk < dk; dk_chunk += TILE_K) {
                        పూర్ణాంకం buf = chunk_id & 1;

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
                        ఒకవేళ (chunk_id >= 2) {
                            et_sem_wait(ET_BARRIER_MINION);
                        }

                        // Prefetch K data for this chunk
                        prefetch_kv_గమ్యం_l2(k_తల, kv_పునాది, dk_chunk, kv_లెక్క, k->nb[1]);

                        pack_k_for_transpose16(scp_kp[buf], k_తల, kv_పునాది, dk_chunk, kv_లెక్క, k->nb[1]);

                        FENCE;
                        flush_గమ్యం_l2(scp_kp[buf], 16, 64);
                        flush_గమ్యం_l2((et_fp16_t *) ((అక్షరం *) scp_kp[buf] + 1024), 16, 64);
                        WAIT_CACHEOPS;

                        // Signal: this buf is ready for hart 0 to consume.
                        et_sem_post(ET_BARRIER_MINION);

                        chunk_id++;
                    }
                }
            }

            // Shire barriers for split-KV merge (hart 1 is a passive arrival).
            ఒకవేళ (k_splits > 1) {
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
        కొరకు (uint32_t d = 0; d < drain; d++) {
            et_sem_wait(ET_BARRIER_MINION);
        }

        // FENCE;
        // et_barrier(ET_BARRIER_GLOBAL);
        తిరిగి 0;
    }

    // Hart 0: tensor engine compute
#నిర్వచితం_కాకపోతే UBERKERNEL_SUPPRESS_SCP_అమరిక
    అమరిక_cache_scp();
#అయితే_ముగింపు
    CLEAR_TENSOR_ERROR;

    // Q converted to F16 (one row at a time)
    et_fp16_t q_f16[FA_DK_MAX] __attribute__((aligned(64)));

    // Score buffer for QK^T output (16 scores per KV tile)
    దశాంశం scores[TILE_KV] __attribute__((aligned(64)));

    // Small buffers for V accumulation
    et_fp16_t w_f16_buf[32] __attribute__((aligned(64)));       // 64 bytes
    et_fp16_t vpanel_buf[8 * 32] __attribute__((aligned(64)));  // 512 bytes

    దశాంశం * acc = (దశాంశం *) et_shire_l2scp_local(scp_పునాది + SCP_ACC_OFF);

    uint32_t chunk_id = 0;

    // Iter-based outer loop (matches hart 1). When k_splits > 1 all teams
    // in a shire iterate the same number of times so the per-row shire
    // barriers stay balanced; iterations with row >= total_rows skip the
    // compute but still participate in the barriers.
    స్థిరరాశి int64_t hart0_అడ్డువరుస_పునాది = (int64_t) shire_id + local_tile_idx * అంకె_COMPUTE_SHIRES;
    int64_t       hart0_max_iters;
    ఒకవేళ (k_splits > 1) {
        hart0_max_iters = (మొత్తం_అడ్డువరుసలు + tiles_stride - 1) / tiles_stride;
    } లేకపోతే {
        hart0_max_iters = (hart0_అడ్డువరుస_పునాది >= మొత్తం_అడ్డువరుసలు) ? 0 : ((మొత్తం_అడ్డువరుసలు - hart0_అడ్డువరుస_పునాది - 1) / tiles_stride + 1);
    }

    కొరకు (int64_t iter = 0; iter < hart0_max_iters; iter++) {
        స్థిరరాశి int64_t అడ్డువరుస = hart0_అడ్డువరుస_పునాది + iter * tiles_stride;
        ఒకవేళ (అడ్డువరుస >= మొత్తం_అడ్డువరుసలు) {
            // No-work iteration: only participate in barriers (k_splits > 1).
            ఒకవేళ (k_splits > 1) {
                et_barrier(ET_BARRIER_SHIRE);  // A
                et_barrier(ET_BARRIER_SHIRE);  // B
            }
            కొనసాగు;
        }

        స్థిరరాశి int64_t iq3 = అడ్డువరుస / (nhq * nq);
        స్థిరరాశి int64_t rem = అడ్డువరుస % (nhq * nq);
        స్థిరరాశి int64_t iq2 = rem / nq;
        స్థిరరాశి int64_t iq1 = rem % nq;
        స్థిరరాశి int64_t ik2 = iq2 / gqa_నిష్పత్తి;

        // Read Q row (F32) and convert to F16
        స్థిరరాశి దశాంశం * pq = (స్థిరరాశి దశాంశం *) (q_దత్తాంశం + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);
        convert_q_అడ్డువరుస_f32_గమ్యం_f16(q_f16, pq, dk);

        // V base for this head + batch (K packing handled by hart 1)
        స్థిరరాశి అక్షరం * v_తల = v_దత్తాంశం + ik2 * v->nb[2] + iq3 * v->nb[3];

        // Output pointer
        దశాంశం * బహిర్గతం = (దశాంశం *) (dst_దత్తాంశం + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        zero_acc_vec(acc, dv);
        దశాంశం        M         = ET_NEG_INF_F;
        దశాంశం        S         = 0.0f;
        స్థిరరాశి అక్షరం * ముసుగు_పునాది = has_ముసుగు ? get_ముసుగు_అడ్డువరుస_పునాది(ముసుగు, iq1, iq2, iq3) : (స్థిరరాశి అక్షరం *) 0;

        // Flush Q_f16 to L2 so tensor_load can see it
        FENCE;
        flush_గమ్యం_l2(q_f16, (dk * 2 + 63) / 64, 64);
        WAIT_CACHEOPS;

        కొరకు (int64_t kv_పునాది = kv_ప్రారంభం; kv_పునాది < kv_చివర; kv_పునాది += TILE_KV) {
            స్థిరరాశి int64_t kv_లెక్క = (kv_పునాది + TILE_KV <= nk) ? TILE_KV : (nk - kv_పునాది);

            // Set tensor_mask for partial tiles
            ఒకవేళ (kv_లెక్క < TILE_KV) {
                uint64_t tmask = (1ULL << kv_లెక్క) - 1;
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
            స్థిరరాశి int64_t  n_dk_chunks = dk / TILE_K;
            స్థిరరాశి uint64_t K_BUFS[2]   = {
                (uint64_t) B_L1_ప్రారంభం,         // 16..31
                (uint64_t) (B_L1_ప్రారంభం + 16),  // 32..47
            };

            // Preload entire Q row into A_L1[0..n_dk_chunks-1] (one tensor_load,
            // one wait, regardless of dk).
            tensor_ఎక్కించు(false, false, A_L1_ప్రారంభం, TENSOR_ఎక్కించు_PLAIN, 0, (uint64_t) q_f16, 0,
                        (uint64_t) (n_dk_chunks - 1), 64, 0);

            // Prologue: wait hart 1's K[0], issue K[0] load, wait both loads.
            {
                పూర్ణాంకం buf = chunk_id & 1;
                et_sem_wait(ET_BARRIER_MINION);
                tensor_ఎక్కించు(false, false, K_BUFS[0], TENSOR_ఎక్కించు_TRANSPOSE16, 0, (uint64_t) scp_kp[buf], 0, 15, 64, 1);
                tensor_wait(TENSOR_ఎక్కించు_WAIT_0);  // Q row complete
                tensor_wait(TENSOR_ఎక్కించు_WAIT_1);  // K[0] complete
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
            కొరకు (int64_t i = 1; i < n_dk_chunks; i++) {
                పూర్ణాంకం buf         = chunk_id & 1;
                పూర్ణాంకం k_స్థానం_prev = (పూర్ణాంకం) ((i - 1) & 1);
                పూర్ణాంకం k_స్థానం      = (పూర్ణాంకం) (i & 1);

                et_sem_wait(ET_BARRIER_MINION);
                tensor_ఎక్కించు(false, false, K_BUFS[k_స్థానం], TENSOR_ఎక్కించు_TRANSPOSE16, 0, (uint64_t) scp_kp[buf], 0, 15, 64,
                            1);

                tensor_fma((kv_లెక్క < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_స్థానం_prev],
                           (uint64_t) (i - 1), TENSOR_FMA_OP_FP16, (i == 1));

                tensor_wait(TENSOR_ఎక్కించు_WAIT_1);  // K[i] in L1
                et_sem_post(ET_BARRIER_MINION);   // release scp_kp[buf] EARLY
                tensor_wait(TENSOR_FMA_WAIT);     // then wait FMA[i-1]
                chunk_id++;
            }

            // Epilogue: FMA on the last chunk (no overlapping load).
            {
                పూర్ణాంకం k_స్థానం_చివరి = (పూర్ణాంకం) ((n_dk_chunks - 1) & 1);
                tensor_fma((kv_లెక్క < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_స్థానం_చివరి],
                           (uint64_t) (n_dk_chunks - 1), TENSOR_FMA_OP_FP16, (n_dk_chunks == 1));
                tensor_wait(TENSOR_FMA_WAIT);
            }

            // Prefetch V rows for this tile.
            // Only useful for the partial-tile path below
            ఒకవేళ (kv_లెక్క < TILE_KV) {
                కొరకు (int64_t d = 0; d < dv; d += 32) {
                    prefetch_kv_గమ్యం_l2(v_తల, kv_పునాది, d, kv_లెక్క, v->nb[1]);
                }
            }

            // Extract QK^T scores from vector register file
            __asm__ అస్థిర("" ::: "f0", "f1");
            {
                గుర్తులేని దీర్ఘ _ms;
                __asm__ అస్థిర(
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

            దశాంశం weights[TILE_KV] __attribute__((aligned(64)));
            {
                // A1: apply mask to scores, pad unused slots
                కొరకు (int64_t j = 0; j < kv_లెక్క; ++j) {
                    ఒకవేళ (has_ముసుగు) {
                        దశాంశం mv = get_ముసుగు_val_from_పునాది(ముసుగు, ముసుగు_పునాది, kv_పునాది + j);
                        ఒకవేళ (mv == ET_NEG_INF_F || mv != mv) {
                            scores[j] = ET_NEG_INF_F;
                        } లేకపోతే {
                            scores[j] += mv;
                        }
                    }
                }
                కొరకు (int64_t j = kv_లెక్క; j < TILE_KV; ++j) {
                    scores[j] = ET_NEG_INF_F;
                }

                // A1b: SIMD horizontal max across all 16 scores
                దశాంశం tile_max;
                {
                    గుర్తులేని దీర్ఘ _ms;
                    __asm__ అస్థిర(
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

                ఒకవేళ (tile_max > ET_NEG_INF_F) {
                    // A2: rescale accumulator if this tile has a new global max
                    ఒకవేళ (tile_max > M) {
                        దశాంశం rescale = et_exp2f((M - tile_max) * 1.4426950408889634f);
                        scale_acc_vec(acc, dv, rescale);
                        S *= rescale;
                        M = tile_max;
                    }

                    // A3: SIMD exp2 + horizontal sum
                    // Interleaved: f2/f3 chains alternate to hide ALU latency.
                    // fexp.ps has multi-cycle latency — the two independent
                    // exp2 calls naturally pipeline.
                    {
                        స్థిరరాశి దశాంశం   log2e = 1.4426950408889634f;
                        దశాంశం         S_tile;
                        గుర్తులేని దీర్ఘ _ms;
                        __asm__ అస్థిర(
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
                        convert_q_అడ్డువరుస_f32_గమ్యం_f16(w_f16_buf, weights, TILE_KV);

                        FENCE;
                        flush_గమ్యం_l2(w_f16_buf, 1, 64);
                        WAIT_CACHEOPS;

                        // Issue weights load (wait_id=0) and the first V chunk
                        // load (wait_id=1) concurrently. Weights comes from
                        // L2 SCP (just flushed); V[0] comes from DRAM via
                        // INTERLEAVE16 — running them in parallel hides the
                        // shorter load behind the longer one. For partial
                        // tiles, V is software-packed below — we only kick
                        // off the early V load on the full-tile fast path.
                        tensor_ఎక్కించు(false, false, A_L1_ప్రారంభం, TENSOR_ఎక్కించు_PLAIN, 0, (uint64_t) w_f16_buf, 0, 0, 64, 0);

                        స్థిరరాశి పూర్ణాంకం       v_నిండిన_tile = (kv_లెక్క == TILE_KV);
                        స్థిరరాశి uintptr_t v_పునాది      = (uintptr_t) v_తల + kv_పునాది * v->nb[1];
                        స్థిరరాశి uint64_t  nb1_v       = (uint64_t) v->nb[1];
                        uint64_t        b_cur       = 8;

                        ఒకవేళ (v_నిండిన_tile) {
                            tensor_ఎక్కించు(false, false, b_cur, TENSOR_ఎక్కించు_INTERLEAVE16, 0, (uint64_t) v_పునాది, 0, 7,
                                        nb1_v, 1);
                        }

                        tensor_wait(TENSOR_ఎక్కించు_WAIT_0);      // weights in A_L1
                        ఒకవేళ (v_నిండిన_tile) {
                            tensor_wait(TENSOR_ఎక్కించు_WAIT_1);  // V[0] in b_cur
                        }

                        // B2: process dv in chunks of 16
                        ఒకవేళ (v_నిండిన_tile) {
                            కొరకు (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                స్థిరరాశి uint64_t b_nxt = b_cur ^ 24;

                                ఒకవేళ (dv_off + 16 < dv) {
                                    tensor_ఎక్కించు(false, false, b_nxt, TENSOR_ఎక్కించు_INTERLEAVE16, 0,
                                                (uint64_t) (v_పునాది + (dv_off + 16) * 2), 0, 7, nb1_v, 1);
                                }

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, b_cur, A_L1_ప్రారంభం,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ అస్థిర("" ::: "f0", "f1");
                                {
                                    గుర్తులేని దీర్ఘ _ms;
                                    __asm__ అస్థిర(
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

                                ఒకవేళ (dv_off + 16 < dv) {
                                    tensor_wait(TENSOR_ఎక్కించు_WAIT_1);
                                    b_cur = b_nxt;
                                }
                            }
                        } లేకపోతే {
                            // Partial tile: software pack, no pipeline
                            కొరకు (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                pack_v_interleaved(vpanel_buf, v_తల, kv_పునాది, dv_off, kv_లెక్క, v->nb[1]);
                                FENCE;
                                flush_గమ్యం_l2(vpanel_buf, 8, 64);
                                WAIT_CACHEOPS;
                                tensor_ఎక్కించు(false, false, B_L1_ప్రారంభం, TENSOR_ఎక్కించు_PLAIN, 0, (uint64_t) vpanel_buf, 0, 7,
                                            64, 0);
                                tensor_wait(TENSOR_ఎక్కించు_WAIT_0);

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, B_L1_ప్రారంభం, A_L1_ప్రారంభం,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ అస్థిర("" ::: "f0", "f1");
                                {
                                    గుర్తులేని దీర్ఘ _ms;
                                    __asm__ అస్థిర(
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
        ఒకవేళ (k_splits > 1) {
            // Publish our partial.
            అస్థిర దశాంశం * my_stats = (అస్థిర దశాంశం *) et_shire_l2scp_local(scp_పునాది + SCP_STATS_OFF);
            my_stats[0]               = M;
            my_stats[1]               = S;
            FENCE;
            evict_range_గమ్యం_l2(acc, (int64_t) dv * (int64_t) పరిమాణం(దశాంశం));
            evict_గమ్యం_l2((స్థిరరాశి శూన్య *) my_stats, 1, 64);
            WAIT_CACHEOPS;

            // A: team members have all written their partials.
            et_barrier(ET_BARRIER_SHIRE);

            ఒకవేళ (k_విభజించు == 0) {
                // Online softmax merge: fold peers 1..k_splits-1 into our
                // own (M_running, S_running, acc). For each peer p:
                //   M_new    = max(M_running, M_p)
                //   α_own    = exp2((M_running - M_new) * log2e)
                //   α_p      = exp2((M_p     - M_new) * log2e)
                //   acc[d]   = α_own * acc[d] + α_p * peer_acc[d]
                //   S_running = α_own * S_running + α_p * S_p
                దశాంశం       M_running = M;
                దశాంశం       S_running = S;
                స్థిరరాశి దశాంశం log2e     = 1.4426950408889634f;

                కొరకు (int64_t p = 1; p < k_splits; p++) {
                    uint64_t         peer_scp   = (local_tile_idx * k_splits + p) * SCP_చొప్పున_MINION;
                    అస్థిర దశాంశం * peer_stats = (అస్థిర దశాంశం *) et_shire_l2scp_local(peer_scp + SCP_STATS_OFF);
                    దశాంశం *          peer_acc   = (దశాంశం *) et_shire_l2scp_local(peer_scp + SCP_ACC_OFF);

                    // Drop stale L1D copies before reading peer's data.
                    evict_గమ్యం_l2((స్థిరరాశి శూన్య *) peer_stats, 1, 64);
                    evict_range_గమ్యం_l2(peer_acc, (int64_t) dv * (int64_t) పరిమాణం(దశాంశం));
                    WAIT_CACHEOPS;

                    స్థిరరాశి దశాంశం M_p = peer_stats[0];
                    స్థిరరాశి దశాంశం S_p = peer_stats[1];

                    స్థిరరాశి దశాంశం M_new     = (M_p > M_running) ? M_p : M_running;
                    స్థిరరాశి దశాంశం alpha_own = (M_running == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_running - M_new) * log2e);
                    స్థిరరాశి దశాంశం alpha_p   = (M_p == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_p - M_new) * log2e);

                    విలీనం_rescale_కూడు_asm(acc, peer_acc, dv, alpha_own, alpha_p);

                    S_running = alpha_own * S_running + alpha_p * S_p;
                    M_running = M_new;
                }

                స్థిరరాశి దశాంశం S_inv = (S_running == 0.0f) ? 0.0f : et_fdiv(1.0f, S_running);
                normalize_store_vec(బహిర్గతం, acc, dv, S_inv, use_fast_store);
            }

            // B: reducer is done, team may reuse its acc/stats slabs.
            et_barrier(ET_BARRIER_SHIRE);
        } లేకపోతే {
            // k_splits == 1 fast path — this minion owns the full row.
            స్థిరరాశి దశాంశం S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
            normalize_store_vec(బహిర్గతం, acc, dv, S_inv, use_fast_store);
        }
    }

    FENCE;
    తిరిగి 0;
}
