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

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"
#ضمن "tensor.h"

#ضمن <etsoc/common/utils.h>
#ضمن <stdint.h>
#ضمن <string.h>

#تعریف_کرو عدد_COMPUTE_SHIRES 32
#تعریف_کرو MINIONS_فی_SHIRE  32

// QK^T tiles: 16 KV positions at a time, K in chunks of 32 F16
#تعریف_کرو TILE_KV 16
#تعریف_کرو TILE_K  32

// L1 scratchpad layout: A (Q) in lines 0-15, B (K interleaved) in lines 16-31
#تعریف_کرو A_L1_شروع 0
#تعریف_کرو B_L1_شروع 16

// Max head dimensions
#تعریف_کرو FA_DV_MAX 512  // max value head dim (dv)
#تعریف_کرو FA_DK_MAX 512  // max key head dim (dk) - some models use hsk > hsv

تعريفنوع uint16_t et_fp16_t;

#تعریف_کرو ET_NEG_INF_F (-3.402823466e+38f)

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
#تعریف_کرو SCP_ACC_OFF     0
#تعریف_کرو SCP_ACC_STRIDE  (FA_DV_MAX * حجم(عائم))       // 2048
#تعریف_کرو SCP_KPANEL_ناپ (32 * 32 * حجم(et_fp16_t))     // 2048
#تعریف_کرو SCP_KP0_OFF     SCP_ACC_STRIDE                    // 2048
#تعریف_کرو SCP_KP1_OFF     (SCP_KP0_OFF + SCP_KPANEL_ناپ)   // 4096
#تعریف_کرو SCP_STATS_OFF   (SCP_KP1_OFF + SCP_KPANEL_ناپ)   // 6144
#تعریف_کرو SCP_STATS_ناپ  64                                // own cache line
#تعریف_کرو SCP_فی_MINION  (SCP_STATS_OFF + SCP_STATS_ناپ)  // 6208

بنية ggml_et_flash_attn_ext_params {
    بنية ggml_tensor src0;  // Q (F32)
    بنية ggml_tensor src1;  // K (F16)
    بنية ggml_tensor src2;  // V (F16)
    بنية ggml_tensor نقاب;  // mask (F16 or F32), zeroed when absent
    بنية ggml_tensor dst;   // Output (F32)
    عائم              scale;
    int32_t            has_نقاب;
};

ساكن مضمن عائم get_نقاب_val(ثابت بنية ggml_tensor * نقاب, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    ثابت حرف * ادھار = (ثابت حرف *) نقاب->انکڑے + iq1 * نقاب->nb[1] + (iq2 % نقاب->ne[2]) * نقاب->nb[2] +
                        (iq3 % نقاب->ne[3]) * نقاب->nb[3];

    إذا (نقاب->type == GGML_TYPE_F32) {
        أرجع *(ثابت عائم *) (ادھار + ik1 * نقاب->nb[0]);
    }
    أرجع fp16_منزل_fp32(*(ثابت uint16_t *) (ادھار + ik1 * نقاب->nb[0]));
}

ساكن مضمن ثابت حرف * get_نقاب_صف_ادھار(ثابت بنية ggml_tensor * نقاب, int64_t iq1, int64_t iq2, int64_t iq3) {
    أرجع (ثابت حرف *) نقاب->انکڑے + iq1 * نقاب->nb[1] + (iq2 % نقاب->ne[2]) * نقاب->nb[2] +
           (iq3 % نقاب->ne[3]) * نقاب->nb[3];
}

ساكن مضمن عائم get_نقاب_val_from_ادھار(ثابت بنية ggml_tensor * نقاب, ثابت حرف * ادھار, int64_t ik1) {
    إذا (نقاب->type == GGML_TYPE_F32) {
        أرجع *(ثابت عائم *) (ادھار + ik1 * نقاب->nb[0]);
    }
    أرجع fp16_منزل_fp32(*(ثابت uint16_t *) (ادھار + ik1 * نقاب->nb[0]));
}

// Pack K rows for TensorLoadTranspose16 (even/odd deinterleave)
ساكن مضمن فراغ __attribute__((always_inline)) pack_k_for_transpose16(et_fp16_t *  باہرلا,
                                                                         ثابت حرف * k_ادھار,
                                                                         int64_t      kv_شروع,
                                                                         int64_t      dk_شروع,
                                                                         int64_t      kv_گنتی,
                                                                         int64_t      nb1_k) {
    بنانشانی لما old_نقاب;
    __asm__ متطاير(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        : [ms] "=&r"(old_نقاب)
        :
        :);

    لكل (صحيح j = 0; j < (صحيح) kv_گنتی; ++j) {
        ثابت et_fp16_t * k_صف    = (ثابت et_fp16_t *) (k_ادھار + (kv_شروع + j) * nb1_k) + dk_شروع;
        et_fp16_t *       even_صف = باہرلا + (j * 2) * 32;
        et_fp16_t *       odd_صف  = باہرلا + (j * 2 + 1) * 32;
        __asm__ متطاير(
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
            : [src0] "r"(k_صف), [src1] "r"(k_صف + 16), [even] "r"(even_صف), [odd] "r"(odd_صف)
            : "f2", "f3", "f4", "f5", "f6", "f7", "memory");
    }

    __asm__ متطاير("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_نقاب));

    لكل (صحيح j = (صحيح) kv_گنتی; j < TILE_KV; ++j) {
        et_fp16_t * even_صف = باہرلا + (j * 2) * 32;
        et_fp16_t * odd_صف  = باہرلا + (j * 2 + 1) * 32;
        لكل (صحيح l = 0; l < TILE_K / 2; ++l) {
            even_صف[l] = 0;
            odd_صف[l]  = 0;
        }
    }
}

// Build interleaved B panel for TensorFMA16A32 (weights @ V).
ساكن مضمن فراغ __attribute__((always_inline)) pack_v_interleaved(et_fp16_t *  باہرلا,
                                                                     ثابت حرف * v_سر,
                                                                     int64_t      kv_ادھار,
                                                                     int64_t      dv_شروع,
                                                                     int64_t      kv_گنتی,
                                                                     int64_t      nb1_v) {
    لكل (صحيح k = 0; k < TILE_KV; ++k) {
        ثابت صحيح         l   = k >> 1;
        ثابت صحيح         r   = k & 1;
        et_fp16_t * ثابت dst = باہرلا + l * 32 + r;
        إذا (k < (صحيح) kv_گنتی) {
            ثابت et_fp16_t * v_صف = (ثابت et_fp16_t *) (v_سر + (kv_ادھار + k) * nb1_v) + dv_شروع;
            لكل (صحيح n = 0; n < 16; ++n) {
                dst[n * 2] = v_صف[n];
            }
        } وإلا {
            لكل (صحيح n = 0; n < 16; ++n) {
                dst[n * 2] = 0;
            }
        }
    }
}

// Prefetch KV rows for one chunk into L2.
ساكن مضمن فراغ __attribute__((always_inline)) prefetch_kv_منزل_l2(ثابت حرف * سر,
                                                                    int64_t      kv_شروع,
                                                                    int64_t      d_شروع,
                                                                    int64_t      kv_گنتی,
                                                                    int64_t      nb1) {
    ثابت فراغ * ادھار = (ثابت فراغ *) (سر + kv_شروع * nb1 + d_شروع * 2);
    l2_prefetch(ادھار, (uint64_t) kv_گنتی, (uint64_t) nb1);
}

ساكن مضمن فراغ __attribute__((always_inline)) convert_q_صف_f32_منزل_f16(et_fp16_t *   dst,
                                                                           ثابت عائم * src,
                                                                           int64_t       n) {
    ساكن ثابت int32_t __attribute__((aligned(32))) offsets[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };

    بنانشانی لما old_نقاب;
    __asm__ متطاير(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 0xFF      \n\t"
        "flw.ps    f1, 0(%[offs])    \n\t"
        : [ms] "=&r"(old_نقاب)
        : [offs] "r"(offsets)
        : "f1");

    لكل (int64_t d = 0; d < n; d += 8) {
        __asm__ متطاير(
            "flw.ps      f2, 0(%[src])    \n\t"
            "fcvt.f16.ps f3, f2           \n\t"
            "fsch.ps     f3, f1(%[dst])   \n\t"
            :
            : [src] "r"(src + d), [dst] "r"(dst + d)
            : "f2", "f3", "memory");
    }

    __asm__ متطاير("mova.m.x  %[ms]             \n\t" : : [ms] "r"(old_نقاب));
}

ساكن مضمن فراغ __attribute__((always_inline)) zero_acc_vec(عائم * acc, int64_t dv) {
    ثابت عائم   zero = 0.0f;
    بنانشانی لما old_نقاب;
    __asm__ متطاير("mova.x.m %0" : "=r"(old_نقاب));
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");
    __asm__ متطاير("fbc.ps  f2, 0(%[z])" ::[z] "r"(&zero) : "f2");

    لكل (int64_t d = 0; d < dv; d += 8) {
        __asm__ متطاير("fsw.ps  f2, 0(%[a])     \n\t" ::[a] "r"(acc + d) : "f2", "memory");
    }

    __asm__ متطاير("mova.m.x %0" ::"r"(old_نقاب));
}

ساكن مضمن فراغ __attribute__((always_inline)) scale_acc_vec(عائم * acc, int64_t dv, عائم scale) {
    بنانشانی لما old_نقاب;
    __asm__ متطاير("mova.x.m %0" : "=r"(old_نقاب));
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");

    لكل (int64_t d = 0; d < dv; d += 8) {
        __asm__ متطاير(
            "fbc.ps    f2, 0(%[s])    \n\t"
            "flw.ps    f3, 0(%[a])    \n\t"
            "fmul.ps   f3, f3, f2     \n\t"
            "fsw.ps    f3, 0(%[a])    \n\t"
            :
            : [s] "r"(&scale), [a] "r"(acc + d)
            : "f2", "f3", "memory");
    }

    __asm__ متطاير("mova.m.x %0" ::"r"(old_نقاب));
}

ساكن مضمن فراغ __attribute__((always_inline)) normalize_store_vec(عائم * باہرلا,
                                                                      عائم * acc,
                                                                      int64_t dv,
                                                                      عائم   inv,
                                                                      صحيح     use_fast_store) {
    بنانشانی لما old_نقاب;
    __asm__ متطاير("mova.x.m %0" : "=r"(old_نقاب));
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");

    لكل (int64_t d = 0; d < dv; d += 8) {
        __asm__ متطاير(
            "fbc.ps    f2, 0(%[inv])   \n\t"
            "flw.ps    f3, 0(%[a])     \n\t"
            "fmul.ps   f3, f3, f2      \n\t"
            "fsw.ps    f3, 0(%[a])     \n\t"
            :
            : [inv] "r"(&inv), [a] "r"(acc + d)
            : "f2", "f3", "memory");
        إذا (use_fast_store) {
            __asm__ متطاير(
                "flw.ps  f4, 0(%[a])     \n\t"
                "fsw.ps  f4, 0(%[o])     \n\t"
                :
                : [a] "r"(acc + d), [o] "r"(باہرلا + d)
                : "f4", "memory");
        } وإلا {
            atomic_store_f32((متطاير عائم *) &باہرلا[d + 0], acc[d + 0]);
            atomic_store_f32((متطاير عائم *) &باہرلا[d + 1], acc[d + 1]);
            atomic_store_f32((متطاير عائم *) &باہرلا[d + 2], acc[d + 2]);
            atomic_store_f32((متطاير عائم *) &باہرلا[d + 3], acc[d + 3]);
            atomic_store_f32((متطاير عائم *) &باہرلا[d + 4], acc[d + 4]);
            atomic_store_f32((متطاير عائم *) &باہرلا[d + 5], acc[d + 5]);
            atomic_store_f32((متطاير عائم *) &باہرلا[d + 6], acc[d + 6]);
            atomic_store_f32((متطاير عائم *) &باہرلا[d + 7], acc[d + 7]);
        }
    }

    __asm__ متطاير("mova.m.x %0" ::"r"(old_نقاب));
}

ساكن مضمن size_t tensor_بائٹاں_fa(ثابت بنية ggml_tensor * t) {
    أرجع (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

// Evict a byte range from L1D to L2 SCP, splitting into batches of ≤16
// cache lines (the hw limit for evict_to_l2). Use before a barrier when
// another minion in the shire needs to read the region, or after a barrier
// on the reader side to drop stale L1D copies before reading peer data.
ساكن مضمن فراغ __attribute__((always_inline)) evict_range_منزل_l2(ثابت فراغ * addr, int64_t بائٹاں) {
    إذا (بائٹاں <= 0) {
        أرجع;
    }
    int64_t      سطریں = (بائٹاں + 63) / 64;
    ثابت حرف * p     = (ثابت حرف *) addr;
    طالما (سطریں > 0) {
        int64_t batch = سطریں > 16 ? 16 : سطریں;
        evict_منزل_l2((ثابت فراغ *) p, (uint64_t) batch, 64);
        p += batch * 64;
        سطریں -= batch;
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
ساكن مضمن فراغ __attribute__((always_inline)) ملاؤ_rescale_جوڑو_asm(عائم *       acc,
                                                                        ثابت عائم * peer_acc,
                                                                        int64_t       dv,
                                                                        عائم         alpha_own,
                                                                        عائم         alpha_peer) {
    بنانشانی لما old_نقاب;
    __asm__ متطاير(
        "mova.x.m  %[ms]              \n\t"
        "mov.m.x   m0, x0, 0xFF       \n\t"
        "fbc.ps    f4, 0(%[ao])       \n\t"  // broadcast alpha_own
        "fbc.ps    f5, 0(%[ap])       \n\t"  // broadcast alpha_peer
        : [ms] "=&r"(old_نقاب)
        : [ao] "r"(&alpha_own), [ap] "r"(&alpha_peer)
        : "f4", "f5");

    لكل (int64_t d = 0; d < dv; d += 8) {
        __asm__ متطاير(
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

    __asm__ متطاير("mova.m.x %0" ::"r"(old_نقاب));
}

صحيح اندراج_نقطہ(بنية ggml_et_flash_attn_ext_params * params, فراغ * env) {
    (فراغ) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    إذا (shire_id >= عدد_COMPUTE_SHIRES) {
        أرجع 0;
    }

    ثابت صحيح is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    بنية ggml_tensor * q        = &params->src0;
    بنية ggml_tensor * k        = &params->src1;
    بنية ggml_tensor * v        = &params->src2;
    بنية ggml_tensor * dst      = &params->dst;
    ثابت int32_t        has_نقاب = params->has_نقاب;
    بنية ggml_tensor * نقاب     = has_نقاب ? &params->نقاب : (بنية ggml_tensor *) 0;

    ثابت حرف * q_انکڑے   = (ثابت حرف *) q->انکڑے;
    ثابت حرف * k_انکڑے   = (ثابت حرف *) k->انکڑے;
    ثابت حرف * v_انکڑے   = (ثابت حرف *) v->انکڑے;
    حرف *       dst_انکڑے = (حرف *) dst->انکڑے;

    // et_barrier(ET_BARRIER_GLOBAL);
    evict_region_past_l2(q->انکڑے, tensor_بائٹاں_fa(q));
    evict_region_past_l2(k->انکڑے, tensor_بائٹاں_fa(k));
    evict_region_past_l2(v->انکڑے, tensor_بائٹاں_fa(v));
    إذا (نقاب) {
        evict_region_past_l2(نقاب->انکڑے, tensor_بائٹاں_fa(نقاب));
    }
    et_barrier(ET_BARRIER_GLOBAL);

    ثابت int64_t dk  = q->ne[0];
    ثابت int64_t nq  = q->ne[1];
    ثابت int64_t nhq = q->ne[2];
    ثابت int64_t no  = q->ne[3];
    ثابت int64_t nk  = k->ne[1];
    ثابت int64_t nhk = k->ne[2];
    ثابت int64_t dv  = v->ne[0];

    إذا (dv > FA_DV_MAX || dk > FA_DK_MAX) {
        أرجع -1;
    }
    إذا (k->nb[0] != 2 || v->nb[0] != 2) {
        أرجع -1;
    }
    إذا ((dk % 8) != 0 || (dv % 16) != 0) {
        أرجع -1;
    }

    ثابت int64_t gqa_تناسب      = nhq / nhk;
    ثابت int64_t کل_صفاں     = nq * nhq * no;
    ثابت عائم   scale          = params->scale;
    ثابت صحيح     use_fast_store = (dv % 16 == 0);

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
    ثابت int64_t nk_tiles      = (nk + TILE_KV - 1) / TILE_KV;
    ثابت int64_t کل_minions = 2 * عدد_COMPUTE_SHIRES * MINIONS_فی_SHIRE;
    int64_t       k_splits      = 1;
    إذا (کل_صفاں < کل_minions) {
        int64_t نشانہ = کل_minions / کل_صفاں;
        int64_t ks     = 1;
        طالما (ks * 2 <= نشانہ && ks * 2 <= MINIONS_فی_SHIRE && ks * 2 <= nk_tiles) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ثابت int64_t tiles_فی_shire = MINIONS_فی_SHIRE / k_splits;
    ثابت int64_t k_ونڈو         = (int64_t) local_minion % k_splits;
    ثابت int64_t local_tile_idx  = (int64_t) local_minion / k_splits;
    ثابت int64_t tiles_stride    = (int64_t) عدد_COMPUTE_SHIRES * tiles_فی_shire;

    // KV slab for this k_split. With k_splits=1 this is the full range.
    ثابت int64_t tiles_فی_ونڈو_rounded = (nk_tiles + k_splits - 1) / k_splits;
    ثابت int64_t tile_شروع              = k_ونڈو * tiles_فی_ونڈو_rounded;
    int64_t       tile_سرا                = tile_شروع + tiles_فی_ونڈو_rounded;
    إذا (tile_سرا > nk_tiles) {
        tile_سرا = nk_tiles;
    }
    ثابت int64_t kv_شروع = tile_شروع * TILE_KV;
    int64_t       kv_سرا   = tile_سرا * TILE_KV;
    إذا (kv_سرا > nk) {
        kv_سرا = nk;
    }

    // L2 SCP pointers for this minion
    uint64_t    scp_ادھار  = local_minion * SCP_فی_MINION;
    et_fp16_t * scp_kp[2] = {
        (et_fp16_t *) et_shire_l2scp_local(scp_ادھار + SCP_KP0_OFF),
        (et_fp16_t *) et_shire_l2scp_local(scp_ادھار + SCP_KP1_OFF),
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
    إذا (is_hart1) {
        uint32_t      chunk_id = 0;
        ثابت int64_t صف_ادھار = (int64_t) shire_id + local_tile_idx * عدد_COMPUTE_SHIRES;

        int64_t max_iters;
        إذا (k_splits > 1) {
            max_iters = (کل_صفاں + tiles_stride - 1) / tiles_stride;
        } وإلا {
            max_iters = (صف_ادھار >= کل_صفاں) ? 0 : ((کل_صفاں - صف_ادھار - 1) / tiles_stride + 1);
        }

        لكل (int64_t iter = 0; iter < max_iters; iter++) {
            ثابت int64_t صف      = صف_ادھار + iter * tiles_stride;
            ثابت صحيح     has_work = (صف < کل_صفاں);

            إذا (has_work) {
                ثابت int64_t iq3 = صف / (nhq * nq);
                ثابت int64_t rem = صف % (nhq * nq);
                ثابت int64_t iq2 = rem / nq;
                ثابت int64_t ik2 = iq2 / gqa_تناسب;

                ثابت حرف * k_سر = k_انکڑے + ik2 * k->nb[2] + iq3 * k->nb[3];

                لكل (int64_t kv_ادھار = kv_شروع; kv_ادھار < kv_سرا; kv_ادھار += TILE_KV) {
                    ثابت int64_t kv_گنتی = (kv_ادھار + TILE_KV <= nk) ? TILE_KV : (nk - kv_ادھار);

                    لكل (int64_t dk_chunk = 0; dk_chunk < dk; dk_chunk += TILE_K) {
                        صحيح buf = chunk_id & 1;

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
                        إذا (chunk_id >= 2) {
                            et_sem_wait(ET_BARRIER_MINION);
                        }

                        // Prefetch K data for this chunk
                        prefetch_kv_منزل_l2(k_سر, kv_ادھار, dk_chunk, kv_گنتی, k->nb[1]);

                        pack_k_for_transpose16(scp_kp[buf], k_سر, kv_ادھار, dk_chunk, kv_گنتی, k->nb[1]);

                        FENCE;
                        flush_منزل_l2(scp_kp[buf], 16, 64);
                        flush_منزل_l2((et_fp16_t *) ((حرف *) scp_kp[buf] + 1024), 16, 64);
                        WAIT_CACHEOPS;

                        // Signal: this buf is ready for hart 0 to consume.
                        et_sem_post(ET_BARRIER_MINION);

                        chunk_id++;
                    }
                }
            }

            // Shire barriers for split-KV merge (hart 1 is a passive arrival).
            إذا (k_splits > 1) {
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
        لكل (uint32_t d = 0; d < drain; d++) {
            et_sem_wait(ET_BARRIER_MINION);
        }

        // FENCE;
        // et_barrier(ET_BARRIER_GLOBAL);
        أرجع 0;
    }

    // Hart 0: tensor engine compute
#جے_بنا_تعریف UBERKERNEL_SUPPRESS_SCP_تیاری
    تیاری_cache_scp();
#جے_انت
    CLEAR_TENSOR_ERROR;

    // Q converted to F16 (one row at a time)
    et_fp16_t q_f16[FA_DK_MAX] __attribute__((aligned(64)));

    // Score buffer for QK^T output (16 scores per KV tile)
    عائم scores[TILE_KV] __attribute__((aligned(64)));

    // Small buffers for V accumulation
    et_fp16_t w_f16_buf[32] __attribute__((aligned(64)));       // 64 bytes
    et_fp16_t vpanel_buf[8 * 32] __attribute__((aligned(64)));  // 512 bytes

    عائم * acc = (عائم *) et_shire_l2scp_local(scp_ادھار + SCP_ACC_OFF);

    uint32_t chunk_id = 0;

    // Iter-based outer loop (matches hart 1). When k_splits > 1 all teams
    // in a shire iterate the same number of times so the per-row shire
    // barriers stay balanced; iterations with row >= total_rows skip the
    // compute but still participate in the barriers.
    ثابت int64_t hart0_صف_ادھار = (int64_t) shire_id + local_tile_idx * عدد_COMPUTE_SHIRES;
    int64_t       hart0_max_iters;
    إذا (k_splits > 1) {
        hart0_max_iters = (کل_صفاں + tiles_stride - 1) / tiles_stride;
    } وإلا {
        hart0_max_iters = (hart0_صف_ادھار >= کل_صفاں) ? 0 : ((کل_صفاں - hart0_صف_ادھار - 1) / tiles_stride + 1);
    }

    لكل (int64_t iter = 0; iter < hart0_max_iters; iter++) {
        ثابت int64_t صف = hart0_صف_ادھار + iter * tiles_stride;
        إذا (صف >= کل_صفاں) {
            // No-work iteration: only participate in barriers (k_splits > 1).
            إذا (k_splits > 1) {
                et_barrier(ET_BARRIER_SHIRE);  // A
                et_barrier(ET_BARRIER_SHIRE);  // B
            }
            استمر;
        }

        ثابت int64_t iq3 = صف / (nhq * nq);
        ثابت int64_t rem = صف % (nhq * nq);
        ثابت int64_t iq2 = rem / nq;
        ثابت int64_t iq1 = rem % nq;
        ثابت int64_t ik2 = iq2 / gqa_تناسب;

        // Read Q row (F32) and convert to F16
        ثابت عائم * pq = (ثابت عائم *) (q_انکڑے + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);
        convert_q_صف_f32_منزل_f16(q_f16, pq, dk);

        // V base for this head + batch (K packing handled by hart 1)
        ثابت حرف * v_سر = v_انکڑے + ik2 * v->nb[2] + iq3 * v->nb[3];

        // Output pointer
        عائم * باہرلا = (عائم *) (dst_انکڑے + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        zero_acc_vec(acc, dv);
        عائم        M         = ET_NEG_INF_F;
        عائم        S         = 0.0f;
        ثابت حرف * نقاب_ادھار = has_نقاب ? get_نقاب_صف_ادھار(نقاب, iq1, iq2, iq3) : (ثابت حرف *) 0;

        // Flush Q_f16 to L2 so tensor_load can see it
        FENCE;
        flush_منزل_l2(q_f16, (dk * 2 + 63) / 64, 64);
        WAIT_CACHEOPS;

        لكل (int64_t kv_ادھار = kv_شروع; kv_ادھار < kv_سرا; kv_ادھار += TILE_KV) {
            ثابت int64_t kv_گنتی = (kv_ادھار + TILE_KV <= nk) ? TILE_KV : (nk - kv_ادھار);

            // Set tensor_mask for partial tiles
            إذا (kv_گنتی < TILE_KV) {
                uint64_t tmask = (1ULL << kv_گنتی) - 1;
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
            ثابت int64_t  n_dk_chunks = dk / TILE_K;
            ثابت uint64_t K_BUFS[2]   = {
                (uint64_t) B_L1_شروع,         // 16..31
                (uint64_t) (B_L1_شروع + 16),  // 32..47
            };

            // Preload entire Q row into A_L1[0..n_dk_chunks-1] (one tensor_load,
            // one wait, regardless of dk).
            tensor_لدو(false, false, A_L1_شروع, TENSOR_لدو_PLAIN, 0, (uint64_t) q_f16, 0,
                        (uint64_t) (n_dk_chunks - 1), 64, 0);

            // Prologue: wait hart 1's K[0], issue K[0] load, wait both loads.
            {
                صحيح buf = chunk_id & 1;
                et_sem_wait(ET_BARRIER_MINION);
                tensor_لدو(false, false, K_BUFS[0], TENSOR_لدو_TRANSPOSE16, 0, (uint64_t) scp_kp[buf], 0, 15, 64, 1);
                tensor_wait(TENSOR_لدو_WAIT_0);  // Q row complete
                tensor_wait(TENSOR_لدو_WAIT_1);  // K[0] complete
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
            لكل (int64_t i = 1; i < n_dk_chunks; i++) {
                صحيح buf         = chunk_id & 1;
                صحيح k_تھاں_prev = (صحيح) ((i - 1) & 1);
                صحيح k_تھاں      = (صحيح) (i & 1);

                et_sem_wait(ET_BARRIER_MINION);
                tensor_لدو(false, false, K_BUFS[k_تھاں], TENSOR_لدو_TRANSPOSE16, 0, (uint64_t) scp_kp[buf], 0, 15, 64,
                            1);

                tensor_fma((kv_گنتی < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_تھاں_prev],
                           (uint64_t) (i - 1), TENSOR_FMA_OP_FP16, (i == 1));

                tensor_wait(TENSOR_لدو_WAIT_1);  // K[i] in L1
                et_sem_post(ET_BARRIER_MINION);   // release scp_kp[buf] EARLY
                tensor_wait(TENSOR_FMA_WAIT);     // then wait FMA[i-1]
                chunk_id++;
            }

            // Epilogue: FMA on the last chunk (no overlapping load).
            {
                صحيح k_تھاں_آخری = (صحيح) ((n_dk_chunks - 1) & 1);
                tensor_fma((kv_گنتی < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_تھاں_آخری],
                           (uint64_t) (n_dk_chunks - 1), TENSOR_FMA_OP_FP16, (n_dk_chunks == 1));
                tensor_wait(TENSOR_FMA_WAIT);
            }

            // Prefetch V rows for this tile.
            // Only useful for the partial-tile path below
            إذا (kv_گنتی < TILE_KV) {
                لكل (int64_t d = 0; d < dv; d += 32) {
                    prefetch_kv_منزل_l2(v_سر, kv_ادھار, d, kv_گنتی, v->nb[1]);
                }
            }

            // Extract QK^T scores from vector register file
            __asm__ متطاير("" ::: "f0", "f1");
            {
                بنانشانی لما _ms;
                __asm__ متطاير(
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

            عائم weights[TILE_KV] __attribute__((aligned(64)));
            {
                // A1: apply mask to scores, pad unused slots
                لكل (int64_t j = 0; j < kv_گنتی; ++j) {
                    إذا (has_نقاب) {
                        عائم mv = get_نقاب_val_from_ادھار(نقاب, نقاب_ادھار, kv_ادھار + j);
                        إذا (mv == ET_NEG_INF_F || mv != mv) {
                            scores[j] = ET_NEG_INF_F;
                        } وإلا {
                            scores[j] += mv;
                        }
                    }
                }
                لكل (int64_t j = kv_گنتی; j < TILE_KV; ++j) {
                    scores[j] = ET_NEG_INF_F;
                }

                // A1b: SIMD horizontal max across all 16 scores
                عائم tile_max;
                {
                    بنانشانی لما _ms;
                    __asm__ متطاير(
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

                إذا (tile_max > ET_NEG_INF_F) {
                    // A2: rescale accumulator if this tile has a new global max
                    إذا (tile_max > M) {
                        عائم rescale = et_exp2f((M - tile_max) * 1.4426950408889634f);
                        scale_acc_vec(acc, dv, rescale);
                        S *= rescale;
                        M = tile_max;
                    }

                    // A3: SIMD exp2 + horizontal sum
                    // Interleaved: f2/f3 chains alternate to hide ALU latency.
                    // fexp.ps has multi-cycle latency — the two independent
                    // exp2 calls naturally pipeline.
                    {
                        ثابت عائم   log2e = 1.4426950408889634f;
                        عائم         S_tile;
                        بنانشانی لما _ms;
                        __asm__ متطاير(
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
                        convert_q_صف_f32_منزل_f16(w_f16_buf, weights, TILE_KV);

                        FENCE;
                        flush_منزل_l2(w_f16_buf, 1, 64);
                        WAIT_CACHEOPS;

                        // Issue weights load (wait_id=0) and the first V chunk
                        // load (wait_id=1) concurrently. Weights comes from
                        // L2 SCP (just flushed); V[0] comes from DRAM via
                        // INTERLEAVE16 — running them in parallel hides the
                        // shorter load behind the longer one. For partial
                        // tiles, V is software-packed below — we only kick
                        // off the early V load on the full-tile fast path.
                        tensor_لدو(false, false, A_L1_شروع, TENSOR_لدو_PLAIN, 0, (uint64_t) w_f16_buf, 0, 0, 64, 0);

                        ثابت صحيح       v_بھریا_tile = (kv_گنتی == TILE_KV);
                        ثابت uintptr_t v_ادھار      = (uintptr_t) v_سر + kv_ادھار * v->nb[1];
                        ثابت uint64_t  nb1_v       = (uint64_t) v->nb[1];
                        uint64_t        b_cur       = 8;

                        إذا (v_بھریا_tile) {
                            tensor_لدو(false, false, b_cur, TENSOR_لدو_INTERLEAVE16, 0, (uint64_t) v_ادھار, 0, 7,
                                        nb1_v, 1);
                        }

                        tensor_wait(TENSOR_لدو_WAIT_0);      // weights in A_L1
                        إذا (v_بھریا_tile) {
                            tensor_wait(TENSOR_لدو_WAIT_1);  // V[0] in b_cur
                        }

                        // B2: process dv in chunks of 16
                        إذا (v_بھریا_tile) {
                            لكل (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                ثابت uint64_t b_nxt = b_cur ^ 24;

                                إذا (dv_off + 16 < dv) {
                                    tensor_لدو(false, false, b_nxt, TENSOR_لدو_INTERLEAVE16, 0,
                                                (uint64_t) (v_ادھار + (dv_off + 16) * 2), 0, 7, nb1_v, 1);
                                }

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, b_cur, A_L1_شروع,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ متطاير("" ::: "f0", "f1");
                                {
                                    بنانشانی لما _ms;
                                    __asm__ متطاير(
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

                                إذا (dv_off + 16 < dv) {
                                    tensor_wait(TENSOR_لدو_WAIT_1);
                                    b_cur = b_nxt;
                                }
                            }
                        } وإلا {
                            // Partial tile: software pack, no pipeline
                            لكل (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                pack_v_interleaved(vpanel_buf, v_سر, kv_ادھار, dv_off, kv_گنتی, v->nb[1]);
                                FENCE;
                                flush_منزل_l2(vpanel_buf, 8, 64);
                                WAIT_CACHEOPS;
                                tensor_لدو(false, false, B_L1_شروع, TENSOR_لدو_PLAIN, 0, (uint64_t) vpanel_buf, 0, 7,
                                            64, 0);
                                tensor_wait(TENSOR_لدو_WAIT_0);

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, B_L1_شروع, A_L1_شروع,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ متطاير("" ::: "f0", "f1");
                                {
                                    بنانشانی لما _ms;
                                    __asm__ متطاير(
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
        إذا (k_splits > 1) {
            // Publish our partial.
            متطاير عائم * my_stats = (متطاير عائم *) et_shire_l2scp_local(scp_ادھار + SCP_STATS_OFF);
            my_stats[0]               = M;
            my_stats[1]               = S;
            FENCE;
            evict_range_منزل_l2(acc, (int64_t) dv * (int64_t) حجم(عائم));
            evict_منزل_l2((ثابت فراغ *) my_stats, 1, 64);
            WAIT_CACHEOPS;

            // A: team members have all written their partials.
            et_barrier(ET_BARRIER_SHIRE);

            إذا (k_ونڈو == 0) {
                // Online softmax merge: fold peers 1..k_splits-1 into our
                // own (M_running, S_running, acc). For each peer p:
                //   M_new    = max(M_running, M_p)
                //   α_own    = exp2((M_running - M_new) * log2e)
                //   α_p      = exp2((M_p     - M_new) * log2e)
                //   acc[d]   = α_own * acc[d] + α_p * peer_acc[d]
                //   S_running = α_own * S_running + α_p * S_p
                عائم       M_running = M;
                عائم       S_running = S;
                ثابت عائم log2e     = 1.4426950408889634f;

                لكل (int64_t p = 1; p < k_splits; p++) {
                    uint64_t         peer_scp   = (local_tile_idx * k_splits + p) * SCP_فی_MINION;
                    متطاير عائم * peer_stats = (متطاير عائم *) et_shire_l2scp_local(peer_scp + SCP_STATS_OFF);
                    عائم *          peer_acc   = (عائم *) et_shire_l2scp_local(peer_scp + SCP_ACC_OFF);

                    // Drop stale L1D copies before reading peer's data.
                    evict_منزل_l2((ثابت فراغ *) peer_stats, 1, 64);
                    evict_range_منزل_l2(peer_acc, (int64_t) dv * (int64_t) حجم(عائم));
                    WAIT_CACHEOPS;

                    ثابت عائم M_p = peer_stats[0];
                    ثابت عائم S_p = peer_stats[1];

                    ثابت عائم M_new     = (M_p > M_running) ? M_p : M_running;
                    ثابت عائم alpha_own = (M_running == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_running - M_new) * log2e);
                    ثابت عائم alpha_p   = (M_p == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_p - M_new) * log2e);

                    ملاؤ_rescale_جوڑو_asm(acc, peer_acc, dv, alpha_own, alpha_p);

                    S_running = alpha_own * S_running + alpha_p * S_p;
                    M_running = M_new;
                }

                ثابت عائم S_inv = (S_running == 0.0f) ? 0.0f : et_fdiv(1.0f, S_running);
                normalize_store_vec(باہرلا, acc, dv, S_inv, use_fast_store);
            }

            // B: reducer is done, team may reuse its acc/stats slabs.
            et_barrier(ET_BARRIER_SHIRE);
        } وإلا {
            // k_splits == 1 fast path — this minion owns the full row.
            ثابت عائم S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
            normalize_store_vec(باہرلا, acc, dv, S_inv, use_fast_store);
        }
    }

    FENCE;
    أرجع 0;
}
