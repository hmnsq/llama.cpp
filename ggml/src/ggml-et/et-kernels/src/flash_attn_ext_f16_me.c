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

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"
#অন্তর্ভাব "tensor.h"

#অন্তর্ভাব <etsoc/common/utils.h>
#অন্তর্ভাব <stdint.h>
#অন্তর্ভাব <string.h>

#define অঙ্ক_COMPUTE_SHIRES 32
#define MINIONS_PER_SHIRE  32

// QK^T tiles: 16 KV positions at a time, K in chunks of 32 F16
#define TILE_KV 16
#define TILE_K  32

// L1 scratchpad layout: A (Q) in lines 0-15, B (K interleaved) in lines 16-31
#define A_L1_START 0
#define B_L1_START 16

// Max head dimensions
#define FA_DV_MAX 512  // max value head dim (dv)
#define FA_DK_MAX 512  // max key head dim (dk) - some models use hsk > hsv

প্রকারলেখ uint16_t et_fp16_t;

#define ET_NEG_INF_F (-3.402823466e+38f)

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
#define SCP_ACC_OFF     0
#define SCP_ACC_STRIDE  (FA_DV_MAX * আকার(দশমিক))       // 2048
#define SCP_KPANEL_আকার (32 * 32 * আকার(et_fp16_t))     // 2048
#define SCP_KP0_OFF     SCP_ACC_STRIDE                    // 2048
#define SCP_KP1_OFF     (SCP_KP0_OFF + SCP_KPANEL_আকার)   // 4096
#define SCP_STATS_OFF   (SCP_KP1_OFF + SCP_KPANEL_আকার)   // 6144
#define SCP_STATS_আকার  64                                // own cache line
#define SCP_PER_MINION  (SCP_STATS_OFF + SCP_STATS_আকার)  // 6208

কাঠামো ggml_et_flash_attn_ext_params {
    কাঠামো ggml_tensor src0;  // Q (F32)
    কাঠামো ggml_tensor src1;  // K (F16)
    কাঠামো ggml_tensor src2;  // V (F16)
    কাঠামো ggml_tensor mask;  // mask (F16 or F32), zeroed when absent
    কাঠামো ggml_tensor dst;   // Output (F32)
    দশমিক              scale;
    int32_t            has_mask;
};

স্থির অন্তঃরেখ দশমিক get_mask_val(ধ্রুবক কাঠামো ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    ধ্রুবক অক্ষর * base = (ধ্রুবক অক্ষর *) mask->তথ্য + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    যদি (mask->type == GGML_TYPE_F32) {
        ফেরত *(ধ্রুবক দশমিক *) (base + ik1 * mask->nb[0]);
    }
    ফেরত fp16_to_fp32(*(ধ্রুবক uint16_t *) (base + ik1 * mask->nb[0]));
}

স্থির অন্তঃরেখ ধ্রুবক অক্ষর * get_mask_row_base(ধ্রুবক কাঠামো ggml_tensor * mask, int64_t iq1, int64_t iq2, int64_t iq3) {
    ফেরত (ধ্রুবক অক্ষর *) mask->তথ্য + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
           (iq3 % mask->ne[3]) * mask->nb[3];
}

স্থির অন্তঃরেখ দশমিক get_mask_val_from_base(ধ্রুবক কাঠামো ggml_tensor * mask, ধ্রুবক অক্ষর * base, int64_t ik1) {
    যদি (mask->type == GGML_TYPE_F32) {
        ফেরত *(ধ্রুবক দশমিক *) (base + ik1 * mask->nb[0]);
    }
    ফেরত fp16_to_fp32(*(ধ্রুবক uint16_t *) (base + ik1 * mask->nb[0]));
}

// Pack K rows for TensorLoadTranspose16 (even/odd deinterleave)
স্থির অন্তঃরেখ শূন্য __attribute__((always_inline)) pack_k_for_transpose16(et_fp16_t *  out,
                                                                         ধ্রুবক অক্ষর * k_base,
                                                                         int64_t      kv_start,
                                                                         int64_t      dk_start,
                                                                         int64_t      kv_গুনতি,
                                                                         int64_t      nb1_k) {
    unsigned long old_mask;
    __asm__ অস্থির(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        : [ms] "=&r"(old_mask)
        :
        :);

    জন্য (পূর্ণসংখ্যা j = 0; j < (পূর্ণসংখ্যা) kv_গুনতি; ++j) {
        ধ্রুবক et_fp16_t * k_row    = (ধ্রুবক et_fp16_t *) (k_base + (kv_start + j) * nb1_k) + dk_start;
        et_fp16_t *       even_row = out + (j * 2) * 32;
        et_fp16_t *       odd_row  = out + (j * 2 + 1) * 32;
        __asm__ অস্থির(
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
            : [src0] "r"(k_row), [src1] "r"(k_row + 16), [even] "r"(even_row), [odd] "r"(odd_row)
            : "f2", "f3", "f4", "f5", "f6", "f7", "memory");
    }

    __asm__ অস্থির("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_mask));

    জন্য (পূর্ণসংখ্যা j = (পূর্ণসংখ্যা) kv_গুনতি; j < TILE_KV; ++j) {
        et_fp16_t * even_row = out + (j * 2) * 32;
        et_fp16_t * odd_row  = out + (j * 2 + 1) * 32;
        জন্য (পূর্ণসংখ্যা l = 0; l < TILE_K / 2; ++l) {
            even_row[l] = 0;
            odd_row[l]  = 0;
        }
    }
}

// Build interleaved B panel for TensorFMA16A32 (weights @ V).
স্থির অন্তঃরেখ শূন্য __attribute__((always_inline)) pack_v_interleaved(et_fp16_t *  out,
                                                                     ধ্রুবক অক্ষর * v_শীর্ষ,
                                                                     int64_t      kv_base,
                                                                     int64_t      dv_start,
                                                                     int64_t      kv_গুনতি,
                                                                     int64_t      nb1_v) {
    জন্য (পূর্ণসংখ্যা k = 0; k < TILE_KV; ++k) {
        ধ্রুবক পূর্ণসংখ্যা         l   = k >> 1;
        ধ্রুবক পূর্ণসংখ্যা         r   = k & 1;
        et_fp16_t * ধ্রুবক dst = out + l * 32 + r;
        যদি (k < (পূর্ণসংখ্যা) kv_গুনতি) {
            ধ্রুবক et_fp16_t * v_row = (ধ্রুবক et_fp16_t *) (v_শীর্ষ + (kv_base + k) * nb1_v) + dv_start;
            জন্য (পূর্ণসংখ্যা n = 0; n < 16; ++n) {
                dst[n * 2] = v_row[n];
            }
        } অন্যথা {
            জন্য (পূর্ণসংখ্যা n = 0; n < 16; ++n) {
                dst[n * 2] = 0;
            }
        }
    }
}

// Prefetch KV rows for one chunk into L2.
স্থির অন্তঃরেখ শূন্য __attribute__((always_inline)) prefetch_kv_to_l2(ধ্রুবক অক্ষর * শীর্ষ,
                                                                    int64_t      kv_start,
                                                                    int64_t      d_start,
                                                                    int64_t      kv_গুনতি,
                                                                    int64_t      nb1) {
    ধ্রুবক শূন্য * base = (ধ্রুবক শূন্য *) (শীর্ষ + kv_start * nb1 + d_start * 2);
    l2_prefetch(base, (uint64_t) kv_গুনতি, (uint64_t) nb1);
}

স্থির অন্তঃরেখ শূন্য __attribute__((always_inline)) convert_q_row_f32_to_f16(et_fp16_t *   dst,
                                                                           ধ্রুবক দশমিক * src,
                                                                           int64_t       n) {
    স্থির ধ্রুবক int32_t __attribute__((aligned(32))) offsets[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };

    unsigned long old_mask;
    __asm__ অস্থির(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 0xFF      \n\t"
        "flw.ps    f1, 0(%[offs])    \n\t"
        : [ms] "=&r"(old_mask)
        : [offs] "r"(offsets)
        : "f1");

    জন্য (int64_t d = 0; d < n; d += 8) {
        __asm__ অস্থির(
            "flw.ps      f2, 0(%[src])    \n\t"
            "fcvt.f16.ps f3, f2           \n\t"
            "fsch.ps     f3, f1(%[dst])   \n\t"
            :
            : [src] "r"(src + d), [dst] "r"(dst + d)
            : "f2", "f3", "memory");
    }

    __asm__ অস্থির("mova.m.x  %[ms]             \n\t" : : [ms] "r"(old_mask));
}

স্থির অন্তঃরেখ শূন্য __attribute__((always_inline)) zero_acc_vec(দশমিক * acc, int64_t dv) {
    ধ্রুবক দশমিক   zero = 0.0f;
    unsigned long old_mask;
    __asm__ অস্থির("mova.x.m %0" : "=r"(old_mask));
    __asm__ অস্থির("mov.m.x m0, x0, 0xFF");
    __asm__ অস্থির("fbc.ps  f2, 0(%[z])" ::[z] "r"(&zero) : "f2");

    জন্য (int64_t d = 0; d < dv; d += 8) {
        __asm__ অস্থির("fsw.ps  f2, 0(%[a])     \n\t" ::[a] "r"(acc + d) : "f2", "memory");
    }

    __asm__ অস্থির("mova.m.x %0" ::"r"(old_mask));
}

স্থির অন্তঃরেখ শূন্য __attribute__((always_inline)) scale_acc_vec(দশমিক * acc, int64_t dv, দশমিক scale) {
    unsigned long old_mask;
    __asm__ অস্থির("mova.x.m %0" : "=r"(old_mask));
    __asm__ অস্থির("mov.m.x m0, x0, 0xFF");

    জন্য (int64_t d = 0; d < dv; d += 8) {
        __asm__ অস্থির(
            "fbc.ps    f2, 0(%[s])    \n\t"
            "flw.ps    f3, 0(%[a])    \n\t"
            "fmul.ps   f3, f3, f2     \n\t"
            "fsw.ps    f3, 0(%[a])    \n\t"
            :
            : [s] "r"(&scale), [a] "r"(acc + d)
            : "f2", "f3", "memory");
    }

    __asm__ অস্থির("mova.m.x %0" ::"r"(old_mask));
}

স্থির অন্তঃরেখ শূন্য __attribute__((always_inline)) normalize_store_vec(দশমিক * out,
                                                                      দশমিক * acc,
                                                                      int64_t dv,
                                                                      দশমিক   inv,
                                                                      পূর্ণসংখ্যা     use_fast_store) {
    unsigned long old_mask;
    __asm__ অস্থির("mova.x.m %0" : "=r"(old_mask));
    __asm__ অস্থির("mov.m.x m0, x0, 0xFF");

    জন্য (int64_t d = 0; d < dv; d += 8) {
        __asm__ অস্থির(
            "fbc.ps    f2, 0(%[inv])   \n\t"
            "flw.ps    f3, 0(%[a])     \n\t"
            "fmul.ps   f3, f3, f2      \n\t"
            "fsw.ps    f3, 0(%[a])     \n\t"
            :
            : [inv] "r"(&inv), [a] "r"(acc + d)
            : "f2", "f3", "memory");
        যদি (use_fast_store) {
            __asm__ অস্থির(
                "flw.ps  f4, 0(%[a])     \n\t"
                "fsw.ps  f4, 0(%[o])     \n\t"
                :
                : [a] "r"(acc + d), [o] "r"(out + d)
                : "f4", "memory");
        } অন্যথা {
            atomic_store_f32((অস্থির দশমিক *) &out[d + 0], acc[d + 0]);
            atomic_store_f32((অস্থির দশমিক *) &out[d + 1], acc[d + 1]);
            atomic_store_f32((অস্থির দশমিক *) &out[d + 2], acc[d + 2]);
            atomic_store_f32((অস্থির দশমিক *) &out[d + 3], acc[d + 3]);
            atomic_store_f32((অস্থির দশমিক *) &out[d + 4], acc[d + 4]);
            atomic_store_f32((অস্থির দশমিক *) &out[d + 5], acc[d + 5]);
            atomic_store_f32((অস্থির দশমিক *) &out[d + 6], acc[d + 6]);
            atomic_store_f32((অস্থির দশমিক *) &out[d + 7], acc[d + 7]);
        }
    }

    __asm__ অস্থির("mova.m.x %0" ::"r"(old_mask));
}

স্থির অন্তঃরেখ size_t tensor_bytes_fa(ধ্রুবক কাঠামো ggml_tensor * t) {
    ফেরত (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

// Evict a byte range from L1D to L2 SCP, splitting into batches of ≤16
// cache lines (the hw limit for evict_to_l2). Use before a barrier when
// another minion in the shire needs to read the region, or after a barrier
// on the reader side to drop stale L1D copies before reading peer data.
স্থির অন্তঃরেখ শূন্য __attribute__((always_inline)) evict_range_to_l2(ধ্রুবক শূন্য * addr, int64_t bytes) {
    যদি (bytes <= 0) {
        ফেরত;
    }
    int64_t      সারিগুলি = (bytes + 63) / 64;
    ধ্রুবক অক্ষর * p     = (ধ্রুবক অক্ষর *) addr;
    যখন (সারিগুলি > 0) {
        int64_t batch = সারিগুলি > 16 ? 16 : সারিগুলি;
        evict_to_l2((ধ্রুবক শূন্য *) p, (uint64_t) batch, 64);
        p += batch * 64;
        সারিগুলি -= batch;
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
স্থির অন্তঃরেখ শূন্য __attribute__((always_inline)) merge_rescale_add_asm(দশমিক *       acc,
                                                                        ধ্রুবক দশমিক * peer_acc,
                                                                        int64_t       dv,
                                                                        দশমিক         alpha_own,
                                                                        দশমিক         alpha_peer) {
    unsigned long old_mask;
    __asm__ অস্থির(
        "mova.x.m  %[ms]              \n\t"
        "mov.m.x   m0, x0, 0xFF       \n\t"
        "fbc.ps    f4, 0(%[ao])       \n\t"  // broadcast alpha_own
        "fbc.ps    f5, 0(%[ap])       \n\t"  // broadcast alpha_peer
        : [ms] "=&r"(old_mask)
        : [ao] "r"(&alpha_own), [ap] "r"(&alpha_peer)
        : "f4", "f5");

    জন্য (int64_t d = 0; d < dv; d += 8) {
        __asm__ অস্থির(
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

    __asm__ অস্থির("mova.m.x %0" ::"r"(old_mask));
}

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_flash_attn_ext_params * params, শূন্য * env) {
    (শূন্য) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    যদি (shire_id >= অঙ্ক_COMPUTE_SHIRES) {
        ফেরত 0;
    }

    ধ্রুবক পূর্ণসংখ্যা is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    কাঠামো ggml_tensor * q        = &params->src0;
    কাঠামো ggml_tensor * k        = &params->src1;
    কাঠামো ggml_tensor * v        = &params->src2;
    কাঠামো ggml_tensor * dst      = &params->dst;
    ধ্রুবক int32_t        has_mask = params->has_mask;
    কাঠামো ggml_tensor * mask     = has_mask ? &params->mask : (কাঠামো ggml_tensor *) 0;

    ধ্রুবক অক্ষর * q_তথ্য   = (ধ্রুবক অক্ষর *) q->তথ্য;
    ধ্রুবক অক্ষর * k_তথ্য   = (ধ্রুবক অক্ষর *) k->তথ্য;
    ধ্রুবক অক্ষর * v_তথ্য   = (ধ্রুবক অক্ষর *) v->তথ্য;
    অক্ষর *       dst_তথ্য = (অক্ষর *) dst->তথ্য;

    // et_barrier(ET_BARRIER_GLOBAL);
    evict_region_past_l2(q->তথ্য, tensor_bytes_fa(q));
    evict_region_past_l2(k->তথ্য, tensor_bytes_fa(k));
    evict_region_past_l2(v->তথ্য, tensor_bytes_fa(v));
    যদি (mask) {
        evict_region_past_l2(mask->তথ্য, tensor_bytes_fa(mask));
    }
    et_barrier(ET_BARRIER_GLOBAL);

    ধ্রুবক int64_t dk  = q->ne[0];
    ধ্রুবক int64_t nq  = q->ne[1];
    ধ্রুবক int64_t nhq = q->ne[2];
    ধ্রুবক int64_t no  = q->ne[3];
    ধ্রুবক int64_t nk  = k->ne[1];
    ধ্রুবক int64_t nhk = k->ne[2];
    ধ্রুবক int64_t dv  = v->ne[0];

    যদি (dv > FA_DV_MAX || dk > FA_DK_MAX) {
        ফেরত -1;
    }
    যদি (k->nb[0] != 2 || v->nb[0] != 2) {
        ফেরত -1;
    }
    যদি ((dk % 8) != 0 || (dv % 16) != 0) {
        ফেরত -1;
    }

    ধ্রুবক int64_t gqa_ratio      = nhq / nhk;
    ধ্রুবক int64_t যোগ_rows     = nq * nhq * no;
    ধ্রুবক দশমিক   scale          = params->scale;
    ধ্রুবক পূর্ণসংখ্যা     use_fast_store = (dv % 16 == 0);

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
    ধ্রুবক int64_t nk_tiles      = (nk + TILE_KV - 1) / TILE_KV;
    ধ্রুবক int64_t যোগ_minions = 2 * অঙ্ক_COMPUTE_SHIRES * MINIONS_PER_SHIRE;
    int64_t       k_splits      = 1;
    যদি (যোগ_rows < যোগ_minions) {
        int64_t target = যোগ_minions / যোগ_rows;
        int64_t ks     = 1;
        যখন (ks * 2 <= target && ks * 2 <= MINIONS_PER_SHIRE && ks * 2 <= nk_tiles) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ধ্রুবক int64_t tiles_per_shire = MINIONS_PER_SHIRE / k_splits;
    ধ্রুবক int64_t k_split         = (int64_t) local_minion % k_splits;
    ধ্রুবক int64_t local_tile_idx  = (int64_t) local_minion / k_splits;
    ধ্রুবক int64_t tiles_stride    = (int64_t) অঙ্ক_COMPUTE_SHIRES * tiles_per_shire;

    // KV slab for this k_split. With k_splits=1 this is the full range.
    ধ্রুবক int64_t tiles_per_split_rounded = (nk_tiles + k_splits - 1) / k_splits;
    ধ্রুবক int64_t tile_start              = k_split * tiles_per_split_rounded;
    int64_t       tile_end                = tile_start + tiles_per_split_rounded;
    যদি (tile_end > nk_tiles) {
        tile_end = nk_tiles;
    }
    ধ্রুবক int64_t kv_start = tile_start * TILE_KV;
    int64_t       kv_end   = tile_end * TILE_KV;
    যদি (kv_end > nk) {
        kv_end = nk;
    }

    // L2 SCP pointers for this minion
    uint64_t    scp_base  = local_minion * SCP_PER_MINION;
    et_fp16_t * scp_kp[2] = {
        (et_fp16_t *) et_shire_l2scp_local(scp_base + SCP_KP0_OFF),
        (et_fp16_t *) et_shire_l2scp_local(scp_base + SCP_KP1_OFF),
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
    যদি (is_hart1) {
        uint32_t      chunk_id = 0;
        ধ্রুবক int64_t row_base = (int64_t) shire_id + local_tile_idx * অঙ্ক_COMPUTE_SHIRES;

        int64_t max_iters;
        যদি (k_splits > 1) {
            max_iters = (যোগ_rows + tiles_stride - 1) / tiles_stride;
        } অন্যথা {
            max_iters = (row_base >= যোগ_rows) ? 0 : ((যোগ_rows - row_base - 1) / tiles_stride + 1);
        }

        জন্য (int64_t iter = 0; iter < max_iters; iter++) {
            ধ্রুবক int64_t row      = row_base + iter * tiles_stride;
            ধ্রুবক পূর্ণসংখ্যা     has_work = (row < যোগ_rows);

            যদি (has_work) {
                ধ্রুবক int64_t iq3 = row / (nhq * nq);
                ধ্রুবক int64_t rem = row % (nhq * nq);
                ধ্রুবক int64_t iq2 = rem / nq;
                ধ্রুবক int64_t ik2 = iq2 / gqa_ratio;

                ধ্রুবক অক্ষর * k_শীর্ষ = k_তথ্য + ik2 * k->nb[2] + iq3 * k->nb[3];

                জন্য (int64_t kv_base = kv_start; kv_base < kv_end; kv_base += TILE_KV) {
                    ধ্রুবক int64_t kv_গুনতি = (kv_base + TILE_KV <= nk) ? TILE_KV : (nk - kv_base);

                    জন্য (int64_t dk_chunk = 0; dk_chunk < dk; dk_chunk += TILE_K) {
                        পূর্ণসংখ্যা buf = chunk_id & 1;

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
                        যদি (chunk_id >= 2) {
                            et_sem_wait(ET_BARRIER_MINION);
                        }

                        // Prefetch K data for this chunk
                        prefetch_kv_to_l2(k_শীর্ষ, kv_base, dk_chunk, kv_গুনতি, k->nb[1]);

                        pack_k_for_transpose16(scp_kp[buf], k_শীর্ষ, kv_base, dk_chunk, kv_গুনতি, k->nb[1]);

                        FENCE;
                        flush_to_l2(scp_kp[buf], 16, 64);
                        flush_to_l2((et_fp16_t *) ((অক্ষর *) scp_kp[buf] + 1024), 16, 64);
                        WAIT_CACHEOPS;

                        // Signal: this buf is ready for hart 0 to consume.
                        et_sem_post(ET_BARRIER_MINION);

                        chunk_id++;
                    }
                }
            }

            // Shire barriers for split-KV merge (hart 1 is a passive arrival).
            যদি (k_splits > 1) {
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
        জন্য (uint32_t d = 0; d < drain; d++) {
            et_sem_wait(ET_BARRIER_MINION);
        }

        // FENCE;
        // et_barrier(ET_BARRIER_GLOBAL);
        ফেরত 0;
    }

    // Hart 0: tensor engine compute
#ifndef UBERKERNEL_SUPPRESS_SCP_SETUP
    setup_cache_scp();
#endif
    CLEAR_TENSOR_ERROR;

    // Q converted to F16 (one row at a time)
    et_fp16_t q_f16[FA_DK_MAX] __attribute__((aligned(64)));

    // Score buffer for QK^T output (16 scores per KV tile)
    দশমিক scores[TILE_KV] __attribute__((aligned(64)));

    // Small buffers for V accumulation
    et_fp16_t w_f16_buf[32] __attribute__((aligned(64)));       // 64 bytes
    et_fp16_t vpanel_buf[8 * 32] __attribute__((aligned(64)));  // 512 bytes

    দশমিক * acc = (দশমিক *) et_shire_l2scp_local(scp_base + SCP_ACC_OFF);

    uint32_t chunk_id = 0;

    // Iter-based outer loop (matches hart 1). When k_splits > 1 all teams
    // in a shire iterate the same number of times so the per-row shire
    // barriers stay balanced; iterations with row >= total_rows skip the
    // compute but still participate in the barriers.
    ধ্রুবক int64_t hart0_row_base = (int64_t) shire_id + local_tile_idx * অঙ্ক_COMPUTE_SHIRES;
    int64_t       hart0_max_iters;
    যদি (k_splits > 1) {
        hart0_max_iters = (যোগ_rows + tiles_stride - 1) / tiles_stride;
    } অন্যথা {
        hart0_max_iters = (hart0_row_base >= যোগ_rows) ? 0 : ((যোগ_rows - hart0_row_base - 1) / tiles_stride + 1);
    }

    জন্য (int64_t iter = 0; iter < hart0_max_iters; iter++) {
        ধ্রুবক int64_t row = hart0_row_base + iter * tiles_stride;
        যদি (row >= যোগ_rows) {
            // No-work iteration: only participate in barriers (k_splits > 1).
            যদি (k_splits > 1) {
                et_barrier(ET_BARRIER_SHIRE);  // A
                et_barrier(ET_BARRIER_SHIRE);  // B
            }
            চালু;
        }

        ধ্রুবক int64_t iq3 = row / (nhq * nq);
        ধ্রুবক int64_t rem = row % (nhq * nq);
        ধ্রুবক int64_t iq2 = rem / nq;
        ধ্রুবক int64_t iq1 = rem % nq;
        ধ্রুবক int64_t ik2 = iq2 / gqa_ratio;

        // Read Q row (F32) and convert to F16
        ধ্রুবক দশমিক * pq = (ধ্রুবক দশমিক *) (q_তথ্য + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);
        convert_q_row_f32_to_f16(q_f16, pq, dk);

        // V base for this head + batch (K packing handled by hart 1)
        ধ্রুবক অক্ষর * v_শীর্ষ = v_তথ্য + ik2 * v->nb[2] + iq3 * v->nb[3];

        // Output pointer
        দশমিক * out = (দশমিক *) (dst_তথ্য + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        zero_acc_vec(acc, dv);
        দশমিক        M         = ET_NEG_INF_F;
        দশমিক        S         = 0.0f;
        ধ্রুবক অক্ষর * mask_base = has_mask ? get_mask_row_base(mask, iq1, iq2, iq3) : (ধ্রুবক অক্ষর *) 0;

        // Flush Q_f16 to L2 so tensor_load can see it
        FENCE;
        flush_to_l2(q_f16, (dk * 2 + 63) / 64, 64);
        WAIT_CACHEOPS;

        জন্য (int64_t kv_base = kv_start; kv_base < kv_end; kv_base += TILE_KV) {
            ধ্রুবক int64_t kv_গুনতি = (kv_base + TILE_KV <= nk) ? TILE_KV : (nk - kv_base);

            // Set tensor_mask for partial tiles
            যদি (kv_গুনতি < TILE_KV) {
                uint64_t tmask = (1ULL << kv_গুনতি) - 1;
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
            ধ্রুবক int64_t  n_dk_chunks = dk / TILE_K;
            ধ্রুবক uint64_t K_BUFS[2]   = {
                (uint64_t) B_L1_START,         // 16..31
                (uint64_t) (B_L1_START + 16),  // 32..47
            };

            // Preload entire Q row into A_L1[0..n_dk_chunks-1] (one tensor_load,
            // one wait, regardless of dk).
            tensor_load(false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0, (uint64_t) q_f16, 0,
                        (uint64_t) (n_dk_chunks - 1), 64, 0);

            // Prologue: wait hart 1's K[0], issue K[0] load, wait both loads.
            {
                পূর্ণসংখ্যা buf = chunk_id & 1;
                et_sem_wait(ET_BARRIER_MINION);
                tensor_load(false, false, K_BUFS[0], TENSOR_LOAD_TRANSPOSE16, 0, (uint64_t) scp_kp[buf], 0, 15, 64, 1);
                tensor_wait(TENSOR_LOAD_WAIT_0);  // Q row complete
                tensor_wait(TENSOR_LOAD_WAIT_1);  // K[0] complete
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
            জন্য (int64_t i = 1; i < n_dk_chunks; i++) {
                পূর্ণসংখ্যা buf         = chunk_id & 1;
                পূর্ণসংখ্যা k_slot_পূর্ব = (পূর্ণসংখ্যা) ((i - 1) & 1);
                পূর্ণসংখ্যা k_slot      = (পূর্ণসংখ্যা) (i & 1);

                et_sem_wait(ET_BARRIER_MINION);
                tensor_load(false, false, K_BUFS[k_slot], TENSOR_LOAD_TRANSPOSE16, 0, (uint64_t) scp_kp[buf], 0, 15, 64,
                            1);

                tensor_fma((kv_গুনতি < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_slot_পূর্ব],
                           (uint64_t) (i - 1), TENSOR_FMA_OP_FP16, (i == 1));

                tensor_wait(TENSOR_LOAD_WAIT_1);  // K[i] in L1
                et_sem_post(ET_BARRIER_MINION);   // release scp_kp[buf] EARLY
                tensor_wait(TENSOR_FMA_WAIT);     // then wait FMA[i-1]
                chunk_id++;
            }

            // Epilogue: FMA on the last chunk (no overlapping load).
            {
                পূর্ণসংখ্যা k_slot_last = (পূর্ণসংখ্যা) ((n_dk_chunks - 1) & 1);
                tensor_fma((kv_গুনতি < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_slot_last],
                           (uint64_t) (n_dk_chunks - 1), TENSOR_FMA_OP_FP16, (n_dk_chunks == 1));
                tensor_wait(TENSOR_FMA_WAIT);
            }

            // Prefetch V rows for this tile.
            // Only useful for the partial-tile path below
            যদি (kv_গুনতি < TILE_KV) {
                জন্য (int64_t d = 0; d < dv; d += 32) {
                    prefetch_kv_to_l2(v_শীর্ষ, kv_base, d, kv_গুনতি, v->nb[1]);
                }
            }

            // Extract QK^T scores from vector register file
            __asm__ অস্থির("" ::: "f0", "f1");
            {
                unsigned long _ms;
                __asm__ অস্থির(
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

            দশমিক weights[TILE_KV] __attribute__((aligned(64)));
            {
                // A1: apply mask to scores, pad unused slots
                জন্য (int64_t j = 0; j < kv_গুনতি; ++j) {
                    যদি (has_mask) {
                        দশমিক mv = get_mask_val_from_base(mask, mask_base, kv_base + j);
                        যদি (mv == ET_NEG_INF_F || mv != mv) {
                            scores[j] = ET_NEG_INF_F;
                        } অন্যথা {
                            scores[j] += mv;
                        }
                    }
                }
                জন্য (int64_t j = kv_গুনতি; j < TILE_KV; ++j) {
                    scores[j] = ET_NEG_INF_F;
                }

                // A1b: SIMD horizontal max across all 16 scores
                দশমিক tile_max;
                {
                    unsigned long _ms;
                    __asm__ অস্থির(
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

                যদি (tile_max > ET_NEG_INF_F) {
                    // A2: rescale accumulator if this tile has a new global max
                    যদি (tile_max > M) {
                        দশমিক rescale = et_exp2f((M - tile_max) * 1.4426950408889634f);
                        scale_acc_vec(acc, dv, rescale);
                        S *= rescale;
                        M = tile_max;
                    }

                    // A3: SIMD exp2 + horizontal sum
                    // Interleaved: f2/f3 chains alternate to hide ALU latency.
                    // fexp.ps has multi-cycle latency — the two independent
                    // exp2 calls naturally pipeline.
                    {
                        ধ্রুবক দশমিক   log2e = 1.4426950408889634f;
                        দশমিক         S_tile;
                        unsigned long _ms;
                        __asm__ অস্থির(
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
                        convert_q_row_f32_to_f16(w_f16_buf, weights, TILE_KV);

                        FENCE;
                        flush_to_l2(w_f16_buf, 1, 64);
                        WAIT_CACHEOPS;

                        // Issue weights load (wait_id=0) and the first V chunk
                        // load (wait_id=1) concurrently. Weights comes from
                        // L2 SCP (just flushed); V[0] comes from DRAM via
                        // INTERLEAVE16 — running them in parallel hides the
                        // shorter load behind the longer one. For partial
                        // tiles, V is software-packed below — we only kick
                        // off the early V load on the full-tile fast path.
                        tensor_load(false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0, (uint64_t) w_f16_buf, 0, 0, 64, 0);

                        ধ্রুবক পূর্ণসংখ্যা       v_full_tile = (kv_গুনতি == TILE_KV);
                        ধ্রুবক uintptr_t v_base      = (uintptr_t) v_শীর্ষ + kv_base * v->nb[1];
                        ধ্রুবক uint64_t  nb1_v       = (uint64_t) v->nb[1];
                        uint64_t        b_cur       = 8;

                        যদি (v_full_tile) {
                            tensor_load(false, false, b_cur, TENSOR_LOAD_INTERLEAVE16, 0, (uint64_t) v_base, 0, 7,
                                        nb1_v, 1);
                        }

                        tensor_wait(TENSOR_LOAD_WAIT_0);      // weights in A_L1
                        যদি (v_full_tile) {
                            tensor_wait(TENSOR_LOAD_WAIT_1);  // V[0] in b_cur
                        }

                        // B2: process dv in chunks of 16
                        যদি (v_full_tile) {
                            জন্য (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                ধ্রুবক uint64_t b_nxt = b_cur ^ 24;

                                যদি (dv_off + 16 < dv) {
                                    tensor_load(false, false, b_nxt, TENSOR_LOAD_INTERLEAVE16, 0,
                                                (uint64_t) (v_base + (dv_off + 16) * 2), 0, 7, nb1_v, 1);
                                }

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, b_cur, A_L1_START,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ অস্থির("" ::: "f0", "f1");
                                {
                                    unsigned long _ms;
                                    __asm__ অস্থির(
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

                                যদি (dv_off + 16 < dv) {
                                    tensor_wait(TENSOR_LOAD_WAIT_1);
                                    b_cur = b_nxt;
                                }
                            }
                        } অন্যথা {
                            // Partial tile: software pack, no pipeline
                            জন্য (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                pack_v_interleaved(vpanel_buf, v_শীর্ষ, kv_base, dv_off, kv_গুনতি, v->nb[1]);
                                FENCE;
                                flush_to_l2(vpanel_buf, 8, 64);
                                WAIT_CACHEOPS;
                                tensor_load(false, false, B_L1_START, TENSOR_LOAD_PLAIN, 0, (uint64_t) vpanel_buf, 0, 7,
                                            64, 0);
                                tensor_wait(TENSOR_LOAD_WAIT_0);

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, B_L1_START, A_L1_START,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ অস্থির("" ::: "f0", "f1");
                                {
                                    unsigned long _ms;
                                    __asm__ অস্থির(
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
        যদি (k_splits > 1) {
            // Publish our partial.
            অস্থির দশমিক * my_stats = (অস্থির দশমিক *) et_shire_l2scp_local(scp_base + SCP_STATS_OFF);
            my_stats[0]               = M;
            my_stats[1]               = S;
            FENCE;
            evict_range_to_l2(acc, (int64_t) dv * (int64_t) আকার(দশমিক));
            evict_to_l2((ধ্রুবক শূন্য *) my_stats, 1, 64);
            WAIT_CACHEOPS;

            // A: team members have all written their partials.
            et_barrier(ET_BARRIER_SHIRE);

            যদি (k_split == 0) {
                // Online softmax merge: fold peers 1..k_splits-1 into our
                // own (M_running, S_running, acc). For each peer p:
                //   M_new    = max(M_running, M_p)
                //   α_own    = exp2((M_running - M_new) * log2e)
                //   α_p      = exp2((M_p     - M_new) * log2e)
                //   acc[d]   = α_own * acc[d] + α_p * peer_acc[d]
                //   S_running = α_own * S_running + α_p * S_p
                দশমিক       M_running = M;
                দশমিক       S_running = S;
                ধ্রুবক দশমিক log2e     = 1.4426950408889634f;

                জন্য (int64_t p = 1; p < k_splits; p++) {
                    uint64_t         peer_scp   = (local_tile_idx * k_splits + p) * SCP_PER_MINION;
                    অস্থির দশমিক * peer_stats = (অস্থির দশমিক *) et_shire_l2scp_local(peer_scp + SCP_STATS_OFF);
                    দশমিক *          peer_acc   = (দশমিক *) et_shire_l2scp_local(peer_scp + SCP_ACC_OFF);

                    // Drop stale L1D copies before reading peer's data.
                    evict_to_l2((ধ্রুবক শূন্য *) peer_stats, 1, 64);
                    evict_range_to_l2(peer_acc, (int64_t) dv * (int64_t) আকার(দশমিক));
                    WAIT_CACHEOPS;

                    ধ্রুবক দশমিক M_p = peer_stats[0];
                    ধ্রুবক দশমিক S_p = peer_stats[1];

                    ধ্রুবক দশমিক M_new     = (M_p > M_running) ? M_p : M_running;
                    ধ্রুবক দশমিক alpha_own = (M_running == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_running - M_new) * log2e);
                    ধ্রুবক দশমিক alpha_p   = (M_p == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_p - M_new) * log2e);

                    merge_rescale_add_asm(acc, peer_acc, dv, alpha_own, alpha_p);

                    S_running = alpha_own * S_running + alpha_p * S_p;
                    M_running = M_new;
                }

                ধ্রুবক দশমিক S_inv = (S_running == 0.0f) ? 0.0f : et_fdiv(1.0f, S_running);
                normalize_store_vec(out, acc, dv, S_inv, use_fast_store);
            }

            // B: reducer is done, team may reuse its acc/stats slabs.
            et_barrier(ET_BARRIER_SHIRE);
        } অন্যথা {
            // k_splits == 1 fast path — this minion owns the full row.
            ধ্রুবক দশমিক S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
            normalize_store_vec(out, acc, dv, S_inv, use_fast_store);
        }
    }

    FENCE;
    ফেরত 0;
}
