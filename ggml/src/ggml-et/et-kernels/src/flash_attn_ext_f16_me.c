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

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"
#अन्तर्भाव "tensor.h"

#अन्तर्भाव <etsoc/common/utils.h>
#अन्तर्भाव <stdint.h>
#अन्तर्भाव <string.h>

#define अंक_COMPUTE_SHIRES 32
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

प्रकारनिर्देश uint16_t et_fp16_t;

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
#define SCP_ACC_STRIDE  (FA_DV_MAX * परिमाण(भिन्न))       // 2048
#define SCP_KPANEL_SIZE (32 * 32 * परिमाण(et_fp16_t))     // 2048
#define SCP_KP0_OFF     SCP_ACC_STRIDE                    // 2048
#define SCP_KP1_OFF     (SCP_KP0_OFF + SCP_KPANEL_SIZE)   // 4096
#define SCP_STATS_OFF   (SCP_KP1_OFF + SCP_KPANEL_SIZE)   // 6144
#define SCP_STATS_SIZE  64                                // own cache line
#define SCP_PER_MINION  (SCP_STATS_OFF + SCP_STATS_SIZE)  // 6208

संरचना ggml_et_flash_attn_ext_params {
    संरचना ggml_tensor src0;  // Q (F32)
    संरचना ggml_tensor src1;  // K (F16)
    संरचना ggml_tensor src2;  // V (F16)
    संरचना ggml_tensor mask;  // mask (F16 or F32), zeroed when absent
    संरचना ggml_tensor dst;   // Output (F32)
    भिन्न              scale;
    int32_t            has_mask;
};

स्थिर अन्तःस्थ भिन्न get_mask_val(ध्रुव संरचना ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    ध्रुव वर्ण * base = (ध्रुव वर्ण *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    यदि (mask->type == GGML_TYPE_F32) {
        प्रत्यागम *(ध्रुव भिन्न *) (base + ik1 * mask->nb[0]);
    }
    प्रत्यागम fp16_to_fp32(*(ध्रुव uint16_t *) (base + ik1 * mask->nb[0]));
}

स्थिर अन्तःस्थ ध्रुव वर्ण * get_mask_row_base(ध्रुव संरचना ggml_tensor * mask, int64_t iq1, int64_t iq2, int64_t iq3) {
    प्रत्यागम (ध्रुव वर्ण *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
           (iq3 % mask->ne[3]) * mask->nb[3];
}

स्थिर अन्तःस्थ भिन्न get_mask_val_from_base(ध्रुव संरचना ggml_tensor * mask, ध्रुव वर्ण * base, int64_t ik1) {
    यदि (mask->type == GGML_TYPE_F32) {
        प्रत्यागम *(ध्रुव भिन्न *) (base + ik1 * mask->nb[0]);
    }
    प्रत्यागम fp16_to_fp32(*(ध्रुव uint16_t *) (base + ik1 * mask->nb[0]));
}

// Pack K rows for TensorLoadTranspose16 (even/odd deinterleave)
स्थिर अन्तःस्थ शून्य __attribute__((always_inline)) pack_k_for_transpose16(et_fp16_t *  out,
                                                                         ध्रुव वर्ण * k_base,
                                                                         int64_t      kv_start,
                                                                         int64_t      dk_start,
                                                                         int64_t      kv_गिनती,
                                                                         int64_t      nb1_k) {
    unsigned long old_mask;
    __asm__ चञ्चल(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        : [ms] "=&r"(old_mask)
        :
        :);

    प्रति (पूर्णाङ्क j = 0; j < (पूर्णाङ्क) kv_गिनती; ++j) {
        ध्रुव et_fp16_t * k_row    = (ध्रुव et_fp16_t *) (k_base + (kv_start + j) * nb1_k) + dk_start;
        et_fp16_t *       even_row = out + (j * 2) * 32;
        et_fp16_t *       odd_row  = out + (j * 2 + 1) * 32;
        __asm__ चञ्चल(
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

    __asm__ चञ्चल("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_mask));

    प्रति (पूर्णाङ्क j = (पूर्णाङ्क) kv_गिनती; j < TILE_KV; ++j) {
        et_fp16_t * even_row = out + (j * 2) * 32;
        et_fp16_t * odd_row  = out + (j * 2 + 1) * 32;
        प्रति (पूर्णाङ्क l = 0; l < TILE_K / 2; ++l) {
            even_row[l] = 0;
            odd_row[l]  = 0;
        }
    }
}

// Build interleaved B panel for TensorFMA16A32 (weights @ V).
स्थिर अन्तःस्थ शून्य __attribute__((always_inline)) pack_v_interleaved(et_fp16_t *  out,
                                                                     ध्रुव वर्ण * v_head,
                                                                     int64_t      kv_base,
                                                                     int64_t      dv_start,
                                                                     int64_t      kv_गिनती,
                                                                     int64_t      nb1_v) {
    प्रति (पूर्णाङ्क k = 0; k < TILE_KV; ++k) {
        ध्रुव पूर्णाङ्क         l   = k >> 1;
        ध्रुव पूर्णाङ्क         r   = k & 1;
        et_fp16_t * ध्रुव dst = out + l * 32 + r;
        यदि (k < (पूर्णाङ्क) kv_गिनती) {
            ध्रुव et_fp16_t * v_row = (ध्रुव et_fp16_t *) (v_head + (kv_base + k) * nb1_v) + dv_start;
            प्रति (पूर्णाङ्क n = 0; n < 16; ++n) {
                dst[n * 2] = v_row[n];
            }
        } अन्यथा {
            प्रति (पूर्णाङ्क n = 0; n < 16; ++n) {
                dst[n * 2] = 0;
            }
        }
    }
}

// Prefetch KV rows for one chunk into L2.
स्थिर अन्तःस्थ शून्य __attribute__((always_inline)) prefetch_kv_to_l2(ध्रुव वर्ण * head,
                                                                    int64_t      kv_start,
                                                                    int64_t      d_start,
                                                                    int64_t      kv_गिनती,
                                                                    int64_t      nb1) {
    ध्रुव शून्य * base = (ध्रुव शून्य *) (head + kv_start * nb1 + d_start * 2);
    l2_prefetch(base, (uint64_t) kv_गिनती, (uint64_t) nb1);
}

स्थिर अन्तःस्थ शून्य __attribute__((always_inline)) convert_q_row_f32_to_f16(et_fp16_t *   dst,
                                                                           ध्रुव भिन्न * src,
                                                                           int64_t       n) {
    स्थिर ध्रुव int32_t __attribute__((aligned(32))) offsets[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };

    unsigned long old_mask;
    __asm__ चञ्चल(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 0xFF      \n\t"
        "flw.ps    f1, 0(%[offs])    \n\t"
        : [ms] "=&r"(old_mask)
        : [offs] "r"(offsets)
        : "f1");

    प्रति (int64_t d = 0; d < n; d += 8) {
        __asm__ चञ्चल(
            "flw.ps      f2, 0(%[src])    \n\t"
            "fcvt.f16.ps f3, f2           \n\t"
            "fsch.ps     f3, f1(%[dst])   \n\t"
            :
            : [src] "r"(src + d), [dst] "r"(dst + d)
            : "f2", "f3", "memory");
    }

    __asm__ चञ्चल("mova.m.x  %[ms]             \n\t" : : [ms] "r"(old_mask));
}

स्थिर अन्तःस्थ शून्य __attribute__((always_inline)) zero_acc_vec(भिन्न * acc, int64_t dv) {
    ध्रुव भिन्न   zero = 0.0f;
    unsigned long old_mask;
    __asm__ चञ्चल("mova.x.m %0" : "=r"(old_mask));
    __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");
    __asm__ चञ्चल("fbc.ps  f2, 0(%[z])" ::[z] "r"(&zero) : "f2");

    प्रति (int64_t d = 0; d < dv; d += 8) {
        __asm__ चञ्चल("fsw.ps  f2, 0(%[a])     \n\t" ::[a] "r"(acc + d) : "f2", "memory");
    }

    __asm__ चञ्चल("mova.m.x %0" ::"r"(old_mask));
}

स्थिर अन्तःस्थ शून्य __attribute__((always_inline)) scale_acc_vec(भिन्न * acc, int64_t dv, भिन्न scale) {
    unsigned long old_mask;
    __asm__ चञ्चल("mova.x.m %0" : "=r"(old_mask));
    __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");

    प्रति (int64_t d = 0; d < dv; d += 8) {
        __asm__ चञ्चल(
            "fbc.ps    f2, 0(%[s])    \n\t"
            "flw.ps    f3, 0(%[a])    \n\t"
            "fmul.ps   f3, f3, f2     \n\t"
            "fsw.ps    f3, 0(%[a])    \n\t"
            :
            : [s] "r"(&scale), [a] "r"(acc + d)
            : "f2", "f3", "memory");
    }

    __asm__ चञ्चल("mova.m.x %0" ::"r"(old_mask));
}

स्थिर अन्तःस्थ शून्य __attribute__((always_inline)) normalize_store_vec(भिन्न * out,
                                                                      भिन्न * acc,
                                                                      int64_t dv,
                                                                      भिन्न   inv,
                                                                      पूर्णाङ्क     use_fast_store) {
    unsigned long old_mask;
    __asm__ चञ्चल("mova.x.m %0" : "=r"(old_mask));
    __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");

    प्रति (int64_t d = 0; d < dv; d += 8) {
        __asm__ चञ्चल(
            "fbc.ps    f2, 0(%[inv])   \n\t"
            "flw.ps    f3, 0(%[a])     \n\t"
            "fmul.ps   f3, f3, f2      \n\t"
            "fsw.ps    f3, 0(%[a])     \n\t"
            :
            : [inv] "r"(&inv), [a] "r"(acc + d)
            : "f2", "f3", "memory");
        यदि (use_fast_store) {
            __asm__ चञ्चल(
                "flw.ps  f4, 0(%[a])     \n\t"
                "fsw.ps  f4, 0(%[o])     \n\t"
                :
                : [a] "r"(acc + d), [o] "r"(out + d)
                : "f4", "memory");
        } अन्यथा {
            atomic_store_f32((चञ्चल भिन्न *) &out[d + 0], acc[d + 0]);
            atomic_store_f32((चञ्चल भिन्न *) &out[d + 1], acc[d + 1]);
            atomic_store_f32((चञ्चल भिन्न *) &out[d + 2], acc[d + 2]);
            atomic_store_f32((चञ्चल भिन्न *) &out[d + 3], acc[d + 3]);
            atomic_store_f32((चञ्चल भिन्न *) &out[d + 4], acc[d + 4]);
            atomic_store_f32((चञ्चल भिन्न *) &out[d + 5], acc[d + 5]);
            atomic_store_f32((चञ्चल भिन्न *) &out[d + 6], acc[d + 6]);
            atomic_store_f32((चञ्चल भिन्न *) &out[d + 7], acc[d + 7]);
        }
    }

    __asm__ चञ्चल("mova.m.x %0" ::"r"(old_mask));
}

स्थिर अन्तःस्थ size_t tensor_bytes_fa(ध्रुव संरचना ggml_tensor * t) {
    प्रत्यागम (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

// Evict a byte range from L1D to L2 SCP, splitting into batches of ≤16
// cache lines (the hw limit for evict_to_l2). Use before a barrier when
// another minion in the shire needs to read the region, or after a barrier
// on the reader side to drop stale L1D copies before reading peer data.
स्थिर अन्तःस्थ शून्य __attribute__((always_inline)) evict_range_to_l2(ध्रुव शून्य * addr, int64_t bytes) {
    यदि (bytes <= 0) {
        प्रत्यागम;
    }
    int64_t      पंक्तियाँ = (bytes + 63) / 64;
    ध्रुव वर्ण * p     = (ध्रुव वर्ण *) addr;
    यावत् (पंक्तियाँ > 0) {
        int64_t batch = पंक्तियाँ > 16 ? 16 : पंक्तियाँ;
        evict_to_l2((ध्रुव शून्य *) p, (uint64_t) batch, 64);
        p += batch * 64;
        पंक्तियाँ -= batch;
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
स्थिर अन्तःस्थ शून्य __attribute__((always_inline)) merge_rescale_add_asm(भिन्न *       acc,
                                                                        ध्रुव भिन्न * peer_acc,
                                                                        int64_t       dv,
                                                                        भिन्न         alpha_own,
                                                                        भिन्न         alpha_peer) {
    unsigned long old_mask;
    __asm__ चञ्चल(
        "mova.x.m  %[ms]              \n\t"
        "mov.m.x   m0, x0, 0xFF       \n\t"
        "fbc.ps    f4, 0(%[ao])       \n\t"  // broadcast alpha_own
        "fbc.ps    f5, 0(%[ap])       \n\t"  // broadcast alpha_peer
        : [ms] "=&r"(old_mask)
        : [ao] "r"(&alpha_own), [ap] "r"(&alpha_peer)
        : "f4", "f5");

    प्रति (int64_t d = 0; d < dv; d += 8) {
        __asm__ चञ्चल(
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

    __asm__ चञ्चल("mova.m.x %0" ::"r"(old_mask));
}

पूर्णाङ्क entry_point(संरचना ggml_et_flash_attn_ext_params * params, शून्य * env) {
    (शून्य) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    यदि (shire_id >= अंक_COMPUTE_SHIRES) {
        प्रत्यागम 0;
    }

    ध्रुव पूर्णाङ्क is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    संरचना ggml_tensor * q        = &params->src0;
    संरचना ggml_tensor * k        = &params->src1;
    संरचना ggml_tensor * v        = &params->src2;
    संरचना ggml_tensor * dst      = &params->dst;
    ध्रुव int32_t        has_mask = params->has_mask;
    संरचना ggml_tensor * mask     = has_mask ? &params->mask : (संरचना ggml_tensor *) 0;

    ध्रुव वर्ण * q_data   = (ध्रुव वर्ण *) q->data;
    ध्रुव वर्ण * k_data   = (ध्रुव वर्ण *) k->data;
    ध्रुव वर्ण * v_data   = (ध्रुव वर्ण *) v->data;
    वर्ण *       dst_data = (वर्ण *) dst->data;

    // et_barrier(ET_BARRIER_GLOBAL);
    evict_region_past_l2(q->data, tensor_bytes_fa(q));
    evict_region_past_l2(k->data, tensor_bytes_fa(k));
    evict_region_past_l2(v->data, tensor_bytes_fa(v));
    यदि (mask) {
        evict_region_past_l2(mask->data, tensor_bytes_fa(mask));
    }
    et_barrier(ET_BARRIER_GLOBAL);

    ध्रुव int64_t dk  = q->ne[0];
    ध्रुव int64_t nq  = q->ne[1];
    ध्रुव int64_t nhq = q->ne[2];
    ध्रुव int64_t no  = q->ne[3];
    ध्रुव int64_t nk  = k->ne[1];
    ध्रुव int64_t nhk = k->ne[2];
    ध्रुव int64_t dv  = v->ne[0];

    यदि (dv > FA_DV_MAX || dk > FA_DK_MAX) {
        प्रत्यागम -1;
    }
    यदि (k->nb[0] != 2 || v->nb[0] != 2) {
        प्रत्यागम -1;
    }
    यदि ((dk % 8) != 0 || (dv % 16) != 0) {
        प्रत्यागम -1;
    }

    ध्रुव int64_t gqa_ratio      = nhq / nhk;
    ध्रुव int64_t total_rows     = nq * nhq * no;
    ध्रुव भिन्न   scale          = params->scale;
    ध्रुव पूर्णाङ्क     use_fast_store = (dv % 16 == 0);

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
    ध्रुव int64_t nk_tiles      = (nk + TILE_KV - 1) / TILE_KV;
    ध्रुव int64_t total_minions = 2 * अंक_COMPUTE_SHIRES * MINIONS_PER_SHIRE;
    int64_t       k_splits      = 1;
    यदि (total_rows < total_minions) {
        int64_t target = total_minions / total_rows;
        int64_t ks     = 1;
        यावत् (ks * 2 <= target && ks * 2 <= MINIONS_PER_SHIRE && ks * 2 <= nk_tiles) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ध्रुव int64_t tiles_per_shire = MINIONS_PER_SHIRE / k_splits;
    ध्रुव int64_t k_split         = (int64_t) local_minion % k_splits;
    ध्रुव int64_t local_tile_idx  = (int64_t) local_minion / k_splits;
    ध्रुव int64_t tiles_stride    = (int64_t) अंक_COMPUTE_SHIRES * tiles_per_shire;

    // KV slab for this k_split. With k_splits=1 this is the full range.
    ध्रुव int64_t tiles_per_split_rounded = (nk_tiles + k_splits - 1) / k_splits;
    ध्रुव int64_t tile_start              = k_split * tiles_per_split_rounded;
    int64_t       tile_end                = tile_start + tiles_per_split_rounded;
    यदि (tile_end > nk_tiles) {
        tile_end = nk_tiles;
    }
    ध्रुव int64_t kv_start = tile_start * TILE_KV;
    int64_t       kv_end   = tile_end * TILE_KV;
    यदि (kv_end > nk) {
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
    यदि (is_hart1) {
        uint32_t      chunk_id = 0;
        ध्रुव int64_t row_base = (int64_t) shire_id + local_tile_idx * अंक_COMPUTE_SHIRES;

        int64_t max_iters;
        यदि (k_splits > 1) {
            max_iters = (total_rows + tiles_stride - 1) / tiles_stride;
        } अन्यथा {
            max_iters = (row_base >= total_rows) ? 0 : ((total_rows - row_base - 1) / tiles_stride + 1);
        }

        प्रति (int64_t iter = 0; iter < max_iters; iter++) {
            ध्रुव int64_t row      = row_base + iter * tiles_stride;
            ध्रुव पूर्णाङ्क     has_work = (row < total_rows);

            यदि (has_work) {
                ध्रुव int64_t iq3 = row / (nhq * nq);
                ध्रुव int64_t rem = row % (nhq * nq);
                ध्रुव int64_t iq2 = rem / nq;
                ध्रुव int64_t ik2 = iq2 / gqa_ratio;

                ध्रुव वर्ण * k_head = k_data + ik2 * k->nb[2] + iq3 * k->nb[3];

                प्रति (int64_t kv_base = kv_start; kv_base < kv_end; kv_base += TILE_KV) {
                    ध्रुव int64_t kv_गिनती = (kv_base + TILE_KV <= nk) ? TILE_KV : (nk - kv_base);

                    प्रति (int64_t dk_chunk = 0; dk_chunk < dk; dk_chunk += TILE_K) {
                        पूर्णाङ्क buf = chunk_id & 1;

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
                        यदि (chunk_id >= 2) {
                            et_sem_wait(ET_BARRIER_MINION);
                        }

                        // Prefetch K data for this chunk
                        prefetch_kv_to_l2(k_head, kv_base, dk_chunk, kv_गिनती, k->nb[1]);

                        pack_k_for_transpose16(scp_kp[buf], k_head, kv_base, dk_chunk, kv_गिनती, k->nb[1]);

                        FENCE;
                        flush_to_l2(scp_kp[buf], 16, 64);
                        flush_to_l2((et_fp16_t *) ((वर्ण *) scp_kp[buf] + 1024), 16, 64);
                        WAIT_CACHEOPS;

                        // Signal: this buf is ready for hart 0 to consume.
                        et_sem_post(ET_BARRIER_MINION);

                        chunk_id++;
                    }
                }
            }

            // Shire barriers for split-KV merge (hart 1 is a passive arrival).
            यदि (k_splits > 1) {
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
        प्रति (uint32_t d = 0; d < drain; d++) {
            et_sem_wait(ET_BARRIER_MINION);
        }

        // FENCE;
        // et_barrier(ET_BARRIER_GLOBAL);
        प्रत्यागम 0;
    }

    // Hart 0: tensor engine compute
#ifndef UBERKERNEL_SUPPRESS_SCP_SETUP
    setup_cache_scp();
#endif
    CLEAR_TENSOR_ERROR;

    // Q converted to F16 (one row at a time)
    et_fp16_t q_f16[FA_DK_MAX] __attribute__((aligned(64)));

    // Score buffer for QK^T output (16 scores per KV tile)
    भिन्न scores[TILE_KV] __attribute__((aligned(64)));

    // Small buffers for V accumulation
    et_fp16_t w_f16_buf[32] __attribute__((aligned(64)));       // 64 bytes
    et_fp16_t vpanel_buf[8 * 32] __attribute__((aligned(64)));  // 512 bytes

    भिन्न * acc = (भिन्न *) et_shire_l2scp_local(scp_base + SCP_ACC_OFF);

    uint32_t chunk_id = 0;

    // Iter-based outer loop (matches hart 1). When k_splits > 1 all teams
    // in a shire iterate the same number of times so the per-row shire
    // barriers stay balanced; iterations with row >= total_rows skip the
    // compute but still participate in the barriers.
    ध्रुव int64_t hart0_row_base = (int64_t) shire_id + local_tile_idx * अंक_COMPUTE_SHIRES;
    int64_t       hart0_max_iters;
    यदि (k_splits > 1) {
        hart0_max_iters = (total_rows + tiles_stride - 1) / tiles_stride;
    } अन्यथा {
        hart0_max_iters = (hart0_row_base >= total_rows) ? 0 : ((total_rows - hart0_row_base - 1) / tiles_stride + 1);
    }

    प्रति (int64_t iter = 0; iter < hart0_max_iters; iter++) {
        ध्रुव int64_t row = hart0_row_base + iter * tiles_stride;
        यदि (row >= total_rows) {
            // No-work iteration: only participate in barriers (k_splits > 1).
            यदि (k_splits > 1) {
                et_barrier(ET_BARRIER_SHIRE);  // A
                et_barrier(ET_BARRIER_SHIRE);  // B
            }
            अनुवर्त;
        }

        ध्रुव int64_t iq3 = row / (nhq * nq);
        ध्रुव int64_t rem = row % (nhq * nq);
        ध्रुव int64_t iq2 = rem / nq;
        ध्रुव int64_t iq1 = rem % nq;
        ध्रुव int64_t ik2 = iq2 / gqa_ratio;

        // Read Q row (F32) and convert to F16
        ध्रुव भिन्न * pq = (ध्रुव भिन्न *) (q_data + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);
        convert_q_row_f32_to_f16(q_f16, pq, dk);

        // V base for this head + batch (K packing handled by hart 1)
        ध्रुव वर्ण * v_head = v_data + ik2 * v->nb[2] + iq3 * v->nb[3];

        // Output pointer
        भिन्न * out = (भिन्न *) (dst_data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        zero_acc_vec(acc, dv);
        भिन्न        M         = ET_NEG_INF_F;
        भिन्न        S         = 0.0f;
        ध्रुव वर्ण * mask_base = has_mask ? get_mask_row_base(mask, iq1, iq2, iq3) : (ध्रुव वर्ण *) 0;

        // Flush Q_f16 to L2 so tensor_load can see it
        FENCE;
        flush_to_l2(q_f16, (dk * 2 + 63) / 64, 64);
        WAIT_CACHEOPS;

        प्रति (int64_t kv_base = kv_start; kv_base < kv_end; kv_base += TILE_KV) {
            ध्रुव int64_t kv_गिनती = (kv_base + TILE_KV <= nk) ? TILE_KV : (nk - kv_base);

            // Set tensor_mask for partial tiles
            यदि (kv_गिनती < TILE_KV) {
                uint64_t tmask = (1ULL << kv_गिनती) - 1;
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
            ध्रुव int64_t  n_dk_chunks = dk / TILE_K;
            ध्रुव uint64_t K_BUFS[2]   = {
                (uint64_t) B_L1_START,         // 16..31
                (uint64_t) (B_L1_START + 16),  // 32..47
            };

            // Preload entire Q row into A_L1[0..n_dk_chunks-1] (one tensor_load,
            // one wait, regardless of dk).
            tensor_load(false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0, (uint64_t) q_f16, 0,
                        (uint64_t) (n_dk_chunks - 1), 64, 0);

            // Prologue: wait hart 1's K[0], issue K[0] load, wait both loads.
            {
                पूर्णाङ्क buf = chunk_id & 1;
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
            प्रति (int64_t i = 1; i < n_dk_chunks; i++) {
                पूर्णाङ्क buf         = chunk_id & 1;
                पूर्णाङ्क k_slot_prev = (पूर्णाङ्क) ((i - 1) & 1);
                पूर्णाङ्क k_slot      = (पूर्णाङ्क) (i & 1);

                et_sem_wait(ET_BARRIER_MINION);
                tensor_load(false, false, K_BUFS[k_slot], TENSOR_LOAD_TRANSPOSE16, 0, (uint64_t) scp_kp[buf], 0, 15, 64,
                            1);

                tensor_fma((kv_गिनती < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_slot_prev],
                           (uint64_t) (i - 1), TENSOR_FMA_OP_FP16, (i == 1));

                tensor_wait(TENSOR_LOAD_WAIT_1);  // K[i] in L1
                et_sem_post(ET_BARRIER_MINION);   // release scp_kp[buf] EARLY
                tensor_wait(TENSOR_FMA_WAIT);     // then wait FMA[i-1]
                chunk_id++;
            }

            // Epilogue: FMA on the last chunk (no overlapping load).
            {
                पूर्णाङ्क k_slot_last = (पूर्णाङ्क) ((n_dk_chunks - 1) & 1);
                tensor_fma((kv_गिनती < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_slot_last],
                           (uint64_t) (n_dk_chunks - 1), TENSOR_FMA_OP_FP16, (n_dk_chunks == 1));
                tensor_wait(TENSOR_FMA_WAIT);
            }

            // Prefetch V rows for this tile.
            // Only useful for the partial-tile path below
            यदि (kv_गिनती < TILE_KV) {
                प्रति (int64_t d = 0; d < dv; d += 32) {
                    prefetch_kv_to_l2(v_head, kv_base, d, kv_गिनती, v->nb[1]);
                }
            }

            // Extract QK^T scores from vector register file
            __asm__ चञ्चल("" ::: "f0", "f1");
            {
                unsigned long _ms;
                __asm__ चञ्चल(
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

            भिन्न weights[TILE_KV] __attribute__((aligned(64)));
            {
                // A1: apply mask to scores, pad unused slots
                प्रति (int64_t j = 0; j < kv_गिनती; ++j) {
                    यदि (has_mask) {
                        भिन्न mv = get_mask_val_from_base(mask, mask_base, kv_base + j);
                        यदि (mv == ET_NEG_INF_F || mv != mv) {
                            scores[j] = ET_NEG_INF_F;
                        } अन्यथा {
                            scores[j] += mv;
                        }
                    }
                }
                प्रति (int64_t j = kv_गिनती; j < TILE_KV; ++j) {
                    scores[j] = ET_NEG_INF_F;
                }

                // A1b: SIMD horizontal max across all 16 scores
                भिन्न tile_max;
                {
                    unsigned long _ms;
                    __asm__ चञ्चल(
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

                यदि (tile_max > ET_NEG_INF_F) {
                    // A2: rescale accumulator if this tile has a new global max
                    यदि (tile_max > M) {
                        भिन्न rescale = et_exp2f((M - tile_max) * 1.4426950408889634f);
                        scale_acc_vec(acc, dv, rescale);
                        S *= rescale;
                        M = tile_max;
                    }

                    // A3: SIMD exp2 + horizontal sum
                    // Interleaved: f2/f3 chains alternate to hide ALU latency.
                    // fexp.ps has multi-cycle latency — the two independent
                    // exp2 calls naturally pipeline.
                    {
                        ध्रुव भिन्न   log2e = 1.4426950408889634f;
                        भिन्न         S_tile;
                        unsigned long _ms;
                        __asm__ चञ्चल(
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

                        ध्रुव पूर्णाङ्क       v_full_tile = (kv_गिनती == TILE_KV);
                        ध्रुव uintptr_t v_base      = (uintptr_t) v_head + kv_base * v->nb[1];
                        ध्रुव uint64_t  nb1_v       = (uint64_t) v->nb[1];
                        uint64_t        b_cur       = 8;

                        यदि (v_full_tile) {
                            tensor_load(false, false, b_cur, TENSOR_LOAD_INTERLEAVE16, 0, (uint64_t) v_base, 0, 7,
                                        nb1_v, 1);
                        }

                        tensor_wait(TENSOR_LOAD_WAIT_0);      // weights in A_L1
                        यदि (v_full_tile) {
                            tensor_wait(TENSOR_LOAD_WAIT_1);  // V[0] in b_cur
                        }

                        // B2: process dv in chunks of 16
                        यदि (v_full_tile) {
                            प्रति (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                ध्रुव uint64_t b_nxt = b_cur ^ 24;

                                यदि (dv_off + 16 < dv) {
                                    tensor_load(false, false, b_nxt, TENSOR_LOAD_INTERLEAVE16, 0,
                                                (uint64_t) (v_base + (dv_off + 16) * 2), 0, 7, nb1_v, 1);
                                }

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, b_cur, A_L1_START,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ चञ्चल("" ::: "f0", "f1");
                                {
                                    unsigned long _ms;
                                    __asm__ चञ्चल(
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

                                यदि (dv_off + 16 < dv) {
                                    tensor_wait(TENSOR_LOAD_WAIT_1);
                                    b_cur = b_nxt;
                                }
                            }
                        } अन्यथा {
                            // Partial tile: software pack, no pipeline
                            प्रति (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                pack_v_interleaved(vpanel_buf, v_head, kv_base, dv_off, kv_गिनती, v->nb[1]);
                                FENCE;
                                flush_to_l2(vpanel_buf, 8, 64);
                                WAIT_CACHEOPS;
                                tensor_load(false, false, B_L1_START, TENSOR_LOAD_PLAIN, 0, (uint64_t) vpanel_buf, 0, 7,
                                            64, 0);
                                tensor_wait(TENSOR_LOAD_WAIT_0);

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, B_L1_START, A_L1_START,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ चञ्चल("" ::: "f0", "f1");
                                {
                                    unsigned long _ms;
                                    __asm__ चञ्चल(
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
        यदि (k_splits > 1) {
            // Publish our partial.
            चञ्चल भिन्न * my_stats = (चञ्चल भिन्न *) et_shire_l2scp_local(scp_base + SCP_STATS_OFF);
            my_stats[0]               = M;
            my_stats[1]               = S;
            FENCE;
            evict_range_to_l2(acc, (int64_t) dv * (int64_t) परिमाण(भिन्न));
            evict_to_l2((ध्रुव शून्य *) my_stats, 1, 64);
            WAIT_CACHEOPS;

            // A: team members have all written their partials.
            et_barrier(ET_BARRIER_SHIRE);

            यदि (k_split == 0) {
                // Online softmax merge: fold peers 1..k_splits-1 into our
                // own (M_running, S_running, acc). For each peer p:
                //   M_new    = max(M_running, M_p)
                //   α_own    = exp2((M_running - M_new) * log2e)
                //   α_p      = exp2((M_p     - M_new) * log2e)
                //   acc[d]   = α_own * acc[d] + α_p * peer_acc[d]
                //   S_running = α_own * S_running + α_p * S_p
                भिन्न       M_running = M;
                भिन्न       S_running = S;
                ध्रुव भिन्न log2e     = 1.4426950408889634f;

                प्रति (int64_t p = 1; p < k_splits; p++) {
                    uint64_t         peer_scp   = (local_tile_idx * k_splits + p) * SCP_PER_MINION;
                    चञ्चल भिन्न * peer_stats = (चञ्चल भिन्न *) et_shire_l2scp_local(peer_scp + SCP_STATS_OFF);
                    भिन्न *          peer_acc   = (भिन्न *) et_shire_l2scp_local(peer_scp + SCP_ACC_OFF);

                    // Drop stale L1D copies before reading peer's data.
                    evict_to_l2((ध्रुव शून्य *) peer_stats, 1, 64);
                    evict_range_to_l2(peer_acc, (int64_t) dv * (int64_t) परिमाण(भिन्न));
                    WAIT_CACHEOPS;

                    ध्रुव भिन्न M_p = peer_stats[0];
                    ध्रुव भिन्न S_p = peer_stats[1];

                    ध्रुव भिन्न M_new     = (M_p > M_running) ? M_p : M_running;
                    ध्रुव भिन्न alpha_own = (M_running == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_running - M_new) * log2e);
                    ध्रुव भिन्न alpha_p   = (M_p == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_p - M_new) * log2e);

                    merge_rescale_add_asm(acc, peer_acc, dv, alpha_own, alpha_p);

                    S_running = alpha_own * S_running + alpha_p * S_p;
                    M_running = M_new;
                }

                ध्रुव भिन्न S_inv = (S_running == 0.0f) ? 0.0f : et_fdiv(1.0f, S_running);
                normalize_store_vec(out, acc, dv, S_inv, use_fast_store);
            }

            // B: reducer is done, team may reuse its acc/stats slabs.
            et_barrier(ET_BARRIER_SHIRE);
        } अन्यथा {
            // k_splits == 1 fast path — this minion owns the full row.
            ध्रुव भिन्न S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
            normalize_store_vec(out, acc, dv, S_inv, use_fast_store);
        }
    }

    FENCE;
    प्रत्यागम 0;
}
