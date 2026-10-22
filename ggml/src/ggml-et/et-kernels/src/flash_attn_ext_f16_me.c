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

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "math_fp.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"
#ଅନ୍ତର୍ଭାଵ "tensor.h"

#ଅନ୍ତର୍ଭାଵ <etsoc/common/utils.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>
#ଅନ୍ତର୍ଭାଵ <string.h>

#define ଅଙ୍କ_COMPUTE_SHIRES 32
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

ପ୍ରକାରବ୍ୟାଖ୍ୟା uint16_t et_fp16_t;

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
#define SCP_ACC_STRIDE  (FA_DV_MAX * ଆକାର(ଦଶମିକ))       // 2048
#define SCP_KPANEL_SIZE (32 * 32 * ଆକାର(et_fp16_t))     // 2048
#define SCP_KP0_OFF     SCP_ACC_STRIDE                    // 2048
#define SCP_KP1_OFF     (SCP_KP0_OFF + SCP_KPANEL_SIZE)   // 4096
#define SCP_STATS_OFF   (SCP_KP1_OFF + SCP_KPANEL_SIZE)   // 6144
#define SCP_STATS_SIZE  64                                // own cache line
#define SCP_PER_MINION  (SCP_STATS_OFF + SCP_STATS_SIZE)  // 6208

ସଂରଚନା ggml_et_flash_attn_ext_params {
    ସଂରଚନା ggml_tensor src0;  // Q (F32)
    ସଂରଚନା ggml_tensor src1;  // K (F16)
    ସଂରଚନା ggml_tensor src2;  // V (F16)
    ସଂରଚନା ggml_tensor mask;  // mask (F16 or F32), zeroed when absent
    ସଂରଚନା ggml_tensor dst;   // Output (F32)
    ଦଶମିକ              scale;
    int32_t            has_mask;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ get_mask_val(ଅଚଳ ସଂରଚନା ggml_tensor * mask, int64_t iq1, int64_t ik1, int64_t iq2, int64_t iq3) {
    ଅଚଳ ଅକ୍ଷର * base = (ଅଚଳ ଅକ୍ଷର *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
                        (iq3 % mask->ne[3]) * mask->nb[3];

    ଯଦି (mask->type == GGML_TYPE_F32) {
        ଫେରନ୍ତୁ *(ଅଚଳ ଦଶମିକ *) (base + ik1 * mask->nb[0]);
    }
    ଫେରନ୍ତୁ fp16_to_fp32(*(ଅଚଳ uint16_t *) (base + ik1 * mask->nb[0]));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଅଚଳ ଅକ୍ଷର * get_mask_row_base(ଅଚଳ ସଂରଚନା ggml_tensor * mask, int64_t iq1, int64_t iq2, int64_t iq3) {
    ଫେରନ୍ତୁ (ଅଚଳ ଅକ୍ଷର *) mask->data + iq1 * mask->nb[1] + (iq2 % mask->ne[2]) * mask->nb[2] +
           (iq3 % mask->ne[3]) * mask->nb[3];
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ get_mask_val_from_base(ଅଚଳ ସଂରଚନା ggml_tensor * mask, ଅଚଳ ଅକ୍ଷର * base, int64_t ik1) {
    ଯଦି (mask->type == GGML_TYPE_F32) {
        ଫେରନ୍ତୁ *(ଅଚଳ ଦଶମିକ *) (base + ik1 * mask->nb[0]);
    }
    ଫେରନ୍ତୁ fp16_to_fp32(*(ଅଚଳ uint16_t *) (base + ik1 * mask->nb[0]));
}

// Pack K rows for TensorLoadTranspose16 (even/odd deinterleave)
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ __attribute__((always_inline)) pack_k_for_transpose16(et_fp16_t *  out,
                                                                         ଅଚଳ ଅକ୍ଷର * k_base,
                                                                         int64_t      kv_start,
                                                                         int64_t      dk_start,
                                                                         int64_t      kv_ଗଣନା,
                                                                         int64_t      nb1_k) {
    unsigned long old_mask;
    __asm__ ଅସ୍ଥିର(
        "mova.x.m  %[ms]            \n\t"
        "mov.m.x   m0, x0, 0xFF     \n\t"
        : [ms] "=&r"(old_mask)
        :
        :);

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 0; j < (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kv_ଗଣନା; ++j) {
        ଅଚଳ et_fp16_t * k_row    = (ଅଚଳ et_fp16_t *) (k_base + (kv_start + j) * nb1_k) + dk_start;
        et_fp16_t *       even_row = out + (j * 2) * 32;
        et_fp16_t *       odd_row  = out + (j * 2 + 1) * 32;
        __asm__ ଅସ୍ଥିର(
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

    __asm__ ଅସ୍ଥିର("mova.m.x  %[ms]            \n\t" : : [ms] "r"(old_mask));

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kv_ଗଣନା; j < TILE_KV; ++j) {
        et_fp16_t * even_row = out + (j * 2) * 32;
        et_fp16_t * odd_row  = out + (j * 2 + 1) * 32;
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା l = 0; l < TILE_K / 2; ++l) {
            even_row[l] = 0;
            odd_row[l]  = 0;
        }
    }
}

// Build interleaved B panel for TensorFMA16A32 (weights @ V).
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ __attribute__((always_inline)) pack_v_interleaved(et_fp16_t *  out,
                                                                     ଅଚଳ ଅକ୍ଷର * v_head,
                                                                     int64_t      kv_base,
                                                                     int64_t      dv_start,
                                                                     int64_t      kv_ଗଣନା,
                                                                     int64_t      nb1_v) {
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା k = 0; k < TILE_KV; ++k) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା         l   = k >> 1;
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା         r   = k & 1;
        et_fp16_t * ଅଚଳ dst = out + l * 32 + r;
        ଯଦି (k < (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) kv_ଗଣନା) {
            ଅଚଳ et_fp16_t * v_row = (ଅଚଳ et_fp16_t *) (v_head + (kv_base + k) * nb1_v) + dv_start;
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = 0; n < 16; ++n) {
                dst[n * 2] = v_row[n];
            }
        } ଅନ୍ୟଥା {
            ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା n = 0; n < 16; ++n) {
                dst[n * 2] = 0;
            }
        }
    }
}

// Prefetch KV rows for one chunk into L2.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ __attribute__((always_inline)) prefetch_kv_to_l2(ଅଚଳ ଅକ୍ଷର * head,
                                                                    int64_t      kv_start,
                                                                    int64_t      d_start,
                                                                    int64_t      kv_ଗଣନା,
                                                                    int64_t      nb1) {
    ଅଚଳ ଶୂନ୍ୟ * base = (ଅଚଳ ଶୂନ୍ୟ *) (head + kv_start * nb1 + d_start * 2);
    l2_prefetch(base, (uint64_t) kv_ଗଣନା, (uint64_t) nb1);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ __attribute__((always_inline)) convert_q_row_f32_to_f16(et_fp16_t *   dst,
                                                                           ଅଚଳ ଦଶମିକ * src,
                                                                           int64_t       n) {
    ସ୍ଥିର ଅଚଳ int32_t __attribute__((aligned(32))) offsets[8] = { 0, 2, 4, 6, 8, 10, 12, 14 };

    unsigned long old_mask;
    __asm__ ଅସ୍ଥିର(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 0xFF      \n\t"
        "flw.ps    f1, 0(%[offs])    \n\t"
        : [ms] "=&r"(old_mask)
        : [offs] "r"(offsets)
        : "f1");

    ପାଇଁ (int64_t d = 0; d < n; d += 8) {
        __asm__ ଅସ୍ଥିର(
            "flw.ps      f2, 0(%[src])    \n\t"
            "fcvt.f16.ps f3, f2           \n\t"
            "fsch.ps     f3, f1(%[dst])   \n\t"
            :
            : [src] "r"(src + d), [dst] "r"(dst + d)
            : "f2", "f3", "memory");
    }

    __asm__ ଅସ୍ଥିର("mova.m.x  %[ms]             \n\t" : : [ms] "r"(old_mask));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ __attribute__((always_inline)) zero_acc_vec(ଦଶମିକ * acc, int64_t dv) {
    ଅଚଳ ଦଶମିକ   zero = 0.0f;
    unsigned long old_mask;
    __asm__ ଅସ୍ଥିର("mova.x.m %0" : "=r"(old_mask));
    __asm__ ଅସ୍ଥିର("mov.m.x m0, x0, 0xFF");
    __asm__ ଅସ୍ଥିର("fbc.ps  f2, 0(%[z])" ::[z] "r"(&zero) : "f2");

    ପାଇଁ (int64_t d = 0; d < dv; d += 8) {
        __asm__ ଅସ୍ଥିର("fsw.ps  f2, 0(%[a])     \n\t" ::[a] "r"(acc + d) : "f2", "memory");
    }

    __asm__ ଅସ୍ଥିର("mova.m.x %0" ::"r"(old_mask));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ __attribute__((always_inline)) scale_acc_vec(ଦଶମିକ * acc, int64_t dv, ଦଶମିକ scale) {
    unsigned long old_mask;
    __asm__ ଅସ୍ଥିର("mova.x.m %0" : "=r"(old_mask));
    __asm__ ଅସ୍ଥିର("mov.m.x m0, x0, 0xFF");

    ପାଇଁ (int64_t d = 0; d < dv; d += 8) {
        __asm__ ଅସ୍ଥିର(
            "fbc.ps    f2, 0(%[s])    \n\t"
            "flw.ps    f3, 0(%[a])    \n\t"
            "fmul.ps   f3, f3, f2     \n\t"
            "fsw.ps    f3, 0(%[a])    \n\t"
            :
            : [s] "r"(&scale), [a] "r"(acc + d)
            : "f2", "f3", "memory");
    }

    __asm__ ଅସ୍ଥିର("mova.m.x %0" ::"r"(old_mask));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ __attribute__((always_inline)) normalize_store_vec(ଦଶମିକ * out,
                                                                      ଦଶମିକ * acc,
                                                                      int64_t dv,
                                                                      ଦଶମିକ   inv,
                                                                      ପୂର୍ଣ୍ଣସଂଖ୍ୟା     use_fast_store) {
    unsigned long old_mask;
    __asm__ ଅସ୍ଥିର("mova.x.m %0" : "=r"(old_mask));
    __asm__ ଅସ୍ଥିର("mov.m.x m0, x0, 0xFF");

    ପାଇଁ (int64_t d = 0; d < dv; d += 8) {
        __asm__ ଅସ୍ଥିର(
            "fbc.ps    f2, 0(%[inv])   \n\t"
            "flw.ps    f3, 0(%[a])     \n\t"
            "fmul.ps   f3, f3, f2      \n\t"
            "fsw.ps    f3, 0(%[a])     \n\t"
            :
            : [inv] "r"(&inv), [a] "r"(acc + d)
            : "f2", "f3", "memory");
        ଯଦି (use_fast_store) {
            __asm__ ଅସ୍ଥିର(
                "flw.ps  f4, 0(%[a])     \n\t"
                "fsw.ps  f4, 0(%[o])     \n\t"
                :
                : [a] "r"(acc + d), [o] "r"(out + d)
                : "f4", "memory");
        } ଅନ୍ୟଥା {
            atomic_store_f32((ଅସ୍ଥିର ଦଶମିକ *) &out[d + 0], acc[d + 0]);
            atomic_store_f32((ଅସ୍ଥିର ଦଶମିକ *) &out[d + 1], acc[d + 1]);
            atomic_store_f32((ଅସ୍ଥିର ଦଶମିକ *) &out[d + 2], acc[d + 2]);
            atomic_store_f32((ଅସ୍ଥିର ଦଶମିକ *) &out[d + 3], acc[d + 3]);
            atomic_store_f32((ଅସ୍ଥିର ଦଶମିକ *) &out[d + 4], acc[d + 4]);
            atomic_store_f32((ଅସ୍ଥିର ଦଶମିକ *) &out[d + 5], acc[d + 5]);
            atomic_store_f32((ଅସ୍ଥିର ଦଶମିକ *) &out[d + 6], acc[d + 6]);
            atomic_store_f32((ଅସ୍ଥିର ଦଶମିକ *) &out[d + 7], acc[d + 7]);
        }
    }

    __asm__ ଅସ୍ଥିର("mova.m.x %0" ::"r"(old_mask));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t tensor_bytes_fa(ଅଚଳ ସଂରଚନା ggml_tensor * t) {
    ଫେରନ୍ତୁ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

// Evict a byte range from L1D to L2 SCP, splitting into batches of ≤16
// cache lines (the hw limit for evict_to_l2). Use before a barrier when
// another minion in the shire needs to read the region, or after a barrier
// on the reader side to drop stale L1D copies before reading peer data.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ __attribute__((always_inline)) evict_range_to_l2(ଅଚଳ ଶୂନ୍ୟ * addr, int64_t bytes) {
    ଯଦି (bytes <= 0) {
        ଫେରନ୍ତୁ;
    }
    int64_t      ଧାଡ଼ିଗୁଡ଼ିକ = (bytes + 63) / 64;
    ଅଚଳ ଅକ୍ଷର * p     = (ଅଚଳ ଅକ୍ଷର *) addr;
    ଯେତେବେଳେ (ଧାଡ଼ିଗୁଡ଼ିକ > 0) {
        int64_t batch = ଧାଡ଼ିଗୁଡ଼ିକ > 16 ? 16 : ଧାଡ଼ିଗୁଡ଼ିକ;
        evict_to_l2((ଅଚଳ ଶୂନ୍ୟ *) p, (uint64_t) batch, 64);
        p += batch * 64;
        ଧାଡ଼ିଗୁଡ଼ିକ -= batch;
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
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ __attribute__((always_inline)) merge_rescale_add_asm(ଦଶମିକ *       acc,
                                                                        ଅଚଳ ଦଶମିକ * peer_acc,
                                                                        int64_t       dv,
                                                                        ଦଶମିକ         alpha_own,
                                                                        ଦଶମିକ         alpha_peer) {
    unsigned long old_mask;
    __asm__ ଅସ୍ଥିର(
        "mova.x.m  %[ms]              \n\t"
        "mov.m.x   m0, x0, 0xFF       \n\t"
        "fbc.ps    f4, 0(%[ao])       \n\t"  // broadcast alpha_own
        "fbc.ps    f5, 0(%[ap])       \n\t"  // broadcast alpha_peer
        : [ms] "=&r"(old_mask)
        : [ao] "r"(&alpha_own), [ap] "r"(&alpha_peer)
        : "f4", "f5");

    ପାଇଁ (int64_t d = 0; d < dv; d += 8) {
        __asm__ ଅସ୍ଥିର(
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

    __asm__ ଅସ୍ଥିର("mova.m.x %0" ::"r"(old_mask));
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_flash_attn_ext_params * params, ଶୂନ୍ୟ * env) {
    (ଶୂନ୍ୟ) env;

    uint64_t hart_id  = get_hart_id();
    uint64_t shire_id = get_shire_id();

    ଯଦି (shire_id >= ଅଙ୍କ_COMPUTE_SHIRES) {
        ଫେରନ୍ତୁ 0;
    }

    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା is_hart1     = hart_id & 1;
    uint64_t  local_minion = (hart_id >> 1) & 0x1F;

    ସଂରଚନା ggml_tensor * q        = &params->src0;
    ସଂରଚନା ggml_tensor * k        = &params->src1;
    ସଂରଚନା ggml_tensor * v        = &params->src2;
    ସଂରଚନା ggml_tensor * dst      = &params->dst;
    ଅଚଳ int32_t        has_mask = params->has_mask;
    ସଂରଚନା ggml_tensor * mask     = has_mask ? &params->mask : (ସଂରଚନା ggml_tensor *) 0;

    ଅଚଳ ଅକ୍ଷର * q_data   = (ଅଚଳ ଅକ୍ଷର *) q->data;
    ଅଚଳ ଅକ୍ଷର * k_data   = (ଅଚଳ ଅକ୍ଷର *) k->data;
    ଅଚଳ ଅକ୍ଷର * v_data   = (ଅଚଳ ଅକ୍ଷର *) v->data;
    ଅକ୍ଷର *       dst_data = (ଅକ୍ଷର *) dst->data;

    // et_barrier(ET_BARRIER_GLOBAL);
    evict_region_past_l2(q->data, tensor_bytes_fa(q));
    evict_region_past_l2(k->data, tensor_bytes_fa(k));
    evict_region_past_l2(v->data, tensor_bytes_fa(v));
    ଯଦି (mask) {
        evict_region_past_l2(mask->data, tensor_bytes_fa(mask));
    }
    et_barrier(ET_BARRIER_GLOBAL);

    ଅଚଳ int64_t dk  = q->ne[0];
    ଅଚଳ int64_t nq  = q->ne[1];
    ଅଚଳ int64_t nhq = q->ne[2];
    ଅଚଳ int64_t no  = q->ne[3];
    ଅଚଳ int64_t nk  = k->ne[1];
    ଅଚଳ int64_t nhk = k->ne[2];
    ଅଚଳ int64_t dv  = v->ne[0];

    ଯଦି (dv > FA_DV_MAX || dk > FA_DK_MAX) {
        ଫେରନ୍ତୁ -1;
    }
    ଯଦି (k->nb[0] != 2 || v->nb[0] != 2) {
        ଫେରନ୍ତୁ -1;
    }
    ଯଦି ((dk % 8) != 0 || (dv % 16) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ int64_t gqa_ratio      = nhq / nhk;
    ଅଚଳ int64_t total_rows     = nq * nhq * no;
    ଅଚଳ ଦଶମିକ   scale          = params->scale;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     use_fast_store = (dv % 16 == 0);

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
    ଅଚଳ int64_t nk_tiles      = (nk + TILE_KV - 1) / TILE_KV;
    ଅଚଳ int64_t total_minions = 2 * ଅଙ୍କ_COMPUTE_SHIRES * MINIONS_PER_SHIRE;
    int64_t       k_splits      = 1;
    ଯଦି (total_rows < total_minions) {
        int64_t target = total_minions / total_rows;
        int64_t ks     = 1;
        ଯେତେବେଳେ (ks * 2 <= target && ks * 2 <= MINIONS_PER_SHIRE && ks * 2 <= nk_tiles) {
            ks *= 2;
        }
        k_splits = ks;
    }

    ଅଚଳ int64_t tiles_per_shire = MINIONS_PER_SHIRE / k_splits;
    ଅଚଳ int64_t k_split         = (int64_t) local_minion % k_splits;
    ଅଚଳ int64_t local_tile_idx  = (int64_t) local_minion / k_splits;
    ଅଚଳ int64_t tiles_stride    = (int64_t) ଅଙ୍କ_COMPUTE_SHIRES * tiles_per_shire;

    // KV slab for this k_split. With k_splits=1 this is the full range.
    ଅଚଳ int64_t tiles_per_split_rounded = (nk_tiles + k_splits - 1) / k_splits;
    ଅଚଳ int64_t tile_start              = k_split * tiles_per_split_rounded;
    int64_t       tile_end                = tile_start + tiles_per_split_rounded;
    ଯଦି (tile_end > nk_tiles) {
        tile_end = nk_tiles;
    }
    ଅଚଳ int64_t kv_start = tile_start * TILE_KV;
    int64_t       kv_end   = tile_end * TILE_KV;
    ଯଦି (kv_end > nk) {
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
    ଯଦି (is_hart1) {
        uint32_t      chunk_id = 0;
        ଅଚଳ int64_t row_base = (int64_t) shire_id + local_tile_idx * ଅଙ୍କ_COMPUTE_SHIRES;

        int64_t max_iters;
        ଯଦି (k_splits > 1) {
            max_iters = (total_rows + tiles_stride - 1) / tiles_stride;
        } ଅନ୍ୟଥା {
            max_iters = (row_base >= total_rows) ? 0 : ((total_rows - row_base - 1) / tiles_stride + 1);
        }

        ପାଇଁ (int64_t iter = 0; iter < max_iters; iter++) {
            ଅଚଳ int64_t row      = row_base + iter * tiles_stride;
            ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     has_work = (row < total_rows);

            ଯଦି (has_work) {
                ଅଚଳ int64_t iq3 = row / (nhq * nq);
                ଅଚଳ int64_t rem = row % (nhq * nq);
                ଅଚଳ int64_t iq2 = rem / nq;
                ଅଚଳ int64_t ik2 = iq2 / gqa_ratio;

                ଅଚଳ ଅକ୍ଷର * k_head = k_data + ik2 * k->nb[2] + iq3 * k->nb[3];

                ପାଇଁ (int64_t kv_base = kv_start; kv_base < kv_end; kv_base += TILE_KV) {
                    ଅଚଳ int64_t kv_ଗଣନା = (kv_base + TILE_KV <= nk) ? TILE_KV : (nk - kv_base);

                    ପାଇଁ (int64_t dk_chunk = 0; dk_chunk < dk; dk_chunk += TILE_K) {
                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf = chunk_id & 1;

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
                        ଯଦି (chunk_id >= 2) {
                            et_sem_wait(ET_BARRIER_MINION);
                        }

                        // Prefetch K data for this chunk
                        prefetch_kv_to_l2(k_head, kv_base, dk_chunk, kv_ଗଣନା, k->nb[1]);

                        pack_k_for_transpose16(scp_kp[buf], k_head, kv_base, dk_chunk, kv_ଗଣନା, k->nb[1]);

                        FENCE;
                        flush_to_l2(scp_kp[buf], 16, 64);
                        flush_to_l2((et_fp16_t *) ((ଅକ୍ଷର *) scp_kp[buf] + 1024), 16, 64);
                        WAIT_CACHEOPS;

                        // Signal: this buf is ready for hart 0 to consume.
                        et_sem_post(ET_BARRIER_MINION);

                        chunk_id++;
                    }
                }
            }

            // Shire barriers for split-KV merge (hart 1 is a passive arrival).
            ଯଦି (k_splits > 1) {
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
        ପାଇଁ (uint32_t d = 0; d < drain; d++) {
            et_sem_wait(ET_BARRIER_MINION);
        }

        // FENCE;
        // et_barrier(ET_BARRIER_GLOBAL);
        ଫେରନ୍ତୁ 0;
    }

    // Hart 0: tensor engine compute
#ifndef UBERKERNEL_SUPPRESS_SCP_SETUP
    setup_cache_scp();
#endif
    CLEAR_TENSOR_ERROR;

    // Q converted to F16 (one row at a time)
    et_fp16_t q_f16[FA_DK_MAX] __attribute__((aligned(64)));

    // Score buffer for QK^T output (16 scores per KV tile)
    ଦଶମିକ scores[TILE_KV] __attribute__((aligned(64)));

    // Small buffers for V accumulation
    et_fp16_t w_f16_buf[32] __attribute__((aligned(64)));       // 64 bytes
    et_fp16_t vpanel_buf[8 * 32] __attribute__((aligned(64)));  // 512 bytes

    ଦଶମିକ * acc = (ଦଶମିକ *) et_shire_l2scp_local(scp_base + SCP_ACC_OFF);

    uint32_t chunk_id = 0;

    // Iter-based outer loop (matches hart 1). When k_splits > 1 all teams
    // in a shire iterate the same number of times so the per-row shire
    // barriers stay balanced; iterations with row >= total_rows skip the
    // compute but still participate in the barriers.
    ଅଚଳ int64_t hart0_row_base = (int64_t) shire_id + local_tile_idx * ଅଙ୍କ_COMPUTE_SHIRES;
    int64_t       hart0_max_iters;
    ଯଦି (k_splits > 1) {
        hart0_max_iters = (total_rows + tiles_stride - 1) / tiles_stride;
    } ଅନ୍ୟଥା {
        hart0_max_iters = (hart0_row_base >= total_rows) ? 0 : ((total_rows - hart0_row_base - 1) / tiles_stride + 1);
    }

    ପାଇଁ (int64_t iter = 0; iter < hart0_max_iters; iter++) {
        ଅଚଳ int64_t row = hart0_row_base + iter * tiles_stride;
        ଯଦି (row >= total_rows) {
            // No-work iteration: only participate in barriers (k_splits > 1).
            ଯଦି (k_splits > 1) {
                et_barrier(ET_BARRIER_SHIRE);  // A
                et_barrier(ET_BARRIER_SHIRE);  // B
            }
            ଜାରି;
        }

        ଅଚଳ int64_t iq3 = row / (nhq * nq);
        ଅଚଳ int64_t rem = row % (nhq * nq);
        ଅଚଳ int64_t iq2 = rem / nq;
        ଅଚଳ int64_t iq1 = rem % nq;
        ଅଚଳ int64_t ik2 = iq2 / gqa_ratio;

        // Read Q row (F32) and convert to F16
        ଅଚଳ ଦଶମିକ * pq = (ଅଚଳ ଦଶମିକ *) (q_data + iq1 * q->nb[1] + iq2 * q->nb[2] + iq3 * q->nb[3]);
        convert_q_row_f32_to_f16(q_f16, pq, dk);

        // V base for this head + batch (K packing handled by hart 1)
        ଅଚଳ ଅକ୍ଷର * v_head = v_data + ik2 * v->nb[2] + iq3 * v->nb[3];

        // Output pointer
        ଦଶମିକ * out = (ଦଶମିକ *) (dst_data + iq2 * dst->nb[1] + iq1 * dst->nb[2] + iq3 * dst->nb[3]);

        zero_acc_vec(acc, dv);
        ଦଶମିକ        M         = ET_NEG_INF_F;
        ଦଶମିକ        S         = 0.0f;
        ଅଚଳ ଅକ୍ଷର * mask_base = has_mask ? get_mask_row_base(mask, iq1, iq2, iq3) : (ଅଚଳ ଅକ୍ଷର *) 0;

        // Flush Q_f16 to L2 so tensor_load can see it
        FENCE;
        flush_to_l2(q_f16, (dk * 2 + 63) / 64, 64);
        WAIT_CACHEOPS;

        ପାଇଁ (int64_t kv_base = kv_start; kv_base < kv_end; kv_base += TILE_KV) {
            ଅଚଳ int64_t kv_ଗଣନା = (kv_base + TILE_KV <= nk) ? TILE_KV : (nk - kv_base);

            // Set tensor_mask for partial tiles
            ଯଦି (kv_ଗଣନା < TILE_KV) {
                uint64_t tmask = (1ULL << kv_ଗଣନା) - 1;
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
            ଅଚଳ int64_t  n_dk_chunks = dk / TILE_K;
            ଅଚଳ uint64_t K_BUFS[2]   = {
                (uint64_t) B_L1_START,         // 16..31
                (uint64_t) (B_L1_START + 16),  // 32..47
            };

            // Preload entire Q row into A_L1[0..n_dk_chunks-1] (one tensor_load,
            // one wait, regardless of dk).
            tensor_load(false, false, A_L1_START, TENSOR_LOAD_PLAIN, 0, (uint64_t) q_f16, 0,
                        (uint64_t) (n_dk_chunks - 1), 64, 0);

            // Prologue: wait hart 1's K[0], issue K[0] load, wait both loads.
            {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf = chunk_id & 1;
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
            ପାଇଁ (int64_t i = 1; i < n_dk_chunks; i++) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା buf         = chunk_id & 1;
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_slot_prev = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ((i - 1) & 1);
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_slot      = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (i & 1);

                et_sem_wait(ET_BARRIER_MINION);
                tensor_load(false, false, K_BUFS[k_slot], TENSOR_LOAD_TRANSPOSE16, 0, (uint64_t) scp_kp[buf], 0, 15, 64,
                            1);

                tensor_fma((kv_ଗଣନା < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_slot_prev],
                           (uint64_t) (i - 1), TENSOR_FMA_OP_FP16, (i == 1));

                tensor_wait(TENSOR_LOAD_WAIT_1);  // K[i] in L1
                et_sem_post(ET_BARRIER_MINION);   // release scp_kp[buf] EARLY
                tensor_wait(TENSOR_FMA_WAIT);     // then wait FMA[i-1]
                chunk_id++;
            }

            // Epilogue: FMA on the last chunk (no overlapping load).
            {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା k_slot_last = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) ((n_dk_chunks - 1) & 1);
                tensor_fma((kv_ଗଣନା < TILE_KV), 3, 0, 15, 0, false, false, false, false, K_BUFS[k_slot_last],
                           (uint64_t) (n_dk_chunks - 1), TENSOR_FMA_OP_FP16, (n_dk_chunks == 1));
                tensor_wait(TENSOR_FMA_WAIT);
            }

            // Prefetch V rows for this tile.
            // Only useful for the partial-tile path below
            ଯଦି (kv_ଗଣନା < TILE_KV) {
                ପାଇଁ (int64_t d = 0; d < dv; d += 32) {
                    prefetch_kv_to_l2(v_head, kv_base, d, kv_ଗଣନା, v->nb[1]);
                }
            }

            // Extract QK^T scores from vector register file
            __asm__ ଅସ୍ଥିର("" ::: "f0", "f1");
            {
                unsigned long _ms;
                __asm__ ଅସ୍ଥିର(
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

            ଦଶମିକ weights[TILE_KV] __attribute__((aligned(64)));
            {
                // A1: apply mask to scores, pad unused slots
                ପାଇଁ (int64_t j = 0; j < kv_ଗଣନା; ++j) {
                    ଯଦି (has_mask) {
                        ଦଶମିକ mv = get_mask_val_from_base(mask, mask_base, kv_base + j);
                        ଯଦି (mv == ET_NEG_INF_F || mv != mv) {
                            scores[j] = ET_NEG_INF_F;
                        } ଅନ୍ୟଥା {
                            scores[j] += mv;
                        }
                    }
                }
                ପାଇଁ (int64_t j = kv_ଗଣନା; j < TILE_KV; ++j) {
                    scores[j] = ET_NEG_INF_F;
                }

                // A1b: SIMD horizontal max across all 16 scores
                ଦଶମିକ tile_max;
                {
                    unsigned long _ms;
                    __asm__ ଅସ୍ଥିର(
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

                ଯଦି (tile_max > ET_NEG_INF_F) {
                    // A2: rescale accumulator if this tile has a new global max
                    ଯଦି (tile_max > M) {
                        ଦଶମିକ rescale = et_exp2f((M - tile_max) * 1.4426950408889634f);
                        scale_acc_vec(acc, dv, rescale);
                        S *= rescale;
                        M = tile_max;
                    }

                    // A3: SIMD exp2 + horizontal sum
                    // Interleaved: f2/f3 chains alternate to hide ALU latency.
                    // fexp.ps has multi-cycle latency — the two independent
                    // exp2 calls naturally pipeline.
                    {
                        ଅଚଳ ଦଶମିକ   log2e = 1.4426950408889634f;
                        ଦଶମିକ         S_tile;
                        unsigned long _ms;
                        __asm__ ଅସ୍ଥିର(
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

                        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା       v_full_tile = (kv_ଗଣନା == TILE_KV);
                        ଅଚଳ uintptr_t v_base      = (uintptr_t) v_head + kv_base * v->nb[1];
                        ଅଚଳ uint64_t  nb1_v       = (uint64_t) v->nb[1];
                        uint64_t        b_cur       = 8;

                        ଯଦି (v_full_tile) {
                            tensor_load(false, false, b_cur, TENSOR_LOAD_INTERLEAVE16, 0, (uint64_t) v_base, 0, 7,
                                        nb1_v, 1);
                        }

                        tensor_wait(TENSOR_LOAD_WAIT_0);      // weights in A_L1
                        ଯଦି (v_full_tile) {
                            tensor_wait(TENSOR_LOAD_WAIT_1);  // V[0] in b_cur
                        }

                        // B2: process dv in chunks of 16
                        ଯଦି (v_full_tile) {
                            ପାଇଁ (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                ଅଚଳ uint64_t b_nxt = b_cur ^ 24;

                                ଯଦି (dv_off + 16 < dv) {
                                    tensor_load(false, false, b_nxt, TENSOR_LOAD_INTERLEAVE16, 0,
                                                (uint64_t) (v_base + (dv_off + 16) * 2), 0, 7, nb1_v, 1);
                                }

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, b_cur, A_L1_START,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ ଅସ୍ଥିର("" ::: "f0", "f1");
                                {
                                    unsigned long _ms;
                                    __asm__ ଅସ୍ଥିର(
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

                                ଯଦି (dv_off + 16 < dv) {
                                    tensor_wait(TENSOR_LOAD_WAIT_1);
                                    b_cur = b_nxt;
                                }
                            }
                        } ଅନ୍ୟଥା {
                            // Partial tile: software pack, no pipeline
                            ପାଇଁ (int64_t dv_off = 0; dv_off < dv; dv_off += 16) {
                                pack_v_interleaved(vpanel_buf, v_head, kv_base, dv_off, kv_ଗଣନା, v->nb[1]);
                                FENCE;
                                flush_to_l2(vpanel_buf, 8, 64);
                                WAIT_CACHEOPS;
                                tensor_load(false, false, B_L1_START, TENSOR_LOAD_PLAIN, 0, (uint64_t) vpanel_buf, 0, 7,
                                            64, 0);
                                tensor_wait(TENSOR_LOAD_WAIT_0);

                                tensor_fma(false, 3, 0, 7, 0, false, false, false, false, B_L1_START, A_L1_START,
                                           TENSOR_FMA_OP_FP16, true);
                                tensor_wait(TENSOR_FMA_WAIT);

                                __asm__ ଅସ୍ଥିର("" ::: "f0", "f1");
                                {
                                    unsigned long _ms;
                                    __asm__ ଅସ୍ଥିର(
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
        ଯଦି (k_splits > 1) {
            // Publish our partial.
            ଅସ୍ଥିର ଦଶମିକ * my_stats = (ଅସ୍ଥିର ଦଶମିକ *) et_shire_l2scp_local(scp_base + SCP_STATS_OFF);
            my_stats[0]               = M;
            my_stats[1]               = S;
            FENCE;
            evict_range_to_l2(acc, (int64_t) dv * (int64_t) ଆକାର(ଦଶମିକ));
            evict_to_l2((ଅଚଳ ଶୂନ୍ୟ *) my_stats, 1, 64);
            WAIT_CACHEOPS;

            // A: team members have all written their partials.
            et_barrier(ET_BARRIER_SHIRE);

            ଯଦି (k_split == 0) {
                // Online softmax merge: fold peers 1..k_splits-1 into our
                // own (M_running, S_running, acc). For each peer p:
                //   M_new    = max(M_running, M_p)
                //   α_own    = exp2((M_running - M_new) * log2e)
                //   α_p      = exp2((M_p     - M_new) * log2e)
                //   acc[d]   = α_own * acc[d] + α_p * peer_acc[d]
                //   S_running = α_own * S_running + α_p * S_p
                ଦଶମିକ       M_running = M;
                ଦଶମିକ       S_running = S;
                ଅଚଳ ଦଶମିକ log2e     = 1.4426950408889634f;

                ପାଇଁ (int64_t p = 1; p < k_splits; p++) {
                    uint64_t         peer_scp   = (local_tile_idx * k_splits + p) * SCP_PER_MINION;
                    ଅସ୍ଥିର ଦଶମିକ * peer_stats = (ଅସ୍ଥିର ଦଶମିକ *) et_shire_l2scp_local(peer_scp + SCP_STATS_OFF);
                    ଦଶମିକ *          peer_acc   = (ଦଶମିକ *) et_shire_l2scp_local(peer_scp + SCP_ACC_OFF);

                    // Drop stale L1D copies before reading peer's data.
                    evict_to_l2((ଅଚଳ ଶୂନ୍ୟ *) peer_stats, 1, 64);
                    evict_range_to_l2(peer_acc, (int64_t) dv * (int64_t) ଆକାର(ଦଶମିକ));
                    WAIT_CACHEOPS;

                    ଅଚଳ ଦଶମିକ M_p = peer_stats[0];
                    ଅଚଳ ଦଶମିକ S_p = peer_stats[1];

                    ଅଚଳ ଦଶମିକ M_new     = (M_p > M_running) ? M_p : M_running;
                    ଅଚଳ ଦଶମିକ alpha_own = (M_running == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_running - M_new) * log2e);
                    ଅଚଳ ଦଶମିକ alpha_p   = (M_p == ET_NEG_INF_F) ? 0.0f : et_exp2f((M_p - M_new) * log2e);

                    merge_rescale_add_asm(acc, peer_acc, dv, alpha_own, alpha_p);

                    S_running = alpha_own * S_running + alpha_p * S_p;
                    M_running = M_new;
                }

                ଅଚଳ ଦଶମିକ S_inv = (S_running == 0.0f) ? 0.0f : et_fdiv(1.0f, S_running);
                normalize_store_vec(out, acc, dv, S_inv, use_fast_store);
            }

            // B: reducer is done, team may reuse its acc/stats slabs.
            et_barrier(ET_BARRIER_SHIRE);
        } ଅନ୍ୟଥା {
            // k_splits == 1 fast path — this minion owns the full row.
            ଅଚଳ ଦଶମିକ S_inv = S == 0.0f ? 0.0f : et_fdiv(1.0f, S);
            normalize_store_vec(out, acc, dv, S_inv, use_fast_store);
        }
    }

    FENCE;
    ଫେରନ୍ତୁ 0;
}
