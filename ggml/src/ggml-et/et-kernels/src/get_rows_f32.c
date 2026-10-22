//******************************************************************************
// Bare Metal GET_ROWS F32 Kernel
// Extracts specific rows from a source tensor based on row indices
//
// Algorithm:
// 1. Read row indices from src1 (int32 tensor)
// 2. For each index, extract the corresponding row from src0
// 3. Copy the row data to the output tensor dst
// 4. Handle different input types: F32, Q8_0, Q4_0, and Q4_K (quantized)
//
// Operation: dst[i] = src0[indices[i]] for i = 0..num_indices
//
// Features supported:
// - F32 input data (direct copy)
// - Q4_0 quantized input data (dequantized to F32)
// - Q8_0 quantized input data (dequantized to F32)
// - Q4_K quantized input data (dequantized to F32)
// - Int32 row indices
// - Multi-dimensional tensor support
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "platform.h"
#ضمن "quants.h"

#ضمن <assert.h>
#ضمن <stdbool.h>
#ضمن <stdint.h>

#وصف_ڪريو CACHE_سٽ_سائيز_بائيٽون 64

بنية ggml_et_get_صفون_params {
    بنية ggml_tensor src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    بنية ggml_tensor src1;  // Row indices tensor (I32)
    بنية ggml_tensor dst;   // Output tensor (F32)
};

#وصف_ڪريو CACHE_سٽ_سائيز_بائيٽون     64
#وصف_ڪريو CACHE_ELEMENTS(elem_سائيز) (CACHE_سٽ_سائيز_بائيٽون / (elem_سائيز))

// Copy a row of F32 data from source to destination
ساكن فراغ copy_f32_صف(عائم * dst, ثابت عائم * src, int64_t انگ_elements) {
    // Simple memcpy for F32 data - no conversion needed
    لكل (int64_t i = 0; i < انگ_elements; i++) {
        dst[i] = src[i];
    }
}

ساكن فراغ copy_f16_صف(عائم * dst, ثابت uint16_t * src, int64_t انگ_elements) {
    لكل (int64_t i = 0; i < انگ_elements; i++) {
        dst[i] = fp16_منزل_fp32(src[i]);
    }
}

// Copy a row of F32 data from source to destination, aligned to cache line boundaries
// using FP32 load/store instructions. They don't perform data conversion so is fine.
// Requirement: n_bytes is a multiple of CACHE_LINE_SIZE (64 bytes)
ساكن فراغ copy_صف_cache_align(عائم * dst, ثابت عائم * src, int64_t n_بائيٽون) {
    صحيح انگ_f32_elem = n_بائيٽون / حجم(عائم);

    // Unrolled to do an entire cache line at a time
    __asm__ متطاير(
        "1: \n\t"
        // --- Process 64 Bytes (1 Cache Line) ---
        // Load 256 bits (32 bytes) into f0 and the other into f1
        "flq2 f0, 0(%[src]) \n\t"
        "flq2 f1, 32(%[src]) \n\t"

        // Store 256 bits (32 bytes) from f0 and f1
        "fsq2 f0, 0(%[dst]) \n\t"
        "fsq2 f1, 32(%[dst]) \n\t"

        // Increment Pointers by 64 bytes
        "addi %[src], %[src], 64 \n\t"
        "addi %[dst], %[dst], 64 \n\t"

        // Decrement count by 16 elements
        "addi %[n], %[n], -16 \n\t"

        // Loop if at least 16 elements remain
        "bge %[n], %[stride_count], 1b \n\t"

        : [dst] "+r"(dst), [src] "+r"(src), [n] "+r"(انگ_f32_elem)
        : [stride_ڳڻپ] "r"(16L)
        : "f0", "f1", "memory");
}

// Copied from GGML: copy a row of Q4_0 data to F32 destination (with dequantization)
ساكن فراغ copy_q4_0_صف(عائم * dst, ثابت بلاڪ_q4_0 * src_بلاڪون, int64_t انگ_elements) {
    ثابت int64_t انگ_بلاڪون = (انگ_elements + QK4_0 - 1) / QK4_0;

    لكل (int64_t بلاڪ_idx = 0; بلاڪ_idx < انگ_بلاڪون; بلاڪ_idx++) {
        ثابت int64_t elements_اندريون_بلاڪ = (بلاڪ_idx == انگ_بلاڪون - 1) ? (انگ_elements - بلاڪ_idx * QK4_0) : QK4_0;

        عائم عارضي_بفر[QK4_0];
        dequantize_q4_0_بلاڪ(&src_بلاڪون[بلاڪ_idx], عارضي_بفر);

        لكل (int64_t i = 0; i < elements_اندريون_بلاڪ; i++) {
            dst[بلاڪ_idx * QK4_0 + i] = عارضي_بفر[i];
        }
    }
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
ساكن فراغ copy_q8_0_صف(عائم * dst, ثابت بلاڪ_q8_0 * src_بلاڪون, int64_t انگ_elements) {
    // Number of Q8_0 blocks needed for this row
    ثابت int64_t انگ_بلاڪون = (انگ_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    لكل (int64_t بلاڪ_idx = 0; بلاڪ_idx < انگ_بلاڪون; بلاڪ_idx++) {
        ثابت int64_t elements_اندريون_بلاڪ =
            (بلاڪ_idx == انگ_بلاڪون - 1) ? (انگ_elements - بلاڪ_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        عائم عارضي_بفر[QK8_0];
        dequantize_q8_0_بلاڪ(&src_بلاڪون[بلاڪ_idx], عارضي_بفر);

        // Copy dequantized values to destination
        لكل (int64_t i = 0; i < elements_اندريون_بلاڪ; i++) {
            dst[بلاڪ_idx * QK8_0 + i] = عارضي_بفر[i];
        }
    }
}

// Copy a row of Q4_K data to F32 destination (with dequantization)
ساكن فراغ copy_q4_K_صف(عائم * dst, ثابت بلاڪ_q4_K * src_بلاڪون, int64_t انگ_elements) {
    ثابت int64_t انگ_بلاڪون = (انگ_elements + QK_K - 1) / QK_K;

    لكل (int64_t بلاڪ_idx = 0; بلاڪ_idx < انگ_بلاڪون; بلاڪ_idx++) {
        ثابت int64_t elements_اندريون_بلاڪ = (بلاڪ_idx == انگ_بلاڪون - 1) ? (انگ_elements - بلاڪ_idx * QK_K) : QK_K;

        عائم عارضي_بفر[QK_K];
        dequantize_q4_K_بلاڪ(&src_بلاڪون[بلاڪ_idx], عارضي_بفر);

        لكل (int64_t i = 0; i < elements_اندريون_بلاڪ; i++) {
            dst[بلاڪ_idx * QK_K + i] = عارضي_بفر[i];
        }
    }
}

ساكن فراغ dequantize_q8_0_بلاڪ_cache_aligned(ثابت بلاڪ_q8_0 * بلاڪ, عائم * dst) {
    ثابت int8_t * qs_ptr = بلاڪ->qs;

    uint64_t عارضي_نقاب;
    __asm__ متطاير("mova.x.m %0" : "=r"(عارضي_نقاب));  // Save current mask
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    ثابت int32_t __attribute__((aligned(32))) vec_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    عائم                                      scale          = fp16_منزل_fp32(بلاڪ->d);
    __asm__ متطاير(
        "fbcx.ps     f0, %0       \n\t"  // Broadcast integer scale to all lanes
        "flq2        f1, 0(%1)    \n\t"  // Load gether indicies
        ::"r"(scale),
        "r"(vec_indices)
        : "f0", "f1");

    لكل (صحيح i = 0; i < 4; i++) {
        __asm__ متطاير(
            "fgb.ps      f2, f1(%0)   \n\t"  // Loads 8 bytes from (qs_ptr + indices) and sign-extends to 32-bit int.
            "fcvt.ps.pw  f2, f2, rne  \n\t"  // Convert Int32 to Float32
            "fmul.ps     f2, f2, f0   \n\t"  // f2 = f2 * f0 (scale)
            "fsq2        f2, 0(%1)    \n\t"  // Store 256 bits (8 floats) to dst.

            ::"r"(qs_ptr),
            "r"(dst)
            : "f2", "memory");

        // Advance pointers in C
        qs_ptr += 8;
        dst += 8;
    }
    __asm__ متطاير("mova.m.x %0" ::"r"(عارضي_نقاب));
}

// Copy a row of Q4_0 data to F32 destination (with dequantization), cache-aligned
ساكن فراغ copy_q4_0_صف_cache_aligned(عائم * dst, ثابت بلاڪ_q4_0 * src_بلاڪون, int64_t انگ_elements) {
    ثابت int64_t انگ_بلاڪون = (انگ_elements + QK4_0 - 1) / QK4_0;

    // Scatter byte offsets: even lanes -> dst[j], odd lanes -> dst[j + QK4_0/2]
    // For 4 consecutive packed bytes producing [low0, high0, low1, high1, low2, high2, low3, high3]:
    //   low_i  -> byte offset i*4       (positions 0,1,2,3 in first half)
    //   high_i -> byte offset (16+i)*4  (positions 16,17,18,19 in second half)
    ثابت int32_t __attribute__((aligned(32))) scatter_offsets[8] = { 0 * 4, 16 * 4, 1 * 4, 17 * 4,
                                                                      2 * 4, 18 * 4, 3 * 4, 19 * 4 };

    // Gather indices: each byte loaded twice for low/high nibble extraction
    ثابت int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 0, 1, 1, 2, 2, 3, 3 };

    uint64_t عارضي_نقاب;
    __asm__ متطاير("mova.x.m %0" : "=r"(عارضي_نقاب));  // Save current mask
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load constant vectors once — shared across all blocks and iterations
    __asm__ متطاير(
        "flq2        f4, 0(%0)    \n\t"  // f4 = scatter offsets
        "flq2        f1, 0(%1)    \n\t"  // f1 = gather indices {0,0,1,1,2,2,3,3}
        ::"r"(scatter_offsets),
        "r"(gather_indices)
        : "f1", "f4");

    لكل (int64_t بلاڪ_idx = 0; بلاڪ_idx < انگ_بلاڪون; بلاڪ_idx++) {
        ثابت بلاڪ_q4_0 * بلاڪ     = &src_بلاڪون[بلاڪ_idx];
        ثابت uint8_t *    qs        = بلاڪ->qs;
        عائم *            بلاڪ_dst = dst + بلاڪ_idx * QK4_0;

        عائم scale = fp16_منزل_fp32(بلاڪ->d);
        عائم bias  = -8.0f * scale;

        // Per-block: broadcast scale and bias
        __asm__ متطاير(
            "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(scale)
            "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-8 * scale)
            ::"r"(scale),
            "r"(bias)
            : "f0", "f3");

        // 4 iterations x 4 packed bytes = 16 bytes = full block -> 32 floats
        لكل (صحيح i = 0; i < 4; i++) {
            __asm__ متطاير(
                "fgb.ps      f2, f1(%0)    \n\t"  // Gather: [b0,b0,b1,b1,b2,b2,b3,b3]
                "mov.m.x     m0, x0, 0xAA  \n\t"  // Odd lanes only (fills gather latency)
                "fsrli.pi    f2, f2, 4     \n\t"  // Odd lanes: byte >> 4 (high nibble)
                "mov.m.x     m0, x0, 0xFF  \n\t"  // Restore full mask
                "fslli.pi    f2, f2, 28    \n\t"  // Isolate low 4 bits: shift left 28
                "fsrli.pi    f2, f2, 28    \n\t"  //   then right 28 -> nibble in [3:0]
                "fcvt.ps.pw  f2, f2, rne   \n\t"  // Int32 -> Float32
                "fmul.ps     f2, f2, f0    \n\t"  // * scale
                "fadd.ps     f2, f2, f3    \n\t"  // + bias -> (nibble - 8) * scale
                "fscw.ps     f2, f4(%1)    \n\t"  // Scatter to GGML positions

                ::"r"(qs),
                "r"(بلاڪ_dst)
                : "f2", "memory");

            qs += 4;         // 4 packed bytes consumed
            بلاڪ_dst += 4;  // Advance base by 4 float positions
        }
    }

    __asm__ متطاير("mova.m.x %0" ::"r"(عارضي_نقاب));  // Restore mask
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
ساكن فراغ copy_q8_0_صف_cache_aligned(عائم * dst, ثابت بلاڪ_q8_0 * src_بلاڪون, int64_t انگ_elements) {
    // Number of Q8_0 blocks needed for this row
    ثابت int64_t انگ_بلاڪون = (انگ_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    لكل (int64_t بلاڪ_idx = 0; بلاڪ_idx < انگ_بلاڪون; بلاڪ_idx++) {
        ثابت int64_t elements_اندريون_بلاڪ =
            (بلاڪ_idx == انگ_بلاڪون - 1) ? (انگ_elements - بلاڪ_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        عائم عارضي_بفر[QK8_0];
        dequantize_q8_0_بلاڪ_cache_aligned(&src_بلاڪون[بلاڪ_idx], عارضي_بفر);

        // Copy dequantized values to destination
        لكل (int64_t i = 0; i < elements_اندريون_بلاڪ; i++) {
            dst[بلاڪ_idx * QK8_0 + i] = عارضي_بفر[i];
        }
    }
}

// Vectorized dequantization of a Q4_K super-block (256 elements) to F32
// Processes 8 groups of 32 elements, using ET SIMD for the inner loops.
// Output is sequential (no scatter needed unlike Q4_0).
ساكن فراغ copy_q4_K_صف_cache_aligned(عائم * dst, ثابت بلاڪ_q4_K * src_بلاڪون, int64_t انگ_elements) {
    ثابت int64_t انگ_بلاڪون = (انگ_elements + QK_K - 1) / QK_K;

    // Gather indices for sequential byte access: {0,1,2,3,4,5,6,7}
    ثابت int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    uint64_t عارضي_نقاب;
    __asm__ متطاير("mova.x.m %0" : "=r"(عارضي_نقاب));  // Save current mask
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load gather indices once — shared across all blocks
    __asm__ متطاير("flq2        f1, 0(%0)    \n\t"  // f1 = gather indices {0,1,2,3,4,5,6,7}
                     ::"r"(gather_indices)
                     : "f1");

    لكل (int64_t بلاڪ_idx = 0; بلاڪ_idx < انگ_بلاڪون; بلاڪ_idx++) {
        ثابت بلاڪ_q4_K * بلاڪ     = &src_بلاڪون[بلاڪ_idx];
        ثابت uint8_t *    qs        = بلاڪ->qs;
        عائم *            بلاڪ_dst = dst + بلاڪ_idx * QK_K;

        ثابت عائم d   = fp16_منزل_fp32(بلاڪ->d);
        ثابت عائم min = fp16_منزل_fp32(بلاڪ->dmin);

        صحيح is = 0;
        لكل (صحيح j = 0; j < QK_K; j += 64) {
            // Extract per-group scales and mins (scalar — only 8 pairs per super-block)
            uint8_t sc, m;
            get_scale_min_k4(is + 0, بلاڪ->scales, &sc, &m);
            ثابت عائم d1     = d * sc;
            ثابت عائم neg_m1 = -(min * m);
            get_scale_min_k4(is + 1, بلاڪ->scales, &sc, &m);
            ثابت عائم d2     = d * sc;
            ثابت عائم neg_m2 = -(min * m);

            // Low nibbles: 32 elements using d1, neg_m1
            __asm__ متطاير(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d1)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m1)
                ::"r"(d1),
                "r"(neg_m1)
                : "f0", "f3");

            ثابت uint8_t * qs_lo  = qs;
            عائم *         dst_lo = بلاڪ_dst + j;
            لكل (صحيح k = 0; k < 4; k++) {
                __asm__ متطاير(
                    "fgb.ps      f2, f1(%0)   \n\t"   // Gather 8 bytes, sign-extend to int32
                    "fandi.pi    f2, f2, 0xF   \n\t"  // Mask low nibble (imm10=15)
                    "fcvt.ps.pw  f2, f2, rne   \n\t"  // Int32 -> Float32
                    "fmadd.ps    f2, f2, f0, f3\n\t"  // d1 * nibble + (-m1)
                    "fsq2        f2, 0(%1)     \n\t"  // Store 8 floats
                    ::"r"(qs_lo),
                    "r"(dst_lo)
                    : "f2", "memory");
                qs_lo += 8;
                dst_lo += 8;
            }

            // High nibbles: 32 elements using d2, neg_m2
            __asm__ متطاير(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d2)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m2)
                ::"r"(d2),
                "r"(neg_m2)
                : "f0", "f3");

            ثابت uint8_t * qs_hi  = qs;
            عائم *         dst_hi = بلاڪ_dst + j + 32;
            لكل (صحيح k = 0; k < 4; k++) {
                __asm__ متطاير(
                    "fgb.ps      f2, f1(%0)   \n\t"   // Gather 8 bytes, sign-extend to int32
                    "fsrli.pi    f2, f2, 4     \n\t"  // Shift right 4: high nibble
                    "fandi.pi    f2, f2, 0xF   \n\t"  // Mask to 4 bits (clean any sign-ext artifacts)
                    "fcvt.ps.pw  f2, f2, rne   \n\t"  // Int32 -> Float32
                    "fmadd.ps    f2, f2, f0, f3\n\t"  // d2 * nibble + (-m2)
                    "fsq2        f2, 0(%1)     \n\t"  // Store 8 floats
                    ::"r"(qs_hi),
                    "r"(dst_hi)
                    : "f2", "memory");
                qs_hi += 8;
                dst_hi += 8;
            }

            qs += 32;  // Advance to next 32 packed bytes
            is += 2;
        }
    }

    __asm__ متطاير("mova.m.x %0" ::"r"(عارضي_نقاب));  // Restore mask
}

// Determine the number of F32 elements per work unit for a given source type.
// For F32: 1 cacheline (16 elements)
// For quantized types: 1 quant block
ساكن int64_t get_elements_في_work_unit(صحيح type) {
    ثابت int64_t elements_في_cacheline = CACHE_سٽ_سائيز_بائيٽون / حجم(عائم);  // 16
    بدل (type) {
        حالة GGML_TYPE_Q8_0:
            أرجع QK8_0;                   // 32 elements = 2 cachelines
        حالة GGML_TYPE_Q4_0:
            أرجع QK4_0;                   // 32 elements = 2 cachelines
        حالة GGML_TYPE_Q4_K:
            أرجع QK_K;                    // 256 elements = 16 cachelines
        افتراضي:
            أرجع elements_في_cacheline;  // 16 elements = 1 cacheline
    }
}

ساكن صحيح get_صف_f32_mc_cacheline_aligned(بنية ggml_et_get_صفون_params * params, فراغ * env) {
    kernel_environment_t * kernel_env  = (kernel_environment_t *) env;
    صحيح                    thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح                    انگ_ڌاڳا = get_انگ_ڌاڳا(kernel_env->shire_نقاب);

    بنية ggml_tensor * src0 = &params->src0;  // Data tensor
    بنية ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    بنية ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    ثابت int64_t ne00 = src0->ne[0];           // Source columns (row width)
    ثابت int64_t ne01 = src0->ne[1];           // Source rows (total available rows)
    ثابت int64_t ne02 = src0->ne[2];           // Source batch dimension
    ثابت int64_t ne03 = src0->ne[3];           // Source outer batch dimension

    ثابت int64_t ne10 = src1->ne[0];           // Number of indices in dimension 0
    ثابت int64_t ne11 = src1->ne[1];           // Number of indices in dimension 1
    ثابت int64_t ne12 = src1->ne[2];           // Batch dimension for indices
    ثابت int64_t ne13 = src1->ne[3];           // Outer batch dimension for indices

    ثابت int64_t ڪل_صفون_منزل_ڪڍو = ne10 * ne11 * ne12 * ne13;

    // Determine work unit size based on source type
    ثابت int64_t elements_في_wu = get_elements_في_work_unit(src0->type);
    ثابت int64_t wus_في_صف     = ne00 / elements_في_wu;
    ثابت int64_t ڪل_wus       = ڪل_صفون_منزل_ڪڍو * wus_في_صف;

    // Distribute work units across threads (contiguous ranges)
    ثابت int64_t wus_في_thread = (ڪل_wus + انگ_ڌاڳا - 1) / انگ_ڌاڳا;
    ثابت int64_t wu_شروعات       = thread_id * wus_في_thread;
    int64_t       wu_پڇاڙي         = wu_شروعات + wus_في_thread;
    إذا (wu_پڇاڙي > ڪل_wus) {
        wu_پڇاڙي = ڪل_wus;
    }

    فراغ *    src0_معلومات = src0->معلومات;
    int32_t * src1_معلومات = (int32_t *) src1->معلومات;
    عائم *   dst_معلومات  = (عائم *) dst->معلومات;

    int64_t wu = wu_شروعات;
    طالما (wu < wu_پڇاڙي) {
        // Determine which row this work unit belongs to and offset within row
        ثابت int64_t صف_idx   = wu / wus_في_صف;
        ثابت int64_t wu_اندريون_صف = wu % wus_في_صف;

        // How many work units to process in this row (batch contiguous WUs in same row)
        int64_t wus_بچيل_اندريون_صف = wus_في_صف - wu_اندريون_صف;
        int64_t wus_منزل_process       = wu_پڇاڙي - wu;
        إذا (wus_بچيل_اندريون_صف < wus_منزل_process) {
            wus_منزل_process = wus_بچيل_اندريون_صف;
        }

        // Calculate multi-dimensional index for this row
        ثابت int64_t i       = صف_idx;
        ثابت int64_t i13_idx = i / (ne12 * ne11 * ne10);
        ثابت int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        ثابت int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        ثابت int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        ثابت int64_t اشاريو_وٿي = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        ثابت int32_t صف_اشاريو    = src1_معلومات[اشاريو_وٿي];

        إذا (صف_اشاريو < 0 || صف_اشاريو >= ne01) {
            أرجع -1;  // Index out of bounds
        }

        ثابت int64_t batch_وٿي =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        ثابت int64_t elem_وٿي_اندريون_صف = wu_اندريون_صف * elements_في_wu;
        ثابت int64_t انگ_elements       = wus_منزل_process * elements_في_wu;

        عائم * dst_صف = dst_معلومات + صف_idx * ne00 + elem_وٿي_اندريون_صف;

        إذا (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy of cacheline-aligned chunk
            ثابت عائم * src_صف = (ثابت عائم *) src0_معلومات + صف_اشاريو * ne00 + batch_وٿي + elem_وٿي_اندريون_صف;
            copy_صف_cache_align(dst_صف, src_صف, انگ_elements * حجم(عائم));
        } وإلا إذا (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion over a destination-aligned write chunk.
            ثابت uint16_t * src_صف =
                (ثابت uint16_t *) src0_معلومات + صف_اشاريو * ne00 + batch_وٿي + elem_وٿي_اندريون_صف;
            copy_f16_صف(dst_صف, src_صف, انگ_elements);
        } وإلا إذا (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize work-unit-aligned blocks
            ثابت int64_t      بلاڪون_في_صف   = (ne00 + QK8_0 - 1) / QK8_0;
            ثابت int64_t      src_بلاڪ_وٿي = (صف_اشاريو * بلاڪون_في_صف) + (batch_وٿي / ne00) * بلاڪون_في_صف;
            ثابت int64_t      بلاڪ_شروعات      = elem_وٿي_اندريون_صف / QK8_0;
            ثابت بلاڪ_q8_0 * src_بلاڪون       = (ثابت بلاڪ_q8_0 *) src0_معلومات + src_بلاڪ_وٿي + بلاڪ_شروعات;
            copy_q8_0_صف_cache_aligned(dst_صف, src_بلاڪون, انگ_elements);
        } وإلا إذا (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize work-unit-aligned blocks
            ثابت int64_t      بلاڪون_في_صف   = (ne00 + QK4_0 - 1) / QK4_0;
            ثابت int64_t      src_بلاڪ_وٿي = (صف_اشاريو * بلاڪون_في_صف) + (batch_وٿي / ne00) * بلاڪون_في_صف;
            ثابت int64_t      بلاڪ_شروعات      = elem_وٿي_اندريون_صف / QK4_0;
            ثابت بلاڪ_q4_0 * src_بلاڪون       = (ثابت بلاڪ_q4_0 *) src0_معلومات + src_بلاڪ_وٿي + بلاڪ_شروعات;
            copy_q4_0_صف_cache_aligned(dst_صف, src_بلاڪون, انگ_elements);
        } وإلا إذا (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize work-unit-aligned blocks
            ثابت int64_t      بلاڪون_في_صف   = (ne00 + QK_K - 1) / QK_K;
            ثابت int64_t      src_بلاڪ_وٿي = (صف_اشاريو * بلاڪون_في_صف) + (batch_وٿي / ne00) * بلاڪون_في_صف;
            ثابت int64_t      بلاڪ_شروعات      = elem_وٿي_اندريون_صف / QK_K;
            ثابت بلاڪ_q4_K * src_بلاڪون       = (ثابت بلاڪ_q4_K *) src0_معلومات + src_بلاڪ_وٿي + بلاڪ_شروعات;
            copy_q4_K_صف_cache_aligned(dst_صف, src_بلاڪون, انگ_elements);
        }

        wu += wus_منزل_process;
    }

    أرجع 0;
}

ساكن مضمن size_t tensor_بائيٽون(ثابت بنية ggml_tensor * t) {
    أرجع (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

صحيح داخلا_نقطو(بنية ggml_et_get_صفون_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    إذا (!kernel_env) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    بنية ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    بنية ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    // Fast path - we know how to deal with them multi-core
    إذا ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_Q8_0 ||
         src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_K) &&
        src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32 && dst->ne[0] % CACHE_ELEMENTS(حجم(عائم)) == 0) {
        أرجع get_صف_f32_mc_cacheline_aligned(params, env);
    }

    صحيح thread_id = get_relative_thread_id(kernel_env->shire_نقاب);
    إذا (thread_id < 0) {
        أرجع 0;
    }

    إذا (thread_id != 0) {
        أرجع 0;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;  // Invalid pointer
    }

    إذا (dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        أرجع -1;  // Invalid output or index type
    }

    إذا (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q4_K) {
        أرجع -1;  // Unsupported input type
    }

    فراغ *    src0_معلومات = src0->معلومات;
    int32_t * src1_معلومات = (int32_t *) src1->معلومات;
    عائم *   dst_معلومات  = (عائم *) dst->معلومات;
#جيڪڏهن_وصف_ٿيل ET_UBERKERNEL
    evict_region_past_l2(src0_معلومات, tensor_بائيٽون(src0));
    evict_region_past_l2(src1_معلومات, tensor_بائيٽون(src1));
    evict_region_past_l2(dst_معلومات, tensor_بائيٽون(dst));
#جيڪڏهن_پورو

    إذا (!src0_معلومات || !src1_معلومات || !dst_معلومات) {
        أرجع -1;  // Null data pointer
    }

    ثابت int64_t ne00 = src0->ne[0];  // Source columns (row width)
    ثابت int64_t ne01 = src0->ne[1];  // Source rows (total available rows)
    ثابت int64_t ne02 = src0->ne[2];  // Source batch dimension
    ثابت int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    ثابت int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    ثابت int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    ثابت int64_t ne12 = src1->ne[2];  // Batch dimension for indices
    ثابت int64_t ne13 = src1->ne[3];  // Outer batch dimension for indices

    ثابت int64_t ڪل_صفون_منزل_ڪڍو = ne10 * ne11 * ne12 * ne13;
#جيڪڏهن_وصف_ٿيل ET_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
#جيڪڏهن_پورو
    // Naive single-threaded implementation - process all rows sequentially
    // XXX: Do we really need a single-threaded implementation?
    لكل (int64_t i = 0; i < ڪل_صفون_منزل_ڪڍو; i++) {
        // Calculate multi-dimensional index for the current output position
        ثابت int64_t i13_idx = i / (ne12 * ne11 * ne10);
        ثابت int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        ثابت int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        ثابت int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        ثابت int64_t اشاريو_وٿي = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        ثابت int32_t صف_اشاريو    = src1_معلومات[اشاريو_وٿي];

        إذا (صف_اشاريو < 0 || صف_اشاريو >= ne01) {
            أرجع -1;  // Index out of bounds
        }

        ثابت int64_t batch_وٿي =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        ثابت int64_t dst_وٿي = i;

        إذا (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy
            ثابت عائم * src_صف = (ثابت عائم *) src0_معلومات + صف_اشاريو * ne00 + batch_وٿي;
            عائم *       dst_صف = dst_معلومات + dst_وٿي * ne00;
            copy_f32_صف(dst_صف, src_صف, ne00);
        } وإلا إذا (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion
            ثابت uint16_t * src_صف = (ثابت uint16_t *) src0_معلومات + صف_اشاريو * ne00 + batch_وٿي;
            عائم *          dst_صف = dst_معلومات + dst_وٿي * ne00;
            copy_f16_صف(dst_صف, src_صف, ne00);
        } وإلا إذا (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize while copying
            ثابت int64_t      بلاڪون_في_صف   = (ne00 + QK8_0 - 1) / QK8_0;
            ثابت int64_t      src_بلاڪ_وٿي = (صف_اشاريو * بلاڪون_في_صف) + (batch_وٿي / ne00) * بلاڪون_في_صف;
            ثابت بلاڪ_q8_0 * src_بلاڪون       = (ثابت بلاڪ_q8_0 *) src0_معلومات + src_بلاڪ_وٿي;
            عائم *            dst_صف          = dst_معلومات + dst_وٿي * ne00;
            copy_q8_0_صف(dst_صف, src_بلاڪون, ne00);
        } وإلا إذا (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize while copying
            ثابت int64_t      بلاڪون_في_صف   = (ne00 + QK4_0 - 1) / QK4_0;
            ثابت int64_t      src_بلاڪ_وٿي = (صف_اشاريو * بلاڪون_في_صف) + (batch_وٿي / ne00) * بلاڪون_في_صف;
            ثابت بلاڪ_q4_0 * src_بلاڪون       = (ثابت بلاڪ_q4_0 *) src0_معلومات + src_بلاڪ_وٿي;
            عائم *            dst_صف          = dst_معلومات + dst_وٿي * ne00;
            copy_q4_0_صف(dst_صف, src_بلاڪون, ne00);
        } وإلا إذا (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize while copying
            ثابت int64_t      بلاڪون_في_صف   = (ne00 + QK_K - 1) / QK_K;
            ثابت int64_t      src_بلاڪ_وٿي = (صف_اشاريو * بلاڪون_في_صف) + (batch_وٿي / ne00) * بلاڪون_في_صف;
            ثابت بلاڪ_q4_K * src_بلاڪون       = (ثابت بلاڪ_q4_K *) src0_معلومات + src_بلاڪ_وٿي;
            عائم *            dst_صف          = dst_معلومات + dst_وٿي * ne00;
            copy_q4_K_صف(dst_صف, src_بلاڪون, ne00);
        }
    }

    أرجع 0;
}
