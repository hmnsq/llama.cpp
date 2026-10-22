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

#تعریف_کړه CACHE_کرښه_اندازه_بایټونه 64

بنية ggml_et_get_لیکې_params {
    بنية ggml_tensor src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    بنية ggml_tensor src1;  // Row indices tensor (I32)
    بنية ggml_tensor dst;   // Output tensor (F32)
};

#تعریف_کړه CACHE_کرښه_اندازه_بایټونه     64
#تعریف_کړه CACHE_ELEMENTS(elem_اندازه) (CACHE_کرښه_اندازه_بایټونه / (elem_اندازه))

// Copy a row of F32 data from source to destination
ساكن فراغ copy_f32_لیکه(عائم * dst, ثابت عائم * src, int64_t عدد_elements) {
    // Simple memcpy for F32 data - no conversion needed
    لكل (int64_t i = 0; i < عدد_elements; i++) {
        dst[i] = src[i];
    }
}

ساكن فراغ copy_f16_لیکه(عائم * dst, ثابت uint16_t * src, int64_t عدد_elements) {
    لكل (int64_t i = 0; i < عدد_elements; i++) {
        dst[i] = fp16_منزل_fp32(src[i]);
    }
}

// Copy a row of F32 data from source to destination, aligned to cache line boundaries
// using FP32 load/store instructions. They don't perform data conversion so is fine.
// Requirement: n_bytes is a multiple of CACHE_LINE_SIZE (64 bytes)
ساكن فراغ copy_لیکه_cache_align(عائم * dst, ثابت عائم * src, int64_t n_بایټونه) {
    صحيح عدد_f32_elem = n_بایټونه / حجم(عائم);

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

        : [dst] "+r"(dst), [src] "+r"(src), [n] "+r"(عدد_f32_elem)
        : [stride_شمېر] "r"(16L)
        : "f0", "f1", "memory");
}

// Copied from GGML: copy a row of Q4_0 data to F32 destination (with dequantization)
ساكن فراغ copy_q4_0_لیکه(عائم * dst, ثابت بلاک_q4_0 * src_بلاکونه, int64_t عدد_elements) {
    ثابت int64_t عدد_بلاکونه = (عدد_elements + QK4_0 - 1) / QK4_0;

    لكل (int64_t بلاک_idx = 0; بلاک_idx < عدد_بلاکونه; بلاک_idx++) {
        ثابت int64_t elements_دننه_بلاک = (بلاک_idx == عدد_بلاکونه - 1) ? (عدد_elements - بلاک_idx * QK4_0) : QK4_0;

        عائم لنډمهاله_بفر[QK4_0];
        dequantize_q4_0_بلاک(&src_بلاکونه[بلاک_idx], لنډمهاله_بفر);

        لكل (int64_t i = 0; i < elements_دننه_بلاک; i++) {
            dst[بلاک_idx * QK4_0 + i] = لنډمهاله_بفر[i];
        }
    }
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
ساكن فراغ copy_q8_0_لیکه(عائم * dst, ثابت بلاک_q8_0 * src_بلاکونه, int64_t عدد_elements) {
    // Number of Q8_0 blocks needed for this row
    ثابت int64_t عدد_بلاکونه = (عدد_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    لكل (int64_t بلاک_idx = 0; بلاک_idx < عدد_بلاکونه; بلاک_idx++) {
        ثابت int64_t elements_دننه_بلاک =
            (بلاک_idx == عدد_بلاکونه - 1) ? (عدد_elements - بلاک_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        عائم لنډمهاله_بفر[QK8_0];
        dequantize_q8_0_بلاک(&src_بلاکونه[بلاک_idx], لنډمهاله_بفر);

        // Copy dequantized values to destination
        لكل (int64_t i = 0; i < elements_دننه_بلاک; i++) {
            dst[بلاک_idx * QK8_0 + i] = لنډمهاله_بفر[i];
        }
    }
}

// Copy a row of Q4_K data to F32 destination (with dequantization)
ساكن فراغ copy_q4_K_لیکه(عائم * dst, ثابت بلاک_q4_K * src_بلاکونه, int64_t عدد_elements) {
    ثابت int64_t عدد_بلاکونه = (عدد_elements + QK_K - 1) / QK_K;

    لكل (int64_t بلاک_idx = 0; بلاک_idx < عدد_بلاکونه; بلاک_idx++) {
        ثابت int64_t elements_دننه_بلاک = (بلاک_idx == عدد_بلاکونه - 1) ? (عدد_elements - بلاک_idx * QK_K) : QK_K;

        عائم لنډمهاله_بفر[QK_K];
        dequantize_q4_K_بلاک(&src_بلاکونه[بلاک_idx], لنډمهاله_بفر);

        لكل (int64_t i = 0; i < elements_دننه_بلاک; i++) {
            dst[بلاک_idx * QK_K + i] = لنډمهاله_بفر[i];
        }
    }
}

ساكن فراغ dequantize_q8_0_بلاک_cache_aligned(ثابت بلاک_q8_0 * بلاک, عائم * dst) {
    ثابت int8_t * qs_ptr = بلاک->qs;

    uint64_t لنډمهاله_ماسک;
    __asm__ متطاير("mova.x.m %0" : "=r"(لنډمهاله_ماسک));  // Save current mask
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    ثابت int32_t __attribute__((aligned(32))) vec_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    عائم                                      scale          = fp16_منزل_fp32(بلاک->d);
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
    __asm__ متطاير("mova.m.x %0" ::"r"(لنډمهاله_ماسک));
}

// Copy a row of Q4_0 data to F32 destination (with dequantization), cache-aligned
ساكن فراغ copy_q4_0_لیکه_cache_aligned(عائم * dst, ثابت بلاک_q4_0 * src_بلاکونه, int64_t عدد_elements) {
    ثابت int64_t عدد_بلاکونه = (عدد_elements + QK4_0 - 1) / QK4_0;

    // Scatter byte offsets: even lanes -> dst[j], odd lanes -> dst[j + QK4_0/2]
    // For 4 consecutive packed bytes producing [low0, high0, low1, high1, low2, high2, low3, high3]:
    //   low_i  -> byte offset i*4       (positions 0,1,2,3 in first half)
    //   high_i -> byte offset (16+i)*4  (positions 16,17,18,19 in second half)
    ثابت int32_t __attribute__((aligned(32))) scatter_offsets[8] = { 0 * 4, 16 * 4, 1 * 4, 17 * 4,
                                                                      2 * 4, 18 * 4, 3 * 4, 19 * 4 };

    // Gather indices: each byte loaded twice for low/high nibble extraction
    ثابت int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 0, 1, 1, 2, 2, 3, 3 };

    uint64_t لنډمهاله_ماسک;
    __asm__ متطاير("mova.x.m %0" : "=r"(لنډمهاله_ماسک));  // Save current mask
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load constant vectors once — shared across all blocks and iterations
    __asm__ متطاير(
        "flq2        f4, 0(%0)    \n\t"  // f4 = scatter offsets
        "flq2        f1, 0(%1)    \n\t"  // f1 = gather indices {0,0,1,1,2,2,3,3}
        ::"r"(scatter_offsets),
        "r"(gather_indices)
        : "f1", "f4");

    لكل (int64_t بلاک_idx = 0; بلاک_idx < عدد_بلاکونه; بلاک_idx++) {
        ثابت بلاک_q4_0 * بلاک     = &src_بلاکونه[بلاک_idx];
        ثابت uint8_t *    qs        = بلاک->qs;
        عائم *            بلاک_dst = dst + بلاک_idx * QK4_0;

        عائم scale = fp16_منزل_fp32(بلاک->d);
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
                "r"(بلاک_dst)
                : "f2", "memory");

            qs += 4;         // 4 packed bytes consumed
            بلاک_dst += 4;  // Advance base by 4 float positions
        }
    }

    __asm__ متطاير("mova.m.x %0" ::"r"(لنډمهاله_ماسک));  // Restore mask
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
ساكن فراغ copy_q8_0_لیکه_cache_aligned(عائم * dst, ثابت بلاک_q8_0 * src_بلاکونه, int64_t عدد_elements) {
    // Number of Q8_0 blocks needed for this row
    ثابت int64_t عدد_بلاکونه = (عدد_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    لكل (int64_t بلاک_idx = 0; بلاک_idx < عدد_بلاکونه; بلاک_idx++) {
        ثابت int64_t elements_دننه_بلاک =
            (بلاک_idx == عدد_بلاکونه - 1) ? (عدد_elements - بلاک_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        عائم لنډمهاله_بفر[QK8_0];
        dequantize_q8_0_بلاک_cache_aligned(&src_بلاکونه[بلاک_idx], لنډمهاله_بفر);

        // Copy dequantized values to destination
        لكل (int64_t i = 0; i < elements_دننه_بلاک; i++) {
            dst[بلاک_idx * QK8_0 + i] = لنډمهاله_بفر[i];
        }
    }
}

// Vectorized dequantization of a Q4_K super-block (256 elements) to F32
// Processes 8 groups of 32 elements, using ET SIMD for the inner loops.
// Output is sequential (no scatter needed unlike Q4_0).
ساكن فراغ copy_q4_K_لیکه_cache_aligned(عائم * dst, ثابت بلاک_q4_K * src_بلاکونه, int64_t عدد_elements) {
    ثابت int64_t عدد_بلاکونه = (عدد_elements + QK_K - 1) / QK_K;

    // Gather indices for sequential byte access: {0,1,2,3,4,5,6,7}
    ثابت int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    uint64_t لنډمهاله_ماسک;
    __asm__ متطاير("mova.x.m %0" : "=r"(لنډمهاله_ماسک));  // Save current mask
    __asm__ متطاير("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load gather indices once — shared across all blocks
    __asm__ متطاير("flq2        f1, 0(%0)    \n\t"  // f1 = gather indices {0,1,2,3,4,5,6,7}
                     ::"r"(gather_indices)
                     : "f1");

    لكل (int64_t بلاک_idx = 0; بلاک_idx < عدد_بلاکونه; بلاک_idx++) {
        ثابت بلاک_q4_K * بلاک     = &src_بلاکونه[بلاک_idx];
        ثابت uint8_t *    qs        = بلاک->qs;
        عائم *            بلاک_dst = dst + بلاک_idx * QK_K;

        ثابت عائم d   = fp16_منزل_fp32(بلاک->d);
        ثابت عائم min = fp16_منزل_fp32(بلاک->dmin);

        صحيح is = 0;
        لكل (صحيح j = 0; j < QK_K; j += 64) {
            // Extract per-group scales and mins (scalar — only 8 pairs per super-block)
            uint8_t sc, m;
            get_scale_min_k4(is + 0, بلاک->scales, &sc, &m);
            ثابت عائم d1     = d * sc;
            ثابت عائم neg_m1 = -(min * m);
            get_scale_min_k4(is + 1, بلاک->scales, &sc, &m);
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
            عائم *         dst_lo = بلاک_dst + j;
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
            عائم *         dst_hi = بلاک_dst + j + 32;
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

    __asm__ متطاير("mova.m.x %0" ::"r"(لنډمهاله_ماسک));  // Restore mask
}

// Determine the number of F32 elements per work unit for a given source type.
// For F32: 1 cacheline (16 elements)
// For quantized types: 1 quant block
ساكن int64_t get_elements_فی_work_unit(صحيح type) {
    ثابت int64_t elements_فی_cacheline = CACHE_کرښه_اندازه_بایټونه / حجم(عائم);  // 16
    بدل (type) {
        حالة GGML_TYPE_Q8_0:
            أرجع QK8_0;                   // 32 elements = 2 cachelines
        حالة GGML_TYPE_Q4_0:
            أرجع QK4_0;                   // 32 elements = 2 cachelines
        حالة GGML_TYPE_Q4_K:
            أرجع QK_K;                    // 256 elements = 16 cachelines
        افتراضي:
            أرجع elements_فی_cacheline;  // 16 elements = 1 cacheline
    }
}

ساكن صحيح get_لیکه_f32_mc_cacheline_aligned(بنية ggml_et_get_لیکې_params * params, فراغ * env) {
    kernel_environment_t * kernel_env  = (kernel_environment_t *) env;
    صحيح                    thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح                    عدد_تارونه = get_عدد_تارونه(kernel_env->shire_ماسک);

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

    ثابت int64_t ټول_لیکې_منزل_ایستل = ne10 * ne11 * ne12 * ne13;

    // Determine work unit size based on source type
    ثابت int64_t elements_فی_wu = get_elements_فی_work_unit(src0->type);
    ثابت int64_t wus_فی_لیکه     = ne00 / elements_فی_wu;
    ثابت int64_t ټول_wus       = ټول_لیکې_منزل_ایستل * wus_فی_لیکه;

    // Distribute work units across threads (contiguous ranges)
    ثابت int64_t wus_فی_thread = (ټول_wus + عدد_تارونه - 1) / عدد_تارونه;
    ثابت int64_t wu_پیل       = thread_id * wus_فی_thread;
    int64_t       wu_پای_ټکی         = wu_پیل + wus_فی_thread;
    إذا (wu_پای_ټکی > ټول_wus) {
        wu_پای_ټکی = ټول_wus;
    }

    فراغ *    src0_معلومات = src0->معلومات;
    int32_t * src1_معلومات = (int32_t *) src1->معلومات;
    عائم *   dst_معلومات  = (عائم *) dst->معلومات;

    int64_t wu = wu_پیل;
    طالما (wu < wu_پای_ټکی) {
        // Determine which row this work unit belongs to and offset within row
        ثابت int64_t لیکه_idx   = wu / wus_فی_لیکه;
        ثابت int64_t wu_دننه_لیکه = wu % wus_فی_لیکه;

        // How many work units to process in this row (batch contiguous WUs in same row)
        int64_t wus_پاتې_شوی_دننه_لیکه = wus_فی_لیکه - wu_دننه_لیکه;
        int64_t wus_منزل_process       = wu_پای_ټکی - wu;
        إذا (wus_پاتې_شوی_دننه_لیکه < wus_منزل_process) {
            wus_منزل_process = wus_پاتې_شوی_دننه_لیکه;
        }

        // Calculate multi-dimensional index for this row
        ثابت int64_t i       = لیکه_idx;
        ثابت int64_t i13_idx = i / (ne12 * ne11 * ne10);
        ثابت int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        ثابت int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        ثابت int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        ثابت int64_t شاخص_دننه_واټن = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        ثابت int32_t لیکه_شاخص    = src1_معلومات[شاخص_دننه_واټن];

        إذا (لیکه_شاخص < 0 || لیکه_شاخص >= ne01) {
            أرجع -1;  // Index out of bounds
        }

        ثابت int64_t batch_دننه_واټن =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        ثابت int64_t elem_دننه_واټن_دننه_لیکه = wu_دننه_لیکه * elements_فی_wu;
        ثابت int64_t عدد_elements       = wus_منزل_process * elements_فی_wu;

        عائم * dst_لیکه = dst_معلومات + لیکه_idx * ne00 + elem_دننه_واټن_دننه_لیکه;

        إذا (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy of cacheline-aligned chunk
            ثابت عائم * src_لیکه = (ثابت عائم *) src0_معلومات + لیکه_شاخص * ne00 + batch_دننه_واټن + elem_دننه_واټن_دننه_لیکه;
            copy_لیکه_cache_align(dst_لیکه, src_لیکه, عدد_elements * حجم(عائم));
        } وإلا إذا (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion over a destination-aligned write chunk.
            ثابت uint16_t * src_لیکه =
                (ثابت uint16_t *) src0_معلومات + لیکه_شاخص * ne00 + batch_دننه_واټن + elem_دننه_واټن_دننه_لیکه;
            copy_f16_لیکه(dst_لیکه, src_لیکه, عدد_elements);
        } وإلا إذا (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize work-unit-aligned blocks
            ثابت int64_t      بلاکونه_فی_لیکه   = (ne00 + QK8_0 - 1) / QK8_0;
            ثابت int64_t      src_بلاک_دننه_واټن = (لیکه_شاخص * بلاکونه_فی_لیکه) + (batch_دننه_واټن / ne00) * بلاکونه_فی_لیکه;
            ثابت int64_t      بلاک_پیل      = elem_دننه_واټن_دننه_لیکه / QK8_0;
            ثابت بلاک_q8_0 * src_بلاکونه       = (ثابت بلاک_q8_0 *) src0_معلومات + src_بلاک_دننه_واټن + بلاک_پیل;
            copy_q8_0_لیکه_cache_aligned(dst_لیکه, src_بلاکونه, عدد_elements);
        } وإلا إذا (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize work-unit-aligned blocks
            ثابت int64_t      بلاکونه_فی_لیکه   = (ne00 + QK4_0 - 1) / QK4_0;
            ثابت int64_t      src_بلاک_دننه_واټن = (لیکه_شاخص * بلاکونه_فی_لیکه) + (batch_دننه_واټن / ne00) * بلاکونه_فی_لیکه;
            ثابت int64_t      بلاک_پیل      = elem_دننه_واټن_دننه_لیکه / QK4_0;
            ثابت بلاک_q4_0 * src_بلاکونه       = (ثابت بلاک_q4_0 *) src0_معلومات + src_بلاک_دننه_واټن + بلاک_پیل;
            copy_q4_0_لیکه_cache_aligned(dst_لیکه, src_بلاکونه, عدد_elements);
        } وإلا إذا (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize work-unit-aligned blocks
            ثابت int64_t      بلاکونه_فی_لیکه   = (ne00 + QK_K - 1) / QK_K;
            ثابت int64_t      src_بلاک_دننه_واټن = (لیکه_شاخص * بلاکونه_فی_لیکه) + (batch_دننه_واټن / ne00) * بلاکونه_فی_لیکه;
            ثابت int64_t      بلاک_پیل      = elem_دننه_واټن_دننه_لیکه / QK_K;
            ثابت بلاک_q4_K * src_بلاکونه       = (ثابت بلاک_q4_K *) src0_معلومات + src_بلاک_دننه_واټن + بلاک_پیل;
            copy_q4_K_لیکه_cache_aligned(dst_لیکه, src_بلاکونه, عدد_elements);
        }

        wu += wus_منزل_process;
    }

    أرجع 0;
}

ساكن مضمن size_t tensor_بایټونه(ثابت بنية ggml_tensor * t) {
    أرجع (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

صحيح ثبت_ټکی(بنية ggml_et_get_لیکې_params * params, فراغ * env) {
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
        أرجع get_لیکه_f32_mc_cacheline_aligned(params, env);
    }

    صحيح thread_id = get_relative_thread_id(kernel_env->shire_ماسک);
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
#که_تعریف_شوی ET_UBERKERNEL
    evict_region_past_l2(src0_معلومات, tensor_بایټونه(src0));
    evict_region_past_l2(src1_معلومات, tensor_بایټونه(src1));
    evict_region_past_l2(dst_معلومات, tensor_بایټونه(dst));
#د_که_پای

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

    ثابت int64_t ټول_لیکې_منزل_ایستل = ne10 * ne11 * ne12 * ne13;
#که_تعریف_شوی ET_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
#د_که_پای
    // Naive single-threaded implementation - process all rows sequentially
    // XXX: Do we really need a single-threaded implementation?
    لكل (int64_t i = 0; i < ټول_لیکې_منزل_ایستل; i++) {
        // Calculate multi-dimensional index for the current output position
        ثابت int64_t i13_idx = i / (ne12 * ne11 * ne10);
        ثابت int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        ثابت int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        ثابت int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        ثابت int64_t شاخص_دننه_واټن = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        ثابت int32_t لیکه_شاخص    = src1_معلومات[شاخص_دننه_واټن];

        إذا (لیکه_شاخص < 0 || لیکه_شاخص >= ne01) {
            أرجع -1;  // Index out of bounds
        }

        ثابت int64_t batch_دننه_واټن =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        ثابت int64_t dst_دننه_واټن = i;

        إذا (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy
            ثابت عائم * src_لیکه = (ثابت عائم *) src0_معلومات + لیکه_شاخص * ne00 + batch_دننه_واټن;
            عائم *       dst_لیکه = dst_معلومات + dst_دننه_واټن * ne00;
            copy_f32_لیکه(dst_لیکه, src_لیکه, ne00);
        } وإلا إذا (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion
            ثابت uint16_t * src_لیکه = (ثابت uint16_t *) src0_معلومات + لیکه_شاخص * ne00 + batch_دننه_واټن;
            عائم *          dst_لیکه = dst_معلومات + dst_دننه_واټن * ne00;
            copy_f16_لیکه(dst_لیکه, src_لیکه, ne00);
        } وإلا إذا (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize while copying
            ثابت int64_t      بلاکونه_فی_لیکه   = (ne00 + QK8_0 - 1) / QK8_0;
            ثابت int64_t      src_بلاک_دننه_واټن = (لیکه_شاخص * بلاکونه_فی_لیکه) + (batch_دننه_واټن / ne00) * بلاکونه_فی_لیکه;
            ثابت بلاک_q8_0 * src_بلاکونه       = (ثابت بلاک_q8_0 *) src0_معلومات + src_بلاک_دننه_واټن;
            عائم *            dst_لیکه          = dst_معلومات + dst_دننه_واټن * ne00;
            copy_q8_0_لیکه(dst_لیکه, src_بلاکونه, ne00);
        } وإلا إذا (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize while copying
            ثابت int64_t      بلاکونه_فی_لیکه   = (ne00 + QK4_0 - 1) / QK4_0;
            ثابت int64_t      src_بلاک_دننه_واټن = (لیکه_شاخص * بلاکونه_فی_لیکه) + (batch_دننه_واټن / ne00) * بلاکونه_فی_لیکه;
            ثابت بلاک_q4_0 * src_بلاکونه       = (ثابت بلاک_q4_0 *) src0_معلومات + src_بلاک_دننه_واټن;
            عائم *            dst_لیکه          = dst_معلومات + dst_دننه_واټن * ne00;
            copy_q4_0_لیکه(dst_لیکه, src_بلاکونه, ne00);
        } وإلا إذا (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize while copying
            ثابت int64_t      بلاکونه_فی_لیکه   = (ne00 + QK_K - 1) / QK_K;
            ثابت int64_t      src_بلاک_دننه_واټن = (لیکه_شاخص * بلاکونه_فی_لیکه) + (batch_دننه_واټن / ne00) * بلاکونه_فی_لیکه;
            ثابت بلاک_q4_K * src_بلاکونه       = (ثابت بلاک_q4_K *) src0_معلومات + src_بلاک_دننه_واټن;
            عائم *            dst_لیکه          = dst_معلومات + dst_دننه_واټن * ne00;
            copy_q4_K_لیکه(dst_لیکه, src_بلاکونه, ne00);
        }
    }

    أرجع 0;
}
