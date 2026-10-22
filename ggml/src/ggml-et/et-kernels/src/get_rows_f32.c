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

#شامل "ggml_tensor.h"
#شامل "platform.h"
#شامل "quants.h"

#شامل <assert.h>
#شامل <stdbool.h>
#شامل <stdint.h>

#تعریف_کرو CACHE_سطر_سائز_بائٹیں 64

ساخت ggml_et_get_صفیں_params {
    ساخت ggml_tensor src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    ساخت ggml_tensor src1;  // Row indices tensor (I32)
    ساخت ggml_tensor dst;   // Output tensor (F32)
};

#تعریف_کرو CACHE_سطر_سائز_بائٹیں     64
#تعریف_کرو CACHE_ELEMENTS(elem_سائز) (CACHE_سطر_سائز_بائٹیں / (elem_سائز))

// Copy a row of F32 data from source to destination
ساکن خالی copy_f32_صف(اعشاری * dst, ثابت اعشاری * src, int64_t عدد_elements) {
    // Simple memcpy for F32 data - no conversion needed
    برائے (int64_t i = 0; i < عدد_elements; i++) {
        dst[i] = src[i];
    }
}

ساکن خالی copy_f16_صف(اعشاری * dst, ثابت uint16_t * src, int64_t عدد_elements) {
    برائے (int64_t i = 0; i < عدد_elements; i++) {
        dst[i] = fp16_مقصد_fp32(src[i]);
    }
}

// Copy a row of F32 data from source to destination, aligned to cache line boundaries
// using FP32 load/store instructions. They don't perform data conversion so is fine.
// Requirement: n_bytes is a multiple of CACHE_LINE_SIZE (64 bytes)
ساکن خالی copy_صف_cache_align(اعشاری * dst, ثابت اعشاری * src, int64_t n_بائٹیں) {
    صحیح عدد_f32_elem = n_بائٹیں / حجم(اعشاری);

    // Unrolled to do an entire cache line at a time
    __asm__ متلون(
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
        : [stride_گنتی] "r"(16L)
        : "f0", "f1", "memory");
}

// Copied from GGML: copy a row of Q4_0 data to F32 destination (with dequantization)
ساکن خالی copy_q4_0_صف(اعشاری * dst, ثابت بلاک_q4_0 * src_بلاکیں, int64_t عدد_elements) {
    ثابت int64_t عدد_بلاکیں = (عدد_elements + QK4_0 - 1) / QK4_0;

    برائے (int64_t بلاک_idx = 0; بلاک_idx < عدد_بلاکیں; بلاک_idx++) {
        ثابت int64_t elements_اندر_بلاک = (بلاک_idx == عدد_بلاکیں - 1) ? (عدد_elements - بلاک_idx * QK4_0) : QK4_0;

        اعشاری عارضی_بفر[QK4_0];
        dequantize_q4_0_بلاک(&src_بلاکیں[بلاک_idx], عارضی_بفر);

        برائے (int64_t i = 0; i < elements_اندر_بلاک; i++) {
            dst[بلاک_idx * QK4_0 + i] = عارضی_بفر[i];
        }
    }
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
ساکن خالی copy_q8_0_صف(اعشاری * dst, ثابت بلاک_q8_0 * src_بلاکیں, int64_t عدد_elements) {
    // Number of Q8_0 blocks needed for this row
    ثابت int64_t عدد_بلاکیں = (عدد_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    برائے (int64_t بلاک_idx = 0; بلاک_idx < عدد_بلاکیں; بلاک_idx++) {
        ثابت int64_t elements_اندر_بلاک =
            (بلاک_idx == عدد_بلاکیں - 1) ? (عدد_elements - بلاک_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        اعشاری عارضی_بفر[QK8_0];
        dequantize_q8_0_بلاک(&src_بلاکیں[بلاک_idx], عارضی_بفر);

        // Copy dequantized values to destination
        برائے (int64_t i = 0; i < elements_اندر_بلاک; i++) {
            dst[بلاک_idx * QK8_0 + i] = عارضی_بفر[i];
        }
    }
}

// Copy a row of Q4_K data to F32 destination (with dequantization)
ساکن خالی copy_q4_K_صف(اعشاری * dst, ثابت بلاک_q4_K * src_بلاکیں, int64_t عدد_elements) {
    ثابت int64_t عدد_بلاکیں = (عدد_elements + QK_K - 1) / QK_K;

    برائے (int64_t بلاک_idx = 0; بلاک_idx < عدد_بلاکیں; بلاک_idx++) {
        ثابت int64_t elements_اندر_بلاک = (بلاک_idx == عدد_بلاکیں - 1) ? (عدد_elements - بلاک_idx * QK_K) : QK_K;

        اعشاری عارضی_بفر[QK_K];
        dequantize_q4_K_بلاک(&src_بلاکیں[بلاک_idx], عارضی_بفر);

        برائے (int64_t i = 0; i < elements_اندر_بلاک; i++) {
            dst[بلاک_idx * QK_K + i] = عارضی_بفر[i];
        }
    }
}

ساکن خالی dequantize_q8_0_بلاک_cache_aligned(ثابت بلاک_q8_0 * بلاک, اعشاری * dst) {
    ثابت int8_t * qs_ptr = بلاک->qs;

    uint64_t عارضی_نقاب;
    __asm__ متلون("mova.x.m %0" : "=r"(عارضی_نقاب));  // Save current mask
    __asm__ متلون("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    ثابت int32_t __attribute__((aligned(32))) vec_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    اعشاری                                      scale          = fp16_مقصد_fp32(بلاک->d);
    __asm__ متلون(
        "fbcx.ps     f0, %0       \n\t"  // Broadcast integer scale to all lanes
        "flq2        f1, 0(%1)    \n\t"  // Load gether indicies
        ::"r"(scale),
        "r"(vec_indices)
        : "f0", "f1");

    برائے (صحیح i = 0; i < 4; i++) {
        __asm__ متلون(
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
    __asm__ متلون("mova.m.x %0" ::"r"(عارضی_نقاب));
}

// Copy a row of Q4_0 data to F32 destination (with dequantization), cache-aligned
ساکن خالی copy_q4_0_صف_cache_aligned(اعشاری * dst, ثابت بلاک_q4_0 * src_بلاکیں, int64_t عدد_elements) {
    ثابت int64_t عدد_بلاکیں = (عدد_elements + QK4_0 - 1) / QK4_0;

    // Scatter byte offsets: even lanes -> dst[j], odd lanes -> dst[j + QK4_0/2]
    // For 4 consecutive packed bytes producing [low0, high0, low1, high1, low2, high2, low3, high3]:
    //   low_i  -> byte offset i*4       (positions 0,1,2,3 in first half)
    //   high_i -> byte offset (16+i)*4  (positions 16,17,18,19 in second half)
    ثابت int32_t __attribute__((aligned(32))) scatter_offsets[8] = { 0 * 4, 16 * 4, 1 * 4, 17 * 4,
                                                                      2 * 4, 18 * 4, 3 * 4, 19 * 4 };

    // Gather indices: each byte loaded twice for low/high nibble extraction
    ثابت int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 0, 1, 1, 2, 2, 3, 3 };

    uint64_t عارضی_نقاب;
    __asm__ متلون("mova.x.m %0" : "=r"(عارضی_نقاب));  // Save current mask
    __asm__ متلون("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load constant vectors once — shared across all blocks and iterations
    __asm__ متلون(
        "flq2        f4, 0(%0)    \n\t"  // f4 = scatter offsets
        "flq2        f1, 0(%1)    \n\t"  // f1 = gather indices {0,0,1,1,2,2,3,3}
        ::"r"(scatter_offsets),
        "r"(gather_indices)
        : "f1", "f4");

    برائے (int64_t بلاک_idx = 0; بلاک_idx < عدد_بلاکیں; بلاک_idx++) {
        ثابت بلاک_q4_0 * بلاک     = &src_بلاکیں[بلاک_idx];
        ثابت uint8_t *    qs        = بلاک->qs;
        اعشاری *            بلاک_dst = dst + بلاک_idx * QK4_0;

        اعشاری scale = fp16_مقصد_fp32(بلاک->d);
        اعشاری bias  = -8.0f * scale;

        // Per-block: broadcast scale and bias
        __asm__ متلون(
            "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(scale)
            "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-8 * scale)
            ::"r"(scale),
            "r"(bias)
            : "f0", "f3");

        // 4 iterations x 4 packed bytes = 16 bytes = full block -> 32 floats
        برائے (صحیح i = 0; i < 4; i++) {
            __asm__ متلون(
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

    __asm__ متلون("mova.m.x %0" ::"r"(عارضی_نقاب));  // Restore mask
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
ساکن خالی copy_q8_0_صف_cache_aligned(اعشاری * dst, ثابت بلاک_q8_0 * src_بلاکیں, int64_t عدد_elements) {
    // Number of Q8_0 blocks needed for this row
    ثابت int64_t عدد_بلاکیں = (عدد_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    برائے (int64_t بلاک_idx = 0; بلاک_idx < عدد_بلاکیں; بلاک_idx++) {
        ثابت int64_t elements_اندر_بلاک =
            (بلاک_idx == عدد_بلاکیں - 1) ? (عدد_elements - بلاک_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        اعشاری عارضی_بفر[QK8_0];
        dequantize_q8_0_بلاک_cache_aligned(&src_بلاکیں[بلاک_idx], عارضی_بفر);

        // Copy dequantized values to destination
        برائے (int64_t i = 0; i < elements_اندر_بلاک; i++) {
            dst[بلاک_idx * QK8_0 + i] = عارضی_بفر[i];
        }
    }
}

// Vectorized dequantization of a Q4_K super-block (256 elements) to F32
// Processes 8 groups of 32 elements, using ET SIMD for the inner loops.
// Output is sequential (no scatter needed unlike Q4_0).
ساکن خالی copy_q4_K_صف_cache_aligned(اعشاری * dst, ثابت بلاک_q4_K * src_بلاکیں, int64_t عدد_elements) {
    ثابت int64_t عدد_بلاکیں = (عدد_elements + QK_K - 1) / QK_K;

    // Gather indices for sequential byte access: {0,1,2,3,4,5,6,7}
    ثابت int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    uint64_t عارضی_نقاب;
    __asm__ متلون("mova.x.m %0" : "=r"(عارضی_نقاب));  // Save current mask
    __asm__ متلون("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load gather indices once — shared across all blocks
    __asm__ متلون("flq2        f1, 0(%0)    \n\t"  // f1 = gather indices {0,1,2,3,4,5,6,7}
                     ::"r"(gather_indices)
                     : "f1");

    برائے (int64_t بلاک_idx = 0; بلاک_idx < عدد_بلاکیں; بلاک_idx++) {
        ثابت بلاک_q4_K * بلاک     = &src_بلاکیں[بلاک_idx];
        ثابت uint8_t *    qs        = بلاک->qs;
        اعشاری *            بلاک_dst = dst + بلاک_idx * QK_K;

        ثابت اعشاری d   = fp16_مقصد_fp32(بلاک->d);
        ثابت اعشاری min = fp16_مقصد_fp32(بلاک->dmin);

        صحیح is = 0;
        برائے (صحیح j = 0; j < QK_K; j += 64) {
            // Extract per-group scales and mins (scalar — only 8 pairs per super-block)
            uint8_t sc, m;
            get_scale_min_k4(is + 0, بلاک->scales, &sc, &m);
            ثابت اعشاری d1     = d * sc;
            ثابت اعشاری neg_m1 = -(min * m);
            get_scale_min_k4(is + 1, بلاک->scales, &sc, &m);
            ثابت اعشاری d2     = d * sc;
            ثابت اعشاری neg_m2 = -(min * m);

            // Low nibbles: 32 elements using d1, neg_m1
            __asm__ متلون(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d1)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m1)
                ::"r"(d1),
                "r"(neg_m1)
                : "f0", "f3");

            ثابت uint8_t * qs_lo  = qs;
            اعشاری *         dst_lo = بلاک_dst + j;
            برائے (صحیح k = 0; k < 4; k++) {
                __asm__ متلون(
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
            __asm__ متلون(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d2)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m2)
                ::"r"(d2),
                "r"(neg_m2)
                : "f0", "f3");

            ثابت uint8_t * qs_hi  = qs;
            اعشاری *         dst_hi = بلاک_dst + j + 32;
            برائے (صحیح k = 0; k < 4; k++) {
                __asm__ متلون(
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

    __asm__ متلون("mova.m.x %0" ::"r"(عارضی_نقاب));  // Restore mask
}

// Determine the number of F32 elements per work unit for a given source type.
// For F32: 1 cacheline (16 elements)
// For quantized types: 1 quant block
ساکن int64_t get_elements_فی_work_unit(صحیح type) {
    ثابت int64_t elements_فی_cacheline = CACHE_سطر_سائز_بائٹیں / حجم(اعشاری);  // 16
    انتخاب (type) {
        حالت GGML_TYPE_Q8_0:
            واپس QK8_0;                   // 32 elements = 2 cachelines
        حالت GGML_TYPE_Q4_0:
            واپس QK4_0;                   // 32 elements = 2 cachelines
        حالت GGML_TYPE_Q4_K:
            واپس QK_K;                    // 256 elements = 16 cachelines
        طےشدہ:
            واپس elements_فی_cacheline;  // 16 elements = 1 cacheline
    }
}

ساکن صحیح get_صف_f32_mc_cacheline_aligned(ساخت ggml_et_get_صفیں_params * params, خالی * env) {
    kernel_environment_t * kernel_env  = (kernel_environment_t *) env;
    صحیح                    thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح                    عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    ساخت ggml_tensor * src0 = &params->src0;  // Data tensor
    ساخت ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    ساخت ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    ثابت int64_t ne00 = src0->ne[0];           // Source columns (row width)
    ثابت int64_t ne01 = src0->ne[1];           // Source rows (total available rows)
    ثابت int64_t ne02 = src0->ne[2];           // Source batch dimension
    ثابت int64_t ne03 = src0->ne[3];           // Source outer batch dimension

    ثابت int64_t ne10 = src1->ne[0];           // Number of indices in dimension 0
    ثابت int64_t ne11 = src1->ne[1];           // Number of indices in dimension 1
    ثابت int64_t ne12 = src1->ne[2];           // Batch dimension for indices
    ثابت int64_t ne13 = src1->ne[3];           // Outer batch dimension for indices

    ثابت int64_t کل_صفیں_مقصد_نکالو = ne10 * ne11 * ne12 * ne13;

    // Determine work unit size based on source type
    ثابت int64_t elements_فی_wu = get_elements_فی_work_unit(src0->type);
    ثابت int64_t wus_فی_صف     = ne00 / elements_فی_wu;
    ثابت int64_t کل_wus       = کل_صفیں_مقصد_نکالو * wus_فی_صف;

    // Distribute work units across threads (contiguous ranges)
    ثابت int64_t wus_فی_thread = (کل_wus + عدد_دھاگے - 1) / عدد_دھاگے;
    ثابت int64_t wu_آغاز       = thread_id * wus_فی_thread;
    int64_t       wu_انتہا         = wu_آغاز + wus_فی_thread;
    اگر (wu_انتہا > کل_wus) {
        wu_انتہا = کل_wus;
    }

    خالی *    src0_ڈیٹا = src0->ڈیٹا;
    int32_t * src1_ڈیٹا = (int32_t *) src1->ڈیٹا;
    اعشاری *   dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    int64_t wu = wu_آغاز;
    جب (wu < wu_انتہا) {
        // Determine which row this work unit belongs to and offset within row
        ثابت int64_t صف_idx   = wu / wus_فی_صف;
        ثابت int64_t wu_اندر_صف = wu % wus_فی_صف;

        // How many work units to process in this row (batch contiguous WUs in same row)
        int64_t wus_بقایا_اندر_صف = wus_فی_صف - wu_اندر_صف;
        int64_t wus_مقصد_process       = wu_انتہا - wu;
        اگر (wus_بقایا_اندر_صف < wus_مقصد_process) {
            wus_مقصد_process = wus_بقایا_اندر_صف;
        }

        // Calculate multi-dimensional index for this row
        ثابت int64_t i       = صف_idx;
        ثابت int64_t i13_idx = i / (ne12 * ne11 * ne10);
        ثابت int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        ثابت int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        ثابت int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        ثابت int64_t اشاریہ_انحراف = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        ثابت int32_t صف_اشاریہ    = src1_ڈیٹا[اشاریہ_انحراف];

        اگر (صف_اشاریہ < 0 || صف_اشاریہ >= ne01) {
            واپس -1;  // Index out of bounds
        }

        ثابت int64_t batch_انحراف =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        ثابت int64_t elem_انحراف_اندر_صف = wu_اندر_صف * elements_فی_wu;
        ثابت int64_t عدد_elements       = wus_مقصد_process * elements_فی_wu;

        اعشاری * dst_صف = dst_ڈیٹا + صف_idx * ne00 + elem_انحراف_اندر_صف;

        اگر (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy of cacheline-aligned chunk
            ثابت اعشاری * src_صف = (ثابت اعشاری *) src0_ڈیٹا + صف_اشاریہ * ne00 + batch_انحراف + elem_انحراف_اندر_صف;
            copy_صف_cache_align(dst_صف, src_صف, عدد_elements * حجم(اعشاری));
        } ورنہ اگر (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion over a destination-aligned write chunk.
            ثابت uint16_t * src_صف =
                (ثابت uint16_t *) src0_ڈیٹا + صف_اشاریہ * ne00 + batch_انحراف + elem_انحراف_اندر_صف;
            copy_f16_صف(dst_صف, src_صف, عدد_elements);
        } ورنہ اگر (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize work-unit-aligned blocks
            ثابت int64_t      بلاکیں_فی_صف   = (ne00 + QK8_0 - 1) / QK8_0;
            ثابت int64_t      src_بلاک_انحراف = (صف_اشاریہ * بلاکیں_فی_صف) + (batch_انحراف / ne00) * بلاکیں_فی_صف;
            ثابت int64_t      بلاک_آغاز      = elem_انحراف_اندر_صف / QK8_0;
            ثابت بلاک_q8_0 * src_بلاکیں       = (ثابت بلاک_q8_0 *) src0_ڈیٹا + src_بلاک_انحراف + بلاک_آغاز;
            copy_q8_0_صف_cache_aligned(dst_صف, src_بلاکیں, عدد_elements);
        } ورنہ اگر (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize work-unit-aligned blocks
            ثابت int64_t      بلاکیں_فی_صف   = (ne00 + QK4_0 - 1) / QK4_0;
            ثابت int64_t      src_بلاک_انحراف = (صف_اشاریہ * بلاکیں_فی_صف) + (batch_انحراف / ne00) * بلاکیں_فی_صف;
            ثابت int64_t      بلاک_آغاز      = elem_انحراف_اندر_صف / QK4_0;
            ثابت بلاک_q4_0 * src_بلاکیں       = (ثابت بلاک_q4_0 *) src0_ڈیٹا + src_بلاک_انحراف + بلاک_آغاز;
            copy_q4_0_صف_cache_aligned(dst_صف, src_بلاکیں, عدد_elements);
        } ورنہ اگر (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize work-unit-aligned blocks
            ثابت int64_t      بلاکیں_فی_صف   = (ne00 + QK_K - 1) / QK_K;
            ثابت int64_t      src_بلاک_انحراف = (صف_اشاریہ * بلاکیں_فی_صف) + (batch_انحراف / ne00) * بلاکیں_فی_صف;
            ثابت int64_t      بلاک_آغاز      = elem_انحراف_اندر_صف / QK_K;
            ثابت بلاک_q4_K * src_بلاکیں       = (ثابت بلاک_q4_K *) src0_ڈیٹا + src_بلاک_انحراف + بلاک_آغاز;
            copy_q4_K_صف_cache_aligned(dst_صف, src_بلاکیں, عدد_elements);
        }

        wu += wus_مقصد_process;
    }

    واپس 0;
}

ساکن اندرونی size_t tensor_بائٹیں(ثابت ساخت ggml_tensor * t) {
    واپس (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

صحیح اندراج_نقطہ(ساخت ggml_et_get_صفیں_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    اگر (!kernel_env) {
        واپس -1;
    }

    ساخت ggml_tensor * src0 = &params->src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    ساخت ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    ساخت ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    // Fast path - we know how to deal with them multi-core
    اگر ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_Q8_0 ||
         src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_K) &&
        src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32 && dst->ne[0] % CACHE_ELEMENTS(حجم(اعشاری)) == 0) {
        واپس get_صف_f32_mc_cacheline_aligned(params, env);
    }

    صحیح thread_id = get_relative_thread_id(kernel_env->shire_نقاب);
    اگر (thread_id < 0) {
        واپس 0;
    }

    اگر (thread_id != 0) {
        واپس 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;  // Invalid pointer
    }

    اگر (dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        واپس -1;  // Invalid output or index type
    }

    اگر (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q4_K) {
        واپس -1;  // Unsupported input type
    }

    خالی *    src0_ڈیٹا = src0->ڈیٹا;
    int32_t * src1_ڈیٹا = (int32_t *) src1->ڈیٹا;
    اعشاری *   dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;
#اگر_تعریف_شدہ ET_UBERKERNEL
    evict_region_past_l2(src0_ڈیٹا, tensor_بائٹیں(src0));
    evict_region_past_l2(src1_ڈیٹا, tensor_بائٹیں(src1));
    evict_region_past_l2(dst_ڈیٹا, tensor_بائٹیں(dst));
#اگر_ختم

    اگر (!src0_ڈیٹا || !src1_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;  // Null data pointer
    }

    ثابت int64_t ne00 = src0->ne[0];  // Source columns (row width)
    ثابت int64_t ne01 = src0->ne[1];  // Source rows (total available rows)
    ثابت int64_t ne02 = src0->ne[2];  // Source batch dimension
    ثابت int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    ثابت int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    ثابت int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    ثابت int64_t ne12 = src1->ne[2];  // Batch dimension for indices
    ثابت int64_t ne13 = src1->ne[3];  // Outer batch dimension for indices

    ثابت int64_t کل_صفیں_مقصد_نکالو = ne10 * ne11 * ne12 * ne13;
#اگر_تعریف_شدہ ET_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
#اگر_ختم
    // Naive single-threaded implementation - process all rows sequentially
    // XXX: Do we really need a single-threaded implementation?
    برائے (int64_t i = 0; i < کل_صفیں_مقصد_نکالو; i++) {
        // Calculate multi-dimensional index for the current output position
        ثابت int64_t i13_idx = i / (ne12 * ne11 * ne10);
        ثابت int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        ثابت int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        ثابت int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        ثابت int64_t اشاریہ_انحراف = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        ثابت int32_t صف_اشاریہ    = src1_ڈیٹا[اشاریہ_انحراف];

        اگر (صف_اشاریہ < 0 || صف_اشاریہ >= ne01) {
            واپس -1;  // Index out of bounds
        }

        ثابت int64_t batch_انحراف =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        ثابت int64_t dst_انحراف = i;

        اگر (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy
            ثابت اعشاری * src_صف = (ثابت اعشاری *) src0_ڈیٹا + صف_اشاریہ * ne00 + batch_انحراف;
            اعشاری *       dst_صف = dst_ڈیٹا + dst_انحراف * ne00;
            copy_f32_صف(dst_صف, src_صف, ne00);
        } ورنہ اگر (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion
            ثابت uint16_t * src_صف = (ثابت uint16_t *) src0_ڈیٹا + صف_اشاریہ * ne00 + batch_انحراف;
            اعشاری *          dst_صف = dst_ڈیٹا + dst_انحراف * ne00;
            copy_f16_صف(dst_صف, src_صف, ne00);
        } ورنہ اگر (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize while copying
            ثابت int64_t      بلاکیں_فی_صف   = (ne00 + QK8_0 - 1) / QK8_0;
            ثابت int64_t      src_بلاک_انحراف = (صف_اشاریہ * بلاکیں_فی_صف) + (batch_انحراف / ne00) * بلاکیں_فی_صف;
            ثابت بلاک_q8_0 * src_بلاکیں       = (ثابت بلاک_q8_0 *) src0_ڈیٹا + src_بلاک_انحراف;
            اعشاری *            dst_صف          = dst_ڈیٹا + dst_انحراف * ne00;
            copy_q8_0_صف(dst_صف, src_بلاکیں, ne00);
        } ورنہ اگر (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize while copying
            ثابت int64_t      بلاکیں_فی_صف   = (ne00 + QK4_0 - 1) / QK4_0;
            ثابت int64_t      src_بلاک_انحراف = (صف_اشاریہ * بلاکیں_فی_صف) + (batch_انحراف / ne00) * بلاکیں_فی_صف;
            ثابت بلاک_q4_0 * src_بلاکیں       = (ثابت بلاک_q4_0 *) src0_ڈیٹا + src_بلاک_انحراف;
            اعشاری *            dst_صف          = dst_ڈیٹا + dst_انحراف * ne00;
            copy_q4_0_صف(dst_صف, src_بلاکیں, ne00);
        } ورنہ اگر (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize while copying
            ثابت int64_t      بلاکیں_فی_صف   = (ne00 + QK_K - 1) / QK_K;
            ثابت int64_t      src_بلاک_انحراف = (صف_اشاریہ * بلاکیں_فی_صف) + (batch_انحراف / ne00) * بلاکیں_فی_صف;
            ثابت بلاک_q4_K * src_بلاکیں       = (ثابت بلاک_q4_K *) src0_ڈیٹا + src_بلاک_انحراف;
            اعشاری *            dst_صف          = dst_ڈیٹا + dst_انحراف * ne00;
            copy_q4_K_صف(dst_صف, src_بلاکیں, ne00);
        }
    }

    واپس 0;
}
