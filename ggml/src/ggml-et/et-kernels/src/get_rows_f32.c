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

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "platform.h"
#అన్తర్భావ "quants.h"

#అన్తర్భావ <assert.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>

#నిర్వచించు CACHE_పంక్తి_కొలత_బైట్లు 64

నిర్మాణం ggml_et_get_అడ్డువరుసలు_params {
    నిర్మాణం ggml_tensor src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    నిర్మాణం ggml_tensor src1;  // Row indices tensor (I32)
    నిర్మాణం ggml_tensor dst;   // Output tensor (F32)
};

#నిర్వచించు CACHE_పంక్తి_కొలత_బైట్లు     64
#నిర్వచించు CACHE_ELEMENTS(elem_కొలత) (CACHE_పంక్తి_కొలత_బైట్లు / (elem_కొలత))

// Copy a row of F32 data from source to destination
స్థిర శూన్య copy_f32_అడ్డువరుస(దశాంశం * dst, స్థిరరాశి దశాంశం * src, int64_t అంకె_elements) {
    // Simple memcpy for F32 data - no conversion needed
    కొరకు (int64_t i = 0; i < అంకె_elements; i++) {
        dst[i] = src[i];
    }
}

స్థిర శూన్య copy_f16_అడ్డువరుస(దశాంశం * dst, స్థిరరాశి uint16_t * src, int64_t అంకె_elements) {
    కొరకు (int64_t i = 0; i < అంకె_elements; i++) {
        dst[i] = fp16_గమ్యం_fp32(src[i]);
    }
}

// Copy a row of F32 data from source to destination, aligned to cache line boundaries
// using FP32 load/store instructions. They don't perform data conversion so is fine.
// Requirement: n_bytes is a multiple of CACHE_LINE_SIZE (64 bytes)
స్థిర శూన్య copy_అడ్డువరుస_cache_align(దశాంశం * dst, స్థిరరాశి దశాంశం * src, int64_t n_బైట్లు) {
    పూర్ణాంకం అంకె_f32_elem = n_బైట్లు / పరిమాణం(దశాంశం);

    // Unrolled to do an entire cache line at a time
    __asm__ అస్థిర(
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

        : [dst] "+r"(dst), [src] "+r"(src), [n] "+r"(అంకె_f32_elem)
        : [stride_లెక్క] "r"(16L)
        : "f0", "f1", "memory");
}

// Copied from GGML: copy a row of Q4_0 data to F32 destination (with dequantization)
స్థిర శూన్య copy_q4_0_అడ్డువరుస(దశాంశం * dst, స్థిరరాశి ఖండం_q4_0 * src_ఖండాలు, int64_t అంకె_elements) {
    స్థిరరాశి int64_t అంకె_ఖండాలు = (అంకె_elements + QK4_0 - 1) / QK4_0;

    కొరకు (int64_t ఖండం_idx = 0; ఖండం_idx < అంకె_ఖండాలు; ఖండం_idx++) {
        స్థిరరాశి int64_t elements_లోపల_ఖండం = (ఖండం_idx == అంకె_ఖండాలు - 1) ? (అంకె_elements - ఖండం_idx * QK4_0) : QK4_0;

        దశాంశం తాత్కాలిక_బఫరు[QK4_0];
        dequantize_q4_0_ఖండం(&src_ఖండాలు[ఖండం_idx], తాత్కాలిక_బఫరు);

        కొరకు (int64_t i = 0; i < elements_లోపల_ఖండం; i++) {
            dst[ఖండం_idx * QK4_0 + i] = తాత్కాలిక_బఫరు[i];
        }
    }
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
స్థిర శూన్య copy_q8_0_అడ్డువరుస(దశాంశం * dst, స్థిరరాశి ఖండం_q8_0 * src_ఖండాలు, int64_t అంకె_elements) {
    // Number of Q8_0 blocks needed for this row
    స్థిరరాశి int64_t అంకె_ఖండాలు = (అంకె_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    కొరకు (int64_t ఖండం_idx = 0; ఖండం_idx < అంకె_ఖండాలు; ఖండం_idx++) {
        స్థిరరాశి int64_t elements_లోపల_ఖండం =
            (ఖండం_idx == అంకె_ఖండాలు - 1) ? (అంకె_elements - ఖండం_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        దశాంశం తాత్కాలిక_బఫరు[QK8_0];
        dequantize_q8_0_ఖండం(&src_ఖండాలు[ఖండం_idx], తాత్కాలిక_బఫరు);

        // Copy dequantized values to destination
        కొరకు (int64_t i = 0; i < elements_లోపల_ఖండం; i++) {
            dst[ఖండం_idx * QK8_0 + i] = తాత్కాలిక_బఫరు[i];
        }
    }
}

// Copy a row of Q4_K data to F32 destination (with dequantization)
స్థిర శూన్య copy_q4_K_అడ్డువరుస(దశాంశం * dst, స్థిరరాశి ఖండం_q4_K * src_ఖండాలు, int64_t అంకె_elements) {
    స్థిరరాశి int64_t అంకె_ఖండాలు = (అంకె_elements + QK_K - 1) / QK_K;

    కొరకు (int64_t ఖండం_idx = 0; ఖండం_idx < అంకె_ఖండాలు; ఖండం_idx++) {
        స్థిరరాశి int64_t elements_లోపల_ఖండం = (ఖండం_idx == అంకె_ఖండాలు - 1) ? (అంకె_elements - ఖండం_idx * QK_K) : QK_K;

        దశాంశం తాత్కాలిక_బఫరు[QK_K];
        dequantize_q4_K_ఖండం(&src_ఖండాలు[ఖండం_idx], తాత్కాలిక_బఫరు);

        కొరకు (int64_t i = 0; i < elements_లోపల_ఖండం; i++) {
            dst[ఖండం_idx * QK_K + i] = తాత్కాలిక_బఫరు[i];
        }
    }
}

స్థిర శూన్య dequantize_q8_0_ఖండం_cache_aligned(స్థిరరాశి ఖండం_q8_0 * ఖండం, దశాంశం * dst) {
    స్థిరరాశి int8_t * qs_ptr = ఖండం->qs;

    uint64_t తాత్కాలిక_ముసుగు;
    __asm__ అస్థిర("mova.x.m %0" : "=r"(తాత్కాలిక_ముసుగు));  // Save current mask
    __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    స్థిరరాశి int32_t __attribute__((aligned(32))) vec_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    దశాంశం                                      scale          = fp16_గమ్యం_fp32(ఖండం->d);
    __asm__ అస్థిర(
        "fbcx.ps     f0, %0       \n\t"  // Broadcast integer scale to all lanes
        "flq2        f1, 0(%1)    \n\t"  // Load gether indicies
        ::"r"(scale),
        "r"(vec_indices)
        : "f0", "f1");

    కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
        __asm__ అస్థిర(
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
    __asm__ అస్థిర("mova.m.x %0" ::"r"(తాత్కాలిక_ముసుగు));
}

// Copy a row of Q4_0 data to F32 destination (with dequantization), cache-aligned
స్థిర శూన్య copy_q4_0_అడ్డువరుస_cache_aligned(దశాంశం * dst, స్థిరరాశి ఖండం_q4_0 * src_ఖండాలు, int64_t అంకె_elements) {
    స్థిరరాశి int64_t అంకె_ఖండాలు = (అంకె_elements + QK4_0 - 1) / QK4_0;

    // Scatter byte offsets: even lanes -> dst[j], odd lanes -> dst[j + QK4_0/2]
    // For 4 consecutive packed bytes producing [low0, high0, low1, high1, low2, high2, low3, high3]:
    //   low_i  -> byte offset i*4       (positions 0,1,2,3 in first half)
    //   high_i -> byte offset (16+i)*4  (positions 16,17,18,19 in second half)
    స్థిరరాశి int32_t __attribute__((aligned(32))) scatter_offsets[8] = { 0 * 4, 16 * 4, 1 * 4, 17 * 4,
                                                                      2 * 4, 18 * 4, 3 * 4, 19 * 4 };

    // Gather indices: each byte loaded twice for low/high nibble extraction
    స్థిరరాశి int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 0, 1, 1, 2, 2, 3, 3 };

    uint64_t తాత్కాలిక_ముసుగు;
    __asm__ అస్థిర("mova.x.m %0" : "=r"(తాత్కాలిక_ముసుగు));  // Save current mask
    __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load constant vectors once — shared across all blocks and iterations
    __asm__ అస్థిర(
        "flq2        f4, 0(%0)    \n\t"  // f4 = scatter offsets
        "flq2        f1, 0(%1)    \n\t"  // f1 = gather indices {0,0,1,1,2,2,3,3}
        ::"r"(scatter_offsets),
        "r"(gather_indices)
        : "f1", "f4");

    కొరకు (int64_t ఖండం_idx = 0; ఖండం_idx < అంకె_ఖండాలు; ఖండం_idx++) {
        స్థిరరాశి ఖండం_q4_0 * ఖండం     = &src_ఖండాలు[ఖండం_idx];
        స్థిరరాశి uint8_t *    qs        = ఖండం->qs;
        దశాంశం *            ఖండం_dst = dst + ఖండం_idx * QK4_0;

        దశాంశం scale = fp16_గమ్యం_fp32(ఖండం->d);
        దశాంశం bias  = -8.0f * scale;

        // Per-block: broadcast scale and bias
        __asm__ అస్థిర(
            "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(scale)
            "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-8 * scale)
            ::"r"(scale),
            "r"(bias)
            : "f0", "f3");

        // 4 iterations x 4 packed bytes = 16 bytes = full block -> 32 floats
        కొరకు (పూర్ణాంకం i = 0; i < 4; i++) {
            __asm__ అస్థిర(
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
                "r"(ఖండం_dst)
                : "f2", "memory");

            qs += 4;         // 4 packed bytes consumed
            ఖండం_dst += 4;  // Advance base by 4 float positions
        }
    }

    __asm__ అస్థిర("mova.m.x %0" ::"r"(తాత్కాలిక_ముసుగు));  // Restore mask
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
స్థిర శూన్య copy_q8_0_అడ్డువరుస_cache_aligned(దశాంశం * dst, స్థిరరాశి ఖండం_q8_0 * src_ఖండాలు, int64_t అంకె_elements) {
    // Number of Q8_0 blocks needed for this row
    స్థిరరాశి int64_t అంకె_ఖండాలు = (అంకె_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    కొరకు (int64_t ఖండం_idx = 0; ఖండం_idx < అంకె_ఖండాలు; ఖండం_idx++) {
        స్థిరరాశి int64_t elements_లోపల_ఖండం =
            (ఖండం_idx == అంకె_ఖండాలు - 1) ? (అంకె_elements - ఖండం_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        దశాంశం తాత్కాలిక_బఫరు[QK8_0];
        dequantize_q8_0_ఖండం_cache_aligned(&src_ఖండాలు[ఖండం_idx], తాత్కాలిక_బఫరు);

        // Copy dequantized values to destination
        కొరకు (int64_t i = 0; i < elements_లోపల_ఖండం; i++) {
            dst[ఖండం_idx * QK8_0 + i] = తాత్కాలిక_బఫరు[i];
        }
    }
}

// Vectorized dequantization of a Q4_K super-block (256 elements) to F32
// Processes 8 groups of 32 elements, using ET SIMD for the inner loops.
// Output is sequential (no scatter needed unlike Q4_0).
స్థిర శూన్య copy_q4_K_అడ్డువరుస_cache_aligned(దశాంశం * dst, స్థిరరాశి ఖండం_q4_K * src_ఖండాలు, int64_t అంకె_elements) {
    స్థిరరాశి int64_t అంకె_ఖండాలు = (అంకె_elements + QK_K - 1) / QK_K;

    // Gather indices for sequential byte access: {0,1,2,3,4,5,6,7}
    స్థిరరాశి int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    uint64_t తాత్కాలిక_ముసుగు;
    __asm__ అస్థిర("mova.x.m %0" : "=r"(తాత్కాలిక_ముసుగు));  // Save current mask
    __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load gather indices once — shared across all blocks
    __asm__ అస్థిర("flq2        f1, 0(%0)    \n\t"  // f1 = gather indices {0,1,2,3,4,5,6,7}
                     ::"r"(gather_indices)
                     : "f1");

    కొరకు (int64_t ఖండం_idx = 0; ఖండం_idx < అంకె_ఖండాలు; ఖండం_idx++) {
        స్థిరరాశి ఖండం_q4_K * ఖండం     = &src_ఖండాలు[ఖండం_idx];
        స్థిరరాశి uint8_t *    qs        = ఖండం->qs;
        దశాంశం *            ఖండం_dst = dst + ఖండం_idx * QK_K;

        స్థిరరాశి దశాంశం d   = fp16_గమ్యం_fp32(ఖండం->d);
        స్థిరరాశి దశాంశం min = fp16_గమ్యం_fp32(ఖండం->dmin);

        పూర్ణాంకం is = 0;
        కొరకు (పూర్ణాంకం j = 0; j < QK_K; j += 64) {
            // Extract per-group scales and mins (scalar — only 8 pairs per super-block)
            uint8_t sc, m;
            get_scale_min_k4(is + 0, ఖండం->scales, &sc, &m);
            స్థిరరాశి దశాంశం d1     = d * sc;
            స్థిరరాశి దశాంశం neg_m1 = -(min * m);
            get_scale_min_k4(is + 1, ఖండం->scales, &sc, &m);
            స్థిరరాశి దశాంశం d2     = d * sc;
            స్థిరరాశి దశాంశం neg_m2 = -(min * m);

            // Low nibbles: 32 elements using d1, neg_m1
            __asm__ అస్థిర(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d1)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m1)
                ::"r"(d1),
                "r"(neg_m1)
                : "f0", "f3");

            స్థిరరాశి uint8_t * qs_lo  = qs;
            దశాంశం *         dst_lo = ఖండం_dst + j;
            కొరకు (పూర్ణాంకం k = 0; k < 4; k++) {
                __asm__ అస్థిర(
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
            __asm__ అస్థిర(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d2)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m2)
                ::"r"(d2),
                "r"(neg_m2)
                : "f0", "f3");

            స్థిరరాశి uint8_t * qs_hi  = qs;
            దశాంశం *         dst_hi = ఖండం_dst + j + 32;
            కొరకు (పూర్ణాంకం k = 0; k < 4; k++) {
                __asm__ అస్థిర(
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

    __asm__ అస్థిర("mova.m.x %0" ::"r"(తాత్కాలిక_ముసుగు));  // Restore mask
}

// Determine the number of F32 elements per work unit for a given source type.
// For F32: 1 cacheline (16 elements)
// For quantized types: 1 quant block
స్థిర int64_t get_elements_చొప్పున_work_unit(పూర్ణాంకం type) {
    స్థిరరాశి int64_t elements_చొప్పున_cacheline = CACHE_పంక్తి_కొలత_బైట్లు / పరిమాణం(దశాంశం);  // 16
    ఎంపిక (type) {
        స్థితి GGML_TYPE_Q8_0:
            తిరిగి QK8_0;                   // 32 elements = 2 cachelines
        స్థితి GGML_TYPE_Q4_0:
            తిరిగి QK4_0;                   // 32 elements = 2 cachelines
        స్థితి GGML_TYPE_Q4_K:
            తిరిగి QK_K;                    // 256 elements = 16 cachelines
        అప్రమేయ:
            తిరిగి elements_చొప్పున_cacheline;  // 16 elements = 1 cacheline
    }
}

స్థిర పూర్ణాంకం get_అడ్డువరుస_f32_mc_cacheline_aligned(నిర్మాణం ggml_et_get_అడ్డువరుసలు_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env  = (kernel_environment_t *) env;
    పూర్ణాంకం                    thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం                    అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    నిర్మాణం ggml_tensor * src0 = &params->src0;  // Data tensor
    నిర్మాణం ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    నిర్మాణం ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    స్థిరరాశి int64_t ne00 = src0->ne[0];           // Source columns (row width)
    స్థిరరాశి int64_t ne01 = src0->ne[1];           // Source rows (total available rows)
    స్థిరరాశి int64_t ne02 = src0->ne[2];           // Source batch dimension
    స్థిరరాశి int64_t ne03 = src0->ne[3];           // Source outer batch dimension

    స్థిరరాశి int64_t ne10 = src1->ne[0];           // Number of indices in dimension 0
    స్థిరరాశి int64_t ne11 = src1->ne[1];           // Number of indices in dimension 1
    స్థిరరాశి int64_t ne12 = src1->ne[2];           // Batch dimension for indices
    స్థిరరాశి int64_t ne13 = src1->ne[3];           // Outer batch dimension for indices

    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు_గమ్యం_వెలికితీయి = ne10 * ne11 * ne12 * ne13;

    // Determine work unit size based on source type
    స్థిరరాశి int64_t elements_చొప్పున_wu = get_elements_చొప్పున_work_unit(src0->type);
    స్థిరరాశి int64_t wus_చొప్పున_అడ్డువరుస     = ne00 / elements_చొప్పున_wu;
    స్థిరరాశి int64_t మొత్తం_wus       = మొత్తం_అడ్డువరుసలు_గమ్యం_వెలికితీయి * wus_చొప్పున_అడ్డువరుస;

    // Distribute work units across threads (contiguous ranges)
    స్థిరరాశి int64_t wus_చొప్పున_thread = (మొత్తం_wus + అంకె_దారాలు - 1) / అంకె_దారాలు;
    స్థిరరాశి int64_t wu_ప్రారంభం       = thread_id * wus_చొప్పున_thread;
    int64_t       wu_చివర         = wu_ప్రారంభం + wus_చొప్పున_thread;
    ఒకవేళ (wu_చివర > మొత్తం_wus) {
        wu_చివర = మొత్తం_wus;
    }

    శూన్య *    src0_దత్తాంశం = src0->దత్తాంశం;
    int32_t * src1_దత్తాంశం = (int32_t *) src1->దత్తాంశం;
    దశాంశం *   dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;

    int64_t wu = wu_ప్రారంభం;
    అయితే (wu < wu_చివర) {
        // Determine which row this work unit belongs to and offset within row
        స్థిరరాశి int64_t అడ్డువరుస_idx   = wu / wus_చొప్పున_అడ్డువరుస;
        స్థిరరాశి int64_t wu_లోపల_అడ్డువరుస = wu % wus_చొప్పున_అడ్డువరుస;

        // How many work units to process in this row (batch contiguous WUs in same row)
        int64_t wus_మిగిలిన_లోపల_అడ్డువరుస = wus_చొప్పున_అడ్డువరుస - wu_లోపల_అడ్డువరుస;
        int64_t wus_గమ్యం_process       = wu_చివర - wu;
        ఒకవేళ (wus_మిగిలిన_లోపల_అడ్డువరుస < wus_గమ్యం_process) {
            wus_గమ్యం_process = wus_మిగిలిన_లోపల_అడ్డువరుస;
        }

        // Calculate multi-dimensional index for this row
        స్థిరరాశి int64_t i       = అడ్డువరుస_idx;
        స్థిరరాశి int64_t i13_idx = i / (ne12 * ne11 * ne10);
        స్థిరరాశి int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        స్థిరరాశి int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        స్థిరరాశి int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        స్థిరరాశి int64_t సూచీ_స్థానభ్రంశం = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        స్థిరరాశి int32_t అడ్డువరుస_సూచీ    = src1_దత్తాంశం[సూచీ_స్థానభ్రంశం];

        ఒకవేళ (అడ్డువరుస_సూచీ < 0 || అడ్డువరుస_సూచీ >= ne01) {
            తిరిగి -1;  // Index out of bounds
        }

        స్థిరరాశి int64_t batch_స్థానభ్రంశం =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        స్థిరరాశి int64_t elem_స్థానభ్రంశం_లోపల_అడ్డువరుస = wu_లోపల_అడ్డువరుస * elements_చొప్పున_wu;
        స్థిరరాశి int64_t అంకె_elements       = wus_గమ్యం_process * elements_చొప్పున_wu;

        దశాంశం * dst_అడ్డువరుస = dst_దత్తాంశం + అడ్డువరుస_idx * ne00 + elem_స్థానభ్రంశం_లోపల_అడ్డువరుస;

        ఒకవేళ (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy of cacheline-aligned chunk
            స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) src0_దత్తాంశం + అడ్డువరుస_సూచీ * ne00 + batch_స్థానభ్రంశం + elem_స్థానభ్రంశం_లోపల_అడ్డువరుస;
            copy_అడ్డువరుస_cache_align(dst_అడ్డువరుస, src_అడ్డువరుస, అంకె_elements * పరిమాణం(దశాంశం));
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion over a destination-aligned write chunk.
            స్థిరరాశి uint16_t * src_అడ్డువరుస =
                (స్థిరరాశి uint16_t *) src0_దత్తాంశం + అడ్డువరుస_సూచీ * ne00 + batch_స్థానభ్రంశం + elem_స్థానభ్రంశం_లోపల_అడ్డువరుస;
            copy_f16_అడ్డువరుస(dst_అడ్డువరుస, src_అడ్డువరుస, అంకె_elements);
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize work-unit-aligned blocks
            స్థిరరాశి int64_t      ఖండాలు_చొప్పున_అడ్డువరుస   = (ne00 + QK8_0 - 1) / QK8_0;
            స్థిరరాశి int64_t      src_ఖండం_స్థానభ్రంశం = (అడ్డువరుస_సూచీ * ఖండాలు_చొప్పున_అడ్డువరుస) + (batch_స్థానభ్రంశం / ne00) * ఖండాలు_చొప్పున_అడ్డువరుస;
            స్థిరరాశి int64_t      ఖండం_ప్రారంభం      = elem_స్థానభ్రంశం_లోపల_అడ్డువరుస / QK8_0;
            స్థిరరాశి ఖండం_q8_0 * src_ఖండాలు       = (స్థిరరాశి ఖండం_q8_0 *) src0_దత్తాంశం + src_ఖండం_స్థానభ్రంశం + ఖండం_ప్రారంభం;
            copy_q8_0_అడ్డువరుస_cache_aligned(dst_అడ్డువరుస, src_ఖండాలు, అంకె_elements);
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize work-unit-aligned blocks
            స్థిరరాశి int64_t      ఖండాలు_చొప్పున_అడ్డువరుస   = (ne00 + QK4_0 - 1) / QK4_0;
            స్థిరరాశి int64_t      src_ఖండం_స్థానభ్రంశం = (అడ్డువరుస_సూచీ * ఖండాలు_చొప్పున_అడ్డువరుస) + (batch_స్థానభ్రంశం / ne00) * ఖండాలు_చొప్పున_అడ్డువరుస;
            స్థిరరాశి int64_t      ఖండం_ప్రారంభం      = elem_స్థానభ్రంశం_లోపల_అడ్డువరుస / QK4_0;
            స్థిరరాశి ఖండం_q4_0 * src_ఖండాలు       = (స్థిరరాశి ఖండం_q4_0 *) src0_దత్తాంశం + src_ఖండం_స్థానభ్రంశం + ఖండం_ప్రారంభం;
            copy_q4_0_అడ్డువరుస_cache_aligned(dst_అడ్డువరుస, src_ఖండాలు, అంకె_elements);
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize work-unit-aligned blocks
            స్థిరరాశి int64_t      ఖండాలు_చొప్పున_అడ్డువరుస   = (ne00 + QK_K - 1) / QK_K;
            స్థిరరాశి int64_t      src_ఖండం_స్థానభ్రంశం = (అడ్డువరుస_సూచీ * ఖండాలు_చొప్పున_అడ్డువరుస) + (batch_స్థానభ్రంశం / ne00) * ఖండాలు_చొప్పున_అడ్డువరుస;
            స్థిరరాశి int64_t      ఖండం_ప్రారంభం      = elem_స్థానభ్రంశం_లోపల_అడ్డువరుస / QK_K;
            స్థిరరాశి ఖండం_q4_K * src_ఖండాలు       = (స్థిరరాశి ఖండం_q4_K *) src0_దత్తాంశం + src_ఖండం_స్థానభ్రంశం + ఖండం_ప్రారంభం;
            copy_q4_K_అడ్డువరుస_cache_aligned(dst_అడ్డువరుస, src_ఖండాలు, అంకె_elements);
        }

        wu += wus_గమ్యం_process;
    }

    తిరిగి 0;
}

స్థిర అంతఃవరుస size_t tensor_బైట్లు(స్థిరరాశి నిర్మాణం ggml_tensor * t) {
    తిరిగి (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_get_అడ్డువరుసలు_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    నిర్మాణం ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    నిర్మాణం ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    // Fast path - we know how to deal with them multi-core
    ఒకవేళ ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_Q8_0 ||
         src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_K) &&
        src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32 && dst->ne[0] % CACHE_ELEMENTS(పరిమాణం(దశాంశం)) == 0) {
        తిరిగి get_అడ్డువరుస_f32_mc_cacheline_aligned(params, env);
    }

    పూర్ణాంకం thread_id = get_relative_thread_id(kernel_env->shire_ముసుగు);
    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    ఒకవేళ (thread_id != 0) {
        తిరిగి 0;
    }

    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;  // Invalid pointer
    }

    ఒకవేళ (dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        తిరిగి -1;  // Invalid output or index type
    }

    ఒకవేళ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q4_K) {
        తిరిగి -1;  // Unsupported input type
    }

    శూన్య *    src0_దత్తాంశం = src0->దత్తాంశం;
    int32_t * src1_దత్తాంశం = (int32_t *) src1->దత్తాంశం;
    దశాంశం *   dst_దత్తాంశం  = (దశాంశం *) dst->దత్తాంశం;
#నిర్వచితమైతే ET_UBERKERNEL
    evict_region_past_l2(src0_దత్తాంశం, tensor_బైట్లు(src0));
    evict_region_past_l2(src1_దత్తాంశం, tensor_బైట్లు(src1));
    evict_region_past_l2(dst_దత్తాంశం, tensor_బైట్లు(dst));
#అయితే_ముగింపు

    ఒకవేళ (!src0_దత్తాంశం || !src1_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;  // Null data pointer
    }

    స్థిరరాశి int64_t ne00 = src0->ne[0];  // Source columns (row width)
    స్థిరరాశి int64_t ne01 = src0->ne[1];  // Source rows (total available rows)
    స్థిరరాశి int64_t ne02 = src0->ne[2];  // Source batch dimension
    స్థిరరాశి int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    స్థిరరాశి int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    స్థిరరాశి int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    స్థిరరాశి int64_t ne12 = src1->ne[2];  // Batch dimension for indices
    స్థిరరాశి int64_t ne13 = src1->ne[3];  // Outer batch dimension for indices

    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు_గమ్యం_వెలికితీయి = ne10 * ne11 * ne12 * ne13;
#నిర్వచితమైతే ET_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
#అయితే_ముగింపు
    // Naive single-threaded implementation - process all rows sequentially
    // XXX: Do we really need a single-threaded implementation?
    కొరకు (int64_t i = 0; i < మొత్తం_అడ్డువరుసలు_గమ్యం_వెలికితీయి; i++) {
        // Calculate multi-dimensional index for the current output position
        స్థిరరాశి int64_t i13_idx = i / (ne12 * ne11 * ne10);
        స్థిరరాశి int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        స్థిరరాశి int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        స్థిరరాశి int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        స్థిరరాశి int64_t సూచీ_స్థానభ్రంశం = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        స్థిరరాశి int32_t అడ్డువరుస_సూచీ    = src1_దత్తాంశం[సూచీ_స్థానభ్రంశం];

        ఒకవేళ (అడ్డువరుస_సూచీ < 0 || అడ్డువరుస_సూచీ >= ne01) {
            తిరిగి -1;  // Index out of bounds
        }

        స్థిరరాశి int64_t batch_స్థానభ్రంశం =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        స్థిరరాశి int64_t dst_స్థానభ్రంశం = i;

        ఒకవేళ (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy
            స్థిరరాశి దశాంశం * src_అడ్డువరుస = (స్థిరరాశి దశాంశం *) src0_దత్తాంశం + అడ్డువరుస_సూచీ * ne00 + batch_స్థానభ్రంశం;
            దశాంశం *       dst_అడ్డువరుస = dst_దత్తాంశం + dst_స్థానభ్రంశం * ne00;
            copy_f32_అడ్డువరుస(dst_అడ్డువరుస, src_అడ్డువరుస, ne00);
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion
            స్థిరరాశి uint16_t * src_అడ్డువరుస = (స్థిరరాశి uint16_t *) src0_దత్తాంశం + అడ్డువరుస_సూచీ * ne00 + batch_స్థానభ్రంశం;
            దశాంశం *          dst_అడ్డువరుస = dst_దత్తాంశం + dst_స్థానభ్రంశం * ne00;
            copy_f16_అడ్డువరుస(dst_అడ్డువరుస, src_అడ్డువరుస, ne00);
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize while copying
            స్థిరరాశి int64_t      ఖండాలు_చొప్పున_అడ్డువరుస   = (ne00 + QK8_0 - 1) / QK8_0;
            స్థిరరాశి int64_t      src_ఖండం_స్థానభ్రంశం = (అడ్డువరుస_సూచీ * ఖండాలు_చొప్పున_అడ్డువరుస) + (batch_స్థానభ్రంశం / ne00) * ఖండాలు_చొప్పున_అడ్డువరుస;
            స్థిరరాశి ఖండం_q8_0 * src_ఖండాలు       = (స్థిరరాశి ఖండం_q8_0 *) src0_దత్తాంశం + src_ఖండం_స్థానభ్రంశం;
            దశాంశం *            dst_అడ్డువరుస          = dst_దత్తాంశం + dst_స్థానభ్రంశం * ne00;
            copy_q8_0_అడ్డువరుస(dst_అడ్డువరుస, src_ఖండాలు, ne00);
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize while copying
            స్థిరరాశి int64_t      ఖండాలు_చొప్పున_అడ్డువరుస   = (ne00 + QK4_0 - 1) / QK4_0;
            స్థిరరాశి int64_t      src_ఖండం_స్థానభ్రంశం = (అడ్డువరుస_సూచీ * ఖండాలు_చొప్పున_అడ్డువరుస) + (batch_స్థానభ్రంశం / ne00) * ఖండాలు_చొప్పున_అడ్డువరుస;
            స్థిరరాశి ఖండం_q4_0 * src_ఖండాలు       = (స్థిరరాశి ఖండం_q4_0 *) src0_దత్తాంశం + src_ఖండం_స్థానభ్రంశం;
            దశాంశం *            dst_అడ్డువరుస          = dst_దత్తాంశం + dst_స్థానభ్రంశం * ne00;
            copy_q4_0_అడ్డువరుస(dst_అడ్డువరుస, src_ఖండాలు, ne00);
        } లేకపోతే ఒకవేళ (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize while copying
            స్థిరరాశి int64_t      ఖండాలు_చొప్పున_అడ్డువరుస   = (ne00 + QK_K - 1) / QK_K;
            స్థిరరాశి int64_t      src_ఖండం_స్థానభ్రంశం = (అడ్డువరుస_సూచీ * ఖండాలు_చొప్పున_అడ్డువరుస) + (batch_స్థానభ్రంశం / ne00) * ఖండాలు_చొప్పున_అడ్డువరుస;
            స్థిరరాశి ఖండం_q4_K * src_ఖండాలు       = (స్థిరరాశి ఖండం_q4_K *) src0_దత్తాంశం + src_ఖండం_స్థానభ్రంశం;
            దశాంశం *            dst_అడ్డువరుస          = dst_దత్తాంశం + dst_స్థానభ్రంశం * ne00;
            copy_q4_K_అడ్డువరుస(dst_అడ్డువరుస, src_ఖండాలు, ne00);
        }
    }

    తిరిగి 0;
}
