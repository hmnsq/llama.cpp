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

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "quants.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CACHE_ਸਤਰ_ਮਾਪ_ਬਾਈਟਾਂ 64

ਬਣਤਰ ggml_et_get_ਪੰਕਤੀਆਂ_params {
    ਬਣਤਰ ggml_tensor src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    ਬਣਤਰ ggml_tensor src1;  // Row indices tensor (I32)
    ਬਣਤਰ ggml_tensor dst;   // Output tensor (F32)
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CACHE_ਸਤਰ_ਮਾਪ_ਬਾਈਟਾਂ     64
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ CACHE_ELEMENTS(elem_ਮਾਪ) (CACHE_ਸਤਰ_ਮਾਪ_ਬਾਈਟਾਂ / (elem_ਮਾਪ))

// Copy a row of F32 data from source to destination
ਸਥਿਰ ਖਾਲੀ copy_f32_ਪੰਕਤੀ(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src, int64_t ਅੰਕ_elements) {
    // Simple memcpy for F32 data - no conversion needed
    ਲਈ (int64_t i = 0; i < ਅੰਕ_elements; i++) {
        dst[i] = src[i];
    }
}

ਸਥਿਰ ਖਾਲੀ copy_f16_ਪੰਕਤੀ(ਦਸ਼ਮਲਵ * dst, ਅਚਲ uint16_t * src, int64_t ਅੰਕ_elements) {
    ਲਈ (int64_t i = 0; i < ਅੰਕ_elements; i++) {
        dst[i] = fp16_ਮੰਜ਼ਿਲ_fp32(src[i]);
    }
}

// Copy a row of F32 data from source to destination, aligned to cache line boundaries
// using FP32 load/store instructions. They don't perform data conversion so is fine.
// Requirement: n_bytes is a multiple of CACHE_LINE_SIZE (64 bytes)
ਸਥਿਰ ਖਾਲੀ copy_ਪੰਕਤੀ_cache_align(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਦਸ਼ਮਲਵ * src, int64_t n_ਬਾਈਟਾਂ) {
    ਪੂਰਨ ਅੰਕ_f32_elem = n_ਬਾਈਟਾਂ / ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    // Unrolled to do an entire cache line at a time
    __asm__ ਅਸਥਿਰ(
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

        : [dst] "+r"(dst), [src] "+r"(src), [n] "+r"(ਅੰਕ_f32_elem)
        : [stride_ਗਿਣਤੀ] "r"(16L)
        : "f0", "f1", "memory");
}

// Copied from GGML: copy a row of Q4_0 data to F32 destination (with dequantization)
ਸਥਿਰ ਖਾਲੀ copy_q4_0_ਪੰਕਤੀ(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਖੰਡ_q4_0 * src_ਖੰਡਾਂ, int64_t ਅੰਕ_elements) {
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (ਅੰਕ_elements + QK4_0 - 1) / QK4_0;

    ਲਈ (int64_t ਖੰਡ_idx = 0; ਖੰਡ_idx < ਅੰਕ_ਖੰਡਾਂ; ਖੰਡ_idx++) {
        ਅਚਲ int64_t elements_ਵਿੱਚ_ਖੰਡ = (ਖੰਡ_idx == ਅੰਕ_ਖੰਡਾਂ - 1) ? (ਅੰਕ_elements - ਖੰਡ_idx * QK4_0) : QK4_0;

        ਦਸ਼ਮਲਵ ਆਰਜ਼ੀ_ਬਫਰ[QK4_0];
        dequantize_q4_0_ਖੰਡ(&src_ਖੰਡਾਂ[ਖੰਡ_idx], ਆਰਜ਼ੀ_ਬਫਰ);

        ਲਈ (int64_t i = 0; i < elements_ਵਿੱਚ_ਖੰਡ; i++) {
            dst[ਖੰਡ_idx * QK4_0 + i] = ਆਰਜ਼ੀ_ਬਫਰ[i];
        }
    }
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
ਸਥਿਰ ਖਾਲੀ copy_q8_0_ਪੰਕਤੀ(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਖੰਡ_q8_0 * src_ਖੰਡਾਂ, int64_t ਅੰਕ_elements) {
    // Number of Q8_0 blocks needed for this row
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (ਅੰਕ_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    ਲਈ (int64_t ਖੰਡ_idx = 0; ਖੰਡ_idx < ਅੰਕ_ਖੰਡਾਂ; ਖੰਡ_idx++) {
        ਅਚਲ int64_t elements_ਵਿੱਚ_ਖੰਡ =
            (ਖੰਡ_idx == ਅੰਕ_ਖੰਡਾਂ - 1) ? (ਅੰਕ_elements - ਖੰਡ_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        ਦਸ਼ਮਲਵ ਆਰਜ਼ੀ_ਬਫਰ[QK8_0];
        dequantize_q8_0_ਖੰਡ(&src_ਖੰਡਾਂ[ਖੰਡ_idx], ਆਰਜ਼ੀ_ਬਫਰ);

        // Copy dequantized values to destination
        ਲਈ (int64_t i = 0; i < elements_ਵਿੱਚ_ਖੰਡ; i++) {
            dst[ਖੰਡ_idx * QK8_0 + i] = ਆਰਜ਼ੀ_ਬਫਰ[i];
        }
    }
}

// Copy a row of Q4_K data to F32 destination (with dequantization)
ਸਥਿਰ ਖਾਲੀ copy_q4_K_ਪੰਕਤੀ(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਖੰਡ_q4_K * src_ਖੰਡਾਂ, int64_t ਅੰਕ_elements) {
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (ਅੰਕ_elements + QK_K - 1) / QK_K;

    ਲਈ (int64_t ਖੰਡ_idx = 0; ਖੰਡ_idx < ਅੰਕ_ਖੰਡਾਂ; ਖੰਡ_idx++) {
        ਅਚਲ int64_t elements_ਵਿੱਚ_ਖੰਡ = (ਖੰਡ_idx == ਅੰਕ_ਖੰਡਾਂ - 1) ? (ਅੰਕ_elements - ਖੰਡ_idx * QK_K) : QK_K;

        ਦਸ਼ਮਲਵ ਆਰਜ਼ੀ_ਬਫਰ[QK_K];
        dequantize_q4_K_ਖੰਡ(&src_ਖੰਡਾਂ[ਖੰਡ_idx], ਆਰਜ਼ੀ_ਬਫਰ);

        ਲਈ (int64_t i = 0; i < elements_ਵਿੱਚ_ਖੰਡ; i++) {
            dst[ਖੰਡ_idx * QK_K + i] = ਆਰਜ਼ੀ_ਬਫਰ[i];
        }
    }
}

ਸਥਿਰ ਖਾਲੀ dequantize_q8_0_ਖੰਡ_cache_aligned(ਅਚਲ ਖੰਡ_q8_0 * ਖੰਡ, ਦਸ਼ਮਲਵ * dst) {
    ਅਚਲ int8_t * qs_ptr = ਖੰਡ->qs;

    uint64_t ਆਰਜ਼ੀ_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));  // Save current mask
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    ਅਚਲ int32_t __attribute__((aligned(32))) vec_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    ਦਸ਼ਮਲਵ                                      scale          = fp16_ਮੰਜ਼ਿਲ_fp32(ਖੰਡ->d);
    __asm__ ਅਸਥਿਰ(
        "fbcx.ps     f0, %0       \n\t"  // Broadcast integer scale to all lanes
        "flq2        f1, 0(%1)    \n\t"  // Load gether indicies
        ::"r"(scale),
        "r"(vec_indices)
        : "f0", "f1");

    ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
        __asm__ ਅਸਥਿਰ(
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
    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));
}

// Copy a row of Q4_0 data to F32 destination (with dequantization), cache-aligned
ਸਥਿਰ ਖਾਲੀ copy_q4_0_ਪੰਕਤੀ_cache_aligned(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਖੰਡ_q4_0 * src_ਖੰਡਾਂ, int64_t ਅੰਕ_elements) {
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (ਅੰਕ_elements + QK4_0 - 1) / QK4_0;

    // Scatter byte offsets: even lanes -> dst[j], odd lanes -> dst[j + QK4_0/2]
    // For 4 consecutive packed bytes producing [low0, high0, low1, high1, low2, high2, low3, high3]:
    //   low_i  -> byte offset i*4       (positions 0,1,2,3 in first half)
    //   high_i -> byte offset (16+i)*4  (positions 16,17,18,19 in second half)
    ਅਚਲ int32_t __attribute__((aligned(32))) scatter_offsets[8] = { 0 * 4, 16 * 4, 1 * 4, 17 * 4,
                                                                      2 * 4, 18 * 4, 3 * 4, 19 * 4 };

    // Gather indices: each byte loaded twice for low/high nibble extraction
    ਅਚਲ int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 0, 1, 1, 2, 2, 3, 3 };

    uint64_t ਆਰਜ਼ੀ_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));  // Save current mask
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load constant vectors once — shared across all blocks and iterations
    __asm__ ਅਸਥਿਰ(
        "flq2        f4, 0(%0)    \n\t"  // f4 = scatter offsets
        "flq2        f1, 0(%1)    \n\t"  // f1 = gather indices {0,0,1,1,2,2,3,3}
        ::"r"(scatter_offsets),
        "r"(gather_indices)
        : "f1", "f4");

    ਲਈ (int64_t ਖੰਡ_idx = 0; ਖੰਡ_idx < ਅੰਕ_ਖੰਡਾਂ; ਖੰਡ_idx++) {
        ਅਚਲ ਖੰਡ_q4_0 * ਖੰਡ     = &src_ਖੰਡਾਂ[ਖੰਡ_idx];
        ਅਚਲ uint8_t *    qs        = ਖੰਡ->qs;
        ਦਸ਼ਮਲਵ *            ਖੰਡ_dst = dst + ਖੰਡ_idx * QK4_0;

        ਦਸ਼ਮਲਵ scale = fp16_ਮੰਜ਼ਿਲ_fp32(ਖੰਡ->d);
        ਦਸ਼ਮਲਵ bias  = -8.0f * scale;

        // Per-block: broadcast scale and bias
        __asm__ ਅਸਥਿਰ(
            "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(scale)
            "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-8 * scale)
            ::"r"(scale),
            "r"(bias)
            : "f0", "f3");

        // 4 iterations x 4 packed bytes = 16 bytes = full block -> 32 floats
        ਲਈ (ਪੂਰਨ i = 0; i < 4; i++) {
            __asm__ ਅਸਥਿਰ(
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
                "r"(ਖੰਡ_dst)
                : "f2", "memory");

            qs += 4;         // 4 packed bytes consumed
            ਖੰਡ_dst += 4;  // Advance base by 4 float positions
        }
    }

    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));  // Restore mask
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
ਸਥਿਰ ਖਾਲੀ copy_q8_0_ਪੰਕਤੀ_cache_aligned(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਖੰਡ_q8_0 * src_ਖੰਡਾਂ, int64_t ਅੰਕ_elements) {
    // Number of Q8_0 blocks needed for this row
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (ਅੰਕ_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    ਲਈ (int64_t ਖੰਡ_idx = 0; ਖੰਡ_idx < ਅੰਕ_ਖੰਡਾਂ; ਖੰਡ_idx++) {
        ਅਚਲ int64_t elements_ਵਿੱਚ_ਖੰਡ =
            (ਖੰਡ_idx == ਅੰਕ_ਖੰਡਾਂ - 1) ? (ਅੰਕ_elements - ਖੰਡ_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        ਦਸ਼ਮਲਵ ਆਰਜ਼ੀ_ਬਫਰ[QK8_0];
        dequantize_q8_0_ਖੰਡ_cache_aligned(&src_ਖੰਡਾਂ[ਖੰਡ_idx], ਆਰਜ਼ੀ_ਬਫਰ);

        // Copy dequantized values to destination
        ਲਈ (int64_t i = 0; i < elements_ਵਿੱਚ_ਖੰਡ; i++) {
            dst[ਖੰਡ_idx * QK8_0 + i] = ਆਰਜ਼ੀ_ਬਫਰ[i];
        }
    }
}

// Vectorized dequantization of a Q4_K super-block (256 elements) to F32
// Processes 8 groups of 32 elements, using ET SIMD for the inner loops.
// Output is sequential (no scatter needed unlike Q4_0).
ਸਥਿਰ ਖਾਲੀ copy_q4_K_ਪੰਕਤੀ_cache_aligned(ਦਸ਼ਮਲਵ * dst, ਅਚਲ ਖੰਡ_q4_K * src_ਖੰਡਾਂ, int64_t ਅੰਕ_elements) {
    ਅਚਲ int64_t ਅੰਕ_ਖੰਡਾਂ = (ਅੰਕ_elements + QK_K - 1) / QK_K;

    // Gather indices for sequential byte access: {0,1,2,3,4,5,6,7}
    ਅਚਲ int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    uint64_t ਆਰਜ਼ੀ_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));  // Save current mask
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load gather indices once — shared across all blocks
    __asm__ ਅਸਥਿਰ("flq2        f1, 0(%0)    \n\t"  // f1 = gather indices {0,1,2,3,4,5,6,7}
                     ::"r"(gather_indices)
                     : "f1");

    ਲਈ (int64_t ਖੰਡ_idx = 0; ਖੰਡ_idx < ਅੰਕ_ਖੰਡਾਂ; ਖੰਡ_idx++) {
        ਅਚਲ ਖੰਡ_q4_K * ਖੰਡ     = &src_ਖੰਡਾਂ[ਖੰਡ_idx];
        ਅਚਲ uint8_t *    qs        = ਖੰਡ->qs;
        ਦਸ਼ਮਲਵ *            ਖੰਡ_dst = dst + ਖੰਡ_idx * QK_K;

        ਅਚਲ ਦਸ਼ਮਲਵ d   = fp16_ਮੰਜ਼ਿਲ_fp32(ਖੰਡ->d);
        ਅਚਲ ਦਸ਼ਮਲਵ min = fp16_ਮੰਜ਼ਿਲ_fp32(ਖੰਡ->dmin);

        ਪੂਰਨ is = 0;
        ਲਈ (ਪੂਰਨ j = 0; j < QK_K; j += 64) {
            // Extract per-group scales and mins (scalar — only 8 pairs per super-block)
            uint8_t sc, m;
            get_scale_min_k4(is + 0, ਖੰਡ->scales, &sc, &m);
            ਅਚਲ ਦਸ਼ਮਲਵ d1     = d * sc;
            ਅਚਲ ਦਸ਼ਮਲਵ neg_m1 = -(min * m);
            get_scale_min_k4(is + 1, ਖੰਡ->scales, &sc, &m);
            ਅਚਲ ਦਸ਼ਮਲਵ d2     = d * sc;
            ਅਚਲ ਦਸ਼ਮਲਵ neg_m2 = -(min * m);

            // Low nibbles: 32 elements using d1, neg_m1
            __asm__ ਅਸਥਿਰ(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d1)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m1)
                ::"r"(d1),
                "r"(neg_m1)
                : "f0", "f3");

            ਅਚਲ uint8_t * qs_lo  = qs;
            ਦਸ਼ਮਲਵ *         dst_lo = ਖੰਡ_dst + j;
            ਲਈ (ਪੂਰਨ k = 0; k < 4; k++) {
                __asm__ ਅਸਥਿਰ(
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
            __asm__ ਅਸਥਿਰ(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d2)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m2)
                ::"r"(d2),
                "r"(neg_m2)
                : "f0", "f3");

            ਅਚਲ uint8_t * qs_hi  = qs;
            ਦਸ਼ਮਲਵ *         dst_hi = ਖੰਡ_dst + j + 32;
            ਲਈ (ਪੂਰਨ k = 0; k < 4; k++) {
                __asm__ ਅਸਥਿਰ(
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

    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));  // Restore mask
}

// Determine the number of F32 elements per work unit for a given source type.
// For F32: 1 cacheline (16 elements)
// For quantized types: 1 quant block
ਸਥਿਰ int64_t get_elements_ਪ੍ਰਤੀ_work_unit(ਪੂਰਨ type) {
    ਅਚਲ int64_t elements_ਪ੍ਰਤੀ_cacheline = CACHE_ਸਤਰ_ਮਾਪ_ਬਾਈਟਾਂ / ਆਕਾਰ(ਦਸ਼ਮਲਵ);  // 16
    ਚੋਣ (type) {
        ਹਾਲਤ GGML_TYPE_Q8_0:
            ਵਾਪਸ QK8_0;                   // 32 elements = 2 cachelines
        ਹਾਲਤ GGML_TYPE_Q4_0:
            ਵਾਪਸ QK4_0;                   // 32 elements = 2 cachelines
        ਹਾਲਤ GGML_TYPE_Q4_K:
            ਵਾਪਸ QK_K;                    // 256 elements = 16 cachelines
        ਮੂਲ:
            ਵਾਪਸ elements_ਪ੍ਰਤੀ_cacheline;  // 16 elements = 1 cacheline
    }
}

ਸਥਿਰ ਪੂਰਨ get_ਪੰਕਤੀ_f32_mc_cacheline_aligned(ਬਣਤਰ ggml_et_get_ਪੰਕਤੀਆਂ_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env  = (kernel_environment_t *) env;
    ਪੂਰਨ                    thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ                    ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;  // Data tensor
    ਬਣਤਰ ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    ਅਚਲ int64_t ne00 = src0->ne[0];           // Source columns (row width)
    ਅਚਲ int64_t ne01 = src0->ne[1];           // Source rows (total available rows)
    ਅਚਲ int64_t ne02 = src0->ne[2];           // Source batch dimension
    ਅਚਲ int64_t ne03 = src0->ne[3];           // Source outer batch dimension

    ਅਚਲ int64_t ne10 = src1->ne[0];           // Number of indices in dimension 0
    ਅਚਲ int64_t ne11 = src1->ne[1];           // Number of indices in dimension 1
    ਅਚਲ int64_t ne12 = src1->ne[2];           // Batch dimension for indices
    ਅਚਲ int64_t ne13 = src1->ne[3];           // Outer batch dimension for indices

    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ_ਮੰਜ਼ਿਲ_ਕੱਢੋ = ne10 * ne11 * ne12 * ne13;

    // Determine work unit size based on source type
    ਅਚਲ int64_t elements_ਪ੍ਰਤੀ_wu = get_elements_ਪ੍ਰਤੀ_work_unit(src0->type);
    ਅਚਲ int64_t wus_ਪ੍ਰਤੀ_ਪੰਕਤੀ     = ne00 / elements_ਪ੍ਰਤੀ_wu;
    ਅਚਲ int64_t ਕੁੱਲ_wus       = ਕੁੱਲ_ਪੰਕਤੀਆਂ_ਮੰਜ਼ਿਲ_ਕੱਢੋ * wus_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

    // Distribute work units across threads (contiguous ranges)
    ਅਚਲ int64_t wus_ਪ੍ਰਤੀ_thread = (ਕੁੱਲ_wus + ਅੰਕ_ਧਾਗੇ - 1) / ਅੰਕ_ਧਾਗੇ;
    ਅਚਲ int64_t wu_ਸ਼ੁਰੂ       = thread_id * wus_ਪ੍ਰਤੀ_thread;
    int64_t       wu_ਸਿਰਾ         = wu_ਸ਼ੁਰੂ + wus_ਪ੍ਰਤੀ_thread;
    ਜੇ (wu_ਸਿਰਾ > ਕੁੱਲ_wus) {
        wu_ਸਿਰਾ = ਕੁੱਲ_wus;
    }

    ਖਾਲੀ *    src0_ਅੰਕੜੇ = src0->ਅੰਕੜੇ;
    int32_t * src1_ਅੰਕੜੇ = (int32_t *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *   dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    int64_t wu = wu_ਸ਼ੁਰੂ;
    ਜਦੋਂ (wu < wu_ਸਿਰਾ) {
        // Determine which row this work unit belongs to and offset within row
        ਅਚਲ int64_t ਪੰਕਤੀ_idx   = wu / wus_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
        ਅਚਲ int64_t wu_ਵਿੱਚ_ਪੰਕਤੀ = wu % wus_ਪ੍ਰਤੀ_ਪੰਕਤੀ;

        // How many work units to process in this row (batch contiguous WUs in same row)
        int64_t wus_ਬਚਿਆ_ਵਿੱਚ_ਪੰਕਤੀ = wus_ਪ੍ਰਤੀ_ਪੰਕਤੀ - wu_ਵਿੱਚ_ਪੰਕਤੀ;
        int64_t wus_ਮੰਜ਼ਿਲ_process       = wu_ਸਿਰਾ - wu;
        ਜੇ (wus_ਬਚਿਆ_ਵਿੱਚ_ਪੰਕਤੀ < wus_ਮੰਜ਼ਿਲ_process) {
            wus_ਮੰਜ਼ਿਲ_process = wus_ਬਚਿਆ_ਵਿੱਚ_ਪੰਕਤੀ;
        }

        // Calculate multi-dimensional index for this row
        ਅਚਲ int64_t i       = ਪੰਕਤੀ_idx;
        ਅਚਲ int64_t i13_idx = i / (ne12 * ne11 * ne10);
        ਅਚਲ int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        ਅਚਲ int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        ਅਚਲ int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        ਅਚਲ int64_t ਸੂਚਕਾਂਕ_ਵਿੱਥ = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        ਅਚਲ int32_t ਪੰਕਤੀ_ਸੂਚਕਾਂਕ    = src1_ਅੰਕੜੇ[ਸੂਚਕਾਂਕ_ਵਿੱਥ];

        ਜੇ (ਪੰਕਤੀ_ਸੂਚਕਾਂਕ < 0 || ਪੰਕਤੀ_ਸੂਚਕਾਂਕ >= ne01) {
            ਵਾਪਸ -1;  // Index out of bounds
        }

        ਅਚਲ int64_t batch_ਵਿੱਥ =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        ਅਚਲ int64_t elem_ਵਿੱਥ_ਵਿੱਚ_ਪੰਕਤੀ = wu_ਵਿੱਚ_ਪੰਕਤੀ * elements_ਪ੍ਰਤੀ_wu;
        ਅਚਲ int64_t ਅੰਕ_elements       = wus_ਮੰਜ਼ਿਲ_process * elements_ਪ੍ਰਤੀ_wu;

        ਦਸ਼ਮਲਵ * dst_ਪੰਕਤੀ = dst_ਅੰਕੜੇ + ਪੰਕਤੀ_idx * ne00 + elem_ਵਿੱਥ_ਵਿੱਚ_ਪੰਕਤੀ;

        ਜੇ (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy of cacheline-aligned chunk
            ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0_ਅੰਕੜੇ + ਪੰਕਤੀ_ਸੂਚਕਾਂਕ * ne00 + batch_ਵਿੱਥ + elem_ਵਿੱਥ_ਵਿੱਚ_ਪੰਕਤੀ;
            copy_ਪੰਕਤੀ_cache_align(dst_ਪੰਕਤੀ, src_ਪੰਕਤੀ, ਅੰਕ_elements * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion over a destination-aligned write chunk.
            ਅਚਲ uint16_t * src_ਪੰਕਤੀ =
                (ਅਚਲ uint16_t *) src0_ਅੰਕੜੇ + ਪੰਕਤੀ_ਸੂਚਕਾਂਕ * ne00 + batch_ਵਿੱਥ + elem_ਵਿੱਥ_ਵਿੱਚ_ਪੰਕਤੀ;
            copy_f16_ਪੰਕਤੀ(dst_ਪੰਕਤੀ, src_ਪੰਕਤੀ, ਅੰਕ_elements);
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize work-unit-aligned blocks
            ਅਚਲ int64_t      ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ   = (ne00 + QK8_0 - 1) / QK8_0;
            ਅਚਲ int64_t      src_ਖੰਡ_ਵਿੱਥ = (ਪੰਕਤੀ_ਸੂਚਕਾਂਕ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) + (batch_ਵਿੱਥ / ne00) * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
            ਅਚਲ int64_t      ਖੰਡ_ਸ਼ੁਰੂ      = elem_ਵਿੱਥ_ਵਿੱਚ_ਪੰਕਤੀ / QK8_0;
            ਅਚਲ ਖੰਡ_q8_0 * src_ਖੰਡਾਂ       = (ਅਚਲ ਖੰਡ_q8_0 *) src0_ਅੰਕੜੇ + src_ਖੰਡ_ਵਿੱਥ + ਖੰਡ_ਸ਼ੁਰੂ;
            copy_q8_0_ਪੰਕਤੀ_cache_aligned(dst_ਪੰਕਤੀ, src_ਖੰਡਾਂ, ਅੰਕ_elements);
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize work-unit-aligned blocks
            ਅਚਲ int64_t      ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ   = (ne00 + QK4_0 - 1) / QK4_0;
            ਅਚਲ int64_t      src_ਖੰਡ_ਵਿੱਥ = (ਪੰਕਤੀ_ਸੂਚਕਾਂਕ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) + (batch_ਵਿੱਥ / ne00) * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
            ਅਚਲ int64_t      ਖੰਡ_ਸ਼ੁਰੂ      = elem_ਵਿੱਥ_ਵਿੱਚ_ਪੰਕਤੀ / QK4_0;
            ਅਚਲ ਖੰਡ_q4_0 * src_ਖੰਡਾਂ       = (ਅਚਲ ਖੰਡ_q4_0 *) src0_ਅੰਕੜੇ + src_ਖੰਡ_ਵਿੱਥ + ਖੰਡ_ਸ਼ੁਰੂ;
            copy_q4_0_ਪੰਕਤੀ_cache_aligned(dst_ਪੰਕਤੀ, src_ਖੰਡਾਂ, ਅੰਕ_elements);
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize work-unit-aligned blocks
            ਅਚਲ int64_t      ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ   = (ne00 + QK_K - 1) / QK_K;
            ਅਚਲ int64_t      src_ਖੰਡ_ਵਿੱਥ = (ਪੰਕਤੀ_ਸੂਚਕਾਂਕ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) + (batch_ਵਿੱਥ / ne00) * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
            ਅਚਲ int64_t      ਖੰਡ_ਸ਼ੁਰੂ      = elem_ਵਿੱਥ_ਵਿੱਚ_ਪੰਕਤੀ / QK_K;
            ਅਚਲ ਖੰਡ_q4_K * src_ਖੰਡਾਂ       = (ਅਚਲ ਖੰਡ_q4_K *) src0_ਅੰਕੜੇ + src_ਖੰਡ_ਵਿੱਥ + ਖੰਡ_ਸ਼ੁਰੂ;
            copy_q4_K_ਪੰਕਤੀ_cache_aligned(dst_ਪੰਕਤੀ, src_ਖੰਡਾਂ, ਅੰਕ_elements);
        }

        wu += wus_ਮੰਜ਼ਿਲ_process;
    }

    ਵਾਪਸ 0;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t tensor_ਬਾਈਟਾਂ(ਅਚਲ ਬਣਤਰ ggml_tensor * t) {
    ਵਾਪਸ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_get_ਪੰਕਤੀਆਂ_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    ਬਣਤਰ ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    // Fast path - we know how to deal with them multi-core
    ਜੇ ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_Q8_0 ||
         src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_K) &&
        src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32 && dst->ne[0] % CACHE_ELEMENTS(ਆਕਾਰ(ਦਸ਼ਮਲਵ)) == 0) {
        ਵਾਪਸ get_ਪੰਕਤੀ_f32_mc_cacheline_aligned(params, env);
    }

    ਪੂਰਨ thread_id = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    ਜੇ (thread_id != 0) {
        ਵਾਪਸ 0;
    }

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;  // Invalid pointer
    }

    ਜੇ (dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        ਵਾਪਸ -1;  // Invalid output or index type
    }

    ਜੇ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q4_K) {
        ਵਾਪਸ -1;  // Unsupported input type
    }

    ਖਾਲੀ *    src0_ਅੰਕੜੇ = src0->ਅੰਕੜੇ;
    int32_t * src1_ਅੰਕੜੇ = (int32_t *) src1->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ *   dst_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ET_UBERKERNEL
    evict_region_past_l2(src0_ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ(src0));
    evict_region_past_l2(src1_ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ(src1));
    evict_region_past_l2(dst_ਅੰਕੜੇ, tensor_ਬਾਈਟਾਂ(dst));
#ਜੇ_ਅੰਤ

    ਜੇ (!src0_ਅੰਕੜੇ || !src1_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;  // Null data pointer
    }

    ਅਚਲ int64_t ne00 = src0->ne[0];  // Source columns (row width)
    ਅਚਲ int64_t ne01 = src0->ne[1];  // Source rows (total available rows)
    ਅਚਲ int64_t ne02 = src0->ne[2];  // Source batch dimension
    ਅਚਲ int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    ਅਚਲ int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    ਅਚਲ int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    ਅਚਲ int64_t ne12 = src1->ne[2];  // Batch dimension for indices
    ਅਚਲ int64_t ne13 = src1->ne[3];  // Outer batch dimension for indices

    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ_ਮੰਜ਼ਿਲ_ਕੱਢੋ = ne10 * ne11 * ne12 * ne13;
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ET_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
#ਜੇ_ਅੰਤ
    // Naive single-threaded implementation - process all rows sequentially
    // XXX: Do we really need a single-threaded implementation?
    ਲਈ (int64_t i = 0; i < ਕੁੱਲ_ਪੰਕਤੀਆਂ_ਮੰਜ਼ਿਲ_ਕੱਢੋ; i++) {
        // Calculate multi-dimensional index for the current output position
        ਅਚਲ int64_t i13_idx = i / (ne12 * ne11 * ne10);
        ਅਚਲ int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        ਅਚਲ int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        ਅਚਲ int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        ਅਚਲ int64_t ਸੂਚਕਾਂਕ_ਵਿੱਥ = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        ਅਚਲ int32_t ਪੰਕਤੀ_ਸੂਚਕਾਂਕ    = src1_ਅੰਕੜੇ[ਸੂਚਕਾਂਕ_ਵਿੱਥ];

        ਜੇ (ਪੰਕਤੀ_ਸੂਚਕਾਂਕ < 0 || ਪੰਕਤੀ_ਸੂਚਕਾਂਕ >= ne01) {
            ਵਾਪਸ -1;  // Index out of bounds
        }

        ਅਚਲ int64_t batch_ਵਿੱਥ =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        ਅਚਲ int64_t dst_ਵਿੱਥ = i;

        ਜੇ (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy
            ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0_ਅੰਕੜੇ + ਪੰਕਤੀ_ਸੂਚਕਾਂਕ * ne00 + batch_ਵਿੱਥ;
            ਦਸ਼ਮਲਵ *       dst_ਪੰਕਤੀ = dst_ਅੰਕੜੇ + dst_ਵਿੱਥ * ne00;
            copy_f32_ਪੰਕਤੀ(dst_ਪੰਕਤੀ, src_ਪੰਕਤੀ, ne00);
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion
            ਅਚਲ uint16_t * src_ਪੰਕਤੀ = (ਅਚਲ uint16_t *) src0_ਅੰਕੜੇ + ਪੰਕਤੀ_ਸੂਚਕਾਂਕ * ne00 + batch_ਵਿੱਥ;
            ਦਸ਼ਮਲਵ *          dst_ਪੰਕਤੀ = dst_ਅੰਕੜੇ + dst_ਵਿੱਥ * ne00;
            copy_f16_ਪੰਕਤੀ(dst_ਪੰਕਤੀ, src_ਪੰਕਤੀ, ne00);
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize while copying
            ਅਚਲ int64_t      ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ   = (ne00 + QK8_0 - 1) / QK8_0;
            ਅਚਲ int64_t      src_ਖੰਡ_ਵਿੱਥ = (ਪੰਕਤੀ_ਸੂਚਕਾਂਕ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) + (batch_ਵਿੱਥ / ne00) * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
            ਅਚਲ ਖੰਡ_q8_0 * src_ਖੰਡਾਂ       = (ਅਚਲ ਖੰਡ_q8_0 *) src0_ਅੰਕੜੇ + src_ਖੰਡ_ਵਿੱਥ;
            ਦਸ਼ਮਲਵ *            dst_ਪੰਕਤੀ          = dst_ਅੰਕੜੇ + dst_ਵਿੱਥ * ne00;
            copy_q8_0_ਪੰਕਤੀ(dst_ਪੰਕਤੀ, src_ਖੰਡਾਂ, ne00);
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize while copying
            ਅਚਲ int64_t      ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ   = (ne00 + QK4_0 - 1) / QK4_0;
            ਅਚਲ int64_t      src_ਖੰਡ_ਵਿੱਥ = (ਪੰਕਤੀ_ਸੂਚਕਾਂਕ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) + (batch_ਵਿੱਥ / ne00) * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
            ਅਚਲ ਖੰਡ_q4_0 * src_ਖੰਡਾਂ       = (ਅਚਲ ਖੰਡ_q4_0 *) src0_ਅੰਕੜੇ + src_ਖੰਡ_ਵਿੱਥ;
            ਦਸ਼ਮਲਵ *            dst_ਪੰਕਤੀ          = dst_ਅੰਕੜੇ + dst_ਵਿੱਥ * ne00;
            copy_q4_0_ਪੰਕਤੀ(dst_ਪੰਕਤੀ, src_ਖੰਡਾਂ, ne00);
        } ਨਹੀਂਤਾਂ ਜੇ (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize while copying
            ਅਚਲ int64_t      ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ   = (ne00 + QK_K - 1) / QK_K;
            ਅਚਲ int64_t      src_ਖੰਡ_ਵਿੱਥ = (ਪੰਕਤੀ_ਸੂਚਕਾਂਕ * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ) + (batch_ਵਿੱਥ / ne00) * ਖੰਡਾਂ_ਪ੍ਰਤੀ_ਪੰਕਤੀ;
            ਅਚਲ ਖੰਡ_q4_K * src_ਖੰਡਾਂ       = (ਅਚਲ ਖੰਡ_q4_K *) src0_ਅੰਕੜੇ + src_ਖੰਡ_ਵਿੱਥ;
            ਦਸ਼ਮਲਵ *            dst_ਪੰਕਤੀ          = dst_ਅੰਕੜੇ + dst_ਵਿੱਥ * ne00;
            copy_q4_K_ਪੰਕਤੀ(dst_ਪੰਕਤੀ, src_ਖੰਡਾਂ, ne00);
        }
    }

    ਵਾਪਸ 0;
}
