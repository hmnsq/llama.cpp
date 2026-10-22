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

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "platform.h"
#ಅನ್ತರ್ಭಾವ "quants.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

#ವ್ಯಾಖ್ಯಾನಿಸು CACHE_ಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು 64

ರಚನೆ ggml_et_get_ಅಡ್ಡಸಾಲುಗಳು_params {
    ರಚನೆ ggml_tensor src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    ರಚನೆ ggml_tensor src1;  // Row indices tensor (I32)
    ರಚನೆ ggml_tensor dst;   // Output tensor (F32)
};

#ವ್ಯಾಖ್ಯಾನಿಸು CACHE_ಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು     64
#ವ್ಯಾಖ್ಯಾನಿಸು CACHE_ELEMENTS(elem_ಗಾತ್ರ) (CACHE_ಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು / (elem_ಗಾತ್ರ))

// Copy a row of F32 data from source to destination
ಸ್ಥಿರ ಶೂನ್ಯ copy_f32_ಅಡ್ಡಸಾಲು(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, int64_t ಅಂಕೆ_elements) {
    // Simple memcpy for F32 data - no conversion needed
    ಗಾಗಿ (int64_t i = 0; i < ಅಂಕೆ_elements; i++) {
        dst[i] = src[i];
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ copy_f16_ಅಡ್ಡಸಾಲು(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ uint16_t * src, int64_t ಅಂಕೆ_elements) {
    ಗಾಗಿ (int64_t i = 0; i < ಅಂಕೆ_elements; i++) {
        dst[i] = fp16_ಗಮ್ಯ_fp32(src[i]);
    }
}

// Copy a row of F32 data from source to destination, aligned to cache line boundaries
// using FP32 load/store instructions. They don't perform data conversion so is fine.
// Requirement: n_bytes is a multiple of CACHE_LINE_SIZE (64 bytes)
ಸ್ಥಿರ ಶೂನ್ಯ copy_ಅಡ್ಡಸಾಲು_cache_align(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src, int64_t n_ಬೈಟುಗಳು) {
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_f32_elem = n_ಬೈಟುಗಳು / ಗಾತ್ರ(ದಶಮಾಂಶ);

    // Unrolled to do an entire cache line at a time
    __asm__ ಅಸ್ಥಿರ(
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

        : [dst] "+r"(dst), [src] "+r"(src), [n] "+r"(ಅಂಕೆ_f32_elem)
        : [stride_ಎಣಿಕೆ] "r"(16L)
        : "f0", "f1", "memory");
}

// Copied from GGML: copy a row of Q4_0 data to F32 destination (with dequantization)
ಸ್ಥಿರ ಶೂನ್ಯ copy_q4_0_ಅಡ್ಡಸಾಲು(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * src_ಖಂಡಗಳು, int64_t ಅಂಕೆ_elements) {
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (ಅಂಕೆ_elements + QK4_0 - 1) / QK4_0;

    ಗಾಗಿ (int64_t ಖಂಡ_idx = 0; ಖಂಡ_idx < ಅಂಕೆ_ಖಂಡಗಳು; ಖಂಡ_idx++) {
        ಸ್ಥಿರಾಂಕ int64_t elements_ಒಳ_ಖಂಡ = (ಖಂಡ_idx == ಅಂಕೆ_ಖಂಡಗಳು - 1) ? (ಅಂಕೆ_elements - ಖಂಡ_idx * QK4_0) : QK4_0;

        ದಶಮಾಂಶ ತಾತ್ಕಾಲಿಕ_ಬಫರ್[QK4_0];
        dequantize_q4_0_ಖಂಡ(&src_ಖಂಡಗಳು[ಖಂಡ_idx], ತಾತ್ಕಾಲಿಕ_ಬಫರ್);

        ಗಾಗಿ (int64_t i = 0; i < elements_ಒಳ_ಖಂಡ; i++) {
            dst[ಖಂಡ_idx * QK4_0 + i] = ತಾತ್ಕಾಲಿಕ_ಬಫರ್[i];
        }
    }
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
ಸ್ಥಿರ ಶೂನ್ಯ copy_q8_0_ಅಡ್ಡಸಾಲು(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * src_ಖಂಡಗಳು, int64_t ಅಂಕೆ_elements) {
    // Number of Q8_0 blocks needed for this row
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (ಅಂಕೆ_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    ಗಾಗಿ (int64_t ಖಂಡ_idx = 0; ಖಂಡ_idx < ಅಂಕೆ_ಖಂಡಗಳು; ಖಂಡ_idx++) {
        ಸ್ಥಿರಾಂಕ int64_t elements_ಒಳ_ಖಂಡ =
            (ಖಂಡ_idx == ಅಂಕೆ_ಖಂಡಗಳು - 1) ? (ಅಂಕೆ_elements - ಖಂಡ_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        ದಶಮಾಂಶ ತಾತ್ಕಾಲಿಕ_ಬಫರ್[QK8_0];
        dequantize_q8_0_ಖಂಡ(&src_ಖಂಡಗಳು[ಖಂಡ_idx], ತಾತ್ಕಾಲಿಕ_ಬಫರ್);

        // Copy dequantized values to destination
        ಗಾಗಿ (int64_t i = 0; i < elements_ಒಳ_ಖಂಡ; i++) {
            dst[ಖಂಡ_idx * QK8_0 + i] = ತಾತ್ಕಾಲಿಕ_ಬಫರ್[i];
        }
    }
}

// Copy a row of Q4_K data to F32 destination (with dequantization)
ಸ್ಥಿರ ಶೂನ್ಯ copy_q4_K_ಅಡ್ಡಸಾಲು(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * src_ಖಂಡಗಳು, int64_t ಅಂಕೆ_elements) {
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (ಅಂಕೆ_elements + QK_K - 1) / QK_K;

    ಗಾಗಿ (int64_t ಖಂಡ_idx = 0; ಖಂಡ_idx < ಅಂಕೆ_ಖಂಡಗಳು; ಖಂಡ_idx++) {
        ಸ್ಥಿರಾಂಕ int64_t elements_ಒಳ_ಖಂಡ = (ಖಂಡ_idx == ಅಂಕೆ_ಖಂಡಗಳು - 1) ? (ಅಂಕೆ_elements - ಖಂಡ_idx * QK_K) : QK_K;

        ದಶಮಾಂಶ ತಾತ್ಕಾಲಿಕ_ಬಫರ್[QK_K];
        dequantize_q4_K_ಖಂಡ(&src_ಖಂಡಗಳು[ಖಂಡ_idx], ತಾತ್ಕಾಲಿಕ_ಬಫರ್);

        ಗಾಗಿ (int64_t i = 0; i < elements_ಒಳ_ಖಂಡ; i++) {
            dst[ಖಂಡ_idx * QK_K + i] = ತಾತ್ಕಾಲಿಕ_ಬಫರ್[i];
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ dequantize_q8_0_ಖಂಡ_cache_aligned(ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * ಖಂಡ, ದಶಮಾಂಶ * dst) {
    ಸ್ಥಿರಾಂಕ int8_t * qs_ptr = ಖಂಡ->qs;

    uint64_t ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));  // Save current mask
    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    ಸ್ಥಿರಾಂಕ int32_t __attribute__((aligned(32))) vec_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    ದಶಮಾಂಶ                                      scale          = fp16_ಗಮ್ಯ_fp32(ಖಂಡ->d);
    __asm__ ಅಸ್ಥಿರ(
        "fbcx.ps     f0, %0       \n\t"  // Broadcast integer scale to all lanes
        "flq2        f1, 0(%1)    \n\t"  // Load gether indicies
        ::"r"(scale),
        "r"(vec_indices)
        : "f0", "f1");

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
        __asm__ ಅಸ್ಥಿರ(
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
    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));
}

// Copy a row of Q4_0 data to F32 destination (with dequantization), cache-aligned
ಸ್ಥಿರ ಶೂನ್ಯ copy_q4_0_ಅಡ್ಡಸಾಲು_cache_aligned(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * src_ಖಂಡಗಳು, int64_t ಅಂಕೆ_elements) {
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (ಅಂಕೆ_elements + QK4_0 - 1) / QK4_0;

    // Scatter byte offsets: even lanes -> dst[j], odd lanes -> dst[j + QK4_0/2]
    // For 4 consecutive packed bytes producing [low0, high0, low1, high1, low2, high2, low3, high3]:
    //   low_i  -> byte offset i*4       (positions 0,1,2,3 in first half)
    //   high_i -> byte offset (16+i)*4  (positions 16,17,18,19 in second half)
    ಸ್ಥಿರಾಂಕ int32_t __attribute__((aligned(32))) scatter_offsets[8] = { 0 * 4, 16 * 4, 1 * 4, 17 * 4,
                                                                      2 * 4, 18 * 4, 3 * 4, 19 * 4 };

    // Gather indices: each byte loaded twice for low/high nibble extraction
    ಸ್ಥಿರಾಂಕ int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 0, 1, 1, 2, 2, 3, 3 };

    uint64_t ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));  // Save current mask
    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load constant vectors once — shared across all blocks and iterations
    __asm__ ಅಸ್ಥಿರ(
        "flq2        f4, 0(%0)    \n\t"  // f4 = scatter offsets
        "flq2        f1, 0(%1)    \n\t"  // f1 = gather indices {0,0,1,1,2,2,3,3}
        ::"r"(scatter_offsets),
        "r"(gather_indices)
        : "f1", "f4");

    ಗಾಗಿ (int64_t ಖಂಡ_idx = 0; ಖಂಡ_idx < ಅಂಕೆ_ಖಂಡಗಳು; ಖಂಡ_idx++) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * ಖಂಡ     = &src_ಖಂಡಗಳು[ಖಂಡ_idx];
        ಸ್ಥಿರಾಂಕ uint8_t *    qs        = ಖಂಡ->qs;
        ದಶಮಾಂಶ *            ಖಂಡ_dst = dst + ಖಂಡ_idx * QK4_0;

        ದಶಮಾಂಶ scale = fp16_ಗಮ್ಯ_fp32(ಖಂಡ->d);
        ದಶಮಾಂಶ bias  = -8.0f * scale;

        // Per-block: broadcast scale and bias
        __asm__ ಅಸ್ಥಿರ(
            "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(scale)
            "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-8 * scale)
            ::"r"(scale),
            "r"(bias)
            : "f0", "f3");

        // 4 iterations x 4 packed bytes = 16 bytes = full block -> 32 floats
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < 4; i++) {
            __asm__ ಅಸ್ಥಿರ(
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
                "r"(ಖಂಡ_dst)
                : "f2", "memory");

            qs += 4;         // 4 packed bytes consumed
            ಖಂಡ_dst += 4;  // Advance base by 4 float positions
        }
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));  // Restore mask
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
ಸ್ಥಿರ ಶೂನ್ಯ copy_q8_0_ಅಡ್ಡಸಾಲು_cache_aligned(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * src_ಖಂಡಗಳು, int64_t ಅಂಕೆ_elements) {
    // Number of Q8_0 blocks needed for this row
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (ಅಂಕೆ_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    ಗಾಗಿ (int64_t ಖಂಡ_idx = 0; ಖಂಡ_idx < ಅಂಕೆ_ಖಂಡಗಳು; ಖಂಡ_idx++) {
        ಸ್ಥಿರಾಂಕ int64_t elements_ಒಳ_ಖಂಡ =
            (ಖಂಡ_idx == ಅಂಕೆ_ಖಂಡಗಳು - 1) ? (ಅಂಕೆ_elements - ಖಂಡ_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        ದಶಮಾಂಶ ತಾತ್ಕಾಲಿಕ_ಬಫರ್[QK8_0];
        dequantize_q8_0_ಖಂಡ_cache_aligned(&src_ಖಂಡಗಳು[ಖಂಡ_idx], ತಾತ್ಕಾಲಿಕ_ಬಫರ್);

        // Copy dequantized values to destination
        ಗಾಗಿ (int64_t i = 0; i < elements_ಒಳ_ಖಂಡ; i++) {
            dst[ಖಂಡ_idx * QK8_0 + i] = ತಾತ್ಕಾಲಿಕ_ಬಫರ್[i];
        }
    }
}

// Vectorized dequantization of a Q4_K super-block (256 elements) to F32
// Processes 8 groups of 32 elements, using ET SIMD for the inner loops.
// Output is sequential (no scatter needed unlike Q4_0).
ಸ್ಥಿರ ಶೂನ್ಯ copy_q4_K_ಅಡ್ಡಸಾಲು_cache_aligned(ದಶಮಾಂಶ * dst, ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * src_ಖಂಡಗಳು, int64_t ಅಂಕೆ_elements) {
    ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_ಖಂಡಗಳು = (ಅಂಕೆ_elements + QK_K - 1) / QK_K;

    // Gather indices for sequential byte access: {0,1,2,3,4,5,6,7}
    ಸ್ಥಿರಾಂಕ int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    uint64_t ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));  // Save current mask
    __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load gather indices once — shared across all blocks
    __asm__ ಅಸ್ಥಿರ("flq2        f1, 0(%0)    \n\t"  // f1 = gather indices {0,1,2,3,4,5,6,7}
                     ::"r"(gather_indices)
                     : "f1");

    ಗಾಗಿ (int64_t ಖಂಡ_idx = 0; ಖಂಡ_idx < ಅಂಕೆ_ಖಂಡಗಳು; ಖಂಡ_idx++) {
        ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * ಖಂಡ     = &src_ಖಂಡಗಳು[ಖಂಡ_idx];
        ಸ್ಥಿರಾಂಕ uint8_t *    qs        = ಖಂಡ->qs;
        ದಶಮಾಂಶ *            ಖಂಡ_dst = dst + ಖಂಡ_idx * QK_K;

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d   = fp16_ಗಮ್ಯ_fp32(ಖಂಡ->d);
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ min = fp16_ಗಮ್ಯ_fp32(ಖಂಡ->dmin);

        ಪೂರ್ಣಾಂಕ is = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < QK_K; j += 64) {
            // Extract per-group scales and mins (scalar — only 8 pairs per super-block)
            uint8_t sc, m;
            get_scale_min_k4(is + 0, ಖಂಡ->scales, &sc, &m);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d1     = d * sc;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ neg_m1 = -(min * m);
            get_scale_min_k4(is + 1, ಖಂಡ->scales, &sc, &m);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ d2     = d * sc;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ neg_m2 = -(min * m);

            // Low nibbles: 32 elements using d1, neg_m1
            __asm__ ಅಸ್ಥಿರ(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d1)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m1)
                ::"r"(d1),
                "r"(neg_m1)
                : "f0", "f3");

            ಸ್ಥಿರಾಂಕ uint8_t * qs_lo  = qs;
            ದಶಮಾಂಶ *         dst_lo = ಖಂಡ_dst + j;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; k++) {
                __asm__ ಅಸ್ಥಿರ(
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
            __asm__ ಅಸ್ಥಿರ(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d2)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m2)
                ::"r"(d2),
                "r"(neg_m2)
                : "f0", "f3");

            ಸ್ಥಿರಾಂಕ uint8_t * qs_hi  = qs;
            ದಶಮಾಂಶ *         dst_hi = ಖಂಡ_dst + j + 32;
            ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 4; k++) {
                __asm__ ಅಸ್ಥಿರ(
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

    __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));  // Restore mask
}

// Determine the number of F32 elements per work unit for a given source type.
// For F32: 1 cacheline (16 elements)
// For quantized types: 1 quant block
ಸ್ಥಿರ int64_t get_elements_ಪ್ರತಿ_work_unit(ಪೂರ್ಣಾಂಕ type) {
    ಸ್ಥಿರಾಂಕ int64_t elements_ಪ್ರತಿ_cacheline = CACHE_ಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು / ಗಾತ್ರ(ದಶಮಾಂಶ);  // 16
    ಆಯ್ಕೆ (type) {
        ಸ್ಥಿತಿ GGML_TYPE_Q8_0:
            ಹಿಂತಿರುಗಿ QK8_0;                   // 32 elements = 2 cachelines
        ಸ್ಥಿತಿ GGML_TYPE_Q4_0:
            ಹಿಂತಿರುಗಿ QK4_0;                   // 32 elements = 2 cachelines
        ಸ್ಥಿತಿ GGML_TYPE_Q4_K:
            ಹಿಂತಿರುಗಿ QK_K;                    // 256 elements = 16 cachelines
        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ elements_ಪ್ರತಿ_cacheline;  // 16 elements = 1 cacheline
    }
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ get_ಅಡ್ಡಸಾಲು_f32_mc_cacheline_aligned(ರಚನೆ ggml_et_get_ಅಡ್ಡಸಾಲುಗಳು_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env  = (kernel_environment_t *) env;
    ಪೂರ್ಣಾಂಕ                    thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ                    ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ರಚನೆ ggml_tensor * src0 = &params->src0;  // Data tensor
    ರಚನೆ ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    ರಚನೆ ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];           // Source columns (row width)
    ಸ್ಥಿರಾಂಕ int64_t ne01 = src0->ne[1];           // Source rows (total available rows)
    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2];           // Source batch dimension
    ಸ್ಥಿರಾಂಕ int64_t ne03 = src0->ne[3];           // Source outer batch dimension

    ಸ್ಥಿರಾಂಕ int64_t ne10 = src1->ne[0];           // Number of indices in dimension 0
    ಸ್ಥಿರಾಂಕ int64_t ne11 = src1->ne[1];           // Number of indices in dimension 1
    ಸ್ಥಿರಾಂಕ int64_t ne12 = src1->ne[2];           // Batch dimension for indices
    ಸ್ಥಿರಾಂಕ int64_t ne13 = src1->ne[3];           // Outer batch dimension for indices

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_ಗಮ್ಯ_ಹೊರತೆಗೆ = ne10 * ne11 * ne12 * ne13;

    // Determine work unit size based on source type
    ಸ್ಥಿರಾಂಕ int64_t elements_ಪ್ರತಿ_wu = get_elements_ಪ್ರತಿ_work_unit(src0->type);
    ಸ್ಥಿರಾಂಕ int64_t wus_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು     = ne00 / elements_ಪ್ರತಿ_wu;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_wus       = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_ಗಮ್ಯ_ಹೊರತೆಗೆ * wus_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

    // Distribute work units across threads (contiguous ranges)
    ಸ್ಥಿರಾಂಕ int64_t wus_ಪ್ರತಿ_thread = (ಒಟ್ಟು_wus + ಅಂಕೆ_ಎಳೆಗಳು - 1) / ಅಂಕೆ_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t wu_ಆರಂಭ       = thread_id * wus_ಪ್ರತಿ_thread;
    int64_t       wu_ಕೊನೆ         = wu_ಆರಂಭ + wus_ಪ್ರತಿ_thread;
    ಆದರೆ (wu_ಕೊನೆ > ಒಟ್ಟು_wus) {
        wu_ಕೊನೆ = ಒಟ್ಟು_wus;
    }

    ಶೂನ್ಯ *    src0_ದತ್ತಾಂಶ = src0->ದತ್ತಾಂಶ;
    int32_t * src1_ದತ್ತಾಂಶ = (int32_t *) src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *   dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    int64_t wu = wu_ಆರಂಭ;
    ಆಗ (wu < wu_ಕೊನೆ) {
        // Determine which row this work unit belongs to and offset within row
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_idx   = wu / wus_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
        ಸ್ಥಿರಾಂಕ int64_t wu_ಒಳ_ಅಡ್ಡಸಾಲು = wu % wus_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;

        // How many work units to process in this row (batch contiguous WUs in same row)
        int64_t wus_ಉಳಿದ_ಒಳ_ಅಡ್ಡಸಾಲು = wus_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು - wu_ಒಳ_ಅಡ್ಡಸಾಲು;
        int64_t wus_ಗಮ್ಯ_process       = wu_ಕೊನೆ - wu;
        ಆದರೆ (wus_ಉಳಿದ_ಒಳ_ಅಡ್ಡಸಾಲು < wus_ಗಮ್ಯ_process) {
            wus_ಗಮ್ಯ_process = wus_ಉಳಿದ_ಒಳ_ಅಡ್ಡಸಾಲು;
        }

        // Calculate multi-dimensional index for this row
        ಸ್ಥಿರಾಂಕ int64_t i       = ಅಡ್ಡಸಾಲು_idx;
        ಸ್ಥಿರಾಂಕ int64_t i13_idx = i / (ne12 * ne11 * ne10);
        ಸ್ಥಿರಾಂಕ int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        ಸ್ಥಿರಾಂಕ int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        ಸ್ಥಿರಾಂಕ int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        ಸ್ಥಿರಾಂಕ int64_t ಸೂಚ್ಯಂಕ_ಸ್ಥಾನಾಂತರ = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        ಸ್ಥಿರಾಂಕ int32_t ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ    = src1_ದತ್ತಾಂಶ[ಸೂಚ್ಯಂಕ_ಸ್ಥಾನಾಂತರ];

        ಆದರೆ (ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ < 0 || ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ >= ne01) {
            ಹಿಂತಿರುಗಿ -1;  // Index out of bounds
        }

        ಸ್ಥಿರಾಂಕ int64_t batch_ಸ್ಥಾನಾಂತರ =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        ಸ್ಥಿರಾಂಕ int64_t elem_ಸ್ಥಾನಾಂತರ_ಒಳ_ಅಡ್ಡಸಾಲು = wu_ಒಳ_ಅಡ್ಡಸಾಲು * elements_ಪ್ರತಿ_wu;
        ಸ್ಥಿರಾಂಕ int64_t ಅಂಕೆ_elements       = wus_ಗಮ್ಯ_process * elements_ಪ್ರತಿ_wu;

        ದಶಮಾಂಶ * dst_ಅಡ್ಡಸಾಲು = dst_ದತ್ತಾಂಶ + ಅಡ್ಡಸಾಲು_idx * ne00 + elem_ಸ್ಥಾನಾಂತರ_ಒಳ_ಅಡ್ಡಸಾಲು;

        ಆದರೆ (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy of cacheline-aligned chunk
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0_ದತ್ತಾಂಶ + ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ * ne00 + batch_ಸ್ಥಾನಾಂತರ + elem_ಸ್ಥಾನಾಂತರ_ಒಳ_ಅಡ್ಡಸಾಲು;
            copy_ಅಡ್ಡಸಾಲು_cache_align(dst_ಅಡ್ಡಸಾಲು, src_ಅಡ್ಡಸಾಲು, ಅಂಕೆ_elements * ಗಾತ್ರ(ದಶಮಾಂಶ));
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion over a destination-aligned write chunk.
            ಸ್ಥಿರಾಂಕ uint16_t * src_ಅಡ್ಡಸಾಲು =
                (ಸ್ಥಿರಾಂಕ uint16_t *) src0_ದತ್ತಾಂಶ + ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ * ne00 + batch_ಸ್ಥಾನಾಂತರ + elem_ಸ್ಥಾನಾಂತರ_ಒಳ_ಅಡ್ಡಸಾಲು;
            copy_f16_ಅಡ್ಡಸಾಲು(dst_ಅಡ್ಡಸಾಲು, src_ಅಡ್ಡಸಾಲು, ಅಂಕೆ_elements);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize work-unit-aligned blocks
            ಸ್ಥಿರಾಂಕ int64_t      ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು   = (ne00 + QK8_0 - 1) / QK8_0;
            ಸ್ಥಿರಾಂಕ int64_t      src_ಖಂಡ_ಸ್ಥಾನಾಂತರ = (ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) + (batch_ಸ್ಥಾನಾಂತರ / ne00) * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
            ಸ್ಥಿರಾಂಕ int64_t      ಖಂಡ_ಆರಂಭ      = elem_ಸ್ಥಾನಾಂತರ_ಒಳ_ಅಡ್ಡಸಾಲು / QK8_0;
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * src_ಖಂಡಗಳು       = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) src0_ದತ್ತಾಂಶ + src_ಖಂಡ_ಸ್ಥಾನಾಂತರ + ಖಂಡ_ಆರಂಭ;
            copy_q8_0_ಅಡ್ಡಸಾಲು_cache_aligned(dst_ಅಡ್ಡಸಾಲು, src_ಖಂಡಗಳು, ಅಂಕೆ_elements);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize work-unit-aligned blocks
            ಸ್ಥಿರಾಂಕ int64_t      ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು   = (ne00 + QK4_0 - 1) / QK4_0;
            ಸ್ಥಿರಾಂಕ int64_t      src_ಖಂಡ_ಸ್ಥಾನಾಂತರ = (ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) + (batch_ಸ್ಥಾನಾಂತರ / ne00) * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
            ಸ್ಥಿರಾಂಕ int64_t      ಖಂಡ_ಆರಂಭ      = elem_ಸ್ಥಾನಾಂತರ_ಒಳ_ಅಡ್ಡಸಾಲು / QK4_0;
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * src_ಖಂಡಗಳು       = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) src0_ದತ್ತಾಂಶ + src_ಖಂಡ_ಸ್ಥಾನಾಂತರ + ಖಂಡ_ಆರಂಭ;
            copy_q4_0_ಅಡ್ಡಸಾಲು_cache_aligned(dst_ಅಡ್ಡಸಾಲು, src_ಖಂಡಗಳು, ಅಂಕೆ_elements);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize work-unit-aligned blocks
            ಸ್ಥಿರಾಂಕ int64_t      ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು   = (ne00 + QK_K - 1) / QK_K;
            ಸ್ಥಿರಾಂಕ int64_t      src_ಖಂಡ_ಸ್ಥಾನಾಂತರ = (ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) + (batch_ಸ್ಥಾನಾಂತರ / ne00) * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
            ಸ್ಥಿರಾಂಕ int64_t      ಖಂಡ_ಆರಂಭ      = elem_ಸ್ಥಾನಾಂತರ_ಒಳ_ಅಡ್ಡಸಾಲು / QK_K;
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * src_ಖಂಡಗಳು       = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K *) src0_ದತ್ತಾಂಶ + src_ಖಂಡ_ಸ್ಥಾನಾಂತರ + ಖಂಡ_ಆರಂಭ;
            copy_q4_K_ಅಡ್ಡಸಾಲು_cache_aligned(dst_ಅಡ್ಡಸಾಲು, src_ಖಂಡಗಳು, ಅಂಕೆ_elements);
        }

        wu += wus_ಗಮ್ಯ_process;
    }

    ಹಿಂತಿರುಗಿ 0;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t tensor_ಬೈಟುಗಳು(ಸ್ಥಿರಾಂಕ ರಚನೆ ggml_tensor * t) {
    ಹಿಂತಿರುಗಿ (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_get_ಅಡ್ಡಸಾಲುಗಳು_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ರಚನೆ ggml_tensor * src0 = &params->src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    ರಚನೆ ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    ರಚನೆ ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    // Fast path - we know how to deal with them multi-core
    ಆದರೆ ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_Q8_0 ||
         src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_K) &&
        src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32 && dst->ne[0] % CACHE_ELEMENTS(ಗಾತ್ರ(ದಶಮಾಂಶ)) == 0) {
        ಹಿಂತಿರುಗಿ get_ಅಡ್ಡಸಾಲು_f32_mc_cacheline_aligned(params, env);
    }

    ಪೂರ್ಣಾಂಕ thread_id = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (thread_id != 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಆದರೆ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;  // Invalid pointer
    }

    ಆದರೆ (dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        ಹಿಂತಿರುಗಿ -1;  // Invalid output or index type
    }

    ಆದರೆ (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q4_K) {
        ಹಿಂತಿರುಗಿ -1;  // Unsupported input type
    }

    ಶೂನ್ಯ *    src0_ದತ್ತಾಂಶ = src0->ದತ್ತಾಂಶ;
    int32_t * src1_ದತ್ತಾಂಶ = (int32_t *) src1->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ *   dst_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ET_UBERKERNEL
    evict_region_past_l2(src0_ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು(src0));
    evict_region_past_l2(src1_ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು(src1));
    evict_region_past_l2(dst_ದತ್ತಾಂಶ, tensor_ಬೈಟುಗಳು(dst));
#ಆದರೆ_ಅಂತ್ಯ

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !src1_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;  // Null data pointer
    }

    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];  // Source columns (row width)
    ಸ್ಥಿರಾಂಕ int64_t ne01 = src0->ne[1];  // Source rows (total available rows)
    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2];  // Source batch dimension
    ಸ್ಥಿರಾಂಕ int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    ಸ್ಥಿರಾಂಕ int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    ಸ್ಥಿರಾಂಕ int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    ಸ್ಥಿರಾಂಕ int64_t ne12 = src1->ne[2];  // Batch dimension for indices
    ಸ್ಥಿರಾಂಕ int64_t ne13 = src1->ne[3];  // Outer batch dimension for indices

    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_ಗಮ್ಯ_ಹೊರತೆಗೆ = ne10 * ne11 * ne12 * ne13;
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ET_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
#ಆದರೆ_ಅಂತ್ಯ
    // Naive single-threaded implementation - process all rows sequentially
    // XXX: Do we really need a single-threaded implementation?
    ಗಾಗಿ (int64_t i = 0; i < ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು_ಗಮ್ಯ_ಹೊರತೆಗೆ; i++) {
        // Calculate multi-dimensional index for the current output position
        ಸ್ಥಿರಾಂಕ int64_t i13_idx = i / (ne12 * ne11 * ne10);
        ಸ್ಥಿರಾಂಕ int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        ಸ್ಥಿರಾಂಕ int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        ಸ್ಥಿರಾಂಕ int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        ಸ್ಥಿರಾಂಕ int64_t ಸೂಚ್ಯಂಕ_ಸ್ಥಾನಾಂತರ = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        ಸ್ಥಿರಾಂಕ int32_t ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ    = src1_ದತ್ತಾಂಶ[ಸೂಚ್ಯಂಕ_ಸ್ಥಾನಾಂತರ];

        ಆದರೆ (ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ < 0 || ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ >= ne01) {
            ಹಿಂತಿರುಗಿ -1;  // Index out of bounds
        }

        ಸ್ಥಿರಾಂಕ int64_t batch_ಸ್ಥಾನಾಂತರ =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        ಸ್ಥಿರಾಂಕ int64_t dst_ಸ್ಥಾನಾಂತರ = i;

        ಆದರೆ (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0_ದತ್ತಾಂಶ + ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ * ne00 + batch_ಸ್ಥಾನಾಂತರ;
            ದಶಮಾಂಶ *       dst_ಅಡ್ಡಸಾಲು = dst_ದತ್ತಾಂಶ + dst_ಸ್ಥಾನಾಂತರ * ne00;
            copy_f32_ಅಡ್ಡಸಾಲು(dst_ಅಡ್ಡಸಾಲು, src_ಅಡ್ಡಸಾಲು, ne00);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion
            ಸ್ಥಿರಾಂಕ uint16_t * src_ಅಡ್ಡಸಾಲು = (ಸ್ಥಿರಾಂಕ uint16_t *) src0_ದತ್ತಾಂಶ + ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ * ne00 + batch_ಸ್ಥಾನಾಂತರ;
            ದಶಮಾಂಶ *          dst_ಅಡ್ಡಸಾಲು = dst_ದತ್ತಾಂಶ + dst_ಸ್ಥಾನಾಂತರ * ne00;
            copy_f16_ಅಡ್ಡಸಾಲು(dst_ಅಡ್ಡಸಾಲು, src_ಅಡ್ಡಸಾಲು, ne00);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize while copying
            ಸ್ಥಿರಾಂಕ int64_t      ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು   = (ne00 + QK8_0 - 1) / QK8_0;
            ಸ್ಥಿರಾಂಕ int64_t      src_ಖಂಡ_ಸ್ಥಾನಾಂತರ = (ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) + (batch_ಸ್ಥಾನಾಂತರ / ne00) * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 * src_ಖಂಡಗಳು       = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q8_0 *) src0_ದತ್ತಾಂಶ + src_ಖಂಡ_ಸ್ಥಾನಾಂತರ;
            ದಶಮಾಂಶ *            dst_ಅಡ್ಡಸಾಲು          = dst_ದತ್ತಾಂಶ + dst_ಸ್ಥಾನಾಂತರ * ne00;
            copy_q8_0_ಅಡ್ಡಸಾಲು(dst_ಅಡ್ಡಸಾಲು, src_ಖಂಡಗಳು, ne00);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize while copying
            ಸ್ಥಿರಾಂಕ int64_t      ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು   = (ne00 + QK4_0 - 1) / QK4_0;
            ಸ್ಥಿರಾಂಕ int64_t      src_ಖಂಡ_ಸ್ಥಾನಾಂತರ = (ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) + (batch_ಸ್ಥಾನಾಂತರ / ne00) * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 * src_ಖಂಡಗಳು       = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_0 *) src0_ದತ್ತಾಂಶ + src_ಖಂಡ_ಸ್ಥಾನಾಂತರ;
            ದಶಮಾಂಶ *            dst_ಅಡ್ಡಸಾಲು          = dst_ದತ್ತಾಂಶ + dst_ಸ್ಥಾನಾಂತರ * ne00;
            copy_q4_0_ಅಡ್ಡಸಾಲು(dst_ಅಡ್ಡಸಾಲು, src_ಖಂಡಗಳು, ne00);
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize while copying
            ಸ್ಥಿರಾಂಕ int64_t      ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು   = (ne00 + QK_K - 1) / QK_K;
            ಸ್ಥಿರಾಂಕ int64_t      src_ಖಂಡ_ಸ್ಥಾನಾಂತರ = (ಅಡ್ಡಸಾಲು_ಸೂಚ್ಯಂಕ * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು) + (batch_ಸ್ಥಾನಾಂತರ / ne00) * ಖಂಡಗಳು_ಪ್ರತಿ_ಅಡ್ಡಸಾಲು;
            ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K * src_ಖಂಡಗಳು       = (ಸ್ಥಿರಾಂಕ ಖಂಡ_q4_K *) src0_ದತ್ತಾಂಶ + src_ಖಂಡ_ಸ್ಥಾನಾಂತರ;
            ದಶಮಾಂಶ *            dst_ಅಡ್ಡಸಾಲು          = dst_ದತ್ತಾಂಶ + dst_ಸ್ಥಾನಾಂತರ * ne00;
            copy_q4_K_ಅಡ್ಡಸಾಲು(dst_ಅಡ್ಡಸಾಲು, src_ಖಂಡಗಳು, ne00);
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
