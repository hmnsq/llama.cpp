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

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "platform.h"
#અન્તર્ભાવ "quants.h"

#અન્તર્ભાવ <assert.h>
#અન્તર્ભાવ <stdbool.h>
#અન્તર્ભાવ <stdint.h>

#define CACHE_લીટી_SIZE_BYTES 64

સંરચના ggml_et_get_rows_params {
    સંરચના ggml_tensor src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    સંરચના ggml_tensor src1;  // Row indices tensor (I32)
    સંરચના ggml_tensor dst;   // Output tensor (F32)
};

#define CACHE_લીટી_SIZE_BYTES     64
#define CACHE_ELEMENTS(elem_size) (CACHE_લીટી_SIZE_BYTES / (elem_size))

// Copy a row of F32 data from source to destination
સ્થિર શૂન્ય copy_f32_row(દશાંશ * dst, અચલ દશાંશ * src, int64_t અંક_elements) {
    // Simple memcpy for F32 data - no conversion needed
    માટે (int64_t i = 0; i < અંક_elements; i++) {
        dst[i] = src[i];
    }
}

સ્થિર શૂન્ય copy_f16_row(દશાંશ * dst, અચલ uint16_t * src, int64_t અંક_elements) {
    માટે (int64_t i = 0; i < અંક_elements; i++) {
        dst[i] = fp16_to_fp32(src[i]);
    }
}

// Copy a row of F32 data from source to destination, aligned to cache line boundaries
// using FP32 load/store instructions. They don't perform data conversion so is fine.
// Requirement: n_bytes is a multiple of CACHE_LINE_SIZE (64 bytes)
સ્થિર શૂન્ય copy_row_cache_align(દશાંશ * dst, અચલ દશાંશ * src, int64_t n_bytes) {
    પૂર્ણાંક અંક_f32_elem = n_bytes / કદ(દશાંશ);

    // Unrolled to do an entire cache line at a time
    __asm__ અસ્થિર(
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

        : [dst] "+r"(dst), [src] "+r"(src), [n] "+r"(અંક_f32_elem)
        : [stride_ગણતરી] "r"(16L)
        : "f0", "f1", "memory");
}

// Copied from GGML: copy a row of Q4_0 data to F32 destination (with dequantization)
સ્થિર શૂન્ય copy_q4_0_row(દશાંશ * dst, અચલ block_q4_0 * src_blocks, int64_t અંક_elements) {
    અચલ int64_t અંક_blocks = (અંક_elements + QK4_0 - 1) / QK4_0;

    માટે (int64_t block_idx = 0; block_idx < અંક_blocks; block_idx++) {
        અચલ int64_t elements_in_block = (block_idx == અંક_blocks - 1) ? (અંક_elements - block_idx * QK4_0) : QK4_0;

        દશાંશ temp_buffer[QK4_0];
        dequantize_q4_0_block(&src_blocks[block_idx], temp_buffer);

        માટે (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK4_0 + i] = temp_buffer[i];
        }
    }
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
સ્થિર શૂન્ય copy_q8_0_row(દશાંશ * dst, અચલ block_q8_0 * src_blocks, int64_t અંક_elements) {
    // Number of Q8_0 blocks needed for this row
    અચલ int64_t અંક_blocks = (અંક_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    માટે (int64_t block_idx = 0; block_idx < અંક_blocks; block_idx++) {
        અચલ int64_t elements_in_block =
            (block_idx == અંક_blocks - 1) ? (અંક_elements - block_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        દશાંશ temp_buffer[QK8_0];
        dequantize_q8_0_block(&src_blocks[block_idx], temp_buffer);

        // Copy dequantized values to destination
        માટે (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK8_0 + i] = temp_buffer[i];
        }
    }
}

// Copy a row of Q4_K data to F32 destination (with dequantization)
સ્થિર શૂન્ય copy_q4_K_row(દશાંશ * dst, અચલ block_q4_K * src_blocks, int64_t અંક_elements) {
    અચલ int64_t અંક_blocks = (અંક_elements + QK_K - 1) / QK_K;

    માટે (int64_t block_idx = 0; block_idx < અંક_blocks; block_idx++) {
        અચલ int64_t elements_in_block = (block_idx == અંક_blocks - 1) ? (અંક_elements - block_idx * QK_K) : QK_K;

        દશાંશ temp_buffer[QK_K];
        dequantize_q4_K_block(&src_blocks[block_idx], temp_buffer);

        માટે (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK_K + i] = temp_buffer[i];
        }
    }
}

સ્થિર શૂન્ય dequantize_q8_0_block_cache_aligned(અચલ block_q8_0 * block, દશાંશ * dst) {
    અચલ int8_t * qs_ptr = block->qs;

    uint64_t temp_mask;
    __asm__ અસ્થિર("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    અચલ int32_t __attribute__((aligned(32))) vec_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    દશાંશ                                      scale          = fp16_to_fp32(block->d);
    __asm__ અસ્થિર(
        "fbcx.ps     f0, %0       \n\t"  // Broadcast integer scale to all lanes
        "flq2        f1, 0(%1)    \n\t"  // Load gether indicies
        ::"r"(scale),
        "r"(vec_indices)
        : "f0", "f1");

    માટે (પૂર્ણાંક i = 0; i < 4; i++) {
        __asm__ અસ્થિર(
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
    __asm__ અસ્થિર("mova.m.x %0" ::"r"(temp_mask));
}

// Copy a row of Q4_0 data to F32 destination (with dequantization), cache-aligned
સ્થિર શૂન્ય copy_q4_0_row_cache_aligned(દશાંશ * dst, અચલ block_q4_0 * src_blocks, int64_t અંક_elements) {
    અચલ int64_t અંક_blocks = (અંક_elements + QK4_0 - 1) / QK4_0;

    // Scatter byte offsets: even lanes -> dst[j], odd lanes -> dst[j + QK4_0/2]
    // For 4 consecutive packed bytes producing [low0, high0, low1, high1, low2, high2, low3, high3]:
    //   low_i  -> byte offset i*4       (positions 0,1,2,3 in first half)
    //   high_i -> byte offset (16+i)*4  (positions 16,17,18,19 in second half)
    અચલ int32_t __attribute__((aligned(32))) scatter_offsets[8] = { 0 * 4, 16 * 4, 1 * 4, 17 * 4,
                                                                      2 * 4, 18 * 4, 3 * 4, 19 * 4 };

    // Gather indices: each byte loaded twice for low/high nibble extraction
    અચલ int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 0, 1, 1, 2, 2, 3, 3 };

    uint64_t temp_mask;
    __asm__ અસ્થિર("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load constant vectors once — shared across all blocks and iterations
    __asm__ અસ્થિર(
        "flq2        f4, 0(%0)    \n\t"  // f4 = scatter offsets
        "flq2        f1, 0(%1)    \n\t"  // f1 = gather indices {0,0,1,1,2,2,3,3}
        ::"r"(scatter_offsets),
        "r"(gather_indices)
        : "f1", "f4");

    માટે (int64_t block_idx = 0; block_idx < અંક_blocks; block_idx++) {
        અચલ block_q4_0 * block     = &src_blocks[block_idx];
        અચલ uint8_t *    qs        = block->qs;
        દશાંશ *            block_dst = dst + block_idx * QK4_0;

        દશાંશ scale = fp16_to_fp32(block->d);
        દશાંશ bias  = -8.0f * scale;

        // Per-block: broadcast scale and bias
        __asm__ અસ્થિર(
            "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(scale)
            "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-8 * scale)
            ::"r"(scale),
            "r"(bias)
            : "f0", "f3");

        // 4 iterations x 4 packed bytes = 16 bytes = full block -> 32 floats
        માટે (પૂર્ણાંક i = 0; i < 4; i++) {
            __asm__ અસ્થિર(
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
                "r"(block_dst)
                : "f2", "memory");

            qs += 4;         // 4 packed bytes consumed
            block_dst += 4;  // Advance base by 4 float positions
        }
    }

    __asm__ અસ્થિર("mova.m.x %0" ::"r"(temp_mask));  // Restore mask
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
સ્થિર શૂન્ય copy_q8_0_row_cache_aligned(દશાંશ * dst, અચલ block_q8_0 * src_blocks, int64_t અંક_elements) {
    // Number of Q8_0 blocks needed for this row
    અચલ int64_t અંક_blocks = (અંક_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    માટે (int64_t block_idx = 0; block_idx < અંક_blocks; block_idx++) {
        અચલ int64_t elements_in_block =
            (block_idx == અંક_blocks - 1) ? (અંક_elements - block_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        દશાંશ temp_buffer[QK8_0];
        dequantize_q8_0_block_cache_aligned(&src_blocks[block_idx], temp_buffer);

        // Copy dequantized values to destination
        માટે (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK8_0 + i] = temp_buffer[i];
        }
    }
}

// Vectorized dequantization of a Q4_K super-block (256 elements) to F32
// Processes 8 groups of 32 elements, using ET SIMD for the inner loops.
// Output is sequential (no scatter needed unlike Q4_0).
સ્થિર શૂન્ય copy_q4_K_row_cache_aligned(દશાંશ * dst, અચલ block_q4_K * src_blocks, int64_t અંક_elements) {
    અચલ int64_t અંક_blocks = (અંક_elements + QK_K - 1) / QK_K;

    // Gather indices for sequential byte access: {0,1,2,3,4,5,6,7}
    અચલ int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    uint64_t temp_mask;
    __asm__ અસ્થિર("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load gather indices once — shared across all blocks
    __asm__ અસ્થિર("flq2        f1, 0(%0)    \n\t"  // f1 = gather indices {0,1,2,3,4,5,6,7}
                     ::"r"(gather_indices)
                     : "f1");

    માટે (int64_t block_idx = 0; block_idx < અંક_blocks; block_idx++) {
        અચલ block_q4_K * block     = &src_blocks[block_idx];
        અચલ uint8_t *    qs        = block->qs;
        દશાંશ *            block_dst = dst + block_idx * QK_K;

        અચલ દશાંશ d   = fp16_to_fp32(block->d);
        અચલ દશાંશ min = fp16_to_fp32(block->dmin);

        પૂર્ણાંક is = 0;
        માટે (પૂર્ણાંક j = 0; j < QK_K; j += 64) {
            // Extract per-group scales and mins (scalar — only 8 pairs per super-block)
            uint8_t sc, m;
            get_scale_min_k4(is + 0, block->scales, &sc, &m);
            અચલ દશાંશ d1     = d * sc;
            અચલ દશાંશ neg_m1 = -(min * m);
            get_scale_min_k4(is + 1, block->scales, &sc, &m);
            અચલ દશાંશ d2     = d * sc;
            અચલ દશાંશ neg_m2 = -(min * m);

            // Low nibbles: 32 elements using d1, neg_m1
            __asm__ અસ્થિર(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d1)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m1)
                ::"r"(d1),
                "r"(neg_m1)
                : "f0", "f3");

            અચલ uint8_t * qs_lo  = qs;
            દશાંશ *         dst_lo = block_dst + j;
            માટે (પૂર્ણાંક k = 0; k < 4; k++) {
                __asm__ અસ્થિર(
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
            __asm__ અસ્થિર(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d2)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m2)
                ::"r"(d2),
                "r"(neg_m2)
                : "f0", "f3");

            અચલ uint8_t * qs_hi  = qs;
            દશાંશ *         dst_hi = block_dst + j + 32;
            માટે (પૂર્ણાંક k = 0; k < 4; k++) {
                __asm__ અસ્થિર(
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

    __asm__ અસ્થિર("mova.m.x %0" ::"r"(temp_mask));  // Restore mask
}

// Determine the number of F32 elements per work unit for a given source type.
// For F32: 1 cacheline (16 elements)
// For quantized types: 1 quant block
સ્થિર int64_t get_elements_per_work_unit(પૂર્ણાંક type) {
    અચલ int64_t elements_per_cacheline = CACHE_લીટી_SIZE_BYTES / કદ(દશાંશ);  // 16
    પસંદગી (type) {
        સ્થિતિ GGML_TYPE_Q8_0:
            પાછા QK8_0;                   // 32 elements = 2 cachelines
        સ્થિતિ GGML_TYPE_Q4_0:
            પાછા QK4_0;                   // 32 elements = 2 cachelines
        સ્થિતિ GGML_TYPE_Q4_K:
            પાછા QK_K;                    // 256 elements = 16 cachelines
        મૂળભૂત:
            પાછા elements_per_cacheline;  // 16 elements = 1 cacheline
    }
}

સ્થિર પૂર્ણાંક get_row_f32_mc_cacheline_aligned(સંરચના ggml_et_get_rows_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env  = (kernel_environment_t *) env;
    પૂર્ણાંક                    thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક                    અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    સંરચના ggml_tensor * src0 = &params->src0;  // Data tensor
    સંરચના ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    સંરચના ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    અચલ int64_t ne00 = src0->ne[0];           // Source columns (row width)
    અચલ int64_t ne01 = src0->ne[1];           // Source rows (total available rows)
    અચલ int64_t ne02 = src0->ne[2];           // Source batch dimension
    અચલ int64_t ne03 = src0->ne[3];           // Source outer batch dimension

    અચલ int64_t ne10 = src1->ne[0];           // Number of indices in dimension 0
    અચલ int64_t ne11 = src1->ne[1];           // Number of indices in dimension 1
    અચલ int64_t ne12 = src1->ne[2];           // Batch dimension for indices
    અચલ int64_t ne13 = src1->ne[3];           // Outer batch dimension for indices

    અચલ int64_t total_rows_to_extract = ne10 * ne11 * ne12 * ne13;

    // Determine work unit size based on source type
    અચલ int64_t elements_per_wu = get_elements_per_work_unit(src0->type);
    અચલ int64_t wus_per_row     = ne00 / elements_per_wu;
    અચલ int64_t total_wus       = total_rows_to_extract * wus_per_row;

    // Distribute work units across threads (contiguous ranges)
    અચલ int64_t wus_per_thread = (total_wus + અંક_threads - 1) / અંક_threads;
    અચલ int64_t wu_start       = thread_id * wus_per_thread;
    int64_t       wu_end         = wu_start + wus_per_thread;
    જો (wu_end > total_wus) {
        wu_end = total_wus;
    }

    શૂન્ય *    src0_data = src0->data;
    int32_t * src1_data = (int32_t *) src1->data;
    દશાંશ *   dst_data  = (દશાંશ *) dst->data;

    int64_t wu = wu_start;
    જ્યારે (wu < wu_end) {
        // Determine which row this work unit belongs to and offset within row
        અચલ int64_t row_idx   = wu / wus_per_row;
        અચલ int64_t wu_in_row = wu % wus_per_row;

        // How many work units to process in this row (batch contiguous WUs in same row)
        int64_t wus_remaining_in_row = wus_per_row - wu_in_row;
        int64_t wus_to_process       = wu_end - wu;
        જો (wus_remaining_in_row < wus_to_process) {
            wus_to_process = wus_remaining_in_row;
        }

        // Calculate multi-dimensional index for this row
        અચલ int64_t i       = row_idx;
        અચલ int64_t i13_idx = i / (ne12 * ne11 * ne10);
        અચલ int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        અચલ int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        અચલ int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        અચલ int64_t index_offset = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        અચલ int32_t row_index    = src1_data[index_offset];

        જો (row_index < 0 || row_index >= ne01) {
            પાછા -1;  // Index out of bounds
        }

        અચલ int64_t batch_offset =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        અચલ int64_t elem_offset_in_row = wu_in_row * elements_per_wu;
        અચલ int64_t અંક_elements       = wus_to_process * elements_per_wu;

        દશાંશ * dst_row = dst_data + row_idx * ne00 + elem_offset_in_row;

        જો (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy of cacheline-aligned chunk
            અચલ દશાંશ * src_row = (અચલ દશાંશ *) src0_data + row_index * ne00 + batch_offset + elem_offset_in_row;
            copy_row_cache_align(dst_row, src_row, અંક_elements * કદ(દશાંશ));
        } નહીંતર જો (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion over a destination-aligned write chunk.
            અચલ uint16_t * src_row =
                (અચલ uint16_t *) src0_data + row_index * ne00 + batch_offset + elem_offset_in_row;
            copy_f16_row(dst_row, src_row, અંક_elements);
        } નહીંતર જો (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize work-unit-aligned blocks
            અચલ int64_t      blocks_per_row   = (ne00 + QK8_0 - 1) / QK8_0;
            અચલ int64_t      src_block_offset = (row_index * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            અચલ int64_t      block_start      = elem_offset_in_row / QK8_0;
            અચલ block_q8_0 * src_blocks       = (અચલ block_q8_0 *) src0_data + src_block_offset + block_start;
            copy_q8_0_row_cache_aligned(dst_row, src_blocks, અંક_elements);
        } નહીંતર જો (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize work-unit-aligned blocks
            અચલ int64_t      blocks_per_row   = (ne00 + QK4_0 - 1) / QK4_0;
            અચલ int64_t      src_block_offset = (row_index * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            અચલ int64_t      block_start      = elem_offset_in_row / QK4_0;
            અચલ block_q4_0 * src_blocks       = (અચલ block_q4_0 *) src0_data + src_block_offset + block_start;
            copy_q4_0_row_cache_aligned(dst_row, src_blocks, અંક_elements);
        } નહીંતર જો (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize work-unit-aligned blocks
            અચલ int64_t      blocks_per_row   = (ne00 + QK_K - 1) / QK_K;
            અચલ int64_t      src_block_offset = (row_index * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            અચલ int64_t      block_start      = elem_offset_in_row / QK_K;
            અચલ block_q4_K * src_blocks       = (અચલ block_q4_K *) src0_data + src_block_offset + block_start;
            copy_q4_K_row_cache_aligned(dst_row, src_blocks, અંક_elements);
        }

        wu += wus_to_process;
    }

    પાછા 0;
}

સ્થિર અંતઃરેખ size_t tensor_bytes(અચલ સંરચના ggml_tensor * t) {
    પાછા (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

પૂર્ણાંક entry_point(સંરચના ggml_et_get_rows_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    જો (!kernel_env) {
        પાછા -1;
    }

    સંરચના ggml_tensor * src0 = &params->src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    સંરચના ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    સંરચના ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    // Fast path - we know how to deal with them multi-core
    જો ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_Q8_0 ||
         src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_K) &&
        src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32 && dst->ne[0] % CACHE_ELEMENTS(કદ(દશાંશ)) == 0) {
        પાછા get_row_f32_mc_cacheline_aligned(params, env);
    }

    પૂર્ણાંક thread_id = get_relative_thread_id(kernel_env->shire_mask);
    જો (thread_id < 0) {
        પાછા 0;
    }

    જો (thread_id != 0) {
        પાછા 0;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;  // Invalid pointer
    }

    જો (dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        પાછા -1;  // Invalid output or index type
    }

    જો (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q4_K) {
        પાછા -1;  // Unsupported input type
    }

    શૂન્ય *    src0_data = src0->data;
    int32_t * src1_data = (int32_t *) src1->data;
    દશાંશ *   dst_data  = (દશાંશ *) dst->data;
#ifdef ET_UBERKERNEL
    evict_region_past_l2(src0_data, tensor_bytes(src0));
    evict_region_past_l2(src1_data, tensor_bytes(src1));
    evict_region_past_l2(dst_data, tensor_bytes(dst));
#endif

    જો (!src0_data || !src1_data || !dst_data) {
        પાછા -1;  // Null data pointer
    }

    અચલ int64_t ne00 = src0->ne[0];  // Source columns (row width)
    અચલ int64_t ne01 = src0->ne[1];  // Source rows (total available rows)
    અચલ int64_t ne02 = src0->ne[2];  // Source batch dimension
    અચલ int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    અચલ int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    અચલ int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    અચલ int64_t ne12 = src1->ne[2];  // Batch dimension for indices
    અચલ int64_t ne13 = src1->ne[3];  // Outer batch dimension for indices

    અચલ int64_t total_rows_to_extract = ne10 * ne11 * ne12 * ne13;
#ifdef ET_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    // Naive single-threaded implementation - process all rows sequentially
    // XXX: Do we really need a single-threaded implementation?
    માટે (int64_t i = 0; i < total_rows_to_extract; i++) {
        // Calculate multi-dimensional index for the current output position
        અચલ int64_t i13_idx = i / (ne12 * ne11 * ne10);
        અચલ int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        અચલ int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        અચલ int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        અચલ int64_t index_offset = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        અચલ int32_t row_index    = src1_data[index_offset];

        જો (row_index < 0 || row_index >= ne01) {
            પાછા -1;  // Index out of bounds
        }

        અચલ int64_t batch_offset =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        અચલ int64_t dst_offset = i;

        જો (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy
            અચલ દશાંશ * src_row = (અચલ દશાંશ *) src0_data + row_index * ne00 + batch_offset;
            દશાંશ *       dst_row = dst_data + dst_offset * ne00;
            copy_f32_row(dst_row, src_row, ne00);
        } નહીંતર જો (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion
            અચલ uint16_t * src_row = (અચલ uint16_t *) src0_data + row_index * ne00 + batch_offset;
            દશાંશ *          dst_row = dst_data + dst_offset * ne00;
            copy_f16_row(dst_row, src_row, ne00);
        } નહીંતર જો (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize while copying
            અચલ int64_t      blocks_per_row   = (ne00 + QK8_0 - 1) / QK8_0;
            અચલ int64_t      src_block_offset = (row_index * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            અચલ block_q8_0 * src_blocks       = (અચલ block_q8_0 *) src0_data + src_block_offset;
            દશાંશ *            dst_row          = dst_data + dst_offset * ne00;
            copy_q8_0_row(dst_row, src_blocks, ne00);
        } નહીંતર જો (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize while copying
            અચલ int64_t      blocks_per_row   = (ne00 + QK4_0 - 1) / QK4_0;
            અચલ int64_t      src_block_offset = (row_index * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            અચલ block_q4_0 * src_blocks       = (અચલ block_q4_0 *) src0_data + src_block_offset;
            દશાંશ *            dst_row          = dst_data + dst_offset * ne00;
            copy_q4_0_row(dst_row, src_blocks, ne00);
        } નહીંતર જો (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize while copying
            અચલ int64_t      blocks_per_row   = (ne00 + QK_K - 1) / QK_K;
            અચલ int64_t      src_block_offset = (row_index * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            અચલ block_q4_K * src_blocks       = (અચલ block_q4_K *) src0_data + src_block_offset;
            દશાંશ *            dst_row          = dst_data + dst_offset * ne00;
            copy_q4_K_row(dst_row, src_blocks, ne00);
        }
    }

    પાછા 0;
}
