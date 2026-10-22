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

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "platform.h"
#অন্তর্ভাব "quants.h"

#অন্তর্ভাব <assert.h>
#অন্তর্ভাব <stdbool.h>
#অন্তর্ভাব <stdint.h>

#define CACHE_সারি_আকার_BYTES 64

কাঠামো ggml_et_get_rows_params {
    কাঠামো ggml_tensor src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    কাঠামো ggml_tensor src1;  // Row indices tensor (I32)
    কাঠামো ggml_tensor dst;   // Output tensor (F32)
};

#define CACHE_সারি_আকার_BYTES     64
#define CACHE_ELEMENTS(elem_আকার) (CACHE_সারি_আকার_BYTES / (elem_আকার))

// Copy a row of F32 data from source to destination
স্থির শূন্য copy_f32_row(দশমিক * dst, ধ্রুবক দশমিক * src, int64_t অঙ্ক_elements) {
    // Simple memcpy for F32 data - no conversion needed
    জন্য (int64_t i = 0; i < অঙ্ক_elements; i++) {
        dst[i] = src[i];
    }
}

স্থির শূন্য copy_f16_row(দশমিক * dst, ধ্রুবক uint16_t * src, int64_t অঙ্ক_elements) {
    জন্য (int64_t i = 0; i < অঙ্ক_elements; i++) {
        dst[i] = fp16_to_fp32(src[i]);
    }
}

// Copy a row of F32 data from source to destination, aligned to cache line boundaries
// using FP32 load/store instructions. They don't perform data conversion so is fine.
// Requirement: n_bytes is a multiple of CACHE_LINE_SIZE (64 bytes)
স্থির শূন্য copy_row_cache_align(দশমিক * dst, ধ্রুবক দশমিক * src, int64_t n_bytes) {
    পূর্ণসংখ্যা অঙ্ক_f32_elem = n_bytes / আকার(দশমিক);

    // Unrolled to do an entire cache line at a time
    __asm__ অস্থির(
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

        : [dst] "+r"(dst), [src] "+r"(src), [n] "+r"(অঙ্ক_f32_elem)
        : [stride_গুনতি] "r"(16L)
        : "f0", "f1", "memory");
}

// Copied from GGML: copy a row of Q4_0 data to F32 destination (with dequantization)
স্থির শূন্য copy_q4_0_row(দশমিক * dst, ধ্রুবক block_q4_0 * src_blocks, int64_t অঙ্ক_elements) {
    ধ্রুবক int64_t অঙ্ক_blocks = (অঙ্ক_elements + QK4_0 - 1) / QK4_0;

    জন্য (int64_t block_idx = 0; block_idx < অঙ্ক_blocks; block_idx++) {
        ধ্রুবক int64_t elements_in_block = (block_idx == অঙ্ক_blocks - 1) ? (অঙ্ক_elements - block_idx * QK4_0) : QK4_0;

        দশমিক অস্থায়ী_বাফার[QK4_0];
        dequantize_q4_0_block(&src_blocks[block_idx], অস্থায়ী_বাফার);

        জন্য (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK4_0 + i] = অস্থায়ী_বাফার[i];
        }
    }
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
স্থির শূন্য copy_q8_0_row(দশমিক * dst, ধ্রুবক block_q8_0 * src_blocks, int64_t অঙ্ক_elements) {
    // Number of Q8_0 blocks needed for this row
    ধ্রুবক int64_t অঙ্ক_blocks = (অঙ্ক_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    জন্য (int64_t block_idx = 0; block_idx < অঙ্ক_blocks; block_idx++) {
        ধ্রুবক int64_t elements_in_block =
            (block_idx == অঙ্ক_blocks - 1) ? (অঙ্ক_elements - block_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        দশমিক অস্থায়ী_বাফার[QK8_0];
        dequantize_q8_0_block(&src_blocks[block_idx], অস্থায়ী_বাফার);

        // Copy dequantized values to destination
        জন্য (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK8_0 + i] = অস্থায়ী_বাফার[i];
        }
    }
}

// Copy a row of Q4_K data to F32 destination (with dequantization)
স্থির শূন্য copy_q4_K_row(দশমিক * dst, ধ্রুবক block_q4_K * src_blocks, int64_t অঙ্ক_elements) {
    ধ্রুবক int64_t অঙ্ক_blocks = (অঙ্ক_elements + QK_K - 1) / QK_K;

    জন্য (int64_t block_idx = 0; block_idx < অঙ্ক_blocks; block_idx++) {
        ধ্রুবক int64_t elements_in_block = (block_idx == অঙ্ক_blocks - 1) ? (অঙ্ক_elements - block_idx * QK_K) : QK_K;

        দশমিক অস্থায়ী_বাফার[QK_K];
        dequantize_q4_K_block(&src_blocks[block_idx], অস্থায়ী_বাফার);

        জন্য (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK_K + i] = অস্থায়ী_বাফার[i];
        }
    }
}

স্থির শূন্য dequantize_q8_0_block_cache_aligned(ধ্রুবক block_q8_0 * block, দশমিক * dst) {
    ধ্রুবক int8_t * qs_ptr = block->qs;

    uint64_t অস্থায়ী_mask;
    __asm__ অস্থির("mova.x.m %0" : "=r"(অস্থায়ী_mask));  // Save current mask
    __asm__ অস্থির("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    ধ্রুবক int32_t __attribute__((aligned(32))) vec_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    দশমিক                                      scale          = fp16_to_fp32(block->d);
    __asm__ অস্থির(
        "fbcx.ps     f0, %0       \n\t"  // Broadcast integer scale to all lanes
        "flq2        f1, 0(%1)    \n\t"  // Load gether indicies
        ::"r"(scale),
        "r"(vec_indices)
        : "f0", "f1");

    জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
        __asm__ অস্থির(
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
    __asm__ অস্থির("mova.m.x %0" ::"r"(অস্থায়ী_mask));
}

// Copy a row of Q4_0 data to F32 destination (with dequantization), cache-aligned
স্থির শূন্য copy_q4_0_row_cache_aligned(দশমিক * dst, ধ্রুবক block_q4_0 * src_blocks, int64_t অঙ্ক_elements) {
    ধ্রুবক int64_t অঙ্ক_blocks = (অঙ্ক_elements + QK4_0 - 1) / QK4_0;

    // Scatter byte offsets: even lanes -> dst[j], odd lanes -> dst[j + QK4_0/2]
    // For 4 consecutive packed bytes producing [low0, high0, low1, high1, low2, high2, low3, high3]:
    //   low_i  -> byte offset i*4       (positions 0,1,2,3 in first half)
    //   high_i -> byte offset (16+i)*4  (positions 16,17,18,19 in second half)
    ধ্রুবক int32_t __attribute__((aligned(32))) scatter_offsets[8] = { 0 * 4, 16 * 4, 1 * 4, 17 * 4,
                                                                      2 * 4, 18 * 4, 3 * 4, 19 * 4 };

    // Gather indices: each byte loaded twice for low/high nibble extraction
    ধ্রুবক int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 0, 1, 1, 2, 2, 3, 3 };

    uint64_t অস্থায়ী_mask;
    __asm__ অস্থির("mova.x.m %0" : "=r"(অস্থায়ী_mask));  // Save current mask
    __asm__ অস্থির("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load constant vectors once — shared across all blocks and iterations
    __asm__ অস্থির(
        "flq2        f4, 0(%0)    \n\t"  // f4 = scatter offsets
        "flq2        f1, 0(%1)    \n\t"  // f1 = gather indices {0,0,1,1,2,2,3,3}
        ::"r"(scatter_offsets),
        "r"(gather_indices)
        : "f1", "f4");

    জন্য (int64_t block_idx = 0; block_idx < অঙ্ক_blocks; block_idx++) {
        ধ্রুবক block_q4_0 * block     = &src_blocks[block_idx];
        ধ্রুবক uint8_t *    qs        = block->qs;
        দশমিক *            block_dst = dst + block_idx * QK4_0;

        দশমিক scale = fp16_to_fp32(block->d);
        দশমিক bias  = -8.0f * scale;

        // Per-block: broadcast scale and bias
        __asm__ অস্থির(
            "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(scale)
            "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-8 * scale)
            ::"r"(scale),
            "r"(bias)
            : "f0", "f3");

        // 4 iterations x 4 packed bytes = 16 bytes = full block -> 32 floats
        জন্য (পূর্ণসংখ্যা i = 0; i < 4; i++) {
            __asm__ অস্থির(
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

    __asm__ অস্থির("mova.m.x %0" ::"r"(অস্থায়ী_mask));  // Restore mask
}

// Copy a row of Q8_0 data to F32 destination (with dequantization)
স্থির শূন্য copy_q8_0_row_cache_aligned(দশমিক * dst, ধ্রুবক block_q8_0 * src_blocks, int64_t অঙ্ক_elements) {
    // Number of Q8_0 blocks needed for this row
    ধ্রুবক int64_t অঙ্ক_blocks = (অঙ্ক_elements + QK8_0 - 1) / QK8_0;  // Round up to handle partial blocks

    জন্য (int64_t block_idx = 0; block_idx < অঙ্ক_blocks; block_idx++) {
        ধ্রুবক int64_t elements_in_block =
            (block_idx == অঙ্ক_blocks - 1) ? (অঙ্ক_elements - block_idx * QK8_0) : QK8_0;  // Handle last partial block

        // Dequantize the block
        দশমিক অস্থায়ী_বাফার[QK8_0];
        dequantize_q8_0_block_cache_aligned(&src_blocks[block_idx], অস্থায়ী_বাফার);

        // Copy dequantized values to destination
        জন্য (int64_t i = 0; i < elements_in_block; i++) {
            dst[block_idx * QK8_0 + i] = অস্থায়ী_বাফার[i];
        }
    }
}

// Vectorized dequantization of a Q4_K super-block (256 elements) to F32
// Processes 8 groups of 32 elements, using ET SIMD for the inner loops.
// Output is sequential (no scatter needed unlike Q4_0).
স্থির শূন্য copy_q4_K_row_cache_aligned(দশমিক * dst, ধ্রুবক block_q4_K * src_blocks, int64_t অঙ্ক_elements) {
    ধ্রুবক int64_t অঙ্ক_blocks = (অঙ্ক_elements + QK_K - 1) / QK_K;

    // Gather indices for sequential byte access: {0,1,2,3,4,5,6,7}
    ধ্রুবক int32_t __attribute__((aligned(32))) gather_indices[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    uint64_t অস্থায়ী_mask;
    __asm__ অস্থির("mova.x.m %0" : "=r"(অস্থায়ী_mask));  // Save current mask
    __asm__ অস্থির("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Load gather indices once — shared across all blocks
    __asm__ অস্থির("flq2        f1, 0(%0)    \n\t"  // f1 = gather indices {0,1,2,3,4,5,6,7}
                     ::"r"(gather_indices)
                     : "f1");

    জন্য (int64_t block_idx = 0; block_idx < অঙ্ক_blocks; block_idx++) {
        ধ্রুবক block_q4_K * block     = &src_blocks[block_idx];
        ধ্রুবক uint8_t *    qs        = block->qs;
        দশমিক *            block_dst = dst + block_idx * QK_K;

        ধ্রুবক দশমিক d   = fp16_to_fp32(block->d);
        ধ্রুবক দশমিক min = fp16_to_fp32(block->dmin);

        পূর্ণসংখ্যা is = 0;
        জন্য (পূর্ণসংখ্যা j = 0; j < QK_K; j += 64) {
            // Extract per-group scales and mins (scalar — only 8 pairs per super-block)
            uint8_t sc, m;
            get_scale_min_k4(is + 0, block->scales, &sc, &m);
            ধ্রুবক দশমিক d1     = d * sc;
            ধ্রুবক দশমিক neg_m1 = -(min * m);
            get_scale_min_k4(is + 1, block->scales, &sc, &m);
            ধ্রুবক দশমিক d2     = d * sc;
            ধ্রুবক দশমিক neg_m2 = -(min * m);

            // Low nibbles: 32 elements using d1, neg_m1
            __asm__ অস্থির(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d1)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m1)
                ::"r"(d1),
                "r"(neg_m1)
                : "f0", "f3");

            ধ্রুবক uint8_t * qs_lo  = qs;
            দশমিক *         dst_lo = block_dst + j;
            জন্য (পূর্ণসংখ্যা k = 0; k < 4; k++) {
                __asm__ অস্থির(
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
            __asm__ অস্থির(
                "fbcx.ps     f0, %0       \n\t"  // f0 = broadcast(d2)
                "fbcx.ps     f3, %1       \n\t"  // f3 = broadcast(-m2)
                ::"r"(d2),
                "r"(neg_m2)
                : "f0", "f3");

            ধ্রুবক uint8_t * qs_hi  = qs;
            দশমিক *         dst_hi = block_dst + j + 32;
            জন্য (পূর্ণসংখ্যা k = 0; k < 4; k++) {
                __asm__ অস্থির(
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

    __asm__ অস্থির("mova.m.x %0" ::"r"(অস্থায়ী_mask));  // Restore mask
}

// Determine the number of F32 elements per work unit for a given source type.
// For F32: 1 cacheline (16 elements)
// For quantized types: 1 quant block
স্থির int64_t get_elements_per_work_unit(পূর্ণসংখ্যা type) {
    ধ্রুবক int64_t elements_per_cacheline = CACHE_সারি_আকার_BYTES / আকার(দশমিক);  // 16
    নির্বাচন (type) {
        ক্ষেত্র GGML_TYPE_Q8_0:
            ফেরত QK8_0;                   // 32 elements = 2 cachelines
        ক্ষেত্র GGML_TYPE_Q4_0:
            ফেরত QK4_0;                   // 32 elements = 2 cachelines
        ক্ষেত্র GGML_TYPE_Q4_K:
            ফেরত QK_K;                    // 256 elements = 16 cachelines
        পূর্বনির্ধারিত:
            ফেরত elements_per_cacheline;  // 16 elements = 1 cacheline
    }
}

স্থির পূর্ণসংখ্যা get_row_f32_mc_cacheline_aligned(কাঠামো ggml_et_get_rows_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env  = (kernel_environment_t *) env;
    পূর্ণসংখ্যা                    thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা                    অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    কাঠামো ggml_tensor * src0 = &params->src0;  // Data tensor
    কাঠামো ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    কাঠামো ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    ধ্রুবক int64_t ne00 = src0->ne[0];           // Source columns (row width)
    ধ্রুবক int64_t ne01 = src0->ne[1];           // Source rows (total available rows)
    ধ্রুবক int64_t ne02 = src0->ne[2];           // Source batch dimension
    ধ্রুবক int64_t ne03 = src0->ne[3];           // Source outer batch dimension

    ধ্রুবক int64_t ne10 = src1->ne[0];           // Number of indices in dimension 0
    ধ্রুবক int64_t ne11 = src1->ne[1];           // Number of indices in dimension 1
    ধ্রুবক int64_t ne12 = src1->ne[2];           // Batch dimension for indices
    ধ্রুবক int64_t ne13 = src1->ne[3];           // Outer batch dimension for indices

    ধ্রুবক int64_t যোগ_rows_to_extract = ne10 * ne11 * ne12 * ne13;

    // Determine work unit size based on source type
    ধ্রুবক int64_t elements_per_wu = get_elements_per_work_unit(src0->type);
    ধ্রুবক int64_t wus_per_row     = ne00 / elements_per_wu;
    ধ্রুবক int64_t যোগ_wus       = যোগ_rows_to_extract * wus_per_row;

    // Distribute work units across threads (contiguous ranges)
    ধ্রুবক int64_t wus_per_thread = (যোগ_wus + অঙ্ক_threads - 1) / অঙ্ক_threads;
    ধ্রুবক int64_t wu_start       = thread_id * wus_per_thread;
    int64_t       wu_end         = wu_start + wus_per_thread;
    যদি (wu_end > যোগ_wus) {
        wu_end = যোগ_wus;
    }

    শূন্য *    src0_তথ্য = src0->তথ্য;
    int32_t * src1_তথ্য = (int32_t *) src1->তথ্য;
    দশমিক *   dst_তথ্য  = (দশমিক *) dst->তথ্য;

    int64_t wu = wu_start;
    যখন (wu < wu_end) {
        // Determine which row this work unit belongs to and offset within row
        ধ্রুবক int64_t row_idx   = wu / wus_per_row;
        ধ্রুবক int64_t wu_in_row = wu % wus_per_row;

        // How many work units to process in this row (batch contiguous WUs in same row)
        int64_t wus_remaining_in_row = wus_per_row - wu_in_row;
        int64_t wus_to_process       = wu_end - wu;
        যদি (wus_remaining_in_row < wus_to_process) {
            wus_to_process = wus_remaining_in_row;
        }

        // Calculate multi-dimensional index for this row
        ধ্রুবক int64_t i       = row_idx;
        ধ্রুবক int64_t i13_idx = i / (ne12 * ne11 * ne10);
        ধ্রুবক int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        ধ্রুবক int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        ধ্রুবক int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        ধ্রুবক int64_t সূচক_offset = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        ধ্রুবক int32_t row_সূচক    = src1_তথ্য[সূচক_offset];

        যদি (row_সূচক < 0 || row_সূচক >= ne01) {
            ফেরত -1;  // Index out of bounds
        }

        ধ্রুবক int64_t batch_offset =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        ধ্রুবক int64_t elem_offset_in_row = wu_in_row * elements_per_wu;
        ধ্রুবক int64_t অঙ্ক_elements       = wus_to_process * elements_per_wu;

        দশমিক * dst_row = dst_তথ্য + row_idx * ne00 + elem_offset_in_row;

        যদি (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy of cacheline-aligned chunk
            ধ্রুবক দশমিক * src_row = (ধ্রুবক দশমিক *) src0_তথ্য + row_সূচক * ne00 + batch_offset + elem_offset_in_row;
            copy_row_cache_align(dst_row, src_row, অঙ্ক_elements * আকার(দশমিক));
        } অন্যথা যদি (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion over a destination-aligned write chunk.
            ধ্রুবক uint16_t * src_row =
                (ধ্রুবক uint16_t *) src0_তথ্য + row_সূচক * ne00 + batch_offset + elem_offset_in_row;
            copy_f16_row(dst_row, src_row, অঙ্ক_elements);
        } অন্যথা যদি (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize work-unit-aligned blocks
            ধ্রুবক int64_t      blocks_per_row   = (ne00 + QK8_0 - 1) / QK8_0;
            ধ্রুবক int64_t      src_block_offset = (row_সূচক * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            ধ্রুবক int64_t      block_start      = elem_offset_in_row / QK8_0;
            ধ্রুবক block_q8_0 * src_blocks       = (ধ্রুবক block_q8_0 *) src0_তথ্য + src_block_offset + block_start;
            copy_q8_0_row_cache_aligned(dst_row, src_blocks, অঙ্ক_elements);
        } অন্যথা যদি (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize work-unit-aligned blocks
            ধ্রুবক int64_t      blocks_per_row   = (ne00 + QK4_0 - 1) / QK4_0;
            ধ্রুবক int64_t      src_block_offset = (row_সূচক * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            ধ্রুবক int64_t      block_start      = elem_offset_in_row / QK4_0;
            ধ্রুবক block_q4_0 * src_blocks       = (ধ্রুবক block_q4_0 *) src0_তথ্য + src_block_offset + block_start;
            copy_q4_0_row_cache_aligned(dst_row, src_blocks, অঙ্ক_elements);
        } অন্যথা যদি (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize work-unit-aligned blocks
            ধ্রুবক int64_t      blocks_per_row   = (ne00 + QK_K - 1) / QK_K;
            ধ্রুবক int64_t      src_block_offset = (row_সূচক * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            ধ্রুবক int64_t      block_start      = elem_offset_in_row / QK_K;
            ধ্রুবক block_q4_K * src_blocks       = (ধ্রুবক block_q4_K *) src0_তথ্য + src_block_offset + block_start;
            copy_q4_K_row_cache_aligned(dst_row, src_blocks, অঙ্ক_elements);
        }

        wu += wus_to_process;
    }

    ফেরত 0;
}

স্থির অন্তঃরেখ size_t tensor_bytes(ধ্রুবক কাঠামো ggml_tensor * t) {
    ফেরত (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_get_rows_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;
    যদি (!kernel_env) {
        ফেরত -1;
    }

    কাঠামো ggml_tensor * src0 = &params->src0;  // Data tensor (F32, Q4_0, Q8_0, or Q4_K)
    কাঠামো ggml_tensor * src1 = &params->src1;  // Row indices tensor (I32)
    কাঠামো ggml_tensor * dst  = &params->dst;   // Output tensor (F32)

    // Fast path - we know how to deal with them multi-core
    যদি ((src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_Q8_0 ||
         src0->type == GGML_TYPE_Q4_0 || src0->type == GGML_TYPE_Q4_K) &&
        src1->type == GGML_TYPE_I32 && dst->type == GGML_TYPE_F32 && dst->ne[0] % CACHE_ELEMENTS(আকার(দশমিক)) == 0) {
        ফেরত get_row_f32_mc_cacheline_aligned(params, env);
    }

    পূর্ণসংখ্যা thread_id = get_relative_thread_id(kernel_env->shire_mask);
    যদি (thread_id < 0) {
        ফেরত 0;
    }

    যদি (thread_id != 0) {
        ফেরত 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;  // Invalid pointer
    }

    যদি (dst->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32) {
        ফেরত -1;  // Invalid output or index type
    }

    যদি (src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_Q8_0 &&
        src0->type != GGML_TYPE_Q4_0 && src0->type != GGML_TYPE_Q4_K) {
        ফেরত -1;  // Unsupported input type
    }

    শূন্য *    src0_তথ্য = src0->তথ্য;
    int32_t * src1_তথ্য = (int32_t *) src1->তথ্য;
    দশমিক *   dst_তথ্য  = (দশমিক *) dst->তথ্য;
#ifdef ET_UBERKERNEL
    evict_region_past_l2(src0_তথ্য, tensor_bytes(src0));
    evict_region_past_l2(src1_তথ্য, tensor_bytes(src1));
    evict_region_past_l2(dst_তথ্য, tensor_bytes(dst));
#endif

    যদি (!src0_তথ্য || !src1_তথ্য || !dst_তথ্য) {
        ফেরত -1;  // Null data pointer
    }

    ধ্রুবক int64_t ne00 = src0->ne[0];  // Source columns (row width)
    ধ্রুবক int64_t ne01 = src0->ne[1];  // Source rows (total available rows)
    ধ্রুবক int64_t ne02 = src0->ne[2];  // Source batch dimension
    ধ্রুবক int64_t ne03 = src0->ne[3];  // Source outer batch dimension

    ধ্রুবক int64_t ne10 = src1->ne[0];  // Number of indices in dimension 0
    ধ্রুবক int64_t ne11 = src1->ne[1];  // Number of indices in dimension 1
    ধ্রুবক int64_t ne12 = src1->ne[2];  // Batch dimension for indices
    ধ্রুবক int64_t ne13 = src1->ne[3];  // Outer batch dimension for indices

    ধ্রুবক int64_t যোগ_rows_to_extract = ne10 * ne11 * ne12 * ne13;
#ifdef ET_UBERKERNEL
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    // Naive single-threaded implementation - process all rows sequentially
    // XXX: Do we really need a single-threaded implementation?
    জন্য (int64_t i = 0; i < যোগ_rows_to_extract; i++) {
        // Calculate multi-dimensional index for the current output position
        ধ্রুবক int64_t i13_idx = i / (ne12 * ne11 * ne10);
        ধ্রুবক int64_t i12_idx = (i - i13_idx * ne12 * ne11 * ne10) / (ne11 * ne10);
        ধ্রুবক int64_t i11_idx = (i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10) / ne10;
        ধ্রুবক int64_t i10_idx = i - i13_idx * ne12 * ne11 * ne10 - i12_idx * ne11 * ne10 - i11_idx * ne10;

        // Get the row index from src1
        ধ্রুবক int64_t সূচক_offset = i13_idx * ne12 * ne11 * ne10 + i12_idx * ne11 * ne10 + i11_idx * ne10 + i10_idx;
        ধ্রুবক int32_t row_সূচক    = src1_তথ্য[সূচক_offset];

        যদি (row_সূচক < 0 || row_সূচক >= ne01) {
            ফেরত -1;  // Index out of bounds
        }

        ধ্রুবক int64_t batch_offset =
            i11_idx * ne01 * ne00 + i12_idx * ne02 * ne01 * ne00 + i13_idx * ne03 * ne02 * ne01 * ne00;

        ধ্রুবক int64_t dst_offset = i;

        যদি (src0->type == GGML_TYPE_F32) {
            // F32 source: direct copy
            ধ্রুবক দশমিক * src_row = (ধ্রুবক দশমিক *) src0_তথ্য + row_সূচক * ne00 + batch_offset;
            দশমিক *       dst_row = dst_তথ্য + dst_offset * ne00;
            copy_f32_row(dst_row, src_row, ne00);
        } অন্যথা যদি (src0->type == GGML_TYPE_F16) {
            // F16 source: scalar conversion
            ধ্রুবক uint16_t * src_row = (ধ্রুবক uint16_t *) src0_তথ্য + row_সূচক * ne00 + batch_offset;
            দশমিক *          dst_row = dst_তথ্য + dst_offset * ne00;
            copy_f16_row(dst_row, src_row, ne00);
        } অন্যথা যদি (src0->type == GGML_TYPE_Q8_0) {
            // Q8_0 source: dequantize while copying
            ধ্রুবক int64_t      blocks_per_row   = (ne00 + QK8_0 - 1) / QK8_0;
            ধ্রুবক int64_t      src_block_offset = (row_সূচক * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            ধ্রুবক block_q8_0 * src_blocks       = (ধ্রুবক block_q8_0 *) src0_তথ্য + src_block_offset;
            দশমিক *            dst_row          = dst_তথ্য + dst_offset * ne00;
            copy_q8_0_row(dst_row, src_blocks, ne00);
        } অন্যথা যদি (src0->type == GGML_TYPE_Q4_0) {
            // Q4_0 source: dequantize while copying
            ধ্রুবক int64_t      blocks_per_row   = (ne00 + QK4_0 - 1) / QK4_0;
            ধ্রুবক int64_t      src_block_offset = (row_সূচক * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            ধ্রুবক block_q4_0 * src_blocks       = (ধ্রুবক block_q4_0 *) src0_তথ্য + src_block_offset;
            দশমিক *            dst_row          = dst_তথ্য + dst_offset * ne00;
            copy_q4_0_row(dst_row, src_blocks, ne00);
        } অন্যথা যদি (src0->type == GGML_TYPE_Q4_K) {
            // Q4_K source: dequantize while copying
            ধ্রুবক int64_t      blocks_per_row   = (ne00 + QK_K - 1) / QK_K;
            ধ্রুবক int64_t      src_block_offset = (row_সূচক * blocks_per_row) + (batch_offset / ne00) * blocks_per_row;
            ধ্রুবক block_q4_K * src_blocks       = (ধ্রুবক block_q4_K *) src0_তথ্য + src_block_offset;
            দশমিক *            dst_row          = dst_তথ্য + dst_offset * ne00;
            copy_q4_K_row(dst_row, src_blocks, ne00);
        }
    }

    ফেরত 0;
}
