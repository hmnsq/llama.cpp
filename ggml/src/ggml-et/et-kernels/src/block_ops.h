//******************************************************************************
// ET Vectorized Block Operations Library
// Provides optimized block-level operations using ET hardware vector instructions
//******************************************************************************

#ifndef BLOCK_OPS_H
#    define BLOCK_OPS_H

#    include "math_fp.h"
#    include "quants.h"

#    include <stdint.h>

//******************************************************************************
// Block Dot Product Operations
//******************************************************************************
ओळीत शून्य __attribute__((always_inline)) excl_mode(uint64_t val) {
    __asm__ __volatile__("csrw 0x7d3, %[csr_enc]\n" : : [csr_enc] "r"(val) : "x31");
}

स्थिर ओळीत दशांश compute_block_dot_product_q4_0(स्थिरांक block_q4_0 * a_block, स्थिरांक दशांश * b_col_start) {
    // Set mask register to enable all 8 vector elements
    अचिन्हित दीर्घ temp_mask;
    __asm__ अस्थिर("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Use f10 as accumulator, init to 0
    __asm__ अस्थिर("fbci.ps f10, 0" ::: "f10");

    स्थिर स्थिरांक int32_t gather_pattern[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ अस्थिर("flw.ps f31, %[gather]\n" : : [gather] "m"(*(स्थिरांक int32_t (*)[8]) gather_pattern) : "f31");

    // Process 32 elements in 2 chunks of 16 elements (8 bytes) each
    साठी (पूर्णांक chunk = 0; chunk < 2; chunk++) {
        पूर्णांक offset_a      = chunk * 8;
        पूर्णांक offset_b_खालचा  = chunk * 8;       // Activations for lower nibbles
        पूर्णांक offset_b_वरचा = chunk * 8 + 16;  // Activations for upper nibbles (16 elements later)

        __asm__ अस्थिर(
            "fgb.ps f11, f31(%[a_ptr])\n"  // Gather 8 bytes (16 packed q4_0 weights)

            // 1. Extract & Multiply Lower Nibbles
            "fandi.pi f12, f11, 15\n"             // Mask lower 4 bits (x & 0xF)
            "faddi.pi f12, f12, -8\n"             // GGML offset to signed: (x & 0xF) - 8
            "fcvt.ps.pw f12, f12, rne\n"          // Convert INT32 to FP32
            "flw.ps f13, 0(%[b_low])\n"           // Load 8 B values (floats)
            "fmadd.ps f10, f12, f13, f10, rne\n"  // acc += A_low * B_low

            // 2. Extract & Multiply Upper Nibbles
            "fsrli.pi f14, f11, 4\n"              // Shift upper 4 bits down
            "fandi.pi f14, f14, 15\n"             // Mask new lower 4 bits
            "faddi.pi f14, f14, -8\n"             // GGML offset to signed
            "fcvt.ps.pw f14, f14, rne\n"          // Convert INT32 to FP32
            "flw.ps f15, 0(%[b_high])\n"          // Load next 8 B values (floats)
            "fmadd.ps f10, f14, f15, f10, rne\n"  // acc += A_high * B_high
            :
            : [a_ptr] "r"(&a_block->qs[offset_a]), [b_खालचा] "r"(&b_col_start[offset_b_खालचा]),
              [b_वरचा] "r"(&b_col_start[offset_b_वरचा])
            // Note: f10 is explicitly NOT listed in the clobbers here to ensure the compiler
            // preserves the running sum across C loop iterations safely.
            : "f11", "f12", "f13", "f14", "f15");
    }

    // Horizontal sum: reduce f10 into a single scalar
    दशांश final_sum;
    __asm__ __volatile__(
        // Pairwise sum within each 128-bit half
        "fswizz.ps f1, f10, 0xB1 \n\t"  // Swaps: e0<->e1 and e2<->e3
        "fadd.ps   f2, f10, f1, rne \n\t"
        // Complete the sum for each 128-bit half
        "fswizz.ps f3, f2, 0x4E \n\t"  // Swaps: e0,e1 <-> e2,e3
        "fadd.ps   f4, f2, f3, rne \n\t"
        // Sum across the two 128b halfs
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(final_sum)::"t0", "f1", "f2", "f3", "f4", "f5", "f10");

    // Restore original mask
    __asm__ अस्थिर("mova.m.x %0" ::"r"(temp_mask));

    स्थिरांक दशांश scale = fp16_to_fp32(a_block->d);
    परत final_sum * scale;
}

// Compute dot product between dequantized q8_0 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 32 int8 values (QK8_0)
स्थिर ओळीत दशांश compute_block_dot_product_q8_0(स्थिरांक block_q8_0 * a_block, स्थिरांक दशांश * b_col_start) {
    // Set mask register to enable all 8 vector elements
    अचिन्हित दीर्घ temp_mask;
    __asm__ अस्थिर("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
    __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements
    __asm__ अस्थिर("fbci.pi f10, 0" ::: "f10");       // Use f10 as accumulator, init to 0

    स्थिर स्थिरांक int32_t gather_pattern[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    __asm__ अस्थिर("flw.ps f31, %[gather]\n" : : [gather] "m"(*(स्थिरांक int32_t (*)[8]) gather_pattern) : "f31");

    // Process 32 elements in 4 chunks of 8 elements each
    साठी (पूर्णांक chunk = 0; chunk < 4; chunk++) {
        पूर्णांक offset = chunk << 3;  // chunk * 8

        __asm__ अस्थिर(
            "flw.ps f12, %[b_vec]\n"         // Load 8 B values (floats)
            "fgb.ps f11, f31(%[a_ptr])\n"    // Gather 8 int8 bytes from A using pattern
            "fcvt.ps.pw f11, f11\n"          // Convert int8 vector to float vector
            "fmadd.ps f10, f11, f12, f10\n"  // acc += a_vec * b_vec (8-wide)
            :
            : [a_ptr] "r"(&a_block->qs[offset]), [b_vec] "m"(*(स्थिरांक दशांश (*)[8]) & b_col_start[offset]),
              [scale] "m"(a_block->d)
            : "f10", "f11", "f12");
    }

    // Horizontal sum: reduce f10 into a single scalar
    दशांश final_sum;
    __asm__ __volatile__(
        // Pairwise sum within each 128-bit half
        "fswizz.ps f1, f10, 0xB1 \n\t"  // Swaps: e0<->e1 and e2<->e3
        "fadd.ps   f2, f10, f1, rne \n\t"
        // Complete the sum for each 128-bit half
        "fswizz.ps f3, f2, 0x4E \n\t"  // Swaps: e0,e1 <-> e2,e3
        "fadd.ps   f4, f2, f3, rne \n\t"
        // Sum across the two 128b halfs
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(final_sum)::"t0", "f10", "f2", "f3", "f4", "f5");

    // Restore original mask
    __asm__ अस्थिर("mova.m.x %0" ::"r"(temp_mask));

    स्थिरांक दशांश scale = fp16_to_fp32(a_block->d);
    परत final_sum * scale;
}

//******************************************************************************
// Split-phase Q8_0 dot product API
//
//   q8_dot_begin(st)      — save mask, set mask 0xFF
//   q8_dot_reset()        — zero vector accumulator f20
//   q8_dot_tile(q, b, n)  — accumulate n Q8_0 blocks into f20
//   q8_dot_reduce()       — horizontal sum of f20, return scalar float
//   q8_dot_teardown(st)   — restore original mask
//
// Register contract:
//   f20       — row accumulator (persistent across tiles, reset per row)
//   f31       — gather pattern (reloaded per q8_dot_tile call)
//   f10-f12   — scratch within tile
//   f15       — scale broadcast within tile
//   f1-f5, t0 — scratch within reduce
//******************************************************************************

स्थिर ओळीत शून्य __attribute__((always_inline)) q8_dot_reset(शून्य) {
    __asm__ अस्थिर("fbci.pi f20, 0" ::: "f20");
}

// Accumulate n_blocks Q8_0 blocks into f20.
// Uses fg32b.ps (fast gather with scalar pattern) for aligned chunks,
// falls back to fgb.ps for chunks crossing a 32-byte boundary.
स्थिर ओळीत शून्य __attribute__((always_inline)) q8_dot_tile(स्थिरांक block_q8_0 * q_row,
                                                              स्थिरांक दशांश *      b_col,
                                                              int64_t            n_blocks) {
    स्थिरांक int32_t  gather_pattern[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    स्थिरांक uint64_t gather_0_to_7     = 0x398a418820ULL;

    __asm__ अस्थिर("flw.ps f31, %[g]\n" : : [g] "m"(*(स्थिरांक int32_t (*)[8]) gather_pattern) : "f31");

    साठी (int64_t kb = 0; kb < n_blocks; kb++) {
        स्थिरांक block_q8_0 * blk         = q_row + kb;
        स्थिरांक दशांश *      b_ptr       = b_col + (kb << 5);
        स्थिरांक uintptr_t    qs_addr     = (uintptr_t) blk->qs;
        स्थिरांक uintptr_t    qs_aligned  = qs_addr & ~(uintptr_t) 31;
        स्थिरांक uintptr_t    qs_खालचा      = qs_addr & 31;
        स्थिरांक पूर्णांक          fast_chunks = (पूर्णांक) ((32 - qs_खालचा) >> 3);

        जर (fast_chunks >= 3) {
            __asm__ अस्थिर(
                "fbci.pi     f10, 0\n"
                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f11, %[gi](%[ap0])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv1]\n"
                "fg32b.ps    f11, %[gi](%[ap1])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv2]\n"
                "fg32b.ps    f11, %[gi](%[ap2])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv3]\n"
                "fgb.ps      f11, f31(%[ap3])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                :
                : [gi] "r"(gather_0_to_7), [ap0] "r"(qs_addr), [ap1] "r"(qs_aligned | ((qs_addr + 8) & 31)),
                  [ap2] "r"(qs_aligned | ((qs_addr + 16) & 31)), [ap3] "r"(&blk->qs[24]),
                  [bv0] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[0]), [bv1] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[8]),
                  [bv2] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[16]), [bv3] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12");
        } अन्यथा जर (fast_chunks == 2) {
            __asm__ अस्थिर(
                "fbci.pi     f10, 0\n"
                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f11, %[gi](%[ap0])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv1]\n"
                "fg32b.ps    f11, %[gi](%[ap1])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv2]\n"
                "fgb.ps      f11, f31(%[ap2])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv3]\n"
                "fgb.ps      f11, f31(%[ap3])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                :
                : [gi] "r"(gather_0_to_7), [ap0] "r"(qs_addr), [ap1] "r"(qs_aligned | ((qs_addr + 8) & 31)),
                  [ap2] "r"(&blk->qs[16]), [ap3] "r"(&blk->qs[24]), [bv0] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[8]), [bv2] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12");
        } अन्यथा जर (fast_chunks == 1) {
            __asm__ अस्थिर(
                "fbci.pi     f10, 0\n"
                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f11, %[gi](%[ap0])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv1]\n"
                "fgb.ps      f11, f31(%[ap1])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv2]\n"
                "fgb.ps      f11, f31(%[ap2])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv3]\n"
                "fgb.ps      f11, f31(%[ap3])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                :
                : [gi] "r"(gather_0_to_7), [ap0] "r"(qs_addr), [ap1] "r"(&blk->qs[8]), [ap2] "r"(&blk->qs[16]),
                  [ap3] "r"(&blk->qs[24]), [bv0] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[8]), [bv2] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12");
        } अन्यथा {
            __asm__ अस्थिर(
                "fbci.pi     f10, 0\n"
                "flw.ps      f12, %[bv0]\n"
                "fgb.ps      f11, f31(%[ap0])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv1]\n"
                "fgb.ps      f11, f31(%[ap1])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv2]\n"
                "fgb.ps      f11, f31(%[ap2])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                "flw.ps      f12, %[bv3]\n"
                "fgb.ps      f11, f31(%[ap3])\n"
                "fcvt.ps.pw  f11, f11\n"
                "fmadd.ps    f10, f11, f12, f10\n"
                :
                : [ap0] "r"(&blk->qs[0]), [ap1] "r"(&blk->qs[8]), [ap2] "r"(&blk->qs[16]), [ap3] "r"(&blk->qs[24]),
                  [bv0] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[0]), [bv1] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[8]),
                  [bv2] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[16]), [bv3] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12");
        }

        // f20 += f10 * broadcast(scale) — hardware fp16→fp32 via FCVT.PS.F16
        uint32_t scale_raw = (uint32_t) blk->d;
        __asm__ अस्थिर(
            "fbcx.ps f15, %[sb]\n"
            "fcvt.ps.f16 f15, f15\n"
            "fmadd.ps f20, f10, f15, f20\n"
            :
            : [sb] "r"(scale_raw)
            : "f15", "f20");
    }
}

// Horizontal sum of 8-element vector accumulator f20.
स्थिर ओळीत दशांश __attribute__((always_inline)) q8_dot_reduce(शून्य) {
    दशांश result;
    __asm__ __volatile__(
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result)::"t0", "f1", "f2", "f3", "f4", "f5");
    परत result;
}

// Full-row dot product (convenience wrapper)
स्थिर ओळीत दशांश compute_row_dot_q8_0(स्थिरांक block_q8_0 * q_row, स्थिरांक दशांश * b_col, int64_t K_blocks) {
    अचिन्हित दीर्घ saved_mask;
    __asm__ अस्थिर("mova.x.m %0" : "=r"(saved_mask));
    __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");
    q8_dot_reset();
    q8_dot_tile(q_row, b_col, K_blocks);
    दशांश result = q8_dot_reduce();
    __asm__ अस्थिर("mova.m.x %0" ::"r"(saved_mask));
    परत result;
}

//******************************************************************************
// Hoisted Q8_0 dot API
//
// q8_dot_begin/end save/restore the vector mask once around a long sequence of
// dot products, so the per-row mask shuffles are hoisted out of the inner
// loops. q8_dot_compute does a full-row dot (no mask handling). The _x2
// variant computes two rows together while reusing each loaded B chunk —
// only safe when both row pointers share the same 32-byte alignment phase
// (i.e. the Q8 row stride is a multiple of 32).
//******************************************************************************

प्रकारव्याख्या संरचना {
    अचिन्हित दीर्घ saved_mask;
} q8_dot_state;

स्थिर ओळीत शून्य q8_dot_begin(q8_dot_state * state) {
    __asm__ अस्थिर("mova.x.m %0" : "=r"(state->saved_mask));
    __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");
}

स्थिर ओळीत शून्य q8_dot_end(स्थिरांक q8_dot_state * state) {
    __asm__ अस्थिर("mova.m.x %0" ::"r"(state->saved_mask));
}

// Equivalent to q8_dot_reset+tile+reduce, without touching the mask register.
// Caller is responsible for q8_dot_begin/end around the surrounding loop.
स्थिर ओळीत दशांश q8_dot_compute(स्थिरांक block_q8_0 * q_row, स्थिरांक दशांश * b_col, int64_t K_blocks) {
    q8_dot_reset();
    q8_dot_tile(q_row, b_col, K_blocks);
    परत q8_dot_reduce();
}

// Compute two row dots together while reusing the same loaded B chunks.
//
// Safe when every row starts at the same 32-byte offset, i.e. the Q8 row stride
// is a multiple of 32. In that case the gather/alignment pattern is the same
// for both rows at a given `kb`, so one set of B vector loads feeds both row
// accumulators.
स्थिर ओळीत शून्य q8_dot_compute_x2_aligned(स्थिरांक block_q8_0 * q_row0,
                                             स्थिरांक block_q8_0 * q_row1,
                                             स्थिरांक दशांश *      b_col,
                                             int64_t            K_blocks,
                                             दशांश *            out0,
                                             दशांश *            out1) {
    स्थिरांक int32_t  gather_pattern[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    स्थिरांक uint64_t gather_0_to_7     = 0x398a418820ULL;
    __asm__ अस्थिर("flw.ps f31, %[g]\n" : : [g] "m"(*(स्थिरांक int32_t (*)[8]) gather_pattern) : "f31");
    __asm__ अस्थिर(
        "fbci.pi f20, 0\n"
        "fbci.pi f21, 0\n" ::
            : "f20", "f21");

    साठी (int64_t kb = 0; kb < K_blocks; kb++) {
        स्थिरांक block_q8_0 * blk0  = q_row0 + kb;
        स्थिरांक block_q8_0 * blk1  = q_row1 + kb;
        स्थिरांक दशांश *      b_ptr = b_col + (kb << 5);

        स्थिरांक uintptr_t qs_addr0    = (uintptr_t) blk0->qs;
        स्थिरांक uintptr_t qs_addr1    = (uintptr_t) blk1->qs;
        स्थिरांक uintptr_t qs_aligned0 = qs_addr0 & ~(uintptr_t) 31;
        स्थिरांक uintptr_t qs_aligned1 = qs_addr1 & ~(uintptr_t) 31;
        स्थिरांक पूर्णांक       fast_chunks = (पूर्णांक) ((32 - (qs_addr0 & 31)) >> 3);

        जर (fast_chunks >= 3) {
            __asm__ अस्थिर(
                "fbci.pi     f10, 0\n"
                "fbci.pi     f11, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f16, %[gi](%[r0ap0])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f12, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap0])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f12, f11\n"

                "flw.ps      f13, %[bv1]\n"
                "fg32b.ps    f16, %[gi](%[r0ap1])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f13, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap1])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f13, f11\n"

                "flw.ps      f14, %[bv2]\n"
                "fg32b.ps    f16, %[gi](%[r0ap2])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f14, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap2])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f14, f11\n"

                "flw.ps      f15, %[bv3]\n"
                "fgb.ps      f16, f31(%[r0ap3])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f15, f10\n"
                "fgb.ps      f17, f31(%[r1ap3])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f15, f11\n"
                :
                : [gi] "r"(gather_0_to_7), [r0ap0] "r"(qs_addr0), [r0ap1] "r"(qs_aligned0 | ((qs_addr0 + 8) & 31)),
                  [r0ap2] "r"(qs_aligned0 | ((qs_addr0 + 16) & 31)), [r0ap3] "r"(&blk0->qs[24]), [r1ap0] "r"(qs_addr1),
                  [r1ap1] "r"(qs_aligned1 | ((qs_addr1 + 8) & 31)), [r1ap2] "r"(qs_aligned1 | ((qs_addr1 + 16) & 31)),
                  [r1ap3] "r"(&blk1->qs[24]), [bv0] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[8]), [bv2] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17");
        } अन्यथा जर (fast_chunks == 2) {
            __asm__ अस्थिर(
                "fbci.pi     f10, 0\n"
                "fbci.pi     f11, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f16, %[gi](%[r0ap0])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f12, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap0])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f12, f11\n"

                "flw.ps      f13, %[bv1]\n"
                "fg32b.ps    f16, %[gi](%[r0ap1])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f13, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap1])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f13, f11\n"

                "flw.ps      f14, %[bv2]\n"
                "fgb.ps      f16, f31(%[r0ap2])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f14, f10\n"
                "fgb.ps      f17, f31(%[r1ap2])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f14, f11\n"

                "flw.ps      f15, %[bv3]\n"
                "fgb.ps      f16, f31(%[r0ap3])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f15, f10\n"
                "fgb.ps      f17, f31(%[r1ap3])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f15, f11\n"
                :
                : [gi] "r"(gather_0_to_7), [r0ap0] "r"(qs_addr0), [r0ap1] "r"(qs_aligned0 | ((qs_addr0 + 8) & 31)),
                  [r0ap2] "r"(&blk0->qs[16]), [r0ap3] "r"(&blk0->qs[24]), [r1ap0] "r"(qs_addr1),
                  [r1ap1] "r"(qs_aligned1 | ((qs_addr1 + 8) & 31)), [r1ap2] "r"(&blk1->qs[16]),
                  [r1ap3] "r"(&blk1->qs[24]), [bv0] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[8]), [bv2] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17");
        } अन्यथा जर (fast_chunks == 1) {
            __asm__ अस्थिर(
                "fbci.pi     f10, 0\n"
                "fbci.pi     f11, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fg32b.ps    f16, %[gi](%[r0ap0])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f12, f10\n"
                "fg32b.ps    f17, %[gi](%[r1ap0])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f12, f11\n"

                "flw.ps      f13, %[bv1]\n"
                "fgb.ps      f16, f31(%[r0ap1])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f13, f10\n"
                "fgb.ps      f17, f31(%[r1ap1])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f13, f11\n"

                "flw.ps      f14, %[bv2]\n"
                "fgb.ps      f16, f31(%[r0ap2])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f14, f10\n"
                "fgb.ps      f17, f31(%[r1ap2])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f14, f11\n"

                "flw.ps      f15, %[bv3]\n"
                "fgb.ps      f16, f31(%[r0ap3])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f15, f10\n"
                "fgb.ps      f17, f31(%[r1ap3])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f15, f11\n"
                :
                : [gi] "r"(gather_0_to_7), [r0ap0] "r"(qs_addr0), [r0ap1] "r"(&blk0->qs[8]), [r0ap2] "r"(&blk0->qs[16]),
                  [r0ap3] "r"(&blk0->qs[24]), [r1ap0] "r"(qs_addr1), [r1ap1] "r"(&blk1->qs[8]),
                  [r1ap2] "r"(&blk1->qs[16]), [r1ap3] "r"(&blk1->qs[24]), [bv0] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[8]), [bv2] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17");
        } अन्यथा {
            __asm__ अस्थिर(
                "fbci.pi     f10, 0\n"
                "fbci.pi     f11, 0\n"

                "flw.ps      f12, %[bv0]\n"
                "fgb.ps      f16, f31(%[r0ap0])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f12, f10\n"
                "fgb.ps      f17, f31(%[r1ap0])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f12, f11\n"

                "flw.ps      f13, %[bv1]\n"
                "fgb.ps      f16, f31(%[r0ap1])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f13, f10\n"
                "fgb.ps      f17, f31(%[r1ap1])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f13, f11\n"

                "flw.ps      f14, %[bv2]\n"
                "fgb.ps      f16, f31(%[r0ap2])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f14, f10\n"
                "fgb.ps      f17, f31(%[r1ap2])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f14, f11\n"

                "flw.ps      f15, %[bv3]\n"
                "fgb.ps      f16, f31(%[r0ap3])\n"
                "fcvt.ps.pw  f16, f16\n"
                "fmadd.ps    f10, f16, f15, f10\n"
                "fgb.ps      f17, f31(%[r1ap3])\n"
                "fcvt.ps.pw  f17, f17\n"
                "fmadd.ps    f11, f17, f15, f11\n"
                :
                : [r0ap0] "r"(&blk0->qs[0]), [r0ap1] "r"(&blk0->qs[8]), [r0ap2] "r"(&blk0->qs[16]),
                  [r0ap3] "r"(&blk0->qs[24]), [r1ap0] "r"(&blk1->qs[0]), [r1ap1] "r"(&blk1->qs[8]),
                  [r1ap2] "r"(&blk1->qs[16]), [r1ap3] "r"(&blk1->qs[24]), [bv0] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[8]), [bv2] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17");
        }

        स्थिरांक uint32_t scale_raw0 = (uint32_t) blk0->d;
        स्थिरांक uint32_t scale_raw1 = (uint32_t) blk1->d;
        __asm__ अस्थिर(
            "fbcx.ps     f24, %[s0]\n"
            "fcvt.ps.f16 f24, f24\n"
            "fmadd.ps    f20, f10, f24, f20\n"
            "fbcx.ps     f25, %[s1]\n"
            "fcvt.ps.f16 f25, f25\n"
            "fmadd.ps    f21, f11, f25, f21\n"
            :
            : [s0] "r"(scale_raw0), [s1] "r"(scale_raw1)
            : "f20", "f21", "f24", "f25");
    }

    दशांश result0;
    दशांश result1;
    __asm__ __volatile__(
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result0)::"t0", "f1", "f2", "f3", "f4", "f5");
    __asm__ __volatile__(
        "fswizz.ps f1, f21, 0xB1 \n\t"
        "fadd.ps   f2, f21, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result1)::"t0", "f1", "f2", "f3", "f4", "f5");

    *out0 = result0;
    *out1 = result1;
}

// Compute dot product between f16 block and f32 column vector (NAIVE VERSION)
// Scalar implementation for debugging - no vectorization
// Block size: 32 f16 values (64 bytes = 1 cache line)
स्थिर ओळीत दशांश compute_block_dot_product_f16_naive(स्थिरांक uint16_t * a_block, स्थिरांक दशांश * b_col_start) {
    दशांश                acc_vec[8] __attribute__((aligned(32))) = { 0.0f };
    // Byte offsets for 16-bit (half-word) elements
    स्थिर स्थिरांक int32_t gather_pattern[8]                       = { 0, 2, 4, 6, 8, 10, 12, 14 };
    अचिन्हित दीर्घ        temp_mask;

    __asm__ अस्थिर("mova.x.m %0" : "=r"(temp_mask));
    __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");

    // Load the pattern once into f31 for the duration of all 4 chunks
    __asm__ अस्थिर("flw.ps f31, %[gather]\n" : : [gather] "m"(*(स्थिरांक int32_t (*)[8]) gather_pattern) : "f31");

    साठी (पूर्णांक chunk = 0; chunk < 4; chunk++) {
        // Correct pointers:
        // a_block elements are 2 bytes, b_col elements are 4 bytes
        स्थिरांक uint16_t * a_ptr = &a_block[chunk << 3];      // chunk * 8
        स्थिरांक दशांश *    b_ptr = &b_col_start[chunk << 3];  // chunk * 8

        __asm__ अस्थिर(
            "flw.ps f10, %[acc]\n"
            "fgh.ps f11, f31(%[a_p])\n"  // Uses {0,2,4,6,8,10,12,14} byte offsets
            "fcvt.ps.f16 f11, f11\n"
            "flw.ps f12, (%[b_p])\n"     // Standard vector load (32-bit floats)
            "fmadd.ps f10, f11, f12, f10\n"
            "fsw.ps f10, %[result]\n"

            : [result] "=m"(*(दशांश (*)[8]) acc_vec)
            : [acc] "m"(*(स्थिरांक दशांश (*)[8]) acc_vec), [a_p] "r"(a_ptr), [b_p] "r"(b_ptr)
            : "f10", "f11", "f12");
    }

    __asm__ अस्थिर("mova.m.x %0" ::"r"(temp_mask));

    परत acc_vec[0] + acc_vec[1] + acc_vec[2] + acc_vec[3] + acc_vec[4] + acc_vec[5] + acc_vec[6] + acc_vec[7];
}

// Compute dot product between f16 block and f32 column vector
// SCALAR implementation for partial blocks
// Block size: up to 32 f16 values (can handle partial blocks for misaligned K)
स्थिर ओळीत दशांश compute_block_dot_product_f16_partial(स्थिरांक uint16_t * a_block,
                                                          स्थिरांक दशांश *    b_col_start,
                                                          पूर्णांक              elements) {
    // This matches compute_block_dot_product_f16_naive behavior
    दशांश sum = 0.0f;

    साठी (पूर्णांक i = 0; i < elements; i++) {
        दशांश a_val = fp16_to_fp32(a_block[i]);
        दशांश b_val = b_col_start[i];
        sum += a_val * b_val;
    }

    परत sum;
}

// Compute dot product between f16 block and f16 column vector
// Scalar implementation for generic non-matrix-engine fallback paths.
स्थिर ओळीत दशांश compute_block_dot_product_f16_f16_partial(स्थिरांक uint16_t * a_block,
                                                              स्थिरांक uint16_t * b_col_start,
                                                              पूर्णांक              elements) {
    दशांश sum = 0.0f;

    साठी (पूर्णांक i = 0; i < elements; i++) {
        sum += fp16_to_fp32(a_block[i]) * fp16_to_fp32(b_col_start[i]);
    }

    परत sum;
}

// Compute dot product between f16 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 32 f16 values (64 bytes = 1 cache line)
स्थिर ओळीत दशांश compute_block_dot_product_f16(स्थिरांक uint16_t * a_block, स्थिरांक दशांश * b_col_start) {
    परत compute_block_dot_product_f16_partial(a_block, b_col_start, QK_F16);
}

// Compute dot product between f32 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: up to 16 f32 values (can handle partial blocks for misaligned K)
स्थिर ओळीत दशांश compute_block_dot_product_f32_partial(स्थिरांक दशांश * a_block,
                                                          स्थिरांक दशांश * b_col_start,
                                                          पूर्णांक           elements) {
    दशांश acc_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };  // Accumulator vector

    // Calculate how many full 8-element chunks we can process
    पूर्णांक vec_end = (elements / 8) * 8;

    जर (vec_end > 0) {
        // Set mask register to enable all 8 vector elements
        अचिन्हित दीर्घ temp_mask;
        __asm__ अस्थिर("mova.x.m %0" : "=r"(temp_mask));  // Save current mask
        __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

        // Process full 8-element chunks
        साठी (पूर्णांक i = 0; i < vec_end; i += 8) {
            // Vectorized f32 multiply-accumulate
            __asm__ अस्थिर(
                "flw.ps f10, %[acc]\n"           // Load current accumulator (8 floats)
                "flw.ps f11, %[a_vec]\n"         // Load 8 A values (f32)
                "flw.ps f12, %[b_vec]\n"         // Load 8 B values (f32)
                "fmadd.ps f10, f11, f12, f10\n"  // acc += a_vec * b_vec (8-wide)
                "fsw.ps f10, %[result]\n"        // Store back to accumulator

                : [result] "=m"(*(दशांश (*)[8]) acc_vec)
                : [acc] "m"(*(स्थिरांक दशांश (*)[8]) acc_vec), [a_vec] "m"(*(स्थिरांक दशांश (*)[8])(a_block + i)),
                  [b_vec] "m"(*(स्थिरांक दशांश (*)[8])(b_col_start + i))
                : "f10", "f11", "f12");
        }

        // Restore original mask
        __asm__ अस्थिर("mova.m.x %0" ::"r"(temp_mask));
    }

    // Horizontal sum: reduce 8 accumulator elements to single scalar
    दशांश final_sum = 0.0f;
    साठी (पूर्णांक i = 0; i < 8; i++) {
        final_sum += acc_vec[i];
    }

    // Handle remaining elements (< 8) with scalar operations
    साठी (पूर्णांक i = vec_end; i < elements; i++) {
        final_sum += a_block[i] * b_col_start[i];
    }

    परत final_sum;
}

// Compute dot product between f32 block and f16 column vector
// Scalar implementation for generic non-matrix-engine fallback paths.
स्थिर ओळीत दशांश compute_block_dot_product_f32_f16_partial(स्थिरांक दशांश *    a_block,
                                                              स्थिरांक uint16_t * b_col_start,
                                                              पूर्णांक              elements) {
    दशांश sum = 0.0f;

    साठी (पूर्णांक i = 0; i < elements; i++) {
        sum += a_block[i] * fp16_to_fp32(b_col_start[i]);
    }

    परत sum;
}

// Compute dot product between f32 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 16 f32 values (64 bytes = 1 cache line)
स्थिर ओळीत दशांश compute_block_dot_product_f32(स्थिरांक दशांश * a_block, स्थिरांक दशांश * b_col_start) {
    परत compute_block_dot_product_f32_partial(a_block, b_col_start, QK_F32);

    // float acc_vec[8];
    // unsigned long old_mask;
    // __asm__ volatile(
    //     // Save current mask
    //     "mova.x.m %[old_mask]\n"
    //     // Enable all 8 lanes
    //     "mov.m.x m0, x0, 0xFF\n"

    //     "flw.ps  f11, %[a]\n"
    //     "flw.ps  f12, %[b]\n"
    //     "fmadd.ps f10, f11, f12, f10\n"
    //     "fsw.ps  f10, %[out]\n"
    //     "mova.m.x %[old_mask]\n"

    //     : [out] "=m" (*(float(*)[8])acc_vec),
    //       [old_mask] "=r"(old_mask)
    //     : [a] "m" (*(const float(*)[8])a_block),
    //       [b] "m" (*(const float(*)[8])b_col_start)
    //     : "f10", "f11", "f12"
    // );

    // // Horizontal reduction
    // return acc_vec[0] + acc_vec[1] + acc_vec[2] + acc_vec[3] +
    //        acc_vec[4] + acc_vec[5] + acc_vec[6] + acc_vec[7];
}

#endif  // BLOCK_OPS_H

स्थिर ओळीत शून्य __attribute__((always_inline)) q4_dot_reset(शून्य) {
    __asm__ अस्थिर("fbci.pi f20, 0" ::: "f20");
}

स्थिर ओळीत शून्य __attribute__((always_inline)) q4_dot_tile(स्थिरांक block_q4_0 * q_row,
                                                              स्थिरांक दशांश *      b_col,
                                                              int64_t            n_blocks) {
    स्थिरांक int32_t gather_pattern[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ अस्थिर("flw.ps f31, %[g]\n" : : [g] "m"(*(स्थिरांक int32_t (*)[8]) gather_pattern) : "f31");

    साठी (int64_t kb = 0; kb < n_blocks; kb++) {
        स्थिरांक block_q4_0 * blk   = q_row + kb;
        स्थिरांक दशांश *      b_ptr = b_col + (kb << 5);

        __asm__ अस्थिर(
            "fbci.pi     f10, 0\n"

            "fgb.ps      f11, f31(%[a_ptr0])\n"
            "fandi.pi    f12, f11, 15\n"
            "faddi.pi    f12, f12, -8\n"
            "fcvt.ps.pw  f12, f12, rne\n"
            "flw.ps      f13, %[b_low0]\n"
            "fmadd.ps    f10, f12, f13, f10, rne\n"

            "fsrli.pi    f14, f11, 4\n"
            "fandi.pi    f14, f14, 15\n"
            "faddi.pi    f14, f14, -8\n"
            "fcvt.ps.pw  f14, f14, rne\n"
            "flw.ps      f15, %[b_high0]\n"
            "fmadd.ps    f10, f14, f15, f10, rne\n"

            "fgb.ps      f11, f31(%[a_ptr1])\n"
            "fandi.pi    f12, f11, 15\n"
            "faddi.pi    f12, f12, -8\n"
            "fcvt.ps.pw  f12, f12, rne\n"
            "flw.ps      f13, %[b_low1]\n"
            "fmadd.ps    f10, f12, f13, f10, rne\n"

            "fsrli.pi    f14, f11, 4\n"
            "fandi.pi    f14, f14, 15\n"
            "faddi.pi    f14, f14, -8\n"
            "fcvt.ps.pw  f14, f14, rne\n"
            "flw.ps      f15, %[b_high1]\n"
            "fmadd.ps    f10, f14, f15, f10, rne\n"
            :
            : [a_ptr0] "r"(&blk->qs[0]), [b_low0] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[0]),
              [b_high0] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[16]), [a_ptr1] "r"(&blk->qs[8]),
              [b_low1] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[8]), [b_high1] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[24])
            : "f10", "f11", "f12", "f13", "f14", "f15");

        uint32_t scale_raw = (uint32_t) blk->d;
        __asm__ अस्थिर(
            "fbcx.ps f15, %[sb]\n"
            "fcvt.ps.f16 f15, f15\n"
            "fmadd.ps f20, f10, f15, f20\n"
            :
            : [sb] "r"(scale_raw)
            : "f15", "f20");
    }
}

स्थिर ओळीत दशांश __attribute__((always_inline)) q4_dot_reduce(शून्य) {
    दशांश result;
    __asm__ __volatile__(
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result)::"t0", "f1", "f2", "f3", "f4", "f5");
    परत result;
}

स्थिर ओळीत दशांश compute_row_dot_q4_0(स्थिरांक block_q4_0 * q_row, स्थिरांक दशांश * b_col, int64_t K_blocks) {
    अचिन्हित दीर्घ saved_mask;
    __asm__ अस्थिर("mova.x.m %0" : "=r"(saved_mask));
    __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");
    q4_dot_reset();
    q4_dot_tile(q_row, b_col, K_blocks);
    दशांश result = q4_dot_reduce();
    __asm__ अस्थिर("mova.m.x %0" ::"r"(saved_mask));
    परत result;
}

प्रकारव्याख्या संरचना {
    अचिन्हित दीर्घ saved_mask;
} q4_dot_state;

स्थिर ओळीत शून्य q4_dot_begin(q4_dot_state * state) {
    __asm__ अस्थिर("mova.x.m %0" : "=r"(state->saved_mask));
    __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");
}

स्थिर ओळीत शून्य q4_dot_end(स्थिरांक q4_dot_state * state) {
    __asm__ अस्थिर("mova.m.x %0" ::"r"(state->saved_mask));
}

स्थिर ओळीत दशांश q4_dot_compute(स्थिरांक block_q4_0 * q_row, स्थिरांक दशांश * b_col, int64_t K_blocks) {
    q4_dot_reset();
    q4_dot_tile(q_row, b_col, K_blocks);
    परत q4_dot_reduce();
}

स्थिर ओळीत शून्य q4_dot_compute_x2_aligned(स्थिरांक block_q4_0 * q_row0,
                                             स्थिरांक block_q4_0 * q_row1,
                                             स्थिरांक दशांश *      b_col,
                                             int64_t            K_blocks,
                                             दशांश *            out0,
                                             दशांश *            out1) {
    स्थिरांक int32_t gather_pattern[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ अस्थिर("flw.ps f31, %[g]\n" : : [g] "m"(*(स्थिरांक int32_t (*)[8]) gather_pattern) : "f31");
    __asm__ अस्थिर(
        "fbci.pi f20, 0\n"
        "fbci.pi f21, 0\n" ::
            : "f20", "f21");

    साठी (int64_t kb = 0; kb < K_blocks; kb++) {
        स्थिरांक block_q4_0 * blk0  = q_row0 + kb;
        स्थिरांक block_q4_0 * blk1  = q_row1 + kb;
        स्थिरांक दशांश *      b_ptr = b_col + (kb << 5);

        __asm__ अस्थिर(
            "fbci.pi     f10, 0\n"
            "fbci.pi     f16, 0\n"

            "flw.ps      f13, %[b_low0]\n"
            "flw.ps      f15, %[b_high0]\n"

            "fgb.ps      f11, f31(%[a_ptr0_0])\n"
            "fgb.ps      f17, f31(%[a_ptr1_0])\n"

            "fandi.pi    f12, f11, 15\n"
            "faddi.pi    f12, f12, -8\n"
            "fcvt.ps.pw  f12, f12, rne\n"
            "fmadd.ps    f10, f12, f13, f10, rne\n"

            "fandi.pi    f18, f17, 15\n"
            "faddi.pi    f18, f18, -8\n"
            "fcvt.ps.pw  f18, f18, rne\n"
            "fmadd.ps    f16, f18, f13, f16, rne\n"

            "fsrli.pi    f14, f11, 4\n"
            "fandi.pi    f14, f14, 15\n"
            "faddi.pi    f14, f14, -8\n"
            "fcvt.ps.pw  f14, f14, rne\n"
            "fmadd.ps    f10, f14, f15, f10, rne\n"

            "fsrli.pi    f19, f17, 4\n"
            "fandi.pi    f19, f19, 15\n"
            "faddi.pi    f19, f19, -8\n"
            "fcvt.ps.pw  f19, f19, rne\n"
            "fmadd.ps    f16, f19, f15, f16, rne\n"

            "flw.ps      f13, %[b_low1]\n"
            "flw.ps      f15, %[b_high1]\n"

            "fgb.ps      f11, f31(%[a_ptr0_1])\n"
            "fgb.ps      f17, f31(%[a_ptr1_1])\n"

            "fandi.pi    f12, f11, 15\n"
            "faddi.pi    f12, f12, -8\n"
            "fcvt.ps.pw  f12, f12, rne\n"
            "fmadd.ps    f10, f12, f13, f10, rne\n"

            "fandi.pi    f18, f17, 15\n"
            "faddi.pi    f18, f18, -8\n"
            "fcvt.ps.pw  f18, f18, rne\n"
            "fmadd.ps    f16, f18, f13, f16, rne\n"

            "fsrli.pi    f14, f11, 4\n"
            "fandi.pi    f14, f14, 15\n"
            "faddi.pi    f14, f14, -8\n"
            "fcvt.ps.pw  f14, f14, rne\n"
            "fmadd.ps    f10, f14, f15, f10, rne\n"

            "fsrli.pi    f19, f17, 4\n"
            "fandi.pi    f19, f19, 15\n"
            "faddi.pi    f19, f19, -8\n"
            "fcvt.ps.pw  f19, f19, rne\n"
            "fmadd.ps    f16, f19, f15, f16, rne\n"
            :
            : [a_ptr0_0] "r"(&blk0->qs[0]), [a_ptr0_1] "r"(&blk0->qs[8]), [a_ptr1_0] "r"(&blk1->qs[0]),
              [a_ptr1_1] "r"(&blk1->qs[8]), [b_low0] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[0]),
              [b_high0] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[16]), [b_low1] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[8]),
              [b_high1] "m"(*(स्थिरांक दशांश (*)[8]) & b_ptr[24])
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19");

        स्थिरांक uint32_t scale_raw0 = (uint32_t) blk0->d;
        स्थिरांक uint32_t scale_raw1 = (uint32_t) blk1->d;
        __asm__ अस्थिर(
            "fbcx.ps     f24, %[s0]\n"
            "fcvt.ps.f16 f24, f24\n"
            "fmadd.ps    f20, f10, f24, f20\n"
            "fbcx.ps     f25, %[s1]\n"
            "fcvt.ps.f16 f25, f25\n"
            "fmadd.ps    f21, f16, f25, f21\n"
            :
            : [s0] "r"(scale_raw0), [s1] "r"(scale_raw1)
            : "f20", "f21", "f24", "f25");
    }

    दशांश result0, result1;
    __asm__ __volatile__(
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result0)::"t0", "f1", "f2", "f3", "f4", "f5");
    __asm__ __volatile__(
        "fswizz.ps f1, f21, 0xB1 \n\t"
        "fadd.ps   f2, f21, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(result1)::"t0", "f1", "f2", "f3", "f4", "f5");

    *out0 = result0;
    *out1 = result1;
}
