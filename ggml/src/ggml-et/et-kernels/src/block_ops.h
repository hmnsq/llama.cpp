//******************************************************************************
// ET Vectorized Block Operations Library
// Provides optimized block-level operations using ET hardware vector instructions
//******************************************************************************

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ ਖੰਡ_OPS_H
#    define ਖੰਡ_OPS_H

#    include "math_fp.h"
#    include "quants.h"

#    include <stdint.h>

//******************************************************************************
// Block Dot Product Operations
//******************************************************************************
ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) excl_ਬਹੁਲਕ(uint64_t val) {
    __asm__ __volatile__("csrw 0x7d3, %[csr_enc]\n" : : [csr_enc] "r"(val) : "x31");
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_q4_0(ਅਚਲ ਖੰਡ_q4_0 * a_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ਸ਼ੁਰੂ) {
    // Set mask register to enable all 8 vector elements
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));  // Save current mask
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

    // Use f10 as accumulator, init to 0
    __asm__ ਅਸਥਿਰ("fbci.ps f10, 0" ::: "f10");

    ਸਥਿਰ ਅਚਲ int32_t gather_ਨਮੂਨਾ[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ ਅਸਥਿਰ("flw.ps f31, %[gather]\n" : : [gather] "m"(*(ਅਚਲ int32_t (*)[8]) gather_ਨਮੂਨਾ) : "f31");

    // Process 32 elements in 2 chunks of 16 elements (8 bytes) each
    ਲਈ (ਪੂਰਨ chunk = 0; chunk < 2; chunk++) {
        ਪੂਰਨ ਵਿੱਥ_a      = chunk * 8;
        ਪੂਰਨ ਵਿੱਥ_b_ਨੀਵਾਂ  = chunk * 8;       // Activations for lower nibbles
        ਪੂਰਨ ਵਿੱਥ_b_ਉੱਚਾ = chunk * 8 + 16;  // Activations for upper nibbles (16 elements later)

        __asm__ ਅਸਥਿਰ(
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
            : [a_ptr] "r"(&a_ਖੰਡ->qs[ਵਿੱਥ_a]), [b_ਨੀਵਾਂ] "r"(&b_ਖਾਨਾ_ਸ਼ੁਰੂ[ਵਿੱਥ_b_ਨੀਵਾਂ]),
              [b_ਉੱਚਾ] "r"(&b_ਖਾਨਾ_ਸ਼ੁਰੂ[ਵਿੱਥ_b_ਉੱਚਾ])
            // Note: f10 is explicitly NOT listed in the clobbers here to ensure the compiler
            // preserves the running sum across C loop iterations safely.
            : "f11", "f12", "f13", "f14", "f15");
    }

    // Horizontal sum: reduce f10 into a single scalar
    ਦਸ਼ਮਲਵ final_ਜੋੜ;
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
        : [vout] "=f"(final_ਜੋੜ)::"t0", "f1", "f2", "f3", "f4", "f5", "f10");

    // Restore original mask
    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));

    ਅਚਲ ਦਸ਼ਮਲਵ scale = fp16_ਮੰਜ਼ਿਲ_fp32(a_ਖੰਡ->d);
    ਵਾਪਸ final_ਜੋੜ * scale;
}

// Compute dot product between dequantized q8_0 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 32 int8 values (QK8_0)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_q8_0(ਅਚਲ ਖੰਡ_q8_0 * a_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ਸ਼ੁਰੂ) {
    // Set mask register to enable all 8 vector elements
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));  // Save current mask
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements
    __asm__ ਅਸਥਿਰ("fbci.pi f10, 0" ::: "f10");       // Use f10 as accumulator, init to 0

    ਸਥਿਰ ਅਚਲ int32_t gather_ਨਮੂਨਾ[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };

    __asm__ ਅਸਥਿਰ("flw.ps f31, %[gather]\n" : : [gather] "m"(*(ਅਚਲ int32_t (*)[8]) gather_ਨਮੂਨਾ) : "f31");

    // Process 32 elements in 4 chunks of 8 elements each
    ਲਈ (ਪੂਰਨ chunk = 0; chunk < 4; chunk++) {
        ਪੂਰਨ ਵਿੱਥ = chunk << 3;  // chunk * 8

        __asm__ ਅਸਥਿਰ(
            "flw.ps f12, %[b_vec]\n"         // Load 8 B values (floats)
            "fgb.ps f11, f31(%[a_ptr])\n"    // Gather 8 int8 bytes from A using pattern
            "fcvt.ps.pw f11, f11\n"          // Convert int8 vector to float vector
            "fmadd.ps f10, f11, f12, f10\n"  // acc += a_vec * b_vec (8-wide)
            :
            : [a_ptr] "r"(&a_ਖੰਡ->qs[ਵਿੱਥ]), [b_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ਖਾਨਾ_ਸ਼ੁਰੂ[ਵਿੱਥ]),
              [scale] "m"(a_ਖੰਡ->d)
            : "f10", "f11", "f12");
    }

    // Horizontal sum: reduce f10 into a single scalar
    ਦਸ਼ਮਲਵ final_ਜੋੜ;
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
        : [vout] "=f"(final_ਜੋੜ)::"t0", "f10", "f2", "f3", "f4", "f5");

    // Restore original mask
    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));

    ਅਚਲ ਦਸ਼ਮਲਵ scale = fp16_ਮੰਜ਼ਿਲ_fp32(a_ਖੰਡ->d);
    ਵਾਪਸ final_ਜੋੜ * scale;
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

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) q8_dot_reset(ਖਾਲੀ) {
    __asm__ ਅਸਥਿਰ("fbci.pi f20, 0" ::: "f20");
}

// Accumulate n_blocks Q8_0 blocks into f20.
// Uses fg32b.ps (fast gather with scalar pattern) for aligned chunks,
// falls back to fgb.ps for chunks crossing a 32-byte boundary.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) q8_dot_tile(ਅਚਲ ਖੰਡ_q8_0 * q_ਪੰਕਤੀ,
                                                              ਅਚਲ ਦਸ਼ਮਲਵ *      b_ਖਾਨਾ,
                                                              int64_t            n_ਖੰਡਾਂ) {
    ਅਚਲ int32_t  gather_ਨਮੂਨਾ[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    ਅਚਲ uint64_t gather_0_ਮੰਜ਼ਿਲ_7     = 0x398a418820ULL;

    __asm__ ਅਸਥਿਰ("flw.ps f31, %[g]\n" : : [g] "m"(*(ਅਚਲ int32_t (*)[8]) gather_ਨਮੂਨਾ) : "f31");

    ਲਈ (int64_t kb = 0; kb < n_ਖੰਡਾਂ; kb++) {
        ਅਚਲ ਖੰਡ_q8_0 * blk         = q_ਪੰਕਤੀ + kb;
        ਅਚਲ ਦਸ਼ਮਲਵ *      b_ptr       = b_ਖਾਨਾ + (kb << 5);
        ਅਚਲ uintptr_t    qs_addr     = (uintptr_t) blk->qs;
        ਅਚਲ uintptr_t    qs_aligned  = qs_addr & ~(uintptr_t) 31;
        ਅਚਲ uintptr_t    qs_ਨੀਵਾਂ      = qs_addr & 31;
        ਅਚਲ ਪੂਰਨ          fast_chunks = (ਪੂਰਨ) ((32 - qs_ਨੀਵਾਂ) >> 3);

        ਜੇ (fast_chunks >= 3) {
            __asm__ ਅਸਥਿਰ(
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
                : [gi] "r"(gather_0_ਮੰਜ਼ਿਲ_7), [ap0] "r"(qs_addr), [ap1] "r"(qs_aligned | ((qs_addr + 8) & 31)),
                  [ap2] "r"(qs_aligned | ((qs_addr + 16) & 31)), [ap3] "r"(&blk->qs[24]),
                  [bv0] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[0]), [bv1] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[8]),
                  [bv2] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[16]), [bv3] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12");
        } ਨਹੀਂਤਾਂ ਜੇ (fast_chunks == 2) {
            __asm__ ਅਸਥਿਰ(
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
                : [gi] "r"(gather_0_ਮੰਜ਼ਿਲ_7), [ap0] "r"(qs_addr), [ap1] "r"(qs_aligned | ((qs_addr + 8) & 31)),
                  [ap2] "r"(&blk->qs[16]), [ap3] "r"(&blk->qs[24]), [bv0] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[8]), [bv2] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12");
        } ਨਹੀਂਤਾਂ ਜੇ (fast_chunks == 1) {
            __asm__ ਅਸਥਿਰ(
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
                : [gi] "r"(gather_0_ਮੰਜ਼ਿਲ_7), [ap0] "r"(qs_addr), [ap1] "r"(&blk->qs[8]), [ap2] "r"(&blk->qs[16]),
                  [ap3] "r"(&blk->qs[24]), [bv0] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[8]), [bv2] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12");
        } ਨਹੀਂਤਾਂ {
            __asm__ ਅਸਥਿਰ(
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
                  [bv0] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[0]), [bv1] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[8]),
                  [bv2] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[16]), [bv3] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12");
        }

        // f20 += f10 * broadcast(scale) — hardware fp16→fp32 via FCVT.PS.F16
        uint32_t scale_raw = (uint32_t) blk->d;
        __asm__ ਅਸਥਿਰ(
            "fbcx.ps f15, %[sb]\n"
            "fcvt.ps.f16 f15, f15\n"
            "fmadd.ps f20, f10, f15, f20\n"
            :
            : [sb] "r"(scale_raw)
            : "f15", "f20");
    }
}

// Horizontal sum of 8-element vector accumulator f20.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ __attribute__((always_inline)) q8_dot_reduce(ਖਾਲੀ) {
    ਦਸ਼ਮਲਵ ਨਤੀਜਾ;
    __asm__ __volatile__(
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(ਨਤੀਜਾ)::"t0", "f1", "f2", "f3", "f4", "f5");
    ਵਾਪਸ ਨਤੀਜਾ;
}

// Full-row dot product (convenience wrapper)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ compute_ਪੰਕਤੀ_dot_q8_0(ਅਚਲ ਖੰਡ_q8_0 * q_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ, int64_t K_ਖੰਡਾਂ) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ saved_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(saved_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");
    q8_dot_reset();
    q8_dot_tile(q_ਪੰਕਤੀ, b_ਖਾਨਾ, K_ਖੰਡਾਂ);
    ਦਸ਼ਮਲਵ ਨਤੀਜਾ = q8_dot_reduce();
    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(saved_ਨਕਾਬ));
    ਵਾਪਸ ਨਤੀਜਾ;
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

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ saved_ਨਕਾਬ;
} q8_dot_state;

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ q8_dot_begin(q8_dot_state * state) {
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(state->saved_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ q8_dot_ਸਿਰਾ(ਅਚਲ q8_dot_state * state) {
    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(state->saved_ਨਕਾਬ));
}

// Equivalent to q8_dot_reset+tile+reduce, without touching the mask register.
// Caller is responsible for q8_dot_begin/end around the surrounding loop.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ q8_dot_compute(ਅਚਲ ਖੰਡ_q8_0 * q_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ, int64_t K_ਖੰਡਾਂ) {
    q8_dot_reset();
    q8_dot_tile(q_ਪੰਕਤੀ, b_ਖਾਨਾ, K_ਖੰਡਾਂ);
    ਵਾਪਸ q8_dot_reduce();
}

// Compute two row dots together while reusing the same loaded B chunks.
//
// Safe when every row starts at the same 32-byte offset, i.e. the Q8 row stride
// is a multiple of 32. In that case the gather/alignment pattern is the same
// for both rows at a given `kb`, so one set of B vector loads feeds both row
// accumulators.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ q8_dot_compute_x2_aligned(ਅਚਲ ਖੰਡ_q8_0 * q_row0,
                                             ਅਚਲ ਖੰਡ_q8_0 * q_row1,
                                             ਅਚਲ ਦਸ਼ਮਲਵ *      b_ਖਾਨਾ,
                                             int64_t            K_ਖੰਡਾਂ,
                                             ਦਸ਼ਮਲਵ *            out0,
                                             ਦਸ਼ਮਲਵ *            out1) {
    ਅਚਲ int32_t  gather_ਨਮੂਨਾ[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    ਅਚਲ uint64_t gather_0_ਮੰਜ਼ਿਲ_7     = 0x398a418820ULL;
    __asm__ ਅਸਥਿਰ("flw.ps f31, %[g]\n" : : [g] "m"(*(ਅਚਲ int32_t (*)[8]) gather_ਨਮੂਨਾ) : "f31");
    __asm__ ਅਸਥਿਰ(
        "fbci.pi f20, 0\n"
        "fbci.pi f21, 0\n" ::
            : "f20", "f21");

    ਲਈ (int64_t kb = 0; kb < K_ਖੰਡਾਂ; kb++) {
        ਅਚਲ ਖੰਡ_q8_0 * blk0  = q_row0 + kb;
        ਅਚਲ ਖੰਡ_q8_0 * blk1  = q_row1 + kb;
        ਅਚਲ ਦਸ਼ਮਲਵ *      b_ptr = b_ਖਾਨਾ + (kb << 5);

        ਅਚਲ uintptr_t qs_addr0    = (uintptr_t) blk0->qs;
        ਅਚਲ uintptr_t qs_addr1    = (uintptr_t) blk1->qs;
        ਅਚਲ uintptr_t qs_aligned0 = qs_addr0 & ~(uintptr_t) 31;
        ਅਚਲ uintptr_t qs_aligned1 = qs_addr1 & ~(uintptr_t) 31;
        ਅਚਲ ਪੂਰਨ       fast_chunks = (ਪੂਰਨ) ((32 - (qs_addr0 & 31)) >> 3);

        ਜੇ (fast_chunks >= 3) {
            __asm__ ਅਸਥਿਰ(
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
                : [gi] "r"(gather_0_ਮੰਜ਼ਿਲ_7), [r0ap0] "r"(qs_addr0), [r0ap1] "r"(qs_aligned0 | ((qs_addr0 + 8) & 31)),
                  [r0ap2] "r"(qs_aligned0 | ((qs_addr0 + 16) & 31)), [r0ap3] "r"(&blk0->qs[24]), [r1ap0] "r"(qs_addr1),
                  [r1ap1] "r"(qs_aligned1 | ((qs_addr1 + 8) & 31)), [r1ap2] "r"(qs_aligned1 | ((qs_addr1 + 16) & 31)),
                  [r1ap3] "r"(&blk1->qs[24]), [bv0] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[8]), [bv2] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17");
        } ਨਹੀਂਤਾਂ ਜੇ (fast_chunks == 2) {
            __asm__ ਅਸਥਿਰ(
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
                : [gi] "r"(gather_0_ਮੰਜ਼ਿਲ_7), [r0ap0] "r"(qs_addr0), [r0ap1] "r"(qs_aligned0 | ((qs_addr0 + 8) & 31)),
                  [r0ap2] "r"(&blk0->qs[16]), [r0ap3] "r"(&blk0->qs[24]), [r1ap0] "r"(qs_addr1),
                  [r1ap1] "r"(qs_aligned1 | ((qs_addr1 + 8) & 31)), [r1ap2] "r"(&blk1->qs[16]),
                  [r1ap3] "r"(&blk1->qs[24]), [bv0] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[8]), [bv2] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17");
        } ਨਹੀਂਤਾਂ ਜੇ (fast_chunks == 1) {
            __asm__ ਅਸਥਿਰ(
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
                : [gi] "r"(gather_0_ਮੰਜ਼ਿਲ_7), [r0ap0] "r"(qs_addr0), [r0ap1] "r"(&blk0->qs[8]), [r0ap2] "r"(&blk0->qs[16]),
                  [r0ap3] "r"(&blk0->qs[24]), [r1ap0] "r"(qs_addr1), [r1ap1] "r"(&blk1->qs[8]),
                  [r1ap2] "r"(&blk1->qs[16]), [r1ap3] "r"(&blk1->qs[24]), [bv0] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[8]), [bv2] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17");
        } ਨਹੀਂਤਾਂ {
            __asm__ ਅਸਥਿਰ(
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
                  [r1ap2] "r"(&blk1->qs[16]), [r1ap3] "r"(&blk1->qs[24]), [bv0] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[0]),
                  [bv1] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[8]), [bv2] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[16]),
                  [bv3] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[24])
                : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17");
        }

        ਅਚਲ uint32_t scale_raw0 = (uint32_t) blk0->d;
        ਅਚਲ uint32_t scale_raw1 = (uint32_t) blk1->d;
        __asm__ ਅਸਥਿਰ(
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

    ਦਸ਼ਮਲਵ result0;
    ਦਸ਼ਮਲਵ result1;
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
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f16_naive(ਅਚਲ uint16_t * a_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ਸ਼ੁਰੂ) {
    ਦਸ਼ਮਲਵ                acc_vec[8] __attribute__((aligned(32))) = { 0.0f };
    // Byte offsets for 16-bit (half-word) elements
    ਸਥਿਰ ਅਚਲ int32_t gather_ਨਮੂਨਾ[8]                       = { 0, 2, 4, 6, 8, 10, 12, 14 };
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ        ਆਰਜ਼ੀ_ਨਕਾਬ;

    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

    // Load the pattern once into f31 for the duration of all 4 chunks
    __asm__ ਅਸਥਿਰ("flw.ps f31, %[gather]\n" : : [gather] "m"(*(ਅਚਲ int32_t (*)[8]) gather_ਨਮੂਨਾ) : "f31");

    ਲਈ (ਪੂਰਨ chunk = 0; chunk < 4; chunk++) {
        // Correct pointers:
        // a_block elements are 2 bytes, b_col elements are 4 bytes
        ਅਚਲ uint16_t * a_ptr = &a_ਖੰਡ[chunk << 3];      // chunk * 8
        ਅਚਲ ਦਸ਼ਮਲਵ *    b_ptr = &b_ਖਾਨਾ_ਸ਼ੁਰੂ[chunk << 3];  // chunk * 8

        __asm__ ਅਸਥਿਰ(
            "flw.ps f10, %[acc]\n"
            "fgh.ps f11, f31(%[a_p])\n"  // Uses {0,2,4,6,8,10,12,14} byte offsets
            "fcvt.ps.f16 f11, f11\n"
            "flw.ps f12, (%[b_p])\n"     // Standard vector load (32-bit floats)
            "fmadd.ps f10, f11, f12, f10\n"
            "fsw.ps f10, %[result]\n"

            : [ਨਤੀਜਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) acc_vec)
            : [acc] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) acc_vec), [a_p] "r"(a_ptr), [b_p] "r"(b_ptr)
            : "f10", "f11", "f12");
    }

    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));

    ਵਾਪਸ acc_vec[0] + acc_vec[1] + acc_vec[2] + acc_vec[3] + acc_vec[4] + acc_vec[5] + acc_vec[6] + acc_vec[7];
}

// Compute dot product between f16 block and f32 column vector
// SCALAR implementation for partial blocks
// Block size: up to 32 f16 values (can handle partial blocks for misaligned K)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f16_partial(ਅਚਲ uint16_t * a_ਖੰਡ,
                                                          ਅਚਲ ਦਸ਼ਮਲਵ *    b_ਖਾਨਾ_ਸ਼ੁਰੂ,
                                                          ਪੂਰਨ              elements) {
    // This matches compute_block_dot_product_f16_naive behavior
    ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;

    ਲਈ (ਪੂਰਨ i = 0; i < elements; i++) {
        ਦਸ਼ਮਲਵ a_val = fp16_ਮੰਜ਼ਿਲ_fp32(a_ਖੰਡ[i]);
        ਦਸ਼ਮਲਵ b_val = b_ਖਾਨਾ_ਸ਼ੁਰੂ[i];
        ਜੋੜ += a_val * b_val;
    }

    ਵਾਪਸ ਜੋੜ;
}

// Compute dot product between f16 block and f16 column vector
// Scalar implementation for generic non-matrix-engine fallback paths.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f16_f16_partial(ਅਚਲ uint16_t * a_ਖੰਡ,
                                                              ਅਚਲ uint16_t * b_ਖਾਨਾ_ਸ਼ੁਰੂ,
                                                              ਪੂਰਨ              elements) {
    ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;

    ਲਈ (ਪੂਰਨ i = 0; i < elements; i++) {
        ਜੋੜ += fp16_ਮੰਜ਼ਿਲ_fp32(a_ਖੰਡ[i]) * fp16_ਮੰਜ਼ਿਲ_fp32(b_ਖਾਨਾ_ਸ਼ੁਰੂ[i]);
    }

    ਵਾਪਸ ਜੋੜ;
}

// Compute dot product between f16 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 32 f16 values (64 bytes = 1 cache line)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f16(ਅਚਲ uint16_t * a_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ਸ਼ੁਰੂ) {
    ਵਾਪਸ compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f16_partial(a_ਖੰਡ, b_ਖਾਨਾ_ਸ਼ੁਰੂ, QK_F16);
}

// Compute dot product between f32 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: up to 16 f32 values (can handle partial blocks for misaligned K)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f32_partial(ਅਚਲ ਦਸ਼ਮਲਵ * a_ਖੰਡ,
                                                          ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ਸ਼ੁਰੂ,
                                                          ਪੂਰਨ           elements) {
    ਦਸ਼ਮਲਵ acc_vec[8] = { 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f, 0.0f };  // Accumulator vector

    // Calculate how many full 8-element chunks we can process
    ਪੂਰਨ vec_ਸਿਰਾ = (elements / 8) * 8;

    ਜੇ (vec_ਸਿਰਾ > 0) {
        // Set mask register to enable all 8 vector elements
        ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ_ਨਕਾਬ;
        __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));  // Save current mask
        __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");           // Enable all 8 elements

        // Process full 8-element chunks
        ਲਈ (ਪੂਰਨ i = 0; i < vec_ਸਿਰਾ; i += 8) {
            // Vectorized f32 multiply-accumulate
            __asm__ ਅਸਥਿਰ(
                "flw.ps f10, %[acc]\n"           // Load current accumulator (8 floats)
                "flw.ps f11, %[a_vec]\n"         // Load 8 A values (f32)
                "flw.ps f12, %[b_vec]\n"         // Load 8 B values (f32)
                "fmadd.ps f10, f11, f12, f10\n"  // acc += a_vec * b_vec (8-wide)
                "fsw.ps f10, %[result]\n"        // Store back to accumulator

                : [ਨਤੀਜਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) acc_vec)
                : [acc] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) acc_vec), [a_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8])(a_ਖੰਡ + i)),
                  [b_vec] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8])(b_ਖਾਨਾ_ਸ਼ੁਰੂ + i))
                : "f10", "f11", "f12");
        }

        // Restore original mask
        __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));
    }

    // Horizontal sum: reduce 8 accumulator elements to single scalar
    ਦਸ਼ਮਲਵ final_ਜੋੜ = 0.0f;
    ਲਈ (ਪੂਰਨ i = 0; i < 8; i++) {
        final_ਜੋੜ += acc_vec[i];
    }

    // Handle remaining elements (< 8) with scalar operations
    ਲਈ (ਪੂਰਨ i = vec_ਸਿਰਾ; i < elements; i++) {
        final_ਜੋੜ += a_ਖੰਡ[i] * b_ਖਾਨਾ_ਸ਼ੁਰੂ[i];
    }

    ਵਾਪਸ final_ਜੋੜ;
}

// Compute dot product between f32 block and f16 column vector
// Scalar implementation for generic non-matrix-engine fallback paths.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f32_f16_partial(ਅਚਲ ਦਸ਼ਮਲਵ *    a_ਖੰਡ,
                                                              ਅਚਲ uint16_t * b_ਖਾਨਾ_ਸ਼ੁਰੂ,
                                                              ਪੂਰਨ              elements) {
    ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;

    ਲਈ (ਪੂਰਨ i = 0; i < elements; i++) {
        ਜੋੜ += a_ਖੰਡ[i] * fp16_ਮੰਜ਼ਿਲ_fp32(b_ਖਾਨਾ_ਸ਼ੁਰੂ[i]);
    }

    ਵਾਪਸ ਜੋੜ;
}

// Compute dot product between f32 block and f32 column vector
// Vectorized: processes 8 elements at a time using ET vector instructions
// Block size: 16 f32 values (64 bytes = 1 cache line)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f32(ਅਚਲ ਦਸ਼ਮਲਵ * a_ਖੰਡ, ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ_ਸ਼ੁਰੂ) {
    ਵਾਪਸ compute_ਖੰਡ_dot_ਗੁਣਨਫਲ_f32_partial(a_ਖੰਡ, b_ਖਾਨਾ_ਸ਼ੁਰੂ, QK_F32);

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

#ਜੇ_ਅੰਤ  // BLOCK_OPS_H

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) q4_dot_reset(ਖਾਲੀ) {
    __asm__ ਅਸਥਿਰ("fbci.pi f20, 0" ::: "f20");
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ __attribute__((always_inline)) q4_dot_tile(ਅਚਲ ਖੰਡ_q4_0 * q_ਪੰਕਤੀ,
                                                              ਅਚਲ ਦਸ਼ਮਲਵ *      b_ਖਾਨਾ,
                                                              int64_t            n_ਖੰਡਾਂ) {
    ਅਚਲ int32_t gather_ਨਮੂਨਾ[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ ਅਸਥਿਰ("flw.ps f31, %[g]\n" : : [g] "m"(*(ਅਚਲ int32_t (*)[8]) gather_ਨਮੂਨਾ) : "f31");

    ਲਈ (int64_t kb = 0; kb < n_ਖੰਡਾਂ; kb++) {
        ਅਚਲ ਖੰਡ_q4_0 * blk   = q_ਪੰਕਤੀ + kb;
        ਅਚਲ ਦਸ਼ਮਲਵ *      b_ptr = b_ਖਾਨਾ + (kb << 5);

        __asm__ ਅਸਥਿਰ(
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
            : [a_ptr0] "r"(&blk->qs[0]), [b_low0] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[0]),
              [b_high0] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[16]), [a_ptr1] "r"(&blk->qs[8]),
              [b_low1] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[8]), [b_high1] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[24])
            : "f10", "f11", "f12", "f13", "f14", "f15");

        uint32_t scale_raw = (uint32_t) blk->d;
        __asm__ ਅਸਥਿਰ(
            "fbcx.ps f15, %[sb]\n"
            "fcvt.ps.f16 f15, f15\n"
            "fmadd.ps f20, f10, f15, f20\n"
            :
            : [sb] "r"(scale_raw)
            : "f15", "f20");
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ __attribute__((always_inline)) q4_dot_reduce(ਖਾਲੀ) {
    ਦਸ਼ਮਲਵ ਨਤੀਜਾ;
    __asm__ __volatile__(
        "fswizz.ps f1, f20, 0xB1 \n\t"
        "fadd.ps   f2, f20, f1, rne \n\t"
        "fswizz.ps f3, f2, 0x4E \n\t"
        "fadd.ps   f4, f2, f3, rne \n\t"
        "fmvz.x.ps t0, f4, 4 \n\t"
        "fbcx.ps   f5, t0 \n\t"
        "fadd.ps   %[vout], f4, f5, rne \n\t"
        : [vout] "=f"(ਨਤੀਜਾ)::"t0", "f1", "f2", "f3", "f4", "f5");
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ compute_ਪੰਕਤੀ_dot_q4_0(ਅਚਲ ਖੰਡ_q4_0 * q_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ, int64_t K_ਖੰਡਾਂ) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ saved_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(saved_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");
    q4_dot_reset();
    q4_dot_tile(q_ਪੰਕਤੀ, b_ਖਾਨਾ, K_ਖੰਡਾਂ);
    ਦਸ਼ਮਲਵ ਨਤੀਜਾ = q4_dot_reduce();
    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(saved_ਨਕਾਬ));
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ saved_ਨਕਾਬ;
} q4_dot_state;

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ q4_dot_begin(q4_dot_state * state) {
    __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(state->saved_ਨਕਾਬ));
    __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ q4_dot_ਸਿਰਾ(ਅਚਲ q4_dot_state * state) {
    __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(state->saved_ਨਕਾਬ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ q4_dot_compute(ਅਚਲ ਖੰਡ_q4_0 * q_ਪੰਕਤੀ, ਅਚਲ ਦਸ਼ਮਲਵ * b_ਖਾਨਾ, int64_t K_ਖੰਡਾਂ) {
    q4_dot_reset();
    q4_dot_tile(q_ਪੰਕਤੀ, b_ਖਾਨਾ, K_ਖੰਡਾਂ);
    ਵਾਪਸ q4_dot_reduce();
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ q4_dot_compute_x2_aligned(ਅਚਲ ਖੰਡ_q4_0 * q_row0,
                                             ਅਚਲ ਖੰਡ_q4_0 * q_row1,
                                             ਅਚਲ ਦਸ਼ਮਲਵ *      b_ਖਾਨਾ,
                                             int64_t            K_ਖੰਡਾਂ,
                                             ਦਸ਼ਮਲਵ *            out0,
                                             ਦਸ਼ਮਲਵ *            out1) {
    ਅਚਲ int32_t gather_ਨਮੂਨਾ[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
    __asm__ ਅਸਥਿਰ("flw.ps f31, %[g]\n" : : [g] "m"(*(ਅਚਲ int32_t (*)[8]) gather_ਨਮੂਨਾ) : "f31");
    __asm__ ਅਸਥਿਰ(
        "fbci.pi f20, 0\n"
        "fbci.pi f21, 0\n" ::
            : "f20", "f21");

    ਲਈ (int64_t kb = 0; kb < K_ਖੰਡਾਂ; kb++) {
        ਅਚਲ ਖੰਡ_q4_0 * blk0  = q_row0 + kb;
        ਅਚਲ ਖੰਡ_q4_0 * blk1  = q_row1 + kb;
        ਅਚਲ ਦਸ਼ਮਲਵ *      b_ptr = b_ਖਾਨਾ + (kb << 5);

        __asm__ ਅਸਥਿਰ(
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
              [a_ptr1_1] "r"(&blk1->qs[8]), [b_low0] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[0]),
              [b_high0] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[16]), [b_low1] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[8]),
              [b_high1] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & b_ptr[24])
            : "f10", "f11", "f12", "f13", "f14", "f15", "f16", "f17", "f18", "f19");

        ਅਚਲ uint32_t scale_raw0 = (uint32_t) blk0->d;
        ਅਚਲ uint32_t scale_raw1 = (uint32_t) blk1->d;
        __asm__ ਅਸਥਿਰ(
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

    ਦਸ਼ਮਲਵ result0, result1;
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
