//******************************************************************************
// Bare Metal Softmax F32 Kernel
// Softmax function: y[i] = exp(x[i] - max) / sum(exp(x[j] - max))
//
// Algorithm:
// 1. Apply scaling: x' = x * scale
// 2. Add mask/bias if present: x' = x' + mask * slope (ALiBi support)
// 3. Find max value for numerical stability: max = max(x')
// 4. Compute exponentials: exp_vals[i] = exp(x'[i] - max)
// 5. Compute sum: sum = sum(exp_vals)
// 6. Normalize: y[i] = exp_vals[i] / sum
//
// Features supported:
// - Temperature scaling via scale parameter
// - Attention masking (transformer masks)
// - ALiBi (Attention with Linear Biases) positional encoding
// - Numerical stability (subtract max before exp)
// - ggml broadcasting rules for mask tensors
//
// Mask Broadcasting Rules (ggml-specific, not standard numpy):
// - Dimension 0: mask.ne[0] == input.ne[0] (exact match required)
// - Dimension 1: mask.ne[1] >= input.ne[1] (allows larger pre-allocated masks)
// - Dimension 2: input.ne[2] % mask.ne[2] == 0 (modulo broadcasting)
// - Dimension 3: input.ne[3] % mask.ne[3] == 0 (modulo broadcasting)
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

// Softmax kernel parameters structure (from ggml-et-ops.h)
ರಚನೆ ggml_et_softmax_params {
    ರಚನೆ ggml_tensor src0;      // F32 input tensor
    ರಚನೆ ggml_tensor src1;      // F32 mask tensor (optional, may be zeroed if not used)
    ರಚನೆ ggml_tensor src2;      // F32 sinks tensor (optional, may be zeroed if not used)
    ರಚನೆ ggml_tensor dst;       // F32 output tensor
    ದಶಮಾಂಶ              scale;     // Scale factor (temperature scaling)
    ದಶಮಾಂಶ              max_bias;  // Max bias for ALiBi (0.0f if not used)
};

#ವ್ಯಾಖ್ಯಾನಿಸು LOG2E_F 1.4426950408889634f

ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    ದಶಮಾಂಶ    max_val;
    ದಶಮಾಂಶ    ಮೊತ್ತ_val;
    uint32_t valid_ಮುಖವಾಡ;
} softmax_params_t;

ಸ್ಥಿರ ಸಾಲೊಳಗೆ bool softmax_lane_is_valid(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ (x == x) && (x != -INFINITY) && (x != INFINITY);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ softmax_params_t softmax_params_ಖಾಲಿಗಳು(ಶೂನ್ಯ) {
    softmax_params_t p;
    p.max_val    = -INFINITY;
    p.ಮೊತ್ತ_val    = 0.0f;
    p.valid_ಮುಖವಾಡ = 0;
    ಹಿಂತಿರುಗಿ p;
}

// chunk_transform_ps_8_branchless_mask
//
// Vector transform for 8 logits:
//
//   x = src * scale + (mask ? mask * slope : 0)
//
// Implemented branchlessly so masked and unmasked paths share the same
// instruction stream. Used by pass1 and pass2 vector loops.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ chunk_transform_ps_8_branchless_ಮುಖವಾಡ(ದಶಮಾಂಶ *       tmp8,
                                                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src,
                                                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಮುಖವಾಡ,
                                                        ದಶಮಾಂಶ         scale,
                                                        ದಶಮಾಂಶ         ಇಳಿಜಾರು) {
    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ       ms;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ         zero         = 0.0f;
    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ಮುಖವಾಡ_ತುಂಬು_m0 = (ಮುಖವಾಡ != ಖಾಲಿ) ? 0xFFul : 0x00ul;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *       mp           = (ಮುಖವಾಡ != ಖಾಲಿ) ? ಮುಖವಾಡ : &zero;

    __asm__ ಅಸ್ಥಿರ(
        "mova.x.m  %[ms]                \n\t"

        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f10, 0(%[p_scale])   \n\t"
        "fbc.ps    f11, 0(%[p_slope])   \n\t"
        "fbc.ps    f1, 0(%[p_zero])    \n\t"

        "mov.m.x   m0, %[maskm0], 0     \n\t"  // load mask if needed
        "flw.ps    f1, 0(%[mp])         \n\t"

        "mov.m.x   m0, x0, 0xFF         \n\t"

        "flw.ps    f0, 0(%[sp])         \n\t"
        "fmul.ps   f0, f0, f10          \n\t"
        "fmul.ps   f1, f1, f11          \n\t"
        "fadd.ps   f0, f0, f1, rne      \n\t"
        "fsw.ps    f0, 0(%[tp])         \n\t"

        "mova.m.x  %[ms]                \n\t"
        : [ms] "=&r"(ms)
        : [tp] "r"(tmp8), [sp] "r"(src), [mp] "r"(mp), [p_zero] "r"(&zero), [p_scale] "r"(&scale),
          [p_ಇಳಿಜಾರು] "r"(&ಇಳಿಜಾರು), [maskm0] "r"(ಮುಖವಾಡ_ತುಂಬು_m0)
        : "f0", "f1", "f10", "f11", "memory");
}

// chunk_transform_ps_8_tail
//
// Same as chunk_transform_ps_8_branchless_mask but gates loads, compute,
// and stores with a caller-supplied m0 mask so that only `count` elements
// (1-7) are touched.  Used for the last sub-8 chunk of a non-aligned row.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ chunk_transform_ps_8_ಬಾಲ(ದಶಮಾಂಶ *       tmp8,
                                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src,
                                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಮುಖವಾಡ,
                                             ದಶಮಾಂಶ         scale,
                                             ದಶಮಾಂಶ         ಇಳಿಜಾರು,
                                             ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ಬಾಲ_m0) {
    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ       ms;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ         zero         = 0.0f;
    ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ಮುಖವಾಡ_ತುಂಬು_m0 = (ಮುಖವಾಡ != ಖಾಲಿ) ? ಬಾಲ_m0 : 0x00ul;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *       mp           = (ಮುಖವಾಡ != ಖಾಲಿ) ? ಮುಖವಾಡ : &zero;

    __asm__ ಅಸ್ಥಿರ(
        "mova.x.m  %[ms]                \n\t"

        // Broadcast constants with all lanes enabled
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f10, 0(%[p_scale])   \n\t"
        "fbc.ps    f11, 0(%[p_slope])   \n\t"
        "fbc.ps    f1, 0(%[p_zero])    \n\t"

        // Load mask data gated by tail mask
        "mov.m.x   m0, %[maskm0], 0     \n\t"
        "flw.ps    f1, 0(%[mp])         \n\t"

        // Load source, compute, and store gated by tail mask
        "mov.m.x   m0, %[tailm0], 0     \n\t"

        "flw.ps    f0, 0(%[sp])         \n\t"
        "fmul.ps   f0, f0, f10          \n\t"
        "fmul.ps   f1, f1, f11          \n\t"
        "fadd.ps   f0, f0, f1, rne      \n\t"
        "fsw.ps    f0, 0(%[tp])         \n\t"

        "mova.m.x  %[ms]                \n\t"
        : [ms] "=&r"(ms)
        : [tp] "r"(tmp8), [sp] "r"(src), [mp] "r"(mp), [p_zero] "r"(&zero), [p_scale] "r"(&scale),
          [p_ಇಳಿಜಾರು] "r"(&ಇಳಿಜಾರು), [maskm0] "r"(ಮುಖವಾಡ_ತುಂಬು_m0), [tailm0] "r"(ಬಾಲ_m0)
        : "f0", "f1", "f10", "f11", "memory");
}

// softmax_pass1_range
//
// Computes the numerically-stable softmax scan over a sub-range of a row.
//
// This implements the 1st pass of online softmax
//
//   max' = max(max, x)
//   sum' = sum * exp(old_max - max') + exp(x - max')
//
// and returns a partial result containing:
//
//   - max_val : maximum logit observed in this range
//   - sum_val : exp-normalized sum relative to max_val
//
// These partial results can be merged with softmax_params_merge() to obtain
// the result for the full row.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ softmax_params_t softmax_pass1_range(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src,
                                                   ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಮುಖವಾಡ,
                                                   ಪೂರ್ಣಾಂಕ           begin,
                                                   ಪೂರ್ಣಾಂಕ           ಕೊನೆ,
                                                   ದಶಮಾಂಶ         scale,
                                                   ದಶಮಾಂಶ         ಇಳಿಜಾರು) {
    __attribute__((aligned(32))) ದಶಮಾಂಶ lane_max[8];
    __attribute__((aligned(32))) ದಶಮಾಂಶ lane_ಮೊತ್ತ[8];
    __attribute__((aligned(32))) ದಶಮಾಂಶ tmp[8];

    uint8_t valid_ಮುಖವಾಡ = 0;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ one_f   = 1.0f;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ zero_f  = 0.0f;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ neg_inf = -INFINITY;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ log2e   = LOG2E_F;

    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ms;

    __asm__ ಅಸ್ಥಿರ(
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f20, 0(%[p_ninf])    \n\t"
        "fbc.ps    f21, 0(%[p_zero])    \n\t"
        "fbc.ps    f22, 0(%[p_one])     \n\t"
        "fbc.ps    f23, 0(%[p_log2e])   \n\t"
        : [ms] "=&r"(ms)
        : [p_ninf] "r"(&neg_inf), [p_zero] "r"(&zero_f), [p_one] "r"(&one_f), [p_log2e] "r"(&log2e)
        : "f20", "f21", "f22", "f23");

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ aligned_ಕೊನೆ = begin + ((ಕೊನೆ - begin) & ~7);

    // Process full 8-element chunks
    ಪೂರ್ಣಾಂಕ i = begin;
    ಗಾಗಿ (; i < aligned_ಕೊನೆ; i += 8) {
        chunk_transform_ps_8_branchless_ಮುಖವಾಡ(tmp, src + i, ಮುಖವಾಡ ? (ಮುಖವಾಡ + i) : ಖಾಲಿ, scale, ಇಳಿಜಾರು);

        uint8_t cur_ಮುಖವಾಡ = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
            ಆದರೆ (softmax_lane_is_valid(tmp[j])) {
                cur_ಮುಖವಾಡ |= (uint8_t) (1u << j);
            }
        }

        ಸ್ಥಿರಾಂಕ uint8_t init_ಮುಖವಾಡ = (uint8_t) (cur_ಮುಖವಾಡ & ~valid_ಮುಖವಾಡ);
        ಸ್ಥಿರಾಂಕ uint8_t upd_ಮುಖವಾಡ  = (uint8_t) (cur_ಮುಖವಾಡ & valid_ಮುಖವಾಡ);

        ಆದರೆ (init_ಮುಖವಾಡ || upd_ಮುಖವಾಡ) {
            __asm__ ಅಸ್ಥಿರ(
                "flw.ps    f0, 0(%[p_tmp])       \n\t"

                "mov.m.x   m0, %[initm], 0       \n\t"
                "fcmovm.ps f20, f0,  f20         \n\t"
                "fcmovm.ps f21, f22, f21         \n\t"

                "mov.m.x   m0, %[updm], 0        \n\t"
                "fmax.ps   f1, f20, f0           \n\t"

                "fsub.ps   f2, f20, f1, rne      \n\t"
                "fmul.ps   f2, f2,  f23          \n\t"
                "fexp.ps   f2, f2                \n\t"

                "fsub.ps   f3, f0,  f1, rne      \n\t"
                "fmul.ps   f3, f3,  f23          \n\t"
                "fexp.ps   f3, f3                \n\t"

                "fmul.ps   f21, f21, f2          \n\t"
                "fadd.ps   f21, f21, f3, rne     \n\t"
                "fcmovm.ps f20, f1,  f20         \n\t"

                "mov.m.x   m0, x0, 0xFF          \n\t"
                :
                : [p_tmp] "r"(tmp), [initm] "r"((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) init_ಮುಖವಾಡ), [updm] "r"((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) upd_ಮುಖವಾಡ)
                : "f0", "f1", "f2", "f3", "memory");

            valid_ಮುಖವಾಡ |= cur_ಮುಖವಾಡ;
        }
    }

    // Tail chunk: m0-gated load/compute/store for remaining 1-7 elements
    ಆದರೆ (i < ಕೊನೆ) {
        ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ಬಾಲ_m0 = (1ul << (ಕೊನೆ - i)) - 1;

        // Fill tmp with NaN so invalid lanes fail softmax_lane_is_valid
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; j++) {
            tmp[j] = __builtin_nanf("");
        }

        chunk_transform_ps_8_ಬಾಲ(tmp, src + i, ಮುಖವಾಡ ? (ಮುಖವಾಡ + i) : ಖಾಲಿ, scale, ಇಳಿಜಾರು, ಬಾಲ_m0);

        uint8_t cur_ಮುಖವಾಡ = 0;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 0; j < 8; ++j) {
            ಆದರೆ (softmax_lane_is_valid(tmp[j])) {
                cur_ಮುಖವಾಡ |= (uint8_t) (1u << j);
            }
        }

        ಸ್ಥಿರಾಂಕ uint8_t init_ಮುಖವಾಡ = (uint8_t) (cur_ಮುಖವಾಡ & ~valid_ಮುಖವಾಡ);
        ಸ್ಥಿರಾಂಕ uint8_t upd_ಮುಖವಾಡ  = (uint8_t) (cur_ಮುಖವಾಡ & valid_ಮುಖವಾಡ);

        ಆದರೆ (init_ಮುಖವಾಡ || upd_ಮುಖವಾಡ) {
            __asm__ ಅಸ್ಥಿರ(
                "flw.ps    f0, 0(%[p_tmp])       \n\t"

                "mov.m.x   m0, %[initm], 0       \n\t"
                "fcmovm.ps f20, f0,  f20         \n\t"
                "fcmovm.ps f21, f22, f21         \n\t"

                "mov.m.x   m0, %[updm], 0        \n\t"
                "fmax.ps   f1, f20, f0           \n\t"

                "fsub.ps   f2, f20, f1, rne      \n\t"
                "fmul.ps   f2, f2,  f23          \n\t"
                "fexp.ps   f2, f2                \n\t"

                "fsub.ps   f3, f0,  f1, rne      \n\t"
                "fmul.ps   f3, f3,  f23          \n\t"
                "fexp.ps   f3, f3                \n\t"

                "fmul.ps   f21, f21, f2          \n\t"
                "fadd.ps   f21, f21, f3, rne     \n\t"
                "fcmovm.ps f20, f1,  f20         \n\t"

                "mov.m.x   m0, x0, 0xFF          \n\t"
                :
                : [p_tmp] "r"(tmp), [initm] "r"((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) init_ಮುಖವಾಡ), [updm] "r"((ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ) upd_ಮುಖವಾಡ)
                : "f0", "f1", "f2", "f3", "memory");

            valid_ಮುಖವಾಡ |= cur_ಮುಖವಾಡ;
        }
    }

    __asm__ ಅಸ್ಥಿರ(
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fsw.ps    f20, 0(%[p_lmax])    \n\t"
        "fsw.ps    f21, 0(%[p_lsum])    \n\t"
        "mova.m.x  %[ms]                \n\t"
        :
        : [p_lmax] "r"(lane_max), [p_lsum] "r"(lane_ಮೊತ್ತ), [ms] "r"(ms)
        : "memory");

    softmax_params_t ಹೊರಹೋಗು = softmax_params_ಖಾಲಿಗಳು();
    ಹೊರಹೋಗು.valid_ಮುಖವಾಡ       = valid_ಮುಖವಾಡ;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; ++k) {
        ಆದರೆ (valid_ಮುಖವಾಡ & (1u << k)) {
            ಆದರೆ (ಹೊರಹೋಗು.valid_ಮುಖವಾಡ == (1u << k) || ಹೊರಹೋಗು.max_val == -INFINITY || lane_max[k] > ಹೊರಹೋಗು.max_val) {
                ಹೊರಹೋಗು.max_val = lane_max[k];
            }
        }
    }

    ಆದರೆ (ಹೊರಹೋಗು.max_val != -INFINITY) {
        // Compute lane correction factors via fexp.ps to stay consistent
        // with the fexp.ps used inside the online softmax loop above.
        // corr[k] = exp2((lane_max[k] - out.max_val) * LOG2E) = exp(lane_max[k] - out.max_val)
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ                        neg_max_l2 = -ಹೊರಹೋಗು.max_val * LOG2E_F;
        __attribute__((aligned(32))) ದಶಮಾಂಶ corr[8];
        __asm__ ಅಸ್ಥಿರ(
            "mova.x.m  %[ms]              \n\t"
            "mov.m.x   m0, x0, 0xFF       \n\t"
            "fbc.ps    f0, 0(%[p_nml2])   \n\t"
            "fbc.ps    f2, 0(%[p_l2e])    \n\t"
            "flw.ps    f1, 0(%[p_lmax])   \n\t"
            "fmadd.ps  f0, f1, f2, f0     \n\t"
            "fexp.ps   f0, f0             \n\t"
            "fsw.ps    f0, 0(%[p_corr])   \n\t"
            "mova.m.x  %[ms]              \n\t"
            :
            : [p_nml2] "r"(&neg_max_l2), [p_l2e] "r"(&log2e), [p_lmax] "r"(lane_max), [p_corr] "r"(corr), [ms] "r"(ms)
            : "f0", "f1", "f2", "memory");
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ k = 0; k < 8; ++k) {
            ಆದರೆ (valid_ಮುಖವಾಡ & (1u << k)) {
                ಹೊರಹೋಗು.ಮೊತ್ತ_val += lane_ಮೊತ್ತ[k] * corr[k];
            }
        }
    }

    ಹಿಂತಿರುಗಿ ಹೊರಹೋಗು;
}

// Pass 2 (normalize) over [begin, end).
//
// Computes: dst[i] = exp(x[i]*scale + mask[i]*slope - max) / sum
//
// Uses fexp.ps for the numerator; the denominator (params.sum_val) must
// already be fully computed by the caller (pass1 + any sink merge).
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ softmax_pass2_range(ದಶಮಾಂಶ *          dst,
                                       ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *    src,
                                       ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *    ಮುಖವಾಡ,
                                       ಪೂರ್ಣಾಂಕ              begin,
                                       ಪೂರ್ಣಾಂಕ              ಕೊನೆ,
                                       ದಶಮಾಂಶ            scale,
                                       ದಶಮಾಂಶ            ಇಳಿಜಾರು,
                                       softmax_params_t params) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ s2      = scale * LOG2E_F;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ sl2     = ಇಳಿಜಾರು * LOG2E_F;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ neg_ml2 = -params.max_val * LOG2E_F;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ inv_ಮೊತ್ತ = et_fdiv(1.0f, params.ಮೊತ್ತ_val);

    ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ms;

    __asm__ ಅಸ್ಥಿರ(
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f10, 0(%[p_s2])      \n\t"
        "fbc.ps    f12, 0(%[p_nml2])    \n\t"
        "fbc.ps    f13, 0(%[p_inv])     \n\t"
        : [ms] "=&r"(ms)
        : [p_s2] "r"(&s2), [p_nml2] "r"(&neg_ml2), [p_inv] "r"(&inv_ಮೊತ್ತ)
        : "f10", "f12", "f13");

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ aligned_ಕೊನೆ = begin + ((ಕೊನೆ - begin) & ~7);

    ಆದರೆ (ಮುಖವಾಡ != ಖಾಲಿ) {
        __asm__ ಅಸ್ಥಿರ("fbc.ps    f11, 0(%[p_sl2]) \n\t" : : [p_sl2] "r"(&sl2) : "f11");

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = begin; c < aligned_ಕೊನೆ; c += 8) {
            __asm__ ಅಸ್ಥಿರ(
                "flw.ps    f0, 0(%[sp])           \n\t"
                "flw.ps    f1, 0(%[mp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fmadd.ps  f0, f1, f11, f0        \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                :
                : [sp] "r"(src + c), [mp] "r"(ಮುಖವಾಡ + c), [dp] "r"(dst + c)
                : "f0", "f1", "memory");
        }

        // Tail chunk with m0 gating
        ಆದರೆ (aligned_ಕೊನೆ < ಕೊನೆ) {
            ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ಬಾಲ_m0 = (1ul << (ಕೊನೆ - aligned_ಕೊನೆ)) - 1;
            __asm__ ಅಸ್ಥಿರ(
                "mov.m.x   m0, %[tm], 0              \n\t"
                "flw.ps    f0, 0(%[sp])           \n\t"
                "flw.ps    f1, 0(%[mp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fmadd.ps  f0, f1, f11, f0        \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                "mov.m.x   m0, x0, 0xFF           \n\t"
                :
                : [sp] "r"(src + aligned_ಕೊನೆ), [mp] "r"(ಮುಖವಾಡ + aligned_ಕೊನೆ), [dp] "r"(dst + aligned_ಕೊನೆ),
                  [tm] "r"(ಬಾಲ_m0)
                : "f0", "f1", "memory");
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ c = begin; c < aligned_ಕೊನೆ; c += 8) {
            __asm__ ಅಸ್ಥಿರ(
                "flw.ps    f0, 0(%[sp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                :
                : [sp] "r"(src + c), [dp] "r"(dst + c)
                : "f0", "memory");
        }

        // Tail chunk with m0 gating
        ಆದರೆ (aligned_ಕೊನೆ < ಕೊನೆ) {
            ಸ್ಥಿರಾಂಕ ಚಿಹ್ನೆರಹಿತ ದೀರ್ಘ ಬಾಲ_m0 = (1ul << (ಕೊನೆ - aligned_ಕೊನೆ)) - 1;
            __asm__ ಅಸ್ಥಿರ(
                "mov.m.x   m0, %[tm], 0              \n\t"
                "flw.ps    f0, 0(%[sp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                "mov.m.x   m0, x0, 0xFF           \n\t"
                :
                : [sp] "r"(src + aligned_ಕೊನೆ), [dp] "r"(dst + aligned_ಕೊನೆ), [tm] "r"(ಬಾಲ_m0)
                : "f0", "memory");
        }
    }

    __asm__ ಅಸ್ಥಿರ("mova.m.x  %[ms] \n\t" ::[ms] "r"(ms));
}

// Single-core row path.
// pass1_range and pass2_range handle non-8-aligned cols internally via
// m0-gated tail chunks, so this function just passes cols directly.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ compute_softmax_ಅಡ್ಡಸಾಲು(ದಶಮಾಂಶ *       dst,
                                       ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src,
                                       ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಮುಖವಾಡ,
                                       ಪೂರ್ಣಾಂಕ           ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2,
                                       ದಶಮಾಂಶ         scale,
                                       ದಶಮಾಂಶ         ಇಳಿಜಾರು,
                                       ದಶಮಾಂಶ         sink_ಮೌಲ್ಯ,
                                       bool          use_sinks) {
    softmax_params_t params = softmax_pass1_range(src, ಮುಖವಾಡ, 0, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2, scale, ಇಳಿಜಾರು);

    ಆದರೆ (use_sinks) {
        // For sinks, use fully scalar et_expf to match the reference CPU
        // backend's expf precision.  Sink tests use small arrays (ne<=32)
        // so the scalar path has negligible performance impact.
        ದಶಮಾಂಶ max_val = params.max_val;
        ಆದರೆ (sink_ಮೌಲ್ಯ > max_val) {
            max_val = sink_ಮೌಲ್ಯ;
        }

        // Compute sum = Σ exp(x'[i] - max) + exp(sink - max)  (scalar)
        ದಶಮಾಂಶ ಮೊತ್ತ = 0.0f;
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2; ++i) {
            ದಶಮಾಂಶ x = src[i] * scale;
            ಆದರೆ (ಮುಖವಾಡ != ಖಾಲಿ) {
                x += ಮುಖವಾಡ[i] * ಇಳಿಜಾರು;
            }
            ಮೊತ್ತ += et_expf(x - max_val);
        }
        ಮೊತ್ತ += et_expf(sink_ಮೌಲ್ಯ - max_val);

        // Normalize: dst[i] = exp(x'[i] - max) / sum  (scalar)
        ದಶಮಾಂಶ inv_ಮೊತ್ತ = et_fdiv(1.0f, ಮೊತ್ತ);
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2; ++i) {
            ದಶಮಾಂಶ x = src[i] * scale;
            ಆದರೆ (ಮುಖವಾಡ != ಖಾಲಿ) {
                x += ಮುಖವಾಡ[i] * ಇಳಿಜಾರು;
            }
            dst[i] = et_expf(x - max_val) * inv_ಮೊತ್ತ;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (!params.valid_ಮುಖವಾಡ) {
            ಹಿಂತಿರುಗಿ;
        }
        softmax_pass2_range(dst, src, ಮುಖವಾಡ, 0, ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_2, scale, ಇಳಿಜಾರು, params);
    }
}

// Main entry point for Softmax kernel
ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_softmax_params * params, ಶೂನ್ಯ * env) {
    // Cast env to proper type
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    // Validate environment pointer
    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    // Get thread info using shire mask from environment
    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    // Return early if this hart is not active
    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ 0;
    }

    // Basic safety check on params
    ಆದರೆ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;  // Invalid pointer
    }

    // Extract tensor references
    ರಚನೆ ggml_tensor * src0     = &params->src0;     // Input tensor
    ರಚನೆ ggml_tensor * src1     = &params->src1;     // Mask tensor (optional)
    ರಚನೆ ggml_tensor * src2     = &params->src2;     // Sinks tensor (optional)
    ರಚನೆ ggml_tensor * dst      = &params->dst;      // Output tensor
    ದಶಮಾಂಶ                scale    = params->scale;     // Scale factor
    ದಶಮಾಂಶ                max_bias = params->max_bias;  // ALiBi max bias

    // Validate tensor types (F32 only)
    ಆದರೆ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;  // Unsupported type combination
    }

    // Check if mask is used and validate type
    bool use_ಮುಖವಾಡ = (src1->ದತ್ತಾಂಶ != ಖಾಲಿ && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16));

    bool use_sinks = (src2->ದತ್ತಾಂಶ != ಖಾಲಿ && src2->type == GGML_TYPE_F32);

    ದಶಮಾಂಶ * src0_ದತ್ತಾಂಶ  = (ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * dst_ದತ್ತಾಂಶ   = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;
    ದಶಮಾಂಶ * ಮುಖವಾಡ_ದತ್ತಾಂಶ  = use_ಮುಖವಾಡ ? (ದಶಮಾಂಶ *) src1->ದತ್ತಾಂಶ : ಖಾಲಿ;
    ದಶಮಾಂಶ * sinks_ದತ್ತಾಂಶ = use_sinks ? (ದಶಮಾಂಶ *) src2->ದತ್ತಾಂಶ : ಖಾಲಿ;

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;  // Null data pointer
    }


    ಸ್ಥಿರಾಂಕ int64_t ne00 = src0->ne[0];  // Sequence length (columns)
    ಸ್ಥಿರಾಂಕ int64_t ne01 = src0->ne[1];  // Number of rows
    ಸ್ಥಿರಾಂಕ int64_t ne02 = src0->ne[2];  // Batch/head dimension
    ಸ್ಥಿರಾಂಕ int64_t ne03 = src0->ne[3];  // Outer batch dimension

    // Fast path: softmax of a single element is always 1.0
    // (exp(x) / exp(x) == 1 for any x, regardless of scale/mask/bias)
    // Skip all ALiBi, mask, and sink setup.
    //
    // Each output element is 4 bytes.  A cache line is 64 bytes = 16 floats.
    // L1 is not coherent across harts, so each thread must own whole cache
    // lines to avoid cross-hart conflicts.
    ಆದರೆ (ne00 == 1) {
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_elems  = ne01 * ne02 * ne03;
        ಸ್ಥಿರಾಂಕ int64_t elems_ಪ್ರತಿ_cl = ET_CACHE_ಸಾಲು_ಗಾತ್ರ_ಬೈಟುಗಳು / (int64_t) ಗಾತ್ರ(ದಶಮಾಂಶ);  // 16
        ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_cls    = (ಒಟ್ಟು_elems + elems_ಪ್ರತಿ_cl - 1) / elems_ಪ್ರತಿ_cl;

        ಗಾಗಿ (int64_t cl = thread_id; cl < ಒಟ್ಟು_cls; cl += ಅಂಕೆ_ಎಳೆಗಳು) {
            ಸ್ಥಿರಾಂಕ int64_t ಆರಂಭ = cl * elems_ಪ್ರತಿ_cl;
            int64_t       ಕೊನೆ   = ಆರಂಭ + elems_ಪ್ರತಿ_cl;
            ಆದರೆ (ಕೊನೆ > ಒಟ್ಟು_elems) {
                ಕೊನೆ = ಒಟ್ಟು_elems;
            }
            ಗಾಗಿ (int64_t idx = ಆರಂಭ; idx < ಕೊನೆ; idx++) {
                dst_ದತ್ತಾಂಶ[idx] = 1.0f;
            }
        }
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ int64_t ne10 = use_ಮುಖವಾಡ ? src1->ne[0] : 0;  // Mask sequence length
    ಸ್ಥಿರಾಂಕ int64_t ne11 = use_ಮುಖವಾಡ ? src1->ne[1] : 0;  // Mask rows
    ಸ್ಥಿರಾಂಕ int64_t ne12 = use_ಮುಖವಾಡ ? src1->ne[2] : 0;  // Mask batch/head dimension
    ಸ್ಥಿರಾಂಕ int64_t ne13 = use_ಮುಖವಾಡ ? src1->ne[3] : 0;  // Mask outer batch dimension

    ಆದರೆ (use_ಮುಖವಾಡ) {
        // - Dimension 0: mask must equal input exactly
        // - Dimension 1: mask must be >= input (allows larger pre-allocated masks)
        // - Dimension 2: input must be divisible by mask (modulo broadcasting)
        // - Dimension 3: input must be divisible by mask (modulo broadcasting)
        ಆದರೆ (ne10 != ne00 ||                    // Dimension 0: exact match required
            ne11 < ne01 ||                     // Dimension 1: mask >= input
            (ne12 > 0 && ne02 % ne12 != 0) ||  // Dimension 2: input % mask == 0
            (ne13 > 0 && ne03 % ne13 != 0)) {  // Dimension 3: input % mask == 0
            ಹಿಂತಿರುಗಿ -1;                         // Incompatible dimensions for ggml softmax broadcasting
        }
    }

    // ALiBi slope calculation - compute per attention head
    ಸ್ಥಿರಾಂಕ uint32_t n_ತಲೆ      = (uint32_t) ne02;
    uint32_t       n_ತಲೆ_log2 = 0;
    ದಶಮಾಂಶ          m0          = 1.0f;
    ದಶಮಾಂಶ          m1          = 1.0f;

    ಆದರೆ (max_bias > 0.0f) {
        // This is equivalent to: 1 << floor(log2(n_head))
        n_ತಲೆ_log2 = 1;
        ಆಗ (n_ತಲೆ_log2 < n_ತಲೆ) {
            n_ತಲೆ_log2 <<= 1;
        }
        ಆದರೆ (n_ತಲೆ_log2 > n_ತಲೆ) {
            n_ತಲೆ_log2 >>= 1;
        }

        // Compute base slopes for ALiBi
        // m0 = 2^(-max_bias / n_head_log2)
        // m1 = 2^(-max_bias / (2 * n_head_log2))
        ದಶಮಾಂಶ inv_n_ತಲೆ_log2 = et_fdiv(1.0f, (ದಶಮಾಂಶ) n_ತಲೆ_log2);
        m0                    = et_expf(-max_bias * 0.69314718f * inv_n_ತಲೆ_log2);  // 0.69314718 = ln(2)
        m1                    = et_expf(-max_bias * 0.69314718f * inv_n_ತಲೆ_log2 * 0.5f);
    }

    // Process tensor row by row in parallel across flattened rows.
    // Flattened row index spans [i03, i02, i01] with row length ne00.
    //
    // When ne00 * sizeof(float) is not a multiple of the cache line size,
    // adjacent rows share cache lines.  Assign contiguous write groups to
    // each thread so every thread's write footprint covers whole cache
    // lines, preventing cross-hart L1 coherency issues.  When rows ARE
    // cache-line aligned, rows_per_wg == 1 and this degenerates to the
    // original stride-by-num_threads distribution.
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_i03 = ne02 * ne01;
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು   = ne03 * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_i03;
    ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_wg  = et_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_cacheline_ಗುಂಪು(ne00, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_wgs    = (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_wg - 1) / ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_wg;

    ಗಾಗಿ (int64_t wg = thread_id; wg < ಒಟ್ಟು_wgs; wg += ಅಂಕೆ_ಎಳೆಗಳು) {
        ಸ್ಥಿರಾಂಕ int64_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = wg * ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_wg;
        int64_t       ಅಡ್ಡಸಾಲು_ಕೊನೆ   = ಅಡ್ಡಸಾಲು_ಆರಂಭ + ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_wg;
        ಆದರೆ (ಅಡ್ಡಸಾಲು_ಕೊನೆ > ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು) {
            ಅಡ್ಡಸಾಲು_ಕೊನೆ = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
        }

        ಗಾಗಿ (int64_t ಅಡ್ಡಸಾಲು = ಅಡ್ಡಸಾಲು_ಆರಂಭ; ಅಡ್ಡಸಾಲು < ಅಡ್ಡಸಾಲು_ಕೊನೆ; ಅಡ್ಡಸಾಲು++) {
            ಸ್ಥಿರಾಂಕ int64_t i03 = ಅಡ್ಡಸಾಲು / ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_i03;
            ಸ್ಥಿರಾಂಕ int64_t rem = ಅಡ್ಡಸಾಲು % ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_i03;
            ಸ್ಥಿರಾಂಕ int64_t i02 = rem / ne01;
            ಸ್ಥಿರಾಂಕ int64_t i01 = rem % ne01;

            // Calculate ALiBi slope for this attention head
            ದಶಮಾಂಶ ಇಳಿಜಾರು = 1.0f;
            ಆದರೆ (max_bias > 0.0f) {
                ಸ್ಥಿರಾಂಕ uint32_t h = (uint32_t) i02;  // head index
                ಆದರೆ (h < n_ತಲೆ_log2) {
                    ಇಳಿಜಾರು = m0;
                    ಗಾಗಿ (uint32_t i = 0; i < h; i++) {
                        ಇಳಿಜಾರು *= m0;
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಸ್ಥಿರಾಂಕ uint32_t ಘಾತಾಂಕ = 2 * (h - n_ತಲೆ_log2) + 1;
                    ಇಳಿಜಾರು              = m1;
                    ಗಾಗಿ (uint32_t i = 1; i < ಘಾತಾಂಕ; i++) {
                        ಇಳಿಜಾರು *= m1;
                    }
                }
            }

            ದಶಮಾಂಶ sink_ಮೌಲ್ಯ = 0.0f;
            ಆದರೆ (use_sinks && sinks_ದತ್ತಾಂಶ) {
                sink_ಮೌಲ್ಯ = sinks_ದತ್ತಾಂಶ[i02];
            }

            ಸ್ಥಿರಾಂಕ int64_t src_ಸ್ಥಾನಾಂತರ = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00;

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * src_ಅಡ್ಡಸಾಲು  = src0_ದತ್ತಾಂಶ + src_ಸ್ಥಾನಾಂತರ;
            ದಶಮಾಂಶ *       dst_ಅಡ್ಡಸಾಲು  = dst_ದತ್ತಾಂಶ + src_ಸ್ಥಾನಾಂತರ;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು = ಖಾಲಿ;

            ಆದರೆ (use_ಮುಖವಾಡ && ಮುಖವಾಡ_ದತ್ತಾಂಶ) {
                ಸ್ಥಿರಾಂಕ int64_t ಮುಖವಾಡ_i03 = (ne13 > 0) ? i03 % ne13 : 0;
                ಸ್ಥಿರಾಂಕ int64_t ಮುಖವಾಡ_i02 = (ne12 > 0) ? i02 % ne12 : 0;
                ಸ್ಥಿರಾಂಕ int64_t ಮುಖವಾಡ_i01 = i01;

                ಸ್ಥಿರಾಂಕ int64_t ಮುಖವಾಡ_ಸ್ಥಾನಾಂತರ = ಮುಖವಾಡ_i03 * ne12 * ne11 * ne10 + ಮುಖವಾಡ_i02 * ne11 * ne10 + ಮುಖವಾಡ_i01 * ne10;

                ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು = ಮುಖವಾಡ_ದತ್ತಾಂಶ + ಮುಖವಾಡ_ಸ್ಥಾನಾಂತರ;
            }

            compute_softmax_ಅಡ್ಡಸಾಲು(dst_ಅಡ್ಡಸಾಲು, src_ಅಡ್ಡಸಾಲು, ಮುಖವಾಡ_ಅಡ್ಡಸಾಲು, (ಪೂರ್ಣಾಂಕ) ne00, scale, ಇಳಿಜಾರು, sink_ಮೌಲ್ಯ, use_sinks);
        }
    }

    ಹಿಂತಿರುಗಿ 0;  // Success
}
