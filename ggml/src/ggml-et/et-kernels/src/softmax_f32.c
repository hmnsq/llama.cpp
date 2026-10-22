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

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

// Softmax kernel parameters structure (from ggml-et-ops.h)
ਬਣਤਰ ggml_et_softmax_params {
    ਬਣਤਰ ggml_tensor src0;      // F32 input tensor
    ਬਣਤਰ ggml_tensor src1;      // F32 mask tensor (optional, may be zeroed if not used)
    ਬਣਤਰ ggml_tensor src2;      // F32 sinks tensor (optional, may be zeroed if not used)
    ਬਣਤਰ ggml_tensor dst;       // F32 output tensor
    ਦਸ਼ਮਲਵ              scale;     // Scale factor (temperature scaling)
    ਦਸ਼ਮਲਵ              max_bias;  // Max bias for ALiBi (0.0f if not used)
};

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ LOG2E_F 1.4426950408889634f

ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    ਦਸ਼ਮਲਵ    max_val;
    ਦਸ਼ਮਲਵ    ਜੋੜ_val;
    uint32_t valid_ਨਕਾਬ;
} softmax_params_t;

ਸਥਿਰ ਸਤਰਅੰਦਰ bool softmax_lane_is_valid(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ (x == x) && (x != -INFINITY) && (x != INFINITY);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ softmax_params_t softmax_params_ਖਾਲੀ_ਥਾਵਾਂ(ਖਾਲੀ) {
    softmax_params_t p;
    p.max_val    = -INFINITY;
    p.ਜੋੜ_val    = 0.0f;
    p.valid_ਨਕਾਬ = 0;
    ਵਾਪਸ p;
}

// chunk_transform_ps_8_branchless_mask
//
// Vector transform for 8 logits:
//
//   x = src * scale + (mask ? mask * slope : 0)
//
// Implemented branchlessly so masked and unmasked paths share the same
// instruction stream. Used by pass1 and pass2 vector loops.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ chunk_transform_ps_8_branchless_ਨਕਾਬ(ਦਸ਼ਮਲਵ *       tmp8,
                                                        ਅਚਲ ਦਸ਼ਮਲਵ * src,
                                                        ਅਚਲ ਦਸ਼ਮਲਵ * ਨਕਾਬ,
                                                        ਦਸ਼ਮਲਵ         scale,
                                                        ਦਸ਼ਮਲਵ         ਢਲਾਨ) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ       ms;
    ਅਚਲ ਦਸ਼ਮਲਵ         zero         = 0.0f;
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਨਕਾਬ_ਲੱਦੋ_m0 = (ਨਕਾਬ != ਨਿਰਾ) ? 0xFFul : 0x00ul;
    ਅਚਲ ਦਸ਼ਮਲਵ *       mp           = (ਨਕਾਬ != ਨਿਰਾ) ? ਨਕਾਬ : &zero;

    __asm__ ਅਸਥਿਰ(
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
          [p_ਢਲਾਨ] "r"(&ਢਲਾਨ), [maskm0] "r"(ਨਕਾਬ_ਲੱਦੋ_m0)
        : "f0", "f1", "f10", "f11", "memory");
}

// chunk_transform_ps_8_tail
//
// Same as chunk_transform_ps_8_branchless_mask but gates loads, compute,
// and stores with a caller-supplied m0 mask so that only `count` elements
// (1-7) are touched.  Used for the last sub-8 chunk of a non-aligned row.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ chunk_transform_ps_8_ਪੂਛ(ਦਸ਼ਮਲਵ *       tmp8,
                                             ਅਚਲ ਦਸ਼ਮਲਵ * src,
                                             ਅਚਲ ਦਸ਼ਮਲਵ * ਨਕਾਬ,
                                             ਦਸ਼ਮਲਵ         scale,
                                             ਦਸ਼ਮਲਵ         ਢਲਾਨ,
                                             ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਪੂਛ_m0) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ       ms;
    ਅਚਲ ਦਸ਼ਮਲਵ         zero         = 0.0f;
    ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਨਕਾਬ_ਲੱਦੋ_m0 = (ਨਕਾਬ != ਨਿਰਾ) ? ਪੂਛ_m0 : 0x00ul;
    ਅਚਲ ਦਸ਼ਮਲਵ *       mp           = (ਨਕਾਬ != ਨਿਰਾ) ? ਨਕਾਬ : &zero;

    __asm__ ਅਸਥਿਰ(
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
          [p_ਢਲਾਨ] "r"(&ਢਲਾਨ), [maskm0] "r"(ਨਕਾਬ_ਲੱਦੋ_m0), [tailm0] "r"(ਪੂਛ_m0)
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
ਸਥਿਰ ਸਤਰਅੰਦਰ softmax_params_t softmax_pass1_range(ਅਚਲ ਦਸ਼ਮਲਵ * src,
                                                   ਅਚਲ ਦਸ਼ਮਲਵ * ਨਕਾਬ,
                                                   ਪੂਰਨ           begin,
                                                   ਪੂਰਨ           ਸਿਰਾ,
                                                   ਦਸ਼ਮਲਵ         scale,
                                                   ਦਸ਼ਮਲਵ         ਢਲਾਨ) {
    __attribute__((aligned(32))) ਦਸ਼ਮਲਵ lane_max[8];
    __attribute__((aligned(32))) ਦਸ਼ਮਲਵ lane_ਜੋੜ[8];
    __attribute__((aligned(32))) ਦਸ਼ਮਲਵ tmp[8];

    uint8_t valid_ਨਕਾਬ = 0;

    ਅਚਲ ਦਸ਼ਮਲਵ one_f   = 1.0f;
    ਅਚਲ ਦਸ਼ਮਲਵ zero_f  = 0.0f;
    ਅਚਲ ਦਸ਼ਮਲਵ neg_inf = -INFINITY;
    ਅਚਲ ਦਸ਼ਮਲਵ log2e   = LOG2E_F;

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ms;

    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f20, 0(%[p_ninf])    \n\t"
        "fbc.ps    f21, 0(%[p_zero])    \n\t"
        "fbc.ps    f22, 0(%[p_one])     \n\t"
        "fbc.ps    f23, 0(%[p_log2e])   \n\t"
        : [ms] "=&r"(ms)
        : [p_ninf] "r"(&neg_inf), [p_zero] "r"(&zero_f), [p_one] "r"(&one_f), [p_log2e] "r"(&log2e)
        : "f20", "f21", "f22", "f23");

    ਅਚਲ ਪੂਰਨ aligned_ਸਿਰਾ = begin + ((ਸਿਰਾ - begin) & ~7);

    // Process full 8-element chunks
    ਪੂਰਨ i = begin;
    ਲਈ (; i < aligned_ਸਿਰਾ; i += 8) {
        chunk_transform_ps_8_branchless_ਨਕਾਬ(tmp, src + i, ਨਕਾਬ ? (ਨਕਾਬ + i) : ਨਿਰਾ, scale, ਢਲਾਨ);

        uint8_t cur_ਨਕਾਬ = 0;
        ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
            ਜੇ (softmax_lane_is_valid(tmp[j])) {
                cur_ਨਕਾਬ |= (uint8_t) (1u << j);
            }
        }

        ਅਚਲ uint8_t init_ਨਕਾਬ = (uint8_t) (cur_ਨਕਾਬ & ~valid_ਨਕਾਬ);
        ਅਚਲ uint8_t upd_ਨਕਾਬ  = (uint8_t) (cur_ਨਕਾਬ & valid_ਨਕਾਬ);

        ਜੇ (init_ਨਕਾਬ || upd_ਨਕਾਬ) {
            __asm__ ਅਸਥਿਰ(
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
                : [p_tmp] "r"(tmp), [initm] "r"((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) init_ਨਕਾਬ), [updm] "r"((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) upd_ਨਕਾਬ)
                : "f0", "f1", "f2", "f3", "memory");

            valid_ਨਕਾਬ |= cur_ਨਕਾਬ;
        }
    }

    // Tail chunk: m0-gated load/compute/store for remaining 1-7 elements
    ਜੇ (i < ਸਿਰਾ) {
        ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਪੂਛ_m0 = (1ul << (ਸਿਰਾ - i)) - 1;

        // Fill tmp with NaN so invalid lanes fail softmax_lane_is_valid
        ਲਈ (ਪੂਰਨ j = 0; j < 8; j++) {
            tmp[j] = __builtin_nanf("");
        }

        chunk_transform_ps_8_ਪੂਛ(tmp, src + i, ਨਕਾਬ ? (ਨਕਾਬ + i) : ਨਿਰਾ, scale, ਢਲਾਨ, ਪੂਛ_m0);

        uint8_t cur_ਨਕਾਬ = 0;
        ਲਈ (ਪੂਰਨ j = 0; j < 8; ++j) {
            ਜੇ (softmax_lane_is_valid(tmp[j])) {
                cur_ਨਕਾਬ |= (uint8_t) (1u << j);
            }
        }

        ਅਚਲ uint8_t init_ਨਕਾਬ = (uint8_t) (cur_ਨਕਾਬ & ~valid_ਨਕਾਬ);
        ਅਚਲ uint8_t upd_ਨਕਾਬ  = (uint8_t) (cur_ਨਕਾਬ & valid_ਨਕਾਬ);

        ਜੇ (init_ਨਕਾਬ || upd_ਨਕਾਬ) {
            __asm__ ਅਸਥਿਰ(
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
                : [p_tmp] "r"(tmp), [initm] "r"((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) init_ਨਕਾਬ), [updm] "r"((ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ) upd_ਨਕਾਬ)
                : "f0", "f1", "f2", "f3", "memory");

            valid_ਨਕਾਬ |= cur_ਨਕਾਬ;
        }
    }

    __asm__ ਅਸਥਿਰ(
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fsw.ps    f20, 0(%[p_lmax])    \n\t"
        "fsw.ps    f21, 0(%[p_lsum])    \n\t"
        "mova.m.x  %[ms]                \n\t"
        :
        : [p_lmax] "r"(lane_max), [p_lsum] "r"(lane_ਜੋੜ), [ms] "r"(ms)
        : "memory");

    softmax_params_t ਬਾਹਰਲਾ = softmax_params_ਖਾਲੀ_ਥਾਵਾਂ();
    ਬਾਹਰਲਾ.valid_ਨਕਾਬ       = valid_ਨਕਾਬ;

    ਲਈ (ਪੂਰਨ k = 0; k < 8; ++k) {
        ਜੇ (valid_ਨਕਾਬ & (1u << k)) {
            ਜੇ (ਬਾਹਰਲਾ.valid_ਨਕਾਬ == (1u << k) || ਬਾਹਰਲਾ.max_val == -INFINITY || lane_max[k] > ਬਾਹਰਲਾ.max_val) {
                ਬਾਹਰਲਾ.max_val = lane_max[k];
            }
        }
    }

    ਜੇ (ਬਾਹਰਲਾ.max_val != -INFINITY) {
        // Compute lane correction factors via fexp.ps to stay consistent
        // with the fexp.ps used inside the online softmax loop above.
        // corr[k] = exp2((lane_max[k] - out.max_val) * LOG2E) = exp(lane_max[k] - out.max_val)
        ਅਚਲ ਦਸ਼ਮਲਵ                        neg_max_l2 = -ਬਾਹਰਲਾ.max_val * LOG2E_F;
        __attribute__((aligned(32))) ਦਸ਼ਮਲਵ corr[8];
        __asm__ ਅਸਥਿਰ(
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
        ਲਈ (ਪੂਰਨ k = 0; k < 8; ++k) {
            ਜੇ (valid_ਨਕਾਬ & (1u << k)) {
                ਬਾਹਰਲਾ.ਜੋੜ_val += lane_ਜੋੜ[k] * corr[k];
            }
        }
    }

    ਵਾਪਸ ਬਾਹਰਲਾ;
}

// Pass 2 (normalize) over [begin, end).
//
// Computes: dst[i] = exp(x[i]*scale + mask[i]*slope - max) / sum
//
// Uses fexp.ps for the numerator; the denominator (params.sum_val) must
// already be fully computed by the caller (pass1 + any sink merge).
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ softmax_pass2_range(ਦਸ਼ਮਲਵ *          dst,
                                       ਅਚਲ ਦਸ਼ਮਲਵ *    src,
                                       ਅਚਲ ਦਸ਼ਮਲਵ *    ਨਕਾਬ,
                                       ਪੂਰਨ              begin,
                                       ਪੂਰਨ              ਸਿਰਾ,
                                       ਦਸ਼ਮਲਵ            scale,
                                       ਦਸ਼ਮਲਵ            ਢਲਾਨ,
                                       softmax_params_t params) {
    ਅਚਲ ਦਸ਼ਮਲਵ s2      = scale * LOG2E_F;
    ਅਚਲ ਦਸ਼ਮਲਵ sl2     = ਢਲਾਨ * LOG2E_F;
    ਅਚਲ ਦਸ਼ਮਲਵ neg_ml2 = -params.max_val * LOG2E_F;
    ਅਚਲ ਦਸ਼ਮਲਵ inv_ਜੋੜ = et_fdiv(1.0f, params.ਜੋੜ_val);

    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ms;

    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f10, 0(%[p_s2])      \n\t"
        "fbc.ps    f12, 0(%[p_nml2])    \n\t"
        "fbc.ps    f13, 0(%[p_inv])     \n\t"
        : [ms] "=&r"(ms)
        : [p_s2] "r"(&s2), [p_nml2] "r"(&neg_ml2), [p_inv] "r"(&inv_ਜੋੜ)
        : "f10", "f12", "f13");

    ਅਚਲ ਪੂਰਨ aligned_ਸਿਰਾ = begin + ((ਸਿਰਾ - begin) & ~7);

    ਜੇ (ਨਕਾਬ != ਨਿਰਾ) {
        __asm__ ਅਸਥਿਰ("fbc.ps    f11, 0(%[p_sl2]) \n\t" : : [p_sl2] "r"(&sl2) : "f11");

        ਲਈ (ਪੂਰਨ c = begin; c < aligned_ਸਿਰਾ; c += 8) {
            __asm__ ਅਸਥਿਰ(
                "flw.ps    f0, 0(%[sp])           \n\t"
                "flw.ps    f1, 0(%[mp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fmadd.ps  f0, f1, f11, f0        \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                :
                : [sp] "r"(src + c), [mp] "r"(ਨਕਾਬ + c), [dp] "r"(dst + c)
                : "f0", "f1", "memory");
        }

        // Tail chunk with m0 gating
        ਜੇ (aligned_ਸਿਰਾ < ਸਿਰਾ) {
            ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਪੂਛ_m0 = (1ul << (ਸਿਰਾ - aligned_ਸਿਰਾ)) - 1;
            __asm__ ਅਸਥਿਰ(
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
                : [sp] "r"(src + aligned_ਸਿਰਾ), [mp] "r"(ਨਕਾਬ + aligned_ਸਿਰਾ), [dp] "r"(dst + aligned_ਸਿਰਾ),
                  [tm] "r"(ਪੂਛ_m0)
                : "f0", "f1", "memory");
        }
    } ਨਹੀਂਤਾਂ {
        ਲਈ (ਪੂਰਨ c = begin; c < aligned_ਸਿਰਾ; c += 8) {
            __asm__ ਅਸਥਿਰ(
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
        ਜੇ (aligned_ਸਿਰਾ < ਸਿਰਾ) {
            ਅਚਲ ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਪੂਛ_m0 = (1ul << (ਸਿਰਾ - aligned_ਸਿਰਾ)) - 1;
            __asm__ ਅਸਥਿਰ(
                "mov.m.x   m0, %[tm], 0              \n\t"
                "flw.ps    f0, 0(%[sp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                "mov.m.x   m0, x0, 0xFF           \n\t"
                :
                : [sp] "r"(src + aligned_ਸਿਰਾ), [dp] "r"(dst + aligned_ਸਿਰਾ), [tm] "r"(ਪੂਛ_m0)
                : "f0", "memory");
        }
    }

    __asm__ ਅਸਥਿਰ("mova.m.x  %[ms] \n\t" ::[ms] "r"(ms));
}

// Single-core row path.
// pass1_range and pass2_range handle non-8-aligned cols internally via
// m0-gated tail chunks, so this function just passes cols directly.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ compute_softmax_ਪੰਕਤੀ(ਦਸ਼ਮਲਵ *       dst,
                                       ਅਚਲ ਦਸ਼ਮਲਵ * src,
                                       ਅਚਲ ਦਸ਼ਮਲਵ * ਨਕਾਬ,
                                       ਪੂਰਨ           ਖਾਨੇ_ਗਿਣਤੀ_2,
                                       ਦਸ਼ਮਲਵ         scale,
                                       ਦਸ਼ਮਲਵ         ਢਲਾਨ,
                                       ਦਸ਼ਮਲਵ         sink_ਮੁੱਲ,
                                       bool          use_sinks) {
    softmax_params_t params = softmax_pass1_range(src, ਨਕਾਬ, 0, ਖਾਨੇ_ਗਿਣਤੀ_2, scale, ਢਲਾਨ);

    ਜੇ (use_sinks) {
        // For sinks, use fully scalar et_expf to match the reference CPU
        // backend's expf precision.  Sink tests use small arrays (ne<=32)
        // so the scalar path has negligible performance impact.
        ਦਸ਼ਮਲਵ max_val = params.max_val;
        ਜੇ (sink_ਮੁੱਲ > max_val) {
            max_val = sink_ਮੁੱਲ;
        }

        // Compute sum = Σ exp(x'[i] - max) + exp(sink - max)  (scalar)
        ਦਸ਼ਮਲਵ ਜੋੜ = 0.0f;
        ਲਈ (ਪੂਰਨ i = 0; i < ਖਾਨੇ_ਗਿਣਤੀ_2; ++i) {
            ਦਸ਼ਮਲਵ x = src[i] * scale;
            ਜੇ (ਨਕਾਬ != ਨਿਰਾ) {
                x += ਨਕਾਬ[i] * ਢਲਾਨ;
            }
            ਜੋੜ += et_expf(x - max_val);
        }
        ਜੋੜ += et_expf(sink_ਮੁੱਲ - max_val);

        // Normalize: dst[i] = exp(x'[i] - max) / sum  (scalar)
        ਦਸ਼ਮਲਵ inv_ਜੋੜ = et_fdiv(1.0f, ਜੋੜ);
        ਲਈ (ਪੂਰਨ i = 0; i < ਖਾਨੇ_ਗਿਣਤੀ_2; ++i) {
            ਦਸ਼ਮਲਵ x = src[i] * scale;
            ਜੇ (ਨਕਾਬ != ਨਿਰਾ) {
                x += ਨਕਾਬ[i] * ਢਲਾਨ;
            }
            dst[i] = et_expf(x - max_val) * inv_ਜੋੜ;
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (!params.valid_ਨਕਾਬ) {
            ਵਾਪਸ;
        }
        softmax_pass2_range(dst, src, ਨਕਾਬ, 0, ਖਾਨੇ_ਗਿਣਤੀ_2, scale, ਢਲਾਨ, params);
    }
}

// Main entry point for Softmax kernel
ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_softmax_params * params, ਖਾਲੀ * env) {
    // Cast env to proper type
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    // Validate environment pointer
    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    // Get thread info using shire mask from environment
    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    // Return early if this hart is not active
    ਜੇ (thread_id < 0) {
        ਵਾਪਸ 0;
    }

    // Basic safety check on params
    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;  // Invalid pointer
    }

    // Extract tensor references
    ਬਣਤਰ ggml_tensor * src0     = &params->src0;     // Input tensor
    ਬਣਤਰ ggml_tensor * src1     = &params->src1;     // Mask tensor (optional)
    ਬਣਤਰ ggml_tensor * src2     = &params->src2;     // Sinks tensor (optional)
    ਬਣਤਰ ggml_tensor * dst      = &params->dst;      // Output tensor
    ਦਸ਼ਮਲਵ                scale    = params->scale;     // Scale factor
    ਦਸ਼ਮਲਵ                max_bias = params->max_bias;  // ALiBi max bias

    // Validate tensor types (F32 only)
    ਜੇ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;  // Unsupported type combination
    }

    // Check if mask is used and validate type
    bool use_ਨਕਾਬ = (src1->ਅੰਕੜੇ != ਨਿਰਾ && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16));

    bool use_sinks = (src2->ਅੰਕੜੇ != ਨਿਰਾ && src2->type == GGML_TYPE_F32);

    ਦਸ਼ਮਲਵ * src0_ਅੰਕੜੇ  = (ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * dst_ਅੰਕੜੇ   = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;
    ਦਸ਼ਮਲਵ * ਨਕਾਬ_ਅੰਕੜੇ  = use_ਨਕਾਬ ? (ਦਸ਼ਮਲਵ *) src1->ਅੰਕੜੇ : ਨਿਰਾ;
    ਦਸ਼ਮਲਵ * sinks_ਅੰਕੜੇ = use_sinks ? (ਦਸ਼ਮਲਵ *) src2->ਅੰਕੜੇ : ਨਿਰਾ;

    ਜੇ (!src0_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;  // Null data pointer
    }


    ਅਚਲ int64_t ne00 = src0->ne[0];  // Sequence length (columns)
    ਅਚਲ int64_t ne01 = src0->ne[1];  // Number of rows
    ਅਚਲ int64_t ne02 = src0->ne[2];  // Batch/head dimension
    ਅਚਲ int64_t ne03 = src0->ne[3];  // Outer batch dimension

    // Fast path: softmax of a single element is always 1.0
    // (exp(x) / exp(x) == 1 for any x, regardless of scale/mask/bias)
    // Skip all ALiBi, mask, and sink setup.
    //
    // Each output element is 4 bytes.  A cache line is 64 bytes = 16 floats.
    // L1 is not coherent across harts, so each thread must own whole cache
    // lines to avoid cross-hart conflicts.
    ਜੇ (ne00 == 1) {
        ਅਚਲ int64_t ਕੁੱਲ_elems  = ne01 * ne02 * ne03;
        ਅਚਲ int64_t elems_ਪ੍ਰਤੀ_cl = ET_CACHE_ਸਤਰ_ਮਾਪ_ਬਾਈਟਾਂ / (int64_t) ਆਕਾਰ(ਦਸ਼ਮਲਵ);  // 16
        ਅਚਲ int64_t ਕੁੱਲ_cls    = (ਕੁੱਲ_elems + elems_ਪ੍ਰਤੀ_cl - 1) / elems_ਪ੍ਰਤੀ_cl;

        ਲਈ (int64_t cl = thread_id; cl < ਕੁੱਲ_cls; cl += ਅੰਕ_ਧਾਗੇ) {
            ਅਚਲ int64_t ਸ਼ੁਰੂ = cl * elems_ਪ੍ਰਤੀ_cl;
            int64_t       ਸਿਰਾ   = ਸ਼ੁਰੂ + elems_ਪ੍ਰਤੀ_cl;
            ਜੇ (ਸਿਰਾ > ਕੁੱਲ_elems) {
                ਸਿਰਾ = ਕੁੱਲ_elems;
            }
            ਲਈ (int64_t idx = ਸ਼ੁਰੂ; idx < ਸਿਰਾ; idx++) {
                dst_ਅੰਕੜੇ[idx] = 1.0f;
            }
        }
        ਵਾਪਸ 0;
    }

    ਅਚਲ int64_t ne10 = use_ਨਕਾਬ ? src1->ne[0] : 0;  // Mask sequence length
    ਅਚਲ int64_t ne11 = use_ਨਕਾਬ ? src1->ne[1] : 0;  // Mask rows
    ਅਚਲ int64_t ne12 = use_ਨਕਾਬ ? src1->ne[2] : 0;  // Mask batch/head dimension
    ਅਚਲ int64_t ne13 = use_ਨਕਾਬ ? src1->ne[3] : 0;  // Mask outer batch dimension

    ਜੇ (use_ਨਕਾਬ) {
        // - Dimension 0: mask must equal input exactly
        // - Dimension 1: mask must be >= input (allows larger pre-allocated masks)
        // - Dimension 2: input must be divisible by mask (modulo broadcasting)
        // - Dimension 3: input must be divisible by mask (modulo broadcasting)
        ਜੇ (ne10 != ne00 ||                    // Dimension 0: exact match required
            ne11 < ne01 ||                     // Dimension 1: mask >= input
            (ne12 > 0 && ne02 % ne12 != 0) ||  // Dimension 2: input % mask == 0
            (ne13 > 0 && ne03 % ne13 != 0)) {  // Dimension 3: input % mask == 0
            ਵਾਪਸ -1;                         // Incompatible dimensions for ggml softmax broadcasting
        }
    }

    // ALiBi slope calculation - compute per attention head
    ਅਚਲ uint32_t n_ਸਿਰ      = (uint32_t) ne02;
    uint32_t       n_ਸਿਰ_log2 = 0;
    ਦਸ਼ਮਲਵ          m0          = 1.0f;
    ਦਸ਼ਮਲਵ          m1          = 1.0f;

    ਜੇ (max_bias > 0.0f) {
        // This is equivalent to: 1 << floor(log2(n_head))
        n_ਸਿਰ_log2 = 1;
        ਜਦੋਂ (n_ਸਿਰ_log2 < n_ਸਿਰ) {
            n_ਸਿਰ_log2 <<= 1;
        }
        ਜੇ (n_ਸਿਰ_log2 > n_ਸਿਰ) {
            n_ਸਿਰ_log2 >>= 1;
        }

        // Compute base slopes for ALiBi
        // m0 = 2^(-max_bias / n_head_log2)
        // m1 = 2^(-max_bias / (2 * n_head_log2))
        ਦਸ਼ਮਲਵ inv_n_ਸਿਰ_log2 = et_fdiv(1.0f, (ਦਸ਼ਮਲਵ) n_ਸਿਰ_log2);
        m0                    = et_expf(-max_bias * 0.69314718f * inv_n_ਸਿਰ_log2);  // 0.69314718 = ln(2)
        m1                    = et_expf(-max_bias * 0.69314718f * inv_n_ਸਿਰ_log2 * 0.5f);
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
    ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_i03 = ne02 * ne01;
    ਅਚਲ int64_t ਕੁੱਲ_ਪੰਕਤੀਆਂ   = ne03 * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_i03;
    ਅਚਲ int64_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_wg  = et_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_cacheline_ਸਮੂਹ(ne00, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਅਚਲ int64_t ਕੁੱਲ_wgs    = (ਕੁੱਲ_ਪੰਕਤੀਆਂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_wg - 1) / ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_wg;

    ਲਈ (int64_t wg = thread_id; wg < ਕੁੱਲ_wgs; wg += ਅੰਕ_ਧਾਗੇ) {
        ਅਚਲ int64_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = wg * ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_wg;
        int64_t       ਪੰਕਤੀ_ਸਿਰਾ   = ਪੰਕਤੀ_ਸ਼ੁਰੂ + ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_wg;
        ਜੇ (ਪੰਕਤੀ_ਸਿਰਾ > ਕੁੱਲ_ਪੰਕਤੀਆਂ) {
            ਪੰਕਤੀ_ਸਿਰਾ = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
        }

        ਲਈ (int64_t ਪੰਕਤੀ = ਪੰਕਤੀ_ਸ਼ੁਰੂ; ਪੰਕਤੀ < ਪੰਕਤੀ_ਸਿਰਾ; ਪੰਕਤੀ++) {
            ਅਚਲ int64_t i03 = ਪੰਕਤੀ / ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_i03;
            ਅਚਲ int64_t rem = ਪੰਕਤੀ % ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_i03;
            ਅਚਲ int64_t i02 = rem / ne01;
            ਅਚਲ int64_t i01 = rem % ne01;

            // Calculate ALiBi slope for this attention head
            ਦਸ਼ਮਲਵ ਢਲਾਨ = 1.0f;
            ਜੇ (max_bias > 0.0f) {
                ਅਚਲ uint32_t h = (uint32_t) i02;  // head index
                ਜੇ (h < n_ਸਿਰ_log2) {
                    ਢਲਾਨ = m0;
                    ਲਈ (uint32_t i = 0; i < h; i++) {
                        ਢਲਾਨ *= m0;
                    }
                } ਨਹੀਂਤਾਂ {
                    ਅਚਲ uint32_t ਘਾਤਾਂਕ = 2 * (h - n_ਸਿਰ_log2) + 1;
                    ਢਲਾਨ              = m1;
                    ਲਈ (uint32_t i = 1; i < ਘਾਤਾਂਕ; i++) {
                        ਢਲਾਨ *= m1;
                    }
                }
            }

            ਦਸ਼ਮਲਵ sink_ਮੁੱਲ = 0.0f;
            ਜੇ (use_sinks && sinks_ਅੰਕੜੇ) {
                sink_ਮੁੱਲ = sinks_ਅੰਕੜੇ[i02];
            }

            ਅਚਲ int64_t src_ਵਿੱਥ = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00;

            ਅਚਲ ਦਸ਼ਮਲਵ * src_ਪੰਕਤੀ  = src0_ਅੰਕੜੇ + src_ਵਿੱਥ;
            ਦਸ਼ਮਲਵ *       dst_ਪੰਕਤੀ  = dst_ਅੰਕੜੇ + src_ਵਿੱਥ;
            ਅਚਲ ਦਸ਼ਮਲਵ * ਨਕਾਬ_ਪੰਕਤੀ = ਨਿਰਾ;

            ਜੇ (use_ਨਕਾਬ && ਨਕਾਬ_ਅੰਕੜੇ) {
                ਅਚਲ int64_t ਨਕਾਬ_i03 = (ne13 > 0) ? i03 % ne13 : 0;
                ਅਚਲ int64_t ਨਕਾਬ_i02 = (ne12 > 0) ? i02 % ne12 : 0;
                ਅਚਲ int64_t ਨਕਾਬ_i01 = i01;

                ਅਚਲ int64_t ਨਕਾਬ_ਵਿੱਥ = ਨਕਾਬ_i03 * ne12 * ne11 * ne10 + ਨਕਾਬ_i02 * ne11 * ne10 + ਨਕਾਬ_i01 * ne10;

                ਨਕਾਬ_ਪੰਕਤੀ = ਨਕਾਬ_ਅੰਕੜੇ + ਨਕਾਬ_ਵਿੱਥ;
            }

            compute_softmax_ਪੰਕਤੀ(dst_ਪੰਕਤੀ, src_ਪੰਕਤੀ, ਨਕਾਬ_ਪੰਕਤੀ, (ਪੂਰਨ) ne00, scale, ਢਲਾਨ, sink_ਮੁੱਲ, use_sinks);
        }
    }

    ਵਾਪਸ 0;  // Success
}
