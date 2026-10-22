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

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "math_fp.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <assert.h>
#அந்தர்பாவ <math.h>
#அந்தர்பாவ <stdbool.h>
#அந்தர்பாவ <stdint.h>

// Softmax kernel parameters structure (from ggml-et-ops.h)
கட்டமைப்பு ggml_et_softmax_params {
    கட்டமைப்பு ggml_tensor src0;      // F32 input tensor
    கட்டமைப்பு ggml_tensor src1;      // F32 mask tensor (optional, may be zeroed if not used)
    கட்டமைப்பு ggml_tensor src2;      // F32 sinks tensor (optional, may be zeroed if not used)
    கட்டமைப்பு ggml_tensor dst;       // F32 output tensor
    தசம              scale;     // Scale factor (temperature scaling)
    தசம              max_bias;  // Max bias for ALiBi (0.0f if not used)
};

#define LOG2E_F 1.4426950408889634f

வகைவரை கட்டமைப்பு {
    தசம    max_val;
    தசம    sum_val;
    uint32_t valid_mask;
} softmax_params_t;

நிலையான உள்வரி bool softmax_lane_is_valid(தசம x) {
    திரும்பு (x == x) && (x != -INFINITY) && (x != INFINITY);
}

நிலையான உள்வரி softmax_params_t softmax_params_empty(வெறுமை) {
    softmax_params_t p;
    p.max_val    = -INFINITY;
    p.sum_val    = 0.0f;
    p.valid_mask = 0;
    திரும்பு p;
}

// chunk_transform_ps_8_branchless_mask
//
// Vector transform for 8 logits:
//
//   x = src * scale + (mask ? mask * slope : 0)
//
// Implemented branchlessly so masked and unmasked paths share the same
// instruction stream. Used by pass1 and pass2 vector loops.
நிலையான உள்வரி வெறுமை chunk_transform_ps_8_branchless_mask(தசம *       tmp8,
                                                        மாறிலி தசம * src,
                                                        மாறிலி தசம * mask,
                                                        தசம         scale,
                                                        தசம         slope) {
    unsigned long       ms;
    மாறிலி தசம         zero         = 0.0f;
    மாறிலி unsigned long mask_load_m0 = (mask != வெற்று) ? 0xFFul : 0x00ul;
    மாறிலி தசம *       mp           = (mask != வெற்று) ? mask : &zero;

    __asm__ நிலையற்ற(
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
          [p_slope] "r"(&slope), [maskm0] "r"(mask_load_m0)
        : "f0", "f1", "f10", "f11", "memory");
}

// chunk_transform_ps_8_tail
//
// Same as chunk_transform_ps_8_branchless_mask but gates loads, compute,
// and stores with a caller-supplied m0 mask so that only `count` elements
// (1-7) are touched.  Used for the last sub-8 chunk of a non-aligned row.
நிலையான உள்வரி வெறுமை chunk_transform_ps_8_tail(தசம *       tmp8,
                                             மாறிலி தசம * src,
                                             மாறிலி தசம * mask,
                                             தசம         scale,
                                             தசம         slope,
                                             unsigned long tail_m0) {
    unsigned long       ms;
    மாறிலி தசம         zero         = 0.0f;
    மாறிலி unsigned long mask_load_m0 = (mask != வெற்று) ? tail_m0 : 0x00ul;
    மாறிலி தசம *       mp           = (mask != வெற்று) ? mask : &zero;

    __asm__ நிலையற்ற(
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
          [p_slope] "r"(&slope), [maskm0] "r"(mask_load_m0), [tailm0] "r"(tail_m0)
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
நிலையான உள்வரி softmax_params_t softmax_pass1_range(மாறிலி தசம * src,
                                                   மாறிலி தசம * mask,
                                                   முழு           begin,
                                                   முழு           end,
                                                   தசம         scale,
                                                   தசம         slope) {
    __attribute__((aligned(32))) தசம lane_max[8];
    __attribute__((aligned(32))) தசம lane_sum[8];
    __attribute__((aligned(32))) தசம tmp[8];

    uint8_t valid_mask = 0;

    மாறிலி தசம one_f   = 1.0f;
    மாறிலி தசம zero_f  = 0.0f;
    மாறிலி தசம neg_inf = -INFINITY;
    மாறிலி தசம log2e   = LOG2E_F;

    unsigned long ms;

    __asm__ நிலையற்ற(
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f20, 0(%[p_ninf])    \n\t"
        "fbc.ps    f21, 0(%[p_zero])    \n\t"
        "fbc.ps    f22, 0(%[p_one])     \n\t"
        "fbc.ps    f23, 0(%[p_log2e])   \n\t"
        : [ms] "=&r"(ms)
        : [p_ninf] "r"(&neg_inf), [p_zero] "r"(&zero_f), [p_one] "r"(&one_f), [p_log2e] "r"(&log2e)
        : "f20", "f21", "f22", "f23");

    மாறிலி முழு aligned_end = begin + ((end - begin) & ~7);

    // Process full 8-element chunks
    முழு i = begin;
    க்கு (; i < aligned_end; i += 8) {
        chunk_transform_ps_8_branchless_mask(tmp, src + i, mask ? (mask + i) : வெற்று, scale, slope);

        uint8_t cur_mask = 0;
        க்கு (முழு j = 0; j < 8; ++j) {
            எனில் (softmax_lane_is_valid(tmp[j])) {
                cur_mask |= (uint8_t) (1u << j);
            }
        }

        மாறிலி uint8_t init_mask = (uint8_t) (cur_mask & ~valid_mask);
        மாறிலி uint8_t upd_mask  = (uint8_t) (cur_mask & valid_mask);

        எனில் (init_mask || upd_mask) {
            __asm__ நிலையற்ற(
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
                : [p_tmp] "r"(tmp), [initm] "r"((unsigned long) init_mask), [updm] "r"((unsigned long) upd_mask)
                : "f0", "f1", "f2", "f3", "memory");

            valid_mask |= cur_mask;
        }
    }

    // Tail chunk: m0-gated load/compute/store for remaining 1-7 elements
    எனில் (i < end) {
        மாறிலி unsigned long tail_m0 = (1ul << (end - i)) - 1;

        // Fill tmp with NaN so invalid lanes fail softmax_lane_is_valid
        க்கு (முழு j = 0; j < 8; j++) {
            tmp[j] = __builtin_nanf("");
        }

        chunk_transform_ps_8_tail(tmp, src + i, mask ? (mask + i) : வெற்று, scale, slope, tail_m0);

        uint8_t cur_mask = 0;
        க்கு (முழு j = 0; j < 8; ++j) {
            எனில் (softmax_lane_is_valid(tmp[j])) {
                cur_mask |= (uint8_t) (1u << j);
            }
        }

        மாறிலி uint8_t init_mask = (uint8_t) (cur_mask & ~valid_mask);
        மாறிலி uint8_t upd_mask  = (uint8_t) (cur_mask & valid_mask);

        எனில் (init_mask || upd_mask) {
            __asm__ நிலையற்ற(
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
                : [p_tmp] "r"(tmp), [initm] "r"((unsigned long) init_mask), [updm] "r"((unsigned long) upd_mask)
                : "f0", "f1", "f2", "f3", "memory");

            valid_mask |= cur_mask;
        }
    }

    __asm__ நிலையற்ற(
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fsw.ps    f20, 0(%[p_lmax])    \n\t"
        "fsw.ps    f21, 0(%[p_lsum])    \n\t"
        "mova.m.x  %[ms]                \n\t"
        :
        : [p_lmax] "r"(lane_max), [p_lsum] "r"(lane_sum), [ms] "r"(ms)
        : "memory");

    softmax_params_t out = softmax_params_empty();
    out.valid_mask       = valid_mask;

    க்கு (முழு k = 0; k < 8; ++k) {
        எனில் (valid_mask & (1u << k)) {
            எனில் (out.valid_mask == (1u << k) || out.max_val == -INFINITY || lane_max[k] > out.max_val) {
                out.max_val = lane_max[k];
            }
        }
    }

    எனில் (out.max_val != -INFINITY) {
        // Compute lane correction factors via fexp.ps to stay consistent
        // with the fexp.ps used inside the online softmax loop above.
        // corr[k] = exp2((lane_max[k] - out.max_val) * LOG2E) = exp(lane_max[k] - out.max_val)
        மாறிலி தசம                        neg_max_l2 = -out.max_val * LOG2E_F;
        __attribute__((aligned(32))) தசம corr[8];
        __asm__ நிலையற்ற(
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
        க்கு (முழு k = 0; k < 8; ++k) {
            எனில் (valid_mask & (1u << k)) {
                out.sum_val += lane_sum[k] * corr[k];
            }
        }
    }

    திரும்பு out;
}

// Pass 2 (normalize) over [begin, end).
//
// Computes: dst[i] = exp(x[i]*scale + mask[i]*slope - max) / sum
//
// Uses fexp.ps for the numerator; the denominator (params.sum_val) must
// already be fully computed by the caller (pass1 + any sink merge).
நிலையான உள்வரி வெறுமை softmax_pass2_range(தசம *          dst,
                                       மாறிலி தசம *    src,
                                       மாறிலி தசம *    mask,
                                       முழு              begin,
                                       முழு              end,
                                       தசம            scale,
                                       தசம            slope,
                                       softmax_params_t params) {
    மாறிலி தசம s2      = scale * LOG2E_F;
    மாறிலி தசம sl2     = slope * LOG2E_F;
    மாறிலி தசம neg_ml2 = -params.max_val * LOG2E_F;
    மாறிலி தசம inv_sum = et_fdiv(1.0f, params.sum_val);

    unsigned long ms;

    __asm__ நிலையற்ற(
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f10, 0(%[p_s2])      \n\t"
        "fbc.ps    f12, 0(%[p_nml2])    \n\t"
        "fbc.ps    f13, 0(%[p_inv])     \n\t"
        : [ms] "=&r"(ms)
        : [p_s2] "r"(&s2), [p_nml2] "r"(&neg_ml2), [p_inv] "r"(&inv_sum)
        : "f10", "f12", "f13");

    மாறிலி முழு aligned_end = begin + ((end - begin) & ~7);

    எனில் (mask != வெற்று) {
        __asm__ நிலையற்ற("fbc.ps    f11, 0(%[p_sl2]) \n\t" : : [p_sl2] "r"(&sl2) : "f11");

        க்கு (முழு c = begin; c < aligned_end; c += 8) {
            __asm__ நிலையற்ற(
                "flw.ps    f0, 0(%[sp])           \n\t"
                "flw.ps    f1, 0(%[mp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fmadd.ps  f0, f1, f11, f0        \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                :
                : [sp] "r"(src + c), [mp] "r"(mask + c), [dp] "r"(dst + c)
                : "f0", "f1", "memory");
        }

        // Tail chunk with m0 gating
        எனில் (aligned_end < end) {
            மாறிலி unsigned long tail_m0 = (1ul << (end - aligned_end)) - 1;
            __asm__ நிலையற்ற(
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
                : [sp] "r"(src + aligned_end), [mp] "r"(mask + aligned_end), [dp] "r"(dst + aligned_end),
                  [tm] "r"(tail_m0)
                : "f0", "f1", "memory");
        }
    } இல்லையெனில் {
        க்கு (முழு c = begin; c < aligned_end; c += 8) {
            __asm__ நிலையற்ற(
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
        எனில் (aligned_end < end) {
            மாறிலி unsigned long tail_m0 = (1ul << (end - aligned_end)) - 1;
            __asm__ நிலையற்ற(
                "mov.m.x   m0, %[tm], 0              \n\t"
                "flw.ps    f0, 0(%[sp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                "mov.m.x   m0, x0, 0xFF           \n\t"
                :
                : [sp] "r"(src + aligned_end), [dp] "r"(dst + aligned_end), [tm] "r"(tail_m0)
                : "f0", "memory");
        }
    }

    __asm__ நிலையற்ற("mova.m.x  %[ms] \n\t" ::[ms] "r"(ms));
}

// Single-core row path.
// pass1_range and pass2_range handle non-8-aligned cols internally via
// m0-gated tail chunks, so this function just passes cols directly.
நிலையான உள்வரி வெறுமை compute_softmax_row(தசம *       dst,
                                       மாறிலி தசம * src,
                                       மாறிலி தசம * mask,
                                       முழு           cols,
                                       தசம         scale,
                                       தசம         slope,
                                       தசம         sink_மதிப்பு,
                                       bool          use_sinks) {
    softmax_params_t params = softmax_pass1_range(src, mask, 0, cols, scale, slope);

    எனில் (use_sinks) {
        // For sinks, use fully scalar et_expf to match the reference CPU
        // backend's expf precision.  Sink tests use small arrays (ne<=32)
        // so the scalar path has negligible performance impact.
        தசம max_val = params.max_val;
        எனில் (sink_மதிப்பு > max_val) {
            max_val = sink_மதிப்பு;
        }

        // Compute sum = Σ exp(x'[i] - max) + exp(sink - max)  (scalar)
        தசம sum = 0.0f;
        க்கு (முழு i = 0; i < cols; ++i) {
            தசம x = src[i] * scale;
            எனில் (mask != வெற்று) {
                x += mask[i] * slope;
            }
            sum += et_expf(x - max_val);
        }
        sum += et_expf(sink_மதிப்பு - max_val);

        // Normalize: dst[i] = exp(x'[i] - max) / sum  (scalar)
        தசம inv_sum = et_fdiv(1.0f, sum);
        க்கு (முழு i = 0; i < cols; ++i) {
            தசம x = src[i] * scale;
            எனில் (mask != வெற்று) {
                x += mask[i] * slope;
            }
            dst[i] = et_expf(x - max_val) * inv_sum;
        }
    } இல்லையெனில் {
        எனில் (!params.valid_mask) {
            திரும்பு;
        }
        softmax_pass2_range(dst, src, mask, 0, cols, scale, slope, params);
    }
}

// Main entry point for Softmax kernel
முழு entry_point(கட்டமைப்பு ggml_et_softmax_params * params, வெறுமை * env) {
    // Cast env to proper type
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    // Validate environment pointer
    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    // Get thread info using shire mask from environment
    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    // Return early if this hart is not active
    எனில் (thread_id < 0) {
        திரும்பு 0;
    }

    // Basic safety check on params
    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;  // Invalid pointer
    }

    // Extract tensor references
    கட்டமைப்பு ggml_tensor * src0     = &params->src0;     // Input tensor
    கட்டமைப்பு ggml_tensor * src1     = &params->src1;     // Mask tensor (optional)
    கட்டமைப்பு ggml_tensor * src2     = &params->src2;     // Sinks tensor (optional)
    கட்டமைப்பு ggml_tensor * dst      = &params->dst;      // Output tensor
    தசம                scale    = params->scale;     // Scale factor
    தசம                max_bias = params->max_bias;  // ALiBi max bias

    // Validate tensor types (F32 only)
    எனில் (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        திரும்பு -1;  // Unsupported type combination
    }

    // Check if mask is used and validate type
    bool use_mask = (src1->data != வெற்று && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16));

    bool use_sinks = (src2->data != வெற்று && src2->type == GGML_TYPE_F32);

    தசம * src0_data  = (தசம *) src0->data;
    தசம * dst_data   = (தசம *) dst->data;
    தசம * mask_data  = use_mask ? (தசம *) src1->data : வெற்று;
    தசம * sinks_data = use_sinks ? (தசம *) src2->data : வெற்று;

    எனில் (!src0_data || !dst_data) {
        திரும்பு -1;  // Null data pointer
    }


    மாறிலி int64_t ne00 = src0->ne[0];  // Sequence length (columns)
    மாறிலி int64_t ne01 = src0->ne[1];  // Number of rows
    மாறிலி int64_t ne02 = src0->ne[2];  // Batch/head dimension
    மாறிலி int64_t ne03 = src0->ne[3];  // Outer batch dimension

    // Fast path: softmax of a single element is always 1.0
    // (exp(x) / exp(x) == 1 for any x, regardless of scale/mask/bias)
    // Skip all ALiBi, mask, and sink setup.
    //
    // Each output element is 4 bytes.  A cache line is 64 bytes = 16 floats.
    // L1 is not coherent across harts, so each thread must own whole cache
    // lines to avoid cross-hart conflicts.
    எனில் (ne00 == 1) {
        மாறிலி int64_t total_elems  = ne01 * ne02 * ne03;
        மாறிலி int64_t elems_per_cl = ET_CACHE_வரி_SIZE_BYTES / (int64_t) அளவு(தசம);  // 16
        மாறிலி int64_t total_cls    = (total_elems + elems_per_cl - 1) / elems_per_cl;

        க்கு (int64_t cl = thread_id; cl < total_cls; cl += இலக்கம்_threads) {
            மாறிலி int64_t start = cl * elems_per_cl;
            int64_t       end   = start + elems_per_cl;
            எனில் (end > total_elems) {
                end = total_elems;
            }
            க்கு (int64_t idx = start; idx < end; idx++) {
                dst_data[idx] = 1.0f;
            }
        }
        திரும்பு 0;
    }

    மாறிலி int64_t ne10 = use_mask ? src1->ne[0] : 0;  // Mask sequence length
    மாறிலி int64_t ne11 = use_mask ? src1->ne[1] : 0;  // Mask rows
    மாறிலி int64_t ne12 = use_mask ? src1->ne[2] : 0;  // Mask batch/head dimension
    மாறிலி int64_t ne13 = use_mask ? src1->ne[3] : 0;  // Mask outer batch dimension

    எனில் (use_mask) {
        // - Dimension 0: mask must equal input exactly
        // - Dimension 1: mask must be >= input (allows larger pre-allocated masks)
        // - Dimension 2: input must be divisible by mask (modulo broadcasting)
        // - Dimension 3: input must be divisible by mask (modulo broadcasting)
        எனில் (ne10 != ne00 ||                    // Dimension 0: exact match required
            ne11 < ne01 ||                     // Dimension 1: mask >= input
            (ne12 > 0 && ne02 % ne12 != 0) ||  // Dimension 2: input % mask == 0
            (ne13 > 0 && ne03 % ne13 != 0)) {  // Dimension 3: input % mask == 0
            திரும்பு -1;                         // Incompatible dimensions for ggml softmax broadcasting
        }
    }

    // ALiBi slope calculation - compute per attention head
    மாறிலி uint32_t n_head      = (uint32_t) ne02;
    uint32_t       n_head_log2 = 0;
    தசம          m0          = 1.0f;
    தசம          m1          = 1.0f;

    எனில் (max_bias > 0.0f) {
        // This is equivalent to: 1 << floor(log2(n_head))
        n_head_log2 = 1;
        போது (n_head_log2 < n_head) {
            n_head_log2 <<= 1;
        }
        எனில் (n_head_log2 > n_head) {
            n_head_log2 >>= 1;
        }

        // Compute base slopes for ALiBi
        // m0 = 2^(-max_bias / n_head_log2)
        // m1 = 2^(-max_bias / (2 * n_head_log2))
        தசம inv_n_head_log2 = et_fdiv(1.0f, (தசம) n_head_log2);
        m0                    = et_expf(-max_bias * 0.69314718f * inv_n_head_log2);  // 0.69314718 = ln(2)
        m1                    = et_expf(-max_bias * 0.69314718f * inv_n_head_log2 * 0.5f);
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
    மாறிலி int64_t rows_per_i03 = ne02 * ne01;
    மாறிலி int64_t total_rows   = ne03 * rows_per_i03;
    மாறிலி int64_t rows_per_wg  = et_rows_per_cacheline_group(ne00, அளவு(தசம));
    மாறிலி int64_t total_wgs    = (total_rows + rows_per_wg - 1) / rows_per_wg;

    க்கு (int64_t wg = thread_id; wg < total_wgs; wg += இலக்கம்_threads) {
        மாறிலி int64_t row_start = wg * rows_per_wg;
        int64_t       row_end   = row_start + rows_per_wg;
        எனில் (row_end > total_rows) {
            row_end = total_rows;
        }

        க்கு (int64_t row = row_start; row < row_end; row++) {
            மாறிலி int64_t i03 = row / rows_per_i03;
            மாறிலி int64_t rem = row % rows_per_i03;
            மாறிலி int64_t i02 = rem / ne01;
            மாறிலி int64_t i01 = rem % ne01;

            // Calculate ALiBi slope for this attention head
            தசம slope = 1.0f;
            எனில் (max_bias > 0.0f) {
                மாறிலி uint32_t h = (uint32_t) i02;  // head index
                எனில் (h < n_head_log2) {
                    slope = m0;
                    க்கு (uint32_t i = 0; i < h; i++) {
                        slope *= m0;
                    }
                } இல்லையெனில் {
                    மாறிலி uint32_t exp = 2 * (h - n_head_log2) + 1;
                    slope              = m1;
                    க்கு (uint32_t i = 1; i < exp; i++) {
                        slope *= m1;
                    }
                }
            }

            தசம sink_மதிப்பு = 0.0f;
            எனில் (use_sinks && sinks_data) {
                sink_மதிப்பு = sinks_data[i02];
            }

            மாறிலி int64_t src_offset = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00;

            மாறிலி தசம * src_row  = src0_data + src_offset;
            தசம *       dst_row  = dst_data + src_offset;
            மாறிலி தசம * mask_row = வெற்று;

            எனில் (use_mask && mask_data) {
                மாறிலி int64_t mask_i03 = (ne13 > 0) ? i03 % ne13 : 0;
                மாறிலி int64_t mask_i02 = (ne12 > 0) ? i02 % ne12 : 0;
                மாறிலி int64_t mask_i01 = i01;

                மாறிலி int64_t mask_offset = mask_i03 * ne12 * ne11 * ne10 + mask_i02 * ne11 * ne10 + mask_i01 * ne10;

                mask_row = mask_data + mask_offset;
            }

            compute_softmax_row(dst_row, src_row, mask_row, (முழு) ne00, scale, slope, sink_மதிப்பு, use_sinks);
        }
    }

    திரும்பு 0;  // Success
}
