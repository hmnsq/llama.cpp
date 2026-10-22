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

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <assert.h>
#ضمن <math.h>
#ضمن <stdbool.h>
#ضمن <stdint.h>

// Softmax kernel parameters structure (from ggml-et-ops.h)
بنية ggml_et_softmax_params {
    بنية ggml_tensor src0;      // F32 input tensor
    بنية ggml_tensor src1;      // F32 mask tensor (optional, may be zeroed if not used)
    بنية ggml_tensor src2;      // F32 sinks tensor (optional, may be zeroed if not used)
    بنية ggml_tensor dst;       // F32 output tensor
    عائم              scale;     // Scale factor (temperature scaling)
    عائم              max_bias;  // Max bias for ALiBi (0.0f if not used)
};

#تعریف_کن LOG2E_F 1.4426950408889634f

تعريفنوع بنية {
    عائم    max_val;
    عائم    جمع_val;
    uint32_t valid_ماسک;
} softmax_params_t;

ساكن مضمن bool softmax_lane_is_valid(عائم x) {
    أرجع (x == x) && (x != -INFINITY) && (x != INFINITY);
}

ساكن مضمن softmax_params_t softmax_params_خالی(فراغ) {
    softmax_params_t p;
    p.max_val    = -INFINITY;
    p.جمع_val    = 0.0f;
    p.valid_ماسک = 0;
    أرجع p;
}

// chunk_transform_ps_8_branchless_mask
//
// Vector transform for 8 logits:
//
//   x = src * scale + (mask ? mask * slope : 0)
//
// Implemented branchlessly so masked and unmasked paths share the same
// instruction stream. Used by pass1 and pass2 vector loops.
ساكن مضمن فراغ chunk_transform_ps_8_branchless_ماسک(عائم *       tmp8,
                                                        ثابت عائم * src,
                                                        ثابت عائم * ماسک,
                                                        عائم         scale,
                                                        عائم         شیب) {
    بیعلامت دراز       ms;
    ثابت عائم         zero         = 0.0f;
    ثابت بیعلامت دراز ماسک_بارگیری_m0 = (ماسک != عدم) ? 0xFFul : 0x00ul;
    ثابت عائم *       mp           = (ماسک != عدم) ? ماسک : &zero;

    __asm__ متطاير(
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
          [p_شیب] "r"(&شیب), [maskm0] "r"(ماسک_بارگیری_m0)
        : "f0", "f1", "f10", "f11", "memory");
}

// chunk_transform_ps_8_tail
//
// Same as chunk_transform_ps_8_branchless_mask but gates loads, compute,
// and stores with a caller-supplied m0 mask so that only `count` elements
// (1-7) are touched.  Used for the last sub-8 chunk of a non-aligned row.
ساكن مضمن فراغ chunk_transform_ps_8_دم(عائم *       tmp8,
                                             ثابت عائم * src,
                                             ثابت عائم * ماسک,
                                             عائم         scale,
                                             عائم         شیب,
                                             بیعلامت دراز دم_m0) {
    بیعلامت دراز       ms;
    ثابت عائم         zero         = 0.0f;
    ثابت بیعلامت دراز ماسک_بارگیری_m0 = (ماسک != عدم) ? دم_m0 : 0x00ul;
    ثابت عائم *       mp           = (ماسک != عدم) ? ماسک : &zero;

    __asm__ متطاير(
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
          [p_شیب] "r"(&شیب), [maskm0] "r"(ماسک_بارگیری_m0), [tailm0] "r"(دم_m0)
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
ساكن مضمن softmax_params_t softmax_pass1_range(ثابت عائم * src,
                                                   ثابت عائم * ماسک,
                                                   صحيح           begin,
                                                   صحيح           انتها,
                                                   عائم         scale,
                                                   عائم         شیب) {
    __attribute__((aligned(32))) عائم lane_max[8];
    __attribute__((aligned(32))) عائم lane_جمع[8];
    __attribute__((aligned(32))) عائم tmp[8];

    uint8_t valid_ماسک = 0;

    ثابت عائم one_f   = 1.0f;
    ثابت عائم zero_f  = 0.0f;
    ثابت عائم neg_inf = -INFINITY;
    ثابت عائم log2e   = LOG2E_F;

    بیعلامت دراز ms;

    __asm__ متطاير(
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f20, 0(%[p_ninf])    \n\t"
        "fbc.ps    f21, 0(%[p_zero])    \n\t"
        "fbc.ps    f22, 0(%[p_one])     \n\t"
        "fbc.ps    f23, 0(%[p_log2e])   \n\t"
        : [ms] "=&r"(ms)
        : [p_ninf] "r"(&neg_inf), [p_zero] "r"(&zero_f), [p_one] "r"(&one_f), [p_log2e] "r"(&log2e)
        : "f20", "f21", "f22", "f23");

    ثابت صحيح aligned_انتها = begin + ((انتها - begin) & ~7);

    // Process full 8-element chunks
    صحيح i = begin;
    لكل (; i < aligned_انتها; i += 8) {
        chunk_transform_ps_8_branchless_ماسک(tmp, src + i, ماسک ? (ماسک + i) : عدم, scale, شیب);

        uint8_t cur_ماسک = 0;
        لكل (صحيح j = 0; j < 8; ++j) {
            إذا (softmax_lane_is_valid(tmp[j])) {
                cur_ماسک |= (uint8_t) (1u << j);
            }
        }

        ثابت uint8_t init_ماسک = (uint8_t) (cur_ماسک & ~valid_ماسک);
        ثابت uint8_t upd_ماسک  = (uint8_t) (cur_ماسک & valid_ماسک);

        إذا (init_ماسک || upd_ماسک) {
            __asm__ متطاير(
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
                : [p_tmp] "r"(tmp), [initm] "r"((بیعلامت دراز) init_ماسک), [updm] "r"((بیعلامت دراز) upd_ماسک)
                : "f0", "f1", "f2", "f3", "memory");

            valid_ماسک |= cur_ماسک;
        }
    }

    // Tail chunk: m0-gated load/compute/store for remaining 1-7 elements
    إذا (i < انتها) {
        ثابت بیعلامت دراز دم_m0 = (1ul << (انتها - i)) - 1;

        // Fill tmp with NaN so invalid lanes fail softmax_lane_is_valid
        لكل (صحيح j = 0; j < 8; j++) {
            tmp[j] = __builtin_nanf("");
        }

        chunk_transform_ps_8_دم(tmp, src + i, ماسک ? (ماسک + i) : عدم, scale, شیب, دم_m0);

        uint8_t cur_ماسک = 0;
        لكل (صحيح j = 0; j < 8; ++j) {
            إذا (softmax_lane_is_valid(tmp[j])) {
                cur_ماسک |= (uint8_t) (1u << j);
            }
        }

        ثابت uint8_t init_ماسک = (uint8_t) (cur_ماسک & ~valid_ماسک);
        ثابت uint8_t upd_ماسک  = (uint8_t) (cur_ماسک & valid_ماسک);

        إذا (init_ماسک || upd_ماسک) {
            __asm__ متطاير(
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
                : [p_tmp] "r"(tmp), [initm] "r"((بیعلامت دراز) init_ماسک), [updm] "r"((بیعلامت دراز) upd_ماسک)
                : "f0", "f1", "f2", "f3", "memory");

            valid_ماسک |= cur_ماسک;
        }
    }

    __asm__ متطاير(
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fsw.ps    f20, 0(%[p_lmax])    \n\t"
        "fsw.ps    f21, 0(%[p_lsum])    \n\t"
        "mova.m.x  %[ms]                \n\t"
        :
        : [p_lmax] "r"(lane_max), [p_lsum] "r"(lane_جمع), [ms] "r"(ms)
        : "memory");

    softmax_params_t بیرون = softmax_params_خالی();
    بیرون.valid_ماسک       = valid_ماسک;

    لكل (صحيح k = 0; k < 8; ++k) {
        إذا (valid_ماسک & (1u << k)) {
            إذا (بیرون.valid_ماسک == (1u << k) || بیرون.max_val == -INFINITY || lane_max[k] > بیرون.max_val) {
                بیرون.max_val = lane_max[k];
            }
        }
    }

    إذا (بیرون.max_val != -INFINITY) {
        // Compute lane correction factors via fexp.ps to stay consistent
        // with the fexp.ps used inside the online softmax loop above.
        // corr[k] = exp2((lane_max[k] - out.max_val) * LOG2E) = exp(lane_max[k] - out.max_val)
        ثابت عائم                        neg_max_l2 = -بیرون.max_val * LOG2E_F;
        __attribute__((aligned(32))) عائم corr[8];
        __asm__ متطاير(
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
        لكل (صحيح k = 0; k < 8; ++k) {
            إذا (valid_ماسک & (1u << k)) {
                بیرون.جمع_val += lane_جمع[k] * corr[k];
            }
        }
    }

    أرجع بیرون;
}

// Pass 2 (normalize) over [begin, end).
//
// Computes: dst[i] = exp(x[i]*scale + mask[i]*slope - max) / sum
//
// Uses fexp.ps for the numerator; the denominator (params.sum_val) must
// already be fully computed by the caller (pass1 + any sink merge).
ساكن مضمن فراغ softmax_pass2_range(عائم *          dst,
                                       ثابت عائم *    src,
                                       ثابت عائم *    ماسک,
                                       صحيح              begin,
                                       صحيح              انتها,
                                       عائم            scale,
                                       عائم            شیب,
                                       softmax_params_t params) {
    ثابت عائم s2      = scale * LOG2E_F;
    ثابت عائم sl2     = شیب * LOG2E_F;
    ثابت عائم neg_ml2 = -params.max_val * LOG2E_F;
    ثابت عائم inv_جمع = et_fdiv(1.0f, params.جمع_val);

    بیعلامت دراز ms;

    __asm__ متطاير(
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f10, 0(%[p_s2])      \n\t"
        "fbc.ps    f12, 0(%[p_nml2])    \n\t"
        "fbc.ps    f13, 0(%[p_inv])     \n\t"
        : [ms] "=&r"(ms)
        : [p_s2] "r"(&s2), [p_nml2] "r"(&neg_ml2), [p_inv] "r"(&inv_جمع)
        : "f10", "f12", "f13");

    ثابت صحيح aligned_انتها = begin + ((انتها - begin) & ~7);

    إذا (ماسک != عدم) {
        __asm__ متطاير("fbc.ps    f11, 0(%[p_sl2]) \n\t" : : [p_sl2] "r"(&sl2) : "f11");

        لكل (صحيح c = begin; c < aligned_انتها; c += 8) {
            __asm__ متطاير(
                "flw.ps    f0, 0(%[sp])           \n\t"
                "flw.ps    f1, 0(%[mp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fmadd.ps  f0, f1, f11, f0        \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                :
                : [sp] "r"(src + c), [mp] "r"(ماسک + c), [dp] "r"(dst + c)
                : "f0", "f1", "memory");
        }

        // Tail chunk with m0 gating
        إذا (aligned_انتها < انتها) {
            ثابت بیعلامت دراز دم_m0 = (1ul << (انتها - aligned_انتها)) - 1;
            __asm__ متطاير(
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
                : [sp] "r"(src + aligned_انتها), [mp] "r"(ماسک + aligned_انتها), [dp] "r"(dst + aligned_انتها),
                  [tm] "r"(دم_m0)
                : "f0", "f1", "memory");
        }
    } وإلا {
        لكل (صحيح c = begin; c < aligned_انتها; c += 8) {
            __asm__ متطاير(
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
        إذا (aligned_انتها < انتها) {
            ثابت بیعلامت دراز دم_m0 = (1ul << (انتها - aligned_انتها)) - 1;
            __asm__ متطاير(
                "mov.m.x   m0, %[tm], 0              \n\t"
                "flw.ps    f0, 0(%[sp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                "mov.m.x   m0, x0, 0xFF           \n\t"
                :
                : [sp] "r"(src + aligned_انتها), [dp] "r"(dst + aligned_انتها), [tm] "r"(دم_m0)
                : "f0", "memory");
        }
    }

    __asm__ متطاير("mova.m.x  %[ms] \n\t" ::[ms] "r"(ms));
}

// Single-core row path.
// pass1_range and pass2_range handle non-8-aligned cols internally via
// m0-gated tail chunks, so this function just passes cols directly.
ساكن مضمن فراغ compute_softmax_ردیف(عائم *       dst,
                                       ثابت عائم * src,
                                       ثابت عائم * ماسک,
                                       صحيح           تعداد_ستون_2,
                                       عائم         scale,
                                       عائم         شیب,
                                       عائم         sink_مقدار,
                                       bool          use_sinks) {
    softmax_params_t params = softmax_pass1_range(src, ماسک, 0, تعداد_ستون_2, scale, شیب);

    إذا (use_sinks) {
        // For sinks, use fully scalar et_expf to match the reference CPU
        // backend's expf precision.  Sink tests use small arrays (ne<=32)
        // so the scalar path has negligible performance impact.
        عائم max_val = params.max_val;
        إذا (sink_مقدار > max_val) {
            max_val = sink_مقدار;
        }

        // Compute sum = Σ exp(x'[i] - max) + exp(sink - max)  (scalar)
        عائم جمع = 0.0f;
        لكل (صحيح i = 0; i < تعداد_ستون_2; ++i) {
            عائم x = src[i] * scale;
            إذا (ماسک != عدم) {
                x += ماسک[i] * شیب;
            }
            جمع += et_expf(x - max_val);
        }
        جمع += et_expf(sink_مقدار - max_val);

        // Normalize: dst[i] = exp(x'[i] - max) / sum  (scalar)
        عائم inv_جمع = et_fdiv(1.0f, جمع);
        لكل (صحيح i = 0; i < تعداد_ستون_2; ++i) {
            عائم x = src[i] * scale;
            إذا (ماسک != عدم) {
                x += ماسک[i] * شیب;
            }
            dst[i] = et_expf(x - max_val) * inv_جمع;
        }
    } وإلا {
        إذا (!params.valid_ماسک) {
            أرجع;
        }
        softmax_pass2_range(dst, src, ماسک, 0, تعداد_ستون_2, scale, شیب, params);
    }
}

// Main entry point for Softmax kernel
صحيح مدخل_نقطه(بنية ggml_et_softmax_params * params, فراغ * env) {
    // Cast env to proper type
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    // Validate environment pointer
    إذا (!kernel_env) {
        أرجع -1;
    }

    // Get thread info using shire mask from environment
    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_ماسک);
    صحيح رقم_نخها = get_رقم_نخها(kernel_env->shire_ماسک);

    // Return early if this hart is not active
    إذا (thread_id < 0) {
        أرجع 0;
    }

    // Basic safety check on params
    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;  // Invalid pointer
    }

    // Extract tensor references
    بنية ggml_tensor * src0     = &params->src0;     // Input tensor
    بنية ggml_tensor * src1     = &params->src1;     // Mask tensor (optional)
    بنية ggml_tensor * src2     = &params->src2;     // Sinks tensor (optional)
    بنية ggml_tensor * dst      = &params->dst;      // Output tensor
    عائم                scale    = params->scale;     // Scale factor
    عائم                max_bias = params->max_bias;  // ALiBi max bias

    // Validate tensor types (F32 only)
    إذا (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;  // Unsupported type combination
    }

    // Check if mask is used and validate type
    bool use_ماسک = (src1->داده != عدم && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16));

    bool use_sinks = (src2->داده != عدم && src2->type == GGML_TYPE_F32);

    عائم * src0_داده  = (عائم *) src0->داده;
    عائم * dst_داده   = (عائم *) dst->داده;
    عائم * ماسک_داده  = use_ماسک ? (عائم *) src1->داده : عدم;
    عائم * sinks_داده = use_sinks ? (عائم *) src2->داده : عدم;

    إذا (!src0_داده || !dst_داده) {
        أرجع -1;  // Null data pointer
    }


    ثابت int64_t ne00 = src0->ne[0];  // Sequence length (columns)
    ثابت int64_t ne01 = src0->ne[1];  // Number of rows
    ثابت int64_t ne02 = src0->ne[2];  // Batch/head dimension
    ثابت int64_t ne03 = src0->ne[3];  // Outer batch dimension

    // Fast path: softmax of a single element is always 1.0
    // (exp(x) / exp(x) == 1 for any x, regardless of scale/mask/bias)
    // Skip all ALiBi, mask, and sink setup.
    //
    // Each output element is 4 bytes.  A cache line is 64 bytes = 16 floats.
    // L1 is not coherent across harts, so each thread must own whole cache
    // lines to avoid cross-hart conflicts.
    إذا (ne00 == 1) {
        ثابت int64_t مجموع_elems  = ne01 * ne02 * ne03;
        ثابت int64_t elems_در_هر_cl = ET_CACHE_سطر_اندازه_بایتها / (int64_t) حجم(عائم);  // 16
        ثابت int64_t مجموع_cls    = (مجموع_elems + elems_در_هر_cl - 1) / elems_در_هر_cl;

        لكل (int64_t cl = thread_id; cl < مجموع_cls; cl += رقم_نخها) {
            ثابت int64_t شروع = cl * elems_در_هر_cl;
            int64_t       انتها   = شروع + elems_در_هر_cl;
            إذا (انتها > مجموع_elems) {
                انتها = مجموع_elems;
            }
            لكل (int64_t idx = شروع; idx < انتها; idx++) {
                dst_داده[idx] = 1.0f;
            }
        }
        أرجع 0;
    }

    ثابت int64_t ne10 = use_ماسک ? src1->ne[0] : 0;  // Mask sequence length
    ثابت int64_t ne11 = use_ماسک ? src1->ne[1] : 0;  // Mask rows
    ثابت int64_t ne12 = use_ماسک ? src1->ne[2] : 0;  // Mask batch/head dimension
    ثابت int64_t ne13 = use_ماسک ? src1->ne[3] : 0;  // Mask outer batch dimension

    إذا (use_ماسک) {
        // - Dimension 0: mask must equal input exactly
        // - Dimension 1: mask must be >= input (allows larger pre-allocated masks)
        // - Dimension 2: input must be divisible by mask (modulo broadcasting)
        // - Dimension 3: input must be divisible by mask (modulo broadcasting)
        إذا (ne10 != ne00 ||                    // Dimension 0: exact match required
            ne11 < ne01 ||                     // Dimension 1: mask >= input
            (ne12 > 0 && ne02 % ne12 != 0) ||  // Dimension 2: input % mask == 0
            (ne13 > 0 && ne03 % ne13 != 0)) {  // Dimension 3: input % mask == 0
            أرجع -1;                         // Incompatible dimensions for ggml softmax broadcasting
        }
    }

    // ALiBi slope calculation - compute per attention head
    ثابت uint32_t n_سر      = (uint32_t) ne02;
    uint32_t       n_سر_log2 = 0;
    عائم          m0          = 1.0f;
    عائم          m1          = 1.0f;

    إذا (max_bias > 0.0f) {
        // This is equivalent to: 1 << floor(log2(n_head))
        n_سر_log2 = 1;
        طالما (n_سر_log2 < n_سر) {
            n_سر_log2 <<= 1;
        }
        إذا (n_سر_log2 > n_سر) {
            n_سر_log2 >>= 1;
        }

        // Compute base slopes for ALiBi
        // m0 = 2^(-max_bias / n_head_log2)
        // m1 = 2^(-max_bias / (2 * n_head_log2))
        عائم inv_n_سر_log2 = et_fdiv(1.0f, (عائم) n_سر_log2);
        m0                    = et_expf(-max_bias * 0.69314718f * inv_n_سر_log2);  // 0.69314718 = ln(2)
        m1                    = et_expf(-max_bias * 0.69314718f * inv_n_سر_log2 * 0.5f);
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
    ثابت int64_t ردیفها_در_هر_i03 = ne02 * ne01;
    ثابت int64_t مجموع_ردیفها   = ne03 * ردیفها_در_هر_i03;
    ثابت int64_t ردیفها_در_هر_wg  = et_ردیفها_در_هر_cacheline_گروه(ne00, حجم(عائم));
    ثابت int64_t مجموع_wgs    = (مجموع_ردیفها + ردیفها_در_هر_wg - 1) / ردیفها_در_هر_wg;

    لكل (int64_t wg = thread_id; wg < مجموع_wgs; wg += رقم_نخها) {
        ثابت int64_t ردیف_شروع = wg * ردیفها_در_هر_wg;
        int64_t       ردیف_انتها   = ردیف_شروع + ردیفها_در_هر_wg;
        إذا (ردیف_انتها > مجموع_ردیفها) {
            ردیف_انتها = مجموع_ردیفها;
        }

        لكل (int64_t ردیف = ردیف_شروع; ردیف < ردیف_انتها; ردیف++) {
            ثابت int64_t i03 = ردیف / ردیفها_در_هر_i03;
            ثابت int64_t rem = ردیف % ردیفها_در_هر_i03;
            ثابت int64_t i02 = rem / ne01;
            ثابت int64_t i01 = rem % ne01;

            // Calculate ALiBi slope for this attention head
            عائم شیب = 1.0f;
            إذا (max_bias > 0.0f) {
                ثابت uint32_t h = (uint32_t) i02;  // head index
                إذا (h < n_سر_log2) {
                    شیب = m0;
                    لكل (uint32_t i = 0; i < h; i++) {
                        شیب *= m0;
                    }
                } وإلا {
                    ثابت uint32_t نمایی_تابع = 2 * (h - n_سر_log2) + 1;
                    شیب              = m1;
                    لكل (uint32_t i = 1; i < نمایی_تابع; i++) {
                        شیب *= m1;
                    }
                }
            }

            عائم sink_مقدار = 0.0f;
            إذا (use_sinks && sinks_داده) {
                sink_مقدار = sinks_داده[i02];
            }

            ثابت int64_t src_جابه_جایی = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00;

            ثابت عائم * src_ردیف  = src0_داده + src_جابه_جایی;
            عائم *       dst_ردیف  = dst_داده + src_جابه_جایی;
            ثابت عائم * ماسک_ردیف = عدم;

            إذا (use_ماسک && ماسک_داده) {
                ثابت int64_t ماسک_i03 = (ne13 > 0) ? i03 % ne13 : 0;
                ثابت int64_t ماسک_i02 = (ne12 > 0) ? i02 % ne12 : 0;
                ثابت int64_t ماسک_i01 = i01;

                ثابت int64_t ماسک_جابه_جایی = ماسک_i03 * ne12 * ne11 * ne10 + ماسک_i02 * ne11 * ne10 + ماسک_i01 * ne10;

                ماسک_ردیف = ماسک_داده + ماسک_جابه_جایی;
            }

            compute_softmax_ردیف(dst_ردیف, src_ردیف, ماسک_ردیف, (صحيح) ne00, scale, شیب, sink_مقدار, use_sinks);
        }
    }

    أرجع 0;  // Success
}
