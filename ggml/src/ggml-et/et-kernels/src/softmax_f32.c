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

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <assert.h>
#شامل <math.h>
#شامل <stdbool.h>
#شامل <stdint.h>

// Softmax kernel parameters structure (from ggml-et-ops.h)
ساخت ggml_et_softmax_params {
    ساخت ggml_tensor src0;      // F32 input tensor
    ساخت ggml_tensor src1;      // F32 mask tensor (optional, may be zeroed if not used)
    ساخت ggml_tensor src2;      // F32 sinks tensor (optional, may be zeroed if not used)
    ساخت ggml_tensor dst;       // F32 output tensor
    اعشاری              scale;     // Scale factor (temperature scaling)
    اعشاری              max_bias;  // Max bias for ALiBi (0.0f if not used)
};

#تعریف_کرو LOG2E_F 1.4426950408889634f

قسمتعریف ساخت {
    اعشاری    max_val;
    اعشاری    مجموع_val;
    uint32_t valid_نقاب;
} softmax_params_t;

ساکن اندرونی bool softmax_lane_is_valid(اعشاری x) {
    واپس (x == x) && (x != -INFINITY) && (x != INFINITY);
}

ساکن اندرونی softmax_params_t softmax_params_خالی_خانے(خالی) {
    softmax_params_t p;
    p.max_val    = -INFINITY;
    p.مجموع_val    = 0.0f;
    p.valid_نقاب = 0;
    واپس p;
}

// chunk_transform_ps_8_branchless_mask
//
// Vector transform for 8 logits:
//
//   x = src * scale + (mask ? mask * slope : 0)
//
// Implemented branchlessly so masked and unmasked paths share the same
// instruction stream. Used by pass1 and pass2 vector loops.
ساکن اندرونی خالی chunk_transform_ps_8_branchless_نقاب(اعشاری *       tmp8,
                                                        ثابت اعشاری * src,
                                                        ثابت اعشاری * نقاب,
                                                        اعشاری         scale,
                                                        اعشاری         ڈھلان) {
    غیرعلامتی طویل       ms;
    ثابت اعشاری         zero         = 0.0f;
    ثابت غیرعلامتی طویل نقاب_بھرو_m0 = (نقاب != تہی) ? 0xFFul : 0x00ul;
    ثابت اعشاری *       mp           = (نقاب != تہی) ? نقاب : &zero;

    __asm__ متلون(
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
          [p_ڈھلان] "r"(&ڈھلان), [maskm0] "r"(نقاب_بھرو_m0)
        : "f0", "f1", "f10", "f11", "memory");
}

// chunk_transform_ps_8_tail
//
// Same as chunk_transform_ps_8_branchless_mask but gates loads, compute,
// and stores with a caller-supplied m0 mask so that only `count` elements
// (1-7) are touched.  Used for the last sub-8 chunk of a non-aligned row.
ساکن اندرونی خالی chunk_transform_ps_8_دم(اعشاری *       tmp8,
                                             ثابت اعشاری * src,
                                             ثابت اعشاری * نقاب,
                                             اعشاری         scale,
                                             اعشاری         ڈھلان,
                                             غیرعلامتی طویل دم_m0) {
    غیرعلامتی طویل       ms;
    ثابت اعشاری         zero         = 0.0f;
    ثابت غیرعلامتی طویل نقاب_بھرو_m0 = (نقاب != تہی) ? دم_m0 : 0x00ul;
    ثابت اعشاری *       mp           = (نقاب != تہی) ? نقاب : &zero;

    __asm__ متلون(
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
          [p_ڈھلان] "r"(&ڈھلان), [maskm0] "r"(نقاب_بھرو_m0), [tailm0] "r"(دم_m0)
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
ساکن اندرونی softmax_params_t softmax_pass1_range(ثابت اعشاری * src,
                                                   ثابت اعشاری * نقاب,
                                                   صحیح           begin,
                                                   صحیح           انتہا,
                                                   اعشاری         scale,
                                                   اعشاری         ڈھلان) {
    __attribute__((aligned(32))) اعشاری lane_max[8];
    __attribute__((aligned(32))) اعشاری lane_مجموع[8];
    __attribute__((aligned(32))) اعشاری tmp[8];

    uint8_t valid_نقاب = 0;

    ثابت اعشاری one_f   = 1.0f;
    ثابت اعشاری zero_f  = 0.0f;
    ثابت اعشاری neg_inf = -INFINITY;
    ثابت اعشاری log2e   = LOG2E_F;

    غیرعلامتی طویل ms;

    __asm__ متلون(
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f20, 0(%[p_ninf])    \n\t"
        "fbc.ps    f21, 0(%[p_zero])    \n\t"
        "fbc.ps    f22, 0(%[p_one])     \n\t"
        "fbc.ps    f23, 0(%[p_log2e])   \n\t"
        : [ms] "=&r"(ms)
        : [p_ninf] "r"(&neg_inf), [p_zero] "r"(&zero_f), [p_one] "r"(&one_f), [p_log2e] "r"(&log2e)
        : "f20", "f21", "f22", "f23");

    ثابت صحیح aligned_انتہا = begin + ((انتہا - begin) & ~7);

    // Process full 8-element chunks
    صحیح i = begin;
    برائے (; i < aligned_انتہا; i += 8) {
        chunk_transform_ps_8_branchless_نقاب(tmp, src + i, نقاب ? (نقاب + i) : تہی, scale, ڈھلان);

        uint8_t cur_نقاب = 0;
        برائے (صحیح j = 0; j < 8; ++j) {
            اگر (softmax_lane_is_valid(tmp[j])) {
                cur_نقاب |= (uint8_t) (1u << j);
            }
        }

        ثابت uint8_t init_نقاب = (uint8_t) (cur_نقاب & ~valid_نقاب);
        ثابت uint8_t upd_نقاب  = (uint8_t) (cur_نقاب & valid_نقاب);

        اگر (init_نقاب || upd_نقاب) {
            __asm__ متلون(
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
                : [p_tmp] "r"(tmp), [initm] "r"((غیرعلامتی طویل) init_نقاب), [updm] "r"((غیرعلامتی طویل) upd_نقاب)
                : "f0", "f1", "f2", "f3", "memory");

            valid_نقاب |= cur_نقاب;
        }
    }

    // Tail chunk: m0-gated load/compute/store for remaining 1-7 elements
    اگر (i < انتہا) {
        ثابت غیرعلامتی طویل دم_m0 = (1ul << (انتہا - i)) - 1;

        // Fill tmp with NaN so invalid lanes fail softmax_lane_is_valid
        برائے (صحیح j = 0; j < 8; j++) {
            tmp[j] = __builtin_nanf("");
        }

        chunk_transform_ps_8_دم(tmp, src + i, نقاب ? (نقاب + i) : تہی, scale, ڈھلان, دم_m0);

        uint8_t cur_نقاب = 0;
        برائے (صحیح j = 0; j < 8; ++j) {
            اگر (softmax_lane_is_valid(tmp[j])) {
                cur_نقاب |= (uint8_t) (1u << j);
            }
        }

        ثابت uint8_t init_نقاب = (uint8_t) (cur_نقاب & ~valid_نقاب);
        ثابت uint8_t upd_نقاب  = (uint8_t) (cur_نقاب & valid_نقاب);

        اگر (init_نقاب || upd_نقاب) {
            __asm__ متلون(
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
                : [p_tmp] "r"(tmp), [initm] "r"((غیرعلامتی طویل) init_نقاب), [updm] "r"((غیرعلامتی طویل) upd_نقاب)
                : "f0", "f1", "f2", "f3", "memory");

            valid_نقاب |= cur_نقاب;
        }
    }

    __asm__ متلون(
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fsw.ps    f20, 0(%[p_lmax])    \n\t"
        "fsw.ps    f21, 0(%[p_lsum])    \n\t"
        "mova.m.x  %[ms]                \n\t"
        :
        : [p_lmax] "r"(lane_max), [p_lsum] "r"(lane_مجموع), [ms] "r"(ms)
        : "memory");

    softmax_params_t برآمد = softmax_params_خالی_خانے();
    برآمد.valid_نقاب       = valid_نقاب;

    برائے (صحیح k = 0; k < 8; ++k) {
        اگر (valid_نقاب & (1u << k)) {
            اگر (برآمد.valid_نقاب == (1u << k) || برآمد.max_val == -INFINITY || lane_max[k] > برآمد.max_val) {
                برآمد.max_val = lane_max[k];
            }
        }
    }

    اگر (برآمد.max_val != -INFINITY) {
        // Compute lane correction factors via fexp.ps to stay consistent
        // with the fexp.ps used inside the online softmax loop above.
        // corr[k] = exp2((lane_max[k] - out.max_val) * LOG2E) = exp(lane_max[k] - out.max_val)
        ثابت اعشاری                        neg_max_l2 = -برآمد.max_val * LOG2E_F;
        __attribute__((aligned(32))) اعشاری corr[8];
        __asm__ متلون(
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
        برائے (صحیح k = 0; k < 8; ++k) {
            اگر (valid_نقاب & (1u << k)) {
                برآمد.مجموع_val += lane_مجموع[k] * corr[k];
            }
        }
    }

    واپس برآمد;
}

// Pass 2 (normalize) over [begin, end).
//
// Computes: dst[i] = exp(x[i]*scale + mask[i]*slope - max) / sum
//
// Uses fexp.ps for the numerator; the denominator (params.sum_val) must
// already be fully computed by the caller (pass1 + any sink merge).
ساکن اندرونی خالی softmax_pass2_range(اعشاری *          dst,
                                       ثابت اعشاری *    src,
                                       ثابت اعشاری *    نقاب,
                                       صحیح              begin,
                                       صحیح              انتہا,
                                       اعشاری            scale,
                                       اعشاری            ڈھلان,
                                       softmax_params_t params) {
    ثابت اعشاری s2      = scale * LOG2E_F;
    ثابت اعشاری sl2     = ڈھلان * LOG2E_F;
    ثابت اعشاری neg_ml2 = -params.max_val * LOG2E_F;
    ثابت اعشاری inv_مجموع = et_fdiv(1.0f, params.مجموع_val);

    غیرعلامتی طویل ms;

    __asm__ متلون(
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f10, 0(%[p_s2])      \n\t"
        "fbc.ps    f12, 0(%[p_nml2])    \n\t"
        "fbc.ps    f13, 0(%[p_inv])     \n\t"
        : [ms] "=&r"(ms)
        : [p_s2] "r"(&s2), [p_nml2] "r"(&neg_ml2), [p_inv] "r"(&inv_مجموع)
        : "f10", "f12", "f13");

    ثابت صحیح aligned_انتہا = begin + ((انتہا - begin) & ~7);

    اگر (نقاب != تہی) {
        __asm__ متلون("fbc.ps    f11, 0(%[p_sl2]) \n\t" : : [p_sl2] "r"(&sl2) : "f11");

        برائے (صحیح c = begin; c < aligned_انتہا; c += 8) {
            __asm__ متلون(
                "flw.ps    f0, 0(%[sp])           \n\t"
                "flw.ps    f1, 0(%[mp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fmadd.ps  f0, f1, f11, f0        \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                :
                : [sp] "r"(src + c), [mp] "r"(نقاب + c), [dp] "r"(dst + c)
                : "f0", "f1", "memory");
        }

        // Tail chunk with m0 gating
        اگر (aligned_انتہا < انتہا) {
            ثابت غیرعلامتی طویل دم_m0 = (1ul << (انتہا - aligned_انتہا)) - 1;
            __asm__ متلون(
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
                : [sp] "r"(src + aligned_انتہا), [mp] "r"(نقاب + aligned_انتہا), [dp] "r"(dst + aligned_انتہا),
                  [tm] "r"(دم_m0)
                : "f0", "f1", "memory");
        }
    } ورنہ {
        برائے (صحیح c = begin; c < aligned_انتہا; c += 8) {
            __asm__ متلون(
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
        اگر (aligned_انتہا < انتہا) {
            ثابت غیرعلامتی طویل دم_m0 = (1ul << (انتہا - aligned_انتہا)) - 1;
            __asm__ متلون(
                "mov.m.x   m0, %[tm], 0              \n\t"
                "flw.ps    f0, 0(%[sp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                "mov.m.x   m0, x0, 0xFF           \n\t"
                :
                : [sp] "r"(src + aligned_انتہا), [dp] "r"(dst + aligned_انتہا), [tm] "r"(دم_m0)
                : "f0", "memory");
        }
    }

    __asm__ متلون("mova.m.x  %[ms] \n\t" ::[ms] "r"(ms));
}

// Single-core row path.
// pass1_range and pass2_range handle non-8-aligned cols internally via
// m0-gated tail chunks, so this function just passes cols directly.
ساکن اندرونی خالی compute_softmax_صف(اعشاری *       dst,
                                       ثابت اعشاری * src,
                                       ثابت اعشاری * نقاب,
                                       صحیح           ستون_تعداد_2,
                                       اعشاری         scale,
                                       اعشاری         ڈھلان,
                                       اعشاری         sink_قدر,
                                       bool          use_sinks) {
    softmax_params_t params = softmax_pass1_range(src, نقاب, 0, ستون_تعداد_2, scale, ڈھلان);

    اگر (use_sinks) {
        // For sinks, use fully scalar et_expf to match the reference CPU
        // backend's expf precision.  Sink tests use small arrays (ne<=32)
        // so the scalar path has negligible performance impact.
        اعشاری max_val = params.max_val;
        اگر (sink_قدر > max_val) {
            max_val = sink_قدر;
        }

        // Compute sum = Σ exp(x'[i] - max) + exp(sink - max)  (scalar)
        اعشاری مجموع = 0.0f;
        برائے (صحیح i = 0; i < ستون_تعداد_2; ++i) {
            اعشاری x = src[i] * scale;
            اگر (نقاب != تہی) {
                x += نقاب[i] * ڈھلان;
            }
            مجموع += et_expf(x - max_val);
        }
        مجموع += et_expf(sink_قدر - max_val);

        // Normalize: dst[i] = exp(x'[i] - max) / sum  (scalar)
        اعشاری inv_مجموع = et_fdiv(1.0f, مجموع);
        برائے (صحیح i = 0; i < ستون_تعداد_2; ++i) {
            اعشاری x = src[i] * scale;
            اگر (نقاب != تہی) {
                x += نقاب[i] * ڈھلان;
            }
            dst[i] = et_expf(x - max_val) * inv_مجموع;
        }
    } ورنہ {
        اگر (!params.valid_نقاب) {
            واپس;
        }
        softmax_pass2_range(dst, src, نقاب, 0, ستون_تعداد_2, scale, ڈھلان, params);
    }
}

// Main entry point for Softmax kernel
صحیح اندراج_نقطہ(ساخت ggml_et_softmax_params * params, خالی * env) {
    // Cast env to proper type
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    // Validate environment pointer
    اگر (!kernel_env) {
        واپس -1;
    }

    // Get thread info using shire mask from environment
    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    // Return early if this hart is not active
    اگر (thread_id < 0) {
        واپس 0;
    }

    // Basic safety check on params
    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;  // Invalid pointer
    }

    // Extract tensor references
    ساخت ggml_tensor * src0     = &params->src0;     // Input tensor
    ساخت ggml_tensor * src1     = &params->src1;     // Mask tensor (optional)
    ساخت ggml_tensor * src2     = &params->src2;     // Sinks tensor (optional)
    ساخت ggml_tensor * dst      = &params->dst;      // Output tensor
    اعشاری                scale    = params->scale;     // Scale factor
    اعشاری                max_bias = params->max_bias;  // ALiBi max bias

    // Validate tensor types (F32 only)
    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;  // Unsupported type combination
    }

    // Check if mask is used and validate type
    bool use_نقاب = (src1->ڈیٹا != تہی && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16));

    bool use_sinks = (src2->ڈیٹا != تہی && src2->type == GGML_TYPE_F32);

    اعشاری * src0_ڈیٹا  = (اعشاری *) src0->ڈیٹا;
    اعشاری * dst_ڈیٹا   = (اعشاری *) dst->ڈیٹا;
    اعشاری * نقاب_ڈیٹا  = use_نقاب ? (اعشاری *) src1->ڈیٹا : تہی;
    اعشاری * sinks_ڈیٹا = use_sinks ? (اعشاری *) src2->ڈیٹا : تہی;

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;  // Null data pointer
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
    اگر (ne00 == 1) {
        ثابت int64_t کل_elems  = ne01 * ne02 * ne03;
        ثابت int64_t elems_فی_cl = ET_CACHE_سطر_سائز_بائٹیں / (int64_t) حجم(اعشاری);  // 16
        ثابت int64_t کل_cls    = (کل_elems + elems_فی_cl - 1) / elems_فی_cl;

        برائے (int64_t cl = thread_id; cl < کل_cls; cl += عدد_دھاگے) {
            ثابت int64_t آغاز = cl * elems_فی_cl;
            int64_t       انتہا   = آغاز + elems_فی_cl;
            اگر (انتہا > کل_elems) {
                انتہا = کل_elems;
            }
            برائے (int64_t idx = آغاز; idx < انتہا; idx++) {
                dst_ڈیٹا[idx] = 1.0f;
            }
        }
        واپس 0;
    }

    ثابت int64_t ne10 = use_نقاب ? src1->ne[0] : 0;  // Mask sequence length
    ثابت int64_t ne11 = use_نقاب ? src1->ne[1] : 0;  // Mask rows
    ثابت int64_t ne12 = use_نقاب ? src1->ne[2] : 0;  // Mask batch/head dimension
    ثابت int64_t ne13 = use_نقاب ? src1->ne[3] : 0;  // Mask outer batch dimension

    اگر (use_نقاب) {
        // - Dimension 0: mask must equal input exactly
        // - Dimension 1: mask must be >= input (allows larger pre-allocated masks)
        // - Dimension 2: input must be divisible by mask (modulo broadcasting)
        // - Dimension 3: input must be divisible by mask (modulo broadcasting)
        اگر (ne10 != ne00 ||                    // Dimension 0: exact match required
            ne11 < ne01 ||                     // Dimension 1: mask >= input
            (ne12 > 0 && ne02 % ne12 != 0) ||  // Dimension 2: input % mask == 0
            (ne13 > 0 && ne03 % ne13 != 0)) {  // Dimension 3: input % mask == 0
            واپس -1;                         // Incompatible dimensions for ggml softmax broadcasting
        }
    }

    // ALiBi slope calculation - compute per attention head
    ثابت uint32_t n_سر      = (uint32_t) ne02;
    uint32_t       n_سر_log2 = 0;
    اعشاری          m0          = 1.0f;
    اعشاری          m1          = 1.0f;

    اگر (max_bias > 0.0f) {
        // This is equivalent to: 1 << floor(log2(n_head))
        n_سر_log2 = 1;
        جب (n_سر_log2 < n_سر) {
            n_سر_log2 <<= 1;
        }
        اگر (n_سر_log2 > n_سر) {
            n_سر_log2 >>= 1;
        }

        // Compute base slopes for ALiBi
        // m0 = 2^(-max_bias / n_head_log2)
        // m1 = 2^(-max_bias / (2 * n_head_log2))
        اعشاری inv_n_سر_log2 = et_fdiv(1.0f, (اعشاری) n_سر_log2);
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
    ثابت int64_t صفیں_فی_i03 = ne02 * ne01;
    ثابت int64_t کل_صفیں   = ne03 * صفیں_فی_i03;
    ثابت int64_t صفیں_فی_wg  = et_صفیں_فی_cacheline_گروہ(ne00, حجم(اعشاری));
    ثابت int64_t کل_wgs    = (کل_صفیں + صفیں_فی_wg - 1) / صفیں_فی_wg;

    برائے (int64_t wg = thread_id; wg < کل_wgs; wg += عدد_دھاگے) {
        ثابت int64_t صف_آغاز = wg * صفیں_فی_wg;
        int64_t       صف_انتہا   = صف_آغاز + صفیں_فی_wg;
        اگر (صف_انتہا > کل_صفیں) {
            صف_انتہا = کل_صفیں;
        }

        برائے (int64_t صف = صف_آغاز; صف < صف_انتہا; صف++) {
            ثابت int64_t i03 = صف / صفیں_فی_i03;
            ثابت int64_t rem = صف % صفیں_فی_i03;
            ثابت int64_t i02 = rem / ne01;
            ثابت int64_t i01 = rem % ne01;

            // Calculate ALiBi slope for this attention head
            اعشاری ڈھلان = 1.0f;
            اگر (max_bias > 0.0f) {
                ثابت uint32_t h = (uint32_t) i02;  // head index
                اگر (h < n_سر_log2) {
                    ڈھلان = m0;
                    برائے (uint32_t i = 0; i < h; i++) {
                        ڈھلان *= m0;
                    }
                } ورنہ {
                    ثابت uint32_t قوت_نما = 2 * (h - n_سر_log2) + 1;
                    ڈھلان              = m1;
                    برائے (uint32_t i = 1; i < قوت_نما; i++) {
                        ڈھلان *= m1;
                    }
                }
            }

            اعشاری sink_قدر = 0.0f;
            اگر (use_sinks && sinks_ڈیٹا) {
                sink_قدر = sinks_ڈیٹا[i02];
            }

            ثابت int64_t src_انحراف = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00;

            ثابت اعشاری * src_صف  = src0_ڈیٹا + src_انحراف;
            اعشاری *       dst_صف  = dst_ڈیٹا + src_انحراف;
            ثابت اعشاری * نقاب_صف = تہی;

            اگر (use_نقاب && نقاب_ڈیٹا) {
                ثابت int64_t نقاب_i03 = (ne13 > 0) ? i03 % ne13 : 0;
                ثابت int64_t نقاب_i02 = (ne12 > 0) ? i02 % ne12 : 0;
                ثابت int64_t نقاب_i01 = i01;

                ثابت int64_t نقاب_انحراف = نقاب_i03 * ne12 * ne11 * ne10 + نقاب_i02 * ne11 * ne10 + نقاب_i01 * ne10;

                نقاب_صف = نقاب_ڈیٹا + نقاب_انحراف;
            }

            compute_softmax_صف(dst_صف, src_صف, نقاب_صف, (صحیح) ne00, scale, ڈھلان, sink_قدر, use_sinks);
        }
    }

    واپس 0;  // Success
}
