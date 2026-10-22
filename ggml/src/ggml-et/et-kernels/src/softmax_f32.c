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

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <assert.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ <stdint.h>

// Softmax kernel parameters structure (from ggml-et-ops.h)
నిర్మాణం ggml_et_softmax_params {
    నిర్మాణం ggml_tensor src0;      // F32 input tensor
    నిర్మాణం ggml_tensor src1;      // F32 mask tensor (optional, may be zeroed if not used)
    నిర్మాణం ggml_tensor src2;      // F32 sinks tensor (optional, may be zeroed if not used)
    నిర్మాణం ggml_tensor dst;       // F32 output tensor
    దశాంశం              scale;     // Scale factor (temperature scaling)
    దశాంశం              max_bias;  // Max bias for ALiBi (0.0f if not used)
};

#నిర్వచించు LOG2E_F 1.4426950408889634f

రకనిర్వచనం నిర్మాణం {
    దశాంశం    max_val;
    దశాంశం    కూడిక_val;
    uint32_t valid_ముసుగు;
} softmax_params_t;

స్థిర అంతఃవరుస bool softmax_lane_is_valid(దశాంశం x) {
    తిరిగి (x == x) && (x != -INFINITY) && (x != INFINITY);
}

స్థిర అంతఃవరుస softmax_params_t softmax_params_ఖాళీలు(శూన్య) {
    softmax_params_t p;
    p.max_val    = -INFINITY;
    p.కూడిక_val    = 0.0f;
    p.valid_ముసుగు = 0;
    తిరిగి p;
}

// chunk_transform_ps_8_branchless_mask
//
// Vector transform for 8 logits:
//
//   x = src * scale + (mask ? mask * slope : 0)
//
// Implemented branchlessly so masked and unmasked paths share the same
// instruction stream. Used by pass1 and pass2 vector loops.
స్థిర అంతఃవరుస శూన్య chunk_transform_ps_8_branchless_ముసుగు(దశాంశం *       tmp8,
                                                        స్థిరరాశి దశాంశం * src,
                                                        స్థిరరాశి దశాంశం * ముసుగు,
                                                        దశాంశం         scale,
                                                        దశాంశం         వాలు) {
    గుర్తులేని దీర్ఘ       ms;
    స్థిరరాశి దశాంశం         zero         = 0.0f;
    స్థిరరాశి గుర్తులేని దీర్ఘ ముసుగు_ఎక్కించు_m0 = (ముసుగు != ఖాళీ) ? 0xFFul : 0x00ul;
    స్థిరరాశి దశాంశం *       mp           = (ముసుగు != ఖాళీ) ? ముసుగు : &zero;

    __asm__ అస్థిర(
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
          [p_వాలు] "r"(&వాలు), [maskm0] "r"(ముసుగు_ఎక్కించు_m0)
        : "f0", "f1", "f10", "f11", "memory");
}

// chunk_transform_ps_8_tail
//
// Same as chunk_transform_ps_8_branchless_mask but gates loads, compute,
// and stores with a caller-supplied m0 mask so that only `count` elements
// (1-7) are touched.  Used for the last sub-8 chunk of a non-aligned row.
స్థిర అంతఃవరుస శూన్య chunk_transform_ps_8_తోక(దశాంశం *       tmp8,
                                             స్థిరరాశి దశాంశం * src,
                                             స్థిరరాశి దశాంశం * ముసుగు,
                                             దశాంశం         scale,
                                             దశాంశం         వాలు,
                                             గుర్తులేని దీర్ఘ తోక_m0) {
    గుర్తులేని దీర్ఘ       ms;
    స్థిరరాశి దశాంశం         zero         = 0.0f;
    స్థిరరాశి గుర్తులేని దీర్ఘ ముసుగు_ఎక్కించు_m0 = (ముసుగు != ఖాళీ) ? తోక_m0 : 0x00ul;
    స్థిరరాశి దశాంశం *       mp           = (ముసుగు != ఖాళీ) ? ముసుగు : &zero;

    __asm__ అస్థిర(
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
          [p_వాలు] "r"(&వాలు), [maskm0] "r"(ముసుగు_ఎక్కించు_m0), [tailm0] "r"(తోక_m0)
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
స్థిర అంతఃవరుస softmax_params_t softmax_pass1_range(స్థిరరాశి దశాంశం * src,
                                                   స్థిరరాశి దశాంశం * ముసుగు,
                                                   పూర్ణాంకం           begin,
                                                   పూర్ణాంకం           చివర,
                                                   దశాంశం         scale,
                                                   దశాంశం         వాలు) {
    __attribute__((aligned(32))) దశాంశం lane_max[8];
    __attribute__((aligned(32))) దశాంశం lane_కూడిక[8];
    __attribute__((aligned(32))) దశాంశం tmp[8];

    uint8_t valid_ముసుగు = 0;

    స్థిరరాశి దశాంశం one_f   = 1.0f;
    స్థిరరాశి దశాంశం zero_f  = 0.0f;
    స్థిరరాశి దశాంశం neg_inf = -INFINITY;
    స్థిరరాశి దశాంశం log2e   = LOG2E_F;

    గుర్తులేని దీర్ఘ ms;

    __asm__ అస్థిర(
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f20, 0(%[p_ninf])    \n\t"
        "fbc.ps    f21, 0(%[p_zero])    \n\t"
        "fbc.ps    f22, 0(%[p_one])     \n\t"
        "fbc.ps    f23, 0(%[p_log2e])   \n\t"
        : [ms] "=&r"(ms)
        : [p_ninf] "r"(&neg_inf), [p_zero] "r"(&zero_f), [p_one] "r"(&one_f), [p_log2e] "r"(&log2e)
        : "f20", "f21", "f22", "f23");

    స్థిరరాశి పూర్ణాంకం aligned_చివర = begin + ((చివర - begin) & ~7);

    // Process full 8-element chunks
    పూర్ణాంకం i = begin;
    కొరకు (; i < aligned_చివర; i += 8) {
        chunk_transform_ps_8_branchless_ముసుగు(tmp, src + i, ముసుగు ? (ముసుగు + i) : ఖాళీ, scale, వాలు);

        uint8_t cur_ముసుగు = 0;
        కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) {
            ఒకవేళ (softmax_lane_is_valid(tmp[j])) {
                cur_ముసుగు |= (uint8_t) (1u << j);
            }
        }

        స్థిరరాశి uint8_t init_ముసుగు = (uint8_t) (cur_ముసుగు & ~valid_ముసుగు);
        స్థిరరాశి uint8_t upd_ముసుగు  = (uint8_t) (cur_ముసుగు & valid_ముసుగు);

        ఒకవేళ (init_ముసుగు || upd_ముసుగు) {
            __asm__ అస్థిర(
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
                : [p_tmp] "r"(tmp), [initm] "r"((గుర్తులేని దీర్ఘ) init_ముసుగు), [updm] "r"((గుర్తులేని దీర్ఘ) upd_ముసుగు)
                : "f0", "f1", "f2", "f3", "memory");

            valid_ముసుగు |= cur_ముసుగు;
        }
    }

    // Tail chunk: m0-gated load/compute/store for remaining 1-7 elements
    ఒకవేళ (i < చివర) {
        స్థిరరాశి గుర్తులేని దీర్ఘ తోక_m0 = (1ul << (చివర - i)) - 1;

        // Fill tmp with NaN so invalid lanes fail softmax_lane_is_valid
        కొరకు (పూర్ణాంకం j = 0; j < 8; j++) {
            tmp[j] = __builtin_nanf("");
        }

        chunk_transform_ps_8_తోక(tmp, src + i, ముసుగు ? (ముసుగు + i) : ఖాళీ, scale, వాలు, తోక_m0);

        uint8_t cur_ముసుగు = 0;
        కొరకు (పూర్ణాంకం j = 0; j < 8; ++j) {
            ఒకవేళ (softmax_lane_is_valid(tmp[j])) {
                cur_ముసుగు |= (uint8_t) (1u << j);
            }
        }

        స్థిరరాశి uint8_t init_ముసుగు = (uint8_t) (cur_ముసుగు & ~valid_ముసుగు);
        స్థిరరాశి uint8_t upd_ముసుగు  = (uint8_t) (cur_ముసుగు & valid_ముసుగు);

        ఒకవేళ (init_ముసుగు || upd_ముసుగు) {
            __asm__ అస్థిర(
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
                : [p_tmp] "r"(tmp), [initm] "r"((గుర్తులేని దీర్ఘ) init_ముసుగు), [updm] "r"((గుర్తులేని దీర్ఘ) upd_ముసుగు)
                : "f0", "f1", "f2", "f3", "memory");

            valid_ముసుగు |= cur_ముసుగు;
        }
    }

    __asm__ అస్థిర(
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fsw.ps    f20, 0(%[p_lmax])    \n\t"
        "fsw.ps    f21, 0(%[p_lsum])    \n\t"
        "mova.m.x  %[ms]                \n\t"
        :
        : [p_lmax] "r"(lane_max), [p_lsum] "r"(lane_కూడిక), [ms] "r"(ms)
        : "memory");

    softmax_params_t బహిర్గతం = softmax_params_ఖాళీలు();
    బహిర్గతం.valid_ముసుగు       = valid_ముసుగు;

    కొరకు (పూర్ణాంకం k = 0; k < 8; ++k) {
        ఒకవేళ (valid_ముసుగు & (1u << k)) {
            ఒకవేళ (బహిర్గతం.valid_ముసుగు == (1u << k) || బహిర్గతం.max_val == -INFINITY || lane_max[k] > బహిర్గతం.max_val) {
                బహిర్గతం.max_val = lane_max[k];
            }
        }
    }

    ఒకవేళ (బహిర్గతం.max_val != -INFINITY) {
        // Compute lane correction factors via fexp.ps to stay consistent
        // with the fexp.ps used inside the online softmax loop above.
        // corr[k] = exp2((lane_max[k] - out.max_val) * LOG2E) = exp(lane_max[k] - out.max_val)
        స్థిరరాశి దశాంశం                        neg_max_l2 = -బహిర్గతం.max_val * LOG2E_F;
        __attribute__((aligned(32))) దశాంశం corr[8];
        __asm__ అస్థిర(
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
        కొరకు (పూర్ణాంకం k = 0; k < 8; ++k) {
            ఒకవేళ (valid_ముసుగు & (1u << k)) {
                బహిర్గతం.కూడిక_val += lane_కూడిక[k] * corr[k];
            }
        }
    }

    తిరిగి బహిర్గతం;
}

// Pass 2 (normalize) over [begin, end).
//
// Computes: dst[i] = exp(x[i]*scale + mask[i]*slope - max) / sum
//
// Uses fexp.ps for the numerator; the denominator (params.sum_val) must
// already be fully computed by the caller (pass1 + any sink merge).
స్థిర అంతఃవరుస శూన్య softmax_pass2_range(దశాంశం *          dst,
                                       స్థిరరాశి దశాంశం *    src,
                                       స్థిరరాశి దశాంశం *    ముసుగు,
                                       పూర్ణాంకం              begin,
                                       పూర్ణాంకం              చివర,
                                       దశాంశం            scale,
                                       దశాంశం            వాలు,
                                       softmax_params_t params) {
    స్థిరరాశి దశాంశం s2      = scale * LOG2E_F;
    స్థిరరాశి దశాంశం sl2     = వాలు * LOG2E_F;
    స్థిరరాశి దశాంశం neg_ml2 = -params.max_val * LOG2E_F;
    స్థిరరాశి దశాంశం inv_కూడిక = et_fdiv(1.0f, params.కూడిక_val);

    గుర్తులేని దీర్ఘ ms;

    __asm__ అస్థిర(
        "mova.x.m  %[ms]                \n\t"
        "mov.m.x   m0, x0, 0xFF         \n\t"
        "fbc.ps    f10, 0(%[p_s2])      \n\t"
        "fbc.ps    f12, 0(%[p_nml2])    \n\t"
        "fbc.ps    f13, 0(%[p_inv])     \n\t"
        : [ms] "=&r"(ms)
        : [p_s2] "r"(&s2), [p_nml2] "r"(&neg_ml2), [p_inv] "r"(&inv_కూడిక)
        : "f10", "f12", "f13");

    స్థిరరాశి పూర్ణాంకం aligned_చివర = begin + ((చివర - begin) & ~7);

    ఒకవేళ (ముసుగు != ఖాళీ) {
        __asm__ అస్థిర("fbc.ps    f11, 0(%[p_sl2]) \n\t" : : [p_sl2] "r"(&sl2) : "f11");

        కొరకు (పూర్ణాంకం c = begin; c < aligned_చివర; c += 8) {
            __asm__ అస్థిర(
                "flw.ps    f0, 0(%[sp])           \n\t"
                "flw.ps    f1, 0(%[mp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fmadd.ps  f0, f1, f11, f0        \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                :
                : [sp] "r"(src + c), [mp] "r"(ముసుగు + c), [dp] "r"(dst + c)
                : "f0", "f1", "memory");
        }

        // Tail chunk with m0 gating
        ఒకవేళ (aligned_చివర < చివర) {
            స్థిరరాశి గుర్తులేని దీర్ఘ తోక_m0 = (1ul << (చివర - aligned_చివర)) - 1;
            __asm__ అస్థిర(
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
                : [sp] "r"(src + aligned_చివర), [mp] "r"(ముసుగు + aligned_చివర), [dp] "r"(dst + aligned_చివర),
                  [tm] "r"(తోక_m0)
                : "f0", "f1", "memory");
        }
    } లేకపోతే {
        కొరకు (పూర్ణాంకం c = begin; c < aligned_చివర; c += 8) {
            __asm__ అస్థిర(
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
        ఒకవేళ (aligned_చివర < చివర) {
            స్థిరరాశి గుర్తులేని దీర్ఘ తోక_m0 = (1ul << (చివర - aligned_చివర)) - 1;
            __asm__ అస్థిర(
                "mov.m.x   m0, %[tm], 0              \n\t"
                "flw.ps    f0, 0(%[sp])           \n\t"
                "fmadd.ps  f0, f0, f10, f12       \n\t"
                "fexp.ps   f0, f0                 \n\t"
                "fmul.ps   f0, f0, f13            \n\t"
                "fsw.ps    f0, 0(%[dp])           \n\t"
                "mov.m.x   m0, x0, 0xFF           \n\t"
                :
                : [sp] "r"(src + aligned_చివర), [dp] "r"(dst + aligned_చివర), [tm] "r"(తోక_m0)
                : "f0", "memory");
        }
    }

    __asm__ అస్థిర("mova.m.x  %[ms] \n\t" ::[ms] "r"(ms));
}

// Single-core row path.
// pass1_range and pass2_range handle non-8-aligned cols internally via
// m0-gated tail chunks, so this function just passes cols directly.
స్థిర అంతఃవరుస శూన్య compute_softmax_అడ్డువరుస(దశాంశం *       dst,
                                       స్థిరరాశి దశాంశం * src,
                                       స్థిరరాశి దశాంశం * ముసుగు,
                                       పూర్ణాంకం           నిలువుల_సంఖ్య_2,
                                       దశాంశం         scale,
                                       దశాంశం         వాలు,
                                       దశాంశం         sink_విలువ,
                                       bool          use_sinks) {
    softmax_params_t params = softmax_pass1_range(src, ముసుగు, 0, నిలువుల_సంఖ్య_2, scale, వాలు);

    ఒకవేళ (use_sinks) {
        // For sinks, use fully scalar et_expf to match the reference CPU
        // backend's expf precision.  Sink tests use small arrays (ne<=32)
        // so the scalar path has negligible performance impact.
        దశాంశం max_val = params.max_val;
        ఒకవేళ (sink_విలువ > max_val) {
            max_val = sink_విలువ;
        }

        // Compute sum = Σ exp(x'[i] - max) + exp(sink - max)  (scalar)
        దశాంశం కూడిక = 0.0f;
        కొరకు (పూర్ణాంకం i = 0; i < నిలువుల_సంఖ్య_2; ++i) {
            దశాంశం x = src[i] * scale;
            ఒకవేళ (ముసుగు != ఖాళీ) {
                x += ముసుగు[i] * వాలు;
            }
            కూడిక += et_expf(x - max_val);
        }
        కూడిక += et_expf(sink_విలువ - max_val);

        // Normalize: dst[i] = exp(x'[i] - max) / sum  (scalar)
        దశాంశం inv_కూడిక = et_fdiv(1.0f, కూడిక);
        కొరకు (పూర్ణాంకం i = 0; i < నిలువుల_సంఖ్య_2; ++i) {
            దశాంశం x = src[i] * scale;
            ఒకవేళ (ముసుగు != ఖాళీ) {
                x += ముసుగు[i] * వాలు;
            }
            dst[i] = et_expf(x - max_val) * inv_కూడిక;
        }
    } లేకపోతే {
        ఒకవేళ (!params.valid_ముసుగు) {
            తిరిగి;
        }
        softmax_pass2_range(dst, src, ముసుగు, 0, నిలువుల_సంఖ్య_2, scale, వాలు, params);
    }
}

// Main entry point for Softmax kernel
పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_softmax_params * params, శూన్య * env) {
    // Cast env to proper type
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    // Validate environment pointer
    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    // Get thread info using shire mask from environment
    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    // Return early if this hart is not active
    ఒకవేళ (thread_id < 0) {
        తిరిగి 0;
    }

    // Basic safety check on params
    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;  // Invalid pointer
    }

    // Extract tensor references
    నిర్మాణం ggml_tensor * src0     = &params->src0;     // Input tensor
    నిర్మాణం ggml_tensor * src1     = &params->src1;     // Mask tensor (optional)
    నిర్మాణం ggml_tensor * src2     = &params->src2;     // Sinks tensor (optional)
    నిర్మాణం ggml_tensor * dst      = &params->dst;      // Output tensor
    దశాంశం                scale    = params->scale;     // Scale factor
    దశాంశం                max_bias = params->max_bias;  // ALiBi max bias

    // Validate tensor types (F32 only)
    ఒకవేళ (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;  // Unsupported type combination
    }

    // Check if mask is used and validate type
    bool use_ముసుగు = (src1->దత్తాంశం != ఖాళీ && (src1->type == GGML_TYPE_F32 || src1->type == GGML_TYPE_F16));

    bool use_sinks = (src2->దత్తాంశం != ఖాళీ && src2->type == GGML_TYPE_F32);

    దశాంశం * src0_దత్తాంశం  = (దశాంశం *) src0->దత్తాంశం;
    దశాంశం * dst_దత్తాంశం   = (దశాంశం *) dst->దత్తాంశం;
    దశాంశం * ముసుగు_దత్తాంశం  = use_ముసుగు ? (దశాంశం *) src1->దత్తాంశం : ఖాళీ;
    దశాంశం * sinks_దత్తాంశం = use_sinks ? (దశాంశం *) src2->దత్తాంశం : ఖాళీ;

    ఒకవేళ (!src0_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;  // Null data pointer
    }


    స్థిరరాశి int64_t ne00 = src0->ne[0];  // Sequence length (columns)
    స్థిరరాశి int64_t ne01 = src0->ne[1];  // Number of rows
    స్థిరరాశి int64_t ne02 = src0->ne[2];  // Batch/head dimension
    స్థిరరాశి int64_t ne03 = src0->ne[3];  // Outer batch dimension

    // Fast path: softmax of a single element is always 1.0
    // (exp(x) / exp(x) == 1 for any x, regardless of scale/mask/bias)
    // Skip all ALiBi, mask, and sink setup.
    //
    // Each output element is 4 bytes.  A cache line is 64 bytes = 16 floats.
    // L1 is not coherent across harts, so each thread must own whole cache
    // lines to avoid cross-hart conflicts.
    ఒకవేళ (ne00 == 1) {
        స్థిరరాశి int64_t మొత్తం_elems  = ne01 * ne02 * ne03;
        స్థిరరాశి int64_t elems_చొప్పున_cl = ET_CACHE_పంక్తి_కొలత_బైట్లు / (int64_t) పరిమాణం(దశాంశం);  // 16
        స్థిరరాశి int64_t మొత్తం_cls    = (మొత్తం_elems + elems_చొప్పున_cl - 1) / elems_చొప్పున_cl;

        కొరకు (int64_t cl = thread_id; cl < మొత్తం_cls; cl += అంకె_దారాలు) {
            స్థిరరాశి int64_t ప్రారంభం = cl * elems_చొప్పున_cl;
            int64_t       చివర   = ప్రారంభం + elems_చొప్పున_cl;
            ఒకవేళ (చివర > మొత్తం_elems) {
                చివర = మొత్తం_elems;
            }
            కొరకు (int64_t idx = ప్రారంభం; idx < చివర; idx++) {
                dst_దత్తాంశం[idx] = 1.0f;
            }
        }
        తిరిగి 0;
    }

    స్థిరరాశి int64_t ne10 = use_ముసుగు ? src1->ne[0] : 0;  // Mask sequence length
    స్థిరరాశి int64_t ne11 = use_ముసుగు ? src1->ne[1] : 0;  // Mask rows
    స్థిరరాశి int64_t ne12 = use_ముసుగు ? src1->ne[2] : 0;  // Mask batch/head dimension
    స్థిరరాశి int64_t ne13 = use_ముసుగు ? src1->ne[3] : 0;  // Mask outer batch dimension

    ఒకవేళ (use_ముసుగు) {
        // - Dimension 0: mask must equal input exactly
        // - Dimension 1: mask must be >= input (allows larger pre-allocated masks)
        // - Dimension 2: input must be divisible by mask (modulo broadcasting)
        // - Dimension 3: input must be divisible by mask (modulo broadcasting)
        ఒకవేళ (ne10 != ne00 ||                    // Dimension 0: exact match required
            ne11 < ne01 ||                     // Dimension 1: mask >= input
            (ne12 > 0 && ne02 % ne12 != 0) ||  // Dimension 2: input % mask == 0
            (ne13 > 0 && ne03 % ne13 != 0)) {  // Dimension 3: input % mask == 0
            తిరిగి -1;                         // Incompatible dimensions for ggml softmax broadcasting
        }
    }

    // ALiBi slope calculation - compute per attention head
    స్థిరరాశి uint32_t n_తల      = (uint32_t) ne02;
    uint32_t       n_తల_log2 = 0;
    దశాంశం          m0          = 1.0f;
    దశాంశం          m1          = 1.0f;

    ఒకవేళ (max_bias > 0.0f) {
        // This is equivalent to: 1 << floor(log2(n_head))
        n_తల_log2 = 1;
        అయితే (n_తల_log2 < n_తల) {
            n_తల_log2 <<= 1;
        }
        ఒకవేళ (n_తల_log2 > n_తల) {
            n_తల_log2 >>= 1;
        }

        // Compute base slopes for ALiBi
        // m0 = 2^(-max_bias / n_head_log2)
        // m1 = 2^(-max_bias / (2 * n_head_log2))
        దశాంశం inv_n_తల_log2 = et_fdiv(1.0f, (దశాంశం) n_తల_log2);
        m0                    = et_expf(-max_bias * 0.69314718f * inv_n_తల_log2);  // 0.69314718 = ln(2)
        m1                    = et_expf(-max_bias * 0.69314718f * inv_n_తల_log2 * 0.5f);
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
    స్థిరరాశి int64_t అడ్డువరుసలు_చొప్పున_i03 = ne02 * ne01;
    స్థిరరాశి int64_t మొత్తం_అడ్డువరుసలు   = ne03 * అడ్డువరుసలు_చొప్పున_i03;
    స్థిరరాశి int64_t అడ్డువరుసలు_చొప్పున_wg  = et_అడ్డువరుసలు_చొప్పున_cacheline_సమూహం(ne00, పరిమాణం(దశాంశం));
    స్థిరరాశి int64_t మొత్తం_wgs    = (మొత్తం_అడ్డువరుసలు + అడ్డువరుసలు_చొప్పున_wg - 1) / అడ్డువరుసలు_చొప్పున_wg;

    కొరకు (int64_t wg = thread_id; wg < మొత్తం_wgs; wg += అంకె_దారాలు) {
        స్థిరరాశి int64_t అడ్డువరుస_ప్రారంభం = wg * అడ్డువరుసలు_చొప్పున_wg;
        int64_t       అడ్డువరుస_చివర   = అడ్డువరుస_ప్రారంభం + అడ్డువరుసలు_చొప్పున_wg;
        ఒకవేళ (అడ్డువరుస_చివర > మొత్తం_అడ్డువరుసలు) {
            అడ్డువరుస_చివర = మొత్తం_అడ్డువరుసలు;
        }

        కొరకు (int64_t అడ్డువరుస = అడ్డువరుస_ప్రారంభం; అడ్డువరుస < అడ్డువరుస_చివర; అడ్డువరుస++) {
            స్థిరరాశి int64_t i03 = అడ్డువరుస / అడ్డువరుసలు_చొప్పున_i03;
            స్థిరరాశి int64_t rem = అడ్డువరుస % అడ్డువరుసలు_చొప్పున_i03;
            స్థిరరాశి int64_t i02 = rem / ne01;
            స్థిరరాశి int64_t i01 = rem % ne01;

            // Calculate ALiBi slope for this attention head
            దశాంశం వాలు = 1.0f;
            ఒకవేళ (max_bias > 0.0f) {
                స్థిరరాశి uint32_t h = (uint32_t) i02;  // head index
                ఒకవేళ (h < n_తల_log2) {
                    వాలు = m0;
                    కొరకు (uint32_t i = 0; i < h; i++) {
                        వాలు *= m0;
                    }
                } లేకపోతే {
                    స్థిరరాశి uint32_t ఘాతాంకం = 2 * (h - n_తల_log2) + 1;
                    వాలు              = m1;
                    కొరకు (uint32_t i = 1; i < ఘాతాంకం; i++) {
                        వాలు *= m1;
                    }
                }
            }

            దశాంశం sink_విలువ = 0.0f;
            ఒకవేళ (use_sinks && sinks_దత్తాంశం) {
                sink_విలువ = sinks_దత్తాంశం[i02];
            }

            స్థిరరాశి int64_t src_స్థానభ్రంశం = i03 * ne02 * ne01 * ne00 + i02 * ne01 * ne00 + i01 * ne00;

            స్థిరరాశి దశాంశం * src_అడ్డువరుస  = src0_దత్తాంశం + src_స్థానభ్రంశం;
            దశాంశం *       dst_అడ్డువరుస  = dst_దత్తాంశం + src_స్థానభ్రంశం;
            స్థిరరాశి దశాంశం * ముసుగు_అడ్డువరుస = ఖాళీ;

            ఒకవేళ (use_ముసుగు && ముసుగు_దత్తాంశం) {
                స్థిరరాశి int64_t ముసుగు_i03 = (ne13 > 0) ? i03 % ne13 : 0;
                స్థిరరాశి int64_t ముసుగు_i02 = (ne12 > 0) ? i02 % ne12 : 0;
                స్థిరరాశి int64_t ముసుగు_i01 = i01;

                స్థిరరాశి int64_t ముసుగు_స్థానభ్రంశం = ముసుగు_i03 * ne12 * ne11 * ne10 + ముసుగు_i02 * ne11 * ne10 + ముసుగు_i01 * ne10;

                ముసుగు_అడ్డువరుస = ముసుగు_దత్తాంశం + ముసుగు_స్థానభ్రంశం;
            }

            compute_softmax_అడ్డువరుస(dst_అడ్డువరుస, src_అడ్డువరుస, ముసుగు_అడ్డువరుస, (పూర్ణాంకం) ne00, scale, వాలు, sink_విలువ, use_sinks);
        }
    }

    తిరిగి 0;  // Success
}
