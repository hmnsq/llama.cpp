//******************************************************************************
// Unary F32 Kernel
// Element-wise unary operations: dst[i] = f(src0[i])
// All ops vectorized using 8-wide ET SIMD (fexp.ps, frcp.ps, flog.ps, etc.)
//
// Supports: ABS, SGN, NEG, STEP, TANH, ELU, RELU, SIGMOID, GELU, GELU_QUICK,
//           SILU, HARDSWISH, HARDSIGMOID, EXP, EXPM1, SOFTPLUS, GELU_ERF
//******************************************************************************

#অন্তর্ভাৱ "ggml_tensor.h"
#অন্তর্ভাৱ "math_fp.h"
#অন্তর্ভাৱ "platform.h"

#অন্তর্ভাৱ <stdint.h>

// Unary kernel parameters structure
সংরচনা ggml_et_unary_params {
    সংরচনা ggml_tensor src0;      // F32 input tensor
    সংরচনা ggml_tensor dst;       // F32 output tensor
    int32_t            unary_op;  // ggml_unary_op enum value
};

//******************************************************************************
// Vectorized 8-wide block operations
// All process exactly 8 floats per call using ET vector instructions.
// ne0 is guaranteed % 16 == 0, so the inner loop always calls with i0 += 8.
//******************************************************************************

// NEG: dst = -x  (zero - x)
স্থির অন্তঃস্থ শূন্য vec_neg(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    ভিন্ন zero = 0.0f;
    বাবে (int32_t i = 0; i < n; i += 8) {
        __asm__ চঞ্চল(
            "fbc.ps  f10, %[z]\n"
            "flw.ps  f11, %[x]\n"
            "fsub.ps f12, f10, f11\n"
            "fsw.ps  f12, %[r]\n"
            : [r] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src[i]), [z] "m"(zero)
            : "f10", "f11", "f12");
    }
}

// ABS: dst = |x|  (negate negative values: abs = x * sgn, or max(x, -x))
// Uses: negate then fmax.ps
স্থির অন্তঃস্থ শূন্য vec_abs(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    ভিন্ন zero = 0.0f;
    বাবে (int32_t i = 0; i < n; i += 8) {
        __asm__ চঞ্চল(
            "fbc.ps  f10, %[z]\n"
            "flw.ps  f11, %[x]\n"
            "fsub.ps f12, f10, f11\n"  // f12 = -x
            "fmax.ps f13, f11, f12\n"  // f13 = max(x, -x) = |x|
            "fsw.ps  f13, %[r]\n"
            : [r] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src[i]), [z] "m"(zero)
            : "f10", "f11", "f12", "f13");
    }
}

// RELU: dst = max(0, x)
স্থির অন্তঃস্থ শূন্য vec_relu(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    ভিন্ন zero = 0.0f;
    বাবে (int32_t i = 0; i < n; i += 8) {
        __asm__ চঞ্চল(
            "fbc.ps  f10, %[z]\n"
            "flw.ps  f11, %[x]\n"
            "fmax.ps f12, f10, f11\n"  // max(0, x)
            "fsw.ps  f12, %[r]\n"
            : [r] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src[i]), [z] "m"(zero)
            : "f10", "f11", "f12");
    }
}

// STEP: dst = x > 0 ? 1 : 0  (clamp to [0,1] via max then min-ish, or use sign bit)
// Trick: relu(x) then frcp gives inf for 0 and finite for >0, but simpler:
// step(x) = min(1, relu(x) * huge) ... too fragile.  Scalar is fine for step/sgn.
স্থির অন্তঃস্থ শূন্য vec_step(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    বাবে (int32_t i = 0; i < n; i++) {
        dst[i] = (src[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

// SGN: dst = sign(x) = x>0 ? 1 : (x<0 ? -1 : 0)
স্থির অন্তঃস্থ শূন্য vec_sgn(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    বাবে (int32_t i = 0; i < n; i++) {
        dst[i] = (src[i] > 0.0f) ? 1.0f : ((src[i] < 0.0f) ? -1.0f : 0.0f);
    }
}

// EXP: dst = exp(x)
// fexp.ps computes 2^x, so feed x * log2(e)
স্থির অন্তঃস্থ শূন্য vec_exp(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    ভিন্ন log2e = 1.4426950408889634f;
    বাবে (int32_t i = 0; i < n; i += 8) {
        __asm__ চঞ্চল(
            "flw.ps  f10, %[x]\n"
            "fbc.ps  f11, %[l2e]\n"
            "fmul.ps f12, f10, f11\n"  // x * log2(e)
            "fexp.ps f13, f12\n"       // 2^(x*log2e) = exp(x)
            "fsw.ps  f13, %[r]\n"
            : [r] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src[i]), [l2e] "m"(log2e)
            : "f10", "f11", "f12", "f13");
    }
}

// EXPM1: dst = exp(x) - 1
স্থির অন্তঃস্থ শূন্য vec_expm1(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    ভিন্ন log2e = 1.4426950408889634f;
    ভিন্ন one   = 1.0f;
    বাবে (int32_t i = 0; i < n; i += 8) {
        __asm__ চঞ্চল(
            "flw.ps  f10, %[x]\n"
            "fbc.ps  f11, %[l2e]\n"
            "fbc.ps  f14, %[one]\n"
            "fmul.ps f12, f10, f11\n"  // x * log2(e)
            "fexp.ps f13, f12\n"       // exp(x)
            "fsub.ps f13, f13, f14\n"  // exp(x) - 1
            "fsw.ps  f13, %[r]\n"
            : [r] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src[i]), [l2e] "m"(log2e), [one] "m"(one)
            : "f10", "f11", "f12", "f13", "f14");
    }
}

// SIGMOID: dst = 1 / (1 + exp(-x))
// Same pattern as SwiGLU: exp(-x) via fexp.ps, then frcp.ps
স্থির অন্তঃস্থ শূন্য vec_sigmoid(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    ভিন্ন zero  = 0.0f;
    ভিন্ন one   = 1.0f;
    ভিন্ন log2e = 1.4426950408889634f;
    বাবে (int32_t i = 0; i < n; i += 8) {
        __asm__ চঞ্চল(
            "flw.ps  f10, %[x]\n"
            "fbc.ps  f20, %[z]\n"
            "fbc.ps  f21, %[one]\n"
            "fbc.ps  f22, %[l2e]\n"
            "fsub.ps f12, f20, f10\n"  // -x
            "fmul.ps f13, f12, f22\n"  // -x * log2(e)
            "fexp.ps f14, f13\n"       // exp(-x)
            "fadd.ps f15, f14, f21\n"  // 1 + exp(-x)
            "frcp.ps f16, f15\n"       // 1 / (1 + exp(-x))
            "fsw.ps  f16, %[r]\n"
            : [r] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src[i]), [z] "m"(zero), [one] "m"(one), [l2e] "m"(log2e)
            : "f10", "f12", "f13", "f14", "f15", "f16", "f20", "f21", "f22");
    }
}

// TANH: dst = (exp(2x) - 1) / (exp(2x) + 1)
// Rewrite as: 1 - 2/(exp(2x) + 1) to use frcp.ps
// Or equivalently: 2*sigmoid(2x) - 1
স্থির অন্তঃস্থ শূন্য vec_tanh(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    ভিন্ন one       = 1.0f;
    ভিন্ন two       = 2.0f;
    ভিন্ন two_log2e = 2.8853900817779268f;  // 2 * log2(e)
    বাবে (int32_t i = 0; i < n; i += 8) {
        __asm__ চঞ্চল(
            "flw.ps  f10, %[x]\n"
            "fbc.ps  f20, %[one]\n"
            "fbc.ps  f21, %[two]\n"
            "fbc.ps  f22, %[tl2e]\n"
            // exp(2x) via fexp.ps: feed 2x * log2(e)
            "fmul.ps f12, f10, f22\n"  // 2x * log2(e)
            "fexp.ps f13, f12\n"       // exp(2x)
            "fadd.ps f14, f13, f20\n"  // exp(2x) + 1
            "frcp.ps f15, f14\n"       // 1 / (exp(2x) + 1)
            "fmul.ps f16, f21, f15\n"  // 2 / (exp(2x) + 1)
            "fsub.ps f17, f20, f16\n"  // 1 - 2/(exp(2x)+1) = tanh(x)
            "fsw.ps  f17, %[r]\n"
            : [r] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src[i]), [one] "m"(one), [two] "m"(two), [tl2e] "m"(two_log2e)
            : "f10", "f12", "f13", "f14", "f15", "f16", "f17", "f20", "f21", "f22");
    }
}

// SILU: dst = x / (1 + exp(-x)) = x * sigmoid(x)
// Copied from SwiGLU pattern but without the gate multiply
স্থির অন্তঃস্থ শূন্য vec_silu(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    ভিন্ন zero  = 0.0f;
    ভিন্ন one   = 1.0f;
    ভিন্ন log2e = 1.4426950408889634f;
    বাবে (int32_t i = 0; i < n; i += 8) {
        __asm__ চঞ্চল(
            "flw.ps  f10, %[x]\n"
            "fbc.ps  f20, %[z]\n"
            "fbc.ps  f21, %[one]\n"
            "fbc.ps  f22, %[l2e]\n"
            "fsub.ps f12, f20, f10\n"  // -x
            "fmul.ps f13, f12, f22\n"  // -x * log2(e)
            "fexp.ps f14, f13\n"       // exp(-x)
            "fadd.ps f15, f14, f21\n"  // 1 + exp(-x)
            "frcp.ps f16, f15\n"       // 1 / (1 + exp(-x))
            "fmul.ps f17, f10, f16\n"  // x * sigmoid(x)
            "fsw.ps  f17, %[r]\n"
            : [r] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src[i]), [z] "m"(zero), [one] "m"(one), [l2e] "m"(log2e)
            : "f10", "f12", "f13", "f14", "f15", "f16", "f17", "f20", "f21", "f22");
    }
}

// ELU: dst = x > 0 ? x : exp(x) - 1
// Vector: compute exp(x)-1 for all lanes, then fmax(x, exp(x)-1)
// Works because for x>0: x > exp(x)-1 is not always true...
// Actually for x>0, exp(x)-1 > x (since exp(x) > x+1 for x>0).
// So fmax won't work. Use: compute both, blend via comparison.
// Simpler: exp(x)-1 for all, then for x>0 overwrite with x.
// Without per-lane masking, do scalar for ELU.
স্থির অন্তঃস্থ শূন্য vec_elu(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    ভিন্ন log2e = 1.4426950408889634f;
    ভিন্ন one   = 1.0f;
    // Compute exp(x)-1 vectorized, then fixup positive elements
    বাবে (int32_t i = 0; i < n; i += 8) {
        __asm__ চঞ্চল(
            "flw.ps  f10, %[x]\n"
            "fbc.ps  f11, %[l2e]\n"
            "fbc.ps  f14, %[one]\n"
            "fmul.ps f12, f10, f11\n"  // x * log2(e)
            "fexp.ps f13, f12\n"       // exp(x)
            "fsub.ps f13, f13, f14\n"  // exp(x) - 1
            "fsw.ps  f13, %[r]\n"      // store exp(x)-1
            : [r] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src[i]), [l2e] "m"(log2e), [one] "m"(one)
            : "f10", "f11", "f12", "f13", "f14");
        // Fixup: for x > 0, dst = x
        বাবে (int32_t j = 0; j < 8 && (i + j) < n; j++) {
            যদি (src[i + j] > 0.0f) {
                dst[i + j] = src[i + j];
            }
        }
    }
}

// GELU: 0.5*x*(1 + tanh(sqrt(2/pi) * x * (1 + 0.044715*x^2)))
// Reformulated as: x * (1 - 1/(exp(2z)+1)) where z = sqrt(2/pi)*x*(1+0.044715*x^2)
// NaN-safe: avoids inf*0.  Copied from GeGLU block pattern.
স্থির অন্তঃস্থ শূন্য vec_gelu(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    ভিন্ন one       = 1.0f;
    ভিন্ন half      = 0.5f;
    ভিন্ন coef_a    = 0.044715f;
    ভিন্ন sqrt2pi   = 0.79788456080286535587989211986876f;
    ভিন্ন two_log2e = 2.8853900817779268f;  // 2 * log2(e)
    বাবে (int32_t i = 0; i < n; i += 8) {
        __asm__ চঞ্চল(
            "flw.ps  f10, %[x]\n"
            "fbc.ps  f20, %[one]\n"
            "fbc.ps  f21, %[half]\n"
            "fbc.ps  f22, %[coef]\n"
            "fbc.ps  f23, %[s2pi]\n"
            "fbc.ps  f24, %[tl2e]\n"
            // inner = 1 + 0.044715 * x^2
            "fmul.ps f12, f10, f10\n"        // x^2
            "fmadd.ps f13, f22, f12, f20\n"  // 1 + 0.044715*x^2
            // z = sqrt(2/pi) * x * inner
            "fmul.ps f14, f23, f10\n"  // sqrt(2/pi) * x
            "fmul.ps f14, f14, f13\n"  // z
            // exp(2z) via fexp.ps
            "fmul.ps f15, f14, f24\n"  // 2z * log2(e)
            "fexp.ps f15, f15\n"       // exp(2z)
            // gelu(x) = 0.5 * x * (1 + tanh(z))
            //         = 0.5 * x * (1 + 1 - 2/(exp(2z)+1))
            //         = x * (1 - 1/(exp(2z)+1))  ... wait, that's tanh-based
            // Actually: 0.5*x*(1 + tanh) = 0.5*x*(1 + 1 - 2/(e2z+1)) = x*(1 - 1/(e2z+1))
            // Hmm: tanh = (e2z-1)/(e2z+1) = 1 - 2/(e2z+1)
            // So 0.5*(1+tanh) = 0.5*(2 - 2/(e2z+1)) = 1 - 1/(e2z+1)
            // gelu = x * (1 - 1/(e2z+1))  -- matches GeGLU pattern exactly
            "fadd.ps f16, f15, f20\n"  // exp(2z) + 1
            "frcp.ps f16, f16\n"       // 1/(exp(2z) + 1)
            "fsub.ps f16, f20, f16\n"  // 1 - 1/(exp(2z)+1) = sigmoid(2z)
            "fmul.ps f17, f10, f16\n"  // x * sigmoid(2z) = gelu(x)
            "fsw.ps  f17, %[r]\n"
            : [r] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src[i]), [one] "m"(one), [half] "m"(half), [coef] "m"(coef_a),
              [s2pi] "m"(sqrt2pi), [tl2e] "m"(two_log2e)
            : "f10", "f12", "f13", "f14", "f15", "f16", "f17", "f20", "f21", "f22", "f23", "f24");
    }
}

// GELU_QUICK: x * sigmoid(1.702 * x) = x / (1 + exp(-1.702*x))
স্থির অন্তঃস্থ শূন্য vec_gelu_quick(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    ভিন্ন one            = 1.0f;
    // -1.702 * log2(e) precomputed
    ভিন্ন neg_coef_log2e = -1.702f * 1.4426950408889634f;  // ~ -2.4542
    বাবে (int32_t i = 0; i < n; i += 8) {
        __asm__ চঞ্চল(
            "flw.ps  f10, %[x]\n"
            "fbc.ps  f20, %[one]\n"
            "fbc.ps  f21, %[ncl2e]\n"
            // exp(-1.702*x): feed -1.702*x*log2(e) = x * (-1.702*log2(e))
            "fmul.ps f12, f10, f21\n"  // x * (-1.702*log2(e))
            "fexp.ps f13, f12\n"       // exp(-1.702*x)
            "fadd.ps f14, f13, f20\n"  // 1 + exp(-1.702*x)
            "frcp.ps f15, f14\n"       // sigmoid(1.702*x)
            "fmul.ps f16, f10, f15\n"  // x * sigmoid(1.702*x)
            "fsw.ps  f16, %[r]\n"
            : [r] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src[i]), [one] "m"(one), [ncl2e] "m"(neg_coef_log2e)
            : "f10", "f12", "f13", "f14", "f15", "f16", "f20", "f21");
    }
}

// GELU_ERF: 0.5 * x * (1 + erf(x / sqrt(2)))
// erf approximation (Abramowitz & Stegun) is hard to vectorize cleanly, keep scalar
// but use et_expf for the exp(-z^2) part
স্থির অন্তঃস্থ শূন্য vec_gelu_erf(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    ধ্রুৱ ভিন্ন SQRT_2_INV = 0.70710678118654752440084436210484f;
    বাবে (int32_t i = 0; i < n; i++) {
        ভিন্ন x  = src[i];
        ভিন্ন z  = x * SQRT_2_INV;
        ভিন্ন az = z < 0.0f ? -z : z;

        ভিন্ন t  = et_fdiv(1.0f, 1.0f + 0.3275911f * az);
        ভিন্ন t2 = t * t;
        ভিন্ন t3 = t2 * t;
        ভিন্ন t4 = t3 * t;
        ভিন্ন t5 = t4 * t;

        ভিন্ন poly = 0.254829592f * t - 0.284496736f * t2 + 1.421413741f * t3 - 1.453152027f * t4 + 1.061405429f * t5;

        ভিন্ন erf_pos = 1.0f - poly * et_expf(-(az * az));
        ভিন্ন erf_val = (z < 0.0f) ? -erf_pos : erf_pos;
        dst[i]        = 0.5f * x * (1.0f + erf_val);
    }
}

// HARDSIGMOID: min(1, max(0, (x + 3) / 6))
// Vector: compute (x+3)/6 via frcp, then clamp with fmax(0) and fmin(1)
স্থির অন্তঃস্থ শূন্য vec_hardsigmoid(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    ভিন্ন zero  = 0.0f;
    ভিন্ন one   = 1.0f;
    ভিন্ন three = 3.0f;
    ভিন্ন inv6  = 0.16666666666666666f;  // 1/6
    বাবে (int32_t i = 0; i < n; i += 8) {
        __asm__ চঞ্চল(
            "flw.ps  f10, %[x]\n"
            "fbc.ps  f20, %[z]\n"
            "fbc.ps  f21, %[one]\n"
            "fbc.ps  f22, %[thr]\n"
            "fbc.ps  f23, %[inv]\n"
            "fadd.ps f12, f10, f22\n"  // x + 3
            "fmul.ps f13, f12, f23\n"  // (x + 3) / 6
            "fmax.ps f14, f13, f20\n"  // max(0, ...)
            "fmin.ps f15, f14, f21\n"  // min(1, ...)
            "fsw.ps  f15, %[r]\n"
            : [r] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src[i]), [z] "m"(zero), [one] "m"(one), [thr] "m"(three), [inv] "m"(inv6)
            : "f10", "f12", "f13", "f14", "f15", "f20", "f21", "f22", "f23");
    }
}

// HARDSWISH: x * hardsigmoid(x) = x * min(1, max(0, (x+3)/6))
স্থির অন্তঃস্থ শূন্য vec_hardswish(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    ভিন্ন zero  = 0.0f;
    ভিন্ন one   = 1.0f;
    ভিন্ন three = 3.0f;
    ভিন্ন inv6  = 0.16666666666666666f;
    বাবে (int32_t i = 0; i < n; i += 8) {
        __asm__ চঞ্চল(
            "flw.ps  f10, %[x]\n"
            "fbc.ps  f20, %[z]\n"
            "fbc.ps  f21, %[one]\n"
            "fbc.ps  f22, %[thr]\n"
            "fbc.ps  f23, %[inv]\n"
            "fadd.ps f12, f10, f22\n"  // x + 3
            "fmul.ps f13, f12, f23\n"  // (x + 3) / 6
            "fmax.ps f14, f13, f20\n"  // max(0, ...)
            "fmin.ps f15, f14, f21\n"  // min(1, ...)
            "fmul.ps f16, f10, f15\n"  // x * hardsigmoid(x)
            "fsw.ps  f16, %[r]\n"
            : [r] "=m"(*(ভিন্ন (*)[8]) & dst[i])
            : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & src[i]), [z] "m"(zero), [one] "m"(one), [thr] "m"(three), [inv] "m"(inv6)
            : "f10", "f12", "f13", "f14", "f15", "f16", "f20", "f21", "f22", "f23");
    }
}

// FLOOR: largest integer <= x
স্থির অন্তঃস্থ শূন্য vec_floor(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    বাবে (int32_t i = 0; i < n; i++) {
        ভিন্ন x = src[i];
        ভিন্ন t = (ভিন্ন) (int32_t) x;
        dst[i]  = (t > x) ? t - 1.0f : t;
    }
}

// CEIL: smallest integer >= x
স্থির অন্তঃস্থ শূন্য vec_ceil(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    বাবে (int32_t i = 0; i < n; i++) {
        ভিন্ন x = src[i];
        ভিন্ন t = (ভিন্ন) (int32_t) x;
        dst[i]  = (t < x) ? t + 1.0f : t;
    }
}

// TRUNC: round towards zero
স্থির অন্তঃস্থ শূন্য vec_trunc(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    বাবে (int32_t i = 0; i < n; i++) {
        dst[i] = (ভিন্ন) (int32_t) src[i];
    }
}

// ROUND: round to nearest, ties to even (banker's rounding)
স্থির অন্তঃস্থ শূন্য vec_round(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    বাবে (int32_t i = 0; i < n; i++) {
        ভিন্ন x    = src[i];
        ভিন্ন t    = (ভিন্ন) (int32_t) x;
        ভিন্ন diff = x - t;
        যদি (diff > 0.5f || (diff == 0.5f && ((int32_t) t & 1))) {
            t += 1.0f;
        } অন্যথা যদি (diff < -0.5f || (diff == -0.5f && ((int32_t) t & 1))) {
            t -= 1.0f;
        }
        dst[i] = t;
    }
}

// SOFTPLUS: log(1 + exp(x))
// For large x (>20), softplus(x) ~ x.  For moderate x, use fexp + flog.
// Scalar fallback since flog.ps computes log2, need conversion, and overflow guard
স্থির অন্তঃস্থ শূন্য vec_softplus(ভিন্ন * dst, ধ্রুৱ ভিন্ন * src, int32_t n) {
    বাবে (int32_t i = 0; i < n; i++) {
        ভিন্ন x = src[i];
        dst[i]  = (x > 20.0f) ? x : et_logf(1.0f + et_expf(x));
    }
}

স্থির অন্তঃস্থ size_t tensor_bytes(ধ্রুৱ সংরচনা ggml_tensor * t) {
    ঘূৰাই (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

//******************************************************************************
// Main entry point
//******************************************************************************

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_unary_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ঘূৰাই -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ঘূৰাই 0;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ঘূৰাই -1;
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * dst  = &params->dst;

    // evict_region_past_l2(&params->unary_op, sizeof(int32_t));
    // WAIT_CACHEOPS;
    // FENCE;

    int32_t unary_op = params->unary_op;

    যদি (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ঘূৰাই -1;
    }

    ভিন্ন * src0_data = (ভিন্ন *) src0->data;
    ভিন্ন * dst_data  = (ভিন্ন *) dst->data;

    যদি (!src0_data || !dst_data) {
        ঘূৰাই -1;
    }

    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(dst_data, tensor_bytes(dst));
    // WAIT_CACHEOPS;
    // FENCE;
    // et_barrier(ET_BARRIER_GLOBAL);

    // Tensor layout: src and dst are F32 with at least dim-0 contiguity
    //   - nb[0] == sizeof(float) (rows are dense; SIMD loads stay legal)
    //   - nb[1], nb[2], nb[3] may all be arbitrary strides for 4D views
    //
    // We walk rows independently and decompose row index r into (i1,i2,i3),
    // computing per-row byte offsets via nb[1..3] of each tensor.
    ধ্রুৱ int64_t nc             = dst->ne[0];  // row width (logical)
    ধ্রুৱ int64_t ne1            = dst->ne[1];
    ধ্রুৱ int64_t ne2            = dst->ne[2];
    ধ্রুৱ int64_t nr             = ne1 * ne2 * dst->ne[3];  // total rows
    ধ্রুৱ int64_t total_elements = nr * nc;
    ধ্রুৱ size_t  s_nb1 = src0->nb[1], s_nb2 = src0->nb[2], s_nb3 = src0->nb[3];
    ধ্রুৱ size_t  d_nb1 = dst->nb[1], d_nb2 = dst->nb[2], d_nb3 = dst->nb[3];

    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(dst_data, tensor_bytes(dst));
    // FENCE;
    // WAIT_CACHEOPS;
    // et_barrier(ET_BARRIER_GLOBAL);
    ধ্রুৱ int64_t elements_per_cacheline = 16;  // 64 bytes / 4 bytes per float
    ধ্রুৱ int64_t total_cachelines       = (total_elements + elements_per_cacheline - 1) / elements_per_cacheline;

    ধ্রুৱ int64_t cl_per_thread = (total_cachelines + অঙ্ক_threads - 1) / অঙ্ক_threads;
    ধ্রুৱ int64_t cl_start      = thread_id * cl_per_thread;
    int64_t       cl_end        = cl_start + cl_per_thread;
    যদি (cl_end > total_cachelines) {
        cl_end = total_cachelines;
    }

    যদি (cl_start >= total_cachelines) {
        ঘূৰাই 0;
    }

    ধ্রুৱ int64_t elem_start = cl_start * elements_per_cacheline;
    int64_t       elem_end   = cl_end * elements_per_cacheline;
    যদি (elem_end > total_elements) {
        elem_end = total_elements;
    }

    // Fast path: tensor is fully contiguous (no view), walk it as a flat array.
    // This preserves perf for the common case and avoids the per-row dispatch loop.
    ধ্রুৱ size_t row_bytes = (size_t) nc * পরিমাণ(ভিন্ন);
    // evict_region_past_l2((src0_data + elem_start), row_bytes);
    // // evict_region_past_l2((dst_data + elem_start), row_bytes);
    // FENCE;
    // WAIT_CACHEOPS;
    // et_barrier(ET_BARRIER_GLOBAL);

    ধ্রুৱ পূর্ণাঙ্ক is_flat = s_nb1 == row_bytes && s_nb2 == s_nb1 * (size_t) ne1 && s_nb3 == s_nb2 * (size_t) ne2 &&
                        d_nb1 == row_bytes && d_nb2 == d_nb1 * (size_t) ne1 && d_nb3 == d_nb2 * (size_t) ne2;

    যদি (is_flat) {
        ভিন্ন *       src_ptr = src0_data + elem_start;
        // evict_region_past_l2(src_ptr, 1024);
        ভিন্ন *       dst_ptr = dst_data + elem_start;
        ধ্রুৱ int32_t গুনতি   = (int32_t) (elem_end - elem_start);
        ৱিকল্প (unary_op) {
            পক্ষ GGML_UNARY_OP_NEG:
                vec_neg(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_ABS:
                vec_abs(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_SGN:
                vec_sgn(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_STEP:
                vec_step(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_RELU:
                vec_relu(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_EXP:
                vec_exp(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_EXPM1:
                vec_expm1(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_SIGMOID:
                vec_sigmoid(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_TANH:
                vec_tanh(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_SILU:
                vec_silu(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_ELU:
                vec_elu(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_GELU:
                vec_gelu(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_GELU_QUICK:
                vec_gelu_quick(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_GELU_ERF:
                vec_gelu_erf(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_HARDSWISH:
                vec_hardswish(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_HARDSIGMOID:
                vec_hardsigmoid(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_SOFTPLUS:
                vec_softplus(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_FLOOR:
                vec_floor(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_CEIL:
                vec_ceil(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_ROUND:
                vec_round(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_TRUNC:
                vec_trunc(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            স্ৱভাৱ:
                ঘূৰাই -1;
        }
        ঘূৰাই 0;
    }

    // Slow path: arbitrary 4D-strided view. Walk the assigned element range
    // row-by-row, clipping each segment to a row boundary so we never cross
    // nb[1]. For each row index r, decompose into (i1,i2,i3) and add the
    // corresponding nb[*] byte offsets to the base pointers.
    int64_t e = elem_start;
    যেতিয়ালৈ (e < elem_end) {
        int64_t row  = e / nc;
        int64_t col  = e % nc;
        int64_t take = nc - col;
        যদি (take > elem_end - e) {
            take = elem_end - e;
        }

        // Decompose row into (i3,i2,i1) using row-major linearization
        ধ্রুৱ int64_t i1 = row % ne1;
        ধ্রুৱ int64_t r2 = row / ne1;
        ধ্রুৱ int64_t i2 = r2 % ne2;
        ধ্রুৱ int64_t i3 = r2 / ne2;

        ভিন্ন *       src_ptr = (ভিন্ন *) ((ৱর্ণ *) src0_data + i3 * s_nb3 + i2 * s_nb2 + i1 * s_nb1) + col;
        ভিন্ন *       dst_ptr = (ভিন্ন *) ((ৱর্ণ *) dst_data + i3 * d_nb3 + i2 * d_nb2 + i1 * d_nb1) + col;
        ধ্রুৱ int32_t গুনতি   = (int32_t) take;

        // evict_region_past_l2(src_ptr, 1024);
        // FENCE;
        // et_barrier(ET_BARRIER_GLOBAL);

        ৱিকল্প (unary_op) {
            পক্ষ GGML_UNARY_OP_NEG:
                vec_neg(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_ABS:
                vec_abs(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_SGN:
                vec_sgn(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_STEP:
                vec_step(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_RELU:
                vec_relu(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_EXP:
                vec_exp(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_EXPM1:
                vec_expm1(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_SIGMOID:
                vec_sigmoid(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_TANH:
                vec_tanh(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_SILU:
                vec_silu(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_ELU:
                vec_elu(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_GELU:
                vec_gelu(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_GELU_QUICK:
                vec_gelu_quick(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_GELU_ERF:
                vec_gelu_erf(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_HARDSWISH:
                vec_hardswish(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_HARDSIGMOID:
                vec_hardsigmoid(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_SOFTPLUS:
                vec_softplus(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_FLOOR:
                vec_floor(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_CEIL:
                vec_ceil(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_ROUND:
                vec_round(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            পক্ষ GGML_UNARY_OP_TRUNC:
                vec_trunc(dst_ptr, src_ptr, গুনতি);
                ভঙ্গ;
            স্ৱভাৱ:
                ঘূৰাই -1;
        }

        e += take;
    }

    ঘূৰাই 0;
}
