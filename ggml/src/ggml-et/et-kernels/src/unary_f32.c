//******************************************************************************
// Unary F32 Kernel
// Element-wise unary operations: dst[i] = f(src0[i])
// All ops vectorized using 8-wide ET SIMD (fexp.ps, frcp.ps, flog.ps, etc.)
//
// Supports: ABS, SGN, NEG, STEP, TANH, ELU, RELU, SIGMOID, GELU, GELU_QUICK,
//           SILU, HARDSWISH, HARDSIGMOID, EXP, EXPM1, SOFTPLUS, GELU_ERF
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <stdint.h>

// Unary kernel parameters structure
ساخت ggml_et_unary_params {
    ساخت ggml_tensor src0;      // F32 input tensor
    ساخت ggml_tensor dst;       // F32 output tensor
    int32_t            unary_op;  // ggml_unary_op enum value
};

//******************************************************************************
// Vectorized 8-wide block operations
// All process exactly 8 floats per call using ET vector instructions.
// ne0 is guaranteed % 16 == 0, so the inner loop always calls with i0 += 8.
//******************************************************************************

// NEG: dst = -x  (zero - x)
ساکن اندرونی خالی vec_neg(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    اعشاری zero = 0.0f;
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
            "fbc.ps  f10, %[z]\n"
            "flw.ps  f11, %[x]\n"
            "fsub.ps f12, f10, f11\n"
            "fsw.ps  f12, %[r]\n"
            : [r] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [x] "m"(*(ثابت اعشاری (*)[8]) & src[i]), [z] "m"(zero)
            : "f10", "f11", "f12");
    }
}

// ABS: dst = |x|  (negate negative values: abs = x * sgn, or max(x, -x))
// Uses: negate then fmax.ps
ساکن اندرونی خالی vec_abs(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    اعشاری zero = 0.0f;
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
            "fbc.ps  f10, %[z]\n"
            "flw.ps  f11, %[x]\n"
            "fsub.ps f12, f10, f11\n"  // f12 = -x
            "fmax.ps f13, f11, f12\n"  // f13 = max(x, -x) = |x|
            "fsw.ps  f13, %[r]\n"
            : [r] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [x] "m"(*(ثابت اعشاری (*)[8]) & src[i]), [z] "m"(zero)
            : "f10", "f11", "f12", "f13");
    }
}

// RELU: dst = max(0, x)
ساکن اندرونی خالی vec_relu(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    اعشاری zero = 0.0f;
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
            "fbc.ps  f10, %[z]\n"
            "flw.ps  f11, %[x]\n"
            "fmax.ps f12, f10, f11\n"  // max(0, x)
            "fsw.ps  f12, %[r]\n"
            : [r] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [x] "m"(*(ثابت اعشاری (*)[8]) & src[i]), [z] "m"(zero)
            : "f10", "f11", "f12");
    }
}

// STEP: dst = x > 0 ? 1 : 0  (clamp to [0,1] via max then min-ish, or use sign bit)
// Trick: relu(x) then frcp gives inf for 0 and finite for >0, but simpler:
// step(x) = min(1, relu(x) * huge) ... too fragile.  Scalar is fine for step/sgn.
ساکن اندرونی خالی vec_قدم(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    برائے (int32_t i = 0; i < n; i++) {
        dst[i] = (src[i] > 0.0f) ? 1.0f : 0.0f;
    }
}

// SGN: dst = sign(x) = x>0 ? 1 : (x<0 ? -1 : 0)
ساکن اندرونی خالی vec_sgn(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    برائے (int32_t i = 0; i < n; i++) {
        dst[i] = (src[i] > 0.0f) ? 1.0f : ((src[i] < 0.0f) ? -1.0f : 0.0f);
    }
}

// EXP: dst = exp(x)
// fexp.ps computes 2^x, so feed x * log2(e)
ساکن اندرونی خالی vec_exp(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    اعشاری log2e = 1.4426950408889634f;
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
            "flw.ps  f10, %[x]\n"
            "fbc.ps  f11, %[l2e]\n"
            "fmul.ps f12, f10, f11\n"  // x * log2(e)
            "fexp.ps f13, f12\n"       // 2^(x*log2e) = exp(x)
            "fsw.ps  f13, %[r]\n"
            : [r] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [x] "m"(*(ثابت اعشاری (*)[8]) & src[i]), [l2e] "m"(log2e)
            : "f10", "f11", "f12", "f13");
    }
}

// EXPM1: dst = exp(x) - 1
ساکن اندرونی خالی vec_expm1(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    اعشاری log2e = 1.4426950408889634f;
    اعشاری one   = 1.0f;
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
            "flw.ps  f10, %[x]\n"
            "fbc.ps  f11, %[l2e]\n"
            "fbc.ps  f14, %[one]\n"
            "fmul.ps f12, f10, f11\n"  // x * log2(e)
            "fexp.ps f13, f12\n"       // exp(x)
            "fsub.ps f13, f13, f14\n"  // exp(x) - 1
            "fsw.ps  f13, %[r]\n"
            : [r] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [x] "m"(*(ثابت اعشاری (*)[8]) & src[i]), [l2e] "m"(log2e), [one] "m"(one)
            : "f10", "f11", "f12", "f13", "f14");
    }
}

// SIGMOID: dst = 1 / (1 + exp(-x))
// Same pattern as SwiGLU: exp(-x) via fexp.ps, then frcp.ps
ساکن اندرونی خالی vec_sigmoid(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    اعشاری zero  = 0.0f;
    اعشاری one   = 1.0f;
    اعشاری log2e = 1.4426950408889634f;
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
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
            : [r] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [x] "m"(*(ثابت اعشاری (*)[8]) & src[i]), [z] "m"(zero), [one] "m"(one), [l2e] "m"(log2e)
            : "f10", "f12", "f13", "f14", "f15", "f16", "f20", "f21", "f22");
    }
}

// TANH: dst = (exp(2x) - 1) / (exp(2x) + 1)
// Rewrite as: 1 - 2/(exp(2x) + 1) to use frcp.ps
// Or equivalently: 2*sigmoid(2x) - 1
ساکن اندرونی خالی vec_tanh(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    اعشاری one       = 1.0f;
    اعشاری two       = 2.0f;
    اعشاری two_log2e = 2.8853900817779268f;  // 2 * log2(e)
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
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
            : [r] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [x] "m"(*(ثابت اعشاری (*)[8]) & src[i]), [one] "m"(one), [two] "m"(two), [tl2e] "m"(two_log2e)
            : "f10", "f12", "f13", "f14", "f15", "f16", "f17", "f20", "f21", "f22");
    }
}

// SILU: dst = x / (1 + exp(-x)) = x * sigmoid(x)
// Copied from SwiGLU pattern but without the gate multiply
ساکن اندرونی خالی vec_silu(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    اعشاری zero  = 0.0f;
    اعشاری one   = 1.0f;
    اعشاری log2e = 1.4426950408889634f;
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
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
            : [r] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [x] "m"(*(ثابت اعشاری (*)[8]) & src[i]), [z] "m"(zero), [one] "m"(one), [l2e] "m"(log2e)
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
ساکن اندرونی خالی vec_elu(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    اعشاری log2e = 1.4426950408889634f;
    اعشاری one   = 1.0f;
    // Compute exp(x)-1 vectorized, then fixup positive elements
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
            "flw.ps  f10, %[x]\n"
            "fbc.ps  f11, %[l2e]\n"
            "fbc.ps  f14, %[one]\n"
            "fmul.ps f12, f10, f11\n"  // x * log2(e)
            "fexp.ps f13, f12\n"       // exp(x)
            "fsub.ps f13, f13, f14\n"  // exp(x) - 1
            "fsw.ps  f13, %[r]\n"      // store exp(x)-1
            : [r] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [x] "m"(*(ثابت اعشاری (*)[8]) & src[i]), [l2e] "m"(log2e), [one] "m"(one)
            : "f10", "f11", "f12", "f13", "f14");
        // Fixup: for x > 0, dst = x
        برائے (int32_t j = 0; j < 8 && (i + j) < n; j++) {
            اگر (src[i + j] > 0.0f) {
                dst[i + j] = src[i + j];
            }
        }
    }
}

// GELU: 0.5*x*(1 + tanh(sqrt(2/pi) * x * (1 + 0.044715*x^2)))
// Reformulated as: x * (1 - 1/(exp(2z)+1)) where z = sqrt(2/pi)*x*(1+0.044715*x^2)
// NaN-safe: avoids inf*0.  Copied from GeGLU block pattern.
ساکن اندرونی خالی vec_gelu(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    اعشاری one       = 1.0f;
    اعشاری half      = 0.5f;
    اعشاری coef_a    = 0.044715f;
    اعشاری sqrt2pi   = 0.79788456080286535587989211986876f;
    اعشاری two_log2e = 2.8853900817779268f;  // 2 * log2(e)
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
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
            : [r] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [x] "m"(*(ثابت اعشاری (*)[8]) & src[i]), [one] "m"(one), [half] "m"(half), [coef] "m"(coef_a),
              [s2pi] "m"(sqrt2pi), [tl2e] "m"(two_log2e)
            : "f10", "f12", "f13", "f14", "f15", "f16", "f17", "f20", "f21", "f22", "f23", "f24");
    }
}

// GELU_QUICK: x * sigmoid(1.702 * x) = x / (1 + exp(-1.702*x))
ساکن اندرونی خالی vec_gelu_تیز(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    اعشاری one            = 1.0f;
    // -1.702 * log2(e) precomputed
    اعشاری neg_coef_log2e = -1.702f * 1.4426950408889634f;  // ~ -2.4542
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
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
            : [r] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [x] "m"(*(ثابت اعشاری (*)[8]) & src[i]), [one] "m"(one), [ncl2e] "m"(neg_coef_log2e)
            : "f10", "f12", "f13", "f14", "f15", "f16", "f20", "f21");
    }
}

// GELU_ERF: 0.5 * x * (1 + erf(x / sqrt(2)))
// erf approximation (Abramowitz & Stegun) is hard to vectorize cleanly, keep scalar
// but use et_expf for the exp(-z^2) part
ساکن اندرونی خالی vec_gelu_erf(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    ثابت اعشاری SQRT_2_INV = 0.70710678118654752440084436210484f;
    برائے (int32_t i = 0; i < n; i++) {
        اعشاری x  = src[i];
        اعشاری z  = x * SQRT_2_INV;
        اعشاری az = z < 0.0f ? -z : z;

        اعشاری t  = et_fdiv(1.0f, 1.0f + 0.3275911f * az);
        اعشاری t2 = t * t;
        اعشاری t3 = t2 * t;
        اعشاری t4 = t3 * t;
        اعشاری t5 = t4 * t;

        اعشاری poly = 0.254829592f * t - 0.284496736f * t2 + 1.421413741f * t3 - 1.453152027f * t4 + 1.061405429f * t5;

        اعشاری erf_pos = 1.0f - poly * et_expf(-(az * az));
        اعشاری erf_val = (z < 0.0f) ? -erf_pos : erf_pos;
        dst[i]        = 0.5f * x * (1.0f + erf_val);
    }
}

// HARDSIGMOID: min(1, max(0, (x + 3) / 6))
// Vector: compute (x+3)/6 via frcp, then clamp with fmax(0) and fmin(1)
ساکن اندرونی خالی vec_hardsigmoid(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    اعشاری zero  = 0.0f;
    اعشاری one   = 1.0f;
    اعشاری three = 3.0f;
    اعشاری inv6  = 0.16666666666666666f;  // 1/6
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
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
            : [r] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [x] "m"(*(ثابت اعشاری (*)[8]) & src[i]), [z] "m"(zero), [one] "m"(one), [thr] "m"(three), [inv] "m"(inv6)
            : "f10", "f12", "f13", "f14", "f15", "f20", "f21", "f22", "f23");
    }
}

// HARDSWISH: x * hardsigmoid(x) = x * min(1, max(0, (x+3)/6))
ساکن اندرونی خالی vec_hardswish(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    اعشاری zero  = 0.0f;
    اعشاری one   = 1.0f;
    اعشاری three = 3.0f;
    اعشاری inv6  = 0.16666666666666666f;
    برائے (int32_t i = 0; i < n; i += 8) {
        __asm__ متلون(
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
            : [r] "=m"(*(اعشاری (*)[8]) & dst[i])
            : [x] "m"(*(ثابت اعشاری (*)[8]) & src[i]), [z] "m"(zero), [one] "m"(one), [thr] "m"(three), [inv] "m"(inv6)
            : "f10", "f12", "f13", "f14", "f15", "f16", "f20", "f21", "f22", "f23");
    }
}

// FLOOR: largest integer <= x
ساکن اندرونی خالی vec_floor(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    برائے (int32_t i = 0; i < n; i++) {
        اعشاری x = src[i];
        اعشاری t = (اعشاری) (int32_t) x;
        dst[i]  = (t > x) ? t - 1.0f : t;
    }
}

// CEIL: smallest integer >= x
ساکن اندرونی خالی vec_ceil(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    برائے (int32_t i = 0; i < n; i++) {
        اعشاری x = src[i];
        اعشاری t = (اعشاری) (int32_t) x;
        dst[i]  = (t < x) ? t + 1.0f : t;
    }
}

// TRUNC: round towards zero
ساکن اندرونی خالی vec_trunc(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    برائے (int32_t i = 0; i < n; i++) {
        dst[i] = (اعشاری) (int32_t) src[i];
    }
}

// ROUND: round to nearest, ties to even (banker's rounding)
ساکن اندرونی خالی vec_باری(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    برائے (int32_t i = 0; i < n; i++) {
        اعشاری x    = src[i];
        اعشاری t    = (اعشاری) (int32_t) x;
        اعشاری diff = x - t;
        اگر (diff > 0.5f || (diff == 0.5f && ((int32_t) t & 1))) {
            t += 1.0f;
        } ورنہ اگر (diff < -0.5f || (diff == -0.5f && ((int32_t) t & 1))) {
            t -= 1.0f;
        }
        dst[i] = t;
    }
}

// SOFTPLUS: log(1 + exp(x))
// For large x (>20), softplus(x) ~ x.  For moderate x, use fexp + flog.
// Scalar fallback since flog.ps computes log2, need conversion, and overflow guard
ساکن اندرونی خالی vec_softplus(اعشاری * dst, ثابت اعشاری * src, int32_t n) {
    برائے (int32_t i = 0; i < n; i++) {
        اعشاری x = src[i];
        dst[i]  = (x > 20.0f) ? x : et_logf(1.0f + et_expf(x));
    }
}

ساکن اندرونی size_t tensor_بائٹیں(ثابت ساخت ggml_tensor * t) {
    واپس (size_t) t->ne[0] * t->ne[1] * t->ne[2] * t->ne[3] * t->nb[0];
}

//******************************************************************************
// Main entry point
//******************************************************************************

صحیح اندراج_نقطہ(ساخت ggml_et_unary_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس 0;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * dst  = &params->dst;

    // evict_region_past_l2(&params->unary_op, sizeof(int32_t));
    // WAIT_CACHEOPS;
    // FENCE;

    int32_t unary_op = params->unary_op;

    اگر (src0->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    اعشاری * src0_ڈیٹا = (اعشاری *) src0->ڈیٹا;
    اعشاری * dst_ڈیٹا  = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
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
    ثابت int64_t nc             = dst->ne[0];  // row width (logical)
    ثابت int64_t ne1            = dst->ne[1];
    ثابت int64_t ne2            = dst->ne[2];
    ثابت int64_t nr             = ne1 * ne2 * dst->ne[3];  // total rows
    ثابت int64_t کل_elements = nr * nc;
    ثابت size_t  s_nb1 = src0->nb[1], s_nb2 = src0->nb[2], s_nb3 = src0->nb[3];
    ثابت size_t  d_nb1 = dst->nb[1], d_nb2 = dst->nb[2], d_nb3 = dst->nb[3];

    // evict_region_past_l2(src0_data, tensor_bytes(src0));
    // evict_region_past_l2(dst_data, tensor_bytes(dst));
    // FENCE;
    // WAIT_CACHEOPS;
    // et_barrier(ET_BARRIER_GLOBAL);
    ثابت int64_t elements_فی_cacheline = 16;  // 64 bytes / 4 bytes per float
    ثابت int64_t کل_cachelines       = (کل_elements + elements_فی_cacheline - 1) / elements_فی_cacheline;

    ثابت int64_t cl_فی_thread = (کل_cachelines + عدد_دھاگے - 1) / عدد_دھاگے;
    ثابت int64_t cl_آغاز      = thread_id * cl_فی_thread;
    int64_t       cl_انتہا        = cl_آغاز + cl_فی_thread;
    اگر (cl_انتہا > کل_cachelines) {
        cl_انتہا = کل_cachelines;
    }

    اگر (cl_آغاز >= کل_cachelines) {
        واپس 0;
    }

    ثابت int64_t elem_آغاز = cl_آغاز * elements_فی_cacheline;
    int64_t       elem_انتہا   = cl_انتہا * elements_فی_cacheline;
    اگر (elem_انتہا > کل_elements) {
        elem_انتہا = کل_elements;
    }

    // Fast path: tensor is fully contiguous (no view), walk it as a flat array.
    // This preserves perf for the common case and avoids the per-row dispatch loop.
    ثابت size_t صف_بائٹیں = (size_t) nc * حجم(اعشاری);
    // evict_region_past_l2((src0_data + elem_start), row_bytes);
    // // evict_region_past_l2((dst_data + elem_start), row_bytes);
    // FENCE;
    // WAIT_CACHEOPS;
    // et_barrier(ET_BARRIER_GLOBAL);

    ثابت صحیح is_flat = s_nb1 == صف_بائٹیں && s_nb2 == s_nb1 * (size_t) ne1 && s_nb3 == s_nb2 * (size_t) ne2 &&
                        d_nb1 == صف_بائٹیں && d_nb2 == d_nb1 * (size_t) ne1 && d_nb3 == d_nb2 * (size_t) ne2;

    اگر (is_flat) {
        اعشاری *       src_ptr = src0_ڈیٹا + elem_آغاز;
        // evict_region_past_l2(src_ptr, 1024);
        اعشاری *       dst_ptr = dst_ڈیٹا + elem_آغاز;
        ثابت int32_t گنتی   = (int32_t) (elem_انتہا - elem_آغاز);
        انتخاب (unary_op) {
            حالت GGML_UNARY_OP_NEG:
                vec_neg(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_ABS:
                vec_abs(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_SGN:
                vec_sgn(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_قدم:
                vec_قدم(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_RELU:
                vec_relu(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_EXP:
                vec_exp(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_EXPM1:
                vec_expm1(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_SIGMOID:
                vec_sigmoid(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_TANH:
                vec_tanh(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_SILU:
                vec_silu(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_ELU:
                vec_elu(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_GELU:
                vec_gelu(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_GELU_تیز:
                vec_gelu_تیز(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_GELU_ERF:
                vec_gelu_erf(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_HARDSWISH:
                vec_hardswish(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_HARDSIGMOID:
                vec_hardsigmoid(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_SOFTPLUS:
                vec_softplus(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_FLOOR:
                vec_floor(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_CEIL:
                vec_ceil(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_باری:
                vec_باری(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_TRUNC:
                vec_trunc(dst_ptr, src_ptr, گنتی);
                توڑو;
            طےشدہ:
                واپس -1;
        }
        واپس 0;
    }

    // Slow path: arbitrary 4D-strided view. Walk the assigned element range
    // row-by-row, clipping each segment to a row boundary so we never cross
    // nb[1]. For each row index r, decompose into (i1,i2,i3) and add the
    // corresponding nb[*] byte offsets to the base pointers.
    int64_t e = elem_آغاز;
    جب (e < elem_انتہا) {
        int64_t صف  = e / nc;
        int64_t ستون  = e % nc;
        int64_t take = nc - ستون;
        اگر (take > elem_انتہا - e) {
            take = elem_انتہا - e;
        }

        // Decompose row into (i3,i2,i1) using row-major linearization
        ثابت int64_t i1 = صف % ne1;
        ثابت int64_t r2 = صف / ne1;
        ثابت int64_t i2 = r2 % ne2;
        ثابت int64_t i3 = r2 / ne2;

        اعشاری *       src_ptr = (اعشاری *) ((حرف *) src0_ڈیٹا + i3 * s_nb3 + i2 * s_nb2 + i1 * s_nb1) + ستون;
        اعشاری *       dst_ptr = (اعشاری *) ((حرف *) dst_ڈیٹا + i3 * d_nb3 + i2 * d_nb2 + i1 * d_nb1) + ستون;
        ثابت int32_t گنتی   = (int32_t) take;

        // evict_region_past_l2(src_ptr, 1024);
        // FENCE;
        // et_barrier(ET_BARRIER_GLOBAL);

        انتخاب (unary_op) {
            حالت GGML_UNARY_OP_NEG:
                vec_neg(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_ABS:
                vec_abs(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_SGN:
                vec_sgn(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_قدم:
                vec_قدم(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_RELU:
                vec_relu(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_EXP:
                vec_exp(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_EXPM1:
                vec_expm1(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_SIGMOID:
                vec_sigmoid(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_TANH:
                vec_tanh(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_SILU:
                vec_silu(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_ELU:
                vec_elu(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_GELU:
                vec_gelu(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_GELU_تیز:
                vec_gelu_تیز(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_GELU_ERF:
                vec_gelu_erf(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_HARDSWISH:
                vec_hardswish(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_HARDSIGMOID:
                vec_hardsigmoid(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_SOFTPLUS:
                vec_softplus(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_FLOOR:
                vec_floor(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_CEIL:
                vec_ceil(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_باری:
                vec_باری(dst_ptr, src_ptr, گنتی);
                توڑو;
            حالت GGML_UNARY_OP_TRUNC:
                vec_trunc(dst_ptr, src_ptr, گنتی);
                توڑو;
            طےشدہ:
                واپس -1;
        }

        e += take;
    }

    واپس 0;
}
