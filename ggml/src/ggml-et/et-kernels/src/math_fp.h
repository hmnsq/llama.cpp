//******************************************************************************
// ET Floating Point Math Library
// Provides ET hardware-specific math functions, FP16 conversion, and trig functions
// for bare metal kernels
//******************************************************************************

#اگر_غیر_تعریف_شدہ MATH_FP_H
#تعریف_کرو MATH_FP_H

#شامل <stdint.h>

//******************************************************************************
// ET Hardware Math Functions
//******************************************************************************

// ET hardware division function (uses FRCP.PS instruction)
ساکن اندرونی اعشاری et_fdiv(اعشاری a, اعشاری b) {
    اعشاری         d;
    غیرعلامتی طویل عارضی;

    __asm__ متلون(
        "mova.x.m  %[temp]              \n\t"
        "mov.m.x   m0, x0, 1            \n\t"
        "frcp.ps   %[d], %[b]           \n\t"
        "fmul.s    %[d], %[d], %[a]     \n\t"
        "mova.m.x  %[temp]              \n\t"
        : [عارضی] "=&r"(عارضی), [d] "=&f"(d)
        : [a] "f"(a), [b] "f"(b));

    واپس d;
}

// Power function using ET hardware vector instructions
// Implements pow(base, exp) = exp(exp * ln(base)) using FLOG.PS and FEXP.PS
ساکن اندرونی اعشاری et_powf(اعشاری بنیاد, اعشاری قوت_نما) {
    // Handle special cases
    اگر (بنیاد <= 0.0f) {
        اگر (بنیاد == 0.0f) {
            اگر (قوت_نما > 0.0f) {
                واپس 0.0f;
            }

            // For exp <= 0, return +infinity (IEEE 754: sign=0, exp=0xFF, mantissa=0)
            اتحاد {
                اعشاری    f;
                uint32_t i;
            } inf = { .i = 0x7F800000 };

            واپس inf.f;
        }

        // For negative base, return NaN (IEEE 754: exp=0xFF, mantissa!=0)
        اتحاد {
            اعشاری    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        واپس nan.f;
    }
    اگر (بنیاد == 1.0f) {
        واپس 1.0f;
    }
    اگر (قوت_نما == 0.0f) {
        واپس 1.0f;
    }
    اگر (قوت_نما == 1.0f) {
        واپس بنیاد;
    }

    // Use ET hardware instructions following DNN library pattern:
    // pow(base, exp) = exp(exp * ln(base))
    اعشاری         نتیجہ;
    غیرعلامتی طویل عارضی;

    __asm__ متلون(
        "mova.x.m  %[temp]              \n\t"      // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"      // Set mask register m0 to enable element 0
        "flog.ps %[result], %[base]     \n\t"      // result = ln(base)
        "fmul.s %[result], %[result], %[exp]\n\t"  // result = ln(base) * exp
        "fexp.ps %[result], %[result]   \n\t"      // result = exp(ln(base) * exp) = base^exp
        "mova.m.x  %[temp]              \n\t"      // Restore mask state
        : [عارضی] "=&r"(عارضی), [نتیجہ] "=&f"(نتیجہ)
        : [بنیاد] "f"(بنیاد), [قوت_نما] "f"(قوت_نما));

    واپس نتیجہ;
}

// Natural logarithm.
ساکن اندرونی اعشاری et_logf(اعشاری x) {
    // Handle special cases
    اگر (x < 0.0f) {
        // Return NaN for negative input
        اتحاد {
            اعشاری    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        واپس nan.f;
    }
    اگر (x == 0.0f) {
        // Return -infinity for log(0)
        اتحاد {
            اعشاری    f;
            uint32_t i;
        } inf = { .i = 0xFF800000 };

        واپس inf.f;
    }
    اگر (x == 1.0f) {
        واپس 0.0f;
    }

    اعشاری         log2_نتیجہ;
    غیرعلامتی طویل عارضی;

    __asm__ متلون(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "flog.ps %[result], %[x]        \n\t"  // result = log2(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [عارضی] "=&r"(عارضی), [نتیجہ] "=&f"(log2_نتیجہ)
        : [x] "f"(x));

    // Convert log2 to natural log: ln(x) = log2(x) * ln(2)
    ثابت اعشاری ln2 = 0.69314718055994530942f;
    واپس log2_نتیجہ * ln2;
}

// Square root function implemented as et_powf(x, 0.5)
ساکن اندرونی اعشاری et_sqrtf(اعشاری x) {
    // Handle special cases
    اگر (x < 0.0f) {
        // Return NaN for negative input (IEEE 754: exp=0xFF, mantissa!=0)
        اتحاد {
            اعشاری    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        واپس nan.f;
    }
    اگر (x == 0.0f) {
        واپس 0.0f;
    }

    واپس et_powf(x, 0.5f);
}

// Base-2 exponential: returns 2^x using the ET hardware FEXP.PS instruction.
// No base conversion, no special-case clamping — this is the raw hardware op
// with just the mask save/restore wrapper.  Caller is responsible for ensuring
// x is in a range that produces a useful result (roughly [-126, 128] for fp32).
ساکن اندرونی اعشاری __attribute__((always_inline)) et_exp2f(اعشاری x) {
    غیرعلامتی طویل old_نقاب;
    اعشاری         برآمد;
    __asm__ متلون(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 1         \n\t"
        "fexp.ps   %[out], %[x]      \n\t"
        "mova.m.x  %[ms]             \n\t"
        : [ms] "=&r"(old_نقاب), [برآمد] "=&f"(برآمد)
        : [x] "f"(x));
    واپس برآمد;
}

// Exponential function using ET hardware FEXP.PS instruction
// Note: FEXP.PS computes 2^x, so we need to convert: exp(x) = 2^(x * log2(e))
ساکن اندرونی اعشاری et_expf(اعشاری x) {
    // Handle special cases
    اگر (x > 88.0f) {
        // For x > 88, exp(x) would overflow, return +infinity
        اتحاد {
            اعشاری    f;
            uint32_t i;
        } inf = { .i = 0x7F800000 };

        واپس inf.f;
    }
    اگر (x < -87.0f) {
        // For x < -87, exp(x) is essentially 0
        واپس 0.0f;
    }

    // Convert to base-2 exponent: x * log2(e)
    ثابت اعشاری log2e   = 1.4426950408889634f;  // log2(e)
    اعشاری       x_log2e = x * log2e;

    // Use ET hardware instruction: fexp.ps computes 2^x
    اعشاری         نتیجہ;
    غیرعلامتی طویل عارضی;

    __asm__ متلون(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fexp.ps %[result], %[x_log2e]  \n\t"  // result = 2^(x * log2(e)) = exp(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [عارضی] "=&r"(عارضی), [نتیجہ] "=&f"(نتیجہ)
        : [x_log2e] "f"(x_log2e));

    واپس نتیجہ;
}

//******************************************************************************
// Trigonometric Functions
//******************************************************************************

// FSIN.PS

// Sine function using Taylor series
ساکن اندرونی اعشاری et_sinf(اعشاری x) {
    ثابت اعشاری pi        = 3.14159265358979323846f;
    ثابت اعشاری two_pi    = 6.28318530717958647693f;
    ثابت اعشاری pi_over_2 = 1.57079632679489661923f;

    اگر (x > pi || x < -pi) {
        اعشاری ادوار = x * et_fdiv(1.0f, two_pi);
        صحیح   n      = (صحیح) ادوار;
        اگر (x < 0.0f) {
            n--;  // Floor for negative
        }
        x = x - (اعشاری) n * two_pi;
    }

    // sin(x) = sin(π - x) for x in [π/2, π]
    // sin(x) = -sin(-π - x) for x in [-π, -π/2]
    صحیح negate = 0;
    اگر (x > pi_over_2) {
        x = pi - x;
    } ورنہ اگر (x < -pi_over_2) {
        x      = -pi - x;
        negate = 1;
    }

    // sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    ثابت اعشاری x2  = x * x;
    ثابت اعشاری x3  = x2 * x;
    ثابت اعشاری x5  = x3 * x2;
    ثابت اعشاری x7  = x5 * x2;
    ثابت اعشاری x9  = x7 * x2;
    ثابت اعشاری x11 = x9 * x2;

    اعشاری نتیجہ = x - x3 * et_fdiv(1.0f, 6.0f)         // x^3/3!
                   + x5 * et_fdiv(1.0f, 120.0f)         // x^5/5!
                   - x7 * et_fdiv(1.0f, 5040.0f)        // x^7/7!
                   + x9 * et_fdiv(1.0f, 362880.0f)      // x^9/9!
                   - x11 * et_fdiv(1.0f, 39916800.0f);  // x^11/11!

    واپس negate ? -نتیجہ : نتیجہ;
}

// Cosine function using identity cos(x) = sin(x + π/2)
ساکن اندرونی اعشاری et_cosf(اعشاری x) {
    ثابت اعشاری pi_over_2 = 1.57079632679489661923f;
    واپس et_sinf(x + pi_over_2);
}

//******************************************************************************
// FP16 <-> FP32 Conversion Functions
//******************************************************************************

// Convert FP16 (IEEE 754 half precision) to FP32 (single precision)
// Uses ET hardware FCVT.PS.F16 instruction for accurate conversion
ساکن اندرونی اعشاری fp16_مقصد_fp32(uint16_t h) {
    اعشاری         نتیجہ;
    غیرعلامتی طویل عارضی;
    uint32_t      raw = (uint32_t) h;

    __asm__ متلون(
        "mova.x.m  %[temp]              \n\t"    // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"    // Set mask register m0 to enable element 0
        "fbcx.ps   %[result], %[raw]    \n\t"    // Broadcast raw FP16 bits into vector register
        "fcvt.ps.f16 %[result], %[result] \n\t"  // Convert FP16 to FP32
        "mova.m.x  %[temp]              \n\t"    // Restore mask state
        : [عارضی] "=&r"(عارضی), [نتیجہ] "=&f"(نتیجہ)
        : [raw] "r"(raw));

    واپس نتیجہ;
}

// Convert FP32 (single precision) to FP16 (IEEE 754 half precision)
// Uses ET hardware FCVT.F16.PS instruction for accurate conversion
ساکن اندرونی uint16_t fp32_مقصد_fp16(اعشاری f) {
    اعشاری         نتیجہ_f;
    غیرعلامتی طویل عارضی;

    __asm__ متلون(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fcvt.f16.ps %[result], %[f]    \n\t"  // Convert FP32 to FP16 (result in lower 16 bits)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [عارضی] "=&r"(عارضی), [نتیجہ] "=&f"(نتیجہ_f)
        : [f] "f"(f));

    // Extract lower 16 bits containing the FP16 value
    // The instruction zero-extends to 32 bits, so upper 16 bits are 0
    uint32_t نتیجہ_بٹ = *(uint32_t *) &نتیجہ_f;
    واپس (uint16_t) نتیجہ_بٹ;
}

#اگر_ختم  // MATH_FP_H
