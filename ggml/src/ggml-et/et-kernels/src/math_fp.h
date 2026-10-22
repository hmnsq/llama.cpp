//******************************************************************************
// ET Floating Point Math Library
// Provides ET hardware-specific math functions, FP16 conversion, and trig functions
// for bare metal kernels
//******************************************************************************

#ifndef MATH_FP_H
#define MATH_FP_H

#অন্তর্ভাব <stdint.h>

//******************************************************************************
// ET Hardware Math Functions
//******************************************************************************

// ET hardware division function (uses FRCP.PS instruction)
স্থির অন্তঃস্থ ভিন্ন et_fdiv(ভিন্ন a, ভিন্ন b) {
    ভিন্ন         d;
    unsigned long temp;

    __asm__ চঞ্চল(
        "mova.x.m  %[temp]              \n\t"
        "mov.m.x   m0, x0, 1            \n\t"
        "frcp.ps   %[d], %[b]           \n\t"
        "fmul.s    %[d], %[d], %[a]     \n\t"
        "mova.m.x  %[temp]              \n\t"
        : [temp] "=&r"(temp), [d] "=&f"(d)
        : [a] "f"(a), [b] "f"(b));

    প্রত্যাগম d;
}

// Power function using ET hardware vector instructions
// Implements pow(base, exp) = exp(exp * ln(base)) using FLOG.PS and FEXP.PS
স্থির অন্তঃস্থ ভিন্ন et_powf(ভিন্ন base, ভিন্ন exp) {
    // Handle special cases
    যদি (base <= 0.0f) {
        যদি (base == 0.0f) {
            যদি (exp > 0.0f) {
                প্রত্যাগম 0.0f;
            }

            // For exp <= 0, return +infinity (IEEE 754: sign=0, exp=0xFF, mantissa=0)
            সঙ্ঘ {
                ভিন্ন    f;
                uint32_t i;
            } inf = { .i = 0x7F800000 };

            প্রত্যাগম inf.f;
        }

        // For negative base, return NaN (IEEE 754: exp=0xFF, mantissa!=0)
        সঙ্ঘ {
            ভিন্ন    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        প্রত্যাগম nan.f;
    }
    যদি (base == 1.0f) {
        প্রত্যাগম 1.0f;
    }
    যদি (exp == 0.0f) {
        প্রত্যাগম 1.0f;
    }
    যদি (exp == 1.0f) {
        প্রত্যাগম base;
    }

    // Use ET hardware instructions following DNN library pattern:
    // pow(base, exp) = exp(exp * ln(base))
    ভিন্ন         result;
    unsigned long temp;

    __asm__ চঞ্চল(
        "mova.x.m  %[temp]              \n\t"      // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"      // Set mask register m0 to enable element 0
        "flog.ps %[result], %[base]     \n\t"      // result = ln(base)
        "fmul.s %[result], %[result], %[exp]\n\t"  // result = ln(base) * exp
        "fexp.ps %[result], %[result]   \n\t"      // result = exp(ln(base) * exp) = base^exp
        "mova.m.x  %[temp]              \n\t"      // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result)
        : [base] "f"(base), [exp] "f"(exp));

    প্রত্যাগম result;
}

// Natural logarithm.
স্থির অন্তঃস্থ ভিন্ন et_logf(ভিন্ন x) {
    // Handle special cases
    যদি (x < 0.0f) {
        // Return NaN for negative input
        সঙ্ঘ {
            ভিন্ন    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        প্রত্যাগম nan.f;
    }
    যদি (x == 0.0f) {
        // Return -infinity for log(0)
        সঙ্ঘ {
            ভিন্ন    f;
            uint32_t i;
        } inf = { .i = 0xFF800000 };

        প্রত্যাগম inf.f;
    }
    যদি (x == 1.0f) {
        প্রত্যাগম 0.0f;
    }

    ভিন্ন         log2_result;
    unsigned long temp;

    __asm__ চঞ্চল(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "flog.ps %[result], %[x]        \n\t"  // result = log2(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(log2_result)
        : [x] "f"(x));

    // Convert log2 to natural log: ln(x) = log2(x) * ln(2)
    ধ্রুব ভিন্ন ln2 = 0.69314718055994530942f;
    প্রত্যাগম log2_result * ln2;
}

// Square root function implemented as et_powf(x, 0.5)
স্থির অন্তঃস্থ ভিন্ন et_sqrtf(ভিন্ন x) {
    // Handle special cases
    যদি (x < 0.0f) {
        // Return NaN for negative input (IEEE 754: exp=0xFF, mantissa!=0)
        সঙ্ঘ {
            ভিন্ন    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        প্রত্যাগম nan.f;
    }
    যদি (x == 0.0f) {
        প্রত্যাগম 0.0f;
    }

    প্রত্যাগম et_powf(x, 0.5f);
}

// Base-2 exponential: returns 2^x using the ET hardware FEXP.PS instruction.
// No base conversion, no special-case clamping — this is the raw hardware op
// with just the mask save/restore wrapper.  Caller is responsible for ensuring
// x is in a range that produces a useful result (roughly [-126, 128] for fp32).
স্থির অন্তঃস্থ ভিন্ন __attribute__((always_inline)) et_exp2f(ভিন্ন x) {
    unsigned long old_mask;
    ভিন্ন         out;
    __asm__ চঞ্চল(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 1         \n\t"
        "fexp.ps   %[out], %[x]      \n\t"
        "mova.m.x  %[ms]             \n\t"
        : [ms] "=&r"(old_mask), [out] "=&f"(out)
        : [x] "f"(x));
    প্রত্যাগম out;
}

// Exponential function using ET hardware FEXP.PS instruction
// Note: FEXP.PS computes 2^x, so we need to convert: exp(x) = 2^(x * log2(e))
স্থির অন্তঃস্থ ভিন্ন et_expf(ভিন্ন x) {
    // Handle special cases
    যদি (x > 88.0f) {
        // For x > 88, exp(x) would overflow, return +infinity
        সঙ্ঘ {
            ভিন্ন    f;
            uint32_t i;
        } inf = { .i = 0x7F800000 };

        প্রত্যাগম inf.f;
    }
    যদি (x < -87.0f) {
        // For x < -87, exp(x) is essentially 0
        প্রত্যাগম 0.0f;
    }

    // Convert to base-2 exponent: x * log2(e)
    ধ্রুব ভিন্ন log2e   = 1.4426950408889634f;  // log2(e)
    ভিন্ন       x_log2e = x * log2e;

    // Use ET hardware instruction: fexp.ps computes 2^x
    ভিন্ন         result;
    unsigned long temp;

    __asm__ চঞ্চল(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fexp.ps %[result], %[x_log2e]  \n\t"  // result = 2^(x * log2(e)) = exp(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result)
        : [x_log2e] "f"(x_log2e));

    প্রত্যাগম result;
}

//******************************************************************************
// Trigonometric Functions
//******************************************************************************

// FSIN.PS

// Sine function using Taylor series
স্থির অন্তঃস্থ ভিন্ন et_sinf(ভিন্ন x) {
    ধ্রুব ভিন্ন pi        = 3.14159265358979323846f;
    ধ্রুব ভিন্ন two_pi    = 6.28318530717958647693f;
    ধ্রুব ভিন্ন pi_over_2 = 1.57079632679489661923f;

    যদি (x > pi || x < -pi) {
        ভিন্ন cycles = x * et_fdiv(1.0f, two_pi);
        পূর্ণাঙ্ক   n      = (পূর্ণাঙ্ক) cycles;
        যদি (x < 0.0f) {
            n--;  // Floor for negative
        }
        x = x - (ভিন্ন) n * two_pi;
    }

    // sin(x) = sin(π - x) for x in [π/2, π]
    // sin(x) = -sin(-π - x) for x in [-π, -π/2]
    পূর্ণাঙ্ক negate = 0;
    যদি (x > pi_over_2) {
        x = pi - x;
    } অন্যথা যদি (x < -pi_over_2) {
        x      = -pi - x;
        negate = 1;
    }

    // sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    ধ্রুব ভিন্ন x2  = x * x;
    ধ্রুব ভিন্ন x3  = x2 * x;
    ধ্রুব ভিন্ন x5  = x3 * x2;
    ধ্রুব ভিন্ন x7  = x5 * x2;
    ধ্রুব ভিন্ন x9  = x7 * x2;
    ধ্রুব ভিন্ন x11 = x9 * x2;

    ভিন্ন result = x - x3 * et_fdiv(1.0f, 6.0f)         // x^3/3!
                   + x5 * et_fdiv(1.0f, 120.0f)         // x^5/5!
                   - x7 * et_fdiv(1.0f, 5040.0f)        // x^7/7!
                   + x9 * et_fdiv(1.0f, 362880.0f)      // x^9/9!
                   - x11 * et_fdiv(1.0f, 39916800.0f);  // x^11/11!

    প্রত্যাগম negate ? -result : result;
}

// Cosine function using identity cos(x) = sin(x + π/2)
স্থির অন্তঃস্থ ভিন্ন et_cosf(ভিন্ন x) {
    ধ্রুব ভিন্ন pi_over_2 = 1.57079632679489661923f;
    প্রত্যাগম et_sinf(x + pi_over_2);
}

//******************************************************************************
// FP16 <-> FP32 Conversion Functions
//******************************************************************************

// Convert FP16 (IEEE 754 half precision) to FP32 (single precision)
// Uses ET hardware FCVT.PS.F16 instruction for accurate conversion
স্থির অন্তঃস্থ ভিন্ন fp16_to_fp32(uint16_t h) {
    ভিন্ন         result;
    unsigned long temp;
    uint32_t      raw = (uint32_t) h;

    __asm__ চঞ্চল(
        "mova.x.m  %[temp]              \n\t"    // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"    // Set mask register m0 to enable element 0
        "fbcx.ps   %[result], %[raw]    \n\t"    // Broadcast raw FP16 bits into vector register
        "fcvt.ps.f16 %[result], %[result] \n\t"  // Convert FP16 to FP32
        "mova.m.x  %[temp]              \n\t"    // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result)
        : [raw] "r"(raw));

    প্রত্যাগম result;
}

// Convert FP32 (single precision) to FP16 (IEEE 754 half precision)
// Uses ET hardware FCVT.F16.PS instruction for accurate conversion
স্থির অন্তঃস্থ uint16_t fp32_to_fp16(ভিন্ন f) {
    ভিন্ন         result_f;
    unsigned long temp;

    __asm__ চঞ্চল(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fcvt.f16.ps %[result], %[f]    \n\t"  // Convert FP32 to FP16 (result in lower 16 bits)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result_f)
        : [f] "f"(f));

    // Extract lower 16 bits containing the FP16 value
    // The instruction zero-extends to 32 bits, so upper 16 bits are 0
    uint32_t result_bits = *(uint32_t *) &result_f;
    প্রত্যাগম (uint16_t) result_bits;
}

#endif  // MATH_FP_H
