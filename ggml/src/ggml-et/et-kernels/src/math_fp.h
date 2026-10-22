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
স্থির অন্তঃরেখ দশমিক et_fdiv(দশমিক a, দশমিক b) {
    দশমিক         d;
    unsigned long অস্থায়ী;

    __asm__ অস্থির(
        "mova.x.m  %[temp]              \n\t"
        "mov.m.x   m0, x0, 1            \n\t"
        "frcp.ps   %[d], %[b]           \n\t"
        "fmul.s    %[d], %[d], %[a]     \n\t"
        "mova.m.x  %[temp]              \n\t"
        : [অস্থায়ী] "=&r"(অস্থায়ী), [d] "=&f"(d)
        : [a] "f"(a), [b] "f"(b));

    ফেরত d;
}

// Power function using ET hardware vector instructions
// Implements pow(base, exp) = exp(exp * ln(base)) using FLOG.PS and FEXP.PS
স্থির অন্তঃরেখ দশমিক et_powf(দশমিক base, দশমিক exp) {
    // Handle special cases
    যদি (base <= 0.0f) {
        যদি (base == 0.0f) {
            যদি (exp > 0.0f) {
                ফেরত 0.0f;
            }

            // For exp <= 0, return +infinity (IEEE 754: sign=0, exp=0xFF, mantissa=0)
            সংঘ {
                দশমিক    f;
                uint32_t i;
            } inf = { .i = 0x7F800000 };

            ফেরত inf.f;
        }

        // For negative base, return NaN (IEEE 754: exp=0xFF, mantissa!=0)
        সংঘ {
            দশমিক    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        ফেরত nan.f;
    }
    যদি (base == 1.0f) {
        ফেরত 1.0f;
    }
    যদি (exp == 0.0f) {
        ফেরত 1.0f;
    }
    যদি (exp == 1.0f) {
        ফেরত base;
    }

    // Use ET hardware instructions following DNN library pattern:
    // pow(base, exp) = exp(exp * ln(base))
    দশমিক         ফলাফল;
    unsigned long অস্থায়ী;

    __asm__ অস্থির(
        "mova.x.m  %[temp]              \n\t"      // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"      // Set mask register m0 to enable element 0
        "flog.ps %[result], %[base]     \n\t"      // result = ln(base)
        "fmul.s %[result], %[result], %[exp]\n\t"  // result = ln(base) * exp
        "fexp.ps %[result], %[result]   \n\t"      // result = exp(ln(base) * exp) = base^exp
        "mova.m.x  %[temp]              \n\t"      // Restore mask state
        : [অস্থায়ী] "=&r"(অস্থায়ী), [ফলাফল] "=&f"(ফলাফল)
        : [base] "f"(base), [exp] "f"(exp));

    ফেরত ফলাফল;
}

// Natural logarithm.
স্থির অন্তঃরেখ দশমিক et_logf(দশমিক x) {
    // Handle special cases
    যদি (x < 0.0f) {
        // Return NaN for negative input
        সংঘ {
            দশমিক    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        ফেরত nan.f;
    }
    যদি (x == 0.0f) {
        // Return -infinity for log(0)
        সংঘ {
            দশমিক    f;
            uint32_t i;
        } inf = { .i = 0xFF800000 };

        ফেরত inf.f;
    }
    যদি (x == 1.0f) {
        ফেরত 0.0f;
    }

    দশমিক         log2_ফলাফল;
    unsigned long অস্থায়ী;

    __asm__ অস্থির(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "flog.ps %[result], %[x]        \n\t"  // result = log2(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [অস্থায়ী] "=&r"(অস্থায়ী), [ফলাফল] "=&f"(log2_ফলাফল)
        : [x] "f"(x));

    // Convert log2 to natural log: ln(x) = log2(x) * ln(2)
    ধ্রুবক দশমিক ln2 = 0.69314718055994530942f;
    ফেরত log2_ফলাফল * ln2;
}

// Square root function implemented as et_powf(x, 0.5)
স্থির অন্তঃরেখ দশমিক et_sqrtf(দশমিক x) {
    // Handle special cases
    যদি (x < 0.0f) {
        // Return NaN for negative input (IEEE 754: exp=0xFF, mantissa!=0)
        সংঘ {
            দশমিক    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        ফেরত nan.f;
    }
    যদি (x == 0.0f) {
        ফেরত 0.0f;
    }

    ফেরত et_powf(x, 0.5f);
}

// Base-2 exponential: returns 2^x using the ET hardware FEXP.PS instruction.
// No base conversion, no special-case clamping — this is the raw hardware op
// with just the mask save/restore wrapper.  Caller is responsible for ensuring
// x is in a range that produces a useful result (roughly [-126, 128] for fp32).
স্থির অন্তঃরেখ দশমিক __attribute__((always_inline)) et_exp2f(দশমিক x) {
    unsigned long old_mask;
    দশমিক         out;
    __asm__ অস্থির(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 1         \n\t"
        "fexp.ps   %[out], %[x]      \n\t"
        "mova.m.x  %[ms]             \n\t"
        : [ms] "=&r"(old_mask), [out] "=&f"(out)
        : [x] "f"(x));
    ফেরত out;
}

// Exponential function using ET hardware FEXP.PS instruction
// Note: FEXP.PS computes 2^x, so we need to convert: exp(x) = 2^(x * log2(e))
স্থির অন্তঃরেখ দশমিক et_expf(দশমিক x) {
    // Handle special cases
    যদি (x > 88.0f) {
        // For x > 88, exp(x) would overflow, return +infinity
        সংঘ {
            দশমিক    f;
            uint32_t i;
        } inf = { .i = 0x7F800000 };

        ফেরত inf.f;
    }
    যদি (x < -87.0f) {
        // For x < -87, exp(x) is essentially 0
        ফেরত 0.0f;
    }

    // Convert to base-2 exponent: x * log2(e)
    ধ্রুবক দশমিক log2e   = 1.4426950408889634f;  // log2(e)
    দশমিক       x_log2e = x * log2e;

    // Use ET hardware instruction: fexp.ps computes 2^x
    দশমিক         ফলাফল;
    unsigned long অস্থায়ী;

    __asm__ অস্থির(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fexp.ps %[result], %[x_log2e]  \n\t"  // result = 2^(x * log2(e)) = exp(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [অস্থায়ী] "=&r"(অস্থায়ী), [ফলাফল] "=&f"(ফলাফল)
        : [x_log2e] "f"(x_log2e));

    ফেরত ফলাফল;
}

//******************************************************************************
// Trigonometric Functions
//******************************************************************************

// FSIN.PS

// Sine function using Taylor series
স্থির অন্তঃরেখ দশমিক et_sinf(দশমিক x) {
    ধ্রুবক দশমিক pi        = 3.14159265358979323846f;
    ধ্রুবক দশমিক two_pi    = 6.28318530717958647693f;
    ধ্রুবক দশমিক pi_over_2 = 1.57079632679489661923f;

    যদি (x > pi || x < -pi) {
        দশমিক cycles = x * et_fdiv(1.0f, two_pi);
        পূর্ণসংখ্যা   n      = (পূর্ণসংখ্যা) cycles;
        যদি (x < 0.0f) {
            n--;  // Floor for negative
        }
        x = x - (দশমিক) n * two_pi;
    }

    // sin(x) = sin(π - x) for x in [π/2, π]
    // sin(x) = -sin(-π - x) for x in [-π, -π/2]
    পূর্ণসংখ্যা negate = 0;
    যদি (x > pi_over_2) {
        x = pi - x;
    } অন্যথা যদি (x < -pi_over_2) {
        x      = -pi - x;
        negate = 1;
    }

    // sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    ধ্রুবক দশমিক x2  = x * x;
    ধ্রুবক দশমিক x3  = x2 * x;
    ধ্রুবক দশমিক x5  = x3 * x2;
    ধ্রুবক দশমিক x7  = x5 * x2;
    ধ্রুবক দশমিক x9  = x7 * x2;
    ধ্রুবক দশমিক x11 = x9 * x2;

    দশমিক ফলাফল = x - x3 * et_fdiv(1.0f, 6.0f)         // x^3/3!
                   + x5 * et_fdiv(1.0f, 120.0f)         // x^5/5!
                   - x7 * et_fdiv(1.0f, 5040.0f)        // x^7/7!
                   + x9 * et_fdiv(1.0f, 362880.0f)      // x^9/9!
                   - x11 * et_fdiv(1.0f, 39916800.0f);  // x^11/11!

    ফেরত negate ? -ফলাফল : ফলাফল;
}

// Cosine function using identity cos(x) = sin(x + π/2)
স্থির অন্তঃরেখ দশমিক et_cosf(দশমিক x) {
    ধ্রুবক দশমিক pi_over_2 = 1.57079632679489661923f;
    ফেরত et_sinf(x + pi_over_2);
}

//******************************************************************************
// FP16 <-> FP32 Conversion Functions
//******************************************************************************

// Convert FP16 (IEEE 754 half precision) to FP32 (single precision)
// Uses ET hardware FCVT.PS.F16 instruction for accurate conversion
স্থির অন্তঃরেখ দশমিক fp16_to_fp32(uint16_t h) {
    দশমিক         ফলাফল;
    unsigned long অস্থায়ী;
    uint32_t      raw = (uint32_t) h;

    __asm__ অস্থির(
        "mova.x.m  %[temp]              \n\t"    // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"    // Set mask register m0 to enable element 0
        "fbcx.ps   %[result], %[raw]    \n\t"    // Broadcast raw FP16 bits into vector register
        "fcvt.ps.f16 %[result], %[result] \n\t"  // Convert FP16 to FP32
        "mova.m.x  %[temp]              \n\t"    // Restore mask state
        : [অস্থায়ী] "=&r"(অস্থায়ী), [ফলাফল] "=&f"(ফলাফল)
        : [raw] "r"(raw));

    ফেরত ফলাফল;
}

// Convert FP32 (single precision) to FP16 (IEEE 754 half precision)
// Uses ET hardware FCVT.F16.PS instruction for accurate conversion
স্থির অন্তঃরেখ uint16_t fp32_to_fp16(দশমিক f) {
    দশমিক         ফলাফল_f;
    unsigned long অস্থায়ী;

    __asm__ অস্থির(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fcvt.f16.ps %[result], %[f]    \n\t"  // Convert FP32 to FP16 (result in lower 16 bits)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [অস্থায়ী] "=&r"(অস্থায়ী), [ফলাফল] "=&f"(ফলাফল_f)
        : [f] "f"(f));

    // Extract lower 16 bits containing the FP16 value
    // The instruction zero-extends to 32 bits, so upper 16 bits are 0
    uint32_t ফলাফল_bits = *(uint32_t *) &ফলাফল_f;
    ফেরত (uint16_t) ফলাফল_bits;
}

#endif  // MATH_FP_H
