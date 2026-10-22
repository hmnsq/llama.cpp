//******************************************************************************
// ET Floating Point Math Library
// Provides ET hardware-specific math functions, FP16 conversion, and trig functions
// for bare metal kernels
//******************************************************************************

#ifndef MATH_FP_H
#define MATH_FP_H

#ଅନ୍ତର୍ଭାଵ <stdint.h>

//******************************************************************************
// ET Hardware Math Functions
//******************************************************************************

// ET hardware division function (uses FRCP.PS instruction)
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ et_fdiv(ଦଶମିକ a, ଦଶମିକ b) {
    ଦଶମିକ         d;
    unsigned long temp;

    __asm__ ଅସ୍ଥିର(
        "mova.x.m  %[temp]              \n\t"
        "mov.m.x   m0, x0, 1            \n\t"
        "frcp.ps   %[d], %[b]           \n\t"
        "fmul.s    %[d], %[d], %[a]     \n\t"
        "mova.m.x  %[temp]              \n\t"
        : [temp] "=&r"(temp), [d] "=&f"(d)
        : [a] "f"(a), [b] "f"(b));

    ଫେରନ୍ତୁ d;
}

// Power function using ET hardware vector instructions
// Implements pow(base, exp) = exp(exp * ln(base)) using FLOG.PS and FEXP.PS
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ et_powf(ଦଶମିକ base, ଦଶମିକ exp) {
    // Handle special cases
    ଯଦି (base <= 0.0f) {
        ଯଦି (base == 0.0f) {
            ଯଦି (exp > 0.0f) {
                ଫେରନ୍ତୁ 0.0f;
            }

            // For exp <= 0, return +infinity (IEEE 754: sign=0, exp=0xFF, mantissa=0)
            ସଂଘ {
                ଦଶମିକ    f;
                uint32_t i;
            } inf = { .i = 0x7F800000 };

            ଫେରନ୍ତୁ inf.f;
        }

        // For negative base, return NaN (IEEE 754: exp=0xFF, mantissa!=0)
        ସଂଘ {
            ଦଶମିକ    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        ଫେରନ୍ତୁ nan.f;
    }
    ଯଦି (base == 1.0f) {
        ଫେରନ୍ତୁ 1.0f;
    }
    ଯଦି (exp == 0.0f) {
        ଫେରନ୍ତୁ 1.0f;
    }
    ଯଦି (exp == 1.0f) {
        ଫେରନ୍ତୁ base;
    }

    // Use ET hardware instructions following DNN library pattern:
    // pow(base, exp) = exp(exp * ln(base))
    ଦଶମିକ         result;
    unsigned long temp;

    __asm__ ଅସ୍ଥିର(
        "mova.x.m  %[temp]              \n\t"      // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"      // Set mask register m0 to enable element 0
        "flog.ps %[result], %[base]     \n\t"      // result = ln(base)
        "fmul.s %[result], %[result], %[exp]\n\t"  // result = ln(base) * exp
        "fexp.ps %[result], %[result]   \n\t"      // result = exp(ln(base) * exp) = base^exp
        "mova.m.x  %[temp]              \n\t"      // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result)
        : [base] "f"(base), [exp] "f"(exp));

    ଫେରନ୍ତୁ result;
}

// Natural logarithm.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ et_logf(ଦଶମିକ x) {
    // Handle special cases
    ଯଦି (x < 0.0f) {
        // Return NaN for negative input
        ସଂଘ {
            ଦଶମିକ    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        ଫେରନ୍ତୁ nan.f;
    }
    ଯଦି (x == 0.0f) {
        // Return -infinity for log(0)
        ସଂଘ {
            ଦଶମିକ    f;
            uint32_t i;
        } inf = { .i = 0xFF800000 };

        ଫେରନ୍ତୁ inf.f;
    }
    ଯଦି (x == 1.0f) {
        ଫେରନ୍ତୁ 0.0f;
    }

    ଦଶମିକ         log2_result;
    unsigned long temp;

    __asm__ ଅସ୍ଥିର(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "flog.ps %[result], %[x]        \n\t"  // result = log2(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(log2_result)
        : [x] "f"(x));

    // Convert log2 to natural log: ln(x) = log2(x) * ln(2)
    ଅଚଳ ଦଶମିକ ln2 = 0.69314718055994530942f;
    ଫେରନ୍ତୁ log2_result * ln2;
}

// Square root function implemented as et_powf(x, 0.5)
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ et_sqrtf(ଦଶମିକ x) {
    // Handle special cases
    ଯଦି (x < 0.0f) {
        // Return NaN for negative input (IEEE 754: exp=0xFF, mantissa!=0)
        ସଂଘ {
            ଦଶମିକ    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        ଫେରନ୍ତୁ nan.f;
    }
    ଯଦି (x == 0.0f) {
        ଫେରନ୍ତୁ 0.0f;
    }

    ଫେରନ୍ତୁ et_powf(x, 0.5f);
}

// Base-2 exponential: returns 2^x using the ET hardware FEXP.PS instruction.
// No base conversion, no special-case clamping — this is the raw hardware op
// with just the mask save/restore wrapper.  Caller is responsible for ensuring
// x is in a range that produces a useful result (roughly [-126, 128] for fp32).
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ __attribute__((always_inline)) et_exp2f(ଦଶମିକ x) {
    unsigned long old_mask;
    ଦଶମିକ         out;
    __asm__ ଅସ୍ଥିର(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 1         \n\t"
        "fexp.ps   %[out], %[x]      \n\t"
        "mova.m.x  %[ms]             \n\t"
        : [ms] "=&r"(old_mask), [out] "=&f"(out)
        : [x] "f"(x));
    ଫେରନ୍ତୁ out;
}

// Exponential function using ET hardware FEXP.PS instruction
// Note: FEXP.PS computes 2^x, so we need to convert: exp(x) = 2^(x * log2(e))
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ et_expf(ଦଶମିକ x) {
    // Handle special cases
    ଯଦି (x > 88.0f) {
        // For x > 88, exp(x) would overflow, return +infinity
        ସଂଘ {
            ଦଶମିକ    f;
            uint32_t i;
        } inf = { .i = 0x7F800000 };

        ଫେରନ୍ତୁ inf.f;
    }
    ଯଦି (x < -87.0f) {
        // For x < -87, exp(x) is essentially 0
        ଫେରନ୍ତୁ 0.0f;
    }

    // Convert to base-2 exponent: x * log2(e)
    ଅଚଳ ଦଶମିକ log2e   = 1.4426950408889634f;  // log2(e)
    ଦଶମିକ       x_log2e = x * log2e;

    // Use ET hardware instruction: fexp.ps computes 2^x
    ଦଶମିକ         result;
    unsigned long temp;

    __asm__ ଅସ୍ଥିର(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fexp.ps %[result], %[x_log2e]  \n\t"  // result = 2^(x * log2(e)) = exp(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result)
        : [x_log2e] "f"(x_log2e));

    ଫେରନ୍ତୁ result;
}

//******************************************************************************
// Trigonometric Functions
//******************************************************************************

// FSIN.PS

// Sine function using Taylor series
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ et_sinf(ଦଶମିକ x) {
    ଅଚଳ ଦଶମିକ pi        = 3.14159265358979323846f;
    ଅଚଳ ଦଶମିକ two_pi    = 6.28318530717958647693f;
    ଅଚଳ ଦଶମିକ pi_over_2 = 1.57079632679489661923f;

    ଯଦି (x > pi || x < -pi) {
        ଦଶମିକ cycles = x * et_fdiv(1.0f, two_pi);
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା   n      = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) cycles;
        ଯଦି (x < 0.0f) {
            n--;  // Floor for negative
        }
        x = x - (ଦଶମିକ) n * two_pi;
    }

    // sin(x) = sin(π - x) for x in [π/2, π]
    // sin(x) = -sin(-π - x) for x in [-π, -π/2]
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା negate = 0;
    ଯଦି (x > pi_over_2) {
        x = pi - x;
    } ଅନ୍ୟଥା ଯଦି (x < -pi_over_2) {
        x      = -pi - x;
        negate = 1;
    }

    // sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    ଅଚଳ ଦଶମିକ x2  = x * x;
    ଅଚଳ ଦଶମିକ x3  = x2 * x;
    ଅଚଳ ଦଶମିକ x5  = x3 * x2;
    ଅଚଳ ଦଶମିକ x7  = x5 * x2;
    ଅଚଳ ଦଶମିକ x9  = x7 * x2;
    ଅଚଳ ଦଶମିକ x11 = x9 * x2;

    ଦଶମିକ result = x - x3 * et_fdiv(1.0f, 6.0f)         // x^3/3!
                   + x5 * et_fdiv(1.0f, 120.0f)         // x^5/5!
                   - x7 * et_fdiv(1.0f, 5040.0f)        // x^7/7!
                   + x9 * et_fdiv(1.0f, 362880.0f)      // x^9/9!
                   - x11 * et_fdiv(1.0f, 39916800.0f);  // x^11/11!

    ଫେରନ୍ତୁ negate ? -result : result;
}

// Cosine function using identity cos(x) = sin(x + π/2)
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ et_cosf(ଦଶମିକ x) {
    ଅଚଳ ଦଶମିକ pi_over_2 = 1.57079632679489661923f;
    ଫେରନ୍ତୁ et_sinf(x + pi_over_2);
}

//******************************************************************************
// FP16 <-> FP32 Conversion Functions
//******************************************************************************

// Convert FP16 (IEEE 754 half precision) to FP32 (single precision)
// Uses ET hardware FCVT.PS.F16 instruction for accurate conversion
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ fp16_to_fp32(uint16_t h) {
    ଦଶମିକ         result;
    unsigned long temp;
    uint32_t      raw = (uint32_t) h;

    __asm__ ଅସ୍ଥିର(
        "mova.x.m  %[temp]              \n\t"    // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"    // Set mask register m0 to enable element 0
        "fbcx.ps   %[result], %[raw]    \n\t"    // Broadcast raw FP16 bits into vector register
        "fcvt.ps.f16 %[result], %[result] \n\t"  // Convert FP16 to FP32
        "mova.m.x  %[temp]              \n\t"    // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result)
        : [raw] "r"(raw));

    ଫେରନ୍ତୁ result;
}

// Convert FP32 (single precision) to FP16 (IEEE 754 half precision)
// Uses ET hardware FCVT.F16.PS instruction for accurate conversion
ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint16_t fp32_to_fp16(ଦଶମିକ f) {
    ଦଶମିକ         result_f;
    unsigned long temp;

    __asm__ ଅସ୍ଥିର(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fcvt.f16.ps %[result], %[f]    \n\t"  // Convert FP32 to FP16 (result in lower 16 bits)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result_f)
        : [f] "f"(f));

    // Extract lower 16 bits containing the FP16 value
    // The instruction zero-extends to 32 bits, so upper 16 bits are 0
    uint32_t result_bits = *(uint32_t *) &result_f;
    ଫେରନ୍ତୁ (uint16_t) result_bits;
}

#endif  // MATH_FP_H
