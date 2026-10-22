//******************************************************************************
// ET Floating Point Math Library
// Provides ET hardware-specific math functions, FP16 conversion, and trig functions
// for bare metal kernels
//******************************************************************************

#ifndef MATH_FP_H
#define MATH_FP_H

#અન્તર્ભાવ <stdint.h>

//******************************************************************************
// ET Hardware Math Functions
//******************************************************************************

// ET hardware division function (uses FRCP.PS instruction)
સ્થિર અંતઃરેખ દશાંશ et_fdiv(દશાંશ a, દશાંશ b) {
    દશાંશ         d;
    unsigned long temp;

    __asm__ અસ્થિર(
        "mova.x.m  %[temp]              \n\t"
        "mov.m.x   m0, x0, 1            \n\t"
        "frcp.ps   %[d], %[b]           \n\t"
        "fmul.s    %[d], %[d], %[a]     \n\t"
        "mova.m.x  %[temp]              \n\t"
        : [temp] "=&r"(temp), [d] "=&f"(d)
        : [a] "f"(a), [b] "f"(b));

    પાછા d;
}

// Power function using ET hardware vector instructions
// Implements pow(base, exp) = exp(exp * ln(base)) using FLOG.PS and FEXP.PS
સ્થિર અંતઃરેખ દશાંશ et_powf(દશાંશ base, દશાંશ exp) {
    // Handle special cases
    જો (base <= 0.0f) {
        જો (base == 0.0f) {
            જો (exp > 0.0f) {
                પાછા 0.0f;
            }

            // For exp <= 0, return +infinity (IEEE 754: sign=0, exp=0xFF, mantissa=0)
            સંઘ {
                દશાંશ    f;
                uint32_t i;
            } inf = { .i = 0x7F800000 };

            પાછા inf.f;
        }

        // For negative base, return NaN (IEEE 754: exp=0xFF, mantissa!=0)
        સંઘ {
            દશાંશ    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        પાછા nan.f;
    }
    જો (base == 1.0f) {
        પાછા 1.0f;
    }
    જો (exp == 0.0f) {
        પાછા 1.0f;
    }
    જો (exp == 1.0f) {
        પાછા base;
    }

    // Use ET hardware instructions following DNN library pattern:
    // pow(base, exp) = exp(exp * ln(base))
    દશાંશ         result;
    unsigned long temp;

    __asm__ અસ્થિર(
        "mova.x.m  %[temp]              \n\t"      // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"      // Set mask register m0 to enable element 0
        "flog.ps %[result], %[base]     \n\t"      // result = ln(base)
        "fmul.s %[result], %[result], %[exp]\n\t"  // result = ln(base) * exp
        "fexp.ps %[result], %[result]   \n\t"      // result = exp(ln(base) * exp) = base^exp
        "mova.m.x  %[temp]              \n\t"      // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result)
        : [base] "f"(base), [exp] "f"(exp));

    પાછા result;
}

// Natural logarithm.
સ્થિર અંતઃરેખ દશાંશ et_logf(દશાંશ x) {
    // Handle special cases
    જો (x < 0.0f) {
        // Return NaN for negative input
        સંઘ {
            દશાંશ    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        પાછા nan.f;
    }
    જો (x == 0.0f) {
        // Return -infinity for log(0)
        સંઘ {
            દશાંશ    f;
            uint32_t i;
        } inf = { .i = 0xFF800000 };

        પાછા inf.f;
    }
    જો (x == 1.0f) {
        પાછા 0.0f;
    }

    દશાંશ         log2_result;
    unsigned long temp;

    __asm__ અસ્થિર(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "flog.ps %[result], %[x]        \n\t"  // result = log2(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(log2_result)
        : [x] "f"(x));

    // Convert log2 to natural log: ln(x) = log2(x) * ln(2)
    અચલ દશાંશ ln2 = 0.69314718055994530942f;
    પાછા log2_result * ln2;
}

// Square root function implemented as et_powf(x, 0.5)
સ્થિર અંતઃરેખ દશાંશ et_sqrtf(દશાંશ x) {
    // Handle special cases
    જો (x < 0.0f) {
        // Return NaN for negative input (IEEE 754: exp=0xFF, mantissa!=0)
        સંઘ {
            દશાંશ    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        પાછા nan.f;
    }
    જો (x == 0.0f) {
        પાછા 0.0f;
    }

    પાછા et_powf(x, 0.5f);
}

// Base-2 exponential: returns 2^x using the ET hardware FEXP.PS instruction.
// No base conversion, no special-case clamping — this is the raw hardware op
// with just the mask save/restore wrapper.  Caller is responsible for ensuring
// x is in a range that produces a useful result (roughly [-126, 128] for fp32).
સ્થિર અંતઃરેખ દશાંશ __attribute__((always_inline)) et_exp2f(દશાંશ x) {
    unsigned long old_mask;
    દશાંશ         out;
    __asm__ અસ્થિર(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 1         \n\t"
        "fexp.ps   %[out], %[x]      \n\t"
        "mova.m.x  %[ms]             \n\t"
        : [ms] "=&r"(old_mask), [out] "=&f"(out)
        : [x] "f"(x));
    પાછા out;
}

// Exponential function using ET hardware FEXP.PS instruction
// Note: FEXP.PS computes 2^x, so we need to convert: exp(x) = 2^(x * log2(e))
સ્થિર અંતઃરેખ દશાંશ et_expf(દશાંશ x) {
    // Handle special cases
    જો (x > 88.0f) {
        // For x > 88, exp(x) would overflow, return +infinity
        સંઘ {
            દશાંશ    f;
            uint32_t i;
        } inf = { .i = 0x7F800000 };

        પાછા inf.f;
    }
    જો (x < -87.0f) {
        // For x < -87, exp(x) is essentially 0
        પાછા 0.0f;
    }

    // Convert to base-2 exponent: x * log2(e)
    અચલ દશાંશ log2e   = 1.4426950408889634f;  // log2(e)
    દશાંશ       x_log2e = x * log2e;

    // Use ET hardware instruction: fexp.ps computes 2^x
    દશાંશ         result;
    unsigned long temp;

    __asm__ અસ્થિર(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fexp.ps %[result], %[x_log2e]  \n\t"  // result = 2^(x * log2(e)) = exp(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result)
        : [x_log2e] "f"(x_log2e));

    પાછા result;
}

//******************************************************************************
// Trigonometric Functions
//******************************************************************************

// FSIN.PS

// Sine function using Taylor series
સ્થિર અંતઃરેખ દશાંશ et_sinf(દશાંશ x) {
    અચલ દશાંશ pi        = 3.14159265358979323846f;
    અચલ દશાંશ two_pi    = 6.28318530717958647693f;
    અચલ દશાંશ pi_over_2 = 1.57079632679489661923f;

    જો (x > pi || x < -pi) {
        દશાંશ cycles = x * et_fdiv(1.0f, two_pi);
        પૂર્ણાંક   n      = (પૂર્ણાંક) cycles;
        જો (x < 0.0f) {
            n--;  // Floor for negative
        }
        x = x - (દશાંશ) n * two_pi;
    }

    // sin(x) = sin(π - x) for x in [π/2, π]
    // sin(x) = -sin(-π - x) for x in [-π, -π/2]
    પૂર્ણાંક negate = 0;
    જો (x > pi_over_2) {
        x = pi - x;
    } નહીંતર જો (x < -pi_over_2) {
        x      = -pi - x;
        negate = 1;
    }

    // sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    અચલ દશાંશ x2  = x * x;
    અચલ દશાંશ x3  = x2 * x;
    અચલ દશાંશ x5  = x3 * x2;
    અચલ દશાંશ x7  = x5 * x2;
    અચલ દશાંશ x9  = x7 * x2;
    અચલ દશાંશ x11 = x9 * x2;

    દશાંશ result = x - x3 * et_fdiv(1.0f, 6.0f)         // x^3/3!
                   + x5 * et_fdiv(1.0f, 120.0f)         // x^5/5!
                   - x7 * et_fdiv(1.0f, 5040.0f)        // x^7/7!
                   + x9 * et_fdiv(1.0f, 362880.0f)      // x^9/9!
                   - x11 * et_fdiv(1.0f, 39916800.0f);  // x^11/11!

    પાછા negate ? -result : result;
}

// Cosine function using identity cos(x) = sin(x + π/2)
સ્થિર અંતઃરેખ દશાંશ et_cosf(દશાંશ x) {
    અચલ દશાંશ pi_over_2 = 1.57079632679489661923f;
    પાછા et_sinf(x + pi_over_2);
}

//******************************************************************************
// FP16 <-> FP32 Conversion Functions
//******************************************************************************

// Convert FP16 (IEEE 754 half precision) to FP32 (single precision)
// Uses ET hardware FCVT.PS.F16 instruction for accurate conversion
સ્થિર અંતઃરેખ દશાંશ fp16_to_fp32(uint16_t h) {
    દશાંશ         result;
    unsigned long temp;
    uint32_t      raw = (uint32_t) h;

    __asm__ અસ્થિર(
        "mova.x.m  %[temp]              \n\t"    // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"    // Set mask register m0 to enable element 0
        "fbcx.ps   %[result], %[raw]    \n\t"    // Broadcast raw FP16 bits into vector register
        "fcvt.ps.f16 %[result], %[result] \n\t"  // Convert FP16 to FP32
        "mova.m.x  %[temp]              \n\t"    // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result)
        : [raw] "r"(raw));

    પાછા result;
}

// Convert FP32 (single precision) to FP16 (IEEE 754 half precision)
// Uses ET hardware FCVT.F16.PS instruction for accurate conversion
સ્થિર અંતઃરેખ uint16_t fp32_to_fp16(દશાંશ f) {
    દશાંશ         result_f;
    unsigned long temp;

    __asm__ અસ્થિર(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fcvt.f16.ps %[result], %[f]    \n\t"  // Convert FP32 to FP16 (result in lower 16 bits)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [temp] "=&r"(temp), [result] "=&f"(result_f)
        : [f] "f"(f));

    // Extract lower 16 bits containing the FP16 value
    // The instruction zero-extends to 32 bits, so upper 16 bits are 0
    uint32_t result_bits = *(uint32_t *) &result_f;
    પાછા (uint16_t) result_bits;
}

#endif  // MATH_FP_H
