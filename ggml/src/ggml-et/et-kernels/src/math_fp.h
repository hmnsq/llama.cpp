//******************************************************************************
// ET Floating Point Math Library
// Provides ET hardware-specific math functions, FP16 conversion, and trig functions
// for bare metal kernels
//******************************************************************************

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ MATH_FP_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MATH_FP_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

//******************************************************************************
// ET Hardware Math Functions
//******************************************************************************

// ET hardware division function (uses FRCP.PS instruction)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ et_fdiv(ਦਸ਼ਮਲਵ a, ਦਸ਼ਮਲਵ b) {
    ਦਸ਼ਮਲਵ         d;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ;

    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[temp]              \n\t"
        "mov.m.x   m0, x0, 1            \n\t"
        "frcp.ps   %[d], %[b]           \n\t"
        "fmul.s    %[d], %[d], %[a]     \n\t"
        "mova.m.x  %[temp]              \n\t"
        : [ਆਰਜ਼ੀ] "=&r"(ਆਰਜ਼ੀ), [d] "=&f"(d)
        : [a] "f"(a), [b] "f"(b));

    ਵਾਪਸ d;
}

// Power function using ET hardware vector instructions
// Implements pow(base, exp) = exp(exp * ln(base)) using FLOG.PS and FEXP.PS
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ et_powf(ਦਸ਼ਮਲਵ ਅਧਾਰ, ਦਸ਼ਮਲਵ ਘਾਤਾਂਕ) {
    // Handle special cases
    ਜੇ (ਅਧਾਰ <= 0.0f) {
        ਜੇ (ਅਧਾਰ == 0.0f) {
            ਜੇ (ਘਾਤਾਂਕ > 0.0f) {
                ਵਾਪਸ 0.0f;
            }

            // For exp <= 0, return +infinity (IEEE 754: sign=0, exp=0xFF, mantissa=0)
            ਸੰਘ {
                ਦਸ਼ਮਲਵ    f;
                uint32_t i;
            } inf = { .i = 0x7F800000 };

            ਵਾਪਸ inf.f;
        }

        // For negative base, return NaN (IEEE 754: exp=0xFF, mantissa!=0)
        ਸੰਘ {
            ਦਸ਼ਮਲਵ    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        ਵਾਪਸ nan.f;
    }
    ਜੇ (ਅਧਾਰ == 1.0f) {
        ਵਾਪਸ 1.0f;
    }
    ਜੇ (ਘਾਤਾਂਕ == 0.0f) {
        ਵਾਪਸ 1.0f;
    }
    ਜੇ (ਘਾਤਾਂਕ == 1.0f) {
        ਵਾਪਸ ਅਧਾਰ;
    }

    // Use ET hardware instructions following DNN library pattern:
    // pow(base, exp) = exp(exp * ln(base))
    ਦਸ਼ਮਲਵ         ਨਤੀਜਾ;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ;

    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[temp]              \n\t"      // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"      // Set mask register m0 to enable element 0
        "flog.ps %[result], %[base]     \n\t"      // result = ln(base)
        "fmul.s %[result], %[result], %[exp]\n\t"  // result = ln(base) * exp
        "fexp.ps %[result], %[result]   \n\t"      // result = exp(ln(base) * exp) = base^exp
        "mova.m.x  %[temp]              \n\t"      // Restore mask state
        : [ਆਰਜ਼ੀ] "=&r"(ਆਰਜ਼ੀ), [ਨਤੀਜਾ] "=&f"(ਨਤੀਜਾ)
        : [ਅਧਾਰ] "f"(ਅਧਾਰ), [ਘਾਤਾਂਕ] "f"(ਘਾਤਾਂਕ));

    ਵਾਪਸ ਨਤੀਜਾ;
}

// Natural logarithm.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ et_logf(ਦਸ਼ਮਲਵ x) {
    // Handle special cases
    ਜੇ (x < 0.0f) {
        // Return NaN for negative input
        ਸੰਘ {
            ਦਸ਼ਮਲਵ    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        ਵਾਪਸ nan.f;
    }
    ਜੇ (x == 0.0f) {
        // Return -infinity for log(0)
        ਸੰਘ {
            ਦਸ਼ਮਲਵ    f;
            uint32_t i;
        } inf = { .i = 0xFF800000 };

        ਵਾਪਸ inf.f;
    }
    ਜੇ (x == 1.0f) {
        ਵਾਪਸ 0.0f;
    }

    ਦਸ਼ਮਲਵ         log2_ਨਤੀਜਾ;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ;

    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "flog.ps %[result], %[x]        \n\t"  // result = log2(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [ਆਰਜ਼ੀ] "=&r"(ਆਰਜ਼ੀ), [ਨਤੀਜਾ] "=&f"(log2_ਨਤੀਜਾ)
        : [x] "f"(x));

    // Convert log2 to natural log: ln(x) = log2(x) * ln(2)
    ਅਚਲ ਦਸ਼ਮਲਵ ln2 = 0.69314718055994530942f;
    ਵਾਪਸ log2_ਨਤੀਜਾ * ln2;
}

// Square root function implemented as et_powf(x, 0.5)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ et_sqrtf(ਦਸ਼ਮਲਵ x) {
    // Handle special cases
    ਜੇ (x < 0.0f) {
        // Return NaN for negative input (IEEE 754: exp=0xFF, mantissa!=0)
        ਸੰਘ {
            ਦਸ਼ਮਲਵ    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        ਵਾਪਸ nan.f;
    }
    ਜੇ (x == 0.0f) {
        ਵਾਪਸ 0.0f;
    }

    ਵਾਪਸ et_powf(x, 0.5f);
}

// Base-2 exponential: returns 2^x using the ET hardware FEXP.PS instruction.
// No base conversion, no special-case clamping — this is the raw hardware op
// with just the mask save/restore wrapper.  Caller is responsible for ensuring
// x is in a range that produces a useful result (roughly [-126, 128] for fp32).
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ __attribute__((always_inline)) et_exp2f(ਦਸ਼ਮਲਵ x) {
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ old_ਨਕਾਬ;
    ਦਸ਼ਮਲਵ         ਬਾਹਰਲਾ;
    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 1         \n\t"
        "fexp.ps   %[out], %[x]      \n\t"
        "mova.m.x  %[ms]             \n\t"
        : [ms] "=&r"(old_ਨਕਾਬ), [ਬਾਹਰਲਾ] "=&f"(ਬਾਹਰਲਾ)
        : [x] "f"(x));
    ਵਾਪਸ ਬਾਹਰਲਾ;
}

// Exponential function using ET hardware FEXP.PS instruction
// Note: FEXP.PS computes 2^x, so we need to convert: exp(x) = 2^(x * log2(e))
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ et_expf(ਦਸ਼ਮਲਵ x) {
    // Handle special cases
    ਜੇ (x > 88.0f) {
        // For x > 88, exp(x) would overflow, return +infinity
        ਸੰਘ {
            ਦਸ਼ਮਲਵ    f;
            uint32_t i;
        } inf = { .i = 0x7F800000 };

        ਵਾਪਸ inf.f;
    }
    ਜੇ (x < -87.0f) {
        // For x < -87, exp(x) is essentially 0
        ਵਾਪਸ 0.0f;
    }

    // Convert to base-2 exponent: x * log2(e)
    ਅਚਲ ਦਸ਼ਮਲਵ log2e   = 1.4426950408889634f;  // log2(e)
    ਦਸ਼ਮਲਵ       x_log2e = x * log2e;

    // Use ET hardware instruction: fexp.ps computes 2^x
    ਦਸ਼ਮਲਵ         ਨਤੀਜਾ;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ;

    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fexp.ps %[result], %[x_log2e]  \n\t"  // result = 2^(x * log2(e)) = exp(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [ਆਰਜ਼ੀ] "=&r"(ਆਰਜ਼ੀ), [ਨਤੀਜਾ] "=&f"(ਨਤੀਜਾ)
        : [x_log2e] "f"(x_log2e));

    ਵਾਪਸ ਨਤੀਜਾ;
}

//******************************************************************************
// Trigonometric Functions
//******************************************************************************

// FSIN.PS

// Sine function using Taylor series
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ et_sinf(ਦਸ਼ਮਲਵ x) {
    ਅਚਲ ਦਸ਼ਮਲਵ pi        = 3.14159265358979323846f;
    ਅਚਲ ਦਸ਼ਮਲਵ two_pi    = 6.28318530717958647693f;
    ਅਚਲ ਦਸ਼ਮਲਵ pi_over_2 = 1.57079632679489661923f;

    ਜੇ (x > pi || x < -pi) {
        ਦਸ਼ਮਲਵ ਚੱਕਰਾਂ = x * et_fdiv(1.0f, two_pi);
        ਪੂਰਨ   n      = (ਪੂਰਨ) ਚੱਕਰਾਂ;
        ਜੇ (x < 0.0f) {
            n--;  // Floor for negative
        }
        x = x - (ਦਸ਼ਮਲਵ) n * two_pi;
    }

    // sin(x) = sin(π - x) for x in [π/2, π]
    // sin(x) = -sin(-π - x) for x in [-π, -π/2]
    ਪੂਰਨ negate = 0;
    ਜੇ (x > pi_over_2) {
        x = pi - x;
    } ਨਹੀਂਤਾਂ ਜੇ (x < -pi_over_2) {
        x      = -pi - x;
        negate = 1;
    }

    // sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    ਅਚਲ ਦਸ਼ਮਲਵ x2  = x * x;
    ਅਚਲ ਦਸ਼ਮਲਵ x3  = x2 * x;
    ਅਚਲ ਦਸ਼ਮਲਵ x5  = x3 * x2;
    ਅਚਲ ਦਸ਼ਮਲਵ x7  = x5 * x2;
    ਅਚਲ ਦਸ਼ਮਲਵ x9  = x7 * x2;
    ਅਚਲ ਦਸ਼ਮਲਵ x11 = x9 * x2;

    ਦਸ਼ਮਲਵ ਨਤੀਜਾ = x - x3 * et_fdiv(1.0f, 6.0f)         // x^3/3!
                   + x5 * et_fdiv(1.0f, 120.0f)         // x^5/5!
                   - x7 * et_fdiv(1.0f, 5040.0f)        // x^7/7!
                   + x9 * et_fdiv(1.0f, 362880.0f)      // x^9/9!
                   - x11 * et_fdiv(1.0f, 39916800.0f);  // x^11/11!

    ਵਾਪਸ negate ? -ਨਤੀਜਾ : ਨਤੀਜਾ;
}

// Cosine function using identity cos(x) = sin(x + π/2)
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ et_cosf(ਦਸ਼ਮਲਵ x) {
    ਅਚਲ ਦਸ਼ਮਲਵ pi_over_2 = 1.57079632679489661923f;
    ਵਾਪਸ et_sinf(x + pi_over_2);
}

//******************************************************************************
// FP16 <-> FP32 Conversion Functions
//******************************************************************************

// Convert FP16 (IEEE 754 half precision) to FP32 (single precision)
// Uses ET hardware FCVT.PS.F16 instruction for accurate conversion
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ fp16_ਮੰਜ਼ਿਲ_fp32(uint16_t h) {
    ਦਸ਼ਮਲਵ         ਨਤੀਜਾ;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ;
    uint32_t      raw = (uint32_t) h;

    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[temp]              \n\t"    // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"    // Set mask register m0 to enable element 0
        "fbcx.ps   %[result], %[raw]    \n\t"    // Broadcast raw FP16 bits into vector register
        "fcvt.ps.f16 %[result], %[result] \n\t"  // Convert FP16 to FP32
        "mova.m.x  %[temp]              \n\t"    // Restore mask state
        : [ਆਰਜ਼ੀ] "=&r"(ਆਰਜ਼ੀ), [ਨਤੀਜਾ] "=&f"(ਨਤੀਜਾ)
        : [raw] "r"(raw));

    ਵਾਪਸ ਨਤੀਜਾ;
}

// Convert FP32 (single precision) to FP16 (IEEE 754 half precision)
// Uses ET hardware FCVT.F16.PS instruction for accurate conversion
ਸਥਿਰ ਸਤਰਅੰਦਰ uint16_t fp32_ਮੰਜ਼ਿਲ_fp16(ਦਸ਼ਮਲਵ f) {
    ਦਸ਼ਮਲਵ         ਨਤੀਜਾ_f;
    ਬਿਨਾਨਿਸ਼ਾਨੀ ਲੰਮਾ ਆਰਜ਼ੀ;

    __asm__ ਅਸਥਿਰ(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fcvt.f16.ps %[result], %[f]    \n\t"  // Convert FP32 to FP16 (result in lower 16 bits)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [ਆਰਜ਼ੀ] "=&r"(ਆਰਜ਼ੀ), [ਨਤੀਜਾ] "=&f"(ਨਤੀਜਾ_f)
        : [f] "f"(f));

    // Extract lower 16 bits containing the FP16 value
    // The instruction zero-extends to 32 bits, so upper 16 bits are 0
    uint32_t ਨਤੀਜਾ_ਬਿੱਟ = *(uint32_t *) &ਨਤੀਜਾ_f;
    ਵਾਪਸ (uint16_t) ਨਤੀਜਾ_ਬਿੱਟ;
}

#ਜੇ_ਅੰਤ  // MATH_FP_H
