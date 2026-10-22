//******************************************************************************
// ET Floating Point Math Library
// Provides ET hardware-specific math functions, FP16 conversion, and trig functions
// for bare metal kernels
//******************************************************************************

#अगर_अघोषित MATH_FP_H
#घोषणा MATH_FP_H

#समावेश <stdint.h>

//******************************************************************************
// ET Hardware Math Functions
//******************************************************************************

// ET hardware division function (uses FRCP.PS instruction)
अटल अंतरभूत भग्न et_fdiv(भग्न a, भग्न b) {
    भग्न         d;
    अचिन्हित दीर्घ अस्थायी;

    __asm__ अस्थिर(
        "mova.x.m  %[temp]              \n\t"
        "mov.m.x   m0, x0, 1            \n\t"
        "frcp.ps   %[d], %[b]           \n\t"
        "fmul.s    %[d], %[d], %[a]     \n\t"
        "mova.m.x  %[temp]              \n\t"
        : [अस्थायी] "=&r"(अस्थायी), [d] "=&f"(d)
        : [a] "f"(a), [b] "f"(b));

    वापस d;
}

// Power function using ET hardware vector instructions
// Implements pow(base, exp) = exp(exp * ln(base)) using FLOG.PS and FEXP.PS
अटल अंतरभूत भग्न et_powf(भग्न base, भग्न exp) {
    // Handle special cases
    अगर (base <= 0.0f) {
        अगर (base == 0.0f) {
            अगर (exp > 0.0f) {
                वापस 0.0f;
            }

            // For exp <= 0, return +infinity (IEEE 754: sign=0, exp=0xFF, mantissa=0)
            जोड़ {
                भग्न    f;
                uint32_t i;
            } inf = { .i = 0x7F800000 };

            वापस inf.f;
        }

        // For negative base, return NaN (IEEE 754: exp=0xFF, mantissa!=0)
        जोड़ {
            भग्न    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        वापस nan.f;
    }
    अगर (base == 1.0f) {
        वापस 1.0f;
    }
    अगर (exp == 0.0f) {
        वापस 1.0f;
    }
    अगर (exp == 1.0f) {
        वापस base;
    }

    // Use ET hardware instructions following DNN library pattern:
    // pow(base, exp) = exp(exp * ln(base))
    भग्न         परिणाम;
    अचिन्हित दीर्घ अस्थायी;

    __asm__ अस्थिर(
        "mova.x.m  %[temp]              \n\t"      // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"      // Set mask register m0 to enable element 0
        "flog.ps %[result], %[base]     \n\t"      // result = ln(base)
        "fmul.s %[result], %[result], %[exp]\n\t"  // result = ln(base) * exp
        "fexp.ps %[result], %[result]   \n\t"      // result = exp(ln(base) * exp) = base^exp
        "mova.m.x  %[temp]              \n\t"      // Restore mask state
        : [अस्थायी] "=&r"(अस्थायी), [परिणाम] "=&f"(परिणाम)
        : [base] "f"(base), [exp] "f"(exp));

    वापस परिणाम;
}

// Natural logarithm.
अटल अंतरभूत भग्न et_logf(भग्न x) {
    // Handle special cases
    अगर (x < 0.0f) {
        // Return NaN for negative input
        जोड़ {
            भग्न    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        वापस nan.f;
    }
    अगर (x == 0.0f) {
        // Return -infinity for log(0)
        जोड़ {
            भग्न    f;
            uint32_t i;
        } inf = { .i = 0xFF800000 };

        वापस inf.f;
    }
    अगर (x == 1.0f) {
        वापस 0.0f;
    }

    भग्न         log2_परिणाम;
    अचिन्हित दीर्घ अस्थायी;

    __asm__ अस्थिर(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "flog.ps %[result], %[x]        \n\t"  // result = log2(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [अस्थायी] "=&r"(अस्थायी), [परिणाम] "=&f"(log2_परिणाम)
        : [x] "f"(x));

    // Convert log2 to natural log: ln(x) = log2(x) * ln(2)
    स्थिर भग्न ln2 = 0.69314718055994530942f;
    वापस log2_परिणाम * ln2;
}

// Square root function implemented as et_powf(x, 0.5)
अटल अंतरभूत भग्न et_sqrtf(भग्न x) {
    // Handle special cases
    अगर (x < 0.0f) {
        // Return NaN for negative input (IEEE 754: exp=0xFF, mantissa!=0)
        जोड़ {
            भग्न    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        वापस nan.f;
    }
    अगर (x == 0.0f) {
        वापस 0.0f;
    }

    वापस et_powf(x, 0.5f);
}

// Base-2 exponential: returns 2^x using the ET hardware FEXP.PS instruction.
// No base conversion, no special-case clamping — this is the raw hardware op
// with just the mask save/restore wrapper.  Caller is responsible for ensuring
// x is in a range that produces a useful result (roughly [-126, 128] for fp32).
अटल अंतरभूत भग्न __attribute__((always_inline)) et_exp2f(भग्न x) {
    अचिन्हित दीर्घ old_mask;
    भग्न         out;
    __asm__ अस्थिर(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 1         \n\t"
        "fexp.ps   %[out], %[x]      \n\t"
        "mova.m.x  %[ms]             \n\t"
        : [ms] "=&r"(old_mask), [out] "=&f"(out)
        : [x] "f"(x));
    वापस out;
}

// Exponential function using ET hardware FEXP.PS instruction
// Note: FEXP.PS computes 2^x, so we need to convert: exp(x) = 2^(x * log2(e))
अटल अंतरभूत भग्न et_expf(भग्न x) {
    // Handle special cases
    अगर (x > 88.0f) {
        // For x > 88, exp(x) would overflow, return +infinity
        जोड़ {
            भग्न    f;
            uint32_t i;
        } inf = { .i = 0x7F800000 };

        वापस inf.f;
    }
    अगर (x < -87.0f) {
        // For x < -87, exp(x) is essentially 0
        वापस 0.0f;
    }

    // Convert to base-2 exponent: x * log2(e)
    स्थिर भग्न log2e   = 1.4426950408889634f;  // log2(e)
    भग्न       x_log2e = x * log2e;

    // Use ET hardware instruction: fexp.ps computes 2^x
    भग्न         परिणाम;
    अचिन्हित दीर्घ अस्थायी;

    __asm__ अस्थिर(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fexp.ps %[result], %[x_log2e]  \n\t"  // result = 2^(x * log2(e)) = exp(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [अस्थायी] "=&r"(अस्थायी), [परिणाम] "=&f"(परिणाम)
        : [x_log2e] "f"(x_log2e));

    वापस परिणाम;
}

//******************************************************************************
// Trigonometric Functions
//******************************************************************************

// FSIN.PS

// Sine function using Taylor series
अटल अंतरभूत भग्न et_sinf(भग्न x) {
    स्थिर भग्न pi        = 3.14159265358979323846f;
    स्थिर भग्न two_pi    = 6.28318530717958647693f;
    स्थिर भग्न pi_over_2 = 1.57079632679489661923f;

    अगर (x > pi || x < -pi) {
        भग्न cycles = x * et_fdiv(1.0f, two_pi);
        पूर्णांक   n      = (पूर्णांक) cycles;
        अगर (x < 0.0f) {
            n--;  // Floor for negative
        }
        x = x - (भग्न) n * two_pi;
    }

    // sin(x) = sin(π - x) for x in [π/2, π]
    // sin(x) = -sin(-π - x) for x in [-π, -π/2]
    पूर्णांक negate = 0;
    अगर (x > pi_over_2) {
        x = pi - x;
    } अन्यथा अगर (x < -pi_over_2) {
        x      = -pi - x;
        negate = 1;
    }

    // sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    स्थिर भग्न x2  = x * x;
    स्थिर भग्न x3  = x2 * x;
    स्थिर भग्न x5  = x3 * x2;
    स्थिर भग्न x7  = x5 * x2;
    स्थिर भग्न x9  = x7 * x2;
    स्थिर भग्न x11 = x9 * x2;

    भग्न परिणाम = x - x3 * et_fdiv(1.0f, 6.0f)         // x^3/3!
                   + x5 * et_fdiv(1.0f, 120.0f)         // x^5/5!
                   - x7 * et_fdiv(1.0f, 5040.0f)        // x^7/7!
                   + x9 * et_fdiv(1.0f, 362880.0f)      // x^9/9!
                   - x11 * et_fdiv(1.0f, 39916800.0f);  // x^11/11!

    वापस negate ? -परिणाम : परिणाम;
}

// Cosine function using identity cos(x) = sin(x + π/2)
अटल अंतरभूत भग्न et_cosf(भग्न x) {
    स्थिर भग्न pi_over_2 = 1.57079632679489661923f;
    वापस et_sinf(x + pi_over_2);
}

//******************************************************************************
// FP16 <-> FP32 Conversion Functions
//******************************************************************************

// Convert FP16 (IEEE 754 half precision) to FP32 (single precision)
// Uses ET hardware FCVT.PS.F16 instruction for accurate conversion
अटल अंतरभूत भग्न fp16_to_fp32(uint16_t h) {
    भग्न         परिणाम;
    अचिन्हित दीर्घ अस्थायी;
    uint32_t      raw = (uint32_t) h;

    __asm__ अस्थिर(
        "mova.x.m  %[temp]              \n\t"    // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"    // Set mask register m0 to enable element 0
        "fbcx.ps   %[result], %[raw]    \n\t"    // Broadcast raw FP16 bits into vector register
        "fcvt.ps.f16 %[result], %[result] \n\t"  // Convert FP16 to FP32
        "mova.m.x  %[temp]              \n\t"    // Restore mask state
        : [अस्थायी] "=&r"(अस्थायी), [परिणाम] "=&f"(परिणाम)
        : [raw] "r"(raw));

    वापस परिणाम;
}

// Convert FP32 (single precision) to FP16 (IEEE 754 half precision)
// Uses ET hardware FCVT.F16.PS instruction for accurate conversion
अटल अंतरभूत uint16_t fp32_to_fp16(भग्न f) {
    भग्न         परिणाम_f;
    अचिन्हित दीर्घ अस्थायी;

    __asm__ अस्थिर(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fcvt.f16.ps %[result], %[f]    \n\t"  // Convert FP32 to FP16 (result in lower 16 bits)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [अस्थायी] "=&r"(अस्थायी), [परिणाम] "=&f"(परिणाम_f)
        : [f] "f"(f));

    // Extract lower 16 bits containing the FP16 value
    // The instruction zero-extends to 32 bits, so upper 16 bits are 0
    uint32_t परिणाम_bits = *(uint32_t *) &परिणाम_f;
    वापस (uint16_t) परिणाम_bits;
}

#पूर्ण_अगर  // MATH_FP_H
