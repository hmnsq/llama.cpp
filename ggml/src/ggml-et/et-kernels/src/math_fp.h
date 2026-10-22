//******************************************************************************
// ET Floating Point Math Library
// Provides ET hardware-specific math functions, FP16 conversion, and trig functions
// for bare metal kernels
//******************************************************************************

#यदि_अपरिभाषितम् MATH_FP_H
#परिभाषय MATH_FP_H

#अन्तर्भाव <stdint.h>

//******************************************************************************
// ET Hardware Math Functions
//******************************************************************************

// ET hardware division function (uses FRCP.PS instruction)
स्थिर अन्तःस्थ भिन्न et_fdiv(भिन्न a, भिन्न b) {
    भिन्न         d;
    अचिह्नित दीर्घ अस्थायी;

    __asm__ चञ्चल(
        "mova.x.m  %[temp]              \n\t"
        "mov.m.x   m0, x0, 1            \n\t"
        "frcp.ps   %[d], %[b]           \n\t"
        "fmul.s    %[d], %[d], %[a]     \n\t"
        "mova.m.x  %[temp]              \n\t"
        : [अस्थायी] "=&r"(अस्थायी), [d] "=&f"(d)
        : [a] "f"(a), [b] "f"(b));

    प्रत्यागम d;
}

// Power function using ET hardware vector instructions
// Implements pow(base, exp) = exp(exp * ln(base)) using FLOG.PS and FEXP.PS
स्थिर अन्तःस्थ भिन्न et_powf(भिन्न आधारः, भिन्न घातांकः) {
    // Handle special cases
    यदि (आधारः <= 0.0f) {
        यदि (आधारः == 0.0f) {
            यदि (घातांकः > 0.0f) {
                प्रत्यागम 0.0f;
            }

            // For exp <= 0, return +infinity (IEEE 754: sign=0, exp=0xFF, mantissa=0)
            सङ्घ {
                भिन्न    f;
                uint32_t i;
            } inf = { .i = 0x7F800000 };

            प्रत्यागम inf.f;
        }

        // For negative base, return NaN (IEEE 754: exp=0xFF, mantissa!=0)
        सङ्घ {
            भिन्न    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        प्रत्यागम nan.f;
    }
    यदि (आधारः == 1.0f) {
        प्रत्यागम 1.0f;
    }
    यदि (घातांकः == 0.0f) {
        प्रत्यागम 1.0f;
    }
    यदि (घातांकः == 1.0f) {
        प्रत्यागम आधारः;
    }

    // Use ET hardware instructions following DNN library pattern:
    // pow(base, exp) = exp(exp * ln(base))
    भिन्न         परिणाम;
    अचिह्नित दीर्घ अस्थायी;

    __asm__ चञ्चल(
        "mova.x.m  %[temp]              \n\t"      // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"      // Set mask register m0 to enable element 0
        "flog.ps %[result], %[base]     \n\t"      // result = ln(base)
        "fmul.s %[result], %[result], %[exp]\n\t"  // result = ln(base) * exp
        "fexp.ps %[result], %[result]   \n\t"      // result = exp(ln(base) * exp) = base^exp
        "mova.m.x  %[temp]              \n\t"      // Restore mask state
        : [अस्थायी] "=&r"(अस्थायी), [परिणाम] "=&f"(परिणाम)
        : [आधारः] "f"(आधारः), [घातांकः] "f"(घातांकः));

    प्रत्यागम परिणाम;
}

// Natural logarithm.
स्थिर अन्तःस्थ भिन्न et_logf(भिन्न x) {
    // Handle special cases
    यदि (x < 0.0f) {
        // Return NaN for negative input
        सङ्घ {
            भिन्न    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        प्रत्यागम nan.f;
    }
    यदि (x == 0.0f) {
        // Return -infinity for log(0)
        सङ्घ {
            भिन्न    f;
            uint32_t i;
        } inf = { .i = 0xFF800000 };

        प्रत्यागम inf.f;
    }
    यदि (x == 1.0f) {
        प्रत्यागम 0.0f;
    }

    भिन्न         log2_परिणाम;
    अचिह्नित दीर्घ अस्थायी;

    __asm__ चञ्चल(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "flog.ps %[result], %[x]        \n\t"  // result = log2(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [अस्थायी] "=&r"(अस्थायी), [परिणाम] "=&f"(log2_परिणाम)
        : [x] "f"(x));

    // Convert log2 to natural log: ln(x) = log2(x) * ln(2)
    ध्रुव भिन्न ln2 = 0.69314718055994530942f;
    प्रत्यागम log2_परिणाम * ln2;
}

// Square root function implemented as et_powf(x, 0.5)
स्थिर अन्तःस्थ भिन्न et_sqrtf(भिन्न x) {
    // Handle special cases
    यदि (x < 0.0f) {
        // Return NaN for negative input (IEEE 754: exp=0xFF, mantissa!=0)
        सङ्घ {
            भिन्न    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        प्रत्यागम nan.f;
    }
    यदि (x == 0.0f) {
        प्रत्यागम 0.0f;
    }

    प्रत्यागम et_powf(x, 0.5f);
}

// Base-2 exponential: returns 2^x using the ET hardware FEXP.PS instruction.
// No base conversion, no special-case clamping — this is the raw hardware op
// with just the mask save/restore wrapper.  Caller is responsible for ensuring
// x is in a range that produces a useful result (roughly [-126, 128] for fp32).
स्थिर अन्तःस्थ भिन्न __attribute__((always_inline)) et_exp2f(भिन्न x) {
    अचिह्नित दीर्घ old_आवरणम्;
    भिन्न         बहिर्गमनम्;
    __asm__ चञ्चल(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 1         \n\t"
        "fexp.ps   %[out], %[x]      \n\t"
        "mova.m.x  %[ms]             \n\t"
        : [ms] "=&r"(old_आवरणम्), [बहिर्गमनम्] "=&f"(बहिर्गमनम्)
        : [x] "f"(x));
    प्रत्यागम बहिर्गमनम्;
}

// Exponential function using ET hardware FEXP.PS instruction
// Note: FEXP.PS computes 2^x, so we need to convert: exp(x) = 2^(x * log2(e))
स्थिर अन्तःस्थ भिन्न et_expf(भिन्न x) {
    // Handle special cases
    यदि (x > 88.0f) {
        // For x > 88, exp(x) would overflow, return +infinity
        सङ्घ {
            भिन्न    f;
            uint32_t i;
        } inf = { .i = 0x7F800000 };

        प्रत्यागम inf.f;
    }
    यदि (x < -87.0f) {
        // For x < -87, exp(x) is essentially 0
        प्रत्यागम 0.0f;
    }

    // Convert to base-2 exponent: x * log2(e)
    ध्रुव भिन्न log2e   = 1.4426950408889634f;  // log2(e)
    भिन्न       x_log2e = x * log2e;

    // Use ET hardware instruction: fexp.ps computes 2^x
    भिन्न         परिणाम;
    अचिह्नित दीर्घ अस्थायी;

    __asm__ चञ्चल(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fexp.ps %[result], %[x_log2e]  \n\t"  // result = 2^(x * log2(e)) = exp(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [अस्थायी] "=&r"(अस्थायी), [परिणाम] "=&f"(परिणाम)
        : [x_log2e] "f"(x_log2e));

    प्रत्यागम परिणाम;
}

//******************************************************************************
// Trigonometric Functions
//******************************************************************************

// FSIN.PS

// Sine function using Taylor series
स्थिर अन्तःस्थ भिन्न et_sinf(भिन्न x) {
    ध्रुव भिन्न pi        = 3.14159265358979323846f;
    ध्रुव भिन्न two_pi    = 6.28318530717958647693f;
    ध्रुव भिन्न pi_over_2 = 1.57079632679489661923f;

    यदि (x > pi || x < -pi) {
        भिन्न आवृत्तयः = x * et_fdiv(1.0f, two_pi);
        पूर्णाङ्क   n      = (पूर्णाङ्क) आवृत्तयः;
        यदि (x < 0.0f) {
            n--;  // Floor for negative
        }
        x = x - (भिन्न) n * two_pi;
    }

    // sin(x) = sin(π - x) for x in [π/2, π]
    // sin(x) = -sin(-π - x) for x in [-π, -π/2]
    पूर्णाङ्क negate = 0;
    यदि (x > pi_over_2) {
        x = pi - x;
    } अन्यथा यदि (x < -pi_over_2) {
        x      = -pi - x;
        negate = 1;
    }

    // sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    ध्रुव भिन्न x2  = x * x;
    ध्रुव भिन्न x3  = x2 * x;
    ध्रुव भिन्न x5  = x3 * x2;
    ध्रुव भिन्न x7  = x5 * x2;
    ध्रुव भिन्न x9  = x7 * x2;
    ध्रुव भिन्न x11 = x9 * x2;

    भिन्न परिणाम = x - x3 * et_fdiv(1.0f, 6.0f)         // x^3/3!
                   + x5 * et_fdiv(1.0f, 120.0f)         // x^5/5!
                   - x7 * et_fdiv(1.0f, 5040.0f)        // x^7/7!
                   + x9 * et_fdiv(1.0f, 362880.0f)      // x^9/9!
                   - x11 * et_fdiv(1.0f, 39916800.0f);  // x^11/11!

    प्रत्यागम negate ? -परिणाम : परिणाम;
}

// Cosine function using identity cos(x) = sin(x + π/2)
स्थिर अन्तःस्थ भिन्न et_cosf(भिन्न x) {
    ध्रुव भिन्न pi_over_2 = 1.57079632679489661923f;
    प्रत्यागम et_sinf(x + pi_over_2);
}

//******************************************************************************
// FP16 <-> FP32 Conversion Functions
//******************************************************************************

// Convert FP16 (IEEE 754 half precision) to FP32 (single precision)
// Uses ET hardware FCVT.PS.F16 instruction for accurate conversion
स्थिर अन्तःस्थ भिन्न fp16_गन्तव्यम्_fp32(uint16_t h) {
    भिन्न         परिणाम;
    अचिह्नित दीर्घ अस्थायी;
    uint32_t      raw = (uint32_t) h;

    __asm__ चञ्चल(
        "mova.x.m  %[temp]              \n\t"    // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"    // Set mask register m0 to enable element 0
        "fbcx.ps   %[result], %[raw]    \n\t"    // Broadcast raw FP16 bits into vector register
        "fcvt.ps.f16 %[result], %[result] \n\t"  // Convert FP16 to FP32
        "mova.m.x  %[temp]              \n\t"    // Restore mask state
        : [अस्थायी] "=&r"(अस्थायी), [परिणाम] "=&f"(परिणाम)
        : [raw] "r"(raw));

    प्रत्यागम परिणाम;
}

// Convert FP32 (single precision) to FP16 (IEEE 754 half precision)
// Uses ET hardware FCVT.F16.PS instruction for accurate conversion
स्थिर अन्तःस्थ uint16_t fp32_गन्तव्यम्_fp16(भिन्न f) {
    भिन्न         परिणाम_f;
    अचिह्नित दीर्घ अस्थायी;

    __asm__ चञ्चल(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fcvt.f16.ps %[result], %[f]    \n\t"  // Convert FP32 to FP16 (result in lower 16 bits)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [अस्थायी] "=&r"(अस्थायी), [परिणाम] "=&f"(परिणाम_f)
        : [f] "f"(f));

    // Extract lower 16 bits containing the FP16 value
    // The instruction zero-extends to 32 bits, so upper 16 bits are 0
    uint32_t परिणाम_द्वयङ्काः = *(uint32_t *) &परिणाम_f;
    प्रत्यागम (uint16_t) परिणाम_द्वयङ्काः;
}

#यदि_अन्तः  // MATH_FP_H
