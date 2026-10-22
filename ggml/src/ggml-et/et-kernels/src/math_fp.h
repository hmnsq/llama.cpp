//******************************************************************************
// ET Floating Point Math Library
// Provides ET hardware-specific math functions, FP16 conversion, and trig functions
// for bare metal kernels
//******************************************************************************

#నిర్వచితం_కాకపోతే MATH_FP_H
#నిర్వచించు MATH_FP_H

#అన్తర్భావ <stdint.h>

//******************************************************************************
// ET Hardware Math Functions
//******************************************************************************

// ET hardware division function (uses FRCP.PS instruction)
స్థిర అంతఃవరుస దశాంశం et_fdiv(దశాంశం a, దశాంశం b) {
    దశాంశం         d;
    గుర్తులేని దీర్ఘ తాత్కాలిక;

    __asm__ అస్థిర(
        "mova.x.m  %[temp]              \n\t"
        "mov.m.x   m0, x0, 1            \n\t"
        "frcp.ps   %[d], %[b]           \n\t"
        "fmul.s    %[d], %[d], %[a]     \n\t"
        "mova.m.x  %[temp]              \n\t"
        : [తాత్కాలిక] "=&r"(తాత్కాలిక), [d] "=&f"(d)
        : [a] "f"(a), [b] "f"(b));

    తిరిగి d;
}

// Power function using ET hardware vector instructions
// Implements pow(base, exp) = exp(exp * ln(base)) using FLOG.PS and FEXP.PS
స్థిర అంతఃవరుస దశాంశం et_powf(దశాంశం పునాది, దశాంశం ఘాతాంకం) {
    // Handle special cases
    ఒకవేళ (పునాది <= 0.0f) {
        ఒకవేళ (పునాది == 0.0f) {
            ఒకవేళ (ఘాతాంకం > 0.0f) {
                తిరిగి 0.0f;
            }

            // For exp <= 0, return +infinity (IEEE 754: sign=0, exp=0xFF, mantissa=0)
            సంఘం {
                దశాంశం    f;
                uint32_t i;
            } inf = { .i = 0x7F800000 };

            తిరిగి inf.f;
        }

        // For negative base, return NaN (IEEE 754: exp=0xFF, mantissa!=0)
        సంఘం {
            దశాంశం    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        తిరిగి nan.f;
    }
    ఒకవేళ (పునాది == 1.0f) {
        తిరిగి 1.0f;
    }
    ఒకవేళ (ఘాతాంకం == 0.0f) {
        తిరిగి 1.0f;
    }
    ఒకవేళ (ఘాతాంకం == 1.0f) {
        తిరిగి పునాది;
    }

    // Use ET hardware instructions following DNN library pattern:
    // pow(base, exp) = exp(exp * ln(base))
    దశాంశం         ఫలితం;
    గుర్తులేని దీర్ఘ తాత్కాలిక;

    __asm__ అస్థిర(
        "mova.x.m  %[temp]              \n\t"      // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"      // Set mask register m0 to enable element 0
        "flog.ps %[result], %[base]     \n\t"      // result = ln(base)
        "fmul.s %[result], %[result], %[exp]\n\t"  // result = ln(base) * exp
        "fexp.ps %[result], %[result]   \n\t"      // result = exp(ln(base) * exp) = base^exp
        "mova.m.x  %[temp]              \n\t"      // Restore mask state
        : [తాత్కాలిక] "=&r"(తాత్కాలిక), [ఫలితం] "=&f"(ఫలితం)
        : [పునాది] "f"(పునాది), [ఘాతాంకం] "f"(ఘాతాంకం));

    తిరిగి ఫలితం;
}

// Natural logarithm.
స్థిర అంతఃవరుస దశాంశం et_logf(దశాంశం x) {
    // Handle special cases
    ఒకవేళ (x < 0.0f) {
        // Return NaN for negative input
        సంఘం {
            దశాంశం    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        తిరిగి nan.f;
    }
    ఒకవేళ (x == 0.0f) {
        // Return -infinity for log(0)
        సంఘం {
            దశాంశం    f;
            uint32_t i;
        } inf = { .i = 0xFF800000 };

        తిరిగి inf.f;
    }
    ఒకవేళ (x == 1.0f) {
        తిరిగి 0.0f;
    }

    దశాంశం         log2_ఫలితం;
    గుర్తులేని దీర్ఘ తాత్కాలిక;

    __asm__ అస్థిర(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "flog.ps %[result], %[x]        \n\t"  // result = log2(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [తాత్కాలిక] "=&r"(తాత్కాలిక), [ఫలితం] "=&f"(log2_ఫలితం)
        : [x] "f"(x));

    // Convert log2 to natural log: ln(x) = log2(x) * ln(2)
    స్థిరరాశి దశాంశం ln2 = 0.69314718055994530942f;
    తిరిగి log2_ఫలితం * ln2;
}

// Square root function implemented as et_powf(x, 0.5)
స్థిర అంతఃవరుస దశాంశం et_sqrtf(దశాంశం x) {
    // Handle special cases
    ఒకవేళ (x < 0.0f) {
        // Return NaN for negative input (IEEE 754: exp=0xFF, mantissa!=0)
        సంఘం {
            దశాంశం    f;
            uint32_t i;
        } nan = { .i = 0x7FC00000 };

        తిరిగి nan.f;
    }
    ఒకవేళ (x == 0.0f) {
        తిరిగి 0.0f;
    }

    తిరిగి et_powf(x, 0.5f);
}

// Base-2 exponential: returns 2^x using the ET hardware FEXP.PS instruction.
// No base conversion, no special-case clamping — this is the raw hardware op
// with just the mask save/restore wrapper.  Caller is responsible for ensuring
// x is in a range that produces a useful result (roughly [-126, 128] for fp32).
స్థిర అంతఃవరుస దశాంశం __attribute__((always_inline)) et_exp2f(దశాంశం x) {
    గుర్తులేని దీర్ఘ old_ముసుగు;
    దశాంశం         బహిర్గతం;
    __asm__ అస్థిర(
        "mova.x.m  %[ms]             \n\t"
        "mov.m.x   m0, x0, 1         \n\t"
        "fexp.ps   %[out], %[x]      \n\t"
        "mova.m.x  %[ms]             \n\t"
        : [ms] "=&r"(old_ముసుగు), [బహిర్గతం] "=&f"(బహిర్గతం)
        : [x] "f"(x));
    తిరిగి బహిర్గతం;
}

// Exponential function using ET hardware FEXP.PS instruction
// Note: FEXP.PS computes 2^x, so we need to convert: exp(x) = 2^(x * log2(e))
స్థిర అంతఃవరుస దశాంశం et_expf(దశాంశం x) {
    // Handle special cases
    ఒకవేళ (x > 88.0f) {
        // For x > 88, exp(x) would overflow, return +infinity
        సంఘం {
            దశాంశం    f;
            uint32_t i;
        } inf = { .i = 0x7F800000 };

        తిరిగి inf.f;
    }
    ఒకవేళ (x < -87.0f) {
        // For x < -87, exp(x) is essentially 0
        తిరిగి 0.0f;
    }

    // Convert to base-2 exponent: x * log2(e)
    స్థిరరాశి దశాంశం log2e   = 1.4426950408889634f;  // log2(e)
    దశాంశం       x_log2e = x * log2e;

    // Use ET hardware instruction: fexp.ps computes 2^x
    దశాంశం         ఫలితం;
    గుర్తులేని దీర్ఘ తాత్కాలిక;

    __asm__ అస్థిర(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fexp.ps %[result], %[x_log2e]  \n\t"  // result = 2^(x * log2(e)) = exp(x)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [తాత్కాలిక] "=&r"(తాత్కాలిక), [ఫలితం] "=&f"(ఫలితం)
        : [x_log2e] "f"(x_log2e));

    తిరిగి ఫలితం;
}

//******************************************************************************
// Trigonometric Functions
//******************************************************************************

// FSIN.PS

// Sine function using Taylor series
స్థిర అంతఃవరుస దశాంశం et_sinf(దశాంశం x) {
    స్థిరరాశి దశాంశం pi        = 3.14159265358979323846f;
    స్థిరరాశి దశాంశం two_pi    = 6.28318530717958647693f;
    స్థిరరాశి దశాంశం pi_over_2 = 1.57079632679489661923f;

    ఒకవేళ (x > pi || x < -pi) {
        దశాంశం చక్రాలు = x * et_fdiv(1.0f, two_pi);
        పూర్ణాంకం   n      = (పూర్ణాంకం) చక్రాలు;
        ఒకవేళ (x < 0.0f) {
            n--;  // Floor for negative
        }
        x = x - (దశాంశం) n * two_pi;
    }

    // sin(x) = sin(π - x) for x in [π/2, π]
    // sin(x) = -sin(-π - x) for x in [-π, -π/2]
    పూర్ణాంకం negate = 0;
    ఒకవేళ (x > pi_over_2) {
        x = pi - x;
    } లేకపోతే ఒకవేళ (x < -pi_over_2) {
        x      = -pi - x;
        negate = 1;
    }

    // sin(x) ≈ x - x^3/3! + x^5/5! - x^7/7! + x^9/9! - x^11/11!
    స్థిరరాశి దశాంశం x2  = x * x;
    స్థిరరాశి దశాంశం x3  = x2 * x;
    స్థిరరాశి దశాంశం x5  = x3 * x2;
    స్థిరరాశి దశాంశం x7  = x5 * x2;
    స్థిరరాశి దశాంశం x9  = x7 * x2;
    స్థిరరాశి దశాంశం x11 = x9 * x2;

    దశాంశం ఫలితం = x - x3 * et_fdiv(1.0f, 6.0f)         // x^3/3!
                   + x5 * et_fdiv(1.0f, 120.0f)         // x^5/5!
                   - x7 * et_fdiv(1.0f, 5040.0f)        // x^7/7!
                   + x9 * et_fdiv(1.0f, 362880.0f)      // x^9/9!
                   - x11 * et_fdiv(1.0f, 39916800.0f);  // x^11/11!

    తిరిగి negate ? -ఫలితం : ఫలితం;
}

// Cosine function using identity cos(x) = sin(x + π/2)
స్థిర అంతఃవరుస దశాంశం et_cosf(దశాంశం x) {
    స్థిరరాశి దశాంశం pi_over_2 = 1.57079632679489661923f;
    తిరిగి et_sinf(x + pi_over_2);
}

//******************************************************************************
// FP16 <-> FP32 Conversion Functions
//******************************************************************************

// Convert FP16 (IEEE 754 half precision) to FP32 (single precision)
// Uses ET hardware FCVT.PS.F16 instruction for accurate conversion
స్థిర అంతఃవరుస దశాంశం fp16_గమ్యం_fp32(uint16_t h) {
    దశాంశం         ఫలితం;
    గుర్తులేని దీర్ఘ తాత్కాలిక;
    uint32_t      raw = (uint32_t) h;

    __asm__ అస్థిర(
        "mova.x.m  %[temp]              \n\t"    // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"    // Set mask register m0 to enable element 0
        "fbcx.ps   %[result], %[raw]    \n\t"    // Broadcast raw FP16 bits into vector register
        "fcvt.ps.f16 %[result], %[result] \n\t"  // Convert FP16 to FP32
        "mova.m.x  %[temp]              \n\t"    // Restore mask state
        : [తాత్కాలిక] "=&r"(తాత్కాలిక), [ఫలితం] "=&f"(ఫలితం)
        : [raw] "r"(raw));

    తిరిగి ఫలితం;
}

// Convert FP32 (single precision) to FP16 (IEEE 754 half precision)
// Uses ET hardware FCVT.F16.PS instruction for accurate conversion
స్థిర అంతఃవరుస uint16_t fp32_గమ్యం_fp16(దశాంశం f) {
    దశాంశం         ఫలితం_f;
    గుర్తులేని దీర్ఘ తాత్కాలిక;

    __asm__ అస్థిర(
        "mova.x.m  %[temp]              \n\t"  // Save current mask state
        "mov.m.x   m0, x0, 1            \n\t"  // Set mask register m0 to enable element 0
        "fcvt.f16.ps %[result], %[f]    \n\t"  // Convert FP32 to FP16 (result in lower 16 bits)
        "mova.m.x  %[temp]              \n\t"  // Restore mask state
        : [తాత్కాలిక] "=&r"(తాత్కాలిక), [ఫలితం] "=&f"(ఫలితం_f)
        : [f] "f"(f));

    // Extract lower 16 bits containing the FP16 value
    // The instruction zero-extends to 32 bits, so upper 16 bits are 0
    uint32_t ఫలితం_బిట్లు = *(uint32_t *) &ఫలితం_f;
    తిరిగి (uint16_t) ఫలితం_బిట్లు;
}

#అయితే_ముగింపు  // MATH_FP_H
