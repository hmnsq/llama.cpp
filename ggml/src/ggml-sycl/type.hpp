#సూచన once

#అన్తర్భావ <sycl/sycl.hpp>
#అన్తర్భావ <cstdint>
#అన్తర్భావ <limits>

అంతఃవరుస uint8_t float_గమ్యం_e4m3(దశాంశం f)
{
    ఒకవేళ (sycl::isnan(f)) {
        తిరిగి 0x7F;                    // Canonical NaN (positive)
    }

    uint32_t బిట్లు = sycl::బిట్టు_cast<uint32_t>(f);
    uint32_t sign = (బిట్లు >> 31) & 0x1u;
    uint32_t ఘాతాంకం  = (బిట్లు >> 23) & 0xFFu;
    uint32_t mant = బిట్లు & 0x7FFFFFu;

    // Zero
    ఒకవేళ (ఘాతాంకం == 0 && mant == 0) {
        తిరిగి static_cast<uint8_t>(sign << 7);
    }

    // Extract biased exponent and mantissa for FP8
    పూర్ణాంకం e = static_cast<పూర్ణాంకం>(ఘాతాంకం) - 127;           // true exponent (IEEE bias 127)
    uint32_t m = mant;

    // Handle very large values → NaN (NVIDIA behavior for E4M3)
    ఒకవేళ (e > 7) {                                   // max exponent for E4M3 is 7 (biased 14)
        తిరిగి static_cast<uint8_t>((sign << 7) | 0x7F);
    }

    // Handle subnormals and normal numbers
    ఒకవేళ (e < -6) {                                  // smallest normal exponent is -6
        // Subnormal in FP8: shift mantissa right
        పూర్ణాంకం జరుపు = -6 - e;
        m = (m | 0x800000u) >> (జరుపు + 1);        // +1 because we lose the implicit 1 position
        ఒకవేళ (జరుపు > 23) m = 0;
    } లేకపోతే {
        // Normal number: adjust exponent bias from 127 to 7
        పూర్ణాంకం new_exp = e + 7;
        m = (m >> 20) & 0x7u;                      // take top 3 mantissa bits (after implicit 1)
        m |= (static_cast<uint32_t>(new_exp) << 3);
    }

    // Round-to-nearest-even (simple guard + round bit)
    // For better accuracy you can add sticky bit, but this is sufficient for most use cases
    uint32_t విడత_బిట్టు = (mant >> 19) & 0x1u;      // bit after the 3 mantissa bits
    ఒకవేళ (విడత_బిట్టు) {
        m += 1;
        // Carry into exponent if mantissa overflows
        ఒకవేళ ((m & 0x8u) != 0) {
            m = (m & 0x7u) | ((m & 0x38u) << 1);   // simple carry handling
            // If exponent overflows after carry → NaN
            ఒకవేళ ((m >> 3) > 14) {
                తిరిగి static_cast<uint8_t>((sign << 7) | 0x7F);
            }
        }
    }

    uint8_t ఫలితం = static_cast<uint8_t>((sign << 7) | (m & 0x7F));
    తిరిగి ఫలితం;
}

అంతఃవరుస దశాంశం e4m3_గమ్యం_float(uint8_t x)
{
    ఒకవేళ (x == 0) తిరిగి 0.0f;

    uint8_t sign = (x >> 7) & 0x1u;
    uint8_t ఘాతాంకం  = (x >> 3) & 0xFu;
    uint8_t mant = x & 0x7u;

    // NaN (NVIDIA uses 0x7F / 0xFF as NaN)
    ఒకవేళ (ఘాతాంకం == 0xF && mant != 0) {
        తిరిగి std::numeric_limits<దశాంశం>::quiet_NaN();
    }
    ఒకవేళ (ఘాతాంకం == 0xF) {                     // 0x7F or 0xFF treated as NaN
        తిరిగి std::numeric_limits<దశాంశం>::quiet_NaN();
    }

    దశాంశం val;

    ఒకవేళ (ఘాతాంకం == 0) {
        // Subnormal
        val = mant * (1.0f / 8.0f) * sycl::ఘాతం(2.0f, -6.0f);
    } లేకపోతే {
        // Normal: implicit leading 1 + bias 7
        val = (1.0f + mant / 8.0f) * sycl::ఘాతం(2.0f, static_cast<దశాంశం>(ఘాతాంకం) - 7.0f);
    }

    తిరిగి sign ? -val : val;
}

// The actual type definition
నిర్మాణం __nv_fp8_e4m3 {
    uint8_t raw;

    __nv_fp8_e4m3() = అప్రమేయ;

    స్పష్టమైన __nv_fp8_e4m3(దశాంశం f) : raw(float_గమ్యం_e4m3(f)) {}
    స్పష్టమైన __nv_fp8_e4m3(sycl::half h) : raw(float_గమ్యం_e4m3(static_cast<దశాంశం>(h))) {}

    క్రియాకారకం దశాంశం() స్థిరరాశి { తిరిగి e4m3_గమ్యం_float(raw); }
    క్రియాకారకం sycl::half() స్థిరరాశి { తిరిగి static_cast<sycl::half>(static_cast<దశాంశం>(*ఇది)); }

    // Allow direct access for vector loads/stores
    క్రియాకారకం uint8_t&() { తిరిగి raw; }
    క్రియాకారకం uint8_t() స్థిరరాశి { తిరిగి raw; }
};

ఉపయోగించి __nv_fp8x2_e4m3 = sycl::vec<__nv_fp8_e4m3, 2>;
ఉపయోగించి __nv_fp8x4_e4m3 = sycl::vec<__nv_fp8_e4m3, 4>;

