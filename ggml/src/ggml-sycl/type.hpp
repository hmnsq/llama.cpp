#निर्देशः once

#अन्तर्भाव <sycl/sycl.hpp>
#अन्तर्भाव <cstdint>
#अन्तर्भाव <limits>

अन्तःस्थ uint8_t float_गन्तव्यम्_e4m3(भिन्न f)
{
    यदि (sycl::isnan(f)) {
        प्रत्यागम 0x7F;                    // Canonical NaN (positive)
    }

    uint32_t द्वयङ्काः = sycl::bit_cast<uint32_t>(f);
    uint32_t sign = (द्वयङ्काः >> 31) & 0x1u;
    uint32_t घातांकः  = (द्वयङ्काः >> 23) & 0xFFu;
    uint32_t mant = द्वयङ्काः & 0x7FFFFFu;

    // Zero
    यदि (घातांकः == 0 && mant == 0) {
        प्रत्यागम static_cast<uint8_t>(sign << 7);
    }

    // Extract biased exponent and mantissa for FP8
    पूर्णाङ्क e = static_cast<पूर्णाङ्क>(घातांकः) - 127;           // true exponent (IEEE bias 127)
    uint32_t m = mant;

    // Handle very large values → NaN (NVIDIA behavior for E4M3)
    यदि (e > 7) {                                   // max exponent for E4M3 is 7 (biased 14)
        प्रत्यागम static_cast<uint8_t>((sign << 7) | 0x7F);
    }

    // Handle subnormals and normal numbers
    यदि (e < -6) {                                  // smallest normal exponent is -6
        // Subnormal in FP8: shift mantissa right
        पूर्णाङ्क सरणम् = -6 - e;
        m = (m | 0x800000u) >> (सरणम् + 1);        // +1 because we lose the implicit 1 position
        यदि (सरणम् > 23) m = 0;
    } अन्यथा {
        // Normal number: adjust exponent bias from 127 to 7
        पूर्णाङ्क new_exp = e + 7;
        m = (m >> 20) & 0x7u;                      // take top 3 mantissa bits (after implicit 1)
        m |= (static_cast<uint32_t>(new_exp) << 3);
    }

    // Round-to-nearest-even (simple guard + round bit)
    // For better accuracy you can add sticky bit, but this is sufficient for most use cases
    uint32_t आवर्तनम्_bit = (mant >> 19) & 0x1u;      // bit after the 3 mantissa bits
    यदि (आवर्तनम्_bit) {
        m += 1;
        // Carry into exponent if mantissa overflows
        यदि ((m & 0x8u) != 0) {
            m = (m & 0x7u) | ((m & 0x38u) << 1);   // simple carry handling
            // If exponent overflows after carry → NaN
            यदि ((m >> 3) > 14) {
                प्रत्यागम static_cast<uint8_t>((sign << 7) | 0x7F);
            }
        }
    }

    uint8_t परिणाम = static_cast<uint8_t>((sign << 7) | (m & 0x7F));
    प्रत्यागम परिणाम;
}

अन्तःस्थ भिन्न e4m3_गन्तव्यम्_float(uint8_t x)
{
    यदि (x == 0) प्रत्यागम 0.0f;

    uint8_t sign = (x >> 7) & 0x1u;
    uint8_t घातांकः  = (x >> 3) & 0xFu;
    uint8_t mant = x & 0x7u;

    // NaN (NVIDIA uses 0x7F / 0xFF as NaN)
    यदि (घातांकः == 0xF && mant != 0) {
        प्रत्यागम std::numeric_limits<भिन्न>::quiet_NaN();
    }
    यदि (घातांकः == 0xF) {                     // 0x7F or 0xFF treated as NaN
        प्रत्यागम std::numeric_limits<भिन्न>::quiet_NaN();
    }

    भिन्न val;

    यदि (घातांकः == 0) {
        // Subnormal
        val = mant * (1.0f / 8.0f) * sycl::घातः(2.0f, -6.0f);
    } अन्यथा {
        // Normal: implicit leading 1 + bias 7
        val = (1.0f + mant / 8.0f) * sycl::घातः(2.0f, static_cast<भिन्न>(घातांकः) - 7.0f);
    }

    प्रत्यागम sign ? -val : val;
}

// The actual type definition
संरचना __nv_fp8_e4m3 {
    uint8_t raw;

    __nv_fp8_e4m3() = स्वभाव;

    स्पष्ट __nv_fp8_e4m3(भिन्न f) : raw(float_गन्तव्यम्_e4m3(f)) {}
    स्पष्ट __nv_fp8_e4m3(sycl::half h) : raw(float_गन्तव्यम्_e4m3(static_cast<भिन्न>(h))) {}

    संक्रियकः भिन्न() ध्रुव { प्रत्यागम e4m3_गन्तव्यम्_float(raw); }
    संक्रियकः sycl::half() ध्रुव { प्रत्यागम static_cast<sycl::half>(static_cast<भिन्न>(*एतत्)); }

    // Allow direct access for vector loads/stores
    संक्रियकः uint8_t&() { प्रत्यागम raw; }
    संक्रियकः uint8_t() ध्रुव { प्रत्यागम raw; }
};

उपयुज्य __nv_fp8x2_e4m3 = sycl::vec<__nv_fp8_e4m3, 2>;
उपयुज्य __nv_fp8x4_e4m3 = sycl::vec<__nv_fp8_e4m3, 4>;

