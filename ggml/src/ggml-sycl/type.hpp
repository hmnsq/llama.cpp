#आशय once

#समावेश <sycl/sycl.hpp>
#समावेश <cstdint>
#समावेश <limits>

अंतरभूत uint8_t float_to_e4m3(भग्न f)
{
    अगर (sycl::isnan(f)) {
        वापस 0x7F;                    // Canonical NaN (positive)
    }

    uint32_t bits = sycl::bit_cast<uint32_t>(f);
    uint32_t sign = (bits >> 31) & 0x1u;
    uint32_t exp  = (bits >> 23) & 0xFFu;
    uint32_t mant = bits & 0x7FFFFFu;

    // Zero
    अगर (exp == 0 && mant == 0) {
        वापस static_cast<uint8_t>(sign << 7);
    }

    // Extract biased exponent and mantissa for FP8
    पूर्णांक e = static_cast<पूर्णांक>(exp) - 127;           // true exponent (IEEE bias 127)
    uint32_t m = mant;

    // Handle very large values → NaN (NVIDIA behavior for E4M3)
    अगर (e > 7) {                                   // max exponent for E4M3 is 7 (biased 14)
        वापस static_cast<uint8_t>((sign << 7) | 0x7F);
    }

    // Handle subnormals and normal numbers
    अगर (e < -6) {                                  // smallest normal exponent is -6
        // Subnormal in FP8: shift mantissa right
        पूर्णांक shift = -6 - e;
        m = (m | 0x800000u) >> (shift + 1);        // +1 because we lose the implicit 1 position
        अगर (shift > 23) m = 0;
    } अन्यथा {
        // Normal number: adjust exponent bias from 127 to 7
        पूर्णांक new_exp = e + 7;
        m = (m >> 20) & 0x7u;                      // take top 3 mantissa bits (after implicit 1)
        m |= (static_cast<uint32_t>(new_exp) << 3);
    }

    // Round-to-nearest-even (simple guard + round bit)
    // For better accuracy you can add sticky bit, but this is sufficient for most use cases
    uint32_t round_bit = (mant >> 19) & 0x1u;      // bit after the 3 mantissa bits
    अगर (round_bit) {
        m += 1;
        // Carry into exponent if mantissa overflows
        अगर ((m & 0x8u) != 0) {
            m = (m & 0x7u) | ((m & 0x38u) << 1);   // simple carry handling
            // If exponent overflows after carry → NaN
            अगर ((m >> 3) > 14) {
                वापस static_cast<uint8_t>((sign << 7) | 0x7F);
            }
        }
    }

    uint8_t परिणाम = static_cast<uint8_t>((sign << 7) | (m & 0x7F));
    वापस परिणाम;
}

अंतरभूत भग्न e4m3_to_float(uint8_t x)
{
    अगर (x == 0) वापस 0.0f;

    uint8_t sign = (x >> 7) & 0x1u;
    uint8_t exp  = (x >> 3) & 0xFu;
    uint8_t mant = x & 0x7u;

    // NaN (NVIDIA uses 0x7F / 0xFF as NaN)
    अगर (exp == 0xF && mant != 0) {
        वापस std::numeric_limits<भग्न>::quiet_NaN();
    }
    अगर (exp == 0xF) {                     // 0x7F or 0xFF treated as NaN
        वापस std::numeric_limits<भग्न>::quiet_NaN();
    }

    भग्न val;

    अगर (exp == 0) {
        // Subnormal
        val = mant * (1.0f / 8.0f) * sycl::घात(2.0f, -6.0f);
    } अन्यथा {
        // Normal: implicit leading 1 + bias 7
        val = (1.0f + mant / 8.0f) * sycl::घात(2.0f, static_cast<भग्न>(exp) - 7.0f);
    }

    वापस sign ? -val : val;
}

// The actual type definition
काष्ठा __nv_fp8_e4m3 {
    uint8_t raw;

    __nv_fp8_e4m3() = शेष;

    explicit __nv_fp8_e4m3(भग्न f) : raw(float_to_e4m3(f)) {}
    explicit __nv_fp8_e4m3(sycl::half h) : raw(float_to_e4m3(static_cast<भग्न>(h))) {}

    चालक भग्न() स्थिर { वापस e4m3_to_float(raw); }
    चालक sycl::half() स्थिर { वापस static_cast<sycl::half>(static_cast<भग्न>(*यह)); }

    // Allow direct access for vector loads/stores
    चालक uint8_t&() { वापस raw; }
    चालक uint8_t() स्थिर { वापस raw; }
};

using __nv_fp8x2_e4m3 = sycl::vec<__nv_fp8_e4m3, 2>;
using __nv_fp8x4_e4m3 = sycl::vec<__nv_fp8_e4m3, 4>;

