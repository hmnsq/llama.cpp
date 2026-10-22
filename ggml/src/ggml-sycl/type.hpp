#pragma once

#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <limits>

ಅನ್ತಃಸ್ಥ uint8_t float_to_e4m3(ಭಿನ್ನ f)
{
    ಯದಿ (sycl::isnan(f)) {
        ಪ್ರತ್ಯಾಗಮ 0x7F;                    // Canonical NaN (positive)
    }

    uint32_t bits = sycl::bit_cast<uint32_t>(f);
    uint32_t sign = (bits >> 31) & 0x1u;
    uint32_t exp  = (bits >> 23) & 0xFFu;
    uint32_t mant = bits & 0x7FFFFFu;

    // Zero
    ಯದಿ (exp == 0 && mant == 0) {
        ಪ್ರತ್ಯಾಗಮ static_cast<uint8_t>(sign << 7);
    }

    // Extract biased exponent and mantissa for FP8
    ಪೂರ್ಣಾಙ್ಕ e = static_cast<ಪೂರ್ಣಾಙ್ಕ>(exp) - 127;           // true exponent (IEEE bias 127)
    uint32_t m = mant;

    // Handle very large values → NaN (NVIDIA behavior for E4M3)
    ಯದಿ (e > 7) {                                   // max exponent for E4M3 is 7 (biased 14)
        ಪ್ರತ್ಯಾಗಮ static_cast<uint8_t>((sign << 7) | 0x7F);
    }

    // Handle subnormals and normal numbers
    ಯದಿ (e < -6) {                                  // smallest normal exponent is -6
        // Subnormal in FP8: shift mantissa right
        ಪೂರ್ಣಾಙ್ಕ shift = -6 - e;
        m = (m | 0x800000u) >> (shift + 1);        // +1 because we lose the implicit 1 position
        ಯದಿ (shift > 23) m = 0;
    } ಅನ್ಯಥಾ {
        // Normal number: adjust exponent bias from 127 to 7
        ಪೂರ್ಣಾಙ್ಕ new_exp = e + 7;
        m = (m >> 20) & 0x7u;                      // take top 3 mantissa bits (after implicit 1)
        m |= (static_cast<uint32_t>(new_exp) << 3);
    }

    // Round-to-nearest-even (simple guard + round bit)
    // For better accuracy you can add sticky bit, but this is sufficient for most use cases
    uint32_t round_bit = (mant >> 19) & 0x1u;      // bit after the 3 mantissa bits
    ಯದಿ (round_bit) {
        m += 1;
        // Carry into exponent if mantissa overflows
        ಯದಿ ((m & 0x8u) != 0) {
            m = (m & 0x7u) | ((m & 0x38u) << 1);   // simple carry handling
            // If exponent overflows after carry → NaN
            ಯದಿ ((m >> 3) > 14) {
                ಪ್ರತ್ಯಾಗಮ static_cast<uint8_t>((sign << 7) | 0x7F);
            }
        }
    }

    uint8_t result = static_cast<uint8_t>((sign << 7) | (m & 0x7F));
    ಪ್ರತ್ಯಾಗಮ result;
}

ಅನ್ತಃಸ್ಥ ಭಿನ್ನ e4m3_to_float(uint8_t x)
{
    ಯದಿ (x == 0) ಪ್ರತ್ಯಾಗಮ 0.0f;

    uint8_t sign = (x >> 7) & 0x1u;
    uint8_t exp  = (x >> 3) & 0xFu;
    uint8_t mant = x & 0x7u;

    // NaN (NVIDIA uses 0x7F / 0xFF as NaN)
    ಯದಿ (exp == 0xF && mant != 0) {
        ಪ್ರತ್ಯಾಗಮ std::numeric_limits<ಭಿನ್ನ>::quiet_NaN();
    }
    ಯದಿ (exp == 0xF) {                     // 0x7F or 0xFF treated as NaN
        ಪ್ರತ್ಯಾಗಮ std::numeric_limits<ಭಿನ್ನ>::quiet_NaN();
    }

    ಭಿನ್ನ val;

    ಯದಿ (exp == 0) {
        // Subnormal
        val = mant * (1.0f / 8.0f) * sycl::pow(2.0f, -6.0f);
    } ಅನ್ಯಥಾ {
        // Normal: implicit leading 1 + bias 7
        val = (1.0f + mant / 8.0f) * sycl::pow(2.0f, static_cast<ಭಿನ್ನ>(exp) - 7.0f);
    }

    ಪ್ರತ್ಯಾಗಮ sign ? -val : val;
}

// The actual type definition
ಸಂರಚನಾ __nv_fp8_e4m3 {
    uint8_t raw;

    __nv_fp8_e4m3() = ಸ್ವಭಾವ;

    explicit __nv_fp8_e4m3(ಭಿನ್ನ f) : raw(float_to_e4m3(f)) {}
    explicit __nv_fp8_e4m3(sycl::half h) : raw(float_to_e4m3(static_cast<ಭಿನ್ನ>(h))) {}

    operator ಭಿನ್ನ() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ e4m3_to_float(raw); }
    operator sycl::half() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ static_cast<sycl::half>(static_cast<ಭಿನ್ನ>(*ಏತತ್)); }

    // Allow direct access for vector loads/stores
    operator uint8_t&() { ಪ್ರತ್ಯಾಗಮ raw; }
    operator uint8_t() ಧ್ರುವ { ಪ್ರತ್ಯಾಗಮ raw; }
};

using __nv_fp8x2_e4m3 = sycl::vec<__nv_fp8_e4m3, 2>;
using __nv_fp8x4_e4m3 = sycl::vec<__nv_fp8_e4m3, 4>;

