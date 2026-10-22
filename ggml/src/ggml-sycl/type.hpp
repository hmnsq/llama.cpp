#pragma once

#অন্তর্ভাব <sycl/sycl.hpp>
#অন্তর্ভাব <cstdint>
#অন্তর্ভাব <limits>

অন্তঃরেখ uint8_t float_to_e4m3(দশমিক f)
{
    যদি (sycl::isnan(f)) {
        ফেরত 0x7F;                    // Canonical NaN (positive)
    }

    uint32_t bits = sycl::bit_cast<uint32_t>(f);
    uint32_t sign = (bits >> 31) & 0x1u;
    uint32_t exp  = (bits >> 23) & 0xFFu;
    uint32_t mant = bits & 0x7FFFFFu;

    // Zero
    যদি (exp == 0 && mant == 0) {
        ফেরত static_cast<uint8_t>(sign << 7);
    }

    // Extract biased exponent and mantissa for FP8
    পূর্ণসংখ্যা e = static_cast<পূর্ণসংখ্যা>(exp) - 127;           // true exponent (IEEE bias 127)
    uint32_t m = mant;

    // Handle very large values → NaN (NVIDIA behavior for E4M3)
    যদি (e > 7) {                                   // max exponent for E4M3 is 7 (biased 14)
        ফেরত static_cast<uint8_t>((sign << 7) | 0x7F);
    }

    // Handle subnormals and normal numbers
    যদি (e < -6) {                                  // smallest normal exponent is -6
        // Subnormal in FP8: shift mantissa right
        পূর্ণসংখ্যা shift = -6 - e;
        m = (m | 0x800000u) >> (shift + 1);        // +1 because we lose the implicit 1 position
        যদি (shift > 23) m = 0;
    } অন্যথা {
        // Normal number: adjust exponent bias from 127 to 7
        পূর্ণসংখ্যা new_exp = e + 7;
        m = (m >> 20) & 0x7u;                      // take top 3 mantissa bits (after implicit 1)
        m |= (static_cast<uint32_t>(new_exp) << 3);
    }

    // Round-to-nearest-even (simple guard + round bit)
    // For better accuracy you can add sticky bit, but this is sufficient for most use cases
    uint32_t round_bit = (mant >> 19) & 0x1u;      // bit after the 3 mantissa bits
    যদি (round_bit) {
        m += 1;
        // Carry into exponent if mantissa overflows
        যদি ((m & 0x8u) != 0) {
            m = (m & 0x7u) | ((m & 0x38u) << 1);   // simple carry handling
            // If exponent overflows after carry → NaN
            যদি ((m >> 3) > 14) {
                ফেরত static_cast<uint8_t>((sign << 7) | 0x7F);
            }
        }
    }

    uint8_t ফলাফল = static_cast<uint8_t>((sign << 7) | (m & 0x7F));
    ফেরত ফলাফল;
}

অন্তঃরেখ দশমিক e4m3_to_float(uint8_t x)
{
    যদি (x == 0) ফেরত 0.0f;

    uint8_t sign = (x >> 7) & 0x1u;
    uint8_t exp  = (x >> 3) & 0xFu;
    uint8_t mant = x & 0x7u;

    // NaN (NVIDIA uses 0x7F / 0xFF as NaN)
    যদি (exp == 0xF && mant != 0) {
        ফেরত std::numeric_limits<দশমিক>::quiet_NaN();
    }
    যদি (exp == 0xF) {                     // 0x7F or 0xFF treated as NaN
        ফেরত std::numeric_limits<দশমিক>::quiet_NaN();
    }

    দশমিক val;

    যদি (exp == 0) {
        // Subnormal
        val = mant * (1.0f / 8.0f) * sycl::pow(2.0f, -6.0f);
    } অন্যথা {
        // Normal: implicit leading 1 + bias 7
        val = (1.0f + mant / 8.0f) * sycl::pow(2.0f, static_cast<দশমিক>(exp) - 7.0f);
    }

    ফেরত sign ? -val : val;
}

// The actual type definition
কাঠামো __nv_fp8_e4m3 {
    uint8_t raw;

    __nv_fp8_e4m3() = পূর্বনির্ধারিত;

    explicit __nv_fp8_e4m3(দশমিক f) : raw(float_to_e4m3(f)) {}
    explicit __nv_fp8_e4m3(sycl::half h) : raw(float_to_e4m3(static_cast<দশমিক>(h))) {}

    operator দশমিক() ধ্রুবক { ফেরত e4m3_to_float(raw); }
    operator sycl::half() ধ্রুবক { ফেরত static_cast<sycl::half>(static_cast<দশমিক>(*এই)); }

    // Allow direct access for vector loads/stores
    operator uint8_t&() { ফেরত raw; }
    operator uint8_t() ধ্রুবক { ফেরত raw; }
};

using __nv_fp8x2_e4m3 = sycl::vec<__nv_fp8_e4m3, 2>;
using __nv_fp8x4_e4m3 = sycl::vec<__nv_fp8_e4m3, 4>;

