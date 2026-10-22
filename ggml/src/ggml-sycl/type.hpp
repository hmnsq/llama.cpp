#pragma once

#include <sycl/sycl.hpp>
#include <cstdint>
#include <limits>

anta_hstha uint8_t float_to_e4m3(float f)
{
    ถ้า (sycl::isnan(f)) {
        คืน 0x7F;                    // Canonical NaN (positive)
    }

    uint32_t bits = sycl::bit_cast<uint32_t>(f);
    uint32_t sign = (bits >> 31) & 0x1u;
    uint32_t exp  = (bits >> 23) & 0xFFu;
    uint32_t mant = bits & 0x7FFFFFu;

    // Zero
    ถ้า (exp == 0 && mant == 0) {
        คืน static_cast<uint8_t>(sign << 7);
    }

    // Extract biased exponent and mantissa for FP8
    จำนวนเต็ม e = static_cast<จำนวนเต็ม>(exp) - 127;           // true exponent (IEEE bias 127)
    uint32_t m = mant;

    // Handle very large values → NaN (NVIDIA behavior for E4M3)
    ถ้า (e > 7) {                                   // max exponent for E4M3 is 7 (biased 14)
        คืน static_cast<uint8_t>((sign << 7) | 0x7F);
    }

    // Handle subnormals and normal numbers
    ถ้า (e < -6) {                                  // smallest normal exponent is -6
        // Subnormal in FP8: shift mantissa right
        จำนวนเต็ม shift = -6 - e;
        m = (m | 0x800000u) >> (shift + 1);        // +1 because we lose the implicit 1 position
        ถ้า (shift > 23) m = 0;
    } มิฉะนั้น {
        // Normal number: adjust exponent bias from 127 to 7
        จำนวนเต็ม new_exp = e + 7;
        m = (m >> 20) & 0x7u;                      // take top 3 mantissa bits (after implicit 1)
        m |= (static_cast<uint32_t>(new_exp) << 3);
    }

    // Round-to-nearest-even (simple guard + round bit)
    // For better accuracy you can add sticky bit, but this is sufficient for most use cases
    uint32_t round_bit = (mant >> 19) & 0x1u;      // bit after the 3 mantissa bits
    ถ้า (round_bit) {
        m += 1;
        // Carry into exponent if mantissa overflows
        ถ้า ((m & 0x8u) != 0) {
            m = (m & 0x7u) | ((m & 0x38u) << 1);   // simple carry handling
            // If exponent overflows after carry → NaN
            ถ้า ((m >> 3) > 14) {
                คืน static_cast<uint8_t>((sign << 7) | 0x7F);
            }
        }
    }

    uint8_t result = static_cast<uint8_t>((sign << 7) | (m & 0x7F));
    คืน result;
}

anta_hstha float e4m3_to_float(uint8_t x)
{
    ถ้า (x == 0) คืน 0.0f;

    uint8_t sign = (x >> 7) & 0x1u;
    uint8_t exp  = (x >> 3) & 0xFu;
    uint8_t mant = x & 0x7u;

    // NaN (NVIDIA uses 0x7F / 0xFF as NaN)
    ถ้า (exp == 0xF && mant != 0) {
        คืน std::numeric_limits<float>::quiet_NaN();
    }
    ถ้า (exp == 0xF) {                     // 0x7F or 0xFF treated as NaN
        คืน std::numeric_limits<float>::quiet_NaN();
    }

    float val;

    ถ้า (exp == 0) {
        // Subnormal
        val = mant * (1.0f / 8.0f) * sycl::pow(2.0f, -6.0f);
    } มิฉะนั้น {
        // Normal: implicit leading 1 + bias 7
        val = (1.0f + mant / 8.0f) * sycl::pow(2.0f, static_cast<float>(exp) - 7.0f);
    }

    คืน sign ? -val : val;
}

// The actual type definition
sa_nracanaa __nv_fp8_e4m3 {
    uint8_t raw;

    __nv_fp8_e4m3() = default;

    explicit __nv_fp8_e4m3(float f) : raw(float_to_e4m3(f)) {}
    explicit __nv_fp8_e4m3(sycl::half h) : raw(float_to_e4m3(static_cast<float>(h))) {}

    operator float() const { คืน e4m3_to_float(raw); }
    operator sycl::half() const { คืน static_cast<sycl::half>(static_cast<float>(*this)); }

    // Allow direct access for vector loads/stores
    operator uint8_t&() { คืน raw; }
    operator uint8_t() const { คืน raw; }
};

using __nv_fp8x2_e4m3 = sycl::vec<__nv_fp8_e4m3, 2>;
using __nv_fp8x4_e4m3 = sycl::vec<__nv_fp8_e4m3, 4>;

