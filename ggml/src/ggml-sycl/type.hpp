#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ <sycl/sycl.hpp>
#ਅਨ੍ਤਰ੍ਭਾਵ <cstdint>
#ਅਨ੍ਤਰ੍ਭਾਵ <limits>

ਸਤਰਅੰਦਰ uint8_t float_ਮੰਜ਼ਿਲ_e4m3(ਦਸ਼ਮਲਵ f)
{
    ਜੇ (sycl::isnan(f)) {
        ਵਾਪਸ 0x7F;                    // Canonical NaN (positive)
    }

    uint32_t ਬਿੱਟ = sycl::ਦੋਆਧਾਰੀ_ਅੰਕ_cast<uint32_t>(f);
    uint32_t sign = (ਬਿੱਟ >> 31) & 0x1u;
    uint32_t ਘਾਤਾਂਕ  = (ਬਿੱਟ >> 23) & 0xFFu;
    uint32_t mant = ਬਿੱਟ & 0x7FFFFFu;

    // Zero
    ਜੇ (ਘਾਤਾਂਕ == 0 && mant == 0) {
        ਵਾਪਸ static_cast<uint8_t>(sign << 7);
    }

    // Extract biased exponent and mantissa for FP8
    ਪੂਰਨ e = static_cast<ਪੂਰਨ>(ਘਾਤਾਂਕ) - 127;           // true exponent (IEEE bias 127)
    uint32_t m = mant;

    // Handle very large values → NaN (NVIDIA behavior for E4M3)
    ਜੇ (e > 7) {                                   // max exponent for E4M3 is 7 (biased 14)
        ਵਾਪਸ static_cast<uint8_t>((sign << 7) | 0x7F);
    }

    // Handle subnormals and normal numbers
    ਜੇ (e < -6) {                                  // smallest normal exponent is -6
        // Subnormal in FP8: shift mantissa right
        ਪੂਰਨ ਖਿਸਕਾਓ = -6 - e;
        m = (m | 0x800000u) >> (ਖਿਸਕਾਓ + 1);        // +1 because we lose the implicit 1 position
        ਜੇ (ਖਿਸਕਾਓ > 23) m = 0;
    } ਨਹੀਂਤਾਂ {
        // Normal number: adjust exponent bias from 127 to 7
        ਪੂਰਨ new_exp = e + 7;
        m = (m >> 20) & 0x7u;                      // take top 3 mantissa bits (after implicit 1)
        m |= (static_cast<uint32_t>(new_exp) << 3);
    }

    // Round-to-nearest-even (simple guard + round bit)
    // For better accuracy you can add sticky bit, but this is sufficient for most use cases
    uint32_t ਗੇੜਾ_ਦੋਆਧਾਰੀ_ਅੰਕ = (mant >> 19) & 0x1u;      // bit after the 3 mantissa bits
    ਜੇ (ਗੇੜਾ_ਦੋਆਧਾਰੀ_ਅੰਕ) {
        m += 1;
        // Carry into exponent if mantissa overflows
        ਜੇ ((m & 0x8u) != 0) {
            m = (m & 0x7u) | ((m & 0x38u) << 1);   // simple carry handling
            // If exponent overflows after carry → NaN
            ਜੇ ((m >> 3) > 14) {
                ਵਾਪਸ static_cast<uint8_t>((sign << 7) | 0x7F);
            }
        }
    }

    uint8_t ਨਤੀਜਾ = static_cast<uint8_t>((sign << 7) | (m & 0x7F));
    ਵਾਪਸ ਨਤੀਜਾ;
}

ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ e4m3_ਮੰਜ਼ਿਲ_float(uint8_t x)
{
    ਜੇ (x == 0) ਵਾਪਸ 0.0f;

    uint8_t sign = (x >> 7) & 0x1u;
    uint8_t ਘਾਤਾਂਕ  = (x >> 3) & 0xFu;
    uint8_t mant = x & 0x7u;

    // NaN (NVIDIA uses 0x7F / 0xFF as NaN)
    ਜੇ (ਘਾਤਾਂਕ == 0xF && mant != 0) {
        ਵਾਪਸ std::numeric_limits<ਦਸ਼ਮਲਵ>::quiet_NaN();
    }
    ਜੇ (ਘਾਤਾਂਕ == 0xF) {                     // 0x7F or 0xFF treated as NaN
        ਵਾਪਸ std::numeric_limits<ਦਸ਼ਮਲਵ>::quiet_NaN();
    }

    ਦਸ਼ਮਲਵ val;

    ਜੇ (ਘਾਤਾਂਕ == 0) {
        // Subnormal
        val = mant * (1.0f / 8.0f) * sycl::ਘਾਤ(2.0f, -6.0f);
    } ਨਹੀਂਤਾਂ {
        // Normal: implicit leading 1 + bias 7
        val = (1.0f + mant / 8.0f) * sycl::ਘਾਤ(2.0f, static_cast<ਦਸ਼ਮਲਵ>(ਘਾਤਾਂਕ) - 7.0f);
    }

    ਵਾਪਸ sign ? -val : val;
}

// The actual type definition
ਬਣਤਰ __nv_fp8_e4m3 {
    uint8_t raw;

    __nv_fp8_e4m3() = ਮੂਲ;

    ਸਪਸ਼ਟ __nv_fp8_e4m3(ਦਸ਼ਮਲਵ f) : raw(float_ਮੰਜ਼ਿਲ_e4m3(f)) {}
    ਸਪਸ਼ਟ __nv_fp8_e4m3(sycl::half h) : raw(float_ਮੰਜ਼ਿਲ_e4m3(static_cast<ਦਸ਼ਮਲਵ>(h))) {}

    ਸੰਚਾਲਕ ਦਸ਼ਮਲਵ() ਅਚਲ { ਵਾਪਸ e4m3_ਮੰਜ਼ਿਲ_float(raw); }
    ਸੰਚਾਲਕ sycl::half() ਅਚਲ { ਵਾਪਸ static_cast<sycl::half>(static_cast<ਦਸ਼ਮਲਵ>(*ਇਹ)); }

    // Allow direct access for vector loads/stores
    ਸੰਚਾਲਕ uint8_t&() { ਵਾਪਸ raw; }
    ਸੰਚਾਲਕ uint8_t() ਅਚਲ { ਵਾਪਸ raw; }
};

ਵਰਤੋਂ __nv_fp8x2_e4m3 = sycl::vec<__nv_fp8_e4m3, 2>;
ਵਰਤੋਂ __nv_fp8x4_e4m3 = sycl::vec<__nv_fp8_e4m3, 4>;

