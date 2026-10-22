#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ <sycl/sycl.hpp>
#ಅನ್ತರ್ಭಾವ <cstdint>
#ಅನ್ತರ್ಭಾವ <limits>

ಸಾಲೊಳಗೆ uint8_t float_ಗಮ್ಯ_e4m3(ದಶಮಾಂಶ f)
{
    ಆದರೆ (sycl::isnan(f)) {
        ಹಿಂತಿರುಗಿ 0x7F;                    // Canonical NaN (positive)
    }

    uint32_t ಬಿಟ್ಟುಗಳು = sycl::ಬಿಟ್_cast<uint32_t>(f);
    uint32_t sign = (ಬಿಟ್ಟುಗಳು >> 31) & 0x1u;
    uint32_t ಘಾತಾಂಕ  = (ಬಿಟ್ಟುಗಳು >> 23) & 0xFFu;
    uint32_t mant = ಬಿಟ್ಟುಗಳು & 0x7FFFFFu;

    // Zero
    ಆದರೆ (ಘಾತಾಂಕ == 0 && mant == 0) {
        ಹಿಂತಿರುಗಿ static_cast<uint8_t>(sign << 7);
    }

    // Extract biased exponent and mantissa for FP8
    ಪೂರ್ಣಾಂಕ e = static_cast<ಪೂರ್ಣಾಂಕ>(ಘಾತಾಂಕ) - 127;           // true exponent (IEEE bias 127)
    uint32_t m = mant;

    // Handle very large values → NaN (NVIDIA behavior for E4M3)
    ಆದರೆ (e > 7) {                                   // max exponent for E4M3 is 7 (biased 14)
        ಹಿಂತಿರುಗಿ static_cast<uint8_t>((sign << 7) | 0x7F);
    }

    // Handle subnormals and normal numbers
    ಆದರೆ (e < -6) {                                  // smallest normal exponent is -6
        // Subnormal in FP8: shift mantissa right
        ಪೂರ್ಣಾಂಕ ಸರಿಸು = -6 - e;
        m = (m | 0x800000u) >> (ಸರಿಸು + 1);        // +1 because we lose the implicit 1 position
        ಆದರೆ (ಸರಿಸು > 23) m = 0;
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Normal number: adjust exponent bias from 127 to 7
        ಪೂರ್ಣಾಂಕ new_exp = e + 7;
        m = (m >> 20) & 0x7u;                      // take top 3 mantissa bits (after implicit 1)
        m |= (static_cast<uint32_t>(new_exp) << 3);
    }

    // Round-to-nearest-even (simple guard + round bit)
    // For better accuracy you can add sticky bit, but this is sufficient for most use cases
    uint32_t ಸುತ್ತು_ಬಿಟ್ = (mant >> 19) & 0x1u;      // bit after the 3 mantissa bits
    ಆದರೆ (ಸುತ್ತು_ಬಿಟ್) {
        m += 1;
        // Carry into exponent if mantissa overflows
        ಆದರೆ ((m & 0x8u) != 0) {
            m = (m & 0x7u) | ((m & 0x38u) << 1);   // simple carry handling
            // If exponent overflows after carry → NaN
            ಆದರೆ ((m >> 3) > 14) {
                ಹಿಂತಿರುಗಿ static_cast<uint8_t>((sign << 7) | 0x7F);
            }
        }
    }

    uint8_t ಫಲಿತಾಂಶ = static_cast<uint8_t>((sign << 7) | (m & 0x7F));
    ಹಿಂತಿರುಗಿ ಫಲಿತಾಂಶ;
}

ಸಾಲೊಳಗೆ ದಶಮಾಂಶ e4m3_ಗಮ್ಯ_float(uint8_t x)
{
    ಆದರೆ (x == 0) ಹಿಂತಿರುಗಿ 0.0f;

    uint8_t sign = (x >> 7) & 0x1u;
    uint8_t ಘಾತಾಂಕ  = (x >> 3) & 0xFu;
    uint8_t mant = x & 0x7u;

    // NaN (NVIDIA uses 0x7F / 0xFF as NaN)
    ಆದರೆ (ಘಾತಾಂಕ == 0xF && mant != 0) {
        ಹಿಂತಿರುಗಿ std::numeric_limits<ದಶಮಾಂಶ>::quiet_NaN();
    }
    ಆದರೆ (ಘಾತಾಂಕ == 0xF) {                     // 0x7F or 0xFF treated as NaN
        ಹಿಂತಿರುಗಿ std::numeric_limits<ದಶಮಾಂಶ>::quiet_NaN();
    }

    ದಶಮಾಂಶ val;

    ಆದರೆ (ಘಾತಾಂಕ == 0) {
        // Subnormal
        val = mant * (1.0f / 8.0f) * sycl::ಘಾತ(2.0f, -6.0f);
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Normal: implicit leading 1 + bias 7
        val = (1.0f + mant / 8.0f) * sycl::ಘಾತ(2.0f, static_cast<ದಶಮಾಂಶ>(ಘಾತಾಂಕ) - 7.0f);
    }

    ಹಿಂತಿರುಗಿ sign ? -val : val;
}

// The actual type definition
ರಚನೆ __nv_fp8_e4m3 {
    uint8_t raw;

    __nv_fp8_e4m3() = ಪೂರ್ವನಿಯೋಜಿತ;

    ಸ್ಪಷ್ಟ __nv_fp8_e4m3(ದಶಮಾಂಶ f) : raw(float_ಗಮ್ಯ_e4m3(f)) {}
    ಸ್ಪಷ್ಟ __nv_fp8_e4m3(sycl::half h) : raw(float_ಗಮ್ಯ_e4m3(static_cast<ದಶಮಾಂಶ>(h))) {}

    ಕ್ರಿಯಾಕಾರಕ ದಶಮಾಂಶ() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ e4m3_ಗಮ್ಯ_float(raw); }
    ಕ್ರಿಯಾಕಾರಕ sycl::half() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ static_cast<sycl::half>(static_cast<ದಶಮಾಂಶ>(*ಇದು)); }

    // Allow direct access for vector loads/stores
    ಕ್ರಿಯಾಕಾರಕ uint8_t&() { ಹಿಂತಿರುಗಿ raw; }
    ಕ್ರಿಯಾಕಾರಕ uint8_t() ಸ್ಥಿರಾಂಕ { ಹಿಂತಿರುಗಿ raw; }
};

ಬಳಸಿ __nv_fp8x2_e4m3 = sycl::vec<__nv_fp8_e4m3, 2>;
ಬಳಸಿ __nv_fp8x4_e4m3 = sycl::vec<__nv_fp8_e4m3, 4>;

