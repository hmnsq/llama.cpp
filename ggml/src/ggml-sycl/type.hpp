#ہدایت once

#شامل <sycl/sycl.hpp>
#شامل <cstdint>
#شامل <limits>

اندرونی uint8_t float_مقصد_e4m3(اعشاری f)
{
    اگر (sycl::isnan(f)) {
        واپس 0x7F;                    // Canonical NaN (positive)
    }

    uint32_t بٹ = sycl::ثنائی_ہندسہ_cast<uint32_t>(f);
    uint32_t sign = (بٹ >> 31) & 0x1u;
    uint32_t قوت_نما  = (بٹ >> 23) & 0xFFu;
    uint32_t mant = بٹ & 0x7FFFFFu;

    // Zero
    اگر (قوت_نما == 0 && mant == 0) {
        واپس static_cast<uint8_t>(sign << 7);
    }

    // Extract biased exponent and mantissa for FP8
    صحیح e = static_cast<صحیح>(قوت_نما) - 127;           // true exponent (IEEE bias 127)
    uint32_t m = mant;

    // Handle very large values → NaN (NVIDIA behavior for E4M3)
    اگر (e > 7) {                                   // max exponent for E4M3 is 7 (biased 14)
        واپس static_cast<uint8_t>((sign << 7) | 0x7F);
    }

    // Handle subnormals and normal numbers
    اگر (e < -6) {                                  // smallest normal exponent is -6
        // Subnormal in FP8: shift mantissa right
        صحیح سرکاؤ = -6 - e;
        m = (m | 0x800000u) >> (سرکاؤ + 1);        // +1 because we lose the implicit 1 position
        اگر (سرکاؤ > 23) m = 0;
    } ورنہ {
        // Normal number: adjust exponent bias from 127 to 7
        صحیح new_exp = e + 7;
        m = (m >> 20) & 0x7u;                      // take top 3 mantissa bits (after implicit 1)
        m |= (static_cast<uint32_t>(new_exp) << 3);
    }

    // Round-to-nearest-even (simple guard + round bit)
    // For better accuracy you can add sticky bit, but this is sufficient for most use cases
    uint32_t باری_ثنائی_ہندسہ = (mant >> 19) & 0x1u;      // bit after the 3 mantissa bits
    اگر (باری_ثنائی_ہندسہ) {
        m += 1;
        // Carry into exponent if mantissa overflows
        اگر ((m & 0x8u) != 0) {
            m = (m & 0x7u) | ((m & 0x38u) << 1);   // simple carry handling
            // If exponent overflows after carry → NaN
            اگر ((m >> 3) > 14) {
                واپس static_cast<uint8_t>((sign << 7) | 0x7F);
            }
        }
    }

    uint8_t نتیجہ = static_cast<uint8_t>((sign << 7) | (m & 0x7F));
    واپس نتیجہ;
}

اندرونی اعشاری e4m3_مقصد_float(uint8_t x)
{
    اگر (x == 0) واپس 0.0f;

    uint8_t sign = (x >> 7) & 0x1u;
    uint8_t قوت_نما  = (x >> 3) & 0xFu;
    uint8_t mant = x & 0x7u;

    // NaN (NVIDIA uses 0x7F / 0xFF as NaN)
    اگر (قوت_نما == 0xF && mant != 0) {
        واپس std::numeric_limits<اعشاری>::quiet_NaN();
    }
    اگر (قوت_نما == 0xF) {                     // 0x7F or 0xFF treated as NaN
        واپس std::numeric_limits<اعشاری>::quiet_NaN();
    }

    اعشاری val;

    اگر (قوت_نما == 0) {
        // Subnormal
        val = mant * (1.0f / 8.0f) * sycl::قوت(2.0f, -6.0f);
    } ورنہ {
        // Normal: implicit leading 1 + bias 7
        val = (1.0f + mant / 8.0f) * sycl::قوت(2.0f, static_cast<اعشاری>(قوت_نما) - 7.0f);
    }

    واپس sign ? -val : val;
}

// The actual type definition
ساخت __nv_fp8_e4m3 {
    uint8_t raw;

    __nv_fp8_e4m3() = طےشدہ;

    صریح __nv_fp8_e4m3(اعشاری f) : raw(float_مقصد_e4m3(f)) {}
    صریح __nv_fp8_e4m3(sycl::half h) : raw(float_مقصد_e4m3(static_cast<اعشاری>(h))) {}

    عامل اعشاری() ثابت { واپس e4m3_مقصد_float(raw); }
    عامل sycl::half() ثابت { واپس static_cast<sycl::half>(static_cast<اعشاری>(*یہ)); }

    // Allow direct access for vector loads/stores
    عامل uint8_t&() { واپس raw; }
    عامل uint8_t() ثابت { واپس raw; }
};

استعمال __nv_fp8x2_e4m3 = sycl::vec<__nv_fp8_e4m3, 2>;
استعمال __nv_fp8x4_e4m3 = sycl::vec<__nv_fp8_e4m3, 4>;

