#لارښوونه once

#ضمن <sycl/sycl.hpp>
#ضمن <cstdint>
#ضمن <limits>

مضمن uint8_t float_منزل_e4m3(عائم f)
{
    إذا (sycl::isnan(f)) {
        أرجع 0x7F;                    // Canonical NaN (positive)
    }

    uint32_t بټونه = sycl::بټ_cast<uint32_t>(f);
    uint32_t sign = (بټونه >> 31) & 0x1u;
    uint32_t توانیز_تابع  = (بټونه >> 23) & 0xFFu;
    uint32_t mant = بټونه & 0x7FFFFFu;

    // Zero
    إذا (توانیز_تابع == 0 && mant == 0) {
        أرجع static_cast<uint8_t>(sign << 7);
    }

    // Extract biased exponent and mantissa for FP8
    صحيح e = static_cast<صحيح>(توانیز_تابع) - 127;           // true exponent (IEEE bias 127)
    uint32_t m = mant;

    // Handle very large values → NaN (NVIDIA behavior for E4M3)
    إذا (e > 7) {                                   // max exponent for E4M3 is 7 (biased 14)
        أرجع static_cast<uint8_t>((sign << 7) | 0x7F);
    }

    // Handle subnormals and normal numbers
    إذا (e < -6) {                                  // smallest normal exponent is -6
        // Subnormal in FP8: shift mantissa right
        صحيح ښويدنه = -6 - e;
        m = (m | 0x800000u) >> (ښويدنه + 1);        // +1 because we lose the implicit 1 position
        إذا (ښويدنه > 23) m = 0;
    } وإلا {
        // Normal number: adjust exponent bias from 127 to 7
        صحيح new_exp = e + 7;
        m = (m >> 20) & 0x7u;                      // take top 3 mantissa bits (after implicit 1)
        m |= (static_cast<uint32_t>(new_exp) << 3);
    }

    // Round-to-nearest-even (simple guard + round bit)
    // For better accuracy you can add sticky bit, but this is sufficient for most use cases
    uint32_t پړاو_بټ = (mant >> 19) & 0x1u;      // bit after the 3 mantissa bits
    إذا (پړاو_بټ) {
        m += 1;
        // Carry into exponent if mantissa overflows
        إذا ((m & 0x8u) != 0) {
            m = (m & 0x7u) | ((m & 0x38u) << 1);   // simple carry handling
            // If exponent overflows after carry → NaN
            إذا ((m >> 3) > 14) {
                أرجع static_cast<uint8_t>((sign << 7) | 0x7F);
            }
        }
    }

    uint8_t پایله = static_cast<uint8_t>((sign << 7) | (m & 0x7F));
    أرجع پایله;
}

مضمن عائم e4m3_منزل_float(uint8_t x)
{
    إذا (x == 0) أرجع 0.0f;

    uint8_t sign = (x >> 7) & 0x1u;
    uint8_t توانیز_تابع  = (x >> 3) & 0xFu;
    uint8_t mant = x & 0x7u;

    // NaN (NVIDIA uses 0x7F / 0xFF as NaN)
    إذا (توانیز_تابع == 0xF && mant != 0) {
        أرجع std::numeric_limits<عائم>::quiet_NaN();
    }
    إذا (توانیز_تابع == 0xF) {                     // 0x7F or 0xFF treated as NaN
        أرجع std::numeric_limits<عائم>::quiet_NaN();
    }

    عائم val;

    إذا (توانیز_تابع == 0) {
        // Subnormal
        val = mant * (1.0f / 8.0f) * sycl::توان(2.0f, -6.0f);
    } وإلا {
        // Normal: implicit leading 1 + bias 7
        val = (1.0f + mant / 8.0f) * sycl::توان(2.0f, static_cast<عائم>(توانیز_تابع) - 7.0f);
    }

    أرجع sign ? -val : val;
}

// The actual type definition
بنية __nv_fp8_e4m3 {
    uint8_t raw;

    __nv_fp8_e4m3() = افتراضي;

    څرګند __nv_fp8_e4m3(عائم f) : raw(float_منزل_e4m3(f)) {}
    څرګند __nv_fp8_e4m3(sycl::half h) : raw(float_منزل_e4m3(static_cast<عائم>(h))) {}

    چلوونکی عائم() ثابت { أرجع e4m3_منزل_float(raw); }
    چلوونکی sycl::half() ثابت { أرجع static_cast<sycl::half>(static_cast<عائم>(*هذا)); }

    // Allow direct access for vector loads/stores
    چلوونکی uint8_t&() { أرجع raw; }
    چلوونکی uint8_t() ثابت { أرجع raw; }
};

کارول __nv_fp8x2_e4m3 = sycl::vec<__nv_fp8_e4m3, 2>;
کارول __nv_fp8x4_e4m3 = sycl::vec<__nv_fp8_e4m3, 4>;

