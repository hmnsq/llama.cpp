#رهنمود once

#ضمن "ggml-metal-impl.h"

#ضمن <metal_stdlib>

#اگر_تعریف_شده GGML_METAL_HAS_TENSOR
#ضمن <metal_tensor>

#ضمن <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#پایان_اگر

using namespace metal;

#تعریف_کن MAX(x, y) ((x) > (y) ? (x) : (y))
#تعریف_کن MIN(x, y) ((x) < (y) ? (x) : (y))
#تعریف_کن جابجا(x, y) { خودکار tmp = (x); (x) = (y); (y) = tmp; }

#تعریف_کن PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#تعریف_کن FOR_UNROLL(x) _Pragma("clang loop unroll(full)") لكل (x)

#تعریف_کن N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#إذا __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#لغو_تعریف GGML_METAL_HAS_BF16
#پایان_اگر

#إذا defined(GGML_METAL_HAS_BF16)
تعريفنوع ماتریس<bfloat, 4, 4> bfloat4x4;
تعريفنوع ماتریس<bfloat, 2, 4> bfloat2x4;
#پایان_اگر

constexpr constant ساكن عائم kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant ساكن عائم kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

ساكن مضمن صحيح بهترین_شاخص_int8(صحيح n, constant عائم * val, عائم x) {
    إذا (x <= val[0]) أرجع 0;
    إذا (x >= val[n-1]) أرجع n-1;
    صحيح ml = 0, mu = n-1;
    طالما (mu-ml > 1) {
        صحيح mav = (ml+mu)/2;
        إذا (x < val[mav]) mu = mav; وإلا ml = mav;
    }
    أرجع x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

ساكن مضمن عائم e8m0_مقصد_fp32(uint8_t x) {
    uint32_t بیتها;

    إذا (x == 0) {
        بیتها = 0x00400000;
    } وإلا {
        بیتها = (uint32_t) x << 23;
    }

    أرجع as_type<عائم>(بیتها);
}

ساكن مضمن عائم dot(عائم x, عائم y) {
    أرجع x*y;
}

ساكن مضمن عائم جمع(عائم x) {
    أرجع x;
}

ساكن مضمن عائم جمع(float4 x) {
    أرجع x[0] + x[1] + x[2] + x[3];
}

تعداد ggml_مرتب_سازی_ترتیب {
    GGML_مرتب_سازی_ترتیب_ASC,
    GGML_مرتب_سازی_ترتیب_DESC,
};

constant عائم GELU_COEF_A     = 0.044715f;
constant عائم GELU_سریع_COEF = -1.702f;
constant عائم SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant عائم SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant عائم p_erf  = 0.3275911f;
constant عائم a1_erf = 0.254829592f;
constant عائم a2_erf = -0.284496736f;
constant عائم a3_erf = 1.421413741f;
constant عائم a4_erf = -1.453152027f;
constant عائم a5_erf = 1.061405429f;

قالب<typename T>
مضمن T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * نمایی_تابع(-x * x);
    أرجع sign_x * y;
}

قالب<typename T> T elu_approx(T x);

قالب<> مضمن عائم elu_approx<عائم>(عائم x) {
    أرجع (x > 0.f) ? x : (نمایی_تابع(x) - 1);
}

قالب<> مضمن float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (نمایی_تابع(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (نمایی_تابع(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (نمایی_تابع(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (نمایی_تابع(x[3]) - 1.0f);

    أرجع res;
}
