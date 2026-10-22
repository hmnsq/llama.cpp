#pragma once

#অন্তর্ভাব "ggml-metal-impl.h"

#অন্তর্ভাব <metal_stdlib>

#ifdef GGML_METAL_HAS_TENSOR
#অন্তর্ভাব <metal_tensor>

#অন্তর্ভাব <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") জন্য (x)

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#যদি __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#undef GGML_METAL_HAS_BF16
#endif

#যদি defined(GGML_METAL_HAS_BF16)
প্রকারলেখ matrix<bfloat, 4, 4> bfloat4x4;
প্রকারলেখ matrix<bfloat, 2, 4> bfloat2x4;
#endif

constexpr constant স্থির দশমিক kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant স্থির দশমিক kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

স্থির অন্তঃরেখ পূর্ণসংখ্যা best_সূচক_int8(পূর্ণসংখ্যা n, constant দশমিক * val, দশমিক x) {
    যদি (x <= val[0]) ফেরত 0;
    যদি (x >= val[n-1]) ফেরত n-1;
    পূর্ণসংখ্যা ml = 0, mu = n-1;
    যখন (mu-ml > 1) {
        পূর্ণসংখ্যা mav = (ml+mu)/2;
        যদি (x < val[mav]) mu = mav; অন্যথা ml = mav;
    }
    ফেরত x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

স্থির অন্তঃরেখ দশমিক e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    যদি (x == 0) {
        bits = 0x00400000;
    } অন্যথা {
        bits = (uint32_t) x << 23;
    }

    ফেরত as_type<দশমিক>(bits);
}

স্থির অন্তঃরেখ দশমিক dot(দশমিক x, দশমিক y) {
    ফেরত x*y;
}

স্থির অন্তঃরেখ দশমিক যোগ_2(দশমিক x) {
    ফেরত x;
}

স্থির অন্তঃরেখ দশমিক যোগ_2(float4 x) {
    ফেরত x[0] + x[1] + x[2] + x[3];
}

গণনা ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

constant দশমিক GELU_COEF_A     = 0.044715f;
constant দশমিক GELU_QUICK_COEF = -1.702f;
constant দশমিক SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant দশমিক SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant দশমিক p_erf  = 0.3275911f;
constant দশমিক a1_erf = 0.254829592f;
constant দশমিক a2_erf = -0.284496736f;
constant দশমিক a3_erf = 1.421413741f;
constant দশমিক a4_erf = -1.453152027f;
constant দশমিক a5_erf = 1.061405429f;

template<typename T>
অন্তঃরেখ T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x);
    ফেরত sign_x * y;
}

template<typename T> T elu_approx(T x);

template<> অন্তঃরেখ দশমিক elu_approx<দশমিক>(দশমিক x) {
    ফেরত (x > 0.f) ? x : (exp(x) - 1);
}

template<> অন্তঃরেখ float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (exp(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (exp(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (exp(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (exp(x[3]) - 1.0f);

    ফেরত res;
}
