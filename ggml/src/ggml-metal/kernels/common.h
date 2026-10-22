#pragma once

#شامل "ggml-metal-impl.h"

#شامل <metal_stdlib>

#ifdef GGML_METAL_HAS_TENSOR
#شامل <metal_tensor>

#شامل <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") برای (x)

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#اگر __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#undef GGML_METAL_HAS_BF16
#endif

#اگر defined(GGML_METAL_HAS_BF16)
تعریفنوع matrix<bfloat, 4, 4> bfloat4x4;
تعریفنوع matrix<bfloat, 2, 4> bfloat2x4;
#endif

constexpr constant ایستا اعشاری kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant ایستا اعشاری kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

ایستا درونخطی صحیح best_index_int8(صحیح n, constant اعشاری * val, اعشاری x) {
    اگر (x <= val[0]) بازگشت 0;
    اگر (x >= val[n-1]) بازگشت n-1;
    صحیح ml = 0, mu = n-1;
    وقتی (mu-ml > 1) {
        صحیح mav = (ml+mu)/2;
        اگر (x < val[mav]) mu = mav; وگرنه ml = mav;
    }
    بازگشت x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

ایستا درونخطی اعشاری e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    اگر (x == 0) {
        bits = 0x00400000;
    } وگرنه {
        bits = (uint32_t) x << 23;
    }

    بازگشت as_type<اعشاری>(bits);
}

ایستا درونخطی اعشاری dot(اعشاری x, اعشاری y) {
    بازگشت x*y;
}

ایستا درونخطی اعشاری sum(اعشاری x) {
    بازگشت x;
}

ایستا درونخطی اعشاری sum(float4 x) {
    بازگشت x[0] + x[1] + x[2] + x[3];
}

شمارش ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

constant اعشاری GELU_COEF_A     = 0.044715f;
constant اعشاری GELU_QUICK_COEF = -1.702f;
constant اعشاری SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant اعشاری SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant اعشاری p_erf  = 0.3275911f;
constant اعشاری a1_erf = 0.254829592f;
constant اعشاری a2_erf = -0.284496736f;
constant اعشاری a3_erf = 1.421413741f;
constant اعشاری a4_erf = -1.453152027f;
constant اعشاری a5_erf = 1.061405429f;

template<typename T>
درونخطی T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x);
    بازگشت sign_x * y;
}

template<typename T> T elu_approx(T x);

template<> درونخطی اعشاری elu_approx<اعشاری>(اعشاری x) {
    بازگشت (x > 0.f) ? x : (exp(x) - 1);
}

template<> درونخطی float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (exp(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (exp(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (exp(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (exp(x[3]) - 1.0f);

    بازگشت res;
}
