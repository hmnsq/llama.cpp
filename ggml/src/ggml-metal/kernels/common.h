#pragma once

#ಅನ್ತರ್ಭಾವ "ggml-metal-impl.h"

#ಅನ್ತರ್ಭಾವ <metal_stdlib>

#ifdef GGML_METAL_HAS_TENSOR
#ಅನ್ತರ್ಭಾವ <metal_tensor>

#ಅನ್ತರ್ಭಾವ <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") ಪ್ರತಿ (x)

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#ಯದಿ __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#undef GGML_METAL_HAS_BF16
#endif

#ಯದಿ defined(GGML_METAL_HAS_BF16)
ಪ್ರಕಾರನಿರ್ದೇಶ matrix<bfloat, 4, 4> bfloat4x4;
ಪ್ರಕಾರನಿರ್ದೇಶ matrix<bfloat, 2, 4> bfloat2x4;
#endif

constexpr constant ಸ್ಥಿರ ಭಿನ್ನ kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant ಸ್ಥಿರ ಭಿನ್ನ kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಪೂರ್ಣಾಙ್ಕ best_index_int8(ಪೂರ್ಣಾಙ್ಕ n, constant ಭಿನ್ನ * val, ಭಿನ್ನ x) {
    ಯದಿ (x <= val[0]) ಪ್ರತ್ಯಾಗಮ 0;
    ಯದಿ (x >= val[n-1]) ಪ್ರತ್ಯಾಗಮ n-1;
    ಪೂರ್ಣಾಙ್ಕ ml = 0, mu = n-1;
    ಯಾವತ್ (mu-ml > 1) {
        ಪೂರ್ಣಾಙ್ಕ mav = (ml+mu)/2;
        ಯದಿ (x < val[mav]) mu = mav; ಅನ್ಯಥಾ ml = mav;
    }
    ಪ್ರತ್ಯಾಗಮ x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    ಯದಿ (x == 0) {
        bits = 0x00400000;
    } ಅನ್ಯಥಾ {
        bits = (uint32_t) x << 23;
    }

    ಪ್ರತ್ಯಾಗಮ as_type<ಭಿನ್ನ>(bits);
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ dot(ಭಿನ್ನ x, ಭಿನ್ನ y) {
    ಪ್ರತ್ಯಾಗಮ x*y;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ sum(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ x;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಭಿನ್ನ sum(float4 x) {
    ಪ್ರತ್ಯಾಗಮ x[0] + x[1] + x[2] + x[3];
}

ಗಣನಾ ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

constant ಭಿನ್ನ GELU_COEF_A     = 0.044715f;
constant ಭಿನ್ನ GELU_QUICK_COEF = -1.702f;
constant ಭಿನ್ನ SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant ಭಿನ್ನ SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant ಭಿನ್ನ p_erf  = 0.3275911f;
constant ಭಿನ್ನ a1_erf = 0.254829592f;
constant ಭಿನ್ನ a2_erf = -0.284496736f;
constant ಭಿನ್ನ a3_erf = 1.421413741f;
constant ಭಿನ್ನ a4_erf = -1.453152027f;
constant ಭಿನ್ನ a5_erf = 1.061405429f;

template<typename T>
ಅನ್ತಃಸ್ಥ T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x);
    ಪ್ರತ್ಯಾಗಮ sign_x * y;
}

template<typename T> T elu_approx(T x);

template<> ಅನ್ತಃಸ್ಥ ಭಿನ್ನ elu_approx<ಭಿನ್ನ>(ಭಿನ್ನ x) {
    ಪ್ರತ್ಯಾಗಮ (x > 0.f) ? x : (exp(x) - 1);
}

template<> ಅನ್ತಃಸ್ಥ float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (exp(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (exp(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (exp(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (exp(x[3]) - 1.0f);

    ಪ್ರತ್ಯಾಗಮ res;
}
