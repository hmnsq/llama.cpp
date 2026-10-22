#pragma once

#अन्तर्भाव "ggml-metal-impl.h"

#अन्तर्भाव <metal_stdlib>

#ifdef GGML_METAL_HAS_TENSOR
#अन्तर्भाव <metal_tensor>

#अन्तर्भाव <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { स्वतः tmp = (x); (x) = (y); (y) = tmp; }

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") साठी (x)

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#जर __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#undef GGML_METAL_HAS_BF16
#endif

#जर defined(GGML_METAL_HAS_BF16)
प्रकारव्याख्या matrix<bfloat, 4, 4> bfloat4x4;
प्रकारव्याख्या matrix<bfloat, 2, 4> bfloat2x4;
#endif

constexpr constant स्थिर दशांश kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant स्थिर दशांश kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

स्थिर ओळीत पूर्णांक best_index_int8(पूर्णांक n, constant दशांश * val, दशांश x) {
    जर (x <= val[0]) परत 0;
    जर (x >= val[n-1]) परत n-1;
    पूर्णांक ml = 0, mu = n-1;
    पर्यंत (mu-ml > 1) {
        पूर्णांक mav = (ml+mu)/2;
        जर (x < val[mav]) mu = mav; अन्यथा ml = mav;
    }
    परत x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

स्थिर ओळीत दशांश e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    जर (x == 0) {
        bits = 0x00400000;
    } अन्यथा {
        bits = (uint32_t) x << 23;
    }

    परत as_type<दशांश>(bits);
}

स्थिर ओळीत दशांश dot(दशांश x, दशांश y) {
    परत x*y;
}

स्थिर ओळीत दशांश sum(दशांश x) {
    परत x;
}

स्थिर ओळीत दशांश sum(float4 x) {
    परत x[0] + x[1] + x[2] + x[3];
}

गणना ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

constant दशांश GELU_COEF_A     = 0.044715f;
constant दशांश GELU_QUICK_COEF = -1.702f;
constant दशांश SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant दशांश SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant दशांश p_erf  = 0.3275911f;
constant दशांश a1_erf = 0.254829592f;
constant दशांश a2_erf = -0.284496736f;
constant दशांश a3_erf = 1.421413741f;
constant दशांश a4_erf = -1.453152027f;
constant दशांश a5_erf = 1.061405429f;

ढाँचा<typename T>
ओळीत T erf_approx(T x) {
    T sign_x = sign(x);
    x = भ_असल(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x);
    परत sign_x * y;
}

ढाँचा<typename T> T elu_approx(T x);

ढाँचा<> ओळीत दशांश elu_approx<दशांश>(दशांश x) {
    परत (x > 0.f) ? x : (exp(x) - 1);
}

ढाँचा<> ओळीत float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (exp(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (exp(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (exp(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (exp(x[3]) - 1.0f);

    परत res;
}
