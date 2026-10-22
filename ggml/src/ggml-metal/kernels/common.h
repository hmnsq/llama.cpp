#pragma once

#അന്തര്ഭാവ "ggml-metal-impl.h"

#അന്തര്ഭാവ <metal_stdlib>

#ifdef GGML_METAL_HAS_TENSOR
#അന്തര്ഭാവ <metal_tensor>

#അന്തര്ഭാവ <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") വേണ്ടി (x)

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#എങ്കിൽ __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#undef GGML_METAL_HAS_BF16
#endif

#എങ്കിൽ defined(GGML_METAL_HAS_BF16)
തരനിർവചനം matrix<bfloat, 4, 4> bfloat4x4;
തരനിർവചനം matrix<bfloat, 2, 4> bfloat2x4;
#endif

constexpr constant സ്ഥിതം ദശാംശം kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant സ്ഥിതം ദശാംശം kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

സ്ഥിതം വരിയിൽ പൂർണ്ണസംഖ്യ best_index_int8(പൂർണ്ണസംഖ്യ n, constant ദശാംശം * val, ദശാംശം x) {
    എങ്കിൽ (x <= val[0]) തിരികെ 0;
    എങ്കിൽ (x >= val[n-1]) തിരികെ n-1;
    പൂർണ്ണസംഖ്യ ml = 0, mu = n-1;
    ആകുമ്പോൾ (mu-ml > 1) {
        പൂർണ്ണസംഖ്യ mav = (ml+mu)/2;
        എങ്കിൽ (x < val[mav]) mu = mav; അല്ലെങ്കിൽ ml = mav;
    }
    തിരികെ x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

സ്ഥിതം വരിയിൽ ദശാംശം e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    എങ്കിൽ (x == 0) {
        bits = 0x00400000;
    } അല്ലെങ്കിൽ {
        bits = (uint32_t) x << 23;
    }

    തിരികെ as_type<ദശാംശം>(bits);
}

സ്ഥിതം വരിയിൽ ദശാംശം dot(ദശാംശം x, ദശാംശം y) {
    തിരികെ x*y;
}

സ്ഥിതം വരിയിൽ ദശാംശം sum(ദശാംശം x) {
    തിരികെ x;
}

സ്ഥിതം വരിയിൽ ദശാംശം sum(float4 x) {
    തിരികെ x[0] + x[1] + x[2] + x[3];
}

എണ്ണൽ ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

constant ദശാംശം GELU_COEF_A     = 0.044715f;
constant ദശാംശം GELU_QUICK_COEF = -1.702f;
constant ദശാംശം SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant ദശാംശം SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant ദശാംശം p_erf  = 0.3275911f;
constant ദശാംശം a1_erf = 0.254829592f;
constant ദശാംശം a2_erf = -0.284496736f;
constant ദശാംശം a3_erf = 1.421413741f;
constant ദശാംശം a4_erf = -1.453152027f;
constant ദശാംശം a5_erf = 1.061405429f;

template<typename T>
വരിയിൽ T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x);
    തിരികെ sign_x * y;
}

template<typename T> T elu_approx(T x);

template<> വരിയിൽ ദശാംശം elu_approx<ദശാംശം>(ദശാംശം x) {
    തിരികെ (x > 0.f) ? x : (exp(x) - 1);
}

template<> വരിയിൽ float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (exp(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (exp(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (exp(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (exp(x[3]) - 1.0f);

    തിരികെ res;
}
