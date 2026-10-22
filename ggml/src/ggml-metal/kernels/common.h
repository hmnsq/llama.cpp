#pragma once

#அந்தர்பாவ "ggml-metal-impl.h"

#அந்தர்பாவ <metal_stdlib>

#ifdef GGML_METAL_HAS_TENSOR
#அந்தர்பாவ <metal_tensor>

#அந்தர்பாவ <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#endif

using namespace metal;

#define MAX(x, y) ((x) > (y) ? (x) : (y))
#define MIN(x, y) ((x) < (y) ? (x) : (y))
#define SWAP(x, y) { auto tmp = (x); (x) = (y); (y) = tmp; }

#define PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#define FOR_UNROLL(x) _Pragma("clang loop unroll(full)") க்கு (x)

#define N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#எனில் __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#undef GGML_METAL_HAS_BF16
#endif

#எனில் defined(GGML_METAL_HAS_BF16)
வகைவரை matrix<bfloat, 4, 4> bfloat4x4;
வகைவரை matrix<bfloat, 2, 4> bfloat2x4;
#endif

constexpr constant நிலையான தசம kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant நிலையான தசம kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

நிலையான உள்வரி முழு best_index_int8(முழு n, constant தசம * val, தசம x) {
    எனில் (x <= val[0]) திரும்பு 0;
    எனில் (x >= val[n-1]) திரும்பு n-1;
    முழு ml = 0, mu = n-1;
    போது (mu-ml > 1) {
        முழு mav = (ml+mu)/2;
        எனில் (x < val[mav]) mu = mav; இல்லையெனில் ml = mav;
    }
    திரும்பு x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

நிலையான உள்வரி தசம e8m0_to_fp32(uint8_t x) {
    uint32_t bits;

    எனில் (x == 0) {
        bits = 0x00400000;
    } இல்லையெனில் {
        bits = (uint32_t) x << 23;
    }

    திரும்பு as_type<தசம>(bits);
}

நிலையான உள்வரி தசம dot(தசம x, தசம y) {
    திரும்பு x*y;
}

நிலையான உள்வரி தசம sum(தசம x) {
    திரும்பு x;
}

நிலையான உள்வரி தசம sum(float4 x) {
    திரும்பு x[0] + x[1] + x[2] + x[3];
}

எண்ணிக்கை ggml_sort_order {
    GGML_SORT_ORDER_ASC,
    GGML_SORT_ORDER_DESC,
};

constant தசம GELU_COEF_A     = 0.044715f;
constant தசம GELU_QUICK_COEF = -1.702f;
constant தசம SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant தசம SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant தசம p_erf  = 0.3275911f;
constant தசம a1_erf = 0.254829592f;
constant தசம a2_erf = -0.284496736f;
constant தசம a3_erf = 1.421413741f;
constant தசம a4_erf = -1.453152027f;
constant தசம a5_erf = 1.061405429f;

template<typename T>
உள்வரி T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * exp(-x * x);
    திரும்பு sign_x * y;
}

template<typename T> T elu_approx(T x);

template<> உள்வரி தசம elu_approx<தசம>(தசம x) {
    திரும்பு (x > 0.f) ? x : (exp(x) - 1);
}

template<> உள்வரி float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (exp(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (exp(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (exp(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (exp(x[3]) - 1.0f);

    திரும்பு res;
}
