#ਹਦਾਇਤ once

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-metal-impl.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <metal_stdlib>

#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ GGML_METAL_HAS_TENSOR
#ਅਨ੍ਤਰ੍ਭਾਵ <metal_tensor>

#ਅਨ੍ਤਰ੍ਭਾਵ <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#ਜੇ_ਅੰਤ

using namespace metal;

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX(x, y) ((x) > (y) ? (x) : (y))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MIN(x, y) ((x) < (y) ? (x) : (y))
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ਅਦਲਾ_ਬਦਲੀ(x, y) { ਸਵੈਚਾਲਿਤ tmp = (x); (x) = (y); (y) = tmp; }

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ FOR_UNROLL(x) _Pragma("clang loop unroll(full)") ਲਈ (x)

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#ਜੇ __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#ਪਰਿਭਾਸ਼ਾ_ਰੱਦ GGML_METAL_HAS_BF16
#ਜੇ_ਅੰਤ

#ਜੇ defined(GGML_METAL_HAS_BF16)
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਮੈਟ੍ਰਿਕਸ<bfloat, 4, 4> bfloat4x4;
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਮੈਟ੍ਰਿਕਸ<bfloat, 2, 4> bfloat2x4;
#ਜੇ_ਅੰਤ

constexpr constant ਸਥਿਰ ਦਸ਼ਮਲਵ kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant ਸਥਿਰ ਦਸ਼ਮਲਵ kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਪੂਰਨ ਉੱਤਮ_ਸੂਚਕਾਂਕ_int8(ਪੂਰਨ n, constant ਦਸ਼ਮਲਵ * val, ਦਸ਼ਮਲਵ x) {
    ਜੇ (x <= val[0]) ਵਾਪਸ 0;
    ਜੇ (x >= val[n-1]) ਵਾਪਸ n-1;
    ਪੂਰਨ ml = 0, mu = n-1;
    ਜਦੋਂ (mu-ml > 1) {
        ਪੂਰਨ mav = (ml+mu)/2;
        ਜੇ (x < val[mav]) mu = mav; ਨਹੀਂਤਾਂ ml = mav;
    }
    ਵਾਪਸ x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ e8m0_ਮੰਜ਼ਿਲ_fp32(uint8_t x) {
    uint32_t ਬਿੱਟ;

    ਜੇ (x == 0) {
        ਬਿੱਟ = 0x00400000;
    } ਨਹੀਂਤਾਂ {
        ਬਿੱਟ = (uint32_t) x << 23;
    }

    ਵਾਪਸ as_type<ਦਸ਼ਮਲਵ>(ਬਿੱਟ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ dot(ਦਸ਼ਮਲਵ x, ਦਸ਼ਮਲਵ y) {
    ਵਾਪਸ x*y;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ਜੋੜ(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ x;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ ਜੋੜ(float4 x) {
    ਵਾਪਸ x[0] + x[1] + x[2] + x[3];
}

ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ {
    GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC,
    GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC,
};

constant ਦਸ਼ਮਲਵ GELU_COEF_A     = 0.044715f;
constant ਦਸ਼ਮਲਵ GELU_ਤੇਜ਼_COEF = -1.702f;
constant ਦਸ਼ਮਲਵ SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant ਦਸ਼ਮਲਵ SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant ਦਸ਼ਮਲਵ p_erf  = 0.3275911f;
constant ਦਸ਼ਮਲਵ a1_erf = 0.254829592f;
constant ਦਸ਼ਮਲਵ a2_erf = -0.284496736f;
constant ਦਸ਼ਮਲਵ a3_erf = 1.421413741f;
constant ਦਸ਼ਮਲਵ a4_erf = -1.453152027f;
constant ਦਸ਼ਮਲਵ a5_erf = 1.061405429f;

ਸਾਂਚਾ<typename T>
ਸਤਰਅੰਦਰ T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * ਘਾਤਾਂਕ(-x * x);
    ਵਾਪਸ sign_x * y;
}

ਸਾਂਚਾ<typename T> T elu_approx(T x);

ਸਾਂਚਾ<> ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ elu_approx<ਦਸ਼ਮਲਵ>(ਦਸ਼ਮਲਵ x) {
    ਵਾਪਸ (x > 0.f) ? x : (ਘਾਤਾਂਕ(x) - 1);
}

ਸਾਂਚਾ<> ਸਤਰਅੰਦਰ float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (ਘਾਤਾਂਕ(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (ਘਾਤਾਂਕ(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (ਘਾਤਾਂਕ(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (ਘਾਤਾਂਕ(x[3]) - 1.0f);

    ਵਾਪਸ res;
}
