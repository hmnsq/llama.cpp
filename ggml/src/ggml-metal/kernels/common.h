#ಸೂಚನೆ once

#ಅನ್ತರ್ಭಾವ "ggml-metal-impl.h"

#ಅನ್ತರ್ಭಾವ <metal_stdlib>

#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ GGML_METAL_HAS_TENSOR
#ಅನ್ತರ್ಭಾವ <metal_tensor>

#ಅನ್ತರ್ಭಾವ <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#ಆದರೆ_ಅಂತ್ಯ

using namespace metal;

#ವ್ಯಾಖ್ಯಾನಿಸು MAX(x, y) ((x) > (y) ? (x) : (y))
#ವ್ಯಾಖ್ಯಾನಿಸು MIN(x, y) ((x) < (y) ? (x) : (y))
#ವ್ಯಾಖ್ಯಾನಿಸು ಅದಲುಬದಲು(x, y) { ಸ್ವಯಂಚಾಲಿತ tmp = (x); (x) = (y); (y) = tmp; }

#ವ್ಯಾಖ್ಯಾನಿಸು PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#ವ್ಯಾಖ್ಯಾನಿಸು FOR_UNROLL(x) _Pragma("clang loop unroll(full)") ಗಾಗಿ (x)

#ವ್ಯಾಖ್ಯಾನಿಸು N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#ಆದರೆ __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#ವ್ಯಾಖ್ಯಾನ_ರದ್ದು GGML_METAL_HAS_BF16
#ಆದರೆ_ಅಂತ್ಯ

#ಆದರೆ defined(GGML_METAL_HAS_BF16)
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಮಾತೃಕೆ<bfloat, 4, 4> bfloat4x4;
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ಮಾತೃಕೆ<bfloat, 2, 4> bfloat2x4;
#ಆದರೆ_ಅಂತ್ಯ

constexpr constant ಸ್ಥಿರ ದಶಮಾಂಶ kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant ಸ್ಥಿರ ದಶಮಾಂಶ kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಪೂರ್ಣಾಂಕ ಉತ್ತಮ_ಸೂಚ್ಯಂಕ_int8(ಪೂರ್ಣಾಂಕ n, constant ದಶಮಾಂಶ * val, ದಶಮಾಂಶ x) {
    ಆದರೆ (x <= val[0]) ಹಿಂತಿರುಗಿ 0;
    ಆದರೆ (x >= val[n-1]) ಹಿಂತಿರುಗಿ n-1;
    ಪೂರ್ಣಾಂಕ ml = 0, mu = n-1;
    ಆಗ (mu-ml > 1) {
        ಪೂರ್ಣಾಂಕ mav = (ml+mu)/2;
        ಆದರೆ (x < val[mav]) mu = mav; ಇಲ್ಲದಿದ್ದರೆ ml = mav;
    }
    ಹಿಂತಿರುಗಿ x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ e8m0_ಗಮ್ಯ_fp32(uint8_t x) {
    uint32_t ಬಿಟ್ಟುಗಳು;

    ಆದರೆ (x == 0) {
        ಬಿಟ್ಟುಗಳು = 0x00400000;
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಬಿಟ್ಟುಗಳು = (uint32_t) x << 23;
    }

    ಹಿಂತಿರುಗಿ as_type<ದಶಮಾಂಶ>(ಬಿಟ್ಟುಗಳು);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ dot(ದಶಮಾಂಶ x, ದಶಮಾಂಶ y) {
    ಹಿಂತಿರುಗಿ x*y;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ಮೊತ್ತ(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ x;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ ಮೊತ್ತ(float4 x) {
    ಹಿಂತಿರುಗಿ x[0] + x[1] + x[2] + x[3];
}

ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ {
    GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC,
    GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC,
};

constant ದಶಮಾಂಶ GELU_COEF_A     = 0.044715f;
constant ದಶಮಾಂಶ GELU_ಶೀಘ್ರ_COEF = -1.702f;
constant ದಶಮಾಂಶ SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant ದಶಮಾಂಶ SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant ದಶಮಾಂಶ p_erf  = 0.3275911f;
constant ದಶಮಾಂಶ a1_erf = 0.254829592f;
constant ದಶಮಾಂಶ a2_erf = -0.284496736f;
constant ದಶಮಾಂಶ a3_erf = 1.421413741f;
constant ದಶಮಾಂಶ a4_erf = -1.453152027f;
constant ದಶಮಾಂಶ a5_erf = 1.061405429f;

ಮಾದರಿ<typename T>
ಸಾಲೊಳಗೆ T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * ಘಾತಾಂಕ(-x * x);
    ಹಿಂತಿರುಗಿ sign_x * y;
}

ಮಾದರಿ<typename T> T elu_approx(T x);

ಮಾದರಿ<> ಸಾಲೊಳಗೆ ದಶಮಾಂಶ elu_approx<ದಶಮಾಂಶ>(ದಶಮಾಂಶ x) {
    ಹಿಂತಿರುಗಿ (x > 0.f) ? x : (ಘಾತಾಂಕ(x) - 1);
}

ಮಾದರಿ<> ಸಾಲೊಳಗೆ float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (ಘಾತಾಂಕ(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (ಘಾತಾಂಕ(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (ಘಾತಾಂಕ(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (ಘಾತಾಂಕ(x[3]) - 1.0f);

    ಹಿಂತಿರುಗಿ res;
}
