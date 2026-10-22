#निर्देशः once

#अन्तर्भाव "ggml-metal-impl.h"

#अन्तर्भाव <metal_stdlib>

#यदि_परिभाषितम् GGML_METAL_HAS_TENSOR
#अन्तर्भाव <metal_tensor>

#अन्तर्भाव <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#यदि_अन्तः

using namespace metal;

#परिभाषय MAX(x, y) ((x) > (y) ? (x) : (y))
#परिभाषय MIN(x, y) ((x) < (y) ? (x) : (y))
#परिभाषय SWAP(x, y) { स्वचालित tmp = (x); (x) = (y); (y) = tmp; }

#परिभाषय PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#परिभाषय FOR_UNROLL(x) _Pragma("clang loop unroll(full)") प्रति (x)

#परिभाषय N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#यदि __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#अपरिभाषय GGML_METAL_HAS_BF16
#यदि_अन्तः

#यदि defined(GGML_METAL_HAS_BF16)
प्रकारनिर्देश matrix<bfloat, 4, 4> bfloat4x4;
प्रकारनिर्देश matrix<bfloat, 2, 4> bfloat2x4;
#यदि_अन्तः

constexpr constant स्थिर भिन्न kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant स्थिर भिन्न kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

स्थिर अन्तःस्थ पूर्णाङ्क best_सूचकांक_int8(पूर्णाङ्क n, constant भिन्न * val, भिन्न x) {
    यदि (x <= val[0]) प्रत्यागम 0;
    यदि (x >= val[n-1]) प्रत्यागम n-1;
    पूर्णाङ्क ml = 0, mu = n-1;
    यावत् (mu-ml > 1) {
        पूर्णाङ्क mav = (ml+mu)/2;
        यदि (x < val[mav]) mu = mav; अन्यथा ml = mav;
    }
    प्रत्यागम x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

स्थिर अन्तःस्थ भिन्न e8m0_गन्तव्यम्_fp32(uint8_t x) {
    uint32_t द्वयङ्काः;

    यदि (x == 0) {
        द्वयङ्काः = 0x00400000;
    } अन्यथा {
        द्वयङ्काः = (uint32_t) x << 23;
    }

    प्रत्यागम as_type<भिन्न>(द्वयङ्काः);
}

स्थिर अन्तःस्थ भिन्न dot(भिन्न x, भिन्न y) {
    प्रत्यागम x*y;
}

स्थिर अन्तःस्थ भिन्न योग_2(भिन्न x) {
    प्रत्यागम x;
}

स्थिर अन्तःस्थ भिन्न योग_2(float4 x) {
    प्रत्यागम x[0] + x[1] + x[2] + x[3];
}

गणना ggml_क्रमय_क्रमः {
    GGML_क्रमय_क्रमः_ASC,
    GGML_क्रमय_क्रमः_DESC,
};

constant भिन्न GELU_COEF_A     = 0.044715f;
constant भिन्न GELU_शीघ्र_COEF = -1.702f;
constant भिन्न SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant भिन्न SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant भिन्न p_erf  = 0.3275911f;
constant भिन्न a1_erf = 0.254829592f;
constant भिन्न a2_erf = -0.284496736f;
constant भिन्न a3_erf = 1.421413741f;
constant भिन्न a4_erf = -1.453152027f;
constant भिन्न a5_erf = 1.061405429f;

प्रतिरूपम्<typename T>
अन्तःस्थ T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * घातांकः(-x * x);
    प्रत्यागम sign_x * y;
}

प्रतिरूपम्<typename T> T elu_approx(T x);

प्रतिरूपम्<> अन्तःस्थ भिन्न elu_approx<भिन्न>(भिन्न x) {
    प्रत्यागम (x > 0.f) ? x : (घातांकः(x) - 1);
}

प्रतिरूपम्<> अन्तःस्थ float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (घातांकः(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (घातांकः(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (घातांकः(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (घातांकः(x[3]) - 1.0f);

    प्रत्यागम res;
}
