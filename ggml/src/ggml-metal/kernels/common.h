#సూచన once

#అన్తర్భావ "ggml-metal-impl.h"

#అన్తర్భావ <metal_stdlib>

#నిర్వచితమైతే GGML_METAL_HAS_TENSOR
#అన్తర్భావ <metal_tensor>

#అన్తర్భావ <MetalPerformancePrimitives/MetalPerformancePrimitives.h>
#అయితే_ముగింపు

using namespace metal;

#నిర్వచించు MAX(x, y) ((x) > (y) ? (x) : (y))
#నిర్వచించు MIN(x, y) ((x) < (y) ? (x) : (y))
#నిర్వచించు మార్పిడి(x, y) { స్వయంచాలక tmp = (x); (x) = (y); (y) = tmp; }

#నిర్వచించు PAD2(x, n) (((x) + (n) - 1) & ~((n) - 1))

#నిర్వచించు FOR_UNROLL(x) _Pragma("clang loop unroll(full)") కొరకు (x)

#నిర్వచించు N_SIMDWIDTH 32 // assuming SIMD group size is 32

// ref: https://developer.apple.com/metal/Metal-Shading-Language-Specification.pdf
//
// cmd:
//   .../usr/bin/metal -dM -E -c                             ggml/src/ggml-metal/kernels/<src>.metal
//   .../usr/bin/metal -dM -E -c -target air64-apple-ios14.0 ggml/src/ggml-metal/kernels/<src>.metal
//
#ఒకవేళ __METAL_VERSION__ < 310 && defined(GGML_METAL_HAS_BF16)
#నిర్వచనం_రద్దు GGML_METAL_HAS_BF16
#అయితే_ముగింపు

#ఒకవేళ defined(GGML_METAL_HAS_BF16)
రకనిర్వచనం మాతృక<bfloat, 4, 4> bfloat4x4;
రకనిర్వచనం మాతృక<bfloat, 2, 4> bfloat2x4;
#అయితే_ముగింపు

constexpr constant స్థిర దశాంశం kvalues_iq4nl_f[16] = {
    -127.f, -104.f, -83.f, -65.f, -49.f, -35.f, -22.f, -10.f, 1.f, 13.f, 25.f, 38.f, 53.f, 69.f, 89.f, 113.f
};

constexpr constant స్థిర దశాంశం kvalues_mxfp4_f[16] = {
    0, .5f, 1.f, 1.5f, 2.f, 3.f, 4.f, 6.f, -0, -.5f, -1.f, -1.5f, -2.f, -3.f, -4.f, -6.f
};

స్థిర అంతఃవరుస పూర్ణాంకం ఉత్తమ_సూచీ_int8(పూర్ణాంకం n, constant దశాంశం * val, దశాంశం x) {
    ఒకవేళ (x <= val[0]) తిరిగి 0;
    ఒకవేళ (x >= val[n-1]) తిరిగి n-1;
    పూర్ణాంకం ml = 0, mu = n-1;
    అయితే (mu-ml > 1) {
        పూర్ణాంకం mav = (ml+mu)/2;
        ఒకవేళ (x < val[mav]) mu = mav; లేకపోతే ml = mav;
    }
    తిరిగి x - val[mu-1] < val[mu] - x ? mu-1 : mu;
}

స్థిర అంతఃవరుస దశాంశం e8m0_గమ్యం_fp32(uint8_t x) {
    uint32_t బిట్లు;

    ఒకవేళ (x == 0) {
        బిట్లు = 0x00400000;
    } లేకపోతే {
        బిట్లు = (uint32_t) x << 23;
    }

    తిరిగి as_type<దశాంశం>(బిట్లు);
}

స్థిర అంతఃవరుస దశాంశం dot(దశాంశం x, దశాంశం y) {
    తిరిగి x*y;
}

స్థిర అంతఃవరుస దశాంశం కూడిక(దశాంశం x) {
    తిరిగి x;
}

స్థిర అంతఃవరుస దశాంశం కూడిక(float4 x) {
    తిరిగి x[0] + x[1] + x[2] + x[3];
}

గణన ggml_క్రమీకరించు_క్రమం {
    GGML_క్రమీకరించు_క్రమం_ASC,
    GGML_క్రమీకరించు_క్రమం_DESC,
};

constant దశాంశం GELU_COEF_A     = 0.044715f;
constant దశాంశం GELU_శీఘ్ర_COEF = -1.702f;
constant దశాంశం SQRT_2_OVER_PI  = 0.79788456080286535587989211986876f;
constant దశాంశం SQRT_2_INV      = 0.70710678118654752440084436210484f;

// based on Abramowitz and Stegun formula 7.1.26 or similar Hastings' approximation
// ref: https://www.johndcook.com/blog/python_erf/
constant దశాంశం p_erf  = 0.3275911f;
constant దశాంశం a1_erf = 0.254829592f;
constant దశాంశం a2_erf = -0.284496736f;
constant దశాంశం a3_erf = 1.421413741f;
constant దశాంశం a4_erf = -1.453152027f;
constant దశాంశం a5_erf = 1.061405429f;

మూస<typename T>
అంతఃవరుస T erf_approx(T x) {
    T sign_x = sign(x);
    x = fabs(x);
    T t = 1.0f / (1.0f + p_erf * x);
    T y = 1.0f - (((((a5_erf * t + a4_erf) * t) + a3_erf) * t + a2_erf) * t + a1_erf) * t * ఘాతాంకం(-x * x);
    తిరిగి sign_x * y;
}

మూస<typename T> T elu_approx(T x);

మూస<> అంతఃవరుస దశాంశం elu_approx<దశాంశం>(దశాంశం x) {
    తిరిగి (x > 0.f) ? x : (ఘాతాంకం(x) - 1);
}

మూస<> అంతఃవరుస float4 elu_approx<float4>(float4 x) {
    float4 res;

    res[0] = (x[0] > 0.0f) ? x[0] : (ఘాతాంకం(x[0]) - 1.0f);
    res[1] = (x[1] > 0.0f) ? x[1] : (ఘాతాంకం(x[1]) - 1.0f);
    res[2] = (x[2] > 0.0f) ? x[2] : (ఘాతాంకం(x[2]) - 1.0f);
    res[3] = (x[3] > 0.0f) ? x[3] : (ఘాతాంకం(x[3]) - 1.0f);

    తిరిగి res;
}
