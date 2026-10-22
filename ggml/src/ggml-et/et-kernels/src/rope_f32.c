//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#ಅನ್ತರ್ಭಾವ "ggml_tensor.h"
#ಅನ್ತರ್ಭಾವ "math_fp.h"
#ಅನ್ತರ್ಭಾವ "platform.h"

#ಅನ್ತರ್ಭಾವ <etsoc/common/utils.h>
#ಅನ್ತರ್ಭಾವ <stdint.h>

// ROPE constants (matching GGML definitions)
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_ROPE_TYPE_NEOX   2
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_ROPE_TYPE_MROPE  8
#ವ್ಯಾಖ್ಯಾನಿಸು GGML_ROPE_TYPE_IMROPE 40
#ವ್ಯಾಖ್ಯಾನಿಸು MAX_ROPE_HALF_DIMS    256  // supports up to n_dims=512

#ವ್ಯಾಖ್ಯಾನಿಸು ROPE_VEC_ಅಗಲ 8

#ವ್ಯಾಖ್ಯಾನಿಸು ROPE_PI         3.14159265358979323846f
#ವ್ಯಾಖ್ಯಾನಿಸು ROPE_TWO_PI     6.28318530717958647693f
#ವ್ಯಾಖ್ಯಾನಿಸು ROPE_PI_OVER_2  1.57079632679489661923f
#ವ್ಯಾಖ್ಯಾನಿಸು ROPE_INV_TWO_PI 0.15915494309189533577f

// ROPE operation parameters structure (matches ggml-et-ops.h)
ಪ್ರಕಾರವ್ಯಾಖ್ಯೆ ರಚನೆ {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t ಬಹುಲಕ;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    ದಶಮಾಂಶ   freq_ತಳ;    // Base frequency (usually 10000.0f)
    ದಶಮಾಂಶ   freq_scale;   // Frequency scaling factor
    ದಶಮಾಂಶ   ext_ಗುಣಕ;   // Extension factor for YaRN
    ದಶಮಾಂಶ   attn_ಗುಣಕ;  // Attention factor for YaRN
    ದಶಮಾಂಶ   beta_fast;    // Fast beta for YaRN
    ದಶಮಾಂಶ   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
ರಚನೆ ggml_et_rope_params {
    ರಚನೆ ggml_tensor src0;  // F32 input tensor
    ರಚನೆ ggml_tensor src1;  // I32 position tensor
    ರಚನೆ ggml_tensor src2;  // F32 frequency factors (optional)
    ರಚನೆ ggml_tensor dst;   // F32 output tensor
    rope_params_t      rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

// floor/ceil with ±inf and NaN passthrough.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ rope_floorf(ದಶಮಾಂಶ x) {
    ಒಕ್ಕೂಟ {
        ದಶಮಾಂಶ    f;
        uint32_t u;
    } v = { .f = x };

    ಸ್ಥಿರಾಂಕ uint32_t expo = (v.u >> 23) & 0xFF;
    ಆದರೆ (expo == 0xFF) {
        ಹಿಂತಿರುಗಿ x;  // inf or NaN
    }
    ಆದರೆ (expo >= 23 + 127) {
        ಹಿಂತಿರುಗಿ x;  // already integer-valued
    }
    ಪೂರ್ಣಾಂಕ i = (ಪೂರ್ಣಾಂಕ) x;
    ಹಿಂತಿರುಗಿ (x < 0.0f && (ದಶಮಾಂಶ) i != x) ? (ದಶಮಾಂಶ) (i - 1) : (ದಶಮಾಂಶ) i;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ rope_ceilf(ದಶಮಾಂಶ x) {
    ಒಕ್ಕೂಟ {
        ದಶಮಾಂಶ    f;
        uint32_t u;
    } v = { .f = x };

    ಸ್ಥಿರಾಂಕ uint32_t expo = (v.u >> 23) & 0xFF;
    ಆದರೆ (expo == 0xFF) {
        ಹಿಂತಿರುಗಿ x;  // inf or NaN
    }
    ಆದರೆ (expo >= 23 + 127) {
        ಹಿಂತಿರುಗಿ x;  // already integer-valued
    }
    ಪೂರ್ಣಾಂಕ i = (ಪೂರ್ಣಾಂಕ) x;
    ಹಿಂತಿರುಗಿ (x > 0.0f && (ದಶಮಾಂಶ) i != x) ? (ದಶಮಾಂಶ) (i + 1) : (ದಶಮಾಂಶ) i;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ rope_yarn_ramp(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಕೆಳಗಿನ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಮೇಲಿನ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0) {
    ದಶಮಾಂಶ denom = ಮೇಲಿನ - ಕೆಳಗಿನ;
    ಆದರೆ (denom < 0.001f) {
        denom = 0.001f;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y       = et_fdiv((ದಶಮಾಂಶ) (i0 / 2) - ಕೆಳಗಿನ, denom);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    ಹಿಂತಿರುಗಿ 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ rope_yarn_corr_dim(ಪೂರ್ಣಾಂಕ n_dims, ಪೂರ್ಣಾಂಕ n_ctx_orig, ದಶಮಾಂಶ beta, ದಶಮಾಂಶ freq_ತಳ) {
    ಹಿಂತಿರುಗಿ (ದಶಮಾಂಶ) n_dims *
           et_fdiv(et_logf(et_fdiv((ದಶಮಾಂಶ) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_ತಳ));
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rope_yarn_corr_dims(ಪೂರ್ಣಾಂಕ   n_dims,
                                       ಪೂರ್ಣಾಂಕ   n_ctx_orig,
                                       ದಶಮಾಂಶ freq_ತಳ,
                                       ದಶಮಾಂಶ beta_fast,
                                       ದಶಮಾಂಶ beta_slow,
                                       ದಶಮಾಂಶ dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    ದಶಮಾಂಶ ಆರಂಭ = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_ತಳ));
    ದಶಮಾಂಶ ಕೊನೆ   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_ತಳ));

    dims[0] = ಆರಂಭ > 0.0f ? ಆರಂಭ : 0.0f;
    dims[1] = ಕೊನೆ < (ದಶಮಾಂಶ) (n_dims - 1) ? ಕೊನೆ : (ದಶಮಾಂಶ) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ rope_ps_one[ROPE_VEC_ಅಗಲ]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ rope_ps_c3[ROPE_VEC_ಅಗಲ]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ rope_ps_c5[ROPE_VEC_ಅಗಲ]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ rope_ps_c7[ROPE_VEC_ಅಗಲ]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ rope_ps_c9[ROPE_VEC_ಅಗಲ]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ rope_ps_c11[ROPE_VEC_ಅಗಲ]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint64_t rope_ps_enter_fullmask(ಶೂನ್ಯ) {
    uint64_t old_ಮುಖವಾಡ;
    __asm__ ಅಸ್ಥಿರ(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_ಮುಖವಾಡ)
        :
        : "t0", "memory");
    ಹಿಂತಿರುಗಿ old_ಮುಖವಾಡ;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rope_ps_leave_fullmask(uint64_t old_ಮುಖವಾಡ) {
    __asm__ ಅಸ್ಥಿರ("mova.m.x %0           \n\t" : : "r"(old_ಮುಖವಾಡ) : "memory");
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rope_poly_sin_block8(ದಶಮಾಂಶ * ಹೊರಹೋಗು, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * x) {
    __asm__ ಅಸ್ಥಿರ(
        "flw.ps    f0,  %[x]           \n\t"
        "fmul.ps   f1,  f0,  f0        \n\t"

        "flw.ps    f2,  %[c11]         \n\t"
        "flw.ps    f3,  %[c9]          \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "flw.ps    f3,  %[c7]          \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "flw.ps    f3,  %[c5]          \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "flw.ps    f3,  %[c3]          \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "flw.ps    f3,  %[one]         \n\t"
        "fnmsub.ps f2,  f1,  f2,  f3   \n\t"

        "fmul.ps   f4,  f0,  f2        \n\t"
        "fsw.ps    f4,  %[out]         \n\t"
        : [ಹೊರಹೋಗು] "=m"(*(ದಶಮಾಂಶ (*)[ROPE_VEC_ಅಗಲ]) ಹೊರಹೋಗು)
        : [x] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[ROPE_VEC_ಅಗಲ]) x), [one] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[ROPE_VEC_ಅಗಲ]) rope_ps_one),
          [c3] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[ROPE_VEC_ಅಗಲ]) rope_ps_c3),
          [c5] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[ROPE_VEC_ಅಗಲ]) rope_ps_c5),
          [c7] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[ROPE_VEC_ಅಗಲ]) rope_ps_c7),
          [c9] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[ROPE_VEC_ಅಗಲ]) rope_ps_c9),
          [c11] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[ROPE_VEC_ಅಗಲ]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rope_sincos_block8(ದಶಮಾಂಶ * sin8, ದಶಮಾಂಶ * cos8, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * theta8) {
    ದಶಮಾಂಶ sin_fold[ROPE_VEC_ಅಗಲ] __attribute__((aligned(32)));
    ದಶಮಾಂಶ cos_fold[ROPE_VEC_ಅಗಲ] __attribute__((aligned(32)));
    ದಶಮಾಂಶ sin_sign[ROPE_VEC_ಅಗಲ] __attribute__((aligned(32)));
    ದಶಮಾಂಶ cos_sign[ROPE_VEC_ಅಗಲ] __attribute__((aligned(32)));

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ROPE_VEC_ಅಗಲ; ++i) {
        ದಶಮಾಂಶ x = theta8[i];

        ಆದರೆ (x > ROPE_PI || x < -ROPE_PI) {
            ದಶಮಾಂಶ ಚಕ್ರಗಳು = x * ROPE_INV_TWO_PI;
            ಪೂರ್ಣಾಂಕ   n      = (ಪೂರ್ಣಾಂಕ) ಚಕ್ರಗಳು;
            ಆದರೆ (x < 0.0f) {
                n--;
            }
            x = x - (ದಶಮಾಂಶ) n * ROPE_TWO_PI;
        }

        {
            ದಶಮಾಂಶ y = x;
            ದಶಮಾಂಶ s = 1.0f;
            ಆದರೆ (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            ದಶಮಾಂಶ y = x + ROPE_PI_OVER_2;
            ಆದರೆ (y > ROPE_PI || y < -ROPE_PI) {
                ದಶಮಾಂಶ ಚಕ್ರಗಳು = y * ROPE_INV_TWO_PI;
                ಪೂರ್ಣಾಂಕ   n      = (ಪೂರ್ಣಾಂಕ) ಚಕ್ರಗಳು;
                ಆದರೆ (y < 0.0f) {
                    n--;
                }
                y = y - (ದಶಮಾಂಶ) n * ROPE_TWO_PI;
            }

            ದಶಮಾಂಶ s = 1.0f;
            ಆದರೆ (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            cos_fold[i] = y;
            cos_sign[i] = s;
        }
    }

    {
        ಸ್ಥಿರಾಂಕ uint64_t saved_ಮುಖವಾಡ = rope_ps_enter_fullmask();

        rope_poly_sin_block8(sin8, sin_fold);
        rope_poly_sin_block8(cos8, cos_fold);

        __asm__ ಅಸ್ಥಿರ(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(ದಶಮಾಂಶ (*)[ROPE_VEC_ಅಗಲ]) sin8), [cout] "=m"(*(ದಶಮಾಂಶ (*)[ROPE_VEC_ಅಗಲ]) cos8)
            : [sinv] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[ROPE_VEC_ಅಗಲ]) sin8),
              [sinsgn] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[ROPE_VEC_ಅಗಲ]) sin_sign),
              [cosv] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[ROPE_VEC_ಅಗಲ]) cos8),
              [cossgn] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[ROPE_VEC_ಅಗಲ]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_ಮುಖವಾಡ);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rope_yarn_scalar(ದಶಮಾಂಶ       theta_extrap,
                                    ದಶಮಾಂಶ       freq_scale,
                                    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ corr_dims[2],
                                    int64_t     i0,
                                    ದಶಮಾಂಶ       ext_ಗುಣಕ,
                                    ದಶಮಾಂಶ       mscale,
                                    ದಶಮಾಂಶ *     cos_theta,
                                    ದಶಮಾಂಶ *     sin_theta) {
    ದಶಮಾಂಶ theta_interp = freq_scale * theta_extrap;
    ದಶಮಾಂಶ theta        = theta_interp;

    ಆದರೆ (ext_ಗುಣಕ != 0.0f) {
        ದಶಮಾಂಶ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (ಪೂರ್ಣಾಂಕ) i0) * ext_ಗುಣಕ;
        theta          = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
        mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
    }

    *cos_theta = et_cosf(theta) * mscale;
    *sin_theta = et_sinf(theta) * mscale;
}

// Populate cos/sin cache for a given position using running theta product
// Experiment 1:
//   - theta construction and YaRN mixing stay scalar
//   - actual sin/cos approximation is done in vec8 blocks
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ compute_rope_cache(ದಶಮಾಂಶ *       cos_cache,
                                      ದಶಮಾಂಶ *       sin_cache,
                                      int32_t       n_dims,
                                      ದಶಮಾಂಶ         theta_scale,
                                      int32_t       pos,
                                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * freq_factors,
                                      ದಶಮಾಂಶ         freq_scale,
                                      ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   corr_dims[2],
                                      ದಶಮಾಂಶ         ext_ಗುಣಕ,
                                      ದಶಮಾಂಶ         attn_ಗುಣಕ) {
    ಸ್ಥಿರಾಂಕ int32_t half_dims = n_dims / 2;
    ದಶಮಾಂಶ         theta     = 1.0f;

    int32_t dim_idx = 0;

    ಗಾಗಿ (; dim_idx + ROPE_VEC_ಅಗಲ <= half_dims; dim_idx += ROPE_VEC_ಅಗಲ) {
        ದಶಮಾಂಶ theta_ಖಂಡ[ROPE_VEC_ಅಗಲ] __attribute__((aligned(32)));
        ದಶಮಾಂಶ theta_local = theta;
        ದಶಮಾಂಶ mscale      = attn_ಗುಣಕ;

        ಆದರೆ (ext_ಗುಣಕ != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ROPE_VEC_ಅಗಲ; ++i) {
            ಸ್ಥಿರಾಂಕ int32_t pair_idx     = dim_idx + i;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   theta_ತಳ   = (ದಶಮಾಂಶ) pos * theta_local;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   theta_extrap = et_fdiv(theta_ತಳ, ff);

            ದಶಮಾಂಶ theta_interp = freq_scale * theta_extrap;
            ದಶಮಾಂಶ theta_mix    = theta_interp;

            ಆದರೆ (ext_ಗುಣಕ != 0.0f) {
                ದಶಮಾಂಶ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_ಗುಣಕ;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_ಖಂಡ[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_ಖಂಡ);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ROPE_VEC_ಅಗಲ; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    ಗಾಗಿ (; dim_idx < half_dims; ++dim_idx) {
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_ತಳ = (ದಶಮಾಂಶ) pos * theta;

        rope_yarn_scalar(et_fdiv(theta_ತಳ, ff), freq_scale, corr_dims, dim_idx * 2, ext_ಗುಣಕ, attn_ಗುಣಕ,
                         &cos_cache[dim_idx], &sin_cache[dim_idx]);

        theta *= theta_scale;
    }
}

//------------------------------------------------------------------------------
// IMROPE cache build (interleaved multi-modal RoPE for Qwen3VL)
//------------------------------------------------------------------------------

// Builds cos/sin cache with 4 interleaved position channels.
// Each dimension pair selects from {theta_t, theta_h, theta_w, theta_e}
// using a mod-3 sector pattern, matching the CPU reference exactly.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ compute_imrope_cache(ದಶಮಾಂಶ *       cos_cache,
                                        ದಶಮಾಂಶ *       sin_cache,
                                        int32_t       n_dims,
                                        ದಶಮಾಂಶ         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        ಸ್ಥಿರಾಂಕ int32_t sections[4],
                                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * freq_factors,
                                        ದಶಮಾಂಶ         freq_scale,
                                        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   corr_dims[2],
                                        ದಶಮಾಂಶ         ext_ಗುಣಕ,
                                        ದಶಮಾಂಶ         attn_ಗುಣಕ) {
    ಸ್ಥಿರಾಂಕ int32_t half_dims = n_dims / 2;
    ಸ್ಥಿರಾಂಕ int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    ದಶಮಾಂಶ theta_t = (ದಶಮಾಂಶ) pos_t;
    ದಶಮಾಂಶ theta_h = (ದಶಮಾಂಶ) pos_h;
    ದಶಮಾಂಶ theta_w = (ದಶಮಾಂಶ) pos_w;
    ದಶಮಾಂಶ theta_e = (ದಶಮಾಂಶ) pos_e;

    int32_t dim_idx = 0;

    ಗಾಗಿ (; dim_idx + ROPE_VEC_ಅಗಲ <= half_dims; dim_idx += ROPE_VEC_ಅಗಲ) {
        ದಶಮಾಂಶ theta_ಖಂಡ[ROPE_VEC_ಅಗಲ] __attribute__((aligned(32)));
        ದಶಮಾಂಶ mscale = attn_ಗುಣಕ;

        ಆದರೆ (ext_ಗುಣಕ != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ROPE_VEC_ಅಗಲ; ++i) {
            ಸ್ಥಿರಾಂಕ int32_t pair_idx = dim_idx + i;
            ಸ್ಥಿರಾಂಕ int32_t sector   = pair_idx % sect_dims;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            ದಶಮಾಂಶ theta;
            ಆದರೆ (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } ಇಲ್ಲದಿದ್ದರೆ {
                theta = theta_e;
            }

            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta_extrap = et_fdiv(theta, ff);
            ದಶಮಾಂಶ       theta_interp = freq_scale * theta_extrap;
            ದಶಮಾಂಶ       theta_mix    = theta_interp;

            ಆದರೆ (ext_ಗುಣಕ != 0.0f) {
                ದಶಮಾಂಶ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_ಗುಣಕ;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_ಖಂಡ[i] = theta_mix;

            // All 4 thetas advance every iteration
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_ಖಂಡ);

        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < ROPE_VEC_ಅಗಲ; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    ಗಾಗಿ (; dim_idx < half_dims; ++dim_idx) {
        ಸ್ಥಿರಾಂಕ int32_t sector = dim_idx % sect_dims;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        ದಶಮಾಂಶ theta;
        ಆದರೆ (sector % 3 == 1 && sector < 3 * sections[1]) {
            theta = theta_h;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector % 3 == 2 && sector < 3 * sections[2]) {
            theta = theta_w;
        } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector % 3 == 0 && sector < 3 * sections[0]) {
            theta = theta_t;
        } ಇಲ್ಲದಿದ್ದರೆ {
            theta = theta_e;
        }

        rope_yarn_scalar(et_fdiv(theta, ff), freq_scale, corr_dims, dim_idx * 2, ext_ಗುಣಕ, attn_ಗುಣಕ,
                         &cos_cache[dim_idx], &sin_cache[dim_idx]);

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }
}

//------------------------------------------------------------------------------
// Entry point
//------------------------------------------------------------------------------

ಪೂರ್ಣಾಂಕ ನಮೂದು_ಬಿಂದು(ರಚನೆ ggml_et_rope_params * params, ಶೂನ್ಯ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ಆದರೆ (!kernel_env) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಪೂರ್ಣಾಂಕ thread_id   = get_relative_thread_id(kernel_env->shire_ಮುಖವಾಡ);
    ಪೂರ್ಣಾಂಕ ಅಂಕೆ_ಎಳೆಗಳು = get_ಅಂಕೆ_ಎಳೆಗಳು(kernel_env->shire_ಮುಖವಾಡ);

    ಆದರೆ (thread_id < 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಆದರೆ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ರಚನೆ ggml_tensor * src0 = &params->src0;
    ರಚನೆ ggml_tensor * src1 = &params->src1;
    ರಚನೆ ggml_tensor * src2 = &params->src2;
    ರಚನೆ ggml_tensor * dst  = &params->dst;

    ಆದರೆ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   src0_ದತ್ತಾಂಶ    = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src0->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ int32_t * src1_ದತ್ತಾಂಶ    = (ಸ್ಥಿರಾಂಕ int32_t *) src1->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   freq_factors = (src2 && src2->ದತ್ತಾಂಶ) ? (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) src2->ದತ್ತಾಂಶ : ಖಾಲಿ;
    ದಶಮಾಂಶ *         dst_ದತ್ತಾಂಶ     = (ದಶಮಾಂಶ *) dst->ದತ್ತಾಂಶ;

    ಆದರೆ (!src0_ದತ್ತಾಂಶ || !src1_ದತ್ತಾಂಶ || !dst_ದತ್ತಾಂಶ) {
        ಹಿಂತಿರುಗಿ -1;
    }
#ವ್ಯಾಖ್ಯಾನಿತವಾದರೆ ET_UBERKERNEL
    ಸ್ಥಿರಾಂಕ size_t src0_ಬೈಟುಗಳು = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ಸ್ಥಿರಾಂಕ size_t src1_ಬೈಟುಗಳು = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_ದತ್ತಾಂಶ, src0_ಬೈಟುಗಳು);
    evict_region_past_l2(src1_ದತ್ತಾಂಶ, src1_ಬೈಟುಗಳು);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#ಆದರೆ_ಅಂತ್ಯ
    ಸ್ಥಿರಾಂಕ int64_t ತಲೆ_dim = src0->ne[0];
    ಸ್ಥಿರಾಂಕ int64_t heads    = src0->ne[1];
    ಸ್ಥಿರಾಂಕ int64_t seq_len  = src0->ne[2];
    ಸ್ಥಿರಾಂಕ int64_t batch    = src0->ne[3];

    ಸ್ಥಿರಾಂಕ rope_params_t * rope_params = &params->rope_params;
    ಸ್ಥಿರಾಂಕ int32_t         n_dims      = rope_params->n_dims;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ           freq_ತಳ   = rope_params->freq_ತಳ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ           freq_scale  = rope_params->freq_scale;
    ಸ್ಥಿರಾಂಕ int32_t         ಬಹುಲಕ        = rope_params->ಬಹುಲಕ;

    ಆದರೆ (n_dims <= 0 || n_dims > ತಲೆ_dim || (n_dims & 1) != 0) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ಆದರೆ (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        ಹಿಂತಿರುಗಿ -1;
    }

    ದಶಮಾಂಶ cos_cache[MAX_ROPE_HALF_DIMS];
    ದಶಮಾಂಶ sin_cache[MAX_ROPE_HALF_DIMS];

    ದಶಮಾಂಶ corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_ತಳ, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    ಸ್ಥಿರಾಂಕ int64_t ಒಟ್ಟು_heads = batch * seq_len * heads;
    ಸ್ಥಿರಾಂಕ int64_t ಆರಂಭ_wu    = (ಒಟ್ಟು_heads * thread_id) / ಅಂಕೆ_ಎಳೆಗಳು;
    ಸ್ಥಿರಾಂಕ int64_t ಕೊನೆ_wu      = (ಒಟ್ಟು_heads * (thread_id + 1)) / ಅಂಕೆ_ಎಳೆಗಳು;

    ಆದರೆ (ಆರಂಭ_wu >= ಕೊನೆ_wu) {
        ಹಿಂತಿರುಗಿ 0;
    }

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   theta_scale       = et_powf(freq_ತಳ, et_fdiv(-2.0f, (ದಶಮಾಂಶ) n_dims));
    ಸ್ಥಿರಾಂಕ int32_t half_dims         = n_dims / 2;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     is_neox           = (ಬಹುಲಕ & GGML_ROPE_TYPE_NEOX) != 0;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     is_imrope         = (ಬಹುಲಕ == GGML_ROPE_TYPE_IMROPE);
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t ಕೊನೆಯ_pos   = -1;
    int32_t ಕೊನೆಯ_pos_h = -1;
    int32_t ಕೊನೆಯ_pos_w = -1;
    int32_t ಕೊನೆಯ_pos_e = -1;

    ಗಾಗಿ (int64_t wu = ಆರಂಭ_wu; wu < ಕೊನೆ_wu; ++wu) {
        ಸ್ಥಿರಾಂಕ int64_t h = wu % heads;
        ಸ್ಥಿರಾಂಕ int64_t s = (wu / heads) % seq_len;
        ಸ್ಥಿರಾಂಕ int64_t b = wu / (heads * seq_len);

        ಆದರೆ (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            ಸ್ಥಿರಾಂಕ int32_t pt = src1_ದತ್ತಾಂಶ[s] + rope_params->n_past;
            ಸ್ಥಿರಾಂಕ int32_t ph = src1_ದತ್ತಾಂಶ[s + seq_len] + rope_params->n_past;
            ಸ್ಥಿರಾಂಕ int32_t pw = src1_ದತ್ತಾಂಶ[s + seq_len * 2] + rope_params->n_past;
            ಸ್ಥಿರಾಂಕ int32_t pe = src1_ದತ್ತಾಂಶ[s + seq_len * 3] + rope_params->n_past;

            ಆದರೆ (pt != ಕೊನೆಯ_pos || ph != ಕೊನೆಯ_pos_h || pw != ಕೊನೆಯ_pos_w || pe != ಕೊನೆಯ_pos_e) {
                compute_imrope_cache(cos_cache, sin_cache, n_dims, theta_scale, pt, ph, pw, pe, rope_params->sections,
                                     freq_factors, freq_scale, corr_dims, rope_params->ext_ಗುಣಕ,
                                     rope_params->attn_ಗುಣಕ);
                ಕೊನೆಯ_pos   = pt;
                ಕೊನೆಯ_pos_h = ph;
                ಕೊನೆಯ_pos_w = pw;
                ಕೊನೆಯ_pos_e = pe;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ int32_t pos = src1_ದತ್ತಾಂಶ[s] + rope_params->n_past;

            ಆದರೆ (pos != ಕೊನೆಯ_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_ಗುಣಕ, rope_params->attn_ಗುಣಕ);
                ಕೊನೆಯ_pos = pos;
            }
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ತಲೆ_src =
            (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) ((ಸ್ಥಿರಾಂಕ ಅಕ್ಷರ *) src0_ದತ್ತಾಂಶ + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        ದಶಮಾಂಶ * ತಲೆ_dst = (ದಶಮಾಂಶ *) ((ಅಕ್ಷರ *) dst_ದತ್ತಾಂಶ + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        ಗಾಗಿ (int64_t d = n_dims; d < ತಲೆ_dim; ++d) {
            ತಲೆ_dst[d] = ತಲೆ_src[d];
        }

        ಆದರೆ (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ;
            __asm__ ಅಸ್ಥಿರ("mova.x.m %0" : "=r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));
            __asm__ ಅಸ್ಥಿರ("mov.m.x m0, x0, 0xFF");

            ಗಾಗಿ (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                __asm__ ಅಸ್ಥಿರ(
                    "flw.ps f0, %[x0_src]       \n\t"
                    "flw.ps f1, %[x1_src]       \n\t"
                    "flw.ps f2, %[sin_cache]    \n\t"
                    "flw.ps f3, %[cos_cache]    \n\t"
                    "fmul.ps f4, f0, f3         \n\t"
                    "fmul.ps f5, f0, f2         \n\t"
                    "fnmsub.ps f4, f1, f2, f4   \n\t"
                    "fmadd.ps f5, f1, f3, f5    \n\t"
                    "fsw.ps f4, %[x0_dst]       \n\t"
                    "fsw.ps f5, %[x1_dst]       \n\t"
                    : [x0_dst] "=m"(*(ದಶಮಾಂಶ (*)[8]) & ತಲೆ_dst[dim_idx]), [x1_dst] "=m"(*(ದಶಮಾಂಶ (*)[8]) &
                                                                                        ತಲೆ_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & ತಲೆ_src[dim_idx]),
                      [x1_src] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & ತಲೆ_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ ಅಸ್ಥಿರ("mova.m.x %0" ::"r"(ತಾತ್ಕಾಲಿಕ_ಮುಖವಾಡ));
        } ಇಲ್ಲದಿದ್ದರೆ {
            // Standard: adjacent pairs (2i, 2i+1)
            ಗಾಗಿ (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                ಸ್ಥಿರಾಂಕ int32_t dim_ಒಳ_ತಲೆ = pair_idx * 2;
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   x0          = ತಲೆ_src[dim_ಒಳ_ತಲೆ];
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ   x1          = ತಲೆ_src[dim_ಒಳ_ತಲೆ + 1];

                ತಲೆ_dst[dim_ಒಳ_ತಲೆ]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                ತಲೆ_dst[dim_ಒಳ_ತಲೆ + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    ಹಿಂತಿರುಗಿ 0;
}
