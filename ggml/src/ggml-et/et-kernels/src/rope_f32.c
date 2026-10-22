//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#ਅਨ੍ਤਰ੍ਭਾਵ "ggml_tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "math_fp.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "platform.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <etsoc/common/utils.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>

// ROPE constants (matching GGML definitions)
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_ROPE_TYPE_NEOX   2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_ROPE_TYPE_MROPE  8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_ROPE_TYPE_IMROPE 40
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MAX_ROPE_HALF_DIMS    256  // supports up to n_dims=512

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ROPE_VEC_ਚੌੜਾਈ 8

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ROPE_PI         3.14159265358979323846f
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ROPE_TWO_PI     6.28318530717958647693f
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ROPE_PI_OVER_2  1.57079632679489661923f
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ ROPE_INV_TWO_PI 0.15915494309189533577f

// ROPE operation parameters structure (matches ggml-et-ops.h)
ਕਿਸਮਪਰਿਭਾਸ਼ਾ ਬਣਤਰ {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t ਬਹੁਲਕ;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    ਦਸ਼ਮਲਵ   freq_ਅਧਾਰ;    // Base frequency (usually 10000.0f)
    ਦਸ਼ਮਲਵ   freq_scale;   // Frequency scaling factor
    ਦਸ਼ਮਲਵ   ext_ਗੁਣਕ;   // Extension factor for YaRN
    ਦਸ਼ਮਲਵ   attn_ਗੁਣਕ;  // Attention factor for YaRN
    ਦਸ਼ਮਲਵ   beta_fast;    // Fast beta for YaRN
    ਦਸ਼ਮਲਵ   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
ਬਣਤਰ ggml_et_rope_params {
    ਬਣਤਰ ggml_tensor src0;  // F32 input tensor
    ਬਣਤਰ ggml_tensor src1;  // I32 position tensor
    ਬਣਤਰ ggml_tensor src2;  // F32 frequency factors (optional)
    ਬਣਤਰ ggml_tensor dst;   // F32 output tensor
    rope_params_t      rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

// floor/ceil with ±inf and NaN passthrough.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ rope_floorf(ਦਸ਼ਮਲਵ x) {
    ਸੰਘ {
        ਦਸ਼ਮਲਵ    f;
        uint32_t u;
    } v = { .f = x };

    ਅਚਲ uint32_t expo = (v.u >> 23) & 0xFF;
    ਜੇ (expo == 0xFF) {
        ਵਾਪਸ x;  // inf or NaN
    }
    ਜੇ (expo >= 23 + 127) {
        ਵਾਪਸ x;  // already integer-valued
    }
    ਪੂਰਨ i = (ਪੂਰਨ) x;
    ਵਾਪਸ (x < 0.0f && (ਦਸ਼ਮਲਵ) i != x) ? (ਦਸ਼ਮਲਵ) (i - 1) : (ਦਸ਼ਮਲਵ) i;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ rope_ceilf(ਦਸ਼ਮਲਵ x) {
    ਸੰਘ {
        ਦਸ਼ਮਲਵ    f;
        uint32_t u;
    } v = { .f = x };

    ਅਚਲ uint32_t expo = (v.u >> 23) & 0xFF;
    ਜੇ (expo == 0xFF) {
        ਵਾਪਸ x;  // inf or NaN
    }
    ਜੇ (expo >= 23 + 127) {
        ਵਾਪਸ x;  // already integer-valued
    }
    ਪੂਰਨ i = (ਪੂਰਨ) x;
    ਵਾਪਸ (x > 0.0f && (ਦਸ਼ਮਲਵ) i != x) ? (ਦਸ਼ਮਲਵ) (i + 1) : (ਦਸ਼ਮਲਵ) i;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ rope_yarn_ramp(ਅਚਲ ਦਸ਼ਮਲਵ ਨੀਵਾਂ, ਅਚਲ ਦਸ਼ਮਲਵ ਉੱਚਾ, ਅਚਲ ਪੂਰਨ i0) {
    ਦਸ਼ਮਲਵ denom = ਉੱਚਾ - ਨੀਵਾਂ;
    ਜੇ (denom < 0.001f) {
        denom = 0.001f;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ y       = et_fdiv((ਦਸ਼ਮਲਵ) (i0 / 2) - ਨੀਵਾਂ, denom);
    ਅਚਲ ਦਸ਼ਮਲਵ clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    ਵਾਪਸ 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ rope_yarn_corr_dim(ਪੂਰਨ n_dims, ਪੂਰਨ n_ctx_orig, ਦਸ਼ਮਲਵ beta, ਦਸ਼ਮਲਵ freq_ਅਧਾਰ) {
    ਵਾਪਸ (ਦਸ਼ਮਲਵ) n_dims *
           et_fdiv(et_logf(et_fdiv((ਦਸ਼ਮਲਵ) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_ਅਧਾਰ));
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rope_yarn_corr_dims(ਪੂਰਨ   n_dims,
                                       ਪੂਰਨ   n_ctx_orig,
                                       ਦਸ਼ਮਲਵ freq_ਅਧਾਰ,
                                       ਦਸ਼ਮਲਵ beta_fast,
                                       ਦਸ਼ਮਲਵ beta_slow,
                                       ਦਸ਼ਮਲਵ dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    ਦਸ਼ਮਲਵ ਸ਼ੁਰੂ = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_ਅਧਾਰ));
    ਦਸ਼ਮਲਵ ਸਿਰਾ   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_ਅਧਾਰ));

    dims[0] = ਸ਼ੁਰੂ > 0.0f ? ਸ਼ੁਰੂ : 0.0f;
    dims[1] = ਸਿਰਾ < (ਦਸ਼ਮਲਵ) (n_dims - 1) ? ਸਿਰਾ : (ਦਸ਼ਮਲਵ) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ rope_ps_one[ROPE_VEC_ਚੌੜਾਈ]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ rope_ps_c3[ROPE_VEC_ਚੌੜਾਈ]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ rope_ps_c5[ROPE_VEC_ਚੌੜਾਈ]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ rope_ps_c7[ROPE_VEC_ਚੌੜਾਈ]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ rope_ps_c9[ROPE_VEC_ਚੌੜਾਈ]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ rope_ps_c11[ROPE_VEC_ਚੌੜਾਈ]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

ਸਥਿਰ ਸਤਰਅੰਦਰ uint64_t rope_ps_enter_fullmask(ਖਾਲੀ) {
    uint64_t old_ਨਕਾਬ;
    __asm__ ਅਸਥਿਰ(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_ਨਕਾਬ)
        :
        : "t0", "memory");
    ਵਾਪਸ old_ਨਕਾਬ;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rope_ps_leave_fullmask(uint64_t old_ਨਕਾਬ) {
    __asm__ ਅਸਥਿਰ("mova.m.x %0           \n\t" : : "r"(old_ਨਕਾਬ) : "memory");
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rope_poly_sin_block8(ਦਸ਼ਮਲਵ * ਬਾਹਰਲਾ, ਅਚਲ ਦਸ਼ਮਲਵ * x) {
    __asm__ ਅਸਥਿਰ(
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
        : [ਬਾਹਰਲਾ] "=m"(*(ਦਸ਼ਮਲਵ (*)[ROPE_VEC_ਚੌੜਾਈ]) ਬਾਹਰਲਾ)
        : [x] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[ROPE_VEC_ਚੌੜਾਈ]) x), [one] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[ROPE_VEC_ਚੌੜਾਈ]) rope_ps_one),
          [c3] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[ROPE_VEC_ਚੌੜਾਈ]) rope_ps_c3),
          [c5] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[ROPE_VEC_ਚੌੜਾਈ]) rope_ps_c5),
          [c7] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[ROPE_VEC_ਚੌੜਾਈ]) rope_ps_c7),
          [c9] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[ROPE_VEC_ਚੌੜਾਈ]) rope_ps_c9),
          [c11] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[ROPE_VEC_ਚੌੜਾਈ]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rope_sincos_block8(ਦਸ਼ਮਲਵ * sin8, ਦਸ਼ਮਲਵ * cos8, ਅਚਲ ਦਸ਼ਮਲਵ * theta8) {
    ਦਸ਼ਮਲਵ sin_fold[ROPE_VEC_ਚੌੜਾਈ] __attribute__((aligned(32)));
    ਦਸ਼ਮਲਵ cos_fold[ROPE_VEC_ਚੌੜਾਈ] __attribute__((aligned(32)));
    ਦਸ਼ਮਲਵ sin_sign[ROPE_VEC_ਚੌੜਾਈ] __attribute__((aligned(32)));
    ਦਸ਼ਮਲਵ cos_sign[ROPE_VEC_ਚੌੜਾਈ] __attribute__((aligned(32)));

    ਲਈ (ਪੂਰਨ i = 0; i < ROPE_VEC_ਚੌੜਾਈ; ++i) {
        ਦਸ਼ਮਲਵ x = theta8[i];

        ਜੇ (x > ROPE_PI || x < -ROPE_PI) {
            ਦਸ਼ਮਲਵ ਚੱਕਰਾਂ = x * ROPE_INV_TWO_PI;
            ਪੂਰਨ   n      = (ਪੂਰਨ) ਚੱਕਰਾਂ;
            ਜੇ (x < 0.0f) {
                n--;
            }
            x = x - (ਦਸ਼ਮਲਵ) n * ROPE_TWO_PI;
        }

        {
            ਦਸ਼ਮਲਵ y = x;
            ਦਸ਼ਮਲਵ s = 1.0f;
            ਜੇ (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } ਨਹੀਂਤਾਂ ਜੇ (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            ਦਸ਼ਮਲਵ y = x + ROPE_PI_OVER_2;
            ਜੇ (y > ROPE_PI || y < -ROPE_PI) {
                ਦਸ਼ਮਲਵ ਚੱਕਰਾਂ = y * ROPE_INV_TWO_PI;
                ਪੂਰਨ   n      = (ਪੂਰਨ) ਚੱਕਰਾਂ;
                ਜੇ (y < 0.0f) {
                    n--;
                }
                y = y - (ਦਸ਼ਮਲਵ) n * ROPE_TWO_PI;
            }

            ਦਸ਼ਮਲਵ s = 1.0f;
            ਜੇ (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } ਨਹੀਂਤਾਂ ਜੇ (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            cos_fold[i] = y;
            cos_sign[i] = s;
        }
    }

    {
        ਅਚਲ uint64_t saved_ਨਕਾਬ = rope_ps_enter_fullmask();

        rope_poly_sin_block8(sin8, sin_fold);
        rope_poly_sin_block8(cos8, cos_fold);

        __asm__ ਅਸਥਿਰ(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(ਦਸ਼ਮਲਵ (*)[ROPE_VEC_ਚੌੜਾਈ]) sin8), [cout] "=m"(*(ਦਸ਼ਮਲਵ (*)[ROPE_VEC_ਚੌੜਾਈ]) cos8)
            : [sinv] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[ROPE_VEC_ਚੌੜਾਈ]) sin8),
              [sinsgn] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[ROPE_VEC_ਚੌੜਾਈ]) sin_sign),
              [cosv] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[ROPE_VEC_ਚੌੜਾਈ]) cos8),
              [cossgn] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[ROPE_VEC_ਚੌੜਾਈ]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_ਨਕਾਬ);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rope_yarn_scalar(ਦਸ਼ਮਲਵ       theta_extrap,
                                    ਦਸ਼ਮਲਵ       freq_scale,
                                    ਅਚਲ ਦਸ਼ਮਲਵ corr_dims[2],
                                    int64_t     i0,
                                    ਦਸ਼ਮਲਵ       ext_ਗੁਣਕ,
                                    ਦਸ਼ਮਲਵ       mscale,
                                    ਦਸ਼ਮਲਵ *     cos_theta,
                                    ਦਸ਼ਮਲਵ *     sin_theta) {
    ਦਸ਼ਮਲਵ theta_interp = freq_scale * theta_extrap;
    ਦਸ਼ਮਲਵ theta        = theta_interp;

    ਜੇ (ext_ਗੁਣਕ != 0.0f) {
        ਦਸ਼ਮਲਵ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (ਪੂਰਨ) i0) * ext_ਗੁਣਕ;
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
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ compute_rope_cache(ਦਸ਼ਮਲਵ *       cos_cache,
                                      ਦਸ਼ਮਲਵ *       sin_cache,
                                      int32_t       n_dims,
                                      ਦਸ਼ਮਲਵ         theta_scale,
                                      int32_t       pos,
                                      ਅਚਲ ਦਸ਼ਮਲਵ * freq_factors,
                                      ਦਸ਼ਮਲਵ         freq_scale,
                                      ਅਚਲ ਦਸ਼ਮਲਵ   corr_dims[2],
                                      ਦਸ਼ਮਲਵ         ext_ਗੁਣਕ,
                                      ਦਸ਼ਮਲਵ         attn_ਗੁਣਕ) {
    ਅਚਲ int32_t half_dims = n_dims / 2;
    ਦਸ਼ਮਲਵ         theta     = 1.0f;

    int32_t dim_idx = 0;

    ਲਈ (; dim_idx + ROPE_VEC_ਚੌੜਾਈ <= half_dims; dim_idx += ROPE_VEC_ਚੌੜਾਈ) {
        ਦਸ਼ਮਲਵ theta_ਖੰਡ[ROPE_VEC_ਚੌੜਾਈ] __attribute__((aligned(32)));
        ਦਸ਼ਮਲਵ theta_local = theta;
        ਦਸ਼ਮਲਵ mscale      = attn_ਗੁਣਕ;

        ਜੇ (ext_ਗੁਣਕ != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        ਲਈ (ਪੂਰਨ i = 0; i < ROPE_VEC_ਚੌੜਾਈ; ++i) {
            ਅਚਲ int32_t pair_idx     = dim_idx + i;
            ਅਚਲ ਦਸ਼ਮਲਵ   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            ਅਚਲ ਦਸ਼ਮਲਵ   theta_ਅਧਾਰ   = (ਦਸ਼ਮਲਵ) pos * theta_local;
            ਅਚਲ ਦਸ਼ਮਲਵ   theta_extrap = et_fdiv(theta_ਅਧਾਰ, ff);

            ਦਸ਼ਮਲਵ theta_interp = freq_scale * theta_extrap;
            ਦਸ਼ਮਲਵ theta_mix    = theta_interp;

            ਜੇ (ext_ਗੁਣਕ != 0.0f) {
                ਦਸ਼ਮਲਵ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_ਗੁਣਕ;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_ਖੰਡ[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_ਖੰਡ);

        ਲਈ (ਪੂਰਨ i = 0; i < ROPE_VEC_ਚੌੜਾਈ; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    ਲਈ (; dim_idx < half_dims; ++dim_idx) {
        ਅਚਲ ਦਸ਼ਮਲਵ ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        ਅਚਲ ਦਸ਼ਮਲਵ theta_ਅਧਾਰ = (ਦਸ਼ਮਲਵ) pos * theta;

        rope_yarn_scalar(et_fdiv(theta_ਅਧਾਰ, ff), freq_scale, corr_dims, dim_idx * 2, ext_ਗੁਣਕ, attn_ਗੁਣਕ,
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
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ compute_imrope_cache(ਦਸ਼ਮਲਵ *       cos_cache,
                                        ਦਸ਼ਮਲਵ *       sin_cache,
                                        int32_t       n_dims,
                                        ਦਸ਼ਮਲਵ         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        ਅਚਲ int32_t sections[4],
                                        ਅਚਲ ਦਸ਼ਮਲਵ * freq_factors,
                                        ਦਸ਼ਮਲਵ         freq_scale,
                                        ਅਚਲ ਦਸ਼ਮਲਵ   corr_dims[2],
                                        ਦਸ਼ਮਲਵ         ext_ਗੁਣਕ,
                                        ਦਸ਼ਮਲਵ         attn_ਗੁਣਕ) {
    ਅਚਲ int32_t half_dims = n_dims / 2;
    ਅਚਲ int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    ਦਸ਼ਮਲਵ theta_t = (ਦਸ਼ਮਲਵ) pos_t;
    ਦਸ਼ਮਲਵ theta_h = (ਦਸ਼ਮਲਵ) pos_h;
    ਦਸ਼ਮਲਵ theta_w = (ਦਸ਼ਮਲਵ) pos_w;
    ਦਸ਼ਮਲਵ theta_e = (ਦਸ਼ਮਲਵ) pos_e;

    int32_t dim_idx = 0;

    ਲਈ (; dim_idx + ROPE_VEC_ਚੌੜਾਈ <= half_dims; dim_idx += ROPE_VEC_ਚੌੜਾਈ) {
        ਦਸ਼ਮਲਵ theta_ਖੰਡ[ROPE_VEC_ਚੌੜਾਈ] __attribute__((aligned(32)));
        ਦਸ਼ਮਲਵ mscale = attn_ਗੁਣਕ;

        ਜੇ (ext_ਗੁਣਕ != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        ਲਈ (ਪੂਰਨ i = 0; i < ROPE_VEC_ਚੌੜਾਈ; ++i) {
            ਅਚਲ int32_t pair_idx = dim_idx + i;
            ਅਚਲ int32_t sector   = pair_idx % sect_dims;
            ਅਚਲ ਦਸ਼ਮਲਵ   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            ਦਸ਼ਮਲਵ theta;
            ਜੇ (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } ਨਹੀਂਤਾਂ ਜੇ (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } ਨਹੀਂਤਾਂ ਜੇ (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } ਨਹੀਂਤਾਂ {
                theta = theta_e;
            }

            ਅਚਲ ਦਸ਼ਮਲਵ theta_extrap = et_fdiv(theta, ff);
            ਦਸ਼ਮਲਵ       theta_interp = freq_scale * theta_extrap;
            ਦਸ਼ਮਲਵ       theta_mix    = theta_interp;

            ਜੇ (ext_ਗੁਣਕ != 0.0f) {
                ਦਸ਼ਮਲਵ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_ਗੁਣਕ;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_ਖੰਡ[i] = theta_mix;

            // All 4 thetas advance every iteration
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_ਖੰਡ);

        ਲਈ (ਪੂਰਨ i = 0; i < ROPE_VEC_ਚੌੜਾਈ; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    ਲਈ (; dim_idx < half_dims; ++dim_idx) {
        ਅਚਲ int32_t sector = dim_idx % sect_dims;
        ਅਚਲ ਦਸ਼ਮਲਵ   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        ਦਸ਼ਮਲਵ theta;
        ਜੇ (sector % 3 == 1 && sector < 3 * sections[1]) {
            theta = theta_h;
        } ਨਹੀਂਤਾਂ ਜੇ (sector % 3 == 2 && sector < 3 * sections[2]) {
            theta = theta_w;
        } ਨਹੀਂਤਾਂ ਜੇ (sector % 3 == 0 && sector < 3 * sections[0]) {
            theta = theta_t;
        } ਨਹੀਂਤਾਂ {
            theta = theta_e;
        }

        rope_yarn_scalar(et_fdiv(theta, ff), freq_scale, corr_dims, dim_idx * 2, ext_ਗੁਣਕ, attn_ਗੁਣਕ,
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

ਪੂਰਨ ਇੰਦਰਾਜ_ਬਿੰਦੂ(ਬਣਤਰ ggml_et_rope_params * params, ਖਾਲੀ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ਜੇ (!kernel_env) {
        ਵਾਪਸ -1;
    }

    ਪੂਰਨ thread_id   = get_relative_thread_id(kernel_env->shire_ਨਕਾਬ);
    ਪੂਰਨ ਅੰਕ_ਧਾਗੇ = get_ਅੰਕ_ਧਾਗੇ(kernel_env->shire_ਨਕਾਬ);

    ਜੇ (thread_id < 0) {
        ਵਾਪਸ -1;
    }

    ਜੇ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ਵਾਪਸ -1;
    }

    ਬਣਤਰ ggml_tensor * src0 = &params->src0;
    ਬਣਤਰ ggml_tensor * src1 = &params->src1;
    ਬਣਤਰ ggml_tensor * src2 = &params->src2;
    ਬਣਤਰ ggml_tensor * dst  = &params->dst;

    ਜੇ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        ਵਾਪਸ -1;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ *   src0_ਅੰਕੜੇ    = (ਅਚਲ ਦਸ਼ਮਲਵ *) src0->ਅੰਕੜੇ;
    ਅਚਲ int32_t * src1_ਅੰਕੜੇ    = (ਅਚਲ int32_t *) src1->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ *   freq_factors = (src2 && src2->ਅੰਕੜੇ) ? (ਅਚਲ ਦਸ਼ਮਲਵ *) src2->ਅੰਕੜੇ : ਨਿਰਾ;
    ਦਸ਼ਮਲਵ *         dst_ਅੰਕੜੇ     = (ਦਸ਼ਮਲਵ *) dst->ਅੰਕੜੇ;

    ਜੇ (!src0_ਅੰਕੜੇ || !src1_ਅੰਕੜੇ || !dst_ਅੰਕੜੇ) {
        ਵਾਪਸ -1;
    }
#ਜੇ_ਪਰਿਭਾਸ਼ਿਤ ET_UBERKERNEL
    ਅਚਲ size_t src0_ਬਾਈਟਾਂ = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ਅਚਲ size_t src1_ਬਾਈਟਾਂ = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_ਅੰਕੜੇ, src0_ਬਾਈਟਾਂ);
    evict_region_past_l2(src1_ਅੰਕੜੇ, src1_ਬਾਈਟਾਂ);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#ਜੇ_ਅੰਤ
    ਅਚਲ int64_t ਸਿਰ_dim = src0->ne[0];
    ਅਚਲ int64_t heads    = src0->ne[1];
    ਅਚਲ int64_t seq_len  = src0->ne[2];
    ਅਚਲ int64_t batch    = src0->ne[3];

    ਅਚਲ rope_params_t * rope_params = &params->rope_params;
    ਅਚਲ int32_t         n_dims      = rope_params->n_dims;
    ਅਚਲ ਦਸ਼ਮਲਵ           freq_ਅਧਾਰ   = rope_params->freq_ਅਧਾਰ;
    ਅਚਲ ਦਸ਼ਮਲਵ           freq_scale  = rope_params->freq_scale;
    ਅਚਲ int32_t         ਬਹੁਲਕ        = rope_params->ਬਹੁਲਕ;

    ਜੇ (n_dims <= 0 || n_dims > ਸਿਰ_dim || (n_dims & 1) != 0) {
        ਵਾਪਸ -1;
    }

    ਜੇ (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        ਵਾਪਸ -1;
    }

    ਦਸ਼ਮਲਵ cos_cache[MAX_ROPE_HALF_DIMS];
    ਦਸ਼ਮਲਵ sin_cache[MAX_ROPE_HALF_DIMS];

    ਦਸ਼ਮਲਵ corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_ਅਧਾਰ, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    ਅਚਲ int64_t ਕੁੱਲ_heads = batch * seq_len * heads;
    ਅਚਲ int64_t ਸ਼ੁਰੂ_wu    = (ਕੁੱਲ_heads * thread_id) / ਅੰਕ_ਧਾਗੇ;
    ਅਚਲ int64_t ਸਿਰਾ_wu      = (ਕੁੱਲ_heads * (thread_id + 1)) / ਅੰਕ_ਧਾਗੇ;

    ਜੇ (ਸ਼ੁਰੂ_wu >= ਸਿਰਾ_wu) {
        ਵਾਪਸ 0;
    }

    ਅਚਲ ਦਸ਼ਮਲਵ   theta_scale       = et_powf(freq_ਅਧਾਰ, et_fdiv(-2.0f, (ਦਸ਼ਮਲਵ) n_dims));
    ਅਚਲ int32_t half_dims         = n_dims / 2;
    ਅਚਲ ਪੂਰਨ     is_neox           = (ਬਹੁਲਕ & GGML_ROPE_TYPE_NEOX) != 0;
    ਅਚਲ ਪੂਰਨ     is_imrope         = (ਬਹੁਲਕ == GGML_ROPE_TYPE_IMROPE);
    ਅਚਲ ਪੂਰਨ     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t ਆਖਰੀ_pos   = -1;
    int32_t ਆਖਰੀ_pos_h = -1;
    int32_t ਆਖਰੀ_pos_w = -1;
    int32_t ਆਖਰੀ_pos_e = -1;

    ਲਈ (int64_t wu = ਸ਼ੁਰੂ_wu; wu < ਸਿਰਾ_wu; ++wu) {
        ਅਚਲ int64_t h = wu % heads;
        ਅਚਲ int64_t s = (wu / heads) % seq_len;
        ਅਚਲ int64_t b = wu / (heads * seq_len);

        ਜੇ (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            ਅਚਲ int32_t pt = src1_ਅੰਕੜੇ[s] + rope_params->n_past;
            ਅਚਲ int32_t ph = src1_ਅੰਕੜੇ[s + seq_len] + rope_params->n_past;
            ਅਚਲ int32_t pw = src1_ਅੰਕੜੇ[s + seq_len * 2] + rope_params->n_past;
            ਅਚਲ int32_t pe = src1_ਅੰਕੜੇ[s + seq_len * 3] + rope_params->n_past;

            ਜੇ (pt != ਆਖਰੀ_pos || ph != ਆਖਰੀ_pos_h || pw != ਆਖਰੀ_pos_w || pe != ਆਖਰੀ_pos_e) {
                compute_imrope_cache(cos_cache, sin_cache, n_dims, theta_scale, pt, ph, pw, pe, rope_params->sections,
                                     freq_factors, freq_scale, corr_dims, rope_params->ext_ਗੁਣਕ,
                                     rope_params->attn_ਗੁਣਕ);
                ਆਖਰੀ_pos   = pt;
                ਆਖਰੀ_pos_h = ph;
                ਆਖਰੀ_pos_w = pw;
                ਆਖਰੀ_pos_e = pe;
            }
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ int32_t pos = src1_ਅੰਕੜੇ[s] + rope_params->n_past;

            ਜੇ (pos != ਆਖਰੀ_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_ਗੁਣਕ, rope_params->attn_ਗੁਣਕ);
                ਆਖਰੀ_pos = pos;
            }
        }

        ਅਚਲ ਦਸ਼ਮਲਵ * ਸਿਰ_src =
            (ਅਚਲ ਦਸ਼ਮਲਵ *) ((ਅਚਲ ਅੱਖਰ *) src0_ਅੰਕੜੇ + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        ਦਸ਼ਮਲਵ * ਸਿਰ_dst = (ਦਸ਼ਮਲਵ *) ((ਅੱਖਰ *) dst_ਅੰਕੜੇ + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        ਲਈ (int64_t d = n_dims; d < ਸਿਰ_dim; ++d) {
            ਸਿਰ_dst[d] = ਸਿਰ_src[d];
        }

        ਜੇ (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t ਆਰਜ਼ੀ_ਨਕਾਬ;
            __asm__ ਅਸਥਿਰ("mova.x.m %0" : "=r"(ਆਰਜ਼ੀ_ਨਕਾਬ));
            __asm__ ਅਸਥਿਰ("mov.m.x m0, x0, 0xFF");

            ਲਈ (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                __asm__ ਅਸਥਿਰ(
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
                    : [x0_dst] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) & ਸਿਰ_dst[dim_idx]), [x1_dst] "=m"(*(ਦਸ਼ਮਲਵ (*)[8]) &
                                                                                        ਸਿਰ_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & ਸਿਰ_src[dim_idx]),
                      [x1_src] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & ਸਿਰ_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(ਅਚਲ ਦਸ਼ਮਲਵ (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ ਅਸਥਿਰ("mova.m.x %0" ::"r"(ਆਰਜ਼ੀ_ਨਕਾਬ));
        } ਨਹੀਂਤਾਂ {
            // Standard: adjacent pairs (2i, 2i+1)
            ਲਈ (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                ਅਚਲ int32_t dim_ਵਿੱਚ_ਸਿਰ = pair_idx * 2;
                ਅਚਲ ਦਸ਼ਮਲਵ   x0          = ਸਿਰ_src[dim_ਵਿੱਚ_ਸਿਰ];
                ਅਚਲ ਦਸ਼ਮਲਵ   x1          = ਸਿਰ_src[dim_ਵਿੱਚ_ਸਿਰ + 1];

                ਸਿਰ_dst[dim_ਵਿੱਚ_ਸਿਰ]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                ਸਿਰ_dst[dim_ਵਿੱਚ_ਸਿਰ + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    ਵਾਪਸ 0;
}
