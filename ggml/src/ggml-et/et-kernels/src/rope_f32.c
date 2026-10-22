//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#ଅନ୍ତର୍ଭାଵ "ggml_tensor.h"
#ଅନ୍ତର୍ଭାଵ "math_fp.h"
#ଅନ୍ତର୍ଭାଵ "platform.h"

#ଅନ୍ତର୍ଭାଵ <etsoc/common/utils.h>
#ଅନ୍ତର୍ଭାଵ <stdint.h>

// ROPE constants (matching GGML definitions)
#define GGML_ROPE_TYPE_NEOX   2
#define GGML_ROPE_TYPE_MROPE  8
#define GGML_ROPE_TYPE_IMROPE 40
#define MAX_ROPE_HALF_DIMS    256  // supports up to n_dims=512

#define ROPE_VEC_WIDTH 8

#define ROPE_PI         3.14159265358979323846f
#define ROPE_TWO_PI     6.28318530717958647693f
#define ROPE_PI_OVER_2  1.57079632679489661923f
#define ROPE_INV_TWO_PI 0.15915494309189533577f

// ROPE operation parameters structure (matches ggml-et-ops.h)
ପ୍ରକାରବ୍ୟାଖ୍ୟା ସଂରଚନା {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t mode;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    ଦଶମିକ   freq_base;    // Base frequency (usually 10000.0f)
    ଦଶମିକ   freq_scale;   // Frequency scaling factor
    ଦଶମିକ   ext_factor;   // Extension factor for YaRN
    ଦଶମିକ   attn_factor;  // Attention factor for YaRN
    ଦଶମିକ   beta_fast;    // Fast beta for YaRN
    ଦଶମିକ   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
ସଂରଚନା ggml_et_rope_params {
    ସଂରଚନା ggml_tensor src0;  // F32 input tensor
    ସଂରଚନା ggml_tensor src1;  // I32 position tensor
    ସଂରଚନା ggml_tensor src2;  // F32 frequency factors (optional)
    ସଂରଚନା ggml_tensor dst;   // F32 output tensor
    rope_params_t      rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

// floor/ceil with ±inf and NaN passthrough.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ rope_floorf(ଦଶମିକ x) {
    ସଂଘ {
        ଦଶମିକ    f;
        uint32_t u;
    } v = { .f = x };

    ଅଚଳ uint32_t expo = (v.u >> 23) & 0xFF;
    ଯଦି (expo == 0xFF) {
        ଫେରନ୍ତୁ x;  // inf or NaN
    }
    ଯଦି (expo >= 23 + 127) {
        ଫେରନ୍ତୁ x;  // already integer-valued
    }
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) x;
    ଫେରନ୍ତୁ (x < 0.0f && (ଦଶମିକ) i != x) ? (ଦଶମିକ) (i - 1) : (ଦଶମିକ) i;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ rope_ceilf(ଦଶମିକ x) {
    ସଂଘ {
        ଦଶମିକ    f;
        uint32_t u;
    } v = { .f = x };

    ଅଚଳ uint32_t expo = (v.u >> 23) & 0xFF;
    ଯଦି (expo == 0xFF) {
        ଫେରନ୍ତୁ x;  // inf or NaN
    }
    ଯଦି (expo >= 23 + 127) {
        ଫେରନ୍ତୁ x;  // already integer-valued
    }
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) x;
    ଫେରନ୍ତୁ (x > 0.0f && (ଦଶମିକ) i != x) ? (ଦଶମିକ) (i + 1) : (ଦଶମିକ) i;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ rope_yarn_ramp(ଅଚଳ ଦଶମିକ ତଳ, ଅଚଳ ଦଶମିକ ଉପର, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0) {
    ଦଶମିକ denom = ଉପର - ତଳ;
    ଯଦି (denom < 0.001f) {
        denom = 0.001f;
    }

    ଅଚଳ ଦଶମିକ y       = et_fdiv((ଦଶମିକ) (i0 / 2) - ତଳ, denom);
    ଅଚଳ ଦଶମିକ clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    ଫେରନ୍ତୁ 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ rope_yarn_corr_dim(ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_dims, ପୂର୍ଣ୍ଣସଂଖ୍ୟା n_ctx_orig, ଦଶମିକ beta, ଦଶମିକ freq_base) {
    ଫେରନ୍ତୁ (ଦଶମିକ) n_dims *
           et_fdiv(et_logf(et_fdiv((ଦଶମିକ) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_base));
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rope_yarn_corr_dims(ପୂର୍ଣ୍ଣସଂଖ୍ୟା   n_dims,
                                       ପୂର୍ଣ୍ଣସଂଖ୍ୟା   n_ctx_orig,
                                       ଦଶମିକ freq_base,
                                       ଦଶମିକ beta_fast,
                                       ଦଶମିକ beta_slow,
                                       ଦଶମିକ dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    ଦଶମିକ start = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    ଦଶମିକ end   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));

    dims[0] = start > 0.0f ? start : 0.0f;
    dims[1] = end < (ଦଶମିକ) (n_dims - 1) ? end : (ଦଶମିକ) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

ସ୍ଥିର ଅଚଳ ଦଶମିକ rope_ps_one[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
ସ୍ଥିର ଅଚଳ ଦଶମିକ rope_ps_c3[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
ସ୍ଥିର ଅଚଳ ଦଶମିକ rope_ps_c5[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
ସ୍ଥିର ଅଚଳ ଦଶମିକ rope_ps_c7[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
ସ୍ଥିର ଅଚଳ ଦଶମିକ rope_ps_c9[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
ସ୍ଥିର ଅଚଳ ଦଶମିକ rope_ps_c11[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint64_t rope_ps_enter_fullmask(ଶୂନ୍ୟ) {
    uint64_t old_mask;
    __asm__ ଅସ୍ଥିର(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_mask)
        :
        : "t0", "memory");
    ଫେରନ୍ତୁ old_mask;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rope_ps_leave_fullmask(uint64_t old_mask) {
    __asm__ ଅସ୍ଥିର("mova.m.x %0           \n\t" : : "r"(old_mask) : "memory");
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rope_poly_sin_block8(ଦଶମିକ * out, ଅଚଳ ଦଶମିକ * x) {
    __asm__ ଅସ୍ଥିର(
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
        : [out] "=m"(*(ଦଶମିକ (*)[ROPE_VEC_WIDTH]) out)
        : [x] "m"(*(ଅଚଳ ଦଶମିକ (*)[ROPE_VEC_WIDTH]) x), [one] "m"(*(ଅଚଳ ଦଶମିକ (*)[ROPE_VEC_WIDTH]) rope_ps_one),
          [c3] "m"(*(ଅଚଳ ଦଶମିକ (*)[ROPE_VEC_WIDTH]) rope_ps_c3),
          [c5] "m"(*(ଅଚଳ ଦଶମିକ (*)[ROPE_VEC_WIDTH]) rope_ps_c5),
          [c7] "m"(*(ଅଚଳ ଦଶମିକ (*)[ROPE_VEC_WIDTH]) rope_ps_c7),
          [c9] "m"(*(ଅଚଳ ଦଶମିକ (*)[ROPE_VEC_WIDTH]) rope_ps_c9),
          [c11] "m"(*(ଅଚଳ ଦଶମିକ (*)[ROPE_VEC_WIDTH]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rope_sincos_block8(ଦଶମିକ * sin8, ଦଶମିକ * cos8, ଅଚଳ ଦଶମିକ * theta8) {
    ଦଶମିକ sin_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    ଦଶମିକ cos_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    ଦଶମିକ sin_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    ଦଶମିକ cos_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ROPE_VEC_WIDTH; ++i) {
        ଦଶମିକ x = theta8[i];

        ଯଦି (x > ROPE_PI || x < -ROPE_PI) {
            ଦଶମିକ cycles = x * ROPE_INV_TWO_PI;
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା   n      = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) cycles;
            ଯଦି (x < 0.0f) {
                n--;
            }
            x = x - (ଦଶମିକ) n * ROPE_TWO_PI;
        }

        {
            ଦଶମିକ y = x;
            ଦଶମିକ s = 1.0f;
            ଯଦି (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } ଅନ୍ୟଥା ଯଦି (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            ଦଶମିକ y = x + ROPE_PI_OVER_2;
            ଯଦି (y > ROPE_PI || y < -ROPE_PI) {
                ଦଶମିକ cycles = y * ROPE_INV_TWO_PI;
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା   n      = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) cycles;
                ଯଦି (y < 0.0f) {
                    n--;
                }
                y = y - (ଦଶମିକ) n * ROPE_TWO_PI;
            }

            ଦଶମିକ s = 1.0f;
            ଯଦି (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } ଅନ୍ୟଥା ଯଦି (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            cos_fold[i] = y;
            cos_sign[i] = s;
        }
    }

    {
        ଅଚଳ uint64_t saved_mask = rope_ps_enter_fullmask();

        rope_poly_sin_block8(sin8, sin_fold);
        rope_poly_sin_block8(cos8, cos_fold);

        __asm__ ଅସ୍ଥିର(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(ଦଶମିକ (*)[ROPE_VEC_WIDTH]) sin8), [cout] "=m"(*(ଦଶମିକ (*)[ROPE_VEC_WIDTH]) cos8)
            : [sinv] "m"(*(ଅଚଳ ଦଶମିକ (*)[ROPE_VEC_WIDTH]) sin8),
              [sinsgn] "m"(*(ଅଚଳ ଦଶମିକ (*)[ROPE_VEC_WIDTH]) sin_sign),
              [cosv] "m"(*(ଅଚଳ ଦଶମିକ (*)[ROPE_VEC_WIDTH]) cos8),
              [cossgn] "m"(*(ଅଚଳ ଦଶମିକ (*)[ROPE_VEC_WIDTH]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_mask);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rope_yarn_scalar(ଦଶମିକ       theta_extrap,
                                    ଦଶମିକ       freq_scale,
                                    ଅଚଳ ଦଶମିକ corr_dims[2],
                                    int64_t     i0,
                                    ଦଶମିକ       ext_factor,
                                    ଦଶମିକ       mscale,
                                    ଦଶମିକ *     cos_theta,
                                    ଦଶମିକ *     sin_theta) {
    ଦଶମିକ theta_interp = freq_scale * theta_extrap;
    ଦଶମିକ theta        = theta_interp;

    ଯଦି (ext_factor != 0.0f) {
        ଦଶମିକ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) i0) * ext_factor;
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
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ compute_rope_cache(ଦଶମିକ *       cos_cache,
                                      ଦଶମିକ *       sin_cache,
                                      int32_t       n_dims,
                                      ଦଶମିକ         theta_scale,
                                      int32_t       pos,
                                      ଅଚଳ ଦଶମିକ * freq_factors,
                                      ଦଶମିକ         freq_scale,
                                      ଅଚଳ ଦଶମିକ   corr_dims[2],
                                      ଦଶମିକ         ext_factor,
                                      ଦଶମିକ         attn_factor) {
    ଅଚଳ int32_t half_dims = n_dims / 2;
    ଦଶମିକ         theta     = 1.0f;

    int32_t dim_idx = 0;

    ପାଇଁ (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        ଦଶମିକ theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        ଦଶମିକ theta_local = theta;
        ଦଶମିକ mscale      = attn_factor;

        ଯଦି (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ROPE_VEC_WIDTH; ++i) {
            ଅଚଳ int32_t pair_idx     = dim_idx + i;
            ଅଚଳ ଦଶମିକ   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            ଅଚଳ ଦଶମିକ   theta_base   = (ଦଶମିକ) pos * theta_local;
            ଅଚଳ ଦଶମିକ   theta_extrap = et_fdiv(theta_base, ff);

            ଦଶମିକ theta_interp = freq_scale * theta_extrap;
            ଦଶମିକ theta_mix    = theta_interp;

            ଯଦି (ext_factor != 0.0f) {
                ଦଶମିକ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_block[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_block);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    ପାଇଁ (; dim_idx < half_dims; ++dim_idx) {
        ଅଚଳ ଦଶମିକ ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        ଅଚଳ ଦଶମିକ theta_base = (ଦଶମିକ) pos * theta;

        rope_yarn_scalar(et_fdiv(theta_base, ff), freq_scale, corr_dims, dim_idx * 2, ext_factor, attn_factor,
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
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ compute_imrope_cache(ଦଶମିକ *       cos_cache,
                                        ଦଶମିକ *       sin_cache,
                                        int32_t       n_dims,
                                        ଦଶମିକ         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        ଅଚଳ int32_t sections[4],
                                        ଅଚଳ ଦଶମିକ * freq_factors,
                                        ଦଶମିକ         freq_scale,
                                        ଅଚଳ ଦଶମିକ   corr_dims[2],
                                        ଦଶମିକ         ext_factor,
                                        ଦଶମିକ         attn_factor) {
    ଅଚଳ int32_t half_dims = n_dims / 2;
    ଅଚଳ int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    ଦଶମିକ theta_t = (ଦଶମିକ) pos_t;
    ଦଶମିକ theta_h = (ଦଶମିକ) pos_h;
    ଦଶମିକ theta_w = (ଦଶମିକ) pos_w;
    ଦଶମିକ theta_e = (ଦଶମିକ) pos_e;

    int32_t dim_idx = 0;

    ପାଇଁ (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        ଦଶମିକ theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        ଦଶମିକ mscale = attn_factor;

        ଯଦି (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ROPE_VEC_WIDTH; ++i) {
            ଅଚଳ int32_t pair_idx = dim_idx + i;
            ଅଚଳ int32_t sector   = pair_idx % sect_dims;
            ଅଚଳ ଦଶମିକ   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            ଦଶମିକ theta;
            ଯଦି (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } ଅନ୍ୟଥା ଯଦି (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } ଅନ୍ୟଥା ଯଦି (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } ଅନ୍ୟଥା {
                theta = theta_e;
            }

            ଅଚଳ ଦଶମିକ theta_extrap = et_fdiv(theta, ff);
            ଦଶମିକ       theta_interp = freq_scale * theta_extrap;
            ଦଶମିକ       theta_mix    = theta_interp;

            ଯଦି (ext_factor != 0.0f) {
                ଦଶମିକ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_block[i] = theta_mix;

            // All 4 thetas advance every iteration
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_block);

        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    ପାଇଁ (; dim_idx < half_dims; ++dim_idx) {
        ଅଚଳ int32_t sector = dim_idx % sect_dims;
        ଅଚଳ ଦଶମିକ   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        ଦଶମିକ theta;
        ଯଦି (sector % 3 == 1 && sector < 3 * sections[1]) {
            theta = theta_h;
        } ଅନ୍ୟଥା ଯଦି (sector % 3 == 2 && sector < 3 * sections[2]) {
            theta = theta_w;
        } ଅନ୍ୟଥା ଯଦି (sector % 3 == 0 && sector < 3 * sections[0]) {
            theta = theta_t;
        } ଅନ୍ୟଥା {
            theta = theta_e;
        }

        rope_yarn_scalar(et_fdiv(theta, ff), freq_scale, corr_dims, dim_idx * 2, ext_factor, attn_factor,
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

ପୂର୍ଣ୍ଣସଂଖ୍ୟା entry_point(ସଂରଚନା ggml_et_rope_params * params, ଶୂନ୍ୟ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ଯଦି (!kernel_env) {
        ଫେରନ୍ତୁ -1;
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଅଙ୍କ_threads = get_ଅଙ୍କ_threads(kernel_env->shire_mask);

    ଯଦି (thread_id < 0) {
        ଫେରନ୍ତୁ -1;
    }

    ଯଦି (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ସଂରଚନା ggml_tensor * src0 = &params->src0;
    ସଂରଚନା ggml_tensor * src1 = &params->src1;
    ସଂରଚନା ggml_tensor * src2 = &params->src2;
    ସଂରଚନା ggml_tensor * dst  = &params->dst;

    ଯଦି (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        ଫେରନ୍ତୁ -1;
    }

    ଅଚଳ ଦଶମିକ *   src0_data    = (ଅଚଳ ଦଶମିକ *) src0->data;
    ଅଚଳ int32_t * src1_data    = (ଅଚଳ int32_t *) src1->data;
    ଅଚଳ ଦଶମିକ *   freq_factors = (src2 && src2->data) ? (ଅଚଳ ଦଶମିକ *) src2->data : ଖାଲି;
    ଦଶମିକ *         dst_data     = (ଦଶମିକ *) dst->data;

    ଯଦି (!src0_data || !src1_data || !dst_data) {
        ଫେରନ୍ତୁ -1;
    }
#ifdef ET_UBERKERNEL
    ଅଚଳ size_t src0_bytes = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ଅଚଳ size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_data, src0_bytes);
    evict_region_past_l2(src1_data, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    ଅଚଳ int64_t head_dim = src0->ne[0];
    ଅଚଳ int64_t heads    = src0->ne[1];
    ଅଚଳ int64_t seq_len  = src0->ne[2];
    ଅଚଳ int64_t batch    = src0->ne[3];

    ଅଚଳ rope_params_t * rope_params = &params->rope_params;
    ଅଚଳ int32_t         n_dims      = rope_params->n_dims;
    ଅଚଳ ଦଶମିକ           freq_base   = rope_params->freq_base;
    ଅଚଳ ଦଶମିକ           freq_scale  = rope_params->freq_scale;
    ଅଚଳ int32_t         mode        = rope_params->mode;

    ଯଦି (n_dims <= 0 || n_dims > head_dim || (n_dims & 1) != 0) {
        ଫେରନ୍ତୁ -1;
    }

    ଯଦି (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        ଫେରନ୍ତୁ -1;
    }

    ଦଶମିକ cos_cache[MAX_ROPE_HALF_DIMS];
    ଦଶମିକ sin_cache[MAX_ROPE_HALF_DIMS];

    ଦଶମିକ corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_base, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    ଅଚଳ int64_t total_heads = batch * seq_len * heads;
    ଅଚଳ int64_t start_wu    = (total_heads * thread_id) / ଅଙ୍କ_threads;
    ଅଚଳ int64_t end_wu      = (total_heads * (thread_id + 1)) / ଅଙ୍କ_threads;

    ଯଦି (start_wu >= end_wu) {
        ଫେରନ୍ତୁ 0;
    }

    ଅଚଳ ଦଶମିକ   theta_scale       = et_powf(freq_base, et_fdiv(-2.0f, (ଦଶମିକ) n_dims));
    ଅଚଳ int32_t half_dims         = n_dims / 2;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     is_neox           = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     is_imrope         = (mode == GGML_ROPE_TYPE_IMROPE);
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t last_pos   = -1;
    int32_t last_pos_h = -1;
    int32_t last_pos_w = -1;
    int32_t last_pos_e = -1;

    ପାଇଁ (int64_t wu = start_wu; wu < end_wu; ++wu) {
        ଅଚଳ int64_t h = wu % heads;
        ଅଚଳ int64_t s = (wu / heads) % seq_len;
        ଅଚଳ int64_t b = wu / (heads * seq_len);

        ଯଦି (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            ଅଚଳ int32_t pt = src1_data[s] + rope_params->n_past;
            ଅଚଳ int32_t ph = src1_data[s + seq_len] + rope_params->n_past;
            ଅଚଳ int32_t pw = src1_data[s + seq_len * 2] + rope_params->n_past;
            ଅଚଳ int32_t pe = src1_data[s + seq_len * 3] + rope_params->n_past;

            ଯଦି (pt != last_pos || ph != last_pos_h || pw != last_pos_w || pe != last_pos_e) {
                compute_imrope_cache(cos_cache, sin_cache, n_dims, theta_scale, pt, ph, pw, pe, rope_params->sections,
                                     freq_factors, freq_scale, corr_dims, rope_params->ext_factor,
                                     rope_params->attn_factor);
                last_pos   = pt;
                last_pos_h = ph;
                last_pos_w = pw;
                last_pos_e = pe;
            }
        } ଅନ୍ୟଥା {
            ଅଚଳ int32_t pos = src1_data[s] + rope_params->n_past;

            ଯଦି (pos != last_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_factor, rope_params->attn_factor);
                last_pos = pos;
            }
        }

        ଅଚଳ ଦଶମିକ * head_src =
            (ଅଚଳ ଦଶମିକ *) ((ଅଚଳ ଅକ୍ଷର *) src0_data + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        ଦଶମିକ * head_dst = (ଦଶମିକ *) ((ଅକ୍ଷର *) dst_data + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        ପାଇଁ (int64_t d = n_dims; d < head_dim; ++d) {
            head_dst[d] = head_src[d];
        }

        ଯଦି (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t temp_mask;
            __asm__ ଅସ୍ଥିର("mova.x.m %0" : "=r"(temp_mask));
            __asm__ ଅସ୍ଥିର("mov.m.x m0, x0, 0xFF");

            ପାଇଁ (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                __asm__ ଅସ୍ଥିର(
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
                    : [x0_dst] "=m"(*(ଦଶମିକ (*)[8]) & head_dst[dim_idx]), [x1_dst] "=m"(*(ଦଶମିକ (*)[8]) &
                                                                                        head_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & head_src[dim_idx]),
                      [x1_src] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & head_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(ଅଚଳ ଦଶମିକ (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ ଅସ୍ଥିର("mova.m.x %0" ::"r"(temp_mask));
        } ଅନ୍ୟଥା {
            // Standard: adjacent pairs (2i, 2i+1)
            ପାଇଁ (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                ଅଚଳ int32_t dim_in_head = pair_idx * 2;
                ଅଚଳ ଦଶମିକ   x0          = head_src[dim_in_head];
                ଅଚଳ ଦଶମିକ   x1          = head_src[dim_in_head + 1];

                head_dst[dim_in_head]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                head_dst[dim_in_head + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    ଫେରନ୍ତୁ 0;
}
