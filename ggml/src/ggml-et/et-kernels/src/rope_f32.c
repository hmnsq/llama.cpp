//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#अन्तर्भाव "ggml_tensor.h"
#अन्तर्भाव "math_fp.h"
#अन्तर्भाव "platform.h"

#अन्तर्भाव <etsoc/common/utils.h>
#अन्तर्भाव <stdint.h>

// ROPE constants (matching GGML definitions)
#परिभाषय GGML_ROPE_TYPE_NEOX   2
#परिभाषय GGML_ROPE_TYPE_MROPE  8
#परिभाषय GGML_ROPE_TYPE_IMROPE 40
#परिभाषय MAX_ROPE_HALF_DIMS    256  // supports up to n_dims=512

#परिभाषय ROPE_VEC_विस्तारः 8

#परिभाषय ROPE_PI         3.14159265358979323846f
#परिभाषय ROPE_TWO_PI     6.28318530717958647693f
#परिभाषय ROPE_PI_OVER_2  1.57079632679489661923f
#परिभाषय ROPE_INV_TWO_PI 0.15915494309189533577f

// ROPE operation parameters structure (matches ggml-et-ops.h)
प्रकारनिर्देश संरचना {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t बहुलकम्;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    भिन्न   freq_आधारः;    // Base frequency (usually 10000.0f)
    भिन्न   freq_scale;   // Frequency scaling factor
    भिन्न   ext_गुणकः;   // Extension factor for YaRN
    भिन्न   attn_गुणकः;  // Attention factor for YaRN
    भिन्न   beta_fast;    // Fast beta for YaRN
    भिन्न   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
संरचना ggml_et_rope_params {
    संरचना ggml_tensor src0;  // F32 input tensor
    संरचना ggml_tensor src1;  // I32 position tensor
    संरचना ggml_tensor src2;  // F32 frequency factors (optional)
    संरचना ggml_tensor dst;   // F32 output tensor
    rope_params_t      rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

// floor/ceil with ±inf and NaN passthrough.
स्थिर अन्तःस्थ भिन्न rope_floorf(भिन्न x) {
    सङ्घ {
        भिन्न    f;
        uint32_t u;
    } v = { .f = x };

    ध्रुव uint32_t expo = (v.u >> 23) & 0xFF;
    यदि (expo == 0xFF) {
        प्रत्यागम x;  // inf or NaN
    }
    यदि (expo >= 23 + 127) {
        प्रत्यागम x;  // already integer-valued
    }
    पूर्णाङ्क i = (पूर्णाङ्क) x;
    प्रत्यागम (x < 0.0f && (भिन्न) i != x) ? (भिन्न) (i - 1) : (भिन्न) i;
}

स्थिर अन्तःस्थ भिन्न rope_ceilf(भिन्न x) {
    सङ्घ {
        भिन्न    f;
        uint32_t u;
    } v = { .f = x };

    ध्रुव uint32_t expo = (v.u >> 23) & 0xFF;
    यदि (expo == 0xFF) {
        प्रत्यागम x;  // inf or NaN
    }
    यदि (expo >= 23 + 127) {
        प्रत्यागम x;  // already integer-valued
    }
    पूर्णाङ्क i = (पूर्णाङ्क) x;
    प्रत्यागम (x > 0.0f && (भिन्न) i != x) ? (भिन्न) (i + 1) : (भिन्न) i;
}

स्थिर अन्तःस्थ भिन्न rope_yarn_ramp(ध्रुव भिन्न नीचः, ध्रुव भिन्न उच्चः, ध्रुव पूर्णाङ्क i0) {
    भिन्न denom = उच्चः - नीचः;
    यदि (denom < 0.001f) {
        denom = 0.001f;
    }

    ध्रुव भिन्न y       = et_fdiv((भिन्न) (i0 / 2) - नीचः, denom);
    ध्रुव भिन्न clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    प्रत्यागम 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
स्थिर अन्तःस्थ भिन्न rope_yarn_corr_dim(पूर्णाङ्क n_dims, पूर्णाङ्क n_ctx_orig, भिन्न beta, भिन्न freq_आधारः) {
    प्रत्यागम (भिन्न) n_dims *
           et_fdiv(et_logf(et_fdiv((भिन्न) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_आधारः));
}

स्थिर अन्तःस्थ शून्य rope_yarn_corr_dims(पूर्णाङ्क   n_dims,
                                       पूर्णाङ्क   n_ctx_orig,
                                       भिन्न freq_आधारः,
                                       भिन्न beta_fast,
                                       भिन्न beta_slow,
                                       भिन्न dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    भिन्न start = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_आधारः));
    भिन्न अन्तः_2   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_आधारः));

    dims[0] = start > 0.0f ? start : 0.0f;
    dims[1] = अन्तः_2 < (भिन्न) (n_dims - 1) ? अन्तः_2 : (भिन्न) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

स्थिर ध्रुव भिन्न rope_ps_one[ROPE_VEC_विस्तारः]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
स्थिर ध्रुव भिन्न rope_ps_c3[ROPE_VEC_विस्तारः]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
स्थिर ध्रुव भिन्न rope_ps_c5[ROPE_VEC_विस्तारः]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
स्थिर ध्रुव भिन्न rope_ps_c7[ROPE_VEC_विस्तारः]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
स्थिर ध्रुव भिन्न rope_ps_c9[ROPE_VEC_विस्तारः]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
स्थिर ध्रुव भिन्न rope_ps_c11[ROPE_VEC_विस्तारः]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

स्थिर अन्तःस्थ uint64_t rope_ps_enter_fullmask(शून्य) {
    uint64_t old_आवरणम्;
    __asm__ चञ्चल(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_आवरणम्)
        :
        : "t0", "memory");
    प्रत्यागम old_आवरणम्;
}

स्थिर अन्तःस्थ शून्य rope_ps_leave_fullmask(uint64_t old_आवरणम्) {
    __asm__ चञ्चल("mova.m.x %0           \n\t" : : "r"(old_आवरणम्) : "memory");
}

स्थिर अन्तःस्थ शून्य rope_poly_sin_block8(भिन्न * बहिर्गमनम्, ध्रुव भिन्न * x) {
    __asm__ चञ्चल(
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
        : [बहिर्गमनम्] "=m"(*(भिन्न (*)[ROPE_VEC_विस्तारः]) बहिर्गमनम्)
        : [x] "m"(*(ध्रुव भिन्न (*)[ROPE_VEC_विस्तारः]) x), [one] "m"(*(ध्रुव भिन्न (*)[ROPE_VEC_विस्तारः]) rope_ps_one),
          [c3] "m"(*(ध्रुव भिन्न (*)[ROPE_VEC_विस्तारः]) rope_ps_c3),
          [c5] "m"(*(ध्रुव भिन्न (*)[ROPE_VEC_विस्तारः]) rope_ps_c5),
          [c7] "m"(*(ध्रुव भिन्न (*)[ROPE_VEC_विस्तारः]) rope_ps_c7),
          [c9] "m"(*(ध्रुव भिन्न (*)[ROPE_VEC_विस्तारः]) rope_ps_c9),
          [c11] "m"(*(ध्रुव भिन्न (*)[ROPE_VEC_विस्तारः]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

स्थिर अन्तःस्थ शून्य rope_sincos_block8(भिन्न * sin8, भिन्न * cos8, ध्रुव भिन्न * theta8) {
    भिन्न sin_fold[ROPE_VEC_विस्तारः] __attribute__((aligned(32)));
    भिन्न cos_fold[ROPE_VEC_विस्तारः] __attribute__((aligned(32)));
    भिन्न sin_sign[ROPE_VEC_विस्तारः] __attribute__((aligned(32)));
    भिन्न cos_sign[ROPE_VEC_विस्तारः] __attribute__((aligned(32)));

    प्रति (पूर्णाङ्क i = 0; i < ROPE_VEC_विस्तारः; ++i) {
        भिन्न x = theta8[i];

        यदि (x > ROPE_PI || x < -ROPE_PI) {
            भिन्न आवृत्तयः = x * ROPE_INV_TWO_PI;
            पूर्णाङ्क   n      = (पूर्णाङ्क) आवृत्तयः;
            यदि (x < 0.0f) {
                n--;
            }
            x = x - (भिन्न) n * ROPE_TWO_PI;
        }

        {
            भिन्न y = x;
            भिन्न s = 1.0f;
            यदि (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } अन्यथा यदि (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            भिन्न y = x + ROPE_PI_OVER_2;
            यदि (y > ROPE_PI || y < -ROPE_PI) {
                भिन्न आवृत्तयः = y * ROPE_INV_TWO_PI;
                पूर्णाङ्क   n      = (पूर्णाङ्क) आवृत्तयः;
                यदि (y < 0.0f) {
                    n--;
                }
                y = y - (भिन्न) n * ROPE_TWO_PI;
            }

            भिन्न s = 1.0f;
            यदि (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } अन्यथा यदि (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            cos_fold[i] = y;
            cos_sign[i] = s;
        }
    }

    {
        ध्रुव uint64_t saved_आवरणम् = rope_ps_enter_fullmask();

        rope_poly_sin_block8(sin8, sin_fold);
        rope_poly_sin_block8(cos8, cos_fold);

        __asm__ चञ्चल(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(भिन्न (*)[ROPE_VEC_विस्तारः]) sin8), [cout] "=m"(*(भिन्न (*)[ROPE_VEC_विस्तारः]) cos8)
            : [sinv] "m"(*(ध्रुव भिन्न (*)[ROPE_VEC_विस्तारः]) sin8),
              [sinsgn] "m"(*(ध्रुव भिन्न (*)[ROPE_VEC_विस्तारः]) sin_sign),
              [cosv] "m"(*(ध्रुव भिन्न (*)[ROPE_VEC_विस्तारः]) cos8),
              [cossgn] "m"(*(ध्रुव भिन्न (*)[ROPE_VEC_विस्तारः]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_आवरणम्);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
स्थिर अन्तःस्थ शून्य rope_yarn_scalar(भिन्न       theta_extrap,
                                    भिन्न       freq_scale,
                                    ध्रुव भिन्न corr_dims[2],
                                    int64_t     i0,
                                    भिन्न       ext_गुणकः,
                                    भिन्न       mscale,
                                    भिन्न *     cos_theta,
                                    भिन्न *     sin_theta) {
    भिन्न theta_interp = freq_scale * theta_extrap;
    भिन्न theta        = theta_interp;

    यदि (ext_गुणकः != 0.0f) {
        भिन्न ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (पूर्णाङ्क) i0) * ext_गुणकः;
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
स्थिर अन्तःस्थ शून्य compute_rope_cache(भिन्न *       cos_cache,
                                      भिन्न *       sin_cache,
                                      int32_t       n_dims,
                                      भिन्न         theta_scale,
                                      int32_t       pos,
                                      ध्रुव भिन्न * freq_factors,
                                      भिन्न         freq_scale,
                                      ध्रुव भिन्न   corr_dims[2],
                                      भिन्न         ext_गुणकः,
                                      भिन्न         attn_गुणकः) {
    ध्रुव int32_t half_dims = n_dims / 2;
    भिन्न         theta     = 1.0f;

    int32_t dim_idx = 0;

    प्रति (; dim_idx + ROPE_VEC_विस्तारः <= half_dims; dim_idx += ROPE_VEC_विस्तारः) {
        भिन्न theta_खण्डम्[ROPE_VEC_विस्तारः] __attribute__((aligned(32)));
        भिन्न theta_local = theta;
        भिन्न mscale      = attn_गुणकः;

        यदि (ext_गुणकः != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        प्रति (पूर्णाङ्क i = 0; i < ROPE_VEC_विस्तारः; ++i) {
            ध्रुव int32_t pair_idx     = dim_idx + i;
            ध्रुव भिन्न   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            ध्रुव भिन्न   theta_आधारः   = (भिन्न) pos * theta_local;
            ध्रुव भिन्न   theta_extrap = et_fdiv(theta_आधारः, ff);

            भिन्न theta_interp = freq_scale * theta_extrap;
            भिन्न theta_mix    = theta_interp;

            यदि (ext_गुणकः != 0.0f) {
                भिन्न ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_गुणकः;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_खण्डम्[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_खण्डम्);

        प्रति (पूर्णाङ्क i = 0; i < ROPE_VEC_विस्तारः; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    प्रति (; dim_idx < half_dims; ++dim_idx) {
        ध्रुव भिन्न ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        ध्रुव भिन्न theta_आधारः = (भिन्न) pos * theta;

        rope_yarn_scalar(et_fdiv(theta_आधारः, ff), freq_scale, corr_dims, dim_idx * 2, ext_गुणकः, attn_गुणकः,
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
स्थिर अन्तःस्थ शून्य compute_imrope_cache(भिन्न *       cos_cache,
                                        भिन्न *       sin_cache,
                                        int32_t       n_dims,
                                        भिन्न         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        ध्रुव int32_t sections[4],
                                        ध्रुव भिन्न * freq_factors,
                                        भिन्न         freq_scale,
                                        ध्रुव भिन्न   corr_dims[2],
                                        भिन्न         ext_गुणकः,
                                        भिन्न         attn_गुणकः) {
    ध्रुव int32_t half_dims = n_dims / 2;
    ध्रुव int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    भिन्न theta_t = (भिन्न) pos_t;
    भिन्न theta_h = (भिन्न) pos_h;
    भिन्न theta_w = (भिन्न) pos_w;
    भिन्न theta_e = (भिन्न) pos_e;

    int32_t dim_idx = 0;

    प्रति (; dim_idx + ROPE_VEC_विस्तारः <= half_dims; dim_idx += ROPE_VEC_विस्तारः) {
        भिन्न theta_खण्डम्[ROPE_VEC_विस्तारः] __attribute__((aligned(32)));
        भिन्न mscale = attn_गुणकः;

        यदि (ext_गुणकः != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        प्रति (पूर्णाङ्क i = 0; i < ROPE_VEC_विस्तारः; ++i) {
            ध्रुव int32_t pair_idx = dim_idx + i;
            ध्रुव int32_t sector   = pair_idx % sect_dims;
            ध्रुव भिन्न   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            भिन्न theta;
            यदि (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } अन्यथा यदि (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } अन्यथा यदि (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } अन्यथा {
                theta = theta_e;
            }

            ध्रुव भिन्न theta_extrap = et_fdiv(theta, ff);
            भिन्न       theta_interp = freq_scale * theta_extrap;
            भिन्न       theta_mix    = theta_interp;

            यदि (ext_गुणकः != 0.0f) {
                भिन्न ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_गुणकः;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_खण्डम्[i] = theta_mix;

            // All 4 thetas advance every iteration
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_खण्डम्);

        प्रति (पूर्णाङ्क i = 0; i < ROPE_VEC_विस्तारः; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    प्रति (; dim_idx < half_dims; ++dim_idx) {
        ध्रुव int32_t sector = dim_idx % sect_dims;
        ध्रुव भिन्न   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        भिन्न theta;
        यदि (sector % 3 == 1 && sector < 3 * sections[1]) {
            theta = theta_h;
        } अन्यथा यदि (sector % 3 == 2 && sector < 3 * sections[2]) {
            theta = theta_w;
        } अन्यथा यदि (sector % 3 == 0 && sector < 3 * sections[0]) {
            theta = theta_t;
        } अन्यथा {
            theta = theta_e;
        }

        rope_yarn_scalar(et_fdiv(theta, ff), freq_scale, corr_dims, dim_idx * 2, ext_गुणकः, attn_गुणकः,
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

पूर्णाङ्क प्रविष्टिः_point(संरचना ggml_et_rope_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    यदि (!kernel_env) {
        प्रत्यागम -1;
    }

    पूर्णाङ्क thread_id   = get_relative_thread_id(kernel_env->shire_आवरणम्);
    पूर्णाङ्क अङ्कः_तन्तवः = get_अङ्कः_तन्तवः(kernel_env->shire_आवरणम्);

    यदि (thread_id < 0) {
        प्रत्यागम -1;
    }

    यदि (params == 0 || ((uint64_t) params & 0x7) != 0) {
        प्रत्यागम -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * src2 = &params->src2;
    संरचना ggml_tensor * dst  = &params->dst;

    यदि (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        प्रत्यागम -1;
    }

    ध्रुव भिन्न *   src0_दत्त    = (ध्रुव भिन्न *) src0->दत्त;
    ध्रुव int32_t * src1_दत्त    = (ध्रुव int32_t *) src1->दत्त;
    ध्रुव भिन्न *   freq_factors = (src2 && src2->दत्त) ? (ध्रुव भिन्न *) src2->दत्त : रिक्त;
    भिन्न *         dst_दत्त     = (भिन्न *) dst->दत्त;

    यदि (!src0_दत्त || !src1_दत्त || !dst_दत्त) {
        प्रत्यागम -1;
    }
#यदि_परिभाषितम् ET_UBERKERNEL
    ध्रुव size_t src0_अष्टकानि = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ध्रुव size_t src1_अष्टकानि = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_दत्त, src0_अष्टकानि);
    evict_region_past_l2(src1_दत्त, src1_अष्टकानि);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#यदि_अन्तः
    ध्रुव int64_t शीर्ष_dim = src0->ne[0];
    ध्रुव int64_t heads    = src0->ne[1];
    ध्रुव int64_t seq_len  = src0->ne[2];
    ध्रुव int64_t batch    = src0->ne[3];

    ध्रुव rope_params_t * rope_params = &params->rope_params;
    ध्रुव int32_t         n_dims      = rope_params->n_dims;
    ध्रुव भिन्न           freq_आधारः   = rope_params->freq_आधारः;
    ध्रुव भिन्न           freq_scale  = rope_params->freq_scale;
    ध्रुव int32_t         बहुलकम्        = rope_params->बहुलकम्;

    यदि (n_dims <= 0 || n_dims > शीर्ष_dim || (n_dims & 1) != 0) {
        प्रत्यागम -1;
    }

    यदि (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        प्रत्यागम -1;
    }

    भिन्न cos_cache[MAX_ROPE_HALF_DIMS];
    भिन्न sin_cache[MAX_ROPE_HALF_DIMS];

    भिन्न corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_आधारः, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    ध्रुव int64_t योग_heads = batch * seq_len * heads;
    ध्रुव int64_t start_wu    = (योग_heads * thread_id) / अङ्कः_तन्तवः;
    ध्रुव int64_t अन्तः_wu      = (योग_heads * (thread_id + 1)) / अङ्कः_तन्तवः;

    यदि (start_wu >= अन्तः_wu) {
        प्रत्यागम 0;
    }

    ध्रुव भिन्न   theta_scale       = et_powf(freq_आधारः, et_fdiv(-2.0f, (भिन्न) n_dims));
    ध्रुव int32_t half_dims         = n_dims / 2;
    ध्रुव पूर्णाङ्क     is_neox           = (बहुलकम् & GGML_ROPE_TYPE_NEOX) != 0;
    ध्रुव पूर्णाङ्क     is_imrope         = (बहुलकम् == GGML_ROPE_TYPE_IMROPE);
    ध्रुव पूर्णाङ्क     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t चरमः_pos   = -1;
    int32_t चरमः_pos_h = -1;
    int32_t चरमः_pos_w = -1;
    int32_t चरमः_pos_e = -1;

    प्रति (int64_t wu = start_wu; wu < अन्तः_wu; ++wu) {
        ध्रुव int64_t h = wu % heads;
        ध्रुव int64_t s = (wu / heads) % seq_len;
        ध्रुव int64_t b = wu / (heads * seq_len);

        यदि (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            ध्रुव int32_t pt = src1_दत्त[s] + rope_params->n_past;
            ध्रुव int32_t ph = src1_दत्त[s + seq_len] + rope_params->n_past;
            ध्रुव int32_t pw = src1_दत्त[s + seq_len * 2] + rope_params->n_past;
            ध्रुव int32_t pe = src1_दत्त[s + seq_len * 3] + rope_params->n_past;

            यदि (pt != चरमः_pos || ph != चरमः_pos_h || pw != चरमः_pos_w || pe != चरमः_pos_e) {
                compute_imrope_cache(cos_cache, sin_cache, n_dims, theta_scale, pt, ph, pw, pe, rope_params->sections,
                                     freq_factors, freq_scale, corr_dims, rope_params->ext_गुणकः,
                                     rope_params->attn_गुणकः);
                चरमः_pos   = pt;
                चरमः_pos_h = ph;
                चरमः_pos_w = pw;
                चरमः_pos_e = pe;
            }
        } अन्यथा {
            ध्रुव int32_t pos = src1_दत्त[s] + rope_params->n_past;

            यदि (pos != चरमः_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_गुणकः, rope_params->attn_गुणकः);
                चरमः_pos = pos;
            }
        }

        ध्रुव भिन्न * शीर्ष_src =
            (ध्रुव भिन्न *) ((ध्रुव वर्ण *) src0_दत्त + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        भिन्न * शीर्ष_dst = (भिन्न *) ((वर्ण *) dst_दत्त + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        प्रति (int64_t d = n_dims; d < शीर्ष_dim; ++d) {
            शीर्ष_dst[d] = शीर्ष_src[d];
        }

        यदि (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t अस्थायी_आवरणम्;
            __asm__ चञ्चल("mova.x.m %0" : "=r"(अस्थायी_आवरणम्));
            __asm__ चञ्चल("mov.m.x m0, x0, 0xFF");

            प्रति (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                __asm__ चञ्चल(
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
                    : [x0_dst] "=m"(*(भिन्न (*)[8]) & शीर्ष_dst[dim_idx]), [x1_dst] "=m"(*(भिन्न (*)[8]) &
                                                                                        शीर्ष_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(ध्रुव भिन्न (*)[8]) & शीर्ष_src[dim_idx]),
                      [x1_src] "m"(*(ध्रुव भिन्न (*)[8]) & शीर्ष_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(ध्रुव भिन्न (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(ध्रुव भिन्न (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ चञ्चल("mova.m.x %0" ::"r"(अस्थायी_आवरणम्));
        } अन्यथा {
            // Standard: adjacent pairs (2i, 2i+1)
            प्रति (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                ध्रुव int32_t dim_अन्तर्_शीर्ष = pair_idx * 2;
                ध्रुव भिन्न   x0          = शीर्ष_src[dim_अन्तर्_शीर्ष];
                ध्रुव भिन्न   x1          = शीर्ष_src[dim_अन्तर्_शीर्ष + 1];

                शीर्ष_dst[dim_अन्तर्_शीर्ष]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                शीर्ष_dst[dim_अन्तर्_शीर्ष + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    प्रत्यागम 0;
}
