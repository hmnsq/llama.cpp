//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#समावेश "ggml_tensor.h"
#समावेश "math_fp.h"
#समावेश "platform.h"

#समावेश <etsoc/common/utils.h>
#समावेश <stdint.h>

// ROPE constants (matching GGML definitions)
#घोषणा GGML_ROPE_TYPE_NEOX   2
#घोषणा GGML_ROPE_TYPE_MROPE  8
#घोषणा GGML_ROPE_TYPE_IMROPE 40
#घोषणा MAX_ROPE_HALF_DIMS    256  // supports up to n_dims=512

#घोषणा ROPE_VEC_WIDTH 8

#घोषणा ROPE_PI         3.14159265358979323846f
#घोषणा ROPE_TWO_PI     6.28318530717958647693f
#घोषणा ROPE_PI_OVER_2  1.57079632679489661923f
#घोषणा ROPE_INV_TWO_PI 0.15915494309189533577f

// ROPE operation parameters structure (matches ggml-et-ops.h)
प्रकार काष्ठा {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t mode;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    भग्न   freq_base;    // Base frequency (usually 10000.0f)
    भग्न   freq_scale;   // Frequency scaling factor
    भग्न   ext_factor;   // Extension factor for YaRN
    भग्न   attn_factor;  // Attention factor for YaRN
    भग्न   beta_fast;    // Fast beta for YaRN
    भग्न   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
काष्ठा ggml_et_rope_params {
    काष्ठा ggml_tensor src0;  // F32 input tensor
    काष्ठा ggml_tensor src1;  // I32 position tensor
    काष्ठा ggml_tensor src2;  // F32 frequency factors (optional)
    काष्ठा ggml_tensor dst;   // F32 output tensor
    rope_params_t      rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

// floor/ceil with ±inf and NaN passthrough.
अटल अंतरभूत भग्न rope_floorf(भग्न x) {
    जोड़ {
        भग्न    f;
        uint32_t u;
    } v = { .f = x };

    स्थिर uint32_t expo = (v.u >> 23) & 0xFF;
    अगर (expo == 0xFF) {
        वापस x;  // inf or NaN
    }
    अगर (expo >= 23 + 127) {
        वापस x;  // already integer-valued
    }
    पूर्णांक i = (पूर्णांक) x;
    वापस (x < 0.0f && (भग्न) i != x) ? (भग्न) (i - 1) : (भग्न) i;
}

अटल अंतरभूत भग्न rope_ceilf(भग्न x) {
    जोड़ {
        भग्न    f;
        uint32_t u;
    } v = { .f = x };

    स्थिर uint32_t expo = (v.u >> 23) & 0xFF;
    अगर (expo == 0xFF) {
        वापस x;  // inf or NaN
    }
    अगर (expo >= 23 + 127) {
        वापस x;  // already integer-valued
    }
    पूर्णांक i = (पूर्णांक) x;
    वापस (x > 0.0f && (भग्न) i != x) ? (भग्न) (i + 1) : (भग्न) i;
}

अटल अंतरभूत भग्न rope_yarn_ramp(स्थिर भग्न निचला, स्थिर भग्न ऊपरी, स्थिर पूर्णांक i0) {
    भग्न denom = ऊपरी - निचला;
    अगर (denom < 0.001f) {
        denom = 0.001f;
    }

    स्थिर भग्न y       = et_fdiv((भग्न) (i0 / 2) - निचला, denom);
    स्थिर भग्न clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    वापस 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
अटल अंतरभूत भग्न rope_yarn_corr_dim(पूर्णांक n_dims, पूर्णांक n_ctx_orig, भग्न beta, भग्न freq_base) {
    वापस (भग्न) n_dims *
           et_fdiv(et_logf(et_fdiv((भग्न) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_base));
}

अटल अंतरभूत व्योम rope_yarn_corr_dims(पूर्णांक   n_dims,
                                       पूर्णांक   n_ctx_orig,
                                       भग्न freq_base,
                                       भग्न beta_fast,
                                       भग्न beta_slow,
                                       भग्न dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    भग्न start = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    भग्न end   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));

    dims[0] = start > 0.0f ? start : 0.0f;
    dims[1] = end < (भग्न) (n_dims - 1) ? end : (भग्न) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

अटल स्थिर भग्न rope_ps_one[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
अटल स्थिर भग्न rope_ps_c3[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
अटल स्थिर भग्न rope_ps_c5[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
अटल स्थिर भग्न rope_ps_c7[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
अटल स्थिर भग्न rope_ps_c9[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
अटल स्थिर भग्न rope_ps_c11[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

अटल अंतरभूत uint64_t rope_ps_enter_fullmask(व्योम) {
    uint64_t old_mask;
    __asm__ अस्थिर(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_mask)
        :
        : "t0", "memory");
    वापस old_mask;
}

अटल अंतरभूत व्योम rope_ps_leave_fullmask(uint64_t old_mask) {
    __asm__ अस्थिर("mova.m.x %0           \n\t" : : "r"(old_mask) : "memory");
}

अटल अंतरभूत व्योम rope_poly_sin_block8(भग्न * out, स्थिर भग्न * x) {
    __asm__ अस्थिर(
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
        : [out] "=m"(*(भग्न (*)[ROPE_VEC_WIDTH]) out)
        : [x] "m"(*(स्थिर भग्न (*)[ROPE_VEC_WIDTH]) x), [one] "m"(*(स्थिर भग्न (*)[ROPE_VEC_WIDTH]) rope_ps_one),
          [c3] "m"(*(स्थिर भग्न (*)[ROPE_VEC_WIDTH]) rope_ps_c3),
          [c5] "m"(*(स्थिर भग्न (*)[ROPE_VEC_WIDTH]) rope_ps_c5),
          [c7] "m"(*(स्थिर भग्न (*)[ROPE_VEC_WIDTH]) rope_ps_c7),
          [c9] "m"(*(स्थिर भग्न (*)[ROPE_VEC_WIDTH]) rope_ps_c9),
          [c11] "m"(*(स्थिर भग्न (*)[ROPE_VEC_WIDTH]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

अटल अंतरभूत व्योम rope_sincos_block8(भग्न * sin8, भग्न * cos8, स्थिर भग्न * theta8) {
    भग्न sin_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    भग्न cos_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    भग्न sin_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    भग्न cos_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));

    क्रम (पूर्णांक i = 0; i < ROPE_VEC_WIDTH; ++i) {
        भग्न x = theta8[i];

        अगर (x > ROPE_PI || x < -ROPE_PI) {
            भग्न cycles = x * ROPE_INV_TWO_PI;
            पूर्णांक   n      = (पूर्णांक) cycles;
            अगर (x < 0.0f) {
                n--;
            }
            x = x - (भग्न) n * ROPE_TWO_PI;
        }

        {
            भग्न y = x;
            भग्न s = 1.0f;
            अगर (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } अन्यथा अगर (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            भग्न y = x + ROPE_PI_OVER_2;
            अगर (y > ROPE_PI || y < -ROPE_PI) {
                भग्न cycles = y * ROPE_INV_TWO_PI;
                पूर्णांक   n      = (पूर्णांक) cycles;
                अगर (y < 0.0f) {
                    n--;
                }
                y = y - (भग्न) n * ROPE_TWO_PI;
            }

            भग्न s = 1.0f;
            अगर (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } अन्यथा अगर (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            cos_fold[i] = y;
            cos_sign[i] = s;
        }
    }

    {
        स्थिर uint64_t saved_mask = rope_ps_enter_fullmask();

        rope_poly_sin_block8(sin8, sin_fold);
        rope_poly_sin_block8(cos8, cos_fold);

        __asm__ अस्थिर(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(भग्न (*)[ROPE_VEC_WIDTH]) sin8), [cout] "=m"(*(भग्न (*)[ROPE_VEC_WIDTH]) cos8)
            : [sinv] "m"(*(स्थिर भग्न (*)[ROPE_VEC_WIDTH]) sin8),
              [sinsgn] "m"(*(स्थिर भग्न (*)[ROPE_VEC_WIDTH]) sin_sign),
              [cosv] "m"(*(स्थिर भग्न (*)[ROPE_VEC_WIDTH]) cos8),
              [cossgn] "m"(*(स्थिर भग्न (*)[ROPE_VEC_WIDTH]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_mask);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
अटल अंतरभूत व्योम rope_yarn_scalar(भग्न       theta_extrap,
                                    भग्न       freq_scale,
                                    स्थिर भग्न corr_dims[2],
                                    int64_t     i0,
                                    भग्न       ext_factor,
                                    भग्न       mscale,
                                    भग्न *     cos_theta,
                                    भग्न *     sin_theta) {
    भग्न theta_interp = freq_scale * theta_extrap;
    भग्न theta        = theta_interp;

    अगर (ext_factor != 0.0f) {
        भग्न ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (पूर्णांक) i0) * ext_factor;
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
अटल अंतरभूत व्योम compute_rope_cache(भग्न *       cos_cache,
                                      भग्न *       sin_cache,
                                      int32_t       n_dims,
                                      भग्न         theta_scale,
                                      int32_t       pos,
                                      स्थिर भग्न * freq_factors,
                                      भग्न         freq_scale,
                                      स्थिर भग्न   corr_dims[2],
                                      भग्न         ext_factor,
                                      भग्न         attn_factor) {
    स्थिर int32_t half_dims = n_dims / 2;
    भग्न         theta     = 1.0f;

    int32_t dim_idx = 0;

    क्रम (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        भग्न theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        भग्न theta_local = theta;
        भग्न mscale      = attn_factor;

        अगर (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        क्रम (पूर्णांक i = 0; i < ROPE_VEC_WIDTH; ++i) {
            स्थिर int32_t pair_idx     = dim_idx + i;
            स्थिर भग्न   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            स्थिर भग्न   theta_base   = (भग्न) pos * theta_local;
            स्थिर भग्न   theta_extrap = et_fdiv(theta_base, ff);

            भग्न theta_interp = freq_scale * theta_extrap;
            भग्न theta_mix    = theta_interp;

            अगर (ext_factor != 0.0f) {
                भग्न ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_block[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_block);

        क्रम (पूर्णांक i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    क्रम (; dim_idx < half_dims; ++dim_idx) {
        स्थिर भग्न ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        स्थिर भग्न theta_base = (भग्न) pos * theta;

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
अटल अंतरभूत व्योम compute_imrope_cache(भग्न *       cos_cache,
                                        भग्न *       sin_cache,
                                        int32_t       n_dims,
                                        भग्न         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        स्थिर int32_t sections[4],
                                        स्थिर भग्न * freq_factors,
                                        भग्न         freq_scale,
                                        स्थिर भग्न   corr_dims[2],
                                        भग्न         ext_factor,
                                        भग्न         attn_factor) {
    स्थिर int32_t half_dims = n_dims / 2;
    स्थिर int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    भग्न theta_t = (भग्न) pos_t;
    भग्न theta_h = (भग्न) pos_h;
    भग्न theta_w = (भग्न) pos_w;
    भग्न theta_e = (भग्न) pos_e;

    int32_t dim_idx = 0;

    क्रम (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        भग्न theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        भग्न mscale = attn_factor;

        अगर (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        क्रम (पूर्णांक i = 0; i < ROPE_VEC_WIDTH; ++i) {
            स्थिर int32_t pair_idx = dim_idx + i;
            स्थिर int32_t sector   = pair_idx % sect_dims;
            स्थिर भग्न   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            भग्न theta;
            अगर (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } अन्यथा अगर (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } अन्यथा अगर (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } अन्यथा {
                theta = theta_e;
            }

            स्थिर भग्न theta_extrap = et_fdiv(theta, ff);
            भग्न       theta_interp = freq_scale * theta_extrap;
            भग्न       theta_mix    = theta_interp;

            अगर (ext_factor != 0.0f) {
                भग्न ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
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

        क्रम (पूर्णांक i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    क्रम (; dim_idx < half_dims; ++dim_idx) {
        स्थिर int32_t sector = dim_idx % sect_dims;
        स्थिर भग्न   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        भग्न theta;
        अगर (sector % 3 == 1 && sector < 3 * sections[1]) {
            theta = theta_h;
        } अन्यथा अगर (sector % 3 == 2 && sector < 3 * sections[2]) {
            theta = theta_w;
        } अन्यथा अगर (sector % 3 == 0 && sector < 3 * sections[0]) {
            theta = theta_t;
        } अन्यथा {
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

पूर्णांक entry_point(काष्ठा ggml_et_rope_params * params, व्योम * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    अगर (!kernel_env) {
        वापस -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    अगर (thread_id < 0) {
        वापस -1;
    }

    अगर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        वापस -1;
    }

    काष्ठा ggml_tensor * src0 = &params->src0;
    काष्ठा ggml_tensor * src1 = &params->src1;
    काष्ठा ggml_tensor * src2 = &params->src2;
    काष्ठा ggml_tensor * dst  = &params->dst;

    अगर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        वापस -1;
    }

    स्थिर भग्न *   src0_तथ्य    = (स्थिर भग्न *) src0->तथ्य;
    स्थिर int32_t * src1_तथ्य    = (स्थिर int32_t *) src1->तथ्य;
    स्थिर भग्न *   freq_factors = (src2 && src2->तथ्य) ? (स्थिर भग्न *) src2->तथ्य : शून्य;
    भग्न *         dst_तथ्य     = (भग्न *) dst->तथ्य;

    अगर (!src0_तथ्य || !src1_तथ्य || !dst_तथ्य) {
        वापस -1;
    }
#अगर_घोषित ET_UBERKERNEL
    स्थिर माप_प्रकार src0_bytes = (माप_प्रकार) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    स्थिर माप_प्रकार src1_bytes = (माप_प्रकार) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_तथ्य, src0_bytes);
    evict_region_past_l2(src1_तथ्य, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#पूर्ण_अगर
    स्थिर int64_t शीर्ष_dim = src0->ne[0];
    स्थिर int64_t heads    = src0->ne[1];
    स्थिर int64_t seq_len  = src0->ne[2];
    स्थिर int64_t batch    = src0->ne[3];

    स्थिर rope_params_t * rope_params = &params->rope_params;
    स्थिर int32_t         n_dims      = rope_params->n_dims;
    स्थिर भग्न           freq_base   = rope_params->freq_base;
    स्थिर भग्न           freq_scale  = rope_params->freq_scale;
    स्थिर int32_t         mode        = rope_params->mode;

    अगर (n_dims <= 0 || n_dims > शीर्ष_dim || (n_dims & 1) != 0) {
        वापस -1;
    }

    अगर (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        वापस -1;
    }

    भग्न cos_cache[MAX_ROPE_HALF_DIMS];
    भग्न sin_cache[MAX_ROPE_HALF_DIMS];

    भग्न corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_base, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    स्थिर int64_t योग_heads = batch * seq_len * heads;
    स्थिर int64_t start_wu    = (योग_heads * thread_id) / अंक_threads;
    स्थिर int64_t end_wu      = (योग_heads * (thread_id + 1)) / अंक_threads;

    अगर (start_wu >= end_wu) {
        वापस 0;
    }

    स्थिर भग्न   theta_scale       = et_powf(freq_base, et_fdiv(-2.0f, (भग्न) n_dims));
    स्थिर int32_t half_dims         = n_dims / 2;
    स्थिर पूर्णांक     is_neox           = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    स्थिर पूर्णांक     is_imrope         = (mode == GGML_ROPE_TYPE_IMROPE);
    स्थिर पूर्णांक     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t last_pos   = -1;
    int32_t last_pos_h = -1;
    int32_t last_pos_w = -1;
    int32_t last_pos_e = -1;

    क्रम (int64_t wu = start_wu; wu < end_wu; ++wu) {
        स्थिर int64_t h = wu % heads;
        स्थिर int64_t s = (wu / heads) % seq_len;
        स्थिर int64_t b = wu / (heads * seq_len);

        अगर (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            स्थिर int32_t pt = src1_तथ्य[s] + rope_params->n_past;
            स्थिर int32_t ph = src1_तथ्य[s + seq_len] + rope_params->n_past;
            स्थिर int32_t pw = src1_तथ्य[s + seq_len * 2] + rope_params->n_past;
            स्थिर int32_t pe = src1_तथ्य[s + seq_len * 3] + rope_params->n_past;

            अगर (pt != last_pos || ph != last_pos_h || pw != last_pos_w || pe != last_pos_e) {
                compute_imrope_cache(cos_cache, sin_cache, n_dims, theta_scale, pt, ph, pw, pe, rope_params->sections,
                                     freq_factors, freq_scale, corr_dims, rope_params->ext_factor,
                                     rope_params->attn_factor);
                last_pos   = pt;
                last_pos_h = ph;
                last_pos_w = pw;
                last_pos_e = pe;
            }
        } अन्यथा {
            स्थिर int32_t pos = src1_तथ्य[s] + rope_params->n_past;

            अगर (pos != last_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_factor, rope_params->attn_factor);
                last_pos = pos;
            }
        }

        स्थिर भग्न * शीर्ष_src =
            (स्थिर भग्न *) ((स्थिर अक्षर *) src0_तथ्य + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        भग्न * शीर्ष_dst = (भग्न *) ((अक्षर *) dst_तथ्य + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        क्रम (int64_t d = n_dims; d < शीर्ष_dim; ++d) {
            शीर्ष_dst[d] = शीर्ष_src[d];
        }

        अगर (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t अस्थायी_mask;
            __asm__ अस्थिर("mova.x.m %0" : "=r"(अस्थायी_mask));
            __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");

            क्रम (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                __asm__ अस्थिर(
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
                    : [x0_dst] "=m"(*(भग्न (*)[8]) & शीर्ष_dst[dim_idx]), [x1_dst] "=m"(*(भग्न (*)[8]) &
                                                                                        शीर्ष_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(स्थिर भग्न (*)[8]) & शीर्ष_src[dim_idx]),
                      [x1_src] "m"(*(स्थिर भग्न (*)[8]) & शीर्ष_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(स्थिर भग्न (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(स्थिर भग्न (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ अस्थिर("mova.m.x %0" ::"r"(अस्थायी_mask));
        } अन्यथा {
            // Standard: adjacent pairs (2i, 2i+1)
            क्रम (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                स्थिर int32_t dim_in_शीर्ष = pair_idx * 2;
                स्थिर भग्न   x0          = शीर्ष_src[dim_in_शीर्ष];
                स्थिर भग्न   x1          = शीर्ष_src[dim_in_शीर्ष + 1];

                शीर्ष_dst[dim_in_शीर्ष]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                शीर्ष_dst[dim_in_शीर्ष + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    वापस 0;
}
