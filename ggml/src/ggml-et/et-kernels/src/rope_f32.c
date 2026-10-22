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
प्रकारव्याख्या संरचना {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t mode;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    दशांश   freq_base;    // Base frequency (usually 10000.0f)
    दशांश   freq_scale;   // Frequency scaling factor
    दशांश   ext_factor;   // Extension factor for YaRN
    दशांश   attn_factor;  // Attention factor for YaRN
    दशांश   beta_fast;    // Fast beta for YaRN
    दशांश   beta_slow;    // Slow beta for YaRN
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
स्थिर ओळीत दशांश rope_floorf(दशांश x) {
    संघ {
        दशांश    f;
        uint32_t u;
    } v = { .f = x };

    स्थिरांक uint32_t expo = (v.u >> 23) & 0xFF;
    जर (expo == 0xFF) {
        परत x;  // inf or NaN
    }
    जर (expo >= 23 + 127) {
        परत x;  // already integer-valued
    }
    पूर्णांक i = (पूर्णांक) x;
    परत (x < 0.0f && (दशांश) i != x) ? (दशांश) (i - 1) : (दशांश) i;
}

स्थिर ओळीत दशांश rope_ceilf(दशांश x) {
    संघ {
        दशांश    f;
        uint32_t u;
    } v = { .f = x };

    स्थिरांक uint32_t expo = (v.u >> 23) & 0xFF;
    जर (expo == 0xFF) {
        परत x;  // inf or NaN
    }
    जर (expo >= 23 + 127) {
        परत x;  // already integer-valued
    }
    पूर्णांक i = (पूर्णांक) x;
    परत (x > 0.0f && (दशांश) i != x) ? (दशांश) (i + 1) : (दशांश) i;
}

स्थिर ओळीत दशांश rope_yarn_ramp(स्थिरांक दशांश खालचा, स्थिरांक दशांश वरचा, स्थिरांक पूर्णांक i0) {
    दशांश denom = वरचा - खालचा;
    जर (denom < 0.001f) {
        denom = 0.001f;
    }

    स्थिरांक दशांश y       = et_fdiv((दशांश) (i0 / 2) - खालचा, denom);
    स्थिरांक दशांश clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    परत 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
स्थिर ओळीत दशांश rope_yarn_corr_dim(पूर्णांक n_dims, पूर्णांक n_ctx_orig, दशांश beta, दशांश freq_base) {
    परत (दशांश) n_dims *
           et_fdiv(et_logf(et_fdiv((दशांश) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_base));
}

स्थिर ओळीत शून्य rope_yarn_corr_dims(पूर्णांक   n_dims,
                                       पूर्णांक   n_ctx_orig,
                                       दशांश freq_base,
                                       दशांश beta_fast,
                                       दशांश beta_slow,
                                       दशांश dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    दशांश start = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    दशांश end   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));

    dims[0] = start > 0.0f ? start : 0.0f;
    dims[1] = end < (दशांश) (n_dims - 1) ? end : (दशांश) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

स्थिर स्थिरांक दशांश rope_ps_one[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
स्थिर स्थिरांक दशांश rope_ps_c3[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
स्थिर स्थिरांक दशांश rope_ps_c5[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
स्थिर स्थिरांक दशांश rope_ps_c7[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
स्थिर स्थिरांक दशांश rope_ps_c9[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
स्थिर स्थिरांक दशांश rope_ps_c11[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

स्थिर ओळीत uint64_t rope_ps_enter_fullmask(शून्य) {
    uint64_t old_mask;
    __asm__ अस्थिर(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_mask)
        :
        : "t0", "memory");
    परत old_mask;
}

स्थिर ओळीत शून्य rope_ps_leave_fullmask(uint64_t old_mask) {
    __asm__ अस्थिर("mova.m.x %0           \n\t" : : "r"(old_mask) : "memory");
}

स्थिर ओळीत शून्य rope_poly_sin_block8(दशांश * out, स्थिरांक दशांश * x) {
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
        : [out] "=m"(*(दशांश (*)[ROPE_VEC_WIDTH]) out)
        : [x] "m"(*(स्थिरांक दशांश (*)[ROPE_VEC_WIDTH]) x), [one] "m"(*(स्थिरांक दशांश (*)[ROPE_VEC_WIDTH]) rope_ps_one),
          [c3] "m"(*(स्थिरांक दशांश (*)[ROPE_VEC_WIDTH]) rope_ps_c3),
          [c5] "m"(*(स्थिरांक दशांश (*)[ROPE_VEC_WIDTH]) rope_ps_c5),
          [c7] "m"(*(स्थिरांक दशांश (*)[ROPE_VEC_WIDTH]) rope_ps_c7),
          [c9] "m"(*(स्थिरांक दशांश (*)[ROPE_VEC_WIDTH]) rope_ps_c9),
          [c11] "m"(*(स्थिरांक दशांश (*)[ROPE_VEC_WIDTH]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

स्थिर ओळीत शून्य rope_sincos_block8(दशांश * sin8, दशांश * cos8, स्थिरांक दशांश * theta8) {
    दशांश sin_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    दशांश cos_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    दशांश sin_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    दशांश cos_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));

    साठी (पूर्णांक i = 0; i < ROPE_VEC_WIDTH; ++i) {
        दशांश x = theta8[i];

        जर (x > ROPE_PI || x < -ROPE_PI) {
            दशांश cycles = x * ROPE_INV_TWO_PI;
            पूर्णांक   n      = (पूर्णांक) cycles;
            जर (x < 0.0f) {
                n--;
            }
            x = x - (दशांश) n * ROPE_TWO_PI;
        }

        {
            दशांश y = x;
            दशांश s = 1.0f;
            जर (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } अन्यथा जर (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            दशांश y = x + ROPE_PI_OVER_2;
            जर (y > ROPE_PI || y < -ROPE_PI) {
                दशांश cycles = y * ROPE_INV_TWO_PI;
                पूर्णांक   n      = (पूर्णांक) cycles;
                जर (y < 0.0f) {
                    n--;
                }
                y = y - (दशांश) n * ROPE_TWO_PI;
            }

            दशांश s = 1.0f;
            जर (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } अन्यथा जर (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            cos_fold[i] = y;
            cos_sign[i] = s;
        }
    }

    {
        स्थिरांक uint64_t saved_mask = rope_ps_enter_fullmask();

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
            : [sout] "=m"(*(दशांश (*)[ROPE_VEC_WIDTH]) sin8), [cout] "=m"(*(दशांश (*)[ROPE_VEC_WIDTH]) cos8)
            : [sinv] "m"(*(स्थिरांक दशांश (*)[ROPE_VEC_WIDTH]) sin8),
              [sinsgn] "m"(*(स्थिरांक दशांश (*)[ROPE_VEC_WIDTH]) sin_sign),
              [cosv] "m"(*(स्थिरांक दशांश (*)[ROPE_VEC_WIDTH]) cos8),
              [cossgn] "m"(*(स्थिरांक दशांश (*)[ROPE_VEC_WIDTH]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_mask);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
स्थिर ओळीत शून्य rope_yarn_scalar(दशांश       theta_extrap,
                                    दशांश       freq_scale,
                                    स्थिरांक दशांश corr_dims[2],
                                    int64_t     i0,
                                    दशांश       ext_factor,
                                    दशांश       mscale,
                                    दशांश *     cos_theta,
                                    दशांश *     sin_theta) {
    दशांश theta_interp = freq_scale * theta_extrap;
    दशांश theta        = theta_interp;

    जर (ext_factor != 0.0f) {
        दशांश ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (पूर्णांक) i0) * ext_factor;
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
स्थिर ओळीत शून्य compute_rope_cache(दशांश *       cos_cache,
                                      दशांश *       sin_cache,
                                      int32_t       n_dims,
                                      दशांश         theta_scale,
                                      int32_t       pos,
                                      स्थिरांक दशांश * freq_factors,
                                      दशांश         freq_scale,
                                      स्थिरांक दशांश   corr_dims[2],
                                      दशांश         ext_factor,
                                      दशांश         attn_factor) {
    स्थिरांक int32_t half_dims = n_dims / 2;
    दशांश         theta     = 1.0f;

    int32_t dim_idx = 0;

    साठी (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        दशांश theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        दशांश theta_local = theta;
        दशांश mscale      = attn_factor;

        जर (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        साठी (पूर्णांक i = 0; i < ROPE_VEC_WIDTH; ++i) {
            स्थिरांक int32_t pair_idx     = dim_idx + i;
            स्थिरांक दशांश   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            स्थिरांक दशांश   theta_base   = (दशांश) pos * theta_local;
            स्थिरांक दशांश   theta_extrap = et_fdiv(theta_base, ff);

            दशांश theta_interp = freq_scale * theta_extrap;
            दशांश theta_mix    = theta_interp;

            जर (ext_factor != 0.0f) {
                दशांश ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_block[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_block);

        साठी (पूर्णांक i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    साठी (; dim_idx < half_dims; ++dim_idx) {
        स्थिरांक दशांश ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        स्थिरांक दशांश theta_base = (दशांश) pos * theta;

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
स्थिर ओळीत शून्य compute_imrope_cache(दशांश *       cos_cache,
                                        दशांश *       sin_cache,
                                        int32_t       n_dims,
                                        दशांश         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        स्थिरांक int32_t sections[4],
                                        स्थिरांक दशांश * freq_factors,
                                        दशांश         freq_scale,
                                        स्थिरांक दशांश   corr_dims[2],
                                        दशांश         ext_factor,
                                        दशांश         attn_factor) {
    स्थिरांक int32_t half_dims = n_dims / 2;
    स्थिरांक int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    दशांश theta_t = (दशांश) pos_t;
    दशांश theta_h = (दशांश) pos_h;
    दशांश theta_w = (दशांश) pos_w;
    दशांश theta_e = (दशांश) pos_e;

    int32_t dim_idx = 0;

    साठी (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        दशांश theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        दशांश mscale = attn_factor;

        जर (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        साठी (पूर्णांक i = 0; i < ROPE_VEC_WIDTH; ++i) {
            स्थिरांक int32_t pair_idx = dim_idx + i;
            स्थिरांक int32_t sector   = pair_idx % sect_dims;
            स्थिरांक दशांश   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            दशांश theta;
            जर (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } अन्यथा जर (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } अन्यथा जर (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } अन्यथा {
                theta = theta_e;
            }

            स्थिरांक दशांश theta_extrap = et_fdiv(theta, ff);
            दशांश       theta_interp = freq_scale * theta_extrap;
            दशांश       theta_mix    = theta_interp;

            जर (ext_factor != 0.0f) {
                दशांश ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
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

        साठी (पूर्णांक i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    साठी (; dim_idx < half_dims; ++dim_idx) {
        स्थिरांक int32_t sector = dim_idx % sect_dims;
        स्थिरांक दशांश   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        दशांश theta;
        जर (sector % 3 == 1 && sector < 3 * sections[1]) {
            theta = theta_h;
        } अन्यथा जर (sector % 3 == 2 && sector < 3 * sections[2]) {
            theta = theta_w;
        } अन्यथा जर (sector % 3 == 0 && sector < 3 * sections[0]) {
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

पूर्णांक entry_point(संरचना ggml_et_rope_params * params, शून्य * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    जर (!kernel_env) {
        परत -1;
    }

    पूर्णांक thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    पूर्णांक अंक_threads = get_अंक_threads(kernel_env->shire_mask);

    जर (thread_id < 0) {
        परत -1;
    }

    जर (params == 0 || ((uint64_t) params & 0x7) != 0) {
        परत -1;
    }

    संरचना ggml_tensor * src0 = &params->src0;
    संरचना ggml_tensor * src1 = &params->src1;
    संरचना ggml_tensor * src2 = &params->src2;
    संरचना ggml_tensor * dst  = &params->dst;

    जर (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        परत -1;
    }

    स्थिरांक दशांश *   src0_data    = (स्थिरांक दशांश *) src0->data;
    स्थिरांक int32_t * src1_data    = (स्थिरांक int32_t *) src1->data;
    स्थिरांक दशांश *   freq_factors = (src2 && src2->data) ? (स्थिरांक दशांश *) src2->data : रिक्त;
    दशांश *         dst_data     = (दशांश *) dst->data;

    जर (!src0_data || !src1_data || !dst_data) {
        परत -1;
    }
#ifdef ET_UBERKERNEL
    स्थिरांक माप_प्रकार src0_bytes = (माप_प्रकार) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    स्थिरांक माप_प्रकार src1_bytes = (माप_प्रकार) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_data, src0_bytes);
    evict_region_past_l2(src1_data, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    स्थिरांक int64_t head_dim = src0->ne[0];
    स्थिरांक int64_t heads    = src0->ne[1];
    स्थिरांक int64_t seq_len  = src0->ne[2];
    स्थिरांक int64_t batch    = src0->ne[3];

    स्थिरांक rope_params_t * rope_params = &params->rope_params;
    स्थिरांक int32_t         n_dims      = rope_params->n_dims;
    स्थिरांक दशांश           freq_base   = rope_params->freq_base;
    स्थिरांक दशांश           freq_scale  = rope_params->freq_scale;
    स्थिरांक int32_t         mode        = rope_params->mode;

    जर (n_dims <= 0 || n_dims > head_dim || (n_dims & 1) != 0) {
        परत -1;
    }

    जर (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        परत -1;
    }

    दशांश cos_cache[MAX_ROPE_HALF_DIMS];
    दशांश sin_cache[MAX_ROPE_HALF_DIMS];

    दशांश corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_base, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    स्थिरांक int64_t total_heads = batch * seq_len * heads;
    स्थिरांक int64_t start_wu    = (total_heads * thread_id) / अंक_threads;
    स्थिरांक int64_t end_wu      = (total_heads * (thread_id + 1)) / अंक_threads;

    जर (start_wu >= end_wu) {
        परत 0;
    }

    स्थिरांक दशांश   theta_scale       = et_powf(freq_base, et_fdiv(-2.0f, (दशांश) n_dims));
    स्थिरांक int32_t half_dims         = n_dims / 2;
    स्थिरांक पूर्णांक     is_neox           = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    स्थिरांक पूर्णांक     is_imrope         = (mode == GGML_ROPE_TYPE_IMROPE);
    स्थिरांक पूर्णांक     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t last_pos   = -1;
    int32_t last_pos_h = -1;
    int32_t last_pos_w = -1;
    int32_t last_pos_e = -1;

    साठी (int64_t wu = start_wu; wu < end_wu; ++wu) {
        स्थिरांक int64_t h = wu % heads;
        स्थिरांक int64_t s = (wu / heads) % seq_len;
        स्थिरांक int64_t b = wu / (heads * seq_len);

        जर (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            स्थिरांक int32_t pt = src1_data[s] + rope_params->n_past;
            स्थिरांक int32_t ph = src1_data[s + seq_len] + rope_params->n_past;
            स्थिरांक int32_t pw = src1_data[s + seq_len * 2] + rope_params->n_past;
            स्थिरांक int32_t pe = src1_data[s + seq_len * 3] + rope_params->n_past;

            जर (pt != last_pos || ph != last_pos_h || pw != last_pos_w || pe != last_pos_e) {
                compute_imrope_cache(cos_cache, sin_cache, n_dims, theta_scale, pt, ph, pw, pe, rope_params->sections,
                                     freq_factors, freq_scale, corr_dims, rope_params->ext_factor,
                                     rope_params->attn_factor);
                last_pos   = pt;
                last_pos_h = ph;
                last_pos_w = pw;
                last_pos_e = pe;
            }
        } अन्यथा {
            स्थिरांक int32_t pos = src1_data[s] + rope_params->n_past;

            जर (pos != last_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_factor, rope_params->attn_factor);
                last_pos = pos;
            }
        }

        स्थिरांक दशांश * head_src =
            (स्थिरांक दशांश *) ((स्थिरांक अक्षर *) src0_data + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        दशांश * head_dst = (दशांश *) ((अक्षर *) dst_data + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        साठी (int64_t d = n_dims; d < head_dim; ++d) {
            head_dst[d] = head_src[d];
        }

        जर (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t temp_mask;
            __asm__ अस्थिर("mova.x.m %0" : "=r"(temp_mask));
            __asm__ अस्थिर("mov.m.x m0, x0, 0xFF");

            साठी (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
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
                    : [x0_dst] "=m"(*(दशांश (*)[8]) & head_dst[dim_idx]), [x1_dst] "=m"(*(दशांश (*)[8]) &
                                                                                        head_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(स्थिरांक दशांश (*)[8]) & head_src[dim_idx]),
                      [x1_src] "m"(*(स्थिरांक दशांश (*)[8]) & head_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(स्थिरांक दशांश (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(स्थिरांक दशांश (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ अस्थिर("mova.m.x %0" ::"r"(temp_mask));
        } अन्यथा {
            // Standard: adjacent pairs (2i, 2i+1)
            साठी (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                स्थिरांक int32_t dim_in_head = pair_idx * 2;
                स्थिरांक दशांश   x0          = head_src[dim_in_head];
                स्थिरांक दशांश   x1          = head_src[dim_in_head + 1];

                head_dst[dim_in_head]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                head_dst[dim_in_head + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    परत 0;
}
