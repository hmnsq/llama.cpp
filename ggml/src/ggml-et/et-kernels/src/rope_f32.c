//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#ضمن "ggml_tensor.h"
#ضمن "math_fp.h"
#ضمن "platform.h"

#ضمن <etsoc/common/utils.h>
#ضمن <stdint.h>

// ROPE constants (matching GGML definitions)
#تعریف_کرو GGML_ROPE_TYPE_NEOX   2
#تعریف_کرو GGML_ROPE_TYPE_MROPE  8
#تعریف_کرو GGML_ROPE_TYPE_IMROPE 40
#تعریف_کرو MAX_ROPE_HALF_DIMS    256  // supports up to n_dims=512

#تعریف_کرو ROPE_VEC_چوڑائی 8

#تعریف_کرو ROPE_PI         3.14159265358979323846f
#تعریف_کرو ROPE_TWO_PI     6.28318530717958647693f
#تعریف_کرو ROPE_PI_OVER_2  1.57079632679489661923f
#تعریف_کرو ROPE_INV_TWO_PI 0.15915494309189533577f

// ROPE operation parameters structure (matches ggml-et-ops.h)
تعريفنوع بنية {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t منوال;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    عائم   freq_ادھار;    // Base frequency (usually 10000.0f)
    عائم   freq_scale;   // Frequency scaling factor
    عائم   ext_ضارب;   // Extension factor for YaRN
    عائم   attn_ضارب;  // Attention factor for YaRN
    عائم   beta_fast;    // Fast beta for YaRN
    عائم   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
بنية ggml_et_rope_params {
    بنية ggml_tensor src0;  // F32 input tensor
    بنية ggml_tensor src1;  // I32 position tensor
    بنية ggml_tensor src2;  // F32 frequency factors (optional)
    بنية ggml_tensor dst;   // F32 output tensor
    rope_params_t      rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

// floor/ceil with ±inf and NaN passthrough.
ساكن مضمن عائم rope_floorf(عائم x) {
    اتحاد {
        عائم    f;
        uint32_t u;
    } v = { .f = x };

    ثابت uint32_t expo = (v.u >> 23) & 0xFF;
    إذا (expo == 0xFF) {
        أرجع x;  // inf or NaN
    }
    إذا (expo >= 23 + 127) {
        أرجع x;  // already integer-valued
    }
    صحيح i = (صحيح) x;
    أرجع (x < 0.0f && (عائم) i != x) ? (عائم) (i - 1) : (عائم) i;
}

ساكن مضمن عائم rope_ceilf(عائم x) {
    اتحاد {
        عائم    f;
        uint32_t u;
    } v = { .f = x };

    ثابت uint32_t expo = (v.u >> 23) & 0xFF;
    إذا (expo == 0xFF) {
        أرجع x;  // inf or NaN
    }
    إذا (expo >= 23 + 127) {
        أرجع x;  // already integer-valued
    }
    صحيح i = (صحيح) x;
    أرجع (x > 0.0f && (عائم) i != x) ? (عائم) (i + 1) : (عائم) i;
}

ساكن مضمن عائم rope_yarn_ramp(ثابت عائم کم, ثابت عائم بلند, ثابت صحيح i0) {
    عائم denom = بلند - کم;
    إذا (denom < 0.001f) {
        denom = 0.001f;
    }

    ثابت عائم y       = et_fdiv((عائم) (i0 / 2) - کم, denom);
    ثابت عائم clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    أرجع 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
ساكن مضمن عائم rope_yarn_corr_dim(صحيح n_dims, صحيح n_ctx_orig, عائم beta, عائم freq_ادھار) {
    أرجع (عائم) n_dims *
           et_fdiv(et_logf(et_fdiv((عائم) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_ادھار));
}

ساكن مضمن فراغ rope_yarn_corr_dims(صحيح   n_dims,
                                       صحيح   n_ctx_orig,
                                       عائم freq_ادھار,
                                       عائم beta_fast,
                                       عائم beta_slow,
                                       عائم dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    عائم شروع = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_ادھار));
    عائم سرا   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_ادھار));

    dims[0] = شروع > 0.0f ? شروع : 0.0f;
    dims[1] = سرا < (عائم) (n_dims - 1) ? سرا : (عائم) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

ساكن ثابت عائم rope_ps_one[ROPE_VEC_چوڑائی]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
ساكن ثابت عائم rope_ps_c3[ROPE_VEC_چوڑائی]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
ساكن ثابت عائم rope_ps_c5[ROPE_VEC_چوڑائی]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
ساكن ثابت عائم rope_ps_c7[ROPE_VEC_چوڑائی]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
ساكن ثابت عائم rope_ps_c9[ROPE_VEC_چوڑائی]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
ساكن ثابت عائم rope_ps_c11[ROPE_VEC_چوڑائی]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

ساكن مضمن uint64_t rope_ps_enter_fullmask(فراغ) {
    uint64_t old_نقاب;
    __asm__ متطاير(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_نقاب)
        :
        : "t0", "memory");
    أرجع old_نقاب;
}

ساكن مضمن فراغ rope_ps_leave_fullmask(uint64_t old_نقاب) {
    __asm__ متطاير("mova.m.x %0           \n\t" : : "r"(old_نقاب) : "memory");
}

ساكن مضمن فراغ rope_poly_sin_block8(عائم * باہرلا, ثابت عائم * x) {
    __asm__ متطاير(
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
        : [باہرلا] "=m"(*(عائم (*)[ROPE_VEC_چوڑائی]) باہرلا)
        : [x] "m"(*(ثابت عائم (*)[ROPE_VEC_چوڑائی]) x), [one] "m"(*(ثابت عائم (*)[ROPE_VEC_چوڑائی]) rope_ps_one),
          [c3] "m"(*(ثابت عائم (*)[ROPE_VEC_چوڑائی]) rope_ps_c3),
          [c5] "m"(*(ثابت عائم (*)[ROPE_VEC_چوڑائی]) rope_ps_c5),
          [c7] "m"(*(ثابت عائم (*)[ROPE_VEC_چوڑائی]) rope_ps_c7),
          [c9] "m"(*(ثابت عائم (*)[ROPE_VEC_چوڑائی]) rope_ps_c9),
          [c11] "m"(*(ثابت عائم (*)[ROPE_VEC_چوڑائی]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

ساكن مضمن فراغ rope_sincos_block8(عائم * sin8, عائم * cos8, ثابت عائم * theta8) {
    عائم sin_fold[ROPE_VEC_چوڑائی] __attribute__((aligned(32)));
    عائم cos_fold[ROPE_VEC_چوڑائی] __attribute__((aligned(32)));
    عائم sin_sign[ROPE_VEC_چوڑائی] __attribute__((aligned(32)));
    عائم cos_sign[ROPE_VEC_چوڑائی] __attribute__((aligned(32)));

    لكل (صحيح i = 0; i < ROPE_VEC_چوڑائی; ++i) {
        عائم x = theta8[i];

        إذا (x > ROPE_PI || x < -ROPE_PI) {
            عائم چکراں = x * ROPE_INV_TWO_PI;
            صحيح   n      = (صحيح) چکراں;
            إذا (x < 0.0f) {
                n--;
            }
            x = x - (عائم) n * ROPE_TWO_PI;
        }

        {
            عائم y = x;
            عائم s = 1.0f;
            إذا (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } وإلا إذا (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            عائم y = x + ROPE_PI_OVER_2;
            إذا (y > ROPE_PI || y < -ROPE_PI) {
                عائم چکراں = y * ROPE_INV_TWO_PI;
                صحيح   n      = (صحيح) چکراں;
                إذا (y < 0.0f) {
                    n--;
                }
                y = y - (عائم) n * ROPE_TWO_PI;
            }

            عائم s = 1.0f;
            إذا (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } وإلا إذا (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            cos_fold[i] = y;
            cos_sign[i] = s;
        }
    }

    {
        ثابت uint64_t saved_نقاب = rope_ps_enter_fullmask();

        rope_poly_sin_block8(sin8, sin_fold);
        rope_poly_sin_block8(cos8, cos_fold);

        __asm__ متطاير(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(عائم (*)[ROPE_VEC_چوڑائی]) sin8), [cout] "=m"(*(عائم (*)[ROPE_VEC_چوڑائی]) cos8)
            : [sinv] "m"(*(ثابت عائم (*)[ROPE_VEC_چوڑائی]) sin8),
              [sinsgn] "m"(*(ثابت عائم (*)[ROPE_VEC_چوڑائی]) sin_sign),
              [cosv] "m"(*(ثابت عائم (*)[ROPE_VEC_چوڑائی]) cos8),
              [cossgn] "m"(*(ثابت عائم (*)[ROPE_VEC_چوڑائی]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_نقاب);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
ساكن مضمن فراغ rope_yarn_scalar(عائم       theta_extrap,
                                    عائم       freq_scale,
                                    ثابت عائم corr_dims[2],
                                    int64_t     i0,
                                    عائم       ext_ضارب,
                                    عائم       mscale,
                                    عائم *     cos_theta,
                                    عائم *     sin_theta) {
    عائم theta_interp = freq_scale * theta_extrap;
    عائم theta        = theta_interp;

    إذا (ext_ضارب != 0.0f) {
        عائم ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (صحيح) i0) * ext_ضارب;
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
ساكن مضمن فراغ compute_rope_cache(عائم *       cos_cache,
                                      عائم *       sin_cache,
                                      int32_t       n_dims,
                                      عائم         theta_scale,
                                      int32_t       pos,
                                      ثابت عائم * freq_factors,
                                      عائم         freq_scale,
                                      ثابت عائم   corr_dims[2],
                                      عائم         ext_ضارب,
                                      عائم         attn_ضارب) {
    ثابت int32_t half_dims = n_dims / 2;
    عائم         theta     = 1.0f;

    int32_t dim_idx = 0;

    لكل (; dim_idx + ROPE_VEC_چوڑائی <= half_dims; dim_idx += ROPE_VEC_چوڑائی) {
        عائم theta_بلاک[ROPE_VEC_چوڑائی] __attribute__((aligned(32)));
        عائم theta_local = theta;
        عائم mscale      = attn_ضارب;

        إذا (ext_ضارب != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        لكل (صحيح i = 0; i < ROPE_VEC_چوڑائی; ++i) {
            ثابت int32_t pair_idx     = dim_idx + i;
            ثابت عائم   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            ثابت عائم   theta_ادھار   = (عائم) pos * theta_local;
            ثابت عائم   theta_extrap = et_fdiv(theta_ادھار, ff);

            عائم theta_interp = freq_scale * theta_extrap;
            عائم theta_mix    = theta_interp;

            إذا (ext_ضارب != 0.0f) {
                عائم ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_ضارب;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_بلاک[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_بلاک);

        لكل (صحيح i = 0; i < ROPE_VEC_چوڑائی; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    لكل (; dim_idx < half_dims; ++dim_idx) {
        ثابت عائم ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        ثابت عائم theta_ادھار = (عائم) pos * theta;

        rope_yarn_scalar(et_fdiv(theta_ادھار, ff), freq_scale, corr_dims, dim_idx * 2, ext_ضارب, attn_ضارب,
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
ساكن مضمن فراغ compute_imrope_cache(عائم *       cos_cache,
                                        عائم *       sin_cache,
                                        int32_t       n_dims,
                                        عائم         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        ثابت int32_t sections[4],
                                        ثابت عائم * freq_factors,
                                        عائم         freq_scale,
                                        ثابت عائم   corr_dims[2],
                                        عائم         ext_ضارب,
                                        عائم         attn_ضارب) {
    ثابت int32_t half_dims = n_dims / 2;
    ثابت int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    عائم theta_t = (عائم) pos_t;
    عائم theta_h = (عائم) pos_h;
    عائم theta_w = (عائم) pos_w;
    عائم theta_e = (عائم) pos_e;

    int32_t dim_idx = 0;

    لكل (; dim_idx + ROPE_VEC_چوڑائی <= half_dims; dim_idx += ROPE_VEC_چوڑائی) {
        عائم theta_بلاک[ROPE_VEC_چوڑائی] __attribute__((aligned(32)));
        عائم mscale = attn_ضارب;

        إذا (ext_ضارب != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        لكل (صحيح i = 0; i < ROPE_VEC_چوڑائی; ++i) {
            ثابت int32_t pair_idx = dim_idx + i;
            ثابت int32_t sector   = pair_idx % sect_dims;
            ثابت عائم   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            عائم theta;
            إذا (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } وإلا إذا (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } وإلا إذا (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } وإلا {
                theta = theta_e;
            }

            ثابت عائم theta_extrap = et_fdiv(theta, ff);
            عائم       theta_interp = freq_scale * theta_extrap;
            عائم       theta_mix    = theta_interp;

            إذا (ext_ضارب != 0.0f) {
                عائم ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_ضارب;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_بلاک[i] = theta_mix;

            // All 4 thetas advance every iteration
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_بلاک);

        لكل (صحيح i = 0; i < ROPE_VEC_چوڑائی; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    لكل (; dim_idx < half_dims; ++dim_idx) {
        ثابت int32_t sector = dim_idx % sect_dims;
        ثابت عائم   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        عائم theta;
        إذا (sector % 3 == 1 && sector < 3 * sections[1]) {
            theta = theta_h;
        } وإلا إذا (sector % 3 == 2 && sector < 3 * sections[2]) {
            theta = theta_w;
        } وإلا إذا (sector % 3 == 0 && sector < 3 * sections[0]) {
            theta = theta_t;
        } وإلا {
            theta = theta_e;
        }

        rope_yarn_scalar(et_fdiv(theta, ff), freq_scale, corr_dims, dim_idx * 2, ext_ضارب, attn_ضارب,
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

صحيح اندراج_نقطہ(بنية ggml_et_rope_params * params, فراغ * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    إذا (!kernel_env) {
        أرجع -1;
    }

    صحيح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحيح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    إذا (thread_id < 0) {
        أرجع -1;
    }

    إذا (params == 0 || ((uint64_t) params & 0x7) != 0) {
        أرجع -1;
    }

    بنية ggml_tensor * src0 = &params->src0;
    بنية ggml_tensor * src1 = &params->src1;
    بنية ggml_tensor * src2 = &params->src2;
    بنية ggml_tensor * dst  = &params->dst;

    إذا (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        أرجع -1;
    }

    ثابت عائم *   src0_انکڑے    = (ثابت عائم *) src0->انکڑے;
    ثابت int32_t * src1_انکڑے    = (ثابت int32_t *) src1->انکڑے;
    ثابت عائم *   freq_factors = (src2 && src2->انکڑے) ? (ثابت عائم *) src2->انکڑے : عدم;
    عائم *         dst_انکڑے     = (عائم *) dst->انکڑے;

    إذا (!src0_انکڑے || !src1_انکڑے || !dst_انکڑے) {
        أرجع -1;
    }
#جے_تعریف_شدہ ET_UBERKERNEL
    ثابت size_t src0_بائٹاں = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ثابت size_t src1_بائٹاں = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_انکڑے, src0_بائٹاں);
    evict_region_past_l2(src1_انکڑے, src1_بائٹاں);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#جے_انت
    ثابت int64_t سر_dim = src0->ne[0];
    ثابت int64_t heads    = src0->ne[1];
    ثابت int64_t seq_len  = src0->ne[2];
    ثابت int64_t batch    = src0->ne[3];

    ثابت rope_params_t * rope_params = &params->rope_params;
    ثابت int32_t         n_dims      = rope_params->n_dims;
    ثابت عائم           freq_ادھار   = rope_params->freq_ادھار;
    ثابت عائم           freq_scale  = rope_params->freq_scale;
    ثابت int32_t         منوال        = rope_params->منوال;

    إذا (n_dims <= 0 || n_dims > سر_dim || (n_dims & 1) != 0) {
        أرجع -1;
    }

    إذا (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        أرجع -1;
    }

    عائم cos_cache[MAX_ROPE_HALF_DIMS];
    عائم sin_cache[MAX_ROPE_HALF_DIMS];

    عائم corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_ادھار, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    ثابت int64_t کل_heads = batch * seq_len * heads;
    ثابت int64_t شروع_wu    = (کل_heads * thread_id) / عدد_دھاگے;
    ثابت int64_t سرا_wu      = (کل_heads * (thread_id + 1)) / عدد_دھاگے;

    إذا (شروع_wu >= سرا_wu) {
        أرجع 0;
    }

    ثابت عائم   theta_scale       = et_powf(freq_ادھار, et_fdiv(-2.0f, (عائم) n_dims));
    ثابت int32_t half_dims         = n_dims / 2;
    ثابت صحيح     is_neox           = (منوال & GGML_ROPE_TYPE_NEOX) != 0;
    ثابت صحيح     is_imrope         = (منوال == GGML_ROPE_TYPE_IMROPE);
    ثابت صحيح     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t آخری_pos   = -1;
    int32_t آخری_pos_h = -1;
    int32_t آخری_pos_w = -1;
    int32_t آخری_pos_e = -1;

    لكل (int64_t wu = شروع_wu; wu < سرا_wu; ++wu) {
        ثابت int64_t h = wu % heads;
        ثابت int64_t s = (wu / heads) % seq_len;
        ثابت int64_t b = wu / (heads * seq_len);

        إذا (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            ثابت int32_t pt = src1_انکڑے[s] + rope_params->n_past;
            ثابت int32_t ph = src1_انکڑے[s + seq_len] + rope_params->n_past;
            ثابت int32_t pw = src1_انکڑے[s + seq_len * 2] + rope_params->n_past;
            ثابت int32_t pe = src1_انکڑے[s + seq_len * 3] + rope_params->n_past;

            إذا (pt != آخری_pos || ph != آخری_pos_h || pw != آخری_pos_w || pe != آخری_pos_e) {
                compute_imrope_cache(cos_cache, sin_cache, n_dims, theta_scale, pt, ph, pw, pe, rope_params->sections,
                                     freq_factors, freq_scale, corr_dims, rope_params->ext_ضارب,
                                     rope_params->attn_ضارب);
                آخری_pos   = pt;
                آخری_pos_h = ph;
                آخری_pos_w = pw;
                آخری_pos_e = pe;
            }
        } وإلا {
            ثابت int32_t pos = src1_انکڑے[s] + rope_params->n_past;

            إذا (pos != آخری_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_ضارب, rope_params->attn_ضارب);
                آخری_pos = pos;
            }
        }

        ثابت عائم * سر_src =
            (ثابت عائم *) ((ثابت حرف *) src0_انکڑے + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        عائم * سر_dst = (عائم *) ((حرف *) dst_انکڑے + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        لكل (int64_t d = n_dims; d < سر_dim; ++d) {
            سر_dst[d] = سر_src[d];
        }

        إذا (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t عارضی_نقاب;
            __asm__ متطاير("mova.x.m %0" : "=r"(عارضی_نقاب));
            __asm__ متطاير("mov.m.x m0, x0, 0xFF");

            لكل (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                __asm__ متطاير(
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
                    : [x0_dst] "=m"(*(عائم (*)[8]) & سر_dst[dim_idx]), [x1_dst] "=m"(*(عائم (*)[8]) &
                                                                                        سر_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(ثابت عائم (*)[8]) & سر_src[dim_idx]),
                      [x1_src] "m"(*(ثابت عائم (*)[8]) & سر_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(ثابت عائم (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(ثابت عائم (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ متطاير("mova.m.x %0" ::"r"(عارضی_نقاب));
        } وإلا {
            // Standard: adjacent pairs (2i, 2i+1)
            لكل (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                ثابت int32_t dim_وچ_سر = pair_idx * 2;
                ثابت عائم   x0          = سر_src[dim_وچ_سر];
                ثابت عائم   x1          = سر_src[dim_وچ_سر + 1];

                سر_dst[dim_وچ_سر]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                سر_dst[dim_وچ_سر + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    أرجع 0;
}
