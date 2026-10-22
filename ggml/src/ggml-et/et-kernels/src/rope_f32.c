//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#અન્તર્ભાવ "ggml_tensor.h"
#અન્તર્ભાવ "math_fp.h"
#અન્તર્ભાવ "platform.h"

#અન્તર્ભાવ <etsoc/common/utils.h>
#અન્તર્ભાવ <stdint.h>

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
પ્રકારવ્યાખ્યા સંરચના {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t mode;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    દશાંશ   freq_base;    // Base frequency (usually 10000.0f)
    દશાંશ   freq_scale;   // Frequency scaling factor
    દશાંશ   ext_factor;   // Extension factor for YaRN
    દશાંશ   attn_factor;  // Attention factor for YaRN
    દશાંશ   beta_fast;    // Fast beta for YaRN
    દશાંશ   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
સંરચના ggml_et_rope_params {
    સંરચના ggml_tensor src0;  // F32 input tensor
    સંરચના ggml_tensor src1;  // I32 position tensor
    સંરચના ggml_tensor src2;  // F32 frequency factors (optional)
    સંરચના ggml_tensor dst;   // F32 output tensor
    rope_params_t      rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

// floor/ceil with ±inf and NaN passthrough.
સ્થિર અંતઃરેખ દશાંશ rope_floorf(દશાંશ x) {
    સંઘ {
        દશાંશ    f;
        uint32_t u;
    } v = { .f = x };

    અચલ uint32_t expo = (v.u >> 23) & 0xFF;
    જો (expo == 0xFF) {
        પાછા x;  // inf or NaN
    }
    જો (expo >= 23 + 127) {
        પાછા x;  // already integer-valued
    }
    પૂર્ણાંક i = (પૂર્ણાંક) x;
    પાછા (x < 0.0f && (દશાંશ) i != x) ? (દશાંશ) (i - 1) : (દશાંશ) i;
}

સ્થિર અંતઃરેખ દશાંશ rope_ceilf(દશાંશ x) {
    સંઘ {
        દશાંશ    f;
        uint32_t u;
    } v = { .f = x };

    અચલ uint32_t expo = (v.u >> 23) & 0xFF;
    જો (expo == 0xFF) {
        પાછા x;  // inf or NaN
    }
    જો (expo >= 23 + 127) {
        પાછા x;  // already integer-valued
    }
    પૂર્ણાંક i = (પૂર્ણાંક) x;
    પાછા (x > 0.0f && (દશાંશ) i != x) ? (દશાંશ) (i + 1) : (દશાંશ) i;
}

સ્થિર અંતઃરેખ દશાંશ rope_yarn_ramp(અચલ દશાંશ નીચલું, અચલ દશાંશ ઉપલું, અચલ પૂર્ણાંક i0) {
    દશાંશ denom = ઉપલું - નીચલું;
    જો (denom < 0.001f) {
        denom = 0.001f;
    }

    અચલ દશાંશ y       = et_fdiv((દશાંશ) (i0 / 2) - નીચલું, denom);
    અચલ દશાંશ clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    પાછા 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
સ્થિર અંતઃરેખ દશાંશ rope_yarn_corr_dim(પૂર્ણાંક n_dims, પૂર્ણાંક n_ctx_orig, દશાંશ beta, દશાંશ freq_base) {
    પાછા (દશાંશ) n_dims *
           et_fdiv(et_logf(et_fdiv((દશાંશ) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_base));
}

સ્થિર અંતઃરેખ શૂન્ય rope_yarn_corr_dims(પૂર્ણાંક   n_dims,
                                       પૂર્ણાંક   n_ctx_orig,
                                       દશાંશ freq_base,
                                       દશાંશ beta_fast,
                                       દશાંશ beta_slow,
                                       દશાંશ dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    દશાંશ start = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    દશાંશ end   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));

    dims[0] = start > 0.0f ? start : 0.0f;
    dims[1] = end < (દશાંશ) (n_dims - 1) ? end : (દશાંશ) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

સ્થિર અચલ દશાંશ rope_ps_one[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
સ્થિર અચલ દશાંશ rope_ps_c3[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
સ્થિર અચલ દશાંશ rope_ps_c5[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
સ્થિર અચલ દશાંશ rope_ps_c7[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
સ્થિર અચલ દશાંશ rope_ps_c9[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
સ્થિર અચલ દશાંશ rope_ps_c11[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

સ્થિર અંતઃરેખ uint64_t rope_ps_enter_fullmask(શૂન્ય) {
    uint64_t old_mask;
    __asm__ અસ્થિર(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_mask)
        :
        : "t0", "memory");
    પાછા old_mask;
}

સ્થિર અંતઃરેખ શૂન્ય rope_ps_leave_fullmask(uint64_t old_mask) {
    __asm__ અસ્થિર("mova.m.x %0           \n\t" : : "r"(old_mask) : "memory");
}

સ્થિર અંતઃરેખ શૂન્ય rope_poly_sin_block8(દશાંશ * out, અચલ દશાંશ * x) {
    __asm__ અસ્થિર(
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
        : [out] "=m"(*(દશાંશ (*)[ROPE_VEC_WIDTH]) out)
        : [x] "m"(*(અચલ દશાંશ (*)[ROPE_VEC_WIDTH]) x), [one] "m"(*(અચલ દશાંશ (*)[ROPE_VEC_WIDTH]) rope_ps_one),
          [c3] "m"(*(અચલ દશાંશ (*)[ROPE_VEC_WIDTH]) rope_ps_c3),
          [c5] "m"(*(અચલ દશાંશ (*)[ROPE_VEC_WIDTH]) rope_ps_c5),
          [c7] "m"(*(અચલ દશાંશ (*)[ROPE_VEC_WIDTH]) rope_ps_c7),
          [c9] "m"(*(અચલ દશાંશ (*)[ROPE_VEC_WIDTH]) rope_ps_c9),
          [c11] "m"(*(અચલ દશાંશ (*)[ROPE_VEC_WIDTH]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

સ્થિર અંતઃરેખ શૂન્ય rope_sincos_block8(દશાંશ * sin8, દશાંશ * cos8, અચલ દશાંશ * theta8) {
    દશાંશ sin_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    દશાંશ cos_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    દશાંશ sin_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    દશાંશ cos_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));

    માટે (પૂર્ણાંક i = 0; i < ROPE_VEC_WIDTH; ++i) {
        દશાંશ x = theta8[i];

        જો (x > ROPE_PI || x < -ROPE_PI) {
            દશાંશ cycles = x * ROPE_INV_TWO_PI;
            પૂર્ણાંક   n      = (પૂર્ણાંક) cycles;
            જો (x < 0.0f) {
                n--;
            }
            x = x - (દશાંશ) n * ROPE_TWO_PI;
        }

        {
            દશાંશ y = x;
            દશાંશ s = 1.0f;
            જો (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } નહીંતર જો (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            દશાંશ y = x + ROPE_PI_OVER_2;
            જો (y > ROPE_PI || y < -ROPE_PI) {
                દશાંશ cycles = y * ROPE_INV_TWO_PI;
                પૂર્ણાંક   n      = (પૂર્ણાંક) cycles;
                જો (y < 0.0f) {
                    n--;
                }
                y = y - (દશાંશ) n * ROPE_TWO_PI;
            }

            દશાંશ s = 1.0f;
            જો (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } નહીંતર જો (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            cos_fold[i] = y;
            cos_sign[i] = s;
        }
    }

    {
        અચલ uint64_t saved_mask = rope_ps_enter_fullmask();

        rope_poly_sin_block8(sin8, sin_fold);
        rope_poly_sin_block8(cos8, cos_fold);

        __asm__ અસ્થિર(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(દશાંશ (*)[ROPE_VEC_WIDTH]) sin8), [cout] "=m"(*(દશાંશ (*)[ROPE_VEC_WIDTH]) cos8)
            : [sinv] "m"(*(અચલ દશાંશ (*)[ROPE_VEC_WIDTH]) sin8),
              [sinsgn] "m"(*(અચલ દશાંશ (*)[ROPE_VEC_WIDTH]) sin_sign),
              [cosv] "m"(*(અચલ દશાંશ (*)[ROPE_VEC_WIDTH]) cos8),
              [cossgn] "m"(*(અચલ દશાંશ (*)[ROPE_VEC_WIDTH]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_mask);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
સ્થિર અંતઃરેખ શૂન્ય rope_yarn_scalar(દશાંશ       theta_extrap,
                                    દશાંશ       freq_scale,
                                    અચલ દશાંશ corr_dims[2],
                                    int64_t     i0,
                                    દશાંશ       ext_factor,
                                    દશાંશ       mscale,
                                    દશાંશ *     cos_theta,
                                    દશાંશ *     sin_theta) {
    દશાંશ theta_interp = freq_scale * theta_extrap;
    દશાંશ theta        = theta_interp;

    જો (ext_factor != 0.0f) {
        દશાંશ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (પૂર્ણાંક) i0) * ext_factor;
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
સ્થિર અંતઃરેખ શૂન્ય compute_rope_cache(દશાંશ *       cos_cache,
                                      દશાંશ *       sin_cache,
                                      int32_t       n_dims,
                                      દશાંશ         theta_scale,
                                      int32_t       pos,
                                      અચલ દશાંશ * freq_factors,
                                      દશાંશ         freq_scale,
                                      અચલ દશાંશ   corr_dims[2],
                                      દશાંશ         ext_factor,
                                      દશાંશ         attn_factor) {
    અચલ int32_t half_dims = n_dims / 2;
    દશાંશ         theta     = 1.0f;

    int32_t dim_idx = 0;

    માટે (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        દશાંશ theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        દશાંશ theta_local = theta;
        દશાંશ mscale      = attn_factor;

        જો (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        માટે (પૂર્ણાંક i = 0; i < ROPE_VEC_WIDTH; ++i) {
            અચલ int32_t pair_idx     = dim_idx + i;
            અચલ દશાંશ   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            અચલ દશાંશ   theta_base   = (દશાંશ) pos * theta_local;
            અચલ દશાંશ   theta_extrap = et_fdiv(theta_base, ff);

            દશાંશ theta_interp = freq_scale * theta_extrap;
            દશાંશ theta_mix    = theta_interp;

            જો (ext_factor != 0.0f) {
                દશાંશ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_block[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_block);

        માટે (પૂર્ણાંક i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    માટે (; dim_idx < half_dims; ++dim_idx) {
        અચલ દશાંશ ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        અચલ દશાંશ theta_base = (દશાંશ) pos * theta;

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
સ્થિર અંતઃરેખ શૂન્ય compute_imrope_cache(દશાંશ *       cos_cache,
                                        દશાંશ *       sin_cache,
                                        int32_t       n_dims,
                                        દશાંશ         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        અચલ int32_t sections[4],
                                        અચલ દશાંશ * freq_factors,
                                        દશાંશ         freq_scale,
                                        અચલ દશાંશ   corr_dims[2],
                                        દશાંશ         ext_factor,
                                        દશાંશ         attn_factor) {
    અચલ int32_t half_dims = n_dims / 2;
    અચલ int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    દશાંશ theta_t = (દશાંશ) pos_t;
    દશાંશ theta_h = (દશાંશ) pos_h;
    દશાંશ theta_w = (દશાંશ) pos_w;
    દશાંશ theta_e = (દશાંશ) pos_e;

    int32_t dim_idx = 0;

    માટે (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        દશાંશ theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        દશાંશ mscale = attn_factor;

        જો (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        માટે (પૂર્ણાંક i = 0; i < ROPE_VEC_WIDTH; ++i) {
            અચલ int32_t pair_idx = dim_idx + i;
            અચલ int32_t sector   = pair_idx % sect_dims;
            અચલ દશાંશ   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            દશાંશ theta;
            જો (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } નહીંતર જો (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } નહીંતર જો (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } નહીંતર {
                theta = theta_e;
            }

            અચલ દશાંશ theta_extrap = et_fdiv(theta, ff);
            દશાંશ       theta_interp = freq_scale * theta_extrap;
            દશાંશ       theta_mix    = theta_interp;

            જો (ext_factor != 0.0f) {
                દશાંશ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
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

        માટે (પૂર્ણાંક i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    માટે (; dim_idx < half_dims; ++dim_idx) {
        અચલ int32_t sector = dim_idx % sect_dims;
        અચલ દશાંશ   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        દશાંશ theta;
        જો (sector % 3 == 1 && sector < 3 * sections[1]) {
            theta = theta_h;
        } નહીંતર જો (sector % 3 == 2 && sector < 3 * sections[2]) {
            theta = theta_w;
        } નહીંતર જો (sector % 3 == 0 && sector < 3 * sections[0]) {
            theta = theta_t;
        } નહીંતર {
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

પૂર્ણાંક entry_point(સંરચના ggml_et_rope_params * params, શૂન્ય * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    જો (!kernel_env) {
        પાછા -1;
    }

    પૂર્ણાંક thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    પૂર્ણાંક અંક_threads = get_અંક_threads(kernel_env->shire_mask);

    જો (thread_id < 0) {
        પાછા -1;
    }

    જો (params == 0 || ((uint64_t) params & 0x7) != 0) {
        પાછા -1;
    }

    સંરચના ggml_tensor * src0 = &params->src0;
    સંરચના ggml_tensor * src1 = &params->src1;
    સંરચના ggml_tensor * src2 = &params->src2;
    સંરચના ggml_tensor * dst  = &params->dst;

    જો (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        પાછા -1;
    }

    અચલ દશાંશ *   src0_data    = (અચલ દશાંશ *) src0->data;
    અચલ int32_t * src1_data    = (અચલ int32_t *) src1->data;
    અચલ દશાંશ *   freq_factors = (src2 && src2->data) ? (અચલ દશાંશ *) src2->data : ખાલી;
    દશાંશ *         dst_data     = (દશાંશ *) dst->data;

    જો (!src0_data || !src1_data || !dst_data) {
        પાછા -1;
    }
#ifdef ET_UBERKERNEL
    અચલ size_t src0_bytes = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    અચલ size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_data, src0_bytes);
    evict_region_past_l2(src1_data, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    અચલ int64_t head_dim = src0->ne[0];
    અચલ int64_t heads    = src0->ne[1];
    અચલ int64_t seq_len  = src0->ne[2];
    અચલ int64_t batch    = src0->ne[3];

    અચલ rope_params_t * rope_params = &params->rope_params;
    અચલ int32_t         n_dims      = rope_params->n_dims;
    અચલ દશાંશ           freq_base   = rope_params->freq_base;
    અચલ દશાંશ           freq_scale  = rope_params->freq_scale;
    અચલ int32_t         mode        = rope_params->mode;

    જો (n_dims <= 0 || n_dims > head_dim || (n_dims & 1) != 0) {
        પાછા -1;
    }

    જો (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        પાછા -1;
    }

    દશાંશ cos_cache[MAX_ROPE_HALF_DIMS];
    દશાંશ sin_cache[MAX_ROPE_HALF_DIMS];

    દશાંશ corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_base, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    અચલ int64_t total_heads = batch * seq_len * heads;
    અચલ int64_t start_wu    = (total_heads * thread_id) / અંક_threads;
    અચલ int64_t end_wu      = (total_heads * (thread_id + 1)) / અંક_threads;

    જો (start_wu >= end_wu) {
        પાછા 0;
    }

    અચલ દશાંશ   theta_scale       = et_powf(freq_base, et_fdiv(-2.0f, (દશાંશ) n_dims));
    અચલ int32_t half_dims         = n_dims / 2;
    અચલ પૂર્ણાંક     is_neox           = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    અચલ પૂર્ણાંક     is_imrope         = (mode == GGML_ROPE_TYPE_IMROPE);
    અચલ પૂર્ણાંક     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t last_pos   = -1;
    int32_t last_pos_h = -1;
    int32_t last_pos_w = -1;
    int32_t last_pos_e = -1;

    માટે (int64_t wu = start_wu; wu < end_wu; ++wu) {
        અચલ int64_t h = wu % heads;
        અચલ int64_t s = (wu / heads) % seq_len;
        અચલ int64_t b = wu / (heads * seq_len);

        જો (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            અચલ int32_t pt = src1_data[s] + rope_params->n_past;
            અચલ int32_t ph = src1_data[s + seq_len] + rope_params->n_past;
            અચલ int32_t pw = src1_data[s + seq_len * 2] + rope_params->n_past;
            અચલ int32_t pe = src1_data[s + seq_len * 3] + rope_params->n_past;

            જો (pt != last_pos || ph != last_pos_h || pw != last_pos_w || pe != last_pos_e) {
                compute_imrope_cache(cos_cache, sin_cache, n_dims, theta_scale, pt, ph, pw, pe, rope_params->sections,
                                     freq_factors, freq_scale, corr_dims, rope_params->ext_factor,
                                     rope_params->attn_factor);
                last_pos   = pt;
                last_pos_h = ph;
                last_pos_w = pw;
                last_pos_e = pe;
            }
        } નહીંતર {
            અચલ int32_t pos = src1_data[s] + rope_params->n_past;

            જો (pos != last_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_factor, rope_params->attn_factor);
                last_pos = pos;
            }
        }

        અચલ દશાંશ * head_src =
            (અચલ દશાંશ *) ((અચલ અક્ષર *) src0_data + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        દશાંશ * head_dst = (દશાંશ *) ((અક્ષર *) dst_data + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        માટે (int64_t d = n_dims; d < head_dim; ++d) {
            head_dst[d] = head_src[d];
        }

        જો (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t temp_mask;
            __asm__ અસ્થિર("mova.x.m %0" : "=r"(temp_mask));
            __asm__ અસ્થિર("mov.m.x m0, x0, 0xFF");

            માટે (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                __asm__ અસ્થિર(
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
                    : [x0_dst] "=m"(*(દશાંશ (*)[8]) & head_dst[dim_idx]), [x1_dst] "=m"(*(દશાંશ (*)[8]) &
                                                                                        head_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(અચલ દશાંશ (*)[8]) & head_src[dim_idx]),
                      [x1_src] "m"(*(અચલ દશાંશ (*)[8]) & head_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(અચલ દશાંશ (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(અચલ દશાંશ (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ અસ્થિર("mova.m.x %0" ::"r"(temp_mask));
        } નહીંતર {
            // Standard: adjacent pairs (2i, 2i+1)
            માટે (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                અચલ int32_t dim_in_head = pair_idx * 2;
                અચલ દશાંશ   x0          = head_src[dim_in_head];
                અચલ દશાંશ   x1          = head_src[dim_in_head + 1];

                head_dst[dim_in_head]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                head_dst[dim_in_head + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    પાછા 0;
}
