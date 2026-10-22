//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#അന്തര്ഭാവ "ggml_tensor.h"
#അന്തര്ഭാവ "math_fp.h"
#അന്തര്ഭാവ "platform.h"

#അന്തര്ഭാവ <etsoc/common/utils.h>
#അന്തര്ഭാവ <stdint.h>

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
തരനിർവചനം ഘടന {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t mode;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    ദശാംശം   freq_base;    // Base frequency (usually 10000.0f)
    ദശാംശം   freq_scale;   // Frequency scaling factor
    ദശാംശം   ext_factor;   // Extension factor for YaRN
    ദശാംശം   attn_factor;  // Attention factor for YaRN
    ദശാംശം   beta_fast;    // Fast beta for YaRN
    ദശാംശം   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
ഘടന ggml_et_rope_params {
    ഘടന ggml_tensor src0;  // F32 input tensor
    ഘടന ggml_tensor src1;  // I32 position tensor
    ഘടന ggml_tensor src2;  // F32 frequency factors (optional)
    ഘടന ggml_tensor dst;   // F32 output tensor
    rope_params_t      rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

// floor/ceil with ±inf and NaN passthrough.
സ്ഥിതം വരിയിൽ ദശാംശം rope_floorf(ദശാംശം x) {
    സംഘം {
        ദശാംശം    f;
        uint32_t u;
    } v = { .f = x };

    സ്ഥിരം uint32_t expo = (v.u >> 23) & 0xFF;
    എങ്കിൽ (expo == 0xFF) {
        തിരികെ x;  // inf or NaN
    }
    എങ്കിൽ (expo >= 23 + 127) {
        തിരികെ x;  // already integer-valued
    }
    പൂർണ്ണസംഖ്യ i = (പൂർണ്ണസംഖ്യ) x;
    തിരികെ (x < 0.0f && (ദശാംശം) i != x) ? (ദശാംശം) (i - 1) : (ദശാംശം) i;
}

സ്ഥിതം വരിയിൽ ദശാംശം rope_ceilf(ദശാംശം x) {
    സംഘം {
        ദശാംശം    f;
        uint32_t u;
    } v = { .f = x };

    സ്ഥിരം uint32_t expo = (v.u >> 23) & 0xFF;
    എങ്കിൽ (expo == 0xFF) {
        തിരികെ x;  // inf or NaN
    }
    എങ്കിൽ (expo >= 23 + 127) {
        തിരികെ x;  // already integer-valued
    }
    പൂർണ്ണസംഖ്യ i = (പൂർണ്ണസംഖ്യ) x;
    തിരികെ (x > 0.0f && (ദശാംശം) i != x) ? (ദശാംശം) (i + 1) : (ദശാംശം) i;
}

സ്ഥിതം വരിയിൽ ദശാംശം rope_yarn_ramp(സ്ഥിരം ദശാംശം താഴെയുള്ള, സ്ഥിരം ദശാംശം മുകളിലെ, സ്ഥിരം പൂർണ്ണസംഖ്യ i0) {
    ദശാംശം denom = മുകളിലെ - താഴെയുള്ള;
    എങ്കിൽ (denom < 0.001f) {
        denom = 0.001f;
    }

    സ്ഥിരം ദശാംശം y       = et_fdiv((ദശാംശം) (i0 / 2) - താഴെയുള്ള, denom);
    സ്ഥിരം ദശാംശം clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    തിരികെ 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
സ്ഥിതം വരിയിൽ ദശാംശം rope_yarn_corr_dim(പൂർണ്ണസംഖ്യ n_dims, പൂർണ്ണസംഖ്യ n_ctx_orig, ദശാംശം beta, ദശാംശം freq_base) {
    തിരികെ (ദശാംശം) n_dims *
           et_fdiv(et_logf(et_fdiv((ദശാംശം) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_base));
}

സ്ഥിതം വരിയിൽ ശൂന്യം rope_yarn_corr_dims(പൂർണ്ണസംഖ്യ   n_dims,
                                       പൂർണ്ണസംഖ്യ   n_ctx_orig,
                                       ദശാംശം freq_base,
                                       ദശാംശം beta_fast,
                                       ദശാംശം beta_slow,
                                       ദശാംശം dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    ദശാംശം start = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    ദശാംശം end   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));

    dims[0] = start > 0.0f ? start : 0.0f;
    dims[1] = end < (ദശാംശം) (n_dims - 1) ? end : (ദശാംശം) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

സ്ഥിതം സ്ഥിരം ദശാംശം rope_ps_one[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
സ്ഥിതം സ്ഥിരം ദശാംശം rope_ps_c3[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
സ്ഥിതം സ്ഥിരം ദശാംശം rope_ps_c5[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
സ്ഥിതം സ്ഥിരം ദശാംശം rope_ps_c7[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
സ്ഥിതം സ്ഥിരം ദശാംശം rope_ps_c9[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
സ്ഥിതം സ്ഥിരം ദശാംശം rope_ps_c11[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

സ്ഥിതം വരിയിൽ uint64_t rope_ps_enter_fullmask(ശൂന്യം) {
    uint64_t old_mask;
    __asm__ അസ്ഥിര(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_mask)
        :
        : "t0", "memory");
    തിരികെ old_mask;
}

സ്ഥിതം വരിയിൽ ശൂന്യം rope_ps_leave_fullmask(uint64_t old_mask) {
    __asm__ അസ്ഥിര("mova.m.x %0           \n\t" : : "r"(old_mask) : "memory");
}

സ്ഥിതം വരിയിൽ ശൂന്യം rope_poly_sin_block8(ദശാംശം * out, സ്ഥിരം ദശാംശം * x) {
    __asm__ അസ്ഥിര(
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
        : [out] "=m"(*(ദശാംശം (*)[ROPE_VEC_WIDTH]) out)
        : [x] "m"(*(സ്ഥിരം ദശാംശം (*)[ROPE_VEC_WIDTH]) x), [one] "m"(*(സ്ഥിരം ദശാംശം (*)[ROPE_VEC_WIDTH]) rope_ps_one),
          [c3] "m"(*(സ്ഥിരം ദശാംശം (*)[ROPE_VEC_WIDTH]) rope_ps_c3),
          [c5] "m"(*(സ്ഥിരം ദശാംശം (*)[ROPE_VEC_WIDTH]) rope_ps_c5),
          [c7] "m"(*(സ്ഥിരം ദശാംശം (*)[ROPE_VEC_WIDTH]) rope_ps_c7),
          [c9] "m"(*(സ്ഥിരം ദശാംശം (*)[ROPE_VEC_WIDTH]) rope_ps_c9),
          [c11] "m"(*(സ്ഥിരം ദശാംശം (*)[ROPE_VEC_WIDTH]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

സ്ഥിതം വരിയിൽ ശൂന്യം rope_sincos_block8(ദശാംശം * sin8, ദശാംശം * cos8, സ്ഥിരം ദശാംശം * theta8) {
    ദശാംശം sin_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    ദശാംശം cos_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    ദശാംശം sin_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    ദശാംശം cos_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));

    വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ROPE_VEC_WIDTH; ++i) {
        ദശാംശം x = theta8[i];

        എങ്കിൽ (x > ROPE_PI || x < -ROPE_PI) {
            ദശാംശം cycles = x * ROPE_INV_TWO_PI;
            പൂർണ്ണസംഖ്യ   n      = (പൂർണ്ണസംഖ്യ) cycles;
            എങ്കിൽ (x < 0.0f) {
                n--;
            }
            x = x - (ദശാംശം) n * ROPE_TWO_PI;
        }

        {
            ദശാംശം y = x;
            ദശാംശം s = 1.0f;
            എങ്കിൽ (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } അല്ലെങ്കിൽ എങ്കിൽ (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            ദശാംശം y = x + ROPE_PI_OVER_2;
            എങ്കിൽ (y > ROPE_PI || y < -ROPE_PI) {
                ദശാംശം cycles = y * ROPE_INV_TWO_PI;
                പൂർണ്ണസംഖ്യ   n      = (പൂർണ്ണസംഖ്യ) cycles;
                എങ്കിൽ (y < 0.0f) {
                    n--;
                }
                y = y - (ദശാംശം) n * ROPE_TWO_PI;
            }

            ദശാംശം s = 1.0f;
            എങ്കിൽ (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } അല്ലെങ്കിൽ എങ്കിൽ (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            cos_fold[i] = y;
            cos_sign[i] = s;
        }
    }

    {
        സ്ഥിരം uint64_t saved_mask = rope_ps_enter_fullmask();

        rope_poly_sin_block8(sin8, sin_fold);
        rope_poly_sin_block8(cos8, cos_fold);

        __asm__ അസ്ഥിര(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(ദശാംശം (*)[ROPE_VEC_WIDTH]) sin8), [cout] "=m"(*(ദശാംശം (*)[ROPE_VEC_WIDTH]) cos8)
            : [sinv] "m"(*(സ്ഥിരം ദശാംശം (*)[ROPE_VEC_WIDTH]) sin8),
              [sinsgn] "m"(*(സ്ഥിരം ദശാംശം (*)[ROPE_VEC_WIDTH]) sin_sign),
              [cosv] "m"(*(സ്ഥിരം ദശാംശം (*)[ROPE_VEC_WIDTH]) cos8),
              [cossgn] "m"(*(സ്ഥിരം ദശാംശം (*)[ROPE_VEC_WIDTH]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_mask);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
സ്ഥിതം വരിയിൽ ശൂന്യം rope_yarn_scalar(ദശാംശം       theta_extrap,
                                    ദശാംശം       freq_scale,
                                    സ്ഥിരം ദശാംശം corr_dims[2],
                                    int64_t     i0,
                                    ദശാംശം       ext_factor,
                                    ദശാംശം       mscale,
                                    ദശാംശം *     cos_theta,
                                    ദശാംശം *     sin_theta) {
    ദശാംശം theta_interp = freq_scale * theta_extrap;
    ദശാംശം theta        = theta_interp;

    എങ്കിൽ (ext_factor != 0.0f) {
        ദശാംശം ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (പൂർണ്ണസംഖ്യ) i0) * ext_factor;
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
സ്ഥിതം വരിയിൽ ശൂന്യം compute_rope_cache(ദശാംശം *       cos_cache,
                                      ദശാംശം *       sin_cache,
                                      int32_t       n_dims,
                                      ദശാംശം         theta_scale,
                                      int32_t       pos,
                                      സ്ഥിരം ദശാംശം * freq_factors,
                                      ദശാംശം         freq_scale,
                                      സ്ഥിരം ദശാംശം   corr_dims[2],
                                      ദശാംശം         ext_factor,
                                      ദശാംശം         attn_factor) {
    സ്ഥിരം int32_t half_dims = n_dims / 2;
    ദശാംശം         theta     = 1.0f;

    int32_t dim_idx = 0;

    വേണ്ടി (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        ദശാംശം theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        ദശാംശം theta_local = theta;
        ദശാംശം mscale      = attn_factor;

        എങ്കിൽ (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ROPE_VEC_WIDTH; ++i) {
            സ്ഥിരം int32_t pair_idx     = dim_idx + i;
            സ്ഥിരം ദശാംശം   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            സ്ഥിരം ദശാംശം   theta_base   = (ദശാംശം) pos * theta_local;
            സ്ഥിരം ദശാംശം   theta_extrap = et_fdiv(theta_base, ff);

            ദശാംശം theta_interp = freq_scale * theta_extrap;
            ദശാംശം theta_mix    = theta_interp;

            എങ്കിൽ (ext_factor != 0.0f) {
                ദശാംശം ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_block[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_block);

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    വേണ്ടി (; dim_idx < half_dims; ++dim_idx) {
        സ്ഥിരം ദശാംശം ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        സ്ഥിരം ദശാംശം theta_base = (ദശാംശം) pos * theta;

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
സ്ഥിതം വരിയിൽ ശൂന്യം compute_imrope_cache(ദശാംശം *       cos_cache,
                                        ദശാംശം *       sin_cache,
                                        int32_t       n_dims,
                                        ദശാംശം         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        സ്ഥിരം int32_t sections[4],
                                        സ്ഥിരം ദശാംശം * freq_factors,
                                        ദശാംശം         freq_scale,
                                        സ്ഥിരം ദശാംശം   corr_dims[2],
                                        ദശാംശം         ext_factor,
                                        ദശാംശം         attn_factor) {
    സ്ഥിരം int32_t half_dims = n_dims / 2;
    സ്ഥിരം int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    ദശാംശം theta_t = (ദശാംശം) pos_t;
    ദശാംശം theta_h = (ദശാംശം) pos_h;
    ദശാംശം theta_w = (ദശാംശം) pos_w;
    ദശാംശം theta_e = (ദശാംശം) pos_e;

    int32_t dim_idx = 0;

    വേണ്ടി (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        ദശാംശം theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        ദശാംശം mscale = attn_factor;

        എങ്കിൽ (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ROPE_VEC_WIDTH; ++i) {
            സ്ഥിരം int32_t pair_idx = dim_idx + i;
            സ്ഥിരം int32_t sector   = pair_idx % sect_dims;
            സ്ഥിരം ദശാംശം   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            ദശാംശം theta;
            എങ്കിൽ (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } അല്ലെങ്കിൽ എങ്കിൽ (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } അല്ലെങ്കിൽ എങ്കിൽ (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } അല്ലെങ്കിൽ {
                theta = theta_e;
            }

            സ്ഥിരം ദശാംശം theta_extrap = et_fdiv(theta, ff);
            ദശാംശം       theta_interp = freq_scale * theta_extrap;
            ദശാംശം       theta_mix    = theta_interp;

            എങ്കിൽ (ext_factor != 0.0f) {
                ദശാംശം ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
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

        വേണ്ടി (പൂർണ്ണസംഖ്യ i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    വേണ്ടി (; dim_idx < half_dims; ++dim_idx) {
        സ്ഥിരം int32_t sector = dim_idx % sect_dims;
        സ്ഥിരം ദശാംശം   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        ദശാംശം theta;
        എങ്കിൽ (sector % 3 == 1 && sector < 3 * sections[1]) {
            theta = theta_h;
        } അല്ലെങ്കിൽ എങ്കിൽ (sector % 3 == 2 && sector < 3 * sections[2]) {
            theta = theta_w;
        } അല്ലെങ്കിൽ എങ്കിൽ (sector % 3 == 0 && sector < 3 * sections[0]) {
            theta = theta_t;
        } അല്ലെങ്കിൽ {
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

പൂർണ്ണസംഖ്യ entry_point(ഘടന ggml_et_rope_params * params, ശൂന്യം * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    എങ്കിൽ (!kernel_env) {
        തിരികെ -1;
    }

    പൂർണ്ണസംഖ്യ thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    പൂർണ്ണസംഖ്യ അക്കം_threads = get_അക്കം_threads(kernel_env->shire_mask);

    എങ്കിൽ (thread_id < 0) {
        തിരികെ -1;
    }

    എങ്കിൽ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        തിരികെ -1;
    }

    ഘടന ggml_tensor * src0 = &params->src0;
    ഘടന ggml_tensor * src1 = &params->src1;
    ഘടന ggml_tensor * src2 = &params->src2;
    ഘടന ggml_tensor * dst  = &params->dst;

    എങ്കിൽ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        തിരികെ -1;
    }

    സ്ഥിരം ദശാംശം *   src0_data    = (സ്ഥിരം ദശാംശം *) src0->data;
    സ്ഥിരം int32_t * src1_data    = (സ്ഥിരം int32_t *) src1->data;
    സ്ഥിരം ദശാംശം *   freq_factors = (src2 && src2->data) ? (സ്ഥിരം ദശാംശം *) src2->data : ശൂന്യ;
    ദശാംശം *         dst_data     = (ദശാംശം *) dst->data;

    എങ്കിൽ (!src0_data || !src1_data || !dst_data) {
        തിരികെ -1;
    }
#ifdef ET_UBERKERNEL
    സ്ഥിരം size_t src0_bytes = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    സ്ഥിരം size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_data, src0_bytes);
    evict_region_past_l2(src1_data, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    സ്ഥിരം int64_t head_dim = src0->ne[0];
    സ്ഥിരം int64_t heads    = src0->ne[1];
    സ്ഥിരം int64_t seq_len  = src0->ne[2];
    സ്ഥിരം int64_t batch    = src0->ne[3];

    സ്ഥിരം rope_params_t * rope_params = &params->rope_params;
    സ്ഥിരം int32_t         n_dims      = rope_params->n_dims;
    സ്ഥിരം ദശാംശം           freq_base   = rope_params->freq_base;
    സ്ഥിരം ദശാംശം           freq_scale  = rope_params->freq_scale;
    സ്ഥിരം int32_t         mode        = rope_params->mode;

    എങ്കിൽ (n_dims <= 0 || n_dims > head_dim || (n_dims & 1) != 0) {
        തിരികെ -1;
    }

    എങ്കിൽ (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        തിരികെ -1;
    }

    ദശാംശം cos_cache[MAX_ROPE_HALF_DIMS];
    ദശാംശം sin_cache[MAX_ROPE_HALF_DIMS];

    ദശാംശം corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_base, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    സ്ഥിരം int64_t total_heads = batch * seq_len * heads;
    സ്ഥിരം int64_t start_wu    = (total_heads * thread_id) / അക്കം_threads;
    സ്ഥിരം int64_t end_wu      = (total_heads * (thread_id + 1)) / അക്കം_threads;

    എങ്കിൽ (start_wu >= end_wu) {
        തിരികെ 0;
    }

    സ്ഥിരം ദശാംശം   theta_scale       = et_powf(freq_base, et_fdiv(-2.0f, (ദശാംശം) n_dims));
    സ്ഥിരം int32_t half_dims         = n_dims / 2;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     is_neox           = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    സ്ഥിരം പൂർണ്ണസംഖ്യ     is_imrope         = (mode == GGML_ROPE_TYPE_IMROPE);
    സ്ഥിരം പൂർണ്ണസംഖ്യ     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t last_pos   = -1;
    int32_t last_pos_h = -1;
    int32_t last_pos_w = -1;
    int32_t last_pos_e = -1;

    വേണ്ടി (int64_t wu = start_wu; wu < end_wu; ++wu) {
        സ്ഥിരം int64_t h = wu % heads;
        സ്ഥിരം int64_t s = (wu / heads) % seq_len;
        സ്ഥിരം int64_t b = wu / (heads * seq_len);

        എങ്കിൽ (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            സ്ഥിരം int32_t pt = src1_data[s] + rope_params->n_past;
            സ്ഥിരം int32_t ph = src1_data[s + seq_len] + rope_params->n_past;
            സ്ഥിരം int32_t pw = src1_data[s + seq_len * 2] + rope_params->n_past;
            സ്ഥിരം int32_t pe = src1_data[s + seq_len * 3] + rope_params->n_past;

            എങ്കിൽ (pt != last_pos || ph != last_pos_h || pw != last_pos_w || pe != last_pos_e) {
                compute_imrope_cache(cos_cache, sin_cache, n_dims, theta_scale, pt, ph, pw, pe, rope_params->sections,
                                     freq_factors, freq_scale, corr_dims, rope_params->ext_factor,
                                     rope_params->attn_factor);
                last_pos   = pt;
                last_pos_h = ph;
                last_pos_w = pw;
                last_pos_e = pe;
            }
        } അല്ലെങ്കിൽ {
            സ്ഥിരം int32_t pos = src1_data[s] + rope_params->n_past;

            എങ്കിൽ (pos != last_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_factor, rope_params->attn_factor);
                last_pos = pos;
            }
        }

        സ്ഥിരം ദശാംശം * head_src =
            (സ്ഥിരം ദശാംശം *) ((സ്ഥിരം അക്ഷരം *) src0_data + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        ദശാംശം * head_dst = (ദശാംശം *) ((അക്ഷരം *) dst_data + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        വേണ്ടി (int64_t d = n_dims; d < head_dim; ++d) {
            head_dst[d] = head_src[d];
        }

        എങ്കിൽ (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t temp_mask;
            __asm__ അസ്ഥിര("mova.x.m %0" : "=r"(temp_mask));
            __asm__ അസ്ഥിര("mov.m.x m0, x0, 0xFF");

            വേണ്ടി (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                __asm__ അസ്ഥിര(
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
                    : [x0_dst] "=m"(*(ദശാംശം (*)[8]) & head_dst[dim_idx]), [x1_dst] "=m"(*(ദശാംശം (*)[8]) &
                                                                                        head_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & head_src[dim_idx]),
                      [x1_src] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & head_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(സ്ഥിരം ദശാംശം (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ അസ്ഥിര("mova.m.x %0" ::"r"(temp_mask));
        } അല്ലെങ്കിൽ {
            // Standard: adjacent pairs (2i, 2i+1)
            വേണ്ടി (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                സ്ഥിരം int32_t dim_in_head = pair_idx * 2;
                സ്ഥിരം ദശാംശം   x0          = head_src[dim_in_head];
                സ്ഥിരം ദശാംശം   x1          = head_src[dim_in_head + 1];

                head_dst[dim_in_head]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                head_dst[dim_in_head + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    തിരികെ 0;
}
