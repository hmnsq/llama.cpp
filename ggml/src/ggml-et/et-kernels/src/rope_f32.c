//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#অন্তর্ভাৱ "ggml_tensor.h"
#অন্তর্ভাৱ "math_fp.h"
#অন্তর্ভাৱ "platform.h"

#অন্তর্ভাৱ <etsoc/common/utils.h>
#অন্তর্ভাৱ <stdint.h>

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
প্রকারনির্দেশ সংরচনা {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t mode;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    ভিন্ন   freq_base;    // Base frequency (usually 10000.0f)
    ভিন্ন   freq_scale;   // Frequency scaling factor
    ভিন্ন   ext_factor;   // Extension factor for YaRN
    ভিন্ন   attn_factor;  // Attention factor for YaRN
    ভিন্ন   beta_fast;    // Fast beta for YaRN
    ভিন্ন   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
সংরচনা ggml_et_rope_params {
    সংরচনা ggml_tensor src0;  // F32 input tensor
    সংরচনা ggml_tensor src1;  // I32 position tensor
    সংরচনা ggml_tensor src2;  // F32 frequency factors (optional)
    সংরচনা ggml_tensor dst;   // F32 output tensor
    rope_params_t      rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

// floor/ceil with ±inf and NaN passthrough.
স্থির অন্তঃস্থ ভিন্ন rope_floorf(ভিন্ন x) {
    সঙ্ঘ {
        ভিন্ন    f;
        uint32_t u;
    } v = { .f = x };

    ধ্রুৱ uint32_t expo = (v.u >> 23) & 0xFF;
    যদি (expo == 0xFF) {
        ঘূৰাই x;  // inf or NaN
    }
    যদি (expo >= 23 + 127) {
        ঘূৰাই x;  // already integer-valued
    }
    পূর্ণাঙ্ক i = (পূর্ণাঙ্ক) x;
    ঘূৰাই (x < 0.0f && (ভিন্ন) i != x) ? (ভিন্ন) (i - 1) : (ভিন্ন) i;
}

স্থির অন্তঃস্থ ভিন্ন rope_ceilf(ভিন্ন x) {
    সঙ্ঘ {
        ভিন্ন    f;
        uint32_t u;
    } v = { .f = x };

    ধ্রুৱ uint32_t expo = (v.u >> 23) & 0xFF;
    যদি (expo == 0xFF) {
        ঘূৰাই x;  // inf or NaN
    }
    যদি (expo >= 23 + 127) {
        ঘূৰাই x;  // already integer-valued
    }
    পূর্ণাঙ্ক i = (পূর্ণাঙ্ক) x;
    ঘূৰাই (x > 0.0f && (ভিন্ন) i != x) ? (ভিন্ন) (i + 1) : (ভিন্ন) i;
}

স্থির অন্তঃস্থ ভিন্ন rope_yarn_ramp(ধ্রুৱ ভিন্ন নিচেৰ, ধ্রুৱ ভিন্ন উপৰেৰ, ধ্রুৱ পূর্ণাঙ্ক i0) {
    ভিন্ন denom = উপৰেৰ - নিচেৰ;
    যদি (denom < 0.001f) {
        denom = 0.001f;
    }

    ধ্রুৱ ভিন্ন y       = et_fdiv((ভিন্ন) (i0 / 2) - নিচেৰ, denom);
    ধ্রুৱ ভিন্ন clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    ঘূৰাই 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
স্থির অন্তঃস্থ ভিন্ন rope_yarn_corr_dim(পূর্ণাঙ্ক n_dims, পূর্ণাঙ্ক n_ctx_orig, ভিন্ন beta, ভিন্ন freq_base) {
    ঘূৰাই (ভিন্ন) n_dims *
           et_fdiv(et_logf(et_fdiv((ভিন্ন) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_base));
}

স্থির অন্তঃস্থ শূন্য rope_yarn_corr_dims(পূর্ণাঙ্ক   n_dims,
                                       পূর্ণাঙ্ক   n_ctx_orig,
                                       ভিন্ন freq_base,
                                       ভিন্ন beta_fast,
                                       ভিন্ন beta_slow,
                                       ভিন্ন dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    ভিন্ন start = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    ভিন্ন end   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));

    dims[0] = start > 0.0f ? start : 0.0f;
    dims[1] = end < (ভিন্ন) (n_dims - 1) ? end : (ভিন্ন) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

স্থির ধ্রুৱ ভিন্ন rope_ps_one[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
স্থির ধ্রুৱ ভিন্ন rope_ps_c3[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
স্থির ধ্রুৱ ভিন্ন rope_ps_c5[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
স্থির ধ্রুৱ ভিন্ন rope_ps_c7[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
স্থির ধ্রুৱ ভিন্ন rope_ps_c9[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
স্থির ধ্রুৱ ভিন্ন rope_ps_c11[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

স্থির অন্তঃস্থ uint64_t rope_ps_enter_fullmask(শূন্য) {
    uint64_t old_mask;
    __asm__ চঞ্চল(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_mask)
        :
        : "t0", "memory");
    ঘূৰাই old_mask;
}

স্থির অন্তঃস্থ শূন্য rope_ps_leave_fullmask(uint64_t old_mask) {
    __asm__ চঞ্চল("mova.m.x %0           \n\t" : : "r"(old_mask) : "memory");
}

স্থির অন্তঃস্থ শূন্য rope_poly_sin_block8(ভিন্ন * out, ধ্রুৱ ভিন্ন * x) {
    __asm__ চঞ্চল(
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
        : [out] "=m"(*(ভিন্ন (*)[ROPE_VEC_WIDTH]) out)
        : [x] "m"(*(ধ্রুৱ ভিন্ন (*)[ROPE_VEC_WIDTH]) x), [one] "m"(*(ধ্রুৱ ভিন্ন (*)[ROPE_VEC_WIDTH]) rope_ps_one),
          [c3] "m"(*(ধ্রুৱ ভিন্ন (*)[ROPE_VEC_WIDTH]) rope_ps_c3),
          [c5] "m"(*(ধ্রুৱ ভিন্ন (*)[ROPE_VEC_WIDTH]) rope_ps_c5),
          [c7] "m"(*(ধ্রুৱ ভিন্ন (*)[ROPE_VEC_WIDTH]) rope_ps_c7),
          [c9] "m"(*(ধ্রুৱ ভিন্ন (*)[ROPE_VEC_WIDTH]) rope_ps_c9),
          [c11] "m"(*(ধ্রুৱ ভিন্ন (*)[ROPE_VEC_WIDTH]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

স্থির অন্তঃস্থ শূন্য rope_sincos_block8(ভিন্ন * sin8, ভিন্ন * cos8, ধ্রুৱ ভিন্ন * theta8) {
    ভিন্ন sin_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    ভিন্ন cos_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    ভিন্ন sin_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    ভিন্ন cos_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));

    বাবে (পূর্ণাঙ্ক i = 0; i < ROPE_VEC_WIDTH; ++i) {
        ভিন্ন x = theta8[i];

        যদি (x > ROPE_PI || x < -ROPE_PI) {
            ভিন্ন cycles = x * ROPE_INV_TWO_PI;
            পূর্ণাঙ্ক   n      = (পূর্ণাঙ্ক) cycles;
            যদি (x < 0.0f) {
                n--;
            }
            x = x - (ভিন্ন) n * ROPE_TWO_PI;
        }

        {
            ভিন্ন y = x;
            ভিন্ন s = 1.0f;
            যদি (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } অন্যথা যদি (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            ভিন্ন y = x + ROPE_PI_OVER_2;
            যদি (y > ROPE_PI || y < -ROPE_PI) {
                ভিন্ন cycles = y * ROPE_INV_TWO_PI;
                পূর্ণাঙ্ক   n      = (পূর্ণাঙ্ক) cycles;
                যদি (y < 0.0f) {
                    n--;
                }
                y = y - (ভিন্ন) n * ROPE_TWO_PI;
            }

            ভিন্ন s = 1.0f;
            যদি (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } অন্যথা যদি (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            cos_fold[i] = y;
            cos_sign[i] = s;
        }
    }

    {
        ধ্রুৱ uint64_t saved_mask = rope_ps_enter_fullmask();

        rope_poly_sin_block8(sin8, sin_fold);
        rope_poly_sin_block8(cos8, cos_fold);

        __asm__ চঞ্চল(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(ভিন্ন (*)[ROPE_VEC_WIDTH]) sin8), [cout] "=m"(*(ভিন্ন (*)[ROPE_VEC_WIDTH]) cos8)
            : [sinv] "m"(*(ধ্রুৱ ভিন্ন (*)[ROPE_VEC_WIDTH]) sin8),
              [sinsgn] "m"(*(ধ্রুৱ ভিন্ন (*)[ROPE_VEC_WIDTH]) sin_sign),
              [cosv] "m"(*(ধ্রুৱ ভিন্ন (*)[ROPE_VEC_WIDTH]) cos8),
              [cossgn] "m"(*(ধ্রুৱ ভিন্ন (*)[ROPE_VEC_WIDTH]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_mask);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
স্থির অন্তঃস্থ শূন্য rope_yarn_scalar(ভিন্ন       theta_extrap,
                                    ভিন্ন       freq_scale,
                                    ধ্রুৱ ভিন্ন corr_dims[2],
                                    int64_t     i0,
                                    ভিন্ন       ext_factor,
                                    ভিন্ন       mscale,
                                    ভিন্ন *     cos_theta,
                                    ভিন্ন *     sin_theta) {
    ভিন্ন theta_interp = freq_scale * theta_extrap;
    ভিন্ন theta        = theta_interp;

    যদি (ext_factor != 0.0f) {
        ভিন্ন ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (পূর্ণাঙ্ক) i0) * ext_factor;
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
স্থির অন্তঃস্থ শূন্য compute_rope_cache(ভিন্ন *       cos_cache,
                                      ভিন্ন *       sin_cache,
                                      int32_t       n_dims,
                                      ভিন্ন         theta_scale,
                                      int32_t       pos,
                                      ধ্রুৱ ভিন্ন * freq_factors,
                                      ভিন্ন         freq_scale,
                                      ধ্রুৱ ভিন্ন   corr_dims[2],
                                      ভিন্ন         ext_factor,
                                      ভিন্ন         attn_factor) {
    ধ্রুৱ int32_t half_dims = n_dims / 2;
    ভিন্ন         theta     = 1.0f;

    int32_t dim_idx = 0;

    বাবে (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        ভিন্ন theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        ভিন্ন theta_local = theta;
        ভিন্ন mscale      = attn_factor;

        যদি (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        বাবে (পূর্ণাঙ্ক i = 0; i < ROPE_VEC_WIDTH; ++i) {
            ধ্রুৱ int32_t pair_idx     = dim_idx + i;
            ধ্রুৱ ভিন্ন   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            ধ্রুৱ ভিন্ন   theta_base   = (ভিন্ন) pos * theta_local;
            ধ্রুৱ ভিন্ন   theta_extrap = et_fdiv(theta_base, ff);

            ভিন্ন theta_interp = freq_scale * theta_extrap;
            ভিন্ন theta_mix    = theta_interp;

            যদি (ext_factor != 0.0f) {
                ভিন্ন ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_block[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_block);

        বাবে (পূর্ণাঙ্ক i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    বাবে (; dim_idx < half_dims; ++dim_idx) {
        ধ্রুৱ ভিন্ন ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        ধ্রুৱ ভিন্ন theta_base = (ভিন্ন) pos * theta;

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
স্থির অন্তঃস্থ শূন্য compute_imrope_cache(ভিন্ন *       cos_cache,
                                        ভিন্ন *       sin_cache,
                                        int32_t       n_dims,
                                        ভিন্ন         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        ধ্রুৱ int32_t sections[4],
                                        ধ্রুৱ ভিন্ন * freq_factors,
                                        ভিন্ন         freq_scale,
                                        ধ্রুৱ ভিন্ন   corr_dims[2],
                                        ভিন্ন         ext_factor,
                                        ভিন্ন         attn_factor) {
    ধ্রুৱ int32_t half_dims = n_dims / 2;
    ধ্রুৱ int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    ভিন্ন theta_t = (ভিন্ন) pos_t;
    ভিন্ন theta_h = (ভিন্ন) pos_h;
    ভিন্ন theta_w = (ভিন্ন) pos_w;
    ভিন্ন theta_e = (ভিন্ন) pos_e;

    int32_t dim_idx = 0;

    বাবে (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        ভিন্ন theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        ভিন্ন mscale = attn_factor;

        যদি (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        বাবে (পূর্ণাঙ্ক i = 0; i < ROPE_VEC_WIDTH; ++i) {
            ধ্রুৱ int32_t pair_idx = dim_idx + i;
            ধ্রুৱ int32_t sector   = pair_idx % sect_dims;
            ধ্রুৱ ভিন্ন   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            ভিন্ন theta;
            যদি (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } অন্যথা যদি (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } অন্যথা যদি (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } অন্যথা {
                theta = theta_e;
            }

            ধ্রুৱ ভিন্ন theta_extrap = et_fdiv(theta, ff);
            ভিন্ন       theta_interp = freq_scale * theta_extrap;
            ভিন্ন       theta_mix    = theta_interp;

            যদি (ext_factor != 0.0f) {
                ভিন্ন ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
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

        বাবে (পূর্ণাঙ্ক i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    বাবে (; dim_idx < half_dims; ++dim_idx) {
        ধ্রুৱ int32_t sector = dim_idx % sect_dims;
        ধ্রুৱ ভিন্ন   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        ভিন্ন theta;
        যদি (sector % 3 == 1 && sector < 3 * sections[1]) {
            theta = theta_h;
        } অন্যথা যদি (sector % 3 == 2 && sector < 3 * sections[2]) {
            theta = theta_w;
        } অন্যথা যদি (sector % 3 == 0 && sector < 3 * sections[0]) {
            theta = theta_t;
        } অন্যথা {
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

পূর্ণাঙ্ক entry_point(সংরচনা ggml_et_rope_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ঘূৰাই -1;
    }

    পূর্ণাঙ্ক thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণাঙ্ক অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ঘূৰাই -1;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ঘূৰাই -1;
    }

    সংরচনা ggml_tensor * src0 = &params->src0;
    সংরচনা ggml_tensor * src1 = &params->src1;
    সংরচনা ggml_tensor * src2 = &params->src2;
    সংরচনা ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        ঘূৰাই -1;
    }

    ধ্রুৱ ভিন্ন *   src0_data    = (ধ্রুৱ ভিন্ন *) src0->data;
    ধ্রুৱ int32_t * src1_data    = (ধ্রুৱ int32_t *) src1->data;
    ধ্রুৱ ভিন্ন *   freq_factors = (src2 && src2->data) ? (ধ্রুৱ ভিন্ন *) src2->data : রিক্ত;
    ভিন্ন *         dst_data     = (ভিন্ন *) dst->data;

    যদি (!src0_data || !src1_data || !dst_data) {
        ঘূৰাই -1;
    }
#ifdef ET_UBERKERNEL
    ধ্রুৱ size_t src0_bytes = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ধ্রুৱ size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_data, src0_bytes);
    evict_region_past_l2(src1_data, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    ধ্রুৱ int64_t head_dim = src0->ne[0];
    ধ্রুৱ int64_t heads    = src0->ne[1];
    ধ্রুৱ int64_t seq_len  = src0->ne[2];
    ধ্রুৱ int64_t batch    = src0->ne[3];

    ধ্রুৱ rope_params_t * rope_params = &params->rope_params;
    ধ্রুৱ int32_t         n_dims      = rope_params->n_dims;
    ধ্রুৱ ভিন্ন           freq_base   = rope_params->freq_base;
    ধ্রুৱ ভিন্ন           freq_scale  = rope_params->freq_scale;
    ধ্রুৱ int32_t         mode        = rope_params->mode;

    যদি (n_dims <= 0 || n_dims > head_dim || (n_dims & 1) != 0) {
        ঘূৰাই -1;
    }

    যদি (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        ঘূৰাই -1;
    }

    ভিন্ন cos_cache[MAX_ROPE_HALF_DIMS];
    ভিন্ন sin_cache[MAX_ROPE_HALF_DIMS];

    ভিন্ন corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_base, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    ধ্রুৱ int64_t total_heads = batch * seq_len * heads;
    ধ্রুৱ int64_t start_wu    = (total_heads * thread_id) / অঙ্ক_threads;
    ধ্রুৱ int64_t end_wu      = (total_heads * (thread_id + 1)) / অঙ্ক_threads;

    যদি (start_wu >= end_wu) {
        ঘূৰাই 0;
    }

    ধ্রুৱ ভিন্ন   theta_scale       = et_powf(freq_base, et_fdiv(-2.0f, (ভিন্ন) n_dims));
    ধ্রুৱ int32_t half_dims         = n_dims / 2;
    ধ্রুৱ পূর্ণাঙ্ক     is_neox           = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    ধ্রুৱ পূর্ণাঙ্ক     is_imrope         = (mode == GGML_ROPE_TYPE_IMROPE);
    ধ্রুৱ পূর্ণাঙ্ক     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t last_pos   = -1;
    int32_t last_pos_h = -1;
    int32_t last_pos_w = -1;
    int32_t last_pos_e = -1;

    বাবে (int64_t wu = start_wu; wu < end_wu; ++wu) {
        ধ্রুৱ int64_t h = wu % heads;
        ধ্রুৱ int64_t s = (wu / heads) % seq_len;
        ধ্রুৱ int64_t b = wu / (heads * seq_len);

        যদি (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            ধ্রুৱ int32_t pt = src1_data[s] + rope_params->n_past;
            ধ্রুৱ int32_t ph = src1_data[s + seq_len] + rope_params->n_past;
            ধ্রুৱ int32_t pw = src1_data[s + seq_len * 2] + rope_params->n_past;
            ধ্রুৱ int32_t pe = src1_data[s + seq_len * 3] + rope_params->n_past;

            যদি (pt != last_pos || ph != last_pos_h || pw != last_pos_w || pe != last_pos_e) {
                compute_imrope_cache(cos_cache, sin_cache, n_dims, theta_scale, pt, ph, pw, pe, rope_params->sections,
                                     freq_factors, freq_scale, corr_dims, rope_params->ext_factor,
                                     rope_params->attn_factor);
                last_pos   = pt;
                last_pos_h = ph;
                last_pos_w = pw;
                last_pos_e = pe;
            }
        } অন্যথা {
            ধ্রুৱ int32_t pos = src1_data[s] + rope_params->n_past;

            যদি (pos != last_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_factor, rope_params->attn_factor);
                last_pos = pos;
            }
        }

        ধ্রুৱ ভিন্ন * head_src =
            (ধ্রুৱ ভিন্ন *) ((ধ্রুৱ ৱর্ণ *) src0_data + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        ভিন্ন * head_dst = (ভিন্ন *) ((ৱর্ণ *) dst_data + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        বাবে (int64_t d = n_dims; d < head_dim; ++d) {
            head_dst[d] = head_src[d];
        }

        যদি (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t temp_mask;
            __asm__ চঞ্চল("mova.x.m %0" : "=r"(temp_mask));
            __asm__ চঞ্চল("mov.m.x m0, x0, 0xFF");

            বাবে (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                __asm__ চঞ্চল(
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
                    : [x0_dst] "=m"(*(ভিন্ন (*)[8]) & head_dst[dim_idx]), [x1_dst] "=m"(*(ভিন্ন (*)[8]) &
                                                                                        head_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & head_src[dim_idx]),
                      [x1_src] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & head_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(ধ্রুৱ ভিন্ন (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ চঞ্চল("mova.m.x %0" ::"r"(temp_mask));
        } অন্যথা {
            // Standard: adjacent pairs (2i, 2i+1)
            বাবে (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                ধ্রুৱ int32_t dim_in_head = pair_idx * 2;
                ধ্রুৱ ভিন্ন   x0          = head_src[dim_in_head];
                ধ্রুৱ ভিন্ন   x1          = head_src[dim_in_head + 1];

                head_dst[dim_in_head]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                head_dst[dim_in_head + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    ঘূৰাই 0;
}
