//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#অন্তর্ভাব "ggml_tensor.h"
#অন্তর্ভাব "math_fp.h"
#অন্তর্ভাব "platform.h"

#অন্তর্ভাব <etsoc/common/utils.h>
#অন্তর্ভাব <stdint.h>

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
প্রকারলেখ কাঠামো {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t mode;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    দশমিক   freq_base;    // Base frequency (usually 10000.0f)
    দশমিক   freq_scale;   // Frequency scaling factor
    দশমিক   ext_factor;   // Extension factor for YaRN
    দশমিক   attn_factor;  // Attention factor for YaRN
    দশমিক   beta_fast;    // Fast beta for YaRN
    দশমিক   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
কাঠামো ggml_et_rope_params {
    কাঠামো ggml_tensor src0;  // F32 input tensor
    কাঠামো ggml_tensor src1;  // I32 position tensor
    কাঠামো ggml_tensor src2;  // F32 frequency factors (optional)
    কাঠামো ggml_tensor dst;   // F32 output tensor
    rope_params_t      rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

// floor/ceil with ±inf and NaN passthrough.
স্থির অন্তঃরেখ দশমিক rope_floorf(দশমিক x) {
    সংঘ {
        দশমিক    f;
        uint32_t u;
    } v = { .f = x };

    ধ্রুবক uint32_t expo = (v.u >> 23) & 0xFF;
    যদি (expo == 0xFF) {
        ফেরত x;  // inf or NaN
    }
    যদি (expo >= 23 + 127) {
        ফেরত x;  // already integer-valued
    }
    পূর্ণসংখ্যা i = (পূর্ণসংখ্যা) x;
    ফেরত (x < 0.0f && (দশমিক) i != x) ? (দশমিক) (i - 1) : (দশমিক) i;
}

স্থির অন্তঃরেখ দশমিক rope_ceilf(দশমিক x) {
    সংঘ {
        দশমিক    f;
        uint32_t u;
    } v = { .f = x };

    ধ্রুবক uint32_t expo = (v.u >> 23) & 0xFF;
    যদি (expo == 0xFF) {
        ফেরত x;  // inf or NaN
    }
    যদি (expo >= 23 + 127) {
        ফেরত x;  // already integer-valued
    }
    পূর্ণসংখ্যা i = (পূর্ণসংখ্যা) x;
    ফেরত (x > 0.0f && (দশমিক) i != x) ? (দশমিক) (i + 1) : (দশমিক) i;
}

স্থির অন্তঃরেখ দশমিক rope_yarn_ramp(ধ্রুবক দশমিক নিচের, ধ্রুবক দশমিক উপরের, ধ্রুবক পূর্ণসংখ্যা i0) {
    দশমিক denom = উপরের - নিচের;
    যদি (denom < 0.001f) {
        denom = 0.001f;
    }

    ধ্রুবক দশমিক y       = et_fdiv((দশমিক) (i0 / 2) - নিচের, denom);
    ধ্রুবক দশমিক clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    ফেরত 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
স্থির অন্তঃরেখ দশমিক rope_yarn_corr_dim(পূর্ণসংখ্যা n_dims, পূর্ণসংখ্যা n_ctx_orig, দশমিক beta, দশমিক freq_base) {
    ফেরত (দশমিক) n_dims *
           et_fdiv(et_logf(et_fdiv((দশমিক) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_base));
}

স্থির অন্তঃরেখ শূন্য rope_yarn_corr_dims(পূর্ণসংখ্যা   n_dims,
                                       পূর্ণসংখ্যা   n_ctx_orig,
                                       দশমিক freq_base,
                                       দশমিক beta_fast,
                                       দশমিক beta_slow,
                                       দশমিক dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    দশমিক start = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    দশমিক end   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));

    dims[0] = start > 0.0f ? start : 0.0f;
    dims[1] = end < (দশমিক) (n_dims - 1) ? end : (দশমিক) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

স্থির ধ্রুবক দশমিক rope_ps_one[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
স্থির ধ্রুবক দশমিক rope_ps_c3[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
স্থির ধ্রুবক দশমিক rope_ps_c5[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
স্থির ধ্রুবক দশমিক rope_ps_c7[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
স্থির ধ্রুবক দশমিক rope_ps_c9[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
স্থির ধ্রুবক দশমিক rope_ps_c11[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

স্থির অন্তঃরেখ uint64_t rope_ps_enter_fullmask(শূন্য) {
    uint64_t old_mask;
    __asm__ অস্থির(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_mask)
        :
        : "t0", "memory");
    ফেরত old_mask;
}

স্থির অন্তঃরেখ শূন্য rope_ps_leave_fullmask(uint64_t old_mask) {
    __asm__ অস্থির("mova.m.x %0           \n\t" : : "r"(old_mask) : "memory");
}

স্থির অন্তঃরেখ শূন্য rope_poly_sin_block8(দশমিক * out, ধ্রুবক দশমিক * x) {
    __asm__ অস্থির(
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
        : [out] "=m"(*(দশমিক (*)[ROPE_VEC_WIDTH]) out)
        : [x] "m"(*(ধ্রুবক দশমিক (*)[ROPE_VEC_WIDTH]) x), [one] "m"(*(ধ্রুবক দশমিক (*)[ROPE_VEC_WIDTH]) rope_ps_one),
          [c3] "m"(*(ধ্রুবক দশমিক (*)[ROPE_VEC_WIDTH]) rope_ps_c3),
          [c5] "m"(*(ধ্রুবক দশমিক (*)[ROPE_VEC_WIDTH]) rope_ps_c5),
          [c7] "m"(*(ধ্রুবক দশমিক (*)[ROPE_VEC_WIDTH]) rope_ps_c7),
          [c9] "m"(*(ধ্রুবক দশমিক (*)[ROPE_VEC_WIDTH]) rope_ps_c9),
          [c11] "m"(*(ধ্রুবক দশমিক (*)[ROPE_VEC_WIDTH]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

স্থির অন্তঃরেখ শূন্য rope_sincos_block8(দশমিক * sin8, দশমিক * cos8, ধ্রুবক দশমিক * theta8) {
    দশমিক sin_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    দশমিক cos_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    দশমিক sin_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    দশমিক cos_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));

    জন্য (পূর্ণসংখ্যা i = 0; i < ROPE_VEC_WIDTH; ++i) {
        দশমিক x = theta8[i];

        যদি (x > ROPE_PI || x < -ROPE_PI) {
            দশমিক cycles = x * ROPE_INV_TWO_PI;
            পূর্ণসংখ্যা   n      = (পূর্ণসংখ্যা) cycles;
            যদি (x < 0.0f) {
                n--;
            }
            x = x - (দশমিক) n * ROPE_TWO_PI;
        }

        {
            দশমিক y = x;
            দশমিক s = 1.0f;
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
            দশমিক y = x + ROPE_PI_OVER_2;
            যদি (y > ROPE_PI || y < -ROPE_PI) {
                দশমিক cycles = y * ROPE_INV_TWO_PI;
                পূর্ণসংখ্যা   n      = (পূর্ণসংখ্যা) cycles;
                যদি (y < 0.0f) {
                    n--;
                }
                y = y - (দশমিক) n * ROPE_TWO_PI;
            }

            দশমিক s = 1.0f;
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
        ধ্রুবক uint64_t saved_mask = rope_ps_enter_fullmask();

        rope_poly_sin_block8(sin8, sin_fold);
        rope_poly_sin_block8(cos8, cos_fold);

        __asm__ অস্থির(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(দশমিক (*)[ROPE_VEC_WIDTH]) sin8), [cout] "=m"(*(দশমিক (*)[ROPE_VEC_WIDTH]) cos8)
            : [sinv] "m"(*(ধ্রুবক দশমিক (*)[ROPE_VEC_WIDTH]) sin8),
              [sinsgn] "m"(*(ধ্রুবক দশমিক (*)[ROPE_VEC_WIDTH]) sin_sign),
              [cosv] "m"(*(ধ্রুবক দশমিক (*)[ROPE_VEC_WIDTH]) cos8),
              [cossgn] "m"(*(ধ্রুবক দশমিক (*)[ROPE_VEC_WIDTH]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_mask);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
স্থির অন্তঃরেখ শূন্য rope_yarn_scalar(দশমিক       theta_extrap,
                                    দশমিক       freq_scale,
                                    ধ্রুবক দশমিক corr_dims[2],
                                    int64_t     i0,
                                    দশমিক       ext_factor,
                                    দশমিক       mscale,
                                    দশমিক *     cos_theta,
                                    দশমিক *     sin_theta) {
    দশমিক theta_interp = freq_scale * theta_extrap;
    দশমিক theta        = theta_interp;

    যদি (ext_factor != 0.0f) {
        দশমিক ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (পূর্ণসংখ্যা) i0) * ext_factor;
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
স্থির অন্তঃরেখ শূন্য compute_rope_cache(দশমিক *       cos_cache,
                                      দশমিক *       sin_cache,
                                      int32_t       n_dims,
                                      দশমিক         theta_scale,
                                      int32_t       pos,
                                      ধ্রুবক দশমিক * freq_factors,
                                      দশমিক         freq_scale,
                                      ধ্রুবক দশমিক   corr_dims[2],
                                      দশমিক         ext_factor,
                                      দশমিক         attn_factor) {
    ধ্রুবক int32_t half_dims = n_dims / 2;
    দশমিক         theta     = 1.0f;

    int32_t dim_idx = 0;

    জন্য (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        দশমিক theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        দশমিক theta_local = theta;
        দশমিক mscale      = attn_factor;

        যদি (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        জন্য (পূর্ণসংখ্যা i = 0; i < ROPE_VEC_WIDTH; ++i) {
            ধ্রুবক int32_t pair_idx     = dim_idx + i;
            ধ্রুবক দশমিক   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            ধ্রুবক দশমিক   theta_base   = (দশমিক) pos * theta_local;
            ধ্রুবক দশমিক   theta_extrap = et_fdiv(theta_base, ff);

            দশমিক theta_interp = freq_scale * theta_extrap;
            দশমিক theta_mix    = theta_interp;

            যদি (ext_factor != 0.0f) {
                দশমিক ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_block[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_block);

        জন্য (পূর্ণসংখ্যা i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    জন্য (; dim_idx < half_dims; ++dim_idx) {
        ধ্রুবক দশমিক ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        ধ্রুবক দশমিক theta_base = (দশমিক) pos * theta;

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
স্থির অন্তঃরেখ শূন্য compute_imrope_cache(দশমিক *       cos_cache,
                                        দশমিক *       sin_cache,
                                        int32_t       n_dims,
                                        দশমিক         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        ধ্রুবক int32_t sections[4],
                                        ধ্রুবক দশমিক * freq_factors,
                                        দশমিক         freq_scale,
                                        ধ্রুবক দশমিক   corr_dims[2],
                                        দশমিক         ext_factor,
                                        দশমিক         attn_factor) {
    ধ্রুবক int32_t half_dims = n_dims / 2;
    ধ্রুবক int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    দশমিক theta_t = (দশমিক) pos_t;
    দশমিক theta_h = (দশমিক) pos_h;
    দশমিক theta_w = (দশমিক) pos_w;
    দশমিক theta_e = (দশমিক) pos_e;

    int32_t dim_idx = 0;

    জন্য (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        দশমিক theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        দশমিক mscale = attn_factor;

        যদি (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        জন্য (পূর্ণসংখ্যা i = 0; i < ROPE_VEC_WIDTH; ++i) {
            ধ্রুবক int32_t pair_idx = dim_idx + i;
            ধ্রুবক int32_t sector   = pair_idx % sect_dims;
            ধ্রুবক দশমিক   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            দশমিক theta;
            যদি (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } অন্যথা যদি (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } অন্যথা যদি (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } অন্যথা {
                theta = theta_e;
            }

            ধ্রুবক দশমিক theta_extrap = et_fdiv(theta, ff);
            দশমিক       theta_interp = freq_scale * theta_extrap;
            দশমিক       theta_mix    = theta_interp;

            যদি (ext_factor != 0.0f) {
                দশমিক ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
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

        জন্য (পূর্ণসংখ্যা i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    জন্য (; dim_idx < half_dims; ++dim_idx) {
        ধ্রুবক int32_t sector = dim_idx % sect_dims;
        ধ্রুবক দশমিক   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        দশমিক theta;
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

পূর্ণসংখ্যা entry_point(কাঠামো ggml_et_rope_params * params, শূন্য * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    যদি (!kernel_env) {
        ফেরত -1;
    }

    পূর্ণসংখ্যা thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    পূর্ণসংখ্যা অঙ্ক_threads = get_অঙ্ক_threads(kernel_env->shire_mask);

    যদি (thread_id < 0) {
        ফেরত -1;
    }

    যদি (params == 0 || ((uint64_t) params & 0x7) != 0) {
        ফেরত -1;
    }

    কাঠামো ggml_tensor * src0 = &params->src0;
    কাঠামো ggml_tensor * src1 = &params->src1;
    কাঠামো ggml_tensor * src2 = &params->src2;
    কাঠামো ggml_tensor * dst  = &params->dst;

    যদি (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        ফেরত -1;
    }

    ধ্রুবক দশমিক *   src0_তথ্য    = (ধ্রুবক দশমিক *) src0->তথ্য;
    ধ্রুবক int32_t * src1_তথ্য    = (ধ্রুবক int32_t *) src1->তথ্য;
    ধ্রুবক দশমিক *   freq_factors = (src2 && src2->তথ্য) ? (ধ্রুবক দশমিক *) src2->তথ্য : ফাঁকা;
    দশমিক *         dst_তথ্য     = (দশমিক *) dst->তথ্য;

    যদি (!src0_তথ্য || !src1_তথ্য || !dst_তথ্য) {
        ফেরত -1;
    }
#ifdef ET_UBERKERNEL
    ধ্রুবক size_t src0_bytes = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ধ্রুবক size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_তথ্য, src0_bytes);
    evict_region_past_l2(src1_তথ্য, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    ধ্রুবক int64_t শীর্ষ_dim = src0->ne[0];
    ধ্রুবক int64_t heads    = src0->ne[1];
    ধ্রুবক int64_t seq_len  = src0->ne[2];
    ধ্রুবক int64_t batch    = src0->ne[3];

    ধ্রুবক rope_params_t * rope_params = &params->rope_params;
    ধ্রুবক int32_t         n_dims      = rope_params->n_dims;
    ধ্রুবক দশমিক           freq_base   = rope_params->freq_base;
    ধ্রুবক দশমিক           freq_scale  = rope_params->freq_scale;
    ধ্রুবক int32_t         mode        = rope_params->mode;

    যদি (n_dims <= 0 || n_dims > শীর্ষ_dim || (n_dims & 1) != 0) {
        ফেরত -1;
    }

    যদি (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        ফেরত -1;
    }

    দশমিক cos_cache[MAX_ROPE_HALF_DIMS];
    দশমিক sin_cache[MAX_ROPE_HALF_DIMS];

    দশমিক corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_base, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    ধ্রুবক int64_t যোগ_heads = batch * seq_len * heads;
    ধ্রুবক int64_t start_wu    = (যোগ_heads * thread_id) / অঙ্ক_threads;
    ধ্রুবক int64_t end_wu      = (যোগ_heads * (thread_id + 1)) / অঙ্ক_threads;

    যদি (start_wu >= end_wu) {
        ফেরত 0;
    }

    ধ্রুবক দশমিক   theta_scale       = et_powf(freq_base, et_fdiv(-2.0f, (দশমিক) n_dims));
    ধ্রুবক int32_t half_dims         = n_dims / 2;
    ধ্রুবক পূর্ণসংখ্যা     is_neox           = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    ধ্রুবক পূর্ণসংখ্যা     is_imrope         = (mode == GGML_ROPE_TYPE_IMROPE);
    ধ্রুবক পূর্ণসংখ্যা     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t last_pos   = -1;
    int32_t last_pos_h = -1;
    int32_t last_pos_w = -1;
    int32_t last_pos_e = -1;

    জন্য (int64_t wu = start_wu; wu < end_wu; ++wu) {
        ধ্রুবক int64_t h = wu % heads;
        ধ্রুবক int64_t s = (wu / heads) % seq_len;
        ধ্রুবক int64_t b = wu / (heads * seq_len);

        যদি (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            ধ্রুবক int32_t pt = src1_তথ্য[s] + rope_params->n_past;
            ধ্রুবক int32_t ph = src1_তথ্য[s + seq_len] + rope_params->n_past;
            ধ্রুবক int32_t pw = src1_তথ্য[s + seq_len * 2] + rope_params->n_past;
            ধ্রুবক int32_t pe = src1_তথ্য[s + seq_len * 3] + rope_params->n_past;

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
            ধ্রুবক int32_t pos = src1_তথ্য[s] + rope_params->n_past;

            যদি (pos != last_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_factor, rope_params->attn_factor);
                last_pos = pos;
            }
        }

        ধ্রুবক দশমিক * শীর্ষ_src =
            (ধ্রুবক দশমিক *) ((ধ্রুবক অক্ষর *) src0_তথ্য + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        দশমিক * শীর্ষ_dst = (দশমিক *) ((অক্ষর *) dst_তথ্য + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        জন্য (int64_t d = n_dims; d < শীর্ষ_dim; ++d) {
            শীর্ষ_dst[d] = শীর্ষ_src[d];
        }

        যদি (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t অস্থায়ী_mask;
            __asm__ অস্থির("mova.x.m %0" : "=r"(অস্থায়ী_mask));
            __asm__ অস্থির("mov.m.x m0, x0, 0xFF");

            জন্য (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                __asm__ অস্থির(
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
                    : [x0_dst] "=m"(*(দশমিক (*)[8]) & শীর্ষ_dst[dim_idx]), [x1_dst] "=m"(*(দশমিক (*)[8]) &
                                                                                        শীর্ষ_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(ধ্রুবক দশমিক (*)[8]) & শীর্ষ_src[dim_idx]),
                      [x1_src] "m"(*(ধ্রুবক দশমিক (*)[8]) & শীর্ষ_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(ধ্রুবক দশমিক (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(ধ্রুবক দশমিক (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ অস্থির("mova.m.x %0" ::"r"(অস্থায়ী_mask));
        } অন্যথা {
            // Standard: adjacent pairs (2i, 2i+1)
            জন্য (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                ধ্রুবক int32_t dim_in_শীর্ষ = pair_idx * 2;
                ধ্রুবক দশমিক   x0          = শীর্ষ_src[dim_in_শীর্ষ];
                ধ্রুবক দশমিক   x1          = শীর্ষ_src[dim_in_শীর্ষ + 1];

                শীর্ষ_dst[dim_in_শীর্ষ]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                শীর্ষ_dst[dim_in_শীর্ষ + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    ফেরত 0;
}
