//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#شامل "ggml_tensor.h"
#شامل "math_fp.h"
#شامل "platform.h"

#شامل <etsoc/common/utils.h>
#شامل <stdint.h>

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
قسمتعریف ساخت {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t منوال;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    اعشاری   freq_بنیاد;    // Base frequency (usually 10000.0f)
    اعشاری   freq_scale;   // Frequency scaling factor
    اعشاری   ext_ضربی_عدد;   // Extension factor for YaRN
    اعشاری   attn_ضربی_عدد;  // Attention factor for YaRN
    اعشاری   beta_fast;    // Fast beta for YaRN
    اعشاری   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
ساخت ggml_et_rope_params {
    ساخت ggml_tensor src0;  // F32 input tensor
    ساخت ggml_tensor src1;  // I32 position tensor
    ساخت ggml_tensor src2;  // F32 frequency factors (optional)
    ساخت ggml_tensor dst;   // F32 output tensor
    rope_params_t      rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

// floor/ceil with ±inf and NaN passthrough.
ساکن اندرونی اعشاری rope_floorf(اعشاری x) {
    اتحاد {
        اعشاری    f;
        uint32_t u;
    } v = { .f = x };

    ثابت uint32_t expo = (v.u >> 23) & 0xFF;
    اگر (expo == 0xFF) {
        واپس x;  // inf or NaN
    }
    اگر (expo >= 23 + 127) {
        واپس x;  // already integer-valued
    }
    صحیح i = (صحیح) x;
    واپس (x < 0.0f && (اعشاری) i != x) ? (اعشاری) (i - 1) : (اعشاری) i;
}

ساکن اندرونی اعشاری rope_ceilf(اعشاری x) {
    اتحاد {
        اعشاری    f;
        uint32_t u;
    } v = { .f = x };

    ثابت uint32_t expo = (v.u >> 23) & 0xFF;
    اگر (expo == 0xFF) {
        واپس x;  // inf or NaN
    }
    اگر (expo >= 23 + 127) {
        واپس x;  // already integer-valued
    }
    صحیح i = (صحیح) x;
    واپس (x > 0.0f && (اعشاری) i != x) ? (اعشاری) (i + 1) : (اعشاری) i;
}

ساکن اندرونی اعشاری rope_yarn_ramp(ثابت اعشاری کم, ثابت اعشاری بلند, ثابت صحیح i0) {
    اعشاری denom = بلند - کم;
    اگر (denom < 0.001f) {
        denom = 0.001f;
    }

    ثابت اعشاری y       = et_fdiv((اعشاری) (i0 / 2) - کم, denom);
    ثابت اعشاری clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    واپس 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
ساکن اندرونی اعشاری rope_yarn_corr_dim(صحیح n_dims, صحیح n_ctx_orig, اعشاری beta, اعشاری freq_بنیاد) {
    واپس (اعشاری) n_dims *
           et_fdiv(et_logf(et_fdiv((اعشاری) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_بنیاد));
}

ساکن اندرونی خالی rope_yarn_corr_dims(صحیح   n_dims,
                                       صحیح   n_ctx_orig,
                                       اعشاری freq_بنیاد,
                                       اعشاری beta_fast,
                                       اعشاری beta_slow,
                                       اعشاری dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    اعشاری آغاز = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_بنیاد));
    اعشاری انتہا   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_بنیاد));

    dims[0] = آغاز > 0.0f ? آغاز : 0.0f;
    dims[1] = انتہا < (اعشاری) (n_dims - 1) ? انتہا : (اعشاری) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

ساکن ثابت اعشاری rope_ps_one[ROPE_VEC_چوڑائی]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
ساکن ثابت اعشاری rope_ps_c3[ROPE_VEC_چوڑائی]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
ساکن ثابت اعشاری rope_ps_c5[ROPE_VEC_چوڑائی]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
ساکن ثابت اعشاری rope_ps_c7[ROPE_VEC_چوڑائی]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
ساکن ثابت اعشاری rope_ps_c9[ROPE_VEC_چوڑائی]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
ساکن ثابت اعشاری rope_ps_c11[ROPE_VEC_چوڑائی]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

ساکن اندرونی uint64_t rope_ps_enter_fullmask(خالی) {
    uint64_t old_نقاب;
    __asm__ متلون(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_نقاب)
        :
        : "t0", "memory");
    واپس old_نقاب;
}

ساکن اندرونی خالی rope_ps_leave_fullmask(uint64_t old_نقاب) {
    __asm__ متلون("mova.m.x %0           \n\t" : : "r"(old_نقاب) : "memory");
}

ساکن اندرونی خالی rope_poly_sin_block8(اعشاری * برآمد, ثابت اعشاری * x) {
    __asm__ متلون(
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
        : [برآمد] "=m"(*(اعشاری (*)[ROPE_VEC_چوڑائی]) برآمد)
        : [x] "m"(*(ثابت اعشاری (*)[ROPE_VEC_چوڑائی]) x), [one] "m"(*(ثابت اعشاری (*)[ROPE_VEC_چوڑائی]) rope_ps_one),
          [c3] "m"(*(ثابت اعشاری (*)[ROPE_VEC_چوڑائی]) rope_ps_c3),
          [c5] "m"(*(ثابت اعشاری (*)[ROPE_VEC_چوڑائی]) rope_ps_c5),
          [c7] "m"(*(ثابت اعشاری (*)[ROPE_VEC_چوڑائی]) rope_ps_c7),
          [c9] "m"(*(ثابت اعشاری (*)[ROPE_VEC_چوڑائی]) rope_ps_c9),
          [c11] "m"(*(ثابت اعشاری (*)[ROPE_VEC_چوڑائی]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

ساکن اندرونی خالی rope_sincos_block8(اعشاری * sin8, اعشاری * cos8, ثابت اعشاری * theta8) {
    اعشاری sin_fold[ROPE_VEC_چوڑائی] __attribute__((aligned(32)));
    اعشاری cos_fold[ROPE_VEC_چوڑائی] __attribute__((aligned(32)));
    اعشاری sin_sign[ROPE_VEC_چوڑائی] __attribute__((aligned(32)));
    اعشاری cos_sign[ROPE_VEC_چوڑائی] __attribute__((aligned(32)));

    برائے (صحیح i = 0; i < ROPE_VEC_چوڑائی; ++i) {
        اعشاری x = theta8[i];

        اگر (x > ROPE_PI || x < -ROPE_PI) {
            اعشاری ادوار = x * ROPE_INV_TWO_PI;
            صحیح   n      = (صحیح) ادوار;
            اگر (x < 0.0f) {
                n--;
            }
            x = x - (اعشاری) n * ROPE_TWO_PI;
        }

        {
            اعشاری y = x;
            اعشاری s = 1.0f;
            اگر (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } ورنہ اگر (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            اعشاری y = x + ROPE_PI_OVER_2;
            اگر (y > ROPE_PI || y < -ROPE_PI) {
                اعشاری ادوار = y * ROPE_INV_TWO_PI;
                صحیح   n      = (صحیح) ادوار;
                اگر (y < 0.0f) {
                    n--;
                }
                y = y - (اعشاری) n * ROPE_TWO_PI;
            }

            اعشاری s = 1.0f;
            اگر (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } ورنہ اگر (y < -ROPE_PI_OVER_2) {
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

        __asm__ متلون(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(اعشاری (*)[ROPE_VEC_چوڑائی]) sin8), [cout] "=m"(*(اعشاری (*)[ROPE_VEC_چوڑائی]) cos8)
            : [sinv] "m"(*(ثابت اعشاری (*)[ROPE_VEC_چوڑائی]) sin8),
              [sinsgn] "m"(*(ثابت اعشاری (*)[ROPE_VEC_چوڑائی]) sin_sign),
              [cosv] "m"(*(ثابت اعشاری (*)[ROPE_VEC_چوڑائی]) cos8),
              [cossgn] "m"(*(ثابت اعشاری (*)[ROPE_VEC_چوڑائی]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_نقاب);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
ساکن اندرونی خالی rope_yarn_scalar(اعشاری       theta_extrap,
                                    اعشاری       freq_scale,
                                    ثابت اعشاری corr_dims[2],
                                    int64_t     i0,
                                    اعشاری       ext_ضربی_عدد,
                                    اعشاری       mscale,
                                    اعشاری *     cos_theta,
                                    اعشاری *     sin_theta) {
    اعشاری theta_interp = freq_scale * theta_extrap;
    اعشاری theta        = theta_interp;

    اگر (ext_ضربی_عدد != 0.0f) {
        اعشاری ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (صحیح) i0) * ext_ضربی_عدد;
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
ساکن اندرونی خالی compute_rope_cache(اعشاری *       cos_cache,
                                      اعشاری *       sin_cache,
                                      int32_t       n_dims,
                                      اعشاری         theta_scale,
                                      int32_t       pos,
                                      ثابت اعشاری * freq_factors,
                                      اعشاری         freq_scale,
                                      ثابت اعشاری   corr_dims[2],
                                      اعشاری         ext_ضربی_عدد,
                                      اعشاری         attn_ضربی_عدد) {
    ثابت int32_t half_dims = n_dims / 2;
    اعشاری         theta     = 1.0f;

    int32_t dim_idx = 0;

    برائے (; dim_idx + ROPE_VEC_چوڑائی <= half_dims; dim_idx += ROPE_VEC_چوڑائی) {
        اعشاری theta_بلاک[ROPE_VEC_چوڑائی] __attribute__((aligned(32)));
        اعشاری theta_local = theta;
        اعشاری mscale      = attn_ضربی_عدد;

        اگر (ext_ضربی_عدد != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        برائے (صحیح i = 0; i < ROPE_VEC_چوڑائی; ++i) {
            ثابت int32_t pair_idx     = dim_idx + i;
            ثابت اعشاری   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            ثابت اعشاری   theta_بنیاد   = (اعشاری) pos * theta_local;
            ثابت اعشاری   theta_extrap = et_fdiv(theta_بنیاد, ff);

            اعشاری theta_interp = freq_scale * theta_extrap;
            اعشاری theta_mix    = theta_interp;

            اگر (ext_ضربی_عدد != 0.0f) {
                اعشاری ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_ضربی_عدد;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_بلاک[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_بلاک);

        برائے (صحیح i = 0; i < ROPE_VEC_چوڑائی; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    برائے (; dim_idx < half_dims; ++dim_idx) {
        ثابت اعشاری ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        ثابت اعشاری theta_بنیاد = (اعشاری) pos * theta;

        rope_yarn_scalar(et_fdiv(theta_بنیاد, ff), freq_scale, corr_dims, dim_idx * 2, ext_ضربی_عدد, attn_ضربی_عدد,
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
ساکن اندرونی خالی compute_imrope_cache(اعشاری *       cos_cache,
                                        اعشاری *       sin_cache,
                                        int32_t       n_dims,
                                        اعشاری         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        ثابت int32_t sections[4],
                                        ثابت اعشاری * freq_factors,
                                        اعشاری         freq_scale,
                                        ثابت اعشاری   corr_dims[2],
                                        اعشاری         ext_ضربی_عدد,
                                        اعشاری         attn_ضربی_عدد) {
    ثابت int32_t half_dims = n_dims / 2;
    ثابت int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    اعشاری theta_t = (اعشاری) pos_t;
    اعشاری theta_h = (اعشاری) pos_h;
    اعشاری theta_w = (اعشاری) pos_w;
    اعشاری theta_e = (اعشاری) pos_e;

    int32_t dim_idx = 0;

    برائے (; dim_idx + ROPE_VEC_چوڑائی <= half_dims; dim_idx += ROPE_VEC_چوڑائی) {
        اعشاری theta_بلاک[ROPE_VEC_چوڑائی] __attribute__((aligned(32)));
        اعشاری mscale = attn_ضربی_عدد;

        اگر (ext_ضربی_عدد != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        برائے (صحیح i = 0; i < ROPE_VEC_چوڑائی; ++i) {
            ثابت int32_t pair_idx = dim_idx + i;
            ثابت int32_t sector   = pair_idx % sect_dims;
            ثابت اعشاری   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            اعشاری theta;
            اگر (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } ورنہ اگر (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } ورنہ اگر (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } ورنہ {
                theta = theta_e;
            }

            ثابت اعشاری theta_extrap = et_fdiv(theta, ff);
            اعشاری       theta_interp = freq_scale * theta_extrap;
            اعشاری       theta_mix    = theta_interp;

            اگر (ext_ضربی_عدد != 0.0f) {
                اعشاری ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_ضربی_عدد;
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

        برائے (صحیح i = 0; i < ROPE_VEC_چوڑائی; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    برائے (; dim_idx < half_dims; ++dim_idx) {
        ثابت int32_t sector = dim_idx % sect_dims;
        ثابت اعشاری   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        اعشاری theta;
        اگر (sector % 3 == 1 && sector < 3 * sections[1]) {
            theta = theta_h;
        } ورنہ اگر (sector % 3 == 2 && sector < 3 * sections[2]) {
            theta = theta_w;
        } ورنہ اگر (sector % 3 == 0 && sector < 3 * sections[0]) {
            theta = theta_t;
        } ورنہ {
            theta = theta_e;
        }

        rope_yarn_scalar(et_fdiv(theta, ff), freq_scale, corr_dims, dim_idx * 2, ext_ضربی_عدد, attn_ضربی_عدد,
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

صحیح اندراج_نقطہ(ساخت ggml_et_rope_params * params, خالی * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    اگر (!kernel_env) {
        واپس -1;
    }

    صحیح thread_id   = get_relative_thread_id(kernel_env->shire_نقاب);
    صحیح عدد_دھاگے = get_عدد_دھاگے(kernel_env->shire_نقاب);

    اگر (thread_id < 0) {
        واپس -1;
    }

    اگر (params == 0 || ((uint64_t) params & 0x7) != 0) {
        واپس -1;
    }

    ساخت ggml_tensor * src0 = &params->src0;
    ساخت ggml_tensor * src1 = &params->src1;
    ساخت ggml_tensor * src2 = &params->src2;
    ساخت ggml_tensor * dst  = &params->dst;

    اگر (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        واپس -1;
    }

    ثابت اعشاری *   src0_ڈیٹا    = (ثابت اعشاری *) src0->ڈیٹا;
    ثابت int32_t * src1_ڈیٹا    = (ثابت int32_t *) src1->ڈیٹا;
    ثابت اعشاری *   freq_factors = (src2 && src2->ڈیٹا) ? (ثابت اعشاری *) src2->ڈیٹا : تہی;
    اعشاری *         dst_ڈیٹا     = (اعشاری *) dst->ڈیٹا;

    اگر (!src0_ڈیٹا || !src1_ڈیٹا || !dst_ڈیٹا) {
        واپس -1;
    }
#اگر_تعریف_شدہ ET_UBERKERNEL
    ثابت size_t src0_بائٹیں = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    ثابت size_t src1_بائٹیں = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_ڈیٹا, src0_بائٹیں);
    evict_region_past_l2(src1_ڈیٹا, src1_بائٹیں);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#اگر_ختم
    ثابت int64_t سر_dim = src0->ne[0];
    ثابت int64_t heads    = src0->ne[1];
    ثابت int64_t seq_len  = src0->ne[2];
    ثابت int64_t batch    = src0->ne[3];

    ثابت rope_params_t * rope_params = &params->rope_params;
    ثابت int32_t         n_dims      = rope_params->n_dims;
    ثابت اعشاری           freq_بنیاد   = rope_params->freq_بنیاد;
    ثابت اعشاری           freq_scale  = rope_params->freq_scale;
    ثابت int32_t         منوال        = rope_params->منوال;

    اگر (n_dims <= 0 || n_dims > سر_dim || (n_dims & 1) != 0) {
        واپس -1;
    }

    اگر (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        واپس -1;
    }

    اعشاری cos_cache[MAX_ROPE_HALF_DIMS];
    اعشاری sin_cache[MAX_ROPE_HALF_DIMS];

    اعشاری corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_بنیاد, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    ثابت int64_t کل_heads = batch * seq_len * heads;
    ثابت int64_t آغاز_wu    = (کل_heads * thread_id) / عدد_دھاگے;
    ثابت int64_t انتہا_wu      = (کل_heads * (thread_id + 1)) / عدد_دھاگے;

    اگر (آغاز_wu >= انتہا_wu) {
        واپس 0;
    }

    ثابت اعشاری   theta_scale       = et_powf(freq_بنیاد, et_fdiv(-2.0f, (اعشاری) n_dims));
    ثابت int32_t half_dims         = n_dims / 2;
    ثابت صحیح     is_neox           = (منوال & GGML_ROPE_TYPE_NEOX) != 0;
    ثابت صحیح     is_imrope         = (منوال == GGML_ROPE_TYPE_IMROPE);
    ثابت صحیح     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t آخری_pos   = -1;
    int32_t آخری_pos_h = -1;
    int32_t آخری_pos_w = -1;
    int32_t آخری_pos_e = -1;

    برائے (int64_t wu = آغاز_wu; wu < انتہا_wu; ++wu) {
        ثابت int64_t h = wu % heads;
        ثابت int64_t s = (wu / heads) % seq_len;
        ثابت int64_t b = wu / (heads * seq_len);

        اگر (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            ثابت int32_t pt = src1_ڈیٹا[s] + rope_params->n_past;
            ثابت int32_t ph = src1_ڈیٹا[s + seq_len] + rope_params->n_past;
            ثابت int32_t pw = src1_ڈیٹا[s + seq_len * 2] + rope_params->n_past;
            ثابت int32_t pe = src1_ڈیٹا[s + seq_len * 3] + rope_params->n_past;

            اگر (pt != آخری_pos || ph != آخری_pos_h || pw != آخری_pos_w || pe != آخری_pos_e) {
                compute_imrope_cache(cos_cache, sin_cache, n_dims, theta_scale, pt, ph, pw, pe, rope_params->sections,
                                     freq_factors, freq_scale, corr_dims, rope_params->ext_ضربی_عدد,
                                     rope_params->attn_ضربی_عدد);
                آخری_pos   = pt;
                آخری_pos_h = ph;
                آخری_pos_w = pw;
                آخری_pos_e = pe;
            }
        } ورنہ {
            ثابت int32_t pos = src1_ڈیٹا[s] + rope_params->n_past;

            اگر (pos != آخری_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_ضربی_عدد, rope_params->attn_ضربی_عدد);
                آخری_pos = pos;
            }
        }

        ثابت اعشاری * سر_src =
            (ثابت اعشاری *) ((ثابت حرف *) src0_ڈیٹا + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        اعشاری * سر_dst = (اعشاری *) ((حرف *) dst_ڈیٹا + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        برائے (int64_t d = n_dims; d < سر_dim; ++d) {
            سر_dst[d] = سر_src[d];
        }

        اگر (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t عارضی_نقاب;
            __asm__ متلون("mova.x.m %0" : "=r"(عارضی_نقاب));
            __asm__ متلون("mov.m.x m0, x0, 0xFF");

            برائے (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                __asm__ متلون(
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
                    : [x0_dst] "=m"(*(اعشاری (*)[8]) & سر_dst[dim_idx]), [x1_dst] "=m"(*(اعشاری (*)[8]) &
                                                                                        سر_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(ثابت اعشاری (*)[8]) & سر_src[dim_idx]),
                      [x1_src] "m"(*(ثابت اعشاری (*)[8]) & سر_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(ثابت اعشاری (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(ثابت اعشاری (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ متلون("mova.m.x %0" ::"r"(عارضی_نقاب));
        } ورنہ {
            // Standard: adjacent pairs (2i, 2i+1)
            برائے (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                ثابت int32_t dim_اندر_سر = pair_idx * 2;
                ثابت اعشاری   x0          = سر_src[dim_اندر_سر];
                ثابت اعشاری   x1          = سر_src[dim_اندر_سر + 1];

                سر_dst[dim_اندر_سر]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                سر_dst[dim_اندر_سر + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    واپس 0;
}
