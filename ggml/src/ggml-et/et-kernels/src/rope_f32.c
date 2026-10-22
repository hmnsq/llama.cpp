//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#அந்தர்பாவ "ggml_tensor.h"
#அந்தர்பாவ "math_fp.h"
#அந்தர்பாவ "platform.h"

#அந்தர்பாவ <etsoc/common/utils.h>
#அந்தர்பாவ <stdint.h>

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
வகைவரை கட்டமைப்பு {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t mode;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    தசம   freq_base;    // Base frequency (usually 10000.0f)
    தசம   freq_scale;   // Frequency scaling factor
    தசம   ext_factor;   // Extension factor for YaRN
    தசம   attn_factor;  // Attention factor for YaRN
    தசம   beta_fast;    // Fast beta for YaRN
    தசம   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
கட்டமைப்பு ggml_et_rope_params {
    கட்டமைப்பு ggml_tensor src0;  // F32 input tensor
    கட்டமைப்பு ggml_tensor src1;  // I32 position tensor
    கட்டமைப்பு ggml_tensor src2;  // F32 frequency factors (optional)
    கட்டமைப்பு ggml_tensor dst;   // F32 output tensor
    rope_params_t      rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

// floor/ceil with ±inf and NaN passthrough.
நிலையான உள்வரி தசம rope_floorf(தசம x) {
    ஒன்றிப்பு {
        தசம    f;
        uint32_t u;
    } v = { .f = x };

    மாறிலி uint32_t expo = (v.u >> 23) & 0xFF;
    எனில் (expo == 0xFF) {
        திரும்பு x;  // inf or NaN
    }
    எனில் (expo >= 23 + 127) {
        திரும்பு x;  // already integer-valued
    }
    முழு i = (முழு) x;
    திரும்பு (x < 0.0f && (தசம) i != x) ? (தசம) (i - 1) : (தசம) i;
}

நிலையான உள்வரி தசம rope_ceilf(தசம x) {
    ஒன்றிப்பு {
        தசம    f;
        uint32_t u;
    } v = { .f = x };

    மாறிலி uint32_t expo = (v.u >> 23) & 0xFF;
    எனில் (expo == 0xFF) {
        திரும்பு x;  // inf or NaN
    }
    எனில் (expo >= 23 + 127) {
        திரும்பு x;  // already integer-valued
    }
    முழு i = (முழு) x;
    திரும்பு (x > 0.0f && (தசம) i != x) ? (தசம) (i + 1) : (தசம) i;
}

நிலையான உள்வரி தசம rope_yarn_ramp(மாறிலி தசம கீழ், மாறிலி தசம உயர்வு, மாறிலி முழு i0) {
    தசம denom = உயர்வு - கீழ்;
    எனில் (denom < 0.001f) {
        denom = 0.001f;
    }

    மாறிலி தசம y       = et_fdiv((தசம) (i0 / 2) - கீழ், denom);
    மாறிலி தசம clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    திரும்பு 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
நிலையான உள்வரி தசம rope_yarn_corr_dim(முழு n_dims, முழு n_ctx_orig, தசம beta, தசம freq_base) {
    திரும்பு (தசம) n_dims *
           et_fdiv(et_logf(et_fdiv((தசம) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_base));
}

நிலையான உள்வரி வெறுமை rope_yarn_corr_dims(முழு   n_dims,
                                       முழு   n_ctx_orig,
                                       தசம freq_base,
                                       தசம beta_fast,
                                       தசம beta_slow,
                                       தசம dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    தசம start = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_base));
    தசம end   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_base));

    dims[0] = start > 0.0f ? start : 0.0f;
    dims[1] = end < (தசம) (n_dims - 1) ? end : (தசம) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

நிலையான மாறிலி தசம rope_ps_one[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
நிலையான மாறிலி தசம rope_ps_c3[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
நிலையான மாறிலி தசம rope_ps_c5[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
நிலையான மாறிலி தசம rope_ps_c7[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
நிலையான மாறிலி தசம rope_ps_c9[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
நிலையான மாறிலி தசம rope_ps_c11[ROPE_VEC_WIDTH]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

நிலையான உள்வரி uint64_t rope_ps_enter_fullmask(வெறுமை) {
    uint64_t old_mask;
    __asm__ நிலையற்ற(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_mask)
        :
        : "t0", "memory");
    திரும்பு old_mask;
}

நிலையான உள்வரி வெறுமை rope_ps_leave_fullmask(uint64_t old_mask) {
    __asm__ நிலையற்ற("mova.m.x %0           \n\t" : : "r"(old_mask) : "memory");
}

நிலையான உள்வரி வெறுமை rope_poly_sin_block8(தசம * out, மாறிலி தசம * x) {
    __asm__ நிலையற்ற(
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
        : [out] "=m"(*(தசம (*)[ROPE_VEC_WIDTH]) out)
        : [x] "m"(*(மாறிலி தசம (*)[ROPE_VEC_WIDTH]) x), [one] "m"(*(மாறிலி தசம (*)[ROPE_VEC_WIDTH]) rope_ps_one),
          [c3] "m"(*(மாறிலி தசம (*)[ROPE_VEC_WIDTH]) rope_ps_c3),
          [c5] "m"(*(மாறிலி தசம (*)[ROPE_VEC_WIDTH]) rope_ps_c5),
          [c7] "m"(*(மாறிலி தசம (*)[ROPE_VEC_WIDTH]) rope_ps_c7),
          [c9] "m"(*(மாறிலி தசம (*)[ROPE_VEC_WIDTH]) rope_ps_c9),
          [c11] "m"(*(மாறிலி தசம (*)[ROPE_VEC_WIDTH]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

நிலையான உள்வரி வெறுமை rope_sincos_block8(தசம * sin8, தசம * cos8, மாறிலி தசம * theta8) {
    தசம sin_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    தசம cos_fold[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    தசம sin_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
    தசம cos_sign[ROPE_VEC_WIDTH] __attribute__((aligned(32)));

    க்கு (முழு i = 0; i < ROPE_VEC_WIDTH; ++i) {
        தசம x = theta8[i];

        எனில் (x > ROPE_PI || x < -ROPE_PI) {
            தசம cycles = x * ROPE_INV_TWO_PI;
            முழு   n      = (முழு) cycles;
            எனில் (x < 0.0f) {
                n--;
            }
            x = x - (தசம) n * ROPE_TWO_PI;
        }

        {
            தசம y = x;
            தசம s = 1.0f;
            எனில் (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } இல்லையெனில் எனில் (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            தசம y = x + ROPE_PI_OVER_2;
            எனில் (y > ROPE_PI || y < -ROPE_PI) {
                தசம cycles = y * ROPE_INV_TWO_PI;
                முழு   n      = (முழு) cycles;
                எனில் (y < 0.0f) {
                    n--;
                }
                y = y - (தசம) n * ROPE_TWO_PI;
            }

            தசம s = 1.0f;
            எனில் (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } இல்லையெனில் எனில் (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            cos_fold[i] = y;
            cos_sign[i] = s;
        }
    }

    {
        மாறிலி uint64_t saved_mask = rope_ps_enter_fullmask();

        rope_poly_sin_block8(sin8, sin_fold);
        rope_poly_sin_block8(cos8, cos_fold);

        __asm__ நிலையற்ற(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(தசம (*)[ROPE_VEC_WIDTH]) sin8), [cout] "=m"(*(தசம (*)[ROPE_VEC_WIDTH]) cos8)
            : [sinv] "m"(*(மாறிலி தசம (*)[ROPE_VEC_WIDTH]) sin8),
              [sinsgn] "m"(*(மாறிலி தசம (*)[ROPE_VEC_WIDTH]) sin_sign),
              [cosv] "m"(*(மாறிலி தசம (*)[ROPE_VEC_WIDTH]) cos8),
              [cossgn] "m"(*(மாறிலி தசம (*)[ROPE_VEC_WIDTH]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_mask);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
நிலையான உள்வரி வெறுமை rope_yarn_scalar(தசம       theta_extrap,
                                    தசம       freq_scale,
                                    மாறிலி தசம corr_dims[2],
                                    int64_t     i0,
                                    தசம       ext_factor,
                                    தசம       mscale,
                                    தசம *     cos_theta,
                                    தசம *     sin_theta) {
    தசம theta_interp = freq_scale * theta_extrap;
    தசம theta        = theta_interp;

    எனில் (ext_factor != 0.0f) {
        தசம ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (முழு) i0) * ext_factor;
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
நிலையான உள்வரி வெறுமை compute_rope_cache(தசம *       cos_cache,
                                      தசம *       sin_cache,
                                      int32_t       n_dims,
                                      தசம         theta_scale,
                                      int32_t       pos,
                                      மாறிலி தசம * freq_factors,
                                      தசம         freq_scale,
                                      மாறிலி தசம   corr_dims[2],
                                      தசம         ext_factor,
                                      தசம         attn_factor) {
    மாறிலி int32_t half_dims = n_dims / 2;
    தசம         theta     = 1.0f;

    int32_t dim_idx = 0;

    க்கு (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        தசம theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        தசம theta_local = theta;
        தசம mscale      = attn_factor;

        எனில் (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        க்கு (முழு i = 0; i < ROPE_VEC_WIDTH; ++i) {
            மாறிலி int32_t pair_idx     = dim_idx + i;
            மாறிலி தசம   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            மாறிலி தசம   theta_base   = (தசம) pos * theta_local;
            மாறிலி தசம   theta_extrap = et_fdiv(theta_base, ff);

            தசம theta_interp = freq_scale * theta_extrap;
            தசம theta_mix    = theta_interp;

            எனில் (ext_factor != 0.0f) {
                தசம ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_block[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_block);

        க்கு (முழு i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    க்கு (; dim_idx < half_dims; ++dim_idx) {
        மாறிலி தசம ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        மாறிலி தசம theta_base = (தசம) pos * theta;

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
நிலையான உள்வரி வெறுமை compute_imrope_cache(தசம *       cos_cache,
                                        தசம *       sin_cache,
                                        int32_t       n_dims,
                                        தசம         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        மாறிலி int32_t sections[4],
                                        மாறிலி தசம * freq_factors,
                                        தசம         freq_scale,
                                        மாறிலி தசம   corr_dims[2],
                                        தசம         ext_factor,
                                        தசம         attn_factor) {
    மாறிலி int32_t half_dims = n_dims / 2;
    மாறிலி int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    தசம theta_t = (தசம) pos_t;
    தசம theta_h = (தசம) pos_h;
    தசம theta_w = (தசம) pos_w;
    தசம theta_e = (தசம) pos_e;

    int32_t dim_idx = 0;

    க்கு (; dim_idx + ROPE_VEC_WIDTH <= half_dims; dim_idx += ROPE_VEC_WIDTH) {
        தசம theta_block[ROPE_VEC_WIDTH] __attribute__((aligned(32)));
        தசம mscale = attn_factor;

        எனில் (ext_factor != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        க்கு (முழு i = 0; i < ROPE_VEC_WIDTH; ++i) {
            மாறிலி int32_t pair_idx = dim_idx + i;
            மாறிலி int32_t sector   = pair_idx % sect_dims;
            மாறிலி தசம   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            தசம theta;
            எனில் (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } இல்லையெனில் எனில் (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } இல்லையெனில் எனில் (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } இல்லையெனில் {
                theta = theta_e;
            }

            மாறிலி தசம theta_extrap = et_fdiv(theta, ff);
            தசம       theta_interp = freq_scale * theta_extrap;
            தசம       theta_mix    = theta_interp;

            எனில் (ext_factor != 0.0f) {
                தசம ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_factor;
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

        க்கு (முழு i = 0; i < ROPE_VEC_WIDTH; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    க்கு (; dim_idx < half_dims; ++dim_idx) {
        மாறிலி int32_t sector = dim_idx % sect_dims;
        மாறிலி தசம   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        தசம theta;
        எனில் (sector % 3 == 1 && sector < 3 * sections[1]) {
            theta = theta_h;
        } இல்லையெனில் எனில் (sector % 3 == 2 && sector < 3 * sections[2]) {
            theta = theta_w;
        } இல்லையெனில் எனில் (sector % 3 == 0 && sector < 3 * sections[0]) {
            theta = theta_t;
        } இல்லையெனில் {
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

முழு entry_point(கட்டமைப்பு ggml_et_rope_params * params, வெறுமை * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    எனில் (!kernel_env) {
        திரும்பு -1;
    }

    முழு thread_id   = get_relative_thread_id(kernel_env->shire_mask);
    முழு இலக்கம்_threads = get_இலக்கம்_threads(kernel_env->shire_mask);

    எனில் (thread_id < 0) {
        திரும்பு -1;
    }

    எனில் (params == 0 || ((uint64_t) params & 0x7) != 0) {
        திரும்பு -1;
    }

    கட்டமைப்பு ggml_tensor * src0 = &params->src0;
    கட்டமைப்பு ggml_tensor * src1 = &params->src1;
    கட்டமைப்பு ggml_tensor * src2 = &params->src2;
    கட்டமைப்பு ggml_tensor * dst  = &params->dst;

    எனில் (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        திரும்பு -1;
    }

    மாறிலி தசம *   src0_data    = (மாறிலி தசம *) src0->data;
    மாறிலி int32_t * src1_data    = (மாறிலி int32_t *) src1->data;
    மாறிலி தசம *   freq_factors = (src2 && src2->data) ? (மாறிலி தசம *) src2->data : வெற்று;
    தசம *         dst_data     = (தசம *) dst->data;

    எனில் (!src0_data || !src1_data || !dst_data) {
        திரும்பு -1;
    }
#ifdef ET_UBERKERNEL
    மாறிலி size_t src0_bytes = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    மாறிலி size_t src1_bytes = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_data, src0_bytes);
    evict_region_past_l2(src1_data, src1_bytes);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#endif
    மாறிலி int64_t head_dim = src0->ne[0];
    மாறிலி int64_t heads    = src0->ne[1];
    மாறிலி int64_t seq_len  = src0->ne[2];
    மாறிலி int64_t batch    = src0->ne[3];

    மாறிலி rope_params_t * rope_params = &params->rope_params;
    மாறிலி int32_t         n_dims      = rope_params->n_dims;
    மாறிலி தசம           freq_base   = rope_params->freq_base;
    மாறிலி தசம           freq_scale  = rope_params->freq_scale;
    மாறிலி int32_t         mode        = rope_params->mode;

    எனில் (n_dims <= 0 || n_dims > head_dim || (n_dims & 1) != 0) {
        திரும்பு -1;
    }

    எனில் (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        திரும்பு -1;
    }

    தசம cos_cache[MAX_ROPE_HALF_DIMS];
    தசம sin_cache[MAX_ROPE_HALF_DIMS];

    தசம corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_base, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    மாறிலி int64_t total_heads = batch * seq_len * heads;
    மாறிலி int64_t start_wu    = (total_heads * thread_id) / இலக்கம்_threads;
    மாறிலி int64_t end_wu      = (total_heads * (thread_id + 1)) / இலக்கம்_threads;

    எனில் (start_wu >= end_wu) {
        திரும்பு 0;
    }

    மாறிலி தசம   theta_scale       = et_powf(freq_base, et_fdiv(-2.0f, (தசம) n_dims));
    மாறிலி int32_t half_dims         = n_dims / 2;
    மாறிலி முழு     is_neox           = (mode & GGML_ROPE_TYPE_NEOX) != 0;
    மாறிலி முழு     is_imrope         = (mode == GGML_ROPE_TYPE_IMROPE);
    மாறிலி முழு     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t last_pos   = -1;
    int32_t last_pos_h = -1;
    int32_t last_pos_w = -1;
    int32_t last_pos_e = -1;

    க்கு (int64_t wu = start_wu; wu < end_wu; ++wu) {
        மாறிலி int64_t h = wu % heads;
        மாறிலி int64_t s = (wu / heads) % seq_len;
        மாறிலி int64_t b = wu / (heads * seq_len);

        எனில் (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            மாறிலி int32_t pt = src1_data[s] + rope_params->n_past;
            மாறிலி int32_t ph = src1_data[s + seq_len] + rope_params->n_past;
            மாறிலி int32_t pw = src1_data[s + seq_len * 2] + rope_params->n_past;
            மாறிலி int32_t pe = src1_data[s + seq_len * 3] + rope_params->n_past;

            எனில் (pt != last_pos || ph != last_pos_h || pw != last_pos_w || pe != last_pos_e) {
                compute_imrope_cache(cos_cache, sin_cache, n_dims, theta_scale, pt, ph, pw, pe, rope_params->sections,
                                     freq_factors, freq_scale, corr_dims, rope_params->ext_factor,
                                     rope_params->attn_factor);
                last_pos   = pt;
                last_pos_h = ph;
                last_pos_w = pw;
                last_pos_e = pe;
            }
        } இல்லையெனில் {
            மாறிலி int32_t pos = src1_data[s] + rope_params->n_past;

            எனில் (pos != last_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_factor, rope_params->attn_factor);
                last_pos = pos;
            }
        }

        மாறிலி தசம * head_src =
            (மாறிலி தசம *) ((மாறிலி எழுத்து *) src0_data + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        தசம * head_dst = (தசம *) ((எழுத்து *) dst_data + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        க்கு (int64_t d = n_dims; d < head_dim; ++d) {
            head_dst[d] = head_src[d];
        }

        எனில் (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t temp_mask;
            __asm__ நிலையற்ற("mova.x.m %0" : "=r"(temp_mask));
            __asm__ நிலையற்ற("mov.m.x m0, x0, 0xFF");

            க்கு (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                __asm__ நிலையற்ற(
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
                    : [x0_dst] "=m"(*(தசம (*)[8]) & head_dst[dim_idx]), [x1_dst] "=m"(*(தசம (*)[8]) &
                                                                                        head_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(மாறிலி தசம (*)[8]) & head_src[dim_idx]),
                      [x1_src] "m"(*(மாறிலி தசம (*)[8]) & head_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(மாறிலி தசம (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(மாறிலி தசம (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ நிலையற்ற("mova.m.x %0" ::"r"(temp_mask));
        } இல்லையெனில் {
            // Standard: adjacent pairs (2i, 2i+1)
            க்கு (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                மாறிலி int32_t dim_in_head = pair_idx * 2;
                மாறிலி தசம   x0          = head_src[dim_in_head];
                மாறிலி தசம   x1          = head_src[dim_in_head + 1];

                head_dst[dim_in_head]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                head_dst[dim_in_head + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    திரும்பு 0;
}
