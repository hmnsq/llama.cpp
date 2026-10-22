//******************************************************************************
// ROPE (Rotary Position Encoding) Kernel
// Experiment 1:
//   - Keep old scheduling and rotate logic
//   - ONLY SIMD-ize sin/cos approximation inside compute_rope_cache()
//******************************************************************************

#అన్తర్భావ "ggml_tensor.h"
#అన్తర్భావ "math_fp.h"
#అన్తర్భావ "platform.h"

#అన్తర్భావ <etsoc/common/utils.h>
#అన్తర్భావ <stdint.h>

// ROPE constants (matching GGML definitions)
#నిర్వచించు GGML_ROPE_TYPE_NEOX   2
#నిర్వచించు GGML_ROPE_TYPE_MROPE  8
#నిర్వచించు GGML_ROPE_TYPE_IMROPE 40
#నిర్వచించు MAX_ROPE_HALF_DIMS    256  // supports up to n_dims=512

#నిర్వచించు ROPE_VEC_వెడల్పు 8

#నిర్వచించు ROPE_PI         3.14159265358979323846f
#నిర్వచించు ROPE_TWO_PI     6.28318530717958647693f
#నిర్వచించు ROPE_PI_OVER_2  1.57079632679489661923f
#నిర్వచించు ROPE_INV_TWO_PI 0.15915494309189533577f

// ROPE operation parameters structure (matches ggml-et-ops.h)
రకనిర్వచనం నిర్మాణం {
    int32_t n_past;
    int32_t n_dims;  // Number of dimensions to apply ROPE to (must be even)
    int32_t బహుళకం;    // ROPE mode (0=normal, 2=neox)
    int32_t n_ctx;
    int32_t n_ctx_orig;
    దశాంశం   freq_పునాది;    // Base frequency (usually 10000.0f)
    దశాంశం   freq_scale;   // Frequency scaling factor
    దశాంశం   ext_గుణకం;   // Extension factor for YaRN
    దశాంశం   attn_గుణకం;  // Attention factor for YaRN
    దశాంశం   beta_fast;    // Fast beta for YaRN
    దశాంశం   beta_slow;    // Slow beta for YaRN
    int32_t sections[4];  // Sections for multi-modal ROPE
} rope_params_t;

// ROPE kernel parameters structure (matches ggml_et_rope_params)
నిర్మాణం ggml_et_rope_params {
    నిర్మాణం ggml_tensor src0;  // F32 input tensor
    నిర్మాణం ggml_tensor src1;  // I32 position tensor
    నిర్మాణం ggml_tensor src2;  // F32 frequency factors (optional)
    నిర్మాణం ggml_tensor dst;   // F32 output tensor
    rope_params_t      rope_params;
};

//------------------------------------------------------------------------------
// Existing scalar helpers
//------------------------------------------------------------------------------

// floor/ceil with ±inf and NaN passthrough.
స్థిర అంతఃవరుస దశాంశం rope_floorf(దశాంశం x) {
    సంఘం {
        దశాంశం    f;
        uint32_t u;
    } v = { .f = x };

    స్థిరరాశి uint32_t expo = (v.u >> 23) & 0xFF;
    ఒకవేళ (expo == 0xFF) {
        తిరిగి x;  // inf or NaN
    }
    ఒకవేళ (expo >= 23 + 127) {
        తిరిగి x;  // already integer-valued
    }
    పూర్ణాంకం i = (పూర్ణాంకం) x;
    తిరిగి (x < 0.0f && (దశాంశం) i != x) ? (దశాంశం) (i - 1) : (దశాంశం) i;
}

స్థిర అంతఃవరుస దశాంశం rope_ceilf(దశాంశం x) {
    సంఘం {
        దశాంశం    f;
        uint32_t u;
    } v = { .f = x };

    స్థిరరాశి uint32_t expo = (v.u >> 23) & 0xFF;
    ఒకవేళ (expo == 0xFF) {
        తిరిగి x;  // inf or NaN
    }
    ఒకవేళ (expo >= 23 + 127) {
        తిరిగి x;  // already integer-valued
    }
    పూర్ణాంకం i = (పూర్ణాంకం) x;
    తిరిగి (x > 0.0f && (దశాంశం) i != x) ? (దశాంశం) (i + 1) : (దశాంశం) i;
}

స్థిర అంతఃవరుస దశాంశం rope_yarn_ramp(స్థిరరాశి దశాంశం దిగువ, స్థిరరాశి దశాంశం ఎగువ, స్థిరరాశి పూర్ణాంకం i0) {
    దశాంశం denom = ఎగువ - దిగువ;
    ఒకవేళ (denom < 0.001f) {
        denom = 0.001f;
    }

    స్థిరరాశి దశాంశం y       = et_fdiv((దశాంశం) (i0 / 2) - దిగువ, denom);
    స్థిరరాశి దశాంశం clamped = y < 0.0f ? 0.0f : (y > 1.0f ? 1.0f : y);
    తిరిగి 1.0f - clamped;
}

// Matches CPU reference (ggml_rope_yarn_corr_dim).
స్థిర అంతఃవరుస దశాంశం rope_yarn_corr_dim(పూర్ణాంకం n_dims, పూర్ణాంకం n_ctx_orig, దశాంశం beta, దశాంశం freq_పునాది) {
    తిరిగి (దశాంశం) n_dims *
           et_fdiv(et_logf(et_fdiv((దశాంశం) n_ctx_orig, beta * ROPE_TWO_PI)), 2.0f * et_logf(freq_పునాది));
}

స్థిర అంతఃవరుస శూన్య rope_yarn_corr_dims(పూర్ణాంకం   n_dims,
                                       పూర్ణాంకం   n_ctx_orig,
                                       దశాంశం freq_పునాది,
                                       దశాంశం beta_fast,
                                       దశాంశం beta_slow,
                                       దశాంశం dims[2]) {
    // Match CPU: floor on start, ceil on end, then clamp to [0, n_dims-1].
    దశాంశం ప్రారంభం = rope_floorf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_fast, freq_పునాది));
    దశాంశం చివర   = rope_ceilf(rope_yarn_corr_dim(n_dims, n_ctx_orig, beta_slow, freq_పునాది));

    dims[0] = ప్రారంభం > 0.0f ? ప్రారంభం : 0.0f;
    dims[1] = చివర < (దశాంశం) (n_dims - 1) ? చివర : (దశాంశం) (n_dims - 1);
}

//------------------------------------------------------------------------------
// SIMD sin/cos approximation
//------------------------------------------------------------------------------

స్థిర స్థిరరాశి దశాంశం rope_ps_one[ROPE_VEC_వెడల్పు]
    __attribute__((aligned(32))) = { 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f, 1.f };
స్థిర స్థిరరాశి దశాంశం rope_ps_c3[ROPE_VEC_వెడల్పు]
    __attribute__((aligned(32))) = { 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f,
                                     1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f, 1.0f / 6.0f };
స్థిర స్థిరరాశి దశాంశం rope_ps_c5[ROPE_VEC_వెడల్పు]
    __attribute__((aligned(32))) = { 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f,
                                     1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f, 1.0f / 120.0f };
స్థిర స్థిరరాశి దశాంశం rope_ps_c7[ROPE_VEC_వెడల్పు]
    __attribute__((aligned(32))) = { 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f,
                                     1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f, 1.0f / 5040.0f };
స్థిర స్థిరరాశి దశాంశం rope_ps_c9[ROPE_VEC_వెడల్పు]
    __attribute__((aligned(32))) = { 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f,
                                     1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f, 1.0f / 362880.0f };
స్థిర స్థిరరాశి దశాంశం rope_ps_c11[ROPE_VEC_వెడల్పు]
    __attribute__((aligned(32))) = { 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f,
                                     1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f, 1.0f / 39916800.0f };

స్థిర అంతఃవరుస uint64_t rope_ps_enter_fullmask(శూన్య) {
    uint64_t old_ముసుగు;
    __asm__ అస్థిర(
        "mova.x.m %0           \n\t"
        "li       t0, -1       \n\t"
        "mova.m.x t0           \n\t"
        : "=r"(old_ముసుగు)
        :
        : "t0", "memory");
    తిరిగి old_ముసుగు;
}

స్థిర అంతఃవరుస శూన్య rope_ps_leave_fullmask(uint64_t old_ముసుగు) {
    __asm__ అస్థిర("mova.m.x %0           \n\t" : : "r"(old_ముసుగు) : "memory");
}

స్థిర అంతఃవరుస శూన్య rope_poly_sin_block8(దశాంశం * బహిర్గతం, స్థిరరాశి దశాంశం * x) {
    __asm__ అస్థిర(
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
        : [బహిర్గతం] "=m"(*(దశాంశం (*)[ROPE_VEC_వెడల్పు]) బహిర్గతం)
        : [x] "m"(*(స్థిరరాశి దశాంశం (*)[ROPE_VEC_వెడల్పు]) x), [one] "m"(*(స్థిరరాశి దశాంశం (*)[ROPE_VEC_వెడల్పు]) rope_ps_one),
          [c3] "m"(*(స్థిరరాశి దశాంశం (*)[ROPE_VEC_వెడల్పు]) rope_ps_c3),
          [c5] "m"(*(స్థిరరాశి దశాంశం (*)[ROPE_VEC_వెడల్పు]) rope_ps_c5),
          [c7] "m"(*(స్థిరరాశి దశాంశం (*)[ROPE_VEC_వెడల్పు]) rope_ps_c7),
          [c9] "m"(*(స్థిరరాశి దశాంశం (*)[ROPE_VEC_వెడల్పు]) rope_ps_c9),
          [c11] "m"(*(స్థిరరాశి దశాంశం (*)[ROPE_VEC_వెడల్పు]) rope_ps_c11)
        : "f0", "f1", "f2", "f3", "f4", "memory");
}

స్థిర అంతఃవరుస శూన్య rope_sincos_block8(దశాంశం * sin8, దశాంశం * cos8, స్థిరరాశి దశాంశం * theta8) {
    దశాంశం sin_fold[ROPE_VEC_వెడల్పు] __attribute__((aligned(32)));
    దశాంశం cos_fold[ROPE_VEC_వెడల్పు] __attribute__((aligned(32)));
    దశాంశం sin_sign[ROPE_VEC_వెడల్పు] __attribute__((aligned(32)));
    దశాంశం cos_sign[ROPE_VEC_వెడల్పు] __attribute__((aligned(32)));

    కొరకు (పూర్ణాంకం i = 0; i < ROPE_VEC_వెడల్పు; ++i) {
        దశాంశం x = theta8[i];

        ఒకవేళ (x > ROPE_PI || x < -ROPE_PI) {
            దశాంశం చక్రాలు = x * ROPE_INV_TWO_PI;
            పూర్ణాంకం   n      = (పూర్ణాంకం) చక్రాలు;
            ఒకవేళ (x < 0.0f) {
                n--;
            }
            x = x - (దశాంశం) n * ROPE_TWO_PI;
        }

        {
            దశాంశం y = x;
            దశాంశం s = 1.0f;
            ఒకవేళ (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } లేకపోతే ఒకవేళ (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            sin_fold[i] = y;
            sin_sign[i] = s;
        }

        {
            దశాంశం y = x + ROPE_PI_OVER_2;
            ఒకవేళ (y > ROPE_PI || y < -ROPE_PI) {
                దశాంశం చక్రాలు = y * ROPE_INV_TWO_PI;
                పూర్ణాంకం   n      = (పూర్ణాంకం) చక్రాలు;
                ఒకవేళ (y < 0.0f) {
                    n--;
                }
                y = y - (దశాంశం) n * ROPE_TWO_PI;
            }

            దశాంశం s = 1.0f;
            ఒకవేళ (y > ROPE_PI_OVER_2) {
                y = ROPE_PI - y;
            } లేకపోతే ఒకవేళ (y < -ROPE_PI_OVER_2) {
                y = -ROPE_PI - y;
                s = -1.0f;
            }
            cos_fold[i] = y;
            cos_sign[i] = s;
        }
    }

    {
        స్థిరరాశి uint64_t saved_ముసుగు = rope_ps_enter_fullmask();

        rope_poly_sin_block8(sin8, sin_fold);
        rope_poly_sin_block8(cos8, cos_fold);

        __asm__ అస్థిర(
            "flw.ps    f0, %[sinv]         \n\t"
            "flw.ps    f1, %[sinsgn]       \n\t"
            "fmul.ps   f2, f0, f1          \n\t"
            "fsw.ps    f2, %[sout]         \n\t"

            "flw.ps    f3, %[cosv]         \n\t"
            "flw.ps    f4, %[cossgn]       \n\t"
            "fmul.ps   f5, f3, f4          \n\t"
            "fsw.ps    f5, %[cout]         \n\t"
            : [sout] "=m"(*(దశాంశం (*)[ROPE_VEC_వెడల్పు]) sin8), [cout] "=m"(*(దశాంశం (*)[ROPE_VEC_వెడల్పు]) cos8)
            : [sinv] "m"(*(స్థిరరాశి దశాంశం (*)[ROPE_VEC_వెడల్పు]) sin8),
              [sinsgn] "m"(*(స్థిరరాశి దశాంశం (*)[ROPE_VEC_వెడల్పు]) sin_sign),
              [cosv] "m"(*(స్థిరరాశి దశాంశం (*)[ROPE_VEC_వెడల్పు]) cos8),
              [cossgn] "m"(*(స్థిరరాశి దశాంశం (*)[ROPE_VEC_వెడల్పు]) cos_sign)
            : "f0", "f1", "f2", "f3", "f4", "f5", "memory");

        rope_ps_leave_fullmask(saved_ముసుగు);
    }
}

//------------------------------------------------------------------------------
// Cache build
//------------------------------------------------------------------------------

// scalar fallback for tail / tiny sizes
స్థిర అంతఃవరుస శూన్య rope_yarn_scalar(దశాంశం       theta_extrap,
                                    దశాంశం       freq_scale,
                                    స్థిరరాశి దశాంశం corr_dims[2],
                                    int64_t     i0,
                                    దశాంశం       ext_గుణకం,
                                    దశాంశం       mscale,
                                    దశాంశం *     cos_theta,
                                    దశాంశం *     sin_theta) {
    దశాంశం theta_interp = freq_scale * theta_extrap;
    దశాంశం theta        = theta_interp;

    ఒకవేళ (ext_గుణకం != 0.0f) {
        దశాంశం ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], (పూర్ణాంకం) i0) * ext_గుణకం;
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
స్థిర అంతఃవరుస శూన్య compute_rope_cache(దశాంశం *       cos_cache,
                                      దశాంశం *       sin_cache,
                                      int32_t       n_dims,
                                      దశాంశం         theta_scale,
                                      int32_t       pos,
                                      స్థిరరాశి దశాంశం * freq_factors,
                                      దశాంశం         freq_scale,
                                      స్థిరరాశి దశాంశం   corr_dims[2],
                                      దశాంశం         ext_గుణకం,
                                      దశాంశం         attn_గుణకం) {
    స్థిరరాశి int32_t half_dims = n_dims / 2;
    దశాంశం         theta     = 1.0f;

    int32_t dim_idx = 0;

    కొరకు (; dim_idx + ROPE_VEC_వెడల్పు <= half_dims; dim_idx += ROPE_VEC_వెడల్పు) {
        దశాంశం theta_ఖండం[ROPE_VEC_వెడల్పు] __attribute__((aligned(32)));
        దశాంశం theta_local = theta;
        దశాంశం mscale      = attn_గుణకం;

        ఒకవేళ (ext_గుణకం != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        కొరకు (పూర్ణాంకం i = 0; i < ROPE_VEC_వెడల్పు; ++i) {
            స్థిరరాశి int32_t pair_idx     = dim_idx + i;
            స్థిరరాశి దశాంశం   ff           = freq_factors ? freq_factors[pair_idx] : 1.0f;
            స్థిరరాశి దశాంశం   theta_పునాది   = (దశాంశం) pos * theta_local;
            స్థిరరాశి దశాంశం   theta_extrap = et_fdiv(theta_పునాది, ff);

            దశాంశం theta_interp = freq_scale * theta_extrap;
            దశాంశం theta_mix    = theta_interp;

            ఒకవేళ (ext_గుణకం != 0.0f) {
                దశాంశం ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_గుణకం;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_ఖండం[i] = theta_mix;
            theta_local *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_ఖండం);

        కొరకు (పూర్ణాంకం i = 0; i < ROPE_VEC_వెడల్పు; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }

        theta = theta_local;
    }

    // tail fallback
    కొరకు (; dim_idx < half_dims; ++dim_idx) {
        స్థిరరాశి దశాంశం ff         = freq_factors ? freq_factors[dim_idx] : 1.0f;
        స్థిరరాశి దశాంశం theta_పునాది = (దశాంశం) pos * theta;

        rope_yarn_scalar(et_fdiv(theta_పునాది, ff), freq_scale, corr_dims, dim_idx * 2, ext_గుణకం, attn_గుణకం,
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
స్థిర అంతఃవరుస శూన్య compute_imrope_cache(దశాంశం *       cos_cache,
                                        దశాంశం *       sin_cache,
                                        int32_t       n_dims,
                                        దశాంశం         theta_scale,
                                        int32_t       pos_t,
                                        int32_t       pos_h,
                                        int32_t       pos_w,
                                        int32_t       pos_e,
                                        స్థిరరాశి int32_t sections[4],
                                        స్థిరరాశి దశాంశం * freq_factors,
                                        దశాంశం         freq_scale,
                                        స్థిరరాశి దశాంశం   corr_dims[2],
                                        దశాంశం         ext_గుణకం,
                                        దశాంశం         attn_గుణకం) {
    స్థిరరాశి int32_t half_dims = n_dims / 2;
    స్థిరరాశి int32_t sect_dims = sections[0] + sections[1] + sections[2] + sections[3];

    దశాంశం theta_t = (దశాంశం) pos_t;
    దశాంశం theta_h = (దశాంశం) pos_h;
    దశాంశం theta_w = (దశాంశం) pos_w;
    దశాంశం theta_e = (దశాంశం) pos_e;

    int32_t dim_idx = 0;

    కొరకు (; dim_idx + ROPE_VEC_వెడల్పు <= half_dims; dim_idx += ROPE_VEC_వెడల్పు) {
        దశాంశం theta_ఖండం[ROPE_VEC_వెడల్పు] __attribute__((aligned(32)));
        దశాంశం mscale = attn_గుణకం;

        ఒకవేళ (ext_గుణకం != 0.0f) {
            mscale *= 1.0f + 0.1f * et_logf(et_fdiv(1.0f, freq_scale));
        }

        కొరకు (పూర్ణాంకం i = 0; i < ROPE_VEC_వెడల్పు; ++i) {
            స్థిరరాశి int32_t pair_idx = dim_idx + i;
            స్థిరరాశి int32_t sector   = pair_idx % sect_dims;
            స్థిరరాశి దశాంశం   ff       = freq_factors ? freq_factors[pair_idx] : 1.0f;

            // Interleaved sector assignment (mod-3 pattern)
            దశాంశం theta;
            ఒకవేళ (sector % 3 == 1 && sector < 3 * sections[1]) {
                theta = theta_h;
            } లేకపోతే ఒకవేళ (sector % 3 == 2 && sector < 3 * sections[2]) {
                theta = theta_w;
            } లేకపోతే ఒకవేళ (sector % 3 == 0 && sector < 3 * sections[0]) {
                theta = theta_t;
            } లేకపోతే {
                theta = theta_e;
            }

            స్థిరరాశి దశాంశం theta_extrap = et_fdiv(theta, ff);
            దశాంశం       theta_interp = freq_scale * theta_extrap;
            దశాంశం       theta_mix    = theta_interp;

            ఒకవేళ (ext_గుణకం != 0.0f) {
                దశాంశం ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], pair_idx * 2) * ext_గుణకం;
                theta_mix      = theta_interp * (1.0f - ramp_mix) + theta_extrap * ramp_mix;
            }

            theta_ఖండం[i] = theta_mix;

            // All 4 thetas advance every iteration
            theta_t *= theta_scale;
            theta_h *= theta_scale;
            theta_w *= theta_scale;
            theta_e *= theta_scale;
        }

        rope_sincos_block8(&sin_cache[dim_idx], &cos_cache[dim_idx], theta_ఖండం);

        కొరకు (పూర్ణాంకం i = 0; i < ROPE_VEC_వెడల్పు; ++i) {
            sin_cache[dim_idx + i] *= mscale;
            cos_cache[dim_idx + i] *= mscale;
        }
    }

    // Scalar tail
    కొరకు (; dim_idx < half_dims; ++dim_idx) {
        స్థిరరాశి int32_t sector = dim_idx % sect_dims;
        స్థిరరాశి దశాంశం   ff     = freq_factors ? freq_factors[dim_idx] : 1.0f;

        దశాంశం theta;
        ఒకవేళ (sector % 3 == 1 && sector < 3 * sections[1]) {
            theta = theta_h;
        } లేకపోతే ఒకవేళ (sector % 3 == 2 && sector < 3 * sections[2]) {
            theta = theta_w;
        } లేకపోతే ఒకవేళ (sector % 3 == 0 && sector < 3 * sections[0]) {
            theta = theta_t;
        } లేకపోతే {
            theta = theta_e;
        }

        rope_yarn_scalar(et_fdiv(theta, ff), freq_scale, corr_dims, dim_idx * 2, ext_గుణకం, attn_గుణకం,
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

పూర్ణాంకం నమోదు_బిందువు(నిర్మాణం ggml_et_rope_params * params, శూన్య * env) {
    kernel_environment_t * kernel_env = (kernel_environment_t *) env;

    ఒకవేళ (!kernel_env) {
        తిరిగి -1;
    }

    పూర్ణాంకం thread_id   = get_relative_thread_id(kernel_env->shire_ముసుగు);
    పూర్ణాంకం అంకె_దారాలు = get_అంకె_దారాలు(kernel_env->shire_ముసుగు);

    ఒకవేళ (thread_id < 0) {
        తిరిగి -1;
    }

    ఒకవేళ (params == 0 || ((uint64_t) params & 0x7) != 0) {
        తిరిగి -1;
    }

    నిర్మాణం ggml_tensor * src0 = &params->src0;
    నిర్మాణం ggml_tensor * src1 = &params->src1;
    నిర్మాణం ggml_tensor * src2 = &params->src2;
    నిర్మాణం ggml_tensor * dst  = &params->dst;

    ఒకవేళ (src0->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_I32 || dst->type != GGML_TYPE_F32) {
        తిరిగి -1;
    }

    స్థిరరాశి దశాంశం *   src0_దత్తాంశం    = (స్థిరరాశి దశాంశం *) src0->దత్తాంశం;
    స్థిరరాశి int32_t * src1_దత్తాంశం    = (స్థిరరాశి int32_t *) src1->దత్తాంశం;
    స్థిరరాశి దశాంశం *   freq_factors = (src2 && src2->దత్తాంశం) ? (స్థిరరాశి దశాంశం *) src2->దత్తాంశం : ఖాళీ;
    దశాంశం *         dst_దత్తాంశం     = (దశాంశం *) dst->దత్తాంశం;

    ఒకవేళ (!src0_దత్తాంశం || !src1_దత్తాంశం || !dst_దత్తాంశం) {
        తిరిగి -1;
    }
#నిర్వచితమైతే ET_UBERKERNEL
    స్థిరరాశి size_t src0_బైట్లు = (size_t) src0->ne[0] * src0->ne[1] * src0->ne[2] * src0->ne[3] * src0->nb[0];
    స్థిరరాశి size_t src1_బైట్లు = (size_t) src1->ne[0] * src1->ne[1] * src1->ne[2] * src1->ne[3] * src1->nb[0];
    evict_region_past_l2(src0_దత్తాంశం, src0_బైట్లు);
    evict_region_past_l2(src1_దత్తాంశం, src1_బైట్లు);
    WAIT_CACHEOPS;
    FENCE;
    et_barrier(ET_BARRIER_GLOBAL);
#అయితే_ముగింపు
    స్థిరరాశి int64_t తల_dim = src0->ne[0];
    స్థిరరాశి int64_t heads    = src0->ne[1];
    స్థిరరాశి int64_t seq_len  = src0->ne[2];
    స్థిరరాశి int64_t batch    = src0->ne[3];

    స్థిరరాశి rope_params_t * rope_params = &params->rope_params;
    స్థిరరాశి int32_t         n_dims      = rope_params->n_dims;
    స్థిరరాశి దశాంశం           freq_పునాది   = rope_params->freq_పునాది;
    స్థిరరాశి దశాంశం           freq_scale  = rope_params->freq_scale;
    స్థిరరాశి int32_t         బహుళకం        = rope_params->బహుళకం;

    ఒకవేళ (n_dims <= 0 || n_dims > తల_dim || (n_dims & 1) != 0) {
        తిరిగి -1;
    }

    ఒకవేళ (n_dims / 2 > MAX_ROPE_HALF_DIMS) {
        తిరిగి -1;
    }

    దశాంశం cos_cache[MAX_ROPE_HALF_DIMS];
    దశాంశం sin_cache[MAX_ROPE_HALF_DIMS];

    దశాంశం corr_dims[2];
    rope_yarn_corr_dims(n_dims, rope_params->n_ctx_orig, freq_పునాది, rope_params->beta_fast, rope_params->beta_slow,
                        corr_dims);
    et_barrier(ET_BARRIER_GLOBAL);

    // Distribute by individual heads: total = batch * seq_len * heads.
    స్థిరరాశి int64_t మొత్తం_heads = batch * seq_len * heads;
    స్థిరరాశి int64_t ప్రారంభం_wu    = (మొత్తం_heads * thread_id) / అంకె_దారాలు;
    స్థిరరాశి int64_t చివర_wu      = (మొత్తం_heads * (thread_id + 1)) / అంకె_దారాలు;

    ఒకవేళ (ప్రారంభం_wu >= చివర_wu) {
        తిరిగి 0;
    }

    స్థిరరాశి దశాంశం   theta_scale       = et_powf(freq_పునాది, et_fdiv(-2.0f, (దశాంశం) n_dims));
    స్థిరరాశి int32_t half_dims         = n_dims / 2;
    స్థిరరాశి పూర్ణాంకం     is_neox           = (బహుళకం & GGML_ROPE_TYPE_NEOX) != 0;
    స్థిరరాశి పూర్ణాంకం     is_imrope         = (బహుళకం == GGML_ROPE_TYPE_IMROPE);
    స్థిరరాశి పూర్ణాంకం     use_neox_rotation = is_neox || is_imrope;

    // For IMROPE position cache invalidation: track all 4 channels
    int32_t చివరి_pos   = -1;
    int32_t చివరి_pos_h = -1;
    int32_t చివరి_pos_w = -1;
    int32_t చివరి_pos_e = -1;

    కొరకు (int64_t wu = ప్రారంభం_wu; wu < చివర_wu; ++wu) {
        స్థిరరాశి int64_t h = wu % heads;
        స్థిరరాశి int64_t s = (wu / heads) % seq_len;
        స్థిరరాశి int64_t b = wu / (heads * seq_len);

        ఒకవేళ (is_imrope) {
            // IMROPE: src1 layout is [p_t(0..S-1), p_h(0..S-1), p_w(0..S-1), p_e(0..S-1)]
            స్థిరరాశి int32_t pt = src1_దత్తాంశం[s] + rope_params->n_past;
            స్థిరరాశి int32_t ph = src1_దత్తాంశం[s + seq_len] + rope_params->n_past;
            స్థిరరాశి int32_t pw = src1_దత్తాంశం[s + seq_len * 2] + rope_params->n_past;
            స్థిరరాశి int32_t pe = src1_దత్తాంశం[s + seq_len * 3] + rope_params->n_past;

            ఒకవేళ (pt != చివరి_pos || ph != చివరి_pos_h || pw != చివరి_pos_w || pe != చివరి_pos_e) {
                compute_imrope_cache(cos_cache, sin_cache, n_dims, theta_scale, pt, ph, pw, pe, rope_params->sections,
                                     freq_factors, freq_scale, corr_dims, rope_params->ext_గుణకం,
                                     rope_params->attn_గుణకం);
                చివరి_pos   = pt;
                చివరి_pos_h = ph;
                చివరి_pos_w = pw;
                చివరి_pos_e = pe;
            }
        } లేకపోతే {
            స్థిరరాశి int32_t pos = src1_దత్తాంశం[s] + rope_params->n_past;

            ఒకవేళ (pos != చివరి_pos) {
                compute_rope_cache(cos_cache, sin_cache, n_dims, theta_scale, pos, freq_factors, freq_scale, corr_dims,
                                   rope_params->ext_గుణకం, rope_params->attn_గుణకం);
                చివరి_pos = pos;
            }
        }

        స్థిరరాశి దశాంశం * తల_src =
            (స్థిరరాశి దశాంశం *) ((స్థిరరాశి అక్షరం *) src0_దత్తాంశం + b * src0->nb[3] + s * src0->nb[2] + h * src0->nb[1]);

        దశాంశం * తల_dst = (దశాంశం *) ((అక్షరం *) dst_దత్తాంశం + b * dst->nb[3] + s * dst->nb[2] + h * dst->nb[1]);

        // Copy dimensions beyond n_dims unchanged
        కొరకు (int64_t d = n_dims; d < తల_dim; ++d) {
            తల_dst[d] = తల_src[d];
        }

        ఒకవేళ (use_neox_rotation) {
            // NEOX/IMROPE: pairs at (i, i+half_dims)
            uint64_t తాత్కాలిక_ముసుగు;
            __asm__ అస్థిర("mova.x.m %0" : "=r"(తాత్కాలిక_ముసుగు));
            __asm__ అస్థిర("mov.m.x m0, x0, 0xFF");

            కొరకు (int32_t dim_idx = 0; dim_idx < half_dims; dim_idx += 8) {
                __asm__ అస్థిర(
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
                    : [x0_dst] "=m"(*(దశాంశం (*)[8]) & తల_dst[dim_idx]), [x1_dst] "=m"(*(దశాంశం (*)[8]) &
                                                                                        తల_dst[dim_idx + half_dims])
                    : [x0_src] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & తల_src[dim_idx]),
                      [x1_src] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & తల_src[dim_idx + half_dims]),
                      [sin_cache] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & sin_cache[dim_idx]),
                      [cos_cache] "m"(*(స్థిరరాశి దశాంశం (*)[8]) & cos_cache[dim_idx])
                    : "f0", "f1", "f2", "f3", "f4", "f5", "memory");
            }

            __asm__ అస్థిర("mova.m.x %0" ::"r"(తాత్కాలిక_ముసుగు));
        } లేకపోతే {
            // Standard: adjacent pairs (2i, 2i+1)
            కొరకు (int32_t pair_idx = 0; pair_idx < half_dims; ++pair_idx) {
                స్థిరరాశి int32_t dim_లోపల_తల = pair_idx * 2;
                స్థిరరాశి దశాంశం   x0          = తల_src[dim_లోపల_తల];
                స్థిరరాశి దశాంశం   x1          = తల_src[dim_లోపల_తల + 1];

                తల_dst[dim_లోపల_తల]     = x0 * cos_cache[pair_idx] - x1 * sin_cache[pair_idx];
                తల_dst[dim_లోపల_తల + 1] = x0 * sin_cache[pair_idx] + x1 * cos_cache[pair_idx];
            }
        }
    }

    తిరిగి 0;
}
