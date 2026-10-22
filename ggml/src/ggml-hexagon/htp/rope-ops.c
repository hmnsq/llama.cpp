#ಸೂಚನೆ clang diagnostic ignored "-Wunused-variable"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-function"
#ಸೂಚನೆ clang diagnostic ignored "-Wunused-but-set-variable"

#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>

#ಅನ್ತರ್ಭಾವ "dma-queue.h"
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "rope-ops.h"

// Redefined the rope type constants as we can't include ggml.h
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_ROPE_TYPE_ಸಾಮಾನ್ಯ 0
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_ROPE_TYPE_NEOX   2
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_ROPE_TYPE_MROPE  8
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_ROPE_TYPE_VISION 24
#ವ್ಯಾಖ್ಯಾನಿಸು HTP_ROPE_TYPE_IMROPE 40

#ವ್ಯಾಖ್ಯಾನಿಸು htp_rope_preamble              \
    ಸ್ಥಿರಾಂಕ uint32_t ne00 = src0->ne[0]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne01 = src0->ne[1]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne02 = src0->ne[2]; \
    ಸ್ಥಿರಾಂಕ uint32_t ne03 = src0->ne[3]; \
                                       \
    ಸ್ಥಿರಾಂಕ uint32_t ne0 = dst->ne[0];   \
    ಸ್ಥಿರಾಂಕ uint32_t ne1 = dst->ne[1];   \
    ಸ್ಥಿರಾಂಕ uint32_t ne2 = dst->ne[2];   \
    ಸ್ಥಿರಾಂಕ uint32_t ne3 = dst->ne[3];   \
                                       \
    ಸ್ಥಿರಾಂಕ uint32_t nb00 = src0->nb[0]; \
    ಸ್ಥಿರಾಂಕ uint32_t nb01 = src0->nb[1]; \
    ಸ್ಥಿರಾಂಕ uint32_t nb02 = src0->nb[2]; \
    ಸ್ಥಿರಾಂಕ uint32_t nb03 = src0->nb[3]; \
                                       \
    ಸ್ಥಿರಾಂಕ uint32_t nb0 = dst->nb[0];   \
    ಸ್ಥಿರಾಂಕ uint32_t nb1 = dst->nb[1];   \
    ಸ್ಥಿರಾಂಕ uint32_t nb2 = dst->nb[2];   \
    ಸ್ಥಿರಾಂಕ uint32_t nb3 = dst->nb[3];

ರಚನೆ htp_rope_context {
    int32_t n_dims;
    int32_t n_offs;
    int32_t ಬಹುಲಕ;
    int32_t n_ctx_orig;
    int32_t sections[4];

    ದಶಮಾಂಶ freq_ತಳ;
    ದಶಮಾಂಶ freq_scale;
    ದಶಮಾಂಶ ext_ಗುಣಕ;
    ದಶಮಾಂಶ attn_ಗುಣಕ;
    ದಶಮಾಂಶ beta_fast;
    ದಶಮಾಂಶ beta_slow;
    ದಶಮಾಂಶ theta_scale;
    ದಶಮಾಂಶ theta_scale_32;
    ದಶಮಾಂಶ theta_powers[32];
    ದಶಮಾಂಶ corr_dims[2];

    uint32_t src0_nrows_ಪ್ರತಿ_thread;

    ರಚನೆ htp_ops_context * octx;

    uint8_t * vtcm_ತಳ;
    size_t    spad_ಪ್ರತಿ_thread;
    size_t    theta_cache_ಸ್ಥಾನಾಂತರ;

    size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    size_t src0_ಅಡ್ಡಸಾಲು_stride;
    size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    size_t dst_ಅಡ್ಡಸಾಲು_stride;
    size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;
    uint32_t src0_nrows;
    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    uint32_t nrows;

    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne2_ne1;
    ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು div_ne1;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * freq_factors;
};

ಸ್ಥಿರ ದಶಮಾಂಶ rope_yarn_ramp(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಕೆಳಗಿನ, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ಮೇಲಿನ, ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ i0) {
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ y = (i0 / 2 - ಕೆಳಗಿನ) / MAX(0.001f, ಮೇಲಿನ - ಕೆಳಗಿನ);

    ಹಿಂತಿರುಗಿ (1 - MIN(1, MAX(0, y)));
}

// Compute one (cos, sin) pair into cache[i0], cache[i0+1] applying YaRN scaling.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rope_yarn_one(ದಶಮಾಂಶ theta, ದಶಮಾಂಶ freq_scale, ದಶಮಾಂಶ * corr_dims,
                                 uint32_t i0, ದಶಮಾಂಶ ext_ಗುಣಕ, ದಶಮಾಂಶ mscale,
                                 ದಶಮಾಂಶ * cache) {
    ದಶಮಾಂಶ theta_extrap = theta;

    // Get n-d rotational scaling corrected for extrapolation
    ದಶಮಾಂಶ theta_interp = freq_scale * theta_extrap;
    ದಶಮಾಂಶ theta_final  = theta_interp;
    ದಶಮಾಂಶ mscale_final = mscale;

    ಆದರೆ (ext_ಗುಣಕ != 0.0f) {
        ದಶಮಾಂಶ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_ಗುಣಕ;
        theta_final    = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale_final  *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    ಸ್ಥಿರಾಂಕ uint32_t b = i0 / 64;
    ಸ್ಥಿರಾಂಕ uint32_t k = (i0 % 64) / 2;
    cache[b * 64 + k]      = cosf(theta_final) * mscale_final;
    cache[b * 64 + 32 + k] = sinf(theta_final) * mscale_final;
}

// 32 thetas -> 32 deinterleaved pairs [cos[32] | sin[32]] at cache[i0].
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ rope_cache_hvx_32(ದಶಮಾಂಶ * cache, uint32_t i0,
                                     HVX_ಸದಿಶ v_theta,
                                     ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * freq_factors,
                                     HVX_ಸದಿಶ v_freq_scale,
                                     HVX_ಸದಿಶ v_mscale) {
    ಆದರೆ (freq_factors) {
        HVX_ಸದಿಶ v_ff = hvx_vmemu(freq_factors + i0 / 2);
        v_theta = hvx_vec_mul_f32_f32(v_theta, hvx_vec_inverse_f32(v_ff));
    }

    HVX_ಸದಿಶ v_theta_final = hvx_vec_mul_f32_f32(v_theta, v_freq_scale);
    HVX_ಸದಿಶ vcos;
    HVX_ಸದಿಶ vsin;
    hvx_vec_sincos_f32(v_theta_final, &vcos, &vsin);
    vcos = hvx_vec_mul_f32_f32(vcos, v_mscale);
    vsin = hvx_vec_mul_f32_f32(vsin, v_mscale);

    ಆದರೆ (((uintptr_t) (cache + i0)) % 128 == 0) {
        hvx_vmem(cache + i0 + 0)  = vcos;
        hvx_vmem(cache + i0 + 32) = vsin;
    } ಇಲ್ಲದಿದ್ದರೆ {
        hvx_vec_store_u(cache + i0 + 0,  32 * ಗಾತ್ರ(ದಶಮಾಂಶ), vcos);
        hvx_vec_store_u(cache + i0 + 32, 32 * ಗಾತ್ರ(ದಶಮಾಂಶ), vsin);
    }
}

ಸ್ಥಿರ __attribute__((noinline)) ಶೂನ್ಯ rope_cache_init(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    theta_ತಳ,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    freq_scale,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *  freq_factors,
                            ದಶಮಾಂಶ *        corr_dims,
                            ಸ್ಥಿರಾಂಕ uint32_t n_cache,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    ext_ಗುಣಕ,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    mscale,
                            ದಶಮಾಂಶ *        cache,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    theta_scale,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *  theta_powers,
                            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    theta_scale_32) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    ಆದರೆ (ext_ಗುಣಕ == 0.0f) {
        // Fast path: fully vectorized
        // We process 32 pairs (64 elements) per iteration.
        ಸ್ಥಿರಾಂಕ uint32_t n_ಖಂಡಗಳು = n_cache / 64;

        HVX_ಸದಿಶ v_theta_powers = hvx_vmemu(theta_powers);
        HVX_ಸದಿಶ v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_ಸದಿಶ v_mscale = hvx_vec_splat_f32(mscale);

        ದಶಮಾಂಶ theta_ಖಂಡ = theta_ತಳ;

        ಗಾಗಿ (uint32_t b = 0; b < n_ಖಂಡಗಳು; b++) {
            uint32_t i0 = b * 64;
            HVX_ಸದಿಶ v_theta_ತಳ = hvx_vec_splat_f32(theta_ಖಂಡ);
            HVX_ಸದಿಶ v_theta = hvx_vec_mul_f32_f32(v_theta_ತಳ, v_theta_powers);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            theta_ಖಂಡ *= theta_scale_32;
        }

        // Leftovers
        ದಶಮಾಂಶ theta = theta_ಖಂಡ;
        ಗಾಗಿ (uint32_t i0 = n_ಖಂಡಗಳು * 64; i0 < n_cache; i0 += 2) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ಗುಣಕ, mscale, cache);
            theta *= theta_scale;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ದಶಮಾಂಶ theta = theta_ತಳ;
        ಗಾಗಿ (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ಗುಣಕ, mscale, cache);
            theta *= theta_scale;
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ mrope_pick_theta(ದಶಮಾಂಶ theta_t, ದಶಮಾಂಶ theta_h, ದಶಮಾಂಶ theta_w, ದಶಮಾಂಶ theta_e,
                                     ಪೂರ್ಣಾಂಕ sector, ಸ್ಥಿರಾಂಕ int32_t sections[4], ಪೂರ್ಣಾಂಕ sec_w, ಪೂರ್ಣಾಂಕ sec_e,
                                     bool is_imrope) {
    ಆದರೆ (is_imrope) {
        ಆದರೆ      (sector % 3 == 0 && sector < 3 * sections[0]) { ಹಿಂತಿರುಗಿ theta_t; }
        ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector % 3 == 1 && sector < 3 * sections[1]) { ಹಿಂತಿರುಗಿ theta_h; }
        ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector % 3 == 2 && sector < 3 * sections[2]) { ಹಿಂತಿರುಗಿ theta_w; }
        ಇಲ್ಲದಿದ್ದರೆ                                                   { ಹಿಂತಿರುಗಿ theta_e; }
    }
    ಆದರೆ      (sector < sections[0]) { ಹಿಂತಿರುಗಿ theta_t; }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector < sec_w)       { ಹಿಂತಿರುಗಿ theta_h; }
    ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector < sec_e)       { ಹಿಂತಿರುಗಿ theta_w; }
    ಇಲ್ಲದಿದ್ದರೆ                           { ಹಿಂತಿರುಗಿ theta_e; }
}

// lane j is 1 when (j % 3) == rem
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ __attribute__((aligned(128))) mrope_mod3_eq0[32] = {
    1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0
};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ __attribute__((aligned(128))) mrope_mod3_eq1[32] = {
    0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1
};
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ __attribute__((aligned(128))) mrope_mod3_eq2[32] = {
    0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0
};

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ __attribute__((aligned(128))) mrope_k_ramp[32] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ_Pred mrope_ಮುಖವಾಡ_eq1(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * m) {
    ಹಿಂತಿರುಗಿ Q6_Q_vcmp_gt_VsfVsf(hvx_vmemu(m), Q6_V_vzero());
}

// IMROPE without wrap: theta[k] = pos[k % 3] * scale^k
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ mrope_thetas_imrope_mod3(ದಶಮಾಂಶ pos_t, ದಶಮಾಂಶ pos_h, ದಶಮಾಂಶ pos_w,
                                                  uint32_t k0, HVX_ಸದಿಶ v_powers, ದಶಮಾಂಶ scale_ಖಂಡ) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ r = (ಪೂರ್ಣಾಂಕ) (k0 % 3);
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * mt = (r == 0) ? mrope_mod3_eq0 : (r == 1) ? mrope_mod3_eq2 : mrope_mod3_eq1;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * mh = (r == 0) ? mrope_mod3_eq1 : (r == 1) ? mrope_mod3_eq0 : mrope_mod3_eq2;

    HVX_ಸದಿಶ v = hvx_vec_splat_f32(pos_w);
    v = Q6_V_vmux_QVV(mrope_ಮುಖವಾಡ_eq1(mh), hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(mrope_ಮುಖವಾಡ_eq1(mt), hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    ಹಿಂತಿರುಗಿ hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_ಖಂಡ));
}

// Contiguous MROPE without wrap: theta[k] = pos[section(k)] * scale^k
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ mrope_thetas_contig(ದಶಮಾಂಶ pos_t, ದಶಮಾಂಶ pos_h, ದಶಮಾಂಶ pos_w, ದಶಮಾಂಶ pos_e,
                                             uint32_t k0, ಪೂರ್ಣಾಂಕ s0, ಪೂರ್ಣಾಂಕ sec_w, ಪೂರ್ಣಾಂಕ sec_e,
                                             HVX_ಸದಿಶ v_powers, ದಶಮಾಂಶ scale_ಖಂಡ) {
    HVX_ಸದಿಶ v_k = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vec_splat_f32((ದಶಮಾಂಶ) k0), hvx_vmemu(mrope_k_ramp));
    HVX_ಸದಿಶ_Pred lt_s0 = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((ದಶಮಾಂಶ) s0), v_k);
    HVX_ಸದಿಶ_Pred lt_sw = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((ದಶಮಾಂಶ) sec_w), v_k);
    HVX_ಸದಿಶ_Pred lt_se = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((ದಶಮಾಂಶ) sec_e), v_k);

    HVX_ಸದಿಶ v = hvx_vec_splat_f32(pos_e);
    v = Q6_V_vmux_QVV(lt_se, hvx_vec_splat_f32(pos_w), v);
    v = Q6_V_vmux_QVV(lt_sw, hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(lt_s0, hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    ಹಿಂತಿರುಗಿ hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_ಖಂಡ));
}

// pos_t/h/w/e: the four position ids for this sequence step (t=time, h=height, w=width, e=extra).
// sections[4]: number of head dims assigned to each position component.
ಸ್ಥಿರ __attribute__((noinline)) ಶೂನ್ಯ mrope_cache_init(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    pos_t,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    pos_h,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    pos_w,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    pos_e,
                             ಸ್ಥಿರಾಂಕ int32_t  sections[4],
                             ಸ್ಥಿರಾಂಕ bool     is_imrope,
                             ಸ್ಥಿರಾಂಕ bool     indep_sects,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    freq_scale,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *  freq_factors,
                             ದಶಮಾಂಶ *        corr_dims,
                             ಸ್ಥಿರಾಂಕ uint32_t n_cache,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    ext_ಗುಣಕ,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    mscale,
                             ದಶಮಾಂಶ *        cache,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    theta_scale,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *  theta_powers,
                             ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ    theta_scale_32) {
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sec_w     = sections[0] + sections[1];
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sec_e     = sec_w + sections[2];
    ಸ್ಥಿರಾಂಕ uint32_t n_ಜೋಡಿಗಳು = n_cache / 2;

    ಸ್ಥಿರಾಂಕ bool no_wrap = (sect_dims > 0) && (n_ಜೋಡಿಗಳು <= (uint32_t) sect_dims);
    ಸ್ಥಿರಾಂಕ bool imrope_mod3 = is_imrope && !indep_sects && no_wrap
        && sections[0] > 0 && sections[1] > 0 && sections[2] > 0
        && n_ಜೋಡಿಗಳು <= (uint32_t) (3 * sections[0])
        && n_ಜೋಡಿಗಳು <= (uint32_t) (3 * sections[1])
        && n_ಜೋಡಿಗಳು <= (uint32_t) (3 * sections[2]);
    ಸ್ಥಿರಾಂಕ bool contig = !is_imrope && !indep_sects && no_wrap;

    ಆದರೆ (ext_ಗುಣಕ == 0.0f && (imrope_mod3 || contig)) {
        HVX_ಸದಿಶ v_powers     = hvx_vmemu(theta_powers);
        HVX_ಸದಿಶ v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_ಸದಿಶ v_mscale     = hvx_vec_splat_f32(mscale);
        ದಶಮಾಂಶ scale_ಖಂಡ = 1.0f;
        ಸ್ಥಿರಾಂಕ uint32_t n_ಖಂಡಗಳು = n_cache / 64;

        ಗಾಗಿ (uint32_t b = 0; b < n_ಖಂಡಗಳು; b++) {
            ಸ್ಥಿರಾಂಕ uint32_t i0 = b * 64;
            ಸ್ಥಿರಾಂಕ uint32_t k0 = b * 32;
            HVX_ಸದಿಶ v_theta = imrope_mod3
                ? mrope_thetas_imrope_mod3(pos_t, pos_h, pos_w, k0, v_powers, scale_ಖಂಡ)
                : mrope_thetas_contig(pos_t, pos_h, pos_w, pos_e, k0, sections[0], sec_w, sec_e,
                                      v_powers, scale_ಖಂಡ);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            scale_ಖಂಡ *= theta_scale_32;
        }

        ದಶಮಾಂಶ theta_k = scale_ಖಂಡ;
        ಗಾಗಿ (uint32_t k = n_ಖಂಡಗಳು * 32; k < n_ಜೋಡಿಗಳು; k++) {
            ಸ್ಥಿರಾಂಕ uint32_t i0 = 2 * k;
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ pos = mrope_pick_theta(pos_t, pos_h, pos_w, pos_e,
                                               (ಪೂರ್ಣಾಂಕ) k, sections, sec_w, sec_e, is_imrope);
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ff = freq_factors ? freq_factors[k] : 1.0f;
            rope_yarn_one(pos * theta_k / ff, freq_scale, corr_dims, i0, ext_ಗುಣಕ, mscale, cache);
            theta_k *= theta_scale;
        }
        ಹಿಂತಿರುಗಿ;
    }

    ದಶಮಾಂಶ theta_t = pos_t;
    ದಶಮಾಂಶ theta_h = pos_h;
    ದಶಮಾಂಶ theta_w = pos_w;
    ದಶಮಾಂಶ theta_e = pos_e;

    ಸ್ಥಿರಾಂಕ bool use_hvx = (ext_ಗುಣಕ == 0.0f);
    ದಶಮಾಂಶ __attribute__((aligned(128))) thetas[32];
    uint32_t n_thetas = 0;
    uint32_t ಖಂಡ_i0 = 0;

    HVX_ಸದಿಶ v_freq_scale = hvx_vec_splat_f32(freq_scale);
    HVX_ಸದಿಶ v_mscale     = hvx_vec_splat_f32(mscale);

    ಗಾಗಿ (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ sector = (i0 / 2) % sect_dims;

        ಆದರೆ (indep_sects) {
            // Reset theta when crossing into a new section.
            ಆದರೆ      (sector == 0)           { theta_t = pos_t; }
            ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector == sections[0]) { theta_h = pos_h; }
            ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector == sec_w)       { theta_w = pos_w; }
            ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (sector == sec_e)       { theta_e = pos_e; }
        }

        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ theta = mrope_pick_theta(theta_t, theta_h, theta_w, theta_e,
                                             sector, sections, sec_w, sec_e, is_imrope);

        ಆದರೆ (use_hvx) {
            ಆದರೆ (n_thetas == 0) {
                ಖಂಡ_i0 = i0;
            }
            thetas[n_thetas++] = theta;
            ಆದರೆ (n_thetas == 32) {
                rope_cache_hvx_32(cache, ಖಂಡ_i0, hvx_vmemu(thetas), freq_factors, v_freq_scale, v_mscale);
                n_thetas = 0;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ಗುಣಕ, mscale, cache);
        }

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }

    ಗಾಗಿ (uint32_t k = 0; k < n_thetas; k++) {
        ಸ್ಥಿರಾಂಕ uint32_t i0 = ಖಂಡ_i0 + 2 * k;
        ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn_one(thetas[k] / ff, freq_scale, corr_dims, i0, ext_ಗುಣಕ, mscale, cache);
    }
}

#ವ್ಯಾಖ್ಯಾನಿಸು M_PI 3.1415926535897932384626433

ಸ್ಥಿರ ಶೂನ್ಯ rope_corr_dims(ಪೂರ್ಣಾಂಕ     n_dims,
                           ಪೂರ್ಣಾಂಕ     n_ctx_orig,
                           ದಶಮಾಂಶ   freq_ತಳ,
                           ದಶಮಾಂಶ   beta_fast,
                           ದಶಮಾಂಶ   beta_slow,
                           ದಶಮಾಂಶ * dims) {
    ದಶಮಾಂಶ ಆರಂಭ = floorf(n_dims * logf(n_ctx_orig / (beta_fast * 2 * (ದಶಮಾಂಶ) M_PI)) / (2 * logf(freq_ತಳ)));
    ದಶಮಾಂಶ ಕೊನೆ   = ceilf(n_dims * logf(n_ctx_orig / (beta_slow * 2 * (ದಶಮಾಂಶ) M_PI)) / (2 * logf(freq_ತಳ)));
    dims[0]     = MAX(0, ಆರಂಭ);
    dims[1]     = MIN(n_dims - 1, ಕೊನೆ);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_rope_neox_mul(HVX_ಸದಿಶ v0, HVX_ಸದಿಶ v1, HVX_ಸದಿಶ vcos, HVX_ಸದಿಶ vsin,
                                     HVX_ಸದಿಶ * o0, HVX_ಸದಿಶ * o1) {
    HVX_ಸದಿಶ vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, vcos);
    HVX_ಸದಿಶ vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, vsin);
    HVX_ಸದಿಶ vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, vcos);
    HVX_ಸದಿಶ vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, vsin);
    *o0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s));
    *o1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c));
}

// theta_cache full 32-pair blocks are deinterleaved [cos | sin].
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_rope_neox_f32_aa(ದಶಮಾಂಶ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src0, uint32_t ne, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ theta_cache) {
    ಸ್ಥಿರಾಂಕ uint32_t he = ne / 2;
    ಸ್ಥಿರಾಂಕ uint32_t nvec = he / 32;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = he % 32;

    ಆದರೆ (nloe == 0) {
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * vs = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src0;
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * vt = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) theta_cache;
        HVX_ಸದಿಶ * vd = (HVX_ಸದಿಶ *) dst;
        ಗಾಗಿ (uint32_t i = 0; i < nvec; i++) {
            HVX_ಸದಿಶ o0, o1;
            hvx_rope_neox_mul(vs[i], vs[nvec + i], vt[i * 2 + 0], vt[i * 2 + 1], &o0, &o1);
            vd[i] = o0;
            vd[nvec + i] = o1;
        }
        ಹಿಂತಿರುಗಿ;
    }

    ಗಾಗಿ (uint32_t i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ o0, o1;
        hvx_rope_neox_mul(((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src0)[i],
                          hvx_vmemu(src0 + he + i * 32),
                          ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) theta_cache)[i * 2 + 0],
                          ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) theta_cache)[i * 2 + 1],
                          &o0, &o1);
        ((HVX_ಸದಿಶ *) dst)[i] = o0;
        hvx_vmemu(dst + he + i * 32) = o1;
    }

    HVX_ಸದಿಶ v0 = hvx_vmemu(src0 + nvec * 32);
    HVX_ಸದಿಶ v1 = hvx_vmemu(src0 + he + nvec * 32);
    HVX_ಸದಿಶ vcos = hvx_vmemu(theta_cache + nvec * 64);
    HVX_ಸದಿಶ vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);
    HVX_ಸದಿಶ o0, o1;
    hvx_rope_neox_mul(v0, v1, vcos, vsin, &o0, &o1);
    hvx_vec_store_u(dst + nvec * 32, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), o0);
    hvx_vec_store_u(dst + he + nvec * 32, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), o1);
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_rope_f32_aa(ದಶಮಾಂಶ * ಸೀಮಿತ dst, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ src0, uint32_t ne, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ theta_cache) {
    ಸ್ಥಿರಾಂಕ uint32_t nvec = ne / 64;
    ಸ್ಥಿರಾಂಕ uint32_t nloe = ne % 64;

    ಗಾಗಿ (uint32_t i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v0 = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src0)[i * 2 + 0];
        HVX_ಸದಿಶ v1 = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src0)[i * 2 + 1];

        HVX_ಸದಿಶ vcos = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) theta_cache)[i * 2 + 0];
        HVX_ಸದಿಶ vsin = ((ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) theta_cache)[i * 2 + 1];

        HVX_ಸದಿಶ_Pair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

        HVX_ಸದಿಶ vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
        HVX_ಸದಿಶ vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
        HVX_ಸದಿಶ vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
        HVX_ಸದಿಶ vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

        HVX_ಸದಿಶ v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_ಸದಿಶ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        HVX_ಸದಿಶ_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

        ((HVX_ಸದಿಶ *) dst)[i * 2 + 0] = Q6_V_lo_W(vstore);
        ((HVX_ಸದಿಶ *) dst)[i * 2 + 1] = Q6_V_hi_W(vstore);
    }

    ಆದರೆ (nloe > 0) {
        ಆದರೆ (nloe <= 32) {
            HVX_ಸದಿಶ v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_ಸದಿಶ vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_ಸದಿಶ vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_ಸದಿಶ_Pair vx0_x1 = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, -4);

            HVX_ಸದಿಶ vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_ಸದಿಶ vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_ಸದಿಶ vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_ಸದಿಶ vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_ಸದಿಶ v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_ಸದಿಶ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_ಸದಿಶ_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            hvx_vec_store_u(dst + nvec * 64, nloe * ಗಾತ್ರ(ದಶಮಾಂಶ), Q6_V_lo_W(vstore));
        } ಇಲ್ಲದಿದ್ದರೆ {
            HVX_ಸದಿಶ v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_ಸದಿಶ v1 = hvx_vmemu(src0 + nvec * 64 + 32);

            HVX_ಸದಿಶ vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_ಸದಿಶ vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_ಸದಿಶ_Pair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

            HVX_ಸದಿಶ vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_ಸದಿಶ vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_ಸದಿಶ vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_ಸದಿಶ vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_ಸದಿಶ v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_ಸದಿಶ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_ಸದಿಶ_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            ((HVX_ಸದಿಶ *) dst)[nvec * 2 + 0] = Q6_V_lo_W(vstore);
            hvx_vec_store_u(dst + nvec * 64 + 32, (nloe - 32) * ಗಾತ್ರ(ದಶಮಾಂಶ), Q6_V_hi_W(vstore));
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ಸಾಲೊಳಗೆ rope_basic_f32_inplace(ರಚನೆ htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ theta_cache) {
    ಸ್ಥಿರಾಂಕ uint32_t n_offs = rctx->n_offs;
    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (uint32_t i = 0; i < nr; i++) {
        ದಶಮಾಂಶ * s = (ದಶಮಾಂಶ *) (src + i * rctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        hvx_rope_f32_aa(s + n_offs, s + n_offs, rctx->n_dims, theta_cache);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ಸಾಲೊಳಗೆ rope_neox_f32_inplace(ರಚನೆ htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, uint32_t ne, ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ theta_cache) {
    ಸ್ಥಿರಾಂಕ uint32_t n_offs = rctx->n_offs;
    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (uint32_t i = 0; i < nr; i++) {
        ದಶಮಾಂಶ * s = (ದಶಮಾಂಶ *) (src + i * rctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
        hvx_rope_neox_f32_aa(s + n_offs, s + n_offs, ne, theta_cache);
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ rope_ಕೆಲಸ_f32(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ nth, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_rope_context * rctx = (ರಚನೆ htp_rope_context *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = rctx->octx;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src2 = octx->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    htp_rope_preamble;

    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows = rctx->nrows;
    ಸ್ಥಿರಾಂಕ uint32_t src0_nrows_ಪ್ರತಿ_thread = rctx->src0_nrows_ಪ್ರತಿ_thread;

    ಸ್ಥಿರಾಂಕ uint32_t src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು = rctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows_ಪ್ರತಿ_thread * ith;
    ಸ್ಥಿರಾಂಕ uint32_t src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು   = MIN(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು + src0_nrows_ಪ್ರತಿ_thread, rctx->ಅಡ್ಡಸಾಲು_ಆರಂಭ + src0_nrows);

    // no work for this thread
    ಆದರೆ (src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
        ಹಿಂತಿರುಗಿ;
    }

    ಸ್ಥಿರಾಂಕ int32_t ಬಹುಲಕ    = rctx->ಬಹುಲಕ;
    // MROPE, IMROPE and VISION use NEOX-style pairing for the rotation
    ಸ್ಥಿರಾಂಕ bool    is_neox = (ಬಹುಲಕ & HTP_ROPE_TYPE_NEOX) || (ಬಹುಲಕ & HTP_ROPE_TYPE_MROPE);
    ಸ್ಥಿರಾಂಕ bool    is_vision = (ಬಹುಲಕ == HTP_ROPE_TYPE_VISION);

    // VTCM setup
    uint8_t * src0_spad_ತಳ = rctx->vtcm_ತಳ + (ith * rctx->spad_ಪ್ರತಿ_thread);
    ದಶಮಾಂಶ *   theta_cache    = (ದಶಮಾಂಶ *) (src0_spad_ತಳ);
              src0_spad_ತಳ = src0_spad_ತಳ + rctx->theta_cache_ಸ್ಥಾನಾಂತರ;

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];
    ಸ್ಥಿರಾಂಕ int32_t * pos = (ಸ್ಥಿರಾಂಕ int32_t *) (uintptr_t) src1->ದತ್ತಾಂಶ;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * freq_factors = rctx->freq_factors;

    ಸ್ಥಿರಾಂಕ uint32_t i3_ಆರಂಭ = fastdiv(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು, &rctx->div_ne2_ne1);
    ಸ್ಥಿರಾಂಕ uint32_t rem      = fastmodulo(src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು, ne2 * ne1, &rctx->div_ne2_ne1);
    ಸ್ಥಿರಾಂಕ uint32_t i2_ಆರಂಭ = fastdiv(rem, &rctx->div_ne1);
    ಸ್ಥಿರಾಂಕ uint32_t i1_ಆರಂಭ = fastmodulo(rem, ne1, &rctx->div_ne1);

    uint32_t ir = src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು;
    uint32_t prev_i2 = (uint32_t) -1;
    uint32_t cur_ಸ್ಥಾನ = 0;

    ಗಾಗಿ (uint32_t i3 = i3_ಆರಂಭ; i3 < ne3; i3++) { // batch
        ಸ್ಥಿರಾಂಕ uint32_t i2_init = (i3 == i3_ಆರಂಭ) ? i2_ಆರಂಭ : 0;
        ಗಾಗಿ (uint32_t i2 = i2_init; i2 < ne2; i2++) { // seq-len
            ಸ್ಥಿರಾಂಕ uint32_t i1_init = (i3 == i3_ಆರಂಭ && i2 == i2_ಆರಂಭ) ? i1_ಆರಂಭ : 0;
            ಗಾಗಿ (uint32_t i1 = i1_init; i1 < ne1; ) { // attn-heads
                ಆದರೆ (ir >= src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು) ಹೋಗು ಮುಗಿದ;

                // Rows in this block
                ಸ್ಥಿರಾಂಕ uint32_t nrows = MIN(src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು - ir, ne1 - i1);

                // Depth before prefetch
                ಸ್ಥಿರಾಂಕ uint32_t dma_ಆಳ = dma_ಸರತಿ_ಆಳ(dma_q);

                // Prefetch up to 2 blocks
                ಸ್ಥಿರಾಂಕ uint32_t p_nrows = MIN(nrows, 2 * HTP_ROPE_SPAD_ಖಂಡ);
                ಗಾಗಿ (uint32_t pr = 0; pr < p_nrows; pr += HTP_ROPE_SPAD_ಖಂಡ) {
                    ಸ್ಥಿರಾಂಕ uint32_t pnr = MIN(nrows - pr, HTP_ROPE_SPAD_ಖಂಡ);
                    ಸ್ಥಿರಾಂಕ uint32_t ಸ್ಥಾನ = (cur_ಸ್ಥಾನ + pr / HTP_ROPE_SPAD_ಖಂಡ) % HTP_ROPE_SPAD_NSLOTS;
                    uint8_t * spad_ಸ್ಥಾನ = rope_spad_ಸ್ಥಾನ(src0_spad_ತಳ, ಸ್ಥಾನ, rctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
                    ಸ್ಥಿರಾಂಕ dma_addr_t src0_ದತ್ತಾಂಶ = src0->ದತ್ತಾಂಶ + i3 * nb03 + i2 * nb02 + (i1 + pr) * nb01;

                    // Dummy DMA transaction for sequencing (interleaving wr, rd, wr, rd, ...)
                    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst->ದತ್ತಾಂಶ, spad_ಸ್ಥಾನ), 0, 0, 0, 0);

                    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(spad_ಸ್ಥಾನ, src0_ದತ್ತಾಂಶ),
                        rctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, rctx->src0_ಅಡ್ಡಸಾಲು_stride, rctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, pnr);
                }

                // Update theta cache
                ಆದರೆ (i2 != prev_i2) {
                    prev_i2 = i2;

                    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                    // VISION rotates the full row; other modes only rotate n_dims.
                    ಸ್ಥಿರಾಂಕ uint32_t n_cache = is_vision ? ne0 : (uint32_t) rctx->n_dims;
                    ಸ್ಥಿರಾಂಕ bool is_mrope = (rctx->ಬಹುಲಕ & HTP_ROPE_TYPE_MROPE) != 0;
                    ಆದರೆ (is_mrope) {
                        // src1 holds four position arrays stacked along ne0:
                        // pos[i2], pos[i2+ne2], pos[i2+ne2*2], pos[i2+ne2*3]
                        ಸ್ಥಿರಾಂಕ bool is_imrope = (rctx->ಬಹುಲಕ == HTP_ROPE_TYPE_IMROPE);
                        mrope_cache_init(
                            (ದಶಮಾಂಶ) pos[i2],
                            (ದಶಮಾಂಶ) pos[i2 + ne2],
                            (ದಶಮಾಂಶ) pos[i2 + ne2 * 2],
                            (ದಶಮಾಂಶ) pos[i2 + ne2 * 3],
                            rctx->sections, is_imrope, is_vision,
                            rctx->freq_scale, freq_factors, rctx->corr_dims,
                            n_cache, rctx->ext_ಗುಣಕ, rctx->attn_ಗುಣಕ,
                            theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                       rope_cache_init(pos[i2], rctx->freq_scale, freq_factors, rctx->corr_dims,
                                        n_cache, rctx->ext_ಗುಣಕ, rctx->attn_ಗುಣಕ,
                                        theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                }

                // Skip output DMA transactions from prev block (if any)
                ಗಾಗಿ (uint32_t d = 0; d < dma_ಆಳ; d++) { dma_ಸರತಿ_ತೆಗೆ_nowait(dma_q); }

                // Compute loop
                ಸ್ಥಿರಾಂಕ uint32_t ne = is_vision ? ne0 : rctx->n_dims;
                ಸ್ಥಿರಾಂಕ uint32_t ತಳ_i1 = i1;
                ಸ್ಥಿರಾಂಕ uint32_t ತಳ_ir = ir;

                ಗಾಗಿ (uint32_t cnr = 0, cr = 0; cr < nrows; cr += cnr) {
                    cnr = MIN(nrows - cr, HTP_ROPE_SPAD_ಖಂಡ);
                    ಸ್ಥಿರಾಂಕ uint32_t ಸ್ಥಾನ = (cur_ಸ್ಥಾನ + cr / HTP_ROPE_SPAD_ಖಂಡ) % HTP_ROPE_SPAD_NSLOTS;
                    ಸ್ಥಿರಾಂಕ uint32_t cur_ir   = ತಳ_ir + cr;
                    ಸ್ಥಿರಾಂಕ uint32_t cur_i1   = ತಳ_i1 + cr;

                    dma_ಸರತಿ_ತೆಗೆ(dma_q);
                    uint8_t * cur_spad = (uint8_t *) dma_ಸರತಿ_ತೆಗೆ(dma_q).dst;

                    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);
                    ಆದರೆ (is_neox || is_vision) {
                        rope_neox_f32_inplace(rctx, cur_spad, cnr, ne, theta_cache);
                    } ಇಲ್ಲದಿದ್ದರೆ {
                        rope_basic_f32_inplace(rctx, cur_spad, cnr, theta_cache);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);

                    ಸ್ಥಿರಾಂಕ dma_addr_t dst_ದತ್ತಾಂಶ = dst->ದತ್ತಾಂಶ + i3 * nb3 + i2 * nb2 + cur_i1 * nb1;
                    dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_ದತ್ತಾಂಶ, cur_spad),
                        rctx->dst_ಅಡ್ಡಸಾಲು_stride, rctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, rctx->dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, cnr);

                    // Prefetch 2 blocks ahead into the slot just freed
                    ಆದರೆ ((cr + 2 * HTP_ROPE_SPAD_ಖಂಡ) < nrows) {
                        ಸ್ಥಿರಾಂಕ uint32_t p_cr   = cr + 2 * HTP_ROPE_SPAD_ಖಂಡ;
                        ಸ್ಥಿರಾಂಕ uint32_t pnr    = MIN(nrows - p_cr, HTP_ROPE_SPAD_ಖಂಡ);
                        ಸ್ಥಿರಾಂಕ uint32_t p_ಸ್ಥಾನ = (cur_ಸ್ಥಾನ + p_cr / HTP_ROPE_SPAD_ಖಂಡ) % HTP_ROPE_SPAD_NSLOTS;
                        uint8_t * p_spad      = rope_spad_ಸ್ಥಾನ(src0_spad_ತಳ, p_ಸ್ಥಾನ, rctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned);
                        ಸ್ಥಿರಾಂಕ dma_addr_t p_src0_ದತ್ತಾಂಶ = src0->ದತ್ತಾಂಶ + i3 * nb03 + i2 * nb02 + (ತಳ_i1 + p_cr) * nb01;

                        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(p_spad, p_src0_ದತ್ತಾಂಶ),
                            rctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned, rctx->src0_ಅಡ್ಡಸಾಲು_stride, rctx->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, pnr);
                    }
                }

                ಸ್ಥಿರಾಂಕ uint32_t n_chunks = (nrows + HTP_ROPE_SPAD_ಖಂಡ - 1) / HTP_ROPE_SPAD_ಖಂಡ;
                cur_ಸ್ಥಾನ = (cur_ಸ್ಥಾನ + n_chunks) % HTP_ROPE_SPAD_NSLOTS;

                ir += nrows;
                i1 += nrows;
            }
        }
    }

ಮುಗಿದ:
    dma_ಸರತಿ_flush(dma_q);

    FARF(ಮೇಲಿನ_2, "rope-f32: %d/%d: (%u:%u)\n", ith, nth, src0_ಆರಂಭ_ಅಡ್ಡಸಾಲು, src0_ಕೊನೆ_ಅಡ್ಡಸಾಲು);
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ execute_op_rope_f32(ರಚನೆ htp_ops_context * octx) {
    ಪೂರ್ಣಾಂಕ err = HTP_ಸ್ಥಿತಿ_OK;

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src1 = octx->src[1];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src2 = octx->src[2];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಆಯ್ಕೆ (octx->op) {
        ಸ್ಥಿತಿ HTP_OP_ROPE:
            ನಿಲ್ಲಿಸು;

        ಪೂರ್ವನಿಯೋಜಿತ:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_rope_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_rope_kernel_params *) octx->kernel_params;
    ಆದರೆ (!htp_ops_context_ಗಣ_n_ಎಳೆಗಳು(octx, kparams->n_ಎಳೆಗಳು)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_INVAL_PARAMS;
    }
    ಪ್ರತಿಜ್ಞಾ(octx->ctx->vtcm_ಗಾತ್ರ >= kparams->vtcm_ಗಾತ್ರ);

    ಆದರೆ (htp_tensor_is_extended(src1)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ size_t dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = dst->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ);

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t nrows     = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;

    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 0;
        htp_tensor_mdev_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk(dst, ಗಾತ್ರ(ದಶಮಾಂಶ), (uint32_t) dst_ದತ್ತಾಂಶ_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, &ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk);
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(
            ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        nrows     = range.ಎಣಿಕೆ_2;
    }

    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ಸ್ಥಿರಾಂಕ uint32_t n_ಎಳೆಗಳು = octx->n_ಎಳೆಗಳು;

    ಸ್ಥಿರಾಂಕ uint32_t ne0 = dst->ne[0];
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ   = src0->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t src0_ಅಡ್ಡಸಾಲು_stride = src0->nb[1];
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ    = dst->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_stride  = dst->nb[1];

    ರಚನೆ htp_rope_context rctx;
    memset(&rctx, 0, ಗಾತ್ರ(ರಚನೆ htp_rope_context));

    rctx.octx                  = octx;
    rctx.vtcm_ತಳ             = (uint8_t *) octx->ctx->vtcm_ತಳ;
    rctx.spad_ಪ್ರತಿ_thread       = kparams->spad_ಪ್ರತಿ_thread;
    rctx.theta_cache_ಸ್ಥಾನಾಂತರ    = kparams->theta_cache_ಸ್ಥಾನಾಂತರ;

    ಆದರೆ (src2) {
        dma_ಸರತಿ * dma_q = octx->ctx->dma[0];
        ಸ್ಥಿರಾಂಕ size_t ff_ಗಾತ್ರ = src2->ne[0] * ಗಾತ್ರ(ದಶಮಾಂಶ);
        ದಶಮಾಂಶ * vtcm_freq_factors = (ದಶಮಾಂಶ *) (rctx.vtcm_ತಳ + kparams->freq_factors_ಸ್ಥಾನಾಂತರ);
        dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(vtcm_freq_factors, src2->ದತ್ತಾಂಶ),
                       kparams->freq_factors_ಗಾತ್ರ, 0, ff_ಗಾತ್ರ, 1);
        dma_ಸರತಿ_ತೆಗೆ(dma_q);
        rctx.freq_factors = vtcm_freq_factors;
    }

    ಸ್ಥಿರಾಂಕ int32_t * op_params = &octx->op_params[0];
    rctx.n_dims     = ((ಸ್ಥಿರಾಂಕ int32_t *) op_params)[1];
    rctx.ಬಹುಲಕ       = ((ಸ್ಥಿರಾಂಕ int32_t *) op_params)[2];
    rctx.n_ctx_orig = ((ಸ್ಥಿರಾಂಕ int32_t *) op_params)[4];
    rctx.n_offs     = ((ಸ್ಥಿರಾಂಕ int32_t *) op_params)[15];

    ಸ್ಮರಣೆ_ನಕಲು(&rctx.freq_ತಳ,   (int32_t *) op_params + 5,  ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&rctx.freq_scale,  (int32_t *) op_params + 6,  ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&rctx.ext_ಗುಣಕ,  (int32_t *) op_params + 7,  ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&rctx.attn_ಗುಣಕ, (int32_t *) op_params + 8,  ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&rctx.beta_fast,   (int32_t *) op_params + 9,  ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&rctx.beta_slow,   (int32_t *) op_params + 10, ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಮರಣೆ_ನಕಲು(&rctx.sections,    (int32_t *) op_params + 11, ಗಾತ್ರ(ಪೂರ್ಣಾಂಕ) * 4);

    rctx.theta_scale = powf(rctx.freq_ತಳ, -2.0f / rctx.n_dims);
    rctx.theta_powers[0] = 1.0f;
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ j = 1; j < 32; j++) {
        rctx.theta_powers[j] = rctx.theta_powers[j - 1] * rctx.theta_scale;
    }
    rctx.theta_scale_32 = rctx.theta_powers[31] * rctx.theta_scale;

    rope_corr_dims(rctx.n_dims, rctx.n_ctx_orig, rctx.freq_ತಳ, rctx.beta_fast, rctx.beta_slow, rctx.corr_dims);

    rctx.src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ         = src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    rctx.src0_ಅಡ್ಡಸಾಲು_stride       = src0_ಅಡ್ಡಸಾಲು_stride;
    rctx.dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ          = dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    rctx.dst_ಅಡ್ಡಸಾಲು_stride        = dst_ಅಡ್ಡಸಾಲು_stride;
    rctx.src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned = kparams->src0_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ_aligned;

    rctx.src0_nrows            = nrows;
    rctx.nrows                 = nrows;
    rctx.ಅಡ್ಡಸಾಲು_ಆರಂಭ             = ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    rctx.src0_nrows_ಪ್ರತಿ_thread = fastdiv(nrows + n_ಎಳೆಗಳು - 1, &octx->n_ಎಳೆಗಳು_div);
    rctx.div_ne2_ne1           = kparams->div_ne2_ne1;
    rctx.div_ne1               = kparams->div_ne1;

    FARF(ಮೇಲಿನ_2, "rope-f32 n-rows %u n-dims %d ne0 %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n", rctx.src0_nrows, rctx.n_dims, ne0,
         rctx.ext_ಗುಣಕ, rctx.theta_scale, rctx.attn_ಗುಣಕ);

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, rope_ಕೆಲಸ_f32, &rctx, n_ಎಳೆಗಳು);

    ಹಿಂತಿರುಗಿ err;
}

ಪೂರ್ಣಾಂಕ op_rope(ರಚನೆ htp_ops_context * octx) {
    ಆಯ್ಕೆ (octx->src[0]->type) {
        ಸ್ಥಿತಿ HTP_TYPE_F32:
            ಹಿಂತಿರುಗಿ execute_op_rope_f32(octx);

        ಪೂರ್ವನಿಯೋಜಿತ:
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }
}
