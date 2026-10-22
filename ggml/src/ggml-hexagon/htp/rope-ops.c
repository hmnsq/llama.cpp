#निर्देशः clang diagnostic ignored "-Wunused-variable"
#निर्देशः clang diagnostic ignored "-Wunused-function"
#निर्देशः clang diagnostic ignored "-Wunused-but-set-variable"

#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#अन्तर्भाव <math.h>
#अन्तर्भाव <string.h>
#अन्तर्भाव <stdlib.h>

#अन्तर्भाव "dma-queue.h"
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "hex-fastdiv.h"

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "rope-ops.h"

// Redefined the rope type constants as we can't include ggml.h
#परिभाषय HTP_ROPE_TYPE_प्रसामान्यम् 0
#परिभाषय HTP_ROPE_TYPE_NEOX   2
#परिभाषय HTP_ROPE_TYPE_MROPE  8
#परिभाषय HTP_ROPE_TYPE_VISION 24
#परिभाषय HTP_ROPE_TYPE_IMROPE 40

#परिभाषय htp_rope_preamble              \
    ध्रुव uint32_t ne00 = src0->ne[0]; \
    ध्रुव uint32_t ne01 = src0->ne[1]; \
    ध्रुव uint32_t ne02 = src0->ne[2]; \
    ध्रुव uint32_t ne03 = src0->ne[3]; \
                                       \
    ध्रुव uint32_t ne0 = dst->ne[0];   \
    ध्रुव uint32_t ne1 = dst->ne[1];   \
    ध्रुव uint32_t ne2 = dst->ne[2];   \
    ध्रुव uint32_t ne3 = dst->ne[3];   \
                                       \
    ध्रुव uint32_t nb00 = src0->nb[0]; \
    ध्रुव uint32_t nb01 = src0->nb[1]; \
    ध्रुव uint32_t nb02 = src0->nb[2]; \
    ध्रुव uint32_t nb03 = src0->nb[3]; \
                                       \
    ध्रुव uint32_t nb0 = dst->nb[0];   \
    ध्रुव uint32_t nb1 = dst->nb[1];   \
    ध्रुव uint32_t nb2 = dst->nb[2];   \
    ध्रुव uint32_t nb3 = dst->nb[3];

संरचना htp_rope_context {
    int32_t n_dims;
    int32_t n_offs;
    int32_t बहुलकम्;
    int32_t n_ctx_orig;
    int32_t sections[4];

    भिन्न freq_आधारः;
    भिन्न freq_scale;
    भिन्न ext_गुणकः;
    भिन्न attn_गुणकः;
    भिन्न beta_fast;
    भिन्न beta_slow;
    भिन्न theta_scale;
    भिन्न theta_scale_32;
    भिन्न theta_powers[32];
    भिन्न corr_dims[2];

    uint32_t src0_nrows_प्रत्येकम्_thread;

    संरचना htp_ops_context * octx;

    uint8_t * vtcm_आधारः;
    size_t    spad_प्रत्येकम्_thread;
    size_t    theta_cache_विस्थापनम्;

    size_t src0_row_आकार;
    size_t src0_row_stride;
    size_t dst_row_आकार;
    size_t dst_row_stride;
    size_t src0_row_आकार_aligned;
    uint32_t src0_nrows;
    uint32_t row_start;
    uint32_t nrows;

    संरचना fastdiv_मानानि div_ne2_ne1;
    संरचना fastdiv_मानानि div_ne1;

    ध्रुव भिन्न * freq_factors;
};

स्थिर भिन्न rope_yarn_ramp(ध्रुव भिन्न नीचः, ध्रुव भिन्न उच्चः, ध्रुव पूर्णाङ्क i0) {
    ध्रुव भिन्न y = (i0 / 2 - नीचः) / MAX(0.001f, उच्चः - नीचः);

    प्रत्यागम (1 - MIN(1, MAX(0, y)));
}

// Compute one (cos, sin) pair into cache[i0], cache[i0+1] applying YaRN scaling.
स्थिर अन्तःस्थ शून्य rope_yarn_one(भिन्न theta, भिन्न freq_scale, भिन्न * corr_dims,
                                 uint32_t i0, भिन्न ext_गुणकः, भिन्न mscale,
                                 भिन्न * cache) {
    भिन्न theta_extrap = theta;

    // Get n-d rotational scaling corrected for extrapolation
    भिन्न theta_interp = freq_scale * theta_extrap;
    भिन्न theta_final  = theta_interp;
    भिन्न mscale_final = mscale;

    यदि (ext_गुणकः != 0.0f) {
        भिन्न ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_गुणकः;
        theta_final    = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale_final  *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    ध्रुव uint32_t b = i0 / 64;
    ध्रुव uint32_t k = (i0 % 64) / 2;
    cache[b * 64 + k]      = cosf(theta_final) * mscale_final;
    cache[b * 64 + 32 + k] = sinf(theta_final) * mscale_final;
}

// 32 thetas -> 32 deinterleaved pairs [cos[32] | sin[32]] at cache[i0].
स्थिर अन्तःस्थ शून्य rope_cache_hvx_32(भिन्न * cache, uint32_t i0,
                                     HVX_सदिशम् v_theta,
                                     ध्रुव भिन्न * freq_factors,
                                     HVX_सदिशम् v_freq_scale,
                                     HVX_सदिशम् v_mscale) {
    यदि (freq_factors) {
        HVX_सदिशम् v_ff = hvx_vmemu(freq_factors + i0 / 2);
        v_theta = hvx_vec_mul_f32_f32(v_theta, hvx_vec_inverse_f32(v_ff));
    }

    HVX_सदिशम् v_theta_final = hvx_vec_mul_f32_f32(v_theta, v_freq_scale);
    HVX_सदिशम् vcos;
    HVX_सदिशम् vsin;
    hvx_vec_sincos_f32(v_theta_final, &vcos, &vsin);
    vcos = hvx_vec_mul_f32_f32(vcos, v_mscale);
    vsin = hvx_vec_mul_f32_f32(vsin, v_mscale);

    यदि (((uintptr_t) (cache + i0)) % 128 == 0) {
        hvx_vmem(cache + i0 + 0)  = vcos;
        hvx_vmem(cache + i0 + 32) = vsin;
    } अन्यथा {
        hvx_vec_store_u(cache + i0 + 0,  32 * परिमाण(भिन्न), vcos);
        hvx_vec_store_u(cache + i0 + 32, 32 * परिमाण(भिन्न), vsin);
    }
}

स्थिर __attribute__((noinline)) शून्य rope_cache_init(ध्रुव भिन्न    theta_आधारः,
                            ध्रुव भिन्न    freq_scale,
                            ध्रुव भिन्न *  freq_factors,
                            भिन्न *        corr_dims,
                            ध्रुव uint32_t n_cache,
                            ध्रुव भिन्न    ext_गुणकः,
                            ध्रुव भिन्न    mscale,
                            भिन्न *        cache,
                            ध्रुव भिन्न    theta_scale,
                            ध्रुव भिन्न *  theta_powers,
                            ध्रुव भिन्न    theta_scale_32) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    यदि (ext_गुणकः == 0.0f) {
        // Fast path: fully vectorized
        // We process 32 pairs (64 elements) per iteration.
        ध्रुव uint32_t n_खण्डानि = n_cache / 64;

        HVX_सदिशम् v_theta_powers = hvx_vmemu(theta_powers);
        HVX_सदिशम् v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_सदिशम् v_mscale = hvx_vec_splat_f32(mscale);

        भिन्न theta_खण्डम् = theta_आधारः;

        प्रति (uint32_t b = 0; b < n_खण्डानि; b++) {
            uint32_t i0 = b * 64;
            HVX_सदिशम् v_theta_आधारः = hvx_vec_splat_f32(theta_खण्डम्);
            HVX_सदिशम् v_theta = hvx_vec_mul_f32_f32(v_theta_आधारः, v_theta_powers);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            theta_खण्डम् *= theta_scale_32;
        }

        // Leftovers
        भिन्न theta = theta_खण्डम्;
        प्रति (uint32_t i0 = n_खण्डानि * 64; i0 < n_cache; i0 += 2) {
            ध्रुव भिन्न ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_गुणकः, mscale, cache);
            theta *= theta_scale;
        }
    } अन्यथा {
        भिन्न theta = theta_आधारः;
        प्रति (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
            ध्रुव भिन्न ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_गुणकः, mscale, cache);
            theta *= theta_scale;
        }
    }
}

स्थिर अन्तःस्थ भिन्न mrope_pick_theta(भिन्न theta_t, भिन्न theta_h, भिन्न theta_w, भिन्न theta_e,
                                     पूर्णाङ्क sector, ध्रुव int32_t sections[4], पूर्णाङ्क sec_w, पूर्णाङ्क sec_e,
                                     bool is_imrope) {
    यदि (is_imrope) {
        यदि      (sector % 3 == 0 && sector < 3 * sections[0]) { प्रत्यागम theta_t; }
        अन्यथा यदि (sector % 3 == 1 && sector < 3 * sections[1]) { प्रत्यागम theta_h; }
        अन्यथा यदि (sector % 3 == 2 && sector < 3 * sections[2]) { प्रत्यागम theta_w; }
        अन्यथा                                                   { प्रत्यागम theta_e; }
    }
    यदि      (sector < sections[0]) { प्रत्यागम theta_t; }
    अन्यथा यदि (sector < sec_w)       { प्रत्यागम theta_h; }
    अन्यथा यदि (sector < sec_e)       { प्रत्यागम theta_w; }
    अन्यथा                           { प्रत्यागम theta_e; }
}

// lane j is 1 when (j % 3) == rem
स्थिर ध्रुव भिन्न __attribute__((aligned(128))) mrope_mod3_eq0[32] = {
    1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0
};
स्थिर ध्रुव भिन्न __attribute__((aligned(128))) mrope_mod3_eq1[32] = {
    0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1
};
स्थिर ध्रुव भिन्न __attribute__((aligned(128))) mrope_mod3_eq2[32] = {
    0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0
};

स्थिर ध्रुव भिन्न __attribute__((aligned(128))) mrope_k_ramp[32] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
};

स्थिर अन्तःस्थ HVX_सदिशम्_Pred mrope_आवरणम्_eq1(ध्रुव भिन्न * m) {
    प्रत्यागम Q6_Q_vcmp_gt_VsfVsf(hvx_vmemu(m), Q6_V_vzero());
}

// IMROPE without wrap: theta[k] = pos[k % 3] * scale^k
स्थिर अन्तःस्थ HVX_सदिशम् mrope_thetas_imrope_mod3(भिन्न pos_t, भिन्न pos_h, भिन्न pos_w,
                                                  uint32_t k0, HVX_सदिशम् v_powers, भिन्न scale_खण्डम्) {
    ध्रुव पूर्णाङ्क r = (पूर्णाङ्क) (k0 % 3);
    ध्रुव भिन्न * mt = (r == 0) ? mrope_mod3_eq0 : (r == 1) ? mrope_mod3_eq2 : mrope_mod3_eq1;
    ध्रुव भिन्न * mh = (r == 0) ? mrope_mod3_eq1 : (r == 1) ? mrope_mod3_eq0 : mrope_mod3_eq2;

    HVX_सदिशम् v = hvx_vec_splat_f32(pos_w);
    v = Q6_V_vmux_QVV(mrope_आवरणम्_eq1(mh), hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(mrope_आवरणम्_eq1(mt), hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    प्रत्यागम hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_खण्डम्));
}

// Contiguous MROPE without wrap: theta[k] = pos[section(k)] * scale^k
स्थिर अन्तःस्थ HVX_सदिशम् mrope_thetas_contig(भिन्न pos_t, भिन्न pos_h, भिन्न pos_w, भिन्न pos_e,
                                             uint32_t k0, पूर्णाङ्क s0, पूर्णाङ्क sec_w, पूर्णाङ्क sec_e,
                                             HVX_सदिशम् v_powers, भिन्न scale_खण्डम्) {
    HVX_सदिशम् v_k = hvx_vec_सङ्कलय_f32_f32(hvx_vec_splat_f32((भिन्न) k0), hvx_vmemu(mrope_k_ramp));
    HVX_सदिशम्_Pred lt_s0 = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((भिन्न) s0), v_k);
    HVX_सदिशम्_Pred lt_sw = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((भिन्न) sec_w), v_k);
    HVX_सदिशम्_Pred lt_se = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((भिन्न) sec_e), v_k);

    HVX_सदिशम् v = hvx_vec_splat_f32(pos_e);
    v = Q6_V_vmux_QVV(lt_se, hvx_vec_splat_f32(pos_w), v);
    v = Q6_V_vmux_QVV(lt_sw, hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(lt_s0, hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    प्रत्यागम hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_खण्डम्));
}

// pos_t/h/w/e: the four position ids for this sequence step (t=time, h=height, w=width, e=extra).
// sections[4]: number of head dims assigned to each position component.
स्थिर __attribute__((noinline)) शून्य mrope_cache_init(ध्रुव भिन्न    pos_t,
                             ध्रुव भिन्न    pos_h,
                             ध्रुव भिन्न    pos_w,
                             ध्रुव भिन्न    pos_e,
                             ध्रुव int32_t  sections[4],
                             ध्रुव bool     is_imrope,
                             ध्रुव bool     indep_sects,
                             ध्रुव भिन्न    freq_scale,
                             ध्रुव भिन्न *  freq_factors,
                             भिन्न *        corr_dims,
                             ध्रुव uint32_t n_cache,
                             ध्रुव भिन्न    ext_गुणकः,
                             ध्रुव भिन्न    mscale,
                             भिन्न *        cache,
                             ध्रुव भिन्न    theta_scale,
                             ध्रुव भिन्न *  theta_powers,
                             ध्रुव भिन्न    theta_scale_32) {
    ध्रुव पूर्णाङ्क sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    ध्रुव पूर्णाङ्क sec_w     = sections[0] + sections[1];
    ध्रुव पूर्णाङ्क sec_e     = sec_w + sections[2];
    ध्रुव uint32_t n_युग्मानि = n_cache / 2;

    ध्रुव bool no_wrap = (sect_dims > 0) && (n_युग्मानि <= (uint32_t) sect_dims);
    ध्रुव bool imrope_mod3 = is_imrope && !indep_sects && no_wrap
        && sections[0] > 0 && sections[1] > 0 && sections[2] > 0
        && n_युग्मानि <= (uint32_t) (3 * sections[0])
        && n_युग्मानि <= (uint32_t) (3 * sections[1])
        && n_युग्मानि <= (uint32_t) (3 * sections[2]);
    ध्रुव bool contig = !is_imrope && !indep_sects && no_wrap;

    यदि (ext_गुणकः == 0.0f && (imrope_mod3 || contig)) {
        HVX_सदिशम् v_powers     = hvx_vmemu(theta_powers);
        HVX_सदिशम् v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_सदिशम् v_mscale     = hvx_vec_splat_f32(mscale);
        भिन्न scale_खण्डम् = 1.0f;
        ध्रुव uint32_t n_खण्डानि = n_cache / 64;

        प्रति (uint32_t b = 0; b < n_खण्डानि; b++) {
            ध्रुव uint32_t i0 = b * 64;
            ध्रुव uint32_t k0 = b * 32;
            HVX_सदिशम् v_theta = imrope_mod3
                ? mrope_thetas_imrope_mod3(pos_t, pos_h, pos_w, k0, v_powers, scale_खण्डम्)
                : mrope_thetas_contig(pos_t, pos_h, pos_w, pos_e, k0, sections[0], sec_w, sec_e,
                                      v_powers, scale_खण्डम्);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            scale_खण्डम् *= theta_scale_32;
        }

        भिन्न theta_k = scale_खण्डम्;
        प्रति (uint32_t k = n_खण्डानि * 32; k < n_युग्मानि; k++) {
            ध्रुव uint32_t i0 = 2 * k;
            ध्रुव भिन्न pos = mrope_pick_theta(pos_t, pos_h, pos_w, pos_e,
                                               (पूर्णाङ्क) k, sections, sec_w, sec_e, is_imrope);
            ध्रुव भिन्न ff = freq_factors ? freq_factors[k] : 1.0f;
            rope_yarn_one(pos * theta_k / ff, freq_scale, corr_dims, i0, ext_गुणकः, mscale, cache);
            theta_k *= theta_scale;
        }
        प्रत्यागम;
    }

    भिन्न theta_t = pos_t;
    भिन्न theta_h = pos_h;
    भिन्न theta_w = pos_w;
    भिन्न theta_e = pos_e;

    ध्रुव bool use_hvx = (ext_गुणकः == 0.0f);
    भिन्न __attribute__((aligned(128))) thetas[32];
    uint32_t n_thetas = 0;
    uint32_t खण्डम्_i0 = 0;

    HVX_सदिशम् v_freq_scale = hvx_vec_splat_f32(freq_scale);
    HVX_सदिशम् v_mscale     = hvx_vec_splat_f32(mscale);

    प्रति (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
        ध्रुव पूर्णाङ्क sector = (i0 / 2) % sect_dims;

        यदि (indep_sects) {
            // Reset theta when crossing into a new section.
            यदि      (sector == 0)           { theta_t = pos_t; }
            अन्यथा यदि (sector == sections[0]) { theta_h = pos_h; }
            अन्यथा यदि (sector == sec_w)       { theta_w = pos_w; }
            अन्यथा यदि (sector == sec_e)       { theta_e = pos_e; }
        }

        ध्रुव भिन्न theta = mrope_pick_theta(theta_t, theta_h, theta_w, theta_e,
                                             sector, sections, sec_w, sec_e, is_imrope);

        यदि (use_hvx) {
            यदि (n_thetas == 0) {
                खण्डम्_i0 = i0;
            }
            thetas[n_thetas++] = theta;
            यदि (n_thetas == 32) {
                rope_cache_hvx_32(cache, खण्डम्_i0, hvx_vmemu(thetas), freq_factors, v_freq_scale, v_mscale);
                n_thetas = 0;
            }
        } अन्यथा {
            ध्रुव भिन्न ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_गुणकः, mscale, cache);
        }

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }

    प्रति (uint32_t k = 0; k < n_thetas; k++) {
        ध्रुव uint32_t i0 = खण्डम्_i0 + 2 * k;
        ध्रुव भिन्न ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn_one(thetas[k] / ff, freq_scale, corr_dims, i0, ext_गुणकः, mscale, cache);
    }
}

#परिभाषय M_PI 3.1415926535897932384626433

स्थिर शून्य rope_corr_dims(पूर्णाङ्क     n_dims,
                           पूर्णाङ्क     n_ctx_orig,
                           भिन्न   freq_आधारः,
                           भिन्न   beta_fast,
                           भिन्न   beta_slow,
                           भिन्न * dims) {
    भिन्न start = floorf(n_dims * logf(n_ctx_orig / (beta_fast * 2 * (भिन्न) M_PI)) / (2 * logf(freq_आधारः)));
    भिन्न अन्तः_2   = ceilf(n_dims * logf(n_ctx_orig / (beta_slow * 2 * (भिन्न) M_PI)) / (2 * logf(freq_आधारः)));
    dims[0]     = MAX(0, start);
    dims[1]     = MIN(n_dims - 1, अन्तः_2);
}

स्थिर अन्तःस्थ शून्य hvx_rope_neox_mul(HVX_सदिशम् v0, HVX_सदिशम् v1, HVX_सदिशम् vcos, HVX_सदिशम् vsin,
                                     HVX_सदिशम् * o0, HVX_सदिशम् * o1) {
    HVX_सदिशम् vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, vcos);
    HVX_सदिशम् vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, vsin);
    HVX_सदिशम् vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, vcos);
    HVX_सदिशम् vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, vsin);
    *o0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s));
    *o1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c));
}

// theta_cache full 32-pair blocks are deinterleaved [cos | sin].
स्थिर अन्तःस्थ शून्य hvx_rope_neox_f32_aa(भिन्न * सीमित dst, ध्रुव भिन्न * सीमित src0, uint32_t ne, ध्रुव भिन्न * सीमित theta_cache) {
    ध्रुव uint32_t he = ne / 2;
    ध्रुव uint32_t nvec = he / 32;
    ध्रुव uint32_t nloe = he % 32;

    यदि (nloe == 0) {
        ध्रुव HVX_सदिशम् * vs = (ध्रुव HVX_सदिशम् *) src0;
        ध्रुव HVX_सदिशम् * vt = (ध्रुव HVX_सदिशम् *) theta_cache;
        HVX_सदिशम् * vd = (HVX_सदिशम् *) dst;
        प्रति (uint32_t i = 0; i < nvec; i++) {
            HVX_सदिशम् o0, o1;
            hvx_rope_neox_mul(vs[i], vs[nvec + i], vt[i * 2 + 0], vt[i * 2 + 1], &o0, &o1);
            vd[i] = o0;
            vd[nvec + i] = o1;
        }
        प्रत्यागम;
    }

    प्रति (uint32_t i = 0; i < nvec; i++) {
        HVX_सदिशम् o0, o1;
        hvx_rope_neox_mul(((ध्रुव HVX_सदिशम् *) src0)[i],
                          hvx_vmemu(src0 + he + i * 32),
                          ((ध्रुव HVX_सदिशम् *) theta_cache)[i * 2 + 0],
                          ((ध्रुव HVX_सदिशम् *) theta_cache)[i * 2 + 1],
                          &o0, &o1);
        ((HVX_सदिशम् *) dst)[i] = o0;
        hvx_vmemu(dst + he + i * 32) = o1;
    }

    HVX_सदिशम् v0 = hvx_vmemu(src0 + nvec * 32);
    HVX_सदिशम् v1 = hvx_vmemu(src0 + he + nvec * 32);
    HVX_सदिशम् vcos = hvx_vmemu(theta_cache + nvec * 64);
    HVX_सदिशम् vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);
    HVX_सदिशम् o0, o1;
    hvx_rope_neox_mul(v0, v1, vcos, vsin, &o0, &o1);
    hvx_vec_store_u(dst + nvec * 32, nloe * परिमाण(भिन्न), o0);
    hvx_vec_store_u(dst + he + nvec * 32, nloe * परिमाण(भिन्न), o1);
}

स्थिर अन्तःस्थ शून्य hvx_rope_f32_aa(भिन्न * सीमित dst, ध्रुव भिन्न * सीमित src0, uint32_t ne, ध्रुव भिन्न * सीमित theta_cache) {
    ध्रुव uint32_t nvec = ne / 64;
    ध्रुव uint32_t nloe = ne % 64;

    प्रति (uint32_t i = 0; i < nvec; i++) {
        HVX_सदिशम् v0 = ((ध्रुव HVX_सदिशम् *) src0)[i * 2 + 0];
        HVX_सदिशम् v1 = ((ध्रुव HVX_सदिशम् *) src0)[i * 2 + 1];

        HVX_सदिशम् vcos = ((ध्रुव HVX_सदिशम् *) theta_cache)[i * 2 + 0];
        HVX_सदिशम् vsin = ((ध्रुव HVX_सदिशम् *) theta_cache)[i * 2 + 1];

        HVX_सदिशम्_Pair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

        HVX_सदिशम् vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
        HVX_सदिशम् vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
        HVX_सदिशम् vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
        HVX_सदिशम् vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

        HVX_सदिशम् v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_सदिशम् v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        HVX_सदिशम्_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

        ((HVX_सदिशम् *) dst)[i * 2 + 0] = Q6_V_lo_W(vstore);
        ((HVX_सदिशम् *) dst)[i * 2 + 1] = Q6_V_hi_W(vstore);
    }

    यदि (nloe > 0) {
        यदि (nloe <= 32) {
            HVX_सदिशम् v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_सदिशम् vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_सदिशम् vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_सदिशम्_Pair vx0_x1 = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, -4);

            HVX_सदिशम् vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_सदिशम् vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_सदिशम् vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_सदिशम् vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_सदिशम् v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_सदिशम् v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_सदिशम्_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            hvx_vec_store_u(dst + nvec * 64, nloe * परिमाण(भिन्न), Q6_V_lo_W(vstore));
        } अन्यथा {
            HVX_सदिशम् v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_सदिशम् v1 = hvx_vmemu(src0 + nvec * 64 + 32);

            HVX_सदिशम् vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_सदिशम् vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_सदिशम्_Pair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

            HVX_सदिशम् vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_सदिशम् vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_सदिशम् vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_सदिशम् vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_सदिशम् v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_सदिशम् v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_सदिशम्_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            ((HVX_सदिशम् *) dst)[nvec * 2 + 0] = Q6_V_lo_W(vstore);
            hvx_vec_store_u(dst + nvec * 64 + 32, (nloe - 32) * परिमाण(भिन्न), Q6_V_hi_W(vstore));
        }
    }
}

स्थिर शून्य अन्तःस्थ rope_basic_f32_inplace(संरचना htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, ध्रुव भिन्न * सीमित theta_cache) {
    ध्रुव uint32_t n_offs = rctx->n_offs;
    #निर्देशः unroll(4)
    प्रति (uint32_t i = 0; i < nr; i++) {
        भिन्न * s = (भिन्न *) (src + i * rctx->src0_row_आकार_aligned);
        hvx_rope_f32_aa(s + n_offs, s + n_offs, rctx->n_dims, theta_cache);
    }
}

स्थिर शून्य अन्तःस्थ rope_neox_f32_inplace(संरचना htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, uint32_t ne, ध्रुव भिन्न * सीमित theta_cache) {
    ध्रुव uint32_t n_offs = rctx->n_offs;
    #निर्देशः unroll(4)
    प्रति (uint32_t i = 0; i < nr; i++) {
        भिन्न * s = (भिन्न *) (src + i * rctx->src0_row_आकार_aligned);
        hvx_rope_neox_f32_aa(s + n_offs, s + n_offs, ne, theta_cache);
    }
}

स्थिर शून्य rope_job_f32(अचिह्नित पूर्णाङ्क nth, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_rope_context * rctx = (संरचना htp_rope_context *) दत्त;
    संरचना htp_ops_context * octx = rctx->octx;

    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];
    ध्रुव संरचना htp_tensor * src2 = octx->src[2];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    htp_rope_preamble;

    ध्रुव uint32_t src0_nrows = rctx->nrows;
    ध्रुव uint32_t src0_nrows_प्रत्येकम्_thread = rctx->src0_nrows_प्रत्येकम्_thread;

    ध्रुव uint32_t src0_start_row = rctx->row_start + src0_nrows_प्रत्येकम्_thread * ith;
    ध्रुव uint32_t src0_अन्तः_row   = MIN(src0_start_row + src0_nrows_प्रत्येकम्_thread, rctx->row_start + src0_nrows);

    // no work for this thread
    यदि (src0_start_row >= src0_अन्तः_row) {
        प्रत्यागम;
    }

    ध्रुव int32_t बहुलकम्    = rctx->बहुलकम्;
    // MROPE, IMROPE and VISION use NEOX-style pairing for the rotation
    ध्रुव bool    is_neox = (बहुलकम् & HTP_ROPE_TYPE_NEOX) || (बहुलकम् & HTP_ROPE_TYPE_MROPE);
    ध्रुव bool    is_vision = (बहुलकम् == HTP_ROPE_TYPE_VISION);

    // VTCM setup
    uint8_t * src0_spad_आधारः = rctx->vtcm_आधारः + (ith * rctx->spad_प्रत्येकम्_thread);
    भिन्न *   theta_cache    = (भिन्न *) (src0_spad_आधारः);
              src0_spad_आधारः = src0_spad_आधारः + rctx->theta_cache_विस्थापनम्;

    dma_queue * dma_q = octx->ctx->dma[ith];
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];
    ध्रुव int32_t * pos = (ध्रुव int32_t *) (uintptr_t) src1->दत्त;
    ध्रुव भिन्न * freq_factors = rctx->freq_factors;

    ध्रुव uint32_t i3_start = fastdiv(src0_start_row, &rctx->div_ne2_ne1);
    ध्रुव uint32_t rem      = fastmodulo(src0_start_row, ne2 * ne1, &rctx->div_ne2_ne1);
    ध्रुव uint32_t i2_start = fastdiv(rem, &rctx->div_ne1);
    ध्रुव uint32_t i1_start = fastmodulo(rem, ne1, &rctx->div_ne1);

    uint32_t ir = src0_start_row;
    uint32_t पूर्व_i2 = (uint32_t) -1;
    uint32_t cur_slot = 0;

    प्रति (uint32_t i3 = i3_start; i3 < ne3; i3++) { // batch
        ध्रुव uint32_t i2_init = (i3 == i3_start) ? i2_start : 0;
        प्रति (uint32_t i2 = i2_init; i2 < ne2; i2++) { // seq-len
            ध्रुव uint32_t i1_init = (i3 == i3_start && i2 == i2_start) ? i1_start : 0;
            प्रति (uint32_t i1 = i1_init; i1 < ne1; ) { // attn-heads
                यदि (ir >= src0_अन्तः_row) गच्छ समाप्तम्;

                // Rows in this block
                ध्रुव uint32_t nrows = MIN(src0_अन्तः_row - ir, ne1 - i1);

                // Depth before prefetch
                ध्रुव uint32_t dma_गभीरता = dma_queue_गभीरता(dma_q);

                // Prefetch up to 2 blocks
                ध्रुव uint32_t p_nrows = MIN(nrows, 2 * HTP_ROPE_SPAD_खण्डम्);
                प्रति (uint32_t pr = 0; pr < p_nrows; pr += HTP_ROPE_SPAD_खण्डम्) {
                    ध्रुव uint32_t pnr = MIN(nrows - pr, HTP_ROPE_SPAD_खण्डम्);
                    ध्रुव uint32_t slot = (cur_slot + pr / HTP_ROPE_SPAD_खण्डम्) % HTP_ROPE_SPAD_NSLOTS;
                    uint8_t * spad_slot = rope_spad_slot(src0_spad_आधारः, slot, rctx->src0_row_आकार_aligned);
                    ध्रुव dma_addr_t src0_दत्त = src0->दत्त + i3 * nb03 + i2 * nb02 + (i1 + pr) * nb01;

                    // Dummy DMA transaction for sequencing (interleaving wr, rd, wr, rd, ...)
                    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst->दत्त, spad_slot), 0, 0, 0, 0);

                    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(spad_slot, src0_दत्त),
                        rctx->src0_row_आकार_aligned, rctx->src0_row_stride, rctx->src0_row_आकार, pnr);
                }

                // Update theta cache
                यदि (i2 != पूर्व_i2) {
                    पूर्व_i2 = i2;

                    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                    // VISION rotates the full row; other modes only rotate n_dims.
                    ध्रुव uint32_t n_cache = is_vision ? ne0 : (uint32_t) rctx->n_dims;
                    ध्रुव bool is_mrope = (rctx->बहुलकम् & HTP_ROPE_TYPE_MROPE) != 0;
                    यदि (is_mrope) {
                        // src1 holds four position arrays stacked along ne0:
                        // pos[i2], pos[i2+ne2], pos[i2+ne2*2], pos[i2+ne2*3]
                        ध्रुव bool is_imrope = (rctx->बहुलकम् == HTP_ROPE_TYPE_IMROPE);
                        mrope_cache_init(
                            (भिन्न) pos[i2],
                            (भिन्न) pos[i2 + ne2],
                            (भिन्न) pos[i2 + ne2 * 2],
                            (भिन्न) pos[i2 + ne2 * 3],
                            rctx->sections, is_imrope, is_vision,
                            rctx->freq_scale, freq_factors, rctx->corr_dims,
                            n_cache, rctx->ext_गुणकः, rctx->attn_गुणकः,
                            theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    } अन्यथा {
                       rope_cache_init(pos[i2], rctx->freq_scale, freq_factors, rctx->corr_dims,
                                        n_cache, rctx->ext_गुणकः, rctx->attn_गुणकः,
                                        theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                }

                // Skip output DMA transactions from prev block (if any)
                प्रति (uint32_t d = 0; d < dma_गभीरता; d++) { dma_queue_निष्कासय_nowait(dma_q); }

                // Compute loop
                ध्रुव uint32_t ne = is_vision ? ne0 : rctx->n_dims;
                ध्रुव uint32_t आधारः_i1 = i1;
                ध्रुव uint32_t आधारः_ir = ir;

                प्रति (uint32_t cnr = 0, cr = 0; cr < nrows; cr += cnr) {
                    cnr = MIN(nrows - cr, HTP_ROPE_SPAD_खण्डम्);
                    ध्रुव uint32_t slot = (cur_slot + cr / HTP_ROPE_SPAD_खण्डम्) % HTP_ROPE_SPAD_NSLOTS;
                    ध्रुव uint32_t cur_ir   = आधारः_ir + cr;
                    ध्रुव uint32_t cur_i1   = आधारः_i1 + cr;

                    dma_queue_निष्कासय(dma_q);
                    uint8_t * cur_spad = (uint8_t *) dma_queue_निष्कासय(dma_q).dst;

                    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);
                    यदि (is_neox || is_vision) {
                        rope_neox_f32_inplace(rctx, cur_spad, cnr, ne, theta_cache);
                    } अन्यथा {
                        rope_basic_f32_inplace(rctx, cur_spad, cnr, theta_cache);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);

                    ध्रुव dma_addr_t dst_दत्त = dst->दत्त + i3 * nb3 + i2 * nb2 + cur_i1 * nb1;
                    dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_दत्त, cur_spad),
                        rctx->dst_row_stride, rctx->src0_row_आकार_aligned, rctx->dst_row_आकार, cnr);

                    // Prefetch 2 blocks ahead into the slot just freed
                    यदि ((cr + 2 * HTP_ROPE_SPAD_खण्डम्) < nrows) {
                        ध्रुव uint32_t p_cr   = cr + 2 * HTP_ROPE_SPAD_खण्डम्;
                        ध्रुव uint32_t pnr    = MIN(nrows - p_cr, HTP_ROPE_SPAD_खण्डम्);
                        ध्रुव uint32_t p_slot = (cur_slot + p_cr / HTP_ROPE_SPAD_खण्डम्) % HTP_ROPE_SPAD_NSLOTS;
                        uint8_t * p_spad      = rope_spad_slot(src0_spad_आधारः, p_slot, rctx->src0_row_आकार_aligned);
                        ध्रुव dma_addr_t p_src0_दत्त = src0->दत्त + i3 * nb03 + i2 * nb02 + (आधारः_i1 + p_cr) * nb01;

                        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(p_spad, p_src0_दत्त),
                            rctx->src0_row_आकार_aligned, rctx->src0_row_stride, rctx->src0_row_आकार, pnr);
                    }
                }

                ध्रुव uint32_t n_chunks = (nrows + HTP_ROPE_SPAD_खण्डम् - 1) / HTP_ROPE_SPAD_खण्डम्;
                cur_slot = (cur_slot + n_chunks) % HTP_ROPE_SPAD_NSLOTS;

                ir += nrows;
                i1 += nrows;
            }
        }
    }

समाप्तम्:
    dma_queue_flush(dma_q);

    FARF(उच्चः_2, "rope-f32: %d/%d: (%u:%u)\n", ith, nth, src0_start_row, src0_अन्तः_row);
}

स्थिर पूर्णाङ्क execute_op_rope_f32(संरचना htp_ops_context * octx) {
    पूर्णाङ्क err = HTP_स्थितिः_OK;

    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * src1 = octx->src[1];
    ध्रुव संरचना htp_tensor * src2 = octx->src[2];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    विकल्प (octx->op) {
        पक्ष HTP_OP_ROPE:
            भङ्ग;

        स्वभाव:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव संरचना htp_rope_kernel_params * kparams = (ध्रुव संरचना htp_rope_kernel_params *) octx->kernel_params;
    यदि (!htp_ops_context_समुच्चयः_n_तन्तवः(octx, kparams->n_तन्तवः)) {
        प्रत्यागम HTP_स्थितिः_INVAL_PARAMS;
    }
    प्रतिज्ञा(octx->ctx->vtcm_आकार >= kparams->vtcm_आकार);

    यदि (htp_tensor_is_extended(src1)) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव uint32_t योग_पङ्क्तिकाः = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव size_t dst_दत्त_row_आकार = dst->ne[0] * परिमाण(भिन्न);

    uint32_t row_start = 0;
    uint32_t nrows     = योग_पङ्क्तिकाः;

    यदि (octx->ctx->mdev.गणनम् > 1) {
        uint32_t पङ्क्तिकाः_प्रत्येकम्_chunk = 0;
        htp_tensor_mdev_पङ्क्तिकाः_प्रत्येकम्_chunk(dst, परिमाण(भिन्न), (uint32_t) dst_दत्त_row_आकार, &पङ्क्तिकाः_प्रत्येकम्_chunk);
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            योग_पङ्क्तिकाः, पङ्क्तिकाः_प्रत्येकम्_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        nrows     = range.गणनम्;
    }

    यदि (nrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    ध्रुव uint32_t n_तन्तवः = octx->n_तन्तवः;

    ध्रुव uint32_t ne0 = dst->ne[0];
    ध्रुव size_t src0_row_आकार   = src0->ne[0] * परिमाण(भिन्न);
    ध्रुव size_t src0_row_stride = src0->nb[1];
    ध्रुव size_t dst_row_आकार    = dst->ne[0] * परिमाण(भिन्न);
    ध्रुव size_t dst_row_stride  = dst->nb[1];

    संरचना htp_rope_context rctx;
    स्मृतिं_पूरय(&rctx, 0, परिमाण(संरचना htp_rope_context));

    rctx.octx                  = octx;
    rctx.vtcm_आधारः             = (uint8_t *) octx->ctx->vtcm_आधारः;
    rctx.spad_प्रत्येकम्_thread       = kparams->spad_प्रत्येकम्_thread;
    rctx.theta_cache_विस्थापनम्    = kparams->theta_cache_विस्थापनम्;

    यदि (src2) {
        dma_queue * dma_q = octx->ctx->dma[0];
        ध्रुव size_t ff_आकार = src2->ne[0] * परिमाण(भिन्न);
        भिन्न * vtcm_freq_factors = (भिन्न *) (rctx.vtcm_आधारः + kparams->freq_factors_विस्थापनम्);
        dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(vtcm_freq_factors, src2->दत्त),
                       kparams->freq_factors_आकार, 0, ff_आकार, 1);
        dma_queue_निष्कासय(dma_q);
        rctx.freq_factors = vtcm_freq_factors;
    }

    ध्रुव int32_t * op_params = &octx->op_params[0];
    rctx.n_dims     = ((ध्रुव int32_t *) op_params)[1];
    rctx.बहुलकम्       = ((ध्रुव int32_t *) op_params)[2];
    rctx.n_ctx_orig = ((ध्रुव int32_t *) op_params)[4];
    rctx.n_offs     = ((ध्रुव int32_t *) op_params)[15];

    स्मृतिं_प्रतिलिख(&rctx.freq_आधारः,   (int32_t *) op_params + 5,  परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&rctx.freq_scale,  (int32_t *) op_params + 6,  परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&rctx.ext_गुणकः,  (int32_t *) op_params + 7,  परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&rctx.attn_गुणकः, (int32_t *) op_params + 8,  परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&rctx.beta_fast,   (int32_t *) op_params + 9,  परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&rctx.beta_slow,   (int32_t *) op_params + 10, परिमाण(भिन्न));
    स्मृतिं_प्रतिलिख(&rctx.sections,    (int32_t *) op_params + 11, परिमाण(पूर्णाङ्क) * 4);

    rctx.theta_scale = powf(rctx.freq_आधारः, -2.0f / rctx.n_dims);
    rctx.theta_powers[0] = 1.0f;
    प्रति (पूर्णाङ्क j = 1; j < 32; j++) {
        rctx.theta_powers[j] = rctx.theta_powers[j - 1] * rctx.theta_scale;
    }
    rctx.theta_scale_32 = rctx.theta_powers[31] * rctx.theta_scale;

    rope_corr_dims(rctx.n_dims, rctx.n_ctx_orig, rctx.freq_आधारः, rctx.beta_fast, rctx.beta_slow, rctx.corr_dims);

    rctx.src0_row_आकार         = src0_row_आकार;
    rctx.src0_row_stride       = src0_row_stride;
    rctx.dst_row_आकार          = dst_row_आकार;
    rctx.dst_row_stride        = dst_row_stride;
    rctx.src0_row_आकार_aligned = kparams->src0_row_आकार_aligned;

    rctx.src0_nrows            = nrows;
    rctx.nrows                 = nrows;
    rctx.row_start             = row_start;
    rctx.src0_nrows_प्रत्येकम्_thread = fastdiv(nrows + n_तन्तवः - 1, &octx->n_तन्तवः_div);
    rctx.div_ne2_ne1           = kparams->div_ne2_ne1;
    rctx.div_ne1               = kparams->div_ne1;

    FARF(उच्चः_2, "rope-f32 n-rows %u n-dims %d ne0 %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n", rctx.src0_nrows, rctx.n_dims, ne0,
         rctx.ext_गुणकः, rctx.theta_scale, rctx.attn_गुणकः);

    work_queue_सततम्(octx->ctx->work_queue, rope_job_f32, &rctx, n_तन्तवः);

    प्रत्यागम err;
}

पूर्णाङ्क op_rope(संरचना htp_ops_context * octx) {
    विकल्प (octx->src[0]->type) {
        पक्ष HTP_TYPE_F32:
            प्रत्यागम execute_op_rope_f32(octx);

        स्वभाव:
            प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }
}
