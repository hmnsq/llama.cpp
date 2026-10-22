#సూచన clang diagnostic ignored "-Wunused-variable"
#సూచన clang diagnostic ignored "-Wunused-function"
#సూచన clang diagnostic ignored "-Wunused-but-set-variable"

#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>

#అన్తర్భావ <math.h>
#అన్తర్భావ <string.h>
#అన్తర్భావ <stdlib.h>

#అన్తర్భావ "dma-queue.h"
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "hex-fastdiv.h"

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "rope-ops.h"

// Redefined the rope type constants as we can't include ggml.h
#నిర్వచించు HTP_ROPE_TYPE_సాధారణ 0
#నిర్వచించు HTP_ROPE_TYPE_NEOX   2
#నిర్వచించు HTP_ROPE_TYPE_MROPE  8
#నిర్వచించు HTP_ROPE_TYPE_VISION 24
#నిర్వచించు HTP_ROPE_TYPE_IMROPE 40

#నిర్వచించు htp_rope_preamble              \
    స్థిరరాశి uint32_t ne00 = src0->ne[0]; \
    స్థిరరాశి uint32_t ne01 = src0->ne[1]; \
    స్థిరరాశి uint32_t ne02 = src0->ne[2]; \
    స్థిరరాశి uint32_t ne03 = src0->ne[3]; \
                                       \
    స్థిరరాశి uint32_t ne0 = dst->ne[0];   \
    స్థిరరాశి uint32_t ne1 = dst->ne[1];   \
    స్థిరరాశి uint32_t ne2 = dst->ne[2];   \
    స్థిరరాశి uint32_t ne3 = dst->ne[3];   \
                                       \
    స్థిరరాశి uint32_t nb00 = src0->nb[0]; \
    స్థిరరాశి uint32_t nb01 = src0->nb[1]; \
    స్థిరరాశి uint32_t nb02 = src0->nb[2]; \
    స్థిరరాశి uint32_t nb03 = src0->nb[3]; \
                                       \
    స్థిరరాశి uint32_t nb0 = dst->nb[0];   \
    స్థిరరాశి uint32_t nb1 = dst->nb[1];   \
    స్థిరరాశి uint32_t nb2 = dst->nb[2];   \
    స్థిరరాశి uint32_t nb3 = dst->nb[3];

నిర్మాణం htp_rope_context {
    int32_t n_dims;
    int32_t n_offs;
    int32_t బహుళకం;
    int32_t n_ctx_orig;
    int32_t sections[4];

    దశాంశం freq_పునాది;
    దశాంశం freq_scale;
    దశాంశం ext_గుణకం;
    దశాంశం attn_గుణకం;
    దశాంశం beta_fast;
    దశాంశం beta_slow;
    దశాంశం theta_scale;
    దశాంశం theta_scale_32;
    దశాంశం theta_powers[32];
    దశాంశం corr_dims[2];

    uint32_t src0_nrows_చొప్పున_thread;

    నిర్మాణం htp_ops_context * octx;

    uint8_t * vtcm_పునాది;
    size_t    spad_చొప్పున_thread;
    size_t    theta_cache_స్థానభ్రంశం;

    size_t src0_అడ్డువరుస_కొలత;
    size_t src0_అడ్డువరుస_stride;
    size_t dst_అడ్డువరుస_కొలత;
    size_t dst_అడ్డువరుస_stride;
    size_t src0_అడ్డువరుస_కొలత_aligned;
    uint32_t src0_nrows;
    uint32_t అడ్డువరుస_ప్రారంభం;
    uint32_t nrows;

    నిర్మాణం fastdiv_విలువలు div_ne2_ne1;
    నిర్మాణం fastdiv_విలువలు div_ne1;

    స్థిరరాశి దశాంశం * freq_factors;
};

స్థిర దశాంశం rope_yarn_ramp(స్థిరరాశి దశాంశం దిగువ, స్థిరరాశి దశాంశం ఎగువ, స్థిరరాశి పూర్ణాంకం i0) {
    స్థిరరాశి దశాంశం y = (i0 / 2 - దిగువ) / MAX(0.001f, ఎగువ - దిగువ);

    తిరిగి (1 - MIN(1, MAX(0, y)));
}

// Compute one (cos, sin) pair into cache[i0], cache[i0+1] applying YaRN scaling.
స్థిర అంతఃవరుస శూన్య rope_yarn_one(దశాంశం theta, దశాంశం freq_scale, దశాంశం * corr_dims,
                                 uint32_t i0, దశాంశం ext_గుణకం, దశాంశం mscale,
                                 దశాంశం * cache) {
    దశాంశం theta_extrap = theta;

    // Get n-d rotational scaling corrected for extrapolation
    దశాంశం theta_interp = freq_scale * theta_extrap;
    దశాంశం theta_final  = theta_interp;
    దశాంశం mscale_final = mscale;

    ఒకవేళ (ext_గుణకం != 0.0f) {
        దశాంశం ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_గుణకం;
        theta_final    = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale_final  *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    స్థిరరాశి uint32_t b = i0 / 64;
    స్థిరరాశి uint32_t k = (i0 % 64) / 2;
    cache[b * 64 + k]      = cosf(theta_final) * mscale_final;
    cache[b * 64 + 32 + k] = sinf(theta_final) * mscale_final;
}

// 32 thetas -> 32 deinterleaved pairs [cos[32] | sin[32]] at cache[i0].
స్థిర అంతఃవరుస శూన్య rope_cache_hvx_32(దశాంశం * cache, uint32_t i0,
                                     HVX_సదిశ v_theta,
                                     స్థిరరాశి దశాంశం * freq_factors,
                                     HVX_సదిశ v_freq_scale,
                                     HVX_సదిశ v_mscale) {
    ఒకవేళ (freq_factors) {
        HVX_సదిశ v_ff = hvx_vmemu(freq_factors + i0 / 2);
        v_theta = hvx_vec_mul_f32_f32(v_theta, hvx_vec_inverse_f32(v_ff));
    }

    HVX_సదిశ v_theta_final = hvx_vec_mul_f32_f32(v_theta, v_freq_scale);
    HVX_సదిశ vcos;
    HVX_సదిశ vsin;
    hvx_vec_sincos_f32(v_theta_final, &vcos, &vsin);
    vcos = hvx_vec_mul_f32_f32(vcos, v_mscale);
    vsin = hvx_vec_mul_f32_f32(vsin, v_mscale);

    ఒకవేళ (((uintptr_t) (cache + i0)) % 128 == 0) {
        hvx_vmem(cache + i0 + 0)  = vcos;
        hvx_vmem(cache + i0 + 32) = vsin;
    } లేకపోతే {
        hvx_vec_store_u(cache + i0 + 0,  32 * పరిమాణం(దశాంశం), vcos);
        hvx_vec_store_u(cache + i0 + 32, 32 * పరిమాణం(దశాంశం), vsin);
    }
}

స్థిర __attribute__((noinline)) శూన్య rope_cache_init(స్థిరరాశి దశాంశం    theta_పునాది,
                            స్థిరరాశి దశాంశం    freq_scale,
                            స్థిరరాశి దశాంశం *  freq_factors,
                            దశాంశం *        corr_dims,
                            స్థిరరాశి uint32_t n_cache,
                            స్థిరరాశి దశాంశం    ext_గుణకం,
                            స్థిరరాశి దశాంశం    mscale,
                            దశాంశం *        cache,
                            స్థిరరాశి దశాంశం    theta_scale,
                            స్థిరరాశి దశాంశం *  theta_powers,
                            స్థిరరాశి దశాంశం    theta_scale_32) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    ఒకవేళ (ext_గుణకం == 0.0f) {
        // Fast path: fully vectorized
        // We process 32 pairs (64 elements) per iteration.
        స్థిరరాశి uint32_t n_ఖండాలు = n_cache / 64;

        HVX_సదిశ v_theta_powers = hvx_vmemu(theta_powers);
        HVX_సదిశ v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_సదిశ v_mscale = hvx_vec_splat_f32(mscale);

        దశాంశం theta_ఖండం = theta_పునాది;

        కొరకు (uint32_t b = 0; b < n_ఖండాలు; b++) {
            uint32_t i0 = b * 64;
            HVX_సదిశ v_theta_పునాది = hvx_vec_splat_f32(theta_ఖండం);
            HVX_సదిశ v_theta = hvx_vec_mul_f32_f32(v_theta_పునాది, v_theta_powers);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            theta_ఖండం *= theta_scale_32;
        }

        // Leftovers
        దశాంశం theta = theta_ఖండం;
        కొరకు (uint32_t i0 = n_ఖండాలు * 64; i0 < n_cache; i0 += 2) {
            స్థిరరాశి దశాంశం ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_గుణకం, mscale, cache);
            theta *= theta_scale;
        }
    } లేకపోతే {
        దశాంశం theta = theta_పునాది;
        కొరకు (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
            స్థిరరాశి దశాంశం ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_గుణకం, mscale, cache);
            theta *= theta_scale;
        }
    }
}

స్థిర అంతఃవరుస దశాంశం mrope_pick_theta(దశాంశం theta_t, దశాంశం theta_h, దశాంశం theta_w, దశాంశం theta_e,
                                     పూర్ణాంకం sector, స్థిరరాశి int32_t sections[4], పూర్ణాంకం sec_w, పూర్ణాంకం sec_e,
                                     bool is_imrope) {
    ఒకవేళ (is_imrope) {
        ఒకవేళ      (sector % 3 == 0 && sector < 3 * sections[0]) { తిరిగి theta_t; }
        లేకపోతే ఒకవేళ (sector % 3 == 1 && sector < 3 * sections[1]) { తిరిగి theta_h; }
        లేకపోతే ఒకవేళ (sector % 3 == 2 && sector < 3 * sections[2]) { తిరిగి theta_w; }
        లేకపోతే                                                   { తిరిగి theta_e; }
    }
    ఒకవేళ      (sector < sections[0]) { తిరిగి theta_t; }
    లేకపోతే ఒకవేళ (sector < sec_w)       { తిరిగి theta_h; }
    లేకపోతే ఒకవేళ (sector < sec_e)       { తిరిగి theta_w; }
    లేకపోతే                           { తిరిగి theta_e; }
}

// lane j is 1 when (j % 3) == rem
స్థిర స్థిరరాశి దశాంశం __attribute__((aligned(128))) mrope_mod3_eq0[32] = {
    1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0
};
స్థిర స్థిరరాశి దశాంశం __attribute__((aligned(128))) mrope_mod3_eq1[32] = {
    0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1
};
స్థిర స్థిరరాశి దశాంశం __attribute__((aligned(128))) mrope_mod3_eq2[32] = {
    0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0
};

స్థిర స్థిరరాశి దశాంశం __attribute__((aligned(128))) mrope_k_ramp[32] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
};

స్థిర అంతఃవరుస HVX_సదిశ_Pred mrope_ముసుగు_eq1(స్థిరరాశి దశాంశం * m) {
    తిరిగి Q6_Q_vcmp_gt_VsfVsf(hvx_vmemu(m), Q6_V_vzero());
}

// IMROPE without wrap: theta[k] = pos[k % 3] * scale^k
స్థిర అంతఃవరుస HVX_సదిశ mrope_thetas_imrope_mod3(దశాంశం pos_t, దశాంశం pos_h, దశాంశం pos_w,
                                                  uint32_t k0, HVX_సదిశ v_powers, దశాంశం scale_ఖండం) {
    స్థిరరాశి పూర్ణాంకం r = (పూర్ణాంకం) (k0 % 3);
    స్థిరరాశి దశాంశం * mt = (r == 0) ? mrope_mod3_eq0 : (r == 1) ? mrope_mod3_eq2 : mrope_mod3_eq1;
    స్థిరరాశి దశాంశం * mh = (r == 0) ? mrope_mod3_eq1 : (r == 1) ? mrope_mod3_eq0 : mrope_mod3_eq2;

    HVX_సదిశ v = hvx_vec_splat_f32(pos_w);
    v = Q6_V_vmux_QVV(mrope_ముసుగు_eq1(mh), hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(mrope_ముసుగు_eq1(mt), hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    తిరిగి hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_ఖండం));
}

// Contiguous MROPE without wrap: theta[k] = pos[section(k)] * scale^k
స్థిర అంతఃవరుస HVX_సదిశ mrope_thetas_contig(దశాంశం pos_t, దశాంశం pos_h, దశాంశం pos_w, దశాంశం pos_e,
                                             uint32_t k0, పూర్ణాంకం s0, పూర్ణాంకం sec_w, పూర్ణాంకం sec_e,
                                             HVX_సదిశ v_powers, దశాంశం scale_ఖండం) {
    HVX_సదిశ v_k = hvx_vec_కూడు_f32_f32(hvx_vec_splat_f32((దశాంశం) k0), hvx_vmemu(mrope_k_ramp));
    HVX_సదిశ_Pred lt_s0 = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((దశాంశం) s0), v_k);
    HVX_సదిశ_Pred lt_sw = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((దశాంశం) sec_w), v_k);
    HVX_సదిశ_Pred lt_se = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((దశాంశం) sec_e), v_k);

    HVX_సదిశ v = hvx_vec_splat_f32(pos_e);
    v = Q6_V_vmux_QVV(lt_se, hvx_vec_splat_f32(pos_w), v);
    v = Q6_V_vmux_QVV(lt_sw, hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(lt_s0, hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    తిరిగి hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_ఖండం));
}

// pos_t/h/w/e: the four position ids for this sequence step (t=time, h=height, w=width, e=extra).
// sections[4]: number of head dims assigned to each position component.
స్థిర __attribute__((noinline)) శూన్య mrope_cache_init(స్థిరరాశి దశాంశం    pos_t,
                             స్థిరరాశి దశాంశం    pos_h,
                             స్థిరరాశి దశాంశం    pos_w,
                             స్థిరరాశి దశాంశం    pos_e,
                             స్థిరరాశి int32_t  sections[4],
                             స్థిరరాశి bool     is_imrope,
                             స్థిరరాశి bool     indep_sects,
                             స్థిరరాశి దశాంశం    freq_scale,
                             స్థిరరాశి దశాంశం *  freq_factors,
                             దశాంశం *        corr_dims,
                             స్థిరరాశి uint32_t n_cache,
                             స్థిరరాశి దశాంశం    ext_గుణకం,
                             స్థిరరాశి దశాంశం    mscale,
                             దశాంశం *        cache,
                             స్థిరరాశి దశాంశం    theta_scale,
                             స్థిరరాశి దశాంశం *  theta_powers,
                             స్థిరరాశి దశాంశం    theta_scale_32) {
    స్థిరరాశి పూర్ణాంకం sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    స్థిరరాశి పూర్ణాంకం sec_w     = sections[0] + sections[1];
    స్థిరరాశి పూర్ణాంకం sec_e     = sec_w + sections[2];
    స్థిరరాశి uint32_t n_జతలు = n_cache / 2;

    స్థిరరాశి bool no_wrap = (sect_dims > 0) && (n_జతలు <= (uint32_t) sect_dims);
    స్థిరరాశి bool imrope_mod3 = is_imrope && !indep_sects && no_wrap
        && sections[0] > 0 && sections[1] > 0 && sections[2] > 0
        && n_జతలు <= (uint32_t) (3 * sections[0])
        && n_జతలు <= (uint32_t) (3 * sections[1])
        && n_జతలు <= (uint32_t) (3 * sections[2]);
    స్థిరరాశి bool contig = !is_imrope && !indep_sects && no_wrap;

    ఒకవేళ (ext_గుణకం == 0.0f && (imrope_mod3 || contig)) {
        HVX_సదిశ v_powers     = hvx_vmemu(theta_powers);
        HVX_సదిశ v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_సదిశ v_mscale     = hvx_vec_splat_f32(mscale);
        దశాంశం scale_ఖండం = 1.0f;
        స్థిరరాశి uint32_t n_ఖండాలు = n_cache / 64;

        కొరకు (uint32_t b = 0; b < n_ఖండాలు; b++) {
            స్థిరరాశి uint32_t i0 = b * 64;
            స్థిరరాశి uint32_t k0 = b * 32;
            HVX_సదిశ v_theta = imrope_mod3
                ? mrope_thetas_imrope_mod3(pos_t, pos_h, pos_w, k0, v_powers, scale_ఖండం)
                : mrope_thetas_contig(pos_t, pos_h, pos_w, pos_e, k0, sections[0], sec_w, sec_e,
                                      v_powers, scale_ఖండం);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            scale_ఖండం *= theta_scale_32;
        }

        దశాంశం theta_k = scale_ఖండం;
        కొరకు (uint32_t k = n_ఖండాలు * 32; k < n_జతలు; k++) {
            స్థిరరాశి uint32_t i0 = 2 * k;
            స్థిరరాశి దశాంశం pos = mrope_pick_theta(pos_t, pos_h, pos_w, pos_e,
                                               (పూర్ణాంకం) k, sections, sec_w, sec_e, is_imrope);
            స్థిరరాశి దశాంశం ff = freq_factors ? freq_factors[k] : 1.0f;
            rope_yarn_one(pos * theta_k / ff, freq_scale, corr_dims, i0, ext_గుణకం, mscale, cache);
            theta_k *= theta_scale;
        }
        తిరిగి;
    }

    దశాంశం theta_t = pos_t;
    దశాంశం theta_h = pos_h;
    దశాంశం theta_w = pos_w;
    దశాంశం theta_e = pos_e;

    స్థిరరాశి bool use_hvx = (ext_గుణకం == 0.0f);
    దశాంశం __attribute__((aligned(128))) thetas[32];
    uint32_t n_thetas = 0;
    uint32_t ఖండం_i0 = 0;

    HVX_సదిశ v_freq_scale = hvx_vec_splat_f32(freq_scale);
    HVX_సదిశ v_mscale     = hvx_vec_splat_f32(mscale);

    కొరకు (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
        స్థిరరాశి పూర్ణాంకం sector = (i0 / 2) % sect_dims;

        ఒకవేళ (indep_sects) {
            // Reset theta when crossing into a new section.
            ఒకవేళ      (sector == 0)           { theta_t = pos_t; }
            లేకపోతే ఒకవేళ (sector == sections[0]) { theta_h = pos_h; }
            లేకపోతే ఒకవేళ (sector == sec_w)       { theta_w = pos_w; }
            లేకపోతే ఒకవేళ (sector == sec_e)       { theta_e = pos_e; }
        }

        స్థిరరాశి దశాంశం theta = mrope_pick_theta(theta_t, theta_h, theta_w, theta_e,
                                             sector, sections, sec_w, sec_e, is_imrope);

        ఒకవేళ (use_hvx) {
            ఒకవేళ (n_thetas == 0) {
                ఖండం_i0 = i0;
            }
            thetas[n_thetas++] = theta;
            ఒకవేళ (n_thetas == 32) {
                rope_cache_hvx_32(cache, ఖండం_i0, hvx_vmemu(thetas), freq_factors, v_freq_scale, v_mscale);
                n_thetas = 0;
            }
        } లేకపోతే {
            స్థిరరాశి దశాంశం ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_గుణకం, mscale, cache);
        }

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }

    కొరకు (uint32_t k = 0; k < n_thetas; k++) {
        స్థిరరాశి uint32_t i0 = ఖండం_i0 + 2 * k;
        స్థిరరాశి దశాంశం ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn_one(thetas[k] / ff, freq_scale, corr_dims, i0, ext_గుణకం, mscale, cache);
    }
}

#నిర్వచించు M_PI 3.1415926535897932384626433

స్థిర శూన్య rope_corr_dims(పూర్ణాంకం     n_dims,
                           పూర్ణాంకం     n_ctx_orig,
                           దశాంశం   freq_పునాది,
                           దశాంశం   beta_fast,
                           దశాంశం   beta_slow,
                           దశాంశం * dims) {
    దశాంశం ప్రారంభం = floorf(n_dims * logf(n_ctx_orig / (beta_fast * 2 * (దశాంశం) M_PI)) / (2 * logf(freq_పునాది)));
    దశాంశం చివర   = ceilf(n_dims * logf(n_ctx_orig / (beta_slow * 2 * (దశాంశం) M_PI)) / (2 * logf(freq_పునాది)));
    dims[0]     = MAX(0, ప్రారంభం);
    dims[1]     = MIN(n_dims - 1, చివర);
}

స్థిర అంతఃవరుస శూన్య hvx_rope_neox_mul(HVX_సదిశ v0, HVX_సదిశ v1, HVX_సదిశ vcos, HVX_సదిశ vsin,
                                     HVX_సదిశ * o0, HVX_సదిశ * o1) {
    HVX_సదిశ vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, vcos);
    HVX_సదిశ vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, vsin);
    HVX_సదిశ vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, vcos);
    HVX_సదిశ vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, vsin);
    *o0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s));
    *o1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c));
}

// theta_cache full 32-pair blocks are deinterleaved [cos | sin].
స్థిర అంతఃవరుస శూన్య hvx_rope_neox_f32_aa(దశాంశం * పరిమిత dst, స్థిరరాశి దశాంశం * పరిమిత src0, uint32_t ne, స్థిరరాశి దశాంశం * పరిమిత theta_cache) {
    స్థిరరాశి uint32_t he = ne / 2;
    స్థిరరాశి uint32_t nvec = he / 32;
    స్థిరరాశి uint32_t nloe = he % 32;

    ఒకవేళ (nloe == 0) {
        స్థిరరాశి HVX_సదిశ * vs = (స్థిరరాశి HVX_సదిశ *) src0;
        స్థిరరాశి HVX_సదిశ * vt = (స్థిరరాశి HVX_సదిశ *) theta_cache;
        HVX_సదిశ * vd = (HVX_సదిశ *) dst;
        కొరకు (uint32_t i = 0; i < nvec; i++) {
            HVX_సదిశ o0, o1;
            hvx_rope_neox_mul(vs[i], vs[nvec + i], vt[i * 2 + 0], vt[i * 2 + 1], &o0, &o1);
            vd[i] = o0;
            vd[nvec + i] = o1;
        }
        తిరిగి;
    }

    కొరకు (uint32_t i = 0; i < nvec; i++) {
        HVX_సదిశ o0, o1;
        hvx_rope_neox_mul(((స్థిరరాశి HVX_సదిశ *) src0)[i],
                          hvx_vmemu(src0 + he + i * 32),
                          ((స్థిరరాశి HVX_సదిశ *) theta_cache)[i * 2 + 0],
                          ((స్థిరరాశి HVX_సదిశ *) theta_cache)[i * 2 + 1],
                          &o0, &o1);
        ((HVX_సదిశ *) dst)[i] = o0;
        hvx_vmemu(dst + he + i * 32) = o1;
    }

    HVX_సదిశ v0 = hvx_vmemu(src0 + nvec * 32);
    HVX_సదిశ v1 = hvx_vmemu(src0 + he + nvec * 32);
    HVX_సదిశ vcos = hvx_vmemu(theta_cache + nvec * 64);
    HVX_సదిశ vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);
    HVX_సదిశ o0, o1;
    hvx_rope_neox_mul(v0, v1, vcos, vsin, &o0, &o1);
    hvx_vec_store_u(dst + nvec * 32, nloe * పరిమాణం(దశాంశం), o0);
    hvx_vec_store_u(dst + he + nvec * 32, nloe * పరిమాణం(దశాంశం), o1);
}

స్థిర అంతఃవరుస శూన్య hvx_rope_f32_aa(దశాంశం * పరిమిత dst, స్థిరరాశి దశాంశం * పరిమిత src0, uint32_t ne, స్థిరరాశి దశాంశం * పరిమిత theta_cache) {
    స్థిరరాశి uint32_t nvec = ne / 64;
    స్థిరరాశి uint32_t nloe = ne % 64;

    కొరకు (uint32_t i = 0; i < nvec; i++) {
        HVX_సదిశ v0 = ((స్థిరరాశి HVX_సదిశ *) src0)[i * 2 + 0];
        HVX_సదిశ v1 = ((స్థిరరాశి HVX_సదిశ *) src0)[i * 2 + 1];

        HVX_సదిశ vcos = ((స్థిరరాశి HVX_సదిశ *) theta_cache)[i * 2 + 0];
        HVX_సదిశ vsin = ((స్థిరరాశి HVX_సదిశ *) theta_cache)[i * 2 + 1];

        HVX_సదిశ_Pair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

        HVX_సదిశ vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
        HVX_సదిశ vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
        HVX_సదిశ vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
        HVX_సదిశ vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

        HVX_సదిశ v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_సదిశ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        HVX_సదిశ_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

        ((HVX_సదిశ *) dst)[i * 2 + 0] = Q6_V_lo_W(vstore);
        ((HVX_సదిశ *) dst)[i * 2 + 1] = Q6_V_hi_W(vstore);
    }

    ఒకవేళ (nloe > 0) {
        ఒకవేళ (nloe <= 32) {
            HVX_సదిశ v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_సదిశ vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_సదిశ vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_సదిశ_Pair vx0_x1 = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, -4);

            HVX_సదిశ vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_సదిశ vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_సదిశ vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_సదిశ vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_సదిశ v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_సదిశ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_సదిశ_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            hvx_vec_store_u(dst + nvec * 64, nloe * పరిమాణం(దశాంశం), Q6_V_lo_W(vstore));
        } లేకపోతే {
            HVX_సదిశ v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_సదిశ v1 = hvx_vmemu(src0 + nvec * 64 + 32);

            HVX_సదిశ vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_సదిశ vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_సదిశ_Pair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

            HVX_సదిశ vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_సదిశ vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_సదిశ vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_సదిశ vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_సదిశ v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_సదిశ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_సదిశ_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            ((HVX_సదిశ *) dst)[nvec * 2 + 0] = Q6_V_lo_W(vstore);
            hvx_vec_store_u(dst + nvec * 64 + 32, (nloe - 32) * పరిమాణం(దశాంశం), Q6_V_hi_W(vstore));
        }
    }
}

స్థిర శూన్య అంతఃవరుస rope_basic_f32_inplace(నిర్మాణం htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, స్థిరరాశి దశాంశం * పరిమిత theta_cache) {
    స్థిరరాశి uint32_t n_offs = rctx->n_offs;
    #సూచన unroll(4)
    కొరకు (uint32_t i = 0; i < nr; i++) {
        దశాంశం * s = (దశాంశం *) (src + i * rctx->src0_అడ్డువరుస_కొలత_aligned);
        hvx_rope_f32_aa(s + n_offs, s + n_offs, rctx->n_dims, theta_cache);
    }
}

స్థిర శూన్య అంతఃవరుస rope_neox_f32_inplace(నిర్మాణం htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, uint32_t ne, స్థిరరాశి దశాంశం * పరిమిత theta_cache) {
    స్థిరరాశి uint32_t n_offs = rctx->n_offs;
    #సూచన unroll(4)
    కొరకు (uint32_t i = 0; i < nr; i++) {
        దశాంశం * s = (దశాంశం *) (src + i * rctx->src0_అడ్డువరుస_కొలత_aligned);
        hvx_rope_neox_f32_aa(s + n_offs, s + n_offs, ne, theta_cache);
    }
}

స్థిర శూన్య rope_పని_f32(గుర్తులేని పూర్ణాంకం nth, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_rope_context * rctx = (నిర్మాణం htp_rope_context *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = rctx->octx;

    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * src2 = octx->src[2];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    htp_rope_preamble;

    స్థిరరాశి uint32_t src0_nrows = rctx->nrows;
    స్థిరరాశి uint32_t src0_nrows_చొప్పున_thread = rctx->src0_nrows_చొప్పున_thread;

    స్థిరరాశి uint32_t src0_ప్రారంభం_అడ్డువరుస = rctx->అడ్డువరుస_ప్రారంభం + src0_nrows_చొప్పున_thread * ith;
    స్థిరరాశి uint32_t src0_చివర_అడ్డువరుస   = MIN(src0_ప్రారంభం_అడ్డువరుస + src0_nrows_చొప్పున_thread, rctx->అడ్డువరుస_ప్రారంభం + src0_nrows);

    // no work for this thread
    ఒకవేళ (src0_ప్రారంభం_అడ్డువరుస >= src0_చివర_అడ్డువరుస) {
        తిరిగి;
    }

    స్థిరరాశి int32_t బహుళకం    = rctx->బహుళకం;
    // MROPE, IMROPE and VISION use NEOX-style pairing for the rotation
    స్థిరరాశి bool    is_neox = (బహుళకం & HTP_ROPE_TYPE_NEOX) || (బహుళకం & HTP_ROPE_TYPE_MROPE);
    స్థిరరాశి bool    is_vision = (బహుళకం == HTP_ROPE_TYPE_VISION);

    // VTCM setup
    uint8_t * src0_spad_పునాది = rctx->vtcm_పునాది + (ith * rctx->spad_చొప్పున_thread);
    దశాంశం *   theta_cache    = (దశాంశం *) (src0_spad_పునాది);
              src0_spad_పునాది = src0_spad_పునాది + rctx->theta_cache_స్థానభ్రంశం;

    dma_వరుస * dma_q = octx->ctx->dma[ith];
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];
    స్థిరరాశి int32_t * pos = (స్థిరరాశి int32_t *) (uintptr_t) src1->దత్తాంశం;
    స్థిరరాశి దశాంశం * freq_factors = rctx->freq_factors;

    స్థిరరాశి uint32_t i3_ప్రారంభం = fastdiv(src0_ప్రారంభం_అడ్డువరుస, &rctx->div_ne2_ne1);
    స్థిరరాశి uint32_t rem      = fastmodulo(src0_ప్రారంభం_అడ్డువరుస, ne2 * ne1, &rctx->div_ne2_ne1);
    స్థిరరాశి uint32_t i2_ప్రారంభం = fastdiv(rem, &rctx->div_ne1);
    స్థిరరాశి uint32_t i1_ప్రారంభం = fastmodulo(rem, ne1, &rctx->div_ne1);

    uint32_t ir = src0_ప్రారంభం_అడ్డువరుస;
    uint32_t prev_i2 = (uint32_t) -1;
    uint32_t cur_స్థానం = 0;

    కొరకు (uint32_t i3 = i3_ప్రారంభం; i3 < ne3; i3++) { // batch
        స్థిరరాశి uint32_t i2_init = (i3 == i3_ప్రారంభం) ? i2_ప్రారంభం : 0;
        కొరకు (uint32_t i2 = i2_init; i2 < ne2; i2++) { // seq-len
            స్థిరరాశి uint32_t i1_init = (i3 == i3_ప్రారంభం && i2 == i2_ప్రారంభం) ? i1_ప్రారంభం : 0;
            కొరకు (uint32_t i1 = i1_init; i1 < ne1; ) { // attn-heads
                ఒకవేళ (ir >= src0_చివర_అడ్డువరుస) వెళ్ళు పూర్తి;

                // Rows in this block
                స్థిరరాశి uint32_t nrows = MIN(src0_చివర_అడ్డువరుస - ir, ne1 - i1);

                // Depth before prefetch
                స్థిరరాశి uint32_t dma_లోతు = dma_వరుస_లోతు(dma_q);

                // Prefetch up to 2 blocks
                స్థిరరాశి uint32_t p_nrows = MIN(nrows, 2 * HTP_ROPE_SPAD_ఖండం);
                కొరకు (uint32_t pr = 0; pr < p_nrows; pr += HTP_ROPE_SPAD_ఖండం) {
                    స్థిరరాశి uint32_t pnr = MIN(nrows - pr, HTP_ROPE_SPAD_ఖండం);
                    స్థిరరాశి uint32_t స్థానం = (cur_స్థానం + pr / HTP_ROPE_SPAD_ఖండం) % HTP_ROPE_SPAD_NSLOTS;
                    uint8_t * spad_స్థానం = rope_spad_స్థానం(src0_spad_పునాది, స్థానం, rctx->src0_అడ్డువరుస_కొలత_aligned);
                    స్థిరరాశి dma_addr_t src0_దత్తాంశం = src0->దత్తాంశం + i3 * nb03 + i2 * nb02 + (i1 + pr) * nb01;

                    // Dummy DMA transaction for sequencing (interleaving wr, rd, wr, rd, ...)
                    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst->దత్తాంశం, spad_స్థానం), 0, 0, 0, 0);

                    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(spad_స్థానం, src0_దత్తాంశం),
                        rctx->src0_అడ్డువరుస_కొలత_aligned, rctx->src0_అడ్డువరుస_stride, rctx->src0_అడ్డువరుస_కొలత, pnr);
                }

                // Update theta cache
                ఒకవేళ (i2 != prev_i2) {
                    prev_i2 = i2;

                    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                    // VISION rotates the full row; other modes only rotate n_dims.
                    స్థిరరాశి uint32_t n_cache = is_vision ? ne0 : (uint32_t) rctx->n_dims;
                    స్థిరరాశి bool is_mrope = (rctx->బహుళకం & HTP_ROPE_TYPE_MROPE) != 0;
                    ఒకవేళ (is_mrope) {
                        // src1 holds four position arrays stacked along ne0:
                        // pos[i2], pos[i2+ne2], pos[i2+ne2*2], pos[i2+ne2*3]
                        స్థిరరాశి bool is_imrope = (rctx->బహుళకం == HTP_ROPE_TYPE_IMROPE);
                        mrope_cache_init(
                            (దశాంశం) pos[i2],
                            (దశాంశం) pos[i2 + ne2],
                            (దశాంశం) pos[i2 + ne2 * 2],
                            (దశాంశం) pos[i2 + ne2 * 3],
                            rctx->sections, is_imrope, is_vision,
                            rctx->freq_scale, freq_factors, rctx->corr_dims,
                            n_cache, rctx->ext_గుణకం, rctx->attn_గుణకం,
                            theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    } లేకపోతే {
                       rope_cache_init(pos[i2], rctx->freq_scale, freq_factors, rctx->corr_dims,
                                        n_cache, rctx->ext_గుణకం, rctx->attn_గుణకం,
                                        theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                }

                // Skip output DMA transactions from prev block (if any)
                కొరకు (uint32_t d = 0; d < dma_లోతు; d++) { dma_వరుస_తీయి_nowait(dma_q); }

                // Compute loop
                స్థిరరాశి uint32_t ne = is_vision ? ne0 : rctx->n_dims;
                స్థిరరాశి uint32_t పునాది_i1 = i1;
                స్థిరరాశి uint32_t పునాది_ir = ir;

                కొరకు (uint32_t cnr = 0, cr = 0; cr < nrows; cr += cnr) {
                    cnr = MIN(nrows - cr, HTP_ROPE_SPAD_ఖండం);
                    స్థిరరాశి uint32_t స్థానం = (cur_స్థానం + cr / HTP_ROPE_SPAD_ఖండం) % HTP_ROPE_SPAD_NSLOTS;
                    స్థిరరాశి uint32_t cur_ir   = పునాది_ir + cr;
                    స్థిరరాశి uint32_t cur_i1   = పునాది_i1 + cr;

                    dma_వరుస_తీయి(dma_q);
                    uint8_t * cur_spad = (uint8_t *) dma_వరుస_తీయి(dma_q).dst;

                    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);
                    ఒకవేళ (is_neox || is_vision) {
                        rope_neox_f32_inplace(rctx, cur_spad, cnr, ne, theta_cache);
                    } లేకపోతే {
                        rope_basic_f32_inplace(rctx, cur_spad, cnr, theta_cache);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);

                    స్థిరరాశి dma_addr_t dst_దత్తాంశం = dst->దత్తాంశం + i3 * nb3 + i2 * nb2 + cur_i1 * nb1;
                    dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_దత్తాంశం, cur_spad),
                        rctx->dst_అడ్డువరుస_stride, rctx->src0_అడ్డువరుస_కొలత_aligned, rctx->dst_అడ్డువరుస_కొలత, cnr);

                    // Prefetch 2 blocks ahead into the slot just freed
                    ఒకవేళ ((cr + 2 * HTP_ROPE_SPAD_ఖండం) < nrows) {
                        స్థిరరాశి uint32_t p_cr   = cr + 2 * HTP_ROPE_SPAD_ఖండం;
                        స్థిరరాశి uint32_t pnr    = MIN(nrows - p_cr, HTP_ROPE_SPAD_ఖండం);
                        స్థిరరాశి uint32_t p_స్థానం = (cur_స్థానం + p_cr / HTP_ROPE_SPAD_ఖండం) % HTP_ROPE_SPAD_NSLOTS;
                        uint8_t * p_spad      = rope_spad_స్థానం(src0_spad_పునాది, p_స్థానం, rctx->src0_అడ్డువరుస_కొలత_aligned);
                        స్థిరరాశి dma_addr_t p_src0_దత్తాంశం = src0->దత్తాంశం + i3 * nb03 + i2 * nb02 + (పునాది_i1 + p_cr) * nb01;

                        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(p_spad, p_src0_దత్తాంశం),
                            rctx->src0_అడ్డువరుస_కొలత_aligned, rctx->src0_అడ్డువరుస_stride, rctx->src0_అడ్డువరుస_కొలత, pnr);
                    }
                }

                స్థిరరాశి uint32_t n_chunks = (nrows + HTP_ROPE_SPAD_ఖండం - 1) / HTP_ROPE_SPAD_ఖండం;
                cur_స్థానం = (cur_స్థానం + n_chunks) % HTP_ROPE_SPAD_NSLOTS;

                ir += nrows;
                i1 += nrows;
            }
        }
    }

పూర్తి:
    dma_వరుస_flush(dma_q);

    FARF(ఎగువ_2, "rope-f32: %d/%d: (%u:%u)\n", ith, nth, src0_ప్రారంభం_అడ్డువరుస, src0_చివర_అడ్డువరుస);
}

స్థిర పూర్ణాంకం execute_op_rope_f32(నిర్మాణం htp_ops_context * octx) {
    పూర్ణాంకం err = HTP_స్థితి_OK;

    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * src1 = octx->src[1];
    స్థిరరాశి నిర్మాణం htp_tensor * src2 = octx->src[2];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    ఎంపిక (octx->op) {
        స్థితి HTP_OP_ROPE:
            ఆపు;

        అప్రమేయ:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి నిర్మాణం htp_rope_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_rope_kernel_params *) octx->kernel_params;
    ఒకవేళ (!htp_ops_context_సమితి_n_దారాలు(octx, kparams->n_దారాలు)) {
        తిరిగి HTP_స్థితి_INVAL_PARAMS;
    }
    ప్రతిజ్ఞా(octx->ctx->vtcm_కొలత >= kparams->vtcm_కొలత);

    ఒకవేళ (htp_tensor_is_extended(src1)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి size_t dst_దత్తాంశం_అడ్డువరుస_కొలత = dst->ne[0] * పరిమాణం(దశాంశం);

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t nrows     = మొత్తం_అడ్డువరుసలు;

    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        uint32_t అడ్డువరుసలు_చొప్పున_chunk = 0;
        htp_tensor_mdev_అడ్డువరుసలు_చొప్పున_chunk(dst, పరిమాణం(దశాంశం), (uint32_t) dst_దత్తాంశం_అడ్డువరుస_కొలత, &అడ్డువరుసలు_చొప్పున_chunk);
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(
            మొత్తం_అడ్డువరుసలు, అడ్డువరుసలు_చొప్పున_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        nrows     = range.లెక్క;
    }

    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    స్థిరరాశి uint32_t n_దారాలు = octx->n_దారాలు;

    స్థిరరాశి uint32_t ne0 = dst->ne[0];
    స్థిరరాశి size_t src0_అడ్డువరుస_కొలత   = src0->ne[0] * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t src0_అడ్డువరుస_stride = src0->nb[1];
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత    = dst->ne[0] * పరిమాణం(దశాంశం);
    స్థిరరాశి size_t dst_అడ్డువరుస_stride  = dst->nb[1];

    నిర్మాణం htp_rope_context rctx;
    memset(&rctx, 0, పరిమాణం(నిర్మాణం htp_rope_context));

    rctx.octx                  = octx;
    rctx.vtcm_పునాది             = (uint8_t *) octx->ctx->vtcm_పునాది;
    rctx.spad_చొప్పున_thread       = kparams->spad_చొప్పున_thread;
    rctx.theta_cache_స్థానభ్రంశం    = kparams->theta_cache_స్థానభ్రంశం;

    ఒకవేళ (src2) {
        dma_వరుస * dma_q = octx->ctx->dma[0];
        స్థిరరాశి size_t ff_కొలత = src2->ne[0] * పరిమాణం(దశాంశం);
        దశాంశం * vtcm_freq_factors = (దశాంశం *) (rctx.vtcm_పునాది + kparams->freq_factors_స్థానభ్రంశం);
        dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(vtcm_freq_factors, src2->దత్తాంశం),
                       kparams->freq_factors_కొలత, 0, ff_కొలత, 1);
        dma_వరుస_తీయి(dma_q);
        rctx.freq_factors = vtcm_freq_factors;
    }

    స్థిరరాశి int32_t * op_params = &octx->op_params[0];
    rctx.n_dims     = ((స్థిరరాశి int32_t *) op_params)[1];
    rctx.బహుళకం       = ((స్థిరరాశి int32_t *) op_params)[2];
    rctx.n_ctx_orig = ((స్థిరరాశి int32_t *) op_params)[4];
    rctx.n_offs     = ((స్థిరరాశి int32_t *) op_params)[15];

    జ్ఞాపకం_నకలు(&rctx.freq_పునాది,   (int32_t *) op_params + 5,  పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&rctx.freq_scale,  (int32_t *) op_params + 6,  పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&rctx.ext_గుణకం,  (int32_t *) op_params + 7,  పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&rctx.attn_గుణకం, (int32_t *) op_params + 8,  పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&rctx.beta_fast,   (int32_t *) op_params + 9,  పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&rctx.beta_slow,   (int32_t *) op_params + 10, పరిమాణం(దశాంశం));
    జ్ఞాపకం_నకలు(&rctx.sections,    (int32_t *) op_params + 11, పరిమాణం(పూర్ణాంకం) * 4);

    rctx.theta_scale = powf(rctx.freq_పునాది, -2.0f / rctx.n_dims);
    rctx.theta_powers[0] = 1.0f;
    కొరకు (పూర్ణాంకం j = 1; j < 32; j++) {
        rctx.theta_powers[j] = rctx.theta_powers[j - 1] * rctx.theta_scale;
    }
    rctx.theta_scale_32 = rctx.theta_powers[31] * rctx.theta_scale;

    rope_corr_dims(rctx.n_dims, rctx.n_ctx_orig, rctx.freq_పునాది, rctx.beta_fast, rctx.beta_slow, rctx.corr_dims);

    rctx.src0_అడ్డువరుస_కొలత         = src0_అడ్డువరుస_కొలత;
    rctx.src0_అడ్డువరుస_stride       = src0_అడ్డువరుస_stride;
    rctx.dst_అడ్డువరుస_కొలత          = dst_అడ్డువరుస_కొలత;
    rctx.dst_అడ్డువరుస_stride        = dst_అడ్డువరుస_stride;
    rctx.src0_అడ్డువరుస_కొలత_aligned = kparams->src0_అడ్డువరుస_కొలత_aligned;

    rctx.src0_nrows            = nrows;
    rctx.nrows                 = nrows;
    rctx.అడ్డువరుస_ప్రారంభం             = అడ్డువరుస_ప్రారంభం;
    rctx.src0_nrows_చొప్పున_thread = fastdiv(nrows + n_దారాలు - 1, &octx->n_దారాలు_div);
    rctx.div_ne2_ne1           = kparams->div_ne2_ne1;
    rctx.div_ne1               = kparams->div_ne1;

    FARF(ఎగువ_2, "rope-f32 n-rows %u n-dims %d ne0 %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n", rctx.src0_nrows, rctx.n_dims, ne0,
         rctx.ext_గుణకం, rctx.theta_scale, rctx.attn_గుణకం);

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, rope_పని_f32, &rctx, n_దారాలు);

    తిరిగి err;
}

పూర్ణాంకం op_rope(నిర్మాణం htp_ops_context * octx) {
    ఎంపిక (octx->src[0]->type) {
        స్థితి HTP_TYPE_F32:
            తిరిగి execute_op_rope_f32(octx);

        అప్రమేయ:
            తిరిగి HTP_స్థితి_NO_SUPPORT;
    }
}
