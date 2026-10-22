#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-variable"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-function"
#ਹਦਾਇਤ clang diagnostic ignored "-Wunused-but-set-variable"

#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>

#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>

#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "rope-ops.h"

// Redefined the rope type constants as we can't include ggml.h
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_ROPE_TYPE_ਸਾਧਾਰਨ 0
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_ROPE_TYPE_NEOX   2
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_ROPE_TYPE_MROPE  8
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_ROPE_TYPE_VISION 24
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HTP_ROPE_TYPE_IMROPE 40

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ htp_rope_preamble              \
    ਅਚਲ uint32_t ne00 = src0->ne[0]; \
    ਅਚਲ uint32_t ne01 = src0->ne[1]; \
    ਅਚਲ uint32_t ne02 = src0->ne[2]; \
    ਅਚਲ uint32_t ne03 = src0->ne[3]; \
                                       \
    ਅਚਲ uint32_t ne0 = dst->ne[0];   \
    ਅਚਲ uint32_t ne1 = dst->ne[1];   \
    ਅਚਲ uint32_t ne2 = dst->ne[2];   \
    ਅਚਲ uint32_t ne3 = dst->ne[3];   \
                                       \
    ਅਚਲ uint32_t nb00 = src0->nb[0]; \
    ਅਚਲ uint32_t nb01 = src0->nb[1]; \
    ਅਚਲ uint32_t nb02 = src0->nb[2]; \
    ਅਚਲ uint32_t nb03 = src0->nb[3]; \
                                       \
    ਅਚਲ uint32_t nb0 = dst->nb[0];   \
    ਅਚਲ uint32_t nb1 = dst->nb[1];   \
    ਅਚਲ uint32_t nb2 = dst->nb[2];   \
    ਅਚਲ uint32_t nb3 = dst->nb[3];

ਬਣਤਰ htp_rope_context {
    int32_t n_dims;
    int32_t n_offs;
    int32_t ਬਹੁਲਕ;
    int32_t n_ctx_orig;
    int32_t sections[4];

    ਦਸ਼ਮਲਵ freq_ਅਧਾਰ;
    ਦਸ਼ਮਲਵ freq_scale;
    ਦਸ਼ਮਲਵ ext_ਗੁਣਕ;
    ਦਸ਼ਮਲਵ attn_ਗੁਣਕ;
    ਦਸ਼ਮਲਵ beta_fast;
    ਦਸ਼ਮਲਵ beta_slow;
    ਦਸ਼ਮਲਵ theta_scale;
    ਦਸ਼ਮਲਵ theta_scale_32;
    ਦਸ਼ਮਲਵ theta_powers[32];
    ਦਸ਼ਮਲਵ corr_dims[2];

    uint32_t src0_nrows_ਪ੍ਰਤੀ_thread;

    ਬਣਤਰ htp_ops_context * octx;

    uint8_t * vtcm_ਅਧਾਰ;
    size_t    spad_ਪ੍ਰਤੀ_thread;
    size_t    theta_cache_ਵਿੱਥ;

    size_t src0_ਪੰਕਤੀ_ਮਾਪ;
    size_t src0_ਪੰਕਤੀ_stride;
    size_t dst_ਪੰਕਤੀ_ਮਾਪ;
    size_t dst_ਪੰਕਤੀ_stride;
    size_t src0_ਪੰਕਤੀ_ਮਾਪ_aligned;
    uint32_t src0_nrows;
    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    uint32_t nrows;

    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne2_ne1;
    ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ div_ne1;

    ਅਚਲ ਦਸ਼ਮਲਵ * freq_factors;
};

ਸਥਿਰ ਦਸ਼ਮਲਵ rope_yarn_ramp(ਅਚਲ ਦਸ਼ਮਲਵ ਨੀਵਾਂ, ਅਚਲ ਦਸ਼ਮਲਵ ਉੱਚਾ, ਅਚਲ ਪੂਰਨ i0) {
    ਅਚਲ ਦਸ਼ਮਲਵ y = (i0 / 2 - ਨੀਵਾਂ) / MAX(0.001f, ਉੱਚਾ - ਨੀਵਾਂ);

    ਵਾਪਸ (1 - MIN(1, MAX(0, y)));
}

// Compute one (cos, sin) pair into cache[i0], cache[i0+1] applying YaRN scaling.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rope_yarn_one(ਦਸ਼ਮਲਵ theta, ਦਸ਼ਮਲਵ freq_scale, ਦਸ਼ਮਲਵ * corr_dims,
                                 uint32_t i0, ਦਸ਼ਮਲਵ ext_ਗੁਣਕ, ਦਸ਼ਮਲਵ mscale,
                                 ਦਸ਼ਮਲਵ * cache) {
    ਦਸ਼ਮਲਵ theta_extrap = theta;

    // Get n-d rotational scaling corrected for extrapolation
    ਦਸ਼ਮਲਵ theta_interp = freq_scale * theta_extrap;
    ਦਸ਼ਮਲਵ theta_final  = theta_interp;
    ਦਸ਼ਮਲਵ mscale_final = mscale;

    ਜੇ (ext_ਗੁਣਕ != 0.0f) {
        ਦਸ਼ਮਲਵ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_ਗੁਣਕ;
        theta_final    = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale_final  *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    ਅਚਲ uint32_t b = i0 / 64;
    ਅਚਲ uint32_t k = (i0 % 64) / 2;
    cache[b * 64 + k]      = cosf(theta_final) * mscale_final;
    cache[b * 64 + 32 + k] = sinf(theta_final) * mscale_final;
}

// 32 thetas -> 32 deinterleaved pairs [cos[32] | sin[32]] at cache[i0].
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ rope_cache_hvx_32(ਦਸ਼ਮਲਵ * cache, uint32_t i0,
                                     HVX_ਸਦਿਸ਼ v_theta,
                                     ਅਚਲ ਦਸ਼ਮਲਵ * freq_factors,
                                     HVX_ਸਦਿਸ਼ v_freq_scale,
                                     HVX_ਸਦਿਸ਼ v_mscale) {
    ਜੇ (freq_factors) {
        HVX_ਸਦਿਸ਼ v_ff = hvx_vmemu(freq_factors + i0 / 2);
        v_theta = hvx_vec_mul_f32_f32(v_theta, hvx_vec_inverse_f32(v_ff));
    }

    HVX_ਸਦਿਸ਼ v_theta_final = hvx_vec_mul_f32_f32(v_theta, v_freq_scale);
    HVX_ਸਦਿਸ਼ vcos;
    HVX_ਸਦਿਸ਼ vsin;
    hvx_vec_sincos_f32(v_theta_final, &vcos, &vsin);
    vcos = hvx_vec_mul_f32_f32(vcos, v_mscale);
    vsin = hvx_vec_mul_f32_f32(vsin, v_mscale);

    ਜੇ (((uintptr_t) (cache + i0)) % 128 == 0) {
        hvx_vmem(cache + i0 + 0)  = vcos;
        hvx_vmem(cache + i0 + 32) = vsin;
    } ਨਹੀਂਤਾਂ {
        hvx_vec_store_u(cache + i0 + 0,  32 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), vcos);
        hvx_vec_store_u(cache + i0 + 32, 32 * ਆਕਾਰ(ਦਸ਼ਮਲਵ), vsin);
    }
}

ਸਥਿਰ __attribute__((noinline)) ਖਾਲੀ rope_cache_init(ਅਚਲ ਦਸ਼ਮਲਵ    theta_ਅਧਾਰ,
                            ਅਚਲ ਦਸ਼ਮਲਵ    freq_scale,
                            ਅਚਲ ਦਸ਼ਮਲਵ *  freq_factors,
                            ਦਸ਼ਮਲਵ *        corr_dims,
                            ਅਚਲ uint32_t n_cache,
                            ਅਚਲ ਦਸ਼ਮਲਵ    ext_ਗੁਣਕ,
                            ਅਚਲ ਦਸ਼ਮਲਵ    mscale,
                            ਦਸ਼ਮਲਵ *        cache,
                            ਅਚਲ ਦਸ਼ਮਲਵ    theta_scale,
                            ਅਚਲ ਦਸ਼ਮਲਵ *  theta_powers,
                            ਅਚਲ ਦਸ਼ਮਲਵ    theta_scale_32) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    ਜੇ (ext_ਗੁਣਕ == 0.0f) {
        // Fast path: fully vectorized
        // We process 32 pairs (64 elements) per iteration.
        ਅਚਲ uint32_t n_ਖੰਡਾਂ = n_cache / 64;

        HVX_ਸਦਿਸ਼ v_theta_powers = hvx_vmemu(theta_powers);
        HVX_ਸਦਿਸ਼ v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_ਸਦਿਸ਼ v_mscale = hvx_vec_splat_f32(mscale);

        ਦਸ਼ਮਲਵ theta_ਖੰਡ = theta_ਅਧਾਰ;

        ਲਈ (uint32_t b = 0; b < n_ਖੰਡਾਂ; b++) {
            uint32_t i0 = b * 64;
            HVX_ਸਦਿਸ਼ v_theta_ਅਧਾਰ = hvx_vec_splat_f32(theta_ਖੰਡ);
            HVX_ਸਦਿਸ਼ v_theta = hvx_vec_mul_f32_f32(v_theta_ਅਧਾਰ, v_theta_powers);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            theta_ਖੰਡ *= theta_scale_32;
        }

        // Leftovers
        ਦਸ਼ਮਲਵ theta = theta_ਖੰਡ;
        ਲਈ (uint32_t i0 = n_ਖੰਡਾਂ * 64; i0 < n_cache; i0 += 2) {
            ਅਚਲ ਦਸ਼ਮਲਵ ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ਗੁਣਕ, mscale, cache);
            theta *= theta_scale;
        }
    } ਨਹੀਂਤਾਂ {
        ਦਸ਼ਮਲਵ theta = theta_ਅਧਾਰ;
        ਲਈ (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
            ਅਚਲ ਦਸ਼ਮਲਵ ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ਗੁਣਕ, mscale, cache);
            theta *= theta_scale;
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਦਸ਼ਮਲਵ mrope_pick_theta(ਦਸ਼ਮਲਵ theta_t, ਦਸ਼ਮਲਵ theta_h, ਦਸ਼ਮਲਵ theta_w, ਦਸ਼ਮਲਵ theta_e,
                                     ਪੂਰਨ sector, ਅਚਲ int32_t sections[4], ਪੂਰਨ sec_w, ਪੂਰਨ sec_e,
                                     bool is_imrope) {
    ਜੇ (is_imrope) {
        ਜੇ      (sector % 3 == 0 && sector < 3 * sections[0]) { ਵਾਪਸ theta_t; }
        ਨਹੀਂਤਾਂ ਜੇ (sector % 3 == 1 && sector < 3 * sections[1]) { ਵਾਪਸ theta_h; }
        ਨਹੀਂਤਾਂ ਜੇ (sector % 3 == 2 && sector < 3 * sections[2]) { ਵਾਪਸ theta_w; }
        ਨਹੀਂਤਾਂ                                                   { ਵਾਪਸ theta_e; }
    }
    ਜੇ      (sector < sections[0]) { ਵਾਪਸ theta_t; }
    ਨਹੀਂਤਾਂ ਜੇ (sector < sec_w)       { ਵਾਪਸ theta_h; }
    ਨਹੀਂਤਾਂ ਜੇ (sector < sec_e)       { ਵਾਪਸ theta_w; }
    ਨਹੀਂਤਾਂ                           { ਵਾਪਸ theta_e; }
}

// lane j is 1 when (j % 3) == rem
ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ __attribute__((aligned(128))) mrope_mod3_eq0[32] = {
    1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0
};
ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ __attribute__((aligned(128))) mrope_mod3_eq1[32] = {
    0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1
};
ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ __attribute__((aligned(128))) mrope_mod3_eq2[32] = {
    0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0
};

ਸਥਿਰ ਅਚਲ ਦਸ਼ਮਲਵ __attribute__((aligned(128))) mrope_k_ramp[32] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
};

ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼_Pred mrope_ਨਕਾਬ_eq1(ਅਚਲ ਦਸ਼ਮਲਵ * m) {
    ਵਾਪਸ Q6_Q_vcmp_gt_VsfVsf(hvx_vmemu(m), Q6_V_vzero());
}

// IMROPE without wrap: theta[k] = pos[k % 3] * scale^k
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ mrope_thetas_imrope_mod3(ਦਸ਼ਮਲਵ pos_t, ਦਸ਼ਮਲਵ pos_h, ਦਸ਼ਮਲਵ pos_w,
                                                  uint32_t k0, HVX_ਸਦਿਸ਼ v_powers, ਦਸ਼ਮਲਵ scale_ਖੰਡ) {
    ਅਚਲ ਪੂਰਨ r = (ਪੂਰਨ) (k0 % 3);
    ਅਚਲ ਦਸ਼ਮਲਵ * mt = (r == 0) ? mrope_mod3_eq0 : (r == 1) ? mrope_mod3_eq2 : mrope_mod3_eq1;
    ਅਚਲ ਦਸ਼ਮਲਵ * mh = (r == 0) ? mrope_mod3_eq1 : (r == 1) ? mrope_mod3_eq0 : mrope_mod3_eq2;

    HVX_ਸਦਿਸ਼ v = hvx_vec_splat_f32(pos_w);
    v = Q6_V_vmux_QVV(mrope_ਨਕਾਬ_eq1(mh), hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(mrope_ਨਕਾਬ_eq1(mt), hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    ਵਾਪਸ hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_ਖੰਡ));
}

// Contiguous MROPE without wrap: theta[k] = pos[section(k)] * scale^k
ਸਥਿਰ ਸਤਰਅੰਦਰ HVX_ਸਦਿਸ਼ mrope_thetas_contig(ਦਸ਼ਮਲਵ pos_t, ਦਸ਼ਮਲਵ pos_h, ਦਸ਼ਮਲਵ pos_w, ਦਸ਼ਮਲਵ pos_e,
                                             uint32_t k0, ਪੂਰਨ s0, ਪੂਰਨ sec_w, ਪੂਰਨ sec_e,
                                             HVX_ਸਦਿਸ਼ v_powers, ਦਸ਼ਮਲਵ scale_ਖੰਡ) {
    HVX_ਸਦਿਸ਼ v_k = hvx_vec_ਜੋੜੋ_f32_f32(hvx_vec_splat_f32((ਦਸ਼ਮਲਵ) k0), hvx_vmemu(mrope_k_ramp));
    HVX_ਸਦਿਸ਼_Pred lt_s0 = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((ਦਸ਼ਮਲਵ) s0), v_k);
    HVX_ਸਦਿਸ਼_Pred lt_sw = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((ਦਸ਼ਮਲਵ) sec_w), v_k);
    HVX_ਸਦਿਸ਼_Pred lt_se = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((ਦਸ਼ਮਲਵ) sec_e), v_k);

    HVX_ਸਦਿਸ਼ v = hvx_vec_splat_f32(pos_e);
    v = Q6_V_vmux_QVV(lt_se, hvx_vec_splat_f32(pos_w), v);
    v = Q6_V_vmux_QVV(lt_sw, hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(lt_s0, hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    ਵਾਪਸ hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_ਖੰਡ));
}

// pos_t/h/w/e: the four position ids for this sequence step (t=time, h=height, w=width, e=extra).
// sections[4]: number of head dims assigned to each position component.
ਸਥਿਰ __attribute__((noinline)) ਖਾਲੀ mrope_cache_init(ਅਚਲ ਦਸ਼ਮਲਵ    pos_t,
                             ਅਚਲ ਦਸ਼ਮਲਵ    pos_h,
                             ਅਚਲ ਦਸ਼ਮਲਵ    pos_w,
                             ਅਚਲ ਦਸ਼ਮਲਵ    pos_e,
                             ਅਚਲ int32_t  sections[4],
                             ਅਚਲ bool     is_imrope,
                             ਅਚਲ bool     indep_sects,
                             ਅਚਲ ਦਸ਼ਮਲਵ    freq_scale,
                             ਅਚਲ ਦਸ਼ਮਲਵ *  freq_factors,
                             ਦਸ਼ਮਲਵ *        corr_dims,
                             ਅਚਲ uint32_t n_cache,
                             ਅਚਲ ਦਸ਼ਮਲਵ    ext_ਗੁਣਕ,
                             ਅਚਲ ਦਸ਼ਮਲਵ    mscale,
                             ਦਸ਼ਮਲਵ *        cache,
                             ਅਚਲ ਦਸ਼ਮਲਵ    theta_scale,
                             ਅਚਲ ਦਸ਼ਮਲਵ *  theta_powers,
                             ਅਚਲ ਦਸ਼ਮਲਵ    theta_scale_32) {
    ਅਚਲ ਪੂਰਨ sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    ਅਚਲ ਪੂਰਨ sec_w     = sections[0] + sections[1];
    ਅਚਲ ਪੂਰਨ sec_e     = sec_w + sections[2];
    ਅਚਲ uint32_t n_ਜੋੜੇ = n_cache / 2;

    ਅਚਲ bool no_wrap = (sect_dims > 0) && (n_ਜੋੜੇ <= (uint32_t) sect_dims);
    ਅਚਲ bool imrope_mod3 = is_imrope && !indep_sects && no_wrap
        && sections[0] > 0 && sections[1] > 0 && sections[2] > 0
        && n_ਜੋੜੇ <= (uint32_t) (3 * sections[0])
        && n_ਜੋੜੇ <= (uint32_t) (3 * sections[1])
        && n_ਜੋੜੇ <= (uint32_t) (3 * sections[2]);
    ਅਚਲ bool contig = !is_imrope && !indep_sects && no_wrap;

    ਜੇ (ext_ਗੁਣਕ == 0.0f && (imrope_mod3 || contig)) {
        HVX_ਸਦਿਸ਼ v_powers     = hvx_vmemu(theta_powers);
        HVX_ਸਦਿਸ਼ v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_ਸਦਿਸ਼ v_mscale     = hvx_vec_splat_f32(mscale);
        ਦਸ਼ਮਲਵ scale_ਖੰਡ = 1.0f;
        ਅਚਲ uint32_t n_ਖੰਡਾਂ = n_cache / 64;

        ਲਈ (uint32_t b = 0; b < n_ਖੰਡਾਂ; b++) {
            ਅਚਲ uint32_t i0 = b * 64;
            ਅਚਲ uint32_t k0 = b * 32;
            HVX_ਸਦਿਸ਼ v_theta = imrope_mod3
                ? mrope_thetas_imrope_mod3(pos_t, pos_h, pos_w, k0, v_powers, scale_ਖੰਡ)
                : mrope_thetas_contig(pos_t, pos_h, pos_w, pos_e, k0, sections[0], sec_w, sec_e,
                                      v_powers, scale_ਖੰਡ);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            scale_ਖੰਡ *= theta_scale_32;
        }

        ਦਸ਼ਮਲਵ theta_k = scale_ਖੰਡ;
        ਲਈ (uint32_t k = n_ਖੰਡਾਂ * 32; k < n_ਜੋੜੇ; k++) {
            ਅਚਲ uint32_t i0 = 2 * k;
            ਅਚਲ ਦਸ਼ਮਲਵ pos = mrope_pick_theta(pos_t, pos_h, pos_w, pos_e,
                                               (ਪੂਰਨ) k, sections, sec_w, sec_e, is_imrope);
            ਅਚਲ ਦਸ਼ਮਲਵ ff = freq_factors ? freq_factors[k] : 1.0f;
            rope_yarn_one(pos * theta_k / ff, freq_scale, corr_dims, i0, ext_ਗੁਣਕ, mscale, cache);
            theta_k *= theta_scale;
        }
        ਵਾਪਸ;
    }

    ਦਸ਼ਮਲਵ theta_t = pos_t;
    ਦਸ਼ਮਲਵ theta_h = pos_h;
    ਦਸ਼ਮਲਵ theta_w = pos_w;
    ਦਸ਼ਮਲਵ theta_e = pos_e;

    ਅਚਲ bool use_hvx = (ext_ਗੁਣਕ == 0.0f);
    ਦਸ਼ਮਲਵ __attribute__((aligned(128))) thetas[32];
    uint32_t n_thetas = 0;
    uint32_t ਖੰਡ_i0 = 0;

    HVX_ਸਦਿਸ਼ v_freq_scale = hvx_vec_splat_f32(freq_scale);
    HVX_ਸਦਿਸ਼ v_mscale     = hvx_vec_splat_f32(mscale);

    ਲਈ (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
        ਅਚਲ ਪੂਰਨ sector = (i0 / 2) % sect_dims;

        ਜੇ (indep_sects) {
            // Reset theta when crossing into a new section.
            ਜੇ      (sector == 0)           { theta_t = pos_t; }
            ਨਹੀਂਤਾਂ ਜੇ (sector == sections[0]) { theta_h = pos_h; }
            ਨਹੀਂਤਾਂ ਜੇ (sector == sec_w)       { theta_w = pos_w; }
            ਨਹੀਂਤਾਂ ਜੇ (sector == sec_e)       { theta_e = pos_e; }
        }

        ਅਚਲ ਦਸ਼ਮਲਵ theta = mrope_pick_theta(theta_t, theta_h, theta_w, theta_e,
                                             sector, sections, sec_w, sec_e, is_imrope);

        ਜੇ (use_hvx) {
            ਜੇ (n_thetas == 0) {
                ਖੰਡ_i0 = i0;
            }
            thetas[n_thetas++] = theta;
            ਜੇ (n_thetas == 32) {
                rope_cache_hvx_32(cache, ਖੰਡ_i0, hvx_vmemu(thetas), freq_factors, v_freq_scale, v_mscale);
                n_thetas = 0;
            }
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ ਦਸ਼ਮਲਵ ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ਗੁਣਕ, mscale, cache);
        }

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }

    ਲਈ (uint32_t k = 0; k < n_thetas; k++) {
        ਅਚਲ uint32_t i0 = ਖੰਡ_i0 + 2 * k;
        ਅਚਲ ਦਸ਼ਮਲਵ ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn_one(thetas[k] / ff, freq_scale, corr_dims, i0, ext_ਗੁਣਕ, mscale, cache);
    }
}

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ M_PI 3.1415926535897932384626433

ਸਥਿਰ ਖਾਲੀ rope_corr_dims(ਪੂਰਨ     n_dims,
                           ਪੂਰਨ     n_ctx_orig,
                           ਦਸ਼ਮਲਵ   freq_ਅਧਾਰ,
                           ਦਸ਼ਮਲਵ   beta_fast,
                           ਦਸ਼ਮਲਵ   beta_slow,
                           ਦਸ਼ਮਲਵ * dims) {
    ਦਸ਼ਮਲਵ ਸ਼ੁਰੂ = floorf(n_dims * logf(n_ctx_orig / (beta_fast * 2 * (ਦਸ਼ਮਲਵ) M_PI)) / (2 * logf(freq_ਅਧਾਰ)));
    ਦਸ਼ਮਲਵ ਸਿਰਾ   = ceilf(n_dims * logf(n_ctx_orig / (beta_slow * 2 * (ਦਸ਼ਮਲਵ) M_PI)) / (2 * logf(freq_ਅਧਾਰ)));
    dims[0]     = MAX(0, ਸ਼ੁਰੂ);
    dims[1]     = MIN(n_dims - 1, ਸਿਰਾ);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_rope_neox_mul(HVX_ਸਦਿਸ਼ v0, HVX_ਸਦਿਸ਼ v1, HVX_ਸਦਿਸ਼ vcos, HVX_ਸਦਿਸ਼ vsin,
                                     HVX_ਸਦਿਸ਼ * o0, HVX_ਸਦਿਸ਼ * o1) {
    HVX_ਸਦਿਸ਼ vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, vcos);
    HVX_ਸਦਿਸ਼ vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, vsin);
    HVX_ਸਦਿਸ਼ vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, vcos);
    HVX_ਸਦਿਸ਼ vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, vsin);
    *o0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s));
    *o1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c));
}

// theta_cache full 32-pair blocks are deinterleaved [cos | sin].
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_rope_neox_f32_aa(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src0, uint32_t ne, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ theta_cache) {
    ਅਚਲ uint32_t he = ne / 2;
    ਅਚਲ uint32_t nvec = he / 32;
    ਅਚਲ uint32_t nloe = he % 32;

    ਜੇ (nloe == 0) {
        ਅਚਲ HVX_ਸਦਿਸ਼ * vs = (ਅਚਲ HVX_ਸਦਿਸ਼ *) src0;
        ਅਚਲ HVX_ਸਦਿਸ਼ * vt = (ਅਚਲ HVX_ਸਦਿਸ਼ *) theta_cache;
        HVX_ਸਦਿਸ਼ * vd = (HVX_ਸਦਿਸ਼ *) dst;
        ਲਈ (uint32_t i = 0; i < nvec; i++) {
            HVX_ਸਦਿਸ਼ o0, o1;
            hvx_rope_neox_mul(vs[i], vs[nvec + i], vt[i * 2 + 0], vt[i * 2 + 1], &o0, &o1);
            vd[i] = o0;
            vd[nvec + i] = o1;
        }
        ਵਾਪਸ;
    }

    ਲਈ (uint32_t i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ o0, o1;
        hvx_rope_neox_mul(((ਅਚਲ HVX_ਸਦਿਸ਼ *) src0)[i],
                          hvx_vmemu(src0 + he + i * 32),
                          ((ਅਚਲ HVX_ਸਦਿਸ਼ *) theta_cache)[i * 2 + 0],
                          ((ਅਚਲ HVX_ਸਦਿਸ਼ *) theta_cache)[i * 2 + 1],
                          &o0, &o1);
        ((HVX_ਸਦਿਸ਼ *) dst)[i] = o0;
        hvx_vmemu(dst + he + i * 32) = o1;
    }

    HVX_ਸਦਿਸ਼ v0 = hvx_vmemu(src0 + nvec * 32);
    HVX_ਸਦਿਸ਼ v1 = hvx_vmemu(src0 + he + nvec * 32);
    HVX_ਸਦਿਸ਼ vcos = hvx_vmemu(theta_cache + nvec * 64);
    HVX_ਸਦਿਸ਼ vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);
    HVX_ਸਦਿਸ਼ o0, o1;
    hvx_rope_neox_mul(v0, v1, vcos, vsin, &o0, &o1);
    hvx_vec_store_u(dst + nvec * 32, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), o0);
    hvx_vec_store_u(dst + he + nvec * 32, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), o1);
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hvx_rope_f32_aa(ਦਸ਼ਮਲਵ * ਸੀਮਿਤ dst, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ src0, uint32_t ne, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ theta_cache) {
    ਅਚਲ uint32_t nvec = ne / 64;
    ਅਚਲ uint32_t nloe = ne % 64;

    ਲਈ (uint32_t i = 0; i < nvec; i++) {
        HVX_ਸਦਿਸ਼ v0 = ((ਅਚਲ HVX_ਸਦਿਸ਼ *) src0)[i * 2 + 0];
        HVX_ਸਦਿਸ਼ v1 = ((ਅਚਲ HVX_ਸਦਿਸ਼ *) src0)[i * 2 + 1];

        HVX_ਸਦਿਸ਼ vcos = ((ਅਚਲ HVX_ਸਦਿਸ਼ *) theta_cache)[i * 2 + 0];
        HVX_ਸਦਿਸ਼ vsin = ((ਅਚਲ HVX_ਸਦਿਸ਼ *) theta_cache)[i * 2 + 1];

        HVX_ਸਦਿਸ਼_Pair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

        HVX_ਸਦਿਸ਼ vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
        HVX_ਸਦਿਸ਼ vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
        HVX_ਸਦਿਸ਼ vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
        HVX_ਸਦਿਸ਼ vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

        HVX_ਸਦਿਸ਼ v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_ਸਦਿਸ਼ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        HVX_ਸਦਿਸ਼_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

        ((HVX_ਸਦਿਸ਼ *) dst)[i * 2 + 0] = Q6_V_lo_W(vstore);
        ((HVX_ਸਦਿਸ਼ *) dst)[i * 2 + 1] = Q6_V_hi_W(vstore);
    }

    ਜੇ (nloe > 0) {
        ਜੇ (nloe <= 32) {
            HVX_ਸਦਿਸ਼ v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_ਸਦਿਸ਼ vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_ਸਦਿਸ਼ vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_ਸਦਿਸ਼_Pair vx0_x1 = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, -4);

            HVX_ਸਦਿਸ਼ vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_ਸਦਿਸ਼ vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_ਸਦਿਸ਼ vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_ਸਦਿਸ਼ vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_ਸਦਿਸ਼ v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_ਸਦਿਸ਼ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_ਸਦਿਸ਼_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            hvx_vec_store_u(dst + nvec * 64, nloe * ਆਕਾਰ(ਦਸ਼ਮਲਵ), Q6_V_lo_W(vstore));
        } ਨਹੀਂਤਾਂ {
            HVX_ਸਦਿਸ਼ v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_ਸਦਿਸ਼ v1 = hvx_vmemu(src0 + nvec * 64 + 32);

            HVX_ਸਦਿਸ਼ vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_ਸਦਿਸ਼ vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_ਸਦਿਸ਼_Pair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

            HVX_ਸਦਿਸ਼ vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_ਸਦਿਸ਼ vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_ਸਦਿਸ਼ vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_ਸਦਿਸ਼ vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_ਸਦਿਸ਼ v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_ਸਦਿਸ਼ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_ਸਦਿਸ਼_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            ((HVX_ਸਦਿਸ਼ *) dst)[nvec * 2 + 0] = Q6_V_lo_W(vstore);
            hvx_vec_store_u(dst + nvec * 64 + 32, (nloe - 32) * ਆਕਾਰ(ਦਸ਼ਮਲਵ), Q6_V_hi_W(vstore));
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ਸਤਰਅੰਦਰ rope_basic_f32_inplace(ਬਣਤਰ htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ theta_cache) {
    ਅਚਲ uint32_t n_offs = rctx->n_offs;
    #ਹਦਾਇਤ unroll(4)
    ਲਈ (uint32_t i = 0; i < nr; i++) {
        ਦਸ਼ਮਲਵ * s = (ਦਸ਼ਮਲਵ *) (src + i * rctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        hvx_rope_f32_aa(s + n_offs, s + n_offs, rctx->n_dims, theta_cache);
    }
}

ਸਥਿਰ ਖਾਲੀ ਸਤਰਅੰਦਰ rope_neox_f32_inplace(ਬਣਤਰ htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, uint32_t ne, ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ theta_cache) {
    ਅਚਲ uint32_t n_offs = rctx->n_offs;
    #ਹਦਾਇਤ unroll(4)
    ਲਈ (uint32_t i = 0; i < nr; i++) {
        ਦਸ਼ਮਲਵ * s = (ਦਸ਼ਮਲਵ *) (src + i * rctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
        hvx_rope_neox_f32_aa(s + n_offs, s + n_offs, ne, theta_cache);
    }
}

ਸਥਿਰ ਖਾਲੀ rope_ਕੰਮ_f32(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ nth, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_rope_context * rctx = (ਬਣਤਰ htp_rope_context *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = rctx->octx;

    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * src2 = octx->src[2];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    htp_rope_preamble;

    ਅਚਲ uint32_t src0_nrows = rctx->nrows;
    ਅਚਲ uint32_t src0_nrows_ਪ੍ਰਤੀ_thread = rctx->src0_nrows_ਪ੍ਰਤੀ_thread;

    ਅਚਲ uint32_t src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ = rctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows_ਪ੍ਰਤੀ_thread * ith;
    ਅਚਲ uint32_t src0_ਸਿਰਾ_ਪੰਕਤੀ   = MIN(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ + src0_nrows_ਪ੍ਰਤੀ_thread, rctx->ਪੰਕਤੀ_ਸ਼ੁਰੂ + src0_nrows);

    // no work for this thread
    ਜੇ (src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ >= src0_ਸਿਰਾ_ਪੰਕਤੀ) {
        ਵਾਪਸ;
    }

    ਅਚਲ int32_t ਬਹੁਲਕ    = rctx->ਬਹੁਲਕ;
    // MROPE, IMROPE and VISION use NEOX-style pairing for the rotation
    ਅਚਲ bool    is_neox = (ਬਹੁਲਕ & HTP_ROPE_TYPE_NEOX) || (ਬਹੁਲਕ & HTP_ROPE_TYPE_MROPE);
    ਅਚਲ bool    is_vision = (ਬਹੁਲਕ == HTP_ROPE_TYPE_VISION);

    // VTCM setup
    uint8_t * src0_spad_ਅਧਾਰ = rctx->vtcm_ਅਧਾਰ + (ith * rctx->spad_ਪ੍ਰਤੀ_thread);
    ਦਸ਼ਮਲਵ *   theta_cache    = (ਦਸ਼ਮਲਵ *) (src0_spad_ਅਧਾਰ);
              src0_spad_ਅਧਾਰ = src0_spad_ਅਧਾਰ + rctx->theta_cache_ਵਿੱਥ;

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];
    ਅਚਲ int32_t * pos = (ਅਚਲ int32_t *) (uintptr_t) src1->ਅੰਕੜੇ;
    ਅਚਲ ਦਸ਼ਮਲਵ * freq_factors = rctx->freq_factors;

    ਅਚਲ uint32_t i3_ਸ਼ੁਰੂ = fastdiv(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ, &rctx->div_ne2_ne1);
    ਅਚਲ uint32_t rem      = fastmodulo(src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ, ne2 * ne1, &rctx->div_ne2_ne1);
    ਅਚਲ uint32_t i2_ਸ਼ੁਰੂ = fastdiv(rem, &rctx->div_ne1);
    ਅਚਲ uint32_t i1_ਸ਼ੁਰੂ = fastmodulo(rem, ne1, &rctx->div_ne1);

    uint32_t ir = src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ;
    uint32_t prev_i2 = (uint32_t) -1;
    uint32_t cur_ਥਾਂ = 0;

    ਲਈ (uint32_t i3 = i3_ਸ਼ੁਰੂ; i3 < ne3; i3++) { // batch
        ਅਚਲ uint32_t i2_init = (i3 == i3_ਸ਼ੁਰੂ) ? i2_ਸ਼ੁਰੂ : 0;
        ਲਈ (uint32_t i2 = i2_init; i2 < ne2; i2++) { // seq-len
            ਅਚਲ uint32_t i1_init = (i3 == i3_ਸ਼ੁਰੂ && i2 == i2_ਸ਼ੁਰੂ) ? i1_ਸ਼ੁਰੂ : 0;
            ਲਈ (uint32_t i1 = i1_init; i1 < ne1; ) { // attn-heads
                ਜੇ (ir >= src0_ਸਿਰਾ_ਪੰਕਤੀ) ਜਾਓ ਪੂਰਾ;

                // Rows in this block
                ਅਚਲ uint32_t nrows = MIN(src0_ਸਿਰਾ_ਪੰਕਤੀ - ir, ne1 - i1);

                // Depth before prefetch
                ਅਚਲ uint32_t dma_ਡੂੰਘਾਈ = dma_ਕਤਾਰ_ਡੂੰਘਾਈ(dma_q);

                // Prefetch up to 2 blocks
                ਅਚਲ uint32_t p_nrows = MIN(nrows, 2 * HTP_ROPE_SPAD_ਖੰਡ);
                ਲਈ (uint32_t pr = 0; pr < p_nrows; pr += HTP_ROPE_SPAD_ਖੰਡ) {
                    ਅਚਲ uint32_t pnr = MIN(nrows - pr, HTP_ROPE_SPAD_ਖੰਡ);
                    ਅਚਲ uint32_t ਥਾਂ = (cur_ਥਾਂ + pr / HTP_ROPE_SPAD_ਖੰਡ) % HTP_ROPE_SPAD_NSLOTS;
                    uint8_t * spad_ਥਾਂ = rope_spad_ਥਾਂ(src0_spad_ਅਧਾਰ, ਥਾਂ, rctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
                    ਅਚਲ dma_addr_t src0_ਅੰਕੜੇ = src0->ਅੰਕੜੇ + i3 * nb03 + i2 * nb02 + (i1 + pr) * nb01;

                    // Dummy DMA transaction for sequencing (interleaving wr, rd, wr, rd, ...)
                    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst->ਅੰਕੜੇ, spad_ਥਾਂ), 0, 0, 0, 0);

                    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(spad_ਥਾਂ, src0_ਅੰਕੜੇ),
                        rctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, rctx->src0_ਪੰਕਤੀ_stride, rctx->src0_ਪੰਕਤੀ_ਮਾਪ, pnr);
                }

                // Update theta cache
                ਜੇ (i2 != prev_i2) {
                    prev_i2 = i2;

                    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                    // VISION rotates the full row; other modes only rotate n_dims.
                    ਅਚਲ uint32_t n_cache = is_vision ? ne0 : (uint32_t) rctx->n_dims;
                    ਅਚਲ bool is_mrope = (rctx->ਬਹੁਲਕ & HTP_ROPE_TYPE_MROPE) != 0;
                    ਜੇ (is_mrope) {
                        // src1 holds four position arrays stacked along ne0:
                        // pos[i2], pos[i2+ne2], pos[i2+ne2*2], pos[i2+ne2*3]
                        ਅਚਲ bool is_imrope = (rctx->ਬਹੁਲਕ == HTP_ROPE_TYPE_IMROPE);
                        mrope_cache_init(
                            (ਦਸ਼ਮਲਵ) pos[i2],
                            (ਦਸ਼ਮਲਵ) pos[i2 + ne2],
                            (ਦਸ਼ਮਲਵ) pos[i2 + ne2 * 2],
                            (ਦਸ਼ਮਲਵ) pos[i2 + ne2 * 3],
                            rctx->sections, is_imrope, is_vision,
                            rctx->freq_scale, freq_factors, rctx->corr_dims,
                            n_cache, rctx->ext_ਗੁਣਕ, rctx->attn_ਗੁਣਕ,
                            theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    } ਨਹੀਂਤਾਂ {
                       rope_cache_init(pos[i2], rctx->freq_scale, freq_factors, rctx->corr_dims,
                                        n_cache, rctx->ext_ਗੁਣਕ, rctx->attn_ਗੁਣਕ,
                                        theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                }

                // Skip output DMA transactions from prev block (if any)
                ਲਈ (uint32_t d = 0; d < dma_ਡੂੰਘਾਈ; d++) { dma_ਕਤਾਰ_ਕੱਢੋ_nowait(dma_q); }

                // Compute loop
                ਅਚਲ uint32_t ne = is_vision ? ne0 : rctx->n_dims;
                ਅਚਲ uint32_t ਅਧਾਰ_i1 = i1;
                ਅਚਲ uint32_t ਅਧਾਰ_ir = ir;

                ਲਈ (uint32_t cnr = 0, cr = 0; cr < nrows; cr += cnr) {
                    cnr = MIN(nrows - cr, HTP_ROPE_SPAD_ਖੰਡ);
                    ਅਚਲ uint32_t ਥਾਂ = (cur_ਥਾਂ + cr / HTP_ROPE_SPAD_ਖੰਡ) % HTP_ROPE_SPAD_NSLOTS;
                    ਅਚਲ uint32_t cur_ir   = ਅਧਾਰ_ir + cr;
                    ਅਚਲ uint32_t cur_i1   = ਅਧਾਰ_i1 + cr;

                    dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
                    uint8_t * cur_spad = (uint8_t *) dma_ਕਤਾਰ_ਕੱਢੋ(dma_q).dst;

                    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);
                    ਜੇ (is_neox || is_vision) {
                        rope_neox_f32_inplace(rctx, cur_spad, cnr, ne, theta_cache);
                    } ਨਹੀਂਤਾਂ {
                        rope_basic_f32_inplace(rctx, cur_spad, cnr, theta_cache);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);

                    ਅਚਲ dma_addr_t dst_ਅੰਕੜੇ = dst->ਅੰਕੜੇ + i3 * nb3 + i2 * nb2 + cur_i1 * nb1;
                    dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_ਅੰਕੜੇ, cur_spad),
                        rctx->dst_ਪੰਕਤੀ_stride, rctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, rctx->dst_ਪੰਕਤੀ_ਮਾਪ, cnr);

                    // Prefetch 2 blocks ahead into the slot just freed
                    ਜੇ ((cr + 2 * HTP_ROPE_SPAD_ਖੰਡ) < nrows) {
                        ਅਚਲ uint32_t p_cr   = cr + 2 * HTP_ROPE_SPAD_ਖੰਡ;
                        ਅਚਲ uint32_t pnr    = MIN(nrows - p_cr, HTP_ROPE_SPAD_ਖੰਡ);
                        ਅਚਲ uint32_t p_ਥਾਂ = (cur_ਥਾਂ + p_cr / HTP_ROPE_SPAD_ਖੰਡ) % HTP_ROPE_SPAD_NSLOTS;
                        uint8_t * p_spad      = rope_spad_ਥਾਂ(src0_spad_ਅਧਾਰ, p_ਥਾਂ, rctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned);
                        ਅਚਲ dma_addr_t p_src0_ਅੰਕੜੇ = src0->ਅੰਕੜੇ + i3 * nb03 + i2 * nb02 + (ਅਧਾਰ_i1 + p_cr) * nb01;

                        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(p_spad, p_src0_ਅੰਕੜੇ),
                            rctx->src0_ਪੰਕਤੀ_ਮਾਪ_aligned, rctx->src0_ਪੰਕਤੀ_stride, rctx->src0_ਪੰਕਤੀ_ਮਾਪ, pnr);
                    }
                }

                ਅਚਲ uint32_t n_chunks = (nrows + HTP_ROPE_SPAD_ਖੰਡ - 1) / HTP_ROPE_SPAD_ਖੰਡ;
                cur_ਥਾਂ = (cur_ਥਾਂ + n_chunks) % HTP_ROPE_SPAD_NSLOTS;

                ir += nrows;
                i1 += nrows;
            }
        }
    }

ਪੂਰਾ:
    dma_ਕਤਾਰ_flush(dma_q);

    FARF(ਉੱਚਾ_2, "rope-f32: %d/%d: (%u:%u)\n", ith, nth, src0_ਸ਼ੁਰੂ_ਪੰਕਤੀ, src0_ਸਿਰਾ_ਪੰਕਤੀ);
}

ਸਥਿਰ ਪੂਰਨ execute_op_rope_f32(ਬਣਤਰ htp_ops_context * octx) {
    ਪੂਰਨ err = HTP_ਸਥਿਤੀ_OK;

    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * src1 = octx->src[1];
    ਅਚਲ ਬਣਤਰ htp_tensor * src2 = octx->src[2];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਚੋਣ (octx->op) {
        ਹਾਲਤ HTP_OP_ROPE:
            ਰੋਕੋ;

        ਮੂਲ:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ ਬਣਤਰ htp_rope_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_rope_kernel_params *) octx->kernel_params;
    ਜੇ (!htp_ops_context_ਸੈੱਟ_n_ਧਾਗੇ(octx, kparams->n_ਧਾਗੇ)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_INVAL_PARAMS;
    }
    ਪ੍ਰਤਿਜ੍ਞਾ(octx->ctx->vtcm_ਮਾਪ >= kparams->vtcm_ਮਾਪ);

    ਜੇ (htp_tensor_is_extended(src1)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ size_t dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ = dst->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t nrows     = ਕੁੱਲ_ਪੰਕਤੀਆਂ;

    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 0;
        htp_tensor_mdev_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk(dst, ਆਕਾਰ(ਦਸ਼ਮਲਵ), (uint32_t) dst_ਅੰਕੜੇ_ਪੰਕਤੀ_ਮਾਪ, &ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk);
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
            ਕੁੱਲ_ਪੰਕਤੀਆਂ, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        nrows     = range.ਗਿਣਤੀ_2;
    }

    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਅਚਲ uint32_t n_ਧਾਗੇ = octx->n_ਧਾਗੇ;

    ਅਚਲ uint32_t ne0 = dst->ne[0];
    ਅਚਲ size_t src0_ਪੰਕਤੀ_ਮਾਪ   = src0->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t src0_ਪੰਕਤੀ_stride = src0->nb[1];
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ    = dst->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ size_t dst_ਪੰਕਤੀ_stride  = dst->nb[1];

    ਬਣਤਰ htp_rope_context rctx;
    memset(&rctx, 0, ਆਕਾਰ(ਬਣਤਰ htp_rope_context));

    rctx.octx                  = octx;
    rctx.vtcm_ਅਧਾਰ             = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;
    rctx.spad_ਪ੍ਰਤੀ_thread       = kparams->spad_ਪ੍ਰਤੀ_thread;
    rctx.theta_cache_ਵਿੱਥ    = kparams->theta_cache_ਵਿੱਥ;

    ਜੇ (src2) {
        dma_ਕਤਾਰ * dma_q = octx->ctx->dma[0];
        ਅਚਲ size_t ff_ਮਾਪ = src2->ne[0] * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
        ਦਸ਼ਮਲਵ * vtcm_freq_factors = (ਦਸ਼ਮਲਵ *) (rctx.vtcm_ਅਧਾਰ + kparams->freq_factors_ਵਿੱਥ);
        dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(vtcm_freq_factors, src2->ਅੰਕੜੇ),
                       kparams->freq_factors_ਮਾਪ, 0, ff_ਮਾਪ, 1);
        dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);
        rctx.freq_factors = vtcm_freq_factors;
    }

    ਅਚਲ int32_t * op_params = &octx->op_params[0];
    rctx.n_dims     = ((ਅਚਲ int32_t *) op_params)[1];
    rctx.ਬਹੁਲਕ       = ((ਅਚਲ int32_t *) op_params)[2];
    rctx.n_ctx_orig = ((ਅਚਲ int32_t *) op_params)[4];
    rctx.n_offs     = ((ਅਚਲ int32_t *) op_params)[15];

    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&rctx.freq_ਅਧਾਰ,   (int32_t *) op_params + 5,  ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&rctx.freq_scale,  (int32_t *) op_params + 6,  ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&rctx.ext_ਗੁਣਕ,  (int32_t *) op_params + 7,  ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&rctx.attn_ਗੁਣਕ, (int32_t *) op_params + 8,  ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&rctx.beta_fast,   (int32_t *) op_params + 9,  ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&rctx.beta_slow,   (int32_t *) op_params + 10, ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਸਮ੍ਰਿਤੀ_ਨਕਲ(&rctx.sections,    (int32_t *) op_params + 11, ਆਕਾਰ(ਪੂਰਨ) * 4);

    rctx.theta_scale = powf(rctx.freq_ਅਧਾਰ, -2.0f / rctx.n_dims);
    rctx.theta_powers[0] = 1.0f;
    ਲਈ (ਪੂਰਨ j = 1; j < 32; j++) {
        rctx.theta_powers[j] = rctx.theta_powers[j - 1] * rctx.theta_scale;
    }
    rctx.theta_scale_32 = rctx.theta_powers[31] * rctx.theta_scale;

    rope_corr_dims(rctx.n_dims, rctx.n_ctx_orig, rctx.freq_ਅਧਾਰ, rctx.beta_fast, rctx.beta_slow, rctx.corr_dims);

    rctx.src0_ਪੰਕਤੀ_ਮਾਪ         = src0_ਪੰਕਤੀ_ਮਾਪ;
    rctx.src0_ਪੰਕਤੀ_stride       = src0_ਪੰਕਤੀ_stride;
    rctx.dst_ਪੰਕਤੀ_ਮਾਪ          = dst_ਪੰਕਤੀ_ਮਾਪ;
    rctx.dst_ਪੰਕਤੀ_stride        = dst_ਪੰਕਤੀ_stride;
    rctx.src0_ਪੰਕਤੀ_ਮਾਪ_aligned = kparams->src0_ਪੰਕਤੀ_ਮਾਪ_aligned;

    rctx.src0_nrows            = nrows;
    rctx.nrows                 = nrows;
    rctx.ਪੰਕਤੀ_ਸ਼ੁਰੂ             = ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    rctx.src0_nrows_ਪ੍ਰਤੀ_thread = fastdiv(nrows + n_ਧਾਗੇ - 1, &octx->n_ਧਾਗੇ_div);
    rctx.div_ne2_ne1           = kparams->div_ne2_ne1;
    rctx.div_ne1               = kparams->div_ne1;

    FARF(ਉੱਚਾ_2, "rope-f32 n-rows %u n-dims %d ne0 %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n", rctx.src0_nrows, rctx.n_dims, ne0,
         rctx.ext_ਗੁਣਕ, rctx.theta_scale, rctx.attn_ਗੁਣਕ);

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, rope_ਕੰਮ_f32, &rctx, n_ਧਾਗੇ);

    ਵਾਪਸ err;
}

ਪੂਰਨ op_rope(ਬਣਤਰ htp_ops_context * octx) {
    ਚੋਣ (octx->src[0]->type) {
        ਹਾਲਤ HTP_TYPE_F32:
            ਵਾਪਸ execute_op_rope_f32(octx);

        ਮੂਲ:
            ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }
}
