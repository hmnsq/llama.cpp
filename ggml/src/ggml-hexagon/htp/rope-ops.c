#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#شامل <math.h>
#شامل <string.h>
#شامل <stdlib.h>

#شامل "dma-queue.h"
#شامل "hvx-utils.h"
#شامل "hex-fastdiv.h"

#تعریف_کرو GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "rope-ops.h"

// Redefined the rope type constants as we can't include ggml.h
#تعریف_کرو HTP_ROPE_TYPE_معمول 0
#تعریف_کرو HTP_ROPE_TYPE_NEOX   2
#تعریف_کرو HTP_ROPE_TYPE_MROPE  8
#تعریف_کرو HTP_ROPE_TYPE_VISION 24
#تعریف_کرو HTP_ROPE_TYPE_IMROPE 40

#تعریف_کرو htp_rope_preamble              \
    ثابت uint32_t ne00 = src0->ne[0]; \
    ثابت uint32_t ne01 = src0->ne[1]; \
    ثابت uint32_t ne02 = src0->ne[2]; \
    ثابت uint32_t ne03 = src0->ne[3]; \
                                       \
    ثابت uint32_t ne0 = dst->ne[0];   \
    ثابت uint32_t ne1 = dst->ne[1];   \
    ثابت uint32_t ne2 = dst->ne[2];   \
    ثابت uint32_t ne3 = dst->ne[3];   \
                                       \
    ثابت uint32_t nb00 = src0->nb[0]; \
    ثابت uint32_t nb01 = src0->nb[1]; \
    ثابت uint32_t nb02 = src0->nb[2]; \
    ثابت uint32_t nb03 = src0->nb[3]; \
                                       \
    ثابت uint32_t nb0 = dst->nb[0];   \
    ثابت uint32_t nb1 = dst->nb[1];   \
    ثابت uint32_t nb2 = dst->nb[2];   \
    ثابت uint32_t nb3 = dst->nb[3];

ساخت htp_rope_context {
    int32_t n_dims;
    int32_t n_offs;
    int32_t منوال;
    int32_t n_ctx_orig;
    int32_t sections[4];

    اعشاری freq_بنیاد;
    اعشاری freq_scale;
    اعشاری ext_ضربی_عدد;
    اعشاری attn_ضربی_عدد;
    اعشاری beta_fast;
    اعشاری beta_slow;
    اعشاری theta_scale;
    اعشاری theta_scale_32;
    اعشاری theta_powers[32];
    اعشاری corr_dims[2];

    uint32_t src0_nrows_فی_thread;

    ساخت htp_ops_context * octx;

    uint8_t * vtcm_بنیاد;
    size_t    spad_فی_thread;
    size_t    theta_cache_انحراف;

    size_t src0_صف_سائز;
    size_t src0_صف_stride;
    size_t dst_صف_سائز;
    size_t dst_صف_stride;
    size_t src0_صف_سائز_aligned;
    uint32_t src0_nrows;
    uint32_t صف_آغاز;
    uint32_t nrows;

    ساخت fastdiv_اقدار div_ne2_ne1;
    ساخت fastdiv_اقدار div_ne1;

    ثابت اعشاری * freq_factors;
};

ساکن اعشاری rope_yarn_ramp(ثابت اعشاری کم, ثابت اعشاری بلند, ثابت صحیح i0) {
    ثابت اعشاری y = (i0 / 2 - کم) / MAX(0.001f, بلند - کم);

    واپس (1 - MIN(1, MAX(0, y)));
}

// Compute one (cos, sin) pair into cache[i0], cache[i0+1] applying YaRN scaling.
ساکن اندرونی خالی rope_yarn_one(اعشاری theta, اعشاری freq_scale, اعشاری * corr_dims,
                                 uint32_t i0, اعشاری ext_ضربی_عدد, اعشاری mscale,
                                 اعشاری * cache) {
    اعشاری theta_extrap = theta;

    // Get n-d rotational scaling corrected for extrapolation
    اعشاری theta_interp = freq_scale * theta_extrap;
    اعشاری theta_final  = theta_interp;
    اعشاری mscale_final = mscale;

    اگر (ext_ضربی_عدد != 0.0f) {
        اعشاری ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_ضربی_عدد;
        theta_final    = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale_final  *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    ثابت uint32_t b = i0 / 64;
    ثابت uint32_t k = (i0 % 64) / 2;
    cache[b * 64 + k]      = cosf(theta_final) * mscale_final;
    cache[b * 64 + 32 + k] = sinf(theta_final) * mscale_final;
}

// 32 thetas -> 32 deinterleaved pairs [cos[32] | sin[32]] at cache[i0].
ساکن اندرونی خالی rope_cache_hvx_32(اعشاری * cache, uint32_t i0,
                                     HVX_سمتیہ v_theta,
                                     ثابت اعشاری * freq_factors,
                                     HVX_سمتیہ v_freq_scale,
                                     HVX_سمتیہ v_mscale) {
    اگر (freq_factors) {
        HVX_سمتیہ v_ff = hvx_vmemu(freq_factors + i0 / 2);
        v_theta = hvx_vec_mul_f32_f32(v_theta, hvx_vec_inverse_f32(v_ff));
    }

    HVX_سمتیہ v_theta_final = hvx_vec_mul_f32_f32(v_theta, v_freq_scale);
    HVX_سمتیہ vcos;
    HVX_سمتیہ vsin;
    hvx_vec_sincos_f32(v_theta_final, &vcos, &vsin);
    vcos = hvx_vec_mul_f32_f32(vcos, v_mscale);
    vsin = hvx_vec_mul_f32_f32(vsin, v_mscale);

    اگر (((uintptr_t) (cache + i0)) % 128 == 0) {
        hvx_vmem(cache + i0 + 0)  = vcos;
        hvx_vmem(cache + i0 + 32) = vsin;
    } ورنہ {
        hvx_vec_store_u(cache + i0 + 0,  32 * حجم(اعشاری), vcos);
        hvx_vec_store_u(cache + i0 + 32, 32 * حجم(اعشاری), vsin);
    }
}

ساکن __attribute__((noinline)) خالی rope_cache_init(ثابت اعشاری    theta_بنیاد,
                            ثابت اعشاری    freq_scale,
                            ثابت اعشاری *  freq_factors,
                            اعشاری *        corr_dims,
                            ثابت uint32_t n_cache,
                            ثابت اعشاری    ext_ضربی_عدد,
                            ثابت اعشاری    mscale,
                            اعشاری *        cache,
                            ثابت اعشاری    theta_scale,
                            ثابت اعشاری *  theta_powers,
                            ثابت اعشاری    theta_scale_32) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    اگر (ext_ضربی_عدد == 0.0f) {
        // Fast path: fully vectorized
        // We process 32 pairs (64 elements) per iteration.
        ثابت uint32_t n_بلاکیں = n_cache / 64;

        HVX_سمتیہ v_theta_powers = hvx_vmemu(theta_powers);
        HVX_سمتیہ v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_سمتیہ v_mscale = hvx_vec_splat_f32(mscale);

        اعشاری theta_بلاک = theta_بنیاد;

        برائے (uint32_t b = 0; b < n_بلاکیں; b++) {
            uint32_t i0 = b * 64;
            HVX_سمتیہ v_theta_بنیاد = hvx_vec_splat_f32(theta_بلاک);
            HVX_سمتیہ v_theta = hvx_vec_mul_f32_f32(v_theta_بنیاد, v_theta_powers);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            theta_بلاک *= theta_scale_32;
        }

        // Leftovers
        اعشاری theta = theta_بلاک;
        برائے (uint32_t i0 = n_بلاکیں * 64; i0 < n_cache; i0 += 2) {
            ثابت اعشاری ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ضربی_عدد, mscale, cache);
            theta *= theta_scale;
        }
    } ورنہ {
        اعشاری theta = theta_بنیاد;
        برائے (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
            ثابت اعشاری ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ضربی_عدد, mscale, cache);
            theta *= theta_scale;
        }
    }
}

ساکن اندرونی اعشاری mrope_pick_theta(اعشاری theta_t, اعشاری theta_h, اعشاری theta_w, اعشاری theta_e,
                                     صحیح sector, ثابت int32_t sections[4], صحیح sec_w, صحیح sec_e,
                                     bool is_imrope) {
    اگر (is_imrope) {
        اگر      (sector % 3 == 0 && sector < 3 * sections[0]) { واپس theta_t; }
        ورنہ اگر (sector % 3 == 1 && sector < 3 * sections[1]) { واپس theta_h; }
        ورنہ اگر (sector % 3 == 2 && sector < 3 * sections[2]) { واپس theta_w; }
        ورنہ                                                   { واپس theta_e; }
    }
    اگر      (sector < sections[0]) { واپس theta_t; }
    ورنہ اگر (sector < sec_w)       { واپس theta_h; }
    ورنہ اگر (sector < sec_e)       { واپس theta_w; }
    ورنہ                           { واپس theta_e; }
}

// lane j is 1 when (j % 3) == rem
ساکن ثابت اعشاری __attribute__((aligned(128))) mrope_mod3_eq0[32] = {
    1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0
};
ساکن ثابت اعشاری __attribute__((aligned(128))) mrope_mod3_eq1[32] = {
    0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1
};
ساکن ثابت اعشاری __attribute__((aligned(128))) mrope_mod3_eq2[32] = {
    0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0
};

ساکن ثابت اعشاری __attribute__((aligned(128))) mrope_k_ramp[32] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
};

ساکن اندرونی HVX_سمتیہ_Pred mrope_نقاب_eq1(ثابت اعشاری * m) {
    واپس Q6_Q_vcmp_gt_VsfVsf(hvx_vmemu(m), Q6_V_vzero());
}

// IMROPE without wrap: theta[k] = pos[k % 3] * scale^k
ساکن اندرونی HVX_سمتیہ mrope_thetas_imrope_mod3(اعشاری pos_t, اعشاری pos_h, اعشاری pos_w,
                                                  uint32_t k0, HVX_سمتیہ v_powers, اعشاری scale_بلاک) {
    ثابت صحیح r = (صحیح) (k0 % 3);
    ثابت اعشاری * mt = (r == 0) ? mrope_mod3_eq0 : (r == 1) ? mrope_mod3_eq2 : mrope_mod3_eq1;
    ثابت اعشاری * mh = (r == 0) ? mrope_mod3_eq1 : (r == 1) ? mrope_mod3_eq0 : mrope_mod3_eq2;

    HVX_سمتیہ v = hvx_vec_splat_f32(pos_w);
    v = Q6_V_vmux_QVV(mrope_نقاب_eq1(mh), hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(mrope_نقاب_eq1(mt), hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    واپس hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_بلاک));
}

// Contiguous MROPE without wrap: theta[k] = pos[section(k)] * scale^k
ساکن اندرونی HVX_سمتیہ mrope_thetas_contig(اعشاری pos_t, اعشاری pos_h, اعشاری pos_w, اعشاری pos_e,
                                             uint32_t k0, صحیح s0, صحیح sec_w, صحیح sec_e,
                                             HVX_سمتیہ v_powers, اعشاری scale_بلاک) {
    HVX_سمتیہ v_k = hvx_vec_جوڑو_f32_f32(hvx_vec_splat_f32((اعشاری) k0), hvx_vmemu(mrope_k_ramp));
    HVX_سمتیہ_Pred lt_s0 = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((اعشاری) s0), v_k);
    HVX_سمتیہ_Pred lt_sw = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((اعشاری) sec_w), v_k);
    HVX_سمتیہ_Pred lt_se = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((اعشاری) sec_e), v_k);

    HVX_سمتیہ v = hvx_vec_splat_f32(pos_e);
    v = Q6_V_vmux_QVV(lt_se, hvx_vec_splat_f32(pos_w), v);
    v = Q6_V_vmux_QVV(lt_sw, hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(lt_s0, hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    واپس hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_بلاک));
}

// pos_t/h/w/e: the four position ids for this sequence step (t=time, h=height, w=width, e=extra).
// sections[4]: number of head dims assigned to each position component.
ساکن __attribute__((noinline)) خالی mrope_cache_init(ثابت اعشاری    pos_t,
                             ثابت اعشاری    pos_h,
                             ثابت اعشاری    pos_w,
                             ثابت اعشاری    pos_e,
                             ثابت int32_t  sections[4],
                             ثابت bool     is_imrope,
                             ثابت bool     indep_sects,
                             ثابت اعشاری    freq_scale,
                             ثابت اعشاری *  freq_factors,
                             اعشاری *        corr_dims,
                             ثابت uint32_t n_cache,
                             ثابت اعشاری    ext_ضربی_عدد,
                             ثابت اعشاری    mscale,
                             اعشاری *        cache,
                             ثابت اعشاری    theta_scale,
                             ثابت اعشاری *  theta_powers,
                             ثابت اعشاری    theta_scale_32) {
    ثابت صحیح sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    ثابت صحیح sec_w     = sections[0] + sections[1];
    ثابت صحیح sec_e     = sec_w + sections[2];
    ثابت uint32_t n_جوڑے = n_cache / 2;

    ثابت bool no_wrap = (sect_dims > 0) && (n_جوڑے <= (uint32_t) sect_dims);
    ثابت bool imrope_mod3 = is_imrope && !indep_sects && no_wrap
        && sections[0] > 0 && sections[1] > 0 && sections[2] > 0
        && n_جوڑے <= (uint32_t) (3 * sections[0])
        && n_جوڑے <= (uint32_t) (3 * sections[1])
        && n_جوڑے <= (uint32_t) (3 * sections[2]);
    ثابت bool contig = !is_imrope && !indep_sects && no_wrap;

    اگر (ext_ضربی_عدد == 0.0f && (imrope_mod3 || contig)) {
        HVX_سمتیہ v_powers     = hvx_vmemu(theta_powers);
        HVX_سمتیہ v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_سمتیہ v_mscale     = hvx_vec_splat_f32(mscale);
        اعشاری scale_بلاک = 1.0f;
        ثابت uint32_t n_بلاکیں = n_cache / 64;

        برائے (uint32_t b = 0; b < n_بلاکیں; b++) {
            ثابت uint32_t i0 = b * 64;
            ثابت uint32_t k0 = b * 32;
            HVX_سمتیہ v_theta = imrope_mod3
                ? mrope_thetas_imrope_mod3(pos_t, pos_h, pos_w, k0, v_powers, scale_بلاک)
                : mrope_thetas_contig(pos_t, pos_h, pos_w, pos_e, k0, sections[0], sec_w, sec_e,
                                      v_powers, scale_بلاک);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            scale_بلاک *= theta_scale_32;
        }

        اعشاری theta_k = scale_بلاک;
        برائے (uint32_t k = n_بلاکیں * 32; k < n_جوڑے; k++) {
            ثابت uint32_t i0 = 2 * k;
            ثابت اعشاری pos = mrope_pick_theta(pos_t, pos_h, pos_w, pos_e,
                                               (صحیح) k, sections, sec_w, sec_e, is_imrope);
            ثابت اعشاری ff = freq_factors ? freq_factors[k] : 1.0f;
            rope_yarn_one(pos * theta_k / ff, freq_scale, corr_dims, i0, ext_ضربی_عدد, mscale, cache);
            theta_k *= theta_scale;
        }
        واپس;
    }

    اعشاری theta_t = pos_t;
    اعشاری theta_h = pos_h;
    اعشاری theta_w = pos_w;
    اعشاری theta_e = pos_e;

    ثابت bool use_hvx = (ext_ضربی_عدد == 0.0f);
    اعشاری __attribute__((aligned(128))) thetas[32];
    uint32_t n_thetas = 0;
    uint32_t بلاک_i0 = 0;

    HVX_سمتیہ v_freq_scale = hvx_vec_splat_f32(freq_scale);
    HVX_سمتیہ v_mscale     = hvx_vec_splat_f32(mscale);

    برائے (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
        ثابت صحیح sector = (i0 / 2) % sect_dims;

        اگر (indep_sects) {
            // Reset theta when crossing into a new section.
            اگر      (sector == 0)           { theta_t = pos_t; }
            ورنہ اگر (sector == sections[0]) { theta_h = pos_h; }
            ورنہ اگر (sector == sec_w)       { theta_w = pos_w; }
            ورنہ اگر (sector == sec_e)       { theta_e = pos_e; }
        }

        ثابت اعشاری theta = mrope_pick_theta(theta_t, theta_h, theta_w, theta_e,
                                             sector, sections, sec_w, sec_e, is_imrope);

        اگر (use_hvx) {
            اگر (n_thetas == 0) {
                بلاک_i0 = i0;
            }
            thetas[n_thetas++] = theta;
            اگر (n_thetas == 32) {
                rope_cache_hvx_32(cache, بلاک_i0, hvx_vmemu(thetas), freq_factors, v_freq_scale, v_mscale);
                n_thetas = 0;
            }
        } ورنہ {
            ثابت اعشاری ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ضربی_عدد, mscale, cache);
        }

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }

    برائے (uint32_t k = 0; k < n_thetas; k++) {
        ثابت uint32_t i0 = بلاک_i0 + 2 * k;
        ثابت اعشاری ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn_one(thetas[k] / ff, freq_scale, corr_dims, i0, ext_ضربی_عدد, mscale, cache);
    }
}

#تعریف_کرو M_PI 3.1415926535897932384626433

ساکن خالی rope_corr_dims(صحیح     n_dims,
                           صحیح     n_ctx_orig,
                           اعشاری   freq_بنیاد,
                           اعشاری   beta_fast,
                           اعشاری   beta_slow,
                           اعشاری * dims) {
    اعشاری آغاز = floorf(n_dims * logf(n_ctx_orig / (beta_fast * 2 * (اعشاری) M_PI)) / (2 * logf(freq_بنیاد)));
    اعشاری انتہا   = ceilf(n_dims * logf(n_ctx_orig / (beta_slow * 2 * (اعشاری) M_PI)) / (2 * logf(freq_بنیاد)));
    dims[0]     = MAX(0, آغاز);
    dims[1]     = MIN(n_dims - 1, انتہا);
}

ساکن اندرونی خالی hvx_rope_neox_mul(HVX_سمتیہ v0, HVX_سمتیہ v1, HVX_سمتیہ vcos, HVX_سمتیہ vsin,
                                     HVX_سمتیہ * o0, HVX_سمتیہ * o1) {
    HVX_سمتیہ vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, vcos);
    HVX_سمتیہ vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, vsin);
    HVX_سمتیہ vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, vcos);
    HVX_سمتیہ vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, vsin);
    *o0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s));
    *o1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c));
}

// theta_cache full 32-pair blocks are deinterleaved [cos | sin].
ساکن اندرونی خالی hvx_rope_neox_f32_aa(اعشاری * محدود dst, ثابت اعشاری * محدود src0, uint32_t ne, ثابت اعشاری * محدود theta_cache) {
    ثابت uint32_t he = ne / 2;
    ثابت uint32_t nvec = he / 32;
    ثابت uint32_t nloe = he % 32;

    اگر (nloe == 0) {
        ثابت HVX_سمتیہ * vs = (ثابت HVX_سمتیہ *) src0;
        ثابت HVX_سمتیہ * vt = (ثابت HVX_سمتیہ *) theta_cache;
        HVX_سمتیہ * vd = (HVX_سمتیہ *) dst;
        برائے (uint32_t i = 0; i < nvec; i++) {
            HVX_سمتیہ o0, o1;
            hvx_rope_neox_mul(vs[i], vs[nvec + i], vt[i * 2 + 0], vt[i * 2 + 1], &o0, &o1);
            vd[i] = o0;
            vd[nvec + i] = o1;
        }
        واپس;
    }

    برائے (uint32_t i = 0; i < nvec; i++) {
        HVX_سمتیہ o0, o1;
        hvx_rope_neox_mul(((ثابت HVX_سمتیہ *) src0)[i],
                          hvx_vmemu(src0 + he + i * 32),
                          ((ثابت HVX_سمتیہ *) theta_cache)[i * 2 + 0],
                          ((ثابت HVX_سمتیہ *) theta_cache)[i * 2 + 1],
                          &o0, &o1);
        ((HVX_سمتیہ *) dst)[i] = o0;
        hvx_vmemu(dst + he + i * 32) = o1;
    }

    HVX_سمتیہ v0 = hvx_vmemu(src0 + nvec * 32);
    HVX_سمتیہ v1 = hvx_vmemu(src0 + he + nvec * 32);
    HVX_سمتیہ vcos = hvx_vmemu(theta_cache + nvec * 64);
    HVX_سمتیہ vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);
    HVX_سمتیہ o0, o1;
    hvx_rope_neox_mul(v0, v1, vcos, vsin, &o0, &o1);
    hvx_vec_store_u(dst + nvec * 32, nloe * حجم(اعشاری), o0);
    hvx_vec_store_u(dst + he + nvec * 32, nloe * حجم(اعشاری), o1);
}

ساکن اندرونی خالی hvx_rope_f32_aa(اعشاری * محدود dst, ثابت اعشاری * محدود src0, uint32_t ne, ثابت اعشاری * محدود theta_cache) {
    ثابت uint32_t nvec = ne / 64;
    ثابت uint32_t nloe = ne % 64;

    برائے (uint32_t i = 0; i < nvec; i++) {
        HVX_سمتیہ v0 = ((ثابت HVX_سمتیہ *) src0)[i * 2 + 0];
        HVX_سمتیہ v1 = ((ثابت HVX_سمتیہ *) src0)[i * 2 + 1];

        HVX_سمتیہ vcos = ((ثابت HVX_سمتیہ *) theta_cache)[i * 2 + 0];
        HVX_سمتیہ vsin = ((ثابت HVX_سمتیہ *) theta_cache)[i * 2 + 1];

        HVX_سمتیہ_Pair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

        HVX_سمتیہ vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
        HVX_سمتیہ vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
        HVX_سمتیہ vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
        HVX_سمتیہ vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

        HVX_سمتیہ v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_سمتیہ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        HVX_سمتیہ_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

        ((HVX_سمتیہ *) dst)[i * 2 + 0] = Q6_V_lo_W(vstore);
        ((HVX_سمتیہ *) dst)[i * 2 + 1] = Q6_V_hi_W(vstore);
    }

    اگر (nloe > 0) {
        اگر (nloe <= 32) {
            HVX_سمتیہ v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_سمتیہ vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_سمتیہ vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_سمتیہ_Pair vx0_x1 = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, -4);

            HVX_سمتیہ vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_سمتیہ vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_سمتیہ vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_سمتیہ vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_سمتیہ v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_سمتیہ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_سمتیہ_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            hvx_vec_store_u(dst + nvec * 64, nloe * حجم(اعشاری), Q6_V_lo_W(vstore));
        } ورنہ {
            HVX_سمتیہ v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_سمتیہ v1 = hvx_vmemu(src0 + nvec * 64 + 32);

            HVX_سمتیہ vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_سمتیہ vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_سمتیہ_Pair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

            HVX_سمتیہ vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_سمتیہ vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_سمتیہ vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_سمتیہ vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_سمتیہ v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_سمتیہ v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_سمتیہ_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            ((HVX_سمتیہ *) dst)[nvec * 2 + 0] = Q6_V_lo_W(vstore);
            hvx_vec_store_u(dst + nvec * 64 + 32, (nloe - 32) * حجم(اعشاری), Q6_V_hi_W(vstore));
        }
    }
}

ساکن خالی اندرونی rope_basic_f32_inplace(ساخت htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, ثابت اعشاری * محدود theta_cache) {
    ثابت uint32_t n_offs = rctx->n_offs;
    #ہدایت unroll(4)
    برائے (uint32_t i = 0; i < nr; i++) {
        اعشاری * s = (اعشاری *) (src + i * rctx->src0_صف_سائز_aligned);
        hvx_rope_f32_aa(s + n_offs, s + n_offs, rctx->n_dims, theta_cache);
    }
}

ساکن خالی اندرونی rope_neox_f32_inplace(ساخت htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, uint32_t ne, ثابت اعشاری * محدود theta_cache) {
    ثابت uint32_t n_offs = rctx->n_offs;
    #ہدایت unroll(4)
    برائے (uint32_t i = 0; i < nr; i++) {
        اعشاری * s = (اعشاری *) (src + i * rctx->src0_صف_سائز_aligned);
        hvx_rope_neox_f32_aa(s + n_offs, s + n_offs, ne, theta_cache);
    }
}

ساکن خالی rope_کام_f32(غیرعلامتی صحیح nth, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_rope_context * rctx = (ساخت htp_rope_context *) ڈیٹا;
    ساخت htp_ops_context * octx = rctx->octx;

    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * src1 = octx->src[1];
    ثابت ساخت htp_tensor * src2 = octx->src[2];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    htp_rope_preamble;

    ثابت uint32_t src0_nrows = rctx->nrows;
    ثابت uint32_t src0_nrows_فی_thread = rctx->src0_nrows_فی_thread;

    ثابت uint32_t src0_آغاز_صف = rctx->صف_آغاز + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_انتہا_صف   = MIN(src0_آغاز_صف + src0_nrows_فی_thread, rctx->صف_آغاز + src0_nrows);

    // no work for this thread
    اگر (src0_آغاز_صف >= src0_انتہا_صف) {
        واپس;
    }

    ثابت int32_t منوال    = rctx->منوال;
    // MROPE, IMROPE and VISION use NEOX-style pairing for the rotation
    ثابت bool    is_neox = (منوال & HTP_ROPE_TYPE_NEOX) || (منوال & HTP_ROPE_TYPE_MROPE);
    ثابت bool    is_vision = (منوال == HTP_ROPE_TYPE_VISION);

    // VTCM setup
    uint8_t * src0_spad_بنیاد = rctx->vtcm_بنیاد + (ith * rctx->spad_فی_thread);
    اعشاری *   theta_cache    = (اعشاری *) (src0_spad_بنیاد);
              src0_spad_بنیاد = src0_spad_بنیاد + rctx->theta_cache_انحراف;

    dma_قطار * dma_q = octx->ctx->dma[ith];
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];
    ثابت int32_t * pos = (ثابت int32_t *) (uintptr_t) src1->ڈیٹا;
    ثابت اعشاری * freq_factors = rctx->freq_factors;

    ثابت uint32_t i3_آغاز = fastdiv(src0_آغاز_صف, &rctx->div_ne2_ne1);
    ثابت uint32_t rem      = fastmodulo(src0_آغاز_صف, ne2 * ne1, &rctx->div_ne2_ne1);
    ثابت uint32_t i2_آغاز = fastdiv(rem, &rctx->div_ne1);
    ثابت uint32_t i1_آغاز = fastmodulo(rem, ne1, &rctx->div_ne1);

    uint32_t ir = src0_آغاز_صف;
    uint32_t پچھلا_i2 = (uint32_t) -1;
    uint32_t cur_خانہ = 0;

    برائے (uint32_t i3 = i3_آغاز; i3 < ne3; i3++) { // batch
        ثابت uint32_t i2_init = (i3 == i3_آغاز) ? i2_آغاز : 0;
        برائے (uint32_t i2 = i2_init; i2 < ne2; i2++) { // seq-len
            ثابت uint32_t i1_init = (i3 == i3_آغاز && i2 == i2_آغاز) ? i1_آغاز : 0;
            برائے (uint32_t i1 = i1_init; i1 < ne1; ) { // attn-heads
                اگر (ir >= src0_انتہا_صف) جاؤ مکمل;

                // Rows in this block
                ثابت uint32_t nrows = MIN(src0_انتہا_صف - ir, ne1 - i1);

                // Depth before prefetch
                ثابت uint32_t dma_گہرائی = dma_قطار_گہرائی(dma_q);

                // Prefetch up to 2 blocks
                ثابت uint32_t p_nrows = MIN(nrows, 2 * HTP_ROPE_SPAD_بلاک);
                برائے (uint32_t pr = 0; pr < p_nrows; pr += HTP_ROPE_SPAD_بلاک) {
                    ثابت uint32_t pnr = MIN(nrows - pr, HTP_ROPE_SPAD_بلاک);
                    ثابت uint32_t خانہ = (cur_خانہ + pr / HTP_ROPE_SPAD_بلاک) % HTP_ROPE_SPAD_NSLOTS;
                    uint8_t * spad_خانہ = rope_spad_خانہ(src0_spad_بنیاد, خانہ, rctx->src0_صف_سائز_aligned);
                    ثابت dma_addr_t src0_ڈیٹا = src0->ڈیٹا + i3 * nb03 + i2 * nb02 + (i1 + pr) * nb01;

                    // Dummy DMA transaction for sequencing (interleaving wr, rd, wr, rd, ...)
                    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst->ڈیٹا, spad_خانہ), 0, 0, 0, 0);

                    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(spad_خانہ, src0_ڈیٹا),
                        rctx->src0_صف_سائز_aligned, rctx->src0_صف_stride, rctx->src0_صف_سائز, pnr);
                }

                // Update theta cache
                اگر (i2 != پچھلا_i2) {
                    پچھلا_i2 = i2;

                    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                    // VISION rotates the full row; other modes only rotate n_dims.
                    ثابت uint32_t n_cache = is_vision ? ne0 : (uint32_t) rctx->n_dims;
                    ثابت bool is_mrope = (rctx->منوال & HTP_ROPE_TYPE_MROPE) != 0;
                    اگر (is_mrope) {
                        // src1 holds four position arrays stacked along ne0:
                        // pos[i2], pos[i2+ne2], pos[i2+ne2*2], pos[i2+ne2*3]
                        ثابت bool is_imrope = (rctx->منوال == HTP_ROPE_TYPE_IMROPE);
                        mrope_cache_init(
                            (اعشاری) pos[i2],
                            (اعشاری) pos[i2 + ne2],
                            (اعشاری) pos[i2 + ne2 * 2],
                            (اعشاری) pos[i2 + ne2 * 3],
                            rctx->sections, is_imrope, is_vision,
                            rctx->freq_scale, freq_factors, rctx->corr_dims,
                            n_cache, rctx->ext_ضربی_عدد, rctx->attn_ضربی_عدد,
                            theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    } ورنہ {
                       rope_cache_init(pos[i2], rctx->freq_scale, freq_factors, rctx->corr_dims,
                                        n_cache, rctx->ext_ضربی_عدد, rctx->attn_ضربی_عدد,
                                        theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                }

                // Skip output DMA transactions from prev block (if any)
                برائے (uint32_t d = 0; d < dma_گہرائی; d++) { dma_قطار_نکالو_nowait(dma_q); }

                // Compute loop
                ثابت uint32_t ne = is_vision ? ne0 : rctx->n_dims;
                ثابت uint32_t بنیاد_i1 = i1;
                ثابت uint32_t بنیاد_ir = ir;

                برائے (uint32_t cnr = 0, cr = 0; cr < nrows; cr += cnr) {
                    cnr = MIN(nrows - cr, HTP_ROPE_SPAD_بلاک);
                    ثابت uint32_t خانہ = (cur_خانہ + cr / HTP_ROPE_SPAD_بلاک) % HTP_ROPE_SPAD_NSLOTS;
                    ثابت uint32_t cur_ir   = بنیاد_ir + cr;
                    ثابت uint32_t cur_i1   = بنیاد_i1 + cr;

                    dma_قطار_نکالو(dma_q);
                    uint8_t * cur_spad = (uint8_t *) dma_قطار_نکالو(dma_q).dst;

                    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);
                    اگر (is_neox || is_vision) {
                        rope_neox_f32_inplace(rctx, cur_spad, cnr, ne, theta_cache);
                    } ورنہ {
                        rope_basic_f32_inplace(rctx, cur_spad, cnr, theta_cache);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);

                    ثابت dma_addr_t dst_ڈیٹا = dst->ڈیٹا + i3 * nb3 + i2 * nb2 + cur_i1 * nb1;
                    dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_ڈیٹا, cur_spad),
                        rctx->dst_صف_stride, rctx->src0_صف_سائز_aligned, rctx->dst_صف_سائز, cnr);

                    // Prefetch 2 blocks ahead into the slot just freed
                    اگر ((cr + 2 * HTP_ROPE_SPAD_بلاک) < nrows) {
                        ثابت uint32_t p_cr   = cr + 2 * HTP_ROPE_SPAD_بلاک;
                        ثابت uint32_t pnr    = MIN(nrows - p_cr, HTP_ROPE_SPAD_بلاک);
                        ثابت uint32_t p_خانہ = (cur_خانہ + p_cr / HTP_ROPE_SPAD_بلاک) % HTP_ROPE_SPAD_NSLOTS;
                        uint8_t * p_spad      = rope_spad_خانہ(src0_spad_بنیاد, p_خانہ, rctx->src0_صف_سائز_aligned);
                        ثابت dma_addr_t p_src0_ڈیٹا = src0->ڈیٹا + i3 * nb03 + i2 * nb02 + (بنیاد_i1 + p_cr) * nb01;

                        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(p_spad, p_src0_ڈیٹا),
                            rctx->src0_صف_سائز_aligned, rctx->src0_صف_stride, rctx->src0_صف_سائز, pnr);
                    }
                }

                ثابت uint32_t n_chunks = (nrows + HTP_ROPE_SPAD_بلاک - 1) / HTP_ROPE_SPAD_بلاک;
                cur_خانہ = (cur_خانہ + n_chunks) % HTP_ROPE_SPAD_NSLOTS;

                ir += nrows;
                i1 += nrows;
            }
        }
    }

مکمل:
    dma_قطار_flush(dma_q);

    FARF(بلند_2, "rope-f32: %d/%d: (%u:%u)\n", ith, nth, src0_آغاز_صف, src0_انتہا_صف);
}

ساکن صحیح execute_op_rope_f32(ساخت htp_ops_context * octx) {
    صحیح err = HTP_حالت_OK;

    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * src1 = octx->src[1];
    ثابت ساخت htp_tensor * src2 = octx->src[2];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    انتخاب (octx->op) {
        حالت HTP_OP_ROPE:
            توڑو;

        طےشدہ:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت ساخت htp_rope_kernel_params * kparams = (ثابت ساخت htp_rope_kernel_params *) octx->kernel_params;
    اگر (!htp_ops_context_سیٹ_n_دھاگے(octx, kparams->n_دھاگے)) {
        واپس HTP_حالت_INVAL_PARAMS;
    }
    أكد(octx->ctx->vtcm_سائز >= kparams->vtcm_سائز);

    اگر (htp_tensor_is_extended(src1)) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t کل_صفیں = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_ڈیٹا_صف_سائز = dst->ne[0] * حجم(اعشاری);

    uint32_t صف_آغاز = 0;
    uint32_t nrows     = کل_صفیں;

    اگر (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفیں_فی_chunk = 0;
        htp_tensor_mdev_صفیں_فی_chunk(dst, حجم(اعشاری), (uint32_t) dst_ڈیٹا_صف_سائز, &صفیں_فی_chunk);
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(
            کل_صفیں, صفیں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        nrows     = range.گنتی;
    }

    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    ثابت uint32_t ne0 = dst->ne[0];
    ثابت size_t src0_صف_سائز   = src0->ne[0] * حجم(اعشاری);
    ثابت size_t src0_صف_stride = src0->nb[1];
    ثابت size_t dst_صف_سائز    = dst->ne[0] * حجم(اعشاری);
    ثابت size_t dst_صف_stride  = dst->nb[1];

    ساخت htp_rope_context rctx;
    memset(&rctx, 0, حجم(ساخت htp_rope_context));

    rctx.octx                  = octx;
    rctx.vtcm_بنیاد             = (uint8_t *) octx->ctx->vtcm_بنیاد;
    rctx.spad_فی_thread       = kparams->spad_فی_thread;
    rctx.theta_cache_انحراف    = kparams->theta_cache_انحراف;

    اگر (src2) {
        dma_قطار * dma_q = octx->ctx->dma[0];
        ثابت size_t ff_سائز = src2->ne[0] * حجم(اعشاری);
        اعشاری * vtcm_freq_factors = (اعشاری *) (rctx.vtcm_بنیاد + kparams->freq_factors_انحراف);
        dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(vtcm_freq_factors, src2->ڈیٹا),
                       kparams->freq_factors_سائز, 0, ff_سائز, 1);
        dma_قطار_نکالو(dma_q);
        rctx.freq_factors = vtcm_freq_factors;
    }

    ثابت int32_t * op_params = &octx->op_params[0];
    rctx.n_dims     = ((ثابت int32_t *) op_params)[1];
    rctx.منوال       = ((ثابت int32_t *) op_params)[2];
    rctx.n_ctx_orig = ((ثابت int32_t *) op_params)[4];
    rctx.n_offs     = ((ثابت int32_t *) op_params)[15];

    حافظہ_نقل(&rctx.freq_بنیاد,   (int32_t *) op_params + 5,  حجم(اعشاری));
    حافظہ_نقل(&rctx.freq_scale,  (int32_t *) op_params + 6,  حجم(اعشاری));
    حافظہ_نقل(&rctx.ext_ضربی_عدد,  (int32_t *) op_params + 7,  حجم(اعشاری));
    حافظہ_نقل(&rctx.attn_ضربی_عدد, (int32_t *) op_params + 8,  حجم(اعشاری));
    حافظہ_نقل(&rctx.beta_fast,   (int32_t *) op_params + 9,  حجم(اعشاری));
    حافظہ_نقل(&rctx.beta_slow,   (int32_t *) op_params + 10, حجم(اعشاری));
    حافظہ_نقل(&rctx.sections,    (int32_t *) op_params + 11, حجم(صحیح) * 4);

    rctx.theta_scale = powf(rctx.freq_بنیاد, -2.0f / rctx.n_dims);
    rctx.theta_powers[0] = 1.0f;
    برائے (صحیح j = 1; j < 32; j++) {
        rctx.theta_powers[j] = rctx.theta_powers[j - 1] * rctx.theta_scale;
    }
    rctx.theta_scale_32 = rctx.theta_powers[31] * rctx.theta_scale;

    rope_corr_dims(rctx.n_dims, rctx.n_ctx_orig, rctx.freq_بنیاد, rctx.beta_fast, rctx.beta_slow, rctx.corr_dims);

    rctx.src0_صف_سائز         = src0_صف_سائز;
    rctx.src0_صف_stride       = src0_صف_stride;
    rctx.dst_صف_سائز          = dst_صف_سائز;
    rctx.dst_صف_stride        = dst_صف_stride;
    rctx.src0_صف_سائز_aligned = kparams->src0_صف_سائز_aligned;

    rctx.src0_nrows            = nrows;
    rctx.nrows                 = nrows;
    rctx.صف_آغاز             = صف_آغاز;
    rctx.src0_nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div);
    rctx.div_ne2_ne1           = kparams->div_ne2_ne1;
    rctx.div_ne1               = kparams->div_ne1;

    FARF(بلند_2, "rope-f32 n-rows %u n-dims %d ne0 %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n", rctx.src0_nrows, rctx.n_dims, ne0,
         rctx.ext_ضربی_عدد, rctx.theta_scale, rctx.attn_ضربی_عدد);

    work_قطار_تسلسل(octx->ctx->work_قطار, rope_کام_f32, &rctx, n_دھاگے);

    واپس err;
}

صحیح op_rope(ساخت htp_ops_context * octx) {
    انتخاب (octx->src[0]->type) {
        حالت HTP_TYPE_F32:
            واپس execute_op_rope_f32(octx);

        طےشدہ:
            واپس HTP_حالت_NO_SUPPORT;
    }
}
