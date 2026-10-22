#ہدایت clang diagnostic ignored "-Wunused-variable"
#ہدایت clang diagnostic ignored "-Wunused-function"
#ہدایت clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <math.h>
#ضمن <string.h>
#ضمن <stdlib.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"
#ضمن "hex-fastdiv.h"

#تعریف_کرو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "rope-ops.h"

// Redefined the rope type constants as we can't include ggml.h
#تعریف_کرو HTP_ROPE_TYPE_عام 0
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

بنية htp_rope_context {
    int32_t n_dims;
    int32_t n_offs;
    int32_t منوال;
    int32_t n_ctx_orig;
    int32_t sections[4];

    عائم freq_ادھار;
    عائم freq_scale;
    عائم ext_ضارب;
    عائم attn_ضارب;
    عائم beta_fast;
    عائم beta_slow;
    عائم theta_scale;
    عائم theta_scale_32;
    عائم theta_powers[32];
    عائم corr_dims[2];

    uint32_t src0_nrows_فی_thread;

    بنية htp_ops_context * octx;

    uint8_t * vtcm_ادھار;
    size_t    spad_فی_thread;
    size_t    theta_cache_وتھ;

    size_t src0_صف_ناپ;
    size_t src0_صف_stride;
    size_t dst_صف_ناپ;
    size_t dst_صف_stride;
    size_t src0_صف_ناپ_aligned;
    uint32_t src0_nrows;
    uint32_t صف_شروع;
    uint32_t nrows;

    بنية fastdiv_اقدار div_ne2_ne1;
    بنية fastdiv_اقدار div_ne1;

    ثابت عائم * freq_factors;
};

ساكن عائم rope_yarn_ramp(ثابت عائم کم, ثابت عائم بلند, ثابت صحيح i0) {
    ثابت عائم y = (i0 / 2 - کم) / MAX(0.001f, بلند - کم);

    أرجع (1 - MIN(1, MAX(0, y)));
}

// Compute one (cos, sin) pair into cache[i0], cache[i0+1] applying YaRN scaling.
ساكن مضمن فراغ rope_yarn_one(عائم theta, عائم freq_scale, عائم * corr_dims,
                                 uint32_t i0, عائم ext_ضارب, عائم mscale,
                                 عائم * cache) {
    عائم theta_extrap = theta;

    // Get n-d rotational scaling corrected for extrapolation
    عائم theta_interp = freq_scale * theta_extrap;
    عائم theta_final  = theta_interp;
    عائم mscale_final = mscale;

    إذا (ext_ضارب != 0.0f) {
        عائم ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_ضارب;
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
ساكن مضمن فراغ rope_cache_hvx_32(عائم * cache, uint32_t i0,
                                     HVX_سمتیہ v_theta,
                                     ثابت عائم * freq_factors,
                                     HVX_سمتیہ v_freq_scale,
                                     HVX_سمتیہ v_mscale) {
    إذا (freq_factors) {
        HVX_سمتیہ v_ff = hvx_vmemu(freq_factors + i0 / 2);
        v_theta = hvx_vec_mul_f32_f32(v_theta, hvx_vec_inverse_f32(v_ff));
    }

    HVX_سمتیہ v_theta_final = hvx_vec_mul_f32_f32(v_theta, v_freq_scale);
    HVX_سمتیہ vcos;
    HVX_سمتیہ vsin;
    hvx_vec_sincos_f32(v_theta_final, &vcos, &vsin);
    vcos = hvx_vec_mul_f32_f32(vcos, v_mscale);
    vsin = hvx_vec_mul_f32_f32(vsin, v_mscale);

    إذا (((uintptr_t) (cache + i0)) % 128 == 0) {
        hvx_vmem(cache + i0 + 0)  = vcos;
        hvx_vmem(cache + i0 + 32) = vsin;
    } وإلا {
        hvx_vec_store_u(cache + i0 + 0,  32 * حجم(عائم), vcos);
        hvx_vec_store_u(cache + i0 + 32, 32 * حجم(عائم), vsin);
    }
}

ساكن __attribute__((noinline)) فراغ rope_cache_init(ثابت عائم    theta_ادھار,
                            ثابت عائم    freq_scale,
                            ثابت عائم *  freq_factors,
                            عائم *        corr_dims,
                            ثابت uint32_t n_cache,
                            ثابت عائم    ext_ضارب,
                            ثابت عائم    mscale,
                            عائم *        cache,
                            ثابت عائم    theta_scale,
                            ثابت عائم *  theta_powers,
                            ثابت عائم    theta_scale_32) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    إذا (ext_ضارب == 0.0f) {
        // Fast path: fully vectorized
        // We process 32 pairs (64 elements) per iteration.
        ثابت uint32_t n_بلاکاں = n_cache / 64;

        HVX_سمتیہ v_theta_powers = hvx_vmemu(theta_powers);
        HVX_سمتیہ v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_سمتیہ v_mscale = hvx_vec_splat_f32(mscale);

        عائم theta_بلاک = theta_ادھار;

        لكل (uint32_t b = 0; b < n_بلاکاں; b++) {
            uint32_t i0 = b * 64;
            HVX_سمتیہ v_theta_ادھار = hvx_vec_splat_f32(theta_بلاک);
            HVX_سمتیہ v_theta = hvx_vec_mul_f32_f32(v_theta_ادھار, v_theta_powers);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            theta_بلاک *= theta_scale_32;
        }

        // Leftovers
        عائم theta = theta_بلاک;
        لكل (uint32_t i0 = n_بلاکاں * 64; i0 < n_cache; i0 += 2) {
            ثابت عائم ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ضارب, mscale, cache);
            theta *= theta_scale;
        }
    } وإلا {
        عائم theta = theta_ادھار;
        لكل (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
            ثابت عائم ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ضارب, mscale, cache);
            theta *= theta_scale;
        }
    }
}

ساكن مضمن عائم mrope_pick_theta(عائم theta_t, عائم theta_h, عائم theta_w, عائم theta_e,
                                     صحيح sector, ثابت int32_t sections[4], صحيح sec_w, صحيح sec_e,
                                     bool is_imrope) {
    إذا (is_imrope) {
        إذا      (sector % 3 == 0 && sector < 3 * sections[0]) { أرجع theta_t; }
        وإلا إذا (sector % 3 == 1 && sector < 3 * sections[1]) { أرجع theta_h; }
        وإلا إذا (sector % 3 == 2 && sector < 3 * sections[2]) { أرجع theta_w; }
        وإلا                                                   { أرجع theta_e; }
    }
    إذا      (sector < sections[0]) { أرجع theta_t; }
    وإلا إذا (sector < sec_w)       { أرجع theta_h; }
    وإلا إذا (sector < sec_e)       { أرجع theta_w; }
    وإلا                           { أرجع theta_e; }
}

// lane j is 1 when (j % 3) == rem
ساكن ثابت عائم __attribute__((aligned(128))) mrope_mod3_eq0[32] = {
    1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0
};
ساكن ثابت عائم __attribute__((aligned(128))) mrope_mod3_eq1[32] = {
    0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1
};
ساكن ثابت عائم __attribute__((aligned(128))) mrope_mod3_eq2[32] = {
    0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0
};

ساكن ثابت عائم __attribute__((aligned(128))) mrope_k_ramp[32] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
};

ساكن مضمن HVX_سمتیہ_Pred mrope_نقاب_eq1(ثابت عائم * m) {
    أرجع Q6_Q_vcmp_gt_VsfVsf(hvx_vmemu(m), Q6_V_vzero());
}

// IMROPE without wrap: theta[k] = pos[k % 3] * scale^k
ساكن مضمن HVX_سمتیہ mrope_thetas_imrope_mod3(عائم pos_t, عائم pos_h, عائم pos_w,
                                                  uint32_t k0, HVX_سمتیہ v_powers, عائم scale_بلاک) {
    ثابت صحيح r = (صحيح) (k0 % 3);
    ثابت عائم * mt = (r == 0) ? mrope_mod3_eq0 : (r == 1) ? mrope_mod3_eq2 : mrope_mod3_eq1;
    ثابت عائم * mh = (r == 0) ? mrope_mod3_eq1 : (r == 1) ? mrope_mod3_eq0 : mrope_mod3_eq2;

    HVX_سمتیہ v = hvx_vec_splat_f32(pos_w);
    v = Q6_V_vmux_QVV(mrope_نقاب_eq1(mh), hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(mrope_نقاب_eq1(mt), hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    أرجع hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_بلاک));
}

// Contiguous MROPE without wrap: theta[k] = pos[section(k)] * scale^k
ساكن مضمن HVX_سمتیہ mrope_thetas_contig(عائم pos_t, عائم pos_h, عائم pos_w, عائم pos_e,
                                             uint32_t k0, صحيح s0, صحيح sec_w, صحيح sec_e,
                                             HVX_سمتیہ v_powers, عائم scale_بلاک) {
    HVX_سمتیہ v_k = hvx_vec_جوڑو_f32_f32(hvx_vec_splat_f32((عائم) k0), hvx_vmemu(mrope_k_ramp));
    HVX_سمتیہ_Pred lt_s0 = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((عائم) s0), v_k);
    HVX_سمتیہ_Pred lt_sw = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((عائم) sec_w), v_k);
    HVX_سمتیہ_Pred lt_se = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((عائم) sec_e), v_k);

    HVX_سمتیہ v = hvx_vec_splat_f32(pos_e);
    v = Q6_V_vmux_QVV(lt_se, hvx_vec_splat_f32(pos_w), v);
    v = Q6_V_vmux_QVV(lt_sw, hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(lt_s0, hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    أرجع hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_بلاک));
}

// pos_t/h/w/e: the four position ids for this sequence step (t=time, h=height, w=width, e=extra).
// sections[4]: number of head dims assigned to each position component.
ساكن __attribute__((noinline)) فراغ mrope_cache_init(ثابت عائم    pos_t,
                             ثابت عائم    pos_h,
                             ثابت عائم    pos_w,
                             ثابت عائم    pos_e,
                             ثابت int32_t  sections[4],
                             ثابت bool     is_imrope,
                             ثابت bool     indep_sects,
                             ثابت عائم    freq_scale,
                             ثابت عائم *  freq_factors,
                             عائم *        corr_dims,
                             ثابت uint32_t n_cache,
                             ثابت عائم    ext_ضارب,
                             ثابت عائم    mscale,
                             عائم *        cache,
                             ثابت عائم    theta_scale,
                             ثابت عائم *  theta_powers,
                             ثابت عائم    theta_scale_32) {
    ثابت صحيح sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    ثابت صحيح sec_w     = sections[0] + sections[1];
    ثابت صحيح sec_e     = sec_w + sections[2];
    ثابت uint32_t n_جوڑے = n_cache / 2;

    ثابت bool no_wrap = (sect_dims > 0) && (n_جوڑے <= (uint32_t) sect_dims);
    ثابت bool imrope_mod3 = is_imrope && !indep_sects && no_wrap
        && sections[0] > 0 && sections[1] > 0 && sections[2] > 0
        && n_جوڑے <= (uint32_t) (3 * sections[0])
        && n_جوڑے <= (uint32_t) (3 * sections[1])
        && n_جوڑے <= (uint32_t) (3 * sections[2]);
    ثابت bool contig = !is_imrope && !indep_sects && no_wrap;

    إذا (ext_ضارب == 0.0f && (imrope_mod3 || contig)) {
        HVX_سمتیہ v_powers     = hvx_vmemu(theta_powers);
        HVX_سمتیہ v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_سمتیہ v_mscale     = hvx_vec_splat_f32(mscale);
        عائم scale_بلاک = 1.0f;
        ثابت uint32_t n_بلاکاں = n_cache / 64;

        لكل (uint32_t b = 0; b < n_بلاکاں; b++) {
            ثابت uint32_t i0 = b * 64;
            ثابت uint32_t k0 = b * 32;
            HVX_سمتیہ v_theta = imrope_mod3
                ? mrope_thetas_imrope_mod3(pos_t, pos_h, pos_w, k0, v_powers, scale_بلاک)
                : mrope_thetas_contig(pos_t, pos_h, pos_w, pos_e, k0, sections[0], sec_w, sec_e,
                                      v_powers, scale_بلاک);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            scale_بلاک *= theta_scale_32;
        }

        عائم theta_k = scale_بلاک;
        لكل (uint32_t k = n_بلاکاں * 32; k < n_جوڑے; k++) {
            ثابت uint32_t i0 = 2 * k;
            ثابت عائم pos = mrope_pick_theta(pos_t, pos_h, pos_w, pos_e,
                                               (صحيح) k, sections, sec_w, sec_e, is_imrope);
            ثابت عائم ff = freq_factors ? freq_factors[k] : 1.0f;
            rope_yarn_one(pos * theta_k / ff, freq_scale, corr_dims, i0, ext_ضارب, mscale, cache);
            theta_k *= theta_scale;
        }
        أرجع;
    }

    عائم theta_t = pos_t;
    عائم theta_h = pos_h;
    عائم theta_w = pos_w;
    عائم theta_e = pos_e;

    ثابت bool use_hvx = (ext_ضارب == 0.0f);
    عائم __attribute__((aligned(128))) thetas[32];
    uint32_t n_thetas = 0;
    uint32_t بلاک_i0 = 0;

    HVX_سمتیہ v_freq_scale = hvx_vec_splat_f32(freq_scale);
    HVX_سمتیہ v_mscale     = hvx_vec_splat_f32(mscale);

    لكل (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
        ثابت صحيح sector = (i0 / 2) % sect_dims;

        إذا (indep_sects) {
            // Reset theta when crossing into a new section.
            إذا      (sector == 0)           { theta_t = pos_t; }
            وإلا إذا (sector == sections[0]) { theta_h = pos_h; }
            وإلا إذا (sector == sec_w)       { theta_w = pos_w; }
            وإلا إذا (sector == sec_e)       { theta_e = pos_e; }
        }

        ثابت عائم theta = mrope_pick_theta(theta_t, theta_h, theta_w, theta_e,
                                             sector, sections, sec_w, sec_e, is_imrope);

        إذا (use_hvx) {
            إذا (n_thetas == 0) {
                بلاک_i0 = i0;
            }
            thetas[n_thetas++] = theta;
            إذا (n_thetas == 32) {
                rope_cache_hvx_32(cache, بلاک_i0, hvx_vmemu(thetas), freq_factors, v_freq_scale, v_mscale);
                n_thetas = 0;
            }
        } وإلا {
            ثابت عائم ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ضارب, mscale, cache);
        }

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }

    لكل (uint32_t k = 0; k < n_thetas; k++) {
        ثابت uint32_t i0 = بلاک_i0 + 2 * k;
        ثابت عائم ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn_one(thetas[k] / ff, freq_scale, corr_dims, i0, ext_ضارب, mscale, cache);
    }
}

#تعریف_کرو M_PI 3.1415926535897932384626433

ساكن فراغ rope_corr_dims(صحيح     n_dims,
                           صحيح     n_ctx_orig,
                           عائم   freq_ادھار,
                           عائم   beta_fast,
                           عائم   beta_slow,
                           عائم * dims) {
    عائم شروع = floorf(n_dims * logf(n_ctx_orig / (beta_fast * 2 * (عائم) M_PI)) / (2 * logf(freq_ادھار)));
    عائم سرا   = ceilf(n_dims * logf(n_ctx_orig / (beta_slow * 2 * (عائم) M_PI)) / (2 * logf(freq_ادھار)));
    dims[0]     = MAX(0, شروع);
    dims[1]     = MIN(n_dims - 1, سرا);
}

ساكن مضمن فراغ hvx_rope_neox_mul(HVX_سمتیہ v0, HVX_سمتیہ v1, HVX_سمتیہ vcos, HVX_سمتیہ vsin,
                                     HVX_سمتیہ * o0, HVX_سمتیہ * o1) {
    HVX_سمتیہ vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, vcos);
    HVX_سمتیہ vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, vsin);
    HVX_سمتیہ vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, vcos);
    HVX_سمتیہ vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, vsin);
    *o0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s));
    *o1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c));
}

// theta_cache full 32-pair blocks are deinterleaved [cos | sin].
ساكن مضمن فراغ hvx_rope_neox_f32_aa(عائم * محدود dst, ثابت عائم * محدود src0, uint32_t ne, ثابت عائم * محدود theta_cache) {
    ثابت uint32_t he = ne / 2;
    ثابت uint32_t nvec = he / 32;
    ثابت uint32_t nloe = he % 32;

    إذا (nloe == 0) {
        ثابت HVX_سمتیہ * vs = (ثابت HVX_سمتیہ *) src0;
        ثابت HVX_سمتیہ * vt = (ثابت HVX_سمتیہ *) theta_cache;
        HVX_سمتیہ * vd = (HVX_سمتیہ *) dst;
        لكل (uint32_t i = 0; i < nvec; i++) {
            HVX_سمتیہ o0, o1;
            hvx_rope_neox_mul(vs[i], vs[nvec + i], vt[i * 2 + 0], vt[i * 2 + 1], &o0, &o1);
            vd[i] = o0;
            vd[nvec + i] = o1;
        }
        أرجع;
    }

    لكل (uint32_t i = 0; i < nvec; i++) {
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
    hvx_vec_store_u(dst + nvec * 32, nloe * حجم(عائم), o0);
    hvx_vec_store_u(dst + he + nvec * 32, nloe * حجم(عائم), o1);
}

ساكن مضمن فراغ hvx_rope_f32_aa(عائم * محدود dst, ثابت عائم * محدود src0, uint32_t ne, ثابت عائم * محدود theta_cache) {
    ثابت uint32_t nvec = ne / 64;
    ثابت uint32_t nloe = ne % 64;

    لكل (uint32_t i = 0; i < nvec; i++) {
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

    إذا (nloe > 0) {
        إذا (nloe <= 32) {
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

            hvx_vec_store_u(dst + nvec * 64, nloe * حجم(عائم), Q6_V_lo_W(vstore));
        } وإلا {
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
            hvx_vec_store_u(dst + nvec * 64 + 32, (nloe - 32) * حجم(عائم), Q6_V_hi_W(vstore));
        }
    }
}

ساكن فراغ مضمن rope_basic_f32_inplace(بنية htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, ثابت عائم * محدود theta_cache) {
    ثابت uint32_t n_offs = rctx->n_offs;
    #ہدایت unroll(4)
    لكل (uint32_t i = 0; i < nr; i++) {
        عائم * s = (عائم *) (src + i * rctx->src0_صف_ناپ_aligned);
        hvx_rope_f32_aa(s + n_offs, s + n_offs, rctx->n_dims, theta_cache);
    }
}

ساكن فراغ مضمن rope_neox_f32_inplace(بنية htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, uint32_t ne, ثابت عائم * محدود theta_cache) {
    ثابت uint32_t n_offs = rctx->n_offs;
    #ہدایت unroll(4)
    لكل (uint32_t i = 0; i < nr; i++) {
        عائم * s = (عائم *) (src + i * rctx->src0_صف_ناپ_aligned);
        hvx_rope_neox_f32_aa(s + n_offs, s + n_offs, ne, theta_cache);
    }
}

ساكن فراغ rope_کم_f32(بنانشانی صحيح nth, بنانشانی صحيح ith, فراغ * انکڑے) {
    بنية htp_rope_context * rctx = (بنية htp_rope_context *) انکڑے;
    بنية htp_ops_context * octx = rctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * src2 = octx->src[2];
    ثابت بنية htp_tensor * dst  = octx->dst;

    htp_rope_preamble;

    ثابت uint32_t src0_nrows = rctx->nrows;
    ثابت uint32_t src0_nrows_فی_thread = rctx->src0_nrows_فی_thread;

    ثابت uint32_t src0_شروع_صف = rctx->صف_شروع + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_سرا_صف   = MIN(src0_شروع_صف + src0_nrows_فی_thread, rctx->صف_شروع + src0_nrows);

    // no work for this thread
    إذا (src0_شروع_صف >= src0_سرا_صف) {
        أرجع;
    }

    ثابت int32_t منوال    = rctx->منوال;
    // MROPE, IMROPE and VISION use NEOX-style pairing for the rotation
    ثابت bool    is_neox = (منوال & HTP_ROPE_TYPE_NEOX) || (منوال & HTP_ROPE_TYPE_MROPE);
    ثابت bool    is_vision = (منوال == HTP_ROPE_TYPE_VISION);

    // VTCM setup
    uint8_t * src0_spad_ادھار = rctx->vtcm_ادھار + (ith * rctx->spad_فی_thread);
    عائم *   theta_cache    = (عائم *) (src0_spad_ادھار);
              src0_spad_ادھار = src0_spad_ادھار + rctx->theta_cache_وتھ;

    dma_قطار * dma_q = octx->ctx->dma[ith];
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    ثابت int32_t * pos = (ثابت int32_t *) (uintptr_t) src1->انکڑے;
    ثابت عائم * freq_factors = rctx->freq_factors;

    ثابت uint32_t i3_شروع = fastdiv(src0_شروع_صف, &rctx->div_ne2_ne1);
    ثابت uint32_t rem      = fastmodulo(src0_شروع_صف, ne2 * ne1, &rctx->div_ne2_ne1);
    ثابت uint32_t i2_شروع = fastdiv(rem, &rctx->div_ne1);
    ثابت uint32_t i1_شروع = fastmodulo(rem, ne1, &rctx->div_ne1);

    uint32_t ir = src0_شروع_صف;
    uint32_t prev_i2 = (uint32_t) -1;
    uint32_t cur_تھاں = 0;

    لكل (uint32_t i3 = i3_شروع; i3 < ne3; i3++) { // batch
        ثابت uint32_t i2_init = (i3 == i3_شروع) ? i2_شروع : 0;
        لكل (uint32_t i2 = i2_init; i2 < ne2; i2++) { // seq-len
            ثابت uint32_t i1_init = (i3 == i3_شروع && i2 == i2_شروع) ? i1_شروع : 0;
            لكل (uint32_t i1 = i1_init; i1 < ne1; ) { // attn-heads
                إذا (ir >= src0_سرا_صف) اذهب پورا;

                // Rows in this block
                ثابت uint32_t nrows = MIN(src0_سرا_صف - ir, ne1 - i1);

                // Depth before prefetch
                ثابت uint32_t dma_ڈونگھائی = dma_قطار_ڈونگھائی(dma_q);

                // Prefetch up to 2 blocks
                ثابت uint32_t p_nrows = MIN(nrows, 2 * HTP_ROPE_SPAD_بلاک);
                لكل (uint32_t pr = 0; pr < p_nrows; pr += HTP_ROPE_SPAD_بلاک) {
                    ثابت uint32_t pnr = MIN(nrows - pr, HTP_ROPE_SPAD_بلاک);
                    ثابت uint32_t تھاں = (cur_تھاں + pr / HTP_ROPE_SPAD_بلاک) % HTP_ROPE_SPAD_NSLOTS;
                    uint8_t * spad_تھاں = rope_spad_تھاں(src0_spad_ادھار, تھاں, rctx->src0_صف_ناپ_aligned);
                    ثابت dma_addr_t src0_انکڑے = src0->انکڑے + i3 * nb03 + i2 * nb02 + (i1 + pr) * nb01;

                    // Dummy DMA transaction for sequencing (interleaving wr, rd, wr, rd, ...)
                    dma_قطار_ڈالو(dma_q, dma_make_انکڑے(dst->انکڑے, spad_تھاں), 0, 0, 0, 0);

                    dma_قطار_ڈالو(dma_q, dma_make_انکڑے(spad_تھاں, src0_انکڑے),
                        rctx->src0_صف_ناپ_aligned, rctx->src0_صف_stride, rctx->src0_صف_ناپ, pnr);
                }

                // Update theta cache
                إذا (i2 != prev_i2) {
                    prev_i2 = i2;

                    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                    // VISION rotates the full row; other modes only rotate n_dims.
                    ثابت uint32_t n_cache = is_vision ? ne0 : (uint32_t) rctx->n_dims;
                    ثابت bool is_mrope = (rctx->منوال & HTP_ROPE_TYPE_MROPE) != 0;
                    إذا (is_mrope) {
                        // src1 holds four position arrays stacked along ne0:
                        // pos[i2], pos[i2+ne2], pos[i2+ne2*2], pos[i2+ne2*3]
                        ثابت bool is_imrope = (rctx->منوال == HTP_ROPE_TYPE_IMROPE);
                        mrope_cache_init(
                            (عائم) pos[i2],
                            (عائم) pos[i2 + ne2],
                            (عائم) pos[i2 + ne2 * 2],
                            (عائم) pos[i2 + ne2 * 3],
                            rctx->sections, is_imrope, is_vision,
                            rctx->freq_scale, freq_factors, rctx->corr_dims,
                            n_cache, rctx->ext_ضارب, rctx->attn_ضارب,
                            theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    } وإلا {
                       rope_cache_init(pos[i2], rctx->freq_scale, freq_factors, rctx->corr_dims,
                                        n_cache, rctx->ext_ضارب, rctx->attn_ضارب,
                                        theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                }

                // Skip output DMA transactions from prev block (if any)
                لكل (uint32_t d = 0; d < dma_ڈونگھائی; d++) { dma_قطار_نکالو_nowait(dma_q); }

                // Compute loop
                ثابت uint32_t ne = is_vision ? ne0 : rctx->n_dims;
                ثابت uint32_t ادھار_i1 = i1;
                ثابت uint32_t ادھار_ir = ir;

                لكل (uint32_t cnr = 0, cr = 0; cr < nrows; cr += cnr) {
                    cnr = MIN(nrows - cr, HTP_ROPE_SPAD_بلاک);
                    ثابت uint32_t تھاں = (cur_تھاں + cr / HTP_ROPE_SPAD_بلاک) % HTP_ROPE_SPAD_NSLOTS;
                    ثابت uint32_t cur_ir   = ادھار_ir + cr;
                    ثابت uint32_t cur_i1   = ادھار_i1 + cr;

                    dma_قطار_نکالو(dma_q);
                    uint8_t * cur_spad = (uint8_t *) dma_قطار_نکالو(dma_q).dst;

                    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);
                    إذا (is_neox || is_vision) {
                        rope_neox_f32_inplace(rctx, cur_spad, cnr, ne, theta_cache);
                    } وإلا {
                        rope_basic_f32_inplace(rctx, cur_spad, cnr, theta_cache);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);

                    ثابت dma_addr_t dst_انکڑے = dst->انکڑے + i3 * nb3 + i2 * nb2 + cur_i1 * nb1;
                    dma_قطار_ڈالو(dma_q, dma_make_انکڑے(dst_انکڑے, cur_spad),
                        rctx->dst_صف_stride, rctx->src0_صف_ناپ_aligned, rctx->dst_صف_ناپ, cnr);

                    // Prefetch 2 blocks ahead into the slot just freed
                    إذا ((cr + 2 * HTP_ROPE_SPAD_بلاک) < nrows) {
                        ثابت uint32_t p_cr   = cr + 2 * HTP_ROPE_SPAD_بلاک;
                        ثابت uint32_t pnr    = MIN(nrows - p_cr, HTP_ROPE_SPAD_بلاک);
                        ثابت uint32_t p_تھاں = (cur_تھاں + p_cr / HTP_ROPE_SPAD_بلاک) % HTP_ROPE_SPAD_NSLOTS;
                        uint8_t * p_spad      = rope_spad_تھاں(src0_spad_ادھار, p_تھاں, rctx->src0_صف_ناپ_aligned);
                        ثابت dma_addr_t p_src0_انکڑے = src0->انکڑے + i3 * nb03 + i2 * nb02 + (ادھار_i1 + p_cr) * nb01;

                        dma_قطار_ڈالو(dma_q, dma_make_انکڑے(p_spad, p_src0_انکڑے),
                            rctx->src0_صف_ناپ_aligned, rctx->src0_صف_stride, rctx->src0_صف_ناپ, pnr);
                    }
                }

                ثابت uint32_t n_chunks = (nrows + HTP_ROPE_SPAD_بلاک - 1) / HTP_ROPE_SPAD_بلاک;
                cur_تھاں = (cur_تھاں + n_chunks) % HTP_ROPE_SPAD_NSLOTS;

                ir += nrows;
                i1 += nrows;
            }
        }
    }

پورا:
    dma_قطار_flush(dma_q);

    FARF(بلند_2, "rope-f32: %d/%d: (%u:%u)\n", ith, nth, src0_شروع_صف, src0_سرا_صف);
}

ساكن صحيح execute_op_rope_f32(بنية htp_ops_context * octx) {
    صحيح err = HTP_حالت_OK;

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * src2 = octx->src[2];
    ثابت بنية htp_tensor * dst  = octx->dst;

    بدل (octx->op) {
        حالة HTP_OP_ROPE:
            اكسر;

        افتراضي:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت بنية htp_rope_kernel_params * kparams = (ثابت بنية htp_rope_kernel_params *) octx->kernel_params;
    إذا (!htp_ops_context_سیٹ_n_دھاگے(octx, kparams->n_دھاگے)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }
    أكد(octx->ctx->vtcm_ناپ >= kparams->vtcm_ناپ);

    إذا (htp_tensor_is_extended(src1)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t کل_صفاں = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_انکڑے_صف_ناپ = dst->ne[0] * حجم(عائم);

    uint32_t صف_شروع = 0;
    uint32_t nrows     = کل_صفاں;

    إذا (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفاں_فی_chunk = 0;
        htp_tensor_mdev_صفاں_فی_chunk(dst, حجم(عائم), (uint32_t) dst_انکڑے_صف_ناپ, &صفاں_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(
            کل_صفاں, صفاں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_شروع = range.شروع;
        nrows     = range.گنتی;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_دھاگے = octx->n_دھاگے;

    ثابت uint32_t ne0 = dst->ne[0];
    ثابت size_t src0_صف_ناپ   = src0->ne[0] * حجم(عائم);
    ثابت size_t src0_صف_stride = src0->nb[1];
    ثابت size_t dst_صف_ناپ    = dst->ne[0] * حجم(عائم);
    ثابت size_t dst_صف_stride  = dst->nb[1];

    بنية htp_rope_context rctx;
    memset(&rctx, 0, حجم(بنية htp_rope_context));

    rctx.octx                  = octx;
    rctx.vtcm_ادھار             = (uint8_t *) octx->ctx->vtcm_ادھار;
    rctx.spad_فی_thread       = kparams->spad_فی_thread;
    rctx.theta_cache_وتھ    = kparams->theta_cache_وتھ;

    إذا (src2) {
        dma_قطار * dma_q = octx->ctx->dma[0];
        ثابت size_t ff_ناپ = src2->ne[0] * حجم(عائم);
        عائم * vtcm_freq_factors = (عائم *) (rctx.vtcm_ادھار + kparams->freq_factors_وتھ);
        dma_قطار_ڈالو(dma_q, dma_make_انکڑے(vtcm_freq_factors, src2->انکڑے),
                       kparams->freq_factors_ناپ, 0, ff_ناپ, 1);
        dma_قطار_نکالو(dma_q);
        rctx.freq_factors = vtcm_freq_factors;
    }

    ثابت int32_t * op_params = &octx->op_params[0];
    rctx.n_dims     = ((ثابت int32_t *) op_params)[1];
    rctx.منوال       = ((ثابت int32_t *) op_params)[2];
    rctx.n_ctx_orig = ((ثابت int32_t *) op_params)[4];
    rctx.n_offs     = ((ثابت int32_t *) op_params)[15];

    حافظہ_نقل(&rctx.freq_ادھار,   (int32_t *) op_params + 5,  حجم(عائم));
    حافظہ_نقل(&rctx.freq_scale,  (int32_t *) op_params + 6,  حجم(عائم));
    حافظہ_نقل(&rctx.ext_ضارب,  (int32_t *) op_params + 7,  حجم(عائم));
    حافظہ_نقل(&rctx.attn_ضارب, (int32_t *) op_params + 8,  حجم(عائم));
    حافظہ_نقل(&rctx.beta_fast,   (int32_t *) op_params + 9,  حجم(عائم));
    حافظہ_نقل(&rctx.beta_slow,   (int32_t *) op_params + 10, حجم(عائم));
    حافظہ_نقل(&rctx.sections,    (int32_t *) op_params + 11, حجم(صحيح) * 4);

    rctx.theta_scale = powf(rctx.freq_ادھار, -2.0f / rctx.n_dims);
    rctx.theta_powers[0] = 1.0f;
    لكل (صحيح j = 1; j < 32; j++) {
        rctx.theta_powers[j] = rctx.theta_powers[j - 1] * rctx.theta_scale;
    }
    rctx.theta_scale_32 = rctx.theta_powers[31] * rctx.theta_scale;

    rope_corr_dims(rctx.n_dims, rctx.n_ctx_orig, rctx.freq_ادھار, rctx.beta_fast, rctx.beta_slow, rctx.corr_dims);

    rctx.src0_صف_ناپ         = src0_صف_ناپ;
    rctx.src0_صف_stride       = src0_صف_stride;
    rctx.dst_صف_ناپ          = dst_صف_ناپ;
    rctx.dst_صف_stride        = dst_صف_stride;
    rctx.src0_صف_ناپ_aligned = kparams->src0_صف_ناپ_aligned;

    rctx.src0_nrows            = nrows;
    rctx.nrows                 = nrows;
    rctx.صف_شروع             = صف_شروع;
    rctx.src0_nrows_فی_thread = fastdiv(nrows + n_دھاگے - 1, &octx->n_دھاگے_div);
    rctx.div_ne2_ne1           = kparams->div_ne2_ne1;
    rctx.div_ne1               = kparams->div_ne1;

    FARF(بلند_2, "rope-f32 n-rows %u n-dims %d ne0 %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n", rctx.src0_nrows, rctx.n_dims, ne0,
         rctx.ext_ضارب, rctx.theta_scale, rctx.attn_ضارب);

    work_قطار_لگاتار(octx->ctx->work_قطار, rope_کم_f32, &rctx, n_دھاگے);

    أرجع err;
}

صحيح op_rope(بنية htp_ops_context * octx) {
    بدل (octx->src[0]->type) {
        حالة HTP_TYPE_F32:
            أرجع execute_op_rope_f32(octx);

        افتراضي:
            أرجع HTP_حالت_NO_SUPPORT;
    }
}
