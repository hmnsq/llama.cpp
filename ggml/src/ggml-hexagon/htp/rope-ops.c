#لارښوونه clang diagnostic ignored "-Wunused-variable"
#لارښوونه clang diagnostic ignored "-Wunused-function"
#لارښوونه clang diagnostic ignored "-Wunused-but-set-variable"

#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#ضمن <math.h>
#ضمن <string.h>
#ضمن <stdlib.h>

#ضمن "dma-queue.h"
#ضمن "hvx-utils.h"
#ضمن "hex-fastdiv.h"

#تعریف_کړه GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "rope-ops.h"

// Redefined the rope type constants as we can't include ggml.h
#تعریف_کړه HTP_ROPE_TYPE_نورمال 0
#تعریف_کړه HTP_ROPE_TYPE_NEOX   2
#تعریف_کړه HTP_ROPE_TYPE_MROPE  8
#تعریف_کړه HTP_ROPE_TYPE_VISION 24
#تعریف_کړه HTP_ROPE_TYPE_IMROPE 40

#تعریف_کړه htp_rope_preamble              \
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
    int32_t ډېرپېښ;
    int32_t n_ctx_orig;
    int32_t sections[4];

    عائم freq_بنسټ;
    عائم freq_scale;
    عائم ext_ضریب;
    عائم attn_ضریب;
    عائم beta_fast;
    عائم beta_slow;
    عائم theta_scale;
    عائم theta_scale_32;
    عائم theta_powers[32];
    عائم corr_dims[2];

    uint32_t src0_nrows_فی_thread;

    بنية htp_ops_context * octx;

    uint8_t * vtcm_بنسټ;
    size_t    spad_فی_thread;
    size_t    theta_cache_دننه_واټن;

    size_t src0_لیکه_اندازه;
    size_t src0_لیکه_stride;
    size_t dst_لیکه_اندازه;
    size_t dst_لیکه_stride;
    size_t src0_لیکه_اندازه_aligned;
    uint32_t src0_nrows;
    uint32_t لیکه_پیل;
    uint32_t nrows;

    بنية fastdiv_ارزښتونه div_ne2_ne1;
    بنية fastdiv_ارزښتونه div_ne1;

    ثابت عائم * freq_factors;
};

ساكن عائم rope_yarn_ramp(ثابت عائم ټیټ, ثابت عائم لوړ, ثابت صحيح i0) {
    ثابت عائم y = (i0 / 2 - ټیټ) / MAX(0.001f, لوړ - ټیټ);

    أرجع (1 - MIN(1, MAX(0, y)));
}

// Compute one (cos, sin) pair into cache[i0], cache[i0+1] applying YaRN scaling.
ساكن مضمن فراغ rope_yarn_one(عائم theta, عائم freq_scale, عائم * corr_dims,
                                 uint32_t i0, عائم ext_ضریب, عائم mscale,
                                 عائم * cache) {
    عائم theta_extrap = theta;

    // Get n-d rotational scaling corrected for extrapolation
    عائم theta_interp = freq_scale * theta_extrap;
    عائم theta_final  = theta_interp;
    عائم mscale_final = mscale;

    إذا (ext_ضریب != 0.0f) {
        عائم ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_ضریب;
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
                                     HVX_ویکټور v_theta,
                                     ثابت عائم * freq_factors,
                                     HVX_ویکټور v_freq_scale,
                                     HVX_ویکټور v_mscale) {
    إذا (freq_factors) {
        HVX_ویکټور v_ff = hvx_vmemu(freq_factors + i0 / 2);
        v_theta = hvx_vec_mul_f32_f32(v_theta, hvx_vec_inverse_f32(v_ff));
    }

    HVX_ویکټور v_theta_final = hvx_vec_mul_f32_f32(v_theta, v_freq_scale);
    HVX_ویکټور vcos;
    HVX_ویکټور vsin;
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

ساكن __attribute__((noinline)) فراغ rope_cache_init(ثابت عائم    theta_بنسټ,
                            ثابت عائم    freq_scale,
                            ثابت عائم *  freq_factors,
                            عائم *        corr_dims,
                            ثابت uint32_t n_cache,
                            ثابت عائم    ext_ضریب,
                            ثابت عائم    mscale,
                            عائم *        cache,
                            ثابت عائم    theta_scale,
                            ثابت عائم *  theta_powers,
                            ثابت عائم    theta_scale_32) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    إذا (ext_ضریب == 0.0f) {
        // Fast path: fully vectorized
        // We process 32 pairs (64 elements) per iteration.
        ثابت uint32_t n_بلاکونه = n_cache / 64;

        HVX_ویکټور v_theta_powers = hvx_vmemu(theta_powers);
        HVX_ویکټور v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_ویکټور v_mscale = hvx_vec_splat_f32(mscale);

        عائم theta_بلاک = theta_بنسټ;

        لكل (uint32_t b = 0; b < n_بلاکونه; b++) {
            uint32_t i0 = b * 64;
            HVX_ویکټور v_theta_بنسټ = hvx_vec_splat_f32(theta_بلاک);
            HVX_ویکټور v_theta = hvx_vec_mul_f32_f32(v_theta_بنسټ, v_theta_powers);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            theta_بلاک *= theta_scale_32;
        }

        // Leftovers
        عائم theta = theta_بلاک;
        لكل (uint32_t i0 = n_بلاکونه * 64; i0 < n_cache; i0 += 2) {
            ثابت عائم ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ضریب, mscale, cache);
            theta *= theta_scale;
        }
    } وإلا {
        عائم theta = theta_بنسټ;
        لكل (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
            ثابت عائم ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ضریب, mscale, cache);
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

ساكن مضمن HVX_ویکټور_Pred mrope_ماسک_eq1(ثابت عائم * m) {
    أرجع Q6_Q_vcmp_gt_VsfVsf(hvx_vmemu(m), Q6_V_vzero());
}

// IMROPE without wrap: theta[k] = pos[k % 3] * scale^k
ساكن مضمن HVX_ویکټور mrope_thetas_imrope_mod3(عائم pos_t, عائم pos_h, عائم pos_w,
                                                  uint32_t k0, HVX_ویکټور v_powers, عائم scale_بلاک) {
    ثابت صحيح r = (صحيح) (k0 % 3);
    ثابت عائم * mt = (r == 0) ? mrope_mod3_eq0 : (r == 1) ? mrope_mod3_eq2 : mrope_mod3_eq1;
    ثابت عائم * mh = (r == 0) ? mrope_mod3_eq1 : (r == 1) ? mrope_mod3_eq0 : mrope_mod3_eq2;

    HVX_ویکټور v = hvx_vec_splat_f32(pos_w);
    v = Q6_V_vmux_QVV(mrope_ماسک_eq1(mh), hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(mrope_ماسک_eq1(mt), hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    أرجع hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_بلاک));
}

// Contiguous MROPE without wrap: theta[k] = pos[section(k)] * scale^k
ساكن مضمن HVX_ویکټور mrope_thetas_contig(عائم pos_t, عائم pos_h, عائم pos_w, عائم pos_e,
                                             uint32_t k0, صحيح s0, صحيح sec_w, صحيح sec_e,
                                             HVX_ویکټور v_powers, عائم scale_بلاک) {
    HVX_ویکټور v_k = hvx_vec_جمع_کړه_f32_f32(hvx_vec_splat_f32((عائم) k0), hvx_vmemu(mrope_k_ramp));
    HVX_ویکټور_Pred lt_s0 = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((عائم) s0), v_k);
    HVX_ویکټور_Pred lt_sw = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((عائم) sec_w), v_k);
    HVX_ویکټور_Pred lt_se = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((عائم) sec_e), v_k);

    HVX_ویکټور v = hvx_vec_splat_f32(pos_e);
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
                             ثابت عائم    ext_ضریب,
                             ثابت عائم    mscale,
                             عائم *        cache,
                             ثابت عائم    theta_scale,
                             ثابت عائم *  theta_powers,
                             ثابت عائم    theta_scale_32) {
    ثابت صحيح sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    ثابت صحيح sec_w     = sections[0] + sections[1];
    ثابت صحيح sec_e     = sec_w + sections[2];
    ثابت uint32_t n_جوړې = n_cache / 2;

    ثابت bool no_wrap = (sect_dims > 0) && (n_جوړې <= (uint32_t) sect_dims);
    ثابت bool imrope_mod3 = is_imrope && !indep_sects && no_wrap
        && sections[0] > 0 && sections[1] > 0 && sections[2] > 0
        && n_جوړې <= (uint32_t) (3 * sections[0])
        && n_جوړې <= (uint32_t) (3 * sections[1])
        && n_جوړې <= (uint32_t) (3 * sections[2]);
    ثابت bool contig = !is_imrope && !indep_sects && no_wrap;

    إذا (ext_ضریب == 0.0f && (imrope_mod3 || contig)) {
        HVX_ویکټور v_powers     = hvx_vmemu(theta_powers);
        HVX_ویکټور v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_ویکټور v_mscale     = hvx_vec_splat_f32(mscale);
        عائم scale_بلاک = 1.0f;
        ثابت uint32_t n_بلاکونه = n_cache / 64;

        لكل (uint32_t b = 0; b < n_بلاکونه; b++) {
            ثابت uint32_t i0 = b * 64;
            ثابت uint32_t k0 = b * 32;
            HVX_ویکټور v_theta = imrope_mod3
                ? mrope_thetas_imrope_mod3(pos_t, pos_h, pos_w, k0, v_powers, scale_بلاک)
                : mrope_thetas_contig(pos_t, pos_h, pos_w, pos_e, k0, sections[0], sec_w, sec_e,
                                      v_powers, scale_بلاک);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            scale_بلاک *= theta_scale_32;
        }

        عائم theta_k = scale_بلاک;
        لكل (uint32_t k = n_بلاکونه * 32; k < n_جوړې; k++) {
            ثابت uint32_t i0 = 2 * k;
            ثابت عائم pos = mrope_pick_theta(pos_t, pos_h, pos_w, pos_e,
                                               (صحيح) k, sections, sec_w, sec_e, is_imrope);
            ثابت عائم ff = freq_factors ? freq_factors[k] : 1.0f;
            rope_yarn_one(pos * theta_k / ff, freq_scale, corr_dims, i0, ext_ضریب, mscale, cache);
            theta_k *= theta_scale;
        }
        أرجع;
    }

    عائم theta_t = pos_t;
    عائم theta_h = pos_h;
    عائم theta_w = pos_w;
    عائم theta_e = pos_e;

    ثابت bool use_hvx = (ext_ضریب == 0.0f);
    عائم __attribute__((aligned(128))) thetas[32];
    uint32_t n_thetas = 0;
    uint32_t بلاک_i0 = 0;

    HVX_ویکټور v_freq_scale = hvx_vec_splat_f32(freq_scale);
    HVX_ویکټور v_mscale     = hvx_vec_splat_f32(mscale);

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
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_ضریب, mscale, cache);
        }

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }

    لكل (uint32_t k = 0; k < n_thetas; k++) {
        ثابت uint32_t i0 = بلاک_i0 + 2 * k;
        ثابت عائم ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn_one(thetas[k] / ff, freq_scale, corr_dims, i0, ext_ضریب, mscale, cache);
    }
}

#تعریف_کړه M_PI 3.1415926535897932384626433

ساكن فراغ rope_corr_dims(صحيح     n_dims,
                           صحيح     n_ctx_orig,
                           عائم   freq_بنسټ,
                           عائم   beta_fast,
                           عائم   beta_slow,
                           عائم * dims) {
    عائم پیل = floorf(n_dims * logf(n_ctx_orig / (beta_fast * 2 * (عائم) M_PI)) / (2 * logf(freq_بنسټ)));
    عائم پای_ټکی   = ceilf(n_dims * logf(n_ctx_orig / (beta_slow * 2 * (عائم) M_PI)) / (2 * logf(freq_بنسټ)));
    dims[0]     = MAX(0, پیل);
    dims[1]     = MIN(n_dims - 1, پای_ټکی);
}

ساكن مضمن فراغ hvx_rope_neox_mul(HVX_ویکټور v0, HVX_ویکټور v1, HVX_ویکټور vcos, HVX_ویکټور vsin,
                                     HVX_ویکټور * o0, HVX_ویکټور * o1) {
    HVX_ویکټور vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, vcos);
    HVX_ویکټور vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, vsin);
    HVX_ویکټور vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, vcos);
    HVX_ویکټور vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, vsin);
    *o0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s));
    *o1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c));
}

// theta_cache full 32-pair blocks are deinterleaved [cos | sin].
ساكن مضمن فراغ hvx_rope_neox_f32_aa(عائم * محدود dst, ثابت عائم * محدود src0, uint32_t ne, ثابت عائم * محدود theta_cache) {
    ثابت uint32_t he = ne / 2;
    ثابت uint32_t nvec = he / 32;
    ثابت uint32_t nloe = he % 32;

    إذا (nloe == 0) {
        ثابت HVX_ویکټور * vs = (ثابت HVX_ویکټور *) src0;
        ثابت HVX_ویکټور * vt = (ثابت HVX_ویکټور *) theta_cache;
        HVX_ویکټور * vd = (HVX_ویکټور *) dst;
        لكل (uint32_t i = 0; i < nvec; i++) {
            HVX_ویکټور o0, o1;
            hvx_rope_neox_mul(vs[i], vs[nvec + i], vt[i * 2 + 0], vt[i * 2 + 1], &o0, &o1);
            vd[i] = o0;
            vd[nvec + i] = o1;
        }
        أرجع;
    }

    لكل (uint32_t i = 0; i < nvec; i++) {
        HVX_ویکټور o0, o1;
        hvx_rope_neox_mul(((ثابت HVX_ویکټور *) src0)[i],
                          hvx_vmemu(src0 + he + i * 32),
                          ((ثابت HVX_ویکټور *) theta_cache)[i * 2 + 0],
                          ((ثابت HVX_ویکټور *) theta_cache)[i * 2 + 1],
                          &o0, &o1);
        ((HVX_ویکټور *) dst)[i] = o0;
        hvx_vmemu(dst + he + i * 32) = o1;
    }

    HVX_ویکټور v0 = hvx_vmemu(src0 + nvec * 32);
    HVX_ویکټور v1 = hvx_vmemu(src0 + he + nvec * 32);
    HVX_ویکټور vcos = hvx_vmemu(theta_cache + nvec * 64);
    HVX_ویکټور vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);
    HVX_ویکټور o0, o1;
    hvx_rope_neox_mul(v0, v1, vcos, vsin, &o0, &o1);
    hvx_vec_store_u(dst + nvec * 32, nloe * حجم(عائم), o0);
    hvx_vec_store_u(dst + he + nvec * 32, nloe * حجم(عائم), o1);
}

ساكن مضمن فراغ hvx_rope_f32_aa(عائم * محدود dst, ثابت عائم * محدود src0, uint32_t ne, ثابت عائم * محدود theta_cache) {
    ثابت uint32_t nvec = ne / 64;
    ثابت uint32_t nloe = ne % 64;

    لكل (uint32_t i = 0; i < nvec; i++) {
        HVX_ویکټور v0 = ((ثابت HVX_ویکټور *) src0)[i * 2 + 0];
        HVX_ویکټور v1 = ((ثابت HVX_ویکټور *) src0)[i * 2 + 1];

        HVX_ویکټور vcos = ((ثابت HVX_ویکټور *) theta_cache)[i * 2 + 0];
        HVX_ویکټور vsin = ((ثابت HVX_ویکټور *) theta_cache)[i * 2 + 1];

        HVX_ویکټور_Pair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

        HVX_ویکټور vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
        HVX_ویکټور vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
        HVX_ویکټور vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
        HVX_ویکټور vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

        HVX_ویکټور v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_ویکټور v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        HVX_ویکټور_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

        ((HVX_ویکټور *) dst)[i * 2 + 0] = Q6_V_lo_W(vstore);
        ((HVX_ویکټور *) dst)[i * 2 + 1] = Q6_V_hi_W(vstore);
    }

    إذا (nloe > 0) {
        إذا (nloe <= 32) {
            HVX_ویکټور v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_ویکټور vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_ویکټور vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_ویکټور_Pair vx0_x1 = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, -4);

            HVX_ویکټور vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_ویکټور vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_ویکټور vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_ویکټور vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_ویکټور v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_ویکټور v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_ویکټور_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            hvx_vec_store_u(dst + nvec * 64, nloe * حجم(عائم), Q6_V_lo_W(vstore));
        } وإلا {
            HVX_ویکټور v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_ویکټور v1 = hvx_vmemu(src0 + nvec * 64 + 32);

            HVX_ویکټور vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_ویکټور vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_ویکټور_Pair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

            HVX_ویکټور vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_ویکټور vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_ویکټور vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_ویکټور vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_ویکټور v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_ویکټور v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_ویکټور_Pair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            ((HVX_ویکټور *) dst)[nvec * 2 + 0] = Q6_V_lo_W(vstore);
            hvx_vec_store_u(dst + nvec * 64 + 32, (nloe - 32) * حجم(عائم), Q6_V_hi_W(vstore));
        }
    }
}

ساكن فراغ مضمن rope_basic_f32_inplace(بنية htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, ثابت عائم * محدود theta_cache) {
    ثابت uint32_t n_offs = rctx->n_offs;
    #لارښوونه unroll(4)
    لكل (uint32_t i = 0; i < nr; i++) {
        عائم * s = (عائم *) (src + i * rctx->src0_لیکه_اندازه_aligned);
        hvx_rope_f32_aa(s + n_offs, s + n_offs, rctx->n_dims, theta_cache);
    }
}

ساكن فراغ مضمن rope_neox_f32_inplace(بنية htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, uint32_t ne, ثابت عائم * محدود theta_cache) {
    ثابت uint32_t n_offs = rctx->n_offs;
    #لارښوونه unroll(4)
    لكل (uint32_t i = 0; i < nr; i++) {
        عائم * s = (عائم *) (src + i * rctx->src0_لیکه_اندازه_aligned);
        hvx_rope_neox_f32_aa(s + n_offs, s + n_offs, ne, theta_cache);
    }
}

ساكن فراغ rope_دنده_f32(بینښه صحيح nth, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_rope_context * rctx = (بنية htp_rope_context *) معلومات;
    بنية htp_ops_context * octx = rctx->octx;

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * src1 = octx->src[1];
    ثابت بنية htp_tensor * src2 = octx->src[2];
    ثابت بنية htp_tensor * dst  = octx->dst;

    htp_rope_preamble;

    ثابت uint32_t src0_nrows = rctx->nrows;
    ثابت uint32_t src0_nrows_فی_thread = rctx->src0_nrows_فی_thread;

    ثابت uint32_t src0_پیل_لیکه = rctx->لیکه_پیل + src0_nrows_فی_thread * ith;
    ثابت uint32_t src0_پای_ټکی_لیکه   = MIN(src0_پیل_لیکه + src0_nrows_فی_thread, rctx->لیکه_پیل + src0_nrows);

    // no work for this thread
    إذا (src0_پیل_لیکه >= src0_پای_ټکی_لیکه) {
        أرجع;
    }

    ثابت int32_t ډېرپېښ    = rctx->ډېرپېښ;
    // MROPE, IMROPE and VISION use NEOX-style pairing for the rotation
    ثابت bool    is_neox = (ډېرپېښ & HTP_ROPE_TYPE_NEOX) || (ډېرپېښ & HTP_ROPE_TYPE_MROPE);
    ثابت bool    is_vision = (ډېرپېښ == HTP_ROPE_TYPE_VISION);

    // VTCM setup
    uint8_t * src0_spad_بنسټ = rctx->vtcm_بنسټ + (ith * rctx->spad_فی_thread);
    عائم *   theta_cache    = (عائم *) (src0_spad_بنسټ);
              src0_spad_بنسټ = src0_spad_بنسټ + rctx->theta_cache_دننه_واټن;

    dma_کتار * dma_q = octx->ctx->dma[ith];
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];
    ثابت int32_t * pos = (ثابت int32_t *) (uintptr_t) src1->معلومات;
    ثابت عائم * freq_factors = rctx->freq_factors;

    ثابت uint32_t i3_پیل = fastdiv(src0_پیل_لیکه, &rctx->div_ne2_ne1);
    ثابت uint32_t rem      = fastmodulo(src0_پیل_لیکه, ne2 * ne1, &rctx->div_ne2_ne1);
    ثابت uint32_t i2_پیل = fastdiv(rem, &rctx->div_ne1);
    ثابت uint32_t i1_پیل = fastmodulo(rem, ne1, &rctx->div_ne1);

    uint32_t ir = src0_پیل_لیکه;
    uint32_t prev_i2 = (uint32_t) -1;
    uint32_t cur_خونه = 0;

    لكل (uint32_t i3 = i3_پیل; i3 < ne3; i3++) { // batch
        ثابت uint32_t i2_init = (i3 == i3_پیل) ? i2_پیل : 0;
        لكل (uint32_t i2 = i2_init; i2 < ne2; i2++) { // seq-len
            ثابت uint32_t i1_init = (i3 == i3_پیل && i2 == i2_پیل) ? i1_پیل : 0;
            لكل (uint32_t i1 = i1_init; i1 < ne1; ) { // attn-heads
                إذا (ir >= src0_پای_ټکی_لیکه) اذهب بشپړ;

                // Rows in this block
                ثابت uint32_t nrows = MIN(src0_پای_ټکی_لیکه - ir, ne1 - i1);

                // Depth before prefetch
                ثابت uint32_t dma_ژوروالی = dma_کتار_ژوروالی(dma_q);

                // Prefetch up to 2 blocks
                ثابت uint32_t p_nrows = MIN(nrows, 2 * HTP_ROPE_SPAD_بلاک);
                لكل (uint32_t pr = 0; pr < p_nrows; pr += HTP_ROPE_SPAD_بلاک) {
                    ثابت uint32_t pnr = MIN(nrows - pr, HTP_ROPE_SPAD_بلاک);
                    ثابت uint32_t خونه = (cur_خونه + pr / HTP_ROPE_SPAD_بلاک) % HTP_ROPE_SPAD_NSLOTS;
                    uint8_t * spad_خونه = rope_spad_خونه(src0_spad_بنسټ, خونه, rctx->src0_لیکه_اندازه_aligned);
                    ثابت dma_addr_t src0_معلومات = src0->معلومات + i3 * nb03 + i2 * nb02 + (i1 + pr) * nb01;

                    // Dummy DMA transaction for sequencing (interleaving wr, rd, wr, rd, ...)
                    dma_کتار_ټېل(dma_q, dma_make_معلومات(dst->معلومات, spad_خونه), 0, 0, 0, 0);

                    dma_کتار_ټېل(dma_q, dma_make_معلومات(spad_خونه, src0_معلومات),
                        rctx->src0_لیکه_اندازه_aligned, rctx->src0_لیکه_stride, rctx->src0_لیکه_اندازه, pnr);
                }

                // Update theta cache
                إذا (i2 != prev_i2) {
                    prev_i2 = i2;

                    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                    // VISION rotates the full row; other modes only rotate n_dims.
                    ثابت uint32_t n_cache = is_vision ? ne0 : (uint32_t) rctx->n_dims;
                    ثابت bool is_mrope = (rctx->ډېرپېښ & HTP_ROPE_TYPE_MROPE) != 0;
                    إذا (is_mrope) {
                        // src1 holds four position arrays stacked along ne0:
                        // pos[i2], pos[i2+ne2], pos[i2+ne2*2], pos[i2+ne2*3]
                        ثابت bool is_imrope = (rctx->ډېرپېښ == HTP_ROPE_TYPE_IMROPE);
                        mrope_cache_init(
                            (عائم) pos[i2],
                            (عائم) pos[i2 + ne2],
                            (عائم) pos[i2 + ne2 * 2],
                            (عائم) pos[i2 + ne2 * 3],
                            rctx->sections, is_imrope, is_vision,
                            rctx->freq_scale, freq_factors, rctx->corr_dims,
                            n_cache, rctx->ext_ضریب, rctx->attn_ضریب,
                            theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    } وإلا {
                       rope_cache_init(pos[i2], rctx->freq_scale, freq_factors, rctx->corr_dims,
                                        n_cache, rctx->ext_ضریب, rctx->attn_ضریب,
                                        theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                }

                // Skip output DMA transactions from prev block (if any)
                لكل (uint32_t d = 0; d < dma_ژوروالی; d++) { dma_کتار_راوباسه_nowait(dma_q); }

                // Compute loop
                ثابت uint32_t ne = is_vision ? ne0 : rctx->n_dims;
                ثابت uint32_t بنسټ_i1 = i1;
                ثابت uint32_t بنسټ_ir = ir;

                لكل (uint32_t cnr = 0, cr = 0; cr < nrows; cr += cnr) {
                    cnr = MIN(nrows - cr, HTP_ROPE_SPAD_بلاک);
                    ثابت uint32_t خونه = (cur_خونه + cr / HTP_ROPE_SPAD_بلاک) % HTP_ROPE_SPAD_NSLOTS;
                    ثابت uint32_t cur_ir   = بنسټ_ir + cr;
                    ثابت uint32_t cur_i1   = بنسټ_i1 + cr;

                    dma_کتار_راوباسه(dma_q);
                    uint8_t * cur_spad = (uint8_t *) dma_کتار_راوباسه(dma_q).dst;

                    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);
                    إذا (is_neox || is_vision) {
                        rope_neox_f32_inplace(rctx, cur_spad, cnr, ne, theta_cache);
                    } وإلا {
                        rope_basic_f32_inplace(rctx, cur_spad, cnr, theta_cache);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);

                    ثابت dma_addr_t dst_معلومات = dst->معلومات + i3 * nb3 + i2 * nb2 + cur_i1 * nb1;
                    dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_معلومات, cur_spad),
                        rctx->dst_لیکه_stride, rctx->src0_لیکه_اندازه_aligned, rctx->dst_لیکه_اندازه, cnr);

                    // Prefetch 2 blocks ahead into the slot just freed
                    إذا ((cr + 2 * HTP_ROPE_SPAD_بلاک) < nrows) {
                        ثابت uint32_t p_cr   = cr + 2 * HTP_ROPE_SPAD_بلاک;
                        ثابت uint32_t pnr    = MIN(nrows - p_cr, HTP_ROPE_SPAD_بلاک);
                        ثابت uint32_t p_خونه = (cur_خونه + p_cr / HTP_ROPE_SPAD_بلاک) % HTP_ROPE_SPAD_NSLOTS;
                        uint8_t * p_spad      = rope_spad_خونه(src0_spad_بنسټ, p_خونه, rctx->src0_لیکه_اندازه_aligned);
                        ثابت dma_addr_t p_src0_معلومات = src0->معلومات + i3 * nb03 + i2 * nb02 + (بنسټ_i1 + p_cr) * nb01;

                        dma_کتار_ټېل(dma_q, dma_make_معلومات(p_spad, p_src0_معلومات),
                            rctx->src0_لیکه_اندازه_aligned, rctx->src0_لیکه_stride, rctx->src0_لیکه_اندازه, pnr);
                    }
                }

                ثابت uint32_t n_chunks = (nrows + HTP_ROPE_SPAD_بلاک - 1) / HTP_ROPE_SPAD_بلاک;
                cur_خونه = (cur_خونه + n_chunks) % HTP_ROPE_SPAD_NSLOTS;

                ir += nrows;
                i1 += nrows;
            }
        }
    }

بشپړ:
    dma_کتار_flush(dma_q);

    FARF(لوړ_2, "rope-f32: %d/%d: (%u:%u)\n", ith, nth, src0_پیل_لیکه, src0_پای_ټکی_لیکه);
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
    إذا (!htp_ops_context_سېټ_n_تارونه(octx, kparams->n_تارونه)) {
        أرجع HTP_حالت_INVAL_PARAMS;
    }
    أكد(octx->ctx->vtcm_اندازه >= kparams->vtcm_اندازه);

    إذا (htp_tensor_is_extended(src1)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t ټول_لیکې = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_معلومات_لیکه_اندازه = dst->ne[0] * حجم(عائم);

    uint32_t لیکه_پیل = 0;
    uint32_t nrows     = ټول_لیکې;

    إذا (octx->ctx->mdev.شمېر > 1) {
        uint32_t لیکې_فی_chunk = 0;
        htp_tensor_mdev_لیکې_فی_chunk(dst, حجم(عائم), (uint32_t) dst_معلومات_لیکه_اندازه, &لیکې_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(
            ټول_لیکې, لیکې_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        لیکه_پیل = range.پیل;
        nrows     = range.شمېر;
    }

    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    ثابت uint32_t n_تارونه = octx->n_تارونه;

    ثابت uint32_t ne0 = dst->ne[0];
    ثابت size_t src0_لیکه_اندازه   = src0->ne[0] * حجم(عائم);
    ثابت size_t src0_لیکه_stride = src0->nb[1];
    ثابت size_t dst_لیکه_اندازه    = dst->ne[0] * حجم(عائم);
    ثابت size_t dst_لیکه_stride  = dst->nb[1];

    بنية htp_rope_context rctx;
    memset(&rctx, 0, حجم(بنية htp_rope_context));

    rctx.octx                  = octx;
    rctx.vtcm_بنسټ             = (uint8_t *) octx->ctx->vtcm_بنسټ;
    rctx.spad_فی_thread       = kparams->spad_فی_thread;
    rctx.theta_cache_دننه_واټن    = kparams->theta_cache_دننه_واټن;

    إذا (src2) {
        dma_کتار * dma_q = octx->ctx->dma[0];
        ثابت size_t ff_اندازه = src2->ne[0] * حجم(عائم);
        عائم * vtcm_freq_factors = (عائم *) (rctx.vtcm_بنسټ + kparams->freq_factors_دننه_واټن);
        dma_کتار_ټېل(dma_q, dma_make_معلومات(vtcm_freq_factors, src2->معلومات),
                       kparams->freq_factors_اندازه, 0, ff_اندازه, 1);
        dma_کتار_راوباسه(dma_q);
        rctx.freq_factors = vtcm_freq_factors;
    }

    ثابت int32_t * op_params = &octx->op_params[0];
    rctx.n_dims     = ((ثابت int32_t *) op_params)[1];
    rctx.ډېرپېښ       = ((ثابت int32_t *) op_params)[2];
    rctx.n_ctx_orig = ((ثابت int32_t *) op_params)[4];
    rctx.n_offs     = ((ثابت int32_t *) op_params)[15];

    حافظه_نقل(&rctx.freq_بنسټ,   (int32_t *) op_params + 5,  حجم(عائم));
    حافظه_نقل(&rctx.freq_scale,  (int32_t *) op_params + 6,  حجم(عائم));
    حافظه_نقل(&rctx.ext_ضریب,  (int32_t *) op_params + 7,  حجم(عائم));
    حافظه_نقل(&rctx.attn_ضریب, (int32_t *) op_params + 8,  حجم(عائم));
    حافظه_نقل(&rctx.beta_fast,   (int32_t *) op_params + 9,  حجم(عائم));
    حافظه_نقل(&rctx.beta_slow,   (int32_t *) op_params + 10, حجم(عائم));
    حافظه_نقل(&rctx.sections,    (int32_t *) op_params + 11, حجم(صحيح) * 4);

    rctx.theta_scale = powf(rctx.freq_بنسټ, -2.0f / rctx.n_dims);
    rctx.theta_powers[0] = 1.0f;
    لكل (صحيح j = 1; j < 32; j++) {
        rctx.theta_powers[j] = rctx.theta_powers[j - 1] * rctx.theta_scale;
    }
    rctx.theta_scale_32 = rctx.theta_powers[31] * rctx.theta_scale;

    rope_corr_dims(rctx.n_dims, rctx.n_ctx_orig, rctx.freq_بنسټ, rctx.beta_fast, rctx.beta_slow, rctx.corr_dims);

    rctx.src0_لیکه_اندازه         = src0_لیکه_اندازه;
    rctx.src0_لیکه_stride       = src0_لیکه_stride;
    rctx.dst_لیکه_اندازه          = dst_لیکه_اندازه;
    rctx.dst_لیکه_stride        = dst_لیکه_stride;
    rctx.src0_لیکه_اندازه_aligned = kparams->src0_لیکه_اندازه_aligned;

    rctx.src0_nrows            = nrows;
    rctx.nrows                 = nrows;
    rctx.لیکه_پیل             = لیکه_پیل;
    rctx.src0_nrows_فی_thread = fastdiv(nrows + n_تارونه - 1, &octx->n_تارونه_div);
    rctx.div_ne2_ne1           = kparams->div_ne2_ne1;
    rctx.div_ne1               = kparams->div_ne1;

    FARF(لوړ_2, "rope-f32 n-rows %u n-dims %d ne0 %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n", rctx.src0_nrows, rctx.n_dims, ne0,
         rctx.ext_ضریب, rctx.theta_scale, rctx.attn_ضریب);

    work_کتار_پرلپسې(octx->ctx->work_کتار, rope_دنده_f32, &rctx, n_تارونه);

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
