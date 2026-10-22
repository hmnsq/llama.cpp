#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#شامل <math.h>
#شامل <string.h>
#شامل <stdlib.h>

#شامل "dma-queue.h"
#شامل "hvx-utils.h"
#شامل "hex-fastdiv.h"

#define GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "rope-ops.h"

// Redefined the rope type constants as we can't include ggml.h
#define HTP_ROPE_TYPE_NORMAL 0
#define HTP_ROPE_TYPE_NEOX   2
#define HTP_ROPE_TYPE_MROPE  8
#define HTP_ROPE_TYPE_VISION 24
#define HTP_ROPE_TYPE_IMROPE 40

#define htp_rope_preamble              \
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

ساختار htp_rope_context {
    int32_t n_dims;
    int32_t n_offs;
    int32_t mode;
    int32_t n_ctx_orig;
    int32_t sections[4];

    اعشاری freq_base;
    اعشاری freq_scale;
    اعشاری ext_factor;
    اعشاری attn_factor;
    اعشاری beta_fast;
    اعشاری beta_slow;
    اعشاری theta_scale;
    اعشاری theta_scale_32;
    اعشاری theta_powers[32];
    اعشاری corr_dims[2];

    uint32_t src0_nrows_per_thread;

    ساختار htp_ops_context * octx;

    uint8_t * vtcm_base;
    size_t    spad_per_thread;
    size_t    theta_cache_offset;

    size_t src0_row_size;
    size_t src0_row_stride;
    size_t dst_row_size;
    size_t dst_row_stride;
    size_t src0_row_size_aligned;
    uint32_t src0_nrows;
    uint32_t row_start;
    uint32_t nrows;

    ساختار fastdiv_مقادیر div_ne2_ne1;
    ساختار fastdiv_مقادیر div_ne1;

    ثابت اعشاری * freq_factors;
};

ایستا اعشاری rope_yarn_ramp(ثابت اعشاری کم, ثابت اعشاری بالا, ثابت صحیح i0) {
    ثابت اعشاری y = (i0 / 2 - کم) / MAX(0.001f, بالا - کم);

    بازگشت (1 - MIN(1, MAX(0, y)));
}

// Compute one (cos, sin) pair into cache[i0], cache[i0+1] applying YaRN scaling.
ایستا درونخطی تهی rope_yarn_one(اعشاری theta, اعشاری freq_scale, اعشاری * corr_dims,
                                 uint32_t i0, اعشاری ext_factor, اعشاری mscale,
                                 اعشاری * cache) {
    اعشاری theta_extrap = theta;

    // Get n-d rotational scaling corrected for extrapolation
    اعشاری theta_interp = freq_scale * theta_extrap;
    اعشاری theta_final  = theta_interp;
    اعشاری mscale_final = mscale;

    اگر (ext_factor != 0.0f) {
        اعشاری ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
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
ایستا درونخطی تهی rope_cache_hvx_32(اعشاری * cache, uint32_t i0,
                                     HVX_Vector v_theta,
                                     ثابت اعشاری * freq_factors,
                                     HVX_Vector v_freq_scale,
                                     HVX_Vector v_mscale) {
    اگر (freq_factors) {
        HVX_Vector v_ff = hvx_vmemu(freq_factors + i0 / 2);
        v_theta = hvx_vec_mul_f32_f32(v_theta, hvx_vec_inverse_f32(v_ff));
    }

    HVX_Vector v_theta_final = hvx_vec_mul_f32_f32(v_theta, v_freq_scale);
    HVX_Vector vcos;
    HVX_Vector vsin;
    hvx_vec_sincos_f32(v_theta_final, &vcos, &vsin);
    vcos = hvx_vec_mul_f32_f32(vcos, v_mscale);
    vsin = hvx_vec_mul_f32_f32(vsin, v_mscale);

    اگر (((uintptr_t) (cache + i0)) % 128 == 0) {
        hvx_vmem(cache + i0 + 0)  = vcos;
        hvx_vmem(cache + i0 + 32) = vsin;
    } وگرنه {
        hvx_vec_store_u(cache + i0 + 0,  32 * اندازه(اعشاری), vcos);
        hvx_vec_store_u(cache + i0 + 32, 32 * اندازه(اعشاری), vsin);
    }
}

ایستا __attribute__((noinline)) تهی rope_cache_init(ثابت اعشاری    theta_base,
                            ثابت اعشاری    freq_scale,
                            ثابت اعشاری *  freq_factors,
                            اعشاری *        corr_dims,
                            ثابت uint32_t n_cache,
                            ثابت اعشاری    ext_factor,
                            ثابت اعشاری    mscale,
                            اعشاری *        cache,
                            ثابت اعشاری    theta_scale,
                            ثابت اعشاری *  theta_powers,
                            ثابت اعشاری    theta_scale_32) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    اگر (ext_factor == 0.0f) {
        // Fast path: fully vectorized
        // We process 32 pairs (64 elements) per iteration.
        ثابت uint32_t n_blocks = n_cache / 64;

        HVX_Vector v_theta_powers = hvx_vmemu(theta_powers);
        HVX_Vector v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_Vector v_mscale = hvx_vec_splat_f32(mscale);

        اعشاری theta_block = theta_base;

        برای (uint32_t b = 0; b < n_blocks; b++) {
            uint32_t i0 = b * 64;
            HVX_Vector v_theta_base = hvx_vec_splat_f32(theta_block);
            HVX_Vector v_theta = hvx_vec_mul_f32_f32(v_theta_base, v_theta_powers);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            theta_block *= theta_scale_32;
        }

        // Leftovers
        اعشاری theta = theta_block;
        برای (uint32_t i0 = n_blocks * 64; i0 < n_cache; i0 += 2) {
            ثابت اعشاری ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta *= theta_scale;
        }
    } وگرنه {
        اعشاری theta = theta_base;
        برای (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
            ثابت اعشاری ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta *= theta_scale;
        }
    }
}

ایستا درونخطی اعشاری mrope_pick_theta(اعشاری theta_t, اعشاری theta_h, اعشاری theta_w, اعشاری theta_e,
                                     صحیح sector, ثابت int32_t sections[4], صحیح sec_w, صحیح sec_e,
                                     bool is_imrope) {
    اگر (is_imrope) {
        اگر      (sector % 3 == 0 && sector < 3 * sections[0]) { بازگشت theta_t; }
        وگرنه اگر (sector % 3 == 1 && sector < 3 * sections[1]) { بازگشت theta_h; }
        وگرنه اگر (sector % 3 == 2 && sector < 3 * sections[2]) { بازگشت theta_w; }
        وگرنه                                                   { بازگشت theta_e; }
    }
    اگر      (sector < sections[0]) { بازگشت theta_t; }
    وگرنه اگر (sector < sec_w)       { بازگشت theta_h; }
    وگرنه اگر (sector < sec_e)       { بازگشت theta_w; }
    وگرنه                           { بازگشت theta_e; }
}

// lane j is 1 when (j % 3) == rem
ایستا ثابت اعشاری __attribute__((aligned(128))) mrope_mod3_eq0[32] = {
    1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0
};
ایستا ثابت اعشاری __attribute__((aligned(128))) mrope_mod3_eq1[32] = {
    0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1
};
ایستا ثابت اعشاری __attribute__((aligned(128))) mrope_mod3_eq2[32] = {
    0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0
};

ایستا ثابت اعشاری __attribute__((aligned(128))) mrope_k_ramp[32] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
};

ایستا درونخطی HVX_VectorPred mrope_mask_eq1(ثابت اعشاری * m) {
    بازگشت Q6_Q_vcmp_gt_VsfVsf(hvx_vmemu(m), Q6_V_vzero());
}

// IMROPE without wrap: theta[k] = pos[k % 3] * scale^k
ایستا درونخطی HVX_Vector mrope_thetas_imrope_mod3(اعشاری pos_t, اعشاری pos_h, اعشاری pos_w,
                                                  uint32_t k0, HVX_Vector v_powers, اعشاری scale_block) {
    ثابت صحیح r = (صحیح) (k0 % 3);
    ثابت اعشاری * mt = (r == 0) ? mrope_mod3_eq0 : (r == 1) ? mrope_mod3_eq2 : mrope_mod3_eq1;
    ثابت اعشاری * mh = (r == 0) ? mrope_mod3_eq1 : (r == 1) ? mrope_mod3_eq0 : mrope_mod3_eq2;

    HVX_Vector v = hvx_vec_splat_f32(pos_w);
    v = Q6_V_vmux_QVV(mrope_mask_eq1(mh), hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(mrope_mask_eq1(mt), hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    بازگشت hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_block));
}

// Contiguous MROPE without wrap: theta[k] = pos[section(k)] * scale^k
ایستا درونخطی HVX_Vector mrope_thetas_contig(اعشاری pos_t, اعشاری pos_h, اعشاری pos_w, اعشاری pos_e,
                                             uint32_t k0, صحیح s0, صحیح sec_w, صحیح sec_e,
                                             HVX_Vector v_powers, اعشاری scale_block) {
    HVX_Vector v_k = hvx_vec_add_f32_f32(hvx_vec_splat_f32((اعشاری) k0), hvx_vmemu(mrope_k_ramp));
    HVX_VectorPred lt_s0 = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((اعشاری) s0), v_k);
    HVX_VectorPred lt_sw = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((اعشاری) sec_w), v_k);
    HVX_VectorPred lt_se = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((اعشاری) sec_e), v_k);

    HVX_Vector v = hvx_vec_splat_f32(pos_e);
    v = Q6_V_vmux_QVV(lt_se, hvx_vec_splat_f32(pos_w), v);
    v = Q6_V_vmux_QVV(lt_sw, hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(lt_s0, hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    بازگشت hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_block));
}

// pos_t/h/w/e: the four position ids for this sequence step (t=time, h=height, w=width, e=extra).
// sections[4]: number of head dims assigned to each position component.
ایستا __attribute__((noinline)) تهی mrope_cache_init(ثابت اعشاری    pos_t,
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
                             ثابت اعشاری    ext_factor,
                             ثابت اعشاری    mscale,
                             اعشاری *        cache,
                             ثابت اعشاری    theta_scale,
                             ثابت اعشاری *  theta_powers,
                             ثابت اعشاری    theta_scale_32) {
    ثابت صحیح sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    ثابت صحیح sec_w     = sections[0] + sections[1];
    ثابت صحیح sec_e     = sec_w + sections[2];
    ثابت uint32_t n_pairs = n_cache / 2;

    ثابت bool no_wrap = (sect_dims > 0) && (n_pairs <= (uint32_t) sect_dims);
    ثابت bool imrope_mod3 = is_imrope && !indep_sects && no_wrap
        && sections[0] > 0 && sections[1] > 0 && sections[2] > 0
        && n_pairs <= (uint32_t) (3 * sections[0])
        && n_pairs <= (uint32_t) (3 * sections[1])
        && n_pairs <= (uint32_t) (3 * sections[2]);
    ثابت bool contig = !is_imrope && !indep_sects && no_wrap;

    اگر (ext_factor == 0.0f && (imrope_mod3 || contig)) {
        HVX_Vector v_powers     = hvx_vmemu(theta_powers);
        HVX_Vector v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_Vector v_mscale     = hvx_vec_splat_f32(mscale);
        اعشاری scale_block = 1.0f;
        ثابت uint32_t n_blocks = n_cache / 64;

        برای (uint32_t b = 0; b < n_blocks; b++) {
            ثابت uint32_t i0 = b * 64;
            ثابت uint32_t k0 = b * 32;
            HVX_Vector v_theta = imrope_mod3
                ? mrope_thetas_imrope_mod3(pos_t, pos_h, pos_w, k0, v_powers, scale_block)
                : mrope_thetas_contig(pos_t, pos_h, pos_w, pos_e, k0, sections[0], sec_w, sec_e,
                                      v_powers, scale_block);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            scale_block *= theta_scale_32;
        }

        اعشاری theta_k = scale_block;
        برای (uint32_t k = n_blocks * 32; k < n_pairs; k++) {
            ثابت uint32_t i0 = 2 * k;
            ثابت اعشاری pos = mrope_pick_theta(pos_t, pos_h, pos_w, pos_e,
                                               (صحیح) k, sections, sec_w, sec_e, is_imrope);
            ثابت اعشاری ff = freq_factors ? freq_factors[k] : 1.0f;
            rope_yarn_one(pos * theta_k / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta_k *= theta_scale;
        }
        بازگشت;
    }

    اعشاری theta_t = pos_t;
    اعشاری theta_h = pos_h;
    اعشاری theta_w = pos_w;
    اعشاری theta_e = pos_e;

    ثابت bool use_hvx = (ext_factor == 0.0f);
    اعشاری __attribute__((aligned(128))) thetas[32];
    uint32_t n_thetas = 0;
    uint32_t block_i0 = 0;

    HVX_Vector v_freq_scale = hvx_vec_splat_f32(freq_scale);
    HVX_Vector v_mscale     = hvx_vec_splat_f32(mscale);

    برای (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
        ثابت صحیح sector = (i0 / 2) % sect_dims;

        اگر (indep_sects) {
            // Reset theta when crossing into a new section.
            اگر      (sector == 0)           { theta_t = pos_t; }
            وگرنه اگر (sector == sections[0]) { theta_h = pos_h; }
            وگرنه اگر (sector == sec_w)       { theta_w = pos_w; }
            وگرنه اگر (sector == sec_e)       { theta_e = pos_e; }
        }

        ثابت اعشاری theta = mrope_pick_theta(theta_t, theta_h, theta_w, theta_e,
                                             sector, sections, sec_w, sec_e, is_imrope);

        اگر (use_hvx) {
            اگر (n_thetas == 0) {
                block_i0 = i0;
            }
            thetas[n_thetas++] = theta;
            اگر (n_thetas == 32) {
                rope_cache_hvx_32(cache, block_i0, hvx_vmemu(thetas), freq_factors, v_freq_scale, v_mscale);
                n_thetas = 0;
            }
        } وگرنه {
            ثابت اعشاری ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
        }

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }

    برای (uint32_t k = 0; k < n_thetas; k++) {
        ثابت uint32_t i0 = block_i0 + 2 * k;
        ثابت اعشاری ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn_one(thetas[k] / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
    }
}

#define M_PI 3.1415926535897932384626433

ایستا تهی rope_corr_dims(صحیح     n_dims,
                           صحیح     n_ctx_orig,
                           اعشاری   freq_base,
                           اعشاری   beta_fast,
                           اعشاری   beta_slow,
                           اعشاری * dims) {
    اعشاری start = floorf(n_dims * logf(n_ctx_orig / (beta_fast * 2 * (اعشاری) M_PI)) / (2 * logf(freq_base)));
    اعشاری end   = ceilf(n_dims * logf(n_ctx_orig / (beta_slow * 2 * (اعشاری) M_PI)) / (2 * logf(freq_base)));
    dims[0]     = MAX(0, start);
    dims[1]     = MIN(n_dims - 1, end);
}

ایستا درونخطی تهی hvx_rope_neox_mul(HVX_Vector v0, HVX_Vector v1, HVX_Vector vcos, HVX_Vector vsin,
                                     HVX_Vector * o0, HVX_Vector * o1) {
    HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, vcos);
    HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, vsin);
    HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, vcos);
    HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, vsin);
    *o0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s));
    *o1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c));
}

// theta_cache full 32-pair blocks are deinterleaved [cos | sin].
ایستا درونخطی تهی hvx_rope_neox_f32_aa(اعشاری * restrict dst, ثابت اعشاری * restrict src0, uint32_t ne, ثابت اعشاری * restrict theta_cache) {
    ثابت uint32_t he = ne / 2;
    ثابت uint32_t nvec = he / 32;
    ثابت uint32_t nloe = he % 32;

    اگر (nloe == 0) {
        ثابت HVX_Vector * vs = (ثابت HVX_Vector *) src0;
        ثابت HVX_Vector * vt = (ثابت HVX_Vector *) theta_cache;
        HVX_Vector * vd = (HVX_Vector *) dst;
        برای (uint32_t i = 0; i < nvec; i++) {
            HVX_Vector o0, o1;
            hvx_rope_neox_mul(vs[i], vs[nvec + i], vt[i * 2 + 0], vt[i * 2 + 1], &o0, &o1);
            vd[i] = o0;
            vd[nvec + i] = o1;
        }
        بازگشت;
    }

    برای (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector o0, o1;
        hvx_rope_neox_mul(((ثابت HVX_Vector *) src0)[i],
                          hvx_vmemu(src0 + he + i * 32),
                          ((ثابت HVX_Vector *) theta_cache)[i * 2 + 0],
                          ((ثابت HVX_Vector *) theta_cache)[i * 2 + 1],
                          &o0, &o1);
        ((HVX_Vector *) dst)[i] = o0;
        hvx_vmemu(dst + he + i * 32) = o1;
    }

    HVX_Vector v0 = hvx_vmemu(src0 + nvec * 32);
    HVX_Vector v1 = hvx_vmemu(src0 + he + nvec * 32);
    HVX_Vector vcos = hvx_vmemu(theta_cache + nvec * 64);
    HVX_Vector vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);
    HVX_Vector o0, o1;
    hvx_rope_neox_mul(v0, v1, vcos, vsin, &o0, &o1);
    hvx_vec_store_u(dst + nvec * 32, nloe * اندازه(اعشاری), o0);
    hvx_vec_store_u(dst + he + nvec * 32, nloe * اندازه(اعشاری), o1);
}

ایستا درونخطی تهی hvx_rope_f32_aa(اعشاری * restrict dst, ثابت اعشاری * restrict src0, uint32_t ne, ثابت اعشاری * restrict theta_cache) {
    ثابت uint32_t nvec = ne / 64;
    ثابت uint32_t nloe = ne % 64;

    برای (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v0 = ((ثابت HVX_Vector *) src0)[i * 2 + 0];
        HVX_Vector v1 = ((ثابت HVX_Vector *) src0)[i * 2 + 1];

        HVX_Vector vcos = ((ثابت HVX_Vector *) theta_cache)[i * 2 + 0];
        HVX_Vector vsin = ((ثابت HVX_Vector *) theta_cache)[i * 2 + 1];

        HVX_VectorPair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

        HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
        HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
        HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
        HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

        HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
        HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

        HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

        ((HVX_Vector *) dst)[i * 2 + 0] = Q6_V_lo_W(vstore);
        ((HVX_Vector *) dst)[i * 2 + 1] = Q6_V_hi_W(vstore);
    }

    اگر (nloe > 0) {
        اگر (nloe <= 32) {
            HVX_Vector v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_Vector vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_Vector vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_VectorPair vx0_x1 = Q6_W_vdeal_VVR(Q6_V_vzero(), v0, -4);

            HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            hvx_vec_store_u(dst + nvec * 64, nloe * اندازه(اعشاری), Q6_V_lo_W(vstore));
        } وگرنه {
            HVX_Vector v0 = hvx_vmemu(src0 + nvec * 64);
            HVX_Vector v1 = hvx_vmemu(src0 + nvec * 64 + 32);

            HVX_Vector vcos = hvx_vmemu(theta_cache + nvec * 64);
            HVX_Vector vsin = hvx_vmemu(theta_cache + nvec * 64 + 32);

            HVX_VectorPair vx0_x1 = Q6_W_vdeal_VVR(v1, v0, -4);

            HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vcos);
            HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(vx0_x1), vsin);
            HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vcos);
            HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(vx0_x1), vsin);

            HVX_Vector v4 = Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s);
            HVX_Vector v5 = Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c);

            HVX_VectorPair vstore = Q6_W_vshuff_VVR(Q6_Vsf_equals_Vqf32(v5), Q6_Vsf_equals_Vqf32(v4), -4);

            ((HVX_Vector *) dst)[nvec * 2 + 0] = Q6_V_lo_W(vstore);
            hvx_vec_store_u(dst + nvec * 64 + 32, (nloe - 32) * اندازه(اعشاری), Q6_V_hi_W(vstore));
        }
    }
}

ایستا تهی درونخطی rope_basic_f32_inplace(ساختار htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, ثابت اعشاری * restrict theta_cache) {
    ثابت uint32_t n_offs = rctx->n_offs;
    #pragma unroll(4)
    برای (uint32_t i = 0; i < nr; i++) {
        اعشاری * s = (اعشاری *) (src + i * rctx->src0_row_size_aligned);
        hvx_rope_f32_aa(s + n_offs, s + n_offs, rctx->n_dims, theta_cache);
    }
}

ایستا تهی درونخطی rope_neox_f32_inplace(ساختار htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, uint32_t ne, ثابت اعشاری * restrict theta_cache) {
    ثابت uint32_t n_offs = rctx->n_offs;
    #pragma unroll(4)
    برای (uint32_t i = 0; i < nr; i++) {
        اعشاری * s = (اعشاری *) (src + i * rctx->src0_row_size_aligned);
        hvx_rope_neox_f32_aa(s + n_offs, s + n_offs, ne, theta_cache);
    }
}

ایستا تهی rope_job_f32(unsigned صحیح nth, unsigned صحیح ith, تهی * data) {
    ساختار htp_rope_context * rctx = (ساختار htp_rope_context *) data;
    ساختار htp_ops_context * octx = rctx->octx;

    ثابت ساختار htp_tensor * src0 = octx->src[0];
    ثابت ساختار htp_tensor * src1 = octx->src[1];
    ثابت ساختار htp_tensor * src2 = octx->src[2];
    ثابت ساختار htp_tensor * dst  = octx->dst;

    htp_rope_preamble;

    ثابت uint32_t src0_nrows = rctx->nrows;
    ثابت uint32_t src0_nrows_per_thread = rctx->src0_nrows_per_thread;

    ثابت uint32_t src0_start_row = rctx->row_start + src0_nrows_per_thread * ith;
    ثابت uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, rctx->row_start + src0_nrows);

    // no work for this thread
    اگر (src0_start_row >= src0_end_row) {
        بازگشت;
    }

    ثابت int32_t mode    = rctx->mode;
    // MROPE, IMROPE and VISION use NEOX-style pairing for the rotation
    ثابت bool    is_neox = (mode & HTP_ROPE_TYPE_NEOX) || (mode & HTP_ROPE_TYPE_MROPE);
    ثابت bool    is_vision = (mode == HTP_ROPE_TYPE_VISION);

    // VTCM setup
    uint8_t * src0_spad_base = rctx->vtcm_base + (ith * rctx->spad_per_thread);
    اعشاری *   theta_cache    = (اعشاری *) (src0_spad_base);
              src0_spad_base = src0_spad_base + rctx->theta_cache_offset;

    dma_queue * dma_q = octx->ctx->dma[ith];
    ساختار htp_thread_trace * tr = &octx->ctx->trace[ith];
    ثابت int32_t * pos = (ثابت int32_t *) (uintptr_t) src1->data;
    ثابت اعشاری * freq_factors = rctx->freq_factors;

    ثابت uint32_t i3_start = fastdiv(src0_start_row, &rctx->div_ne2_ne1);
    ثابت uint32_t rem      = fastmodulo(src0_start_row, ne2 * ne1, &rctx->div_ne2_ne1);
    ثابت uint32_t i2_start = fastdiv(rem, &rctx->div_ne1);
    ثابت uint32_t i1_start = fastmodulo(rem, ne1, &rctx->div_ne1);

    uint32_t ir = src0_start_row;
    uint32_t prev_i2 = (uint32_t) -1;
    uint32_t cur_slot = 0;

    برای (uint32_t i3 = i3_start; i3 < ne3; i3++) { // batch
        ثابت uint32_t i2_init = (i3 == i3_start) ? i2_start : 0;
        برای (uint32_t i2 = i2_init; i2 < ne2; i2++) { // seq-len
            ثابت uint32_t i1_init = (i3 == i3_start && i2 == i2_start) ? i1_start : 0;
            برای (uint32_t i1 = i1_init; i1 < ne1; ) { // attn-heads
                اگر (ir >= src0_end_row) برو done;

                // Rows in this block
                ثابت uint32_t nrows = MIN(src0_end_row - ir, ne1 - i1);

                // Depth before prefetch
                ثابت uint32_t dma_depth = dma_queue_depth(dma_q);

                // Prefetch up to 2 blocks
                ثابت uint32_t p_nrows = MIN(nrows, 2 * HTP_ROPE_SPAD_BLOCK);
                برای (uint32_t pr = 0; pr < p_nrows; pr += HTP_ROPE_SPAD_BLOCK) {
                    ثابت uint32_t pnr = MIN(nrows - pr, HTP_ROPE_SPAD_BLOCK);
                    ثابت uint32_t slot = (cur_slot + pr / HTP_ROPE_SPAD_BLOCK) % HTP_ROPE_SPAD_NSLOTS;
                    uint8_t * spad_slot = rope_spad_slot(src0_spad_base, slot, rctx->src0_row_size_aligned);
                    ثابت dma_addr_t src0_data = src0->data + i3 * nb03 + i2 * nb02 + (i1 + pr) * nb01;

                    // Dummy DMA transaction for sequencing (interleaving wr, rd, wr, rd, ...)
                    dma_queue_افزودن(dma_q, dma_make_data(dst->data, spad_slot), 0, 0, 0, 0);

                    dma_queue_افزودن(dma_q, dma_make_data(spad_slot, src0_data),
                        rctx->src0_row_size_aligned, rctx->src0_row_stride, rctx->src0_row_size, pnr);
                }

                // Update theta cache
                اگر (i2 != prev_i2) {
                    prev_i2 = i2;

                    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                    // VISION rotates the full row; other modes only rotate n_dims.
                    ثابت uint32_t n_cache = is_vision ? ne0 : (uint32_t) rctx->n_dims;
                    ثابت bool is_mrope = (rctx->mode & HTP_ROPE_TYPE_MROPE) != 0;
                    اگر (is_mrope) {
                        // src1 holds four position arrays stacked along ne0:
                        // pos[i2], pos[i2+ne2], pos[i2+ne2*2], pos[i2+ne2*3]
                        ثابت bool is_imrope = (rctx->mode == HTP_ROPE_TYPE_IMROPE);
                        mrope_cache_init(
                            (اعشاری) pos[i2],
                            (اعشاری) pos[i2 + ne2],
                            (اعشاری) pos[i2 + ne2 * 2],
                            (اعشاری) pos[i2 + ne2 * 3],
                            rctx->sections, is_imrope, is_vision,
                            rctx->freq_scale, freq_factors, rctx->corr_dims,
                            n_cache, rctx->ext_factor, rctx->attn_factor,
                            theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    } وگرنه {
                       rope_cache_init(pos[i2], rctx->freq_scale, freq_factors, rctx->corr_dims,
                                        n_cache, rctx->ext_factor, rctx->attn_factor,
                                        theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                }

                // Skip output DMA transactions from prev block (if any)
                برای (uint32_t d = 0; d < dma_depth; d++) { dma_queue_برداشت_nowait(dma_q); }

                // Compute loop
                ثابت uint32_t ne = is_vision ? ne0 : rctx->n_dims;
                ثابت uint32_t base_i1 = i1;
                ثابت uint32_t base_ir = ir;

                برای (uint32_t cnr = 0, cr = 0; cr < nrows; cr += cnr) {
                    cnr = MIN(nrows - cr, HTP_ROPE_SPAD_BLOCK);
                    ثابت uint32_t slot = (cur_slot + cr / HTP_ROPE_SPAD_BLOCK) % HTP_ROPE_SPAD_NSLOTS;
                    ثابت uint32_t cur_ir   = base_ir + cr;
                    ثابت uint32_t cur_i1   = base_i1 + cr;

                    dma_queue_برداشت(dma_q);
                    uint8_t * cur_spad = (uint8_t *) dma_queue_برداشت(dma_q).dst;

                    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);
                    اگر (is_neox || is_vision) {
                        rope_neox_f32_inplace(rctx, cur_spad, cnr, ne, theta_cache);
                    } وگرنه {
                        rope_basic_f32_inplace(rctx, cur_spad, cnr, theta_cache);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);

                    ثابت dma_addr_t dst_data = dst->data + i3 * nb3 + i2 * nb2 + cur_i1 * nb1;
                    dma_queue_افزودن(dma_q, dma_make_data(dst_data, cur_spad),
                        rctx->dst_row_stride, rctx->src0_row_size_aligned, rctx->dst_row_size, cnr);

                    // Prefetch 2 blocks ahead into the slot just freed
                    اگر ((cr + 2 * HTP_ROPE_SPAD_BLOCK) < nrows) {
                        ثابت uint32_t p_cr   = cr + 2 * HTP_ROPE_SPAD_BLOCK;
                        ثابت uint32_t pnr    = MIN(nrows - p_cr, HTP_ROPE_SPAD_BLOCK);
                        ثابت uint32_t p_slot = (cur_slot + p_cr / HTP_ROPE_SPAD_BLOCK) % HTP_ROPE_SPAD_NSLOTS;
                        uint8_t * p_spad      = rope_spad_slot(src0_spad_base, p_slot, rctx->src0_row_size_aligned);
                        ثابت dma_addr_t p_src0_data = src0->data + i3 * nb03 + i2 * nb02 + (base_i1 + p_cr) * nb01;

                        dma_queue_افزودن(dma_q, dma_make_data(p_spad, p_src0_data),
                            rctx->src0_row_size_aligned, rctx->src0_row_stride, rctx->src0_row_size, pnr);
                    }
                }

                ثابت uint32_t n_chunks = (nrows + HTP_ROPE_SPAD_BLOCK - 1) / HTP_ROPE_SPAD_BLOCK;
                cur_slot = (cur_slot + n_chunks) % HTP_ROPE_SPAD_NSLOTS;

                ir += nrows;
                i1 += nrows;
            }
        }
    }

done:
    dma_queue_flush(dma_q);

    FARF(بالا_2, "rope-f32: %d/%d: (%u:%u)\n", ith, nth, src0_start_row, src0_end_row);
}

ایستا صحیح execute_op_rope_f32(ساختار htp_ops_context * octx) {
    صحیح err = HTP_STATUS_OK;

    ثابت ساختار htp_tensor * src0 = octx->src[0];
    ثابت ساختار htp_tensor * src1 = octx->src[1];
    ثابت ساختار htp_tensor * src2 = octx->src[2];
    ثابت ساختار htp_tensor * dst  = octx->dst;

    انتخاب (octx->op) {
        حالت HTP_OP_ROPE:
            بشکن;

        پیشفرض:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            بازگشت HTP_STATUS_NO_SUPPORT;
    }

    ثابت ساختار htp_rope_kernel_params * kparams = (ثابت ساختار htp_rope_kernel_params *) octx->kernel_params;
    اگر (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        بازگشت HTP_STATUS_INVAL_PARAMS;
    }
    أكد(octx->ctx->vtcm_size >= kparams->vtcm_size);

    اگر (htp_tensor_is_extended(src1)) {
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    ثابت uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_data_row_size = dst->ne[0] * اندازه(اعشاری);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    اگر (octx->ctx->mdev.شمار > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, اندازه(اعشاری), (uint32_t) dst_data_row_size, &rows_per_chunk);
        ثابت ساختار htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        row_start = range.start;
        nrows     = range.شمار;
    }

    اگر (nrows == 0) {
        بازگشت HTP_STATUS_OK;
    }

    ثابت uint32_t n_threads = octx->n_threads;

    ثابت uint32_t ne0 = dst->ne[0];
    ثابت size_t src0_row_size   = src0->ne[0] * اندازه(اعشاری);
    ثابت size_t src0_row_stride = src0->nb[1];
    ثابت size_t dst_row_size    = dst->ne[0] * اندازه(اعشاری);
    ثابت size_t dst_row_stride  = dst->nb[1];

    ساختار htp_rope_context rctx;
    memset(&rctx, 0, اندازه(ساختار htp_rope_context));

    rctx.octx                  = octx;
    rctx.vtcm_base             = (uint8_t *) octx->ctx->vtcm_base;
    rctx.spad_per_thread       = kparams->spad_per_thread;
    rctx.theta_cache_offset    = kparams->theta_cache_offset;

    اگر (src2) {
        dma_queue * dma_q = octx->ctx->dma[0];
        ثابت size_t ff_size = src2->ne[0] * اندازه(اعشاری);
        اعشاری * vtcm_freq_factors = (اعشاری *) (rctx.vtcm_base + kparams->freq_factors_offset);
        dma_queue_افزودن(dma_q, dma_make_data(vtcm_freq_factors, src2->data),
                       kparams->freq_factors_size, 0, ff_size, 1);
        dma_queue_برداشت(dma_q);
        rctx.freq_factors = vtcm_freq_factors;
    }

    ثابت int32_t * op_params = &octx->op_params[0];
    rctx.n_dims     = ((ثابت int32_t *) op_params)[1];
    rctx.mode       = ((ثابت int32_t *) op_params)[2];
    rctx.n_ctx_orig = ((ثابت int32_t *) op_params)[4];
    rctx.n_offs     = ((ثابت int32_t *) op_params)[15];

    memcpy(&rctx.freq_base,   (int32_t *) op_params + 5,  اندازه(اعشاری));
    memcpy(&rctx.freq_scale,  (int32_t *) op_params + 6,  اندازه(اعشاری));
    memcpy(&rctx.ext_factor,  (int32_t *) op_params + 7,  اندازه(اعشاری));
    memcpy(&rctx.attn_factor, (int32_t *) op_params + 8,  اندازه(اعشاری));
    memcpy(&rctx.beta_fast,   (int32_t *) op_params + 9,  اندازه(اعشاری));
    memcpy(&rctx.beta_slow,   (int32_t *) op_params + 10, اندازه(اعشاری));
    memcpy(&rctx.sections,    (int32_t *) op_params + 11, اندازه(صحیح) * 4);

    rctx.theta_scale = powf(rctx.freq_base, -2.0f / rctx.n_dims);
    rctx.theta_powers[0] = 1.0f;
    برای (صحیح j = 1; j < 32; j++) {
        rctx.theta_powers[j] = rctx.theta_powers[j - 1] * rctx.theta_scale;
    }
    rctx.theta_scale_32 = rctx.theta_powers[31] * rctx.theta_scale;

    rope_corr_dims(rctx.n_dims, rctx.n_ctx_orig, rctx.freq_base, rctx.beta_fast, rctx.beta_slow, rctx.corr_dims);

    rctx.src0_row_size         = src0_row_size;
    rctx.src0_row_stride       = src0_row_stride;
    rctx.dst_row_size          = dst_row_size;
    rctx.dst_row_stride        = dst_row_stride;
    rctx.src0_row_size_aligned = kparams->src0_row_size_aligned;

    rctx.src0_nrows            = nrows;
    rctx.nrows                 = nrows;
    rctx.row_start             = row_start;
    rctx.src0_nrows_per_thread = fastdiv(nrows + n_threads - 1, &octx->n_threads_div);
    rctx.div_ne2_ne1           = kparams->div_ne2_ne1;
    rctx.div_ne1               = kparams->div_ne1;

    FARF(بالا_2, "rope-f32 n-rows %u n-dims %d ne0 %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n", rctx.src0_nrows, rctx.n_dims, ne0,
         rctx.ext_factor, rctx.theta_scale, rctx.attn_factor);

    work_queue_run(octx->ctx->work_queue, rope_job_f32, &rctx, n_threads);

    بازگشت err;
}

صحیح op_rope(ساختار htp_ops_context * octx) {
    انتخاب (octx->src[0]->type) {
        حالت HTP_TYPE_F32:
            بازگشت execute_op_rope_f32(octx);

        پیشفرض:
            بازگشت HTP_STATUS_NO_SUPPORT;
    }
}
