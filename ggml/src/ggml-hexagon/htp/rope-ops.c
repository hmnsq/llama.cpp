#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#অন্তর্ভাৱ <HAP_farf.h>
#অন্তর্ভাৱ <HAP_perf.h>

#অন্তর্ভাৱ <math.h>
#অন্তর্ভাৱ <string.h>
#অন্তর্ভাৱ <stdlib.h>

#অন্তর্ভাৱ "dma-queue.h"
#অন্তর্ভাৱ "hvx-utils.h"
#অন্তর্ভাৱ "hex-fastdiv.h"

#define GGML_COMMON_DECL_C
#অন্তর্ভাৱ "ggml-common.h"
#অন্তর্ভাৱ "htp-ctx.h"
#অন্তর্ভাৱ "htp-ops.h"
#অন্তর্ভাৱ "htp-tensor.h"
#অন্তর্ভাৱ "rope-ops.h"

// Redefined the rope type constants as we can't include ggml.h
#define HTP_ROPE_TYPE_NORMAL 0
#define HTP_ROPE_TYPE_NEOX   2
#define HTP_ROPE_TYPE_MROPE  8
#define HTP_ROPE_TYPE_VISION 24
#define HTP_ROPE_TYPE_IMROPE 40

#define htp_rope_preamble              \
    ধ্রুৱ uint32_t ne00 = src0->ne[0]; \
    ধ্রুৱ uint32_t ne01 = src0->ne[1]; \
    ধ্রুৱ uint32_t ne02 = src0->ne[2]; \
    ধ্রুৱ uint32_t ne03 = src0->ne[3]; \
                                       \
    ধ্রুৱ uint32_t ne0 = dst->ne[0];   \
    ধ্রুৱ uint32_t ne1 = dst->ne[1];   \
    ধ্রুৱ uint32_t ne2 = dst->ne[2];   \
    ধ্রুৱ uint32_t ne3 = dst->ne[3];   \
                                       \
    ধ্রুৱ uint32_t nb00 = src0->nb[0]; \
    ধ্রুৱ uint32_t nb01 = src0->nb[1]; \
    ধ্রুৱ uint32_t nb02 = src0->nb[2]; \
    ধ্রুৱ uint32_t nb03 = src0->nb[3]; \
                                       \
    ধ্রুৱ uint32_t nb0 = dst->nb[0];   \
    ধ্রুৱ uint32_t nb1 = dst->nb[1];   \
    ধ্রুৱ uint32_t nb2 = dst->nb[2];   \
    ধ্রুৱ uint32_t nb3 = dst->nb[3];

সংরচনা htp_rope_context {
    int32_t n_dims;
    int32_t n_offs;
    int32_t mode;
    int32_t n_ctx_orig;
    int32_t sections[4];

    ভিন্ন freq_base;
    ভিন্ন freq_scale;
    ভিন্ন ext_factor;
    ভিন্ন attn_factor;
    ভিন্ন beta_fast;
    ভিন্ন beta_slow;
    ভিন্ন theta_scale;
    ভিন্ন theta_scale_32;
    ভিন্ন theta_powers[32];
    ভিন্ন corr_dims[2];

    uint32_t src0_nrows_per_thread;

    সংরচনা htp_ops_context * octx;

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

    সংরচনা fastdiv_মানসমূহ div_ne2_ne1;
    সংরচনা fastdiv_মানসমূহ div_ne1;

    ধ্রুৱ ভিন্ন * freq_factors;
};

স্থির ভিন্ন rope_yarn_ramp(ধ্রুৱ ভিন্ন নিচেৰ, ধ্রুৱ ভিন্ন উপৰেৰ, ধ্রুৱ পূর্ণাঙ্ক i0) {
    ধ্রুৱ ভিন্ন y = (i0 / 2 - নিচেৰ) / MAX(0.001f, উপৰেৰ - নিচেৰ);

    ঘূৰাই (1 - MIN(1, MAX(0, y)));
}

// Compute one (cos, sin) pair into cache[i0], cache[i0+1] applying YaRN scaling.
স্থির অন্তঃস্থ শূন্য rope_yarn_one(ভিন্ন theta, ভিন্ন freq_scale, ভিন্ন * corr_dims,
                                 uint32_t i0, ভিন্ন ext_factor, ভিন্ন mscale,
                                 ভিন্ন * cache) {
    ভিন্ন theta_extrap = theta;

    // Get n-d rotational scaling corrected for extrapolation
    ভিন্ন theta_interp = freq_scale * theta_extrap;
    ভিন্ন theta_final  = theta_interp;
    ভিন্ন mscale_final = mscale;

    যদি (ext_factor != 0.0f) {
        ভিন্ন ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta_final    = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale_final  *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    ধ্রুৱ uint32_t b = i0 / 64;
    ধ্রুৱ uint32_t k = (i0 % 64) / 2;
    cache[b * 64 + k]      = cosf(theta_final) * mscale_final;
    cache[b * 64 + 32 + k] = sinf(theta_final) * mscale_final;
}

// 32 thetas -> 32 deinterleaved pairs [cos[32] | sin[32]] at cache[i0].
স্থির অন্তঃস্থ শূন্য rope_cache_hvx_32(ভিন্ন * cache, uint32_t i0,
                                     HVX_Vector v_theta,
                                     ধ্রুৱ ভিন্ন * freq_factors,
                                     HVX_Vector v_freq_scale,
                                     HVX_Vector v_mscale) {
    যদি (freq_factors) {
        HVX_Vector v_ff = hvx_vmemu(freq_factors + i0 / 2);
        v_theta = hvx_vec_mul_f32_f32(v_theta, hvx_vec_inverse_f32(v_ff));
    }

    HVX_Vector v_theta_final = hvx_vec_mul_f32_f32(v_theta, v_freq_scale);
    HVX_Vector vcos;
    HVX_Vector vsin;
    hvx_vec_sincos_f32(v_theta_final, &vcos, &vsin);
    vcos = hvx_vec_mul_f32_f32(vcos, v_mscale);
    vsin = hvx_vec_mul_f32_f32(vsin, v_mscale);

    যদি (((uintptr_t) (cache + i0)) % 128 == 0) {
        hvx_vmem(cache + i0 + 0)  = vcos;
        hvx_vmem(cache + i0 + 32) = vsin;
    } অন্যথা {
        hvx_vec_store_u(cache + i0 + 0,  32 * পরিমাণ(ভিন্ন), vcos);
        hvx_vec_store_u(cache + i0 + 32, 32 * পরিমাণ(ভিন্ন), vsin);
    }
}

স্থির __attribute__((noinline)) শূন্য rope_cache_init(ধ্রুৱ ভিন্ন    theta_base,
                            ধ্রুৱ ভিন্ন    freq_scale,
                            ধ্রুৱ ভিন্ন *  freq_factors,
                            ভিন্ন *        corr_dims,
                            ধ্রুৱ uint32_t n_cache,
                            ধ্রুৱ ভিন্ন    ext_factor,
                            ধ্রুৱ ভিন্ন    mscale,
                            ভিন্ন *        cache,
                            ধ্রুৱ ভিন্ন    theta_scale,
                            ধ্রুৱ ভিন্ন *  theta_powers,
                            ধ্রুৱ ভিন্ন    theta_scale_32) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    যদি (ext_factor == 0.0f) {
        // Fast path: fully vectorized
        // We process 32 pairs (64 elements) per iteration.
        ধ্রুৱ uint32_t n_blocks = n_cache / 64;

        HVX_Vector v_theta_powers = hvx_vmemu(theta_powers);
        HVX_Vector v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_Vector v_mscale = hvx_vec_splat_f32(mscale);

        ভিন্ন theta_block = theta_base;

        বাবে (uint32_t b = 0; b < n_blocks; b++) {
            uint32_t i0 = b * 64;
            HVX_Vector v_theta_base = hvx_vec_splat_f32(theta_block);
            HVX_Vector v_theta = hvx_vec_mul_f32_f32(v_theta_base, v_theta_powers);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            theta_block *= theta_scale_32;
        }

        // Leftovers
        ভিন্ন theta = theta_block;
        বাবে (uint32_t i0 = n_blocks * 64; i0 < n_cache; i0 += 2) {
            ধ্রুৱ ভিন্ন ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta *= theta_scale;
        }
    } অন্যথা {
        ভিন্ন theta = theta_base;
        বাবে (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
            ধ্রুৱ ভিন্ন ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta *= theta_scale;
        }
    }
}

স্থির অন্তঃস্থ ভিন্ন mrope_pick_theta(ভিন্ন theta_t, ভিন্ন theta_h, ভিন্ন theta_w, ভিন্ন theta_e,
                                     পূর্ণাঙ্ক sector, ধ্রুৱ int32_t sections[4], পূর্ণাঙ্ক sec_w, পূর্ণাঙ্ক sec_e,
                                     bool is_imrope) {
    যদি (is_imrope) {
        যদি      (sector % 3 == 0 && sector < 3 * sections[0]) { ঘূৰাই theta_t; }
        অন্যথা যদি (sector % 3 == 1 && sector < 3 * sections[1]) { ঘূৰাই theta_h; }
        অন্যথা যদি (sector % 3 == 2 && sector < 3 * sections[2]) { ঘূৰাই theta_w; }
        অন্যথা                                                   { ঘূৰাই theta_e; }
    }
    যদি      (sector < sections[0]) { ঘূৰাই theta_t; }
    অন্যথা যদি (sector < sec_w)       { ঘূৰাই theta_h; }
    অন্যথা যদি (sector < sec_e)       { ঘূৰাই theta_w; }
    অন্যথা                           { ঘূৰাই theta_e; }
}

// lane j is 1 when (j % 3) == rem
স্থির ধ্রুৱ ভিন্ন __attribute__((aligned(128))) mrope_mod3_eq0[32] = {
    1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0
};
স্থির ধ্রুৱ ভিন্ন __attribute__((aligned(128))) mrope_mod3_eq1[32] = {
    0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1
};
স্থির ধ্রুৱ ভিন্ন __attribute__((aligned(128))) mrope_mod3_eq2[32] = {
    0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0
};

স্থির ধ্রুৱ ভিন্ন __attribute__((aligned(128))) mrope_k_ramp[32] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
};

স্থির অন্তঃস্থ HVX_VectorPred mrope_mask_eq1(ধ্রুৱ ভিন্ন * m) {
    ঘূৰাই Q6_Q_vcmp_gt_VsfVsf(hvx_vmemu(m), Q6_V_vzero());
}

// IMROPE without wrap: theta[k] = pos[k % 3] * scale^k
স্থির অন্তঃস্থ HVX_Vector mrope_thetas_imrope_mod3(ভিন্ন pos_t, ভিন্ন pos_h, ভিন্ন pos_w,
                                                  uint32_t k0, HVX_Vector v_powers, ভিন্ন scale_block) {
    ধ্রুৱ পূর্ণাঙ্ক r = (পূর্ণাঙ্ক) (k0 % 3);
    ধ্রুৱ ভিন্ন * mt = (r == 0) ? mrope_mod3_eq0 : (r == 1) ? mrope_mod3_eq2 : mrope_mod3_eq1;
    ধ্রুৱ ভিন্ন * mh = (r == 0) ? mrope_mod3_eq1 : (r == 1) ? mrope_mod3_eq0 : mrope_mod3_eq2;

    HVX_Vector v = hvx_vec_splat_f32(pos_w);
    v = Q6_V_vmux_QVV(mrope_mask_eq1(mh), hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(mrope_mask_eq1(mt), hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    ঘূৰাই hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_block));
}

// Contiguous MROPE without wrap: theta[k] = pos[section(k)] * scale^k
স্থির অন্তঃস্থ HVX_Vector mrope_thetas_contig(ভিন্ন pos_t, ভিন্ন pos_h, ভিন্ন pos_w, ভিন্ন pos_e,
                                             uint32_t k0, পূর্ণাঙ্ক s0, পূর্ণাঙ্ক sec_w, পূর্ণাঙ্ক sec_e,
                                             HVX_Vector v_powers, ভিন্ন scale_block) {
    HVX_Vector v_k = hvx_vec_add_f32_f32(hvx_vec_splat_f32((ভিন্ন) k0), hvx_vmemu(mrope_k_ramp));
    HVX_VectorPred lt_s0 = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((ভিন্ন) s0), v_k);
    HVX_VectorPred lt_sw = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((ভিন্ন) sec_w), v_k);
    HVX_VectorPred lt_se = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((ভিন্ন) sec_e), v_k);

    HVX_Vector v = hvx_vec_splat_f32(pos_e);
    v = Q6_V_vmux_QVV(lt_se, hvx_vec_splat_f32(pos_w), v);
    v = Q6_V_vmux_QVV(lt_sw, hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(lt_s0, hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    ঘূৰাই hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_block));
}

// pos_t/h/w/e: the four position ids for this sequence step (t=time, h=height, w=width, e=extra).
// sections[4]: number of head dims assigned to each position component.
স্থির __attribute__((noinline)) শূন্য mrope_cache_init(ধ্রুৱ ভিন্ন    pos_t,
                             ধ্রুৱ ভিন্ন    pos_h,
                             ধ্রুৱ ভিন্ন    pos_w,
                             ধ্রুৱ ভিন্ন    pos_e,
                             ধ্রুৱ int32_t  sections[4],
                             ধ্রুৱ bool     is_imrope,
                             ধ্রুৱ bool     indep_sects,
                             ধ্রুৱ ভিন্ন    freq_scale,
                             ধ্রুৱ ভিন্ন *  freq_factors,
                             ভিন্ন *        corr_dims,
                             ধ্রুৱ uint32_t n_cache,
                             ধ্রুৱ ভিন্ন    ext_factor,
                             ধ্রুৱ ভিন্ন    mscale,
                             ভিন্ন *        cache,
                             ধ্রুৱ ভিন্ন    theta_scale,
                             ধ্রুৱ ভিন্ন *  theta_powers,
                             ধ্রুৱ ভিন্ন    theta_scale_32) {
    ধ্রুৱ পূর্ণাঙ্ক sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    ধ্রুৱ পূর্ণাঙ্ক sec_w     = sections[0] + sections[1];
    ধ্রুৱ পূর্ণাঙ্ক sec_e     = sec_w + sections[2];
    ধ্রুৱ uint32_t n_pairs = n_cache / 2;

    ধ্রুৱ bool no_wrap = (sect_dims > 0) && (n_pairs <= (uint32_t) sect_dims);
    ধ্রুৱ bool imrope_mod3 = is_imrope && !indep_sects && no_wrap
        && sections[0] > 0 && sections[1] > 0 && sections[2] > 0
        && n_pairs <= (uint32_t) (3 * sections[0])
        && n_pairs <= (uint32_t) (3 * sections[1])
        && n_pairs <= (uint32_t) (3 * sections[2]);
    ধ্রুৱ bool contig = !is_imrope && !indep_sects && no_wrap;

    যদি (ext_factor == 0.0f && (imrope_mod3 || contig)) {
        HVX_Vector v_powers     = hvx_vmemu(theta_powers);
        HVX_Vector v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_Vector v_mscale     = hvx_vec_splat_f32(mscale);
        ভিন্ন scale_block = 1.0f;
        ধ্রুৱ uint32_t n_blocks = n_cache / 64;

        বাবে (uint32_t b = 0; b < n_blocks; b++) {
            ধ্রুৱ uint32_t i0 = b * 64;
            ধ্রুৱ uint32_t k0 = b * 32;
            HVX_Vector v_theta = imrope_mod3
                ? mrope_thetas_imrope_mod3(pos_t, pos_h, pos_w, k0, v_powers, scale_block)
                : mrope_thetas_contig(pos_t, pos_h, pos_w, pos_e, k0, sections[0], sec_w, sec_e,
                                      v_powers, scale_block);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            scale_block *= theta_scale_32;
        }

        ভিন্ন theta_k = scale_block;
        বাবে (uint32_t k = n_blocks * 32; k < n_pairs; k++) {
            ধ্রুৱ uint32_t i0 = 2 * k;
            ধ্রুৱ ভিন্ন pos = mrope_pick_theta(pos_t, pos_h, pos_w, pos_e,
                                               (পূর্ণাঙ্ক) k, sections, sec_w, sec_e, is_imrope);
            ধ্রুৱ ভিন্ন ff = freq_factors ? freq_factors[k] : 1.0f;
            rope_yarn_one(pos * theta_k / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta_k *= theta_scale;
        }
        ঘূৰাই;
    }

    ভিন্ন theta_t = pos_t;
    ভিন্ন theta_h = pos_h;
    ভিন্ন theta_w = pos_w;
    ভিন্ন theta_e = pos_e;

    ধ্রুৱ bool use_hvx = (ext_factor == 0.0f);
    ভিন্ন __attribute__((aligned(128))) thetas[32];
    uint32_t n_thetas = 0;
    uint32_t block_i0 = 0;

    HVX_Vector v_freq_scale = hvx_vec_splat_f32(freq_scale);
    HVX_Vector v_mscale     = hvx_vec_splat_f32(mscale);

    বাবে (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
        ধ্রুৱ পূর্ণাঙ্ক sector = (i0 / 2) % sect_dims;

        যদি (indep_sects) {
            // Reset theta when crossing into a new section.
            যদি      (sector == 0)           { theta_t = pos_t; }
            অন্যথা যদি (sector == sections[0]) { theta_h = pos_h; }
            অন্যথা যদি (sector == sec_w)       { theta_w = pos_w; }
            অন্যথা যদি (sector == sec_e)       { theta_e = pos_e; }
        }

        ধ্রুৱ ভিন্ন theta = mrope_pick_theta(theta_t, theta_h, theta_w, theta_e,
                                             sector, sections, sec_w, sec_e, is_imrope);

        যদি (use_hvx) {
            যদি (n_thetas == 0) {
                block_i0 = i0;
            }
            thetas[n_thetas++] = theta;
            যদি (n_thetas == 32) {
                rope_cache_hvx_32(cache, block_i0, hvx_vmemu(thetas), freq_factors, v_freq_scale, v_mscale);
                n_thetas = 0;
            }
        } অন্যথা {
            ধ্রুৱ ভিন্ন ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
        }

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }

    বাবে (uint32_t k = 0; k < n_thetas; k++) {
        ধ্রুৱ uint32_t i0 = block_i0 + 2 * k;
        ধ্রুৱ ভিন্ন ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn_one(thetas[k] / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
    }
}

#define M_PI 3.1415926535897932384626433

স্থির শূন্য rope_corr_dims(পূর্ণাঙ্ক     n_dims,
                           পূর্ণাঙ্ক     n_ctx_orig,
                           ভিন্ন   freq_base,
                           ভিন্ন   beta_fast,
                           ভিন্ন   beta_slow,
                           ভিন্ন * dims) {
    ভিন্ন start = floorf(n_dims * logf(n_ctx_orig / (beta_fast * 2 * (ভিন্ন) M_PI)) / (2 * logf(freq_base)));
    ভিন্ন end   = ceilf(n_dims * logf(n_ctx_orig / (beta_slow * 2 * (ভিন্ন) M_PI)) / (2 * logf(freq_base)));
    dims[0]     = MAX(0, start);
    dims[1]     = MIN(n_dims - 1, end);
}

স্থির অন্তঃস্থ শূন্য hvx_rope_neox_mul(HVX_Vector v0, HVX_Vector v1, HVX_Vector vcos, HVX_Vector vsin,
                                     HVX_Vector * o0, HVX_Vector * o1) {
    HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, vcos);
    HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, vsin);
    HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, vcos);
    HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, vsin);
    *o0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s));
    *o1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c));
}

// theta_cache full 32-pair blocks are deinterleaved [cos | sin].
স্থির অন্তঃস্থ শূন্য hvx_rope_neox_f32_aa(ভিন্ন * restrict dst, ধ্রুৱ ভিন্ন * restrict src0, uint32_t ne, ধ্রুৱ ভিন্ন * restrict theta_cache) {
    ধ্রুৱ uint32_t he = ne / 2;
    ধ্রুৱ uint32_t nvec = he / 32;
    ধ্রুৱ uint32_t nloe = he % 32;

    যদি (nloe == 0) {
        ধ্রুৱ HVX_Vector * vs = (ধ্রুৱ HVX_Vector *) src0;
        ধ্রুৱ HVX_Vector * vt = (ধ্রুৱ HVX_Vector *) theta_cache;
        HVX_Vector * vd = (HVX_Vector *) dst;
        বাবে (uint32_t i = 0; i < nvec; i++) {
            HVX_Vector o0, o1;
            hvx_rope_neox_mul(vs[i], vs[nvec + i], vt[i * 2 + 0], vt[i * 2 + 1], &o0, &o1);
            vd[i] = o0;
            vd[nvec + i] = o1;
        }
        ঘূৰাই;
    }

    বাবে (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector o0, o1;
        hvx_rope_neox_mul(((ধ্রুৱ HVX_Vector *) src0)[i],
                          hvx_vmemu(src0 + he + i * 32),
                          ((ধ্রুৱ HVX_Vector *) theta_cache)[i * 2 + 0],
                          ((ধ্রুৱ HVX_Vector *) theta_cache)[i * 2 + 1],
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
    hvx_vec_store_u(dst + nvec * 32, nloe * পরিমাণ(ভিন্ন), o0);
    hvx_vec_store_u(dst + he + nvec * 32, nloe * পরিমাণ(ভিন্ন), o1);
}

স্থির অন্তঃস্থ শূন্য hvx_rope_f32_aa(ভিন্ন * restrict dst, ধ্রুৱ ভিন্ন * restrict src0, uint32_t ne, ধ্রুৱ ভিন্ন * restrict theta_cache) {
    ধ্রুৱ uint32_t nvec = ne / 64;
    ধ্রুৱ uint32_t nloe = ne % 64;

    বাবে (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v0 = ((ধ্রুৱ HVX_Vector *) src0)[i * 2 + 0];
        HVX_Vector v1 = ((ধ্রুৱ HVX_Vector *) src0)[i * 2 + 1];

        HVX_Vector vcos = ((ধ্রুৱ HVX_Vector *) theta_cache)[i * 2 + 0];
        HVX_Vector vsin = ((ধ্রুৱ HVX_Vector *) theta_cache)[i * 2 + 1];

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

    যদি (nloe > 0) {
        যদি (nloe <= 32) {
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

            hvx_vec_store_u(dst + nvec * 64, nloe * পরিমাণ(ভিন্ন), Q6_V_lo_W(vstore));
        } অন্যথা {
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
            hvx_vec_store_u(dst + nvec * 64 + 32, (nloe - 32) * পরিমাণ(ভিন্ন), Q6_V_hi_W(vstore));
        }
    }
}

স্থির শূন্য অন্তঃস্থ rope_basic_f32_inplace(সংরচনা htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, ধ্রুৱ ভিন্ন * restrict theta_cache) {
    ধ্রুৱ uint32_t n_offs = rctx->n_offs;
    #pragma unroll(4)
    বাবে (uint32_t i = 0; i < nr; i++) {
        ভিন্ন * s = (ভিন্ন *) (src + i * rctx->src0_row_size_aligned);
        hvx_rope_f32_aa(s + n_offs, s + n_offs, rctx->n_dims, theta_cache);
    }
}

স্থির শূন্য অন্তঃস্থ rope_neox_f32_inplace(সংরচনা htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, uint32_t ne, ধ্রুৱ ভিন্ন * restrict theta_cache) {
    ধ্রুৱ uint32_t n_offs = rctx->n_offs;
    #pragma unroll(4)
    বাবে (uint32_t i = 0; i < nr; i++) {
        ভিন্ন * s = (ভিন্ন *) (src + i * rctx->src0_row_size_aligned);
        hvx_rope_neox_f32_aa(s + n_offs, s + n_offs, ne, theta_cache);
    }
}

স্থির শূন্য rope_job_f32(unsigned পূর্ণাঙ্ক nth, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা htp_rope_context * rctx = (সংরচনা htp_rope_context *) data;
    সংরচনা htp_ops_context * octx = rctx->octx;

    ধ্রুৱ সংরচনা htp_tensor * src0 = octx->src[0];
    ধ্রুৱ সংরচনা htp_tensor * src1 = octx->src[1];
    ধ্রুৱ সংরচনা htp_tensor * src2 = octx->src[2];
    ধ্রুৱ সংরচনা htp_tensor * dst  = octx->dst;

    htp_rope_preamble;

    ধ্রুৱ uint32_t src0_nrows = rctx->nrows;
    ধ্রুৱ uint32_t src0_nrows_per_thread = rctx->src0_nrows_per_thread;

    ধ্রুৱ uint32_t src0_start_row = rctx->row_start + src0_nrows_per_thread * ith;
    ধ্রুৱ uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, rctx->row_start + src0_nrows);

    // no work for this thread
    যদি (src0_start_row >= src0_end_row) {
        ঘূৰাই;
    }

    ধ্রুৱ int32_t mode    = rctx->mode;
    // MROPE, IMROPE and VISION use NEOX-style pairing for the rotation
    ধ্রুৱ bool    is_neox = (mode & HTP_ROPE_TYPE_NEOX) || (mode & HTP_ROPE_TYPE_MROPE);
    ধ্রুৱ bool    is_vision = (mode == HTP_ROPE_TYPE_VISION);

    // VTCM setup
    uint8_t * src0_spad_base = rctx->vtcm_base + (ith * rctx->spad_per_thread);
    ভিন্ন *   theta_cache    = (ভিন্ন *) (src0_spad_base);
              src0_spad_base = src0_spad_base + rctx->theta_cache_offset;

    dma_queue * dma_q = octx->ctx->dma[ith];
    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];
    ধ্রুৱ int32_t * pos = (ধ্রুৱ int32_t *) (uintptr_t) src1->data;
    ধ্রুৱ ভিন্ন * freq_factors = rctx->freq_factors;

    ধ্রুৱ uint32_t i3_start = fastdiv(src0_start_row, &rctx->div_ne2_ne1);
    ধ্রুৱ uint32_t rem      = fastmodulo(src0_start_row, ne2 * ne1, &rctx->div_ne2_ne1);
    ধ্রুৱ uint32_t i2_start = fastdiv(rem, &rctx->div_ne1);
    ধ্রুৱ uint32_t i1_start = fastmodulo(rem, ne1, &rctx->div_ne1);

    uint32_t ir = src0_start_row;
    uint32_t prev_i2 = (uint32_t) -1;
    uint32_t cur_slot = 0;

    বাবে (uint32_t i3 = i3_start; i3 < ne3; i3++) { // batch
        ধ্রুৱ uint32_t i2_init = (i3 == i3_start) ? i2_start : 0;
        বাবে (uint32_t i2 = i2_init; i2 < ne2; i2++) { // seq-len
            ধ্রুৱ uint32_t i1_init = (i3 == i3_start && i2 == i2_start) ? i1_start : 0;
            বাবে (uint32_t i1 = i1_init; i1 < ne1; ) { // attn-heads
                যদি (ir >= src0_end_row) গচ্ছ done;

                // Rows in this block
                ধ্রুৱ uint32_t nrows = MIN(src0_end_row - ir, ne1 - i1);

                // Depth before prefetch
                ধ্রুৱ uint32_t dma_depth = dma_queue_depth(dma_q);

                // Prefetch up to 2 blocks
                ধ্রুৱ uint32_t p_nrows = MIN(nrows, 2 * HTP_ROPE_SPAD_BLOCK);
                বাবে (uint32_t pr = 0; pr < p_nrows; pr += HTP_ROPE_SPAD_BLOCK) {
                    ধ্রুৱ uint32_t pnr = MIN(nrows - pr, HTP_ROPE_SPAD_BLOCK);
                    ধ্রুৱ uint32_t slot = (cur_slot + pr / HTP_ROPE_SPAD_BLOCK) % HTP_ROPE_SPAD_NSLOTS;
                    uint8_t * spad_slot = rope_spad_slot(src0_spad_base, slot, rctx->src0_row_size_aligned);
                    ধ্রুৱ dma_addr_t src0_data = src0->data + i3 * nb03 + i2 * nb02 + (i1 + pr) * nb01;

                    // Dummy DMA transaction for sequencing (interleaving wr, rd, wr, rd, ...)
                    dma_queue_ঠেলো(dma_q, dma_make_data(dst->data, spad_slot), 0, 0, 0, 0);

                    dma_queue_ঠেলো(dma_q, dma_make_data(spad_slot, src0_data),
                        rctx->src0_row_size_aligned, rctx->src0_row_stride, rctx->src0_row_size, pnr);
                }

                // Update theta cache
                যদি (i2 != prev_i2) {
                    prev_i2 = i2;

                    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                    // VISION rotates the full row; other modes only rotate n_dims.
                    ধ্রুৱ uint32_t n_cache = is_vision ? ne0 : (uint32_t) rctx->n_dims;
                    ধ্রুৱ bool is_mrope = (rctx->mode & HTP_ROPE_TYPE_MROPE) != 0;
                    যদি (is_mrope) {
                        // src1 holds four position arrays stacked along ne0:
                        // pos[i2], pos[i2+ne2], pos[i2+ne2*2], pos[i2+ne2*3]
                        ধ্রুৱ bool is_imrope = (rctx->mode == HTP_ROPE_TYPE_IMROPE);
                        mrope_cache_init(
                            (ভিন্ন) pos[i2],
                            (ভিন্ন) pos[i2 + ne2],
                            (ভিন্ন) pos[i2 + ne2 * 2],
                            (ভিন্ন) pos[i2 + ne2 * 3],
                            rctx->sections, is_imrope, is_vision,
                            rctx->freq_scale, freq_factors, rctx->corr_dims,
                            n_cache, rctx->ext_factor, rctx->attn_factor,
                            theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    } অন্যথা {
                       rope_cache_init(pos[i2], rctx->freq_scale, freq_factors, rctx->corr_dims,
                                        n_cache, rctx->ext_factor, rctx->attn_factor,
                                        theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                }

                // Skip output DMA transactions from prev block (if any)
                বাবে (uint32_t d = 0; d < dma_depth; d++) { dma_queue_বেৰ_কৰো_nowait(dma_q); }

                // Compute loop
                ধ্রুৱ uint32_t ne = is_vision ? ne0 : rctx->n_dims;
                ধ্রুৱ uint32_t base_i1 = i1;
                ধ্রুৱ uint32_t base_ir = ir;

                বাবে (uint32_t cnr = 0, cr = 0; cr < nrows; cr += cnr) {
                    cnr = MIN(nrows - cr, HTP_ROPE_SPAD_BLOCK);
                    ধ্রুৱ uint32_t slot = (cur_slot + cr / HTP_ROPE_SPAD_BLOCK) % HTP_ROPE_SPAD_NSLOTS;
                    ধ্রুৱ uint32_t cur_ir   = base_ir + cr;
                    ধ্রুৱ uint32_t cur_i1   = base_i1 + cr;

                    dma_queue_বেৰ_কৰো(dma_q);
                    uint8_t * cur_spad = (uint8_t *) dma_queue_বেৰ_কৰো(dma_q).dst;

                    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);
                    যদি (is_neox || is_vision) {
                        rope_neox_f32_inplace(rctx, cur_spad, cnr, ne, theta_cache);
                    } অন্যথা {
                        rope_basic_f32_inplace(rctx, cur_spad, cnr, theta_cache);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);

                    ধ্রুৱ dma_addr_t dst_data = dst->data + i3 * nb3 + i2 * nb2 + cur_i1 * nb1;
                    dma_queue_ঠেলো(dma_q, dma_make_data(dst_data, cur_spad),
                        rctx->dst_row_stride, rctx->src0_row_size_aligned, rctx->dst_row_size, cnr);

                    // Prefetch 2 blocks ahead into the slot just freed
                    যদি ((cr + 2 * HTP_ROPE_SPAD_BLOCK) < nrows) {
                        ধ্রুৱ uint32_t p_cr   = cr + 2 * HTP_ROPE_SPAD_BLOCK;
                        ধ্রুৱ uint32_t pnr    = MIN(nrows - p_cr, HTP_ROPE_SPAD_BLOCK);
                        ধ্রুৱ uint32_t p_slot = (cur_slot + p_cr / HTP_ROPE_SPAD_BLOCK) % HTP_ROPE_SPAD_NSLOTS;
                        uint8_t * p_spad      = rope_spad_slot(src0_spad_base, p_slot, rctx->src0_row_size_aligned);
                        ধ্রুৱ dma_addr_t p_src0_data = src0->data + i3 * nb03 + i2 * nb02 + (base_i1 + p_cr) * nb01;

                        dma_queue_ঠেলো(dma_q, dma_make_data(p_spad, p_src0_data),
                            rctx->src0_row_size_aligned, rctx->src0_row_stride, rctx->src0_row_size, pnr);
                    }
                }

                ধ্রুৱ uint32_t n_chunks = (nrows + HTP_ROPE_SPAD_BLOCK - 1) / HTP_ROPE_SPAD_BLOCK;
                cur_slot = (cur_slot + n_chunks) % HTP_ROPE_SPAD_NSLOTS;

                ir += nrows;
                i1 += nrows;
            }
        }
    }

done:
    dma_queue_flush(dma_q);

    FARF(উপৰেৰ_2, "rope-f32: %d/%d: (%u:%u)\n", ith, nth, src0_start_row, src0_end_row);
}

স্থির পূর্ণাঙ্ক execute_op_rope_f32(সংরচনা htp_ops_context * octx) {
    পূর্ণাঙ্ক err = HTP_STATUS_OK;

    ধ্রুৱ সংরচনা htp_tensor * src0 = octx->src[0];
    ধ্রুৱ সংরচনা htp_tensor * src1 = octx->src[1];
    ধ্রুৱ সংরচনা htp_tensor * src2 = octx->src[2];
    ধ্রুৱ সংরচনা htp_tensor * dst  = octx->dst;

    ৱিকল্প (octx->op) {
        পক্ষ HTP_OP_ROPE:
            ভঙ্গ;

        স্ৱভাৱ:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            ঘূৰাই HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুৱ সংরচনা htp_rope_kernel_params * kparams = (ধ্রুৱ সংরচনা htp_rope_kernel_params *) octx->kernel_params;
    যদি (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ঘূৰাই HTP_STATUS_INVAL_PARAMS;
    }
    প্রতিজ্ঞা(octx->ctx->vtcm_size >= kparams->vtcm_size);

    যদি (htp_tensor_is_extended(src1)) {
        ঘূৰাই HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুৱ uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুৱ size_t dst_data_row_size = dst->ne[0] * পরিমাণ(ভিন্ন);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, পরিমাণ(ভিন্ন), (uint32_t) dst_data_row_size, &rows_per_chunk);
        ধ্রুৱ সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        nrows     = range.গুনতি;
    }

    যদি (nrows == 0) {
        ঘূৰাই HTP_STATUS_OK;
    }

    ধ্রুৱ uint32_t n_threads = octx->n_threads;

    ধ্রুৱ uint32_t ne0 = dst->ne[0];
    ধ্রুৱ size_t src0_row_size   = src0->ne[0] * পরিমাণ(ভিন্ন);
    ধ্রুৱ size_t src0_row_stride = src0->nb[1];
    ধ্রুৱ size_t dst_row_size    = dst->ne[0] * পরিমাণ(ভিন্ন);
    ধ্রুৱ size_t dst_row_stride  = dst->nb[1];

    সংরচনা htp_rope_context rctx;
    memset(&rctx, 0, পরিমাণ(সংরচনা htp_rope_context));

    rctx.octx                  = octx;
    rctx.vtcm_base             = (uint8_t *) octx->ctx->vtcm_base;
    rctx.spad_per_thread       = kparams->spad_per_thread;
    rctx.theta_cache_offset    = kparams->theta_cache_offset;

    যদি (src2) {
        dma_queue * dma_q = octx->ctx->dma[0];
        ধ্রুৱ size_t ff_size = src2->ne[0] * পরিমাণ(ভিন্ন);
        ভিন্ন * vtcm_freq_factors = (ভিন্ন *) (rctx.vtcm_base + kparams->freq_factors_offset);
        dma_queue_ঠেলো(dma_q, dma_make_data(vtcm_freq_factors, src2->data),
                       kparams->freq_factors_size, 0, ff_size, 1);
        dma_queue_বেৰ_কৰো(dma_q);
        rctx.freq_factors = vtcm_freq_factors;
    }

    ধ্রুৱ int32_t * op_params = &octx->op_params[0];
    rctx.n_dims     = ((ধ্রুৱ int32_t *) op_params)[1];
    rctx.mode       = ((ধ্রুৱ int32_t *) op_params)[2];
    rctx.n_ctx_orig = ((ধ্রুৱ int32_t *) op_params)[4];
    rctx.n_offs     = ((ধ্রুৱ int32_t *) op_params)[15];

    memcpy(&rctx.freq_base,   (int32_t *) op_params + 5,  পরিমাণ(ভিন্ন));
    memcpy(&rctx.freq_scale,  (int32_t *) op_params + 6,  পরিমাণ(ভিন্ন));
    memcpy(&rctx.ext_factor,  (int32_t *) op_params + 7,  পরিমাণ(ভিন্ন));
    memcpy(&rctx.attn_factor, (int32_t *) op_params + 8,  পরিমাণ(ভিন্ন));
    memcpy(&rctx.beta_fast,   (int32_t *) op_params + 9,  পরিমাণ(ভিন্ন));
    memcpy(&rctx.beta_slow,   (int32_t *) op_params + 10, পরিমাণ(ভিন্ন));
    memcpy(&rctx.sections,    (int32_t *) op_params + 11, পরিমাণ(পূর্ণাঙ্ক) * 4);

    rctx.theta_scale = powf(rctx.freq_base, -2.0f / rctx.n_dims);
    rctx.theta_powers[0] = 1.0f;
    বাবে (পূর্ণাঙ্ক j = 1; j < 32; j++) {
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

    FARF(উপৰেৰ_2, "rope-f32 n-rows %u n-dims %d ne0 %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n", rctx.src0_nrows, rctx.n_dims, ne0,
         rctx.ext_factor, rctx.theta_scale, rctx.attn_factor);

    work_queue_run(octx->ctx->work_queue, rope_job_f32, &rctx, n_threads);

    ঘূৰাই err;
}

পূর্ণাঙ্ক op_rope(সংরচনা htp_ops_context * octx) {
    ৱিকল্প (octx->src[0]->type) {
        পক্ষ HTP_TYPE_F32:
            ঘূৰাই execute_op_rope_f32(octx);

        স্ৱভাৱ:
            ঘূৰাই HTP_STATUS_NO_SUPPORT;
    }
}
