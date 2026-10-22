#pragma clang diagnostic ignored "-Wunused-variable"
#pragma clang diagnostic ignored "-Wunused-function"
#pragma clang diagnostic ignored "-Wunused-but-set-variable"

#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <HAP_perf.h>

#ଅନ୍ତର୍ଭାଵ <math.h>
#ଅନ୍ତର୍ଭାଵ <string.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h>

#ଅନ୍ତର୍ଭାଵ "dma-queue.h"
#ଅନ୍ତର୍ଭାଵ "hvx-utils.h"
#ଅନ୍ତର୍ଭାଵ "hex-fastdiv.h"

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"
#ଅନ୍ତର୍ଭାଵ "rope-ops.h"

// Redefined the rope type constants as we can't include ggml.h
#define HTP_ROPE_TYPE_NORMAL 0
#define HTP_ROPE_TYPE_NEOX   2
#define HTP_ROPE_TYPE_MROPE  8
#define HTP_ROPE_TYPE_VISION 24
#define HTP_ROPE_TYPE_IMROPE 40

#define htp_rope_preamble              \
    ଅଚଳ uint32_t ne00 = src0->ne[0]; \
    ଅଚଳ uint32_t ne01 = src0->ne[1]; \
    ଅଚଳ uint32_t ne02 = src0->ne[2]; \
    ଅଚଳ uint32_t ne03 = src0->ne[3]; \
                                       \
    ଅଚଳ uint32_t ne0 = dst->ne[0];   \
    ଅଚଳ uint32_t ne1 = dst->ne[1];   \
    ଅଚଳ uint32_t ne2 = dst->ne[2];   \
    ଅଚଳ uint32_t ne3 = dst->ne[3];   \
                                       \
    ଅଚଳ uint32_t nb00 = src0->nb[0]; \
    ଅଚଳ uint32_t nb01 = src0->nb[1]; \
    ଅଚଳ uint32_t nb02 = src0->nb[2]; \
    ଅଚଳ uint32_t nb03 = src0->nb[3]; \
                                       \
    ଅଚଳ uint32_t nb0 = dst->nb[0];   \
    ଅଚଳ uint32_t nb1 = dst->nb[1];   \
    ଅଚଳ uint32_t nb2 = dst->nb[2];   \
    ଅଚଳ uint32_t nb3 = dst->nb[3];

ସଂରଚନା htp_rope_context {
    int32_t n_dims;
    int32_t n_offs;
    int32_t mode;
    int32_t n_ctx_orig;
    int32_t sections[4];

    ଦଶମିକ freq_base;
    ଦଶମିକ freq_scale;
    ଦଶମିକ ext_factor;
    ଦଶମିକ attn_factor;
    ଦଶମିକ beta_fast;
    ଦଶମିକ beta_slow;
    ଦଶମିକ theta_scale;
    ଦଶମିକ theta_scale_32;
    ଦଶମିକ theta_powers[32];
    ଦଶମିକ corr_dims[2];

    uint32_t src0_nrows_per_thread;

    ସଂରଚନା htp_ops_context * octx;

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

    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne2_ne1;
    ସଂରଚନା fastdiv_ମୂଲ୍ୟଗୁଡ଼ିକ div_ne1;

    ଅଚଳ ଦଶମିକ * freq_factors;
};

ସ୍ଥିର ଦଶମିକ rope_yarn_ramp(ଅଚଳ ଦଶମିକ ତଳ, ଅଚଳ ଦଶମିକ ଉପର, ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା i0) {
    ଅଚଳ ଦଶମିକ y = (i0 / 2 - ତଳ) / MAX(0.001f, ଉପର - ତଳ);

    ଫେରନ୍ତୁ (1 - MIN(1, MAX(0, y)));
}

// Compute one (cos, sin) pair into cache[i0], cache[i0+1] applying YaRN scaling.
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rope_yarn_one(ଦଶମିକ theta, ଦଶମିକ freq_scale, ଦଶମିକ * corr_dims,
                                 uint32_t i0, ଦଶମିକ ext_factor, ଦଶମିକ mscale,
                                 ଦଶମିକ * cache) {
    ଦଶମିକ theta_extrap = theta;

    // Get n-d rotational scaling corrected for extrapolation
    ଦଶମିକ theta_interp = freq_scale * theta_extrap;
    ଦଶମିକ theta_final  = theta_interp;
    ଦଶମିକ mscale_final = mscale;

    ଯଦି (ext_factor != 0.0f) {
        ଦଶମିକ ramp_mix = rope_yarn_ramp(corr_dims[0], corr_dims[1], i0) * ext_factor;
        theta_final    = theta_interp * (1 - ramp_mix) + theta_extrap * ramp_mix;

        // Get n-d magnitude scaling corrected for interpolation
        mscale_final  *= 1.0f + 0.1f * logf(1.0f / freq_scale);
    }

    ଅଚଳ uint32_t b = i0 / 64;
    ଅଚଳ uint32_t k = (i0 % 64) / 2;
    cache[b * 64 + k]      = cosf(theta_final) * mscale_final;
    cache[b * 64 + 32 + k] = sinf(theta_final) * mscale_final;
}

// 32 thetas -> 32 deinterleaved pairs [cos[32] | sin[32]] at cache[i0].
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ rope_cache_hvx_32(ଦଶମିକ * cache, uint32_t i0,
                                     HVX_Vector v_theta,
                                     ଅଚଳ ଦଶମିକ * freq_factors,
                                     HVX_Vector v_freq_scale,
                                     HVX_Vector v_mscale) {
    ଯଦି (freq_factors) {
        HVX_Vector v_ff = hvx_vmemu(freq_factors + i0 / 2);
        v_theta = hvx_vec_mul_f32_f32(v_theta, hvx_vec_inverse_f32(v_ff));
    }

    HVX_Vector v_theta_final = hvx_vec_mul_f32_f32(v_theta, v_freq_scale);
    HVX_Vector vcos;
    HVX_Vector vsin;
    hvx_vec_sincos_f32(v_theta_final, &vcos, &vsin);
    vcos = hvx_vec_mul_f32_f32(vcos, v_mscale);
    vsin = hvx_vec_mul_f32_f32(vsin, v_mscale);

    ଯଦି (((uintptr_t) (cache + i0)) % 128 == 0) {
        hvx_vmem(cache + i0 + 0)  = vcos;
        hvx_vmem(cache + i0 + 32) = vsin;
    } ଅନ୍ୟଥା {
        hvx_vec_store_u(cache + i0 + 0,  32 * ଆକାର(ଦଶମିକ), vcos);
        hvx_vec_store_u(cache + i0 + 32, 32 * ଆକାର(ଦଶମିକ), vsin);
    }
}

ସ୍ଥିର __attribute__((noinline)) ଶୂନ୍ୟ rope_cache_init(ଅଚଳ ଦଶମିକ    theta_base,
                            ଅଚଳ ଦଶମିକ    freq_scale,
                            ଅଚଳ ଦଶମିକ *  freq_factors,
                            ଦଶମିକ *        corr_dims,
                            ଅଚଳ uint32_t n_cache,
                            ଅଚଳ ଦଶମିକ    ext_factor,
                            ଅଚଳ ଦଶମିକ    mscale,
                            ଦଶମିକ *        cache,
                            ଅଚଳ ଦଶମିକ    theta_scale,
                            ଅଚଳ ଦଶମିକ *  theta_powers,
                            ଅଚଳ ଦଶମିକ    theta_scale_32) {
    // ref: https://github.com/jquesnelle/yarn/blob/master/scaled_rope/LlamaYaRNScaledRotaryEmbedding.py
    ଯଦି (ext_factor == 0.0f) {
        // Fast path: fully vectorized
        // We process 32 pairs (64 elements) per iteration.
        ଅଚଳ uint32_t n_blocks = n_cache / 64;

        HVX_Vector v_theta_powers = hvx_vmemu(theta_powers);
        HVX_Vector v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_Vector v_mscale = hvx_vec_splat_f32(mscale);

        ଦଶମିକ theta_block = theta_base;

        ପାଇଁ (uint32_t b = 0; b < n_blocks; b++) {
            uint32_t i0 = b * 64;
            HVX_Vector v_theta_base = hvx_vec_splat_f32(theta_block);
            HVX_Vector v_theta = hvx_vec_mul_f32_f32(v_theta_base, v_theta_powers);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            theta_block *= theta_scale_32;
        }

        // Leftovers
        ଦଶମିକ theta = theta_block;
        ପାଇଁ (uint32_t i0 = n_blocks * 64; i0 < n_cache; i0 += 2) {
            ଅଚଳ ଦଶମିକ ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta *= theta_scale;
        }
    } ଅନ୍ୟଥା {
        ଦଶମିକ theta = theta_base;
        ପାଇଁ (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
            ଅଚଳ ଦଶମିକ ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta *= theta_scale;
        }
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଦଶମିକ mrope_pick_theta(ଦଶମିକ theta_t, ଦଶମିକ theta_h, ଦଶମିକ theta_w, ଦଶମିକ theta_e,
                                     ପୂର୍ଣ୍ଣସଂଖ୍ୟା sector, ଅଚଳ int32_t sections[4], ପୂର୍ଣ୍ଣସଂଖ୍ୟା sec_w, ପୂର୍ଣ୍ଣସଂଖ୍ୟା sec_e,
                                     bool is_imrope) {
    ଯଦି (is_imrope) {
        ଯଦି      (sector % 3 == 0 && sector < 3 * sections[0]) { ଫେରନ୍ତୁ theta_t; }
        ଅନ୍ୟଥା ଯଦି (sector % 3 == 1 && sector < 3 * sections[1]) { ଫେରନ୍ତୁ theta_h; }
        ଅନ୍ୟଥା ଯଦି (sector % 3 == 2 && sector < 3 * sections[2]) { ଫେରନ୍ତୁ theta_w; }
        ଅନ୍ୟଥା                                                   { ଫେରନ୍ତୁ theta_e; }
    }
    ଯଦି      (sector < sections[0]) { ଫେରନ୍ତୁ theta_t; }
    ଅନ୍ୟଥା ଯଦି (sector < sec_w)       { ଫେରନ୍ତୁ theta_h; }
    ଅନ୍ୟଥା ଯଦି (sector < sec_e)       { ଫେରନ୍ତୁ theta_w; }
    ଅନ୍ୟଥା                           { ଫେରନ୍ତୁ theta_e; }
}

// lane j is 1 when (j % 3) == rem
ସ୍ଥିର ଅଚଳ ଦଶମିକ __attribute__((aligned(128))) mrope_mod3_eq0[32] = {
    1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0
};
ସ୍ଥିର ଅଚଳ ଦଶମିକ __attribute__((aligned(128))) mrope_mod3_eq1[32] = {
    0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1
};
ସ୍ଥିର ଅଚଳ ଦଶମିକ __attribute__((aligned(128))) mrope_mod3_eq2[32] = {
    0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0,1,0,0
};

ସ୍ଥିର ଅଚଳ ଦଶମିକ __attribute__((aligned(128))) mrope_k_ramp[32] = {
    0,1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,
    16,17,18,19,20,21,22,23,24,25,26,27,28,29,30,31
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ HVX_VectorPred mrope_mask_eq1(ଅଚଳ ଦଶମିକ * m) {
    ଫେରନ୍ତୁ Q6_Q_vcmp_gt_VsfVsf(hvx_vmemu(m), Q6_V_vzero());
}

// IMROPE without wrap: theta[k] = pos[k % 3] * scale^k
ସ୍ଥିର ଧାଡ଼ିଭିତରେ HVX_Vector mrope_thetas_imrope_mod3(ଦଶମିକ pos_t, ଦଶମିକ pos_h, ଦଶମିକ pos_w,
                                                  uint32_t k0, HVX_Vector v_powers, ଦଶମିକ scale_block) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା r = (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) (k0 % 3);
    ଅଚଳ ଦଶମିକ * mt = (r == 0) ? mrope_mod3_eq0 : (r == 1) ? mrope_mod3_eq2 : mrope_mod3_eq1;
    ଅଚଳ ଦଶମିକ * mh = (r == 0) ? mrope_mod3_eq1 : (r == 1) ? mrope_mod3_eq0 : mrope_mod3_eq2;

    HVX_Vector v = hvx_vec_splat_f32(pos_w);
    v = Q6_V_vmux_QVV(mrope_mask_eq1(mh), hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(mrope_mask_eq1(mt), hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    ଫେରନ୍ତୁ hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_block));
}

// Contiguous MROPE without wrap: theta[k] = pos[section(k)] * scale^k
ସ୍ଥିର ଧାଡ଼ିଭିତରେ HVX_Vector mrope_thetas_contig(ଦଶମିକ pos_t, ଦଶମିକ pos_h, ଦଶମିକ pos_w, ଦଶମିକ pos_e,
                                             uint32_t k0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା s0, ପୂର୍ଣ୍ଣସଂଖ୍ୟା sec_w, ପୂର୍ଣ୍ଣସଂଖ୍ୟା sec_e,
                                             HVX_Vector v_powers, ଦଶମିକ scale_block) {
    HVX_Vector v_k = hvx_vec_add_f32_f32(hvx_vec_splat_f32((ଦଶମିକ) k0), hvx_vmemu(mrope_k_ramp));
    HVX_VectorPred lt_s0 = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((ଦଶମିକ) s0), v_k);
    HVX_VectorPred lt_sw = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((ଦଶମିକ) sec_w), v_k);
    HVX_VectorPred lt_se = Q6_Q_vcmp_gt_VsfVsf(hvx_vec_splat_f32((ଦଶମିକ) sec_e), v_k);

    HVX_Vector v = hvx_vec_splat_f32(pos_e);
    v = Q6_V_vmux_QVV(lt_se, hvx_vec_splat_f32(pos_w), v);
    v = Q6_V_vmux_QVV(lt_sw, hvx_vec_splat_f32(pos_h), v);
    v = Q6_V_vmux_QVV(lt_s0, hvx_vec_splat_f32(pos_t), v);
    v = hvx_vec_mul_f32_f32(v, v_powers);
    ଫେରନ୍ତୁ hvx_vec_mul_f32_f32(v, hvx_vec_splat_f32(scale_block));
}

// pos_t/h/w/e: the four position ids for this sequence step (t=time, h=height, w=width, e=extra).
// sections[4]: number of head dims assigned to each position component.
ସ୍ଥିର __attribute__((noinline)) ଶୂନ୍ୟ mrope_cache_init(ଅଚଳ ଦଶମିକ    pos_t,
                             ଅଚଳ ଦଶମିକ    pos_h,
                             ଅଚଳ ଦଶମିକ    pos_w,
                             ଅଚଳ ଦଶମିକ    pos_e,
                             ଅଚଳ int32_t  sections[4],
                             ଅଚଳ bool     is_imrope,
                             ଅଚଳ bool     indep_sects,
                             ଅଚଳ ଦଶମିକ    freq_scale,
                             ଅଚଳ ଦଶମିକ *  freq_factors,
                             ଦଶମିକ *        corr_dims,
                             ଅଚଳ uint32_t n_cache,
                             ଅଚଳ ଦଶମିକ    ext_factor,
                             ଅଚଳ ଦଶମିକ    mscale,
                             ଦଶମିକ *        cache,
                             ଅଚଳ ଦଶମିକ    theta_scale,
                             ଅଚଳ ଦଶମିକ *  theta_powers,
                             ଅଚଳ ଦଶମିକ    theta_scale_32) {
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sect_dims = sections[0] + sections[1] + sections[2] + sections[3];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sec_w     = sections[0] + sections[1];
    ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sec_e     = sec_w + sections[2];
    ଅଚଳ uint32_t n_pairs = n_cache / 2;

    ଅଚଳ bool no_wrap = (sect_dims > 0) && (n_pairs <= (uint32_t) sect_dims);
    ଅଚଳ bool imrope_mod3 = is_imrope && !indep_sects && no_wrap
        && sections[0] > 0 && sections[1] > 0 && sections[2] > 0
        && n_pairs <= (uint32_t) (3 * sections[0])
        && n_pairs <= (uint32_t) (3 * sections[1])
        && n_pairs <= (uint32_t) (3 * sections[2]);
    ଅଚଳ bool contig = !is_imrope && !indep_sects && no_wrap;

    ଯଦି (ext_factor == 0.0f && (imrope_mod3 || contig)) {
        HVX_Vector v_powers     = hvx_vmemu(theta_powers);
        HVX_Vector v_freq_scale = hvx_vec_splat_f32(freq_scale);
        HVX_Vector v_mscale     = hvx_vec_splat_f32(mscale);
        ଦଶମିକ scale_block = 1.0f;
        ଅଚଳ uint32_t n_blocks = n_cache / 64;

        ପାଇଁ (uint32_t b = 0; b < n_blocks; b++) {
            ଅଚଳ uint32_t i0 = b * 64;
            ଅଚଳ uint32_t k0 = b * 32;
            HVX_Vector v_theta = imrope_mod3
                ? mrope_thetas_imrope_mod3(pos_t, pos_h, pos_w, k0, v_powers, scale_block)
                : mrope_thetas_contig(pos_t, pos_h, pos_w, pos_e, k0, sections[0], sec_w, sec_e,
                                      v_powers, scale_block);
            rope_cache_hvx_32(cache, i0, v_theta, freq_factors, v_freq_scale, v_mscale);
            scale_block *= theta_scale_32;
        }

        ଦଶମିକ theta_k = scale_block;
        ପାଇଁ (uint32_t k = n_blocks * 32; k < n_pairs; k++) {
            ଅଚଳ uint32_t i0 = 2 * k;
            ଅଚଳ ଦଶମିକ pos = mrope_pick_theta(pos_t, pos_h, pos_w, pos_e,
                                               (ପୂର୍ଣ୍ଣସଂଖ୍ୟା) k, sections, sec_w, sec_e, is_imrope);
            ଅଚଳ ଦଶମିକ ff = freq_factors ? freq_factors[k] : 1.0f;
            rope_yarn_one(pos * theta_k / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
            theta_k *= theta_scale;
        }
        ଫେରନ୍ତୁ;
    }

    ଦଶମିକ theta_t = pos_t;
    ଦଶମିକ theta_h = pos_h;
    ଦଶମିକ theta_w = pos_w;
    ଦଶମିକ theta_e = pos_e;

    ଅଚଳ bool use_hvx = (ext_factor == 0.0f);
    ଦଶମିକ __attribute__((aligned(128))) thetas[32];
    uint32_t n_thetas = 0;
    uint32_t block_i0 = 0;

    HVX_Vector v_freq_scale = hvx_vec_splat_f32(freq_scale);
    HVX_Vector v_mscale     = hvx_vec_splat_f32(mscale);

    ପାଇଁ (uint32_t i0 = 0; i0 < n_cache; i0 += 2) {
        ଅଚଳ ପୂର୍ଣ୍ଣସଂଖ୍ୟା sector = (i0 / 2) % sect_dims;

        ଯଦି (indep_sects) {
            // Reset theta when crossing into a new section.
            ଯଦି      (sector == 0)           { theta_t = pos_t; }
            ଅନ୍ୟଥା ଯଦି (sector == sections[0]) { theta_h = pos_h; }
            ଅନ୍ୟଥା ଯଦି (sector == sec_w)       { theta_w = pos_w; }
            ଅନ୍ୟଥା ଯଦି (sector == sec_e)       { theta_e = pos_e; }
        }

        ଅଚଳ ଦଶମିକ theta = mrope_pick_theta(theta_t, theta_h, theta_w, theta_e,
                                             sector, sections, sec_w, sec_e, is_imrope);

        ଯଦି (use_hvx) {
            ଯଦି (n_thetas == 0) {
                block_i0 = i0;
            }
            thetas[n_thetas++] = theta;
            ଯଦି (n_thetas == 32) {
                rope_cache_hvx_32(cache, block_i0, hvx_vmemu(thetas), freq_factors, v_freq_scale, v_mscale);
                n_thetas = 0;
            }
        } ଅନ୍ୟଥା {
            ଅଚଳ ଦଶମିକ ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
            rope_yarn_one(theta / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
        }

        theta_t *= theta_scale;
        theta_h *= theta_scale;
        theta_w *= theta_scale;
        theta_e *= theta_scale;
    }

    ପାଇଁ (uint32_t k = 0; k < n_thetas; k++) {
        ଅଚଳ uint32_t i0 = block_i0 + 2 * k;
        ଅଚଳ ଦଶମିକ ff = freq_factors ? freq_factors[i0 / 2] : 1.0f;
        rope_yarn_one(thetas[k] / ff, freq_scale, corr_dims, i0, ext_factor, mscale, cache);
    }
}

#define M_PI 3.1415926535897932384626433

ସ୍ଥିର ଶୂନ୍ୟ rope_corr_dims(ପୂର୍ଣ୍ଣସଂଖ୍ୟା     n_dims,
                           ପୂର୍ଣ୍ଣସଂଖ୍ୟା     n_ctx_orig,
                           ଦଶମିକ   freq_base,
                           ଦଶମିକ   beta_fast,
                           ଦଶମିକ   beta_slow,
                           ଦଶମିକ * dims) {
    ଦଶମିକ start = floorf(n_dims * logf(n_ctx_orig / (beta_fast * 2 * (ଦଶମିକ) M_PI)) / (2 * logf(freq_base)));
    ଦଶମିକ end   = ceilf(n_dims * logf(n_ctx_orig / (beta_slow * 2 * (ଦଶମିକ) M_PI)) / (2 * logf(freq_base)));
    dims[0]     = MAX(0, start);
    dims[1]     = MIN(n_dims - 1, end);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_rope_neox_mul(HVX_Vector v0, HVX_Vector v1, HVX_Vector vcos, HVX_Vector vsin,
                                     HVX_Vector * o0, HVX_Vector * o1) {
    HVX_Vector vx0_c = Q6_Vqf32_vmpy_VsfVsf(v0, vcos);
    HVX_Vector vx0_s = Q6_Vqf32_vmpy_VsfVsf(v0, vsin);
    HVX_Vector vx1_c = Q6_Vqf32_vmpy_VsfVsf(v1, vcos);
    HVX_Vector vx1_s = Q6_Vqf32_vmpy_VsfVsf(v1, vsin);
    *o0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vsub_Vqf32Vqf32(vx0_c, vx1_s));
    *o1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vadd_Vqf32Vqf32(vx0_s, vx1_c));
}

// theta_cache full 32-pair blocks are deinterleaved [cos | sin].
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_rope_neox_f32_aa(ଦଶମିକ * restrict dst, ଅଚଳ ଦଶମିକ * restrict src0, uint32_t ne, ଅଚଳ ଦଶମିକ * restrict theta_cache) {
    ଅଚଳ uint32_t he = ne / 2;
    ଅଚଳ uint32_t nvec = he / 32;
    ଅଚଳ uint32_t nloe = he % 32;

    ଯଦି (nloe == 0) {
        ଅଚଳ HVX_Vector * vs = (ଅଚଳ HVX_Vector *) src0;
        ଅଚଳ HVX_Vector * vt = (ଅଚଳ HVX_Vector *) theta_cache;
        HVX_Vector * vd = (HVX_Vector *) dst;
        ପାଇଁ (uint32_t i = 0; i < nvec; i++) {
            HVX_Vector o0, o1;
            hvx_rope_neox_mul(vs[i], vs[nvec + i], vt[i * 2 + 0], vt[i * 2 + 1], &o0, &o1);
            vd[i] = o0;
            vd[nvec + i] = o1;
        }
        ଫେରନ୍ତୁ;
    }

    ପାଇଁ (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector o0, o1;
        hvx_rope_neox_mul(((ଅଚଳ HVX_Vector *) src0)[i],
                          hvx_vmemu(src0 + he + i * 32),
                          ((ଅଚଳ HVX_Vector *) theta_cache)[i * 2 + 0],
                          ((ଅଚଳ HVX_Vector *) theta_cache)[i * 2 + 1],
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
    hvx_vec_store_u(dst + nvec * 32, nloe * ଆକାର(ଦଶମିକ), o0);
    hvx_vec_store_u(dst + he + nvec * 32, nloe * ଆକାର(ଦଶମିକ), o1);
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ hvx_rope_f32_aa(ଦଶମିକ * restrict dst, ଅଚଳ ଦଶମିକ * restrict src0, uint32_t ne, ଅଚଳ ଦଶମିକ * restrict theta_cache) {
    ଅଚଳ uint32_t nvec = ne / 64;
    ଅଚଳ uint32_t nloe = ne % 64;

    ପାଇଁ (uint32_t i = 0; i < nvec; i++) {
        HVX_Vector v0 = ((ଅଚଳ HVX_Vector *) src0)[i * 2 + 0];
        HVX_Vector v1 = ((ଅଚଳ HVX_Vector *) src0)[i * 2 + 1];

        HVX_Vector vcos = ((ଅଚଳ HVX_Vector *) theta_cache)[i * 2 + 0];
        HVX_Vector vsin = ((ଅଚଳ HVX_Vector *) theta_cache)[i * 2 + 1];

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

    ଯଦି (nloe > 0) {
        ଯଦି (nloe <= 32) {
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

            hvx_vec_store_u(dst + nvec * 64, nloe * ଆକାର(ଦଶମିକ), Q6_V_lo_W(vstore));
        } ଅନ୍ୟଥା {
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
            hvx_vec_store_u(dst + nvec * 64 + 32, (nloe - 32) * ଆକାର(ଦଶମିକ), Q6_V_hi_W(vstore));
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ଧାଡ଼ିଭିତରେ rope_basic_f32_inplace(ସଂରଚନା htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, ଅଚଳ ଦଶମିକ * restrict theta_cache) {
    ଅଚଳ uint32_t n_offs = rctx->n_offs;
    #pragma unroll(4)
    ପାଇଁ (uint32_t i = 0; i < nr; i++) {
        ଦଶମିକ * s = (ଦଶମିକ *) (src + i * rctx->src0_row_size_aligned);
        hvx_rope_f32_aa(s + n_offs, s + n_offs, rctx->n_dims, theta_cache);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ଧାଡ଼ିଭିତରେ rope_neox_f32_inplace(ସଂରଚନା htp_rope_context * rctx, uint8_t * src,
                   uint32_t nr, uint32_t ne, ଅଚଳ ଦଶମିକ * restrict theta_cache) {
    ଅଚଳ uint32_t n_offs = rctx->n_offs;
    #pragma unroll(4)
    ପାଇଁ (uint32_t i = 0; i < nr; i++) {
        ଦଶମିକ * s = (ଦଶମିକ *) (src + i * rctx->src0_row_size_aligned);
        hvx_rope_neox_f32_aa(s + n_offs, s + n_offs, ne, theta_cache);
    }
}

ସ୍ଥିର ଶୂନ୍ୟ rope_job_f32(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା nth, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ସଂରଚନା htp_rope_context * rctx = (ସଂରଚନା htp_rope_context *) data;
    ସଂରଚନା htp_ops_context * octx = rctx->octx;

    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * src1 = octx->src[1];
    ଅଚଳ ସଂରଚନା htp_tensor * src2 = octx->src[2];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

    htp_rope_preamble;

    ଅଚଳ uint32_t src0_nrows = rctx->nrows;
    ଅଚଳ uint32_t src0_nrows_per_thread = rctx->src0_nrows_per_thread;

    ଅଚଳ uint32_t src0_start_row = rctx->row_start + src0_nrows_per_thread * ith;
    ଅଚଳ uint32_t src0_end_row   = MIN(src0_start_row + src0_nrows_per_thread, rctx->row_start + src0_nrows);

    // no work for this thread
    ଯଦି (src0_start_row >= src0_end_row) {
        ଫେରନ୍ତୁ;
    }

    ଅଚଳ int32_t mode    = rctx->mode;
    // MROPE, IMROPE and VISION use NEOX-style pairing for the rotation
    ଅଚଳ bool    is_neox = (mode & HTP_ROPE_TYPE_NEOX) || (mode & HTP_ROPE_TYPE_MROPE);
    ଅଚଳ bool    is_vision = (mode == HTP_ROPE_TYPE_VISION);

    // VTCM setup
    uint8_t * src0_spad_base = rctx->vtcm_base + (ith * rctx->spad_per_thread);
    ଦଶମିକ *   theta_cache    = (ଦଶମିକ *) (src0_spad_base);
              src0_spad_base = src0_spad_base + rctx->theta_cache_offset;

    dma_queue * dma_q = octx->ctx->dma[ith];
    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];
    ଅଚଳ int32_t * pos = (ଅଚଳ int32_t *) (uintptr_t) src1->data;
    ଅଚଳ ଦଶମିକ * freq_factors = rctx->freq_factors;

    ଅଚଳ uint32_t i3_start = fastdiv(src0_start_row, &rctx->div_ne2_ne1);
    ଅଚଳ uint32_t rem      = fastmodulo(src0_start_row, ne2 * ne1, &rctx->div_ne2_ne1);
    ଅଚଳ uint32_t i2_start = fastdiv(rem, &rctx->div_ne1);
    ଅଚଳ uint32_t i1_start = fastmodulo(rem, ne1, &rctx->div_ne1);

    uint32_t ir = src0_start_row;
    uint32_t prev_i2 = (uint32_t) -1;
    uint32_t cur_slot = 0;

    ପାଇଁ (uint32_t i3 = i3_start; i3 < ne3; i3++) { // batch
        ଅଚଳ uint32_t i2_init = (i3 == i3_start) ? i2_start : 0;
        ପାଇଁ (uint32_t i2 = i2_init; i2 < ne2; i2++) { // seq-len
            ଅଚଳ uint32_t i1_init = (i3 == i3_start && i2 == i2_start) ? i1_start : 0;
            ପାଇଁ (uint32_t i1 = i1_init; i1 < ne1; ) { // attn-heads
                ଯଦି (ir >= src0_end_row) ଯାଅ done;

                // Rows in this block
                ଅଚଳ uint32_t nrows = MIN(src0_end_row - ir, ne1 - i1);

                // Depth before prefetch
                ଅଚଳ uint32_t dma_depth = dma_queue_depth(dma_q);

                // Prefetch up to 2 blocks
                ଅଚଳ uint32_t p_nrows = MIN(nrows, 2 * HTP_ROPE_SPAD_BLOCK);
                ପାଇଁ (uint32_t pr = 0; pr < p_nrows; pr += HTP_ROPE_SPAD_BLOCK) {
                    ଅଚଳ uint32_t pnr = MIN(nrows - pr, HTP_ROPE_SPAD_BLOCK);
                    ଅଚଳ uint32_t slot = (cur_slot + pr / HTP_ROPE_SPAD_BLOCK) % HTP_ROPE_SPAD_NSLOTS;
                    uint8_t * spad_slot = rope_spad_slot(src0_spad_base, slot, rctx->src0_row_size_aligned);
                    ଅଚଳ dma_addr_t src0_data = src0->data + i3 * nb03 + i2 * nb02 + (i1 + pr) * nb01;

                    // Dummy DMA transaction for sequencing (interleaving wr, rd, wr, rd, ...)
                    dma_queue_ଠେଲ(dma_q, dma_make_data(dst->data, spad_slot), 0, 0, 0, 0);

                    dma_queue_ଠେଲ(dma_q, dma_make_data(spad_slot, src0_data),
                        rctx->src0_row_size_aligned, rctx->src0_row_stride, rctx->src0_row_size, pnr);
                }

                // Update theta cache
                ଯଦି (i2 != prev_i2) {
                    prev_i2 = i2;

                    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                    // VISION rotates the full row; other modes only rotate n_dims.
                    ଅଚଳ uint32_t n_cache = is_vision ? ne0 : (uint32_t) rctx->n_dims;
                    ଅଚଳ bool is_mrope = (rctx->mode & HTP_ROPE_TYPE_MROPE) != 0;
                    ଯଦି (is_mrope) {
                        // src1 holds four position arrays stacked along ne0:
                        // pos[i2], pos[i2+ne2], pos[i2+ne2*2], pos[i2+ne2*3]
                        ଅଚଳ bool is_imrope = (rctx->mode == HTP_ROPE_TYPE_IMROPE);
                        mrope_cache_init(
                            (ଦଶମିକ) pos[i2],
                            (ଦଶମିକ) pos[i2 + ne2],
                            (ଦଶମିକ) pos[i2 + ne2 * 2],
                            (ଦଶମିକ) pos[i2 + ne2 * 3],
                            rctx->sections, is_imrope, is_vision,
                            rctx->freq_scale, freq_factors, rctx->corr_dims,
                            n_cache, rctx->ext_factor, rctx->attn_factor,
                            theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    } ଅନ୍ୟଥା {
                       rope_cache_init(pos[i2], rctx->freq_scale, freq_factors, rctx->corr_dims,
                                        n_cache, rctx->ext_factor, rctx->attn_factor,
                                        theta_cache, rctx->theta_scale, rctx->theta_powers, rctx->theta_scale_32);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_A_PREP, i2);
                }

                // Skip output DMA transactions from prev block (if any)
                ପାଇଁ (uint32_t d = 0; d < dma_depth; d++) { dma_queue_କାଢ଼_nowait(dma_q); }

                // Compute loop
                ଅଚଳ uint32_t ne = is_vision ? ne0 : rctx->n_dims;
                ଅଚଳ uint32_t base_i1 = i1;
                ଅଚଳ uint32_t base_ir = ir;

                ପାଇଁ (uint32_t cnr = 0, cr = 0; cr < nrows; cr += cnr) {
                    cnr = MIN(nrows - cr, HTP_ROPE_SPAD_BLOCK);
                    ଅଚଳ uint32_t slot = (cur_slot + cr / HTP_ROPE_SPAD_BLOCK) % HTP_ROPE_SPAD_NSLOTS;
                    ଅଚଳ uint32_t cur_ir   = base_ir + cr;
                    ଅଚଳ uint32_t cur_i1   = base_i1 + cr;

                    dma_queue_କାଢ଼(dma_q);
                    uint8_t * cur_spad = (uint8_t *) dma_queue_କାଢ଼(dma_q).dst;

                    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);
                    ଯଦି (is_neox || is_vision) {
                        rope_neox_f32_inplace(rctx, cur_spad, cnr, ne, theta_cache);
                    } ଅନ୍ୟଥା {
                        rope_basic_f32_inplace(rctx, cur_spad, cnr, theta_cache);
                    }
                    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, cur_ir);

                    ଅଚଳ dma_addr_t dst_data = dst->data + i3 * nb3 + i2 * nb2 + cur_i1 * nb1;
                    dma_queue_ଠେଲ(dma_q, dma_make_data(dst_data, cur_spad),
                        rctx->dst_row_stride, rctx->src0_row_size_aligned, rctx->dst_row_size, cnr);

                    // Prefetch 2 blocks ahead into the slot just freed
                    ଯଦି ((cr + 2 * HTP_ROPE_SPAD_BLOCK) < nrows) {
                        ଅଚଳ uint32_t p_cr   = cr + 2 * HTP_ROPE_SPAD_BLOCK;
                        ଅଚଳ uint32_t pnr    = MIN(nrows - p_cr, HTP_ROPE_SPAD_BLOCK);
                        ଅଚଳ uint32_t p_slot = (cur_slot + p_cr / HTP_ROPE_SPAD_BLOCK) % HTP_ROPE_SPAD_NSLOTS;
                        uint8_t * p_spad      = rope_spad_slot(src0_spad_base, p_slot, rctx->src0_row_size_aligned);
                        ଅଚଳ dma_addr_t p_src0_data = src0->data + i3 * nb03 + i2 * nb02 + (base_i1 + p_cr) * nb01;

                        dma_queue_ଠେଲ(dma_q, dma_make_data(p_spad, p_src0_data),
                            rctx->src0_row_size_aligned, rctx->src0_row_stride, rctx->src0_row_size, pnr);
                    }
                }

                ଅଚଳ uint32_t n_chunks = (nrows + HTP_ROPE_SPAD_BLOCK - 1) / HTP_ROPE_SPAD_BLOCK;
                cur_slot = (cur_slot + n_chunks) % HTP_ROPE_SPAD_NSLOTS;

                ir += nrows;
                i1 += nrows;
            }
        }
    }

done:
    dma_queue_flush(dma_q);

    FARF(ଉପର_2, "rope-f32: %d/%d: (%u:%u)\n", ith, nth, src0_start_row, src0_end_row);
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା execute_op_rope_f32(ସଂରଚନା htp_ops_context * octx) {
    ପୂର୍ଣ୍ଣସଂଖ୍ୟା err = HTP_STATUS_OK;

    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * src1 = octx->src[1];
    ଅଚଳ ସଂରଚନା htp_tensor * src2 = octx->src[2];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

    ବାଛ (octx->op) {
        ସ୍ଥିତି HTP_OP_ROPE:
            ଭାଙ୍ଗ;

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            FARF(ERROR, "Unsupported Op %u\n", octx->op);
            ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଅଚଳ ସଂରଚନା htp_rope_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_rope_kernel_params *) octx->kernel_params;
    ଯଦି (!htp_ops_context_set_n_threads(octx, kparams->n_threads)) {
        ଫେରନ୍ତୁ HTP_STATUS_INVAL_PARAMS;
    }
    ପ୍ରତିଜ୍ଞା(octx->ctx->vtcm_size >= kparams->vtcm_size);

    ଯଦି (htp_tensor_is_extended(src1)) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଅଚଳ uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ଅଚଳ size_t dst_data_row_size = dst->ne[0] * ଆକାର(ଦଶମିକ);

    uint32_t row_start = 0;
    uint32_t nrows     = total_rows;

    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, ଆକାର(ଦଶମିକ), (uint32_t) dst_data_row_size, &rows_per_chunk);
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        row_start = range.start;
        nrows     = range.ଗଣନା_2;
    }

    ଯଦି (nrows == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ଅଚଳ uint32_t n_threads = octx->n_threads;

    ଅଚଳ uint32_t ne0 = dst->ne[0];
    ଅଚଳ size_t src0_row_size   = src0->ne[0] * ଆକାର(ଦଶମିକ);
    ଅଚଳ size_t src0_row_stride = src0->nb[1];
    ଅଚଳ size_t dst_row_size    = dst->ne[0] * ଆକାର(ଦଶମିକ);
    ଅଚଳ size_t dst_row_stride  = dst->nb[1];

    ସଂରଚନା htp_rope_context rctx;
    memset(&rctx, 0, ଆକାର(ସଂରଚନା htp_rope_context));

    rctx.octx                  = octx;
    rctx.vtcm_base             = (uint8_t *) octx->ctx->vtcm_base;
    rctx.spad_per_thread       = kparams->spad_per_thread;
    rctx.theta_cache_offset    = kparams->theta_cache_offset;

    ଯଦି (src2) {
        dma_queue * dma_q = octx->ctx->dma[0];
        ଅଚଳ size_t ff_size = src2->ne[0] * ଆକାର(ଦଶମିକ);
        ଦଶମିକ * vtcm_freq_factors = (ଦଶମିକ *) (rctx.vtcm_base + kparams->freq_factors_offset);
        dma_queue_ଠେଲ(dma_q, dma_make_data(vtcm_freq_factors, src2->data),
                       kparams->freq_factors_size, 0, ff_size, 1);
        dma_queue_କାଢ଼(dma_q);
        rctx.freq_factors = vtcm_freq_factors;
    }

    ଅଚଳ int32_t * op_params = &octx->op_params[0];
    rctx.n_dims     = ((ଅଚଳ int32_t *) op_params)[1];
    rctx.mode       = ((ଅଚଳ int32_t *) op_params)[2];
    rctx.n_ctx_orig = ((ଅଚଳ int32_t *) op_params)[4];
    rctx.n_offs     = ((ଅଚଳ int32_t *) op_params)[15];

    memcpy(&rctx.freq_base,   (int32_t *) op_params + 5,  ଆକାର(ଦଶମିକ));
    memcpy(&rctx.freq_scale,  (int32_t *) op_params + 6,  ଆକାର(ଦଶମିକ));
    memcpy(&rctx.ext_factor,  (int32_t *) op_params + 7,  ଆକାର(ଦଶମିକ));
    memcpy(&rctx.attn_factor, (int32_t *) op_params + 8,  ଆକାର(ଦଶମିକ));
    memcpy(&rctx.beta_fast,   (int32_t *) op_params + 9,  ଆକାର(ଦଶମିକ));
    memcpy(&rctx.beta_slow,   (int32_t *) op_params + 10, ଆକାର(ଦଶମିକ));
    memcpy(&rctx.sections,    (int32_t *) op_params + 11, ଆକାର(ପୂର୍ଣ୍ଣସଂଖ୍ୟା) * 4);

    rctx.theta_scale = powf(rctx.freq_base, -2.0f / rctx.n_dims);
    rctx.theta_powers[0] = 1.0f;
    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = 1; j < 32; j++) {
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

    FARF(ଉପର_2, "rope-f32 n-rows %u n-dims %d ne0 %u ext-factor %.6f theta-scale %.6f attn-factor %.6f\n", rctx.src0_nrows, rctx.n_dims, ne0,
         rctx.ext_factor, rctx.theta_scale, rctx.attn_factor);

    work_queue_run(octx->ctx->work_queue, rope_job_f32, &rctx, n_threads);

    ଫେରନ୍ତୁ err;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_rope(ସଂରଚନା htp_ops_context * octx) {
    ବାଛ (octx->src[0]->type) {
        ସ୍ଥିତି HTP_TYPE_F32:
            ଫେରନ୍ତୁ execute_op_rope_f32(octx);

        ପୂର୍ବନିର୍ଦ୍ଧାରିତ:
            ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }
}
