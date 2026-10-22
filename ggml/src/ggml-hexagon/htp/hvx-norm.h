#నిర్వచితం_కాకపోతే HVX_NORM_H
#నిర్వచించు HVX_NORM_H

#అన్తర్భావ <stdint.h>
#అన్తర్భావ "hvx-base.h"
#అన్తర్భావ "hvx-reduce.h"
#అన్తర్భావ "hvx-inverse.h"
#అన్తర్భావ "hvx-sqrt.h"
#అన్తర్భావ "hvx-repl.h"

స్థిర అంతఃవరుస శూన్య hvx_fast_rms_norm_f32(స్థిరరాశి uint8_t * పరిమిత src,
                                         uint8_t * పరిమిత dst,
                                         స్థిరరాశి పూర్ణాంకం అంకె_elems,
                                         దశాంశం     epsilon) {

    స్థిరరాశి HVX_సదిశ * పరిమిత v_src = (HVX_సదిశ *) src;
    HVX_సదిశ * పరిమిత v_dst       = (HVX_సదిశ *) dst;

    స్థిరరాశి పూర్ణాంకం nvec = అంకె_elems / VLEN_FP32;    // number of full vectors
    స్థిరరాశి పూర్ణాంకం nloe = అంకె_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_సదిశ కూడిక_v = Q6_V_vsplat_R(0x00000000);
    HVX_సదిశ epsilon_v = hvx_vec_splat_f32(epsilon);

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ v1 = v_src[i];
        HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        కూడిక_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_సదిశ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        కూడిక_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_v, v2);
    }

    // Reduce HVX sum
    కూడిక_v = hvx_vec_reduce_కూడిక_f32(Q6_Vsf_equals_Vqf32(కూడిక_v));

    HVX_సదిశ t_v            = hvx_vec_splat_f32((దశాంశం) అంకె_elems);
    HVX_సదిశ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_సదిశ సగటు_v         = Q6_Vqf32_vmpy_VsfVsf(కూడిక_v, denom_v);
    HVX_సదిశ సగటు_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(సగటు_v, epsilon_v);

    // Scale full vectors
    HVX_సదిశ scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(సగటు_epsilon_v));

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ v1 = v_src[i];
        HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v2);
    }

    // Handle tail elements using vectorized ops with masking
    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_సదిశ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_సదిశ ఫలితం = Q6_Vsf_equals_Vqf32(v2);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, ఫలితం);
    }
}

స్థిర అంతఃవరుస శూన్య hvx_fast_rms_norm_mul_f32(స్థిరరాశి uint8_t * పరిమిత src,
                                             స్థిరరాశి uint8_t * పరిమిత బరువు,
                                             uint8_t * పరిమిత dst,
                                             స్థిరరాశి పూర్ణాంకం అంకె_elems,
                                             దశాంశం     epsilon) {
    స్థిరరాశి HVX_సదిశ * పరిమిత v_src    = (స్థిరరాశి HVX_సదిశ *) src;
    స్థిరరాశి HVX_సదిశ * పరిమిత v_బరువు = (స్థిరరాశి HVX_సదిశ *) బరువు;
    HVX_సదిశ * పరిమిత v_dst          = (HVX_సదిశ *) dst;

    స్థిరరాశి పూర్ణాంకం nvec = అంకె_elems / VLEN_FP32;    // number of full vectors
    స్థిరరాశి పూర్ణాంకం nloe = అంకె_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_సదిశ కూడిక_v = Q6_V_vsplat_R(0x00000000);
    HVX_సదిశ epsilon_v = hvx_vec_splat_f32(epsilon);

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ v1 = v_src[i];
        HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        కూడిక_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_సదిశ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        కూడిక_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_v, v2);
    }

    // Reduce HVX sum
    కూడిక_v = hvx_vec_reduce_కూడిక_f32(Q6_Vsf_equals_Vqf32(కూడిక_v));

    HVX_సదిశ t_v            = hvx_vec_splat_f32((దశాంశం) అంకె_elems);
    HVX_సదిశ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_సదిశ సగటు_v         = Q6_Vqf32_vmpy_VsfVsf(కూడిక_v, denom_v);
    HVX_సదిశ సగటు_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(సగటు_v, epsilon_v);

    // Scale and multiply
    HVX_సదిశ scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(సగటు_epsilon_v));

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ v1 = v_src[i];
        HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_సదిశ v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_సదిశ ఫలితం = Q6_Vqf32_vmpy_VsfVsf(v3, v_బరువు[i]);
        v_dst[i] = Q6_Vsf_equals_Vqf32(ఫలితం);
    }

    // Handle tail elements using vectorized ops with masking
    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_సదిశ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_సదిశ v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_సదిశ ఫలితం = Q6_Vqf32_vmpy_VsfVsf(v3, v_బరువు[nvec]);
        HVX_సదిశ res_v = Q6_Vsf_equals_Vqf32(ఫలితం);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

స్థిర అంతఃవరుస శూన్య hvx_fast_norm_f32(స్థిరరాశి uint8_t * పరిమిత src,
                                     uint8_t * పరిమిత dst,
                                     స్థిరరాశి పూర్ణాంకం అంకె_elems,
                                     దశాంశం     epsilon) {

    స్థిరరాశి HVX_సదిశ * పరిమిత v_src = (HVX_సదిశ *) src;
    HVX_సదిశ * పరిమిత v_dst       = (HVX_సదిశ *) dst;

    స్థిరరాశి పూర్ణాంకం nvec = అంకె_elems / VLEN_FP32;    // number of full vectors
    స్థిరరాశి పూర్ణాంకం nloe = అంకె_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares and sum of values for full vectors
    HVX_సదిశ కూడిక_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_సదిశ కూడిక_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_సదిశ epsilon_v = hvx_vec_splat_f32(epsilon);

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ v1 = v_src[i];
        HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        కూడిక_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_sq_v, v2);
        కూడిక_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Handle tail elements using vectorized ops with masking
    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_సదిశ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_సదిశ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        కూడిక_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_sq_v, v2);
        కూడిక_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Reduce HVX sums
    కూడిక_sq_v = hvx_vec_reduce_కూడిక_f32(Q6_Vsf_equals_Vqf32(కూడిక_sq_v));
    కూడిక_x_v  = hvx_vec_reduce_కూడిక_f32(Q6_Vsf_equals_Vqf32(కూడిక_x_v));

    HVX_సదిశ t_v            = hvx_vec_splat_f32((దశాంశం) అంకె_elems);
    HVX_సదిశ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_సదిశ సగటు_sq_v      = Q6_Vqf32_vmpy_VsfVsf(కూడిక_sq_v, denom_v);
    HVX_సదిశ సగటు_x_v       = Q6_Vqf32_vmpy_VsfVsf(కూడిక_x_v,  denom_v);
    HVX_సదిశ సగటు_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(సగటు_x_v), Q6_Vsf_equals_Vqf32(సగటు_x_v));
    HVX_సదిశ var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(సగటు_sq_v, సగటు_x_sq_v);
    HVX_సదిశ var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    // scale = rsqrt(variance + epsilon),  mean_x broadcast for subtraction
    HVX_సదిశ scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_సదిశ సగటు_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(సగటు_x_v));

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ v1 = v_src[i];
        HVX_సదిశ v2 = Q6_Vqf32_vsub_VsfVsf(v1, సగటు_x_b);
        HVX_సదిశ v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v3);
    }

    // Handle tail elements using vectorized ops with masking
    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_సదిశ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_సదిశ v2 = Q6_Vqf32_vsub_VsfVsf(v1, సగటు_x_b);
        HVX_సదిశ v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        HVX_సదిశ ఫలితం = Q6_Vsf_equals_Vqf32(v3);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, ఫలితం);
    }
}

స్థిర అంతఃవరుస శూన్య hvx_fast_l2_norm_f32(స్థిరరాశి uint8_t * పరిమిత src,
                                        uint8_t * పరిమిత dst,
                                        స్థిరరాశి పూర్ణాంకం అంకె_elems,
                                        దశాంశం     epsilon) {

    స్థిరరాశి HVX_సదిశ * పరిమిత v_src = (HVX_సదిశ *) src;
    HVX_సదిశ * పరిమిత v_dst       = (HVX_సదిశ *) dst;

    HVX_సదిశ కూడిక_v = hvx_vec_splat_f32(0.0f);

    స్థిరరాశి పూర్ణాంకం nvec = అంకె_elems / VLEN_FP32;
    స్థిరరాశి పూర్ణాంకం nloe = అంకె_elems % VLEN_FP32;

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ v1 = v_src[i];
        HVX_సదిశ sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        కూడిక_v         = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_v, sq);
    }

    // Include tail elements in the sum-of-squares using a predicate mask
    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_సదిశ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_సదిశ sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        కూడిక_v         = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_v, sq);
    }

    // Compute scale = 1/fmax(sqrt(sum), epsilon) entirely in HVX registers.
    // hvx_vec_rsqrt_f32 + hvx_vec_inverse_f32 avoids scalar extraction.
    HVX_సదిశ కూడిక_sf    = hvx_vec_reduce_కూడిక_f32(Q6_Vsf_equals_Vqf32(కూడిక_v));
    HVX_సదిశ rsqrt_v   = hvx_vec_rsqrt_f32(కూడిక_sf);              // 1/sqrt(sum)
    HVX_సదిశ sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);            // sqrt(sum)
    HVX_సదిశ epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_సదిశ denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);  // fmax(sqrt(sum), epsilon)
    HVX_సదిశ scale_v   = hvx_vec_inverse_f32(denom_v);            // 1/fmax(sqrt(sum), epsilon)

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ v1 = v_src[i];
        v_dst[i]      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
    }

    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_సదిశ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_సదిశ ఫలితం = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, ఫలితం);
    }
}

// F16 norm kernels: reduce and scale in f32 (via promote/narrow), matching the
// precision-preserving pattern used by the flash-attn f16 kernels.

స్థిర అంతఃవరుస శూన్య hvx_fast_rms_norm_f16(స్థిరరాశి uint8_t * పరిమిత src,
                                         uint8_t * పరిమిత dst,
                                         స్థిరరాశి పూర్ణాంకం అంకె_elems,
                                         దశాంశం     epsilon) {

    స్థిరరాశి HVX_సదిశ * పరిమిత v_src = (HVX_సదిశ *) src;
    HVX_సదిశ * పరిమిత v_dst       = (HVX_సదిశ *) dst;

    స్థిరరాశి పూర్ణాంకం nvec = అంకె_elems / VLEN_FP16;    // number of full f16 vectors
    స్థిరరాశి పూర్ణాంకం nloe = అంకె_elems % VLEN_FP16;    // leftover elements

    HVX_సదిశ కూడిక_v = Q6_V_vsplat_R(0x00000000);
    HVX_సదిశ epsilon_v = hvx_vec_splat_f32(epsilon);

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(v_src[i]);
        HVX_సదిశ p0 = Q6_V_lo_W(p);
        HVX_సదిశ p1 = Q6_V_hi_W(p);
        కూడిక_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        కూడిక_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_సదిశ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(v1);
        HVX_సదిశ p0 = Q6_V_lo_W(p);
        HVX_సదిశ p1 = Q6_V_hi_W(p);
        కూడిక_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        కూడిక_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    కూడిక_v = hvx_vec_reduce_కూడిక_f32(Q6_Vsf_equals_Vqf32(కూడిక_v));

    HVX_సదిశ t_v            = hvx_vec_splat_f32((దశాంశం) అంకె_elems);
    HVX_సదిశ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_సదిశ సగటు_v         = Q6_Vqf32_vmpy_VsfVsf(కూడిక_v, denom_v);
    HVX_సదిశ సగటు_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(సగటు_v, epsilon_v);

    HVX_సదిశ scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(సగటు_epsilon_v));

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(v_src[i]);
        HVX_సదిశ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_సదిశ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_గమ్యం_f16(r0, r1);
    }

    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_సదిశ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(v1);
        HVX_సదిశ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_సదిశ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_సదిశ ఫలితం = hvx_vec_f32_గమ్యం_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, ఫలితం);
    }
}

స్థిర అంతఃవరుస శూన్య hvx_fast_norm_f16(స్థిరరాశి uint8_t * పరిమిత src,
                                     uint8_t * పరిమిత dst,
                                     స్థిరరాశి పూర్ణాంకం అంకె_elems,
                                     దశాంశం     epsilon) {

    స్థిరరాశి HVX_సదిశ * పరిమిత v_src = (HVX_సదిశ *) src;
    HVX_సదిశ * పరిమిత v_dst       = (HVX_సదిశ *) dst;

    స్థిరరాశి పూర్ణాంకం nvec = అంకె_elems / VLEN_FP16;
    స్థిరరాశి పూర్ణాంకం nloe = అంకె_elems % VLEN_FP16;

    HVX_సదిశ కూడిక_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_సదిశ కూడిక_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_సదిశ epsilon_v = hvx_vec_splat_f32(epsilon);

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(v_src[i]);
        HVX_సదిశ p0 = Q6_V_lo_W(p);
        HVX_సదిశ p1 = Q6_V_hi_W(p);
        కూడిక_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        కూడిక_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        కూడిక_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        కూడిక_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_సదిశ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(v1);
        HVX_సదిశ p0 = Q6_V_lo_W(p);
        HVX_సదిశ p1 = Q6_V_hi_W(p);
        కూడిక_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        కూడిక_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        కూడిక_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        కూడిక_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    కూడిక_sq_v = hvx_vec_reduce_కూడిక_f32(Q6_Vsf_equals_Vqf32(కూడిక_sq_v));
    కూడిక_x_v  = hvx_vec_reduce_కూడిక_f32(Q6_Vsf_equals_Vqf32(కూడిక_x_v));

    HVX_సదిశ t_v            = hvx_vec_splat_f32((దశాంశం) అంకె_elems);
    HVX_సదిశ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_సదిశ సగటు_sq_v      = Q6_Vqf32_vmpy_VsfVsf(కూడిక_sq_v, denom_v);
    HVX_సదిశ సగటు_x_v       = Q6_Vqf32_vmpy_VsfVsf(కూడిక_x_v,  denom_v);
    HVX_సదిశ సగటు_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(సగటు_x_v), Q6_Vsf_equals_Vqf32(సగటు_x_v));
    HVX_సదిశ var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(సగటు_sq_v, సగటు_x_sq_v);
    HVX_సదిశ var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    HVX_సదిశ scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_సదిశ సగటు_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(సగటు_x_v));

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(v_src[i]);
        HVX_సదిశ d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), సగటు_x_b);
        HVX_సదిశ d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), సగటు_x_b);
        HVX_సదిశ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_సదిశ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        v_dst[i] = hvx_vec_f32_గమ్యం_f16(r0, r1);
    }

    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_సదిశ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(v1);
        HVX_సదిశ d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), సగటు_x_b);
        HVX_సదిశ d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), సగటు_x_b);
        HVX_సదిశ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_సదిశ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        HVX_సదిశ ఫలితం = hvx_vec_f32_గమ్యం_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, ఫలితం);
    }
}

స్థిర అంతఃవరుస శూన్య hvx_fast_l2_norm_f16(స్థిరరాశి uint8_t * పరిమిత src,
                                        uint8_t * పరిమిత dst,
                                        స్థిరరాశి పూర్ణాంకం అంకె_elems,
                                        దశాంశం     epsilon) {

    స్థిరరాశి HVX_సదిశ * పరిమిత v_src = (HVX_సదిశ *) src;
    HVX_సదిశ * పరిమిత v_dst       = (HVX_సదిశ *) dst;

    స్థిరరాశి పూర్ణాంకం nvec = అంకె_elems / VLEN_FP16;
    స్థిరరాశి పూర్ణాంకం nloe = అంకె_elems % VLEN_FP16;

    HVX_సదిశ కూడిక_v = hvx_vec_splat_f32(0.0f);

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(v_src[i]);
        HVX_సదిశ p0 = Q6_V_lo_W(p);
        HVX_సదిశ p1 = Q6_V_hi_W(p);
        కూడిక_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        కూడిక_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_సదిశ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(v1);
        HVX_సదిశ p0 = Q6_V_lo_W(p);
        HVX_సదిశ p1 = Q6_V_hi_W(p);
        కూడిక_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        కూడిక_v = Q6_Vqf32_vadd_Vqf32Vqf32(కూడిక_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    HVX_సదిశ కూడిక_sf    = hvx_vec_reduce_కూడిక_f32(Q6_Vsf_equals_Vqf32(కూడిక_v));
    HVX_సదిశ rsqrt_v   = hvx_vec_rsqrt_f32(కూడిక_sf);
    HVX_సదిశ sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);
    HVX_సదిశ epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_సదిశ denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);
    HVX_సదిశ scale_v   = hvx_vec_inverse_f32(denom_v);

    #సూచన unroll(4)
    కొరకు (పూర్ణాంకం i = 0; i < nvec; i++) {
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(v_src[i]);
        HVX_సదిశ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_సదిశ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_గమ్యం_f16(r0, r1);
    }

    ఒకవేళ (nloe > 0) {
        HVX_సదిశ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_సదిశ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_సదిశ_Pair p = hvx_vec_f16_గమ్యం_f32(v1);
        HVX_సదిశ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_సదిశ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_సదిశ ఫలితం = hvx_vec_f32_గమ్యం_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, ఫలితం);
    }
}

#అయితే_ముగింపు // HVX_NORM_H
