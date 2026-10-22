#यदि_अपरिभाषितम् HVX_NORM_H
#परिभाषय HVX_NORM_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव "hvx-base.h"
#अन्तर्भाव "hvx-reduce.h"
#अन्तर्भाव "hvx-inverse.h"
#अन्तर्भाव "hvx-sqrt.h"
#अन्तर्भाव "hvx-repl.h"

स्थिर अन्तःस्थ शून्य hvx_fast_rms_norm_f32(ध्रुव uint8_t * सीमित src,
                                         uint8_t * सीमित dst,
                                         ध्रुव पूर्णाङ्क अङ्कः_elems,
                                         भिन्न     epsilon) {

    ध्रुव HVX_सदिशम् * सीमित v_src = (HVX_सदिशम् *) src;
    HVX_सदिशम् * सीमित v_dst       = (HVX_सदिशम् *) dst;

    ध्रुव पूर्णाङ्क nvec = अङ्कः_elems / VLEN_FP32;    // number of full vectors
    ध्रुव पूर्णाङ्क nloe = अङ्कः_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_सदिशम् योग_v = Q6_V_vsplat_R(0x00000000);
    HVX_सदिशम् epsilon_v = hvx_vec_splat_f32(epsilon);

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम् v1 = v_src[i];
        HVX_सदिशम् v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    यदि (nloe > 0) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_सदिशम् v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_सदिशम् v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, v2);
    }

    // Reduce HVX sum
    योग_v = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_v));

    HVX_सदिशम् t_v            = hvx_vec_splat_f32((भिन्न) अङ्कः_elems);
    HVX_सदिशम् denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_सदिशम् माध्यम्_v         = Q6_Vqf32_vmpy_VsfVsf(योग_v, denom_v);
    HVX_सदिशम् माध्यम्_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(माध्यम्_v, epsilon_v);

    // Scale full vectors
    HVX_सदिशम् scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(माध्यम्_epsilon_v));

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम् v1 = v_src[i];
        HVX_सदिशम् v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v2);
    }

    // Handle tail elements using vectorized ops with masking
    यदि (nloe > 0) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_सदिशम् v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_सदिशम् v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_सदिशम् परिणाम = Q6_Vsf_equals_Vqf32(v2);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, परिणाम);
    }
}

स्थिर अन्तःस्थ शून्य hvx_fast_rms_norm_mul_f32(ध्रुव uint8_t * सीमित src,
                                             ध्रुव uint8_t * सीमित weight,
                                             uint8_t * सीमित dst,
                                             ध्रुव पूर्णाङ्क अङ्कः_elems,
                                             भिन्न     epsilon) {
    ध्रुव HVX_सदिशम् * सीमित v_src    = (ध्रुव HVX_सदिशम् *) src;
    ध्रुव HVX_सदिशम् * सीमित v_weight = (ध्रुव HVX_सदिशम् *) weight;
    HVX_सदिशम् * सीमित v_dst          = (HVX_सदिशम् *) dst;

    ध्रुव पूर्णाङ्क nvec = अङ्कः_elems / VLEN_FP32;    // number of full vectors
    ध्रुव पूर्णाङ्क nloe = अङ्कः_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_सदिशम् योग_v = Q6_V_vsplat_R(0x00000000);
    HVX_सदिशम् epsilon_v = hvx_vec_splat_f32(epsilon);

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम् v1 = v_src[i];
        HVX_सदिशम् v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    यदि (nloe > 0) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_सदिशम् v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_सदिशम् v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, v2);
    }

    // Reduce HVX sum
    योग_v = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_v));

    HVX_सदिशम् t_v            = hvx_vec_splat_f32((भिन्न) अङ्कः_elems);
    HVX_सदिशम् denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_सदिशम् माध्यम्_v         = Q6_Vqf32_vmpy_VsfVsf(योग_v, denom_v);
    HVX_सदिशम् माध्यम्_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(माध्यम्_v, epsilon_v);

    // Scale and multiply
    HVX_सदिशम् scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(माध्यम्_epsilon_v));

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम् v1 = v_src[i];
        HVX_सदिशम् v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_सदिशम् v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_सदिशम् परिणाम = Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[i]);
        v_dst[i] = Q6_Vsf_equals_Vqf32(परिणाम);
    }

    // Handle tail elements using vectorized ops with masking
    यदि (nloe > 0) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_सदिशम् v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_सदिशम् v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_सदिशम् v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_सदिशम् परिणाम = Q6_Vqf32_vmpy_VsfVsf(v3, v_weight[nvec]);
        HVX_सदिशम् res_v = Q6_Vsf_equals_Vqf32(परिणाम);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

स्थिर अन्तःस्थ शून्य hvx_fast_norm_f32(ध्रुव uint8_t * सीमित src,
                                     uint8_t * सीमित dst,
                                     ध्रुव पूर्णाङ्क अङ्कः_elems,
                                     भिन्न     epsilon) {

    ध्रुव HVX_सदिशम् * सीमित v_src = (HVX_सदिशम् *) src;
    HVX_सदिशम् * सीमित v_dst       = (HVX_सदिशम् *) dst;

    ध्रुव पूर्णाङ्क nvec = अङ्कः_elems / VLEN_FP32;    // number of full vectors
    ध्रुव पूर्णाङ्क nloe = अङ्कः_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares and sum of values for full vectors
    HVX_सदिशम् योग_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_सदिशम् योग_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_सदिशम् epsilon_v = hvx_vec_splat_f32(epsilon);

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम् v1 = v_src[i];
        HVX_सदिशम् v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_sq_v, v2);
        योग_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(योग_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Handle tail elements using vectorized ops with masking
    यदि (nloe > 0) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_सदिशम् v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_सदिशम् v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_sq_v, v2);
        योग_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(योग_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Reduce HVX sums
    योग_sq_v = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_sq_v));
    योग_x_v  = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_x_v));

    HVX_सदिशम् t_v            = hvx_vec_splat_f32((भिन्न) अङ्कः_elems);
    HVX_सदिशम् denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_सदिशम् माध्यम्_sq_v      = Q6_Vqf32_vmpy_VsfVsf(योग_sq_v, denom_v);
    HVX_सदिशम् माध्यम्_x_v       = Q6_Vqf32_vmpy_VsfVsf(योग_x_v,  denom_v);
    HVX_सदिशम् माध्यम्_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(माध्यम्_x_v), Q6_Vsf_equals_Vqf32(माध्यम्_x_v));
    HVX_सदिशम् var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(माध्यम्_sq_v, माध्यम्_x_sq_v);
    HVX_सदिशम् var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    // scale = rsqrt(variance + epsilon),  mean_x broadcast for subtraction
    HVX_सदिशम् scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_सदिशम् माध्यम्_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(माध्यम्_x_v));

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम् v1 = v_src[i];
        HVX_सदिशम् v2 = Q6_Vqf32_vsub_VsfVsf(v1, माध्यम्_x_b);
        HVX_सदिशम् v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v3);
    }

    // Handle tail elements using vectorized ops with masking
    यदि (nloe > 0) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_सदिशम् v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_सदिशम् v2 = Q6_Vqf32_vsub_VsfVsf(v1, माध्यम्_x_b);
        HVX_सदिशम् v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        HVX_सदिशम् परिणाम = Q6_Vsf_equals_Vqf32(v3);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, परिणाम);
    }
}

स्थिर अन्तःस्थ शून्य hvx_fast_l2_norm_f32(ध्रुव uint8_t * सीमित src,
                                        uint8_t * सीमित dst,
                                        ध्रुव पूर्णाङ्क अङ्कः_elems,
                                        भिन्न     epsilon) {

    ध्रुव HVX_सदिशम् * सीमित v_src = (HVX_सदिशम् *) src;
    HVX_सदिशम् * सीमित v_dst       = (HVX_सदिशम् *) dst;

    HVX_सदिशम् योग_v = hvx_vec_splat_f32(0.0f);

    ध्रुव पूर्णाङ्क nvec = अङ्कः_elems / VLEN_FP32;
    ध्रुव पूर्णाङ्क nloe = अङ्कः_elems % VLEN_FP32;

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम् v1 = v_src[i];
        HVX_सदिशम् sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_v         = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, sq);
    }

    // Include tail elements in the sum-of-squares using a predicate mask
    यदि (nloe > 0) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_सदिशम् v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_सदिशम् sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        योग_v         = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, sq);
    }

    // Compute scale = 1/fmax(sqrt(sum), epsilon) entirely in HVX registers.
    // hvx_vec_rsqrt_f32 + hvx_vec_inverse_f32 avoids scalar extraction.
    HVX_सदिशम् योग_sf    = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_v));
    HVX_सदिशम् rsqrt_v   = hvx_vec_rsqrt_f32(योग_sf);              // 1/sqrt(sum)
    HVX_सदिशम् sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);            // sqrt(sum)
    HVX_सदिशम् epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_सदिशम् denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);  // fmax(sqrt(sum), epsilon)
    HVX_सदिशम् scale_v   = hvx_vec_inverse_f32(denom_v);            // 1/fmax(sqrt(sum), epsilon)

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम् v1 = v_src[i];
        v_dst[i]      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
    }

    यदि (nloe > 0) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_सदिशम् v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_सदिशम् परिणाम = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, परिणाम);
    }
}

// F16 norm kernels: reduce and scale in f32 (via promote/narrow), matching the
// precision-preserving pattern used by the flash-attn f16 kernels.

स्थिर अन्तःस्थ शून्य hvx_fast_rms_norm_f16(ध्रुव uint8_t * सीमित src,
                                         uint8_t * सीमित dst,
                                         ध्रुव पूर्णाङ्क अङ्कः_elems,
                                         भिन्न     epsilon) {

    ध्रुव HVX_सदिशम् * सीमित v_src = (HVX_सदिशम् *) src;
    HVX_सदिशम् * सीमित v_dst       = (HVX_सदिशम् *) dst;

    ध्रुव पूर्णाङ्क nvec = अङ्कः_elems / VLEN_FP16;    // number of full f16 vectors
    ध्रुव पूर्णाङ्क nloe = अङ्कः_elems % VLEN_FP16;    // leftover elements

    HVX_सदिशम् योग_v = Q6_V_vsplat_R(0x00000000);
    HVX_सदिशम् epsilon_v = hvx_vec_splat_f32(epsilon);

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(v_src[i]);
        HVX_सदिशम् p0 = Q6_V_lo_W(p);
        HVX_सदिशम् p1 = Q6_V_hi_W(p);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    यदि (nloe > 0) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_सदिशम् v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(v1);
        HVX_सदिशम् p0 = Q6_V_lo_W(p);
        HVX_सदिशम् p1 = Q6_V_hi_W(p);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    योग_v = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_v));

    HVX_सदिशम् t_v            = hvx_vec_splat_f32((भिन्न) अङ्कः_elems);
    HVX_सदिशम् denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_सदिशम् माध्यम्_v         = Q6_Vqf32_vmpy_VsfVsf(योग_v, denom_v);
    HVX_सदिशम् माध्यम्_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(माध्यम्_v, epsilon_v);

    HVX_सदिशम् scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(माध्यम्_epsilon_v));

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(v_src[i]);
        HVX_सदिशम् r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_सदिशम् r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_गन्तव्यम्_f16(r0, r1);
    }

    यदि (nloe > 0) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_सदिशम् v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(v1);
        HVX_सदिशम् r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_सदिशम् r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_सदिशम् परिणाम = hvx_vec_f32_गन्तव्यम्_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, परिणाम);
    }
}

स्थिर अन्तःस्थ शून्य hvx_fast_norm_f16(ध्रुव uint8_t * सीमित src,
                                     uint8_t * सीमित dst,
                                     ध्रुव पूर्णाङ्क अङ्कः_elems,
                                     भिन्न     epsilon) {

    ध्रुव HVX_सदिशम् * सीमित v_src = (HVX_सदिशम् *) src;
    HVX_सदिशम् * सीमित v_dst       = (HVX_सदिशम् *) dst;

    ध्रुव पूर्णाङ्क nvec = अङ्कः_elems / VLEN_FP16;
    ध्रुव पूर्णाङ्क nloe = अङ्कः_elems % VLEN_FP16;

    HVX_सदिशम् योग_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_सदिशम् योग_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_सदिशम् epsilon_v = hvx_vec_splat_f32(epsilon);

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(v_src[i]);
        HVX_सदिशम् p0 = Q6_V_lo_W(p);
        HVX_सदिशम् p1 = Q6_V_hi_W(p);
        योग_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        योग_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        योग_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(योग_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        योग_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(योग_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    यदि (nloe > 0) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_सदिशम् v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(v1);
        HVX_सदिशम् p0 = Q6_V_lo_W(p);
        HVX_सदिशम् p1 = Q6_V_hi_W(p);
        योग_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        योग_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        योग_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(योग_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        योग_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(योग_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    योग_sq_v = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_sq_v));
    योग_x_v  = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_x_v));

    HVX_सदिशम् t_v            = hvx_vec_splat_f32((भिन्न) अङ्कः_elems);
    HVX_सदिशम् denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_सदिशम् माध्यम्_sq_v      = Q6_Vqf32_vmpy_VsfVsf(योग_sq_v, denom_v);
    HVX_सदिशम् माध्यम्_x_v       = Q6_Vqf32_vmpy_VsfVsf(योग_x_v,  denom_v);
    HVX_सदिशम् माध्यम्_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(माध्यम्_x_v), Q6_Vsf_equals_Vqf32(माध्यम्_x_v));
    HVX_सदिशम् var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(माध्यम्_sq_v, माध्यम्_x_sq_v);
    HVX_सदिशम् var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    HVX_सदिशम् scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_सदिशम् माध्यम्_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(माध्यम्_x_v));

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(v_src[i]);
        HVX_सदिशम् d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), माध्यम्_x_b);
        HVX_सदिशम् d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), माध्यम्_x_b);
        HVX_सदिशम् r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_सदिशम् r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        v_dst[i] = hvx_vec_f32_गन्तव्यम्_f16(r0, r1);
    }

    यदि (nloe > 0) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_सदिशम् v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(v1);
        HVX_सदिशम् d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), माध्यम्_x_b);
        HVX_सदिशम् d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), माध्यम्_x_b);
        HVX_सदिशम् r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_सदिशम् r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        HVX_सदिशम् परिणाम = hvx_vec_f32_गन्तव्यम्_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, परिणाम);
    }
}

स्थिर अन्तःस्थ शून्य hvx_fast_l2_norm_f16(ध्रुव uint8_t * सीमित src,
                                        uint8_t * सीमित dst,
                                        ध्रुव पूर्णाङ्क अङ्कः_elems,
                                        भिन्न     epsilon) {

    ध्रुव HVX_सदिशम् * सीमित v_src = (HVX_सदिशम् *) src;
    HVX_सदिशम् * सीमित v_dst       = (HVX_सदिशम् *) dst;

    ध्रुव पूर्णाङ्क nvec = अङ्कः_elems / VLEN_FP16;
    ध्रुव पूर्णाङ्क nloe = अङ्कः_elems % VLEN_FP16;

    HVX_सदिशम् योग_v = hvx_vec_splat_f32(0.0f);

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(v_src[i]);
        HVX_सदिशम् p0 = Q6_V_lo_W(p);
        HVX_सदिशम् p1 = Q6_V_hi_W(p);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    यदि (nloe > 0) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_सदिशम् v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(v1);
        HVX_सदिशम् p0 = Q6_V_lo_W(p);
        HVX_सदिशम् p1 = Q6_V_hi_W(p);
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        योग_v = Q6_Vqf32_vadd_Vqf32Vqf32(योग_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    HVX_सदिशम् योग_sf    = hvx_vec_reduce_योग_f32(Q6_Vsf_equals_Vqf32(योग_v));
    HVX_सदिशम् rsqrt_v   = hvx_vec_rsqrt_f32(योग_sf);
    HVX_सदिशम् sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);
    HVX_सदिशम् epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_सदिशम् denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);
    HVX_सदिशम् scale_v   = hvx_vec_inverse_f32(denom_v);

    #निर्देशः unroll(4)
    प्रति (पूर्णाङ्क i = 0; i < nvec; i++) {
        HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(v_src[i]);
        HVX_सदिशम् r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_सदिशम् r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_गन्तव्यम्_f16(r0, r1);
    }

    यदि (nloe > 0) {
        HVX_सदिशम्_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_सदिशम् v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_सदिशम्_Pair p = hvx_vec_f16_गन्तव्यम्_f32(v1);
        HVX_सदिशम् r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_सदिशम् r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_सदिशम् परिणाम = hvx_vec_f32_गन्तव्यम्_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, परिणाम);
    }
}

#यदि_अन्तः // HVX_NORM_H
