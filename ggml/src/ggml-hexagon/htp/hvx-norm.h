#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_NORM_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_NORM_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ "hvx-base.h"
#ಅನ್ತರ್ಭಾವ "hvx-reduce.h"
#ಅನ್ತರ್ಭಾವ "hvx-inverse.h"
#ಅನ್ತರ್ಭಾವ "hvx-sqrt.h"
#ಅನ್ತರ್ಭಾವ "hvx-repl.h"

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_fast_rms_norm_f32(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src,
                                         uint8_t * ಸೀಮಿತ dst,
                                         ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems,
                                         ದಶಮಾಂಶ     epsilon) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src = (HVX_ಸದಿಶ *) src;
    HVX_ಸದಿಶ * ಸೀಮಿತ v_dst       = (HVX_ಸದಿಶ *) dst;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nvec = ಅಂಕೆ_elems / VLEN_FP32;    // number of full vectors
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nloe = ಅಂಕೆ_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_ಸದಿಶ ಮೊತ್ತ_v = Q6_V_vsplat_R(0x00000000);
    HVX_ಸದಿಶ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v1 = v_src[i];
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ಮೊತ್ತ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ಸದಿಶ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ಮೊತ್ತ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_v, v2);
    }

    // Reduce HVX sum
    ಮೊತ್ತ_v = hvx_vec_reduce_ಮೊತ್ತ_f32(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_v));

    HVX_ಸದಿಶ t_v            = hvx_vec_splat_f32((ದಶಮಾಂಶ) ಅಂಕೆ_elems);
    HVX_ಸದಿಶ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ಸದಿಶ ಸರಾಸರಿ_v         = Q6_Vqf32_vmpy_VsfVsf(ಮೊತ್ತ_v, denom_v);
    HVX_ಸದಿಶ ಸರಾಸರಿ_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(ಸರಾಸರಿ_v, epsilon_v);

    // Scale full vectors
    HVX_ಸದಿಶ scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(ಸರಾಸರಿ_epsilon_v));

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v1 = v_src[i];
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v2);
    }

    // Handle tail elements using vectorized ops with masking
    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ಸದಿಶ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_ಸದಿಶ ಫಲಿತಾಂಶ = Q6_Vsf_equals_Vqf32(v2);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, ಫಲಿತಾಂಶ);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_fast_rms_norm_mul_f32(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src,
                                             ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ ತೂಕ,
                                             uint8_t * ಸೀಮಿತ dst,
                                             ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems,
                                             ದಶಮಾಂಶ     epsilon) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src    = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) src;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_ತೂಕ = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) ತೂಕ;
    HVX_ಸದಿಶ * ಸೀಮಿತ v_dst          = (HVX_ಸದಿಶ *) dst;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nvec = ಅಂಕೆ_elems / VLEN_FP32;    // number of full vectors
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nloe = ಅಂಕೆ_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares for full vectors
    HVX_ಸದಿಶ ಮೊತ್ತ_v = Q6_V_vsplat_R(0x00000000);
    HVX_ಸದಿಶ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v1 = v_src[i];
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ಮೊತ್ತ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_v, v2);
    }

    // Handle tail elements using vectorized ops with masking
    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ಸದಿಶ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ಮೊತ್ತ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_v, v2);
    }

    // Reduce HVX sum
    ಮೊತ್ತ_v = hvx_vec_reduce_ಮೊತ್ತ_f32(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_v));

    HVX_ಸದಿಶ t_v            = hvx_vec_splat_f32((ದಶಮಾಂಶ) ಅಂಕೆ_elems);
    HVX_ಸದಿಶ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ಸದಿಶ ಸರಾಸರಿ_v         = Q6_Vqf32_vmpy_VsfVsf(ಮೊತ್ತ_v, denom_v);
    HVX_ಸದಿಶ ಸರಾಸರಿ_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(ಸರಾಸರಿ_v, epsilon_v);

    // Scale and multiply
    HVX_ಸದಿಶ scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(ಸರಾಸರಿ_epsilon_v));

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v1 = v_src[i];
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_ಸದಿಶ v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_ಸದಿಶ ಫಲಿತಾಂಶ = Q6_Vqf32_vmpy_VsfVsf(v3, v_ತೂಕ[i]);
        v_dst[i] = Q6_Vsf_equals_Vqf32(ಫಲಿತಾಂಶ);
    }

    // Handle tail elements using vectorized ops with masking
    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ಸದಿಶ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, scale_v);
        HVX_ಸದಿಶ v3 = Q6_Vsf_equals_Vqf32(v2);
        HVX_ಸದಿಶ ಫಲಿತಾಂಶ = Q6_Vqf32_vmpy_VsfVsf(v3, v_ತೂಕ[nvec]);
        HVX_ಸದಿಶ res_v = Q6_Vsf_equals_Vqf32(ಫಲಿತಾಂಶ);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, res_v);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_fast_norm_f32(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src,
                                     uint8_t * ಸೀಮಿತ dst,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems,
                                     ದಶಮಾಂಶ     epsilon) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src = (HVX_ಸದಿಶ *) src;
    HVX_ಸದಿಶ * ಸೀಮಿತ v_dst       = (HVX_ಸದಿಶ *) dst;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nvec = ಅಂಕೆ_elems / VLEN_FP32;    // number of full vectors
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nloe = ಅಂಕೆ_elems % VLEN_FP32;    // leftover elements

    // Compute sum of squares and sum of values for full vectors
    HVX_ಸದಿಶ ಮೊತ್ತ_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_ಸದಿಶ ಮೊತ್ತ_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_ಸದಿಶ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v1 = v_src[i];
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ಮೊತ್ತ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_sq_v, v2);
        ಮೊತ್ತ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Handle tail elements using vectorized ops with masking
    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ಸದಿಶ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ಮೊತ್ತ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_sq_v, v2);
        ಮೊತ್ತ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_x_v,  Q6_Vqf32_vadd_VsfVsf(v1, Q6_V_vzero()));
    }

    // Reduce HVX sums
    ಮೊತ್ತ_sq_v = hvx_vec_reduce_ಮೊತ್ತ_f32(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_sq_v));
    ಮೊತ್ತ_x_v  = hvx_vec_reduce_ಮೊತ್ತ_f32(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_x_v));

    HVX_ಸದಿಶ t_v            = hvx_vec_splat_f32((ದಶಮಾಂಶ) ಅಂಕೆ_elems);
    HVX_ಸದಿಶ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ಸದಿಶ ಸರಾಸರಿ_sq_v      = Q6_Vqf32_vmpy_VsfVsf(ಮೊತ್ತ_sq_v, denom_v);
    HVX_ಸದಿಶ ಸರಾಸರಿ_x_v       = Q6_Vqf32_vmpy_VsfVsf(ಮೊತ್ತ_x_v,  denom_v);
    HVX_ಸದಿಶ ಸರಾಸರಿ_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(ಸರಾಸರಿ_x_v), Q6_Vsf_equals_Vqf32(ಸರಾಸರಿ_x_v));
    HVX_ಸದಿಶ var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(ಸರಾಸರಿ_sq_v, ಸರಾಸರಿ_x_sq_v);
    HVX_ಸದಿಶ var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    // scale = rsqrt(variance + epsilon),  mean_x broadcast for subtraction
    HVX_ಸದಿಶ scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_ಸದಿಶ ಸರಾಸರಿ_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(ಸರಾಸರಿ_x_v));

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v1 = v_src[i];
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vsub_VsfVsf(v1, ಸರಾಸರಿ_x_b);
        HVX_ಸದಿಶ v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        v_dst[i] = Q6_Vsf_equals_Vqf32(v3);
    }

    // Handle tail elements using vectorized ops with masking
    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ಸದಿಶ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ಸದಿಶ v2 = Q6_Vqf32_vsub_VsfVsf(v1, ಸರಾಸರಿ_x_b);
        HVX_ಸದಿಶ v3 = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(v2), scale_v);
        HVX_ಸದಿಶ ಫಲಿತಾಂಶ = Q6_Vsf_equals_Vqf32(v3);

        // Store with masking to avoid overwriting memory beyond the tensor
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, ಫಲಿತಾಂಶ);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_fast_l2_norm_f32(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src,
                                        uint8_t * ಸೀಮಿತ dst,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems,
                                        ದಶಮಾಂಶ     epsilon) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src = (HVX_ಸದಿಶ *) src;
    HVX_ಸದಿಶ * ಸೀಮಿತ v_dst       = (HVX_ಸದಿಶ *) dst;

    HVX_ಸದಿಶ ಮೊತ್ತ_v = hvx_vec_splat_f32(0.0f);

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nvec = ಅಂಕೆ_elems / VLEN_FP32;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nloe = ಅಂಕೆ_elems % VLEN_FP32;

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v1 = v_src[i];
        HVX_ಸದಿಶ sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ಮೊತ್ತ_v         = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_v, sq);
    }

    // Include tail elements in the sum-of-squares using a predicate mask
    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ಸದಿಶ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ಸದಿಶ sq = Q6_Vqf32_vmpy_VsfVsf(v1, v1);
        ಮೊತ್ತ_v         = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_v, sq);
    }

    // Compute scale = 1/fmax(sqrt(sum), epsilon) entirely in HVX registers.
    // hvx_vec_rsqrt_f32 + hvx_vec_inverse_f32 avoids scalar extraction.
    HVX_ಸದಿಶ ಮೊತ್ತ_sf    = hvx_vec_reduce_ಮೊತ್ತ_f32(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_v));
    HVX_ಸದಿಶ rsqrt_v   = hvx_vec_rsqrt_f32(ಮೊತ್ತ_sf);              // 1/sqrt(sum)
    HVX_ಸದಿಶ sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);            // sqrt(sum)
    HVX_ಸದಿಶ epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_ಸದಿಶ denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);  // fmax(sqrt(sum), epsilon)
    HVX_ಸದಿಶ scale_v   = hvx_vec_inverse_f32(denom_v);            // 1/fmax(sqrt(sum), epsilon)

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ v1 = v_src[i];
        v_dst[i]      = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
    }

    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * 4);
        HVX_ಸದಿಶ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ಸದಿಶ ಫಲಿತಾಂಶ = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(v1, scale_v));
        hvx_vec_store_a(&v_dst[nvec], nloe * 4, ಫಲಿತಾಂಶ);
    }
}

// F16 norm kernels: reduce and scale in f32 (via promote/narrow), matching the
// precision-preserving pattern used by the flash-attn f16 kernels.

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_fast_rms_norm_f16(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src,
                                         uint8_t * ಸೀಮಿತ dst,
                                         ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems,
                                         ದಶಮಾಂಶ     epsilon) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src = (HVX_ಸದಿಶ *) src;
    HVX_ಸದಿಶ * ಸೀಮಿತ v_dst       = (HVX_ಸದಿಶ *) dst;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nvec = ಅಂಕೆ_elems / VLEN_FP16;    // number of full f16 vectors
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nloe = ಅಂಕೆ_elems % VLEN_FP16;    // leftover elements

    HVX_ಸದಿಶ ಮೊತ್ತ_v = Q6_V_vsplat_R(0x00000000);
    HVX_ಸದಿಶ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(v_src[i]);
        HVX_ಸದಿಶ p0 = Q6_V_lo_W(p);
        HVX_ಸದಿಶ p1 = Q6_V_hi_W(p);
        ಮೊತ್ತ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        ಮೊತ್ತ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ಸದಿಶ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(v1);
        HVX_ಸದಿಶ p0 = Q6_V_lo_W(p);
        HVX_ಸದಿಶ p1 = Q6_V_hi_W(p);
        ಮೊತ್ತ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        ಮೊತ್ತ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    ಮೊತ್ತ_v = hvx_vec_reduce_ಮೊತ್ತ_f32(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_v));

    HVX_ಸದಿಶ t_v            = hvx_vec_splat_f32((ದಶಮಾಂಶ) ಅಂಕೆ_elems);
    HVX_ಸದಿಶ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ಸದಿಶ ಸರಾಸರಿ_v         = Q6_Vqf32_vmpy_VsfVsf(ಮೊತ್ತ_v, denom_v);
    HVX_ಸದಿಶ ಸರಾಸರಿ_epsilon_v = Q6_Vqf32_vadd_Vqf32Vsf(ಸರಾಸರಿ_v, epsilon_v);

    HVX_ಸದಿಶ scale_v = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(ಸರಾಸರಿ_epsilon_v));

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(v_src[i]);
        HVX_ಸದಿಶ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ಸದಿಶ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_ಗಮ್ಯ_f16(r0, r1);
    }

    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ಸದಿಶ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(v1);
        HVX_ಸದಿಶ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ಸದಿಶ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_ಸದಿಶ ಫಲಿತಾಂಶ = hvx_vec_f32_ಗಮ್ಯ_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, ಫಲಿತಾಂಶ);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_fast_norm_f16(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src,
                                     uint8_t * ಸೀಮಿತ dst,
                                     ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems,
                                     ದಶಮಾಂಶ     epsilon) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src = (HVX_ಸದಿಶ *) src;
    HVX_ಸದಿಶ * ಸೀಮಿತ v_dst       = (HVX_ಸದಿಶ *) dst;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nvec = ಅಂಕೆ_elems / VLEN_FP16;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nloe = ಅಂಕೆ_elems % VLEN_FP16;

    HVX_ಸದಿಶ ಮೊತ್ತ_sq_v = Q6_V_vsplat_R(0x00000000);
    HVX_ಸದಿಶ ಮೊತ್ತ_x_v  = Q6_V_vsplat_R(0x00000000);
    HVX_ಸದಿಶ epsilon_v = hvx_vec_splat_f32(epsilon);

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(v_src[i]);
        HVX_ಸದಿಶ p0 = Q6_V_lo_W(p);
        HVX_ಸದಿಶ p1 = Q6_V_hi_W(p);
        ಮೊತ್ತ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        ಮೊತ್ತ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        ಮೊತ್ತ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        ಮೊತ್ತ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ಸದಿಶ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(v1);
        HVX_ಸದಿಶ p0 = Q6_V_lo_W(p);
        HVX_ಸದಿಶ p1 = Q6_V_hi_W(p);
        ಮೊತ್ತ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        ಮೊತ್ತ_sq_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_sq_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
        ಮೊತ್ತ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_x_v,  Q6_Vqf32_vadd_VsfVsf(p0, Q6_V_vzero()));
        ಮೊತ್ತ_x_v  = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_x_v,  Q6_Vqf32_vadd_VsfVsf(p1, Q6_V_vzero()));
    }

    ಮೊತ್ತ_sq_v = hvx_vec_reduce_ಮೊತ್ತ_f32(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_sq_v));
    ಮೊತ್ತ_x_v  = hvx_vec_reduce_ಮೊತ್ತ_f32(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_x_v));

    HVX_ಸದಿಶ t_v            = hvx_vec_splat_f32((ದಶಮಾಂಶ) ಅಂಕೆ_elems);
    HVX_ಸದಿಶ denom_v        = hvx_vec_inverse_f32(t_v);
    HVX_ಸದಿಶ ಸರಾಸರಿ_sq_v      = Q6_Vqf32_vmpy_VsfVsf(ಮೊತ್ತ_sq_v, denom_v);
    HVX_ಸದಿಶ ಸರಾಸರಿ_x_v       = Q6_Vqf32_vmpy_VsfVsf(ಮೊತ್ತ_x_v,  denom_v);
    HVX_ಸದಿಶ ಸರಾಸರಿ_x_sq_v    = Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(ಸರಾಸರಿ_x_v), Q6_Vsf_equals_Vqf32(ಸರಾಸರಿ_x_v));
    HVX_ಸದಿಶ var_v          = Q6_Vqf32_vsub_Vqf32Vqf32(ಸರಾಸರಿ_sq_v, ಸರಾಸರಿ_x_sq_v);
    HVX_ಸದಿಶ var_epsilon_v  = Q6_Vqf32_vadd_Vqf32Vsf(var_v, epsilon_v);

    HVX_ಸದಿಶ scale_v  = hvx_vec_rsqrt_f32(Q6_Vsf_equals_Vqf32(var_epsilon_v));
    HVX_ಸದಿಶ ಸರಾಸರಿ_x_b = hvx_vec_repl_f32(Q6_Vsf_equals_Vqf32(ಸರಾಸರಿ_x_v));

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(v_src[i]);
        HVX_ಸದಿಶ d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), ಸರಾಸರಿ_x_b);
        HVX_ಸದಿಶ d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), ಸರಾಸರಿ_x_b);
        HVX_ಸದಿಶ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_ಸದಿಶ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        v_dst[i] = hvx_vec_f32_ಗಮ್ಯ_f16(r0, r1);
    }

    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ಸದಿಶ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(v1);
        HVX_ಸದಿಶ d0 = Q6_Vqf32_vsub_VsfVsf(Q6_V_lo_W(p), ಸರಾಸರಿ_x_b);
        HVX_ಸದಿಶ d1 = Q6_Vqf32_vsub_VsfVsf(Q6_V_hi_W(p), ಸರಾಸರಿ_x_b);
        HVX_ಸದಿಶ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d0), scale_v));
        HVX_ಸದಿಶ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_Vsf_equals_Vqf32(d1), scale_v));
        HVX_ಸದಿಶ ಫಲಿತಾಂಶ = hvx_vec_f32_ಗಮ್ಯ_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, ಫಲಿತಾಂಶ);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hvx_fast_l2_norm_f16(ಸ್ಥಿರಾಂಕ uint8_t * ಸೀಮಿತ src,
                                        uint8_t * ಸೀಮಿತ dst,
                                        ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ ಅಂಕೆ_elems,
                                        ದಶಮಾಂಶ     epsilon) {

    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * ಸೀಮಿತ v_src = (HVX_ಸದಿಶ *) src;
    HVX_ಸದಿಶ * ಸೀಮಿತ v_dst       = (HVX_ಸದಿಶ *) dst;

    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nvec = ಅಂಕೆ_elems / VLEN_FP16;
    ಸ್ಥಿರಾಂಕ ಪೂರ್ಣಾಂಕ nloe = ಅಂಕೆ_elems % VLEN_FP16;

    HVX_ಸದಿಶ ಮೊತ್ತ_v = hvx_vec_splat_f32(0.0f);

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(v_src[i]);
        HVX_ಸದಿಶ p0 = Q6_V_lo_W(p);
        HVX_ಸದಿಶ p1 = Q6_V_hi_W(p);
        ಮೊತ್ತ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        ಮೊತ್ತ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ಸದಿಶ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(v1);
        HVX_ಸದಿಶ p0 = Q6_V_lo_W(p);
        HVX_ಸದಿಶ p1 = Q6_V_hi_W(p);
        ಮೊತ್ತ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_v, Q6_Vqf32_vmpy_VsfVsf(p0, p0));
        ಮೊತ್ತ_v = Q6_Vqf32_vadd_Vqf32Vqf32(ಮೊತ್ತ_v, Q6_Vqf32_vmpy_VsfVsf(p1, p1));
    }

    HVX_ಸದಿಶ ಮೊತ್ತ_sf    = hvx_vec_reduce_ಮೊತ್ತ_f32(Q6_Vsf_equals_Vqf32(ಮೊತ್ತ_v));
    HVX_ಸದಿಶ rsqrt_v   = hvx_vec_rsqrt_f32(ಮೊತ್ತ_sf);
    HVX_ಸದಿಶ sqrt_v    = hvx_vec_inverse_f32(rsqrt_v);
    HVX_ಸದಿಶ epsilon_v = hvx_vec_splat_f32(epsilon);
    HVX_ಸದಿಶ denom_v   = Q6_Vsf_vmax_VsfVsf(sqrt_v, epsilon_v);
    HVX_ಸದಿಶ scale_v   = hvx_vec_inverse_f32(denom_v);

    #ಸೂಚನೆ unroll(4)
    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ i = 0; i < nvec; i++) {
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(v_src[i]);
        HVX_ಸದಿಶ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ಸದಿಶ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        v_dst[i] = hvx_vec_f32_ಗಮ್ಯ_f16(r0, r1);
    }

    ಆದರೆ (nloe > 0) {
        HVX_ಸದಿಶ_Pred bmask = Q6_Q_vsetq_R(nloe * SIZEOF_FP16);
        HVX_ಸದಿಶ v1 = Q6_V_vand_QV(bmask, v_src[nvec]);
        HVX_ಸದಿಶ_Pair p = hvx_vec_f16_ಗಮ್ಯ_f32(v1);
        HVX_ಸದಿಶ r0 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_lo_W(p), scale_v));
        HVX_ಸದಿಶ r1 = Q6_Vsf_equals_Vqf32(Q6_Vqf32_vmpy_VsfVsf(Q6_V_hi_W(p), scale_v));
        HVX_ಸದಿಶ ಫಲಿತಾಂಶ = hvx_vec_f32_ಗಮ್ಯ_f16(r0, r1);
        hvx_vec_store_a(&v_dst[nvec], nloe * SIZEOF_FP16, ಫಲಿತಾಂಶ);
    }
}

#ಆದರೆ_ಅಂತ್ಯ // HVX_NORM_H
