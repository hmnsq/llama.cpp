#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HVX_FLASH_ATTN_H
#ವ್ಯಾಖ್ಯಾನಿಸು HVX_FLASH_ATTN_H

#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"

// Scalar helper to compute a single ALiBi slope.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ದಶಮಾಂಶ alibi_ಇಳಿಜಾರು(uint32_t h, uint32_t n_ತಲೆ_log2, ದಶಮಾಂಶ m0, ದಶಮಾಂಶ m1) {
    ಹಿಂತಿರುಗಿ (h < n_ತಲೆ_log2) ? powf(m0, h + 1) : powf(m1, 2 * (h - n_ತಲೆ_log2) + 1);
}

// Vectorized helper to compute 32 ALiBi slopes starting from (kv_head * G).
ಸ್ಥಿರ ಸಾಲೊಳಗೆ HVX_ಸದಿಶ hvx_alibi_slopes(
    uint32_t kv_ತಲೆ,
    uint32_t G,
    uint32_t n_ತಲೆ_log2,
    ದಶಮಾಂಶ m0,
    ದಶಮಾಂಶ m1
) {
    ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ ramp_32[32] __attribute__((aligned(128))) = {
        0.0f, 1.0f, 2.0f, 3.0f, 4.0f, 5.0f, 6.0f, 7.0f,
        8.0f, 9.0f, 10.0f, 11.0f, 12.0f, 13.0f, 14.0f, 15.0f,
        16.0f, 17.0f, 18.0f, 19.0f, 20.0f, 21.0f, 22.0f, 23.0f,
        24.0f, 25.0f, 26.0f, 27.0f, 28.0f, 29.0f, 30.0f, 31.0f
    };
    HVX_ಸದಿಶ v_ramp = hvx_vmem(ramp_32);
    HVX_ಸದಿಶ v_h_ತಳ = hvx_vec_splat_f32((ದಶಮಾಂಶ)(kv_ತಲೆ * G));
    HVX_ಸದಿಶ v_h = hvx_vec_ಕೂಡಿಸು_f32_f32(v_h_ತಳ, v_ramp);

    // Compute exponent_m0: h + 1
    HVX_ಸದಿಶ v_exp_m0 = hvx_vec_ಕೂಡಿಸು_f32_f32(v_h, hvx_vec_splat_f32(1.0f));

    // Compute exponent_m1: 2 * (h - n_head_log2) + 1
    HVX_ಸದಿಶ v_n_ತಲೆ_log2 = hvx_vec_splat_f32((ದಶಮಾಂಶ)n_ತಲೆ_log2);
    HVX_ಸದಿಶ v_h_minus = hvx_vec_sub_f32_f32(v_h, v_n_ತಲೆ_log2);
    HVX_ಸದಿಶ v_exp_m1 = hvx_vec_ಕೂಡಿಸು_f32_f32(hvx_vec_mul_f32_f32(hvx_vec_splat_f32(2.0f), v_h_minus), hvx_vec_splat_f32(1.0f));

    // Compute powers
    HVX_ಸದಿಶ v_pow_m0 = hvx_vec_pow_const_ತಳ_f32(m0, v_exp_m0);
    HVX_ಸದಿಶ v_pow_m1 = hvx_vec_pow_const_ತಳ_f32(m1, v_exp_m1);

    // Select based on h < n_head_log2
    HVX_ಸದಿಶ_Pred p_cond = Q6_Q_vcmp_gt_VsfVsf(v_n_ತಲೆ_log2, v_h); // v_n_head_log2 > v_h <=> h < n_head_log2
    ಹಿಂತಿರುಗಿ Q6_V_vmux_QVV(p_cond, v_pow_m0, v_pow_m1);
}

#ಆದರೆ_ಅಂತ್ಯ /* HVX_FLASH_ATTN_H */
