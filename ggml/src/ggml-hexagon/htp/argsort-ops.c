#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#ವ್ಯಾಖ್ಯಾನಿಸು GGML_COMMON_DECL_C
#ಅನ್ತರ್ಭಾವ "ggml-common.h"
#ಅನ್ತರ್ಭಾವ "ggml.h"

#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "hvx-copy.h"
#ಅನ್ತರ್ಭಾವ "hvx-reduce.h"
#ಅನ್ತರ್ಭಾವ "dma-queue.h"

#ಅನ್ತರ್ಭಾವ "hex-common.h"
#ಅನ್ತರ್ಭಾವ "htp-ctx.h"
#ಅನ್ತರ್ಭಾವ "htp-ops.h"
#ಅನ್ತರ್ಭಾವ "htp-tensor.h"
#ಅನ್ತರ್ಭಾವ "argsort-ops.h"

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ MIN
#ವ್ಯಾಖ್ಯಾನಿಸು MIN(a, b) ((a) < (b) ? (a) : (b))
#ಆದರೆ_ಅಂತ್ಯ


ಸ್ಥಿರ ಶೂನ್ಯ quicksort_ಮೌಲ್ಯಗಳು_indices_asc(ದಶಮಾಂಶ * ಮೌಲ್ಯಗಳು, int32_t * indices, ಪೂರ್ಣಾಂಕ ಎಡ_2, ಪೂರ್ಣಾಂಕ ಬಲ_2) {
    ಆಗ (ಎಡ_2 < ಬಲ_2) {
        ದಶಮಾಂಶ ಅಚ್ಚು = ಮೌಲ್ಯಗಳು[ಎಡ_2 + (ಬಲ_2 - ಎಡ_2) / 2];
        ಪೂರ್ಣಾಂಕ i = ಎಡ_2;
        ಪೂರ್ಣಾಂಕ j = ಬಲ_2;

        ಆಗ (i <= j) {
            ಆಗ (ಮೌಲ್ಯಗಳು[i] < ಅಚ್ಚು) i++;
            ಆಗ (ಮೌಲ್ಯಗಳು[j] > ಅಚ್ಚು) j--;

            ಆದರೆ (i <= j) {
                ದಶಮಾಂಶ tmp_val = ಮೌಲ್ಯಗಳು[i];
                ಮೌಲ್ಯಗಳು[i] = ಮೌಲ್ಯಗಳು[j];
                ಮೌಲ್ಯಗಳು[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        ಆದರೆ (j - ಎಡ_2 < ಬಲ_2 - i) {
            ಆದರೆ (ಎಡ_2 < j) quicksort_ಮೌಲ್ಯಗಳು_indices_asc(ಮೌಲ್ಯಗಳು, indices, ಎಡ_2, j);
            ಎಡ_2 = i;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (i < ಬಲ_2) quicksort_ಮೌಲ್ಯಗಳು_indices_asc(ಮೌಲ್ಯಗಳು, indices, i, ಬಲ_2);
            ಬಲ_2 = j;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ quicksort_ಮೌಲ್ಯಗಳು_indices_desc(ದಶಮಾಂಶ * ಮೌಲ್ಯಗಳು, int32_t * indices, ಪೂರ್ಣಾಂಕ ಎಡ_2, ಪೂರ್ಣಾಂಕ ಬಲ_2) {
    ಆಗ (ಎಡ_2 < ಬಲ_2) {
        ದಶಮಾಂಶ ಅಚ್ಚು = ಮೌಲ್ಯಗಳು[ಎಡ_2 + (ಬಲ_2 - ಎಡ_2) / 2];
        ಪೂರ್ಣಾಂಕ i = ಎಡ_2;
        ಪೂರ್ಣಾಂಕ j = ಬಲ_2;

        ಆಗ (i <= j) {
            ಆಗ (ಮೌಲ್ಯಗಳು[i] > ಅಚ್ಚು) i++;
            ಆಗ (ಮೌಲ್ಯಗಳು[j] < ಅಚ್ಚು) j--;

            ಆದರೆ (i <= j) {
                ದಶಮಾಂಶ tmp_val = ಮೌಲ್ಯಗಳು[i];
                ಮೌಲ್ಯಗಳು[i] = ಮೌಲ್ಯಗಳು[j];
                ಮೌಲ್ಯಗಳು[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        ಆದರೆ (j - ಎಡ_2 < ಬಲ_2 - i) {
            ಆದರೆ (ಎಡ_2 < j) quicksort_ಮೌಲ್ಯಗಳು_indices_desc(ಮೌಲ್ಯಗಳು, indices, ಎಡ_2, j);
            ಎಡ_2 = i;
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (i < ಬಲ_2) quicksort_ಮೌಲ್ಯಗಳು_indices_desc(ಮೌಲ್ಯಗಳು, indices, i, ಬಲ_2);
            ಬಲ_2 = j;
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ uint32_t ತುದಿ_k_max_ಮೌಲ್ಯ_ಸೂಚ್ಯಂಕ(ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಮೌಲ್ಯಗಳು, uint32_t n, ದಶಮಾಂಶ * ಮೌಲ್ಯ) {
    int32_t max_idx = 0;
    hvx_argmax_f32(ಮೌಲ್ಯಗಳು, n, 0, ಮೌಲ್ಯ, &max_idx);
    ಹಿಂತಿರುಗಿ (uint32_t) max_idx;
}

ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ init_indices_ramp(int32_t * indices_buf, uint32_t ಎಣಿಕೆ_2, uint32_t ತಳ) {
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ind_init_vec = Q6_Vw_vadd_VwVw(*(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) argsort_ramp_lut, Q6_V_vsplat_R(ತಳ));
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_ಸದಿಶ * indices_buf_vec = (HVX_ಸದಿಶ *) indices_buf;
    uint32_t ಅಂಕೆ_vecs = ಎಣಿಕೆ_2 / 32;
    HVX_ಸದಿಶ curr_ind_vec = ind_init_vec;
    ಗಾಗಿ (uint32_t j = 0; j < ಅಂಕೆ_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = ಎಣಿಕೆ_2 % 32;
    ಆದರೆ (rem > 0) {
        hvx_vec_store_a((ಶೂನ್ಯ *) &indices_buf_vec[ಅಂಕೆ_vecs], rem * ಗಾತ್ರ(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ vec_cas(HVX_ಸದಿಶ * X_val, HVX_ಸದಿಶ * X_idx, HVX_ಸದಿಶ * Y_val, HVX_ಸದಿಶ * Y_idx, bool asc) {
    HVX_ಸದಿಶ_Pred pred = asc ? Q6_Q_vcmp_gt_VsfVsf(*X_val, *Y_val)
                              : Q6_Q_vcmp_gt_VsfVsf(*Y_val, *X_val);
    HVX_ಸದಿಶ ಮುಂದಿನ_X_val = Q6_V_vmux_QVV(pred, *Y_val, *X_val);
    HVX_ಸದಿಶ ಮುಂದಿನ_Y_val = Q6_V_vmux_QVV(pred, *X_val, *Y_val);
    HVX_ಸದಿಶ ಮುಂದಿನ_X_idx = Q6_V_vmux_QVV(pred, *Y_idx, *X_idx);
    HVX_ಸದಿಶ Y_tmp_idx  = Q6_V_vmux_QVV(pred, *X_idx, *Y_idx);
    *X_val = ಮುಂದಿನ_X_val;
    *Y_val = ಮುಂದಿನ_Y_val;
    *X_idx = ಮುಂದಿನ_X_idx;
    *Y_idx = Y_tmp_idx;
}

__attribute__((always_inline))
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ bitonic_cas_32(HVX_ಸದಿಶ * V, HVX_ಸದಿಶ * I, ಪೂರ್ಣಾಂಕ d, HVX_ಸದಿಶ_Pred dir_ಮುಖವಾಡ, HVX_ಸದಿಶ idx_vec, HVX_ಸದಿಶ zero_vec) {
    HVX_ಸದಿಶ_Pred ಮುಖವಾಡ_ಎಡ;
    HVX_ಸದಿಶ V_rot_ಎಡ, V_rot_ಬಲ;
    HVX_ಸದಿಶ I_rot_ಎಡ, I_rot_ಬಲ;

    ಆದರೆ (d == 1) {
        ಮುಖವಾಡ_ಎಡ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_ಎಡ = Q6_V_vror_VR(*V, 4);
        V_rot_ಬಲ = Q6_V_vror_VR(*V, 124);
        I_rot_ಎಡ = Q6_V_vror_VR(*I, 4);
        I_rot_ಬಲ = Q6_V_vror_VR(*I, 124);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (d == 2) {
        ಮುಖವಾಡ_ಎಡ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_ಎಡ = Q6_V_vror_VR(*V, 8);
        V_rot_ಬಲ = Q6_V_vror_VR(*V, 120);
        I_rot_ಎಡ = Q6_V_vror_VR(*I, 8);
        I_rot_ಬಲ = Q6_V_vror_VR(*I, 120);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (d == 4) {
        ಮುಖವಾಡ_ಎಡ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_ಎಡ = Q6_V_vror_VR(*V, 16);
        V_rot_ಬಲ = Q6_V_vror_VR(*V, 112);
        I_rot_ಎಡ = Q6_V_vror_VR(*I, 16);
        I_rot_ಬಲ = Q6_V_vror_VR(*I, 112);
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (d == 8) {
        ಮುಖವಾಡ_ಎಡ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_ಎಡ = Q6_V_vror_VR(*V, 32);
        V_rot_ಬಲ = Q6_V_vror_VR(*V, 96);
        I_rot_ಎಡ = Q6_V_vror_VR(*I, 32);
        I_rot_ಬಲ = Q6_V_vror_VR(*I, 96);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಮುಖವಾಡ_ಎಡ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_ಎಡ = Q6_V_vror_VR(*V, 64);
        V_rot_ಬಲ = Q6_V_vror_VR(*V, 64);
        I_rot_ಎಡ = Q6_V_vror_VR(*I, 64);
        I_rot_ಬಲ = Q6_V_vror_VR(*I, 64);
    }

    HVX_ಸದಿಶ V_paired = Q6_V_vmux_QVV(ಮುಖವಾಡ_ಎಡ, V_rot_ಎಡ, V_rot_ಬಲ);
    HVX_ಸದಿಶ I_paired = Q6_V_vmux_QVV(ಮುಖವಾಡ_ಎಡ, I_rot_ಎಡ, I_rot_ಬಲ);

    HVX_ಸದಿಶ_Pred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_ಸದಿಶ_Pred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_ಸದಿಶ_Pred ಮುಖವಾಡ_ಬಲ = Q6_Q_not_Q(ಮುಖವಾಡ_ಎಡ);
    HVX_ಸದಿಶ_Pred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(ಮುಖವಾಡ_ಎಡ, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, ಮುಖವಾಡ_ಬಲ)
    );
    HVX_ಸದಿಶ_Pred Q_ಅದಲುಬದಲು = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_ಮುಖವಾಡ, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_ಮುಖವಾಡ), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_ಅದಲುಬದಲು, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_ಅದಲುಬದಲು, I_paired, *I);
}

__attribute__((always_inline))
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ bitonic_ವಿಂಗಡಿಸು_generic_hvx(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, ಪೂರ್ಣಾಂಕ K, bool asc_ಕ್ರಮ) {
    HVX_ಸದಿಶ V[32];
    HVX_ಸದಿಶ I[32];

    HVX_ಸದಿಶ zero_vec = Q6_V_vzero();
    HVX_ಸದಿಶ idx_vec = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) argsort_ramp_lut;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ v = 0; v < K; v++) {
        V[v] = *(HVX_ಸದಿಶ *)(ಮೌಲ್ಯಗಳು + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_ಸದಿಶ_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_ಸದಿಶ_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    ಪೂರ್ಣಾಂಕ M = 5;
    ಆಗ ((1 << (M - 5)) < K) M++;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 1; s <= M; s++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ stage_d = s - 1; stage_d >= 0; stage_d--) {
            ಪೂರ್ಣಾಂಕ d = 1 << stage_d;
            ಆದರೆ (d >= 32) {
                ಪೂರ್ಣಾಂಕ v_dist = d / 32;
                ಗಾಗಿ (ಪೂರ್ಣಾಂಕ v1 = 0; v1 < K; v1++) {
                    ಆದರೆ ((v1 & v_dist) == 0) {
                        ಪೂರ್ಣಾಂಕ v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_ಕ್ರಮ;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (s < 5) {
                    HVX_ಸದಿಶ_Pred dir_ಮುಖವಾಡ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_ಮುಖವಾಡ, idx_vec, zero_vec);
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_ಕ್ರಮ;
                        HVX_ಸದಿಶ_Pred dir_ಮುಖವಾಡ = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_ಮುಖವಾಡ, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ v = 0; v < K; v++) {
        *(HVX_ಸದಿಶ *)(ಮೌಲ್ಯಗಳು + v * 128)  = V[v];
        *(HVX_ಸದಿಶ *)(indices + v * 128) = I[v];
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ bitonic_ವಿಂಗಡಿಸು_vtcm_desc(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_ಸದಿಶ zero_vec = Q6_V_vzero();
    HVX_ಸದಿಶ idx_vec = *(ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) argsort_ramp_lut;

    HVX_ಸದಿಶ_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_ಸದಿಶ_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    ಆದರೆ (init_indices) {
        ಗಾಗಿ (uint32_t v = 0; v < n_vec; v++) {
            HVX_ಸದಿಶ idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_ಸದಿಶ *)(indices + v * 128) = idx;
        }
    }

    ಪೂರ್ಣಾಂಕ M = 5;
    ಆಗ ((1u << (M - 5)) < n_vec) M++;

    ಗಾಗಿ (ಪೂರ್ಣಾಂಕ s = 1; s <= M; s++) {
        ಗಾಗಿ (ಪೂರ್ಣಾಂಕ stage_d = s - 1; stage_d >= 0; stage_d--) {
            ಪೂರ್ಣಾಂಕ d = 1 << stage_d;
            ಆದರೆ (d >= 32) {
                uint32_t v_dist = d / 32;
                ಗಾಗಿ (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    ಆದರೆ ((v1 & v_dist) == 0) {
                        uint32_t v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : false;

                        HVX_ಸದಿಶ Vv1 = *(HVX_ಸದಿಶ *)(ಮೌಲ್ಯಗಳು + v1 * 128);
                        HVX_ಸದಿಶ Iv1 = *(HVX_ಸದಿಶ *)(indices + v1 * 128);
                        HVX_ಸದಿಶ Vv2 = *(HVX_ಸದಿಶ *)(ಮೌಲ್ಯಗಳು + v2 * 128);
                        HVX_ಸದಿಶ Iv2 = *(HVX_ಸದಿಶ *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_ಸದಿಶ *)(ಮೌಲ್ಯಗಳು + v1 * 128)  = Vv1;
                        *(HVX_ಸದಿಶ *)(indices + v1 * 128) = Iv1;
                        *(HVX_ಸದಿಶ *)(ಮೌಲ್ಯಗಳು + v2 * 128)  = Vv2;
                        *(HVX_ಸದಿಶ *)(indices + v2 * 128) = Iv2;
                    }
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (s < 5) {
                    HVX_ಸದಿಶ_Pred dir_ಮುಖವಾಡ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    ಗಾಗಿ (uint32_t v = 0; v < n_vec; v++) {
                        HVX_ಸದಿಶ Vv = *(HVX_ಸದಿಶ *)(ಮೌಲ್ಯಗಳು + v * 128);
                        HVX_ಸದಿಶ Iv = *(HVX_ಸದಿಶ *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_ಮುಖವಾಡ, idx_vec, zero_vec);

                        *(HVX_ಸದಿಶ *)(ಮೌಲ್ಯಗಳು + v * 128)  = Vv;
                        *(HVX_ಸದಿಶ *)(indices + v * 128) = Iv;
                    }
                } ಇಲ್ಲದಿದ್ದರೆ {
                    ಗಾಗಿ (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_ಸದಿಶ_Pred dir_ಮುಖವಾಡ = asc ? pred_all_1s : pred_all_0s;

                        HVX_ಸದಿಶ Vv = *(HVX_ಸದಿಶ *)(ಮೌಲ್ಯಗಳು + v * 128);
                        HVX_ಸದಿಶ Iv = *(HVX_ಸದಿಶ *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_ಮುಖವಾಡ, idx_vec, zero_vec);

                        *(HVX_ಸದಿಶ *)(ಮೌಲ್ಯಗಳು + v * 128)  = Vv;
                        *(HVX_ಸದಿಶ *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ತುದಿ_k_select_tiled(dma_ಸರತಿ * dma_q, ರಚನೆ htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               ದಶಮಾಂಶ * ಮೌಲ್ಯಗಳು_buf, int32_t * indices_buf,
                               ದಶಮಾಂಶ * ಹೊರಹೋಗು_ಮೌಲ್ಯಗಳು, int32_t * ಹೊರಹೋಗು_indices,
                               uint32_t chunk_ತಳ) {
    ಸ್ಥಿರಾಂಕ uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_ಎಣಿಕೆ    = n_tiles * k;
    uint32_t ವಿಲೀನ_n_vec        = hmx_ceil_div(candidate_ಎಣಿಕೆ, 32);
    uint32_t ವಿಲೀನ_n_vec_pow2   = 1;
    ಆಗ (ವಿಲೀನ_n_vec_pow2 < ವಿಲೀನ_n_vec) ವಿಲೀನ_n_vec_pow2 <<= 1;
    uint32_t ವಿಲೀನ_elems        = ವಿಲೀನ_n_vec_pow2 * 32;
    ದಶಮಾಂಶ * candidate_ಮೌಲ್ಯಗಳು    = ಮೌಲ್ಯಗಳು_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    ಗಾಗಿ (uint32_t ಸ್ಥಾನಾಂತರ = 0; ಸ್ಥಾನಾಂತರ < n; ಸ್ಥಾನಾಂತರ += tile_elems) {
        uint32_t tile_ಎಣಿಕೆ = MIN(tile_elems, n - ಸ್ಥಾನಾಂತರ);
        dma_sync_read(dma_q, ಮೌಲ್ಯಗಳು_buf, src_addr + (size_t) ಸ್ಥಾನಾಂತರ * ಗಾತ್ರ(ದಶಮಾಂಶ), tile_ಎಣಿಕೆ * ಗಾತ್ರ(ದಶಮಾಂಶ));

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ಸ್ಥಾನಾಂತರ);
        ಆದರೆ (tile_ಎಣಿಕೆ < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (ಮೌಲ್ಯಗಳು_buf + tile_ಎಣಿಕೆ), -INFINITY, tile_elems - tile_ಎಣಿಕೆ);
        }

        bitonic_ವಿಂಗಡಿಸು_vtcm_desc((uint8_t *) ಮೌಲ್ಯಗಳು_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_ಎಣಿಕೆ);
        ಗಾಗಿ (uint32_t j = 0; j < tile_k; j++) {
            candidate_ಮೌಲ್ಯಗಳು[candidate_pos] = ಮೌಲ್ಯಗಳು_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) ಸ್ಥಾನಾಂತರ;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ಸ್ಥಾನಾಂತರ);
    }

    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    ಆದರೆ (ವಿಲೀನ_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_ಮೌಲ್ಯಗಳು + candidate_pos), -INFINITY, ವಿಲೀನ_elems - candidate_pos);
        ಗಾಗಿ (uint32_t j = candidate_pos; j < ವಿಲೀನ_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_ವಿಂಗಡಿಸು_vtcm_desc((uint8_t *) candidate_ಮೌಲ್ಯಗಳು, (uint8_t *) candidate_indices, ವಿಲೀನ_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    ಗಾಗಿ (uint32_t j = 0; j < k; j++) {
        ಹೊರಹೋಗು_ಮೌಲ್ಯಗಳು[j]  = candidate_ಮೌಲ್ಯಗಳು[j];
        ಹೊರಹೋಗು_indices[j] = candidate_indices[j] + (int32_t) chunk_ತಳ;
    }
}

__attribute__((always_inline))
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ sort32_f32_hvx(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ) {
    bitonic_ವಿಂಗಡಿಸು_generic_hvx(ಮೌಲ್ಯಗಳು, indices, 1, ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC);
}

__attribute__((always_inline))
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ sort64_f32_hvx(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ) {
    bitonic_ವಿಂಗಡಿಸು_generic_hvx(ಮೌಲ್ಯಗಳು, indices, 2, ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC);
}

__attribute__((always_inline))
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ sort128_f32_hvx(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ) {
    bitonic_ವಿಂಗಡಿಸು_generic_hvx(ಮೌಲ್ಯಗಳು, indices, 4, ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC);
}

__attribute__((always_inline))
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ sort256_f32_hvx(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ) {
    bitonic_ವಿಂಗಡಿಸು_generic_hvx(ಮೌಲ್ಯಗಳು, indices, 8, ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC);
}

__attribute__((always_inline))
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ sort512_f32_hvx(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ) {
    bitonic_ವಿಂಗಡಿಸು_generic_hvx(ಮೌಲ್ಯಗಳು, indices, 16, ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC);
}

__attribute__((always_inline))
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ sort1024_f32_hvx(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ) {
    bitonic_ವಿಂಗಡಿಸು_generic_hvx(ಮೌಲ್ಯಗಳು, indices, 32, ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC);
}

ಸ್ಥಿರ ಶೂನ್ಯ ವಿಲೀನ_runs(
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ ಒಳ_val0, ಸ್ಥಿರಾಂಕ int32_t * ಸೀಮಿತ ಒಳ_idx0, uint32_t n0,
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ * ಸೀಮಿತ ಒಳ_val1, ಸ್ಥಿರಾಂಕ int32_t * ಸೀಮಿತ ಒಳ_idx1, uint32_t n1,
    ದಶಮಾಂಶ * ಸೀಮಿತ ಹೊರಹೋಗು_val, int32_t * ಸೀಮಿತ ಹೊರಹೋಗು_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t ಹೊರಹೋಗು = 0;

    ಆದರೆ (asc) {
        ಆಗ (i0 < n0 && i1 < n1) {
            ಆದರೆ (ಒಳ_val0[i0] <= ಒಳ_val1[i1]) {
                ಆದರೆ (ಹೊರಹೋಗು_val) ಹೊರಹೋಗು_val[ಹೊರಹೋಗು] = ಒಳ_val0[i0];
                ಹೊರಹೋಗು_idx[ಹೊರಹೋಗು] = ಒಳ_idx0[i0];
                i0++;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (ಹೊರಹೋಗು_val) ಹೊರಹೋಗು_val[ಹೊರಹೋಗು] = ಒಳ_val1[i1];
                ಹೊರಹೋಗು_idx[ಹೊರಹೋಗು] = ಒಳ_idx1[i1];
                i1++;
            }
            ಹೊರಹೋಗು++;
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆಗ (i0 < n0 && i1 < n1) {
            ಆದರೆ (ಒಳ_val0[i0] >= ಒಳ_val1[i1]) {
                ಆದರೆ (ಹೊರಹೋಗು_val) ಹೊರಹೋಗು_val[ಹೊರಹೋಗು] = ಒಳ_val0[i0];
                ಹೊರಹೋಗು_idx[ಹೊರಹೋಗು] = ಒಳ_idx0[i0];
                i0++;
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಆದರೆ (ಹೊರಹೋಗು_val) ಹೊರಹೋಗು_val[ಹೊರಹೋಗು] = ಒಳ_val1[i1];
                ಹೊರಹೋಗು_idx[ಹೊರಹೋಗು] = ಒಳ_idx1[i1];
                i1++;
            }
            ಹೊರಹೋಗು++;
        }
    }

    ಆಗ (i0 < n0) {
        ಆದರೆ (ಹೊರಹೋಗು_val) ಹೊರಹೋಗು_val[ಹೊರಹೋಗು] = ಒಳ_val0[i0];
        ಹೊರಹೋಗು_idx[ಹೊರಹೋಗು] = ಒಳ_idx0[i0];
        i0++;
        ಹೊರಹೋಗು++;
    }

    ಆಗ (i1 < n1) {
        ಆದರೆ (ಹೊರಹೋಗು_val) ಹೊರಹೋಗು_val[ಹೊರಹೋಗು] = ಒಳ_val1[i1];
        ಹೊರಹೋಗು_idx[ಹೊರಹೋಗು] = ಒಳ_idx1[i1];
        i1++;
        ಹೊರಹೋಗು++;
    }
}

ರಚನೆ htp_ವಿಂಗಡಿಸು_chunk_ctx {
    ರಚನೆ htp_ops_context *              octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params * kparams;
    uint8_t *                             vtcm_ತಳ;
    uint32_t                              real_ಎಣಿಕೆ[HTP_MAX_NTHREADS];
};

ಸ್ಥಿರ ಶೂನ್ಯ htp_ವಿಂಗಡಿಸು_chunk_ಕೆಲಸ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_ವಿಂಗಡಿಸು_chunk_ctx * cctx = (ರಚನೆ htp_ವಿಂಗಡಿಸು_chunk_ctx *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = cctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params * kparams = cctx->kparams;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_ತಳ  = i * chunk_elems;

    uint32_t real_ಎಣಿಕೆ = (chunk_ತಳ < ne00) ? MIN(chunk_elems, ne00 - chunk_ತಳ) : 0;
    cctx->real_ಎಣಿಕೆ[i] = real_ಎಣಿಕೆ;

    uint8_t * spad = cctx->vtcm_ತಳ + (size_t) kparams->phase1_ಸ್ಥಾನ_ಗಾತ್ರ * i;
    ದಶಮಾಂಶ *   ಮೌಲ್ಯಗಳು_buf  = (ದಶಮಾಂಶ *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * ಗಾತ್ರ(ದಶಮಾಂಶ));

    ದಶಮಾಂಶ *   ವಿಲೀನ_ಮೌಲ್ಯಗಳು  = (ದಶಮಾಂಶ *)   (cctx->vtcm_ತಳ + kparams->ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off);
    int32_t * ವಿಲೀನ_indices = (int32_t *) (cctx->vtcm_ತಳ + kparams->ವಿಲೀನ_indices_off);

    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[i];
    ಸ್ಥಿರಾಂಕ dma_addr_t src_addr = src0->ದತ್ತಾಂಶ + (size_t) chunk_ತಳ * ಗಾತ್ರ(ದಶಮಾಂಶ);

    ಆದರೆ (kparams->is_ತುದಿ_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        ಆದರೆ (real_ಎಣಿಕೆ == 0) {
            ಗಾಗಿ (uint32_t j = 0; j < local_k; j++) {
                ವಿಲೀನ_ಮೌಲ್ಯಗಳು[i * local_k + j]  = -INFINITY;
                ವಿಲೀನ_indices[i * local_k + j] = 0;
            }
            ಹಿಂತಿರುಗಿ;
        }

        ಆದರೆ (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], ಮೌಲ್ಯಗಳು_buf, src_addr, real_ಎಣಿಕೆ * ಗಾತ್ರ(ದಶಮಾಂಶ));

            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            ದಶಮಾಂಶ max_ಮೌಲ್ಯ;
            uint32_t max_ಸೂಚ್ಯಂಕ = ತುದಿ_k_max_ಮೌಲ್ಯ_ಸೂಚ್ಯಂಕ(ಮೌಲ್ಯಗಳು_buf, real_ಎಣಿಕೆ, &max_ಮೌಲ್ಯ);
            ವಿಲೀನ_ಮೌಲ್ಯಗಳು[i] = max_ಮೌಲ್ಯ;
            ವಿಲೀನ_indices[i] = (int32_t) (max_ಸೂಚ್ಯಂಕ + chunk_ತಳ);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            ಹಿಂತಿರುಗಿ;
        }

        ಆದರೆ (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            ತುದಿ_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_ಎಣಿಕೆ, local_k,
                               ಮೌಲ್ಯಗಳು_buf, indices_buf,
                               &ವಿಲೀನ_ಮೌಲ್ಯಗಳು[i * local_k], &ವಿಲೀನ_indices[i * local_k],
                               chunk_ತಳ);
            ಹಿಂತಿರುಗಿ;
        }

        dma_sync_read(octx->ctx->dma[i], ಮೌಲ್ಯಗಳು_buf, src_addr, real_ಎಣಿಕೆ * ಗಾತ್ರ(ದಶಮಾಂಶ));

        htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        ಆದರೆ (chunk_elems > real_ಎಣಿಕೆ) {
            hvx_splat_f32_u((uint8_t *)(ಮೌಲ್ಯಗಳು_buf + real_ಎಣಿಕೆ), -INFINITY, chunk_elems - real_ಎಣಿಕೆ);
        }

        bitonic_ವಿಂಗಡಿಸು_vtcm_desc((uint8_t *) ಮೌಲ್ಯಗಳು_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        ಗಾಗಿ (uint32_t j = 0; j < local_k; j++) {
            ವಿಲೀನ_ಮೌಲ್ಯಗಳು[i * local_k + j]  = ಮೌಲ್ಯಗಳು_buf[j];
            ವಿಲೀನ_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_ತಳ;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (real_ಎಣಿಕೆ > 0) {
            dma_sync_read(octx->ctx->dma[i], ಮೌಲ್ಯಗಳು_buf, src_addr, real_ಎಣಿಕೆ * ಗಾತ್ರ(ದಶಮಾಂಶ));

            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_ಎಣಿಕೆ, chunk_ತಳ);

            ಆದರೆ (kparams->ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC) {
                quicksort_ಮೌಲ್ಯಗಳು_indices_asc(ಮೌಲ್ಯಗಳು_buf, indices_buf, 0, real_ಎಣಿಕೆ - 1);
            } ಇಲ್ಲದಿದ್ದರೆ {
                quicksort_ಮೌಲ್ಯಗಳು_indices_desc(ಮೌಲ್ಯಗಳು_buf, indices_buf, 0, real_ಎಣಿಕೆ - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

ರಚನೆ htp_argsort_ವಿಲೀನ_l1_ctx {
    ರಚನೆ htp_ವಿಂಗಡಿಸು_chunk_ctx * cctx;
    bool                        asc;
};

ಸ್ಥಿರ ಶೂನ್ಯ htp_argsort_ವಿಲೀನ_l1_ಕೆಲಸ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_argsort_ವಿಲೀನ_l1_ctx * mctx = (ರಚನೆ htp_argsort_ವಿಲೀನ_l1_ctx *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ವಿಂಗಡಿಸು_chunk_ctx * cctx = mctx->cctx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_ಎಣಿಕೆ[c0];
    uint32_t count1 = cctx->real_ಎಣಿಕೆ[c1];

    size_t ಸ್ಥಾನ_ಗಾತ್ರ = (size_t) kparams->phase1_ಸ್ಥಾನ_ಗಾತ್ರ;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   val0 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)   (cctx->vtcm_ತಳ + ಸ್ಥಾನ_ಗಾತ್ರ * c0);
    ಸ್ಥಿರಾಂಕ int32_t * idx0 = (ಸ್ಥಿರಾಂಕ int32_t *) (cctx->vtcm_ತಳ + ಸ್ಥಾನ_ಗಾತ್ರ * c0 + chunk_elems * ಗಾತ್ರ(ದಶಮಾಂಶ));
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   val1 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)   (cctx->vtcm_ತಳ + ಸ್ಥಾನ_ಗಾತ್ರ * c1);
    ಸ್ಥಿರಾಂಕ int32_t * idx1 = (ಸ್ಥಿರಾಂಕ int32_t *) (cctx->vtcm_ತಳ + ಸ್ಥಾನ_ಗಾತ್ರ * c1 + chunk_elems * ಗಾತ್ರ(ದಶಮಾಂಶ));

    uint32_t ಹೊರಹೋಗು_ಸ್ಥಾನಾಂತರ = (2 * i) * chunk_elems;
    ದಶಮಾಂಶ *   ಹೊರಹೋಗು_val = (ದಶಮಾಂಶ *)   (cctx->vtcm_ತಳ + kparams->ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off)  + ಹೊರಹೋಗು_ಸ್ಥಾನಾಂತರ;
    int32_t * ಹೊರಹೋಗು_idx = (int32_t *) (cctx->vtcm_ತಳ + kparams->ವಿಲೀನ_indices_off) + ಹೊರಹೋಗು_ಸ್ಥಾನಾಂತರ;

    ರಚನೆ htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    ವಿಲೀನ_runs(val0, idx0, count0, val1, idx1, count1, ಹೊರಹೋಗು_val, ಹೊರಹೋಗು_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

ರಚನೆ htp_argsort_ವಿಲೀನ_l2_ctx {
    ರಚನೆ htp_ವಿಂಗಡಿಸು_chunk_ctx * cctx;
    bool                        asc;
};

ಸ್ಥಿರ ಶೂನ್ಯ htp_argsort_ವಿಲೀನ_l2_ಕೆಲಸ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ i, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_argsort_ವಿಲೀನ_l2_ctx * mctx = (ರಚನೆ htp_argsort_ವಿಲೀನ_l2_ctx *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ವಿಂಗಡಿಸು_chunk_ctx * cctx = mctx->cctx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params * kparams = cctx->kparams;

    size_t ಸ್ಥಾನ_ಗಾತ್ರ = (size_t) kparams->phase1_ಸ್ಥಾನ_ಗಾತ್ರ;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   val0 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)   (cctx->vtcm_ತಳ + kparams->ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off)  + off0;
    ಸ್ಥಿರಾಂಕ int32_t * idx0 = (ಸ್ಥಿರಾಂಕ int32_t *) (cctx->vtcm_ತಳ + kparams->ವಿಲೀನ_indices_off) + off0;
    ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   val1 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)   (cctx->vtcm_ತಳ + kparams->ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off)  + off1;
    ಸ್ಥಿರಾಂಕ int32_t * idx1 = (ಸ್ಥಿರಾಂಕ int32_t *) (cctx->vtcm_ತಳ + kparams->ವಿಲೀನ_indices_off) + off1;

    uint32_t count0 = cctx->real_ಎಣಿಕೆ[4*i + 0] + cctx->real_ಎಣಿಕೆ[4*i + 1];
    uint32_t count1 = cctx->real_ಎಣಿಕೆ[4*i + 2] + cctx->real_ಎಣಿಕೆ[4*i + 3];

    uint8_t * ಸ್ಥಾನ_ತಳ = cctx->vtcm_ತಳ + (4 * i) * ಸ್ಥಾನ_ಗಾತ್ರ;
    ದಶಮಾಂಶ *   ಹೊರಹೋಗು_val   = (ದಶಮಾಂಶ *) ಸ್ಥಾನ_ತಳ;
    int32_t * ಹೊರಹೋಗು_idx   = (int32_t *) (ಸ್ಥಾನ_ತಳ + 4 * chunk_elems * ಗಾತ್ರ(ದಶಮಾಂಶ));

    ರಚನೆ htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    ವಿಲೀನ_runs(val0, idx0, count0, val1, idx1, count1, ಹೊರಹೋಗು_val, ಹೊರಹೋಗು_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

ರಚನೆ htp_ವಿಂಗಡಿಸು_multi_ಅಡ್ಡಸಾಲು_ctx {
    ರಚನೆ htp_ops_context *              octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params * kparams;
    uint8_t *                             vtcm_ತಳ;
    uint32_t                              nrows_ಪ್ರತಿ_thread;
};

ರಚನೆ htp_tensor_ಅಡ್ಡಸಾಲು_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ರಚನೆ htp_tensor_ಅಡ್ಡಸಾಲು_coord init_ಅಡ್ಡಸಾಲು_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    ಹಿಂತಿರುಗಿ (ರಚನೆ htp_tensor_ಅಡ್ಡಸಾಲು_coord){ i01, i02, i03 };
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲು_coord(ರಚನೆ htp_tensor_ಅಡ್ಡಸಾಲು_coord * c, uint32_t ne01, uint32_t ne02) {
    c->i01++;
    ಆದರೆ (c->i01 == ne01) {
        c->i01 = 0;
        c->i02++;
        ಆದರೆ (c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ size_t coord_ಸ್ಥಾನಾಂತರ(ರಚನೆ htp_tensor_ಅಡ್ಡಸಾಲು_coord c, size_t nb1, size_t nb2, size_t nb3) {
    ಹಿಂತಿರುಗಿ (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ compute_ಅಡ್ಡಸಾಲು_ವಿಂಗಡಿಸು(
    ದಶಮಾಂಶ * cur_ಮೌಲ್ಯಗಳು,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_ತುದಿ_k,
    ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ
) {
    ಆದರೆ (is_ತುದಿ_k) {
        ಆದರೆ (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            ಆಯ್ಕೆ (ne00) {
                ಸ್ಥಿತಿ 32:   sort32_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 64:   sort64_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 128:  sort128_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 256:  sort256_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 512:  sort512_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 1024: sort1024_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC); ನಿಲ್ಲಿಸು;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಆದರೆ (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_ಮೌಲ್ಯಗಳು + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_ವಿಂಗಡಿಸು_vtcm_desc((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಆದರೆ (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            ಆಯ್ಕೆ (ne00) {
                ಸ್ಥಿತಿ 32:   sort32_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, ಕ್ರಮ); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 64:   sort64_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, ಕ್ರಮ); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 128:  sort128_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, ಕ್ರಮ); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 256:  sort256_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, ಕ್ರಮ); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 512:  sort512_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, ಕ್ರಮ); ನಿಲ್ಲಿಸು;
                ಸ್ಥಿತಿ 1024: sort1024_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, ಕ್ರಮ); ನಿಲ್ಲಿಸು;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            init_indices_ramp(cur_indices, ne00, 0);
            ಆದರೆ (ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC) {
                quicksort_ಮೌಲ್ಯಗಳು_indices_asc(cur_ಮೌಲ್ಯಗಳು, cur_indices, 0, ne00 - 1);
            } ಇಲ್ಲದಿದ್ದರೆ {
                quicksort_ಮೌಲ್ಯಗಳು_indices_desc(cur_ಮೌಲ್ಯಗಳು, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ htp_ವಿಂಗಡಿಸು_multi_ಅಡ್ಡಸಾಲು_ಕೆಲಸ(ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ n, ಚಿಹ್ನೆರಹಿತ ಪೂರ್ಣಾಂಕ ith, ಶೂನ್ಯ * ದತ್ತಾಂಶ) {
    ರಚನೆ htp_ವಿಂಗಡಿಸು_multi_ಅಡ್ಡಸಾಲು_ctx * mctx = (ರಚನೆ htp_ವಿಂಗಡಿಸು_multi_ಅಡ್ಡಸಾಲು_ctx *) ದತ್ತಾಂಶ;
    ರಚನೆ htp_ops_context * octx = mctx->octx;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params * kparams = mctx->kparams;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    uint32_t ne00 = (uint32_t) kparams->ne00;
    uint32_t ne01 = src0->ne[1];
    uint32_t ne02 = src0->ne[2];

    uint32_t nb01 = src0->nb[1];
    uint32_t nb02 = src0->nb[2];
    uint32_t nb03 = src0->nb[3];

    uint32_t nb1 = dst->nb[1];
    uint32_t nb2 = dst->nb[2];
    uint32_t nb3 = dst->nb[3];

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = (uint32_t) kparams->ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ   = (uint32_t) kparams->ಅಡ್ಡಸಾಲು_ಕೊನೆ;
    uint32_t r_ಆರಂಭ   = ಅಡ್ಡಸಾಲು_ಆರಂಭ + ith * mctx->nrows_ಪ್ರತಿ_thread;
    uint32_t r_ಕೊನೆ     = MIN(r_ಆರಂಭ + mctx->nrows_ಪ್ರತಿ_thread, ಅಡ್ಡಸಾಲು_ಕೊನೆ);

    ಆದರೆ (r_ಆರಂಭ >= r_ಕೊನೆ) ಹಿಂತಿರುಗಿ;

    size_t spad_ಸ್ಥಾನಾಂತರ = ith * (size_t) (kparams->phase1_ಸ್ಥಾನ_ಗಾತ್ರ * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_ತಳ + spad_ಸ್ಥಾನಾಂತರ;
    size_t ಸ್ಥಾನ_ಗಾತ್ರ = (size_t) kparams->phase1_ಸ್ಥಾನ_ಗಾತ್ರ;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    ದಶಮಾಂಶ *   ಮೌಲ್ಯಗಳು_buf  = (ದಶಮಾಂಶ *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * ಗಾತ್ರ(ದಶಮಾಂಶ));

    dma_ಸರತಿ * dma_q = octx->ctx->dma[ith];
    ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_ತುದಿ_k = kparams->is_ತುದಿ_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ ಕ್ರಮ = (ಎಣಿಕೆ ggml_ವಿಂಗಡಿಸು_ಕ್ರಮ) kparams->ಕ್ರಮ;

    ಸ್ಥಿರಾಂಕ uint32_t src_ಬೈಟುಗಳು = ne00 * ಗಾತ್ರ(ದಶಮಾಂಶ);
    ಸ್ಥಿರಾಂಕ uint32_t dst_ಬೈಟುಗಳು = (is_ತುದಿ_k ? k : ne00) * ಗಾತ್ರ(int32_t);

    ಆದರೆ (kparams->n_slots == 2) {
        ರಚನೆ htp_tensor_ಅಡ್ಡಸಾಲು_coord ಅವಿಭಾಜ್ಯ_coord = init_ಅಡ್ಡಸಾಲು_coord(r_ಆರಂಭ, ne01, ne02);
        ಗಾಗಿ (uint32_t r = r_ಆರಂಭ, spad_idx = 0; r < r_ಕೊನೆ && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * ಸ್ಥಾನ_ಗಾತ್ರ;
            ದಶಮಾಂಶ *   cur_ಮೌಲ್ಯಗಳು  = (ದಶಮಾಂಶ *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * ಗಾತ್ರ(ದಶಮಾಂಶ));

            dma_addr_t dst_addr = dst->ದತ್ತಾಂಶ + coord_ಸ್ಥಾನಾಂತರ(ಅವಿಭಾಜ್ಯ_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->ದತ್ತಾಂಶ + coord_ಸ್ಥಾನಾಂತರ(ಅವಿಭಾಜ್ಯ_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_addr, cur_indices),
                           dst_ಬೈಟುಗಳು, dst_ಬೈಟುಗಳು, dst_ಬೈಟುಗಳು, 0);

            // Prefetch input row
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(cur_ಮೌಲ್ಯಗಳು, src_addr),
                           src_ಬೈಟುಗಳು, src_ಬೈಟುಗಳು, src_ಬೈಟುಗಳು, 1);

            ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲು_coord(&ಅವಿಭಾಜ್ಯ_coord, ne01, ne02);
        }

        ರಚನೆ htp_tensor_ಅಡ್ಡಸಾಲು_coord dst_coord = init_ಅಡ್ಡಸಾಲು_coord(r_ಆರಂಭ, ne01, ne02);
        ರಚನೆ htp_tensor_ಅಡ್ಡಸಾಲು_coord ಮುಂದಿನ_src_coord = init_ಅಡ್ಡಸಾಲು_coord(r_ಆರಂಭ + 2, ne01, ne02);

        ಗಾಗಿ (uint32_t r = r_ಆರಂಭ; r < r_ಕೊನೆ; r++) {
            uint32_t cur_ಸ್ಥಾನ = (r - r_ಆರಂಭ) & 1;
            uint8_t * cur_spad = thread_spad + cur_ಸ್ಥಾನ * ಸ್ಥಾನ_ಗಾತ್ರ;
            ದಶಮಾಂಶ *   cur_ಮೌಲ್ಯಗಳು  = (ದಶಮಾಂಶ *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * ಗಾತ್ರ(ದಶಮಾಂಶ));

            // Wait for previous writeback of this slot to complete
            dma_ಸರತಿ_ತೆಗೆ(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_ಸರತಿ_ತೆಗೆ(dma_q);

            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_ಅಡ್ಡಸಾಲು_ವಿಂಗಡಿಸು(cur_ಮೌಲ್ಯಗಳು, cur_indices, ne00, chunk_elems, is_ತುದಿ_k, ಕ್ರಮ);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->ದತ್ತಾಂಶ + coord_ಸ್ಥಾನಾಂತರ(dst_coord, nb1, nb2, nb3);
            dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(dst_addr, cur_indices),
                           dst_ಬೈಟುಗಳು, dst_ಬೈಟುಗಳು, dst_ಬೈಟುಗಳು, 1);
            ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲು_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            ಸ್ಥಿರಾಂಕ uint32_t ಮುಂದಿನ_ಅಡ್ಡಸಾಲು = r + 2;
            ಆದರೆ (ಮುಂದಿನ_ಅಡ್ಡಸಾಲು < r_ಕೊನೆ) {
                dma_addr_t ಮುಂದಿನ_src_addr = src0->ದತ್ತಾಂಶ + coord_ಸ್ಥಾನಾಂತರ(ಮುಂದಿನ_src_coord, nb01, nb02, nb03);
                dma_ಸರತಿ_ತಳ್ಳು(dma_q, dma_make_ದತ್ತಾಂಶ(cur_ಮೌಲ್ಯಗಳು, ಮುಂದಿನ_src_addr),
                               src_ಬೈಟುಗಳು, src_ಬೈಟುಗಳು, src_ಬೈಟುಗಳು, 1);
                ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲು_coord(&ಮುಂದಿನ_src_coord, ne01, ne02);
            }
        }

        dma_ಸರತಿ_flush(dma_q);
    } ಇಲ್ಲದಿದ್ದರೆ {
        ರಚನೆ htp_tensor_ಅಡ್ಡಸಾಲು_coord coord = init_ಅಡ್ಡಸಾಲು_coord(r_ಆರಂಭ, ne01, ne02);
        ಗಾಗಿ (uint32_t r = r_ಆರಂಭ; r < r_ಕೊನೆ; r++) {
            dma_addr_t src_addr = src0->ದತ್ತಾಂಶ + coord_ಸ್ಥಾನಾಂತರ(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->ದತ್ತಾಂಶ + coord_ಸ್ಥಾನಾಂತರ(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, ಮೌಲ್ಯಗಳು_buf, src_addr, src_ಬೈಟುಗಳು);

            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_ಅಡ್ಡಸಾಲು_ವಿಂಗಡಿಸು(ಮೌಲ್ಯಗಳು_buf, indices_buf, ne00, chunk_elems, is_ತುದಿ_k, ಕ್ರಮ);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_ಬೈಟುಗಳು);
            ಹೆಜ್ಜೆ_ಅಡ್ಡಸಾಲು_coord(&coord, ne01, ne02);
        }
    }
}

ಸ್ಥಿರ ಪೂರ್ಣಾಂಕ op_ವಿಂಗಡಿಸು_common(ರಚನೆ htp_ops_context * octx, bool is_ತುದಿ_k) {
    ಆದರೆ (octx->src[0]->type != HTP_TYPE_F32) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * src0 = octx->src[0];
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor * dst  = octx->dst;

    ಆದರೆ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_NO_SUPPORT;
    }

    ಸ್ಥಿರಾಂಕ uint32_t ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಸ್ಥಿರಾಂಕ size_t dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ = dst->ne[0] * ಗಾತ್ರ(int32_t);

    uint32_t ಅಡ್ಡಸಾಲು_ಆರಂಭ = 0;
    uint32_t ಅಡ್ಡಸಾಲು_ಕೊನೆ   = ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
    ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 0;
        htp_tensor_mdev_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk(dst, ಗಾತ್ರ(int32_t), (uint32_t) dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ, &ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk);
        ಸ್ಥಿರಾಂಕ ರಚನೆ htp_tensor_mdev_range range = htp_tensor_mdev_ವಿಭಜನೆ(
            ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ_2, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        ಅಡ್ಡಸಾಲು_ಆರಂಭ = range.ಆರಂಭ;
        ಅಡ್ಡಸಾಲು_ಕೊನೆ   = range.ಆರಂಭ + range.ಎಣಿಕೆ_2;
    }

    ಸ್ಥಿರಾಂಕ uint32_t nrows = ಅಡ್ಡಸಾಲು_ಕೊನೆ - ಅಡ್ಡಸಾಲು_ಆರಂಭ;
    ಆದರೆ (nrows == 0) {
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params kparams_local;
    ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params * kparams = (ಸ್ಥಿರಾಂಕ ರಚನೆ htp_ವಿಂಗಡಿಸು_kernel_params *) octx->kernel_params;

    ಆದರೆ (kparams->n_ಎಳೆಗಳು == 0) {
        ರಚನೆ htp_ವಿಂಗಡಿಸು_vtcm_layout layout;
        ಆದರೆ (!htp_ವಿಂಗಡಿಸು_solve_layout(&layout, ne00, ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು, k, octx->n_ಎಳೆಗಳು, octx->ctx->vtcm_ಗಾತ್ರ, is_ತುದಿ_k)) {
            ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, ಗಾತ್ರ(kparams_local));
        kparams_local.n_ಎಳೆಗಳು         = (int32_t) layout.n_ಎಳೆಗಳು;
        kparams_local.ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು        = (int32_t) ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು;
        kparams_local.ಅಡ್ಡಸಾಲು_ಆರಂಭ         = (int32_t) ಅಡ್ಡಸಾಲು_ಆರಂಭ;
        kparams_local.ಅಡ್ಡಸಾಲು_ಕೊನೆ           = (int32_t) ಅಡ್ಡಸಾಲು_ಕೊನೆ;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.ಕ್ರಮ             = is_ತುದಿ_k ? GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_ತುದಿ_k          = is_ತುದಿ_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_ಗಾತ್ರ         = (int32_t) layout.ಒಟ್ಟು_ಬೈಟುಗಳು;
        kparams_local.phase1_ಸ್ಥಾನ_ಗಾತ್ರ  = (int32_t) layout.phase1_ಸ್ಥಾನ_ಗಾತ್ರ;
        kparams_local.ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off  = (int32_t) layout.ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off;
        kparams_local.ವಿಲೀನ_indices_off = (int32_t) layout.ವಿಲೀನ_indices_off;
        kparams_local.ವಿಲೀನ_elems       = (int32_t) layout.ವಿಲೀನ_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (octx->ctx->mdev.ಎಣಿಕೆ_2 > 1) {
        ಸ್ಮರಣೆ_ನಕಲು(&kparams_local, kparams, ಗಾತ್ರ(kparams_local));
        kparams_local.ಅಡ್ಡಸಾಲು_ಆರಂಭ = (int32_t) ಅಡ್ಡಸಾಲು_ಆರಂಭ;
        kparams_local.ಅಡ್ಡಸಾಲು_ಕೊನೆ   = (int32_t) ಅಡ್ಡಸಾಲು_ಕೊನೆ;
        kparams = &kparams_local;
    }

    ಆದರೆ (octx->ctx->vtcm_ಗಾತ್ರ < (size_t) kparams->vtcm_ಗಾತ್ರ) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_ಗಾತ್ರ, octx->ctx->vtcm_ಗಾತ್ರ);
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_ತಳ = (uint8_t *) octx->ctx->vtcm_ತಳ;

    ಆದರೆ (ಒಟ್ಟು_ಅಡ್ಡಸಾಲುಗಳು == 1 && kparams->n_chunks > 1) {
        ರಚನೆ htp_ವಿಂಗಡಿಸು_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_ತಳ = vtcm_ತಳ;
        memset(cctx.real_ಎಣಿಕೆ, 0, ಗಾತ್ರ(cctx.real_ಎಣಿಕೆ));

        work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, htp_ವಿಂಗಡಿಸು_chunk_ಕೆಲಸ, &cctx, (uint32_t) kparams->n_chunks);

        ಆದರೆ (is_ತುದಿ_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            ಆದರೆ (k == 1 && ne00 >= 128*1024) {
                ದಶಮಾಂಶ *   cand_vals = (ದಶಮಾಂಶ *)   (vtcm_ತಳ + kparams->ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_ತಳ + kparams->ವಿಲೀನ_indices_off);
                ದಶಮಾಂಶ ಉತ್ತಮ_val = cand_vals[0];
                int32_t ಉತ್ತಮ_idx = cand_idxs[0];
                ಗಾಗಿ (int32_t c = 1; c < kparams->n_chunks; c++) {
                    ಆದರೆ (cand_vals[c] > ಉತ್ತಮ_val) {
                        ಉತ್ತಮ_val = cand_vals[c];
                        ಉತ್ತಮ_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->ದತ್ತಾಂಶ;
                dst_ptr[0] = ಉತ್ತಮ_idx;
                ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
            }

            uint32_t ಒಟ್ಟು_candidates = (uint32_t) kparams->n_chunks * local_k;
            ದಶಮಾಂಶ *   cand_vals = (ದಶಮಾಂಶ *)   (vtcm_ತಳ + kparams->ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_ತಳ + kparams->ವಿಲೀನ_indices_off);
            uint32_t ವಿಲೀನ_elems = (uint32_t) kparams->ವಿಲೀನ_elems;

            ಆದರೆ (ವಿಲೀನ_elems > ಒಟ್ಟು_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + ಒಟ್ಟು_candidates), -INFINITY, ವಿಲೀನ_elems - ಒಟ್ಟು_candidates);
                ಗಾಗಿ (uint32_t j = ಒಟ್ಟು_candidates; j < ವಿಲೀನ_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_ವಿಂಗಡಿಸು_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, ವಿಲೀನ_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->ದತ್ತಾಂಶ, cand_idxs, k * ಗಾತ್ರ(int32_t));
        } ಇಲ್ಲದಿದ್ದರೆ {
            bool asc = kparams->ಕ್ರಮ == GGML_ವಿಂಗಡಿಸು_ಕ್ರಮ_ASC;
            ಆದರೆ (kparams->n_chunks == 8) {
                ರಚನೆ htp_argsort_ವಿಲೀನ_l1_ctx l1ctx = { &cctx, asc };
                work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, htp_argsort_ವಿಲೀನ_l1_ಕೆಲಸ, &l1ctx, 4);

                ರಚನೆ htp_argsort_ವಿಲೀನ_l2_ctx l2ctx = { &cctx, asc };
                work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, htp_argsort_ವಿಲೀನ_l2_ಕೆಲಸ, &l2ctx, 2);

                size_t ಸ್ಥಾನ_ಗಾತ್ರ = (size_t) kparams->phase1_ಸ್ಥಾನ_ಗಾತ್ರ;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   val0 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) vtcm_ತಳ;
                ಸ್ಥಿರಾಂಕ int32_t * idx0 = (ಸ್ಥಿರಾಂಕ int32_t *) (vtcm_ತಳ + 4 * chunk_elems * ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   val1 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *) (vtcm_ತಳ + 4 * ಸ್ಥಾನ_ಗಾತ್ರ);
                ಸ್ಥಿರಾಂಕ int32_t * idx1 = (ಸ್ಥಿರಾಂಕ int32_t *) (vtcm_ತಳ + 4 * ಸ್ಥಾನ_ಗಾತ್ರ + 4 * chunk_elems * ಗಾತ್ರ(ದಶಮಾಂಶ));

                uint32_t n01 = cctx.real_ಎಣಿಕೆ[0] + cctx.real_ಎಣಿಕೆ[1] + cctx.real_ಎಣಿಕೆ[2] + cctx.real_ಎಣಿಕೆ[3];
                uint32_t n23 = cctx.real_ಎಣಿಕೆ[4] + cctx.real_ಎಣಿಕೆ[5] + cctx.real_ಎಣಿಕೆ[6] + cctx.real_ಎಣಿಕೆ[7];

                int32_t * final_idx = (int32_t *) (vtcm_ತಳ + kparams->ವಿಲೀನ_indices_off);
                ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ವಿಲೀನ_runs(val0, idx0, n01, val1, idx1, n23, ಖಾಲಿ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->ದತ್ತಾಂಶ, final_idx, ne00 * ಗಾತ್ರ(int32_t));
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kparams->n_chunks == 4) {
                ರಚನೆ htp_argsort_ವಿಲೀನ_l1_ctx l1ctx = { &cctx, asc };
                work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, htp_argsort_ವಿಲೀನ_l1_ಕೆಲಸ, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_ಎಣಿಕೆ[0] + cctx.real_ಎಣಿಕೆ[1];
                uint32_t n23 = cctx.real_ಎಣಿಕೆ[2] + cctx.real_ಎಣಿಕೆ[3];

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   val01 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)   (vtcm_ತಳ + kparams->ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off);
                ಸ್ಥಿರಾಂಕ int32_t * idx01 = (ಸ್ಥಿರಾಂಕ int32_t *) (vtcm_ತಳ + kparams->ವಿಲೀನ_indices_off);
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   val23 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)   (vtcm_ತಳ + kparams->ವಿಲೀನ_ಮೌಲ್ಯಗಳು_off)  + 2 * chunk_elems;
                ಸ್ಥಿರಾಂಕ int32_t * idx23 = (ಸ್ಥಿರಾಂಕ int32_t *) (vtcm_ತಳ + kparams->ವಿಲೀನ_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_ತಳ;
                ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ವಿಲೀನ_runs(val01, idx01, n01, val23, idx23, n23, ಖಾಲಿ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->ದತ್ತಾಂಶ, final_idx, ne00 * ಗಾತ್ರ(int32_t));
            } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (kparams->n_chunks == 2) {
                size_t ಸ್ಥಾನ_ಗಾತ್ರ = (size_t) kparams->phase1_ಸ್ಥಾನ_ಗಾತ್ರ;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   val0 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)   (vtcm_ತಳ);
                ಸ್ಥಿರಾಂಕ int32_t * idx0 = (ಸ್ಥಿರಾಂಕ int32_t *) (vtcm_ತಳ + chunk_elems * ಗಾತ್ರ(ದಶಮಾಂಶ));
                ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *   val1 = (ಸ್ಥಿರಾಂಕ ದಶಮಾಂಶ *)   (vtcm_ತಳ + ಸ್ಥಾನ_ಗಾತ್ರ);
                ಸ್ಥಿರಾಂಕ int32_t * idx1 = (ಸ್ಥಿರಾಂಕ int32_t *) (vtcm_ತಳ + ಸ್ಥಾನ_ಗಾತ್ರ + chunk_elems * ಗಾತ್ರ(ದಶಮಾಂಶ));

                int32_t * final_idx = (int32_t *) (vtcm_ತಳ + kparams->ವಿಲೀನ_indices_off);
                ರಚನೆ htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_ಆರಂಭ(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ವಿಲೀನ_runs(val0, idx0, cctx.real_ಎಣಿಕೆ[0], val1, idx1, cctx.real_ಎಣಿಕೆ[1], ಖಾಲಿ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->ದತ್ತಾಂಶ, final_idx, ne00 * ಗಾತ್ರ(int32_t));
            } ಇಲ್ಲದಿದ್ದರೆ {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_ತಳ + chunk_elems * ಗಾತ್ರ(ದಶಮಾಂಶ));
                dma_sync_write(octx->ctx->dma[0], dst->ದತ್ತಾಂಶ, final_idx, ne00 * ಗಾತ್ರ(int32_t));
            }
        }
        ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
    }

    ರಚನೆ htp_ವಿಂಗಡಿಸು_multi_ಅಡ್ಡಸಾಲು_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_ತಳ        = vtcm_ತಳ;

    uint32_t dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು = (uint32_t) dst_ಅಡ್ಡಸಾಲು_ಗಾತ್ರ;
    uint32_t ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 0;
    htp_tensor_mdev_ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk(dst, ಗಾತ್ರ(int32_t), dst_ಅಡ್ಡಸಾಲು_ಬೈಟುಗಳು, &ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk);
    ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk == 0) {
        ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_ಎಳೆಗಳು - 1) / (uint32_t) kparams->n_ಎಳೆಗಳು;
    ಆದರೆ (ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk > 1) {
        dr = hex_ಸುತ್ತು_ಮೇಲೆ(dr, ಅಡ್ಡಸಾಲುಗಳು_ಪ್ರತಿ_chunk);
    }
    mctx.nrows_ಪ್ರತಿ_thread = dr;

    work_ಸರತಿ_ಸತತ(octx->ctx->work_ಸರತಿ, htp_ವಿಂಗಡಿಸು_multi_ಅಡ್ಡಸಾಲು_ಕೆಲಸ, &mctx, (uint32_t) kparams->n_ಎಳೆಗಳು);

    ಹಿಂತಿರುಗಿ HTP_ಸ್ಥಿತಿ_OK;
}

ಪೂರ್ಣಾಂಕ op_argsort(ರಚನೆ htp_ops_context * octx) {
    ಹಿಂತಿರುಗಿ op_ವಿಂಗಡಿಸು_common(octx, false);
}

ಪೂರ್ಣಾಂಕ op_ತುದಿ_k(ರಚನೆ htp_ops_context * octx) {
    ಹಿಂತಿರುಗಿ op_ವಿಂಗಡಿಸು_common(octx, true);
}
