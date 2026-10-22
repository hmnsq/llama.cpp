#ಅನ್ತರ್ಭಾವ <string.h>
#ಅನ್ತರ್ಭಾವ <stdlib.h>
#ಅನ್ತರ್ಭಾವ <math.h>
#ಅನ್ತರ್ಭಾವ <HAP_farf.h>
#ಅನ್ತರ್ಭಾವ <HAP_perf.h>

#define GGML_COMMON_DECL_C
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

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif


ಸ್ಥಿರ ಶೂನ್ಯ quicksort_ಮೌಲ್ಯಗಳು_indices_asc(ಭಿನ್ನ * ಮೌಲ್ಯಗಳು, int32_t * indices, ಪೂರ್ಣಾಙ್ಕ ಎಡ, ಪೂರ್ಣಾಙ್ಕ ಬಲ) {
    ಯಾವತ್ (ಎಡ < ಬಲ) {
        ಭಿನ್ನ pivot = ಮೌಲ್ಯಗಳು[ಎಡ + (ಬಲ - ಎಡ) / 2];
        ಪೂರ್ಣಾಙ್ಕ i = ಎಡ;
        ಪೂರ್ಣಾಙ್ಕ j = ಬಲ;

        ಯಾವತ್ (i <= j) {
            ಯಾವತ್ (ಮೌಲ್ಯಗಳು[i] < pivot) i++;
            ಯಾವತ್ (ಮೌಲ್ಯಗಳು[j] > pivot) j--;

            ಯದಿ (i <= j) {
                ಭಿನ್ನ tmp_val = ಮೌಲ್ಯಗಳು[i];
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
        ಯದಿ (j - ಎಡ < ಬಲ - i) {
            ಯದಿ (ಎಡ < j) quicksort_ಮೌಲ್ಯಗಳು_indices_asc(ಮೌಲ್ಯಗಳು, indices, ಎಡ, j);
            ಎಡ = i;
        } ಅನ್ಯಥಾ {
            ಯದಿ (i < ಬಲ) quicksort_ಮೌಲ್ಯಗಳು_indices_asc(ಮೌಲ್ಯಗಳು, indices, i, ಬಲ);
            ಬಲ = j;
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ quicksort_ಮೌಲ್ಯಗಳು_indices_desc(ಭಿನ್ನ * ಮೌಲ್ಯಗಳು, int32_t * indices, ಪೂರ್ಣಾಙ್ಕ ಎಡ, ಪೂರ್ಣಾಙ್ಕ ಬಲ) {
    ಯಾವತ್ (ಎಡ < ಬಲ) {
        ಭಿನ್ನ pivot = ಮೌಲ್ಯಗಳು[ಎಡ + (ಬಲ - ಎಡ) / 2];
        ಪೂರ್ಣಾಙ್ಕ i = ಎಡ;
        ಪೂರ್ಣಾಙ್ಕ j = ಬಲ;

        ಯಾವತ್ (i <= j) {
            ಯಾವತ್ (ಮೌಲ್ಯಗಳು[i] > pivot) i++;
            ಯಾವತ್ (ಮೌಲ್ಯಗಳು[j] < pivot) j--;

            ಯದಿ (i <= j) {
                ಭಿನ್ನ tmp_val = ಮೌಲ್ಯಗಳು[i];
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
        ಯದಿ (j - ಎಡ < ಬಲ - i) {
            ಯದಿ (ಎಡ < j) quicksort_ಮೌಲ್ಯಗಳು_indices_desc(ಮೌಲ್ಯಗಳು, indices, ಎಡ, j);
            ಎಡ = i;
        } ಅನ್ಯಥಾ {
            ಯದಿ (i < ಬಲ) quicksort_ಮೌಲ್ಯಗಳು_indices_desc(ಮೌಲ್ಯಗಳು, indices, i, ಬಲ);
            ಬಲ = j;
        }
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ uint32_t ತುದಿ_k_max_ಮೌಲ್ಯ_index(ಧ್ರುವ ಭಿನ್ನ * ಮೌಲ್ಯಗಳು, uint32_t n, ಭಿನ್ನ * ಮೌಲ್ಯ) {
    int32_t max_idx = 0;
    hvx_argmax_f32(ಮೌಲ್ಯಗಳು, n, 0, ಮೌಲ್ಯ, &max_idx);
    ಪ್ರತ್ಯಾಗಮ (uint32_t) max_idx;
}

ಸ್ಥಿರ ಧ್ರುವ int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ init_indices_ramp(int32_t * indices_buf, uint32_t ಎಣಿಕೆ, uint32_t base) {
    ಧ್ರುವ HVX_Vector ind_init_vec = Q6_Vw_vadd_VwVw(*(ಧ್ರುವ HVX_Vector *) argsort_ramp_lut, Q6_V_vsplat_R(base));
    ಧ್ರುವ HVX_Vector ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_Vector * indices_buf_vec = (HVX_Vector *) indices_buf;
    uint32_t ಅಂಕೆ_vecs = ಎಣಿಕೆ / 32;
    HVX_Vector curr_ind_vec = ind_init_vec;
    ಪ್ರತಿ (uint32_t j = 0; j < ಅಂಕೆ_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = ಎಣಿಕೆ % 32;
    ಯದಿ (rem > 0) {
        hvx_vec_store_a((ಶೂನ್ಯ *) &indices_buf_vec[ಅಂಕೆ_vecs], rem * ಪರಿಮಾಣ(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ vec_cas(HVX_Vector * X_val, HVX_Vector * X_idx, HVX_Vector * Y_val, HVX_Vector * Y_idx, bool asc) {
    HVX_VectorPred pred = asc ? Q6_Q_vcmp_gt_VsfVsf(*X_val, *Y_val)
                              : Q6_Q_vcmp_gt_VsfVsf(*Y_val, *X_val);
    HVX_Vector next_X_val = Q6_V_vmux_QVV(pred, *Y_val, *X_val);
    HVX_Vector next_Y_val = Q6_V_vmux_QVV(pred, *X_val, *Y_val);
    HVX_Vector next_X_idx = Q6_V_vmux_QVV(pred, *Y_idx, *X_idx);
    HVX_Vector Y_tmp_idx  = Q6_V_vmux_QVV(pred, *X_idx, *Y_idx);
    *X_val = next_X_val;
    *Y_val = next_Y_val;
    *X_idx = next_X_idx;
    *Y_idx = Y_tmp_idx;
}

__attribute__((always_inline))
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ bitonic_cas_32(HVX_Vector * V, HVX_Vector * I, ಪೂರ್ಣಾಙ್ಕ d, HVX_VectorPred dir_mask, HVX_Vector idx_vec, HVX_Vector zero_vec) {
    HVX_VectorPred mask_ಎಡ;
    HVX_Vector V_rot_ಎಡ, V_rot_ಬಲ;
    HVX_Vector I_rot_ಎಡ, I_rot_ಬಲ;

    ಯದಿ (d == 1) {
        mask_ಎಡ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_ಎಡ = Q6_V_vror_VR(*V, 4);
        V_rot_ಬಲ = Q6_V_vror_VR(*V, 124);
        I_rot_ಎಡ = Q6_V_vror_VR(*I, 4);
        I_rot_ಬಲ = Q6_V_vror_VR(*I, 124);
    } ಅನ್ಯಥಾ ಯದಿ (d == 2) {
        mask_ಎಡ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_ಎಡ = Q6_V_vror_VR(*V, 8);
        V_rot_ಬಲ = Q6_V_vror_VR(*V, 120);
        I_rot_ಎಡ = Q6_V_vror_VR(*I, 8);
        I_rot_ಬಲ = Q6_V_vror_VR(*I, 120);
    } ಅನ್ಯಥಾ ಯದಿ (d == 4) {
        mask_ಎಡ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_ಎಡ = Q6_V_vror_VR(*V, 16);
        V_rot_ಬಲ = Q6_V_vror_VR(*V, 112);
        I_rot_ಎಡ = Q6_V_vror_VR(*I, 16);
        I_rot_ಬಲ = Q6_V_vror_VR(*I, 112);
    } ಅನ್ಯಥಾ ಯದಿ (d == 8) {
        mask_ಎಡ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_ಎಡ = Q6_V_vror_VR(*V, 32);
        V_rot_ಬಲ = Q6_V_vror_VR(*V, 96);
        I_rot_ಎಡ = Q6_V_vror_VR(*I, 32);
        I_rot_ಬಲ = Q6_V_vror_VR(*I, 96);
    } ಅನ್ಯಥಾ {
        mask_ಎಡ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_ಎಡ = Q6_V_vror_VR(*V, 64);
        V_rot_ಬಲ = Q6_V_vror_VR(*V, 64);
        I_rot_ಎಡ = Q6_V_vror_VR(*I, 64);
        I_rot_ಬಲ = Q6_V_vror_VR(*I, 64);
    }

    HVX_Vector V_paired = Q6_V_vmux_QVV(mask_ಎಡ, V_rot_ಎಡ, V_rot_ಬಲ);
    HVX_Vector I_paired = Q6_V_vmux_QVV(mask_ಎಡ, I_rot_ಎಡ, I_rot_ಬಲ);

    HVX_VectorPred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_VectorPred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_VectorPred mask_ಬಲ = Q6_Q_not_Q(mask_ಎಡ);
    HVX_VectorPred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(mask_ಎಡ, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, mask_ಬಲ)
    );
    HVX_VectorPred Q_swap = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_mask, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_mask), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_swap, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_swap, I_paired, *I);
}

__attribute__((always_inline))
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ bitonic_sort_generic_hvx(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, ಪೂರ್ಣಾಙ್ಕ K, bool asc_order) {
    HVX_Vector V[32];
    HVX_Vector I[32];

    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(ಧ್ರುವ HVX_Vector *) argsort_ramp_lut;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ v = 0; v < K; v++) {
        V[v] = *(HVX_Vector *)(ಮೌಲ್ಯಗಳು + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    ಪೂರ್ಣಾಙ್ಕ M = 5;
    ಯಾವತ್ ((1 << (M - 5)) < K) M++;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 1; s <= M; s++) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ stage_d = s - 1; stage_d >= 0; stage_d--) {
            ಪೂರ್ಣಾಙ್ಕ d = 1 << stage_d;
            ಯದಿ (d >= 32) {
                ಪೂರ್ಣಾಙ್ಕ v_dist = d / 32;
                ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ v1 = 0; v1 < K; v1++) {
                    ಯದಿ ((v1 & v_dist) == 0) {
                        ಪೂರ್ಣಾಙ್ಕ v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_order;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } ಅನ್ಯಥಾ {
                ಯದಿ (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                } ಅನ್ಯಥಾ {
                    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_order;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ v = 0; v < K; v++) {
        *(HVX_Vector *)(ಮೌಲ್ಯಗಳು + v * 128)  = V[v];
        *(HVX_Vector *)(indices + v * 128) = I[v];
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ bitonic_sort_vtcm_desc(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(ಧ್ರುವ HVX_Vector *) argsort_ramp_lut;

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    ಯದಿ (init_indices) {
        ಪ್ರತಿ (uint32_t v = 0; v < n_vec; v++) {
            HVX_Vector idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_Vector *)(indices + v * 128) = idx;
        }
    }

    ಪೂರ್ಣಾಙ್ಕ M = 5;
    ಯಾವತ್ ((1u << (M - 5)) < n_vec) M++;

    ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ s = 1; s <= M; s++) {
        ಪ್ರತಿ (ಪೂರ್ಣಾಙ್ಕ stage_d = s - 1; stage_d >= 0; stage_d--) {
            ಪೂರ್ಣಾಙ್ಕ d = 1 << stage_d;
            ಯದಿ (d >= 32) {
                uint32_t v_dist = d / 32;
                ಪ್ರತಿ (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    ಯದಿ ((v1 & v_dist) == 0) {
                        uint32_t v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : false;

                        HVX_Vector Vv1 = *(HVX_Vector *)(ಮೌಲ್ಯಗಳು + v1 * 128);
                        HVX_Vector Iv1 = *(HVX_Vector *)(indices + v1 * 128);
                        HVX_Vector Vv2 = *(HVX_Vector *)(ಮೌಲ್ಯಗಳು + v2 * 128);
                        HVX_Vector Iv2 = *(HVX_Vector *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_Vector *)(ಮೌಲ್ಯಗಳು + v1 * 128)  = Vv1;
                        *(HVX_Vector *)(indices + v1 * 128) = Iv1;
                        *(HVX_Vector *)(ಮೌಲ್ಯಗಳು + v2 * 128)  = Vv2;
                        *(HVX_Vector *)(indices + v2 * 128) = Iv2;
                    }
                }
            } ಅನ್ಯಥಾ {
                ಯದಿ (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    ಪ್ರತಿ (uint32_t v = 0; v < n_vec; v++) {
                        HVX_Vector Vv = *(HVX_Vector *)(ಮೌಲ್ಯಗಳು + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(ಮೌಲ್ಯಗಳು + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                } ಅನ್ಯಥಾ {
                    ಪ್ರತಿ (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;

                        HVX_Vector Vv = *(HVX_Vector *)(ಮೌಲ್ಯಗಳು + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(ಮೌಲ್ಯಗಳು + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ ತುದಿ_k_select_tiled(dma_queue * dma_q, ಸಂರಚನಾ htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               ಭಿನ್ನ * ಮೌಲ್ಯಗಳು_buf, int32_t * indices_buf,
                               ಭಿನ್ನ * out_ಮೌಲ್ಯಗಳು, int32_t * out_indices,
                               uint32_t chunk_base) {
    ಧ್ರುವ uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_ಎಣಿಕೆ    = n_tiles * k;
    uint32_t merge_n_vec        = hmx_ceil_div(candidate_ಎಣಿಕೆ, 32);
    uint32_t merge_n_vec_pow2   = 1;
    ಯಾವತ್ (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
    uint32_t merge_elems        = merge_n_vec_pow2 * 32;
    ಭಿನ್ನ * candidate_ಮೌಲ್ಯಗಳು    = ಮೌಲ್ಯಗಳು_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    ಪ್ರತಿ (uint32_t offset = 0; offset < n; offset += tile_elems) {
        uint32_t tile_ಎಣಿಕೆ = MIN(tile_elems, n - offset);
        dma_sync_read(dma_q, ಮೌಲ್ಯಗಳು_buf, src_addr + (size_t) offset * ಪರಿಮಾಣ(ಭಿನ್ನ), tile_ಎಣಿಕೆ * ಪರಿಮಾಣ(ಭಿನ್ನ));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
        ಯದಿ (tile_ಎಣಿಕೆ < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (ಮೌಲ್ಯಗಳು_buf + tile_ಎಣಿಕೆ), -INFINITY, tile_elems - tile_ಎಣಿಕೆ);
        }

        bitonic_sort_vtcm_desc((uint8_t *) ಮೌಲ್ಯಗಳು_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_ಎಣಿಕೆ);
        ಪ್ರತಿ (uint32_t j = 0; j < tile_k; j++) {
            candidate_ಮೌಲ್ಯಗಳು[candidate_pos] = ಮೌಲ್ಯಗಳು_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) offset;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    ಯದಿ (merge_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_ಮೌಲ್ಯಗಳು + candidate_pos), -INFINITY, merge_elems - candidate_pos);
        ಪ್ರತಿ (uint32_t j = candidate_pos; j < merge_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_sort_vtcm_desc((uint8_t *) candidate_ಮೌಲ್ಯಗಳು, (uint8_t *) candidate_indices, merge_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    ಪ್ರತಿ (uint32_t j = 0; j < k; j++) {
        out_ಮೌಲ್ಯಗಳು[j]  = candidate_ಮೌಲ್ಯಗಳು[j];
        out_indices[j] = candidate_indices[j] + (int32_t) chunk_base;
    }
}

__attribute__((always_inline))
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ sort32_f32_hvx(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, ಗಣನಾ ggml_sort_order order) {
    bitonic_sort_generic_hvx(ಮೌಲ್ಯಗಳು, indices, 1, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ sort64_f32_hvx(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, ಗಣನಾ ggml_sort_order order) {
    bitonic_sort_generic_hvx(ಮೌಲ್ಯಗಳು, indices, 2, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ sort128_f32_hvx(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, ಗಣನಾ ggml_sort_order order) {
    bitonic_sort_generic_hvx(ಮೌಲ್ಯಗಳು, indices, 4, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ sort256_f32_hvx(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, ಗಣನಾ ggml_sort_order order) {
    bitonic_sort_generic_hvx(ಮೌಲ್ಯಗಳು, indices, 8, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ sort512_f32_hvx(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, ಗಣನಾ ggml_sort_order order) {
    bitonic_sort_generic_hvx(ಮೌಲ್ಯಗಳು, indices, 16, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ sort1024_f32_hvx(uint8_t * ಮೌಲ್ಯಗಳು, uint8_t * indices, ಗಣನಾ ggml_sort_order order) {
    bitonic_sort_generic_hvx(ಮೌಲ್ಯಗಳು, indices, 32, order == GGML_SORT_ORDER_ASC);
}

ಸ್ಥಿರ ಶೂನ್ಯ merge_runs(
    ಧ್ರುವ ಭಿನ್ನ * restrict in_val0, ಧ್ರುವ int32_t * restrict in_idx0, uint32_t n0,
    ಧ್ರುವ ಭಿನ್ನ * restrict in_val1, ಧ್ರುವ int32_t * restrict in_idx1, uint32_t n1,
    ಭಿನ್ನ * restrict out_val, int32_t * restrict out_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t out = 0;

    ಯದಿ (asc) {
        ಯಾವತ್ (i0 < n0 && i1 < n1) {
            ಯದಿ (in_val0[i0] <= in_val1[i1]) {
                ಯದಿ (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } ಅನ್ಯಥಾ {
                ಯದಿ (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    } ಅನ್ಯಥಾ {
        ಯಾವತ್ (i0 < n0 && i1 < n1) {
            ಯದಿ (in_val0[i0] >= in_val1[i1]) {
                ಯದಿ (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } ಅನ್ಯಥಾ {
                ಯದಿ (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    }

    ಯಾವತ್ (i0 < n0) {
        ಯದಿ (out_val) out_val[out] = in_val0[i0];
        out_idx[out] = in_idx0[i0];
        i0++;
        out++;
    }

    ಯಾವತ್ (i1 < n1) {
        ಯದಿ (out_val) out_val[out] = in_val1[i1];
        out_idx[out] = in_idx1[i1];
        i1++;
        out++;
    }
}

ಸಂರಚನಾ htp_sort_chunk_ctx {
    ಸಂರಚನಾ htp_ops_context *              octx;
    ಧ್ರುವ ಸಂರಚನಾ htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              real_ಎಣಿಕೆ[HTP_MAX_NTHREADS];
};

ಸ್ಥಿರ ಶೂನ್ಯ htp_sort_chunk_job(unsigned ಪೂರ್ಣಾಙ್ಕ n, unsigned ಪೂರ್ಣಾಙ್ಕ i, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ htp_sort_chunk_ctx * cctx = (ಸಂರಚನಾ htp_sort_chunk_ctx *) data;
    ಸಂರಚನಾ htp_ops_context * octx = cctx->octx;
    ಧ್ರುವ ಸಂರಚನಾ htp_sort_kernel_params * kparams = cctx->kparams;
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_base  = i * chunk_elems;

    uint32_t real_ಎಣಿಕೆ = (chunk_base < ne00) ? MIN(chunk_elems, ne00 - chunk_base) : 0;
    cctx->real_ಎಣಿಕೆ[i] = real_ಎಣಿಕೆ;

    uint8_t * spad = cctx->vtcm_base + (size_t) kparams->phase1_slot_size * i;
    ಭಿನ್ನ *   ಮೌಲ್ಯಗಳು_buf  = (ಭಿನ್ನ *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಭಿನ್ನ *   merge_ಮೌಲ್ಯಗಳು  = (ಭಿನ್ನ *)   (cctx->vtcm_base + kparams->merge_ಮೌಲ್ಯಗಳು_off);
    int32_t * merge_indices = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off);

    ಸಂರಚನಾ htp_thread_trace * tr = &octx->ctx->trace[i];
    ಧ್ರುವ dma_addr_t src_addr = src0->data + (size_t) chunk_base * ಪರಿಮಾಣ(ಭಿನ್ನ);

    ಯದಿ (kparams->is_ತುದಿ_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        ಯದಿ (real_ಎಣಿಕೆ == 0) {
            ಪ್ರತಿ (uint32_t j = 0; j < local_k; j++) {
                merge_ಮೌಲ್ಯಗಳು[i * local_k + j]  = -INFINITY;
                merge_indices[i * local_k + j] = 0;
            }
            ಪ್ರತ್ಯಾಗಮ;
        }

        ಯದಿ (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], ಮೌಲ್ಯಗಳು_buf, src_addr, real_ಎಣಿಕೆ * ಪರಿಮಾಣ(ಭಿನ್ನ));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            ಭಿನ್ನ max_ಮೌಲ್ಯ;
            uint32_t max_index = ತುದಿ_k_max_ಮೌಲ್ಯ_index(ಮೌಲ್ಯಗಳು_buf, real_ಎಣಿಕೆ, &max_ಮೌಲ್ಯ);
            merge_ಮೌಲ್ಯಗಳು[i] = max_ಮೌಲ್ಯ;
            merge_indices[i] = (int32_t) (max_index + chunk_base);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            ಪ್ರತ್ಯಾಗಮ;
        }

        ಯದಿ (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            ತುದಿ_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_ಎಣಿಕೆ, local_k,
                               ಮೌಲ್ಯಗಳು_buf, indices_buf,
                               &merge_ಮೌಲ್ಯಗಳು[i * local_k], &merge_indices[i * local_k],
                               chunk_base);
            ಪ್ರತ್ಯಾಗಮ;
        }

        dma_sync_read(octx->ctx->dma[i], ಮೌಲ್ಯಗಳು_buf, src_addr, real_ಎಣಿಕೆ * ಪರಿಮಾಣ(ಭಿನ್ನ));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        ಯದಿ (chunk_elems > real_ಎಣಿಕೆ) {
            hvx_splat_f32_u((uint8_t *)(ಮೌಲ್ಯಗಳು_buf + real_ಎಣಿಕೆ), -INFINITY, chunk_elems - real_ಎಣಿಕೆ);
        }

        bitonic_sort_vtcm_desc((uint8_t *) ಮೌಲ್ಯಗಳು_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        ಪ್ರತಿ (uint32_t j = 0; j < local_k; j++) {
            merge_ಮೌಲ್ಯಗಳು[i * local_k + j]  = ಮೌಲ್ಯಗಳು_buf[j];
            merge_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_base;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } ಅನ್ಯಥಾ {
        ಯದಿ (real_ಎಣಿಕೆ > 0) {
            dma_sync_read(octx->ctx->dma[i], ಮೌಲ್ಯಗಳು_buf, src_addr, real_ಎಣಿಕೆ * ಪರಿಮಾಣ(ಭಿನ್ನ));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_ಎಣಿಕೆ, chunk_base);

            ಯದಿ (kparams->order == GGML_SORT_ORDER_ASC) {
                quicksort_ಮೌಲ್ಯಗಳು_indices_asc(ಮೌಲ್ಯಗಳು_buf, indices_buf, 0, real_ಎಣಿಕೆ - 1);
            } ಅನ್ಯಥಾ {
                quicksort_ಮೌಲ್ಯಗಳು_indices_desc(ಮೌಲ್ಯಗಳು_buf, indices_buf, 0, real_ಎಣಿಕೆ - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

ಸಂರಚನಾ htp_argsort_merge_l1_ctx {
    ಸಂರಚನಾ htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

ಸ್ಥಿರ ಶೂನ್ಯ htp_argsort_merge_l1_job(unsigned ಪೂರ್ಣಾಙ್ಕ n, unsigned ಪೂರ್ಣಾಙ್ಕ i, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ htp_argsort_merge_l1_ctx * mctx = (ಸಂರಚನಾ htp_argsort_merge_l1_ctx *) data;
    ಸಂರಚನಾ htp_sort_chunk_ctx * cctx = mctx->cctx;
    ಧ್ರುವ ಸಂರಚನಾ htp_sort_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_ಎಣಿಕೆ[c0];
    uint32_t count1 = cctx->real_ಎಣಿಕೆ[c1];

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    ಧ್ರುವ ಭಿನ್ನ *   val0 = (ಧ್ರುವ ಭಿನ್ನ *)   (cctx->vtcm_base + slot_size * c0);
    ಧ್ರುವ int32_t * idx0 = (ಧ್ರುವ int32_t *) (cctx->vtcm_base + slot_size * c0 + chunk_elems * ಪರಿಮಾಣ(ಭಿನ್ನ));
    ಧ್ರುವ ಭಿನ್ನ *   val1 = (ಧ್ರುವ ಭಿನ್ನ *)   (cctx->vtcm_base + slot_size * c1);
    ಧ್ರುವ int32_t * idx1 = (ಧ್ರುವ int32_t *) (cctx->vtcm_base + slot_size * c1 + chunk_elems * ಪರಿಮಾಣ(ಭಿನ್ನ));

    uint32_t out_offset = (2 * i) * chunk_elems;
    ಭಿನ್ನ *   out_val = (ಭಿನ್ನ *)   (cctx->vtcm_base + kparams->merge_ಮೌಲ್ಯಗಳು_off)  + out_offset;
    int32_t * out_idx = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + out_offset;

    ಸಂರಚನಾ htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

ಸಂರಚನಾ htp_argsort_merge_l2_ctx {
    ಸಂರಚನಾ htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

ಸ್ಥಿರ ಶೂನ್ಯ htp_argsort_merge_l2_job(unsigned ಪೂರ್ಣಾಙ್ಕ n, unsigned ಪೂರ್ಣಾಙ್ಕ i, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ htp_argsort_merge_l2_ctx * mctx = (ಸಂರಚನಾ htp_argsort_merge_l2_ctx *) data;
    ಸಂರಚನಾ htp_sort_chunk_ctx * cctx = mctx->cctx;
    ಧ್ರುವ ಸಂರಚನಾ htp_sort_kernel_params * kparams = cctx->kparams;

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    ಧ್ರುವ ಭಿನ್ನ *   val0 = (ಧ್ರುವ ಭಿನ್ನ *)   (cctx->vtcm_base + kparams->merge_ಮೌಲ್ಯಗಳು_off)  + off0;
    ಧ್ರುವ int32_t * idx0 = (ಧ್ರುವ int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off0;
    ಧ್ರುವ ಭಿನ್ನ *   val1 = (ಧ್ರುವ ಭಿನ್ನ *)   (cctx->vtcm_base + kparams->merge_ಮೌಲ್ಯಗಳು_off)  + off1;
    ಧ್ರುವ int32_t * idx1 = (ಧ್ರುವ int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off1;

    uint32_t count0 = cctx->real_ಎಣಿಕೆ[4*i + 0] + cctx->real_ಎಣಿಕೆ[4*i + 1];
    uint32_t count1 = cctx->real_ಎಣಿಕೆ[4*i + 2] + cctx->real_ಎಣಿಕೆ[4*i + 3];

    uint8_t * slot_base = cctx->vtcm_base + (4 * i) * slot_size;
    ಭಿನ್ನ *   out_val   = (ಭಿನ್ನ *) slot_base;
    int32_t * out_idx   = (int32_t *) (slot_base + 4 * chunk_elems * ಪರಿಮಾಣ(ಭಿನ್ನ));

    ಸಂರಚನಾ htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

ಸಂರಚನಾ htp_sort_multi_row_ctx {
    ಸಂರಚನಾ htp_ops_context *              octx;
    ಧ್ರುವ ಸಂರಚನಾ htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              nrows_per_thread;
};

ಸಂರಚನಾ htp_tensor_row_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಸಂರಚನಾ htp_tensor_row_coord init_row_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    ಪ್ರತ್ಯಾಗಮ (ಸಂರಚನಾ htp_tensor_row_coord){ i01, i02, i03 };
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ step_row_coord(ಸಂರಚನಾ htp_tensor_row_coord * c, uint32_t ne01, uint32_t ne02) {
    c->i01++;
    ಯದಿ (c->i01 == ne01) {
        c->i01 = 0;
        c->i02++;
        ಯದಿ (c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ size_t coord_offset(ಸಂರಚನಾ htp_tensor_row_coord c, size_t nb1, size_t nb2, size_t nb3) {
    ಪ್ರತ್ಯಾಗಮ (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

ಸ್ಥಿರ ಅನ್ತಃಸ್ಥ ಶೂನ್ಯ compute_row_sort(
    ಭಿನ್ನ * cur_ಮೌಲ್ಯಗಳು,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_ತುದಿ_k,
    ಗಣನಾ ggml_sort_order order
) {
    ಯದಿ (is_ತುದಿ_k) {
        ಯದಿ (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            ವಿಕಲ್ಪ (ne00) {
                ಪಕ್ಷ 32:   sort32_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ಭಙ್ಗ;
                ಪಕ್ಷ 64:   sort64_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ಭಙ್ಗ;
                ಪಕ್ಷ 128:  sort128_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ಭಙ್ಗ;
                ಪಕ್ಷ 256:  sort256_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ಭಙ್ಗ;
                ಪಕ್ಷ 512:  sort512_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ಭಙ್ಗ;
                ಪಕ್ಷ 1024: sort1024_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ಭಙ್ಗ;
            }
        } ಅನ್ಯಥಾ {
            ಯದಿ (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_ಮೌಲ್ಯಗಳು + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_sort_vtcm_desc((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } ಅನ್ಯಥಾ {
        ಯದಿ (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            ವಿಕಲ್ಪ (ne00) {
                ಪಕ್ಷ 32:   sort32_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, order); ಭಙ್ಗ;
                ಪಕ್ಷ 64:   sort64_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, order); ಭಙ್ಗ;
                ಪಕ್ಷ 128:  sort128_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, order); ಭಙ್ಗ;
                ಪಕ್ಷ 256:  sort256_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, order); ಭಙ್ಗ;
                ಪಕ್ಷ 512:  sort512_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, order); ಭಙ್ಗ;
                ಪಕ್ಷ 1024: sort1024_f32_hvx((uint8_t *) cur_ಮೌಲ್ಯಗಳು, (uint8_t *) cur_indices, order); ಭಙ್ಗ;
            }
        } ಅನ್ಯಥಾ {
            init_indices_ramp(cur_indices, ne00, 0);
            ಯದಿ (order == GGML_SORT_ORDER_ASC) {
                quicksort_ಮೌಲ್ಯಗಳು_indices_asc(cur_ಮೌಲ್ಯಗಳು, cur_indices, 0, ne00 - 1);
            } ಅನ್ಯಥಾ {
                quicksort_ಮೌಲ್ಯಗಳು_indices_desc(cur_ಮೌಲ್ಯಗಳು, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

ಸ್ಥಿರ ಶೂನ್ಯ htp_sort_multi_row_job(unsigned ಪೂರ್ಣಾಙ್ಕ n, unsigned ಪೂರ್ಣಾಙ್ಕ ith, ಶೂನ್ಯ * data) {
    ಸಂರಚನಾ htp_sort_multi_row_ctx * mctx = (ಸಂರಚನಾ htp_sort_multi_row_ctx *) data;
    ಸಂರಚನಾ htp_ops_context * octx = mctx->octx;
    ಧ್ರುವ ಸಂರಚನಾ htp_sort_kernel_params * kparams = mctx->kparams;
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst  = octx->dst;

    uint32_t ne00 = (uint32_t) kparams->ne00;
    uint32_t ne01 = src0->ne[1];
    uint32_t ne02 = src0->ne[2];

    uint32_t nb01 = src0->nb[1];
    uint32_t nb02 = src0->nb[2];
    uint32_t nb03 = src0->nb[3];

    uint32_t nb1 = dst->nb[1];
    uint32_t nb2 = dst->nb[2];
    uint32_t nb3 = dst->nb[3];

    uint32_t row_start = (uint32_t) kparams->row_start;
    uint32_t row_end   = (uint32_t) kparams->row_end;
    uint32_t r_start   = row_start + ith * mctx->nrows_per_thread;
    uint32_t r_end     = MIN(r_start + mctx->nrows_per_thread, row_end);

    ಯದಿ (r_start >= r_end) ಪ್ರತ್ಯಾಗಮ;

    size_t spad_offset = ith * (size_t) (kparams->phase1_slot_size * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_base + spad_offset;
    size_t slot_size = (size_t) kparams->phase1_slot_size;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    ಭಿನ್ನ *   ಮೌಲ್ಯಗಳು_buf  = (ಭಿನ್ನ *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * ಪರಿಮಾಣ(ಭಿನ್ನ));

    dma_queue * dma_q = octx->ctx->dma[ith];
    ಸಂರಚನಾ htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_ತುದಿ_k = kparams->is_ತುದಿ_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    ಗಣನಾ ggml_sort_order order = (ಗಣನಾ ggml_sort_order) kparams->order;

    ಧ್ರುವ uint32_t src_bytes = ne00 * ಪರಿಮಾಣ(ಭಿನ್ನ);
    ಧ್ರುವ uint32_t dst_bytes = (is_ತುದಿ_k ? k : ne00) * ಪರಿಮಾಣ(int32_t);

    ಯದಿ (kparams->n_slots == 2) {
        ಸಂರಚನಾ htp_tensor_row_coord ಅವಿಭಾಜ್ಯ_coord = init_row_coord(r_start, ne01, ne02);
        ಪ್ರತಿ (uint32_t r = r_start, spad_idx = 0; r < r_end && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * slot_size;
            ಭಿನ್ನ *   cur_ಮೌಲ್ಯಗಳು  = (ಭಿನ್ನ *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * ಪರಿಮಾಣ(ಭಿನ್ನ));

            dma_addr_t dst_addr = dst->data + coord_offset(ಅವಿಭಾಜ್ಯ_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->data + coord_offset(ಅವಿಭಾಜ್ಯ_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_queue_ತಳ್ಳು(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 0);

            // Prefetch input row
            dma_queue_ತಳ್ಳು(dma_q, dma_make_data(cur_ಮೌಲ್ಯಗಳು, src_addr),
                           src_bytes, src_bytes, src_bytes, 1);

            step_row_coord(&ಅವಿಭಾಜ್ಯ_coord, ne01, ne02);
        }

        ಸಂರಚನಾ htp_tensor_row_coord dst_coord = init_row_coord(r_start, ne01, ne02);
        ಸಂರಚನಾ htp_tensor_row_coord next_src_coord = init_row_coord(r_start + 2, ne01, ne02);

        ಪ್ರತಿ (uint32_t r = r_start; r < r_end; r++) {
            uint32_t cur_slot = (r - r_start) & 1;
            uint8_t * cur_spad = thread_spad + cur_slot * slot_size;
            ಭಿನ್ನ *   cur_ಮೌಲ್ಯಗಳು  = (ಭಿನ್ನ *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * ಪರಿಮಾಣ(ಭಿನ್ನ));

            // Wait for previous writeback of this slot to complete
            dma_queue_ತೆಗೆ(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_queue_ತೆಗೆ(dma_q);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(cur_ಮೌಲ್ಯಗಳು, cur_indices, ne00, chunk_elems, is_ತುದಿ_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->data + coord_offset(dst_coord, nb1, nb2, nb3);
            dma_queue_ತಳ್ಳು(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 1);
            step_row_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            ಧ್ರುವ uint32_t next_row = r + 2;
            ಯದಿ (next_row < r_end) {
                dma_addr_t next_src_addr = src0->data + coord_offset(next_src_coord, nb01, nb02, nb03);
                dma_queue_ತಳ್ಳು(dma_q, dma_make_data(cur_ಮೌಲ್ಯಗಳು, next_src_addr),
                               src_bytes, src_bytes, src_bytes, 1);
                step_row_coord(&next_src_coord, ne01, ne02);
            }
        }

        dma_queue_flush(dma_q);
    } ಅನ್ಯಥಾ {
        ಸಂರಚನಾ htp_tensor_row_coord coord = init_row_coord(r_start, ne01, ne02);
        ಪ್ರತಿ (uint32_t r = r_start; r < r_end; r++) {
            dma_addr_t src_addr = src0->data + coord_offset(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->data + coord_offset(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, ಮೌಲ್ಯಗಳು_buf, src_addr, src_bytes);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(ಮೌಲ್ಯಗಳು_buf, indices_buf, ne00, chunk_elems, is_ತುದಿ_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_bytes);
            step_row_coord(&coord, ne01, ne02);
        }
    }
}

ಸ್ಥಿರ ಪೂರ್ಣಾಙ್ಕ op_sort_common(ಸಂರಚನಾ htp_ops_context * octx, bool is_ತುದಿ_k) {
    ಯದಿ (octx->src[0]->type != HTP_TYPE_F32) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * src0 = octx->src[0];
    ಧ್ರುವ ಸಂರಚನಾ htp_tensor * dst  = octx->dst;

    ಯದಿ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_NO_SUPPORT;
    }

    ಧ್ರುವ uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ಧ್ರುವ size_t dst_row_size = dst->ne[0] * ಪರಿಮಾಣ(int32_t);

    uint32_t row_start = 0;
    uint32_t row_end   = total_rows;
    ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, ಪರಿಮಾಣ(int32_t), (uint32_t) dst_row_size, &rows_per_chunk);
        ಧ್ರುವ ಸಂರಚನಾ htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ಎಣಿಕೆ, &octx->ctx->mdev.ಎಣಿಕೆ_div);
        row_start = range.start;
        row_end   = range.start + range.ಎಣಿಕೆ;
    }

    ಧ್ರುವ uint32_t nrows = row_end - row_start;
    ಯದಿ (nrows == 0) {
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    ಸಂರಚನಾ htp_sort_kernel_params kparams_local;
    ಧ್ರುವ ಸಂರಚನಾ htp_sort_kernel_params * kparams = (ಧ್ರುವ ಸಂರಚನಾ htp_sort_kernel_params *) octx->kernel_params;

    ಯದಿ (kparams->n_threads == 0) {
        ಸಂರಚನಾ htp_sort_vtcm_layout layout;
        ಯದಿ (!htp_sort_solve_layout(&layout, ne00, total_rows, k, octx->n_threads, octx->ctx->vtcm_size, is_ತುದಿ_k)) {
            ಪ್ರತ್ಯಾಗಮ HTP_STATUS_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, ಪರಿಮಾಣ(kparams_local));
        kparams_local.n_threads         = (int32_t) layout.n_threads;
        kparams_local.total_rows        = (int32_t) total_rows;
        kparams_local.row_start         = (int32_t) row_start;
        kparams_local.row_end           = (int32_t) row_end;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.order             = is_ತುದಿ_k ? GGML_SORT_ORDER_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_ತುದಿ_k          = is_ತುದಿ_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_size         = (int32_t) layout.total_bytes;
        kparams_local.phase1_slot_size  = (int32_t) layout.phase1_slot_size;
        kparams_local.merge_ಮೌಲ್ಯಗಳು_off  = (int32_t) layout.merge_ಮೌಲ್ಯಗಳು_off;
        kparams_local.merge_indices_off = (int32_t) layout.merge_indices_off;
        kparams_local.merge_elems       = (int32_t) layout.merge_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } ಅನ್ಯಥಾ ಯದಿ (octx->ctx->mdev.ಎಣಿಕೆ > 1) {
        memcpy(&kparams_local, kparams, ಪರಿಮಾಣ(kparams_local));
        kparams_local.row_start = (int32_t) row_start;
        kparams_local.row_end   = (int32_t) row_end;
        kparams = &kparams_local;
    }

    ಯದಿ (octx->ctx->vtcm_size < (size_t) kparams->vtcm_size) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_size, octx->ctx->vtcm_size);
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    ಯದಿ (total_rows == 1 && kparams->n_chunks > 1) {
        ಸಂರಚನಾ htp_sort_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_base = vtcm_base;
        memset(cctx.real_ಎಣಿಕೆ, 0, ಪರಿಮಾಣ(cctx.real_ಎಣಿಕೆ));

        work_queue_run(octx->ctx->work_queue, htp_sort_chunk_job, &cctx, (uint32_t) kparams->n_chunks);

        ಯದಿ (is_ತುದಿ_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            ಯದಿ (k == 1 && ne00 >= 128*1024) {
                ಭಿನ್ನ *   cand_vals = (ಭಿನ್ನ *)   (vtcm_base + kparams->merge_ಮೌಲ್ಯಗಳು_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                ಭಿನ್ನ best_val = cand_vals[0];
                int32_t best_idx = cand_idxs[0];
                ಪ್ರತಿ (int32_t c = 1; c < kparams->n_chunks; c++) {
                    ಯದಿ (cand_vals[c] > best_val) {
                        best_val = cand_vals[c];
                        best_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->data;
                dst_ptr[0] = best_idx;
                ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
            }

            uint32_t total_candidates = (uint32_t) kparams->n_chunks * local_k;
            ಭಿನ್ನ *   cand_vals = (ಭಿನ್ನ *)   (vtcm_base + kparams->merge_ಮೌಲ್ಯಗಳು_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
            uint32_t merge_elems = (uint32_t) kparams->merge_elems;

            ಯದಿ (merge_elems > total_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + total_candidates), -INFINITY, merge_elems - total_candidates);
                ಪ್ರತಿ (uint32_t j = total_candidates; j < merge_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            ಸಂರಚನಾ htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_sort_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, merge_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->data, cand_idxs, k * ಪರಿಮಾಣ(int32_t));
        } ಅನ್ಯಥಾ {
            bool asc = kparams->order == GGML_SORT_ORDER_ASC;
            ಯದಿ (kparams->n_chunks == 8) {
                ಸಂರಚನಾ htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 4);

                ಸಂರಚನಾ htp_argsort_merge_l2_ctx l2ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l2_job, &l2ctx, 2);

                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ಧ್ರುವ ಭಿನ್ನ *   val0 = (ಧ್ರುವ ಭಿನ್ನ *) vtcm_base;
                ಧ್ರುವ int32_t * idx0 = (ಧ್ರುವ int32_t *) (vtcm_base + 4 * chunk_elems * ಪರಿಮಾಣ(ಭಿನ್ನ));
                ಧ್ರುವ ಭಿನ್ನ *   val1 = (ಧ್ರುವ ಭಿನ್ನ *) (vtcm_base + 4 * slot_size);
                ಧ್ರುವ int32_t * idx1 = (ಧ್ರುವ int32_t *) (vtcm_base + 4 * slot_size + 4 * chunk_elems * ಪರಿಮಾಣ(ಭಿನ್ನ));

                uint32_t n01 = cctx.real_ಎಣಿಕೆ[0] + cctx.real_ಎಣಿಕೆ[1] + cctx.real_ಎಣಿಕೆ[2] + cctx.real_ಎಣಿಕೆ[3];
                uint32_t n23 = cctx.real_ಎಣಿಕೆ[4] + cctx.real_ಎಣಿಕೆ[5] + cctx.real_ಎಣಿಕೆ[6] + cctx.real_ಎಣಿಕೆ[7];

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                ಸಂರಚನಾ htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, n01, val1, idx1, n23, ರಿಕ್ತ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * ಪರಿಮಾಣ(int32_t));
            } ಅನ್ಯಥಾ ಯದಿ (kparams->n_chunks == 4) {
                ಸಂರಚನಾ htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_ಎಣಿಕೆ[0] + cctx.real_ಎಣಿಕೆ[1];
                uint32_t n23 = cctx.real_ಎಣಿಕೆ[2] + cctx.real_ಎಣಿಕೆ[3];

                ಧ್ರುವ ಭಿನ್ನ *   val01 = (ಧ್ರುವ ಭಿನ್ನ *)   (vtcm_base + kparams->merge_ಮೌಲ್ಯಗಳು_off);
                ಧ್ರುವ int32_t * idx01 = (ಧ್ರುವ int32_t *) (vtcm_base + kparams->merge_indices_off);
                ಧ್ರುವ ಭಿನ್ನ *   val23 = (ಧ್ರುವ ಭಿನ್ನ *)   (vtcm_base + kparams->merge_ಮೌಲ್ಯಗಳು_off)  + 2 * chunk_elems;
                ಧ್ರುವ int32_t * idx23 = (ಧ್ರುವ int32_t *) (vtcm_base + kparams->merge_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_base;
                ಸಂರಚನಾ htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val01, idx01, n01, val23, idx23, n23, ರಿಕ್ತ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * ಪರಿಮಾಣ(int32_t));
            } ಅನ್ಯಥಾ ಯದಿ (kparams->n_chunks == 2) {
                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ಧ್ರುವ ಭಿನ್ನ *   val0 = (ಧ್ರುವ ಭಿನ್ನ *)   (vtcm_base);
                ಧ್ರುವ int32_t * idx0 = (ಧ್ರುವ int32_t *) (vtcm_base + chunk_elems * ಪರಿಮಾಣ(ಭಿನ್ನ));
                ಧ್ರುವ ಭಿನ್ನ *   val1 = (ಧ್ರುವ ಭಿನ್ನ *)   (vtcm_base + slot_size);
                ಧ್ರುವ int32_t * idx1 = (ಧ್ರುವ int32_t *) (vtcm_base + slot_size + chunk_elems * ಪರಿಮಾಣ(ಭಿನ್ನ));

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                ಸಂರಚನಾ htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, cctx.real_ಎಣಿಕೆ[0], val1, idx1, cctx.real_ಎಣಿಕೆ[1], ರಿಕ್ತ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * ಪರಿಮಾಣ(int32_t));
            } ಅನ್ಯಥಾ {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_base + chunk_elems * ಪರಿಮಾಣ(ಭಿನ್ನ));
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * ಪರಿಮಾಣ(int32_t));
            }
        }
        ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
    }

    ಸಂರಚನಾ htp_sort_multi_row_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_base        = vtcm_base;

    uint32_t dst_row_bytes = (uint32_t) dst_row_size;
    uint32_t rows_per_chunk = 0;
    htp_tensor_mdev_rows_per_chunk(dst, ಪರಿಮಾಣ(int32_t), dst_row_bytes, &rows_per_chunk);
    ಯದಿ (rows_per_chunk == 0) {
        rows_per_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_threads - 1) / (uint32_t) kparams->n_threads;
    ಯದಿ (rows_per_chunk > 1) {
        dr = hex_round_up(dr, rows_per_chunk);
    }
    mctx.nrows_per_thread = dr;

    work_queue_run(octx->ctx->work_queue, htp_sort_multi_row_job, &mctx, (uint32_t) kparams->n_threads);

    ಪ್ರತ್ಯಾಗಮ HTP_STATUS_OK;
}

ಪೂರ್ಣಾಙ್ಕ op_argsort(ಸಂರಚನಾ htp_ops_context * octx) {
    ಪ್ರತ್ಯಾಗಮ op_sort_common(octx, false);
}

ಪೂರ್ಣಾಙ್ಕ op_ತುದಿ_k(ಸಂರಚನಾ htp_ops_context * octx) {
    ಪ್ರತ್ಯಾಗಮ op_sort_common(octx, true);
}
