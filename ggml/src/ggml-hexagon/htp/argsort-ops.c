#ଅନ୍ତର୍ଭାଵ <string.h>
#ଅନ୍ତର୍ଭାଵ <stdlib.h>
#ଅନ୍ତର୍ଭାଵ <math.h>
#ଅନ୍ତର୍ଭାଵ <HAP_farf.h>
#ଅନ୍ତର୍ଭାଵ <HAP_perf.h>

#define GGML_COMMON_DECL_C
#ଅନ୍ତର୍ଭାଵ "ggml-common.h"
#ଅନ୍ତର୍ଭାଵ "ggml.h"

#ଅନ୍ତର୍ଭାଵ "hvx-utils.h"
#ଅନ୍ତର୍ଭାଵ "hvx-copy.h"
#ଅନ୍ତର୍ଭାଵ "hvx-reduce.h"
#ଅନ୍ତର୍ଭାଵ "dma-queue.h"

#ଅନ୍ତର୍ଭାଵ "hex-common.h"
#ଅନ୍ତର୍ଭାଵ "htp-ctx.h"
#ଅନ୍ତର୍ଭାଵ "htp-ops.h"
#ଅନ୍ତର୍ଭାଵ "htp-tensor.h"
#ଅନ୍ତର୍ଭାଵ "argsort-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif


ସ୍ଥିର ଶୂନ୍ୟ quicksort_ମୂଲ୍ୟଗୁଡ଼ିକ_indices_asc(ଦଶମିକ * ମୂଲ୍ୟଗୁଡ଼ିକ, int32_t * indices, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ବାମ, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଡାହାଣ) {
    ଯେତେବେଳେ (ବାମ < ଡାହାଣ) {
        ଦଶମିକ pivot = ମୂଲ୍ୟଗୁଡ଼ିକ[ବାମ + (ଡାହାଣ - ବାମ) / 2];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = ବାମ;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = ଡାହାଣ;

        ଯେତେବେଳେ (i <= j) {
            ଯେତେବେଳେ (ମୂଲ୍ୟଗୁଡ଼ିକ[i] < pivot) i++;
            ଯେତେବେଳେ (ମୂଲ୍ୟଗୁଡ଼ିକ[j] > pivot) j--;

            ଯଦି (i <= j) {
                ଦଶମିକ tmp_val = ମୂଲ୍ୟଗୁଡ଼ିକ[i];
                ମୂଲ୍ୟଗୁଡ଼ିକ[i] = ମୂଲ୍ୟଗୁଡ଼ିକ[j];
                ମୂଲ୍ୟଗୁଡ଼ିକ[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        ଯଦି (j - ବାମ < ଡାହାଣ - i) {
            ଯଦି (ବାମ < j) quicksort_ମୂଲ୍ୟଗୁଡ଼ିକ_indices_asc(ମୂଲ୍ୟଗୁଡ଼ିକ, indices, ବାମ, j);
            ବାମ = i;
        } ଅନ୍ୟଥା {
            ଯଦି (i < ଡାହାଣ) quicksort_ମୂଲ୍ୟଗୁଡ଼ିକ_indices_asc(ମୂଲ୍ୟଗୁଡ଼ିକ, indices, i, ଡାହାଣ);
            ଡାହାଣ = j;
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ quicksort_ମୂଲ୍ୟଗୁଡ଼ିକ_indices_desc(ଦଶମିକ * ମୂଲ୍ୟଗୁଡ଼ିକ, int32_t * indices, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ବାମ, ପୂର୍ଣ୍ଣସଂଖ୍ୟା ଡାହାଣ) {
    ଯେତେବେଳେ (ବାମ < ଡାହାଣ) {
        ଦଶମିକ pivot = ମୂଲ୍ୟଗୁଡ଼ିକ[ବାମ + (ଡାହାଣ - ବାମ) / 2];
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା i = ବାମ;
        ପୂର୍ଣ୍ଣସଂଖ୍ୟା j = ଡାହାଣ;

        ଯେତେବେଳେ (i <= j) {
            ଯେତେବେଳେ (ମୂଲ୍ୟଗୁଡ଼ିକ[i] > pivot) i++;
            ଯେତେବେଳେ (ମୂଲ୍ୟଗୁଡ଼ିକ[j] < pivot) j--;

            ଯଦି (i <= j) {
                ଦଶମିକ tmp_val = ମୂଲ୍ୟଗୁଡ଼ିକ[i];
                ମୂଲ୍ୟଗୁଡ଼ିକ[i] = ମୂଲ୍ୟଗୁଡ଼ିକ[j];
                ମୂଲ୍ୟଗୁଡ଼ିକ[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        ଯଦି (j - ବାମ < ଡାହାଣ - i) {
            ଯଦି (ବାମ < j) quicksort_ମୂଲ୍ୟଗୁଡ଼ିକ_indices_desc(ମୂଲ୍ୟଗୁଡ଼ିକ, indices, ବାମ, j);
            ବାମ = i;
        } ଅନ୍ୟଥା {
            ଯଦି (i < ଡାହାଣ) quicksort_ମୂଲ୍ୟଗୁଡ଼ିକ_indices_desc(ମୂଲ୍ୟଗୁଡ଼ିକ, indices, i, ଡାହାଣ);
            ଡାହାଣ = j;
        }
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ uint32_t ଶୀର୍ଷ_k_max_ମୂଲ୍ୟ_index(ଅଚଳ ଦଶମିକ * ମୂଲ୍ୟଗୁଡ଼ିକ, uint32_t n, ଦଶମିକ * ମୂଲ୍ୟ) {
    int32_t max_idx = 0;
    hvx_argmax_f32(ମୂଲ୍ୟଗୁଡ଼ିକ, n, 0, ମୂଲ୍ୟ, &max_idx);
    ଫେରନ୍ତୁ (uint32_t) max_idx;
}

ସ୍ଥିର ଅଚଳ int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ init_indices_ramp(int32_t * indices_buf, uint32_t ଗଣନା_2, uint32_t base) {
    ଅଚଳ HVX_Vector ind_init_vec = Q6_Vw_vadd_VwVw(*(ଅଚଳ HVX_Vector *) argsort_ramp_lut, Q6_V_vsplat_R(base));
    ଅଚଳ HVX_Vector ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_Vector * indices_buf_vec = (HVX_Vector *) indices_buf;
    uint32_t ଅଙ୍କ_vecs = ଗଣନା_2 / 32;
    HVX_Vector curr_ind_vec = ind_init_vec;
    ପାଇଁ (uint32_t j = 0; j < ଅଙ୍କ_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = ଗଣନା_2 % 32;
    ଯଦି (rem > 0) {
        hvx_vec_store_a((ଶୂନ୍ୟ *) &indices_buf_vec[ଅଙ୍କ_vecs], rem * ଆକାର(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ vec_cas(HVX_Vector * X_val, HVX_Vector * X_idx, HVX_Vector * Y_val, HVX_Vector * Y_idx, bool asc) {
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
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ bitonic_cas_32(HVX_Vector * V, HVX_Vector * I, ପୂର୍ଣ୍ଣସଂଖ୍ୟା d, HVX_VectorPred dir_mask, HVX_Vector idx_vec, HVX_Vector zero_vec) {
    HVX_VectorPred mask_ବାମ;
    HVX_Vector V_rot_ବାମ, V_rot_ଡାହାଣ;
    HVX_Vector I_rot_ବାମ, I_rot_ଡାହାଣ;

    ଯଦି (d == 1) {
        mask_ବାମ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_ବାମ = Q6_V_vror_VR(*V, 4);
        V_rot_ଡାହାଣ = Q6_V_vror_VR(*V, 124);
        I_rot_ବାମ = Q6_V_vror_VR(*I, 4);
        I_rot_ଡାହାଣ = Q6_V_vror_VR(*I, 124);
    } ଅନ୍ୟଥା ଯଦି (d == 2) {
        mask_ବାମ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_ବାମ = Q6_V_vror_VR(*V, 8);
        V_rot_ଡାହାଣ = Q6_V_vror_VR(*V, 120);
        I_rot_ବାମ = Q6_V_vror_VR(*I, 8);
        I_rot_ଡାହାଣ = Q6_V_vror_VR(*I, 120);
    } ଅନ୍ୟଥା ଯଦି (d == 4) {
        mask_ବାମ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_ବାମ = Q6_V_vror_VR(*V, 16);
        V_rot_ଡାହାଣ = Q6_V_vror_VR(*V, 112);
        I_rot_ବାମ = Q6_V_vror_VR(*I, 16);
        I_rot_ଡାହାଣ = Q6_V_vror_VR(*I, 112);
    } ଅନ୍ୟଥା ଯଦି (d == 8) {
        mask_ବାମ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_ବାମ = Q6_V_vror_VR(*V, 32);
        V_rot_ଡାହାଣ = Q6_V_vror_VR(*V, 96);
        I_rot_ବାମ = Q6_V_vror_VR(*I, 32);
        I_rot_ଡାହାଣ = Q6_V_vror_VR(*I, 96);
    } ଅନ୍ୟଥା {
        mask_ବାମ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_ବାମ = Q6_V_vror_VR(*V, 64);
        V_rot_ଡାହାଣ = Q6_V_vror_VR(*V, 64);
        I_rot_ବାମ = Q6_V_vror_VR(*I, 64);
        I_rot_ଡାହାଣ = Q6_V_vror_VR(*I, 64);
    }

    HVX_Vector V_paired = Q6_V_vmux_QVV(mask_ବାମ, V_rot_ବାମ, V_rot_ଡାହାଣ);
    HVX_Vector I_paired = Q6_V_vmux_QVV(mask_ବାମ, I_rot_ବାମ, I_rot_ଡାହାଣ);

    HVX_VectorPred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_VectorPred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_VectorPred mask_ଡାହାଣ = Q6_Q_not_Q(mask_ବାମ);
    HVX_VectorPred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(mask_ବାମ, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, mask_ଡାହାଣ)
    );
    HVX_VectorPred Q_swap = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_mask, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_mask), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_swap, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_swap, I_paired, *I);
}

__attribute__((always_inline))
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ bitonic_sort_generic_hvx(uint8_t * ମୂଲ୍ୟଗୁଡ଼ିକ, uint8_t * indices, ପୂର୍ଣ୍ଣସଂଖ୍ୟା K, bool asc_order) {
    HVX_Vector V[32];
    HVX_Vector I[32];

    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(ଅଚଳ HVX_Vector *) argsort_ramp_lut;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = 0; v < K; v++) {
        V[v] = *(HVX_Vector *)(ମୂଲ୍ୟଗୁଡ଼ିକ + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା M = 5;
    ଯେତେବେଳେ ((1 << (M - 5)) < K) M++;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 1; s <= M; s++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା stage_d = s - 1; stage_d >= 0; stage_d--) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା d = 1 << stage_d;
            ଯଦି (d >= 32) {
                ପୂର୍ଣ୍ଣସଂଖ୍ୟା v_dist = d / 32;
                ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା v1 = 0; v1 < K; v1++) {
                    ଯଦି ((v1 & v_dist) == 0) {
                        ପୂର୍ଣ୍ଣସଂଖ୍ୟା v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_order;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } ଅନ୍ୟଥା {
                ଯଦି (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                } ଅନ୍ୟଥା {
                    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_order;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା v = 0; v < K; v++) {
        *(HVX_Vector *)(ମୂଲ୍ୟଗୁଡ଼ିକ + v * 128)  = V[v];
        *(HVX_Vector *)(indices + v * 128) = I[v];
    }
}

ସ୍ଥିର ଶୂନ୍ୟ bitonic_sort_vtcm_desc(uint8_t * ମୂଲ୍ୟଗୁଡ଼ିକ, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(ଅଚଳ HVX_Vector *) argsort_ramp_lut;

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    ଯଦି (init_indices) {
        ପାଇଁ (uint32_t v = 0; v < n_vec; v++) {
            HVX_Vector idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_Vector *)(indices + v * 128) = idx;
        }
    }

    ପୂର୍ଣ୍ଣସଂଖ୍ୟା M = 5;
    ଯେତେବେଳେ ((1u << (M - 5)) < n_vec) M++;

    ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା s = 1; s <= M; s++) {
        ପାଇଁ (ପୂର୍ଣ୍ଣସଂଖ୍ୟା stage_d = s - 1; stage_d >= 0; stage_d--) {
            ପୂର୍ଣ୍ଣସଂଖ୍ୟା d = 1 << stage_d;
            ଯଦି (d >= 32) {
                uint32_t v_dist = d / 32;
                ପାଇଁ (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    ଯଦି ((v1 & v_dist) == 0) {
                        uint32_t v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : false;

                        HVX_Vector Vv1 = *(HVX_Vector *)(ମୂଲ୍ୟଗୁଡ଼ିକ + v1 * 128);
                        HVX_Vector Iv1 = *(HVX_Vector *)(indices + v1 * 128);
                        HVX_Vector Vv2 = *(HVX_Vector *)(ମୂଲ୍ୟଗୁଡ଼ିକ + v2 * 128);
                        HVX_Vector Iv2 = *(HVX_Vector *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_Vector *)(ମୂଲ୍ୟଗୁଡ଼ିକ + v1 * 128)  = Vv1;
                        *(HVX_Vector *)(indices + v1 * 128) = Iv1;
                        *(HVX_Vector *)(ମୂଲ୍ୟଗୁଡ଼ିକ + v2 * 128)  = Vv2;
                        *(HVX_Vector *)(indices + v2 * 128) = Iv2;
                    }
                }
            } ଅନ୍ୟଥା {
                ଯଦି (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    ପାଇଁ (uint32_t v = 0; v < n_vec; v++) {
                        HVX_Vector Vv = *(HVX_Vector *)(ମୂଲ୍ୟଗୁଡ଼ିକ + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(ମୂଲ୍ୟଗୁଡ଼ିକ + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                } ଅନ୍ୟଥା {
                    ପାଇଁ (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;

                        HVX_Vector Vv = *(HVX_Vector *)(ମୂଲ୍ୟଗୁଡ଼ିକ + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(ମୂଲ୍ୟଗୁଡ଼ିକ + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ ଶୀର୍ଷ_k_select_tiled(dma_queue * dma_q, ସଂରଚନା htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               ଦଶମିକ * ମୂଲ୍ୟଗୁଡ଼ିକ_buf, int32_t * indices_buf,
                               ଦଶମିକ * out_ମୂଲ୍ୟଗୁଡ଼ିକ, int32_t * out_indices,
                               uint32_t chunk_base) {
    ଅଚଳ uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_ଗଣନା    = n_tiles * k;
    uint32_t merge_n_vec        = hmx_ceil_div(candidate_ଗଣନା, 32);
    uint32_t merge_n_vec_pow2   = 1;
    ଯେତେବେଳେ (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
    uint32_t merge_elems        = merge_n_vec_pow2 * 32;
    ଦଶମିକ * candidate_ମୂଲ୍ୟଗୁଡ଼ିକ    = ମୂଲ୍ୟଗୁଡ଼ିକ_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    ପାଇଁ (uint32_t offset = 0; offset < n; offset += tile_elems) {
        uint32_t tile_ଗଣନା = MIN(tile_elems, n - offset);
        dma_sync_read(dma_q, ମୂଲ୍ୟଗୁଡ଼ିକ_buf, src_addr + (size_t) offset * ଆକାର(ଦଶମିକ), tile_ଗଣନା * ଆକାର(ଦଶମିକ));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
        ଯଦି (tile_ଗଣନା < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (ମୂଲ୍ୟଗୁଡ଼ିକ_buf + tile_ଗଣନା), -INFINITY, tile_elems - tile_ଗଣନା);
        }

        bitonic_sort_vtcm_desc((uint8_t *) ମୂଲ୍ୟଗୁଡ଼ିକ_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_ଗଣନା);
        ପାଇଁ (uint32_t j = 0; j < tile_k; j++) {
            candidate_ମୂଲ୍ୟଗୁଡ଼ିକ[candidate_pos] = ମୂଲ୍ୟଗୁଡ଼ିକ_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) offset;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    ଯଦି (merge_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_ମୂଲ୍ୟଗୁଡ଼ିକ + candidate_pos), -INFINITY, merge_elems - candidate_pos);
        ପାଇଁ (uint32_t j = candidate_pos; j < merge_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_sort_vtcm_desc((uint8_t *) candidate_ମୂଲ୍ୟଗୁଡ଼ିକ, (uint8_t *) candidate_indices, merge_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    ପାଇଁ (uint32_t j = 0; j < k; j++) {
        out_ମୂଲ୍ୟଗୁଡ଼ିକ[j]  = candidate_ମୂଲ୍ୟଗୁଡ଼ିକ[j];
        out_indices[j] = candidate_indices[j] + (int32_t) chunk_base;
    }
}

__attribute__((always_inline))
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ sort32_f32_hvx(uint8_t * ମୂଲ୍ୟଗୁଡ଼ିକ, uint8_t * indices, ଗଣନା ggml_sort_order order) {
    bitonic_sort_generic_hvx(ମୂଲ୍ୟଗୁଡ଼ିକ, indices, 1, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ sort64_f32_hvx(uint8_t * ମୂଲ୍ୟଗୁଡ଼ିକ, uint8_t * indices, ଗଣନା ggml_sort_order order) {
    bitonic_sort_generic_hvx(ମୂଲ୍ୟଗୁଡ଼ିକ, indices, 2, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ sort128_f32_hvx(uint8_t * ମୂଲ୍ୟଗୁଡ଼ିକ, uint8_t * indices, ଗଣନା ggml_sort_order order) {
    bitonic_sort_generic_hvx(ମୂଲ୍ୟଗୁଡ଼ିକ, indices, 4, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ sort256_f32_hvx(uint8_t * ମୂଲ୍ୟଗୁଡ଼ିକ, uint8_t * indices, ଗଣନା ggml_sort_order order) {
    bitonic_sort_generic_hvx(ମୂଲ୍ୟଗୁଡ଼ିକ, indices, 8, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ sort512_f32_hvx(uint8_t * ମୂଲ୍ୟଗୁଡ଼ିକ, uint8_t * indices, ଗଣନା ggml_sort_order order) {
    bitonic_sort_generic_hvx(ମୂଲ୍ୟଗୁଡ଼ିକ, indices, 16, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ sort1024_f32_hvx(uint8_t * ମୂଲ୍ୟଗୁଡ଼ିକ, uint8_t * indices, ଗଣନା ggml_sort_order order) {
    bitonic_sort_generic_hvx(ମୂଲ୍ୟଗୁଡ଼ିକ, indices, 32, order == GGML_SORT_ORDER_ASC);
}

ସ୍ଥିର ଶୂନ୍ୟ merge_runs(
    ଅଚଳ ଦଶମିକ * restrict in_val0, ଅଚଳ int32_t * restrict in_idx0, uint32_t n0,
    ଅଚଳ ଦଶମିକ * restrict in_val1, ଅଚଳ int32_t * restrict in_idx1, uint32_t n1,
    ଦଶମିକ * restrict out_val, int32_t * restrict out_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t out = 0;

    ଯଦି (asc) {
        ଯେତେବେଳେ (i0 < n0 && i1 < n1) {
            ଯଦି (in_val0[i0] <= in_val1[i1]) {
                ଯଦି (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } ଅନ୍ୟଥା {
                ଯଦି (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    } ଅନ୍ୟଥା {
        ଯେତେବେଳେ (i0 < n0 && i1 < n1) {
            ଯଦି (in_val0[i0] >= in_val1[i1]) {
                ଯଦି (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } ଅନ୍ୟଥା {
                ଯଦି (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    }

    ଯେତେବେଳେ (i0 < n0) {
        ଯଦି (out_val) out_val[out] = in_val0[i0];
        out_idx[out] = in_idx0[i0];
        i0++;
        out++;
    }

    ଯେତେବେଳେ (i1 < n1) {
        ଯଦି (out_val) out_val[out] = in_val1[i1];
        out_idx[out] = in_idx1[i1];
        i1++;
        out++;
    }
}

ସଂରଚନା htp_sort_chunk_ctx {
    ସଂରଚନା htp_ops_context *              octx;
    ଅଚଳ ସଂରଚନା htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              real_ଗଣନା[HTP_MAX_NTHREADS];
};

ସ୍ଥିର ଶୂନ୍ୟ htp_sort_chunk_job(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଶୂନ୍ୟ * data) {
    ସଂରଚନା htp_sort_chunk_ctx * cctx = (ସଂରଚନା htp_sort_chunk_ctx *) data;
    ସଂରଚନା htp_ops_context * octx = cctx->octx;
    ଅଚଳ ସଂରଚନା htp_sort_kernel_params * kparams = cctx->kparams;
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_base  = i * chunk_elems;

    uint32_t real_ଗଣନା = (chunk_base < ne00) ? MIN(chunk_elems, ne00 - chunk_base) : 0;
    cctx->real_ଗଣନା[i] = real_ଗଣନା;

    uint8_t * spad = cctx->vtcm_base + (size_t) kparams->phase1_slot_size * i;
    ଦଶମିକ *   ମୂଲ୍ୟଗୁଡ଼ିକ_buf  = (ଦଶମିକ *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * ଆକାର(ଦଶମିକ));

    ଦଶମିକ *   merge_ମୂଲ୍ୟଗୁଡ଼ିକ  = (ଦଶମିକ *)   (cctx->vtcm_base + kparams->merge_ମୂଲ୍ୟଗୁଡ଼ିକ_off);
    int32_t * merge_indices = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off);

    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[i];
    ଅଚଳ dma_addr_t src_addr = src0->data + (size_t) chunk_base * ଆକାର(ଦଶମିକ);

    ଯଦି (kparams->is_ଶୀର୍ଷ_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        ଯଦି (real_ଗଣନା == 0) {
            ପାଇଁ (uint32_t j = 0; j < local_k; j++) {
                merge_ମୂଲ୍ୟଗୁଡ଼ିକ[i * local_k + j]  = -INFINITY;
                merge_indices[i * local_k + j] = 0;
            }
            ଫେରନ୍ତୁ;
        }

        ଯଦି (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], ମୂଲ୍ୟଗୁଡ଼ିକ_buf, src_addr, real_ଗଣନା * ଆକାର(ଦଶମିକ));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            ଦଶମିକ max_ମୂଲ୍ୟ;
            uint32_t max_index = ଶୀର୍ଷ_k_max_ମୂଲ୍ୟ_index(ମୂଲ୍ୟଗୁଡ଼ିକ_buf, real_ଗଣନା, &max_ମୂଲ୍ୟ);
            merge_ମୂଲ୍ୟଗୁଡ଼ିକ[i] = max_ମୂଲ୍ୟ;
            merge_indices[i] = (int32_t) (max_index + chunk_base);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            ଫେରନ୍ତୁ;
        }

        ଯଦି (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            ଶୀର୍ଷ_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_ଗଣନା, local_k,
                               ମୂଲ୍ୟଗୁଡ଼ିକ_buf, indices_buf,
                               &merge_ମୂଲ୍ୟଗୁଡ଼ିକ[i * local_k], &merge_indices[i * local_k],
                               chunk_base);
            ଫେରନ୍ତୁ;
        }

        dma_sync_read(octx->ctx->dma[i], ମୂଲ୍ୟଗୁଡ଼ିକ_buf, src_addr, real_ଗଣନା * ଆକାର(ଦଶମିକ));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        ଯଦି (chunk_elems > real_ଗଣନା) {
            hvx_splat_f32_u((uint8_t *)(ମୂଲ୍ୟଗୁଡ଼ିକ_buf + real_ଗଣନା), -INFINITY, chunk_elems - real_ଗଣନା);
        }

        bitonic_sort_vtcm_desc((uint8_t *) ମୂଲ୍ୟଗୁଡ଼ିକ_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        ପାଇଁ (uint32_t j = 0; j < local_k; j++) {
            merge_ମୂଲ୍ୟଗୁଡ଼ିକ[i * local_k + j]  = ମୂଲ୍ୟଗୁଡ଼ିକ_buf[j];
            merge_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_base;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } ଅନ୍ୟଥା {
        ଯଦି (real_ଗଣନା > 0) {
            dma_sync_read(octx->ctx->dma[i], ମୂଲ୍ୟଗୁଡ଼ିକ_buf, src_addr, real_ଗଣନା * ଆକାର(ଦଶମିକ));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_ଗଣନା, chunk_base);

            ଯଦି (kparams->order == GGML_SORT_ORDER_ASC) {
                quicksort_ମୂଲ୍ୟଗୁଡ଼ିକ_indices_asc(ମୂଲ୍ୟଗୁଡ଼ିକ_buf, indices_buf, 0, real_ଗଣନା - 1);
            } ଅନ୍ୟଥା {
                quicksort_ମୂଲ୍ୟଗୁଡ଼ିକ_indices_desc(ମୂଲ୍ୟଗୁଡ଼ିକ_buf, indices_buf, 0, real_ଗଣନା - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

ସଂରଚନା htp_argsort_merge_l1_ctx {
    ସଂରଚନା htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

ସ୍ଥିର ଶୂନ୍ୟ htp_argsort_merge_l1_job(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଶୂନ୍ୟ * data) {
    ସଂରଚନା htp_argsort_merge_l1_ctx * mctx = (ସଂରଚନା htp_argsort_merge_l1_ctx *) data;
    ସଂରଚନା htp_sort_chunk_ctx * cctx = mctx->cctx;
    ଅଚଳ ସଂରଚନା htp_sort_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_ଗଣନା[c0];
    uint32_t count1 = cctx->real_ଗଣନା[c1];

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    ଅଚଳ ଦଶମିକ *   val0 = (ଅଚଳ ଦଶମିକ *)   (cctx->vtcm_base + slot_size * c0);
    ଅଚଳ int32_t * idx0 = (ଅଚଳ int32_t *) (cctx->vtcm_base + slot_size * c0 + chunk_elems * ଆକାର(ଦଶମିକ));
    ଅଚଳ ଦଶମିକ *   val1 = (ଅଚଳ ଦଶମିକ *)   (cctx->vtcm_base + slot_size * c1);
    ଅଚଳ int32_t * idx1 = (ଅଚଳ int32_t *) (cctx->vtcm_base + slot_size * c1 + chunk_elems * ଆକାର(ଦଶମିକ));

    uint32_t out_offset = (2 * i) * chunk_elems;
    ଦଶମିକ *   out_val = (ଦଶମିକ *)   (cctx->vtcm_base + kparams->merge_ମୂଲ୍ୟଗୁଡ଼ିକ_off)  + out_offset;
    int32_t * out_idx = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + out_offset;

    ସଂରଚନା htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

ସଂରଚନା htp_argsort_merge_l2_ctx {
    ସଂରଚନା htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

ସ୍ଥିର ଶୂନ୍ୟ htp_argsort_merge_l2_job(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା i, ଶୂନ୍ୟ * data) {
    ସଂରଚନା htp_argsort_merge_l2_ctx * mctx = (ସଂରଚନା htp_argsort_merge_l2_ctx *) data;
    ସଂରଚନା htp_sort_chunk_ctx * cctx = mctx->cctx;
    ଅଚଳ ସଂରଚନା htp_sort_kernel_params * kparams = cctx->kparams;

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    ଅଚଳ ଦଶମିକ *   val0 = (ଅଚଳ ଦଶମିକ *)   (cctx->vtcm_base + kparams->merge_ମୂଲ୍ୟଗୁଡ଼ିକ_off)  + off0;
    ଅଚଳ int32_t * idx0 = (ଅଚଳ int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off0;
    ଅଚଳ ଦଶମିକ *   val1 = (ଅଚଳ ଦଶମିକ *)   (cctx->vtcm_base + kparams->merge_ମୂଲ୍ୟଗୁଡ଼ିକ_off)  + off1;
    ଅଚଳ int32_t * idx1 = (ଅଚଳ int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off1;

    uint32_t count0 = cctx->real_ଗଣନା[4*i + 0] + cctx->real_ଗଣନା[4*i + 1];
    uint32_t count1 = cctx->real_ଗଣନା[4*i + 2] + cctx->real_ଗଣନା[4*i + 3];

    uint8_t * slot_base = cctx->vtcm_base + (4 * i) * slot_size;
    ଦଶମିକ *   out_val   = (ଦଶମିକ *) slot_base;
    int32_t * out_idx   = (int32_t *) (slot_base + 4 * chunk_elems * ଆକାର(ଦଶମିକ));

    ସଂରଚନା htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

ସଂରଚନା htp_sort_multi_row_ctx {
    ସଂରଚନା htp_ops_context *              octx;
    ଅଚଳ ସଂରଚନା htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              nrows_per_thread;
};

ସଂରଚନା htp_tensor_row_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ସଂରଚନା htp_tensor_row_coord init_row_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    ଫେରନ୍ତୁ (ସଂରଚନା htp_tensor_row_coord){ i01, i02, i03 };
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ step_row_coord(ସଂରଚନା htp_tensor_row_coord * c, uint32_t ne01, uint32_t ne02) {
    c->i01++;
    ଯଦି (c->i01 == ne01) {
        c->i01 = 0;
        c->i02++;
        ଯଦି (c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ size_t coord_offset(ସଂରଚନା htp_tensor_row_coord c, size_t nb1, size_t nb2, size_t nb3) {
    ଫେରନ୍ତୁ (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

ସ୍ଥିର ଧାଡ଼ିଭିତରେ ଶୂନ୍ୟ compute_row_sort(
    ଦଶମିକ * cur_ମୂଲ୍ୟଗୁଡ଼ିକ,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_ଶୀର୍ଷ_k,
    ଗଣନା ggml_sort_order order
) {
    ଯଦି (is_ଶୀର୍ଷ_k) {
        ଯଦି (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            ବାଛ (ne00) {
                ସ୍ଥିତି 32:   sort32_f32_hvx((uint8_t *) cur_ମୂଲ୍ୟଗୁଡ଼ିକ, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ଭାଙ୍ଗ;
                ସ୍ଥିତି 64:   sort64_f32_hvx((uint8_t *) cur_ମୂଲ୍ୟଗୁଡ଼ିକ, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ଭାଙ୍ଗ;
                ସ୍ଥିତି 128:  sort128_f32_hvx((uint8_t *) cur_ମୂଲ୍ୟଗୁଡ଼ିକ, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ଭାଙ୍ଗ;
                ସ୍ଥିତି 256:  sort256_f32_hvx((uint8_t *) cur_ମୂଲ୍ୟଗୁଡ଼ିକ, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ଭାଙ୍ଗ;
                ସ୍ଥିତି 512:  sort512_f32_hvx((uint8_t *) cur_ମୂଲ୍ୟଗୁଡ଼ିକ, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ଭାଙ୍ଗ;
                ସ୍ଥିତି 1024: sort1024_f32_hvx((uint8_t *) cur_ମୂଲ୍ୟଗୁଡ଼ିକ, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ଭାଙ୍ଗ;
            }
        } ଅନ୍ୟଥା {
            ଯଦି (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_ମୂଲ୍ୟଗୁଡ଼ିକ + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_sort_vtcm_desc((uint8_t *) cur_ମୂଲ୍ୟଗୁଡ଼ିକ, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } ଅନ୍ୟଥା {
        ଯଦି (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            ବାଛ (ne00) {
                ସ୍ଥିତି 32:   sort32_f32_hvx((uint8_t *) cur_ମୂଲ୍ୟଗୁଡ଼ିକ, (uint8_t *) cur_indices, order); ଭାଙ୍ଗ;
                ସ୍ଥିତି 64:   sort64_f32_hvx((uint8_t *) cur_ମୂଲ୍ୟଗୁଡ଼ିକ, (uint8_t *) cur_indices, order); ଭାଙ୍ଗ;
                ସ୍ଥିତି 128:  sort128_f32_hvx((uint8_t *) cur_ମୂଲ୍ୟଗୁଡ଼ିକ, (uint8_t *) cur_indices, order); ଭାଙ୍ଗ;
                ସ୍ଥିତି 256:  sort256_f32_hvx((uint8_t *) cur_ମୂଲ୍ୟଗୁଡ଼ିକ, (uint8_t *) cur_indices, order); ଭାଙ୍ଗ;
                ସ୍ଥିତି 512:  sort512_f32_hvx((uint8_t *) cur_ମୂଲ୍ୟଗୁଡ଼ିକ, (uint8_t *) cur_indices, order); ଭାଙ୍ଗ;
                ସ୍ଥିତି 1024: sort1024_f32_hvx((uint8_t *) cur_ମୂଲ୍ୟଗୁଡ଼ିକ, (uint8_t *) cur_indices, order); ଭାଙ୍ଗ;
            }
        } ଅନ୍ୟଥା {
            init_indices_ramp(cur_indices, ne00, 0);
            ଯଦି (order == GGML_SORT_ORDER_ASC) {
                quicksort_ମୂଲ୍ୟଗୁଡ଼ିକ_indices_asc(cur_ମୂଲ୍ୟଗୁଡ଼ିକ, cur_indices, 0, ne00 - 1);
            } ଅନ୍ୟଥା {
                quicksort_ମୂଲ୍ୟଗୁଡ଼ିକ_indices_desc(cur_ମୂଲ୍ୟଗୁଡ଼ିକ, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

ସ୍ଥିର ଶୂନ୍ୟ htp_sort_multi_row_job(unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା n, unsigned ପୂର୍ଣ୍ଣସଂଖ୍ୟା ith, ଶୂନ୍ୟ * data) {
    ସଂରଚନା htp_sort_multi_row_ctx * mctx = (ସଂରଚନା htp_sort_multi_row_ctx *) data;
    ସଂରଚନା htp_ops_context * octx = mctx->octx;
    ଅଚଳ ସଂରଚନା htp_sort_kernel_params * kparams = mctx->kparams;
    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

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

    ଯଦି (r_start >= r_end) ଫେରନ୍ତୁ;

    size_t spad_offset = ith * (size_t) (kparams->phase1_slot_size * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_base + spad_offset;
    size_t slot_size = (size_t) kparams->phase1_slot_size;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    ଦଶମିକ *   ମୂଲ୍ୟଗୁଡ଼ିକ_buf  = (ଦଶମିକ *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * ଆକାର(ଦଶମିକ));

    dma_queue * dma_q = octx->ctx->dma[ith];
    ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_ଶୀର୍ଷ_k = kparams->is_ଶୀର୍ଷ_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    ଗଣନା ggml_sort_order order = (ଗଣନା ggml_sort_order) kparams->order;

    ଅଚଳ uint32_t src_bytes = ne00 * ଆକାର(ଦଶମିକ);
    ଅଚଳ uint32_t dst_bytes = (is_ଶୀର୍ଷ_k ? k : ne00) * ଆକାର(int32_t);

    ଯଦି (kparams->n_slots == 2) {
        ସଂରଚନା htp_tensor_row_coord ମୌଳିକ_coord = init_row_coord(r_start, ne01, ne02);
        ପାଇଁ (uint32_t r = r_start, spad_idx = 0; r < r_end && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * slot_size;
            ଦଶମିକ *   cur_ମୂଲ୍ୟଗୁଡ଼ିକ  = (ଦଶମିକ *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * ଆକାର(ଦଶମିକ));

            dma_addr_t dst_addr = dst->data + coord_offset(ମୌଳିକ_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->data + coord_offset(ମୌଳିକ_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_queue_ଠେଲ(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 0);

            // Prefetch input row
            dma_queue_ଠେଲ(dma_q, dma_make_data(cur_ମୂଲ୍ୟଗୁଡ଼ିକ, src_addr),
                           src_bytes, src_bytes, src_bytes, 1);

            step_row_coord(&ମୌଳିକ_coord, ne01, ne02);
        }

        ସଂରଚନା htp_tensor_row_coord dst_coord = init_row_coord(r_start, ne01, ne02);
        ସଂରଚନା htp_tensor_row_coord next_src_coord = init_row_coord(r_start + 2, ne01, ne02);

        ପାଇଁ (uint32_t r = r_start; r < r_end; r++) {
            uint32_t cur_slot = (r - r_start) & 1;
            uint8_t * cur_spad = thread_spad + cur_slot * slot_size;
            ଦଶମିକ *   cur_ମୂଲ୍ୟଗୁଡ଼ିକ  = (ଦଶମିକ *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * ଆକାର(ଦଶମିକ));

            // Wait for previous writeback of this slot to complete
            dma_queue_କାଢ଼(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_queue_କାଢ଼(dma_q);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(cur_ମୂଲ୍ୟଗୁଡ଼ିକ, cur_indices, ne00, chunk_elems, is_ଶୀର୍ଷ_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->data + coord_offset(dst_coord, nb1, nb2, nb3);
            dma_queue_ଠେଲ(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 1);
            step_row_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            ଅଚଳ uint32_t next_row = r + 2;
            ଯଦି (next_row < r_end) {
                dma_addr_t next_src_addr = src0->data + coord_offset(next_src_coord, nb01, nb02, nb03);
                dma_queue_ଠେଲ(dma_q, dma_make_data(cur_ମୂଲ୍ୟଗୁଡ଼ିକ, next_src_addr),
                               src_bytes, src_bytes, src_bytes, 1);
                step_row_coord(&next_src_coord, ne01, ne02);
            }
        }

        dma_queue_flush(dma_q);
    } ଅନ୍ୟଥା {
        ସଂରଚନା htp_tensor_row_coord coord = init_row_coord(r_start, ne01, ne02);
        ପାଇଁ (uint32_t r = r_start; r < r_end; r++) {
            dma_addr_t src_addr = src0->data + coord_offset(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->data + coord_offset(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, ମୂଲ୍ୟଗୁଡ଼ିକ_buf, src_addr, src_bytes);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(ମୂଲ୍ୟଗୁଡ଼ିକ_buf, indices_buf, ne00, chunk_elems, is_ଶୀର୍ଷ_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_bytes);
            step_row_coord(&coord, ne01, ne02);
        }
    }
}

ସ୍ଥିର ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_sort_common(ସଂରଚନା htp_ops_context * octx, bool is_ଶୀର୍ଷ_k) {
    ଯଦି (octx->src[0]->type != HTP_TYPE_F32) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଅଚଳ ସଂରଚନା htp_tensor * src0 = octx->src[0];
    ଅଚଳ ସଂରଚନା htp_tensor * dst  = octx->dst;

    ଯଦି (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ଫେରନ୍ତୁ HTP_STATUS_NO_SUPPORT;
    }

    ଅଚଳ uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ଅଚଳ size_t dst_row_size = dst->ne[0] * ଆକାର(int32_t);

    uint32_t row_start = 0;
    uint32_t row_end   = total_rows;
    ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, ଆକାର(int32_t), (uint32_t) dst_row_size, &rows_per_chunk);
        ଅଚଳ ସଂରଚନା htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ଗଣନା_2, &octx->ctx->mdev.ଗଣନା_div);
        row_start = range.start;
        row_end   = range.start + range.ଗଣନା_2;
    }

    ଅଚଳ uint32_t nrows = row_end - row_start;
    ଯଦି (nrows == 0) {
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    ସଂରଚନା htp_sort_kernel_params kparams_local;
    ଅଚଳ ସଂରଚନା htp_sort_kernel_params * kparams = (ଅଚଳ ସଂରଚନା htp_sort_kernel_params *) octx->kernel_params;

    ଯଦି (kparams->n_threads == 0) {
        ସଂରଚନା htp_sort_vtcm_layout layout;
        ଯଦି (!htp_sort_solve_layout(&layout, ne00, total_rows, k, octx->n_threads, octx->ctx->vtcm_size, is_ଶୀର୍ଷ_k)) {
            ଫେରନ୍ତୁ HTP_STATUS_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, ଆକାର(kparams_local));
        kparams_local.n_threads         = (int32_t) layout.n_threads;
        kparams_local.total_rows        = (int32_t) total_rows;
        kparams_local.row_start         = (int32_t) row_start;
        kparams_local.row_end           = (int32_t) row_end;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.order             = is_ଶୀର୍ଷ_k ? GGML_SORT_ORDER_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_ଶୀର୍ଷ_k          = is_ଶୀର୍ଷ_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_size         = (int32_t) layout.total_bytes;
        kparams_local.phase1_slot_size  = (int32_t) layout.phase1_slot_size;
        kparams_local.merge_ମୂଲ୍ୟଗୁଡ଼ିକ_off  = (int32_t) layout.merge_ମୂଲ୍ୟଗୁଡ଼ିକ_off;
        kparams_local.merge_indices_off = (int32_t) layout.merge_indices_off;
        kparams_local.merge_elems       = (int32_t) layout.merge_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } ଅନ୍ୟଥା ଯଦି (octx->ctx->mdev.ଗଣନା_2 > 1) {
        memcpy(&kparams_local, kparams, ଆକାର(kparams_local));
        kparams_local.row_start = (int32_t) row_start;
        kparams_local.row_end   = (int32_t) row_end;
        kparams = &kparams_local;
    }

    ଯଦି (octx->ctx->vtcm_size < (size_t) kparams->vtcm_size) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_size, octx->ctx->vtcm_size);
        ଫେରନ୍ତୁ HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    ଯଦି (total_rows == 1 && kparams->n_chunks > 1) {
        ସଂରଚନା htp_sort_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_base = vtcm_base;
        memset(cctx.real_ଗଣନା, 0, ଆକାର(cctx.real_ଗଣନା));

        work_queue_run(octx->ctx->work_queue, htp_sort_chunk_job, &cctx, (uint32_t) kparams->n_chunks);

        ଯଦି (is_ଶୀର୍ଷ_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            ଯଦି (k == 1 && ne00 >= 128*1024) {
                ଦଶମିକ *   cand_vals = (ଦଶମିକ *)   (vtcm_base + kparams->merge_ମୂଲ୍ୟଗୁଡ଼ିକ_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                ଦଶମିକ best_val = cand_vals[0];
                int32_t best_idx = cand_idxs[0];
                ପାଇଁ (int32_t c = 1; c < kparams->n_chunks; c++) {
                    ଯଦି (cand_vals[c] > best_val) {
                        best_val = cand_vals[c];
                        best_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->data;
                dst_ptr[0] = best_idx;
                ଫେରନ୍ତୁ HTP_STATUS_OK;
            }

            uint32_t total_candidates = (uint32_t) kparams->n_chunks * local_k;
            ଦଶମିକ *   cand_vals = (ଦଶମିକ *)   (vtcm_base + kparams->merge_ମୂଲ୍ୟଗୁଡ଼ିକ_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
            uint32_t merge_elems = (uint32_t) kparams->merge_elems;

            ଯଦି (merge_elems > total_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + total_candidates), -INFINITY, merge_elems - total_candidates);
                ପାଇଁ (uint32_t j = total_candidates; j < merge_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_sort_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, merge_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->data, cand_idxs, k * ଆକାର(int32_t));
        } ଅନ୍ୟଥା {
            bool asc = kparams->order == GGML_SORT_ORDER_ASC;
            ଯଦି (kparams->n_chunks == 8) {
                ସଂରଚନା htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 4);

                ସଂରଚନା htp_argsort_merge_l2_ctx l2ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l2_job, &l2ctx, 2);

                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ଅଚଳ ଦଶମିକ *   val0 = (ଅଚଳ ଦଶମିକ *) vtcm_base;
                ଅଚଳ int32_t * idx0 = (ଅଚଳ int32_t *) (vtcm_base + 4 * chunk_elems * ଆକାର(ଦଶମିକ));
                ଅଚଳ ଦଶମିକ *   val1 = (ଅଚଳ ଦଶମିକ *) (vtcm_base + 4 * slot_size);
                ଅଚଳ int32_t * idx1 = (ଅଚଳ int32_t *) (vtcm_base + 4 * slot_size + 4 * chunk_elems * ଆକାର(ଦଶମିକ));

                uint32_t n01 = cctx.real_ଗଣନା[0] + cctx.real_ଗଣନା[1] + cctx.real_ଗଣନା[2] + cctx.real_ଗଣନା[3];
                uint32_t n23 = cctx.real_ଗଣନା[4] + cctx.real_ଗଣନା[5] + cctx.real_ଗଣନା[6] + cctx.real_ଗଣନା[7];

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, n01, val1, idx1, n23, ଖାଲି, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * ଆକାର(int32_t));
            } ଅନ୍ୟଥା ଯଦି (kparams->n_chunks == 4) {
                ସଂରଚନା htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_ଗଣନା[0] + cctx.real_ଗଣନା[1];
                uint32_t n23 = cctx.real_ଗଣନା[2] + cctx.real_ଗଣନା[3];

                ଅଚଳ ଦଶମିକ *   val01 = (ଅଚଳ ଦଶମିକ *)   (vtcm_base + kparams->merge_ମୂଲ୍ୟଗୁଡ଼ିକ_off);
                ଅଚଳ int32_t * idx01 = (ଅଚଳ int32_t *) (vtcm_base + kparams->merge_indices_off);
                ଅଚଳ ଦଶମିକ *   val23 = (ଅଚଳ ଦଶମିକ *)   (vtcm_base + kparams->merge_ମୂଲ୍ୟଗୁଡ଼ିକ_off)  + 2 * chunk_elems;
                ଅଚଳ int32_t * idx23 = (ଅଚଳ int32_t *) (vtcm_base + kparams->merge_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_base;
                ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val01, idx01, n01, val23, idx23, n23, ଖାଲି, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * ଆକାର(int32_t));
            } ଅନ୍ୟଥା ଯଦି (kparams->n_chunks == 2) {
                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ଅଚଳ ଦଶମିକ *   val0 = (ଅଚଳ ଦଶମିକ *)   (vtcm_base);
                ଅଚଳ int32_t * idx0 = (ଅଚଳ int32_t *) (vtcm_base + chunk_elems * ଆକାର(ଦଶମିକ));
                ଅଚଳ ଦଶମିକ *   val1 = (ଅଚଳ ଦଶମିକ *)   (vtcm_base + slot_size);
                ଅଚଳ int32_t * idx1 = (ଅଚଳ int32_t *) (vtcm_base + slot_size + chunk_elems * ଆକାର(ଦଶମିକ));

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                ସଂରଚନା htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, cctx.real_ଗଣନା[0], val1, idx1, cctx.real_ଗଣନା[1], ଖାଲି, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * ଆକାର(int32_t));
            } ଅନ୍ୟଥା {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_base + chunk_elems * ଆକାର(ଦଶମିକ));
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * ଆକାର(int32_t));
            }
        }
        ଫେରନ୍ତୁ HTP_STATUS_OK;
    }

    ସଂରଚନା htp_sort_multi_row_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_base        = vtcm_base;

    uint32_t dst_row_bytes = (uint32_t) dst_row_size;
    uint32_t rows_per_chunk = 0;
    htp_tensor_mdev_rows_per_chunk(dst, ଆକାର(int32_t), dst_row_bytes, &rows_per_chunk);
    ଯଦି (rows_per_chunk == 0) {
        rows_per_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_threads - 1) / (uint32_t) kparams->n_threads;
    ଯଦି (rows_per_chunk > 1) {
        dr = hex_round_up(dr, rows_per_chunk);
    }
    mctx.nrows_per_thread = dr;

    work_queue_run(octx->ctx->work_queue, htp_sort_multi_row_job, &mctx, (uint32_t) kparams->n_threads);

    ଫେରନ୍ତୁ HTP_STATUS_OK;
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_argsort(ସଂରଚନା htp_ops_context * octx) {
    ଫେରନ୍ତୁ op_sort_common(octx, false);
}

ପୂର୍ଣ୍ଣସଂଖ୍ୟା op_ଶୀର୍ଷ_k(ସଂରଚନା htp_ops_context * octx) {
    ଫେରନ୍ତୁ op_sort_common(octx, true);
}
