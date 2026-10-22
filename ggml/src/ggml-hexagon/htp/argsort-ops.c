#ਅਨ੍ਤਰ੍ਭਾਵ <string.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdlib.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <math.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_farf.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <HAP_perf.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ GGML_COMMON_DECL_C
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "ggml.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-copy.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-reduce.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "dma-queue.h"

#ਅਨ੍ਤਰ੍ਭਾਵ "hex-common.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ctx.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-ops.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "htp-tensor.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "argsort-ops.h"

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ MIN
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ MIN(a, b) ((a) < (b) ? (a) : (b))
#ਜੇ_ਅੰਤ


ਸਥਿਰ ਖਾਲੀ quicksort_ਮੁੱਲਾਂ_indices_asc(ਦਸ਼ਮਲਵ * ਮੁੱਲਾਂ, int32_t * indices, ਪੂਰਨ ਖੱਬਾ, ਪੂਰਨ ਸੱਜਾ) {
    ਜਦੋਂ (ਖੱਬਾ < ਸੱਜਾ) {
        ਦਸ਼ਮਲਵ ਧੁਰਾ = ਮੁੱਲਾਂ[ਖੱਬਾ + (ਸੱਜਾ - ਖੱਬਾ) / 2];
        ਪੂਰਨ i = ਖੱਬਾ;
        ਪੂਰਨ j = ਸੱਜਾ;

        ਜਦੋਂ (i <= j) {
            ਜਦੋਂ (ਮੁੱਲਾਂ[i] < ਧੁਰਾ) i++;
            ਜਦੋਂ (ਮੁੱਲਾਂ[j] > ਧੁਰਾ) j--;

            ਜੇ (i <= j) {
                ਦਸ਼ਮਲਵ tmp_val = ਮੁੱਲਾਂ[i];
                ਮੁੱਲਾਂ[i] = ਮੁੱਲਾਂ[j];
                ਮੁੱਲਾਂ[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        ਜੇ (j - ਖੱਬਾ < ਸੱਜਾ - i) {
            ਜੇ (ਖੱਬਾ < j) quicksort_ਮੁੱਲਾਂ_indices_asc(ਮੁੱਲਾਂ, indices, ਖੱਬਾ, j);
            ਖੱਬਾ = i;
        } ਨਹੀਂਤਾਂ {
            ਜੇ (i < ਸੱਜਾ) quicksort_ਮੁੱਲਾਂ_indices_asc(ਮੁੱਲਾਂ, indices, i, ਸੱਜਾ);
            ਸੱਜਾ = j;
        }
    }
}

ਸਥਿਰ ਖਾਲੀ quicksort_ਮੁੱਲਾਂ_indices_desc(ਦਸ਼ਮਲਵ * ਮੁੱਲਾਂ, int32_t * indices, ਪੂਰਨ ਖੱਬਾ, ਪੂਰਨ ਸੱਜਾ) {
    ਜਦੋਂ (ਖੱਬਾ < ਸੱਜਾ) {
        ਦਸ਼ਮਲਵ ਧੁਰਾ = ਮੁੱਲਾਂ[ਖੱਬਾ + (ਸੱਜਾ - ਖੱਬਾ) / 2];
        ਪੂਰਨ i = ਖੱਬਾ;
        ਪੂਰਨ j = ਸੱਜਾ;

        ਜਦੋਂ (i <= j) {
            ਜਦੋਂ (ਮੁੱਲਾਂ[i] > ਧੁਰਾ) i++;
            ਜਦੋਂ (ਮੁੱਲਾਂ[j] < ਧੁਰਾ) j--;

            ਜੇ (i <= j) {
                ਦਸ਼ਮਲਵ tmp_val = ਮੁੱਲਾਂ[i];
                ਮੁੱਲਾਂ[i] = ਮੁੱਲਾਂ[j];
                ਮੁੱਲਾਂ[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        ਜੇ (j - ਖੱਬਾ < ਸੱਜਾ - i) {
            ਜੇ (ਖੱਬਾ < j) quicksort_ਮੁੱਲਾਂ_indices_desc(ਮੁੱਲਾਂ, indices, ਖੱਬਾ, j);
            ਖੱਬਾ = i;
        } ਨਹੀਂਤਾਂ {
            ਜੇ (i < ਸੱਜਾ) quicksort_ਮੁੱਲਾਂ_indices_desc(ਮੁੱਲਾਂ, indices, i, ਸੱਜਾ);
            ਸੱਜਾ = j;
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ uint32_t ਸਿਖਰ_k_max_ਮੁੱਲ_ਸੂਚਕਾਂਕ(ਅਚਲ ਦਸ਼ਮਲਵ * ਮੁੱਲਾਂ, uint32_t n, ਦਸ਼ਮਲਵ * ਮੁੱਲ) {
    int32_t max_idx = 0;
    hvx_argmax_f32(ਮੁੱਲਾਂ, n, 0, ਮੁੱਲ, &max_idx);
    ਵਾਪਸ (uint32_t) max_idx;
}

ਸਥਿਰ ਅਚਲ int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ init_indices_ramp(int32_t * indices_buf, uint32_t ਗਿਣਤੀ_2, uint32_t ਅਧਾਰ) {
    ਅਚਲ HVX_ਸਦਿਸ਼ ind_init_vec = Q6_Vw_vadd_VwVw(*(ਅਚਲ HVX_ਸਦਿਸ਼ *) argsort_ramp_lut, Q6_V_vsplat_R(ਅਧਾਰ));
    ਅਚਲ HVX_ਸਦਿਸ਼ ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_ਸਦਿਸ਼ * indices_buf_vec = (HVX_ਸਦਿਸ਼ *) indices_buf;
    uint32_t ਅੰਕ_vecs = ਗਿਣਤੀ_2 / 32;
    HVX_ਸਦਿਸ਼ curr_ind_vec = ind_init_vec;
    ਲਈ (uint32_t j = 0; j < ਅੰਕ_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = ਗਿਣਤੀ_2 % 32;
    ਜੇ (rem > 0) {
        hvx_vec_store_a((ਖਾਲੀ *) &indices_buf_vec[ਅੰਕ_vecs], rem * ਆਕਾਰ(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ vec_cas(HVX_ਸਦਿਸ਼ * X_val, HVX_ਸਦਿਸ਼ * X_idx, HVX_ਸਦਿਸ਼ * Y_val, HVX_ਸਦਿਸ਼ * Y_idx, bool asc) {
    HVX_ਸਦਿਸ਼_Pred pred = asc ? Q6_Q_vcmp_gt_VsfVsf(*X_val, *Y_val)
                              : Q6_Q_vcmp_gt_VsfVsf(*Y_val, *X_val);
    HVX_ਸਦਿਸ਼ ਅਗਲਾ_X_val = Q6_V_vmux_QVV(pred, *Y_val, *X_val);
    HVX_ਸਦਿਸ਼ ਅਗਲਾ_Y_val = Q6_V_vmux_QVV(pred, *X_val, *Y_val);
    HVX_ਸਦਿਸ਼ ਅਗਲਾ_X_idx = Q6_V_vmux_QVV(pred, *Y_idx, *X_idx);
    HVX_ਸਦਿਸ਼ Y_tmp_idx  = Q6_V_vmux_QVV(pred, *X_idx, *Y_idx);
    *X_val = ਅਗਲਾ_X_val;
    *Y_val = ਅਗਲਾ_Y_val;
    *X_idx = ਅਗਲਾ_X_idx;
    *Y_idx = Y_tmp_idx;
}

__attribute__((always_inline))
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ bitonic_cas_32(HVX_ਸਦਿਸ਼ * V, HVX_ਸਦਿਸ਼ * I, ਪੂਰਨ d, HVX_ਸਦਿਸ਼_Pred dir_ਨਕਾਬ, HVX_ਸਦਿਸ਼ idx_vec, HVX_ਸਦਿਸ਼ zero_vec) {
    HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ_ਖੱਬਾ;
    HVX_ਸਦਿਸ਼ V_rot_ਖੱਬਾ, V_rot_ਸੱਜਾ;
    HVX_ਸਦਿਸ਼ I_rot_ਖੱਬਾ, I_rot_ਸੱਜਾ;

    ਜੇ (d == 1) {
        ਨਕਾਬ_ਖੱਬਾ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_ਖੱਬਾ = Q6_V_vror_VR(*V, 4);
        V_rot_ਸੱਜਾ = Q6_V_vror_VR(*V, 124);
        I_rot_ਖੱਬਾ = Q6_V_vror_VR(*I, 4);
        I_rot_ਸੱਜਾ = Q6_V_vror_VR(*I, 124);
    } ਨਹੀਂਤਾਂ ਜੇ (d == 2) {
        ਨਕਾਬ_ਖੱਬਾ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_ਖੱਬਾ = Q6_V_vror_VR(*V, 8);
        V_rot_ਸੱਜਾ = Q6_V_vror_VR(*V, 120);
        I_rot_ਖੱਬਾ = Q6_V_vror_VR(*I, 8);
        I_rot_ਸੱਜਾ = Q6_V_vror_VR(*I, 120);
    } ਨਹੀਂਤਾਂ ਜੇ (d == 4) {
        ਨਕਾਬ_ਖੱਬਾ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_ਖੱਬਾ = Q6_V_vror_VR(*V, 16);
        V_rot_ਸੱਜਾ = Q6_V_vror_VR(*V, 112);
        I_rot_ਖੱਬਾ = Q6_V_vror_VR(*I, 16);
        I_rot_ਸੱਜਾ = Q6_V_vror_VR(*I, 112);
    } ਨਹੀਂਤਾਂ ਜੇ (d == 8) {
        ਨਕਾਬ_ਖੱਬਾ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_ਖੱਬਾ = Q6_V_vror_VR(*V, 32);
        V_rot_ਸੱਜਾ = Q6_V_vror_VR(*V, 96);
        I_rot_ਖੱਬਾ = Q6_V_vror_VR(*I, 32);
        I_rot_ਸੱਜਾ = Q6_V_vror_VR(*I, 96);
    } ਨਹੀਂਤਾਂ {
        ਨਕਾਬ_ਖੱਬਾ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_ਖੱਬਾ = Q6_V_vror_VR(*V, 64);
        V_rot_ਸੱਜਾ = Q6_V_vror_VR(*V, 64);
        I_rot_ਖੱਬਾ = Q6_V_vror_VR(*I, 64);
        I_rot_ਸੱਜਾ = Q6_V_vror_VR(*I, 64);
    }

    HVX_ਸਦਿਸ਼ V_paired = Q6_V_vmux_QVV(ਨਕਾਬ_ਖੱਬਾ, V_rot_ਖੱਬਾ, V_rot_ਸੱਜਾ);
    HVX_ਸਦਿਸ਼ I_paired = Q6_V_vmux_QVV(ਨਕਾਬ_ਖੱਬਾ, I_rot_ਖੱਬਾ, I_rot_ਸੱਜਾ);

    HVX_ਸਦਿਸ਼_Pred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_ਸਦਿਸ਼_Pred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_ਸਦਿਸ਼_Pred ਨਕਾਬ_ਸੱਜਾ = Q6_Q_not_Q(ਨਕਾਬ_ਖੱਬਾ);
    HVX_ਸਦਿਸ਼_Pred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(ਨਕਾਬ_ਖੱਬਾ, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, ਨਕਾਬ_ਸੱਜਾ)
    );
    HVX_ਸਦਿਸ਼_Pred Q_ਅਦਲਾ_ਬਦਲੀ = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_ਨਕਾਬ, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_ਨਕਾਬ), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_ਅਦਲਾ_ਬਦਲੀ, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_ਅਦਲਾ_ਬਦਲੀ, I_paired, *I);
}

__attribute__((always_inline))
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ bitonic_ਛਾਂਟੋ_generic_hvx(uint8_t * ਮੁੱਲਾਂ, uint8_t * indices, ਪੂਰਨ K, bool asc_ਕ੍ਰਮ) {
    HVX_ਸਦਿਸ਼ V[32];
    HVX_ਸਦਿਸ਼ I[32];

    HVX_ਸਦਿਸ਼ zero_vec = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ idx_vec = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) argsort_ramp_lut;

    ਲਈ (ਪੂਰਨ v = 0; v < K; v++) {
        V[v] = *(HVX_ਸਦਿਸ਼ *)(ਮੁੱਲਾਂ + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_ਸਦਿਸ਼_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_ਸਦਿਸ਼_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    ਪੂਰਨ M = 5;
    ਜਦੋਂ ((1 << (M - 5)) < K) M++;

    ਲਈ (ਪੂਰਨ s = 1; s <= M; s++) {
        ਲਈ (ਪੂਰਨ stage_d = s - 1; stage_d >= 0; stage_d--) {
            ਪੂਰਨ d = 1 << stage_d;
            ਜੇ (d >= 32) {
                ਪੂਰਨ v_dist = d / 32;
                ਲਈ (ਪੂਰਨ v1 = 0; v1 < K; v1++) {
                    ਜੇ ((v1 & v_dist) == 0) {
                        ਪੂਰਨ v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_ਕ੍ਰਮ;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } ਨਹੀਂਤਾਂ {
                ਜੇ (s < 5) {
                    HVX_ਸਦਿਸ਼_Pred dir_ਨਕਾਬ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    ਲਈ (ਪੂਰਨ v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_ਨਕਾਬ, idx_vec, zero_vec);
                    }
                } ਨਹੀਂਤਾਂ {
                    ਲਈ (ਪੂਰਨ v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_ਕ੍ਰਮ;
                        HVX_ਸਦਿਸ਼_Pred dir_ਨਕਾਬ = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_ਨਕਾਬ, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    ਲਈ (ਪੂਰਨ v = 0; v < K; v++) {
        *(HVX_ਸਦਿਸ਼ *)(ਮੁੱਲਾਂ + v * 128)  = V[v];
        *(HVX_ਸਦਿਸ਼ *)(indices + v * 128) = I[v];
    }
}

ਸਥਿਰ ਖਾਲੀ bitonic_ਛਾਂਟੋ_vtcm_desc(uint8_t * ਮੁੱਲਾਂ, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_ਸਦਿਸ਼ zero_vec = Q6_V_vzero();
    HVX_ਸਦਿਸ਼ idx_vec = *(ਅਚਲ HVX_ਸਦਿਸ਼ *) argsort_ramp_lut;

    HVX_ਸਦਿਸ਼_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_ਸਦਿਸ਼_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    ਜੇ (init_indices) {
        ਲਈ (uint32_t v = 0; v < n_vec; v++) {
            HVX_ਸਦਿਸ਼ idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_ਸਦਿਸ਼ *)(indices + v * 128) = idx;
        }
    }

    ਪੂਰਨ M = 5;
    ਜਦੋਂ ((1u << (M - 5)) < n_vec) M++;

    ਲਈ (ਪੂਰਨ s = 1; s <= M; s++) {
        ਲਈ (ਪੂਰਨ stage_d = s - 1; stage_d >= 0; stage_d--) {
            ਪੂਰਨ d = 1 << stage_d;
            ਜੇ (d >= 32) {
                uint32_t v_dist = d / 32;
                ਲਈ (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    ਜੇ ((v1 & v_dist) == 0) {
                        uint32_t v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : false;

                        HVX_ਸਦਿਸ਼ Vv1 = *(HVX_ਸਦਿਸ਼ *)(ਮੁੱਲਾਂ + v1 * 128);
                        HVX_ਸਦਿਸ਼ Iv1 = *(HVX_ਸਦਿਸ਼ *)(indices + v1 * 128);
                        HVX_ਸਦਿਸ਼ Vv2 = *(HVX_ਸਦਿਸ਼ *)(ਮੁੱਲਾਂ + v2 * 128);
                        HVX_ਸਦਿਸ਼ Iv2 = *(HVX_ਸਦਿਸ਼ *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_ਸਦਿਸ਼ *)(ਮੁੱਲਾਂ + v1 * 128)  = Vv1;
                        *(HVX_ਸਦਿਸ਼ *)(indices + v1 * 128) = Iv1;
                        *(HVX_ਸਦਿਸ਼ *)(ਮੁੱਲਾਂ + v2 * 128)  = Vv2;
                        *(HVX_ਸਦਿਸ਼ *)(indices + v2 * 128) = Iv2;
                    }
                }
            } ਨਹੀਂਤਾਂ {
                ਜੇ (s < 5) {
                    HVX_ਸਦਿਸ਼_Pred dir_ਨਕਾਬ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    ਲਈ (uint32_t v = 0; v < n_vec; v++) {
                        HVX_ਸਦਿਸ਼ Vv = *(HVX_ਸਦਿਸ਼ *)(ਮੁੱਲਾਂ + v * 128);
                        HVX_ਸਦਿਸ਼ Iv = *(HVX_ਸਦਿਸ਼ *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_ਨਕਾਬ, idx_vec, zero_vec);

                        *(HVX_ਸਦਿਸ਼ *)(ਮੁੱਲਾਂ + v * 128)  = Vv;
                        *(HVX_ਸਦਿਸ਼ *)(indices + v * 128) = Iv;
                    }
                } ਨਹੀਂਤਾਂ {
                    ਲਈ (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_ਸਦਿਸ਼_Pred dir_ਨਕਾਬ = asc ? pred_all_1s : pred_all_0s;

                        HVX_ਸਦਿਸ਼ Vv = *(HVX_ਸਦਿਸ਼ *)(ਮੁੱਲਾਂ + v * 128);
                        HVX_ਸਦਿਸ਼ Iv = *(HVX_ਸਦਿਸ਼ *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_ਨਕਾਬ, idx_vec, zero_vec);

                        *(HVX_ਸਦਿਸ਼ *)(ਮੁੱਲਾਂ + v * 128)  = Vv;
                        *(HVX_ਸਦਿਸ਼ *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ ਸਿਖਰ_k_select_tiled(dma_ਕਤਾਰ * dma_q, ਬਣਤਰ htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               ਦਸ਼ਮਲਵ * ਮੁੱਲਾਂ_buf, int32_t * indices_buf,
                               ਦਸ਼ਮਲਵ * ਬਾਹਰਲਾ_ਮੁੱਲਾਂ, int32_t * ਬਾਹਰਲਾ_indices,
                               uint32_t chunk_ਅਧਾਰ) {
    ਅਚਲ uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_ਗਿਣਤੀ    = n_tiles * k;
    uint32_t ਮਿਲਾਓ_n_vec        = hmx_ceil_div(candidate_ਗਿਣਤੀ, 32);
    uint32_t ਮਿਲਾਓ_n_vec_pow2   = 1;
    ਜਦੋਂ (ਮਿਲਾਓ_n_vec_pow2 < ਮਿਲਾਓ_n_vec) ਮਿਲਾਓ_n_vec_pow2 <<= 1;
    uint32_t ਮਿਲਾਓ_elems        = ਮਿਲਾਓ_n_vec_pow2 * 32;
    ਦਸ਼ਮਲਵ * candidate_ਮੁੱਲਾਂ    = ਮੁੱਲਾਂ_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    ਲਈ (uint32_t ਵਿੱਥ = 0; ਵਿੱਥ < n; ਵਿੱਥ += tile_elems) {
        uint32_t tile_ਗਿਣਤੀ = MIN(tile_elems, n - ਵਿੱਥ);
        dma_sync_read(dma_q, ਮੁੱਲਾਂ_buf, src_addr + (size_t) ਵਿੱਥ * ਆਕਾਰ(ਦਸ਼ਮਲਵ), tile_ਗਿਣਤੀ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ਵਿੱਥ);
        ਜੇ (tile_ਗਿਣਤੀ < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (ਮੁੱਲਾਂ_buf + tile_ਗਿਣਤੀ), -INFINITY, tile_elems - tile_ਗਿਣਤੀ);
        }

        bitonic_ਛਾਂਟੋ_vtcm_desc((uint8_t *) ਮੁੱਲਾਂ_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_ਗਿਣਤੀ);
        ਲਈ (uint32_t j = 0; j < tile_k; j++) {
            candidate_ਮੁੱਲਾਂ[candidate_pos] = ਮੁੱਲਾਂ_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) ਵਿੱਥ;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) ਵਿੱਥ);
    }

    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    ਜੇ (ਮਿਲਾਓ_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_ਮੁੱਲਾਂ + candidate_pos), -INFINITY, ਮਿਲਾਓ_elems - candidate_pos);
        ਲਈ (uint32_t j = candidate_pos; j < ਮਿਲਾਓ_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_ਛਾਂਟੋ_vtcm_desc((uint8_t *) candidate_ਮੁੱਲਾਂ, (uint8_t *) candidate_indices, ਮਿਲਾਓ_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    ਲਈ (uint32_t j = 0; j < k; j++) {
        ਬਾਹਰਲਾ_ਮੁੱਲਾਂ[j]  = candidate_ਮੁੱਲਾਂ[j];
        ਬਾਹਰਲਾ_indices[j] = candidate_indices[j] + (int32_t) chunk_ਅਧਾਰ;
    }
}

__attribute__((always_inline))
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ sort32_f32_hvx(uint8_t * ਮੁੱਲਾਂ, uint8_t * indices, ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ) {
    bitonic_ਛਾਂਟੋ_generic_hvx(ਮੁੱਲਾਂ, indices, 1, ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC);
}

__attribute__((always_inline))
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ sort64_f32_hvx(uint8_t * ਮੁੱਲਾਂ, uint8_t * indices, ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ) {
    bitonic_ਛਾਂਟੋ_generic_hvx(ਮੁੱਲਾਂ, indices, 2, ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC);
}

__attribute__((always_inline))
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ sort128_f32_hvx(uint8_t * ਮੁੱਲਾਂ, uint8_t * indices, ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ) {
    bitonic_ਛਾਂਟੋ_generic_hvx(ਮੁੱਲਾਂ, indices, 4, ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC);
}

__attribute__((always_inline))
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ sort256_f32_hvx(uint8_t * ਮੁੱਲਾਂ, uint8_t * indices, ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ) {
    bitonic_ਛਾਂਟੋ_generic_hvx(ਮੁੱਲਾਂ, indices, 8, ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC);
}

__attribute__((always_inline))
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ sort512_f32_hvx(uint8_t * ਮੁੱਲਾਂ, uint8_t * indices, ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ) {
    bitonic_ਛਾਂਟੋ_generic_hvx(ਮੁੱਲਾਂ, indices, 16, ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC);
}

__attribute__((always_inline))
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ sort1024_f32_hvx(uint8_t * ਮੁੱਲਾਂ, uint8_t * indices, ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ) {
    bitonic_ਛਾਂਟੋ_generic_hvx(ਮੁੱਲਾਂ, indices, 32, ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC);
}

ਸਥਿਰ ਖਾਲੀ ਮਿਲਾਓ_runs(
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ ਵਿੱਚ_val0, ਅਚਲ int32_t * ਸੀਮਿਤ ਵਿੱਚ_idx0, uint32_t n0,
    ਅਚਲ ਦਸ਼ਮਲਵ * ਸੀਮਿਤ ਵਿੱਚ_val1, ਅਚਲ int32_t * ਸੀਮਿਤ ਵਿੱਚ_idx1, uint32_t n1,
    ਦਸ਼ਮਲਵ * ਸੀਮਿਤ ਬਾਹਰਲਾ_val, int32_t * ਸੀਮਿਤ ਬਾਹਰਲਾ_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t ਬਾਹਰਲਾ = 0;

    ਜੇ (asc) {
        ਜਦੋਂ (i0 < n0 && i1 < n1) {
            ਜੇ (ਵਿੱਚ_val0[i0] <= ਵਿੱਚ_val1[i1]) {
                ਜੇ (ਬਾਹਰਲਾ_val) ਬਾਹਰਲਾ_val[ਬਾਹਰਲਾ] = ਵਿੱਚ_val0[i0];
                ਬਾਹਰਲਾ_idx[ਬਾਹਰਲਾ] = ਵਿੱਚ_idx0[i0];
                i0++;
            } ਨਹੀਂਤਾਂ {
                ਜੇ (ਬਾਹਰਲਾ_val) ਬਾਹਰਲਾ_val[ਬਾਹਰਲਾ] = ਵਿੱਚ_val1[i1];
                ਬਾਹਰਲਾ_idx[ਬਾਹਰਲਾ] = ਵਿੱਚ_idx1[i1];
                i1++;
            }
            ਬਾਹਰਲਾ++;
        }
    } ਨਹੀਂਤਾਂ {
        ਜਦੋਂ (i0 < n0 && i1 < n1) {
            ਜੇ (ਵਿੱਚ_val0[i0] >= ਵਿੱਚ_val1[i1]) {
                ਜੇ (ਬਾਹਰਲਾ_val) ਬਾਹਰਲਾ_val[ਬਾਹਰਲਾ] = ਵਿੱਚ_val0[i0];
                ਬਾਹਰਲਾ_idx[ਬਾਹਰਲਾ] = ਵਿੱਚ_idx0[i0];
                i0++;
            } ਨਹੀਂਤਾਂ {
                ਜੇ (ਬਾਹਰਲਾ_val) ਬਾਹਰਲਾ_val[ਬਾਹਰਲਾ] = ਵਿੱਚ_val1[i1];
                ਬਾਹਰਲਾ_idx[ਬਾਹਰਲਾ] = ਵਿੱਚ_idx1[i1];
                i1++;
            }
            ਬਾਹਰਲਾ++;
        }
    }

    ਜਦੋਂ (i0 < n0) {
        ਜੇ (ਬਾਹਰਲਾ_val) ਬਾਹਰਲਾ_val[ਬਾਹਰਲਾ] = ਵਿੱਚ_val0[i0];
        ਬਾਹਰਲਾ_idx[ਬਾਹਰਲਾ] = ਵਿੱਚ_idx0[i0];
        i0++;
        ਬਾਹਰਲਾ++;
    }

    ਜਦੋਂ (i1 < n1) {
        ਜੇ (ਬਾਹਰਲਾ_val) ਬਾਹਰਲਾ_val[ਬਾਹਰਲਾ] = ਵਿੱਚ_val1[i1];
        ਬਾਹਰਲਾ_idx[ਬਾਹਰਲਾ] = ਵਿੱਚ_idx1[i1];
        i1++;
        ਬਾਹਰਲਾ++;
    }
}

ਬਣਤਰ htp_ਛਾਂਟੋ_chunk_ctx {
    ਬਣਤਰ htp_ops_context *              octx;
    ਅਚਲ ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params * kparams;
    uint8_t *                             vtcm_ਅਧਾਰ;
    uint32_t                              real_ਗਿਣਤੀ[HTP_MAX_NTHREADS];
};

ਸਥਿਰ ਖਾਲੀ htp_ਛਾਂਟੋ_chunk_ਕੰਮ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_ਛਾਂਟੋ_chunk_ctx * cctx = (ਬਣਤਰ htp_ਛਾਂਟੋ_chunk_ctx *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = cctx->octx;
    ਅਚਲ ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params * kparams = cctx->kparams;
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_ਅਧਾਰ  = i * chunk_elems;

    uint32_t real_ਗਿਣਤੀ = (chunk_ਅਧਾਰ < ne00) ? MIN(chunk_elems, ne00 - chunk_ਅਧਾਰ) : 0;
    cctx->real_ਗਿਣਤੀ[i] = real_ਗਿਣਤੀ;

    uint8_t * spad = cctx->vtcm_ਅਧਾਰ + (size_t) kparams->phase1_ਥਾਂ_ਮਾਪ * i;
    ਦਸ਼ਮਲਵ *   ਮੁੱਲਾਂ_buf  = (ਦਸ਼ਮਲਵ *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਦਸ਼ਮਲਵ *   ਮਿਲਾਓ_ਮੁੱਲਾਂ  = (ਦਸ਼ਮਲਵ *)   (cctx->vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_ਮੁੱਲਾਂ_off);
    int32_t * ਮਿਲਾਓ_indices = (int32_t *) (cctx->vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_indices_off);

    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[i];
    ਅਚਲ dma_addr_t src_addr = src0->ਅੰਕੜੇ + (size_t) chunk_ਅਧਾਰ * ਆਕਾਰ(ਦਸ਼ਮਲਵ);

    ਜੇ (kparams->is_ਸਿਖਰ_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        ਜੇ (real_ਗਿਣਤੀ == 0) {
            ਲਈ (uint32_t j = 0; j < local_k; j++) {
                ਮਿਲਾਓ_ਮੁੱਲਾਂ[i * local_k + j]  = -INFINITY;
                ਮਿਲਾਓ_indices[i * local_k + j] = 0;
            }
            ਵਾਪਸ;
        }

        ਜੇ (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], ਮੁੱਲਾਂ_buf, src_addr, real_ਗਿਣਤੀ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            ਦਸ਼ਮਲਵ max_ਮੁੱਲ;
            uint32_t max_ਸੂਚਕਾਂਕ = ਸਿਖਰ_k_max_ਮੁੱਲ_ਸੂਚਕਾਂਕ(ਮੁੱਲਾਂ_buf, real_ਗਿਣਤੀ, &max_ਮੁੱਲ);
            ਮਿਲਾਓ_ਮੁੱਲਾਂ[i] = max_ਮੁੱਲ;
            ਮਿਲਾਓ_indices[i] = (int32_t) (max_ਸੂਚਕਾਂਕ + chunk_ਅਧਾਰ);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            ਵਾਪਸ;
        }

        ਜੇ (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            ਸਿਖਰ_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_ਗਿਣਤੀ, local_k,
                               ਮੁੱਲਾਂ_buf, indices_buf,
                               &ਮਿਲਾਓ_ਮੁੱਲਾਂ[i * local_k], &ਮਿਲਾਓ_indices[i * local_k],
                               chunk_ਅਧਾਰ);
            ਵਾਪਸ;
        }

        dma_sync_read(octx->ctx->dma[i], ਮੁੱਲਾਂ_buf, src_addr, real_ਗਿਣਤੀ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

        htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        ਜੇ (chunk_elems > real_ਗਿਣਤੀ) {
            hvx_splat_f32_u((uint8_t *)(ਮੁੱਲਾਂ_buf + real_ਗਿਣਤੀ), -INFINITY, chunk_elems - real_ਗਿਣਤੀ);
        }

        bitonic_ਛਾਂਟੋ_vtcm_desc((uint8_t *) ਮੁੱਲਾਂ_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        ਲਈ (uint32_t j = 0; j < local_k; j++) {
            ਮਿਲਾਓ_ਮੁੱਲਾਂ[i * local_k + j]  = ਮੁੱਲਾਂ_buf[j];
            ਮਿਲਾਓ_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_ਅਧਾਰ;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } ਨਹੀਂਤਾਂ {
        ਜੇ (real_ਗਿਣਤੀ > 0) {
            dma_sync_read(octx->ctx->dma[i], ਮੁੱਲਾਂ_buf, src_addr, real_ਗਿਣਤੀ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_ਗਿਣਤੀ, chunk_ਅਧਾਰ);

            ਜੇ (kparams->ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC) {
                quicksort_ਮੁੱਲਾਂ_indices_asc(ਮੁੱਲਾਂ_buf, indices_buf, 0, real_ਗਿਣਤੀ - 1);
            } ਨਹੀਂਤਾਂ {
                quicksort_ਮੁੱਲਾਂ_indices_desc(ਮੁੱਲਾਂ_buf, indices_buf, 0, real_ਗਿਣਤੀ - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

ਬਣਤਰ htp_argsort_ਮਿਲਾਓ_l1_ctx {
    ਬਣਤਰ htp_ਛਾਂਟੋ_chunk_ctx * cctx;
    bool                        asc;
};

ਸਥਿਰ ਖਾਲੀ htp_argsort_ਮਿਲਾਓ_l1_ਕੰਮ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_argsort_ਮਿਲਾਓ_l1_ctx * mctx = (ਬਣਤਰ htp_argsort_ਮਿਲਾਓ_l1_ctx *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ਛਾਂਟੋ_chunk_ctx * cctx = mctx->cctx;
    ਅਚਲ ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_ਗਿਣਤੀ[c0];
    uint32_t count1 = cctx->real_ਗਿਣਤੀ[c1];

    size_t ਥਾਂ_ਮਾਪ = (size_t) kparams->phase1_ਥਾਂ_ਮਾਪ;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    ਅਚਲ ਦਸ਼ਮਲਵ *   val0 = (ਅਚਲ ਦਸ਼ਮਲਵ *)   (cctx->vtcm_ਅਧਾਰ + ਥਾਂ_ਮਾਪ * c0);
    ਅਚਲ int32_t * idx0 = (ਅਚਲ int32_t *) (cctx->vtcm_ਅਧਾਰ + ਥਾਂ_ਮਾਪ * c0 + chunk_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
    ਅਚਲ ਦਸ਼ਮਲਵ *   val1 = (ਅਚਲ ਦਸ਼ਮਲਵ *)   (cctx->vtcm_ਅਧਾਰ + ਥਾਂ_ਮਾਪ * c1);
    ਅਚਲ int32_t * idx1 = (ਅਚਲ int32_t *) (cctx->vtcm_ਅਧਾਰ + ਥਾਂ_ਮਾਪ * c1 + chunk_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    uint32_t ਬਾਹਰਲਾ_ਵਿੱਥ = (2 * i) * chunk_elems;
    ਦਸ਼ਮਲਵ *   ਬਾਹਰਲਾ_val = (ਦਸ਼ਮਲਵ *)   (cctx->vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_ਮੁੱਲਾਂ_off)  + ਬਾਹਰਲਾ_ਵਿੱਥ;
    int32_t * ਬਾਹਰਲਾ_idx = (int32_t *) (cctx->vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_indices_off) + ਬਾਹਰਲਾ_ਵਿੱਥ;

    ਬਣਤਰ htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    ਮਿਲਾਓ_runs(val0, idx0, count0, val1, idx1, count1, ਬਾਹਰਲਾ_val, ਬਾਹਰਲਾ_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

ਬਣਤਰ htp_argsort_ਮਿਲਾਓ_l2_ctx {
    ਬਣਤਰ htp_ਛਾਂਟੋ_chunk_ctx * cctx;
    bool                        asc;
};

ਸਥਿਰ ਖਾਲੀ htp_argsort_ਮਿਲਾਓ_l2_ਕੰਮ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ i, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_argsort_ਮਿਲਾਓ_l2_ctx * mctx = (ਬਣਤਰ htp_argsort_ਮਿਲਾਓ_l2_ctx *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ਛਾਂਟੋ_chunk_ctx * cctx = mctx->cctx;
    ਅਚਲ ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params * kparams = cctx->kparams;

    size_t ਥਾਂ_ਮਾਪ = (size_t) kparams->phase1_ਥਾਂ_ਮਾਪ;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    ਅਚਲ ਦਸ਼ਮਲਵ *   val0 = (ਅਚਲ ਦਸ਼ਮਲਵ *)   (cctx->vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_ਮੁੱਲਾਂ_off)  + off0;
    ਅਚਲ int32_t * idx0 = (ਅਚਲ int32_t *) (cctx->vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_indices_off) + off0;
    ਅਚਲ ਦਸ਼ਮਲਵ *   val1 = (ਅਚਲ ਦਸ਼ਮਲਵ *)   (cctx->vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_ਮੁੱਲਾਂ_off)  + off1;
    ਅਚਲ int32_t * idx1 = (ਅਚਲ int32_t *) (cctx->vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_indices_off) + off1;

    uint32_t count0 = cctx->real_ਗਿਣਤੀ[4*i + 0] + cctx->real_ਗਿਣਤੀ[4*i + 1];
    uint32_t count1 = cctx->real_ਗਿਣਤੀ[4*i + 2] + cctx->real_ਗਿਣਤੀ[4*i + 3];

    uint8_t * ਥਾਂ_ਅਧਾਰ = cctx->vtcm_ਅਧਾਰ + (4 * i) * ਥਾਂ_ਮਾਪ;
    ਦਸ਼ਮਲਵ *   ਬਾਹਰਲਾ_val   = (ਦਸ਼ਮਲਵ *) ਥਾਂ_ਅਧਾਰ;
    int32_t * ਬਾਹਰਲਾ_idx   = (int32_t *) (ਥਾਂ_ਅਧਾਰ + 4 * chunk_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    ਬਣਤਰ htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    ਮਿਲਾਓ_runs(val0, idx0, count0, val1, idx1, count1, ਬਾਹਰਲਾ_val, ਬਾਹਰਲਾ_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

ਬਣਤਰ htp_ਛਾਂਟੋ_multi_ਪੰਕਤੀ_ctx {
    ਬਣਤਰ htp_ops_context *              octx;
    ਅਚਲ ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params * kparams;
    uint8_t *                             vtcm_ਅਧਾਰ;
    uint32_t                              nrows_ਪ੍ਰਤੀ_thread;
};

ਬਣਤਰ htp_tensor_ਪੰਕਤੀ_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ਸਥਿਰ ਸਤਰਅੰਦਰ ਬਣਤਰ htp_tensor_ਪੰਕਤੀ_coord init_ਪੰਕਤੀ_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    ਵਾਪਸ (ਬਣਤਰ htp_tensor_ਪੰਕਤੀ_coord){ i01, i02, i03 };
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ ਕਦਮ_ਪੰਕਤੀ_coord(ਬਣਤਰ htp_tensor_ਪੰਕਤੀ_coord * c, uint32_t ne01, uint32_t ne02) {
    c->i01++;
    ਜੇ (c->i01 == ne01) {
        c->i01 = 0;
        c->i02++;
        ਜੇ (c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ size_t coord_ਵਿੱਥ(ਬਣਤਰ htp_tensor_ਪੰਕਤੀ_coord c, size_t nb1, size_t nb2, size_t nb3) {
    ਵਾਪਸ (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ compute_ਪੰਕਤੀ_ਛਾਂਟੋ(
    ਦਸ਼ਮਲਵ * cur_ਮੁੱਲਾਂ,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_ਸਿਖਰ_k,
    ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ
) {
    ਜੇ (is_ਸਿਖਰ_k) {
        ਜੇ (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            ਚੋਣ (ne00) {
                ਹਾਲਤ 32:   sort32_f32_hvx((uint8_t *) cur_ਮੁੱਲਾਂ, (uint8_t *) cur_indices, GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC); ਰੋਕੋ;
                ਹਾਲਤ 64:   sort64_f32_hvx((uint8_t *) cur_ਮੁੱਲਾਂ, (uint8_t *) cur_indices, GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC); ਰੋਕੋ;
                ਹਾਲਤ 128:  sort128_f32_hvx((uint8_t *) cur_ਮੁੱਲਾਂ, (uint8_t *) cur_indices, GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC); ਰੋਕੋ;
                ਹਾਲਤ 256:  sort256_f32_hvx((uint8_t *) cur_ਮੁੱਲਾਂ, (uint8_t *) cur_indices, GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC); ਰੋਕੋ;
                ਹਾਲਤ 512:  sort512_f32_hvx((uint8_t *) cur_ਮੁੱਲਾਂ, (uint8_t *) cur_indices, GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC); ਰੋਕੋ;
                ਹਾਲਤ 1024: sort1024_f32_hvx((uint8_t *) cur_ਮੁੱਲਾਂ, (uint8_t *) cur_indices, GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC); ਰੋਕੋ;
            }
        } ਨਹੀਂਤਾਂ {
            ਜੇ (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_ਮੁੱਲਾਂ + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_ਛਾਂਟੋ_vtcm_desc((uint8_t *) cur_ਮੁੱਲਾਂ, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } ਨਹੀਂਤਾਂ {
        ਜੇ (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            ਚੋਣ (ne00) {
                ਹਾਲਤ 32:   sort32_f32_hvx((uint8_t *) cur_ਮੁੱਲਾਂ, (uint8_t *) cur_indices, ਕ੍ਰਮ); ਰੋਕੋ;
                ਹਾਲਤ 64:   sort64_f32_hvx((uint8_t *) cur_ਮੁੱਲਾਂ, (uint8_t *) cur_indices, ਕ੍ਰਮ); ਰੋਕੋ;
                ਹਾਲਤ 128:  sort128_f32_hvx((uint8_t *) cur_ਮੁੱਲਾਂ, (uint8_t *) cur_indices, ਕ੍ਰਮ); ਰੋਕੋ;
                ਹਾਲਤ 256:  sort256_f32_hvx((uint8_t *) cur_ਮੁੱਲਾਂ, (uint8_t *) cur_indices, ਕ੍ਰਮ); ਰੋਕੋ;
                ਹਾਲਤ 512:  sort512_f32_hvx((uint8_t *) cur_ਮੁੱਲਾਂ, (uint8_t *) cur_indices, ਕ੍ਰਮ); ਰੋਕੋ;
                ਹਾਲਤ 1024: sort1024_f32_hvx((uint8_t *) cur_ਮੁੱਲਾਂ, (uint8_t *) cur_indices, ਕ੍ਰਮ); ਰੋਕੋ;
            }
        } ਨਹੀਂਤਾਂ {
            init_indices_ramp(cur_indices, ne00, 0);
            ਜੇ (ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC) {
                quicksort_ਮੁੱਲਾਂ_indices_asc(cur_ਮੁੱਲਾਂ, cur_indices, 0, ne00 - 1);
            } ਨਹੀਂਤਾਂ {
                quicksort_ਮੁੱਲਾਂ_indices_desc(cur_ਮੁੱਲਾਂ, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

ਸਥਿਰ ਖਾਲੀ htp_ਛਾਂਟੋ_multi_ਪੰਕਤੀ_ਕੰਮ(ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ n, ਬਿਨਾਨਿਸ਼ਾਨੀ ਪੂਰਨ ith, ਖਾਲੀ * ਅੰਕੜੇ) {
    ਬਣਤਰ htp_ਛਾਂਟੋ_multi_ਪੰਕਤੀ_ctx * mctx = (ਬਣਤਰ htp_ਛਾਂਟੋ_multi_ਪੰਕਤੀ_ctx *) ਅੰਕੜੇ;
    ਬਣਤਰ htp_ops_context * octx = mctx->octx;
    ਅਚਲ ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params * kparams = mctx->kparams;
    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    uint32_t ne00 = (uint32_t) kparams->ne00;
    uint32_t ne01 = src0->ne[1];
    uint32_t ne02 = src0->ne[2];

    uint32_t nb01 = src0->nb[1];
    uint32_t nb02 = src0->nb[2];
    uint32_t nb03 = src0->nb[3];

    uint32_t nb1 = dst->nb[1];
    uint32_t nb2 = dst->nb[2];
    uint32_t nb3 = dst->nb[3];

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = (uint32_t) kparams->ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    uint32_t ਪੰਕਤੀ_ਸਿਰਾ   = (uint32_t) kparams->ਪੰਕਤੀ_ਸਿਰਾ;
    uint32_t r_ਸ਼ੁਰੂ   = ਪੰਕਤੀ_ਸ਼ੁਰੂ + ith * mctx->nrows_ਪ੍ਰਤੀ_thread;
    uint32_t r_ਸਿਰਾ     = MIN(r_ਸ਼ੁਰੂ + mctx->nrows_ਪ੍ਰਤੀ_thread, ਪੰਕਤੀ_ਸਿਰਾ);

    ਜੇ (r_ਸ਼ੁਰੂ >= r_ਸਿਰਾ) ਵਾਪਸ;

    size_t spad_ਵਿੱਥ = ith * (size_t) (kparams->phase1_ਥਾਂ_ਮਾਪ * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_ਅਧਾਰ + spad_ਵਿੱਥ;
    size_t ਥਾਂ_ਮਾਪ = (size_t) kparams->phase1_ਥਾਂ_ਮਾਪ;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    ਦਸ਼ਮਲਵ *   ਮੁੱਲਾਂ_buf  = (ਦਸ਼ਮਲਵ *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

    dma_ਕਤਾਰ * dma_q = octx->ctx->dma[ith];
    ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_ਸਿਖਰ_k = kparams->is_ਸਿਖਰ_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ ਕ੍ਰਮ = (ਗਿਣਤੀ ggml_ਛਾਂਟੋ_ਕ੍ਰਮ) kparams->ਕ੍ਰਮ;

    ਅਚਲ uint32_t src_ਬਾਈਟਾਂ = ne00 * ਆਕਾਰ(ਦਸ਼ਮਲਵ);
    ਅਚਲ uint32_t dst_ਬਾਈਟਾਂ = (is_ਸਿਖਰ_k ? k : ne00) * ਆਕਾਰ(int32_t);

    ਜੇ (kparams->n_slots == 2) {
        ਬਣਤਰ htp_tensor_ਪੰਕਤੀ_coord ਅਭਾਜ_coord = init_ਪੰਕਤੀ_coord(r_ਸ਼ੁਰੂ, ne01, ne02);
        ਲਈ (uint32_t r = r_ਸ਼ੁਰੂ, spad_idx = 0; r < r_ਸਿਰਾ && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * ਥਾਂ_ਮਾਪ;
            ਦਸ਼ਮਲਵ *   cur_ਮੁੱਲਾਂ  = (ਦਸ਼ਮਲਵ *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

            dma_addr_t dst_addr = dst->ਅੰਕੜੇ + coord_ਵਿੱਥ(ਅਭਾਜ_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->ਅੰਕੜੇ + coord_ਵਿੱਥ(ਅਭਾਜ_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_addr, cur_indices),
                           dst_ਬਾਈਟਾਂ, dst_ਬਾਈਟਾਂ, dst_ਬਾਈਟਾਂ, 0);

            // Prefetch input row
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(cur_ਮੁੱਲਾਂ, src_addr),
                           src_ਬਾਈਟਾਂ, src_ਬਾਈਟਾਂ, src_ਬਾਈਟਾਂ, 1);

            ਕਦਮ_ਪੰਕਤੀ_coord(&ਅਭਾਜ_coord, ne01, ne02);
        }

        ਬਣਤਰ htp_tensor_ਪੰਕਤੀ_coord dst_coord = init_ਪੰਕਤੀ_coord(r_ਸ਼ੁਰੂ, ne01, ne02);
        ਬਣਤਰ htp_tensor_ਪੰਕਤੀ_coord ਅਗਲਾ_src_coord = init_ਪੰਕਤੀ_coord(r_ਸ਼ੁਰੂ + 2, ne01, ne02);

        ਲਈ (uint32_t r = r_ਸ਼ੁਰੂ; r < r_ਸਿਰਾ; r++) {
            uint32_t cur_ਥਾਂ = (r - r_ਸ਼ੁਰੂ) & 1;
            uint8_t * cur_spad = thread_spad + cur_ਥਾਂ * ਥਾਂ_ਮਾਪ;
            ਦਸ਼ਮਲਵ *   cur_ਮੁੱਲਾਂ  = (ਦਸ਼ਮਲਵ *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

            // Wait for previous writeback of this slot to complete
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_ਕਤਾਰ_ਕੱਢੋ(dma_q);

            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_ਪੰਕਤੀ_ਛਾਂਟੋ(cur_ਮੁੱਲਾਂ, cur_indices, ne00, chunk_elems, is_ਸਿਖਰ_k, ਕ੍ਰਮ);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->ਅੰਕੜੇ + coord_ਵਿੱਥ(dst_coord, nb1, nb2, nb3);
            dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(dst_addr, cur_indices),
                           dst_ਬਾਈਟਾਂ, dst_ਬਾਈਟਾਂ, dst_ਬਾਈਟਾਂ, 1);
            ਕਦਮ_ਪੰਕਤੀ_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            ਅਚਲ uint32_t ਅਗਲਾ_ਪੰਕਤੀ = r + 2;
            ਜੇ (ਅਗਲਾ_ਪੰਕਤੀ < r_ਸਿਰਾ) {
                dma_addr_t ਅਗਲਾ_src_addr = src0->ਅੰਕੜੇ + coord_ਵਿੱਥ(ਅਗਲਾ_src_coord, nb01, nb02, nb03);
                dma_ਕਤਾਰ_ਧੱਕੋ(dma_q, dma_make_ਅੰਕੜੇ(cur_ਮੁੱਲਾਂ, ਅਗਲਾ_src_addr),
                               src_ਬਾਈਟਾਂ, src_ਬਾਈਟਾਂ, src_ਬਾਈਟਾਂ, 1);
                ਕਦਮ_ਪੰਕਤੀ_coord(&ਅਗਲਾ_src_coord, ne01, ne02);
            }
        }

        dma_ਕਤਾਰ_flush(dma_q);
    } ਨਹੀਂਤਾਂ {
        ਬਣਤਰ htp_tensor_ਪੰਕਤੀ_coord coord = init_ਪੰਕਤੀ_coord(r_ਸ਼ੁਰੂ, ne01, ne02);
        ਲਈ (uint32_t r = r_ਸ਼ੁਰੂ; r < r_ਸਿਰਾ; r++) {
            dma_addr_t src_addr = src0->ਅੰਕੜੇ + coord_ਵਿੱਥ(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->ਅੰਕੜੇ + coord_ਵਿੱਥ(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, ਮੁੱਲਾਂ_buf, src_addr, src_ਬਾਈਟਾਂ);

            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_ਪੰਕਤੀ_ਛਾਂਟੋ(ਮੁੱਲਾਂ_buf, indices_buf, ne00, chunk_elems, is_ਸਿਖਰ_k, ਕ੍ਰਮ);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_ਬਾਈਟਾਂ);
            ਕਦਮ_ਪੰਕਤੀ_coord(&coord, ne01, ne02);
        }
    }
}

ਸਥਿਰ ਪੂਰਨ op_ਛਾਂਟੋ_common(ਬਣਤਰ htp_ops_context * octx, bool is_ਸਿਖਰ_k) {
    ਜੇ (octx->src[0]->type != HTP_TYPE_F32) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ ਬਣਤਰ htp_tensor * src0 = octx->src[0];
    ਅਚਲ ਬਣਤਰ htp_tensor * dst  = octx->dst;

    ਜੇ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_NO_SUPPORT;
    }

    ਅਚਲ uint32_t ਕੁੱਲ_ਪੰਕਤੀਆਂ = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ਅਚਲ size_t dst_ਪੰਕਤੀ_ਮਾਪ = dst->ne[0] * ਆਕਾਰ(int32_t);

    uint32_t ਪੰਕਤੀ_ਸ਼ੁਰੂ = 0;
    uint32_t ਪੰਕਤੀ_ਸਿਰਾ   = ਕੁੱਲ_ਪੰਕਤੀਆਂ;
    ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 0;
        htp_tensor_mdev_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk(dst, ਆਕਾਰ(int32_t), (uint32_t) dst_ਪੰਕਤੀ_ਮਾਪ, &ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk);
        ਅਚਲ ਬਣਤਰ htp_tensor_mdev_range range = htp_tensor_mdev_ਹਿੱਸੇ_ਕਰੋ(
            ਕੁੱਲ_ਪੰਕਤੀਆਂ, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ਗਿਣਤੀ_2, &octx->ctx->mdev.ਗਿਣਤੀ_div);
        ਪੰਕਤੀ_ਸ਼ੁਰੂ = range.ਸ਼ੁਰੂ;
        ਪੰਕਤੀ_ਸਿਰਾ   = range.ਸ਼ੁਰੂ + range.ਗਿਣਤੀ_2;
    }

    ਅਚਲ uint32_t nrows = ਪੰਕਤੀ_ਸਿਰਾ - ਪੰਕਤੀ_ਸ਼ੁਰੂ;
    ਜੇ (nrows == 0) {
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params kparams_local;
    ਅਚਲ ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params * kparams = (ਅਚਲ ਬਣਤਰ htp_ਛਾਂਟੋ_kernel_params *) octx->kernel_params;

    ਜੇ (kparams->n_ਧਾਗੇ == 0) {
        ਬਣਤਰ htp_ਛਾਂਟੋ_vtcm_layout layout;
        ਜੇ (!htp_ਛਾਂਟੋ_solve_layout(&layout, ne00, ਕੁੱਲ_ਪੰਕਤੀਆਂ, k, octx->n_ਧਾਗੇ, octx->ctx->vtcm_ਮਾਪ, is_ਸਿਖਰ_k)) {
            ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, ਆਕਾਰ(kparams_local));
        kparams_local.n_ਧਾਗੇ         = (int32_t) layout.n_ਧਾਗੇ;
        kparams_local.ਕੁੱਲ_ਪੰਕਤੀਆਂ        = (int32_t) ਕੁੱਲ_ਪੰਕਤੀਆਂ;
        kparams_local.ਪੰਕਤੀ_ਸ਼ੁਰੂ         = (int32_t) ਪੰਕਤੀ_ਸ਼ੁਰੂ;
        kparams_local.ਪੰਕਤੀ_ਸਿਰਾ           = (int32_t) ਪੰਕਤੀ_ਸਿਰਾ;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.ਕ੍ਰਮ             = is_ਸਿਖਰ_k ? GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_ਸਿਖਰ_k          = is_ਸਿਖਰ_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_ਮਾਪ         = (int32_t) layout.ਕੁੱਲ_ਬਾਈਟਾਂ;
        kparams_local.phase1_ਥਾਂ_ਮਾਪ  = (int32_t) layout.phase1_ਥਾਂ_ਮਾਪ;
        kparams_local.ਮਿਲਾਓ_ਮੁੱਲਾਂ_off  = (int32_t) layout.ਮਿਲਾਓ_ਮੁੱਲਾਂ_off;
        kparams_local.ਮਿਲਾਓ_indices_off = (int32_t) layout.ਮਿਲਾਓ_indices_off;
        kparams_local.ਮਿਲਾਓ_elems       = (int32_t) layout.ਮਿਲਾਓ_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } ਨਹੀਂਤਾਂ ਜੇ (octx->ctx->mdev.ਗਿਣਤੀ_2 > 1) {
        ਸਮ੍ਰਿਤੀ_ਨਕਲ(&kparams_local, kparams, ਆਕਾਰ(kparams_local));
        kparams_local.ਪੰਕਤੀ_ਸ਼ੁਰੂ = (int32_t) ਪੰਕਤੀ_ਸ਼ੁਰੂ;
        kparams_local.ਪੰਕਤੀ_ਸਿਰਾ   = (int32_t) ਪੰਕਤੀ_ਸਿਰਾ;
        kparams = &kparams_local;
    }

    ਜੇ (octx->ctx->vtcm_ਮਾਪ < (size_t) kparams->vtcm_ਮਾਪ) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_ਮਾਪ, octx->ctx->vtcm_ਮਾਪ);
        ਵਾਪਸ HTP_ਸਥਿਤੀ_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_ਅਧਾਰ = (uint8_t *) octx->ctx->vtcm_ਅਧਾਰ;

    ਜੇ (ਕੁੱਲ_ਪੰਕਤੀਆਂ == 1 && kparams->n_chunks > 1) {
        ਬਣਤਰ htp_ਛਾਂਟੋ_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_ਅਧਾਰ = vtcm_ਅਧਾਰ;
        memset(cctx.real_ਗਿਣਤੀ, 0, ਆਕਾਰ(cctx.real_ਗਿਣਤੀ));

        work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, htp_ਛਾਂਟੋ_chunk_ਕੰਮ, &cctx, (uint32_t) kparams->n_chunks);

        ਜੇ (is_ਸਿਖਰ_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            ਜੇ (k == 1 && ne00 >= 128*1024) {
                ਦਸ਼ਮਲਵ *   cand_vals = (ਦਸ਼ਮਲਵ *)   (vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_ਮੁੱਲਾਂ_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_indices_off);
                ਦਸ਼ਮਲਵ ਉੱਤਮ_val = cand_vals[0];
                int32_t ਉੱਤਮ_idx = cand_idxs[0];
                ਲਈ (int32_t c = 1; c < kparams->n_chunks; c++) {
                    ਜੇ (cand_vals[c] > ਉੱਤਮ_val) {
                        ਉੱਤਮ_val = cand_vals[c];
                        ਉੱਤਮ_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->ਅੰਕੜੇ;
                dst_ptr[0] = ਉੱਤਮ_idx;
                ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
            }

            uint32_t ਕੁੱਲ_candidates = (uint32_t) kparams->n_chunks * local_k;
            ਦਸ਼ਮਲਵ *   cand_vals = (ਦਸ਼ਮਲਵ *)   (vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_ਮੁੱਲਾਂ_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_indices_off);
            uint32_t ਮਿਲਾਓ_elems = (uint32_t) kparams->ਮਿਲਾਓ_elems;

            ਜੇ (ਮਿਲਾਓ_elems > ਕੁੱਲ_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + ਕੁੱਲ_candidates), -INFINITY, ਮਿਲਾਓ_elems - ਕੁੱਲ_candidates);
                ਲਈ (uint32_t j = ਕੁੱਲ_candidates; j < ਮਿਲਾਓ_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_ਛਾਂਟੋ_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, ਮਿਲਾਓ_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->ਅੰਕੜੇ, cand_idxs, k * ਆਕਾਰ(int32_t));
        } ਨਹੀਂਤਾਂ {
            bool asc = kparams->ਕ੍ਰਮ == GGML_ਛਾਂਟੋ_ਕ੍ਰਮ_ASC;
            ਜੇ (kparams->n_chunks == 8) {
                ਬਣਤਰ htp_argsort_ਮਿਲਾਓ_l1_ctx l1ctx = { &cctx, asc };
                work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, htp_argsort_ਮਿਲਾਓ_l1_ਕੰਮ, &l1ctx, 4);

                ਬਣਤਰ htp_argsort_ਮਿਲਾਓ_l2_ctx l2ctx = { &cctx, asc };
                work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, htp_argsort_ਮਿਲਾਓ_l2_ਕੰਮ, &l2ctx, 2);

                size_t ਥਾਂ_ਮਾਪ = (size_t) kparams->phase1_ਥਾਂ_ਮਾਪ;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ਅਚਲ ਦਸ਼ਮਲਵ *   val0 = (ਅਚਲ ਦਸ਼ਮਲਵ *) vtcm_ਅਧਾਰ;
                ਅਚਲ int32_t * idx0 = (ਅਚਲ int32_t *) (vtcm_ਅਧਾਰ + 4 * chunk_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਅਚਲ ਦਸ਼ਮਲਵ *   val1 = (ਅਚਲ ਦਸ਼ਮਲਵ *) (vtcm_ਅਧਾਰ + 4 * ਥਾਂ_ਮਾਪ);
                ਅਚਲ int32_t * idx1 = (ਅਚਲ int32_t *) (vtcm_ਅਧਾਰ + 4 * ਥਾਂ_ਮਾਪ + 4 * chunk_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

                uint32_t n01 = cctx.real_ਗਿਣਤੀ[0] + cctx.real_ਗਿਣਤੀ[1] + cctx.real_ਗਿਣਤੀ[2] + cctx.real_ਗਿਣਤੀ[3];
                uint32_t n23 = cctx.real_ਗਿਣਤੀ[4] + cctx.real_ਗਿਣਤੀ[5] + cctx.real_ਗਿਣਤੀ[6] + cctx.real_ਗਿਣਤੀ[7];

                int32_t * final_idx = (int32_t *) (vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_indices_off);
                ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ਮਿਲਾਓ_runs(val0, idx0, n01, val1, idx1, n23, ਨਿਰਾ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->ਅੰਕੜੇ, final_idx, ne00 * ਆਕਾਰ(int32_t));
            } ਨਹੀਂਤਾਂ ਜੇ (kparams->n_chunks == 4) {
                ਬਣਤਰ htp_argsort_ਮਿਲਾਓ_l1_ctx l1ctx = { &cctx, asc };
                work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, htp_argsort_ਮਿਲਾਓ_l1_ਕੰਮ, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_ਗਿਣਤੀ[0] + cctx.real_ਗਿਣਤੀ[1];
                uint32_t n23 = cctx.real_ਗਿਣਤੀ[2] + cctx.real_ਗਿਣਤੀ[3];

                ਅਚਲ ਦਸ਼ਮਲਵ *   val01 = (ਅਚਲ ਦਸ਼ਮਲਵ *)   (vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_ਮੁੱਲਾਂ_off);
                ਅਚਲ int32_t * idx01 = (ਅਚਲ int32_t *) (vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_indices_off);
                ਅਚਲ ਦਸ਼ਮਲਵ *   val23 = (ਅਚਲ ਦਸ਼ਮਲਵ *)   (vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_ਮੁੱਲਾਂ_off)  + 2 * chunk_elems;
                ਅਚਲ int32_t * idx23 = (ਅਚਲ int32_t *) (vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_ਅਧਾਰ;
                ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ਮਿਲਾਓ_runs(val01, idx01, n01, val23, idx23, n23, ਨਿਰਾ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->ਅੰਕੜੇ, final_idx, ne00 * ਆਕਾਰ(int32_t));
            } ਨਹੀਂਤਾਂ ਜੇ (kparams->n_chunks == 2) {
                size_t ਥਾਂ_ਮਾਪ = (size_t) kparams->phase1_ਥਾਂ_ਮਾਪ;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ਅਚਲ ਦਸ਼ਮਲਵ *   val0 = (ਅਚਲ ਦਸ਼ਮਲਵ *)   (vtcm_ਅਧਾਰ);
                ਅਚਲ int32_t * idx0 = (ਅਚਲ int32_t *) (vtcm_ਅਧਾਰ + chunk_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                ਅਚਲ ਦਸ਼ਮਲਵ *   val1 = (ਅਚਲ ਦਸ਼ਮਲਵ *)   (vtcm_ਅਧਾਰ + ਥਾਂ_ਮਾਪ);
                ਅਚਲ int32_t * idx1 = (ਅਚਲ int32_t *) (vtcm_ਅਧਾਰ + ਥਾਂ_ਮਾਪ + chunk_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ));

                int32_t * final_idx = (int32_t *) (vtcm_ਅਧਾਰ + kparams->ਮਿਲਾਓ_indices_off);
                ਬਣਤਰ htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_ਸ਼ੁਰੂ(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ਮਿਲਾਓ_runs(val0, idx0, cctx.real_ਗਿਣਤੀ[0], val1, idx1, cctx.real_ਗਿਣਤੀ[1], ਨਿਰਾ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->ਅੰਕੜੇ, final_idx, ne00 * ਆਕਾਰ(int32_t));
            } ਨਹੀਂਤਾਂ {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_ਅਧਾਰ + chunk_elems * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
                dma_sync_write(octx->ctx->dma[0], dst->ਅੰਕੜੇ, final_idx, ne00 * ਆਕਾਰ(int32_t));
            }
        }
        ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
    }

    ਬਣਤਰ htp_ਛਾਂਟੋ_multi_ਪੰਕਤੀ_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_ਅਧਾਰ        = vtcm_ਅਧਾਰ;

    uint32_t dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ = (uint32_t) dst_ਪੰਕਤੀ_ਮਾਪ;
    uint32_t ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 0;
    htp_tensor_mdev_ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk(dst, ਆਕਾਰ(int32_t), dst_ਪੰਕਤੀ_ਬਾਈਟਾਂ, &ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk);
    ਜੇ (ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk == 0) {
        ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_ਧਾਗੇ - 1) / (uint32_t) kparams->n_ਧਾਗੇ;
    ਜੇ (ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk > 1) {
        dr = hex_ਗੇੜਾ_ਉੱਪਰ(dr, ਪੰਕਤੀਆਂ_ਪ੍ਰਤੀ_chunk);
    }
    mctx.nrows_ਪ੍ਰਤੀ_thread = dr;

    work_ਕਤਾਰ_ਲੜੀਵਾਰ(octx->ctx->work_ਕਤਾਰ, htp_ਛਾਂਟੋ_multi_ਪੰਕਤੀ_ਕੰਮ, &mctx, (uint32_t) kparams->n_ਧਾਗੇ);

    ਵਾਪਸ HTP_ਸਥਿਤੀ_OK;
}

ਪੂਰਨ op_argsort(ਬਣਤਰ htp_ops_context * octx) {
    ਵਾਪਸ op_ਛਾਂਟੋ_common(octx, false);
}

ਪੂਰਨ op_ਸਿਖਰ_k(ਬਣਤਰ htp_ops_context * octx) {
    ਵਾਪਸ op_ਛਾਂਟੋ_common(octx, true);
}
