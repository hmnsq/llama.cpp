#समावेश <string.h>
#समावेश <stdlib.h>
#समावेश <math.h>
#समावेश <HAP_farf.h>
#समावेश <HAP_perf.h>

#घोषणा GGML_COMMON_DECL_C
#समावेश "ggml-common.h"
#समावेश "ggml.h"

#समावेश "hvx-utils.h"
#समावेश "hvx-copy.h"
#समावेश "hvx-reduce.h"
#समावेश "dma-queue.h"

#समावेश "hex-common.h"
#समावेश "htp-ctx.h"
#समावेश "htp-ops.h"
#समावेश "htp-tensor.h"
#समावेश "argsort-ops.h"

#अगर_अघोषित MIN
#घोषणा MIN(a, b) ((a) < (b) ? (a) : (b))
#पूर्ण_अगर


अटल व्योम quicksort_मूल्यावली_indices_asc(भग्न * मूल्यावली, int32_t * indices, पूर्णांक बायाँ, पूर्णांक दायाँ) {
    जबतक (बायाँ < दायाँ) {
        भग्न pivot = मूल्यावली[बायाँ + (दायाँ - बायाँ) / 2];
        पूर्णांक i = बायाँ;
        पूर्णांक j = दायाँ;

        जबतक (i <= j) {
            जबतक (मूल्यावली[i] < pivot) i++;
            जबतक (मूल्यावली[j] > pivot) j--;

            अगर (i <= j) {
                भग्न tmp_val = मूल्यावली[i];
                मूल्यावली[i] = मूल्यावली[j];
                मूल्यावली[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        अगर (j - बायाँ < दायाँ - i) {
            अगर (बायाँ < j) quicksort_मूल्यावली_indices_asc(मूल्यावली, indices, बायाँ, j);
            बायाँ = i;
        } अन्यथा {
            अगर (i < दायाँ) quicksort_मूल्यावली_indices_asc(मूल्यावली, indices, i, दायाँ);
            दायाँ = j;
        }
    }
}

अटल व्योम quicksort_मूल्यावली_indices_desc(भग्न * मूल्यावली, int32_t * indices, पूर्णांक बायाँ, पूर्णांक दायाँ) {
    जबतक (बायाँ < दायाँ) {
        भग्न pivot = मूल्यावली[बायाँ + (दायाँ - बायाँ) / 2];
        पूर्णांक i = बायाँ;
        पूर्णांक j = दायाँ;

        जबतक (i <= j) {
            जबतक (मूल्यावली[i] > pivot) i++;
            जबतक (मूल्यावली[j] < pivot) j--;

            अगर (i <= j) {
                भग्न tmp_val = मूल्यावली[i];
                मूल्यावली[i] = मूल्यावली[j];
                मूल्यावली[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        अगर (j - बायाँ < दायाँ - i) {
            अगर (बायाँ < j) quicksort_मूल्यावली_indices_desc(मूल्यावली, indices, बायाँ, j);
            बायाँ = i;
        } अन्यथा {
            अगर (i < दायाँ) quicksort_मूल्यावली_indices_desc(मूल्यावली, indices, i, दायाँ);
            दायाँ = j;
        }
    }
}

अटल अंतरभूत uint32_t शीर्ष_k_max_मूल्य_सूचकांक(स्थिर भग्न * मूल्यावली, uint32_t n, भग्न * मूल्य) {
    int32_t max_idx = 0;
    hvx_argmax_f32(मूल्यावली, n, 0, मूल्य, &max_idx);
    वापस (uint32_t) max_idx;
}

अटल स्थिर int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

अटल अंतरभूत व्योम init_indices_ramp(int32_t * indices_buf, uint32_t गिनती, uint32_t base) {
    स्थिर HVX_Vector ind_init_vec = Q6_Vw_vadd_VwVw(*(स्थिर HVX_Vector *) argsort_ramp_lut, Q6_V_vsplat_R(base));
    स्थिर HVX_Vector ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_Vector * indices_buf_vec = (HVX_Vector *) indices_buf;
    uint32_t अंक_vecs = गिनती / 32;
    HVX_Vector curr_ind_vec = ind_init_vec;
    क्रम (uint32_t j = 0; j < अंक_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = गिनती % 32;
    अगर (rem > 0) {
        hvx_vec_store_a((व्योम *) &indices_buf_vec[अंक_vecs], rem * माप(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
अटल अंतरभूत व्योम vec_cas(HVX_Vector * X_val, HVX_Vector * X_idx, HVX_Vector * Y_val, HVX_Vector * Y_idx, bool asc) {
    HVX_VectorPred pred = asc ? Q6_Q_vcmp_gt_VsfVsf(*X_val, *Y_val)
                              : Q6_Q_vcmp_gt_VsfVsf(*Y_val, *X_val);
    HVX_Vector अग्र_X_val = Q6_V_vmux_QVV(pred, *Y_val, *X_val);
    HVX_Vector अग्र_Y_val = Q6_V_vmux_QVV(pred, *X_val, *Y_val);
    HVX_Vector अग्र_X_idx = Q6_V_vmux_QVV(pred, *Y_idx, *X_idx);
    HVX_Vector Y_tmp_idx  = Q6_V_vmux_QVV(pred, *X_idx, *Y_idx);
    *X_val = अग्र_X_val;
    *Y_val = अग्र_Y_val;
    *X_idx = अग्र_X_idx;
    *Y_idx = Y_tmp_idx;
}

__attribute__((always_inline))
अटल अंतरभूत व्योम bitonic_cas_32(HVX_Vector * V, HVX_Vector * I, पूर्णांक d, HVX_VectorPred dir_mask, HVX_Vector idx_vec, HVX_Vector zero_vec) {
    HVX_VectorPred mask_बायाँ;
    HVX_Vector V_rot_बायाँ, V_rot_दायाँ;
    HVX_Vector I_rot_बायाँ, I_rot_दायाँ;

    अगर (d == 1) {
        mask_बायाँ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_बायाँ = Q6_V_vror_VR(*V, 4);
        V_rot_दायाँ = Q6_V_vror_VR(*V, 124);
        I_rot_बायाँ = Q6_V_vror_VR(*I, 4);
        I_rot_दायाँ = Q6_V_vror_VR(*I, 124);
    } अन्यथा अगर (d == 2) {
        mask_बायाँ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_बायाँ = Q6_V_vror_VR(*V, 8);
        V_rot_दायाँ = Q6_V_vror_VR(*V, 120);
        I_rot_बायाँ = Q6_V_vror_VR(*I, 8);
        I_rot_दायाँ = Q6_V_vror_VR(*I, 120);
    } अन्यथा अगर (d == 4) {
        mask_बायाँ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_बायाँ = Q6_V_vror_VR(*V, 16);
        V_rot_दायाँ = Q6_V_vror_VR(*V, 112);
        I_rot_बायाँ = Q6_V_vror_VR(*I, 16);
        I_rot_दायाँ = Q6_V_vror_VR(*I, 112);
    } अन्यथा अगर (d == 8) {
        mask_बायाँ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_बायाँ = Q6_V_vror_VR(*V, 32);
        V_rot_दायाँ = Q6_V_vror_VR(*V, 96);
        I_rot_बायाँ = Q6_V_vror_VR(*I, 32);
        I_rot_दायाँ = Q6_V_vror_VR(*I, 96);
    } अन्यथा {
        mask_बायाँ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_बायाँ = Q6_V_vror_VR(*V, 64);
        V_rot_दायाँ = Q6_V_vror_VR(*V, 64);
        I_rot_बायाँ = Q6_V_vror_VR(*I, 64);
        I_rot_दायाँ = Q6_V_vror_VR(*I, 64);
    }

    HVX_Vector V_paired = Q6_V_vmux_QVV(mask_बायाँ, V_rot_बायाँ, V_rot_दायाँ);
    HVX_Vector I_paired = Q6_V_vmux_QVV(mask_बायाँ, I_rot_बायाँ, I_rot_दायाँ);

    HVX_VectorPred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_VectorPred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_VectorPred mask_दायाँ = Q6_Q_not_Q(mask_बायाँ);
    HVX_VectorPred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(mask_बायाँ, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, mask_दायाँ)
    );
    HVX_VectorPred Q_swap = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_mask, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_mask), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_swap, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_swap, I_paired, *I);
}

__attribute__((always_inline))
अटल अंतरभूत व्योम bitonic_sort_generic_hvx(uint8_t * मूल्यावली, uint8_t * indices, पूर्णांक K, bool asc_order) {
    HVX_Vector V[32];
    HVX_Vector I[32];

    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(स्थिर HVX_Vector *) argsort_ramp_lut;

    क्रम (पूर्णांक v = 0; v < K; v++) {
        V[v] = *(HVX_Vector *)(मूल्यावली + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    पूर्णांक M = 5;
    जबतक ((1 << (M - 5)) < K) M++;

    क्रम (पूर्णांक s = 1; s <= M; s++) {
        क्रम (पूर्णांक stage_d = s - 1; stage_d >= 0; stage_d--) {
            पूर्णांक d = 1 << stage_d;
            अगर (d >= 32) {
                पूर्णांक v_dist = d / 32;
                क्रम (पूर्णांक v1 = 0; v1 < K; v1++) {
                    अगर ((v1 & v_dist) == 0) {
                        पूर्णांक v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_order;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } अन्यथा {
                अगर (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    क्रम (पूर्णांक v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                } अन्यथा {
                    क्रम (पूर्णांक v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_order;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    क्रम (पूर्णांक v = 0; v < K; v++) {
        *(HVX_Vector *)(मूल्यावली + v * 128)  = V[v];
        *(HVX_Vector *)(indices + v * 128) = I[v];
    }
}

अटल व्योम bitonic_sort_vtcm_desc(uint8_t * मूल्यावली, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(स्थिर HVX_Vector *) argsort_ramp_lut;

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    अगर (init_indices) {
        क्रम (uint32_t v = 0; v < n_vec; v++) {
            HVX_Vector idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_Vector *)(indices + v * 128) = idx;
        }
    }

    पूर्णांक M = 5;
    जबतक ((1u << (M - 5)) < n_vec) M++;

    क्रम (पूर्णांक s = 1; s <= M; s++) {
        क्रम (पूर्णांक stage_d = s - 1; stage_d >= 0; stage_d--) {
            पूर्णांक d = 1 << stage_d;
            अगर (d >= 32) {
                uint32_t v_dist = d / 32;
                क्रम (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    अगर ((v1 & v_dist) == 0) {
                        uint32_t v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : false;

                        HVX_Vector Vv1 = *(HVX_Vector *)(मूल्यावली + v1 * 128);
                        HVX_Vector Iv1 = *(HVX_Vector *)(indices + v1 * 128);
                        HVX_Vector Vv2 = *(HVX_Vector *)(मूल्यावली + v2 * 128);
                        HVX_Vector Iv2 = *(HVX_Vector *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_Vector *)(मूल्यावली + v1 * 128)  = Vv1;
                        *(HVX_Vector *)(indices + v1 * 128) = Iv1;
                        *(HVX_Vector *)(मूल्यावली + v2 * 128)  = Vv2;
                        *(HVX_Vector *)(indices + v2 * 128) = Iv2;
                    }
                }
            } अन्यथा {
                अगर (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    क्रम (uint32_t v = 0; v < n_vec; v++) {
                        HVX_Vector Vv = *(HVX_Vector *)(मूल्यावली + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(मूल्यावली + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                } अन्यथा {
                    क्रम (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;

                        HVX_Vector Vv = *(HVX_Vector *)(मूल्यावली + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(मूल्यावली + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

अटल व्योम शीर्ष_k_select_tiled(dma_queue * dma_q, काष्ठा htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               भग्न * मूल्यावली_buf, int32_t * indices_buf,
                               भग्न * out_मूल्यावली, int32_t * out_indices,
                               uint32_t chunk_base) {
    स्थिर uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_गिनती    = n_tiles * k;
    uint32_t merge_n_vec        = hmx_ceil_div(candidate_गिनती, 32);
    uint32_t merge_n_vec_pow2   = 1;
    जबतक (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
    uint32_t merge_elems        = merge_n_vec_pow2 * 32;
    भग्न * candidate_मूल्यावली    = मूल्यावली_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    क्रम (uint32_t offset = 0; offset < n; offset += tile_elems) {
        uint32_t tile_गिनती = MIN(tile_elems, n - offset);
        dma_sync_read(dma_q, मूल्यावली_buf, src_addr + (माप_प्रकार) offset * माप(भग्न), tile_गिनती * माप(भग्न));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
        अगर (tile_गिनती < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (मूल्यावली_buf + tile_गिनती), -अनन्त, tile_elems - tile_गिनती);
        }

        bitonic_sort_vtcm_desc((uint8_t *) मूल्यावली_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_गिनती);
        क्रम (uint32_t j = 0; j < tile_k; j++) {
            candidate_मूल्यावली[candidate_pos] = मूल्यावली_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) offset;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    अगर (merge_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_मूल्यावली + candidate_pos), -अनन्त, merge_elems - candidate_pos);
        क्रम (uint32_t j = candidate_pos; j < merge_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_sort_vtcm_desc((uint8_t *) candidate_मूल्यावली, (uint8_t *) candidate_indices, merge_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    क्रम (uint32_t j = 0; j < k; j++) {
        out_मूल्यावली[j]  = candidate_मूल्यावली[j];
        out_indices[j] = candidate_indices[j] + (int32_t) chunk_base;
    }
}

__attribute__((always_inline))
अटल अंतरभूत व्योम sort32_f32_hvx(uint8_t * मूल्यावली, uint8_t * indices, क्रमागत ggml_sort_order order) {
    bitonic_sort_generic_hvx(मूल्यावली, indices, 1, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
अटल अंतरभूत व्योम sort64_f32_hvx(uint8_t * मूल्यावली, uint8_t * indices, क्रमागत ggml_sort_order order) {
    bitonic_sort_generic_hvx(मूल्यावली, indices, 2, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
अटल अंतरभूत व्योम sort128_f32_hvx(uint8_t * मूल्यावली, uint8_t * indices, क्रमागत ggml_sort_order order) {
    bitonic_sort_generic_hvx(मूल्यावली, indices, 4, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
अटल अंतरभूत व्योम sort256_f32_hvx(uint8_t * मूल्यावली, uint8_t * indices, क्रमागत ggml_sort_order order) {
    bitonic_sort_generic_hvx(मूल्यावली, indices, 8, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
अटल अंतरभूत व्योम sort512_f32_hvx(uint8_t * मूल्यावली, uint8_t * indices, क्रमागत ggml_sort_order order) {
    bitonic_sort_generic_hvx(मूल्यावली, indices, 16, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
अटल अंतरभूत व्योम sort1024_f32_hvx(uint8_t * मूल्यावली, uint8_t * indices, क्रमागत ggml_sort_order order) {
    bitonic_sort_generic_hvx(मूल्यावली, indices, 32, order == GGML_SORT_ORDER_ASC);
}

अटल व्योम merge_runs(
    स्थिर भग्न * restrict in_val0, स्थिर int32_t * restrict in_idx0, uint32_t n0,
    स्थिर भग्न * restrict in_val1, स्थिर int32_t * restrict in_idx1, uint32_t n1,
    भग्न * restrict out_val, int32_t * restrict out_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t out = 0;

    अगर (asc) {
        जबतक (i0 < n0 && i1 < n1) {
            अगर (in_val0[i0] <= in_val1[i1]) {
                अगर (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } अन्यथा {
                अगर (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    } अन्यथा {
        जबतक (i0 < n0 && i1 < n1) {
            अगर (in_val0[i0] >= in_val1[i1]) {
                अगर (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } अन्यथा {
                अगर (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    }

    जबतक (i0 < n0) {
        अगर (out_val) out_val[out] = in_val0[i0];
        out_idx[out] = in_idx0[i0];
        i0++;
        out++;
    }

    जबतक (i1 < n1) {
        अगर (out_val) out_val[out] = in_val1[i1];
        out_idx[out] = in_idx1[i1];
        i1++;
        out++;
    }
}

काष्ठा htp_sort_chunk_ctx {
    काष्ठा htp_ops_context *              octx;
    स्थिर काष्ठा htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              real_गिनती[HTP_MAX_NTHREADS];
};

अटल व्योम htp_sort_chunk_job(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, व्योम * तथ्य) {
    काष्ठा htp_sort_chunk_ctx * cctx = (काष्ठा htp_sort_chunk_ctx *) तथ्य;
    काष्ठा htp_ops_context * octx = cctx->octx;
    स्थिर काष्ठा htp_sort_kernel_params * kparams = cctx->kparams;
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_base  = i * chunk_elems;

    uint32_t real_गिनती = (chunk_base < ne00) ? MIN(chunk_elems, ne00 - chunk_base) : 0;
    cctx->real_गिनती[i] = real_गिनती;

    uint8_t * spad = cctx->vtcm_base + (माप_प्रकार) kparams->phase1_slot_आकार * i;
    भग्न *   मूल्यावली_buf  = (भग्न *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * माप(भग्न));

    भग्न *   merge_मूल्यावली  = (भग्न *)   (cctx->vtcm_base + kparams->merge_मूल्यावली_off);
    int32_t * merge_indices = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off);

    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[i];
    स्थिर dma_addr_t src_addr = src0->तथ्य + (माप_प्रकार) chunk_base * माप(भग्न);

    अगर (kparams->is_शीर्ष_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        अगर (real_गिनती == 0) {
            क्रम (uint32_t j = 0; j < local_k; j++) {
                merge_मूल्यावली[i * local_k + j]  = -अनन्त;
                merge_indices[i * local_k + j] = 0;
            }
            वापस;
        }

        अगर (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], मूल्यावली_buf, src_addr, real_गिनती * माप(भग्न));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            भग्न max_मूल्य;
            uint32_t max_सूचकांक = शीर्ष_k_max_मूल्य_सूचकांक(मूल्यावली_buf, real_गिनती, &max_मूल्य);
            merge_मूल्यावली[i] = max_मूल्य;
            merge_indices[i] = (int32_t) (max_सूचकांक + chunk_base);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            वापस;
        }

        अगर (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            शीर्ष_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_गिनती, local_k,
                               मूल्यावली_buf, indices_buf,
                               &merge_मूल्यावली[i * local_k], &merge_indices[i * local_k],
                               chunk_base);
            वापस;
        }

        dma_sync_read(octx->ctx->dma[i], मूल्यावली_buf, src_addr, real_गिनती * माप(भग्न));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        अगर (chunk_elems > real_गिनती) {
            hvx_splat_f32_u((uint8_t *)(मूल्यावली_buf + real_गिनती), -अनन्त, chunk_elems - real_गिनती);
        }

        bitonic_sort_vtcm_desc((uint8_t *) मूल्यावली_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        क्रम (uint32_t j = 0; j < local_k; j++) {
            merge_मूल्यावली[i * local_k + j]  = मूल्यावली_buf[j];
            merge_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_base;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } अन्यथा {
        अगर (real_गिनती > 0) {
            dma_sync_read(octx->ctx->dma[i], मूल्यावली_buf, src_addr, real_गिनती * माप(भग्न));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_गिनती, chunk_base);

            अगर (kparams->order == GGML_SORT_ORDER_ASC) {
                quicksort_मूल्यावली_indices_asc(मूल्यावली_buf, indices_buf, 0, real_गिनती - 1);
            } अन्यथा {
                quicksort_मूल्यावली_indices_desc(मूल्यावली_buf, indices_buf, 0, real_गिनती - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

काष्ठा htp_argsort_merge_l1_ctx {
    काष्ठा htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

अटल व्योम htp_argsort_merge_l1_job(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, व्योम * तथ्य) {
    काष्ठा htp_argsort_merge_l1_ctx * mctx = (काष्ठा htp_argsort_merge_l1_ctx *) तथ्य;
    काष्ठा htp_sort_chunk_ctx * cctx = mctx->cctx;
    स्थिर काष्ठा htp_sort_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_गिनती[c0];
    uint32_t count1 = cctx->real_गिनती[c1];

    माप_प्रकार slot_आकार = (माप_प्रकार) kparams->phase1_slot_आकार;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    स्थिर भग्न *   val0 = (स्थिर भग्न *)   (cctx->vtcm_base + slot_आकार * c0);
    स्थिर int32_t * idx0 = (स्थिर int32_t *) (cctx->vtcm_base + slot_आकार * c0 + chunk_elems * माप(भग्न));
    स्थिर भग्न *   val1 = (स्थिर भग्न *)   (cctx->vtcm_base + slot_आकार * c1);
    स्थिर int32_t * idx1 = (स्थिर int32_t *) (cctx->vtcm_base + slot_आकार * c1 + chunk_elems * माप(भग्न));

    uint32_t out_offset = (2 * i) * chunk_elems;
    भग्न *   out_val = (भग्न *)   (cctx->vtcm_base + kparams->merge_मूल्यावली_off)  + out_offset;
    int32_t * out_idx = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + out_offset;

    काष्ठा htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

काष्ठा htp_argsort_merge_l2_ctx {
    काष्ठा htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

अटल व्योम htp_argsort_merge_l2_job(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक i, व्योम * तथ्य) {
    काष्ठा htp_argsort_merge_l2_ctx * mctx = (काष्ठा htp_argsort_merge_l2_ctx *) तथ्य;
    काष्ठा htp_sort_chunk_ctx * cctx = mctx->cctx;
    स्थिर काष्ठा htp_sort_kernel_params * kparams = cctx->kparams;

    माप_प्रकार slot_आकार = (माप_प्रकार) kparams->phase1_slot_आकार;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    स्थिर भग्न *   val0 = (स्थिर भग्न *)   (cctx->vtcm_base + kparams->merge_मूल्यावली_off)  + off0;
    स्थिर int32_t * idx0 = (स्थिर int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off0;
    स्थिर भग्न *   val1 = (स्थिर भग्न *)   (cctx->vtcm_base + kparams->merge_मूल्यावली_off)  + off1;
    स्थिर int32_t * idx1 = (स्थिर int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off1;

    uint32_t count0 = cctx->real_गिनती[4*i + 0] + cctx->real_गिनती[4*i + 1];
    uint32_t count1 = cctx->real_गिनती[4*i + 2] + cctx->real_गिनती[4*i + 3];

    uint8_t * slot_base = cctx->vtcm_base + (4 * i) * slot_आकार;
    भग्न *   out_val   = (भग्न *) slot_base;
    int32_t * out_idx   = (int32_t *) (slot_base + 4 * chunk_elems * माप(भग्न));

    काष्ठा htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

काष्ठा htp_sort_multi_row_ctx {
    काष्ठा htp_ops_context *              octx;
    स्थिर काष्ठा htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              nrows_per_thread;
};

काष्ठा htp_tensor_row_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

अटल अंतरभूत काष्ठा htp_tensor_row_coord init_row_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    वापस (काष्ठा htp_tensor_row_coord){ i01, i02, i03 };
}

अटल अंतरभूत व्योम step_row_coord(काष्ठा htp_tensor_row_coord * c, uint32_t ne01, uint32_t ne02) {
    c->i01++;
    अगर (c->i01 == ne01) {
        c->i01 = 0;
        c->i02++;
        अगर (c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

अटल अंतरभूत माप_प्रकार coord_offset(काष्ठा htp_tensor_row_coord c, माप_प्रकार nb1, माप_प्रकार nb2, माप_प्रकार nb3) {
    वापस (माप_प्रकार) c.i01 * nb1 + (माप_प्रकार) c.i02 * nb2 + (माप_प्रकार) c.i03 * nb3;
}

अटल अंतरभूत व्योम compute_row_sort(
    भग्न * cur_मूल्यावली,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_शीर्ष_k,
    क्रमागत ggml_sort_order order
) {
    अगर (is_शीर्ष_k) {
        अगर (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            चयन (ne00) {
                हाल 32:   sort32_f32_hvx((uint8_t *) cur_मूल्यावली, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); अवरोध;
                हाल 64:   sort64_f32_hvx((uint8_t *) cur_मूल्यावली, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); अवरोध;
                हाल 128:  sort128_f32_hvx((uint8_t *) cur_मूल्यावली, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); अवरोध;
                हाल 256:  sort256_f32_hvx((uint8_t *) cur_मूल्यावली, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); अवरोध;
                हाल 512:  sort512_f32_hvx((uint8_t *) cur_मूल्यावली, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); अवरोध;
                हाल 1024: sort1024_f32_hvx((uint8_t *) cur_मूल्यावली, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); अवरोध;
            }
        } अन्यथा {
            अगर (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_मूल्यावली + ne00), -अनन्त, chunk_elems - ne00);
            }
            bitonic_sort_vtcm_desc((uint8_t *) cur_मूल्यावली, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } अन्यथा {
        अगर (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            चयन (ne00) {
                हाल 32:   sort32_f32_hvx((uint8_t *) cur_मूल्यावली, (uint8_t *) cur_indices, order); अवरोध;
                हाल 64:   sort64_f32_hvx((uint8_t *) cur_मूल्यावली, (uint8_t *) cur_indices, order); अवरोध;
                हाल 128:  sort128_f32_hvx((uint8_t *) cur_मूल्यावली, (uint8_t *) cur_indices, order); अवरोध;
                हाल 256:  sort256_f32_hvx((uint8_t *) cur_मूल्यावली, (uint8_t *) cur_indices, order); अवरोध;
                हाल 512:  sort512_f32_hvx((uint8_t *) cur_मूल्यावली, (uint8_t *) cur_indices, order); अवरोध;
                हाल 1024: sort1024_f32_hvx((uint8_t *) cur_मूल्यावली, (uint8_t *) cur_indices, order); अवरोध;
            }
        } अन्यथा {
            init_indices_ramp(cur_indices, ne00, 0);
            अगर (order == GGML_SORT_ORDER_ASC) {
                quicksort_मूल्यावली_indices_asc(cur_मूल्यावली, cur_indices, 0, ne00 - 1);
            } अन्यथा {
                quicksort_मूल्यावली_indices_desc(cur_मूल्यावली, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

अटल व्योम htp_sort_multi_row_job(अचिन्हित पूर्णांक n, अचिन्हित पूर्णांक ith, व्योम * तथ्य) {
    काष्ठा htp_sort_multi_row_ctx * mctx = (काष्ठा htp_sort_multi_row_ctx *) तथ्य;
    काष्ठा htp_ops_context * octx = mctx->octx;
    स्थिर काष्ठा htp_sort_kernel_params * kparams = mctx->kparams;
    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

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

    अगर (r_start >= r_end) वापस;

    माप_प्रकार spad_offset = ith * (माप_प्रकार) (kparams->phase1_slot_आकार * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_base + spad_offset;
    माप_प्रकार slot_आकार = (माप_प्रकार) kparams->phase1_slot_आकार;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    भग्न *   मूल्यावली_buf  = (भग्न *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * माप(भग्न));

    dma_queue * dma_q = octx->ctx->dma[ith];
    काष्ठा htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_शीर्ष_k = kparams->is_शीर्ष_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    क्रमागत ggml_sort_order order = (क्रमागत ggml_sort_order) kparams->order;

    स्थिर uint32_t src_bytes = ne00 * माप(भग्न);
    स्थिर uint32_t dst_bytes = (is_शीर्ष_k ? k : ne00) * माप(int32_t);

    अगर (kparams->n_slots == 2) {
        काष्ठा htp_tensor_row_coord अभाज्य_coord = init_row_coord(r_start, ne01, ne02);
        क्रम (uint32_t r = r_start, spad_idx = 0; r < r_end && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * slot_आकार;
            भग्न *   cur_मूल्यावली  = (भग्न *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * माप(भग्न));

            dma_addr_t dst_addr = dst->तथ्य + coord_offset(अभाज्य_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->तथ्य + coord_offset(अभाज्य_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 0);

            // Prefetch input row
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(cur_मूल्यावली, src_addr),
                           src_bytes, src_bytes, src_bytes, 1);

            step_row_coord(&अभाज्य_coord, ne01, ne02);
        }

        काष्ठा htp_tensor_row_coord dst_coord = init_row_coord(r_start, ne01, ne02);
        काष्ठा htp_tensor_row_coord अग्र_src_coord = init_row_coord(r_start + 2, ne01, ne02);

        क्रम (uint32_t r = r_start; r < r_end; r++) {
            uint32_t cur_slot = (r - r_start) & 1;
            uint8_t * cur_spad = thread_spad + cur_slot * slot_आकार;
            भग्न *   cur_मूल्यावली  = (भग्न *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * माप(भग्न));

            // Wait for previous writeback of this slot to complete
            dma_queue_निकालो(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_queue_निकालो(dma_q);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(cur_मूल्यावली, cur_indices, ne00, chunk_elems, is_शीर्ष_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->तथ्य + coord_offset(dst_coord, nb1, nb2, nb3);
            dma_queue_धकेलो(dma_q, dma_make_तथ्य(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 1);
            step_row_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            स्थिर uint32_t अग्र_row = r + 2;
            अगर (अग्र_row < r_end) {
                dma_addr_t अग्र_src_addr = src0->तथ्य + coord_offset(अग्र_src_coord, nb01, nb02, nb03);
                dma_queue_धकेलो(dma_q, dma_make_तथ्य(cur_मूल्यावली, अग्र_src_addr),
                               src_bytes, src_bytes, src_bytes, 1);
                step_row_coord(&अग्र_src_coord, ne01, ne02);
            }
        }

        dma_queue_flush(dma_q);
    } अन्यथा {
        काष्ठा htp_tensor_row_coord coord = init_row_coord(r_start, ne01, ne02);
        क्रम (uint32_t r = r_start; r < r_end; r++) {
            dma_addr_t src_addr = src0->तथ्य + coord_offset(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->तथ्य + coord_offset(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, मूल्यावली_buf, src_addr, src_bytes);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(मूल्यावली_buf, indices_buf, ne00, chunk_elems, is_शीर्ष_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_bytes);
            step_row_coord(&coord, ne01, ne02);
        }
    }
}

अटल पूर्णांक op_sort_common(काष्ठा htp_ops_context * octx, bool is_शीर्ष_k) {
    अगर (octx->src[0]->type != HTP_TYPE_F32) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    स्थिर काष्ठा htp_tensor * src0 = octx->src[0];
    स्थिर काष्ठा htp_tensor * dst  = octx->dst;

    अगर (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        वापस HTP_STATUS_NO_SUPPORT;
    }

    स्थिर uint32_t योग_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    स्थिर माप_प्रकार dst_row_आकार = dst->ne[0] * माप(int32_t);

    uint32_t row_start = 0;
    uint32_t row_end   = योग_rows;
    अगर (octx->ctx->mdev.गिनती > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, माप(int32_t), (uint32_t) dst_row_आकार, &rows_per_chunk);
        स्थिर काष्ठा htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            योग_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गिनती, &octx->ctx->mdev.गिनती_div);
        row_start = range.start;
        row_end   = range.start + range.गिनती;
    }

    स्थिर uint32_t nrows = row_end - row_start;
    अगर (nrows == 0) {
        वापस HTP_STATUS_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    काष्ठा htp_sort_kernel_params kparams_local;
    स्थिर काष्ठा htp_sort_kernel_params * kparams = (स्थिर काष्ठा htp_sort_kernel_params *) octx->kernel_params;

    अगर (kparams->n_threads == 0) {
        काष्ठा htp_sort_vtcm_layout layout;
        अगर (!htp_sort_solve_layout(&layout, ne00, योग_rows, k, octx->n_threads, octx->ctx->vtcm_आकार, is_शीर्ष_k)) {
            वापस HTP_STATUS_VTCM_TOO_SMALL;
        }
        स_रखो(&kparams_local, 0, माप(kparams_local));
        kparams_local.n_threads         = (int32_t) layout.n_threads;
        kparams_local.योग_rows        = (int32_t) योग_rows;
        kparams_local.row_start         = (int32_t) row_start;
        kparams_local.row_end           = (int32_t) row_end;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.order             = is_शीर्ष_k ? GGML_SORT_ORDER_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_शीर्ष_k          = is_शीर्ष_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_आकार         = (int32_t) layout.योग_bytes;
        kparams_local.phase1_slot_आकार  = (int32_t) layout.phase1_slot_आकार;
        kparams_local.merge_मूल्यावली_off  = (int32_t) layout.merge_मूल्यावली_off;
        kparams_local.merge_indices_off = (int32_t) layout.merge_indices_off;
        kparams_local.merge_elems       = (int32_t) layout.merge_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } अन्यथा अगर (octx->ctx->mdev.गिनती > 1) {
        स_नकल(&kparams_local, kparams, माप(kparams_local));
        kparams_local.row_start = (int32_t) row_start;
        kparams_local.row_end   = (int32_t) row_end;
        kparams = &kparams_local;
    }

    अगर (octx->ctx->vtcm_आकार < (माप_प्रकार) kparams->vtcm_आकार) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_आकार, octx->ctx->vtcm_आकार);
        वापस HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    अगर (योग_rows == 1 && kparams->n_chunks > 1) {
        काष्ठा htp_sort_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_base = vtcm_base;
        स_रखो(cctx.real_गिनती, 0, माप(cctx.real_गिनती));

        work_queue_run(octx->ctx->work_queue, htp_sort_chunk_job, &cctx, (uint32_t) kparams->n_chunks);

        अगर (is_शीर्ष_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            अगर (k == 1 && ne00 >= 128*1024) {
                भग्न *   cand_vals = (भग्न *)   (vtcm_base + kparams->merge_मूल्यावली_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                भग्न best_val = cand_vals[0];
                int32_t best_idx = cand_idxs[0];
                क्रम (int32_t c = 1; c < kparams->n_chunks; c++) {
                    अगर (cand_vals[c] > best_val) {
                        best_val = cand_vals[c];
                        best_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->तथ्य;
                dst_ptr[0] = best_idx;
                वापस HTP_STATUS_OK;
            }

            uint32_t योग_candidates = (uint32_t) kparams->n_chunks * local_k;
            भग्न *   cand_vals = (भग्न *)   (vtcm_base + kparams->merge_मूल्यावली_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
            uint32_t merge_elems = (uint32_t) kparams->merge_elems;

            अगर (merge_elems > योग_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + योग_candidates), -अनन्त, merge_elems - योग_candidates);
                क्रम (uint32_t j = योग_candidates; j < merge_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            काष्ठा htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_sort_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, merge_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->तथ्य, cand_idxs, k * माप(int32_t));
        } अन्यथा {
            bool asc = kparams->order == GGML_SORT_ORDER_ASC;
            अगर (kparams->n_chunks == 8) {
                काष्ठा htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 4);

                काष्ठा htp_argsort_merge_l2_ctx l2ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l2_job, &l2ctx, 2);

                माप_प्रकार slot_आकार = (माप_प्रकार) kparams->phase1_slot_आकार;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                स्थिर भग्न *   val0 = (स्थिर भग्न *) vtcm_base;
                स्थिर int32_t * idx0 = (स्थिर int32_t *) (vtcm_base + 4 * chunk_elems * माप(भग्न));
                स्थिर भग्न *   val1 = (स्थिर भग्न *) (vtcm_base + 4 * slot_आकार);
                स्थिर int32_t * idx1 = (स्थिर int32_t *) (vtcm_base + 4 * slot_आकार + 4 * chunk_elems * माप(भग्न));

                uint32_t n01 = cctx.real_गिनती[0] + cctx.real_गिनती[1] + cctx.real_गिनती[2] + cctx.real_गिनती[3];
                uint32_t n23 = cctx.real_गिनती[4] + cctx.real_गिनती[5] + cctx.real_गिनती[6] + cctx.real_गिनती[7];

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                काष्ठा htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, n01, val1, idx1, n23, शून्य, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->तथ्य, final_idx, ne00 * माप(int32_t));
            } अन्यथा अगर (kparams->n_chunks == 4) {
                काष्ठा htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_गिनती[0] + cctx.real_गिनती[1];
                uint32_t n23 = cctx.real_गिनती[2] + cctx.real_गिनती[3];

                स्थिर भग्न *   val01 = (स्थिर भग्न *)   (vtcm_base + kparams->merge_मूल्यावली_off);
                स्थिर int32_t * idx01 = (स्थिर int32_t *) (vtcm_base + kparams->merge_indices_off);
                स्थिर भग्न *   val23 = (स्थिर भग्न *)   (vtcm_base + kparams->merge_मूल्यावली_off)  + 2 * chunk_elems;
                स्थिर int32_t * idx23 = (स्थिर int32_t *) (vtcm_base + kparams->merge_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_base;
                काष्ठा htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val01, idx01, n01, val23, idx23, n23, शून्य, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->तथ्य, final_idx, ne00 * माप(int32_t));
            } अन्यथा अगर (kparams->n_chunks == 2) {
                माप_प्रकार slot_आकार = (माप_प्रकार) kparams->phase1_slot_आकार;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                स्थिर भग्न *   val0 = (स्थिर भग्न *)   (vtcm_base);
                स्थिर int32_t * idx0 = (स्थिर int32_t *) (vtcm_base + chunk_elems * माप(भग्न));
                स्थिर भग्न *   val1 = (स्थिर भग्न *)   (vtcm_base + slot_आकार);
                स्थिर int32_t * idx1 = (स्थिर int32_t *) (vtcm_base + slot_आकार + chunk_elems * माप(भग्न));

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                काष्ठा htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, cctx.real_गिनती[0], val1, idx1, cctx.real_गिनती[1], शून्य, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->तथ्य, final_idx, ne00 * माप(int32_t));
            } अन्यथा {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_base + chunk_elems * माप(भग्न));
                dma_sync_write(octx->ctx->dma[0], dst->तथ्य, final_idx, ne00 * माप(int32_t));
            }
        }
        वापस HTP_STATUS_OK;
    }

    काष्ठा htp_sort_multi_row_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_base        = vtcm_base;

    uint32_t dst_row_bytes = (uint32_t) dst_row_आकार;
    uint32_t rows_per_chunk = 0;
    htp_tensor_mdev_rows_per_chunk(dst, माप(int32_t), dst_row_bytes, &rows_per_chunk);
    अगर (rows_per_chunk == 0) {
        rows_per_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_threads - 1) / (uint32_t) kparams->n_threads;
    अगर (rows_per_chunk > 1) {
        dr = hex_round_up(dr, rows_per_chunk);
    }
    mctx.nrows_per_thread = dr;

    work_queue_run(octx->ctx->work_queue, htp_sort_multi_row_job, &mctx, (uint32_t) kparams->n_threads);

    वापस HTP_STATUS_OK;
}

पूर्णांक op_argsort(काष्ठा htp_ops_context * octx) {
    वापस op_sort_common(octx, false);
}

पूर्णांक op_शीर्ष_k(काष्ठा htp_ops_context * octx) {
    वापस op_sort_common(octx, true);
}
