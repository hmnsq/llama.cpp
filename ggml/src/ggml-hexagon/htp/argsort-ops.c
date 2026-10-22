#अन्तर्भाव <string.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#परिभाषय GGML_COMMON_DECL_C
#अन्तर्भाव "ggml-common.h"
#अन्तर्भाव "ggml.h"

#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "hvx-copy.h"
#अन्तर्भाव "hvx-reduce.h"
#अन्तर्भाव "dma-queue.h"

#अन्तर्भाव "hex-common.h"
#अन्तर्भाव "htp-ctx.h"
#अन्तर्भाव "htp-ops.h"
#अन्तर्भाव "htp-tensor.h"
#अन्तर्भाव "argsort-ops.h"

#यदि_अपरिभाषितम् MIN
#परिभाषय MIN(a, b) ((a) < (b) ? (a) : (b))
#यदि_अन्तः


स्थिर शून्य quicksort_मानानि_indices_asc(भिन्न * मानानि, int32_t * indices, पूर्णाङ्क वामः, पूर्णाङ्क दक्षिणः) {
    यावत् (वामः < दक्षिणः) {
        भिन्न pivot = मानानि[वामः + (दक्षिणः - वामः) / 2];
        पूर्णाङ्क i = वामः;
        पूर्णाङ्क j = दक्षिणः;

        यावत् (i <= j) {
            यावत् (मानानि[i] < pivot) i++;
            यावत् (मानानि[j] > pivot) j--;

            यदि (i <= j) {
                भिन्न tmp_val = मानानि[i];
                मानानि[i] = मानानि[j];
                मानानि[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        यदि (j - वामः < दक्षिणः - i) {
            यदि (वामः < j) quicksort_मानानि_indices_asc(मानानि, indices, वामः, j);
            वामः = i;
        } अन्यथा {
            यदि (i < दक्षिणः) quicksort_मानानि_indices_asc(मानानि, indices, i, दक्षिणः);
            दक्षिणः = j;
        }
    }
}

स्थिर शून्य quicksort_मानानि_indices_desc(भिन्न * मानानि, int32_t * indices, पूर्णाङ्क वामः, पूर्णाङ्क दक्षिणः) {
    यावत् (वामः < दक्षिणः) {
        भिन्न pivot = मानानि[वामः + (दक्षिणः - वामः) / 2];
        पूर्णाङ्क i = वामः;
        पूर्णाङ्क j = दक्षिणः;

        यावत् (i <= j) {
            यावत् (मानानि[i] > pivot) i++;
            यावत् (मानानि[j] < pivot) j--;

            यदि (i <= j) {
                भिन्न tmp_val = मानानि[i];
                मानानि[i] = मानानि[j];
                मानानि[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        यदि (j - वामः < दक्षिणः - i) {
            यदि (वामः < j) quicksort_मानानि_indices_desc(मानानि, indices, वामः, j);
            वामः = i;
        } अन्यथा {
            यदि (i < दक्षिणः) quicksort_मानानि_indices_desc(मानानि, indices, i, दक्षिणः);
            दक्षिणः = j;
        }
    }
}

स्थिर अन्तःस्थ uint32_t शिखरम्_k_max_मानम्_सूचकांक(ध्रुव भिन्न * मानानि, uint32_t n, भिन्न * मानम्) {
    int32_t max_idx = 0;
    hvx_argmax_f32(मानानि, n, 0, मानम्, &max_idx);
    प्रत्यागम (uint32_t) max_idx;
}

स्थिर ध्रुव int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

स्थिर अन्तःस्थ शून्य init_indices_ramp(int32_t * indices_buf, uint32_t गणनम्, uint32_t आधारः) {
    ध्रुव HVX_सदिशम् ind_init_vec = Q6_Vw_vadd_VwVw(*(ध्रुव HVX_सदिशम् *) argsort_ramp_lut, Q6_V_vsplat_R(आधारः));
    ध्रुव HVX_सदिशम् ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_सदिशम् * indices_buf_vec = (HVX_सदिशम् *) indices_buf;
    uint32_t अङ्कः_vecs = गणनम् / 32;
    HVX_सदिशम् curr_ind_vec = ind_init_vec;
    प्रति (uint32_t j = 0; j < अङ्कः_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = गणनम् % 32;
    यदि (rem > 0) {
        hvx_vec_store_a((शून्य *) &indices_buf_vec[अङ्कः_vecs], rem * परिमाण(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य vec_cas(HVX_सदिशम् * X_val, HVX_सदिशम् * X_idx, HVX_सदिशम् * Y_val, HVX_सदिशम् * Y_idx, bool asc) {
    HVX_सदिशम्_Pred pred = asc ? Q6_Q_vcmp_gt_VsfVsf(*X_val, *Y_val)
                              : Q6_Q_vcmp_gt_VsfVsf(*Y_val, *X_val);
    HVX_सदिशम् अग्र_X_val = Q6_V_vmux_QVV(pred, *Y_val, *X_val);
    HVX_सदिशम् अग्र_Y_val = Q6_V_vmux_QVV(pred, *X_val, *Y_val);
    HVX_सदिशम् अग्र_X_idx = Q6_V_vmux_QVV(pred, *Y_idx, *X_idx);
    HVX_सदिशम् Y_tmp_idx  = Q6_V_vmux_QVV(pred, *X_idx, *Y_idx);
    *X_val = अग्र_X_val;
    *Y_val = अग्र_Y_val;
    *X_idx = अग्र_X_idx;
    *Y_idx = Y_tmp_idx;
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य bitonic_cas_32(HVX_सदिशम् * V, HVX_सदिशम् * I, पूर्णाङ्क d, HVX_सदिशम्_Pred dir_आवरणम्, HVX_सदिशम् idx_vec, HVX_सदिशम् zero_vec) {
    HVX_सदिशम्_Pred आवरणम्_वामः;
    HVX_सदिशम् V_rot_वामः, V_rot_दक्षिणः;
    HVX_सदिशम् I_rot_वामः, I_rot_दक्षिणः;

    यदि (d == 1) {
        आवरणम्_वामः = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_वामः = Q6_V_vror_VR(*V, 4);
        V_rot_दक्षिणः = Q6_V_vror_VR(*V, 124);
        I_rot_वामः = Q6_V_vror_VR(*I, 4);
        I_rot_दक्षिणः = Q6_V_vror_VR(*I, 124);
    } अन्यथा यदि (d == 2) {
        आवरणम्_वामः = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_वामः = Q6_V_vror_VR(*V, 8);
        V_rot_दक्षिणः = Q6_V_vror_VR(*V, 120);
        I_rot_वामः = Q6_V_vror_VR(*I, 8);
        I_rot_दक्षिणः = Q6_V_vror_VR(*I, 120);
    } अन्यथा यदि (d == 4) {
        आवरणम्_वामः = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_वामः = Q6_V_vror_VR(*V, 16);
        V_rot_दक्षिणः = Q6_V_vror_VR(*V, 112);
        I_rot_वामः = Q6_V_vror_VR(*I, 16);
        I_rot_दक्षिणः = Q6_V_vror_VR(*I, 112);
    } अन्यथा यदि (d == 8) {
        आवरणम्_वामः = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_वामः = Q6_V_vror_VR(*V, 32);
        V_rot_दक्षिणः = Q6_V_vror_VR(*V, 96);
        I_rot_वामः = Q6_V_vror_VR(*I, 32);
        I_rot_दक्षिणः = Q6_V_vror_VR(*I, 96);
    } अन्यथा {
        आवरणम्_वामः = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_वामः = Q6_V_vror_VR(*V, 64);
        V_rot_दक्षिणः = Q6_V_vror_VR(*V, 64);
        I_rot_वामः = Q6_V_vror_VR(*I, 64);
        I_rot_दक्षिणः = Q6_V_vror_VR(*I, 64);
    }

    HVX_सदिशम् V_paired = Q6_V_vmux_QVV(आवरणम्_वामः, V_rot_वामः, V_rot_दक्षिणः);
    HVX_सदिशम् I_paired = Q6_V_vmux_QVV(आवरणम्_वामः, I_rot_वामः, I_rot_दक्षिणः);

    HVX_सदिशम्_Pred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_सदिशम्_Pred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_सदिशम्_Pred आवरणम्_दक्षिणः = Q6_Q_not_Q(आवरणम्_वामः);
    HVX_सदिशम्_Pred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(आवरणम्_वामः, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, आवरणम्_दक्षिणः)
    );
    HVX_सदिशम्_Pred Q_swap = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_आवरणम्, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_आवरणम्), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_swap, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_swap, I_paired, *I);
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य bitonic_क्रमय_generic_hvx(uint8_t * मानानि, uint8_t * indices, पूर्णाङ्क K, bool asc_क्रमः) {
    HVX_सदिशम् V[32];
    HVX_सदिशम् I[32];

    HVX_सदिशम् zero_vec = Q6_V_vzero();
    HVX_सदिशम् idx_vec = *(ध्रुव HVX_सदिशम् *) argsort_ramp_lut;

    प्रति (पूर्णाङ्क v = 0; v < K; v++) {
        V[v] = *(HVX_सदिशम् *)(मानानि + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_सदिशम्_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_सदिशम्_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    पूर्णाङ्क M = 5;
    यावत् ((1 << (M - 5)) < K) M++;

    प्रति (पूर्णाङ्क s = 1; s <= M; s++) {
        प्रति (पूर्णाङ्क stage_d = s - 1; stage_d >= 0; stage_d--) {
            पूर्णाङ्क d = 1 << stage_d;
            यदि (d >= 32) {
                पूर्णाङ्क v_dist = d / 32;
                प्रति (पूर्णाङ्क v1 = 0; v1 < K; v1++) {
                    यदि ((v1 & v_dist) == 0) {
                        पूर्णाङ्क v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_क्रमः;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } अन्यथा {
                यदि (s < 5) {
                    HVX_सदिशम्_Pred dir_आवरणम् = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    प्रति (पूर्णाङ्क v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_आवरणम्, idx_vec, zero_vec);
                    }
                } अन्यथा {
                    प्रति (पूर्णाङ्क v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_क्रमः;
                        HVX_सदिशम्_Pred dir_आवरणम् = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_आवरणम्, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    प्रति (पूर्णाङ्क v = 0; v < K; v++) {
        *(HVX_सदिशम् *)(मानानि + v * 128)  = V[v];
        *(HVX_सदिशम् *)(indices + v * 128) = I[v];
    }
}

स्थिर शून्य bitonic_क्रमय_vtcm_desc(uint8_t * मानानि, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_सदिशम् zero_vec = Q6_V_vzero();
    HVX_सदिशम् idx_vec = *(ध्रुव HVX_सदिशम् *) argsort_ramp_lut;

    HVX_सदिशम्_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_सदिशम्_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    यदि (init_indices) {
        प्रति (uint32_t v = 0; v < n_vec; v++) {
            HVX_सदिशम् idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_सदिशम् *)(indices + v * 128) = idx;
        }
    }

    पूर्णाङ्क M = 5;
    यावत् ((1u << (M - 5)) < n_vec) M++;

    प्रति (पूर्णाङ्क s = 1; s <= M; s++) {
        प्रति (पूर्णाङ्क stage_d = s - 1; stage_d >= 0; stage_d--) {
            पूर्णाङ्क d = 1 << stage_d;
            यदि (d >= 32) {
                uint32_t v_dist = d / 32;
                प्रति (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    यदि ((v1 & v_dist) == 0) {
                        uint32_t v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : false;

                        HVX_सदिशम् Vv1 = *(HVX_सदिशम् *)(मानानि + v1 * 128);
                        HVX_सदिशम् Iv1 = *(HVX_सदिशम् *)(indices + v1 * 128);
                        HVX_सदिशम् Vv2 = *(HVX_सदिशम् *)(मानानि + v2 * 128);
                        HVX_सदिशम् Iv2 = *(HVX_सदिशम् *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_सदिशम् *)(मानानि + v1 * 128)  = Vv1;
                        *(HVX_सदिशम् *)(indices + v1 * 128) = Iv1;
                        *(HVX_सदिशम् *)(मानानि + v2 * 128)  = Vv2;
                        *(HVX_सदिशम् *)(indices + v2 * 128) = Iv2;
                    }
                }
            } अन्यथा {
                यदि (s < 5) {
                    HVX_सदिशम्_Pred dir_आवरणम् = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    प्रति (uint32_t v = 0; v < n_vec; v++) {
                        HVX_सदिशम् Vv = *(HVX_सदिशम् *)(मानानि + v * 128);
                        HVX_सदिशम् Iv = *(HVX_सदिशम् *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_आवरणम्, idx_vec, zero_vec);

                        *(HVX_सदिशम् *)(मानानि + v * 128)  = Vv;
                        *(HVX_सदिशम् *)(indices + v * 128) = Iv;
                    }
                } अन्यथा {
                    प्रति (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_सदिशम्_Pred dir_आवरणम् = asc ? pred_all_1s : pred_all_0s;

                        HVX_सदिशम् Vv = *(HVX_सदिशम् *)(मानानि + v * 128);
                        HVX_सदिशम् Iv = *(HVX_सदिशम् *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_आवरणम्, idx_vec, zero_vec);

                        *(HVX_सदिशम् *)(मानानि + v * 128)  = Vv;
                        *(HVX_सदिशम् *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

स्थिर शून्य शिखरम्_k_select_tiled(dma_queue * dma_q, संरचना htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               भिन्न * मानानि_buf, int32_t * indices_buf,
                               भिन्न * बहिर्गमनम्_मानानि, int32_t * बहिर्गमनम्_indices,
                               uint32_t chunk_आधारः) {
    ध्रुव uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_गणनम्    = n_tiles * k;
    uint32_t merge_n_vec        = hmx_ceil_div(candidate_गणनम्, 32);
    uint32_t merge_n_vec_pow2   = 1;
    यावत् (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
    uint32_t merge_elems        = merge_n_vec_pow2 * 32;
    भिन्न * candidate_मानानि    = मानानि_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    प्रति (uint32_t विस्थापनम् = 0; विस्थापनम् < n; विस्थापनम् += tile_elems) {
        uint32_t tile_गणनम् = MIN(tile_elems, n - विस्थापनम्);
        dma_sync_read(dma_q, मानानि_buf, src_addr + (size_t) विस्थापनम् * परिमाण(भिन्न), tile_गणनम् * परिमाण(भिन्न));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) विस्थापनम्);
        यदि (tile_गणनम् < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (मानानि_buf + tile_गणनम्), -INFINITY, tile_elems - tile_गणनम्);
        }

        bitonic_क्रमय_vtcm_desc((uint8_t *) मानानि_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_गणनम्);
        प्रति (uint32_t j = 0; j < tile_k; j++) {
            candidate_मानानि[candidate_pos] = मानानि_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) विस्थापनम्;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) विस्थापनम्);
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    यदि (merge_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_मानानि + candidate_pos), -INFINITY, merge_elems - candidate_pos);
        प्रति (uint32_t j = candidate_pos; j < merge_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_क्रमय_vtcm_desc((uint8_t *) candidate_मानानि, (uint8_t *) candidate_indices, merge_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    प्रति (uint32_t j = 0; j < k; j++) {
        बहिर्गमनम्_मानानि[j]  = candidate_मानानि[j];
        बहिर्गमनम्_indices[j] = candidate_indices[j] + (int32_t) chunk_आधारः;
    }
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य sort32_f32_hvx(uint8_t * मानानि, uint8_t * indices, गणना ggml_क्रमय_क्रमः क्रमः) {
    bitonic_क्रमय_generic_hvx(मानानि, indices, 1, क्रमः == GGML_क्रमय_क्रमः_ASC);
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य sort64_f32_hvx(uint8_t * मानानि, uint8_t * indices, गणना ggml_क्रमय_क्रमः क्रमः) {
    bitonic_क्रमय_generic_hvx(मानानि, indices, 2, क्रमः == GGML_क्रमय_क्रमः_ASC);
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य sort128_f32_hvx(uint8_t * मानानि, uint8_t * indices, गणना ggml_क्रमय_क्रमः क्रमः) {
    bitonic_क्रमय_generic_hvx(मानानि, indices, 4, क्रमः == GGML_क्रमय_क्रमः_ASC);
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य sort256_f32_hvx(uint8_t * मानानि, uint8_t * indices, गणना ggml_क्रमय_क्रमः क्रमः) {
    bitonic_क्रमय_generic_hvx(मानानि, indices, 8, क्रमः == GGML_क्रमय_क्रमः_ASC);
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य sort512_f32_hvx(uint8_t * मानानि, uint8_t * indices, गणना ggml_क्रमय_क्रमः क्रमः) {
    bitonic_क्रमय_generic_hvx(मानानि, indices, 16, क्रमः == GGML_क्रमय_क्रमः_ASC);
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य sort1024_f32_hvx(uint8_t * मानानि, uint8_t * indices, गणना ggml_क्रमय_क्रमः क्रमः) {
    bitonic_क्रमय_generic_hvx(मानानि, indices, 32, क्रमः == GGML_क्रमय_क्रमः_ASC);
}

स्थिर शून्य merge_runs(
    ध्रुव भिन्न * सीमित अन्तर्_val0, ध्रुव int32_t * सीमित अन्तर्_idx0, uint32_t n0,
    ध्रुव भिन्न * सीमित अन्तर्_val1, ध्रुव int32_t * सीमित अन्तर्_idx1, uint32_t n1,
    भिन्न * सीमित बहिर्गमनम्_val, int32_t * सीमित बहिर्गमनम्_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t बहिर्गमनम् = 0;

    यदि (asc) {
        यावत् (i0 < n0 && i1 < n1) {
            यदि (अन्तर्_val0[i0] <= अन्तर्_val1[i1]) {
                यदि (बहिर्गमनम्_val) बहिर्गमनम्_val[बहिर्गमनम्] = अन्तर्_val0[i0];
                बहिर्गमनम्_idx[बहिर्गमनम्] = अन्तर्_idx0[i0];
                i0++;
            } अन्यथा {
                यदि (बहिर्गमनम्_val) बहिर्गमनम्_val[बहिर्गमनम्] = अन्तर्_val1[i1];
                बहिर्गमनम्_idx[बहिर्गमनम्] = अन्तर्_idx1[i1];
                i1++;
            }
            बहिर्गमनम्++;
        }
    } अन्यथा {
        यावत् (i0 < n0 && i1 < n1) {
            यदि (अन्तर्_val0[i0] >= अन्तर्_val1[i1]) {
                यदि (बहिर्गमनम्_val) बहिर्गमनम्_val[बहिर्गमनम्] = अन्तर्_val0[i0];
                बहिर्गमनम्_idx[बहिर्गमनम्] = अन्तर्_idx0[i0];
                i0++;
            } अन्यथा {
                यदि (बहिर्गमनम्_val) बहिर्गमनम्_val[बहिर्गमनम्] = अन्तर्_val1[i1];
                बहिर्गमनम्_idx[बहिर्गमनम्] = अन्तर्_idx1[i1];
                i1++;
            }
            बहिर्गमनम्++;
        }
    }

    यावत् (i0 < n0) {
        यदि (बहिर्गमनम्_val) बहिर्गमनम्_val[बहिर्गमनम्] = अन्तर्_val0[i0];
        बहिर्गमनम्_idx[बहिर्गमनम्] = अन्तर्_idx0[i0];
        i0++;
        बहिर्गमनम्++;
    }

    यावत् (i1 < n1) {
        यदि (बहिर्गमनम्_val) बहिर्गमनम्_val[बहिर्गमनम्] = अन्तर्_val1[i1];
        बहिर्गमनम्_idx[बहिर्गमनम्] = अन्तर्_idx1[i1];
        i1++;
        बहिर्गमनम्++;
    }
}

संरचना htp_क्रमय_chunk_ctx {
    संरचना htp_ops_context *              octx;
    ध्रुव संरचना htp_क्रमय_kernel_params * kparams;
    uint8_t *                             vtcm_आधारः;
    uint32_t                              real_गणनम्[HTP_MAX_NTHREADS];
};

स्थिर शून्य htp_क्रमय_chunk_job(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त) {
    संरचना htp_क्रमय_chunk_ctx * cctx = (संरचना htp_क्रमय_chunk_ctx *) दत्त;
    संरचना htp_ops_context * octx = cctx->octx;
    ध्रुव संरचना htp_क्रमय_kernel_params * kparams = cctx->kparams;
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_आधारः  = i * chunk_elems;

    uint32_t real_गणनम् = (chunk_आधारः < ne00) ? MIN(chunk_elems, ne00 - chunk_आधारः) : 0;
    cctx->real_गणनम्[i] = real_गणनम्;

    uint8_t * spad = cctx->vtcm_आधारः + (size_t) kparams->phase1_slot_आकार * i;
    भिन्न *   मानानि_buf  = (भिन्न *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * परिमाण(भिन्न));

    भिन्न *   merge_मानानि  = (भिन्न *)   (cctx->vtcm_आधारः + kparams->merge_मानानि_off);
    int32_t * merge_indices = (int32_t *) (cctx->vtcm_आधारः + kparams->merge_indices_off);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[i];
    ध्रुव dma_addr_t src_addr = src0->दत्त + (size_t) chunk_आधारः * परिमाण(भिन्न);

    यदि (kparams->is_शिखरम्_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        यदि (real_गणनम् == 0) {
            प्रति (uint32_t j = 0; j < local_k; j++) {
                merge_मानानि[i * local_k + j]  = -INFINITY;
                merge_indices[i * local_k + j] = 0;
            }
            प्रत्यागम;
        }

        यदि (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], मानानि_buf, src_addr, real_गणनम् * परिमाण(भिन्न));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            भिन्न max_मानम्;
            uint32_t max_सूचकांक = शिखरम्_k_max_मानम्_सूचकांक(मानानि_buf, real_गणनम्, &max_मानम्);
            merge_मानानि[i] = max_मानम्;
            merge_indices[i] = (int32_t) (max_सूचकांक + chunk_आधारः);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            प्रत्यागम;
        }

        यदि (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            शिखरम्_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_गणनम्, local_k,
                               मानानि_buf, indices_buf,
                               &merge_मानानि[i * local_k], &merge_indices[i * local_k],
                               chunk_आधारः);
            प्रत्यागम;
        }

        dma_sync_read(octx->ctx->dma[i], मानानि_buf, src_addr, real_गणनम् * परिमाण(भिन्न));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        यदि (chunk_elems > real_गणनम्) {
            hvx_splat_f32_u((uint8_t *)(मानानि_buf + real_गणनम्), -INFINITY, chunk_elems - real_गणनम्);
        }

        bitonic_क्रमय_vtcm_desc((uint8_t *) मानानि_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        प्रति (uint32_t j = 0; j < local_k; j++) {
            merge_मानानि[i * local_k + j]  = मानानि_buf[j];
            merge_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_आधारः;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } अन्यथा {
        यदि (real_गणनम् > 0) {
            dma_sync_read(octx->ctx->dma[i], मानानि_buf, src_addr, real_गणनम् * परिमाण(भिन्न));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_गणनम्, chunk_आधारः);

            यदि (kparams->क्रमः == GGML_क्रमय_क्रमः_ASC) {
                quicksort_मानानि_indices_asc(मानानि_buf, indices_buf, 0, real_गणनम् - 1);
            } अन्यथा {
                quicksort_मानानि_indices_desc(मानानि_buf, indices_buf, 0, real_गणनम् - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

संरचना htp_argsort_merge_l1_ctx {
    संरचना htp_क्रमय_chunk_ctx * cctx;
    bool                        asc;
};

स्थिर शून्य htp_argsort_merge_l1_job(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त) {
    संरचना htp_argsort_merge_l1_ctx * mctx = (संरचना htp_argsort_merge_l1_ctx *) दत्त;
    संरचना htp_क्रमय_chunk_ctx * cctx = mctx->cctx;
    ध्रुव संरचना htp_क्रमय_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_गणनम्[c0];
    uint32_t count1 = cctx->real_गणनम्[c1];

    size_t slot_आकार = (size_t) kparams->phase1_slot_आकार;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    ध्रुव भिन्न *   val0 = (ध्रुव भिन्न *)   (cctx->vtcm_आधारः + slot_आकार * c0);
    ध्रुव int32_t * idx0 = (ध्रुव int32_t *) (cctx->vtcm_आधारः + slot_आकार * c0 + chunk_elems * परिमाण(भिन्न));
    ध्रुव भिन्न *   val1 = (ध्रुव भिन्न *)   (cctx->vtcm_आधारः + slot_आकार * c1);
    ध्रुव int32_t * idx1 = (ध्रुव int32_t *) (cctx->vtcm_आधारः + slot_आकार * c1 + chunk_elems * परिमाण(भिन्न));

    uint32_t बहिर्गमनम्_विस्थापनम् = (2 * i) * chunk_elems;
    भिन्न *   बहिर्गमनम्_val = (भिन्न *)   (cctx->vtcm_आधारः + kparams->merge_मानानि_off)  + बहिर्गमनम्_विस्थापनम्;
    int32_t * बहिर्गमनम्_idx = (int32_t *) (cctx->vtcm_आधारः + kparams->merge_indices_off) + बहिर्गमनम्_विस्थापनम्;

    संरचना htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, बहिर्गमनम्_val, बहिर्गमनम्_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

संरचना htp_argsort_merge_l2_ctx {
    संरचना htp_क्रमय_chunk_ctx * cctx;
    bool                        asc;
};

स्थिर शून्य htp_argsort_merge_l2_job(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क i, शून्य * दत्त) {
    संरचना htp_argsort_merge_l2_ctx * mctx = (संरचना htp_argsort_merge_l2_ctx *) दत्त;
    संरचना htp_क्रमय_chunk_ctx * cctx = mctx->cctx;
    ध्रुव संरचना htp_क्रमय_kernel_params * kparams = cctx->kparams;

    size_t slot_आकार = (size_t) kparams->phase1_slot_आकार;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    ध्रुव भिन्न *   val0 = (ध्रुव भिन्न *)   (cctx->vtcm_आधारः + kparams->merge_मानानि_off)  + off0;
    ध्रुव int32_t * idx0 = (ध्रुव int32_t *) (cctx->vtcm_आधारः + kparams->merge_indices_off) + off0;
    ध्रुव भिन्न *   val1 = (ध्रुव भिन्न *)   (cctx->vtcm_आधारः + kparams->merge_मानानि_off)  + off1;
    ध्रुव int32_t * idx1 = (ध्रुव int32_t *) (cctx->vtcm_आधारः + kparams->merge_indices_off) + off1;

    uint32_t count0 = cctx->real_गणनम्[4*i + 0] + cctx->real_गणनम्[4*i + 1];
    uint32_t count1 = cctx->real_गणनम्[4*i + 2] + cctx->real_गणनम्[4*i + 3];

    uint8_t * slot_आधारः = cctx->vtcm_आधारः + (4 * i) * slot_आकार;
    भिन्न *   बहिर्गमनम्_val   = (भिन्न *) slot_आधारः;
    int32_t * बहिर्गमनम्_idx   = (int32_t *) (slot_आधारः + 4 * chunk_elems * परिमाण(भिन्न));

    संरचना htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, बहिर्गमनम्_val, बहिर्गमनम्_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

संरचना htp_क्रमय_multi_row_ctx {
    संरचना htp_ops_context *              octx;
    ध्रुव संरचना htp_क्रमय_kernel_params * kparams;
    uint8_t *                             vtcm_आधारः;
    uint32_t                              nrows_प्रत्येकम्_thread;
};

संरचना htp_tensor_row_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

स्थिर अन्तःस्थ संरचना htp_tensor_row_coord init_row_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    प्रत्यागम (संरचना htp_tensor_row_coord){ i01, i02, i03 };
}

स्थिर अन्तःस्थ शून्य चरणः_row_coord(संरचना htp_tensor_row_coord * c, uint32_t ne01, uint32_t ne02) {
    c->i01++;
    यदि (c->i01 == ne01) {
        c->i01 = 0;
        c->i02++;
        यदि (c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

स्थिर अन्तःस्थ size_t coord_विस्थापनम्(संरचना htp_tensor_row_coord c, size_t nb1, size_t nb2, size_t nb3) {
    प्रत्यागम (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

स्थिर अन्तःस्थ शून्य compute_row_क्रमय(
    भिन्न * cur_मानानि,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_शिखरम्_k,
    गणना ggml_क्रमय_क्रमः क्रमः
) {
    यदि (is_शिखरम्_k) {
        यदि (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            विकल्प (ne00) {
                पक्ष 32:   sort32_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, GGML_क्रमय_क्रमः_DESC); भङ्ग;
                पक्ष 64:   sort64_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, GGML_क्रमय_क्रमः_DESC); भङ्ग;
                पक्ष 128:  sort128_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, GGML_क्रमय_क्रमः_DESC); भङ्ग;
                पक्ष 256:  sort256_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, GGML_क्रमय_क्रमः_DESC); भङ्ग;
                पक्ष 512:  sort512_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, GGML_क्रमय_क्रमः_DESC); भङ्ग;
                पक्ष 1024: sort1024_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, GGML_क्रमय_क्रमः_DESC); भङ्ग;
            }
        } अन्यथा {
            यदि (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_मानानि + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_क्रमय_vtcm_desc((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } अन्यथा {
        यदि (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            विकल्प (ne00) {
                पक्ष 32:   sort32_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, क्रमः); भङ्ग;
                पक्ष 64:   sort64_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, क्रमः); भङ्ग;
                पक्ष 128:  sort128_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, क्रमः); भङ्ग;
                पक्ष 256:  sort256_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, क्रमः); भङ्ग;
                पक्ष 512:  sort512_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, क्रमः); भङ्ग;
                पक्ष 1024: sort1024_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, क्रमः); भङ्ग;
            }
        } अन्यथा {
            init_indices_ramp(cur_indices, ne00, 0);
            यदि (क्रमः == GGML_क्रमय_क्रमः_ASC) {
                quicksort_मानानि_indices_asc(cur_मानानि, cur_indices, 0, ne00 - 1);
            } अन्यथा {
                quicksort_मानानि_indices_desc(cur_मानानि, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

स्थिर शून्य htp_क्रमय_multi_row_job(अचिह्नित पूर्णाङ्क n, अचिह्नित पूर्णाङ्क ith, शून्य * दत्त) {
    संरचना htp_क्रमय_multi_row_ctx * mctx = (संरचना htp_क्रमय_multi_row_ctx *) दत्त;
    संरचना htp_ops_context * octx = mctx->octx;
    ध्रुव संरचना htp_क्रमय_kernel_params * kparams = mctx->kparams;
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

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
    uint32_t row_अन्तः   = (uint32_t) kparams->row_अन्तः;
    uint32_t r_start   = row_start + ith * mctx->nrows_प्रत्येकम्_thread;
    uint32_t r_अन्तः     = MIN(r_start + mctx->nrows_प्रत्येकम्_thread, row_अन्तः);

    यदि (r_start >= r_अन्तः) प्रत्यागम;

    size_t spad_विस्थापनम् = ith * (size_t) (kparams->phase1_slot_आकार * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_आधारः + spad_विस्थापनम्;
    size_t slot_आकार = (size_t) kparams->phase1_slot_आकार;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    भिन्न *   मानानि_buf  = (भिन्न *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * परिमाण(भिन्न));

    dma_queue * dma_q = octx->ctx->dma[ith];
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_शिखरम्_k = kparams->is_शिखरम्_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    गणना ggml_क्रमय_क्रमः क्रमः = (गणना ggml_क्रमय_क्रमः) kparams->क्रमः;

    ध्रुव uint32_t src_अष्टकानि = ne00 * परिमाण(भिन्न);
    ध्रुव uint32_t dst_अष्टकानि = (is_शिखरम्_k ? k : ne00) * परिमाण(int32_t);

    यदि (kparams->n_slots == 2) {
        संरचना htp_tensor_row_coord अविभाज्यः_coord = init_row_coord(r_start, ne01, ne02);
        प्रति (uint32_t r = r_start, spad_idx = 0; r < r_अन्तः && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * slot_आकार;
            भिन्न *   cur_मानानि  = (भिन्न *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * परिमाण(भिन्न));

            dma_addr_t dst_addr = dst->दत्त + coord_विस्थापनम्(अविभाज्यः_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->दत्त + coord_विस्थापनम्(अविभाज्यः_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_addr, cur_indices),
                           dst_अष्टकानि, dst_अष्टकानि, dst_अष्टकानि, 0);

            // Prefetch input row
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(cur_मानानि, src_addr),
                           src_अष्टकानि, src_अष्टकानि, src_अष्टकानि, 1);

            चरणः_row_coord(&अविभाज्यः_coord, ne01, ne02);
        }

        संरचना htp_tensor_row_coord dst_coord = init_row_coord(r_start, ne01, ne02);
        संरचना htp_tensor_row_coord अग्र_src_coord = init_row_coord(r_start + 2, ne01, ne02);

        प्रति (uint32_t r = r_start; r < r_अन्तः; r++) {
            uint32_t cur_slot = (r - r_start) & 1;
            uint8_t * cur_spad = thread_spad + cur_slot * slot_आकार;
            भिन्न *   cur_मानानि  = (भिन्न *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * परिमाण(भिन्न));

            // Wait for previous writeback of this slot to complete
            dma_queue_निष्कासय(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_queue_निष्कासय(dma_q);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_क्रमय(cur_मानानि, cur_indices, ne00, chunk_elems, is_शिखरम्_k, क्रमः);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->दत्त + coord_विस्थापनम्(dst_coord, nb1, nb2, nb3);
            dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(dst_addr, cur_indices),
                           dst_अष्टकानि, dst_अष्टकानि, dst_अष्टकानि, 1);
            चरणः_row_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            ध्रुव uint32_t अग्र_row = r + 2;
            यदि (अग्र_row < r_अन्तः) {
                dma_addr_t अग्र_src_addr = src0->दत्त + coord_विस्थापनम्(अग्र_src_coord, nb01, nb02, nb03);
                dma_queue_प्रक्षिप(dma_q, dma_make_दत्त(cur_मानानि, अग्र_src_addr),
                               src_अष्टकानि, src_अष्टकानि, src_अष्टकानि, 1);
                चरणः_row_coord(&अग्र_src_coord, ne01, ne02);
            }
        }

        dma_queue_flush(dma_q);
    } अन्यथा {
        संरचना htp_tensor_row_coord coord = init_row_coord(r_start, ne01, ne02);
        प्रति (uint32_t r = r_start; r < r_अन्तः; r++) {
            dma_addr_t src_addr = src0->दत्त + coord_विस्थापनम्(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->दत्त + coord_विस्थापनम्(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, मानानि_buf, src_addr, src_अष्टकानि);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_क्रमय(मानानि_buf, indices_buf, ne00, chunk_elems, is_शिखरम्_k, क्रमः);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_अष्टकानि);
            चरणः_row_coord(&coord, ne01, ne02);
        }
    }
}

स्थिर पूर्णाङ्क op_क्रमय_common(संरचना htp_ops_context * octx, bool is_शिखरम्_k) {
    यदि (octx->src[0]->type != HTP_TYPE_F32) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    यदि (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        प्रत्यागम HTP_स्थितिः_NO_SUPPORT;
    }

    ध्रुव uint32_t योग_पङ्क्तिकाः = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव size_t dst_row_आकार = dst->ne[0] * परिमाण(int32_t);

    uint32_t row_start = 0;
    uint32_t row_अन्तः   = योग_पङ्क्तिकाः;
    यदि (octx->ctx->mdev.गणनम् > 1) {
        uint32_t पङ्क्तिकाः_प्रत्येकम्_chunk = 0;
        htp_tensor_mdev_पङ्क्तिकाः_प्रत्येकम्_chunk(dst, परिमाण(int32_t), (uint32_t) dst_row_आकार, &पङ्क्तिकाः_प्रत्येकम्_chunk);
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            योग_पङ्क्तिकाः, पङ्क्तिकाः_प्रत्येकम्_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        row_अन्तः   = range.start + range.गणनम्;
    }

    ध्रुव uint32_t nrows = row_अन्तः - row_start;
    यदि (nrows == 0) {
        प्रत्यागम HTP_स्थितिः_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    संरचना htp_क्रमय_kernel_params kparams_local;
    ध्रुव संरचना htp_क्रमय_kernel_params * kparams = (ध्रुव संरचना htp_क्रमय_kernel_params *) octx->kernel_params;

    यदि (kparams->n_तन्तवः == 0) {
        संरचना htp_क्रमय_vtcm_layout layout;
        यदि (!htp_क्रमय_solve_layout(&layout, ne00, योग_पङ्क्तिकाः, k, octx->n_तन्तवः, octx->ctx->vtcm_आकार, is_शिखरम्_k)) {
            प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
        }
        स्मृतिं_पूरय(&kparams_local, 0, परिमाण(kparams_local));
        kparams_local.n_तन्तवः         = (int32_t) layout.n_तन्तवः;
        kparams_local.योग_पङ्क्तिकाः        = (int32_t) योग_पङ्क्तिकाः;
        kparams_local.row_start         = (int32_t) row_start;
        kparams_local.row_अन्तः           = (int32_t) row_अन्तः;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.क्रमः             = is_शिखरम्_k ? GGML_क्रमय_क्रमः_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_शिखरम्_k          = is_शिखरम्_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_आकार         = (int32_t) layout.योग_अष्टकानि;
        kparams_local.phase1_slot_आकार  = (int32_t) layout.phase1_slot_आकार;
        kparams_local.merge_मानानि_off  = (int32_t) layout.merge_मानानि_off;
        kparams_local.merge_indices_off = (int32_t) layout.merge_indices_off;
        kparams_local.merge_elems       = (int32_t) layout.merge_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } अन्यथा यदि (octx->ctx->mdev.गणनम् > 1) {
        स्मृतिं_प्रतिलिख(&kparams_local, kparams, परिमाण(kparams_local));
        kparams_local.row_start = (int32_t) row_start;
        kparams_local.row_अन्तः   = (int32_t) row_अन्तः;
        kparams = &kparams_local;
    }

    यदि (octx->ctx->vtcm_आकार < (size_t) kparams->vtcm_आकार) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_आकार, octx->ctx->vtcm_आकार);
        प्रत्यागम HTP_स्थितिः_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_आधारः = (uint8_t *) octx->ctx->vtcm_आधारः;

    यदि (योग_पङ्क्तिकाः == 1 && kparams->n_chunks > 1) {
        संरचना htp_क्रमय_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_आधारः = vtcm_आधारः;
        स्मृतिं_पूरय(cctx.real_गणनम्, 0, परिमाण(cctx.real_गणनम्));

        work_queue_सततम्(octx->ctx->work_queue, htp_क्रमय_chunk_job, &cctx, (uint32_t) kparams->n_chunks);

        यदि (is_शिखरम्_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            यदि (k == 1 && ne00 >= 128*1024) {
                भिन्न *   cand_vals = (भिन्न *)   (vtcm_आधारः + kparams->merge_मानानि_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_आधारः + kparams->merge_indices_off);
                भिन्न best_val = cand_vals[0];
                int32_t best_idx = cand_idxs[0];
                प्रति (int32_t c = 1; c < kparams->n_chunks; c++) {
                    यदि (cand_vals[c] > best_val) {
                        best_val = cand_vals[c];
                        best_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->दत्त;
                dst_ptr[0] = best_idx;
                प्रत्यागम HTP_स्थितिः_OK;
            }

            uint32_t योग_candidates = (uint32_t) kparams->n_chunks * local_k;
            भिन्न *   cand_vals = (भिन्न *)   (vtcm_आधारः + kparams->merge_मानानि_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_आधारः + kparams->merge_indices_off);
            uint32_t merge_elems = (uint32_t) kparams->merge_elems;

            यदि (merge_elems > योग_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + योग_candidates), -INFINITY, merge_elems - योग_candidates);
                प्रति (uint32_t j = योग_candidates; j < merge_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_क्रमय_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, merge_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->दत्त, cand_idxs, k * परिमाण(int32_t));
        } अन्यथा {
            bool asc = kparams->क्रमः == GGML_क्रमय_क्रमः_ASC;
            यदि (kparams->n_chunks == 8) {
                संरचना htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_सततम्(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 4);

                संरचना htp_argsort_merge_l2_ctx l2ctx = { &cctx, asc };
                work_queue_सततम्(octx->ctx->work_queue, htp_argsort_merge_l2_job, &l2ctx, 2);

                size_t slot_आकार = (size_t) kparams->phase1_slot_आकार;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ध्रुव भिन्न *   val0 = (ध्रुव भिन्न *) vtcm_आधारः;
                ध्रुव int32_t * idx0 = (ध्रुव int32_t *) (vtcm_आधारः + 4 * chunk_elems * परिमाण(भिन्न));
                ध्रुव भिन्न *   val1 = (ध्रुव भिन्न *) (vtcm_आधारः + 4 * slot_आकार);
                ध्रुव int32_t * idx1 = (ध्रुव int32_t *) (vtcm_आधारः + 4 * slot_आकार + 4 * chunk_elems * परिमाण(भिन्न));

                uint32_t n01 = cctx.real_गणनम्[0] + cctx.real_गणनम्[1] + cctx.real_गणनम्[2] + cctx.real_गणनम्[3];
                uint32_t n23 = cctx.real_गणनम्[4] + cctx.real_गणनम्[5] + cctx.real_गणनम्[6] + cctx.real_गणनम्[7];

                int32_t * final_idx = (int32_t *) (vtcm_आधारः + kparams->merge_indices_off);
                संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, n01, val1, idx1, n23, रिक्त, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->दत्त, final_idx, ne00 * परिमाण(int32_t));
            } अन्यथा यदि (kparams->n_chunks == 4) {
                संरचना htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_सततम्(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_गणनम्[0] + cctx.real_गणनम्[1];
                uint32_t n23 = cctx.real_गणनम्[2] + cctx.real_गणनम्[3];

                ध्रुव भिन्न *   val01 = (ध्रुव भिन्न *)   (vtcm_आधारः + kparams->merge_मानानि_off);
                ध्रुव int32_t * idx01 = (ध्रुव int32_t *) (vtcm_आधारः + kparams->merge_indices_off);
                ध्रुव भिन्न *   val23 = (ध्रुव भिन्न *)   (vtcm_आधारः + kparams->merge_मानानि_off)  + 2 * chunk_elems;
                ध्रुव int32_t * idx23 = (ध्रुव int32_t *) (vtcm_आधारः + kparams->merge_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_आधारः;
                संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val01, idx01, n01, val23, idx23, n23, रिक्त, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->दत्त, final_idx, ne00 * परिमाण(int32_t));
            } अन्यथा यदि (kparams->n_chunks == 2) {
                size_t slot_आकार = (size_t) kparams->phase1_slot_आकार;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ध्रुव भिन्न *   val0 = (ध्रुव भिन्न *)   (vtcm_आधारः);
                ध्रुव int32_t * idx0 = (ध्रुव int32_t *) (vtcm_आधारः + chunk_elems * परिमाण(भिन्न));
                ध्रुव भिन्न *   val1 = (ध्रुव भिन्न *)   (vtcm_आधारः + slot_आकार);
                ध्रुव int32_t * idx1 = (ध्रुव int32_t *) (vtcm_आधारः + slot_आकार + chunk_elems * परिमाण(भिन्न));

                int32_t * final_idx = (int32_t *) (vtcm_आधारः + kparams->merge_indices_off);
                संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, cctx.real_गणनम्[0], val1, idx1, cctx.real_गणनम्[1], रिक्त, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->दत्त, final_idx, ne00 * परिमाण(int32_t));
            } अन्यथा {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_आधारः + chunk_elems * परिमाण(भिन्न));
                dma_sync_write(octx->ctx->dma[0], dst->दत्त, final_idx, ne00 * परिमाण(int32_t));
            }
        }
        प्रत्यागम HTP_स्थितिः_OK;
    }

    संरचना htp_क्रमय_multi_row_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_आधारः        = vtcm_आधारः;

    uint32_t dst_row_अष्टकानि = (uint32_t) dst_row_आकार;
    uint32_t पङ्क्तिकाः_प्रत्येकम्_chunk = 0;
    htp_tensor_mdev_पङ्क्तिकाः_प्रत्येकम्_chunk(dst, परिमाण(int32_t), dst_row_अष्टकानि, &पङ्क्तिकाः_प्रत्येकम्_chunk);
    यदि (पङ्क्तिकाः_प्रत्येकम्_chunk == 0) {
        पङ्क्तिकाः_प्रत्येकम्_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_तन्तवः - 1) / (uint32_t) kparams->n_तन्तवः;
    यदि (पङ्क्तिकाः_प्रत्येकम्_chunk > 1) {
        dr = hex_आवर्तनम्_ऊर्ध्वम्(dr, पङ्क्तिकाः_प्रत्येकम्_chunk);
    }
    mctx.nrows_प्रत्येकम्_thread = dr;

    work_queue_सततम्(octx->ctx->work_queue, htp_क्रमय_multi_row_job, &mctx, (uint32_t) kparams->n_तन्तवः);

    प्रत्यागम HTP_स्थितिः_OK;
}

पूर्णाङ्क op_argsort(संरचना htp_ops_context * octx) {
    प्रत्यागम op_क्रमय_common(octx, false);
}

पूर्णाङ्क op_शिखरम्_k(संरचना htp_ops_context * octx) {
    प्रत्यागम op_क्रमय_common(octx, true);
}
