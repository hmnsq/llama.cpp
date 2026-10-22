#अन्तर्भाव <string.h>
#अन्तर्भाव <stdlib.h>
#अन्तर्भाव <math.h>
#अन्तर्भाव <HAP_farf.h>
#अन्तर्भाव <HAP_perf.h>

#define GGML_COMMON_DECL_C
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

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif


स्थिर शून्य quicksort_मानानि_indices_asc(भिन्न * मानानि, int32_t * indices, पूर्णाङ्क वामः, पूर्णाङ्क दक्षिणः) {
    जाव (वामः < दक्षिणः) {
        भिन्न pivot = मानानि[वामः + (दक्षिणः - वामः) / 2];
        पूर्णाङ्क i = वामः;
        पूर्णाङ्क j = दक्षिणः;

        जाव (i <= j) {
            जाव (मानानि[i] < pivot) i++;
            जाव (मानानि[j] > pivot) j--;

            जइ (i <= j) {
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
        जइ (j - वामः < दक्षिणः - i) {
            जइ (वामः < j) quicksort_मानानि_indices_asc(मानानि, indices, वामः, j);
            वामः = i;
        } अण्णहा {
            जइ (i < दक्षिणः) quicksort_मानानि_indices_asc(मानानि, indices, i, दक्षिणः);
            दक्षिणः = j;
        }
    }
}

स्थिर शून्य quicksort_मानानि_indices_desc(भिन्न * मानानि, int32_t * indices, पूर्णाङ्क वामः, पूर्णाङ्क दक्षिणः) {
    जाव (वामः < दक्षिणः) {
        भिन्न pivot = मानानि[वामः + (दक्षिणः - वामः) / 2];
        पूर्णाङ्क i = वामः;
        पूर्णाङ्क j = दक्षिणः;

        जाव (i <= j) {
            जाव (मानानि[i] > pivot) i++;
            जाव (मानानि[j] < pivot) j--;

            जइ (i <= j) {
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
        जइ (j - वामः < दक्षिणः - i) {
            जइ (वामः < j) quicksort_मानानि_indices_desc(मानानि, indices, वामः, j);
            वामः = i;
        } अण्णहा {
            जइ (i < दक्षिणः) quicksort_मानानि_indices_desc(मानानि, indices, i, दक्षिणः);
            दक्षिणः = j;
        }
    }
}

स्थिर अन्तःस्थ uint32_t शिखरम्_k_max_मानम्_index(ध्रुव भिन्न * मानानि, uint32_t n, भिन्न * मानम्) {
    int32_t max_idx = 0;
    hvx_argmax_f32(मानानि, n, 0, मानम्, &max_idx);
    प्रत्यागम (uint32_t) max_idx;
}

स्थिर ध्रुव int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

स्थिर अन्तःस्थ शून्य init_indices_ramp(int32_t * indices_buf, uint32_t गणनम्, uint32_t base) {
    ध्रुव HVX_Vector ind_init_vec = Q6_Vw_vadd_VwVw(*(ध्रुव HVX_Vector *) argsort_ramp_lut, Q6_V_vsplat_R(base));
    ध्रुव HVX_Vector ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_Vector * indices_buf_vec = (HVX_Vector *) indices_buf;
    uint32_t अङ्कः_vecs = गणनम् / 32;
    HVX_Vector curr_ind_vec = ind_init_vec;
    प्रति (uint32_t j = 0; j < अङ्कः_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = गणनम् % 32;
    जइ (rem > 0) {
        hvx_vec_store_a((शून्य *) &indices_buf_vec[अङ्कः_vecs], rem * परिमाण(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य vec_cas(HVX_Vector * X_val, HVX_Vector * X_idx, HVX_Vector * Y_val, HVX_Vector * Y_idx, bool asc) {
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
स्थिर अन्तःस्थ शून्य bitonic_cas_32(HVX_Vector * V, HVX_Vector * I, पूर्णाङ्क d, HVX_VectorPred dir_mask, HVX_Vector idx_vec, HVX_Vector zero_vec) {
    HVX_VectorPred mask_वामः;
    HVX_Vector V_rot_वामः, V_rot_दक्षिणः;
    HVX_Vector I_rot_वामः, I_rot_दक्षिणः;

    जइ (d == 1) {
        mask_वामः = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_वामः = Q6_V_vror_VR(*V, 4);
        V_rot_दक्षिणः = Q6_V_vror_VR(*V, 124);
        I_rot_वामः = Q6_V_vror_VR(*I, 4);
        I_rot_दक्षिणः = Q6_V_vror_VR(*I, 124);
    } अण्णहा जइ (d == 2) {
        mask_वामः = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_वामः = Q6_V_vror_VR(*V, 8);
        V_rot_दक्षिणः = Q6_V_vror_VR(*V, 120);
        I_rot_वामः = Q6_V_vror_VR(*I, 8);
        I_rot_दक्षिणः = Q6_V_vror_VR(*I, 120);
    } अण्णहा जइ (d == 4) {
        mask_वामः = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_वामः = Q6_V_vror_VR(*V, 16);
        V_rot_दक्षिणः = Q6_V_vror_VR(*V, 112);
        I_rot_वामः = Q6_V_vror_VR(*I, 16);
        I_rot_दक्षिणः = Q6_V_vror_VR(*I, 112);
    } अण्णहा जइ (d == 8) {
        mask_वामः = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_वामः = Q6_V_vror_VR(*V, 32);
        V_rot_दक्षिणः = Q6_V_vror_VR(*V, 96);
        I_rot_वामः = Q6_V_vror_VR(*I, 32);
        I_rot_दक्षिणः = Q6_V_vror_VR(*I, 96);
    } अण्णहा {
        mask_वामः = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_वामः = Q6_V_vror_VR(*V, 64);
        V_rot_दक्षिणः = Q6_V_vror_VR(*V, 64);
        I_rot_वामः = Q6_V_vror_VR(*I, 64);
        I_rot_दक्षिणः = Q6_V_vror_VR(*I, 64);
    }

    HVX_Vector V_paired = Q6_V_vmux_QVV(mask_वामः, V_rot_वामः, V_rot_दक्षिणः);
    HVX_Vector I_paired = Q6_V_vmux_QVV(mask_वामः, I_rot_वामः, I_rot_दक्षिणः);

    HVX_VectorPred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_VectorPred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_VectorPred mask_दक्षिणः = Q6_Q_not_Q(mask_वामः);
    HVX_VectorPred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(mask_वामः, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, mask_दक्षिणः)
    );
    HVX_VectorPred Q_swap = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_mask, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_mask), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_swap, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_swap, I_paired, *I);
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य bitonic_sort_generic_hvx(uint8_t * मानानि, uint8_t * indices, पूर्णाङ्क K, bool asc_order) {
    HVX_Vector V[32];
    HVX_Vector I[32];

    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(ध्रुव HVX_Vector *) argsort_ramp_lut;

    प्रति (पूर्णाङ्क v = 0; v < K; v++) {
        V[v] = *(HVX_Vector *)(मानानि + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    पूर्णाङ्क M = 5;
    जाव ((1 << (M - 5)) < K) M++;

    प्रति (पूर्णाङ्क s = 1; s <= M; s++) {
        प्रति (पूर्णाङ्क stage_d = s - 1; stage_d >= 0; stage_d--) {
            पूर्णाङ्क d = 1 << stage_d;
            जइ (d >= 32) {
                पूर्णाङ्क v_dist = d / 32;
                प्रति (पूर्णाङ्क v1 = 0; v1 < K; v1++) {
                    जइ ((v1 & v_dist) == 0) {
                        पूर्णाङ्क v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_order;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } अण्णहा {
                जइ (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    प्रति (पूर्णाङ्क v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                } अण्णहा {
                    प्रति (पूर्णाङ्क v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_order;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    प्रति (पूर्णाङ्क v = 0; v < K; v++) {
        *(HVX_Vector *)(मानानि + v * 128)  = V[v];
        *(HVX_Vector *)(indices + v * 128) = I[v];
    }
}

स्थिर शून्य bitonic_sort_vtcm_desc(uint8_t * मानानि, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(ध्रुव HVX_Vector *) argsort_ramp_lut;

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    जइ (init_indices) {
        प्रति (uint32_t v = 0; v < n_vec; v++) {
            HVX_Vector idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_Vector *)(indices + v * 128) = idx;
        }
    }

    पूर्णाङ्क M = 5;
    जाव ((1u << (M - 5)) < n_vec) M++;

    प्रति (पूर्णाङ्क s = 1; s <= M; s++) {
        प्रति (पूर्णाङ्क stage_d = s - 1; stage_d >= 0; stage_d--) {
            पूर्णाङ्क d = 1 << stage_d;
            जइ (d >= 32) {
                uint32_t v_dist = d / 32;
                प्रति (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    जइ ((v1 & v_dist) == 0) {
                        uint32_t v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : false;

                        HVX_Vector Vv1 = *(HVX_Vector *)(मानानि + v1 * 128);
                        HVX_Vector Iv1 = *(HVX_Vector *)(indices + v1 * 128);
                        HVX_Vector Vv2 = *(HVX_Vector *)(मानानि + v2 * 128);
                        HVX_Vector Iv2 = *(HVX_Vector *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_Vector *)(मानानि + v1 * 128)  = Vv1;
                        *(HVX_Vector *)(indices + v1 * 128) = Iv1;
                        *(HVX_Vector *)(मानानि + v2 * 128)  = Vv2;
                        *(HVX_Vector *)(indices + v2 * 128) = Iv2;
                    }
                }
            } अण्णहा {
                जइ (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    प्रति (uint32_t v = 0; v < n_vec; v++) {
                        HVX_Vector Vv = *(HVX_Vector *)(मानानि + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(मानानि + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                } अण्णहा {
                    प्रति (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;

                        HVX_Vector Vv = *(HVX_Vector *)(मानानि + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(मानानि + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

स्थिर शून्य शिखरम्_k_select_tiled(dma_queue * dma_q, संरचना htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               भिन्न * मानानि_buf, int32_t * indices_buf,
                               भिन्न * out_मानानि, int32_t * out_indices,
                               uint32_t chunk_base) {
    ध्रुव uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_गणनम्    = n_tiles * k;
    uint32_t merge_n_vec        = hmx_ceil_div(candidate_गणनम्, 32);
    uint32_t merge_n_vec_pow2   = 1;
    जाव (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
    uint32_t merge_elems        = merge_n_vec_pow2 * 32;
    भिन्न * candidate_मानानि    = मानानि_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    प्रति (uint32_t offset = 0; offset < n; offset += tile_elems) {
        uint32_t tile_गणनम् = MIN(tile_elems, n - offset);
        dma_sync_read(dma_q, मानानि_buf, src_addr + (size_t) offset * परिमाण(भिन्न), tile_गणनम् * परिमाण(भिन्न));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
        जइ (tile_गणनम् < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (मानानि_buf + tile_गणनम्), -INFINITY, tile_elems - tile_गणनम्);
        }

        bitonic_sort_vtcm_desc((uint8_t *) मानानि_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_गणनम्);
        प्रति (uint32_t j = 0; j < tile_k; j++) {
            candidate_मानानि[candidate_pos] = मानानि_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) offset;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    जइ (merge_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_मानानि + candidate_pos), -INFINITY, merge_elems - candidate_pos);
        प्रति (uint32_t j = candidate_pos; j < merge_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_sort_vtcm_desc((uint8_t *) candidate_मानानि, (uint8_t *) candidate_indices, merge_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    प्रति (uint32_t j = 0; j < k; j++) {
        out_मानानि[j]  = candidate_मानानि[j];
        out_indices[j] = candidate_indices[j] + (int32_t) chunk_base;
    }
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य sort32_f32_hvx(uint8_t * मानानि, uint8_t * indices, गणना ggml_sort_order order) {
    bitonic_sort_generic_hvx(मानानि, indices, 1, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य sort64_f32_hvx(uint8_t * मानानि, uint8_t * indices, गणना ggml_sort_order order) {
    bitonic_sort_generic_hvx(मानानि, indices, 2, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य sort128_f32_hvx(uint8_t * मानानि, uint8_t * indices, गणना ggml_sort_order order) {
    bitonic_sort_generic_hvx(मानानि, indices, 4, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य sort256_f32_hvx(uint8_t * मानानि, uint8_t * indices, गणना ggml_sort_order order) {
    bitonic_sort_generic_hvx(मानानि, indices, 8, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य sort512_f32_hvx(uint8_t * मानानि, uint8_t * indices, गणना ggml_sort_order order) {
    bitonic_sort_generic_hvx(मानानि, indices, 16, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
स्थिर अन्तःस्थ शून्य sort1024_f32_hvx(uint8_t * मानानि, uint8_t * indices, गणना ggml_sort_order order) {
    bitonic_sort_generic_hvx(मानानि, indices, 32, order == GGML_SORT_ORDER_ASC);
}

स्थिर शून्य merge_runs(
    ध्रुव भिन्न * restrict in_val0, ध्रुव int32_t * restrict in_idx0, uint32_t n0,
    ध्रुव भिन्न * restrict in_val1, ध्रुव int32_t * restrict in_idx1, uint32_t n1,
    भिन्न * restrict out_val, int32_t * restrict out_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t out = 0;

    जइ (asc) {
        जाव (i0 < n0 && i1 < n1) {
            जइ (in_val0[i0] <= in_val1[i1]) {
                जइ (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } अण्णहा {
                जइ (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    } अण्णहा {
        जाव (i0 < n0 && i1 < n1) {
            जइ (in_val0[i0] >= in_val1[i1]) {
                जइ (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } अण्णहा {
                जइ (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    }

    जाव (i0 < n0) {
        जइ (out_val) out_val[out] = in_val0[i0];
        out_idx[out] = in_idx0[i0];
        i0++;
        out++;
    }

    जाव (i1 < n1) {
        जइ (out_val) out_val[out] = in_val1[i1];
        out_idx[out] = in_idx1[i1];
        i1++;
        out++;
    }
}

संरचना htp_sort_chunk_ctx {
    संरचना htp_ops_context *              octx;
    ध्रुव संरचना htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              real_गणनम्[HTP_MAX_NTHREADS];
};

स्थिर शून्य htp_sort_chunk_job(unsigned पूर्णाङ्क n, unsigned पूर्णाङ्क i, शून्य * data) {
    संरचना htp_sort_chunk_ctx * cctx = (संरचना htp_sort_chunk_ctx *) data;
    संरचना htp_ops_context * octx = cctx->octx;
    ध्रुव संरचना htp_sort_kernel_params * kparams = cctx->kparams;
    ध्रुव संरचना htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_base  = i * chunk_elems;

    uint32_t real_गणनम् = (chunk_base < ne00) ? MIN(chunk_elems, ne00 - chunk_base) : 0;
    cctx->real_गणनम्[i] = real_गणनम्;

    uint8_t * spad = cctx->vtcm_base + (size_t) kparams->phase1_slot_size * i;
    भिन्न *   मानानि_buf  = (भिन्न *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * परिमाण(भिन्न));

    भिन्न *   merge_मानानि  = (भिन्न *)   (cctx->vtcm_base + kparams->merge_मानानि_off);
    int32_t * merge_indices = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off);

    संरचना htp_thread_trace * tr = &octx->ctx->trace[i];
    ध्रुव dma_addr_t src_addr = src0->data + (size_t) chunk_base * परिमाण(भिन्न);

    जइ (kparams->is_शिखरम्_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        जइ (real_गणनम् == 0) {
            प्रति (uint32_t j = 0; j < local_k; j++) {
                merge_मानानि[i * local_k + j]  = -INFINITY;
                merge_indices[i * local_k + j] = 0;
            }
            प्रत्यागम;
        }

        जइ (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], मानानि_buf, src_addr, real_गणनम् * परिमाण(भिन्न));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            भिन्न max_मानम्;
            uint32_t max_index = शिखरम्_k_max_मानम्_index(मानानि_buf, real_गणनम्, &max_मानम्);
            merge_मानानि[i] = max_मानम्;
            merge_indices[i] = (int32_t) (max_index + chunk_base);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            प्रत्यागम;
        }

        जइ (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            शिखरम्_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_गणनम्, local_k,
                               मानानि_buf, indices_buf,
                               &merge_मानानि[i * local_k], &merge_indices[i * local_k],
                               chunk_base);
            प्रत्यागम;
        }

        dma_sync_read(octx->ctx->dma[i], मानानि_buf, src_addr, real_गणनम् * परिमाण(भिन्न));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        जइ (chunk_elems > real_गणनम्) {
            hvx_splat_f32_u((uint8_t *)(मानानि_buf + real_गणनम्), -INFINITY, chunk_elems - real_गणनम्);
        }

        bitonic_sort_vtcm_desc((uint8_t *) मानानि_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        प्रति (uint32_t j = 0; j < local_k; j++) {
            merge_मानानि[i * local_k + j]  = मानानि_buf[j];
            merge_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_base;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } अण्णहा {
        जइ (real_गणनम् > 0) {
            dma_sync_read(octx->ctx->dma[i], मानानि_buf, src_addr, real_गणनम् * परिमाण(भिन्न));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_गणनम्, chunk_base);

            जइ (kparams->order == GGML_SORT_ORDER_ASC) {
                quicksort_मानानि_indices_asc(मानानि_buf, indices_buf, 0, real_गणनम् - 1);
            } अण्णहा {
                quicksort_मानानि_indices_desc(मानानि_buf, indices_buf, 0, real_गणनम् - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

संरचना htp_argsort_merge_l1_ctx {
    संरचना htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

स्थिर शून्य htp_argsort_merge_l1_job(unsigned पूर्णाङ्क n, unsigned पूर्णाङ्क i, शून्य * data) {
    संरचना htp_argsort_merge_l1_ctx * mctx = (संरचना htp_argsort_merge_l1_ctx *) data;
    संरचना htp_sort_chunk_ctx * cctx = mctx->cctx;
    ध्रुव संरचना htp_sort_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_गणनम्[c0];
    uint32_t count1 = cctx->real_गणनम्[c1];

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    ध्रुव भिन्न *   val0 = (ध्रुव भिन्न *)   (cctx->vtcm_base + slot_size * c0);
    ध्रुव int32_t * idx0 = (ध्रुव int32_t *) (cctx->vtcm_base + slot_size * c0 + chunk_elems * परिमाण(भिन्न));
    ध्रुव भिन्न *   val1 = (ध्रुव भिन्न *)   (cctx->vtcm_base + slot_size * c1);
    ध्रुव int32_t * idx1 = (ध्रुव int32_t *) (cctx->vtcm_base + slot_size * c1 + chunk_elems * परिमाण(भिन्न));

    uint32_t out_offset = (2 * i) * chunk_elems;
    भिन्न *   out_val = (भिन्न *)   (cctx->vtcm_base + kparams->merge_मानानि_off)  + out_offset;
    int32_t * out_idx = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + out_offset;

    संरचना htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

संरचना htp_argsort_merge_l2_ctx {
    संरचना htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

स्थिर शून्य htp_argsort_merge_l2_job(unsigned पूर्णाङ्क n, unsigned पूर्णाङ्क i, शून्य * data) {
    संरचना htp_argsort_merge_l2_ctx * mctx = (संरचना htp_argsort_merge_l2_ctx *) data;
    संरचना htp_sort_chunk_ctx * cctx = mctx->cctx;
    ध्रुव संरचना htp_sort_kernel_params * kparams = cctx->kparams;

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    ध्रुव भिन्न *   val0 = (ध्रुव भिन्न *)   (cctx->vtcm_base + kparams->merge_मानानि_off)  + off0;
    ध्रुव int32_t * idx0 = (ध्रुव int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off0;
    ध्रुव भिन्न *   val1 = (ध्रुव भिन्न *)   (cctx->vtcm_base + kparams->merge_मानानि_off)  + off1;
    ध्रुव int32_t * idx1 = (ध्रुव int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off1;

    uint32_t count0 = cctx->real_गणनम्[4*i + 0] + cctx->real_गणनम्[4*i + 1];
    uint32_t count1 = cctx->real_गणनम्[4*i + 2] + cctx->real_गणनम्[4*i + 3];

    uint8_t * slot_base = cctx->vtcm_base + (4 * i) * slot_size;
    भिन्न *   out_val   = (भिन्न *) slot_base;
    int32_t * out_idx   = (int32_t *) (slot_base + 4 * chunk_elems * परिमाण(भिन्न));

    संरचना htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

संरचना htp_sort_multi_row_ctx {
    संरचना htp_ops_context *              octx;
    ध्रुव संरचना htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              nrows_per_thread;
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

स्थिर अन्तःस्थ शून्य step_row_coord(संरचना htp_tensor_row_coord * c, uint32_t ne01, uint32_t ne02) {
    c->i01++;
    जइ (c->i01 == ne01) {
        c->i01 = 0;
        c->i02++;
        जइ (c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

स्थिर अन्तःस्थ size_t coord_offset(संरचना htp_tensor_row_coord c, size_t nb1, size_t nb2, size_t nb3) {
    प्रत्यागम (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

स्थिर अन्तःस्थ शून्य compute_row_sort(
    भिन्न * cur_मानानि,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_शिखरम्_k,
    गणना ggml_sort_order order
) {
    जइ (is_शिखरम्_k) {
        जइ (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            विकल्प (ne00) {
                पक्ष 32:   sort32_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); भङ्ग;
                पक्ष 64:   sort64_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); भङ्ग;
                पक्ष 128:  sort128_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); भङ्ग;
                पक्ष 256:  sort256_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); भङ्ग;
                पक्ष 512:  sort512_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); भङ्ग;
                पक्ष 1024: sort1024_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); भङ्ग;
            }
        } अण्णहा {
            जइ (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_मानानि + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_sort_vtcm_desc((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } अण्णहा {
        जइ (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            विकल्प (ne00) {
                पक्ष 32:   sort32_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, order); भङ्ग;
                पक्ष 64:   sort64_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, order); भङ्ग;
                पक्ष 128:  sort128_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, order); भङ्ग;
                पक्ष 256:  sort256_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, order); भङ्ग;
                पक्ष 512:  sort512_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, order); भङ्ग;
                पक्ष 1024: sort1024_f32_hvx((uint8_t *) cur_मानानि, (uint8_t *) cur_indices, order); भङ्ग;
            }
        } अण्णहा {
            init_indices_ramp(cur_indices, ne00, 0);
            जइ (order == GGML_SORT_ORDER_ASC) {
                quicksort_मानानि_indices_asc(cur_मानानि, cur_indices, 0, ne00 - 1);
            } अण्णहा {
                quicksort_मानानि_indices_desc(cur_मानानि, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

स्थिर शून्य htp_sort_multi_row_job(unsigned पूर्णाङ्क n, unsigned पूर्णाङ्क ith, शून्य * data) {
    संरचना htp_sort_multi_row_ctx * mctx = (संरचना htp_sort_multi_row_ctx *) data;
    संरचना htp_ops_context * octx = mctx->octx;
    ध्रुव संरचना htp_sort_kernel_params * kparams = mctx->kparams;
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
    uint32_t row_end   = (uint32_t) kparams->row_end;
    uint32_t r_start   = row_start + ith * mctx->nrows_per_thread;
    uint32_t r_end     = MIN(r_start + mctx->nrows_per_thread, row_end);

    जइ (r_start >= r_end) प्रत्यागम;

    size_t spad_offset = ith * (size_t) (kparams->phase1_slot_size * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_base + spad_offset;
    size_t slot_size = (size_t) kparams->phase1_slot_size;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    भिन्न *   मानानि_buf  = (भिन्न *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * परिमाण(भिन्न));

    dma_queue * dma_q = octx->ctx->dma[ith];
    संरचना htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_शिखरम्_k = kparams->is_शिखरम्_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    गणना ggml_sort_order order = (गणना ggml_sort_order) kparams->order;

    ध्रुव uint32_t src_bytes = ne00 * परिमाण(भिन्न);
    ध्रुव uint32_t dst_bytes = (is_शिखरम्_k ? k : ne00) * परिमाण(int32_t);

    जइ (kparams->n_slots == 2) {
        संरचना htp_tensor_row_coord अविभाज्यः_coord = init_row_coord(r_start, ne01, ne02);
        प्रति (uint32_t r = r_start, spad_idx = 0; r < r_end && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * slot_size;
            भिन्न *   cur_मानानि  = (भिन्न *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * परिमाण(भिन्न));

            dma_addr_t dst_addr = dst->data + coord_offset(अविभाज्यः_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->data + coord_offset(अविभाज्यः_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_queue_प्रक्षिप(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 0);

            // Prefetch input row
            dma_queue_प्रक्षिप(dma_q, dma_make_data(cur_मानानि, src_addr),
                           src_bytes, src_bytes, src_bytes, 1);

            step_row_coord(&अविभाज्यः_coord, ne01, ne02);
        }

        संरचना htp_tensor_row_coord dst_coord = init_row_coord(r_start, ne01, ne02);
        संरचना htp_tensor_row_coord next_src_coord = init_row_coord(r_start + 2, ne01, ne02);

        प्रति (uint32_t r = r_start; r < r_end; r++) {
            uint32_t cur_slot = (r - r_start) & 1;
            uint8_t * cur_spad = thread_spad + cur_slot * slot_size;
            भिन्न *   cur_मानानि  = (भिन्न *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * परिमाण(भिन्न));

            // Wait for previous writeback of this slot to complete
            dma_queue_निष्कासय(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_queue_निष्कासय(dma_q);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(cur_मानानि, cur_indices, ne00, chunk_elems, is_शिखरम्_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->data + coord_offset(dst_coord, nb1, nb2, nb3);
            dma_queue_प्रक्षिप(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 1);
            step_row_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            ध्रुव uint32_t next_row = r + 2;
            जइ (next_row < r_end) {
                dma_addr_t next_src_addr = src0->data + coord_offset(next_src_coord, nb01, nb02, nb03);
                dma_queue_प्रक्षिप(dma_q, dma_make_data(cur_मानानि, next_src_addr),
                               src_bytes, src_bytes, src_bytes, 1);
                step_row_coord(&next_src_coord, ne01, ne02);
            }
        }

        dma_queue_flush(dma_q);
    } अण्णहा {
        संरचना htp_tensor_row_coord coord = init_row_coord(r_start, ne01, ne02);
        प्रति (uint32_t r = r_start; r < r_end; r++) {
            dma_addr_t src_addr = src0->data + coord_offset(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->data + coord_offset(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, मानानि_buf, src_addr, src_bytes);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(मानानि_buf, indices_buf, ne00, chunk_elems, is_शिखरम्_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_bytes);
            step_row_coord(&coord, ne01, ne02);
        }
    }
}

स्थिर पूर्णाङ्क op_sort_common(संरचना htp_ops_context * octx, bool is_शिखरम्_k) {
    जइ (octx->src[0]->type != HTP_TYPE_F32) {
        प्रत्यागम HTP_STATUS_NO_SUPPORT;
    }

    ध्रुव संरचना htp_tensor * src0 = octx->src[0];
    ध्रुव संरचना htp_tensor * dst  = octx->dst;

    जइ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        प्रत्यागम HTP_STATUS_NO_SUPPORT;
    }

    ध्रुव uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ध्रुव size_t dst_row_size = dst->ne[0] * परिमाण(int32_t);

    uint32_t row_start = 0;
    uint32_t row_end   = total_rows;
    जइ (octx->ctx->mdev.गणनम् > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, परिमाण(int32_t), (uint32_t) dst_row_size, &rows_per_chunk);
        ध्रुव संरचना htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.गणनम्, &octx->ctx->mdev.गणनम्_div);
        row_start = range.start;
        row_end   = range.start + range.गणनम्;
    }

    ध्रुव uint32_t nrows = row_end - row_start;
    जइ (nrows == 0) {
        प्रत्यागम HTP_STATUS_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    संरचना htp_sort_kernel_params kparams_local;
    ध्रुव संरचना htp_sort_kernel_params * kparams = (ध्रुव संरचना htp_sort_kernel_params *) octx->kernel_params;

    जइ (kparams->n_threads == 0) {
        संरचना htp_sort_vtcm_layout layout;
        जइ (!htp_sort_solve_layout(&layout, ne00, total_rows, k, octx->n_threads, octx->ctx->vtcm_size, is_शिखरम्_k)) {
            प्रत्यागम HTP_STATUS_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, परिमाण(kparams_local));
        kparams_local.n_threads         = (int32_t) layout.n_threads;
        kparams_local.total_rows        = (int32_t) total_rows;
        kparams_local.row_start         = (int32_t) row_start;
        kparams_local.row_end           = (int32_t) row_end;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.order             = is_शिखरम्_k ? GGML_SORT_ORDER_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_शिखरम्_k          = is_शिखरम्_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_size         = (int32_t) layout.total_bytes;
        kparams_local.phase1_slot_size  = (int32_t) layout.phase1_slot_size;
        kparams_local.merge_मानानि_off  = (int32_t) layout.merge_मानानि_off;
        kparams_local.merge_indices_off = (int32_t) layout.merge_indices_off;
        kparams_local.merge_elems       = (int32_t) layout.merge_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } अण्णहा जइ (octx->ctx->mdev.गणनम् > 1) {
        memcpy(&kparams_local, kparams, परिमाण(kparams_local));
        kparams_local.row_start = (int32_t) row_start;
        kparams_local.row_end   = (int32_t) row_end;
        kparams = &kparams_local;
    }

    जइ (octx->ctx->vtcm_size < (size_t) kparams->vtcm_size) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_size, octx->ctx->vtcm_size);
        प्रत्यागम HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    जइ (total_rows == 1 && kparams->n_chunks > 1) {
        संरचना htp_sort_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_base = vtcm_base;
        memset(cctx.real_गणनम्, 0, परिमाण(cctx.real_गणनम्));

        work_queue_run(octx->ctx->work_queue, htp_sort_chunk_job, &cctx, (uint32_t) kparams->n_chunks);

        जइ (is_शिखरम्_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            जइ (k == 1 && ne00 >= 128*1024) {
                भिन्न *   cand_vals = (भिन्न *)   (vtcm_base + kparams->merge_मानानि_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                भिन्न best_val = cand_vals[0];
                int32_t best_idx = cand_idxs[0];
                प्रति (int32_t c = 1; c < kparams->n_chunks; c++) {
                    जइ (cand_vals[c] > best_val) {
                        best_val = cand_vals[c];
                        best_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->data;
                dst_ptr[0] = best_idx;
                प्रत्यागम HTP_STATUS_OK;
            }

            uint32_t total_candidates = (uint32_t) kparams->n_chunks * local_k;
            भिन्न *   cand_vals = (भिन्न *)   (vtcm_base + kparams->merge_मानानि_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
            uint32_t merge_elems = (uint32_t) kparams->merge_elems;

            जइ (merge_elems > total_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + total_candidates), -INFINITY, merge_elems - total_candidates);
                प्रति (uint32_t j = total_candidates; j < merge_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_sort_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, merge_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->data, cand_idxs, k * परिमाण(int32_t));
        } अण्णहा {
            bool asc = kparams->order == GGML_SORT_ORDER_ASC;
            जइ (kparams->n_chunks == 8) {
                संरचना htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 4);

                संरचना htp_argsort_merge_l2_ctx l2ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l2_job, &l2ctx, 2);

                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ध्रुव भिन्न *   val0 = (ध्रुव भिन्न *) vtcm_base;
                ध्रुव int32_t * idx0 = (ध्रुव int32_t *) (vtcm_base + 4 * chunk_elems * परिमाण(भिन्न));
                ध्रुव भिन्न *   val1 = (ध्रुव भिन्न *) (vtcm_base + 4 * slot_size);
                ध्रुव int32_t * idx1 = (ध्रुव int32_t *) (vtcm_base + 4 * slot_size + 4 * chunk_elems * परिमाण(भिन्न));

                uint32_t n01 = cctx.real_गणनम्[0] + cctx.real_गणनम्[1] + cctx.real_गणनम्[2] + cctx.real_गणनम्[3];
                uint32_t n23 = cctx.real_गणनम्[4] + cctx.real_गणनम्[5] + cctx.real_गणनम्[6] + cctx.real_गणनम्[7];

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, n01, val1, idx1, n23, रिक्त, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * परिमाण(int32_t));
            } अण्णहा जइ (kparams->n_chunks == 4) {
                संरचना htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_गणनम्[0] + cctx.real_गणनम्[1];
                uint32_t n23 = cctx.real_गणनम्[2] + cctx.real_गणनम्[3];

                ध्रुव भिन्न *   val01 = (ध्रुव भिन्न *)   (vtcm_base + kparams->merge_मानानि_off);
                ध्रुव int32_t * idx01 = (ध्रुव int32_t *) (vtcm_base + kparams->merge_indices_off);
                ध्रुव भिन्न *   val23 = (ध्रुव भिन्न *)   (vtcm_base + kparams->merge_मानानि_off)  + 2 * chunk_elems;
                ध्रुव int32_t * idx23 = (ध्रुव int32_t *) (vtcm_base + kparams->merge_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_base;
                संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val01, idx01, n01, val23, idx23, n23, रिक्त, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * परिमाण(int32_t));
            } अण्णहा जइ (kparams->n_chunks == 2) {
                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ध्रुव भिन्न *   val0 = (ध्रुव भिन्न *)   (vtcm_base);
                ध्रुव int32_t * idx0 = (ध्रुव int32_t *) (vtcm_base + chunk_elems * परिमाण(भिन्न));
                ध्रुव भिन्न *   val1 = (ध्रुव भिन्न *)   (vtcm_base + slot_size);
                ध्रुव int32_t * idx1 = (ध्रुव int32_t *) (vtcm_base + slot_size + chunk_elems * परिमाण(भिन्न));

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                संरचना htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, cctx.real_गणनम्[0], val1, idx1, cctx.real_गणनम्[1], रिक्त, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * परिमाण(int32_t));
            } अण्णहा {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_base + chunk_elems * परिमाण(भिन्न));
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * परिमाण(int32_t));
            }
        }
        प्रत्यागम HTP_STATUS_OK;
    }

    संरचना htp_sort_multi_row_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_base        = vtcm_base;

    uint32_t dst_row_bytes = (uint32_t) dst_row_size;
    uint32_t rows_per_chunk = 0;
    htp_tensor_mdev_rows_per_chunk(dst, परिमाण(int32_t), dst_row_bytes, &rows_per_chunk);
    जइ (rows_per_chunk == 0) {
        rows_per_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_threads - 1) / (uint32_t) kparams->n_threads;
    जइ (rows_per_chunk > 1) {
        dr = hex_round_up(dr, rows_per_chunk);
    }
    mctx.nrows_per_thread = dr;

    work_queue_run(octx->ctx->work_queue, htp_sort_multi_row_job, &mctx, (uint32_t) kparams->n_threads);

    प्रत्यागम HTP_STATUS_OK;
}

पूर्णाङ्क op_argsort(संरचना htp_ops_context * octx) {
    प्रत्यागम op_sort_common(octx, false);
}

पूर्णाङ्क op_शिखरम्_k(संरचना htp_ops_context * octx) {
    प्रत्यागम op_sort_common(octx, true);
}
