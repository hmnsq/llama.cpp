#અન્તર્ભાવ <string.h>
#અન્તર્ભાવ <stdlib.h>
#અન્તર્ભાવ <math.h>
#અન્તર્ભાવ <HAP_farf.h>
#અન્તર્ભાવ <HAP_perf.h>

#define GGML_COMMON_DECL_C
#અન્તર્ભાવ "ggml-common.h"
#અન્તર્ભાવ "ggml.h"

#અન્તર્ભાવ "hvx-utils.h"
#અન્તર્ભાવ "hvx-copy.h"
#અન્તર્ભાવ "hvx-reduce.h"
#અન્તર્ભાવ "dma-queue.h"

#અન્તર્ભાવ "hex-common.h"
#અન્તર્ભાવ "htp-ctx.h"
#અન્તર્ભાવ "htp-ops.h"
#અન્તર્ભાવ "htp-tensor.h"
#અન્તર્ભાવ "argsort-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif


સ્થિર શૂન્ય quicksort_કિંમતો_indices_asc(દશાંશ * કિંમતો, int32_t * indices, પૂર્ણાંક ડાબું, પૂર્ણાંક જમણું) {
    જ્યારે (ડાબું < જમણું) {
        દશાંશ pivot = કિંમતો[ડાબું + (જમણું - ડાબું) / 2];
        પૂર્ણાંક i = ડાબું;
        પૂર્ણાંક j = જમણું;

        જ્યારે (i <= j) {
            જ્યારે (કિંમતો[i] < pivot) i++;
            જ્યારે (કિંમતો[j] > pivot) j--;

            જો (i <= j) {
                દશાંશ tmp_val = કિંમતો[i];
                કિંમતો[i] = કિંમતો[j];
                કિંમતો[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        જો (j - ડાબું < જમણું - i) {
            જો (ડાબું < j) quicksort_કિંમતો_indices_asc(કિંમતો, indices, ડાબું, j);
            ડાબું = i;
        } નહીંતર {
            જો (i < જમણું) quicksort_કિંમતો_indices_asc(કિંમતો, indices, i, જમણું);
            જમણું = j;
        }
    }
}

સ્થિર શૂન્ય quicksort_કિંમતો_indices_desc(દશાંશ * કિંમતો, int32_t * indices, પૂર્ણાંક ડાબું, પૂર્ણાંક જમણું) {
    જ્યારે (ડાબું < જમણું) {
        દશાંશ pivot = કિંમતો[ડાબું + (જમણું - ડાબું) / 2];
        પૂર્ણાંક i = ડાબું;
        પૂર્ણાંક j = જમણું;

        જ્યારે (i <= j) {
            જ્યારે (કિંમતો[i] > pivot) i++;
            જ્યારે (કિંમતો[j] < pivot) j--;

            જો (i <= j) {
                દશાંશ tmp_val = કિંમતો[i];
                કિંમતો[i] = કિંમતો[j];
                કિંમતો[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        જો (j - ડાબું < જમણું - i) {
            જો (ડાબું < j) quicksort_કિંમતો_indices_desc(કિંમતો, indices, ડાબું, j);
            ડાબું = i;
        } નહીંતર {
            જો (i < જમણું) quicksort_કિંમતો_indices_desc(કિંમતો, indices, i, જમણું);
            જમણું = j;
        }
    }
}

સ્થિર અંતઃરેખ uint32_t ટોચ_k_max_કિંમત_index(અચલ દશાંશ * કિંમતો, uint32_t n, દશાંશ * કિંમત) {
    int32_t max_idx = 0;
    hvx_argmax_f32(કિંમતો, n, 0, કિંમત, &max_idx);
    પાછા (uint32_t) max_idx;
}

સ્થિર અચલ int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

સ્થિર અંતઃરેખ શૂન્ય init_indices_ramp(int32_t * indices_buf, uint32_t ગણતરી, uint32_t base) {
    અચલ HVX_Vector ind_init_vec = Q6_Vw_vadd_VwVw(*(અચલ HVX_Vector *) argsort_ramp_lut, Q6_V_vsplat_R(base));
    અચલ HVX_Vector ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_Vector * indices_buf_vec = (HVX_Vector *) indices_buf;
    uint32_t અંક_vecs = ગણતરી / 32;
    HVX_Vector curr_ind_vec = ind_init_vec;
    માટે (uint32_t j = 0; j < અંક_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = ગણતરી % 32;
    જો (rem > 0) {
        hvx_vec_store_a((શૂન્ય *) &indices_buf_vec[અંક_vecs], rem * કદ(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
સ્થિર અંતઃરેખ શૂન્ય vec_cas(HVX_Vector * X_val, HVX_Vector * X_idx, HVX_Vector * Y_val, HVX_Vector * Y_idx, bool asc) {
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
સ્થિર અંતઃરેખ શૂન્ય bitonic_cas_32(HVX_Vector * V, HVX_Vector * I, પૂર્ણાંક d, HVX_VectorPred dir_mask, HVX_Vector idx_vec, HVX_Vector zero_vec) {
    HVX_VectorPred mask_ડાબું;
    HVX_Vector V_rot_ડાબું, V_rot_જમણું;
    HVX_Vector I_rot_ડાબું, I_rot_જમણું;

    જો (d == 1) {
        mask_ડાબું = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_ડાબું = Q6_V_vror_VR(*V, 4);
        V_rot_જમણું = Q6_V_vror_VR(*V, 124);
        I_rot_ડાબું = Q6_V_vror_VR(*I, 4);
        I_rot_જમણું = Q6_V_vror_VR(*I, 124);
    } નહીંતર જો (d == 2) {
        mask_ડાબું = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_ડાબું = Q6_V_vror_VR(*V, 8);
        V_rot_જમણું = Q6_V_vror_VR(*V, 120);
        I_rot_ડાબું = Q6_V_vror_VR(*I, 8);
        I_rot_જમણું = Q6_V_vror_VR(*I, 120);
    } નહીંતર જો (d == 4) {
        mask_ડાબું = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_ડાબું = Q6_V_vror_VR(*V, 16);
        V_rot_જમણું = Q6_V_vror_VR(*V, 112);
        I_rot_ડાબું = Q6_V_vror_VR(*I, 16);
        I_rot_જમણું = Q6_V_vror_VR(*I, 112);
    } નહીંતર જો (d == 8) {
        mask_ડાબું = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_ડાબું = Q6_V_vror_VR(*V, 32);
        V_rot_જમણું = Q6_V_vror_VR(*V, 96);
        I_rot_ડાબું = Q6_V_vror_VR(*I, 32);
        I_rot_જમણું = Q6_V_vror_VR(*I, 96);
    } નહીંતર {
        mask_ડાબું = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_ડાબું = Q6_V_vror_VR(*V, 64);
        V_rot_જમણું = Q6_V_vror_VR(*V, 64);
        I_rot_ડાબું = Q6_V_vror_VR(*I, 64);
        I_rot_જમણું = Q6_V_vror_VR(*I, 64);
    }

    HVX_Vector V_paired = Q6_V_vmux_QVV(mask_ડાબું, V_rot_ડાબું, V_rot_જમણું);
    HVX_Vector I_paired = Q6_V_vmux_QVV(mask_ડાબું, I_rot_ડાબું, I_rot_જમણું);

    HVX_VectorPred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_VectorPred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_VectorPred mask_જમણું = Q6_Q_not_Q(mask_ડાબું);
    HVX_VectorPred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(mask_ડાબું, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, mask_જમણું)
    );
    HVX_VectorPred Q_swap = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_mask, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_mask), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_swap, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_swap, I_paired, *I);
}

__attribute__((always_inline))
સ્થિર અંતઃરેખ શૂન્ય bitonic_sort_generic_hvx(uint8_t * કિંમતો, uint8_t * indices, પૂર્ણાંક K, bool asc_order) {
    HVX_Vector V[32];
    HVX_Vector I[32];

    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(અચલ HVX_Vector *) argsort_ramp_lut;

    માટે (પૂર્ણાંક v = 0; v < K; v++) {
        V[v] = *(HVX_Vector *)(કિંમતો + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    પૂર્ણાંક M = 5;
    જ્યારે ((1 << (M - 5)) < K) M++;

    માટે (પૂર્ણાંક s = 1; s <= M; s++) {
        માટે (પૂર્ણાંક stage_d = s - 1; stage_d >= 0; stage_d--) {
            પૂર્ણાંક d = 1 << stage_d;
            જો (d >= 32) {
                પૂર્ણાંક v_dist = d / 32;
                માટે (પૂર્ણાંક v1 = 0; v1 < K; v1++) {
                    જો ((v1 & v_dist) == 0) {
                        પૂર્ણાંક v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_order;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } નહીંતર {
                જો (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    માટે (પૂર્ણાંક v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                } નહીંતર {
                    માટે (પૂર્ણાંક v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_order;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    માટે (પૂર્ણાંક v = 0; v < K; v++) {
        *(HVX_Vector *)(કિંમતો + v * 128)  = V[v];
        *(HVX_Vector *)(indices + v * 128) = I[v];
    }
}

સ્થિર શૂન્ય bitonic_sort_vtcm_desc(uint8_t * કિંમતો, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(અચલ HVX_Vector *) argsort_ramp_lut;

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    જો (init_indices) {
        માટે (uint32_t v = 0; v < n_vec; v++) {
            HVX_Vector idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_Vector *)(indices + v * 128) = idx;
        }
    }

    પૂર્ણાંક M = 5;
    જ્યારે ((1u << (M - 5)) < n_vec) M++;

    માટે (પૂર્ણાંક s = 1; s <= M; s++) {
        માટે (પૂર્ણાંક stage_d = s - 1; stage_d >= 0; stage_d--) {
            પૂર્ણાંક d = 1 << stage_d;
            જો (d >= 32) {
                uint32_t v_dist = d / 32;
                માટે (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    જો ((v1 & v_dist) == 0) {
                        uint32_t v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : false;

                        HVX_Vector Vv1 = *(HVX_Vector *)(કિંમતો + v1 * 128);
                        HVX_Vector Iv1 = *(HVX_Vector *)(indices + v1 * 128);
                        HVX_Vector Vv2 = *(HVX_Vector *)(કિંમતો + v2 * 128);
                        HVX_Vector Iv2 = *(HVX_Vector *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_Vector *)(કિંમતો + v1 * 128)  = Vv1;
                        *(HVX_Vector *)(indices + v1 * 128) = Iv1;
                        *(HVX_Vector *)(કિંમતો + v2 * 128)  = Vv2;
                        *(HVX_Vector *)(indices + v2 * 128) = Iv2;
                    }
                }
            } નહીંતર {
                જો (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    માટે (uint32_t v = 0; v < n_vec; v++) {
                        HVX_Vector Vv = *(HVX_Vector *)(કિંમતો + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(કિંમતો + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                } નહીંતર {
                    માટે (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;

                        HVX_Vector Vv = *(HVX_Vector *)(કિંમતો + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(કિંમતો + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

સ્થિર શૂન્ય ટોચ_k_select_tiled(dma_queue * dma_q, સંરચના htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               દશાંશ * કિંમતો_buf, int32_t * indices_buf,
                               દશાંશ * out_કિંમતો, int32_t * out_indices,
                               uint32_t chunk_base) {
    અચલ uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_ગણતરી    = n_tiles * k;
    uint32_t merge_n_vec        = hmx_ceil_div(candidate_ગણતરી, 32);
    uint32_t merge_n_vec_pow2   = 1;
    જ્યારે (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
    uint32_t merge_elems        = merge_n_vec_pow2 * 32;
    દશાંશ * candidate_કિંમતો    = કિંમતો_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    માટે (uint32_t offset = 0; offset < n; offset += tile_elems) {
        uint32_t tile_ગણતરી = MIN(tile_elems, n - offset);
        dma_sync_read(dma_q, કિંમતો_buf, src_addr + (size_t) offset * કદ(દશાંશ), tile_ગણતરી * કદ(દશાંશ));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
        જો (tile_ગણતરી < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (કિંમતો_buf + tile_ગણતરી), -INFINITY, tile_elems - tile_ગણતરી);
        }

        bitonic_sort_vtcm_desc((uint8_t *) કિંમતો_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_ગણતરી);
        માટે (uint32_t j = 0; j < tile_k; j++) {
            candidate_કિંમતો[candidate_pos] = કિંમતો_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) offset;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    જો (merge_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_કિંમતો + candidate_pos), -INFINITY, merge_elems - candidate_pos);
        માટે (uint32_t j = candidate_pos; j < merge_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_sort_vtcm_desc((uint8_t *) candidate_કિંમતો, (uint8_t *) candidate_indices, merge_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    માટે (uint32_t j = 0; j < k; j++) {
        out_કિંમતો[j]  = candidate_કિંમતો[j];
        out_indices[j] = candidate_indices[j] + (int32_t) chunk_base;
    }
}

__attribute__((always_inline))
સ્થિર અંતઃરેખ શૂન્ય sort32_f32_hvx(uint8_t * કિંમતો, uint8_t * indices, ગણના ggml_sort_order order) {
    bitonic_sort_generic_hvx(કિંમતો, indices, 1, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
સ્થિર અંતઃરેખ શૂન્ય sort64_f32_hvx(uint8_t * કિંમતો, uint8_t * indices, ગણના ggml_sort_order order) {
    bitonic_sort_generic_hvx(કિંમતો, indices, 2, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
સ્થિર અંતઃરેખ શૂન્ય sort128_f32_hvx(uint8_t * કિંમતો, uint8_t * indices, ગણના ggml_sort_order order) {
    bitonic_sort_generic_hvx(કિંમતો, indices, 4, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
સ્થિર અંતઃરેખ શૂન્ય sort256_f32_hvx(uint8_t * કિંમતો, uint8_t * indices, ગણના ggml_sort_order order) {
    bitonic_sort_generic_hvx(કિંમતો, indices, 8, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
સ્થિર અંતઃરેખ શૂન્ય sort512_f32_hvx(uint8_t * કિંમતો, uint8_t * indices, ગણના ggml_sort_order order) {
    bitonic_sort_generic_hvx(કિંમતો, indices, 16, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
સ્થિર અંતઃરેખ શૂન્ય sort1024_f32_hvx(uint8_t * કિંમતો, uint8_t * indices, ગણના ggml_sort_order order) {
    bitonic_sort_generic_hvx(કિંમતો, indices, 32, order == GGML_SORT_ORDER_ASC);
}

સ્થિર શૂન્ય merge_runs(
    અચલ દશાંશ * restrict in_val0, અચલ int32_t * restrict in_idx0, uint32_t n0,
    અચલ દશાંશ * restrict in_val1, અચલ int32_t * restrict in_idx1, uint32_t n1,
    દશાંશ * restrict out_val, int32_t * restrict out_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t out = 0;

    જો (asc) {
        જ્યારે (i0 < n0 && i1 < n1) {
            જો (in_val0[i0] <= in_val1[i1]) {
                જો (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } નહીંતર {
                જો (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    } નહીંતર {
        જ્યારે (i0 < n0 && i1 < n1) {
            જો (in_val0[i0] >= in_val1[i1]) {
                જો (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } નહીંતર {
                જો (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    }

    જ્યારે (i0 < n0) {
        જો (out_val) out_val[out] = in_val0[i0];
        out_idx[out] = in_idx0[i0];
        i0++;
        out++;
    }

    જ્યારે (i1 < n1) {
        જો (out_val) out_val[out] = in_val1[i1];
        out_idx[out] = in_idx1[i1];
        i1++;
        out++;
    }
}

સંરચના htp_sort_chunk_ctx {
    સંરચના htp_ops_context *              octx;
    અચલ સંરચના htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              real_ગણતરી[HTP_MAX_NTHREADS];
};

સ્થિર શૂન્ય htp_sort_chunk_job(unsigned પૂર્ણાંક n, unsigned પૂર્ણાંક i, શૂન્ય * data) {
    સંરચના htp_sort_chunk_ctx * cctx = (સંરચના htp_sort_chunk_ctx *) data;
    સંરચના htp_ops_context * octx = cctx->octx;
    અચલ સંરચના htp_sort_kernel_params * kparams = cctx->kparams;
    અચલ સંરચના htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_base  = i * chunk_elems;

    uint32_t real_ગણતરી = (chunk_base < ne00) ? MIN(chunk_elems, ne00 - chunk_base) : 0;
    cctx->real_ગણતરી[i] = real_ગણતરી;

    uint8_t * spad = cctx->vtcm_base + (size_t) kparams->phase1_slot_size * i;
    દશાંશ *   કિંમતો_buf  = (દશાંશ *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * કદ(દશાંશ));

    દશાંશ *   merge_કિંમતો  = (દશાંશ *)   (cctx->vtcm_base + kparams->merge_કિંમતો_off);
    int32_t * merge_indices = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off);

    સંરચના htp_thread_trace * tr = &octx->ctx->trace[i];
    અચલ dma_addr_t src_addr = src0->data + (size_t) chunk_base * કદ(દશાંશ);

    જો (kparams->is_ટોચ_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        જો (real_ગણતરી == 0) {
            માટે (uint32_t j = 0; j < local_k; j++) {
                merge_કિંમતો[i * local_k + j]  = -INFINITY;
                merge_indices[i * local_k + j] = 0;
            }
            પાછા;
        }

        જો (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], કિંમતો_buf, src_addr, real_ગણતરી * કદ(દશાંશ));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            દશાંશ max_કિંમત;
            uint32_t max_index = ટોચ_k_max_કિંમત_index(કિંમતો_buf, real_ગણતરી, &max_કિંમત);
            merge_કિંમતો[i] = max_કિંમત;
            merge_indices[i] = (int32_t) (max_index + chunk_base);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            પાછા;
        }

        જો (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            ટોચ_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_ગણતરી, local_k,
                               કિંમતો_buf, indices_buf,
                               &merge_કિંમતો[i * local_k], &merge_indices[i * local_k],
                               chunk_base);
            પાછા;
        }

        dma_sync_read(octx->ctx->dma[i], કિંમતો_buf, src_addr, real_ગણતરી * કદ(દશાંશ));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        જો (chunk_elems > real_ગણતરી) {
            hvx_splat_f32_u((uint8_t *)(કિંમતો_buf + real_ગણતરી), -INFINITY, chunk_elems - real_ગણતરી);
        }

        bitonic_sort_vtcm_desc((uint8_t *) કિંમતો_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        માટે (uint32_t j = 0; j < local_k; j++) {
            merge_કિંમતો[i * local_k + j]  = કિંમતો_buf[j];
            merge_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_base;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } નહીંતર {
        જો (real_ગણતરી > 0) {
            dma_sync_read(octx->ctx->dma[i], કિંમતો_buf, src_addr, real_ગણતરી * કદ(દશાંશ));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_ગણતરી, chunk_base);

            જો (kparams->order == GGML_SORT_ORDER_ASC) {
                quicksort_કિંમતો_indices_asc(કિંમતો_buf, indices_buf, 0, real_ગણતરી - 1);
            } નહીંતર {
                quicksort_કિંમતો_indices_desc(કિંમતો_buf, indices_buf, 0, real_ગણતરી - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

સંરચના htp_argsort_merge_l1_ctx {
    સંરચના htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

સ્થિર શૂન્ય htp_argsort_merge_l1_job(unsigned પૂર્ણાંક n, unsigned પૂર્ણાંક i, શૂન્ય * data) {
    સંરચના htp_argsort_merge_l1_ctx * mctx = (સંરચના htp_argsort_merge_l1_ctx *) data;
    સંરચના htp_sort_chunk_ctx * cctx = mctx->cctx;
    અચલ સંરચના htp_sort_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_ગણતરી[c0];
    uint32_t count1 = cctx->real_ગણતરી[c1];

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    અચલ દશાંશ *   val0 = (અચલ દશાંશ *)   (cctx->vtcm_base + slot_size * c0);
    અચલ int32_t * idx0 = (અચલ int32_t *) (cctx->vtcm_base + slot_size * c0 + chunk_elems * કદ(દશાંશ));
    અચલ દશાંશ *   val1 = (અચલ દશાંશ *)   (cctx->vtcm_base + slot_size * c1);
    અચલ int32_t * idx1 = (અચલ int32_t *) (cctx->vtcm_base + slot_size * c1 + chunk_elems * કદ(દશાંશ));

    uint32_t out_offset = (2 * i) * chunk_elems;
    દશાંશ *   out_val = (દશાંશ *)   (cctx->vtcm_base + kparams->merge_કિંમતો_off)  + out_offset;
    int32_t * out_idx = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + out_offset;

    સંરચના htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

સંરચના htp_argsort_merge_l2_ctx {
    સંરચના htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

સ્થિર શૂન્ય htp_argsort_merge_l2_job(unsigned પૂર્ણાંક n, unsigned પૂર્ણાંક i, શૂન્ય * data) {
    સંરચના htp_argsort_merge_l2_ctx * mctx = (સંરચના htp_argsort_merge_l2_ctx *) data;
    સંરચના htp_sort_chunk_ctx * cctx = mctx->cctx;
    અચલ સંરચના htp_sort_kernel_params * kparams = cctx->kparams;

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    અચલ દશાંશ *   val0 = (અચલ દશાંશ *)   (cctx->vtcm_base + kparams->merge_કિંમતો_off)  + off0;
    અચલ int32_t * idx0 = (અચલ int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off0;
    અચલ દશાંશ *   val1 = (અચલ દશાંશ *)   (cctx->vtcm_base + kparams->merge_કિંમતો_off)  + off1;
    અચલ int32_t * idx1 = (અચલ int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off1;

    uint32_t count0 = cctx->real_ગણતરી[4*i + 0] + cctx->real_ગણતરી[4*i + 1];
    uint32_t count1 = cctx->real_ગણતરી[4*i + 2] + cctx->real_ગણતરી[4*i + 3];

    uint8_t * slot_base = cctx->vtcm_base + (4 * i) * slot_size;
    દશાંશ *   out_val   = (દશાંશ *) slot_base;
    int32_t * out_idx   = (int32_t *) (slot_base + 4 * chunk_elems * કદ(દશાંશ));

    સંરચના htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

સંરચના htp_sort_multi_row_ctx {
    સંરચના htp_ops_context *              octx;
    અચલ સંરચના htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              nrows_per_thread;
};

સંરચના htp_tensor_row_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

સ્થિર અંતઃરેખ સંરચના htp_tensor_row_coord init_row_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    પાછા (સંરચના htp_tensor_row_coord){ i01, i02, i03 };
}

સ્થિર અંતઃરેખ શૂન્ય step_row_coord(સંરચના htp_tensor_row_coord * c, uint32_t ne01, uint32_t ne02) {
    c->i01++;
    જો (c->i01 == ne01) {
        c->i01 = 0;
        c->i02++;
        જો (c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

સ્થિર અંતઃરેખ size_t coord_offset(સંરચના htp_tensor_row_coord c, size_t nb1, size_t nb2, size_t nb3) {
    પાછા (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

સ્થિર અંતઃરેખ શૂન્ય compute_row_sort(
    દશાંશ * cur_કિંમતો,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_ટોચ_k,
    ગણના ggml_sort_order order
) {
    જો (is_ટોચ_k) {
        જો (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            પસંદગી (ne00) {
                સ્થિતિ 32:   sort32_f32_hvx((uint8_t *) cur_કિંમતો, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); તોડો;
                સ્થિતિ 64:   sort64_f32_hvx((uint8_t *) cur_કિંમતો, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); તોડો;
                સ્થિતિ 128:  sort128_f32_hvx((uint8_t *) cur_કિંમતો, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); તોડો;
                સ્થિતિ 256:  sort256_f32_hvx((uint8_t *) cur_કિંમતો, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); તોડો;
                સ્થિતિ 512:  sort512_f32_hvx((uint8_t *) cur_કિંમતો, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); તોડો;
                સ્થિતિ 1024: sort1024_f32_hvx((uint8_t *) cur_કિંમતો, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); તોડો;
            }
        } નહીંતર {
            જો (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_કિંમતો + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_sort_vtcm_desc((uint8_t *) cur_કિંમતો, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } નહીંતર {
        જો (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            પસંદગી (ne00) {
                સ્થિતિ 32:   sort32_f32_hvx((uint8_t *) cur_કિંમતો, (uint8_t *) cur_indices, order); તોડો;
                સ્થિતિ 64:   sort64_f32_hvx((uint8_t *) cur_કિંમતો, (uint8_t *) cur_indices, order); તોડો;
                સ્થિતિ 128:  sort128_f32_hvx((uint8_t *) cur_કિંમતો, (uint8_t *) cur_indices, order); તોડો;
                સ્થિતિ 256:  sort256_f32_hvx((uint8_t *) cur_કિંમતો, (uint8_t *) cur_indices, order); તોડો;
                સ્થિતિ 512:  sort512_f32_hvx((uint8_t *) cur_કિંમતો, (uint8_t *) cur_indices, order); તોડો;
                સ્થિતિ 1024: sort1024_f32_hvx((uint8_t *) cur_કિંમતો, (uint8_t *) cur_indices, order); તોડો;
            }
        } નહીંતર {
            init_indices_ramp(cur_indices, ne00, 0);
            જો (order == GGML_SORT_ORDER_ASC) {
                quicksort_કિંમતો_indices_asc(cur_કિંમતો, cur_indices, 0, ne00 - 1);
            } નહીંતર {
                quicksort_કિંમતો_indices_desc(cur_કિંમતો, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

સ્થિર શૂન્ય htp_sort_multi_row_job(unsigned પૂર્ણાંક n, unsigned પૂર્ણાંક ith, શૂન્ય * data) {
    સંરચના htp_sort_multi_row_ctx * mctx = (સંરચના htp_sort_multi_row_ctx *) data;
    સંરચના htp_ops_context * octx = mctx->octx;
    અચલ સંરચના htp_sort_kernel_params * kparams = mctx->kparams;
    અચલ સંરચના htp_tensor * src0 = octx->src[0];
    અચલ સંરચના htp_tensor * dst  = octx->dst;

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

    જો (r_start >= r_end) પાછા;

    size_t spad_offset = ith * (size_t) (kparams->phase1_slot_size * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_base + spad_offset;
    size_t slot_size = (size_t) kparams->phase1_slot_size;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    દશાંશ *   કિંમતો_buf  = (દશાંશ *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * કદ(દશાંશ));

    dma_queue * dma_q = octx->ctx->dma[ith];
    સંરચના htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_ટોચ_k = kparams->is_ટોચ_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    ગણના ggml_sort_order order = (ગણના ggml_sort_order) kparams->order;

    અચલ uint32_t src_bytes = ne00 * કદ(દશાંશ);
    અચલ uint32_t dst_bytes = (is_ટોચ_k ? k : ne00) * કદ(int32_t);

    જો (kparams->n_slots == 2) {
        સંરચના htp_tensor_row_coord અવિભાજ્ય_coord = init_row_coord(r_start, ne01, ne02);
        માટે (uint32_t r = r_start, spad_idx = 0; r < r_end && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * slot_size;
            દશાંશ *   cur_કિંમતો  = (દશાંશ *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * કદ(દશાંશ));

            dma_addr_t dst_addr = dst->data + coord_offset(અવિભાજ્ય_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->data + coord_offset(અવિભાજ્ય_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_queue_ધકેલો(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 0);

            // Prefetch input row
            dma_queue_ધકેલો(dma_q, dma_make_data(cur_કિંમતો, src_addr),
                           src_bytes, src_bytes, src_bytes, 1);

            step_row_coord(&અવિભાજ્ય_coord, ne01, ne02);
        }

        સંરચના htp_tensor_row_coord dst_coord = init_row_coord(r_start, ne01, ne02);
        સંરચના htp_tensor_row_coord next_src_coord = init_row_coord(r_start + 2, ne01, ne02);

        માટે (uint32_t r = r_start; r < r_end; r++) {
            uint32_t cur_slot = (r - r_start) & 1;
            uint8_t * cur_spad = thread_spad + cur_slot * slot_size;
            દશાંશ *   cur_કિંમતો  = (દશાંશ *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * કદ(દશાંશ));

            // Wait for previous writeback of this slot to complete
            dma_queue_કાઢો(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_queue_કાઢો(dma_q);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(cur_કિંમતો, cur_indices, ne00, chunk_elems, is_ટોચ_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->data + coord_offset(dst_coord, nb1, nb2, nb3);
            dma_queue_ધકેલો(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 1);
            step_row_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            અચલ uint32_t next_row = r + 2;
            જો (next_row < r_end) {
                dma_addr_t next_src_addr = src0->data + coord_offset(next_src_coord, nb01, nb02, nb03);
                dma_queue_ધકેલો(dma_q, dma_make_data(cur_કિંમતો, next_src_addr),
                               src_bytes, src_bytes, src_bytes, 1);
                step_row_coord(&next_src_coord, ne01, ne02);
            }
        }

        dma_queue_flush(dma_q);
    } નહીંતર {
        સંરચના htp_tensor_row_coord coord = init_row_coord(r_start, ne01, ne02);
        માટે (uint32_t r = r_start; r < r_end; r++) {
            dma_addr_t src_addr = src0->data + coord_offset(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->data + coord_offset(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, કિંમતો_buf, src_addr, src_bytes);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(કિંમતો_buf, indices_buf, ne00, chunk_elems, is_ટોચ_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_bytes);
            step_row_coord(&coord, ne01, ne02);
        }
    }
}

સ્થિર પૂર્ણાંક op_sort_common(સંરચના htp_ops_context * octx, bool is_ટોચ_k) {
    જો (octx->src[0]->type != HTP_TYPE_F32) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    અચલ સંરચના htp_tensor * src0 = octx->src[0];
    અચલ સંરચના htp_tensor * dst  = octx->dst;

    જો (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        પાછા HTP_STATUS_NO_SUPPORT;
    }

    અચલ uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    અચલ size_t dst_row_size = dst->ne[0] * કદ(int32_t);

    uint32_t row_start = 0;
    uint32_t row_end   = total_rows;
    જો (octx->ctx->mdev.ગણતરી > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, કદ(int32_t), (uint32_t) dst_row_size, &rows_per_chunk);
        અચલ સંરચના htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ગણતરી, &octx->ctx->mdev.ગણતરી_div);
        row_start = range.start;
        row_end   = range.start + range.ગણતરી;
    }

    અચલ uint32_t nrows = row_end - row_start;
    જો (nrows == 0) {
        પાછા HTP_STATUS_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    સંરચના htp_sort_kernel_params kparams_local;
    અચલ સંરચના htp_sort_kernel_params * kparams = (અચલ સંરચના htp_sort_kernel_params *) octx->kernel_params;

    જો (kparams->n_threads == 0) {
        સંરચના htp_sort_vtcm_layout layout;
        જો (!htp_sort_solve_layout(&layout, ne00, total_rows, k, octx->n_threads, octx->ctx->vtcm_size, is_ટોચ_k)) {
            પાછા HTP_STATUS_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, કદ(kparams_local));
        kparams_local.n_threads         = (int32_t) layout.n_threads;
        kparams_local.total_rows        = (int32_t) total_rows;
        kparams_local.row_start         = (int32_t) row_start;
        kparams_local.row_end           = (int32_t) row_end;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.order             = is_ટોચ_k ? GGML_SORT_ORDER_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_ટોચ_k          = is_ટોચ_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_size         = (int32_t) layout.total_bytes;
        kparams_local.phase1_slot_size  = (int32_t) layout.phase1_slot_size;
        kparams_local.merge_કિંમતો_off  = (int32_t) layout.merge_કિંમતો_off;
        kparams_local.merge_indices_off = (int32_t) layout.merge_indices_off;
        kparams_local.merge_elems       = (int32_t) layout.merge_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } નહીંતર જો (octx->ctx->mdev.ગણતરી > 1) {
        memcpy(&kparams_local, kparams, કદ(kparams_local));
        kparams_local.row_start = (int32_t) row_start;
        kparams_local.row_end   = (int32_t) row_end;
        kparams = &kparams_local;
    }

    જો (octx->ctx->vtcm_size < (size_t) kparams->vtcm_size) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_size, octx->ctx->vtcm_size);
        પાછા HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    જો (total_rows == 1 && kparams->n_chunks > 1) {
        સંરચના htp_sort_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_base = vtcm_base;
        memset(cctx.real_ગણતરી, 0, કદ(cctx.real_ગણતરી));

        work_queue_run(octx->ctx->work_queue, htp_sort_chunk_job, &cctx, (uint32_t) kparams->n_chunks);

        જો (is_ટોચ_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            જો (k == 1 && ne00 >= 128*1024) {
                દશાંશ *   cand_vals = (દશાંશ *)   (vtcm_base + kparams->merge_કિંમતો_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                દશાંશ best_val = cand_vals[0];
                int32_t best_idx = cand_idxs[0];
                માટે (int32_t c = 1; c < kparams->n_chunks; c++) {
                    જો (cand_vals[c] > best_val) {
                        best_val = cand_vals[c];
                        best_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->data;
                dst_ptr[0] = best_idx;
                પાછા HTP_STATUS_OK;
            }

            uint32_t total_candidates = (uint32_t) kparams->n_chunks * local_k;
            દશાંશ *   cand_vals = (દશાંશ *)   (vtcm_base + kparams->merge_કિંમતો_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
            uint32_t merge_elems = (uint32_t) kparams->merge_elems;

            જો (merge_elems > total_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + total_candidates), -INFINITY, merge_elems - total_candidates);
                માટે (uint32_t j = total_candidates; j < merge_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            સંરચના htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_sort_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, merge_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->data, cand_idxs, k * કદ(int32_t));
        } નહીંતર {
            bool asc = kparams->order == GGML_SORT_ORDER_ASC;
            જો (kparams->n_chunks == 8) {
                સંરચના htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 4);

                સંરચના htp_argsort_merge_l2_ctx l2ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l2_job, &l2ctx, 2);

                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                અચલ દશાંશ *   val0 = (અચલ દશાંશ *) vtcm_base;
                અચલ int32_t * idx0 = (અચલ int32_t *) (vtcm_base + 4 * chunk_elems * કદ(દશાંશ));
                અચલ દશાંશ *   val1 = (અચલ દશાંશ *) (vtcm_base + 4 * slot_size);
                અચલ int32_t * idx1 = (અચલ int32_t *) (vtcm_base + 4 * slot_size + 4 * chunk_elems * કદ(દશાંશ));

                uint32_t n01 = cctx.real_ગણતરી[0] + cctx.real_ગણતરી[1] + cctx.real_ગણતરી[2] + cctx.real_ગણતરી[3];
                uint32_t n23 = cctx.real_ગણતરી[4] + cctx.real_ગણતરી[5] + cctx.real_ગણતરી[6] + cctx.real_ગણતરી[7];

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                સંરચના htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, n01, val1, idx1, n23, ખાલી, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * કદ(int32_t));
            } નહીંતર જો (kparams->n_chunks == 4) {
                સંરચના htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_ગણતરી[0] + cctx.real_ગણતરી[1];
                uint32_t n23 = cctx.real_ગણતરી[2] + cctx.real_ગણતરી[3];

                અચલ દશાંશ *   val01 = (અચલ દશાંશ *)   (vtcm_base + kparams->merge_કિંમતો_off);
                અચલ int32_t * idx01 = (અચલ int32_t *) (vtcm_base + kparams->merge_indices_off);
                અચલ દશાંશ *   val23 = (અચલ દશાંશ *)   (vtcm_base + kparams->merge_કિંમતો_off)  + 2 * chunk_elems;
                અચલ int32_t * idx23 = (અચલ int32_t *) (vtcm_base + kparams->merge_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_base;
                સંરચના htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val01, idx01, n01, val23, idx23, n23, ખાલી, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * કદ(int32_t));
            } નહીંતર જો (kparams->n_chunks == 2) {
                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                અચલ દશાંશ *   val0 = (અચલ દશાંશ *)   (vtcm_base);
                અચલ int32_t * idx0 = (અચલ int32_t *) (vtcm_base + chunk_elems * કદ(દશાંશ));
                અચલ દશાંશ *   val1 = (અચલ દશાંશ *)   (vtcm_base + slot_size);
                અચલ int32_t * idx1 = (અચલ int32_t *) (vtcm_base + slot_size + chunk_elems * કદ(દશાંશ));

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                સંરચના htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, cctx.real_ગણતરી[0], val1, idx1, cctx.real_ગણતરી[1], ખાલી, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * કદ(int32_t));
            } નહીંતર {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_base + chunk_elems * કદ(દશાંશ));
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * કદ(int32_t));
            }
        }
        પાછા HTP_STATUS_OK;
    }

    સંરચના htp_sort_multi_row_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_base        = vtcm_base;

    uint32_t dst_row_bytes = (uint32_t) dst_row_size;
    uint32_t rows_per_chunk = 0;
    htp_tensor_mdev_rows_per_chunk(dst, કદ(int32_t), dst_row_bytes, &rows_per_chunk);
    જો (rows_per_chunk == 0) {
        rows_per_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_threads - 1) / (uint32_t) kparams->n_threads;
    જો (rows_per_chunk > 1) {
        dr = hex_round_up(dr, rows_per_chunk);
    }
    mctx.nrows_per_thread = dr;

    work_queue_run(octx->ctx->work_queue, htp_sort_multi_row_job, &mctx, (uint32_t) kparams->n_threads);

    પાછા HTP_STATUS_OK;
}

પૂર્ણાંક op_argsort(સંરચના htp_ops_context * octx) {
    પાછા op_sort_common(octx, false);
}

પૂર્ણાંક op_ટોચ_k(સંરચના htp_ops_context * octx) {
    પાછા op_sort_common(octx, true);
}
