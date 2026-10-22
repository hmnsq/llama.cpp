#அந்தர்பாவ <string.h>
#அந்தர்பாவ <stdlib.h>
#அந்தர்பாவ <math.h>
#அந்தர்பாவ <HAP_farf.h>
#அந்தர்பாவ <HAP_perf.h>

#define GGML_COMMON_DECL_C
#அந்தர்பாவ "ggml-common.h"
#அந்தர்பாவ "ggml.h"

#அந்தர்பாவ "hvx-utils.h"
#அந்தர்பாவ "hvx-copy.h"
#அந்தர்பாவ "hvx-reduce.h"
#அந்தர்பாவ "dma-queue.h"

#அந்தர்பாவ "hex-common.h"
#அந்தர்பாவ "htp-ctx.h"
#அந்தர்பாவ "htp-ops.h"
#அந்தர்பாவ "htp-tensor.h"
#அந்தர்பாவ "argsort-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif


நிலையான வெறுமை quicksort_மதிப்புகள்_indices_asc(தசம * மதிப்புகள், int32_t * indices, முழு இடது, முழு வலது) {
    போது (இடது < வலது) {
        தசம pivot = மதிப்புகள்[இடது + (வலது - இடது) / 2];
        முழு i = இடது;
        முழு j = வலது;

        போது (i <= j) {
            போது (மதிப்புகள்[i] < pivot) i++;
            போது (மதிப்புகள்[j] > pivot) j--;

            எனில் (i <= j) {
                தசம tmp_val = மதிப்புகள்[i];
                மதிப்புகள்[i] = மதிப்புகள்[j];
                மதிப்புகள்[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        எனில் (j - இடது < வலது - i) {
            எனில் (இடது < j) quicksort_மதிப்புகள்_indices_asc(மதிப்புகள், indices, இடது, j);
            இடது = i;
        } இல்லையெனில் {
            எனில் (i < வலது) quicksort_மதிப்புகள்_indices_asc(மதிப்புகள், indices, i, வலது);
            வலது = j;
        }
    }
}

நிலையான வெறுமை quicksort_மதிப்புகள்_indices_desc(தசம * மதிப்புகள், int32_t * indices, முழு இடது, முழு வலது) {
    போது (இடது < வலது) {
        தசம pivot = மதிப்புகள்[இடது + (வலது - இடது) / 2];
        முழு i = இடது;
        முழு j = வலது;

        போது (i <= j) {
            போது (மதிப்புகள்[i] > pivot) i++;
            போது (மதிப்புகள்[j] < pivot) j--;

            எனில் (i <= j) {
                தசம tmp_val = மதிப்புகள்[i];
                மதிப்புகள்[i] = மதிப்புகள்[j];
                மதிப்புகள்[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        எனில் (j - இடது < வலது - i) {
            எனில் (இடது < j) quicksort_மதிப்புகள்_indices_desc(மதிப்புகள், indices, இடது, j);
            இடது = i;
        } இல்லையெனில் {
            எனில் (i < வலது) quicksort_மதிப்புகள்_indices_desc(மதிப்புகள், indices, i, வலது);
            வலது = j;
        }
    }
}

நிலையான உள்வரி uint32_t உச்சி_k_max_மதிப்பு_index(மாறிலி தசம * மதிப்புகள், uint32_t n, தசம * மதிப்பு) {
    int32_t max_idx = 0;
    hvx_argmax_f32(மதிப்புகள், n, 0, மதிப்பு, &max_idx);
    திரும்பு (uint32_t) max_idx;
}

நிலையான மாறிலி int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

நிலையான உள்வரி வெறுமை init_indices_ramp(int32_t * indices_buf, uint32_t மொத்தம், uint32_t base) {
    மாறிலி HVX_Vector ind_init_vec = Q6_Vw_vadd_VwVw(*(மாறிலி HVX_Vector *) argsort_ramp_lut, Q6_V_vsplat_R(base));
    மாறிலி HVX_Vector ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_Vector * indices_buf_vec = (HVX_Vector *) indices_buf;
    uint32_t இலக்கம்_vecs = மொத்தம் / 32;
    HVX_Vector curr_ind_vec = ind_init_vec;
    க்கு (uint32_t j = 0; j < இலக்கம்_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = மொத்தம் % 32;
    எனில் (rem > 0) {
        hvx_vec_store_a((வெறுமை *) &indices_buf_vec[இலக்கம்_vecs], rem * அளவு(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
நிலையான உள்வரி வெறுமை vec_cas(HVX_Vector * X_val, HVX_Vector * X_idx, HVX_Vector * Y_val, HVX_Vector * Y_idx, bool asc) {
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
நிலையான உள்வரி வெறுமை bitonic_cas_32(HVX_Vector * V, HVX_Vector * I, முழு d, HVX_VectorPred dir_mask, HVX_Vector idx_vec, HVX_Vector zero_vec) {
    HVX_VectorPred mask_இடது;
    HVX_Vector V_rot_இடது, V_rot_வலது;
    HVX_Vector I_rot_இடது, I_rot_வலது;

    எனில் (d == 1) {
        mask_இடது = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_இடது = Q6_V_vror_VR(*V, 4);
        V_rot_வலது = Q6_V_vror_VR(*V, 124);
        I_rot_இடது = Q6_V_vror_VR(*I, 4);
        I_rot_வலது = Q6_V_vror_VR(*I, 124);
    } இல்லையெனில் எனில் (d == 2) {
        mask_இடது = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_இடது = Q6_V_vror_VR(*V, 8);
        V_rot_வலது = Q6_V_vror_VR(*V, 120);
        I_rot_இடது = Q6_V_vror_VR(*I, 8);
        I_rot_வலது = Q6_V_vror_VR(*I, 120);
    } இல்லையெனில் எனில் (d == 4) {
        mask_இடது = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_இடது = Q6_V_vror_VR(*V, 16);
        V_rot_வலது = Q6_V_vror_VR(*V, 112);
        I_rot_இடது = Q6_V_vror_VR(*I, 16);
        I_rot_வலது = Q6_V_vror_VR(*I, 112);
    } இல்லையெனில் எனில் (d == 8) {
        mask_இடது = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_இடது = Q6_V_vror_VR(*V, 32);
        V_rot_வலது = Q6_V_vror_VR(*V, 96);
        I_rot_இடது = Q6_V_vror_VR(*I, 32);
        I_rot_வலது = Q6_V_vror_VR(*I, 96);
    } இல்லையெனில் {
        mask_இடது = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_இடது = Q6_V_vror_VR(*V, 64);
        V_rot_வலது = Q6_V_vror_VR(*V, 64);
        I_rot_இடது = Q6_V_vror_VR(*I, 64);
        I_rot_வலது = Q6_V_vror_VR(*I, 64);
    }

    HVX_Vector V_paired = Q6_V_vmux_QVV(mask_இடது, V_rot_இடது, V_rot_வலது);
    HVX_Vector I_paired = Q6_V_vmux_QVV(mask_இடது, I_rot_இடது, I_rot_வலது);

    HVX_VectorPred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_VectorPred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_VectorPred mask_வலது = Q6_Q_not_Q(mask_இடது);
    HVX_VectorPred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(mask_இடது, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, mask_வலது)
    );
    HVX_VectorPred Q_swap = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_mask, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_mask), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_swap, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_swap, I_paired, *I);
}

__attribute__((always_inline))
நிலையான உள்வரி வெறுமை bitonic_sort_generic_hvx(uint8_t * மதிப்புகள், uint8_t * indices, முழு K, bool asc_order) {
    HVX_Vector V[32];
    HVX_Vector I[32];

    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(மாறிலி HVX_Vector *) argsort_ramp_lut;

    க்கு (முழு v = 0; v < K; v++) {
        V[v] = *(HVX_Vector *)(மதிப்புகள் + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    முழு M = 5;
    போது ((1 << (M - 5)) < K) M++;

    க்கு (முழு s = 1; s <= M; s++) {
        க்கு (முழு stage_d = s - 1; stage_d >= 0; stage_d--) {
            முழு d = 1 << stage_d;
            எனில் (d >= 32) {
                முழு v_dist = d / 32;
                க்கு (முழு v1 = 0; v1 < K; v1++) {
                    எனில் ((v1 & v_dist) == 0) {
                        முழு v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_order;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } இல்லையெனில் {
                எனில் (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    க்கு (முழு v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                } இல்லையெனில் {
                    க்கு (முழு v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_order;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    க்கு (முழு v = 0; v < K; v++) {
        *(HVX_Vector *)(மதிப்புகள் + v * 128)  = V[v];
        *(HVX_Vector *)(indices + v * 128) = I[v];
    }
}

நிலையான வெறுமை bitonic_sort_vtcm_desc(uint8_t * மதிப்புகள், uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(மாறிலி HVX_Vector *) argsort_ramp_lut;

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    எனில் (init_indices) {
        க்கு (uint32_t v = 0; v < n_vec; v++) {
            HVX_Vector idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_Vector *)(indices + v * 128) = idx;
        }
    }

    முழு M = 5;
    போது ((1u << (M - 5)) < n_vec) M++;

    க்கு (முழு s = 1; s <= M; s++) {
        க்கு (முழு stage_d = s - 1; stage_d >= 0; stage_d--) {
            முழு d = 1 << stage_d;
            எனில் (d >= 32) {
                uint32_t v_dist = d / 32;
                க்கு (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    எனில் ((v1 & v_dist) == 0) {
                        uint32_t v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : false;

                        HVX_Vector Vv1 = *(HVX_Vector *)(மதிப்புகள் + v1 * 128);
                        HVX_Vector Iv1 = *(HVX_Vector *)(indices + v1 * 128);
                        HVX_Vector Vv2 = *(HVX_Vector *)(மதிப்புகள் + v2 * 128);
                        HVX_Vector Iv2 = *(HVX_Vector *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_Vector *)(மதிப்புகள் + v1 * 128)  = Vv1;
                        *(HVX_Vector *)(indices + v1 * 128) = Iv1;
                        *(HVX_Vector *)(மதிப்புகள் + v2 * 128)  = Vv2;
                        *(HVX_Vector *)(indices + v2 * 128) = Iv2;
                    }
                }
            } இல்லையெனில் {
                எனில் (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    க்கு (uint32_t v = 0; v < n_vec; v++) {
                        HVX_Vector Vv = *(HVX_Vector *)(மதிப்புகள் + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(மதிப்புகள் + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                } இல்லையெனில் {
                    க்கு (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;

                        HVX_Vector Vv = *(HVX_Vector *)(மதிப்புகள் + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(மதிப்புகள் + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

நிலையான வெறுமை உச்சி_k_select_tiled(dma_queue * dma_q, கட்டமைப்பு htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               தசம * மதிப்புகள்_buf, int32_t * indices_buf,
                               தசம * out_மதிப்புகள், int32_t * out_indices,
                               uint32_t chunk_base) {
    மாறிலி uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_மொத்தம்    = n_tiles * k;
    uint32_t merge_n_vec        = hmx_ceil_div(candidate_மொத்தம், 32);
    uint32_t merge_n_vec_pow2   = 1;
    போது (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
    uint32_t merge_elems        = merge_n_vec_pow2 * 32;
    தசம * candidate_மதிப்புகள்    = மதிப்புகள்_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    க்கு (uint32_t offset = 0; offset < n; offset += tile_elems) {
        uint32_t tile_மொத்தம் = MIN(tile_elems, n - offset);
        dma_sync_read(dma_q, மதிப்புகள்_buf, src_addr + (size_t) offset * அளவு(தசம), tile_மொத்தம் * அளவு(தசம));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
        எனில் (tile_மொத்தம் < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (மதிப்புகள்_buf + tile_மொத்தம்), -INFINITY, tile_elems - tile_மொத்தம்);
        }

        bitonic_sort_vtcm_desc((uint8_t *) மதிப்புகள்_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_மொத்தம்);
        க்கு (uint32_t j = 0; j < tile_k; j++) {
            candidate_மதிப்புகள்[candidate_pos] = மதிப்புகள்_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) offset;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    எனில் (merge_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_மதிப்புகள் + candidate_pos), -INFINITY, merge_elems - candidate_pos);
        க்கு (uint32_t j = candidate_pos; j < merge_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_sort_vtcm_desc((uint8_t *) candidate_மதிப்புகள், (uint8_t *) candidate_indices, merge_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    க்கு (uint32_t j = 0; j < k; j++) {
        out_மதிப்புகள்[j]  = candidate_மதிப்புகள்[j];
        out_indices[j] = candidate_indices[j] + (int32_t) chunk_base;
    }
}

__attribute__((always_inline))
நிலையான உள்வரி வெறுமை sort32_f32_hvx(uint8_t * மதிப்புகள், uint8_t * indices, எண்ணிக்கை ggml_sort_order order) {
    bitonic_sort_generic_hvx(மதிப்புகள், indices, 1, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
நிலையான உள்வரி வெறுமை sort64_f32_hvx(uint8_t * மதிப்புகள், uint8_t * indices, எண்ணிக்கை ggml_sort_order order) {
    bitonic_sort_generic_hvx(மதிப்புகள், indices, 2, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
நிலையான உள்வரி வெறுமை sort128_f32_hvx(uint8_t * மதிப்புகள், uint8_t * indices, எண்ணிக்கை ggml_sort_order order) {
    bitonic_sort_generic_hvx(மதிப்புகள், indices, 4, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
நிலையான உள்வரி வெறுமை sort256_f32_hvx(uint8_t * மதிப்புகள், uint8_t * indices, எண்ணிக்கை ggml_sort_order order) {
    bitonic_sort_generic_hvx(மதிப்புகள், indices, 8, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
நிலையான உள்வரி வெறுமை sort512_f32_hvx(uint8_t * மதிப்புகள், uint8_t * indices, எண்ணிக்கை ggml_sort_order order) {
    bitonic_sort_generic_hvx(மதிப்புகள், indices, 16, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
நிலையான உள்வரி வெறுமை sort1024_f32_hvx(uint8_t * மதிப்புகள், uint8_t * indices, எண்ணிக்கை ggml_sort_order order) {
    bitonic_sort_generic_hvx(மதிப்புகள், indices, 32, order == GGML_SORT_ORDER_ASC);
}

நிலையான வெறுமை merge_runs(
    மாறிலி தசம * restrict in_val0, மாறிலி int32_t * restrict in_idx0, uint32_t n0,
    மாறிலி தசம * restrict in_val1, மாறிலி int32_t * restrict in_idx1, uint32_t n1,
    தசம * restrict out_val, int32_t * restrict out_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t out = 0;

    எனில் (asc) {
        போது (i0 < n0 && i1 < n1) {
            எனில் (in_val0[i0] <= in_val1[i1]) {
                எனில் (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } இல்லையெனில் {
                எனில் (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    } இல்லையெனில் {
        போது (i0 < n0 && i1 < n1) {
            எனில் (in_val0[i0] >= in_val1[i1]) {
                எனில் (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } இல்லையெனில் {
                எனில் (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    }

    போது (i0 < n0) {
        எனில் (out_val) out_val[out] = in_val0[i0];
        out_idx[out] = in_idx0[i0];
        i0++;
        out++;
    }

    போது (i1 < n1) {
        எனில் (out_val) out_val[out] = in_val1[i1];
        out_idx[out] = in_idx1[i1];
        i1++;
        out++;
    }
}

கட்டமைப்பு htp_sort_chunk_ctx {
    கட்டமைப்பு htp_ops_context *              octx;
    மாறிலி கட்டமைப்பு htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              real_மொத்தம்[HTP_MAX_NTHREADS];
};

நிலையான வெறுமை htp_sort_chunk_job(unsigned முழு n, unsigned முழு i, வெறுமை * data) {
    கட்டமைப்பு htp_sort_chunk_ctx * cctx = (கட்டமைப்பு htp_sort_chunk_ctx *) data;
    கட்டமைப்பு htp_ops_context * octx = cctx->octx;
    மாறிலி கட்டமைப்பு htp_sort_kernel_params * kparams = cctx->kparams;
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_base  = i * chunk_elems;

    uint32_t real_மொத்தம் = (chunk_base < ne00) ? MIN(chunk_elems, ne00 - chunk_base) : 0;
    cctx->real_மொத்தம்[i] = real_மொத்தம்;

    uint8_t * spad = cctx->vtcm_base + (size_t) kparams->phase1_slot_size * i;
    தசம *   மதிப்புகள்_buf  = (தசம *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * அளவு(தசம));

    தசம *   merge_மதிப்புகள்  = (தசம *)   (cctx->vtcm_base + kparams->merge_மதிப்புகள்_off);
    int32_t * merge_indices = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off);

    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[i];
    மாறிலி dma_addr_t src_addr = src0->data + (size_t) chunk_base * அளவு(தசம);

    எனில் (kparams->is_உச்சி_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        எனில் (real_மொத்தம் == 0) {
            க்கு (uint32_t j = 0; j < local_k; j++) {
                merge_மதிப்புகள்[i * local_k + j]  = -INFINITY;
                merge_indices[i * local_k + j] = 0;
            }
            திரும்பு;
        }

        எனில் (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], மதிப்புகள்_buf, src_addr, real_மொத்தம் * அளவு(தசம));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            தசம max_மதிப்பு;
            uint32_t max_index = உச்சி_k_max_மதிப்பு_index(மதிப்புகள்_buf, real_மொத்தம், &max_மதிப்பு);
            merge_மதிப்புகள்[i] = max_மதிப்பு;
            merge_indices[i] = (int32_t) (max_index + chunk_base);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            திரும்பு;
        }

        எனில் (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            உச்சி_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_மொத்தம், local_k,
                               மதிப்புகள்_buf, indices_buf,
                               &merge_மதிப்புகள்[i * local_k], &merge_indices[i * local_k],
                               chunk_base);
            திரும்பு;
        }

        dma_sync_read(octx->ctx->dma[i], மதிப்புகள்_buf, src_addr, real_மொத்தம் * அளவு(தசம));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        எனில் (chunk_elems > real_மொத்தம்) {
            hvx_splat_f32_u((uint8_t *)(மதிப்புகள்_buf + real_மொத்தம்), -INFINITY, chunk_elems - real_மொத்தம்);
        }

        bitonic_sort_vtcm_desc((uint8_t *) மதிப்புகள்_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        க்கு (uint32_t j = 0; j < local_k; j++) {
            merge_மதிப்புகள்[i * local_k + j]  = மதிப்புகள்_buf[j];
            merge_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_base;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } இல்லையெனில் {
        எனில் (real_மொத்தம் > 0) {
            dma_sync_read(octx->ctx->dma[i], மதிப்புகள்_buf, src_addr, real_மொத்தம் * அளவு(தசம));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_மொத்தம், chunk_base);

            எனில் (kparams->order == GGML_SORT_ORDER_ASC) {
                quicksort_மதிப்புகள்_indices_asc(மதிப்புகள்_buf, indices_buf, 0, real_மொத்தம் - 1);
            } இல்லையெனில் {
                quicksort_மதிப்புகள்_indices_desc(மதிப்புகள்_buf, indices_buf, 0, real_மொத்தம் - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

கட்டமைப்பு htp_argsort_merge_l1_ctx {
    கட்டமைப்பு htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

நிலையான வெறுமை htp_argsort_merge_l1_job(unsigned முழு n, unsigned முழு i, வெறுமை * data) {
    கட்டமைப்பு htp_argsort_merge_l1_ctx * mctx = (கட்டமைப்பு htp_argsort_merge_l1_ctx *) data;
    கட்டமைப்பு htp_sort_chunk_ctx * cctx = mctx->cctx;
    மாறிலி கட்டமைப்பு htp_sort_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_மொத்தம்[c0];
    uint32_t count1 = cctx->real_மொத்தம்[c1];

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    மாறிலி தசம *   val0 = (மாறிலி தசம *)   (cctx->vtcm_base + slot_size * c0);
    மாறிலி int32_t * idx0 = (மாறிலி int32_t *) (cctx->vtcm_base + slot_size * c0 + chunk_elems * அளவு(தசம));
    மாறிலி தசம *   val1 = (மாறிலி தசம *)   (cctx->vtcm_base + slot_size * c1);
    மாறிலி int32_t * idx1 = (மாறிலி int32_t *) (cctx->vtcm_base + slot_size * c1 + chunk_elems * அளவு(தசம));

    uint32_t out_offset = (2 * i) * chunk_elems;
    தசம *   out_val = (தசம *)   (cctx->vtcm_base + kparams->merge_மதிப்புகள்_off)  + out_offset;
    int32_t * out_idx = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + out_offset;

    கட்டமைப்பு htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

கட்டமைப்பு htp_argsort_merge_l2_ctx {
    கட்டமைப்பு htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

நிலையான வெறுமை htp_argsort_merge_l2_job(unsigned முழு n, unsigned முழு i, வெறுமை * data) {
    கட்டமைப்பு htp_argsort_merge_l2_ctx * mctx = (கட்டமைப்பு htp_argsort_merge_l2_ctx *) data;
    கட்டமைப்பு htp_sort_chunk_ctx * cctx = mctx->cctx;
    மாறிலி கட்டமைப்பு htp_sort_kernel_params * kparams = cctx->kparams;

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    மாறிலி தசம *   val0 = (மாறிலி தசம *)   (cctx->vtcm_base + kparams->merge_மதிப்புகள்_off)  + off0;
    மாறிலி int32_t * idx0 = (மாறிலி int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off0;
    மாறிலி தசம *   val1 = (மாறிலி தசம *)   (cctx->vtcm_base + kparams->merge_மதிப்புகள்_off)  + off1;
    மாறிலி int32_t * idx1 = (மாறிலி int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off1;

    uint32_t count0 = cctx->real_மொத்தம்[4*i + 0] + cctx->real_மொத்தம்[4*i + 1];
    uint32_t count1 = cctx->real_மொத்தம்[4*i + 2] + cctx->real_மொத்தம்[4*i + 3];

    uint8_t * slot_base = cctx->vtcm_base + (4 * i) * slot_size;
    தசம *   out_val   = (தசம *) slot_base;
    int32_t * out_idx   = (int32_t *) (slot_base + 4 * chunk_elems * அளவு(தசம));

    கட்டமைப்பு htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

கட்டமைப்பு htp_sort_multi_row_ctx {
    கட்டமைப்பு htp_ops_context *              octx;
    மாறிலி கட்டமைப்பு htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              nrows_per_thread;
};

கட்டமைப்பு htp_tensor_row_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

நிலையான உள்வரி கட்டமைப்பு htp_tensor_row_coord init_row_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    திரும்பு (கட்டமைப்பு htp_tensor_row_coord){ i01, i02, i03 };
}

நிலையான உள்வரி வெறுமை step_row_coord(கட்டமைப்பு htp_tensor_row_coord * c, uint32_t ne01, uint32_t ne02) {
    c->i01++;
    எனில் (c->i01 == ne01) {
        c->i01 = 0;
        c->i02++;
        எனில் (c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

நிலையான உள்வரி size_t coord_offset(கட்டமைப்பு htp_tensor_row_coord c, size_t nb1, size_t nb2, size_t nb3) {
    திரும்பு (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

நிலையான உள்வரி வெறுமை compute_row_sort(
    தசம * cur_மதிப்புகள்,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_உச்சி_k,
    எண்ணிக்கை ggml_sort_order order
) {
    எனில் (is_உச்சி_k) {
        எனில் (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            தேர்வு (ne00) {
                நிலை 32:   sort32_f32_hvx((uint8_t *) cur_மதிப்புகள், (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); நிறுத்து;
                நிலை 64:   sort64_f32_hvx((uint8_t *) cur_மதிப்புகள், (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); நிறுத்து;
                நிலை 128:  sort128_f32_hvx((uint8_t *) cur_மதிப்புகள், (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); நிறுத்து;
                நிலை 256:  sort256_f32_hvx((uint8_t *) cur_மதிப்புகள், (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); நிறுத்து;
                நிலை 512:  sort512_f32_hvx((uint8_t *) cur_மதிப்புகள், (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); நிறுத்து;
                நிலை 1024: sort1024_f32_hvx((uint8_t *) cur_மதிப்புகள், (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); நிறுத்து;
            }
        } இல்லையெனில் {
            எனில் (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_மதிப்புகள் + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_sort_vtcm_desc((uint8_t *) cur_மதிப்புகள், (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } இல்லையெனில் {
        எனில் (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            தேர்வு (ne00) {
                நிலை 32:   sort32_f32_hvx((uint8_t *) cur_மதிப்புகள், (uint8_t *) cur_indices, order); நிறுத்து;
                நிலை 64:   sort64_f32_hvx((uint8_t *) cur_மதிப்புகள், (uint8_t *) cur_indices, order); நிறுத்து;
                நிலை 128:  sort128_f32_hvx((uint8_t *) cur_மதிப்புகள், (uint8_t *) cur_indices, order); நிறுத்து;
                நிலை 256:  sort256_f32_hvx((uint8_t *) cur_மதிப்புகள், (uint8_t *) cur_indices, order); நிறுத்து;
                நிலை 512:  sort512_f32_hvx((uint8_t *) cur_மதிப்புகள், (uint8_t *) cur_indices, order); நிறுத்து;
                நிலை 1024: sort1024_f32_hvx((uint8_t *) cur_மதிப்புகள், (uint8_t *) cur_indices, order); நிறுத்து;
            }
        } இல்லையெனில் {
            init_indices_ramp(cur_indices, ne00, 0);
            எனில் (order == GGML_SORT_ORDER_ASC) {
                quicksort_மதிப்புகள்_indices_asc(cur_மதிப்புகள், cur_indices, 0, ne00 - 1);
            } இல்லையெனில் {
                quicksort_மதிப்புகள்_indices_desc(cur_மதிப்புகள், cur_indices, 0, ne00 - 1);
            }
        }
    }
}

நிலையான வெறுமை htp_sort_multi_row_job(unsigned முழு n, unsigned முழு ith, வெறுமை * data) {
    கட்டமைப்பு htp_sort_multi_row_ctx * mctx = (கட்டமைப்பு htp_sort_multi_row_ctx *) data;
    கட்டமைப்பு htp_ops_context * octx = mctx->octx;
    மாறிலி கட்டமைப்பு htp_sort_kernel_params * kparams = mctx->kparams;
    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;

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

    எனில் (r_start >= r_end) திரும்பு;

    size_t spad_offset = ith * (size_t) (kparams->phase1_slot_size * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_base + spad_offset;
    size_t slot_size = (size_t) kparams->phase1_slot_size;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    தசம *   மதிப்புகள்_buf  = (தசம *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * அளவு(தசம));

    dma_queue * dma_q = octx->ctx->dma[ith];
    கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_உச்சி_k = kparams->is_உச்சி_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    எண்ணிக்கை ggml_sort_order order = (எண்ணிக்கை ggml_sort_order) kparams->order;

    மாறிலி uint32_t src_bytes = ne00 * அளவு(தசம);
    மாறிலி uint32_t dst_bytes = (is_உச்சி_k ? k : ne00) * அளவு(int32_t);

    எனில் (kparams->n_slots == 2) {
        கட்டமைப்பு htp_tensor_row_coord பகாஎண்_coord = init_row_coord(r_start, ne01, ne02);
        க்கு (uint32_t r = r_start, spad_idx = 0; r < r_end && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * slot_size;
            தசம *   cur_மதிப்புகள்  = (தசம *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * அளவு(தசம));

            dma_addr_t dst_addr = dst->data + coord_offset(பகாஎண்_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->data + coord_offset(பகாஎண்_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_queue_தள்ளு(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 0);

            // Prefetch input row
            dma_queue_தள்ளு(dma_q, dma_make_data(cur_மதிப்புகள், src_addr),
                           src_bytes, src_bytes, src_bytes, 1);

            step_row_coord(&பகாஎண்_coord, ne01, ne02);
        }

        கட்டமைப்பு htp_tensor_row_coord dst_coord = init_row_coord(r_start, ne01, ne02);
        கட்டமைப்பு htp_tensor_row_coord next_src_coord = init_row_coord(r_start + 2, ne01, ne02);

        க்கு (uint32_t r = r_start; r < r_end; r++) {
            uint32_t cur_slot = (r - r_start) & 1;
            uint8_t * cur_spad = thread_spad + cur_slot * slot_size;
            தசம *   cur_மதிப்புகள்  = (தசம *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * அளவு(தசம));

            // Wait for previous writeback of this slot to complete
            dma_queue_எடு(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_queue_எடு(dma_q);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(cur_மதிப்புகள், cur_indices, ne00, chunk_elems, is_உச்சி_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->data + coord_offset(dst_coord, nb1, nb2, nb3);
            dma_queue_தள்ளு(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 1);
            step_row_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            மாறிலி uint32_t next_row = r + 2;
            எனில் (next_row < r_end) {
                dma_addr_t next_src_addr = src0->data + coord_offset(next_src_coord, nb01, nb02, nb03);
                dma_queue_தள்ளு(dma_q, dma_make_data(cur_மதிப்புகள், next_src_addr),
                               src_bytes, src_bytes, src_bytes, 1);
                step_row_coord(&next_src_coord, ne01, ne02);
            }
        }

        dma_queue_flush(dma_q);
    } இல்லையெனில் {
        கட்டமைப்பு htp_tensor_row_coord coord = init_row_coord(r_start, ne01, ne02);
        க்கு (uint32_t r = r_start; r < r_end; r++) {
            dma_addr_t src_addr = src0->data + coord_offset(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->data + coord_offset(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, மதிப்புகள்_buf, src_addr, src_bytes);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(மதிப்புகள்_buf, indices_buf, ne00, chunk_elems, is_உச்சி_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_bytes);
            step_row_coord(&coord, ne01, ne02);
        }
    }
}

நிலையான முழு op_sort_common(கட்டமைப்பு htp_ops_context * octx, bool is_உச்சி_k) {
    எனில் (octx->src[0]->type != HTP_TYPE_F32) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    மாறிலி கட்டமைப்பு htp_tensor * src0 = octx->src[0];
    மாறிலி கட்டமைப்பு htp_tensor * dst  = octx->dst;

    எனில் (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        திரும்பு HTP_STATUS_NO_SUPPORT;
    }

    மாறிலி uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    மாறிலி size_t dst_row_size = dst->ne[0] * அளவு(int32_t);

    uint32_t row_start = 0;
    uint32_t row_end   = total_rows;
    எனில் (octx->ctx->mdev.மொத்தம் > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, அளவு(int32_t), (uint32_t) dst_row_size, &rows_per_chunk);
        மாறிலி கட்டமைப்பு htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.மொத்தம், &octx->ctx->mdev.மொத்தம்_div);
        row_start = range.start;
        row_end   = range.start + range.மொத்தம்;
    }

    மாறிலி uint32_t nrows = row_end - row_start;
    எனில் (nrows == 0) {
        திரும்பு HTP_STATUS_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    கட்டமைப்பு htp_sort_kernel_params kparams_local;
    மாறிலி கட்டமைப்பு htp_sort_kernel_params * kparams = (மாறிலி கட்டமைப்பு htp_sort_kernel_params *) octx->kernel_params;

    எனில் (kparams->n_threads == 0) {
        கட்டமைப்பு htp_sort_vtcm_layout layout;
        எனில் (!htp_sort_solve_layout(&layout, ne00, total_rows, k, octx->n_threads, octx->ctx->vtcm_size, is_உச்சி_k)) {
            திரும்பு HTP_STATUS_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, அளவு(kparams_local));
        kparams_local.n_threads         = (int32_t) layout.n_threads;
        kparams_local.total_rows        = (int32_t) total_rows;
        kparams_local.row_start         = (int32_t) row_start;
        kparams_local.row_end           = (int32_t) row_end;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.order             = is_உச்சி_k ? GGML_SORT_ORDER_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_உச்சி_k          = is_உச்சி_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_size         = (int32_t) layout.total_bytes;
        kparams_local.phase1_slot_size  = (int32_t) layout.phase1_slot_size;
        kparams_local.merge_மதிப்புகள்_off  = (int32_t) layout.merge_மதிப்புகள்_off;
        kparams_local.merge_indices_off = (int32_t) layout.merge_indices_off;
        kparams_local.merge_elems       = (int32_t) layout.merge_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } இல்லையெனில் எனில் (octx->ctx->mdev.மொத்தம் > 1) {
        memcpy(&kparams_local, kparams, அளவு(kparams_local));
        kparams_local.row_start = (int32_t) row_start;
        kparams_local.row_end   = (int32_t) row_end;
        kparams = &kparams_local;
    }

    எனில் (octx->ctx->vtcm_size < (size_t) kparams->vtcm_size) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_size, octx->ctx->vtcm_size);
        திரும்பு HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    எனில் (total_rows == 1 && kparams->n_chunks > 1) {
        கட்டமைப்பு htp_sort_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_base = vtcm_base;
        memset(cctx.real_மொத்தம், 0, அளவு(cctx.real_மொத்தம்));

        work_queue_run(octx->ctx->work_queue, htp_sort_chunk_job, &cctx, (uint32_t) kparams->n_chunks);

        எனில் (is_உச்சி_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            எனில் (k == 1 && ne00 >= 128*1024) {
                தசம *   cand_vals = (தசம *)   (vtcm_base + kparams->merge_மதிப்புகள்_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                தசம best_val = cand_vals[0];
                int32_t best_idx = cand_idxs[0];
                க்கு (int32_t c = 1; c < kparams->n_chunks; c++) {
                    எனில் (cand_vals[c] > best_val) {
                        best_val = cand_vals[c];
                        best_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->data;
                dst_ptr[0] = best_idx;
                திரும்பு HTP_STATUS_OK;
            }

            uint32_t total_candidates = (uint32_t) kparams->n_chunks * local_k;
            தசம *   cand_vals = (தசம *)   (vtcm_base + kparams->merge_மதிப்புகள்_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
            uint32_t merge_elems = (uint32_t) kparams->merge_elems;

            எனில் (merge_elems > total_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + total_candidates), -INFINITY, merge_elems - total_candidates);
                க்கு (uint32_t j = total_candidates; j < merge_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_sort_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, merge_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->data, cand_idxs, k * அளவு(int32_t));
        } இல்லையெனில் {
            bool asc = kparams->order == GGML_SORT_ORDER_ASC;
            எனில் (kparams->n_chunks == 8) {
                கட்டமைப்பு htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 4);

                கட்டமைப்பு htp_argsort_merge_l2_ctx l2ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l2_job, &l2ctx, 2);

                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                மாறிலி தசம *   val0 = (மாறிலி தசம *) vtcm_base;
                மாறிலி int32_t * idx0 = (மாறிலி int32_t *) (vtcm_base + 4 * chunk_elems * அளவு(தசம));
                மாறிலி தசம *   val1 = (மாறிலி தசம *) (vtcm_base + 4 * slot_size);
                மாறிலி int32_t * idx1 = (மாறிலி int32_t *) (vtcm_base + 4 * slot_size + 4 * chunk_elems * அளவு(தசம));

                uint32_t n01 = cctx.real_மொத்தம்[0] + cctx.real_மொத்தம்[1] + cctx.real_மொத்தம்[2] + cctx.real_மொத்தம்[3];
                uint32_t n23 = cctx.real_மொத்தம்[4] + cctx.real_மொத்தம்[5] + cctx.real_மொத்தம்[6] + cctx.real_மொத்தம்[7];

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, n01, val1, idx1, n23, வெற்று, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * அளவு(int32_t));
            } இல்லையெனில் எனில் (kparams->n_chunks == 4) {
                கட்டமைப்பு htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_மொத்தம்[0] + cctx.real_மொத்தம்[1];
                uint32_t n23 = cctx.real_மொத்தம்[2] + cctx.real_மொத்தம்[3];

                மாறிலி தசம *   val01 = (மாறிலி தசம *)   (vtcm_base + kparams->merge_மதிப்புகள்_off);
                மாறிலி int32_t * idx01 = (மாறிலி int32_t *) (vtcm_base + kparams->merge_indices_off);
                மாறிலி தசம *   val23 = (மாறிலி தசம *)   (vtcm_base + kparams->merge_மதிப்புகள்_off)  + 2 * chunk_elems;
                மாறிலி int32_t * idx23 = (மாறிலி int32_t *) (vtcm_base + kparams->merge_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_base;
                கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val01, idx01, n01, val23, idx23, n23, வெற்று, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * அளவு(int32_t));
            } இல்லையெனில் எனில் (kparams->n_chunks == 2) {
                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                மாறிலி தசம *   val0 = (மாறிலி தசம *)   (vtcm_base);
                மாறிலி int32_t * idx0 = (மாறிலி int32_t *) (vtcm_base + chunk_elems * அளவு(தசம));
                மாறிலி தசம *   val1 = (மாறிலி தசம *)   (vtcm_base + slot_size);
                மாறிலி int32_t * idx1 = (மாறிலி int32_t *) (vtcm_base + slot_size + chunk_elems * அளவு(தசம));

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                கட்டமைப்பு htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, cctx.real_மொத்தம்[0], val1, idx1, cctx.real_மொத்தம்[1], வெற்று, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * அளவு(int32_t));
            } இல்லையெனில் {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_base + chunk_elems * அளவு(தசம));
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * அளவு(int32_t));
            }
        }
        திரும்பு HTP_STATUS_OK;
    }

    கட்டமைப்பு htp_sort_multi_row_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_base        = vtcm_base;

    uint32_t dst_row_bytes = (uint32_t) dst_row_size;
    uint32_t rows_per_chunk = 0;
    htp_tensor_mdev_rows_per_chunk(dst, அளவு(int32_t), dst_row_bytes, &rows_per_chunk);
    எனில் (rows_per_chunk == 0) {
        rows_per_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_threads - 1) / (uint32_t) kparams->n_threads;
    எனில் (rows_per_chunk > 1) {
        dr = hex_round_up(dr, rows_per_chunk);
    }
    mctx.nrows_per_thread = dr;

    work_queue_run(octx->ctx->work_queue, htp_sort_multi_row_job, &mctx, (uint32_t) kparams->n_threads);

    திரும்பு HTP_STATUS_OK;
}

முழு op_argsort(கட்டமைப்பு htp_ops_context * octx) {
    திரும்பு op_sort_common(octx, false);
}

முழு op_உச்சி_k(கட்டமைப்பு htp_ops_context * octx) {
    திரும்பு op_sort_common(octx, true);
}
