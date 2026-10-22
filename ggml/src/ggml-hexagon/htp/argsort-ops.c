#অন্তর্ভাব <string.h>
#অন্তর্ভাব <stdlib.h>
#অন্তর্ভাব <math.h>
#অন্তর্ভাব <HAP_farf.h>
#অন্তর্ভাব <HAP_perf.h>

#define GGML_COMMON_DECL_C
#অন্তর্ভাব "ggml-common.h"
#অন্তর্ভাব "ggml.h"

#অন্তর্ভাব "hvx-utils.h"
#অন্তর্ভাব "hvx-copy.h"
#অন্তর্ভাব "hvx-reduce.h"
#অন্তর্ভাব "dma-queue.h"

#অন্তর্ভাব "hex-common.h"
#অন্তর্ভাব "htp-ctx.h"
#অন্তর্ভাব "htp-ops.h"
#অন্তর্ভাব "htp-tensor.h"
#অন্তর্ভাব "argsort-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif


স্থির শূন্য quicksort_মানসমূহ_indices_asc(ভিন্ন * মানসমূহ, int32_t * indices, পূর্ণাঙ্ক বাম, পূর্ণাঙ্ক ডান) {
    যাবত্ (বাম < ডান) {
        ভিন্ন pivot = মানসমূহ[বাম + (ডান - বাম) / 2];
        পূর্ণাঙ্ক i = বাম;
        পূর্ণাঙ্ক j = ডান;

        যাবত্ (i <= j) {
            যাবত্ (মানসমূহ[i] < pivot) i++;
            যাবত্ (মানসমূহ[j] > pivot) j--;

            যদি (i <= j) {
                ভিন্ন tmp_val = মানসমূহ[i];
                মানসমূহ[i] = মানসমূহ[j];
                মানসমূহ[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        যদি (j - বাম < ডান - i) {
            যদি (বাম < j) quicksort_মানসমূহ_indices_asc(মানসমূহ, indices, বাম, j);
            বাম = i;
        } অন্যথা {
            যদি (i < ডান) quicksort_মানসমূহ_indices_asc(মানসমূহ, indices, i, ডান);
            ডান = j;
        }
    }
}

স্থির শূন্য quicksort_মানসমূহ_indices_desc(ভিন্ন * মানসমূহ, int32_t * indices, পূর্ণাঙ্ক বাম, পূর্ণাঙ্ক ডান) {
    যাবত্ (বাম < ডান) {
        ভিন্ন pivot = মানসমূহ[বাম + (ডান - বাম) / 2];
        পূর্ণাঙ্ক i = বাম;
        পূর্ণাঙ্ক j = ডান;

        যাবত্ (i <= j) {
            যাবত্ (মানসমূহ[i] > pivot) i++;
            যাবত্ (মানসমূহ[j] < pivot) j--;

            যদি (i <= j) {
                ভিন্ন tmp_val = মানসমূহ[i];
                মানসমূহ[i] = মানসমূহ[j];
                মানসমূহ[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        যদি (j - বাম < ডান - i) {
            যদি (বাম < j) quicksort_মানসমূহ_indices_desc(মানসমূহ, indices, বাম, j);
            বাম = i;
        } অন্যথা {
            যদি (i < ডান) quicksort_মানসমূহ_indices_desc(মানসমূহ, indices, i, ডান);
            ডান = j;
        }
    }
}

স্থির অন্তঃস্থ uint32_t শীর্ষ_k_max_মান_index(ধ্রুব ভিন্ন * মানসমূহ, uint32_t n, ভিন্ন * মান) {
    int32_t max_idx = 0;
    hvx_argmax_f32(মানসমূহ, n, 0, মান, &max_idx);
    প্রত্যাগম (uint32_t) max_idx;
}

স্থির ধ্রুব int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

স্থির অন্তঃস্থ শূন্য init_indices_ramp(int32_t * indices_buf, uint32_t গুনতি, uint32_t base) {
    ধ্রুব HVX_Vector ind_init_vec = Q6_Vw_vadd_VwVw(*(ধ্রুব HVX_Vector *) argsort_ramp_lut, Q6_V_vsplat_R(base));
    ধ্রুব HVX_Vector ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_Vector * indices_buf_vec = (HVX_Vector *) indices_buf;
    uint32_t অঙ্ক_vecs = গুনতি / 32;
    HVX_Vector curr_ind_vec = ind_init_vec;
    প্রতি (uint32_t j = 0; j < অঙ্ক_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = গুনতি % 32;
    যদি (rem > 0) {
        hvx_vec_store_a((শূন্য *) &indices_buf_vec[অঙ্ক_vecs], rem * পরিমাণ(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
স্থির অন্তঃস্থ শূন্য vec_cas(HVX_Vector * X_val, HVX_Vector * X_idx, HVX_Vector * Y_val, HVX_Vector * Y_idx, bool asc) {
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
স্থির অন্তঃস্থ শূন্য bitonic_cas_32(HVX_Vector * V, HVX_Vector * I, পূর্ণাঙ্ক d, HVX_VectorPred dir_mask, HVX_Vector idx_vec, HVX_Vector zero_vec) {
    HVX_VectorPred mask_বাম;
    HVX_Vector V_rot_বাম, V_rot_ডান;
    HVX_Vector I_rot_বাম, I_rot_ডান;

    যদি (d == 1) {
        mask_বাম = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_বাম = Q6_V_vror_VR(*V, 4);
        V_rot_ডান = Q6_V_vror_VR(*V, 124);
        I_rot_বাম = Q6_V_vror_VR(*I, 4);
        I_rot_ডান = Q6_V_vror_VR(*I, 124);
    } অন্যথা যদি (d == 2) {
        mask_বাম = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_বাম = Q6_V_vror_VR(*V, 8);
        V_rot_ডান = Q6_V_vror_VR(*V, 120);
        I_rot_বাম = Q6_V_vror_VR(*I, 8);
        I_rot_ডান = Q6_V_vror_VR(*I, 120);
    } অন্যথা যদি (d == 4) {
        mask_বাম = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_বাম = Q6_V_vror_VR(*V, 16);
        V_rot_ডান = Q6_V_vror_VR(*V, 112);
        I_rot_বাম = Q6_V_vror_VR(*I, 16);
        I_rot_ডান = Q6_V_vror_VR(*I, 112);
    } অন্যথা যদি (d == 8) {
        mask_বাম = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_বাম = Q6_V_vror_VR(*V, 32);
        V_rot_ডান = Q6_V_vror_VR(*V, 96);
        I_rot_বাম = Q6_V_vror_VR(*I, 32);
        I_rot_ডান = Q6_V_vror_VR(*I, 96);
    } অন্যথা {
        mask_বাম = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_বাম = Q6_V_vror_VR(*V, 64);
        V_rot_ডান = Q6_V_vror_VR(*V, 64);
        I_rot_বাম = Q6_V_vror_VR(*I, 64);
        I_rot_ডান = Q6_V_vror_VR(*I, 64);
    }

    HVX_Vector V_paired = Q6_V_vmux_QVV(mask_বাম, V_rot_বাম, V_rot_ডান);
    HVX_Vector I_paired = Q6_V_vmux_QVV(mask_বাম, I_rot_বাম, I_rot_ডান);

    HVX_VectorPred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_VectorPred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_VectorPred mask_ডান = Q6_Q_not_Q(mask_বাম);
    HVX_VectorPred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(mask_বাম, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, mask_ডান)
    );
    HVX_VectorPred Q_swap = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_mask, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_mask), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_swap, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_swap, I_paired, *I);
}

__attribute__((always_inline))
স্থির অন্তঃস্থ শূন্য bitonic_sort_generic_hvx(uint8_t * মানসমূহ, uint8_t * indices, পূর্ণাঙ্ক K, bool asc_order) {
    HVX_Vector V[32];
    HVX_Vector I[32];

    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(ধ্রুব HVX_Vector *) argsort_ramp_lut;

    প্রতি (পূর্ণাঙ্ক v = 0; v < K; v++) {
        V[v] = *(HVX_Vector *)(মানসমূহ + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    পূর্ণাঙ্ক M = 5;
    যাবত্ ((1 << (M - 5)) < K) M++;

    প্রতি (পূর্ণাঙ্ক s = 1; s <= M; s++) {
        প্রতি (পূর্ণাঙ্ক stage_d = s - 1; stage_d >= 0; stage_d--) {
            পূর্ণাঙ্ক d = 1 << stage_d;
            যদি (d >= 32) {
                পূর্ণাঙ্ক v_dist = d / 32;
                প্রতি (পূর্ণাঙ্ক v1 = 0; v1 < K; v1++) {
                    যদি ((v1 & v_dist) == 0) {
                        পূর্ণাঙ্ক v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_order;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } অন্যথা {
                যদি (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    প্রতি (পূর্ণাঙ্ক v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                } অন্যথা {
                    প্রতি (পূর্ণাঙ্ক v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_order;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    প্রতি (পূর্ণাঙ্ক v = 0; v < K; v++) {
        *(HVX_Vector *)(মানসমূহ + v * 128)  = V[v];
        *(HVX_Vector *)(indices + v * 128) = I[v];
    }
}

স্থির শূন্য bitonic_sort_vtcm_desc(uint8_t * মানসমূহ, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(ধ্রুব HVX_Vector *) argsort_ramp_lut;

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    যদি (init_indices) {
        প্রতি (uint32_t v = 0; v < n_vec; v++) {
            HVX_Vector idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_Vector *)(indices + v * 128) = idx;
        }
    }

    পূর্ণাঙ্ক M = 5;
    যাবত্ ((1u << (M - 5)) < n_vec) M++;

    প্রতি (পূর্ণাঙ্ক s = 1; s <= M; s++) {
        প্রতি (পূর্ণাঙ্ক stage_d = s - 1; stage_d >= 0; stage_d--) {
            পূর্ণাঙ্ক d = 1 << stage_d;
            যদি (d >= 32) {
                uint32_t v_dist = d / 32;
                প্রতি (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    যদি ((v1 & v_dist) == 0) {
                        uint32_t v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : false;

                        HVX_Vector Vv1 = *(HVX_Vector *)(মানসমূহ + v1 * 128);
                        HVX_Vector Iv1 = *(HVX_Vector *)(indices + v1 * 128);
                        HVX_Vector Vv2 = *(HVX_Vector *)(মানসমূহ + v2 * 128);
                        HVX_Vector Iv2 = *(HVX_Vector *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_Vector *)(মানসমূহ + v1 * 128)  = Vv1;
                        *(HVX_Vector *)(indices + v1 * 128) = Iv1;
                        *(HVX_Vector *)(মানসমূহ + v2 * 128)  = Vv2;
                        *(HVX_Vector *)(indices + v2 * 128) = Iv2;
                    }
                }
            } অন্যথা {
                যদি (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    প্রতি (uint32_t v = 0; v < n_vec; v++) {
                        HVX_Vector Vv = *(HVX_Vector *)(মানসমূহ + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(মানসমূহ + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                } অন্যথা {
                    প্রতি (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;

                        HVX_Vector Vv = *(HVX_Vector *)(মানসমূহ + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(মানসমূহ + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

স্থির শূন্য শীর্ষ_k_select_tiled(dma_queue * dma_q, সংরচনা htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               ভিন্ন * মানসমূহ_buf, int32_t * indices_buf,
                               ভিন্ন * out_মানসমূহ, int32_t * out_indices,
                               uint32_t chunk_base) {
    ধ্রুব uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_গুনতি    = n_tiles * k;
    uint32_t merge_n_vec        = hmx_ceil_div(candidate_গুনতি, 32);
    uint32_t merge_n_vec_pow2   = 1;
    যাবত্ (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
    uint32_t merge_elems        = merge_n_vec_pow2 * 32;
    ভিন্ন * candidate_মানসমূহ    = মানসমূহ_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    প্রতি (uint32_t offset = 0; offset < n; offset += tile_elems) {
        uint32_t tile_গুনতি = MIN(tile_elems, n - offset);
        dma_sync_read(dma_q, মানসমূহ_buf, src_addr + (size_t) offset * পরিমাণ(ভিন্ন), tile_গুনতি * পরিমাণ(ভিন্ন));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
        যদি (tile_গুনতি < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (মানসমূহ_buf + tile_গুনতি), -INFINITY, tile_elems - tile_গুনতি);
        }

        bitonic_sort_vtcm_desc((uint8_t *) মানসমূহ_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_গুনতি);
        প্রতি (uint32_t j = 0; j < tile_k; j++) {
            candidate_মানসমূহ[candidate_pos] = মানসমূহ_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) offset;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    যদি (merge_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_মানসমূহ + candidate_pos), -INFINITY, merge_elems - candidate_pos);
        প্রতি (uint32_t j = candidate_pos; j < merge_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_sort_vtcm_desc((uint8_t *) candidate_মানসমূহ, (uint8_t *) candidate_indices, merge_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    প্রতি (uint32_t j = 0; j < k; j++) {
        out_মানসমূহ[j]  = candidate_মানসমূহ[j];
        out_indices[j] = candidate_indices[j] + (int32_t) chunk_base;
    }
}

__attribute__((always_inline))
স্থির অন্তঃস্থ শূন্য sort32_f32_hvx(uint8_t * মানসমূহ, uint8_t * indices, গণনা ggml_sort_order order) {
    bitonic_sort_generic_hvx(মানসমূহ, indices, 1, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
স্থির অন্তঃস্থ শূন্য sort64_f32_hvx(uint8_t * মানসমূহ, uint8_t * indices, গণনা ggml_sort_order order) {
    bitonic_sort_generic_hvx(মানসমূহ, indices, 2, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
স্থির অন্তঃস্থ শূন্য sort128_f32_hvx(uint8_t * মানসমূহ, uint8_t * indices, গণনা ggml_sort_order order) {
    bitonic_sort_generic_hvx(মানসমূহ, indices, 4, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
স্থির অন্তঃস্থ শূন্য sort256_f32_hvx(uint8_t * মানসমূহ, uint8_t * indices, গণনা ggml_sort_order order) {
    bitonic_sort_generic_hvx(মানসমূহ, indices, 8, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
স্থির অন্তঃস্থ শূন্য sort512_f32_hvx(uint8_t * মানসমূহ, uint8_t * indices, গণনা ggml_sort_order order) {
    bitonic_sort_generic_hvx(মানসমূহ, indices, 16, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
স্থির অন্তঃস্থ শূন্য sort1024_f32_hvx(uint8_t * মানসমূহ, uint8_t * indices, গণনা ggml_sort_order order) {
    bitonic_sort_generic_hvx(মানসমূহ, indices, 32, order == GGML_SORT_ORDER_ASC);
}

স্থির শূন্য merge_runs(
    ধ্রুব ভিন্ন * restrict in_val0, ধ্রুব int32_t * restrict in_idx0, uint32_t n0,
    ধ্রুব ভিন্ন * restrict in_val1, ধ্রুব int32_t * restrict in_idx1, uint32_t n1,
    ভিন্ন * restrict out_val, int32_t * restrict out_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t out = 0;

    যদি (asc) {
        যাবত্ (i0 < n0 && i1 < n1) {
            যদি (in_val0[i0] <= in_val1[i1]) {
                যদি (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } অন্যথা {
                যদি (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    } অন্যথা {
        যাবত্ (i0 < n0 && i1 < n1) {
            যদি (in_val0[i0] >= in_val1[i1]) {
                যদি (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } অন্যথা {
                যদি (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    }

    যাবত্ (i0 < n0) {
        যদি (out_val) out_val[out] = in_val0[i0];
        out_idx[out] = in_idx0[i0];
        i0++;
        out++;
    }

    যাবত্ (i1 < n1) {
        যদি (out_val) out_val[out] = in_val1[i1];
        out_idx[out] = in_idx1[i1];
        i1++;
        out++;
    }
}

সংরচনা htp_sort_chunk_ctx {
    সংরচনা htp_ops_context *              octx;
    ধ্রুব সংরচনা htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              real_গুনতি[HTP_MAX_NTHREADS];
};

স্থির শূন্য htp_sort_chunk_job(unsigned পূর্ণাঙ্ক n, unsigned পূর্ণাঙ্ক i, শূন্য * data) {
    সংরচনা htp_sort_chunk_ctx * cctx = (সংরচনা htp_sort_chunk_ctx *) data;
    সংরচনা htp_ops_context * octx = cctx->octx;
    ধ্রুব সংরচনা htp_sort_kernel_params * kparams = cctx->kparams;
    ধ্রুব সংরচনা htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_base  = i * chunk_elems;

    uint32_t real_গুনতি = (chunk_base < ne00) ? MIN(chunk_elems, ne00 - chunk_base) : 0;
    cctx->real_গুনতি[i] = real_গুনতি;

    uint8_t * spad = cctx->vtcm_base + (size_t) kparams->phase1_slot_size * i;
    ভিন্ন *   মানসমূহ_buf  = (ভিন্ন *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * পরিমাণ(ভিন্ন));

    ভিন্ন *   merge_মানসমূহ  = (ভিন্ন *)   (cctx->vtcm_base + kparams->merge_মানসমূহ_off);
    int32_t * merge_indices = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off);

    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[i];
    ধ্রুব dma_addr_t src_addr = src0->data + (size_t) chunk_base * পরিমাণ(ভিন্ন);

    যদি (kparams->is_শীর্ষ_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        যদি (real_গুনতি == 0) {
            প্রতি (uint32_t j = 0; j < local_k; j++) {
                merge_মানসমূহ[i * local_k + j]  = -INFINITY;
                merge_indices[i * local_k + j] = 0;
            }
            প্রত্যাগম;
        }

        যদি (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], মানসমূহ_buf, src_addr, real_গুনতি * পরিমাণ(ভিন্ন));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            ভিন্ন max_মান;
            uint32_t max_index = শীর্ষ_k_max_মান_index(মানসমূহ_buf, real_গুনতি, &max_মান);
            merge_মানসমূহ[i] = max_মান;
            merge_indices[i] = (int32_t) (max_index + chunk_base);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            প্রত্যাগম;
        }

        যদি (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            শীর্ষ_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_গুনতি, local_k,
                               মানসমূহ_buf, indices_buf,
                               &merge_মানসমূহ[i * local_k], &merge_indices[i * local_k],
                               chunk_base);
            প্রত্যাগম;
        }

        dma_sync_read(octx->ctx->dma[i], মানসমূহ_buf, src_addr, real_গুনতি * পরিমাণ(ভিন্ন));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        যদি (chunk_elems > real_গুনতি) {
            hvx_splat_f32_u((uint8_t *)(মানসমূহ_buf + real_গুনতি), -INFINITY, chunk_elems - real_গুনতি);
        }

        bitonic_sort_vtcm_desc((uint8_t *) মানসমূহ_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        প্রতি (uint32_t j = 0; j < local_k; j++) {
            merge_মানসমূহ[i * local_k + j]  = মানসমূহ_buf[j];
            merge_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_base;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } অন্যথা {
        যদি (real_গুনতি > 0) {
            dma_sync_read(octx->ctx->dma[i], মানসমূহ_buf, src_addr, real_গুনতি * পরিমাণ(ভিন্ন));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_গুনতি, chunk_base);

            যদি (kparams->order == GGML_SORT_ORDER_ASC) {
                quicksort_মানসমূহ_indices_asc(মানসমূহ_buf, indices_buf, 0, real_গুনতি - 1);
            } অন্যথা {
                quicksort_মানসমূহ_indices_desc(মানসমূহ_buf, indices_buf, 0, real_গুনতি - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

সংরচনা htp_argsort_merge_l1_ctx {
    সংরচনা htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

স্থির শূন্য htp_argsort_merge_l1_job(unsigned পূর্ণাঙ্ক n, unsigned পূর্ণাঙ্ক i, শূন্য * data) {
    সংরচনা htp_argsort_merge_l1_ctx * mctx = (সংরচনা htp_argsort_merge_l1_ctx *) data;
    সংরচনা htp_sort_chunk_ctx * cctx = mctx->cctx;
    ধ্রুব সংরচনা htp_sort_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_গুনতি[c0];
    uint32_t count1 = cctx->real_গুনতি[c1];

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    ধ্রুব ভিন্ন *   val0 = (ধ্রুব ভিন্ন *)   (cctx->vtcm_base + slot_size * c0);
    ধ্রুব int32_t * idx0 = (ধ্রুব int32_t *) (cctx->vtcm_base + slot_size * c0 + chunk_elems * পরিমাণ(ভিন্ন));
    ধ্রুব ভিন্ন *   val1 = (ধ্রুব ভিন্ন *)   (cctx->vtcm_base + slot_size * c1);
    ধ্রুব int32_t * idx1 = (ধ্রুব int32_t *) (cctx->vtcm_base + slot_size * c1 + chunk_elems * পরিমাণ(ভিন্ন));

    uint32_t out_offset = (2 * i) * chunk_elems;
    ভিন্ন *   out_val = (ভিন্ন *)   (cctx->vtcm_base + kparams->merge_মানসমূহ_off)  + out_offset;
    int32_t * out_idx = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + out_offset;

    সংরচনা htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

সংরচনা htp_argsort_merge_l2_ctx {
    সংরচনা htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

স্থির শূন্য htp_argsort_merge_l2_job(unsigned পূর্ণাঙ্ক n, unsigned পূর্ণাঙ্ক i, শূন্য * data) {
    সংরচনা htp_argsort_merge_l2_ctx * mctx = (সংরচনা htp_argsort_merge_l2_ctx *) data;
    সংরচনা htp_sort_chunk_ctx * cctx = mctx->cctx;
    ধ্রুব সংরচনা htp_sort_kernel_params * kparams = cctx->kparams;

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    ধ্রুব ভিন্ন *   val0 = (ধ্রুব ভিন্ন *)   (cctx->vtcm_base + kparams->merge_মানসমূহ_off)  + off0;
    ধ্রুব int32_t * idx0 = (ধ্রুব int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off0;
    ধ্রুব ভিন্ন *   val1 = (ধ্রুব ভিন্ন *)   (cctx->vtcm_base + kparams->merge_মানসমূহ_off)  + off1;
    ধ্রুব int32_t * idx1 = (ধ্রুব int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off1;

    uint32_t count0 = cctx->real_গুনতি[4*i + 0] + cctx->real_গুনতি[4*i + 1];
    uint32_t count1 = cctx->real_গুনতি[4*i + 2] + cctx->real_গুনতি[4*i + 3];

    uint8_t * slot_base = cctx->vtcm_base + (4 * i) * slot_size;
    ভিন্ন *   out_val   = (ভিন্ন *) slot_base;
    int32_t * out_idx   = (int32_t *) (slot_base + 4 * chunk_elems * পরিমাণ(ভিন্ন));

    সংরচনা htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

সংরচনা htp_sort_multi_row_ctx {
    সংরচনা htp_ops_context *              octx;
    ধ্রুব সংরচনা htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              nrows_per_thread;
};

সংরচনা htp_tensor_row_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

স্থির অন্তঃস্থ সংরচনা htp_tensor_row_coord init_row_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    প্রত্যাগম (সংরচনা htp_tensor_row_coord){ i01, i02, i03 };
}

স্থির অন্তঃস্থ শূন্য step_row_coord(সংরচনা htp_tensor_row_coord * c, uint32_t ne01, uint32_t ne02) {
    c->i01++;
    যদি (c->i01 == ne01) {
        c->i01 = 0;
        c->i02++;
        যদি (c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

স্থির অন্তঃস্থ size_t coord_offset(সংরচনা htp_tensor_row_coord c, size_t nb1, size_t nb2, size_t nb3) {
    প্রত্যাগম (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

স্থির অন্তঃস্থ শূন্য compute_row_sort(
    ভিন্ন * cur_মানসমূহ,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_শীর্ষ_k,
    গণনা ggml_sort_order order
) {
    যদি (is_শীর্ষ_k) {
        যদি (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            বিকল্প (ne00) {
                পক্ষ 32:   sort32_f32_hvx((uint8_t *) cur_মানসমূহ, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ভঙ্গ;
                পক্ষ 64:   sort64_f32_hvx((uint8_t *) cur_মানসমূহ, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ভঙ্গ;
                পক্ষ 128:  sort128_f32_hvx((uint8_t *) cur_মানসমূহ, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ভঙ্গ;
                পক্ষ 256:  sort256_f32_hvx((uint8_t *) cur_মানসমূহ, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ভঙ্গ;
                পক্ষ 512:  sort512_f32_hvx((uint8_t *) cur_মানসমূহ, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ভঙ্গ;
                পক্ষ 1024: sort1024_f32_hvx((uint8_t *) cur_মানসমূহ, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); ভঙ্গ;
            }
        } অন্যথা {
            যদি (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_মানসমূহ + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_sort_vtcm_desc((uint8_t *) cur_মানসমূহ, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } অন্যথা {
        যদি (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            বিকল্প (ne00) {
                পক্ষ 32:   sort32_f32_hvx((uint8_t *) cur_মানসমূহ, (uint8_t *) cur_indices, order); ভঙ্গ;
                পক্ষ 64:   sort64_f32_hvx((uint8_t *) cur_মানসমূহ, (uint8_t *) cur_indices, order); ভঙ্গ;
                পক্ষ 128:  sort128_f32_hvx((uint8_t *) cur_মানসমূহ, (uint8_t *) cur_indices, order); ভঙ্গ;
                পক্ষ 256:  sort256_f32_hvx((uint8_t *) cur_মানসমূহ, (uint8_t *) cur_indices, order); ভঙ্গ;
                পক্ষ 512:  sort512_f32_hvx((uint8_t *) cur_মানসমূহ, (uint8_t *) cur_indices, order); ভঙ্গ;
                পক্ষ 1024: sort1024_f32_hvx((uint8_t *) cur_মানসমূহ, (uint8_t *) cur_indices, order); ভঙ্গ;
            }
        } অন্যথা {
            init_indices_ramp(cur_indices, ne00, 0);
            যদি (order == GGML_SORT_ORDER_ASC) {
                quicksort_মানসমূহ_indices_asc(cur_মানসমূহ, cur_indices, 0, ne00 - 1);
            } অন্যথা {
                quicksort_মানসমূহ_indices_desc(cur_মানসমূহ, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

স্থির শূন্য htp_sort_multi_row_job(unsigned পূর্ণাঙ্ক n, unsigned পূর্ণাঙ্ক ith, শূন্য * data) {
    সংরচনা htp_sort_multi_row_ctx * mctx = (সংরচনা htp_sort_multi_row_ctx *) data;
    সংরচনা htp_ops_context * octx = mctx->octx;
    ধ্রুব সংরচনা htp_sort_kernel_params * kparams = mctx->kparams;
    ধ্রুব সংরচনা htp_tensor * src0 = octx->src[0];
    ধ্রুব সংরচনা htp_tensor * dst  = octx->dst;

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

    যদি (r_start >= r_end) প্রত্যাগম;

    size_t spad_offset = ith * (size_t) (kparams->phase1_slot_size * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_base + spad_offset;
    size_t slot_size = (size_t) kparams->phase1_slot_size;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    ভিন্ন *   মানসমূহ_buf  = (ভিন্ন *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * পরিমাণ(ভিন্ন));

    dma_queue * dma_q = octx->ctx->dma[ith];
    সংরচনা htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_শীর্ষ_k = kparams->is_শীর্ষ_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    গণনা ggml_sort_order order = (গণনা ggml_sort_order) kparams->order;

    ধ্রুব uint32_t src_bytes = ne00 * পরিমাণ(ভিন্ন);
    ধ্রুব uint32_t dst_bytes = (is_শীর্ষ_k ? k : ne00) * পরিমাণ(int32_t);

    যদি (kparams->n_slots == 2) {
        সংরচনা htp_tensor_row_coord মৌলিক_coord = init_row_coord(r_start, ne01, ne02);
        প্রতি (uint32_t r = r_start, spad_idx = 0; r < r_end && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * slot_size;
            ভিন্ন *   cur_মানসমূহ  = (ভিন্ন *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * পরিমাণ(ভিন্ন));

            dma_addr_t dst_addr = dst->data + coord_offset(মৌলিক_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->data + coord_offset(মৌলিক_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_queue_ঠেলো(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 0);

            // Prefetch input row
            dma_queue_ঠেলো(dma_q, dma_make_data(cur_মানসমূহ, src_addr),
                           src_bytes, src_bytes, src_bytes, 1);

            step_row_coord(&মৌলিক_coord, ne01, ne02);
        }

        সংরচনা htp_tensor_row_coord dst_coord = init_row_coord(r_start, ne01, ne02);
        সংরচনা htp_tensor_row_coord next_src_coord = init_row_coord(r_start + 2, ne01, ne02);

        প্রতি (uint32_t r = r_start; r < r_end; r++) {
            uint32_t cur_slot = (r - r_start) & 1;
            uint8_t * cur_spad = thread_spad + cur_slot * slot_size;
            ভিন্ন *   cur_মানসমূহ  = (ভিন্ন *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * পরিমাণ(ভিন্ন));

            // Wait for previous writeback of this slot to complete
            dma_queue_বের_করো(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_queue_বের_করো(dma_q);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(cur_মানসমূহ, cur_indices, ne00, chunk_elems, is_শীর্ষ_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->data + coord_offset(dst_coord, nb1, nb2, nb3);
            dma_queue_ঠেলো(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 1);
            step_row_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            ধ্রুব uint32_t next_row = r + 2;
            যদি (next_row < r_end) {
                dma_addr_t next_src_addr = src0->data + coord_offset(next_src_coord, nb01, nb02, nb03);
                dma_queue_ঠেলো(dma_q, dma_make_data(cur_মানসমূহ, next_src_addr),
                               src_bytes, src_bytes, src_bytes, 1);
                step_row_coord(&next_src_coord, ne01, ne02);
            }
        }

        dma_queue_flush(dma_q);
    } অন্যথা {
        সংরচনা htp_tensor_row_coord coord = init_row_coord(r_start, ne01, ne02);
        প্রতি (uint32_t r = r_start; r < r_end; r++) {
            dma_addr_t src_addr = src0->data + coord_offset(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->data + coord_offset(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, মানসমূহ_buf, src_addr, src_bytes);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(মানসমূহ_buf, indices_buf, ne00, chunk_elems, is_শীর্ষ_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_bytes);
            step_row_coord(&coord, ne01, ne02);
        }
    }
}

স্থির পূর্ণাঙ্ক op_sort_common(সংরচনা htp_ops_context * octx, bool is_শীর্ষ_k) {
    যদি (octx->src[0]->type != HTP_TYPE_F32) {
        প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুব সংরচনা htp_tensor * src0 = octx->src[0];
    ধ্রুব সংরচনা htp_tensor * dst  = octx->dst;

    যদি (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        প্রত্যাগম HTP_STATUS_NO_SUPPORT;
    }

    ধ্রুব uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ধ্রুব size_t dst_row_size = dst->ne[0] * পরিমাণ(int32_t);

    uint32_t row_start = 0;
    uint32_t row_end   = total_rows;
    যদি (octx->ctx->mdev.গুনতি > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, পরিমাণ(int32_t), (uint32_t) dst_row_size, &rows_per_chunk);
        ধ্রুব সংরচনা htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.গুনতি, &octx->ctx->mdev.গুনতি_div);
        row_start = range.start;
        row_end   = range.start + range.গুনতি;
    }

    ধ্রুব uint32_t nrows = row_end - row_start;
    যদি (nrows == 0) {
        প্রত্যাগম HTP_STATUS_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    সংরচনা htp_sort_kernel_params kparams_local;
    ধ্রুব সংরচনা htp_sort_kernel_params * kparams = (ধ্রুব সংরচনা htp_sort_kernel_params *) octx->kernel_params;

    যদি (kparams->n_threads == 0) {
        সংরচনা htp_sort_vtcm_layout layout;
        যদি (!htp_sort_solve_layout(&layout, ne00, total_rows, k, octx->n_threads, octx->ctx->vtcm_size, is_শীর্ষ_k)) {
            প্রত্যাগম HTP_STATUS_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, পরিমাণ(kparams_local));
        kparams_local.n_threads         = (int32_t) layout.n_threads;
        kparams_local.total_rows        = (int32_t) total_rows;
        kparams_local.row_start         = (int32_t) row_start;
        kparams_local.row_end           = (int32_t) row_end;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.order             = is_শীর্ষ_k ? GGML_SORT_ORDER_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_শীর্ষ_k          = is_শীর্ষ_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_size         = (int32_t) layout.total_bytes;
        kparams_local.phase1_slot_size  = (int32_t) layout.phase1_slot_size;
        kparams_local.merge_মানসমূহ_off  = (int32_t) layout.merge_মানসমূহ_off;
        kparams_local.merge_indices_off = (int32_t) layout.merge_indices_off;
        kparams_local.merge_elems       = (int32_t) layout.merge_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } অন্যথা যদি (octx->ctx->mdev.গুনতি > 1) {
        memcpy(&kparams_local, kparams, পরিমাণ(kparams_local));
        kparams_local.row_start = (int32_t) row_start;
        kparams_local.row_end   = (int32_t) row_end;
        kparams = &kparams_local;
    }

    যদি (octx->ctx->vtcm_size < (size_t) kparams->vtcm_size) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_size, octx->ctx->vtcm_size);
        প্রত্যাগম HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    যদি (total_rows == 1 && kparams->n_chunks > 1) {
        সংরচনা htp_sort_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_base = vtcm_base;
        memset(cctx.real_গুনতি, 0, পরিমাণ(cctx.real_গুনতি));

        work_queue_run(octx->ctx->work_queue, htp_sort_chunk_job, &cctx, (uint32_t) kparams->n_chunks);

        যদি (is_শীর্ষ_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            যদি (k == 1 && ne00 >= 128*1024) {
                ভিন্ন *   cand_vals = (ভিন্ন *)   (vtcm_base + kparams->merge_মানসমূহ_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                ভিন্ন best_val = cand_vals[0];
                int32_t best_idx = cand_idxs[0];
                প্রতি (int32_t c = 1; c < kparams->n_chunks; c++) {
                    যদি (cand_vals[c] > best_val) {
                        best_val = cand_vals[c];
                        best_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->data;
                dst_ptr[0] = best_idx;
                প্রত্যাগম HTP_STATUS_OK;
            }

            uint32_t total_candidates = (uint32_t) kparams->n_chunks * local_k;
            ভিন্ন *   cand_vals = (ভিন্ন *)   (vtcm_base + kparams->merge_মানসমূহ_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
            uint32_t merge_elems = (uint32_t) kparams->merge_elems;

            যদি (merge_elems > total_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + total_candidates), -INFINITY, merge_elems - total_candidates);
                প্রতি (uint32_t j = total_candidates; j < merge_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            সংরচনা htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_sort_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, merge_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->data, cand_idxs, k * পরিমাণ(int32_t));
        } অন্যথা {
            bool asc = kparams->order == GGML_SORT_ORDER_ASC;
            যদি (kparams->n_chunks == 8) {
                সংরচনা htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 4);

                সংরচনা htp_argsort_merge_l2_ctx l2ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l2_job, &l2ctx, 2);

                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ধ্রুব ভিন্ন *   val0 = (ধ্রুব ভিন্ন *) vtcm_base;
                ধ্রুব int32_t * idx0 = (ধ্রুব int32_t *) (vtcm_base + 4 * chunk_elems * পরিমাণ(ভিন্ন));
                ধ্রুব ভিন্ন *   val1 = (ধ্রুব ভিন্ন *) (vtcm_base + 4 * slot_size);
                ধ্রুব int32_t * idx1 = (ধ্রুব int32_t *) (vtcm_base + 4 * slot_size + 4 * chunk_elems * পরিমাণ(ভিন্ন));

                uint32_t n01 = cctx.real_গুনতি[0] + cctx.real_গুনতি[1] + cctx.real_গুনতি[2] + cctx.real_গুনতি[3];
                uint32_t n23 = cctx.real_গুনতি[4] + cctx.real_গুনতি[5] + cctx.real_গুনতি[6] + cctx.real_গুনতি[7];

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                সংরচনা htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, n01, val1, idx1, n23, রিক্ত, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * পরিমাণ(int32_t));
            } অন্যথা যদি (kparams->n_chunks == 4) {
                সংরচনা htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_গুনতি[0] + cctx.real_গুনতি[1];
                uint32_t n23 = cctx.real_গুনতি[2] + cctx.real_গুনতি[3];

                ধ্রুব ভিন্ন *   val01 = (ধ্রুব ভিন্ন *)   (vtcm_base + kparams->merge_মানসমূহ_off);
                ধ্রুব int32_t * idx01 = (ধ্রুব int32_t *) (vtcm_base + kparams->merge_indices_off);
                ধ্রুব ভিন্ন *   val23 = (ধ্রুব ভিন্ন *)   (vtcm_base + kparams->merge_মানসমূহ_off)  + 2 * chunk_elems;
                ধ্রুব int32_t * idx23 = (ধ্রুব int32_t *) (vtcm_base + kparams->merge_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_base;
                সংরচনা htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val01, idx01, n01, val23, idx23, n23, রিক্ত, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * পরিমাণ(int32_t));
            } অন্যথা যদি (kparams->n_chunks == 2) {
                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ধ্রুব ভিন্ন *   val0 = (ধ্রুব ভিন্ন *)   (vtcm_base);
                ধ্রুব int32_t * idx0 = (ধ্রুব int32_t *) (vtcm_base + chunk_elems * পরিমাণ(ভিন্ন));
                ধ্রুব ভিন্ন *   val1 = (ধ্রুব ভিন্ন *)   (vtcm_base + slot_size);
                ধ্রুব int32_t * idx1 = (ধ্রুব int32_t *) (vtcm_base + slot_size + chunk_elems * পরিমাণ(ভিন্ন));

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                সংরচনা htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, cctx.real_গুনতি[0], val1, idx1, cctx.real_গুনতি[1], রিক্ত, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * পরিমাণ(int32_t));
            } অন্যথা {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_base + chunk_elems * পরিমাণ(ভিন্ন));
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * পরিমাণ(int32_t));
            }
        }
        প্রত্যাগম HTP_STATUS_OK;
    }

    সংরচনা htp_sort_multi_row_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_base        = vtcm_base;

    uint32_t dst_row_bytes = (uint32_t) dst_row_size;
    uint32_t rows_per_chunk = 0;
    htp_tensor_mdev_rows_per_chunk(dst, পরিমাণ(int32_t), dst_row_bytes, &rows_per_chunk);
    যদি (rows_per_chunk == 0) {
        rows_per_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_threads - 1) / (uint32_t) kparams->n_threads;
    যদি (rows_per_chunk > 1) {
        dr = hex_round_up(dr, rows_per_chunk);
    }
    mctx.nrows_per_thread = dr;

    work_queue_run(octx->ctx->work_queue, htp_sort_multi_row_job, &mctx, (uint32_t) kparams->n_threads);

    প্রত্যাগম HTP_STATUS_OK;
}

পূর্ণাঙ্ক op_argsort(সংরচনা htp_ops_context * octx) {
    প্রত্যাগম op_sort_common(octx, false);
}

পূর্ণাঙ্ক op_শীর্ষ_k(সংরচনা htp_ops_context * octx) {
    প্রত্যাগম op_sort_common(octx, true);
}
