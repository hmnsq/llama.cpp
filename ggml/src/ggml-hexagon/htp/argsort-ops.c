#شامل <string.h>
#شامل <stdlib.h>
#شامل <math.h>
#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#define GGML_COMMON_DECL_C
#شامل "ggml-common.h"
#شامل "ggml.h"

#شامل "hvx-utils.h"
#شامل "hvx-copy.h"
#شامل "hvx-reduce.h"
#شامل "dma-queue.h"

#شامل "hex-common.h"
#شامل "htp-ctx.h"
#شامل "htp-ops.h"
#شامل "htp-tensor.h"
#شامل "argsort-ops.h"

#ifndef MIN
#define MIN(a, b) ((a) < (b) ? (a) : (b))
#endif


ایستا تهی quicksort_مقادیر_indices_asc(اعشاری * مقادیر, int32_t * indices, صحیح چپ, صحیح راست) {
    وقتی (چپ < راست) {
        اعشاری pivot = مقادیر[چپ + (راست - چپ) / 2];
        صحیح i = چپ;
        صحیح j = راست;

        وقتی (i <= j) {
            وقتی (مقادیر[i] < pivot) i++;
            وقتی (مقادیر[j] > pivot) j--;

            اگر (i <= j) {
                اعشاری tmp_val = مقادیر[i];
                مقادیر[i] = مقادیر[j];
                مقادیر[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        اگر (j - چپ < راست - i) {
            اگر (چپ < j) quicksort_مقادیر_indices_asc(مقادیر, indices, چپ, j);
            چپ = i;
        } وگرنه {
            اگر (i < راست) quicksort_مقادیر_indices_asc(مقادیر, indices, i, راست);
            راست = j;
        }
    }
}

ایستا تهی quicksort_مقادیر_indices_desc(اعشاری * مقادیر, int32_t * indices, صحیح چپ, صحیح راست) {
    وقتی (چپ < راست) {
        اعشاری pivot = مقادیر[چپ + (راست - چپ) / 2];
        صحیح i = چپ;
        صحیح j = راست;

        وقتی (i <= j) {
            وقتی (مقادیر[i] > pivot) i++;
            وقتی (مقادیر[j] < pivot) j--;

            اگر (i <= j) {
                اعشاری tmp_val = مقادیر[i];
                مقادیر[i] = مقادیر[j];
                مقادیر[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        اگر (j - چپ < راست - i) {
            اگر (چپ < j) quicksort_مقادیر_indices_desc(مقادیر, indices, چپ, j);
            چپ = i;
        } وگرنه {
            اگر (i < راست) quicksort_مقادیر_indices_desc(مقادیر, indices, i, راست);
            راست = j;
        }
    }
}

ایستا درونخطی uint32_t بالاترین_k_max_مقدار_index(ثابت اعشاری * مقادیر, uint32_t n, اعشاری * مقدار) {
    int32_t max_idx = 0;
    hvx_argmax_f32(مقادیر, n, 0, مقدار, &max_idx);
    بازگشت (uint32_t) max_idx;
}

ایستا ثابت int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

ایستا درونخطی تهی init_indices_ramp(int32_t * indices_buf, uint32_t شمار, uint32_t base) {
    ثابت HVX_Vector ind_init_vec = Q6_Vw_vadd_VwVw(*(ثابت HVX_Vector *) argsort_ramp_lut, Q6_V_vsplat_R(base));
    ثابت HVX_Vector ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_Vector * indices_buf_vec = (HVX_Vector *) indices_buf;
    uint32_t رقم_vecs = شمار / 32;
    HVX_Vector curr_ind_vec = ind_init_vec;
    برای (uint32_t j = 0; j < رقم_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = شمار % 32;
    اگر (rem > 0) {
        hvx_vec_store_a((تهی *) &indices_buf_vec[رقم_vecs], rem * اندازه(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
ایستا درونخطی تهی vec_cas(HVX_Vector * X_val, HVX_Vector * X_idx, HVX_Vector * Y_val, HVX_Vector * Y_idx, bool asc) {
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
ایستا درونخطی تهی bitonic_cas_32(HVX_Vector * V, HVX_Vector * I, صحیح d, HVX_VectorPred dir_mask, HVX_Vector idx_vec, HVX_Vector zero_vec) {
    HVX_VectorPred mask_چپ;
    HVX_Vector V_rot_چپ, V_rot_راست;
    HVX_Vector I_rot_چپ, I_rot_راست;

    اگر (d == 1) {
        mask_چپ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_چپ = Q6_V_vror_VR(*V, 4);
        V_rot_راست = Q6_V_vror_VR(*V, 124);
        I_rot_چپ = Q6_V_vror_VR(*I, 4);
        I_rot_راست = Q6_V_vror_VR(*I, 124);
    } وگرنه اگر (d == 2) {
        mask_چپ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_چپ = Q6_V_vror_VR(*V, 8);
        V_rot_راست = Q6_V_vror_VR(*V, 120);
        I_rot_چپ = Q6_V_vror_VR(*I, 8);
        I_rot_راست = Q6_V_vror_VR(*I, 120);
    } وگرنه اگر (d == 4) {
        mask_چپ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_چپ = Q6_V_vror_VR(*V, 16);
        V_rot_راست = Q6_V_vror_VR(*V, 112);
        I_rot_چپ = Q6_V_vror_VR(*I, 16);
        I_rot_راست = Q6_V_vror_VR(*I, 112);
    } وگرنه اگر (d == 8) {
        mask_چپ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_چپ = Q6_V_vror_VR(*V, 32);
        V_rot_راست = Q6_V_vror_VR(*V, 96);
        I_rot_چپ = Q6_V_vror_VR(*I, 32);
        I_rot_راست = Q6_V_vror_VR(*I, 96);
    } وگرنه {
        mask_چپ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_چپ = Q6_V_vror_VR(*V, 64);
        V_rot_راست = Q6_V_vror_VR(*V, 64);
        I_rot_چپ = Q6_V_vror_VR(*I, 64);
        I_rot_راست = Q6_V_vror_VR(*I, 64);
    }

    HVX_Vector V_paired = Q6_V_vmux_QVV(mask_چپ, V_rot_چپ, V_rot_راست);
    HVX_Vector I_paired = Q6_V_vmux_QVV(mask_چپ, I_rot_چپ, I_rot_راست);

    HVX_VectorPred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_VectorPred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_VectorPred mask_راست = Q6_Q_not_Q(mask_چپ);
    HVX_VectorPred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(mask_چپ, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, mask_راست)
    );
    HVX_VectorPred Q_swap = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_mask, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_mask), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_swap, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_swap, I_paired, *I);
}

__attribute__((always_inline))
ایستا درونخطی تهی bitonic_sort_generic_hvx(uint8_t * مقادیر, uint8_t * indices, صحیح K, bool asc_order) {
    HVX_Vector V[32];
    HVX_Vector I[32];

    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(ثابت HVX_Vector *) argsort_ramp_lut;

    برای (صحیح v = 0; v < K; v++) {
        V[v] = *(HVX_Vector *)(مقادیر + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    صحیح M = 5;
    وقتی ((1 << (M - 5)) < K) M++;

    برای (صحیح s = 1; s <= M; s++) {
        برای (صحیح stage_d = s - 1; stage_d >= 0; stage_d--) {
            صحیح d = 1 << stage_d;
            اگر (d >= 32) {
                صحیح v_dist = d / 32;
                برای (صحیح v1 = 0; v1 < K; v1++) {
                    اگر ((v1 & v_dist) == 0) {
                        صحیح v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_order;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } وگرنه {
                اگر (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    برای (صحیح v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                } وگرنه {
                    برای (صحیح v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_order;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_mask, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    برای (صحیح v = 0; v < K; v++) {
        *(HVX_Vector *)(مقادیر + v * 128)  = V[v];
        *(HVX_Vector *)(indices + v * 128) = I[v];
    }
}

ایستا تهی bitonic_sort_vtcm_desc(uint8_t * مقادیر, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_Vector zero_vec = Q6_V_vzero();
    HVX_Vector idx_vec = *(ثابت HVX_Vector *) argsort_ramp_lut;

    HVX_VectorPred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_VectorPred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    اگر (init_indices) {
        برای (uint32_t v = 0; v < n_vec; v++) {
            HVX_Vector idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_Vector *)(indices + v * 128) = idx;
        }
    }

    صحیح M = 5;
    وقتی ((1u << (M - 5)) < n_vec) M++;

    برای (صحیح s = 1; s <= M; s++) {
        برای (صحیح stage_d = s - 1; stage_d >= 0; stage_d--) {
            صحیح d = 1 << stage_d;
            اگر (d >= 32) {
                uint32_t v_dist = d / 32;
                برای (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    اگر ((v1 & v_dist) == 0) {
                        uint32_t v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : false;

                        HVX_Vector Vv1 = *(HVX_Vector *)(مقادیر + v1 * 128);
                        HVX_Vector Iv1 = *(HVX_Vector *)(indices + v1 * 128);
                        HVX_Vector Vv2 = *(HVX_Vector *)(مقادیر + v2 * 128);
                        HVX_Vector Iv2 = *(HVX_Vector *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_Vector *)(مقادیر + v1 * 128)  = Vv1;
                        *(HVX_Vector *)(indices + v1 * 128) = Iv1;
                        *(HVX_Vector *)(مقادیر + v2 * 128)  = Vv2;
                        *(HVX_Vector *)(indices + v2 * 128) = Iv2;
                    }
                }
            } وگرنه {
                اگر (s < 5) {
                    HVX_VectorPred dir_mask = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    برای (uint32_t v = 0; v < n_vec; v++) {
                        HVX_Vector Vv = *(HVX_Vector *)(مقادیر + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(مقادیر + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                } وگرنه {
                    برای (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_VectorPred dir_mask = asc ? pred_all_1s : pred_all_0s;

                        HVX_Vector Vv = *(HVX_Vector *)(مقادیر + v * 128);
                        HVX_Vector Iv = *(HVX_Vector *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_mask, idx_vec, zero_vec);

                        *(HVX_Vector *)(مقادیر + v * 128)  = Vv;
                        *(HVX_Vector *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

ایستا تهی بالاترین_k_select_tiled(dma_queue * dma_q, ساختار htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               اعشاری * مقادیر_buf, int32_t * indices_buf,
                               اعشاری * out_مقادیر, int32_t * out_indices,
                               uint32_t chunk_base) {
    ثابت uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_شمار    = n_tiles * k;
    uint32_t merge_n_vec        = hmx_ceil_div(candidate_شمار, 32);
    uint32_t merge_n_vec_pow2   = 1;
    وقتی (merge_n_vec_pow2 < merge_n_vec) merge_n_vec_pow2 <<= 1;
    uint32_t merge_elems        = merge_n_vec_pow2 * 32;
    اعشاری * candidate_مقادیر    = مقادیر_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    برای (uint32_t offset = 0; offset < n; offset += tile_elems) {
        uint32_t tile_شمار = MIN(tile_elems, n - offset);
        dma_sync_read(dma_q, مقادیر_buf, src_addr + (size_t) offset * اندازه(اعشاری), tile_شمار * اندازه(اعشاری));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
        اگر (tile_شمار < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (مقادیر_buf + tile_شمار), -INFINITY, tile_elems - tile_شمار);
        }

        bitonic_sort_vtcm_desc((uint8_t *) مقادیر_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_شمار);
        برای (uint32_t j = 0; j < tile_k; j++) {
            candidate_مقادیر[candidate_pos] = مقادیر_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) offset;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) offset);
    }

    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    اگر (merge_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_مقادیر + candidate_pos), -INFINITY, merge_elems - candidate_pos);
        برای (uint32_t j = candidate_pos; j < merge_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_sort_vtcm_desc((uint8_t *) candidate_مقادیر, (uint8_t *) candidate_indices, merge_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    برای (uint32_t j = 0; j < k; j++) {
        out_مقادیر[j]  = candidate_مقادیر[j];
        out_indices[j] = candidate_indices[j] + (int32_t) chunk_base;
    }
}

__attribute__((always_inline))
ایستا درونخطی تهی sort32_f32_hvx(uint8_t * مقادیر, uint8_t * indices, شمارش ggml_sort_order order) {
    bitonic_sort_generic_hvx(مقادیر, indices, 1, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ایستا درونخطی تهی sort64_f32_hvx(uint8_t * مقادیر, uint8_t * indices, شمارش ggml_sort_order order) {
    bitonic_sort_generic_hvx(مقادیر, indices, 2, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ایستا درونخطی تهی sort128_f32_hvx(uint8_t * مقادیر, uint8_t * indices, شمارش ggml_sort_order order) {
    bitonic_sort_generic_hvx(مقادیر, indices, 4, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ایستا درونخطی تهی sort256_f32_hvx(uint8_t * مقادیر, uint8_t * indices, شمارش ggml_sort_order order) {
    bitonic_sort_generic_hvx(مقادیر, indices, 8, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ایستا درونخطی تهی sort512_f32_hvx(uint8_t * مقادیر, uint8_t * indices, شمارش ggml_sort_order order) {
    bitonic_sort_generic_hvx(مقادیر, indices, 16, order == GGML_SORT_ORDER_ASC);
}

__attribute__((always_inline))
ایستا درونخطی تهی sort1024_f32_hvx(uint8_t * مقادیر, uint8_t * indices, شمارش ggml_sort_order order) {
    bitonic_sort_generic_hvx(مقادیر, indices, 32, order == GGML_SORT_ORDER_ASC);
}

ایستا تهی merge_runs(
    ثابت اعشاری * restrict in_val0, ثابت int32_t * restrict in_idx0, uint32_t n0,
    ثابت اعشاری * restrict in_val1, ثابت int32_t * restrict in_idx1, uint32_t n1,
    اعشاری * restrict out_val, int32_t * restrict out_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t out = 0;

    اگر (asc) {
        وقتی (i0 < n0 && i1 < n1) {
            اگر (in_val0[i0] <= in_val1[i1]) {
                اگر (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } وگرنه {
                اگر (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    } وگرنه {
        وقتی (i0 < n0 && i1 < n1) {
            اگر (in_val0[i0] >= in_val1[i1]) {
                اگر (out_val) out_val[out] = in_val0[i0];
                out_idx[out] = in_idx0[i0];
                i0++;
            } وگرنه {
                اگر (out_val) out_val[out] = in_val1[i1];
                out_idx[out] = in_idx1[i1];
                i1++;
            }
            out++;
        }
    }

    وقتی (i0 < n0) {
        اگر (out_val) out_val[out] = in_val0[i0];
        out_idx[out] = in_idx0[i0];
        i0++;
        out++;
    }

    وقتی (i1 < n1) {
        اگر (out_val) out_val[out] = in_val1[i1];
        out_idx[out] = in_idx1[i1];
        i1++;
        out++;
    }
}

ساختار htp_sort_chunk_ctx {
    ساختار htp_ops_context *              octx;
    ثابت ساختار htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              real_شمار[HTP_MAX_NTHREADS];
};

ایستا تهی htp_sort_chunk_job(unsigned صحیح n, unsigned صحیح i, تهی * data) {
    ساختار htp_sort_chunk_ctx * cctx = (ساختار htp_sort_chunk_ctx *) data;
    ساختار htp_ops_context * octx = cctx->octx;
    ثابت ساختار htp_sort_kernel_params * kparams = cctx->kparams;
    ثابت ساختار htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_base  = i * chunk_elems;

    uint32_t real_شمار = (chunk_base < ne00) ? MIN(chunk_elems, ne00 - chunk_base) : 0;
    cctx->real_شمار[i] = real_شمار;

    uint8_t * spad = cctx->vtcm_base + (size_t) kparams->phase1_slot_size * i;
    اعشاری *   مقادیر_buf  = (اعشاری *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * اندازه(اعشاری));

    اعشاری *   merge_مقادیر  = (اعشاری *)   (cctx->vtcm_base + kparams->merge_مقادیر_off);
    int32_t * merge_indices = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off);

    ساختار htp_thread_trace * tr = &octx->ctx->trace[i];
    ثابت dma_addr_t src_addr = src0->data + (size_t) chunk_base * اندازه(اعشاری);

    اگر (kparams->is_بالاترین_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        اگر (real_شمار == 0) {
            برای (uint32_t j = 0; j < local_k; j++) {
                merge_مقادیر[i * local_k + j]  = -INFINITY;
                merge_indices[i * local_k + j] = 0;
            }
            بازگشت;
        }

        اگر (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], مقادیر_buf, src_addr, real_شمار * اندازه(اعشاری));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            اعشاری max_مقدار;
            uint32_t max_index = بالاترین_k_max_مقدار_index(مقادیر_buf, real_شمار, &max_مقدار);
            merge_مقادیر[i] = max_مقدار;
            merge_indices[i] = (int32_t) (max_index + chunk_base);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            بازگشت;
        }

        اگر (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            بالاترین_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_شمار, local_k,
                               مقادیر_buf, indices_buf,
                               &merge_مقادیر[i * local_k], &merge_indices[i * local_k],
                               chunk_base);
            بازگشت;
        }

        dma_sync_read(octx->ctx->dma[i], مقادیر_buf, src_addr, real_شمار * اندازه(اعشاری));

        htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        اگر (chunk_elems > real_شمار) {
            hvx_splat_f32_u((uint8_t *)(مقادیر_buf + real_شمار), -INFINITY, chunk_elems - real_شمار);
        }

        bitonic_sort_vtcm_desc((uint8_t *) مقادیر_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        برای (uint32_t j = 0; j < local_k; j++) {
            merge_مقادیر[i * local_k + j]  = مقادیر_buf[j];
            merge_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_base;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } وگرنه {
        اگر (real_شمار > 0) {
            dma_sync_read(octx->ctx->dma[i], مقادیر_buf, src_addr, real_شمار * اندازه(اعشاری));

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_شمار, chunk_base);

            اگر (kparams->order == GGML_SORT_ORDER_ASC) {
                quicksort_مقادیر_indices_asc(مقادیر_buf, indices_buf, 0, real_شمار - 1);
            } وگرنه {
                quicksort_مقادیر_indices_desc(مقادیر_buf, indices_buf, 0, real_شمار - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

ساختار htp_argsort_merge_l1_ctx {
    ساختار htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

ایستا تهی htp_argsort_merge_l1_job(unsigned صحیح n, unsigned صحیح i, تهی * data) {
    ساختار htp_argsort_merge_l1_ctx * mctx = (ساختار htp_argsort_merge_l1_ctx *) data;
    ساختار htp_sort_chunk_ctx * cctx = mctx->cctx;
    ثابت ساختار htp_sort_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_شمار[c0];
    uint32_t count1 = cctx->real_شمار[c1];

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    ثابت اعشاری *   val0 = (ثابت اعشاری *)   (cctx->vtcm_base + slot_size * c0);
    ثابت int32_t * idx0 = (ثابت int32_t *) (cctx->vtcm_base + slot_size * c0 + chunk_elems * اندازه(اعشاری));
    ثابت اعشاری *   val1 = (ثابت اعشاری *)   (cctx->vtcm_base + slot_size * c1);
    ثابت int32_t * idx1 = (ثابت int32_t *) (cctx->vtcm_base + slot_size * c1 + chunk_elems * اندازه(اعشاری));

    uint32_t out_offset = (2 * i) * chunk_elems;
    اعشاری *   out_val = (اعشاری *)   (cctx->vtcm_base + kparams->merge_مقادیر_off)  + out_offset;
    int32_t * out_idx = (int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + out_offset;

    ساختار htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

ساختار htp_argsort_merge_l2_ctx {
    ساختار htp_sort_chunk_ctx * cctx;
    bool                        asc;
};

ایستا تهی htp_argsort_merge_l2_job(unsigned صحیح n, unsigned صحیح i, تهی * data) {
    ساختار htp_argsort_merge_l2_ctx * mctx = (ساختار htp_argsort_merge_l2_ctx *) data;
    ساختار htp_sort_chunk_ctx * cctx = mctx->cctx;
    ثابت ساختار htp_sort_kernel_params * kparams = cctx->kparams;

    size_t slot_size = (size_t) kparams->phase1_slot_size;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    ثابت اعشاری *   val0 = (ثابت اعشاری *)   (cctx->vtcm_base + kparams->merge_مقادیر_off)  + off0;
    ثابت int32_t * idx0 = (ثابت int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off0;
    ثابت اعشاری *   val1 = (ثابت اعشاری *)   (cctx->vtcm_base + kparams->merge_مقادیر_off)  + off1;
    ثابت int32_t * idx1 = (ثابت int32_t *) (cctx->vtcm_base + kparams->merge_indices_off) + off1;

    uint32_t count0 = cctx->real_شمار[4*i + 0] + cctx->real_شمار[4*i + 1];
    uint32_t count1 = cctx->real_شمار[4*i + 2] + cctx->real_شمار[4*i + 3];

    uint8_t * slot_base = cctx->vtcm_base + (4 * i) * slot_size;
    اعشاری *   out_val   = (اعشاری *) slot_base;
    int32_t * out_idx   = (int32_t *) (slot_base + 4 * chunk_elems * اندازه(اعشاری));

    ساختار htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    merge_runs(val0, idx0, count0, val1, idx1, count1, out_val, out_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

ساختار htp_sort_multi_row_ctx {
    ساختار htp_ops_context *              octx;
    ثابت ساختار htp_sort_kernel_params * kparams;
    uint8_t *                             vtcm_base;
    uint32_t                              nrows_per_thread;
};

ساختار htp_tensor_row_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ایستا درونخطی ساختار htp_tensor_row_coord init_row_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    بازگشت (ساختار htp_tensor_row_coord){ i01, i02, i03 };
}

ایستا درونخطی تهی step_row_coord(ساختار htp_tensor_row_coord * c, uint32_t ne01, uint32_t ne02) {
    c->i01++;
    اگر (c->i01 == ne01) {
        c->i01 = 0;
        c->i02++;
        اگر (c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

ایستا درونخطی size_t coord_offset(ساختار htp_tensor_row_coord c, size_t nb1, size_t nb2, size_t nb3) {
    بازگشت (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

ایستا درونخطی تهی compute_row_sort(
    اعشاری * cur_مقادیر,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_بالاترین_k,
    شمارش ggml_sort_order order
) {
    اگر (is_بالاترین_k) {
        اگر (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            انتخاب (ne00) {
                حالت 32:   sort32_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); بشکن;
                حالت 64:   sort64_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); بشکن;
                حالت 128:  sort128_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); بشکن;
                حالت 256:  sort256_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); بشکن;
                حالت 512:  sort512_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); بشکن;
                حالت 1024: sort1024_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, GGML_SORT_ORDER_DESC); بشکن;
            }
        } وگرنه {
            اگر (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_مقادیر + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_sort_vtcm_desc((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } وگرنه {
        اگر (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            انتخاب (ne00) {
                حالت 32:   sort32_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, order); بشکن;
                حالت 64:   sort64_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, order); بشکن;
                حالت 128:  sort128_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, order); بشکن;
                حالت 256:  sort256_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, order); بشکن;
                حالت 512:  sort512_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, order); بشکن;
                حالت 1024: sort1024_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, order); بشکن;
            }
        } وگرنه {
            init_indices_ramp(cur_indices, ne00, 0);
            اگر (order == GGML_SORT_ORDER_ASC) {
                quicksort_مقادیر_indices_asc(cur_مقادیر, cur_indices, 0, ne00 - 1);
            } وگرنه {
                quicksort_مقادیر_indices_desc(cur_مقادیر, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

ایستا تهی htp_sort_multi_row_job(unsigned صحیح n, unsigned صحیح ith, تهی * data) {
    ساختار htp_sort_multi_row_ctx * mctx = (ساختار htp_sort_multi_row_ctx *) data;
    ساختار htp_ops_context * octx = mctx->octx;
    ثابت ساختار htp_sort_kernel_params * kparams = mctx->kparams;
    ثابت ساختار htp_tensor * src0 = octx->src[0];
    ثابت ساختار htp_tensor * dst  = octx->dst;

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

    اگر (r_start >= r_end) بازگشت;

    size_t spad_offset = ith * (size_t) (kparams->phase1_slot_size * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_base + spad_offset;
    size_t slot_size = (size_t) kparams->phase1_slot_size;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    اعشاری *   مقادیر_buf  = (اعشاری *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * اندازه(اعشاری));

    dma_queue * dma_q = octx->ctx->dma[ith];
    ساختار htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_بالاترین_k = kparams->is_بالاترین_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    شمارش ggml_sort_order order = (شمارش ggml_sort_order) kparams->order;

    ثابت uint32_t src_bytes = ne00 * اندازه(اعشاری);
    ثابت uint32_t dst_bytes = (is_بالاترین_k ? k : ne00) * اندازه(int32_t);

    اگر (kparams->n_slots == 2) {
        ساختار htp_tensor_row_coord اول_coord = init_row_coord(r_start, ne01, ne02);
        برای (uint32_t r = r_start, spad_idx = 0; r < r_end && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * slot_size;
            اعشاری *   cur_مقادیر  = (اعشاری *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * اندازه(اعشاری));

            dma_addr_t dst_addr = dst->data + coord_offset(اول_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->data + coord_offset(اول_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_queue_افزودن(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 0);

            // Prefetch input row
            dma_queue_افزودن(dma_q, dma_make_data(cur_مقادیر, src_addr),
                           src_bytes, src_bytes, src_bytes, 1);

            step_row_coord(&اول_coord, ne01, ne02);
        }

        ساختار htp_tensor_row_coord dst_coord = init_row_coord(r_start, ne01, ne02);
        ساختار htp_tensor_row_coord next_src_coord = init_row_coord(r_start + 2, ne01, ne02);

        برای (uint32_t r = r_start; r < r_end; r++) {
            uint32_t cur_slot = (r - r_start) & 1;
            uint8_t * cur_spad = thread_spad + cur_slot * slot_size;
            اعشاری *   cur_مقادیر  = (اعشاری *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * اندازه(اعشاری));

            // Wait for previous writeback of this slot to complete
            dma_queue_برداشت(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_queue_برداشت(dma_q);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(cur_مقادیر, cur_indices, ne00, chunk_elems, is_بالاترین_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->data + coord_offset(dst_coord, nb1, nb2, nb3);
            dma_queue_افزودن(dma_q, dma_make_data(dst_addr, cur_indices),
                           dst_bytes, dst_bytes, dst_bytes, 1);
            step_row_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            ثابت uint32_t next_row = r + 2;
            اگر (next_row < r_end) {
                dma_addr_t next_src_addr = src0->data + coord_offset(next_src_coord, nb01, nb02, nb03);
                dma_queue_افزودن(dma_q, dma_make_data(cur_مقادیر, next_src_addr),
                               src_bytes, src_bytes, src_bytes, 1);
                step_row_coord(&next_src_coord, ne01, ne02);
            }
        }

        dma_queue_flush(dma_q);
    } وگرنه {
        ساختار htp_tensor_row_coord coord = init_row_coord(r_start, ne01, ne02);
        برای (uint32_t r = r_start; r < r_end; r++) {
            dma_addr_t src_addr = src0->data + coord_offset(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->data + coord_offset(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, مقادیر_buf, src_addr, src_bytes);

            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_row_sort(مقادیر_buf, indices_buf, ne00, chunk_elems, is_بالاترین_k, order);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_bytes);
            step_row_coord(&coord, ne01, ne02);
        }
    }
}

ایستا صحیح op_sort_common(ساختار htp_ops_context * octx, bool is_بالاترین_k) {
    اگر (octx->src[0]->type != HTP_TYPE_F32) {
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    ثابت ساختار htp_tensor * src0 = octx->src[0];
    ثابت ساختار htp_tensor * dst  = octx->dst;

    اگر (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        بازگشت HTP_STATUS_NO_SUPPORT;
    }

    ثابت uint32_t total_rows = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_row_size = dst->ne[0] * اندازه(int32_t);

    uint32_t row_start = 0;
    uint32_t row_end   = total_rows;
    اگر (octx->ctx->mdev.شمار > 1) {
        uint32_t rows_per_chunk = 0;
        htp_tensor_mdev_rows_per_chunk(dst, اندازه(int32_t), (uint32_t) dst_row_size, &rows_per_chunk);
        ثابت ساختار htp_tensor_mdev_range range = htp_tensor_mdev_partition(
            total_rows, rows_per_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        row_start = range.start;
        row_end   = range.start + range.شمار;
    }

    ثابت uint32_t nrows = row_end - row_start;
    اگر (nrows == 0) {
        بازگشت HTP_STATUS_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    ساختار htp_sort_kernel_params kparams_local;
    ثابت ساختار htp_sort_kernel_params * kparams = (ثابت ساختار htp_sort_kernel_params *) octx->kernel_params;

    اگر (kparams->n_threads == 0) {
        ساختار htp_sort_vtcm_layout layout;
        اگر (!htp_sort_solve_layout(&layout, ne00, total_rows, k, octx->n_threads, octx->ctx->vtcm_size, is_بالاترین_k)) {
            بازگشت HTP_STATUS_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, اندازه(kparams_local));
        kparams_local.n_threads         = (int32_t) layout.n_threads;
        kparams_local.total_rows        = (int32_t) total_rows;
        kparams_local.row_start         = (int32_t) row_start;
        kparams_local.row_end           = (int32_t) row_end;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.order             = is_بالاترین_k ? GGML_SORT_ORDER_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_بالاترین_k          = is_بالاترین_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_size         = (int32_t) layout.total_bytes;
        kparams_local.phase1_slot_size  = (int32_t) layout.phase1_slot_size;
        kparams_local.merge_مقادیر_off  = (int32_t) layout.merge_مقادیر_off;
        kparams_local.merge_indices_off = (int32_t) layout.merge_indices_off;
        kparams_local.merge_elems       = (int32_t) layout.merge_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } وگرنه اگر (octx->ctx->mdev.شمار > 1) {
        memcpy(&kparams_local, kparams, اندازه(kparams_local));
        kparams_local.row_start = (int32_t) row_start;
        kparams_local.row_end   = (int32_t) row_end;
        kparams = &kparams_local;
    }

    اگر (octx->ctx->vtcm_size < (size_t) kparams->vtcm_size) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_size, octx->ctx->vtcm_size);
        بازگشت HTP_STATUS_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_base = (uint8_t *) octx->ctx->vtcm_base;

    اگر (total_rows == 1 && kparams->n_chunks > 1) {
        ساختار htp_sort_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_base = vtcm_base;
        memset(cctx.real_شمار, 0, اندازه(cctx.real_شمار));

        work_queue_run(octx->ctx->work_queue, htp_sort_chunk_job, &cctx, (uint32_t) kparams->n_chunks);

        اگر (is_بالاترین_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            اگر (k == 1 && ne00 >= 128*1024) {
                اعشاری *   cand_vals = (اعشاری *)   (vtcm_base + kparams->merge_مقادیر_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                اعشاری best_val = cand_vals[0];
                int32_t best_idx = cand_idxs[0];
                برای (int32_t c = 1; c < kparams->n_chunks; c++) {
                    اگر (cand_vals[c] > best_val) {
                        best_val = cand_vals[c];
                        best_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->data;
                dst_ptr[0] = best_idx;
                بازگشت HTP_STATUS_OK;
            }

            uint32_t total_candidates = (uint32_t) kparams->n_chunks * local_k;
            اعشاری *   cand_vals = (اعشاری *)   (vtcm_base + kparams->merge_مقادیر_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_base + kparams->merge_indices_off);
            uint32_t merge_elems = (uint32_t) kparams->merge_elems;

            اگر (merge_elems > total_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + total_candidates), -INFINITY, merge_elems - total_candidates);
                برای (uint32_t j = total_candidates; j < merge_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            ساختار htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_sort_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, merge_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->data, cand_idxs, k * اندازه(int32_t));
        } وگرنه {
            bool asc = kparams->order == GGML_SORT_ORDER_ASC;
            اگر (kparams->n_chunks == 8) {
                ساختار htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 4);

                ساختار htp_argsort_merge_l2_ctx l2ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l2_job, &l2ctx, 2);

                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ثابت اعشاری *   val0 = (ثابت اعشاری *) vtcm_base;
                ثابت int32_t * idx0 = (ثابت int32_t *) (vtcm_base + 4 * chunk_elems * اندازه(اعشاری));
                ثابت اعشاری *   val1 = (ثابت اعشاری *) (vtcm_base + 4 * slot_size);
                ثابت int32_t * idx1 = (ثابت int32_t *) (vtcm_base + 4 * slot_size + 4 * chunk_elems * اندازه(اعشاری));

                uint32_t n01 = cctx.real_شمار[0] + cctx.real_شمار[1] + cctx.real_شمار[2] + cctx.real_شمار[3];
                uint32_t n23 = cctx.real_شمار[4] + cctx.real_شمار[5] + cctx.real_شمار[6] + cctx.real_شمار[7];

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                ساختار htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, n01, val1, idx1, n23, پوچ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * اندازه(int32_t));
            } وگرنه اگر (kparams->n_chunks == 4) {
                ساختار htp_argsort_merge_l1_ctx l1ctx = { &cctx, asc };
                work_queue_run(octx->ctx->work_queue, htp_argsort_merge_l1_job, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_شمار[0] + cctx.real_شمار[1];
                uint32_t n23 = cctx.real_شمار[2] + cctx.real_شمار[3];

                ثابت اعشاری *   val01 = (ثابت اعشاری *)   (vtcm_base + kparams->merge_مقادیر_off);
                ثابت int32_t * idx01 = (ثابت int32_t *) (vtcm_base + kparams->merge_indices_off);
                ثابت اعشاری *   val23 = (ثابت اعشاری *)   (vtcm_base + kparams->merge_مقادیر_off)  + 2 * chunk_elems;
                ثابت int32_t * idx23 = (ثابت int32_t *) (vtcm_base + kparams->merge_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_base;
                ساختار htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val01, idx01, n01, val23, idx23, n23, پوچ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * اندازه(int32_t));
            } وگرنه اگر (kparams->n_chunks == 2) {
                size_t slot_size = (size_t) kparams->phase1_slot_size;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ثابت اعشاری *   val0 = (ثابت اعشاری *)   (vtcm_base);
                ثابت int32_t * idx0 = (ثابت int32_t *) (vtcm_base + chunk_elems * اندازه(اعشاری));
                ثابت اعشاری *   val1 = (ثابت اعشاری *)   (vtcm_base + slot_size);
                ثابت int32_t * idx1 = (ثابت int32_t *) (vtcm_base + slot_size + chunk_elems * اندازه(اعشاری));

                int32_t * final_idx = (int32_t *) (vtcm_base + kparams->merge_indices_off);
                ساختار htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_start(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                merge_runs(val0, idx0, cctx.real_شمار[0], val1, idx1, cctx.real_شمار[1], پوچ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * اندازه(int32_t));
            } وگرنه {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_base + chunk_elems * اندازه(اعشاری));
                dma_sync_write(octx->ctx->dma[0], dst->data, final_idx, ne00 * اندازه(int32_t));
            }
        }
        بازگشت HTP_STATUS_OK;
    }

    ساختار htp_sort_multi_row_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_base        = vtcm_base;

    uint32_t dst_row_bytes = (uint32_t) dst_row_size;
    uint32_t rows_per_chunk = 0;
    htp_tensor_mdev_rows_per_chunk(dst, اندازه(int32_t), dst_row_bytes, &rows_per_chunk);
    اگر (rows_per_chunk == 0) {
        rows_per_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_threads - 1) / (uint32_t) kparams->n_threads;
    اگر (rows_per_chunk > 1) {
        dr = hex_round_up(dr, rows_per_chunk);
    }
    mctx.nrows_per_thread = dr;

    work_queue_run(octx->ctx->work_queue, htp_sort_multi_row_job, &mctx, (uint32_t) kparams->n_threads);

    بازگشت HTP_STATUS_OK;
}

صحیح op_argsort(ساختار htp_ops_context * octx) {
    بازگشت op_sort_common(octx, false);
}

صحیح op_بالاترین_k(ساختار htp_ops_context * octx) {
    بازگشت op_sort_common(octx, true);
}
