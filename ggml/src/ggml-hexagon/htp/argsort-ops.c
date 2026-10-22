#شامل <string.h>
#شامل <stdlib.h>
#شامل <math.h>
#شامل <HAP_farf.h>
#شامل <HAP_perf.h>

#تعریف_کرو GGML_COMMON_DECL_C
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

#اگر_غیر_تعریف_شدہ MIN
#تعریف_کرو MIN(a, b) ((a) < (b) ? (a) : (b))
#اگر_ختم


ساکن خالی quicksort_اقدار_indices_asc(اعشاری * اقدار, int32_t * indices, صحیح بایاں, صحیح دایاں) {
    جب (بایاں < دایاں) {
        اعشاری محور = اقدار[بایاں + (دایاں - بایاں) / 2];
        صحیح i = بایاں;
        صحیح j = دایاں;

        جب (i <= j) {
            جب (اقدار[i] < محور) i++;
            جب (اقدار[j] > محور) j--;

            اگر (i <= j) {
                اعشاری tmp_val = اقدار[i];
                اقدار[i] = اقدار[j];
                اقدار[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        اگر (j - بایاں < دایاں - i) {
            اگر (بایاں < j) quicksort_اقدار_indices_asc(اقدار, indices, بایاں, j);
            بایاں = i;
        } ورنہ {
            اگر (i < دایاں) quicksort_اقدار_indices_asc(اقدار, indices, i, دایاں);
            دایاں = j;
        }
    }
}

ساکن خالی quicksort_اقدار_indices_desc(اعشاری * اقدار, int32_t * indices, صحیح بایاں, صحیح دایاں) {
    جب (بایاں < دایاں) {
        اعشاری محور = اقدار[بایاں + (دایاں - بایاں) / 2];
        صحیح i = بایاں;
        صحیح j = دایاں;

        جب (i <= j) {
            جب (اقدار[i] > محور) i++;
            جب (اقدار[j] < محور) j--;

            اگر (i <= j) {
                اعشاری tmp_val = اقدار[i];
                اقدار[i] = اقدار[j];
                اقدار[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        اگر (j - بایاں < دایاں - i) {
            اگر (بایاں < j) quicksort_اقدار_indices_desc(اقدار, indices, بایاں, j);
            بایاں = i;
        } ورنہ {
            اگر (i < دایاں) quicksort_اقدار_indices_desc(اقدار, indices, i, دایاں);
            دایاں = j;
        }
    }
}

ساکن اندرونی uint32_t چوٹی_k_max_قدر_اشاریہ(ثابت اعشاری * اقدار, uint32_t n, اعشاری * قدر) {
    int32_t max_idx = 0;
    hvx_argmax_f32(اقدار, n, 0, قدر, &max_idx);
    واپس (uint32_t) max_idx;
}

ساکن ثابت int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

ساکن اندرونی خالی init_indices_ramp(int32_t * indices_buf, uint32_t گنتی, uint32_t بنیاد) {
    ثابت HVX_سمتیہ ind_init_vec = Q6_Vw_vadd_VwVw(*(ثابت HVX_سمتیہ *) argsort_ramp_lut, Q6_V_vsplat_R(بنیاد));
    ثابت HVX_سمتیہ ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_سمتیہ * indices_buf_vec = (HVX_سمتیہ *) indices_buf;
    uint32_t عدد_vecs = گنتی / 32;
    HVX_سمتیہ curr_ind_vec = ind_init_vec;
    برائے (uint32_t j = 0; j < عدد_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = گنتی % 32;
    اگر (rem > 0) {
        hvx_vec_store_a((خالی *) &indices_buf_vec[عدد_vecs], rem * حجم(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
ساکن اندرونی خالی vec_cas(HVX_سمتیہ * X_val, HVX_سمتیہ * X_idx, HVX_سمتیہ * Y_val, HVX_سمتیہ * Y_idx, bool asc) {
    HVX_سمتیہ_Pred pred = asc ? Q6_Q_vcmp_gt_VsfVsf(*X_val, *Y_val)
                              : Q6_Q_vcmp_gt_VsfVsf(*Y_val, *X_val);
    HVX_سمتیہ اگلا_X_val = Q6_V_vmux_QVV(pred, *Y_val, *X_val);
    HVX_سمتیہ اگلا_Y_val = Q6_V_vmux_QVV(pred, *X_val, *Y_val);
    HVX_سمتیہ اگلا_X_idx = Q6_V_vmux_QVV(pred, *Y_idx, *X_idx);
    HVX_سمتیہ Y_tmp_idx  = Q6_V_vmux_QVV(pred, *X_idx, *Y_idx);
    *X_val = اگلا_X_val;
    *Y_val = اگلا_Y_val;
    *X_idx = اگلا_X_idx;
    *Y_idx = Y_tmp_idx;
}

__attribute__((always_inline))
ساکن اندرونی خالی bitonic_cas_32(HVX_سمتیہ * V, HVX_سمتیہ * I, صحیح d, HVX_سمتیہ_Pred dir_نقاب, HVX_سمتیہ idx_vec, HVX_سمتیہ zero_vec) {
    HVX_سمتیہ_Pred نقاب_بایاں;
    HVX_سمتیہ V_rot_بایاں, V_rot_دایاں;
    HVX_سمتیہ I_rot_بایاں, I_rot_دایاں;

    اگر (d == 1) {
        نقاب_بایاں = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_بایاں = Q6_V_vror_VR(*V, 4);
        V_rot_دایاں = Q6_V_vror_VR(*V, 124);
        I_rot_بایاں = Q6_V_vror_VR(*I, 4);
        I_rot_دایاں = Q6_V_vror_VR(*I, 124);
    } ورنہ اگر (d == 2) {
        نقاب_بایاں = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_بایاں = Q6_V_vror_VR(*V, 8);
        V_rot_دایاں = Q6_V_vror_VR(*V, 120);
        I_rot_بایاں = Q6_V_vror_VR(*I, 8);
        I_rot_دایاں = Q6_V_vror_VR(*I, 120);
    } ورنہ اگر (d == 4) {
        نقاب_بایاں = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_بایاں = Q6_V_vror_VR(*V, 16);
        V_rot_دایاں = Q6_V_vror_VR(*V, 112);
        I_rot_بایاں = Q6_V_vror_VR(*I, 16);
        I_rot_دایاں = Q6_V_vror_VR(*I, 112);
    } ورنہ اگر (d == 8) {
        نقاب_بایاں = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_بایاں = Q6_V_vror_VR(*V, 32);
        V_rot_دایاں = Q6_V_vror_VR(*V, 96);
        I_rot_بایاں = Q6_V_vror_VR(*I, 32);
        I_rot_دایاں = Q6_V_vror_VR(*I, 96);
    } ورنہ {
        نقاب_بایاں = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_بایاں = Q6_V_vror_VR(*V, 64);
        V_rot_دایاں = Q6_V_vror_VR(*V, 64);
        I_rot_بایاں = Q6_V_vror_VR(*I, 64);
        I_rot_دایاں = Q6_V_vror_VR(*I, 64);
    }

    HVX_سمتیہ V_paired = Q6_V_vmux_QVV(نقاب_بایاں, V_rot_بایاں, V_rot_دایاں);
    HVX_سمتیہ I_paired = Q6_V_vmux_QVV(نقاب_بایاں, I_rot_بایاں, I_rot_دایاں);

    HVX_سمتیہ_Pred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_سمتیہ_Pred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_سمتیہ_Pred نقاب_دایاں = Q6_Q_not_Q(نقاب_بایاں);
    HVX_سمتیہ_Pred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(نقاب_بایاں, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, نقاب_دایاں)
    );
    HVX_سمتیہ_Pred Q_تبادلہ = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_نقاب, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_نقاب), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_تبادلہ, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_تبادلہ, I_paired, *I);
}

__attribute__((always_inline))
ساکن اندرونی خالی bitonic_ترتیب_دو_generic_hvx(uint8_t * اقدار, uint8_t * indices, صحیح K, bool asc_ترتیب) {
    HVX_سمتیہ V[32];
    HVX_سمتیہ I[32];

    HVX_سمتیہ zero_vec = Q6_V_vzero();
    HVX_سمتیہ idx_vec = *(ثابت HVX_سمتیہ *) argsort_ramp_lut;

    برائے (صحیح v = 0; v < K; v++) {
        V[v] = *(HVX_سمتیہ *)(اقدار + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_سمتیہ_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_سمتیہ_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    صحیح M = 5;
    جب ((1 << (M - 5)) < K) M++;

    برائے (صحیح s = 1; s <= M; s++) {
        برائے (صحیح stage_d = s - 1; stage_d >= 0; stage_d--) {
            صحیح d = 1 << stage_d;
            اگر (d >= 32) {
                صحیح v_dist = d / 32;
                برائے (صحیح v1 = 0; v1 < K; v1++) {
                    اگر ((v1 & v_dist) == 0) {
                        صحیح v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_ترتیب;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } ورنہ {
                اگر (s < 5) {
                    HVX_سمتیہ_Pred dir_نقاب = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    برائے (صحیح v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_نقاب, idx_vec, zero_vec);
                    }
                } ورنہ {
                    برائے (صحیح v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_ترتیب;
                        HVX_سمتیہ_Pred dir_نقاب = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_نقاب, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    برائے (صحیح v = 0; v < K; v++) {
        *(HVX_سمتیہ *)(اقدار + v * 128)  = V[v];
        *(HVX_سمتیہ *)(indices + v * 128) = I[v];
    }
}

ساکن خالی bitonic_ترتیب_دو_vtcm_desc(uint8_t * اقدار, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_سمتیہ zero_vec = Q6_V_vzero();
    HVX_سمتیہ idx_vec = *(ثابت HVX_سمتیہ *) argsort_ramp_lut;

    HVX_سمتیہ_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_سمتیہ_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    اگر (init_indices) {
        برائے (uint32_t v = 0; v < n_vec; v++) {
            HVX_سمتیہ idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_سمتیہ *)(indices + v * 128) = idx;
        }
    }

    صحیح M = 5;
    جب ((1u << (M - 5)) < n_vec) M++;

    برائے (صحیح s = 1; s <= M; s++) {
        برائے (صحیح stage_d = s - 1; stage_d >= 0; stage_d--) {
            صحیح d = 1 << stage_d;
            اگر (d >= 32) {
                uint32_t v_dist = d / 32;
                برائے (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    اگر ((v1 & v_dist) == 0) {
                        uint32_t v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : false;

                        HVX_سمتیہ Vv1 = *(HVX_سمتیہ *)(اقدار + v1 * 128);
                        HVX_سمتیہ Iv1 = *(HVX_سمتیہ *)(indices + v1 * 128);
                        HVX_سمتیہ Vv2 = *(HVX_سمتیہ *)(اقدار + v2 * 128);
                        HVX_سمتیہ Iv2 = *(HVX_سمتیہ *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_سمتیہ *)(اقدار + v1 * 128)  = Vv1;
                        *(HVX_سمتیہ *)(indices + v1 * 128) = Iv1;
                        *(HVX_سمتیہ *)(اقدار + v2 * 128)  = Vv2;
                        *(HVX_سمتیہ *)(indices + v2 * 128) = Iv2;
                    }
                }
            } ورنہ {
                اگر (s < 5) {
                    HVX_سمتیہ_Pred dir_نقاب = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    برائے (uint32_t v = 0; v < n_vec; v++) {
                        HVX_سمتیہ Vv = *(HVX_سمتیہ *)(اقدار + v * 128);
                        HVX_سمتیہ Iv = *(HVX_سمتیہ *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_نقاب, idx_vec, zero_vec);

                        *(HVX_سمتیہ *)(اقدار + v * 128)  = Vv;
                        *(HVX_سمتیہ *)(indices + v * 128) = Iv;
                    }
                } ورنہ {
                    برائے (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_سمتیہ_Pred dir_نقاب = asc ? pred_all_1s : pred_all_0s;

                        HVX_سمتیہ Vv = *(HVX_سمتیہ *)(اقدار + v * 128);
                        HVX_سمتیہ Iv = *(HVX_سمتیہ *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_نقاب, idx_vec, zero_vec);

                        *(HVX_سمتیہ *)(اقدار + v * 128)  = Vv;
                        *(HVX_سمتیہ *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

ساکن خالی چوٹی_k_select_tiled(dma_قطار * dma_q, ساخت htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               اعشاری * اقدار_buf, int32_t * indices_buf,
                               اعشاری * برآمد_اقدار, int32_t * برآمد_indices,
                               uint32_t chunk_بنیاد) {
    ثابت uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_گنتی    = n_tiles * k;
    uint32_t ملاؤ_n_vec        = hmx_ceil_div(candidate_گنتی, 32);
    uint32_t ملاؤ_n_vec_pow2   = 1;
    جب (ملاؤ_n_vec_pow2 < ملاؤ_n_vec) ملاؤ_n_vec_pow2 <<= 1;
    uint32_t ملاؤ_elems        = ملاؤ_n_vec_pow2 * 32;
    اعشاری * candidate_اقدار    = اقدار_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    برائے (uint32_t انحراف = 0; انحراف < n; انحراف += tile_elems) {
        uint32_t tile_گنتی = MIN(tile_elems, n - انحراف);
        dma_sync_read(dma_q, اقدار_buf, src_addr + (size_t) انحراف * حجم(اعشاری), tile_گنتی * حجم(اعشاری));

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) انحراف);
        اگر (tile_گنتی < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (اقدار_buf + tile_گنتی), -INFINITY, tile_elems - tile_گنتی);
        }

        bitonic_ترتیب_دو_vtcm_desc((uint8_t *) اقدار_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_گنتی);
        برائے (uint32_t j = 0; j < tile_k; j++) {
            candidate_اقدار[candidate_pos] = اقدار_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) انحراف;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) انحراف);
    }

    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    اگر (ملاؤ_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_اقدار + candidate_pos), -INFINITY, ملاؤ_elems - candidate_pos);
        برائے (uint32_t j = candidate_pos; j < ملاؤ_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_ترتیب_دو_vtcm_desc((uint8_t *) candidate_اقدار, (uint8_t *) candidate_indices, ملاؤ_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    برائے (uint32_t j = 0; j < k; j++) {
        برآمد_اقدار[j]  = candidate_اقدار[j];
        برآمد_indices[j] = candidate_indices[j] + (int32_t) chunk_بنیاد;
    }
}

__attribute__((always_inline))
ساکن اندرونی خالی sort32_f32_hvx(uint8_t * اقدار, uint8_t * indices, شمار ggml_ترتیب_دو_ترتیب ترتیب) {
    bitonic_ترتیب_دو_generic_hvx(اقدار, indices, 1, ترتیب == GGML_ترتیب_دو_ترتیب_ASC);
}

__attribute__((always_inline))
ساکن اندرونی خالی sort64_f32_hvx(uint8_t * اقدار, uint8_t * indices, شمار ggml_ترتیب_دو_ترتیب ترتیب) {
    bitonic_ترتیب_دو_generic_hvx(اقدار, indices, 2, ترتیب == GGML_ترتیب_دو_ترتیب_ASC);
}

__attribute__((always_inline))
ساکن اندرونی خالی sort128_f32_hvx(uint8_t * اقدار, uint8_t * indices, شمار ggml_ترتیب_دو_ترتیب ترتیب) {
    bitonic_ترتیب_دو_generic_hvx(اقدار, indices, 4, ترتیب == GGML_ترتیب_دو_ترتیب_ASC);
}

__attribute__((always_inline))
ساکن اندرونی خالی sort256_f32_hvx(uint8_t * اقدار, uint8_t * indices, شمار ggml_ترتیب_دو_ترتیب ترتیب) {
    bitonic_ترتیب_دو_generic_hvx(اقدار, indices, 8, ترتیب == GGML_ترتیب_دو_ترتیب_ASC);
}

__attribute__((always_inline))
ساکن اندرونی خالی sort512_f32_hvx(uint8_t * اقدار, uint8_t * indices, شمار ggml_ترتیب_دو_ترتیب ترتیب) {
    bitonic_ترتیب_دو_generic_hvx(اقدار, indices, 16, ترتیب == GGML_ترتیب_دو_ترتیب_ASC);
}

__attribute__((always_inline))
ساکن اندرونی خالی sort1024_f32_hvx(uint8_t * اقدار, uint8_t * indices, شمار ggml_ترتیب_دو_ترتیب ترتیب) {
    bitonic_ترتیب_دو_generic_hvx(اقدار, indices, 32, ترتیب == GGML_ترتیب_دو_ترتیب_ASC);
}

ساکن خالی ملاؤ_runs(
    ثابت اعشاری * محدود اندر_val0, ثابت int32_t * محدود اندر_idx0, uint32_t n0,
    ثابت اعشاری * محدود اندر_val1, ثابت int32_t * محدود اندر_idx1, uint32_t n1,
    اعشاری * محدود برآمد_val, int32_t * محدود برآمد_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t برآمد = 0;

    اگر (asc) {
        جب (i0 < n0 && i1 < n1) {
            اگر (اندر_val0[i0] <= اندر_val1[i1]) {
                اگر (برآمد_val) برآمد_val[برآمد] = اندر_val0[i0];
                برآمد_idx[برآمد] = اندر_idx0[i0];
                i0++;
            } ورنہ {
                اگر (برآمد_val) برآمد_val[برآمد] = اندر_val1[i1];
                برآمد_idx[برآمد] = اندر_idx1[i1];
                i1++;
            }
            برآمد++;
        }
    } ورنہ {
        جب (i0 < n0 && i1 < n1) {
            اگر (اندر_val0[i0] >= اندر_val1[i1]) {
                اگر (برآمد_val) برآمد_val[برآمد] = اندر_val0[i0];
                برآمد_idx[برآمد] = اندر_idx0[i0];
                i0++;
            } ورنہ {
                اگر (برآمد_val) برآمد_val[برآمد] = اندر_val1[i1];
                برآمد_idx[برآمد] = اندر_idx1[i1];
                i1++;
            }
            برآمد++;
        }
    }

    جب (i0 < n0) {
        اگر (برآمد_val) برآمد_val[برآمد] = اندر_val0[i0];
        برآمد_idx[برآمد] = اندر_idx0[i0];
        i0++;
        برآمد++;
    }

    جب (i1 < n1) {
        اگر (برآمد_val) برآمد_val[برآمد] = اندر_val1[i1];
        برآمد_idx[برآمد] = اندر_idx1[i1];
        i1++;
        برآمد++;
    }
}

ساخت htp_ترتیب_دو_chunk_ctx {
    ساخت htp_ops_context *              octx;
    ثابت ساخت htp_ترتیب_دو_kernel_params * kparams;
    uint8_t *                             vtcm_بنیاد;
    uint32_t                              real_گنتی[HTP_MAX_NTHREADS];
};

ساکن خالی htp_ترتیب_دو_chunk_کام(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    ساخت htp_ترتیب_دو_chunk_ctx * cctx = (ساخت htp_ترتیب_دو_chunk_ctx *) ڈیٹا;
    ساخت htp_ops_context * octx = cctx->octx;
    ثابت ساخت htp_ترتیب_دو_kernel_params * kparams = cctx->kparams;
    ثابت ساخت htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_بنیاد  = i * chunk_elems;

    uint32_t real_گنتی = (chunk_بنیاد < ne00) ? MIN(chunk_elems, ne00 - chunk_بنیاد) : 0;
    cctx->real_گنتی[i] = real_گنتی;

    uint8_t * spad = cctx->vtcm_بنیاد + (size_t) kparams->phase1_خانہ_سائز * i;
    اعشاری *   اقدار_buf  = (اعشاری *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * حجم(اعشاری));

    اعشاری *   ملاؤ_اقدار  = (اعشاری *)   (cctx->vtcm_بنیاد + kparams->ملاؤ_اقدار_off);
    int32_t * ملاؤ_indices = (int32_t *) (cctx->vtcm_بنیاد + kparams->ملاؤ_indices_off);

    ساخت htp_thread_trace * tr = &octx->ctx->trace[i];
    ثابت dma_addr_t src_addr = src0->ڈیٹا + (size_t) chunk_بنیاد * حجم(اعشاری);

    اگر (kparams->is_چوٹی_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        اگر (real_گنتی == 0) {
            برائے (uint32_t j = 0; j < local_k; j++) {
                ملاؤ_اقدار[i * local_k + j]  = -INFINITY;
                ملاؤ_indices[i * local_k + j] = 0;
            }
            واپس;
        }

        اگر (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], اقدار_buf, src_addr, real_گنتی * حجم(اعشاری));

            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            اعشاری max_قدر;
            uint32_t max_اشاریہ = چوٹی_k_max_قدر_اشاریہ(اقدار_buf, real_گنتی, &max_قدر);
            ملاؤ_اقدار[i] = max_قدر;
            ملاؤ_indices[i] = (int32_t) (max_اشاریہ + chunk_بنیاد);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            واپس;
        }

        اگر (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            چوٹی_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_گنتی, local_k,
                               اقدار_buf, indices_buf,
                               &ملاؤ_اقدار[i * local_k], &ملاؤ_indices[i * local_k],
                               chunk_بنیاد);
            واپس;
        }

        dma_sync_read(octx->ctx->dma[i], اقدار_buf, src_addr, real_گنتی * حجم(اعشاری));

        htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        اگر (chunk_elems > real_گنتی) {
            hvx_splat_f32_u((uint8_t *)(اقدار_buf + real_گنتی), -INFINITY, chunk_elems - real_گنتی);
        }

        bitonic_ترتیب_دو_vtcm_desc((uint8_t *) اقدار_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        برائے (uint32_t j = 0; j < local_k; j++) {
            ملاؤ_اقدار[i * local_k + j]  = اقدار_buf[j];
            ملاؤ_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_بنیاد;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } ورنہ {
        اگر (real_گنتی > 0) {
            dma_sync_read(octx->ctx->dma[i], اقدار_buf, src_addr, real_گنتی * حجم(اعشاری));

            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_گنتی, chunk_بنیاد);

            اگر (kparams->ترتیب == GGML_ترتیب_دو_ترتیب_ASC) {
                quicksort_اقدار_indices_asc(اقدار_buf, indices_buf, 0, real_گنتی - 1);
            } ورنہ {
                quicksort_اقدار_indices_desc(اقدار_buf, indices_buf, 0, real_گنتی - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

ساخت htp_argsort_ملاؤ_l1_ctx {
    ساخت htp_ترتیب_دو_chunk_ctx * cctx;
    bool                        asc;
};

ساکن خالی htp_argsort_ملاؤ_l1_کام(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    ساخت htp_argsort_ملاؤ_l1_ctx * mctx = (ساخت htp_argsort_ملاؤ_l1_ctx *) ڈیٹا;
    ساخت htp_ترتیب_دو_chunk_ctx * cctx = mctx->cctx;
    ثابت ساخت htp_ترتیب_دو_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_گنتی[c0];
    uint32_t count1 = cctx->real_گنتی[c1];

    size_t خانہ_سائز = (size_t) kparams->phase1_خانہ_سائز;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    ثابت اعشاری *   val0 = (ثابت اعشاری *)   (cctx->vtcm_بنیاد + خانہ_سائز * c0);
    ثابت int32_t * idx0 = (ثابت int32_t *) (cctx->vtcm_بنیاد + خانہ_سائز * c0 + chunk_elems * حجم(اعشاری));
    ثابت اعشاری *   val1 = (ثابت اعشاری *)   (cctx->vtcm_بنیاد + خانہ_سائز * c1);
    ثابت int32_t * idx1 = (ثابت int32_t *) (cctx->vtcm_بنیاد + خانہ_سائز * c1 + chunk_elems * حجم(اعشاری));

    uint32_t برآمد_انحراف = (2 * i) * chunk_elems;
    اعشاری *   برآمد_val = (اعشاری *)   (cctx->vtcm_بنیاد + kparams->ملاؤ_اقدار_off)  + برآمد_انحراف;
    int32_t * برآمد_idx = (int32_t *) (cctx->vtcm_بنیاد + kparams->ملاؤ_indices_off) + برآمد_انحراف;

    ساخت htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    ملاؤ_runs(val0, idx0, count0, val1, idx1, count1, برآمد_val, برآمد_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

ساخت htp_argsort_ملاؤ_l2_ctx {
    ساخت htp_ترتیب_دو_chunk_ctx * cctx;
    bool                        asc;
};

ساکن خالی htp_argsort_ملاؤ_l2_کام(غیرعلامتی صحیح n, غیرعلامتی صحیح i, خالی * ڈیٹا) {
    ساخت htp_argsort_ملاؤ_l2_ctx * mctx = (ساخت htp_argsort_ملاؤ_l2_ctx *) ڈیٹا;
    ساخت htp_ترتیب_دو_chunk_ctx * cctx = mctx->cctx;
    ثابت ساخت htp_ترتیب_دو_kernel_params * kparams = cctx->kparams;

    size_t خانہ_سائز = (size_t) kparams->phase1_خانہ_سائز;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    ثابت اعشاری *   val0 = (ثابت اعشاری *)   (cctx->vtcm_بنیاد + kparams->ملاؤ_اقدار_off)  + off0;
    ثابت int32_t * idx0 = (ثابت int32_t *) (cctx->vtcm_بنیاد + kparams->ملاؤ_indices_off) + off0;
    ثابت اعشاری *   val1 = (ثابت اعشاری *)   (cctx->vtcm_بنیاد + kparams->ملاؤ_اقدار_off)  + off1;
    ثابت int32_t * idx1 = (ثابت int32_t *) (cctx->vtcm_بنیاد + kparams->ملاؤ_indices_off) + off1;

    uint32_t count0 = cctx->real_گنتی[4*i + 0] + cctx->real_گنتی[4*i + 1];
    uint32_t count1 = cctx->real_گنتی[4*i + 2] + cctx->real_گنتی[4*i + 3];

    uint8_t * خانہ_بنیاد = cctx->vtcm_بنیاد + (4 * i) * خانہ_سائز;
    اعشاری *   برآمد_val   = (اعشاری *) خانہ_بنیاد;
    int32_t * برآمد_idx   = (int32_t *) (خانہ_بنیاد + 4 * chunk_elems * حجم(اعشاری));

    ساخت htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    ملاؤ_runs(val0, idx0, count0, val1, idx1, count1, برآمد_val, برآمد_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

ساخت htp_ترتیب_دو_multi_صف_ctx {
    ساخت htp_ops_context *              octx;
    ثابت ساخت htp_ترتیب_دو_kernel_params * kparams;
    uint8_t *                             vtcm_بنیاد;
    uint32_t                              nrows_فی_thread;
};

ساخت htp_tensor_صف_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ساکن اندرونی ساخت htp_tensor_صف_coord init_صف_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    واپس (ساخت htp_tensor_صف_coord){ i01, i02, i03 };
}

ساکن اندرونی خالی قدم_صف_coord(ساخت htp_tensor_صف_coord * c, uint32_t ne01, uint32_t ne02) {
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

ساکن اندرونی size_t coord_انحراف(ساخت htp_tensor_صف_coord c, size_t nb1, size_t nb2, size_t nb3) {
    واپس (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

ساکن اندرونی خالی compute_صف_ترتیب_دو(
    اعشاری * cur_اقدار,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_چوٹی_k,
    شمار ggml_ترتیب_دو_ترتیب ترتیب
) {
    اگر (is_چوٹی_k) {
        اگر (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            انتخاب (ne00) {
                حالت 32:   sort32_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, GGML_ترتیب_دو_ترتیب_DESC); توڑو;
                حالت 64:   sort64_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, GGML_ترتیب_دو_ترتیب_DESC); توڑو;
                حالت 128:  sort128_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, GGML_ترتیب_دو_ترتیب_DESC); توڑو;
                حالت 256:  sort256_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, GGML_ترتیب_دو_ترتیب_DESC); توڑو;
                حالت 512:  sort512_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, GGML_ترتیب_دو_ترتیب_DESC); توڑو;
                حالت 1024: sort1024_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, GGML_ترتیب_دو_ترتیب_DESC); توڑو;
            }
        } ورنہ {
            اگر (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_اقدار + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_ترتیب_دو_vtcm_desc((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } ورنہ {
        اگر (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            انتخاب (ne00) {
                حالت 32:   sort32_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, ترتیب); توڑو;
                حالت 64:   sort64_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, ترتیب); توڑو;
                حالت 128:  sort128_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, ترتیب); توڑو;
                حالت 256:  sort256_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, ترتیب); توڑو;
                حالت 512:  sort512_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, ترتیب); توڑو;
                حالت 1024: sort1024_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, ترتیب); توڑو;
            }
        } ورنہ {
            init_indices_ramp(cur_indices, ne00, 0);
            اگر (ترتیب == GGML_ترتیب_دو_ترتیب_ASC) {
                quicksort_اقدار_indices_asc(cur_اقدار, cur_indices, 0, ne00 - 1);
            } ورنہ {
                quicksort_اقدار_indices_desc(cur_اقدار, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

ساکن خالی htp_ترتیب_دو_multi_صف_کام(غیرعلامتی صحیح n, غیرعلامتی صحیح ith, خالی * ڈیٹا) {
    ساخت htp_ترتیب_دو_multi_صف_ctx * mctx = (ساخت htp_ترتیب_دو_multi_صف_ctx *) ڈیٹا;
    ساخت htp_ops_context * octx = mctx->octx;
    ثابت ساخت htp_ترتیب_دو_kernel_params * kparams = mctx->kparams;
    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    uint32_t ne00 = (uint32_t) kparams->ne00;
    uint32_t ne01 = src0->ne[1];
    uint32_t ne02 = src0->ne[2];

    uint32_t nb01 = src0->nb[1];
    uint32_t nb02 = src0->nb[2];
    uint32_t nb03 = src0->nb[3];

    uint32_t nb1 = dst->nb[1];
    uint32_t nb2 = dst->nb[2];
    uint32_t nb3 = dst->nb[3];

    uint32_t صف_آغاز = (uint32_t) kparams->صف_آغاز;
    uint32_t صف_انتہا   = (uint32_t) kparams->صف_انتہا;
    uint32_t r_آغاز   = صف_آغاز + ith * mctx->nrows_فی_thread;
    uint32_t r_انتہا     = MIN(r_آغاز + mctx->nrows_فی_thread, صف_انتہا);

    اگر (r_آغاز >= r_انتہا) واپس;

    size_t spad_انحراف = ith * (size_t) (kparams->phase1_خانہ_سائز * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_بنیاد + spad_انحراف;
    size_t خانہ_سائز = (size_t) kparams->phase1_خانہ_سائز;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    اعشاری *   اقدار_buf  = (اعشاری *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * حجم(اعشاری));

    dma_قطار * dma_q = octx->ctx->dma[ith];
    ساخت htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_چوٹی_k = kparams->is_چوٹی_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    شمار ggml_ترتیب_دو_ترتیب ترتیب = (شمار ggml_ترتیب_دو_ترتیب) kparams->ترتیب;

    ثابت uint32_t src_بائٹیں = ne00 * حجم(اعشاری);
    ثابت uint32_t dst_بائٹیں = (is_چوٹی_k ? k : ne00) * حجم(int32_t);

    اگر (kparams->n_slots == 2) {
        ساخت htp_tensor_صف_coord مفرد_coord = init_صف_coord(r_آغاز, ne01, ne02);
        برائے (uint32_t r = r_آغاز, spad_idx = 0; r < r_انتہا && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * خانہ_سائز;
            اعشاری *   cur_اقدار  = (اعشاری *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * حجم(اعشاری));

            dma_addr_t dst_addr = dst->ڈیٹا + coord_انحراف(مفرد_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->ڈیٹا + coord_انحراف(مفرد_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_addr, cur_indices),
                           dst_بائٹیں, dst_بائٹیں, dst_بائٹیں, 0);

            // Prefetch input row
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(cur_اقدار, src_addr),
                           src_بائٹیں, src_بائٹیں, src_بائٹیں, 1);

            قدم_صف_coord(&مفرد_coord, ne01, ne02);
        }

        ساخت htp_tensor_صف_coord dst_coord = init_صف_coord(r_آغاز, ne01, ne02);
        ساخت htp_tensor_صف_coord اگلا_src_coord = init_صف_coord(r_آغاز + 2, ne01, ne02);

        برائے (uint32_t r = r_آغاز; r < r_انتہا; r++) {
            uint32_t cur_خانہ = (r - r_آغاز) & 1;
            uint8_t * cur_spad = thread_spad + cur_خانہ * خانہ_سائز;
            اعشاری *   cur_اقدار  = (اعشاری *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * حجم(اعشاری));

            // Wait for previous writeback of this slot to complete
            dma_قطار_نکالو(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_قطار_نکالو(dma_q);

            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_صف_ترتیب_دو(cur_اقدار, cur_indices, ne00, chunk_elems, is_چوٹی_k, ترتیب);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->ڈیٹا + coord_انحراف(dst_coord, nb1, nb2, nb3);
            dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(dst_addr, cur_indices),
                           dst_بائٹیں, dst_بائٹیں, dst_بائٹیں, 1);
            قدم_صف_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            ثابت uint32_t اگلا_صف = r + 2;
            اگر (اگلا_صف < r_انتہا) {
                dma_addr_t اگلا_src_addr = src0->ڈیٹا + coord_انحراف(اگلا_src_coord, nb01, nb02, nb03);
                dma_قطار_ڈالو(dma_q, dma_make_ڈیٹا(cur_اقدار, اگلا_src_addr),
                               src_بائٹیں, src_بائٹیں, src_بائٹیں, 1);
                قدم_صف_coord(&اگلا_src_coord, ne01, ne02);
            }
        }

        dma_قطار_flush(dma_q);
    } ورنہ {
        ساخت htp_tensor_صف_coord coord = init_صف_coord(r_آغاز, ne01, ne02);
        برائے (uint32_t r = r_آغاز; r < r_انتہا; r++) {
            dma_addr_t src_addr = src0->ڈیٹا + coord_انحراف(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->ڈیٹا + coord_انحراف(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, اقدار_buf, src_addr, src_بائٹیں);

            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_صف_ترتیب_دو(اقدار_buf, indices_buf, ne00, chunk_elems, is_چوٹی_k, ترتیب);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_بائٹیں);
            قدم_صف_coord(&coord, ne01, ne02);
        }
    }
}

ساکن صحیح op_ترتیب_دو_common(ساخت htp_ops_context * octx, bool is_چوٹی_k) {
    اگر (octx->src[0]->type != HTP_TYPE_F32) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت ساخت htp_tensor * src0 = octx->src[0];
    ثابت ساخت htp_tensor * dst  = octx->dst;

    اگر (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        واپس HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t کل_صفیں = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_صف_سائز = dst->ne[0] * حجم(int32_t);

    uint32_t صف_آغاز = 0;
    uint32_t صف_انتہا   = کل_صفیں;
    اگر (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفیں_فی_chunk = 0;
        htp_tensor_mdev_صفیں_فی_chunk(dst, حجم(int32_t), (uint32_t) dst_صف_سائز, &صفیں_فی_chunk);
        ثابت ساخت htp_tensor_mdev_range range = htp_tensor_mdev_تقسیم(
            کل_صفیں, صفیں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_آغاز = range.آغاز;
        صف_انتہا   = range.آغاز + range.گنتی;
    }

    ثابت uint32_t nrows = صف_انتہا - صف_آغاز;
    اگر (nrows == 0) {
        واپس HTP_حالت_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    ساخت htp_ترتیب_دو_kernel_params kparams_local;
    ثابت ساخت htp_ترتیب_دو_kernel_params * kparams = (ثابت ساخت htp_ترتیب_دو_kernel_params *) octx->kernel_params;

    اگر (kparams->n_دھاگے == 0) {
        ساخت htp_ترتیب_دو_vtcm_layout layout;
        اگر (!htp_ترتیب_دو_solve_layout(&layout, ne00, کل_صفیں, k, octx->n_دھاگے, octx->ctx->vtcm_سائز, is_چوٹی_k)) {
            واپس HTP_حالت_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, حجم(kparams_local));
        kparams_local.n_دھاگے         = (int32_t) layout.n_دھاگے;
        kparams_local.کل_صفیں        = (int32_t) کل_صفیں;
        kparams_local.صف_آغاز         = (int32_t) صف_آغاز;
        kparams_local.صف_انتہا           = (int32_t) صف_انتہا;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.ترتیب             = is_چوٹی_k ? GGML_ترتیب_دو_ترتیب_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_چوٹی_k          = is_چوٹی_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_سائز         = (int32_t) layout.کل_بائٹیں;
        kparams_local.phase1_خانہ_سائز  = (int32_t) layout.phase1_خانہ_سائز;
        kparams_local.ملاؤ_اقدار_off  = (int32_t) layout.ملاؤ_اقدار_off;
        kparams_local.ملاؤ_indices_off = (int32_t) layout.ملاؤ_indices_off;
        kparams_local.ملاؤ_elems       = (int32_t) layout.ملاؤ_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } ورنہ اگر (octx->ctx->mdev.گنتی > 1) {
        حافظہ_نقل(&kparams_local, kparams, حجم(kparams_local));
        kparams_local.صف_آغاز = (int32_t) صف_آغاز;
        kparams_local.صف_انتہا   = (int32_t) صف_انتہا;
        kparams = &kparams_local;
    }

    اگر (octx->ctx->vtcm_سائز < (size_t) kparams->vtcm_سائز) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_سائز, octx->ctx->vtcm_سائز);
        واپس HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_بنیاد = (uint8_t *) octx->ctx->vtcm_بنیاد;

    اگر (کل_صفیں == 1 && kparams->n_chunks > 1) {
        ساخت htp_ترتیب_دو_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_بنیاد = vtcm_بنیاد;
        memset(cctx.real_گنتی, 0, حجم(cctx.real_گنتی));

        work_قطار_تسلسل(octx->ctx->work_قطار, htp_ترتیب_دو_chunk_کام, &cctx, (uint32_t) kparams->n_chunks);

        اگر (is_چوٹی_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            اگر (k == 1 && ne00 >= 128*1024) {
                اعشاری *   cand_vals = (اعشاری *)   (vtcm_بنیاد + kparams->ملاؤ_اقدار_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_بنیاد + kparams->ملاؤ_indices_off);
                اعشاری بہترین_val = cand_vals[0];
                int32_t بہترین_idx = cand_idxs[0];
                برائے (int32_t c = 1; c < kparams->n_chunks; c++) {
                    اگر (cand_vals[c] > بہترین_val) {
                        بہترین_val = cand_vals[c];
                        بہترین_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->ڈیٹا;
                dst_ptr[0] = بہترین_idx;
                واپس HTP_حالت_OK;
            }

            uint32_t کل_candidates = (uint32_t) kparams->n_chunks * local_k;
            اعشاری *   cand_vals = (اعشاری *)   (vtcm_بنیاد + kparams->ملاؤ_اقدار_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_بنیاد + kparams->ملاؤ_indices_off);
            uint32_t ملاؤ_elems = (uint32_t) kparams->ملاؤ_elems;

            اگر (ملاؤ_elems > کل_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + کل_candidates), -INFINITY, ملاؤ_elems - کل_candidates);
                برائے (uint32_t j = کل_candidates; j < ملاؤ_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            ساخت htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_ترتیب_دو_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, ملاؤ_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->ڈیٹا, cand_idxs, k * حجم(int32_t));
        } ورنہ {
            bool asc = kparams->ترتیب == GGML_ترتیب_دو_ترتیب_ASC;
            اگر (kparams->n_chunks == 8) {
                ساخت htp_argsort_ملاؤ_l1_ctx l1ctx = { &cctx, asc };
                work_قطار_تسلسل(octx->ctx->work_قطار, htp_argsort_ملاؤ_l1_کام, &l1ctx, 4);

                ساخت htp_argsort_ملاؤ_l2_ctx l2ctx = { &cctx, asc };
                work_قطار_تسلسل(octx->ctx->work_قطار, htp_argsort_ملاؤ_l2_کام, &l2ctx, 2);

                size_t خانہ_سائز = (size_t) kparams->phase1_خانہ_سائز;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ثابت اعشاری *   val0 = (ثابت اعشاری *) vtcm_بنیاد;
                ثابت int32_t * idx0 = (ثابت int32_t *) (vtcm_بنیاد + 4 * chunk_elems * حجم(اعشاری));
                ثابت اعشاری *   val1 = (ثابت اعشاری *) (vtcm_بنیاد + 4 * خانہ_سائز);
                ثابت int32_t * idx1 = (ثابت int32_t *) (vtcm_بنیاد + 4 * خانہ_سائز + 4 * chunk_elems * حجم(اعشاری));

                uint32_t n01 = cctx.real_گنتی[0] + cctx.real_گنتی[1] + cctx.real_گنتی[2] + cctx.real_گنتی[3];
                uint32_t n23 = cctx.real_گنتی[4] + cctx.real_گنتی[5] + cctx.real_گنتی[6] + cctx.real_گنتی[7];

                int32_t * final_idx = (int32_t *) (vtcm_بنیاد + kparams->ملاؤ_indices_off);
                ساخت htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ملاؤ_runs(val0, idx0, n01, val1, idx1, n23, تہی, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->ڈیٹا, final_idx, ne00 * حجم(int32_t));
            } ورنہ اگر (kparams->n_chunks == 4) {
                ساخت htp_argsort_ملاؤ_l1_ctx l1ctx = { &cctx, asc };
                work_قطار_تسلسل(octx->ctx->work_قطار, htp_argsort_ملاؤ_l1_کام, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_گنتی[0] + cctx.real_گنتی[1];
                uint32_t n23 = cctx.real_گنتی[2] + cctx.real_گنتی[3];

                ثابت اعشاری *   val01 = (ثابت اعشاری *)   (vtcm_بنیاد + kparams->ملاؤ_اقدار_off);
                ثابت int32_t * idx01 = (ثابت int32_t *) (vtcm_بنیاد + kparams->ملاؤ_indices_off);
                ثابت اعشاری *   val23 = (ثابت اعشاری *)   (vtcm_بنیاد + kparams->ملاؤ_اقدار_off)  + 2 * chunk_elems;
                ثابت int32_t * idx23 = (ثابت int32_t *) (vtcm_بنیاد + kparams->ملاؤ_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_بنیاد;
                ساخت htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ملاؤ_runs(val01, idx01, n01, val23, idx23, n23, تہی, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->ڈیٹا, final_idx, ne00 * حجم(int32_t));
            } ورنہ اگر (kparams->n_chunks == 2) {
                size_t خانہ_سائز = (size_t) kparams->phase1_خانہ_سائز;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ثابت اعشاری *   val0 = (ثابت اعشاری *)   (vtcm_بنیاد);
                ثابت int32_t * idx0 = (ثابت int32_t *) (vtcm_بنیاد + chunk_elems * حجم(اعشاری));
                ثابت اعشاری *   val1 = (ثابت اعشاری *)   (vtcm_بنیاد + خانہ_سائز);
                ثابت int32_t * idx1 = (ثابت int32_t *) (vtcm_بنیاد + خانہ_سائز + chunk_elems * حجم(اعشاری));

                int32_t * final_idx = (int32_t *) (vtcm_بنیاد + kparams->ملاؤ_indices_off);
                ساخت htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_آغاز(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ملاؤ_runs(val0, idx0, cctx.real_گنتی[0], val1, idx1, cctx.real_گنتی[1], تہی, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->ڈیٹا, final_idx, ne00 * حجم(int32_t));
            } ورنہ {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_بنیاد + chunk_elems * حجم(اعشاری));
                dma_sync_write(octx->ctx->dma[0], dst->ڈیٹا, final_idx, ne00 * حجم(int32_t));
            }
        }
        واپس HTP_حالت_OK;
    }

    ساخت htp_ترتیب_دو_multi_صف_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_بنیاد        = vtcm_بنیاد;

    uint32_t dst_صف_بائٹیں = (uint32_t) dst_صف_سائز;
    uint32_t صفیں_فی_chunk = 0;
    htp_tensor_mdev_صفیں_فی_chunk(dst, حجم(int32_t), dst_صف_بائٹیں, &صفیں_فی_chunk);
    اگر (صفیں_فی_chunk == 0) {
        صفیں_فی_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_دھاگے - 1) / (uint32_t) kparams->n_دھاگے;
    اگر (صفیں_فی_chunk > 1) {
        dr = hex_باری_اوپر(dr, صفیں_فی_chunk);
    }
    mctx.nrows_فی_thread = dr;

    work_قطار_تسلسل(octx->ctx->work_قطار, htp_ترتیب_دو_multi_صف_کام, &mctx, (uint32_t) kparams->n_دھاگے);

    واپس HTP_حالت_OK;
}

صحیح op_argsort(ساخت htp_ops_context * octx) {
    واپس op_ترتیب_دو_common(octx, false);
}

صحیح op_چوٹی_k(ساخت htp_ops_context * octx) {
    واپس op_ترتیب_دو_common(octx, true);
}
