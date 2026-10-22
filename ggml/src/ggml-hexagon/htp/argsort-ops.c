#ضمن <string.h>
#ضمن <stdlib.h>
#ضمن <math.h>
#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#تعریف_کرو GGML_COMMON_DECL_C
#ضمن "ggml-common.h"
#ضمن "ggml.h"

#ضمن "hvx-utils.h"
#ضمن "hvx-copy.h"
#ضمن "hvx-reduce.h"
#ضمن "dma-queue.h"

#ضمن "hex-common.h"
#ضمن "htp-ctx.h"
#ضمن "htp-ops.h"
#ضمن "htp-tensor.h"
#ضمن "argsort-ops.h"

#جے_بنا_تعریف MIN
#تعریف_کرو MIN(a, b) ((a) < (b) ? (a) : (b))
#جے_انت


ساكن فراغ quicksort_اقدار_indices_asc(عائم * اقدار, int32_t * indices, صحيح بایاں, صحيح دایاں) {
    طالما (بایاں < دایاں) {
        عائم دھرا = اقدار[بایاں + (دایاں - بایاں) / 2];
        صحيح i = بایاں;
        صحيح j = دایاں;

        طالما (i <= j) {
            طالما (اقدار[i] < دھرا) i++;
            طالما (اقدار[j] > دھرا) j--;

            إذا (i <= j) {
                عائم tmp_val = اقدار[i];
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
        إذا (j - بایاں < دایاں - i) {
            إذا (بایاں < j) quicksort_اقدار_indices_asc(اقدار, indices, بایاں, j);
            بایاں = i;
        } وإلا {
            إذا (i < دایاں) quicksort_اقدار_indices_asc(اقدار, indices, i, دایاں);
            دایاں = j;
        }
    }
}

ساكن فراغ quicksort_اقدار_indices_desc(عائم * اقدار, int32_t * indices, صحيح بایاں, صحيح دایاں) {
    طالما (بایاں < دایاں) {
        عائم دھرا = اقدار[بایاں + (دایاں - بایاں) / 2];
        صحيح i = بایاں;
        صحيح j = دایاں;

        طالما (i <= j) {
            طالما (اقدار[i] > دھرا) i++;
            طالما (اقدار[j] < دھرا) j--;

            إذا (i <= j) {
                عائم tmp_val = اقدار[i];
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
        إذا (j - بایاں < دایاں - i) {
            إذا (بایاں < j) quicksort_اقدار_indices_desc(اقدار, indices, بایاں, j);
            بایاں = i;
        } وإلا {
            إذا (i < دایاں) quicksort_اقدار_indices_desc(اقدار, indices, i, دایاں);
            دایاں = j;
        }
    }
}

ساكن مضمن uint32_t چوٹی_k_max_قدر_اشاریہ(ثابت عائم * اقدار, uint32_t n, عائم * قدر) {
    int32_t max_idx = 0;
    hvx_argmax_f32(اقدار, n, 0, قدر, &max_idx);
    أرجع (uint32_t) max_idx;
}

ساكن ثابت int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

ساكن مضمن فراغ init_indices_ramp(int32_t * indices_buf, uint32_t گنتی, uint32_t ادھار) {
    ثابت HVX_سمتیہ ind_init_vec = Q6_Vw_vadd_VwVw(*(ثابت HVX_سمتیہ *) argsort_ramp_lut, Q6_V_vsplat_R(ادھار));
    ثابت HVX_سمتیہ ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_سمتیہ * indices_buf_vec = (HVX_سمتیہ *) indices_buf;
    uint32_t عدد_vecs = گنتی / 32;
    HVX_سمتیہ curr_ind_vec = ind_init_vec;
    لكل (uint32_t j = 0; j < عدد_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = گنتی % 32;
    إذا (rem > 0) {
        hvx_vec_store_a((فراغ *) &indices_buf_vec[عدد_vecs], rem * حجم(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
ساكن مضمن فراغ vec_cas(HVX_سمتیہ * X_val, HVX_سمتیہ * X_idx, HVX_سمتیہ * Y_val, HVX_سمتیہ * Y_idx, bool asc) {
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
ساكن مضمن فراغ bitonic_cas_32(HVX_سمتیہ * V, HVX_سمتیہ * I, صحيح d, HVX_سمتیہ_Pred dir_نقاب, HVX_سمتیہ idx_vec, HVX_سمتیہ zero_vec) {
    HVX_سمتیہ_Pred نقاب_بایاں;
    HVX_سمتیہ V_rot_بایاں, V_rot_دایاں;
    HVX_سمتیہ I_rot_بایاں, I_rot_دایاں;

    إذا (d == 1) {
        نقاب_بایاں = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_بایاں = Q6_V_vror_VR(*V, 4);
        V_rot_دایاں = Q6_V_vror_VR(*V, 124);
        I_rot_بایاں = Q6_V_vror_VR(*I, 4);
        I_rot_دایاں = Q6_V_vror_VR(*I, 124);
    } وإلا إذا (d == 2) {
        نقاب_بایاں = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_بایاں = Q6_V_vror_VR(*V, 8);
        V_rot_دایاں = Q6_V_vror_VR(*V, 120);
        I_rot_بایاں = Q6_V_vror_VR(*I, 8);
        I_rot_دایاں = Q6_V_vror_VR(*I, 120);
    } وإلا إذا (d == 4) {
        نقاب_بایاں = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_بایاں = Q6_V_vror_VR(*V, 16);
        V_rot_دایاں = Q6_V_vror_VR(*V, 112);
        I_rot_بایاں = Q6_V_vror_VR(*I, 16);
        I_rot_دایاں = Q6_V_vror_VR(*I, 112);
    } وإلا إذا (d == 8) {
        نقاب_بایاں = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_بایاں = Q6_V_vror_VR(*V, 32);
        V_rot_دایاں = Q6_V_vror_VR(*V, 96);
        I_rot_بایاں = Q6_V_vror_VR(*I, 32);
        I_rot_دایاں = Q6_V_vror_VR(*I, 96);
    } وإلا {
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
    HVX_سمتیہ_Pred Q_وٹاندرا = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_نقاب, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_نقاب), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_وٹاندرا, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_وٹاندرا, I_paired, *I);
}

__attribute__((always_inline))
ساكن مضمن فراغ bitonic_ترتیب_دیو_generic_hvx(uint8_t * اقدار, uint8_t * indices, صحيح K, bool asc_ترتیب) {
    HVX_سمتیہ V[32];
    HVX_سمتیہ I[32];

    HVX_سمتیہ zero_vec = Q6_V_vzero();
    HVX_سمتیہ idx_vec = *(ثابت HVX_سمتیہ *) argsort_ramp_lut;

    لكل (صحيح v = 0; v < K; v++) {
        V[v] = *(HVX_سمتیہ *)(اقدار + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_سمتیہ_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_سمتیہ_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    صحيح M = 5;
    طالما ((1 << (M - 5)) < K) M++;

    لكل (صحيح s = 1; s <= M; s++) {
        لكل (صحيح stage_d = s - 1; stage_d >= 0; stage_d--) {
            صحيح d = 1 << stage_d;
            إذا (d >= 32) {
                صحيح v_dist = d / 32;
                لكل (صحيح v1 = 0; v1 < K; v1++) {
                    إذا ((v1 & v_dist) == 0) {
                        صحيح v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_ترتیب;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } وإلا {
                إذا (s < 5) {
                    HVX_سمتیہ_Pred dir_نقاب = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    لكل (صحيح v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_نقاب, idx_vec, zero_vec);
                    }
                } وإلا {
                    لكل (صحيح v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_ترتیب;
                        HVX_سمتیہ_Pred dir_نقاب = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_نقاب, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    لكل (صحيح v = 0; v < K; v++) {
        *(HVX_سمتیہ *)(اقدار + v * 128)  = V[v];
        *(HVX_سمتیہ *)(indices + v * 128) = I[v];
    }
}

ساكن فراغ bitonic_ترتیب_دیو_vtcm_desc(uint8_t * اقدار, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_سمتیہ zero_vec = Q6_V_vzero();
    HVX_سمتیہ idx_vec = *(ثابت HVX_سمتیہ *) argsort_ramp_lut;

    HVX_سمتیہ_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_سمتیہ_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    إذا (init_indices) {
        لكل (uint32_t v = 0; v < n_vec; v++) {
            HVX_سمتیہ idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_سمتیہ *)(indices + v * 128) = idx;
        }
    }

    صحيح M = 5;
    طالما ((1u << (M - 5)) < n_vec) M++;

    لكل (صحيح s = 1; s <= M; s++) {
        لكل (صحيح stage_d = s - 1; stage_d >= 0; stage_d--) {
            صحيح d = 1 << stage_d;
            إذا (d >= 32) {
                uint32_t v_dist = d / 32;
                لكل (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    إذا ((v1 & v_dist) == 0) {
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
            } وإلا {
                إذا (s < 5) {
                    HVX_سمتیہ_Pred dir_نقاب = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    لكل (uint32_t v = 0; v < n_vec; v++) {
                        HVX_سمتیہ Vv = *(HVX_سمتیہ *)(اقدار + v * 128);
                        HVX_سمتیہ Iv = *(HVX_سمتیہ *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_نقاب, idx_vec, zero_vec);

                        *(HVX_سمتیہ *)(اقدار + v * 128)  = Vv;
                        *(HVX_سمتیہ *)(indices + v * 128) = Iv;
                    }
                } وإلا {
                    لكل (uint32_t v = 0; v < n_vec; v++) {
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

ساكن فراغ چوٹی_k_select_tiled(dma_قطار * dma_q, بنية htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               عائم * اقدار_buf, int32_t * indices_buf,
                               عائم * باہرلا_اقدار, int32_t * باہرلا_indices,
                               uint32_t chunk_ادھار) {
    ثابت uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_گنتی    = n_tiles * k;
    uint32_t ملاؤ_n_vec        = hmx_ceil_div(candidate_گنتی, 32);
    uint32_t ملاؤ_n_vec_pow2   = 1;
    طالما (ملاؤ_n_vec_pow2 < ملاؤ_n_vec) ملاؤ_n_vec_pow2 <<= 1;
    uint32_t ملاؤ_elems        = ملاؤ_n_vec_pow2 * 32;
    عائم * candidate_اقدار    = اقدار_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    لكل (uint32_t وتھ = 0; وتھ < n; وتھ += tile_elems) {
        uint32_t tile_گنتی = MIN(tile_elems, n - وتھ);
        dma_sync_read(dma_q, اقدار_buf, src_addr + (size_t) وتھ * حجم(عائم), tile_گنتی * حجم(عائم));

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) وتھ);
        إذا (tile_گنتی < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (اقدار_buf + tile_گنتی), -INFINITY, tile_elems - tile_گنتی);
        }

        bitonic_ترتیب_دیو_vtcm_desc((uint8_t *) اقدار_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_گنتی);
        لكل (uint32_t j = 0; j < tile_k; j++) {
            candidate_اقدار[candidate_pos] = اقدار_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) وتھ;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) وتھ);
    }

    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    إذا (ملاؤ_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_اقدار + candidate_pos), -INFINITY, ملاؤ_elems - candidate_pos);
        لكل (uint32_t j = candidate_pos; j < ملاؤ_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_ترتیب_دیو_vtcm_desc((uint8_t *) candidate_اقدار, (uint8_t *) candidate_indices, ملاؤ_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    لكل (uint32_t j = 0; j < k; j++) {
        باہرلا_اقدار[j]  = candidate_اقدار[j];
        باہرلا_indices[j] = candidate_indices[j] + (int32_t) chunk_ادھار;
    }
}

__attribute__((always_inline))
ساكن مضمن فراغ sort32_f32_hvx(uint8_t * اقدار, uint8_t * indices, تعداد ggml_ترتیب_دیو_ترتیب ترتیب) {
    bitonic_ترتیب_دیو_generic_hvx(اقدار, indices, 1, ترتیب == GGML_ترتیب_دیو_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort64_f32_hvx(uint8_t * اقدار, uint8_t * indices, تعداد ggml_ترتیب_دیو_ترتیب ترتیب) {
    bitonic_ترتیب_دیو_generic_hvx(اقدار, indices, 2, ترتیب == GGML_ترتیب_دیو_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort128_f32_hvx(uint8_t * اقدار, uint8_t * indices, تعداد ggml_ترتیب_دیو_ترتیب ترتیب) {
    bitonic_ترتیب_دیو_generic_hvx(اقدار, indices, 4, ترتیب == GGML_ترتیب_دیو_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort256_f32_hvx(uint8_t * اقدار, uint8_t * indices, تعداد ggml_ترتیب_دیو_ترتیب ترتیب) {
    bitonic_ترتیب_دیو_generic_hvx(اقدار, indices, 8, ترتیب == GGML_ترتیب_دیو_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort512_f32_hvx(uint8_t * اقدار, uint8_t * indices, تعداد ggml_ترتیب_دیو_ترتیب ترتیب) {
    bitonic_ترتیب_دیو_generic_hvx(اقدار, indices, 16, ترتیب == GGML_ترتیب_دیو_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort1024_f32_hvx(uint8_t * اقدار, uint8_t * indices, تعداد ggml_ترتیب_دیو_ترتیب ترتیب) {
    bitonic_ترتیب_دیو_generic_hvx(اقدار, indices, 32, ترتیب == GGML_ترتیب_دیو_ترتیب_ASC);
}

ساكن فراغ ملاؤ_runs(
    ثابت عائم * محدود وچ_val0, ثابت int32_t * محدود وچ_idx0, uint32_t n0,
    ثابت عائم * محدود وچ_val1, ثابت int32_t * محدود وچ_idx1, uint32_t n1,
    عائم * محدود باہرلا_val, int32_t * محدود باہرلا_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t باہرلا = 0;

    إذا (asc) {
        طالما (i0 < n0 && i1 < n1) {
            إذا (وچ_val0[i0] <= وچ_val1[i1]) {
                إذا (باہرلا_val) باہرلا_val[باہرلا] = وچ_val0[i0];
                باہرلا_idx[باہرلا] = وچ_idx0[i0];
                i0++;
            } وإلا {
                إذا (باہرلا_val) باہرلا_val[باہرلا] = وچ_val1[i1];
                باہرلا_idx[باہرلا] = وچ_idx1[i1];
                i1++;
            }
            باہرلا++;
        }
    } وإلا {
        طالما (i0 < n0 && i1 < n1) {
            إذا (وچ_val0[i0] >= وچ_val1[i1]) {
                إذا (باہرلا_val) باہرلا_val[باہرلا] = وچ_val0[i0];
                باہرلا_idx[باہرلا] = وچ_idx0[i0];
                i0++;
            } وإلا {
                إذا (باہرلا_val) باہرلا_val[باہرلا] = وچ_val1[i1];
                باہرلا_idx[باہرلا] = وچ_idx1[i1];
                i1++;
            }
            باہرلا++;
        }
    }

    طالما (i0 < n0) {
        إذا (باہرلا_val) باہرلا_val[باہرلا] = وچ_val0[i0];
        باہرلا_idx[باہرلا] = وچ_idx0[i0];
        i0++;
        باہرلا++;
    }

    طالما (i1 < n1) {
        إذا (باہرلا_val) باہرلا_val[باہرلا] = وچ_val1[i1];
        باہرلا_idx[باہرلا] = وچ_idx1[i1];
        i1++;
        باہرلا++;
    }
}

بنية htp_ترتیب_دیو_chunk_ctx {
    بنية htp_ops_context *              octx;
    ثابت بنية htp_ترتیب_دیو_kernel_params * kparams;
    uint8_t *                             vtcm_ادھار;
    uint32_t                              real_گنتی[HTP_MAX_NTHREADS];
};

ساكن فراغ htp_ترتیب_دیو_chunk_کم(بنانشانی صحيح n, بنانشانی صحيح i, فراغ * انکڑے) {
    بنية htp_ترتیب_دیو_chunk_ctx * cctx = (بنية htp_ترتیب_دیو_chunk_ctx *) انکڑے;
    بنية htp_ops_context * octx = cctx->octx;
    ثابت بنية htp_ترتیب_دیو_kernel_params * kparams = cctx->kparams;
    ثابت بنية htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_ادھار  = i * chunk_elems;

    uint32_t real_گنتی = (chunk_ادھار < ne00) ? MIN(chunk_elems, ne00 - chunk_ادھار) : 0;
    cctx->real_گنتی[i] = real_گنتی;

    uint8_t * spad = cctx->vtcm_ادھار + (size_t) kparams->phase1_تھاں_ناپ * i;
    عائم *   اقدار_buf  = (عائم *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * حجم(عائم));

    عائم *   ملاؤ_اقدار  = (عائم *)   (cctx->vtcm_ادھار + kparams->ملاؤ_اقدار_off);
    int32_t * ملاؤ_indices = (int32_t *) (cctx->vtcm_ادھار + kparams->ملاؤ_indices_off);

    بنية htp_thread_trace * tr = &octx->ctx->trace[i];
    ثابت dma_addr_t src_addr = src0->انکڑے + (size_t) chunk_ادھار * حجم(عائم);

    إذا (kparams->is_چوٹی_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        إذا (real_گنتی == 0) {
            لكل (uint32_t j = 0; j < local_k; j++) {
                ملاؤ_اقدار[i * local_k + j]  = -INFINITY;
                ملاؤ_indices[i * local_k + j] = 0;
            }
            أرجع;
        }

        إذا (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], اقدار_buf, src_addr, real_گنتی * حجم(عائم));

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            عائم max_قدر;
            uint32_t max_اشاریہ = چوٹی_k_max_قدر_اشاریہ(اقدار_buf, real_گنتی, &max_قدر);
            ملاؤ_اقدار[i] = max_قدر;
            ملاؤ_indices[i] = (int32_t) (max_اشاریہ + chunk_ادھار);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            أرجع;
        }

        إذا (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            چوٹی_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_گنتی, local_k,
                               اقدار_buf, indices_buf,
                               &ملاؤ_اقدار[i * local_k], &ملاؤ_indices[i * local_k],
                               chunk_ادھار);
            أرجع;
        }

        dma_sync_read(octx->ctx->dma[i], اقدار_buf, src_addr, real_گنتی * حجم(عائم));

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        إذا (chunk_elems > real_گنتی) {
            hvx_splat_f32_u((uint8_t *)(اقدار_buf + real_گنتی), -INFINITY, chunk_elems - real_گنتی);
        }

        bitonic_ترتیب_دیو_vtcm_desc((uint8_t *) اقدار_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        لكل (uint32_t j = 0; j < local_k; j++) {
            ملاؤ_اقدار[i * local_k + j]  = اقدار_buf[j];
            ملاؤ_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_ادھار;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } وإلا {
        إذا (real_گنتی > 0) {
            dma_sync_read(octx->ctx->dma[i], اقدار_buf, src_addr, real_گنتی * حجم(عائم));

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_گنتی, chunk_ادھار);

            إذا (kparams->ترتیب == GGML_ترتیب_دیو_ترتیب_ASC) {
                quicksort_اقدار_indices_asc(اقدار_buf, indices_buf, 0, real_گنتی - 1);
            } وإلا {
                quicksort_اقدار_indices_desc(اقدار_buf, indices_buf, 0, real_گنتی - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

بنية htp_argsort_ملاؤ_l1_ctx {
    بنية htp_ترتیب_دیو_chunk_ctx * cctx;
    bool                        asc;
};

ساكن فراغ htp_argsort_ملاؤ_l1_کم(بنانشانی صحيح n, بنانشانی صحيح i, فراغ * انکڑے) {
    بنية htp_argsort_ملاؤ_l1_ctx * mctx = (بنية htp_argsort_ملاؤ_l1_ctx *) انکڑے;
    بنية htp_ترتیب_دیو_chunk_ctx * cctx = mctx->cctx;
    ثابت بنية htp_ترتیب_دیو_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_گنتی[c0];
    uint32_t count1 = cctx->real_گنتی[c1];

    size_t تھاں_ناپ = (size_t) kparams->phase1_تھاں_ناپ;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    ثابت عائم *   val0 = (ثابت عائم *)   (cctx->vtcm_ادھار + تھاں_ناپ * c0);
    ثابت int32_t * idx0 = (ثابت int32_t *) (cctx->vtcm_ادھار + تھاں_ناپ * c0 + chunk_elems * حجم(عائم));
    ثابت عائم *   val1 = (ثابت عائم *)   (cctx->vtcm_ادھار + تھاں_ناپ * c1);
    ثابت int32_t * idx1 = (ثابت int32_t *) (cctx->vtcm_ادھار + تھاں_ناپ * c1 + chunk_elems * حجم(عائم));

    uint32_t باہرلا_وتھ = (2 * i) * chunk_elems;
    عائم *   باہرلا_val = (عائم *)   (cctx->vtcm_ادھار + kparams->ملاؤ_اقدار_off)  + باہرلا_وتھ;
    int32_t * باہرلا_idx = (int32_t *) (cctx->vtcm_ادھار + kparams->ملاؤ_indices_off) + باہرلا_وتھ;

    بنية htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    ملاؤ_runs(val0, idx0, count0, val1, idx1, count1, باہرلا_val, باہرلا_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

بنية htp_argsort_ملاؤ_l2_ctx {
    بنية htp_ترتیب_دیو_chunk_ctx * cctx;
    bool                        asc;
};

ساكن فراغ htp_argsort_ملاؤ_l2_کم(بنانشانی صحيح n, بنانشانی صحيح i, فراغ * انکڑے) {
    بنية htp_argsort_ملاؤ_l2_ctx * mctx = (بنية htp_argsort_ملاؤ_l2_ctx *) انکڑے;
    بنية htp_ترتیب_دیو_chunk_ctx * cctx = mctx->cctx;
    ثابت بنية htp_ترتیب_دیو_kernel_params * kparams = cctx->kparams;

    size_t تھاں_ناپ = (size_t) kparams->phase1_تھاں_ناپ;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    ثابت عائم *   val0 = (ثابت عائم *)   (cctx->vtcm_ادھار + kparams->ملاؤ_اقدار_off)  + off0;
    ثابت int32_t * idx0 = (ثابت int32_t *) (cctx->vtcm_ادھار + kparams->ملاؤ_indices_off) + off0;
    ثابت عائم *   val1 = (ثابت عائم *)   (cctx->vtcm_ادھار + kparams->ملاؤ_اقدار_off)  + off1;
    ثابت int32_t * idx1 = (ثابت int32_t *) (cctx->vtcm_ادھار + kparams->ملاؤ_indices_off) + off1;

    uint32_t count0 = cctx->real_گنتی[4*i + 0] + cctx->real_گنتی[4*i + 1];
    uint32_t count1 = cctx->real_گنتی[4*i + 2] + cctx->real_گنتی[4*i + 3];

    uint8_t * تھاں_ادھار = cctx->vtcm_ادھار + (4 * i) * تھاں_ناپ;
    عائم *   باہرلا_val   = (عائم *) تھاں_ادھار;
    int32_t * باہرلا_idx   = (int32_t *) (تھاں_ادھار + 4 * chunk_elems * حجم(عائم));

    بنية htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    ملاؤ_runs(val0, idx0, count0, val1, idx1, count1, باہرلا_val, باہرلا_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

بنية htp_ترتیب_دیو_multi_صف_ctx {
    بنية htp_ops_context *              octx;
    ثابت بنية htp_ترتیب_دیو_kernel_params * kparams;
    uint8_t *                             vtcm_ادھار;
    uint32_t                              nrows_فی_thread;
};

بنية htp_tensor_صف_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ساكن مضمن بنية htp_tensor_صف_coord init_صف_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    أرجع (بنية htp_tensor_صف_coord){ i01, i02, i03 };
}

ساكن مضمن فراغ قدم_صف_coord(بنية htp_tensor_صف_coord * c, uint32_t ne01, uint32_t ne02) {
    c->i01++;
    إذا (c->i01 == ne01) {
        c->i01 = 0;
        c->i02++;
        إذا (c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

ساكن مضمن size_t coord_وتھ(بنية htp_tensor_صف_coord c, size_t nb1, size_t nb2, size_t nb3) {
    أرجع (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

ساكن مضمن فراغ compute_صف_ترتیب_دیو(
    عائم * cur_اقدار,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_چوٹی_k,
    تعداد ggml_ترتیب_دیو_ترتیب ترتیب
) {
    إذا (is_چوٹی_k) {
        إذا (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            بدل (ne00) {
                حالة 32:   sort32_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, GGML_ترتیب_دیو_ترتیب_DESC); اكسر;
                حالة 64:   sort64_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, GGML_ترتیب_دیو_ترتیب_DESC); اكسر;
                حالة 128:  sort128_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, GGML_ترتیب_دیو_ترتیب_DESC); اكسر;
                حالة 256:  sort256_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, GGML_ترتیب_دیو_ترتیب_DESC); اكسر;
                حالة 512:  sort512_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, GGML_ترتیب_دیو_ترتیب_DESC); اكسر;
                حالة 1024: sort1024_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, GGML_ترتیب_دیو_ترتیب_DESC); اكسر;
            }
        } وإلا {
            إذا (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_اقدار + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_ترتیب_دیو_vtcm_desc((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } وإلا {
        إذا (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            بدل (ne00) {
                حالة 32:   sort32_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 64:   sort64_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 128:  sort128_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 256:  sort256_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 512:  sort512_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 1024: sort1024_f32_hvx((uint8_t *) cur_اقدار, (uint8_t *) cur_indices, ترتیب); اكسر;
            }
        } وإلا {
            init_indices_ramp(cur_indices, ne00, 0);
            إذا (ترتیب == GGML_ترتیب_دیو_ترتیب_ASC) {
                quicksort_اقدار_indices_asc(cur_اقدار, cur_indices, 0, ne00 - 1);
            } وإلا {
                quicksort_اقدار_indices_desc(cur_اقدار, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

ساكن فراغ htp_ترتیب_دیو_multi_صف_کم(بنانشانی صحيح n, بنانشانی صحيح ith, فراغ * انکڑے) {
    بنية htp_ترتیب_دیو_multi_صف_ctx * mctx = (بنية htp_ترتیب_دیو_multi_صف_ctx *) انکڑے;
    بنية htp_ops_context * octx = mctx->octx;
    ثابت بنية htp_ترتیب_دیو_kernel_params * kparams = mctx->kparams;
    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    uint32_t ne00 = (uint32_t) kparams->ne00;
    uint32_t ne01 = src0->ne[1];
    uint32_t ne02 = src0->ne[2];

    uint32_t nb01 = src0->nb[1];
    uint32_t nb02 = src0->nb[2];
    uint32_t nb03 = src0->nb[3];

    uint32_t nb1 = dst->nb[1];
    uint32_t nb2 = dst->nb[2];
    uint32_t nb3 = dst->nb[3];

    uint32_t صف_شروع = (uint32_t) kparams->صف_شروع;
    uint32_t صف_سرا   = (uint32_t) kparams->صف_سرا;
    uint32_t r_شروع   = صف_شروع + ith * mctx->nrows_فی_thread;
    uint32_t r_سرا     = MIN(r_شروع + mctx->nrows_فی_thread, صف_سرا);

    إذا (r_شروع >= r_سرا) أرجع;

    size_t spad_وتھ = ith * (size_t) (kparams->phase1_تھاں_ناپ * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_ادھار + spad_وتھ;
    size_t تھاں_ناپ = (size_t) kparams->phase1_تھاں_ناپ;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    عائم *   اقدار_buf  = (عائم *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * حجم(عائم));

    dma_قطار * dma_q = octx->ctx->dma[ith];
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_چوٹی_k = kparams->is_چوٹی_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    تعداد ggml_ترتیب_دیو_ترتیب ترتیب = (تعداد ggml_ترتیب_دیو_ترتیب) kparams->ترتیب;

    ثابت uint32_t src_بائٹاں = ne00 * حجم(عائم);
    ثابت uint32_t dst_بائٹاں = (is_چوٹی_k ? k : ne00) * حجم(int32_t);

    إذا (kparams->n_slots == 2) {
        بنية htp_tensor_صف_coord مفرد_coord = init_صف_coord(r_شروع, ne01, ne02);
        لكل (uint32_t r = r_شروع, spad_idx = 0; r < r_سرا && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * تھاں_ناپ;
            عائم *   cur_اقدار  = (عائم *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * حجم(عائم));

            dma_addr_t dst_addr = dst->انکڑے + coord_وتھ(مفرد_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->انکڑے + coord_وتھ(مفرد_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(dst_addr, cur_indices),
                           dst_بائٹاں, dst_بائٹاں, dst_بائٹاں, 0);

            // Prefetch input row
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(cur_اقدار, src_addr),
                           src_بائٹاں, src_بائٹاں, src_بائٹاں, 1);

            قدم_صف_coord(&مفرد_coord, ne01, ne02);
        }

        بنية htp_tensor_صف_coord dst_coord = init_صف_coord(r_شروع, ne01, ne02);
        بنية htp_tensor_صف_coord اگلا_src_coord = init_صف_coord(r_شروع + 2, ne01, ne02);

        لكل (uint32_t r = r_شروع; r < r_سرا; r++) {
            uint32_t cur_تھاں = (r - r_شروع) & 1;
            uint8_t * cur_spad = thread_spad + cur_تھاں * تھاں_ناپ;
            عائم *   cur_اقدار  = (عائم *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * حجم(عائم));

            // Wait for previous writeback of this slot to complete
            dma_قطار_نکالو(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_قطار_نکالو(dma_q);

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_صف_ترتیب_دیو(cur_اقدار, cur_indices, ne00, chunk_elems, is_چوٹی_k, ترتیب);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->انکڑے + coord_وتھ(dst_coord, nb1, nb2, nb3);
            dma_قطار_ڈالو(dma_q, dma_make_انکڑے(dst_addr, cur_indices),
                           dst_بائٹاں, dst_بائٹاں, dst_بائٹاں, 1);
            قدم_صف_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            ثابت uint32_t اگلا_صف = r + 2;
            إذا (اگلا_صف < r_سرا) {
                dma_addr_t اگلا_src_addr = src0->انکڑے + coord_وتھ(اگلا_src_coord, nb01, nb02, nb03);
                dma_قطار_ڈالو(dma_q, dma_make_انکڑے(cur_اقدار, اگلا_src_addr),
                               src_بائٹاں, src_بائٹاں, src_بائٹاں, 1);
                قدم_صف_coord(&اگلا_src_coord, ne01, ne02);
            }
        }

        dma_قطار_flush(dma_q);
    } وإلا {
        بنية htp_tensor_صف_coord coord = init_صف_coord(r_شروع, ne01, ne02);
        لكل (uint32_t r = r_شروع; r < r_سرا; r++) {
            dma_addr_t src_addr = src0->انکڑے + coord_وتھ(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->انکڑے + coord_وتھ(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, اقدار_buf, src_addr, src_بائٹاں);

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_صف_ترتیب_دیو(اقدار_buf, indices_buf, ne00, chunk_elems, is_چوٹی_k, ترتیب);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_بائٹاں);
            قدم_صف_coord(&coord, ne01, ne02);
        }
    }
}

ساكن صحيح op_ترتیب_دیو_common(بنية htp_ops_context * octx, bool is_چوٹی_k) {
    إذا (octx->src[0]->type != HTP_TYPE_F32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t کل_صفاں = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_صف_ناپ = dst->ne[0] * حجم(int32_t);

    uint32_t صف_شروع = 0;
    uint32_t صف_سرا   = کل_صفاں;
    إذا (octx->ctx->mdev.گنتی > 1) {
        uint32_t صفاں_فی_chunk = 0;
        htp_tensor_mdev_صفاں_فی_chunk(dst, حجم(int32_t), (uint32_t) dst_صف_ناپ, &صفاں_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصے_کرو(
            کل_صفاں, صفاں_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.گنتی, &octx->ctx->mdev.گنتی_div);
        صف_شروع = range.شروع;
        صف_سرا   = range.شروع + range.گنتی;
    }

    ثابت uint32_t nrows = صف_سرا - صف_شروع;
    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    بنية htp_ترتیب_دیو_kernel_params kparams_local;
    ثابت بنية htp_ترتیب_دیو_kernel_params * kparams = (ثابت بنية htp_ترتیب_دیو_kernel_params *) octx->kernel_params;

    إذا (kparams->n_دھاگے == 0) {
        بنية htp_ترتیب_دیو_vtcm_layout layout;
        إذا (!htp_ترتیب_دیو_solve_layout(&layout, ne00, کل_صفاں, k, octx->n_دھاگے, octx->ctx->vtcm_ناپ, is_چوٹی_k)) {
            أرجع HTP_حالت_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, حجم(kparams_local));
        kparams_local.n_دھاگے         = (int32_t) layout.n_دھاگے;
        kparams_local.کل_صفاں        = (int32_t) کل_صفاں;
        kparams_local.صف_شروع         = (int32_t) صف_شروع;
        kparams_local.صف_سرا           = (int32_t) صف_سرا;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.ترتیب             = is_چوٹی_k ? GGML_ترتیب_دیو_ترتیب_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_چوٹی_k          = is_چوٹی_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_ناپ         = (int32_t) layout.کل_بائٹاں;
        kparams_local.phase1_تھاں_ناپ  = (int32_t) layout.phase1_تھاں_ناپ;
        kparams_local.ملاؤ_اقدار_off  = (int32_t) layout.ملاؤ_اقدار_off;
        kparams_local.ملاؤ_indices_off = (int32_t) layout.ملاؤ_indices_off;
        kparams_local.ملاؤ_elems       = (int32_t) layout.ملاؤ_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } وإلا إذا (octx->ctx->mdev.گنتی > 1) {
        حافظہ_نقل(&kparams_local, kparams, حجم(kparams_local));
        kparams_local.صف_شروع = (int32_t) صف_شروع;
        kparams_local.صف_سرا   = (int32_t) صف_سرا;
        kparams = &kparams_local;
    }

    إذا (octx->ctx->vtcm_ناپ < (size_t) kparams->vtcm_ناپ) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_ناپ, octx->ctx->vtcm_ناپ);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_ادھار = (uint8_t *) octx->ctx->vtcm_ادھار;

    إذا (کل_صفاں == 1 && kparams->n_chunks > 1) {
        بنية htp_ترتیب_دیو_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_ادھار = vtcm_ادھار;
        memset(cctx.real_گنتی, 0, حجم(cctx.real_گنتی));

        work_قطار_لگاتار(octx->ctx->work_قطار, htp_ترتیب_دیو_chunk_کم, &cctx, (uint32_t) kparams->n_chunks);

        إذا (is_چوٹی_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            إذا (k == 1 && ne00 >= 128*1024) {
                عائم *   cand_vals = (عائم *)   (vtcm_ادھار + kparams->ملاؤ_اقدار_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_ادھار + kparams->ملاؤ_indices_off);
                عائم بہترین_val = cand_vals[0];
                int32_t بہترین_idx = cand_idxs[0];
                لكل (int32_t c = 1; c < kparams->n_chunks; c++) {
                    إذا (cand_vals[c] > بہترین_val) {
                        بہترین_val = cand_vals[c];
                        بہترین_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->انکڑے;
                dst_ptr[0] = بہترین_idx;
                أرجع HTP_حالت_OK;
            }

            uint32_t کل_candidates = (uint32_t) kparams->n_chunks * local_k;
            عائم *   cand_vals = (عائم *)   (vtcm_ادھار + kparams->ملاؤ_اقدار_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_ادھار + kparams->ملاؤ_indices_off);
            uint32_t ملاؤ_elems = (uint32_t) kparams->ملاؤ_elems;

            إذا (ملاؤ_elems > کل_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + کل_candidates), -INFINITY, ملاؤ_elems - کل_candidates);
                لكل (uint32_t j = کل_candidates; j < ملاؤ_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            بنية htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_ترتیب_دیو_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, ملاؤ_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->انکڑے, cand_idxs, k * حجم(int32_t));
        } وإلا {
            bool asc = kparams->ترتیب == GGML_ترتیب_دیو_ترتیب_ASC;
            إذا (kparams->n_chunks == 8) {
                بنية htp_argsort_ملاؤ_l1_ctx l1ctx = { &cctx, asc };
                work_قطار_لگاتار(octx->ctx->work_قطار, htp_argsort_ملاؤ_l1_کم, &l1ctx, 4);

                بنية htp_argsort_ملاؤ_l2_ctx l2ctx = { &cctx, asc };
                work_قطار_لگاتار(octx->ctx->work_قطار, htp_argsort_ملاؤ_l2_کم, &l2ctx, 2);

                size_t تھاں_ناپ = (size_t) kparams->phase1_تھاں_ناپ;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ثابت عائم *   val0 = (ثابت عائم *) vtcm_ادھار;
                ثابت int32_t * idx0 = (ثابت int32_t *) (vtcm_ادھار + 4 * chunk_elems * حجم(عائم));
                ثابت عائم *   val1 = (ثابت عائم *) (vtcm_ادھار + 4 * تھاں_ناپ);
                ثابت int32_t * idx1 = (ثابت int32_t *) (vtcm_ادھار + 4 * تھاں_ناپ + 4 * chunk_elems * حجم(عائم));

                uint32_t n01 = cctx.real_گنتی[0] + cctx.real_گنتی[1] + cctx.real_گنتی[2] + cctx.real_گنتی[3];
                uint32_t n23 = cctx.real_گنتی[4] + cctx.real_گنتی[5] + cctx.real_گنتی[6] + cctx.real_گنتی[7];

                int32_t * final_idx = (int32_t *) (vtcm_ادھار + kparams->ملاؤ_indices_off);
                بنية htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ملاؤ_runs(val0, idx0, n01, val1, idx1, n23, عدم, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->انکڑے, final_idx, ne00 * حجم(int32_t));
            } وإلا إذا (kparams->n_chunks == 4) {
                بنية htp_argsort_ملاؤ_l1_ctx l1ctx = { &cctx, asc };
                work_قطار_لگاتار(octx->ctx->work_قطار, htp_argsort_ملاؤ_l1_کم, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_گنتی[0] + cctx.real_گنتی[1];
                uint32_t n23 = cctx.real_گنتی[2] + cctx.real_گنتی[3];

                ثابت عائم *   val01 = (ثابت عائم *)   (vtcm_ادھار + kparams->ملاؤ_اقدار_off);
                ثابت int32_t * idx01 = (ثابت int32_t *) (vtcm_ادھار + kparams->ملاؤ_indices_off);
                ثابت عائم *   val23 = (ثابت عائم *)   (vtcm_ادھار + kparams->ملاؤ_اقدار_off)  + 2 * chunk_elems;
                ثابت int32_t * idx23 = (ثابت int32_t *) (vtcm_ادھار + kparams->ملاؤ_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_ادھار;
                بنية htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ملاؤ_runs(val01, idx01, n01, val23, idx23, n23, عدم, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->انکڑے, final_idx, ne00 * حجم(int32_t));
            } وإلا إذا (kparams->n_chunks == 2) {
                size_t تھاں_ناپ = (size_t) kparams->phase1_تھاں_ناپ;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ثابت عائم *   val0 = (ثابت عائم *)   (vtcm_ادھار);
                ثابت int32_t * idx0 = (ثابت int32_t *) (vtcm_ادھار + chunk_elems * حجم(عائم));
                ثابت عائم *   val1 = (ثابت عائم *)   (vtcm_ادھار + تھاں_ناپ);
                ثابت int32_t * idx1 = (ثابت int32_t *) (vtcm_ادھار + تھاں_ناپ + chunk_elems * حجم(عائم));

                int32_t * final_idx = (int32_t *) (vtcm_ادھار + kparams->ملاؤ_indices_off);
                بنية htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ملاؤ_runs(val0, idx0, cctx.real_گنتی[0], val1, idx1, cctx.real_گنتی[1], عدم, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->انکڑے, final_idx, ne00 * حجم(int32_t));
            } وإلا {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_ادھار + chunk_elems * حجم(عائم));
                dma_sync_write(octx->ctx->dma[0], dst->انکڑے, final_idx, ne00 * حجم(int32_t));
            }
        }
        أرجع HTP_حالت_OK;
    }

    بنية htp_ترتیب_دیو_multi_صف_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_ادھار        = vtcm_ادھار;

    uint32_t dst_صف_بائٹاں = (uint32_t) dst_صف_ناپ;
    uint32_t صفاں_فی_chunk = 0;
    htp_tensor_mdev_صفاں_فی_chunk(dst, حجم(int32_t), dst_صف_بائٹاں, &صفاں_فی_chunk);
    إذا (صفاں_فی_chunk == 0) {
        صفاں_فی_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_دھاگے - 1) / (uint32_t) kparams->n_دھاگے;
    إذا (صفاں_فی_chunk > 1) {
        dr = hex_گیڑا_اتے(dr, صفاں_فی_chunk);
    }
    mctx.nrows_فی_thread = dr;

    work_قطار_لگاتار(octx->ctx->work_قطار, htp_ترتیب_دیو_multi_صف_کم, &mctx, (uint32_t) kparams->n_دھاگے);

    أرجع HTP_حالت_OK;
}

صحيح op_argsort(بنية htp_ops_context * octx) {
    أرجع op_ترتیب_دیو_common(octx, false);
}

صحيح op_چوٹی_k(بنية htp_ops_context * octx) {
    أرجع op_ترتیب_دیو_common(octx, true);
}
