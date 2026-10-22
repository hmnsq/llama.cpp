#ضمن <string.h>
#ضمن <stdlib.h>
#ضمن <math.h>
#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#تعریف_کن GGML_COMMON_DECL_C
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

#اگر_تعریف_نشده MIN
#تعریف_کن MIN(a, b) ((a) < (b) ? (a) : (b))
#پایان_اگر


ساكن فراغ quicksort_مقادیر_indices_asc(عائم * مقادیر, int32_t * indices, صحيح چپ_2, صحيح راست_2) {
    طالما (چپ_2 < راست_2) {
        عائم محور = مقادیر[چپ_2 + (راست_2 - چپ_2) / 2];
        صحيح i = چپ_2;
        صحيح j = راست_2;

        طالما (i <= j) {
            طالما (مقادیر[i] < محور) i++;
            طالما (مقادیر[j] > محور) j--;

            إذا (i <= j) {
                عائم tmp_val = مقادیر[i];
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
        إذا (j - چپ_2 < راست_2 - i) {
            إذا (چپ_2 < j) quicksort_مقادیر_indices_asc(مقادیر, indices, چپ_2, j);
            چپ_2 = i;
        } وإلا {
            إذا (i < راست_2) quicksort_مقادیر_indices_asc(مقادیر, indices, i, راست_2);
            راست_2 = j;
        }
    }
}

ساكن فراغ quicksort_مقادیر_indices_desc(عائم * مقادیر, int32_t * indices, صحيح چپ_2, صحيح راست_2) {
    طالما (چپ_2 < راست_2) {
        عائم محور = مقادیر[چپ_2 + (راست_2 - چپ_2) / 2];
        صحيح i = چپ_2;
        صحيح j = راست_2;

        طالما (i <= j) {
            طالما (مقادیر[i] > محور) i++;
            طالما (مقادیر[j] < محور) j--;

            إذا (i <= j) {
                عائم tmp_val = مقادیر[i];
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
        إذا (j - چپ_2 < راست_2 - i) {
            إذا (چپ_2 < j) quicksort_مقادیر_indices_desc(مقادیر, indices, چپ_2, j);
            چپ_2 = i;
        } وإلا {
            إذا (i < راست_2) quicksort_مقادیر_indices_desc(مقادیر, indices, i, راست_2);
            راست_2 = j;
        }
    }
}

ساكن مضمن uint32_t بالاترین_k_max_مقدار_شاخص(ثابت عائم * مقادیر, uint32_t n, عائم * مقدار) {
    int32_t max_idx = 0;
    hvx_argmax_f32(مقادیر, n, 0, مقدار, &max_idx);
    أرجع (uint32_t) max_idx;
}

ساكن ثابت int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

ساكن مضمن فراغ init_indices_ramp(int32_t * indices_buf, uint32_t شمار, uint32_t پایه) {
    ثابت HVX_وکتور ind_init_vec = Q6_Vw_vadd_VwVw(*(ثابت HVX_وکتور *) argsort_ramp_lut, Q6_V_vsplat_R(پایه));
    ثابت HVX_وکتور ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_وکتور * indices_buf_vec = (HVX_وکتور *) indices_buf;
    uint32_t رقم_vecs = شمار / 32;
    HVX_وکتور curr_ind_vec = ind_init_vec;
    لكل (uint32_t j = 0; j < رقم_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = شمار % 32;
    إذا (rem > 0) {
        hvx_vec_store_a((فراغ *) &indices_buf_vec[رقم_vecs], rem * حجم(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
ساكن مضمن فراغ vec_cas(HVX_وکتور * X_val, HVX_وکتور * X_idx, HVX_وکتور * Y_val, HVX_وکتور * Y_idx, bool asc) {
    HVX_وکتور_Pred pred = asc ? Q6_Q_vcmp_gt_VsfVsf(*X_val, *Y_val)
                              : Q6_Q_vcmp_gt_VsfVsf(*Y_val, *X_val);
    HVX_وکتور بعدی_X_val = Q6_V_vmux_QVV(pred, *Y_val, *X_val);
    HVX_وکتور بعدی_Y_val = Q6_V_vmux_QVV(pred, *X_val, *Y_val);
    HVX_وکتور بعدی_X_idx = Q6_V_vmux_QVV(pred, *Y_idx, *X_idx);
    HVX_وکتور Y_tmp_idx  = Q6_V_vmux_QVV(pred, *X_idx, *Y_idx);
    *X_val = بعدی_X_val;
    *Y_val = بعدی_Y_val;
    *X_idx = بعدی_X_idx;
    *Y_idx = Y_tmp_idx;
}

__attribute__((always_inline))
ساكن مضمن فراغ bitonic_cas_32(HVX_وکتور * V, HVX_وکتور * I, صحيح d, HVX_وکتور_Pred dir_ماسک, HVX_وکتور idx_vec, HVX_وکتور zero_vec) {
    HVX_وکتور_Pred ماسک_چپ;
    HVX_وکتور V_rot_چپ, V_rot_راست;
    HVX_وکتور I_rot_چپ, I_rot_راست;

    إذا (d == 1) {
        ماسک_چپ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_چپ = Q6_V_vror_VR(*V, 4);
        V_rot_راست = Q6_V_vror_VR(*V, 124);
        I_rot_چپ = Q6_V_vror_VR(*I, 4);
        I_rot_راست = Q6_V_vror_VR(*I, 124);
    } وإلا إذا (d == 2) {
        ماسک_چپ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_چپ = Q6_V_vror_VR(*V, 8);
        V_rot_راست = Q6_V_vror_VR(*V, 120);
        I_rot_چپ = Q6_V_vror_VR(*I, 8);
        I_rot_راست = Q6_V_vror_VR(*I, 120);
    } وإلا إذا (d == 4) {
        ماسک_چپ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_چپ = Q6_V_vror_VR(*V, 16);
        V_rot_راست = Q6_V_vror_VR(*V, 112);
        I_rot_چپ = Q6_V_vror_VR(*I, 16);
        I_rot_راست = Q6_V_vror_VR(*I, 112);
    } وإلا إذا (d == 8) {
        ماسک_چپ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_چپ = Q6_V_vror_VR(*V, 32);
        V_rot_راست = Q6_V_vror_VR(*V, 96);
        I_rot_چپ = Q6_V_vror_VR(*I, 32);
        I_rot_راست = Q6_V_vror_VR(*I, 96);
    } وإلا {
        ماسک_چپ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_چپ = Q6_V_vror_VR(*V, 64);
        V_rot_راست = Q6_V_vror_VR(*V, 64);
        I_rot_چپ = Q6_V_vror_VR(*I, 64);
        I_rot_راست = Q6_V_vror_VR(*I, 64);
    }

    HVX_وکتور V_paired = Q6_V_vmux_QVV(ماسک_چپ, V_rot_چپ, V_rot_راست);
    HVX_وکتور I_paired = Q6_V_vmux_QVV(ماسک_چپ, I_rot_چپ, I_rot_راست);

    HVX_وکتور_Pred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_وکتور_Pred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_وکتور_Pred ماسک_راست = Q6_Q_not_Q(ماسک_چپ);
    HVX_وکتور_Pred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(ماسک_چپ, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, ماسک_راست)
    );
    HVX_وکتور_Pred Q_جابجا = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_ماسک, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_ماسک), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_جابجا, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_جابجا, I_paired, *I);
}

__attribute__((always_inline))
ساكن مضمن فراغ bitonic_مرتب_سازی_generic_hvx(uint8_t * مقادیر, uint8_t * indices, صحيح K, bool asc_ترتیب) {
    HVX_وکتور V[32];
    HVX_وکتور I[32];

    HVX_وکتور zero_vec = Q6_V_vzero();
    HVX_وکتور idx_vec = *(ثابت HVX_وکتور *) argsort_ramp_lut;

    لكل (صحيح v = 0; v < K; v++) {
        V[v] = *(HVX_وکتور *)(مقادیر + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_وکتور_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_وکتور_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

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
                    HVX_وکتور_Pred dir_ماسک = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    لكل (صحيح v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_ماسک, idx_vec, zero_vec);
                    }
                } وإلا {
                    لكل (صحيح v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_ترتیب;
                        HVX_وکتور_Pred dir_ماسک = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_ماسک, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    لكل (صحيح v = 0; v < K; v++) {
        *(HVX_وکتور *)(مقادیر + v * 128)  = V[v];
        *(HVX_وکتور *)(indices + v * 128) = I[v];
    }
}

ساكن فراغ bitonic_مرتب_سازی_vtcm_desc(uint8_t * مقادیر, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_وکتور zero_vec = Q6_V_vzero();
    HVX_وکتور idx_vec = *(ثابت HVX_وکتور *) argsort_ramp_lut;

    HVX_وکتور_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_وکتور_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    إذا (init_indices) {
        لكل (uint32_t v = 0; v < n_vec; v++) {
            HVX_وکتور idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_وکتور *)(indices + v * 128) = idx;
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

                        HVX_وکتور Vv1 = *(HVX_وکتور *)(مقادیر + v1 * 128);
                        HVX_وکتور Iv1 = *(HVX_وکتور *)(indices + v1 * 128);
                        HVX_وکتور Vv2 = *(HVX_وکتور *)(مقادیر + v2 * 128);
                        HVX_وکتور Iv2 = *(HVX_وکتور *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_وکتور *)(مقادیر + v1 * 128)  = Vv1;
                        *(HVX_وکتور *)(indices + v1 * 128) = Iv1;
                        *(HVX_وکتور *)(مقادیر + v2 * 128)  = Vv2;
                        *(HVX_وکتور *)(indices + v2 * 128) = Iv2;
                    }
                }
            } وإلا {
                إذا (s < 5) {
                    HVX_وکتور_Pred dir_ماسک = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    لكل (uint32_t v = 0; v < n_vec; v++) {
                        HVX_وکتور Vv = *(HVX_وکتور *)(مقادیر + v * 128);
                        HVX_وکتور Iv = *(HVX_وکتور *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_ماسک, idx_vec, zero_vec);

                        *(HVX_وکتور *)(مقادیر + v * 128)  = Vv;
                        *(HVX_وکتور *)(indices + v * 128) = Iv;
                    }
                } وإلا {
                    لكل (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_وکتور_Pred dir_ماسک = asc ? pred_all_1s : pred_all_0s;

                        HVX_وکتور Vv = *(HVX_وکتور *)(مقادیر + v * 128);
                        HVX_وکتور Iv = *(HVX_وکتور *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_ماسک, idx_vec, zero_vec);

                        *(HVX_وکتور *)(مقادیر + v * 128)  = Vv;
                        *(HVX_وکتور *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

ساكن فراغ بالاترین_k_select_tiled(dma_صف * dma_q, بنية htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               عائم * مقادیر_buf, int32_t * indices_buf,
                               عائم * بیرون_مقادیر, int32_t * بیرون_indices,
                               uint32_t chunk_پایه) {
    ثابت uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_شمار    = n_tiles * k;
    uint32_t ادغام_n_vec        = hmx_ceil_div(candidate_شمار, 32);
    uint32_t ادغام_n_vec_pow2   = 1;
    طالما (ادغام_n_vec_pow2 < ادغام_n_vec) ادغام_n_vec_pow2 <<= 1;
    uint32_t ادغام_elems        = ادغام_n_vec_pow2 * 32;
    عائم * candidate_مقادیر    = مقادیر_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    لكل (uint32_t جابه_جایی = 0; جابه_جایی < n; جابه_جایی += tile_elems) {
        uint32_t tile_شمار = MIN(tile_elems, n - جابه_جایی);
        dma_sync_read(dma_q, مقادیر_buf, src_addr + (size_t) جابه_جایی * حجم(عائم), tile_شمار * حجم(عائم));

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) جابه_جایی);
        إذا (tile_شمار < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (مقادیر_buf + tile_شمار), -INFINITY, tile_elems - tile_شمار);
        }

        bitonic_مرتب_سازی_vtcm_desc((uint8_t *) مقادیر_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_شمار);
        لكل (uint32_t j = 0; j < tile_k; j++) {
            candidate_مقادیر[candidate_pos] = مقادیر_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) جابه_جایی;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) جابه_جایی);
    }

    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    إذا (ادغام_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_مقادیر + candidate_pos), -INFINITY, ادغام_elems - candidate_pos);
        لكل (uint32_t j = candidate_pos; j < ادغام_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_مرتب_سازی_vtcm_desc((uint8_t *) candidate_مقادیر, (uint8_t *) candidate_indices, ادغام_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    لكل (uint32_t j = 0; j < k; j++) {
        بیرون_مقادیر[j]  = candidate_مقادیر[j];
        بیرون_indices[j] = candidate_indices[j] + (int32_t) chunk_پایه;
    }
}

__attribute__((always_inline))
ساكن مضمن فراغ sort32_f32_hvx(uint8_t * مقادیر, uint8_t * indices, تعداد ggml_مرتب_سازی_ترتیب ترتیب) {
    bitonic_مرتب_سازی_generic_hvx(مقادیر, indices, 1, ترتیب == GGML_مرتب_سازی_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort64_f32_hvx(uint8_t * مقادیر, uint8_t * indices, تعداد ggml_مرتب_سازی_ترتیب ترتیب) {
    bitonic_مرتب_سازی_generic_hvx(مقادیر, indices, 2, ترتیب == GGML_مرتب_سازی_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort128_f32_hvx(uint8_t * مقادیر, uint8_t * indices, تعداد ggml_مرتب_سازی_ترتیب ترتیب) {
    bitonic_مرتب_سازی_generic_hvx(مقادیر, indices, 4, ترتیب == GGML_مرتب_سازی_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort256_f32_hvx(uint8_t * مقادیر, uint8_t * indices, تعداد ggml_مرتب_سازی_ترتیب ترتیب) {
    bitonic_مرتب_سازی_generic_hvx(مقادیر, indices, 8, ترتیب == GGML_مرتب_سازی_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort512_f32_hvx(uint8_t * مقادیر, uint8_t * indices, تعداد ggml_مرتب_سازی_ترتیب ترتیب) {
    bitonic_مرتب_سازی_generic_hvx(مقادیر, indices, 16, ترتیب == GGML_مرتب_سازی_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort1024_f32_hvx(uint8_t * مقادیر, uint8_t * indices, تعداد ggml_مرتب_سازی_ترتیب ترتیب) {
    bitonic_مرتب_سازی_generic_hvx(مقادیر, indices, 32, ترتیب == GGML_مرتب_سازی_ترتیب_ASC);
}

ساكن فراغ ادغام_runs(
    ثابت عائم * محدود درون_val0, ثابت int32_t * محدود درون_idx0, uint32_t n0,
    ثابت عائم * محدود درون_val1, ثابت int32_t * محدود درون_idx1, uint32_t n1,
    عائم * محدود بیرون_val, int32_t * محدود بیرون_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t بیرون = 0;

    إذا (asc) {
        طالما (i0 < n0 && i1 < n1) {
            إذا (درون_val0[i0] <= درون_val1[i1]) {
                إذا (بیرون_val) بیرون_val[بیرون] = درون_val0[i0];
                بیرون_idx[بیرون] = درون_idx0[i0];
                i0++;
            } وإلا {
                إذا (بیرون_val) بیرون_val[بیرون] = درون_val1[i1];
                بیرون_idx[بیرون] = درون_idx1[i1];
                i1++;
            }
            بیرون++;
        }
    } وإلا {
        طالما (i0 < n0 && i1 < n1) {
            إذا (درون_val0[i0] >= درون_val1[i1]) {
                إذا (بیرون_val) بیرون_val[بیرون] = درون_val0[i0];
                بیرون_idx[بیرون] = درون_idx0[i0];
                i0++;
            } وإلا {
                إذا (بیرون_val) بیرون_val[بیرون] = درون_val1[i1];
                بیرون_idx[بیرون] = درون_idx1[i1];
                i1++;
            }
            بیرون++;
        }
    }

    طالما (i0 < n0) {
        إذا (بیرون_val) بیرون_val[بیرون] = درون_val0[i0];
        بیرون_idx[بیرون] = درون_idx0[i0];
        i0++;
        بیرون++;
    }

    طالما (i1 < n1) {
        إذا (بیرون_val) بیرون_val[بیرون] = درون_val1[i1];
        بیرون_idx[بیرون] = درون_idx1[i1];
        i1++;
        بیرون++;
    }
}

بنية htp_مرتب_سازی_chunk_ctx {
    بنية htp_ops_context *              octx;
    ثابت بنية htp_مرتب_سازی_kernel_params * kparams;
    uint8_t *                             vtcm_پایه;
    uint32_t                              real_شمار[HTP_MAX_NTHREADS];
};

ساكن فراغ htp_مرتب_سازی_chunk_کار(بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده) {
    بنية htp_مرتب_سازی_chunk_ctx * cctx = (بنية htp_مرتب_سازی_chunk_ctx *) داده;
    بنية htp_ops_context * octx = cctx->octx;
    ثابت بنية htp_مرتب_سازی_kernel_params * kparams = cctx->kparams;
    ثابت بنية htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_پایه  = i * chunk_elems;

    uint32_t real_شمار = (chunk_پایه < ne00) ? MIN(chunk_elems, ne00 - chunk_پایه) : 0;
    cctx->real_شمار[i] = real_شمار;

    uint8_t * spad = cctx->vtcm_پایه + (size_t) kparams->phase1_جایگاه_اندازه * i;
    عائم *   مقادیر_buf  = (عائم *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * حجم(عائم));

    عائم *   ادغام_مقادیر  = (عائم *)   (cctx->vtcm_پایه + kparams->ادغام_مقادیر_off);
    int32_t * ادغام_indices = (int32_t *) (cctx->vtcm_پایه + kparams->ادغام_indices_off);

    بنية htp_thread_trace * tr = &octx->ctx->trace[i];
    ثابت dma_addr_t src_addr = src0->داده + (size_t) chunk_پایه * حجم(عائم);

    إذا (kparams->is_بالاترین_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        إذا (real_شمار == 0) {
            لكل (uint32_t j = 0; j < local_k; j++) {
                ادغام_مقادیر[i * local_k + j]  = -INFINITY;
                ادغام_indices[i * local_k + j] = 0;
            }
            أرجع;
        }

        إذا (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], مقادیر_buf, src_addr, real_شمار * حجم(عائم));

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            عائم max_مقدار;
            uint32_t max_شاخص = بالاترین_k_max_مقدار_شاخص(مقادیر_buf, real_شمار, &max_مقدار);
            ادغام_مقادیر[i] = max_مقدار;
            ادغام_indices[i] = (int32_t) (max_شاخص + chunk_پایه);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            أرجع;
        }

        إذا (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            بالاترین_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_شمار, local_k,
                               مقادیر_buf, indices_buf,
                               &ادغام_مقادیر[i * local_k], &ادغام_indices[i * local_k],
                               chunk_پایه);
            أرجع;
        }

        dma_sync_read(octx->ctx->dma[i], مقادیر_buf, src_addr, real_شمار * حجم(عائم));

        htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        إذا (chunk_elems > real_شمار) {
            hvx_splat_f32_u((uint8_t *)(مقادیر_buf + real_شمار), -INFINITY, chunk_elems - real_شمار);
        }

        bitonic_مرتب_سازی_vtcm_desc((uint8_t *) مقادیر_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        لكل (uint32_t j = 0; j < local_k; j++) {
            ادغام_مقادیر[i * local_k + j]  = مقادیر_buf[j];
            ادغام_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_پایه;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } وإلا {
        إذا (real_شمار > 0) {
            dma_sync_read(octx->ctx->dma[i], مقادیر_buf, src_addr, real_شمار * حجم(عائم));

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_شمار, chunk_پایه);

            إذا (kparams->ترتیب == GGML_مرتب_سازی_ترتیب_ASC) {
                quicksort_مقادیر_indices_asc(مقادیر_buf, indices_buf, 0, real_شمار - 1);
            } وإلا {
                quicksort_مقادیر_indices_desc(مقادیر_buf, indices_buf, 0, real_شمار - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

بنية htp_argsort_ادغام_l1_ctx {
    بنية htp_مرتب_سازی_chunk_ctx * cctx;
    bool                        asc;
};

ساكن فراغ htp_argsort_ادغام_l1_کار(بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده) {
    بنية htp_argsort_ادغام_l1_ctx * mctx = (بنية htp_argsort_ادغام_l1_ctx *) داده;
    بنية htp_مرتب_سازی_chunk_ctx * cctx = mctx->cctx;
    ثابت بنية htp_مرتب_سازی_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_شمار[c0];
    uint32_t count1 = cctx->real_شمار[c1];

    size_t جایگاه_اندازه = (size_t) kparams->phase1_جایگاه_اندازه;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    ثابت عائم *   val0 = (ثابت عائم *)   (cctx->vtcm_پایه + جایگاه_اندازه * c0);
    ثابت int32_t * idx0 = (ثابت int32_t *) (cctx->vtcm_پایه + جایگاه_اندازه * c0 + chunk_elems * حجم(عائم));
    ثابت عائم *   val1 = (ثابت عائم *)   (cctx->vtcm_پایه + جایگاه_اندازه * c1);
    ثابت int32_t * idx1 = (ثابت int32_t *) (cctx->vtcm_پایه + جایگاه_اندازه * c1 + chunk_elems * حجم(عائم));

    uint32_t بیرون_جابه_جایی = (2 * i) * chunk_elems;
    عائم *   بیرون_val = (عائم *)   (cctx->vtcm_پایه + kparams->ادغام_مقادیر_off)  + بیرون_جابه_جایی;
    int32_t * بیرون_idx = (int32_t *) (cctx->vtcm_پایه + kparams->ادغام_indices_off) + بیرون_جابه_جایی;

    بنية htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    ادغام_runs(val0, idx0, count0, val1, idx1, count1, بیرون_val, بیرون_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

بنية htp_argsort_ادغام_l2_ctx {
    بنية htp_مرتب_سازی_chunk_ctx * cctx;
    bool                        asc;
};

ساكن فراغ htp_argsort_ادغام_l2_کار(بیعلامت صحيح n, بیعلامت صحيح i, فراغ * داده) {
    بنية htp_argsort_ادغام_l2_ctx * mctx = (بنية htp_argsort_ادغام_l2_ctx *) داده;
    بنية htp_مرتب_سازی_chunk_ctx * cctx = mctx->cctx;
    ثابت بنية htp_مرتب_سازی_kernel_params * kparams = cctx->kparams;

    size_t جایگاه_اندازه = (size_t) kparams->phase1_جایگاه_اندازه;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    ثابت عائم *   val0 = (ثابت عائم *)   (cctx->vtcm_پایه + kparams->ادغام_مقادیر_off)  + off0;
    ثابت int32_t * idx0 = (ثابت int32_t *) (cctx->vtcm_پایه + kparams->ادغام_indices_off) + off0;
    ثابت عائم *   val1 = (ثابت عائم *)   (cctx->vtcm_پایه + kparams->ادغام_مقادیر_off)  + off1;
    ثابت int32_t * idx1 = (ثابت int32_t *) (cctx->vtcm_پایه + kparams->ادغام_indices_off) + off1;

    uint32_t count0 = cctx->real_شمار[4*i + 0] + cctx->real_شمار[4*i + 1];
    uint32_t count1 = cctx->real_شمار[4*i + 2] + cctx->real_شمار[4*i + 3];

    uint8_t * جایگاه_پایه = cctx->vtcm_پایه + (4 * i) * جایگاه_اندازه;
    عائم *   بیرون_val   = (عائم *) جایگاه_پایه;
    int32_t * بیرون_idx   = (int32_t *) (جایگاه_پایه + 4 * chunk_elems * حجم(عائم));

    بنية htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    ادغام_runs(val0, idx0, count0, val1, idx1, count1, بیرون_val, بیرون_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

بنية htp_مرتب_سازی_multi_ردیف_ctx {
    بنية htp_ops_context *              octx;
    ثابت بنية htp_مرتب_سازی_kernel_params * kparams;
    uint8_t *                             vtcm_پایه;
    uint32_t                              nrows_در_هر_thread;
};

بنية htp_tensor_ردیف_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ساكن مضمن بنية htp_tensor_ردیف_coord init_ردیف_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    أرجع (بنية htp_tensor_ردیف_coord){ i01, i02, i03 };
}

ساكن مضمن فراغ گام_ردیف_coord(بنية htp_tensor_ردیف_coord * c, uint32_t ne01, uint32_t ne02) {
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

ساكن مضمن size_t coord_جابه_جایی(بنية htp_tensor_ردیف_coord c, size_t nb1, size_t nb2, size_t nb3) {
    أرجع (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

ساكن مضمن فراغ compute_ردیف_مرتب_سازی(
    عائم * cur_مقادیر,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_بالاترین_k,
    تعداد ggml_مرتب_سازی_ترتیب ترتیب
) {
    إذا (is_بالاترین_k) {
        إذا (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            بدل (ne00) {
                حالة 32:   sort32_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, GGML_مرتب_سازی_ترتیب_DESC); اكسر;
                حالة 64:   sort64_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, GGML_مرتب_سازی_ترتیب_DESC); اكسر;
                حالة 128:  sort128_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, GGML_مرتب_سازی_ترتیب_DESC); اكسر;
                حالة 256:  sort256_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, GGML_مرتب_سازی_ترتیب_DESC); اكسر;
                حالة 512:  sort512_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, GGML_مرتب_سازی_ترتیب_DESC); اكسر;
                حالة 1024: sort1024_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, GGML_مرتب_سازی_ترتیب_DESC); اكسر;
            }
        } وإلا {
            إذا (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_مقادیر + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_مرتب_سازی_vtcm_desc((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } وإلا {
        إذا (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            بدل (ne00) {
                حالة 32:   sort32_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 64:   sort64_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 128:  sort128_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 256:  sort256_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 512:  sort512_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 1024: sort1024_f32_hvx((uint8_t *) cur_مقادیر, (uint8_t *) cur_indices, ترتیب); اكسر;
            }
        } وإلا {
            init_indices_ramp(cur_indices, ne00, 0);
            إذا (ترتیب == GGML_مرتب_سازی_ترتیب_ASC) {
                quicksort_مقادیر_indices_asc(cur_مقادیر, cur_indices, 0, ne00 - 1);
            } وإلا {
                quicksort_مقادیر_indices_desc(cur_مقادیر, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

ساكن فراغ htp_مرتب_سازی_multi_ردیف_کار(بیعلامت صحيح n, بیعلامت صحيح ith, فراغ * داده) {
    بنية htp_مرتب_سازی_multi_ردیف_ctx * mctx = (بنية htp_مرتب_سازی_multi_ردیف_ctx *) داده;
    بنية htp_ops_context * octx = mctx->octx;
    ثابت بنية htp_مرتب_سازی_kernel_params * kparams = mctx->kparams;
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

    uint32_t ردیف_شروع = (uint32_t) kparams->ردیف_شروع;
    uint32_t ردیف_انتها   = (uint32_t) kparams->ردیف_انتها;
    uint32_t r_شروع   = ردیف_شروع + ith * mctx->nrows_در_هر_thread;
    uint32_t r_انتها     = MIN(r_شروع + mctx->nrows_در_هر_thread, ردیف_انتها);

    إذا (r_شروع >= r_انتها) أرجع;

    size_t spad_جابه_جایی = ith * (size_t) (kparams->phase1_جایگاه_اندازه * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_پایه + spad_جابه_جایی;
    size_t جایگاه_اندازه = (size_t) kparams->phase1_جایگاه_اندازه;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    عائم *   مقادیر_buf  = (عائم *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * حجم(عائم));

    dma_صف * dma_q = octx->ctx->dma[ith];
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_بالاترین_k = kparams->is_بالاترین_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    تعداد ggml_مرتب_سازی_ترتیب ترتیب = (تعداد ggml_مرتب_سازی_ترتیب) kparams->ترتیب;

    ثابت uint32_t src_بایتها = ne00 * حجم(عائم);
    ثابت uint32_t dst_بایتها = (is_بالاترین_k ? k : ne00) * حجم(int32_t);

    إذا (kparams->n_slots == 2) {
        بنية htp_tensor_ردیف_coord اول_coord = init_ردیف_coord(r_شروع, ne01, ne02);
        لكل (uint32_t r = r_شروع, spad_idx = 0; r < r_انتها && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * جایگاه_اندازه;
            عائم *   cur_مقادیر  = (عائم *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * حجم(عائم));

            dma_addr_t dst_addr = dst->داده + coord_جابه_جایی(اول_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->داده + coord_جابه_جایی(اول_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_صف_افزودن(dma_q, dma_make_داده(dst_addr, cur_indices),
                           dst_بایتها, dst_بایتها, dst_بایتها, 0);

            // Prefetch input row
            dma_صف_افزودن(dma_q, dma_make_داده(cur_مقادیر, src_addr),
                           src_بایتها, src_بایتها, src_بایتها, 1);

            گام_ردیف_coord(&اول_coord, ne01, ne02);
        }

        بنية htp_tensor_ردیف_coord dst_coord = init_ردیف_coord(r_شروع, ne01, ne02);
        بنية htp_tensor_ردیف_coord بعدی_src_coord = init_ردیف_coord(r_شروع + 2, ne01, ne02);

        لكل (uint32_t r = r_شروع; r < r_انتها; r++) {
            uint32_t cur_جایگاه = (r - r_شروع) & 1;
            uint8_t * cur_spad = thread_spad + cur_جایگاه * جایگاه_اندازه;
            عائم *   cur_مقادیر  = (عائم *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * حجم(عائم));

            // Wait for previous writeback of this slot to complete
            dma_صف_برداشت(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_صف_برداشت(dma_q);

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_ردیف_مرتب_سازی(cur_مقادیر, cur_indices, ne00, chunk_elems, is_بالاترین_k, ترتیب);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->داده + coord_جابه_جایی(dst_coord, nb1, nb2, nb3);
            dma_صف_افزودن(dma_q, dma_make_داده(dst_addr, cur_indices),
                           dst_بایتها, dst_بایتها, dst_بایتها, 1);
            گام_ردیف_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            ثابت uint32_t بعدی_ردیف = r + 2;
            إذا (بعدی_ردیف < r_انتها) {
                dma_addr_t بعدی_src_addr = src0->داده + coord_جابه_جایی(بعدی_src_coord, nb01, nb02, nb03);
                dma_صف_افزودن(dma_q, dma_make_داده(cur_مقادیر, بعدی_src_addr),
                               src_بایتها, src_بایتها, src_بایتها, 1);
                گام_ردیف_coord(&بعدی_src_coord, ne01, ne02);
            }
        }

        dma_صف_flush(dma_q);
    } وإلا {
        بنية htp_tensor_ردیف_coord coord = init_ردیف_coord(r_شروع, ne01, ne02);
        لكل (uint32_t r = r_شروع; r < r_انتها; r++) {
            dma_addr_t src_addr = src0->داده + coord_جابه_جایی(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->داده + coord_جابه_جایی(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, مقادیر_buf, src_addr, src_بایتها);

            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_ردیف_مرتب_سازی(مقادیر_buf, indices_buf, ne00, chunk_elems, is_بالاترین_k, ترتیب);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_بایتها);
            گام_ردیف_coord(&coord, ne01, ne02);
        }
    }
}

ساكن صحيح op_مرتب_سازی_common(بنية htp_ops_context * octx, bool is_بالاترین_k) {
    إذا (octx->src[0]->type != HTP_TYPE_F32) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_وضعیت_NO_SUPPORT;
    }

    ثابت uint32_t مجموع_ردیفها = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_ردیف_اندازه = dst->ne[0] * حجم(int32_t);

    uint32_t ردیف_شروع = 0;
    uint32_t ردیف_انتها   = مجموع_ردیفها;
    إذا (octx->ctx->mdev.شمار > 1) {
        uint32_t ردیفها_در_هر_chunk = 0;
        htp_tensor_mdev_ردیفها_در_هر_chunk(dst, حجم(int32_t), (uint32_t) dst_ردیف_اندازه, &ردیفها_در_هر_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_افراز(
            مجموع_ردیفها, ردیفها_در_هر_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمار, &octx->ctx->mdev.شمار_div);
        ردیف_شروع = range.شروع;
        ردیف_انتها   = range.شروع + range.شمار;
    }

    ثابت uint32_t nrows = ردیف_انتها - ردیف_شروع;
    إذا (nrows == 0) {
        أرجع HTP_وضعیت_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    بنية htp_مرتب_سازی_kernel_params kparams_local;
    ثابت بنية htp_مرتب_سازی_kernel_params * kparams = (ثابت بنية htp_مرتب_سازی_kernel_params *) octx->kernel_params;

    إذا (kparams->n_نخها == 0) {
        بنية htp_مرتب_سازی_vtcm_layout layout;
        إذا (!htp_مرتب_سازی_solve_layout(&layout, ne00, مجموع_ردیفها, k, octx->n_نخها, octx->ctx->vtcm_اندازه, is_بالاترین_k)) {
            أرجع HTP_وضعیت_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, حجم(kparams_local));
        kparams_local.n_نخها         = (int32_t) layout.n_نخها;
        kparams_local.مجموع_ردیفها        = (int32_t) مجموع_ردیفها;
        kparams_local.ردیف_شروع         = (int32_t) ردیف_شروع;
        kparams_local.ردیف_انتها           = (int32_t) ردیف_انتها;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.ترتیب             = is_بالاترین_k ? GGML_مرتب_سازی_ترتیب_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_بالاترین_k          = is_بالاترین_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_اندازه         = (int32_t) layout.مجموع_بایتها;
        kparams_local.phase1_جایگاه_اندازه  = (int32_t) layout.phase1_جایگاه_اندازه;
        kparams_local.ادغام_مقادیر_off  = (int32_t) layout.ادغام_مقادیر_off;
        kparams_local.ادغام_indices_off = (int32_t) layout.ادغام_indices_off;
        kparams_local.ادغام_elems       = (int32_t) layout.ادغام_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } وإلا إذا (octx->ctx->mdev.شمار > 1) {
        کپی_حافظه(&kparams_local, kparams, حجم(kparams_local));
        kparams_local.ردیف_شروع = (int32_t) ردیف_شروع;
        kparams_local.ردیف_انتها   = (int32_t) ردیف_انتها;
        kparams = &kparams_local;
    }

    إذا (octx->ctx->vtcm_اندازه < (size_t) kparams->vtcm_اندازه) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_اندازه, octx->ctx->vtcm_اندازه);
        أرجع HTP_وضعیت_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_پایه = (uint8_t *) octx->ctx->vtcm_پایه;

    إذا (مجموع_ردیفها == 1 && kparams->n_chunks > 1) {
        بنية htp_مرتب_سازی_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_پایه = vtcm_پایه;
        memset(cctx.real_شمار, 0, حجم(cctx.real_شمار));

        work_صف_پیاپی(octx->ctx->work_صف, htp_مرتب_سازی_chunk_کار, &cctx, (uint32_t) kparams->n_chunks);

        إذا (is_بالاترین_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            إذا (k == 1 && ne00 >= 128*1024) {
                عائم *   cand_vals = (عائم *)   (vtcm_پایه + kparams->ادغام_مقادیر_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_پایه + kparams->ادغام_indices_off);
                عائم بهترین_val = cand_vals[0];
                int32_t بهترین_idx = cand_idxs[0];
                لكل (int32_t c = 1; c < kparams->n_chunks; c++) {
                    إذا (cand_vals[c] > بهترین_val) {
                        بهترین_val = cand_vals[c];
                        بهترین_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->داده;
                dst_ptr[0] = بهترین_idx;
                أرجع HTP_وضعیت_OK;
            }

            uint32_t مجموع_candidates = (uint32_t) kparams->n_chunks * local_k;
            عائم *   cand_vals = (عائم *)   (vtcm_پایه + kparams->ادغام_مقادیر_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_پایه + kparams->ادغام_indices_off);
            uint32_t ادغام_elems = (uint32_t) kparams->ادغام_elems;

            إذا (ادغام_elems > مجموع_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + مجموع_candidates), -INFINITY, ادغام_elems - مجموع_candidates);
                لكل (uint32_t j = مجموع_candidates; j < ادغام_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            بنية htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_مرتب_سازی_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, ادغام_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->داده, cand_idxs, k * حجم(int32_t));
        } وإلا {
            bool asc = kparams->ترتیب == GGML_مرتب_سازی_ترتیب_ASC;
            إذا (kparams->n_chunks == 8) {
                بنية htp_argsort_ادغام_l1_ctx l1ctx = { &cctx, asc };
                work_صف_پیاپی(octx->ctx->work_صف, htp_argsort_ادغام_l1_کار, &l1ctx, 4);

                بنية htp_argsort_ادغام_l2_ctx l2ctx = { &cctx, asc };
                work_صف_پیاپی(octx->ctx->work_صف, htp_argsort_ادغام_l2_کار, &l2ctx, 2);

                size_t جایگاه_اندازه = (size_t) kparams->phase1_جایگاه_اندازه;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ثابت عائم *   val0 = (ثابت عائم *) vtcm_پایه;
                ثابت int32_t * idx0 = (ثابت int32_t *) (vtcm_پایه + 4 * chunk_elems * حجم(عائم));
                ثابت عائم *   val1 = (ثابت عائم *) (vtcm_پایه + 4 * جایگاه_اندازه);
                ثابت int32_t * idx1 = (ثابت int32_t *) (vtcm_پایه + 4 * جایگاه_اندازه + 4 * chunk_elems * حجم(عائم));

                uint32_t n01 = cctx.real_شمار[0] + cctx.real_شمار[1] + cctx.real_شمار[2] + cctx.real_شمار[3];
                uint32_t n23 = cctx.real_شمار[4] + cctx.real_شمار[5] + cctx.real_شمار[6] + cctx.real_شمار[7];

                int32_t * final_idx = (int32_t *) (vtcm_پایه + kparams->ادغام_indices_off);
                بنية htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ادغام_runs(val0, idx0, n01, val1, idx1, n23, عدم, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->داده, final_idx, ne00 * حجم(int32_t));
            } وإلا إذا (kparams->n_chunks == 4) {
                بنية htp_argsort_ادغام_l1_ctx l1ctx = { &cctx, asc };
                work_صف_پیاپی(octx->ctx->work_صف, htp_argsort_ادغام_l1_کار, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_شمار[0] + cctx.real_شمار[1];
                uint32_t n23 = cctx.real_شمار[2] + cctx.real_شمار[3];

                ثابت عائم *   val01 = (ثابت عائم *)   (vtcm_پایه + kparams->ادغام_مقادیر_off);
                ثابت int32_t * idx01 = (ثابت int32_t *) (vtcm_پایه + kparams->ادغام_indices_off);
                ثابت عائم *   val23 = (ثابت عائم *)   (vtcm_پایه + kparams->ادغام_مقادیر_off)  + 2 * chunk_elems;
                ثابت int32_t * idx23 = (ثابت int32_t *) (vtcm_پایه + kparams->ادغام_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_پایه;
                بنية htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ادغام_runs(val01, idx01, n01, val23, idx23, n23, عدم, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->داده, final_idx, ne00 * حجم(int32_t));
            } وإلا إذا (kparams->n_chunks == 2) {
                size_t جایگاه_اندازه = (size_t) kparams->phase1_جایگاه_اندازه;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ثابت عائم *   val0 = (ثابت عائم *)   (vtcm_پایه);
                ثابت int32_t * idx0 = (ثابت int32_t *) (vtcm_پایه + chunk_elems * حجم(عائم));
                ثابت عائم *   val1 = (ثابت عائم *)   (vtcm_پایه + جایگاه_اندازه);
                ثابت int32_t * idx1 = (ثابت int32_t *) (vtcm_پایه + جایگاه_اندازه + chunk_elems * حجم(عائم));

                int32_t * final_idx = (int32_t *) (vtcm_پایه + kparams->ادغام_indices_off);
                بنية htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_شروع(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ادغام_runs(val0, idx0, cctx.real_شمار[0], val1, idx1, cctx.real_شمار[1], عدم, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->داده, final_idx, ne00 * حجم(int32_t));
            } وإلا {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_پایه + chunk_elems * حجم(عائم));
                dma_sync_write(octx->ctx->dma[0], dst->داده, final_idx, ne00 * حجم(int32_t));
            }
        }
        أرجع HTP_وضعیت_OK;
    }

    بنية htp_مرتب_سازی_multi_ردیف_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_پایه        = vtcm_پایه;

    uint32_t dst_ردیف_بایتها = (uint32_t) dst_ردیف_اندازه;
    uint32_t ردیفها_در_هر_chunk = 0;
    htp_tensor_mdev_ردیفها_در_هر_chunk(dst, حجم(int32_t), dst_ردیف_بایتها, &ردیفها_در_هر_chunk);
    إذا (ردیفها_در_هر_chunk == 0) {
        ردیفها_در_هر_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_نخها - 1) / (uint32_t) kparams->n_نخها;
    إذا (ردیفها_در_هر_chunk > 1) {
        dr = hex_دور_به_بالا(dr, ردیفها_در_هر_chunk);
    }
    mctx.nrows_در_هر_thread = dr;

    work_صف_پیاپی(octx->ctx->work_صف, htp_مرتب_سازی_multi_ردیف_کار, &mctx, (uint32_t) kparams->n_نخها);

    أرجع HTP_وضعیت_OK;
}

صحيح op_argsort(بنية htp_ops_context * octx) {
    أرجع op_مرتب_سازی_common(octx, false);
}

صحيح op_بالاترین_k(بنية htp_ops_context * octx) {
    أرجع op_مرتب_سازی_common(octx, true);
}
