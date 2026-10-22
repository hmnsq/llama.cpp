#ضمن <string.h>
#ضمن <stdlib.h>
#ضمن <math.h>
#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#تعریف_کړه GGML_COMMON_DECL_C
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

#که_نه_تعریف_شوی MIN
#تعریف_کړه MIN(a, b) ((a) < (b) ? (a) : (b))
#د_که_پای


ساكن فراغ quicksort_ارزښتونه_indices_asc(عائم * ارزښتونه, int32_t * indices, صحيح کیڼ_2, صحيح ښی_2) {
    طالما (کیڼ_2 < ښی_2) {
        عائم محور = ارزښتونه[کیڼ_2 + (ښی_2 - کیڼ_2) / 2];
        صحيح i = کیڼ_2;
        صحيح j = ښی_2;

        طالما (i <= j) {
            طالما (ارزښتونه[i] < محور) i++;
            طالما (ارزښتونه[j] > محور) j--;

            إذا (i <= j) {
                عائم tmp_val = ارزښتونه[i];
                ارزښتونه[i] = ارزښتونه[j];
                ارزښتونه[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        إذا (j - کیڼ_2 < ښی_2 - i) {
            إذا (کیڼ_2 < j) quicksort_ارزښتونه_indices_asc(ارزښتونه, indices, کیڼ_2, j);
            کیڼ_2 = i;
        } وإلا {
            إذا (i < ښی_2) quicksort_ارزښتونه_indices_asc(ارزښتونه, indices, i, ښی_2);
            ښی_2 = j;
        }
    }
}

ساكن فراغ quicksort_ارزښتونه_indices_desc(عائم * ارزښتونه, int32_t * indices, صحيح کیڼ_2, صحيح ښی_2) {
    طالما (کیڼ_2 < ښی_2) {
        عائم محور = ارزښتونه[کیڼ_2 + (ښی_2 - کیڼ_2) / 2];
        صحيح i = کیڼ_2;
        صحيح j = ښی_2;

        طالما (i <= j) {
            طالما (ارزښتونه[i] > محور) i++;
            طالما (ارزښتونه[j] < محور) j--;

            إذا (i <= j) {
                عائم tmp_val = ارزښتونه[i];
                ارزښتونه[i] = ارزښتونه[j];
                ارزښتونه[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        إذا (j - کیڼ_2 < ښی_2 - i) {
            إذا (کیڼ_2 < j) quicksort_ارزښتونه_indices_desc(ارزښتونه, indices, کیڼ_2, j);
            کیڼ_2 = i;
        } وإلا {
            إذا (i < ښی_2) quicksort_ارزښتونه_indices_desc(ارزښتونه, indices, i, ښی_2);
            ښی_2 = j;
        }
    }
}

ساكن مضمن uint32_t سر_k_max_ارزښت_شاخص(ثابت عائم * ارزښتونه, uint32_t n, عائم * ارزښت) {
    int32_t max_idx = 0;
    hvx_argmax_f32(ارزښتونه, n, 0, ارزښت, &max_idx);
    أرجع (uint32_t) max_idx;
}

ساكن ثابت int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

ساكن مضمن فراغ init_indices_ramp(int32_t * indices_buf, uint32_t شمېر, uint32_t بنسټ) {
    ثابت HVX_ویکټور ind_init_vec = Q6_Vw_vadd_VwVw(*(ثابت HVX_ویکټور *) argsort_ramp_lut, Q6_V_vsplat_R(بنسټ));
    ثابت HVX_ویکټور ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_ویکټور * indices_buf_vec = (HVX_ویکټور *) indices_buf;
    uint32_t عدد_vecs = شمېر / 32;
    HVX_ویکټور curr_ind_vec = ind_init_vec;
    لكل (uint32_t j = 0; j < عدد_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = شمېر % 32;
    إذا (rem > 0) {
        hvx_vec_store_a((فراغ *) &indices_buf_vec[عدد_vecs], rem * حجم(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
ساكن مضمن فراغ vec_cas(HVX_ویکټور * X_val, HVX_ویکټور * X_idx, HVX_ویکټور * Y_val, HVX_ویکټور * Y_idx, bool asc) {
    HVX_ویکټور_Pred pred = asc ? Q6_Q_vcmp_gt_VsfVsf(*X_val, *Y_val)
                              : Q6_Q_vcmp_gt_VsfVsf(*Y_val, *X_val);
    HVX_ویکټور بل_X_val = Q6_V_vmux_QVV(pred, *Y_val, *X_val);
    HVX_ویکټور بل_Y_val = Q6_V_vmux_QVV(pred, *X_val, *Y_val);
    HVX_ویکټور بل_X_idx = Q6_V_vmux_QVV(pred, *Y_idx, *X_idx);
    HVX_ویکټور Y_tmp_idx  = Q6_V_vmux_QVV(pred, *X_idx, *Y_idx);
    *X_val = بل_X_val;
    *Y_val = بل_Y_val;
    *X_idx = بل_X_idx;
    *Y_idx = Y_tmp_idx;
}

__attribute__((always_inline))
ساكن مضمن فراغ bitonic_cas_32(HVX_ویکټور * V, HVX_ویکټور * I, صحيح d, HVX_ویکټور_Pred dir_ماسک, HVX_ویکټور idx_vec, HVX_ویکټور zero_vec) {
    HVX_ویکټور_Pred ماسک_کیڼ;
    HVX_ویکټور V_rot_کیڼ, V_rot_ښی;
    HVX_ویکټور I_rot_کیڼ, I_rot_ښی;

    إذا (d == 1) {
        ماسک_کیڼ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_کیڼ = Q6_V_vror_VR(*V, 4);
        V_rot_ښی = Q6_V_vror_VR(*V, 124);
        I_rot_کیڼ = Q6_V_vror_VR(*I, 4);
        I_rot_ښی = Q6_V_vror_VR(*I, 124);
    } وإلا إذا (d == 2) {
        ماسک_کیڼ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_کیڼ = Q6_V_vror_VR(*V, 8);
        V_rot_ښی = Q6_V_vror_VR(*V, 120);
        I_rot_کیڼ = Q6_V_vror_VR(*I, 8);
        I_rot_ښی = Q6_V_vror_VR(*I, 120);
    } وإلا إذا (d == 4) {
        ماسک_کیڼ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_کیڼ = Q6_V_vror_VR(*V, 16);
        V_rot_ښی = Q6_V_vror_VR(*V, 112);
        I_rot_کیڼ = Q6_V_vror_VR(*I, 16);
        I_rot_ښی = Q6_V_vror_VR(*I, 112);
    } وإلا إذا (d == 8) {
        ماسک_کیڼ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_کیڼ = Q6_V_vror_VR(*V, 32);
        V_rot_ښی = Q6_V_vror_VR(*V, 96);
        I_rot_کیڼ = Q6_V_vror_VR(*I, 32);
        I_rot_ښی = Q6_V_vror_VR(*I, 96);
    } وإلا {
        ماسک_کیڼ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_کیڼ = Q6_V_vror_VR(*V, 64);
        V_rot_ښی = Q6_V_vror_VR(*V, 64);
        I_rot_کیڼ = Q6_V_vror_VR(*I, 64);
        I_rot_ښی = Q6_V_vror_VR(*I, 64);
    }

    HVX_ویکټور V_paired = Q6_V_vmux_QVV(ماسک_کیڼ, V_rot_کیڼ, V_rot_ښی);
    HVX_ویکټور I_paired = Q6_V_vmux_QVV(ماسک_کیڼ, I_rot_کیڼ, I_rot_ښی);

    HVX_ویکټور_Pred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_ویکټور_Pred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_ویکټور_Pred ماسک_ښی = Q6_Q_not_Q(ماسک_کیڼ);
    HVX_ویکټور_Pred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(ماسک_کیڼ, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, ماسک_ښی)
    );
    HVX_ویکټور_Pred Q_بدلون = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_ماسک, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_ماسک), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_بدلون, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_بدلون, I_paired, *I);
}

__attribute__((always_inline))
ساكن مضمن فراغ bitonic_ترتیبول_generic_hvx(uint8_t * ارزښتونه, uint8_t * indices, صحيح K, bool asc_ترتیب) {
    HVX_ویکټور V[32];
    HVX_ویکټور I[32];

    HVX_ویکټور zero_vec = Q6_V_vzero();
    HVX_ویکټور idx_vec = *(ثابت HVX_ویکټور *) argsort_ramp_lut;

    لكل (صحيح v = 0; v < K; v++) {
        V[v] = *(HVX_ویکټور *)(ارزښتونه + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_ویکټور_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_ویکټور_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

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
                    HVX_ویکټور_Pred dir_ماسک = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    لكل (صحيح v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_ماسک, idx_vec, zero_vec);
                    }
                } وإلا {
                    لكل (صحيح v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_ترتیب;
                        HVX_ویکټور_Pred dir_ماسک = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_ماسک, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    لكل (صحيح v = 0; v < K; v++) {
        *(HVX_ویکټور *)(ارزښتونه + v * 128)  = V[v];
        *(HVX_ویکټور *)(indices + v * 128) = I[v];
    }
}

ساكن فراغ bitonic_ترتیبول_vtcm_desc(uint8_t * ارزښتونه, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_ویکټور zero_vec = Q6_V_vzero();
    HVX_ویکټور idx_vec = *(ثابت HVX_ویکټور *) argsort_ramp_lut;

    HVX_ویکټور_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_ویکټور_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    إذا (init_indices) {
        لكل (uint32_t v = 0; v < n_vec; v++) {
            HVX_ویکټور idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_ویکټور *)(indices + v * 128) = idx;
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

                        HVX_ویکټور Vv1 = *(HVX_ویکټور *)(ارزښتونه + v1 * 128);
                        HVX_ویکټور Iv1 = *(HVX_ویکټور *)(indices + v1 * 128);
                        HVX_ویکټور Vv2 = *(HVX_ویکټور *)(ارزښتونه + v2 * 128);
                        HVX_ویکټور Iv2 = *(HVX_ویکټور *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_ویکټور *)(ارزښتونه + v1 * 128)  = Vv1;
                        *(HVX_ویکټور *)(indices + v1 * 128) = Iv1;
                        *(HVX_ویکټور *)(ارزښتونه + v2 * 128)  = Vv2;
                        *(HVX_ویکټور *)(indices + v2 * 128) = Iv2;
                    }
                }
            } وإلا {
                إذا (s < 5) {
                    HVX_ویکټور_Pred dir_ماسک = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    لكل (uint32_t v = 0; v < n_vec; v++) {
                        HVX_ویکټور Vv = *(HVX_ویکټور *)(ارزښتونه + v * 128);
                        HVX_ویکټور Iv = *(HVX_ویکټور *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_ماسک, idx_vec, zero_vec);

                        *(HVX_ویکټور *)(ارزښتونه + v * 128)  = Vv;
                        *(HVX_ویکټور *)(indices + v * 128) = Iv;
                    }
                } وإلا {
                    لكل (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_ویکټور_Pred dir_ماسک = asc ? pred_all_1s : pred_all_0s;

                        HVX_ویکټور Vv = *(HVX_ویکټور *)(ارزښتونه + v * 128);
                        HVX_ویکټور Iv = *(HVX_ویکټور *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_ماسک, idx_vec, zero_vec);

                        *(HVX_ویکټور *)(ارزښتونه + v * 128)  = Vv;
                        *(HVX_ویکټور *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

ساكن فراغ سر_k_select_tiled(dma_کتار * dma_q, بنية htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               عائم * ارزښتونه_buf, int32_t * indices_buf,
                               عائم * بهر_ارزښتونه, int32_t * بهر_indices,
                               uint32_t chunk_بنسټ) {
    ثابت uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_شمېر    = n_tiles * k;
    uint32_t یوځای_n_vec        = hmx_ceil_div(candidate_شمېر, 32);
    uint32_t یوځای_n_vec_pow2   = 1;
    طالما (یوځای_n_vec_pow2 < یوځای_n_vec) یوځای_n_vec_pow2 <<= 1;
    uint32_t یوځای_elems        = یوځای_n_vec_pow2 * 32;
    عائم * candidate_ارزښتونه    = ارزښتونه_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    لكل (uint32_t دننه_واټن = 0; دننه_واټن < n; دننه_واټن += tile_elems) {
        uint32_t tile_شمېر = MIN(tile_elems, n - دننه_واټن);
        dma_sync_read(dma_q, ارزښتونه_buf, src_addr + (size_t) دننه_واټن * حجم(عائم), tile_شمېر * حجم(عائم));

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) دننه_واټن);
        إذا (tile_شمېر < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (ارزښتونه_buf + tile_شمېر), -INFINITY, tile_elems - tile_شمېر);
        }

        bitonic_ترتیبول_vtcm_desc((uint8_t *) ارزښتونه_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_شمېر);
        لكل (uint32_t j = 0; j < tile_k; j++) {
            candidate_ارزښتونه[candidate_pos] = ارزښتونه_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) دننه_واټن;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) دننه_واټن);
    }

    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    إذا (یوځای_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_ارزښتونه + candidate_pos), -INFINITY, یوځای_elems - candidate_pos);
        لكل (uint32_t j = candidate_pos; j < یوځای_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_ترتیبول_vtcm_desc((uint8_t *) candidate_ارزښتونه, (uint8_t *) candidate_indices, یوځای_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    لكل (uint32_t j = 0; j < k; j++) {
        بهر_ارزښتونه[j]  = candidate_ارزښتونه[j];
        بهر_indices[j] = candidate_indices[j] + (int32_t) chunk_بنسټ;
    }
}

__attribute__((always_inline))
ساكن مضمن فراغ sort32_f32_hvx(uint8_t * ارزښتونه, uint8_t * indices, تعداد ggml_ترتیبول_ترتیب ترتیب) {
    bitonic_ترتیبول_generic_hvx(ارزښتونه, indices, 1, ترتیب == GGML_ترتیبول_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort64_f32_hvx(uint8_t * ارزښتونه, uint8_t * indices, تعداد ggml_ترتیبول_ترتیب ترتیب) {
    bitonic_ترتیبول_generic_hvx(ارزښتونه, indices, 2, ترتیب == GGML_ترتیبول_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort128_f32_hvx(uint8_t * ارزښتونه, uint8_t * indices, تعداد ggml_ترتیبول_ترتیب ترتیب) {
    bitonic_ترتیبول_generic_hvx(ارزښتونه, indices, 4, ترتیب == GGML_ترتیبول_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort256_f32_hvx(uint8_t * ارزښتونه, uint8_t * indices, تعداد ggml_ترتیبول_ترتیب ترتیب) {
    bitonic_ترتیبول_generic_hvx(ارزښتونه, indices, 8, ترتیب == GGML_ترتیبول_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort512_f32_hvx(uint8_t * ارزښتونه, uint8_t * indices, تعداد ggml_ترتیبول_ترتیب ترتیب) {
    bitonic_ترتیبول_generic_hvx(ارزښتونه, indices, 16, ترتیب == GGML_ترتیبول_ترتیب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort1024_f32_hvx(uint8_t * ارزښتونه, uint8_t * indices, تعداد ggml_ترتیبول_ترتیب ترتیب) {
    bitonic_ترتیبول_generic_hvx(ارزښتونه, indices, 32, ترتیب == GGML_ترتیبول_ترتیب_ASC);
}

ساكن فراغ یوځای_runs(
    ثابت عائم * محدود دننه_val0, ثابت int32_t * محدود دننه_idx0, uint32_t n0,
    ثابت عائم * محدود دننه_val1, ثابت int32_t * محدود دننه_idx1, uint32_t n1,
    عائم * محدود بهر_val, int32_t * محدود بهر_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t بهر = 0;

    إذا (asc) {
        طالما (i0 < n0 && i1 < n1) {
            إذا (دننه_val0[i0] <= دننه_val1[i1]) {
                إذا (بهر_val) بهر_val[بهر] = دننه_val0[i0];
                بهر_idx[بهر] = دننه_idx0[i0];
                i0++;
            } وإلا {
                إذا (بهر_val) بهر_val[بهر] = دننه_val1[i1];
                بهر_idx[بهر] = دننه_idx1[i1];
                i1++;
            }
            بهر++;
        }
    } وإلا {
        طالما (i0 < n0 && i1 < n1) {
            إذا (دننه_val0[i0] >= دننه_val1[i1]) {
                إذا (بهر_val) بهر_val[بهر] = دننه_val0[i0];
                بهر_idx[بهر] = دننه_idx0[i0];
                i0++;
            } وإلا {
                إذا (بهر_val) بهر_val[بهر] = دننه_val1[i1];
                بهر_idx[بهر] = دننه_idx1[i1];
                i1++;
            }
            بهر++;
        }
    }

    طالما (i0 < n0) {
        إذا (بهر_val) بهر_val[بهر] = دننه_val0[i0];
        بهر_idx[بهر] = دننه_idx0[i0];
        i0++;
        بهر++;
    }

    طالما (i1 < n1) {
        إذا (بهر_val) بهر_val[بهر] = دننه_val1[i1];
        بهر_idx[بهر] = دننه_idx1[i1];
        i1++;
        بهر++;
    }
}

بنية htp_ترتیبول_chunk_ctx {
    بنية htp_ops_context *              octx;
    ثابت بنية htp_ترتیبول_kernel_params * kparams;
    uint8_t *                             vtcm_بنسټ;
    uint32_t                              real_شمېر[HTP_MAX_NTHREADS];
};

ساكن فراغ htp_ترتیبول_chunk_دنده(بینښه صحيح n, بینښه صحيح i, فراغ * معلومات) {
    بنية htp_ترتیبول_chunk_ctx * cctx = (بنية htp_ترتیبول_chunk_ctx *) معلومات;
    بنية htp_ops_context * octx = cctx->octx;
    ثابت بنية htp_ترتیبول_kernel_params * kparams = cctx->kparams;
    ثابت بنية htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_بنسټ  = i * chunk_elems;

    uint32_t real_شمېر = (chunk_بنسټ < ne00) ? MIN(chunk_elems, ne00 - chunk_بنسټ) : 0;
    cctx->real_شمېر[i] = real_شمېر;

    uint8_t * spad = cctx->vtcm_بنسټ + (size_t) kparams->phase1_خونه_اندازه * i;
    عائم *   ارزښتونه_buf  = (عائم *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * حجم(عائم));

    عائم *   یوځای_ارزښتونه  = (عائم *)   (cctx->vtcm_بنسټ + kparams->یوځای_ارزښتونه_off);
    int32_t * یوځای_indices = (int32_t *) (cctx->vtcm_بنسټ + kparams->یوځای_indices_off);

    بنية htp_thread_trace * tr = &octx->ctx->trace[i];
    ثابت dma_addr_t src_addr = src0->معلومات + (size_t) chunk_بنسټ * حجم(عائم);

    إذا (kparams->is_سر_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        إذا (real_شمېر == 0) {
            لكل (uint32_t j = 0; j < local_k; j++) {
                یوځای_ارزښتونه[i * local_k + j]  = -INFINITY;
                یوځای_indices[i * local_k + j] = 0;
            }
            أرجع;
        }

        إذا (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], ارزښتونه_buf, src_addr, real_شمېر * حجم(عائم));

            htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            عائم max_ارزښت;
            uint32_t max_شاخص = سر_k_max_ارزښت_شاخص(ارزښتونه_buf, real_شمېر, &max_ارزښت);
            یوځای_ارزښتونه[i] = max_ارزښت;
            یوځای_indices[i] = (int32_t) (max_شاخص + chunk_بنسټ);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            أرجع;
        }

        إذا (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            سر_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_شمېر, local_k,
                               ارزښتونه_buf, indices_buf,
                               &یوځای_ارزښتونه[i * local_k], &یوځای_indices[i * local_k],
                               chunk_بنسټ);
            أرجع;
        }

        dma_sync_read(octx->ctx->dma[i], ارزښتونه_buf, src_addr, real_شمېر * حجم(عائم));

        htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        إذا (chunk_elems > real_شمېر) {
            hvx_splat_f32_u((uint8_t *)(ارزښتونه_buf + real_شمېر), -INFINITY, chunk_elems - real_شمېر);
        }

        bitonic_ترتیبول_vtcm_desc((uint8_t *) ارزښتونه_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        لكل (uint32_t j = 0; j < local_k; j++) {
            یوځای_ارزښتونه[i * local_k + j]  = ارزښتونه_buf[j];
            یوځای_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_بنسټ;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } وإلا {
        إذا (real_شمېر > 0) {
            dma_sync_read(octx->ctx->dma[i], ارزښتونه_buf, src_addr, real_شمېر * حجم(عائم));

            htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_شمېر, chunk_بنسټ);

            إذا (kparams->ترتیب == GGML_ترتیبول_ترتیب_ASC) {
                quicksort_ارزښتونه_indices_asc(ارزښتونه_buf, indices_buf, 0, real_شمېر - 1);
            } وإلا {
                quicksort_ارزښتونه_indices_desc(ارزښتونه_buf, indices_buf, 0, real_شمېر - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

بنية htp_argsort_یوځای_l1_ctx {
    بنية htp_ترتیبول_chunk_ctx * cctx;
    bool                        asc;
};

ساكن فراغ htp_argsort_یوځای_l1_دنده(بینښه صحيح n, بینښه صحيح i, فراغ * معلومات) {
    بنية htp_argsort_یوځای_l1_ctx * mctx = (بنية htp_argsort_یوځای_l1_ctx *) معلومات;
    بنية htp_ترتیبول_chunk_ctx * cctx = mctx->cctx;
    ثابت بنية htp_ترتیبول_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_شمېر[c0];
    uint32_t count1 = cctx->real_شمېر[c1];

    size_t خونه_اندازه = (size_t) kparams->phase1_خونه_اندازه;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    ثابت عائم *   val0 = (ثابت عائم *)   (cctx->vtcm_بنسټ + خونه_اندازه * c0);
    ثابت int32_t * idx0 = (ثابت int32_t *) (cctx->vtcm_بنسټ + خونه_اندازه * c0 + chunk_elems * حجم(عائم));
    ثابت عائم *   val1 = (ثابت عائم *)   (cctx->vtcm_بنسټ + خونه_اندازه * c1);
    ثابت int32_t * idx1 = (ثابت int32_t *) (cctx->vtcm_بنسټ + خونه_اندازه * c1 + chunk_elems * حجم(عائم));

    uint32_t بهر_دننه_واټن = (2 * i) * chunk_elems;
    عائم *   بهر_val = (عائم *)   (cctx->vtcm_بنسټ + kparams->یوځای_ارزښتونه_off)  + بهر_دننه_واټن;
    int32_t * بهر_idx = (int32_t *) (cctx->vtcm_بنسټ + kparams->یوځای_indices_off) + بهر_دننه_واټن;

    بنية htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    یوځای_runs(val0, idx0, count0, val1, idx1, count1, بهر_val, بهر_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

بنية htp_argsort_یوځای_l2_ctx {
    بنية htp_ترتیبول_chunk_ctx * cctx;
    bool                        asc;
};

ساكن فراغ htp_argsort_یوځای_l2_دنده(بینښه صحيح n, بینښه صحيح i, فراغ * معلومات) {
    بنية htp_argsort_یوځای_l2_ctx * mctx = (بنية htp_argsort_یوځای_l2_ctx *) معلومات;
    بنية htp_ترتیبول_chunk_ctx * cctx = mctx->cctx;
    ثابت بنية htp_ترتیبول_kernel_params * kparams = cctx->kparams;

    size_t خونه_اندازه = (size_t) kparams->phase1_خونه_اندازه;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    ثابت عائم *   val0 = (ثابت عائم *)   (cctx->vtcm_بنسټ + kparams->یوځای_ارزښتونه_off)  + off0;
    ثابت int32_t * idx0 = (ثابت int32_t *) (cctx->vtcm_بنسټ + kparams->یوځای_indices_off) + off0;
    ثابت عائم *   val1 = (ثابت عائم *)   (cctx->vtcm_بنسټ + kparams->یوځای_ارزښتونه_off)  + off1;
    ثابت int32_t * idx1 = (ثابت int32_t *) (cctx->vtcm_بنسټ + kparams->یوځای_indices_off) + off1;

    uint32_t count0 = cctx->real_شمېر[4*i + 0] + cctx->real_شمېر[4*i + 1];
    uint32_t count1 = cctx->real_شمېر[4*i + 2] + cctx->real_شمېر[4*i + 3];

    uint8_t * خونه_بنسټ = cctx->vtcm_بنسټ + (4 * i) * خونه_اندازه;
    عائم *   بهر_val   = (عائم *) خونه_بنسټ;
    int32_t * بهر_idx   = (int32_t *) (خونه_بنسټ + 4 * chunk_elems * حجم(عائم));

    بنية htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    یوځای_runs(val0, idx0, count0, val1, idx1, count1, بهر_val, بهر_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

بنية htp_ترتیبول_multi_لیکه_ctx {
    بنية htp_ops_context *              octx;
    ثابت بنية htp_ترتیبول_kernel_params * kparams;
    uint8_t *                             vtcm_بنسټ;
    uint32_t                              nrows_فی_thread;
};

بنية htp_tensor_لیکه_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

ساكن مضمن بنية htp_tensor_لیکه_coord init_لیکه_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    أرجع (بنية htp_tensor_لیکه_coord){ i01, i02, i03 };
}

ساكن مضمن فراغ ګام_لیکه_coord(بنية htp_tensor_لیکه_coord * c, uint32_t ne01, uint32_t ne02) {
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

ساكن مضمن size_t coord_دننه_واټن(بنية htp_tensor_لیکه_coord c, size_t nb1, size_t nb2, size_t nb3) {
    أرجع (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

ساكن مضمن فراغ compute_لیکه_ترتیبول(
    عائم * cur_ارزښتونه,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_سر_k,
    تعداد ggml_ترتیبول_ترتیب ترتیب
) {
    إذا (is_سر_k) {
        إذا (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            بدل (ne00) {
                حالة 32:   sort32_f32_hvx((uint8_t *) cur_ارزښتونه, (uint8_t *) cur_indices, GGML_ترتیبول_ترتیب_DESC); اكسر;
                حالة 64:   sort64_f32_hvx((uint8_t *) cur_ارزښتونه, (uint8_t *) cur_indices, GGML_ترتیبول_ترتیب_DESC); اكسر;
                حالة 128:  sort128_f32_hvx((uint8_t *) cur_ارزښتونه, (uint8_t *) cur_indices, GGML_ترتیبول_ترتیب_DESC); اكسر;
                حالة 256:  sort256_f32_hvx((uint8_t *) cur_ارزښتونه, (uint8_t *) cur_indices, GGML_ترتیبول_ترتیب_DESC); اكسر;
                حالة 512:  sort512_f32_hvx((uint8_t *) cur_ارزښتونه, (uint8_t *) cur_indices, GGML_ترتیبول_ترتیب_DESC); اكسر;
                حالة 1024: sort1024_f32_hvx((uint8_t *) cur_ارزښتونه, (uint8_t *) cur_indices, GGML_ترتیبول_ترتیب_DESC); اكسر;
            }
        } وإلا {
            إذا (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_ارزښتونه + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_ترتیبول_vtcm_desc((uint8_t *) cur_ارزښتونه, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } وإلا {
        إذا (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            بدل (ne00) {
                حالة 32:   sort32_f32_hvx((uint8_t *) cur_ارزښتونه, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 64:   sort64_f32_hvx((uint8_t *) cur_ارزښتونه, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 128:  sort128_f32_hvx((uint8_t *) cur_ارزښتونه, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 256:  sort256_f32_hvx((uint8_t *) cur_ارزښتونه, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 512:  sort512_f32_hvx((uint8_t *) cur_ارزښتونه, (uint8_t *) cur_indices, ترتیب); اكسر;
                حالة 1024: sort1024_f32_hvx((uint8_t *) cur_ارزښتونه, (uint8_t *) cur_indices, ترتیب); اكسر;
            }
        } وإلا {
            init_indices_ramp(cur_indices, ne00, 0);
            إذا (ترتیب == GGML_ترتیبول_ترتیب_ASC) {
                quicksort_ارزښتونه_indices_asc(cur_ارزښتونه, cur_indices, 0, ne00 - 1);
            } وإلا {
                quicksort_ارزښتونه_indices_desc(cur_ارزښتونه, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

ساكن فراغ htp_ترتیبول_multi_لیکه_دنده(بینښه صحيح n, بینښه صحيح ith, فراغ * معلومات) {
    بنية htp_ترتیبول_multi_لیکه_ctx * mctx = (بنية htp_ترتیبول_multi_لیکه_ctx *) معلومات;
    بنية htp_ops_context * octx = mctx->octx;
    ثابت بنية htp_ترتیبول_kernel_params * kparams = mctx->kparams;
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

    uint32_t لیکه_پیل = (uint32_t) kparams->لیکه_پیل;
    uint32_t لیکه_پای_ټکی   = (uint32_t) kparams->لیکه_پای_ټکی;
    uint32_t r_پیل   = لیکه_پیل + ith * mctx->nrows_فی_thread;
    uint32_t r_پای_ټکی     = MIN(r_پیل + mctx->nrows_فی_thread, لیکه_پای_ټکی);

    إذا (r_پیل >= r_پای_ټکی) أرجع;

    size_t spad_دننه_واټن = ith * (size_t) (kparams->phase1_خونه_اندازه * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_بنسټ + spad_دننه_واټن;
    size_t خونه_اندازه = (size_t) kparams->phase1_خونه_اندازه;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    عائم *   ارزښتونه_buf  = (عائم *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * حجم(عائم));

    dma_کتار * dma_q = octx->ctx->dma[ith];
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_سر_k = kparams->is_سر_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    تعداد ggml_ترتیبول_ترتیب ترتیب = (تعداد ggml_ترتیبول_ترتیب) kparams->ترتیب;

    ثابت uint32_t src_بایټونه = ne00 * حجم(عائم);
    ثابت uint32_t dst_بایټونه = (is_سر_k ? k : ne00) * حجم(int32_t);

    إذا (kparams->n_slots == 2) {
        بنية htp_tensor_لیکه_coord اولي_coord = init_لیکه_coord(r_پیل, ne01, ne02);
        لكل (uint32_t r = r_پیل, spad_idx = 0; r < r_پای_ټکی && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * خونه_اندازه;
            عائم *   cur_ارزښتونه  = (عائم *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * حجم(عائم));

            dma_addr_t dst_addr = dst->معلومات + coord_دننه_واټن(اولي_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->معلومات + coord_دننه_واټن(اولي_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_addr, cur_indices),
                           dst_بایټونه, dst_بایټونه, dst_بایټونه, 0);

            // Prefetch input row
            dma_کتار_ټېل(dma_q, dma_make_معلومات(cur_ارزښتونه, src_addr),
                           src_بایټونه, src_بایټونه, src_بایټونه, 1);

            ګام_لیکه_coord(&اولي_coord, ne01, ne02);
        }

        بنية htp_tensor_لیکه_coord dst_coord = init_لیکه_coord(r_پیل, ne01, ne02);
        بنية htp_tensor_لیکه_coord بل_src_coord = init_لیکه_coord(r_پیل + 2, ne01, ne02);

        لكل (uint32_t r = r_پیل; r < r_پای_ټکی; r++) {
            uint32_t cur_خونه = (r - r_پیل) & 1;
            uint8_t * cur_spad = thread_spad + cur_خونه * خونه_اندازه;
            عائم *   cur_ارزښتونه  = (عائم *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * حجم(عائم));

            // Wait for previous writeback of this slot to complete
            dma_کتار_راوباسه(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_کتار_راوباسه(dma_q);

            htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_لیکه_ترتیبول(cur_ارزښتونه, cur_indices, ne00, chunk_elems, is_سر_k, ترتیب);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->معلومات + coord_دننه_واټن(dst_coord, nb1, nb2, nb3);
            dma_کتار_ټېل(dma_q, dma_make_معلومات(dst_addr, cur_indices),
                           dst_بایټونه, dst_بایټونه, dst_بایټونه, 1);
            ګام_لیکه_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            ثابت uint32_t بل_لیکه = r + 2;
            إذا (بل_لیکه < r_پای_ټکی) {
                dma_addr_t بل_src_addr = src0->معلومات + coord_دننه_واټن(بل_src_coord, nb01, nb02, nb03);
                dma_کتار_ټېل(dma_q, dma_make_معلومات(cur_ارزښتونه, بل_src_addr),
                               src_بایټونه, src_بایټونه, src_بایټونه, 1);
                ګام_لیکه_coord(&بل_src_coord, ne01, ne02);
            }
        }

        dma_کتار_flush(dma_q);
    } وإلا {
        بنية htp_tensor_لیکه_coord coord = init_لیکه_coord(r_پیل, ne01, ne02);
        لكل (uint32_t r = r_پیل; r < r_پای_ټکی; r++) {
            dma_addr_t src_addr = src0->معلومات + coord_دننه_واټن(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->معلومات + coord_دننه_واټن(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, ارزښتونه_buf, src_addr, src_بایټونه);

            htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_لیکه_ترتیبول(ارزښتونه_buf, indices_buf, ne00, chunk_elems, is_سر_k, ترتیب);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_بایټونه);
            ګام_لیکه_coord(&coord, ne01, ne02);
        }
    }
}

ساكن صحيح op_ترتیبول_common(بنية htp_ops_context * octx, bool is_سر_k) {
    إذا (octx->src[0]->type != HTP_TYPE_F32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t ټول_لیکې = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_لیکه_اندازه = dst->ne[0] * حجم(int32_t);

    uint32_t لیکه_پیل = 0;
    uint32_t لیکه_پای_ټکی   = ټول_لیکې;
    إذا (octx->ctx->mdev.شمېر > 1) {
        uint32_t لیکې_فی_chunk = 0;
        htp_tensor_mdev_لیکې_فی_chunk(dst, حجم(int32_t), (uint32_t) dst_لیکه_اندازه, &لیکې_فی_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_برخه_کول(
            ټول_لیکې, لیکې_فی_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.شمېر, &octx->ctx->mdev.شمېر_div);
        لیکه_پیل = range.پیل;
        لیکه_پای_ټکی   = range.پیل + range.شمېر;
    }

    ثابت uint32_t nrows = لیکه_پای_ټکی - لیکه_پیل;
    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    بنية htp_ترتیبول_kernel_params kparams_local;
    ثابت بنية htp_ترتیبول_kernel_params * kparams = (ثابت بنية htp_ترتیبول_kernel_params *) octx->kernel_params;

    إذا (kparams->n_تارونه == 0) {
        بنية htp_ترتیبول_vtcm_layout layout;
        إذا (!htp_ترتیبول_solve_layout(&layout, ne00, ټول_لیکې, k, octx->n_تارونه, octx->ctx->vtcm_اندازه, is_سر_k)) {
            أرجع HTP_حالت_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, حجم(kparams_local));
        kparams_local.n_تارونه         = (int32_t) layout.n_تارونه;
        kparams_local.ټول_لیکې        = (int32_t) ټول_لیکې;
        kparams_local.لیکه_پیل         = (int32_t) لیکه_پیل;
        kparams_local.لیکه_پای_ټکی           = (int32_t) لیکه_پای_ټکی;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.ترتیب             = is_سر_k ? GGML_ترتیبول_ترتیب_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_سر_k          = is_سر_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_اندازه         = (int32_t) layout.ټول_بایټونه;
        kparams_local.phase1_خونه_اندازه  = (int32_t) layout.phase1_خونه_اندازه;
        kparams_local.یوځای_ارزښتونه_off  = (int32_t) layout.یوځای_ارزښتونه_off;
        kparams_local.یوځای_indices_off = (int32_t) layout.یوځای_indices_off;
        kparams_local.یوځای_elems       = (int32_t) layout.یوځای_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } وإلا إذا (octx->ctx->mdev.شمېر > 1) {
        حافظه_نقل(&kparams_local, kparams, حجم(kparams_local));
        kparams_local.لیکه_پیل = (int32_t) لیکه_پیل;
        kparams_local.لیکه_پای_ټکی   = (int32_t) لیکه_پای_ټکی;
        kparams = &kparams_local;
    }

    إذا (octx->ctx->vtcm_اندازه < (size_t) kparams->vtcm_اندازه) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_اندازه, octx->ctx->vtcm_اندازه);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_بنسټ = (uint8_t *) octx->ctx->vtcm_بنسټ;

    إذا (ټول_لیکې == 1 && kparams->n_chunks > 1) {
        بنية htp_ترتیبول_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_بنسټ = vtcm_بنسټ;
        memset(cctx.real_شمېر, 0, حجم(cctx.real_شمېر));

        work_کتار_پرلپسې(octx->ctx->work_کتار, htp_ترتیبول_chunk_دنده, &cctx, (uint32_t) kparams->n_chunks);

        إذا (is_سر_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            إذا (k == 1 && ne00 >= 128*1024) {
                عائم *   cand_vals = (عائم *)   (vtcm_بنسټ + kparams->یوځای_ارزښتونه_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_بنسټ + kparams->یوځای_indices_off);
                عائم غوره_val = cand_vals[0];
                int32_t غوره_idx = cand_idxs[0];
                لكل (int32_t c = 1; c < kparams->n_chunks; c++) {
                    إذا (cand_vals[c] > غوره_val) {
                        غوره_val = cand_vals[c];
                        غوره_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->معلومات;
                dst_ptr[0] = غوره_idx;
                أرجع HTP_حالت_OK;
            }

            uint32_t ټول_candidates = (uint32_t) kparams->n_chunks * local_k;
            عائم *   cand_vals = (عائم *)   (vtcm_بنسټ + kparams->یوځای_ارزښتونه_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_بنسټ + kparams->یوځای_indices_off);
            uint32_t یوځای_elems = (uint32_t) kparams->یوځای_elems;

            إذا (یوځای_elems > ټول_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + ټول_candidates), -INFINITY, یوځای_elems - ټول_candidates);
                لكل (uint32_t j = ټول_candidates; j < یوځای_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            بنية htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_ترتیبول_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, یوځای_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->معلومات, cand_idxs, k * حجم(int32_t));
        } وإلا {
            bool asc = kparams->ترتیب == GGML_ترتیبول_ترتیب_ASC;
            إذا (kparams->n_chunks == 8) {
                بنية htp_argsort_یوځای_l1_ctx l1ctx = { &cctx, asc };
                work_کتار_پرلپسې(octx->ctx->work_کتار, htp_argsort_یوځای_l1_دنده, &l1ctx, 4);

                بنية htp_argsort_یوځای_l2_ctx l2ctx = { &cctx, asc };
                work_کتار_پرلپسې(octx->ctx->work_کتار, htp_argsort_یوځای_l2_دنده, &l2ctx, 2);

                size_t خونه_اندازه = (size_t) kparams->phase1_خونه_اندازه;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ثابت عائم *   val0 = (ثابت عائم *) vtcm_بنسټ;
                ثابت int32_t * idx0 = (ثابت int32_t *) (vtcm_بنسټ + 4 * chunk_elems * حجم(عائم));
                ثابت عائم *   val1 = (ثابت عائم *) (vtcm_بنسټ + 4 * خونه_اندازه);
                ثابت int32_t * idx1 = (ثابت int32_t *) (vtcm_بنسټ + 4 * خونه_اندازه + 4 * chunk_elems * حجم(عائم));

                uint32_t n01 = cctx.real_شمېر[0] + cctx.real_شمېر[1] + cctx.real_شمېر[2] + cctx.real_شمېر[3];
                uint32_t n23 = cctx.real_شمېر[4] + cctx.real_شمېر[5] + cctx.real_شمېر[6] + cctx.real_شمېر[7];

                int32_t * final_idx = (int32_t *) (vtcm_بنسټ + kparams->یوځای_indices_off);
                بنية htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                یوځای_runs(val0, idx0, n01, val1, idx1, n23, عدم, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->معلومات, final_idx, ne00 * حجم(int32_t));
            } وإلا إذا (kparams->n_chunks == 4) {
                بنية htp_argsort_یوځای_l1_ctx l1ctx = { &cctx, asc };
                work_کتار_پرلپسې(octx->ctx->work_کتار, htp_argsort_یوځای_l1_دنده, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_شمېر[0] + cctx.real_شمېر[1];
                uint32_t n23 = cctx.real_شمېر[2] + cctx.real_شمېر[3];

                ثابت عائم *   val01 = (ثابت عائم *)   (vtcm_بنسټ + kparams->یوځای_ارزښتونه_off);
                ثابت int32_t * idx01 = (ثابت int32_t *) (vtcm_بنسټ + kparams->یوځای_indices_off);
                ثابت عائم *   val23 = (ثابت عائم *)   (vtcm_بنسټ + kparams->یوځای_ارزښتونه_off)  + 2 * chunk_elems;
                ثابت int32_t * idx23 = (ثابت int32_t *) (vtcm_بنسټ + kparams->یوځای_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_بنسټ;
                بنية htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                یوځای_runs(val01, idx01, n01, val23, idx23, n23, عدم, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->معلومات, final_idx, ne00 * حجم(int32_t));
            } وإلا إذا (kparams->n_chunks == 2) {
                size_t خونه_اندازه = (size_t) kparams->phase1_خونه_اندازه;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ثابت عائم *   val0 = (ثابت عائم *)   (vtcm_بنسټ);
                ثابت int32_t * idx0 = (ثابت int32_t *) (vtcm_بنسټ + chunk_elems * حجم(عائم));
                ثابت عائم *   val1 = (ثابت عائم *)   (vtcm_بنسټ + خونه_اندازه);
                ثابت int32_t * idx1 = (ثابت int32_t *) (vtcm_بنسټ + خونه_اندازه + chunk_elems * حجم(عائم));

                int32_t * final_idx = (int32_t *) (vtcm_بنسټ + kparams->یوځای_indices_off);
                بنية htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_پیل(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                یوځای_runs(val0, idx0, cctx.real_شمېر[0], val1, idx1, cctx.real_شمېر[1], عدم, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->معلومات, final_idx, ne00 * حجم(int32_t));
            } وإلا {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_بنسټ + chunk_elems * حجم(عائم));
                dma_sync_write(octx->ctx->dma[0], dst->معلومات, final_idx, ne00 * حجم(int32_t));
            }
        }
        أرجع HTP_حالت_OK;
    }

    بنية htp_ترتیبول_multi_لیکه_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_بنسټ        = vtcm_بنسټ;

    uint32_t dst_لیکه_بایټونه = (uint32_t) dst_لیکه_اندازه;
    uint32_t لیکې_فی_chunk = 0;
    htp_tensor_mdev_لیکې_فی_chunk(dst, حجم(int32_t), dst_لیکه_بایټونه, &لیکې_فی_chunk);
    إذا (لیکې_فی_chunk == 0) {
        لیکې_فی_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_تارونه - 1) / (uint32_t) kparams->n_تارونه;
    إذا (لیکې_فی_chunk > 1) {
        dr = hex_پړاو_پورته(dr, لیکې_فی_chunk);
    }
    mctx.nrows_فی_thread = dr;

    work_کتار_پرلپسې(octx->ctx->work_کتار, htp_ترتیبول_multi_لیکه_دنده, &mctx, (uint32_t) kparams->n_تارونه);

    أرجع HTP_حالت_OK;
}

صحيح op_argsort(بنية htp_ops_context * octx) {
    أرجع op_ترتیبول_common(octx, false);
}

صحيح op_سر_k(بنية htp_ops_context * octx) {
    أرجع op_ترتیبول_common(octx, true);
}
