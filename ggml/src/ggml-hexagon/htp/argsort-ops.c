#ضمن <string.h>
#ضمن <stdlib.h>
#ضمن <math.h>
#ضمن <HAP_farf.h>
#ضمن <HAP_perf.h>

#وصف_ڪريو GGML_COMMON_DECL_C
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

#جيڪڏهن_وصف_نه_ٿيل MIN
#وصف_ڪريو MIN(a, b) ((a) < (b) ? (a) : (b))
#جيڪڏهن_پورو


ساكن فراغ quicksort_قدرون_indices_asc(عائم * قدرون, int32_t * indices, صحيح کاٻو, صحيح ساڄو) {
    طالما (کاٻو < ساڄو) {
        عائم محور = قدرون[کاٻو + (ساڄو - کاٻو) / 2];
        صحيح i = کاٻو;
        صحيح j = ساڄو;

        طالما (i <= j) {
            طالما (قدرون[i] < محور) i++;
            طالما (قدرون[j] > محور) j--;

            إذا (i <= j) {
                عائم tmp_val = قدرون[i];
                قدرون[i] = قدرون[j];
                قدرون[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        إذا (j - کاٻو < ساڄو - i) {
            إذا (کاٻو < j) quicksort_قدرون_indices_asc(قدرون, indices, کاٻو, j);
            کاٻو = i;
        } وإلا {
            إذا (i < ساڄو) quicksort_قدرون_indices_asc(قدرون, indices, i, ساڄو);
            ساڄو = j;
        }
    }
}

ساكن فراغ quicksort_قدرون_indices_desc(عائم * قدرون, int32_t * indices, صحيح کاٻو, صحيح ساڄو) {
    طالما (کاٻو < ساڄو) {
        عائم محور = قدرون[کاٻو + (ساڄو - کاٻو) / 2];
        صحيح i = کاٻو;
        صحيح j = ساڄو;

        طالما (i <= j) {
            طالما (قدرون[i] > محور) i++;
            طالما (قدرون[j] < محور) j--;

            إذا (i <= j) {
                عائم tmp_val = قدرون[i];
                قدرون[i] = قدرون[j];
                قدرون[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        إذا (j - کاٻو < ساڄو - i) {
            إذا (کاٻو < j) quicksort_قدرون_indices_desc(قدرون, indices, کاٻو, j);
            کاٻو = i;
        } وإلا {
            إذا (i < ساڄو) quicksort_قدرون_indices_desc(قدرون, indices, i, ساڄو);
            ساڄو = j;
        }
    }
}

ساكن مضمن uint32_t چوٽي_k_max_قدر_اشاريو(ثابت عائم * قدرون, uint32_t n, عائم * قدر) {
    int32_t max_idx = 0;
    hvx_argmax_f32(قدرون, n, 0, قدر, &max_idx);
    أرجع (uint32_t) max_idx;
}

ساكن ثابت int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

ساكن مضمن فراغ init_indices_ramp(int32_t * indices_buf, uint32_t ڳڻپ, uint32_t بنياد) {
    ثابت HVX_ويڪٽر ind_init_vec = Q6_Vw_vadd_VwVw(*(ثابت HVX_ويڪٽر *) argsort_ramp_lut, Q6_V_vsplat_R(بنياد));
    ثابت HVX_ويڪٽر ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_ويڪٽر * indices_buf_vec = (HVX_ويڪٽر *) indices_buf;
    uint32_t انگ_vecs = ڳڻپ / 32;
    HVX_ويڪٽر curr_ind_vec = ind_init_vec;
    لكل (uint32_t j = 0; j < انگ_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = ڳڻپ % 32;
    إذا (rem > 0) {
        hvx_vec_store_a((فراغ *) &indices_buf_vec[انگ_vecs], rem * حجم(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
ساكن مضمن فراغ vec_cas(HVX_ويڪٽر * X_val, HVX_ويڪٽر * X_idx, HVX_ويڪٽر * Y_val, HVX_ويڪٽر * Y_idx, bool asc) {
    HVX_ويڪٽر_Pred pred = asc ? Q6_Q_vcmp_gt_VsfVsf(*X_val, *Y_val)
                              : Q6_Q_vcmp_gt_VsfVsf(*Y_val, *X_val);
    HVX_ويڪٽر ايندڙ_X_val = Q6_V_vmux_QVV(pred, *Y_val, *X_val);
    HVX_ويڪٽر ايندڙ_Y_val = Q6_V_vmux_QVV(pred, *X_val, *Y_val);
    HVX_ويڪٽر ايندڙ_X_idx = Q6_V_vmux_QVV(pred, *Y_idx, *X_idx);
    HVX_ويڪٽر Y_tmp_idx  = Q6_V_vmux_QVV(pred, *X_idx, *Y_idx);
    *X_val = ايندڙ_X_val;
    *Y_val = ايندڙ_Y_val;
    *X_idx = ايندڙ_X_idx;
    *Y_idx = Y_tmp_idx;
}

__attribute__((always_inline))
ساكن مضمن فراغ bitonic_cas_32(HVX_ويڪٽر * V, HVX_ويڪٽر * I, صحيح d, HVX_ويڪٽر_Pred dir_نقاب, HVX_ويڪٽر idx_vec, HVX_ويڪٽر zero_vec) {
    HVX_ويڪٽر_Pred نقاب_کاٻو;
    HVX_ويڪٽر V_rot_کاٻو, V_rot_ساڄو;
    HVX_ويڪٽر I_rot_کاٻو, I_rot_ساڄو;

    إذا (d == 1) {
        نقاب_کاٻو = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_کاٻو = Q6_V_vror_VR(*V, 4);
        V_rot_ساڄو = Q6_V_vror_VR(*V, 124);
        I_rot_کاٻو = Q6_V_vror_VR(*I, 4);
        I_rot_ساڄو = Q6_V_vror_VR(*I, 124);
    } وإلا إذا (d == 2) {
        نقاب_کاٻو = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_کاٻو = Q6_V_vror_VR(*V, 8);
        V_rot_ساڄو = Q6_V_vror_VR(*V, 120);
        I_rot_کاٻو = Q6_V_vror_VR(*I, 8);
        I_rot_ساڄو = Q6_V_vror_VR(*I, 120);
    } وإلا إذا (d == 4) {
        نقاب_کاٻو = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_کاٻو = Q6_V_vror_VR(*V, 16);
        V_rot_ساڄو = Q6_V_vror_VR(*V, 112);
        I_rot_کاٻو = Q6_V_vror_VR(*I, 16);
        I_rot_ساڄو = Q6_V_vror_VR(*I, 112);
    } وإلا إذا (d == 8) {
        نقاب_کاٻو = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_کاٻو = Q6_V_vror_VR(*V, 32);
        V_rot_ساڄو = Q6_V_vror_VR(*V, 96);
        I_rot_کاٻو = Q6_V_vror_VR(*I, 32);
        I_rot_ساڄو = Q6_V_vror_VR(*I, 96);
    } وإلا {
        نقاب_کاٻو = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_کاٻو = Q6_V_vror_VR(*V, 64);
        V_rot_ساڄو = Q6_V_vror_VR(*V, 64);
        I_rot_کاٻو = Q6_V_vror_VR(*I, 64);
        I_rot_ساڄو = Q6_V_vror_VR(*I, 64);
    }

    HVX_ويڪٽر V_paired = Q6_V_vmux_QVV(نقاب_کاٻو, V_rot_کاٻو, V_rot_ساڄو);
    HVX_ويڪٽر I_paired = Q6_V_vmux_QVV(نقاب_کاٻو, I_rot_کاٻو, I_rot_ساڄو);

    HVX_ويڪٽر_Pred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_ويڪٽر_Pred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_ويڪٽر_Pred نقاب_ساڄو = Q6_Q_not_Q(نقاب_کاٻو);
    HVX_ويڪٽر_Pred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(نقاب_کاٻو, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, نقاب_ساڄو)
    );
    HVX_ويڪٽر_Pred Q_مٽا_سٽا = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_نقاب, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_نقاب), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_مٽا_سٽا, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_مٽا_سٽا, I_paired, *I);
}

__attribute__((always_inline))
ساكن مضمن فراغ bitonic_ترتيب_ڏيو_generic_hvx(uint8_t * قدرون, uint8_t * indices, صحيح K, bool asc_ترتيب) {
    HVX_ويڪٽر V[32];
    HVX_ويڪٽر I[32];

    HVX_ويڪٽر zero_vec = Q6_V_vzero();
    HVX_ويڪٽر idx_vec = *(ثابت HVX_ويڪٽر *) argsort_ramp_lut;

    لكل (صحيح v = 0; v < K; v++) {
        V[v] = *(HVX_ويڪٽر *)(قدرون + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_ويڪٽر_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_ويڪٽر_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

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
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_ترتيب;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } وإلا {
                إذا (s < 5) {
                    HVX_ويڪٽر_Pred dir_نقاب = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    لكل (صحيح v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_نقاب, idx_vec, zero_vec);
                    }
                } وإلا {
                    لكل (صحيح v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_ترتيب;
                        HVX_ويڪٽر_Pred dir_نقاب = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_نقاب, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    لكل (صحيح v = 0; v < K; v++) {
        *(HVX_ويڪٽر *)(قدرون + v * 128)  = V[v];
        *(HVX_ويڪٽر *)(indices + v * 128) = I[v];
    }
}

ساكن فراغ bitonic_ترتيب_ڏيو_vtcm_desc(uint8_t * قدرون, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_ويڪٽر zero_vec = Q6_V_vzero();
    HVX_ويڪٽر idx_vec = *(ثابت HVX_ويڪٽر *) argsort_ramp_lut;

    HVX_ويڪٽر_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_ويڪٽر_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    إذا (init_indices) {
        لكل (uint32_t v = 0; v < n_vec; v++) {
            HVX_ويڪٽر idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_ويڪٽر *)(indices + v * 128) = idx;
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

                        HVX_ويڪٽر Vv1 = *(HVX_ويڪٽر *)(قدرون + v1 * 128);
                        HVX_ويڪٽر Iv1 = *(HVX_ويڪٽر *)(indices + v1 * 128);
                        HVX_ويڪٽر Vv2 = *(HVX_ويڪٽر *)(قدرون + v2 * 128);
                        HVX_ويڪٽر Iv2 = *(HVX_ويڪٽر *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_ويڪٽر *)(قدرون + v1 * 128)  = Vv1;
                        *(HVX_ويڪٽر *)(indices + v1 * 128) = Iv1;
                        *(HVX_ويڪٽر *)(قدرون + v2 * 128)  = Vv2;
                        *(HVX_ويڪٽر *)(indices + v2 * 128) = Iv2;
                    }
                }
            } وإلا {
                إذا (s < 5) {
                    HVX_ويڪٽر_Pred dir_نقاب = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    لكل (uint32_t v = 0; v < n_vec; v++) {
                        HVX_ويڪٽر Vv = *(HVX_ويڪٽر *)(قدرون + v * 128);
                        HVX_ويڪٽر Iv = *(HVX_ويڪٽر *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_نقاب, idx_vec, zero_vec);

                        *(HVX_ويڪٽر *)(قدرون + v * 128)  = Vv;
                        *(HVX_ويڪٽر *)(indices + v * 128) = Iv;
                    }
                } وإلا {
                    لكل (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_ويڪٽر_Pred dir_نقاب = asc ? pred_all_1s : pred_all_0s;

                        HVX_ويڪٽر Vv = *(HVX_ويڪٽر *)(قدرون + v * 128);
                        HVX_ويڪٽر Iv = *(HVX_ويڪٽر *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_نقاب, idx_vec, zero_vec);

                        *(HVX_ويڪٽر *)(قدرون + v * 128)  = Vv;
                        *(HVX_ويڪٽر *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

ساكن فراغ چوٽي_k_select_tiled(dma_قطار * dma_q, بنية htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               عائم * قدرون_buf, int32_t * indices_buf,
                               عائم * ٻاهريون_قدرون, int32_t * ٻاهريون_indices,
                               uint32_t chunk_بنياد) {
    ثابت uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_ڳڻپ    = n_tiles * k;
    uint32_t ملايو_n_vec        = hmx_ceil_div(candidate_ڳڻپ, 32);
    uint32_t ملايو_n_vec_pow2   = 1;
    طالما (ملايو_n_vec_pow2 < ملايو_n_vec) ملايو_n_vec_pow2 <<= 1;
    uint32_t ملايو_elems        = ملايو_n_vec_pow2 * 32;
    عائم * candidate_قدرون    = قدرون_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    لكل (uint32_t وٿي = 0; وٿي < n; وٿي += tile_elems) {
        uint32_t tile_ڳڻپ = MIN(tile_elems, n - وٿي);
        dma_sync_read(dma_q, قدرون_buf, src_addr + (size_t) وٿي * حجم(عائم), tile_ڳڻپ * حجم(عائم));

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) وٿي);
        إذا (tile_ڳڻپ < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (قدرون_buf + tile_ڳڻپ), -INFINITY, tile_elems - tile_ڳڻپ);
        }

        bitonic_ترتيب_ڏيو_vtcm_desc((uint8_t *) قدرون_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_ڳڻپ);
        لكل (uint32_t j = 0; j < tile_k; j++) {
            candidate_قدرون[candidate_pos] = قدرون_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) وٿي;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) وٿي);
    }

    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    إذا (ملايو_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_قدرون + candidate_pos), -INFINITY, ملايو_elems - candidate_pos);
        لكل (uint32_t j = candidate_pos; j < ملايو_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_ترتيب_ڏيو_vtcm_desc((uint8_t *) candidate_قدرون, (uint8_t *) candidate_indices, ملايو_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    لكل (uint32_t j = 0; j < k; j++) {
        ٻاهريون_قدرون[j]  = candidate_قدرون[j];
        ٻاهريون_indices[j] = candidate_indices[j] + (int32_t) chunk_بنياد;
    }
}

__attribute__((always_inline))
ساكن مضمن فراغ sort32_f32_hvx(uint8_t * قدرون, uint8_t * indices, تعداد ggml_ترتيب_ڏيو_ترتيب ترتيب) {
    bitonic_ترتيب_ڏيو_generic_hvx(قدرون, indices, 1, ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort64_f32_hvx(uint8_t * قدرون, uint8_t * indices, تعداد ggml_ترتيب_ڏيو_ترتيب ترتيب) {
    bitonic_ترتيب_ڏيو_generic_hvx(قدرون, indices, 2, ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort128_f32_hvx(uint8_t * قدرون, uint8_t * indices, تعداد ggml_ترتيب_ڏيو_ترتيب ترتيب) {
    bitonic_ترتيب_ڏيو_generic_hvx(قدرون, indices, 4, ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort256_f32_hvx(uint8_t * قدرون, uint8_t * indices, تعداد ggml_ترتيب_ڏيو_ترتيب ترتيب) {
    bitonic_ترتيب_ڏيو_generic_hvx(قدرون, indices, 8, ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort512_f32_hvx(uint8_t * قدرون, uint8_t * indices, تعداد ggml_ترتيب_ڏيو_ترتيب ترتيب) {
    bitonic_ترتيب_ڏيو_generic_hvx(قدرون, indices, 16, ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC);
}

__attribute__((always_inline))
ساكن مضمن فراغ sort1024_f32_hvx(uint8_t * قدرون, uint8_t * indices, تعداد ggml_ترتيب_ڏيو_ترتيب ترتيب) {
    bitonic_ترتيب_ڏيو_generic_hvx(قدرون, indices, 32, ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC);
}

ساكن فراغ ملايو_runs(
    ثابت عائم * محدود اندريون_val0, ثابت int32_t * محدود اندريون_idx0, uint32_t n0,
    ثابت عائم * محدود اندريون_val1, ثابت int32_t * محدود اندريون_idx1, uint32_t n1,
    عائم * محدود ٻاهريون_val, int32_t * محدود ٻاهريون_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t ٻاهريون = 0;

    إذا (asc) {
        طالما (i0 < n0 && i1 < n1) {
            إذا (اندريون_val0[i0] <= اندريون_val1[i1]) {
                إذا (ٻاهريون_val) ٻاهريون_val[ٻاهريون] = اندريون_val0[i0];
                ٻاهريون_idx[ٻاهريون] = اندريون_idx0[i0];
                i0++;
            } وإلا {
                إذا (ٻاهريون_val) ٻاهريون_val[ٻاهريون] = اندريون_val1[i1];
                ٻاهريون_idx[ٻاهريون] = اندريون_idx1[i1];
                i1++;
            }
            ٻاهريون++;
        }
    } وإلا {
        طالما (i0 < n0 && i1 < n1) {
            إذا (اندريون_val0[i0] >= اندريون_val1[i1]) {
                إذا (ٻاهريون_val) ٻاهريون_val[ٻاهريون] = اندريون_val0[i0];
                ٻاهريون_idx[ٻاهريون] = اندريون_idx0[i0];
                i0++;
            } وإلا {
                إذا (ٻاهريون_val) ٻاهريون_val[ٻاهريون] = اندريون_val1[i1];
                ٻاهريون_idx[ٻاهريون] = اندريون_idx1[i1];
                i1++;
            }
            ٻاهريون++;
        }
    }

    طالما (i0 < n0) {
        إذا (ٻاهريون_val) ٻاهريون_val[ٻاهريون] = اندريون_val0[i0];
        ٻاهريون_idx[ٻاهريون] = اندريون_idx0[i0];
        i0++;
        ٻاهريون++;
    }

    طالما (i1 < n1) {
        إذا (ٻاهريون_val) ٻاهريون_val[ٻاهريون] = اندريون_val1[i1];
        ٻاهريون_idx[ٻاهريون] = اندريون_idx1[i1];
        i1++;
        ٻاهريون++;
    }
}

بنية htp_ترتيب_ڏيو_chunk_ctx {
    بنية htp_ops_context *              octx;
    ثابت بنية htp_ترتيب_ڏيو_kernel_params * kparams;
    uint8_t *                             vtcm_بنياد;
    uint32_t                              real_ڳڻپ[HTP_MAX_NTHREADS];
};

ساكن فراغ htp_ترتيب_ڏيو_chunk_ڪم(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    بنية htp_ترتيب_ڏيو_chunk_ctx * cctx = (بنية htp_ترتيب_ڏيو_chunk_ctx *) معلومات;
    بنية htp_ops_context * octx = cctx->octx;
    ثابت بنية htp_ترتيب_ڏيو_kernel_params * kparams = cctx->kparams;
    ثابت بنية htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_بنياد  = i * chunk_elems;

    uint32_t real_ڳڻپ = (chunk_بنياد < ne00) ? MIN(chunk_elems, ne00 - chunk_بنياد) : 0;
    cctx->real_ڳڻپ[i] = real_ڳڻپ;

    uint8_t * spad = cctx->vtcm_بنياد + (size_t) kparams->phase1_جڳهه_سائيز * i;
    عائم *   قدرون_buf  = (عائم *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * حجم(عائم));

    عائم *   ملايو_قدرون  = (عائم *)   (cctx->vtcm_بنياد + kparams->ملايو_قدرون_off);
    int32_t * ملايو_indices = (int32_t *) (cctx->vtcm_بنياد + kparams->ملايو_indices_off);

    بنية htp_thread_trace * tr = &octx->ctx->trace[i];
    ثابت dma_addr_t src_addr = src0->معلومات + (size_t) chunk_بنياد * حجم(عائم);

    إذا (kparams->is_چوٽي_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        إذا (real_ڳڻپ == 0) {
            لكل (uint32_t j = 0; j < local_k; j++) {
                ملايو_قدرون[i * local_k + j]  = -INFINITY;
                ملايو_indices[i * local_k + j] = 0;
            }
            أرجع;
        }

        إذا (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], قدرون_buf, src_addr, real_ڳڻپ * حجم(عائم));

            htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            عائم max_قدر;
            uint32_t max_اشاريو = چوٽي_k_max_قدر_اشاريو(قدرون_buf, real_ڳڻپ, &max_قدر);
            ملايو_قدرون[i] = max_قدر;
            ملايو_indices[i] = (int32_t) (max_اشاريو + chunk_بنياد);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            أرجع;
        }

        إذا (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            چوٽي_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_ڳڻپ, local_k,
                               قدرون_buf, indices_buf,
                               &ملايو_قدرون[i * local_k], &ملايو_indices[i * local_k],
                               chunk_بنياد);
            أرجع;
        }

        dma_sync_read(octx->ctx->dma[i], قدرون_buf, src_addr, real_ڳڻپ * حجم(عائم));

        htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        إذا (chunk_elems > real_ڳڻپ) {
            hvx_splat_f32_u((uint8_t *)(قدرون_buf + real_ڳڻپ), -INFINITY, chunk_elems - real_ڳڻپ);
        }

        bitonic_ترتيب_ڏيو_vtcm_desc((uint8_t *) قدرون_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        لكل (uint32_t j = 0; j < local_k; j++) {
            ملايو_قدرون[i * local_k + j]  = قدرون_buf[j];
            ملايو_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_بنياد;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } وإلا {
        إذا (real_ڳڻپ > 0) {
            dma_sync_read(octx->ctx->dma[i], قدرون_buf, src_addr, real_ڳڻپ * حجم(عائم));

            htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_ڳڻپ, chunk_بنياد);

            إذا (kparams->ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC) {
                quicksort_قدرون_indices_asc(قدرون_buf, indices_buf, 0, real_ڳڻپ - 1);
            } وإلا {
                quicksort_قدرون_indices_desc(قدرون_buf, indices_buf, 0, real_ڳڻپ - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

بنية htp_argsort_ملايو_l1_ctx {
    بنية htp_ترتيب_ڏيو_chunk_ctx * cctx;
    bool                        asc;
};

ساكن فراغ htp_argsort_ملايو_l1_ڪم(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    بنية htp_argsort_ملايو_l1_ctx * mctx = (بنية htp_argsort_ملايو_l1_ctx *) معلومات;
    بنية htp_ترتيب_ڏيو_chunk_ctx * cctx = mctx->cctx;
    ثابت بنية htp_ترتيب_ڏيو_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_ڳڻپ[c0];
    uint32_t count1 = cctx->real_ڳڻپ[c1];

    size_t جڳهه_سائيز = (size_t) kparams->phase1_جڳهه_سائيز;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    ثابت عائم *   val0 = (ثابت عائم *)   (cctx->vtcm_بنياد + جڳهه_سائيز * c0);
    ثابت int32_t * idx0 = (ثابت int32_t *) (cctx->vtcm_بنياد + جڳهه_سائيز * c0 + chunk_elems * حجم(عائم));
    ثابت عائم *   val1 = (ثابت عائم *)   (cctx->vtcm_بنياد + جڳهه_سائيز * c1);
    ثابت int32_t * idx1 = (ثابت int32_t *) (cctx->vtcm_بنياد + جڳهه_سائيز * c1 + chunk_elems * حجم(عائم));

    uint32_t ٻاهريون_وٿي = (2 * i) * chunk_elems;
    عائم *   ٻاهريون_val = (عائم *)   (cctx->vtcm_بنياد + kparams->ملايو_قدرون_off)  + ٻاهريون_وٿي;
    int32_t * ٻاهريون_idx = (int32_t *) (cctx->vtcm_بنياد + kparams->ملايو_indices_off) + ٻاهريون_وٿي;

    بنية htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    ملايو_runs(val0, idx0, count0, val1, idx1, count1, ٻاهريون_val, ٻاهريون_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

بنية htp_argsort_ملايو_l2_ctx {
    بنية htp_ترتيب_ڏيو_chunk_ctx * cctx;
    bool                        asc;
};

ساكن فراغ htp_argsort_ملايو_l2_ڪم(بنانشان صحيح n, بنانشان صحيح i, فراغ * معلومات) {
    بنية htp_argsort_ملايو_l2_ctx * mctx = (بنية htp_argsort_ملايو_l2_ctx *) معلومات;
    بنية htp_ترتيب_ڏيو_chunk_ctx * cctx = mctx->cctx;
    ثابت بنية htp_ترتيب_ڏيو_kernel_params * kparams = cctx->kparams;

    size_t جڳهه_سائيز = (size_t) kparams->phase1_جڳهه_سائيز;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    ثابت عائم *   val0 = (ثابت عائم *)   (cctx->vtcm_بنياد + kparams->ملايو_قدرون_off)  + off0;
    ثابت int32_t * idx0 = (ثابت int32_t *) (cctx->vtcm_بنياد + kparams->ملايو_indices_off) + off0;
    ثابت عائم *   val1 = (ثابت عائم *)   (cctx->vtcm_بنياد + kparams->ملايو_قدرون_off)  + off1;
    ثابت int32_t * idx1 = (ثابت int32_t *) (cctx->vtcm_بنياد + kparams->ملايو_indices_off) + off1;

    uint32_t count0 = cctx->real_ڳڻپ[4*i + 0] + cctx->real_ڳڻپ[4*i + 1];
    uint32_t count1 = cctx->real_ڳڻپ[4*i + 2] + cctx->real_ڳڻپ[4*i + 3];

    uint8_t * جڳهه_بنياد = cctx->vtcm_بنياد + (4 * i) * جڳهه_سائيز;
    عائم *   ٻاهريون_val   = (عائم *) جڳهه_بنياد;
    int32_t * ٻاهريون_idx   = (int32_t *) (جڳهه_بنياد + 4 * chunk_elems * حجم(عائم));

    بنية htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    ملايو_runs(val0, idx0, count0, val1, idx1, count1, ٻاهريون_val, ٻاهريون_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

بنية htp_ترتيب_ڏيو_multi_صف_ctx {
    بنية htp_ops_context *              octx;
    ثابت بنية htp_ترتيب_ڏيو_kernel_params * kparams;
    uint8_t *                             vtcm_بنياد;
    uint32_t                              nrows_في_thread;
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

ساكن مضمن size_t coord_وٿي(بنية htp_tensor_صف_coord c, size_t nb1, size_t nb2, size_t nb3) {
    أرجع (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

ساكن مضمن فراغ compute_صف_ترتيب_ڏيو(
    عائم * cur_قدرون,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_چوٽي_k,
    تعداد ggml_ترتيب_ڏيو_ترتيب ترتيب
) {
    إذا (is_چوٽي_k) {
        إذا (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            بدل (ne00) {
                حالة 32:   sort32_f32_hvx((uint8_t *) cur_قدرون, (uint8_t *) cur_indices, GGML_ترتيب_ڏيو_ترتيب_DESC); اكسر;
                حالة 64:   sort64_f32_hvx((uint8_t *) cur_قدرون, (uint8_t *) cur_indices, GGML_ترتيب_ڏيو_ترتيب_DESC); اكسر;
                حالة 128:  sort128_f32_hvx((uint8_t *) cur_قدرون, (uint8_t *) cur_indices, GGML_ترتيب_ڏيو_ترتيب_DESC); اكسر;
                حالة 256:  sort256_f32_hvx((uint8_t *) cur_قدرون, (uint8_t *) cur_indices, GGML_ترتيب_ڏيو_ترتيب_DESC); اكسر;
                حالة 512:  sort512_f32_hvx((uint8_t *) cur_قدرون, (uint8_t *) cur_indices, GGML_ترتيب_ڏيو_ترتيب_DESC); اكسر;
                حالة 1024: sort1024_f32_hvx((uint8_t *) cur_قدرون, (uint8_t *) cur_indices, GGML_ترتيب_ڏيو_ترتيب_DESC); اكسر;
            }
        } وإلا {
            إذا (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_قدرون + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_ترتيب_ڏيو_vtcm_desc((uint8_t *) cur_قدرون, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } وإلا {
        إذا (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            بدل (ne00) {
                حالة 32:   sort32_f32_hvx((uint8_t *) cur_قدرون, (uint8_t *) cur_indices, ترتيب); اكسر;
                حالة 64:   sort64_f32_hvx((uint8_t *) cur_قدرون, (uint8_t *) cur_indices, ترتيب); اكسر;
                حالة 128:  sort128_f32_hvx((uint8_t *) cur_قدرون, (uint8_t *) cur_indices, ترتيب); اكسر;
                حالة 256:  sort256_f32_hvx((uint8_t *) cur_قدرون, (uint8_t *) cur_indices, ترتيب); اكسر;
                حالة 512:  sort512_f32_hvx((uint8_t *) cur_قدرون, (uint8_t *) cur_indices, ترتيب); اكسر;
                حالة 1024: sort1024_f32_hvx((uint8_t *) cur_قدرون, (uint8_t *) cur_indices, ترتيب); اكسر;
            }
        } وإلا {
            init_indices_ramp(cur_indices, ne00, 0);
            إذا (ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC) {
                quicksort_قدرون_indices_asc(cur_قدرون, cur_indices, 0, ne00 - 1);
            } وإلا {
                quicksort_قدرون_indices_desc(cur_قدرون, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

ساكن فراغ htp_ترتيب_ڏيو_multi_صف_ڪم(بنانشان صحيح n, بنانشان صحيح ith, فراغ * معلومات) {
    بنية htp_ترتيب_ڏيو_multi_صف_ctx * mctx = (بنية htp_ترتيب_ڏيو_multi_صف_ctx *) معلومات;
    بنية htp_ops_context * octx = mctx->octx;
    ثابت بنية htp_ترتيب_ڏيو_kernel_params * kparams = mctx->kparams;
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

    uint32_t صف_شروعات = (uint32_t) kparams->صف_شروعات;
    uint32_t صف_پڇاڙي   = (uint32_t) kparams->صف_پڇاڙي;
    uint32_t r_شروعات   = صف_شروعات + ith * mctx->nrows_في_thread;
    uint32_t r_پڇاڙي     = MIN(r_شروعات + mctx->nrows_في_thread, صف_پڇاڙي);

    إذا (r_شروعات >= r_پڇاڙي) أرجع;

    size_t spad_وٿي = ith * (size_t) (kparams->phase1_جڳهه_سائيز * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_بنياد + spad_وٿي;
    size_t جڳهه_سائيز = (size_t) kparams->phase1_جڳهه_سائيز;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    عائم *   قدرون_buf  = (عائم *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * حجم(عائم));

    dma_قطار * dma_q = octx->ctx->dma[ith];
    بنية htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_چوٽي_k = kparams->is_چوٽي_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    تعداد ggml_ترتيب_ڏيو_ترتيب ترتيب = (تعداد ggml_ترتيب_ڏيو_ترتيب) kparams->ترتيب;

    ثابت uint32_t src_بائيٽون = ne00 * حجم(عائم);
    ثابت uint32_t dst_بائيٽون = (is_چوٽي_k ? k : ne00) * حجم(int32_t);

    إذا (kparams->n_slots == 2) {
        بنية htp_tensor_صف_coord مفرد_coord = init_صف_coord(r_شروعات, ne01, ne02);
        لكل (uint32_t r = r_شروعات, spad_idx = 0; r < r_پڇاڙي && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * جڳهه_سائيز;
            عائم *   cur_قدرون  = (عائم *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * حجم(عائم));

            dma_addr_t dst_addr = dst->معلومات + coord_وٿي(مفرد_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->معلومات + coord_وٿي(مفرد_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_addr, cur_indices),
                           dst_بائيٽون, dst_بائيٽون, dst_بائيٽون, 0);

            // Prefetch input row
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(cur_قدرون, src_addr),
                           src_بائيٽون, src_بائيٽون, src_بائيٽون, 1);

            قدم_صف_coord(&مفرد_coord, ne01, ne02);
        }

        بنية htp_tensor_صف_coord dst_coord = init_صف_coord(r_شروعات, ne01, ne02);
        بنية htp_tensor_صف_coord ايندڙ_src_coord = init_صف_coord(r_شروعات + 2, ne01, ne02);

        لكل (uint32_t r = r_شروعات; r < r_پڇاڙي; r++) {
            uint32_t cur_جڳهه = (r - r_شروعات) & 1;
            uint8_t * cur_spad = thread_spad + cur_جڳهه * جڳهه_سائيز;
            عائم *   cur_قدرون  = (عائم *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * حجم(عائم));

            // Wait for previous writeback of this slot to complete
            dma_قطار_ڪڍو(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_قطار_ڪڍو(dma_q);

            htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_صف_ترتيب_ڏيو(cur_قدرون, cur_indices, ne00, chunk_elems, is_چوٽي_k, ترتيب);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->معلومات + coord_وٿي(dst_coord, nb1, nb2, nb3);
            dma_قطار_ڌڪو(dma_q, dma_make_معلومات(dst_addr, cur_indices),
                           dst_بائيٽون, dst_بائيٽون, dst_بائيٽون, 1);
            قدم_صف_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            ثابت uint32_t ايندڙ_صف = r + 2;
            إذا (ايندڙ_صف < r_پڇاڙي) {
                dma_addr_t ايندڙ_src_addr = src0->معلومات + coord_وٿي(ايندڙ_src_coord, nb01, nb02, nb03);
                dma_قطار_ڌڪو(dma_q, dma_make_معلومات(cur_قدرون, ايندڙ_src_addr),
                               src_بائيٽون, src_بائيٽون, src_بائيٽون, 1);
                قدم_صف_coord(&ايندڙ_src_coord, ne01, ne02);
            }
        }

        dma_قطار_flush(dma_q);
    } وإلا {
        بنية htp_tensor_صف_coord coord = init_صف_coord(r_شروعات, ne01, ne02);
        لكل (uint32_t r = r_شروعات; r < r_پڇاڙي; r++) {
            dma_addr_t src_addr = src0->معلومات + coord_وٿي(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->معلومات + coord_وٿي(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, قدرون_buf, src_addr, src_بائيٽون);

            htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_صف_ترتيب_ڏيو(قدرون_buf, indices_buf, ne00, chunk_elems, is_چوٽي_k, ترتيب);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_بائيٽون);
            قدم_صف_coord(&coord, ne01, ne02);
        }
    }
}

ساكن صحيح op_ترتيب_ڏيو_common(بنية htp_ops_context * octx, bool is_چوٽي_k) {
    إذا (octx->src[0]->type != HTP_TYPE_F32) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت بنية htp_tensor * src0 = octx->src[0];
    ثابت بنية htp_tensor * dst  = octx->dst;

    إذا (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        أرجع HTP_حالت_NO_SUPPORT;
    }

    ثابت uint32_t ڪل_صفون = src0->ne[1] * src0->ne[2] * src0->ne[3];
    ثابت size_t dst_صف_سائيز = dst->ne[0] * حجم(int32_t);

    uint32_t صف_شروعات = 0;
    uint32_t صف_پڇاڙي   = ڪل_صفون;
    إذا (octx->ctx->mdev.ڳڻپ > 1) {
        uint32_t صفون_في_chunk = 0;
        htp_tensor_mdev_صفون_في_chunk(dst, حجم(int32_t), (uint32_t) dst_صف_سائيز, &صفون_في_chunk);
        ثابت بنية htp_tensor_mdev_range range = htp_tensor_mdev_حصا_ڪريو(
            ڪل_صفون, صفون_في_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.ڳڻپ, &octx->ctx->mdev.ڳڻپ_div);
        صف_شروعات = range.شروعات;
        صف_پڇاڙي   = range.شروعات + range.ڳڻپ;
    }

    ثابت uint32_t nrows = صف_پڇاڙي - صف_شروعات;
    إذا (nrows == 0) {
        أرجع HTP_حالت_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    بنية htp_ترتيب_ڏيو_kernel_params kparams_local;
    ثابت بنية htp_ترتيب_ڏيو_kernel_params * kparams = (ثابت بنية htp_ترتيب_ڏيو_kernel_params *) octx->kernel_params;

    إذا (kparams->n_ڌاڳا == 0) {
        بنية htp_ترتيب_ڏيو_vtcm_layout layout;
        إذا (!htp_ترتيب_ڏيو_solve_layout(&layout, ne00, ڪل_صفون, k, octx->n_ڌاڳا, octx->ctx->vtcm_سائيز, is_چوٽي_k)) {
            أرجع HTP_حالت_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, حجم(kparams_local));
        kparams_local.n_ڌاڳا         = (int32_t) layout.n_ڌاڳا;
        kparams_local.ڪل_صفون        = (int32_t) ڪل_صفون;
        kparams_local.صف_شروعات         = (int32_t) صف_شروعات;
        kparams_local.صف_پڇاڙي           = (int32_t) صف_پڇاڙي;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.ترتيب             = is_چوٽي_k ? GGML_ترتيب_ڏيو_ترتيب_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_چوٽي_k          = is_چوٽي_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_سائيز         = (int32_t) layout.ڪل_بائيٽون;
        kparams_local.phase1_جڳهه_سائيز  = (int32_t) layout.phase1_جڳهه_سائيز;
        kparams_local.ملايو_قدرون_off  = (int32_t) layout.ملايو_قدرون_off;
        kparams_local.ملايو_indices_off = (int32_t) layout.ملايو_indices_off;
        kparams_local.ملايو_elems       = (int32_t) layout.ملايو_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } وإلا إذا (octx->ctx->mdev.ڳڻپ > 1) {
        يادگيري_نقل(&kparams_local, kparams, حجم(kparams_local));
        kparams_local.صف_شروعات = (int32_t) صف_شروعات;
        kparams_local.صف_پڇاڙي   = (int32_t) صف_پڇاڙي;
        kparams = &kparams_local;
    }

    إذا (octx->ctx->vtcm_سائيز < (size_t) kparams->vtcm_سائيز) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_سائيز, octx->ctx->vtcm_سائيز);
        أرجع HTP_حالت_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_بنياد = (uint8_t *) octx->ctx->vtcm_بنياد;

    إذا (ڪل_صفون == 1 && kparams->n_chunks > 1) {
        بنية htp_ترتيب_ڏيو_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_بنياد = vtcm_بنياد;
        memset(cctx.real_ڳڻپ, 0, حجم(cctx.real_ڳڻپ));

        work_قطار_لاڳيتو(octx->ctx->work_قطار, htp_ترتيب_ڏيو_chunk_ڪم, &cctx, (uint32_t) kparams->n_chunks);

        إذا (is_چوٽي_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            إذا (k == 1 && ne00 >= 128*1024) {
                عائم *   cand_vals = (عائم *)   (vtcm_بنياد + kparams->ملايو_قدرون_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_بنياد + kparams->ملايو_indices_off);
                عائم بهترين_val = cand_vals[0];
                int32_t بهترين_idx = cand_idxs[0];
                لكل (int32_t c = 1; c < kparams->n_chunks; c++) {
                    إذا (cand_vals[c] > بهترين_val) {
                        بهترين_val = cand_vals[c];
                        بهترين_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->معلومات;
                dst_ptr[0] = بهترين_idx;
                أرجع HTP_حالت_OK;
            }

            uint32_t ڪل_candidates = (uint32_t) kparams->n_chunks * local_k;
            عائم *   cand_vals = (عائم *)   (vtcm_بنياد + kparams->ملايو_قدرون_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_بنياد + kparams->ملايو_indices_off);
            uint32_t ملايو_elems = (uint32_t) kparams->ملايو_elems;

            إذا (ملايو_elems > ڪل_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + ڪل_candidates), -INFINITY, ملايو_elems - ڪل_candidates);
                لكل (uint32_t j = ڪل_candidates; j < ملايو_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            بنية htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_ترتيب_ڏيو_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, ملايو_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->معلومات, cand_idxs, k * حجم(int32_t));
        } وإلا {
            bool asc = kparams->ترتيب == GGML_ترتيب_ڏيو_ترتيب_ASC;
            إذا (kparams->n_chunks == 8) {
                بنية htp_argsort_ملايو_l1_ctx l1ctx = { &cctx, asc };
                work_قطار_لاڳيتو(octx->ctx->work_قطار, htp_argsort_ملايو_l1_ڪم, &l1ctx, 4);

                بنية htp_argsort_ملايو_l2_ctx l2ctx = { &cctx, asc };
                work_قطار_لاڳيتو(octx->ctx->work_قطار, htp_argsort_ملايو_l2_ڪم, &l2ctx, 2);

                size_t جڳهه_سائيز = (size_t) kparams->phase1_جڳهه_سائيز;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ثابت عائم *   val0 = (ثابت عائم *) vtcm_بنياد;
                ثابت int32_t * idx0 = (ثابت int32_t *) (vtcm_بنياد + 4 * chunk_elems * حجم(عائم));
                ثابت عائم *   val1 = (ثابت عائم *) (vtcm_بنياد + 4 * جڳهه_سائيز);
                ثابت int32_t * idx1 = (ثابت int32_t *) (vtcm_بنياد + 4 * جڳهه_سائيز + 4 * chunk_elems * حجم(عائم));

                uint32_t n01 = cctx.real_ڳڻپ[0] + cctx.real_ڳڻپ[1] + cctx.real_ڳڻپ[2] + cctx.real_ڳڻپ[3];
                uint32_t n23 = cctx.real_ڳڻپ[4] + cctx.real_ڳڻپ[5] + cctx.real_ڳڻپ[6] + cctx.real_ڳڻپ[7];

                int32_t * final_idx = (int32_t *) (vtcm_بنياد + kparams->ملايو_indices_off);
                بنية htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ملايو_runs(val0, idx0, n01, val1, idx1, n23, عدم, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->معلومات, final_idx, ne00 * حجم(int32_t));
            } وإلا إذا (kparams->n_chunks == 4) {
                بنية htp_argsort_ملايو_l1_ctx l1ctx = { &cctx, asc };
                work_قطار_لاڳيتو(octx->ctx->work_قطار, htp_argsort_ملايو_l1_ڪم, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_ڳڻپ[0] + cctx.real_ڳڻپ[1];
                uint32_t n23 = cctx.real_ڳڻپ[2] + cctx.real_ڳڻپ[3];

                ثابت عائم *   val01 = (ثابت عائم *)   (vtcm_بنياد + kparams->ملايو_قدرون_off);
                ثابت int32_t * idx01 = (ثابت int32_t *) (vtcm_بنياد + kparams->ملايو_indices_off);
                ثابت عائم *   val23 = (ثابت عائم *)   (vtcm_بنياد + kparams->ملايو_قدرون_off)  + 2 * chunk_elems;
                ثابت int32_t * idx23 = (ثابت int32_t *) (vtcm_بنياد + kparams->ملايو_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_بنياد;
                بنية htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ملايو_runs(val01, idx01, n01, val23, idx23, n23, عدم, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->معلومات, final_idx, ne00 * حجم(int32_t));
            } وإلا إذا (kparams->n_chunks == 2) {
                size_t جڳهه_سائيز = (size_t) kparams->phase1_جڳهه_سائيز;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                ثابت عائم *   val0 = (ثابت عائم *)   (vtcm_بنياد);
                ثابت int32_t * idx0 = (ثابت int32_t *) (vtcm_بنياد + chunk_elems * حجم(عائم));
                ثابت عائم *   val1 = (ثابت عائم *)   (vtcm_بنياد + جڳهه_سائيز);
                ثابت int32_t * idx1 = (ثابت int32_t *) (vtcm_بنياد + جڳهه_سائيز + chunk_elems * حجم(عائم));

                int32_t * final_idx = (int32_t *) (vtcm_بنياد + kparams->ملايو_indices_off);
                بنية htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_شروعات(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                ملايو_runs(val0, idx0, cctx.real_ڳڻپ[0], val1, idx1, cctx.real_ڳڻپ[1], عدم, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->معلومات, final_idx, ne00 * حجم(int32_t));
            } وإلا {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_بنياد + chunk_elems * حجم(عائم));
                dma_sync_write(octx->ctx->dma[0], dst->معلومات, final_idx, ne00 * حجم(int32_t));
            }
        }
        أرجع HTP_حالت_OK;
    }

    بنية htp_ترتيب_ڏيو_multi_صف_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_بنياد        = vtcm_بنياد;

    uint32_t dst_صف_بائيٽون = (uint32_t) dst_صف_سائيز;
    uint32_t صفون_في_chunk = 0;
    htp_tensor_mdev_صفون_في_chunk(dst, حجم(int32_t), dst_صف_بائيٽون, &صفون_في_chunk);
    إذا (صفون_في_chunk == 0) {
        صفون_في_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_ڌاڳا - 1) / (uint32_t) kparams->n_ڌاڳا;
    إذا (صفون_في_chunk > 1) {
        dr = hex_ڦيرو_مٿي(dr, صفون_في_chunk);
    }
    mctx.nrows_في_thread = dr;

    work_قطار_لاڳيتو(octx->ctx->work_قطار, htp_ترتيب_ڏيو_multi_صف_ڪم, &mctx, (uint32_t) kparams->n_ڌاڳا);

    أرجع HTP_حالت_OK;
}

صحيح op_argsort(بنية htp_ops_context * octx) {
    أرجع op_ترتيب_ڏيو_common(octx, false);
}

صحيح op_چوٽي_k(بنية htp_ops_context * octx) {
    أرجع op_ترتيب_ڏيو_common(octx, true);
}
