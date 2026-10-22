#అన్తర్భావ <string.h>
#అన్తర్భావ <stdlib.h>
#అన్తర్భావ <math.h>
#అన్తర్భావ <HAP_farf.h>
#అన్తర్భావ <HAP_perf.h>

#నిర్వచించు GGML_COMMON_DECL_C
#అన్తర్భావ "ggml-common.h"
#అన్తర్భావ "ggml.h"

#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "hvx-copy.h"
#అన్తర్భావ "hvx-reduce.h"
#అన్తర్భావ "dma-queue.h"

#అన్తర్భావ "hex-common.h"
#అన్తర్భావ "htp-ctx.h"
#అన్తర్భావ "htp-ops.h"
#అన్తర్భావ "htp-tensor.h"
#అన్తర్భావ "argsort-ops.h"

#నిర్వచితం_కాకపోతే MIN
#నిర్వచించు MIN(a, b) ((a) < (b) ? (a) : (b))
#అయితే_ముగింపు


స్థిర శూన్య quicksort_విలువలు_indices_asc(దశాంశం * విలువలు, int32_t * indices, పూర్ణాంకం ఎడమ_2, పూర్ణాంకం కుడి_2) {
    అయితే (ఎడమ_2 < కుడి_2) {
        దశాంశం ఇరుసు = విలువలు[ఎడమ_2 + (కుడి_2 - ఎడమ_2) / 2];
        పూర్ణాంకం i = ఎడమ_2;
        పూర్ణాంకం j = కుడి_2;

        అయితే (i <= j) {
            అయితే (విలువలు[i] < ఇరుసు) i++;
            అయితే (విలువలు[j] > ఇరుసు) j--;

            ఒకవేళ (i <= j) {
                దశాంశం tmp_val = విలువలు[i];
                విలువలు[i] = విలువలు[j];
                విలువలు[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        ఒకవేళ (j - ఎడమ_2 < కుడి_2 - i) {
            ఒకవేళ (ఎడమ_2 < j) quicksort_విలువలు_indices_asc(విలువలు, indices, ఎడమ_2, j);
            ఎడమ_2 = i;
        } లేకపోతే {
            ఒకవేళ (i < కుడి_2) quicksort_విలువలు_indices_asc(విలువలు, indices, i, కుడి_2);
            కుడి_2 = j;
        }
    }
}

స్థిర శూన్య quicksort_విలువలు_indices_desc(దశాంశం * విలువలు, int32_t * indices, పూర్ణాంకం ఎడమ_2, పూర్ణాంకం కుడి_2) {
    అయితే (ఎడమ_2 < కుడి_2) {
        దశాంశం ఇరుసు = విలువలు[ఎడమ_2 + (కుడి_2 - ఎడమ_2) / 2];
        పూర్ణాంకం i = ఎడమ_2;
        పూర్ణాంకం j = కుడి_2;

        అయితే (i <= j) {
            అయితే (విలువలు[i] > ఇరుసు) i++;
            అయితే (విలువలు[j] < ఇరుసు) j--;

            ఒకవేళ (i <= j) {
                దశాంశం tmp_val = విలువలు[i];
                విలువలు[i] = విలువలు[j];
                విలువలు[j] = tmp_val;

                int32_t tmp_idx = indices[i];
                indices[i] = indices[j];
                indices[j] = tmp_idx;
                i++;
                j--;
            }
        }

        // Tail-recursion elimination to bound stack depth
        ఒకవేళ (j - ఎడమ_2 < కుడి_2 - i) {
            ఒకవేళ (ఎడమ_2 < j) quicksort_విలువలు_indices_desc(విలువలు, indices, ఎడమ_2, j);
            ఎడమ_2 = i;
        } లేకపోతే {
            ఒకవేళ (i < కుడి_2) quicksort_విలువలు_indices_desc(విలువలు, indices, i, కుడి_2);
            కుడి_2 = j;
        }
    }
}

స్థిర అంతఃవరుస uint32_t శిఖరం_k_max_విలువ_సూచీ(స్థిరరాశి దశాంశం * విలువలు, uint32_t n, దశాంశం * విలువ) {
    int32_t max_idx = 0;
    hvx_argmax_f32(విలువలు, n, 0, విలువ, &max_idx);
    తిరిగి (uint32_t) max_idx;
}

స్థిర స్థిరరాశి int32_t argsort_ramp_lut[32] __attribute__((aligned(VLEN))) = {
    0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15,
    16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 31
};

స్థిర అంతఃవరుస శూన్య init_indices_ramp(int32_t * indices_buf, uint32_t లెక్క, uint32_t పునాది) {
    స్థిరరాశి HVX_సదిశ ind_init_vec = Q6_Vw_vadd_VwVw(*(స్థిరరాశి HVX_సదిశ *) argsort_ramp_lut, Q6_V_vsplat_R(పునాది));
    స్థిరరాశి HVX_సదిశ ind_diff_vec = Q6_V_vsplat_R(32);
    HVX_సదిశ * indices_buf_vec = (HVX_సదిశ *) indices_buf;
    uint32_t అంకె_vecs = లెక్క / 32;
    HVX_సదిశ curr_ind_vec = ind_init_vec;
    కొరకు (uint32_t j = 0; j < అంకె_vecs; j++) {
        indices_buf_vec[j] = curr_ind_vec;
        curr_ind_vec = Q6_Vw_vadd_VwVw(curr_ind_vec, ind_diff_vec);
    }
    uint32_t rem = లెక్క % 32;
    ఒకవేళ (rem > 0) {
        hvx_vec_store_a((శూన్య *) &indices_buf_vec[అంకె_vecs], rem * పరిమాణం(int32_t), curr_ind_vec);
    }
}

__attribute__((always_inline))
స్థిర అంతఃవరుస శూన్య vec_cas(HVX_సదిశ * X_val, HVX_సదిశ * X_idx, HVX_సదిశ * Y_val, HVX_సదిశ * Y_idx, bool asc) {
    HVX_సదిశ_Pred pred = asc ? Q6_Q_vcmp_gt_VsfVsf(*X_val, *Y_val)
                              : Q6_Q_vcmp_gt_VsfVsf(*Y_val, *X_val);
    HVX_సదిశ తదుపరి_X_val = Q6_V_vmux_QVV(pred, *Y_val, *X_val);
    HVX_సదిశ తదుపరి_Y_val = Q6_V_vmux_QVV(pred, *X_val, *Y_val);
    HVX_సదిశ తదుపరి_X_idx = Q6_V_vmux_QVV(pred, *Y_idx, *X_idx);
    HVX_సదిశ Y_tmp_idx  = Q6_V_vmux_QVV(pred, *X_idx, *Y_idx);
    *X_val = తదుపరి_X_val;
    *Y_val = తదుపరి_Y_val;
    *X_idx = తదుపరి_X_idx;
    *Y_idx = Y_tmp_idx;
}

__attribute__((always_inline))
స్థిర అంతఃవరుస శూన్య bitonic_cas_32(HVX_సదిశ * V, HVX_సదిశ * I, పూర్ణాంకం d, HVX_సదిశ_Pred dir_ముసుగు, HVX_సదిశ idx_vec, HVX_సదిశ zero_vec) {
    HVX_సదిశ_Pred ముసుగు_ఎడమ;
    HVX_సదిశ V_rot_ఎడమ, V_rot_కుడి;
    HVX_సదిశ I_rot_ఎడమ, I_rot_కుడి;

    ఒకవేళ (d == 1) {
        ముసుగు_ఎడమ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1)), zero_vec);
        V_rot_ఎడమ = Q6_V_vror_VR(*V, 4);
        V_rot_కుడి = Q6_V_vror_VR(*V, 124);
        I_rot_ఎడమ = Q6_V_vror_VR(*I, 4);
        I_rot_కుడి = Q6_V_vror_VR(*I, 124);
    } లేకపోతే ఒకవేళ (d == 2) {
        ముసుగు_ఎడమ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(2)), zero_vec);
        V_rot_ఎడమ = Q6_V_vror_VR(*V, 8);
        V_rot_కుడి = Q6_V_vror_VR(*V, 120);
        I_rot_ఎడమ = Q6_V_vror_VR(*I, 8);
        I_rot_కుడి = Q6_V_vror_VR(*I, 120);
    } లేకపోతే ఒకవేళ (d == 4) {
        ముసుగు_ఎడమ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(4)), zero_vec);
        V_rot_ఎడమ = Q6_V_vror_VR(*V, 16);
        V_rot_కుడి = Q6_V_vror_VR(*V, 112);
        I_rot_ఎడమ = Q6_V_vror_VR(*I, 16);
        I_rot_కుడి = Q6_V_vror_VR(*I, 112);
    } లేకపోతే ఒకవేళ (d == 8) {
        ముసుగు_ఎడమ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(8)), zero_vec);
        V_rot_ఎడమ = Q6_V_vror_VR(*V, 32);
        V_rot_కుడి = Q6_V_vror_VR(*V, 96);
        I_rot_ఎడమ = Q6_V_vror_VR(*I, 32);
        I_rot_కుడి = Q6_V_vror_VR(*I, 96);
    } లేకపోతే {
        ముసుగు_ఎడమ = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(16)), zero_vec);
        V_rot_ఎడమ = Q6_V_vror_VR(*V, 64);
        V_rot_కుడి = Q6_V_vror_VR(*V, 64);
        I_rot_ఎడమ = Q6_V_vror_VR(*I, 64);
        I_rot_కుడి = Q6_V_vror_VR(*I, 64);
    }

    HVX_సదిశ V_paired = Q6_V_vmux_QVV(ముసుగు_ఎడమ, V_rot_ఎడమ, V_rot_కుడి);
    HVX_సదిశ I_paired = Q6_V_vmux_QVV(ముసుగు_ఎడమ, I_rot_ఎడమ, I_rot_కుడి);

    HVX_సదిశ_Pred V_gt_Vpaired = Q6_Q_vcmp_gt_VsfVsf(*V, V_paired);
    HVX_సదిశ_Pred Vpaired_gt_V = Q6_Q_vcmp_gt_VsfVsf(V_paired, *V);
    HVX_సదిశ_Pred ముసుగు_కుడి = Q6_Q_not_Q(ముసుగు_ఎడమ);
    HVX_సదిశ_Pred Q_asc = Q6_Q_or_QQ(
        Q6_Q_and_QQ(ముసుగు_ఎడమ, V_gt_Vpaired),
        Q6_Q_and_QQ(Vpaired_gt_V, ముసుగు_కుడి)
    );
    HVX_సదిశ_Pred Q_మార్పిడి = Q6_Q_or_QQ(
        Q6_Q_and_QQ(dir_ముసుగు, Q_asc),
        Q6_Q_and_QQ(Q6_Q_not_Q(dir_ముసుగు), Q6_Q_not_Q(Q_asc))
    );

    *V = Q6_V_vmux_QVV(Q_మార్పిడి, V_paired, *V);
    *I = Q6_V_vmux_QVV(Q_మార్పిడి, I_paired, *I);
}

__attribute__((always_inline))
స్థిర అంతఃవరుస శూన్య bitonic_క్రమీకరించు_generic_hvx(uint8_t * విలువలు, uint8_t * indices, పూర్ణాంకం K, bool asc_క్రమం) {
    HVX_సదిశ V[32];
    HVX_సదిశ I[32];

    HVX_సదిశ zero_vec = Q6_V_vzero();
    HVX_సదిశ idx_vec = *(స్థిరరాశి HVX_సదిశ *) argsort_ramp_lut;

    కొరకు (పూర్ణాంకం v = 0; v < K; v++) {
        V[v] = *(HVX_సదిశ *)(విలువలు + v * 128);
        I[v] = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
    }

    HVX_సదిశ_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_సదిశ_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    పూర్ణాంకం M = 5;
    అయితే ((1 << (M - 5)) < K) M++;

    కొరకు (పూర్ణాంకం s = 1; s <= M; s++) {
        కొరకు (పూర్ణాంకం stage_d = s - 1; stage_d >= 0; stage_d--) {
            పూర్ణాంకం d = 1 << stage_d;
            ఒకవేళ (d >= 32) {
                పూర్ణాంకం v_dist = d / 32;
                కొరకు (పూర్ణాంకం v1 = 0; v1 < K; v1++) {
                    ఒకవేళ ((v1 & v_dist) == 0) {
                        పూర్ణాంకం v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : asc_క్రమం;
                        vec_cas(&V[v1], &I[v1], &V[v2], &I[v2], asc);
                    }
                }
            } లేకపోతే {
                ఒకవేళ (s < 5) {
                    HVX_సదిశ_Pred dir_ముసుగు = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    కొరకు (పూర్ణాంకం v = 0; v < K; v++) {
                        bitonic_cas_32(&V[v], &I[v], d, dir_ముసుగు, idx_vec, zero_vec);
                    }
                } లేకపోతే {
                    కొరకు (పూర్ణాంకం v = 0; v < K; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : asc_క్రమం;
                        HVX_సదిశ_Pred dir_ముసుగు = asc ? pred_all_1s : pred_all_0s;
                        bitonic_cas_32(&V[v], &I[v], d, dir_ముసుగు, idx_vec, zero_vec);
                    }
                }
            }
        }
    }

    కొరకు (పూర్ణాంకం v = 0; v < K; v++) {
        *(HVX_సదిశ *)(విలువలు + v * 128)  = V[v];
        *(HVX_సదిశ *)(indices + v * 128) = I[v];
    }
}

స్థిర శూన్య bitonic_క్రమీకరించు_vtcm_desc(uint8_t * విలువలు, uint8_t * indices, uint32_t n_vec, bool init_indices) {
    HVX_సదిశ zero_vec = Q6_V_vzero();
    HVX_సదిశ idx_vec = *(స్థిరరాశి HVX_సదిశ *) argsort_ramp_lut;

    HVX_సదిశ_Pred pred_all_1s = Q6_Q_vcmp_eq_VwVw(zero_vec, zero_vec);
    HVX_సదిశ_Pred pred_all_0s = Q6_Q_not_Q(pred_all_1s);

    ఒకవేళ (init_indices) {
        కొరకు (uint32_t v = 0; v < n_vec; v++) {
            HVX_సదిశ idx = Q6_Vw_vadd_VwVw(idx_vec, Q6_V_vsplat_R(v * 32));
            *(HVX_సదిశ *)(indices + v * 128) = idx;
        }
    }

    పూర్ణాంకం M = 5;
    అయితే ((1u << (M - 5)) < n_vec) M++;

    కొరకు (పూర్ణాంకం s = 1; s <= M; s++) {
        కొరకు (పూర్ణాంకం stage_d = s - 1; stage_d >= 0; stage_d--) {
            పూర్ణాంకం d = 1 << stage_d;
            ఒకవేళ (d >= 32) {
                uint32_t v_dist = d / 32;
                కొరకు (uint32_t v1 = 0; v1 < n_vec; v1++) {
                    ఒకవేళ ((v1 & v_dist) == 0) {
                        uint32_t v2 = v1 + v_dist;
                        bool asc = (s < M) ? ((((v1 * 32) >> s) % 2) == 0) : false;

                        HVX_సదిశ Vv1 = *(HVX_సదిశ *)(విలువలు + v1 * 128);
                        HVX_సదిశ Iv1 = *(HVX_సదిశ *)(indices + v1 * 128);
                        HVX_సదిశ Vv2 = *(HVX_సదిశ *)(విలువలు + v2 * 128);
                        HVX_సదిశ Iv2 = *(HVX_సదిశ *)(indices + v2 * 128);

                        vec_cas(&Vv1, &Iv1, &Vv2, &Iv2, asc);

                        *(HVX_సదిశ *)(విలువలు + v1 * 128)  = Vv1;
                        *(HVX_సదిశ *)(indices + v1 * 128) = Iv1;
                        *(HVX_సదిశ *)(విలువలు + v2 * 128)  = Vv2;
                        *(HVX_సదిశ *)(indices + v2 * 128) = Iv2;
                    }
                }
            } లేకపోతే {
                ఒకవేళ (s < 5) {
                    HVX_సదిశ_Pred dir_ముసుగు = Q6_Q_vcmp_eq_VwVw(Q6_V_vand_VV(idx_vec, Q6_V_vsplat_R(1 << s)), zero_vec);
                    కొరకు (uint32_t v = 0; v < n_vec; v++) {
                        HVX_సదిశ Vv = *(HVX_సదిశ *)(విలువలు + v * 128);
                        HVX_సదిశ Iv = *(HVX_సదిశ *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_ముసుగు, idx_vec, zero_vec);

                        *(HVX_సదిశ *)(విలువలు + v * 128)  = Vv;
                        *(HVX_సదిశ *)(indices + v * 128) = Iv;
                    }
                } లేకపోతే {
                    కొరకు (uint32_t v = 0; v < n_vec; v++) {
                        bool asc = (s < M) ? ((((v * 32) >> s) % 2) == 0) : false;
                        HVX_సదిశ_Pred dir_ముసుగు = asc ? pred_all_1s : pred_all_0s;

                        HVX_సదిశ Vv = *(HVX_సదిశ *)(విలువలు + v * 128);
                        HVX_సదిశ Iv = *(HVX_సదిశ *)(indices + v * 128);

                        bitonic_cas_32(&Vv, &Iv, d, dir_ముసుగు, idx_vec, zero_vec);

                        *(HVX_సదిశ *)(విలువలు + v * 128)  = Vv;
                        *(HVX_సదిశ *)(indices + v * 128) = Iv;
                    }
                }
            }
        }
    }
}

స్థిర శూన్య శిఖరం_k_select_tiled(dma_వరుస * dma_q, నిర్మాణం htp_thread_trace * tr,
                               dma_addr_t src_addr, uint32_t n, uint32_t k,
                               దశాంశం * విలువలు_buf, int32_t * indices_buf,
                               దశాంశం * బహిర్గతం_విలువలు, int32_t * బహిర్గతం_indices,
                               uint32_t chunk_పునాది) {
    స్థిరరాశి uint32_t tile_elems = 1024;
    uint32_t n_tiles            = (n + tile_elems - 1) / tile_elems;
    uint32_t candidate_లెక్క    = n_tiles * k;
    uint32_t విలీనం_n_vec        = hmx_ceil_div(candidate_లెక్క, 32);
    uint32_t విలీనం_n_vec_pow2   = 1;
    అయితే (విలీనం_n_vec_pow2 < విలీనం_n_vec) విలీనం_n_vec_pow2 <<= 1;
    uint32_t విలీనం_elems        = విలీనం_n_vec_pow2 * 32;
    దశాంశం * candidate_విలువలు    = విలువలు_buf + tile_elems;
    int32_t * candidate_indices = indices_buf + tile_elems;
    uint32_t candidate_pos      = 0;

    కొరకు (uint32_t స్థానభ్రంశం = 0; స్థానభ్రంశం < n; స్థానభ్రంశం += tile_elems) {
        uint32_t tile_లెక్క = MIN(tile_elems, n - స్థానభ్రంశం);
        dma_sync_read(dma_q, విలువలు_buf, src_addr + (size_t) స్థానభ్రంశం * పరిమాణం(దశాంశం), tile_లెక్క * పరిమాణం(దశాంశం));

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) స్థానభ్రంశం);
        ఒకవేళ (tile_లెక్క < tile_elems) {
            hvx_splat_f32_u((uint8_t *) (విలువలు_buf + tile_లెక్క), -INFINITY, tile_elems - tile_లెక్క);
        }

        bitonic_క్రమీకరించు_vtcm_desc((uint8_t *) విలువలు_buf, (uint8_t *) indices_buf, tile_elems / 32, true);
        uint32_t tile_k = MIN(k, tile_లెక్క);
        కొరకు (uint32_t j = 0; j < tile_k; j++) {
            candidate_విలువలు[candidate_pos] = విలువలు_buf[j];
            candidate_indices[candidate_pos] = indices_buf[j] + (int32_t) స్థానభ్రంశం;
            candidate_pos++;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) స్థానభ్రంశం);
    }

    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, 0);
    ఒకవేళ (విలీనం_elems > candidate_pos) {
        hvx_splat_f32_u((uint8_t *) (candidate_విలువలు + candidate_pos), -INFINITY, విలీనం_elems - candidate_pos);
        కొరకు (uint32_t j = candidate_pos; j < విలీనం_elems; j++) {
            candidate_indices[j] = 0;
        }
    }

    bitonic_క్రమీకరించు_vtcm_desc((uint8_t *) candidate_విలువలు, (uint8_t *) candidate_indices, విలీనం_n_vec_pow2, false);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);

    కొరకు (uint32_t j = 0; j < k; j++) {
        బహిర్గతం_విలువలు[j]  = candidate_విలువలు[j];
        బహిర్గతం_indices[j] = candidate_indices[j] + (int32_t) chunk_పునాది;
    }
}

__attribute__((always_inline))
స్థిర అంతఃవరుస శూన్య sort32_f32_hvx(uint8_t * విలువలు, uint8_t * indices, గణన ggml_క్రమీకరించు_క్రమం క్రమం) {
    bitonic_క్రమీకరించు_generic_hvx(విలువలు, indices, 1, క్రమం == GGML_క్రమీకరించు_క్రమం_ASC);
}

__attribute__((always_inline))
స్థిర అంతఃవరుస శూన్య sort64_f32_hvx(uint8_t * విలువలు, uint8_t * indices, గణన ggml_క్రమీకరించు_క్రమం క్రమం) {
    bitonic_క్రమీకరించు_generic_hvx(విలువలు, indices, 2, క్రమం == GGML_క్రమీకరించు_క్రమం_ASC);
}

__attribute__((always_inline))
స్థిర అంతఃవరుస శూన్య sort128_f32_hvx(uint8_t * విలువలు, uint8_t * indices, గణన ggml_క్రమీకరించు_క్రమం క్రమం) {
    bitonic_క్రమీకరించు_generic_hvx(విలువలు, indices, 4, క్రమం == GGML_క్రమీకరించు_క్రమం_ASC);
}

__attribute__((always_inline))
స్థిర అంతఃవరుస శూన్య sort256_f32_hvx(uint8_t * విలువలు, uint8_t * indices, గణన ggml_క్రమీకరించు_క్రమం క్రమం) {
    bitonic_క్రమీకరించు_generic_hvx(విలువలు, indices, 8, క్రమం == GGML_క్రమీకరించు_క్రమం_ASC);
}

__attribute__((always_inline))
స్థిర అంతఃవరుస శూన్య sort512_f32_hvx(uint8_t * విలువలు, uint8_t * indices, గణన ggml_క్రమీకరించు_క్రమం క్రమం) {
    bitonic_క్రమీకరించు_generic_hvx(విలువలు, indices, 16, క్రమం == GGML_క్రమీకరించు_క్రమం_ASC);
}

__attribute__((always_inline))
స్థిర అంతఃవరుస శూన్య sort1024_f32_hvx(uint8_t * విలువలు, uint8_t * indices, గణన ggml_క్రమీకరించు_క్రమం క్రమం) {
    bitonic_క్రమీకరించు_generic_hvx(విలువలు, indices, 32, క్రమం == GGML_క్రమీకరించు_క్రమం_ASC);
}

స్థిర శూన్య విలీనం_runs(
    స్థిరరాశి దశాంశం * పరిమిత లోపల_val0, స్థిరరాశి int32_t * పరిమిత లోపల_idx0, uint32_t n0,
    స్థిరరాశి దశాంశం * పరిమిత లోపల_val1, స్థిరరాశి int32_t * పరిమిత లోపల_idx1, uint32_t n1,
    దశాంశం * పరిమిత బహిర్గతం_val, int32_t * పరిమిత బహిర్గతం_idx,
    bool asc
) {
    uint32_t i0 = 0;
    uint32_t i1 = 0;
    uint32_t బహిర్గతం = 0;

    ఒకవేళ (asc) {
        అయితే (i0 < n0 && i1 < n1) {
            ఒకవేళ (లోపల_val0[i0] <= లోపల_val1[i1]) {
                ఒకవేళ (బహిర్గతం_val) బహిర్గతం_val[బహిర్గతం] = లోపల_val0[i0];
                బహిర్గతం_idx[బహిర్గతం] = లోపల_idx0[i0];
                i0++;
            } లేకపోతే {
                ఒకవేళ (బహిర్గతం_val) బహిర్గతం_val[బహిర్గతం] = లోపల_val1[i1];
                బహిర్గతం_idx[బహిర్గతం] = లోపల_idx1[i1];
                i1++;
            }
            బహిర్గతం++;
        }
    } లేకపోతే {
        అయితే (i0 < n0 && i1 < n1) {
            ఒకవేళ (లోపల_val0[i0] >= లోపల_val1[i1]) {
                ఒకవేళ (బహిర్గతం_val) బహిర్గతం_val[బహిర్గతం] = లోపల_val0[i0];
                బహిర్గతం_idx[బహిర్గతం] = లోపల_idx0[i0];
                i0++;
            } లేకపోతే {
                ఒకవేళ (బహిర్గతం_val) బహిర్గతం_val[బహిర్గతం] = లోపల_val1[i1];
                బహిర్గతం_idx[బహిర్గతం] = లోపల_idx1[i1];
                i1++;
            }
            బహిర్గతం++;
        }
    }

    అయితే (i0 < n0) {
        ఒకవేళ (బహిర్గతం_val) బహిర్గతం_val[బహిర్గతం] = లోపల_val0[i0];
        బహిర్గతం_idx[బహిర్గతం] = లోపల_idx0[i0];
        i0++;
        బహిర్గతం++;
    }

    అయితే (i1 < n1) {
        ఒకవేళ (బహిర్గతం_val) బహిర్గతం_val[బహిర్గతం] = లోపల_val1[i1];
        బహిర్గతం_idx[బహిర్గతం] = లోపల_idx1[i1];
        i1++;
        బహిర్గతం++;
    }
}

నిర్మాణం htp_క్రమీకరించు_chunk_ctx {
    నిర్మాణం htp_ops_context *              octx;
    స్థిరరాశి నిర్మాణం htp_క్రమీకరించు_kernel_params * kparams;
    uint8_t *                             vtcm_పునాది;
    uint32_t                              real_లెక్క[HTP_MAX_NTHREADS];
};

స్థిర శూన్య htp_క్రమీకరించు_chunk_పని(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_క్రమీకరించు_chunk_ctx * cctx = (నిర్మాణం htp_క్రమీకరించు_chunk_ctx *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = cctx->octx;
    స్థిరరాశి నిర్మాణం htp_క్రమీకరించు_kernel_params * kparams = cctx->kparams;
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    uint32_t ne00        = (uint32_t) kparams->ne00;
    uint32_t chunk_పునాది  = i * chunk_elems;

    uint32_t real_లెక్క = (chunk_పునాది < ne00) ? MIN(chunk_elems, ne00 - chunk_పునాది) : 0;
    cctx->real_లెక్క[i] = real_లెక్క;

    uint8_t * spad = cctx->vtcm_పునాది + (size_t) kparams->phase1_స్థానం_కొలత * i;
    దశాంశం *   విలువలు_buf  = (దశాంశం *) spad;
    int32_t * indices_buf = (int32_t *) (spad + chunk_elems * పరిమాణం(దశాంశం));

    దశాంశం *   విలీనం_విలువలు  = (దశాంశం *)   (cctx->vtcm_పునాది + kparams->విలీనం_విలువలు_off);
    int32_t * విలీనం_indices = (int32_t *) (cctx->vtcm_పునాది + kparams->విలీనం_indices_off);

    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[i];
    స్థిరరాశి dma_addr_t src_addr = src0->దత్తాంశం + (size_t) chunk_పునాది * పరిమాణం(దశాంశం);

    ఒకవేళ (kparams->is_శిఖరం_k) {
        uint32_t k = (uint32_t) kparams->k;
        uint32_t local_k = MIN(k, chunk_elems);

        ఒకవేళ (real_లెక్క == 0) {
            కొరకు (uint32_t j = 0; j < local_k; j++) {
                విలీనం_విలువలు[i * local_k + j]  = -INFINITY;
                విలీనం_indices[i * local_k + j] = 0;
            }
            తిరిగి;
        }

        ఒకవేళ (local_k == 1 && ne00 >= 128*1024) {
            dma_sync_read(octx->ctx->dma[i], విలువలు_buf, src_addr, real_లెక్క * పరిమాణం(దశాంశం));

            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            దశాంశం max_విలువ;
            uint32_t max_సూచీ = శిఖరం_k_max_విలువ_సూచీ(విలువలు_buf, real_లెక్క, &max_విలువ);
            విలీనం_విలువలు[i] = max_విలువ;
            విలీనం_indices[i] = (int32_t) (max_సూచీ + chunk_పునాది);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            తిరిగి;
        }

        ఒకవేళ (local_k > 1 && local_k <= 64 && chunk_elems > 1024) {
            శిఖరం_k_select_tiled(octx->ctx->dma[i], tr, src_addr, real_లెక్క, local_k,
                               విలువలు_buf, indices_buf,
                               &విలీనం_విలువలు[i * local_k], &విలీనం_indices[i * local_k],
                               chunk_పునాది);
            తిరిగి;
        }

        dma_sync_read(octx->ctx->dma[i], విలువలు_buf, src_addr, real_లెక్క * పరిమాణం(దశాంశం));

        htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        ఒకవేళ (chunk_elems > real_లెక్క) {
            hvx_splat_f32_u((uint8_t *)(విలువలు_buf + real_లెక్క), -INFINITY, chunk_elems - real_లెక్క);
        }

        bitonic_క్రమీకరించు_vtcm_desc((uint8_t *) విలువలు_buf, (uint8_t *) indices_buf, chunk_elems / 32, true);

        కొరకు (uint32_t j = 0; j < local_k; j++) {
            విలీనం_విలువలు[i * local_k + j]  = విలువలు_buf[j];
            విలీనం_indices[i * local_k + j] = indices_buf[j] + (int32_t) chunk_పునాది;
        }
        htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    } లేకపోతే {
        ఒకవేళ (real_లెక్క > 0) {
            dma_sync_read(octx->ctx->dma[i], విలువలు_buf, src_addr, real_లెక్క * పరిమాణం(దశాంశం));

            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
            init_indices_ramp(indices_buf, real_లెక్క, chunk_పునాది);

            ఒకవేళ (kparams->క్రమం == GGML_క్రమీకరించు_క్రమం_ASC) {
                quicksort_విలువలు_indices_asc(విలువలు_buf, indices_buf, 0, real_లెక్క - 1);
            } లేకపోతే {
                quicksort_విలువలు_indices_desc(విలువలు_buf, indices_buf, 0, real_లెక్క - 1);
            }
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
        }
    }
}

నిర్మాణం htp_argsort_విలీనం_l1_ctx {
    నిర్మాణం htp_క్రమీకరించు_chunk_ctx * cctx;
    bool                        asc;
};

స్థిర శూన్య htp_argsort_విలీనం_l1_పని(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_argsort_విలీనం_l1_ctx * mctx = (నిర్మాణం htp_argsort_విలీనం_l1_ctx *) దత్తాంశం;
    నిర్మాణం htp_క్రమీకరించు_chunk_ctx * cctx = mctx->cctx;
    స్థిరరాశి నిర్మాణం htp_క్రమీకరించు_kernel_params * kparams = cctx->kparams;

    uint32_t c0 = i * 2;
    uint32_t c1 = c0 + 1;

    uint32_t count0 = cctx->real_లెక్క[c0];
    uint32_t count1 = cctx->real_లెక్క[c1];

    size_t స్థానం_కొలత = (size_t) kparams->phase1_స్థానం_కొలత;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    స్థిరరాశి దశాంశం *   val0 = (స్థిరరాశి దశాంశం *)   (cctx->vtcm_పునాది + స్థానం_కొలత * c0);
    స్థిరరాశి int32_t * idx0 = (స్థిరరాశి int32_t *) (cctx->vtcm_పునాది + స్థానం_కొలత * c0 + chunk_elems * పరిమాణం(దశాంశం));
    స్థిరరాశి దశాంశం *   val1 = (స్థిరరాశి దశాంశం *)   (cctx->vtcm_పునాది + స్థానం_కొలత * c1);
    స్థిరరాశి int32_t * idx1 = (స్థిరరాశి int32_t *) (cctx->vtcm_పునాది + స్థానం_కొలత * c1 + chunk_elems * పరిమాణం(దశాంశం));

    uint32_t బహిర్గతం_స్థానభ్రంశం = (2 * i) * chunk_elems;
    దశాంశం *   బహిర్గతం_val = (దశాంశం *)   (cctx->vtcm_పునాది + kparams->విలీనం_విలువలు_off)  + బహిర్గతం_స్థానభ్రంశం;
    int32_t * బహిర్గతం_idx = (int32_t *) (cctx->vtcm_పునాది + kparams->విలీనం_indices_off) + బహిర్గతం_స్థానభ్రంశం;

    నిర్మాణం htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    విలీనం_runs(val0, idx0, count0, val1, idx1, count1, బహిర్గతం_val, బహిర్గతం_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

నిర్మాణం htp_argsort_విలీనం_l2_ctx {
    నిర్మాణం htp_క్రమీకరించు_chunk_ctx * cctx;
    bool                        asc;
};

స్థిర శూన్య htp_argsort_విలీనం_l2_పని(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం i, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_argsort_విలీనం_l2_ctx * mctx = (నిర్మాణం htp_argsort_విలీనం_l2_ctx *) దత్తాంశం;
    నిర్మాణం htp_క్రమీకరించు_chunk_ctx * cctx = mctx->cctx;
    స్థిరరాశి నిర్మాణం htp_క్రమీకరించు_kernel_params * kparams = cctx->kparams;

    size_t స్థానం_కొలత = (size_t) kparams->phase1_స్థానం_కొలత;
    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

    uint32_t off0 = (4 * i) * chunk_elems;
    uint32_t off1 = (4 * i + 2) * chunk_elems;

    స్థిరరాశి దశాంశం *   val0 = (స్థిరరాశి దశాంశం *)   (cctx->vtcm_పునాది + kparams->విలీనం_విలువలు_off)  + off0;
    స్థిరరాశి int32_t * idx0 = (స్థిరరాశి int32_t *) (cctx->vtcm_పునాది + kparams->విలీనం_indices_off) + off0;
    స్థిరరాశి దశాంశం *   val1 = (స్థిరరాశి దశాంశం *)   (cctx->vtcm_పునాది + kparams->విలీనం_విలువలు_off)  + off1;
    స్థిరరాశి int32_t * idx1 = (స్థిరరాశి int32_t *) (cctx->vtcm_పునాది + kparams->విలీనం_indices_off) + off1;

    uint32_t count0 = cctx->real_లెక్క[4*i + 0] + cctx->real_లెక్క[4*i + 1];
    uint32_t count1 = cctx->real_లెక్క[4*i + 2] + cctx->real_లెక్క[4*i + 3];

    uint8_t * స్థానం_పునాది = cctx->vtcm_పునాది + (4 * i) * స్థానం_కొలత;
    దశాంశం *   బహిర్గతం_val   = (దశాంశం *) స్థానం_పునాది;
    int32_t * బహిర్గతం_idx   = (int32_t *) (స్థానం_పునాది + 4 * chunk_elems * పరిమాణం(దశాంశం));

    నిర్మాణం htp_thread_trace * tr = &cctx->octx->ctx->trace[i];
    htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
    విలీనం_runs(val0, idx0, count0, val1, idx1, count1, బహిర్గతం_val, బహిర్గతం_idx, mctx->asc);
    htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) i);
}

నిర్మాణం htp_క్రమీకరించు_multi_అడ్డువరుస_ctx {
    నిర్మాణం htp_ops_context *              octx;
    స్థిరరాశి నిర్మాణం htp_క్రమీకరించు_kernel_params * kparams;
    uint8_t *                             vtcm_పునాది;
    uint32_t                              nrows_చొప్పున_thread;
};

నిర్మాణం htp_tensor_అడ్డువరుస_coord {
    uint32_t i01;
    uint32_t i02;
    uint32_t i03;
};

స్థిర అంతఃవరుస నిర్మాణం htp_tensor_అడ్డువరుస_coord init_అడ్డువరుస_coord(uint32_t r, uint32_t ne01, uint32_t ne02) {
    uint32_t ne01_ne02 = ne01 * ne02;
    uint32_t i03 = (ne01_ne02 > 0) ? (r / ne01_ne02) : 0;
    uint32_t rem = (ne01_ne02 > 0) ? (r % ne01_ne02) : 0;
    uint32_t i02 = (ne01 > 0) ? (rem / ne01) : 0;
    uint32_t i01 = (ne01 > 0) ? (rem % ne01) : 0;
    తిరిగి (నిర్మాణం htp_tensor_అడ్డువరుస_coord){ i01, i02, i03 };
}

స్థిర అంతఃవరుస శూన్య అడుగు_అడ్డువరుస_coord(నిర్మాణం htp_tensor_అడ్డువరుస_coord * c, uint32_t ne01, uint32_t ne02) {
    c->i01++;
    ఒకవేళ (c->i01 == ne01) {
        c->i01 = 0;
        c->i02++;
        ఒకవేళ (c->i02 == ne02) {
            c->i02 = 0;
            c->i03++;
        }
    }
}

స్థిర అంతఃవరుస size_t coord_స్థానభ్రంశం(నిర్మాణం htp_tensor_అడ్డువరుస_coord c, size_t nb1, size_t nb2, size_t nb3) {
    తిరిగి (size_t) c.i01 * nb1 + (size_t) c.i02 * nb2 + (size_t) c.i03 * nb3;
}

స్థిర అంతఃవరుస శూన్య compute_అడ్డువరుస_క్రమీకరించు(
    దశాంశం * cur_విలువలు,
    int32_t * cur_indices,
    uint32_t ne00,
    uint32_t chunk_elems,
    bool is_శిఖరం_k,
    గణన ggml_క్రమీకరించు_క్రమం క్రమం
) {
    ఒకవేళ (is_శిఖరం_k) {
        ఒకవేళ (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            ఎంపిక (ne00) {
                స్థితి 32:   sort32_f32_hvx((uint8_t *) cur_విలువలు, (uint8_t *) cur_indices, GGML_క్రమీకరించు_క్రమం_DESC); ఆపు;
                స్థితి 64:   sort64_f32_hvx((uint8_t *) cur_విలువలు, (uint8_t *) cur_indices, GGML_క్రమీకరించు_క్రమం_DESC); ఆపు;
                స్థితి 128:  sort128_f32_hvx((uint8_t *) cur_విలువలు, (uint8_t *) cur_indices, GGML_క్రమీకరించు_క్రమం_DESC); ఆపు;
                స్థితి 256:  sort256_f32_hvx((uint8_t *) cur_విలువలు, (uint8_t *) cur_indices, GGML_క్రమీకరించు_క్రమం_DESC); ఆపు;
                స్థితి 512:  sort512_f32_hvx((uint8_t *) cur_విలువలు, (uint8_t *) cur_indices, GGML_క్రమీకరించు_క్రమం_DESC); ఆపు;
                స్థితి 1024: sort1024_f32_hvx((uint8_t *) cur_విలువలు, (uint8_t *) cur_indices, GGML_క్రమీకరించు_క్రమం_DESC); ఆపు;
            }
        } లేకపోతే {
            ఒకవేళ (chunk_elems > ne00) {
                hvx_splat_f32_u((uint8_t *)(cur_విలువలు + ne00), -INFINITY, chunk_elems - ne00);
            }
            bitonic_క్రమీకరించు_vtcm_desc((uint8_t *) cur_విలువలు, (uint8_t *) cur_indices, chunk_elems / 32, true);
        }
    } లేకపోతే {
        ఒకవేళ (ne00 == 32 || ne00 == 64 || ne00 == 128 || ne00 == 256 || ne00 == 512 || ne00 == 1024) {
            ఎంపిక (ne00) {
                స్థితి 32:   sort32_f32_hvx((uint8_t *) cur_విలువలు, (uint8_t *) cur_indices, క్రమం); ఆపు;
                స్థితి 64:   sort64_f32_hvx((uint8_t *) cur_విలువలు, (uint8_t *) cur_indices, క్రమం); ఆపు;
                స్థితి 128:  sort128_f32_hvx((uint8_t *) cur_విలువలు, (uint8_t *) cur_indices, క్రమం); ఆపు;
                స్థితి 256:  sort256_f32_hvx((uint8_t *) cur_విలువలు, (uint8_t *) cur_indices, క్రమం); ఆపు;
                స్థితి 512:  sort512_f32_hvx((uint8_t *) cur_విలువలు, (uint8_t *) cur_indices, క్రమం); ఆపు;
                స్థితి 1024: sort1024_f32_hvx((uint8_t *) cur_విలువలు, (uint8_t *) cur_indices, క్రమం); ఆపు;
            }
        } లేకపోతే {
            init_indices_ramp(cur_indices, ne00, 0);
            ఒకవేళ (క్రమం == GGML_క్రమీకరించు_క్రమం_ASC) {
                quicksort_విలువలు_indices_asc(cur_విలువలు, cur_indices, 0, ne00 - 1);
            } లేకపోతే {
                quicksort_విలువలు_indices_desc(cur_విలువలు, cur_indices, 0, ne00 - 1);
            }
        }
    }
}

స్థిర శూన్య htp_క్రమీకరించు_multi_అడ్డువరుస_పని(గుర్తులేని పూర్ణాంకం n, గుర్తులేని పూర్ణాంకం ith, శూన్య * దత్తాంశం) {
    నిర్మాణం htp_క్రమీకరించు_multi_అడ్డువరుస_ctx * mctx = (నిర్మాణం htp_క్రమీకరించు_multi_అడ్డువరుస_ctx *) దత్తాంశం;
    నిర్మాణం htp_ops_context * octx = mctx->octx;
    స్థిరరాశి నిర్మాణం htp_క్రమీకరించు_kernel_params * kparams = mctx->kparams;
    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    uint32_t ne00 = (uint32_t) kparams->ne00;
    uint32_t ne01 = src0->ne[1];
    uint32_t ne02 = src0->ne[2];

    uint32_t nb01 = src0->nb[1];
    uint32_t nb02 = src0->nb[2];
    uint32_t nb03 = src0->nb[3];

    uint32_t nb1 = dst->nb[1];
    uint32_t nb2 = dst->nb[2];
    uint32_t nb3 = dst->nb[3];

    uint32_t అడ్డువరుస_ప్రారంభం = (uint32_t) kparams->అడ్డువరుస_ప్రారంభం;
    uint32_t అడ్డువరుస_చివర   = (uint32_t) kparams->అడ్డువరుస_చివర;
    uint32_t r_ప్రారంభం   = అడ్డువరుస_ప్రారంభం + ith * mctx->nrows_చొప్పున_thread;
    uint32_t r_చివర     = MIN(r_ప్రారంభం + mctx->nrows_చొప్పున_thread, అడ్డువరుస_చివర);

    ఒకవేళ (r_ప్రారంభం >= r_చివర) తిరిగి;

    size_t spad_స్థానభ్రంశం = ith * (size_t) (kparams->phase1_స్థానం_కొలత * kparams->n_slots);
    uint8_t * thread_spad = mctx->vtcm_పునాది + spad_స్థానభ్రంశం;
    size_t స్థానం_కొలత = (size_t) kparams->phase1_స్థానం_కొలత;

    uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
    దశాంశం *   విలువలు_buf  = (దశాంశం *) thread_spad;
    int32_t * indices_buf = (int32_t *) (thread_spad + chunk_elems * పరిమాణం(దశాంశం));

    dma_వరుస * dma_q = octx->ctx->dma[ith];
    నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[ith];

    bool is_శిఖరం_k = kparams->is_శిఖరం_k != 0;
    uint32_t k = (uint32_t) kparams->k;
    గణన ggml_క్రమీకరించు_క్రమం క్రమం = (గణన ggml_క్రమీకరించు_క్రమం) kparams->క్రమం;

    స్థిరరాశి uint32_t src_బైట్లు = ne00 * పరిమాణం(దశాంశం);
    స్థిరరాశి uint32_t dst_బైట్లు = (is_శిఖరం_k ? k : ne00) * పరిమాణం(int32_t);

    ఒకవేళ (kparams->n_slots == 2) {
        నిర్మాణం htp_tensor_అడ్డువరుస_coord ప్రధానసంఖ్య_coord = init_అడ్డువరుస_coord(r_ప్రారంభం, ne01, ne02);
        కొరకు (uint32_t r = r_ప్రారంభం, spad_idx = 0; r < r_చివర && spad_idx < 2; r++, spad_idx++) {
            uint8_t * cur_spad = thread_spad + spad_idx * స్థానం_కొలత;
            దశాంశం *   cur_విలువలు  = (దశాంశం *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * పరిమాణం(దశాంశం));

            dma_addr_t dst_addr = dst->దత్తాంశం + coord_స్థానభ్రంశం(ప్రధానసంఖ్య_coord, nb1, nb2, nb3);
            dma_addr_t src_addr = src0->దత్తాంశం + coord_స్థానభ్రంశం(ప్రధానసంఖ్య_coord, nb01, nb02, nb03);

            // Dummy dst writeback to establish queue ordering
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_addr, cur_indices),
                           dst_బైట్లు, dst_బైట్లు, dst_బైట్లు, 0);

            // Prefetch input row
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(cur_విలువలు, src_addr),
                           src_బైట్లు, src_బైట్లు, src_బైట్లు, 1);

            అడుగు_అడ్డువరుస_coord(&ప్రధానసంఖ్య_coord, ne01, ne02);
        }

        నిర్మాణం htp_tensor_అడ్డువరుస_coord dst_coord = init_అడ్డువరుస_coord(r_ప్రారంభం, ne01, ne02);
        నిర్మాణం htp_tensor_అడ్డువరుస_coord తదుపరి_src_coord = init_అడ్డువరుస_coord(r_ప్రారంభం + 2, ne01, ne02);

        కొరకు (uint32_t r = r_ప్రారంభం; r < r_చివర; r++) {
            uint32_t cur_స్థానం = (r - r_ప్రారంభం) & 1;
            uint8_t * cur_spad = thread_spad + cur_స్థానం * స్థానం_కొలత;
            దశాంశం *   cur_విలువలు  = (దశాంశం *) cur_spad;
            int32_t * cur_indices = (int32_t *) (cur_spad + chunk_elems * పరిమాణం(దశాంశం));

            // Wait for previous writeback of this slot to complete
            dma_వరుస_తీయి(dma_q);

            // Wait for input row DMA read into this slot to complete
            dma_వరుస_తీయి(dma_q);

            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_అడ్డువరుస_క్రమీకరించు(cur_విలువలు, cur_indices, ne00, chunk_elems, is_శిఖరం_k, క్రమం);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            // Push writeback of current slot
            dma_addr_t dst_addr = dst->దత్తాంశం + coord_స్థానభ్రంశం(dst_coord, nb1, nb2, nb3);
            dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(dst_addr, cur_indices),
                           dst_బైట్లు, dst_బైట్లు, dst_బైట్లు, 1);
            అడుగు_అడ్డువరుస_coord(&dst_coord, ne01, ne02);

            // Prefetch next row into this slot
            స్థిరరాశి uint32_t తదుపరి_అడ్డువరుస = r + 2;
            ఒకవేళ (తదుపరి_అడ్డువరుస < r_చివర) {
                dma_addr_t తదుపరి_src_addr = src0->దత్తాంశం + coord_స్థానభ్రంశం(తదుపరి_src_coord, nb01, nb02, nb03);
                dma_వరుస_నెట్టు(dma_q, dma_make_దత్తాంశం(cur_విలువలు, తదుపరి_src_addr),
                               src_బైట్లు, src_బైట్లు, src_బైట్లు, 1);
                అడుగు_అడ్డువరుస_coord(&తదుపరి_src_coord, ne01, ne02);
            }
        }

        dma_వరుస_flush(dma_q);
    } లేకపోతే {
        నిర్మాణం htp_tensor_అడ్డువరుస_coord coord = init_అడ్డువరుస_coord(r_ప్రారంభం, ne01, ne02);
        కొరకు (uint32_t r = r_ప్రారంభం; r < r_చివర; r++) {
            dma_addr_t src_addr = src0->దత్తాంశం + coord_స్థానభ్రంశం(coord, nb01, nb02, nb03);
            dma_addr_t dst_addr = dst->దత్తాంశం + coord_స్థానభ్రంశం(coord, nb1, nb2, nb3);

            dma_sync_read(dma_q, విలువలు_buf, src_addr, src_బైట్లు);

            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);
            compute_అడ్డువరుస_క్రమీకరించు(విలువలు_buf, indices_buf, ne00, chunk_elems, is_శిఖరం_k, క్రమం);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, (uint16_t) r);

            dma_sync_write(dma_q, dst_addr, indices_buf, dst_బైట్లు);
            అడుగు_అడ్డువరుస_coord(&coord, ne01, ne02);
        }
    }
}

స్థిర పూర్ణాంకం op_క్రమీకరించు_common(నిర్మాణం htp_ops_context * octx, bool is_శిఖరం_k) {
    ఒకవేళ (octx->src[0]->type != HTP_TYPE_F32) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి నిర్మాణం htp_tensor * src0 = octx->src[0];
    స్థిరరాశి నిర్మాణం htp_tensor * dst  = octx->dst;

    ఒకవేళ (htp_tensor_is_extended(src0) || htp_tensor_is_extended(dst)) {
        తిరిగి HTP_స్థితి_NO_SUPPORT;
    }

    స్థిరరాశి uint32_t మొత్తం_అడ్డువరుసలు = src0->ne[1] * src0->ne[2] * src0->ne[3];
    స్థిరరాశి size_t dst_అడ్డువరుస_కొలత = dst->ne[0] * పరిమాణం(int32_t);

    uint32_t అడ్డువరుస_ప్రారంభం = 0;
    uint32_t అడ్డువరుస_చివర   = మొత్తం_అడ్డువరుసలు;
    ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        uint32_t అడ్డువరుసలు_చొప్పున_chunk = 0;
        htp_tensor_mdev_అడ్డువరుసలు_చొప్పున_chunk(dst, పరిమాణం(int32_t), (uint32_t) dst_అడ్డువరుస_కొలత, &అడ్డువరుసలు_చొప్పున_chunk);
        స్థిరరాశి నిర్మాణం htp_tensor_mdev_range range = htp_tensor_mdev_విభజన(
            మొత్తం_అడ్డువరుసలు, అడ్డువరుసలు_చొప్పున_chunk, octx->ctx->mdev.idx, octx->ctx->mdev.లెక్క, &octx->ctx->mdev.లెక్క_div);
        అడ్డువరుస_ప్రారంభం = range.ప్రారంభం;
        అడ్డువరుస_చివర   = range.ప్రారంభం + range.లెక్క;
    }

    స్థిరరాశి uint32_t nrows = అడ్డువరుస_చివర - అడ్డువరుస_ప్రారంభం;
    ఒకవేళ (nrows == 0) {
        తిరిగి HTP_స్థితి_OK;
    }

    uint32_t ne00 = src0->ne[0];
    uint32_t k    = dst->ne[0];

    నిర్మాణం htp_క్రమీకరించు_kernel_params kparams_local;
    స్థిరరాశి నిర్మాణం htp_క్రమీకరించు_kernel_params * kparams = (స్థిరరాశి నిర్మాణం htp_క్రమీకరించు_kernel_params *) octx->kernel_params;

    ఒకవేళ (kparams->n_దారాలు == 0) {
        నిర్మాణం htp_క్రమీకరించు_vtcm_layout layout;
        ఒకవేళ (!htp_క్రమీకరించు_solve_layout(&layout, ne00, మొత్తం_అడ్డువరుసలు, k, octx->n_దారాలు, octx->ctx->vtcm_కొలత, is_శిఖరం_k)) {
            తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
        }
        memset(&kparams_local, 0, పరిమాణం(kparams_local));
        kparams_local.n_దారాలు         = (int32_t) layout.n_దారాలు;
        kparams_local.మొత్తం_అడ్డువరుసలు        = (int32_t) మొత్తం_అడ్డువరుసలు;
        kparams_local.అడ్డువరుస_ప్రారంభం         = (int32_t) అడ్డువరుస_ప్రారంభం;
        kparams_local.అడ్డువరుస_చివర           = (int32_t) అడ్డువరుస_చివర;
        kparams_local.ne00              = (int32_t) ne00;
        kparams_local.k                 = (int32_t) k;
        kparams_local.క్రమం             = is_శిఖరం_k ? GGML_క్రమీకరించు_క్రమం_DESC : (int32_t) octx->op_params[0];
        kparams_local.is_శిఖరం_k          = is_శిఖరం_k ? 1 : 0;
        kparams_local.use_dma           = 1;
        kparams_local.chunk_elems       = (int32_t) layout.chunk_elems;
        kparams_local.n_chunks          = (int32_t) layout.n_chunks;
        kparams_local.vtcm_కొలత         = (int32_t) layout.మొత్తం_బైట్లు;
        kparams_local.phase1_స్థానం_కొలత  = (int32_t) layout.phase1_స్థానం_కొలత;
        kparams_local.విలీనం_విలువలు_off  = (int32_t) layout.విలీనం_విలువలు_off;
        kparams_local.విలీనం_indices_off = (int32_t) layout.విలీనం_indices_off;
        kparams_local.విలీనం_elems       = (int32_t) layout.విలీనం_elems;
        kparams_local.n_slots           = (int32_t) layout.n_slots;
        kparams = &kparams_local;
    } లేకపోతే ఒకవేళ (octx->ctx->mdev.లెక్క > 1) {
        జ్ఞాపకం_నకలు(&kparams_local, kparams, పరిమాణం(kparams_local));
        kparams_local.అడ్డువరుస_ప్రారంభం = (int32_t) అడ్డువరుస_ప్రారంభం;
        kparams_local.అడ్డువరుస_చివర   = (int32_t) అడ్డువరుస_చివర;
        kparams = &kparams_local;
    }

    ఒకవేళ (octx->ctx->vtcm_కొలత < (size_t) kparams->vtcm_కొలత) {
        FARF(ERROR, "sort: VTCM size too small. Needed %d, have %zu", kparams->vtcm_కొలత, octx->ctx->vtcm_కొలత);
        తిరిగి HTP_స్థితి_VTCM_TOO_SMALL;
    }

    uint8_t * vtcm_పునాది = (uint8_t *) octx->ctx->vtcm_పునాది;

    ఒకవేళ (మొత్తం_అడ్డువరుసలు == 1 && kparams->n_chunks > 1) {
        నిర్మాణం htp_క్రమీకరించు_chunk_ctx cctx;
        cctx.octx      = octx;
        cctx.kparams   = kparams;
        cctx.vtcm_పునాది = vtcm_పునాది;
        memset(cctx.real_లెక్క, 0, పరిమాణం(cctx.real_లెక్క));

        work_వరుస_నిరంతరం(octx->ctx->work_వరుస, htp_క్రమీకరించు_chunk_పని, &cctx, (uint32_t) kparams->n_chunks);

        ఒకవేళ (is_శిఖరం_k) {
            uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
            uint32_t local_k     = MIN(k, chunk_elems);

            ఒకవేళ (k == 1 && ne00 >= 128*1024) {
                దశాంశం *   cand_vals = (దశాంశం *)   (vtcm_పునాది + kparams->విలీనం_విలువలు_off);
                int32_t * cand_idxs = (int32_t *) (vtcm_పునాది + kparams->విలీనం_indices_off);
                దశాంశం ఉత్తమ_val = cand_vals[0];
                int32_t ఉత్తమ_idx = cand_idxs[0];
                కొరకు (int32_t c = 1; c < kparams->n_chunks; c++) {
                    ఒకవేళ (cand_vals[c] > ఉత్తమ_val) {
                        ఉత్తమ_val = cand_vals[c];
                        ఉత్తమ_idx = cand_idxs[c];
                    }
                }
                int32_t * dst_ptr = (int32_t *) dst->దత్తాంశం;
                dst_ptr[0] = ఉత్తమ_idx;
                తిరిగి HTP_స్థితి_OK;
            }

            uint32_t మొత్తం_candidates = (uint32_t) kparams->n_chunks * local_k;
            దశాంశం *   cand_vals = (దశాంశం *)   (vtcm_పునాది + kparams->విలీనం_విలువలు_off);
            int32_t * cand_idxs = (int32_t *) (vtcm_పునాది + kparams->విలీనం_indices_off);
            uint32_t విలీనం_elems = (uint32_t) kparams->విలీనం_elems;

            ఒకవేళ (విలీనం_elems > మొత్తం_candidates) {
                hvx_splat_f32_u((uint8_t *) (cand_vals + మొత్తం_candidates), -INFINITY, విలీనం_elems - మొత్తం_candidates);
                కొరకు (uint32_t j = మొత్తం_candidates; j < విలీనం_elems; j++) {
                    cand_idxs[j] = 0;
                }
            }

            నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[0];
            htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            bitonic_క్రమీకరించు_vtcm_desc((uint8_t *) cand_vals, (uint8_t *) cand_idxs, విలీనం_elems / 32, false);
            htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
            dma_sync_write(octx->ctx->dma[0], dst->దత్తాంశం, cand_idxs, k * పరిమాణం(int32_t));
        } లేకపోతే {
            bool asc = kparams->క్రమం == GGML_క్రమీకరించు_క్రమం_ASC;
            ఒకవేళ (kparams->n_chunks == 8) {
                నిర్మాణం htp_argsort_విలీనం_l1_ctx l1ctx = { &cctx, asc };
                work_వరుస_నిరంతరం(octx->ctx->work_వరుస, htp_argsort_విలీనం_l1_పని, &l1ctx, 4);

                నిర్మాణం htp_argsort_విలీనం_l2_ctx l2ctx = { &cctx, asc };
                work_వరుస_నిరంతరం(octx->ctx->work_వరుస, htp_argsort_విలీనం_l2_పని, &l2ctx, 2);

                size_t స్థానం_కొలత = (size_t) kparams->phase1_స్థానం_కొలత;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                స్థిరరాశి దశాంశం *   val0 = (స్థిరరాశి దశాంశం *) vtcm_పునాది;
                స్థిరరాశి int32_t * idx0 = (స్థిరరాశి int32_t *) (vtcm_పునాది + 4 * chunk_elems * పరిమాణం(దశాంశం));
                స్థిరరాశి దశాంశం *   val1 = (స్థిరరాశి దశాంశం *) (vtcm_పునాది + 4 * స్థానం_కొలత);
                స్థిరరాశి int32_t * idx1 = (స్థిరరాశి int32_t *) (vtcm_పునాది + 4 * స్థానం_కొలత + 4 * chunk_elems * పరిమాణం(దశాంశం));

                uint32_t n01 = cctx.real_లెక్క[0] + cctx.real_లెక్క[1] + cctx.real_లెక్క[2] + cctx.real_లెక్క[3];
                uint32_t n23 = cctx.real_లెక్క[4] + cctx.real_లెక్క[5] + cctx.real_లెక్క[6] + cctx.real_లెక్క[7];

                int32_t * final_idx = (int32_t *) (vtcm_పునాది + kparams->విలీనం_indices_off);
                నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                విలీనం_runs(val0, idx0, n01, val1, idx1, n23, ఖాళీ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->దత్తాంశం, final_idx, ne00 * పరిమాణం(int32_t));
            } లేకపోతే ఒకవేళ (kparams->n_chunks == 4) {
                నిర్మాణం htp_argsort_విలీనం_l1_ctx l1ctx = { &cctx, asc };
                work_వరుస_నిరంతరం(octx->ctx->work_వరుస, htp_argsort_విలీనం_l1_పని, &l1ctx, 2);

                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                uint32_t n01 = cctx.real_లెక్క[0] + cctx.real_లెక్క[1];
                uint32_t n23 = cctx.real_లెక్క[2] + cctx.real_లెక్క[3];

                స్థిరరాశి దశాంశం *   val01 = (స్థిరరాశి దశాంశం *)   (vtcm_పునాది + kparams->విలీనం_విలువలు_off);
                స్థిరరాశి int32_t * idx01 = (స్థిరరాశి int32_t *) (vtcm_పునాది + kparams->విలీనం_indices_off);
                స్థిరరాశి దశాంశం *   val23 = (స్థిరరాశి దశాంశం *)   (vtcm_పునాది + kparams->విలీనం_విలువలు_off)  + 2 * chunk_elems;
                స్థిరరాశి int32_t * idx23 = (స్థిరరాశి int32_t *) (vtcm_పునాది + kparams->విలీనం_indices_off) + 2 * chunk_elems;

                int32_t * final_idx = (int32_t *) vtcm_పునాది;
                నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                విలీనం_runs(val01, idx01, n01, val23, idx23, n23, ఖాళీ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->దత్తాంశం, final_idx, ne00 * పరిమాణం(int32_t));
            } లేకపోతే ఒకవేళ (kparams->n_chunks == 2) {
                size_t స్థానం_కొలత = (size_t) kparams->phase1_స్థానం_కొలత;
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;

                స్థిరరాశి దశాంశం *   val0 = (స్థిరరాశి దశాంశం *)   (vtcm_పునాది);
                స్థిరరాశి int32_t * idx0 = (స్థిరరాశి int32_t *) (vtcm_పునాది + chunk_elems * పరిమాణం(దశాంశం));
                స్థిరరాశి దశాంశం *   val1 = (స్థిరరాశి దశాంశం *)   (vtcm_పునాది + స్థానం_కొలత);
                స్థిరరాశి int32_t * idx1 = (స్థిరరాశి int32_t *) (vtcm_పునాది + స్థానం_కొలత + chunk_elems * పరిమాణం(దశాంశం));

                int32_t * final_idx = (int32_t *) (vtcm_పునాది + kparams->విలీనం_indices_off);
                నిర్మాణం htp_thread_trace * tr = &octx->ctx->trace[0];
                htp_trace_event_ప్రారంభం(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                విలీనం_runs(val0, idx0, cctx.real_లెక్క[0], val1, idx1, cctx.real_లెక్క[1], ఖాళీ, final_idx, asc);
                htp_trace_event_stop(tr, HTP_TRACE_EVT_HVX_COMP, 0);
                dma_sync_write(octx->ctx->dma[0], dst->దత్తాంశం, final_idx, ne00 * పరిమాణం(int32_t));
            } లేకపోతే {
                uint32_t chunk_elems = (uint32_t) kparams->chunk_elems;
                int32_t * final_idx = (int32_t *) (vtcm_పునాది + chunk_elems * పరిమాణం(దశాంశం));
                dma_sync_write(octx->ctx->dma[0], dst->దత్తాంశం, final_idx, ne00 * పరిమాణం(int32_t));
            }
        }
        తిరిగి HTP_స్థితి_OK;
    }

    నిర్మాణం htp_క్రమీకరించు_multi_అడ్డువరుస_ctx mctx;
    mctx.octx             = octx;
    mctx.kparams          = kparams;
    mctx.vtcm_పునాది        = vtcm_పునాది;

    uint32_t dst_అడ్డువరుస_బైట్లు = (uint32_t) dst_అడ్డువరుస_కొలత;
    uint32_t అడ్డువరుసలు_చొప్పున_chunk = 0;
    htp_tensor_mdev_అడ్డువరుసలు_చొప్పున_chunk(dst, పరిమాణం(int32_t), dst_అడ్డువరుస_బైట్లు, &అడ్డువరుసలు_చొప్పున_chunk);
    ఒకవేళ (అడ్డువరుసలు_చొప్పున_chunk == 0) {
        అడ్డువరుసలు_చొప్పున_chunk = 1;
    }
    uint32_t dr = (nrows + (uint32_t) kparams->n_దారాలు - 1) / (uint32_t) kparams->n_దారాలు;
    ఒకవేళ (అడ్డువరుసలు_చొప్పున_chunk > 1) {
        dr = hex_విడత_పైకి(dr, అడ్డువరుసలు_చొప్పున_chunk);
    }
    mctx.nrows_చొప్పున_thread = dr;

    work_వరుస_నిరంతరం(octx->ctx->work_వరుస, htp_క్రమీకరించు_multi_అడ్డువరుస_పని, &mctx, (uint32_t) kparams->n_దారాలు);

    తిరిగి HTP_స్థితి_OK;
}

పూర్ణాంకం op_argsort(నిర్మాణం htp_ops_context * octx) {
    తిరిగి op_క్రమీకరించు_common(octx, false);
}

పూర్ణాంకం op_శిఖరం_k(నిర్మాణం htp_ops_context * octx) {
    తిరిగి op_క్రమీకరించు_common(octx, true);
}
