#నిర్వచితం_కాకపోతే HMX_FA_KERNELS_H
#నిర్వచించు HMX_FA_KERNELS_H

#అన్తర్భావ <stdint.h>
#అన్తర్భావ <stddef.h>
#అన్తర్భావ <stdbool.h>
#అన్తర్భావ "hvx-utils.h"
#అన్తర్భావ "hmx-utils.h"
#అన్తర్భావ "hex-fastdiv.h"

// HMX-specific parameters, offsets and inner kernels for Flash Attention

// Scatter offsets for diagonal tile: entry[2i] = i*136, entry[2i+1] = i*136+6
// 136 = 4 * 32 + 8 = byte offset to diagonal in a 32x32 fp16 interleaved tile
స్థిర స్థిరరాశి int16_t d_tile_scatter_offsets[64] __attribute__((aligned(128))) = {
    0 * 136,  0 * 136 + 6,
    1 * 136,  1 * 136 + 6,
    2 * 136,  2 * 136 + 6,
    3 * 136,  3 * 136 + 6,
    4 * 136,  4 * 136 + 6,
    5 * 136,  5 * 136 + 6,
    6 * 136,  6 * 136 + 6,
    7 * 136,  7 * 136 + 6,
    8 * 136,  8 * 136 + 6,
    9 * 136,  9 * 136 + 6,
    10 * 136, 10 * 136 + 6,
    11 * 136, 11 * 136 + 6,
    12 * 136, 12 * 136 + 6,
    13 * 136, 13 * 136 + 6,
    14 * 136, 14 * 136 + 6,
    15 * 136, 15 * 136 + 6,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
    0,        0,
};
// Inner HMX tile computation kernels

స్థిర అంతఃవరుస శూన్య hmx_fa_qk_dot_tile(
    స్థిరరాశి __fp16 * అడ్డువరుస_tiles,
    స్థిరరాశి __fp16 * నిలువు_tiles,
    __fp16 *       బహిర్గతం_tile,
    size_t         n_dot_tiles
) {
    ఒకవేళ (n_dot_tiles == 2) {
        asm అస్థిర(
            HMX_ఎక్కించు_MPY_F16("%1", "%2", "%0")
            HMX_ఎక్కించు_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(అడ్డువరుస_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(నిలువు_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(అడ్డువరుస_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(నిలువు_tiles + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } లేకపోతే ఒకవేళ (n_dot_tiles == 4) {
        asm అస్థిర(
            HMX_ఎక్కించు_MPY_F16("%1", "%2", "%0")
            HMX_ఎక్కించు_MPY_F16("%3", "%4", "%0")
            HMX_ఎక్కించు_MPY_F16("%5", "%6", "%0")
            HMX_ఎక్కించు_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(అడ్డువరుస_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(నిలువు_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(అడ్డువరుస_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(నిలువు_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(అడ్డువరుస_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(నిలువు_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(అడ్డువరుస_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(నిలువు_tiles + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } లేకపోతే ఒకవేళ (n_dot_tiles == 8) {
        asm అస్థిర(
            HMX_ఎక్కించు_MPY_F16("%1", "%2", "%0")
            HMX_ఎక్కించు_MPY_F16("%3", "%4", "%0")
            HMX_ఎక్కించు_MPY_F16("%5", "%6", "%0")
            HMX_ఎక్కించు_MPY_F16("%7", "%8", "%0")
            HMX_ఎక్కించు_MPY_F16("%9", "%10", "%0")
            HMX_ఎక్కించు_MPY_F16("%11", "%12", "%0")
            HMX_ఎక్కించు_MPY_F16("%13", "%14", "%0")
            HMX_ఎక్కించు_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(అడ్డువరుస_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(నిలువు_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(అడ్డువరుస_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(నిలువు_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(అడ్డువరుస_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(నిలువు_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(అడ్డువరుస_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(నిలువు_tiles + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(అడ్డువరుస_tiles + 4 * HMX_FP16_TILE_N_ELMS), "r"(నిలువు_tiles + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(అడ్డువరుస_tiles + 5 * HMX_FP16_TILE_N_ELMS), "r"(నిలువు_tiles + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(అడ్డువరుస_tiles + 6 * HMX_FP16_TILE_N_ELMS), "r"(నిలువు_tiles + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(అడ్డువరుస_tiles + 7 * HMX_FP16_TILE_N_ELMS), "r"(నిలువు_tiles + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } లేకపోతే {
        కొరకు (size_t k = 0; k < n_dot_tiles; ++k) {
            asm అస్థిర(
                HMX_ఎక్కించు_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(అడ్డువరుస_tiles), "r"(నిలువు_tiles)
            );
            అడ్డువరుస_tiles += HMX_FP16_TILE_N_ELMS;
            నిలువు_tiles += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm అస్థిర(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(బహిర్గతం_tile), "r"(0)
        : "memory"
    );
}

స్థిర అంతఃవరుస శూన్య hmx_fa_o_update_tile(
    స్థిరరాశి __fp16 * d_diag,
    స్థిరరాశి __fp16 * o_rc,
    స్థిరరాశి __fp16 * p_tile_లోపల,
    స్థిరరాశి __fp16 * v_tile_లోపల,
    __fp16 *       o_tile_బహిర్గతం,
    size_t         n_నిలువు_tiles
) {
    asm అస్థిర(
        HMX_ఎక్కించు_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    ఒకవేళ (n_నిలువు_tiles == 2) {
        asm అస్థిర(
            HMX_ఎక్కించు_MPY_F16("%1", "%2", "%0")
            HMX_ఎక్కించు_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(p_tile_లోపల + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_లోపల + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_లోపల + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_లోపల + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } లేకపోతే ఒకవేళ (n_నిలువు_tiles == 4) {
        asm అస్థిర(
            HMX_ఎక్కించు_MPY_F16("%1", "%2", "%0")
            HMX_ఎక్కించు_MPY_F16("%3", "%4", "%0")
            HMX_ఎక్కించు_MPY_F16("%5", "%6", "%0")
            HMX_ఎక్కించు_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(p_tile_లోపల + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_లోపల + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_లోపల + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_లోపల + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_లోపల + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_లోపల + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_లోపల + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_లోపల + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } లేకపోతే ఒకవేళ (n_నిలువు_tiles == 8) {
        asm అస్థిర(
            HMX_ఎక్కించు_MPY_F16("%1", "%2", "%0")
            HMX_ఎక్కించు_MPY_F16("%3", "%4", "%0")
            HMX_ఎక్కించు_MPY_F16("%5", "%6", "%0")
            HMX_ఎక్కించు_MPY_F16("%7", "%8", "%0")
            HMX_ఎక్కించు_MPY_F16("%9", "%10", "%0")
            HMX_ఎక్కించు_MPY_F16("%11", "%12", "%0")
            HMX_ఎక్కించు_MPY_F16("%13", "%14", "%0")
            HMX_ఎక్కించు_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(p_tile_లోపల + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_లోపల + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_లోపల + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_లోపల + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_లోపల + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_లోపల + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_లోపల + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_లోపల + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_లోపల + 4 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_లోపల + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_లోపల + 5 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_లోపల + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_లోపల + 6 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_లోపల + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_లోపల + 7 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_లోపల + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } లేకపోతే {
        కొరకు (size_t k = 0; k < n_నిలువు_tiles; ++k) {
            asm అస్థిర(
                HMX_ఎక్కించు_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(p_tile_లోపల), "r"(v_tile_లోపల)
            );
            p_tile_లోపల += HMX_FP16_TILE_N_ELMS;
            v_tile_లోపల += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm అస్థిర(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_tile_బహిర్గతం), "r"(0)
        : "memory"
    );
}

స్థిర అంతఃవరుస శూన్య hmx_fa_o_norm_tile(
    స్థిరరాశి __fp16 * d_diag,
    స్థిరరాశి __fp16 * o_rc,
    __fp16 *       o_బహిర్గతం
) {
    asm అస్థిర(
        HMX_ఎక్కించు_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    asm అస్థిర(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_బహిర్గతం), "r"(0)
        : "memory"
    );
}

స్థిర అంతఃవరుస శూన్య hmx_fa_q_prep_fp32_d2(
    __fp16 * vtcm_q_tiles, స్థిరరాశి uint8_t * తాత్కాలిక_q_vtcm,
    size_t ప్రారంభం, size_t చివర, size_t g_అడ్డువరుసలు_చివర,
    size_t DK, size_t G, size_t n_అడ్డువరుసలు_q,
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_G, bool q_transposed
) {
    కొరకు (size_t r = ప్రారంభం; r < చివర; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
        size_t   r1       = r % HMX_FP16_TILE_N_అడ్డువరుసలు;
        __fp16 * బహిర్గతం_పునాది = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_అడ్డువరుసలు * DK;

        ఒకవేళ (r >= g_అడ్డువరుసలు_చివర) {
            ((HVX_సదిశ *) (బహిర్గతం_పునాది + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            ((HVX_సదిశ *) (బహిర్గతం_పునాది + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            కొనసాగు;
        }

        స్థిరరాశి size_t q_idx0 = fastdiv(r + 0, div_G);
        స్థిరరాశి size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        స్థిరరాశి size_t q_idx1 = fastdiv(r + 1, div_G);
        స్థిరరాశి size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        స్థిరరాశి size_t offset0 = q_transposed ? (h_idx0 * n_అడ్డువరుసలు_q + q_idx0) : (q_idx0 * G + h_idx0);
        స్థిరరాశి size_t offset1 = q_transposed ? (h_idx1 * n_అడ్డువరుసలు_q + q_idx1) : (q_idx1 * G + h_idx1);

        స్థిరరాశి HVX_సదిశ * pv_in0 = (స్థిరరాశి HVX_సదిశ *) (తాత్కాలిక_q_vtcm + offset0 * DK * పరిమాణం(దశాంశం));
        స్థిరరాశి HVX_సదిశ * pv_in1 = (r + 1 < g_అడ్డువరుసలు_చివర)
            ? (స్థిరరాశి HVX_సదిశ *) (తాత్కాలిక_q_vtcm + offset1 * DK * పరిమాణం(దశాంశం))
            : ఖాళీ;

        {
            HVX_సదిశ v0 = pv_in0[0];
            HVX_సదిశ v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_సదిశ v_hf = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);
            ((HVX_సదిశ *) (బహిర్గతం_పునాది + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
        {
            HVX_సదిశ v0 = pv_in0[1];
            HVX_సదిశ v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_సదిశ v_hf = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);
            ((HVX_సదిశ *) (బహిర్గతం_పునాది + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

స్థిర అంతఃవరుస శూన్య hmx_fa_q_prep_fp32_d4(
    __fp16 * vtcm_q_tiles, స్థిరరాశి uint8_t * తాత్కాలిక_q_vtcm,
    size_t ప్రారంభం, size_t చివర, size_t g_అడ్డువరుసలు_చివర,
    size_t DK, size_t G, size_t n_అడ్డువరుసలు_q,
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_G, bool q_transposed
) {
    కొరకు (size_t r = ప్రారంభం; r < చివర; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
        size_t   r1       = r % HMX_FP16_TILE_N_అడ్డువరుసలు;
        __fp16 * బహిర్గతం_పునాది = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_అడ్డువరుసలు * DK;

        ఒకవేళ (r >= g_అడ్డువరుసలు_చివర) {
            కొరకు (uint32_t d = 0; d < 4; ++d) {
                ((HVX_సదిశ *) (బహిర్గతం_పునాది + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            కొనసాగు;
        }

        స్థిరరాశి size_t q_idx0 = fastdiv(r + 0, div_G);
        స్థిరరాశి size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        స్థిరరాశి size_t q_idx1 = fastdiv(r + 1, div_G);
        స్థిరరాశి size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        స్థిరరాశి size_t offset0 = q_transposed ? (h_idx0 * n_అడ్డువరుసలు_q + q_idx0) : (q_idx0 * G + h_idx0);
        స్థిరరాశి size_t offset1 = q_transposed ? (h_idx1 * n_అడ్డువరుసలు_q + q_idx1) : (q_idx1 * G + h_idx1);

        స్థిరరాశి HVX_సదిశ * pv_in0 = (స్థిరరాశి HVX_సదిశ *) (తాత్కాలిక_q_vtcm + offset0 * DK * పరిమాణం(దశాంశం));
        స్థిరరాశి HVX_సదిశ * pv_in1 = (r + 1 < g_అడ్డువరుసలు_చివర)
            ? (స్థిరరాశి HVX_సదిశ *) (తాత్కాలిక_q_vtcm + offset1 * DK * పరిమాణం(దశాంశం))
            : ఖాళీ;

        కొరకు (uint32_t d = 0; d < 4; ++d) {
            HVX_సదిశ v0 = pv_in0[d];
            HVX_సదిశ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_సదిశ v_hf = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);
            ((HVX_సదిశ *) (బహిర్గతం_పునాది + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

స్థిర అంతఃవరుస శూన్య hmx_fa_q_prep_fp32(
    __fp16 * vtcm_q_tiles, స్థిరరాశి uint8_t * తాత్కాలిక_q_vtcm,
    size_t ప్రారంభం, size_t చివర, size_t g_అడ్డువరుసలు_చివర,
    size_t DK, size_t G, size_t n_అడ్డువరుసలు_q,
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_G, uint32_t d_హద్దు, bool q_transposed
) {
    కొరకు (size_t r = ప్రారంభం; r < చివర; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
        size_t   r1       = r % HMX_FP16_TILE_N_అడ్డువరుసలు;
        __fp16 * బహిర్గతం_పునాది = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_అడ్డువరుసలు * DK;

        ఒకవేళ (r >= g_అడ్డువరుసలు_చివర) {
            కొరకు (uint32_t d = 0; d < d_హద్దు; ++d) {
                ((HVX_సదిశ *) (బహిర్గతం_పునాది + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            కొనసాగు;
        }

        స్థిరరాశి size_t q_idx0 = fastdiv(r + 0, div_G);
        స్థిరరాశి size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        స్థిరరాశి size_t q_idx1 = fastdiv(r + 1, div_G);
        స్థిరరాశి size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        స్థిరరాశి size_t offset0 = q_transposed ? (h_idx0 * n_అడ్డువరుసలు_q + q_idx0) : (q_idx0 * G + h_idx0);
        స్థిరరాశి size_t offset1 = q_transposed ? (h_idx1 * n_అడ్డువరుసలు_q + q_idx1) : (q_idx1 * G + h_idx1);

        స్థిరరాశి HVX_సదిశ * pv_in0 = (స్థిరరాశి HVX_సదిశ *) (తాత్కాలిక_q_vtcm + offset0 * DK * పరిమాణం(దశాంశం));
        స్థిరరాశి HVX_సదిశ * pv_in1 = (r + 1 < g_అడ్డువరుసలు_చివర)
            ? (స్థిరరాశి HVX_సదిశ *) (తాత్కాలిక_q_vtcm + offset1 * DK * పరిమాణం(దశాంశం))
            : ఖాళీ;

        కొరకు (uint32_t d = 0; d < d_హద్దు; ++d) {
            HVX_సదిశ v0   = pv_in0[d];
            HVX_సదిశ v1   = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_సదిశ v_hf = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);

            HVX_సదిశ * బహిర్గతం_tile = (HVX_సదిశ *) (బహిర్గతం_పునాది + d * HMX_FP16_TILE_N_ELMS);
            బహిర్గతం_tile[r1 / 2]      = v_hf;
        }
    }
}

స్థిర అంతఃవరుస శూన్య hmx_fa_q_prep_fp16_d1(
    __fp16 * vtcm_q_tiles, స్థిరరాశి uint8_t * తాత్కాలిక_q_vtcm,
    size_t ప్రారంభం, size_t చివర, size_t g_అడ్డువరుసలు_చివర,
    size_t DK, size_t G, size_t n_అడ్డువరుసలు_q,
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_G, bool q_transposed
) {
    కొరకు (size_t r = ప్రారంభం; r < చివర; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
        size_t   r1       = r % HMX_FP16_TILE_N_అడ్డువరుసలు;
        __fp16 * బహిర్గతం_పునాది = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_అడ్డువరుసలు * DK;

        ఒకవేళ (r >= g_అడ్డువరుసలు_చివర) {
            __fp16 *     బహిర్గతం_dtile = బహిర్గతం_పునాది + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_సదిశ * pv_out0   = ((HVX_సదిశ *) బహిర్గతం_dtile) + r1 / 2;
            HVX_సదిశ * pv_out1   = pv_out0 + 16;
            *pv_out0 = Q6_V_vzero();
            *pv_out1 = Q6_V_vzero();
            కొనసాగు;
        }

        స్థిరరాశి size_t q_idx0 = fastdiv(r + 0, div_G);
        స్థిరరాశి size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        స్థిరరాశి size_t q_idx1 = fastdiv(r + 1, div_G);
        స్థిరరాశి size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        స్థిరరాశి size_t offset0 = q_transposed ? (h_idx0 * n_అడ్డువరుసలు_q + q_idx0) : (q_idx0 * G + h_idx0);
        స్థిరరాశి size_t offset1 = q_transposed ? (h_idx1 * n_అడ్డువరుసలు_q + q_idx1) : (q_idx1 * G + h_idx1);

        స్థిరరాశి HVX_సదిశ * pv_in0 = (స్థిరరాశి HVX_సదిశ *) (తాత్కాలిక_q_vtcm + offset0 * DK * పరిమాణం(__fp16));
        స్థిరరాశి HVX_సదిశ * pv_in1 = (r + 1 < g_అడ్డువరుసలు_చివర)
            ? (స్థిరరాశి HVX_సదిశ *) (తాత్కాలిక_q_vtcm + offset1 * DK * పరిమాణం(__fp16))
            : ఖాళీ;

        HVX_సదిశ     v0 = pv_in0[0];
        HVX_సదిశ     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
        HVX_సదిశ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        __fp16 *     బహిర్గతం_dtile = బహిర్గతం_పునాది + 0 * HMX_FP16_TILE_N_ELMS * 2;
        HVX_సదిశ * pv_out0   = ((HVX_సదిశ *) బహిర్గతం_dtile) + r1 / 2;
        HVX_సదిశ * pv_out1   = pv_out0 + 16;

        *pv_out0 = Q6_V_lo_W(vp);
        *pv_out1 = Q6_V_hi_W(vp);
    }
}

స్థిర అంతఃవరుస శూన్య hmx_fa_q_prep_fp16_d2(
    __fp16 * vtcm_q_tiles, స్థిరరాశి uint8_t * తాత్కాలిక_q_vtcm,
    size_t ప్రారంభం, size_t చివర, size_t g_అడ్డువరుసలు_చివర,
    size_t DK, size_t G, size_t n_అడ్డువరుసలు_q,
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_G, bool q_transposed
) {
    కొరకు (size_t r = ప్రారంభం; r < చివర; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
        size_t   r1       = r % HMX_FP16_TILE_N_అడ్డువరుసలు;
        __fp16 * బహిర్గతం_పునాది = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_అడ్డువరుసలు * DK;

        ఒకవేళ (r >= g_అడ్డువరుసలు_చివర) {
            కొరకు (uint32_t d = 0; d < 2; ++d) {
                __fp16 *     బహిర్గతం_dtile = బహిర్గతం_పునాది + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_సదిశ * pv_out0   = ((HVX_సదిశ *) బహిర్గతం_dtile) + r1 / 2;
                HVX_సదిశ * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            కొనసాగు;
        }

        స్థిరరాశి size_t q_idx0 = fastdiv(r + 0, div_G);
        స్థిరరాశి size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        స్థిరరాశి size_t q_idx1 = fastdiv(r + 1, div_G);
        స్థిరరాశి size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        స్థిరరాశి size_t offset0 = q_transposed ? (h_idx0 * n_అడ్డువరుసలు_q + q_idx0) : (q_idx0 * G + h_idx0);
        స్థిరరాశి size_t offset1 = q_transposed ? (h_idx1 * n_అడ్డువరుసలు_q + q_idx1) : (q_idx1 * G + h_idx1);

        స్థిరరాశి HVX_సదిశ * pv_in0 = (స్థిరరాశి HVX_సదిశ *) (తాత్కాలిక_q_vtcm + offset0 * DK * పరిమాణం(__fp16));
        స్థిరరాశి HVX_సదిశ * pv_in1 = (r + 1 < g_అడ్డువరుసలు_చివర)
            ? (స్థిరరాశి HVX_సదిశ *) (తాత్కాలిక_q_vtcm + offset1 * DK * పరిమాణం(__fp16))
            : ఖాళీ;

        {
            HVX_సదిశ     v0 = pv_in0[0];
            HVX_సదిశ     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_సదిశ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     బహిర్గతం_dtile = బహిర్గతం_పునాది + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_సదిశ * pv_out0   = ((HVX_సదిశ *) బహిర్గతం_dtile) + r1 / 2;
            HVX_సదిశ * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
        {
            HVX_సదిశ     v0 = pv_in0[1];
            HVX_సదిశ     v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_సదిశ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     బహిర్గతం_dtile = బహిర్గతం_పునాది + 1 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_సదిశ * pv_out0   = ((HVX_సదిశ *) బహిర్గతం_dtile) + r1 / 2;
            HVX_సదిశ * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}

స్థిర అంతఃవరుస శూన్య hmx_fa_q_prep_fp16(
    __fp16 * vtcm_q_tiles, స్థిరరాశి uint8_t * తాత్కాలిక_q_vtcm,
    size_t ప్రారంభం, size_t చివర, size_t g_అడ్డువరుసలు_చివర,
    size_t DK, size_t G, size_t n_అడ్డువరుసలు_q,
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_G, uint32_t d_హద్దు, bool q_transposed
) {
    కొరకు (size_t r = ప్రారంభం; r < చివర; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
        size_t   r1       = r % HMX_FP16_TILE_N_అడ్డువరుసలు;
        __fp16 * బహిర్గతం_పునాది = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_అడ్డువరుసలు * DK;

        ఒకవేళ (r >= g_అడ్డువరుసలు_చివర) {
            కొరకు (uint32_t d = 0; d < d_హద్దు; ++d) {
                __fp16 *     బహిర్గతం_dtile = బహిర్గతం_పునాది + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_సదిశ * pv_out0   = ((HVX_సదిశ *) బహిర్గతం_dtile) + r1 / 2;
                HVX_సదిశ * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            కొనసాగు;
        }

        స్థిరరాశి size_t q_idx0 = fastdiv(r + 0, div_G);
        స్థిరరాశి size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        స్థిరరాశి size_t q_idx1 = fastdiv(r + 1, div_G);
        స్థిరరాశి size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        స్థిరరాశి size_t offset0 = q_transposed ? (h_idx0 * n_అడ్డువరుసలు_q + q_idx0) : (q_idx0 * G + h_idx0);
        స్థిరరాశి size_t offset1 = q_transposed ? (h_idx1 * n_అడ్డువరుసలు_q + q_idx1) : (q_idx1 * G + h_idx1);

        స్థిరరాశి HVX_సదిశ * pv_in0 = (స్థిరరాశి HVX_సదిశ *) (తాత్కాలిక_q_vtcm + offset0 * DK * పరిమాణం(__fp16));
        స్థిరరాశి HVX_సదిశ * pv_in1 = (r + 1 < g_అడ్డువరుసలు_చివర)
            ? (స్థిరరాశి HVX_సదిశ *) (తాత్కాలిక_q_vtcm + offset1 * DK * పరిమాణం(__fp16))
            : ఖాళీ;

        కొరకు (uint32_t d = 0; d < d_హద్దు; ++d) {
            HVX_సదిశ     v0 = pv_in0[d];
            HVX_సదిశ     v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_సదిశ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     బహిర్గతం_dtile = బహిర్గతం_పునాది + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_సదిశ * pv_out0   = ((HVX_సదిశ *) బహిర్గతం_dtile) + r1 / 2;
            HVX_సదిశ * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}


// Head-dim-padded Q-prep (f32). Used when DK is not a multiple of 64.
స్థిర అంతఃవరుస శూన్య hmx_fa_q_prep_fp32_pad(__fp16 *                      vtcm_q_tiles,
                                          స్థిరరాశి uint8_t *               తాత్కాలిక_q_vtcm,
                                          size_t                        ప్రారంభం,
                                          size_t                        చివర,
                                          size_t                        g_అడ్డువరుసలు_చివర,
                                          size_t                        dk_లోపల,
                                          size_t                        dk_బహిర్గతం,
                                          size_t                        G,
                                          size_t                        n_అడ్డువరుసలు_q,
                                          స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_G,
                                          bool                          q_transposed) {
    స్థిరరాశి uint32_t n_బహిర్గతం_tiles = (uint32_t) (dk_బహిర్గతం / 32);
    కొరకు (size_t r = ప్రారంభం; r < చివర; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
        size_t   r1       = r % HMX_FP16_TILE_N_అడ్డువరుసలు;
        __fp16 * బహిర్గతం_పునాది = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_అడ్డువరుసలు * dk_బహిర్గతం;

        ఒకవేళ (r >= g_అడ్డువరుసలు_చివర) {
            కొరకు (uint32_t d = 0; d < n_బహిర్గతం_tiles; ++d) {
                ((HVX_సదిశ *) (బహిర్గతం_పునాది + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            కొనసాగు;
        }

        స్థిరరాశి size_t q_idx0 = fastdiv(r + 0, div_G);
        స్థిరరాశి size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        స్థిరరాశి size_t q_idx1 = fastdiv(r + 1, div_G);
        స్థిరరాశి size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        స్థిరరాశి size_t offset0 = q_transposed ? (h_idx0 * n_అడ్డువరుసలు_q + q_idx0) : (q_idx0 * G + h_idx0);
        స్థిరరాశి size_t offset1 = q_transposed ? (h_idx1 * n_అడ్డువరుసలు_q + q_idx1) : (q_idx1 * G + h_idx1);

        స్థిరరాశి HVX_UVector * pv_in0 = (స్థిరరాశి HVX_UVector *) (తాత్కాలిక_q_vtcm + offset0 * dk_లోపల * పరిమాణం(దశాంశం));
        స్థిరరాశి HVX_UVector * pv_in1 = (r + 1 < g_అడ్డువరుసలు_చివర) ? (స్థిరరాశి HVX_UVector *) (తాత్కాలిక_q_vtcm + offset1 * dk_లోపల * పరిమాణం(దశాంశం)) : ఖాళీ;

        కొరకు (uint32_t d = 0; d < n_బహిర్గతం_tiles; ++d) {
            HVX_సదిశ * బహిర్గతం_tile   = (HVX_సదిశ *) (బహిర్గతం_పునాది + d * HMX_FP16_TILE_N_ELMS);
            స్థిరరాశి size_t పునాది_lane  = (size_t) d * 32;
            స్థిరరాశి size_t real_lanes = (పునాది_lane < dk_లోపల) ? hex_smin(32, dk_లోపల - పునాది_lane) : 0;

            ఒకవేళ (real_lanes == 0) {
                బహిర్గతం_tile[r1 / 2] = Q6_V_vzero();
                కొనసాగు;
            }

            HVX_సదిశ v0 = pv_in0[d];
            HVX_సదిశ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            ఒకవేళ (real_lanes < 32) {
                // Straddle tile: keep the first real_lanes floats, zero the padded tail so
                // the packed f16 lanes beyond DK are zero.
                స్థిరరాశి HVX_సదిశ_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * పరిమాణం(దశాంశం)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            బహిర్గతం_tile[r1 / 2] = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);
        }
    }
}

// Head-dim-padded Q-prep (f16). Used when DK is not a multiple of 64.
స్థిర అంతఃవరుస శూన్య hmx_fa_q_prep_fp16_pad(__fp16 *                      vtcm_q_tiles,
                                          స్థిరరాశి uint8_t *               తాత్కాలిక_q_vtcm,
                                          size_t                        ప్రారంభం,
                                          size_t                        చివర,
                                          size_t                        g_అడ్డువరుసలు_చివర,
                                          size_t                        dk_లోపల,
                                          size_t                        dk_బహిర్గతం,
                                          size_t                        G,
                                          size_t                        n_అడ్డువరుసలు_q,
                                          స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_G,
                                          bool                          q_transposed) {
    స్థిరరాశి uint32_t n_బహిర్గతం_జతలు = (uint32_t) (dk_బహిర్గతం / 64);
    కొరకు (size_t r = ప్రారంభం; r < చివర; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
        size_t   r1       = r % HMX_FP16_TILE_N_అడ్డువరుసలు;
        __fp16 * బహిర్గతం_పునాది = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_అడ్డువరుసలు * dk_బహిర్గతం;

        ఒకవేళ (r >= g_అడ్డువరుసలు_చివర) {
            కొరకు (uint32_t d = 0; d < n_బహిర్గతం_జతలు; ++d) {
                __fp16 *     బహిర్గతం_dtile = బహిర్గతం_పునాది + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_సదిశ * pv_out0   = ((HVX_సదిశ *) బహిర్గతం_dtile) + r1 / 2;
                HVX_సదిశ * pv_out1   = pv_out0 + 16;
                *pv_out0               = Q6_V_vzero();
                *pv_out1               = Q6_V_vzero();
            }
            కొనసాగు;
        }

        స్థిరరాశి size_t q_idx0 = fastdiv(r + 0, div_G);
        స్థిరరాశి size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        స్థిరరాశి size_t q_idx1 = fastdiv(r + 1, div_G);
        స్థిరరాశి size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        స్థిరరాశి size_t offset0 = q_transposed ? (h_idx0 * n_అడ్డువరుసలు_q + q_idx0) : (q_idx0 * G + h_idx0);
        స్థిరరాశి size_t offset1 = q_transposed ? (h_idx1 * n_అడ్డువరుసలు_q + q_idx1) : (q_idx1 * G + h_idx1);

        స్థిరరాశి HVX_UVector * pv_in0 = (స్థిరరాశి HVX_UVector *) (తాత్కాలిక_q_vtcm + offset0 * dk_లోపల * పరిమాణం(__fp16));
        స్థిరరాశి HVX_UVector * pv_in1 = (r + 1 < g_అడ్డువరుసలు_చివర) ? (స్థిరరాశి HVX_UVector *) (తాత్కాలిక_q_vtcm + offset1 * dk_లోపల * పరిమాణం(__fp16)) : ఖాళీ;

        కొరకు (uint32_t d = 0; d < n_బహిర్గతం_జతలు; ++d) {
            __fp16 *     బహిర్గతం_dtile = బహిర్గతం_పునాది + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_సదిశ * pv_out0   = ((HVX_సదిశ *) బహిర్గతం_dtile) + r1 / 2;
            HVX_సదిశ * pv_out1   = pv_out0 + 16;

            స్థిరరాశి size_t పునాది_lane  = (size_t) d * 64;
            స్థిరరాశి size_t real_lanes = (పునాది_lane < dk_లోపల) ? hex_smin(64, dk_లోపల - పునాది_lane) : 0;

            ఒకవేళ (real_lanes == 0) {
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
                కొనసాగు;
            }

            HVX_సదిశ v0 = pv_in0[d];
            HVX_సదిశ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            ఒకవేళ (real_lanes < 64) {
                స్థిరరాశి HVX_సదిశ_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * పరిమాణం(__fp16)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            HVX_సదిశ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);
            *pv_out0          = Q6_V_lo_W(vp);
            *pv_out1          = Q6_V_hi_W(vp);
        }
    }
}

స్థిర అంతఃవరుస శూన్య hmx_fa_q_prep_fallback(
    __fp16 * vtcm_q_tiles, uintptr_t q_దత్తాంశం,
    size_t q_nb1, size_t q_nb2, size_t q_nb3,
    uint32_t q_ప్రారంభం, uint32_t kv_తల, uint32_t ib3,
    size_t ప్రారంభం, size_t చివర, size_t n_అడ్డువరుసలు_g,
    size_t G, size_t dk_లోపల, size_t dk_బహిర్గతం, bool is_q_fp32,
    స్థిరరాశి నిర్మాణం fastdiv_విలువలు * div_G
) {
    కొరకు (size_t r = ప్రారంభం; r < చివర; r += 2) {
        స్థిరరాశి size_t q_idx0 = fastdiv(r + 0, div_G);
        స్థిరరాశి size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        స్థిరరాశి size_t q_idx1 = fastdiv(r + 1, div_G);
        స్థిరరాశి size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        స్థిరరాశి uint8_t * q_ptr0 = (r + 0 < n_అడ్డువరుసలు_g) ? ((స్థిరరాశి uint8_t *) q_దత్తాంశం + (q_ప్రారంభం + q_idx0) * q_nb1 +
                                                      (kv_తల * G + h_idx0) * q_nb2 + ib3 * q_nb3) :
                                                      ఖాళీ;
        స్థిరరాశి uint8_t * q_ptr1 = (r + 1 < n_అడ్డువరుసలు_g) ? ((స్థిరరాశి uint8_t *) q_దత్తాంశం + (q_ప్రారంభం + q_idx1) * q_nb1 +
                                                      (kv_తల * G + h_idx1) * q_nb2 + ib3 * q_nb3) :
                                                      ఖాళీ;

        size_t   r0       = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
        size_t   r1       = r % HMX_FP16_TILE_N_అడ్డువరుసలు;
        __fp16 * బహిర్గతం_పునాది = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_అడ్డువరుసలు * dk_బహిర్గతం;

        ఒకవేళ (is_q_fp32) {
            స్థిరరాశి HVX_UVector * pv_in0 = q_ptr0 ? (స్థిరరాశి HVX_UVector *) q_ptr0 : ఖాళీ;
            స్థిరరాశి HVX_UVector * pv_in1 = q_ptr1 ? (స్థిరరాశి HVX_UVector *) q_ptr1 : ఖాళీ;

            కొరకు (uint32_t d = 0; d < dk_బహిర్గతం / 32; ++d) {
                HVX_సదిశ * బహిర్గతం_tile   = (HVX_సదిశ *) (బహిర్గతం_పునాది + d * HMX_FP16_TILE_N_ELMS);
                స్థిరరాశి size_t పునాది_lane  = (size_t) d * 32;
                స్థిరరాశి size_t real_lanes = (పునాది_lane < dk_లోపల) ? hex_smin(32, dk_లోపల - పునాది_lane) : 0;

                ఒకవేళ (real_lanes == 0) {
                    బహిర్గతం_tile[r1 / 2] = Q6_V_vzero();
                    కొనసాగు;
                }
                HVX_సదిశ v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_సదిశ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                ఒకవేళ (real_lanes < 32) {
                    స్థిరరాశి HVX_సదిశ_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * పరిమాణం(దశాంశం)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                బహిర్గతం_tile[r1 / 2] = hvx_vec_f32_గమ్యం_f16_shuff(v0, v1);
            }
        } లేకపోతే {
            స్థిరరాశి HVX_UVector * pv_in0 = q_ptr0 ? (స్థిరరాశి HVX_UVector *) q_ptr0 : ఖాళీ;
            స్థిరరాశి HVX_UVector * pv_in1 = q_ptr1 ? (స్థిరరాశి HVX_UVector *) q_ptr1 : ఖాళీ;

            కొరకు (uint32_t d = 0; d < dk_బహిర్గతం / 64; ++d) {
                __fp16 *     బహిర్గతం_dtile = బహిర్గతం_పునాది + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_సదిశ * pv_out0   = ((HVX_సదిశ *) బహిర్గతం_dtile) + r1 / 2;
                HVX_సదిశ * pv_out1   = pv_out0 + 16;

                స్థిరరాశి size_t పునాది_lane  = (size_t) d * 64;
                స్థిరరాశి size_t real_lanes = (పునాది_lane < dk_లోపల) ? hex_smin(64, dk_లోపల - పునాది_lane) : 0;

                ఒకవేళ (real_lanes == 0) {
                    *pv_out0 = Q6_V_vzero();
                    *pv_out1 = Q6_V_vzero();
                    కొనసాగు;
                }
                HVX_సదిశ v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_సదిశ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                ఒకవేళ (real_lanes < 64) {
                    స్థిరరాశి HVX_సదిశ_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * పరిమాణం(__fp16)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                HVX_సదిశ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);
                *pv_out0 = Q6_V_lo_W(vp);
                *pv_out1 = Q6_V_hi_W(vp);
            }
        }
    }
}

#అయితే_ముగింపు /* HMX_FA_KERNELS_H */
