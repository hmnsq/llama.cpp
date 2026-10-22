#यदि_अपरिभाषितम् HMX_FA_KERNELS_H
#परिभाषय HMX_FA_KERNELS_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "hmx-utils.h"
#अन्तर्भाव "hex-fastdiv.h"

// HMX-specific parameters, offsets and inner kernels for Flash Attention

// Scatter offsets for diagonal tile: entry[2i] = i*136, entry[2i+1] = i*136+6
// 136 = 4 * 32 + 8 = byte offset to diagonal in a 32x32 fp16 interleaved tile
स्थिर ध्रुव int16_t d_tile_scatter_offsets[64] __attribute__((aligned(128))) = {
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

स्थिर अन्तःस्थ शून्य hmx_fa_qk_dot_tile(
    ध्रुव __fp16 * row_tiles,
    ध्रुव __fp16 * col_tiles,
    __fp16 *       बहिर्गमनम्_tile,
    size_t         n_dot_tiles
) {
    यदि (n_dot_tiles == 2) {
        asm चञ्चल(
            HMX_आरोपय_MPY_F16("%1", "%2", "%0")
            HMX_आरोपय_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(row_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } अन्यथा यदि (n_dot_tiles == 4) {
        asm चञ्चल(
            HMX_आरोपय_MPY_F16("%1", "%2", "%0")
            HMX_आरोपय_MPY_F16("%3", "%4", "%0")
            HMX_आरोपय_MPY_F16("%5", "%6", "%0")
            HMX_आरोपय_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(row_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } अन्यथा यदि (n_dot_tiles == 8) {
        asm चञ्चल(
            HMX_आरोपय_MPY_F16("%1", "%2", "%0")
            HMX_आरोपय_MPY_F16("%3", "%4", "%0")
            HMX_आरोपय_MPY_F16("%5", "%6", "%0")
            HMX_आरोपय_MPY_F16("%7", "%8", "%0")
            HMX_आरोपय_MPY_F16("%9", "%10", "%0")
            HMX_आरोपय_MPY_F16("%11", "%12", "%0")
            HMX_आरोपय_MPY_F16("%13", "%14", "%0")
            HMX_आरोपय_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(row_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 4 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 5 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 6 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 7 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } अन्यथा {
        प्रति (size_t k = 0; k < n_dot_tiles; ++k) {
            asm चञ्चल(
                HMX_आरोपय_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(row_tiles), "r"(col_tiles)
            );
            row_tiles += HMX_FP16_TILE_N_ELMS;
            col_tiles += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm चञ्चल(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(बहिर्गमनम्_tile), "r"(0)
        : "memory"
    );
}

स्थिर अन्तःस्थ शून्य hmx_fa_o_update_tile(
    ध्रुव __fp16 * d_diag,
    ध्रुव __fp16 * o_rc,
    ध्रुव __fp16 * p_tile_अन्तर्,
    ध्रुव __fp16 * v_tile_अन्तर्,
    __fp16 *       o_tile_बहिर्गमनम्,
    size_t         n_col_tiles
) {
    asm चञ्चल(
        HMX_आरोपय_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    यदि (n_col_tiles == 2) {
        asm चञ्चल(
            HMX_आरोपय_MPY_F16("%1", "%2", "%0")
            HMX_आरोपय_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(p_tile_अन्तर् + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_अन्तर् + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_अन्तर् + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_अन्तर् + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } अन्यथा यदि (n_col_tiles == 4) {
        asm चञ्चल(
            HMX_आरोपय_MPY_F16("%1", "%2", "%0")
            HMX_आरोपय_MPY_F16("%3", "%4", "%0")
            HMX_आरोपय_MPY_F16("%5", "%6", "%0")
            HMX_आरोपय_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(p_tile_अन्तर् + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_अन्तर् + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_अन्तर् + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_अन्तर् + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_अन्तर् + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_अन्तर् + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_अन्तर् + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_अन्तर् + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } अन्यथा यदि (n_col_tiles == 8) {
        asm चञ्चल(
            HMX_आरोपय_MPY_F16("%1", "%2", "%0")
            HMX_आरोपय_MPY_F16("%3", "%4", "%0")
            HMX_आरोपय_MPY_F16("%5", "%6", "%0")
            HMX_आरोपय_MPY_F16("%7", "%8", "%0")
            HMX_आरोपय_MPY_F16("%9", "%10", "%0")
            HMX_आरोपय_MPY_F16("%11", "%12", "%0")
            HMX_आरोपय_MPY_F16("%13", "%14", "%0")
            HMX_आरोपय_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(p_tile_अन्तर् + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_अन्तर् + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_अन्तर् + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_अन्तर् + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_अन्तर् + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_अन्तर् + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_अन्तर् + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_अन्तर् + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_अन्तर् + 4 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_अन्तर् + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_अन्तर् + 5 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_अन्तर् + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_अन्तर् + 6 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_अन्तर् + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_अन्तर् + 7 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_अन्तर् + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } अन्यथा {
        प्रति (size_t k = 0; k < n_col_tiles; ++k) {
            asm चञ्चल(
                HMX_आरोपय_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(p_tile_अन्तर्), "r"(v_tile_अन्तर्)
            );
            p_tile_अन्तर् += HMX_FP16_TILE_N_ELMS;
            v_tile_अन्तर् += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm चञ्चल(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_tile_बहिर्गमनम्), "r"(0)
        : "memory"
    );
}

स्थिर अन्तःस्थ शून्य hmx_fa_o_norm_tile(
    ध्रुव __fp16 * d_diag,
    ध्रुव __fp16 * o_rc,
    __fp16 *       o_बहिर्गमनम्
) {
    asm चञ्चल(
        HMX_आरोपय_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    asm चञ्चल(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_बहिर्गमनम्), "r"(0)
        : "memory"
    );
}

स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp32_d2(
    __fp16 * vtcm_q_tiles, ध्रुव uint8_t * अस्थायी_q_vtcm,
    size_t start, size_t अन्तः_2, size_t g_पङ्क्तिकाः_अन्तः,
    size_t DK, size_t G, size_t n_पङ्क्तिकाः_q,
    ध्रुव संरचना fastdiv_मानानि * div_G, bool q_transposed
) {
    प्रति (size_t r = start; r < अन्तः_2; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_पङ्क्तिकाः;
        size_t   r1       = r % HMX_FP16_TILE_N_पङ्क्तिकाः;
        __fp16 * बहिर्गमनम्_आधारः = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_पङ्क्तिकाः * DK;

        यदि (r >= g_पङ्क्तिकाः_अन्तः) {
            ((HVX_सदिशम् *) (बहिर्गमनम्_आधारः + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            ((HVX_सदिशम् *) (बहिर्गमनम्_आधारः + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            अनुवर्त;
        }

        ध्रुव size_t q_idx0 = fastdiv(r + 0, div_G);
        ध्रुव size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ध्रुव size_t q_idx1 = fastdiv(r + 1, div_G);
        ध्रुव size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ध्रुव size_t offset0 = q_transposed ? (h_idx0 * n_पङ्क्तिकाः_q + q_idx0) : (q_idx0 * G + h_idx0);
        ध्रुव size_t offset1 = q_transposed ? (h_idx1 * n_पङ्क्तिकाः_q + q_idx1) : (q_idx1 * G + h_idx1);

        ध्रुव HVX_सदिशम् * pv_in0 = (ध्रुव HVX_सदिशम् *) (अस्थायी_q_vtcm + offset0 * DK * परिमाण(भिन्न));
        ध्रुव HVX_सदिशम् * pv_in1 = (r + 1 < g_पङ्क्तिकाः_अन्तः)
            ? (ध्रुव HVX_सदिशम् *) (अस्थायी_q_vtcm + offset1 * DK * परिमाण(भिन्न))
            : रिक्त;

        {
            HVX_सदिशम् v0 = pv_in0[0];
            HVX_सदिशम् v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_सदिशम् v_hf = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);
            ((HVX_सदिशम् *) (बहिर्गमनम्_आधारः + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
        {
            HVX_सदिशम् v0 = pv_in0[1];
            HVX_सदिशम् v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_सदिशम् v_hf = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);
            ((HVX_सदिशम् *) (बहिर्गमनम्_आधारः + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp32_d4(
    __fp16 * vtcm_q_tiles, ध्रुव uint8_t * अस्थायी_q_vtcm,
    size_t start, size_t अन्तः_2, size_t g_पङ्क्तिकाः_अन्तः,
    size_t DK, size_t G, size_t n_पङ्क्तिकाः_q,
    ध्रुव संरचना fastdiv_मानानि * div_G, bool q_transposed
) {
    प्रति (size_t r = start; r < अन्तः_2; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_पङ्क्तिकाः;
        size_t   r1       = r % HMX_FP16_TILE_N_पङ्क्तिकाः;
        __fp16 * बहिर्गमनम्_आधारः = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_पङ्क्तिकाः * DK;

        यदि (r >= g_पङ्क्तिकाः_अन्तः) {
            प्रति (uint32_t d = 0; d < 4; ++d) {
                ((HVX_सदिशम् *) (बहिर्गमनम्_आधारः + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            अनुवर्त;
        }

        ध्रुव size_t q_idx0 = fastdiv(r + 0, div_G);
        ध्रुव size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ध्रुव size_t q_idx1 = fastdiv(r + 1, div_G);
        ध्रुव size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ध्रुव size_t offset0 = q_transposed ? (h_idx0 * n_पङ्क्तिकाः_q + q_idx0) : (q_idx0 * G + h_idx0);
        ध्रुव size_t offset1 = q_transposed ? (h_idx1 * n_पङ्क्तिकाः_q + q_idx1) : (q_idx1 * G + h_idx1);

        ध्रुव HVX_सदिशम् * pv_in0 = (ध्रुव HVX_सदिशम् *) (अस्थायी_q_vtcm + offset0 * DK * परिमाण(भिन्न));
        ध्रुव HVX_सदिशम् * pv_in1 = (r + 1 < g_पङ्क्तिकाः_अन्तः)
            ? (ध्रुव HVX_सदिशम् *) (अस्थायी_q_vtcm + offset1 * DK * परिमाण(भिन्न))
            : रिक्त;

        प्रति (uint32_t d = 0; d < 4; ++d) {
            HVX_सदिशम् v0 = pv_in0[d];
            HVX_सदिशम् v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_सदिशम् v_hf = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);
            ((HVX_सदिशम् *) (बहिर्गमनम्_आधारः + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp32(
    __fp16 * vtcm_q_tiles, ध्रुव uint8_t * अस्थायी_q_vtcm,
    size_t start, size_t अन्तः_2, size_t g_पङ्क्तिकाः_अन्तः,
    size_t DK, size_t G, size_t n_पङ्क्तिकाः_q,
    ध्रुव संरचना fastdiv_मानानि * div_G, uint32_t d_सीमा, bool q_transposed
) {
    प्रति (size_t r = start; r < अन्तः_2; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_पङ्क्तिकाः;
        size_t   r1       = r % HMX_FP16_TILE_N_पङ्क्तिकाः;
        __fp16 * बहिर्गमनम्_आधारः = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_पङ्क्तिकाः * DK;

        यदि (r >= g_पङ्क्तिकाः_अन्तः) {
            प्रति (uint32_t d = 0; d < d_सीमा; ++d) {
                ((HVX_सदिशम् *) (बहिर्गमनम्_आधारः + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            अनुवर्त;
        }

        ध्रुव size_t q_idx0 = fastdiv(r + 0, div_G);
        ध्रुव size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ध्रुव size_t q_idx1 = fastdiv(r + 1, div_G);
        ध्रुव size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ध्रुव size_t offset0 = q_transposed ? (h_idx0 * n_पङ्क्तिकाः_q + q_idx0) : (q_idx0 * G + h_idx0);
        ध्रुव size_t offset1 = q_transposed ? (h_idx1 * n_पङ्क्तिकाः_q + q_idx1) : (q_idx1 * G + h_idx1);

        ध्रुव HVX_सदिशम् * pv_in0 = (ध्रुव HVX_सदिशम् *) (अस्थायी_q_vtcm + offset0 * DK * परिमाण(भिन्न));
        ध्रुव HVX_सदिशम् * pv_in1 = (r + 1 < g_पङ्क्तिकाः_अन्तः)
            ? (ध्रुव HVX_सदिशम् *) (अस्थायी_q_vtcm + offset1 * DK * परिमाण(भिन्न))
            : रिक्त;

        प्रति (uint32_t d = 0; d < d_सीमा; ++d) {
            HVX_सदिशम् v0   = pv_in0[d];
            HVX_सदिशम् v1   = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_सदिशम् v_hf = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);

            HVX_सदिशम् * बहिर्गमनम्_tile = (HVX_सदिशम् *) (बहिर्गमनम्_आधारः + d * HMX_FP16_TILE_N_ELMS);
            बहिर्गमनम्_tile[r1 / 2]      = v_hf;
        }
    }
}

स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp16_d1(
    __fp16 * vtcm_q_tiles, ध्रुव uint8_t * अस्थायी_q_vtcm,
    size_t start, size_t अन्तः_2, size_t g_पङ्क्तिकाः_अन्तः,
    size_t DK, size_t G, size_t n_पङ्क्तिकाः_q,
    ध्रुव संरचना fastdiv_मानानि * div_G, bool q_transposed
) {
    प्रति (size_t r = start; r < अन्तः_2; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_पङ्क्तिकाः;
        size_t   r1       = r % HMX_FP16_TILE_N_पङ्क्तिकाः;
        __fp16 * बहिर्गमनम्_आधारः = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_पङ्क्तिकाः * DK;

        यदि (r >= g_पङ्क्तिकाः_अन्तः) {
            __fp16 *     बहिर्गमनम्_dtile = बहिर्गमनम्_आधारः + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_सदिशम् * pv_out0   = ((HVX_सदिशम् *) बहिर्गमनम्_dtile) + r1 / 2;
            HVX_सदिशम् * pv_out1   = pv_out0 + 16;
            *pv_out0 = Q6_V_vzero();
            *pv_out1 = Q6_V_vzero();
            अनुवर्त;
        }

        ध्रुव size_t q_idx0 = fastdiv(r + 0, div_G);
        ध्रुव size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ध्रुव size_t q_idx1 = fastdiv(r + 1, div_G);
        ध्रुव size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ध्रुव size_t offset0 = q_transposed ? (h_idx0 * n_पङ्क्तिकाः_q + q_idx0) : (q_idx0 * G + h_idx0);
        ध्रुव size_t offset1 = q_transposed ? (h_idx1 * n_पङ्क्तिकाः_q + q_idx1) : (q_idx1 * G + h_idx1);

        ध्रुव HVX_सदिशम् * pv_in0 = (ध्रुव HVX_सदिशम् *) (अस्थायी_q_vtcm + offset0 * DK * परिमाण(__fp16));
        ध्रुव HVX_सदिशम् * pv_in1 = (r + 1 < g_पङ्क्तिकाः_अन्तः)
            ? (ध्रुव HVX_सदिशम् *) (अस्थायी_q_vtcm + offset1 * DK * परिमाण(__fp16))
            : रिक्त;

        HVX_सदिशम्     v0 = pv_in0[0];
        HVX_सदिशम्     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
        HVX_सदिशम्_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        __fp16 *     बहिर्गमनम्_dtile = बहिर्गमनम्_आधारः + 0 * HMX_FP16_TILE_N_ELMS * 2;
        HVX_सदिशम् * pv_out0   = ((HVX_सदिशम् *) बहिर्गमनम्_dtile) + r1 / 2;
        HVX_सदिशम् * pv_out1   = pv_out0 + 16;

        *pv_out0 = Q6_V_lo_W(vp);
        *pv_out1 = Q6_V_hi_W(vp);
    }
}

स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp16_d2(
    __fp16 * vtcm_q_tiles, ध्रुव uint8_t * अस्थायी_q_vtcm,
    size_t start, size_t अन्तः_2, size_t g_पङ्क्तिकाः_अन्तः,
    size_t DK, size_t G, size_t n_पङ्क्तिकाः_q,
    ध्रुव संरचना fastdiv_मानानि * div_G, bool q_transposed
) {
    प्रति (size_t r = start; r < अन्तः_2; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_पङ्क्तिकाः;
        size_t   r1       = r % HMX_FP16_TILE_N_पङ्क्तिकाः;
        __fp16 * बहिर्गमनम्_आधारः = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_पङ्क्तिकाः * DK;

        यदि (r >= g_पङ्क्तिकाः_अन्तः) {
            प्रति (uint32_t d = 0; d < 2; ++d) {
                __fp16 *     बहिर्गमनम्_dtile = बहिर्गमनम्_आधारः + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_सदिशम् * pv_out0   = ((HVX_सदिशम् *) बहिर्गमनम्_dtile) + r1 / 2;
                HVX_सदिशम् * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            अनुवर्त;
        }

        ध्रुव size_t q_idx0 = fastdiv(r + 0, div_G);
        ध्रुव size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ध्रुव size_t q_idx1 = fastdiv(r + 1, div_G);
        ध्रुव size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ध्रुव size_t offset0 = q_transposed ? (h_idx0 * n_पङ्क्तिकाः_q + q_idx0) : (q_idx0 * G + h_idx0);
        ध्रुव size_t offset1 = q_transposed ? (h_idx1 * n_पङ्क्तिकाः_q + q_idx1) : (q_idx1 * G + h_idx1);

        ध्रुव HVX_सदिशम् * pv_in0 = (ध्रुव HVX_सदिशम् *) (अस्थायी_q_vtcm + offset0 * DK * परिमाण(__fp16));
        ध्रुव HVX_सदिशम् * pv_in1 = (r + 1 < g_पङ्क्तिकाः_अन्तः)
            ? (ध्रुव HVX_सदिशम् *) (अस्थायी_q_vtcm + offset1 * DK * परिमाण(__fp16))
            : रिक्त;

        {
            HVX_सदिशम्     v0 = pv_in0[0];
            HVX_सदिशम्     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_सदिशम्_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     बहिर्गमनम्_dtile = बहिर्गमनम्_आधारः + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_सदिशम् * pv_out0   = ((HVX_सदिशम् *) बहिर्गमनम्_dtile) + r1 / 2;
            HVX_सदिशम् * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
        {
            HVX_सदिशम्     v0 = pv_in0[1];
            HVX_सदिशम्     v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_सदिशम्_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     बहिर्गमनम्_dtile = बहिर्गमनम्_आधारः + 1 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_सदिशम् * pv_out0   = ((HVX_सदिशम् *) बहिर्गमनम्_dtile) + r1 / 2;
            HVX_सदिशम् * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}

स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp16(
    __fp16 * vtcm_q_tiles, ध्रुव uint8_t * अस्थायी_q_vtcm,
    size_t start, size_t अन्तः_2, size_t g_पङ्क्तिकाः_अन्तः,
    size_t DK, size_t G, size_t n_पङ्क्तिकाः_q,
    ध्रुव संरचना fastdiv_मानानि * div_G, uint32_t d_सीमा, bool q_transposed
) {
    प्रति (size_t r = start; r < अन्तः_2; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_पङ्क्तिकाः;
        size_t   r1       = r % HMX_FP16_TILE_N_पङ्क्तिकाः;
        __fp16 * बहिर्गमनम्_आधारः = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_पङ्क्तिकाः * DK;

        यदि (r >= g_पङ्क्तिकाः_अन्तः) {
            प्रति (uint32_t d = 0; d < d_सीमा; ++d) {
                __fp16 *     बहिर्गमनम्_dtile = बहिर्गमनम्_आधारः + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_सदिशम् * pv_out0   = ((HVX_सदिशम् *) बहिर्गमनम्_dtile) + r1 / 2;
                HVX_सदिशम् * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            अनुवर्त;
        }

        ध्रुव size_t q_idx0 = fastdiv(r + 0, div_G);
        ध्रुव size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ध्रुव size_t q_idx1 = fastdiv(r + 1, div_G);
        ध्रुव size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ध्रुव size_t offset0 = q_transposed ? (h_idx0 * n_पङ्क्तिकाः_q + q_idx0) : (q_idx0 * G + h_idx0);
        ध्रुव size_t offset1 = q_transposed ? (h_idx1 * n_पङ्क्तिकाः_q + q_idx1) : (q_idx1 * G + h_idx1);

        ध्रुव HVX_सदिशम् * pv_in0 = (ध्रुव HVX_सदिशम् *) (अस्थायी_q_vtcm + offset0 * DK * परिमाण(__fp16));
        ध्रुव HVX_सदिशम् * pv_in1 = (r + 1 < g_पङ्क्तिकाः_अन्तः)
            ? (ध्रुव HVX_सदिशम् *) (अस्थायी_q_vtcm + offset1 * DK * परिमाण(__fp16))
            : रिक्त;

        प्रति (uint32_t d = 0; d < d_सीमा; ++d) {
            HVX_सदिशम्     v0 = pv_in0[d];
            HVX_सदिशम्     v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_सदिशम्_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     बहिर्गमनम्_dtile = बहिर्गमनम्_आधारः + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_सदिशम् * pv_out0   = ((HVX_सदिशम् *) बहिर्गमनम्_dtile) + r1 / 2;
            HVX_सदिशम् * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}


// Head-dim-padded Q-prep (f32). Used when DK is not a multiple of 64.
स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp32_pad(__fp16 *                      vtcm_q_tiles,
                                          ध्रुव uint8_t *               अस्थायी_q_vtcm,
                                          size_t                        start,
                                          size_t                        अन्तः_2,
                                          size_t                        g_पङ्क्तिकाः_अन्तः,
                                          size_t                        dk_अन्तर्,
                                          size_t                        dk_बहिर्गमनम्,
                                          size_t                        G,
                                          size_t                        n_पङ्क्तिकाः_q,
                                          ध्रुव संरचना fastdiv_मानानि * div_G,
                                          bool                          q_transposed) {
    ध्रुव uint32_t n_बहिर्गमनम्_tiles = (uint32_t) (dk_बहिर्गमनम् / 32);
    प्रति (size_t r = start; r < अन्तः_2; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_पङ्क्तिकाः;
        size_t   r1       = r % HMX_FP16_TILE_N_पङ्क्तिकाः;
        __fp16 * बहिर्गमनम्_आधारः = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_पङ्क्तिकाः * dk_बहिर्गमनम्;

        यदि (r >= g_पङ्क्तिकाः_अन्तः) {
            प्रति (uint32_t d = 0; d < n_बहिर्गमनम्_tiles; ++d) {
                ((HVX_सदिशम् *) (बहिर्गमनम्_आधारः + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            अनुवर्त;
        }

        ध्रुव size_t q_idx0 = fastdiv(r + 0, div_G);
        ध्रुव size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ध्रुव size_t q_idx1 = fastdiv(r + 1, div_G);
        ध्रुव size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ध्रुव size_t offset0 = q_transposed ? (h_idx0 * n_पङ्क्तिकाः_q + q_idx0) : (q_idx0 * G + h_idx0);
        ध्रुव size_t offset1 = q_transposed ? (h_idx1 * n_पङ्क्तिकाः_q + q_idx1) : (q_idx1 * G + h_idx1);

        ध्रुव HVX_UVector * pv_in0 = (ध्रुव HVX_UVector *) (अस्थायी_q_vtcm + offset0 * dk_अन्तर् * परिमाण(भिन्न));
        ध्रुव HVX_UVector * pv_in1 = (r + 1 < g_पङ्क्तिकाः_अन्तः) ? (ध्रुव HVX_UVector *) (अस्थायी_q_vtcm + offset1 * dk_अन्तर् * परिमाण(भिन्न)) : रिक्त;

        प्रति (uint32_t d = 0; d < n_बहिर्गमनम्_tiles; ++d) {
            HVX_सदिशम् * बहिर्गमनम्_tile   = (HVX_सदिशम् *) (बहिर्गमनम्_आधारः + d * HMX_FP16_TILE_N_ELMS);
            ध्रुव size_t आधारः_lane  = (size_t) d * 32;
            ध्रुव size_t real_lanes = (आधारः_lane < dk_अन्तर्) ? hex_smin(32, dk_अन्तर् - आधारः_lane) : 0;

            यदि (real_lanes == 0) {
                बहिर्गमनम्_tile[r1 / 2] = Q6_V_vzero();
                अनुवर्त;
            }

            HVX_सदिशम् v0 = pv_in0[d];
            HVX_सदिशम् v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            यदि (real_lanes < 32) {
                // Straddle tile: keep the first real_lanes floats, zero the padded tail so
                // the packed f16 lanes beyond DK are zero.
                ध्रुव HVX_सदिशम्_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * परिमाण(भिन्न)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            बहिर्गमनम्_tile[r1 / 2] = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);
        }
    }
}

// Head-dim-padded Q-prep (f16). Used when DK is not a multiple of 64.
स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp16_pad(__fp16 *                      vtcm_q_tiles,
                                          ध्रुव uint8_t *               अस्थायी_q_vtcm,
                                          size_t                        start,
                                          size_t                        अन्तः_2,
                                          size_t                        g_पङ्क्तिकाः_अन्तः,
                                          size_t                        dk_अन्तर्,
                                          size_t                        dk_बहिर्गमनम्,
                                          size_t                        G,
                                          size_t                        n_पङ्क्तिकाः_q,
                                          ध्रुव संरचना fastdiv_मानानि * div_G,
                                          bool                          q_transposed) {
    ध्रुव uint32_t n_बहिर्गमनम्_युग्मानि = (uint32_t) (dk_बहिर्गमनम् / 64);
    प्रति (size_t r = start; r < अन्तः_2; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_पङ्क्तिकाः;
        size_t   r1       = r % HMX_FP16_TILE_N_पङ्क्तिकाः;
        __fp16 * बहिर्गमनम्_आधारः = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_पङ्क्तिकाः * dk_बहिर्गमनम्;

        यदि (r >= g_पङ्क्तिकाः_अन्तः) {
            प्रति (uint32_t d = 0; d < n_बहिर्गमनम्_युग्मानि; ++d) {
                __fp16 *     बहिर्गमनम्_dtile = बहिर्गमनम्_आधारः + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_सदिशम् * pv_out0   = ((HVX_सदिशम् *) बहिर्गमनम्_dtile) + r1 / 2;
                HVX_सदिशम् * pv_out1   = pv_out0 + 16;
                *pv_out0               = Q6_V_vzero();
                *pv_out1               = Q6_V_vzero();
            }
            अनुवर्त;
        }

        ध्रुव size_t q_idx0 = fastdiv(r + 0, div_G);
        ध्रुव size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ध्रुव size_t q_idx1 = fastdiv(r + 1, div_G);
        ध्रुव size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ध्रुव size_t offset0 = q_transposed ? (h_idx0 * n_पङ्क्तिकाः_q + q_idx0) : (q_idx0 * G + h_idx0);
        ध्रुव size_t offset1 = q_transposed ? (h_idx1 * n_पङ्क्तिकाः_q + q_idx1) : (q_idx1 * G + h_idx1);

        ध्रुव HVX_UVector * pv_in0 = (ध्रुव HVX_UVector *) (अस्थायी_q_vtcm + offset0 * dk_अन्तर् * परिमाण(__fp16));
        ध्रुव HVX_UVector * pv_in1 = (r + 1 < g_पङ्क्तिकाः_अन्तः) ? (ध्रुव HVX_UVector *) (अस्थायी_q_vtcm + offset1 * dk_अन्तर् * परिमाण(__fp16)) : रिक्त;

        प्रति (uint32_t d = 0; d < n_बहिर्गमनम्_युग्मानि; ++d) {
            __fp16 *     बहिर्गमनम्_dtile = बहिर्गमनम्_आधारः + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_सदिशम् * pv_out0   = ((HVX_सदिशम् *) बहिर्गमनम्_dtile) + r1 / 2;
            HVX_सदिशम् * pv_out1   = pv_out0 + 16;

            ध्रुव size_t आधारः_lane  = (size_t) d * 64;
            ध्रुव size_t real_lanes = (आधारः_lane < dk_अन्तर्) ? hex_smin(64, dk_अन्तर् - आधारः_lane) : 0;

            यदि (real_lanes == 0) {
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
                अनुवर्त;
            }

            HVX_सदिशम् v0 = pv_in0[d];
            HVX_सदिशम् v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            यदि (real_lanes < 64) {
                ध्रुव HVX_सदिशम्_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * परिमाण(__fp16)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            HVX_सदिशम्_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);
            *pv_out0          = Q6_V_lo_W(vp);
            *pv_out1          = Q6_V_hi_W(vp);
        }
    }
}

स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fallback(
    __fp16 * vtcm_q_tiles, uintptr_t q_दत्त,
    size_t q_nb1, size_t q_nb2, size_t q_nb3,
    uint32_t q_start, uint32_t kv_शीर्ष, uint32_t ib3,
    size_t start, size_t अन्तः_2, size_t n_पङ्क्तिकाः_g,
    size_t G, size_t dk_अन्तर्, size_t dk_बहिर्गमनम्, bool is_q_fp32,
    ध्रुव संरचना fastdiv_मानानि * div_G
) {
    प्रति (size_t r = start; r < अन्तः_2; r += 2) {
        ध्रुव size_t q_idx0 = fastdiv(r + 0, div_G);
        ध्रुव size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ध्रुव size_t q_idx1 = fastdiv(r + 1, div_G);
        ध्रुव size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ध्रुव uint8_t * q_ptr0 = (r + 0 < n_पङ्क्तिकाः_g) ? ((ध्रुव uint8_t *) q_दत्त + (q_start + q_idx0) * q_nb1 +
                                                      (kv_शीर्ष * G + h_idx0) * q_nb2 + ib3 * q_nb3) :
                                                      रिक्त;
        ध्रुव uint8_t * q_ptr1 = (r + 1 < n_पङ्क्तिकाः_g) ? ((ध्रुव uint8_t *) q_दत्त + (q_start + q_idx1) * q_nb1 +
                                                      (kv_शीर्ष * G + h_idx1) * q_nb2 + ib3 * q_nb3) :
                                                      रिक्त;

        size_t   r0       = r / HMX_FP16_TILE_N_पङ्क्तिकाः;
        size_t   r1       = r % HMX_FP16_TILE_N_पङ्क्तिकाः;
        __fp16 * बहिर्गमनम्_आधारः = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_पङ्क्तिकाः * dk_बहिर्गमनम्;

        यदि (is_q_fp32) {
            ध्रुव HVX_UVector * pv_in0 = q_ptr0 ? (ध्रुव HVX_UVector *) q_ptr0 : रिक्त;
            ध्रुव HVX_UVector * pv_in1 = q_ptr1 ? (ध्रुव HVX_UVector *) q_ptr1 : रिक्त;

            प्रति (uint32_t d = 0; d < dk_बहिर्गमनम् / 32; ++d) {
                HVX_सदिशम् * बहिर्गमनम्_tile   = (HVX_सदिशम् *) (बहिर्गमनम्_आधारः + d * HMX_FP16_TILE_N_ELMS);
                ध्रुव size_t आधारः_lane  = (size_t) d * 32;
                ध्रुव size_t real_lanes = (आधारः_lane < dk_अन्तर्) ? hex_smin(32, dk_अन्तर् - आधारः_lane) : 0;

                यदि (real_lanes == 0) {
                    बहिर्गमनम्_tile[r1 / 2] = Q6_V_vzero();
                    अनुवर्त;
                }
                HVX_सदिशम् v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_सदिशम् v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                यदि (real_lanes < 32) {
                    ध्रुव HVX_सदिशम्_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * परिमाण(भिन्न)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                बहिर्गमनम्_tile[r1 / 2] = hvx_vec_f32_गन्तव्यम्_f16_shuff(v0, v1);
            }
        } अन्यथा {
            ध्रुव HVX_UVector * pv_in0 = q_ptr0 ? (ध्रुव HVX_UVector *) q_ptr0 : रिक्त;
            ध्रुव HVX_UVector * pv_in1 = q_ptr1 ? (ध्रुव HVX_UVector *) q_ptr1 : रिक्त;

            प्रति (uint32_t d = 0; d < dk_बहिर्गमनम् / 64; ++d) {
                __fp16 *     बहिर्गमनम्_dtile = बहिर्गमनम्_आधारः + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_सदिशम् * pv_out0   = ((HVX_सदिशम् *) बहिर्गमनम्_dtile) + r1 / 2;
                HVX_सदिशम् * pv_out1   = pv_out0 + 16;

                ध्रुव size_t आधारः_lane  = (size_t) d * 64;
                ध्रुव size_t real_lanes = (आधारः_lane < dk_अन्तर्) ? hex_smin(64, dk_अन्तर् - आधारः_lane) : 0;

                यदि (real_lanes == 0) {
                    *pv_out0 = Q6_V_vzero();
                    *pv_out1 = Q6_V_vzero();
                    अनुवर्त;
                }
                HVX_सदिशम् v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_सदिशम् v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                यदि (real_lanes < 64) {
                    ध्रुव HVX_सदिशम्_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * परिमाण(__fp16)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                HVX_सदिशम्_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);
                *pv_out0 = Q6_V_lo_W(vp);
                *pv_out1 = Q6_V_hi_W(vp);
            }
        }
    }
}

#यदि_अन्तः /* HMX_FA_KERNELS_H */
