#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HMX_FA_KERNELS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_FA_KERNELS_H

#ಅನ್ತರ್ಭಾವ <stdint.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>
#ಅನ್ತರ್ಭಾವ <stdbool.h>
#ಅನ್ತರ್ಭಾವ "hvx-utils.h"
#ಅನ್ತರ್ಭಾವ "hmx-utils.h"
#ಅನ್ತರ್ಭಾವ "hex-fastdiv.h"

// HMX-specific parameters, offsets and inner kernels for Flash Attention

// Scatter offsets for diagonal tile: entry[2i] = i*136, entry[2i+1] = i*136+6
// 136 = 4 * 32 + 8 = byte offset to diagonal in a 32x32 fp16 interleaved tile
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ int16_t d_tile_scatter_offsets[64] __attribute__((aligned(128))) = {
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

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_fa_qk_dot_tile(
    ಸ್ಥಿರಾಂಕ __fp16 * ಅಡ್ಡಸಾಲು_tiles,
    ಸ್ಥಿರಾಂಕ __fp16 * ಲಂಬಸಾಲು_tiles,
    __fp16 *       ಹೊರಹೋಗು_tile,
    size_t         n_dot_tiles
) {
    ಆದರೆ (n_dot_tiles == 2) {
        asm ಅಸ್ಥಿರ(
            HMX_ತುಂಬು_MPY_F16("%1", "%2", "%0")
            HMX_ತುಂಬು_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(ಅಡ್ಡಸಾಲು_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(ಲಂಬಸಾಲು_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(ಅಡ್ಡಸಾಲು_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(ಲಂಬಸಾಲು_tiles + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (n_dot_tiles == 4) {
        asm ಅಸ್ಥಿರ(
            HMX_ತುಂಬು_MPY_F16("%1", "%2", "%0")
            HMX_ತುಂಬು_MPY_F16("%3", "%4", "%0")
            HMX_ತುಂಬು_MPY_F16("%5", "%6", "%0")
            HMX_ತುಂಬು_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(ಅಡ್ಡಸಾಲು_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(ಲಂಬಸಾಲು_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(ಅಡ್ಡಸಾಲು_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(ಲಂಬಸಾಲು_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(ಅಡ್ಡಸಾಲು_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(ಲಂಬಸಾಲು_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(ಅಡ್ಡಸಾಲು_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(ಲಂಬಸಾಲು_tiles + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (n_dot_tiles == 8) {
        asm ಅಸ್ಥಿರ(
            HMX_ತುಂಬು_MPY_F16("%1", "%2", "%0")
            HMX_ತುಂಬು_MPY_F16("%3", "%4", "%0")
            HMX_ತುಂಬು_MPY_F16("%5", "%6", "%0")
            HMX_ತುಂಬು_MPY_F16("%7", "%8", "%0")
            HMX_ತುಂಬು_MPY_F16("%9", "%10", "%0")
            HMX_ತುಂಬು_MPY_F16("%11", "%12", "%0")
            HMX_ತುಂಬು_MPY_F16("%13", "%14", "%0")
            HMX_ತುಂಬು_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(ಅಡ್ಡಸಾಲು_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(ಲಂಬಸಾಲು_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(ಅಡ್ಡಸಾಲು_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(ಲಂಬಸಾಲು_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(ಅಡ್ಡಸಾಲು_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(ಲಂಬಸಾಲು_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(ಅಡ್ಡಸಾಲು_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(ಲಂಬಸಾಲು_tiles + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(ಅಡ್ಡಸಾಲು_tiles + 4 * HMX_FP16_TILE_N_ELMS), "r"(ಲಂಬಸಾಲು_tiles + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(ಅಡ್ಡಸಾಲು_tiles + 5 * HMX_FP16_TILE_N_ELMS), "r"(ಲಂಬಸಾಲು_tiles + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(ಅಡ್ಡಸಾಲು_tiles + 6 * HMX_FP16_TILE_N_ELMS), "r"(ಲಂಬಸಾಲು_tiles + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(ಅಡ್ಡಸಾಲು_tiles + 7 * HMX_FP16_TILE_N_ELMS), "r"(ಲಂಬಸಾಲು_tiles + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (size_t k = 0; k < n_dot_tiles; ++k) {
            asm ಅಸ್ಥಿರ(
                HMX_ತುಂಬು_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(ಅಡ್ಡಸಾಲು_tiles), "r"(ಲಂಬಸಾಲು_tiles)
            );
            ಅಡ್ಡಸಾಲು_tiles += HMX_FP16_TILE_N_ELMS;
            ಲಂಬಸಾಲು_tiles += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm ಅಸ್ಥಿರ(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(ಹೊರಹೋಗು_tile), "r"(0)
        : "memory"
    );
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_fa_o_update_tile(
    ಸ್ಥಿರಾಂಕ __fp16 * d_diag,
    ಸ್ಥಿರಾಂಕ __fp16 * o_rc,
    ಸ್ಥಿರಾಂಕ __fp16 * p_tile_ಒಳ,
    ಸ್ಥಿರಾಂಕ __fp16 * v_tile_ಒಳ,
    __fp16 *       o_tile_ಹೊರಹೋಗು,
    size_t         n_ಲಂಬಸಾಲು_tiles
) {
    asm ಅಸ್ಥಿರ(
        HMX_ತುಂಬು_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    ಆದರೆ (n_ಲಂಬಸಾಲು_tiles == 2) {
        asm ಅಸ್ಥಿರ(
            HMX_ತುಂಬು_MPY_F16("%1", "%2", "%0")
            HMX_ತುಂಬು_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(p_tile_ಒಳ + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ಒಳ + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ಒಳ + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ಒಳ + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (n_ಲಂಬಸಾಲು_tiles == 4) {
        asm ಅಸ್ಥಿರ(
            HMX_ತುಂಬು_MPY_F16("%1", "%2", "%0")
            HMX_ತುಂಬು_MPY_F16("%3", "%4", "%0")
            HMX_ತುಂಬು_MPY_F16("%5", "%6", "%0")
            HMX_ತುಂಬು_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(p_tile_ಒಳ + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ಒಳ + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ಒಳ + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ಒಳ + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ಒಳ + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ಒಳ + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ಒಳ + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ಒಳ + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } ಇಲ್ಲದಿದ್ದರೆ ಆದರೆ (n_ಲಂಬಸಾಲು_tiles == 8) {
        asm ಅಸ್ಥಿರ(
            HMX_ತುಂಬು_MPY_F16("%1", "%2", "%0")
            HMX_ತುಂಬು_MPY_F16("%3", "%4", "%0")
            HMX_ತುಂಬು_MPY_F16("%5", "%6", "%0")
            HMX_ತುಂಬು_MPY_F16("%7", "%8", "%0")
            HMX_ತುಂಬು_MPY_F16("%9", "%10", "%0")
            HMX_ತುಂಬು_MPY_F16("%11", "%12", "%0")
            HMX_ತುಂಬು_MPY_F16("%13", "%14", "%0")
            HMX_ತುಂಬು_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(p_tile_ಒಳ + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ಒಳ + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ಒಳ + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ಒಳ + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ಒಳ + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ಒಳ + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ಒಳ + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ಒಳ + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ಒಳ + 4 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ಒಳ + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ಒಳ + 5 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ಒಳ + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ಒಳ + 6 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ಒಳ + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ಒಳ + 7 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ಒಳ + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } ಇಲ್ಲದಿದ್ದರೆ {
        ಗಾಗಿ (size_t k = 0; k < n_ಲಂಬಸಾಲು_tiles; ++k) {
            asm ಅಸ್ಥಿರ(
                HMX_ತುಂಬು_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(p_tile_ಒಳ), "r"(v_tile_ಒಳ)
            );
            p_tile_ಒಳ += HMX_FP16_TILE_N_ELMS;
            v_tile_ಒಳ += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm ಅಸ್ಥಿರ(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_tile_ಹೊರಹೋಗು), "r"(0)
        : "memory"
    );
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_fa_o_norm_tile(
    ಸ್ಥಿರಾಂಕ __fp16 * d_diag,
    ಸ್ಥಿರಾಂಕ __fp16 * o_rc,
    __fp16 *       o_ಹೊರಹೋಗು
) {
    asm ಅಸ್ಥಿರ(
        HMX_ತುಂಬು_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    asm ಅಸ್ಥಿರ(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_ಹೊರಹೋಗು), "r"(0)
        : "memory"
    );
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_fa_q_prep_fp32_d2(
    __fp16 * vtcm_q_tiles, ಸ್ಥಿರಾಂಕ uint8_t * ತಾತ್ಕಾಲಿಕ_q_vtcm,
    size_t ಆರಂಭ, size_t ಕೊನೆ, size_t g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ,
    size_t DK, size_t G, size_t n_ಅಡ್ಡಸಾಲುಗಳು_q,
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_G, bool q_transposed
) {
    ಗಾಗಿ (size_t r = ಆರಂಭ; r < ಕೊನೆ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        size_t   r1       = r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        __fp16 * ಹೊರಹೋಗು_ತಳ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು * DK;

        ಆದರೆ (r >= g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ) {
            ((HVX_ಸದಿಶ *) (ಹೊರಹೋಗು_ತಳ + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            ((HVX_ಸದಿಶ *) (ಹೊರಹೋಗು_ತಳ + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ size_t q_idx0 = fastdiv(r + 0, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ಸ್ಥಿರಾಂಕ size_t q_idx1 = fastdiv(r + 1, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ಸ್ಥಿರಾಂಕ size_t offset0 = q_transposed ? (h_idx0 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx0) : (q_idx0 * G + h_idx0);
        ಸ್ಥಿರಾಂಕ size_t offset1 = q_transposed ? (h_idx1 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx1) : (q_idx1 * G + h_idx1);

        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset0 * DK * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in1 = (r + 1 < g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ)
            ? (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset1 * DK * ಗಾತ್ರ(ದಶಮಾಂಶ))
            : ಖಾಲಿ;

        {
            HVX_ಸದಿಶ v0 = pv_in0[0];
            HVX_ಸದಿಶ v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_ಸದಿಶ v_hf = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);
            ((HVX_ಸದಿಶ *) (ಹೊರಹೋಗು_ತಳ + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
        {
            HVX_ಸದಿಶ v0 = pv_in0[1];
            HVX_ಸದಿಶ v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_ಸದಿಶ v_hf = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);
            ((HVX_ಸದಿಶ *) (ಹೊರಹೋಗು_ತಳ + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_fa_q_prep_fp32_d4(
    __fp16 * vtcm_q_tiles, ಸ್ಥಿರಾಂಕ uint8_t * ತಾತ್ಕಾಲಿಕ_q_vtcm,
    size_t ಆರಂಭ, size_t ಕೊನೆ, size_t g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ,
    size_t DK, size_t G, size_t n_ಅಡ್ಡಸಾಲುಗಳು_q,
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_G, bool q_transposed
) {
    ಗಾಗಿ (size_t r = ಆರಂಭ; r < ಕೊನೆ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        size_t   r1       = r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        __fp16 * ಹೊರಹೋಗು_ತಳ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು * DK;

        ಆದರೆ (r >= g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ) {
            ಗಾಗಿ (uint32_t d = 0; d < 4; ++d) {
                ((HVX_ಸದಿಶ *) (ಹೊರಹೋಗು_ತಳ + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ size_t q_idx0 = fastdiv(r + 0, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ಸ್ಥಿರಾಂಕ size_t q_idx1 = fastdiv(r + 1, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ಸ್ಥಿರಾಂಕ size_t offset0 = q_transposed ? (h_idx0 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx0) : (q_idx0 * G + h_idx0);
        ಸ್ಥಿರಾಂಕ size_t offset1 = q_transposed ? (h_idx1 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx1) : (q_idx1 * G + h_idx1);

        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset0 * DK * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in1 = (r + 1 < g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ)
            ? (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset1 * DK * ಗಾತ್ರ(ದಶಮಾಂಶ))
            : ಖಾಲಿ;

        ಗಾಗಿ (uint32_t d = 0; d < 4; ++d) {
            HVX_ಸದಿಶ v0 = pv_in0[d];
            HVX_ಸದಿಶ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_ಸದಿಶ v_hf = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);
            ((HVX_ಸದಿಶ *) (ಹೊರಹೋಗು_ತಳ + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_fa_q_prep_fp32(
    __fp16 * vtcm_q_tiles, ಸ್ಥಿರಾಂಕ uint8_t * ತಾತ್ಕಾಲಿಕ_q_vtcm,
    size_t ಆರಂಭ, size_t ಕೊನೆ, size_t g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ,
    size_t DK, size_t G, size_t n_ಅಡ್ಡಸಾಲುಗಳು_q,
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_G, uint32_t d_ಎಲ್ಲೆ, bool q_transposed
) {
    ಗಾಗಿ (size_t r = ಆರಂಭ; r < ಕೊನೆ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        size_t   r1       = r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        __fp16 * ಹೊರಹೋಗು_ತಳ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು * DK;

        ಆದರೆ (r >= g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ) {
            ಗಾಗಿ (uint32_t d = 0; d < d_ಎಲ್ಲೆ; ++d) {
                ((HVX_ಸದಿಶ *) (ಹೊರಹೋಗು_ತಳ + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ size_t q_idx0 = fastdiv(r + 0, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ಸ್ಥಿರಾಂಕ size_t q_idx1 = fastdiv(r + 1, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ಸ್ಥಿರಾಂಕ size_t offset0 = q_transposed ? (h_idx0 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx0) : (q_idx0 * G + h_idx0);
        ಸ್ಥಿರಾಂಕ size_t offset1 = q_transposed ? (h_idx1 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx1) : (q_idx1 * G + h_idx1);

        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset0 * DK * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in1 = (r + 1 < g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ)
            ? (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset1 * DK * ಗಾತ್ರ(ದಶಮಾಂಶ))
            : ಖಾಲಿ;

        ಗಾಗಿ (uint32_t d = 0; d < d_ಎಲ್ಲೆ; ++d) {
            HVX_ಸದಿಶ v0   = pv_in0[d];
            HVX_ಸದಿಶ v1   = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_ಸದಿಶ v_hf = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);

            HVX_ಸದಿಶ * ಹೊರಹೋಗು_tile = (HVX_ಸದಿಶ *) (ಹೊರಹೋಗು_ತಳ + d * HMX_FP16_TILE_N_ELMS);
            ಹೊರಹೋಗು_tile[r1 / 2]      = v_hf;
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_fa_q_prep_fp16_d1(
    __fp16 * vtcm_q_tiles, ಸ್ಥಿರಾಂಕ uint8_t * ತಾತ್ಕಾಲಿಕ_q_vtcm,
    size_t ಆರಂಭ, size_t ಕೊನೆ, size_t g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ,
    size_t DK, size_t G, size_t n_ಅಡ್ಡಸಾಲುಗಳು_q,
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_G, bool q_transposed
) {
    ಗಾಗಿ (size_t r = ಆರಂಭ; r < ಕೊನೆ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        size_t   r1       = r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        __fp16 * ಹೊರಹೋಗು_ತಳ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು * DK;

        ಆದರೆ (r >= g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ) {
            __fp16 *     ಹೊರಹೋಗು_dtile = ಹೊರಹೋಗು_ತಳ + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ಸದಿಶ * pv_out0   = ((HVX_ಸದಿಶ *) ಹೊರಹೋಗು_dtile) + r1 / 2;
            HVX_ಸದಿಶ * pv_out1   = pv_out0 + 16;
            *pv_out0 = Q6_V_vzero();
            *pv_out1 = Q6_V_vzero();
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ size_t q_idx0 = fastdiv(r + 0, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ಸ್ಥಿರಾಂಕ size_t q_idx1 = fastdiv(r + 1, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ಸ್ಥಿರಾಂಕ size_t offset0 = q_transposed ? (h_idx0 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx0) : (q_idx0 * G + h_idx0);
        ಸ್ಥಿರಾಂಕ size_t offset1 = q_transposed ? (h_idx1 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx1) : (q_idx1 * G + h_idx1);

        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset0 * DK * ಗಾತ್ರ(__fp16));
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in1 = (r + 1 < g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ)
            ? (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset1 * DK * ಗಾತ್ರ(__fp16))
            : ಖಾಲಿ;

        HVX_ಸದಿಶ     v0 = pv_in0[0];
        HVX_ಸದಿಶ     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
        HVX_ಸದಿಶ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        __fp16 *     ಹೊರಹೋಗು_dtile = ಹೊರಹೋಗು_ತಳ + 0 * HMX_FP16_TILE_N_ELMS * 2;
        HVX_ಸದಿಶ * pv_out0   = ((HVX_ಸದಿಶ *) ಹೊರಹೋಗು_dtile) + r1 / 2;
        HVX_ಸದಿಶ * pv_out1   = pv_out0 + 16;

        *pv_out0 = Q6_V_lo_W(vp);
        *pv_out1 = Q6_V_hi_W(vp);
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_fa_q_prep_fp16_d2(
    __fp16 * vtcm_q_tiles, ಸ್ಥಿರಾಂಕ uint8_t * ತಾತ್ಕಾಲಿಕ_q_vtcm,
    size_t ಆರಂಭ, size_t ಕೊನೆ, size_t g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ,
    size_t DK, size_t G, size_t n_ಅಡ್ಡಸಾಲುಗಳು_q,
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_G, bool q_transposed
) {
    ಗಾಗಿ (size_t r = ಆರಂಭ; r < ಕೊನೆ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        size_t   r1       = r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        __fp16 * ಹೊರಹೋಗು_ತಳ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು * DK;

        ಆದರೆ (r >= g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ) {
            ಗಾಗಿ (uint32_t d = 0; d < 2; ++d) {
                __fp16 *     ಹೊರಹೋಗು_dtile = ಹೊರಹೋಗು_ತಳ + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ಸದಿಶ * pv_out0   = ((HVX_ಸದಿಶ *) ಹೊರಹೋಗು_dtile) + r1 / 2;
                HVX_ಸದಿಶ * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ size_t q_idx0 = fastdiv(r + 0, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ಸ್ಥಿರಾಂಕ size_t q_idx1 = fastdiv(r + 1, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ಸ್ಥಿರಾಂಕ size_t offset0 = q_transposed ? (h_idx0 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx0) : (q_idx0 * G + h_idx0);
        ಸ್ಥಿರಾಂಕ size_t offset1 = q_transposed ? (h_idx1 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx1) : (q_idx1 * G + h_idx1);

        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset0 * DK * ಗಾತ್ರ(__fp16));
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in1 = (r + 1 < g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ)
            ? (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset1 * DK * ಗಾತ್ರ(__fp16))
            : ಖಾಲಿ;

        {
            HVX_ಸದಿಶ     v0 = pv_in0[0];
            HVX_ಸದಿಶ     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_ಸದಿಶ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     ಹೊರಹೋಗು_dtile = ಹೊರಹೋಗು_ತಳ + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ಸದಿಶ * pv_out0   = ((HVX_ಸದಿಶ *) ಹೊರಹೋಗು_dtile) + r1 / 2;
            HVX_ಸದಿಶ * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
        {
            HVX_ಸದಿಶ     v0 = pv_in0[1];
            HVX_ಸದಿಶ     v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_ಸದಿಶ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     ಹೊರಹೋಗು_dtile = ಹೊರಹೋಗು_ತಳ + 1 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ಸದಿಶ * pv_out0   = ((HVX_ಸದಿಶ *) ಹೊರಹೋಗು_dtile) + r1 / 2;
            HVX_ಸದಿಶ * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_fa_q_prep_fp16(
    __fp16 * vtcm_q_tiles, ಸ್ಥಿರಾಂಕ uint8_t * ತಾತ್ಕಾಲಿಕ_q_vtcm,
    size_t ಆರಂಭ, size_t ಕೊನೆ, size_t g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ,
    size_t DK, size_t G, size_t n_ಅಡ್ಡಸಾಲುಗಳು_q,
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_G, uint32_t d_ಎಲ್ಲೆ, bool q_transposed
) {
    ಗಾಗಿ (size_t r = ಆರಂಭ; r < ಕೊನೆ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        size_t   r1       = r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        __fp16 * ಹೊರಹೋಗು_ತಳ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು * DK;

        ಆದರೆ (r >= g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ) {
            ಗಾಗಿ (uint32_t d = 0; d < d_ಎಲ್ಲೆ; ++d) {
                __fp16 *     ಹೊರಹೋಗು_dtile = ಹೊರಹೋಗು_ತಳ + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ಸದಿಶ * pv_out0   = ((HVX_ಸದಿಶ *) ಹೊರಹೋಗು_dtile) + r1 / 2;
                HVX_ಸದಿಶ * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ size_t q_idx0 = fastdiv(r + 0, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ಸ್ಥಿರಾಂಕ size_t q_idx1 = fastdiv(r + 1, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ಸ್ಥಿರಾಂಕ size_t offset0 = q_transposed ? (h_idx0 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx0) : (q_idx0 * G + h_idx0);
        ಸ್ಥಿರಾಂಕ size_t offset1 = q_transposed ? (h_idx1 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx1) : (q_idx1 * G + h_idx1);

        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset0 * DK * ಗಾತ್ರ(__fp16));
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in1 = (r + 1 < g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ)
            ? (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset1 * DK * ಗಾತ್ರ(__fp16))
            : ಖಾಲಿ;

        ಗಾಗಿ (uint32_t d = 0; d < d_ಎಲ್ಲೆ; ++d) {
            HVX_ಸದಿಶ     v0 = pv_in0[d];
            HVX_ಸದಿಶ     v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_ಸದಿಶ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     ಹೊರಹೋಗು_dtile = ಹೊರಹೋಗು_ತಳ + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ಸದಿಶ * pv_out0   = ((HVX_ಸದಿಶ *) ಹೊರಹೋಗು_dtile) + r1 / 2;
            HVX_ಸದಿಶ * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}


// Head-dim-padded Q-prep (f32). Used when DK is not a multiple of 64.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_fa_q_prep_fp32_pad(__fp16 *                      vtcm_q_tiles,
                                          ಸ್ಥಿರಾಂಕ uint8_t *               ತಾತ್ಕಾಲಿಕ_q_vtcm,
                                          size_t                        ಆರಂಭ,
                                          size_t                        ಕೊನೆ,
                                          size_t                        g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ,
                                          size_t                        dk_ಒಳ,
                                          size_t                        dk_ಹೊರಹೋಗು,
                                          size_t                        G,
                                          size_t                        n_ಅಡ್ಡಸಾಲುಗಳು_q,
                                          ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_G,
                                          bool                          q_transposed) {
    ಸ್ಥಿರಾಂಕ uint32_t n_ಹೊರಹೋಗು_tiles = (uint32_t) (dk_ಹೊರಹೋಗು / 32);
    ಗಾಗಿ (size_t r = ಆರಂಭ; r < ಕೊನೆ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        size_t   r1       = r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        __fp16 * ಹೊರಹೋಗು_ತಳ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು * dk_ಹೊರಹೋಗು;

        ಆದರೆ (r >= g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ) {
            ಗಾಗಿ (uint32_t d = 0; d < n_ಹೊರಹೋಗು_tiles; ++d) {
                ((HVX_ಸದಿಶ *) (ಹೊರಹೋಗು_ತಳ + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ size_t q_idx0 = fastdiv(r + 0, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ಸ್ಥಿರಾಂಕ size_t q_idx1 = fastdiv(r + 1, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ಸ್ಥಿರಾಂಕ size_t offset0 = q_transposed ? (h_idx0 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx0) : (q_idx0 * G + h_idx0);
        ಸ್ಥಿರಾಂಕ size_t offset1 = q_transposed ? (h_idx1 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx1) : (q_idx1 * G + h_idx1);

        ಸ್ಥಿರಾಂಕ HVX_UVector * pv_in0 = (ಸ್ಥಿರಾಂಕ HVX_UVector *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset0 * dk_ಒಳ * ಗಾತ್ರ(ದಶಮಾಂಶ));
        ಸ್ಥಿರಾಂಕ HVX_UVector * pv_in1 = (r + 1 < g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ) ? (ಸ್ಥಿರಾಂಕ HVX_UVector *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset1 * dk_ಒಳ * ಗಾತ್ರ(ದಶಮಾಂಶ)) : ಖಾಲಿ;

        ಗಾಗಿ (uint32_t d = 0; d < n_ಹೊರಹೋಗು_tiles; ++d) {
            HVX_ಸದಿಶ * ಹೊರಹೋಗು_tile   = (HVX_ಸದಿಶ *) (ಹೊರಹೋಗು_ತಳ + d * HMX_FP16_TILE_N_ELMS);
            ಸ್ಥಿರಾಂಕ size_t ತಳ_lane  = (size_t) d * 32;
            ಸ್ಥಿರಾಂಕ size_t real_lanes = (ತಳ_lane < dk_ಒಳ) ? hex_smin(32, dk_ಒಳ - ತಳ_lane) : 0;

            ಆದರೆ (real_lanes == 0) {
                ಹೊರಹೋಗು_tile[r1 / 2] = Q6_V_vzero();
                ಮುಂದುವರಿ;
            }

            HVX_ಸದಿಶ v0 = pv_in0[d];
            HVX_ಸದಿಶ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            ಆದರೆ (real_lanes < 32) {
                // Straddle tile: keep the first real_lanes floats, zero the padded tail so
                // the packed f16 lanes beyond DK are zero.
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * ಗಾತ್ರ(ದಶಮಾಂಶ)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            ಹೊರಹೋಗು_tile[r1 / 2] = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);
        }
    }
}

// Head-dim-padded Q-prep (f16). Used when DK is not a multiple of 64.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_fa_q_prep_fp16_pad(__fp16 *                      vtcm_q_tiles,
                                          ಸ್ಥಿರಾಂಕ uint8_t *               ತಾತ್ಕಾಲಿಕ_q_vtcm,
                                          size_t                        ಆರಂಭ,
                                          size_t                        ಕೊನೆ,
                                          size_t                        g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ,
                                          size_t                        dk_ಒಳ,
                                          size_t                        dk_ಹೊರಹೋಗು,
                                          size_t                        G,
                                          size_t                        n_ಅಡ್ಡಸಾಲುಗಳು_q,
                                          ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_G,
                                          bool                          q_transposed) {
    ಸ್ಥಿರಾಂಕ uint32_t n_ಹೊರಹೋಗು_ಜೋಡಿಗಳು = (uint32_t) (dk_ಹೊರಹೋಗು / 64);
    ಗಾಗಿ (size_t r = ಆರಂಭ; r < ಕೊನೆ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        size_t   r1       = r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        __fp16 * ಹೊರಹೋಗು_ತಳ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು * dk_ಹೊರಹೋಗು;

        ಆದರೆ (r >= g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ) {
            ಗಾಗಿ (uint32_t d = 0; d < n_ಹೊರಹೋಗು_ಜೋಡಿಗಳು; ++d) {
                __fp16 *     ಹೊರಹೋಗು_dtile = ಹೊರಹೋಗು_ತಳ + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ಸದಿಶ * pv_out0   = ((HVX_ಸದಿಶ *) ಹೊರಹೋಗು_dtile) + r1 / 2;
                HVX_ಸದಿಶ * pv_out1   = pv_out0 + 16;
                *pv_out0               = Q6_V_vzero();
                *pv_out1               = Q6_V_vzero();
            }
            ಮುಂದುವರಿ;
        }

        ಸ್ಥಿರಾಂಕ size_t q_idx0 = fastdiv(r + 0, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ಸ್ಥಿರಾಂಕ size_t q_idx1 = fastdiv(r + 1, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ಸ್ಥಿರಾಂಕ size_t offset0 = q_transposed ? (h_idx0 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx0) : (q_idx0 * G + h_idx0);
        ಸ್ಥಿರಾಂಕ size_t offset1 = q_transposed ? (h_idx1 * n_ಅಡ್ಡಸಾಲುಗಳು_q + q_idx1) : (q_idx1 * G + h_idx1);

        ಸ್ಥಿರಾಂಕ HVX_UVector * pv_in0 = (ಸ್ಥಿರಾಂಕ HVX_UVector *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset0 * dk_ಒಳ * ಗಾತ್ರ(__fp16));
        ಸ್ಥಿರಾಂಕ HVX_UVector * pv_in1 = (r + 1 < g_ಅಡ್ಡಸಾಲುಗಳು_ಕೊನೆ) ? (ಸ್ಥಿರಾಂಕ HVX_UVector *) (ತಾತ್ಕಾಲಿಕ_q_vtcm + offset1 * dk_ಒಳ * ಗಾತ್ರ(__fp16)) : ಖಾಲಿ;

        ಗಾಗಿ (uint32_t d = 0; d < n_ಹೊರಹೋಗು_ಜೋಡಿಗಳು; ++d) {
            __fp16 *     ಹೊರಹೋಗು_dtile = ಹೊರಹೋಗು_ತಳ + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ಸದಿಶ * pv_out0   = ((HVX_ಸದಿಶ *) ಹೊರಹೋಗು_dtile) + r1 / 2;
            HVX_ಸದಿಶ * pv_out1   = pv_out0 + 16;

            ಸ್ಥಿರಾಂಕ size_t ತಳ_lane  = (size_t) d * 64;
            ಸ್ಥಿರಾಂಕ size_t real_lanes = (ತಳ_lane < dk_ಒಳ) ? hex_smin(64, dk_ಒಳ - ತಳ_lane) : 0;

            ಆದರೆ (real_lanes == 0) {
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
                ಮುಂದುವರಿ;
            }

            HVX_ಸದಿಶ v0 = pv_in0[d];
            HVX_ಸದಿಶ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            ಆದರೆ (real_lanes < 64) {
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * ಗಾತ್ರ(__fp16)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            HVX_ಸದಿಶ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);
            *pv_out0          = Q6_V_lo_W(vp);
            *pv_out1          = Q6_V_hi_W(vp);
        }
    }
}

ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_fa_q_prep_fallback(
    __fp16 * vtcm_q_tiles, uintptr_t q_ದತ್ತಾಂಶ,
    size_t q_nb1, size_t q_nb2, size_t q_nb3,
    uint32_t q_ಆರಂಭ, uint32_t kv_ತಲೆ, uint32_t ib3,
    size_t ಆರಂಭ, size_t ಕೊನೆ, size_t n_ಅಡ್ಡಸಾಲುಗಳು_g,
    size_t G, size_t dk_ಒಳ, size_t dk_ಹೊರಹೋಗು, bool is_q_fp32,
    ಸ್ಥಿರಾಂಕ ರಚನೆ fastdiv_ಮೌಲ್ಯಗಳು * div_G
) {
    ಗಾಗಿ (size_t r = ಆರಂಭ; r < ಕೊನೆ; r += 2) {
        ಸ್ಥಿರಾಂಕ size_t q_idx0 = fastdiv(r + 0, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ಸ್ಥಿರಾಂಕ size_t q_idx1 = fastdiv(r + 1, div_G);
        ಸ್ಥಿರಾಂಕ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ಸ್ಥಿರಾಂಕ uint8_t * q_ptr0 = (r + 0 < n_ಅಡ್ಡಸಾಲುಗಳು_g) ? ((ಸ್ಥಿರಾಂಕ uint8_t *) q_ದತ್ತಾಂಶ + (q_ಆರಂಭ + q_idx0) * q_nb1 +
                                                      (kv_ತಲೆ * G + h_idx0) * q_nb2 + ib3 * q_nb3) :
                                                      ಖಾಲಿ;
        ಸ್ಥಿರಾಂಕ uint8_t * q_ptr1 = (r + 1 < n_ಅಡ್ಡಸಾಲುಗಳು_g) ? ((ಸ್ಥಿರಾಂಕ uint8_t *) q_ದತ್ತಾಂಶ + (q_ಆರಂಭ + q_idx1) * q_nb1 +
                                                      (kv_ತಲೆ * G + h_idx1) * q_nb2 + ib3 * q_nb3) :
                                                      ಖಾಲಿ;

        size_t   r0       = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        size_t   r1       = r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        __fp16 * ಹೊರಹೋಗು_ತಳ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು * dk_ಹೊರಹೋಗು;

        ಆದರೆ (is_q_fp32) {
            ಸ್ಥಿರಾಂಕ HVX_UVector * pv_in0 = q_ptr0 ? (ಸ್ಥಿರಾಂಕ HVX_UVector *) q_ptr0 : ಖಾಲಿ;
            ಸ್ಥಿರಾಂಕ HVX_UVector * pv_in1 = q_ptr1 ? (ಸ್ಥಿರಾಂಕ HVX_UVector *) q_ptr1 : ಖಾಲಿ;

            ಗಾಗಿ (uint32_t d = 0; d < dk_ಹೊರಹೋಗು / 32; ++d) {
                HVX_ಸದಿಶ * ಹೊರಹೋಗು_tile   = (HVX_ಸದಿಶ *) (ಹೊರಹೋಗು_ತಳ + d * HMX_FP16_TILE_N_ELMS);
                ಸ್ಥಿರಾಂಕ size_t ತಳ_lane  = (size_t) d * 32;
                ಸ್ಥಿರಾಂಕ size_t real_lanes = (ತಳ_lane < dk_ಒಳ) ? hex_smin(32, dk_ಒಳ - ತಳ_lane) : 0;

                ಆದರೆ (real_lanes == 0) {
                    ಹೊರಹೋಗು_tile[r1 / 2] = Q6_V_vzero();
                    ಮುಂದುವರಿ;
                }
                HVX_ಸದಿಶ v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_ಸದಿಶ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                ಆದರೆ (real_lanes < 32) {
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * ಗಾತ್ರ(ದಶಮಾಂಶ)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                ಹೊರಹೋಗು_tile[r1 / 2] = hvx_vec_f32_ಗಮ್ಯ_f16_shuff(v0, v1);
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ HVX_UVector * pv_in0 = q_ptr0 ? (ಸ್ಥಿರಾಂಕ HVX_UVector *) q_ptr0 : ಖಾಲಿ;
            ಸ್ಥಿರಾಂಕ HVX_UVector * pv_in1 = q_ptr1 ? (ಸ್ಥಿರಾಂಕ HVX_UVector *) q_ptr1 : ಖಾಲಿ;

            ಗಾಗಿ (uint32_t d = 0; d < dk_ಹೊರಹೋಗು / 64; ++d) {
                __fp16 *     ಹೊರಹೋಗು_dtile = ಹೊರಹೋಗು_ತಳ + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ಸದಿಶ * pv_out0   = ((HVX_ಸದಿಶ *) ಹೊರಹೋಗು_dtile) + r1 / 2;
                HVX_ಸದಿಶ * pv_out1   = pv_out0 + 16;

                ಸ್ಥಿರಾಂಕ size_t ತಳ_lane  = (size_t) d * 64;
                ಸ್ಥಿರಾಂಕ size_t real_lanes = (ತಳ_lane < dk_ಒಳ) ? hex_smin(64, dk_ಒಳ - ತಳ_lane) : 0;

                ಆದರೆ (real_lanes == 0) {
                    *pv_out0 = Q6_V_vzero();
                    *pv_out1 = Q6_V_vzero();
                    ಮುಂದುವರಿ;
                }
                HVX_ಸದಿಶ v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_ಸದಿಶ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                ಆದರೆ (real_lanes < 64) {
                    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * ಗಾತ್ರ(__fp16)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                HVX_ಸದಿಶ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);
                *pv_out0 = Q6_V_lo_W(vp);
                *pv_out1 = Q6_V_hi_W(vp);
            }
        }
    }
}

#ಆದರೆ_ಅಂತ್ಯ /* HMX_FA_KERNELS_H */
