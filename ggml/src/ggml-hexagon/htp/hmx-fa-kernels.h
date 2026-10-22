#ifndef HMX_FA_KERNELS_H
#define HMX_FA_KERNELS_H

#അന്തര്ഭാവ <stdint.h>
#അന്തര്ഭാവ <stddef.h>
#അന്തര്ഭാവ <stdbool.h>
#അന്തര്ഭാവ "hvx-utils.h"
#അന്തര്ഭാവ "hmx-utils.h"
#അന്തര്ഭാവ "hex-fastdiv.h"

// HMX-specific parameters, offsets and inner kernels for Flash Attention

// Scatter offsets for diagonal tile: entry[2i] = i*136, entry[2i+1] = i*136+6
// 136 = 4 * 32 + 8 = byte offset to diagonal in a 32x32 fp16 interleaved tile
സ്ഥിതം സ്ഥിരം int16_t d_tile_scatter_offsets[64] __attribute__((aligned(128))) = {
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

സ്ഥിതം വരിയിൽ ശൂന്യം hmx_fa_qk_dot_tile(
    സ്ഥിരം __fp16 * row_tiles,
    സ്ഥിരം __fp16 * col_tiles,
    __fp16 *       out_tile,
    size_t         n_dot_tiles
) {
    എങ്കിൽ (n_dot_tiles == 2) {
        asm അസ്ഥിര(
            HMX_LOAD_MPY_F16("%1", "%2", "%0")
            HMX_LOAD_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(row_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } അല്ലെങ്കിൽ എങ്കിൽ (n_dot_tiles == 4) {
        asm അസ്ഥിര(
            HMX_LOAD_MPY_F16("%1", "%2", "%0")
            HMX_LOAD_MPY_F16("%3", "%4", "%0")
            HMX_LOAD_MPY_F16("%5", "%6", "%0")
            HMX_LOAD_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(row_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } അല്ലെങ്കിൽ എങ്കിൽ (n_dot_tiles == 8) {
        asm അസ്ഥിര(
            HMX_LOAD_MPY_F16("%1", "%2", "%0")
            HMX_LOAD_MPY_F16("%3", "%4", "%0")
            HMX_LOAD_MPY_F16("%5", "%6", "%0")
            HMX_LOAD_MPY_F16("%7", "%8", "%0")
            HMX_LOAD_MPY_F16("%9", "%10", "%0")
            HMX_LOAD_MPY_F16("%11", "%12", "%0")
            HMX_LOAD_MPY_F16("%13", "%14", "%0")
            HMX_LOAD_MPY_F16("%15", "%16", "%0")
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
    } അല്ലെങ്കിൽ {
        വേണ്ടി (size_t k = 0; k < n_dot_tiles; ++k) {
            asm അസ്ഥിര(
                HMX_LOAD_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(row_tiles), "r"(col_tiles)
            );
            row_tiles += HMX_FP16_TILE_N_ELMS;
            col_tiles += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm അസ്ഥിര(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(out_tile), "r"(0)
        : "memory"
    );
}

സ്ഥിതം വരിയിൽ ശൂന്യം hmx_fa_o_update_tile(
    സ്ഥിരം __fp16 * d_diag,
    സ്ഥിരം __fp16 * o_rc,
    സ്ഥിരം __fp16 * p_tile_in,
    സ്ഥിരം __fp16 * v_tile_in,
    __fp16 *       o_tile_out,
    size_t         n_col_tiles
) {
    asm അസ്ഥിര(
        HMX_LOAD_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    എങ്കിൽ (n_col_tiles == 2) {
        asm അസ്ഥിര(
            HMX_LOAD_MPY_F16("%1", "%2", "%0")
            HMX_LOAD_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(p_tile_in + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } അല്ലെങ്കിൽ എങ്കിൽ (n_col_tiles == 4) {
        asm അസ്ഥിര(
            HMX_LOAD_MPY_F16("%1", "%2", "%0")
            HMX_LOAD_MPY_F16("%3", "%4", "%0")
            HMX_LOAD_MPY_F16("%5", "%6", "%0")
            HMX_LOAD_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(p_tile_in + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } അല്ലെങ്കിൽ എങ്കിൽ (n_col_tiles == 8) {
        asm അസ്ഥിര(
            HMX_LOAD_MPY_F16("%1", "%2", "%0")
            HMX_LOAD_MPY_F16("%3", "%4", "%0")
            HMX_LOAD_MPY_F16("%5", "%6", "%0")
            HMX_LOAD_MPY_F16("%7", "%8", "%0")
            HMX_LOAD_MPY_F16("%9", "%10", "%0")
            HMX_LOAD_MPY_F16("%11", "%12", "%0")
            HMX_LOAD_MPY_F16("%13", "%14", "%0")
            HMX_LOAD_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(p_tile_in + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 4 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 5 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 6 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 7 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } അല്ലെങ്കിൽ {
        വേണ്ടി (size_t k = 0; k < n_col_tiles; ++k) {
            asm അസ്ഥിര(
                HMX_LOAD_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(p_tile_in), "r"(v_tile_in)
            );
            p_tile_in += HMX_FP16_TILE_N_ELMS;
            v_tile_in += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm അസ്ഥിര(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_tile_out), "r"(0)
        : "memory"
    );
}

സ്ഥിതം വരിയിൽ ശൂന്യം hmx_fa_o_norm_tile(
    സ്ഥിരം __fp16 * d_diag,
    സ്ഥിരം __fp16 * o_rc,
    __fp16 *       o_out
) {
    asm അസ്ഥിര(
        HMX_LOAD_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    asm അസ്ഥിര(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_out), "r"(0)
        : "memory"
    );
}

സ്ഥിതം വരിയിൽ ശൂന്യം hmx_fa_q_prep_fp32_d2(
    __fp16 * vtcm_q_tiles, സ്ഥിരം uint8_t * temp_q_vtcm,
    size_t start, size_t end, size_t g_rows_end,
    size_t DK, size_t G, size_t n_rows_q,
    സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div_G, bool q_transposed
) {
    വേണ്ടി (size_t r = start; r < end; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ROWS;
        size_t   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * DK;

        എങ്കിൽ (r >= g_rows_end) {
            ((HVX_Vector *) (out_base + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            ((HVX_Vector *) (out_base + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            തുടരുക;
        }

        സ്ഥിരം size_t q_idx0 = fastdiv(r + 0, div_G);
        സ്ഥിരം size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        സ്ഥിരം size_t q_idx1 = fastdiv(r + 1, div_G);
        സ്ഥിരം size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        സ്ഥിരം size_t offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        സ്ഥിരം size_t offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        സ്ഥിരം HVX_Vector * pv_in0 = (സ്ഥിരം HVX_Vector *) (temp_q_vtcm + offset0 * DK * വലുപ്പം(ദശാംശം));
        സ്ഥിരം HVX_Vector * pv_in1 = (r + 1 < g_rows_end)
            ? (സ്ഥിരം HVX_Vector *) (temp_q_vtcm + offset1 * DK * വലുപ്പം(ദശാംശം))
            : ശൂന്യ;

        {
            HVX_Vector v0 = pv_in0[0];
            HVX_Vector v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_Vector v_hf = hvx_vec_f32_to_f16_shuff(v0, v1);
            ((HVX_Vector *) (out_base + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
        {
            HVX_Vector v0 = pv_in0[1];
            HVX_Vector v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_Vector v_hf = hvx_vec_f32_to_f16_shuff(v0, v1);
            ((HVX_Vector *) (out_base + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം hmx_fa_q_prep_fp32_d4(
    __fp16 * vtcm_q_tiles, സ്ഥിരം uint8_t * temp_q_vtcm,
    size_t start, size_t end, size_t g_rows_end,
    size_t DK, size_t G, size_t n_rows_q,
    സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div_G, bool q_transposed
) {
    വേണ്ടി (size_t r = start; r < end; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ROWS;
        size_t   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * DK;

        എങ്കിൽ (r >= g_rows_end) {
            വേണ്ടി (uint32_t d = 0; d < 4; ++d) {
                ((HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            തുടരുക;
        }

        സ്ഥിരം size_t q_idx0 = fastdiv(r + 0, div_G);
        സ്ഥിരം size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        സ്ഥിരം size_t q_idx1 = fastdiv(r + 1, div_G);
        സ്ഥിരം size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        സ്ഥിരം size_t offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        സ്ഥിരം size_t offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        സ്ഥിരം HVX_Vector * pv_in0 = (സ്ഥിരം HVX_Vector *) (temp_q_vtcm + offset0 * DK * വലുപ്പം(ദശാംശം));
        സ്ഥിരം HVX_Vector * pv_in1 = (r + 1 < g_rows_end)
            ? (സ്ഥിരം HVX_Vector *) (temp_q_vtcm + offset1 * DK * വലുപ്പം(ദശാംശം))
            : ശൂന്യ;

        വേണ്ടി (uint32_t d = 0; d < 4; ++d) {
            HVX_Vector v0 = pv_in0[d];
            HVX_Vector v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_Vector v_hf = hvx_vec_f32_to_f16_shuff(v0, v1);
            ((HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം hmx_fa_q_prep_fp32(
    __fp16 * vtcm_q_tiles, സ്ഥിരം uint8_t * temp_q_vtcm,
    size_t start, size_t end, size_t g_rows_end,
    size_t DK, size_t G, size_t n_rows_q,
    സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div_G, uint32_t d_limit, bool q_transposed
) {
    വേണ്ടി (size_t r = start; r < end; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ROWS;
        size_t   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * DK;

        എങ്കിൽ (r >= g_rows_end) {
            വേണ്ടി (uint32_t d = 0; d < d_limit; ++d) {
                ((HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            തുടരുക;
        }

        സ്ഥിരം size_t q_idx0 = fastdiv(r + 0, div_G);
        സ്ഥിരം size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        സ്ഥിരം size_t q_idx1 = fastdiv(r + 1, div_G);
        സ്ഥിരം size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        സ്ഥിരം size_t offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        സ്ഥിരം size_t offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        സ്ഥിരം HVX_Vector * pv_in0 = (സ്ഥിരം HVX_Vector *) (temp_q_vtcm + offset0 * DK * വലുപ്പം(ദശാംശം));
        സ്ഥിരം HVX_Vector * pv_in1 = (r + 1 < g_rows_end)
            ? (സ്ഥിരം HVX_Vector *) (temp_q_vtcm + offset1 * DK * വലുപ്പം(ദശാംശം))
            : ശൂന്യ;

        വേണ്ടി (uint32_t d = 0; d < d_limit; ++d) {
            HVX_Vector v0   = pv_in0[d];
            HVX_Vector v1   = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_Vector v_hf = hvx_vec_f32_to_f16_shuff(v0, v1);

            HVX_Vector * out_tile = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
            out_tile[r1 / 2]      = v_hf;
        }
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം hmx_fa_q_prep_fp16_d1(
    __fp16 * vtcm_q_tiles, സ്ഥിരം uint8_t * temp_q_vtcm,
    size_t start, size_t end, size_t g_rows_end,
    size_t DK, size_t G, size_t n_rows_q,
    സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div_G, bool q_transposed
) {
    വേണ്ടി (size_t r = start; r < end; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ROWS;
        size_t   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * DK;

        എങ്കിൽ (r >= g_rows_end) {
            __fp16 *     out_dtile = out_base + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
            HVX_Vector * pv_out1   = pv_out0 + 16;
            *pv_out0 = Q6_V_vzero();
            *pv_out1 = Q6_V_vzero();
            തുടരുക;
        }

        സ്ഥിരം size_t q_idx0 = fastdiv(r + 0, div_G);
        സ്ഥിരം size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        സ്ഥിരം size_t q_idx1 = fastdiv(r + 1, div_G);
        സ്ഥിരം size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        സ്ഥിരം size_t offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        സ്ഥിരം size_t offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        സ്ഥിരം HVX_Vector * pv_in0 = (സ്ഥിരം HVX_Vector *) (temp_q_vtcm + offset0 * DK * വലുപ്പം(__fp16));
        സ്ഥിരം HVX_Vector * pv_in1 = (r + 1 < g_rows_end)
            ? (സ്ഥിരം HVX_Vector *) (temp_q_vtcm + offset1 * DK * വലുപ്പം(__fp16))
            : ശൂന്യ;

        HVX_Vector     v0 = pv_in0[0];
        HVX_Vector     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
        HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        __fp16 *     out_dtile = out_base + 0 * HMX_FP16_TILE_N_ELMS * 2;
        HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
        HVX_Vector * pv_out1   = pv_out0 + 16;

        *pv_out0 = Q6_V_lo_W(vp);
        *pv_out1 = Q6_V_hi_W(vp);
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം hmx_fa_q_prep_fp16_d2(
    __fp16 * vtcm_q_tiles, സ്ഥിരം uint8_t * temp_q_vtcm,
    size_t start, size_t end, size_t g_rows_end,
    size_t DK, size_t G, size_t n_rows_q,
    സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div_G, bool q_transposed
) {
    വേണ്ടി (size_t r = start; r < end; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ROWS;
        size_t   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * DK;

        എങ്കിൽ (r >= g_rows_end) {
            വേണ്ടി (uint32_t d = 0; d < 2; ++d) {
                __fp16 *     out_dtile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
                HVX_Vector * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            തുടരുക;
        }

        സ്ഥിരം size_t q_idx0 = fastdiv(r + 0, div_G);
        സ്ഥിരം size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        സ്ഥിരം size_t q_idx1 = fastdiv(r + 1, div_G);
        സ്ഥിരം size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        സ്ഥിരം size_t offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        സ്ഥിരം size_t offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        സ്ഥിരം HVX_Vector * pv_in0 = (സ്ഥിരം HVX_Vector *) (temp_q_vtcm + offset0 * DK * വലുപ്പം(__fp16));
        സ്ഥിരം HVX_Vector * pv_in1 = (r + 1 < g_rows_end)
            ? (സ്ഥിരം HVX_Vector *) (temp_q_vtcm + offset1 * DK * വലുപ്പം(__fp16))
            : ശൂന്യ;

        {
            HVX_Vector     v0 = pv_in0[0];
            HVX_Vector     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     out_dtile = out_base + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
            HVX_Vector * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
        {
            HVX_Vector     v0 = pv_in0[1];
            HVX_Vector     v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     out_dtile = out_base + 1 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
            HVX_Vector * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം hmx_fa_q_prep_fp16(
    __fp16 * vtcm_q_tiles, സ്ഥിരം uint8_t * temp_q_vtcm,
    size_t start, size_t end, size_t g_rows_end,
    size_t DK, size_t G, size_t n_rows_q,
    സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div_G, uint32_t d_limit, bool q_transposed
) {
    വേണ്ടി (size_t r = start; r < end; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ROWS;
        size_t   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * DK;

        എങ്കിൽ (r >= g_rows_end) {
            വേണ്ടി (uint32_t d = 0; d < d_limit; ++d) {
                __fp16 *     out_dtile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
                HVX_Vector * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            തുടരുക;
        }

        സ്ഥിരം size_t q_idx0 = fastdiv(r + 0, div_G);
        സ്ഥിരം size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        സ്ഥിരം size_t q_idx1 = fastdiv(r + 1, div_G);
        സ്ഥിരം size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        സ്ഥിരം size_t offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        സ്ഥിരം size_t offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        സ്ഥിരം HVX_Vector * pv_in0 = (സ്ഥിരം HVX_Vector *) (temp_q_vtcm + offset0 * DK * വലുപ്പം(__fp16));
        സ്ഥിരം HVX_Vector * pv_in1 = (r + 1 < g_rows_end)
            ? (സ്ഥിരം HVX_Vector *) (temp_q_vtcm + offset1 * DK * വലുപ്പം(__fp16))
            : ശൂന്യ;

        വേണ്ടി (uint32_t d = 0; d < d_limit; ++d) {
            HVX_Vector     v0 = pv_in0[d];
            HVX_Vector     v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     out_dtile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
            HVX_Vector * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}


// Head-dim-padded Q-prep (f32). Used when DK is not a multiple of 64.
സ്ഥിതം വരിയിൽ ശൂന്യം hmx_fa_q_prep_fp32_pad(__fp16 *                      vtcm_q_tiles,
                                          സ്ഥിരം uint8_t *               temp_q_vtcm,
                                          size_t                        start,
                                          size_t                        end,
                                          size_t                        g_rows_end,
                                          size_t                        dk_in,
                                          size_t                        dk_out,
                                          size_t                        G,
                                          size_t                        n_rows_q,
                                          സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div_G,
                                          bool                          q_transposed) {
    സ്ഥിരം uint32_t n_out_tiles = (uint32_t) (dk_out / 32);
    വേണ്ടി (size_t r = start; r < end; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ROWS;
        size_t   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * dk_out;

        എങ്കിൽ (r >= g_rows_end) {
            വേണ്ടി (uint32_t d = 0; d < n_out_tiles; ++d) {
                ((HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            തുടരുക;
        }

        സ്ഥിരം size_t q_idx0 = fastdiv(r + 0, div_G);
        സ്ഥിരം size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        സ്ഥിരം size_t q_idx1 = fastdiv(r + 1, div_G);
        സ്ഥിരം size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        സ്ഥിരം size_t offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        സ്ഥിരം size_t offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        സ്ഥിരം HVX_UVector * pv_in0 = (സ്ഥിരം HVX_UVector *) (temp_q_vtcm + offset0 * dk_in * വലുപ്പം(ദശാംശം));
        സ്ഥിരം HVX_UVector * pv_in1 = (r + 1 < g_rows_end) ? (സ്ഥിരം HVX_UVector *) (temp_q_vtcm + offset1 * dk_in * വലുപ്പം(ദശാംശം)) : ശൂന്യ;

        വേണ്ടി (uint32_t d = 0; d < n_out_tiles; ++d) {
            HVX_Vector * out_tile   = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
            സ്ഥിരം size_t base_lane  = (size_t) d * 32;
            സ്ഥിരം size_t real_lanes = (base_lane < dk_in) ? hex_smin(32, dk_in - base_lane) : 0;

            എങ്കിൽ (real_lanes == 0) {
                out_tile[r1 / 2] = Q6_V_vzero();
                തുടരുക;
            }

            HVX_Vector v0 = pv_in0[d];
            HVX_Vector v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            എങ്കിൽ (real_lanes < 32) {
                // Straddle tile: keep the first real_lanes floats, zero the padded tail so
                // the packed f16 lanes beyond DK are zero.
                സ്ഥിരം HVX_VectorPred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * വലുപ്പം(ദശാംശം)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            out_tile[r1 / 2] = hvx_vec_f32_to_f16_shuff(v0, v1);
        }
    }
}

// Head-dim-padded Q-prep (f16). Used when DK is not a multiple of 64.
സ്ഥിതം വരിയിൽ ശൂന്യം hmx_fa_q_prep_fp16_pad(__fp16 *                      vtcm_q_tiles,
                                          സ്ഥിരം uint8_t *               temp_q_vtcm,
                                          size_t                        start,
                                          size_t                        end,
                                          size_t                        g_rows_end,
                                          size_t                        dk_in,
                                          size_t                        dk_out,
                                          size_t                        G,
                                          size_t                        n_rows_q,
                                          സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div_G,
                                          bool                          q_transposed) {
    സ്ഥിരം uint32_t n_out_pairs = (uint32_t) (dk_out / 64);
    വേണ്ടി (size_t r = start; r < end; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ROWS;
        size_t   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * dk_out;

        എങ്കിൽ (r >= g_rows_end) {
            വേണ്ടി (uint32_t d = 0; d < n_out_pairs; ++d) {
                __fp16 *     out_dtile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
                HVX_Vector * pv_out1   = pv_out0 + 16;
                *pv_out0               = Q6_V_vzero();
                *pv_out1               = Q6_V_vzero();
            }
            തുടരുക;
        }

        സ്ഥിരം size_t q_idx0 = fastdiv(r + 0, div_G);
        സ്ഥിരം size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        സ്ഥിരം size_t q_idx1 = fastdiv(r + 1, div_G);
        സ്ഥിരം size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        സ്ഥിരം size_t offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        സ്ഥിരം size_t offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        സ്ഥിരം HVX_UVector * pv_in0 = (സ്ഥിരം HVX_UVector *) (temp_q_vtcm + offset0 * dk_in * വലുപ്പം(__fp16));
        സ്ഥിരം HVX_UVector * pv_in1 = (r + 1 < g_rows_end) ? (സ്ഥിരം HVX_UVector *) (temp_q_vtcm + offset1 * dk_in * വലുപ്പം(__fp16)) : ശൂന്യ;

        വേണ്ടി (uint32_t d = 0; d < n_out_pairs; ++d) {
            __fp16 *     out_dtile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
            HVX_Vector * pv_out1   = pv_out0 + 16;

            സ്ഥിരം size_t base_lane  = (size_t) d * 64;
            സ്ഥിരം size_t real_lanes = (base_lane < dk_in) ? hex_smin(64, dk_in - base_lane) : 0;

            എങ്കിൽ (real_lanes == 0) {
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
                തുടരുക;
            }

            HVX_Vector v0 = pv_in0[d];
            HVX_Vector v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            എങ്കിൽ (real_lanes < 64) {
                സ്ഥിരം HVX_VectorPred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * വലുപ്പം(__fp16)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);
            *pv_out0          = Q6_V_lo_W(vp);
            *pv_out1          = Q6_V_hi_W(vp);
        }
    }
}

സ്ഥിതം വരിയിൽ ശൂന്യം hmx_fa_q_prep_fallback(
    __fp16 * vtcm_q_tiles, uintptr_t q_data,
    size_t q_nb1, size_t q_nb2, size_t q_nb3,
    uint32_t q_start, uint32_t kv_head, uint32_t ib3,
    size_t start, size_t end, size_t n_rows_g,
    size_t G, size_t dk_in, size_t dk_out, bool is_q_fp32,
    സ്ഥിരം ഘടന fastdiv_മൂല്യങ്ങൾ * div_G
) {
    വേണ്ടി (size_t r = start; r < end; r += 2) {
        സ്ഥിരം size_t q_idx0 = fastdiv(r + 0, div_G);
        സ്ഥിരം size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        സ്ഥിരം size_t q_idx1 = fastdiv(r + 1, div_G);
        സ്ഥിരം size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        സ്ഥിരം uint8_t * q_ptr0 = (r + 0 < n_rows_g) ? ((സ്ഥിരം uint8_t *) q_data + (q_start + q_idx0) * q_nb1 +
                                                      (kv_head * G + h_idx0) * q_nb2 + ib3 * q_nb3) :
                                                      ശൂന്യ;
        സ്ഥിരം uint8_t * q_ptr1 = (r + 1 < n_rows_g) ? ((സ്ഥിരം uint8_t *) q_data + (q_start + q_idx1) * q_nb1 +
                                                      (kv_head * G + h_idx1) * q_nb2 + ib3 * q_nb3) :
                                                      ശൂന്യ;

        size_t   r0       = r / HMX_FP16_TILE_N_ROWS;
        size_t   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * dk_out;

        എങ്കിൽ (is_q_fp32) {
            സ്ഥിരം HVX_UVector * pv_in0 = q_ptr0 ? (സ്ഥിരം HVX_UVector *) q_ptr0 : ശൂന്യ;
            സ്ഥിരം HVX_UVector * pv_in1 = q_ptr1 ? (സ്ഥിരം HVX_UVector *) q_ptr1 : ശൂന്യ;

            വേണ്ടി (uint32_t d = 0; d < dk_out / 32; ++d) {
                HVX_Vector * out_tile   = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
                സ്ഥിരം size_t base_lane  = (size_t) d * 32;
                സ്ഥിരം size_t real_lanes = (base_lane < dk_in) ? hex_smin(32, dk_in - base_lane) : 0;

                എങ്കിൽ (real_lanes == 0) {
                    out_tile[r1 / 2] = Q6_V_vzero();
                    തുടരുക;
                }
                HVX_Vector v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_Vector v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                എങ്കിൽ (real_lanes < 32) {
                    സ്ഥിരം HVX_VectorPred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * വലുപ്പം(ദശാംശം)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                out_tile[r1 / 2] = hvx_vec_f32_to_f16_shuff(v0, v1);
            }
        } അല്ലെങ്കിൽ {
            സ്ഥിരം HVX_UVector * pv_in0 = q_ptr0 ? (സ്ഥിരം HVX_UVector *) q_ptr0 : ശൂന്യ;
            സ്ഥിരം HVX_UVector * pv_in1 = q_ptr1 ? (സ്ഥിരം HVX_UVector *) q_ptr1 : ശൂന്യ;

            വേണ്ടി (uint32_t d = 0; d < dk_out / 64; ++d) {
                __fp16 *     out_dtile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
                HVX_Vector * pv_out1   = pv_out0 + 16;

                സ്ഥിരം size_t base_lane  = (size_t) d * 64;
                സ്ഥിരം size_t real_lanes = (base_lane < dk_in) ? hex_smin(64, dk_in - base_lane) : 0;

                എങ്കിൽ (real_lanes == 0) {
                    *pv_out0 = Q6_V_vzero();
                    *pv_out1 = Q6_V_vzero();
                    തുടരുക;
                }
                HVX_Vector v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_Vector v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                എങ്കിൽ (real_lanes < 64) {
                    സ്ഥിരം HVX_VectorPred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * വലുപ്പം(__fp16)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);
                *pv_out0 = Q6_V_lo_W(vp);
                *pv_out1 = Q6_V_hi_W(vp);
            }
        }
    }
}

#endif /* HMX_FA_KERNELS_H */
