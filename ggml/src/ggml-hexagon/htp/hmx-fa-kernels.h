#ifndef HMX_FA_KERNELS_H
#define HMX_FA_KERNELS_H

#अन्तर्भाव <stdint.h>
#अन्तर्भाव <stddef.h>
#अन्तर्भाव <stdbool.h>
#अन्तर्भाव "hvx-utils.h"
#अन्तर्भाव "hmx-utils.h"
#अन्तर्भाव "hex-fastdiv.h"

// HMX-specific parameters, offsets and inner kernels for Flash Attention

// Scatter offsets for diagonal tile: entry[2i] = i*136, entry[2i+1] = i*136+6
// 136 = 4 * 32 + 8 = byte offset to diagonal in a 32x32 fp16 interleaved tile
स्थिर अचर int16_t d_tile_scatter_offsets[64] __attribute__((aligned(128))) = {
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
    अचर __fp16 * row_tiles,
    अचर __fp16 * col_tiles,
    __fp16 *       out_tile,
    माप_प्रकार         n_dot_tiles
) {
    यदि (n_dot_tiles == 2) {
        यंत्र चञ्चल(
            HMX_LOAD_MPY_F16("%1", "%2", "%0")
            HMX_LOAD_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(row_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(row_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(col_tiles + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } अन्यथा यदि (n_dot_tiles == 4) {
        यंत्र चञ्चल(
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
    } अन्यथा यदि (n_dot_tiles == 8) {
        यंत्र चञ्चल(
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
    } अन्यथा {
        लागि (माप_प्रकार k = 0; k < n_dot_tiles; ++k) {
            यंत्र चञ्चल(
                HMX_LOAD_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(row_tiles), "r"(col_tiles)
            );
            row_tiles += HMX_FP16_TILE_N_ELMS;
            col_tiles += HMX_FP16_TILE_N_ELMS;
        }
    }
    यंत्र चञ्चल(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(out_tile), "r"(0)
        : "memory"
    );
}

स्थिर अन्तःस्थ शून्य hmx_fa_o_update_tile(
    अचर __fp16 * d_diag,
    अचर __fp16 * o_rc,
    अचर __fp16 * p_tile_in,
    अचर __fp16 * v_tile_in,
    __fp16 *       o_tile_out,
    माप_प्रकार         n_col_tiles
) {
    यंत्र चञ्चल(
        HMX_LOAD_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    यदि (n_col_tiles == 2) {
        यंत्र चञ्चल(
            HMX_LOAD_MPY_F16("%1", "%2", "%0")
            HMX_LOAD_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(p_tile_in + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_in + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_in + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } अन्यथा यदि (n_col_tiles == 4) {
        यंत्र चञ्चल(
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
    } अन्यथा यदि (n_col_tiles == 8) {
        यंत्र चञ्चल(
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
    } अन्यथा {
        लागि (माप_प्रकार k = 0; k < n_col_tiles; ++k) {
            यंत्र चञ्चल(
                HMX_LOAD_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(p_tile_in), "r"(v_tile_in)
            );
            p_tile_in += HMX_FP16_TILE_N_ELMS;
            v_tile_in += HMX_FP16_TILE_N_ELMS;
        }
    }
    यंत्र चञ्चल(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_tile_out), "r"(0)
        : "memory"
    );
}

स्थिर अन्तःस्थ शून्य hmx_fa_o_norm_tile(
    अचर __fp16 * d_diag,
    अचर __fp16 * o_rc,
    __fp16 *       o_out
) {
    यंत्र चञ्चल(
        HMX_LOAD_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    यंत्र चञ्चल(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_out), "r"(0)
        : "memory"
    );
}

स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp32_d2(
    __fp16 * vtcm_q_tiles, अचर uint8_t * temp_q_vtcm,
    माप_प्रकार start, माप_प्रकार end, माप_प्रकार g_rows_end,
    माप_प्रकार DK, माप_प्रकार G, माप_प्रकार n_rows_q,
    अचर संरचना fastdiv_मानहरू * div_G, bool q_transposed
) {
    लागि (माप_प्रकार r = start; r < end; r += 2) {
        माप_प्रकार   r0       = r / HMX_FP16_TILE_N_ROWS;
        माप_प्रकार   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * DK;

        यदि (r >= g_rows_end) {
            ((HVX_Vector *) (out_base + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            ((HVX_Vector *) (out_base + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            जारी;
        }

        अचर माप_प्रकार q_idx0 = fastdiv(r + 0, div_G);
        अचर माप_प्रकार h_idx0 = fastmodulo(r + 0, G, div_G);
        अचर माप_प्रकार q_idx1 = fastdiv(r + 1, div_G);
        अचर माप_प्रकार h_idx1 = fastmodulo(r + 1, G, div_G);

        अचर माप_प्रकार offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        अचर माप_प्रकार offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        अचर HVX_Vector * pv_in0 = (अचर HVX_Vector *) (temp_q_vtcm + offset0 * DK * परिमाण(भिन्न));
        अचर HVX_Vector * pv_in1 = (r + 1 < g_rows_end)
            ? (अचर HVX_Vector *) (temp_q_vtcm + offset1 * DK * परिमाण(भिन्न))
            : रिक्त;

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

स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp32_d4(
    __fp16 * vtcm_q_tiles, अचर uint8_t * temp_q_vtcm,
    माप_प्रकार start, माप_प्रकार end, माप_प्रकार g_rows_end,
    माप_प्रकार DK, माप_प्रकार G, माप_प्रकार n_rows_q,
    अचर संरचना fastdiv_मानहरू * div_G, bool q_transposed
) {
    लागि (माप_प्रकार r = start; r < end; r += 2) {
        माप_प्रकार   r0       = r / HMX_FP16_TILE_N_ROWS;
        माप_प्रकार   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * DK;

        यदि (r >= g_rows_end) {
            लागि (uint32_t d = 0; d < 4; ++d) {
                ((HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            जारी;
        }

        अचर माप_प्रकार q_idx0 = fastdiv(r + 0, div_G);
        अचर माप_प्रकार h_idx0 = fastmodulo(r + 0, G, div_G);
        अचर माप_प्रकार q_idx1 = fastdiv(r + 1, div_G);
        अचर माप_प्रकार h_idx1 = fastmodulo(r + 1, G, div_G);

        अचर माप_प्रकार offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        अचर माप_प्रकार offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        अचर HVX_Vector * pv_in0 = (अचर HVX_Vector *) (temp_q_vtcm + offset0 * DK * परिमाण(भिन्न));
        अचर HVX_Vector * pv_in1 = (r + 1 < g_rows_end)
            ? (अचर HVX_Vector *) (temp_q_vtcm + offset1 * DK * परिमाण(भिन्न))
            : रिक्त;

        लागि (uint32_t d = 0; d < 4; ++d) {
            HVX_Vector v0 = pv_in0[d];
            HVX_Vector v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_Vector v_hf = hvx_vec_f32_to_f16_shuff(v0, v1);
            ((HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp32(
    __fp16 * vtcm_q_tiles, अचर uint8_t * temp_q_vtcm,
    माप_प्रकार start, माप_प्रकार end, माप_प्रकार g_rows_end,
    माप_प्रकार DK, माप_प्रकार G, माप_प्रकार n_rows_q,
    अचर संरचना fastdiv_मानहरू * div_G, uint32_t d_limit, bool q_transposed
) {
    लागि (माप_प्रकार r = start; r < end; r += 2) {
        माप_प्रकार   r0       = r / HMX_FP16_TILE_N_ROWS;
        माप_प्रकार   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * DK;

        यदि (r >= g_rows_end) {
            लागि (uint32_t d = 0; d < d_limit; ++d) {
                ((HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            जारी;
        }

        अचर माप_प्रकार q_idx0 = fastdiv(r + 0, div_G);
        अचर माप_प्रकार h_idx0 = fastmodulo(r + 0, G, div_G);
        अचर माप_प्रकार q_idx1 = fastdiv(r + 1, div_G);
        अचर माप_प्रकार h_idx1 = fastmodulo(r + 1, G, div_G);

        अचर माप_प्रकार offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        अचर माप_प्रकार offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        अचर HVX_Vector * pv_in0 = (अचर HVX_Vector *) (temp_q_vtcm + offset0 * DK * परिमाण(भिन्न));
        अचर HVX_Vector * pv_in1 = (r + 1 < g_rows_end)
            ? (अचर HVX_Vector *) (temp_q_vtcm + offset1 * DK * परिमाण(भिन्न))
            : रिक्त;

        लागि (uint32_t d = 0; d < d_limit; ++d) {
            HVX_Vector v0   = pv_in0[d];
            HVX_Vector v1   = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_Vector v_hf = hvx_vec_f32_to_f16_shuff(v0, v1);

            HVX_Vector * out_tile = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
            out_tile[r1 / 2]      = v_hf;
        }
    }
}

स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp16_d1(
    __fp16 * vtcm_q_tiles, अचर uint8_t * temp_q_vtcm,
    माप_प्रकार start, माप_प्रकार end, माप_प्रकार g_rows_end,
    माप_प्रकार DK, माप_प्रकार G, माप_प्रकार n_rows_q,
    अचर संरचना fastdiv_मानहरू * div_G, bool q_transposed
) {
    लागि (माप_प्रकार r = start; r < end; r += 2) {
        माप_प्रकार   r0       = r / HMX_FP16_TILE_N_ROWS;
        माप_प्रकार   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * DK;

        यदि (r >= g_rows_end) {
            __fp16 *     out_dtile = out_base + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
            HVX_Vector * pv_out1   = pv_out0 + 16;
            *pv_out0 = Q6_V_vzero();
            *pv_out1 = Q6_V_vzero();
            जारी;
        }

        अचर माप_प्रकार q_idx0 = fastdiv(r + 0, div_G);
        अचर माप_प्रकार h_idx0 = fastmodulo(r + 0, G, div_G);
        अचर माप_प्रकार q_idx1 = fastdiv(r + 1, div_G);
        अचर माप_प्रकार h_idx1 = fastmodulo(r + 1, G, div_G);

        अचर माप_प्रकार offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        अचर माप_प्रकार offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        अचर HVX_Vector * pv_in0 = (अचर HVX_Vector *) (temp_q_vtcm + offset0 * DK * परिमाण(__fp16));
        अचर HVX_Vector * pv_in1 = (r + 1 < g_rows_end)
            ? (अचर HVX_Vector *) (temp_q_vtcm + offset1 * DK * परिमाण(__fp16))
            : रिक्त;

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

स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp16_d2(
    __fp16 * vtcm_q_tiles, अचर uint8_t * temp_q_vtcm,
    माप_प्रकार start, माप_प्रकार end, माप_प्रकार g_rows_end,
    माप_प्रकार DK, माप_प्रकार G, माप_प्रकार n_rows_q,
    अचर संरचना fastdiv_मानहरू * div_G, bool q_transposed
) {
    लागि (माप_प्रकार r = start; r < end; r += 2) {
        माप_प्रकार   r0       = r / HMX_FP16_TILE_N_ROWS;
        माप_प्रकार   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * DK;

        यदि (r >= g_rows_end) {
            लागि (uint32_t d = 0; d < 2; ++d) {
                __fp16 *     out_dtile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
                HVX_Vector * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            जारी;
        }

        अचर माप_प्रकार q_idx0 = fastdiv(r + 0, div_G);
        अचर माप_प्रकार h_idx0 = fastmodulo(r + 0, G, div_G);
        अचर माप_प्रकार q_idx1 = fastdiv(r + 1, div_G);
        अचर माप_प्रकार h_idx1 = fastmodulo(r + 1, G, div_G);

        अचर माप_प्रकार offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        अचर माप_प्रकार offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        अचर HVX_Vector * pv_in0 = (अचर HVX_Vector *) (temp_q_vtcm + offset0 * DK * परिमाण(__fp16));
        अचर HVX_Vector * pv_in1 = (r + 1 < g_rows_end)
            ? (अचर HVX_Vector *) (temp_q_vtcm + offset1 * DK * परिमाण(__fp16))
            : रिक्त;

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

स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp16(
    __fp16 * vtcm_q_tiles, अचर uint8_t * temp_q_vtcm,
    माप_प्रकार start, माप_प्रकार end, माप_प्रकार g_rows_end,
    माप_प्रकार DK, माप_प्रकार G, माप_प्रकार n_rows_q,
    अचर संरचना fastdiv_मानहरू * div_G, uint32_t d_limit, bool q_transposed
) {
    लागि (माप_प्रकार r = start; r < end; r += 2) {
        माप_प्रकार   r0       = r / HMX_FP16_TILE_N_ROWS;
        माप_प्रकार   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * DK;

        यदि (r >= g_rows_end) {
            लागि (uint32_t d = 0; d < d_limit; ++d) {
                __fp16 *     out_dtile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
                HVX_Vector * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            जारी;
        }

        अचर माप_प्रकार q_idx0 = fastdiv(r + 0, div_G);
        अचर माप_प्रकार h_idx0 = fastmodulo(r + 0, G, div_G);
        अचर माप_प्रकार q_idx1 = fastdiv(r + 1, div_G);
        अचर माप_प्रकार h_idx1 = fastmodulo(r + 1, G, div_G);

        अचर माप_प्रकार offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        अचर माप_प्रकार offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        अचर HVX_Vector * pv_in0 = (अचर HVX_Vector *) (temp_q_vtcm + offset0 * DK * परिमाण(__fp16));
        अचर HVX_Vector * pv_in1 = (r + 1 < g_rows_end)
            ? (अचर HVX_Vector *) (temp_q_vtcm + offset1 * DK * परिमाण(__fp16))
            : रिक्त;

        लागि (uint32_t d = 0; d < d_limit; ++d) {
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
स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp32_pad(__fp16 *                      vtcm_q_tiles,
                                          अचर uint8_t *               temp_q_vtcm,
                                          माप_प्रकार                        start,
                                          माप_प्रकार                        end,
                                          माप_प्रकार                        g_rows_end,
                                          माप_प्रकार                        dk_in,
                                          माप_प्रकार                        dk_out,
                                          माप_प्रकार                        G,
                                          माप_प्रकार                        n_rows_q,
                                          अचर संरचना fastdiv_मानहरू * div_G,
                                          bool                          q_transposed) {
    अचर uint32_t n_out_tiles = (uint32_t) (dk_out / 32);
    लागि (माप_प्रकार r = start; r < end; r += 2) {
        माप_प्रकार   r0       = r / HMX_FP16_TILE_N_ROWS;
        माप_प्रकार   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * dk_out;

        यदि (r >= g_rows_end) {
            लागि (uint32_t d = 0; d < n_out_tiles; ++d) {
                ((HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            जारी;
        }

        अचर माप_प्रकार q_idx0 = fastdiv(r + 0, div_G);
        अचर माप_प्रकार h_idx0 = fastmodulo(r + 0, G, div_G);
        अचर माप_प्रकार q_idx1 = fastdiv(r + 1, div_G);
        अचर माप_प्रकार h_idx1 = fastmodulo(r + 1, G, div_G);

        अचर माप_प्रकार offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        अचर माप_प्रकार offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        अचर HVX_UVector * pv_in0 = (अचर HVX_UVector *) (temp_q_vtcm + offset0 * dk_in * परिमाण(भिन्न));
        अचर HVX_UVector * pv_in1 = (r + 1 < g_rows_end) ? (अचर HVX_UVector *) (temp_q_vtcm + offset1 * dk_in * परिमाण(भिन्न)) : रिक्त;

        लागि (uint32_t d = 0; d < n_out_tiles; ++d) {
            HVX_Vector * out_tile   = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
            अचर माप_प्रकार base_lane  = (माप_प्रकार) d * 32;
            अचर माप_प्रकार real_lanes = (base_lane < dk_in) ? hex_smin(32, dk_in - base_lane) : 0;

            यदि (real_lanes == 0) {
                out_tile[r1 / 2] = Q6_V_vzero();
                जारी;
            }

            HVX_Vector v0 = pv_in0[d];
            HVX_Vector v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            यदि (real_lanes < 32) {
                // Straddle tile: keep the first real_lanes floats, zero the padded tail so
                // the packed f16 lanes beyond DK are zero.
                अचर HVX_VectorPred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * परिमाण(भिन्न)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            out_tile[r1 / 2] = hvx_vec_f32_to_f16_shuff(v0, v1);
        }
    }
}

// Head-dim-padded Q-prep (f16). Used when DK is not a multiple of 64.
स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fp16_pad(__fp16 *                      vtcm_q_tiles,
                                          अचर uint8_t *               temp_q_vtcm,
                                          माप_प्रकार                        start,
                                          माप_प्रकार                        end,
                                          माप_प्रकार                        g_rows_end,
                                          माप_प्रकार                        dk_in,
                                          माप_प्रकार                        dk_out,
                                          माप_प्रकार                        G,
                                          माप_प्रकार                        n_rows_q,
                                          अचर संरचना fastdiv_मानहरू * div_G,
                                          bool                          q_transposed) {
    अचर uint32_t n_out_pairs = (uint32_t) (dk_out / 64);
    लागि (माप_प्रकार r = start; r < end; r += 2) {
        माप_प्रकार   r0       = r / HMX_FP16_TILE_N_ROWS;
        माप_प्रकार   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * dk_out;

        यदि (r >= g_rows_end) {
            लागि (uint32_t d = 0; d < n_out_pairs; ++d) {
                __fp16 *     out_dtile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
                HVX_Vector * pv_out1   = pv_out0 + 16;
                *pv_out0               = Q6_V_vzero();
                *pv_out1               = Q6_V_vzero();
            }
            जारी;
        }

        अचर माप_प्रकार q_idx0 = fastdiv(r + 0, div_G);
        अचर माप_प्रकार h_idx0 = fastmodulo(r + 0, G, div_G);
        अचर माप_प्रकार q_idx1 = fastdiv(r + 1, div_G);
        अचर माप_प्रकार h_idx1 = fastmodulo(r + 1, G, div_G);

        अचर माप_प्रकार offset0 = q_transposed ? (h_idx0 * n_rows_q + q_idx0) : (q_idx0 * G + h_idx0);
        अचर माप_प्रकार offset1 = q_transposed ? (h_idx1 * n_rows_q + q_idx1) : (q_idx1 * G + h_idx1);

        अचर HVX_UVector * pv_in0 = (अचर HVX_UVector *) (temp_q_vtcm + offset0 * dk_in * परिमाण(__fp16));
        अचर HVX_UVector * pv_in1 = (r + 1 < g_rows_end) ? (अचर HVX_UVector *) (temp_q_vtcm + offset1 * dk_in * परिमाण(__fp16)) : रिक्त;

        लागि (uint32_t d = 0; d < n_out_pairs; ++d) {
            __fp16 *     out_dtile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
            HVX_Vector * pv_out1   = pv_out0 + 16;

            अचर माप_प्रकार base_lane  = (माप_प्रकार) d * 64;
            अचर माप_प्रकार real_lanes = (base_lane < dk_in) ? hex_smin(64, dk_in - base_lane) : 0;

            यदि (real_lanes == 0) {
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
                जारी;
            }

            HVX_Vector v0 = pv_in0[d];
            HVX_Vector v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            यदि (real_lanes < 64) {
                अचर HVX_VectorPred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * परिमाण(__fp16)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            HVX_VectorPair vp = Q6_W_vshuff_VVR(v1, v0, -2);
            *pv_out0          = Q6_V_lo_W(vp);
            *pv_out1          = Q6_V_hi_W(vp);
        }
    }
}

स्थिर अन्तःस्थ शून्य hmx_fa_q_prep_fallback(
    __fp16 * vtcm_q_tiles, uintptr_t q_data,
    माप_प्रकार q_nb1, माप_प्रकार q_nb2, माप_प्रकार q_nb3,
    uint32_t q_start, uint32_t kv_head, uint32_t ib3,
    माप_प्रकार start, माप_प्रकार end, माप_प्रकार n_rows_g,
    माप_प्रकार G, माप_प्रकार dk_in, माप_प्रकार dk_out, bool is_q_fp32,
    अचर संरचना fastdiv_मानहरू * div_G
) {
    लागि (माप_प्रकार r = start; r < end; r += 2) {
        अचर माप_प्रकार q_idx0 = fastdiv(r + 0, div_G);
        अचर माप_प्रकार h_idx0 = fastmodulo(r + 0, G, div_G);
        अचर माप_प्रकार q_idx1 = fastdiv(r + 1, div_G);
        अचर माप_प्रकार h_idx1 = fastmodulo(r + 1, G, div_G);

        अचर uint8_t * q_ptr0 = (r + 0 < n_rows_g) ? ((अचर uint8_t *) q_data + (q_start + q_idx0) * q_nb1 +
                                                      (kv_head * G + h_idx0) * q_nb2 + ib3 * q_nb3) :
                                                      रिक्त;
        अचर uint8_t * q_ptr1 = (r + 1 < n_rows_g) ? ((अचर uint8_t *) q_data + (q_start + q_idx1) * q_nb1 +
                                                      (kv_head * G + h_idx1) * q_nb2 + ib3 * q_nb3) :
                                                      रिक्त;

        माप_प्रकार   r0       = r / HMX_FP16_TILE_N_ROWS;
        माप_प्रकार   r1       = r % HMX_FP16_TILE_N_ROWS;
        __fp16 * out_base = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ROWS * dk_out;

        यदि (is_q_fp32) {
            अचर HVX_UVector * pv_in0 = q_ptr0 ? (अचर HVX_UVector *) q_ptr0 : रिक्त;
            अचर HVX_UVector * pv_in1 = q_ptr1 ? (अचर HVX_UVector *) q_ptr1 : रिक्त;

            लागि (uint32_t d = 0; d < dk_out / 32; ++d) {
                HVX_Vector * out_tile   = (HVX_Vector *) (out_base + d * HMX_FP16_TILE_N_ELMS);
                अचर माप_प्रकार base_lane  = (माप_प्रकार) d * 32;
                अचर माप_प्रकार real_lanes = (base_lane < dk_in) ? hex_smin(32, dk_in - base_lane) : 0;

                यदि (real_lanes == 0) {
                    out_tile[r1 / 2] = Q6_V_vzero();
                    जारी;
                }
                HVX_Vector v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_Vector v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                यदि (real_lanes < 32) {
                    अचर HVX_VectorPred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * परिमाण(भिन्न)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                out_tile[r1 / 2] = hvx_vec_f32_to_f16_shuff(v0, v1);
            }
        } अन्यथा {
            अचर HVX_UVector * pv_in0 = q_ptr0 ? (अचर HVX_UVector *) q_ptr0 : रिक्त;
            अचर HVX_UVector * pv_in1 = q_ptr1 ? (अचर HVX_UVector *) q_ptr1 : रिक्त;

            लागि (uint32_t d = 0; d < dk_out / 64; ++d) {
                __fp16 *     out_dtile = out_base + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_Vector * pv_out0   = ((HVX_Vector *) out_dtile) + r1 / 2;
                HVX_Vector * pv_out1   = pv_out0 + 16;

                अचर माप_प्रकार base_lane  = (माप_प्रकार) d * 64;
                अचर माप_प्रकार real_lanes = (base_lane < dk_in) ? hex_smin(64, dk_in - base_lane) : 0;

                यदि (real_lanes == 0) {
                    *pv_out0 = Q6_V_vzero();
                    *pv_out1 = Q6_V_vzero();
                    जारी;
                }
                HVX_Vector v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_Vector v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                यदि (real_lanes < 64) {
                    अचर HVX_VectorPred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * परिमाण(__fp16)));
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
