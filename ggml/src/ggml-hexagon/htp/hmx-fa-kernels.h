#که_نه_تعریف_شوی HMX_FA_KERNELS_H
#تعریف_کړه HMX_FA_KERNELS_H

#ضمن <stdint.h>
#ضمن <stddef.h>
#ضمن <stdbool.h>
#ضمن "hvx-utils.h"
#ضمن "hmx-utils.h"
#ضمن "hex-fastdiv.h"

// HMX-specific parameters, offsets and inner kernels for Flash Attention

// Scatter offsets for diagonal tile: entry[2i] = i*136, entry[2i+1] = i*136+6
// 136 = 4 * 32 + 8 = byte offset to diagonal in a 32x32 fp16 interleaved tile
ساكن ثابت int16_t d_tile_scatter_offsets[64] __attribute__((aligned(128))) = {
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

ساكن مضمن فراغ hmx_fa_qk_dot_tile(
    ثابت __fp16 * لیکه_tiles,
    ثابت __fp16 * ستون_tiles,
    __fp16 *       بهر_tile,
    size_t         n_dot_tiles
) {
    إذا (n_dot_tiles == 2) {
        asm متطاير(
            HMX_بارول_MPY_F16("%1", "%2", "%0")
            HMX_بارول_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(لیکه_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(لیکه_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } وإلا إذا (n_dot_tiles == 4) {
        asm متطاير(
            HMX_بارول_MPY_F16("%1", "%2", "%0")
            HMX_بارول_MPY_F16("%3", "%4", "%0")
            HMX_بارول_MPY_F16("%5", "%6", "%0")
            HMX_بارول_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(لیکه_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(لیکه_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(لیکه_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(لیکه_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } وإلا إذا (n_dot_tiles == 8) {
        asm متطاير(
            HMX_بارول_MPY_F16("%1", "%2", "%0")
            HMX_بارول_MPY_F16("%3", "%4", "%0")
            HMX_بارول_MPY_F16("%5", "%6", "%0")
            HMX_بارول_MPY_F16("%7", "%8", "%0")
            HMX_بارول_MPY_F16("%9", "%10", "%0")
            HMX_بارول_MPY_F16("%11", "%12", "%0")
            HMX_بارول_MPY_F16("%13", "%14", "%0")
            HMX_بارول_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(لیکه_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(لیکه_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(لیکه_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(لیکه_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(لیکه_tiles + 4 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(لیکه_tiles + 5 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(لیکه_tiles + 6 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(لیکه_tiles + 7 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } وإلا {
        لكل (size_t k = 0; k < n_dot_tiles; ++k) {
            asm متطاير(
                HMX_بارول_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(لیکه_tiles), "r"(ستون_tiles)
            );
            لیکه_tiles += HMX_FP16_TILE_N_ELMS;
            ستون_tiles += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm متطاير(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(بهر_tile), "r"(0)
        : "memory"
    );
}

ساكن مضمن فراغ hmx_fa_o_update_tile(
    ثابت __fp16 * d_diag,
    ثابت __fp16 * o_rc,
    ثابت __fp16 * p_tile_دننه,
    ثابت __fp16 * v_tile_دننه,
    __fp16 *       o_tile_بهر,
    size_t         n_ستون_tiles
) {
    asm متطاير(
        HMX_بارول_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    إذا (n_ستون_tiles == 2) {
        asm متطاير(
            HMX_بارول_MPY_F16("%1", "%2", "%0")
            HMX_بارول_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(p_tile_دننه + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_دننه + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_دننه + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_دننه + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } وإلا إذا (n_ستون_tiles == 4) {
        asm متطاير(
            HMX_بارول_MPY_F16("%1", "%2", "%0")
            HMX_بارول_MPY_F16("%3", "%4", "%0")
            HMX_بارول_MPY_F16("%5", "%6", "%0")
            HMX_بارول_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(p_tile_دننه + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_دننه + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_دننه + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_دننه + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_دننه + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_دننه + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_دننه + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_دننه + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } وإلا إذا (n_ستون_tiles == 8) {
        asm متطاير(
            HMX_بارول_MPY_F16("%1", "%2", "%0")
            HMX_بارول_MPY_F16("%3", "%4", "%0")
            HMX_بارول_MPY_F16("%5", "%6", "%0")
            HMX_بارول_MPY_F16("%7", "%8", "%0")
            HMX_بارول_MPY_F16("%9", "%10", "%0")
            HMX_بارول_MPY_F16("%11", "%12", "%0")
            HMX_بارول_MPY_F16("%13", "%14", "%0")
            HMX_بارول_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(p_tile_دننه + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_دننه + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_دننه + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_دننه + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_دننه + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_دننه + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_دننه + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_دننه + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_دننه + 4 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_دننه + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_دننه + 5 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_دننه + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_دننه + 6 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_دننه + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_دننه + 7 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_دننه + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } وإلا {
        لكل (size_t k = 0; k < n_ستون_tiles; ++k) {
            asm متطاير(
                HMX_بارول_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(p_tile_دننه), "r"(v_tile_دننه)
            );
            p_tile_دننه += HMX_FP16_TILE_N_ELMS;
            v_tile_دننه += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm متطاير(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_tile_بهر), "r"(0)
        : "memory"
    );
}

ساكن مضمن فراغ hmx_fa_o_norm_tile(
    ثابت __fp16 * d_diag,
    ثابت __fp16 * o_rc,
    __fp16 *       o_بهر
) {
    asm متطاير(
        HMX_بارول_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    asm متطاير(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_بهر), "r"(0)
        : "memory"
    );
}

ساكن مضمن فراغ hmx_fa_q_prep_fp32_d2(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * لنډمهاله_q_vtcm,
    size_t پیل, size_t پای_ټکی, size_t g_لیکې_پای_ټکی,
    size_t DK, size_t G, size_t n_لیکې_q,
    ثابت بنية fastdiv_ارزښتونه * div_G, bool q_transposed
) {
    لكل (size_t r = پیل; r < پای_ټکی; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_لیکې;
        size_t   r1       = r % HMX_FP16_TILE_N_لیکې;
        __fp16 * بهر_بنسټ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_لیکې * DK;

        إذا (r >= g_لیکې_پای_ټکی) {
            ((HVX_ویکټور *) (بهر_بنسټ + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            ((HVX_ویکټور *) (بهر_بنسټ + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_لیکې_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_لیکې_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_ویکټور * pv_in0 = (ثابت HVX_ویکټور *) (لنډمهاله_q_vtcm + offset0 * DK * حجم(عائم));
        ثابت HVX_ویکټور * pv_in1 = (r + 1 < g_لیکې_پای_ټکی)
            ? (ثابت HVX_ویکټور *) (لنډمهاله_q_vtcm + offset1 * DK * حجم(عائم))
            : عدم;

        {
            HVX_ویکټور v0 = pv_in0[0];
            HVX_ویکټور v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_ویکټور v_hf = hvx_vec_f32_منزل_f16_shuff(v0, v1);
            ((HVX_ویکټور *) (بهر_بنسټ + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
        {
            HVX_ویکټور v0 = pv_in0[1];
            HVX_ویکټور v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_ویکټور v_hf = hvx_vec_f32_منزل_f16_shuff(v0, v1);
            ((HVX_ویکټور *) (بهر_بنسټ + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

ساكن مضمن فراغ hmx_fa_q_prep_fp32_d4(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * لنډمهاله_q_vtcm,
    size_t پیل, size_t پای_ټکی, size_t g_لیکې_پای_ټکی,
    size_t DK, size_t G, size_t n_لیکې_q,
    ثابت بنية fastdiv_ارزښتونه * div_G, bool q_transposed
) {
    لكل (size_t r = پیل; r < پای_ټکی; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_لیکې;
        size_t   r1       = r % HMX_FP16_TILE_N_لیکې;
        __fp16 * بهر_بنسټ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_لیکې * DK;

        إذا (r >= g_لیکې_پای_ټکی) {
            لكل (uint32_t d = 0; d < 4; ++d) {
                ((HVX_ویکټور *) (بهر_بنسټ + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_لیکې_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_لیکې_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_ویکټور * pv_in0 = (ثابت HVX_ویکټور *) (لنډمهاله_q_vtcm + offset0 * DK * حجم(عائم));
        ثابت HVX_ویکټور * pv_in1 = (r + 1 < g_لیکې_پای_ټکی)
            ? (ثابت HVX_ویکټور *) (لنډمهاله_q_vtcm + offset1 * DK * حجم(عائم))
            : عدم;

        لكل (uint32_t d = 0; d < 4; ++d) {
            HVX_ویکټور v0 = pv_in0[d];
            HVX_ویکټور v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_ویکټور v_hf = hvx_vec_f32_منزل_f16_shuff(v0, v1);
            ((HVX_ویکټور *) (بهر_بنسټ + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

ساكن مضمن فراغ hmx_fa_q_prep_fp32(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * لنډمهاله_q_vtcm,
    size_t پیل, size_t پای_ټکی, size_t g_لیکې_پای_ټکی,
    size_t DK, size_t G, size_t n_لیکې_q,
    ثابت بنية fastdiv_ارزښتونه * div_G, uint32_t d_حد, bool q_transposed
) {
    لكل (size_t r = پیل; r < پای_ټکی; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_لیکې;
        size_t   r1       = r % HMX_FP16_TILE_N_لیکې;
        __fp16 * بهر_بنسټ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_لیکې * DK;

        إذا (r >= g_لیکې_پای_ټکی) {
            لكل (uint32_t d = 0; d < d_حد; ++d) {
                ((HVX_ویکټور *) (بهر_بنسټ + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_لیکې_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_لیکې_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_ویکټور * pv_in0 = (ثابت HVX_ویکټور *) (لنډمهاله_q_vtcm + offset0 * DK * حجم(عائم));
        ثابت HVX_ویکټور * pv_in1 = (r + 1 < g_لیکې_پای_ټکی)
            ? (ثابت HVX_ویکټور *) (لنډمهاله_q_vtcm + offset1 * DK * حجم(عائم))
            : عدم;

        لكل (uint32_t d = 0; d < d_حد; ++d) {
            HVX_ویکټور v0   = pv_in0[d];
            HVX_ویکټور v1   = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_ویکټور v_hf = hvx_vec_f32_منزل_f16_shuff(v0, v1);

            HVX_ویکټور * بهر_tile = (HVX_ویکټور *) (بهر_بنسټ + d * HMX_FP16_TILE_N_ELMS);
            بهر_tile[r1 / 2]      = v_hf;
        }
    }
}

ساكن مضمن فراغ hmx_fa_q_prep_fp16_d1(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * لنډمهاله_q_vtcm,
    size_t پیل, size_t پای_ټکی, size_t g_لیکې_پای_ټکی,
    size_t DK, size_t G, size_t n_لیکې_q,
    ثابت بنية fastdiv_ارزښتونه * div_G, bool q_transposed
) {
    لكل (size_t r = پیل; r < پای_ټکی; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_لیکې;
        size_t   r1       = r % HMX_FP16_TILE_N_لیکې;
        __fp16 * بهر_بنسټ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_لیکې * DK;

        إذا (r >= g_لیکې_پای_ټکی) {
            __fp16 *     بهر_dtile = بهر_بنسټ + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ویکټور * pv_out0   = ((HVX_ویکټور *) بهر_dtile) + r1 / 2;
            HVX_ویکټور * pv_out1   = pv_out0 + 16;
            *pv_out0 = Q6_V_vzero();
            *pv_out1 = Q6_V_vzero();
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_لیکې_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_لیکې_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_ویکټور * pv_in0 = (ثابت HVX_ویکټور *) (لنډمهاله_q_vtcm + offset0 * DK * حجم(__fp16));
        ثابت HVX_ویکټور * pv_in1 = (r + 1 < g_لیکې_پای_ټکی)
            ? (ثابت HVX_ویکټور *) (لنډمهاله_q_vtcm + offset1 * DK * حجم(__fp16))
            : عدم;

        HVX_ویکټور     v0 = pv_in0[0];
        HVX_ویکټور     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
        HVX_ویکټور_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        __fp16 *     بهر_dtile = بهر_بنسټ + 0 * HMX_FP16_TILE_N_ELMS * 2;
        HVX_ویکټور * pv_out0   = ((HVX_ویکټور *) بهر_dtile) + r1 / 2;
        HVX_ویکټور * pv_out1   = pv_out0 + 16;

        *pv_out0 = Q6_V_lo_W(vp);
        *pv_out1 = Q6_V_hi_W(vp);
    }
}

ساكن مضمن فراغ hmx_fa_q_prep_fp16_d2(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * لنډمهاله_q_vtcm,
    size_t پیل, size_t پای_ټکی, size_t g_لیکې_پای_ټکی,
    size_t DK, size_t G, size_t n_لیکې_q,
    ثابت بنية fastdiv_ارزښتونه * div_G, bool q_transposed
) {
    لكل (size_t r = پیل; r < پای_ټکی; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_لیکې;
        size_t   r1       = r % HMX_FP16_TILE_N_لیکې;
        __fp16 * بهر_بنسټ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_لیکې * DK;

        إذا (r >= g_لیکې_پای_ټکی) {
            لكل (uint32_t d = 0; d < 2; ++d) {
                __fp16 *     بهر_dtile = بهر_بنسټ + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ویکټور * pv_out0   = ((HVX_ویکټور *) بهر_dtile) + r1 / 2;
                HVX_ویکټور * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_لیکې_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_لیکې_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_ویکټور * pv_in0 = (ثابت HVX_ویکټور *) (لنډمهاله_q_vtcm + offset0 * DK * حجم(__fp16));
        ثابت HVX_ویکټور * pv_in1 = (r + 1 < g_لیکې_پای_ټکی)
            ? (ثابت HVX_ویکټور *) (لنډمهاله_q_vtcm + offset1 * DK * حجم(__fp16))
            : عدم;

        {
            HVX_ویکټور     v0 = pv_in0[0];
            HVX_ویکټور     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_ویکټور_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     بهر_dtile = بهر_بنسټ + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ویکټور * pv_out0   = ((HVX_ویکټور *) بهر_dtile) + r1 / 2;
            HVX_ویکټور * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
        {
            HVX_ویکټور     v0 = pv_in0[1];
            HVX_ویکټور     v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_ویکټور_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     بهر_dtile = بهر_بنسټ + 1 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ویکټور * pv_out0   = ((HVX_ویکټور *) بهر_dtile) + r1 / 2;
            HVX_ویکټور * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}

ساكن مضمن فراغ hmx_fa_q_prep_fp16(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * لنډمهاله_q_vtcm,
    size_t پیل, size_t پای_ټکی, size_t g_لیکې_پای_ټکی,
    size_t DK, size_t G, size_t n_لیکې_q,
    ثابت بنية fastdiv_ارزښتونه * div_G, uint32_t d_حد, bool q_transposed
) {
    لكل (size_t r = پیل; r < پای_ټکی; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_لیکې;
        size_t   r1       = r % HMX_FP16_TILE_N_لیکې;
        __fp16 * بهر_بنسټ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_لیکې * DK;

        إذا (r >= g_لیکې_پای_ټکی) {
            لكل (uint32_t d = 0; d < d_حد; ++d) {
                __fp16 *     بهر_dtile = بهر_بنسټ + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ویکټور * pv_out0   = ((HVX_ویکټور *) بهر_dtile) + r1 / 2;
                HVX_ویکټور * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_لیکې_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_لیکې_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_ویکټور * pv_in0 = (ثابت HVX_ویکټور *) (لنډمهاله_q_vtcm + offset0 * DK * حجم(__fp16));
        ثابت HVX_ویکټور * pv_in1 = (r + 1 < g_لیکې_پای_ټکی)
            ? (ثابت HVX_ویکټور *) (لنډمهاله_q_vtcm + offset1 * DK * حجم(__fp16))
            : عدم;

        لكل (uint32_t d = 0; d < d_حد; ++d) {
            HVX_ویکټور     v0 = pv_in0[d];
            HVX_ویکټور     v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_ویکټور_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     بهر_dtile = بهر_بنسټ + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ویکټور * pv_out0   = ((HVX_ویکټور *) بهر_dtile) + r1 / 2;
            HVX_ویکټور * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}


// Head-dim-padded Q-prep (f32). Used when DK is not a multiple of 64.
ساكن مضمن فراغ hmx_fa_q_prep_fp32_pad(__fp16 *                      vtcm_q_tiles,
                                          ثابت uint8_t *               لنډمهاله_q_vtcm,
                                          size_t                        پیل,
                                          size_t                        پای_ټکی,
                                          size_t                        g_لیکې_پای_ټکی,
                                          size_t                        dk_دننه,
                                          size_t                        dk_بهر,
                                          size_t                        G,
                                          size_t                        n_لیکې_q,
                                          ثابت بنية fastdiv_ارزښتونه * div_G,
                                          bool                          q_transposed) {
    ثابت uint32_t n_بهر_tiles = (uint32_t) (dk_بهر / 32);
    لكل (size_t r = پیل; r < پای_ټکی; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_لیکې;
        size_t   r1       = r % HMX_FP16_TILE_N_لیکې;
        __fp16 * بهر_بنسټ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_لیکې * dk_بهر;

        إذا (r >= g_لیکې_پای_ټکی) {
            لكل (uint32_t d = 0; d < n_بهر_tiles; ++d) {
                ((HVX_ویکټور *) (بهر_بنسټ + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_لیکې_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_لیکې_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_UVector * pv_in0 = (ثابت HVX_UVector *) (لنډمهاله_q_vtcm + offset0 * dk_دننه * حجم(عائم));
        ثابت HVX_UVector * pv_in1 = (r + 1 < g_لیکې_پای_ټکی) ? (ثابت HVX_UVector *) (لنډمهاله_q_vtcm + offset1 * dk_دننه * حجم(عائم)) : عدم;

        لكل (uint32_t d = 0; d < n_بهر_tiles; ++d) {
            HVX_ویکټور * بهر_tile   = (HVX_ویکټور *) (بهر_بنسټ + d * HMX_FP16_TILE_N_ELMS);
            ثابت size_t بنسټ_lane  = (size_t) d * 32;
            ثابت size_t real_lanes = (بنسټ_lane < dk_دننه) ? hex_smin(32, dk_دننه - بنسټ_lane) : 0;

            إذا (real_lanes == 0) {
                بهر_tile[r1 / 2] = Q6_V_vzero();
                استمر;
            }

            HVX_ویکټور v0 = pv_in0[d];
            HVX_ویکټور v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            إذا (real_lanes < 32) {
                // Straddle tile: keep the first real_lanes floats, zero the padded tail so
                // the packed f16 lanes beyond DK are zero.
                ثابت HVX_ویکټور_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * حجم(عائم)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            بهر_tile[r1 / 2] = hvx_vec_f32_منزل_f16_shuff(v0, v1);
        }
    }
}

// Head-dim-padded Q-prep (f16). Used when DK is not a multiple of 64.
ساكن مضمن فراغ hmx_fa_q_prep_fp16_pad(__fp16 *                      vtcm_q_tiles,
                                          ثابت uint8_t *               لنډمهاله_q_vtcm,
                                          size_t                        پیل,
                                          size_t                        پای_ټکی,
                                          size_t                        g_لیکې_پای_ټکی,
                                          size_t                        dk_دننه,
                                          size_t                        dk_بهر,
                                          size_t                        G,
                                          size_t                        n_لیکې_q,
                                          ثابت بنية fastdiv_ارزښتونه * div_G,
                                          bool                          q_transposed) {
    ثابت uint32_t n_بهر_جوړې = (uint32_t) (dk_بهر / 64);
    لكل (size_t r = پیل; r < پای_ټکی; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_لیکې;
        size_t   r1       = r % HMX_FP16_TILE_N_لیکې;
        __fp16 * بهر_بنسټ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_لیکې * dk_بهر;

        إذا (r >= g_لیکې_پای_ټکی) {
            لكل (uint32_t d = 0; d < n_بهر_جوړې; ++d) {
                __fp16 *     بهر_dtile = بهر_بنسټ + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ویکټور * pv_out0   = ((HVX_ویکټور *) بهر_dtile) + r1 / 2;
                HVX_ویکټور * pv_out1   = pv_out0 + 16;
                *pv_out0               = Q6_V_vzero();
                *pv_out1               = Q6_V_vzero();
            }
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_لیکې_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_لیکې_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_UVector * pv_in0 = (ثابت HVX_UVector *) (لنډمهاله_q_vtcm + offset0 * dk_دننه * حجم(__fp16));
        ثابت HVX_UVector * pv_in1 = (r + 1 < g_لیکې_پای_ټکی) ? (ثابت HVX_UVector *) (لنډمهاله_q_vtcm + offset1 * dk_دننه * حجم(__fp16)) : عدم;

        لكل (uint32_t d = 0; d < n_بهر_جوړې; ++d) {
            __fp16 *     بهر_dtile = بهر_بنسټ + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ویکټور * pv_out0   = ((HVX_ویکټور *) بهر_dtile) + r1 / 2;
            HVX_ویکټور * pv_out1   = pv_out0 + 16;

            ثابت size_t بنسټ_lane  = (size_t) d * 64;
            ثابت size_t real_lanes = (بنسټ_lane < dk_دننه) ? hex_smin(64, dk_دننه - بنسټ_lane) : 0;

            إذا (real_lanes == 0) {
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
                استمر;
            }

            HVX_ویکټور v0 = pv_in0[d];
            HVX_ویکټور v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            إذا (real_lanes < 64) {
                ثابت HVX_ویکټور_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * حجم(__fp16)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            HVX_ویکټور_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);
            *pv_out0          = Q6_V_lo_W(vp);
            *pv_out1          = Q6_V_hi_W(vp);
        }
    }
}

ساكن مضمن فراغ hmx_fa_q_prep_fallback(
    __fp16 * vtcm_q_tiles, uintptr_t q_معلومات,
    size_t q_nb1, size_t q_nb2, size_t q_nb3,
    uint32_t q_پیل, uint32_t kv_سر, uint32_t ib3,
    size_t پیل, size_t پای_ټکی, size_t n_لیکې_g,
    size_t G, size_t dk_دننه, size_t dk_بهر, bool is_q_fp32,
    ثابت بنية fastdiv_ارزښتونه * div_G
) {
    لكل (size_t r = پیل; r < پای_ټکی; r += 2) {
        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت uint8_t * q_ptr0 = (r + 0 < n_لیکې_g) ? ((ثابت uint8_t *) q_معلومات + (q_پیل + q_idx0) * q_nb1 +
                                                      (kv_سر * G + h_idx0) * q_nb2 + ib3 * q_nb3) :
                                                      عدم;
        ثابت uint8_t * q_ptr1 = (r + 1 < n_لیکې_g) ? ((ثابت uint8_t *) q_معلومات + (q_پیل + q_idx1) * q_nb1 +
                                                      (kv_سر * G + h_idx1) * q_nb2 + ib3 * q_nb3) :
                                                      عدم;

        size_t   r0       = r / HMX_FP16_TILE_N_لیکې;
        size_t   r1       = r % HMX_FP16_TILE_N_لیکې;
        __fp16 * بهر_بنسټ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_لیکې * dk_بهر;

        إذا (is_q_fp32) {
            ثابت HVX_UVector * pv_in0 = q_ptr0 ? (ثابت HVX_UVector *) q_ptr0 : عدم;
            ثابت HVX_UVector * pv_in1 = q_ptr1 ? (ثابت HVX_UVector *) q_ptr1 : عدم;

            لكل (uint32_t d = 0; d < dk_بهر / 32; ++d) {
                HVX_ویکټور * بهر_tile   = (HVX_ویکټور *) (بهر_بنسټ + d * HMX_FP16_TILE_N_ELMS);
                ثابت size_t بنسټ_lane  = (size_t) d * 32;
                ثابت size_t real_lanes = (بنسټ_lane < dk_دننه) ? hex_smin(32, dk_دننه - بنسټ_lane) : 0;

                إذا (real_lanes == 0) {
                    بهر_tile[r1 / 2] = Q6_V_vzero();
                    استمر;
                }
                HVX_ویکټور v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_ویکټور v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                إذا (real_lanes < 32) {
                    ثابت HVX_ویکټور_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * حجم(عائم)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                بهر_tile[r1 / 2] = hvx_vec_f32_منزل_f16_shuff(v0, v1);
            }
        } وإلا {
            ثابت HVX_UVector * pv_in0 = q_ptr0 ? (ثابت HVX_UVector *) q_ptr0 : عدم;
            ثابت HVX_UVector * pv_in1 = q_ptr1 ? (ثابت HVX_UVector *) q_ptr1 : عدم;

            لكل (uint32_t d = 0; d < dk_بهر / 64; ++d) {
                __fp16 *     بهر_dtile = بهر_بنسټ + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ویکټور * pv_out0   = ((HVX_ویکټور *) بهر_dtile) + r1 / 2;
                HVX_ویکټور * pv_out1   = pv_out0 + 16;

                ثابت size_t بنسټ_lane  = (size_t) d * 64;
                ثابت size_t real_lanes = (بنسټ_lane < dk_دننه) ? hex_smin(64, dk_دننه - بنسټ_lane) : 0;

                إذا (real_lanes == 0) {
                    *pv_out0 = Q6_V_vzero();
                    *pv_out1 = Q6_V_vzero();
                    استمر;
                }
                HVX_ویکټور v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_ویکټور v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                إذا (real_lanes < 64) {
                    ثابت HVX_ویکټور_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * حجم(__fp16)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                HVX_ویکټور_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);
                *pv_out0 = Q6_V_lo_W(vp);
                *pv_out1 = Q6_V_hi_W(vp);
            }
        }
    }
}

#د_که_پای /* HMX_FA_KERNELS_H */
