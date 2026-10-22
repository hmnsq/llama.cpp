#جيڪڏهن_وصف_نه_ٿيل HMX_FA_KERNELS_H
#وصف_ڪريو HMX_FA_KERNELS_H

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
    ثابت __fp16 * صف_tiles,
    ثابت __fp16 * کانو_tiles,
    __fp16 *       ٻاهريون_tile,
    size_t         n_dot_tiles
) {
    إذا (n_dot_tiles == 2) {
        asm متطاير(
            HMX_ڀريو_MPY_F16("%1", "%2", "%0")
            HMX_ڀريو_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(صف_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(کانو_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(کانو_tiles + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } وإلا إذا (n_dot_tiles == 4) {
        asm متطاير(
            HMX_ڀريو_MPY_F16("%1", "%2", "%0")
            HMX_ڀريو_MPY_F16("%3", "%4", "%0")
            HMX_ڀريو_MPY_F16("%5", "%6", "%0")
            HMX_ڀريو_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(صف_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(کانو_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(کانو_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(کانو_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(کانو_tiles + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } وإلا إذا (n_dot_tiles == 8) {
        asm متطاير(
            HMX_ڀريو_MPY_F16("%1", "%2", "%0")
            HMX_ڀريو_MPY_F16("%3", "%4", "%0")
            HMX_ڀريو_MPY_F16("%5", "%6", "%0")
            HMX_ڀريو_MPY_F16("%7", "%8", "%0")
            HMX_ڀريو_MPY_F16("%9", "%10", "%0")
            HMX_ڀريو_MPY_F16("%11", "%12", "%0")
            HMX_ڀريو_MPY_F16("%13", "%14", "%0")
            HMX_ڀريو_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(صف_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(کانو_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(کانو_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(کانو_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(کانو_tiles + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 4 * HMX_FP16_TILE_N_ELMS), "r"(کانو_tiles + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 5 * HMX_FP16_TILE_N_ELMS), "r"(کانو_tiles + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 6 * HMX_FP16_TILE_N_ELMS), "r"(کانو_tiles + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 7 * HMX_FP16_TILE_N_ELMS), "r"(کانو_tiles + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } وإلا {
        لكل (size_t k = 0; k < n_dot_tiles; ++k) {
            asm متطاير(
                HMX_ڀريو_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(صف_tiles), "r"(کانو_tiles)
            );
            صف_tiles += HMX_FP16_TILE_N_ELMS;
            کانو_tiles += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm متطاير(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(ٻاهريون_tile), "r"(0)
        : "memory"
    );
}

ساكن مضمن فراغ hmx_fa_o_update_tile(
    ثابت __fp16 * d_diag,
    ثابت __fp16 * o_rc,
    ثابت __fp16 * p_tile_اندريون,
    ثابت __fp16 * v_tile_اندريون,
    __fp16 *       o_tile_ٻاهريون,
    size_t         n_کانو_tiles
) {
    asm متطاير(
        HMX_ڀريو_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    إذا (n_کانو_tiles == 2) {
        asm متطاير(
            HMX_ڀريو_MPY_F16("%1", "%2", "%0")
            HMX_ڀريو_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(p_tile_اندريون + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندريون + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندريون + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندريون + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } وإلا إذا (n_کانو_tiles == 4) {
        asm متطاير(
            HMX_ڀريو_MPY_F16("%1", "%2", "%0")
            HMX_ڀريو_MPY_F16("%3", "%4", "%0")
            HMX_ڀريو_MPY_F16("%5", "%6", "%0")
            HMX_ڀريو_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(p_tile_اندريون + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندريون + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندريون + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندريون + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندريون + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندريون + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندريون + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندريون + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } وإلا إذا (n_کانو_tiles == 8) {
        asm متطاير(
            HMX_ڀريو_MPY_F16("%1", "%2", "%0")
            HMX_ڀريو_MPY_F16("%3", "%4", "%0")
            HMX_ڀريو_MPY_F16("%5", "%6", "%0")
            HMX_ڀريو_MPY_F16("%7", "%8", "%0")
            HMX_ڀريو_MPY_F16("%9", "%10", "%0")
            HMX_ڀريو_MPY_F16("%11", "%12", "%0")
            HMX_ڀريو_MPY_F16("%13", "%14", "%0")
            HMX_ڀريو_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(p_tile_اندريون + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندريون + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندريون + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندريون + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندريون + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندريون + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندريون + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندريون + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندريون + 4 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندريون + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندريون + 5 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندريون + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندريون + 6 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندريون + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندريون + 7 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندريون + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } وإلا {
        لكل (size_t k = 0; k < n_کانو_tiles; ++k) {
            asm متطاير(
                HMX_ڀريو_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(p_tile_اندريون), "r"(v_tile_اندريون)
            );
            p_tile_اندريون += HMX_FP16_TILE_N_ELMS;
            v_tile_اندريون += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm متطاير(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_tile_ٻاهريون), "r"(0)
        : "memory"
    );
}

ساكن مضمن فراغ hmx_fa_o_norm_tile(
    ثابت __fp16 * d_diag,
    ثابت __fp16 * o_rc,
    __fp16 *       o_ٻاهريون
) {
    asm متطاير(
        HMX_ڀريو_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    asm متطاير(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_ٻاهريون), "r"(0)
        : "memory"
    );
}

ساكن مضمن فراغ hmx_fa_q_prep_fp32_d2(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * عارضي_q_vtcm,
    size_t شروعات, size_t پڇاڙي, size_t g_صفون_پڇاڙي,
    size_t DK, size_t G, size_t n_صفون_q,
    ثابت بنية fastdiv_قدرون * div_G, bool q_transposed
) {
    لكل (size_t r = شروعات; r < پڇاڙي; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفون;
        size_t   r1       = r % HMX_FP16_TILE_N_صفون;
        __fp16 * ٻاهريون_بنياد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفون * DK;

        إذا (r >= g_صفون_پڇاڙي) {
            ((HVX_ويڪٽر *) (ٻاهريون_بنياد + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            ((HVX_ويڪٽر *) (ٻاهريون_بنياد + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفون_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفون_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_ويڪٽر * pv_in0 = (ثابت HVX_ويڪٽر *) (عارضي_q_vtcm + offset0 * DK * حجم(عائم));
        ثابت HVX_ويڪٽر * pv_in1 = (r + 1 < g_صفون_پڇاڙي)
            ? (ثابت HVX_ويڪٽر *) (عارضي_q_vtcm + offset1 * DK * حجم(عائم))
            : عدم;

        {
            HVX_ويڪٽر v0 = pv_in0[0];
            HVX_ويڪٽر v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_ويڪٽر v_hf = hvx_vec_f32_منزل_f16_shuff(v0, v1);
            ((HVX_ويڪٽر *) (ٻاهريون_بنياد + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
        {
            HVX_ويڪٽر v0 = pv_in0[1];
            HVX_ويڪٽر v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_ويڪٽر v_hf = hvx_vec_f32_منزل_f16_shuff(v0, v1);
            ((HVX_ويڪٽر *) (ٻاهريون_بنياد + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

ساكن مضمن فراغ hmx_fa_q_prep_fp32_d4(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * عارضي_q_vtcm,
    size_t شروعات, size_t پڇاڙي, size_t g_صفون_پڇاڙي,
    size_t DK, size_t G, size_t n_صفون_q,
    ثابت بنية fastdiv_قدرون * div_G, bool q_transposed
) {
    لكل (size_t r = شروعات; r < پڇاڙي; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفون;
        size_t   r1       = r % HMX_FP16_TILE_N_صفون;
        __fp16 * ٻاهريون_بنياد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفون * DK;

        إذا (r >= g_صفون_پڇاڙي) {
            لكل (uint32_t d = 0; d < 4; ++d) {
                ((HVX_ويڪٽر *) (ٻاهريون_بنياد + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفون_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفون_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_ويڪٽر * pv_in0 = (ثابت HVX_ويڪٽر *) (عارضي_q_vtcm + offset0 * DK * حجم(عائم));
        ثابت HVX_ويڪٽر * pv_in1 = (r + 1 < g_صفون_پڇاڙي)
            ? (ثابت HVX_ويڪٽر *) (عارضي_q_vtcm + offset1 * DK * حجم(عائم))
            : عدم;

        لكل (uint32_t d = 0; d < 4; ++d) {
            HVX_ويڪٽر v0 = pv_in0[d];
            HVX_ويڪٽر v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_ويڪٽر v_hf = hvx_vec_f32_منزل_f16_shuff(v0, v1);
            ((HVX_ويڪٽر *) (ٻاهريون_بنياد + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

ساكن مضمن فراغ hmx_fa_q_prep_fp32(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * عارضي_q_vtcm,
    size_t شروعات, size_t پڇاڙي, size_t g_صفون_پڇاڙي,
    size_t DK, size_t G, size_t n_صفون_q,
    ثابت بنية fastdiv_قدرون * div_G, uint32_t d_حد_بندي, bool q_transposed
) {
    لكل (size_t r = شروعات; r < پڇاڙي; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفون;
        size_t   r1       = r % HMX_FP16_TILE_N_صفون;
        __fp16 * ٻاهريون_بنياد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفون * DK;

        إذا (r >= g_صفون_پڇاڙي) {
            لكل (uint32_t d = 0; d < d_حد_بندي; ++d) {
                ((HVX_ويڪٽر *) (ٻاهريون_بنياد + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفون_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفون_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_ويڪٽر * pv_in0 = (ثابت HVX_ويڪٽر *) (عارضي_q_vtcm + offset0 * DK * حجم(عائم));
        ثابت HVX_ويڪٽر * pv_in1 = (r + 1 < g_صفون_پڇاڙي)
            ? (ثابت HVX_ويڪٽر *) (عارضي_q_vtcm + offset1 * DK * حجم(عائم))
            : عدم;

        لكل (uint32_t d = 0; d < d_حد_بندي; ++d) {
            HVX_ويڪٽر v0   = pv_in0[d];
            HVX_ويڪٽر v1   = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_ويڪٽر v_hf = hvx_vec_f32_منزل_f16_shuff(v0, v1);

            HVX_ويڪٽر * ٻاهريون_tile = (HVX_ويڪٽر *) (ٻاهريون_بنياد + d * HMX_FP16_TILE_N_ELMS);
            ٻاهريون_tile[r1 / 2]      = v_hf;
        }
    }
}

ساكن مضمن فراغ hmx_fa_q_prep_fp16_d1(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * عارضي_q_vtcm,
    size_t شروعات, size_t پڇاڙي, size_t g_صفون_پڇاڙي,
    size_t DK, size_t G, size_t n_صفون_q,
    ثابت بنية fastdiv_قدرون * div_G, bool q_transposed
) {
    لكل (size_t r = شروعات; r < پڇاڙي; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفون;
        size_t   r1       = r % HMX_FP16_TILE_N_صفون;
        __fp16 * ٻاهريون_بنياد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفون * DK;

        إذا (r >= g_صفون_پڇاڙي) {
            __fp16 *     ٻاهريون_dtile = ٻاهريون_بنياد + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ويڪٽر * pv_out0   = ((HVX_ويڪٽر *) ٻاهريون_dtile) + r1 / 2;
            HVX_ويڪٽر * pv_out1   = pv_out0 + 16;
            *pv_out0 = Q6_V_vzero();
            *pv_out1 = Q6_V_vzero();
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفون_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفون_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_ويڪٽر * pv_in0 = (ثابت HVX_ويڪٽر *) (عارضي_q_vtcm + offset0 * DK * حجم(__fp16));
        ثابت HVX_ويڪٽر * pv_in1 = (r + 1 < g_صفون_پڇاڙي)
            ? (ثابت HVX_ويڪٽر *) (عارضي_q_vtcm + offset1 * DK * حجم(__fp16))
            : عدم;

        HVX_ويڪٽر     v0 = pv_in0[0];
        HVX_ويڪٽر     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
        HVX_ويڪٽر_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        __fp16 *     ٻاهريون_dtile = ٻاهريون_بنياد + 0 * HMX_FP16_TILE_N_ELMS * 2;
        HVX_ويڪٽر * pv_out0   = ((HVX_ويڪٽر *) ٻاهريون_dtile) + r1 / 2;
        HVX_ويڪٽر * pv_out1   = pv_out0 + 16;

        *pv_out0 = Q6_V_lo_W(vp);
        *pv_out1 = Q6_V_hi_W(vp);
    }
}

ساكن مضمن فراغ hmx_fa_q_prep_fp16_d2(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * عارضي_q_vtcm,
    size_t شروعات, size_t پڇاڙي, size_t g_صفون_پڇاڙي,
    size_t DK, size_t G, size_t n_صفون_q,
    ثابت بنية fastdiv_قدرون * div_G, bool q_transposed
) {
    لكل (size_t r = شروعات; r < پڇاڙي; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفون;
        size_t   r1       = r % HMX_FP16_TILE_N_صفون;
        __fp16 * ٻاهريون_بنياد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفون * DK;

        إذا (r >= g_صفون_پڇاڙي) {
            لكل (uint32_t d = 0; d < 2; ++d) {
                __fp16 *     ٻاهريون_dtile = ٻاهريون_بنياد + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ويڪٽر * pv_out0   = ((HVX_ويڪٽر *) ٻاهريون_dtile) + r1 / 2;
                HVX_ويڪٽر * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفون_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفون_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_ويڪٽر * pv_in0 = (ثابت HVX_ويڪٽر *) (عارضي_q_vtcm + offset0 * DK * حجم(__fp16));
        ثابت HVX_ويڪٽر * pv_in1 = (r + 1 < g_صفون_پڇاڙي)
            ? (ثابت HVX_ويڪٽر *) (عارضي_q_vtcm + offset1 * DK * حجم(__fp16))
            : عدم;

        {
            HVX_ويڪٽر     v0 = pv_in0[0];
            HVX_ويڪٽر     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_ويڪٽر_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     ٻاهريون_dtile = ٻاهريون_بنياد + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ويڪٽر * pv_out0   = ((HVX_ويڪٽر *) ٻاهريون_dtile) + r1 / 2;
            HVX_ويڪٽر * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
        {
            HVX_ويڪٽر     v0 = pv_in0[1];
            HVX_ويڪٽر     v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_ويڪٽر_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     ٻاهريون_dtile = ٻاهريون_بنياد + 1 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ويڪٽر * pv_out0   = ((HVX_ويڪٽر *) ٻاهريون_dtile) + r1 / 2;
            HVX_ويڪٽر * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}

ساكن مضمن فراغ hmx_fa_q_prep_fp16(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * عارضي_q_vtcm,
    size_t شروعات, size_t پڇاڙي, size_t g_صفون_پڇاڙي,
    size_t DK, size_t G, size_t n_صفون_q,
    ثابت بنية fastdiv_قدرون * div_G, uint32_t d_حد_بندي, bool q_transposed
) {
    لكل (size_t r = شروعات; r < پڇاڙي; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفون;
        size_t   r1       = r % HMX_FP16_TILE_N_صفون;
        __fp16 * ٻاهريون_بنياد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفون * DK;

        إذا (r >= g_صفون_پڇاڙي) {
            لكل (uint32_t d = 0; d < d_حد_بندي; ++d) {
                __fp16 *     ٻاهريون_dtile = ٻاهريون_بنياد + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ويڪٽر * pv_out0   = ((HVX_ويڪٽر *) ٻاهريون_dtile) + r1 / 2;
                HVX_ويڪٽر * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفون_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفون_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_ويڪٽر * pv_in0 = (ثابت HVX_ويڪٽر *) (عارضي_q_vtcm + offset0 * DK * حجم(__fp16));
        ثابت HVX_ويڪٽر * pv_in1 = (r + 1 < g_صفون_پڇاڙي)
            ? (ثابت HVX_ويڪٽر *) (عارضي_q_vtcm + offset1 * DK * حجم(__fp16))
            : عدم;

        لكل (uint32_t d = 0; d < d_حد_بندي; ++d) {
            HVX_ويڪٽر     v0 = pv_in0[d];
            HVX_ويڪٽر     v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_ويڪٽر_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     ٻاهريون_dtile = ٻاهريون_بنياد + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ويڪٽر * pv_out0   = ((HVX_ويڪٽر *) ٻاهريون_dtile) + r1 / 2;
            HVX_ويڪٽر * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}


// Head-dim-padded Q-prep (f32). Used when DK is not a multiple of 64.
ساكن مضمن فراغ hmx_fa_q_prep_fp32_pad(__fp16 *                      vtcm_q_tiles,
                                          ثابت uint8_t *               عارضي_q_vtcm,
                                          size_t                        شروعات,
                                          size_t                        پڇاڙي,
                                          size_t                        g_صفون_پڇاڙي,
                                          size_t                        dk_اندريون,
                                          size_t                        dk_ٻاهريون,
                                          size_t                        G,
                                          size_t                        n_صفون_q,
                                          ثابت بنية fastdiv_قدرون * div_G,
                                          bool                          q_transposed) {
    ثابت uint32_t n_ٻاهريون_tiles = (uint32_t) (dk_ٻاهريون / 32);
    لكل (size_t r = شروعات; r < پڇاڙي; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفون;
        size_t   r1       = r % HMX_FP16_TILE_N_صفون;
        __fp16 * ٻاهريون_بنياد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفون * dk_ٻاهريون;

        إذا (r >= g_صفون_پڇاڙي) {
            لكل (uint32_t d = 0; d < n_ٻاهريون_tiles; ++d) {
                ((HVX_ويڪٽر *) (ٻاهريون_بنياد + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفون_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفون_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_UVector * pv_in0 = (ثابت HVX_UVector *) (عارضي_q_vtcm + offset0 * dk_اندريون * حجم(عائم));
        ثابت HVX_UVector * pv_in1 = (r + 1 < g_صفون_پڇاڙي) ? (ثابت HVX_UVector *) (عارضي_q_vtcm + offset1 * dk_اندريون * حجم(عائم)) : عدم;

        لكل (uint32_t d = 0; d < n_ٻاهريون_tiles; ++d) {
            HVX_ويڪٽر * ٻاهريون_tile   = (HVX_ويڪٽر *) (ٻاهريون_بنياد + d * HMX_FP16_TILE_N_ELMS);
            ثابت size_t بنياد_lane  = (size_t) d * 32;
            ثابت size_t real_lanes = (بنياد_lane < dk_اندريون) ? hex_smin(32, dk_اندريون - بنياد_lane) : 0;

            إذا (real_lanes == 0) {
                ٻاهريون_tile[r1 / 2] = Q6_V_vzero();
                استمر;
            }

            HVX_ويڪٽر v0 = pv_in0[d];
            HVX_ويڪٽر v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            إذا (real_lanes < 32) {
                // Straddle tile: keep the first real_lanes floats, zero the padded tail so
                // the packed f16 lanes beyond DK are zero.
                ثابت HVX_ويڪٽر_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * حجم(عائم)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            ٻاهريون_tile[r1 / 2] = hvx_vec_f32_منزل_f16_shuff(v0, v1);
        }
    }
}

// Head-dim-padded Q-prep (f16). Used when DK is not a multiple of 64.
ساكن مضمن فراغ hmx_fa_q_prep_fp16_pad(__fp16 *                      vtcm_q_tiles,
                                          ثابت uint8_t *               عارضي_q_vtcm,
                                          size_t                        شروعات,
                                          size_t                        پڇاڙي,
                                          size_t                        g_صفون_پڇاڙي,
                                          size_t                        dk_اندريون,
                                          size_t                        dk_ٻاهريون,
                                          size_t                        G,
                                          size_t                        n_صفون_q,
                                          ثابت بنية fastdiv_قدرون * div_G,
                                          bool                          q_transposed) {
    ثابت uint32_t n_ٻاهريون_جوڙا = (uint32_t) (dk_ٻاهريون / 64);
    لكل (size_t r = شروعات; r < پڇاڙي; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفون;
        size_t   r1       = r % HMX_FP16_TILE_N_صفون;
        __fp16 * ٻاهريون_بنياد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفون * dk_ٻاهريون;

        إذا (r >= g_صفون_پڇاڙي) {
            لكل (uint32_t d = 0; d < n_ٻاهريون_جوڙا; ++d) {
                __fp16 *     ٻاهريون_dtile = ٻاهريون_بنياد + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ويڪٽر * pv_out0   = ((HVX_ويڪٽر *) ٻاهريون_dtile) + r1 / 2;
                HVX_ويڪٽر * pv_out1   = pv_out0 + 16;
                *pv_out0               = Q6_V_vzero();
                *pv_out1               = Q6_V_vzero();
            }
            استمر;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفون_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفون_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_UVector * pv_in0 = (ثابت HVX_UVector *) (عارضي_q_vtcm + offset0 * dk_اندريون * حجم(__fp16));
        ثابت HVX_UVector * pv_in1 = (r + 1 < g_صفون_پڇاڙي) ? (ثابت HVX_UVector *) (عارضي_q_vtcm + offset1 * dk_اندريون * حجم(__fp16)) : عدم;

        لكل (uint32_t d = 0; d < n_ٻاهريون_جوڙا; ++d) {
            __fp16 *     ٻاهريون_dtile = ٻاهريون_بنياد + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ويڪٽر * pv_out0   = ((HVX_ويڪٽر *) ٻاهريون_dtile) + r1 / 2;
            HVX_ويڪٽر * pv_out1   = pv_out0 + 16;

            ثابت size_t بنياد_lane  = (size_t) d * 64;
            ثابت size_t real_lanes = (بنياد_lane < dk_اندريون) ? hex_smin(64, dk_اندريون - بنياد_lane) : 0;

            إذا (real_lanes == 0) {
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
                استمر;
            }

            HVX_ويڪٽر v0 = pv_in0[d];
            HVX_ويڪٽر v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            إذا (real_lanes < 64) {
                ثابت HVX_ويڪٽر_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * حجم(__fp16)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            HVX_ويڪٽر_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);
            *pv_out0          = Q6_V_lo_W(vp);
            *pv_out1          = Q6_V_hi_W(vp);
        }
    }
}

ساكن مضمن فراغ hmx_fa_q_prep_fallback(
    __fp16 * vtcm_q_tiles, uintptr_t q_معلومات,
    size_t q_nb1, size_t q_nb2, size_t q_nb3,
    uint32_t q_شروعات, uint32_t kv_سر, uint32_t ib3,
    size_t شروعات, size_t پڇاڙي, size_t n_صفون_g,
    size_t G, size_t dk_اندريون, size_t dk_ٻاهريون, bool is_q_fp32,
    ثابت بنية fastdiv_قدرون * div_G
) {
    لكل (size_t r = شروعات; r < پڇاڙي; r += 2) {
        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت uint8_t * q_ptr0 = (r + 0 < n_صفون_g) ? ((ثابت uint8_t *) q_معلومات + (q_شروعات + q_idx0) * q_nb1 +
                                                      (kv_سر * G + h_idx0) * q_nb2 + ib3 * q_nb3) :
                                                      عدم;
        ثابت uint8_t * q_ptr1 = (r + 1 < n_صفون_g) ? ((ثابت uint8_t *) q_معلومات + (q_شروعات + q_idx1) * q_nb1 +
                                                      (kv_سر * G + h_idx1) * q_nb2 + ib3 * q_nb3) :
                                                      عدم;

        size_t   r0       = r / HMX_FP16_TILE_N_صفون;
        size_t   r1       = r % HMX_FP16_TILE_N_صفون;
        __fp16 * ٻاهريون_بنياد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفون * dk_ٻاهريون;

        إذا (is_q_fp32) {
            ثابت HVX_UVector * pv_in0 = q_ptr0 ? (ثابت HVX_UVector *) q_ptr0 : عدم;
            ثابت HVX_UVector * pv_in1 = q_ptr1 ? (ثابت HVX_UVector *) q_ptr1 : عدم;

            لكل (uint32_t d = 0; d < dk_ٻاهريون / 32; ++d) {
                HVX_ويڪٽر * ٻاهريون_tile   = (HVX_ويڪٽر *) (ٻاهريون_بنياد + d * HMX_FP16_TILE_N_ELMS);
                ثابت size_t بنياد_lane  = (size_t) d * 32;
                ثابت size_t real_lanes = (بنياد_lane < dk_اندريون) ? hex_smin(32, dk_اندريون - بنياد_lane) : 0;

                إذا (real_lanes == 0) {
                    ٻاهريون_tile[r1 / 2] = Q6_V_vzero();
                    استمر;
                }
                HVX_ويڪٽر v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_ويڪٽر v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                إذا (real_lanes < 32) {
                    ثابت HVX_ويڪٽر_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * حجم(عائم)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                ٻاهريون_tile[r1 / 2] = hvx_vec_f32_منزل_f16_shuff(v0, v1);
            }
        } وإلا {
            ثابت HVX_UVector * pv_in0 = q_ptr0 ? (ثابت HVX_UVector *) q_ptr0 : عدم;
            ثابت HVX_UVector * pv_in1 = q_ptr1 ? (ثابت HVX_UVector *) q_ptr1 : عدم;

            لكل (uint32_t d = 0; d < dk_ٻاهريون / 64; ++d) {
                __fp16 *     ٻاهريون_dtile = ٻاهريون_بنياد + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ويڪٽر * pv_out0   = ((HVX_ويڪٽر *) ٻاهريون_dtile) + r1 / 2;
                HVX_ويڪٽر * pv_out1   = pv_out0 + 16;

                ثابت size_t بنياد_lane  = (size_t) d * 64;
                ثابت size_t real_lanes = (بنياد_lane < dk_اندريون) ? hex_smin(64, dk_اندريون - بنياد_lane) : 0;

                إذا (real_lanes == 0) {
                    *pv_out0 = Q6_V_vzero();
                    *pv_out1 = Q6_V_vzero();
                    استمر;
                }
                HVX_ويڪٽر v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_ويڪٽر v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                إذا (real_lanes < 64) {
                    ثابت HVX_ويڪٽر_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * حجم(__fp16)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                HVX_ويڪٽر_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);
                *pv_out0 = Q6_V_lo_W(vp);
                *pv_out1 = Q6_V_hi_W(vp);
            }
        }
    }
}

#جيڪڏهن_پورو /* HMX_FA_KERNELS_H */
