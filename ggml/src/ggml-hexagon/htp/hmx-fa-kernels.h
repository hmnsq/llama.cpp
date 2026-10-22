#اگر_غیر_تعریف_شدہ HMX_FA_KERNELS_H
#تعریف_کرو HMX_FA_KERNELS_H

#شامل <stdint.h>
#شامل <stddef.h>
#شامل <stdbool.h>
#شامل "hvx-utils.h"
#شامل "hmx-utils.h"
#شامل "hex-fastdiv.h"

// HMX-specific parameters, offsets and inner kernels for Flash Attention

// Scatter offsets for diagonal tile: entry[2i] = i*136, entry[2i+1] = i*136+6
// 136 = 4 * 32 + 8 = byte offset to diagonal in a 32x32 fp16 interleaved tile
ساکن ثابت int16_t d_tile_scatter_offsets[64] __attribute__((aligned(128))) = {
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

ساکن اندرونی خالی hmx_fa_qk_dot_tile(
    ثابت __fp16 * صف_tiles,
    ثابت __fp16 * ستون_tiles,
    __fp16 *       برآمد_tile,
    size_t         n_dot_tiles
) {
    اگر (n_dot_tiles == 2) {
        asm متلون(
            HMX_بھرو_MPY_F16("%1", "%2", "%0")
            HMX_بھرو_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(صف_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } ورنہ اگر (n_dot_tiles == 4) {
        asm متلون(
            HMX_بھرو_MPY_F16("%1", "%2", "%0")
            HMX_بھرو_MPY_F16("%3", "%4", "%0")
            HMX_بھرو_MPY_F16("%5", "%6", "%0")
            HMX_بھرو_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(صف_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } ورنہ اگر (n_dot_tiles == 8) {
        asm متلون(
            HMX_بھرو_MPY_F16("%1", "%2", "%0")
            HMX_بھرو_MPY_F16("%3", "%4", "%0")
            HMX_بھرو_MPY_F16("%5", "%6", "%0")
            HMX_بھرو_MPY_F16("%7", "%8", "%0")
            HMX_بھرو_MPY_F16("%9", "%10", "%0")
            HMX_بھرو_MPY_F16("%11", "%12", "%0")
            HMX_بھرو_MPY_F16("%13", "%14", "%0")
            HMX_بھرو_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(صف_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 4 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 5 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 6 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(صف_tiles + 7 * HMX_FP16_TILE_N_ELMS), "r"(ستون_tiles + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } ورنہ {
        برائے (size_t k = 0; k < n_dot_tiles; ++k) {
            asm متلون(
                HMX_بھرو_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(صف_tiles), "r"(ستون_tiles)
            );
            صف_tiles += HMX_FP16_TILE_N_ELMS;
            ستون_tiles += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm متلون(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(برآمد_tile), "r"(0)
        : "memory"
    );
}

ساکن اندرونی خالی hmx_fa_o_update_tile(
    ثابت __fp16 * d_diag,
    ثابت __fp16 * o_rc,
    ثابت __fp16 * p_tile_اندر,
    ثابت __fp16 * v_tile_اندر,
    __fp16 *       o_tile_برآمد,
    size_t         n_ستون_tiles
) {
    asm متلون(
        HMX_بھرو_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    اگر (n_ستون_tiles == 2) {
        asm متلون(
            HMX_بھرو_MPY_F16("%1", "%2", "%0")
            HMX_بھرو_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(p_tile_اندر + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندر + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندر + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندر + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } ورنہ اگر (n_ستون_tiles == 4) {
        asm متلون(
            HMX_بھرو_MPY_F16("%1", "%2", "%0")
            HMX_بھرو_MPY_F16("%3", "%4", "%0")
            HMX_بھرو_MPY_F16("%5", "%6", "%0")
            HMX_بھرو_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(p_tile_اندر + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندر + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندر + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندر + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندر + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندر + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندر + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندر + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } ورنہ اگر (n_ستون_tiles == 8) {
        asm متلون(
            HMX_بھرو_MPY_F16("%1", "%2", "%0")
            HMX_بھرو_MPY_F16("%3", "%4", "%0")
            HMX_بھرو_MPY_F16("%5", "%6", "%0")
            HMX_بھرو_MPY_F16("%7", "%8", "%0")
            HMX_بھرو_MPY_F16("%9", "%10", "%0")
            HMX_بھرو_MPY_F16("%11", "%12", "%0")
            HMX_بھرو_MPY_F16("%13", "%14", "%0")
            HMX_بھرو_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(p_tile_اندر + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندر + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندر + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندر + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندر + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندر + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندر + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندر + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندر + 4 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندر + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندر + 5 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندر + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندر + 6 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندر + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_اندر + 7 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_اندر + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } ورنہ {
        برائے (size_t k = 0; k < n_ستون_tiles; ++k) {
            asm متلون(
                HMX_بھرو_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(p_tile_اندر), "r"(v_tile_اندر)
            );
            p_tile_اندر += HMX_FP16_TILE_N_ELMS;
            v_tile_اندر += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm متلون(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_tile_برآمد), "r"(0)
        : "memory"
    );
}

ساکن اندرونی خالی hmx_fa_o_norm_tile(
    ثابت __fp16 * d_diag,
    ثابت __fp16 * o_rc,
    __fp16 *       o_برآمد
) {
    asm متلون(
        HMX_بھرو_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    asm متلون(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_برآمد), "r"(0)
        : "memory"
    );
}

ساکن اندرونی خالی hmx_fa_q_prep_fp32_d2(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * عارضی_q_vtcm,
    size_t آغاز, size_t انتہا, size_t g_صفیں_انتہا,
    size_t DK, size_t G, size_t n_صفیں_q,
    ثابت ساخت fastdiv_اقدار * div_G, bool q_transposed
) {
    برائے (size_t r = آغاز; r < انتہا; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفیں;
        size_t   r1       = r % HMX_FP16_TILE_N_صفیں;
        __fp16 * برآمد_بنیاد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفیں * DK;

        اگر (r >= g_صفیں_انتہا) {
            ((HVX_سمتیہ *) (برآمد_بنیاد + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            ((HVX_سمتیہ *) (برآمد_بنیاد + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            جاری;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفیں_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفیں_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_سمتیہ * pv_in0 = (ثابت HVX_سمتیہ *) (عارضی_q_vtcm + offset0 * DK * حجم(اعشاری));
        ثابت HVX_سمتیہ * pv_in1 = (r + 1 < g_صفیں_انتہا)
            ? (ثابت HVX_سمتیہ *) (عارضی_q_vtcm + offset1 * DK * حجم(اعشاری))
            : تہی;

        {
            HVX_سمتیہ v0 = pv_in0[0];
            HVX_سمتیہ v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_سمتیہ v_hf = hvx_vec_f32_مقصد_f16_shuff(v0, v1);
            ((HVX_سمتیہ *) (برآمد_بنیاد + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
        {
            HVX_سمتیہ v0 = pv_in0[1];
            HVX_سمتیہ v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_سمتیہ v_hf = hvx_vec_f32_مقصد_f16_shuff(v0, v1);
            ((HVX_سمتیہ *) (برآمد_بنیاد + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

ساکن اندرونی خالی hmx_fa_q_prep_fp32_d4(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * عارضی_q_vtcm,
    size_t آغاز, size_t انتہا, size_t g_صفیں_انتہا,
    size_t DK, size_t G, size_t n_صفیں_q,
    ثابت ساخت fastdiv_اقدار * div_G, bool q_transposed
) {
    برائے (size_t r = آغاز; r < انتہا; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفیں;
        size_t   r1       = r % HMX_FP16_TILE_N_صفیں;
        __fp16 * برآمد_بنیاد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفیں * DK;

        اگر (r >= g_صفیں_انتہا) {
            برائے (uint32_t d = 0; d < 4; ++d) {
                ((HVX_سمتیہ *) (برآمد_بنیاد + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            جاری;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفیں_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفیں_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_سمتیہ * pv_in0 = (ثابت HVX_سمتیہ *) (عارضی_q_vtcm + offset0 * DK * حجم(اعشاری));
        ثابت HVX_سمتیہ * pv_in1 = (r + 1 < g_صفیں_انتہا)
            ? (ثابت HVX_سمتیہ *) (عارضی_q_vtcm + offset1 * DK * حجم(اعشاری))
            : تہی;

        برائے (uint32_t d = 0; d < 4; ++d) {
            HVX_سمتیہ v0 = pv_in0[d];
            HVX_سمتیہ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_سمتیہ v_hf = hvx_vec_f32_مقصد_f16_shuff(v0, v1);
            ((HVX_سمتیہ *) (برآمد_بنیاد + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

ساکن اندرونی خالی hmx_fa_q_prep_fp32(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * عارضی_q_vtcm,
    size_t آغاز, size_t انتہا, size_t g_صفیں_انتہا,
    size_t DK, size_t G, size_t n_صفیں_q,
    ثابت ساخت fastdiv_اقدار * div_G, uint32_t d_حد_بندی, bool q_transposed
) {
    برائے (size_t r = آغاز; r < انتہا; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفیں;
        size_t   r1       = r % HMX_FP16_TILE_N_صفیں;
        __fp16 * برآمد_بنیاد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفیں * DK;

        اگر (r >= g_صفیں_انتہا) {
            برائے (uint32_t d = 0; d < d_حد_بندی; ++d) {
                ((HVX_سمتیہ *) (برآمد_بنیاد + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            جاری;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفیں_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفیں_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_سمتیہ * pv_in0 = (ثابت HVX_سمتیہ *) (عارضی_q_vtcm + offset0 * DK * حجم(اعشاری));
        ثابت HVX_سمتیہ * pv_in1 = (r + 1 < g_صفیں_انتہا)
            ? (ثابت HVX_سمتیہ *) (عارضی_q_vtcm + offset1 * DK * حجم(اعشاری))
            : تہی;

        برائے (uint32_t d = 0; d < d_حد_بندی; ++d) {
            HVX_سمتیہ v0   = pv_in0[d];
            HVX_سمتیہ v1   = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_سمتیہ v_hf = hvx_vec_f32_مقصد_f16_shuff(v0, v1);

            HVX_سمتیہ * برآمد_tile = (HVX_سمتیہ *) (برآمد_بنیاد + d * HMX_FP16_TILE_N_ELMS);
            برآمد_tile[r1 / 2]      = v_hf;
        }
    }
}

ساکن اندرونی خالی hmx_fa_q_prep_fp16_d1(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * عارضی_q_vtcm,
    size_t آغاز, size_t انتہا, size_t g_صفیں_انتہا,
    size_t DK, size_t G, size_t n_صفیں_q,
    ثابت ساخت fastdiv_اقدار * div_G, bool q_transposed
) {
    برائے (size_t r = آغاز; r < انتہا; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفیں;
        size_t   r1       = r % HMX_FP16_TILE_N_صفیں;
        __fp16 * برآمد_بنیاد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفیں * DK;

        اگر (r >= g_صفیں_انتہا) {
            __fp16 *     برآمد_dtile = برآمد_بنیاد + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_سمتیہ * pv_out0   = ((HVX_سمتیہ *) برآمد_dtile) + r1 / 2;
            HVX_سمتیہ * pv_out1   = pv_out0 + 16;
            *pv_out0 = Q6_V_vzero();
            *pv_out1 = Q6_V_vzero();
            جاری;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفیں_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفیں_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_سمتیہ * pv_in0 = (ثابت HVX_سمتیہ *) (عارضی_q_vtcm + offset0 * DK * حجم(__fp16));
        ثابت HVX_سمتیہ * pv_in1 = (r + 1 < g_صفیں_انتہا)
            ? (ثابت HVX_سمتیہ *) (عارضی_q_vtcm + offset1 * DK * حجم(__fp16))
            : تہی;

        HVX_سمتیہ     v0 = pv_in0[0];
        HVX_سمتیہ     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
        HVX_سمتیہ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        __fp16 *     برآمد_dtile = برآمد_بنیاد + 0 * HMX_FP16_TILE_N_ELMS * 2;
        HVX_سمتیہ * pv_out0   = ((HVX_سمتیہ *) برآمد_dtile) + r1 / 2;
        HVX_سمتیہ * pv_out1   = pv_out0 + 16;

        *pv_out0 = Q6_V_lo_W(vp);
        *pv_out1 = Q6_V_hi_W(vp);
    }
}

ساکن اندرونی خالی hmx_fa_q_prep_fp16_d2(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * عارضی_q_vtcm,
    size_t آغاز, size_t انتہا, size_t g_صفیں_انتہا,
    size_t DK, size_t G, size_t n_صفیں_q,
    ثابت ساخت fastdiv_اقدار * div_G, bool q_transposed
) {
    برائے (size_t r = آغاز; r < انتہا; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفیں;
        size_t   r1       = r % HMX_FP16_TILE_N_صفیں;
        __fp16 * برآمد_بنیاد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفیں * DK;

        اگر (r >= g_صفیں_انتہا) {
            برائے (uint32_t d = 0; d < 2; ++d) {
                __fp16 *     برآمد_dtile = برآمد_بنیاد + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_سمتیہ * pv_out0   = ((HVX_سمتیہ *) برآمد_dtile) + r1 / 2;
                HVX_سمتیہ * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            جاری;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفیں_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفیں_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_سمتیہ * pv_in0 = (ثابت HVX_سمتیہ *) (عارضی_q_vtcm + offset0 * DK * حجم(__fp16));
        ثابت HVX_سمتیہ * pv_in1 = (r + 1 < g_صفیں_انتہا)
            ? (ثابت HVX_سمتیہ *) (عارضی_q_vtcm + offset1 * DK * حجم(__fp16))
            : تہی;

        {
            HVX_سمتیہ     v0 = pv_in0[0];
            HVX_سمتیہ     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_سمتیہ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     برآمد_dtile = برآمد_بنیاد + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_سمتیہ * pv_out0   = ((HVX_سمتیہ *) برآمد_dtile) + r1 / 2;
            HVX_سمتیہ * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
        {
            HVX_سمتیہ     v0 = pv_in0[1];
            HVX_سمتیہ     v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_سمتیہ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     برآمد_dtile = برآمد_بنیاد + 1 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_سمتیہ * pv_out0   = ((HVX_سمتیہ *) برآمد_dtile) + r1 / 2;
            HVX_سمتیہ * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}

ساکن اندرونی خالی hmx_fa_q_prep_fp16(
    __fp16 * vtcm_q_tiles, ثابت uint8_t * عارضی_q_vtcm,
    size_t آغاز, size_t انتہا, size_t g_صفیں_انتہا,
    size_t DK, size_t G, size_t n_صفیں_q,
    ثابت ساخت fastdiv_اقدار * div_G, uint32_t d_حد_بندی, bool q_transposed
) {
    برائے (size_t r = آغاز; r < انتہا; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفیں;
        size_t   r1       = r % HMX_FP16_TILE_N_صفیں;
        __fp16 * برآمد_بنیاد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفیں * DK;

        اگر (r >= g_صفیں_انتہا) {
            برائے (uint32_t d = 0; d < d_حد_بندی; ++d) {
                __fp16 *     برآمد_dtile = برآمد_بنیاد + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_سمتیہ * pv_out0   = ((HVX_سمتیہ *) برآمد_dtile) + r1 / 2;
                HVX_سمتیہ * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            جاری;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفیں_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفیں_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_سمتیہ * pv_in0 = (ثابت HVX_سمتیہ *) (عارضی_q_vtcm + offset0 * DK * حجم(__fp16));
        ثابت HVX_سمتیہ * pv_in1 = (r + 1 < g_صفیں_انتہا)
            ? (ثابت HVX_سمتیہ *) (عارضی_q_vtcm + offset1 * DK * حجم(__fp16))
            : تہی;

        برائے (uint32_t d = 0; d < d_حد_بندی; ++d) {
            HVX_سمتیہ     v0 = pv_in0[d];
            HVX_سمتیہ     v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_سمتیہ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     برآمد_dtile = برآمد_بنیاد + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_سمتیہ * pv_out0   = ((HVX_سمتیہ *) برآمد_dtile) + r1 / 2;
            HVX_سمتیہ * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}


// Head-dim-padded Q-prep (f32). Used when DK is not a multiple of 64.
ساکن اندرونی خالی hmx_fa_q_prep_fp32_pad(__fp16 *                      vtcm_q_tiles,
                                          ثابت uint8_t *               عارضی_q_vtcm,
                                          size_t                        آغاز,
                                          size_t                        انتہا,
                                          size_t                        g_صفیں_انتہا,
                                          size_t                        dk_اندر,
                                          size_t                        dk_برآمد,
                                          size_t                        G,
                                          size_t                        n_صفیں_q,
                                          ثابت ساخت fastdiv_اقدار * div_G,
                                          bool                          q_transposed) {
    ثابت uint32_t n_برآمد_tiles = (uint32_t) (dk_برآمد / 32);
    برائے (size_t r = آغاز; r < انتہا; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفیں;
        size_t   r1       = r % HMX_FP16_TILE_N_صفیں;
        __fp16 * برآمد_بنیاد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفیں * dk_برآمد;

        اگر (r >= g_صفیں_انتہا) {
            برائے (uint32_t d = 0; d < n_برآمد_tiles; ++d) {
                ((HVX_سمتیہ *) (برآمد_بنیاد + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            جاری;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفیں_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفیں_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_UVector * pv_in0 = (ثابت HVX_UVector *) (عارضی_q_vtcm + offset0 * dk_اندر * حجم(اعشاری));
        ثابت HVX_UVector * pv_in1 = (r + 1 < g_صفیں_انتہا) ? (ثابت HVX_UVector *) (عارضی_q_vtcm + offset1 * dk_اندر * حجم(اعشاری)) : تہی;

        برائے (uint32_t d = 0; d < n_برآمد_tiles; ++d) {
            HVX_سمتیہ * برآمد_tile   = (HVX_سمتیہ *) (برآمد_بنیاد + d * HMX_FP16_TILE_N_ELMS);
            ثابت size_t بنیاد_lane  = (size_t) d * 32;
            ثابت size_t real_lanes = (بنیاد_lane < dk_اندر) ? hex_smin(32, dk_اندر - بنیاد_lane) : 0;

            اگر (real_lanes == 0) {
                برآمد_tile[r1 / 2] = Q6_V_vzero();
                جاری;
            }

            HVX_سمتیہ v0 = pv_in0[d];
            HVX_سمتیہ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            اگر (real_lanes < 32) {
                // Straddle tile: keep the first real_lanes floats, zero the padded tail so
                // the packed f16 lanes beyond DK are zero.
                ثابت HVX_سمتیہ_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * حجم(اعشاری)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            برآمد_tile[r1 / 2] = hvx_vec_f32_مقصد_f16_shuff(v0, v1);
        }
    }
}

// Head-dim-padded Q-prep (f16). Used when DK is not a multiple of 64.
ساکن اندرونی خالی hmx_fa_q_prep_fp16_pad(__fp16 *                      vtcm_q_tiles,
                                          ثابت uint8_t *               عارضی_q_vtcm,
                                          size_t                        آغاز,
                                          size_t                        انتہا,
                                          size_t                        g_صفیں_انتہا,
                                          size_t                        dk_اندر,
                                          size_t                        dk_برآمد,
                                          size_t                        G,
                                          size_t                        n_صفیں_q,
                                          ثابت ساخت fastdiv_اقدار * div_G,
                                          bool                          q_transposed) {
    ثابت uint32_t n_برآمد_جوڑے = (uint32_t) (dk_برآمد / 64);
    برائے (size_t r = آغاز; r < انتہا; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_صفیں;
        size_t   r1       = r % HMX_FP16_TILE_N_صفیں;
        __fp16 * برآمد_بنیاد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفیں * dk_برآمد;

        اگر (r >= g_صفیں_انتہا) {
            برائے (uint32_t d = 0; d < n_برآمد_جوڑے; ++d) {
                __fp16 *     برآمد_dtile = برآمد_بنیاد + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_سمتیہ * pv_out0   = ((HVX_سمتیہ *) برآمد_dtile) + r1 / 2;
                HVX_سمتیہ * pv_out1   = pv_out0 + 16;
                *pv_out0               = Q6_V_vzero();
                *pv_out1               = Q6_V_vzero();
            }
            جاری;
        }

        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت size_t offset0 = q_transposed ? (h_idx0 * n_صفیں_q + q_idx0) : (q_idx0 * G + h_idx0);
        ثابت size_t offset1 = q_transposed ? (h_idx1 * n_صفیں_q + q_idx1) : (q_idx1 * G + h_idx1);

        ثابت HVX_UVector * pv_in0 = (ثابت HVX_UVector *) (عارضی_q_vtcm + offset0 * dk_اندر * حجم(__fp16));
        ثابت HVX_UVector * pv_in1 = (r + 1 < g_صفیں_انتہا) ? (ثابت HVX_UVector *) (عارضی_q_vtcm + offset1 * dk_اندر * حجم(__fp16)) : تہی;

        برائے (uint32_t d = 0; d < n_برآمد_جوڑے; ++d) {
            __fp16 *     برآمد_dtile = برآمد_بنیاد + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_سمتیہ * pv_out0   = ((HVX_سمتیہ *) برآمد_dtile) + r1 / 2;
            HVX_سمتیہ * pv_out1   = pv_out0 + 16;

            ثابت size_t بنیاد_lane  = (size_t) d * 64;
            ثابت size_t real_lanes = (بنیاد_lane < dk_اندر) ? hex_smin(64, dk_اندر - بنیاد_lane) : 0;

            اگر (real_lanes == 0) {
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
                جاری;
            }

            HVX_سمتیہ v0 = pv_in0[d];
            HVX_سمتیہ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            اگر (real_lanes < 64) {
                ثابت HVX_سمتیہ_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * حجم(__fp16)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            HVX_سمتیہ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);
            *pv_out0          = Q6_V_lo_W(vp);
            *pv_out1          = Q6_V_hi_W(vp);
        }
    }
}

ساکن اندرونی خالی hmx_fa_q_prep_fallback(
    __fp16 * vtcm_q_tiles, uintptr_t q_ڈیٹا,
    size_t q_nb1, size_t q_nb2, size_t q_nb3,
    uint32_t q_آغاز, uint32_t kv_سر, uint32_t ib3,
    size_t آغاز, size_t انتہا, size_t n_صفیں_g,
    size_t G, size_t dk_اندر, size_t dk_برآمد, bool is_q_fp32,
    ثابت ساخت fastdiv_اقدار * div_G
) {
    برائے (size_t r = آغاز; r < انتہا; r += 2) {
        ثابت size_t q_idx0 = fastdiv(r + 0, div_G);
        ثابت size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ثابت size_t q_idx1 = fastdiv(r + 1, div_G);
        ثابت size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ثابت uint8_t * q_ptr0 = (r + 0 < n_صفیں_g) ? ((ثابت uint8_t *) q_ڈیٹا + (q_آغاز + q_idx0) * q_nb1 +
                                                      (kv_سر * G + h_idx0) * q_nb2 + ib3 * q_nb3) :
                                                      تہی;
        ثابت uint8_t * q_ptr1 = (r + 1 < n_صفیں_g) ? ((ثابت uint8_t *) q_ڈیٹا + (q_آغاز + q_idx1) * q_nb1 +
                                                      (kv_سر * G + h_idx1) * q_nb2 + ib3 * q_nb3) :
                                                      تہی;

        size_t   r0       = r / HMX_FP16_TILE_N_صفیں;
        size_t   r1       = r % HMX_FP16_TILE_N_صفیں;
        __fp16 * برآمد_بنیاد = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_صفیں * dk_برآمد;

        اگر (is_q_fp32) {
            ثابت HVX_UVector * pv_in0 = q_ptr0 ? (ثابت HVX_UVector *) q_ptr0 : تہی;
            ثابت HVX_UVector * pv_in1 = q_ptr1 ? (ثابت HVX_UVector *) q_ptr1 : تہی;

            برائے (uint32_t d = 0; d < dk_برآمد / 32; ++d) {
                HVX_سمتیہ * برآمد_tile   = (HVX_سمتیہ *) (برآمد_بنیاد + d * HMX_FP16_TILE_N_ELMS);
                ثابت size_t بنیاد_lane  = (size_t) d * 32;
                ثابت size_t real_lanes = (بنیاد_lane < dk_اندر) ? hex_smin(32, dk_اندر - بنیاد_lane) : 0;

                اگر (real_lanes == 0) {
                    برآمد_tile[r1 / 2] = Q6_V_vzero();
                    جاری;
                }
                HVX_سمتیہ v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_سمتیہ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                اگر (real_lanes < 32) {
                    ثابت HVX_سمتیہ_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * حجم(اعشاری)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                برآمد_tile[r1 / 2] = hvx_vec_f32_مقصد_f16_shuff(v0, v1);
            }
        } ورنہ {
            ثابت HVX_UVector * pv_in0 = q_ptr0 ? (ثابت HVX_UVector *) q_ptr0 : تہی;
            ثابت HVX_UVector * pv_in1 = q_ptr1 ? (ثابت HVX_UVector *) q_ptr1 : تہی;

            برائے (uint32_t d = 0; d < dk_برآمد / 64; ++d) {
                __fp16 *     برآمد_dtile = برآمد_بنیاد + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_سمتیہ * pv_out0   = ((HVX_سمتیہ *) برآمد_dtile) + r1 / 2;
                HVX_سمتیہ * pv_out1   = pv_out0 + 16;

                ثابت size_t بنیاد_lane  = (size_t) d * 64;
                ثابت size_t real_lanes = (بنیاد_lane < dk_اندر) ? hex_smin(64, dk_اندر - بنیاد_lane) : 0;

                اگر (real_lanes == 0) {
                    *pv_out0 = Q6_V_vzero();
                    *pv_out1 = Q6_V_vzero();
                    جاری;
                }
                HVX_سمتیہ v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_سمتیہ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                اگر (real_lanes < 64) {
                    ثابت HVX_سمتیہ_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * حجم(__fp16)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                HVX_سمتیہ_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);
                *pv_out0 = Q6_V_lo_W(vp);
                *pv_out1 = Q6_V_hi_W(vp);
            }
        }
    }
}

#اگر_ختم /* HMX_FA_KERNELS_H */
