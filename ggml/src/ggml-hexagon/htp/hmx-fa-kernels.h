#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HMX_FA_KERNELS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_FA_KERNELS_H

#ਅਨ੍ਤਰ੍ਭਾਵ <stdint.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stdbool.h>
#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hmx-utils.h"
#ਅਨ੍ਤਰ੍ਭਾਵ "hex-fastdiv.h"

// HMX-specific parameters, offsets and inner kernels for Flash Attention

// Scatter offsets for diagonal tile: entry[2i] = i*136, entry[2i+1] = i*136+6
// 136 = 4 * 32 + 8 = byte offset to diagonal in a 32x32 fp16 interleaved tile
ਸਥਿਰ ਅਚਲ int16_t d_tile_scatter_offsets[64] __attribute__((aligned(128))) = {
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

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_fa_qk_dot_tile(
    ਅਚਲ __fp16 * ਪੰਕਤੀ_tiles,
    ਅਚਲ __fp16 * ਖਾਨਾ_tiles,
    __fp16 *       ਬਾਹਰਲਾ_tile,
    size_t         n_dot_tiles
) {
    ਜੇ (n_dot_tiles == 2) {
        asm ਅਸਥਿਰ(
            HMX_ਲੱਦੋ_MPY_F16("%1", "%2", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(ਪੰਕਤੀ_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(ਖਾਨਾ_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(ਪੰਕਤੀ_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(ਖਾਨਾ_tiles + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } ਨਹੀਂਤਾਂ ਜੇ (n_dot_tiles == 4) {
        asm ਅਸਥਿਰ(
            HMX_ਲੱਦੋ_MPY_F16("%1", "%2", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%3", "%4", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%5", "%6", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(ਪੰਕਤੀ_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(ਖਾਨਾ_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(ਪੰਕਤੀ_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(ਖਾਨਾ_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(ਪੰਕਤੀ_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(ਖਾਨਾ_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(ਪੰਕਤੀ_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(ਖਾਨਾ_tiles + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } ਨਹੀਂਤਾਂ ਜੇ (n_dot_tiles == 8) {
        asm ਅਸਥਿਰ(
            HMX_ਲੱਦੋ_MPY_F16("%1", "%2", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%3", "%4", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%5", "%6", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%7", "%8", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%9", "%10", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%11", "%12", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%13", "%14", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(ਪੰਕਤੀ_tiles + 0 * HMX_FP16_TILE_N_ELMS), "r"(ਖਾਨਾ_tiles + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(ਪੰਕਤੀ_tiles + 1 * HMX_FP16_TILE_N_ELMS), "r"(ਖਾਨਾ_tiles + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(ਪੰਕਤੀ_tiles + 2 * HMX_FP16_TILE_N_ELMS), "r"(ਖਾਨਾ_tiles + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(ਪੰਕਤੀ_tiles + 3 * HMX_FP16_TILE_N_ELMS), "r"(ਖਾਨਾ_tiles + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(ਪੰਕਤੀ_tiles + 4 * HMX_FP16_TILE_N_ELMS), "r"(ਖਾਨਾ_tiles + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(ਪੰਕਤੀ_tiles + 5 * HMX_FP16_TILE_N_ELMS), "r"(ਖਾਨਾ_tiles + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(ਪੰਕਤੀ_tiles + 6 * HMX_FP16_TILE_N_ELMS), "r"(ਖਾਨਾ_tiles + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(ਪੰਕਤੀ_tiles + 7 * HMX_FP16_TILE_N_ELMS), "r"(ਖਾਨਾ_tiles + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } ਨਹੀਂਤਾਂ {
        ਲਈ (size_t k = 0; k < n_dot_tiles; ++k) {
            asm ਅਸਥਿਰ(
                HMX_ਲੱਦੋ_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(ਪੰਕਤੀ_tiles), "r"(ਖਾਨਾ_tiles)
            );
            ਪੰਕਤੀ_tiles += HMX_FP16_TILE_N_ELMS;
            ਖਾਨਾ_tiles += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm ਅਸਥਿਰ(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(ਬਾਹਰਲਾ_tile), "r"(0)
        : "memory"
    );
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_fa_o_update_tile(
    ਅਚਲ __fp16 * d_diag,
    ਅਚਲ __fp16 * o_rc,
    ਅਚਲ __fp16 * p_tile_ਵਿੱਚ,
    ਅਚਲ __fp16 * v_tile_ਵਿੱਚ,
    __fp16 *       o_tile_ਬਾਹਰਲਾ,
    size_t         n_ਖਾਨਾ_tiles
) {
    asm ਅਸਥਿਰ(
        HMX_ਲੱਦੋ_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    ਜੇ (n_ਖਾਨਾ_tiles == 2) {
        asm ਅਸਥਿਰ(
            HMX_ਲੱਦੋ_MPY_F16("%1", "%2", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%3", "%4", "%0")
            :
            : "r"(2047),
              "r"(p_tile_ਵਿੱਚ + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ਵਿੱਚ + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ਵਿੱਚ + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ਵਿੱਚ + 1 * HMX_FP16_TILE_N_ELMS)
        );
    } ਨਹੀਂਤਾਂ ਜੇ (n_ਖਾਨਾ_tiles == 4) {
        asm ਅਸਥਿਰ(
            HMX_ਲੱਦੋ_MPY_F16("%1", "%2", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%3", "%4", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%5", "%6", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%7", "%8", "%0")
            :
            : "r"(2047),
              "r"(p_tile_ਵਿੱਚ + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ਵਿੱਚ + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ਵਿੱਚ + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ਵਿੱਚ + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ਵਿੱਚ + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ਵਿੱਚ + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ਵਿੱਚ + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ਵਿੱਚ + 3 * HMX_FP16_TILE_N_ELMS)
        );
    } ਨਹੀਂਤਾਂ ਜੇ (n_ਖਾਨਾ_tiles == 8) {
        asm ਅਸਥਿਰ(
            HMX_ਲੱਦੋ_MPY_F16("%1", "%2", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%3", "%4", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%5", "%6", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%7", "%8", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%9", "%10", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%11", "%12", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%13", "%14", "%0")
            HMX_ਲੱਦੋ_MPY_F16("%15", "%16", "%0")
            :
            : "r"(2047),
              "r"(p_tile_ਵਿੱਚ + 0 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ਵਿੱਚ + 0 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ਵਿੱਚ + 1 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ਵਿੱਚ + 1 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ਵਿੱਚ + 2 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ਵਿੱਚ + 2 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ਵਿੱਚ + 3 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ਵਿੱਚ + 3 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ਵਿੱਚ + 4 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ਵਿੱਚ + 4 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ਵਿੱਚ + 5 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ਵਿੱਚ + 5 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ਵਿੱਚ + 6 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ਵਿੱਚ + 6 * HMX_FP16_TILE_N_ELMS),
              "r"(p_tile_ਵਿੱਚ + 7 * HMX_FP16_TILE_N_ELMS), "r"(v_tile_ਵਿੱਚ + 7 * HMX_FP16_TILE_N_ELMS)
        );
    } ਨਹੀਂਤਾਂ {
        ਲਈ (size_t k = 0; k < n_ਖਾਨਾ_tiles; ++k) {
            asm ਅਸਥਿਰ(
                HMX_ਲੱਦੋ_MPY_F16("%1", "%2", "%0")
                :
                : "r"(2047), "r"(p_tile_ਵਿੱਚ), "r"(v_tile_ਵਿੱਚ)
            );
            p_tile_ਵਿੱਚ += HMX_FP16_TILE_N_ELMS;
            v_tile_ਵਿੱਚ += HMX_FP16_TILE_N_ELMS;
        }
    }
    asm ਅਸਥਿਰ(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_tile_ਬਾਹਰਲਾ), "r"(0)
        : "memory"
    );
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_fa_o_norm_tile(
    ਅਚਲ __fp16 * d_diag,
    ਅਚਲ __fp16 * o_rc,
    __fp16 *       o_ਬਾਹਰਲਾ
) {
    asm ਅਸਥਿਰ(
        HMX_ਲੱਦੋ_MPY_F16("%1", "%2", "%0")
        :
        : "r"(2047), "r"(d_diag), "r"(o_rc)
    );
    asm ਅਸਥਿਰ(
        HMX_STORE_AFTER_F16("%0", "%1")
        :
        : "r"(o_ਬਾਹਰਲਾ), "r"(0)
        : "memory"
    );
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_fa_q_prep_fp32_d2(
    __fp16 * vtcm_q_tiles, ਅਚਲ uint8_t * ਆਰਜ਼ੀ_q_vtcm,
    size_t ਸ਼ੁਰੂ, size_t ਸਿਰਾ, size_t g_ਪੰਕਤੀਆਂ_ਸਿਰਾ,
    size_t DK, size_t G, size_t n_ਪੰਕਤੀਆਂ_q,
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_G, bool q_transposed
) {
    ਲਈ (size_t r = ਸ਼ੁਰੂ; r < ਸਿਰਾ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        size_t   r1       = r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        __fp16 * ਬਾਹਰਲਾ_ਅਧਾਰ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ * DK;

        ਜੇ (r >= g_ਪੰਕਤੀਆਂ_ਸਿਰਾ) {
            ((HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ_ਅਧਾਰ + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            ((HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ_ਅਧਾਰ + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            ਜਾਰੀ;
        }

        ਅਚਲ size_t q_idx0 = fastdiv(r + 0, div_G);
        ਅਚਲ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ਅਚਲ size_t q_idx1 = fastdiv(r + 1, div_G);
        ਅਚਲ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ਅਚਲ size_t offset0 = q_transposed ? (h_idx0 * n_ਪੰਕਤੀਆਂ_q + q_idx0) : (q_idx0 * G + h_idx0);
        ਅਚਲ size_t offset1 = q_transposed ? (h_idx1 * n_ਪੰਕਤੀਆਂ_q + q_idx1) : (q_idx1 * G + h_idx1);

        ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਆਰਜ਼ੀ_q_vtcm + offset0 * DK * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in1 = (r + 1 < g_ਪੰਕਤੀਆਂ_ਸਿਰਾ)
            ? (ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਆਰਜ਼ੀ_q_vtcm + offset1 * DK * ਆਕਾਰ(ਦਸ਼ਮਲਵ))
            : ਨਿਰਾ;

        {
            HVX_ਸਦਿਸ਼ v0 = pv_in0[0];
            HVX_ਸਦਿਸ਼ v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_ਸਦਿਸ਼ v_hf = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);
            ((HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ_ਅਧਾਰ + 0 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
        {
            HVX_ਸਦਿਸ਼ v0 = pv_in0[1];
            HVX_ਸਦਿਸ਼ v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_ਸਦਿਸ਼ v_hf = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);
            ((HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ_ਅਧਾਰ + 1 * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_fa_q_prep_fp32_d4(
    __fp16 * vtcm_q_tiles, ਅਚਲ uint8_t * ਆਰਜ਼ੀ_q_vtcm,
    size_t ਸ਼ੁਰੂ, size_t ਸਿਰਾ, size_t g_ਪੰਕਤੀਆਂ_ਸਿਰਾ,
    size_t DK, size_t G, size_t n_ਪੰਕਤੀਆਂ_q,
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_G, bool q_transposed
) {
    ਲਈ (size_t r = ਸ਼ੁਰੂ; r < ਸਿਰਾ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        size_t   r1       = r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        __fp16 * ਬਾਹਰਲਾ_ਅਧਾਰ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ * DK;

        ਜੇ (r >= g_ਪੰਕਤੀਆਂ_ਸਿਰਾ) {
            ਲਈ (uint32_t d = 0; d < 4; ++d) {
                ((HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            ਜਾਰੀ;
        }

        ਅਚਲ size_t q_idx0 = fastdiv(r + 0, div_G);
        ਅਚਲ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ਅਚਲ size_t q_idx1 = fastdiv(r + 1, div_G);
        ਅਚਲ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ਅਚਲ size_t offset0 = q_transposed ? (h_idx0 * n_ਪੰਕਤੀਆਂ_q + q_idx0) : (q_idx0 * G + h_idx0);
        ਅਚਲ size_t offset1 = q_transposed ? (h_idx1 * n_ਪੰਕਤੀਆਂ_q + q_idx1) : (q_idx1 * G + h_idx1);

        ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਆਰਜ਼ੀ_q_vtcm + offset0 * DK * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in1 = (r + 1 < g_ਪੰਕਤੀਆਂ_ਸਿਰਾ)
            ? (ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਆਰਜ਼ੀ_q_vtcm + offset1 * DK * ਆਕਾਰ(ਦਸ਼ਮਲਵ))
            : ਨਿਰਾ;

        ਲਈ (uint32_t d = 0; d < 4; ++d) {
            HVX_ਸਦਿਸ਼ v0 = pv_in0[d];
            HVX_ਸਦਿਸ਼ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_ਸਦਿਸ਼ v_hf = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);
            ((HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = v_hf;
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_fa_q_prep_fp32(
    __fp16 * vtcm_q_tiles, ਅਚਲ uint8_t * ਆਰਜ਼ੀ_q_vtcm,
    size_t ਸ਼ੁਰੂ, size_t ਸਿਰਾ, size_t g_ਪੰਕਤੀਆਂ_ਸਿਰਾ,
    size_t DK, size_t G, size_t n_ਪੰਕਤੀਆਂ_q,
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_G, uint32_t d_ਹੱਦਬੰਦੀ, bool q_transposed
) {
    ਲਈ (size_t r = ਸ਼ੁਰੂ; r < ਸਿਰਾ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        size_t   r1       = r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        __fp16 * ਬਾਹਰਲਾ_ਅਧਾਰ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ * DK;

        ਜੇ (r >= g_ਪੰਕਤੀਆਂ_ਸਿਰਾ) {
            ਲਈ (uint32_t d = 0; d < d_ਹੱਦਬੰਦੀ; ++d) {
                ((HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            ਜਾਰੀ;
        }

        ਅਚਲ size_t q_idx0 = fastdiv(r + 0, div_G);
        ਅਚਲ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ਅਚਲ size_t q_idx1 = fastdiv(r + 1, div_G);
        ਅਚਲ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ਅਚਲ size_t offset0 = q_transposed ? (h_idx0 * n_ਪੰਕਤੀਆਂ_q + q_idx0) : (q_idx0 * G + h_idx0);
        ਅਚਲ size_t offset1 = q_transposed ? (h_idx1 * n_ਪੰਕਤੀਆਂ_q + q_idx1) : (q_idx1 * G + h_idx1);

        ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਆਰਜ਼ੀ_q_vtcm + offset0 * DK * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in1 = (r + 1 < g_ਪੰਕਤੀਆਂ_ਸਿਰਾ)
            ? (ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਆਰਜ਼ੀ_q_vtcm + offset1 * DK * ਆਕਾਰ(ਦਸ਼ਮਲਵ))
            : ਨਿਰਾ;

        ਲਈ (uint32_t d = 0; d < d_ਹੱਦਬੰਦੀ; ++d) {
            HVX_ਸਦਿਸ਼ v0   = pv_in0[d];
            HVX_ਸਦਿਸ਼ v1   = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_ਸਦਿਸ਼ v_hf = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);

            HVX_ਸਦਿਸ਼ * ਬਾਹਰਲਾ_tile = (HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS);
            ਬਾਹਰਲਾ_tile[r1 / 2]      = v_hf;
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_fa_q_prep_fp16_d1(
    __fp16 * vtcm_q_tiles, ਅਚਲ uint8_t * ਆਰਜ਼ੀ_q_vtcm,
    size_t ਸ਼ੁਰੂ, size_t ਸਿਰਾ, size_t g_ਪੰਕਤੀਆਂ_ਸਿਰਾ,
    size_t DK, size_t G, size_t n_ਪੰਕਤੀਆਂ_q,
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_G, bool q_transposed
) {
    ਲਈ (size_t r = ਸ਼ੁਰੂ; r < ਸਿਰਾ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        size_t   r1       = r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        __fp16 * ਬਾਹਰਲਾ_ਅਧਾਰ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ * DK;

        ਜੇ (r >= g_ਪੰਕਤੀਆਂ_ਸਿਰਾ) {
            __fp16 *     ਬਾਹਰਲਾ_dtile = ਬਾਹਰਲਾ_ਅਧਾਰ + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ਸਦਿਸ਼ * pv_out0   = ((HVX_ਸਦਿਸ਼ *) ਬਾਹਰਲਾ_dtile) + r1 / 2;
            HVX_ਸਦਿਸ਼ * pv_out1   = pv_out0 + 16;
            *pv_out0 = Q6_V_vzero();
            *pv_out1 = Q6_V_vzero();
            ਜਾਰੀ;
        }

        ਅਚਲ size_t q_idx0 = fastdiv(r + 0, div_G);
        ਅਚਲ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ਅਚਲ size_t q_idx1 = fastdiv(r + 1, div_G);
        ਅਚਲ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ਅਚਲ size_t offset0 = q_transposed ? (h_idx0 * n_ਪੰਕਤੀਆਂ_q + q_idx0) : (q_idx0 * G + h_idx0);
        ਅਚਲ size_t offset1 = q_transposed ? (h_idx1 * n_ਪੰਕਤੀਆਂ_q + q_idx1) : (q_idx1 * G + h_idx1);

        ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਆਰਜ਼ੀ_q_vtcm + offset0 * DK * ਆਕਾਰ(__fp16));
        ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in1 = (r + 1 < g_ਪੰਕਤੀਆਂ_ਸਿਰਾ)
            ? (ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਆਰਜ਼ੀ_q_vtcm + offset1 * DK * ਆਕਾਰ(__fp16))
            : ਨਿਰਾ;

        HVX_ਸਦਿਸ਼     v0 = pv_in0[0];
        HVX_ਸਦਿਸ਼     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
        HVX_ਸਦਿਸ਼_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

        __fp16 *     ਬਾਹਰਲਾ_dtile = ਬਾਹਰਲਾ_ਅਧਾਰ + 0 * HMX_FP16_TILE_N_ELMS * 2;
        HVX_ਸਦਿਸ਼ * pv_out0   = ((HVX_ਸਦਿਸ਼ *) ਬਾਹਰਲਾ_dtile) + r1 / 2;
        HVX_ਸਦਿਸ਼ * pv_out1   = pv_out0 + 16;

        *pv_out0 = Q6_V_lo_W(vp);
        *pv_out1 = Q6_V_hi_W(vp);
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_fa_q_prep_fp16_d2(
    __fp16 * vtcm_q_tiles, ਅਚਲ uint8_t * ਆਰਜ਼ੀ_q_vtcm,
    size_t ਸ਼ੁਰੂ, size_t ਸਿਰਾ, size_t g_ਪੰਕਤੀਆਂ_ਸਿਰਾ,
    size_t DK, size_t G, size_t n_ਪੰਕਤੀਆਂ_q,
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_G, bool q_transposed
) {
    ਲਈ (size_t r = ਸ਼ੁਰੂ; r < ਸਿਰਾ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        size_t   r1       = r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        __fp16 * ਬਾਹਰਲਾ_ਅਧਾਰ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ * DK;

        ਜੇ (r >= g_ਪੰਕਤੀਆਂ_ਸਿਰਾ) {
            ਲਈ (uint32_t d = 0; d < 2; ++d) {
                __fp16 *     ਬਾਹਰਲਾ_dtile = ਬਾਹਰਲਾ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ਸਦਿਸ਼ * pv_out0   = ((HVX_ਸਦਿਸ਼ *) ਬਾਹਰਲਾ_dtile) + r1 / 2;
                HVX_ਸਦਿਸ਼ * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            ਜਾਰੀ;
        }

        ਅਚਲ size_t q_idx0 = fastdiv(r + 0, div_G);
        ਅਚਲ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ਅਚਲ size_t q_idx1 = fastdiv(r + 1, div_G);
        ਅਚਲ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ਅਚਲ size_t offset0 = q_transposed ? (h_idx0 * n_ਪੰਕਤੀਆਂ_q + q_idx0) : (q_idx0 * G + h_idx0);
        ਅਚਲ size_t offset1 = q_transposed ? (h_idx1 * n_ਪੰਕਤੀਆਂ_q + q_idx1) : (q_idx1 * G + h_idx1);

        ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਆਰਜ਼ੀ_q_vtcm + offset0 * DK * ਆਕਾਰ(__fp16));
        ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in1 = (r + 1 < g_ਪੰਕਤੀਆਂ_ਸਿਰਾ)
            ? (ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਆਰਜ਼ੀ_q_vtcm + offset1 * DK * ਆਕਾਰ(__fp16))
            : ਨਿਰਾ;

        {
            HVX_ਸਦਿਸ਼     v0 = pv_in0[0];
            HVX_ਸਦਿਸ਼     v1 = pv_in1 ? pv_in1[0] : Q6_V_vzero();
            HVX_ਸਦਿਸ਼_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     ਬਾਹਰਲਾ_dtile = ਬਾਹਰਲਾ_ਅਧਾਰ + 0 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ਸਦਿਸ਼ * pv_out0   = ((HVX_ਸਦਿਸ਼ *) ਬਾਹਰਲਾ_dtile) + r1 / 2;
            HVX_ਸਦਿਸ਼ * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
        {
            HVX_ਸਦਿਸ਼     v0 = pv_in0[1];
            HVX_ਸਦਿਸ਼     v1 = pv_in1 ? pv_in1[1] : Q6_V_vzero();
            HVX_ਸਦਿਸ਼_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     ਬਾਹਰਲਾ_dtile = ਬਾਹਰਲਾ_ਅਧਾਰ + 1 * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ਸਦਿਸ਼ * pv_out0   = ((HVX_ਸਦਿਸ਼ *) ਬਾਹਰਲਾ_dtile) + r1 / 2;
            HVX_ਸਦਿਸ਼ * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_fa_q_prep_fp16(
    __fp16 * vtcm_q_tiles, ਅਚਲ uint8_t * ਆਰਜ਼ੀ_q_vtcm,
    size_t ਸ਼ੁਰੂ, size_t ਸਿਰਾ, size_t g_ਪੰਕਤੀਆਂ_ਸਿਰਾ,
    size_t DK, size_t G, size_t n_ਪੰਕਤੀਆਂ_q,
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_G, uint32_t d_ਹੱਦਬੰਦੀ, bool q_transposed
) {
    ਲਈ (size_t r = ਸ਼ੁਰੂ; r < ਸਿਰਾ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        size_t   r1       = r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        __fp16 * ਬਾਹਰਲਾ_ਅਧਾਰ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ * DK;

        ਜੇ (r >= g_ਪੰਕਤੀਆਂ_ਸਿਰਾ) {
            ਲਈ (uint32_t d = 0; d < d_ਹੱਦਬੰਦੀ; ++d) {
                __fp16 *     ਬਾਹਰਲਾ_dtile = ਬਾਹਰਲਾ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ਸਦਿਸ਼ * pv_out0   = ((HVX_ਸਦਿਸ਼ *) ਬਾਹਰਲਾ_dtile) + r1 / 2;
                HVX_ਸਦਿਸ਼ * pv_out1   = pv_out0 + 16;
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
            }
            ਜਾਰੀ;
        }

        ਅਚਲ size_t q_idx0 = fastdiv(r + 0, div_G);
        ਅਚਲ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ਅਚਲ size_t q_idx1 = fastdiv(r + 1, div_G);
        ਅਚਲ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ਅਚਲ size_t offset0 = q_transposed ? (h_idx0 * n_ਪੰਕਤੀਆਂ_q + q_idx0) : (q_idx0 * G + h_idx0);
        ਅਚਲ size_t offset1 = q_transposed ? (h_idx1 * n_ਪੰਕਤੀਆਂ_q + q_idx1) : (q_idx1 * G + h_idx1);

        ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਆਰਜ਼ੀ_q_vtcm + offset0 * DK * ਆਕਾਰ(__fp16));
        ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in1 = (r + 1 < g_ਪੰਕਤੀਆਂ_ਸਿਰਾ)
            ? (ਅਚਲ HVX_ਸਦਿਸ਼ *) (ਆਰਜ਼ੀ_q_vtcm + offset1 * DK * ਆਕਾਰ(__fp16))
            : ਨਿਰਾ;

        ਲਈ (uint32_t d = 0; d < d_ਹੱਦਬੰਦੀ; ++d) {
            HVX_ਸਦਿਸ਼     v0 = pv_in0[d];
            HVX_ਸਦਿਸ਼     v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            HVX_ਸਦਿਸ਼_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);

            __fp16 *     ਬਾਹਰਲਾ_dtile = ਬਾਹਰਲਾ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ਸਦਿਸ਼ * pv_out0   = ((HVX_ਸਦਿਸ਼ *) ਬਾਹਰਲਾ_dtile) + r1 / 2;
            HVX_ਸਦਿਸ਼ * pv_out1   = pv_out0 + 16;

            *pv_out0 = Q6_V_lo_W(vp);
            *pv_out1 = Q6_V_hi_W(vp);
        }
    }
}


// Head-dim-padded Q-prep (f32). Used when DK is not a multiple of 64.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_fa_q_prep_fp32_pad(__fp16 *                      vtcm_q_tiles,
                                          ਅਚਲ uint8_t *               ਆਰਜ਼ੀ_q_vtcm,
                                          size_t                        ਸ਼ੁਰੂ,
                                          size_t                        ਸਿਰਾ,
                                          size_t                        g_ਪੰਕਤੀਆਂ_ਸਿਰਾ,
                                          size_t                        dk_ਵਿੱਚ,
                                          size_t                        dk_ਬਾਹਰਲਾ,
                                          size_t                        G,
                                          size_t                        n_ਪੰਕਤੀਆਂ_q,
                                          ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_G,
                                          bool                          q_transposed) {
    ਅਚਲ uint32_t n_ਬਾਹਰਲਾ_tiles = (uint32_t) (dk_ਬਾਹਰਲਾ / 32);
    ਲਈ (size_t r = ਸ਼ੁਰੂ; r < ਸਿਰਾ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        size_t   r1       = r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        __fp16 * ਬਾਹਰਲਾ_ਅਧਾਰ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ * dk_ਬਾਹਰਲਾ;

        ਜੇ (r >= g_ਪੰਕਤੀਆਂ_ਸਿਰਾ) {
            ਲਈ (uint32_t d = 0; d < n_ਬਾਹਰਲਾ_tiles; ++d) {
                ((HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS))[r1 / 2] = Q6_V_vzero();
            }
            ਜਾਰੀ;
        }

        ਅਚਲ size_t q_idx0 = fastdiv(r + 0, div_G);
        ਅਚਲ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ਅਚਲ size_t q_idx1 = fastdiv(r + 1, div_G);
        ਅਚਲ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ਅਚਲ size_t offset0 = q_transposed ? (h_idx0 * n_ਪੰਕਤੀਆਂ_q + q_idx0) : (q_idx0 * G + h_idx0);
        ਅਚਲ size_t offset1 = q_transposed ? (h_idx1 * n_ਪੰਕਤੀਆਂ_q + q_idx1) : (q_idx1 * G + h_idx1);

        ਅਚਲ HVX_UVector * pv_in0 = (ਅਚਲ HVX_UVector *) (ਆਰਜ਼ੀ_q_vtcm + offset0 * dk_ਵਿੱਚ * ਆਕਾਰ(ਦਸ਼ਮਲਵ));
        ਅਚਲ HVX_UVector * pv_in1 = (r + 1 < g_ਪੰਕਤੀਆਂ_ਸਿਰਾ) ? (ਅਚਲ HVX_UVector *) (ਆਰਜ਼ੀ_q_vtcm + offset1 * dk_ਵਿੱਚ * ਆਕਾਰ(ਦਸ਼ਮਲਵ)) : ਨਿਰਾ;

        ਲਈ (uint32_t d = 0; d < n_ਬਾਹਰਲਾ_tiles; ++d) {
            HVX_ਸਦਿਸ਼ * ਬਾਹਰਲਾ_tile   = (HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS);
            ਅਚਲ size_t ਅਧਾਰ_lane  = (size_t) d * 32;
            ਅਚਲ size_t real_lanes = (ਅਧਾਰ_lane < dk_ਵਿੱਚ) ? hex_smin(32, dk_ਵਿੱਚ - ਅਧਾਰ_lane) : 0;

            ਜੇ (real_lanes == 0) {
                ਬਾਹਰਲਾ_tile[r1 / 2] = Q6_V_vzero();
                ਜਾਰੀ;
            }

            HVX_ਸਦਿਸ਼ v0 = pv_in0[d];
            HVX_ਸਦਿਸ਼ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            ਜੇ (real_lanes < 32) {
                // Straddle tile: keep the first real_lanes floats, zero the padded tail so
                // the packed f16 lanes beyond DK are zero.
                ਅਚਲ HVX_ਸਦਿਸ਼_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * ਆਕਾਰ(ਦਸ਼ਮਲਵ)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            ਬਾਹਰਲਾ_tile[r1 / 2] = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);
        }
    }
}

// Head-dim-padded Q-prep (f16). Used when DK is not a multiple of 64.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_fa_q_prep_fp16_pad(__fp16 *                      vtcm_q_tiles,
                                          ਅਚਲ uint8_t *               ਆਰਜ਼ੀ_q_vtcm,
                                          size_t                        ਸ਼ੁਰੂ,
                                          size_t                        ਸਿਰਾ,
                                          size_t                        g_ਪੰਕਤੀਆਂ_ਸਿਰਾ,
                                          size_t                        dk_ਵਿੱਚ,
                                          size_t                        dk_ਬਾਹਰਲਾ,
                                          size_t                        G,
                                          size_t                        n_ਪੰਕਤੀਆਂ_q,
                                          ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_G,
                                          bool                          q_transposed) {
    ਅਚਲ uint32_t n_ਬਾਹਰਲਾ_ਜੋੜੇ = (uint32_t) (dk_ਬਾਹਰਲਾ / 64);
    ਲਈ (size_t r = ਸ਼ੁਰੂ; r < ਸਿਰਾ; r += 2) {
        size_t   r0       = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        size_t   r1       = r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        __fp16 * ਬਾਹਰਲਾ_ਅਧਾਰ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ * dk_ਬਾਹਰਲਾ;

        ਜੇ (r >= g_ਪੰਕਤੀਆਂ_ਸਿਰਾ) {
            ਲਈ (uint32_t d = 0; d < n_ਬਾਹਰਲਾ_ਜੋੜੇ; ++d) {
                __fp16 *     ਬਾਹਰਲਾ_dtile = ਬਾਹਰਲਾ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ਸਦਿਸ਼ * pv_out0   = ((HVX_ਸਦਿਸ਼ *) ਬਾਹਰਲਾ_dtile) + r1 / 2;
                HVX_ਸਦਿਸ਼ * pv_out1   = pv_out0 + 16;
                *pv_out0               = Q6_V_vzero();
                *pv_out1               = Q6_V_vzero();
            }
            ਜਾਰੀ;
        }

        ਅਚਲ size_t q_idx0 = fastdiv(r + 0, div_G);
        ਅਚਲ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ਅਚਲ size_t q_idx1 = fastdiv(r + 1, div_G);
        ਅਚਲ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ਅਚਲ size_t offset0 = q_transposed ? (h_idx0 * n_ਪੰਕਤੀਆਂ_q + q_idx0) : (q_idx0 * G + h_idx0);
        ਅਚਲ size_t offset1 = q_transposed ? (h_idx1 * n_ਪੰਕਤੀਆਂ_q + q_idx1) : (q_idx1 * G + h_idx1);

        ਅਚਲ HVX_UVector * pv_in0 = (ਅਚਲ HVX_UVector *) (ਆਰਜ਼ੀ_q_vtcm + offset0 * dk_ਵਿੱਚ * ਆਕਾਰ(__fp16));
        ਅਚਲ HVX_UVector * pv_in1 = (r + 1 < g_ਪੰਕਤੀਆਂ_ਸਿਰਾ) ? (ਅਚਲ HVX_UVector *) (ਆਰਜ਼ੀ_q_vtcm + offset1 * dk_ਵਿੱਚ * ਆਕਾਰ(__fp16)) : ਨਿਰਾ;

        ਲਈ (uint32_t d = 0; d < n_ਬਾਹਰਲਾ_ਜੋੜੇ; ++d) {
            __fp16 *     ਬਾਹਰਲਾ_dtile = ਬਾਹਰਲਾ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS * 2;
            HVX_ਸਦਿਸ਼ * pv_out0   = ((HVX_ਸਦਿਸ਼ *) ਬਾਹਰਲਾ_dtile) + r1 / 2;
            HVX_ਸਦਿਸ਼ * pv_out1   = pv_out0 + 16;

            ਅਚਲ size_t ਅਧਾਰ_lane  = (size_t) d * 64;
            ਅਚਲ size_t real_lanes = (ਅਧਾਰ_lane < dk_ਵਿੱਚ) ? hex_smin(64, dk_ਵਿੱਚ - ਅਧਾਰ_lane) : 0;

            ਜੇ (real_lanes == 0) {
                *pv_out0 = Q6_V_vzero();
                *pv_out1 = Q6_V_vzero();
                ਜਾਰੀ;
            }

            HVX_ਸਦਿਸ਼ v0 = pv_in0[d];
            HVX_ਸਦਿਸ਼ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
            ਜੇ (real_lanes < 64) {
                ਅਚਲ HVX_ਸਦਿਸ਼_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * ਆਕਾਰ(__fp16)));
                v0                        = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                v1                        = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
            }
            HVX_ਸਦਿਸ਼_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);
            *pv_out0          = Q6_V_lo_W(vp);
            *pv_out1          = Q6_V_hi_W(vp);
        }
    }
}

ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_fa_q_prep_fallback(
    __fp16 * vtcm_q_tiles, uintptr_t q_ਅੰਕੜੇ,
    size_t q_nb1, size_t q_nb2, size_t q_nb3,
    uint32_t q_ਸ਼ੁਰੂ, uint32_t kv_ਸਿਰ, uint32_t ib3,
    size_t ਸ਼ੁਰੂ, size_t ਸਿਰਾ, size_t n_ਪੰਕਤੀਆਂ_g,
    size_t G, size_t dk_ਵਿੱਚ, size_t dk_ਬਾਹਰਲਾ, bool is_q_fp32,
    ਅਚਲ ਬਣਤਰ fastdiv_ਮੁੱਲਾਂ * div_G
) {
    ਲਈ (size_t r = ਸ਼ੁਰੂ; r < ਸਿਰਾ; r += 2) {
        ਅਚਲ size_t q_idx0 = fastdiv(r + 0, div_G);
        ਅਚਲ size_t h_idx0 = fastmodulo(r + 0, G, div_G);
        ਅਚਲ size_t q_idx1 = fastdiv(r + 1, div_G);
        ਅਚਲ size_t h_idx1 = fastmodulo(r + 1, G, div_G);

        ਅਚਲ uint8_t * q_ptr0 = (r + 0 < n_ਪੰਕਤੀਆਂ_g) ? ((ਅਚਲ uint8_t *) q_ਅੰਕੜੇ + (q_ਸ਼ੁਰੂ + q_idx0) * q_nb1 +
                                                      (kv_ਸਿਰ * G + h_idx0) * q_nb2 + ib3 * q_nb3) :
                                                      ਨਿਰਾ;
        ਅਚਲ uint8_t * q_ptr1 = (r + 1 < n_ਪੰਕਤੀਆਂ_g) ? ((ਅਚਲ uint8_t *) q_ਅੰਕੜੇ + (q_ਸ਼ੁਰੂ + q_idx1) * q_nb1 +
                                                      (kv_ਸਿਰ * G + h_idx1) * q_nb2 + ib3 * q_nb3) :
                                                      ਨਿਰਾ;

        size_t   r0       = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        size_t   r1       = r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        __fp16 * ਬਾਹਰਲਾ_ਅਧਾਰ = vtcm_q_tiles + r0 * HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ * dk_ਬਾਹਰਲਾ;

        ਜੇ (is_q_fp32) {
            ਅਚਲ HVX_UVector * pv_in0 = q_ptr0 ? (ਅਚਲ HVX_UVector *) q_ptr0 : ਨਿਰਾ;
            ਅਚਲ HVX_UVector * pv_in1 = q_ptr1 ? (ਅਚਲ HVX_UVector *) q_ptr1 : ਨਿਰਾ;

            ਲਈ (uint32_t d = 0; d < dk_ਬਾਹਰਲਾ / 32; ++d) {
                HVX_ਸਦਿਸ਼ * ਬਾਹਰਲਾ_tile   = (HVX_ਸਦਿਸ਼ *) (ਬਾਹਰਲਾ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS);
                ਅਚਲ size_t ਅਧਾਰ_lane  = (size_t) d * 32;
                ਅਚਲ size_t real_lanes = (ਅਧਾਰ_lane < dk_ਵਿੱਚ) ? hex_smin(32, dk_ਵਿੱਚ - ਅਧਾਰ_lane) : 0;

                ਜੇ (real_lanes == 0) {
                    ਬਾਹਰਲਾ_tile[r1 / 2] = Q6_V_vzero();
                    ਜਾਰੀ;
                }
                HVX_ਸਦਿਸ਼ v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_ਸਦਿਸ਼ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                ਜੇ (real_lanes < 32) {
                    ਅਚਲ HVX_ਸਦਿਸ਼_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * ਆਕਾਰ(ਦਸ਼ਮਲਵ)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                ਬਾਹਰਲਾ_tile[r1 / 2] = hvx_vec_f32_ਮੰਜ਼ਿਲ_f16_shuff(v0, v1);
            }
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ HVX_UVector * pv_in0 = q_ptr0 ? (ਅਚਲ HVX_UVector *) q_ptr0 : ਨਿਰਾ;
            ਅਚਲ HVX_UVector * pv_in1 = q_ptr1 ? (ਅਚਲ HVX_UVector *) q_ptr1 : ਨਿਰਾ;

            ਲਈ (uint32_t d = 0; d < dk_ਬਾਹਰਲਾ / 64; ++d) {
                __fp16 *     ਬਾਹਰਲਾ_dtile = ਬਾਹਰਲਾ_ਅਧਾਰ + d * HMX_FP16_TILE_N_ELMS * 2;
                HVX_ਸਦਿਸ਼ * pv_out0   = ((HVX_ਸਦਿਸ਼ *) ਬਾਹਰਲਾ_dtile) + r1 / 2;
                HVX_ਸਦਿਸ਼ * pv_out1   = pv_out0 + 16;

                ਅਚਲ size_t ਅਧਾਰ_lane  = (size_t) d * 64;
                ਅਚਲ size_t real_lanes = (ਅਧਾਰ_lane < dk_ਵਿੱਚ) ? hex_smin(64, dk_ਵਿੱਚ - ਅਧਾਰ_lane) : 0;

                ਜੇ (real_lanes == 0) {
                    *pv_out0 = Q6_V_vzero();
                    *pv_out1 = Q6_V_vzero();
                    ਜਾਰੀ;
                }
                HVX_ਸਦਿਸ਼ v0 = pv_in0 ? pv_in0[d] : Q6_V_vzero();
                HVX_ਸਦਿਸ਼ v1 = pv_in1 ? pv_in1[d] : Q6_V_vzero();
                ਜੇ (real_lanes < 64) {
                    ਅਚਲ HVX_ਸਦਿਸ਼_Pred keep = Q6_Q_vsetq_R((uint32_t) (real_lanes * ਆਕਾਰ(__fp16)));
                    v0 = Q6_V_vmux_QVV(keep, v0, Q6_V_vzero());
                    v1 = Q6_V_vmux_QVV(keep, v1, Q6_V_vzero());
                }
                HVX_ਸਦਿਸ਼_Pair vp = Q6_W_vshuff_VVR(v1, v0, -2);
                *pv_out0 = Q6_V_lo_W(vp);
                *pv_out1 = Q6_V_hi_W(vp);
            }
        }
    }
}

#ਜੇ_ਅੰਤ /* HMX_FA_KERNELS_H */
