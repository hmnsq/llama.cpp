// HMX tile-level inline helpers (FP16 32x32 tile operations).
// Ported from htp-ops-lib/include/dsp/hmx_utils.h. (https://github.com/haozixu/htp-ops-lib)

#ವ್ಯಾಖ್ಯಾನಿತವಾಗದಿದ್ದರೆ HMX_UTILS_H
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_UTILS_H

#ಅನ್ತರ್ಭಾವ "hvx-base.h"

#ಅನ್ತರ್ಭಾವ <assert.h>
#ಅನ್ತರ್ಭಾವ <hexagon_types.h>
#ಅನ್ತರ್ಭಾವ <stddef.h>

#ವ್ಯಾಖ್ಯಾನಿಸು HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು 32
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_FP16_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ 32
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_FP16_TILE_N_ELMS 1024
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_FP16_TILE_ಗಾತ್ರ   2048

// Initialise aligned 256-byte area with scale vector + zero padding.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_init_ಸ್ತಂಭ_scales(ಶೂನ್ಯ *ಹೊರಹೋಗು_scales, HVX_ಸದಿಶ v_scale) {
    ಅಸ್ಥಿರ HVX_ಸದಿಶ *pv = (HVX_ಸದಿಶ *) ಹೊರಹೋಗು_scales;
    pv[0] = v_scale;
    pv[1] = Q6_V_vzero();
}

// --- Shared scatter offsets and interleave helper ---

// vscatter offsets for fused dequant+transpose: write K-values directly to [K][N] tile.
// word[i] = i*128 maps K-row-pair i to byte offset i*128.
// Column offset (n*4) is added at runtime.  Entries 0..15 cover one tile (region 2047);
// entries 16..31 cover the next adjacent tile (region 4095) - pick region size at the
// call site to scatter into one tile (masked) or two contiguous tiles (unmasked).
ಸ್ಥಿರ ಸ್ಥಿರಾಂಕ int32_t hmx_transpose_scatter_offsets[32] __attribute__((aligned(VLEN))) = {
    0 * 128,  1 * 128,  2 * 128,  3 * 128,  4 * 128,  5 * 128,  6 * 128,  7 * 128,  8 * 128,  9 * 128,  10 * 128,
    11 * 128, 12 * 128, 13 * 128, 14 * 128, 15 * 128, 16 * 128, 17 * 128, 18 * 128, 19 * 128, 20 * 128, 21 * 128,
    22 * 128, 23 * 128, 24 * 128, 25 * 128, 26 * 128, 27 * 128, 28 * 128, 29 * 128, 30 * 128, 31 * 128,
};

// Scatter row-major FP16 data (in VTCM scratch) into transposed [K][N] tiles.
// vtcm_src: [n_cols][src_stride] row-major fp16 (only first k elements per row are used)
// vtcm_dst: [n_col_tiles][n_k_tiles][HMX_FP16_TILE_N_ELMS] tile-major interleaved fp16
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_cols.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_interleave_ಅಡ್ಡಸಾಲುಗಳು_ಗಮ್ಯ_tiles(__fp16 * ಸೀಮಿತ vtcm_dst,
                                            ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ vtcm_src,
                                            uint32_t n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ,
                                            uint32_t k,
                                            size_t src_stride,
                                            uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು,
                                            uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
    ಪ್ರತಿಜ್ಞಾ(k % HMX_FP16_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ == 0);

    ಸ್ಥಿರಾಂಕ uint32_t       n_k_tiles     = k / HMX_FP16_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ     v_scat_ತಳ   = hvx_vmem(hmx_transpose_scatter_offsets);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ     v_scat_ಹೆಜ್ಜೆ   = Q6_V_vsplat_R(4);
    ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ_Pred q_mask64      = Q6_Q_vsetq_R(64);
    // Each hvx_vmemu load brings 64 fp16 = 128 bytes covering 2 adjacent K-tiles.
    // When n_k_tiles is even, scatter into 2 K-tiles per call (region 4095, no mask)
    // using the upper half of hmx_transpose_scatter_offsets.  Tail one K-tile (when
    // n_k_tiles is odd) falls back to single-tile masked scatter.
    ಸ್ಥಿರಾಂಕ bool           pair_scatter  = (n_k_tiles & 1) == 0;
    ಸ್ಥಿರಾಂಕ size_t         pair_region   = (size_t) (2 * HMX_FP16_TILE_ಗಾತ್ರ - 1);
    ಸ್ಥಿರಾಂಕ size_t         single_region = (size_t) (HMX_FP16_TILE_ಗಾತ್ರ - 1);
    __builtin_assume(k > 0);
    __builtin_assume(ಕೊನೆ_ಅಡ್ಡಸಾಲು > ಆರಂಭ_ಅಡ್ಡಸಾಲು);

    ಆದರೆ (pair_scatter) {
        // Step c by 64 fp16 (two K-tiles per scatter), advance dst by 2 tiles per iter.
        ಸ್ಥಿರಾಂಕ uint32_t c_ಹೆಜ್ಜೆ      = 2 * HMX_FP16_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
        ಸ್ಥಿರಾಂಕ size_t   c_byte_ಹೆಜ್ಜೆ = (size_t) c_ಹೆಜ್ಜೆ * ಗಾತ್ರ(__fp16);
        ಸ್ಥಿರಾಂಕ size_t   dst_ಹೆಜ್ಜೆ    = 2 * (size_t) HMX_FP16_TILE_N_ELMS;
        ಸ್ಥಿರಾಂಕ uint32_t n_c_iters   = k / c_ಹೆಜ್ಜೆ;

        ಗಾಗಿ (uint32_t r = ಆರಂಭ_ಅಡ್ಡಸಾಲು; r < ಕೊನೆ_ಅಡ್ಡಸಾಲು; r += 2) {
            ಸ್ಥಿರಾಂಕ uint32_t   ct             = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
            ಸ್ಥಿರಾಂಕ uint32_t   local_r        = r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
            ಸ್ಥಿರಾಂಕ bool       row0_valid     = r < n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            ಸ್ಥಿರಾಂಕ bool       ಮುಂದಿನ_ಅಡ್ಡಸಾಲು_valid = (r + 1) < ಕೊನೆ_ಅಡ್ಡಸಾಲು && (r + 1) < n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_off0         = Q6_Vw_vadd_VwVw(v_scat_ತಳ, Q6_V_vsplat_R(local_r * 4));
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_ಹೆಜ್ಜೆ);

            __fp16 * tile_ತಳ = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
            ಸ್ಥಿರಾಂಕ uint8_t * p0 = row0_valid ? (ಸ್ಥಿರಾಂಕ uint8_t *) (vtcm_src + r * src_stride) : ಖಾಲಿ;
            ಸ್ಥಿರಾಂಕ uint8_t * p1 = ಮುಂದಿನ_ಅಡ್ಡಸಾಲು_valid ? (ಸ್ಥಿರಾಂಕ uint8_t *) (vtcm_src + (r + 1) * src_stride) : ಖಾಲಿ;

            ಪ್ರತಿಜ್ಞಾ(!p0 || hex_is_aligned(p0, 128));
            ಪ್ರತಿಜ್ಞಾ(!p1 || hex_is_aligned(p1, 128));
            ಪ್ರತಿಜ್ಞಾ(c_byte_ಹೆಜ್ಜೆ % 128 == 0);

            ಆದರೆ (p0 && p1) {
                ಗಾಗಿ (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_ಸದಿಶ v0 = hvx_vmem(p0); p0 += c_byte_ಹೆಜ್ಜೆ;
                    HVX_ಸದಿಶ v1 = hvx_vmem(p1); p1 += c_byte_ಹೆಜ್ಜೆ;
                    Q6_vscatter_RMVwV((size_t) tile_ತಳ, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_ತಳ, pair_region, v_off1, v1);
                    tile_ತಳ += dst_ಹೆಜ್ಜೆ;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vzero = Q6_V_vzero();
                ಗಾಗಿ (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_ಸದಿಶ v0 = p0 ? hvx_vmem(p0) : vzero;
                    ಆದರೆ (p0) p0 += c_byte_ಹೆಜ್ಜೆ;
                    HVX_ಸದಿಶ v1 = p1 ? hvx_vmem(p1) : vzero;
                    ಆದರೆ (p1) p1 += c_byte_ಹೆಜ್ಜೆ;
                    Q6_vscatter_RMVwV((size_t) tile_ತಳ, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_ತಳ, pair_region, v_off1, v1);
                    tile_ತಳ += dst_ಹೆಜ್ಜೆ;
                }
            }
        }
    } ಇಲ್ಲದಿದ್ದರೆ {
        // Fallback: scatter one K-tile per call (region 2047, masked).
        ಸ್ಥಿರಾಂಕ uint32_t c_ಹೆಜ್ಜೆ      = HMX_FP16_TILE_N_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
        ಸ್ಥಿರಾಂಕ size_t   c_byte_ಹೆಜ್ಜೆ = (size_t) c_ಹೆಜ್ಜೆ * ಗಾತ್ರ(__fp16);
        ಸ್ಥಿರಾಂಕ size_t   dst_ಹೆಜ್ಜೆ    = (size_t) HMX_FP16_TILE_N_ELMS;
        ಸ್ಥಿರಾಂಕ uint32_t n_c_iters   = k / c_ಹೆಜ್ಜೆ;

        ಗಾಗಿ (uint32_t r = ಆರಂಭ_ಅಡ್ಡಸಾಲು; r < ಕೊನೆ_ಅಡ್ಡಸಾಲು; r += 2) {
            ಸ್ಥಿರಾಂಕ uint32_t   ct             = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
            ಸ್ಥಿರಾಂಕ uint32_t   local_r        = r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
            ಸ್ಥಿರಾಂಕ bool       row0_valid     = r < n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            ಸ್ಥಿರಾಂಕ bool       ಮುಂದಿನ_ಅಡ್ಡಸಾಲು_valid = (r + 1) < ಕೊನೆ_ಅಡ್ಡಸಾಲು && (r + 1) < n_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ;
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_off0         = Q6_Vw_vadd_VwVw(v_scat_ತಳ, Q6_V_vsplat_R(local_r * 4));
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_ಹೆಜ್ಜೆ);

            __fp16 * tile_ತಳ = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
            ಸ್ಥಿರಾಂಕ uint8_t * p0 = row0_valid ? (ಸ್ಥಿರಾಂಕ uint8_t *) (vtcm_src + r * src_stride) : ಖಾಲಿ;
            ಸ್ಥಿರಾಂಕ uint8_t * p1 = ಮುಂದಿನ_ಅಡ್ಡಸಾಲು_valid ? (ಸ್ಥಿರಾಂಕ uint8_t *) (vtcm_src + (r + 1) * src_stride) : ಖಾಲಿ;

            ಆದರೆ (p0 && p1) {
                ಗಾಗಿ (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_ಸದಿಶ v0 = hvx_vmemu(p0); p0 += c_byte_ಹೆಜ್ಜೆ;
                    HVX_ಸದಿಶ v1 = hvx_vmemu(p1); p1 += c_byte_ಹೆಜ್ಜೆ;
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_ತಳ, single_region, v_off0, v0);
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_ತಳ, single_region, v_off1, v1);
                    tile_ತಳ += dst_ಹೆಜ್ಜೆ;
                }
            } ಇಲ್ಲದಿದ್ದರೆ {
                ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vzero = Q6_V_vzero();
                ಗಾಗಿ (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_ಸದಿಶ v0 = p0 ? hvx_vmemu(p0) : vzero;
                    ಆದರೆ (p0) p0 += c_byte_ಹೆಜ್ಜೆ;
                    HVX_ಸದಿಶ v1 = p1 ? hvx_vmemu(p1) : vzero;
                    ಆದರೆ (p1) p1 += c_byte_ಹೆಜ್ಜೆ;
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_ತಳ, single_region, v_off0, v0);
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_ತಳ, single_region, v_off1, v1);
                    tile_ತಳ += dst_ಹೆಜ್ಜೆ;
                }
            }
        }
    }
}

// Interleave row-major FP16 data into column-major tile format.
// Input: [n_rows, head_dim] row-major.  Output: tile[dim_tile][row_tile].
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_rows.
ಸ್ಥಿರ ಸಾಲೊಳಗೆ ಶೂನ್ಯ hmx_interleave_ಲಂಬಸಾಲು_ಸಂಖ್ಯೆ_ಗಮ್ಯ_tiles(__fp16 * ಸೀಮಿತ tiles_ಹೊರಹೋಗು,
                                            ಸ್ಥಿರಾಂಕ __fp16 * ಸೀಮಿತ src,
                                            uint32_t n_ಅಡ್ಡಸಾಲುಗಳು,
                                            uint32_t ತಲೆ_dim,
                                            size_t src_stride,
                                            uint32_t n_ಅಡ್ಡಸಾಲು_tiles,
                                            uint32_t ಆರಂಭ_ಅಡ್ಡಸಾಲು,
                                            uint32_t ಕೊನೆ_ಅಡ್ಡಸಾಲು) {
    __builtin_assume(ತಲೆ_dim > 0);
    ಸ್ಥಿರಾಂಕ size_t tile_stride_elms = (size_t) n_ಅಡ್ಡಸಾಲು_tiles * HMX_FP16_TILE_N_ELMS;

    ಗಾಗಿ (uint32_t r = ಆರಂಭ_ಅಡ್ಡಸಾಲು; r < ಕೊನೆ_ಅಡ್ಡಸಾಲು; r += 2) {
        ಸ್ಥಿರಾಂಕ bool ಮುಂದಿನ_ಅಡ್ಡಸಾಲು_valid = (r + 1) < ಕೊನೆ_ಅಡ್ಡಸಾಲು && (r + 1) < n_ಅಡ್ಡಸಾಲುಗಳು;

        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in0 = (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (src + r * src_stride);
        ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ * pv_in1 = ಮುಂದಿನ_ಅಡ್ಡಸಾಲು_valid ? (ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ *) (src + (r + 1) * src_stride) : ಖಾಲಿ;

        // Row-pair invariants hoisted out of the c loop.
        ಸ್ಥಿರಾಂಕ uint32_t r0      = r / HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು;
        ಸ್ಥಿರಾಂಕ uint32_t r1_half = (r % HMX_FP16_TILE_N_ಅಡ್ಡಸಾಲುಗಳು) / 2;

        // tb0 starts at tile (c0=0, r0); tb1 at the adjacent dim-tile (c0=1, r0).
        // Each c step (+= 64) advances both by 2 dim-tiles worth of fp16.
        __fp16 *     tb0     = tiles_ಹೊರಹೋಗು + (size_t) r0 * HMX_FP16_TILE_N_ELMS;
        __fp16 *     tb1     = tb0 + tile_stride_elms;
        ಸ್ಥಿರಾಂಕ size_t tb_ಹೆಜ್ಜೆ = 2 * tile_stride_elms;

        ಆದರೆ (pv_in1) {
            ಗಾಗಿ (uint32_t c = 0; c < ತಲೆ_dim; c += 64) {
                HVX_ಸದಿಶ     v0             = *pv_in0++;
                HVX_ಸದಿಶ     v1             = *pv_in1++;
                HVX_ಸದಿಶ_Pair vp             = Q6_W_vshuff_VVR(v1, v0, -2);
                ((HVX_ಸದಿಶ *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_ಸದಿಶ *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += tb_ಹೆಜ್ಜೆ;
                tb1 += tb_ಹೆಜ್ಜೆ;
            }
        } ಇಲ್ಲದಿದ್ದರೆ {
            ಸ್ಥಿರಾಂಕ HVX_ಸದಿಶ vzero = Q6_V_vzero();
            ಗಾಗಿ (uint32_t c = 0; c < ತಲೆ_dim; c += 64) {
                HVX_ಸದಿಶ     v0             = *pv_in0++;
                HVX_ಸದಿಶ_Pair vp             = Q6_W_vshuff_VVR(vzero, v0, -2);
                ((HVX_ಸದಿಶ *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_ಸದಿಶ *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += tb_ಹೆಜ್ಜೆ;
                tb1 += tb_ಹೆಜ್ಜೆ;
            }
        }
    }
}

// --- HMX inline asm macros for load-store packetization ---
#ವ್ಯಾಖ್ಯಾನಿಸು HMX_ತುಂಬು_MPY_F16(act, wt, range)              \
    "{\n"                                             \
    "    activation.hf = mxmem(" act ", " range ")\n" \
    "    weight.hf = mxmem(" wt ", " range ")\n"      \
    "}\n"

#ವ್ಯಾಖ್ಯಾನಿಸು HMX_ತುಂಬು_MPY_DEEP_F16(act, wt, range)              \
    "{\n"                                                  \
    "    activation.hf = mxmem(" act ", " range "):deep\n" \
    "    weight.hf = mxmem(" wt ", " range ")\n"           \
    "}\n"

#ವ್ಯಾಖ್ಯಾನಿಸು HMX_STORE_AFTER_F16(ಹೊರಹೋಗು, scale_reg) \
    "mxmem(" ಹೊರಹೋಗು ", " scale_reg "):after.hf = acc\n"

#ವ್ಯಾಖ್ಯಾನಿಸು HMX_ಗಣ_BIAS(scales) \
    "bias = mxmem2(" scales ")\n"

#ವ್ಯಾಖ್ಯಾನಿಸು HMX_CLRACC_F16() \
    "mxclracc.hf\n"

#ಆದರೆ_ಅಂತ್ಯ // HMX_UTILS_H
