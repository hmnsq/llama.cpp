// HMX tile-level inline helpers (FP16 32x32 tile operations).
// Ported from htp-ops-lib/include/dsp/hmx_utils.h. (https://github.com/haozixu/htp-ops-lib)

#ਜੇ_ਅਪਰਿਭਾਸ਼ਿਤ HMX_UTILS_H
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_UTILS_H

#ਅਨ੍ਤਰ੍ਭਾਵ "hvx-base.h"

#ਅਨ੍ਤਰ੍ਭਾਵ <assert.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <hexagon_types.h>
#ਅਨ੍ਤਰ੍ਭਾਵ <stddef.h>

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_FP16_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ 32
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_FP16_TILE_N_ELMS 1024
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_FP16_TILE_ਮਾਪ   2048

// Initialise aligned 256-byte area with scale vector + zero padding.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_init_ਕਾਲਮ_scales(ਖਾਲੀ *ਬਾਹਰਲਾ_scales, HVX_ਸਦਿਸ਼ v_scale) {
    ਅਸਥਿਰ HVX_ਸਦਿਸ਼ *pv = (HVX_ਸਦਿਸ਼ *) ਬਾਹਰਲਾ_scales;
    pv[0] = v_scale;
    pv[1] = Q6_V_vzero();
}

// --- Shared scatter offsets and interleave helper ---

// vscatter offsets for fused dequant+transpose: write K-values directly to [K][N] tile.
// word[i] = i*128 maps K-row-pair i to byte offset i*128.
// Column offset (n*4) is added at runtime.  Entries 0..15 cover one tile (region 2047);
// entries 16..31 cover the next adjacent tile (region 4095) - pick region size at the
// call site to scatter into one tile (masked) or two contiguous tiles (unmasked).
ਸਥਿਰ ਅਚਲ int32_t hmx_transpose_scatter_offsets[32] __attribute__((aligned(VLEN))) = {
    0 * 128,  1 * 128,  2 * 128,  3 * 128,  4 * 128,  5 * 128,  6 * 128,  7 * 128,  8 * 128,  9 * 128,  10 * 128,
    11 * 128, 12 * 128, 13 * 128, 14 * 128, 15 * 128, 16 * 128, 17 * 128, 18 * 128, 19 * 128, 20 * 128, 21 * 128,
    22 * 128, 23 * 128, 24 * 128, 25 * 128, 26 * 128, 27 * 128, 28 * 128, 29 * 128, 30 * 128, 31 * 128,
};

// Scatter row-major FP16 data (in VTCM scratch) into transposed [K][N] tiles.
// vtcm_src: [n_cols][src_stride] row-major fp16 (only first k elements per row are used)
// vtcm_dst: [n_col_tiles][n_k_tiles][HMX_FP16_TILE_N_ELMS] tile-major interleaved fp16
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_cols.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_interleave_ਪੰਕਤੀਆਂ_ਮੰਜ਼ਿਲ_tiles(__fp16 * ਸੀਮਿਤ vtcm_dst,
                                            ਅਚਲ __fp16 * ਸੀਮਿਤ vtcm_src,
                                            uint32_t n_ਖਾਨੇ_ਗਿਣਤੀ,
                                            uint32_t k,
                                            size_t src_stride,
                                            uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ,
                                            uint32_t ਸਿਰਾ_ਪੰਕਤੀ) {
    ਪ੍ਰਤਿਜ੍ਞਾ(k % HMX_FP16_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ == 0);

    ਅਚਲ uint32_t       n_k_tiles     = k / HMX_FP16_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
    ਅਚਲ HVX_ਸਦਿਸ਼     v_scat_ਅਧਾਰ   = hvx_vmem(hmx_transpose_scatter_offsets);
    ਅਚਲ HVX_ਸਦਿਸ਼     v_scat_ਕਦਮ   = Q6_V_vsplat_R(4);
    ਅਚਲ HVX_ਸਦਿਸ਼_Pred q_mask64      = Q6_Q_vsetq_R(64);
    // Each hvx_vmemu load brings 64 fp16 = 128 bytes covering 2 adjacent K-tiles.
    // When n_k_tiles is even, scatter into 2 K-tiles per call (region 4095, no mask)
    // using the upper half of hmx_transpose_scatter_offsets.  Tail one K-tile (when
    // n_k_tiles is odd) falls back to single-tile masked scatter.
    ਅਚਲ bool           pair_scatter  = (n_k_tiles & 1) == 0;
    ਅਚਲ size_t         pair_region   = (size_t) (2 * HMX_FP16_TILE_ਮਾਪ - 1);
    ਅਚਲ size_t         single_region = (size_t) (HMX_FP16_TILE_ਮਾਪ - 1);
    __builtin_assume(k > 0);
    __builtin_assume(ਸਿਰਾ_ਪੰਕਤੀ > ਸ਼ੁਰੂ_ਪੰਕਤੀ);

    ਜੇ (pair_scatter) {
        // Step c by 64 fp16 (two K-tiles per scatter), advance dst by 2 tiles per iter.
        ਅਚਲ uint32_t c_ਕਦਮ      = 2 * HMX_FP16_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
        ਅਚਲ size_t   c_byte_ਕਦਮ = (size_t) c_ਕਦਮ * ਆਕਾਰ(__fp16);
        ਅਚਲ size_t   dst_ਕਦਮ    = 2 * (size_t) HMX_FP16_TILE_N_ELMS;
        ਅਚਲ uint32_t n_c_iters   = k / c_ਕਦਮ;

        ਲਈ (uint32_t r = ਸ਼ੁਰੂ_ਪੰਕਤੀ; r < ਸਿਰਾ_ਪੰਕਤੀ; r += 2) {
            ਅਚਲ uint32_t   ct             = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
            ਅਚਲ uint32_t   local_r        = r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
            ਅਚਲ bool       row0_valid     = r < n_ਖਾਨੇ_ਗਿਣਤੀ;
            ਅਚਲ bool       ਅਗਲਾ_ਪੰਕਤੀ_valid = (r + 1) < ਸਿਰਾ_ਪੰਕਤੀ && (r + 1) < n_ਖਾਨੇ_ਗਿਣਤੀ;
            ਅਚਲ HVX_ਸਦਿਸ਼ v_off0         = Q6_Vw_vadd_VwVw(v_scat_ਅਧਾਰ, Q6_V_vsplat_R(local_r * 4));
            ਅਚਲ HVX_ਸਦਿਸ਼ v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_ਕਦਮ);

            __fp16 * tile_ਅਧਾਰ = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
            ਅਚਲ uint8_t * p0 = row0_valid ? (ਅਚਲ uint8_t *) (vtcm_src + r * src_stride) : ਨਿਰਾ;
            ਅਚਲ uint8_t * p1 = ਅਗਲਾ_ਪੰਕਤੀ_valid ? (ਅਚਲ uint8_t *) (vtcm_src + (r + 1) * src_stride) : ਨਿਰਾ;

            ਪ੍ਰਤਿਜ੍ਞਾ(!p0 || hex_is_aligned(p0, 128));
            ਪ੍ਰਤਿਜ੍ਞਾ(!p1 || hex_is_aligned(p1, 128));
            ਪ੍ਰਤਿਜ੍ਞਾ(c_byte_ਕਦਮ % 128 == 0);

            ਜੇ (p0 && p1) {
                ਲਈ (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_ਸਦਿਸ਼ v0 = hvx_vmem(p0); p0 += c_byte_ਕਦਮ;
                    HVX_ਸਦਿਸ਼ v1 = hvx_vmem(p1); p1 += c_byte_ਕਦਮ;
                    Q6_vscatter_RMVwV((size_t) tile_ਅਧਾਰ, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_ਅਧਾਰ, pair_region, v_off1, v1);
                    tile_ਅਧਾਰ += dst_ਕਦਮ;
                }
            } ਨਹੀਂਤਾਂ {
                ਅਚਲ HVX_ਸਦਿਸ਼ vzero = Q6_V_vzero();
                ਲਈ (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_ਸਦਿਸ਼ v0 = p0 ? hvx_vmem(p0) : vzero;
                    ਜੇ (p0) p0 += c_byte_ਕਦਮ;
                    HVX_ਸਦਿਸ਼ v1 = p1 ? hvx_vmem(p1) : vzero;
                    ਜੇ (p1) p1 += c_byte_ਕਦਮ;
                    Q6_vscatter_RMVwV((size_t) tile_ਅਧਾਰ, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_ਅਧਾਰ, pair_region, v_off1, v1);
                    tile_ਅਧਾਰ += dst_ਕਦਮ;
                }
            }
        }
    } ਨਹੀਂਤਾਂ {
        // Fallback: scatter one K-tile per call (region 2047, masked).
        ਅਚਲ uint32_t c_ਕਦਮ      = HMX_FP16_TILE_N_ਖਾਨੇ_ਗਿਣਤੀ;
        ਅਚਲ size_t   c_byte_ਕਦਮ = (size_t) c_ਕਦਮ * ਆਕਾਰ(__fp16);
        ਅਚਲ size_t   dst_ਕਦਮ    = (size_t) HMX_FP16_TILE_N_ELMS;
        ਅਚਲ uint32_t n_c_iters   = k / c_ਕਦਮ;

        ਲਈ (uint32_t r = ਸ਼ੁਰੂ_ਪੰਕਤੀ; r < ਸਿਰਾ_ਪੰਕਤੀ; r += 2) {
            ਅਚਲ uint32_t   ct             = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
            ਅਚਲ uint32_t   local_r        = r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
            ਅਚਲ bool       row0_valid     = r < n_ਖਾਨੇ_ਗਿਣਤੀ;
            ਅਚਲ bool       ਅਗਲਾ_ਪੰਕਤੀ_valid = (r + 1) < ਸਿਰਾ_ਪੰਕਤੀ && (r + 1) < n_ਖਾਨੇ_ਗਿਣਤੀ;
            ਅਚਲ HVX_ਸਦਿਸ਼ v_off0         = Q6_Vw_vadd_VwVw(v_scat_ਅਧਾਰ, Q6_V_vsplat_R(local_r * 4));
            ਅਚਲ HVX_ਸਦਿਸ਼ v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_ਕਦਮ);

            __fp16 * tile_ਅਧਾਰ = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
            ਅਚਲ uint8_t * p0 = row0_valid ? (ਅਚਲ uint8_t *) (vtcm_src + r * src_stride) : ਨਿਰਾ;
            ਅਚਲ uint8_t * p1 = ਅਗਲਾ_ਪੰਕਤੀ_valid ? (ਅਚਲ uint8_t *) (vtcm_src + (r + 1) * src_stride) : ਨਿਰਾ;

            ਜੇ (p0 && p1) {
                ਲਈ (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_ਸਦਿਸ਼ v0 = hvx_vmemu(p0); p0 += c_byte_ਕਦਮ;
                    HVX_ਸਦਿਸ਼ v1 = hvx_vmemu(p1); p1 += c_byte_ਕਦਮ;
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_ਅਧਾਰ, single_region, v_off0, v0);
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_ਅਧਾਰ, single_region, v_off1, v1);
                    tile_ਅਧਾਰ += dst_ਕਦਮ;
                }
            } ਨਹੀਂਤਾਂ {
                ਅਚਲ HVX_ਸਦਿਸ਼ vzero = Q6_V_vzero();
                ਲਈ (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_ਸਦਿਸ਼ v0 = p0 ? hvx_vmemu(p0) : vzero;
                    ਜੇ (p0) p0 += c_byte_ਕਦਮ;
                    HVX_ਸਦਿਸ਼ v1 = p1 ? hvx_vmemu(p1) : vzero;
                    ਜੇ (p1) p1 += c_byte_ਕਦਮ;
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_ਅਧਾਰ, single_region, v_off0, v0);
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_ਅਧਾਰ, single_region, v_off1, v1);
                    tile_ਅਧਾਰ += dst_ਕਦਮ;
                }
            }
        }
    }
}

// Interleave row-major FP16 data into column-major tile format.
// Input: [n_rows, head_dim] row-major.  Output: tile[dim_tile][row_tile].
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_rows.
ਸਥਿਰ ਸਤਰਅੰਦਰ ਖਾਲੀ hmx_interleave_ਖਾਨੇ_ਗਿਣਤੀ_ਮੰਜ਼ਿਲ_tiles(__fp16 * ਸੀਮਿਤ tiles_ਬਾਹਰਲਾ,
                                            ਅਚਲ __fp16 * ਸੀਮਿਤ src,
                                            uint32_t n_ਪੰਕਤੀਆਂ,
                                            uint32_t ਸਿਰ_dim,
                                            size_t src_stride,
                                            uint32_t n_ਪੰਕਤੀ_tiles,
                                            uint32_t ਸ਼ੁਰੂ_ਪੰਕਤੀ,
                                            uint32_t ਸਿਰਾ_ਪੰਕਤੀ) {
    __builtin_assume(ਸਿਰ_dim > 0);
    ਅਚਲ size_t tile_stride_elms = (size_t) n_ਪੰਕਤੀ_tiles * HMX_FP16_TILE_N_ELMS;

    ਲਈ (uint32_t r = ਸ਼ੁਰੂ_ਪੰਕਤੀ; r < ਸਿਰਾ_ਪੰਕਤੀ; r += 2) {
        ਅਚਲ bool ਅਗਲਾ_ਪੰਕਤੀ_valid = (r + 1) < ਸਿਰਾ_ਪੰਕਤੀ && (r + 1) < n_ਪੰਕਤੀਆਂ;

        ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in0 = (ਅਚਲ HVX_ਸਦਿਸ਼ *) (src + r * src_stride);
        ਅਚਲ HVX_ਸਦਿਸ਼ * pv_in1 = ਅਗਲਾ_ਪੰਕਤੀ_valid ? (ਅਚਲ HVX_ਸਦਿਸ਼ *) (src + (r + 1) * src_stride) : ਨਿਰਾ;

        // Row-pair invariants hoisted out of the c loop.
        ਅਚਲ uint32_t r0      = r / HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ;
        ਅਚਲ uint32_t r1_half = (r % HMX_FP16_TILE_N_ਪੰਕਤੀਆਂ) / 2;

        // tb0 starts at tile (c0=0, r0); tb1 at the adjacent dim-tile (c0=1, r0).
        // Each c step (+= 64) advances both by 2 dim-tiles worth of fp16.
        __fp16 *     tb0     = tiles_ਬਾਹਰਲਾ + (size_t) r0 * HMX_FP16_TILE_N_ELMS;
        __fp16 *     tb1     = tb0 + tile_stride_elms;
        ਅਚਲ size_t tb_ਕਦਮ = 2 * tile_stride_elms;

        ਜੇ (pv_in1) {
            ਲਈ (uint32_t c = 0; c < ਸਿਰ_dim; c += 64) {
                HVX_ਸਦਿਸ਼     v0             = *pv_in0++;
                HVX_ਸਦਿਸ਼     v1             = *pv_in1++;
                HVX_ਸਦਿਸ਼_Pair vp             = Q6_W_vshuff_VVR(v1, v0, -2);
                ((HVX_ਸਦਿਸ਼ *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_ਸਦਿਸ਼ *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += tb_ਕਦਮ;
                tb1 += tb_ਕਦਮ;
            }
        } ਨਹੀਂਤਾਂ {
            ਅਚਲ HVX_ਸਦਿਸ਼ vzero = Q6_V_vzero();
            ਲਈ (uint32_t c = 0; c < ਸਿਰ_dim; c += 64) {
                HVX_ਸਦਿਸ਼     v0             = *pv_in0++;
                HVX_ਸਦਿਸ਼_Pair vp             = Q6_W_vshuff_VVR(vzero, v0, -2);
                ((HVX_ਸਦਿਸ਼ *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_ਸਦਿਸ਼ *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += tb_ਕਦਮ;
                tb1 += tb_ਕਦਮ;
            }
        }
    }
}

// --- HMX inline asm macros for load-store packetization ---
#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_ਲੱਦੋ_MPY_F16(act, wt, range)              \
    "{\n"                                             \
    "    activation.hf = mxmem(" act ", " range ")\n" \
    "    weight.hf = mxmem(" wt ", " range ")\n"      \
    "}\n"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_ਲੱਦੋ_MPY_DEEP_F16(act, wt, range)              \
    "{\n"                                                  \
    "    activation.hf = mxmem(" act ", " range "):deep\n" \
    "    weight.hf = mxmem(" wt ", " range ")\n"           \
    "}\n"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_STORE_AFTER_F16(ਬਾਹਰਲਾ, scale_reg) \
    "mxmem(" ਬਾਹਰਲਾ ", " scale_reg "):after.hf = acc\n"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_ਸੈੱਟ_BIAS(scales) \
    "bias = mxmem2(" scales ")\n"

#ਪਰਿਭਾਸ਼ਿਤ_ਕਰੋ HMX_CLRACC_F16() \
    "mxclracc.hf\n"

#ਜੇ_ਅੰਤ // HMX_UTILS_H
