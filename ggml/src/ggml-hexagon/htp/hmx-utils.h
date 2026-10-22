// HMX tile-level inline helpers (FP16 32x32 tile operations).
// Ported from htp-ops-lib/include/dsp/hmx_utils.h. (https://github.com/haozixu/htp-ops-lib)

#که_نه_تعریف_شوی HMX_UTILS_H
#تعریف_کړه HMX_UTILS_H

#ضمن "hvx-base.h"

#ضمن <assert.h>
#ضمن <hexagon_types.h>
#ضمن <stddef.h>

#تعریف_کړه HMX_FP16_TILE_N_لیکې 32
#تعریف_کړه HMX_FP16_TILE_N_ستون_شمېر 32
#تعریف_کړه HMX_FP16_TILE_N_ELMS 1024
#تعریف_کړه HMX_FP16_TILE_اندازه   2048

// Initialise aligned 256-byte area with scale vector + zero padding.
ساكن مضمن فراغ hmx_init_ستنه_scales(فراغ *بهر_scales, HVX_ویکټور v_scale) {
    متطاير HVX_ویکټور *pv = (HVX_ویکټور *) بهر_scales;
    pv[0] = v_scale;
    pv[1] = Q6_V_vzero();
}

// --- Shared scatter offsets and interleave helper ---

// vscatter offsets for fused dequant+transpose: write K-values directly to [K][N] tile.
// word[i] = i*128 maps K-row-pair i to byte offset i*128.
// Column offset (n*4) is added at runtime.  Entries 0..15 cover one tile (region 2047);
// entries 16..31 cover the next adjacent tile (region 4095) - pick region size at the
// call site to scatter into one tile (masked) or two contiguous tiles (unmasked).
ساكن ثابت int32_t hmx_transpose_scatter_offsets[32] __attribute__((aligned(VLEN))) = {
    0 * 128,  1 * 128,  2 * 128,  3 * 128,  4 * 128,  5 * 128,  6 * 128,  7 * 128,  8 * 128,  9 * 128,  10 * 128,
    11 * 128, 12 * 128, 13 * 128, 14 * 128, 15 * 128, 16 * 128, 17 * 128, 18 * 128, 19 * 128, 20 * 128, 21 * 128,
    22 * 128, 23 * 128, 24 * 128, 25 * 128, 26 * 128, 27 * 128, 28 * 128, 29 * 128, 30 * 128, 31 * 128,
};

// Scatter row-major FP16 data (in VTCM scratch) into transposed [K][N] tiles.
// vtcm_src: [n_cols][src_stride] row-major fp16 (only first k elements per row are used)
// vtcm_dst: [n_col_tiles][n_k_tiles][HMX_FP16_TILE_N_ELMS] tile-major interleaved fp16
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_cols.
ساكن مضمن فراغ hmx_interleave_لیکې_منزل_tiles(__fp16 * محدود vtcm_dst,
                                            ثابت __fp16 * محدود vtcm_src,
                                            uint32_t n_ستون_شمېر,
                                            uint32_t k,
                                            size_t src_stride,
                                            uint32_t پیل_لیکه,
                                            uint32_t پای_ټکی_لیکه) {
    أكد(k % HMX_FP16_TILE_N_ستون_شمېر == 0);

    ثابت uint32_t       n_k_tiles     = k / HMX_FP16_TILE_N_ستون_شمېر;
    ثابت HVX_ویکټور     v_scat_بنسټ   = hvx_vmem(hmx_transpose_scatter_offsets);
    ثابت HVX_ویکټور     v_scat_ګام   = Q6_V_vsplat_R(4);
    ثابت HVX_ویکټور_Pred q_mask64      = Q6_Q_vsetq_R(64);
    // Each hvx_vmemu load brings 64 fp16 = 128 bytes covering 2 adjacent K-tiles.
    // When n_k_tiles is even, scatter into 2 K-tiles per call (region 4095, no mask)
    // using the upper half of hmx_transpose_scatter_offsets.  Tail one K-tile (when
    // n_k_tiles is odd) falls back to single-tile masked scatter.
    ثابت bool           pair_scatter  = (n_k_tiles & 1) == 0;
    ثابت size_t         pair_region   = (size_t) (2 * HMX_FP16_TILE_اندازه - 1);
    ثابت size_t         single_region = (size_t) (HMX_FP16_TILE_اندازه - 1);
    __builtin_assume(k > 0);
    __builtin_assume(پای_ټکی_لیکه > پیل_لیکه);

    إذا (pair_scatter) {
        // Step c by 64 fp16 (two K-tiles per scatter), advance dst by 2 tiles per iter.
        ثابت uint32_t c_ګام      = 2 * HMX_FP16_TILE_N_ستون_شمېر;
        ثابت size_t   c_byte_ګام = (size_t) c_ګام * حجم(__fp16);
        ثابت size_t   dst_ګام    = 2 * (size_t) HMX_FP16_TILE_N_ELMS;
        ثابت uint32_t n_c_iters   = k / c_ګام;

        لكل (uint32_t r = پیل_لیکه; r < پای_ټکی_لیکه; r += 2) {
            ثابت uint32_t   ct             = r / HMX_FP16_TILE_N_لیکې;
            ثابت uint32_t   local_r        = r % HMX_FP16_TILE_N_لیکې;
            ثابت bool       row0_valid     = r < n_ستون_شمېر;
            ثابت bool       بل_لیکه_valid = (r + 1) < پای_ټکی_لیکه && (r + 1) < n_ستون_شمېر;
            ثابت HVX_ویکټور v_off0         = Q6_Vw_vadd_VwVw(v_scat_بنسټ, Q6_V_vsplat_R(local_r * 4));
            ثابت HVX_ویکټور v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_ګام);

            __fp16 * tile_بنسټ = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
            ثابت uint8_t * p0 = row0_valid ? (ثابت uint8_t *) (vtcm_src + r * src_stride) : عدم;
            ثابت uint8_t * p1 = بل_لیکه_valid ? (ثابت uint8_t *) (vtcm_src + (r + 1) * src_stride) : عدم;

            أكد(!p0 || hex_is_aligned(p0, 128));
            أكد(!p1 || hex_is_aligned(p1, 128));
            أكد(c_byte_ګام % 128 == 0);

            إذا (p0 && p1) {
                لكل (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_ویکټور v0 = hvx_vmem(p0); p0 += c_byte_ګام;
                    HVX_ویکټور v1 = hvx_vmem(p1); p1 += c_byte_ګام;
                    Q6_vscatter_RMVwV((size_t) tile_بنسټ, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_بنسټ, pair_region, v_off1, v1);
                    tile_بنسټ += dst_ګام;
                }
            } وإلا {
                ثابت HVX_ویکټور vzero = Q6_V_vzero();
                لكل (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_ویکټور v0 = p0 ? hvx_vmem(p0) : vzero;
                    إذا (p0) p0 += c_byte_ګام;
                    HVX_ویکټور v1 = p1 ? hvx_vmem(p1) : vzero;
                    إذا (p1) p1 += c_byte_ګام;
                    Q6_vscatter_RMVwV((size_t) tile_بنسټ, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_بنسټ, pair_region, v_off1, v1);
                    tile_بنسټ += dst_ګام;
                }
            }
        }
    } وإلا {
        // Fallback: scatter one K-tile per call (region 2047, masked).
        ثابت uint32_t c_ګام      = HMX_FP16_TILE_N_ستون_شمېر;
        ثابت size_t   c_byte_ګام = (size_t) c_ګام * حجم(__fp16);
        ثابت size_t   dst_ګام    = (size_t) HMX_FP16_TILE_N_ELMS;
        ثابت uint32_t n_c_iters   = k / c_ګام;

        لكل (uint32_t r = پیل_لیکه; r < پای_ټکی_لیکه; r += 2) {
            ثابت uint32_t   ct             = r / HMX_FP16_TILE_N_لیکې;
            ثابت uint32_t   local_r        = r % HMX_FP16_TILE_N_لیکې;
            ثابت bool       row0_valid     = r < n_ستون_شمېر;
            ثابت bool       بل_لیکه_valid = (r + 1) < پای_ټکی_لیکه && (r + 1) < n_ستون_شمېر;
            ثابت HVX_ویکټور v_off0         = Q6_Vw_vadd_VwVw(v_scat_بنسټ, Q6_V_vsplat_R(local_r * 4));
            ثابت HVX_ویکټور v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_ګام);

            __fp16 * tile_بنسټ = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
            ثابت uint8_t * p0 = row0_valid ? (ثابت uint8_t *) (vtcm_src + r * src_stride) : عدم;
            ثابت uint8_t * p1 = بل_لیکه_valid ? (ثابت uint8_t *) (vtcm_src + (r + 1) * src_stride) : عدم;

            إذا (p0 && p1) {
                لكل (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_ویکټور v0 = hvx_vmemu(p0); p0 += c_byte_ګام;
                    HVX_ویکټور v1 = hvx_vmemu(p1); p1 += c_byte_ګام;
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_بنسټ, single_region, v_off0, v0);
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_بنسټ, single_region, v_off1, v1);
                    tile_بنسټ += dst_ګام;
                }
            } وإلا {
                ثابت HVX_ویکټور vzero = Q6_V_vzero();
                لكل (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_ویکټور v0 = p0 ? hvx_vmemu(p0) : vzero;
                    إذا (p0) p0 += c_byte_ګام;
                    HVX_ویکټور v1 = p1 ? hvx_vmemu(p1) : vzero;
                    إذا (p1) p1 += c_byte_ګام;
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_بنسټ, single_region, v_off0, v0);
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_بنسټ, single_region, v_off1, v1);
                    tile_بنسټ += dst_ګام;
                }
            }
        }
    }
}

// Interleave row-major FP16 data into column-major tile format.
// Input: [n_rows, head_dim] row-major.  Output: tile[dim_tile][row_tile].
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_rows.
ساكن مضمن فراغ hmx_interleave_ستون_شمېر_منزل_tiles(__fp16 * محدود tiles_بهر,
                                            ثابت __fp16 * محدود src,
                                            uint32_t n_لیکې,
                                            uint32_t سر_dim,
                                            size_t src_stride,
                                            uint32_t n_لیکه_tiles,
                                            uint32_t پیل_لیکه,
                                            uint32_t پای_ټکی_لیکه) {
    __builtin_assume(سر_dim > 0);
    ثابت size_t tile_stride_elms = (size_t) n_لیکه_tiles * HMX_FP16_TILE_N_ELMS;

    لكل (uint32_t r = پیل_لیکه; r < پای_ټکی_لیکه; r += 2) {
        ثابت bool بل_لیکه_valid = (r + 1) < پای_ټکی_لیکه && (r + 1) < n_لیکې;

        ثابت HVX_ویکټور * pv_in0 = (ثابت HVX_ویکټور *) (src + r * src_stride);
        ثابت HVX_ویکټور * pv_in1 = بل_لیکه_valid ? (ثابت HVX_ویکټور *) (src + (r + 1) * src_stride) : عدم;

        // Row-pair invariants hoisted out of the c loop.
        ثابت uint32_t r0      = r / HMX_FP16_TILE_N_لیکې;
        ثابت uint32_t r1_half = (r % HMX_FP16_TILE_N_لیکې) / 2;

        // tb0 starts at tile (c0=0, r0); tb1 at the adjacent dim-tile (c0=1, r0).
        // Each c step (+= 64) advances both by 2 dim-tiles worth of fp16.
        __fp16 *     tb0     = tiles_بهر + (size_t) r0 * HMX_FP16_TILE_N_ELMS;
        __fp16 *     tb1     = tb0 + tile_stride_elms;
        ثابت size_t tb_ګام = 2 * tile_stride_elms;

        إذا (pv_in1) {
            لكل (uint32_t c = 0; c < سر_dim; c += 64) {
                HVX_ویکټور     v0             = *pv_in0++;
                HVX_ویکټور     v1             = *pv_in1++;
                HVX_ویکټور_Pair vp             = Q6_W_vshuff_VVR(v1, v0, -2);
                ((HVX_ویکټور *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_ویکټور *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += tb_ګام;
                tb1 += tb_ګام;
            }
        } وإلا {
            ثابت HVX_ویکټور vzero = Q6_V_vzero();
            لكل (uint32_t c = 0; c < سر_dim; c += 64) {
                HVX_ویکټور     v0             = *pv_in0++;
                HVX_ویکټور_Pair vp             = Q6_W_vshuff_VVR(vzero, v0, -2);
                ((HVX_ویکټور *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_ویکټور *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += tb_ګام;
                tb1 += tb_ګام;
            }
        }
    }
}

// --- HMX inline asm macros for load-store packetization ---
#تعریف_کړه HMX_بارول_MPY_F16(act, wt, range)              \
    "{\n"                                             \
    "    activation.hf = mxmem(" act ", " range ")\n" \
    "    weight.hf = mxmem(" wt ", " range ")\n"      \
    "}\n"

#تعریف_کړه HMX_بارول_MPY_DEEP_F16(act, wt, range)              \
    "{\n"                                                  \
    "    activation.hf = mxmem(" act ", " range "):deep\n" \
    "    weight.hf = mxmem(" wt ", " range ")\n"           \
    "}\n"

#تعریف_کړه HMX_STORE_AFTER_F16(بهر, scale_reg) \
    "mxmem(" بهر ", " scale_reg "):after.hf = acc\n"

#تعریف_کړه HMX_سېټ_BIAS(scales) \
    "bias = mxmem2(" scales ")\n"

#تعریف_کړه HMX_CLRACC_F16() \
    "mxclracc.hf\n"

#د_که_پای // HMX_UTILS_H
