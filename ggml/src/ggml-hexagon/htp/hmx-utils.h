// HMX tile-level inline helpers (FP16 32x32 tile operations).
// Ported from htp-ops-lib/include/dsp/hmx_utils.h. (https://github.com/haozixu/htp-ops-lib)

#నిర్వచితం_కాకపోతే HMX_UTILS_H
#నిర్వచించు HMX_UTILS_H

#అన్తర్భావ "hvx-base.h"

#అన్తర్భావ <assert.h>
#అన్తర్భావ <hexagon_types.h>
#అన్తర్భావ <stddef.h>

#నిర్వచించు HMX_FP16_TILE_N_అడ్డువరుసలు 32
#నిర్వచించు HMX_FP16_TILE_N_నిలువుల_సంఖ్య 32
#నిర్వచించు HMX_FP16_TILE_N_ELMS 1024
#నిర్వచించు HMX_FP16_TILE_కొలత   2048

// Initialise aligned 256-byte area with scale vector + zero padding.
స్థిర అంతఃవరుస శూన్య hmx_init_స్తంభం_scales(శూన్య *బహిర్గతం_scales, HVX_సదిశ v_scale) {
    అస్థిర HVX_సదిశ *pv = (HVX_సదిశ *) బహిర్గతం_scales;
    pv[0] = v_scale;
    pv[1] = Q6_V_vzero();
}

// --- Shared scatter offsets and interleave helper ---

// vscatter offsets for fused dequant+transpose: write K-values directly to [K][N] tile.
// word[i] = i*128 maps K-row-pair i to byte offset i*128.
// Column offset (n*4) is added at runtime.  Entries 0..15 cover one tile (region 2047);
// entries 16..31 cover the next adjacent tile (region 4095) - pick region size at the
// call site to scatter into one tile (masked) or two contiguous tiles (unmasked).
స్థిర స్థిరరాశి int32_t hmx_transpose_scatter_offsets[32] __attribute__((aligned(VLEN))) = {
    0 * 128,  1 * 128,  2 * 128,  3 * 128,  4 * 128,  5 * 128,  6 * 128,  7 * 128,  8 * 128,  9 * 128,  10 * 128,
    11 * 128, 12 * 128, 13 * 128, 14 * 128, 15 * 128, 16 * 128, 17 * 128, 18 * 128, 19 * 128, 20 * 128, 21 * 128,
    22 * 128, 23 * 128, 24 * 128, 25 * 128, 26 * 128, 27 * 128, 28 * 128, 29 * 128, 30 * 128, 31 * 128,
};

// Scatter row-major FP16 data (in VTCM scratch) into transposed [K][N] tiles.
// vtcm_src: [n_cols][src_stride] row-major fp16 (only first k elements per row are used)
// vtcm_dst: [n_col_tiles][n_k_tiles][HMX_FP16_TILE_N_ELMS] tile-major interleaved fp16
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_cols.
స్థిర అంతఃవరుస శూన్య hmx_interleave_అడ్డువరుసలు_గమ్యం_tiles(__fp16 * పరిమిత vtcm_dst,
                                            స్థిరరాశి __fp16 * పరిమిత vtcm_src,
                                            uint32_t n_నిలువుల_సంఖ్య,
                                            uint32_t k,
                                            size_t src_stride,
                                            uint32_t ప్రారంభం_అడ్డువరుస,
                                            uint32_t చివర_అడ్డువరుస) {
    ప్రతిజ్ఞా(k % HMX_FP16_TILE_N_నిలువుల_సంఖ్య == 0);

    స్థిరరాశి uint32_t       n_k_tiles     = k / HMX_FP16_TILE_N_నిలువుల_సంఖ్య;
    స్థిరరాశి HVX_సదిశ     v_scat_పునాది   = hvx_vmem(hmx_transpose_scatter_offsets);
    స్థిరరాశి HVX_సదిశ     v_scat_అడుగు   = Q6_V_vsplat_R(4);
    స్థిరరాశి HVX_సదిశ_Pred q_mask64      = Q6_Q_vsetq_R(64);
    // Each hvx_vmemu load brings 64 fp16 = 128 bytes covering 2 adjacent K-tiles.
    // When n_k_tiles is even, scatter into 2 K-tiles per call (region 4095, no mask)
    // using the upper half of hmx_transpose_scatter_offsets.  Tail one K-tile (when
    // n_k_tiles is odd) falls back to single-tile masked scatter.
    స్థిరరాశి bool           pair_scatter  = (n_k_tiles & 1) == 0;
    స్థిరరాశి size_t         pair_region   = (size_t) (2 * HMX_FP16_TILE_కొలత - 1);
    స్థిరరాశి size_t         single_region = (size_t) (HMX_FP16_TILE_కొలత - 1);
    __builtin_assume(k > 0);
    __builtin_assume(చివర_అడ్డువరుస > ప్రారంభం_అడ్డువరుస);

    ఒకవేళ (pair_scatter) {
        // Step c by 64 fp16 (two K-tiles per scatter), advance dst by 2 tiles per iter.
        స్థిరరాశి uint32_t c_అడుగు      = 2 * HMX_FP16_TILE_N_నిలువుల_సంఖ్య;
        స్థిరరాశి size_t   c_byte_అడుగు = (size_t) c_అడుగు * పరిమాణం(__fp16);
        స్థిరరాశి size_t   dst_అడుగు    = 2 * (size_t) HMX_FP16_TILE_N_ELMS;
        స్థిరరాశి uint32_t n_c_iters   = k / c_అడుగు;

        కొరకు (uint32_t r = ప్రారంభం_అడ్డువరుస; r < చివర_అడ్డువరుస; r += 2) {
            స్థిరరాశి uint32_t   ct             = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
            స్థిరరాశి uint32_t   local_r        = r % HMX_FP16_TILE_N_అడ్డువరుసలు;
            స్థిరరాశి bool       row0_valid     = r < n_నిలువుల_సంఖ్య;
            స్థిరరాశి bool       తదుపరి_అడ్డువరుస_valid = (r + 1) < చివర_అడ్డువరుస && (r + 1) < n_నిలువుల_సంఖ్య;
            స్థిరరాశి HVX_సదిశ v_off0         = Q6_Vw_vadd_VwVw(v_scat_పునాది, Q6_V_vsplat_R(local_r * 4));
            స్థిరరాశి HVX_సదిశ v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_అడుగు);

            __fp16 * tile_పునాది = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
            స్థిరరాశి uint8_t * p0 = row0_valid ? (స్థిరరాశి uint8_t *) (vtcm_src + r * src_stride) : ఖాళీ;
            స్థిరరాశి uint8_t * p1 = తదుపరి_అడ్డువరుస_valid ? (స్థిరరాశి uint8_t *) (vtcm_src + (r + 1) * src_stride) : ఖాళీ;

            ప్రతిజ్ఞా(!p0 || hex_is_aligned(p0, 128));
            ప్రతిజ్ఞా(!p1 || hex_is_aligned(p1, 128));
            ప్రతిజ్ఞా(c_byte_అడుగు % 128 == 0);

            ఒకవేళ (p0 && p1) {
                కొరకు (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_సదిశ v0 = hvx_vmem(p0); p0 += c_byte_అడుగు;
                    HVX_సదిశ v1 = hvx_vmem(p1); p1 += c_byte_అడుగు;
                    Q6_vscatter_RMVwV((size_t) tile_పునాది, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_పునాది, pair_region, v_off1, v1);
                    tile_పునాది += dst_అడుగు;
                }
            } లేకపోతే {
                స్థిరరాశి HVX_సదిశ vzero = Q6_V_vzero();
                కొరకు (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_సదిశ v0 = p0 ? hvx_vmem(p0) : vzero;
                    ఒకవేళ (p0) p0 += c_byte_అడుగు;
                    HVX_సదిశ v1 = p1 ? hvx_vmem(p1) : vzero;
                    ఒకవేళ (p1) p1 += c_byte_అడుగు;
                    Q6_vscatter_RMVwV((size_t) tile_పునాది, pair_region, v_off0, v0);
                    Q6_vscatter_RMVwV((size_t) tile_పునాది, pair_region, v_off1, v1);
                    tile_పునాది += dst_అడుగు;
                }
            }
        }
    } లేకపోతే {
        // Fallback: scatter one K-tile per call (region 2047, masked).
        స్థిరరాశి uint32_t c_అడుగు      = HMX_FP16_TILE_N_నిలువుల_సంఖ్య;
        స్థిరరాశి size_t   c_byte_అడుగు = (size_t) c_అడుగు * పరిమాణం(__fp16);
        స్థిరరాశి size_t   dst_అడుగు    = (size_t) HMX_FP16_TILE_N_ELMS;
        స్థిరరాశి uint32_t n_c_iters   = k / c_అడుగు;

        కొరకు (uint32_t r = ప్రారంభం_అడ్డువరుస; r < చివర_అడ్డువరుస; r += 2) {
            స్థిరరాశి uint32_t   ct             = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
            స్థిరరాశి uint32_t   local_r        = r % HMX_FP16_TILE_N_అడ్డువరుసలు;
            స్థిరరాశి bool       row0_valid     = r < n_నిలువుల_సంఖ్య;
            స్థిరరాశి bool       తదుపరి_అడ్డువరుస_valid = (r + 1) < చివర_అడ్డువరుస && (r + 1) < n_నిలువుల_సంఖ్య;
            స్థిరరాశి HVX_సదిశ v_off0         = Q6_Vw_vadd_VwVw(v_scat_పునాది, Q6_V_vsplat_R(local_r * 4));
            స్థిరరాశి HVX_సదిశ v_off1         = Q6_Vw_vadd_VwVw(v_off0, v_scat_అడుగు);

            __fp16 * tile_పునాది = vtcm_dst + (size_t) ct * n_k_tiles * HMX_FP16_TILE_N_ELMS;
            స్థిరరాశి uint8_t * p0 = row0_valid ? (స్థిరరాశి uint8_t *) (vtcm_src + r * src_stride) : ఖాళీ;
            స్థిరరాశి uint8_t * p1 = తదుపరి_అడ్డువరుస_valid ? (స్థిరరాశి uint8_t *) (vtcm_src + (r + 1) * src_stride) : ఖాళీ;

            ఒకవేళ (p0 && p1) {
                కొరకు (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_సదిశ v0 = hvx_vmemu(p0); p0 += c_byte_అడుగు;
                    HVX_సదిశ v1 = hvx_vmemu(p1); p1 += c_byte_అడుగు;
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_పునాది, single_region, v_off0, v0);
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_పునాది, single_region, v_off1, v1);
                    tile_పునాది += dst_అడుగు;
                }
            } లేకపోతే {
                స్థిరరాశి HVX_సదిశ vzero = Q6_V_vzero();
                కొరకు (uint32_t i = 0; i < n_c_iters; ++i) {
                    HVX_సదిశ v0 = p0 ? hvx_vmemu(p0) : vzero;
                    ఒకవేళ (p0) p0 += c_byte_అడుగు;
                    HVX_సదిశ v1 = p1 ? hvx_vmemu(p1) : vzero;
                    ఒకవేళ (p1) p1 += c_byte_అడుగు;
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_పునాది, single_region, v_off0, v0);
                    Q6_vscatter_QRMVwV(q_mask64, (size_t) tile_పునాది, single_region, v_off1, v1);
                    tile_పునాది += dst_అడుగు;
                }
            }
        }
    }
}

// Interleave row-major FP16 data into column-major tile format.
// Input: [n_rows, head_dim] row-major.  Output: tile[dim_tile][row_tile].
// Processes rows [start_row, end_row) for multi-thread slicing.
// Full range: start_row=0, end_row=n_rows.
స్థిర అంతఃవరుస శూన్య hmx_interleave_నిలువుల_సంఖ్య_గమ్యం_tiles(__fp16 * పరిమిత tiles_బహిర్గతం,
                                            స్థిరరాశి __fp16 * పరిమిత src,
                                            uint32_t n_అడ్డువరుసలు,
                                            uint32_t తల_dim,
                                            size_t src_stride,
                                            uint32_t n_అడ్డువరుస_tiles,
                                            uint32_t ప్రారంభం_అడ్డువరుస,
                                            uint32_t చివర_అడ్డువరుస) {
    __builtin_assume(తల_dim > 0);
    స్థిరరాశి size_t tile_stride_elms = (size_t) n_అడ్డువరుస_tiles * HMX_FP16_TILE_N_ELMS;

    కొరకు (uint32_t r = ప్రారంభం_అడ్డువరుస; r < చివర_అడ్డువరుస; r += 2) {
        స్థిరరాశి bool తదుపరి_అడ్డువరుస_valid = (r + 1) < చివర_అడ్డువరుస && (r + 1) < n_అడ్డువరుసలు;

        స్థిరరాశి HVX_సదిశ * pv_in0 = (స్థిరరాశి HVX_సదిశ *) (src + r * src_stride);
        స్థిరరాశి HVX_సదిశ * pv_in1 = తదుపరి_అడ్డువరుస_valid ? (స్థిరరాశి HVX_సదిశ *) (src + (r + 1) * src_stride) : ఖాళీ;

        // Row-pair invariants hoisted out of the c loop.
        స్థిరరాశి uint32_t r0      = r / HMX_FP16_TILE_N_అడ్డువరుసలు;
        స్థిరరాశి uint32_t r1_half = (r % HMX_FP16_TILE_N_అడ్డువరుసలు) / 2;

        // tb0 starts at tile (c0=0, r0); tb1 at the adjacent dim-tile (c0=1, r0).
        // Each c step (+= 64) advances both by 2 dim-tiles worth of fp16.
        __fp16 *     tb0     = tiles_బహిర్గతం + (size_t) r0 * HMX_FP16_TILE_N_ELMS;
        __fp16 *     tb1     = tb0 + tile_stride_elms;
        స్థిరరాశి size_t tb_అడుగు = 2 * tile_stride_elms;

        ఒకవేళ (pv_in1) {
            కొరకు (uint32_t c = 0; c < తల_dim; c += 64) {
                HVX_సదిశ     v0             = *pv_in0++;
                HVX_సదిశ     v1             = *pv_in1++;
                HVX_సదిశ_Pair vp             = Q6_W_vshuff_VVR(v1, v0, -2);
                ((HVX_సదిశ *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_సదిశ *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += tb_అడుగు;
                tb1 += tb_అడుగు;
            }
        } లేకపోతే {
            స్థిరరాశి HVX_సదిశ vzero = Q6_V_vzero();
            కొరకు (uint32_t c = 0; c < తల_dim; c += 64) {
                HVX_సదిశ     v0             = *pv_in0++;
                HVX_సదిశ_Pair vp             = Q6_W_vshuff_VVR(vzero, v0, -2);
                ((HVX_సదిశ *) tb0)[r1_half] = Q6_V_lo_W(vp);
                ((HVX_సదిశ *) tb1)[r1_half] = Q6_V_hi_W(vp);
                tb0 += tb_అడుగు;
                tb1 += tb_అడుగు;
            }
        }
    }
}

// --- HMX inline asm macros for load-store packetization ---
#నిర్వచించు HMX_ఎక్కించు_MPY_F16(act, wt, range)              \
    "{\n"                                             \
    "    activation.hf = mxmem(" act ", " range ")\n" \
    "    weight.hf = mxmem(" wt ", " range ")\n"      \
    "}\n"

#నిర్వచించు HMX_ఎక్కించు_MPY_DEEP_F16(act, wt, range)              \
    "{\n"                                                  \
    "    activation.hf = mxmem(" act ", " range "):deep\n" \
    "    weight.hf = mxmem(" wt ", " range ")\n"           \
    "}\n"

#నిర్వచించు HMX_STORE_AFTER_F16(బహిర్గతం, scale_reg) \
    "mxmem(" బహిర్గతం ", " scale_reg "):after.hf = acc\n"

#నిర్వచించు HMX_సమితి_BIAS(scales) \
    "bias = mxmem2(" scales ")\n"

#నిర్వచించు HMX_CLRACC_F16() \
    "mxclracc.hf\n"

#అయితే_ముగింపు // HMX_UTILS_H
