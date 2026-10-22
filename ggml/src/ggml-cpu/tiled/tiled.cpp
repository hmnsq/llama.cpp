#अन्तर्भाव "tiled.h"
#अन्तर्भाव "tiled-kernel.h"

#अन्तर्भाव "ggml-cpu-impl.h"
#अन्तर्भाव "ggml-cpu.h"
#अन्तर्भाव "ggml.h"

// kvalues table (impl section) for the iq4_xs unpack
#define GGML_COMMON_IMPL_CPP
#अन्तर्भाव "ggml-common.h"

#अन्तर्भाव <cassert>
#अन्तर्भाव <cstdlib>
#अन्तर्भाव <cstring>

#अन्तर्भाव <mutex>

#define UNUSED GGML_UNUSED

// unpack routines for various quant types src0
स्थिर शून्य tiled_unpack_src0(स्थिरांक block_q4_K * rows, int64_t row_stride, पूर्णांक n_rows, tiled_tile_src0 * tile, पूर्णांक अंक_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr पूर्णांक NB = 8; // 32-wide subblocks
    // 12-byte packed scale/min decode, same extraction as the reference kernels
    स्थिर स्थिरांक uint32_t kmask1 = 0x3f3f3f3f;
    स्थिर स्थिरांक uint32_t kmask2 = 0x0f0f0f0f;
    स्थिर स्थिरांक uint32_t kmask3 = 0x03030303;

    स्थिरांक पूर्णांक qk_stride = अंक_k * TILED_TILE_K;
    स्थिरांक पूर्णांक nb_stride = NB * अंक_k;
    साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
        साठी (पूर्णांक r = 0; r < n_rows; r++) {
            स्थिरांक पूर्णांक d_off = slab * TILED_MICRO + r;
            स्थिरांक पूर्णांक q_off = r * qk_stride + slab * TILED_TILE_K;
            स्थिरांक पूर्णांक s_off = r * nb_stride + slab * NB;
            स्थिरांक block_q4_K & x = rows[r * row_stride + slab];

            tile->d[d_off]    = ggml_fp16_to_fp32(x.data.data.d);
            tile->dmin[d_off] = ggml_fp16_to_fp32(x.data.data.dmin);

            uint32_t utmp[4];
            स_नकल(utmp, x.scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            स्थिरांक uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            स्थिरांक uint8_t * scales = (स्थिरांक uint8_t *) &utmp[0];
            स्थिरांक uint8_t * mins = (स्थिरांक uint8_t *) &utmp[2];
            साठी (पूर्णांक s = 0; s < NB; s++) {
                tile->scales[s_off + s] = (int32_t) scales[s];
                tile->mins[s_off + s] = (int32_t) mins[s];
            }

            // extract the 4-bit codes (low 4 + high 4), same extraction as the reference kernels
            uint8_t * q = &tile->q[q_off];
            tiled_unpk_nib4(x.qs + 0,  q + 0,   q + 32);
            tiled_unpk_nib4(x.qs + 32, q + 64,  q + 96);
            tiled_unpk_nib4(x.qs + 64, q + 128, q + 160);
            tiled_unpk_nib4(x.qs + 96, q + 192, q + 224);
        }
    }
}

स्थिर शून्य tiled_unpack_src0(स्थिरांक block_q5_K * rows, int64_t row_stride, पूर्णांक n_rows, tiled_tile_src0 * tile, पूर्णांक अंक_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr पूर्णांक NB = 8; // 32-wide subblocks
    // 12-byte packed scale/min decode, same extraction as the reference kernels
    स्थिर स्थिरांक uint32_t kmask1 = 0x3f3f3f3f;
    स्थिर स्थिरांक uint32_t kmask2 = 0x0f0f0f0f;
    स्थिर स्थिरांक uint32_t kmask3 = 0x03030303;

    स्थिरांक पूर्णांक qk_stride = अंक_k * TILED_TILE_K;
    स्थिरांक पूर्णांक nb_stride = NB * अंक_k;
    साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
        साठी (पूर्णांक r = 0; r < n_rows; r++) {
            स्थिरांक पूर्णांक d_off = slab * TILED_MICRO + r;
            स्थिरांक पूर्णांक q_off = r * qk_stride + slab * TILED_TILE_K;
            स्थिरांक पूर्णांक s_off = r * nb_stride + slab * NB;
            स्थिरांक block_q5_K & x = rows[r * row_stride + slab];

            tile->d[d_off]    = ggml_fp16_to_fp32(x.data.data.d);
            tile->dmin[d_off] = ggml_fp16_to_fp32(x.data.data.dmin);

            uint32_t utmp[4];
            स_नकल(utmp, x.scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            स्थिरांक uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            स्थिरांक uint8_t * scales = (स्थिरांक uint8_t *) &utmp[0];
            स्थिरांक uint8_t * mins = (स्थिरांक uint8_t *) &utmp[2];
            साठी (पूर्णांक s = 0; s < NB; s++) {
                tile->scales[s_off + s] = (int32_t) scales[s];
                tile->mins[s_off + s] = (int32_t) mins[s];
            }

            // extract the 5-bit codes (4 low bits + 1 high bit), same as the generic kernels:
            // 64-element chunk j uses qh bits 2j (low 32) and 2j+1 (high 32); OR adds the 5th bit
            // (no overlap with the 4-bit codes, identical to the reference ADD)
            uint8_t * q = &tile->q[q_off];
            tiled_unpk_nib4(x.qs + 0,  q + 0,   q + 32);
            tiled_unpk_nib4(x.qs + 32, q + 64,  q + 96);
            tiled_unpk_nib4(x.qs + 64, q + 128, q + 160);
            tiled_unpk_nib4(x.qs + 96, q + 192, q + 224);
            tiled_unpk_or<0, 4, 1>(q + 0,   x.qh);
            tiled_unpk_or<1, 4, 1>(q + 32,  x.qh);
            tiled_unpk_or<2, 4, 1>(q + 64,  x.qh);
            tiled_unpk_or<3, 4, 1>(q + 96,  x.qh);
            tiled_unpk_or<4, 4, 1>(q + 128, x.qh);
            tiled_unpk_or<5, 4, 1>(q + 160, x.qh);
            tiled_unpk_or<6, 4, 1>(q + 192, x.qh);
            tiled_unpk_or<7, 4, 1>(q + 224, x.qh);
        }
    }
}

स्थिर शून्य tiled_unpack_src0(स्थिरांक block_q6_K * rows, int64_t row_stride, पूर्णांक n_rows, tiled_tile_src0 * tile, पूर्णांक अंक_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr पूर्णांक NB = 16; // 16-wide subblocks
    स्थिरांक पूर्णांक qk_stride = अंक_k * TILED_TILE_K;
    स्थिरांक पूर्णांक nb_stride = NB * अंक_k;
    साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
        साठी (पूर्णांक r = 0; r < n_rows; r++) {
            स्थिरांक पूर्णांक d_off = slab * TILED_MICRO + r;
            स्थिरांक पूर्णांक q_off = r * qk_stride + slab * TILED_TILE_K;
            स्थिरांक पूर्णांक s_off = r * nb_stride + slab * NB;
            स्थिरांक block_q6_K & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d);

            // 6-bit code = 4 low bits (ql) | 2 high bits (qh); see ggml_vec_dot_q6_K_q8_K_generic
            // per half the lanes are [ql lo(0:32)] [ql lo(32:64)] [ql hi(0:32)] [ql hi(32:64)]
            uint8_t * q = &tile->q[q_off];
            साठी (पूर्णांक half = 0; half < 2; half++) {
                uint8_t * out = q + 128 * half;
                tiled_unpk_nib4(x.ql + 64 * half + 0,  out + 0,  out + 64);
                tiled_unpk_nib4(x.ql + 64 * half + 32, out + 32, out + 96);
                tiled_unpk_or<0, 4, 3>(out + 0,  x.qh + 32 * half);
                tiled_unpk_or<2, 4, 3>(out + 32, x.qh + 32 * half);
                tiled_unpk_or<4, 4, 3>(out + 64, x.qh + 32 * half);
                tiled_unpk_or<6, 4, 3>(out + 96, x.qh + 32 * half);
            }
            // scale is a plain int8 per 16-element subblock (16 per 256-K)
            साठी (पूर्णांक s = 0; s < NB; s++) { tile->scales[s_off + s] = (int32_t) (int8_t) x.scales[s]; }
        }
    }
}

स्थिर शून्य tiled_unpack_src0(स्थिरांक block_q3_K * rows, int64_t row_stride, पूर्णांक n_rows, tiled_tile_src0 * tile, पूर्णांक अंक_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr पूर्णांक NB = 16; // 16-wide subblocks
    स्थिरांक पूर्णांक qk_stride = अंक_k * TILED_TILE_K;
    स्थिरांक पूर्णांक nb_stride = NB * अंक_k;
    साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
        साठी (पूर्णांक r = 0; r < n_rows; r++) {
            स्थिरांक पूर्णांक d_off = slab * TILED_MICRO + r;
            स्थिरांक पूर्णांक q_off = r * qk_stride + slab * TILED_TILE_K;
            स्थिरांक पूर्णांक s_off = r * nb_stride + slab * NB;
            स्थिरांक block_q3_K & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d);

            // 3-bit code = 2 low bits (qs) | (1 high bit from hmask << 2)
            // element e (0..255): half=e>>7, el=e&127, group=el>>5, l=el&31
            //   low2 = (qs[half*32 + l] >> 2*group) & 3
            //   high = (hmask[l] >> (half*4 + group)) & 1
            // see ggml_vec_dot_q3_K_q8_K_generic
            uint8_t * q = &tile->q[q_off];
            स्थिरांक uint8_t * s0 = x.qs;
            स्थिरांक uint8_t * s1 = x.qs + 32;
            uint8_t * o0 = q;
            uint8_t * o1 = q + 128;
            tiled_unpk_2bit<0>(s0, o0);      tiled_unpk_or<0, 2, 1>(o0,      x.hmask);
            tiled_unpk_2bit<2>(s0, o0 + 32); tiled_unpk_or<1, 2, 1>(o0 + 32, x.hmask);
            tiled_unpk_2bit<4>(s0, o0 + 64); tiled_unpk_or<2, 2, 1>(o0 + 64, x.hmask);
            tiled_unpk_2bit<6>(s0, o0 + 96); tiled_unpk_or<3, 2, 1>(o0 + 96, x.hmask);
            tiled_unpk_2bit<0>(s1, o1);      tiled_unpk_or<4, 2, 1>(o1,      x.hmask);
            tiled_unpk_2bit<2>(s1, o1 + 32); tiled_unpk_or<5, 2, 1>(o1 + 32, x.hmask);
            tiled_unpk_2bit<4>(s1, o1 + 64); tiled_unpk_or<6, 2, 1>(o1 + 64, x.hmask);
            tiled_unpk_2bit<6>(s1, o1 + 96); tiled_unpk_or<7, 2, 1>(o1 + 96, x.hmask);
            // 6-bit scale decode (same kmask trick as the reference), stored as (scales - 32)
            स्थिर स्थिरांक uint32_t kmask1 = 0x03030303;
            स्थिर स्थिरांक uint32_t kmask2 = 0x0f0f0f0f;
            uint32_t auxs[4];
            स_नकल(auxs, x.scales, 12);
            स्थिरांक uint32_t tmp = auxs[2];
            auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            स्थिरांक int8_t * scales = (स्थिरांक int8_t *) &auxs[0];
            साठी (पूर्णांक s = 0; s < NB; s++) { tile->scales[s_off + s] = (int32_t) scales[s] - 32; }
        }
    }
}

स्थिर शून्य tiled_unpack_src0(स्थिरांक block_q2_K * rows, int64_t row_stride, पूर्णांक n_rows, tiled_tile_src0 * tile, पूर्णांक अंक_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr पूर्णांक NB = 16; // 16-wide subblocks
    स्थिरांक पूर्णांक qk_stride = अंक_k * TILED_TILE_K;
    स्थिरांक पूर्णांक nb_stride = NB * अंक_k;
    साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
        साठी (पूर्णांक r = 0; r < n_rows; r++) {
            स्थिरांक पूर्णांक d_off = slab * TILED_MICRO + r;
            स्थिरांक पूर्णांक q_off = r * qk_stride + slab * TILED_TILE_K;
            स्थिरांक पूर्णांक s_off = r * nb_stride + slab * NB;
            स्थिरांक block_q2_K & x = rows[r * row_stride + slab];
            tile->d[d_off]    = ggml_fp16_to_fp32(x.data.data.d);
            tile->dmin[d_off] = ggml_fp16_to_fp32(x.data.data.dmin);

            // 2-bit code: element e -> half=e>>7, el=e&127
            //   byte = half*32 + (el & 31), shift = 2*(el >> 5)
            // see ggml_vec_dot_q2_K_q8_K_generic
            uint8_t * q = &tile->q[q_off];
            साठी (पूर्णांक half = 0; half < 2; half++) {
                स्थिरांक uint8_t * s = x.qs + 32 * half;
                uint8_t * out = q + 128 * half;
                tiled_unpk_2bit<0>(s, out + 0);
                tiled_unpk_2bit<2>(s, out + 32);
                tiled_unpk_2bit<4>(s, out + 64);
                tiled_unpk_2bit<6>(s, out + 96);
            }
            // scale/min packed in one byte per 16-element subblock: low 4 bits = scale, high 4 = min
            साठी (पूर्णांक s = 0; s < NB; s++) {
                tile->scales[s_off + s] = (int32_t) (x.scales[s] & 0xF);
                tile->mins[s_off + s] = (int32_t) (x.scales[s] >> 4);
            }
        }
    }
}

// iq4_xs: 4-bit codes through the kvalues_iq4nl LUT, 6-bit scales per 32, no min.
// LUT values are stored as kvalues + 128 so the +128 shift the kernel applies to
// the activations cancels against the bsums correction (BIAS = 128); the +128 is
// folded into the table up front so the expansion is a plain lookup
स्थिर शून्य tiled_unpack_src0(स्थिरांक block_iq4_xs * rows, int64_t row_stride, पूर्णांक n_rows, tiled_tile_src0 * tile, पूर्णांक अंक_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr पूर्णांक NB = 8; // 32-wide subblocks

    स्थिर स्थिरांक uint8_t lut[16] = {
        (uint8_t) (kvalues_iq4nl[0] + 128),  (uint8_t) (kvalues_iq4nl[1] + 128),
        (uint8_t) (kvalues_iq4nl[2] + 128),  (uint8_t) (kvalues_iq4nl[3] + 128),
        (uint8_t) (kvalues_iq4nl[4] + 128),  (uint8_t) (kvalues_iq4nl[5] + 128),
        (uint8_t) (kvalues_iq4nl[6] + 128),  (uint8_t) (kvalues_iq4nl[7] + 128),
        (uint8_t) (kvalues_iq4nl[8] + 128),  (uint8_t) (kvalues_iq4nl[9] + 128),
        (uint8_t) (kvalues_iq4nl[10] + 128), (uint8_t) (kvalues_iq4nl[11] + 128),
        (uint8_t) (kvalues_iq4nl[12] + 128), (uint8_t) (kvalues_iq4nl[13] + 128),
        (uint8_t) (kvalues_iq4nl[14] + 128), (uint8_t) (kvalues_iq4nl[15] + 128),
    };

    स्थिरांक पूर्णांक qk_stride = अंक_k * TILED_TILE_K;
    स्थिरांक पूर्णांक nb_stride = NB * अंक_k;
    साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
        साठी (पूर्णांक r = 0; r < n_rows; r++) {
            स्थिरांक पूर्णांक d_off = slab * TILED_MICRO + r;
            स्थिरांक पूर्णांक q_off = r * qk_stride + slab * TILED_TILE_K;
            स्थिरांक पूर्णांक s_off = r * nb_stride + slab * NB;
            स्थिरांक block_iq4_xs & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d);

            // 6-bit scale per 32, stored as (ls - 32); same extraction as dequantize_row_iq4_xs
            साठी (पूर्णांक s = 0; s < NB; s++) {
                स्थिरांक पूर्णांक ls = ((x.scales_l[s / 2] >> 4 * (s % 2)) & 0xf) | (((x.scales_h >> 2 * s) & 3) << 4);
                tile->scales[s_off + s] = (int32_t) ls - 32;
            }

            // 4-bit codes through the LUT: low nibbles of a byte pair come first
            uint8_t * q = &tile->q[q_off];
            uint8_t lo[32], hi[32];
            साठी (पूर्णांक u = 0; u < 4; u++) {
                tiled_unpk_nib4(x.qs + 32 * u, lo, hi);
                tiled_lut8(lut, lo + 0,  q + 64 * u + 0);
                tiled_lut8(lut, hi + 0,  q + 64 * u + 16);
                tiled_lut8(lut, lo + 16, q + 64 * u + 32);
                tiled_lut8(lut, hi + 16, q + 64 * u + 48);
            }
        }
    }
}

// iq2_xxs: 2-bit grids through the iq2xxs_grid LUT, 4-bit scale per 32, no min
// codes stored as (value + 128), matching BIAS = 128
स्थिर शून्य tiled_unpack_src0(स्थिरांक block_iq2_xxs * rows, int64_t row_stride, पूर्णांक n_rows, tiled_tile_src0 * tile, पूर्णांक अंक_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr पूर्णांक NB = 8; // 32-wide subblocks

    स्थिरांक पूर्णांक qk_stride = अंक_k * TILED_TILE_K;
    स्थिरांक पूर्णांक nb_stride = NB * अंक_k;
    साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
        साठी (पूर्णांक r = 0; r < n_rows; r++) {
            स्थिरांक पूर्णांक d_off = slab * TILED_MICRO + r;
            स्थिरांक पूर्णांक q_off = r * qk_stride + slab * TILED_TILE_K;
            स्थिरांक पूर्णांक s_off = r * nb_stride + slab * NB;
            स्थिरांक block_iq2_xxs & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d) * 0.125f;

            uint32_t aux32[2];
            स्थिरांक uint8_t * aux8 = (स्थिरांक uint8_t *) aux32;
            uint64_t g[4];
            uint8_t signs4[4];
            साठी (पूर्णांक ib32 = 0; ib32 < NB; ib32++) {
                स_नकल(aux32, x.qs + 4 * ib32, 2 * आकार(uint32_t));
                tile->scales[s_off + ib32] = (int32_t) (2 * (aux32[1] >> 28) + 1);
                साठी (पूर्णांक l = 0; l < 4; l++) {
                    g[l] = iq2xxs_grid[aux8[l]];
                    signs4[l] = ksigns_iq2xs[(aux32[1] >> (7 * l)) & 127];
                }
                tiled_unpk_sign32(g[0], g[1], g[2], g[3], signs4, &tile->q[q_off + 32 * ib32]);
            }
        }
    }
}

// iq2_xs: 2-bit grids through the iq2xs_grid LUT, 4-bit scale per 16 (two per 32), no min
// codes stored as (value + 128), matching BIAS = 128
स्थिर शून्य tiled_unpack_src0(स्थिरांक block_iq2_xs * rows, int64_t row_stride, पूर्णांक n_rows, tiled_tile_src0 * tile, पूर्णांक अंक_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr पूर्णांक NB = 16; // 16-wide subblocks

    स्थिरांक पूर्णांक qk_stride = अंक_k * TILED_TILE_K;
    स्थिरांक पूर्णांक nb_stride = NB * अंक_k;
    साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
        साठी (पूर्णांक r = 0; r < n_rows; r++) {
            स्थिरांक पूर्णांक d_off = slab * TILED_MICRO + r;
            स्थिरांक पूर्णांक q_off = r * qk_stride + slab * TILED_TILE_K;
            स्थिरांक पूर्णांक s_off = r * nb_stride + slab * NB;
            स्थिरांक block_iq2_xs & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d) * 0.125f;

            uint64_t g[4];
            uint8_t signs4[4];
            साठी (पूर्णांक ib32 = 0; ib32 < QK_K / 32; ib32++) {
                tile->scales[s_off + 2 * ib32 + 0] = (int32_t) (2 * (x.scales[ib32] & 0xf) + 1);
                tile->scales[s_off + 2 * ib32 + 1] = (int32_t) (2 * (x.scales[ib32] >> 4) + 1);
                स्थिरांक uint16_t * q = x.qs + 4 * ib32;
                साठी (पूर्णांक l = 0; l < 4; l++) {
                    g[l] = iq2xs_grid[q[l] & 511];
                    signs4[l] = ksigns_iq2xs[q[l] >> 9];
                }
                tiled_unpk_sign32(g[0], g[1], g[2], g[3], signs4, &tile->q[q_off + 32 * ib32]);
            }
        }
    }
}

// iq2_s: 2-bit grids through the iq2s_grid LUT, 4-bit scale per 16 (two per 32), no min
// codes stored as (value + 128), matching BIAS = 128
स्थिर शून्य tiled_unpack_src0(स्थिरांक block_iq2_s * rows, int64_t row_stride, पूर्णांक n_rows, tiled_tile_src0 * tile, पूर्णांक अंक_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr पूर्णांक NB = 16; // 16-wide subblocks

    स्थिरांक पूर्णांक qk_stride = अंक_k * TILED_TILE_K;
    स्थिरांक पूर्णांक nb_stride = NB * अंक_k;
    साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
        साठी (पूर्णांक r = 0; r < n_rows; r++) {
            स्थिरांक पूर्णांक d_off = slab * TILED_MICRO + r;
            स्थिरांक पूर्णांक q_off = r * qk_stride + slab * TILED_TILE_K;
            स्थिरांक पूर्णांक s_off = r * nb_stride + slab * NB;
            स्थिरांक block_iq2_s & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d) * 0.125f;

            स्थिरांक uint8_t * qs = x.qs;
            स्थिरांक uint8_t * signs = x.qs + QK_K / 8; // packed sign bytes share the qs array, same as the reference
            uint64_t g[4];
            साठी (पूर्णांक ib32 = 0; ib32 < QK_K / 32; ib32++) {
                tile->scales[s_off + 2 * ib32 + 0] = (int32_t) (2 * (x.scales[ib32] & 0xf) + 1);
                tile->scales[s_off + 2 * ib32 + 1] = (int32_t) (2 * (x.scales[ib32] >> 4) + 1);
                साठी (पूर्णांक l = 0; l < 4; l++) {
                    g[l] = iq2s_grid[qs[l] | (x.qh[ib32] << (8 - 2 * l) & 0x300)];
                }
                tiled_unpk_sign32(g[0], g[1], g[2], g[3], signs, &tile->q[q_off + 32 * ib32]);
                qs += 4;
                signs += 4;
            }
        }
    }
}

// iq3_xxs: 3-bit grids through the iq3xxs_grid LUT, 4-bit scale per 32, no min
// codes stored as (value + 128), matching BIAS = 128
स्थिर शून्य tiled_unpack_src0(स्थिरांक block_iq3_xxs * rows, int64_t row_stride, पूर्णांक n_rows, tiled_tile_src0 * tile, पूर्णांक अंक_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr पूर्णांक NB = 8; // 32-wide subblocks

    स्थिरांक पूर्णांक qk_stride = अंक_k * TILED_TILE_K;
    स्थिरांक पूर्णांक nb_stride = NB * अंक_k;
    साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
        साठी (पूर्णांक r = 0; r < n_rows; r++) {
            स्थिरांक पूर्णांक d_off = slab * TILED_MICRO + r;
            स्थिरांक पूर्णांक q_off = r * qk_stride + slab * TILED_TILE_K;
            स्थिरांक पूर्णांक s_off = r * nb_stride + slab * NB;
            स्थिरांक block_iq3_xxs & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d) * 0.25f;

            स्थिरांक uint8_t * qs = x.qs;
            स्थिरांक uint8_t * scales_and_signs = x.qs + QK_K / 4; // 4 bytes per 32: code bits in the top nibble, signs in 7-bit chunks
            uint64_t g[4];
            uint8_t signs4[4];
            साठी (पूर्णांक ib32 = 0; ib32 < QK_K / 32; ib32++) {
                स्थिरांक uint32_t aux32 = *(स्थिरांक uint32_t *) (scales_and_signs + 4 * ib32);
                tile->scales[s_off + ib32] = (int32_t) (2 * (aux32 >> 28) + 1);
                साठी (पूर्णांक l = 0; l < 4; l++) {
                    // bytes 8*l .. 8*l+7: low entry e1 (4 values) then high entry e2 (4 values)
                    g[l] = (uint64_t) iq3xxs_grid[qs[2 * l + 1]] << 32 | iq3xxs_grid[qs[2 * l + 0]];
                    signs4[l] = ksigns_iq2xs[(aux32 >> (7 * l)) & 127];
                }
                tiled_unpk_sign32(g[0], g[1], g[2], g[3], signs4, &tile->q[q_off + 32 * ib32]);
                qs += 8;
            }
        }
    }
}

// iq3_s: 3-bit grids through the iq3s_grid LUT, 4-bit scale per 32 (two per scale byte), no min
// codes stored as (value + 128), matching BIAS = 128
स्थिर शून्य tiled_unpack_src0(स्थिरांक block_iq3_s * rows, int64_t row_stride, पूर्णांक n_rows, tiled_tile_src0 * tile, पूर्णांक अंक_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr पूर्णांक NB = 8; // 32-wide subblocks

    स्थिरांक पूर्णांक qk_stride = अंक_k * TILED_TILE_K;
    स्थिरांक पूर्णांक nb_stride = NB * अंक_k;
    साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
        साठी (पूर्णांक r = 0; r < n_rows; r++) {
            स्थिरांक पूर्णांक d_off = slab * TILED_MICRO + r;
            स्थिरांक पूर्णांक q_off = r * qk_stride + slab * TILED_TILE_K;
            स्थिरांक पूर्णांक s_off = r * nb_stride + slab * NB;
            स्थिरांक block_iq3_s & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d);

            स्थिरांक uint8_t * qs = x.qs;
            स्थिरांक uint8_t * qh = x.qh;
            स्थिरांक uint8_t * signs = x.signs;
            साठी (पूर्णांक ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
                tile->scales[s_off + ib32 + 0] = (int32_t) (1 + 2 * (x.scales[ib32 / 2] & 0xf));
                tile->scales[s_off + ib32 + 1] = (int32_t) (1 + 2 * (x.scales[ib32 / 2] >> 4));
                साठी (पूर्णांक h = 0; h < 2; h++) {
                    uint64_t g[4];
                    साठी (पूर्णांक l = 0; l < 4; l++) {
                        // bytes 8*l .. 8*l+7: low entry e1 (4 values) then high entry e2 (4 values)
                        g[l] = (uint64_t) iq3s_grid[qs[2 * l + 1] | ((qh[h] << (7 - 2 * l)) & 256)] << 32
                             |  iq3s_grid[qs[2 * l + 0] | ((qh[h] << (8 - 2 * l)) & 256)];
                    }
                    tiled_unpk_sign32(g[0], g[1], g[2], g[3], signs, &tile->q[q_off + (ib32 + h) * 32]);
                    qs += 8;
                    signs += 4;
                }
                qh += 2;
            }
        }
    }
}

// iq1_s: ternary grid (values +-1/0) scaled by 8 to leave room for the +-1 delta offset,
// the /8 folds into d; 3-bit scale per 32, no min
// codes stored as (8 * grid + delta + 128), matching BIAS = 128
स्थिर शून्य tiled_unpack_src0(स्थिरांक block_iq1_s * rows, int64_t row_stride, पूर्णांक n_rows, tiled_tile_src0 * tile, पूर्णांक अंक_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr पूर्णांक NB = 8; // 32-wide subblocks

    स्थिरांक पूर्णांक qk_stride = अंक_k * TILED_TILE_K;
    स्थिरांक पूर्णांक nb_stride = NB * अंक_k;
    साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
        साठी (पूर्णांक r = 0; r < n_rows; r++) {
            स्थिरांक पूर्णांक d_off = slab * TILED_MICRO + r;
            स्थिरांक पूर्णांक q_off = r * qk_stride + slab * TILED_TILE_K;
            स्थिरांक पूर्णांक s_off = r * nb_stride + slab * NB;
            स्थिरांक block_iq1_s & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d) * 0.125f;

            स्थिरांक uint8_t * qs = x.qs;
            साठी (पूर्णांक ib = 0; ib < QK_K / 32; ib++) {
                स्थिरांक uint16_t hw = x.qh[ib];
                tile->scales[s_off + ib] = (int32_t) (2 * ((hw >> 12) & 7) + 1);
                स्थिरांक int8_t delta = (hw & 0x8000) ? -1 : 1;
                साठी (पूर्णांक l = 0; l < 4; l++) {
                    स्थिरांक uint64_t entry = iq1s_grid[qs[l] | (((hw >> (3 * l)) & 7) << 8)];
                    tiled_unpk_tern8((स्थिरांक uint8_t *) &entry, delta, &tile->q[q_off + 32 * ib + 8 * l]);
                }
                qs += 4;
            }
        }
    }
}

// iq1_m: like iq1_s but the fp16 scale is packed across the 4 scale bytes (no d field) and the
// delta offset is per 8, giving one scale per 16; 3-bit scale per 16, no min
// codes stored as (8 * grid + delta + 128), matching BIAS = 128
स्थिर शून्य tiled_unpack_src0(स्थिरांक block_iq1_m * rows, int64_t row_stride, पूर्णांक n_rows, tiled_tile_src0 * tile, पूर्णांक अंक_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr पूर्णांक NB = 16; // 16-wide subblocks

    स्थिरांक पूर्णांक qk_stride = अंक_k * TILED_TILE_K;
    स्थिरांक पूर्णांक nb_stride = NB * अंक_k;
    साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
        साठी (पूर्णांक r = 0; r < n_rows; r++) {
            स्थिरांक पूर्णांक d_off = slab * TILED_MICRO + r;
            स्थिरांक पूर्णांक q_off = r * qk_stride + slab * TILED_TILE_K;
            स्थिरांक पूर्णांक s_off = r * nb_stride + slab * NB;
            स्थिरांक block_iq1_m & x = rows[r * row_stride + slab];

            स्थिरांक uint16_t * sc = (स्थिरांक uint16_t *) x.scales;
            iq1m_scale_t scale;
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            tile->d[d_off] = ggml_fp16_to_fp32(scale.f16) * 0.125f;

            स्थिरांक uint8_t * qs = x.qs;
            स्थिरांक uint8_t * qh = x.qh;
            साठी (पूर्णांक ib = 0; ib < QK_K / 32; ib++) {
                स्थिरांक uint16_t hw = sc[ib / 2];
                स्थिरांक पूर्णांक sh = 6 * (ib % 2);
                tile->scales[s_off + 2 * ib + 0] = (int32_t) (2 * ((hw >> sh) & 7) + 1);
                tile->scales[s_off + 2 * ib + 1] = (int32_t) (2 * ((hw >> (sh + 3)) & 7) + 1);

                स्थिरांक uint16_t idx[4] = {
                    (uint16_t) (qs[0] | ((qh[0] << 8) & 0x700)),
                    (uint16_t) (qs[1] | ((qh[0] << 4) & 0x700)),
                    (uint16_t) (qs[2] | ((qh[1] << 8) & 0x700)),
                    (uint16_t) (qs[3] | ((qh[1] << 4) & 0x700)),
                };
                स्थिरांक int8_t delta[4] = {
                    (int8_t) ((qh[0] & 0x08) ? -1 : 1), (int8_t) ((qh[0] & 0x80) ? -1 : 1),
                    (int8_t) ((qh[1] & 0x08) ? -1 : 1), (int8_t) ((qh[1] & 0x80) ? -1 : 1),
                };
                साठी (पूर्णांक l = 0; l < 4; l++) {
                    स्थिरांक uint64_t entry = iq1s_grid[idx[l]];
                    tiled_unpk_tern8((स्थिरांक uint8_t *) &entry, delta[l], &tile->q[q_off + 32 * ib + 8 * l]);
                }
                qs += 4;
                qh += 2;
            }
        }
    }
}

// unpack src1 tile from q8_K rows. num_k = K-blocks per row: each row decodes num_k consecutive
// source blocks (rows[r][kblk + slab]) into the tile at row stride num_k*256, so each activation
// row is one long stream. num_k=1 is the standard single-slab path (byte-identical to before).
स्थिर शून्य tiled_unpack_src1_q8_K(स्थिरांक block_q8_K * स्थिरांक * rows, पूर्णांक n_rows, tiled_tile_src1 * tile,
                                   पूर्णांक kblk, पूर्णांक अंक_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    स्थिरांक पूर्णांक n_padded = (n_rows + TILED_MICRO - 1) & ~(TILED_MICRO - 1);
    स्थिरांक पूर्णांक bs_stride = (अंक_k == 1) ? TILED_TILE_ROWS : TILED_MICRO;
    स्थिरांक पूर्णांक n16 = TILED_TILE_K / TILED_MICRO;
    // natural [row][k] fill, zero-pad ragged tail; codes stay natural here, the driver
    // repacks them just-in-time via tiled_repack_src1 (per 16-row band on the standard path,
    // the whole chunk on the narrow path) so each repacked region is L1-hot for its uses
    साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
        स्थिरांक पूर्णांक q_off = slab * TILED_TILE_K;
        साठी (पूर्णांक r = 0; r < n_padded; r++) {
            जर (r < n_rows) {
                स_नकल(&tile->q[r * (अंक_k * TILED_TILE_K) + q_off], rows[r][kblk + slab].qs, TILED_TILE_K);
            } अन्यथा {
                स_रखो(&tile->q[r * (अंक_k * TILED_TILE_K) + q_off], 0, TILED_TILE_K);
            }
        }
        // d and bsums (ISA-independent)
        साठी (पूर्णांक r = 0; r < n_padded; r++) {
            जर (r < n_rows) {
                स्थिरांक block_q8_K & x = rows[r][kblk + slab];
                साठी (पूर्णांक s = 0; s < n16; s++) { tile->bsums[(slab * n16 + s) * bs_stride + r] = (int32_t) x.bsums[s]; }
                tile->d[slab * TILED_MICRO + r] = x.d;
            } अन्यथा {
                साठी (पूर्णांक s = 0; s < n16; s++) { tile->bsums[(slab * n16 + s) * bs_stride + r] = 0; }
                tile->d[slab * TILED_MICRO + r] = 0.0f;
            }
        }
    }
}

// GGML_CPU_TILED_MM: master switch, on by default. If off, we fast return false and normal vec_dot mul_mat resumes
स्थिर तार्किक ggml_tiled_matmul_enabled(शून्य) {
    स्थिर तार्किक enabled = सत्य;
    स्थिर std::once_flag flag;
    std::call_once(flag, []() {
        स्थिरांक अक्षर * env = दो_पर्या("GGML_CPU_TILED_MM");
        enabled = env == रिक्त || म_से_प(env) != 0;
    });
    परत enabled;
}

// GGML_CPU_TILED_MM_FORCE: test/bench only, take the tiled path even when unprofitable
स्थिर तार्किक ggml_tiled_matmul_forced(शून्य) {
    स्थिर तार्किक forced = असत्य;
    स्थिर std::once_flag flag;
    std::call_once(flag, []() {
        स्थिरांक अक्षर * env = दो_पर्या("GGML_CPU_TILED_MM_FORCE");
        forced = env != रिक्त && म_से_प(env) == 1;
    });
    परत forced;
}

// hard constraints shared by the MUL_MAT and MUL_MAT_ID entries; the src0 type gate is the
// entries' type switch

#जर !defined(__AVX512VNNI__) && !defined(__AVX2__) && !defined(__AVX__)
स्थिर तार्किक ggml_tiled_supported(स्थिरांक संरचना ggml_tensor * src0,
                                 स्थिरांक संरचना ggml_tensor * src1) {
    UNUSED(src0);
    UNUSED(src1);
    परत असत्य;
}
#अन्यथा
स्थिर तार्किक ggml_tiled_supported(स्थिरांक संरचना ggml_tensor * src0,
                                 स्थिरांक संरचना ggml_tensor * src1) {
    जर (!ggml_tiled_matmul_enabled()) {
        परत असत्य;
    }

    // repack-buffer weights hold a repacked layout, let that kernel handle
    जर (src0->extra != रिक्त) {
        परत असत्य;
    }
    // Supported quant types for src0
    निवड (src0->type) {
        स्थिती GGML_TYPE_Q6_K:
        स्थिती GGML_TYPE_Q5_K:
        स्थिती GGML_TYPE_Q4_K:
        स्थिती GGML_TYPE_Q3_K:
        स्थिती GGML_TYPE_Q2_K:
        स्थिती GGML_TYPE_IQ4_XS:
        स्थिती GGML_TYPE_IQ2_XXS:
        स्थिती GGML_TYPE_IQ2_XS:
        स्थिती GGML_TYPE_IQ2_S:
        स्थिती GGML_TYPE_IQ3_XXS:
        स्थिती GGML_TYPE_IQ3_S:
        स्थिती GGML_TYPE_IQ1_S:
        स्थिती GGML_TYPE_IQ1_M:
            परत सत्य;
        पूर्वनिर्धारित:
            परत असत्य;
    }
    जर (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_Q8_K) {
        परत असत्य;
    }

    जर (src1->type == GGML_TYPE_Q8_K && !ggml_is_contiguous(src1)) {
        // We can handle noncontiguous floats because we're repacking to q8_k anyways
        परत असत्य;
    }
    परत सत्य;
}
#endif

// per-thread workspace slot size (0 when tiled is disabled or unsupported on this arch)
स्थिर माप_प्रकार ggml_tiled_ws_size(शून्य) {
    जर (!ggml_tiled_matmul_enabled()) {
        परत 0;
    }
    परत TILED_WS_SLOT; // clean 512KB slot, rounded up from sizeof(tiled_ws)
}

माप_प्रकार ggml_tiled_wdata_size(पूर्णांक n_tasks, संरचना ggml_tensor * dst) {
    जर (! ggml_tiled_supported(dst->src[0], dst->src[1])) {
        परत 0; // unsupported, don't allocate
    }
    परत 64 + n_tasks * ggml_tiled_ws_size();  // 64 for alignment plus one 512KB slot per thread
}


// narrow-path K chunk: how many K's worth of weights to read contiguously into long/skinny tiles
// Assumes 32kb L1 cache budget
स्थिर पूर्णांक ggml_tiled_narrow_k_extent(int64_t n_rows, int64_t ne00) {
    पूर्णांक l1 = 31 * 1024;  // fit weights in 31k to leave room for scales

    पूर्णांक rows = (पूर्णांक) n_rows;
    जर (rows < TILED_MICRO) rows = TILED_MICRO;
    पूर्णांक ke = l1 / (TILED_MICRO + rows);
    ke &= ~(TILED_TILE_K - 1);  // floor to a multiple of 256
    स्थिरांक पूर्णांक ke_max = TILED_TILE_ROWS * TILED_TILE_K / TILED_MICRO; // buffer ceiling
    जर (ke > ke_max) ke = ke_max;
    पर्यंत (ke >= TILED_TILE_K && ne00 % ke != 0) {
        ke -= TILED_TILE_K;
    }
    परत ke < TILED_TILE_K ? 0 : ke;
}

// Writeback of the 256x256 window: buf is j-major (row stride buf_stride), dst is i-major (column stride dst_stride).
स्थिर शून्य tiled_store_window(स्थिरांक दशांश * buf, पूर्णांक n_src0, पूर्णांक n_src1, पूर्णांक buf_stride,
                               दशांश * dst, माप_प्रकार dst_stride) {
    पूर्णांक ri = 0;
    साठी (; ri + 16 <= n_src0; ri += 16) {
        पूर्णांक rj = 0;
        साठी (; rj + 8 <= n_src1; rj += 8) {
            दशांश r[16][8];
            साठी (पूर्णांक t = 0; t < 16; t++) {
                साठी (पूर्णांक u = 0; u < 8; u++) {
                    r[t][u] = buf[(ri + t) * buf_stride + rj + u];
                }
            }
            साठी (पूर्णांक u = 0; u < 8; u++) {
                साठी (पूर्णांक t = 0; t < 16; t++) {
                    dst[(ri + t) + (माप_प्रकार) (rj + u) * dst_stride] = r[t][u];
                }
            }
        }
        // ragged j tail
        साठी (; rj < n_src1; rj++) {
            साठी (पूर्णांक t = 0; t < 16; t++) {
                dst[(ri + t) + (माप_प्रकार) rj * dst_stride] = buf[(ri + t) * buf_stride + rj];
            }
        }
    }
    // ragged i tail
    साठी (; ri < n_src0; ri++) {
        साठी (पूर्णांक j = 0; j < n_src1; j++) {
            dst[ri + (माप_प्रकार) j * dst_stride] = buf[ri * buf_stride + j];
        }
    }
}

// MUL_MAT_ID (MoE): src1 rows are gathered per output row (expert dispatch) and dst rows are
// scattered back. The expert's cne1 gathered q8_K rows are staged once per expert into
// thread-local scratch (contiguous rows, plus the [k/4][row][4] interleave on VNNI) so the
// unpack and microtile are reused unchanged from mul_mat.

// src0 rows per g group; finer than the TILED_TILE_K kernel tile so all threads stay busy at small ne01
#define TILED_MMID_GROUP 64

// like tiled_store_window, but the dst columns are not contiguous: col_ptrs[j] points at the
// start of dst column j, whose rows are contiguous (dim 0, nb0 == 4 bytes).
स्थिर शून्य tiled_store_window_scatter(स्थिरांक दशांश * buf, पूर्णांक n_src0, पूर्णांक n_src1, पूर्णांक buf_stride,
                                       दशांश * स्थिरांक * col_ptrs) {
    पूर्णांक ri = 0;
    साठी (; ri + 16 <= n_src0; ri += 16) {
        पूर्णांक rj = 0;
        साठी (; rj + 8 <= n_src1; rj += 8) {
            दशांश r[16][8];
            साठी (पूर्णांक t = 0; t < 16; t++) {
                साठी (पूर्णांक u = 0; u < 8; u++) {
                    r[t][u] = buf[(ri + t) * buf_stride + rj + u];
                }
            }
            साठी (पूर्णांक u = 0; u < 8; u++) {
                साठी (पूर्णांक t = 0; t < 16; t++) {
                    col_ptrs[rj + u][ri + t] = r[t][u];
                }
            }
        }
        // ragged j tail
        साठी (; rj < n_src1; rj++) {
            साठी (पूर्णांक t = 0; t < 16; t++) {
                col_ptrs[rj][ri + t] = buf[(ri + t) * buf_stride + rj];
            }
        }
    }
    // ragged i tail
    साठी (; ri < n_src0; ri++) {
        साठी (पूर्णांक j = 0; j < n_src1; j++) {
            col_ptrs[j][ri] = buf[ri * buf_stride + j];
        }
    }
}


// one (g, k) macrotile: zero the acc window, sweep K in 256 element slabs, scatter the result rows
// rows points at this k window's routed src1 rows, row r at its base block
ढाँचा <typename B, पूर्णांक SUBBLK, तार्किक HAS_MIN, पूर्णांक BIAS, तार्किक ACTBIAS>
स्थिर शून्य tiled_mmid_gemm_window(संरचना ggml_tensor * dst, स्थिरांक संरचना ggml_tensor * src0,
                                   स्थिरांक अक्षर * src0_cur, int64_t r, int64_t k, int64_t nrows,
                                   स्थिरांक int32_t * expert_rows,
                                   स्थिरांक block_q8_K * स्थिरांक * rows,
                                   tiled_ws * ws) {
    स्थिरांक int64_t ne00 = src0->ne[0];
    स्थिरांक int64_t ne01 = src0->ne[1];

    स्थिरांक int64_t r_end = MIN(r + TILED_MMID_GROUP, ne01);
    स्थिरांक पूर्णांक n_src0 = (पूर्णांक) (r_end - r);

    स्थिरांक माप_प्रकार src0_bs = ggml_type_size(src0->type);
    स्थिरांक int64_t src0_stride = src0->nb[1] / src0_bs;

    // the window is at most TILED_MMID_GROUP x TILED_TILE_K, so zero only that region of acc
    साठी (int64_t i = 0; i < n_src0; i++) {
        स_रखो(&ws->acc[i * TILED_TILE_ROWS], 0, nrows * आकार(दशांश));
    }

    // scattered writeback: column m goes to its routed dst row; r * nb[0] is the window row offset
    दशांश * col_ptrs[TILED_TILE_ROWS];
    साठी (int64_t m = 0; m < nrows; m++) {
        col_ptrs[m] = (दशांश *) ((अक्षर *) dst->data + r * dst->nb[0] + expert_rows[2 * (k + m) + 0] * dst->nb[1] +
                                 expert_rows[2 * (k + m) + 1] * dst->nb[2]);
    }

    // narrow path (small nrows): we're memory bound in this case so do longer, skinny tiled in order
    // to have contiguous reads and improve memory bandwidth
    जर (nrows <= TILED_MICRO) {
        स्थिरांक पूर्णांक k_extent = ggml_tiled_narrow_k_extent(nrows, ne00);
        स्थिरांक पूर्णांक अंक_k = k_extent / TILED_TILE_K;
        साठी (int64_t k0 = 0; k0 < ne00; k0 += k_extent) {
            स्थिरांक पूर्णांक kstart = (पूर्णांक) (k0 / TILED_TILE_K);
            // activation: one 16-row band (nrows <= 16). Unpack the whole k_extent chunk (all
            // slabs) so each activation row is a long stream, then repack each slab in place
            tiled_unpack_src1_q8_K(rows, nrows, &ws->src1, kstart, अंक_k);
            // repack the whole k_extent chunk in place: num_k slabs x 4 tiles, 16 rows, row stride num_k*256
            tiled_repack_src1(&ws->src1, 0, अंक_k, ACTBIAS);
            // weight groups (16 at a time): unpack the k_extent chunk of the group (the long per-row
            // read), then one standard MAC per slab accumulating into the same acc rows
            साठी (int64_t ir0 = r; ir0 < r_end; ir0 += TILED_MICRO) {
                स्थिरांक पूर्णांक n0 = (पूर्णांक) MIN(TILED_MICRO, r_end - ir0);
                स्थिरांक B * wbase = (स्थिरांक B *) (src0_cur + ir0 * src0->nb[1] + kstart * src0_bs);
                tiled_unpack_src0(wbase, src0_stride, n0, &ws->src0, अंक_k);
                tiled_repack_src0<SUBBLK>(&ws->src0, n0, अंक_k, BIAS, ACTBIAS);
                दशांश * buf = ws->acc + (ir0 - r) * TILED_TILE_ROWS;
                साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
                    tiled_run_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1, 0, 0, अंक_k, slab, buf, TILED_TILE_ROWS);
                }
            }
        }
    } अन्यथा {
        // standard per-slab path: K is stepped in 256-element slabs
        साठी (int64_t ib = 0; ib < ne00; ib += TILED_TILE_K) {
            स्थिरांक पूर्णांक kblk = (पूर्णांक) (ib / TILED_TILE_K);
            tiled_unpack_src0((स्थिरांक B *) (src0_cur + r * src0->nb[1] + kblk * src0_bs), src0_stride, n_src0, &ws->src0, 1);
            tiled_repack_src0<SUBBLK>(&ws->src0, n_src0, 1, BIAS, ACTBIAS);
            tiled_unpack_src1_q8_K(rows, nrows, &ws->src1, kblk, 1);
            // 16x16 microtiles sweeping the window; repack each src1 band just-in-time
            // (j0-outer) so only the bands actually used are repacked and each is L1-hot
            साठी (int64_t ir1 = 0; ir1 < nrows; ir1 += TILED_MICRO) {
                tiled_repack_src1(&ws->src1, (पूर्णांक) ir1, 1, ACTBIAS);
                साठी (int64_t ir0 = r; ir0 < r_end; ir0 += TILED_MICRO) {
                    tiled_run_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1,
                        (पूर्णांक) (ir0 - r), (पूर्णांक) ir1, 1, 0,
                        ws->acc, TILED_TILE_ROWS);
                }
            }
        }
    }

    tiled_store_window_scatter(ws->acc, n_src0, nrows, TILED_TILE_ROWS, col_ptrs);
}

// one expert of MUL_MAT_ID: each k window's dispatched rows are pointed at by a per-window
// row pointer list and swept over the thread's row windows, the dst rows are scattered back
// per window
ढाँचा <typename B, पूर्णांक SUBBLK, तार्किक HAS_MIN, पूर्णांक BIAS, तार्किक ACTBIAS>
स्थिर शून्य ggml_compute_forward_mul_mat_id_tiled_one_expert(
        स्थिरांक संरचना ggml_compute_params * params,
              संरचना ggml_tensor *         dst,
        int64_t                            cur_a,
        int64_t                            cne1,
        स्थिरांक int32_t *                    expert_rows,
        अक्षर *                             scratch) {

    स्थिरांक संरचना ggml_tensor * src0 = dst->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    स्थिरांक गणना ggml_type vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;

    स्थिरांक माप_प्रकार nbw1 = (src1->type == vec_dot_type) ? nb11 : ggml_row_size(vec_dot_type, ne10);
    स्थिरांक शून्य * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;

    स्थिरांक अक्षर * src0_cur = (स्थिरांक अक्षर *) src0->data + cur_a * nb02;

    // this thread's workspace
    tiled_ws * ws = (tiled_ws *) (scratch + (माप_प्रकार) ith * ggml_tiled_ws_size());

    // groups of TILED_MMID_GROUP rows; rounded up, the window tail is clamped in the gemm
    स्थिरांक int64_t ngroups = (ne01 + TILED_MMID_GROUP - 1) / TILED_MMID_GROUP;

    स्थिरांक int64_t g0 = (ngroups * ith) / nth;
    स्थिरांक int64_t g1 = (ngroups * (ith + 1)) / nth;

    // no rows for this thread; nothing to do
    जर (g0 >= g1) {
        परत;
    }

    // MUL_MAT_ID scatters the src1 rows per routed row: point each k window's rows
    // at the routed src1 rows, then sweep the row windows
    साठी (int64_t k = 0; k < cne1; k += TILED_TILE_K) {
        स्थिरांक int64_t nrows = MIN(TILED_TILE_K, cne1 - k);

        स्थिरांक block_q8_K * rows[TILED_TILE_ROWS];
        साठी (int64_t i = 0; i < nrows; i++) {
            स्थिरांक int64_t i11 = expert_rows[2 * (k + i) + 0] % ne11;
            स्थिरांक int64_t i12 = expert_rows[2 * (k + i) + 1];
            rows[i] = (स्थिरांक block_q8_K *) ((स्थिरांक अक्षर *) wdata + (i11 + i12 * ne11) * nbw1);
        }

        साठी (int64_t g = g0; g < g1; g++) {
            स्थिरांक int64_t r = g * TILED_MMID_GROUP;

            tiled_mmid_gemm_window<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(dst, src0, src0_cur, r, k, nrows, expert_rows, rows, ws);
        }
    }
}

ढाँचा <typename B, पूर्णांक SUBBLK, तार्किक HAS_MIN, पूर्णांक BIAS, तार्किक ACTBIAS>
स्थिर शून्य ggml_compute_forward_mul_mat_tiled_one_chunk(
    स्थिरांक संरचना ggml_compute_params * params,
    संरचना ggml_tensor * dst,
    स्थिरांक int64_t ir0_start,
    स्थिरांक int64_t ir0_end,
    स्थिरांक int64_t ir1_start,
    स्थिरांक int64_t ir1_end,
    tiled_ws * ws) {

    स्थिरांक संरचना ggml_tensor * src0 = dst->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक गणना ggml_type vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;

    // broadcast factors
    स्थिरांक int64_t r2 = ne12 / ne02;
    स्थिरांक int64_t r3 = ne13 / ne03;

    जर (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        परत;
    }

    स्थिरांक शून्य * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    स्थिरांक माप_प्रकार row_size = ggml_row_size(vec_dot_type, ne10);
    स्थिरांक माप_प्रकार src0_bs  = ggml_type_size(src0->type);
    स्थिरांक माप_प्रकार src1_bs  = ggml_type_size(vec_dot_type);

    GGML_ASSERT(ne00 % 256 == 0);
    प्रतिज्ञा(ne12 % ne02 == 0);
    प्रतिज्ञा(ne13 % ne03 == 0);

    स्थिरांक int64_t src0_stride = nb01 / src0_bs;  // blocks between src0 rows
    स्थिरांक int64_t src1_stride = (src1->type == vec_dot_type ? src1->nb[1] : row_size) / src1_bs;

    स्थिरांक int64_t TILE = 256;
    स्थिरांक int64_t MICRO = 16;

    // 256-wide windows over the chunk. The iir1 window is additionally clamped at the
    // src1 batch (ne11) boundary: the tiles require a constant batch index (i12/i13)
    // within a window, so advance by the clamped end, not a fixed 256.
    साठी (int64_t iir1 = ir1_start; iir1 < ir1_end; ) {
        int64_t iir1_end = MIN(iir1 + TILE, ir1_end);
        स्थिरांक int64_t bnd = (iir1 / ne11 + 1) * ne11;
        जर (bnd < iir1_end) {
            iir1_end = bnd;
        }

        स्थिरांक पूर्णांक n_src1 = (पूर्णांक) (iir1_end - iir1);

        // batch coords, constant within the clamped window
        स्थिरांक int64_t i13 = iir1 / (ne12 * ne11);
        स्थिरांक int64_t i12 = (iir1 - i13 * ne12 * ne11) / ne11;
        // within-batch row; dst_col below holds the i12/i13 batch offset, so the store
        // applies i11 * nb1 (not the flattened iir1, which spans all batch dims)
        स्थिरांक int64_t i11 = iir1 - i13 * ne12 * ne11 - i12 * ne11;

        // dst batches == src1 batches (ggml_mul_mat), so the loop batch coords are in
        // src1 space; src0 batches are broadcast over them, map down into src0's batch
        स्थिरांक int64_t i02 = i12 / r2;
        स्थिरांक int64_t i03 = i13 / r3;

        स्थिरांक अक्षर * src0_row = (स्थिरांक अक्षर *) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3];
        अक्षर * dst_col = (अक्षर *) dst->data + i12 * nb2 + i13 * nb3;

        // rows[r] is the window's row r base block; the k-slab offset is applied in the unpack
        स्थिरांक block_q8_K * rows[TILED_TILE_ROWS];
        साठी (पूर्णांक r = 0; r < n_src1; r++) {
            rows[r] = (स्थिरांक block_q8_K *) ((स्थिरांक अक्षर *) wdata + (iir1 + r) * src1_stride * src1_bs);
        }

        साठी (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += TILE) {
            int64_t iir0_end = MIN(iir0 + TILE, ir0_end);
            स्थिरांक पूर्णांक n_src0 = (पूर्णांक) (iir0_end - iir0);

            // result buffer zeroed once per macrotile
            स_रखो(ws->acc, 0, (माप_प्रकार)TILED_TILE_ROWS * TILED_TILE_ROWS * आकार(दशांश));

            जर (n_src1 <= TILED_MICRO) {
                // narrow path: if we have few enough rows, we're memory bound
                // Do longer, skinner tiles so we have longer contiguous reads and improve bandwidth
                स्थिरांक पूर्णांक k_extent = ggml_tiled_narrow_k_extent(n_src1, ne00);
                स्थिरांक पूर्णांक अंक_k = k_extent / TILED_TILE_K;
                साठी (int64_t k0 = 0; k0 < ne00; k0 += k_extent) {
                    स्थिरांक पूर्णांक kstart = (पूर्णांक) (k0 / TILED_TILE_K);
                    // activation: one 16-row band (n_src1 <= 16). Unpack the whole k_extent chunk
                    // (all slabs) so each activation row is a long stream, then repack each slab in place
                    tiled_unpack_src1_q8_K(rows, n_src1, &ws->src1, kstart, अंक_k);
                    // repack the whole k_extent chunk in place, if kernel wants to: num_k slabs x 4 tiles, 16 rows, row stride num_k*256
                    tiled_repack_src1(&ws->src1, 0, अंक_k, ACTBIAS);
                    // weight groups: unpack the k_extent chunk of each 16-row group (the long per-row
                    // read), then one standard MAC per slab accumulating into the same acc rows
                    साठी (int64_t ir0 = iir0; ir0 < iir0_end; ir0 += MICRO) {
                        स्थिरांक पूर्णांक n0 = (पूर्णांक) MIN(MICRO, iir0_end - ir0);
                        स्थिरांक B * wbase = (स्थिरांक B *) (src0_row + ir0 * nb01 + kstart * src0_bs);
                        tiled_unpack_src0(wbase, src0_stride, n0, &ws->src0, अंक_k);
                        tiled_repack_src0<SUBBLK>(&ws->src0, n0, अंक_k, BIAS, ACTBIAS);
                        दशांश * buf = ws->acc + (ir0 - iir0) * TILED_TILE_ROWS;
                        साठी (पूर्णांक slab = 0; slab < अंक_k; slab++) {
                            tiled_run_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1, 0, 0, अंक_k, slab, buf, TILED_TILE_ROWS);
                        }
                    }
                }
            } अन्यथा {
                // standard per-slab path: K is stepped in 256-element chunks
                साठी (int64_t ib = 0; ib < ne00; ib += TILE) {
                    स्थिरांक पूर्णांक kblk = (पूर्णांक) (ib / TILE);
                    tiled_unpack_src0((स्थिरांक B *) (src0_row + iir0 * nb01 + kblk * src0_bs), src0_stride, n_src0, &ws->src0, 1);
                    tiled_repack_src0<SUBBLK>(&ws->src0, n_src0, 1, BIAS, ACTBIAS);
                    tiled_unpack_src1_q8_K(rows, n_src1, &ws->src1, kblk, 1);

                    // 16x16 microtiles sweeping the window; repack each src1 band before first use
                    साठी (int64_t ir1 = iir1; ir1 < iir1_end; ir1 += MICRO) {
                        tiled_repack_src1(&ws->src1, (पूर्णांक) (ir1 - iir1), 1, ACTBIAS);
                        साठी (int64_t ir0 = iir0; ir0 < iir0_end; ir0 += MICRO) {
                            tiled_run_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1,
                                (पूर्णांक) (ir0 - iir0), (पूर्णांक) (ir1 - iir1), 1, 0,
                                ws->acc, TILED_TILE_ROWS);
                        }
                    }
                }
            }
            // write acc back out from L2 to main memory
            tiled_store_window(ws->acc, n_src0, n_src1, TILED_TILE_ROWS,
                               (दशांश *) (dst_col + iir0 * nb0 + i11 * nb1), nb1 / nb0);
        }
        iir1 = iir1_end;
    }
}

ढाँचा <typename B, पूर्णांक SUBBLK, तार्किक HAS_MIN, पूर्णांक BIAS, तार्किक ACTBIAS>
स्थिर शून्य ggml_compute_forward_mul_mat_tiled_driver(
        स्थिरांक संरचना ggml_compute_params * params,
              संरचना ggml_tensor * dst) {

    स्थिरांक संरचना ggml_tensor * src0 = dst->src[0];
    स्थिरांक संरचना ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_द्विमान_OP_LOCALS

    स्थिरांक पूर्णांक ith = params->ith;
    स्थिरांक पूर्णांक nth = params->nth;

    गणना ggml_type      स्थिरांक vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
    ggml_from_float_t   स्थिरांक from_float   = ggml_get_type_traits_cpu(vec_dot_type)->from_float;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == आकार(दशांश));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    जर (src1->type != vec_dot_type) {
        अक्षर * wdata = (अक्षर *) params->wdata;

        स्थिरांक माप_प्रकार nbw0 = ggml_type_size(vec_dot_type);
        स्थिरांक माप_प्रकार nbw1 = ggml_row_size(vec_dot_type, ne10);
        स्थिरांक माप_प्रकार nbw2 = nbw1*ne11;
        स्थिरांक माप_प्रकार nbw3 = nbw2*ne12;

        प्रतिज्ञा(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        साठी (int64_t i13 = 0; i13 < ne13; ++i13) {
            साठी (int64_t i12 = 0; i12 < ne12; ++i12) {
                साठी (int64_t i11 = 0; i11 < ne11; ++i11) {
                    माप_प्रकार bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((दशांश *)((अक्षर *) src1->data + i13*src1->nb[3] + i12*src1->nb[2] + i11*src1->nb[1] + ne10_block_start*bs*src1->nb[0]),
                               (शून्य *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    }

    जर (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth. This saves a bit of coordination right at the start.
        ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    स्थिरांक int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    स्थिरांक int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    पूर्णांक chunk_size = 256; //TILED_TILE_ROWS;

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim. CEIL(nr/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // Step down chunk size if too few chunks to saturate cores, minimum is microtile size
    पर्यंत (nchunk0 * nchunk1 < nth * 4 && chunk_size > 16) {
        chunk_size = chunk_size / 2;
        nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
    }

    // The number of elements in each chunk
    स्थिरांक int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    स्थिरांक int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    पूर्णांक current_chunk = ith;

    // per-thread workspace: after any converted src1 data in wdata, aligned to 64
    अक्षर * ws_base = (अक्षर *) params->wdata;
    जर (src1->type != vec_dot_type) {
        ws_base += GGML_PAD(ggml_row_size(vec_dot_type, ggml_nelements(src1)), 64);
    }
    ws_base = (अक्षर *) (((uintptr_t) ws_base + 63) & ~(uintptr_t) 63);
    tiled_ws * ws = (tiled_ws *) (ws_base + (माप_प्रकार) ith * ggml_tiled_ws_size());

    // TODO:  if we KNOW we're on a machine where all cores are equal, we could skip the coordination/work-stealing and just assign chunks deterministically
    पर्यंत (current_chunk < nchunk0 * nchunk1) {
        स्थिरांक int64_t ith0 = current_chunk % nchunk0;
        स्थिरांक int64_t ith1 = current_chunk / nchunk0;

        स्थिरांक int64_t ir0_start = dr0 * ith0;
        स्थिरांक int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        स्थिरांक int64_t ir1_start = dr1 * ith1;
        स्थिरांक int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        ggml_compute_forward_mul_mat_tiled_one_chunk<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(params, dst, ir0_start, ir0_end, ir1_start, ir1_end, ws);

        जर (nth >= nchunk0 * nchunk1) {
            थांबा;
        }

        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

// src0 type dispatch, shared by the MUL_MAT and MUL_MAT_ID entries: one expert for
// MUL_MAT_ID (expert_rows != NULL), the full op for MUL_MAT
ढाँचा <typename B, पूर्णांक SUBBLK, तार्किक HAS_MIN, पूर्णांक BIAS, तार्किक ACTBIAS>
स्थिर तार्किक tiled_matmul_dispatch(स्थिरांक संरचना ggml_compute_params * params,
                                  संरचना ggml_tensor * dst,
                                  स्थिरांक int32_t * expert_rows,
                                  int64_t cur_a,
                                  int64_t cne1,
                                  अक्षर * scratch) {
    जर (expert_rows == रिक्त) {
        ggml_compute_forward_mul_mat_tiled_driver<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(params, dst);
    } अन्यथा {
        ggml_compute_forward_mul_mat_id_tiled_one_expert<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(params, dst, cur_a, cne1, expert_rows, scratch);
    }
    परत सत्य;
}

// the supported src0 types, one list for both ops; false to fall through to vec_dot
स्थिर तार्किक ggml_tiled_matmul_type_dispatch(स्थिरांक संरचना ggml_compute_params * params,
                                            संरचना ggml_tensor * dst,
                                            स्थिरांक int32_t * expert_rows = रिक्त,
                                            int64_t cur_a = 0,
                                            int64_t cne1 = 0,
                                            अक्षर * scratch = रिक्त) {
    निवड (dst->src[0]->type) {
        स्थिती GGML_TYPE_Q6_K:
            परत tiled_matmul_dispatch<block_q6_K, 16, असत्य, 32, सत्य>(params, dst, expert_rows, cur_a, cne1, scratch);
        स्थिती GGML_TYPE_Q5_K:
            परत tiled_matmul_dispatch<block_q5_K, 32, सत्य,  0, असत्य>(params, dst, expert_rows, cur_a, cne1, scratch);
        स्थिती GGML_TYPE_Q4_K:
            परत tiled_matmul_dispatch<block_q4_K, 32, सत्य,  0, असत्य>(params, dst, expert_rows, cur_a, cne1, scratch);
        स्थिती GGML_TYPE_Q3_K:
            परत tiled_matmul_dispatch<block_q3_K, 16, असत्य,  4, सत्य>(params, dst, expert_rows, cur_a, cne1, scratch);
        स्थिती GGML_TYPE_Q2_K:
            परत tiled_matmul_dispatch<block_q2_K, 16, सत्य,  0, असत्य>(params, dst, expert_rows, cur_a, cne1, scratch);
        स्थिती GGML_TYPE_IQ4_XS:
            परत tiled_matmul_dispatch<block_iq4_xs, 32, असत्य, 128, असत्य>(params, dst, expert_rows, cur_a, cne1, scratch);
        स्थिती GGML_TYPE_IQ2_XXS:
            परत tiled_matmul_dispatch<block_iq2_xxs, 32, असत्य, 128, सत्य>(params, dst, expert_rows, cur_a, cne1, scratch);
        स्थिती GGML_TYPE_IQ2_XS:
            परत tiled_matmul_dispatch<block_iq2_xs, 16, असत्य, 128, सत्य>(params, dst, expert_rows, cur_a, cne1, scratch);
        स्थिती GGML_TYPE_IQ2_S:
            परत tiled_matmul_dispatch<block_iq2_s, 16, असत्य, 128, सत्य>(params, dst, expert_rows, cur_a, cne1, scratch);
        स्थिती GGML_TYPE_IQ3_XXS:
            परत tiled_matmul_dispatch<block_iq3_xxs, 32, असत्य, 128, सत्य>(params, dst, expert_rows, cur_a, cne1, scratch);
        स्थिती GGML_TYPE_IQ3_S:
            परत tiled_matmul_dispatch<block_iq3_s, 32, असत्य, 128, सत्य>(params, dst, expert_rows, cur_a, cne1, scratch);
        स्थिती GGML_TYPE_IQ1_S:
            परत tiled_matmul_dispatch<block_iq1_s, 32, असत्य, 128, सत्य>(params, dst, expert_rows, cur_a, cne1, scratch);
        स्थिती GGML_TYPE_IQ1_M:
            परत tiled_matmul_dispatch<block_iq1_m, 16, असत्य, 128, सत्य>(params, dst, expert_rows, cur_a, cne1, scratch);
        पूर्वनिर्धारित:
            परत असत्य;
    }
}

स्थिर तार्किक ggml_tiled_min_batch(int64_t rows) {
    //  Profitable at rows >= 8, take even when unprofitable if we're forced
    परत rows >= 8 || ggml_tiled_matmul_forced();
}

// tiled K-quant matmul; returns true if the op was computed here,
// false to fall through to the stock path
तार्किक ggml_compute_forward_mul_mat_tiled(
        स्थिरांक संरचना ggml_compute_params * params,
              संरचना ggml_tensor * dst) {
    // --use-ref means bail out and go back to vec_dot reference impl
    जर (params->use_ref) {
        परत असत्य;
    }
    जर (!ggml_tiled_supported(dst->src[0], dst->src[1])) {
        परत असत्य;
    }
    जर (!ggml_tiled_min_batch(dst->src[1]->ne[1])) {
        परत असत्य;
    }
    परत ggml_tiled_matmul_type_dispatch(params, dst);
}

// MUL_MAT_ID (MoE), one expert; returns true if the expert was computed here,
// per expert eligibility (type gate, batch floor) is decided here
तार्किक ggml_compute_forward_mul_mat_id_tiled(
        स्थिरांक संरचना ggml_compute_params * params,
              संरचना ggml_tensor *         dst,
        int64_t                            cur_a,
        int64_t                            cne1,
        स्थिरांक int32_t *                    expert_rows,
        अक्षर *                             scratch) {
    जर (params->use_ref) {
        परत असत्य;
    }
    जर (!ggml_tiled_supported(dst->src[0], dst->src[1])) {
        परत असत्य;
    }
    // profitability is per expert: the rows routed to this expert
    जर (!ggml_tiled_min_batch(cne1)) {
        परत असत्य;
    }
    परत ggml_tiled_matmul_type_dispatch(params, dst, expert_rows, cur_a, cne1, scratch);
}
