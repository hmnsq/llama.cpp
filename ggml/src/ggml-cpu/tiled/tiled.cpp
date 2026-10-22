#અન્તર્ભાવ "tiled.h"
#અન્તર્ભાવ "tiled-kernel.h"

#અન્તર્ભાવ "ggml-cpu-impl.h"
#અન્તર્ભાવ "ggml-cpu.h"
#અન્તર્ભાવ "ggml.h"

// kvalues table (impl section) for the iq4_xs unpack
#define GGML_COMMON_IMPL_CPP
#અન્તર્ભાવ "ggml-common.h"

#અન્તર્ભાવ <cassert>
#અન્તર્ભાવ <cstdlib>
#અન્તર્ભાવ <cstring>

#અન્તર્ભાવ <mutex>

#define UNUSED GGML_UNUSED

// unpack routines for various quant types src0
સ્થિર શૂન્ય tiled_unpack_src0(અચલ block_q4_K * rows, int64_t row_stride, પૂર્ણાંક n_rows, tiled_tile_src0 * tile, પૂર્ણાંક અંક_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr પૂર્ણાંક NB = 8; // 32-wide subblocks
    // 12-byte packed scale/min decode, same extraction as the reference kernels
    સ્થિર અચલ uint32_t kmask1 = 0x3f3f3f3f;
    સ્થિર અચલ uint32_t kmask2 = 0x0f0f0f0f;
    સ્થિર અચલ uint32_t kmask3 = 0x03030303;

    અચલ પૂર્ણાંક qk_stride = અંક_k * TILED_TILE_K;
    અચલ પૂર્ણાંક nb_stride = NB * અંક_k;
    માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
        માટે (પૂર્ણાંક r = 0; r < n_rows; r++) {
            અચલ પૂર્ણાંક d_off = slab * TILED_MICRO + r;
            અચલ પૂર્ણાંક q_off = r * qk_stride + slab * TILED_TILE_K;
            અચલ પૂર્ણાંક s_off = r * nb_stride + slab * NB;
            અચલ block_q4_K & x = rows[r * row_stride + slab];

            tile->d[d_off]    = ggml_fp16_to_fp32(x.data.data.d);
            tile->dmin[d_off] = ggml_fp16_to_fp32(x.data.data.dmin);

            uint32_t utmp[4];
            memcpy(utmp, x.scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            અચલ uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            અચલ uint8_t * scales = (અચલ uint8_t *) &utmp[0];
            અચલ uint8_t * mins = (અચલ uint8_t *) &utmp[2];
            માટે (પૂર્ણાંક s = 0; s < NB; s++) {
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

સ્થિર શૂન્ય tiled_unpack_src0(અચલ block_q5_K * rows, int64_t row_stride, પૂર્ણાંક n_rows, tiled_tile_src0 * tile, પૂર્ણાંક અંક_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr પૂર્ણાંક NB = 8; // 32-wide subblocks
    // 12-byte packed scale/min decode, same extraction as the reference kernels
    સ્થિર અચલ uint32_t kmask1 = 0x3f3f3f3f;
    સ્થિર અચલ uint32_t kmask2 = 0x0f0f0f0f;
    સ્થિર અચલ uint32_t kmask3 = 0x03030303;

    અચલ પૂર્ણાંક qk_stride = અંક_k * TILED_TILE_K;
    અચલ પૂર્ણાંક nb_stride = NB * અંક_k;
    માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
        માટે (પૂર્ણાંક r = 0; r < n_rows; r++) {
            અચલ પૂર્ણાંક d_off = slab * TILED_MICRO + r;
            અચલ પૂર્ણાંક q_off = r * qk_stride + slab * TILED_TILE_K;
            અચલ પૂર્ણાંક s_off = r * nb_stride + slab * NB;
            અચલ block_q5_K & x = rows[r * row_stride + slab];

            tile->d[d_off]    = ggml_fp16_to_fp32(x.data.data.d);
            tile->dmin[d_off] = ggml_fp16_to_fp32(x.data.data.dmin);

            uint32_t utmp[4];
            memcpy(utmp, x.scales, 12);
            utmp[3] = ((utmp[2] >> 4) & kmask2) | (((utmp[1] >> 6) & kmask3) << 4);
            અચલ uint32_t uaux = utmp[1] & kmask1;
            utmp[1] = (utmp[2] & kmask2) | (((utmp[0] >> 6) & kmask3) << 4);
            utmp[2] = uaux;
            utmp[0] &= kmask1;

            અચલ uint8_t * scales = (અચલ uint8_t *) &utmp[0];
            અચલ uint8_t * mins = (અચલ uint8_t *) &utmp[2];
            માટે (પૂર્ણાંક s = 0; s < NB; s++) {
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

સ્થિર શૂન્ય tiled_unpack_src0(અચલ block_q6_K * rows, int64_t row_stride, પૂર્ણાંક n_rows, tiled_tile_src0 * tile, પૂર્ણાંક અંક_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr પૂર્ણાંક NB = 16; // 16-wide subblocks
    અચલ પૂર્ણાંક qk_stride = અંક_k * TILED_TILE_K;
    અચલ પૂર્ણાંક nb_stride = NB * અંક_k;
    માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
        માટે (પૂર્ણાંક r = 0; r < n_rows; r++) {
            અચલ પૂર્ણાંક d_off = slab * TILED_MICRO + r;
            અચલ પૂર્ણાંક q_off = r * qk_stride + slab * TILED_TILE_K;
            અચલ પૂર્ણાંક s_off = r * nb_stride + slab * NB;
            અચલ block_q6_K & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d);

            // 6-bit code = 4 low bits (ql) | 2 high bits (qh); see ggml_vec_dot_q6_K_q8_K_generic
            // per half the lanes are [ql lo(0:32)] [ql lo(32:64)] [ql hi(0:32)] [ql hi(32:64)]
            uint8_t * q = &tile->q[q_off];
            માટે (પૂર્ણાંક half = 0; half < 2; half++) {
                uint8_t * out = q + 128 * half;
                tiled_unpk_nib4(x.ql + 64 * half + 0,  out + 0,  out + 64);
                tiled_unpk_nib4(x.ql + 64 * half + 32, out + 32, out + 96);
                tiled_unpk_or<0, 4, 3>(out + 0,  x.qh + 32 * half);
                tiled_unpk_or<2, 4, 3>(out + 32, x.qh + 32 * half);
                tiled_unpk_or<4, 4, 3>(out + 64, x.qh + 32 * half);
                tiled_unpk_or<6, 4, 3>(out + 96, x.qh + 32 * half);
            }
            // scale is a plain int8 per 16-element subblock (16 per 256-K)
            માટે (પૂર્ણાંક s = 0; s < NB; s++) { tile->scales[s_off + s] = (int32_t) (int8_t) x.scales[s]; }
        }
    }
}

સ્થિર શૂન્ય tiled_unpack_src0(અચલ block_q3_K * rows, int64_t row_stride, પૂર્ણાંક n_rows, tiled_tile_src0 * tile, પૂર્ણાંક અંક_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr પૂર્ણાંક NB = 16; // 16-wide subblocks
    અચલ પૂર્ણાંક qk_stride = અંક_k * TILED_TILE_K;
    અચલ પૂર્ણાંક nb_stride = NB * અંક_k;
    માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
        માટે (પૂર્ણાંક r = 0; r < n_rows; r++) {
            અચલ પૂર્ણાંક d_off = slab * TILED_MICRO + r;
            અચલ પૂર્ણાંક q_off = r * qk_stride + slab * TILED_TILE_K;
            અચલ પૂર્ણાંક s_off = r * nb_stride + slab * NB;
            અચલ block_q3_K & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d);

            // 3-bit code = 2 low bits (qs) | (1 high bit from hmask << 2)
            // element e (0..255): half=e>>7, el=e&127, group=el>>5, l=el&31
            //   low2 = (qs[half*32 + l] >> 2*group) & 3
            //   high = (hmask[l] >> (half*4 + group)) & 1
            // see ggml_vec_dot_q3_K_q8_K_generic
            uint8_t * q = &tile->q[q_off];
            અચલ uint8_t * s0 = x.qs;
            અચલ uint8_t * s1 = x.qs + 32;
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
            સ્થિર અચલ uint32_t kmask1 = 0x03030303;
            સ્થિર અચલ uint32_t kmask2 = 0x0f0f0f0f;
            uint32_t auxs[4];
            memcpy(auxs, x.scales, 12);
            અચલ uint32_t tmp = auxs[2];
            auxs[2] = ((auxs[0] >> 4) & kmask2) | (((tmp >> 4) & kmask1) << 4);
            auxs[3] = ((auxs[1] >> 4) & kmask2) | (((tmp >> 6) & kmask1) << 4);
            auxs[0] = (auxs[0] & kmask2) | (((tmp >> 0) & kmask1) << 4);
            auxs[1] = (auxs[1] & kmask2) | (((tmp >> 2) & kmask1) << 4);
            અચલ int8_t * scales = (અચલ int8_t *) &auxs[0];
            માટે (પૂર્ણાંક s = 0; s < NB; s++) { tile->scales[s_off + s] = (int32_t) scales[s] - 32; }
        }
    }
}

સ્થિર શૂન્ય tiled_unpack_src0(અચલ block_q2_K * rows, int64_t row_stride, પૂર્ણાંક n_rows, tiled_tile_src0 * tile, પૂર્ણાંક અંક_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr પૂર્ણાંક NB = 16; // 16-wide subblocks
    અચલ પૂર્ણાંક qk_stride = અંક_k * TILED_TILE_K;
    અચલ પૂર્ણાંક nb_stride = NB * અંક_k;
    માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
        માટે (પૂર્ણાંક r = 0; r < n_rows; r++) {
            અચલ પૂર્ણાંક d_off = slab * TILED_MICRO + r;
            અચલ પૂર્ણાંક q_off = r * qk_stride + slab * TILED_TILE_K;
            અચલ પૂર્ણાંક s_off = r * nb_stride + slab * NB;
            અચલ block_q2_K & x = rows[r * row_stride + slab];
            tile->d[d_off]    = ggml_fp16_to_fp32(x.data.data.d);
            tile->dmin[d_off] = ggml_fp16_to_fp32(x.data.data.dmin);

            // 2-bit code: element e -> half=e>>7, el=e&127
            //   byte = half*32 + (el & 31), shift = 2*(el >> 5)
            // see ggml_vec_dot_q2_K_q8_K_generic
            uint8_t * q = &tile->q[q_off];
            માટે (પૂર્ણાંક half = 0; half < 2; half++) {
                અચલ uint8_t * s = x.qs + 32 * half;
                uint8_t * out = q + 128 * half;
                tiled_unpk_2bit<0>(s, out + 0);
                tiled_unpk_2bit<2>(s, out + 32);
                tiled_unpk_2bit<4>(s, out + 64);
                tiled_unpk_2bit<6>(s, out + 96);
            }
            // scale/min packed in one byte per 16-element subblock: low 4 bits = scale, high 4 = min
            માટે (પૂર્ણાંક s = 0; s < NB; s++) {
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
સ્થિર શૂન્ય tiled_unpack_src0(અચલ block_iq4_xs * rows, int64_t row_stride, પૂર્ણાંક n_rows, tiled_tile_src0 * tile, પૂર્ણાંક અંક_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr પૂર્ણાંક NB = 8; // 32-wide subblocks

    સ્થિર અચલ uint8_t lut[16] = {
        (uint8_t) (kvalues_iq4nl[0] + 128),  (uint8_t) (kvalues_iq4nl[1] + 128),
        (uint8_t) (kvalues_iq4nl[2] + 128),  (uint8_t) (kvalues_iq4nl[3] + 128),
        (uint8_t) (kvalues_iq4nl[4] + 128),  (uint8_t) (kvalues_iq4nl[5] + 128),
        (uint8_t) (kvalues_iq4nl[6] + 128),  (uint8_t) (kvalues_iq4nl[7] + 128),
        (uint8_t) (kvalues_iq4nl[8] + 128),  (uint8_t) (kvalues_iq4nl[9] + 128),
        (uint8_t) (kvalues_iq4nl[10] + 128), (uint8_t) (kvalues_iq4nl[11] + 128),
        (uint8_t) (kvalues_iq4nl[12] + 128), (uint8_t) (kvalues_iq4nl[13] + 128),
        (uint8_t) (kvalues_iq4nl[14] + 128), (uint8_t) (kvalues_iq4nl[15] + 128),
    };

    અચલ પૂર્ણાંક qk_stride = અંક_k * TILED_TILE_K;
    અચલ પૂર્ણાંક nb_stride = NB * અંક_k;
    માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
        માટે (પૂર્ણાંક r = 0; r < n_rows; r++) {
            અચલ પૂર્ણાંક d_off = slab * TILED_MICRO + r;
            અચલ પૂર્ણાંક q_off = r * qk_stride + slab * TILED_TILE_K;
            અચલ પૂર્ણાંક s_off = r * nb_stride + slab * NB;
            અચલ block_iq4_xs & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d);

            // 6-bit scale per 32, stored as (ls - 32); same extraction as dequantize_row_iq4_xs
            માટે (પૂર્ણાંક s = 0; s < NB; s++) {
                અચલ પૂર્ણાંક ls = ((x.scales_l[s / 2] >> 4 * (s % 2)) & 0xf) | (((x.scales_h >> 2 * s) & 3) << 4);
                tile->scales[s_off + s] = (int32_t) ls - 32;
            }

            // 4-bit codes through the LUT: low nibbles of a byte pair come first
            uint8_t * q = &tile->q[q_off];
            uint8_t lo[32], hi[32];
            માટે (પૂર્ણાંક u = 0; u < 4; u++) {
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
સ્થિર શૂન્ય tiled_unpack_src0(અચલ block_iq2_xxs * rows, int64_t row_stride, પૂર્ણાંક n_rows, tiled_tile_src0 * tile, પૂર્ણાંક અંક_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr પૂર્ણાંક NB = 8; // 32-wide subblocks

    અચલ પૂર્ણાંક qk_stride = અંક_k * TILED_TILE_K;
    અચલ પૂર્ણાંક nb_stride = NB * અંક_k;
    માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
        માટે (પૂર્ણાંક r = 0; r < n_rows; r++) {
            અચલ પૂર્ણાંક d_off = slab * TILED_MICRO + r;
            અચલ પૂર્ણાંક q_off = r * qk_stride + slab * TILED_TILE_K;
            અચલ પૂર્ણાંક s_off = r * nb_stride + slab * NB;
            અચલ block_iq2_xxs & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d) * 0.125f;

            uint32_t aux32[2];
            અચલ uint8_t * aux8 = (અચલ uint8_t *) aux32;
            uint64_t g[4];
            uint8_t signs4[4];
            માટે (પૂર્ણાંક ib32 = 0; ib32 < NB; ib32++) {
                memcpy(aux32, x.qs + 4 * ib32, 2 * કદ(uint32_t));
                tile->scales[s_off + ib32] = (int32_t) (2 * (aux32[1] >> 28) + 1);
                માટે (પૂર્ણાંક l = 0; l < 4; l++) {
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
સ્થિર શૂન્ય tiled_unpack_src0(અચલ block_iq2_xs * rows, int64_t row_stride, પૂર્ણાંક n_rows, tiled_tile_src0 * tile, પૂર્ણાંક અંક_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr પૂર્ણાંક NB = 16; // 16-wide subblocks

    અચલ પૂર્ણાંક qk_stride = અંક_k * TILED_TILE_K;
    અચલ પૂર્ણાંક nb_stride = NB * અંક_k;
    માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
        માટે (પૂર્ણાંક r = 0; r < n_rows; r++) {
            અચલ પૂર્ણાંક d_off = slab * TILED_MICRO + r;
            અચલ પૂર્ણાંક q_off = r * qk_stride + slab * TILED_TILE_K;
            અચલ પૂર્ણાંક s_off = r * nb_stride + slab * NB;
            અચલ block_iq2_xs & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d) * 0.125f;

            uint64_t g[4];
            uint8_t signs4[4];
            માટે (પૂર્ણાંક ib32 = 0; ib32 < QK_K / 32; ib32++) {
                tile->scales[s_off + 2 * ib32 + 0] = (int32_t) (2 * (x.scales[ib32] & 0xf) + 1);
                tile->scales[s_off + 2 * ib32 + 1] = (int32_t) (2 * (x.scales[ib32] >> 4) + 1);
                અચલ uint16_t * q = x.qs + 4 * ib32;
                માટે (પૂર્ણાંક l = 0; l < 4; l++) {
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
સ્થિર શૂન્ય tiled_unpack_src0(અચલ block_iq2_s * rows, int64_t row_stride, પૂર્ણાંક n_rows, tiled_tile_src0 * tile, પૂર્ણાંક અંક_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr પૂર્ણાંક NB = 16; // 16-wide subblocks

    અચલ પૂર્ણાંક qk_stride = અંક_k * TILED_TILE_K;
    અચલ પૂર્ણાંક nb_stride = NB * અંક_k;
    માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
        માટે (પૂર્ણાંક r = 0; r < n_rows; r++) {
            અચલ પૂર્ણાંક d_off = slab * TILED_MICRO + r;
            અચલ પૂર્ણાંક q_off = r * qk_stride + slab * TILED_TILE_K;
            અચલ પૂર્ણાંક s_off = r * nb_stride + slab * NB;
            અચલ block_iq2_s & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d) * 0.125f;

            અચલ uint8_t * qs = x.qs;
            અચલ uint8_t * signs = x.qs + QK_K / 8; // packed sign bytes share the qs array, same as the reference
            uint64_t g[4];
            માટે (પૂર્ણાંક ib32 = 0; ib32 < QK_K / 32; ib32++) {
                tile->scales[s_off + 2 * ib32 + 0] = (int32_t) (2 * (x.scales[ib32] & 0xf) + 1);
                tile->scales[s_off + 2 * ib32 + 1] = (int32_t) (2 * (x.scales[ib32] >> 4) + 1);
                માટે (પૂર્ણાંક l = 0; l < 4; l++) {
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
સ્થિર શૂન્ય tiled_unpack_src0(અચલ block_iq3_xxs * rows, int64_t row_stride, પૂર્ણાંક n_rows, tiled_tile_src0 * tile, પૂર્ણાંક અંક_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr પૂર્ણાંક NB = 8; // 32-wide subblocks

    અચલ પૂર્ણાંક qk_stride = અંક_k * TILED_TILE_K;
    અચલ પૂર્ણાંક nb_stride = NB * અંક_k;
    માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
        માટે (પૂર્ણાંક r = 0; r < n_rows; r++) {
            અચલ પૂર્ણાંક d_off = slab * TILED_MICRO + r;
            અચલ પૂર્ણાંક q_off = r * qk_stride + slab * TILED_TILE_K;
            અચલ પૂર્ણાંક s_off = r * nb_stride + slab * NB;
            અચલ block_iq3_xxs & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d) * 0.25f;

            અચલ uint8_t * qs = x.qs;
            અચલ uint8_t * scales_and_signs = x.qs + QK_K / 4; // 4 bytes per 32: code bits in the top nibble, signs in 7-bit chunks
            uint64_t g[4];
            uint8_t signs4[4];
            માટે (પૂર્ણાંક ib32 = 0; ib32 < QK_K / 32; ib32++) {
                અચલ uint32_t aux32 = *(અચલ uint32_t *) (scales_and_signs + 4 * ib32);
                tile->scales[s_off + ib32] = (int32_t) (2 * (aux32 >> 28) + 1);
                માટે (પૂર્ણાંક l = 0; l < 4; l++) {
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
સ્થિર શૂન્ય tiled_unpack_src0(અચલ block_iq3_s * rows, int64_t row_stride, પૂર્ણાંક n_rows, tiled_tile_src0 * tile, પૂર્ણાંક અંક_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr પૂર્ણાંક NB = 8; // 32-wide subblocks

    અચલ પૂર્ણાંક qk_stride = અંક_k * TILED_TILE_K;
    અચલ પૂર્ણાંક nb_stride = NB * અંક_k;
    માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
        માટે (પૂર્ણાંક r = 0; r < n_rows; r++) {
            અચલ પૂર્ણાંક d_off = slab * TILED_MICRO + r;
            અચલ પૂર્ણાંક q_off = r * qk_stride + slab * TILED_TILE_K;
            અચલ પૂર્ણાંક s_off = r * nb_stride + slab * NB;
            અચલ block_iq3_s & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d);

            અચલ uint8_t * qs = x.qs;
            અચલ uint8_t * qh = x.qh;
            અચલ uint8_t * signs = x.signs;
            માટે (પૂર્ણાંક ib32 = 0; ib32 < QK_K / 32; ib32 += 2) {
                tile->scales[s_off + ib32 + 0] = (int32_t) (1 + 2 * (x.scales[ib32 / 2] & 0xf));
                tile->scales[s_off + ib32 + 1] = (int32_t) (1 + 2 * (x.scales[ib32 / 2] >> 4));
                માટે (પૂર્ણાંક h = 0; h < 2; h++) {
                    uint64_t g[4];
                    માટે (પૂર્ણાંક l = 0; l < 4; l++) {
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
સ્થિર શૂન્ય tiled_unpack_src0(અચલ block_iq1_s * rows, int64_t row_stride, પૂર્ણાંક n_rows, tiled_tile_src0 * tile, પૂર્ણાંક અંક_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr પૂર્ણાંક NB = 8; // 32-wide subblocks

    અચલ પૂર્ણાંક qk_stride = અંક_k * TILED_TILE_K;
    અચલ પૂર્ણાંક nb_stride = NB * અંક_k;
    માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
        માટે (પૂર્ણાંક r = 0; r < n_rows; r++) {
            અચલ પૂર્ણાંક d_off = slab * TILED_MICRO + r;
            અચલ પૂર્ણાંક q_off = r * qk_stride + slab * TILED_TILE_K;
            અચલ પૂર્ણાંક s_off = r * nb_stride + slab * NB;
            અચલ block_iq1_s & x = rows[r * row_stride + slab];
            tile->d[d_off] = ggml_fp16_to_fp32(x.d) * 0.125f;

            અચલ uint8_t * qs = x.qs;
            માટે (પૂર્ણાંક ib = 0; ib < QK_K / 32; ib++) {
                અચલ uint16_t hw = x.qh[ib];
                tile->scales[s_off + ib] = (int32_t) (2 * ((hw >> 12) & 7) + 1);
                અચલ int8_t delta = (hw & 0x8000) ? -1 : 1;
                માટે (પૂર્ણાંક l = 0; l < 4; l++) {
                    અચલ uint64_t entry = iq1s_grid[qs[l] | (((hw >> (3 * l)) & 7) << 8)];
                    tiled_unpk_tern8((અચલ uint8_t *) &entry, delta, &tile->q[q_off + 32 * ib + 8 * l]);
                }
                qs += 4;
            }
        }
    }
}

// iq1_m: like iq1_s but the fp16 scale is packed across the 4 scale bytes (no d field) and the
// delta offset is per 8, giving one scale per 16; 3-bit scale per 16, no min
// codes stored as (8 * grid + delta + 128), matching BIAS = 128
સ્થિર શૂન્ય tiled_unpack_src0(અચલ block_iq1_m * rows, int64_t row_stride, પૂર્ણાંક n_rows, tiled_tile_src0 * tile, પૂર્ણાંક અંક_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    constexpr પૂર્ણાંક NB = 16; // 16-wide subblocks

    અચલ પૂર્ણાંક qk_stride = અંક_k * TILED_TILE_K;
    અચલ પૂર્ણાંક nb_stride = NB * અંક_k;
    માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
        માટે (પૂર્ણાંક r = 0; r < n_rows; r++) {
            અચલ પૂર્ણાંક d_off = slab * TILED_MICRO + r;
            અચલ પૂર્ણાંક q_off = r * qk_stride + slab * TILED_TILE_K;
            અચલ પૂર્ણાંક s_off = r * nb_stride + slab * NB;
            અચલ block_iq1_m & x = rows[r * row_stride + slab];

            અચલ uint16_t * sc = (અચલ uint16_t *) x.scales;
            iq1m_scale_t scale;
            scale.u16 = (sc[0] >> 12) | ((sc[1] >> 8) & 0x00f0) | ((sc[2] >> 4) & 0x0f00) | (sc[3] & 0xf000);
            tile->d[d_off] = ggml_fp16_to_fp32(scale.f16) * 0.125f;

            અચલ uint8_t * qs = x.qs;
            અચલ uint8_t * qh = x.qh;
            માટે (પૂર્ણાંક ib = 0; ib < QK_K / 32; ib++) {
                અચલ uint16_t hw = sc[ib / 2];
                અચલ પૂર્ણાંક sh = 6 * (ib % 2);
                tile->scales[s_off + 2 * ib + 0] = (int32_t) (2 * ((hw >> sh) & 7) + 1);
                tile->scales[s_off + 2 * ib + 1] = (int32_t) (2 * ((hw >> (sh + 3)) & 7) + 1);

                અચલ uint16_t idx[4] = {
                    (uint16_t) (qs[0] | ((qh[0] << 8) & 0x700)),
                    (uint16_t) (qs[1] | ((qh[0] << 4) & 0x700)),
                    (uint16_t) (qs[2] | ((qh[1] << 8) & 0x700)),
                    (uint16_t) (qs[3] | ((qh[1] << 4) & 0x700)),
                };
                અચલ int8_t delta[4] = {
                    (int8_t) ((qh[0] & 0x08) ? -1 : 1), (int8_t) ((qh[0] & 0x80) ? -1 : 1),
                    (int8_t) ((qh[1] & 0x08) ? -1 : 1), (int8_t) ((qh[1] & 0x80) ? -1 : 1),
                };
                માટે (પૂર્ણાંક l = 0; l < 4; l++) {
                    અચલ uint64_t entry = iq1s_grid[idx[l]];
                    tiled_unpk_tern8((અચલ uint8_t *) &entry, delta[l], &tile->q[q_off + 32 * ib + 8 * l]);
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
સ્થિર શૂન્ય tiled_unpack_src1_q8_K(અચલ block_q8_K * અચલ * rows, પૂર્ણાંક n_rows, tiled_tile_src1 * tile,
                                   પૂર્ણાંક kblk, પૂર્ણાંક અંક_k) {
    GGML_ASSERT(n_rows <= TILED_TILE_ROWS);
    અચલ પૂર્ણાંક n_padded = (n_rows + TILED_MICRO - 1) & ~(TILED_MICRO - 1);
    અચલ પૂર્ણાંક bs_stride = (અંક_k == 1) ? TILED_TILE_ROWS : TILED_MICRO;
    અચલ પૂર્ણાંક n16 = TILED_TILE_K / TILED_MICRO;
    // natural [row][k] fill, zero-pad ragged tail; codes stay natural here, the driver
    // repacks them just-in-time via tiled_repack_src1 (per 16-row band on the standard path,
    // the whole chunk on the narrow path) so each repacked region is L1-hot for its uses
    માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
        અચલ પૂર્ણાંક q_off = slab * TILED_TILE_K;
        માટે (પૂર્ણાંક r = 0; r < n_padded; r++) {
            જો (r < n_rows) {
                memcpy(&tile->q[r * (અંક_k * TILED_TILE_K) + q_off], rows[r][kblk + slab].qs, TILED_TILE_K);
            } નહીંતર {
                memset(&tile->q[r * (અંક_k * TILED_TILE_K) + q_off], 0, TILED_TILE_K);
            }
        }
        // d and bsums (ISA-independent)
        માટે (પૂર્ણાંક r = 0; r < n_padded; r++) {
            જો (r < n_rows) {
                અચલ block_q8_K & x = rows[r][kblk + slab];
                માટે (પૂર્ણાંક s = 0; s < n16; s++) { tile->bsums[(slab * n16 + s) * bs_stride + r] = (int32_t) x.bsums[s]; }
                tile->d[slab * TILED_MICRO + r] = x.d;
            } નહીંતર {
                માટે (પૂર્ણાંક s = 0; s < n16; s++) { tile->bsums[(slab * n16 + s) * bs_stride + r] = 0; }
                tile->d[slab * TILED_MICRO + r] = 0.0f;
            }
        }
    }
}

// GGML_CPU_TILED_MM: master switch, on by default. If off, we fast return false and normal vec_dot mul_mat resumes
સ્થિર તાર્કિક ggml_tiled_matmul_enabled(શૂન્ય) {
    સ્થિર તાર્કિક enabled = સાચું;
    સ્થિર std::once_flag flag;
    std::call_once(flag, []() {
        અચલ અક્ષર * env = getenv("GGML_CPU_TILED_MM");
        enabled = env == ખાલી || atoi(env) != 0;
    });
    પાછા enabled;
}

// GGML_CPU_TILED_MM_FORCE: test/bench only, take the tiled path even when unprofitable
સ્થિર તાર્કિક ggml_tiled_matmul_forced(શૂન્ય) {
    સ્થિર તાર્કિક forced = ખોટું;
    સ્થિર std::once_flag flag;
    std::call_once(flag, []() {
        અચલ અક્ષર * env = getenv("GGML_CPU_TILED_MM_FORCE");
        forced = env != ખાલી && atoi(env) == 1;
    });
    પાછા forced;
}

// hard constraints shared by the MUL_MAT and MUL_MAT_ID entries; the src0 type gate is the
// entries' type switch

#જો !defined(__AVX512VNNI__) && !defined(__AVX2__) && !defined(__AVX__)
સ્થિર તાર્કિક ggml_tiled_supported(અચલ સંરચના ggml_tensor * src0,
                                 અચલ સંરચના ggml_tensor * src1) {
    UNUSED(src0);
    UNUSED(src1);
    પાછા ખોટું;
}
#નહીંતર
સ્થિર તાર્કિક ggml_tiled_supported(અચલ સંરચના ggml_tensor * src0,
                                 અચલ સંરચના ggml_tensor * src1) {
    જો (!ggml_tiled_matmul_enabled()) {
        પાછા ખોટું;
    }

    // repack-buffer weights hold a repacked layout, let that kernel handle
    જો (src0->extra != ખાલી) {
        પાછા ખોટું;
    }
    // Supported quant types for src0
    પસંદગી (src0->type) {
        સ્થિતિ GGML_TYPE_Q6_K:
        સ્થિતિ GGML_TYPE_Q5_K:
        સ્થિતિ GGML_TYPE_Q4_K:
        સ્થિતિ GGML_TYPE_Q3_K:
        સ્થિતિ GGML_TYPE_Q2_K:
        સ્થિતિ GGML_TYPE_IQ4_XS:
        સ્થિતિ GGML_TYPE_IQ2_XXS:
        સ્થિતિ GGML_TYPE_IQ2_XS:
        સ્થિતિ GGML_TYPE_IQ2_S:
        સ્થિતિ GGML_TYPE_IQ3_XXS:
        સ્થિતિ GGML_TYPE_IQ3_S:
        સ્થિતિ GGML_TYPE_IQ1_S:
        સ્થિતિ GGML_TYPE_IQ1_M:
            પાછા સાચું;
        મૂળભૂત:
            પાછા ખોટું;
    }
    જો (src1->type != GGML_TYPE_F32 && src1->type != GGML_TYPE_Q8_K) {
        પાછા ખોટું;
    }

    જો (src1->type == GGML_TYPE_Q8_K && !ggml_is_contiguous(src1)) {
        // We can handle noncontiguous floats because we're repacking to q8_k anyways
        પાછા ખોટું;
    }
    પાછા સાચું;
}
#endif

// per-thread workspace slot size (0 when tiled is disabled or unsupported on this arch)
સ્થિર size_t ggml_tiled_ws_size(શૂન્ય) {
    જો (!ggml_tiled_matmul_enabled()) {
        પાછા 0;
    }
    પાછા TILED_WS_SLOT; // clean 512KB slot, rounded up from sizeof(tiled_ws)
}

size_t ggml_tiled_wdata_size(પૂર્ણાંક n_tasks, સંરચના ggml_tensor * dst) {
    જો (! ggml_tiled_supported(dst->src[0], dst->src[1])) {
        પાછા 0; // unsupported, don't allocate
    }
    પાછા 64 + n_tasks * ggml_tiled_ws_size();  // 64 for alignment plus one 512KB slot per thread
}


// narrow-path K chunk: how many K's worth of weights to read contiguously into long/skinny tiles
// Assumes 32kb L1 cache budget
સ્થિર પૂર્ણાંક ggml_tiled_narrow_k_extent(int64_t n_rows, int64_t ne00) {
    પૂર્ણાંક l1 = 31 * 1024;  // fit weights in 31k to leave room for scales

    પૂર્ણાંક rows = (પૂર્ણાંક) n_rows;
    જો (rows < TILED_MICRO) rows = TILED_MICRO;
    પૂર્ણાંક ke = l1 / (TILED_MICRO + rows);
    ke &= ~(TILED_TILE_K - 1);  // floor to a multiple of 256
    અચલ પૂર્ણાંક ke_max = TILED_TILE_ROWS * TILED_TILE_K / TILED_MICRO; // buffer ceiling
    જો (ke > ke_max) ke = ke_max;
    જ્યારે (ke >= TILED_TILE_K && ne00 % ke != 0) {
        ke -= TILED_TILE_K;
    }
    પાછા ke < TILED_TILE_K ? 0 : ke;
}

// Writeback of the 256x256 window: buf is j-major (row stride buf_stride), dst is i-major (column stride dst_stride).
સ્થિર શૂન્ય tiled_store_window(અચલ દશાંશ * buf, પૂર્ણાંક n_src0, પૂર્ણાંક n_src1, પૂર્ણાંક buf_stride,
                               દશાંશ * dst, size_t dst_stride) {
    પૂર્ણાંક ri = 0;
    માટે (; ri + 16 <= n_src0; ri += 16) {
        પૂર્ણાંક rj = 0;
        માટે (; rj + 8 <= n_src1; rj += 8) {
            દશાંશ r[16][8];
            માટે (પૂર્ણાંક t = 0; t < 16; t++) {
                માટે (પૂર્ણાંક u = 0; u < 8; u++) {
                    r[t][u] = buf[(ri + t) * buf_stride + rj + u];
                }
            }
            માટે (પૂર્ણાંક u = 0; u < 8; u++) {
                માટે (પૂર્ણાંક t = 0; t < 16; t++) {
                    dst[(ri + t) + (size_t) (rj + u) * dst_stride] = r[t][u];
                }
            }
        }
        // ragged j tail
        માટે (; rj < n_src1; rj++) {
            માટે (પૂર્ણાંક t = 0; t < 16; t++) {
                dst[(ri + t) + (size_t) rj * dst_stride] = buf[(ri + t) * buf_stride + rj];
            }
        }
    }
    // ragged i tail
    માટે (; ri < n_src0; ri++) {
        માટે (પૂર્ણાંક j = 0; j < n_src1; j++) {
            dst[ri + (size_t) j * dst_stride] = buf[ri * buf_stride + j];
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
સ્થિર શૂન્ય tiled_store_window_scatter(અચલ દશાંશ * buf, પૂર્ણાંક n_src0, પૂર્ણાંક n_src1, પૂર્ણાંક buf_stride,
                                       દશાંશ * અચલ * col_ptrs) {
    પૂર્ણાંક ri = 0;
    માટે (; ri + 16 <= n_src0; ri += 16) {
        પૂર્ણાંક rj = 0;
        માટે (; rj + 8 <= n_src1; rj += 8) {
            દશાંશ r[16][8];
            માટે (પૂર્ણાંક t = 0; t < 16; t++) {
                માટે (પૂર્ણાંક u = 0; u < 8; u++) {
                    r[t][u] = buf[(ri + t) * buf_stride + rj + u];
                }
            }
            માટે (પૂર્ણાંક u = 0; u < 8; u++) {
                માટે (પૂર્ણાંક t = 0; t < 16; t++) {
                    col_ptrs[rj + u][ri + t] = r[t][u];
                }
            }
        }
        // ragged j tail
        માટે (; rj < n_src1; rj++) {
            માટે (પૂર્ણાંક t = 0; t < 16; t++) {
                col_ptrs[rj][ri + t] = buf[(ri + t) * buf_stride + rj];
            }
        }
    }
    // ragged i tail
    માટે (; ri < n_src0; ri++) {
        માટે (પૂર્ણાંક j = 0; j < n_src1; j++) {
            col_ptrs[j][ri] = buf[ri * buf_stride + j];
        }
    }
}


// one (g, k) macrotile: zero the acc window, sweep K in 256 element slabs, scatter the result rows
// rows points at this k window's routed src1 rows, row r at its base block
template <typename B, પૂર્ણાંક SUBBLK, તાર્કિક HAS_MIN, પૂર્ણાંક BIAS, તાર્કિક ACTBIAS>
સ્થિર શૂન્ય tiled_mmid_gemm_window(સંરચના ggml_tensor * dst, અચલ સંરચના ggml_tensor * src0,
                                   અચલ અક્ષર * src0_cur, int64_t r, int64_t k, int64_t nrows,
                                   અચલ int32_t * expert_rows,
                                   અચલ block_q8_K * અચલ * rows,
                                   tiled_ws * ws) {
    અચલ int64_t ne00 = src0->ne[0];
    અચલ int64_t ne01 = src0->ne[1];

    અચલ int64_t r_end = MIN(r + TILED_MMID_GROUP, ne01);
    અચલ પૂર્ણાંક n_src0 = (પૂર્ણાંક) (r_end - r);

    અચલ size_t src0_bs = ggml_type_size(src0->type);
    અચલ int64_t src0_stride = src0->nb[1] / src0_bs;

    // the window is at most TILED_MMID_GROUP x TILED_TILE_K, so zero only that region of acc
    માટે (int64_t i = 0; i < n_src0; i++) {
        memset(&ws->acc[i * TILED_TILE_ROWS], 0, nrows * કદ(દશાંશ));
    }

    // scattered writeback: column m goes to its routed dst row; r * nb[0] is the window row offset
    દશાંશ * col_ptrs[TILED_TILE_ROWS];
    માટે (int64_t m = 0; m < nrows; m++) {
        col_ptrs[m] = (દશાંશ *) ((અક્ષર *) dst->data + r * dst->nb[0] + expert_rows[2 * (k + m) + 0] * dst->nb[1] +
                                 expert_rows[2 * (k + m) + 1] * dst->nb[2]);
    }

    // narrow path (small nrows): we're memory bound in this case so do longer, skinny tiled in order
    // to have contiguous reads and improve memory bandwidth
    જો (nrows <= TILED_MICRO) {
        અચલ પૂર્ણાંક k_extent = ggml_tiled_narrow_k_extent(nrows, ne00);
        અચલ પૂર્ણાંક અંક_k = k_extent / TILED_TILE_K;
        માટે (int64_t k0 = 0; k0 < ne00; k0 += k_extent) {
            અચલ પૂર્ણાંક kstart = (પૂર્ણાંક) (k0 / TILED_TILE_K);
            // activation: one 16-row band (nrows <= 16). Unpack the whole k_extent chunk (all
            // slabs) so each activation row is a long stream, then repack each slab in place
            tiled_unpack_src1_q8_K(rows, nrows, &ws->src1, kstart, અંક_k);
            // repack the whole k_extent chunk in place: num_k slabs x 4 tiles, 16 rows, row stride num_k*256
            tiled_repack_src1(&ws->src1, 0, અંક_k, ACTBIAS);
            // weight groups (16 at a time): unpack the k_extent chunk of the group (the long per-row
            // read), then one standard MAC per slab accumulating into the same acc rows
            માટે (int64_t ir0 = r; ir0 < r_end; ir0 += TILED_MICRO) {
                અચલ પૂર્ણાંક n0 = (પૂર્ણાંક) MIN(TILED_MICRO, r_end - ir0);
                અચલ B * wbase = (અચલ B *) (src0_cur + ir0 * src0->nb[1] + kstart * src0_bs);
                tiled_unpack_src0(wbase, src0_stride, n0, &ws->src0, અંક_k);
                tiled_repack_src0<SUBBLK>(&ws->src0, n0, અંક_k, BIAS, ACTBIAS);
                દશાંશ * buf = ws->acc + (ir0 - r) * TILED_TILE_ROWS;
                માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
                    tiled_run_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1, 0, 0, અંક_k, slab, buf, TILED_TILE_ROWS);
                }
            }
        }
    } નહીંતર {
        // standard per-slab path: K is stepped in 256-element slabs
        માટે (int64_t ib = 0; ib < ne00; ib += TILED_TILE_K) {
            અચલ પૂર્ણાંક kblk = (પૂર્ણાંક) (ib / TILED_TILE_K);
            tiled_unpack_src0((અચલ B *) (src0_cur + r * src0->nb[1] + kblk * src0_bs), src0_stride, n_src0, &ws->src0, 1);
            tiled_repack_src0<SUBBLK>(&ws->src0, n_src0, 1, BIAS, ACTBIAS);
            tiled_unpack_src1_q8_K(rows, nrows, &ws->src1, kblk, 1);
            // 16x16 microtiles sweeping the window; repack each src1 band just-in-time
            // (j0-outer) so only the bands actually used are repacked and each is L1-hot
            માટે (int64_t ir1 = 0; ir1 < nrows; ir1 += TILED_MICRO) {
                tiled_repack_src1(&ws->src1, (પૂર્ણાંક) ir1, 1, ACTBIAS);
                માટે (int64_t ir0 = r; ir0 < r_end; ir0 += TILED_MICRO) {
                    tiled_run_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1,
                        (પૂર્ણાંક) (ir0 - r), (પૂર્ણાંક) ir1, 1, 0,
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
template <typename B, પૂર્ણાંક SUBBLK, તાર્કિક HAS_MIN, પૂર્ણાંક BIAS, તાર્કિક ACTBIAS>
સ્થિર શૂન્ય ggml_compute_forward_mul_mat_id_tiled_one_expert(
        અચલ સંરચના ggml_compute_params * params,
              સંરચના ggml_tensor *         dst,
        int64_t                            cur_a,
        int64_t                            cne1,
        અચલ int32_t *                    expert_rows,
        અક્ષર *                             scratch) {

    અચલ સંરચના ggml_tensor * src0 = dst->src[0];
    અચલ સંરચના ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    અચલ ગણના ggml_type vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;

    અચલ size_t nbw1 = (src1->type == vec_dot_type) ? nb11 : ggml_row_size(vec_dot_type, ne10);
    અચલ શૂન્ય * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;

    અચલ અક્ષર * src0_cur = (અચલ અક્ષર *) src0->data + cur_a * nb02;

    // this thread's workspace
    tiled_ws * ws = (tiled_ws *) (scratch + (size_t) ith * ggml_tiled_ws_size());

    // groups of TILED_MMID_GROUP rows; rounded up, the window tail is clamped in the gemm
    અચલ int64_t ngroups = (ne01 + TILED_MMID_GROUP - 1) / TILED_MMID_GROUP;

    અચલ int64_t g0 = (ngroups * ith) / nth;
    અચલ int64_t g1 = (ngroups * (ith + 1)) / nth;

    // no rows for this thread; nothing to do
    જો (g0 >= g1) {
        પાછા;
    }

    // MUL_MAT_ID scatters the src1 rows per routed row: point each k window's rows
    // at the routed src1 rows, then sweep the row windows
    માટે (int64_t k = 0; k < cne1; k += TILED_TILE_K) {
        અચલ int64_t nrows = MIN(TILED_TILE_K, cne1 - k);

        અચલ block_q8_K * rows[TILED_TILE_ROWS];
        માટે (int64_t i = 0; i < nrows; i++) {
            અચલ int64_t i11 = expert_rows[2 * (k + i) + 0] % ne11;
            અચલ int64_t i12 = expert_rows[2 * (k + i) + 1];
            rows[i] = (અચલ block_q8_K *) ((અચલ અક્ષર *) wdata + (i11 + i12 * ne11) * nbw1);
        }

        માટે (int64_t g = g0; g < g1; g++) {
            અચલ int64_t r = g * TILED_MMID_GROUP;

            tiled_mmid_gemm_window<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(dst, src0, src0_cur, r, k, nrows, expert_rows, rows, ws);
        }
    }
}

template <typename B, પૂર્ણાંક SUBBLK, તાર્કિક HAS_MIN, પૂર્ણાંક BIAS, તાર્કિક ACTBIAS>
સ્થિર શૂન્ય ggml_compute_forward_mul_mat_tiled_one_chunk(
    અચલ સંરચના ggml_compute_params * params,
    સંરચના ggml_tensor * dst,
    અચલ int64_t ir0_start,
    અચલ int64_t ir0_end,
    અચલ int64_t ir1_start,
    અચલ int64_t ir1_end,
    tiled_ws * ws) {

    અચલ સંરચના ggml_tensor * src0 = dst->src[0];
    અચલ સંરચના ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ ગણના ggml_type vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;

    // broadcast factors
    અચલ int64_t r2 = ne12 / ne02;
    અચલ int64_t r3 = ne13 / ne03;

    જો (ir0_start >= ir0_end || ir1_start >= ir1_end) {
        પાછા;
    }

    અચલ શૂન્ય * wdata = (src1->type == vec_dot_type) ? src1->data : params->wdata;
    અચલ size_t row_size = ggml_row_size(vec_dot_type, ne10);
    અચલ size_t src0_bs  = ggml_type_size(src0->type);
    અચલ size_t src1_bs  = ggml_type_size(vec_dot_type);

    GGML_ASSERT(ne00 % 256 == 0);
    પ્રતિજ્ઞા(ne12 % ne02 == 0);
    પ્રતિજ્ઞા(ne13 % ne03 == 0);

    અચલ int64_t src0_stride = nb01 / src0_bs;  // blocks between src0 rows
    અચલ int64_t src1_stride = (src1->type == vec_dot_type ? src1->nb[1] : row_size) / src1_bs;

    અચલ int64_t TILE = 256;
    અચલ int64_t MICRO = 16;

    // 256-wide windows over the chunk. The iir1 window is additionally clamped at the
    // src1 batch (ne11) boundary: the tiles require a constant batch index (i12/i13)
    // within a window, so advance by the clamped end, not a fixed 256.
    માટે (int64_t iir1 = ir1_start; iir1 < ir1_end; ) {
        int64_t iir1_end = MIN(iir1 + TILE, ir1_end);
        અચલ int64_t bnd = (iir1 / ne11 + 1) * ne11;
        જો (bnd < iir1_end) {
            iir1_end = bnd;
        }

        અચલ પૂર્ણાંક n_src1 = (પૂર્ણાંક) (iir1_end - iir1);

        // batch coords, constant within the clamped window
        અચલ int64_t i13 = iir1 / (ne12 * ne11);
        અચલ int64_t i12 = (iir1 - i13 * ne12 * ne11) / ne11;
        // within-batch row; dst_col below holds the i12/i13 batch offset, so the store
        // applies i11 * nb1 (not the flattened iir1, which spans all batch dims)
        અચલ int64_t i11 = iir1 - i13 * ne12 * ne11 - i12 * ne11;

        // dst batches == src1 batches (ggml_mul_mat), so the loop batch coords are in
        // src1 space; src0 batches are broadcast over them, map down into src0's batch
        અચલ int64_t i02 = i12 / r2;
        અચલ int64_t i03 = i13 / r3;

        અચલ અક્ષર * src0_row = (અચલ અક્ષર *) src0->data + i02 * src0->nb[2] + i03 * src0->nb[3];
        અક્ષર * dst_col = (અક્ષર *) dst->data + i12 * nb2 + i13 * nb3;

        // rows[r] is the window's row r base block; the k-slab offset is applied in the unpack
        અચલ block_q8_K * rows[TILED_TILE_ROWS];
        માટે (પૂર્ણાંક r = 0; r < n_src1; r++) {
            rows[r] = (અચલ block_q8_K *) ((અચલ અક્ષર *) wdata + (iir1 + r) * src1_stride * src1_bs);
        }

        માટે (int64_t iir0 = ir0_start; iir0 < ir0_end; iir0 += TILE) {
            int64_t iir0_end = MIN(iir0 + TILE, ir0_end);
            અચલ પૂર્ણાંક n_src0 = (પૂર્ણાંક) (iir0_end - iir0);

            // result buffer zeroed once per macrotile
            memset(ws->acc, 0, (size_t)TILED_TILE_ROWS * TILED_TILE_ROWS * કદ(દશાંશ));

            જો (n_src1 <= TILED_MICRO) {
                // narrow path: if we have few enough rows, we're memory bound
                // Do longer, skinner tiles so we have longer contiguous reads and improve bandwidth
                અચલ પૂર્ણાંક k_extent = ggml_tiled_narrow_k_extent(n_src1, ne00);
                અચલ પૂર્ણાંક અંક_k = k_extent / TILED_TILE_K;
                માટે (int64_t k0 = 0; k0 < ne00; k0 += k_extent) {
                    અચલ પૂર્ણાંક kstart = (પૂર્ણાંક) (k0 / TILED_TILE_K);
                    // activation: one 16-row band (n_src1 <= 16). Unpack the whole k_extent chunk
                    // (all slabs) so each activation row is a long stream, then repack each slab in place
                    tiled_unpack_src1_q8_K(rows, n_src1, &ws->src1, kstart, અંક_k);
                    // repack the whole k_extent chunk in place, if kernel wants to: num_k slabs x 4 tiles, 16 rows, row stride num_k*256
                    tiled_repack_src1(&ws->src1, 0, અંક_k, ACTBIAS);
                    // weight groups: unpack the k_extent chunk of each 16-row group (the long per-row
                    // read), then one standard MAC per slab accumulating into the same acc rows
                    માટે (int64_t ir0 = iir0; ir0 < iir0_end; ir0 += MICRO) {
                        અચલ પૂર્ણાંક n0 = (પૂર્ણાંક) MIN(MICRO, iir0_end - ir0);
                        અચલ B * wbase = (અચલ B *) (src0_row + ir0 * nb01 + kstart * src0_bs);
                        tiled_unpack_src0(wbase, src0_stride, n0, &ws->src0, અંક_k);
                        tiled_repack_src0<SUBBLK>(&ws->src0, n0, અંક_k, BIAS, ACTBIAS);
                        દશાંશ * buf = ws->acc + (ir0 - iir0) * TILED_TILE_ROWS;
                        માટે (પૂર્ણાંક slab = 0; slab < અંક_k; slab++) {
                            tiled_run_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1, 0, 0, અંક_k, slab, buf, TILED_TILE_ROWS);
                        }
                    }
                }
            } નહીંતર {
                // standard per-slab path: K is stepped in 256-element chunks
                માટે (int64_t ib = 0; ib < ne00; ib += TILE) {
                    અચલ પૂર્ણાંક kblk = (પૂર્ણાંક) (ib / TILE);
                    tiled_unpack_src0((અચલ B *) (src0_row + iir0 * nb01 + kblk * src0_bs), src0_stride, n_src0, &ws->src0, 1);
                    tiled_repack_src0<SUBBLK>(&ws->src0, n_src0, 1, BIAS, ACTBIAS);
                    tiled_unpack_src1_q8_K(rows, n_src1, &ws->src1, kblk, 1);

                    // 16x16 microtiles sweeping the window; repack each src1 band before first use
                    માટે (int64_t ir1 = iir1; ir1 < iir1_end; ir1 += MICRO) {
                        tiled_repack_src1(&ws->src1, (પૂર્ણાંક) (ir1 - iir1), 1, ACTBIAS);
                        માટે (int64_t ir0 = iir0; ir0 < iir0_end; ir0 += MICRO) {
                            tiled_run_microtile<SUBBLK, HAS_MIN, BIAS, ACTBIAS>(ws->src0, ws->src1,
                                (પૂર્ણાંક) (ir0 - iir0), (પૂર્ણાંક) (ir1 - iir1), 1, 0,
                                ws->acc, TILED_TILE_ROWS);
                        }
                    }
                }
            }
            // write acc back out from L2 to main memory
            tiled_store_window(ws->acc, n_src0, n_src1, TILED_TILE_ROWS,
                               (દશાંશ *) (dst_col + iir0 * nb0 + i11 * nb1), nb1 / nb0);
        }
        iir1 = iir1_end;
    }
}

template <typename B, પૂર્ણાંક SUBBLK, તાર્કિક HAS_MIN, પૂર્ણાંક BIAS, તાર્કિક ACTBIAS>
સ્થિર શૂન્ય ggml_compute_forward_mul_mat_tiled_driver(
        અચલ સંરચના ggml_compute_params * params,
              સંરચના ggml_tensor * dst) {

    અચલ સંરચના ggml_tensor * src0 = dst->src[0];
    અચલ સંરચના ggml_tensor * src1 = dst->src[1];

    GGML_TENSOR_દ્વિઅંકી_OP_LOCALS

    અચલ પૂર્ણાંક ith = params->ith;
    અચલ પૂર્ણાંક nth = params->nth;

    ગણના ggml_type      અચલ vec_dot_type = ggml_get_type_traits_cpu(src0->type)->vec_dot_type;
    ggml_from_float_t   અચલ from_float   = ggml_get_type_traits_cpu(vec_dot_type)->from_float;

    GGML_ASSERT(ne0 == ne01);
    GGML_ASSERT(ne1 == ne11);
    GGML_ASSERT(ne2 == ne12);
    GGML_ASSERT(ne3 == ne13);

    // we don't support permuted src0 or src1
    GGML_ASSERT(nb10 == ggml_type_size(src1->type));

    // dst cannot be transposed or permuted
    GGML_ASSERT(nb0 == કદ(દશાંશ));
    GGML_ASSERT(nb0 <= nb1);
    GGML_ASSERT(nb1 <= nb2);
    GGML_ASSERT(nb2 <= nb3);

    જો (src1->type != vec_dot_type) {
        અક્ષર * wdata = (અક્ષર *) params->wdata;

        અચલ size_t nbw0 = ggml_type_size(vec_dot_type);
        અચલ size_t nbw1 = ggml_row_size(vec_dot_type, ne10);
        અચલ size_t nbw2 = nbw1*ne11;
        અચલ size_t nbw3 = nbw2*ne12;

        પ્રતિજ્ઞા(params->wsize >= ne13*nbw3);
        GGML_ASSERT(src1->type == GGML_TYPE_F32);

        માટે (int64_t i13 = 0; i13 < ne13; ++i13) {
            માટે (int64_t i12 = 0; i12 < ne12; ++i12) {
                માટે (int64_t i11 = 0; i11 < ne11; ++i11) {
                    size_t bs = ggml_blck_size(vec_dot_type);
                    int64_t ne10_block_start = (ith * ne10/bs) / nth;
                    int64_t ne10_block_end   = ((ith + 1) * ne10/bs) / nth;
                    from_float((દશાંશ *)((અક્ષર *) src1->data + i13*src1->nb[3] + i12*src1->nb[2] + i11*src1->nb[1] + ne10_block_start*bs*src1->nb[0]),
                               (શૂન્ય *)               (wdata + i13*nbw3 + i12*nbw2 + i11*nbw1 + ne10_block_start*nbw0),
                               (ne10_block_end - ne10_block_start) * bs);
                }
            }
        }
    }

    જો (ith == 0) {
        // Every thread starts at ith, so the first unprocessed chunk is nth. This saves a bit of coordination right at the start.
        ggml_threadpool_chunk_set(params->threadpool, nth);
    }

    ggml_barrier(params->threadpool);

    // This is the size of the first dimension of the result, so we can iterate that way. (see the ASSERT above, these are the same numbers)
    અચલ int64_t nr0 = ne0;

    // This is the size of the rest of the dimensions of the result
    અચલ int64_t nr1 = ne1 * ne2 * ne3;

    // Now select a reasonable chunk size.
    પૂર્ણાંક chunk_size = 256; //TILED_TILE_ROWS;

    // distribute the work across the inner or outer loop based on which one is larger
    // The number of chunks in the 0/1 dim. CEIL(nr/chunk_size)
    int64_t nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
    int64_t nchunk1 = (nr1 + chunk_size - 1) / chunk_size;

    // Step down chunk size if too few chunks to saturate cores, minimum is microtile size
    જ્યારે (nchunk0 * nchunk1 < nth * 4 && chunk_size > 16) {
        chunk_size = chunk_size / 2;
        nchunk0 = (nr0 + chunk_size - 1) / chunk_size;
        nchunk1 = (nr1 + chunk_size - 1) / chunk_size;
    }

    // The number of elements in each chunk
    અચલ int64_t dr0 = (nr0 + nchunk0 - 1) / nchunk0;
    અચલ int64_t dr1 = (nr1 + nchunk1 - 1) / nchunk1;

    // The first chunk comes from our thread_id, the rest will get auto-assigned.
    પૂર્ણાંક current_chunk = ith;

    // per-thread workspace: after any converted src1 data in wdata, aligned to 64
    અક્ષર * ws_base = (અક્ષર *) params->wdata;
    જો (src1->type != vec_dot_type) {
        ws_base += GGML_PAD(ggml_row_size(vec_dot_type, ggml_nelements(src1)), 64);
    }
    ws_base = (અક્ષર *) (((uintptr_t) ws_base + 63) & ~(uintptr_t) 63);
    tiled_ws * ws = (tiled_ws *) (ws_base + (size_t) ith * ggml_tiled_ws_size());

    // TODO:  if we KNOW we're on a machine where all cores are equal, we could skip the coordination/work-stealing and just assign chunks deterministically
    જ્યારે (current_chunk < nchunk0 * nchunk1) {
        અચલ int64_t ith0 = current_chunk % nchunk0;
        અચલ int64_t ith1 = current_chunk / nchunk0;

        અચલ int64_t ir0_start = dr0 * ith0;
        અચલ int64_t ir0_end = MIN(ir0_start + dr0, nr0);

        અચલ int64_t ir1_start = dr1 * ith1;
        અચલ int64_t ir1_end = MIN(ir1_start + dr1, nr1);

        ggml_compute_forward_mul_mat_tiled_one_chunk<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(params, dst, ir0_start, ir0_end, ir1_start, ir1_end, ws);

        જો (nth >= nchunk0 * nchunk1) {
            તોડો;
        }

        current_chunk = ggml_threadpool_chunk_add(params->threadpool, 1);
    }
}

// src0 type dispatch, shared by the MUL_MAT and MUL_MAT_ID entries: one expert for
// MUL_MAT_ID (expert_rows != NULL), the full op for MUL_MAT
template <typename B, પૂર્ણાંક SUBBLK, તાર્કિક HAS_MIN, પૂર્ણાંક BIAS, તાર્કિક ACTBIAS>
સ્થિર તાર્કિક tiled_matmul_dispatch(અચલ સંરચના ggml_compute_params * params,
                                  સંરચના ggml_tensor * dst,
                                  અચલ int32_t * expert_rows,
                                  int64_t cur_a,
                                  int64_t cne1,
                                  અક્ષર * scratch) {
    જો (expert_rows == ખાલી) {
        ggml_compute_forward_mul_mat_tiled_driver<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(params, dst);
    } નહીંતર {
        ggml_compute_forward_mul_mat_id_tiled_one_expert<B, SUBBLK, HAS_MIN, BIAS, ACTBIAS>(params, dst, cur_a, cne1, expert_rows, scratch);
    }
    પાછા સાચું;
}

// the supported src0 types, one list for both ops; false to fall through to vec_dot
સ્થિર તાર્કિક ggml_tiled_matmul_type_dispatch(અચલ સંરચના ggml_compute_params * params,
                                            સંરચના ggml_tensor * dst,
                                            અચલ int32_t * expert_rows = ખાલી,
                                            int64_t cur_a = 0,
                                            int64_t cne1 = 0,
                                            અક્ષર * scratch = ખાલી) {
    પસંદગી (dst->src[0]->type) {
        સ્થિતિ GGML_TYPE_Q6_K:
            પાછા tiled_matmul_dispatch<block_q6_K, 16, ખોટું, 32, સાચું>(params, dst, expert_rows, cur_a, cne1, scratch);
        સ્થિતિ GGML_TYPE_Q5_K:
            પાછા tiled_matmul_dispatch<block_q5_K, 32, સાચું,  0, ખોટું>(params, dst, expert_rows, cur_a, cne1, scratch);
        સ્થિતિ GGML_TYPE_Q4_K:
            પાછા tiled_matmul_dispatch<block_q4_K, 32, સાચું,  0, ખોટું>(params, dst, expert_rows, cur_a, cne1, scratch);
        સ્થિતિ GGML_TYPE_Q3_K:
            પાછા tiled_matmul_dispatch<block_q3_K, 16, ખોટું,  4, સાચું>(params, dst, expert_rows, cur_a, cne1, scratch);
        સ્થિતિ GGML_TYPE_Q2_K:
            પાછા tiled_matmul_dispatch<block_q2_K, 16, સાચું,  0, ખોટું>(params, dst, expert_rows, cur_a, cne1, scratch);
        સ્થિતિ GGML_TYPE_IQ4_XS:
            પાછા tiled_matmul_dispatch<block_iq4_xs, 32, ખોટું, 128, ખોટું>(params, dst, expert_rows, cur_a, cne1, scratch);
        સ્થિતિ GGML_TYPE_IQ2_XXS:
            પાછા tiled_matmul_dispatch<block_iq2_xxs, 32, ખોટું, 128, સાચું>(params, dst, expert_rows, cur_a, cne1, scratch);
        સ્થિતિ GGML_TYPE_IQ2_XS:
            પાછા tiled_matmul_dispatch<block_iq2_xs, 16, ખોટું, 128, સાચું>(params, dst, expert_rows, cur_a, cne1, scratch);
        સ્થિતિ GGML_TYPE_IQ2_S:
            પાછા tiled_matmul_dispatch<block_iq2_s, 16, ખોટું, 128, સાચું>(params, dst, expert_rows, cur_a, cne1, scratch);
        સ્થિતિ GGML_TYPE_IQ3_XXS:
            પાછા tiled_matmul_dispatch<block_iq3_xxs, 32, ખોટું, 128, સાચું>(params, dst, expert_rows, cur_a, cne1, scratch);
        સ્થિતિ GGML_TYPE_IQ3_S:
            પાછા tiled_matmul_dispatch<block_iq3_s, 32, ખોટું, 128, સાચું>(params, dst, expert_rows, cur_a, cne1, scratch);
        સ્થિતિ GGML_TYPE_IQ1_S:
            પાછા tiled_matmul_dispatch<block_iq1_s, 32, ખોટું, 128, સાચું>(params, dst, expert_rows, cur_a, cne1, scratch);
        સ્થિતિ GGML_TYPE_IQ1_M:
            પાછા tiled_matmul_dispatch<block_iq1_m, 16, ખોટું, 128, સાચું>(params, dst, expert_rows, cur_a, cne1, scratch);
        મૂળભૂત:
            પાછા ખોટું;
    }
}

સ્થિર તાર્કિક ggml_tiled_min_batch(int64_t rows) {
    //  Profitable at rows >= 8, take even when unprofitable if we're forced
    પાછા rows >= 8 || ggml_tiled_matmul_forced();
}

// tiled K-quant matmul; returns true if the op was computed here,
// false to fall through to the stock path
તાર્કિક ggml_compute_forward_mul_mat_tiled(
        અચલ સંરચના ggml_compute_params * params,
              સંરચના ggml_tensor * dst) {
    // --use-ref means bail out and go back to vec_dot reference impl
    જો (params->use_ref) {
        પાછા ખોટું;
    }
    જો (!ggml_tiled_supported(dst->src[0], dst->src[1])) {
        પાછા ખોટું;
    }
    જો (!ggml_tiled_min_batch(dst->src[1]->ne[1])) {
        પાછા ખોટું;
    }
    પાછા ggml_tiled_matmul_type_dispatch(params, dst);
}

// MUL_MAT_ID (MoE), one expert; returns true if the expert was computed here,
// per expert eligibility (type gate, batch floor) is decided here
તાર્કિક ggml_compute_forward_mul_mat_id_tiled(
        અચલ સંરચના ggml_compute_params * params,
              સંરચના ggml_tensor *         dst,
        int64_t                            cur_a,
        int64_t                            cne1,
        અચલ int32_t *                    expert_rows,
        અક્ષર *                             scratch) {
    જો (params->use_ref) {
        પાછા ખોટું;
    }
    જો (!ggml_tiled_supported(dst->src[0], dst->src[1])) {
        પાછા ખોટું;
    }
    // profitability is per expert: the rows routed to this expert
    જો (!ggml_tiled_min_batch(cne1)) {
        પાછા ખોટું;
    }
    પાછા ggml_tiled_matmul_type_dispatch(params, dst, expert_rows, cur_a, cne1, scratch);
}
